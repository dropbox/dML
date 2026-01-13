#!/usr/bin/env python3
"""
Repro harness for the MultiheadAttention SDPA *layout* on MPS.

This builds q/k/v with the same (B, H, L, D) strides produced by
`multi_head_attention_forward()`:
  - stride pattern: (H*D, D, B*H*D, 1)

NOTE:
- This file is primarily useful for inspecting the strided layout and as a
  lightweight SDPA stress test.
- For the current known parallel-stream correctness failure, use:
  `tests/repro_transformer_block_race.py`.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SDPAInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def _make_mha_strided_qkv_from_packed_proj(
    proj: torch.Tensor, *, num_heads: int, seq_len: int, batch: int, embed_dim: int
) -> SDPAInputs:
    head_dim = embed_dim // num_heads
    if embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim={embed_dim} not divisible by num_heads={num_heads}")

    # Mimic torch.nn.functional._in_projection_packed (self-attention) output:
    # a single packed buffer with q/k/v stacked in dim0, then sliced.
    if proj.shape != (3, seq_len, batch, embed_dim):
        raise ValueError(f"proj shape {tuple(proj.shape)} != {(3, seq_len, batch, embed_dim)}")
    q0, k0, v0 = proj[0], proj[1], proj[2]

    # Reshape to (L, B*H, D) then transpose to (B*H, L, D) and view to (B, H, L, D).
    # This yields the characteristic MHA SDPA stride pattern: (H*D, D, B*H*D, 1).
    q = q0.reshape(seq_len, batch * num_heads, head_dim).transpose(0, 1).view(batch, num_heads, seq_len, head_dim)
    k = k0.reshape(seq_len, batch * num_heads, head_dim).transpose(0, 1).view(batch, num_heads, seq_len, head_dim)
    v = v0.reshape(seq_len, batch * num_heads, head_dim).transpose(0, 1).view(batch, num_heads, seq_len, head_dim)
    return SDPAInputs(q=q, k=k, v=v)


def _run_sdpa(inputs: SDPAInputs) -> torch.Tensor:
    # No dropout, non-causal (matches TransformerEncoderLayer eval path).
    out = F.scaled_dot_product_attention(inputs.q, inputs.k, inputs.v, dropout_p=0.0, is_causal=False)
    if out.device.type == "mps":
        torch.mps.synchronize()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("SKIP: MPS not available")
        return 0

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    torch.manual_seed(0)
    proj_cpu = torch.randn(3, args.seq_len, args.batch, args.embed_dim, device="cpu", dtype=torch.float32)

    # Golden output from a spawned thread (avoids main-thread vs spawned-thread MPS differences).
    golden_holder: list[Optional[torch.Tensor]] = [None]

    def compute_golden() -> None:
        proj = proj_cpu.to(device=args.device, dtype=dtype)
        inputs = _make_mha_strided_qkv_from_packed_proj(
            proj, num_heads=args.num_heads, seq_len=args.seq_len, batch=args.batch, embed_dim=args.embed_dim
        )
        golden_holder[0] = _run_sdpa(inputs).cpu()

    t = threading.Thread(target=compute_golden)
    t.start()
    t.join()
    golden = golden_holder[0]
    if golden is None:
        raise RuntimeError("Failed to compute golden output")

    max_diff_overall = 0.0

    for it in range(args.iterations):
        barrier = threading.Barrier(args.threads)
        outputs: list[Optional[torch.Tensor]] = [None] * args.threads
        errors: list[str] = []

        def worker(thread_id: int) -> None:
            try:
                proj = proj_cpu.to(device=args.device, dtype=dtype)
                inputs = _make_mha_strided_qkv_from_packed_proj(
                    proj, num_heads=args.num_heads, seq_len=args.seq_len, batch=args.batch, embed_dim=args.embed_dim
                )
                if thread_id == 0:
                    q_stride = tuple(inputs.q.stride())
                    k_stride = tuple(inputs.k.stride())
                    v_stride = tuple(inputs.v.stride())
                    q_offs = int(inputs.q.storage_offset())
                    k_offs = int(inputs.k.storage_offset())
                    v_offs = int(inputs.v.storage_offset())
                    print(
                        f"iter {it}: q stride={q_stride} off={q_offs} "
                        f"k stride={k_stride} off={k_offs} v stride={v_stride} off={v_offs}",
                        flush=True,
                    )
                barrier.wait(timeout=60)
                outputs[thread_id] = _run_sdpa(inputs).cpu()
            except Exception as e:
                try:
                    barrier.abort()
                except Exception:
                    pass
                errors.append(f"thread {thread_id}: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        if errors:
            raise RuntimeError(";\n".join(errors))

        for thread_id, out in enumerate(outputs):
            if out is None:
                raise RuntimeError(f"Missing output for thread {thread_id}")
            diff = (out.float() - golden.float()).abs().max().item()
            max_diff_overall = max(max_diff_overall, diff)
            if diff > args.atol:
                print(f"FAIL: iter {it} thread {thread_id} max_abs_diff={diff:.6e} (atol={args.atol})")
                return 1

        print(f"PASS: iter {it} max_abs_diff={max_diff_overall:.3e}", flush=True)

    print(f"ALL PASS: max_abs_diff={max_diff_overall:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
