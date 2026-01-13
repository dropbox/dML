#!/usr/bin/env python3
# Copyright 2024-2025 Andrew Yates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Trace through PyTorch Generator using forward hooks.

Run in kokoro PyTorch environment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def main():
    ref_path = Path("/tmp/kokoro_ref/generator_traces.npz")
    if not ref_path.exists():
        print("ERROR: Run export_generator_intermediates.py first")
        return 1

    pt_traces = dict(np.load(ref_path))
    ref = np.load("/tmp/kokoro_ref/tensors.npz")

    import torch

    try:
        from kokoro import KModel
    except ImportError:
        print("ERROR: kokoro package not found. Run in kokoro environment.")
        return 1

    print("=== Loading Kokoro Model ===")
    model = KModel().to("cpu").eval()
    gen = model.decoder.generator

    # Inputs
    gen_input_ncl = pt_traces["generator_input_ncl"]  # [1, 512, 126]
    style = ref["style_128"]  # [1, 128]
    f0 = ref["F0_pred"]  # [1, 126]

    gen_input_pt = torch.tensor(gen_input_ncl)
    style_pt = torch.tensor(style)
    f0_pt = torch.tensor(f0)

    print(f"Generator input: {gen_input_pt.shape}")
    print(f"Style: {style_pt.shape}")
    print(f"F0: {f0_pt.shape}")

    traces = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        traces[f"{name}_out_{i}"] = o.detach().cpu().numpy()
            elif isinstance(output, torch.Tensor):
                traces[f"{name}_out"] = output.detach().cpu().numpy()
            # Also save input
            if isinstance(input, tuple) and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    traces[f"{name}_in"] = input[0].detach().cpu().numpy()

        return hook

    # Register hooks on key modules
    handles.append(gen.m_source.register_forward_hook(make_hook("m_source")))
    handles.append(gen.stft.register_forward_hook(make_hook("stft")))

    for i, conv in enumerate(gen.noise_convs):
        handles.append(conv.register_forward_hook(make_hook(f"noise_conv_{i}")))

    for i, res in enumerate(gen.noise_res):
        handles.append(res.register_forward_hook(make_hook(f"noise_res_{i}")))

    for i, up in enumerate(gen.ups):
        handles.append(up.register_forward_hook(make_hook(f"ups_{i}")))

    for i, blk in enumerate(gen.resblocks):
        handles.append(blk.register_forward_hook(make_hook(f"resblock_{i}")))

    handles.append(gen.conv_post.register_forward_hook(make_hook("conv_post")))

    # Run forward
    print("\n=== Running Generator Forward ===")
    with torch.no_grad():
        audio = gen(gen_input_pt, style_pt, f0_pt)
        traces["final_audio"] = audio.detach().cpu().numpy()
        print(
            f"Audio output: {audio.shape}, range: [{audio.min():.4f}, {audio.max():.4f}]"
        )

    # Remove hooks
    for h in handles:
        h.remove()

    # Print trace summary
    print("\n=== Trace Summary ===")
    for k in sorted(traces.keys()):
        v = traces[k]
        print(f"  {k}: {v.shape}, range: [{v.min():.4f}, {v.max():.4f}]")

    # Save traces
    output_path = Path("/tmp/kokoro_ref/generator_internal_traces.npz")
    np.savez_compressed(output_path, **traces)
    print(f"\nSaved {len(traces)} traces to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
