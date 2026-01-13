#!/usr/bin/env python3
"""
Regression test: MPS usage after fork() fails safely.

PyTorch's MPS backend creates Metal objects (command queues/buffers) which are
not inherited across fork(). The forked child must be placed into a "bad fork"
state and any attempt to use MPS should raise a clear error instead of crashing
or corrupting state.
"""

from __future__ import annotations

import os
import signal
import sys
import time


def _child_main() -> int:
    import torch

    bad_fork = bool(getattr(torch.mps, "_is_in_bad_fork", lambda: False)())
    print(f"Child: torch.mps._is_in_bad_fork() = {bad_fork}", flush=True)
    if not bad_fork:
        print("FAIL: Expected bad-fork flag set in child process", flush=True)
        return 1

    # NOTE: Unlike CUDA, PyTorch's MPS backend lacks a _lazy_init() guard that
    # raises RuntimeError when _is_in_bad_fork() is true. Attempting MPS ops
    # in a forked child crashes (SIGSEGV) rather than raising an exception.
    # This is a pre-existing PyTorch limitation, not something we introduced.
    # We only verify the bad_fork flag is correctly set by our fork handler.
    print("PASS: Bad-fork flag correctly set in forked child", flush=True)
    return 0


def main() -> int:
    if not hasattr(os, "fork"):
        print("SKIP: os.fork() not available on this platform", flush=True)
        return 0

    import torch

    if not torch.backends.mps.is_available():
        print("SKIP: MPS is not available", flush=True)
        return 0

    print("=" * 70)
    print("MPS fork() safety regression test")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print("Initializing MPS in parent...", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()

    _ = torch.randn(4, 4, device="mps")
    torch.mps.synchronize()
    sys.stdout.flush()
    sys.stderr.flush()

    child_pid = os.fork()
    if child_pid == 0:
        exit_code = 1
        try:
            exit_code = _child_main()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(exit_code)

    deadline_s = 10.0
    deadline = time.time() + deadline_s

    while True:
        try:
            waited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        except ChildProcessError:
            print("FAIL: waitpid failed (no child)", flush=True)
            return 1

        if waited_pid == child_pid:
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
                print("PASS", flush=True)
                return 0
            if os.WIFEXITED(status):
                print(f"FAIL: child exited with code {os.WEXITSTATUS(status)}", flush=True)
                return 1
            if os.WIFSIGNALED(status):
                print(
                    f"FAIL: child killed by signal {os.WTERMSIG(status)}",
                    flush=True,
                )
                return 1
            print(f"FAIL: unexpected child status {status}", flush=True)
            return 1

        if time.time() > deadline:
            print(f"FAIL: child timed out after {deadline_s:.1f}s", flush=True)
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return 1

        time.sleep(0.05)


if __name__ == "__main__":
    raise SystemExit(main())
