#!/usr/bin/env python3
"""B1: run AnyView's released eval_avb.py unchanged, adding timestamps and peak GPU memory.

Usage (from the repo root): python scripts/b1_profile_avb.py <eval_avb.py args...>
Runs third_party/AnyView-DVS/scripts/eval_avb.py via runpy in its own directory. Each episode
line gets the wall time since the previous one; at exit it prints per-episode timing excluding
the first (warm-up) episode, and torch.cuda peak allocated/reserved memory.
"""

from __future__ import annotations

import builtins
import json
import os
import runpy
import sys
import time
from pathlib import Path

import torch

AV = Path(__file__).resolve().parents[1] / "third_party" / "AnyView-DVS"
stamps: list[tuple[str, float]] = []
t_start = time.time()
_print = builtins.print


def timed_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    now = time.time()
    if "PSNR=" in msg:
        prev = stamps[-1][1] if stamps else t_start
        stamps.append((msg.strip().split(":")[0], now))
        msg = f"{msg}  [{now - prev:.1f}s]"
    _print(f"{time.strftime('%H:%M:%S')} {msg}", **{k: v for k, v in kwargs.items() if k != "sep"})


def main():
    os.chdir(AV)
    sys.path.insert(0, str(AV))
    sys.argv = [str(AV / "scripts" / "eval_avb.py")] + sys.argv[1:]
    builtins.print = timed_print
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        builtins.print = _print
        durs = [b - a for (_, a), (_, b) in zip(stamps, stamps[1:])]
        report = {
            "n_episodes": len(stamps),
            "first_episode_s_incl_load": (stamps[0][1] - t_start) if stamps else None,
            "per_episode_s_after_warmup": {"mean": sum(durs) / len(durs), "min": min(durs), "max": max(durs)} if durs else None,
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        }
        _print("B1_PROFILE " + json.dumps(report))


if __name__ == "__main__":
    main()
