"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


def save(fig, name: str) -> str:
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ROOT)}")
    return path


def dump(results: dict, name: str) -> None:
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o
    path = os.path.join(RES_DIR, name)
    with open(path, "w") as fh:
        json.dump(conv(results), fh, indent=2)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.label}]")
        return self

    def __exit__(self, *exc):
        print(f"  done in {time.time() - self.t0:.1f}s")
