"""Match fixed image A against every rotated copy of image B with one matcher.

Usage: python bench/run_sweep.py <matcher>   (see matchers.MATCHERS)
Writes outputs/raw/<matcher>/<angle>.npz with pA, pB (pixels in the rotated B), conf, seconds.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from matchers import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "data"

name = sys.argv[1]
out = ROOT / "outputs" / "raw" / name
out.mkdir(parents=True, exist_ok=True)
match = build(name)
path_A = str(DATA / "A.jpg")
rotated = sorted((DATA / "rot").glob("B_*.jpg"))

match(path_A, str(rotated[0]))  # warm-up so timings exclude one-off setup
for p in rotated:
    angle = int(p.stem.split("_")[1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t = time.perf_counter()
    pA, pB, conf = match(path_A, str(p))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t
    np.savez(out / f"{angle:03d}.npz", pA=pA, pB=pB, conf=conf, seconds=dt)
    print(f"{name} {angle:3d}° {len(pA):5d} matches {dt:.2f}s", flush=True)
