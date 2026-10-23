"""Score every sweep against a shared reference geometry and export the site data.

A match is "correct" when its Sampson (epipolar) error, after undoing the known rotation of B,
is below TOL pixels under a fundamental matrix fitted (MAGSAC) to the pooled 0-degree matches
of all methods. Writes assets/data.json and the two display images.

Usage: python bench/score_export.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from matchers import MATCHERS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA, RAW, ASSETS = ROOT / "outputs" / "data", ROOT / "outputs" / "raw", ROOT / "assets"
TOL = 3.0          # px at the 2048 px working resolution
SAMPLES = 120      # matches drawn per method/angle on the page
DISPLAY = 1024     # px size of the images shipped with the page
DENSE = {"roma", "romav2", "mast3r", "loftr"}
# chart encoding: colour = method family, line style = variant within the family
STYLE = {
    "loma-b": (0, "solid"), "loma-r": (0, "dash"), "loma-g": (0, "dot"),
    "loftr": (1, "solid"),
    "roma": (2, "solid"), "romav2": (2, "dash"),
    "sp-lg": (3, "solid"), "aliked-lg": (3, "dash"),
    "sift": (4, "solid"),
    "mast3r": (5, "solid"),
}

size = Image.open(DATA / "A.jpg").size[0]
c = size / 2


def load(m, angle):
    """Matches with both ends inside the image circle (the black corners carry no content)."""
    d = np.load(RAW / m / f"{angle:03d}.npz")
    pA, pB = d["pA"].astype(np.float64), d["pB"].astype(np.float64)
    inside = lambda p: np.hypot(p[:, 0] - c, p[:, 1] - c) < c - 2
    keep = inside(pA) & inside(pB)
    return pA[keep], pB[keep], float(d["seconds"])


def unrotate(p, angle):
    # undo PIL's counter-clockwise rotation (image y axis points down)
    t = np.deg2rad(angle)
    x, y = p[:, 0] - c, p[:, 1] - c
    return np.stack([c + x * np.cos(t) - y * np.sin(t), c + x * np.sin(t) + y * np.cos(t)], 1)


def sampson(F, a, b):
    a = np.c_[a, np.ones(len(a))]
    b = np.c_[b, np.ones(len(b))]
    Fa, Ftb = a @ F.T, b @ F
    return np.sqrt(np.sum(b * Fa, 1) ** 2 / (Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2))


methods = sorted((m for m in MATCHERS if (RAW / m / "000.npz").exists()), key=list(STYLE).index)
angles = sorted(int(p.stem) for p in (RAW / methods[0]).glob("*.npz"))

pooled = [load(m, 0) for m in methods]
pA0 = np.concatenate([d[0] for d in pooled])
pB0 = np.concatenate([d[1] for d in pooled])
F, inl = cv2.findFundamentalMat(pA0, pB0, cv2.USAC_MAGSAC, TOL / 2, 0.99999, 50000)
print(f"reference F from {len(pA0)} pooled 0° matches, {int(inl.sum())} inliers")

rng = np.random.default_rng(0)
out = {"angles": angles, "tol_px": TOL, "work_px": size, "methods": []}
for m in methods:
    rec = {"id": m, "name": MATCHERS[m][0], "dense": m in DENSE, "family": STYLE[m][0], "dash": STYLE[m][1],
           "matches": [], "correct": [], "seconds": [], "samples": []}
    for a in angles:
        pA, pB, seconds = load(m, a)
        ok = sampson(F, pA, unrotate(pB, a)) < TOL if len(pA) else np.zeros(0, bool)
        rec["matches"].append(int(len(pA)))
        rec["correct"].append(int(ok.sum()))
        rec["seconds"].append(round(seconds, 3))
        sel = rng.choice(len(pA), min(SAMPLES, len(pA)), replace=False) if len(pA) else []
        rec["samples"].append([[round(float(v), 4) for v in (*(pA[i] / size), *(pB[i] / size))] + [int(ok[i])]
                               for i in sel])
    p = np.array(rec["correct"]) / np.maximum(rec["matches"], 1)
    print(f"{m:8s} precision min {p.min():.3f} mean {p.mean():.3f}  "
          f"median time {np.median(rec['seconds']):.2f}s")
    out["methods"].append(rec)

ASSETS.mkdir(exist_ok=True)
for k in ("A", "B"):
    Image.open(DATA / f"{k}.jpg").resize((DISPLAY, DISPLAY), Image.LANCZOS).save(ASSETS / f"{k}.jpg", quality=85)
json.dump(out, open(ASSETS / "data.json", "w"), separators=(",", ":"))
print(f"wrote {ASSETS / 'data.json'} ({(ASSETS / 'data.json').stat().st_size / 1e6:.1f} MB)")
