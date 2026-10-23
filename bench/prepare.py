"""Make the circular-crop image pair and every rotated copy of image B used by the sweep.

Usage: python bench/prepare.py <image_A> <image_B> [--size 2048] [--step 5]
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "data"


def circular_crop(path: str, size: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    d = min(w, h)
    l, t = (w - d) // 2, (h - d) // 2
    sq = im.crop((l, t, l + d, t + d)).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)
    out = Image.new("RGB", (size, size), (0, 0, 0))
    out.paste(sq, (0, 0), mask)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_A")
    ap.add_argument("image_B")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    (DATA / "rot").mkdir(parents=True, exist_ok=True)
    A = circular_crop(args.image_A, args.size)
    B = circular_crop(args.image_B, args.size)
    A.save(DATA / "A.jpg", quality=95)
    B.save(DATA / "B.jpg", quality=95)
    for angle in range(0, 360, args.step):
        # PIL rotates counter-clockwise; the circular crop keeps all content inside the frame
        B.rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0)).save(
            DATA / "rot" / f"B_{angle:03d}.jpg", quality=95)
    print(f"wrote {DATA}")


if __name__ == "__main__":
    main()
