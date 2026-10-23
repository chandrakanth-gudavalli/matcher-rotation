"""Render README media from assets/data.json: a 2x2 rotation GIF and light/dark precision charts.

Usage: python bench/make_readme_media.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS, MEDIA = ROOT / "assets", ROOT / "assets" / "readme"
MEDIA.mkdir(parents=True, exist_ok=True)

data = json.load(open(ASSETS / "data.json"))
angles = data["angles"]
methods = {m["id"]: m for m in data["methods"]}

FAMILY = {  # categorical slots, light / dark steps
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
}
DASH = {"solid": "-", "dash": (0, (5, 3)), "dot": (0, (1, 2.5))}
THEME = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e1e0d9", axis="#c3c2b7"),
    "dark": dict(surface="#0d1117", text="#f0efec", text2="#c3c2b7", grid="#2c2c2a", axis="#383835"),
}
GOOD, BAD = (12, 163, 12), (208, 59, 59)


def font(size):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def gif(ids, side=250, step=10, path=MEDIA / "rotation.gif"):
    A = Image.open(ASSETS / "A.jpg").convert("RGB").resize((side, side), Image.LANCZOS)
    B = Image.open(ASSETS / "B.jpg").convert("RGB").resize((side, side), Image.LANCZOS)
    f_title, f_stat, f_big = font(16), font(14), font(20)
    pad, head, top = 10, 30, 44
    pw, ph = 2 * side, side + head
    frames = []
    for k, a in enumerate(angles):
        if a % step:
            continue
        frame = Image.new("RGB", (2 * pw + 3 * pad, top + 2 * ph + 2 * pad), (13, 17, 23))
        for j, mid in enumerate(ids):
            m = methods[mid]
            panel = Image.new("RGB", (pw, ph), (13, 17, 23))
            panel.paste(A, (0, head))
            panel.paste(B.rotate(a, resample=Image.BICUBIC), (side, head))
            d = ImageDraw.Draw(panel)
            samples = sorted(m["samples"][k], key=lambda s: s[4], reverse=True)[:90]
            for xa, ya, xb, yb, ok in sorted(samples, key=lambda s: s[4], reverse=True):
                c = GOOD if ok else BAD
                d.line([(xa * side, head + ya * side), (side + xb * side, head + yb * side)], fill=c, width=2)
            prec = m["correct"][k] / max(m["matches"][k], 1)
            d.text((6, 5), m["name"], fill=(240, 239, 236), font=f_title)
            txt = f"{prec:.0%} correct"
            d.text((pw - 6 - d.textlength(txt, font=f_stat), 7), txt,
                   fill=GOOD if prec >= 0.8 else (BAD if prec < 0.3 else (250, 178, 25)), font=f_stat)
            frame.paste(panel, (pad + (j % 2) * (pw + pad), top + (j // 2) * (ph + pad)))
        d = ImageDraw.Draw(frame)
        label = f"right image rotated {a}°"
        d.text((pad, 12), label, fill=(240, 239, 236), font=f_big)
        bar = (frame.width - pad - 260, 20, frame.width - pad, 28)
        d.rounded_rectangle(bar, radius=4, fill=(44, 44, 42))
        d.rounded_rectangle((bar[0], bar[1], bar[0] + max(8, (bar[2] - bar[0]) * a / 355), bar[3]), radius=4,
                            fill=(57, 135, 229))
        frames.append(frame.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


def chart(mode):
    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    xs = angles + [360]
    for m in data["methods"]:
        p = [c / max(n, 1) * 100 for c, n in zip(m["correct"], m["matches"])]
        ax.plot(xs, p + p[:1], color=FAMILY[mode][m["family"]], ls=DASH[m["dash"]], lw=2.2,
                label=m["name"], solid_capstyle="round", dash_capstyle="round")
    ax.set_xlim(0, 360)
    ax.set_ylim(-2, 102)
    ax.set_xticks(range(0, 361, 45), [f"{a}°" for a in range(0, 361, 45)])
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["text2"], labelsize=11, length=0, pad=6)
    ax.set_xlabel("Rotation of the second image", color=t["text2"], fontsize=12)
    ax.set_title("Precision (% of matches that are correct) through a full rotation", color=t["text"],
                 fontsize=14, loc="left", pad=12, fontweight="bold")
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5, frameon=False, fontsize=10.5,
                    handlelength=2.6, labelcolor=t["text"])
    fig.tight_layout()
    path = MEDIA / f"precision_{mode}.png"
    fig.savefig(path, facecolor=t["surface"], bbox_extra_artists=[leg], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


gif(["romav2", "loma-b", "loftr", "sp-lg"])
chart("light")
chart("dark")
