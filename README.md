# Matchers under rotation

An interactive page comparing how ten image point matchers handle in-plane rotation:
one aerial drone frame is held fixed while the overlapping next frame is rotated through
360° in 5° steps, and each matcher is scored at every angle.

**Live page:** `https://chandrakanth-gudavalli.github.io/matcher-rotation/`

Methods: LoMa-B, LoMa-R, LoMa-G, LoFTR, RoMa, RoMa v2, SuperPoint + LightGlue,
ALIKED + LightGlue, SIFT (ratio test + mutual NN), MASt3R.

## How correctness is measured

The rotation applied to the second image is known, so every match can be mapped back to the
unrotated image. A match is correct when its Sampson (epipolar) error is below 3 px (at the
2048 px working resolution) under a fundamental matrix fitted with MAGSAC to the pooled 0°
matches of all methods. Matches with either end outside the image circle are dropped first.

## Reproduce

Each matcher runs in its own environment (they have conflicting dependencies); all were
tested with PyTorch 2.14 + CUDA 12.6 on a GTX TITAN X.

```bash
# third-party code
mkdir -p third_party && cd third_party
git clone https://github.com/davnords/LoMa.git
git clone https://github.com/cvg/LightGlue.git
git clone https://github.com/Parskatt/RoMa.git
git clone https://github.com/Parskatt/RoMaV2.git
git clone --recursive https://github.com/naver/mast3r.git
cd ..

# env "loma": LoMa, LightGlue (SuperPoint, ALIKED), LoFTR (kornia), SIFT (opencv)
pip install -e third_party/LoMa -e third_party/LightGlue
# env "romas": RoMa + RoMa v2 (uninstall fused-local-corr if it has no build for your torch)
pip install -e third_party/RoMa -e third_party/RoMaV2
# env "mast3r"
pip install -r third_party/mast3r/requirements.txt -r third_party/mast3r/dust3r/requirements.txt

python bench/prepare.py <image_A> <image_B>          # circular crops + 72 rotated copies
python bench/run_sweep.py loma-b                     # repeat per matcher, in its env
PYTHONPATH=third_party/mast3r python bench/run_sweep.py mast3r
python bench/score_export.py                         # writes assets/data.json
python -m http.server                                # open http://localhost:8000
```

Without the fused correlation kernel, RoMa v2's native fallback needs more memory than a
12 GB GPU has at its default "precise" setting; this run used a local patch that computes
the local correlation in row chunks (same result, lower peak memory).

## Publish on GitHub Pages

Push this repository to GitHub, then *Settings → Pages → Deploy from a branch → `main` / root*.
