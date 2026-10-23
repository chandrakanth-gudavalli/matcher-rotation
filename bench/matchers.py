"""Uniform wrappers around each matcher: build(name) returns fn(path_A, path_B) -> (pA, pB, conf).

pA, pB are (N, 2) float arrays in pixel coordinates of the input images; conf is (N,).
Imports are lazy so each matcher only needs its own environment.
"""
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_DENSE_SAMPLES = 5000  # matches sampled from dense methods (RoMa, RoMa v2)


def _np(x):
    return x.detach().float().cpu().numpy()


def build_loma(variant):
    from loma import LoMa
    from loma import loma as L

    model = LoMa(getattr(L, variant)())

    @torch.inference_mode()
    def match(pa, pb):
        kA, dA, h1, w1 = model.detect_and_describe(pa)
        kB, dB, h2, w2 = model.detect_and_describe(pb)
        scores = model(kA, kB, dA, dB)["scores"]
        m0, _, ms0, _ = L.filter_matches(scores, model.cfg.filter_threshold)
        idx = torch.where(m0[0] > -1)[0]
        pA = L.to_pixel_coords(kA[0][idx], h1, w1)
        pB = L.to_pixel_coords(kB[0][m0[0][idx]], h2, w2)
        return _np(pA), _np(pB), _np(ms0[0][idx])

    return match


def build_lightglue(features):
    import lightglue
    from lightglue import LightGlue
    from lightglue.utils import load_image, rbd

    Extractor = {"superpoint": lightglue.SuperPoint, "aliked": lightglue.ALIKED}[features]

    extractor = Extractor(max_num_keypoints=2048).eval().to(DEVICE)
    matcher = LightGlue(features=features).eval().to(DEVICE)

    @torch.inference_mode()
    def match(pa, pb):
        f0 = extractor.extract(load_image(pa).to(DEVICE))
        f1 = extractor.extract(load_image(pb).to(DEVICE))
        m = matcher({"image0": f0, "image1": f1})
        f0, f1, m = rbd(f0), rbd(f1), rbd(m)
        ij = m["matches"]
        return _np(f0["keypoints"][ij[:, 0]]), _np(f1["keypoints"][ij[:, 1]]), _np(m["scores"])

    return match


def build_sift(ratio=0.8, n_features=8000):
    import cv2

    sift = cv2.SIFT_create(nfeatures=n_features)

    def detect(p):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        # keep keypoints off the black area outside the circle
        mask = (im > 0).astype(np.uint8) * 255
        kp, desc = sift.detectAndCompute(im, mask)
        return np.array([k.pt for k in kp], np.float32), desc

    def match(pa, pb):
        kA, dA = detect(pa)
        kB, dB = detect(pb)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        ab = bf.knnMatch(dA, dB, k=2)
        ba = {m.queryIdx: m.trainIdx for m in bf.match(dB, dA)}
        keep, r = [], []
        for m, n in ab:  # Lowe ratio test + mutual nearest neighbour
            if m.distance < ratio * n.distance and ba.get(m.trainIdx) == m.queryIdx:
                keep.append((m.queryIdx, m.trainIdx))
                r.append(m.distance / n.distance)
        keep = np.array(keep, int).reshape(-1, 2)
        return kA[keep[:, 0]], kB[keep[:, 1]], 1 - np.array(r, np.float32)

    return match


def build_loftr(size=840):
    import kornia.feature as KF
    from PIL import Image

    model = KF.LoFTR(pretrained="outdoor").eval().to(DEVICE)

    def load(p):
        im = Image.open(p).convert("L")
        W, H = im.size
        # LoFTR works at 1/8 resolution; keep sides divisible by 8
        w, h = size, round(size * H / W / 8) * 8
        t = torch.from_numpy(np.array(im.resize((w, h), Image.LANCZOS))).float()[None, None] / 255
        return t.to(DEVICE), np.array([W / w, H / h], np.float32)

    @torch.inference_mode()
    def match(pa, pb):
        (tA, sA), (tB, sB) = load(pa), load(pb)
        out = model({"image0": tA, "image1": tB})
        # pixel centres: (x + 0.5) * scale - 0.5
        kA = (_np(out["keypoints0"]) + 0.5) * sA - 0.5
        kB = (_np(out["keypoints1"]) + 0.5) * sB - 0.5
        return kA, kB, _np(out["confidence"])

    return match


def build_roma():
    from PIL import Image
    from romatch import roma_outdoor

    # the fused correlation kernel has no build for this torch version; use the native path
    model = roma_outdoor(device=DEVICE, coarse_res=560, upsample_res=(864, 864), use_custom_corr=False)

    @torch.inference_mode()
    def match(pa, pb):
        (wA, hA), (wB, hB) = Image.open(pa).size, Image.open(pb).size
        warp, cert = model.match(pa, pb, device=DEVICE)
        m, c = model.sample(warp, cert, num=NUM_DENSE_SAMPLES)
        kA, kB = model.to_pixel_coordinates(m, hA, wA, hB, wB)
        return _np(kA), _np(kB), _np(c)

    return match


def build_romav2():
    from PIL import Image
    from romav2 import RoMaV2

    model = RoMaV2()

    @torch.inference_mode()
    def match(pa, pb):
        (wA, hA), (wB, hB) = Image.open(pa).size, Image.open(pb).size
        preds = model.match(pa, pb)
        m, overlap, *_ = model.sample(preds, NUM_DENSE_SAMPLES)
        kA, kB = model.to_pixel_coordinates(m, hA, wA, hB, wB)
        return _np(kA), _np(kB), _np(overlap).reshape(-1)

    return match


def build_mast3r(size=512):
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from mast3r.fast_nn import fast_reciprocal_NNs
    from mast3r.model import AsymmetricMASt3R
    from PIL import Image

    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(DEVICE).eval()

    @torch.inference_mode()
    def match(pa, pb):
        images = load_images([pa, pb], size=size, verbose=False)
        out = inference([tuple(images)], model, DEVICE, batch_size=1, verbose=False)
        d1, d2 = out["pred1"]["desc"][0], out["pred2"]["desc"][0]
        m0, m1 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=8, device=DEVICE,
                                     dist="dot", block_size=2 ** 13)
        H0, W0 = out["view1"]["true_shape"][0].tolist()
        H1, W1 = out["view2"]["true_shape"][0].tolist()
        ok = ((m0[:, 0] >= 3) & (m0[:, 0] < W0 - 3) & (m0[:, 1] >= 3) & (m0[:, 1] < H0 - 3)
              & (m1[:, 0] >= 3) & (m1[:, 0] < W1 - 3) & (m1[:, 1] >= 3) & (m1[:, 1] < H1 - 3))
        m0, m1 = m0[ok], m1[ok]
        c0 = _np(out["pred1"]["desc_conf"][0])[m0[:, 1], m0[:, 0]]
        c1 = _np(out["pred2"]["desc_conf"][0])[m1[:, 1], m1[:, 0]]
        # load_images resizes then center-crops; map back to the original pixel grid
        pts = []
        for p, m, (h, w) in ((pa, m0, (H0, W0)), (pb, m1, (H1, W1))):
            W, H = Image.open(p).size
            s = max(W, H) / size
            rw, rh = round(W / s), round(H / s)
            pts.append(((m + [(rw - w) / 2, (rh - h) / 2]) + 0.5) * [W / rw, H / rh] - 0.5)
        return pts[0].astype(np.float32), pts[1].astype(np.float32), np.minimum(c0, c1)

    return match


MATCHERS = {
    "loma-b": ("LoMa-B", lambda: build_loma("LoMaB")),
    "loma-r": ("LoMa-R", lambda: build_loma("LoMaR")),
    "loma-g": ("LoMa-G", lambda: build_loma("LoMaG")),
    "sp-lg": ("SuperPoint + LightGlue", lambda: build_lightglue("superpoint")),
    "aliked-lg": ("ALIKED + LightGlue", lambda: build_lightglue("aliked")),
    "loftr": ("LoFTR", build_loftr),
    "sift": ("SIFT (ratio + mutual NN)", build_sift),
    "roma": ("RoMa", build_roma),
    "romav2": ("RoMa v2", build_romav2),
    "mast3r": ("MASt3R", build_mast3r),
}


def build(name):
    return MATCHERS[name][1]()
