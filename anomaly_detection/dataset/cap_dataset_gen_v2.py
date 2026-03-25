#!/usr/bin/env python3
"""
Bottle Cap Dataset Generator  v2
──────────────────────────────────
Augments source images first (stored independently), then runs the cap
detection pipeline on each augmented image to form the training dataset.

Pipeline order change vs v1
────────────────────────────
  v1: source → detect → crop → augment augmented crop → train/val
  v2: source → augment full-res image → save → detect → crop → train/val

Usage
─────
  python cap_dataset_gen_v2.py --img_dir images/ --out_dir dataset/
  python cap_dataset_gen_v2.py --img_dir images/ --out_dir dataset/ --n_aug 20 --val_frac 0.15

Cap Detection Pipeline (identical to final notebook)
───────────────────────────────────────
  1. normalise_fast   – downsample → conditional gamma 0.4 lift → CLAHE
  2. blob_detect_fast – Otsu sat threshold + morphology @ 500px, circ gate 0.45
  3. find_cap_fast    – scale back + constrained Hough for elongated blobs
  4. find_rim_radius  – 36-angle radial saturation sampling @ 500px
  5. `make_masked_crop` – circle mask drawn at crop scale; neutral grey bg

Augmentation distributions (applied to full-res source images)
───────────────────────────
  gamma        log-normal  μ=0, σ=0.30  → typically 0.70×–1.43×  (applied to V channel)
  saturation   log-normal  μ=0, σ=0.28  → typically 0.76×–1.32×
  brightness   normal      μ=0, σ=12    → ±~24 luma shift (additive, clamped)
  rotation     uniform     0°–360°      (caps are rotationally symmetric)
  skew         uniform     ±8°          (perspective tilt from camera angle)
  noise        half-normal σ=6          → 0–~18 std Gaussian pixel noise
  blur         Bernoulli   p=0.25       → slight Gaussian blur (kernel 3 or 5)

Every augmentation is independently sampled per image, so the combinations
are genuinely random — not a fixed grid.

Output structure
────────────────
  dataset/
      augmented_sources/    ← full-res augmented input images (N per source)
          IMG_4082_aug_001.jpg
          ...
      train/                ← crops detected from augmented sources
          IMG_4082_aug_001.jpg
          ...
      val/
          IMG_4082_aug_015.jpg
          ...
      metadata.json
      contact_sheet_train.jpg
      contact_sheet_val.jpg
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ─── Pipeline constants (locked) ──────────────────────────────────────────────

DETECT_SIZE   = 500
CROP_SIZE     = 256
PADDING       = 0.08
BG_COLOUR     = (0, 0, 0)
MIN_CIRC      = 0.45
MIN_AREA_FRAC = 0.03
MAX_AREA_FRAC = 0.55
N_RIM_ANGLES  = 36
RIM_SAT_MAX   = 85
RIM_VAL_MIN   = 95
HOUGH_P2      = [35, 28, 22]

# ─── Detection pipeline ────────────────────────────────────────────────────────

def normalise_fast(img: np.ndarray) -> np.ndarray:
    """Downsample → conditional gamma 0.4 lift (dark images) → CLAHE."""
    small = cv2.resize(img, (DETECT_SIZE, DETECT_SIZE), interpolation=cv2.INTER_AREA)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.uint8)
    if hsv[..., 2].mean() < 100:
        lut     = np.array([(i / 255.0) ** 0.4 * 255 for i in range(256)], np.uint8)
        hsv[..., 2] = cv2.LUT(hsv[..., 2], lut)
    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def blob_detect_fast(norm_small: np.ndarray, img_area_small: int) -> Optional[tuple]:
    """Find the best cap blob in a 500px normalised image."""
    h, w           = norm_small.shape[:2]
    cx_img, cy_img = w / 2, h / 2

    hsv = cv2.cvtColor(norm_small, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]

    otsu_val, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds   = sorted(set([max(30, int(otsu_val)), 55, 65, 75, 85]))

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    for s_thresh in thresholds:
        mask = ((sat < s_thresh) & (val > 80)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2)

        ctrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score, best = 0, None

        for c in ctrs:
            area = cv2.contourArea(c)
            if not (MIN_AREA_FRAC * img_area_small < area < MAX_AREA_FRAC * img_area_small):
                continue
            if len(c) < 5:
                continue
            ell             = cv2.fitEllipse(c)
            (ecx, ecy), (MA, ma), _ = ell
            circ            = min(MA, ma) / max(MA, ma)
            if circ < MIN_CIRC:
                continue
            dist  = np.hypot((ecx - cx_img) / w, (ecy - cy_img) / h)
            score = circ * max(0, 1 - 2 * dist) * np.log(area)
            if score > best_score:
                best_score, best = score, ell

        if best:
            return best
    return None


def find_cap_fast(img: np.ndarray) -> Optional[tuple]:
    """Detect at 500px, scale back to full-res; Hough-refine elongated blobs."""
    h, w   = img.shape[:2]
    scale  = h / DETECT_SIZE

    norm_s = normalise_fast(img)
    hs, ws = norm_s.shape[:2]
    area_s = hs * ws

    ell_s = blob_detect_fast(norm_s, area_s)
    if ell_s is None:
        raw_s = cv2.resize(img, (DETECT_SIZE, DETECT_SIZE), interpolation=cv2.INTER_AREA)
        ell_s = blob_detect_fast(raw_s, area_s)
    if ell_s is None:
        return None

    (ecx, ecy), (MA, ma), angle = ell_s
    ecx_f, ecy_f = ecx * scale, ecy * scale
    MA_f,  ma_f  = MA * scale,  ma * scale
    circ         = min(MA_f, ma_f) / max(MA_f, ma_f)

    if circ < 0.88:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        win  = int(max(MA_f, ma_f) * 0.75)
        x1 = max(int(ecx_f) - win, 0);  y1 = max(int(ecy_f) - win, 0)
        x2 = min(int(ecx_f) + win, w);  y2 = min(int(ecy_f) + win, h)
        roi   = cv2.GaussianBlur(gray[y1:y2, x1:x2], (13, 13), 0)
        min_r = int(min(MA_f, ma_f) / 2 * 0.7)
        max_r = int(max(MA_f, ma_f) / 2 * 1.3)
        for p2 in HOUGH_P2:
            circles = cv2.HoughCircles(
                roi, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=200, param1=70, param2=p2,
                minRadius=min_r, maxRadius=max_r
            )
            if circles is not None:
                cx, cy, r = np.round(circles[0][0]).astype(int)
                return ((float(cx + x1), float(cy + y1)),
                        (float(r * 2), float(r * 2)), 0.0)

    return ((ecx_f, ecy_f), (MA_f, ma_f), angle)


def find_rim_radius(img: np.ndarray, cx: float, cy: float, r_approx: int) -> int:
    """Radial saturation sampling at 500px; returns true outer rim radius (full-res px)."""
    small   = cv2.resize(img, (DETECT_SIZE, DETECT_SIZE), interpolation=cv2.INTER_AREA)
    h0, w0  = img.shape[:2]
    hs, ws  = small.shape[:2]
    sx      = ws / w0
    cx_s, cy_s = cx * sx, cy * sx
    r_s     = r_approx * sx

    norm_s  = normalise_fast(img)
    hsv     = cv2.cvtColor(norm_s, cv2.COLOR_BGR2HSV)
    sat     = hsv[..., 1]
    val     = hsv[..., 2]

    r_min = int(r_s * 0.60)
    r_max = int(r_s * 1.65)
    outer = []

    for a_deg in np.linspace(0, 360, N_RIM_ANGLES, endpoint=False):
        a        = np.deg2rad(a_deg)
        last_cap = None
        for rr in range(r_min, r_max, 2):
            px = int(cx_s + rr * np.cos(a))
            py = int(cy_s + rr * np.sin(a))
            if not (0 <= px < ws and 0 <= py < hs):
                break
            if sat[py, px] < RIM_SAT_MAX and val[py, px] > RIM_VAL_MIN:
                last_cap = rr
            elif last_cap is not None:
                break
        if last_cap is not None:
            outer.append(last_cap / sx)

    if not outer:
        return r_approx
    clean = [v for v in outer if v >= r_approx * 0.85]
    if len(clean) < 6:
        return int(np.percentile(outer, 90))
    return int(np.median(clean))


def make_masked_crop(
    img: np.ndarray,
    cx: float, cy: float,
    r_true: int,
    crop_size: int = CROP_SIZE,
    padding: float = PADDING,
    bg: Tuple = BG_COLOUR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Circle-masked 256×256 crop; mask drawn at crop scale (fast)."""
    h, w   = img.shape[:2]
    r_pad  = int(r_true * (1 + padding))

    x1 = max(int(cx) - r_pad, 0);  y1 = max(int(cy) - r_pad, 0)
    x2 = min(int(cx) + r_pad, w);  y2 = min(int(cy) + r_pad, h)
    ch, cw = y2 - y1, x2 - x1
    side   = max(ch, cw)

    sq = np.full((side, side, 3), bg, np.uint8)
    ox = (side - cw) // 2
    oy = (side - ch) // 2
    sq[oy:oy + ch, ox:ox + cw] = img[y1:y2, x1:x2]

    crop_n = cv2.resize(sq, (crop_size, crop_size), interpolation=cv2.INTER_AREA)

    mq      = np.zeros((crop_size, crop_size), np.uint8)
    scale_m = crop_size / side
    cx_m    = int((int(cx) - x1 + ox) * scale_m)
    cy_m    = int((int(cy) - y1 + oy) * scale_m)
    r_m     = int(r_true * scale_m)
    cv2.circle(mq, (cx_m, cy_m), r_m, 255, -1)
    crop_n[mq == 0] = bg

    return crop_n, mq


def process_image(img: np.ndarray) -> Optional[dict]:
    """Full pipeline: image → {crop, mask, centre, r_true} or None."""
    ell = find_cap_fast(img)
    if ell is None:
        return None
    (cx, cy), (MA, ma), _ = ell
    r_approx = int(max(MA, ma) / 2)
    r_true   = find_rim_radius(img, cx, cy, r_approx)
    crop, mask = make_masked_crop(img, cx, cy, r_true)
    return dict(crop=crop, mask=mask, centre=(int(cx), int(cy)), r_true=r_true)


# ─── Augmentation engine ───────────────────────────────────────────────────────

class AugParams:
    """Holds one full set of randomly drawn augmentation parameters."""

    def __init__(self, rng: np.random.Generator):
        # Gamma: log-normal, σ=0.30 → ~[0.55, 1.82], tails clipped to [0.4, 2.5]
        self.gamma      = float(np.clip(np.exp(rng.normal(0, 0.30)), 0.40, 2.50))

        # Saturation scale: log-normal σ=0.28 → ~[0.60, 1.67]
        self.sat_scale  = float(np.clip(np.exp(rng.normal(0, 0.28)), 0.40, 2.50))

        # Brightness shift: normal μ=0, σ=12, additive to V channel
        self.brightness = float(np.clip(rng.normal(0, 12), -40, 40))

        # Rotation: uniform [0, 360)  — caps are rotationally symmetric
        self.rotation   = float(rng.uniform(0, 360))

        # Skew (perspective tilt): uniform ±8° → small affine skew
        self.skew_x     = float(rng.uniform(-8, 8))   # degrees of horizontal lean
        self.skew_y     = float(rng.uniform(-8, 8))   # degrees of vertical lean

        # Gaussian noise std: half-normal σ=6 → 0–~18 pixel noise
        self.noise_std  = float(np.abs(rng.normal(0, 6)))

        # Blur: applied with probability 0.25
        self.blur       = rng.random() < 0.25
        self.blur_k     = int(rng.choice([3, 5]))      # kernel size if blur applied

    def describe(self) -> dict:
        return {
            "gamma":      round(self.gamma, 3),
            "sat_scale":  round(self.sat_scale, 3),
            "brightness": round(self.brightness, 2),
            "rotation":   round(self.rotation, 2),
            "skew_x":     round(self.skew_x, 2),
            "skew_y":     round(self.skew_y, 2),
            "noise_std":  round(self.noise_std, 3),
            "blur":       self.blur,
            "blur_k":     self.blur_k,
        }


def apply_colour(img: np.ndarray, p: AugParams) -> np.ndarray:
    """Apply gamma, saturation, brightness in HSV space."""
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Gamma on V channel
    hsv[..., 2] = np.clip((hsv[..., 2] / 255.0) ** (1.0 / p.gamma) * 255.0, 0, 255)

    # Saturation scale
    hsv[..., 1] = np.clip(hsv[..., 1] * p.sat_scale, 0, 255)

    # Brightness additive shift
    hsv[..., 2] = np.clip(hsv[..., 2] + p.brightness, 0, 255)

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_geometry(img: np.ndarray, p: AugParams, bg: Tuple = BG_COLOUR) -> np.ndarray:
    """Apply rotation + perspective skew, keeping the circle inside the frame."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # ── Rotation ──────────────────────────────────────────────────────────────
    M_rot = cv2.getRotationMatrix2D((cx, cy), p.rotation, 1.0)
    img   = cv2.warpAffine(img, M_rot, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=bg)

    # ── Perspective skew ──────────────────────────────────────────────────────
    # Convert skew angles to a fraction of the image dimension for the shift
    dx = np.tan(np.deg2rad(abs(p.skew_x))) * h * 0.5
    dy = np.tan(np.deg2rad(abs(p.skew_y))) * w * 0.5

    sx = np.sign(p.skew_x)  # direction of horizontal lean
    sy = np.sign(p.skew_y)

    # Source corners (full image)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Destination corners: top edge leans left/right, bottom opposite
    dst = np.float32([
        [max(0, sx * dx),      max(0, sy * dy)],
        [min(w, w - sx * dx),  max(0, -sy * dy)],
        [min(w, w + sx * dx),  min(h, h - sy * dy)],
        [max(0, -sx * dx),     min(h, h + sy * dy)],
    ])

    M_persp = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, M_persp, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=bg)
    return img


def apply_noise(img: np.ndarray, p: AugParams) -> np.ndarray:
    """Add Gaussian noise; skip if std ≈ 0."""
    if p.noise_std < 0.5:
        return img
    noise = np.random.normal(0, p.noise_std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_blur(img: np.ndarray, p: AugParams) -> np.ndarray:
    """Slight Gaussian blur if selected."""
    if not p.blur:
        return img
    k = p.blur_k if p.blur_k % 2 == 1 else p.blur_k + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def augment_image(img: np.ndarray, p: AugParams) -> np.ndarray:
    """Apply the full augmentation chain to an image of any size."""
    aug = apply_colour(img, p)
    aug = apply_geometry(aug, p, bg=BG_COLOUR)
    aug = apply_noise(aug, p)
    aug = apply_blur(aug, p)
    return aug


# ─── Contact sheet ─────────────────────────────────────────────────────────────

def make_contact_sheet(
    images: list,    # list of (label_str, np.ndarray[H×W×3])
    cols:   int = 10,
    tile:   int = 128,
    label_h: int = 18,
) -> np.ndarray:
    rows = (len(images) + cols - 1) // cols
    tw   = tile
    th   = tile + label_h
    sheet = np.full((rows * (th + 2) - 2, cols * (tw + 2) - 2, 3), 20, np.uint8)

    for i, (label, img) in enumerate(images):
        row, col = divmod(i, cols)
        y0 = row * (th + 2)
        x0 = col * (tw + 2)

        bar = np.full((label_h, tw, 3), 45, np.uint8)
        cv2.putText(bar, label[:16], (2, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        sheet[y0:y0 + label_h, x0:x0 + tw] = bar

        thumb = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        sheet[y0 + label_h:y0 + th, x0:x0 + tw] = thumb

    return sheet


# ─── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cap detection + training dataset generator v2 "
                    "(augment source images first, then detect)"
    )
    p.add_argument("--img_dir",  default="images",  help="Input image directory")
    p.add_argument("--out_dir",  default="dataset", help="Output dataset directory")
    p.add_argument("--n_aug",    type=int, default=12,
                   help="Augmented images to generate per source image (default: 12)")
    p.add_argument("--val_frac", type=float, default=0.15,
                   help="Fraction of augmented images to put in val/ (default: 0.15)")
    p.add_argument("--seed",     type=int, default=None,
                   help="Random seed for reproducibility (default: random)")
    p.add_argument("--crop_size",type=int, default=256,
                   help="Output crop resolution (default: 256)")
    p.add_argument("--ext",      default="jpg",
                   help="Image extension to glob (default: jpg)")
    return p.parse_args()


def main():
    args    = parse_args()
    seed    = args.seed if args.seed is not None else int(time.time()) % (2**31)
    rng     = np.random.default_rng(seed)
    random.seed(seed)

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)

    # ── Find source images ─────────────────────────────────────────────────────
    exts = [args.ext.lstrip(".")]
    sources = sorted([f for f in img_dir.iterdir()
                      if f.suffix.lstrip(".").lower() in exts])
    if not sources:
        print(f"No .{args.ext} images found in {img_dir}")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Cap Dataset Generator  v2")
    print(f"{'─'*60}")
    print(f"  Source images : {len(sources)}")
    print(f"  Augs per image: {args.n_aug}")
    print(f"  Val fraction  : {args.val_frac:.0%}")
    print(f"  Random seed   : {seed}")
    print(f"  Output        : {out_dir}")
    print(f"{'─'*60}\n")

    # ── Create output dirs ─────────────────────────────────────────────────────
    for d in ["augmented_sources", "train", "val"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    # ── Stage 1: augment source images and save ────────────────────────────────
    print("Stage 1 — Augmenting source images\n")

    # Each entry: (stem, aug_index, aug_path, is_val, AugParams)
    augmented_entries = []
    skipped_sources   = []

    total_aug = len(sources) * args.n_aug

    for path in sources:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  SKIP  {path.name}  (cannot read)")
            skipped_sources.append(path.name)
            continue

        stem = path.stem

        for i in range(1, args.n_aug + 1):
            p   = AugParams(rng)
            aug = augment_image(img, p)

            is_val = (i / args.n_aug) > (1 - args.val_frac)
            fname  = f"{stem}_aug_{i:03d}.jpg"
            aug_path = out_dir / "augmented_sources" / fname
            cv2.imwrite(str(aug_path), aug)

            augmented_entries.append((stem, i, aug_path, is_val, p))

            done = augmented_entries.__len__()
            pct  = done / total_aug * 100
            print(f"  [{done:4d}/{total_aug}  {pct:5.1f}%]  augmented_sources/{fname}",
                  end="\r")

    print(f"\n\n  {len(augmented_entries)} augmented images saved to"
          f" {out_dir / 'augmented_sources'}\n")

    if not augmented_entries:
        print("No augmented images produced — exiting.")
        sys.exit(1)

    # ── Stage 2: detect caps in augmented source images ────────────────────────
    print("Stage 2 — Detecting caps in augmented source images\n")

    metadata = {
        "seed":     seed,
        "n_aug":    args.n_aug,
        "val_frac": args.val_frac,
        "failed":   [],
        "train":    [],
        "val":      [],
    }

    train_thumbs = []
    val_thumbs   = []
    total_detect = len(augmented_entries)

    for idx, (stem, aug_index, aug_path, is_val, p) in enumerate(augmented_entries, 1):
        aug_img = cv2.imread(str(aug_path))
        result  = process_image(aug_img)

        pct = idx / total_detect * 100
        if result is None:
            print(f"  [{idx:4d}/{total_detect}  {pct:5.1f}%]  FAIL  {aug_path.name}"
                  f"  (no cap detected)")
            metadata["failed"].append(aug_path.name)
            continue

        crop   = result["crop"]
        split  = "val" if is_val else "train"
        dst    = out_dir / split / aug_path.name
        cv2.imwrite(str(dst), crop)

        entry = {
            "file":       str(dst.relative_to(out_dir)),
            "source":     stem,
            "aug_index":  aug_index,
            "params":     p.describe(),
        }

        if is_val:
            metadata["val"].append(entry)
            val_thumbs.append((f"{stem[:8]}_{aug_index}", crop))
        else:
            metadata["train"].append(entry)
            train_thumbs.append((f"{stem[:8]}_{aug_index}", crop))

        cx, cy = result["centre"]
        print(f"  [{idx:4d}/{total_detect}  {pct:5.1f}%]  {split}/{aug_path.name}"
              f"  r={result['r_true']}px  centre=({cx},{cy})", end="\r")

    n_failed = len(metadata["failed"])
    n_ok     = total_detect - n_failed
    print(f"\n\n  {n_ok}/{total_detect} detected"
          f"  ({n_failed} failed — no cap found in augmented image)\n")

    # ── Stage 3: save metadata + contact sheets ────────────────────────────────
    print("Stage 3 — Writing metadata and contact sheets\n")

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  metadata.json saved")

    if train_thumbs:
        sheet = make_contact_sheet(train_thumbs, cols=12, tile=128)
        cv2.imwrite(str(out_dir / "contact_sheet_train.jpg"), sheet)
        print(f"  contact_sheet_train.jpg saved  ({sheet.shape[1]}×{sheet.shape[0]}px)")

    if val_thumbs:
        sheet = make_contact_sheet(val_thumbs, cols=12, tile=128)
        cv2.imwrite(str(out_dir / "contact_sheet_val.jpg"), sheet)
        print(f"  contact_sheet_val.jpg saved  ({sheet.shape[1]}×{sheet.shape[0]}px)")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Done.")
    print(f"  Augmented sources : {len(augmented_entries)}"
          f"  →  {out_dir}/augmented_sources/")
    print(f"  Train images      : {len(metadata['train'])}")
    print(f"  Val images        : {len(metadata['val'])}")
    print(f"  Metadata          : {meta_path}")
    if metadata["failed"]:
        print(f"\n  ⚠ Failed to detect cap in {n_failed} augmented image(s)")
    if skipped_sources:
        print(f"  ⚠ Skipped unreadable source(s): {skipped_sources}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
