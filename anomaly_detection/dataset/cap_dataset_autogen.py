#!/usr/bin/env python3
"""
Bottle Cap Dataset Generator  v2
──────────────────────────────────
Augments source images first (stored independently), then runs the cap
detection pipeline on each augmented image to form the training dataset.

Pipeline order
────────────────────────────
START → source → augment full-res image → save → detect → crop → train/val → FIN.

Usage
─────
  python cap_dataset_gen_v2.py --img_dir images/ --out_dir dataset/
  python cap_dataset_gen_v2.py --img_dir images/ --out_dir dataset/ --n_aug 20 --val_frac 0.15

Output structure
────────────────
  dataset/
      augmented_sources/    ← full-res augmented input images (N per source)
      train/                ← crops detected from augmented sources
      val/
      metadata.json
      contact_sheet_train.jpg
      contact_sheet_val.jpg

All detection and augmentation logic lives in cap_detection_pipeline.py.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from cap_detection_pipeline import (
    AugParams,
    augment_image,
    make_contact_sheet,
    process_image,
)

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
