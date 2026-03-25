#!/usr/bin/env python3
"""
preprocess.py — Detect, crop, and augment bottle cap images.

Runs the cap detection pipeline on one image or a directory of images,
producing circle-masked 256×256 crops and optionally augmented variants
for use as input to build_distribution.py.

Usage — single image, crop only
────────────────────────────────
    python preprocess.py --image cap.jpg --out-dir crops/

Usage — single image, crop + augmentations
──────────────────────────────────────────
    python preprocess.py --image cap.jpg --out-dir crops/ --n-aug 10

Usage — directory, full dataset generation
──────────────────────────────────────────
    python preprocess.py --images-dir raw/ --out-dir dataset/ --n-aug 12

Usage — reproducible run
─────────────────────────
    python preprocess.py --images-dir raw/ --out-dir dataset/ --n-aug 12 --seed 42

Output (directory mode)
────────────────────────
    dataset/
        train/                  ← augmented crops (training)
        val/                    ← augmented crops (validation)
        metadata.json
        contact_sheet_train.jpg
        contact_sheet_val.jpg

Exit codes
──────────
    0  Success.
    1  No cap detected (single image mode).
    2  Fatal error (bad arguments, unreadable input, etc.).
"""

import argparse
import sys
from pathlib import Path

import cv2

from anomaly.preprocess import process_image, process_dir


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preprocess.py",
        description=(
            "Detect, crop, and augment bottle cap images.\n"
            "Provide either --image (single) or --images-dir (batch), not both."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image",      metavar="FILE", help="Single source image.")
    input_group.add_argument("--images-dir", metavar="DIR",  help="Directory of source images.")

    parser.add_argument(
        "--out-dir", required=True, metavar="DIR",
        help="Output directory for crops / dataset.",
    )
    parser.add_argument(
        "--n-aug", type=int, default=0, metavar="N",
        help="Augmented variants to generate per image (default: 0 — crop only).",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15, metavar="FLOAT",
        help="Fraction of augments to put in val/ (directory mode, default: 0.15).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Random seed for reproducibility (default: random).",
    )
    parser.add_argument(
        "--ext", default="jpg", metavar="EXT",
        help="Image extension to glob in directory mode (default: jpg).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)

    # ── Directory mode ────────────────────────────────────────────────────────
    if args.images_dir:
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            print(f"\nERROR: --images-dir '{images_dir}' is not a directory.\n",
                  file=sys.stderr)
            return 2

        metadata = process_dir(
            input_dir=images_dir,
            output_dir=out_dir,
            n_aug=args.n_aug,
            val_frac=args.val_frac,
            seed=args.seed,
            ext=args.ext,
        )
        return 0 if metadata else 2

    # ── Single image mode ─────────────────────────────────────────────────────
    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"\nERROR: Image file not found: '{image_path}'\n", file=sys.stderr)
        return 2

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"\nERROR: Cannot read image: '{image_path}'\n", file=sys.stderr)
        return 2

    import numpy as np
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    results = process_image(bgr, n_aug=args.n_aug, rng=rng)

    if not results:
        print(f"\nNo cap detected in '{image_path}'.\n", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    for i, item in enumerate(results):
        if item['aug_params'] is None:
            fname = f"{stem}_crop.jpg"
        else:
            fname = f"{stem}_aug_{i:03d}.jpg"
        cv2.imwrite(str(out_dir / fname), item['crop'])

    cx, cy = results[0]['centre']
    print(f"  Detected cap: r={results[0]['r_true']}px  centre=({cx},{cy})")
    print(f"  Saved {len(results)} image(s) to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
