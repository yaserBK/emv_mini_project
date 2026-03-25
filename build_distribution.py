#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_distribution.py — Build an anomaly detection distribution from known-good crops.

Takes a directory of known-good cropped bottle-cap images (produced by
preprocess.py), extracts ResNet-18 features, optionally reduces
dimensionality via PCA, fits a multivariate Gaussian with Ledoit-Wolf
shrinkage, and saves the result as a calibration .pkl file for use with
inference.py and video_inference.py.

Usage — default (PCA retaining 95% variance)
─────────────────────────────────────────────
    python build_distribution.py --images-dir crops/train/ --output distribution.pkl

Usage — retain 99% variance
─────────────────────────────
    python build_distribution.py --images-dir crops/train/ --output distribution.pkl \\
        --pca-variance 0.99

Usage — fixed component count
──────────────────────────────
    python build_distribution.py --images-dir crops/train/ --output distribution.pkl \\
        --pca-components 64

Usage — no PCA (full 512-dim space)
────────────────────────────────────
    python build_distribution.py --images-dir crops/train/ --output distribution.pkl \\
        --no-pca

Exit codes
──────────
    0  Success.
    1  Fatal error (no images, no detections, etc.).
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from anomaly.distribution import fit_distribution
from anomaly.features import (
    FEATURE_DIM,
    build_feature_extractor,
    build_transform,
    extract_features,
    find_images,
)
from anomaly.io import save_calibration
from anomaly.pca import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_distribution")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build_distribution.py",
        description=(
            "Build an anomaly detection distribution from known-good cap crops.\n"
            "Produces a .pkl calibration file for use with inference.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images-dir", required=True, metavar="DIR",
        help="Directory of known-good cropped cap images.",
    )
    parser.add_argument(
        "--output", required=True, metavar="FILE",
        help="Output path for the calibration .pkl file.",
    )
    parser.add_argument(
        "--device", default="cpu", metavar="DEVICE",
        help="PyTorch device for feature extraction (default: cpu).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, metavar="N",
        help="Images per forward pass (default: 16).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    pca_group = parser.add_mutually_exclusive_group()
    pca_group.add_argument(
        "--pca-components", type=int, metavar="N",
        help="Retain exactly N principal components.",
    )
    pca_group.add_argument(
        "--pca-variance", type=float, metavar="FLOAT",
        help="Retain enough components for this fraction of variance (default: 0.95).",
    )
    pca_group.add_argument(
        "--no-pca", action="store_true",
        help="Disable PCA; operate in the full 512-dim feature space.",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    images_dir  = Path(args.images_dir).resolve()
    output_path = Path(args.output).resolve()

    if not images_dir.is_dir():
        logger.error("--images-dir '%s' does not exist.", images_dir)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        device = torch.device(args.device)
        torch.zeros(1).to(device)
    except (RuntimeError, AssertionError) as exc:
        logger.warning("Device '%s' unavailable (%s) — falling back to CPU.", args.device, exc)
        device = torch.device("cpu")

    image_paths = find_images(images_dir)
    if not image_paths:
        logger.error("No images found in '%s'.", images_dir)
        return 1
    if len(image_paths) < 10:
        logger.warning(
            "Only %d image(s) found. Recommend at least 50–100 for reliable results.",
            len(image_paths),
        )

    transform = build_transform()
    extractor = build_feature_extractor(device)

    features, valid_paths = extract_features(
        image_paths, extractor, transform, device, batch_size=args.batch_size,
    )

    if len(valid_paths) == 0:
        logger.error("No images could be processed.")
        return 1

    # PCA
    pca_data = {"pca_enabled": False}

    if not args.no_pca:
        if args.pca_components is not None:
            pca = PCA(n_components=args.pca_components)
        else:
            variance_target = args.pca_variance if args.pca_variance is not None else 0.95
            pca = PCA(variance_threshold=variance_target)

        features = pca.fit_transform(features)
        _, cumulative = pca.explained_variance_ratio()

        pca_data = {
            "pca_enabled":          True,
            "pca_mean":             pca.mean_,
            "pca_components":       pca.components_,
            "pca_eigenvalues":      pca.eigenvalues_,
            "pca_n_components":     pca.n_components_,
            "pca_variance_explained": float(cumulative[-1]),
        }
        logger.info(
            "PCA: %d → %d dimensions (%.1f%% variance)",
            FEATURE_DIM, pca.n_components_, cumulative[-1] * 100,
        )

    model = fit_distribution(features)
    model.update(pca_data)
    model["metadata"] = {
        "calibration_date":  datetime.now(timezone.utc).isoformat(),
        "images_dir":        str(images_dir),
        "n_images_found":    len(image_paths),
        "n_images_used":     len(valid_paths),
        "n_images_failed":   len(image_paths) - len(valid_paths),
        "feature_extractor": "resnet18-imagenet1k-v1-avgpool",
        "feature_dim":       FEATURE_DIM,
        "device":            str(device),
        "pca_enabled":       pca_data["pca_enabled"],
        "pca_n_components":  pca_data.get("pca_n_components", FEATURE_DIM),
    }

    save_calibration(model, output_path)

    cal_dist   = model["calibration_distances"]
    thresholds = model["thresholds"]

    print()
    print("=" * 60)
    print("  BUILD DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"  Images dir       : {images_dir}")
    print(f"  Images found     : {len(image_paths)}")
    print(f"  Images processed : {len(valid_paths)}")
    if len(image_paths) - len(valid_paths):
        print(f"  Images skipped   : {len(image_paths) - len(valid_paths)}")
    print(f"  Feature extractor: ResNet-18 avgpool (dim={FEATURE_DIM})")
    print(f"  Device           : {device}")
    print()
    if pca_data["pca_enabled"]:
        print("  PCA              : enabled")
        print(f"  PCA dims         : {FEATURE_DIM} → {pca_data['pca_n_components']}")
        print(f"  PCA variance     : {pca_data['pca_variance_explained'] * 100:.1f}% retained")
    else:
        print(f"  PCA              : disabled (full {FEATURE_DIM}-dim)")
    print(f"  L-W shrinkage α  : {model['shrinkage_alpha']:.4f}")
    print()
    print("  Mahalanobis distances (calibration set):")
    print(f"    mean : {cal_dist.mean():.4f}")
    print(f"    std  : {cal_dist.std():.4f}")
    print(f"    min  : {cal_dist.min():.4f}")
    print(f"    max  : {cal_dist.max():.4f}")
    print()
    print("  Decision thresholds:")
    print(f"    p90 : {thresholds['90']:.4f}")
    print(f"    p95 : {thresholds['95']:.4f}")
    print(f"    p99 : {thresholds['99']:.4f}")
    print()
    print(f"  Saved to: {output_path}")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
