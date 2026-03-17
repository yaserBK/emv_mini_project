"""
calibrate.py — Calibration (training) entry point.

Given a directory of known-good cropped bottle-cap images, this script:
  1. Finds all supported images recursively.
  2. Extracts 512-dimensional ResNet-18 feature vectors in batches.
  3. Fits a multivariate Gaussian with Ledoit-Wolf shrinkage.
  4. Saves the calibration model (mean, inverse covariance, thresholds,
     calibration distances, and provenance metadata) to a pickle file.
  5. Prints a human-readable summary to stdout.

Usage
-----
    python calibrate.py \\
        --images-dir ./good_caps/ \\
        --output     ./calibration.pkl \\
        --device     cuda \\
        --batch-size 16

See ``python calibrate.py --help`` for all options.
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

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrate")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="calibrate.py",
        description=(
            "Calibrate the anomaly detector from a directory of known-good images.\n"
            "Produces a calibration .pkl file for use with infer.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        metavar="DIR",
        help="Directory containing known-good (normal) cropped cap images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="Path to write the calibration model (e.g. calibration.pkl).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="PyTorch device for feature extraction (default: cpu). "
             "Use 'cuda' for GPU, 'cuda:0', 'cpu', etc.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Number of images per forward pass (default: 16).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """
    Run the calibration pipeline.

    Returns:
        Exit code: 0 on success, non-zero on error.
    """
    args = parse_args(argv)

    # Apply log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    images_dir = Path(args.images_dir).resolve()
    output_path = Path(args.output).resolve()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not images_dir.is_dir():
        logger.error("--images-dir '%s' is not a directory or does not exist.", images_dir)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve device
    # ------------------------------------------------------------------
    try:
        device = torch.device(args.device)
        # Probe device availability
        torch.zeros(1).to(device)
    except (RuntimeError, AssertionError) as exc:
        logger.error("Could not use device '%s': %s", args.device, exc)
        logger.error("Falling back to CPU.")
        device = torch.device("cpu")

    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Discover images
    # ------------------------------------------------------------------
    image_paths = find_images(images_dir)

    if len(image_paths) == 0:
        logger.error("No images found in '%s'. Nothing to calibrate.", images_dir)
        return 1

    if len(image_paths) < 10:
        logger.warning(
            "Only %d image(s) found.  Calibration will be unreliable. "
            "Recommend at least 50–100 good images.",
            len(image_paths),
        )

    # ------------------------------------------------------------------
    # Build feature extractor
    # ------------------------------------------------------------------
    transform = build_transform()
    extractor = build_feature_extractor(device)

    # ------------------------------------------------------------------
    # Extract features
    # ------------------------------------------------------------------
    features, valid_paths = extract_features(
        image_paths,
        extractor,
        transform,
        device,
        batch_size=args.batch_size,
    )

    n_found = len(image_paths)
    n_processed = len(valid_paths)
    n_failed = n_found - n_processed

    if n_processed == 0:
        logger.error("No images could be processed. Cannot calibrate.")
        return 1

    # ------------------------------------------------------------------
    # Fit distribution
    # ------------------------------------------------------------------
    model = fit_distribution(features)

    # ------------------------------------------------------------------
    # Attach metadata
    # ------------------------------------------------------------------
    model["metadata"] = {
        "calibration_date": datetime.now(timezone.utc).isoformat(),
        "images_dir": str(images_dir),
        "n_images_found": n_found,
        "n_images_used": n_processed,
        "n_images_failed": n_failed,
        "feature_extractor": "resnet18-imagenet1k-v1-avgpool",
        "feature_dim": FEATURE_DIM,
        "device": str(device),
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_calibration(model, output_path)

    # ------------------------------------------------------------------
    # Summary (human-readable stdout)
    # ------------------------------------------------------------------
    cal_dist = model["calibration_distances"]
    thresholds = model["thresholds"]

    print()
    print("=" * 60)
    print("  CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"  Images dir       : {images_dir}")
    print(f"  Images found     : {n_found}")
    print(f"  Images processed : {n_processed}")
    if n_failed:
        print(f"  Images skipped   : {n_failed}  (load failures)")
    print(f"  Feature extractor: ResNet-18 avgpool (dim={FEATURE_DIM})")
    print(f"  Device           : {device}")
    print(f"  L-W shrinkage α  : {model['shrinkage_alpha']:.4f}")
    print()
    print("  Calibration Mahalanobis distances:")
    print(f"    mean  : {cal_dist.mean():.4f}")
    print(f"    std   : {cal_dist.std():.4f}")
    print(f"    min   : {cal_dist.min():.4f}")
    print(f"    max   : {cal_dist.max():.4f}")
    print()
    print("  Decision thresholds (percentiles of calibration distances):")
    print(f"    p90 threshold : {thresholds['90']:.4f}")
    print(f"    p95 threshold : {thresholds['95']:.4f}")
    print(f"    p99 threshold : {thresholds['99']:.4f}")
    print()
    print(f"  Calibration model saved to: {output_path}")
    print("=" * 60)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
