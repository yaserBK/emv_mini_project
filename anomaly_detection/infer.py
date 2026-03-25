"""
infer.py — Inference entry point.

Loads a saved calibration model and scores one or more cropped bottle-cap
images for anomalies using Mahalanobis distance against the fitted normal
distribution.

Usage — single image
--------------------
    python infer.py \\
        --calibration ./calibration.pkl \\
        --image       ./test_cap.jpg \\
        --threshold   99

Usage — directory of images
---------------------------
    python infer.py \\
        --calibration ./calibration.pkl \\
        --images-dir  ./test_caps/ \\
        --threshold   99

Exit codes
----------
    0  All images are GOOD (no anomalies detected).
    1  At least one image is ANOMALOUS.
    2  Fatal error (bad arguments, missing calibration file, etc.).
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from anomaly.distribution import mahalanobis_distances
from anomaly.features import (
    build_feature_extractor,
    build_transform,
    extract_features,
    find_images,
    load_image,
)
from anomaly.io import load_calibration

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("infer")

# Valid percentile threshold keys
_VALID_THRESHOLDS = {90, 95, 99}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description=(
            "Score bottle-cap images for anomalies using a saved calibration model.\n"
            "Provide either --image (single) or --images-dir (batch), not both."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        metavar="FILE",
        help="Path to a single image to score.",
    )
    input_group.add_argument(
        "--images-dir",
        metavar="DIR",
        help="Directory of images to score (processed recursively).",
    )

    parser.add_argument(
        "--calibration",
        required=True,
        metavar="FILE",
        help="Path to the calibration .pkl file produced by calibrate.py.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=99,
        choices=sorted(_VALID_THRESHOLDS),
        metavar="{90,95,99}",
        help="Percentile threshold to use for the GOOD/ANOMALOUS decision "
             "(90, 95, or 99).  Default: 99.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="PyTorch device for feature extraction (default: cpu).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Images per forward pass when processing a directory (default: 16).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING — keeps stdout clean).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

_COL_WIDTH = 55  # max filename width for aligned output


def _verdict(distance: float, threshold: float) -> str:
    """Return 'GOOD' or 'ANOMALOUS' based on distance vs threshold."""
    return "ANOMALOUS" if distance > threshold else "GOOD     "


def _print_header(threshold_value: float, percentile: int) -> None:
    print()
    print(
        f"  {'FILE':<{_COL_WIDTH}}  {'DISTANCE':>10}  {'THRESHOLD':>10}  VERDICT"
    )
    print(
        f"  {'-' * _COL_WIDTH}  {'-' * 10}  {'-' * 10}  {'-' * 9}"
    )


def _print_row(name: str, distance: float, threshold: float) -> None:
    verdict = _verdict(distance, threshold)
    # Truncate long names with an ellipsis
    display = name if len(name) <= _COL_WIDTH else "…" + name[-(  _COL_WIDTH - 1):]
    print(f"  {display:<{_COL_WIDTH}}  {distance:>10.4f}  {threshold:>10.4f}  {verdict}")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    """
    Run the inference pipeline.

    Returns:
        0 — all images are GOOD
        1 — at least one ANOMALOUS image found
        2 — fatal error
    """
    args = parse_args(argv)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ------------------------------------------------------------------
    # Load calibration model
    # ------------------------------------------------------------------
    try:
        cal = load_calibration(Path(args.calibration))
    except (FileNotFoundError, KeyError, Exception) as exc:
        logger.error("Failed to load calibration file: %s", exc)
        print(f"\nERROR: Cannot load calibration file '{args.calibration}': {exc}\n",
              file=sys.stderr)
        return 2

    mean = cal["mean"]
    inv_cov = cal["inv_cov"]
    thresholds = cal["thresholds"]

    threshold_key = str(args.threshold)
    if threshold_key not in thresholds:
        print(
            f"\nERROR: Threshold key '{threshold_key}' not found in calibration file. "
            f"Available: {sorted(thresholds.keys())}\n",
            file=sys.stderr,
        )
        return 2

    threshold_value = thresholds[threshold_key]

    # ------------------------------------------------------------------
    # Resolve device
    # ------------------------------------------------------------------
    try:
        device = torch.device(args.device)
        torch.zeros(1).to(device)
    except (RuntimeError, AssertionError) as exc:
        logger.warning("Device '%s' unavailable (%s). Falling back to CPU.", args.device, exc)
        device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Build feature extractor
    # ------------------------------------------------------------------
    transform = build_transform()
    extractor = build_feature_extractor(device)

    # ------------------------------------------------------------------
    # Collect images to score
    # ------------------------------------------------------------------
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f"\nERROR: Image file not found: '{image_path}'\n", file=sys.stderr)
            return 2
        image_paths = [image_path]
    else:
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            print(
                f"\nERROR: --images-dir '{images_dir}' is not a directory.\n",
                file=sys.stderr,
            )
            return 2
        image_paths = find_images(images_dir)
        if not image_paths:
            print(f"\nERROR: No images found in '{images_dir}'.\n", file=sys.stderr)
            return 2

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

    if len(valid_paths) == 0:
        print("\nERROR: No images could be loaded.\n", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # Apply PCA projection (if calibration was fitted with PCA)
    # ------------------------------------------------------------------
    if cal.get("pca_enabled", False):
        pca_mean = cal["pca_mean"]
        pca_components = cal["pca_components"]
        n_in = features.shape[1]
        n_out = pca_components.shape[1]
        logger.info("PCA: Projecting features from %d to %d dimensions", n_in, n_out)
        features = (features - pca_mean) @ pca_components

    # ------------------------------------------------------------------
    # Compute Mahalanobis distances
    # ------------------------------------------------------------------
    distances = mahalanobis_distances(features, mean, inv_cov)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    n_anomalous = 0

    print()
    print("=" * 90)
    print("  INFERENCE RESULTS")
    print(f"  Threshold : p{args.threshold} = {threshold_value:.4f}")
    if "metadata" in cal:
        meta = cal["metadata"]
        print(f"  Calibrated: {meta.get('calibration_date', 'unknown')[:19]}  "
              f"({meta.get('n_images_used', '?')} good images)")
    if cal.get("pca_enabled", False):
        print(f"  PCA       : {cal['pca_components'].shape[0]}d → "
              f"{cal['pca_n_components']}d  "
              f"({cal.get('pca_variance_explained', 0) * 100:.1f}% variance retained)")
    print("=" * 90)

    _print_header(threshold_value, args.threshold)

    for path, dist in zip(valid_paths, distances):
        _print_row(path.name, float(dist), threshold_value)
        if float(dist) > threshold_value:
            n_anomalous += 1

    # Skipped images
    n_skipped = len(image_paths) - len(valid_paths)

    # Directory summary
    if args.images_dir:
        n_total = len(valid_paths)
        print()
        print("-" * 90)
        print(f"  Total images scored : {n_total}")
        print(f"  GOOD                : {n_total - n_anomalous}")
        print(f"  ANOMALOUS           : {n_anomalous}")
        print(f"  Flag rate           : {n_anomalous / n_total * 100:.1f}%")
        if n_skipped:
            print(f"  Skipped (load fail) : {n_skipped}")
        print("-" * 90)

    print()

    return 1 if n_anomalous > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
