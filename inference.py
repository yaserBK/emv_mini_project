#!/usr/bin/env python3
"""
inference.py -- Score bottle cap images for anomalies.

Loads a calibration model and scores one or more pre-cropped bottle-cap
images using Mahalanobis distance against the fitted normal distribution.

Usage -- single image
---------------------
    python inference.py --calibration distribution.pkl --image test_cap.jpg

Usage -- directory of images
-----------------------------
    python inference.py --calibration distribution.pkl --images-dir test_caps/

Usage -- lower threshold (more sensitive)
-----------------------------------------
    python inference.py --calibration distribution.pkl --images-dir test_caps/ \\
        --threshold 90

Exit codes
----------
    0  All images are GOOD.
    1  At least one ANOMALOUS image found.
    2  Fatal error (bad arguments, missing file, etc.).
"""

import argparse
import logging
import sys
from pathlib import Path

from anomaly.detector import AnomalyDetector
from anomaly.features import find_images

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")

_VALID_THRESHOLDS = {90, 95, 99}
_COL_WIDTH = 55


def parse_args(argv=None)  :
    parser = argparse.ArgumentParser(
        prog="inference.py",
        description=(
            "Score bottle-cap crops for anomalies using a saved calibration model.\n"
            "Provide either --image (single) or --images-dir (batch), not both."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image",      metavar="FILE", help="Single image to score.")
    input_group.add_argument("--images-dir", metavar="DIR",  help="Directory of images to score.")

    parser.add_argument(
        "--calibration", required=True, metavar="FILE",
        help="Calibration .pkl file produced by build_distribution.py.",
    )
    parser.add_argument(
        "--threshold", type=int, default=99,
        choices=sorted(_VALID_THRESHOLDS), metavar="{90,95,99}",
        help="Percentile threshold for GOOD/ANOMALOUS decision (default: 99).",
    )
    parser.add_argument(
        "--device", default="cpu", metavar="DEVICE",
        help="PyTorch device for feature extraction (default: cpu).",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser.parse_args(argv)


def _print_header()  :
    print(f"  {'FILE':<{_COL_WIDTH}}  {'DISTANCE':>10}  {'THRESHOLD':>10}  VERDICT")
    print(f"  {'-' * _COL_WIDTH}  {'-' * 10}  {'-' * 10}  {'-' * 9}")


def _print_row(name , distance , threshold )  :
    verdict = "ANOMALOUS" if distance > threshold else "GOOD     "
    display = name if len(name) <= _COL_WIDTH else "..." + name[-(_COL_WIDTH - 1):]
    print(f"  {display:<{_COL_WIDTH}}  {distance:>10.4f}  {threshold:>10.4f}  {verdict}")


def main(argv=None)  :
    args = parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        detector = AnomalyDetector(
            args.calibration, threshold=args.threshold, device=args.device,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    # Collect image paths
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f"\nERROR: File not found: '{image_path}'\n", file=sys.stderr)
            return 2
        image_paths = [image_path]
    else:
        images_dir = Path(args.images_dir)
        if not images_dir.is_dir():
            print(f"\nERROR: Not a directory: '{images_dir}'\n", file=sys.stderr)
            return 2
        image_paths = find_images(images_dir)
        if not image_paths:
            print(f"\nERROR: No images found in '{images_dir}'.\n", file=sys.stderr)
            return 2

    threshold = detector.threshold_value

    print()
    print("=" * 90)
    print("  INFERENCE RESULTS")
    print(f"  Threshold : p{args.threshold} = {threshold:.4f}")
    print("=" * 90)

    _print_header()

    import cv2
    n_anomalous = 0
    n_skipped   = 0

    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning("Could not read '%s' -- skipping.", path)
            n_skipped += 1
            continue
        distance = detector.score_crop(bgr)
        _print_row(path.name, distance, threshold)
        if distance > threshold:
            n_anomalous += 1

    if args.images_dir:
        n_scored = len(image_paths) - n_skipped
        print()
        print("-" * 90)
        print(f"  Total scored  : {n_scored}")
        print(f"  GOOD          : {n_scored - n_anomalous}")
        print(f"  ANOMALOUS     : {n_anomalous}")
        if n_scored:
            print(f"  Flag rate     : {n_anomalous / n_scored * 100:.1f}%")
        if n_skipped:
            print(f"  Skipped       : {n_skipped}  (load failures)")
        print("-" * 90)

    print()
    return 1 if n_anomalous > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
