"""
live.py — Real-time video anomaly detection.

Opens a webcam or video file, scores each frame using the calibration model,
and displays the result with a coloured overlay:
  - GREEN border/label → GOOD (within normal distribution)
  - RED border/label   → ANOMALOUS (Mahalanobis distance exceeds threshold)

Usage — webcam (default camera index 0)
----------------------------------------
    python live.py --calibration distribution.pkl

Usage — specific camera
-----------------------
    python live.py --calibration distribution.pkl --source 1

Usage — video file
------------------
    python live.py --calibration distribution.pkl --source ./recording.mp4

Usage — change threshold
------------------------
    python live.py --calibration distribution.pkl --threshold 95

Press 'q' or Escape to quit.

Exit codes
----------
    0  Stream ended normally.
    2  Fatal error (bad arguments, missing file, camera failure, etc.).
"""

import argparse
import collections
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from anomaly.distribution import mahalanobis_distances
from anomaly.features import build_feature_extractor, build_transform
from anomaly.io import load_calibration

try:
    import cv2
except ImportError:
    print(
        "ERROR: OpenCV is required for live video mode.\n"
        "Install it with:  pip install opencv-python\n",
        file=sys.stderr,
    )
    sys.exit(2)

logger = logging.getLogger("live")

_VALID_THRESHOLDS = {90, 95, 99}

# Colours in BGR (OpenCV convention)
_GREEN = (30, 210, 30)
_RED   = (30, 30, 220)
_WHITE = (255, 255, 255)
_BLACK = (0,   0,   0)

_BORDER_THICKNESS = 8
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FPS_SMOOTHING = 20  # frames to average for FPS display


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="live.py",
        description=(
            "Real-time anomaly detection on a webcam or video file.\n"
            "Press 'q' or Escape to quit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--calibration",
        required=True,
        metavar="FILE",
        help="Path to the calibration .pkl file produced by calibrate.py.",
    )
    parser.add_argument(
        "--source",
        default="0",
        metavar="SOURCE",
        help=(
            "Camera index (integer) or path to a video file.  "
            "Default: 0 (first webcam)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=99,
        choices=sorted(_VALID_THRESHOLDS),
        metavar="{90,95,99}",
        help="Percentile threshold for GOOD/ANOMALOUS decision (default: 99).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="PyTorch device for feature extraction (default: cpu).",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Score every N-th frame; reuse the previous result in between.  "
            "Default: 1 (every frame).  Increase to reduce CPU load."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser.parse_args(argv)


def _resolve_source(source_str: str):
    """Return an int camera index or a string file path from --source."""
    try:
        return int(source_str)
    except ValueError:
        return source_str


# ---------------------------------------------------------------------------
# Frame scoring
# ---------------------------------------------------------------------------


def score_frame(
    frame: np.ndarray,
    extractor: torch.nn.Module,
    transform,
    mean: np.ndarray,
    inv_cov: np.ndarray,
    device: torch.device,
) -> float:
    """
    Score a single OpenCV BGR frame and return its Mahalanobis distance.

    Args:
        frame:     HxWx3 uint8 BGR array from cv2.VideoCapture.
        extractor: Frozen ResNet-18 feature extractor.
        transform: ImageNet preprocessing transform.
        mean:      Calibration distribution mean (512,).
        inv_cov:   Inverse covariance matrix (512, 512).
        device:    Torch device.

    Returns:
        Scalar Mahalanobis distance (float).
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_img).unsqueeze(0).to(device)       # (1, 3, 224, 224)

    with torch.no_grad():
        out = extractor(tensor)                                # (1, 512, 1, 1)
        feat = out.squeeze(-1).squeeze(-1).cpu().numpy()       # (1, 512)

    distances = mahalanobis_distances(feat, mean, inv_cov)
    return float(distances[0])


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def _draw_overlay(
    frame: np.ndarray,
    distance: float,
    threshold: float,
    fps: float,
    is_anomalous: bool,
) -> np.ndarray:
    """
    Draw a coloured border, verdict badge, and stats banner onto *frame*.

    The frame is modified in-place.  Pass a copy if you need the original.

    Layout
    ------
    - Thick coloured rectangle border (green = GOOD, red = ANOMALOUS).
    - Semi-transparent dark banner at the top with:
        - Large verdict text on the left.
        - Distance / threshold / FPS stats below.

    Args:
        frame:        HxWx3 BGR uint8 array.
        distance:     Current Mahalanobis distance.
        threshold:    Decision threshold value.
        fps:          Smoothed frames-per-second to display.
        is_anomalous: Whether the current frame is flagged as anomalous.

    Returns:
        The annotated frame (same object as *frame*).
    """
    h, w = frame.shape[:2]
    colour = _RED if is_anomalous else _GREEN
    verdict = "ANOMALOUS" if is_anomalous else "GOOD"

    # --- Coloured border ---
    cv2.rectangle(
        frame, (0, 0), (w - 1, h - 1), colour, _BORDER_THICKNESS
    )

    # --- Semi-transparent banner at top ---
    banner_h = 84
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    # --- Verdict text (large) ---
    verdict_scale = 1.7
    verdict_thick = 3
    (_, vh), baseline = cv2.getTextSize(
        verdict, _FONT, verdict_scale, verdict_thick
    )
    cv2.putText(
        frame, verdict,
        (14, vh + 10),
        _FONT, verdict_scale, colour, verdict_thick, cv2.LINE_AA,
    )

    # --- Stats line ---
    margin = abs(distance - threshold)
    sign = "+" if distance >= threshold else "-"
    stats = (
        f"dist={distance:.4f}  thresh={threshold:.4f}  "
        f"margin={sign}{margin:.4f}  {fps:.1f} fps"
    )
    cv2.putText(
        frame, stats,
        (14, banner_h - 12),
        _FONT, 0.52, _WHITE, 1, cv2.LINE_AA,
    )

    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ------------------------------------------------------------------
    # Load calibration model
    # ------------------------------------------------------------------
    try:
        cal = load_calibration(Path(args.calibration))
    except Exception as exc:
        print(f"\nERROR: Cannot load calibration file: {exc}\n", file=sys.stderr)
        return 2

    mean = cal["mean"]
    inv_cov = cal["inv_cov"]

    threshold_key = str(args.threshold)
    if threshold_key not in cal["thresholds"]:
        print(
            f"\nERROR: Threshold key '{threshold_key}' not found in calibration file. "
            f"Available: {sorted(cal['thresholds'].keys())}\n",
            file=sys.stderr,
        )
        return 2

    threshold_value = cal["thresholds"][threshold_key]
    logger.info("Threshold p%s = %.4f", args.threshold, threshold_value)

    # ------------------------------------------------------------------
    # Build feature extractor
    # ------------------------------------------------------------------
    try:
        device = torch.device(args.device)
        torch.zeros(1).to(device)
    except Exception:
        logger.warning("Device '%s' unavailable. Falling back to CPU.", args.device)
        device = torch.device("cpu")

    transform = build_transform()
    extractor = build_feature_extractor(device)

    # ------------------------------------------------------------------
    # Open video source
    # ------------------------------------------------------------------
    source = _resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(
            f"\nERROR: Cannot open video source '{args.source}'.\n"
            "Check that the camera is connected or the file path is correct.\n",
            file=sys.stderr,
        )
        return 2

    window_title = "Anomaly Detection  [Q / Esc = quit]"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    print(
        f"\nStreaming from source={args.source!r}  "
        f"threshold=p{args.threshold}={threshold_value:.4f}  "
        f"device={device}\n"
        "Press Q or Escape to quit.\n"
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    frame_idx = 0
    last_distance = 0.0
    last_is_anomalous = False

    # Rolling FPS window
    frame_times: collections.deque = collections.deque(maxlen=_FPS_SMOOTHING)
    t_prev = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream — no more frames.")
                break

            # Centre-crop to 1024×1024
            fh, fw = frame.shape[:2]
            crop = 1024
            y0 = max((fh - crop) // 2, 0)
            x0 = max((fw - crop) // 2, 0)
            frame = frame[y0:y0 + crop, x0:x0 + crop]

            # Score every N-th frame; reuse previous verdict in between
            if frame_idx % args.every == 0:
                last_distance = score_frame(
                    frame, extractor, transform, mean, inv_cov, device
                )
                last_is_anomalous = last_distance > threshold_value
                logger.debug(
                    "frame=%d  dist=%.4f  anomalous=%s",
                    frame_idx, last_distance, last_is_anomalous,
                )

            # Smooth FPS
            t_now = time.perf_counter()
            frame_times.append(t_now - t_prev)
            t_prev = t_now
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            display = _draw_overlay(
                frame,           # draw on the frame directly (already captured)
                last_distance,
                threshold_value,
                fps,
                last_is_anomalous,
            )

            cv2.imshow(window_title, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 27 = Escape
                logger.info("Quit key pressed — stopping.")
                break

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
