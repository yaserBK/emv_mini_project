#!/usr/bin/env python3
"""
video_inference.py -- Real-time anomaly detection on a webcam or video file.

Opens a video source, detects and crops the bottle cap in each frame,
scores the crop against the calibration model, and displays the result:
  GREEN border/label  -> GOOD (within normal distribution)
  RED border/label    -> ANOMALOUS (Mahalanobis distance exceeds threshold)
  YELLOW border/label -> NO CAP detected in this frame

Press Q or Escape to quit.

Usage -- default webcam
-----------------------
    python video_inference.py --calibration distribution.pkl

Usage -- specific camera index
------------------------------
    python video_inference.py --calibration distribution.pkl --source 1

Usage -- video file
-------------------
    python video_inference.py --calibration distribution.pkl --source ./recording.mp4

Usage -- change threshold / score every N frames
-------------------------------------------------
    python video_inference.py --calibration distribution.pkl --threshold 95 --every 3

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

try:
    import cv2
except ImportError:
    print(
        "ERROR: OpenCV is required.\n"
        "Install it with:  pip install opencv-python\n",
        file=sys.stderr,
    )
    sys.exit(2)

from anomaly.detector import AnomalyDetector

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("video_inference")

_VALID_THRESHOLDS = {90, 95, 99}

# BGR colours
_GREEN  = (30, 210, 30)
_RED    = (30, 30, 220)
_YELLOW = (0, 210, 220)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)

_BORDER_THICKNESS = 8
_FONT              = cv2.FONT_HERSHEY_SIMPLEX
_FPS_SMOOTHING     = 20


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="video_inference.py",
        description=(
            "Real-time anomaly detection on a webcam or video file.\n"
            "Press Q or Escape to quit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--calibration", required=True, metavar="FILE",
        help="Calibration .pkl file produced by build_distribution.py.",
    )
    parser.add_argument(
        "--source", default="0", metavar="SOURCE",
        help="Camera index (int) or path to a video file (default: 0).",
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
        "--every", type=int, default=1, metavar="N",
        help="Score every N-th frame; reuse previous result in between (default: 1).",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser.parse_args(argv)


def _resolve_source(source_str: str):
    try:
        return int(source_str)
    except ValueError:
        return source_str


# -- Overlay rendering ---------------------------------------------------------


def _draw_overlay(
    frame: np.ndarray,
    distance: float,
    threshold: float,
    fps: float,
    is_anomalous: bool,
) -> np.ndarray:
    """Draw verdict border and stats banner onto *frame* (in-place)."""
    h, w   = frame.shape[:2]
    colour  = _RED if is_anomalous else _GREEN
    verdict = "ANOMALOUS" if is_anomalous else "GOOD"

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), colour, _BORDER_THICKNESS)

    banner_h = 84
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    verdict_scale, verdict_thick = 1.7, 3
    (_, vh), _ = cv2.getTextSize(verdict, _FONT, verdict_scale, verdict_thick)
    cv2.putText(frame, verdict, (14, vh + 10),
                _FONT, verdict_scale, colour, verdict_thick, cv2.LINE_AA)

    margin = abs(distance - threshold)
    sign   = "+" if distance >= threshold else "-"
    stats  = (f"dist={distance:.4f}  thresh={threshold:.4f}  "
              f"margin={sign}{margin:.4f}  {fps:.1f} fps")
    cv2.putText(frame, stats, (14, banner_h - 12),
                _FONT, 0.52, _WHITE, 1, cv2.LINE_AA)

    return frame


def _draw_no_cap_overlay(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw a yellow 'NO CAP' banner onto *frame* (in-place)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), _YELLOW, _BORDER_THICKNESS)

    banner_h = 84
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), _BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    verdict_scale, verdict_thick = 1.7, 3
    (_, vh), _ = cv2.getTextSize("NO CAP", _FONT, verdict_scale, verdict_thick)
    cv2.putText(frame, "NO CAP", (14, vh + 10),
                _FONT, verdict_scale, _YELLOW, verdict_thick, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} fps", (14, banner_h - 12),
                _FONT, 0.52, _WHITE, 1, cv2.LINE_AA)

    return frame


# -- Main ----------------------------------------------------------------------


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        detector = AnomalyDetector(
            args.calibration, threshold=args.threshold, device=args.device,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 2

    source = _resolve_source(args.source)
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(
            f"\nERROR: Cannot open video source '{args.source}'.\n"
            "Check the camera is connected or the file path is correct.\n",
            file=sys.stderr,
        )
        return 2

    window_title = "Anomaly Detection  [Q / Esc = quit]"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    print(
        f"\nStreaming  source={args.source!r}  "
        f"threshold=p{args.threshold}={detector.threshold_value:.4f}  "
        f"device={args.device}\n"
        "Press Q or Escape to quit.\n"
    )

    frame_idx         = 0
    last_result       = None
    frame_times: collections.deque = collections.deque(maxlen=_FPS_SMOOTHING)
    t_prev = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream.")
                break

            if frame_idx % args.every == 0:
                last_result = detector.score_image(frame)
                logger.debug(
                    "frame=%d  status=%s  dist=%s",
                    frame_idx,
                    last_result['status'],
                    f"{last_result['distance']:.4f}" if last_result['distance'] else "--",
                )

            t_now = time.perf_counter()
            frame_times.append(t_now - t_prev)
            t_prev = t_now
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            if last_result and last_result['status'] == 'ok':
                display = _draw_overlay(
                    last_result['crop'].copy(),
                    last_result['distance'],
                    detector.threshold_value,
                    fps,
                    last_result['is_anomalous'],
                )
            else:
                display = _draw_no_cap_overlay(frame, fps)

            cv2.imshow(window_title, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                logger.info("Quit key pressed.")
                break

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
