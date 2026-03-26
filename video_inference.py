#!/usr/bin/env python3
"""
video_inference.py -- Real-time anomaly detection on a webcam or video file.

Opens a video source, detects the bottle cap in each frame, scores it against
the calibration model, and renders a split-panel window:

  Left panel (stabilised feed)
      The raw camera frame is cropped and zoomed so the detected cap stays
      centred regardless of where it moves in the scene.  A coloured border
      indicates the current verdict.

  Right panel (stats)
      Large verdict label (GOOD / ANOMALOUS / NO CAP), Mahalanobis distance,
      threshold, margin, and live FPS.

  Overview panel
      Full camera frame scaled to fit the panel, with the region inside the
      detected cap highlighted and the surrounding area darkened.

Colour coding (borders and labels)
-----------------------------------
  GREEN  -> GOOD       (within normal distribution)
  RED    -> ANOMALOUS  (Mahalanobis distance exceeds threshold)
  YELLOW -> NO CAP     (cap not detected in this frame)

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

_BORDER_THICKNESS = 6
_FONT              = cv2.FONT_HERSHEY_SIMPLEX
_FPS_SMOOTHING     = 20

_PANEL_SIZE  = 480   # each half of the split UI (px)
_CAP_PADDING = 0.40  # extra space around r_true in stabilised view


def parse_args(argv=None)  :
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
        "--csi", action="store_true",
        help="Use Jetson CSI camera via nvarguscamerasrc GStreamer pipeline.",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="CSI capture width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="CSI capture height in pixels (default: 720).",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="CSI capture framerate (default: 30).",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )
    return parser.parse_args(argv)


def _csi_pipeline(sensor_id=0, width=1280, height=720, fps=30):
    return (
        "nvarguscamerasrc sensor-id={} ! "
        "video/x-raw(memory:NVMM), width={}, height={}, format=NV12, framerate={}/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    ).format(sensor_id, width, height, fps)


def _resolve_source(source_str ):
    try:
        return int(source_str)
    except ValueError:
        return source_str


# -- Display helpers ----------------------------------------------------------


def _overview_panel(frame, result, panel_size):
    """Build the optional overview panel showing the full camera frame.

    The raw frame is scaled to fit within *panel_size* x *panel_size* while
    preserving its aspect ratio (letter-boxed on a black canvas).  When a cap
    is detected, the region outside the cap circle is darkened to ~30%
    brightness and the interior receives a subtle green tint, drawing the eye
    to the detection without drawing any outline.  When no cap is found the
    entire frame is dimmed to 40% brightness.  A small 'OVERVIEW' label marks
    the panel.
    """
    h, w   = frame.shape[:2]
    scale  = panel_size / max(h, w)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    p      = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Centre the scaled frame on a black canvas
    canvas = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    y0     = (panel_size - new_h) // 2
    x0     = (panel_size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = p

    if result and result['status'] == 'ok':
        cx = int(result['centre'][0] * scale) + x0
        cy = int(result['centre'][1] * scale) + y0
        r  = max(3, int(result['r_true'] * scale))

        # Binary mask: 255 inside the cap circle, 0 outside
        circ_mask = np.zeros((panel_size, panel_size), dtype=np.uint8)
        cv2.circle(circ_mask, (cx, cy), r, 255, cv2.FILLED)

        # Darken area outside the circle
        outside = circ_mask == 0
        canvas[outside] = (canvas[outside].astype(np.float32) * 0.30).astype(np.uint8)

        # Subtle green highlight inside the circle -- blend toward pure green
        inside            = circ_mask == 255
        green_layer        = np.zeros_like(canvas)
        green_layer[:, :] = (0, 80, 0)          # dark green tint (BGR)
        canvas[inside]    = cv2.addWeighted(
            canvas,  0.80,
            green_layer, 0.20, 0,
        )[inside]
    else:
        # Dim the whole frame when no cap is detected
        canvas = (canvas.astype(np.float32) * 0.40).astype(np.uint8)

    # Small label in the top-left corner
    cv2.putText(canvas, "OVERVIEW", (8, 22), _FONT, 0.48,
                (100, 100, 100), 1, cv2.LINE_AA)
    return canvas


def _stabilised_view(frame, cx, cy, r_true, panel_size):
    """Extract a square ROI from *frame* centred on the detected cap.

    Crops a square region of half-size ``r_true * (1 + _CAP_PADDING)`` around
    *(cx, cy)* so the cap remains centred and fills most of the panel as it
    moves around the scene.  Regions that fall outside the frame boundary are
    padded with black.  The result is always exactly *panel_size* x *panel_size*.
    """
    h, w = frame.shape[:2]
    half = max(1, int(r_true * (1.0 + _CAP_PADDING)))

    x1, y1 = int(cx) - half, int(cy) - half
    x2, y2 = int(cx) + half, int(cy) + half

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if roi.size == 0:
        roi = frame
    if pad_l or pad_t or pad_r or pad_b:
        roi = cv2.copyMakeBorder(
            roi, pad_t, pad_b, pad_l, pad_r,
            cv2.BORDER_CONSTANT, value=_BLACK,
        )
    return cv2.resize(roi, (panel_size, panel_size), interpolation=cv2.INTER_LINEAR)


def _stats_panel(panel_size, colour, verdict, distance, threshold, fps):
    """Build the right-side stats panel.

    Renders onto a near-black *panel_size* x *panel_size* canvas:
      - A thin coloured accent bar along the top edge.
      - A large, horizontally centred verdict label (GOOD / ANOMALOUS / NO CAP)
        auto-scaled to stay within the panel width.
      - A separator line.
      - Per-row stats: dist, thresh, margin (omitted when no cap is detected).
      - Live FPS in the bottom-right corner.
    """
    p = np.full((panel_size, panel_size, 3), 18, dtype=np.uint8)  # near-black bg

    # Coloured accent bar along the top edge
    cv2.rectangle(p, (0, 0), (panel_size, 5), colour, cv2.FILLED)

    # -- Verdict label --------------------------------------------------------
    v_scale, v_thick = 2.0, 4
    (tw, th), _ = cv2.getTextSize(verdict, _FONT, v_scale, v_thick)
    max_w = panel_size - 40
    if tw > max_w:
        v_scale = v_scale * max_w / tw
        (tw, th), _ = cv2.getTextSize(verdict, _FONT, v_scale, v_thick)
    tx = (panel_size - tw) // 2
    ty = int(panel_size * 0.30)
    cv2.putText(p, verdict, (tx, ty), _FONT, v_scale, colour, v_thick, cv2.LINE_AA)

    # -- Separator ------------------------------------------------------------
    sep_y = int(panel_size * 0.40)
    cv2.line(p, (20, sep_y), (panel_size - 20, sep_y), (60, 60, 60), 1)

    # -- Stats rows -----------------------------------------------------------
    if distance is not None:
        margin_val = abs(distance - threshold)
        sign = "+" if distance >= threshold else "-"
        rows = [
            ("dist",   f"{distance:.4f}"),
            ("thresh", f"{threshold:.4f}"),
            ("margin", f"{sign}{margin_val:.4f}"),
        ]
    else:
        rows = [("No cap detected in frame", "")]

    row_scale  = 0.55
    row_y0     = int(panel_size * 0.50)
    row_step   = int(panel_size * 0.12)
    label_x    = 28
    value_x    = int(panel_size * 0.52)

    for i, (label, value) in enumerate(rows):
        y = row_y0 + i * row_step
        cv2.putText(p, label, (label_x, y), _FONT, row_scale,
                    (160, 160, 160), 1, cv2.LINE_AA)
        if value:
            cv2.putText(p, value, (value_x, y), _FONT, row_scale,
                        _WHITE, 1, cv2.LINE_AA)

    # -- FPS (bottom-right) ---------------------------------------------------
    fps_text = f"{fps:.1f} fps"
    (fw, _), _ = cv2.getTextSize(fps_text, _FONT, 0.46, 1)
    cv2.putText(p, fps_text, (panel_size - fw - 16, panel_size - 14),
                _FONT, 0.46, (100, 100, 100), 1, cv2.LINE_AA)

    return p


def _build_display(frame, result, threshold_value, fps, panel_size=_PANEL_SIZE):
    """Compose the full display frame from individual panels.

    Always produces a 3-panel layout: overview | stabilised | stats.
    Each panel is *panel_size* x *panel_size*.  Returns a single BGR image
    suitable for ``cv2.imshow``.
    """
    if result and result['status'] == 'ok':
        colour  = _RED if result['is_anomalous'] else _GREEN
        verdict = "ANOMALOUS" if result['is_anomalous'] else "GOOD"

        left = _stabilised_view(
            frame,
            result['centre'][0], result['centre'][1],
            result['r_true'],
            panel_size,
        )
        cv2.rectangle(left, (0, 0), (panel_size - 1, panel_size - 1),
                      colour, _BORDER_THICKNESS)

        right = _stats_panel(
            panel_size, colour, verdict,
            result['distance'], threshold_value, fps,
        )
    else:
        # No cap -- show a centre-cropped raw frame on the left
        h, w  = frame.shape[:2]
        half  = min(h, w) // 2
        cx, cy = w // 2, h // 2
        roi   = frame[max(0, cy - half):min(h, cy + half),
                      max(0, cx - half):min(w, cx + half)]
        left  = cv2.resize(roi, (panel_size, panel_size),
                           interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(left, (0, 0), (panel_size - 1, panel_size - 1),
                      _YELLOW, _BORDER_THICKNESS)

        right = _stats_panel(panel_size, _YELLOW, "NO CAP",
                             None, threshold_value, fps)

    panels = [_overview_panel(frame, result, panel_size), left, right]
    return np.hstack(panels)


# -- Main ----------------------------------------------------------------------


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

    if args.csi:
        source = _csi_pipeline(
            sensor_id=int(args.source),
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
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
    cv2.resizeWindow(window_title, _PANEL_SIZE * 3, _PANEL_SIZE)

    print(
        f"\nStreaming  source={args.source!r}  "
        f"threshold=p{args.threshold}={detector.threshold_value:.4f}  "
        f"device={args.device}\n"
        "Press Q or Escape to quit.\n"
    )

    frame_idx         = 0
    last_result       = None
    frame_times  = collections.deque(maxlen=_FPS_SMOOTHING)
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

            display = _build_display(frame, last_result, detector.threshold_value, fps)
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
