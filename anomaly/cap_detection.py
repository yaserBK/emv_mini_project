"""cap_detection.py -- Bottle cap detection and masked crop extraction.

Cascaded Detect-Localise-Inspect (DLI) pipeline:
  1. Normalise    -- conditional gamma lift + CLAHE on V channel
  2. Blob detect  -- Otsu saturation threshold + morphology + centre-proximity scoring
  3. Hough refine -- constrained Hough snaps elongated blobs to a clean circle
  4. Rim measure  -- radial saturation walk finds true outer rim radius
  5. Masked crop  -- CROP_SIZE x CROP_SIZE output with background fill

Usage::

    from anomaly.cap_detection import process_image

    result = process_image(bgr_frame)       # bgr_frame: HxWx3 uint8
    if result['status'] == 'ok':
        crop     = result['crop']           # CROP_SIZE x CROP_SIZE BGR
        centre   = result['centre']         # (cx, cy) in full-image pixels
        r_true   = result['r_true']         # outer rim radius in full-image pixels
"""

import cv2
import numpy as np
from typing import Optional, Tuple

# -- Configuration ----------------------------------------------------------

DETECT_SIZE   = 500     # px -- downsample resolution for blob detection
CROP_SIZE     = 256     # px -- output crop resolution
PADDING       = 0.08    # fraction of radius added as border around the rim

BG_COLOUR     = (0, 0, 0)   # BGR -- fill colour for masked-out areas

# Blob detection
MIN_CIRC      = 0.45    # minimum blob circularity to consider
MIN_AREA_FRAC = 0.03    # cap must cover >= 3 % of the 500 px image
MAX_AREA_FRAC = 0.55    # cap must cover <= 55 % of the 500 px image

# Rim radius measurement
N_RIM_ANGLES  = 36      # number of radial sample directions
RIM_SAT_MAX   = 85      # max saturation considered "on-cap"
RIM_VAL_MIN   = 95      # min brightness considered "on-cap"

# Hough refinement (tried in order, most -> least strict)
HOUGH_P2_VALUES = [35, 28, 22]

# Low-saturation fallback threshold
SAT_GREY_THRESHOLD = 25  # mean saturation below this -> use gradient rim finder


# -- Stage 1a: normalise ----------------------------------------------------

def normalise_fast(img: np.ndarray) -> np.ndarray:
    """
    Resize to DETECT_SIZE, apply conditional gamma lift for dark images,
    then CLAHE on the V channel.  Returns BGR at DETECT_SIZE x DETECT_SIZE.
    """
    small = cv2.resize(img, (DETECT_SIZE, DETECT_SIZE), interpolation=cv2.INTER_AREA)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.uint8)

    if hsv[..., 2].mean() < 100:
        gamma_table = np.array(
            [(i / 255.0) ** 0.4 * 255 for i in range(256)], np.uint8
        )
        hsv[..., 2] = cv2.LUT(hsv[..., 2], gamma_table)

    clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[..., 2] = clahe.apply(hsv[..., 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# -- Stage 1b: blob detection -----------------------------------------------


def blob_detect_fast(
    norm_small: np.ndarray,
    img_area_small: int,
) -> Optional[tuple]:
    """
    Find the best cap-candidate blob in a normalised 500 px image.

    Scoring: circularity x centre-proximity x log(area).
    Returns an OpenCV ellipse tuple or None.
    """
    h, w           = norm_small.shape[:2]
    cx_img, cy_img = w / 2, h / 2

    hsv = cv2.cvtColor(norm_small, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]

    otsu_val, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds  = sorted(set([max(30, int(otsu_val)), 55, 65, 75, 85]))

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    for s_thresh in thresholds:
        mask = ((sat < s_thresh) & (val > 80)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score, best = 0, None
        for c in contours:
            area = cv2.contourArea(c)
            if not (MIN_AREA_FRAC * img_area_small < area < MAX_AREA_FRAC * img_area_small):
                continue
            if len(c) < 5:
                continue

            ell              = cv2.fitEllipse(c)
            (ecx, ecy), (MA, ma), _ = ell
            circ             = min(MA, ma) / max(MA, ma)

            if circ < MIN_CIRC:
                continue

            dist  = np.hypot((ecx - cx_img) / w, (ecy - cy_img) / h)
            score = circ * max(0, 1 - 2 * dist) * np.log(area)

            if score > best_score:
                best_score, best = score, ell

        if best:
            return best

    return None


# -- Stage 1b fallback: global Hough ---------------------------------------


def hough_global(norm_small: np.ndarray, area_s: int) -> Optional[tuple]:
    """
    Global HoughCircles on grayscale -- fallback for low-saturation images where
    colour-based blob detection finds nothing (e.g. silver/metallic caps on
    similarly-coloured backgrounds).

    Returns an ellipse-compatible tuple ((cx, cy), (MA, ma), angle) in the
    same pixel space as *norm_small*, or None.
    """
    h, w    = norm_small.shape[:2]
    gray    = cv2.cvtColor(norm_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    cx_img, cy_img = w / 2, h / 2
    min_r = max(10, int(np.sqrt(MIN_AREA_FRAC * area_s / np.pi)))
    max_r = int(np.sqrt(MAX_AREA_FRAC * area_s / np.pi))

    for p2 in [40, 32, 25, 18]:
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min(h, w) * 0.3,
            param1=60, param2=p2,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            continue
        best, best_score = None, -1.0
        for cx, cy, r in circles[0]:
            dist  = np.hypot((cx - cx_img) / w, (cy - cy_img) / h)
            score = max(0.0, 1.0 - 2.0 * dist)
            if score > best_score:
                best_score, best = score, (float(cx), float(cy), float(r))
        if best is not None and best_score > 0.0:
            cx, cy, r = best
            return ((cx, cy), (r * 2.0, r * 2.0), 0.0)
    return None


# -- Stage 1: find_cap -----------------------------------------------------


def find_cap(
    img: np.ndarray,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """
    Detect the bottle cap and return its ellipse in full-image coordinates.

    1. Blob detect at 500 px (normalised, then raw fallback).
    2. Scale result back to full resolution.
    3. If circularity < 0.88, run constrained Hough to refine to a circle.

    Returns OpenCV ellipse ``((cx, cy), (MA, ma), angle)`` or ``None``.
    """
    h, w  = img.shape[:2]
    scale = h / DETECT_SIZE

    norm_s = normalise_fast(img)
    hs, ws = norm_s.shape[:2]
    area_s = hs * ws
    ell_s  = blob_detect_fast(norm_s, area_s)

    if ell_s is None:
        raw_s = cv2.resize(img, (DETECT_SIZE, DETECT_SIZE), interpolation=cv2.INTER_AREA)
        ell_s = blob_detect_fast(raw_s, area_s)

    if ell_s is None:
        ell_s = hough_global(norm_s, area_s)

    if ell_s is None:
        return None

    (ecx, ecy), (MA, ma), angle = ell_s
    ecx_f, ecy_f = ecx * scale, ecy * scale
    MA_f,  ma_f  = MA  * scale, ma  * scale
    circ = min(MA_f, ma_f) / max(MA_f, ma_f)

    if circ < 0.88:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        win  = int(max(MA_f, ma_f) * 0.75)
        x1 = max(int(ecx_f) - win, 0);  y1 = max(int(ecy_f) - win, 0)
        x2 = min(int(ecx_f) + win, w);  y2 = min(int(ecy_f) + win, h)
        roi = cv2.GaussianBlur(gray[y1:y2, x1:x2], (13, 13), 0)

        min_r = int(min(MA_f, ma_f) / 2 * 0.7)
        max_r = int(max(MA_f, ma_f) / 2 * 1.3)

        for p2 in HOUGH_P2_VALUES:
            circles = cv2.HoughCircles(
                roi, cv2.HOUGH_GRADIENT, dp=1.2,
                minDist=200, param1=70, param2=p2,
                minRadius=min_r, maxRadius=max_r,
            )
            if circles is not None:
                cx, cy, r = np.round(circles[0][0]).astype(int)
                return (
                    (float(cx + x1), float(cy + y1)),
                    (float(r * 2),   float(r * 2)),
                    0.0,
                )

    return ((ecx_f, ecy_f), (MA_f, ma_f), angle)


# -- Stage 2: rim radius ----------------------------------------------------


def find_rim_radius(
    img: np.ndarray,
    cx: float,
    cy: float,
    r_approx: int,
) -> int:
    """
    Measure the true outer cap rim radius via radial saturation sampling.

    Walks outward from the detected centre in N_RIM_ANGLES directions and
    finds the last pixel still on the cap surface (low saturation, bright).
    Discards angles blocked by adjacent bottles, returns the median.
    """
    h0, w0 = img.shape[:2]
    small  = cv2.resize(img, (DETECT_SIZE, DETECT_SIZE), interpolation=cv2.INTER_AREA)
    hs, ws = small.shape[:2]
    sx     = ws / w0

    cx_s, cy_s = cx * sx, cy * sx
    r_s        = r_approx * sx

    norm_s = normalise_fast(img)
    hsv    = cv2.cvtColor(norm_s, cv2.COLOR_BGR2HSV)
    sat    = hsv[..., 1]
    val    = hsv[..., 2]

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


# -- Stage 2b: gradient rim finder (low-saturation fallback) ---------------


def rim_radius_gradient(
    img: np.ndarray,
    cx: float,
    cy: float,
    r_approx: int,
) -> int:
    """
    Gradient-magnitude rim finder for low-saturation (silver/metallic) caps.

    For each radial ray, finds the pixel with the highest gradient magnitude
    within a tight window around r_approx.  The knurling edge produces a strong
    brightness transition regardless of colour, unlike the saturation-based
    walker which fails when cap and background are both low-saturation.
    """
    h0, w0     = img.shape[:2]
    norm_s     = normalise_fast(img)
    hs, ws     = norm_s.shape[:2]
    sx         = ws / w0
    cx_s, cy_s = cx * sx, cy * sx
    r_s        = r_approx * sx

    gray = cv2.cvtColor(norm_s, cv2.COLOR_BGR2GRAY)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)

    r_min = int(r_s * 0.75)
    r_max = int(r_s * 1.25)
    outer = []

    for a_deg in np.linspace(0, 360, N_RIM_ANGLES, endpoint=False):
        a              = np.deg2rad(a_deg)
        best_r, best_g = None, 0.0
        for rr in range(r_min, r_max):
            px = int(cx_s + rr * np.cos(a))
            py = int(cy_s + rr * np.sin(a))
            if not (0 <= px < ws and 0 <= py < hs):
                break
            g = float(grad[py, px])
            if g > best_g:
                best_g, best_r = g, rr
        if best_r is not None and best_g > 8.0:
            outer.append(best_r / sx)

    if not outer:
        return r_approx
    clean = [v for v in outer if v >= r_approx * 0.85]
    if len(clean) < 6:
        return int(np.percentile(outer, 90))
    return int(np.median(clean))


# -- Stage 3: masked crop ---------------------------------------------------


def make_masked_crop(
    img: np.ndarray,
    cx: float,
    cy: float,
    r_true: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a CROP_SIZE x CROP_SIZE BGR crop masked to the cap circle.

    Outside the circle is filled with BG_COLOUR.  If the crop region
    extends beyond the image boundary it is padded with BG_COLOUR.

    Returns:
        crop -- CROP_SIZE x CROP_SIZE BGR image
        mask -- CROP_SIZE x CROP_SIZE uint8 binary mask
    """
    h, w   = img.shape[:2]
    r_pad  = int(r_true * (1 + PADDING))

    x1 = max(int(cx) - r_pad, 0);  y1 = max(int(cy) - r_pad, 0)
    x2 = min(int(cx) + r_pad, w);  y2 = min(int(cy) + r_pad, h)
    ch, cw = y2 - y1, x2 - x1
    side   = max(ch, cw)

    sq = np.full((side, side, 3), BG_COLOUR, np.uint8)
    ox = (side - cw) // 2
    oy = (side - ch) // 2
    sq[oy:oy + ch, ox:ox + cw] = img[y1:y2, x1:x2]

    crop_n = cv2.resize(sq, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

    mq      = np.zeros((CROP_SIZE, CROP_SIZE), np.uint8)
    scale_m = CROP_SIZE / side
    cx_m    = int((int(cx) - x1 + ox) * scale_m)
    cy_m    = int((int(cy) - y1 + oy) * scale_m)
    r_m     = int(r_true * scale_m)
    cv2.circle(mq, (cx_m, cy_m), r_m, 255, -1)

    crop_n[mq == 0] = BG_COLOUR
    return crop_n, mq


# -- Public API -------------------------------------------------------------


def process_image(img: np.ndarray) -> dict:
    """
    Full pipeline: BGR frame -> masked cap crop.

    Args:
        img: HxWx3 uint8 BGR frame.

    Returns dict with keys:
        status  -- ``'ok'`` or ``'no_cap'``
        crop    -- CROP_SIZE x CROP_SIZE BGR image, or None
        mask    -- CROP_SIZE x CROP_SIZE uint8 mask, or None
        centre  -- ``(cx, cy)`` in full-image pixels, or None
        r_true  -- outer rim radius in full-image pixels, or None
        ellipse -- raw OpenCV ellipse tuple, or None
    """
    result = dict(status='no_cap', crop=None, mask=None,
                  centre=None, r_true=None, ellipse=None)

    ellipse = find_cap(img)
    if ellipse is None:
        return result

    (cx, cy), (MA, ma), _ = ellipse
    r_approx = int(max(MA, ma) / 2)

    # For low-saturation images (silver/metallic caps) the saturation-based rim
    # finder cannot distinguish cap from bottle body -- use gradient peaks instead.
    small_hsv = cv2.cvtColor(
        cv2.resize(img, (DETECT_SIZE, DETECT_SIZE), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2HSV,
    )
    if small_hsv[..., 1].mean() < SAT_GREY_THRESHOLD:
        r_true = rim_radius_gradient(img, cx, cy, r_approx)
    else:
        r_true = find_rim_radius(img, cx, cy, r_approx)

    crop, mask = make_masked_crop(img, cx, cy, r_true)

    result.update(
        status  = 'ok',
        crop    = crop,
        mask    = mask,
        centre  = (int(cx), int(cy)),
        r_true  = r_true,
        ellipse = ellipse,
    )
    return result
