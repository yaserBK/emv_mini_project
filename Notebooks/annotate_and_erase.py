"""
Stamp Erase Tool — Polynomial Fill + Boundary Blend
====================================================
Stage 1 : Click 4 corners — tight quad around stamp text.
Stage 2 : Click 3–9 points — freehand polygon for the blend region.

Controls — Stage 1 (quad, 4 pts):
  Click × 4     — place corners (draggable)
  SPACE         — confirm → Stage 2
  R             — reset    S — skip    B — back    Q — quit

Controls — Stage 2 (blend polygon, 3–9 pts):
  Click         — add point (draggable)
  SPACE         — confirm (min 3 pts)
  R             — reset    B — back to quad    Q — quit

Preview:
  SPACE         — save & next    R — redo from Stage 1    Q — quit

Usage:
  uv run python annotate_and_erase.py --img_dir output/crops --out_dir output/erased
"""

import cv2
import numpy as np
import json
import argparse
import glob
import os
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
DISPLAY_SIZE  = 750
QUAD_COLOUR   = (0, 220, 255)
POLY_COLOUR   = (255, 140, 0)
HANDLE_R      = 7
INK_THRESH    = 180
POLY_DEGREE   = 2
MAX_BLEND_PTS = 9
TAPER_SIGMA   = 15
ANNOT_FILE    = "stamp_annotations.json"

# ── Shared drag state ───────────────────────────────────────────────────────
_drag_idx = None


def _nearest(pts, x, y, r=14):
    for i, (px, py) in enumerate(pts):
        if abs(px-x) < r and abs(py-y) < r:
            return i
    return None


def make_cb(pts, max_pts):
    def cb(event, x, y, flags, param):
        global _drag_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            _drag_idx = None
            idx = _nearest(pts, x, y)
            if idx is not None:
                _drag_idx = idx
            elif len(pts) < max_pts:
                pts.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and _drag_idx is not None:
            pts[_drag_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            _drag_idx = None
    return cb


def draw_poly(base, pts, colour, close=True):
    out = base.copy()
    n   = len(pts)
    if n >= 2:
        for i in range(n):
            if close or i < n-1:
                cv2.line(out, tuple(pts[i]), tuple(pts[(i+1) % n]),
                         colour, 1, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        cv2.circle(out, (px, py), HANDLE_R, colour, -1)
        cv2.circle(out, (px, py), HANDLE_R, (255,255,255), 1)
        cv2.putText(out, str(i+1), (px+9, py-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    return out


# ── Cap mask ─────────────────────────────────────────────────────────────────

def get_cap_mask(img):
    """
    Identify cap pixels by excluding the neutral grey background (180,180,180).
    The background is precisely equal on all channels; cap surface has slight
    colour variation so at least one channel will be outside 168-192.
    """
    bg = (img[:,:,0]>168)&(img[:,:,1]>168)&(img[:,:,2]>168)&\
         (img[:,:,0]<192)&(img[:,:,1]<192)&(img[:,:,2]<192)
    return ~bg


# ── Polynomial surface fill ──────────────────────────────────────────────────

def poly_features(xv, yv, degree):
    cols = []
    for d in range(degree+1):
        for i in range(d+1):
            cols.append((xv**(d-i)) * (yv**i))
    return np.stack(cols, axis=1)


def fit_and_fill(img, fill_mask, sample_mask, degree=POLY_DEGREE):
    H, W   = img.shape[:2]
    out    = img.copy().astype(np.float32)
    ys, xs = np.mgrid[0:H, 0:W]
    fy, fx = np.where(fill_mask)
    if len(fx) == 0:
        return img
    cx_f  = fx.mean();  cy_f  = fy.mean()
    scale = max(fx.max()-fx.min(), fy.max()-fy.min(), 1) / 2.0
    xn    = (xs - cx_f) / scale
    yn    = (ys - cy_f) / scale
    sy, sx = np.where(sample_mask)
    min_s  = (degree+1)*(degree+2)//2
    if len(sx) < min_s:
        return img
    A_s = poly_features(xn[sy, sx], yn[sy, sx], degree)
    A_f = poly_features(xn[fy, fx], yn[fy, fx], degree)
    for c in range(3):
        coeff, _, _, _ = np.linalg.lstsq(A_s, out[:,:,c][sy, sx], rcond=None)
        out[:,:,c][fy, fx] = np.clip(A_f @ coeff, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


# ── Boundary blend ────────────────────────────────────────────────────────────

def blend_boundary(img_orig, filled, quad_mask, blend_poly_mask,
                   bw=6, taper_sigma=TAPER_SIGMA):
    """
    Pass 1: bilateral smooth on narrow band around quad perimeter.
    Pass 2: distance taper outward from quad edge — weight=1 at boundary,
            decays to 0 at taper_sigma pixels out. Inside quad always = filled.
    """
    H, W    = img_orig.shape[:2]
    result  = filled.copy().astype(np.float32)
    orig_f  = img_orig.astype(np.float32)

    # Pass 1 — bilateral band
    k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bw*2+1, bw*2+1))
    dilated = cv2.dilate(quad_mask, k, iterations=1)
    eroded  = cv2.erode(quad_mask,  k, iterations=1)
    band    = (dilated > 0) & (eroded == 0)
    smooth  = cv2.bilateralFilter(filled, d=9, sigmaColor=25, sigmaSpace=9)
    band3   = np.stack([band.astype(np.float32)]*3, axis=-1)
    result  = result*(1-band3) + smooth.astype(np.float32)*band3

    # Pass 2 — outward taper
    # For each pixel OUTSIDE the quad, distance to nearest quad pixel
    outside_quad   = (quad_mask == 0).astype(np.uint8)
    dist_from_edge = cv2.distanceTransform(outside_quad, cv2.DIST_L2, 3)

    weight = np.exp(-(dist_from_edge**2) / (2 * taper_sigma**2))
    weight = np.clip(weight, 0, 1)

    taper_zone            = (blend_poly_mask > 0) & (quad_mask == 0)
    weight[~taper_zone]   = 0
    weight[quad_mask > 0] = 1   # inside quad: always keep filled result

    w3     = np.stack([weight]*3, axis=-1)
    result = result*w3 + orig_f*(1-w3)

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Master erase ─────────────────────────────────────────────────────────────

def erase_stamp(img, quad_pts_orig, blend_pts_orig):
    H, W      = img.shape[:2]
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap_mask  = get_cap_mask(img)

    quad_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(quad_mask,
                 [np.array(quad_pts_orig, np.int32).reshape(-1,1,2)], 255)
    on_quad = quad_mask > 0

    blend_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(blend_mask,
                 [np.array(blend_pts_orig, np.int32).reshape(-1,1,2)], 255)

    fill_mask   = on_quad & cap_mask & (gray <= INK_THRESH)
    sample_mask = (on_quad & cap_mask & (gray > INK_THRESH)) | \
                  ((blend_mask > 0) & ~on_quad & cap_mask)

    # ── Debug output ──────────────────────────────────────────────────
    print(f"    img size      : {W}x{H}")
    print(f"    cap_mask px   : {cap_mask.sum()}")
    print(f"    quad px       : {on_quad.sum()}")
    print(f"    blend px      : {(blend_mask>0).sum()}")
    print(f"    fill_mask px  : {fill_mask.sum()}  (ink inside quad)")
    print(f"    sample_mask px: {sample_mask.sum()}")
    print(f"    gray in quad  : min={gray[on_quad].min() if on_quad.sum() else 0}  "
          f"max={gray[on_quad].max() if on_quad.sum() else 0}  "
          f"mean={gray[on_quad].mean() if on_quad.sum() else 0:.0f}")
    print(f"    ink_thresh    : {INK_THRESH}")
    # ------------------------------------------------------------------

    if on_quad.sum() == 0:
        print("    !! quad is empty — coords may be outside image bounds")
        print(f"    quad_pts: {quad_pts_orig}")
        return img

    if cap_mask[on_quad].sum() == 0:
        print("    !! quad region has no cap pixels — cap_mask failing?")
        print(f"    sample BGR in quad: {img[on_quad][0]}")
        return img

    if fill_mask.sum() < 5:
        print(f"    !! fill_mask too small ({fill_mask.sum()} px) — "
              f"no ink pixels found at threshold {INK_THRESH}")
        # Diagnostic: show what threshold would catch pixels
        for t in [120, 140, 150, 160, 165, 170, 175, 180]:
            n = (on_quad & cap_mask & (gray <= t)).sum()
            print(f"       gray <= {t}: {n} px")
        return img

    filled = fit_and_fill(img, fill_mask, sample_mask)
    result = blend_boundary(img, filled, quad_mask, blend_mask)

    diff = np.abs(result.astype(int) - img.astype(int))
    print(f"    mean diff at fill: {diff[fill_mask].mean():.1f}  "
          f"max diff: {diff.max()}")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hint(img, text):
    cv2.putText(img, text, (5, img.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150,150,150), 1, cv2.LINE_AA)

def _prog(img, text):
    cv2.putText(img, text, (5, img.shape[0]-28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)

def to_orig(pts_d, s):
    return [[int(x/s), int(y/s)] for x, y in pts_d]

def load_display(path):
    img = cv2.imread(path)
    if img is None: return None, 1.0
    h, w = img.shape[:2]
    s    = min(DISPLAY_SIZE/w, DISPLAY_SIZE/h)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA), s

def _save(ann, f):
    with open(f, 'w') as fp: json.dump(ann, fp, indent=2)
    print(f"  → {len(ann)} annotations saved to {f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(img_dir, out_dir, annot_file):
    global _drag_idx

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )
    if not img_paths:
        print(f"No images in {img_dir}"); return

    annotations = {}
    if os.path.exists(annot_file):
        with open(annot_file) as f: annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations")

    remaining = [p for p in img_paths if Path(p).name not in annotations]
    if not remaining:
        print("All done — reprocessing all."); remaining = img_paths

    print(f"Total: {len(img_paths)}  Done: {len(annotations)}  "
          f"Remaining: {len(remaining)}")
    print("Stage 1: click 4 corners tightly around stamp text (draggable)")
    print(f"Stage 2: click 3–{MAX_BLEND_PTS} points for freehand blend polygon\n")

    WIN = "Stamp Erase"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    idx = 0
    while idx < len(remaining):
        path  = remaining[idx]
        fname = Path(path).name
        disp, s = load_display(path)
        if disp is None: idx += 1; continue

        print(f"\n--- {fname}  (scale={s:.3f}  "
              f"display={disp.shape[1]}x{disp.shape[0]}) ---")

        restart = True
        while restart:
            restart = False

            # ── Stage 1: quad ─────────────────────────────────────────
            quad_d = []
            _drag_idx = None
            cv2.setMouseCallback(WIN, make_cb(quad_d, 4))

            s1 = True
            while s1:
                base  = disp.copy()
                n     = len(quad_d)
                _prog(base, f"[{idx+1}/{len(remaining)}]  {fname}  — STAGE 1: 4 quad corners")
                _hint(base, ("4 pts — SPACE to confirm, drag to adjust"
                             if n == 4 else
                             "Click 4 corners  SPACE=confirm  R=reset  S=skip  B=back  Q=quit"))
                frame = draw_poly(base, quad_d, QUAD_COLOUR, close=(n==4))
                cv2.imshow(WIN, frame)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord(' '), 13):
                    if n == 4:
                        q_orig = to_orig(quad_d, s)
                        print(f"  Quad display: {quad_d}")
                        print(f"  Quad original: {q_orig}")
                        s1 = False
                    else:
                        print(f"  Need 4 pts ({n} placed)")
                elif key == ord('r'): quad_d.clear()
                elif key == ord('s'):
                    annotations[fname] = "skip"
                    _save(annotations, annot_file)
                    idx += 1; s1 = False; quad_d.clear()
                elif key == ord('b') and idx > 0:
                    idx -= 1; s1 = False; quad_d.clear()
                elif key in (ord('q'), 27):
                    _save(annotations, annot_file)
                    cv2.destroyAllWindows(); return

            if len(quad_d) != 4: break

            # ── Stage 2: blend polygon ────────────────────────────────
            blend_d = []
            _drag_idx = None
            cv2.setMouseCallback(WIN, make_cb(blend_d, MAX_BLEND_PTS))

            s2 = True
            while s2:
                base = disp.copy()
                n    = len(blend_d)
                _prog(base, (f"[{idx+1}/{len(remaining)}]  {fname}"
                             f"  — STAGE 2: blend polygon  ({n}/{MAX_BLEND_PTS})"))
                _hint(base, (f"Min 3, max {MAX_BLEND_PTS} pts  "
                             "SPACE=confirm  R=reset  B=back  Q=quit"))
                frame = draw_poly(base, quad_d, QUAD_COLOUR, close=True)
                frame = draw_poly(frame, blend_d, POLY_COLOUR, close=(n >= 3))
                cv2.imshow(WIN, frame)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord(' '), 13):
                    if n < 3:
                        print(f"  Need at least 3 pts ({n} placed)")
                    else:
                        b_orig = to_orig(blend_d, s)
                        print(f"  Blend display: {blend_d}")
                        print(f"  Blend original: {b_orig}")
                        s2 = False
                elif key == ord('r'): blend_d.clear()
                elif key == ord('b'): restart = True; s2 = False
                elif key in (ord('q'), 27):
                    _save(annotations, annot_file)
                    cv2.destroyAllWindows(); return

            if restart or len(blend_d) < 3: continue

            # ── Run erase ─────────────────────────────────────────────
            quad_orig  = to_orig(quad_d, s)
            blend_orig = to_orig(blend_d, s)
            img_orig   = cv2.imread(path)

            print(f"  Erasing {fname} ...")
            erased = erase_stamp(img_orig, quad_orig, blend_orig)

            # ── Preview ───────────────────────────────────────────────
            prev = cv2.resize(erased, (disp.shape[1], disp.shape[0]),
                              interpolation=cv2.INTER_AREA)
            cv2.polylines(prev, [np.array(quad_d, np.int32)],
                          True, QUAD_COLOUR, 1, cv2.LINE_AA)
            cv2.polylines(prev, [np.array(blend_d, np.int32)],
                          True, POLY_COLOUR, 1, cv2.LINE_AA)
            _prog(prev, f"[{idx+1}/{len(remaining)}]  {fname}  — PREVIEW")
            _hint(prev, "SPACE=save & next  R=redo from Stage 1  Q=quit")
            cv2.imshow(WIN, prev)

            while True:
                k2 = cv2.waitKey(20) & 0xFF
                if k2 in (ord(' '), 13):
                    out_path = os.path.join(out_dir, fname)
                    cv2.imwrite(out_path, erased)
                    annotations[fname] = {"quad": quad_orig, "blend": blend_orig}
                    _save(annotations, annot_file)
                    print(f"  Saved → {out_path}")
                    idx += 1; break
                elif k2 == ord('r'):
                    restart = True; break
                elif k2 in (ord('q'), 27):
                    _save(annotations, annot_file)
                    cv2.destroyAllWindows(); return

    _save(annotations, annot_file)
    cv2.destroyAllWindows()
    n = len([v for v in annotations.values() if v != "skip"])
    print(f"\nFinished. {n} images erased → {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir",    default="output/crops")
    p.add_argument("--out_dir",    default="output/erased")
    p.add_argument("--annot_file", default=ANNOT_FILE)
    args = p.parse_args()
    run(args.img_dir, args.out_dir, args.annot_file)
