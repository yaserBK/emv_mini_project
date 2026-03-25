"""
Stamp Annotation Tool
=====================
Click and drag to draw a bounding box over the stamp region.
Saves all annotations to stamp_annotations.json on exit.

Controls:
  Click + drag  -- draw bbox
  SPACE or D    -- confirm current bbox and move to next image
  R             -- redo current image (clear bbox)
  S             -- skip image (no stamp / can't see it)
  B             -- go back to previous image
  Q or ESC      -- quit and save progress

Usage:
  cd to your project root, then:
  uv run python annotate_stamps.py
  uv run python annotate_stamps.py --img_dir path/to/crops
"""

import cv2
import numpy as np
import json
import argparse
import glob
import os
from pathlib import Path

# -- Config ----------------------------------------------------------------
DISPLAY_SIZE  = 700      # window size (image is scaled to fit)
BBOX_COLOUR   = (0, 220, 255)
CONFIRM_COLOUR= (0, 255, 80)
ANNOT_FILE    = "stamp_annotations.json"

# -- State -----------------------------------------------------------------
drawing    = False
ix, iy     = -1, -1
fx, fy     = -1, -1
confirmed_bbox = None
current_img    = None
display_img    = None
scale          = 1.0


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, display_img, current_img, confirmed_bbox

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy  = x, y
        fx, fy  = x, y
        confirmed_bbox = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y
        # Live preview
        tmp = current_img.copy()
        cv2.rectangle(tmp, (ix, iy), (fx, fy), BBOX_COLOUR, 2)
        display_img = tmp

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy  = x, y
        x1, y1  = min(ix, fx), min(iy, fy)
        x2, y2  = max(ix, fx), max(iy, fy)
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            confirmed_bbox = (x1, y1, x2, y2)
            tmp = current_img.copy()
            cv2.rectangle(tmp, (x1, y1), (x2, y2), CONFIRM_COLOUR, 2)
            cv2.putText(tmp, "Press SPACE to confirm, R to redo",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        CONFIRM_COLOUR, 1, cv2.LINE_AA)
            display_img = tmp


def scale_bbox_to_original(bbox, scale):
    """Convert display-space bbox back to original image coordinates."""
    x1, y1, x2, y2 = bbox
    return (
        int(x1 / scale), int(y1 / scale),
        int(x2 / scale), int(y2 / scale)
    )


def load_image_for_display(path, size=DISPLAY_SIZE):
    """Load image, scale to fit display window, return (img, scale)."""
    img = cv2.imread(path)
    if img is None:
        return None, 1.0
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    new_w, new_h = int(w * s), int(h * s)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, s


def run(img_dir, annot_file):
    global current_img, display_img, confirmed_bbox, scale

    img_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )

    if not img_paths:
        print(f"No images found in {img_dir}")
        return

    # Load existing annotations
    annotations = {}
    if os.path.exists(annot_file):
        with open(annot_file) as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} existing annotations from {annot_file}")

    cv2.namedWindow("Stamp Annotator", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Stamp Annotator", mouse_cb)

    idx = 0
    # Skip already annotated images
    unannotated = [p for p in img_paths if Path(p).name not in annotations]
    print(f"Total: {len(img_paths)}  Already annotated: {len(annotations)}  "
          f"Remaining: {len(unannotated)}")

    paths_to_annotate = unannotated if unannotated else img_paths
    idx = 0

    while idx < len(paths_to_annotate):
        path  = paths_to_annotate[idx]
        fname = Path(path).name
        confirmed_bbox = None

        img_display, scale = load_image_for_display(path)
        if img_display is None:
            idx += 1
            continue

        current_img = img_display.copy()
        display_img = img_display.copy()

        # Overlay existing annotation if present
        if fname in annotations:
            ann = annotations[fname]
            if ann != "skip":
                x1,y1,x2,y2 = ann
                # Scale to display
                dx1=int(x1*scale); dy1=int(y1*scale)
                dx2=int(x2*scale); dy2=int(y2*scale)
                cv2.rectangle(current_img,(dx1,dy1),(dx2,dy2),CONFIRM_COLOUR,2)
                display_img = current_img.copy()

        # Progress bar and instructions
        prog = f"[{idx+1}/{len(paths_to_annotate)}]  {fname}"
        h, w = current_img.shape[:2]
        info = img_display.copy()
        cv2.putText(info, prog, (5, h-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(info, "Drag=draw  SPACE=confirm  R=redo  S=skip  B=back  Q=quit",
                    (5, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1, cv2.LINE_AA)
        current_img = info
        display_img = info.copy()

        while True:
            cv2.imshow("Stamp Annotator", display_img)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord(' '), ord('d')):          # confirm
                if confirmed_bbox is not None:
                    orig_bbox = scale_bbox_to_original(confirmed_bbox, scale)
                    annotations[fname] = list(orig_bbox)
                    print(f"  OK {fname}: {orig_bbox}")
                    idx += 1
                    break
                else:
                    print("  No bbox drawn -- draw one first")

            elif key == ord('r'):                    # redo
                confirmed_bbox = None
                current_img = img_display.copy()
                h2,w2 = current_img.shape[:2]
                cv2.putText(current_img, prog, (5,h2-30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(200,200,200),1,cv2.LINE_AA)
                cv2.putText(current_img, "Drag=draw  SPACE=confirm  R=redo  S=skip  B=back  Q=quit",
                            (5,h2-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(160,160,160),1,cv2.LINE_AA)
                display_img = current_img.copy()

            elif key == ord('s'):                    # skip
                annotations[fname] = "skip"
                print(f"  -- {fname}: skipped")
                idx += 1
                break

            elif key == ord('b') and idx > 0:        # back
                idx -= 1
                break

            elif key in (ord('q'), 27):              # quit
                _save(annotations, annot_file)
                cv2.destroyAllWindows()
                return

    _save(annotations, annot_file)
    cv2.destroyAllWindows()
    print(f"\nDone! {len([v for v in annotations.values() if v != 'skip'])} "
          f"bboxes saved to {annot_file}")


def _save(annotations, annot_file):
    with open(annot_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved {len(annotations)} annotations -> {annot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stamp bbox annotator")
    parser.add_argument("--img_dir",    default="output/crops",
                        help="Directory of 256x256 cap crops")
    parser.add_argument("--annot_file", default=ANNOT_FILE,
                        help="Output JSON file")
    args = parser.parse_args()
    run(args.img_dir, args.annot_file)
