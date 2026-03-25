"""
preprocess.py -- Bottle cap detection, cropping, and augmentation pipeline.

Provides the implementation used by the top-level preprocess.py script.
Can also be imported directly when building custom dataset generation workflows.

Typical usage
-------------
  from anomaly.preprocess import process_image, process_dir

  # Single image -- base crop only
  results = process_image(bgr)
  if results:
      crop = results[0]['crop']   # 256x256 BGR, circle-masked

  # Single image -- base crop + 10 augmented variants
  results = process_image(bgr, n_aug=10)

  # Directory -- full dataset generation with train/val split
  metadata = process_dir('images/', 'dataset/', n_aug=12, val_frac=0.15, seed=42)
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from anomaly.augment import AugParams, augment_crop
from anomaly.cap_detection import process_image as _detect_and_crop


def process_image(
    bgr ,
    n_aug  = 0,
    rng  = None,
)  :
    """
    Detect the bottle cap in *bgr*, produce a base crop, and optionally
    generate *n_aug* augmented variants.

    Args:
        bgr:   HxWx3 uint8 BGR image (any resolution).
        n_aug: Number of augmented variants to generate in addition to the
               base crop.  Pass 0 for crop-only (no augmentation).
        rng:   NumPy random generator for reproducible augmentation.
               A fresh default_rng() is used when omitted.

    Returns:
        List of dicts.  Empty list if no cap is detected.
        Each dict contains:
          crop       -- CROP_SIZE x CROP_SIZE BGR image
          mask       -- CROP_SIZE x CROP_SIZE uint8 binary mask
          centre     -- (cx, cy) in full-image pixels
          r_true     -- outer rim radius in full-image pixels
          aug_params -- AugParams instance, or None for the base crop
    """
    result = _detect_and_crop(bgr)
    if result['status'] != 'ok':
        return []

    base = {
        'crop':       result['crop'],
        'mask':       result['mask'],
        'centre':     result['centre'],
        'r_true':     result['r_true'],
        'aug_params': None,
    }
    outputs = [base]

    if n_aug > 0:
        if rng is None:
            rng = np.random.default_rng()
        for _ in range(n_aug):
            p = AugParams(rng)
            outputs.append({
                'crop':       augment_crop(result['crop'], p),
                'mask':       result['mask'],
                'centre':     result['centre'],
                'r_true':     result['r_true'],
                'aug_params': p,
            })

    return outputs


def process_dir(
    input_dir,
    output_dir,
    n_aug  = 12,
    val_frac  = 0.15,
    seed  = None,
    ext  = 'jpg',
)  :
    """
    Process all images in *input_dir*: detect caps, crop, augment, and split
    into train/val sets.

    Output structure::

        output_dir/
            train/                  <- augmented crops for training
            val/                    <- augmented crops for validation
            metadata.json           <- seed, params, per-image split assignments
            contact_sheet_train.jpg <- thumbnail grid (visual inspection)
            contact_sheet_val.jpg

    Args:
        input_dir:  Directory containing source images.
        output_dir: Root output directory (created if absent).
        n_aug:      Augmented variants per source image (default 12).
        val_frac:   Fraction of each source's augments sent to val/ (default 0.15).
        seed:       Integer seed for reproducibility.  Random if omitted.
        ext:        File extension to glob for source images (default 'jpg').

    Returns:
        Metadata dict with keys: seed, n_aug, val_frac, failed, train, val.
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    if seed is None:
        seed = int(time.time()) % (2 ** 31)
    rng = np.random.default_rng(seed)

    sources = sorted(f for f in input_dir.iterdir()
                     if f.suffix.lstrip('.').lower() == ext.lstrip('.').lower())
    if not sources:
        print(f"No .{ext} images found in {input_dir}", file=sys.stderr)
        return {}

    for d in ('train', 'val'):
        (output_dir / d).mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 60}")
    print(f"  Preprocess")
    print(f"{'-' * 60}")
    print(f"  Source images : {len(sources)}")
    print(f"  Augs per image: {n_aug}")
    print(f"  Val fraction  : {val_frac:.0%}")
    print(f"  Seed          : {seed}")
    print(f"  Output        : {output_dir}")
    print(f"{'-' * 60}\n")

    metadata  = {
        'seed':     seed,
        'n_aug':    n_aug,
        'val_frac': val_frac,
        'failed':   [],
        'train':    [],
        'val':      [],
    }

    train_thumbs  = []
    val_thumbs    = []
    total = len(sources) * n_aug if n_aug > 0 else len(sources)

    for path in sources:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  SKIP  {path.name}  (cannot read)")
            continue

        result = _detect_and_crop(img)
        if result['status'] != 'ok':
            print(f"  FAIL  {path.name}  (no cap detected)")
            metadata['failed'].append(path.name)
            continue

        crop = result['crop']
        stem = path.stem
        cx, cy = result['centre']
        print(f"  OK    {path.name:<30}  r={result['r_true']}px  centre=({cx},{cy})")

        if n_aug == 0:
            src_idx = sources.index(path)
            is_val  = (src_idx / len(sources)) >= (1 - val_frac)
            split   = 'val' if is_val else 'train'
            fname   = f"{stem}.jpg"
            dst     = output_dir / split / fname
            cv2.imwrite(str(dst), crop)
            entry = {
                'file':      str(dst.relative_to(output_dir)),
                'source':    stem,
                'aug_index': 0,
                'params':    None,
            }
            if is_val:
                metadata['val'].append(entry)
                val_thumbs.append((stem[:16], crop))
            else:
                metadata['train'].append(entry)
                train_thumbs.append((stem[:16], crop))
        else:
            for i in range(1, n_aug + 1):
                p   = AugParams(rng)
                aug = augment_crop(crop, p)

                is_val = (i / n_aug) > (1 - val_frac)
                split  = 'val' if is_val else 'train'
                fname  = f"{stem}_aug_{i:03d}.jpg"
                dst    = output_dir / split / fname
                cv2.imwrite(str(dst), aug)

                entry = {
                    'file':      str(dst.relative_to(output_dir)),
                    'source':    stem,
                    'aug_index': i,
                    'params':    p.describe(),
                }
                if is_val:
                    metadata['val'].append(entry)
                    val_thumbs.append((f"{stem[:8]}_{i}", aug))
                else:
                    metadata['train'].append(entry)
                    train_thumbs.append((f"{stem[:8]}_{i}", aug))

                done = sources.index(path) * n_aug + i
                pct  = done / total * 100
                print(f"  [{done:4d}/{total}  {pct:5.1f}%]  {split}/{fname}", end='\r')

    print(f"\n\n  Train: {len(metadata['train'])} images")
    print(f"  Val  : {len(metadata['val'])} images")

    # Metadata
    meta_path = output_dir / 'metadata.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Contact sheets
    if train_thumbs:
        sheet = make_contact_sheet(train_thumbs, cols=12, tile=128)
        cv2.imwrite(str(output_dir / 'contact_sheet_train.jpg'), sheet)

    if val_thumbs:
        sheet = make_contact_sheet(val_thumbs, cols=12, tile=128)
        cv2.imwrite(str(output_dir / 'contact_sheet_val.jpg'), sheet)

    print(f"\n{'-' * 60}")
    print(f"  Done.")
    print(f"  Train : {len(metadata['train'])}  ->  {output_dir}/train/")
    print(f"  Val   : {len(metadata['val'])}  ->  {output_dir}/val/")
    if metadata['failed']:
        print(f"\n  Failed to detect cap in: {metadata['failed']}")
    print(f"{'-' * 60}\n")

    return metadata


def make_contact_sheet(
    images ,
    cols     = 10,
    tile     = 128,
    label_h  = 18,
)  :
    """
    Build a contact-sheet image from a list of (label, bgr_image) pairs.

    Args:
        images:  List of (label_str, np.ndarray) tuples.
        cols:    Number of columns in the grid.
        tile:    Thumbnail width/height in pixels.
        label_h: Height of the label bar above each thumbnail.

    Returns:
        Contact sheet as a BGR uint8 array.
    """
    rows  = (len(images) + cols - 1) // cols
    tw    = tile
    th    = tile + label_h
    sheet = np.full((rows * (th + 2) - 2, cols * (tw + 2) - 2, 3), 20, np.uint8)

    for i, (label, img) in enumerate(images):
        row, col = divmod(i, cols)
        y0 = row * (th + 2)
        x0 = col * (tw + 2)

        bar = np.full((label_h, tw, 3), 45, np.uint8)
        cv2.putText(bar, label[:16], (2, 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        sheet[y0:y0 + label_h, x0:x0 + tw] = bar

        thumb = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        sheet[y0 + label_h:y0 + th, x0:x0 + tw] = thumb

    return sheet
