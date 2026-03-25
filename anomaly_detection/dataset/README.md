# Bottle Cap Dataset Generator

Generates a training dataset of circle-masked bottle cap crops from raw source images.  
All detection and augmentation logic lives in `cap_detection_pipeline.py`.  
The CLI orchestration lives in `cap_dataset_autogen.py`.

---

## Pipeline Overview

```
source image
    │
    ▼
[Stage 1] Augment full-resolution source image  →  augmented_sources/
    │
    ▼
[Stage 2] Run cap detection pipeline on each augmented image
          normalise → blob detect → Hough refine → find rim radius → masked crop
    │
    ▼
train/  or  val/
```

This order (augment first, then detect) means the detector sees realistic imaging
variation — changed lighting, perspective tilt, noise — rather than augmenting a
pre-cropped image after the fact.

---

## Usage

```bash
# Defaults: 12 augmentations per image, 15% validation split
python cap_dataset_autogen.py --img_dir images/ --out_dir dataset/

# Custom
python cap_dataset_autogen.py --img_dir images/ --out_dir dataset/ --n_aug 20 --val_frac 0.15 --seed 42
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--img_dir` | `images` | Input directory of source images |
| `--out_dir` | `dataset` | Root output directory |
| `--n_aug` | `12` | Augmented variants to generate per source image |
| `--val_frac` | `0.15` | Fraction of each source's augments sent to `val/` |
| `--seed` | random | Integer seed for full reproducibility |
| `--crop_size` | `256` | Output crop resolution in pixels |
| `--ext` | `jpg` | File extension to glob for source images |

---

## Output Structure

```
dataset/
    augmented_sources/      ← full-resolution augmented source images
        IMG_4082_aug_001.jpg
        ...
    train/                  ← 256×256 circle-masked crops (training set)
        IMG_4082_aug_001.jpg
        ...
    val/                    ← 256×256 circle-masked crops (validation set)
        IMG_4082_aug_011.jpg
        ...
    metadata.json           ← seed, params, and split assignment per image
    contact_sheet_train.jpg ← thumbnail grid for visual inspection
    contact_sheet_val.jpg
```

---

## Cap Detection Pipeline

Each augmented source image passes through five stages:

1. **`normalise_fast`** — Downsamples to 500 px, applies a conditional gamma 0.4 lift
   if the image is dark (mean brightness < 100), then runs CLAHE contrast enhancement.

2. **`blob_detect_fast`** — Thresholds the saturation channel (Otsu + fixed fallbacks),
   applies morphological close/open, fits ellipses to contours, and scores candidates
   on circularity and proximity to the image centre.

3. **`find_cap_fast`** — Scales the detection back to full resolution. If the blob is
   too elongated (circularity < 0.88), refines with Hough circle detection on a local ROI.

4. **`find_rim_radius`** — Samples radially outward from the detected centre at 36 angles,
   using saturation and brightness thresholds to locate the true outer rim edge.

5. **`make_masked_crop`** — Crops a square region around the cap, resizes to
   `crop_size × crop_size`, and blanks all pixels outside the detected circle radius
   with a black background.

---

## Augmentation Distributions

Each augmented image is produced by independently sampling every parameter below.
Because the parameters are independent, the joint distribution is the product of
each marginal — dense coverage near the "neutral" centre with exponentially thinning
probability toward extreme combinations, reflecting how real imaging variation behaves.

### Gamma correction — Log-normal

$$\gamma = \text{clip}\!\left(e^{\,\mathcal{N}(0,\; 0.30)},\ 0.4,\ 2.5\right)$$

Applied to the V (brightness) channel in HSV space. Log-normal is used because gamma
is a *multiplicative* factor — strictly positive, and equal relative changes feel
equal perceptually. With σ = 0.30, roughly 95% of samples fall in [0.55, 1.82].

```
     ▓
    ▓▓▓
   ▓▓▓▓▓
  ▓▓▓▓▓▓▓▓
 ▓▓▓▓▓▓▓▓▓▓▓░░░
0.4  0.7  1.0  1.5  2.5
```

### Saturation scale — Log-normal

$$s = \text{clip}\!\left(e^{\,\mathcal{N}(0,\; 0.28)},\ 0.4,\ 2.5\right)$$

Same reasoning as gamma — saturation is a multiplicative scale on the S channel.
Slightly tighter spread (σ = 0.28) keeps most samples in a believable colour range.

### Brightness shift — Truncated Normal

$$b = \text{clip}\!\left(\mathcal{N}(0,\ 12),\ -40,\ 40\right)$$

Additive offset to pixel values (0–255 scale). Symmetric normal centred on zero —
equally likely to be brighter or darker, small shifts dominating, large ones rare.
The hard clip at ±40 has negligible effect since 3σ = 36.

```
          ▓▓▓
        ▓▓▓▓▓▓▓
      ▓▓▓▓▓▓▓▓▓▓▓
   ░░░▓▓▓▓▓▓▓▓▓▓▓▓░░░
  -40  -12   0   12   40
```

### Rotation — Uniform

$$\theta \sim \mathcal{U}(0°,\ 360°)$$

Bottle caps have no inherent "up" orientation, so any angle is equally valid.
Uniform over the full circle is the only unbiased choice.

```
▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
0°                  360°
```

### Perspective skew — Uniform (×2 axes)

$$s_x,\ s_y \sim \mathcal{U}(-8°,\ +8°)$$

Models a camera that is not perfectly perpendicular to the cap. Uniform reflects
no systematic bias in tilt direction. The ±8° range is physically plausible without
introducing distortion that would cause detection failures.

### Noise standard deviation — Half-normal

$$n = \left|\mathcal{N}(0,\ 6)\right|$$

Taking the absolute value of a normal distribution gives a half-normal — always
non-negative, concentrated near zero. Most images receive negligible noise; occasional
images receive a noticeable amount. Mirrors real sensor noise behaviour.

```
▓▓
▓▓▓
▓▓▓▓▓
▓▓▓▓▓▓▓▓░░
0    6    12   18
```

### Blur — Bernoulli

$$\text{blur} \sim \text{Bernoulli}(p = 0.25)$$

Applied as a binary on/off. One in four images receives a slight Gaussian blur
(kernel size 3 or 5, drawn uniformly), modelling occasional slight defocus or
motion blur from the camera.

```
▓▓▓▓▓▓▓▓▓▓▓▓        ▓▓▓▓
    no blur          blur
```

### Summary table

| Parameter | Distribution | Typical range |
|---|---|---|
| Gamma | Log-normal σ=0.30 | 0.55 – 1.82 |
| Saturation scale | Log-normal σ=0.28 | 0.60 – 1.67 |
| Brightness shift | Normal μ=0, σ=12, clipped ±40 | ±~24 pixel values |
| Rotation | Uniform [0°, 360°) | any angle |
| Skew X/Y | Uniform [−8°, +8°] | ±8° perspective tilt |
| Noise std | Half-normal σ=6 | 0 – ~18 pixel std |
| Blur | Bernoulli p=0.25 | on (kernel 3 or 5) or off |

---

## Recommended Dataset Sizes

| Use case | Source images | Augmentations per image | Total crops |
|---|---|---|---|
| Quick prototype | 20–50 | 12 | 240–600 |
| Initial deployment | 50–100 | 15 | 750–1500 |
| Production | 100–200 | 20 | 2000–4000 |

---

## Files

| File | Purpose |
|---|---|
| `cap_dataset_autogen.py` | CLI entry point — orchestrates stages 1–3 |
| `cap_detection_pipeline.py` | All detection and augmentation functions |
| `metadata.json` | Per-image record of source, split, and augmentation params |
