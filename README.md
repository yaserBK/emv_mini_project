# emv_mini_project

A repository for collaborative work on an embedded machine vision mini-project
using the Jetson Nano.

---

## Overview

A lightweight, one-class anomaly detection system for bottle-cap date stamps,
designed to run on **NVIDIA Jetson** edge hardware.  The system learns what a
"good" date stamp looks like from a set of reference images, then flags any new
image that deviates from that learned distribution.

### Full pipeline

```
Camera / images
    │
    ▼
preprocess.py         ← detect cap, crop to 256×256, optionally augment
    │
    ▼
build_distribution.py ← extract ResNet-18 features, fit Mahalanobis model
    │
    ▼
inference.py          ← score new images → GOOD / ANOMALOUS
video_inference.py    ← live scoring from webcam or video file
```

---

## Project Structure

```
emv_mini_project/
  preprocess.py            ← CLI: detect, crop, and augment cap images
  build_distribution.py    ← CLI: fit calibration model from good crops
  inference.py             ← CLI: score images for anomalies
  video_inference.py       ← CLI: live webcam / video scoring
  anomaly/
    __init__.py
    augment.py             ← AugParams, augment_crop (colour/geometry/noise/blur)
    cap_detection.py       ← Hough-circle cap detection, masking, crop
    detector.py            ← AnomalyDetector class (load model, score_crop, score_image)
    distribution.py        ← Ledoit-Wolf shrinkage, Mahalanobis distance, fit_distribution
    features.py            ← frozen ResNet-18 feature extraction, image transforms
    io.py                  ← save / load calibration .pkl
    pca.py                 ← pure-NumPy PCA (optional dimensionality reduction)
    preprocess.py          ← process_image, process_dir, make_contact_sheet
    tests/
      test_distribution.py
      test_pca.py
  research/                ← exploratory notebooks and annotation tools
  anomaly_detection/
    dataset/               ← training image dataset
```

---

## Mathematical Background

### 1. Pretrained CNN as a Feature Extractor

A convolutional neural network (CNN) trained on ImageNet has learned to
recognise an enormous variety of visual patterns: edges, textures, shapes,
and higher-level object parts.  Even though it was trained to classify
everyday photographs, the intermediate representations it builds are
powerful general-purpose descriptions of image content.

We exploit this via **transfer learning**: we use the ResNet-18 backbone
as a fixed, frozen *feature extractor*.  We remove the final classification
layer and keep everything up to the global average-pooling layer.  Given an
input image of shape 3 × 224 × 224, this produces a **512-dimensional
feature vector** that compactly describes the visual content of the image.

When all calibration images are normal date stamps, their 512-dimensional
feature vectors will cluster tightly in feature space.  Anomalous stamps
(wrong digits, smeared ink, missing text) produce feature vectors that lie
far from this cluster — which is exactly what we measure.

### 2. Multivariate Gaussian Model

We model the cluster of normal feature vectors as a **multivariate Gaussian
distribution**:

```
p(x) = (2π)^{−d/2} |Σ|^{−1/2} exp( −½ (x − μ)ᵀ Σ⁻¹ (x − μ) )
```

where:
- **x** ∈ ℝᵈ is a feature vector (d = 512, or fewer after PCA)
- **μ** ∈ ℝᵈ is the distribution mean (centre of the cluster)
- **Σ** ∈ ℝᵈˣᵈ is the covariance matrix
- **Σ⁻¹** is the inverse covariance (precision matrix)

### 3. Mahalanobis Distance

The anomaly score is the **Mahalanobis distance** from the feature vector
**x** to the distribution centre **μ**:

```
d(x) = √[ (x − μ)ᵀ Σ⁻¹ (x − μ) ]
```

| Property | Euclidean | Mahalanobis |
|---|---|---|
| Variance normalisation | No | Yes — high-variance dims contribute less |
| Correlation handling | No | Yes — correlated dims are decorrelated |
| Units | Raw feature units | "Standard deviations from centre" |

### 4. Ledoit-Wolf Covariance Shrinkage

With d = 512 features and only n ≈ 100–500 calibration images, the sample
covariance matrix is a poor estimator — rank-deficient when n < d, and
ill-conditioned otherwise.

Ledoit-Wolf shrinkage regularises by blending the sample covariance S with a
scaled-identity target:

```
Σ̂ = (1 − α) S + α · (tr(S)/d) · I
```

The scalar α ∈ [0, 1] is computed analytically (no hyperparameter search):

```
α* = β̄ / δ̄   (clamped to [0, 1])

where:
  δ̄ = tr(S²) − tr(S)²/d
  β̄ = [Σᵢ ‖xᵢ‖⁴ − n · tr(S²)] / n²
```

### 5. PCA Dimensionality Reduction (optional)

Before fitting the Gaussian, features can be projected to a lower-dimensional
subspace using PCA.  This reduces the covariance matrix size from 512×512 and
can improve conditioning when calibration data is limited.

Enable with `--pca-variance 0.95` (retain 95% of variance) or
`--pca-components N` (retain exactly N components).

### 6. Threshold Selection

After calibration, we record the **90th, 95th, and 99th percentiles** of
Mahalanobis distances on the calibration images:

| Threshold | Expected false-positive rate |
|---|---|
| p90 | ~10% |
| p95 | ~5% |
| p99 (default) | ~1% |

---

## Usage Guide

### Step 1 — Preprocess

Detect the bottle cap in each source image, crop it to 256×256, and
optionally generate augmented variants.

```bash
# Single image — base crop only
python preprocess.py --image cap.jpg --out-dir crops/

# Single image with 10 augmented variants
python preprocess.py --image cap.jpg --out-dir crops/ --n-aug 10

# Directory — full dataset with train/val split
python preprocess.py \
    --images-dir raw_images/ \
    --out-dir    dataset/ \
    --n-aug      12 \
    --val-frac   0.15 \
    --seed       42
```

#### Preprocess CLI options

| Option | Default | Description |
|---|---|---|
| `--image FILE` | — | Single source image |
| `--images-dir DIR` | — | Directory of source images |
| `--out-dir DIR` | (required) | Output directory for crops / dataset |
| `--n-aug N` | `0` | Augmented variants per image |
| `--val-frac FLOAT` | `0.15` | Fraction to put in `val/` (directory mode) |
| `--seed INT` | random | Random seed for reproducibility |
| `--ext EXT` | `jpg` | Image extension to glob (directory mode) |

---

### Step 2 — Build Distribution

Extract ResNet-18 features from the preprocessed good crops and fit the
Ledoit-Wolf Mahalanobis model.

```bash
python build_distribution.py \
    --images-dir dataset/train/ \
    --output     distribution.pkl \
    --device     cuda
```

With PCA:

```bash
python build_distribution.py \
    --images-dir dataset/train/ \
    --output     distribution.pkl \
    --pca-variance 0.95
```

#### Build distribution CLI options

| Option | Default | Description |
|---|---|---|
| `--images-dir DIR` | (required) | Directory of good cropped images |
| `--output FILE` | (required) | Output `.pkl` path |
| `--device DEVICE` | `cpu` | `cpu`, `cuda`, `cuda:0`, etc. |
| `--batch-size N` | `16` | Images per forward pass |
| `--pca-variance FLOAT` | `0.95` | Retain enough PCs to explain this fraction of variance |
| `--pca-components N` | — | Retain exactly N principal components |
| `--no-pca` | — | Disable PCA; operate in full 512-dim space |
| `--log-level LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

### Step 3 — Inference

Score one or more cropped cap images against the calibration model.

```bash
# Single image
python inference.py \
    --calibration distribution.pkl \
    --image       test_cap.jpg

# Directory of images
python inference.py \
    --calibration distribution.pkl \
    --images-dir  test_caps/ \
    --threshold   95
```

Example output:

```
==========================================================================================
  INFERENCE RESULTS
  Threshold : p99 = 10.8773
==========================================================================================
  FILE                                               DISTANCE    THRESHOLD  VERDICT
  -------------------------------------------------------  ----------  ----------  ---------
  cap_001.jpg                                            7.2341     10.8773  GOOD
  cap_002.jpg                                            9.8821     10.8773  GOOD
  cap_003_smeared.jpg                                   21.3302     10.8773  ANOMALOUS
```

#### Inference CLI options

| Option | Default | Description |
|---|---|---|
| `--calibration FILE` | (required) | Calibration `.pkl` file |
| `--image FILE` | — | Single image to score |
| `--images-dir DIR` | — | Directory of images to score |
| `--threshold {90,95,99}` | `99` | Percentile threshold |
| `--device DEVICE` | `cpu` | PyTorch device |
| `--log-level LEVEL` | `WARNING` | Logging verbosity |

#### Exit codes

| Code | Meaning |
|---|---|
| `0` | All images are GOOD |
| `1` | At least one ANOMALOUS image detected |
| `2` | Fatal error (bad arguments, missing file, etc.) |

---

### Step 4 — Video Inference

Score frames from a webcam or video file in real time.

```bash
# Webcam (device 0)
python video_inference.py --calibration distribution.pkl

# Specific camera index or video file
python video_inference.py \
    --calibration distribution.pkl \
    --source      1 \
    --threshold   95

# Score every 3rd frame (faster on slow hardware)
python video_inference.py \
    --calibration distribution.pkl \
    --every       3
```

#### Video inference CLI options

| Option | Default | Description |
|---|---|---|
| `--calibration FILE` | (required) | Calibration `.pkl` file |
| `--source SOURCE` | `0` | Camera index or video file path |
| `--threshold {90,95,99}` | `99` | Percentile threshold |
| `--device DEVICE` | `cpu` | PyTorch device |
| `--every N` | `1` | Score every N-th frame |
| `--log-level LEVEL` | `WARNING` | Logging verbosity |

---

## Running the Tests

```bash
python -m pytest anomaly/tests/ -v
```

No GPU or image data required — tests use synthetic NumPy arrays.

---

## Dependencies

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | ResNet-18 backbone, image transforms |
| `numpy` | All numerical computation |
| `opencv-python` | Image I/O, cap detection, augmentation |
| `Pillow` | Image loading for PyTorch transforms |
| Python stdlib | `argparse`, `logging`, `pickle`, `pathlib`, etc. |

---

## Tuning and Practical Advice

### How many calibration images do I need?

| n | Recommendation |
|---|---|
| < 10 | Avoid — extremely unreliable |
| 10–50 | Quick prototype only; expect high false-positive rates |
| 50–200 | Good for initial deployment |
| 200–500 | Recommended for production |
| > 500 | Diminishing returns |

With d = 512 features, you ideally want n > d for a fully determined sample
covariance.  Ledoit-Wolf shrinkage (and optionally PCA) makes the system work
even when n < d.

### False-positive rate is too high

1. Use a **higher** percentile threshold (`--threshold 99`) so more of the
   calibration set is considered normal.
2. Collect more calibration images covering more variation (angles, lighting,
   embossing depths).
3. Check calibration image quality — anomalous calibration images contaminate
   the learned distribution.

### False-negative rate is too high (missing anomalies)

1. Use a **lower** percentile threshold (`--threshold 90` or `--threshold 95`).
2. Check whether the anomaly type produces a visually distinct feature vector.

### When to recalibrate

Recalibrate whenever any of these change:
- Product line (different cap geometry, ink colour, digit style)
- Camera (new sensor, focal length, or position)
- Lighting (new bulb type, intensity, or angle)
- Crop parameters (different preprocess settings)

---

## Limitations

### What this system will catch

- **Global visual deviation** from normal appearance: smeared ink, missing
  digits, completely wrong text, gross misalignment.

### What this system may miss

- **Subtle single-character errors** (e.g. "2025" vs "2026") — the global
  512-dim feature averages over the whole image; localised digit differences
  may not shift the feature vector enough.
- **Anomalies that look like ImageNet features** may get low anomaly scores.
- **Distribution shift** in non-anomalous ways (lighting change, new camera)
  will cause false positives. Recalibrate after any change.

For higher sensitivity to digit-level errors, consider an OCR-based check as
a complementary stage.

---

## References

- Ledoit, O., Wolf, M. (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices.* Journal of Multivariate Analysis,
  88(2), 365-411.
- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for
  Image Recognition.* CVPR.
- Mahalanobis, P. C. (1936). *On the generalised distance in statistics.*
  Proceedings of the National Institute of Sciences of India.
