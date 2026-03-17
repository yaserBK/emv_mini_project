# Date-Stamp Anomaly Detection

A lightweight, one-class anomaly detection system for bottle-cap date stamps,
designed to run on **NVIDIA Jetson** edge hardware.  The system learns what a
"good" date stamp looks like from a set of reference images, then flags any new
image that deviates from that learned distribution.

---

## Pipeline Context

This system handles **only the anomaly-scoring step**.  It assumes that the
input image has already been cropped to the bottle-cap / date-stamp region of
interest (ROI).  ROI detection (e.g. with YOLOv8) is a separate upstream
component.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Production Pipeline                              │
│                                                                         │
│  Camera → [YOLOv8 ROI detector] → [cropped cap image]                  │
│                                           │                             │
│                                           ▼                             │
│                             ┌─────────────────────────┐                │
│                             │  THIS SYSTEM            │                │
│                             │  (anomaly_detection/)   │                │
│                             │                         │                │
│                             │  1. ResNet-18 features  │                │
│                             │  2. Mahalanobis score   │                │
│                             │  3. GOOD / ANOMALOUS    │                │
│                             └────────────┬────────────┘                │
│                                          │                              │
│                                          ▼                              │
│                          [Reject / Accept / Alert]                      │
└─────────────────────────────────────────────────────────────────────────┘
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
distribution** (also called a multivariate normal):

```
p(x) = (2π)^{−d/2} |Σ|^{−1/2} exp( −½ (x − μ)ᵀ Σ⁻¹ (x − μ) )
```

where:
- **x** ∈ ℝ⁵¹² is a feature vector
- **μ** ∈ ℝ⁵¹² is the distribution mean (centre of the cluster)
- **Σ** ∈ ℝ⁵¹²ˣ⁵¹² is the covariance matrix (shape/orientation of the cluster)
- **Σ⁻¹** is the inverse covariance (precision matrix)

Geometrically, the level sets of the Gaussian are **ellipsoids** in
512-dimensional space.  Points inside the central ellipsoid are normal;
points outside are anomalous.

### 3. Mahalanobis Distance

The anomaly score is the **Mahalanobis distance** from the feature vector
**x** to the distribution centre **μ**:

```
d(x) = √[ (x − μ)ᵀ Σ⁻¹ (x − μ) ]
```

Compared to plain Euclidean distance ‖x − μ‖₂, the Mahalanobis distance
has two important properties:

| Property | Euclidean | Mahalanobis |
|---|---|---|
| Variance normalisation | No | Yes — high-variance dims contribute less |
| Correlation handling | No | Yes — correlated dims are decorrelated |
| Units | Raw feature units | "Standard deviations from centre" |

A Mahalanobis distance of, say, 5 means the point lies roughly 5 standard
deviations from the centre of the normal distribution (in the direction of
the deviation), regardless of the individual scales of the 512 features.

When Σ = I (identity matrix), Mahalanobis distance reduces exactly to
Euclidean distance.

### 4. Ledoit-Wolf Covariance Shrinkage

**Why shrinkage?**

To compute the Mahalanobis distance we need Σ⁻¹.  With d = 512 features and
only n ≈ 100–500 calibration images, the *sample* (empirical) covariance
matrix

```
S = (1/n) Σᵢ (xᵢ − μ)(xᵢ − μ)ᵀ
```

is a poor estimator of the true Σ.  When n < d, S is rank-deficient and
singular (not invertible at all).  Even when n is moderately larger than d,
the eigenvalues of S are dispersed far beyond their true values — small
eigenvalues are biased toward zero and large ones are inflated.  Inverting
such a matrix amplifies these errors catastrophically.

**What shrinkage does:**

Ledoit-Wolf shrinkage regularises S by blending it with a structured,
well-conditioned *target matrix* F.  We use the scaled-identity target
F = μ_F I, where μ_F = tr(S)/d is the mean eigenvalue of S:

```
Σ̂ = (1 − α) S + α · (tr(S)/d) · I
```

The scalar α ∈ [0, 1] is the *shrinkage intensity*:
- α → 0 : use raw S (no regularisation, good when n >> d)
- α → 1 : use scaled identity (full regularisation)
- 0 < α < 1 : weighted blend — the typical case

Geometrically, shrinkage pulls the elongated eigenvalue spectrum of S toward
its mean, replacing the flat ellipsoid of S with a rounder, better-conditioned
one.

**Optimal shrinkage intensity (analytical formula):**

Ledoit & Wolf (2004) derived a closed-form estimator of the optimal α that
requires no knowledge of the true Σ:

```
α* = β̄ / δ̄   (clamped to [0, 1])

where:
  δ̄ = ‖S − μ_F I‖²_F  =  tr(S²) − tr(S)²/d
  β̄ = (1/n²) Σᵢ ‖xᵢ xᵢᵀ − S‖²_F
     = [Σᵢ ‖xᵢ‖⁴ − n · tr(S²)] / n²   (efficient formula)
```

The identity `Σᵢ ‖xᵢ xᵢᵀ − S‖²_F = Σᵢ ‖xᵢ‖⁴ − n · tr(S²)` reduces the
computation from O(n · d²) to O(n · d) — critical for d = 512.

### 5. Threshold Selection

After calibration, we compute the Mahalanobis distance for every calibration
image and record the **90th, 95th, and 99th percentiles** of those distances.

A threshold at the *k*-th percentile means:
- *k*% of known-good calibration images fall **below** the threshold → GOOD
- (100 − k)% of known-good calibration images fall **above** the threshold → would be flagged as ANOMALOUS (false positives)

So the expected **false-positive rate** on genuinely normal images is:
- p90 threshold → ~10% false-positive rate
- p95 threshold → ~5% false-positive rate
- p99 threshold → ~1% false-positive rate (recommended default)

These are rates on the **calibration set**.  On new images from the same
distribution they are approximately correct; on images from a shifted
distribution (different lighting, camera, product line) they may be wrong —
which is a reason to recalibrate when conditions change.

---

## Project Structure

```
anomaly_detection/
├── README.md                  ← this file
├── calibrate.py               ← CLI: fit model from good images
├── infer.py                   ← CLI: score new images
├── anomaly/
│   ├── __init__.py
│   ├── features.py            ← ResNet-18 feature extraction
│   ├── distribution.py        ← Ledoit-Wolf, Mahalanobis, fit_distribution
│   └── io.py                  ← save / load calibration .pkl
└── tests/
    └── test_distribution.py   ← unit tests (pure NumPy, no ML required)
```

---

## Dependencies

Only the following libraries are required (all available on Jetson):

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | ResNet-18 backbone, image transforms |
| `numpy` | All numerical computation |
| `Pillow` | Image loading |
| Python stdlib | `argparse`, `logging`, `pickle`, `pathlib`, etc. |

No scikit-learn, opencv, pandas, anomalib, or other third-party libraries.

---

## Usage Guide

### Step 1 — Calibration

Collect 50–500 images of **normal** (good) bottle caps.  Each image should
already be cropped to the cap / date-stamp region.  Place them in a directory,
e.g. `./good_caps/`.

```bash
python calibrate.py \
    --images-dir ./good_caps/ \
    --output     ./calibration.pkl \
    --device     cuda \
    --batch-size 16
```

This will print a summary like:

```
============================================================
  CALIBRATION SUMMARY
============================================================
  Images dir       : /path/to/good_caps
  Images found     : 312
  Images processed : 312
  Feature extractor: ResNet-18 avgpool (dim=512)
  Device           : cuda
  L-W shrinkage α  : 0.4312

  Calibration Mahalanobis distances:
    mean  : 18.2341
    std   : 2.1045
    min   : 13.8920
    max   : 24.5610

  Decision thresholds (percentiles of calibration distances):
    p90 threshold : 21.0432
    p95 threshold : 22.3187
    p99 threshold : 23.9841

  Calibration model saved to: /path/to/calibration.pkl
============================================================
```

#### Calibration CLI options

| Option | Default | Description |
|---|---|---|
| `--images-dir DIR` | (required) | Directory of good images |
| `--output FILE` | (required) | Output `.pkl` path |
| `--device DEVICE` | `cpu` | `cpu`, `cuda`, `cuda:0`, etc. |
| `--batch-size N` | `16` | Images per forward pass |
| `--log-level LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

### Step 2 — Inference

#### Single image

```bash
python infer.py \
    --calibration ./calibration.pkl \
    --image       ./test_cap.jpg \
    --threshold   99
```

#### Directory of images

```bash
python infer.py \
    --calibration ./calibration.pkl \
    --images-dir  ./test_caps/ \
    --threshold   99
```

Example output:

```
==========================================================================================
  INFERENCE RESULTS
  Threshold : p99 = 23.9841
  Calibrated: 2024-11-15T09:42:17  (312 good images)
==========================================================================================
  FILE                                               DISTANCE    THRESHOLD  VERDICT
  -------------------------------------------------  ----------  ----------  ---------
  cap_001.jpg                                           17.2341     23.9841  GOOD
  cap_002.jpg                                           19.8821     23.9841  GOOD
  cap_003_smeared.jpg                                   41.3302     23.9841  ANOMALOUS
  cap_004.jpg                                           15.9123     23.9841  GOOD

------------------------------------------------------------------------------------------
  Total images scored : 4
  GOOD                : 3
  ANOMALOUS           : 1
  Flag rate           : 25.0%
------------------------------------------------------------------------------------------
```

#### Exit codes

| Code | Meaning |
|---|---|
| `0` | All images are GOOD |
| `1` | At least one ANOMALOUS image detected |
| `2` | Fatal error (bad arguments, missing calibration file, etc.) |

The exit code is useful for scripting:

```bash
python infer.py --calibration cal.pkl --image cap.jpg && echo "OK" || echo "REJECT"
```

#### Inference CLI options

| Option | Default | Description |
|---|---|---|
| `--calibration FILE` | (required) | Calibration `.pkl` file |
| `--image FILE` | — | Single image to score |
| `--images-dir DIR` | — | Directory of images to score |
| `--threshold {90,95,99}` | `99` | Percentile threshold to use |
| `--device DEVICE` | `cpu` | PyTorch device |
| `--batch-size N` | `16` | Images per forward pass |
| `--log-level LEVEL` | `WARNING` | Logging verbosity |

---

## Running the Tests

```bash
cd anomaly_detection/
python -m pytest tests/test_distribution.py -v
```

No GPU or image data is required — tests use synthetic NumPy arrays.

---

## Tuning and Practical Advice

### How many calibration images do I need?

| n | Recommendation |
|---|---|
| < 10 | Avoid — extremely unreliable; system will warn |
| 10–50 | Possible for a quick prototype; expect high false-positive rates |
| 50–200 | Good for initial deployment |
| 200–500 | Recommended for production |
| > 500 | Diminishing returns; the distribution estimate is stable |

With d = 512 features, you ideally want n > d = 512 for a fully determined
sample covariance.  Ledoit-Wolf shrinkage makes the system work even when n < d,
but more data is always better.

### False-positive rate is too high

The system is flagging too many good caps as anomalous.

1. **Lower the threshold** to p95 or p90 (counterintuitively — a *lower*
   percentile means a *higher* distance threshold, catching fewer images).
   Wait — actually the opposite: use a **higher** percentile threshold (p99)
   so that a larger fraction of the calibration set is considered normal.
   If already at p99, try a custom threshold by examining the calibration
   distances saved in the `.pkl` file.
2. **Collect more calibration images** covering more variation (different
   angles, lighting conditions, embossing depths) so the model learns a
   wider "normal" zone.
3. **Check calibration image quality** — if some calibration images are
   actually anomalous, the model learns a contaminated distribution.

### False-negative rate is too high (missing anomalies)

The system is not catching anomalous caps.

1. **Raise the threshold** to p95 or p90 (more sensitive — lower distance
   cutoff flags more images).
2. **Check whether the anomaly type is visually distinct** in the feature
   space (see Limitations below).

### When to recalibrate

Recalibrate whenever any of the following change:
- Product line (different cap geometry, ink colour, digit style)
- Camera (new sensor, focal length, or position)
- Lighting (new bulb type, intensity, or angle)
- Software pipeline upstream (different crop parameters or ROI model)

A quick sanity-check: run `infer.py` on your calibration images and verify
the flag rate is close to (100 − threshold percentile)%.

---

## Limitations

### What this system will catch

- **Global visual deviation** from the normal appearance: smeared ink,
  missing digits, completely wrong text, gross misalignment.
- **Any change** that shifts the ResNet-18 feature representation far from
  the calibration cluster.

### What this system may miss

- **Subtle single-character errors** (e.g. "2025" vs "2026") if the
  overall visual texture is similar.  The 512-dimensional features average
  over the whole image; localised digit-level differences may not produce a
  large shift in the global feature vector.
- **Anomalies that look like ImageNet features** (e.g. a sticker with a
  natural-image pattern) — ResNet-18 features will represent these
  confidently, potentially giving a low anomaly score.
- **Distribution shift** in non-anomalous ways (lighting change, new camera)
  will cause many false positives.  Recalibrate after any change.

For higher sensitivity to digit-level errors, consider augmenting this system
with an OCR-based digit-reading approach as a complementary check.

---

## References

- Ledoit, O., Wolf, M. (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices.* Journal of Multivariate Analysis,
  88(2), 365-411.
- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for
  Image Recognition.* CVPR.
- Mahalanobis, P. C. (1936). *On the generalised distance in statistics.*
  Proceedings of the National Institute of Sciences of India.
