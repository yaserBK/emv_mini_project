# AGENTS.md

## Scope and System Boundary

- Full pipeline: cap detection + cropping → feature extraction → Mahalanobis
  anomaly scoring, all deployable on NVIDIA Jetson edge hardware.
- Four root-level CLI scripts delegate entirely to the `anomaly/` package;
  scripts contain no implementation logic.
- `README.md` is the operational source of truth for usage; code docstrings
  are authoritative for API details.
- Intended runtime target: NVIDIA Jetson (Python 3.13, torch, torchvision,
  opencv-python, numpy, Pillow).

## Architecture in One Sentence

`preprocess.py` (Hough-circle crop + augment) → `build_distribution.py`
(frozen ResNet-18 → optional PCA → Ledoit-Wolf Gaussian) → `inference.py` /
`video_inference.py` (Mahalanobis distance → GOOD / ANOMALOUS).

## Architecture Map

### Root scripts (CLI wrappers only — no implementation)

| Script | Delegates to |
|---|---|
| `preprocess.py` | `anomaly.preprocess.process_image` / `process_dir` |
| `build_distribution.py` | `anomaly.features`, `anomaly.distribution`, `anomaly.pca`, `anomaly.io` |
| `inference.py` | `anomaly.detector.AnomalyDetector` |
| `video_inference.py` | `anomaly.detector.AnomalyDetector` + display helpers (UI concern, stays in script) |

### Package modules (`anomaly/`)

| Module | Role |
|---|---|
| `cap_detection.py` | Blob detect (Otsu sat + morphology) → Hough circle refinement → radial rim radius → 256×256 masked crop |
| `augment.py` | `AugParams` (gamma, saturation, brightness, rotation, perspective skew, noise, blur) + `augment_crop` |
| `preprocess.py` | `process_image`, `process_dir`, `make_contact_sheet` — detect-first augmentation pipeline |
| `features.py` | Frozen ResNet-18 (`FEATURE_DIM = 512`), ImageNet transforms, `find_images`, `extract_features` |
| `distribution.py` | Pure-NumPy Ledoit-Wolf shrinkage + Mahalanobis scoring + `fit_distribution` |
| `pca.py` | Pure-NumPy PCA — `fit`, `transform`, `inverse_transform`, `explained_variance_ratio` |
| `io.py` | Pickle schema, required-key validation, `_format_version` guard, `save_calibration` / `load_calibration` |
| `detector.py` | `AnomalyDetector(calibration_path, threshold, device)` — `score_crop(bgr)`, `score_image(bgr_frame)` |

## The Math Pipeline

### 1. Feature Extraction (`features.py`)
- ResNet-18 pretrained on ImageNet-1K, **final FC layer removed**
- Output: global average-pool → `(B, 512)` float32
- All parameters **frozen** (`requires_grad=False`), model in `eval()` mode
- Images preprocessed: resize 224×224, ToTensor, ImageNet normalisation

### 2. Optional PCA (`pca.py`)
- Fits on calibration features; retained at inference via the `.pkl` model
- `n_components` (exact) or `variance_threshold` (cumulative explained variance)
- `n_components` overrides `variance_threshold` when both are set
- `eigenvalues_` contains all d eigenvalues (not just retained); `components_` is `(d, k)`

### 3. Distribution Fitting (`distribution.py`, `build_distribution.py`)
1. Compute sample mean **μ**
2. Centre data: `X_c = X − μ`
3. Optionally project via PCA: `X_c = X_c @ W` where `W` is `(d, k)`
4. Fit **Ledoit-Wolf shrinkage**: `Σ̂ = (1−α)S + α·(tr(S)/d)·I`
   - α analytically optimal (Ledoit & Wolf 2004), no hyperparameter search
5. Invert: `Σ̂⁻¹` (symmetrised after inversion)
6. Compute calibration distances and percentile thresholds (p90, p95, p99)

### 4. Anomaly Scoring (`detector.py`)
- `score_crop(bgr)`: PIL BGR→RGB → transform → no_grad forward → PCA project if enabled → `mahalanobis_distances`
- `score_image(bgr_frame)`: runs `cap_detection.process_image` first, then `score_crop`
- Vectorised distance: `sqrt(sum((D @ inv_cov) * D, axis=1))` where `D = X − μ`

### 5. Threshold Semantics
| Threshold | Expected false-positive rate |
|---|---|
| p99 (default) | ~1% |
| p95 | ~5% |
| p90 | ~10% |

## Data and Control Flow

**Calibration path:**
```
image paths → BGR frames → cap detection → 256×256 crops
    → augmentation (optional) → ResNet-18 → 512D features
    → PCA (optional) → fit_distribution() → save_calibration()
```

**Inference path:**
```
BGR image → AnomalyDetector.score_crop()
    → PIL transform → ResNet-18 → PCA project → mahalanobis_distances()
    → compare to thresholds[str(percentile)] → GOOD / ANOMALOUS
```

## Calibration Model (.pkl) Schema

```python
{
    "mean":                   np.ndarray (k,)        # post-PCA distribution centre; k=512 if no PCA
    "inv_cov":                np.ndarray (k, k)      # precision matrix
    "thresholds":             {"90": float, "95": float, "99": float}
    "calibration_distances":  np.ndarray (N,)
    "shrinkage_alpha":        float
    "n_samples":              int
    "pca":                    PCA | None             # fitted PCA object, or None
    "metadata":               dict                   # provenance
    "_format_version":        int (= 1)
}
```

## Developer Workflows

```bash
# Step 1 — preprocess a directory of raw images
python preprocess.py --images-dir raw/ --out-dir dataset/ --n-aug 12 --seed 42

# Step 2 — build calibration model
python build_distribution.py --images-dir dataset/train/ --output distribution.pkl --device cpu

# Step 2 (with PCA)
python build_distribution.py --images-dir dataset/train/ --output distribution.pkl --pca-variance 0.95

# Step 3 — score images
python inference.py --calibration distribution.pkl --images-dir dataset/val/ --threshold 99

# Step 4 — live webcam
python video_inference.py --calibration distribution.pkl --source 0

# Run tests
python -m pytest anomaly/tests/ -v
```

Exit codes for `inference.py` and `video_inference.py`:
`0` = all GOOD, `1` = at least one ANOMALOUS, `2` = fatal error.

## Project-Specific Conventions

- Root scripts contain no implementation — only `argparse`, delegation calls,
  and exit codes.  All logic lives in `anomaly/`.
- Keep core statistics in pure NumPy (`distribution.py`, `pca.py`); no
  sklearn/scipy for these paths.
- Keep feature extraction frozen/inference-only: `build_feature_extractor()`
  removes the FC layer and calls `requires_grad_(False)`.
- Threshold keys are strings (`"90"`, `"95"`, `"99"`) throughout — CLIs
  index with stringified percentile ints.
- `inference.py` and `video_inference.py` are automation-friendly: preserve
  exit code semantics.
- Logging (`--log-level`) is separated from stdout summaries.
- `AnomalyDetector` is the single entry point for all scoring code; do not
  duplicate ResNet-18 loading or Mahalanobis logic in scripts.

## Integration Notes and Safe Extension Points

- Primary deps: `torch`, `torchvision`, `numpy`, `opencv-python`, `Pillow`
  (see `pyproject.toml`), Python ≥ 3.13.
- Image discovery is recursive and extension-driven (`SUPPORTED_EXTENSIONS`
  in `features.py`); update that tuple for new formats.
- If the calibration pickle schema changes, bump `_FORMAT_VERSION` in `io.py`
  and keep backward-compat checks explicit.
- Public package surface is controlled in `anomaly/__init__.py`.
- Recommended calibration set size: 50–200 images for initial deployment,
  200–500 for production.

## Known Limitations

- **Subtle digit errors** (e.g. "2025" vs "2026") may not be caught — the
  512-dim global feature averages over the whole image.
- **Distribution shift** (new camera, lighting, crop parameters) causes false
  positives. Recalibrate whenever imaging conditions change.
- **Anomalies that look like ImageNet features** may receive a low anomaly score.
