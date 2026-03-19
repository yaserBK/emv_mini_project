# AGENTS.md

## Scope and System Boundary
- This repository currently ships one runnable component: `anomaly_detection/` (one-class anomaly scoring for already-cropped bottle-cap ROI images).
- Upstream ROI detection is explicitly out of scope; `anomaly_detection/README.md` documents YOLOv8 as a separate pipeline stage.
- Root `README.md` is minimal; treat `anomaly_detection/README.md` and code docstrings as the operational source of truth.
- Intended runtime target: NVIDIA Jetson edge hardware.

## Architecture in One Sentence

Frozen ResNet-18 → 512-dim feature vector → Mahalanobis distance against a Ledoit-Wolf–regularised multivariate Gaussian → GOOD / ANOMALOUS verdict.

## Architecture Map (Read These First)
- `anomaly_detection/calibrate.py`: calibration CLI; discovers images, extracts features, fits distribution, writes `.pkl` model.
- `anomaly_detection/infer.py`: inference CLI; loads calibration, scores one image or directory, prints table, uses exit codes 0/1/2.
- `anomaly_detection/anomaly/features.py`: frozen ResNet-18 feature extraction (`FEATURE_DIM = 512`), image transforms, recursive file discovery.
- `anomaly_detection/anomaly/distribution.py`: pure-NumPy Ledoit-Wolf shrinkage + Mahalanobis scoring + percentile threshold fitting.
- `anomaly_detection/anomaly/io.py`: pickle schema + required key validation + `_format_version` guard.
- `anomaly_detection/tests/test_distribution.py`: canonical behavior checks for math contracts and threshold semantics.

## The Math Pipeline

### 1. Feature Extraction
- ResNet-18 pretrained on ImageNet-1K, **final FC layer removed**
- Output: global average-pool → shape `(B, 512, 1, 1)` → squeezed to `(B, 512)`
- All parameters **frozen** (`requires_grad=False`), model in `eval()` mode
- Images preprocessed: resize to 224×224, ToTensor, ImageNet normalisation

### 2. Distribution Fitting (calibrate.py)
1. Compute sample mean **μ** of all calibration feature vectors
2. Centre data: `X_c = X − μ`
3. Fit **Ledoit-Wolf shrinkage covariance** `Σ̂ = (1−α)S + α·(tr(S)/d)·I`
   - α is analytically optimal (Ledoit & Wolf 2004), no hyperparameter search
   - Critical when n ≈ 100–500 samples and d = 512 features (n < d is common)
4. Invert: `Σ̂⁻¹` (symmetrised after inversion for numerical stability)
5. Compute calibration distances and percentile thresholds (p90, p95, p99)

### 3. Anomaly Scoring (infer.py)
- Mahalanobis distance: `d(x) = √[(x−μ)ᵀ Σ̂⁻¹ (x−μ)]`
- Vectorised batch form: `sqrt(sum((D @ inv_cov) * D, axis=1))` where `D = X − μ`
- Flag if `d(x) > threshold`

### 4. Threshold Semantics
| Threshold | Expected false-positive rate |
|---|---|
| p99 (default) | ~1% |
| p95 | ~5% |
| p90 | ~10% |

## Data and Control Flow
- Calibration path: image paths -> tensors -> 512D features -> `fit_distribution()` -> save model dict via `save_calibration()`.
- Inference path: load model dict -> extract features -> `mahalanobis_distances()` -> compare to `thresholds[str(percentile)]` -> verdict.
- Calibration model contract includes: `mean`, `inv_cov`, `thresholds` (`"90"`, `"95"`, `"99"`), `calibration_distances`, `n_samples`, `shrinkage_alpha`, plus optional `metadata`.

## Calibration Model (.pkl) Schema

```python
{
    "mean":                   np.ndarray (512,)      # distribution centre
    "inv_cov":                np.ndarray (512, 512)  # precision matrix
    "thresholds":             {"90": float, "95": float, "99": float}
    "calibration_distances":  np.ndarray (N,)        # per-image scores
    "shrinkage_alpha":        float                  # L-W intensity used
    "n_samples":              int
    "metadata":               dict                   # provenance
    "_format_version":        int (= 1)
}
```

## Developer Workflows
- Create calibration file:
```bash
python anomaly_detection/calibrate.py --images-dir ./good_caps --output ./calibration.pkl --device cpu --batch-size 16
```
- Run inference on one image:
```bash
python anomaly_detection/infer.py --calibration ./calibration.pkl --image ./cap.jpg --threshold 99
```
- Run inference on a directory:
```bash
python anomaly_detection/infer.py --calibration ./calibration.pkl --images-dir ./bottle_cap_data/ --threshold 99
```
- Run math-focused tests (no GPU/image assets needed):
```bash
cd anomaly_detection
python -m pytest tests/test_distribution.py -v
```

Exit codes: `0` = all GOOD, `1` = at least one ANOMALOUS, `2` = fatal error.

## Project-Specific Conventions
- Keep core statistics in NumPy (`distribution.py`); avoid adding sklearn/pandas/opencv dependencies for this path.
- Keep feature extraction frozen/inference-only: `build_feature_extractor()` removes FC layer and sets `requires_grad_(False)`.
- Preserve threshold key format as strings (`"90"`, `"95"`, `"99"`), since CLIs index thresholds using stringified percentile ints.
- `infer.py` is automation-friendly: preserve exit code semantics (0 all good, 1 anomalies, 2 fatal errors).
- Logging is intentionally separated from stdout summaries (`--log-level` defaults differ between calibrate and infer).

## Integration Notes and Safe Extension Points
- Primary external runtime deps: `torch`, `torchvision`, `numpy`, `Pillow` (see `pyproject.toml`), with Python `>=3.13`.
- No scikit-learn, opencv, pandas, or anomalib.
- Image discovery is recursive and extension-driven (`SUPPORTED_EXTENSIONS` in `features.py`); update that tuple for new formats.
- If calibration pickle schema changes, bump `_FORMAT_VERSION` in `io.py` and keep backward-compat checks explicit.
- Public package surface is controlled in `anomaly_detection/anomaly/__init__.py`; export new APIs there intentionally.
- `anomaly_detection/perception/` is currently empty; do not assume active perception-side contracts yet.
- Recommend 50–200 calibration images for initial deployment, 200–500 for production.

## Known Limitations
- **Subtle digit errors** (e.g. "2025" vs "2026") may not be caught — the 512-dim global feature averages over the whole image; localised single-character differences may not shift the feature vector enough.
- **Distribution shift** (new camera, lighting, crop parameters) causes false positives. Recalibrate whenever the imaging conditions change.
- Anomalies that look like ImageNet features may get low anomaly scores.

## Inference Results Discussion (2026-03-18, 109 images)

- **Threshold**: p99 = 10.8773
- **Flag rate**: 1.8% (2 of 109) — consistent with the expected ~1% for a p99 threshold
- **Flagged images**:
  - `IMG_4138.jpg` — distance 10.8789 (margin above threshold: **+0.0016**, borderline)
  - `IMG_4176.jpg` — distance 10.9896 (margin above threshold: **+0.1123**, more convincing)

These results were produced by scoring the **calibration images against their own calibration model**. A p99 threshold by definition flags ~1% of the calibration set, so 1–2 flags in 109 images is the statistically expected outcome, not evidence of genuine anomalies.

To validate the detector on true anomalies, score images known to be defective and verify those receive distances well above the threshold.
