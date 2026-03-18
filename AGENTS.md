# AGENTS.md

## Scope and System Boundary
- This repository currently ships one runnable component: `anomaly_detection/` (one-class anomaly scoring for already-cropped bottle-cap ROI images).
- Upstream ROI detection is explicitly out of scope; `anomaly_detection/README.md` documents YOLOv8 as a separate pipeline stage.
- Root `README.md` is minimal; treat `anomaly_detection/README.md` and code docstrings as the operational source of truth.

## Architecture Map (Read These First)
- `anomaly_detection/calibrate.py`: calibration CLI; discovers images, extracts features, fits distribution, writes `.pkl` model.
- `anomaly_detection/infer.py`: inference CLI; loads calibration, scores one image or directory, prints table, uses exit codes 0/1/2.
- `anomaly_detection/anomaly/features.py`: frozen ResNet-18 feature extraction (`FEATURE_DIM = 512`), image transforms, recursive file discovery.
- `anomaly_detection/anomaly/distribution.py`: pure-NumPy Ledoit-Wolf shrinkage + Mahalanobis scoring + percentile threshold fitting.
- `anomaly_detection/anomaly/io.py`: pickle schema + required key validation + `_format_version` guard.
- `anomaly_detection/tests/test_distribution.py`: canonical behavior checks for math contracts and threshold semantics.

## Data and Control Flow
- Calibration path: image paths -> tensors -> 512D features -> `fit_distribution()` -> save model dict via `save_calibration()`.
- Inference path: load model dict -> extract features -> `mahalanobis_distances()` -> compare to `thresholds[str(percentile)]` -> verdict.
- Calibration model contract includes: `mean`, `inv_cov`, `thresholds` (`"90"`, `"95"`, `"99"`), `calibration_distances`, `n_samples`, `shrinkage_alpha`, plus optional `metadata`.

## Developer Workflows
- Create calibration file:
```bash
python anomaly_detection/calibrate.py --images-dir ./good_caps --output ./calibration.pkl --device cpu --batch-size 16
```
- Run inference on one image:
```bash
python anomaly_detection/infer.py --calibration ./calibration.pkl --image ./cap.jpg --threshold 99
```
- Run math-focused tests (no GPU/image assets needed):
```bash
cd anomaly_detection
python -m pytest tests/test_distribution.py -v
```

## Project-Specific Conventions
- Keep core statistics in NumPy (`distribution.py`); avoid adding sklearn/pandas/opencv dependencies for this path.
- Keep feature extraction frozen/inference-only: `build_feature_extractor()` removes FC layer and sets `requires_grad_(False)`.
- Preserve threshold key format as strings (`"90"`, `"95"`, `"99"`), since CLIs index thresholds using stringified percentile ints.
- `infer.py` is automation-friendly: preserve exit code semantics (0 all good, 1 anomalies, 2 fatal errors).
- Logging is intentionally separated from stdout summaries (`--log-level` defaults differ between calibrate and infer).

## Integration Notes and Safe Extension Points
- Primary external runtime deps: `torch`, `torchvision`, `numpy`, `Pillow` (see `pyproject.toml`), with Python `>=3.13`.
- Image discovery is recursive and extension-driven (`SUPPORTED_EXTENSIONS` in `features.py`); update that tuple for new formats.
- If calibration pickle schema changes, bump `_FORMAT_VERSION` in `io.py` and keep backward-compat checks explicit.
- Public package surface is controlled in `anomaly_detection/anomaly/__init__.py`; export new APIs there intentionally.
- `anomaly_detection/perception/` is currently empty; do not assume active perception-side contracts yet.

