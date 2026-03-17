"""
anomaly — Date-stamp bottle-cap anomaly detection package.

This package provides the core components for one-class anomaly detection:
  - features.py   : frozen ResNet-18 feature extraction
  - distribution.py : Ledoit-Wolf covariance estimation and Mahalanobis scoring
  - io.py          : calibration model serialisation / deserialisation
"""

from anomaly.features import (
    build_feature_extractor,
    build_transform,
    extract_features,
    find_images,
)
from anomaly.distribution import (
    fit_distribution,
    ledoit_wolf_shrinkage,
    mahalanobis_distance,
    mahalanobis_distances,
)
from anomaly.io import load_calibration, save_calibration

__all__ = [
    "build_feature_extractor",
    "build_transform",
    "extract_features",
    "find_images",
    "fit_distribution",
    "ledoit_wolf_shrinkage",
    "mahalanobis_distance",
    "mahalanobis_distances",
    "load_calibration",
    "save_calibration",
]
