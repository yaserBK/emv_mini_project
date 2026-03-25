"""
anomaly — Bottle-cap anomaly detection package.

Modules
───────
  cap_detection : detect bottle cap in a BGR frame, produce a masked crop
  augment       : randomised augmentation of cap crops (AugParams, augment_crop)
  preprocess    : process_image / process_dir — crop + augment pipeline
  detector      : AnomalyDetector — load calibration model and score images
  features      : frozen ResNet-18 feature extraction
  pca           : PCA dimensionality reduction (pure NumPy)
  distribution  : Ledoit-Wolf covariance estimation and Mahalanobis scoring
  io            : calibration model serialisation / deserialisation
"""

from anomaly.cap_detection import (
    find_cap,
    find_rim_radius,
    make_masked_crop,
    process_image as detect_and_crop,
)
from anomaly.augment import AugParams, augment_crop
from anomaly.detector import AnomalyDetector
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
from anomaly.pca import PCA

__all__ = [
    # cap detection / cropping
    "detect_and_crop",
    "find_cap",
    "find_rim_radius",
    "make_masked_crop",
    # augmentation
    "AugParams",
    "augment_crop",
    # inference
    "AnomalyDetector",
    # feature extraction
    "build_feature_extractor",
    "build_transform",
    "extract_features",
    "find_images",
    # distribution
    "fit_distribution",
    "ledoit_wolf_shrinkage",
    "mahalanobis_distance",
    "mahalanobis_distances",
    # io
    "load_calibration",
    "save_calibration",
    # pca
    "PCA",
]
