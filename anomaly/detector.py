"""
detector.py — AnomalyDetector: load a calibration model and score images.

Provides the implementation used by inference.py and video_inference.py.
Can be imported directly for custom inference pipelines.

Usage
─────
  from anomaly.detector import AnomalyDetector

  detector = AnomalyDetector('distribution.pkl', threshold=99, device='cpu')

  # Score a pre-cropped 256×256 BGR image
  distance = detector.score_crop(bgr_crop)

  # Detect cap in a raw frame and score it in one call
  result = detector.score_image(bgr_frame)
  if result['status'] == 'ok':
      print(result['is_anomalous'], result['distance'])
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from anomaly.cap_detection import process_image as _detect_and_crop
from anomaly.distribution import mahalanobis_distances
from anomaly.features import build_feature_extractor, build_transform
from anomaly.io import load_calibration

logger = logging.getLogger(__name__)

_VALID_THRESHOLDS = {90, 95, 99}


class AnomalyDetector:
    """
    One-class anomaly detector for bottle cap images.

    Loads a calibration model produced by build_distribution.py and exposes
    two scoring methods:

    * ``score_crop``  — score a pre-cropped 256×256 BGR image directly.
    * ``score_image`` — detect the cap in a raw frame, crop it, then score it.
    """

    def __init__(
        self,
        calibration_path,
        threshold: int = 99,
        device: str = 'cpu',
    ):
        """
        Args:
            calibration_path: Path to the .pkl file produced by build_distribution.py.
            threshold:        Percentile key for the decision boundary (90, 95, or 99).
            device:           PyTorch device string (e.g. 'cpu', 'cuda').

        Raises:
            FileNotFoundError: If the calibration file does not exist.
            KeyError:          If required keys are missing from the calibration file.
            ValueError:        If *threshold* is not one of {90, 95, 99}.
        """
        if threshold not in _VALID_THRESHOLDS:
            raise ValueError(
                f"threshold must be one of {sorted(_VALID_THRESHOLDS)}, got {threshold}"
            )

        cal = load_calibration(Path(calibration_path))

        threshold_key = str(threshold)
        if threshold_key not in cal['thresholds']:
            raise KeyError(
                f"Threshold key '{threshold_key}' not found in calibration file. "
                f"Available: {sorted(cal['thresholds'].keys())}"
            )

        self._mean      = cal['mean']
        self._inv_cov   = cal['inv_cov']
        self._threshold = cal['thresholds'][threshold_key]

        self._pca_enabled    = cal.get('pca_enabled', False)
        self._pca_mean       = cal.get('pca_mean')
        self._pca_components = cal.get('pca_components')

        try:
            self._device = torch.device(device)
            torch.zeros(1).to(self._device)
        except (RuntimeError, AssertionError):
            logger.warning("Device '%s' unavailable — falling back to CPU.", device)
            self._device = torch.device('cpu')

        self._transform = build_transform()
        self._extractor = build_feature_extractor(self._device)

        logger.info(
            "AnomalyDetector ready — threshold p%s=%.4f  device=%s  pca=%s",
            threshold, self._threshold, self._device, self._pca_enabled,
        )

    @property
    def threshold_value(self) -> float:
        """Numeric Mahalanobis distance threshold for the chosen percentile."""
        return self._threshold

    def score_crop(self, bgr_crop: np.ndarray) -> float:
        """
        Score a pre-cropped BGR image and return its Mahalanobis distance.

        Args:
            bgr_crop: HxWx3 uint8 BGR image (typically the 256×256 masked cap crop).

        Returns:
            Scalar Mahalanobis distance.  Values above ``threshold_value``
            indicate an anomaly.
        """
        pil_img = Image.fromarray(bgr_crop[..., ::-1].copy())  # BGR → RGB
        tensor  = self._transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            out  = self._extractor(tensor)              # (1, 512, 1, 1)
            feat = out.squeeze(-1).squeeze(-1).cpu().numpy()  # (1, 512)

        if self._pca_enabled:
            feat = (feat - self._pca_mean) @ self._pca_components

        distances = mahalanobis_distances(feat, self._mean, self._inv_cov)
        return float(distances[0])

    def score_image(self, bgr_frame: np.ndarray) -> dict:
        """
        Detect the bottle cap in a raw BGR frame, crop it, and score it.

        Args:
            bgr_frame: HxWx3 uint8 BGR image at any resolution.

        Returns:
            Dict with keys:
              status       — 'ok' or 'no_cap'
              distance     — Mahalanobis distance (float), or None if no cap
              is_anomalous — bool, or None if no cap
              crop         — 256×256 BGR masked crop, or None
              centre       — (cx, cy) in full-image pixels, or None
              r_true       — outer rim radius in full-image pixels, or None
        """
        result = _detect_and_crop(bgr_frame)

        if result['status'] != 'ok':
            return {
                'status':       'no_cap',
                'distance':     None,
                'is_anomalous': None,
                'crop':         None,
                'centre':       None,
                'r_true':       None,
            }

        distance     = self.score_crop(result['crop'])
        is_anomalous = distance > self._threshold

        return {
            'status':       'ok',
            'distance':     distance,
            'is_anomalous': is_anomalous,
            'crop':         result['crop'],
            'centre':       result['centre'],
            'r_true':       result['r_true'],
        }
