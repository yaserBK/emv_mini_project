"""
Serialisation and deserialisation of calibration models.

The calibration model is persisted as a Python ``pickle`` file.  In addition
to the statistical model (mean vector, inverse covariance, thresholds) the
file stores metadata that allows the calibration run to be audited and
re-examined without repeating the feature extraction step.

Saved keys
----------
``mean``                   np.ndarray (512,)        Sample mean of calibration features.
``inv_cov``                np.ndarray (512, 512)    Inverse Ledoit-Wolf shrinkage covariance.
``thresholds``             dict                     Keys '90', '95', '99' -> float.
``calibration_distances``  np.ndarray (N,)          Per-image Mahalanobis distances.
``shrinkage_alpha``        float                    L-W shrinkage intensity used.
``n_samples``              int                      Number of calibration images.
``metadata``               dict                     Provenance information (see below).

Metadata keys
-------------
``calibration_date``   str    ISO-8601 date/time of calibration run.
``images_dir``         str    Absolute path of the image directory used.
``n_images_found``     int    Total images found (including any that failed to load).
``n_images_used``      int    Images successfully processed.
``feature_extractor``  str    Name/description of the CNN backbone used.
``feature_dim``        int    Dimensionality of the feature vectors (512).
"""

import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump this version if the saved format changes incompatibly.
_FORMAT_VERSION = 1


def save_calibration(data  , path )  :
    """
    Save a calibration model dictionary to a pickle file.

    The dictionary is written with ``pickle.HIGHEST_PROTOCOL`` for efficient
    storage of large NumPy arrays.  A ``_format_version`` key is injected
    automatically so that future readers can detect schema changes.

    Args:
        data: Dictionary produced by :func:`anomaly.distribution.fit_distribution`,
              optionally augmented with a ``metadata`` sub-dictionary.
        path: Destination file path.  Parent directories must already exist.

    Raises:
        OSError: If the file cannot be written (e.g. permission denied).
    """
    payload = dict(data)
    payload["_format_version"] = _FORMAT_VERSION

    path = Path(path)
    logger.info("Saving calibration model to '%s' ...", path)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Calibration model saved (%d bytes)", path.stat().st_size)


def load_calibration(path )   :
    """
    Load a calibration model from a pickle file.

    Performs basic validation to confirm that the required keys are present
    and warns if the file was written by a different format version.

    Args:
        path: Path to the ``.pkl`` calibration file produced by
              :func:`save_calibration`.

    Returns:
        The calibration dictionary with at minimum the keys:
        ``mean``, ``inv_cov``, ``thresholds``, ``calibration_distances``,
        ``n_samples``, ``shrinkage_alpha``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If required keys are absent (file is corrupt or wrong format).
        pickle.UnpicklingError: If the file is not a valid pickle.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    logger.info("Loading calibration model from '%s' ...", path)
    with open(path, "rb") as fh:
        data = pickle.load(fh)

    # Version check
    version = data.get("_format_version", 0)
    if version != _FORMAT_VERSION:
        logger.warning(
            "Calibration file version %d does not match expected version %d. "
            "Proceeding, but results may be unpredictable.",
            version,
            _FORMAT_VERSION,
        )

    # Validate required keys
    required = {"mean", "inv_cov", "thresholds", "calibration_distances", "n_samples"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Calibration file is missing required keys: {missing}")

    logger.info(
        "Calibration loaded: n_samples=%d, thresholds=%s",
        data["n_samples"],
        data["thresholds"],
    )
    return data
