"""
Multivariate Gaussian distribution fitting and Mahalanobis anomaly scoring.

This module implements three tightly related pieces of mathematics:

1. **Ledoit-Wolf shrinkage** — a regularised estimator for high-dimensional
   covariance matrices implemented entirely in NumPy (no scikit-learn).

2. **Mahalanobis distance** — the anomaly score for a feature vector against
   the fitted normal distribution.

3. **Distribution fitting** — the end-to-end calibration pipeline that
   combines the above and computes percentile thresholds.

All computations are purely NumPy; no external ML libraries are required.
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage
# ---------------------------------------------------------------------------


def ledoit_wolf_shrinkage(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute the Ledoit-Wolf analytical shrinkage estimator of the covariance matrix.

    Background
    ----------
    When the number of samples *n* is comparable to the feature dimensionality
    *p* (here n ≈ 100–500 and p = 512), the maximum-likelihood (sample)
    covariance matrix

        S = (1/n) Xᵀ X

    is an unreliable estimator of the true covariance Σ.  Its eigenvalues
    exhibit the *Marchenko-Pastur* dispersion: too-small eigenvalues are
    pushed toward zero and too-large ones are inflated.  The resulting matrix
    is often ill-conditioned or even singular, making its inverse numerically
    unreliable.

    **Ledoit-Wolf shrinkage** regularises S by blending it with a structured,
    well-conditioned *target matrix* F.  Here we use the scaled-identity target:

        F = μ I,    μ = tr(S) / p

    which replaces each eigenvalue of S with the mean eigenvalue μ.  The
    shrinkage estimator is a convex combination:

        Σ̂ = (1 − α) S + α F

    where α ∈ [0, 1] is the *shrinkage intensity* (also called the shrinkage
    coefficient or regularisation parameter):

    - α = 0 → use S unchanged (no regularisation)
    - α = 1 → use F = μ I (full regularisation)
    - 0 < α < 1 → intermediate blend

    Optimal Shrinkage Intensity
    ---------------------------
    The theoretically optimal α minimises the expected squared Frobenius
    loss E[‖Σ̂ − Σ‖²_F].  Ledoit & Wolf (2004) derived a *consistent
    plug-in estimator* that requires no knowledge of Σ:

        α* = β̄ / δ̄,   clamped to [0, 1]

    where the two quantities are:

        δ̄ = ‖S − F‖²_F = tr(S²) − tr(S)² / p
        β̄ = (1/n²) Σᵢ ‖xᵢ xᵢᵀ − S‖²_F

    Efficient Computation of β̄
    ---------------------------
    The naive O(n p²) summation over outer products is avoided using the
    identity (derived by expanding the Frobenius norm):

        Σᵢ ‖xᵢ xᵢᵀ − S‖²_F = Σᵢ ‖xᵢ‖⁴ − n · tr(S²)

    **Proof:**

        ‖xᵢ xᵢᵀ − S‖²_F
          = tr((xᵢ xᵢᵀ)²) − 2 tr(S · xᵢ xᵢᵀ) + tr(S²)
          = ‖xᵢ‖⁴ − 2 xᵢᵀ S xᵢ + tr(S²)

    Summing over i = 1…n and using
    Σᵢ xᵢᵀ S xᵢ = tr(S Σᵢ xᵢ xᵢᵀ) = tr(S · nS) = n tr(S²):

        Σᵢ ‖xᵢ xᵢᵀ − S‖²_F = Σᵢ ‖xᵢ‖⁴ − n tr(S²)  ∎

    Therefore:

        β̄ = [Σᵢ ‖xᵢ‖⁴ − n · tr(S²)] / n²

    This reduces to two O(np) operations: computing per-sample squared norms
    and summing their squares.

    Args:
        X: **Centred** sample matrix of shape ``(n, p)``.  Each row is one
           observation after subtracting the sample mean.  It is the caller's
           responsibility to centre the data before calling this function.

    Returns:
        A tuple ``(shrunk_covariance, shrinkage_alpha)`` where:

        - *shrunk_covariance*: ``np.ndarray`` of shape ``(p, p)``, symmetric
          and positive-definite.
        - *shrinkage_alpha*: float in ``[0, 1]``.  A value close to 1
          indicates heavy regularisation was needed (few samples relative to
          dimensionality).

    References:
        Ledoit, O., Wolf, M. (2004). "A well-conditioned estimator for
        large-dimensional covariance matrices." *Journal of Multivariate
        Analysis*, 88(2), 365-411.
    """
    n, p = X.shape

    # MLE sample covariance (divisor n, consistent with L-W derivation)
    S = X.T @ X / n

    trace_S = float(np.trace(S))
    trace_S2 = float(np.sum(S ** 2))  # ‖S‖²_F = tr(S²) for symmetric S

    # Target scaling: mean eigenvalue of S
    mu = trace_S / p

    # δ̄ = ‖S − μI‖²_F = tr(S²) − tr(S)²/p
    delta = trace_S2 - trace_S ** 2 / p

    if delta < 1e-12:
        # S is already proportional to identity; shrinkage has no effect
        logger.debug("Ledoit-Wolf: delta ≈ 0, no shrinkage applied.")
        return S.copy(), 0.0

    # Efficient computation of β̄
    sq_norms = np.sum(X ** 2, axis=1)          # (n,) — squared L2 norm of each row
    sum_fourth_order = float(np.sum(sq_norms ** 2))   # Σᵢ ‖xᵢ‖⁴
    beta_bar = (sum_fourth_order / n ** 2) - (trace_S2 / n)

    # Clamp to [0, 1]: negative β̄ can arise from numerical noise
    alpha = float(np.clip(beta_bar / delta, 0.0, 1.0))

    shrunk = (1.0 - alpha) * S + alpha * mu * np.eye(p, dtype=X.dtype)

    logger.debug(
        "Ledoit-Wolf: n=%d, p=%d, delta=%.4e, beta_bar=%.4e, alpha=%.4f",
        n, p, delta, beta_bar, alpha,
    )
    return shrunk, alpha


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------


def mahalanobis_distance(
    x: np.ndarray,
    mean: np.ndarray,
    inv_cov: np.ndarray,
) -> float:
    """
    Compute the Mahalanobis distance from a single vector to a Gaussian centre.

    Definition
    ----------
    Given a feature vector **x**, distribution mean **μ**, and inverse
    covariance **Σ⁻¹**, the Mahalanobis distance is:

        d(x) = √[ (x − μ)ᵀ Σ⁻¹ (x − μ) ]

    Geometric Interpretation
    ------------------------
    Mahalanobis distance is a *whitened* Euclidean distance.  The
    transformation v = Σ^{−1/2} (x − μ) maps the ellipsoidal level sets of
    the multivariate Gaussian into spherical shells, so that d(x) = ‖v‖₂.

    Compared to plain Euclidean distance ‖x − μ‖₂, the key differences are:

    * **Variance normalisation**: each feature dimension is divided by its
      standard deviation.  A feature with high natural variance contributes
      less to the distance (it is "stretched").  This prevents high-variance
      but unimportant dimensions from dominating the score.

    * **Correlation decorrelation**: correlated features are decorrelated
      before measuring distance.  A deviation that is merely a consequence
      of known feature correlations is not penalised.

    * **When Σ = I** (identity covariance), Mahalanobis distance reduces
      exactly to Euclidean distance.

    These properties make Mahalanobis distance especially suitable for
    anomaly detection in high-dimensional pretrained-feature spaces, where
    features have heterogeneous variances and strong correlations.

    Args:
        x: Feature vector of shape ``(d,)``.
        mean: Distribution mean of shape ``(d,)``.
        inv_cov: Inverse covariance matrix of shape ``(d, d)``.

    Returns:
        Non-negative float — the Mahalanobis distance.
    """
    diff = x - mean
    return float(np.sqrt(max(0.0, float(diff @ inv_cov @ diff))))


def mahalanobis_distances(
    X: np.ndarray,
    mean: np.ndarray,
    inv_cov: np.ndarray,
) -> np.ndarray:
    """
    Compute Mahalanobis distances for a batch of feature vectors.

    This is a vectorised equivalent of calling :func:`mahalanobis_distance`
    for each row of *X*, but avoids the Python-level loop.

    The computation exploits the identity:

        dᵢ² = diffᵢᵀ Σ⁻¹ diffᵢ = Σⱼ (diffᵢ Σ⁻¹)ⱼ · diffᵢⱼ

    which is computed as ``np.sum((D @ inv_cov) * D, axis=1)`` where
    ``D = X − mean``.

    Args:
        X: Feature matrix of shape ``(N, d)``.
        mean: Distribution mean of shape ``(d,)``.
        inv_cov: Inverse covariance matrix of shape ``(d, d)``.

    Returns:
        ``np.ndarray`` of shape ``(N,)`` containing non-negative distances.
    """
    diff = X - mean[np.newaxis, :]          # (N, d)
    maha_sq = np.sum((diff @ inv_cov) * diff, axis=1)   # (N,)
    return np.sqrt(np.maximum(maha_sq, 0.0))


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------


def fit_distribution(features: np.ndarray) -> Dict:
    """
    Fit a multivariate Gaussian model to a set of "good" feature vectors.

    This is the calibration step.  It performs the following operations in
    order:

    1. Compute the **sample mean** μ ∈ ℝ⁵¹².
    2. Centre the data: X_c = features − μ.
    3. Fit the **Ledoit-Wolf shrinkage covariance** Σ̂ from X_c.
    4. Compute the **inverse** Σ̂⁻¹ (well-conditioned thanks to shrinkage).
    5. Compute **Mahalanobis distances** for all calibration samples against
       (μ, Σ̂⁻¹) to characterise the range of normal distances.
    6. Compute **percentile thresholds** at the 90th, 95th, and 99th
       percentiles of the calibration distances.

    Threshold Semantics
    -------------------
    A threshold at the *k*-th percentile means that *k*% of the known-good
    calibration images fall below that threshold.  If the calibration set is
    representative, the expected false-positive rate on future good images is
    roughly (100 − k)%.

    - p90 threshold → ~10 % expected false-positive rate
    - p95 threshold → ~5 % expected false-positive rate
    - p99 threshold → ~1 % expected false-positive rate

    Args:
        features: ``np.ndarray`` of shape ``(N, 512)``.  Each row is the
                  feature vector of one known-good calibration image.

    Returns:
        A dictionary with the following keys:

        ``mean`` (np.ndarray, shape (512,))
            Sample mean of calibration features.
        ``inv_cov`` (np.ndarray, shape (512, 512))
            Inverse of the Ledoit-Wolf shrinkage covariance.
        ``thresholds`` (dict)
            Keys ``'90'``, ``'95'``, ``'99'`` mapping to float threshold values.
        ``calibration_distances`` (np.ndarray, shape (N,))
            Mahalanobis distance of each calibration image from the mean.
            Stored so the distribution can be re-examined later.
        ``shrinkage_alpha`` (float)
            The Ledoit-Wolf shrinkage intensity applied to the covariance.
        ``n_samples`` (int)
            Number of calibration images used.

    Raises:
        ValueError: If *features* is empty.
        np.linalg.LinAlgError: If the shrunk covariance matrix cannot be
            inverted (should not happen with L-W shrinkage and n ≥ 2).
    """
    n, d = features.shape

    if n == 0:
        raise ValueError("Cannot fit a distribution to an empty feature matrix.")

    if n < 10:
        logger.warning(
            "Only %d calibration image(s) provided.  The distribution estimate "
            "may be unreliable.  Aim for at least 50–100 good images for "
            "stable results.",
            n,
        )
    elif n < 50:
        logger.warning(
            "%d calibration images is on the low side for %d-dimensional "
            "features.  Results may be noisy.  Consider collecting more images.",
            n, d,
        )

    # Step 1: sample mean
    mean = features.mean(axis=0)

    # Step 2: centre
    X_centered = features - mean

    # Step 3: Ledoit-Wolf shrinkage covariance
    logger.info("Fitting Ledoit-Wolf covariance (n=%d, d=%d) …", n, d)
    shrunk_cov, alpha = ledoit_wolf_shrinkage(X_centered)
    logger.info("Shrinkage intensity α = %.4f", alpha)

    # Step 4: invert
    try:
        inv_cov = np.linalg.inv(shrunk_cov)
    except np.linalg.LinAlgError as exc:
        logger.error("Covariance inversion failed: %s", exc)
        raise

    # Symmetrise to guard against floating-point asymmetry
    inv_cov = (inv_cov + inv_cov.T) / 2.0

    # Step 5: calibration distances
    cal_distances = mahalanobis_distances(features, mean, inv_cov)

    # Step 6: percentile thresholds
    thresholds = {
        "90": float(np.percentile(cal_distances, 90)),
        "95": float(np.percentile(cal_distances, 95)),
        "99": float(np.percentile(cal_distances, 99)),
    }

    logger.info(
        "Calibration distances — mean: %.3f, std: %.3f, max: %.3f",
        float(cal_distances.mean()),
        float(cal_distances.std()),
        float(cal_distances.max()),
    )
    logger.info(
        "Thresholds — p90: %.3f, p95: %.3f, p99: %.3f",
        thresholds["90"], thresholds["95"], thresholds["99"],
    )

    return {
        "mean": mean,
        "inv_cov": inv_cov,
        "thresholds": thresholds,
        "calibration_distances": cal_distances,
        "shrinkage_alpha": alpha,
        "n_samples": n,
    }
