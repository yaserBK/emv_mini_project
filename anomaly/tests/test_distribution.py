"""
Unit tests for anomaly.distribution -- Ledoit-Wolf shrinkage and Mahalanobis distance.

Tests are pure NumPy; no ML frameworks or external test fixtures are required.
Run with:
    python -m pytest anomaly/tests/test_distribution.py -v
"""

import unittest

import numpy as np

from anomaly.distribution import (
    fit_distribution,
    ledoit_wolf_shrinkage,
    mahalanobis_distance,
    mahalanobis_distances,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_centred(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    """Return a centred (n, p) sample matrix."""
    X = rng.standard_normal((n, p))
    return X - X.mean(axis=0)


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage tests
# ---------------------------------------------------------------------------


class TestLedoitWolfShrinkage(unittest.TestCase):
    """Tests for ledoit_wolf_shrinkage()."""

    def test_output_shape(self):
        """Returned covariance has the right shape."""
        rng = _rng(1)
        X = _make_centred(rng, 30, 10)
        cov, alpha = ledoit_wolf_shrinkage(X)
        self.assertEqual(cov.shape, (10, 10))

    def test_alpha_in_unit_interval(self):
        """Shrinkage intensity must lie in [0, 1]."""
        rng = _rng(2)
        X = _make_centred(rng, 30, 20)
        _, alpha = ledoit_wolf_shrinkage(X)
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)

    def test_shrinkage_happens_when_n_small(self):
        """
        When n is much smaller than p, shrinkage should be substantial (alpha > 0).

        With n=20 and p=50, the sample covariance is rank-deficient and the
        L-W estimator should blend it heavily toward the scaled identity.
        """
        rng = _rng(3)
        X = _make_centred(rng, 20, 50)
        S = X.T @ X / len(X)  # raw sample covariance
        shrunk, alpha = ledoit_wolf_shrinkage(X)

        self.assertGreater(alpha, 0.0, "Expected positive shrinkage for n << p")

        # The shrunk matrix must lie strictly between S and the scaled identity:
        # shrunk = (1-alpha)*S + alpha*muI  =>  componentwise closer to muI than S is.
        p = S.shape[0]
        mu = np.trace(S) / p
        target = mu * np.eye(p)

        dist_shrunk_to_target = np.linalg.norm(shrunk - target, "fro")
        dist_sample_to_target = np.linalg.norm(S - target, "fro")

        self.assertLess(
            dist_shrunk_to_target,
            dist_sample_to_target,
            "Shrunk covariance should be closer to the identity target than S.",
        )

    def test_no_shrinkage_when_n_large(self):
        """
        When n >> p and the true covariance is NOT proportional to the identity,
        alpha should approach 0.

        Key: the data must be non-spherical (Sigma != cI) so that the shrinkage
        target F = (tr(S)/p)I differs meaningfully from S (i.e. delta > 0).
        When Sigma = cI the target IS the truth, making the L-W formula degenerate
        (both numerator and denominator -> 0 at the same rate), so alpha is
        undefined in that limit.

        Here we use Sigma = diag(1,4,9,16,25), giving delta = tr(Sigma^2) - tr(Sigma)^2/p = 10.
        With n=10 000, beta_bar ~= p(p+1)/n ~= 3*10^-3, so alpha* ~= 0.003/10 ~= 3*10^-4.
        """
        rng = _rng(4)
        n, p = 10_000, 5
        # Non-spherical: each column scaled by 1, 2, 3, 4, 5
        scales = np.arange(1.0, p + 1.0)
        X = _make_centred(rng, n, p) * scales
        _, alpha = ledoit_wolf_shrinkage(X)
        self.assertLess(
            alpha,
            0.05,
            f"Expected near-zero shrinkage for n >> p with non-spherical Sigma, got alpha={alpha:.4f}",
        )

    def test_result_is_symmetric(self):
        """The shrunk covariance must be symmetric."""
        rng = _rng(5)
        X = _make_centred(rng, 40, 15)
        shrunk, _ = ledoit_wolf_shrinkage(X)
        np.testing.assert_allclose(
            shrunk, shrunk.T, atol=1e-10, err_msg="Shrunk covariance is not symmetric"
        )

    def test_result_is_positive_definite(self):
        """The shrunk covariance must be positive definite (all eigenvalues > 0)."""
        rng = _rng(6)
        X = _make_centred(rng, 25, 30)
        shrunk, _ = ledoit_wolf_shrinkage(X)
        eigenvalues = np.linalg.eigvalsh(shrunk)
        self.assertGreater(
            eigenvalues.min(),
            0.0,
            f"Shrunk covariance has non-positive eigenvalue: {eigenvalues.min():.4e}",
        )

    def test_isotropic_input_output_still_accurate(self):
        """
        When the true covariance IS proportional to the identity (Sigma = sigma^2I),
        the target F = (tr(S)/p)I ~= sigma^2I equals the truth.  Both alpha = 0 and
        alpha = 1 produce the same result (sigma^2I), so the L-W formula may return
        any alpha -- the value is undefined/degenerate in this limit.

        What matters is that the OUTPUT covariance is still a good estimator
        of sigma^2I regardless of alpha.  We verify this by checking that the
        off-diagonal entries of the shrunk covariance are small.
        """
        rng = _rng(7)
        n, p = 200, 5
        sigma = 3.0
        X = rng.standard_normal((n, p)) * sigma
        X = X - X.mean(axis=0)
        shrunk, _ = ledoit_wolf_shrinkage(X)

        # Off-diagonal elements should be small relative to diagonal
        diag_mean = np.mean(np.diag(shrunk))
        off_diag_rms = np.sqrt(
            np.sum(shrunk ** 2 - np.diag(np.diag(shrunk)) ** 2) / (p * (p - 1))
        )
        self.assertLess(
            off_diag_rms,
            0.3 * diag_mean,
            "Off-diagonal elements are too large for isotropic input",
        )

    def test_shrinkage_blends_sample_and_identity(self):
        """
        The shrunk covariance must equal (1-alpha)*S + alpha*mu*I exactly.
        """
        rng = _rng(8)
        X = _make_centred(rng, 30, 20)
        p = X.shape[1]
        S = X.T @ X / len(X)
        shrunk, alpha = ledoit_wolf_shrinkage(X)
        mu = np.trace(S) / p
        expected = (1 - alpha) * S + alpha * mu * np.eye(p)
        np.testing.assert_allclose(
            shrunk, expected, atol=1e-10,
            err_msg="Shrunk covariance does not match (1-alpha)S + alpha*mu*I"
        )


# ---------------------------------------------------------------------------
# Mahalanobis distance tests
# ---------------------------------------------------------------------------


class TestMahalanobisDistance(unittest.TestCase):
    """Tests for mahalanobis_distance() and mahalanobis_distances()."""

    def test_mean_to_itself_is_zero(self):
        """The Mahalanobis distance from the mean to itself is exactly 0."""
        rng = _rng(10)
        d = 16
        mean = rng.standard_normal(d)
        inv_cov = np.eye(d)
        dist = mahalanobis_distance(mean, mean, inv_cov)
        self.assertAlmostEqual(dist, 0.0, places=10)

    def test_identity_covariance_equals_euclidean(self):
        """
        When Sigma^-1 = I, Mahalanobis distance must equal Euclidean distance.

        This follows directly from the definition:
            d_M(x) = sqrt[(x-mu)^T I (x-mu)] = sqrt[||x-mu||^2] = ||x-mu||_2
        """
        rng = _rng(11)
        d = 32
        x = rng.standard_normal(d)
        mean = rng.standard_normal(d)
        inv_cov = np.eye(d)

        maha = mahalanobis_distance(x, mean, inv_cov)
        euclidean = float(np.linalg.norm(x - mean))

        self.assertAlmostEqual(maha, euclidean, places=8)

    def test_distance_is_non_negative(self):
        """Mahalanobis distance is always >= 0."""
        rng = _rng(12)
        d = 10
        mean = rng.standard_normal(d)
        # Random positive-definite inv_cov via L^T L
        L = rng.standard_normal((d, d))
        inv_cov = L.T @ L + np.eye(d)

        for _ in range(20):
            x = rng.standard_normal(d)
            dist = mahalanobis_distance(x, mean, inv_cov)
            self.assertGreaterEqual(dist, 0.0)

    def test_symmetry_around_mean(self):
        """
        Points equidistant from the mean on opposite sides have the same distance.

        Specifically, d_M(mu + v) == d_M(mu - v) because the Mahalanobis distance
        depends only on ||v||_{Sigma^-1}, not the direction of v.
        """
        rng = _rng(13)
        d = 8
        mean = rng.standard_normal(d)
        L = rng.standard_normal((d, d))
        inv_cov = L.T @ L + np.eye(d)
        v = rng.standard_normal(d)

        d_plus = mahalanobis_distance(mean + v, mean, inv_cov)
        d_minus = mahalanobis_distance(mean - v, mean, inv_cov)
        self.assertAlmostEqual(d_plus, d_minus, places=8)

    def test_batch_equals_individual(self):
        """
        mahalanobis_distances() on a batch must match individual calls to
        mahalanobis_distance() for every element.
        """
        rng = _rng(14)
        d, n = 16, 50
        mean = rng.standard_normal(d)
        L = rng.standard_normal((d, d))
        inv_cov = L.T @ L + np.eye(d)
        X = rng.standard_normal((n, d))

        batch_distances = mahalanobis_distances(X, mean, inv_cov)
        individual_distances = np.array(
            [mahalanobis_distance(X[i], mean, inv_cov) for i in range(n)]
        )
        np.testing.assert_allclose(batch_distances, individual_distances, atol=1e-7)

    def test_scaling_invariance(self):
        """
        Scaling the deviation vector by a scalar k scales the Mahalanobis distance
        by |k|.  This validates that the metric is a proper norm.
        """
        rng = _rng(15)
        d = 12
        mean = np.zeros(d)
        inv_cov = np.eye(d)
        v = rng.standard_normal(d)
        k = 3.5

        d1 = mahalanobis_distance(mean + v, mean, inv_cov)
        d_k = mahalanobis_distance(mean + k * v, mean, inv_cov)

        self.assertAlmostEqual(d_k, k * d1, places=8)

    def test_diagonal_covariance_normalisation(self):
        """
        With a diagonal covariance Sigma = diag(sigma_1^2, ..., sigma_d^2), the Mahalanobis
        distance should equal the root-sum of squared standardised deviations:

            d_M(x) = sqrt[ Sigma_i ((x_i - mu_i) / sigma_i)^2 ]
        """
        rng = _rng(16)
        d = 8
        sigmas = rng.uniform(0.5, 3.0, size=d)
        mean = rng.standard_normal(d)
        inv_cov = np.diag(1.0 / sigmas ** 2)

        x = mean + rng.standard_normal(d)

        maha = mahalanobis_distance(x, mean, inv_cov)
        expected = float(np.sqrt(np.sum(((x - mean) / sigmas) ** 2)))

        self.assertAlmostEqual(maha, expected, places=8)


# ---------------------------------------------------------------------------
# fit_distribution integration tests
# ---------------------------------------------------------------------------


class TestFitDistribution(unittest.TestCase):
    """Integration-level tests for fit_distribution()."""

    def test_output_keys_present(self):
        """fit_distribution must return all required keys."""
        rng = _rng(20)
        features = rng.standard_normal((60, 16)).astype(np.float32)
        result = fit_distribution(features)
        for key in ("mean", "inv_cov", "thresholds", "calibration_distances",
                    "shrinkage_alpha", "n_samples"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_threshold_ordering(self):
        """p90 <= p95 <= p99."""
        rng = _rng(21)
        features = rng.standard_normal((80, 16)).astype(np.float32)
        result = fit_distribution(features)
        t = result["thresholds"]
        self.assertLessEqual(t["90"], t["95"])
        self.assertLessEqual(t["95"], t["99"])

    def test_calibration_distances_count(self):
        """One calibration distance per input image."""
        rng = _rng(22)
        n = 75
        features = rng.standard_normal((n, 16)).astype(np.float32)
        result = fit_distribution(features)
        self.assertEqual(len(result["calibration_distances"]), n)

    def test_empty_input_raises(self):
        """fit_distribution must raise ValueError on empty input."""
        with self.assertRaises(ValueError):
            fit_distribution(np.empty((0, 16), dtype=np.float32))

    def test_known_good_samples_mostly_below_p99(self):
        """
        By definition, at most 1 % of calibration samples should exceed the p99
        threshold.  With n=200 we expect at most 2 above the threshold.
        """
        rng = _rng(23)
        n = 200
        features = rng.standard_normal((n, 16)).astype(np.float32)
        result = fit_distribution(features)
        distances = result["calibration_distances"]
        threshold_99 = result["thresholds"]["99"]
        n_above = int(np.sum(distances > threshold_99))
        # Allow for slight floating-point variation: at most 2 above
        self.assertLessEqual(n_above, 2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main(verbosity=2)
