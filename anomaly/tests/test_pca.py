"""
Unit tests for anomaly.pca -- PCA dimensionality reduction.

All tests use synthetic NumPy arrays; no ML framework or image data required.
Run with:
    python -m pytest anomaly/tests/test_pca.py -v
"""

import unittest

import numpy as np

from anomaly.pca import PCA


def _rng(seed  = 0)  :
    return np.random.default_rng(seed)


class TestPCAFit(unittest.TestCase):
    """Tests for PCA fitting behaviour."""

    def test_fitted_attributes_set(self):
        """After fit(), mean_, components_, eigenvalues_, n_components_ are set."""
        rng = _rng(0)
        X = rng.standard_normal((50, 20))
        pca = PCA(n_components=10)
        pca.fit(X)
        self.assertIsNotNone(pca.mean_)
        self.assertIsNotNone(pca.components_)
        self.assertIsNotNone(pca.eigenvalues_)
        self.assertIsNotNone(pca.n_components_)

    def test_eigenvalues_descending_order(self):
        """Eigenvalues must be in strictly non-increasing order."""
        rng = _rng(1)
        X = rng.standard_normal((100, 30))
        pca = PCA(n_components=30)
        pca.fit(X)
        eigenvalues = pca.eigenvalues_
        self.assertTrue(
            np.all(eigenvalues[:-1] >= eigenvalues[1:]),
            "Eigenvalues are not in descending order",
        )

    def test_eigenvalues_non_negative(self):
        """Covariance eigenvalues must be >= 0 (PSD matrix)."""
        rng = _rng(2)
        X = rng.standard_normal((60, 25))
        pca = PCA(n_components=25)
        pca.fit(X)
        self.assertTrue(
            np.all(pca.eigenvalues_ >= 0),
            f"Negative eigenvalue found: {pca.eigenvalues_.min():.4e}",
        )

    def test_eigenvalues_count_equals_input_dim(self):
        """eigenvalues_ should contain all d eigenvalues, not just retained ones."""
        rng = _rng(3)
        d = 40
        X = rng.standard_normal((80, d))
        pca = PCA(n_components=10)
        pca.fit(X)
        self.assertEqual(len(pca.eigenvalues_), d)

    def test_unfitted_transform_raises(self):
        """Calling transform() before fit() must raise RuntimeError."""
        pca = PCA(n_components=5)
        with self.assertRaises(RuntimeError):
            pca.transform(np.zeros((10, 20)))


class TestPCAComponents(unittest.TestCase):
    """Tests for dimensionality reduction and component selection."""

    def test_fixed_n_components_output_shape(self):
        """With n_components=10, transform output shape must be (N, 10)."""
        rng = _rng(10)
        X = rng.standard_normal((80, 64))
        pca = PCA(n_components=10)
        X_reduced = pca.fit_transform(X)
        self.assertEqual(X_reduced.shape, (80, 10))

    def test_components_matrix_shape(self):
        """components_ must have shape (d, k)."""
        rng = _rng(11)
        d, k = 64, 15
        X = rng.standard_normal((100, d))
        pca = PCA(n_components=k)
        pca.fit(X)
        self.assertEqual(pca.components_.shape, (d, k))

    def test_components_are_orthonormal(self):
        """
        The principal components must be orthonormal: W^T W = I.

        This follows from the eigendecomposition of a symmetric matrix;
        eigenvectors are orthogonal and are normalised by ``eigh``.
        """
        rng = _rng(12)
        X = rng.standard_normal((150, 50))
        pca = PCA(n_components=30)
        pca.fit(X)
        W = pca.components_   # (50, 30)
        gram = W.T @ W        # should be I_30
        np.testing.assert_allclose(
            gram,
            np.eye(30),
            atol=1e-10,
            err_msg="Principal components are not orthonormal (W^T W != I)",
        )

    def test_variance_threshold_selects_few_components(self):
        """
        With synthetic data where only the first 3 dimensions carry variance,
        PCA with variance_threshold=0.95 should retain approximately 3 components.

        Data: first 3 dims have std=10, remaining 50 dims have std=0.01.
        The first 3 PCs should capture > 99% of variance; threshold=0.95
        should therefore select exactly 3 components.
        """
        rng = _rng(13)
        n, d_signal, d_noise = 300, 3, 50
        d_signal + d_noise
        signal = rng.standard_normal((n, d_signal)) * 10.0
        noise = rng.standard_normal((n, d_noise)) * 0.01
        X = np.concatenate([signal, noise], axis=1)

        pca = PCA(variance_threshold=0.95)
        pca.fit(X)

        self.assertLessEqual(
            pca.n_components_,
            d_signal + 2,
            f"Expected <= {d_signal + 2} components for predominantly 3-dim data, "
            f"got {pca.n_components_}",
        )
        self.assertGreaterEqual(
            pca.n_components_,
            d_signal - 1,
            f"Expected >= {d_signal - 1} components, got {pca.n_components_}",
        )

    def test_variance_threshold_cumulative_meets_target(self):
        """The cumulative variance of retained components must meet the threshold."""
        rng = _rng(14)
        X = rng.standard_normal((200, 64))
        threshold = 0.90
        pca = PCA(variance_threshold=threshold)
        pca.fit(X)
        _, cumulative = pca.explained_variance_ratio()
        self.assertGreaterEqual(
            cumulative[-1],
            threshold - 1e-9,
            f"Retained components explain only {cumulative[-1]:.3f} < {threshold}",
        )

    def test_n_components_overrides_variance_threshold(self):
        """When n_components is set, it takes priority over variance_threshold."""
        rng = _rng(15)
        X = rng.standard_normal((100, 40))
        pca = PCA(n_components=7, variance_threshold=0.99)
        pca.fit(X)
        self.assertEqual(pca.n_components_, 7)


class TestPCATransform(unittest.TestCase):
    """Tests for the transform and inverse_transform operations."""

    def test_centering_of_training_data(self):
        """
        The mean of transform(X_train) should be approximately zero.

        After centering X by the training mean and projecting, the projected
        training data must have zero mean along every component axis.
        """
        rng = _rng(20)
        X = rng.standard_normal((120, 30)) + rng.standard_normal(30) * 5
        pca = PCA(n_components=15)
        X_proj = pca.fit_transform(X)
        component_means = np.abs(X_proj.mean(axis=0))
        np.testing.assert_allclose(
            component_means,
            0.0,
            atol=1e-10,
            err_msg="Projected training data does not have zero mean per component",
        )

    def test_reconstruction_full_components(self):
        """
        When all d components are retained, inverse_transform(transform(X))
        should recover X to within floating-point tolerance.
        """
        rng = _rng(21)
        n, d = 80, 20
        X = rng.standard_normal((n, d))
        pca = PCA(n_components=d)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)
        np.testing.assert_allclose(
            X_reconstructed,
            X,
            atol=1e-8,
            err_msg="Full-component reconstruction does not recover original data",
        )

    def test_transform_unseen_data(self):
        """transform() on unseen data should produce correct output shape."""
        rng = _rng(22)
        X_train = rng.standard_normal((100, 32))
        X_test = rng.standard_normal((25, 32))
        pca = PCA(n_components=8)
        pca.fit(X_train)
        X_test_proj = pca.transform(X_test)
        self.assertEqual(X_test_proj.shape, (25, 8))

    def test_fit_transform_matches_fit_then_transform(self):
        """fit_transform(X) must give the same result as fit(X) then transform(X)."""
        rng = _rng(23)
        X = rng.standard_normal((60, 20))
        pca_a = PCA(n_components=10)
        result_a = pca_a.fit_transform(X)

        pca_b = PCA(n_components=10)
        pca_b.fit(X)
        result_b = pca_b.transform(X)

        np.testing.assert_allclose(
            result_a, result_b, atol=1e-12,
            err_msg="fit_transform and fit+transform give different results",
        )


class TestPCAExplainedVariance(unittest.TestCase):
    """Tests for explained_variance_ratio()."""

    def test_ratios_sum_to_cumulative_end(self):
        """Cumulative[-1] must equal sum(ratios)."""
        rng = _rng(30)
        X = rng.standard_normal((100, 20))
        pca = PCA(n_components=10)
        pca.fit(X)
        ratios, cumulative = pca.explained_variance_ratio()
        self.assertAlmostEqual(
            cumulative[-1], ratios.sum(), places=12
        )

    def test_ratios_are_positive(self):
        """All explained variance ratios must be non-negative."""
        rng = _rng(31)
        X = rng.standard_normal((80, 15))
        pca = PCA(n_components=15)
        pca.fit(X)
        ratios, _ = pca.explained_variance_ratio()
        self.assertTrue(np.all(ratios >= 0))

    def test_full_components_explain_all_variance(self):
        """Retaining all d components must explain 100% of variance."""
        rng = _rng(32)
        d = 12
        X = rng.standard_normal((50, d))
        pca = PCA(n_components=d)
        pca.fit(X)
        _, cumulative = pca.explained_variance_ratio()
        self.assertAlmostEqual(cumulative[-1], 1.0, places=8)

    def test_ratios_length_matches_n_components(self):
        """Length of returned arrays must equal n_components_."""
        rng = _rng(33)
        X = rng.standard_normal((100, 30))
        k = 12
        pca = PCA(n_components=k)
        pca.fit(X)
        ratios, cumulative = pca.explained_variance_ratio()
        self.assertEqual(len(ratios), k)
        self.assertEqual(len(cumulative), k)

    def test_unfitted_explained_variance_raises(self):
        """explained_variance_ratio() before fit() must raise RuntimeError."""
        pca = PCA(n_components=5)
        with self.assertRaises(RuntimeError):
            pca.explained_variance_ratio()


class TestPCAEdgeCases(unittest.TestCase):
    """Edge case and validation tests."""

    def test_invalid_n_components_raises(self):
        """n_components < 1 must raise ValueError at construction."""
        with self.assertRaises(ValueError):
            PCA(n_components=0)

    def test_n_components_exceeds_dim_raises(self):
        """n_components > d must raise ValueError at fit time."""
        rng = _rng(40)
        X = rng.standard_normal((50, 10))
        pca = PCA(n_components=20)
        with self.assertRaises(ValueError):
            pca.fit(X)

    def test_invalid_variance_threshold_raises(self):
        """variance_threshold outside (0, 1] must raise ValueError."""
        with self.assertRaises(ValueError):
            PCA(variance_threshold=0.0)
        with self.assertRaises(ValueError):
            PCA(variance_threshold=1.5)

    def test_single_sample_raises(self):
        """N=1 must raise ValueError (cannot form unbiased covariance)."""
        pca = PCA(n_components=1)
        with self.assertRaises(ValueError):
            pca.fit(np.ones((1, 10)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
