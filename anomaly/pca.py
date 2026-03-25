"""
Principal Component Analysis (PCA) for feature-space dimensionality reduction.

This module provides a pure-NumPy PCA implementation used as an intermediate
step between ResNet-18 feature extraction (512-dim) and multivariate Gaussian
fitting.

Why PCA helps here
------------------
A ResNet-18 backbone trained on ImageNet encodes a rich vocabulary of visual
concepts: animal fur, sky gradients, vehicle parts, text, and much more.
When applied to bottle-cap images, most of those 512 feature dimensions encode
concepts that are entirely irrelevant to date stamps.  These irrelevant
dimensions have low variance across cap images -- all caps look roughly the
same along those axes.

Low-variance dimensions are doubly harmful in Mahalanobis scoring:

1. **Noise amplification.** Each dimension contributes independently to the
   Mahalanobis distance.  A dimension where all good caps cluster tightly can
   still push the distance of a genuinely good cap above the threshold if its
   residual in that direction happens to be unlucky.

2. **Ill-conditioning.** With n ~= 100-500 calibration images and 512 features,
   the sample covariance is rank-deficient; the low-variance dimensions
   contribute near-zero eigenvalues that destabilise the inverse.  Ledoit-Wolf
   shrinkage mitigates this but does not eliminate it.

PCA projects the data onto the subspace of *maximum variance*, discarding the
low-variance tail.  After projection, every retained dimension carries genuine
discriminative information, and the distribution model is fitted in a lower-
dimensional, better-conditioned space.

Mathematical Background
-----------------------
**Step 1 -- Centering.**
Subtract the training mean mu from every sample:

    X_c = X - mu

This is required because PCA operates on the covariance structure, which is
defined relative to the mean.

**Step 2 -- Covariance matrix.**
Compute the sample (unbiased) covariance matrix:

    C = (1 / (N - 1)) * X_c^T X_c     in R^{d x d}

C captures the pairwise linear relationships between the d feature dimensions.

**Step 3 -- Eigendecomposition.**
Factorise C as:

    C = V Lambda V^T

where the columns of V in R^{d x d} are the *principal components*
(orthonormal eigenvectors of C) and Lambda = diag(lambda_1, lambda_2, ..., lambda_d) contains the
corresponding eigenvalues in descending order.

Each eigenvalue lambda_i equals the variance of the data projected onto the i-th
principal component.  The eigenvectors are sorted so that lambda_1 >= lambda_2 >= ... >= lambda_d.

We use ``numpy.linalg.eigh`` instead of the general ``eig`` because C is
symmetric positive semi-definite.  ``eigh`` exploits symmetry for faster,
numerically more stable computation.  **Note:** ``eigh`` returns eigenvalues in
*ascending* order, so we reverse them immediately after the call.

**Step 4 -- Component selection.**
Select the top k principal components, where k is chosen by one of two
criteria:

* *Fixed count:*  k = n_components (user-specified).
* *Variance threshold:*  choose the smallest k such that

      sum_{i=1}^k lambda_i / sum_{i=1}^d lambda_i >= variance_threshold

  The ratio lambda_i / Sigma*lambda is the *explained variance ratio* of component i.

The retained projection matrix is W = V[:, :k]  in R^{d x k}.

**Step 5 -- Projection.**
Map a new (possibly unseen) sample x in R^d into the k-dimensional subspace:

    x_reduced = (x - mu) @ W     in R^k

The resulting vector x_reduced is the set of *scores* -- the coordinates of x
in the principal component basis.  Euclidean (and Mahalanobis) distances in
this space are meaningful and stable.

**Inverse transform (reconstruction).**
Given a reduced vector z in R^k, the original-space reconstruction is:

    x^ = z @ W^T + mu

This is exact when k = d and approximate (minimum reconstruction error)
otherwise.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PCA:
    """
    Principal Component Analysis implemented from scratch using NumPy.

    Supports two component-selection strategies:

    * ``n_components`` -- retain exactly this many components.
    * ``variance_threshold`` -- retain the fewest components that together
      explain at least this fraction of total variance (default: 0.95).

    If both are provided, ``n_components`` takes priority.
    If neither is provided, ``variance_threshold=0.95`` is used.

    Attributes set after ``fit()``:
        mean_ (np.ndarray): Training-data mean, shape ``(d,)``.
        components_ (np.ndarray): Projection matrix W, shape ``(d, k)``.
            Columns are orthonormal principal components in descending
            eigenvalue order.
        eigenvalues_ (np.ndarray): All d eigenvalues in descending order,
            shape ``(d,)``.  Includes the discarded tail for diagnostics.
        n_components_ (int): Actual number of retained components k.
    """

    def __init__(
        self,
        n_components  = None,
        variance_threshold  = 0.95,
    )  :
        """
        Initialise PCA.

        Args:
            n_components: Fixed number of components to retain.  If given,
                overrides ``variance_threshold``.
            variance_threshold: Retain the minimum number of components whose
                cumulative explained variance meets this fraction.  Ignored
                when ``n_components`` is set.  Default: 0.95.
        """
        if n_components is not None and n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if variance_threshold is not None and not (0.0 < variance_threshold <= 1.0):
            raise ValueError(
                f"variance_threshold must be in (0, 1], got {variance_threshold}"
            )

        self.n_components = n_components
        self.variance_threshold = variance_threshold

        # Fitted attributes (set by fit())
        self.mean_  = None
        self.components_  = None
        self.eigenvalues_  = None
        self.n_components_  = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X )  :
        """
        Fit PCA to the training data X.

        Computes the sample mean, forms the unbiased covariance matrix, and
        eigendecomposes it using ``numpy.linalg.eigh``.  The top k eigenvectors
        are retained as the projection matrix.

        Args:
            X: Training data of shape ``(N, d)``.  N >= 2 is required to form
               an unbiased covariance estimate.

        Returns:
            self -- for method chaining.

        Raises:
            ValueError: If N < 2 or if ``n_components`` exceeds d.
        """
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape

        if n < 2:
            raise ValueError(
                f"PCA requires at least 2 samples, got {n}."
            )
        if self.n_components is not None and self.n_components > d:
            raise ValueError(
                f"n_components={self.n_components} exceeds input dimensionality d={d}."
            )

        # Step 1: center
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # Step 2: unbiased covariance  C in R^{d x d}
        C = (X_c.T @ X_c) / (n - 1)

        # Step 3: eigendecompose -- eigh is faster & more stable for symmetric matrices
        # Returns eigenvalues in ASCENDING order; we reverse immediately.
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eigenvalues = eigenvalues[::-1].copy()       # descending
        eigenvectors = eigenvectors[:, ::-1].copy()  # match column order

        # Clamp tiny negatives from floating-point to 0
        eigenvalues = np.maximum(eigenvalues, 0.0)
        self.eigenvalues_ = eigenvalues

        # Step 4: select k components
        if self.n_components is not None:
            k = min(self.n_components, d)
        else:
            threshold = self.variance_threshold if self.variance_threshold is not None else 0.95
            total_var = eigenvalues.sum()
            if total_var < 1e-12:
                logger.warning("Total variance is near-zero; retaining all components.")
                k = d
            else:
                cumvar = np.cumsum(eigenvalues) / total_var
                # searchsorted finds first index where cumvar >= threshold
                k = int(np.searchsorted(cumvar, threshold)) + 1
                k = min(k, d)

        self.n_components_ = k
        self.components_ = eigenvectors[:, :k]   # (d, k)

        var_explained = (
            eigenvalues[:k].sum() / eigenvalues.sum()
            if eigenvalues.sum() > 1e-12
            else 1.0
        )
        logger.info(
            "PCA: Reducing feature space from %d to %d dimensions", d, k
        )
        logger.info(
            "PCA: Retained components explain %.1f%% of total variance",
            var_explained * 100,
        )
        logger.info(
            "PCA: Top 10 eigenvalues: %s",
            np.array2string(eigenvalues[:10], precision=4, suppress_small=True),
        )

        return self

    def transform(self, X )  :
        """
        Project data onto the retained principal components.

        Centers X using the training mean, then multiplies by the projection
        matrix W:

            X_reduced = (X - mu) @ W

        Args:
            X: Data of shape ``(N, d)``.  Must have the same d as the training
               data.

        Returns:
            Projected data of shape ``(N, k)`` where k = ``n_components_``.

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X )  :
        """
        Fit PCA to X and return the projected data.

        Equivalent to ``fit(X).transform(X)`` but avoids re-centering.

        Args:
            X: Training data of shape ``(N, d)``.

        Returns:
            Projected data of shape ``(N, k)``.
        """
        self.fit(X)
        # Use pre-centred form directly for numerical consistency
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, X_reduced )  :
        """
        Map reduced-space coordinates back to the original feature space.

        The reconstruction is:

            x^ = X_reduced @ W^T + mu

        This is an exact inverse when k = d, and the minimum reconstruction-
        error approximation otherwise (by the Eckart-Young theorem).

        Args:
            X_reduced: Projected data of shape ``(N, k)``.

        Returns:
            Reconstructed data of shape ``(N, d)``.
        """
        self._check_fitted()
        X_reduced = np.asarray(X_reduced, dtype=np.float64)
        return X_reduced @ self.components_.T + self.mean_

    def explained_variance_ratio(self)   :
        """
        Return per-component and cumulative explained variance ratios.

        The explained variance ratio of component i is:

            r_i = lambda_i / Sigma*lambda

        where lambda_i is the i-th retained eigenvalue and Sigma*lambda is the sum of all d
        eigenvalues (including discarded components).

        Returns:
            A tuple ``(ratios, cumulative)`` where both arrays have length
            ``n_components_``:

            - *ratios*: fraction of total variance per retained component.
            - *cumulative*: cumulative sum of ratios (last entry = total
              variance fraction retained).

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        self._check_fitted()
        total = self.eigenvalues_.sum()
        if total < 1e-12:
            ratios = np.ones(self.n_components_) / self.n_components_
        else:
            ratios = self.eigenvalues_[: self.n_components_] / total
        return ratios, np.cumsum(ratios)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self)  :
        """Raise RuntimeError if fit() has not been called."""
        if self.mean_ is None:
            raise RuntimeError(
                "PCA has not been fitted yet. Call fit() before transform()."
            )
