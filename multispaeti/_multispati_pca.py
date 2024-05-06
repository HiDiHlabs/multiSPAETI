import warnings
from typing import TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, eye, issparse
from scipy.sparse import linalg as sparse_linalg
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array, check_is_fitted

T = TypeVar("T", bound=np.number)
U = TypeVar("U", bound=np.number)


_Csr: TypeAlias = csr_array | csr_matrix
_Csc: TypeAlias = csc_array | csc_matrix
_X: TypeAlias = np.ndarray | _Csr | _Csc
_Connectivity: TypeAlias = np.ndarray | _Csr


class MultispatiPCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """
    MULTISPATI-PCA

    In contrast to Principal component analysis (PCA), MULTISPATI-PCA does not optimize
    the variance explained of each component but rather the product of the variance and
    Moran's I. This can lead to negative eigenvalues i.e. in the case of negative
    auto-correlation.

    The problem is solved by diagonalizing the symmetric matrix
    :math:`H=1/(2n)*X^t(W+W^t)X` where `X` is matrix of `n` observations :math:`\\times`
    `d` features, and `W` is a matrix of the connectivity between observations.

    Parameters
    ----------
    n_components : int or tuple[int, int], optional
        Number of components to keep.
        If None, will keep all components.
        If an int, it will keep the top `n_components`.
        If a tuple, it will keep the top and bottom `n_components`, respectively.
    connectivity : scipy.sparse.sparray or scipy.sparse.spmatrix
        Matrix of row-wise neighbor definitions i.e. c\ :sub:`ij` is the connectivity of
        i :math:`\\to` j. The matrix does not have to be symmetric. It can be a
        binary adjacency matrix or a matrix of connectivities in which case
        c\ :sub:`ij` should be larger if i and j are close.
        A distance matrix should be transformed to connectivities by e.g.
        calculating :math:`1-d/d_{max}` beforehand.

    Attributes
    ----------
    components_ : numpy.ndarray
        The estimated components: Array of shape `(n_components, n_features)`.

    eigenvalues_ : numpy.ndarray
        The eigenvalues corresponding to each of the selected components. Array of shape
        `(n_components,)`.

    variance_ : numpy.ndarray
        The estimated variance part of the eigenvalues. Array of shape `(n_components,)`.

    moransI_ : numpy.ndarray
        The estimated Moran's I part of the eigenvalues. Array of shape `(n_components,)`.

    mean_ : numpy.ndarray or None
        Per-feature empirical mean, estimated from the training set if `X` is not sparse.
        Array of shape `(n_features,)`.

    n_components_ : int
        The estimated number of components.

    n_samples_ : int
        Number of samples in the training data.

    n_features_in_ : int
        Number of features seen during :term:`fit`.


    References
    ----------
    `Dray, Stéphane, Sonia Saïd, and Françis Débias. "Spatial ordination of vegetation
    data using a generalization of Wartenberg's multivariate spatial correlation."
    Journal of vegetation science 19.1 (2008): 45-56.
    <https://onlinelibrary.wiley.com/doi/abs/10.3170/2007-8-18312>`_
    """

    def __init__(
        self,
        n_components: int | tuple[int, int] | None = None,
        *,
        connectivity: _Connectivity | None = None,
    ):
        self.n_components = n_components
        self.connectivity = connectivity

    @staticmethod
    def _validate_connectivity(W: _Connectivity, n: int):
        if W.shape[0] != W.shape[1]:
            raise ValueError("`connectivity` must be square")
        if W.shape[0] != n:
            raise ValueError(
                "#rows in `X` must be the same as dimensions of `connectivity`"
            )

    def _validate_n_components(self, n: int, d: int):
        self._n_components = self.n_components

        m = min(n, d)

        if self.n_components is not None:
            if isinstance(self.n_components, int):
                if self.n_components <= 0:
                    raise ValueError("`n_components` must be a positive integer.")
                elif self.n_components >= m:
                    warnings.warn(
                        "`n_components` should be less than minimum of "
                        "#samples and #features. Using all components."
                    )
                self._n_components = None
            elif isinstance(self.n_components, tuple) and len(self.n_components) == 2:
                if any(
                    not isinstance(i, int) or i < 0 for i in self.n_components
                ) or self.n_components == (0, 0):
                    raise ValueError(
                        "`n_components` must be a tuple of positive integers."
                    )
                elif sum(self.n_components) >= m:
                    warnings.warn(
                        "Sum of `n_components` should be less than minimum of "
                        "#samples and #features. Using all components."
                    )
                    self._n_components = None
            else:
                raise ValueError("`n_components` must be None, int or (int, int)")

    def fit(self, X: _X, y: None = None):
        """
        Fit MULTISPATI-PCA projection.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
            Array of observations x features.
        y : None
            Ignored. scikit-learn compatibility only.

        Raises
        ------
        ValueError
            If `X` does not have the same number of rows as `connectivity`.
            If `n_components` has the wrong type or is negative.
            If `connectivity` is not a square matrix.
        """
        self._fit(X)
        return self

    def _fit(self, X: _X, *, return_transform: bool = False) -> np.ndarray | None:

        X = check_array(X, accept_sparse=["csr", "csc"])
        if self.connectivity is None:
            warnings.warn(
                "`connectivity` has not been set. Defaulting to identity matrix "
                "which will conceptually compute a standard PCA. "
                "It is not recommended to not set `connectivity`."
            )
            W = csr_array(eye(X.shape[0]))
        else:
            W = self.connectivity
        W = check_array(W, accept_sparse="csr")

        n, d = X.shape

        self._validate_connectivity(W, n)
        self._validate_n_components(n, d)

        self.W_ = normalize(W, norm="l1")

        if issparse(X):
            self.mean_ = None
            X_centered = X
        else:
            self.mean_ = X.mean(axis=0)
            X_centered = X - self.mean_
            assert isinstance(X_centered, np.ndarray)

        eig_val, eig_vec = self._multispati_eigendecomposition(X_centered, self.W_)

        self.components_ = eig_vec
        self.eigenvalues_ = eig_val
        self.n_components_ = eig_val.size
        self._n_features_out = self.n_components_
        self.n_features_in_ = d

        X_tr = X_centered @ self.components_.T
        self.variance_, self.moransI_ = self._variance_moransI_decomposition(X_tr)

        if return_transform:
            return X_tr

    def _multispati_eigendecomposition(
        self, X: _X, W: _Connectivity
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # X: observations x features
        # W: row-wise definition of neighbors, row-sums should be 1
        def remove_zero_eigenvalues(
            eigen_values: NDArray[T], eigen_vectors: NDArray[U], n: int
        ) -> tuple[NDArray[T], NDArray[U]]:
            keep_idx = np.sort(np.argpartition(np.abs(eigen_values), -n)[-n:])

            return eigen_values[keep_idx], eigen_vectors[:, keep_idx]

        n, d = X.shape

        H = (X.T @ (W + W.T) @ X) / (2 * n)
        # TODO handle sparse based on density?
        if issparse(H):
            match self._n_components:
                case None:
                    eig_val, eig_vec = sparse_linalg.eigsh(
                        H, k=min(n, d) - 1, which="LM"
                    )
                case (n_pos, 0) | int(n_pos):
                    eig_val, eig_vec = sparse_linalg.eigsh(H, k=n_pos, which="LA")
                case (0, n_neg):
                    eig_val, eig_vec = sparse_linalg.eigsh(H, k=n_neg, which="SA")
                case (n_pos, n_neg):
                    n_comp = min(2 * max(n_neg, n_pos), min(n, d))
                    eig_val, eig_vec = sparse_linalg.eigsh(H, k=n_comp, which="BE")
                    component_indices = self._get_component_indices(
                        n_comp, n_pos, n_neg
                    )
                    eig_val = eig_val[component_indices]
                    eig_vec = eig_vec[:, component_indices]

        else:
            match self._n_components:
                case None:
                    eig_val, eig_vec = linalg.eigh(H)
                    if n < d:
                        eig_val, eig_vec = remove_zero_eigenvalues(eig_val, eig_vec, n)
                case (n_pos, 0) | int(n_pos):
                    eig_val, eig_vec = linalg.eigh(
                        H, subset_by_index=[d - n_pos, d - 1]
                    )
                case (0, n_neg):
                    eig_val, eig_vec = linalg.eigh(H, subset_by_index=[0, n_neg])
                case (n_pos, n_neg):
                    eig_val, eig_vec = linalg.eigh(H)
                    component_indices = self._get_component_indices(d, n_pos, n_neg)
                    eig_val = eig_val[component_indices]
                    eig_vec = eig_vec[:, component_indices]

        return np.flip(eig_val), np.flipud(eig_vec.T)

    @staticmethod
    def _get_component_indices(n: int, n_pos: int, n_neg: int) -> list[int]:
        if n_pos + n_neg > n:
            return list(range(n))
        else:
            return list(range(n_neg)) + list(range(n - n_pos, n))

    def transform(self, X: _X) -> np.ndarray:
        """
        Transform the data using fitted MULTISPATI-PCA projection.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
            Array of observations x features.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If instance has not been fitted.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=["csr", "csc"])
        if self.mean_ is not None and not issparse(X):
            X = X - self.mean_
        return X @ self.components_.T

    def fit_transform(self, X: _X, y: None = None) -> np.ndarray:
        """
        Fit and transform the data using MULTISPATI-PCA projection.

        See :py:meth:`multispaeti.MultispatiPCA` for more information.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
            Array of observations x features.
        y : None
            Ignored. scikit-learn compatibility only.

        Returns
        -------
        numpy.ndarray
        """
        X_tr = self._fit(X, return_transform=True)
        assert isinstance(X_tr, np.ndarray)
        return X_tr

    def transform_spatial_lag(self, X: _X) -> np.ndarray:
        """
        Transform the data using fitted MULTISPATI-PCA projection and calculate the
        spatial lag.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
            Array of observations x features.

        Returns
        -------
        numpy.ndarray

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If instance has not been fitted.
        """
        check_is_fitted(self)
        return self._spatial_lag(self.transform(X))

    def _spatial_lag(self, X: np.ndarray) -> np.ndarray:
        return self.W_ @ X

    def _variance_moransI_decomposition(
        self, X_tr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lag = self._spatial_lag(X_tr)

        # vector of row_Weights from dudi.PCA (we only use default row_weights i.e. 1/n)
        w = 1 / X_tr.shape[0]

        variance = np.sum(X_tr * X_tr * w, axis=0)
        moran = np.sum(X_tr * lag * w, axis=0) / variance

        return variance, moran

    def moransI_bounds(
        self, *, sparse_approx: bool = True
    ) -> tuple[float, float, float]:
        """
        Calculate the minimum and maximum bound for Moran's I given the `connectivity`
        and the expected value given the #observations.

        Parameters
        ----------
        sparse_approx : bool
            Only applicable if `connectivity` is sparse.

        Returns
        -------
        tuple[float, float, float]
            Minimum bound, maximum bound, and expected value.
        """

        # following R package adespatial::moran.bounds
        # sparse approx is following adegenet sPCA as shown in screeplot/summary
        def double_center(W: np.ndarray | csr_array) -> np.ndarray:
            if issparse(W):
                assert isinstance(W, csr_array)
                W = W.toarray()
            assert isinstance(W, np.ndarray)

            row_means = np.mean(W, axis=1, keepdims=True)
            col_means = np.mean(W, axis=0, keepdims=True) - np.mean(row_means)

            return W - row_means - col_means

        # ensure symmetry
        W = 0.5 * (self.W_ + self.W_.T)

        n_sample = W.shape[0]
        s = n_sample / np.sum(W)  # 1 if original W has rowSums or colSums of 1

        if not issparse(W) or not sparse_approx:
            W = double_center(W)

        eigen_values = s * sparse_linalg.eigsh(
            W, k=2, which="BE", return_eigenvectors=False
        )

        I_0 = -1 / (n_sample - 1)
        I_min = min(eigen_values)
        I_max = max(eigen_values)

        return I_min, I_max, I_0
