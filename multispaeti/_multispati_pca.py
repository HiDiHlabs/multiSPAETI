import warnings
from typing import TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, eye, issparse
from scipy.sparse import linalg as sparse_linalg
from scipy.sparse import sparray, spmatrix
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


# TODO: should pass the following check
# from sklearn.utils.estimator_checks import check_estimator
# check_estimator(MultispatiPCA())


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
        If None, will keep all components (only supported for non-sparse `X`).
        If an int, it will keep the top `n_components`.
        If a tuple, it will keep the top and bottom `n_components` respectively.
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
        connectivity: sparray | spmatrix | None = None,
    ):
        self.n_components = n_components
        self.connectivity = connectivity

    @staticmethod
    def _validate_connectivity(W: csr_array, n: int):
        if W.shape[0] != W.shape[1]:
            raise ValueError("`connectivity` must be square")
        if W.shape[0] != n:
            raise ValueError(
                "#rows in `X` must be the same as dimensions of `connectivity`"
            )

    def _validate_n_components(self, n: int):
        n_components = self.n_components

        self._n_neg = 0
        if n_components is None:
            self._n_pos = n_components
        else:
            if isinstance(n_components, int):
                if n_components > n:
                    warnings.warn(
                        "`n_components` should be less or equal than "
                        f"#rows of `connectivity`. Using {n} components."
                    )
                self._n_pos = min(n_components, n)
            elif isinstance(n_components, tuple) and len(n_components) == 2:
                if n < n_components[0] + n_components[1]:
                    warnings.warn(
                        "Sum of `n_components` should be less or equal than "
                        f"#rows of `connectivity`. Using {n} components."
                    )
                    self._n_pos = n
                else:
                    self._n_pos, self._n_neg = n_components
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
            If `n_components` is None and `X` is sparse.
            If (sum of) `n_components` is larger than the smaller dimension of `X`.
            If `connectivity` is not a square matrix.
        """

        X = check_array(X)
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
        self._validate_n_components(n)

        self.W_ = normalize(W, norm="l1")
        assert isinstance(self.W_, csr_array)

        if issparse(X):
            X = csc_array(X)

        assert isinstance(X, (np.ndarray, csc_array))
        if self._n_pos is None:
            if issparse(X):
                raise ValueError(
                    "`n_components` is None, but `X` is a sparse matrix. None is only "
                    "supported for dense matrices."
                )
        elif (self._n_pos + self._n_neg) > X.shape[1]:
            n_comp = self._n_pos + self._n_neg
            n_comp_max = min(n, d)
            raise ValueError(
                f"Requested {n_comp} components but given `X` at most {n_comp_max} "
                "can be calculated."
            )

        eig_val, eig_vec = self._multispati_eigendecomposition(X, self.W_)

        self.components_ = eig_vec
        self.eigenvalues_ = eig_val
        self.n_components_ = eig_val.size
        self._n_features_out = self.n_components_
        self.n_features_in_ = d

        self.variance_, self.moransI_ = self._variance_moransI_decomposition(
            X @ self.components_
        )

        return self

    def _multispati_eigendecomposition(
        self, X: _X, W: _Csr
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # X: beads/bin x gene, must be standardized
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
            # TODO fix can't return all eigenvalues of sparse matrix
            # TODO check that number of eigenvalues does not exceed d
            if self._n_pos is None:
                raise ValueError(
                    "`n_components` is None, but `X` is a sparse matrix. None is only "
                    "supported for dense matrices."
                )
            elif self._n_pos == 0:
                eig_val, eig_vec = sparse_linalg.eigsh(H, k=self._n_neg, which="SA")
            elif self._n_neg == 0:
                eig_val, eig_vec = sparse_linalg.eigsh(H, k=self._n_pos, which="LA")
            else:
                n_comp = 2 * max(self._n_neg, self._n_pos)
                eig_val, eig_vec = sparse_linalg.eigsh(H, k=n_comp, which="BE")
                component_indices = self._get_component_indices(
                    n_comp, self._n_pos, self._n_neg
                )
                eig_val = eig_val[component_indices]
                eig_vec = eig_vec[:, component_indices]

        else:
            if self._n_pos is None:
                eig_val, eig_vec = linalg.eigh(H)
                if n < d:
                    eig_val, eig_vec = remove_zero_eigenvalues(eig_val, eig_vec, n)
            elif self._n_pos == 0:
                eig_val, eig_vec = linalg.eigh(H, subset_by_index=[0, self._n_neg])
            elif self._n_neg == 0:
                eig_val, eig_vec = linalg.eigh(
                    H, subset_by_index=[d - self._n_pos, d - 1]
                )
            else:
                eig_val, eig_vec = linalg.eigh(H)
                component_indices = self._get_component_indices(
                    d, self._n_pos, self._n_neg
                )
                eig_val = eig_val[component_indices]
                eig_vec = eig_vec[:, component_indices]

        return np.flip(eig_val), np.fliplr(eig_vec)

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
        X = check_array(X)
        return X @ self.components_

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

        # vector of row_Weights from dudi.PCA
        # (we only use default row_weights i.e. 1/n)
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

        if issparse(W):
            eigen_values = s * sparse_linalg.eigsh(
                W, k=2, which="BE", return_eigenvectors=False
            )
        else:
            eigen_values = s * linalg.eigvalsh(W, overwrite_a=True)

        I_0 = -1 / (n_sample - 1)
        I_min = min(eigen_values)
        I_max = max(eigen_values)

        return I_min, I_max, I_0
