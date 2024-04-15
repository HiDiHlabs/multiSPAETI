import warnings
from typing import TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, issparse
from scipy.sparse import linalg as sparse_linalg
from scipy.sparse import sparray, spmatrix
from sklearn.preprocessing import scale

T = TypeVar("T", bound=np.number)
U = TypeVar("U", bound=np.number)


_Csr: TypeAlias = csr_array | csr_matrix
_Csc: TypeAlias = csc_array | csc_matrix
_X: TypeAlias = np.ndarray | _Csr | _Csc


class MultispatiPCA:
    """
    MULTISPATI-PCA

    In contrast to Principal component analysis (PCA), MULTISPATI-PCA does not optimize
    the variance explained of each component but rather the product of the variance and
    Moran's I. This can lead to negative eigenvalues i.e. in the case of negative
    auto-correlation.

    The problem is solved by diagonalizing the symmetric matrix
    :math:`H=1/(2n)*X^t(W+W^t)X` where `X` is matrix of `n` observations :math:`\\times`
    `d` features, and `W` is a matrix of the connectivity between observations.

    References
    ----------
    `Dray, Stéphane, Sonia Saïd, and Françis Débias. "Spatial ordination of vegetation
    data using a generalization of Wartenberg's multivariate spatial correlation."
    Journal of vegetation science 19.1 (2008): 45-56.
    <https://onlinelibrary.wiley.com/doi/abs/10.3170/2007-8-18312>`_
    """

    # TODO, should scaling be part of multispati

    def __init__(
        self,
        n_components: int | tuple[int, int] | None = None,
        *,
        connectivity: sparray | spmatrix,
    ):
        """
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
        Raises
        ------
        ValueError
            If connectivity is not a square matrix.
        ZeroDivisionError
            If one of the observations has no neighbors.
        """
        self._fitted = False
        W = csr_array(connectivity)
        if W.shape[0] != W.shape[1]:
            raise ValueError("`connectivity` must be square")
        self.W = self._normalize_connectivities(W)

        n = self.W.shape[0]

        self.n_components = n_components

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

    @staticmethod
    def _normalize_connectivities(W: csr_array) -> csr_array:
        # normalize rowsums to 1 for better interpretability
        # TODO can't handle points without neighbors because of division by zero
        return W.multiply(1 / W.sum(axis=1)[:, np.newaxis])

    def fit(self, X: _X):
        """
        Fit MULTISPATI-PCA projection.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
            Array of observations x features.

        Raises
        ------
        ValueError
            If `X` has not the same number of rows like `connectivity`.
            If `n_components` is None and `X` is sparse.
            If (sum of) `n_components` is larger than the smaller dimension of `X`.
        """
        if issparse(X):
            X = csc_array(X)

        assert isinstance(X, (np.ndarray, csc_array))
        if X.shape[0] != self.W.shape[0]:
            raise ValueError(
                "#rows in `X` must be the same as dimensions of `connectivity`"
            )
        if self._n_pos is None:
            if issparse(X):
                raise ValueError(
                    "`n_components` is None, but `X` is a sparse matrix. None is only "
                    "supported for dense matrices."
                )
        elif (self._n_pos + self._n_neg) > X.shape[1]:
            n, d = X.shape
            n_comp = self._n_pos + self._n_neg
            n_comp_max = min(n, d)
            raise ValueError(
                f"Requested {n_comp} components but given `X` at most {n_comp_max} "
                "can be calculated."
            )

        # Data must be scaled, avoid mean-centering for sparse
        X = scale(X, with_mean=not issparse(X))

        eig_val, eig_vec = self._multispati_eigendecomposition(X, self.W)

        self.components_ = eig_vec
        self.eigenvalues_ = eig_val
        self.n_components_ = eig_val.size
        self.n_features_in_ = X.shape[1]
        self._fitted = True

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
        ValueError
            If instance has not been fitted.
        """
        # Data must be scaled, avoid mean-centering for sparse
        if not self._fitted:
            self._not_fitted()

        X = scale(X, with_mean=not issparse(X))
        return X @ self.components_

    def fit_transform(self, X: _X) -> np.ndarray:
        """
        Fit the MULTISPATI-PCA projection and transform the data.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
            Array of observations x features.

        Returns
        -------
        numpy.ndarray
        """
        self.fit(X)
        return self.transform(X)

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
        ValueError
            If instance has not been fitted.
        """
        if not self._fitted:
            self._not_fitted()
        return self._spatial_lag(self.transform(X))

    def _spatial_lag(self, X: np.ndarray) -> np.ndarray:
        return self.W @ X

    def variance_moransI_decomposition(self, X: _X) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the decomposition of the variance and Moran's I for `n_components`.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
            Array of observations x features.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Variance and Moran's I.

        Raises
        ------
        ValueError
            If instance has not been fitted.
        """
        if not self._fitted:
            self._not_fitted()
        transformed = self.transform(X)
        lag = self._spatial_lag(transformed)

        # vector of row_Weights from dudi.PCA
        # (we only use default row_weights i.e. 1/n anyways)
        w = 1 / X.shape[0]

        variance = np.sum(transformed * transformed * w, axis=0)
        moran = np.sum(transformed * lag * w, axis=0) / variance

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
        def double_center(W):
            if issparse(W):
                W = W.toarray()

            row_means = np.mean(W, axis=1, keepdims=True)
            col_means = np.mean(W, axis=0, keepdims=True) - np.mean(row_means)

            return W - row_means - col_means

        # ensure symmetry
        W = 0.5 * (self.W + self.W.T)

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

    def _not_fitted(self):
        raise ValueError(
            "This MultispatiPCA instance is not fitted yet. "
            "Call 'fit' with appropriate arguments first."
        )
