import warnings
from typing import TYPE_CHECKING, Self, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, eye
from scipy.sparse import linalg as sp_linalg
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
    from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
    from cupyx.scipy.sparse import linalg as csp_linalg


_Csr: TypeAlias = csr_array | csr_matrix
_Csc: TypeAlias = csc_array | csc_matrix
_X: TypeAlias = Union[
    np.ndarray, "cp.ndarray", _Csr, _Csc, "cp_csr_matrix", "cp_csc_matrix"
]
# TODO: support cupy arrays for connectivity?
_Connectivity: TypeAlias = np.ndarray | _Csr

# TODO: enable use of cupy.random.RandomState?


# adapted from scikit-learn
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_arpack.py
def _init_arpack_v0(
    size: int, random_state: int | np.random.RandomState | None
) -> NDArray[np.float64]:
    """
    Initialize the starting vector for iteration in ARPACK functions.

    Initialize a ndarray with values sampled from the uniform distribution on
    [-1, 1]. This initialization model has been chosen to be consistent with
    the ARPACK one as another initialization can lead to convergence issues.

    Parameters
    ----------
    size : int
        The size of the eigenvalue vector to be initialized.

    random_state : int, RandomState instance or None
        The seed of the pseudo random number generator used to generate a
        uniform distribution. If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    v0 : ndarray of shape (size,)
        The initialized vector.
    """
    random_state = check_random_state(random_state)
    assert isinstance(random_state, np.random.RandomState)
    v0 = random_state.uniform(-1, 1, size)
    return v0


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
        Matrix of row-wise neighbor definitions i.e. c\\ :sub:`ij` is the connectivity of
        i :math:`\\to` j. The matrix does not have to be symmetric. It can be a
        binary adjacency matrix or a matrix of connectivities in which case
        c\\ :sub:`ij` should be larger if i and j are close.
        A distance matrix should be transformed to connectivities by e.g.
        calculating :math:`1-d/d_{max}` beforehand.
    center_sparse : bool
        Whether to center `X` if it is a sparse array. By default sparse `X` will not be
        centered as this requires transforming it to a dense array, potentially raising
        out-of-memory errors.
    use_gpu : bool
        Whether to use GPU implementation based on `cupy` and `cupyx.scipy` (not installed
        by default).
        Eigendecomposition using the GPU is not as mature yet. For sparse arrays instead of
        `min(n, d)-1` only `min(n, d)-3` eigenvalues/-vectors can be calculated (which
        in most cases won't be a problem). For dense arrays all eigenvalues have to be calculated
        and subsequently subsetted.
    random_state : int | numpy.random.RandomState | None
        Used when the `X` is sparse and `center_sparse` is False and for Moran's I
        bound estimation.
        Pass an int for reproducible results across multiple function calls.

    Attributes
    ----------
    components_ : numpy.ndarray | cupy.ndarray
        The estimated components: Array of shape `(n_components, n_features)`.

    eigenvalues_ : numpy.ndarray
        The eigenvalues corresponding to each of the selected components. Array of shape
        `(n_components,)`.

    variance_ : numpy.ndarray
        The estimated variance part of the eigenvalues. Array of shape `(n_components,)`.

    moransI_ : numpy.ndarray
        The estimated Moran's I part of the eigenvalues. Array of shape `(n_components,)`.

    mean_ : numpy.ndarray | cupy.ndarray | None
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
        center_sparse: bool = False,
        use_gpu: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.n_components = n_components
        self.connectivity = connectivity
        self.center_sparse = center_sparse
        self.use_gpu = use_gpu
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

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
                self._n_components = (self.n_components, 0)
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

    @staticmethod
    def _import_cupy():
        try:
            global cp
            global cp_sparse
            global csp_linalg
            import cupy as cp
            from cupyx.scipy import sparse as cp_sparse
            from cupyx.scipy.sparse import linalg as csp_linalg

        except ImportError as e:
            raise ImportError(
                "GPU implementation requires `cupy` and `cupyx.scipy`."
            ) from e

    @staticmethod
    def _ensure_gpu_matrix(X):
        if sparse.issparse(X):
            X = cp_sparse.csr_matrix(X)
        elif isinstance(X, (np.ndarray, np.matrix)):
            X = cp.array(X)
        return X

    def fit(self, X: _X, y: None = None) -> Self:
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

    def _fit(
        self, X: _X, *, return_transform: bool = False, stats: bool = True
    ) -> "np.ndarray | cp.ndarray | None":
        if self.use_gpu:
            self._import_cupy()
            X = self._ensure_gpu_matrix(X)
        else:
            X = validate_data(self, X, reset=False, accept_sparse=["csr", "csc"])

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
        self.W_ = self.W_ if not self.use_gpu else cp_sparse.csr_matrix(self.W_)

        is_sparse = sparse.issparse if not self.use_gpu else cp_sparse.issparse

        if self.center_sparse and is_sparse(X):
            X = X.toarray()

        if is_sparse(X):
            self.mean_ = None
            X_centered = X
        else:
            self.mean_ = X.mean(axis=0)
            X_centered = X - self.mean_

        eig_val, eig_vec = self._multispati_eigendecomposition(X_centered, self.W_)

        self.components_ = eig_vec
        self.eigenvalues_ = eig_val.get() if self.use_gpu else eig_val  # type: ignore
        self.n_components_ = eig_val.size
        self._n_features_out = self.n_components_
        self.n_features_in_ = d

        if stats:
            X_tr = X_centered @ self.components_.T
            self.variance_, self.moransI_ = self._variance_moransI_decomposition(X_tr)

        if return_transform:
            return X_tr

    def _multispati_eigendecomposition(
        self, X: _X, W: _Connectivity
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple["cp.ndarray", "cp.ndarray"]
    ):
        # X: observations x features
        # W: row-wise definition of neighbors, row-sums should be 1
        def remove_zero_eigenvalues(eigen_values, eigen_vectors, n: int):
            keep_idx = xp.sort(xp.argpartition(xp.abs(eigen_values), -n)[-n:])

            return eigen_values[keep_idx], eigen_vectors[:, keep_idx]

        xp = np if not self.use_gpu else cp
        is_sparse = sparse.issparse if not self.use_gpu else cp_sparse.issparse
        eigsh = sp_linalg.eigsh if not self.use_gpu else csp_linalg.eigsh

        n, d = X.shape

        H = (X.T @ (W + W.T) @ X) / (2 * n)
        # TODO handle sparse based on density?
        # both scipy and cupy sparse arrays
        if is_sparse(H):
            v0 = _init_arpack_v0(H.shape[0], self.random_state)
            if self.use_gpu:
                v0 = cp.asarray(v0)
            match self._n_components:
                case None:
                    k = min(n, d) - 1
                    if self.use_gpu:
                        k -= 2  # TODO: https://github.com/cupy/cupy/issues/6863
                    eig_val, eig_vec = eigsh(H, k=k, which="LM", v0=v0)
                case (n_pos, 0):
                    eig_val, eig_vec = eigsh(H, k=n_pos, which="LA", v0=v0)
                case (0, n_neg):
                    eig_val, eig_vec = eigsh(H, k=n_neg, which="SA", v0=v0)
                case (n_pos, n_neg):
                    eig_val_hi, eig_vec_hi = eigsh(H, k=n_pos, which="LA", v0=v0)
                    eig_val_lo, eig_vec_lo = eigsh(H, k=n_neg, which="SA", v0=v0)

                    eig_val = xp.concatenate([eig_val_lo, eig_val_hi])
                    eig_vec = xp.concatenate([eig_vec_lo, eig_vec_hi], axis=1)

        # numpy.ndarray
        elif xp.__name__ != "cupy":
            match self._n_components:
                case None:
                    eig_val, eig_vec = linalg.eigh(H)
                    if n < d:
                        eig_val, eig_vec = remove_zero_eigenvalues(eig_val, eig_vec, n)
                case (n_pos, 0):
                    eig_val, eig_vec = linalg.eigh(
                        H, subset_by_index=[d - n_pos, d - 1]
                    )
                case (0, n_neg):
                    eig_val, eig_vec = linalg.eigh(H, subset_by_index=[0, n_neg - 1])
                case (n_pos, n_neg):
                    eig_val, eig_vec = linalg.eigh(H)
                    component_indices = self._get_component_indices(d, n_pos, n_neg)
                    eig_val = eig_val[component_indices]
                    eig_vec = eig_vec[:, component_indices]
        # cupy.ndarray
        else:
            # TODO: improve if https://github.com/cupy/cupy/issues/7901 is implemented
            eig_val, eig_vec = xp.linalg.eigh(H)
            match self._n_components:
                case None:
                    # no subsetting required, matching only for completeness
                    pass
                case (n_pos, n_neg):
                    component_indices = self._get_component_indices(d, n_pos, n_neg)
                    eig_val = eig_val[component_indices]
                    eig_vec = eig_vec[:, component_indices]

        return xp.flip(eig_val), xp.flipud(eig_vec.T)

    @staticmethod
    def _get_component_indices(n: int, n_pos: int, n_neg: int) -> list[int]:
        if n_pos + n_neg > n:
            return list(range(n))
        else:
            return list(range(n_neg)) + list(range(n - n_pos, n))

    def transform(self, X: _X) -> "np.ndarray | cp.ndarray":
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

        if self.use_gpu:
            X = self._ensure_gpu_matrix(X)
        else:
            X = validate_data(self, X, reset=False, accept_sparse=["csr", "csc"])

        is_sparse = sparse.issparse if not self.use_gpu else cp_sparse.issparse
        if self.mean_ is not None and not is_sparse(X):
            X = X - self.mean_
        return X @ self.components_.T

    def fit_transform(self, X: _X, y: None = None) -> "np.ndarray | cp.ndarray":
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
        return X_tr

    def transform_spatial_lag(self, X: _X) -> "np.ndarray | cp.ndarray":
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
        # transform performs all necessary input checks
        return self._spatial_lag(self.transform(X))

    def _spatial_lag(self, X: "np.ndarray | cp.ndarray") -> "np.ndarray | cp.ndarray":
        return self.W_ @ X

    def _variance_moransI_decomposition(
        self, X_tr: "np.ndarray | cp.ndarray"
    ) -> tuple[np.ndarray, np.ndarray]:
        lag = self._spatial_lag(X_tr)

        xp = np if not self.use_gpu else cp

        # vector of row_Weights from dudi.PCA (we only use default row_weights i.e. 1/n)
        w = 1 / X_tr.shape[0]

        variance = xp.sum(X_tr * X_tr * w, axis=0)
        moran = xp.sum(X_tr * lag * w, axis=0) / variance

        if xp.__name__ == "cupy":
            variance = variance.get()
            moran = moran.get()

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
        def double_center(
            W: "np.ndarray | cp.ndarray | csr_array | cp_csr_matrix",
        ) -> "np.ndarray | cp.ndarray":
            if is_sparse(W):
                W = W.toarray()  # type: ignore

            row_means = W.mean(axis=1, keepdims=True)
            col_means = W.mean(axis=0, keepdims=True) - row_means.mean()

            return W - row_means - col_means

        # TODO: the correct function is not only dependent on use_gpu but also W_
        is_sparse = sparse.issparse if not self.use_gpu else cp_sparse.issparse
        eigsh = sp_linalg.eigsh if not self.use_gpu else csp_linalg.eigsh

        # ensure symmetry
        W = 0.5 * (self.W_ + self.W_.T)

        n_sample = W.shape[0]
        s = n_sample / W.sum()  # 1 if original W has rowSums or colSums of 1

        if not is_sparse(W) or not sparse_approx:
            W = double_center(W)

        v0 = _init_arpack_v0(W.shape[0], self.random_state)
        if self.use_gpu:
            v0 = cp.asarray(v0)

        I_0 = -1 / (n_sample - 1)
        I_min = float(
            s * eigsh(W, k=1, which="SA", return_eigenvectors=False, v0=v0)[0]
        )
        I_max = float(
            s * eigsh(W, k=1, which="LA", return_eigenvectors=False, v0=v0)[0]
        )

        return I_min, I_max, I_0


def multispati_pca(
    X: _X,
    n_components: int | tuple[int, int] | None = None,
    *,
    connectivity: _Connectivity | None = None,
    center_sparse: bool = False,
    use_gpu: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]] | tuple["cp.ndarray", "cp.ndarray"]
):
    """
    Calculate MULTISPATI-PCA and return the transformed data matrix and components.

    For more information refer to :py:class:`multispaeti.MultispatiPCA`.

    This function is more efficient than :py:meth:`multispaeti.MultispatiPCA.fit_transform`
    if the additional attributes are not needed.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
        Array of observations x features.
    n_components : int or tuple[int, int], optional
        Number of components to keep.
        If None, will keep all components.
        If an int, it will keep the top `n_components`.
        If a tuple, it will keep the top and bottom `n_components`, respectively.
    connectivity : scipy.sparse.sparray or scipy.sparse.spmatrix
        Matrix of row-wise neighbor definitions i.e. c\\ :sub:`ij` is the connectivity of
        i :math:`\\to` j. The matrix does not have to be symmetric. It can be a
        binary adjacency matrix or a matrix of connectivities in which case
        c\\ :sub:`ij` should be larger if i and j are close.
        A distance matrix should be transformed to connectivities by e.g.
        calculating :math:`1-d/d_{max}` beforehand.
    center_sparse : bool
        Whether to center `X` if it is a sparse array. By default sparse `X` will not be
        centered as this requires transforming it to a dense array, potentially raising
        out-of-memory errors.
    use_gpu : bool
        Whether to use GPU implementation based on `cupy` and `cupyx.scipy` (not installed
        by default).
        Eigendecomposition using the GPU is not as mature yet (especially for sparse arrays).
    random_state : int | numpy.random.RandomState | None
        Used when the `X` is sparse and `center_sparse` is False.
        Pass an int for reproducible results across multiple function calls.

    Returns
    -------
        X_transformed : numpy.ndarray or cupy.ndarray
        components : numpy.ndarray or cupy.ndarray
    """
    ms_pca = MultispatiPCA(
        n_components,
        connectivity=connectivity,
        center_sparse=center_sparse,
        use_gpu=use_gpu,
        random_state=random_state,
    )

    X_tr = ms_pca._fit(X, return_transform=True, stats=False)

    return X_tr, ms_pca.components_
