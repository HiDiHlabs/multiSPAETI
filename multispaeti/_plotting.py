import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ._multispati_pca import MultispatiPCA


def plot_eigenvalues(msPCA: MultispatiPCA, *, n_top: int | None = None) -> Figure:
    """
    Plot the eigenvalues of the MULTISPATI-PCA.

    Parameters
    ----------
    msPCA : MultispatiPCA
    n_top : int, optional
        Plot the `n_top` highest and `n_top` lowest eigenvalues in a zoomed in view.

    Returns
    -------
    matplotlib.figure.Figure
    """
    eigenvalues = msPCA.eigenvalues_

    x_lbl, y_lbl = "Component", "Eigenvalue"
    n = len(eigenvalues)

    if n_top is None:
        fig, ax = plt.subplots()
        ax.bar(range(1, n + 1), eigenvalues, width=1)
        ax.set(xlabel=x_lbl, ylabel=y_lbl)

    else:
        fig = plt.figure(layout="constrained")
        gs = GridSpec(2, 2, figure=fig)

        ax_all = fig.add_subplot(gs[0, :])
        ax_high = fig.add_subplot(gs[1, 0], sharey=ax_all)
        ax_low = fig.add_subplot(gs[1, 1], sharey=ax_all)

        ax_all.bar(np.arange(1, n + 1), eigenvalues, width=1)
        ax_high.bar(np.arange(1, n_top + 1), eigenvalues[:n_top], width=1)
        ax_low.bar(np.arange(n - n_top + 1, n + 1), eigenvalues[-n_top:], width=1)

        ax_all.set(xlabel=x_lbl, ylabel=y_lbl)
        ax_high.set(xlabel=x_lbl, ylabel=y_lbl)
        ax_low.set(xlabel=x_lbl)

        plt.setp(ax_low.get_yticklabels(), visible=False)

    return fig


def plot_variance_moransI_decomposition(
    msPCA: MultispatiPCA, X, *, sparse_approx: bool = True, **kwargs
) -> Figure:
    """
    Plot the decomposition of variance and Moran's I of the MULTISPATI-PCA eigenvalues.

    The bounds of Moran's I and the expected value for uncorrelated data are indicated
    as well.

    Parameters
    ----------
    msPCA : multispaeti.MultispatiPCA
    X : numpy.ndarray or scipy.sparse.csr_array or scipy.sparse.csc_array
        TODO Data to calculate the decomposition for.
    sparse_approx : bool
        Whether to use a sparse approximation to calculate the decomposition.

    Returns
    -------
    matplotlib.figure.Figure
    """

    variance, moranI = msPCA.variance_moranI_decomposition(X)
    I_min, I_max, I_0 = msPCA.moransI_bounds(sparse_approx=sparse_approx)

    fig, ax = plt.subplots(1)
    _ = ax.scatter(x=variance, y=moranI, **kwargs)

    plt.axhline(y=I_0, ls="--")
    plt.axhline(y=I_min, ls="--")
    plt.axhline(y=I_max, ls="--")

    _ = ax.set_xlim(0, None)
    _ = ax.set(xlabel="Variance", ylabel="Moran's I")

    return fig
