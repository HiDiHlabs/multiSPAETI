from typing import TYPE_CHECKING

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ._multispati_pca import MultispatiPCA

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def plot_eigenvalues(msPCA: MultispatiPCA, *, n_top: int | None = None) -> "Figure":
    """
    Plot the eigenvalues of the MULTISPATI-PCA.

    Parameters
    ----------
    msPCA : MultispatiPCA
        An instance of MultispatiPCA that has been fitted so that the eigenvalues
        have been calculated.
    n_top : int, optional
        Plot the `n_top` highest and `n_top` lowest eigenvalues in a zoomed in view.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ModuleNotFoundError
        If `matplotlib` is not installed.
    sklearn.exceptions.NotFittedError
        If the MultispatiPCA has not been fitted.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ModuleNotFoundError as e:
        _raise_matplotlib_load_error(e, "plot_eigenvalues")

    check_is_fitted(msPCA)
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
    msPCA: MultispatiPCA, *, sparse_approx: bool = True, **kwargs
) -> "Figure":
    """
    Plot the decomposition of variance and Moran's I of the MULTISPATI-PCA eigenvalues.

    The bounds of Moran's I and the expected value for uncorrelated data are indicated
    as well.

    Parameters
    ----------
    msPCA : multispaeti.MultispatiPCA
        An instance of MultispatiPCA that has been fitted so that variance and Moran's I
        contributions to the eigenvalues have been calculated.
    sparse_approx : bool
        Whether to use a sparse approximation to calculate the decomposition.
    kwargs
        Other keyword arguments are passed to :py:func:`matplotlib.pyplot.scatter`

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ModuleNotFoundError
        If `matplotlib` is not installed.
    sklearn.exceptions.NotFittedError
        If the MultispatiPCA has not been fitted.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        _raise_matplotlib_load_error(e, "plot_variance_moransI_decomposition")

    check_is_fitted(msPCA)
    I_min, I_max, I_0 = msPCA.moransI_bounds(sparse_approx=sparse_approx)

    fig, ax = plt.subplots(1)
    _ = ax.scatter(x=msPCA.variance_, y=msPCA.moransI_, **kwargs)

    plt.axhline(y=I_0, ls="--")
    plt.axhline(y=I_min, ls="--")
    plt.axhline(y=I_max, ls="--")

    _ = ax.set_xlim(0, None)
    _ = ax.set(xlabel="Variance", ylabel="Moran's I")

    return fig


def _raise_matplotlib_load_error(e: Exception, fn: str):
    raise ModuleNotFoundError(
        f"`{fn}` requires 'matplotlib' to be installed, e.g. via the 'plot' extra."
    ) from e
