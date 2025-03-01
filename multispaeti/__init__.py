from importlib.metadata import PackageNotFoundError, version

from ._multispati_pca import MultispatiPCA, multispati_pca
from ._plotting import plot_eigenvalues, plot_variance_moransI_decomposition

try:
    __version__ = version("multispaeti")
except PackageNotFoundError:
    __version__ = "unknown version"

del PackageNotFoundError, version


__all__ = [
    "MultispatiPCA",
    "multispati_pca",
    "plot_eigenvalues",
    "plot_variance_moransI_decomposition",
]
