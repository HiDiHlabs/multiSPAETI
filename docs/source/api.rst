API
===


The API for :py:class:`multispaeti.MultispatiPCA` follows the design in
`scikit-learn <https://scikit-learn.org/>`_. It inherits from
:py:class:`sklearn.base.BaseEstimator`, :py:class:`sklearn.base.TransformerMixin`, and
:py:class:`sklearn.base.ClassNamePrefixFeaturesOutMixin` making it fully compatible
with `scikit-learn` pipelines.


.. currentmodule:: multispaeti

.. autosummary::
   :nosignatures:
   :toctree: ./generated/
   :template: class.rst

   MultispatiPCA
   multispati_pca
   plot_eigenvalues
   plot_variance_moransI_decomposition
