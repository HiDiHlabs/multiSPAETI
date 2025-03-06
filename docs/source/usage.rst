Usage
=====

A short explanation of MULTISPATI-PCA is provided in the documentation
of :py:class:`multispaeti.MultispatiPCA`. For a more comprehensive understanding
the interested reader is referred to the original publication of
`Dray et al. <https://onlinelibrary.wiley.com/doi/abs/10.3170/2007-8-18312>`_
as well as a related publication by
`Jombart et al. <https://www.nature.com/articles/hdy200834>`_

Briefly, MULTISPATI-PCA tries to maximize the product of variance and Moran's `I` for
each retrieved component. Components with (large) positive eigenvalues are spatially
auto-correlated while (large) negative eigenvalues identify components of negative
auto-correlation.

The general usage follows the patterns in :py:mod:`sklearn.decomposition`.

We first create an instance of :py:class:`multispaeti.MultispatiPCA` specifying a
`connectivity` matrix and optionally the desired number of components to be retrieved.

.. code-block:: python

    from multispaeti import MultispatiPCA

    msPCA = MultispatiPCA(n_components=(30, 5), connectivity=connectivity)

In this case this would retrieve the 30 largest as well as 5 smallest eigenvalues and
their respective eigenvectors.

As for e.g. :py:class:`sklearn.decomposition.PCA` we first need to
:py:meth:`multispaeti.MultispatiPCA.fit` and than
:py:meth:`multispaeti.MultispatiPCA.transform` our data matrix `X` (`n` observations
:math:`\times` `d` features).

.. code-block:: python

    msPCA.fit(X)
    X_transformed = msPCA.transform(X)


Alternatively, this can be achieved in one step by
:py:meth:`multispaeti.MultispatiPCA.fit_transform` which avoids redundant computation.

.. code-block:: python

    X_transformed = msPCA.fit_transform(X)

If only the transformed data matrix and selected components are of interest, the most
efficient is to call :py:func:`multispaeti.multispati_pca` directly as this will avoid
calculating additional statistics such as variance and Moran's I.

.. code-block:: python

    from multispaeti import multispati_pca

    X_transformed, components = multispati_pca(
        X, n_components=(30, 5), connectivity=connectivity
    )

.. Additional, functionality is offered through the method
.. :py:meth:`multispaeti.MultispatiPCA.moransI_bounds` which calculates the minimum and
.. maximum bound as well as the expected value given the `connectivity` matrix

.. .. code-block:: python

..     X_transformed = msPCA.moransI_bounds()
