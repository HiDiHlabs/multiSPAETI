Installation
============


PyPI and ``pip``
----------------

To install ``multispaeti`` from `PyPI <https://pypi.org/>`_ using ``pip`` just run

.. code-block:: bash

    pip install multispaeti

GPU support
___________

In most cases support for GPU computations (using `CuPy <https://cupy.dev/>`_) can be
installed via

.. code-block:: bash

    # for CUDA12
    pip install multispaeti[cuda12]

    # or for CUDA11
    pip install multispaeti[cuda11]

However, in cases where this doesn't work we recommended referring to the
`CuPy documentation <https://docs.cupy.dev/en/stable/install.html>`_.


conda-forge and ``conda``
-------------------------

``multispaeti`` can also be installed from `conda-forge <https://conda-forge.org/>`_ via

.. code-block:: bash

    conda install conda-forge::multispaeti

.. note::

    Of course, it is also possible to use ``mamba`` instead of ``conda``
    to speed up the installation.

GPU support
___________

For support of GPU computations please refer to the
`CuPy documentation <https://docs.cupy.dev/en/stable/install.html>`_.


From GitHub
-----------

You can install the latest versions directly from GitHub. To do so
clone the repository using the ``git clone`` command. Navigate into the downloaded
directory and install using

.. code-block:: bash

    pip install -e .

If you want to install the development version you can install the additional optional
dependencies with

.. code-block:: bash

    pip install -e '.[dev]'
