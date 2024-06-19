Installation
============


PyPI and ``pip``
----------------

To install ``multispaeti`` from `PyPI <https://pypi.org/>`_ using ``pip`` just run

.. code-block:: bash

    pip install multispaeti


conda-forge and ``conda``
-------------------------

``multispaeti`` can also be installed from `conda-forge <https://conda-forge.org/>`_ via

.. code-block:: bash

    conda install conda-forge::multispaeti

.. note::

    Of course, it is also possible to use ``mamba`` instead of ``conda``
    to speed up the installation.


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
