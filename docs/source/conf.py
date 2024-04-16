import importlib.metadata
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "multiSPAETI"
copyright = f"""
{datetime.now():%Y}, Niklas Müller-Bötticher, Naveed Ishaque, Roland Eils,
Berlin Institute of Health @ Charité"""
author = "Niklas Müller-Bötticher"
version = importlib.metadata.version("multispaeti")
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.coverage",
]


autodoc_typehints = "none"
autodoc_typehints_format = "short"
autoclass_content = "class"
autodoc_member_order = "groupwise"

python_use_unqualified_type_names = True  # still experimental

autosummary_generate = True
autosummary_imported_members = True

nitpicky = True
nitpick_ignore = [("py:class", "optional")]


templates_path = ["_templates"]
exclude_patterns: list[str] = []

intersphinx_mapping = dict(
    matplotlib=("https://matplotlib.org/stable/", None),
    numpy=("https://numpy.org/doc/stable/", None),
    python=("https://docs.python.org/3", None),
    scipy=("https://docs.scipy.org/doc/scipy/", None),
    sklearn=("https://scikit-learn.org/stable/", None),
)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
