"""Sphinx configuration for the SimKit documentation.

Run a local build with::

    pip install -e .[docs]
    sphinx-build -b html docs docs/_build/html

The generated HTML lives in ``docs/_build/html/index.html``. The API reference
under ``docs/_build/html/autoapi/`` is generated automatically from the
docstrings in the ``simkit`` package, so adding/improving a docstring is all
that is needed to update the docs.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the package importable for autoapi without requiring an install.
sys.path.insert(0, os.path.abspath(".."))

project = "SimKit"
author = "Otman Benchekroun"
copyright = f"{datetime.now():%Y}, {author}"

try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("simkit")
except Exception:  # pragma: no cover - fall back when not installed
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "myst_parser",
]

# AutoAPI walks the package and turns every docstring into a doc page.
autoapi_type = "python"
autoapi_dirs = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simkit"))]
autoapi_root = "autoapi"
autoapi_keep_files = False
autoapi_add_toctree_entry = True
autoapi_ignore = [
    "*/old/*",
    "*/__pycache__/*",
]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

# Napoleon: accept both NumPy and Google-style docstrings.
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Don't fail the build because an optional heavy dep can't import at doc time.
autodoc_mock_imports = [
    "igl",
    "libigl",
    "polyscope",
    "cv2",
    "opencv-python",
    "cvxopt",
    "sklearn",
    "scikit-learn",
    "bpy",
    "blendertoolbox",
    "cma",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"{project} {release}"
