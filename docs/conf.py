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
    "sphinx_design",
    # myst_nb renders both Markdown (it bundles myst_parser) and the tutorial
    # notebooks that CI checks out under docs/tutorials/.
    "myst_nb",
]

# Render the notebooks' *stored* outputs rather than executing them at build
# time: the tutorials need polyscope/libigl and interactive viewers that won't
# run in a headless CI. Re-run a notebook locally and commit its outputs to the
# simkit-tutorials repo to refresh what the website shows.
nb_execution_mode = "off"

# AutoAPI walks the package and turns every docstring into a doc page.
autoapi_type = "python"
autoapi_dirs = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simkit"))]
autoapi_root = "autoapi"
autoapi_keep_files = False
# autoapi generates ``autoapi/index`` and injects its own nav entry. The landing
# page links to it via a card, so we let autoapi own the sidebar entry here.
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
    "cvxopt",
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
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"

html_theme_options = {
    "github_url": "https://github.com/otmanon/simkit",
    "navbar_align": "left",
    "show_prev_next": False,
    "header_links_before_dropdown": 6,
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/simkit/",
            "icon": "fa-brands fa-python",
        },
    ],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
}

html_context = {
    "github_user": "otmanon",
    "github_repo": "simkit",
    "github_version": "main",
    "doc_path": "docs",
}
