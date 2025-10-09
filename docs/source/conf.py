"""Sphinx configuration for the biomedical-data-generator project."""

from __future__ import annotations

from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version

# --- Project info -------------------------------------------------------------

project = "Biomedical Data Generator"
author = "Sigrun May"
copyright = f"{date.today().year}, {author}"

# Use the installed package version if available; fall back for local builds
try:
    release = pkg_version("biomedical-data-generator")
except PackageNotFoundError:
    release = "0.1.0"

# --- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True

# Docstring style
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True

# Autodoc behavior
autodoc_typehints = "description"
autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    # Keep special/private members off by default; enable in individual pages if needed
    # "special-members": "__call__",
    # "private-members": True,
    "exclude-members": "__weakref__, __init__",
}

# Source file types
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []

# Intersphinx references
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

# --- HTML output --------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_show_sphinx = False

# Optional custom CSS (ensure file exists at docs/_static/css/custom.css)
html_css_files = [
    "css/custom.css",
]

# GitHub integration (used by many RTD templates to render "Edit on GitHub")
html_context = {
    "display_github": True,
    "github_user": "sigrun-may",
    "github_repo": "biomedical-data-generator",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
