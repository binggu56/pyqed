# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import warnings
sys.path.insert(0, os.path.abspath('../..'))

# Keep autodoc imports quiet and deterministic on Read the Docs.
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
warnings.filterwarnings('ignore', category=SyntaxWarning)


# -- Project information -----------------------------------------------------

project = 'PyQED'
copyright = '2022-2026, Bing Gu and PyQED contributors'
author = 'Bing Gu'

# Resolve the documentation version from the package instead of maintaining a
# second, easily stale version string in this file.
try:
    from pyqed import __version__ as release
except (ImportError, AttributeError):
    release = '0+unknown'
version = release


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
]

#.. autoclass:: lime.oqs.LindbladSolver

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    # Old generated API pages for modules that were renamed or moved.
    'Floquet.rst',
    'NAMD_2D.rst',
    'SPO_1D.rst',
    'SPO_1D_NAMD.rst',
    'README.rst',
    'models.rst',
    'namd.rst',
    'modules.rst',
    'pyqed.beam.rst',
    'pyqed.dvr.rst',
    'pyqed.rst',
    'pyqed.pyqed*.rst',
    'pyqed.signal.rst',
]

autodoc_mock_imports = [
    'cv2',
    'gbasis',
    'lime',
    'mayavi',
    'numba',
    'proplot',
    'pyscf',
    'pyqed.qchem',
    'screeninfo',
    'traits',
    'tvtk',
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Consolidate Read the Docs aliases and historical versions around the public
# documentation domain. Sphinx emits a per-page canonical link from this base.
html_baseurl = os.environ.get(
    'READTHEDOCS_CANONICAL_URL',
    'https://docs.pyqed.org/en/latest/',
)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
_static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '_static'))
html_static_path = ['../_static'] if os.path.isdir(_static_path) else []
html_js_files = ['analytics.js'] if html_static_path else []
