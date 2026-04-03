# ORPHEUS Documentation — Sphinx Configuration
# ==============================================

import sys
from pathlib import Path

# Add project root and numbered module directories to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
for d in sorted(project_root.iterdir()):
    if d.is_dir() and d.name[0].isdigit():
        sys.path.insert(0, str(d))

# -- Project information -----------------------------------------------

project = 'ORPHEUS'
copyright = '2026, Rodrigo de Oliveira'
author = 'Rodrigo de Oliveira'
release = '0.1'

# -- General configuration ---------------------------------------------

extensions = [
    'sphinx.ext.autodoc',       # Pull docstrings from Python code
    'sphinx.ext.mathjax',       # LaTeX math rendering
    'sphinx.ext.viewcode',      # [source] links to highlighted code
    'sphinx.ext.intersphinx',   # Cross-reference external docs (numpy, scipy)
    'sphinx.ext.napoleon',      # Google/NumPy-style docstrings
    'matplotlib.sphinxext.plot_directive',  # .. plot:: for auto-generated figures
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Options for mathjax -----------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            # Macros that take a subscript argument to avoid double-subscript errors
            'Sigt': [r'\Sigma_{\mathrm{t},#1}', 1, ''],
            'Sigs': [r'\Sigma_{\mathrm{s},#1}', 1, ''],
            'Siga': [r'\Sigma_{\mathrm{a},#1}', 1, ''],
            'Sigf': [r'\Sigma_{\mathrm{f},#1}', 1, ''],
            'nSigf': [r'\nu\Sigma_{\mathrm{f},#1}', 1, ''],
            'keff': r'k_{\mathrm{eff}}',
            'kinf': r'k_{\infty}',
        },
    },
}

# -- Options for plot directive ----------------------------------------

plot_include_source = True
plot_html_show_source_link = False
plot_formats = ['png']
plot_rcparams = {
    'figure.figsize': (8, 5),
    'font.size': 11,
}

# -- Intersphinx mapping -----------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}
