# ORPHEUS Documentation — Sphinx Configuration
# ==============================================

import sys
from pathlib import Path

# Add project root to Python path (orpheus package lives there)
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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
    'sphinxcontrib.nexus',                  # Knowledge graph extraction
]

templates_path = ['_templates']
exclude_patterns = ['_build', '_generated', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_js_files = ['sortable.js']
html_css_files = ['sortable.css']

# -- Options for autodoc -----------------------------------------------

# -- Options for Nexus knowledge graph ------------------------------------

nexus_extra_source_dirs = ['tests']

# Exclude pedagogical tutorial scripts from the Nexus knowledge graph.
# student_resources/ contains reduced re-implementations whose class
# names intentionally shadow orpheus.* (CPParams, CPResult, TwoGroupXS,
# HomogeneousResult, MCParams, FuelBehaviourResult, KineticsResult).
# Without this exclusion, Sphinx's py-domain xref resolver attaches
# both the orpheus.* class and the tutorial duplicate to every
# api/*.rst page, inflating the graph and triggering false staleness
# flags whenever a tutorial file is touched.
nexus_source_exclude_patterns = ['student_resources/*']

# -- Options for autodoc -----------------------------------------------

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Options for napoleon ---------------------------------------------
#
# Render NumPy-style ``Attributes`` sections as ``:ivar:`` info fields
# rather than standalone ``.. attribute::`` directives. Without this
# flag, dataclass fields are double-documented (once by autodoc and
# once by napoleon), causing "duplicate object description" warnings.

napoleon_use_ivar = True

# -- Auto-generate verification matrix page ---------------------------
#
# Runs `python -m tools.verification.generate_matrix` before Sphinx
# collects sources so `docs/verification/matrix.rst` is always in sync
# with the pytest test registry. Closes ORPHEUS issue #79.

def _regenerate_verification_matrix(app):
    import subprocess
    try:
        subprocess.run(
            [sys.executable, "-m", "tools.verification.generate_matrix"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        app.warn(
            f"verification matrix regeneration failed: {e.stderr}"
        )


# -- Auto-generate Peierls capability matrix --------------------------
#
# Runs `python -m tools.verification.generate_peierls_matrix` before
# Sphinx collects sources so the capability table in
# `docs/theory/peierls_unified.rst` (§theory-peierls-capabilities)
# cannot drift from the registry function
# `orpheus.derivations.peierls_cases.capability_rows()`. Consolidation
# task T2.1.

def _regenerate_peierls_matrix(app):
    import subprocess
    try:
        subprocess.run(
            [sys.executable, "-m", "tools.verification.generate_peierls_matrix"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        app.warn(
            f"peierls capability matrix regeneration failed: {e.stderr}"
        )


def setup(app):
    app.connect("builder-inited", _regenerate_verification_matrix)
    app.connect("builder-inited", _regenerate_peierls_matrix)

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
