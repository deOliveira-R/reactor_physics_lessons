"""Generate RST fragments for Sphinx documentation.

Imports all verification cases from the derivation modules and writes
RST files with LaTeX equations and results tables to docs/_generated/.

Run before sphinx-build:
    python -m derivations.generate_rst
"""

from __future__ import annotations

from pathlib import Path

from .reference_values import all_cases, by_method


OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "_generated"

# Display names for method and geometry codes
_METHOD_LABELS = {
    "homo": "Homogeneous",
    "cp": "Collision Probability",
    "sn": "Discrete Ordinates",
    "moc": "Method of Characteristics",
    "mc": "Monte Carlo",
    "dif": "Diffusion",
}

_GEOM_LABELS = {
    "--": "Homogeneous (0D)",
    "slab": "Slab 1D",
    "cyl1D": "Cylindrical 1D",
    "sph1D": "Spherical 1D",
}

# Sort order for method-grouped display
_METHOD_ORDER = ["homo", "sn", "cp", "moc", "mc", "dif"]


def _write(name: str, content: str) -> None:
    """Write an RST fragment to the output directory."""
    path = OUTPUT_DIR / name
    path.write_text(content)


def _method_fragment(method: str, geometry: str | None = None) -> str:
    """Build an RST fragment for all cases of a given method (and optional geometry)."""
    cases = by_method(method)
    if geometry is not None:
        cases = [c for c in cases if c.geometry == geometry]
    parts = []
    for c in sorted(cases, key=lambda c: c.n_groups):
        parts.append(f"**{c.name}**: {c.description}")
        parts.append("")
        parts.append(c.latex)
        parts.append("")
    return "\n".join(parts) + "\n"


def generate_verification_table() -> None:
    """Master table of all verification cases as a sortable HTML table."""
    def sort_key(c):
        m = _METHOD_ORDER.index(c.method) if c.method in _METHOD_ORDER else 99
        return (m, c.n_groups, c.geometry)

    cases = sorted(all_cases(), key=sort_key)

    rows = []
    for c in cases:
        method_label = _METHOD_LABELS.get(c.method, c.method)
        geom_label = _GEOM_LABELS.get(c.geometry, c.geometry)
        tol_label = c.tolerance or "&mdash;"
        rows.append(
            f"      <tr>"
            f"<td><code>{c.name}</code></td>"
            f"<td>{method_label}</td>"
            f"<td>{geom_label}</td>"
            f"<td>{c.n_groups}</td>"
            f"<td>{c.n_regions}</td>"
            f"<td>{tol_label}</td>"
            f'<td>{c.k_inf:.10f}</td>'
            f"</tr>"
        )

    html = "\n".join([
        '.. raw:: html',
        '',
        '   <table class="sortable docutils align-default">',
        '     <thead>',
        '       <tr>',
        '         <th>Name</th>',
        '         <th>Method</th>',
        '         <th>Geometry</th>',
        '         <th>Groups</th>',
        '         <th>Regions</th>',
        '         <th>Tolerance</th>',
        '         <th><em>k</em><sub>&infin;</sub></th>',
        '       </tr>',
        '     </thead>',
        '     <tbody>',
        *rows,
        '     </tbody>',
        '   </table>',
    ])

    _write("verification_table.rst", html + "\n")


def main() -> None:
    """Generate all RST fragments."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_verification_table()
    _write("homogeneous_derivation.rst", _method_fragment("homo"))
    _write("sn_derivation.rst", _method_fragment("sn"))
    _write("cp_slab_derivation.rst", _method_fragment("cp", "slab"))
    _write("cp_cylinder_derivation.rst", _method_fragment("cp", "cyl1D"))
    _write("moc_derivation.rst", _method_fragment("moc"))
    _write("mc_derivation.rst", _method_fragment("mc"))
    _write("diffusion_derivation.rst", _method_fragment("dif"))

    n_files = len(list(OUTPUT_DIR.glob("*.rst")))
    print(f"Generated {n_files} RST fragments in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
