"""Generate ``docs/verification/matrix.rst`` from the pytest test registry.

Runs ``python -m tests._harness.audit --json`` under the hood to
populate :data:`tests._harness.registry.TEST_REGISTRY`, then emits a
Sphinx RST page with:

- Overall V&V level distribution (L0/L1/L2/L3/foundation/unmarked counts)
- Per-module level × count grid
- Equation coverage table (label → number of declared tests)
- Orphan equations (`.. math:: :label:` blocks with zero declared tests,
  excluding ``:vv-status: documented`` labels)
- Documented-only labels (excluded from the orphan gate)
- ERR-NNN catalog cross-check (from ``tests/l0_error_catalog.md``)
- Unmarked tests listing

The ``foundation`` bucket is orthogonal to the L0..L3 physics ladder
— foundation tests verify software invariants (data-structure
contracts, numerical primitives, factory outputs) rather than physics
equations. They appear in their own column in the module grid and
never contribute to the equation-coverage or orphan-equation tables.
See ``docs/testing/architecture.rst``:ref:`vv-foundation-tests`.

The page is built every time Sphinx rebuilds (the generator is
invoked as a ``pre-build`` step, see ``docs/conf.py`` and the
harness architecture page). It closes ORPHEUS issue #79
("Sphinx L0 verification page generated from test docstrings").

Usage::

    python -m tools.verification.generate_matrix [OUT_RST]

``OUT_RST`` defaults to ``docs/verification/matrix.rst``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "docs" / "verification" / "matrix.rst"


def _run_audit() -> dict:
    """Run the harness audit and return its JSON payload."""
    result = subprocess.run(
        [sys.executable, "-m", "tests._harness.audit", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _rst_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple list-table for Sphinx."""
    if not rows:
        return "   (empty)\n"
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    sep = "   " + "  ".join("=" * w for w in widths) + "\n"
    out = [sep]
    out.append(
        "   " + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + "\n"
    )
    out.append(sep)
    for row in rows:
        out.append(
            "   "
            + "  ".join(str(c).ljust(w) for c, w in zip(row, widths))
            + "\n"
        )
    out.append(sep)
    return "".join(out)


def _render(payload: dict) -> str:
    total = payload["total"]
    by_level = payload["by_level"]
    by_source = payload["by_source"]
    grid = payload["grid"]
    coverage = payload["equation_coverage"]
    orphans = payload["orphan_equations"]
    # ``documented_equations`` was added by the :vv-status: directive
    # work (Phase B.0 of issue #87). Older audit payloads may not
    # include it, so fall back to an empty list for robustness.
    documented = payload.get("documented_equations", [])
    err_coverage = payload["err_coverage"]
    untagged = payload["untagged"]

    lines: list[str] = []

    # Header
    lines.append("Verification Matrix\n")
    lines.append("===================\n\n")
    lines.append(
        ".. note::\n\n"
        "   Auto-generated from ``tests._harness.registry.TEST_REGISTRY``\n"
        "   by ``tools/verification/generate_matrix.py``. Do not edit by\n"
        "   hand — changes will be overwritten on the next rebuild.\n\n"
    )

    lines.append(f"Total tests collected: **{total}**\n\n")

    # V&V level distribution. ``foundation`` is orthogonal to the
    # L0..L3 ladder and reported alongside it for visibility.
    lines.append("V&V level distribution\n")
    lines.append("----------------------\n\n")
    level_rows = []
    for lvl in ("L0", "L1", "L2", "L3", "foundation", "unmarked"):
        count = by_level.get(lvl, 0)
        pct = f"{100 * count / total:.1f}%" if total else "0.0%"
        level_rows.append([lvl, str(count), pct])
    lines.append(".. csv-table::\n")
    lines.append("   :header: Level, Count, Share\n")
    lines.append("   :widths: 15, 10, 10\n\n")
    for row in level_rows:
        lines.append(f"   {row[0]}, {row[1]}, {row[2]}\n")
    lines.append("\n")

    # Tagging source distribution
    lines.append("Tagging source\n")
    lines.append("--------------\n\n")
    lines.append(
        "How each test acquired its V&V level "
        "(see ``tests/conftest.py`` for the precedence chain).\n\n"
    )
    src_rows = []
    for src in (
        "explicit",
        "verify",
        "class-name",
        "func-name",
        "case",
        "unmarked",
    ):
        count = by_source.get(src, 0)
        src_rows.append([src, str(count)])
    lines.append(".. csv-table::\n")
    lines.append("   :header: Source, Count\n")
    lines.append("   :widths: 20, 10\n\n")
    for row in src_rows:
        lines.append(f"   {row[0]}, {row[1]}\n")
    lines.append("\n")

    # Module × level grid. ``FD`` is the foundation column.
    lines.append("Module × level grid\n")
    lines.append("-------------------\n\n")
    mod_rows = []
    for module in sorted(grid):
        row = grid[module]
        mod_rows.append(
            [
                module,
                str(row.get("L0", 0)),
                str(row.get("L1", 0)),
                str(row.get("L2", 0)),
                str(row.get("L3", 0)),
                str(row.get("foundation", 0)),
                str(row.get("unmarked", 0)),
            ]
        )
    lines.append(".. csv-table::\n")
    lines.append("   :header: Module, L0, L1, L2, L3, FD, ??\n")
    lines.append("   :widths: 40, 6, 6, 6, 6, 6, 6\n\n")
    for row in mod_rows:
        lines.append(f"   {', '.join(row)}\n")
    lines.append("\n")

    # Equation coverage
    lines.append("Equation coverage\n")
    lines.append("-----------------\n\n")
    lines.append(
        "Every Sphinx ``.. math:: :label:`` block declared in "
        "``docs/theory/*.rst`` and the number of tests carrying "
        "``@pytest.mark.verifies(\"label\")`` that reference it.\n\n"
    )
    if coverage:
        lines.append(".. csv-table::\n")
        lines.append("   :header: Equation label, Tests\n")
        lines.append("   :widths: 50, 10\n\n")
        for eq in sorted(coverage, key=lambda e: (-len(coverage[e]), e)):
            lines.append(f"   ``{eq}``, {len(coverage[eq])}\n")
    else:
        lines.append("*(no equations declared)*\n")
    lines.append("\n")

    # Orphan equations
    lines.append("Orphan equations\n")
    lines.append("----------------\n\n")
    lines.append(
        f"Equations with zero tests carrying "
        f"``@pytest.mark.verifies(\"label\")``, excluding labels "
        f"explicitly marked ``:vv-status: documented``. "
        f"**{len(orphans)}** of the testable equations found on "
        f"theory pages are orphan.\n\n"
    )
    if orphans:
        for eq in sorted(orphans):
            lines.append(f"- ``{eq}``\n")
    else:
        lines.append("*(none — every testable theory equation has at "
                     "least one verifying test)*\n")
    lines.append("\n")

    # Documented-only equations (excluded from the orphan gate)
    lines.append("Documented-only equations\n")
    lines.append("-------------------------\n\n")
    lines.append(
        f"Theory labels marked ``.. vv-status: <label> documented`` in "
        f"their RST source. These are excluded from the orphan-equation "
        f"gate because they are either definitional (no single "
        f"implementing function — e.g. ``boltzmann``), describe a "
        f"module whose Python port does not yet exist (e.g. the "
        f"thermal-hydraulics / fuel-behaviour / reactor-kinetics "
        f"equations), or have a deliberately deferred test paired with "
        f"a tracking issue. **{len(documented)}** labels carry the "
        f"directive. See ``docs/testing/architecture.rst``"
        f":ref:`vv-status-documented` for the full taxonomy.\n\n"
    )
    if documented:
        for eq in sorted(documented):
            lines.append(f"- ``{eq}``\n")
    else:
        lines.append("*(none)*\n")
    lines.append("\n")

    # ERR catalog cross-check
    lines.append("L0 error-catalog coverage\n")
    lines.append("-------------------------\n\n")
    lines.append(
        "Every ``ERR-NNN`` entry in ``tests/l0_error_catalog.md`` and "
        "the tests that carry ``@pytest.mark.catches(\"ERR-NNN\")`` "
        "to guard it. A missing catcher is a publication-blocker for "
        "the error catalog.\n\n"
    )
    if err_coverage:
        lines.append(".. csv-table::\n")
        lines.append("   :header: Error tag, Catching tests\n")
        lines.append("   :widths: 15, 10\n\n")
        for err in sorted(err_coverage):
            caught = err_coverage[err]
            status = str(len(caught)) if caught else "**0 (MISSING)**"
            lines.append(f"   ``{err}``, {status}\n")
    else:
        lines.append("*(no ERR entries found)*\n")
    lines.append("\n")

    # Untagged tests summary
    lines.append("Unmarked tests\n")
    lines.append("--------------\n\n")
    untagged_count = len(untagged)
    if untagged_count:
        lines.append(
            f"**{untagged_count} tests** have no V&V level marker.\n"
            "This is a gap — every test in the tree should carry either\n"
            "a physics-ladder marker (``l0``..``l3``) or the orthogonal\n"
            "``foundation`` marker (``@pytest.mark.foundation``) for\n"
            "tests that verify software invariants rather than physics\n"
            "equations. See ``docs/testing/architecture.rst``\n"
            ":ref:`vv-foundation-tests` for the taxonomy.\n\n"
        )
        by_file: Counter[str] = Counter()
        for nodeid in untagged:
            by_file[nodeid.split("::", 1)[0]] += 1
        lines.append(".. csv-table::\n")
        lines.append("   :header: File, Unmarked tests\n")
        lines.append("   :widths: 60, 10\n\n")
        for f, c in by_file.most_common():
            lines.append(f"   ``{f}``, {c}\n")
    else:
        lines.append(
            "*(none — every test carries an L0/L1/L2/L3 or "
            "foundation marker)*\n"
        )
    lines.append("\n")

    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out_path = Path(argv[0]) if argv else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _run_audit()
    rst = _render(payload)
    out_path.write_text(rst, encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO_ROOT)} "
          f"({payload['total']} tests, {len(payload['equation_coverage'])} "
          f"equations covered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
