"""V&V audit CLI — dump the level × module × equation grid.

Usage::

    python -m tests._harness.audit              # text report
    python -m tests._harness.audit --json       # machine-readable
    python -m tests._harness.audit --untagged   # list only unmarked tests
    python -m tests._harness.audit --gaps       # equations with no coverage

The tool runs ``pytest --collect-only`` under the hood so
:data:`tests._harness.registry.TEST_REGISTRY` is populated, then queries
the registry. No test code is executed.

Exit codes:
    0  clean run
    1  --strict was passed and the report has gaps or untagged tests
    2  collection failed
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest

from tests._harness import registry
from tests._harness.registry import TestMetadata


def _run_collection() -> int:
    """Invoke pytest in collect-only mode so TEST_REGISTRY gets populated.

    Pytest's own stdout (the flat list of 497 nodeids) is discarded so
    the audit report is the only thing the user sees. If pytest fails,
    its stderr is still routed to the real stream.
    """
    devnull = open(os.devnull, "w")  # noqa: SIM115 — must stay open for duration
    try:
        with contextlib.redirect_stdout(devnull):
            return pytest.main(
                [
                    "--collect-only",
                    "-q",
                    "--no-header",
                    "--disable-warnings",
                    "-p",
                    "no:cacheprovider",
                ]
            )
    finally:
        devnull.close()


def _module_of(file_path: str) -> str:
    """Return the pytest test-file's display name for the module grid.

    The nested tests/ layout (issue #77) groups files by solver
    module, so multiple files share a basename (e.g. both
    ``tests/cp/test_properties.py`` and ``tests/sn/test_properties.py``
    have stem ``test_properties``). Collapsing them to the bare stem
    hides the per-module breakdown, so the grid uses the
    ``parent/stem`` form — e.g. ``cp/test_properties`` — when the
    file lives below a subfolder of ``tests/``. Files at the
    ``tests/`` root (``test_pending_ports``,
    ``test_vv_harness_audit``, ``test_convergence``) keep their
    bare stem since there is no parent folder to disambiguate.
    """
    p = Path(file_path)
    parent = p.parent.name
    if parent and parent != "tests":
        return f"{parent}/{p.stem}"
    return p.stem


def _group_by_module_level(
    items: list[TestMetadata],
) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for m in items:
        out[_module_of(m.file)][m.level or "unmarked"] += 1
    return out


def _equation_coverage(items: list[TestMetadata]) -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = defaultdict(list)
    for m in items:
        for eq in m.equations:
            coverage[eq].append(m.nodeid)
    return coverage


def _caught_tags(items: list[TestMetadata]) -> dict[str, list[str]]:
    caught: dict[str, list[str]] = defaultdict(list)
    for m in items:
        for tag in m.catches:
            caught[tag].append(m.nodeid)
    return caught


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------


def _render_text(
    items: list[TestMetadata],
    *,
    theory_labels: set[str],
    documented_labels: set[str],
    err_tags: set[str],
) -> str:
    lines: list[str] = []
    total = len(items)
    level_totals = Counter(m.level or "unmarked" for m in items)
    source_totals = Counter(m.level_source for m in items)

    lines.append("=" * 72)
    lines.append("ORPHEUS V&V Test Audit")
    lines.append("=" * 72)
    lines.append(f"Total tests collected: {total}")
    lines.append("")
    lines.append("By V&V level:")
    # L0..L3 are the physics-verification ladder. foundation is the
    # orthogonal software-invariant bucket; it is reported here for
    # visibility but is not part of the L0..L3 progression. "unmarked"
    # is a gap that the --strict gate surfaces.
    for lvl in ("L0", "L1", "L2", "L3", "foundation", "unmarked"):
        count = level_totals.get(lvl, 0)
        pct = 100 * count / total if total else 0
        lines.append(f"  {lvl:11} {count:5}   ({pct:4.1f}%)")
    lines.append("")
    lines.append("By tagging source:")
    for src in (
        "explicit",
        "verify",
        "class-name",
        "func-name",
        "case",
        "unmarked",
    ):
        count = source_totals.get(src, 0)
        lines.append(f"  {src:12} {count:5}")
    lines.append("")

    # Module × level grid. ``FD`` column counts foundation-marker tests
    # (software invariants, orthogonal to the physics ladder).
    grid = _group_by_module_level(items)
    lines.append("Module × level grid:")
    header = (
        f"  {'module':<36} "
        f"{'L0':>4} {'L1':>4} {'L2':>4} {'L3':>4} {'FD':>4} {'??':>4}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for module in sorted(grid):
        row = grid[module]
        lines.append(
            f"  {module:<36} "
            f"{row.get('L0', 0):>4} "
            f"{row.get('L1', 0):>4} "
            f"{row.get('L2', 0):>4} "
            f"{row.get('L3', 0):>4} "
            f"{row.get('foundation', 0):>4} "
            f"{row.get('unmarked', 0):>4}"
        )
    lines.append("")

    # Equation coverage
    coverage = _equation_coverage(items)
    lines.append("Equation coverage:")
    if not coverage:
        lines.append("  (no tests declare @pytest.mark.verifies yet)")
    else:
        for eq in sorted(coverage):
            lines.append(f"  {eq:40} {len(coverage[eq]):>3} test(s)")
    lines.append("")

    # Orphan equations (declared in theory pages, never referenced by
    # any test) — excluding labels explicitly marked `.. vv-status: X
    # documented` as definitional or not-yet-implemented.
    testable_labels = theory_labels - documented_labels
    orphans = sorted(testable_labels - coverage.keys())
    if theory_labels:
        lines.append(
            f"Orphan equations ({len(orphans)} of {len(testable_labels)} "
            "testable theory labels have zero test coverage; "
            f"{len(documented_labels)} labels are :vv-status: documented "
            "and excluded from the orphan gate):"
        )
        for eq in orphans:
            lines.append(f"  {eq}")
        lines.append("")

    # ERR catalog cross-check
    caught = _caught_tags(items)
    if err_tags:
        missing = sorted(err_tags - caught.keys())
        lines.append(
            f"l0_error_catalog.md ERR coverage "
            f"({len(err_tags) - len(missing)}/{len(err_tags)} entries have a "
            "catching test):"
        )
        for err in missing:
            lines.append(f"  MISSING {err}")
        lines.append("")

    return "\n".join(lines)


def _render_json(
    items: list[TestMetadata],
    *,
    theory_labels: set[str],
    documented_labels: set[str],
    err_tags: set[str],
) -> str:
    coverage = _equation_coverage(items)
    caught = _caught_tags(items)
    testable_labels = theory_labels - documented_labels
    payload: dict[str, Any] = {
        "total": len(items),
        "by_level": dict(Counter(m.level or "unmarked" for m in items)),
        "by_source": dict(Counter(m.level_source for m in items)),
        "grid": {
            module: dict(counts)
            for module, counts in _group_by_module_level(items).items()
        },
        "equation_coverage": {eq: tests for eq, tests in coverage.items()},
        "orphan_equations": sorted(testable_labels - coverage.keys()),
        "documented_equations": sorted(documented_labels),
        "err_coverage": {err: caught.get(err, []) for err in sorted(err_tags)},
        "untagged": [m.nodeid for m in items if m.level is None],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# External inputs: theory equation labels + ERR catalog
# ---------------------------------------------------------------------------


def _scan_theory_equations(theory_dir: Path) -> tuple[set[str], set[str]]:
    """Scan theory RST pages for equation labels and documented-only markers.

    Returns
    -------
    (all_labels, documented_labels)
        ``all_labels`` is every ``.. math:: :label: foo`` found in
        ``docs/theory/*.rst``. ``documented_labels`` is the subset of
        those labels that carry the V&V-harness-specific sentinel

            .. vv-status: <label> documented

        anywhere in the same file. This is a plain RST comment — the
        ``.. `` prefix followed by text that is not a known directive
        is silently stripped by Sphinx, so the sentinel has no effect
        on the rendered theory page. The audit tool uses it to exclude
        three kinds of label from the orphan-equation gate:

        1. **Pure definitional labels** — e.g. ``boltzmann``,
           ``transport-equation``, ``balance-general``. These name the
           governing equation or a mathematical identity that has no
           single "implementing function" to test against. They
           belong in the theory page for the narrative but cannot be
           paired with a verifying test.
        2. **Not-yet-implemented modules** — e.g. the 19 TH / FB / RK
           equations whose Python ports do not exist yet. A real
           orphan (implemented but untested) is a V&V gap; a
           documented-but-not-implemented equation is a work-in-
           progress marker, not a gap.
        3. **Equations with a pending catching test** — when an
           author deliberately wants to defer writing a test and
           surface it as "acceptable gap" rather than "bug in audit",
           marking ``documented`` is the escape hatch. This should
           be rare and paired with a GitHub issue.

        The orphan gate (and ``--strict``) only fires for labels that
        are *not* in ``documented_labels``.
    """
    import re

    if not theory_dir.is_dir():
        return set(), set()

    labels: set[str] = set()
    documented: set[str] = set()
    status_re = re.compile(r"^\.\.\s+vv-status:\s+(\S+)\s+documented\s*$")

    for rst in theory_dir.rglob("*.rst"):
        try:
            text = rst.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(":label:"):
                labels.add(stripped.split(":", 2)[2].strip())
                continue
            m = status_re.match(stripped)
            if m:
                documented.add(m.group(1))

    # A label may only be "documented" if it also exists as a real
    # `:label:`. If someone writes `.. vv-status: foo documented` but
    # `foo` is not actually a theory label, silently drop it — the
    # sentinel has no effect and the typo is caught on the next audit.
    documented &= labels
    return labels, documented


def _scan_err_catalog(catalog: Path) -> set[str]:
    """Extract ``ERR-NNN`` IDs from the L0 error catalog markdown."""
    if not catalog.is_file():
        return set()
    import re

    text = catalog.read_text(encoding="utf-8")
    return set(re.findall(r"\bERR-\d{3}\b", text))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests._harness.audit",
        description="ORPHEUS V&V test audit",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--untagged",
        action="store_true",
        help="list only tests with no V&V level",
    )
    parser.add_argument(
        "--gaps",
        action="store_true",
        help="list orphan equations + missing ERR catchers",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if untagged tests or orphan equations are present",
    )
    parser.add_argument(
        "--theory-dir",
        type=Path,
        default=Path("docs/theory"),
        help="Sphinx theory directory (for orphan-equation scan)",
    )
    parser.add_argument(
        "--err-catalog",
        type=Path,
        default=Path("tests/l0_error_catalog.md"),
        help="L0 error catalog markdown (for ERR coverage)",
    )
    args = parser.parse_args(argv)

    rc = _run_collection()
    if rc != 0 and rc != 5:  # 5 == pytest "no tests"
        print(f"pytest collection failed (exit {rc})", file=sys.stderr)
        return 2

    items = sorted(registry.TEST_REGISTRY.values(), key=lambda m: m.nodeid)
    theory_labels, documented_labels = _scan_theory_equations(args.theory_dir)
    testable_labels = theory_labels - documented_labels
    err_tags = _scan_err_catalog(args.err_catalog)

    if args.untagged:
        for m in items:
            if m.level is None:
                print(m.nodeid)
    elif args.gaps:
        coverage = _equation_coverage(items)
        caught = _caught_tags(items)
        orphans = sorted(testable_labels - coverage.keys())
        missing_err = sorted(err_tags - caught.keys())
        if orphans:
            print("# Orphan equations (no verifying tests)")
            for eq in orphans:
                print(eq)
        if missing_err:
            if orphans:
                print()
            print("# ERR entries with no catching test")
            for err in missing_err:
                print(err)
    elif args.json:
        print(
            _render_json(
                items,
                theory_labels=theory_labels,
                documented_labels=documented_labels,
                err_tags=err_tags,
            )
        )
    else:
        print(
            _render_text(
                items,
                theory_labels=theory_labels,
                documented_labels=documented_labels,
                err_tags=err_tags,
            )
        )

    if args.strict:
        untagged = sum(1 for m in items if m.level is None)
        coverage = _equation_coverage(items)
        orphans = testable_labels - coverage.keys()
        if untagged or orphans:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
