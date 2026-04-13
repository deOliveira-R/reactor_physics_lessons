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
    """Return the pytest test-file's short module name, e.g. ``test_cp_slab``."""
    return Path(file_path).stem


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
    for lvl in ("L0", "L1", "L2", "L3", "unmarked"):
        count = level_totals.get(lvl, 0)
        pct = 100 * count / total if total else 0
        lines.append(f"  {lvl:9}  {count:5}   ({pct:4.1f}%)")
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

    # Module × level grid
    grid = _group_by_module_level(items)
    lines.append("Module × level grid:")
    header = f"  {'module':<36} {'L0':>4} {'L1':>4} {'L2':>4} {'L3':>4} {'??':>4}"
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

    # Orphan equations (declared in theory pages, never referenced by any test)
    orphans = sorted(theory_labels - coverage.keys())
    if theory_labels:
        lines.append(
            f"Orphan equations ({len(orphans)} of {len(theory_labels)} "
            "theory labels have zero test coverage):"
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
    err_tags: set[str],
) -> str:
    coverage = _equation_coverage(items)
    caught = _caught_tags(items)
    payload: dict[str, Any] = {
        "total": len(items),
        "by_level": dict(Counter(m.level or "unmarked" for m in items)),
        "by_source": dict(Counter(m.level_source for m in items)),
        "grid": {
            module: dict(counts)
            for module, counts in _group_by_module_level(items).items()
        },
        "equation_coverage": {eq: tests for eq, tests in coverage.items()},
        "orphan_equations": sorted(theory_labels - coverage.keys()),
        "err_coverage": {err: caught.get(err, []) for err in sorted(err_tags)},
        "untagged": [m.nodeid for m in items if m.level is None],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# External inputs: theory equation labels + ERR catalog
# ---------------------------------------------------------------------------


def _scan_theory_equation_labels(theory_dir: Path) -> set[str]:
    """Grep ``.. math:: :label: foo`` blocks out of theory RST pages."""
    if not theory_dir.is_dir():
        return set()
    labels: set[str] = set()
    for rst in theory_dir.rglob("*.rst"):
        try:
            text = rst.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(":label:"):
                labels.add(stripped.split(":", 2)[2].strip())
    return labels


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
    theory_labels = _scan_theory_equation_labels(args.theory_dir)
    err_tags = _scan_err_catalog(args.err_catalog)

    if args.untagged:
        for m in items:
            if m.level is None:
                print(m.nodeid)
    elif args.gaps:
        coverage = _equation_coverage(items)
        caught = _caught_tags(items)
        orphans = sorted(theory_labels - coverage.keys())
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
            _render_json(items, theory_labels=theory_labels, err_tags=err_tags)
        )
    else:
        print(
            _render_text(items, theory_labels=theory_labels, err_tags=err_tags)
        )

    if args.strict:
        untagged = sum(1 for m in items if m.level is None)
        coverage = _equation_coverage(items)
        orphans = theory_labels - coverage.keys()
        if untagged or orphans:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
