"""Shared fixtures + V&V auto-tagging for ORPHEUS tests.

Every pytest collection pass populates :data:`tests._harness.registry.TEST_REGISTRY`
with one :class:`~tests._harness.registry.TestMetadata` entry per item.
Each entry carries the resolved V&V level, the source of that resolution
(``"explicit"`` / ``"verify"`` / ``"class-name"`` / ``"func-name"`` /
``"case"`` / ``"unmarked"``), the Sphinx equation labels the test
verifies, the failure-mode / error-catalog tags it catches, and the
``VerificationCase`` names it parametrizes over.

The audit CLI (``python -m tests._harness.audit``) and a future
Sphinx verification-matrix generator read the registry directly.

See ``docs/testing/architecture.rst`` for the design rationale and
the full contributor guide.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from orpheus.derivations.reference_values import get as get_reference
from tests._harness import registry
from tests._harness.registry import TestMetadata

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ref():
    """Access analytical reference values by name.

    Usage in tests::

        def test_something(ref):
            case = ref("homo_1eg")
            assert abs(result.k_inf - case.k_inf) < 1e-10
    """
    return get_reference


# ---------------------------------------------------------------------------
# V&V marker auto-tagging + registry population
# ---------------------------------------------------------------------------

_LEVEL_CLASS_RE = re.compile(r"TestL([0-3])")
_LEVEL_FUNC_RE = re.compile(r"(?:^|_)l([0-3])(?:_|$)")
_LEVEL_MARKERS = ("l0", "l1", "l2", "l3")


def _existing_level(item: pytest.Item) -> tuple[str | None, bool]:
    """Return (level_str, was_already_marked).

    Inspects already-applied markers, including file-level ``pytestmark``
    and class-level markers inherited from `@verify.lN(...)`. If two
    different L<N> markers are present, the numerically highest wins
    (matching pytest's precedence for stacked markers) and a warning is
    emitted so duplicate tagging surfaces early.
    """
    present = [m.name for m in item.iter_markers() if m.name in _LEVEL_MARKERS]
    if not present:
        return None, False
    if len(set(present)) > 1:
        # Surface the conflict; choose deterministically.
        chosen = sorted(set(present))[-1]
        item.warn(
            pytest.PytestUnknownMarkWarning(
                f"{item.nodeid} has conflicting V&V level markers "
                f"{sorted(set(present))}; using {chosen!r}"
            )
        )
        return chosen.upper(), True
    return present[0].upper(), True


def _level_from_class_name(item: pytest.Item) -> str | None:
    cls = getattr(item, "cls", None)
    if cls is None:
        return None
    m = _LEVEL_CLASS_RE.search(cls.__name__)
    return f"L{m.group(1)}" if m else None


def _level_from_func_name(item: pytest.Item) -> str | None:
    name = getattr(item, "originalname", None) or item.name
    m = _LEVEL_FUNC_RE.search(name)
    return f"L{m.group(1)}" if m else None


def _case_names_from_parametrize(item: pytest.Item) -> tuple[str, ...]:
    """Extract VerificationCase.name values if the test parametrizes ``case``.

    Relies on pytest's ``callspec`` attribute populated for parametrized
    items. Returns an empty tuple for non-parametrized tests.
    """
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return ()
    case_obj = callspec.params.get("case")
    if case_obj is None:
        return ()
    name = getattr(case_obj, "name", None)
    if isinstance(name, str):
        return (name,)
    return ()


def _level_from_case(item: pytest.Item) -> tuple[str | None, tuple[str, ...]]:
    """Return (level, equation_labels) inherited from a parametrized VerificationCase."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None, ()
    case_obj = callspec.params.get("case")
    if case_obj is None:
        return None, ()
    level = getattr(case_obj, "vv_level", None)
    labels = tuple(getattr(case_obj, "equation_labels", ()) or ())
    return level, labels


def _collect_str_marker_args(item: pytest.Item, marker_name: str) -> tuple[str, ...]:
    """Flatten string args from every ``@pytest.mark.<name>(*args)`` on the item."""
    out: list[str] = []
    for m in item.iter_markers(name=marker_name):
        for arg in m.args:
            if isinstance(arg, str):
                out.append(arg)
            elif isinstance(arg, (list, tuple)):
                out.extend(str(x) for x in arg)
    return tuple(dict.fromkeys(out))  # dedupe preserving order


def _apply_level_marker(item: pytest.Item, level: str) -> None:
    item.add_marker(getattr(pytest.mark, level.lower()))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Resolve V&V level per item and populate ``TEST_REGISTRY``.

    Precedence (most specific wins):

    1. Explicit marker already on the test (``@pytest.mark.lN``,
       file-level ``pytestmark``, or ``@verify.lN(...)`` which stamps
       an explicit marker).
    2. Class name matching ``TestL<N>Foo``.
    3. Function name matching ``test_l<N>_*``.
    4. ``VerificationCase.vv_level`` inherited through a parametrized
       ``case`` argument (requires PR-2 to populate case metadata; in
       PR-1 this branch is inert until cases are tagged).
    5. Unmarked — recorded in the registry with ``level=None`` so the
       audit tool can surface it.
    """
    registry.clear()
    for item in items:
        level, had_explicit = _existing_level(item)
        source: str = "explicit" if had_explicit else "unmarked"

        if level is None:
            cls_level = _level_from_class_name(item)
            if cls_level is not None:
                level, source = cls_level, "class-name"

        if level is None:
            fn_level = _level_from_func_name(item)
            if fn_level is not None:
                level, source = fn_level, "func-name"

        inherited_equations: tuple[str, ...] = ()
        if level is None:
            case_level, case_labels = _level_from_case(item)
            if case_level is not None:
                level, source = case_level, "case"
                inherited_equations = case_labels

        if level is not None and not had_explicit:
            _apply_level_marker(item, level)

        explicit_equations = _collect_str_marker_args(item, "verifies")
        catches = _collect_str_marker_args(item, "catches")
        case_names = _case_names_from_parametrize(item)
        slow = any(m.name == "slow" for m in item.iter_markers())

        equations = tuple(
            dict.fromkeys(explicit_equations + inherited_equations)
        )

        nodeid = item.nodeid
        file_path = nodeid.split("::", 1)[0]

        registry.record(
            TestMetadata(
                nodeid=nodeid,
                file=file_path,
                level=level,  # type: ignore[arg-type]
                level_source=source,  # type: ignore[arg-type]
                equations=equations,
                catches=catches,
                case_names=case_names,
                slow=slow,
            )
        )
