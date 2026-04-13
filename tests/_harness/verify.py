"""Ergonomic sugar for V&V markers.

Two surfaces:

``verify.l0(...)``, ``verify.l1(...)``, ``verify.l2(...)``, ``verify.l3(...)``
    Class or function decorator that stamps the level marker plus
    optional ``verifies`` and ``catches`` markers in one call.

``vv_cases(level=..., method=..., geometry=...)``
    ``pytest.mark.parametrize`` replacement that auto-selects matching
    :class:`~orpheus.derivations._types.VerificationCase` entries and
    attaches the inherited level marker.

Both are thin layers over raw ``pytest.mark.*``. Teams that prefer
vanilla pytest markers can skip them entirely — the conftest hook
handles either style transparently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

import pytest

VVLevel = Literal["L0", "L1", "L2", "L3"]


def _stamp(
    obj: Any,
    *,
    level: VVLevel,
    equations: Iterable[str] = (),
    catches: Iterable[str] = (),
    slow: bool = False,
) -> Any:
    """Attach level + optional verifies/catches markers to obj.

    ``obj`` can be a function or a class. Both are handled natively by
    pytest's marker API via ``pytestmark``-style application: applying
    a marker to a class propagates to every collected test method.
    """
    level_marker = getattr(pytest.mark, level.lower())
    obj = level_marker(obj)
    if equations:
        obj = pytest.mark.verifies(*tuple(equations))(obj)
    if catches:
        obj = pytest.mark.catches(*tuple(catches))(obj)
    if slow:
        obj = pytest.mark.slow(obj)
    return obj


def _level_decorator(level: VVLevel) -> Callable[..., Callable[[Any], Any]]:
    def decorator_factory(
        *,
        equations: Iterable[str] = (),
        catches: Iterable[str] = (),
        slow: bool = False,
    ) -> Callable[[Any], Any]:
        def apply(obj: Any) -> Any:
            return _stamp(
                obj,
                level=level,
                equations=equations,
                catches=catches,
                slow=slow,
            )

        return apply

    return decorator_factory


class _VerifyNamespace:
    """Expose ``verify.l0(...)``, ``verify.l1(...)``, etc."""

    l0 = staticmethod(_level_decorator("L0"))
    l1 = staticmethod(_level_decorator("L1"))
    l2 = staticmethod(_level_decorator("L2"))
    l3 = staticmethod(_level_decorator("L3"))


verify = _VerifyNamespace()


def vv_cases(
    *,
    level: VVLevel | None = None,
    method: str | None = None,
    geometry: str | None = None,
    n_groups: int | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Parametrize a test over matching :class:`VerificationCase` entries.

    Applies ``pytest.mark.parametrize("case", [...], ids=[...])`` plus
    the inherited level marker (if every matched case shares the same
    ``vv_level``). Cases are pulled from
    :mod:`orpheus.derivations.reference_values` lazily at collection
    time so this module can be imported without triggering solver
    imports.

    Example::

        @vv_cases(level="L1", method="cp", geometry="slab")
        def test_cp_slab_eigenvalue(case):
            result = solve_cp_slab(case.materials, **case.geom_params)
            assert abs(result.k_inf - case.k_inf) < 1e-10
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Lazy import: avoid pulling solver references at import time.
        from orpheus.derivations.reference_values import all_cases

        cases = [
            c
            for c in all_cases()
            if (level is None or c.vv_level == level)
            and (method is None or c.method == method)
            and (geometry is None or c.geometry == geometry)
            and (n_groups is None or c.n_groups == n_groups)
        ]
        if not cases:
            # Don't silently collect zero-parametrize tests — surface
            # the filter mismatch immediately.
            raise ValueError(
                f"vv_cases(level={level!r}, method={method!r}, "
                f"geometry={geometry!r}, n_groups={n_groups!r}) "
                "matched zero VerificationCase entries"
            )

        ids = [c.name for c in cases]
        func = pytest.mark.parametrize("case", cases, ids=ids)(func)

        inherited_levels = {c.vv_level for c in cases if c.vv_level is not None}
        if len(inherited_levels) == 1:
            (only_level,) = inherited_levels
            func = getattr(pytest.mark, only_level.lower())(func)
        return func

    return decorator
