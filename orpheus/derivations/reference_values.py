"""Unified registry of all analytical verification cases.

Imports all derivation modules and collects their VerificationCase
objects into a single dict for easy lookup by tests and documentation.
"""

from __future__ import annotations

from ._types import VerificationCase

# Registry populated lazily on first access
_CASES: dict[str, VerificationCase] | None = None


_SOLVER_CASES_LOADED = False


def _build_registry() -> dict[str, VerificationCase]:
    """Import all derivation modules and collect analytical/semi-analytical cases.

    Solver-computed cases (SN/MOC heterogeneous via Richardson extrapolation)
    are loaded separately by ``_load_solver_cases()`` when solvers are on the path.
    """
    cases: dict[str, VerificationCase] = {}

    from . import homogeneous, cp_slab, cp_cylinder, cp_sphere, diffusion, sn, moc, mc
    for module in [homogeneous, sn, cp_slab, cp_cylinder, cp_sphere, moc, mc, diffusion]:
        for case in module.all_cases():
            cases[case.name] = case

    return cases


def _load_solver_cases() -> None:
    """Load solver-computed heterogeneous cases (requires solver modules on path)."""
    global _SOLVER_CASES_LOADED
    if _SOLVER_CASES_LOADED:
        return
    _SOLVER_CASES_LOADED = True

    cases = _ensure_loaded()
    try:
        from . import sn, moc, diffusion
        for module in [sn, moc, diffusion]:
            if hasattr(module, 'solver_cases'):
                for case in module.solver_cases():
                    cases[case.name] = case
    except ImportError:
        pass  # solvers not on path (e.g. docs build)


def _ensure_loaded() -> dict[str, VerificationCase]:
    global _CASES
    if _CASES is None:
        _CASES = _build_registry()
    return _CASES


def get(name: str) -> VerificationCase:
    """Get a verification case by name.

    Raises KeyError if not found. Tries loading solver-computed cases
    if the name is not in the analytical registry.
    """
    cases = _ensure_loaded()
    if name not in cases:
        _load_solver_cases()
    return cases[name]


def all_names() -> list[str]:
    """List all available verification case names."""
    _load_solver_cases()
    return sorted(_ensure_loaded().keys())


def all_cases() -> list[VerificationCase]:
    """Return all verification cases."""
    _load_solver_cases()
    return list(_ensure_loaded().values())


def by_geometry(geometry: str) -> list[VerificationCase]:
    """Filter cases by geometry type."""
    return [c for c in _ensure_loaded().values() if c.geometry == geometry]


def by_groups(n_groups: int) -> list[VerificationCase]:
    """Filter cases by number of energy groups."""
    return [c for c in _ensure_loaded().values() if c.n_groups == n_groups]


def by_method(method: str) -> list[VerificationCase]:
    """Filter cases by solver method (homo, cp, sn, moc, mc, dif)."""
    return [c for c in _ensure_loaded().values() if c.method == method]
