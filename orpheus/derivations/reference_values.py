"""Unified registry of all verification reference solutions.

Holds two parallel registries during the Phase-0 → Phase-6 migration:

1. ``_CASES`` — the legacy :class:`VerificationCase` registry keyed
   by name, populated from existing ``derive_*`` functions. Every
   currently-green test consumes this.

2. ``_CONTINUOUS`` — the new :class:`ContinuousReferenceSolution`
   registry. Populated by derivations that have been retrofitted to
   the Phase-0 contract. Keys are identical to ``_CASES`` when a
   derivation has been upgraded, so a test can migrate one call site
   at a time.

Retrieval functions:

- :func:`get` — legacy, returns a :class:`VerificationCase`.
- :func:`continuous_get` — returns a :class:`ContinuousReferenceSolution`
  if registered; raises :class:`KeyError` otherwise. Tests that need
  a continuous reference call this; tests that only need a scalar
  ``k_inf`` can stay on :func:`get`.
- :func:`continuous_all_names`, :func:`continuous_all` — enumerate
  the upgraded derivations.

The two registries are additive — upgrading a derivation registers
the new :class:`ContinuousReferenceSolution` **without** removing
the legacy case (the new one can call ``.as_verification_case()`` to
produce a backward-compatible bridge), so the migration is incremental.
"""

from __future__ import annotations

from ._reference import ContinuousReferenceSolution
from ._types import VerificationCase

# Legacy registry populated lazily on first access
_CASES: dict[str, VerificationCase] | None = None

# Phase-0 continuous-reference registry
_CONTINUOUS: dict[str, ContinuousReferenceSolution] | None = None


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
    """Load solver-computed heterogeneous cases (legacy T3 path).

    .. deprecated:: verification-campaign

        ``solver_cases()`` is the entry point for the Richardson-
        extrapolated heterogeneous references that the
        verification campaign is replacing one module at a time:

        - ``sn.py`` — deleted in Phase 2.1a; heterogeneous SN
          verification is now the MMS continuous reference in
          :mod:`orpheus.derivations.sn_mms`.
        - ``moc.py`` — Phase 2.2 target (still T3 at the time of
          writing).
        - ``diffusion.py`` — Phase 1.2 replaced the *continuous*
          registry entry with a transcendental transfer-matrix
          reference, but the legacy Richardson ``solver_cases``
          path is kept alive during the migration window so
          that pre-Phase-1.2 tests (which still read from the
          legacy :func:`get` registry) keep working.

        The loop below iterates over modules that *still* define
        ``solver_cases`` at import time, so deletions are a
        one-line change to the target module.
    """
    global _SOLVER_CASES_LOADED
    if _SOLVER_CASES_LOADED:
        return
    _SOLVER_CASES_LOADED = True

    cases = _ensure_loaded()
    try:
        from . import diffusion, moc, sn
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


# ═══════════════════════════════════════════════════════════════════════
# Phase-0 continuous-reference registry
# ═══════════════════════════════════════════════════════════════════════

def _build_continuous_registry() -> dict[str, ContinuousReferenceSolution]:
    """Import retrofitted derivation modules and collect their continuous references.

    As each module in :mod:`orpheus.derivations` is upgraded to the
    Phase-0 contract, add its import here. The function signature
    modules are expected to expose is ``continuous_cases() -> list[ContinuousReferenceSolution]``.
    """
    refs: dict[str, ContinuousReferenceSolution] = {}

    # Populated incrementally through Phases 1–5 of the verification
    # campaign.
    from . import diffusion, homogeneous, moc_mms, sn, sn_mms
    _continuous_modules: list = [homogeneous, diffusion, sn, sn_mms, moc_mms]

    for module in _continuous_modules:
        if hasattr(module, "continuous_cases"):
            for ref in module.continuous_cases():
                refs[ref.name] = ref

    return refs


def _ensure_continuous_loaded() -> dict[str, ContinuousReferenceSolution]:
    global _CONTINUOUS
    if _CONTINUOUS is None:
        _CONTINUOUS = _build_continuous_registry()
    return _CONTINUOUS


def continuous_register(ref: ContinuousReferenceSolution) -> None:
    """Register a :class:`ContinuousReferenceSolution` into the registry.

    The preferred registration path is to add the producing module
    to the ``_continuous_modules`` list inside
    :func:`_build_continuous_registry`. This explicit entry point
    exists for tests and one-off derivations that need to inject
    a reference at import time without touching the registry source.
    """
    refs = _ensure_continuous_loaded()
    refs[ref.name] = ref


def continuous_get(name: str) -> ContinuousReferenceSolution:
    """Retrieve a continuous reference solution by name.

    Raises :class:`KeyError` if ``name`` has not been upgraded yet.
    Tests that need a continuous reference should call this directly;
    tests that only need a scalar ``k_inf`` should stay on
    :func:`get` until Phase 2 of the migration.
    """
    refs = _ensure_continuous_loaded()
    return refs[name]


def continuous_all_names() -> list[str]:
    """List every registered continuous reference solution name."""
    return sorted(_ensure_continuous_loaded().keys())


def continuous_all() -> list[ContinuousReferenceSolution]:
    """Return every registered continuous reference solution."""
    return list(_ensure_continuous_loaded().values())


def continuous_by_operator_form(form: str) -> list[ContinuousReferenceSolution]:
    """Filter continuous references by :attr:`ContinuousReferenceSolution.operator_form`.

    Used by the verification audit tool to group references by the
    equation form they commit to, and by tests that want to pull
    every reference valid for their solver's operator.
    """
    return [
        r for r in _ensure_continuous_loaded().values()
        if r.operator_form == form
    ]
