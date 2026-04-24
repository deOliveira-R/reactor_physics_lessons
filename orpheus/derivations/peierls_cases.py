r"""Unified continuous-reference registry for Peierls Nyström solvers,
organized by **topological class** instead of shape.

See :file:`.claude/plans/topology-based-consolidation.md` and Sphinx
§\ ``theory-peierls-capabilities`` / §\ ``theory-peierls-naming``.

The two topological classes are:

- **Class A — two-surface** (F.4 applies). Members: slab (two parallel
  faces), hollow annular cylinder (inner + outer ring), hollow sphere
  (inner + outer shell). Shared closure class: Stamm'ler IV Eq. 34 =
  Hébert 2009 Eq. 3.323 (scalar rank-2 per-face).
- **Class B — one-surface compact** (rank-1 Mark only). Members:
  solid cylinder, solid sphere. F.4 structurally collapses.

This module is the canonical entry point for continuous-reference
registration via :func:`cases`. The per-geometry modules
(:mod:`~orpheus.derivations.peierls_slab`,
:mod:`~orpheus.derivations.peierls_cylinder`,
:mod:`~orpheus.derivations.peierls_sphere`) retain their
``_build_*_case`` constructor functions — this module calls them
directly. Their ``continuous_cases()`` hooks return empty lists to
avoid double-registration; the registry-builder's auto-discovery
walks every module and this module is the single source for Peierls
continuous references.

Slab note (2026-04-23): slab's machinery (native E₁ Nyström) has
not yet been absorbed into the curvilinear
:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
polar form — that requires arbitrary-precision quadrature on the
log singularity, scheduled for a future numerics session. Until
then slab lives in its own module but is **registered** here under
Class A alongside the curvilinear hollow cells, because the closure
class (F.4 scalar rank-2 per-face) and the verification methodology
(L19 stability protocol) are identical across Class A regardless of
machinery.
"""
from __future__ import annotations

from ._reference import ContinuousReferenceSolution


# ---------------------------------------------------------------------
# Class A — two-surface (F.4 applies)
# ---------------------------------------------------------------------


def build_two_surface_case(
    shape: str,
    ng_key: str = "1g",
    n_regions: int = 1,
    *,
    inner_radius: float | None = None,
) -> ContinuousReferenceSolution:
    r"""Build a Class-A (two-surface) continuous reference.

    Class A members share the F.4 scalar rank-2 per-face closure
    (:math:numref:`hebert-3-323`). Dispatch on ``shape``:

    - ``"slab"`` — calls
      :func:`orpheus.derivations.peierls_slab._build_peierls_slab_case`.
      ``inner_radius`` is ignored (slab has two parallel faces at
      :math:`x=0` and :math:`x=L`, not a cavity).
    - ``"cylinder-1d"`` — requires ``inner_radius > 0``; calls
      :func:`orpheus.derivations.peierls_cylinder._build_peierls_cylinder_hollow_f4_case`.
    - ``"sphere-1d"`` — requires ``inner_radius > 0``; calls
      :func:`orpheus.derivations.peierls_sphere._build_peierls_sphere_hollow_f4_case`.

    Parameters
    ----------
    shape
        ``"slab"``, ``"cylinder-1d"``, or ``"sphere-1d"``.
    ng_key
        XS-library group key (``"1g"``, ``"2g"``, ``"4g"``). The
        hollow curvilinear cases currently support ``"1g"`` only;
        multi-group lift is Issue #104.
    n_regions
        Number of radial regions. Hollow curvilinear cases support
        1-region only today (single annular shell). Slab supports
        1/2/4.
    inner_radius
        Cavity radius :math:`r_0` for curvilinear hollow cases.
        **Required** for ``"cylinder-1d"`` / ``"sphere-1d"``; must
        be strictly between 0 and the outer radius. Ignored for slab.

    Raises
    ------
    ValueError
        For curvilinear shapes when ``inner_radius`` is missing or
        not in ``(0, R_outer)``. Use
        :func:`build_one_surface_compact_case` for solid geometry.
    """
    if shape == "slab":
        from .peierls_slab import _build_peierls_slab_case
        return _build_peierls_slab_case(
            ng_key, n_regions,
        )
    if shape == "cylinder-1d":
        if inner_radius is None:
            raise ValueError(
                "cylinder-1d is a Class A (two-surface) case only "
                "when inner_radius > 0. Use build_one_surface_compact_case "
                "for solid cylinder."
            )
        from .peierls_cylinder import _build_peierls_cylinder_hollow_f4_case
        # The hollow-f4 builder takes r_0_over_R (unitless), not
        # absolute inner_radius. The single shipped outer radius for
        # 1g 1-region is R=1 (from cp_cylinder._RADII[1][-1]) so the
        # two are numerically equal when R=1 — but be explicit.
        from .cp_cylinder import _RADII as _CYL_RADII
        R_out = float(_CYL_RADII[n_regions][-1])
        return _build_peierls_cylinder_hollow_f4_case(
            r0_over_R=float(inner_radius) / R_out,
        )
    if shape == "sphere-1d":
        if inner_radius is None:
            raise ValueError(
                "sphere-1d is a Class A (two-surface) case only "
                "when inner_radius > 0. Use build_one_surface_compact_case "
                "for solid sphere."
            )
        from .peierls_sphere import _build_peierls_sphere_hollow_f4_case
        from .cp_sphere import _RADII as _SPH_RADII
        R_out = float(_SPH_RADII[n_regions][-1])
        return _build_peierls_sphere_hollow_f4_case(
            r0_over_R=float(inner_radius) / R_out,
        )
    raise ValueError(
        f"build_two_surface_case: unknown shape {shape!r}; "
        f"expected 'slab', 'cylinder-1d', or 'sphere-1d'"
    )


# ---------------------------------------------------------------------
# Class B — one-surface compact (rank-1 Mark only)
# ---------------------------------------------------------------------


def build_one_surface_compact_case(
    shape: str,
    ng_key: str = "1g",
    n_regions: int = 1,
) -> ContinuousReferenceSolution:
    r"""Build a Class-B (one-surface compact) continuous reference.

    Class B members (solid cylinder, solid sphere) ship only the
    rank-1 Mark closure. F.4 collapses to rank-1 Mark on solid
    geometry (no second-face coupling).

    **No references are registered in Class B today.** The rank-1
    Mark floor (21 % err at :math:`R = 1` MFP for cylinder per
    Issue #103) is too loose to serve as an L1 reference for the
    ``cp_{cyl,sph}1D_*`` solver tests. Lifting the floor requires
    Issue #103 (rank-N DP\ :sub:`N` on the single outer face) or
    Issue #101 (chord-based Ki₁ analytical).

    This function exists for future use when one of those lands.
    Until then it unconditionally raises ``NotImplementedError``
    with an explanatory message.
    """
    raise NotImplementedError(
        f"build_one_surface_compact_case({shape!r}) is not yet "
        f"populated. Class B (solid cylinder / solid sphere) has "
        f"no shipped continuous references because the rank-1 Mark "
        f"floor is too loose (21 % err at R=1 MFP per Issue #103). "
        f"Resolution requires rank-N DP_N (Issue #103) or chord-"
        f"based Ki_1 analytical (Issue #101)."
    )


# ---------------------------------------------------------------------
# Registry entry point — auto-discovered by reference_values.py
# ---------------------------------------------------------------------


def _class_a_cases() -> list[ContinuousReferenceSolution]:
    """Class A — two-surface cases. Slab + hollow cylinder/sphere F.4."""
    refs: list[ContinuousReferenceSolution] = []
    # Slab: 2G 2-region (current shipped default)
    refs.append(build_two_surface_case("slab", "2g", 2))
    # Hollow cylinder F.4 at r_0/R ∈ {0.1, 0.2, 0.3}
    for r0 in (0.1, 0.2, 0.3):
        refs.append(build_two_surface_case(
            "cylinder-1d", "1g", 1, inner_radius=r0,
        ))
    # Hollow sphere F.4 at r_0/R ∈ {0.1, 0.2, 0.3}
    for r0 in (0.1, 0.2, 0.3):
        refs.append(build_two_surface_case(
            "sphere-1d", "1g", 1, inner_radius=r0,
        ))
    return refs


def _class_b_cases() -> list[ContinuousReferenceSolution]:
    """Class B — one-surface compact cases. Empty today."""
    return []


def continuous_cases() -> list[ContinuousReferenceSolution]:
    r"""All Peierls continuous references across both topology classes.

    Registered by auto-discovery in
    :func:`orpheus.derivations.reference_values._build_continuous_registry`
    via the standard ``continuous_cases()`` contract.
    """
    return _class_a_cases() + _class_b_cases()


# Alias for readers who want the topology-explicit name.
cases = continuous_cases
