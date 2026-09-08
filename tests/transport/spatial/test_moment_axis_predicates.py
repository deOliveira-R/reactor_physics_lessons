r"""#246 — the named moment-axis predicates: positive + negative (anti-pattern #11).

SKELETON (test-architect, PRE-IMPL for #246). The predicates are NOT written.
These tests pin the CONTRACT and will ERROR (AttributeError) until
``DiscretizationSchemeBase.is_multi_moment`` and
``BulkField.has_spatial_moment_axis`` land (see the spec
``.claude/agent-memory/test-architect/issue_246_moment_axis_predicate_spec.md``).

Two predicates, two pairs:

* ``DiscretizationSchemeBase.is_multi_moment`` (= ``spatial_basis_per_axis > 1``)
  — the SCHEME-level UNCONDITIONAL truth used at the inner-walk sites. LD → True,
  DD → False, regardless of any field provenance.
* ``BulkField.spatial_moments_per_axis`` (the pre-existing moment-width count) —
  the FIELD-level provenance check. (#246 TRIMMED the speculative
  ``has_spatial_moment_axis`` boolean — zero production consumers, Pattern 6; the
  count already answers "does this field carry the moment axis".)
  PROVENANCE-DEPENDENT: ``> 1`` only on output that came through a moment-aware
  producer (the factor is set by the OUTPUT wrap, default 1 everywhere, NOT
  auto-read from the mesh — ``transport/fields/_bases.py:183-194``). A hand-built
  bare field on an LD mesh is ``1`` BY DESIGN — the load-bearing
  construct-general pin (P4').

vv-principles anti-pattern #11: a predicate needs a POSITIVE (real multi-moment →
True, MUST NOT mis-report) AND a NEGATIVE (real single-moment → False). Negative-
only validates the raising/return behaviour but NOT the invariant claim.

``foundation`` level (software invariant — a typed query over a data structure,
no equation ``:label:``; NO ``verifies``). ``-O``-safe: assert via
``pytest.fail`` on the bool, never bare ``assert``.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn import solve_sn_fixed_source
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.spatial import DiamondDifference, LinearDiscontinuous
from orpheus.transport.fields.angular_flux import AngularFlux

pytestmark = pytest.mark.foundation


def _require(predicate: bool, msg: str) -> None:
    r"""-O-safe boolean assert (a bare ``assert`` would be stripped under -O)."""
    if not predicate:
        pytest.fail(msg)


# ═══════════════════════════════════════════════════════════════════════
# P1/P2 — DiscretizationSchemeBase.is_multi_moment (scheme-level, UNCONDITIONAL)
# ═══════════════════════════════════════════════════════════════════════
#
# MUTATION CHECK (run once at impl, document the result in the PR): redefine
# is_multi_moment to a constant True → P2 must redden; constant False → P1 must
# redden. If a mutation does NOT redden, the body is not `spatial_basis_per_axis
# > 1` as claimed (Mode-10 activated-but-unconstrained).


def test_is_multi_moment_true_for_linear_discontinuous() -> None:
    r"""P1 (positive): LD is multi-moment, and the predicate equals the
    underlying ``spatial_basis_per_axis > 1`` (not a hardcoded True)."""
    ld = LinearDiscontinuous()
    _require(
        ld.is_multi_moment is True,
        "LinearDiscontinuous().is_multi_moment should be True",
    )
    _require(
        ld.is_multi_moment == (ld.spatial_basis_per_axis > 1),
        "is_multi_moment must equal spatial_basis_per_axis > 1 (single source)",
    )


def test_is_multi_moment_false_for_diamond_difference() -> None:
    r"""P2 (negative): DD is the slopeless cell-average closure → False."""
    dd = DiamondDifference()
    _require(
        dd.is_multi_moment is False,
        "DiamondDifference().is_multi_moment should be False",
    )
    _require(
        dd.is_multi_moment == (dd.spatial_basis_per_axis > 1),
        "is_multi_moment must equal spatial_basis_per_axis > 1 (single source)",
    )


@pytest.mark.skip(
    reason="Step scheme is a docstring stub today (no live class — see "
    "d5_nd_polymorphism_verification). UN-SKIP when Step lands; it must be "
    "is_multi_moment False (spatial_basis_per_axis == 1)."
)
def test_is_multi_moment_false_for_step() -> None:
    r"""P2' (negative): the Step scheme (slopeless) → False. Gated until Step
    is instantiable."""
    raise NotImplementedError  # pragma: no cover — un-skip when Step exists


# ═══════════════════════════════════════════════════════════════════════
# P3/P4 — BulkField.spatial_moments_per_axis (field-level, PROVENANCE-dependent)
# ═══════════════════════════════════════════════════════════════════════


def _solve_1g_slab(scheme):
    r"""A minimal 1G vacuum slab solve — returns the result whose
    ``angular_flux`` carries (LD) or lacks (DD) the spatial-moment factor."""
    from orpheus.derivations.continuous.mms.sn import _make_1g_mixture

    nx = 8
    materials = {0: _make_1g_mixture(1.0, 0.5)}
    mesh = Mesh1D(
        edges=np.linspace(0.0, 1.0, nx + 1),
        mat_ids=np.zeros(nx, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("vacuum"),
    )
    quad = Quadrature.gauss_legendre(8)
    Q = np.ones((quad.N, 1, nx)) / quad.weights.sum()
    return solve_sn_fixed_source(
        materials, mesh, quad, Q, scheme=scheme,
        boundary_condition="vacuum", inner_solver="krylov",
        max_inner=500, inner_tol=1e-10,
    )


def test_spatial_moment_axis_present_on_moment_aware_producer_output() -> None:
    r"""P3 (positive): a field produced THROUGH the LD moment-aware output wrap
    carries the spatial-moment factor → ``spatial_moments_per_axis > 1``."""
    result = _solve_1g_slab(LinearDiscontinuous())
    bulk = result.angular_flux.interior
    _require(
        bulk.spatial_moments_per_axis > 1,
        "LD producer output should carry the spatial-moment axis",
    )


def test_spatial_moment_axis_absent_on_dd_field() -> None:
    r"""P4 (negative): a DD output field lacks the factor → ``spatial_moments_per_axis == 1``."""
    result = _solve_1g_slab(DiamondDifference())
    bulk = result.angular_flux.interior
    _require(
        bulk.spatial_moments_per_axis == 1,
        "DD output should NOT carry the spatial-moment axis",
    )


def test_spatial_moment_axis_absent_on_hand_built_ld_field() -> None:
    r"""P4' (the CONSTRUCT-GENERAL pin — load-bearing): a hand-built bare field
    on an LD mesh LACKS the factor (``spatial_moments_per_axis == 1``), DESPITE
    the mesh being LD.

    This is INTENTIONAL: the spatial-moment axis is set by the OUTPUT
    wrap (``spatial_moments=per_axis``), default 1 everywhere, NOT auto-read from
    ``mesh.scheme`` (``transport/fields/_bases.py:183-194``). WITHOUT this pin a
    future change that auto-reads the scheme into the field default would pass
    silently and break LD byte-identity. This test is WHY the inner walk uses the
    SCHEME-level ``is_multi_moment``, not a field-level query.

    Build a bare width-1 field (``space=ld_sn_mesh.angular_bulk_space`` —
    NOT the ``angular_trial_space`` widened mint) on an LD ``SNMesh`` and
    assert the field does NOT carry the spatial-moment axis even though the
    mesh's scheme IS multi-moment — widening is the CALLER's selection, by
    property choice since CS4b S5.
    """
    from tests.sn._test_helpers import placeholder_materials

    nx = 5
    mesh = Mesh1D(
        edges=np.linspace(0.0, 1.0, nx + 1),
        mat_ids=np.zeros(nx, dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("vacuum"),
    )
    quad = Quadrature.gauss_legendre(4)
    ld_sn_mesh = SNMesh(
        mesh, quad, placeholder_materials(ng=2), scheme=LinearDiscontinuous(),
    )
    # Premise: the mesh's scheme IS multi-moment — the field's False is NOT a
    # trivial DD-mesh result; it documents the provenance discipline.
    _require(
        ld_sn_mesh.scheme.is_multi_moment is True,
        "premise broken: the SNMesh scheme must be LD (multi-moment)",
    )
    # Bare field — NO spatial_moments=, the construct-general default 1.
    values = np.zeros((quad.N, ld_sn_mesh.ng, nx))
    bulk = AngularFlux(values=values, space=ld_sn_mesh.angular_bulk_space)
    _require(
        bulk.spatial_moments_per_axis == 1,
        "a hand-built bare field on an LD mesh must NOT carry the moment axis "
        "(the factor is producer-set, not mesh-derived; construct-general pin)",
    )
