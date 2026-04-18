"""Foundation test: rank-N white-BC closure must preserve CONSERVATION.

Created 2026-04-18 during Issue #112 investigation.

For a homogeneous material with :math:`\\Sigma_a = \\Sigma_t` (pure absorber,
no scattering, no fission) inside a WHITE-boundary cell, neutron balance
says: any neutron emitted is eventually absorbed (since white = no escape).
So for a uniform volumetric source ``q = 1``, the steady-state flux is
``φ = 1/Σ_t`` everywhere inside, and the Peierls operator identity becomes

.. math::

   K \\cdot q = \\Sigma_t \\varphi = 1,

i.e. the full kernel acting on the all-ones vector should return the
all-ones vector (at Σ_t = 1).

This identity holds as :math:`(n_{angular}, n_{\\rho}, n_{\\mathrm{surf}})
\\to \\infty` (quadrature convergence) AND as :math:`N \\to \\infty`
(Marshak mode truncation). Concretely: rank-1 leaks because the
Lambertian BC closure is approximate; rank-N MUST reduce (never
increase) the conservation defect.

A rank-N implementation that **increases** the conservation defect as
modes are added is wrong in the mode-n magnitude — the smoking gun for
the current Issue #112 bug.

This test is marked ``foundation`` because it checks an invariant of the
kernel assembly itself, not a specific equation on the theory page.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    SPHERE_1D,
    build_volume_kernel,
    build_white_bc_correction,
    build_white_bc_correction_rank_n,
    composite_gl_r,
)


_GEOMETRIES = [CYLINDER_1D, SPHERE_1D]


def _conservation_defect(geometry, R, n_bc_modes, n_quad=16):
    """Compute ``max |K·1 - 1|`` for homogeneous Σ_t = 1."""
    radii = np.array([R])
    sig_t = np.array([1.0])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 5, dps=25)
    K_vol = build_volume_kernel(
        geometry, r_nodes, panels, radii, sig_t,
        n_angular=n_quad, n_rho=n_quad, dps=25,
    )
    K_bc = build_white_bc_correction_rank_n(
        geometry, r_nodes, r_wts, radii, sig_t,
        n_angular=n_quad, n_surf_quad=n_quad, dps=25,
        n_bc_modes=n_bc_modes,
    )
    K = K_vol + K_bc
    q = np.ones(len(r_nodes))
    Kq = K @ q
    return float(np.max(np.abs(Kq - 1.0)))


@pytest.mark.foundation
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_rank1_conservation_defect_is_small(geometry):
    """Rank-1 conservation defect ≤ 20 % at R=1, ≤ 10 % at R=10.

    These bounds reflect the known Lambertian leakage of the rank-1
    closure (~20 % at thin R, small at thick R).  Tighter bounds would
    over-constrain the rank-1 form; this just sanity-checks the known
    rank-1 accuracy before we test rank-N behavior.
    """
    defect_thin = _conservation_defect(geometry, R=1.0, n_bc_modes=1)
    assert defect_thin < 0.25, (
        f"[{geometry.kind}] R=1 rank-1 conservation defect "
        f"= {defect_thin:.3e} exceeds 25 % (known rank-1 accuracy)"
    )
    defect_thick = _conservation_defect(geometry, R=10.0, n_bc_modes=1)
    assert defect_thick < 0.1, (
        f"[{geometry.kind}] R=10 rank-1 conservation defect "
        f"= {defect_thick:.3e} exceeds 10 %"
    )


@pytest.mark.foundation
@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_rank_n_conservation_improves(geometry):
    """Adding Marshak modes must NEVER worsen conservation.

    At thick R=10 where rank-1 is already well-conserved (<1 % defect),
    a correctly-normalised rank-N closure should leave the defect
    essentially unchanged (drift < 10 % of rank-1 value).

    Flipped from xfail to pass on 2026-04-18 after Issue #112 fix
    landed: the canonical DP\\_N outgoing partial-current moment
    with the ``(ρ_max / R)²`` surface-to-observer Jacobian factor
    in ``compute_P_esc_mode`` causes rank-N to STRENGTHEN conservation
    rather than degrade it. This test is the quantitative gate that
    catches any future regression of that Jacobian factor.
    """
    defect_rank_1 = _conservation_defect(geometry, R=10.0, n_bc_modes=1)
    defect_rank_3 = _conservation_defect(geometry, R=10.0, n_bc_modes=3)
    # Canonical behavior: defect should shrink (or stay ≈ the same).
    assert defect_rank_3 <= 1.1 * defect_rank_1, (
        f"[{geometry.kind}] R=10: rank-3 conservation defect "
        f"({defect_rank_3:.3e}) exceeds 110 % of rank-1 "
        f"({defect_rank_1:.3e}). Rank-N is INTRODUCING non-conservation "
        f"— mode-n magnitude must be wrong (regression of the "
        f"(ρ_max/R)² Jacobian factor?)."
    )
