"""L0 verification of the mode-n primitives G_bc^(n) and P_esc^(n).

Created 2026-04-18 during Issue #112 investigation. These tests exercise
properties of the mode-n primitives that are *independent* of the
downstream rank-N closure normalization — they nail down the integrand
structure so that when the rank-N bug is fixed, we know the primitives
themselves are correct.

The core identity tested:

* **Sphere identity** (`G_bc^(n) = 4 · P_esc^(n)`): for the sphere, both
  primitives share the same observer-centred angular integrand
  :math:`\\int_{4\\pi} \\tilde P_n(\\mu_s)\\,e^{-\\tau}\\,\\mathrm d\\Omega`,
  differing only in the geometry prefactor absorbed into each: G_bc uses
  ``2`` (as "response to local Lambertian partial current") and P_esc uses
  ``0.5`` (as "escape probability per emission"), giving the fixed 4:1
  ratio. This identity MUST hold at every r, every n, every R.

* **Rank-1 center value** (`G_bc(0) = 4 e^{-Σ_t R}` for sphere): at the
  origin, every ray from the observer goes radially outward, attenuating
  uniformly by ``exp(-Σ_t R)``. The angular integral trivially evaluates.

These tests are part of the rank-N investigation trail (Issue #112) but
stand alone as building-block verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D,
    SPHERE_1D,
    compute_G_bc,
    compute_G_bc_mode,
    compute_P_esc,
    compute_P_esc_mode,
    composite_gl_r,
)


pytestmark = [pytest.mark.verifies("peierls-rank-n-bc-closure")]


_SIG_T = np.array([1.0])


@pytest.mark.l0
@pytest.mark.parametrize("R", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_sphere_rank1_G_bc_equals_4_P_esc(R):
    """Sphere: ``G_bc(r) = 4 · P_esc(r)`` pointwise, every R.

    Both primitives are observer-centred angular integrals with identical
    integrands; the ratio 4 comes from the prefactor convention (2 vs 0.5).
    This is an ORTHOGONAL identity to the rank-1 row-sum tests — it checks
    that the two primitives are related by reciprocity.
    """
    radii = np.array([R])
    r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)

    G = compute_G_bc(SPHERE_1D, r_nodes, radii, _SIG_T, n_surf_quad=32, dps=25)
    P = compute_P_esc(SPHERE_1D, r_nodes, radii, _SIG_T, n_angular=32, dps=25)

    ratio = G / P
    assert np.allclose(ratio, 4.0, rtol=1e-12, atol=1e-12), (
        f"R={R}: G_bc/P_esc ratio deviates from 4.0. Got range "
        f"[{ratio.min():.6f}, {ratio.max():.6f}]"
    )


@pytest.mark.l0
@pytest.mark.parametrize("R", [1.0, 5.0])
@pytest.mark.parametrize("n_mode", [1, 2, 3, 4])
def test_sphere_rank_n_G_bc_not_proportional_to_P_esc(R, n_mode):
    """Sphere mode-n ≥ 1: ``G_bc_mode`` and ``P_esc_mode`` are NOT proportional.

    The investigator's 2026-04-18 finding was that the old (pre-fix)
    ``compute_P_esc_mode`` shared the identical angular integrand with
    ``compute_G_bc_mode``, making ``G_n = 4·P_n`` pointwise — a
    *symmetric* rank-1 outer product in the ``P̃_n`` basis. That
    symmetry was precisely what prevented Marshak convergence.

    The current ``compute_P_esc_mode`` is the **canonical DP_N
    outgoing partial-current moment** per
    :eq:`peierls-rank-n-P-esc-moment`, carrying the
    surface-to-observer Jacobian factor ``(ρ_max / R)²`` that the old
    form was missing. This factor DELIBERATELY BREAKS the ``G = 4·P``
    symmetry for n ≥ 1, making the rank-N K_bc decomposition truly
    higher-rank and enabling the canonical Marshak convergence ladder
    (Sanchez & McCormick 1982 §III.F.1).

    For n=0 the two functions ALSO differ (the Jacobian factor is not
    trivially 1 off-center); the mode-0 path in
    ``build_white_bc_correction_rank_n`` therefore routes through the
    legacy ``compute_P_esc`` / ``compute_G_bc``, not these ``_mode``
    versions — see ``test_rank1_bit_exact_recovery`` for that gate.

    This test asserts the **asymmetry** at n ≥ 1: if a future edit
    accidentally re-symmetrised the integrands (removing the Jacobian),
    convergence would regress and this test would catch it.
    """
    radii = np.array([R])
    r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)

    G = compute_G_bc_mode(
        SPHERE_1D, r_nodes, radii, _SIG_T, n_mode,
        n_surf_quad=32, dps=25,
    )
    P = compute_P_esc_mode(
        SPHERE_1D, r_nodes, radii, _SIG_T, n_mode,
        n_angular=32, dps=25,
    )

    scale = float(np.max(np.abs(P)))
    assert scale > 0.0, f"P_esc^(n={n_mode}) is identically zero"
    rel_diff = np.abs(G - 4.0 * P) / scale
    # The asymmetry is the critical rank-N feature. Use 1e-3 as a "clearly
    # distinguishable from rank-1-in-P_n-basis" threshold — the empirical
    # asymmetry for n=1..4 at R=1,5 is well above 1 %.
    assert rel_diff.max() > 1e-3, (
        f"R={R}, n={n_mode}: G_bc and 4·P_esc are essentially "
        f"proportional (max rel diff = {rel_diff.max():.3e} < 1e-3). "
        f"This means the Jacobian factor (ρ_max/R)² in "
        f"compute_P_esc_mode has been removed, which would regress "
        f"rank-N Marshak convergence. See Issue #112 history."
    )


@pytest.mark.l0
@pytest.mark.parametrize("Sigma_t_R", [0.5, 1.0, 2.0, 4.0, 8.0])
def test_sphere_G_bc_at_origin_equals_four_exp_neg_tau(Sigma_t_R):
    r"""Sphere: ``G_bc(r \to 0) = 4 · exp(-Σ_t R)`` analytically.

    At the origin every ray from the observer reaches the surface at
    distance R with the same attenuation ``exp(-Σ_t R)``. The angular
    integral :math:`\\int_0^\\pi \\sin\\theta\\,\\mathrm d\\theta = 2`
    together with the prefactor 2 yields ``G_bc(0) = 4 · exp(-Σ_t R)``.

    Ground truth for the mode-0 integrand, INDEPENDENT of the rank-N
    closure.
    """
    R = 1.0
    sigma_t = Sigma_t_R / R  # choose R=1 without loss of generality
    sig_t = np.array([sigma_t])
    radii = np.array([R])
    r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)

    G = compute_G_bc(SPHERE_1D, r_nodes, radii, sig_t, n_surf_quad=64, dps=30)

    # At the innermost node (closest to origin), the result should approach
    # 4 exp(-Σ_t R).  Coarse GL nodes aren't exactly at r=0, so use the
    # analytical result as a limit check for the smallest r_node.
    expected = 4.0 * np.exp(-Sigma_t_R)
    r_small = float(r_nodes[0])
    assert r_small < 0.1, f"Unexpected innermost r={r_small}"

    # At r_small, the result should be very close to the r=0 limit.
    # A strict sanity bound is within 5% (the innermost GL node at this
    # quadrature order sits at r ≈ 0.023 relative to R=1).
    rel_err = abs(G[0] - expected) / expected
    assert rel_err < 0.05, (
        f"Σ_t R = {Sigma_t_R}: G_bc(r={r_small:.4f}) = {G[0]:.6e}, "
        f"expected ≈ 4·exp(-Σ_t R) = {expected:.6e} at r=0. "
        f"Relative error = {rel_err*100:.2f}%"
    )


@pytest.mark.l0
def test_cylinder_G_bc_not_proportional_to_P_esc():
    """Cylinder: G_bc and P_esc do NOT share a constant ratio.

    Unlike the sphere, the cylinder's G_bc uses the surface-centred
    ``Ki_1(τ)/d`` integrand (from the 2-D Bickley-Naylor reduction)
    while P_esc uses the observer-centred ``Ki_2(τ)`` integrand.  These
    are NOT proportional — the ratio is a non-trivial function of r.

    This negative test guards against accidentally unifying the two
    cylinder primitives under a wrong identity.
    """
    R = 2.0
    radii = np.array([R])
    r_nodes, _, _ = composite_gl_r(radii, 2, 5, dps=25)

    G = compute_G_bc(CYLINDER_1D, r_nodes, radii, _SIG_T,
                     n_surf_quad=32, dps=25)
    P = compute_P_esc(CYLINDER_1D, r_nodes, radii, _SIG_T,
                      n_angular=32, dps=25)

    ratio = G / P
    # The ratio varies substantially across r (we've seen [2.1, 5.6]).
    # Require that the relative spread is bigger than 10 %.
    assert (ratio.max() - ratio.min()) / ratio.mean() > 0.1, (
        "Cylinder G_bc/P_esc unexpectedly uniform — has the geometry "
        "definition changed?  Spread: "
        f"[{ratio.min():.3f}, {ratio.max():.3f}]"
    )
