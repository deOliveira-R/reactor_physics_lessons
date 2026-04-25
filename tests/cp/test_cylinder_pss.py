"""Foundation tests for ``compute_P_ss_cylinder`` — surface-to-surface
probability for cylinder white BC.

Promoted from ``derivations/diagnostics/diag_cylinder_hebert_pss.py``
on 2026-04-25 (Issue #132 cylinder follow-up). The primitive itself
is shipped in ``orpheus/derivations/peierls_geometry.py``; this file
pins its derivation against:

- closed-form Bickley-Naylor identities at limiting τ values
- independent Monte Carlo estimate
- multi-region routing invariance (homog-MR equals 1R)
- multi-region order-dependence (chord geometry is asymmetric)

The primitive is the building block for a future cylinder
``boundary="white_hebert"`` closure, which is currently blocked on
Issue #112 Phase C (Knyazev :math:`\\mathrm{Ki}_{2+k}` 3-D angular
normalisation correction in ``compute_G_bc`` for cylinder). Once
#112 Phase C lands, this test suite confirms the P_ss^cyl building
block is ready for production wiring.

Foundation marker: tests assert mathematical invariants of the P_ss
derivation, not equation labels — :ref:`peierls-cyl-Pss-homogeneous`
is the math reference but the closed form is verified here at the
primitive level (no Sphinx :label: dependency).
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations._kernels import ki_n_mp
from orpheus.derivations.peierls_geometry import compute_P_ss_cylinder


pytestmark = [pytest.mark.foundation]


# ═══════════════════════════════════════════════════════════════════════
# 1. Closed-form Bickley-Naylor sanity
# ═══════════════════════════════════════════════════════════════════════

def test_Ki_3_at_zero_matches_closed_form():
    """Ki_3(0) = π/4 — the Wallis closed form for the Bickley-Naylor
    function evaluated at zero argument.

    Foundation gate: confirms the underlying ``ki_n_mp`` infrastructure
    used by ``compute_P_ss_cylinder`` returns the textbook value at the
    limiting argument. Any drift here breaks every cylinder primitive
    that calls Ki_n.
    """
    val = float(ki_n_mp(3, 0.0))
    np.testing.assert_allclose(val, np.pi / 4, rtol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# 2. Asymptotic limits of P_ss^cyl
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tau_R", [1e-6, 1e-4, 1e-2])
def test_pss_thin_cell_limit_approaches_unity(tau_R):
    """As τ_R → 0, P_ss^cyl → 1 (thin cell, all surface neutrons transit)."""
    p = compute_P_ss_cylinder(np.array([tau_R]), np.array([1.0]), n_quad=64)
    # P_ss should be just below 1, departing as O(τ_R) for thin cells
    assert 1.0 - 4 * tau_R < p < 1.0, (
        f"τ_R={tau_R}: P_ss={p}, expected ~ 1 - O(τ_R)"
    )


@pytest.mark.parametrize("tau_R", [10.0, 50.0, 100.0])
def test_pss_thick_cell_limit_approaches_zero(tau_R):
    """As τ_R → ∞, P_ss^cyl → 0 (thick cell, all neutrons absorbed)."""
    p = compute_P_ss_cylinder(np.array([tau_R]), np.array([1.0]), n_quad=64)
    assert p < 0.01, f"τ_R={tau_R}: P_ss={p}, expected exponentially small"


# ═══════════════════════════════════════════════════════════════════════
# 3. Independent Monte Carlo verification
# ═══════════════════════════════════════════════════════════════════════

def _pss_cyl_monte_carlo(tau_R: float, n_samples: int = 400_000,
                          seed: int = 42) -> float:
    """Independent MC estimate of P_ss^cyl via rejection sampling.

    Sample (α, β) with PDF ∝ cos α · sin² β over (0, π/2)², then
    accumulate exp(-2τ_R cos α / sin β). Reproduces the cylinder
    surface-to-surface transmission integral directly without using
    the Bickley-Naylor formulation — independent verification of the
    quadrature path.
    """
    rng = np.random.default_rng(seed)
    n_accept = 0
    sum_exp = 0.0
    while n_accept < n_samples:
        batch = 200_000
        a = rng.uniform(0, np.pi / 2, batch)
        b = rng.uniform(0, np.pi / 2, batch)
        u = rng.uniform(0, 1, batch)
        accept = u < (np.cos(a) * np.sin(b) ** 2)
        a_acc = a[accept]
        b_acc = b[accept]
        chord = 2.0 * tau_R * np.cos(a_acc) / np.sin(b_acc)
        sum_exp += float(np.exp(-chord).sum())
        n_accept += int(accept.sum())
    return sum_exp / n_accept


@pytest.mark.parametrize("tau_R", [0.1, 0.5, 1.0, 2.0])
def test_pss_quadrature_matches_independent_monte_carlo(tau_R):
    """Quadrature P_ss^cyl agrees with MC estimate to ~3·MC stderr."""
    p_quad = compute_P_ss_cylinder(np.array([tau_R]), np.array([1.0]), n_quad=64)
    p_mc = _pss_cyl_monte_carlo(tau_R, n_samples=400_000, seed=42)
    # MC stderr ~ sqrt(p(1-p)/N) ~ 1e-3
    diff = abs(p_quad - p_mc)
    assert diff < 5e-3, (
        f"τ_R={tau_R}: P_ss quad={p_quad:.6f} MC={p_mc:.6f} "
        f"diff={diff:.4e} (expect <5e-3)"
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. Multi-region routing invariance
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("tau_R", [0.5, 1.0, 2.0])
def test_pss_multiregion_with_uniform_sigma_equals_homogeneous(tau_R):
    """MR P_ss with all regions same σ_t equals homogeneous P_ss to
    machine precision — confirms no MR routing bug independent of σ_t
    breakpoints.
    """
    sig_t_val = 1.0
    R = tau_R
    p_homog = compute_P_ss_cylinder(np.array([R]), np.array([sig_t_val]),
                                      n_quad=64)
    p_mr2 = compute_P_ss_cylinder(
        np.array([R / 2, R]), np.array([sig_t_val, sig_t_val]), n_quad=64,
    )
    p_mr4 = compute_P_ss_cylinder(
        np.array([0.4 * R, 0.45 * R, 0.55 * R, R]),
        np.array([sig_t_val] * 4), n_quad=64,
    )
    np.testing.assert_allclose(p_homog, p_mr2, atol=1e-12)
    np.testing.assert_allclose(p_homog, p_mr4, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# 5. Multi-region asymmetry (chord geometry order-dependence)
# ═══════════════════════════════════════════════════════════════════════

def test_pss_multiregion_layer_order_matters_for_grazing_chords():
    """Chord geometry: chords with impact parameter h > r_inner cross
    only the OUTER annulus, so swapping σ_t between annuli changes
    the optical depth on those grazing chords.

    For radii=[0.5, 1.0]:
    - σ_t=[0.1, 2.0]: thin-inner, thick-outer.
      Grazing chords (h > 0.5) accumulate τ_outer = HIGH → low P_ss.
    - σ_t=[2.0, 0.1]: thick-inner, thin-outer.
      Grazing chords accumulate τ_outer = LOW → high P_ss.

    Swap MUST give different P_ss (chord geometry is order-sensitive
    in the tangential limit).
    """
    radii = np.array([0.5, 1.0])
    p_thin_inner = compute_P_ss_cylinder(
        np.array([0.1, 2.0]), radii, n_quad=64,
    )
    p_thick_inner = compute_P_ss_cylinder(
        np.array([2.0, 0.1]), radii, n_quad=64,
    )
    # Strong inequality — grazing chords are much more transmissive
    # when the outer annulus is thin
    assert p_thick_inner > 5 * p_thin_inner, (
        f"Expect strong order-dependence; got "
        f"p(σ=[0.1,2])={p_thin_inner}, p(σ=[2,0.1])={p_thick_inner}, "
        f"ratio={p_thick_inner / p_thin_inner:.2f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# 6. Convergence under quadrature refinement
# ═══════════════════════════════════════════════════════════════════════

def test_pss_quadrature_convergence_at_typical_thickness():
    """At τ_R = 1 (typical Class B cell), P_ss^cyl converges
    monotonically as n_quad doubles. Foundation gate confirms the GL
    quadrature is appropriate for this integrand (no oscillatory
    pathology like Issue #114-style ρ-subdivision).
    """
    p_vals = []
    for n_quad in [16, 32, 64, 128]:
        p = compute_P_ss_cylinder(
            np.array([1.0]), np.array([1.0]), n_quad=n_quad,
        )
        p_vals.append(p)
    # Successive differences should DECREASE monotonically (convergence)
    diffs = [abs(p_vals[i + 1] - p_vals[i]) for i in range(len(p_vals) - 1)]
    assert diffs[1] < diffs[0], f"Non-monotone convergence: {diffs}"
    assert diffs[2] < diffs[1], f"Non-monotone convergence: {diffs}"
    # Final value should be well-converged (<1e-6 between n_quad=64 and 128)
    assert diffs[-1] < 1e-6, (
        f"At n_quad=128 vs 64, diff={diffs[-1]} > 1e-6 — under-converged"
    )
