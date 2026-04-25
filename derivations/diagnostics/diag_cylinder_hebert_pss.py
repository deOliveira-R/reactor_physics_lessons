"""Diagnostic: derive and verify P_ss for cylinder white BC (Issue #132).

Investigation goal: extend the Hébert (2009) §3.8.5 Eq. (3.323) closure
``K_bc^white = K_bc^Mark / (1 - P_ss)`` from sphere (where it gives <1.5%
k_eff agreement on Class B 1G/1R, 2G/1R, 2G/2R) to cylinder. The cylinder
analog of ``compute_P_ss_sphere`` integrates the surface-to-surface
transmission probability over the *full 3-D* inward hemisphere on a
surface point of the lateral cylinder face (NOT just the 2-D in-plane
chord).

DERIVATION
----------
Surface point on infinite cylinder of radius R with inward normal n.
Inward direction Ω parameterised by (α, β):

  α ∈ (-π/2, π/2)  : azimuthal angle from inward normal in transverse plane
  β ∈ (0, π)       : polar angle from cylinder axis (z)

Transverse-plane projection of Ω has magnitude sin β; its 2-D direction
makes angle α with the inward normal. So the chord through the cylinder
in the transverse plane has length 2R cos α (projection of chord onto
the transverse plane). The *3-D* chord along Ω is

  ℓ(α, β) = 2R cos α / sin β              (slanted-chord, sin β = transverse-component of Ω)

Inward partial current weight (Ω·n) = cos α · sin β.

Uniform isotropic inward angular flux ψ⁻ = J⁻/π (with J⁻ = π ψ⁻).

P_ss = (1/J⁻) ∫_{Ω·n>0} (Ω·n) ψ⁻ exp(-Σ_t · ℓ) dΩ
     = (1/π) ∫_{α=-π/2}^{π/2} ∫_{β=0}^{π} sin β · cos α · sin β · exp(-2τ_R cos α / sin β) dα dβ

By symmetry (α ↔ -α and β ↔ π-β):

P_ss = (4/π) ∫₀^{π/2} cos α dα ∫₀^{π/2} sin² β · exp(-2τ_R cos α / sin β) dβ

Recognising the inner integral as the Bickley-Naylor function

  Ki_3(x) := ∫₀^{π/2} sin² θ · exp(-x/sin θ) dθ

gives the COMPACT 1-D form

  ┌─────────────────────────────────────────────────────────────┐
  │  P_ss^cyl(τ_R) = (4/π) ∫₀^{π/2} cos α · Ki_3(2 τ_R cos α) dα │
  └─────────────────────────────────────────────────────────────┘

with τ_R = Σ_t · R for the homogeneous cell. Substituting t = cos α:

  P_ss^cyl(τ_R) = (4/π) ∫₀¹ t · Ki_3(2 τ_R t) / √(1 - t²) dt

The 1/√(1-t²) Chebyshev singularity at t=1 (grazing rays) is integrable
and is handled by GL on α directly (no singularity in α-coordinates).

MULTI-REGION extension
----------------------
For a cylinder partitioned into N annuli with outer radii r_1 < r_2 <
... < r_N = R, a chord at angle α from the inward normal in the transverse
plane crosses annular boundaries at impact parameter h = R sin α. The
2-D chord in annulus k has length

  ℓ_k^{2D}(α) = 2 (√(r_k² - h²) · 𝟙[h<r_k] - √(r_{k-1}² - h²) · 𝟙[h<r_{k-1}])

with r_0 = 0. The total 2-D optical depth is

  τ^{2D}(α) = Σ_k Σ_{t,k} · ℓ_k^{2D}(α)

The 3-D chord along Ω is the 2-D chord scaled by 1/sin β. So:

  P_ss^cyl_MR(α) = (4/π) ∫₀^{π/2} cos α dα ∫₀^{π/2} sin² β · exp(-τ^{2D}(α)/sin β) dβ
                 = (4/π) ∫₀^{π/2} cos α · Ki_3(τ^{2D}(α)) dα

This MIRRORS compute_P_ss_sphere exactly except:
  - sphere uses cos θ' (Lambertian) and exp(-τ^{3D})    [no β to integrate]
  - cylinder uses cos α and Ki_3(τ^{2D}(α))             [β-integral done analytically]
  - sphere chord at angle θ': uses h = R sin θ' identical
  - both use the SAME chord-segment-in-annulus formula

SANITY GATES
------------
- Ki_3(0) = π/4 (exact)
- τ_R → 0:  P_ss → 1  (thin cell, all surface neutrons transit)
- τ_R → ∞:  P_ss → 0  (thick cell, all absorbed)
- MC verification: cosine-sample (α, β) inward, accumulate exp(-2τ_R cos α / sin β)

If this diagnostic passes (Phase 1 of Issue #132), promote to:
``tests/cp/test_cylinder_hebert_pss.py`` as the foundation regression
test for P_ss^cyl that gates the cylinder Hébert closure.
"""
from __future__ import annotations

import numpy as np
import mpmath as mp
import pytest


def Ki_n_mp(n: int, x: float, dps: int = 25) -> float:
    """Bickley-Naylor of order n: ∫₀^{π/2} sin^{n-1}θ · exp(-x/sinθ) dθ."""
    if x <= 0.0:
        # Closed forms at zero
        if n == 1:
            return float(mp.pi / 2)
        if n == 2:
            return 1.0
        if n == 3:
            return float(mp.pi / 4)
        # General: Ki_n(0) = ∫ sin^{n-1} θ dθ
        return float(mp.beta(mp.mpf(n)/2, mp.mpf(1)/2) / 2)
    with mp.workdps(dps):
        val = mp.quad(
            lambda th: mp.sin(th) ** (n - 1) * mp.exp(-x / mp.sin(th)),
            [0, mp.pi / 2],
        )
    return float(val)


def compute_P_ss_cylinder_homogeneous(
    Sigma_t: float, R: float, *, n_quad: int = 64, dps: int = 25
) -> float:
    """Surface-to-surface probability for HOMOGENEOUS cylinder, white BC.

    Closed (semi-)form in 1 dimension:
        P_ss = (4/π) · ∫₀^{π/2} cos α · Ki_3(2 τ_R cos α) dα
    with τ_R = Σ_t R.
    """
    tau_R = Sigma_t * R
    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    a, b = 0.0, np.pi / 2
    alpha = 0.5 * (b - a) * pts + 0.5 * (b + a)
    w = 0.5 * (b - a) * wts
    val = 0.0
    for k in range(n_quad):
        ca = float(np.cos(alpha[k]))
        val += w[k] * ca * Ki_n_mp(3, 2 * tau_R * ca, dps=dps)
    return float(4.0 / np.pi * val)


def compute_P_ss_cylinder_multiregion(
    sig_t_per_region: np.ndarray,
    radii: np.ndarray,
    *,
    n_quad: int = 64,
    dps: int = 25,
) -> float:
    """Multi-region cylinder P_ss with white BC.

    For chord at impact parameter h = R sin α through annular regions:
        τ^{2D}(α) = Σ_k Σ_{t,k} · 2(√(r_k² - h²)·1[h<r_k]
                                  - √(r_{k-1}² - h²)·1[h<r_{k-1}])
    Then:
        P_ss = (4/π) · ∫₀^{π/2} cos α · Ki_3(τ^{2D}(α)) dα
    """
    sig_t = np.asarray(sig_t_per_region, dtype=float)
    radii = np.asarray(radii, dtype=float)
    R = float(radii[-1])
    radii_inner = np.concatenate([[0.0], radii[:-1]])
    radii_outer = radii

    pts, wts = np.polynomial.legendre.leggauss(n_quad)
    a, b = 0.0, np.pi / 2
    alpha = 0.5 * (b - a) * pts + 0.5 * (b + a)
    w = 0.5 * (b - a) * wts

    val = 0.0
    for k in range(n_quad):
        ca = float(np.cos(alpha[k]))
        sa = float(np.sin(alpha[k]))
        h = R * sa
        tau_2d = 0.0
        for n_reg in range(len(radii)):
            r_in = float(radii_inner[n_reg])
            r_out = float(radii_outer[n_reg])
            if h >= r_out:
                continue
            seg_outer = float(np.sqrt(max(r_out**2 - h**2, 0.0)))
            seg_inner = float(np.sqrt(max(r_in**2 - h**2, 0.0))) if h < r_in else 0.0
            chord_2d_in_annulus = 2.0 * (seg_outer - seg_inner)
            tau_2d += float(sig_t[n_reg]) * chord_2d_in_annulus
        val += w[k] * ca * Ki_n_mp(3, tau_2d, dps=dps)
    return float(4.0 / np.pi * val)


def Pss_cyl_mc(tau_R: float, n_samples: int = 500_000, seed: int = 42) -> float:
    """Monte-Carlo estimate of P_ss^cyl: independent verification.

    Sample (α, β) with PDF ∝ cos α · sin² β over (0,π/2)², then
    accumulate exp(-2τ_R cos α / sin β).
    """
    rng = np.random.default_rng(seed)
    n_accept = 0
    sum_exp = 0.0
    while n_accept < n_samples:
        batch = 200_000
        a = rng.uniform(0, np.pi / 2, batch)
        b = rng.uniform(0, np.pi / 2, batch)
        u = rng.uniform(0, 1, batch)
        # max of cos α · sin² β over (0,π/2)² is 1 (α=0, β=π/2)
        accept = u < (np.cos(a) * np.sin(b) ** 2)
        a_acc = a[accept]
        b_acc = b[accept]
        chord = 2.0 * tau_R * np.cos(a_acc) / np.sin(b_acc)
        sum_exp += float(np.exp(-chord).sum())
        n_accept += int(accept.sum())
    return sum_exp / n_accept


# ───────────────────────── Pytest assertions ─────────────────────────

def test_Ki_3_at_zero():
    """Ki_3(0) = π/4 (closed form)."""
    val = Ki_n_mp(3, 0.0)
    assert abs(val - np.pi / 4) < 1e-14


def test_pss_thin_limit():
    """τ_R → 0: P_ss → 1 (thin cell, all surface neutrons transit)."""
    for tau in [1e-6, 1e-4, 1e-2]:
        p = compute_P_ss_cylinder_homogeneous(1.0, tau)
        # Quadratic departure from 1: 1 - O(τ_R)
        assert p > 0.97 * (1 - 2 * tau), f"τ={tau}: P_ss={p}"
        assert p < 1.0


def test_pss_thick_limit():
    """τ_R → ∞: P_ss → 0 (thick cell, all absorbed)."""
    for tau in [10.0, 50.0, 100.0]:
        p = compute_P_ss_cylinder_homogeneous(1.0, tau)
        # Should be exponentially small
        assert p < 0.01, f"τ={tau}: P_ss={p}"


def test_pss_mc_agreement():
    """Quadrature P_ss^cyl matches independent MC estimate to ~3·MC stderr."""
    for tau_R in [0.1, 0.5, 1.0, 2.0]:
        p_q = compute_P_ss_cylinder_homogeneous(1.0, tau_R, n_quad=64)
        p_m = Pss_cyl_mc(tau_R, n_samples=400_000, seed=42)
        # Standard error ~ sqrt(p(1-p)/N) ~ 1e-3
        diff = abs(p_q - p_m)
        assert diff < 5e-3, (
            f"τ_R={tau_R}: P_ss quad={p_q:.6f} MC={p_m:.6f} "
            f"diff={diff:.4e} (expect <5e-3)"
        )


def test_pss_multiregion_reduces_to_homogeneous():
    """MR P_ss with all regions same σ_t equals homogeneous P_ss."""
    for tau_R in [0.5, 1.0, 2.0]:
        sig_t = 1.0
        R = tau_R
        p_homog = compute_P_ss_cylinder_homogeneous(sig_t, R, n_quad=64)
        p_mr2 = compute_P_ss_cylinder_multiregion(
            np.array([sig_t, sig_t]), np.array([R / 2, R]), n_quad=64
        )
        p_mr4 = compute_P_ss_cylinder_multiregion(
            np.array([sig_t] * 4), np.array([0.4 * R, 0.45 * R, 0.55 * R, R]),
            n_quad=64,
        )
        assert abs(p_homog - p_mr2) < 1e-12
        assert abs(p_homog - p_mr4) < 1e-12


def test_pss_multiregion_layer_order_matters():
    """MR P_ss DEPENDS on layer order: chords with impact parameter
    0.5 < h < 1.0 cross only the OUTER annulus, so swapping σ_t
    between annuli changes the optical depth on those grazing chords.

    [0.1 inner, 2.0 outer]: grazing chords accumulate τ_outer = HIGH
    [2.0 inner, 0.1 outer]: grazing chords accumulate τ_outer = LOW
    → P_ss MUCH higher in the second case (more transmission via grazing).
    """
    radii = np.array([0.5, 1.0])
    p_thin_inner = compute_P_ss_cylinder_multiregion(
        np.array([0.1, 2.0]), radii, n_quad=64,
    )
    p_thick_inner = compute_P_ss_cylinder_multiregion(
        np.array([2.0, 0.1]), radii, n_quad=64,
    )
    # Order MUST matter; thick-inner (=thin-outer at grazing) → higher P_ss
    assert p_thick_inner > 5 * p_thin_inner, (
        f"Expect strong order-dependence; got {p_thin_inner=}, "
        f"{p_thick_inner=}, ratio={p_thick_inner/p_thin_inner:.2f}"
    )


if __name__ == "__main__":
    print("=" * 78)
    print("CYLINDER P_ss derivation + verification (Issue #132)")
    print("=" * 78)

    print("\n--- Sanity: Ki_3 closed forms ---")
    print(f"  Ki_3(0) numeric   = {Ki_n_mp(3, 0.0):.12f}")
    print(f"  π/4               = {np.pi/4:.12f}")
    print(f"  Ki_3(1)           = {Ki_n_mp(3, 1.0):.12f}")

    print("\n--- Homogeneous P_ss^cyl(τ_R) ---")
    print(f"  {'τ_R':>10} {'P_ss(quad)':>14} {'P_ss(MC)':>14} {'1/(1-P_ss)':>12}")
    for tau in [1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        p_q = compute_P_ss_cylinder_homogeneous(1.0, tau)
        p_m = Pss_cyl_mc(tau, n_samples=200_000, seed=42)
        factor = 1.0 / (1.0 - p_q) if p_q < 1.0 else float('inf')
        print(f"  {tau:>10.3e} {p_q:>14.10f} {p_m:>14.10f} {factor:>12.6f}")

    print("\n--- Multi-region P_ss^cyl (radii=[0.5, 1.0]) ---")
    print(f"  {'σ_t':>14} {'P_ss(quad)':>14}")
    for sigt in [
        np.array([1.0, 1.0]),     # homogeneous reference
        np.array([0.1, 2.0]),     # thin-inner, thick-outer (Class A-ish)
        np.array([2.0, 0.1]),     # swap (chord-symmetric — must equal above)
        np.array([0.5, 0.5]),
    ]:
        p = compute_P_ss_cylinder_multiregion(sigt, np.array([0.5, 1.0]))
        print(f"  {str(sigt):>14} {p:>14.10f}")
