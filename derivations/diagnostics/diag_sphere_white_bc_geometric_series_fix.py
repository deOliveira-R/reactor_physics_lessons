"""Test the Hébert (2009) white-BC closure: add (1 - β·P_ss)⁻¹ geometric
series factor to the existing rank-1 Mark K_bc.

Per Hébert *Applied Reactor Physics* (3rd ed.) §3.8.5 Eq. (3.323):

    ℙ_white = ℙ_vac + (β⁺/(1 - β⁺·P_ss)) · P_iS · P_Sj^T

The current ORPHEUS rank-1 Mark code computes:

    K = K_vol + G_bc(r_i) · P_esc(r_j) / R²

which corresponds to ℙ_vac + (P_iS · P_Sj^T) — i.e. the rank-1 outer
product but **NOT the (1-P_ss)⁻¹ geometric series factor**. Without
the geometric-series factor, the closure underestimates the boundary
contribution by exactly the factor 1/(1-P_ss). For sphere R=1 MFP,
P_ss ≈ 0.30 so 1/(1-P_ss) ≈ 1.42 — and the rank-1 Mark gives k_eff
= 1.0957 vs k_inf = 1.5 (low by 27 %). Geometric-series factor of
1.42 would boost it substantially.

This script computes P_ss analytically for sphere with white BC,
patches K_bc to include the 1/(1-P_ss) factor, and re-runs sphere
1G/1R + 1G/2R to test if k_eff approaches k_inf.

P_ss for homogeneous sphere
---------------------------
For uniform isotropic inward partial current J⁻ = 1, the angular flux
ψ⁻(r_b, Ω) = J⁻/π = 1/π. The probability a surface neutron transits
the cell without colliding and exits the other side:

    P_ss = ∫_(half-range inward) (Ω·n) · exp(-Σ_t·chord(Ω)) dΩ
         / ∫_(half-range inward) (Ω·n) dΩ

For a surface point on the sphere with inward chord length to the
opposite side = 2R·cos θ' (where θ' = angle from inward normal):

    P_ss = (1/π) · ∫₀^(π/2) cos θ' · sin θ' · exp(-2R·Σ_t·cos θ') · 2π dθ'
         / [(1/π) · π]
         = 2 · ∫₀^(π/2) cos θ' · sin θ' · exp(-2τ_R·cos θ') dθ'
         = (1 - (1 + 2τ_R)·exp(-2τ_R)) / (2 τ_R²)

with τ_R = R·Σ_t. For τ_R = 1: P_ss = (1 - 3·e⁻²)/2 ≈ 0.297, so
1/(1-P_ss) ≈ 1.422.

For multi-region cell, P_ss generalizes by replacing 2R·Σ_t with the
piecewise-integrated optical depth across the chord (which in
spherical geometry is 2·Σ_k Σ_t,k·ℓ_k(θ') per the Hébert chord
geometry).
"""

from __future__ import annotations

import time

import numpy as np
import mpmath

from orpheus.derivations import cp_cylinder, cp_sphere
from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    build_volume_kernel,
    build_white_bc_correction,
    composite_gl_r,
    solve_peierls_mg,
)
from orpheus.derivations import peierls_geometry as _pg


_QUAD = dict(n_panels_per_region=2, p_order=3,
             n_angular=24, n_rho=24, n_surf_quad=24, dps=15)


def compute_P_ss_homogeneous_sphere(Sigma_t: float, R: float) -> float:
    """Analytical P_ss for homogeneous sphere with white BC.

    Closed form: P_ss = (1 - (1 + 2τ_R)·exp(-2τ_R)) / (2 τ_R²)
    where τ_R = Sigma_t · R.
    """
    tau_R = Sigma_t * R
    if tau_R < 1e-10:
        return 1.0  # limit τ→0: P_ss → 1
    P_ss = (1 - (1 + 2 * tau_R) * np.exp(-2 * tau_R)) / (2 * tau_R**2)
    return float(P_ss)


def compute_P_ss_multiregion_sphere(sig_t_per_region: np.ndarray,
                                     radii: np.ndarray,
                                     n_quad: int = 64) -> float:
    """Numerical P_ss for sphere with piecewise σ_t.

    The chord from a surface point inward at angle θ' (from inward normal)
    has length 2R·cos θ' total, but passes through annular regions in
    a known sequence. The optical depth along the chord:

        τ(θ') = Σ_k Σ_t,k · ℓ_k(θ')

    where ℓ_k(θ') is the chord-segment in annulus k. For sphere with
    inner radii [r_0, r_1, ..., r_n=R], a chord from the outer surface
    at angle θ' from the inward normal:
    - if cos θ' > sqrt(1 - (r_{k-1}/R)²) the chord doesn't reach annulus
      below r_{k-1}; it stays in annulus k (outer)
    - in general, the impact parameter h = R·sin θ', and the chord
      crosses annulus boundary r_k iff h < r_k

    P_ss = 2 · ∫₀^(π/2) cos θ' · sin θ' · exp(-τ(θ')) dθ'
    """
    R = float(radii[-1])
    radii_inner = np.concatenate([[0.0], radii[:-1]])  # inner of each annulus
    radii_outer = radii  # outer of each annulus

    # GL quadrature in θ'
    theta_pts, theta_wts = np.polynomial.legendre.leggauss(n_quad)
    # Map [-1, 1] → [0, π/2]
    theta_pts_mapped = 0.5 * (theta_pts + 1) * np.pi / 2
    theta_wts_mapped = theta_wts * np.pi / 4

    P_ss = 0.0
    for k in range(n_quad):
        theta = theta_pts_mapped[k]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        h = R * sin_theta  # impact parameter

        # Compute optical depth along chord
        # The chord passes through annulus k iff r_{k-1} < h < r_k OR h < r_{k-1}
        # For each annulus that the chord traverses, the chord length in that
        # annulus is 2 · sqrt(r_outer^2 - h^2) - 2 · sqrt(r_inner^2 - h^2)
        # if the chord stays inside; if h > r_inner the chord doesn't enter
        tau = 0.0
        for n_reg in range(len(radii)):
            r_in = radii_inner[n_reg]
            r_out = radii_outer[n_reg]
            if h >= r_out:
                # chord doesn't enter this annulus
                continue
            # chord segment in annulus n_reg: from r_out to max(r_in, h)
            seg_outer = np.sqrt(max(r_out**2 - h**2, 0))
            seg_inner = np.sqrt(max(r_in**2 - h**2, 0)) if h < r_in else 0.0
            chord_in_annulus = 2 * (seg_outer - seg_inner)
            tau += sig_t_per_region[n_reg] * chord_in_annulus

        P_ss += theta_wts_mapped[k] * cos_theta * sin_theta * np.exp(-tau)

    P_ss = 2 * P_ss
    return float(P_ss)


def _build_xs(ng_key, n_regions):
    layout = LAYOUTS[n_regions]
    xs_list = [get_xs(region, ng_key) for region in layout]
    return (
        np.vstack([xs["sig_t"] for xs in xs_list]),
        np.stack([xs["sig_s"] for xs in xs_list], axis=0),
        np.vstack([xs["nu"] * xs["sig_f"] for xs in xs_list]),
        np.vstack([xs["chi"] for xs in xs_list]),
    )


def _kinf(ng_key, n_regions):
    return cp_sphere._build_case(ng_key, n_regions).k_inf


def _solve_with_geometric_series_patch(
    ng_key: str, n_regions: int, *, P_ss_per_group: list[float] | None = None,
):
    """Patched solve_peierls_mg with K_bc → K_bc / (1 - P_ss) for each group.

    P_ss_per_group is a list of (n_groups,) — the surface-to-surface
    probability per group. If None, computed analytically per group.
    """
    cp_module = cp_sphere
    sig_t, sig_s, nu_sig_f, chi = _build_xs(ng_key, n_regions)
    radii = np.array(cp_module._RADII[n_regions])
    R_cell = float(radii[-1])

    # Patch the actual code path used by solve_peierls_mg via
    # _build_full_K_per_group → build_white_bc_correction_rank_n.
    original_build = _pg.build_white_bc_correction_rank_n

    n_groups = sig_t.shape[1]
    if P_ss_per_group is None:
        P_ss_per_group = [
            compute_P_ss_multiregion_sphere(sig_t[:, g], radii)
            for g in range(n_groups)
        ]
    P_ss_iter = iter(P_ss_per_group)

    def patched_build(geometry, r_nodes, r_wts, radii_arg, sig_t_arg,
                     n_angular=32, n_surf_quad=32, dps=25, n_bc_modes=1):
        K_bc = original_build(
            geometry, r_nodes, r_wts, radii_arg, sig_t_arg,
            n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
            n_bc_modes=n_bc_modes,
        )
        P_ss = next(P_ss_iter)
        factor = 1.0 / (1.0 - P_ss)
        return K_bc * factor

    _pg.build_white_bc_correction_rank_n = patched_build
    try:
        sol = solve_peierls_mg(
            SPHERE_1D, radii=radii, sig_t=sig_t, sig_s=sig_s,
            nu_sig_f=nu_sig_f, chi=chi,
            boundary="white_rank1_mark", n_bc_modes=1,
            **_QUAD,
        )
    finally:
        _pg.build_white_bc_correction_rank_n = original_build

    return sol.k_eff


def _solve_unpatched(ng_key: str, n_regions: int):
    """Run with the existing rank-1 Mark closure — the BEFORE baseline."""
    sig_t, sig_s, nu_sig_f, chi = _build_xs(ng_key, n_regions)
    radii = np.array(cp_sphere._RADII[n_regions])
    sol = solve_peierls_mg(
        SPHERE_1D, radii=radii, sig_t=sig_t, sig_s=sig_s,
        nu_sig_f=nu_sig_f, chi=chi,
        boundary="white_rank1_mark", n_bc_modes=1,
        **_QUAD,
    )
    return sol.k_eff


def main():
    print("=" * 76)
    print("Hébert (2009) white-BC closure: add (1-P_ss)⁻¹ geometric series factor")
    print("=" * 76)

    print("\n--- P_ss verification (homogeneous sphere) ---")
    print(f"  {'σ_t':>6} {'R':>6} {'τ_R':>6} {'P_ss analytic':>16} {'P_ss numeric':>16} {'1/(1-P_ss)':>12}")
    for sig_t_val, R_val in [(1.0, 1.0), (1.0, 2.0), (1.0, 5.0), (5.0, 1.0), (10.0, 1.0)]:
        P_ss_a = compute_P_ss_homogeneous_sphere(sig_t_val, R_val)
        P_ss_n = compute_P_ss_multiregion_sphere(np.array([sig_t_val]), np.array([R_val]))
        factor = 1.0 / (1.0 - P_ss_a)
        print(f"  {sig_t_val:>6.2f} {R_val:>6.2f} {sig_t_val*R_val:>6.2f} "
              f"{P_ss_a:>16.10f} {P_ss_n:>16.10f} {factor:>12.6f}")

    print("\n" + "=" * 76)
    print("Apply (1-P_ss)⁻¹ geometric series fix to sphere k_eff")
    print("=" * 76)

    cases = [
        ("1g", 1),
        ("1g", 2),
        ("2g", 1),
        ("2g", 2),
    ]

    for ng_key, n_regions in cases:
        kinf = _kinf(ng_key, n_regions)
        print(f"\n--- {ng_key} {n_regions}r — k_inf = {kinf:.6f} ---")

        t0 = time.time()
        k_before = _solve_unpatched(ng_key, n_regions)
        t_before = time.time() - t0
        err_before = (k_before - kinf) / kinf * 100

        # Compute P_ss per group
        sig_t, _, _, _ = _build_xs(ng_key, n_regions)
        radii = np.array(cp_sphere._RADII[n_regions])
        P_ss_per_g = [
            compute_P_ss_multiregion_sphere(sig_t[:, g], radii)
            for g in range(sig_t.shape[1])
        ]

        t0 = time.time()
        k_after = _solve_with_geometric_series_patch(
            ng_key, n_regions, P_ss_per_group=P_ss_per_g,
        )
        t_after = time.time() - t0
        err_after = (k_after - kinf) / kinf * 100

        print(f"  P_ss per group: {[f'{p:.6f}' for p in P_ss_per_g]}")
        print(f"  factor 1/(1-P_ss): {[f'{1/(1-p):.4f}' for p in P_ss_per_g]}")
        print(f"  BEFORE: k_eff = {k_before:.10f}  err = {err_before:+.3f}%  ({t_before:.1f}s)")
        print(f"  AFTER:  k_eff = {k_after:.10f}  err = {err_after:+.3f}%  ({t_after:.1f}s)")


if __name__ == "__main__":
    main()
