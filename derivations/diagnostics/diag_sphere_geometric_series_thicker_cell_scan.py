"""Verify the geometric-series white-BC fix at thicker cells.

If the +10% overshoot on 1G/2R sphere is the legitimate pointwise-vs-
flat-flux difference (pointwise allows fuel-region flux peaking that
flat-flux CP averages out), then at thicker cells (where flux flattens
from absorption/scattering equilibrium) the residual should drop to <1 %.

If the residual STAYS at +10% across cell thicknesses, the fix has a
structural error (perhaps in P_ss for the multi-region case, or in how
the geometric-series factor interacts with a non-uniform σ_t profile).
"""

from __future__ import annotations

import time

import numpy as np

from orpheus.derivations import cp_sphere
from orpheus.derivations._xs_library import LAYOUTS, get_xs
from orpheus.derivations.peierls_geometry import SPHERE_1D, solve_peierls_mg
from orpheus.derivations import peierls_geometry as _pg


_QUAD = dict(n_panels_per_region=2, p_order=3,
             n_angular=24, n_rho=24, n_surf_quad=24, dps=15)


def _compute_P_ss_mr(sig_t_per_region, radii, n_quad=64):
    R = float(radii[-1])
    radii_inner = np.concatenate([[0.0], radii[:-1]])
    radii_outer = radii
    theta_pts, theta_wts = np.polynomial.legendre.leggauss(n_quad)
    theta_pts_mapped = 0.5 * (theta_pts + 1) * np.pi / 2
    theta_wts_mapped = theta_wts * np.pi / 4
    P_ss = 0.0
    for k in range(n_quad):
        theta = theta_pts_mapped[k]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        h = R * sin_theta
        tau = 0.0
        for n_reg in range(len(radii)):
            r_in = radii_inner[n_reg]
            r_out = radii_outer[n_reg]
            if h >= r_out:
                continue
            seg_outer = np.sqrt(max(r_out**2 - h**2, 0))
            seg_inner = np.sqrt(max(r_in**2 - h**2, 0)) if h < r_in else 0.0
            chord_in_annulus = 2 * (seg_outer - seg_inner)
            tau += sig_t_per_region[n_reg] * chord_in_annulus
        P_ss += theta_wts_mapped[k] * cos_theta * sin_theta * np.exp(-tau)
    return float(2 * P_ss)


def _solve_with_fix(ng_key, n_regions, *, R_scale=1.0):
    """Run sphere solver with the (1-P_ss)^-1 fix; optionally scale all radii by R_scale."""
    sig_t, sig_s, nu_sig_f, chi = (
        np.vstack([get_xs(r, ng_key)["sig_t"] for r in LAYOUTS[n_regions]]),
        np.stack([get_xs(r, ng_key)["sig_s"] for r in LAYOUTS[n_regions]], axis=0),
        np.vstack([get_xs(r, ng_key)["nu"] * get_xs(r, ng_key)["sig_f"]
                   for r in LAYOUTS[n_regions]]),
        np.vstack([get_xs(r, ng_key)["chi"] for r in LAYOUTS[n_regions]]),
    )
    radii = np.array(cp_sphere._RADII[n_regions]) * R_scale

    P_ss_per_g = [
        _compute_P_ss_mr(sig_t[:, g], radii)
        for g in range(sig_t.shape[1])
    ]
    P_ss_iter = iter(P_ss_per_g)

    original = _pg.build_white_bc_correction_rank_n

    def patched(geometry, r_nodes, r_wts, radii_arg, sig_t_arg,
                n_angular=32, n_surf_quad=32, dps=25, n_bc_modes=1):
        K_bc = original(
            geometry, r_nodes, r_wts, radii_arg, sig_t_arg,
            n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
            n_bc_modes=n_bc_modes,
        )
        return K_bc / (1 - next(P_ss_iter))

    _pg.build_white_bc_correction_rank_n = patched
    try:
        sol = solve_peierls_mg(
            SPHERE_1D, radii=radii, sig_t=sig_t, sig_s=sig_s,
            nu_sig_f=nu_sig_f, chi=chi,
            boundary="white_rank1_mark", n_bc_modes=1,
            **_QUAD,
        )
    finally:
        _pg.build_white_bc_correction_rank_n = original
    return sol.k_eff, P_ss_per_g


def _solve_unpatched(ng_key, n_regions, *, R_scale=1.0):
    sig_t, sig_s, nu_sig_f, chi = (
        np.vstack([get_xs(r, ng_key)["sig_t"] for r in LAYOUTS[n_regions]]),
        np.stack([get_xs(r, ng_key)["sig_s"] for r in LAYOUTS[n_regions]], axis=0),
        np.vstack([get_xs(r, ng_key)["nu"] * get_xs(r, ng_key)["sig_f"]
                   for r in LAYOUTS[n_regions]]),
        np.vstack([get_xs(r, ng_key)["chi"] for r in LAYOUTS[n_regions]]),
    )
    radii = np.array(cp_sphere._RADII[n_regions]) * R_scale
    sol = solve_peierls_mg(
        SPHERE_1D, radii=radii, sig_t=sig_t, sig_s=sig_s,
        nu_sig_f=nu_sig_f, chi=chi,
        boundary="white_rank1_mark", n_bc_modes=1,
        **_QUAD,
    )
    return sol.k_eff


def main():
    print("=" * 76)
    print("Geometric-series fix: thicker-cell scan to characterize residual")
    print("=" * 76)
    print()
    print("  Note: cp_sphere k_inf depends only on materials, not on R_scale,")
    print("  because R_scale just multiplies all σ_t·R uniformly (which is")
    print("  what determines k_inf via the CP_inf integrals).")
    print("  So we expect k_eff(R_scale) → cp k_inf as R_scale → ∞ (thick cell")
    print("  homogenises the flux structure, eliminating pointwise advantage).")

    # Pre-compute k_inf reference values
    cases = [("1g", 1), ("1g", 2), ("2g", 1), ("2g", 2)]
    k_inf_ref = {(ng, nr): cp_sphere._build_case(ng, nr).k_inf for ng, nr in cases}

    for ng_key, n_regions in cases:
        kinf = k_inf_ref[(ng_key, n_regions)]
        print(f"\n--- {ng_key} {n_regions}r — cp k_inf = {kinf:.6f} ---")
        print(f"  {'R_scale':>8} {'k_FIX':>14} {'err':>9} {'P_ss':>16}")
        for R_scale in [1.0, 2.0, 5.0, 10.0]:
            k_after, P_ss = _solve_with_fix(ng_key, n_regions, R_scale=R_scale)
            err = (k_after - kinf) / kinf * 100
            P_ss_str = ",".join(f"{p:.4f}" for p in P_ss)
            print(
                f"  {R_scale:>8.2f} {k_after:>14.10f} {err:>+8.3f}% "
                f"  P_ss=[{P_ss_str}]"
            )


if __name__ == "__main__":
    main()
