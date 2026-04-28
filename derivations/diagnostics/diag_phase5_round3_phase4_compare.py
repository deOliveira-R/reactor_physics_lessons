"""Round 3 — Compare A1/A2 K_bc per-pair off-diagonal entries to Phase 4
matrix-Galerkin K_bc.

If the off-diagonal entries match Phase 4 closely, then the only
question is "what diagonal does Phase 4 produce that makes k_eff
right". That tells us what value the continuous-µ diagonal SHOULD
be regularised to.
"""
from __future__ import annotations

import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    composite_gl_r,
)

from derivations.diagnostics.diag_phase5_round3_adaptive_quadrature import (
    K_INF, NU_SIG_F, R_THIN, SIG_S, SIG_T,
    compute_K_bc_chord_substitution_half_M1,
    compute_K_bc_per_pair_half_M1,
    keff_from_K,
)


def test_compare_offdiag_to_phase4(capsys):
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 1, 3, dps=15, inner_radius=0.0,
        )
        N = len(r_nodes)
        sig_s_arr = np.full(N, SIG_S)
        nu_sf_arr = np.full(N, NU_SIG_F)
        sig_t_arr = np.full(N, SIG_T)

        # Phase 4 reference: closure='specular_multibounce' at rank-3
        K_p4 = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "specular_multibounce",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=3, dps=20,
        )
        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=3, dps=20,
        )
        K_bc_p4 = K_p4 - K_vol  # extract the BC piece

        # A1 per-pair K_bc (no regularisation)
        K_bc_a1 = compute_K_bc_per_pair_half_M1(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=128,
        )
        # A2 chord substitution K_bc
        K_bc_a2 = compute_K_bc_chord_substitution_half_M1(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=128,
        )

        print(f"\n=== Compare K_bc forms ===")
        print(f"K_bc Phase 4 (specular_mb rank-3):")
        print(K_bc_p4)
        print(f"\nK_bc A1 (per-pair half-M1, Q=128):")
        print(K_bc_a1)
        print(f"\nK_bc A2 (chord subst, Q=128):")
        print(K_bc_a2)
        print(f"\nRatio Phase4 / A2:")
        with np.errstate(divide='ignore', invalid='ignore'):
            print(np.where(np.abs(K_bc_a2) > 1e-15,
                           K_bc_p4 / K_bc_a2, np.nan))


def test_compare_offdiag_only_keff(capsys):
    """If we use Phase 4 diagonal but A2 off-diagonals, do we recover
    Phase 4's k_eff?

    Decomposition: K_bc = K_diag + K_offdiag. If we substitute Phase 4's
    diagonal but use A2's off-diagonal (which is exact to machine
    precision), we can isolate "is the diagonal alone the issue".
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        N = len(r_nodes)
        sig_s_arr = np.full(N, SIG_S)
        nu_sf_arr = np.full(N, NU_SIG_F)
        sig_t_arr = np.full(N, SIG_T)

        K_p4 = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "specular_multibounce",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=3, dps=20,
        )
        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=3, dps=20,
        )
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )

        K_bc_p4 = K_p4 - K_vol
        K_bc_a2 = compute_K_bc_chord_substitution_half_M1(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=128,
        )

        k_p4 = keff_from_K(K_p4, sig_t_arr, nu_sf_arr, sig_s_arr)
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)
        # Hybrid: use Phase 4 diagonal but A2 off-diagonals
        K_bc_hybrid = K_bc_a2.copy()
        for i in range(N):
            K_bc_hybrid[i, i] = K_bc_p4[i, i]
        K_hybrid = K_vol + K_bc_hybrid
        k_hybrid = keff_from_K(K_hybrid, sig_t_arr, nu_sf_arr, sig_s_arr)

        print(f"\n=== Hybrid: A2 off-diag + Phase4 diag ===")
        print(f"  k_inf:   {K_INF:.6f}")
        print(f"  k_heb:   {k_heb:.6f} ({(k_heb-K_INF)/K_INF*100:+.4f}%)")
        print(f"  k_p4:    {k_p4:.6f} ({(k_p4-K_INF)/K_INF*100:+.4f}%)")
        print(f"  k_hybrid: {k_hybrid:.6f} ({(k_hybrid-K_INF)/K_INF*100:+.4f}%)")
        print(f"  N_r = {N}, r_nodes = {r_nodes[:5]}...{r_nodes[-1]}")
        print(f"\nDiagonal comparison (first 5):")
        print(f"  Phase4 diag: {np.diag(K_bc_p4)[:5]}")
        print(f"  A2 diag:     {np.diag(K_bc_a2)[:5]}")
