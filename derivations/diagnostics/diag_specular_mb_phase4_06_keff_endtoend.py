"""Diagnostic: end-to-end k_eff for sphere/cyl/slab specular_multibounce.

Created by numerics-investigator on 2026-04-28.

Verifies the resolvent-norm conclusion against the actual k_eff
produced by the multi-bounce closure across N. To save time we build
K_vol once per geometry (slow mpmath) and evaluate K_bc^bare and
K_bc^MB analytically and add to the cached K_vol.

Test fixture (matches investigation request):
  fuel-A-like 1G: σ_t=0.5, σ_s=0.38, νσ_f=0.025, k_inf=0.20833...
  thin: R/L=5, σ_t=0.5 ⇒ τ_R/τ_L = 2.5
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SLAB_POLAR_1D,
    SPHERE_1D,
    CurvilinearGeometry,
    _build_full_K_per_group,
    _shifted_legendre_eval,
    _shifted_legendre_monomial_coefs,
    _slab_E_n,
    _slab_tau_to_inner_face,
    _slab_tau_to_outer_face,
    composite_gl_r,
    compute_G_bc_cylinder_3d_mode,
    compute_P_esc_cylinder_3d_mode,
    reflection_specular,
)

from derivations.diagnostics.diag_specular_mb_phase4_03_pathology_resolvent import (
    build_T_cyl, build_T_slab,
)

CYL_1D = CurvilinearGeometry(kind="cylinder-1d")

# fuel-A-like XS, k_inf = 0.025 / 0.12 = 0.208333...
SIG_T = 0.5
SIG_S = 0.38
NU_SIG_F = 0.025
K_INF = NU_SIG_F / (SIG_T - SIG_S)


def keff_from_K(K, sig_t_array, nu_sig_f_array, sig_s_array):
    """k_eff via dense generalized eigenproblem."""
    A = np.diag(sig_t_array) - K * sig_s_array[None, :]
    B = K * nu_sig_f_array[None, :]
    M = np.linalg.solve(A, B)
    eigval = np.linalg.eigvals(M)
    return float(np.max(np.real(eigval)))


# ─── Cylinder per-mode P/G (uses 3-D Knyazev primitives) ──────────────


def _build_cyl_PG(R_cell, sig_t_g, r_nodes, r_wts, N_modes, *,
                   n_angular=12, n_surf_quad=12, dps=15):
    radii = np.array([R_cell])
    sig_t_n = np.array([
        sig_t_g[CYL_1D.which_annulus(float(r), radii)] for r in r_nodes
    ])
    rv = np.array([CYL_1D.radial_volume_weight(float(r)) for r in r_nodes])
    divisor = CYL_1D.rank1_surface_divisor(R_cell)
    N_r = len(r_nodes)
    P = np.zeros((N_modes, N_r))
    G = np.zeros((N_r, N_modes))
    for n in range(N_modes):
        P_esc_n = compute_P_esc_cylinder_3d_mode(
            CYL_1D, r_nodes, radii, sig_t_g, n,
            n_angular=n_angular, dps=dps,
        )
        G_bc_n = compute_G_bc_cylinder_3d_mode(
            CYL_1D, r_nodes, radii, sig_t_g, n,
            n_surf_quad=n_surf_quad, dps=dps,
        )
        P[n, :] = rv * r_wts * P_esc_n
        G[:, n] = sig_t_n * G_bc_n / divisor
    return P, G, sig_t_n


# ─── Slab per-face P/G (uses closed-form E_(k+2)) ─────────────────────


def _build_slab_PG_perface(L_cell, sig_t, r_nodes, r_wts, N_modes):
    radii = np.array([L_cell])
    sig_t_g = np.array([sig_t])
    sig_t_n = np.array([
        sig_t_g[SLAB_POLAR_1D.which_annulus(float(r), radii)] for r in r_nodes
    ])
    rv = np.array([SLAB_POLAR_1D.radial_volume_weight(float(r)) for r in r_nodes])
    N_r = len(r_nodes)
    P_o = np.zeros((N_modes, N_r))
    P_i = np.zeros((N_modes, N_r))
    G_o = np.zeros((N_r, N_modes))
    G_i = np.zeros((N_r, N_modes))
    for i in range(N_r):
        x_i = float(r_nodes[i])
        tau_o = _slab_tau_to_outer_face(x_i, radii, sig_t_g)
        tau_n = _slab_tau_to_inner_face(x_i, radii, sig_t_g)
        for n in range(N_modes):
            coefs = _shifted_legendre_monomial_coefs(n)
            Po = Pn = Go = Gn = 0.0
            for k, c in enumerate(coefs):
                if c == 0.0:
                    continue
                E_o = _slab_E_n(k + 2, tau_o)
                E_n_val = _slab_E_n(k + 2, tau_n)
                Po += 0.5 * c * E_o
                Pn += 0.5 * c * E_n_val
                Go += 2.0 * c * E_o
                Gn += 2.0 * c * E_n_val
            P_o[n, i] = Po
            P_i[n, i] = Pn
            G_o[i, n] = Go
            G_i[i, n] = Gn
    DIV = 1.0
    P_o_w = rv * r_wts * P_o
    P_i_w = rv * r_wts * P_i
    G_o_w = sig_t_n[:, None] * G_o / DIV
    G_i_w = sig_t_n[:, None] * G_i / DIV
    P_slab = np.vstack([P_o_w, P_i_w])
    G_slab = np.hstack([G_o_w, G_i_w])
    return P_slab, G_slab, sig_t_n


# ─── Headline k_eff sweeps ────────────────────────────────────────────


def test_keff_sphere_baseline(capsys):
    """Sphere via existing implementation — replicates the documented
    overshoot at N ≥ 4."""
    with capsys.disabled():
        R = 5.0
        radii = np.array([R])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        print(f"\n=== SPHERE thin (R={R}, τ_R={SIG_T*R}, k_inf={K_INF:.6f}) ===")
        print(f"  N |  k_eff (bare)     | k_eff (MB)        | rel_diff_MB(%)")
        for N in (1, 2, 3, 4, 6, 8):
            K_bare = _build_full_K_per_group(
                SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
                "specular",
                n_angular=24, n_rho=24, n_surf_quad=24,
                n_bc_modes=N, dps=20,
            )
            K_mb = _build_full_K_per_group(
                SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
                "specular_multibounce",
                n_angular=24, n_rho=24, n_surf_quad=24,
                n_bc_modes=N, dps=20,
            )
            k_bare = keff_from_K(K_bare, sig_t_arr, nu_sf_arr, sig_s_arr)
            k_mb = keff_from_K(K_mb, sig_t_arr, nu_sf_arr, sig_s_arr)
            rel_bare = (k_bare - K_INF) / K_INF * 100
            rel_mb = (k_mb - K_INF) / K_INF * 100
            print(f"  {N} |  {k_bare:.6f} ({rel_bare:+.3f}%) | "
                  f"{k_mb:.6f} ({rel_mb:+.3f}%) | {rel_mb:+.3f}")


def test_keff_cyl_endtoend_low_N(capsys):
    """Cylinder — multi-bounce k_eff. Time budget tight, restrict to
    low N."""
    with capsys.disabled():
        R = 5.0
        radii = np.array([R])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 1, 5, dps=15, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        # Build K_vol ONCE: subtract the rank-1 K_bc^bare from the
        # _build_full_K specular path to extract K_vol.
        # The K_vol is the same for every N (it doesn't depend on N).
        K_full_n1 = _build_full_K_per_group(
            CYL_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "specular",
            n_angular=12, n_rho=12, n_surf_quad=12,
            n_bc_modes=1, dps=15,
        )
        P1, G1, _ = _build_cyl_PG(R, sig_t_g, r_nodes, r_wts, 1,
                                    n_angular=12, n_surf_quad=12, dps=15)
        R1 = reflection_specular(1)
        K_bc_bare_n1 = G1 @ R1 @ P1
        K_vol = K_full_n1 - K_bc_bare_n1

        print(f"\n=== CYL thin (R={R}, τ_R={SIG_T*R}, k_inf={K_INF:.6f}) ===")
        print(f"  N |  k_eff (bare)      | k_eff (MB)         | rel_MB(%)")
        for N in (1, 2, 3, 4, 6, 8, 12):
            P, G, _ = _build_cyl_PG(R, sig_t_g, r_nodes, r_wts, N,
                                     n_angular=12, n_surf_quad=12, dps=15)
            R_op = reflection_specular(N)
            K_bare = K_vol + G @ R_op @ P
            T = build_T_cyl(SIG_T, R, N)
            ITR = np.eye(N) - T @ R_op
            try:
                K_mb = K_vol + G @ R_op @ np.linalg.solve(ITR, P)
                k_bare = keff_from_K(K_bare, sig_t_arr, nu_sf_arr, sig_s_arr)
                k_mb = keff_from_K(K_mb, sig_t_arr, nu_sf_arr, sig_s_arr)
                rb = (k_bare - K_INF) / K_INF * 100
                rm = (k_mb - K_INF) / K_INF * 100
                print(f"  {N:2d}|  {k_bare:.6f} ({rb:+.3f}%) | "
                      f"{k_mb:.6f} ({rm:+.3f}%) | {rm:+.3f}")
            except np.linalg.LinAlgError as e:
                print(f"  N={N}: SOLVE FAIL {e}")


def test_keff_slab_endtoend(capsys):
    """Slab — multi-bounce k_eff."""
    with capsys.disabled():
        L = 5.0
        radii = np.array([L])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 1, 5, dps=15, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        # K_vol from rank-1 specular (slab K_vol independent of N).
        K_full_n1 = _build_full_K_per_group(
            SLAB_POLAR_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "specular",
            n_angular=12, n_rho=12, n_surf_quad=12,
            n_bc_modes=1, dps=15,
        )
        P1, G1, _ = _build_slab_PG_perface(L, SIG_T, r_nodes, r_wts, 1)
        R1_face = reflection_specular(1)
        R1_slab = np.zeros((2, 2))
        R1_slab[:1, :1] = R1_face
        R1_slab[1:, 1:] = R1_face
        K_bc_bare_n1 = G1 @ R1_slab @ P1
        K_vol = K_full_n1 - K_bc_bare_n1

        print(f"\n=== SLAB thin (L={L}, τ_L={SIG_T*L}, k_inf={K_INF:.6f}) ===")
        print(f"  N |  k_eff (bare)      | k_eff (MB)         | rel_MB(%)")
        for N in (1, 2, 3, 4, 6, 8, 12, 16, 20):
            P, G, _ = _build_slab_PG_perface(L, SIG_T, r_nodes, r_wts, N)
            R_face = reflection_specular(N)
            R_slab = np.zeros((2 * N, 2 * N))
            R_slab[:N, :N] = R_face
            R_slab[N:, N:] = R_face
            K_bare = K_vol + G @ R_slab @ P
            T = build_T_slab(SIG_T, L, N)
            ITR = np.eye(2 * N) - T @ R_slab
            try:
                K_mb = K_vol + G @ R_slab @ np.linalg.solve(ITR, P)
                k_bare = keff_from_K(K_bare, sig_t_arr, nu_sf_arr, sig_s_arr)
                k_mb = keff_from_K(K_mb, sig_t_arr, nu_sf_arr, sig_s_arr)
                rb = (k_bare - K_INF) / K_INF * 100
                rm = (k_mb - K_INF) / K_INF * 100
                print(f"  {N:2d}|  {k_bare:.6f} ({rb:+.3f}%) | "
                      f"{k_mb:.6f} ({rm:+.3f}%) | {rm:+.3f}")
            except np.linalg.LinAlgError as e:
                print(f"  N={N}: SOLVE FAIL {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
