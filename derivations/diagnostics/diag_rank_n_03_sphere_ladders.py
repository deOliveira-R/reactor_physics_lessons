"""Diagnostic: sphere k_eff convergence ladder with various normalization choices.

Tests multiple hypotheses for the sphere mode-n normalization.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspaces/ORPHEUS")

import numpy as np

from orpheus.derivations.peierls_geometry import (
    CYLINDER_1D, SPHERE_1D,
    compute_G_bc, compute_G_bc_mode,
    compute_P_esc, compute_P_esc_mode,
    composite_gl_r,
    build_volume_kernel, build_white_bc_correction,
)


_SIG_T = np.array([1.0])
_SIG_S = np.array([0.5])
_NU_SIG_F = np.array([0.75])
_K_INF = 1.5


def solve_with_custom_Kbc(geometry, R, K_bc_fn,
                          n_panels_per_region=2, p_order=5,
                          n_angular=32, n_rho=32, n_surf_quad=32, dps=25,
                          max_iter=300, tol=1e-10):
    """Solve 1-G Peierls with a user-supplied K_bc matrix."""
    radii = np.array([R])
    r_nodes, r_wts, panels = composite_gl_r(radii, n_panels_per_region, p_order, dps=dps)
    K_vol = build_volume_kernel(
        geometry, r_nodes, panels, radii, _SIG_T,
        n_angular=n_angular, n_rho=n_rho, dps=dps,
    )
    K_bc = K_bc_fn(geometry, r_nodes, r_wts, radii, _SIG_T,
                   n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps)
    K = K_vol + K_bc

    N = len(r_nodes)
    sig_t_n = np.full(N, _SIG_T[0])
    sig_s_n = np.full(N, _SIG_S[0])
    nu_sig_f_n = np.full(N, _NU_SIG_F[0])

    A = np.diag(sig_t_n) - K * sig_s_n[np.newaxis, :]
    B = K * nu_sig_f_n[np.newaxis, :]

    phi = np.ones(N)
    k_val = 1.0
    B_phi = B @ phi
    prod_old = np.abs(B_phi).sum()
    for it in range(max_iter):
        q = B_phi / k_val
        phi_new = np.linalg.solve(A, q)
        B_phi_new = B @ phi_new
        prod_new = np.abs(B_phi_new).sum()
        k_new = k_val * prod_new / prod_old if prod_old > 0 else k_val
        nrm = np.abs(phi_new).sum()
        if nrm > 0:
            phi_new /= nrm
        B_phi_norm = B @ phi_new
        prod_norm = np.abs(B_phi_norm).sum()
        converged = abs(k_new - k_val) < tol and it > 5
        phi, k_val = phi_new, k_new
        B_phi, prod_old = B_phi_norm, prod_norm
        if converged:
            break
    return k_val


def make_rank_n_builder(n_bc_modes, normalization="original"):
    """Return a K_bc builder function with a specific normalization strategy."""
    def builder(geometry, r_nodes, r_wts, radii, sig_t,
                n_angular, n_surf_quad, dps):
        R = float(radii[-1])
        N = len(r_nodes)
        # Mode 0 = existing rank-1
        K = build_white_bc_correction(
            geometry, r_nodes, r_wts, radii, sig_t,
            n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
        )
        if n_bc_modes == 1:
            return K

        sig_t_n = np.array([sig_t[geometry.which_annulus(ri, radii)] for ri in r_nodes])
        rv = np.array([geometry.radial_volume_weight(rj) for rj in r_nodes])
        divisor = geometry.rank1_surface_divisor(R)

        for n_mode in range(1, n_bc_modes):
            P_n = compute_P_esc_mode(geometry, r_nodes, radii, sig_t, n_mode,
                                     n_angular=n_angular, dps=dps)
            G_n = compute_G_bc_mode(geometry, r_nodes, radii, sig_t, n_mode,
                                    n_surf_quad=n_surf_quad, dps=dps)

            if normalization == "original":
                u_n = sig_t_n * G_n / divisor
                v_n = (2 * n_mode + 1) * rv * r_wts * P_n
            elif normalization == "no_2np1":
                u_n = sig_t_n * G_n / divisor
                v_n = rv * r_wts * P_n
            elif normalization == "2np1_on_u":
                u_n = (2 * n_mode + 1) * sig_t_n * G_n / divisor
                v_n = rv * r_wts * P_n
            elif normalization == "no_2np1_G_from_P":
                # Sphere: G^(n) = 4 P^(n) — so u[i] = 4 * Σ_t(i) * P^(n)(i) / divisor
                # (numerically identical to "no_2np1" for sphere; different for cyl)
                u_n = sig_t_n * G_n / divisor
                v_n = rv * r_wts * P_n
            elif normalization == "4_P_instead_of_G":
                # Use 4·P_n as the surface-to-volume response (sphere only).
                u_n = sig_t_n * (4.0 * P_n) / divisor
                v_n = (2 * n_mode + 1) * rv * r_wts * P_n
            else:
                raise ValueError(normalization)
            K = K + np.outer(u_n, v_n)
        return K
    return builder


def run_ladder(geometry, R, n_values, normalization, label=""):
    """Run eigenvalue ladder for various n_bc_modes."""
    header = f"{geometry.kind} R={R} MFP norm={normalization}{(' ' + label) if label else ''}"
    print(f"\n{header}")
    print("  N  k_eff        |err/k_inf|")
    prev_err = None
    for N in n_values:
        K_bc = make_rank_n_builder(N, normalization)
        k = solve_with_custom_Kbc(geometry, R, K_bc, n_angular=24, n_rho=24, n_surf_quad=24)
        err = abs(k - _K_INF) / _K_INF
        ratio_str = ""
        if prev_err is not None and prev_err > 0:
            ratio_str = f"  (ratio vs prev: {err/prev_err:.3f})"
        print(f"  {N}  {k:.6f}  {err*100:+.3f}%{ratio_str}")
        prev_err = err


if __name__ == "__main__":
    # Test the SPHERE at thin and thick R with all normalizations.
    n_values = [1, 2, 3, 4, 5]

    for R in [1.0, 10.0]:
        run_ladder(SPHERE_1D, R, n_values, "original")
        run_ladder(SPHERE_1D, R, n_values, "no_2np1")
        run_ladder(SPHERE_1D, R, n_values, "2np1_on_u")
