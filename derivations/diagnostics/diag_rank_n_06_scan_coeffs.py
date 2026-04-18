"""Scan of rank-N normalization coefficients for sphere.

We try K_bc[i,j] = Σ_n α_n · Σ_t · g_n(r_i) · P_esc^(n)(r_j) · r_j² w_j / R²
for various α_n sequences and observe the convergence ladder at R=1 and R=10.

The correct α_n should:
- make α_0 = 1 (rank-1 match)
- reduce k_eff error monotonically as N increases
- not perturb k_eff at thick R (where rank-1 is already accurate)
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspaces/ORPHEUS")

import numpy as np

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, composite_gl_r,
    compute_G_bc_mode, compute_P_esc_mode,
    build_volume_kernel, build_white_bc_correction,
)


_SIG_T = np.array([1.0])
_SIG_S = np.array([0.5])
_NU_SIG_F = np.array([0.75])
_K_INF = 1.5


def solve_eig(K, N):
    sig_t_n = np.full(N, _SIG_T[0])
    sig_s_n = np.full(N, _SIG_S[0])
    nu_sig_f_n = np.full(N, _NU_SIG_F[0])
    A = np.diag(sig_t_n) - K * sig_s_n[np.newaxis, :]
    B = K * nu_sig_f_n[np.newaxis, :]
    phi = np.ones(N)
    k_val = 1.0
    B_phi = B @ phi
    prod_old = np.abs(B_phi).sum()
    for it in range(300):
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
        converged = abs(k_new - k_val) < 1e-10 and it > 5
        phi, k_val = phi_new, k_new
        B_phi, prod_old = B_phi_norm, prod_norm
        if converged:
            break
    return k_val


def solve_with_alpha(R, n_bc_modes, alpha_fn, n_angular=32, n_rho=32, n_surf_quad=32):
    radii = np.array([R])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 5, dps=25)
    K_vol = build_volume_kernel(SPHERE_1D, r_nodes, panels, radii, _SIG_T,
                                n_angular=n_angular, n_rho=n_rho, dps=25)
    # Start with mode-0 = existing rank-1 (bit-exact).
    K_bc = build_white_bc_correction(
        SPHERE_1D, r_nodes, r_wts, radii, _SIG_T,
        n_angular=n_angular, n_surf_quad=n_surf_quad, dps=25,
    )
    sig_t_n_vec = np.full(len(r_nodes), _SIG_T[0])
    rv = np.array([SPHERE_1D.radial_volume_weight(rj) for rj in r_nodes])
    divisor = SPHERE_1D.rank1_surface_divisor(R)  # = R²

    for n_mode in range(1, n_bc_modes):
        P_n = compute_P_esc_mode(SPHERE_1D, r_nodes, radii, _SIG_T, n_mode,
                                 n_angular=n_angular, dps=25)
        g_n = compute_G_bc_mode(SPHERE_1D, r_nodes, radii, _SIG_T, n_mode,
                                n_surf_quad=n_surf_quad, dps=25)
        alpha = alpha_fn(n_mode)
        # Structure: K_bc_n[i,j] = α * Σ_t · g_n(i) · P_n(j) · r² w / R²
        u = alpha * sig_t_n_vec * g_n / divisor
        v = rv * r_wts * P_n
        K_bc = K_bc + np.outer(u, v)
    return solve_eig(K_vol + K_bc, len(r_nodes))


def run_ladder(R, alphas_dict, N_values=[1, 2, 3, 5, 8]):
    print(f"\n=== Sphere R={R} MFP ===")
    print(f"{'N':>3} " + "  ".join(f"{name:>18}" for name in alphas_dict))
    for N in N_values:
        row = [f"{N:>3}"]
        for name, alpha_fn in alphas_dict.items():
            k = solve_with_alpha(R, N, alpha_fn, n_angular=24, n_rho=24, n_surf_quad=24)
            err = abs(k - _K_INF) / _K_INF
            row.append(f"{err*100:+7.3f}%  k={k:.4f}")
        print("  ".join(row))


if __name__ == "__main__":
    alphas = {
        "α=1": lambda n: 1.0,
        "α=(2n+1)": lambda n: 2 * n + 1,
        "α=1/(2n+1)": lambda n: 1.0 / (2 * n + 1),
        "α=(2n+1)/π": lambda n: (2 * n + 1) / np.pi,
        "α=π(2n+1)": lambda n: np.pi * (2 * n + 1),
    }
    for R in [1.0, 2.0, 5.0, 10.0]:
        run_ladder(R, alphas, N_values=[1, 2, 3, 5, 8])
