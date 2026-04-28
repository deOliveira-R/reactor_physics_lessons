"""Diagnostic 14: restore the (ρ_max/R)² Jacobian in the sphere
specular P primitive and see if thin-cell plateau lifts.

Created by numerics-investigator on 2026-04-27.

User Hypothesis 3: 'My no-Jacobian P primitive may be wrong for thin
cells. The Phase 1 derivation argued that the (ρ_max/R)² Jacobian in
compute_P_esc_mode was spurious because ρ²/s² = 1 for sphere. That's
true GEOMETRICALLY. But maybe the Jacobian was empirically calibrated
for OTHER reasons.'

The diagnosis above (diag 12) shows specular has 26-44 % implied
leakage on thin cell — a strong signal that the K_bc primitive has
the wrong NORMALIZATION (not just the wrong angular shape). The
Jacobian factor `(ρ_max/R)²` is a candidate fix because it converts
observer-Ω measure to surface-A measure, which is REQUIRED for the
P primitive to return Marshak J⁺_n moments (which is what R_spec
expects).

This diagnostic builds the K_bc with both variants and reports
k_eff vs k_inf.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, build_volume_kernel, composite_gl_r,
    compute_G_bc_mode, compute_P_esc_mode,
    reflection_specular,
    gl_float, _shifted_legendre_eval,
)


def _build_specular_K_with_pchoice(
    R, sigt, *, n_bc_modes, p_choice="no_jacobian",
):
    """Build K = K_vol + G·R_spec·P with two choices for P.
    p_choice = 'no_jacobian': inline integrand sin θ · P̃_n · e^-τ
    p_choice = 'with_jacobian': use compute_P_esc_mode (carries (ρ/R)²).
    """
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
    K_vol = build_volume_kernel(SPHERE_1D, r_nodes, panels, radii, sig_t_g,
                                n_angular=24, n_rho=24, dps=20)

    R_cell = float(radii[-1])
    sig_t_n = np.array([
        sig_t_g[SPHERE_1D.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])
    rv = np.array([
        SPHERE_1D.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    divisor = SPHERE_1D.rank1_surface_divisor(R_cell)

    N_r = len(r_nodes)
    N = n_bc_modes
    P = np.zeros((N, N_r))
    G = np.zeros((N_r, N))

    omega_low, omega_high = SPHERE_1D.angular_range
    omega_pts, omega_wts = gl_float(24, omega_low, omega_high, 20)
    cos_omegas = SPHERE_1D.ray_direction_cosine(omega_pts)
    angular_factor = SPHERE_1D.angular_weight(omega_pts)
    pref = SPHERE_1D.prefactor

    for n in range(N):
        if p_choice == "no_jacobian":
            P_esc_n = np.zeros(N_r)
            for i in range(N_r):
                r_i = float(r_nodes[i])
                total = 0.0
                for k_q in range(24):
                    cos_om = cos_omegas[k_q]
                    rho_max_val = SPHERE_1D.rho_max(r_i, cos_om, R_cell)
                    if rho_max_val <= 0.0:
                        continue
                    tau = SPHERE_1D.optical_depth_along_ray(
                        r_i, cos_om, rho_max_val, radii, sig_t_g,
                    )
                    K_esc = SPHERE_1D.escape_kernel_mp(tau, 20)
                    mu_exit = (rho_max_val + r_i * cos_om) / R_cell
                    p_tilde = float(_shifted_legendre_eval(
                        n, np.array([mu_exit]),
                    )[0])
                    total += (
                        omega_wts[k_q] * angular_factor[k_q]
                        * p_tilde * K_esc
                    )
                P_esc_n[i] = pref * total
        elif p_choice == "with_jacobian":
            P_esc_n = compute_P_esc_mode(
                SPHERE_1D, r_nodes, radii, sig_t_g, n,
                n_angular=24, dps=20,
            )
        elif p_choice == "with_mu_weight":
            # Marshak basis: integrand has explicit µ_exit weight
            P_esc_n = np.zeros(N_r)
            for i in range(N_r):
                r_i = float(r_nodes[i])
                total = 0.0
                for k_q in range(24):
                    cos_om = cos_omegas[k_q]
                    rho_max_val = SPHERE_1D.rho_max(r_i, cos_om, R_cell)
                    if rho_max_val <= 0.0:
                        continue
                    tau = SPHERE_1D.optical_depth_along_ray(
                        r_i, cos_om, rho_max_val, radii, sig_t_g,
                    )
                    K_esc = SPHERE_1D.escape_kernel_mp(tau, 20)
                    mu_exit = (rho_max_val + r_i * cos_om) / R_cell
                    p_tilde = float(_shifted_legendre_eval(
                        n, np.array([mu_exit]),
                    )[0])
                    total += (
                        omega_wts[k_q] * angular_factor[k_q]
                        * mu_exit * p_tilde * K_esc
                    )
                P_esc_n[i] = pref * total
        else:
            raise ValueError(f"unknown p_choice: {p_choice}")

        G_bc_n = compute_G_bc_mode(
            SPHERE_1D, r_nodes, radii, sig_t_g, n,
            n_surf_quad=24, dps=20,
        )
        P[n, :] = rv * r_wts * P_esc_n
        G[:, n] = sig_t_n * G_bc_n / divisor

    R_op = reflection_specular(N)
    K_bc = G @ R_op @ P
    return K_vol + K_bc


def _eig(K, sigt, sigs, nuf):
    A = sigt * np.eye(K.shape[0]) - sigs * K
    B = nuf * K
    eigvals = np.linalg.eigvals(np.linalg.solve(A, B))
    real_mask = np.abs(eigvals.imag) < 1e-10
    if not real_mask.any():
        return float("nan")
    return float(eigvals[real_mask].real.max())


def test_jacobian_variants(capsys):
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    k_inf = nuf / (sigt - sigs)
    with capsys.disabled():
        print(f"\n=== thin sphere τR=2.5, k_inf={k_inf:.6f} ===")
        for N in (1, 2, 4, 6):
            print(f"\n  N={N}:")
            for choice in ("no_jacobian", "with_jacobian", "with_mu_weight"):
                K = _build_specular_K_with_pchoice(
                    R, sigt, n_bc_modes=N, p_choice=choice,
                )
                k = _eig(K, sigt, sigs, nuf)
                err = (k - k_inf) / k_inf
                print(f"    p={choice:18s}: k_eff={k:.6f} ({err*100:+.4f}%)")


if __name__ == "__main__":
    R = 5.0; sigt = 0.5; sigs = 0.38; nuf = 0.025
    k_inf = nuf / (sigt - sigs)
    print(f"\n=== thin sphere τR=2.5, k_inf={k_inf:.6f} ===")
    for N in (1, 2, 4, 6):
        print(f"\n  N={N}:")
        for choice in ("no_jacobian", "with_jacobian", "with_mu_weight"):
            K = _build_specular_K_with_pchoice(
                R, sigt, n_bc_modes=N, p_choice=choice,
            )
            k = _eig(K, sigt, sigs, nuf)
            err = (k - k_inf) / k_inf
            print(f"    p={choice:18s}: k_eff={k:.6f} ({err*100:+.4f}%)")
