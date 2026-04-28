"""Diagnostic: does T's quadrature affect overshoot?

Created by numerics-investigator on 2026-04-27.

QUESTION: Maybe T at n_quad=64 mis-represents the kernel near µ=0,
amplifying grazing modes spuriously. Push n_quad up and see.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _shifted_legendre_eval,
    build_volume_kernel,
    compute_G_bc_mode,
    composite_gl_r,
    gl_float,
    reflection_specular,
)


def shifted_legendre(n, mu):
    return _shifted_legendre_eval(n, mu)


def build_T_spec_sphere(sigt, R, N, n_quad):
    nodes, wts = leggauss(n_quad)
    mu = 0.5 * (nodes + 1.0)
    w = 0.5 * wts
    chord = 2.0 * R
    decay = np.exp(-sigt * chord * mu)
    T = np.zeros((N, N))
    for m in range(N):
        Pm = shifted_legendre(m, mu)
        for n in range(N):
            Pn = shifted_legendre(n, mu)
            T[m, n] = 2.0 * np.sum(w * mu * Pm * Pn * decay)
    return T


def _build_K_components(geom, R, sigt, *, n_bc_modes):
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t_g,
        n_angular=24, n_rho=24, dps=20,
    )

    R_cell = float(radii[-1])
    sig_t_n = np.array([
        sig_t_g[geom.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])
    rv = np.array([
        geom.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    divisor = geom.rank1_surface_divisor(R_cell)

    N_r = len(r_nodes)
    N = n_bc_modes
    P = np.zeros((N, N_r))
    G = np.zeros((N_r, N))

    omega_low, omega_high = geom.angular_range
    omega_pts, omega_wts = gl_float(24, omega_low, omega_high, 20)
    cos_omegas = geom.ray_direction_cosine(omega_pts)
    angular_factor = geom.angular_weight(omega_pts)
    pref = geom.prefactor

    for n in range(N):
        P_esc_n = np.zeros(N_r)
        for i in range(N_r):
            r_i = float(r_nodes[i])
            total = 0.0
            for k_q in range(24):
                cos_om = cos_omegas[k_q]
                rho_max_val = geom.rho_max(r_i, cos_om, R_cell)
                if rho_max_val <= 0.0:
                    continue
                tau = geom.optical_depth_along_ray(
                    r_i, cos_om, rho_max_val, radii, sig_t_g,
                )
                K_esc = geom.escape_kernel_mp(tau, 20)
                mu_exit = (rho_max_val + r_i * cos_om) / R_cell
                p_tilde = float(_shifted_legendre_eval(
                    n, np.array([mu_exit]),
                )[0])
                total += (
                    omega_wts[k_q] * angular_factor[k_q]
                    * p_tilde * K_esc
                )
            P_esc_n[i] = pref * total
        G_bc_n = compute_G_bc_mode(
            geom, r_nodes, radii, sig_t_g, n,
            n_surf_quad=24, dps=20,
        )
        P[n, :] = rv * r_wts * P_esc_n
        G[:, n] = sig_t_n * G_bc_n / divisor

    return r_nodes, r_wts, K_vol, P, G


def _solve(K, sigt, sigs, nuf):
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    M = np.linalg.solve(A, B)
    eigvals = np.linalg.eigvals(M)
    real_mask = np.abs(eigvals.imag) < 1e-10
    return float(eigvals[real_mask].real.max())


@pytest.mark.parametrize(
    "tag,R,sigt,sigs,nuf,N",
    [
        ("thin τR=2.5 N=4", 5.0, 0.5, 0.38, 0.025, 4),
        ("thin τR=2.5 N=6", 5.0, 0.5, 0.38, 0.025, 6),
        ("thin τR=2.5 N=8", 5.0, 0.5, 0.38, 0.025, 8),
    ],
)
def test_T_quadrature_convergence(tag, R, sigt, sigs, nuf, N, capsys):
    """Does k_eff stabilize as n_quad in T grows?"""
    with capsys.disabled():
        k_inf = nuf / (sigt - sigs)
        rn, rw, K_vol, P, G = _build_K_components(
            SPHERE_1D, R, sigt, n_bc_modes=N,
        )
        R_op = reflection_specular(N)
        print(f"\n=== {tag}: σ_t={sigt}, R={R}, k_inf={k_inf:.6f} ===")

        for n_quad in (16, 32, 64, 128, 256, 512, 1024, 2048):
            T = build_T_spec_sphere(sigt, R, N, n_quad=n_quad)
            ITR = np.eye(N) - T @ R_op
            K_bc = G @ R_op @ np.linalg.solve(ITR, P)
            k_eff = _solve(K_vol + K_bc, sigt, sigs, nuf)
            err = (k_eff - k_inf) / k_inf
            rho = float(np.max(np.abs(np.linalg.eigvals(T @ R_op))))
            print(f"  n_quad={n_quad:5d}: k_eff={k_eff:.8f} ({err*100:+.4f}%), "
                  f"ρ(T·R)={rho:.6f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
