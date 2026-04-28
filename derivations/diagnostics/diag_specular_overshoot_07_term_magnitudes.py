"""Diagnostic: magnitude of each bounce term in the geometric series.

Created by numerics-investigator on 2026-04-27.

For thin sphere N=8:
- k=0 bare: -3.22% (under k_inf)
- k=1 (1st + 2nd): +3.74%
- The 2nd-bounce term ALONE accounts for a 7pp swing.

But scalar P_ss for thin τ_R=2.5 is only 0.077. Hebert factor is 1.083
— a 8.3% multiplier. So the 2nd-bounce contribution should add ~8% of
the 1st-bounce contribution. 8% × (~5% deficit) → ~0.4% improvement.

Yet at N=8 we see a 7pp jump. The 2nd-bounce term is amplified by
(ρ(T·R) ~ 0.82) — much bigger than scalar P_ss. So the rank-N basis
amplifies grazing-mode bounces enormously.

Let's measure exactly how much of K_bc^bare each bounce contributes,
and see WHICH BASIS MODES dominate.
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


def build_T_spec_sphere(sigt, R, N, n_quad=128):
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


def _build_components(geom, R, sigt, *, n_bc_modes):
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

    R_op = reflection_specular(N)
    T = build_T_spec_sphere(sigt, R, N)
    return r_nodes, r_wts, K_vol, P, R_op, G, T


def test_term_magnitudes_per_mode(capsys):
    """Decompose each bounce term by basis mode and see where amplification
    happens."""
    with capsys.disabled():
        R = 5.0
        sigt = 0.5
        sigs = 0.38
        nuf = 0.025
        k_inf = nuf / (sigt - sigs)

        for N in (4, 8):
            print(f"\n=== N={N} thin sphere ===")
            rn, rw, K_vol, P, R_op, G, T = _build_components(
                SPHERE_1D, R, sigt, n_bc_modes=N,
            )
            TR = T @ R_op

            # Examine the J⁺ vector at each bounce, when q is the
            # (homogeneous) eigenvector (approx uniform).
            q = np.ones(K_vol.shape[1])
            J_arrival = []
            v = P @ q  # initial J⁺
            print(f"  Initial J⁺ (P·q) = {v}")
            J_arrival.append(v)
            for k in range(8):
                v = TR @ v
                J_arrival.append(v)
                print(f"  J⁺ after bounce {k+1}: norm={np.linalg.norm(v):.4f}, "
                      f"first 4 components = {v[:4]}")

            # Now: contribution of each bounce to flux φ_i
            # φ_k = G · R · J⁺_k = G · R · (TR)^k · P · q
            print(f"\n  Per-bounce flux contributions (sum over r) for q=1:")
            phi_total = np.zeros(K_vol.shape[0])
            for k, J_k in enumerate(J_arrival):
                phi_k = G @ R_op @ J_k
                phi_sum = np.sum(phi_k)
                phi_max = np.max(np.abs(phi_k))
                phi_total = phi_total + phi_k
                print(f"    bounce {k}: <φ_k> = {phi_sum:.4e}, "
                      f"max |φ_k| = {phi_max:.4e}, "
                      f"<φ_total> so far = {np.sum(phi_total):.4e}")

            # And the K_vol contribution for reference
            phi_vol = K_vol @ q
            print(f"  <K_vol·q> = {np.sum(phi_vol):.4e}, max = {np.max(phi_vol):.4e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
