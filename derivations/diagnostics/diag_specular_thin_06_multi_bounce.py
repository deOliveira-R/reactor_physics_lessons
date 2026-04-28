"""Diagnostic 06: Multi-bounce surface-to-surface correction for specular.

Created by numerics-investigator on 2026-04-27.

HYPOTHESIS: The K_bc = G·R·P specular construction captures only ONE
boundary bounce per source emission. For thin cells where the surface
contribution dominates k_eff, the missing multi-bounce term is the
analog of Hébert's (1 - P_ss)⁻¹ but in rank-N partial-current space.

For SPHERE specular, a surface emission ψ⁻(µ) at one surface point
travels a chord 2R·µ uncollided to the antipodal surface point
(specular preserves the cosine, sphere symmetry transports the
direction across). The angular-resolved surface-to-surface kernel for
specular is

    P_surf^spec(µ → µ') = e^{-σ_t · 2R · µ} · δ(µ - µ')

For the rank-N partial-current outgoing-mode space, the surface
transfer matrix T (mapping surface emission e_n to re-arriving
partial-current J^+_m) is

    T_mn = 2 ∫_0^1 µ · P̃_m(µ) · P̃_n(µ) · e^{-σ_t · 2R · µ} dµ

(2 because the inward partial-current normalization vs outward).

The corrected K_bc^spec_corrected = G · R · (I - T·R)⁻¹ · P, which
sums the geometric series of bounces: the surface re-emerging
multiplier (I - T·R)⁻¹ is the rank-N analog of (1-P_ss)⁻¹.

Verify: at rank-1, T = 2 ∫ µ e^{-2τ_R µ} dµ = (1 - (1+2τ_R) e^{-2τ_R})
/ (2 τ_R²) = P_ss. R = [[1]], so T·R = P_ss and (I - T·R)⁻¹ =
1/(1-P_ss), which RECOVERS Hébert exactly at rank-1.

If this fix gets thin-sphere specular under 0.5% at N=4, we have the
right structural insight.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    build_volume_kernel,
    composite_gl_r,
    compute_P_ss_sphere,
    reflection_specular,
)


def shifted_legendre(n, mu):
    from scipy.special import legendre
    return legendre(n)(2.0 * mu - 1.0)


def build_T_spec_sphere(sigt, R, N, n_quad=128):
    """T_mn = 2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^{-σ_t · 2R · µ} dµ.

    Surface-to-surface partial-current transfer for specular sphere.
    """
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


def _build_K_specular_components(geom, R, sigt, *, n_bc_modes):
    """Returns r_nodes, r_wts, K_vol, P, R_op, G  such that
    K_bc^spec = G @ R_op @ P.
    """
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
    K_vol = build_volume_kernel(
        geom, r_nodes, panels, radii, sig_t_g,
        n_angular=24, n_rho=24, dps=20,
    )

    # Build the specular K_bc and decompose. Easier: re-build P, G inline
    # mirroring the inline assembly in _build_full_K_per_group.
    from orpheus.derivations.peierls_geometry import (
        gl_float, _shifted_legendre_eval, compute_G_bc_mode,
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
    return r_nodes, r_wts, K_vol, P, R_op, G


def _solve(K, sigt, sigs, nuf):
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    M = np.linalg.solve(A, B)
    eigvals = np.linalg.eigvals(M)
    real_mask = np.abs(eigvals.imag) < 1e-10
    return float(eigvals[real_mask].real.max())


@pytest.mark.parametrize(
    "tag,R,sigt,sigs,nuf",
    [
        ("thin τR=2.5 sphere", 5.0, 0.5, 0.38, 0.025),
        ("thick τR=5.0 sphere fuelA", 5.0, 1.0, 0.5, 0.75),
    ],
)
def test_multi_bounce_correction(tag, R, sigt, sigs, nuf, capsys):
    with capsys.disabled():
        k_inf = nuf / (sigt - sigs)
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        P_ss = compute_P_ss_sphere(radii, sig_t_g, n_quad=24, dps=20)
        print(f"\n=== {tag}: σ_t={sigt}, k_inf={k_inf:.6f}, "
              f"P_ss(scalar)={P_ss:.6f} ===")

        for N in (1, 2, 3, 4, 6):
            rn, rw, K_vol, P, R_op, G = _build_K_specular_components(
                SPHERE_1D, R, sigt, n_bc_modes=N,
            )
            T = build_T_spec_sphere(sigt, R, N)

            # Sanity: at rank-1, T should equal P_ss
            if N == 1:
                print(f"  N=1 sanity: T[0,0]={T[0,0]:.6f}, "
                      f"P_ss={P_ss:.6f}, diff={T[0,0]-P_ss:.2e}")

            # Baseline (single-bounce)
            K_bc = G @ R_op @ P
            k_base = _solve(K_vol + K_bc, sigt, sigs, nuf)

            # Multi-bounce corrected: K_bc_corr = G @ R @ (I - T R)^{-1} @ P
            ITR = np.eye(N) - T @ R_op
            K_bc_corr = G @ R_op @ np.linalg.solve(ITR, P)
            k_corr = _solve(K_vol + K_bc_corr, sigt, sigs, nuf)

            err_b = (k_base - k_inf) / k_inf
            err_c = (k_corr - k_inf) / k_inf
            print(f"  N={N}: baseline k={k_base:.8f} ({err_b*100:+.4f}%); "
                  f"corrected k={k_corr:.8f} ({err_c*100:+.4f}%)")
            print(f"        T eigvals: {np.linalg.eigvals(T)}")


if __name__ == "__main__":
    import sys
    for tag, R, sigt, sigs, nuf in [
        ("thin τR=2.5 sphere", 5.0, 0.5, 0.38, 0.025),
        ("thick τR=5.0 sphere fuelA", 5.0, 1.0, 0.5, 0.75),
        ("very thin τR=1.0", 5.0, 0.2, 0.16, 0.01),
    ]:
        k_inf = nuf / (sigt - sigs)
        radii = np.array([R])
        sig_t_g = np.array([sigt])
        P_ss = compute_P_ss_sphere(radii, sig_t_g, n_quad=24, dps=20)
        print(f"\n=== {tag}: σ_t={sigt}, k_inf={k_inf:.6f}, "
              f"P_ss(scalar)={P_ss:.6f} ===")

        for N in (1, 2, 3, 4, 6, 8):
            rn, rw, K_vol, P, R_op, G = _build_K_specular_components(
                SPHERE_1D, R, sigt, n_bc_modes=N,
            )
            T = build_T_spec_sphere(sigt, R, N)
            if N == 1:
                print(f"  N=1 sanity: T[0,0]={T[0,0]:.6f}, "
                      f"P_ss={P_ss:.6f}, diff={T[0,0]-P_ss:.2e}")

            K_bc = G @ R_op @ P
            k_base = _solve(K_vol + K_bc, sigt, sigs, nuf)

            ITR = np.eye(N) - T @ R_op
            K_bc_corr = G @ R_op @ np.linalg.solve(ITR, P)
            k_corr = _solve(K_vol + K_bc_corr, sigt, sigs, nuf)

            err_b = (k_base - k_inf) / k_inf
            err_c = (k_corr - k_inf) / k_inf
            print(f"  N={N}: baseline k={k_base:.8f} ({err_b*100:+.4f}%); "
                  f"corrected k={k_corr:.8f} ({err_c*100:+.4f}%)")
    sys.exit(0)
