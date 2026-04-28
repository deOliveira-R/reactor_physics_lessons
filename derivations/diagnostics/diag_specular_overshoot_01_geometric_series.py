"""Diagnostic: decompose the geometric series K_bc = G·R·(I-TR)⁻¹·P term-by-term.

Created by numerics-investigator on 2026-04-27.
If this test catches a real bug, promote to ``tests/derivations/test_peierls_specular_bc.py``.

QUESTION: Does the geometric series expansion of K_bc^mb monotonically
APPROACH a fixed K_bc^*, or does it diverge / oscillate?

We compute partial sums of K_bc = G·R·Σ_k (TR)^k ·P up to k_max and
extract k_eff at each k. If the series converges (cleanly), then k_eff(k)
should approach a limit. If it diverges, we'll see runaway. If it
overshoots and stabilizes, we'll see the user's reported behavior.

Also report ρ(TR) (spectral radius) at each N — this is the convergence
parameter of the geometric series.

Anchor: thin sphere fuel-A-like (σ_t=0.5, R=5, k_inf=0.20833).
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
    compute_P_ss_sphere,
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


def _build_K_components(geom, R, sigt, *, n_bc_modes, n_quad_T=128):
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
    T = build_T_spec_sphere(sigt, R, N, n_quad=n_quad_T)
    return r_nodes, r_wts, K_vol, P, R_op, G, T


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
        ("very-thin τR=1.0 N=8", 5.0, 0.2, 0.16, 0.01, 8),
    ],
)
def test_geometric_series_term_by_term(tag, R, sigt, sigs, nuf, N, capsys):
    """Walk the geometric series K_bc = G·R·Σ_k (TR)^k ·P term-by-term."""
    with capsys.disabled():
        k_inf = nuf / (sigt - sigs)

        rn, rw, K_vol, P, R_op, G, T = _build_K_components(
            SPHERE_1D, R, sigt, n_bc_modes=N,
        )

        TR = T @ R_op
        eigs_TR = np.linalg.eigvals(TR)
        rho = float(np.max(np.abs(eigs_TR)))

        print(f"\n=== {tag}: σ_t={sigt}, R={R}, k_inf={k_inf:.6f} ===")
        print(f"  ρ(T·R) = {rho:.6f}")
        print(f"  T·R eigvals: {sorted(np.abs(eigs_TR), reverse=True)}")
        print(f"  T diag: {np.diag(T)}")
        print(f"  R diag: {np.diag(R_op)}")

        # Walk partial sums S_k = Σ_{j=0..k} (TR)^j
        # K_bc^(k) = G · R · S_k · P
        I_N = np.eye(N)
        Sk = np.zeros((N, N))
        TRpow = I_N.copy()  # (TR)^0 = I

        print(f"\n  k | k_eff(partial sum) | rel err vs k_inf | rel err vs S∞")
        print(f"  --|--------------------|-----------------|---------------")

        # Reference: full inverse (limit of geometric series)
        S_inf = np.linalg.solve(I_N - TR, I_N)
        K_bc_inf = G @ R_op @ S_inf @ P
        k_inf_closure = _solve(K_vol + K_bc_inf, sigt, sigs, nuf)

        keff_history = []
        for k in range(31):
            Sk = Sk + TRpow
            K_bc_k = G @ R_op @ Sk @ P
            try:
                k_k = _solve(K_vol + K_bc_k, sigt, sigs, nuf)
            except Exception as e:
                k_k = float('nan')
            err_kinf = (k_k - k_inf) / k_inf
            err_closure = (k_k - k_inf_closure) / k_inf_closure
            keff_history.append(k_k)
            if k <= 12 or k in (15, 20, 25, 30):
                print(f"  {k:2d} | {k_k:.8f}     | {err_kinf*100:+.4f}%       | "
                      f"{err_closure*100:+.4e}")
            TRpow = TRpow @ TR

        print(f"\n  Closure limit k_eff = {k_inf_closure:.8f} "
              f"({(k_inf_closure-k_inf)/k_inf*100:+.4f}% vs k_inf)")
        print(f"  Bare specular  k_eff = {_solve(K_vol + G @ R_op @ P, sigt, sigs, nuf):.8f}")

        # Diagnostic asserts:
        # 1) The series converges to the closure limit (assuming ρ<1).
        # Tolerance scales with ρ: at ρ→1 the geometric tail is huge.
        # ρ=0.5 → ~0.5^30 ≈ 1e-9 ; ρ=0.9 → 0.9^30 ≈ 0.04.
        tail_bound = max(1e-6, rho ** 30 * 1e2)
        if rho < 0.99:
            assert abs(keff_history[-1] - k_inf_closure) / k_inf_closure < tail_bound, (
                f"Geometric series did not converge: k_30={keff_history[-1]} "
                f"vs closure_limit={k_inf_closure} (tol={tail_bound})"
            )

        # 2) Each partial sum should be POSITIVE (k_eff > 0).
        # If any is negative or huge, the series oscillates pathologically.
        for k_idx, k_val in enumerate(keff_history[:10]):
            assert 0 < k_val < 2 * k_inf, (
                f"Partial sum k={k_idx}: k_eff={k_val} blew up (k_inf={k_inf})"
            )


if __name__ == "__main__":
    import sys
    pytest.main([__file__, "-v", "-s"])
    sys.exit(0)
