"""Diagnostic: build K_bc^continuous via direct µ quadrature, check k_eff.

Created by numerics-investigator on 2026-04-27.

Bypass the rank-N basis. Construct K_bc^continuous via:

    K_bc(r_i, r_j) = ∫_0^1 µ-weighted-kernel(r_i, r_j, µ) ·
                            [1 / (1 - e^(-σ·2Rµ))] dµ

where the µ-weighted kernel can be defined by the LINEARITY in N: the
RANK-N bare K_bc with HUGE N approaches the µ-resolved kernel.

Test approach: compute bare K_bc at N=20, 30, 50; check if it converges
to a fixed kernel (the continuous-µ K_bc). If yes, multiply each
"µ-weighted contribution" by the continuous Hebert factor 1/(1-e^(-2σRµ))
and integrate.

Shortcut: the bare K_bc at high N can be DIRECTLY constructed as a
µ-quadrature, bypassing the basis machinery.

Specifically, for the sphere bare K_bc:

  K_bc^bare(r_i, r_j) = ∫_0^1 µ · F_out(r_j, µ) · F_in(r_i, µ) · weight dµ

where F_out(r_j, µ) is the source-to-surface partial-current density
contribution at exit angle µ, and F_in(r_i, µ) is the boundary-to-r_i
contribution at inward angle µ.

For multi-bounce, multiply by 1/(1-e^(-σ·2Rµ)) inside the integral.
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    build_volume_kernel,
    composite_gl_r,
    gl_float,
)


def build_K_bc_continuous(R, sigt, r_nodes, r_wts,
                          n_mu=128, multi_bounce=False):
    """Compute K_bc^bare or K_bc^mb via direct µ quadrature.

    For homogeneous sphere with reflective BC, use the spherically-
    symmetric formulation:

      φ_bc(r_i) = ∫_surface ψ⁻(B, µ) · K_inward(r_i, B, µ) dB dΩ

    For homogeneous + specular, the inward distribution at boundary B
    equals the outward distribution at the corresponding A:
      ψ⁻(B, µ) = ψ⁺(A(B,µ), µ)

    For the multi-bounce sum, ψ⁻_total(B, µ) = ψ⁻_1(B, µ) /
    (1 - e^(-σ·2Rµ)).

    For uniform isotropic source, ψ⁺_1 averaged over surface and azimuth
    is the standard P_esc differential.

    For each (r_i, r_j), the K_bc kernel can be decomposed:

      K_bc(r_i, r_j) = (∫ µ · OUT(r_j, µ) · IN(r_i, µ) · MULT(µ)) dµ

    where:
    - OUT(r_j, µ) is the average ψ⁺(boundary_pt, µ) per unit source at r_j
    - IN(r_i, µ) is the average flux contribution at r_i per unit ψ⁻ at boundary in direction µ
    - MULT(µ) = 1/(1-e^(-σ·2Rµ)) for multi-bounce, =1 for bare

    Specifically using observer-centered (r_i, θ) → boundary at angle θ:
    For sphere, the observer at r_i sees the boundary at distance d(r_i,θ)
    via direction with cos(angle with outward normal at boundary point) = µ_in.
    µ_in = (R - r_i cos θ) / d, with d = √(r_i² + R² - 2 r_i R cos θ).

    For the SOURCE side, source at r_j emits isotropically; the contribution
    to ψ⁺ at the boundary at angle µ_out (with outward normal) per unit
    source is (1/4π) e^(-τ_path) dA / dΩ_obs.

    Constructing this carefully is a major derivation. SHORTCUT: use
    high-N rank-N as the limit, fit µ-resolved decomposition.
    """
    raise NotImplementedError("Use the high-N convergence test below.")


def _build_components(R, sigt, n_bc_modes):
    """Wrap _build_K_specular_components for bare K_bc."""
    from orpheus.derivations.peierls_geometry import (
        SPHERE_1D, _shifted_legendre_eval, build_volume_kernel,
        compute_G_bc_mode, composite_gl_r, gl_float, reflection_specular,
    )
    radii = np.array([R])
    sig_t_g = np.array([sigt])
    r_nodes, r_wts, panels = composite_gl_r(radii, 2, 4, dps=20)
    K_vol = build_volume_kernel(
        SPHERE_1D, r_nodes, panels, radii, sig_t_g,
        n_angular=24, n_rho=24, dps=20,
    )

    R_cell = float(radii[-1])
    geom = SPHERE_1D
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

    return r_nodes, r_wts, K_vol, P, G, reflection_specular(N)


def test_high_N_bare_specular_convergence(capsys):
    """Does bare specular K_bc converge to a limit as N → ∞?"""
    with capsys.disabled():
        R = 5.0
        sigt = 0.5
        sigs = 0.38
        nuf = 0.025
        k_inf = nuf / (sigt - sigs)
        print(f"\n=== bare specular convergence as N → ∞ ===")
        print(f"  σ_t·R = {sigt*R}, k_inf = {k_inf:.6f}")

        prev_K_bc_norm = 0.0
        prev_keff = 0.0
        for N in (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 25):
            try:
                rn, rw, K_vol, P, G, R_op = _build_components(R, sigt, N)
                K_bc_bare = G @ R_op @ P
                K_total = K_vol + K_bc_bare
                k_eff = _solve(K_total, sigt, sigs, nuf)
                err = (k_eff - k_inf) / k_inf

                K_bc_norm = np.linalg.norm(K_bc_bare)
                d_K = np.abs(K_bc_norm - prev_K_bc_norm)
                d_k = abs(k_eff - prev_keff)
                print(f"  N={N:2d}: k_eff={k_eff:.6f} ({err*100:+.3f}%), "
                      f"|K_bc|={K_bc_norm:.4f}, dK={d_K:.2e}, dk={d_k:.2e}")
                prev_K_bc_norm = K_bc_norm
                prev_keff = k_eff
            except Exception as e:
                print(f"  N={N}: FAILED: {e}")
                break


def _solve(K, sigt, sigs, nuf):
    N = K.shape[0]
    A = sigt * np.eye(N) - sigs * K
    B = nuf * K
    M = np.linalg.solve(A, B)
    eigvals = np.linalg.eigvals(M)
    real_mask = np.abs(eigvals.imag) < 1e-10
    return float(eigvals[real_mask].real.max())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
