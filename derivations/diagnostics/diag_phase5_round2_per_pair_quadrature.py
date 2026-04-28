"""Round 2 follow-up — Per-pair µ-quadrature on the joint visibility cone.

Hypothesis: The Q-divergence of M2 is caused by the STEP DISCONTINUITY at
µ = µ_min(r_i) and µ_min(r_j) in the per-bounce integrand. ``F_out`` is
zero for µ < µ_min(r_j); ``G_in`` is zero for µ < µ_min(r_i). Plain GL
on [0,1] hits both discontinuities, giving poor algebraic convergence.

Fix: integrate per-(i,j) on µ ∈ [µ_min_max, 1] where µ_min_max =
max(µ_min(r_i), µ_min(r_j)). The integrand is SMOOTH on this interval.

Test: spectral GL convergence on the joint visibility interval. Then
sum bounces, k_eff smoke test.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    _chord_tau_mu_sphere,
    composite_gl_r,
)

from derivations.diagnostics.diag_phase5_round2_m2_bounce_resolved import (
    K_INF,
    NU_SIG_F,
    R_THIN,
    SIG_S,
    SIG_T,
    keff_from_K,
)


def _F_out_at_mu(r_j, R, mu_q, sig_t_homog):
    """Pointwise F_out(r_j, µ_q) for homogeneous σ_t. Returns 0 outside
    visibility cone."""
    h = R * float(np.sqrt(max(0.0, 1.0 - mu_q * mu_q)))
    if r_j * r_j < h * h - 1e-15:
        return 0.0
    sin2_om = (R * R / max(r_j * r_j, 1e-30)) * (1.0 - mu_q * mu_q)
    cos2_om = max(1.0 - sin2_om, 0.0)
    if cos2_om < 1e-30:
        return 0.0
    cos_om = float(np.sqrt(cos2_om))
    sqrt_rh = float(np.sqrt(max(r_j * r_j - h * h, 0.0)))
    rho_plus = R * mu_q + sqrt_rh
    rho_minus = R * mu_q - sqrt_rh
    K_plus = float(np.exp(-sig_t_homog * rho_plus))
    K_minus = float(np.exp(-sig_t_homog * rho_minus))
    pref = 0.5
    return pref * mu_q / (r_j * r_j * cos_om) * (
        rho_plus**2 * K_plus + rho_minus**2 * K_minus
    )


def _G_in_at_mu(r_i, R, mu_q, sig_t_homog):
    h = R * float(np.sqrt(max(0.0, 1.0 - mu_q * mu_q)))
    if r_i * r_i < h * h - 1e-15:
        return 0.0
    sin2_om = (R * R / max(r_i * r_i, 1e-30)) * (1.0 - mu_q * mu_q)
    cos2_om = max(1.0 - sin2_om, 0.0)
    if cos2_om < 1e-30:
        return 0.0
    cos_om = float(np.sqrt(cos2_om))
    sqrt_rh = float(np.sqrt(max(r_i * r_i - h * h, 0.0)))
    rho_plus = R * mu_q + sqrt_rh
    rho_minus = R * mu_q - sqrt_rh
    decay_sum = (
        float(np.exp(-sig_t_homog * rho_plus))
        + float(np.exp(-sig_t_homog * rho_minus))
    )
    return 2.0 * R * R * mu_q / (r_i * r_i * cos_om) * decay_sum


def compute_K_bc_M2_per_pair(
    geometry, r_nodes, r_wts, radii, sig_t, *,
    n_quad: int = 64, K_max: int = 20,
):
    r"""Per-pair µ-quadrature on joint visibility cone.

    For each (i, j), compute :math:`\mu_{\min}^{ij} = \max(\mu_{\min}(r_i),
    \mu_{\min}(r_j))` where :math:`\mu_{\min}(r) = \sqrt{1 - (r/R)^2}`.
    Then GL on :math:`[\mu_{\min}^{ij}, 1]`. The integrand is smooth on
    this interval.

    Sum bounces with weight ``no_mu`` (Sanchez form: T(µ) = Σ_k e^{-k τ}).
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    sigma = float(sig_t[0])  # homogeneous

    # ORPHEUS volume / divisor weights
    rv = np.array([
        geometry.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    sig_t_n = np.array([
        sig_t[geometry.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])
    divisor = geometry.rank1_surface_divisor(R)

    # GL nodes on [-1, 1]
    nodes_unit, wts_unit = np.polynomial.legendre.leggauss(n_quad)

    N = len(r_nodes)
    K_total = np.zeros((N, N))

    for i in range(N):
        r_i = float(r_nodes[i])
        mu_min_i = float(np.sqrt(max(0.0, 1.0 - (r_i / R) ** 2)))
        for j in range(N):
            r_j = float(r_nodes[j])
            mu_min_j = float(np.sqrt(max(0.0, 1.0 - (r_j / R) ** 2)))
            mu_lo = max(mu_min_i, mu_min_j)
            # Map [-1, 1] → [mu_lo, 1]
            mu_pts = 0.5 * (nodes_unit + 1.0) * (1.0 - mu_lo) + mu_lo
            mu_wts = 0.5 * (1.0 - mu_lo) * wts_unit
            # Integrand at each µ_q
            integrand = np.zeros(n_quad)
            tau_chord = sigma * 2 * R * mu_pts  # homogeneous chord τ(µ)
            # Multi-bounce sum at each q
            T_mu = np.zeros(n_quad)
            for k in range(K_max + 1):
                T_mu += np.exp(-k * tau_chord)
            for q in range(n_quad):
                f_out = _F_out_at_mu(r_j, R, mu_pts[q], sigma)
                g_in = _G_in_at_mu(r_i, R, mu_pts[q], sigma)
                integrand[q] = g_in * f_out * T_mu[q]
            # Apply ORPHEUS weights for K_ij
            K_total[i, j] = (
                2.0 * (sig_t_n[i] / divisor) * rv[j] * r_wts[j]
                * float(np.sum(mu_wts * integrand))
            )
    return K_total


def test_per_pair_q_convergence(capsys):
    """Per-pair Q-convergence at K_max=10 (homogeneous thin sphere)."""
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)

        print(f"\n=== Per-pair Q-conv at K_max=10 ===")
        print(f"  white_hebert: k_eff = {k_heb:.6f} ({(k_heb-K_INF)/K_INF*100:+.4f}%)")
        print(f"  Q   | k_eff       | rel_inf   | rel_heb     | ‖ΔK‖_F/‖K‖_F")
        K_prev = None
        for Q in (16, 32, 64, 128, 256):
            K_bc_n = compute_K_bc_M2_per_pair(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=Q, K_max=10,
            )
            k_n = keff_from_K(
                K_vol + K_bc_n, sig_t_arr, nu_sf_arr, sig_s_arr,
            )
            rel_inf = (k_n - K_INF) / K_INF * 100
            rel_h = (k_n - k_heb) / k_heb * 100
            if K_prev is not None:
                dK = float(
                    np.linalg.norm(K_bc_n - K_prev) / np.linalg.norm(K_bc_n)
                )
            else:
                dK = float("nan")
            print(f"  {Q:3d} | {k_n:.6f}   | {rel_inf:+.4f}%  | {rel_h:+.4f}%  | {dK:.4e}")
            K_prev = K_bc_n


def test_per_pair_K_max_truncation(capsys):
    """Per-pair K_max truncation at fixed Q=128."""
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)
        print(f"\n=== Per-pair K_max sweep at Q=128 ===")
        print(f"  white_hebert: k_eff = {k_heb:.6f} ({(k_heb-K_INF)/K_INF*100:+.4f}%)")
        print(f"  K_max | k_eff       | rel_inf   | rel_heb")
        for K_max in (0, 1, 2, 3, 5, 10, 20, 50):
            K_bc_n = compute_K_bc_M2_per_pair(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=128, K_max=K_max,
            )
            k_n = keff_from_K(
                K_vol + K_bc_n, sig_t_arr, nu_sf_arr, sig_s_arr,
            )
            rel_inf = (k_n - K_INF) / K_INF * 100
            rel_h = (k_n - k_heb) / k_heb * 100
            print(f"  {K_max:3d}   | {k_n:.6f}   | {rel_inf:+.4f}%  | {rel_h:+.4f}%")
