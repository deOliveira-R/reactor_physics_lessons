"""Round 2 follow-up — cos(ω) substitution to remove √ singularity.

The integrand of K_bc^(k) has an integrable √-singularity at
µ = µ_min(r) for ALL k ≥ 0 because

    cos(ω) = √((µ² - µ_min²)/(1 - µ_min²))

vanishes at µ = µ_min, and the integrand carries 1/cos(ω). With the
substitution

    u = cos(ω)        i.e.   µ² = µ_min² + u²(1 - µ_min²)
                              µ = √(µ_min² + u²(1-µ_min²))
                              dµ = u(1-µ_min²)/µ · du

the singularity 1/cos(ω) = 1/u CANCELS the u·du Jacobian, leaving a
SMOOTH integrand on u ∈ [0, 1]. This is the **standard
sphere ω-quadrature** that ORPHEUS already uses in
``compute_P_esc_mode`` / ``compute_G_bc_mode``.

In fact: this is the ω-grid that Front C started from. The whole
attempt to "skip basis projection by integrating µ directly" can be
salvaged IF µ-integration is rephrased as cos(ω)-integration on a
PER-RECEIVER-r interval.

Test: GL on u ∈ [0, 1] with the µ(u) substitution. Expected
spectral Q-convergence.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
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


def compute_K_bc_M2_cosomega(
    geometry, r_nodes, r_wts, radii, sig_t, *,
    n_quad: int = 64, K_max: int = 20,
):
    r"""M2 bounce-resolved K_bc with cos(ω)-substituted µ-quadrature.

    For each (i, j), define :math:`\mu_{\min}^{ij} = \max(\mu_{\min}(r_i),
    \mu_{\min}(r_j))`, then change variables :math:`u = \cos\omega`
    where the relation depends on which RECEIVER carries the cos(ω)
    Jacobian we want to remove.

    SIMPLER: build F_out and G_in WITHOUT the 1/cos(ω) Jacobian (i.e.,
    keep them as functions of ω, parametrized by ω∈(0, π/2]). Then
    integrate against ω directly. Plain GL on ω is spectrally accurate.

    Wait — at receiver r_i and source r_j, the chord cosine µ
    is FIXED at the surface. The only ω that matters is per-receiver:
    ω_i = angle of receiver chord. So G_in(r_i, ω_i) is a function of
    ω_i only. Similarly F_out(r_j, ω_j). They share the same surface
    cosine µ, so ω_i and ω_j are RELATED:

        sin(ω_i) = (R/r_i) sin(θ_surface)  where θ_surface = arccos(µ)
        sin(ω_j) = (R/r_j) sin(θ_surface)

    They depend on receiver/source radius. So integrating over µ is
    equivalent to integrating over θ_surface = arccos(µ) ∈ [0, π/2].

    SECOND ATTEMPT: integrate on θ ∈ [0, π/2], where µ = cos(θ),
    dµ = -sin(θ) dθ. The integrand `G F dµ` has the µ-Jacobian
    1/cos(ω_i) and 1/cos(ω_j) which both vanish at θ = π/2 - 0
    (grazing). But θ at grazing for the smaller of (r_i, r_j) is the
    determinant. Substitute u_i = cos(ω_i): u_i² = 1 - (R/r_i)²(1-µ²)
    = 1 - (R/r_i)²·sin²(θ).

    For receiver alone: rewrite as integral over its u_i.

    The SIMPLEST CORRECT version uses TWO substitutions: factor the
    integrand as G_in · F_out and switch each to its OWN ω. But that
    requires a CHANGE-of-VARIABLES MISMATCH (different u for source
    vs receiver). Instead, integrate over µ on the joint interval,
    then make the µ → u substitution USING THE SMALLER of cos(ω_i)
    or cos(ω_j) (the one that vanishes first).

    Implementation: per (i, j), set µ_min_ij = max(µ_min_i, µ_min_j).
    Substitute u² = (µ² - µ_min_ij²)/(1 - µ_min_ij²), i.e.,
    u = √((µ - µ_min_ij)(µ + µ_min_ij))/√(1 - µ_min_ij²).
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    sigma = float(sig_t[0])  # homogeneous

    rv = np.array([
        geometry.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    sig_t_n = np.array([
        sig_t[geometry.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])
    divisor = geometry.rank1_surface_divisor(R)

    nodes_unit, wts_unit = np.polynomial.legendre.leggauss(n_quad)

    N = len(r_nodes)
    K_total = np.zeros((N, N))

    for i in range(N):
        r_i = float(r_nodes[i])
        mu_min_i_sq = max(0.0, 1.0 - (r_i / R) ** 2)
        for j in range(N):
            r_j = float(r_nodes[j])
            mu_min_j_sq = max(0.0, 1.0 - (r_j / R) ** 2)
            mu_min_sq = max(mu_min_i_sq, mu_min_j_sq)
            mu_min = float(np.sqrt(mu_min_sq))
            # u in [0, 1], µ² = µ_min² + u²(1-µ_min²)
            u_pts = 0.5 * (nodes_unit + 1.0)
            u_wts = 0.5 * wts_unit
            mu_sq_pts = mu_min_sq + u_pts**2 * (1.0 - mu_min_sq)
            mu_pts = np.sqrt(mu_sq_pts)
            # dµ = u·(1-µ_min²) / µ · du
            dmu_du = u_pts * (1.0 - mu_min_sq) / mu_pts
            integrand = np.zeros(n_quad)
            tau_chord = sigma * 2.0 * R * mu_pts
            T_mu = np.zeros(n_quad)
            for k in range(K_max + 1):
                T_mu += np.exp(-k * tau_chord)
            # Compute pointwise integrand
            for q in range(n_quad):
                mu_q = float(mu_pts[q])
                # F_out at (r_j, µ_q)
                h = R * float(np.sqrt(max(0.0, 1.0 - mu_q**2)))
                if r_j**2 < h**2 - 1e-15:
                    f_out = 0.0
                else:
                    sin2_om_j = (R**2 / max(r_j**2, 1e-30)) * (
                        1.0 - mu_q**2
                    )
                    cos2_om_j = max(1.0 - sin2_om_j, 0.0)
                    if cos2_om_j < 1e-30:
                        f_out = 0.0
                    else:
                        cos_om_j = float(np.sqrt(cos2_om_j))
                        sqrt_rh_j = float(np.sqrt(max(r_j**2 - h**2, 0.0)))
                        rho_plus = R * mu_q + sqrt_rh_j
                        rho_minus = R * mu_q - sqrt_rh_j
                        K_plus = float(np.exp(-sigma * rho_plus))
                        K_minus = float(np.exp(-sigma * rho_minus))
                        f_out = 0.5 * mu_q / (r_j**2 * cos_om_j) * (
                            rho_plus**2 * K_plus + rho_minus**2 * K_minus
                        )
                # G_in at (r_i, µ_q)
                if r_i**2 < h**2 - 1e-15:
                    g_in = 0.0
                else:
                    sin2_om_i = (R**2 / max(r_i**2, 1e-30)) * (
                        1.0 - mu_q**2
                    )
                    cos2_om_i = max(1.0 - sin2_om_i, 0.0)
                    if cos2_om_i < 1e-30:
                        g_in = 0.0
                    else:
                        cos_om_i = float(np.sqrt(cos2_om_i))
                        sqrt_rh_i = float(np.sqrt(max(r_i**2 - h**2, 0.0)))
                        rho_p = R * mu_q + sqrt_rh_i
                        rho_m = R * mu_q - sqrt_rh_i
                        decay = (
                            float(np.exp(-sigma * rho_p))
                            + float(np.exp(-sigma * rho_m))
                        )
                        g_in = 2.0 * R**2 * mu_q / (r_i**2 * cos_om_i) * decay
                integrand[q] = g_in * f_out * T_mu[q] * dmu_du[q]
            # The 1/cos(ω) singularity in the integrand is at µ_min,
            # but the dominant one is the larger of µ_min_i and µ_min_j;
            # substituting u kills that one. The OTHER (subdominant)
            # cos(ω) is bounded away from zero on [µ_min_dom, 1] so
            # the integrand is now smooth.
            K_total[i, j] = (
                2.0 * (sig_t_n[i] / divisor) * rv[j] * r_wts[j]
                * float(np.sum(u_wts * integrand))
            )
    return K_total


def test_cosomega_q_convergence(capsys):
    """Q-convergence after cos(ω) substitution at K_max=10."""
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

        print(f"\n=== cos(ω)-subst Q-conv at K_max=10 ===")
        print(f"  white_hebert: k_eff = {k_heb:.6f} ({(k_heb-K_INF)/K_INF*100:+.4f}%)")
        print(f"  Q   | k_eff       | rel_inf   | rel_heb   | ‖ΔK‖/‖K‖")
        K_prev = None
        for Q in (16, 32, 64, 128, 256):
            K_bc_n = compute_K_bc_M2_cosomega(
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
                    np.linalg.norm(K_bc_n - K_prev)
                    / np.linalg.norm(K_bc_n)
                )
            else:
                dK = float("nan")
            print(f"  {Q:3d} | {k_n:.6f}   | {rel_inf:+.4f}%  | {rel_h:+.4f}% | {dK:.4e}")
            K_prev = K_bc_n


def test_cosomega_K_max_truncation(capsys):
    """K_max truncation at fixed Q=64 with cos(ω) substitution."""
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

        for Q in (64, 128, 256):
            print(f"\n=== K_max sweep at Q={Q}, cos(ω) subst ===")
            print(f"  white_hebert: k_eff = {k_heb:.6f}")
            print(f"  K_max | k_eff     | rel_inf  | rel_heb")
            for K_max in (0, 1, 3, 5, 10, 20, 50):
                K_bc_n = compute_K_bc_M2_cosomega(
                    SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                    n_quad=Q, K_max=K_max,
                )
                k_n = keff_from_K(
                    K_vol + K_bc_n, sig_t_arr, nu_sf_arr, sig_s_arr,
                )
                rel_inf = (k_n - K_INF) / K_INF * 100
                rel_h = (k_n - k_heb) / k_heb * 100
                print(f"  {K_max:3d}   | {k_n:.6f} | {rel_inf:+.4f}% | {rel_h:+.4f}%")
