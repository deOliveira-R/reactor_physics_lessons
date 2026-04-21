"""Diagnostic: Numerical probe of rank-N hollow-sphere closure conventions.

Created by numerics-investigator on 2026-04-21.

Tests multiple conventions for assembling `K_bc = G * R_eff * P` with
rank-N per-face mode primitives on a hollow sphere, identifying which
R_eff + primitive-measure combination closes the Wigner-Seitz identity
row-sum (q=1 -> K_bc·1 ~= baseline + small correction).

Result set written inline via run_probe(). Isolates the measure
mismatch: code's P/G primitives compute angular-flux (Lambert) moments
while W encodes partial-current (mu-weighted) moments, so
`(I-W)^-1 · P` couples INCOMPATIBLE bases at mode n>=1.

If promoted, this becomes a regression/progression test for Phase F.5.
See the session report for the full recipe and projected fix.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/workspaces/ORPHEUS")

import numpy as np
from numpy.polynomial.legendre import leggauss

from orpheus.derivations._kernels import _shifted_legendre_eval
from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry, composite_gl_r, gl_float,
    compute_P_esc_outer, compute_P_esc_inner,
    compute_G_bc_outer, compute_G_bc_inner,
    compute_hollow_sph_transmission_rank_n,
)


# ── helpers ────────────────────────────────────────────────────────────
def half_range_gram(N, weight="none", n_quad=128):
    x, w = leggauss(n_quad)
    mu = 0.5 * (x + 1.0); w = 0.5 * w
    P = np.zeros((N, n_quad))
    for n in range(N):
        P[n, :] = _shifted_legendre_eval(n, mu)
    B = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            f = P[n] * P[m]
            if weight == "mu":
                f = mu * f
            B[n, m] = float(np.sum(w * f))
    return B


def _P_esc_outer_mode_with_mu(geom, r_nodes, radii, sig_t, n_mode,
                              n_angular=32, dps=15, mu_power=0):
    """Generalised mode-n outer-face escape primitive.

    Integrand: (1/2) sin theta * mu_exit^mu_power * P_tilde_n(mu_exit) * K_esc(tau).
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    R = float(radii[-1])
    omega_pts, omega_wts = gl_float(n_angular, 0.0, np.pi, dps)
    cos_omegas = np.cos(omega_pts)
    sin_omegas = np.sin(omega_pts)
    pref = 0.5
    P = np.zeros(len(r_nodes))
    for i, r_i in enumerate(r_nodes):
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho = geom.rho_max(float(r_i), cos_om, R)
            if rho <= 0.0:
                continue
            rho_in_m, _ = geom.rho_inner_intersections(float(r_i), cos_om)
            if rho_in_m is not None and rho_in_m < rho:
                continue
            tau = geom.optical_depth_along_ray(float(r_i), cos_om, rho,
                                               radii, sig_t)
            K = geom.escape_kernel_mp(tau, dps)
            mu_exit = (rho + float(r_i) * cos_om) / R
            p_t = float(_shifted_legendre_eval(n_mode,
                                               np.array([mu_exit]))[0])
            mu_w = mu_exit ** mu_power if mu_power > 0 else 1.0
            total += omega_wts[k] * sin_omegas[k] * mu_w * p_t * K
        P[i] = pref * total
    return P


def _P_esc_inner_mode_with_mu(geom, r_nodes, radii, sig_t, n_mode,
                              n_angular=32, dps=15, mu_power=0):
    r_nodes = np.asarray(r_nodes, dtype=float)
    r_0 = float(geom.inner_radius)
    omega_pts, omega_wts = gl_float(n_angular, 0.0, np.pi, dps)
    cos_omegas = np.cos(omega_pts)
    sin_omegas = np.sin(omega_pts)
    pref = 0.5
    P = np.zeros(len(r_nodes))
    for i, r_i in enumerate(r_nodes):
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_in_m, _ = geom.rho_inner_intersections(float(r_i), cos_om)
            if rho_in_m is None:
                continue
            tau = geom.optical_depth_along_ray(float(r_i), cos_om, rho_in_m,
                                               radii, sig_t)
            K = geom.escape_kernel_mp(tau, dps)
            sin_om = np.sqrt(max(0.0, 1.0 - cos_om * cos_om))
            h2 = float(r_i) ** 2 * sin_om ** 2
            mu_exit = np.sqrt(max(0.0, (r_0 ** 2 - h2) / (r_0 ** 2)))
            p_t = float(_shifted_legendre_eval(n_mode,
                                               np.array([mu_exit]))[0])
            mu_w = mu_exit ** mu_power if mu_power > 0 else 1.0
            total += omega_wts[k] * sin_omegas[k] * mu_w * p_t * K
        P[i] = pref * total
    return P


def _G_bc_outer_mode_with_mu(geom, r_nodes, radii, sig_t, n_mode,
                             n_surf_quad=32, dps=15, mu_power=0):
    r_nodes = np.asarray(r_nodes, dtype=float)
    R = float(radii[-1])
    theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_thetas = np.cos(theta_pts); sin_thetas = np.sin(theta_pts)
    G = np.zeros(len(r_nodes))
    for i, r_i in enumerate(r_nodes):
        total = 0.0
        for k in range(n_surf_quad):
            ct = cos_thetas[k]; st = sin_thetas[k]
            rho = geom.rho_max(float(r_i), ct, R)
            if rho <= 0.0:
                continue
            rho_in_m, _ = geom.rho_inner_intersections(float(r_i), ct)
            if rho_in_m is not None and rho_in_m < rho:
                continue
            tau = geom.optical_depth_along_ray(float(r_i), ct, rho,
                                               radii, sig_t)
            mu_s = (rho + float(r_i) * ct) / R
            p_t = float(_shifted_legendre_eval(n_mode,
                                               np.array([mu_s]))[0])
            mu_w = mu_s ** mu_power if mu_power > 0 else 1.0
            total += theta_wts[k] * st * mu_w * p_t * float(np.exp(-tau))
        G[i] = 2.0 * total
    return G


def _G_bc_inner_mode_with_mu(geom, r_nodes, radii, sig_t, n_mode,
                             n_surf_quad=32, dps=15, mu_power=0):
    r_nodes = np.asarray(r_nodes, dtype=float)
    r_0 = float(geom.inner_radius)
    theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_thetas = np.cos(theta_pts); sin_thetas = np.sin(theta_pts)
    G = np.zeros(len(r_nodes))
    for i, r_i in enumerate(r_nodes):
        total = 0.0
        for k in range(n_surf_quad):
            ct = cos_thetas[k]; st = sin_thetas[k]
            rho_in_m, _ = geom.rho_inner_intersections(float(r_i), ct)
            if rho_in_m is None:
                continue
            tau = geom.optical_depth_along_ray(float(r_i), ct, rho_in_m,
                                               radii, sig_t)
            sin_om = np.sqrt(max(0.0, 1.0 - ct * ct))
            h2 = float(r_i) ** 2 * sin_om ** 2
            mu_s = np.sqrt(max(0.0, (r_0 ** 2 - h2) / (r_0 ** 2)))
            p_t = float(_shifted_legendre_eval(n_mode,
                                               np.array([mu_s]))[0])
            mu_w = mu_s ** mu_power if mu_power > 0 else 1.0
            total += theta_wts[k] * st * mu_w * p_t * float(np.exp(-tau))
        G[i] = 2.0 * total
    return G


def build_pg(geom, r_nodes, r_wts, radii, sig_t, N, mu_power_P=0,
             mu_power_G=0, n_angular=24, dps=15):
    r_in = float(geom.inner_radius); R = float(radii[-1])
    div_o = R * R; div_i = r_in * r_in
    sig_t_n = np.array([sig_t[geom.which_annulus(ri, radii)] for ri in r_nodes])
    rv = np.array([geom.radial_volume_weight(rj) for rj in r_nodes])
    P = np.zeros((2 * N, len(r_nodes)))
    G = np.zeros((len(r_nodes), 2 * N))
    for n in range(N):
        # Mode 0 always: legacy (mu_power=0), matches scalar.
        # Mode n>=1: configurable.
        p_pow = 0 if n == 0 else mu_power_P
        g_pow = 0 if n == 0 else mu_power_G
        Po = _P_esc_outer_mode_with_mu(geom, r_nodes, radii, sig_t, n,
                                       n_angular, dps, p_pow)
        Pi = _P_esc_inner_mode_with_mu(geom, r_nodes, radii, sig_t, n,
                                       n_angular, dps, p_pow)
        Go = _G_bc_outer_mode_with_mu(geom, r_nodes, radii, sig_t, n,
                                      n_angular, dps, g_pow)
        Gi = _G_bc_inner_mode_with_mu(geom, r_nodes, radii, sig_t, n,
                                      n_angular, dps, g_pow)
        P[n, :] = rv * r_wts * Po
        P[N + n, :] = rv * r_wts * Pi
        G[:, n] = sig_t_n * Go / div_o
        G[:, N + n] = sig_t_n * Gi / div_i
    return P, G, sig_t_n, rv


# ── diagnostic driver ──────────────────────────────────────────────────
def run_probe(R=5.0, ratio=0.3, N=2, dps=15):
    r_in = ratio * R
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_in)
    r_nodes, r_wts, _ = composite_gl_r(np.array([R]),
                                       n_panels_per_region=2, p_order=4,
                                       dps=dps, inner_radius=r_in)
    radii = np.array([R])
    sig_t = np.array([1.0])

    # Reference N=1 rank-2 closure (Phase F.4 proven, 3.3% residual):
    P1 = np.zeros((2, len(r_nodes))); G1 = np.zeros((len(r_nodes), 2))
    sig_t_n = np.array([sig_t[geom.which_annulus(ri, radii)] for ri in r_nodes])
    rv = np.array([geom.radial_volume_weight(rj) for rj in r_nodes])
    div_o = R * R; div_i = r_in * r_in
    Po1 = compute_P_esc_outer(geom, r_nodes, radii, sig_t, 24, dps)
    Pi1 = compute_P_esc_inner(geom, r_nodes, radii, sig_t, 24, dps)
    Go1 = compute_G_bc_outer(geom, r_nodes, radii, sig_t, 24, dps)
    Gi1 = compute_G_bc_inner(geom, r_nodes, radii, sig_t, 24, dps)
    P1[0, :] = rv * r_wts * Po1; P1[1, :] = rv * r_wts * Pi1
    G1[:, 0] = sig_t_n * Go1 / div_o; G1[:, 1] = sig_t_n * Gi1 / div_i

    W_full = compute_hollow_sph_transmission_rank_n(r_in, R, radii, sig_t,
                                                    n_bc_modes=N, dps=dps)
    W1 = W_full[np.ix_([0, N], [0, N])]
    q1 = np.ones(len(r_nodes))
    rs_N1 = G1 @ np.linalg.inv(np.eye(2) - W1) @ P1 @ q1

    print(f"Hollow sphere R={R}, r_0/R={ratio}, homogeneous sigma_t=1:")
    print(f"  N=1 rank-2 (Phase F.4):  K_bc.1  outer={rs_N1[-1]:.6f}, inner={rs_N1[0]:.6f}")
    print()

    # N=2 variants. Grid over:
    #   mu_power_P in {0, 1}       — extra mu weight in P mode-n>=1
    #   mu_power_G in {0, 1}       — extra mu weight in G mode-n>=1
    #   R_eff family in { (I-W)^-1, (I-W)^-1 D_gelbard, (I-W Bmu_inv)^-1, etc. }
    Bmu = half_range_gram(N, weight="mu")
    Bmu_block = np.block([[Bmu, np.zeros((N, N))], [np.zeros((N, N)), Bmu]])
    Bmu_inv = np.linalg.inv(Bmu_block)
    I4 = np.eye(2 * N)
    D = np.diag([1.0, 3.0, 1.0, 3.0])

    print("Legend: mu_P/mu_G = mu-weight applied to n>=1 primitives in P/G")
    print("         (n=0 always legacy = no mu weight, matches scalar)")
    print()
    print(f"{'mu_P':>5} {'mu_G':>5} {'R_eff':>36} {'outer':>10} {'inner':>10} "
          f"{'d_outer':>10} {'d_inner':>10}")
    for mu_P in (0, 1):
        for mu_G in (0, 1):
            P, G, _, _ = build_pg(geom, r_nodes, r_wts, radii, sig_t, N,
                                   mu_power_P=mu_P, mu_power_G=mu_G)
            for name, R_eff in [
                ("(I-W)^-1", np.linalg.inv(I4 - W_full)),
                ("(I-W)^-1 D_gelbard",
                 np.linalg.inv(I4 - W_full) @ D),
                ("(I - Bmu_inv W)^-1",
                 np.linalg.inv(I4 - Bmu_inv @ W_full)),
            ]:
                rs = G @ R_eff @ P @ q1
                do = rs[-1] - rs_N1[-1]; di = rs[0] - rs_N1[0]
                print(f"{mu_P:>5} {mu_G:>5} {name:>36} {rs[-1]:>10.5f} "
                      f"{rs[0]:>10.5f} {do:>+10.3e} {di:>+10.3e}")


def test_measure_mismatch_premise():
    """The measure converter (B^mu)(B^L)^-1 is not trivially diagonal."""
    N = 3
    B_L = half_range_gram(N, weight="none")
    B_mu = half_range_gram(N, weight="mu")
    C = B_mu @ np.linalg.inv(B_L)
    # At rank-1, C = 0.5 (the hemispheric factor 1/2 for the Lambert
    # angular-flux -> partial-current measure conversion).
    assert abs(C[0, 0] - 0.5) < 1e-10
    # At rank>=2, C has off-diagonal couplings — this is the heart of
    # the measure mismatch. The closure code must be internally consistent
    # in choice of inner product.
    assert abs(C[0, 1]) > 0.1, (
        f"Converter (B^mu)(B^L)^-1 must have off-diagonal coupling "
        f"between modes 0 and 1; got C[0,1] = {C[0,1]:.3e}. If zero, "
        f"the measure mismatch hypothesis is falsified."
    )


if __name__ == "__main__":
    # Match the user's test grid: R=5 (r_0/R=0.3, scalar rank-2 error 3%).
    run_probe(R=5.0, ratio=0.3, N=2)
