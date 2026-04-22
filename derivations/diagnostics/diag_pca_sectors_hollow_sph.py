"""PCA (piecewise-constant angular) sectors for hollow-sphere rank-N closure.

Motivation: Issue #121 — Sanchez & Santandrea 2002 paradigm. Instead of
Legendre polynomials, use indicator functions on angular cones as the
basis. Each "sector" is a cone of solid angle, and the angular flux
within each sector is assumed constant (piecewise-constant basis).

At rank (1 outer-sector, 1 inner-sector), this reduces to F.4 (one
constant outer mode + one constant inner mode).

For rank-M (M sectors per surface), the hemisphere c ∈ [0, 1] is
partitioned as 0 = c_0 < c_1 < ... < c_M = 1, and the basis is:
    φ_k(c) = 1_{[c_k, c_{k+1}]}(c) / norm   for k = 0, ..., M-1

Two partition schemes to test:
(a) Uniform: c_k = k/M.
(b) Physics-informed: split at the c_I critical angle µ_crit =
   √(1-ρ²), and sub-divide within each region.

Goal: see if PCA at rank-(1, 1, 2) or rank-(1, 1, 3) can break F.4 at
similar (σ_t·R, ρ) as the split-basis work.

This is a minimal-viable prototype — no Sanchez-McCormick reciprocity
enforcement, no area factors. Diagnostic-only.
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy import integrate

sys.path.insert(0, '/workspaces/ORPHEUS/derivations/diagnostics')

from diag_cin_aware_split_basis_keff import (
    _unit_legendre_lambdas,
    grazing_lambdas,
    mu_crit,
    chord_oi,
    solve_k_eff,
    run_scalar_f4,
    compute_P_esc_graze_mode,
    compute_P_esc_steep_mode,
    compute_G_bc_graze_mode,
    compute_G_bc_steep_mode,
)
from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry,
    build_volume_kernel,
    composite_gl_r,
    gl_float,
)

K_INF = 1.5


def make_sector_basis(edges):
    """Given sector edges [c_0, c_1, ..., c_M], return M indicator basis functions.

    Each basis function is orthonormal under c-weight on [0,1]:
      ∫_0^1 φ_k(c)² · c dc = 1
    which means φ_k(c) = 1_{[c_k, c_{k+1}]}(c) / √((c_{k+1}² - c_k²)/2).
    """
    funcs = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k+1]
        norm = math.sqrt((hi*hi - lo*lo) / 2.0)
        def make_f(lo=lo, hi=hi, norm=norm):
            def f(c):
                if np.isscalar(c):
                    return 1.0/norm if (lo <= c <= hi) else 0.0
                c_arr = np.asarray(c)
                return np.where((c_arr >= lo) & (c_arr <= hi), 1.0/norm, 0.0)
            return f
        funcs.append(make_f())
    return funcs


def run_pca_sectors(r_0, R, sig_t_val, sig_s_val, nsf_val, inner_edges,
                     n_panels=2, p_order=4, n_ang=32, dps=15):
    """Run rank-(1, 1, M) closure with M inner sectors."""
    geom = CurvilinearGeometry(kind="sphere-1d", inner_radius=r_0)
    radii = np.array([R])
    sig_t = np.array([sig_t_val])
    r_nodes, r_wts, panels = composite_gl_r(radii, n_panels, p_order,
                                             dps=dps, inner_radius=r_0)
    K_vol = build_volume_kernel(geom, r_nodes, panels, radii, sig_t,
                                 n_angular=n_ang, n_rho=n_ang, dps=dps)

    N_g = N_s = 1
    N_i = len(inner_edges) - 1
    rho_val = r_0 / R
    muc = mu_crit(rho_val)

    leg_s = _unit_legendre_lambdas(max(N_s, 1))
    leg_g = grazing_lambdas(max(N_g, 1), muc)
    inner_basis = make_sector_basis(inner_edges)

    D = N_g + N_s + N_i
    N_r = len(r_nodes)
    P = np.zeros((D, N_r))
    G = np.zeros((N_r, D))
    R_out = float(radii[-1])
    A_outer = R_out * R_out
    A_inner = r_0 ** 2
    rv = r_nodes ** 2

    # Grazing
    P_m = compute_P_esc_graze_mode(geom, r_nodes, radii, sig_t, 0, muc, leg_g,
                                     n_angular=n_ang, dps=dps)
    G_m = compute_G_bc_graze_mode(geom, r_nodes, radii, sig_t, 0, muc, leg_g,
                                    n_surf_quad=n_ang, dps=dps)
    P[0, :] = rv * r_wts * P_m
    G[:, 0] = G_m / A_outer

    # Steep
    P_m = compute_P_esc_steep_mode(geom, r_nodes, radii, sig_t, 0, muc, rho_val, leg_s,
                                     n_angular=n_ang, dps=dps)
    G_m = compute_G_bc_steep_mode(geom, r_nodes, radii, sig_t, 0, muc, rho_val, leg_s,
                                    n_surf_quad=n_ang, dps=dps)
    P[1, :] = rv * r_wts * P_m
    G[:, 1] = G_m / A_outer

    # Inner sectors — use the asymptote-basis projection primitives
    from diag_cin_split_asymptote_basis import (
        compute_P_esc_inner_asym,
        compute_G_bc_inner_asym,
    )
    for m in range(N_i):
        P_m = compute_P_esc_inner_asym(geom, r_nodes, radii, sig_t, m,
                                         inner_basis, n_angular=n_ang, dps=dps)
        G_m = compute_G_bc_inner_asym(geom, r_nodes, radii, sig_t, m,
                                        inner_basis, n_surf_quad=n_ang, dps=dps)
        P[2 + m, :] = rv * r_wts * P_m
        G[:, 2 + m] = G_m / A_inner

    # sig_t scaling on G
    sig_t_n = np.empty(N_r)
    for i, ri in enumerate(r_nodes):
        ki = geom.which_annulus(ri, radii)
        sig_t_n[i] = sig_t[ki]
    G = sig_t_n[:, None] * G

    # Build W (split + sectors)
    tau = sig_t_val * R_out
    W = np.zeros((D, D))

    # W_gg
    def integrand_gg(mu, m=0, n=0):
        return leg_g[m](mu) * leg_g[n](mu) * math.exp(-2.0 * tau * mu) * mu
    val, _ = integrate.quad(integrand_gg, 0.0, muc, epsabs=1e-13, epsrel=1e-11)
    W[0, 0] = val

    # W_si, W_is (steep ↔ inner-sector)
    for m in range(N_i):
        for n in range(1):  # just steep mode-0
            def integrand(c, m=m, n=n):
                return inner_basis[m](c) * leg_s[n](c) * \
                       math.exp(-tau * chord_oi(c, rho_val)) * c
            val, _ = integrate.quad(integrand, 0.0, 1.0, epsabs=1e-13, epsrel=1e-11)
            W[2 + m, 1] = val / rho_val
            W[1, 2 + m] = val / rho_val

    # B matrix
    B = np.zeros((D, D))
    B[0, 0] = muc * muc
    B[0, 1] = muc * rho_val
    B[1, 0] = rho_val * muc
    B[1, 1] = rho_val * rho_val
    for m in range(N_i):
        B[2 + m, 2 + m] = 1.0

    M = np.eye(D) - W @ B
    K_bc = G @ B @ np.linalg.inv(M) @ P
    K = K_vol + K_bc
    return solve_k_eff(K, sig_t_val, sig_s_val, nsf_val)


def main():
    print("=" * 80)
    print("PCA sectors on inner surface — rank-(1, 1, M) closure test")
    print("=" * 80)

    sig_t = 1.0
    sig_s = 1.0 / 3.0
    nsf = 1.0

    # Key test points (σ_t·R ≥ 5 where split basis works)
    points = [(5.0, 0.3), (10.0, 0.3), (20.0, 0.3),
              (5.0, 0.5), (10.0, 0.5),
              (5.0, 0.7), (10.0, 0.7)]

    print(f"\n{'σ_t·R':>6} {'ρ':>6} {'F.4':>10} {'M=1 (uni)':>12} "
          f"{'M=2 uni':>10} {'M=2 phys':>10} {'M=3 uni':>10}")
    print("-" * 80)

    for sig_t_R, rho in points:
        R = sig_t_R / sig_t
        r_0 = rho * R
        try:
            k_f4 = run_scalar_f4(r_0, R, sig_t, sig_s, nsf)
            err_f4 = abs(k_f4 - K_INF) / K_INF * 100
        except Exception:
            err_f4 = float('nan')

        # M=1: full hemisphere (should ≈ F.4)
        try:
            k1 = run_pca_sectors(r_0, R, sig_t, sig_s, nsf, [0.0, 1.0])
            e1 = abs(k1 - K_INF) / K_INF * 100
        except Exception as e:
            e1 = float('nan')

        # M=2 uniform: split at 0.5
        try:
            k2u = run_pca_sectors(r_0, R, sig_t, sig_s, nsf, [0.0, 0.5, 1.0])
            e2u = abs(k2u - K_INF) / K_INF * 100
        except Exception:
            e2u = float('nan')

        # M=2 physics-informed: split at c corresponding to the Eddington mean
        # µ where ∫_0^µ c dc = ∫_µ^1 c dc → µ² = 1 - µ² → µ = 1/√2
        try:
            c_eddington = 1.0 / math.sqrt(2.0)
            k2p = run_pca_sectors(r_0, R, sig_t, sig_s, nsf,
                                     [0.0, c_eddington, 1.0])
            e2p = abs(k2p - K_INF) / K_INF * 100
        except Exception:
            e2p = float('nan')

        # M=3 uniform: 1/3, 2/3
        try:
            k3 = run_pca_sectors(r_0, R, sig_t, sig_s, nsf, [0.0, 1/3, 2/3, 1.0])
            e3 = abs(k3 - K_INF) / K_INF * 100
        except Exception:
            e3 = float('nan')

        print(f"{sig_t_R:>6.1f} {rho:>6.2f} {err_f4:>10.4f}% "
              f"{e1:>12.4f}% {e2u:>10.4f}% {e2p:>10.4f}% {e3:>10.4f}%")


if __name__ == "__main__":
    main()
