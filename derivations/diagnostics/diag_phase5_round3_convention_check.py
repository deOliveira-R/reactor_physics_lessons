"""Convention check: HALF-M1 vs FULL-M1 vs Front C convention.

Created by numerics-investigator on 2026-04-28.

Round 2 BACKUP found that matrix-Galerkin K_N → HALF M1:
    K_∞^half = (1/2) ∫_0^1 G·F · µ/(1-e^{-2aµ}) dµ  (1/2 from R=½M^{-1})

Front C used (Sanchez convention):
    K_FrontC = 2 ∫_0^1 G·F · 1/(1-e^{-2aµ}) dµ  (no µ — diverges at µ=0)

These are GENUINELY DIFFERENT integrands. Front C's diverges at µ=0 while
HALF-M1 is bounded at µ=0.

This diagnostic computes both at a single (r_i, r_j) pair on a fine grid
and compares to the rank-1 Phase 4 closure='specular_multibounce' value
(which IS the matrix-Galerkin form for one bounce).

Outcome
-------
Determines which integrand convention to use in Galerkin double-integration
when wired to ORPHEUS conventions.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, _build_full_K_per_group, composite_gl_r,
)
from diag_phase5_round3_visibility_cone_quad import (
    _K_cont_pair_visibility_substitution,
)
from diag_phase5_round3_galerkin_double_integration import (
    _F_out_at, _G_in_at, K_INF, R_THIN, SIG_T,
)


def _K_pair_full_m1(r, rp, sigma, R, Q):
    """FULL M1: K = ∫ G·F·µ/(1-e^{-2aµ}) dµ (with vis-cone + u² subst)."""
    return _K_cont_pair_visibility_substitution(r, rp, sigma, R, Q)


def _K_pair_half_m1(r, rp, sigma, R, Q):
    """HALF M1: K = (1/2) ∫ G·F·µ/(1-e^{-2aµ}) dµ."""
    return 0.5 * _K_cont_pair_visibility_substitution(r, rp, sigma, R, Q)


def _K_pair_front_c(r, rp, sigma, R, Q):
    """Front C: K = 2 ∫ G·F·1/(1-e^{-2aµ}) dµ — DIVERGES at µ=0!"""
    R2 = R * R
    mu_min_i = float(np.sqrt(max(0.0, 1.0 - r * r / R2)))
    mu_min_j = float(np.sqrt(max(0.0, 1.0 - rp * rp / R2)))
    mu_lo = max(mu_min_i, mu_min_j)
    if mu_lo >= 1.0 - 1e-15:
        return 0.0
    nodes, wts = np.polynomial.legendre.leggauss(Q)
    u_pts = 0.5 * (nodes + 1.0)
    u_wts = 0.5 * wts
    mu_pts = mu_lo + (1.0 - mu_lo) * u_pts * u_pts
    dmu_du = 2.0 * (1.0 - mu_lo) * u_pts
    arg = sigma * 2.0 * R * mu_pts
    T_no_mu = 1.0 / (1.0 - np.exp(-arg))  # NO µ — divergent at µ=0
    G = np.array([_G_in_at(r, R, sigma, m) for m in mu_pts])
    F = np.array([_F_out_at(rp, R, sigma, m) for m in mu_pts])
    return 2.0 * float(np.sum(u_wts * dmu_du * G * F * T_no_mu))


def test_convention_compare(capsys):
    """Compare three conventions on (r, r') pairs at thin sphere."""
    with capsys.disabled():
        sigma, R = SIG_T, R_THIN
        Q = 128
        pairs = [
            (0.5, 4.5),
            (1.0, 4.0),
            (2.0, 3.0),
            (3.0, 4.5),
        ]
        print(f"\n=== Convention compare at thin sphere (σR={sigma*R}) ===")
        print(f"  (r,r')         | FULL_M1     | HALF_M1     | "
              f"FRONT_C       | ratio FullM1/FrontC")
        for r, rp in pairs:
            K_full = _K_pair_full_m1(r, rp, sigma, R, Q)
            K_half = _K_pair_half_m1(r, rp, sigma, R, Q)
            K_fc = _K_pair_front_c(r, rp, sigma, R, Q)
            ratio = K_full / K_fc if abs(K_fc) > 1e-30 else float('nan')
            print(f"  ({r}, {rp})    | {K_full:+.4e} | {K_half:+.4e} | "
                  f"{K_fc:+.4e} | {ratio:.4f}")


def _K_pair_no_mu_no_half(r, rp, sigma, R, Q):
    r"""HALF Front C: K = ∫ G·F·1/(1-e^{-2aµ}) dµ (with vis-cone + u² subst).

    NO µ in numerator (1/µ singularity at µ=0), NO outer 2 factor. The
    visibility cone restricts µ ≥ µ_lo > 0 so 1/µ stays bounded.
    """
    R2 = R * R
    mu_min_i = float(np.sqrt(max(0.0, 1.0 - r * r / R2)))
    mu_min_j = float(np.sqrt(max(0.0, 1.0 - rp * rp / R2)))
    mu_lo = max(mu_min_i, mu_min_j)
    if mu_lo >= 1.0 - 1e-15:
        return 0.0
    nodes, wts = np.polynomial.legendre.leggauss(Q)
    u_pts = 0.5 * (nodes + 1.0)
    u_wts = 0.5 * wts
    mu_pts = mu_lo + (1.0 - mu_lo) * u_pts * u_pts
    dmu_du = 2.0 * (1.0 - mu_lo) * u_pts
    arg = sigma * 2.0 * R * mu_pts
    T_no_mu = 1.0 / (1.0 - np.exp(-arg))
    G = np.array([_G_in_at(r, R, sigma, m) for m in mu_pts])
    F = np.array([_F_out_at(rp, R, sigma, m) for m in mu_pts])
    return float(np.sum(u_wts * dmu_du * G * F * T_no_mu))


def test_full_smoke_keff_half_m1(capsys):
    """Smoke test k_eff using HALF M1 form (the BACKUP-recommended one).

    Tries multiple conventions for the multi-bounce factor:
    - HALF M1: K = (1/2) ∫ G·F·µ/(1-e^{-2aµ}) dµ
    - FULL M1: K = ∫ G·F·µ/(1-e^{-2aµ}) dµ
    - HALF Front C: K = ∫ G·F·1/(1-e^{-2aµ}) dµ
    - FULL Front C: K = 2 ∫ G·F·1/(1-e^{-2aµ}) dµ

    Whichever matches white_hebert (-0.12%) at Nyström assembly is the
    correct convention to use in Galerkin double-integration.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), 0.38)
        nu_sf_arr = np.full(len(r_nodes), 0.025)
        sig_t_arr = np.full(len(r_nodes), SIG_T)
        N = len(r_nodes)

        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        A = np.diag(sig_t_arr) - K_heb * sig_s_arr[None, :]
        B = K_heb * nu_sf_arr[None, :]
        k_heb = float(np.max(np.real(np.linalg.eigvals(np.linalg.solve(A, B)))))
        rel_heb = (k_heb - K_INF) / K_INF * 100

        print(f"\n=== Smoke k_eff Nyström conventions ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, k_inf={K_INF:.6f}")
        print(f"white_hebert: k_eff={k_heb:.6f} ({rel_heb:+.4f}%)")

        rv = np.array([SPHERE_1D.radial_volume_weight(float(rj)) for rj in r_nodes])
        sig_t_n = np.full(N, SIG_T)
        divisor = SPHERE_1D.rank1_surface_divisor(R_THIN)

        K_vac = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )

        # Pre-compute per-pair K_full and K_no_mu values (Q=128)
        K_full_mat = np.zeros((N, N))
        K_no_mu_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                K_full_mat[i, j] = _K_pair_full_m1(
                    r_nodes[i], r_nodes[j], SIG_T, R_THIN, 128,
                )
                K_no_mu_mat[i, j] = _K_pair_no_mu_no_half(
                    r_nodes[i], r_nodes[j], SIG_T, R_THIN, 128,
                )

        outer_weight = (sig_t_n[:, None] / divisor) * (rv * r_wts)[None, :]

        for label, K_pair, factor in [
            ("HALF M1 (µ, ×0.5)", K_full_mat, 0.5),
            ("FULL M1 (µ, ×1.0)", K_full_mat, 1.0),
            ("DOUBLE M1 (µ, ×2.0)", K_full_mat, 2.0),
            ("HALF Front C (no µ, ×0.5)", K_no_mu_mat, 0.5),
            ("FULL Front C (no µ, ×1.0)", K_no_mu_mat, 1.0),
            ("DOUBLE Front C (no µ, ×2.0)", K_no_mu_mat, 2.0),
        ]:
            K_bc = outer_weight * K_pair * factor
            K_total = K_vac + K_bc
            A = np.diag(sig_t_arr) - K_total * sig_s_arr[None, :]
            B = K_total * nu_sf_arr[None, :]
            k_g = float(np.max(np.real(np.linalg.eigvals(
                np.linalg.solve(A, B)))))
            rel_inf = (k_g - K_INF) / K_INF * 100
            rel_h = (k_g - k_heb) / k_heb * 100
            print(f"  {label:35s}: k_eff={k_g:.6f}, "
                  f"rel_kinf={rel_inf:+.4f}%, rel_heb={rel_h:+.4f}%")
