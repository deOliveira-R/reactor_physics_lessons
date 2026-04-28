"""Phase 5 Front C — Jacobian audit.

Compute compute_P_esc_mode(n=0) for one r_j directly via the Phase 4
ω-integral, then via the change of variables to µ_exit. Compare term
by term to find the bug in F_out_mu_sphere.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D, _shifted_legendre_eval,
)


SIG_T = 0.5
R_THIN = 5.0


def Kesc_sphere(tau):
    """K_esc for sphere = e^{-τ} (per CurvilinearGeometry.escape_kernel_mp)."""
    return float(np.exp(-tau))


def test_audit_one_r(capsys):
    with capsys.disabled():
        r_j = 2.674  # from grid
        R = R_THIN
        sigma = SIG_T

        # ─── Method 1: Phase 4 ω-integral (compute_P_esc_mode(n=0))
        from orpheus.derivations.peierls_geometry import gl_float
        omega_pts, omega_wts = gl_float(64, 0.0, np.pi, 25)
        cos_omegas = np.cos(omega_pts)
        sin_omegas = np.sin(omega_pts)
        pref = 0.5
        total_phase4 = 0.0
        for k in range(64):
            cos_om = cos_omegas[k]
            sin_om = sin_omegas[k]
            disc = r_j*r_j*cos_om*cos_om + R*R - r_j*r_j
            rho_max = -r_j*cos_om + np.sqrt(max(disc, 0))
            if rho_max <= 0:
                continue
            tau = sigma * rho_max
            K = Kesc_sphere(tau)
            mu_exit = (rho_max + r_j*cos_om) / R
            jacobian = (rho_max / R)**2  # (rho_max/R)^2
            total_phase4 += omega_wts[k] * sin_om * jacobian * K
        P_esc_mode_n0 = pref * total_phase4
        print(f"\n=== r_j = {r_j}, R = {R}, σ_t = {sigma} ===")
        print(f"Phase 4 ω-integral (P_esc_mode n=0): {P_esc_mode_n0:.6e}")

        # Also test what Phase 4's `compute_P_esc_mode(n=0)` gives
        from orpheus.derivations.peierls_geometry import compute_P_esc_mode
        rn = np.array([r_j])
        P_mode = compute_P_esc_mode(
            SPHERE_1D, rn, np.array([R]), np.array([sigma]),
            n_mode=0, n_angular=64, dps=20,
        )
        print(f"compute_P_esc_mode(n=0):              {P_mode[0]:.6e}")

        # ─── Method 2: change-of-variables to µ_exit
        # Per derivation: F_out(r, u) = pref · u/(r²·|cos(ω)|) ·
        # [(R·u - D)²·K_esc(τ_-) + (R·u + D)²·K_esc(τ_+)]
        # where D = √(r² - R²(1-u²)) = r·|cos(ω)|.
        # Range: u ∈ [u_min, 1] where u_min = √(1-(r/R)²).

        u_min = np.sqrt(max(1.0 - (r_j/R)**2, 0.0))
        # Quadrature: GL on [u_min, 1]
        Q = 64
        nodes, wts = np.polynomial.legendre.leggauss(Q)
        u_pts_full = 0.5 * (nodes + 1.0)         # [0, 1]
        u_wts_full = 0.5 * wts
        # Use [u_min, 1]
        u_pts = u_min + (1.0 - u_min) * 0.5 * (nodes + 1.0)
        u_wts = (1.0 - u_min) * 0.5 * wts

        total_native = 0.0
        for q in range(Q):
            u = u_pts[q]
            D2 = r_j*r_j - R*R*(1 - u*u)
            if D2 < 0:
                continue
            D = np.sqrt(D2)
            rho_minus = R*u - D  # short
            rho_plus = R*u + D   # long
            tau_m = sigma * rho_minus
            tau_p = sigma * rho_plus
            K_m = Kesc_sphere(tau_m)
            K_p = Kesc_sphere(tau_p)
            cos_om = D/r_j
            integrand = u/(r_j*r_j*cos_om) * (
                rho_plus*rho_plus*K_p + rho_minus*rho_minus*K_m
            )
            total_native += u_wts[q] * integrand
        F_out_integral = pref * total_native
        print(f"u-integral (manual native):           {F_out_integral:.6e}")
        print(f"  u_min                              = {u_min:.6e}")

        # ── Test 3: numerical change-of-variable, GL on u ∈ [0, 1]
        # but skipping u < u_min.
        total_full = 0.0
        for q in range(Q):
            u = u_pts_full[q]
            if u < u_min:
                continue
            D2 = r_j*r_j - R*R*(1 - u*u)
            if D2 < 0:
                continue
            D = np.sqrt(D2)
            rho_minus = R*u - D
            rho_plus = R*u + D
            tau_m = sigma * rho_minus
            tau_p = sigma * rho_plus
            K_m = Kesc_sphere(tau_m)
            K_p = Kesc_sphere(tau_p)
            cos_om = D/r_j
            integrand = u/(r_j*r_j*cos_om) * (
                rho_plus*rho_plus*K_p + rho_minus*rho_minus*K_m
            )
            total_full += u_wts_full[q] * integrand
        F_out_full = pref * total_full
        print(f"GL on [0,1] (skip u<u_min):           {F_out_full:.6e}")
        print(f"\n  ratio (manual native) / (Phase 4): "
              f"{F_out_integral/P_esc_mode_n0:.4f}")
        print(f"  ratio (GL [0,1])      / (Phase 4): "
              f"{F_out_full/P_esc_mode_n0:.4f}")
