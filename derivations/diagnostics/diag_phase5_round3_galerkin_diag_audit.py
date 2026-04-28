"""Diagnostic audit: K_continuous(r, r', µ) integrand structure for Galerkin Round 3.

Created by numerics-investigator on 2026-04-28.

Audits the µ-integral of `G_in(r,µ) · F_out(r',µ) · f(µ)` for various (r, r')
to understand whether the Galerkin r-r' double-integration smoothing strategy
can succeed.

Hypothesis check
================
M2 found: K[r_i, r_i] (diagonal) Q-DIVERGES because cos(ω_i)·cos(ω_j) → 0 at
µ_min(r). Off-diagonal converges because µ_min_i ≠ µ_min_j.

For Galerkin to work, we need:
1. At r ≠ r' (any non-zero separation), the µ-integral converges spectrally.
2. The 2-D integral ∫∫ K_continuous(r,r') dr dr' is finite (log-integrable
   diagonal singularity is OK in 2-D).
3. With reasonable n_quad_r, the panel sub-quadrature samples enough
   off-diagonal points to keep the diagonal contribution from dominating.

If condition 2 fails (worse than log divergence), Galerkin won't help.
If conditions 2 and 3 hold but my implementation diverges, there's a wiring bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import composite_gl_r

from diag_phase5_round3_galerkin_double_integration import (
    _F_out_at, _G_in_at, K_INF, R_THIN, SIG_T,
)


def _K_cont_pair(r: float, r_p: float, sigma: float, R: float, n_quad_mu: int) -> float:
    """K(r, r') with VISIBILITY-CONE + u²-SUBSTITUTION (corrected version).

    Fixes the original plain-GL bug discovered in test_audit_a (off-diagonal
    pairs were Q-oscillating because GL on [0, 1] missed the visibility step
    discontinuity at µ_visible = max(µ_min(r), µ_min(r')) and the endpoint
    sqrt-singularity at µ_visible).

    Substitution: u ∈ [0, 1], µ = µ_lo + (1 - µ_lo)·u², dµ = 2(1-µ_lo)·u du.
    The Jacobian factor `u du` regularises the 1/√(µ-µ_lo) endpoint
    singularity for off-diagonal pairs.

    DIAGONAL r = r' STILL log-diverges with Q (M2 finding) — the substitution
    cures the off-diagonal sqrt-singularity but not the diagonal log-divergence.
    """
    R2 = R * R
    mu_min_i = float(np.sqrt(max(0.0, 1.0 - r * r / R2)))
    mu_min_j = float(np.sqrt(max(0.0, 1.0 - r_p * r_p / R2)))
    mu_lo = max(mu_min_i, mu_min_j)
    if mu_lo >= 1.0 - 1e-15:
        return 0.0
    nodes, wts = np.polynomial.legendre.leggauss(n_quad_mu)
    u_pts = 0.5 * (nodes + 1.0)
    u_wts = 0.5 * wts
    mu_pts = mu_lo + (1.0 - mu_lo) * u_pts * u_pts
    dmu_du = 2.0 * (1.0 - mu_lo) * u_pts
    a = sigma * R
    arg = sigma * 2.0 * R * mu_pts
    f = np.where(arg > 1e-8,
                 mu_pts / (1.0 - np.exp(-arg)),
                 1.0 / (2.0 * a) + 0.5 * mu_pts)
    G = np.array([_G_in_at(r, R, sigma, m) for m in mu_pts])
    F = np.array([_F_out_at(r_p, R, sigma, m) for m in mu_pts])
    return float(np.sum(u_wts * dmu_du * G * F * f))


def test_audit_a_kcont_q_convergence_offdiag(capsys):
    """At r != r', µ-integral should Q-converge spectrally (M2 finding)."""
    with capsys.disabled():
        sigma, R = SIG_T, R_THIN
        # Several (r, r') pairs at varying separation
        pairs = [
            (0.5, 4.5),    # well-separated
            (1.0, 4.0),    # moderate
            (2.0, 3.0),    # close
            (2.4, 2.6),    # very close
            (2.49, 2.51),  # near-diagonal
        ]
        print(f"\n=== Audit A — K_cont µ-Q convergence at off-diagonal (r,r') ===")
        for r, rp in pairs:
            print(f"\n(r, r') = ({r}, {rp}), |r-r'| = {abs(r-rp)}")
            K_prev = None
            for Q in (16, 32, 64, 128, 256, 512, 1024):
                K = _K_cont_pair(r, rp, sigma, R, Q)
                if K_prev is not None:
                    delta = abs(K - K_prev) / max(abs(K), 1e-30)
                    print(f"  Q={Q:5d}: K = {K:+.6e},  Δrel = {delta:.3e}")
                else:
                    print(f"  Q={Q:5d}: K = {K:+.6e}")
                K_prev = K


def test_audit_b_kcont_q_at_diagonal(capsys):
    """At r = r' (DIAGONAL), expect Q-divergence per M2."""
    with capsys.disabled():
        sigma, R = SIG_T, R_THIN
        print(f"\n=== Audit B — K_cont at DIAGONAL r = r' ===")
        for r in (0.5, 1.0, 2.0, 3.0, 4.0):
            print(f"\nr = r' = {r}, µ_min = {np.sqrt(max(0, 1 - (r/R)**2))}")
            for Q in (16, 64, 256, 1024, 4096):
                K = _K_cont_pair(r, r, sigma, R, Q)
                print(f"  Q={Q:5d}: K = {K:+.6e}")


def test_audit_c_2d_panel_average(capsys):
    """Test 2-D (r, r') integration over a panel cell.

    If the 2-D integral converges (Galerkin smoothing works),
    `∫∫_panel K(r, r') dr dr'` should converge as we refine the
    panel sub-quadrature, even though K(r, r') diverges on the diagonal.

    Test on a single panel of width = R/2. Use varying n_quad_r and a
    fixed n_quad_µ.
    """
    with capsys.disabled():
        sigma, R = SIG_T, R_THIN
        # One panel from R/4 to 3R/4 (no boundary effects)
        a_panel, b_panel = R/4, 3*R/4
        h = 0.5 * (b_panel - a_panel)
        m = 0.5 * (a_panel + b_panel)
        print(f"\n=== Audit C — 2-D panel average ∫∫ K(r,r') dr dr' ===")
        print(f"Panel: [{a_panel}, {b_panel}], width={b_panel-a_panel}")
        for n_quad_µ in (32, 64, 128):
            print(f"\n  n_quad_µ = {n_quad_µ}")
            for n_quad_r in (4, 8, 16, 32):
                nodes, wts = np.polynomial.legendre.leggauss(n_quad_r)
                r_sub = h * nodes + m
                w_sub = h * wts
                # Plain 2-D Galerkin (no Lagrange weights — just total mass)
                I = 0.0
                for k1 in range(n_quad_r):
                    for k2 in range(n_quad_r):
                        K_val = _K_cont_pair(
                            r_sub[k1], r_sub[k2], sigma, R, n_quad_µ,
                        )
                        I += w_sub[k1] * w_sub[k2] * K_val
                print(f"    n_quad_r = {n_quad_r:3d}: ∫∫ K = {I:+.6e}")


def test_audit_d_kcont_separation_scan(capsys):
    """Scan K(r, r') vs |r-r'| at fixed r, large n_quad_µ.

    Expect log-singular behaviour: K(r, r) → ∞, but K(r, r+ε) finite
    with weak (log) growth as ε → 0.
    """
    with capsys.disabled():
        sigma, R = SIG_T, R_THIN
        Q = 1024
        r0 = 2.5
        print(f"\n=== Audit D — K(r, r+ε) vs ε at r={r0}, Q={Q} ===")
        print(f"  ε         | K(r, r+ε)")
        for eps in (1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001):
            K = _K_cont_pair(r0, r0 + eps, sigma, R, Q)
            print(f"  {eps:.4f}    | {K:+.6e}")
        print(f"  0.0       | {_K_cont_pair(r0, r0, sigma, R, Q):+.6e} (DIAGONAL)")
