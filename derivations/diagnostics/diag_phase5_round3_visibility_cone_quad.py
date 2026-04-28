"""Diagnostic: per-pair µ-quadrature on visibility cone with endpoint-aware substitution.

Created by numerics-investigator on 2026-04-28.

The Round 3 SECONDARY mission (Galerkin double-integration) requires a
correctly-converging per-pair K_continuous(r_i, r_j). Audit revealed that
plain GL on [0, 1] does NOT converge even off-diagonal because:

1. Visibility cone: K_cont(r_i, r_j, µ) = 0 for µ < µ_visible(r_i, r_j) =
   max(µ_min(r_i), µ_min(r_j)). The integrand has a STEP DISCONTINUITY
   at µ_visible. Plain GL on [0, 1] misses this.

2. Endpoint singularity: at µ = µ_visible, the larger µ_min cos(ω) → 0,
   giving an integrable square-root-type singularity. Plain GL has poor
   convergence for square-root endpoint singularities.

3. At r_i = r_j (DIAGONAL), BOTH cos(ω) factors vanish at µ_visible,
   giving a NON-integrable 1/(µ-µ_min) log-divergent singularity. This
   is the M2 finding — the diagonal is fundamentally singular.

This diagnostic builds a fixed-up `_K_cont_pair_visibility(r_i, r_j, σ, R, Q)`
that:
- Restricts µ to [µ_visible, 1]
- Uses the substitution ν = √(1 - µ²/(1 - µ_visible²)) ∈ [0, 1] which
  maps the endpoint singularity to a regular integrand on [0, 1]
- Falls back to a finite-part regularisation on the diagonal (or excludes
  the diagonal contribution if Galerkin smoothing absorbs it)

Then re-test Audit A (off-diagonal Q-convergence) to verify before
attempting the Galerkin assembly.
"""
from __future__ import annotations

import numpy as np
import pytest

from diag_phase5_round3_galerkin_double_integration import (
    _F_out_at, _G_in_at, K_INF, R_THIN, SIG_T,
)


def _K_cont_pair_visibility(
    r: float, r_p: float, sigma: float, R: float, n_quad_mu: int,
    *, exclude_diag_eps: float = 1e-12,
) -> float:
    """K(r, r') with µ-quadrature on the visibility cone.

    Restricts µ to [µ_visible, 1] where µ_visible =
    max(µ_min(r), µ_min(r')) = max[√(1-(r/R)²), √(1-(r'/R)²)].

    Uses GL on the restricted interval. NO endpoint substitution yet —
    just to see if the visibility-cone restriction alone makes
    off-diagonal pairs Q-convergent.

    For r ≈ r' (diagonal), this still diverges because µ_min_i = µ_min_j
    and BOTH cos factors vanish at the endpoint.
    """
    R2 = R * R
    mu_min_i = float(np.sqrt(max(0.0, 1.0 - r * r / R2)))
    mu_min_j = float(np.sqrt(max(0.0, 1.0 - r_p * r_p / R2)))
    mu_lo = max(mu_min_i, mu_min_j)
    if mu_lo >= 1.0 - 1e-15:
        return 0.0
    nodes, wts = np.polynomial.legendre.leggauss(n_quad_mu)
    half = 0.5 * (1.0 - mu_lo)
    mid = 0.5 * (1.0 + mu_lo)
    mu_pts = half * nodes + mid
    mu_wts = half * wts
    a = sigma * R
    arg = sigma * 2.0 * R * mu_pts
    f = np.where(arg > 1e-8,
                 mu_pts / (1.0 - np.exp(-arg)),
                 1.0 / (2.0 * a) + 0.5 * mu_pts)
    G = np.array([_G_in_at(r, R, sigma, m) for m in mu_pts])
    F = np.array([_F_out_at(r_p, R, sigma, m) for m in mu_pts])
    return float(np.sum(mu_wts * G * F * f))


def _K_cont_pair_visibility_substitution(
    r: float, r_p: float, sigma: float, R: float, n_quad_mu: int,
) -> float:
    r"""K(r, r') with substitution ν: µ = µ_lo·cos²(θ) + sin²(θ),
    ν = sin(θ), µ-Jacobian dµ/dν = 2(1-µ_lo)·sin(θ)·cos(θ) = 2(1-µ_lo)·ν·√(1-ν²).

    Actually let's use a cleaner substitution: u² = (µ - µ_lo)/(1 - µ_lo),
    µ = µ_lo + (1 - µ_lo)·u². Jacobian dµ = 2(1-µ_lo)·u·du.

    Near µ = µ_lo, the integrand has 1/√(µ²-µ_lo²) ≈ 1/√(2µ_lo(µ-µ_lo))
    behaviour (off-diagonal: only ONE cos vanishes). The Jacobian factor
    `u du` cancels the 1/√(µ-µ_lo) singularity — integrand becomes smooth
    in u ∈ [0, 1].

    For DIAGONAL (r = r'), both factors vanish at µ_lo. Then the
    integrand ~ 1/(µ-µ_lo) and the substitution gives integrand ~ 1/u
    which is still log-divergent. So the substitution helps off-diagonal
    only.
    """
    R2 = R * R
    mu_min_i = float(np.sqrt(max(0.0, 1.0 - r * r / R2)))
    mu_min_j = float(np.sqrt(max(0.0, 1.0 - r_p * r_p / R2)))
    mu_lo = max(mu_min_i, mu_min_j)
    if mu_lo >= 1.0 - 1e-15:
        return 0.0
    # Substitution: u ∈ [0, 1], µ = µ_lo + (1 - µ_lo) u², dµ = 2(1-µ_lo)u du
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


def test_visibility_a_offdiag_qconv(capsys):
    """Off-diagonal pairs should Q-converge with visibility-cone GL."""
    with capsys.disabled():
        sigma, R = SIG_T, R_THIN
        pairs = [
            (0.5, 4.5),    # well-separated
            (1.0, 4.0),
            (2.0, 3.0),
            (2.4, 2.6),
            (2.49, 2.51),
        ]
        print(f"\n=== Visibility A — vis-cone GL (no substitution) ===")
        for r, rp in pairs:
            print(f"\n(r, r') = ({r}, {rp}), |r-r'| = {abs(r-rp)}")
            print(f"  µ_min_i={np.sqrt(max(0,1-(r/R)**2)):.4f}, "
                  f"µ_min_j={np.sqrt(max(0,1-(rp/R)**2)):.4f}")
            K_prev = None
            for Q in (16, 32, 64, 128, 256, 512):
                K = _K_cont_pair_visibility(r, rp, sigma, R, Q)
                if K_prev is not None:
                    delta = abs(K - K_prev) / max(abs(K), 1e-30)
                    print(f"  Q={Q:5d}: K = {K:+.6e},  Δrel = {delta:.3e}")
                else:
                    print(f"  Q={Q:5d}: K = {K:+.6e}")
                K_prev = K


def test_visibility_b_offdiag_qconv_substitution(capsys):
    """With u² substitution, off-diagonal should converge much faster."""
    with capsys.disabled():
        sigma, R = SIG_T, R_THIN
        pairs = [
            (0.5, 4.5),
            (1.0, 4.0),
            (2.0, 3.0),
            (2.4, 2.6),
            (2.49, 2.51),
        ]
        print(f"\n=== Visibility B — vis-cone GL with u² substitution ===")
        for r, rp in pairs:
            print(f"\n(r, r') = ({r}, {rp}), |r-r'| = {abs(r-rp)}")
            K_prev = None
            for Q in (16, 32, 64, 128, 256, 512):
                K = _K_cont_pair_visibility_substitution(
                    r, rp, sigma, R, Q,
                )
                if K_prev is not None:
                    delta = abs(K - K_prev) / max(abs(K), 1e-30)
                    print(f"  Q={Q:5d}: K = {K:+.6e},  Δrel = {delta:.3e}")
                else:
                    print(f"  Q={Q:5d}: K = {K:+.6e}")
                K_prev = K


def test_visibility_c_diag_under_subst(capsys):
    """At diagonal r = r', confirm the substitution does NOT cure the
    log-divergence (M2 finding)."""
    with capsys.disabled():
        sigma, R = SIG_T, R_THIN
        print(f"\n=== Visibility C — DIAGONAL r=r' Q-divergence ===")
        for r in (0.5, 1.0, 2.0, 3.0, 4.0):
            print(f"\nr = r' = {r}")
            for Q in (16, 64, 256, 1024):
                K = _K_cont_pair_visibility_substitution(
                    r, r, sigma, R, Q,
                )
                print(f"  Q={Q:5d}: K = {K:+.6e}")
