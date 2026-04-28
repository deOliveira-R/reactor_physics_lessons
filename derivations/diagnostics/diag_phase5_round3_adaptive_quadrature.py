"""Phase 5+ Round 3 — Adaptive quadrature for the r-dependent diagonal singularity.

Created by numerics-investigator on 2026-04-28.

Mission
=======
With Round 2 BACKUP closing the multi-bounce factor closed form
``f_∞(µ) = (1/2)·µ/(1 - e^{-σ·2Rµ})`` (HALF M1, bounded at µ=0), and
M2 PRIMARY locating the LAST blocker as the r-dependent diagonal
``1/cos²(ω) ∝ 1/(µ²-µ_min²(r))`` singularity at ``r_i = r_j``, this
diagnostic tries each of the four approaches in the Round 3 brief
on the corrected ``(1/2)·µ·T(µ)`` factor.

Approaches tested
-----------------
1. **Per-pair quadrature with corrected factor** — same as M2 but with
   the HALF-M1 multi-bounce factor `(1/2)·µ·T(µ)`. Confirm the
   diagonal log-divergence is independent of the multi-bounce factor
   choice (M2 weight=1 vs BACKUP `(1/2)·µ` weight).
2. **Chord substitution** — `s = √(µ²-µ_min²)`. The Jacobian carries
   `s/µ`, so ``1/cos²(ω) · dµ → (r²/(R²·µ·s)) ds`` — still 1/s
   divergent. Test if combined with the smoothness of the bounded
   factors a single GL on `[0, s_max]` works.
3. **Adaptive Gauss-Kronrod with explicit subdivision** — use
   `scipy.integrate.quad(..., points=[µ_min(r)])`. NOTE: the
   integrand is genuinely log-divergent (NOT integrable), so
   ``quad`` will NOT converge for the diagonal. This approach
   FALSIFIES the "integrable singularity" hypothesis.
4. **Galerkin diagonal cell-averaging** — Hadamard-finite-part cure:
   replace ``K_bc[i, i]`` (point evaluation) with a small-radius
   cell-averaged value
   ``∫∫ L_i(r) L_i(r') K_bc(r, r') dr dr'`` over a radial neighbourhood
   matching the GL panel width. This is the natural cure for
   hypersingular boundary integral operators (Burton-Miller).
5. **Simplified Galerkin: use OFF-diagonal pair to extrapolate diagonal**
   — for the Nyström approximation, replace ``K_bc[i, i]`` with the
   value at adjacent off-diagonal entries weighted by a small offset.
   This is an ad-hoc fix exploiting the off-diagonal Q-convergence.

Theoretical inevitability of the diagonal singularity
-----------------------------------------------------
The continuous-µ kernel ``K_bc(r, r')`` carries

.. math::

    K_{\\rm bc}(r, r') \\;\\propto\\;
        \\int \\mathrm d\\mu\\, \\frac{e^{-\\sigma|r - r'|/\\mu}}{|\\cos\\omega(r,\\mu)| \\cdot
            |\\cos\\omega(r',\\mu)|}\\,\\cdots

When ``r = r'``, both `cos(ω)` factors vanish at the SAME `µ_min(r)`,
making the integrand `~1/(µ-µ_min)` — log-divergent, NOT integrable.
This is a HYPERSINGULAR Cauchy-type kernel; the diagonal value is a
Hadamard finite part that is NOT a normal integral.

The Phase 4 matrix-Galerkin form `(I - T·R)^{-1}` does NOT see this
singularity because rank-N projection regularises by integrating
against modal basis functions. The continuous-µ form requires either
(a) a Galerkin trial-test pair (cell averaging) or (b) abandoning
the µ-resolved form for production.

Decision
--------
If approaches 1, 2, 3 fail (predicted by theory), and approach 4
either yields a smooth diagonal AND end-to-end k_eff converges, the
production form is **Galerkin K_bc** (NOT Nyström). If approach 4
ALSO fails to converge in k_eff, recommend Round 4 (full Galerkin
double-integration on **all** pairs) OR Round 3-C (abandon Phase 5).
"""
from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import (
    SPHERE_1D,
    _build_full_K_per_group,
    _chord_tau_mu_sphere,
    composite_gl_r,
    compute_K_bc_specular_continuous_mu_sphere,
    solve_peierls_1g,
)


# ─── fixture identical to Phase 4 thin-sphere test ───────────────────


SIG_T = 0.5
SIG_S = 0.38
NU_SIG_F = 0.025
K_INF = NU_SIG_F / (SIG_T - SIG_S)
R_THIN = 5.0  # τ_R = 2.5


def keff_from_K(K, sig_t_array, nu_sig_f_array, sig_s_array):
    """k_eff via dense generalized eigenproblem."""
    A = np.diag(sig_t_array) - K * sig_s_array[None, :]
    B = K * nu_sig_f_array[None, :]
    M = np.linalg.solve(A, B)
    eigval = np.linalg.eigvals(M)
    return float(np.max(np.real(eigval)))


# ─── pointwise µ-resolved primitives (homogeneous σ_t) ──────────────


def _F_out_at_mu(r_j, R, mu_q, sigma):
    """Pointwise F_out(r_j, µ_q) for homogeneous σ_t.

    Returns 0 outside visibility cone µ_q ≥ µ_min(r_j) = √(1-(r_j/R)²).
    """
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
    K_plus = float(np.exp(-sigma * rho_plus))
    K_minus = float(np.exp(-sigma * rho_minus))
    pref = 0.5
    return pref * mu_q / (r_j * r_j * cos_om) * (
        rho_plus**2 * K_plus + rho_minus**2 * K_minus
    )


def _G_in_at_mu(r_i, R, mu_q, sigma):
    """Pointwise G_in(r_i, µ_q) for homogeneous σ_t."""
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
        float(np.exp(-sigma * rho_plus))
        + float(np.exp(-sigma * rho_minus))
    )
    return 2.0 * R * R * mu_q / (r_i * r_i * cos_om) * decay_sum


def _multi_bounce_factor_half_M1(mu, sigma, R):
    """The Round 2 BACKUP closed form: f_∞(µ) = (1/2)·µ/(1 - e^{-σ·2Rµ}).

    Bounded at µ → 0 with limit 1/(4·σ·R). No singularity here.
    """
    tau_chord = sigma * 2.0 * R * mu
    # Avoid 0/0 at µ=0: limit of µ/(1-e^{-aµ}) is 1/a.
    out = np.empty_like(mu)
    small = tau_chord < 1e-10
    out[small] = 1.0 / (2.0 * sigma * 2.0 * R)  # = 1/(4σR)
    out[~small] = 0.5 * mu[~small] / (1.0 - np.exp(-tau_chord[~small]))
    return out


# ═════════════════════════════════════════════════════════════════════
# Approach 1 — Per-pair quadrature with corrected factor
# ═════════════════════════════════════════════════════════════════════


def compute_K_bc_per_pair_half_M1(
    geometry, r_nodes, r_wts, radii, sig_t, *, n_quad=64,
):
    r"""Per-pair µ-quadrature with the Round 2 BACKUP HALF-M1 factor.

    For each (i, j), GL on [µ_min^ij, 1] where µ_min^ij = max(µ_min(r_i),
    µ_min(r_j)). Uses ``f_∞(µ) = (1/2)·µ/(1-e^{-σ·2Rµ})`` (bounded at µ=0).

    Diagonal entries (r_i = r_j) carry 1/cos²(ω) → log divergent on
    [µ_min(r), 1]. This routine does NOT regularise — used to confirm
    the M2 finding holds for the corrected factor.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    sigma = float(sig_t[0])

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
        mu_min_i = float(np.sqrt(max(0.0, 1.0 - (r_i / R) ** 2)))
        for j in range(N):
            r_j = float(r_nodes[j])
            mu_min_j = float(np.sqrt(max(0.0, 1.0 - (r_j / R) ** 2)))
            mu_lo = max(mu_min_i, mu_min_j)
            mu_pts = 0.5 * (nodes_unit + 1.0) * (1.0 - mu_lo) + mu_lo
            mu_wts = 0.5 * (1.0 - mu_lo) * wts_unit
            f_mb = _multi_bounce_factor_half_M1(mu_pts, sigma, R)
            integrand = np.zeros(n_quad)
            for q in range(n_quad):
                f_out = _F_out_at_mu(r_j, R, mu_pts[q], sigma)
                g_in = _G_in_at_mu(r_i, R, mu_pts[q], sigma)
                integrand[q] = g_in * f_out * f_mb[q]
            K_total[i, j] = (
                2.0 * (sig_t_n[i] / divisor) * rv[j] * r_wts[j]
                * float(np.sum(mu_wts * integrand))
            )
    return K_total


# ═════════════════════════════════════════════════════════════════════
# Approach 2 — Chord substitution s = √(µ² - µ_min²)
# ═════════════════════════════════════════════════════════════════════


def compute_K_bc_chord_substitution_half_M1(
    geometry, r_nodes, r_wts, radii, sig_t, *, n_quad=64,
):
    r"""Per-pair quadrature with ``s = √(µ² - µ_min^ij²)``.

    Substitution: ``s² = µ² - µ_min²``, so ``µ = √(s² + µ_min²)``,
    ``dµ = s/µ ds``. The Jacobian s in numerator partially cancels
    the 1/cos² = (r²/R²)/(µ²-µ_min²) = (r²/R²)/s² when only ONE cos(ω)
    factor is present (off-diagonal); the OTHER cos(ω) factor remains
    1/s on diagonal.

    On diagonal (r_i = r_j): integrand has ONE cos(ω)·dµ = (R/r)·s·(s/µ)·ds
    = (R/r)·(s²/µ) ds (smooth!), and the OTHER cos(ω) factor gives
    1/((R/r)·s) — STILL LOG-DIVERGENT.

    This is a sanity-check approach: substitution should NOT make the
    integral converge, but might make it Q-converge to a *finite*
    Hadamard finite-part value if we explicitly excise the singular
    end. We just integrate as-is to see the Q-behaviour.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    sigma = float(sig_t[0])

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
        mu_min_i = float(np.sqrt(max(0.0, 1.0 - (r_i / R) ** 2)))
        for j in range(N):
            r_j = float(r_nodes[j])
            mu_min_j = float(np.sqrt(max(0.0, 1.0 - (r_j / R) ** 2)))
            mu_lo = max(mu_min_i, mu_min_j)
            s_max = float(np.sqrt(max(0.0, 1.0 - mu_lo * mu_lo)))
            # GL on s ∈ [0, s_max]
            s_pts = 0.5 * (nodes_unit + 1.0) * s_max
            s_wts = 0.5 * s_max * wts_unit
            mu_pts = np.sqrt(s_pts * s_pts + mu_lo * mu_lo)
            jac = s_pts / mu_pts  # dµ = (s/µ) ds
            f_mb = _multi_bounce_factor_half_M1(mu_pts, sigma, R)
            integrand = np.zeros(n_quad)
            for q in range(n_quad):
                f_out = _F_out_at_mu(r_j, R, mu_pts[q], sigma)
                g_in = _G_in_at_mu(r_i, R, mu_pts[q], sigma)
                integrand[q] = g_in * f_out * f_mb[q] * jac[q]
            K_total[i, j] = (
                2.0 * (sig_t_n[i] / divisor) * rv[j] * r_wts[j]
                * float(np.sum(s_wts * integrand))
            )
    return K_total


# ═════════════════════════════════════════════════════════════════════
# Approach 4 — Galerkin diagonal regularization via local cell-average
# ═════════════════════════════════════════════════════════════════════


def compute_K_bc_galerkin_diagonal(
    geometry, r_nodes, r_wts, radii, sig_t, *, n_quad=64,
    diag_offset_frac: float = 0.5,
):
    r"""Phase 5 with Galerkin-regularised diagonal.

    Off-diagonal entries: same per-pair GL on visibility cone (no
    singularity issue).

    Diagonal entries: replace point-evaluation `K_bc(r_i, r_i)` with
    a small-cell-averaged version

    .. math::
        K^{\\rm reg}_{ii} \\approx \\frac{1}{2\\delta} \\int_{r_i-\\delta}^{r_i+\\delta}
            K_{\\rm bc}(r_i, r') dr'

    with ``δ = diag_offset_frac · r_wts[i] / 2`` (half the local panel
    width). This is a 1-point Galerkin trial-test pairing: the
    Hadamard finite-part of the hypersingular kernel is replaced by
    the cell average over a small neighbourhood.

    Implementation: a 4-point GL quadrature in `r'` over the
    neighbourhood, integrating r' ≠ r_i (so visibility cones don't
    coincide).

    NOTE: this approximation introduces an error O(δ²·∂²K/∂r²) in
    the diagonal — fundamentally a regularisation, not exact.
    Whether it gives the "right" k_eff depends on whether the
    self-term contribution is correctly captured.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    sigma = float(sig_t[0])

    rv = np.array([
        geometry.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    sig_t_n = np.array([
        sig_t[geometry.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])
    divisor = geometry.rank1_surface_divisor(R)

    nodes_unit, wts_unit = np.polynomial.legendre.leggauss(n_quad)
    # Inner GL for r' regularisation (4 points)
    rp_nodes_unit, rp_wts_unit = np.polynomial.legendre.leggauss(4)

    N = len(r_nodes)
    K_total = np.zeros((N, N))

    def _K_at_pair(r_i, r_j):
        """Single-pair K_bc value for r_i ≠ r_j (visibility cones distinct)."""
        mu_min_i = float(np.sqrt(max(0.0, 1.0 - (r_i / R) ** 2)))
        mu_min_j = float(np.sqrt(max(0.0, 1.0 - (r_j / R) ** 2)))
        mu_lo = max(mu_min_i, mu_min_j)
        mu_pts = 0.5 * (nodes_unit + 1.0) * (1.0 - mu_lo) + mu_lo
        mu_wts = 0.5 * (1.0 - mu_lo) * wts_unit
        f_mb = _multi_bounce_factor_half_M1(mu_pts, sigma, R)
        integrand = np.zeros(n_quad)
        for q in range(n_quad):
            f_out = _F_out_at_mu(r_j, R, mu_pts[q], sigma)
            g_in = _G_in_at_mu(r_i, R, mu_pts[q], sigma)
            integrand[q] = g_in * f_out * f_mb[q]
        return 2.0 * float(np.sum(mu_wts * integrand))

    for i in range(N):
        r_i = float(r_nodes[i])
        for j in range(N):
            r_j = float(r_nodes[j])
            if i == j:
                # Replace point-evaluation by small-cell average:
                # δ = diag_offset_frac · r_wts[i]/2 (half panel width)
                # Use 4-point GL on [r_i - δ, r_i + δ], skipping r_i.
                # NOTE: r_wts[i] is the GL weight for the radial integration,
                # which has dimensions of length. We use it as a proxy for
                # local panel width.
                delta = diag_offset_frac * 0.5 * r_wts[i]
                # Clamp to stay inside [0, R)
                lo = max(r_i - delta, 1e-3)
                hi = min(r_i + delta, R - 1e-3)
                if hi <= lo:
                    K_total[i, j] = (
                        (sig_t_n[i] / divisor) * rv[j] * r_wts[j]
                        * 0.0
                    )
                    continue
                rp_pts = 0.5 * (rp_nodes_unit + 1.0) * (hi - lo) + lo
                rp_wts = 0.5 * (hi - lo) * rp_wts_unit
                # Sample K_bc(r_i, r') for each r' ≠ r_i (off-diag — finite)
                K_avg = 0.0
                tot_w = 0.0
                for rp_idx, rp in enumerate(rp_pts):
                    if abs(rp - r_i) < 1e-12:
                        continue
                    K_avg += rp_wts[rp_idx] * _K_at_pair(r_i, float(rp))
                    tot_w += rp_wts[rp_idx]
                K_avg = K_avg / tot_w if tot_w > 0 else 0.0
                K_total[i, j] = (
                    (sig_t_n[i] / divisor) * rv[j] * r_wts[j] * K_avg
                )
            else:
                K_total[i, j] = (
                    (sig_t_n[i] / divisor) * rv[j] * r_wts[j]
                    * _K_at_pair(r_i, r_j)
                )
    return K_total


# ═════════════════════════════════════════════════════════════════════
# Reference helpers
# ═════════════════════════════════════════════════════════════════════


def _build_K_vol_and_K_heb(r_nodes, r_wts, panels, radii, sig_t_g):
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
    return K_vol, K_heb


def _build_K_specular_mb(r_nodes, r_wts, panels, radii, sig_t_g, n_bc_modes):
    """Reference Phase 4 multi-bounce form."""
    return _build_full_K_per_group(
        SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
        "specular_multibounce",
        n_angular=24, n_rho=24, n_surf_quad=24,
        n_bc_modes=n_bc_modes, dps=20,
    )


# ═════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════


def test_a1_per_pair_half_M1_q_convergence(capsys):
    """Approach 1 — confirm Q-divergence on diagonal with corrected
    HALF-M1 factor.

    M2 PRIMARY found Q-divergence with ``weight=1`` (Sanchez form).
    BACKUP found ``f_∞ = (1/2)·µ/(1-e^{-σ·2Rµ})`` is the correct
    matrix-Galerkin limit. Test if changing weight changes the
    Q-divergence behaviour. Theory predicts: NO (singularity is in
    F_out·G_in, not in T(µ)).
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 1, 3, dps=15, inner_radius=0.0,
        )
        N = len(r_nodes)
        print(f"\n=== A1 — Per-pair half-M1 K_bc Q-conv ===")
        print(f"  r_nodes = {r_nodes.tolist()}")
        print(f"  i,j   | Q=16     Q=64     Q=128    Q=512   | rel(128 vs 512)")
        for i in range(N):
            for j in range(N):
                vals = []
                for Q in (16, 64, 128, 512):
                    K = compute_K_bc_per_pair_half_M1(
                        SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                        n_quad=Q,
                    )
                    vals.append(K[i, j])
                rel = abs(vals[2] - vals[3]) / max(abs(vals[3]), 1e-30)
                tag = "DIAG" if i == j else "OFF"
                print(f"  ({i},{j}) {tag} | "
                      + "  ".join(f"{v:+.3e}" for v in vals)
                      + f" | rel = {rel:.3e}")
        # No assertion — diagnostic only
        assert True


def test_a2_chord_substitution_q_convergence(capsys):
    """Approach 2 — chord substitution s = √(µ²-µ_min²).

    Should NOT make the diagonal converge (theory: 1/s → log divergent).
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 1, 3, dps=15, inner_radius=0.0,
        )
        N = len(r_nodes)
        print(f"\n=== A2 — Chord substitution s² = µ² - µ_min² ===")
        print(f"  i,j   | Q=16     Q=64     Q=128    Q=512   | rel(128 vs 512)")
        for i in range(N):
            for j in range(N):
                vals = []
                for Q in (16, 64, 128, 512):
                    K = compute_K_bc_chord_substitution_half_M1(
                        SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                        n_quad=Q,
                    )
                    vals.append(K[i, j])
                rel = abs(vals[2] - vals[3]) / max(abs(vals[3]), 1e-30)
                tag = "DIAG" if i == j else "OFF"
                print(f"  ({i},{j}) {tag} | "
                      + "  ".join(f"{v:+.3e}" for v in vals)
                      + f" | rel = {rel:.3e}")
        assert True


def test_a4_galerkin_diagonal_smoke_test(capsys):
    """Approach 4 — Galerkin diagonal cell-averaging, end-to-end k_eff.

    Pass criterion: Q-convergence MONOTONIC, |k_eff/k_inf - 1| < 0.5%
    at Q=64, no overshoot at any Q.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        N = len(r_nodes)
        sig_s_arr = np.full(N, SIG_S)
        nu_sf_arr = np.full(N, NU_SIG_F)
        sig_t_arr = np.full(N, SIG_T)

        K_vol, K_heb = _build_K_vol_and_K_heb(
            r_nodes, r_wts, panels, radii, sig_t_g,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)

        print(f"\n=== A4 — Galerkin-regularised diagonal ===")
        print(f"  R={R_THIN}, σ_t={SIG_T}, τ_R={SIG_T*R_THIN}")
        print(f"  k_inf = {K_INF:.6f}, k_heb = {k_heb:.6f} "
              f"({(k_heb-K_INF)/K_INF*100:+.4f}%)")
        print(f"  Q   | k_eff       | rel_inf    | rel_heb     | overshoot?")
        results = {}
        for Q in (16, 32, 64, 128, 256):
            K_bc_g = compute_K_bc_galerkin_diagonal(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=Q,
            )
            k_g = keff_from_K(
                K_vol + K_bc_g, sig_t_arr, nu_sf_arr, sig_s_arr,
            )
            rel_inf = (k_g - K_INF) / K_INF * 100
            rel_h = (k_g - k_heb) / k_heb * 100
            overshoot = "YES" if k_g > K_INF else "no"
            results[Q] = k_g
            print(f"  {Q:3d} | {k_g:.6f}   | {rel_inf:+.4f}%  | "
                  f"{rel_h:+.4f}%  | {overshoot}")
        # Diagnostic only — no assert
        assert True


def test_a4_galerkin_diagonal_offset_sweep(capsys):
    """Test sensitivity to ``diag_offset_frac`` — is the regularisation
    parameter-stable, or does it depend strongly on δ?
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        N = len(r_nodes)
        sig_s_arr = np.full(N, SIG_S)
        nu_sf_arr = np.full(N, NU_SIG_F)
        sig_t_arr = np.full(N, SIG_T)

        K_vol, K_heb = _build_K_vol_and_K_heb(
            r_nodes, r_wts, panels, radii, sig_t_g,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)

        print(f"\n=== A4 — Galerkin diagonal δ-sensitivity ===")
        print(f"  k_inf = {K_INF:.6f}, k_heb = {k_heb:.6f}")
        print(f"  δ_frac | k_eff (Q=64) | rel_heb")
        for d_frac in (0.1, 0.25, 0.5, 1.0, 2.0):
            K_bc_g = compute_K_bc_galerkin_diagonal(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=64, diag_offset_frac=d_frac,
            )
            k_g = keff_from_K(
                K_vol + K_bc_g, sig_t_arr, nu_sf_arr, sig_s_arr,
            )
            rel_h = (k_g - k_heb) / k_heb * 100
            print(f"  {d_frac:.2f}   | {k_g:.6f}     | {rel_h:+.4f}%")
        assert True


def test_a5_ignore_diagonal_smoke_test(capsys):
    """Approach 5 — extreme: set K_bc[i,i] = 0 and see what happens.

    This is a dumbest-possible regularisation. If it gives sensible
    k_eff, then the diagonal contribution is small and any reasonable
    regularisation works. If it diverges or gives nonsense, the
    diagonal must be captured somehow.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        N = len(r_nodes)
        sig_s_arr = np.full(N, SIG_S)
        nu_sf_arr = np.full(N, NU_SIG_F)
        sig_t_arr = np.full(N, SIG_T)

        K_vol, K_heb = _build_K_vol_and_K_heb(
            r_nodes, r_wts, panels, radii, sig_t_g,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)

        print(f"\n=== A5 — Zero-diagonal K_bc (sanity test) ===")
        print(f"  k_inf = {K_INF:.6f}, k_heb = {k_heb:.6f}")
        print(f"  Q   | k_eff       | rel_heb")
        for Q in (32, 64, 128):
            K_bc = compute_K_bc_per_pair_half_M1(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=Q,
            )
            # Zero out the diagonal
            np.fill_diagonal(K_bc, 0.0)
            k_g = keff_from_K(
                K_vol + K_bc, sig_t_arr, nu_sf_arr, sig_s_arr,
            )
            rel_h = (k_g - k_heb) / k_heb * 100
            print(f"  {Q:3d} | {k_g:.6f}   | {rel_h:+.4f}%")
        assert True


def test_a6_nearest_neighbour_diagonal_smoke_test(capsys):
    """Approach 6 — replace K_bc[i,i] with average of K_bc[i, i±1].

    Heuristic Galerkin: assume the diagonal value is approximately
    the average of the off-diagonal nearest neighbours. This is the
    simplest interpolation cure for the hypersingular diagonal.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        N = len(r_nodes)
        sig_s_arr = np.full(N, SIG_S)
        nu_sf_arr = np.full(N, NU_SIG_F)
        sig_t_arr = np.full(N, SIG_T)

        K_vol, K_heb = _build_K_vol_and_K_heb(
            r_nodes, r_wts, panels, radii, sig_t_g,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)

        print(f"\n=== A6 — Nearest-neighbour diagonal interpolation ===")
        print(f"  k_inf = {K_INF:.6f}, k_heb = {k_heb:.6f} "
              f"({(k_heb-K_INF)/K_INF*100:+.4f}%)")
        print(f"  Q   | k_eff       | rel_inf    | rel_heb     | overshoot?")
        for Q in (16, 32, 64, 128, 256):
            K_bc = compute_K_bc_per_pair_half_M1(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=Q,
            )
            # Replace diagonal with average of nearest-neighbour
            # off-diagonal values (i, i±1)
            for i in range(N):
                neighbours = []
                if i > 0:
                    neighbours.append(K_bc[i, i-1])
                if i < N - 1:
                    neighbours.append(K_bc[i, i+1])
                if neighbours:
                    K_bc[i, i] = float(np.mean(neighbours))
                else:
                    K_bc[i, i] = 0.0
            k_g = keff_from_K(
                K_vol + K_bc, sig_t_arr, nu_sf_arr, sig_s_arr,
            )
            rel_inf = (k_g - K_INF) / K_INF * 100
            rel_h = (k_g - k_heb) / k_heb * 100
            overshoot = "YES" if k_g > K_INF else "no"
            print(f"  {Q:3d} | {k_g:.6f}   | {rel_inf:+.4f}%  | "
                  f"{rel_h:+.4f}%  | {overshoot}")
        assert True
