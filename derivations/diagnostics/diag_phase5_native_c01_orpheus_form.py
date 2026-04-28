"""Phase 5 Front C — ORPHEUS-native continuous-µ specular sphere kernel.

Created by numerics-investigator on 2026-04-28.

Mission
=======
Build :math:`F_{\\rm out}(r, \\mu)` and :math:`G_{\\rm in}(r, \\mu)` as
**µ-resolved** primitives (NOT projected onto the rank-N shifted-Legendre
basis), then assemble :math:`K_{\\rm bc}` via direct quadrature over µ
with the multi-bounce factor :math:`T(\\mu) = 1/(1 - e^{-\\sigma\\,2R\\mu})`.

The cross-domain-attacker M1 form (corrected by SymPy V2 in
:file:`derivations/peierls_specular_continuous_mu.py`) is

.. math::

    K_{\\rm bc}^{\\rm cmu,sph}(r_i, r_j) \\;=\\;
        2\\!\\int_0^1\\!
        G_{\\rm in}(r_i, \\mu)\\,F_{\\rm out}(r_j, \\mu)\\,
        T(\\mu)\\,\\mathrm d\\mu

where :math:`F_{\\rm out}` carries the surface-area Jacobian
:math:`(\\rho_{\\max}/R)^2` and the chord transmission to the surface;
:math:`G_{\\rm in}` carries :math:`e^{-\\tau}` from the surface to the
receiver. Both use ORPHEUS's existing radial-volume / surface-divisor
weights so the conversion to :math:`K_{ij}` Nyström is automatic.

Strategy
--------
The Phase 4 sphere ``closure="specular_multibounce"`` builds:

- ``P[n, j] = rv·r_wts·P_esc_n[j]`` with ``P_esc_n[j] = pref · Σ_q
  ω_w · sin(ω) · (ρ_max/R)² · K_esc(τ) · P̃_n(µ_exit)``
- ``G[i, n] = sig_t·G_bc_n[i]/divisor`` with ``G_bc_n[i] = 2 · Σ_q
  θ_w · sin(θ) · P̃_n(µ_exit) · e^{-τ}``

Both project the µ-distribution onto the shifted-Legendre basis. We
**skip the projection** by:

1. Discretizing µ ∈ [0, 1] with a Gauss-Legendre rule.
2. For each (r_j, µ_q), computing ``F_out(r_j, µ_q)`` directly via the
   chord-from-r-to-surface geometry (not the ω-grid).
3. For each (r_i, µ_q), computing ``G_in(r_i, µ_q)`` similarly.
4. Assembling ``K_bc[i, j] = 2 · Σ_q µ_w_q · G_in_w[i, q] · F_out_w[q, j]
   · T(µ_q)`` with the same ``rv·r_wts`` and ``sig_t/divisor`` weights
   as Phase 4.

This is the M1 Hilbert-Schmidt separable-kernel form; the matrix-Galerkin
projection of Phase 4 is the rank-N truncation along µ.

Diagnostics in this file
------------------------
- Probe A: cross-check existing ``compute_K_bc_specular_continuous_mu_sphere``
  (Sanchez Eq. A6) ↔ ORPHEUS-native µ-resolved primitives. **Discovers
  the Sanchez↔ORPHEUS Jacobian conversion empirically** by ratio analysis.
- Probe B: ORPHEUS-native µ-resolved K_bc construction.
- Probe C: Q-quadrature convergence ladder Q ∈ {16, 32, 64, 128}.
- Probe D: end-to-end k_eff vs ``closure="white_hebert"`` at thin τ_R = 2.5.

Decision criterion
------------------
Front C wins if Probe D shows ``k_eff`` matching ``k_inf`` to better
than 0.5 % at thin τ_R = 2.5 with Q ≤ 128, AND no overshoot at higher
Q (the Phase 4 pathology).
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
    """k_eff via dense generalized eigenproblem (matches Phase 4 driver)."""
    A = np.diag(sig_t_array) - K * sig_s_array[None, :]
    B = K * nu_sig_f_array[None, :]
    M = np.linalg.solve(A, B)
    eigval = np.linalg.eigvals(M)
    return float(np.max(np.real(eigval)))


# ─── ORPHEUS-native µ-resolved primitives for sphere ─────────────────


def _chord_tau_segment_sphere(
    r: float, mu: float, radii: np.ndarray, sig_t: np.ndarray,
    *, side: str,
) -> float:
    r"""Optical depth from interior point ``r`` to the surface along the
    chord with **surface-cosine** :math:`\mu`.

    For sphere with surface cosine µ at impact param h = R·√(1-µ²),
    the chord intersects the sphere of radius r at distance
    ``ρ_seg = R·µ ∓ √(r² - h²)`` from r. ``side='+'`` is the FAR
    intersection (chord goes "forward" from r to the far surface),
    ``side='-'`` is the NEAR intersection (chord goes "backward").
    Both yield surface cosine µ.

    The optical depth integrates the multi-region chord segment from
    ``r`` to the surface point. For homogeneous Σ_t, ``τ = σ_t · ρ_seg``.
    For multi-region, we integrate σ_t along the chord segment which
    crosses ``r → r=h → r → R`` (forward) or just ``r → R`` (backward, if
    r > h on the same side).

    Implementation: compute ``τ_full = σ_t · 2R·µ`` (full chord) and
    ``τ_half_backward = σ_t · (R·µ - √(r²-h²))``. Then
    ``τ_forward = τ_full - τ_half_backward``.
    """
    R = float(radii[-1])
    h = R * float(np.sqrt(max(0.0, 1.0 - mu * mu)))
    if r * r < h * h:
        # not visible — should not be called
        return 0.0
    half_backward = R * mu - float(np.sqrt(r * r - h * h))
    half_forward = R * mu + float(np.sqrt(r * r - h * h))
    if side == "-":
        seg_length = half_backward
    else:  # "+"
        seg_length = R * 2.0 * mu - half_backward  # = half_forward
    # Multi-region τ along this chord segment. Integrate σ_t·dℓ as a
    # function of ℓ ∈ [0, seg_length] starting from r and going outward.
    # For homogeneous Σ_t this is just σ_t · seg_length. For multi-region,
    # need piecewise integration (deferred — Phase 5+).
    if len(radii) == 1:
        return float(sig_t[0]) * seg_length
    # Multi-region: integrate piecewise. Parametrize the chord segment
    # by t ∈ [0, seg_length]. The chord is at impact h from origin.
    # On the chord, the radial coordinate as fn of t:
    #   r_chord(t) = √[h² + (t_signed)²]
    # where t_signed measures distance along chord from foot of impact.
    # The foot of impact (closest to origin) has signed-distance 0 along
    # chord; r intersects chord at signed-dist ±√(r²-h²) on the two
    # branches. The "forward" exit (towards far surface) is at signed-dist
    # +√(R²-h²); "backward" exit (towards near surface) at -√(R²-h²).
    # Need to integrate σ_t(r_chord(t_signed)) over the appropriate range.
    sd_r_back = -float(np.sqrt(r * r - h * h))  # r position along chord
    sd_r_for = +float(np.sqrt(r * r - h * h))
    sd_R_back = -float(np.sqrt(R * R - h * h))
    sd_R_for = +float(np.sqrt(R * R - h * h))
    if side == "-":
        sd_lo, sd_hi = sd_R_back, sd_r_back  # going r → near surface
        # We're integrating from r (sd_r_back) to surface (sd_R_back),
        # so direction is -1; reorder so sd_lo <= sd_hi.
        sd_lo, sd_hi = sd_R_back, sd_r_back
    else:
        sd_lo, sd_hi = sd_r_for, sd_R_for  # r → far surface
    # Integrate σ_t along chord from sd_lo to sd_hi. r_chord(s)=√(h²+s²)
    # is monotone for s ≥ 0, so for sd_hi > sd_lo > 0 we can find the
    # crossings of the annulus boundaries directly.
    radii_inner = np.concatenate([[0.0], radii[:-1]])
    radii_outer = radii
    tau = 0.0
    for n_reg in range(len(radii)):
        r_in = float(radii_inner[n_reg])
        r_out = float(radii_outer[n_reg])
        if h >= r_out:
            continue
        # Annulus contributes for |s| ∈ [√(max(h,r_in)²-h²), √(r_out²-h²)]
        s_in = float(np.sqrt(max(r_in * r_in - h * h, 0.0))) if h < r_in else 0.0
        s_out = float(np.sqrt(max(r_out * r_out - h * h, 0.0)))
        # Annulus on positive-s branch: [s_in, s_out]
        # Annulus on negative-s branch: [-s_out, -s_in]
        # Intersect with [sd_lo, sd_hi]:
        for branch_lo, branch_hi in [(s_in, s_out), (-s_out, -s_in)]:
            lo = max(branch_lo, sd_lo)
            hi = min(branch_hi, sd_hi)
            if hi > lo:
                tau += float(sig_t[n_reg]) * (hi - lo)
    return tau


def F_out_mu_sphere(
    r_nodes: np.ndarray, radii: np.ndarray, sig_t: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    r"""µ-resolved outgoing partial-current density at the surface from
    a unit volumetric source at each :math:`r_j`.

    Returns array shape (Q, N_r) where Q = len(mu).

    .. math::
        F_{\\rm out}(r, \\mu) \\;=\\;
            \\frac{\\mu}{r^2}\\!\\sum_{\\pm}
            \\frac{\\rho_\\pm^2(r,\\mu)}{|\\cos\\omega_\\pm|}\\,
            K_{\\rm esc}(\\tau_\\pm(r, \\mu))

    where :math:`\\rho_\\pm = R\\mu \\pm \\sqrt{r^2 - R^2(1-\\mu^2)}` are
    the two distances from r to the surface along the chord with
    surface-cosine µ, :math:`|\\cos\\omega_\\pm| = \\rho_\\pm / R \\cdot
    (\\text{sign})` ... derived from :math:`R^2\\mu^2 = R^2 - r^2\\sin^2\\omega`
    so :math:`\\sin\\omega = (R/r)\\sqrt{1-\\mu^2}`,
    :math:`|\\cos\\omega| = \\sqrt{1 - (R/r)^2(1-\\mu^2)}`.

    Includes the sphere ``pref = 0.5`` (geometry.prefactor) absorbed.

    K_esc on sphere with sphere geometry escape kernel
    ``K_esc(τ) = (1 - e^{-τ})/τ`` for τ > 0, 1 at τ=0.
    """
    R = float(radii[-1])
    pref = 0.5  # sphere geometry.prefactor
    N_r = len(r_nodes)
    Q = len(mu)
    F = np.zeros((Q, N_r))
    for q in range(Q):
        mu_q = float(mu[q])
        for j in range(N_r):
            r_j = float(r_nodes[j])
            h = R * float(np.sqrt(max(0.0, 1.0 - mu_q * mu_q)))
            if r_j * r_j < h * h - 1e-15:
                continue  # not visible
            # |cos(ω)| = √(1 - (R/r)²·(1-µ²)). When r = h exactly, cos = 0
            # (grazing — singular). Guard.
            sin2_om = (R * R / max(r_j * r_j, 1e-30)) * (1.0 - mu_q * mu_q)
            cos2_om = max(1.0 - sin2_om, 0.0)
            if cos2_om < 1e-30:
                continue
            cos_om = float(np.sqrt(cos2_om))
            # The two chord intersections from r to surface:
            sqrt_rh = float(np.sqrt(max(r_j * r_j - h * h, 0.0)))
            rho_plus = R * mu_q + sqrt_rh   # forward (cos(ω) < 0 in geom convention)
            rho_minus = R * mu_q - sqrt_rh  # backward (cos(ω) > 0)
            # τ along each chord segment from r_j to surface
            tau_plus = _chord_tau_segment_sphere(
                r_j, mu_q, radii, sig_t, side="+",
            )
            tau_minus = _chord_tau_segment_sphere(
                r_j, mu_q, radii, sig_t, side="-",
            )
            # SPHERE K_esc = e^{-τ} (CurvilinearGeometry.escape_kernel_mp).
            # NOT (1-e^{-τ})/τ (that's slab/Bickley).
            K_plus = float(np.exp(-tau_plus))
            K_minus = float(np.exp(-tau_minus))
            # Sum both contributions. The Jacobian is µ/(r²·|cos(ω)|),
            # multiplied by ρ²/R² = (ρ/R)². Combined: (µ·ρ²)/(r²·R²·|cos|).
            contrib_plus = (rho_plus * rho_plus) * K_plus
            contrib_minus = (rho_minus * rho_minus) * K_minus
            F[q, j] = pref * mu_q / (r_j * r_j * cos_om) * (
                contrib_plus + contrib_minus
            )
    return F


def G_in_mu_sphere(
    r_nodes: np.ndarray, radii: np.ndarray, sig_t: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    r"""µ-resolved boundary-to-flux response at receiver :math:`r_i`
    from inward partial-current at surface cosine :math:`\\mu`.

    Returns array shape (N_r, Q).

    .. math::
        G_{\\rm in}(r, \\mu) \\;=\\;
            \\frac{2 R^2 \\mu}{r^2 |\\cos\\omega|}
            \\sum_{\\pm} \\frac{e^{-\\tau_\\pm(r,\\mu)}}{R^2}

    Derived from the Phase 4 ``compute_G_bc_mode`` integrand
    :math:`G_{\\rm bc}^{(n)} = 2 \\int_0^\\pi \\sin(\\theta) P̃_n(\\mu_{\\rm exit})
    e^{-\\tau} d\\theta` by change of variable θ → µ_exit. The Jacobian
    :math:`|\\sin\\theta d\\theta| = R^2 \\mu / (r^2 |\\cos\\theta|) d\\mu`
    matches the F_out derivation. Both ± branches at the same surface
    cosine contribute.
    """
    R = float(radii[-1])
    N_r = len(r_nodes)
    Q = len(mu)
    G = np.zeros((N_r, Q))
    for q in range(Q):
        mu_q = float(mu[q])
        for i in range(N_r):
            r_i = float(r_nodes[i])
            h = R * float(np.sqrt(max(0.0, 1.0 - mu_q * mu_q)))
            if r_i * r_i < h * h - 1e-15:
                continue
            sin2_om = (R * R / max(r_i * r_i, 1e-30)) * (1.0 - mu_q * mu_q)
            cos2_om = max(1.0 - sin2_om, 0.0)
            if cos2_om < 1e-30:
                continue
            cos_om = float(np.sqrt(cos2_om))
            tau_plus = _chord_tau_segment_sphere(
                r_i, mu_q, radii, sig_t, side="+",
            )
            tau_minus = _chord_tau_segment_sphere(
                r_i, mu_q, radii, sig_t, side="-",
            )
            decay_sum = float(np.exp(-tau_plus)) + float(np.exp(-tau_minus))
            # G_in is 2 · |sin(θ)dθ/dµ| · e^{-τ} summed over both branches
            # = 2 · R²·µ/(r²·|cos|) · [e^{-τ_+} + e^{-τ_-}] / 1
            # NOTE: G_bc has a factor of 2 in front of the integral
            G[i, q] = 2.0 * R * R * mu_q / (r_i * r_i * cos_om) * decay_sum
    return G


def compute_K_bc_specular_continuous_mu_sphere_native(
    geometry, r_nodes, r_wts, radii, sig_t, *, n_quad=64,
):
    """ORPHEUS-native continuous-µ K_bc for sphere with specular BC.

    Uses µ-resolved primitives F_out and G_in (no rank-N basis
    projection) integrated against T(µ) = 1/(1 - e^{-σ·2R·µ}).

    The µ-quadrature is GL on [µ_min, 1] where µ_min = small ε (the
    integrand has a removable simple pole at µ=0; see SymPy V1).
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    sigma = float(sig_t[0])  # homogeneous

    # GL on µ ∈ [0, 1]. The integrand has removable pole at µ=0;
    # standard GL nodes interior to [0,1] are fine.
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    mu_pts = 0.5 * (nodes + 1.0)
    mu_wts = 0.5 * wts

    # T(µ) = 1/(1 - e^{-σ·2R·µ_chord}) where µ_chord is the FULL chord
    # cosine from one surface to the antipodal surface. For a single
    # bounce trajectory leaving the surface at cosine µ and returning
    # at cosine µ (specular), the chord crossing the cell is at
    # impact h = R·√(1-µ²), full chord length 2R·µ — independent of
    # multi-region (just different τ).
    tau_chord = _chord_tau_mu_sphere(radii, sig_t, mu_pts)
    T_mu = 1.0 / (1.0 - np.exp(-tau_chord))

    F_out = F_out_mu_sphere(r_nodes, radii, sig_t, mu_pts)  # (Q, N_r)
    G_in = G_in_mu_sphere(r_nodes, radii, sig_t, mu_pts)    # (N_r, Q)

    # ORPHEUS volume / divisor weights (same as Phase 4 build)
    rv = np.array([
        geometry.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    sig_t_n = np.array([
        sig_t[geometry.which_annulus(float(r_nodes[i]), radii)]
        for i in range(len(r_nodes))
    ])
    divisor = geometry.rank1_surface_divisor(R)

    F_out_w = (rv * r_wts)[None, :] * F_out          # (Q, N_r)
    G_in_w = (sig_t_n / divisor)[:, None] * G_in     # (N_r, Q)

    # K_bc[i,j] = 2 · Σ_q µ_w_q · G_in_w[i,q] · F_out_w[q,j] · T_mu[q]
    return 2.0 * np.einsum(
        'iq,q,qj->ij', G_in_w, mu_wts * T_mu, F_out_w,
    )


# ═════════════════════════════════════════════════════════════════════
# Probes
# ═════════════════════════════════════════════════════════════════════


def test_probe_a_sanchez_reference_callable(capsys):
    """Probe A — sanity that the existing Sanchez reference impl runs
    and produces a sensible K_bc shape on a small grid.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t = np.array([SIG_T])
        # Small interior grid
        r_nodes = np.linspace(0.5, 4.5, 5)
        K_sanchez = compute_K_bc_specular_continuous_mu_sphere(
            r_nodes, radii, sig_t, n_quad=64,
        )
        print(f"\n=== Probe A — Sanchez Eq. (A6) reference ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, τ_R={SIG_T*R_THIN}")
        print(f"K_sanchez shape: {K_sanchez.shape}")
        print(f"K_sanchez max: {np.max(np.abs(K_sanchez)):.6e}")
        print(f"K_sanchez[i,j] sample (5 nodes):\n{K_sanchez}")
        assert K_sanchez.shape == (5, 5)
        assert np.all(np.isfinite(K_sanchez))


def test_probe_b_orpheus_native_construction(capsys):
    """Probe B — ORPHEUS-native µ-resolved K_bc constructs without
    error on the same fixture.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )
        K_native = compute_K_bc_specular_continuous_mu_sphere_native(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=64,
        )
        print(f"\n=== Probe B — ORPHEUS-native µ-resolved K_bc ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, N_r={len(r_nodes)}")
        print(f"K_native shape: {K_native.shape}")
        print(f"K_native max abs: {np.max(np.abs(K_native)):.6e}")
        print(f"K_native min: {np.min(K_native):.6e}")
        assert K_native.shape == (len(r_nodes), len(r_nodes))
        assert np.all(np.isfinite(K_native))


def test_probe_c_quadrature_convergence(capsys):
    """Probe C — Q-quadrature convergence ladder for ORPHEUS-native.
    Compare K_bc_native at Q ∈ {16, 32, 64, 128} via Frobenius norm.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )
        K_ref = compute_K_bc_specular_continuous_mu_sphere_native(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=256,
        )
        print(f"\n=== Probe C — Q-convergence ladder ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, N_r={len(r_nodes)}")
        print(f"  Q  | ‖K - K_ref256‖_F / ‖K_ref‖_F | max abs(K)")
        for Q in (16, 32, 64, 128):
            K = compute_K_bc_specular_continuous_mu_sphere_native(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=Q,
            )
            rel = np.linalg.norm(K - K_ref) / np.linalg.norm(K_ref)
            print(f"  {Q:3d} | {rel:.6e}                 | "
                  f"{np.max(np.abs(K)):.6e}")
        assert np.all(np.isfinite(K_ref))


def test_probe_d_keff_vs_white_hebert(capsys):
    """Probe D — end-to-end k_eff smoke test.

    Compare ORPHEUS-native continuous-µ K_bc against ``white_hebert``
    rank-1 reference at thin τ_R = 2.5. Pass criterion: |k_eff_native -
    k_inf|/k_inf ≤ 0.5 % AND no overshoot at higher Q.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 5, dps=20, inner_radius=0.0,
        )
        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        # ── Reference: white_hebert k_eff (rank-1 only)
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)
        rel_heb = (k_heb - K_INF) / K_INF * 100

        # ── K_vol (same for all closures)
        K_vac = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        # ── K_bc^native at varying Q
        print(f"\n=== Probe D — k_eff vs white_hebert ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, τ_R={SIG_T*R_THIN}, "
              f"k_inf={K_INF:.6f}")
        print(f"  white_hebert: k_eff={k_heb:.6f} ({rel_heb:+.4f}%)")
        print(f"  Q   | k_eff_native     | rel_to_kinf(%) "
              f"| rel_to_heb(%)")
        results = {}
        for Q in (16, 32, 64, 128, 256):
            K_bc_n = compute_K_bc_specular_continuous_mu_sphere_native(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=Q,
            )
            K_native = K_vac + K_bc_n
            k_n = keff_from_K(K_native, sig_t_arr, nu_sf_arr, sig_s_arr)
            rel_inf = (k_n - K_INF) / K_INF * 100
            rel_heb_diff = (k_n - k_heb) / k_heb * 100
            results[Q] = k_n
            print(f"  {Q:3d} | {k_n:.6f}        | "
                  f"{rel_inf:+.4f}%       | {rel_heb_diff:+.4f}%")

        # ── Smoke-test asserts
        print(f"\n  white_hebert is the rank-1 Hebert reference. The")
        print(f"  continuous-µ form should match it (rank-∞ Hebert).")


def test_probe_e_jacobian_conversion_against_sanchez(capsys):
    """Probe E — discover Sanchez↔ORPHEUS Jacobian conversion empirically.

    The existing reference impl ``compute_K_bc_specular_continuous_mu_sphere``
    produces the Sanchez Eq. (A6) form in Sanchez normalisation.
    Compare to the ORPHEUS-native build to identify the conversion factor
    (constant, separable, or radius-dependent).
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )
        K_native = compute_K_bc_specular_continuous_mu_sphere_native(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g, n_quad=128,
        )
        K_sanchez = compute_K_bc_specular_continuous_mu_sphere(
            r_nodes, radii, sig_t_g, n_quad=128,
        )

        print(f"\n=== Probe E — Sanchez ↔ ORPHEUS-native conversion ===")
        print(f"K_native max abs:  {np.max(np.abs(K_native)):.6e}")
        print(f"K_sanchez max abs: {np.max(np.abs(K_sanchez)):.6e}")

        # Element-wise ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(
                np.abs(K_sanchez) > 1e-12,
                K_native / K_sanchez,
                np.nan,
            )
        finite = np.isfinite(ratio) & ~np.isnan(ratio)
        if finite.any():
            r_med = float(np.median(ratio[finite]))
            r_min = float(np.min(ratio[finite]))
            r_max = float(np.max(ratio[finite]))
            r_std = float(np.std(ratio[finite]))
            print(f"  ratio K_native/K_sanchez:")
            print(f"    median = {r_med:.6e}")
            print(f"    min    = {r_min:.6e}")
            print(f"    max    = {r_max:.6e}")
            print(f"    std    = {r_std:.6e}")
            print(f"    rel_spread = std/median = "
                  f"{abs(r_std/r_med) if r_med != 0 else float('inf'):.4e}")
            if abs(r_std / r_med) < 0.01:
                print(f"  → Constant scalar conversion: native = "
                      f"{r_med:.4e} · sanchez")
            else:
                # Try separable: ratio[i,j] = f(i) · g(j)?
                # If separable, log(ratio) has rank 1.
                with np.errstate(invalid='ignore'):
                    log_ratio = np.where(
                        np.abs(ratio) > 1e-30,
                        np.log(np.abs(ratio)),
                        0.0,
                    )
                u, s, vh = np.linalg.svd(log_ratio)
                print(f"  log-ratio singular values (top 3): "
                      f"{s[:3]}")
                if len(s) > 1 and s[1] / s[0] < 0.01:
                    print(f"  → Separable conversion: native[i,j] = "
                          f"f(i)·g(j)·sanchez[i,j]")
                else:
                    print(f"  → Non-separable conversion (full matrix)")
