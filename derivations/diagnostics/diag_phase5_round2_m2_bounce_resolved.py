"""Phase 5+ Round 2 — M2 bounce-resolved expansion of the specular-BC
multi-bounce factor.

Created by numerics-investigator on 2026-04-28.

Mission
=======

Round 1 (Fronts A/B/C) converged: the M1 form

.. math::

    K_{\rm bc}(r_i, r_j) \;=\; 2\!\int_0^1\! G_{\rm in}(r_i,\mu)\,
        F_{\rm out}(r_j,\mu)\, T(\mu)\,\mathrm d\mu

with :math:`T(\mu) = 1/(1 - e^{-\sigma\,2R\mu})` has a non-integrable
:math:`1/\mu` (interior diagonal) or :math:`1/\mu^2` (surface diagonal)
singularity at :math:`\mu \to 0`. Front C confirmed via Q-quadrature
divergence (k_eff oscillates -45 % to +33 % across Q ∈ {16..256}).

The M2 reformulation expands the multi-bounce factor as a geometric series

.. math::

    T(\mu) \;=\; \sum_{k=0}^{\infty} e^{-k\,\sigma\,2R\,\mu}

so

.. math::

    K_{\rm bc}(r_i, r_j) \;=\; \sum_{k=0}^{\infty} K_{\rm bc}^{(k)}(r_i, r_j),
    \quad K_{\rm bc}^{(k)} \;=\; 2\!\int_0^1\! G_{\rm in}\,F_{\rm out}\,
        e^{-k\,\sigma\,2R\,\mu}\,\mathrm d\mu.

Each per-bounce integrand is **bounded at µ → 0** (G_in · F_out
involves a finite chord-to-surface distance and a finite cos(ω) at
each visible µ), so plain Gauss-Legendre is spectral.

Series truncation: each term decays as :math:`e^{-k\,\sigma\,2R\,\mu_{\rm
eff}}` (some effective µ around 0.5), so K_max ≈ ⌈log(1/tol)/(σR)⌉
suffices.

This diagnostic answers (per the dispatch protocol):

1. ``w(µ) = µ`` vs ``w(µ) = 1`` — which is the correct M1 form
2. Per-bounce magnitude decay (geometric vs not)
3. Q-convergence (spectral vs oscillating)
4. Smoke test: k_eff at thin homogeneous sphere vs k_inf and white_hebert
5. K_max truncation study
6. Cross-check: K_bc^(0) ≡ bare specular Phase 4? K_bc^(0) + K_bc^(1) ≈ rank-1 Hebert?
7. Multi-region 1G/2R sphere (if reachable)

Decision criterion: PASS if k_eff matches k_inf within 0.5 % at thin
τ_R = 2.5 with Q ≤ 128, K_max ≤ 20, no oscillation. PASS → ship as
``closure="specular_continuous_mu"`` (replacing the current
``NotImplementedError``).
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
)

# Reuse Front C's primitives (verified against Phase 4 P_esc_mode /
# G_bc_mode at Q=128 within 1-30 % at endpoints, <2 % interior).
from derivations.diagnostics.diag_phase5_native_c01_orpheus_form import (
    F_out_mu_sphere,
    G_in_mu_sphere,
    keff_from_K,
)


# ─── Fixtures matching Phase 4 thin-sphere & Round 1 dispatch ─────────


SIG_T = 0.5
SIG_S = 0.38
NU_SIG_F = 0.025
K_INF = NU_SIG_F / (SIG_T - SIG_S)  # = 0.20833
R_THIN = 5.0  # τ_R = 2.5


# ─── M2 bounce-resolved K_bc ─────────────────────────────────────────


def compute_K_bc_M2_bounce_sphere(
    geometry, r_nodes, r_wts, radii, sig_t, *,
    n_quad: int = 64,
    K_max: int = 20,
    weight_form: str = "mu_in_T",  # "mu_in_T" or "no_mu"
    return_per_bounce: bool = False,
):
    r"""M2 bounce-resolved K_bc for sphere with specular BC.

    Sums

    .. math::

        K_{\rm bc} \;=\; \sum_{k=0}^{K_{\max}} K_{\rm bc}^{(k)},
        \quad K_{\rm bc}^{(k)} \;=\; 2\!\int_0^1 G_{\rm in}(r_i,\mu)\,
            F_{\rm out}(r_j,\mu)\,w(\mu)\,e^{-k\,\sigma\,2R\,\mu}\,\mathrm d\mu

    with weight ``w(µ) = µ`` (form ``mu_in_T``, M1-with-µ-in-numerator,
    geometric-series of ``µ·T(µ)``) or ``w(µ) = 1`` (form ``no_mu``,
    pure Sanchez geometric-series of ``T(µ)``).

    Per-bounce integrand is bounded at µ → 0 in BOTH forms because
    ``G_in · F_out`` is finite at the visibility cutoff.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)

    # GL on µ ∈ [0, 1]
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    mu_pts = 0.5 * (nodes + 1.0)
    mu_wts = 0.5 * wts

    # Multi-region antipodal chord τ(µ) — same per-bounce decay
    # τ_chord(µ) is the FULL antipodal chord (one round trip's optical
    # depth).
    tau_chord = _chord_tau_mu_sphere(radii, sig_t, mu_pts)
    # Per-bounce decay at quadrature node µ_q: e^{-k·τ_chord(µ_q)}
    # K_max=0 baseline is the bare specular kernel (no multi-bounce).

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
    divisor = geometry.rank1_surface_divisor(float(radii[-1]))

    F_out_w = (rv * r_wts)[None, :] * F_out         # (Q, N_r)
    G_in_w = (sig_t_n / divisor)[:, None] * G_in    # (N_r, Q)

    # Choose µ-weight
    if weight_form == "mu_in_T":
        # K^(k) = 2 ∫ G F · µ · e^{-k τ_chord(µ)} dµ
        w_mu = mu_pts.copy()
    elif weight_form == "no_mu":
        # K^(k) = 2 ∫ G F · 1 · e^{-k τ_chord(µ)} dµ — Sanchez direct
        w_mu = np.ones_like(mu_pts)
    else:
        raise ValueError(
            f"weight_form must be 'mu_in_T' or 'no_mu', got {weight_form}"
        )

    K_total = np.zeros((len(r_nodes), len(r_nodes)))
    per_bounce = []
    for k in range(K_max + 1):
        # Per-bounce factor: e^{-k τ_chord(µ)}
        decay_k = np.exp(-k * tau_chord)
        K_k = 2.0 * np.einsum(
            'iq,q,qj->ij', G_in_w, mu_wts * w_mu * decay_k, F_out_w,
        )
        K_total += K_k
        if return_per_bounce:
            per_bounce.append(K_k)

    if return_per_bounce:
        return K_total, per_bounce
    return K_total


# ═════════════════════════════════════════════════════════════════════
# Tests (probes)
# ═════════════════════════════════════════════════════════════════════


def test_p1_weight_form_pinning(capsys):
    """P1 — Pin the µ-weight by rank-1 cross-check.

    The CORRECT weight is determined by geometric-series
    decomposition of the ALREADY-PROVEN Phase 4 specular_multibounce
    rank-1 form (which agrees with white_hebert algebraically per
    `phase4_cyl_slab` memo). Build K_bc with each weight form and check
    rank-1 collapse → k_eff at thin τ_R = 2.5.

    Actually a more direct check: the bare specular Phase 4 closure
    rank-1 form (closure='specular') — which is documented as 'rank-1
    bit-equals Mark' — equals K_bc^(0) (k=0 term only) in the M2 form
    that has the CORRECT weight.

    We compute the bare K_bc^(0) under both weight conventions and
    compare to closure='specular' rank-1 K_bc.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )

        # K_bc^(0) under each weight
        K0_mu = compute_K_bc_M2_bounce_sphere(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
            n_quad=128, K_max=0, weight_form="mu_in_T",
        )
        K0_nomu = compute_K_bc_M2_bounce_sphere(
            SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
            n_quad=128, K_max=0, weight_form="no_mu",
        )

        sig_s_arr = np.full(len(r_nodes), SIG_S)
        nu_sf_arr = np.full(len(r_nodes), NU_SIG_F)
        sig_t_arr = np.full(len(r_nodes), SIG_T)

        # Vacuum K_vol
        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        # Bare specular Phase 4 rank-1 reference
        K_bare = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "specular",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        # Hebert reference (k=0+1+2+... single closed form)
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )

        k_bare = keff_from_K(K_bare, sig_t_arr, nu_sf_arr, sig_s_arr)
        k_heb = keff_from_K(K_heb, sig_t_arr, nu_sf_arr, sig_s_arr)
        k_M2_mu_k0 = keff_from_K(
            K_vol + K0_mu, sig_t_arr, nu_sf_arr, sig_s_arr,
        )
        k_M2_nomu_k0 = keff_from_K(
            K_vol + K0_nomu, sig_t_arr, nu_sf_arr, sig_s_arr,
        )

        print("\n=== P1 — Weight pinning at k=0 (bare specular) ===")
        print(f"R={R_THIN}, σ_t={SIG_T}, τ_R={SIG_T*R_THIN}, k_inf={K_INF:.6f}")
        print(f"  closure='specular' (Phase 4 rank-1 bare):    k_eff = {k_bare:.6f}  ({(k_bare-K_INF)/K_INF*100:+.4f}%)")
        print(f"  closure='white_hebert' (rank-1 multi-bounce): k_eff = {k_heb:.6f}  ({(k_heb-K_INF)/K_INF*100:+.4f}%)")
        print(f"  M2 K^(0) weight=µ:                            k_eff = {k_M2_mu_k0:.6f}  ({(k_M2_mu_k0-K_INF)/K_INF*100:+.4f}%)")
        print(f"  M2 K^(0) weight=1:                            k_eff = {k_M2_nomu_k0:.6f}  ({(k_M2_nomu_k0-K_INF)/K_INF*100:+.4f}%)")
        print(f"\n  CRITERION: weight matching closure='specular' rank-1 wins.")
        print(f"  → |k_M2_mu_k0  - k_bare| = {abs(k_M2_mu_k0 - k_bare):.6e}")
        print(f"  → |k_M2_nomu_k0 - k_bare| = {abs(k_M2_nomu_k0 - k_bare):.6e}")


def test_p2_per_bounce_magnitudes(capsys):
    """P2 — Per-bounce magnitude decay.

    Compute K_bc^(k) for k=0..15 and tabulate ||K^(k)||_F. Should decay
    geometrically with ratio ≈ e^{-σ·2R·µ_eff}.
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, _ = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )

        for weight in ("mu_in_T", "no_mu"):
            K_tot, K_list = compute_K_bc_M2_bounce_sphere(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=128, K_max=15, weight_form=weight,
                return_per_bounce=True,
            )
            print(f"\n=== P2 — Per-bounce decay (weight={weight}) ===")
            print(f"  k  | ||K^(k)||_F        | ratio to ||K^(k-1)||_F")
            prev = None
            for k, Kk in enumerate(K_list):
                fr = float(np.linalg.norm(Kk))
                ratio = (fr / prev) if prev is not None and prev > 0 else float("nan")
                print(f"  {k:2d} | {fr:18.6e} | {ratio:.6f}")
                prev = fr


def test_p3_K_max_truncation(capsys):
    """P3 — K_max truncation study (smoke test).

    For the SELECTED weight_form (decided by P1), sweep K_max ∈
    {1, 2, 3, 5, 10, 20} at Q=128 and tabulate k_eff. The smallest
    K_max that gives k_eff converged to within 0.05 % of K_max=∞
    (= K_max = 50 here) is the production setting.
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
        k_M2_inf = {}
        for weight in ("mu_in_T", "no_mu"):
            K_M2_50 = compute_K_bc_M2_bounce_sphere(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=128, K_max=50, weight_form=weight,
            )
            k_M2_inf[weight] = keff_from_K(
                K_vol + K_M2_50, sig_t_arr, nu_sf_arr, sig_s_arr,
            )

            print(f"\n=== P3 — K_max truncation, weight={weight} ===")
            print(f"  K_max | k_eff_M2     | rel_kinf  | rel_heb")
            for K_max in (0, 1, 2, 3, 5, 10, 20, 50):
                K_bc_n = compute_K_bc_M2_bounce_sphere(
                    SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                    n_quad=128, K_max=K_max, weight_form=weight,
                )
                k_n = keff_from_K(
                    K_vol + K_bc_n, sig_t_arr, nu_sf_arr, sig_s_arr,
                )
                rel_inf = (k_n - K_INF) / K_INF * 100
                rel_h = (k_n - k_heb) / k_heb * 100
                print(f"  {K_max:3d}   | {k_n:.6f}    | {rel_inf:+.4f}%  | {rel_h:+.4f}%")
            print(f"\n  Reference: white_hebert k_eff = {k_heb:.6f} ({(k_heb-K_INF)/K_INF*100:+.4f}%)")
            print(f"             k_inf = {K_INF:.6f}")


def test_p4_q_convergence(capsys):
    """P4 — Q-quadrature convergence at fixed K_max.

    For the SELECTED weight_form and K_max=20, sweep Q ∈ {16, 32, 64,
    128, 256}. Should show MONOTONE convergence (no oscillation —
    the M2 reformulation removed the singularity).
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

        for weight in ("mu_in_T", "no_mu"):
            print(f"\n=== P4 — Q convergence at K_max=20, weight={weight} ===")
            print(f"  Q   | k_eff       | rel_inf   | rel_heb")
            prev = None
            for Q in (16, 32, 64, 128, 256):
                K_bc_n = compute_K_bc_M2_bounce_sphere(
                    SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                    n_quad=Q, K_max=20, weight_form=weight,
                )
                k_n = keff_from_K(
                    K_vol + K_bc_n, sig_t_arr, nu_sf_arr, sig_s_arr,
                )
                rel_inf = (k_n - K_INF) / K_INF * 100
                rel_h = (k_n - k_heb) / k_heb * 100
                delta = (k_n - prev) if prev is not None else float("nan")
                print(f"  {Q:3d} | {k_n:.6f}   | {rel_inf:+.4f}%  | {rel_h:+.4f}%   (Δk={delta:+.2e})")
                prev = k_n


def test_p5_smoke_homogeneous_thin(capsys):
    """P5 — Homogeneous thin sphere smoke test.

    Decision criterion: PASS if k_eff matches k_inf within 0.5 % at
    thin τ_R = 2.5 with Q = 128, K_max = 20, no oscillation.
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

        print("\n=== P5 — SMOKE TEST (thin homogeneous sphere) ===")
        print(f"  R={R_THIN}, σ_t={SIG_T}, τ_R={SIG_T*R_THIN}, k_inf={K_INF:.6f}")
        print(f"  Reference white_hebert: k_eff={k_heb:.6f}, "
              f"rel_inf={(k_heb-K_INF)/K_INF*100:+.4f}%")

        for weight in ("mu_in_T", "no_mu"):
            K_bc_n = compute_K_bc_M2_bounce_sphere(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=128, K_max=20, weight_form=weight,
            )
            k_n = keff_from_K(
                K_vol + K_bc_n, sig_t_arr, nu_sf_arr, sig_s_arr,
            )
            rel_inf = (k_n - K_INF) / K_INF * 100
            rel_h = (k_n - k_heb) / k_heb * 100
            verdict = "PASS" if abs(rel_inf) < 0.5 else "FAIL"
            print(f"  M2 weight={weight}: k_eff={k_n:.6f}  rel_inf={rel_inf:+.4f}% "
                  f"rel_heb={rel_h:+.4f}%  → {verdict}")


def test_p6_phase4_crosscheck(capsys):
    """P6 — Cross-check with Phase 4 matrix form at low N.

    K_bc^(0) (k=0 term) should match Phase 4 closure='specular' rank-1
    (which is bit-equal to Mark for sphere). Total M2 sum at high
    K_max should match white_hebert (rank-1 (1-P_ss)^{-1} closure).
    """
    with capsys.disabled():
        radii = np.array([R_THIN])
        sig_t_g = np.array([SIG_T])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )

        K_vol = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "vacuum",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        K_bare = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "specular",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )
        K_heb = _build_full_K_per_group(
            SPHERE_1D, r_nodes, r_wts, panels, radii, sig_t_g,
            "white_hebert",
            n_angular=24, n_rho=24, n_surf_quad=24,
            n_bc_modes=1, dps=20,
        )

        K_bc_bare_p4 = K_bare - K_vol  # bare K_bc only
        K_bc_heb_p4 = K_heb - K_vol    # Hebert K_bc only

        for weight in ("mu_in_T", "no_mu"):
            K0 = compute_K_bc_M2_bounce_sphere(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=128, K_max=0, weight_form=weight,
            )
            Kinf = compute_K_bc_M2_bounce_sphere(
                SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                n_quad=128, K_max=50, weight_form=weight,
            )

            n_bc_bare = float(np.linalg.norm(K_bc_bare_p4))
            n_bc_heb = float(np.linalg.norm(K_bc_heb_p4))
            print(f"\n=== P6 — Phase 4 cross-check (weight={weight}) ===")
            print(f"  ||K_bc_bare_p4||_F  = {n_bc_bare:.6e}")
            print(f"  ||K^(0)_M2||_F      = {float(np.linalg.norm(K0)):.6e}")
            print(f"  ||K_bc_heb_p4||_F   = {n_bc_heb:.6e}")
            print(f"  ||K^∞_M2||_F        = {float(np.linalg.norm(Kinf)):.6e}")

            # Element ratio at center to detect simple multiplicative
            # gap.
            i, j = len(r_nodes)//2, len(r_nodes)//2
            print(f"  K_bc_bare_p4[{i},{j}] = {K_bc_bare_p4[i,j]:.6e}")
            print(f"  K^(0)_M2     [{i},{j}] = {K0[i,j]:.6e}")
            if K_bc_bare_p4[i,j] != 0:
                print(f"  ratio K_bc_bare / K^(0)_M2 at [{i},{j}] = {K_bc_bare_p4[i,j]/K0[i,j] if K0[i,j] != 0 else 'inf':.6e}")


def test_p7_multiregion_sphere_1g_2r(capsys):
    """P7 — Multi-region 1G/2R sphere smoke test (if reachable).

    R=[3, 5], σ_t=[1.0, 0.4]. Compare to white_hebert and check
    convergence in K_max.
    """
    with capsys.disabled():
        radii = np.array([3.0, 5.0])
        sig_t_g = np.array([1.0, 0.4])
        r_nodes, r_wts, panels = composite_gl_r(
            radii, 2, 4, dps=15, inner_radius=0.0,
        )
        # Per-region material
        sig_s_arr = np.array([
            0.7 if r_nodes[i] < 3.0 else 0.3
            for i in range(len(r_nodes))
        ])
        nu_sf_arr = np.array([
            0.05 if r_nodes[i] < 3.0 else 0.05
            for i in range(len(r_nodes))
        ])
        sig_t_arr = np.array([
            sig_t_g[SPHERE_1D.which_annulus(float(r_nodes[i]), radii)]
            for i in range(len(r_nodes))
        ])

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
        print("\n=== P7 — Multi-region 1G/2R sphere ===")
        print(f"  R={radii.tolist()}, σ_t={sig_t_g.tolist()}")
        print(f"  white_hebert: k_eff = {k_heb:.6f}")
        print(f"  K_max | weight=µ k_eff   | weight=1 k_eff")
        for K_max in (0, 1, 5, 20, 50):
            row = [f"  {K_max:3d}   "]
            for weight in ("mu_in_T", "no_mu"):
                K_bc_n = compute_K_bc_M2_bounce_sphere(
                    SPHERE_1D, r_nodes, r_wts, radii, sig_t_g,
                    n_quad=128, K_max=K_max, weight_form=weight,
                )
                k_n = keff_from_K(
                    K_vol + K_bc_n, sig_t_arr, nu_sf_arr, sig_s_arr,
                )
                row.append(f"{k_n:.6f}")
            print(" | ".join(row))
