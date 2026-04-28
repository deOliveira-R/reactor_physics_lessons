"""Diagnostic: Phase 5+ Round 3 SECONDARY — Galerkin double-integration FAILED.

Created by numerics-investigator on 2026-04-28.

This is the parent-agent-specified entry point for the SECONDARY (Galerkin in
r-space) Phase 5+ Round 3 mission. The conclusion is documented in
`.claude/agent-memory/numerics-investigator/phase5_round3_galerkin_double_integration.md`.

OUTCOME: SECONDARY FAILS for the same root cause as PRIMARY — the µ-integrand
at r = r' has a non-integrable log-divergence (M2 finding, intact). Galerkin
sub-quadrature in r-space samples the µ-integrand at near-diagonal points
(r_qP ≈ r_qQ in adjacent panels), and the µ-integral at fixed Q_µ inherits
the diagonal log(Q_µ) divergence. The 2-D panel integral therefore grows
linearly with log(Q_µ) — empirically each Q_µ-doubling adds a constant
~0.149 to the integrated value at panel `[1, 4]` Q_r=16.

Galerkin smoothing of an integrable 2-D singularity ≠ smoothing of the
underlying µ-integrand divergence. The cross-domain memo's expectation that
"the L_i · L_j weighting smooths the diagonal singularity" was based on
the 2-D log singularity being integrable in `dr dr'`, which IS true in
the EXACT continuum sense. But the FINITE-Q_µ approximation of the µ-
integrand is non-uniformly bounded near r=r', so the panel sum captures
the log(Q_µ) divergence node-by-node.

Files in the SECONDARY investigation
=====================================

1. `diag_phase5_round3_galerkin_double_integration.py` (this file) —
   summary entry point.
2. `diag_phase5_round3_galerkin_diag_audit.py` — 4 audits revealing the
   visibility-step bug and confirming M2's diagonal log-divergence in the
   v1 implementation.
3. `diag_phase5_round3_visibility_cone_quad.py` — vis-cone + u² subst
   per-pair quadrature. **Off-diagonal Q-converges to MACHINE PRECISION
   at Q=16** for 5 pairs with |r-r'| ∈ {0.02, ..., 4.0}. Promotion-worthy
   primitive (held back: only useful as a component of a future
   continuous-µ closure, not standalone).
4. `diag_phase5_round3_convention_check.py` — scans 6 conventions for
   the multi-bounce factor (HALF/FULL/DOUBLE M1 × with-or-without 1/µ).
   None matches Hébert. Convention is NOT the fix; the per-pair K is
   structurally wrong on the diagonal regardless.
5. `diag_phase5_round3_galerkin_v2.py` — corrected Galerkin v2 with
   vis-cone+subst per-pair K. Two tests:
   - test_v2_a_smoke_keff: smoke test fails -34% to -50% across (n_quad_r,
     n_quad_µ).
   - test_v2_b_q_mu_divergence: empirical proof that ∫∫_panel K dr dr'
     grows ~0.149 per Q_µ-doubling (3-figure consistent → log divergent).

Recommendation
==============

Same as PRIMARY's R4-C: ABANDON Phase 5 production wiring. Ship
`closure="specular_multibounce"` at N ≤ 3 (already shipped) as the
production form forever. Phase 5 is research artifact only — the
continuous-µ form is structurally singular at every (r_i, r_i) and no
known reformulation in our toolkit can regularize it without introducing
a gauge ambiguity.

This file ships as a structural placeholder asserting the failure mode
is reproducible (the test below replicates the v2-B Q_µ divergence).

If converted to a permanent regression test, this proves the FAILURE
mode for any future Phase 5 retry. Move to:
`tests/derivations/test_peierls_specular_continuous_mu.py::test_galerkin_diag_log_qmu_divergent`
"""
from __future__ import annotations

import numpy as np
import pytest


# ─── Re-export primitives consumed by sibling diagnostics ────────────
# (visibility-cone quad imports F_out / G_in evaluators from this module)


SIG_T = 0.5
SIG_S = 0.38
NU_SIG_F = 0.025
K_INF = NU_SIG_F / (SIG_T - SIG_S)
R_THIN = 5.0


def _F_out_at(r_p: float, R: float, sigma: float, mu: float) -> float:
    r"""`F_out(r', µ)` for homogeneous sphere — scalar evaluation.

    Mirrors Front C `F_out_mu_sphere`; includes sphere prefactor 0.5
    and the surface-area Jacobian implicit in the chord length.
    """
    h = R * float(np.sqrt(max(0.0, 1.0 - mu * mu)))
    if r_p * r_p < h * h - 1e-15:
        return 0.0
    sin2_om = (R * R / max(r_p * r_p, 1e-30)) * (1.0 - mu * mu)
    cos2_om = max(1.0 - sin2_om, 0.0)
    if cos2_om < 1e-30:
        return 0.0
    cos_om = float(np.sqrt(cos2_om))
    sqrt_rh = float(np.sqrt(max(r_p * r_p - h * h, 0.0)))
    rho_plus = R * mu + sqrt_rh
    rho_minus = R * mu - sqrt_rh
    tau_plus = sigma * rho_plus
    tau_minus = sigma * rho_minus
    K_plus = float(np.exp(-tau_plus))
    K_minus = float(np.exp(-tau_minus))
    contrib_plus = (rho_plus * rho_plus) * K_plus
    contrib_minus = (rho_minus * rho_minus) * K_minus
    pref = 0.5
    return pref * mu / (r_p * r_p * cos_om) * (contrib_plus + contrib_minus)


def _G_in_at(r: float, R: float, sigma: float, mu: float) -> float:
    r"""`G_in(r, µ)` for homogeneous sphere — scalar evaluation."""
    h = R * float(np.sqrt(max(0.0, 1.0 - mu * mu)))
    if r * r < h * h - 1e-15:
        return 0.0
    sin2_om = (R * R / max(r * r, 1e-30)) * (1.0 - mu * mu)
    cos2_om = max(1.0 - sin2_om, 0.0)
    if cos2_om < 1e-30:
        return 0.0
    cos_om = float(np.sqrt(cos2_om))
    sqrt_rh = float(np.sqrt(max(r * r - h * h, 0.0)))
    rho_plus = R * mu + sqrt_rh
    rho_minus = R * mu - sqrt_rh
    tau_plus = sigma * rho_plus
    tau_minus = sigma * rho_minus
    decay_sum = float(np.exp(-tau_plus)) + float(np.exp(-tau_minus))
    return 2.0 * R * R * mu / (r * r * cos_om) * decay_sum


def test_round3_secondary_galerkin_fails_with_q_mu_divergence(capsys):
    """Reproduces the v2-B finding that the 2-D Galerkin integral grows
    with log(Q_µ).

    This test PASSES when the divergence is demonstrated (since the
    purpose is to document the failure mode for posterity). If a future
    Phase 5 reformulation eliminates the diagonal log-divergence, this
    test will FAIL — at which point the test should be re-evaluated
    rather than blindly fixed (the failure is the historical artifact).
    """
    with capsys.disabled():
        # Use the v2 implementation's K_pair and panel sum
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "_v2",
            os.path.join(
                os.path.dirname(__file__),
                "diag_phase5_round3_visibility_cone_quad.py",
            ),
        )
        mod_v2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod_v2)
        Kpair = mod_v2._K_cont_pair_visibility_substitution

        sigma, R = 0.5, 5.0
        a_p, b_p = 1.0, 4.0
        h = 0.5 * (b_p - a_p)
        m = 0.5 * (a_p + b_p)
        Q_r = 16

        print(f"\n=== SECONDARY: Galerkin Q_µ-divergence demo ===")
        print(f"Panel [{a_p}, {b_p}], Q_r = {Q_r}")
        print(f"Q_µ doublings should add a CONSTANT to the 2-D integral —")
        print(f"this is the log(Q_µ) divergence inherited from the diagonal")
        print(f"of the µ-integrand at r = r'.")
        print(f"  Q_µ  | ∫∫ K_pair  | Δ vs prev")

        prev = None
        deltas = []
        for Q_mu in (32, 64, 128, 256):
            nodes, wts = np.polynomial.legendre.leggauss(Q_r)
            r_sub = h * nodes + m
            w_sub = h * wts
            I = 0.0
            for k1 in range(Q_r):
                for k2 in range(Q_r):
                    I += w_sub[k1] * w_sub[k2] * Kpair(
                        r_sub[k1], r_sub[k2], sigma, R, Q_mu,
                    )
            if prev is not None:
                d = I - prev
                deltas.append(d)
                print(f"  {Q_mu:4d} | {I:+.6e} | {d:+.6e}")
            else:
                print(f"  {Q_mu:4d} | {I:+.6e} | n/a")
            prev = I

        # Assertion: deltas should be approximately CONSTANT (proves
        # log(Q_µ) divergence). If they were decaying to zero, the
        # integral would be Q_µ-convergent.
        delta_arr = np.array(deltas)
        ratio_max = np.max(delta_arr) / np.min(delta_arr)
        print(f"\n  Δ values: {deltas}")
        print(f"  max/min ratio: {ratio_max:.4f}")
        print(f"  → Δ ≈ constant means log(Q_µ) divergence (FAILURE mode).")
        # The Δ values are within 1-2 % of each other, consistent with
        # log divergence (small higher-order corrections).
        assert ratio_max < 1.1, (
            f"Δ values not constant ({ratio_max:.4f} > 1.1) — Galerkin "
            f"may have CONVERGED contrary to historical record. Re-check."
        )
