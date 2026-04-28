"""Diagnostic: Phase 5+ Round 2 BACKUP — symbolic continuous-µ limit of
the matrix-Galerkin Phase 4 form.

Created by numerics-investigator on 2026-04-28.

If this test catches a real bug, promote to ``tests/derivations/`` —
file name ``test_peierls_specular_continuous_mu_limit.py``.

Mission
-------

Identify the closed-form continuous-µ kernel that the matrix-Galerkin
Phase 4 form

.. math::

   K_{\rm bc} = G \cdot R \cdot (I - T \cdot R)^{-1} \cdot P,

with building blocks (in the shifted-Legendre basis
:math:`\tilde P_n(\mu) = P_n(2\mu-1)`)

.. math::

   M_{nm}     &= \int_0^1 \mu\,\tilde P_n(\mu)\,\tilde P_m(\mu)\,d\mu, \\
   T_{mn}     &= 2\int_0^1 \mu\,\tilde P_m(\mu)\,\tilde P_n(\mu)\,
                      e^{-\sigma 2R\mu}\,d\mu, \\
   R          &= \tfrac{1}{2}\,M^{-1},

converges to as :math:`N \to \infty`.

Key empirical finding (this script)
-----------------------------------

Using a synthetic separable kernel :math:`F(\mu) = G(\mu) = e^{-a\mu}`
(so K_bc collapses to a scalar), the matrix-Galerkin K_N converges
**monotonically from below** to

.. math::

   K_\infty^{\rm matrix}
       \;=\; \tfrac{1}{2}\!\int_0^1
             G(\mu)\,F(\mu)\,\frac{\mu}{1 - e^{-\sigma 2R\mu}}\,d\mu.

Critical: the multi-bounce factor in the limit is

.. math::

   f_\infty(\mu) \;=\; \tfrac{1}{2}\,\frac{\mu}{1 - e^{-\sigma 2R\mu}},

which is **bounded at µ → 0** (limit = :math:`1/(4 a)`). The factor
of :math:`1/2` comes from the half-Marshak normalisation embedded in
:math:`R = (1/2) M^{-1}`.

This **disproves the M1 sketch hypothesis** (cross-domain memo's
:math:`f_\infty(\mu) = \mu/(1-e^{-2a\mu})`) by a factor of 2.

Empirical evidence — at three a values:

==========  ========  ==========  ==========  ==============
σR          K_∞^full   K_∞^half   K_12        rel-half
==========  ========  ==========  ==========  ==============
0.5         0.7775    0.3888      0.3858     -0.7%
1.25        0.2155    0.1078      0.1066     -1.1%
2.5         0.0642    0.0321      0.0315     -1.7%
==========  ========  ==========  ==========  ==============

K_12 is at -50% of the FULL M1 form for all σR — the matrix limit IS
the HALF M1 form. M1 is wrong by 2×.

Decision
--------

Phase 5 production wiring should use the **HALF M1 form** as the
continuous-µ K_bc target. The 1/2 factor must NOT be omitted:

.. code-block:: python

   K_bc^∞(r_i, r_j) = (1/2) ∫_0^1 G_in(r_i, µ) F_out(r_j, µ) ·
                              µ / (1 - exp(-σ 2R µ)) dµ

This integrand is bounded at µ=0 (the µ in the numerator absorbs the
1/(2aµ) singularity of T(µ)), so standard GL or adaptive quadrature
on µ ∈ [0,1] should converge spectrally. **No singularity-subtraction
is needed.** This was the round-1 blocker for Front C.

Cross-validation against the M2 (bounce-resolved) primary dispatch
should produce the same answer; this BACKUP is the orthogonal
verification.
"""
from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
import sympy as sp


# ---------------------------------------------------------------------
# Helpers — symbolic (SymPy) and numerical (mpmath) building blocks
# ---------------------------------------------------------------------


def _build_matrices_symbolic(N: int, mu_sym: sp.Symbol, a_sym: sp.Symbol):
    """Build M, T, R symbolically at rank N for homogeneous sphere.

    Returns (M, T, R, P_tilde) — sp.Matrix at rank N, integrals exact.
    """
    P_tilde = [sp.legendre(n, 2 * mu_sym - 1).expand() for n in range(N)]

    M = sp.zeros(N, N)
    for n in range(N):
        for m in range(N):
            M[n, m] = sp.integrate(
                mu_sym * P_tilde[n] * P_tilde[m], (mu_sym, 0, 1),
            )

    T = sp.zeros(N, N)
    decay = sp.exp(-2 * a_sym * mu_sym)
    for m in range(N):
        for n in range(N):
            T[m, n] = 2 * sp.integrate(
                mu_sym * P_tilde[m] * P_tilde[n] * decay, (mu_sym, 0, 1),
            )

    R = sp.Rational(1, 2) * M.inv()
    return M, T, R, P_tilde


def _P_tilde_mp(n: int, mu_pt):
    """Shifted Legendre P̃_n(µ) = P_n(2µ-1) via Bonnet's recurrence
    (mpmath-friendly, exact in mpmath rationals)."""
    x = 2 * mu_pt - 1
    if n == 0:
        return mp.mpf(1)
    if n == 1:
        return x
    Pnm1 = mp.mpf(1)
    Pn = x
    for k in range(1, n):
        Pnp1 = ((2 * k + 1) * x * Pn - k * Pnm1) / (k + 1)
        Pnm1, Pn = Pn, Pnp1
    return Pn


def _K_N_separable_mp(N: int, a_val: mp.mpf) -> float:
    """Compute K_N = G^T · R · (I - T·R)^{-1} · P for the separable
    test kernel F(µ) = G(µ) = e^{-aµ} in mpmath at high precision.

    This is the matrix-Galerkin Phase 4 form collapsed to a scalar.
    """
    M = mp.matrix(N, N)
    for n in range(N):
        M[n, n] = mp.mpf(1) / (2 * (2 * n + 1))
        if n + 1 < N:
            v = mp.mpf(n + 1) / (2 * (2 * n + 1) * (2 * n + 3))
            M[n, n + 1] = v
            M[n + 1, n] = v

    T = mp.matrix(N, N)
    for m in range(N):
        for n in range(N):
            T[m, n] = 2 * mp.quad(
                lambda mu_pt, m=m, n=n: (
                    mu_pt * _P_tilde_mp(m, mu_pt) * _P_tilde_mp(n, mu_pt)
                    * mp.exp(-2 * a_val * mu_pt)
                ),
                [0, 1],
            )

    R = mp.mpf(1) / 2 * M ** -1

    Pv = mp.matrix(N, 1)
    Gv = mp.matrix(N, 1)
    for n in range(N):
        Pv[n, 0] = mp.quad(
            lambda mu_pt, n=n: (
                mu_pt * _P_tilde_mp(n, mu_pt) * mp.exp(-a_val * mu_pt)
            ),
            [0, 1],
        )
        Gv[n, 0] = Pv[n, 0]

    I_N = mp.eye(N)
    resolvent = (I_N - T * R) ** -1
    return float((Gv.T * R * resolvent * Pv)[0, 0])


def _K_inf_full_mp(a_val: mp.mpf) -> float:
    """The M1 sketch's continuous-µ integrand (no 1/2 prefactor)."""
    return float(mp.quad(
        lambda mu_pt: (
            mp.exp(-a_val * mu_pt) ** 2 * mu_pt
            / (1 - mp.exp(-2 * a_val * mu_pt))
        ),
        [0, 1],
    ))


def _K_inf_half_mp(a_val: mp.mpf) -> float:
    """The corrected continuous-µ integrand (with 1/2 prefactor)."""
    return _K_inf_full_mp(a_val) / 2


# ---------------------------------------------------------------------
# Test 1 — multi-bounce factor identity µ·T(µ) → 1/(2a) (V1 reproof)
# ---------------------------------------------------------------------


def test_mu_T_limit_bounded_at_zero():
    """Reproof of V1: µ · 1/(1 - e^{-2aµ}) → 1/(2a) as µ → 0+.

    Confirms the µ-numerator factor (which arises from the µ-weighted
    Gram measure of M) absorbs the 1/µ singularity of T(µ) at the
    grazing limit. The corrected ``f_∞(µ) = (1/2) µ T(µ)`` therefore
    has limit 1/(4a) at µ=0 — bounded.
    """
    mu, a = sp.symbols("mu a", positive=True, real=True)
    T_mu = 1 / (1 - sp.exp(-2 * a * mu))
    f_M1 = mu * T_mu
    lim = sp.limit(f_M1, mu, 0, "+")
    expected = 1 / (2 * a)
    assert sp.simplify(lim - expected) == 0, (
        f"µ·T(µ) → {lim}, expected {expected}"
    )

    # Also check the corrected limit (1/2)µT(µ) → 1/(4a)
    f_half = sp.Rational(1, 2) * mu * T_mu
    lim_half = sp.limit(f_half, mu, 0, "+")
    expected_half = 1 / (4 * a)
    assert sp.simplify(lim_half - expected_half) == 0


# ---------------------------------------------------------------------
# Test 2 — operator-theoretic identification: (TR) M = M (RT)
# ---------------------------------------------------------------------


def test_TR_is_matrix_rep_of_M_phi():
    """Both T·R and R·T are matrix representations of multiplication
    by e^{-2aµ} on L²([0,1], µ dµ), differing only by basis convention
    (J^+ moments vs basis coefficients).

    This is captured by the identity (T·R)·M = M·(R·T): the Gram
    matrix M intertwines the two representations.

    Verified at N = 1, 2, 3 with SymPy in closed form.
    """
    mu, a = sp.symbols("mu a", positive=True, real=True)
    a_val = sp.Rational(5, 4)

    for N in (1, 2, 3):
        M, T, R, _ = _build_matrices_symbolic(N, mu, a)
        M_n = sp.simplify(M.subs(a, a_val))
        T_n = sp.simplify(T.subs(a, a_val))
        R_n = sp.simplify(R.subs(a, a_val))

        TR = T_n * R_n
        RT = R_n * T_n

        lhs = sp.simplify(TR * M_n)
        rhs = sp.simplify(M_n * RT)
        diff = sp.simplify(lhs - rhs)
        assert diff == sp.zeros(N, N), (
            f"N={N}: TR M ≠ M RT. diff = {diff}"
        )


# ---------------------------------------------------------------------
# Test 3 — KEY EMPIRICAL FINDING: K_N → HALF M1, not full M1
# ---------------------------------------------------------------------


def test_KN_matches_HALF_M1_not_full_M1():
    """**KEY FINDING.** The matrix-Galerkin K_N converges to

        K_∞^matrix = (1/2) · ∫_0^1 e^{-2aµ} · µ/(1-e^{-2aµ}) dµ

    NOT to the M1 sketch's

        K_∞^M1 = ∫_0^1 e^{-2aµ} · µ/(1-e^{-2aµ}) dµ.

    The M1 sketch is wrong by a factor of 2. The 1/2 comes from
    R = (1/2) M^{-1} (Marshak reflection).

    PASS criteria at three a values (σR = 0.5, 1.25, 2.5):
    - K_12 within 5% of K_∞^half (matrix limit)
    - K_12 at ~-50% of K_∞^full (proves M1 is off by 2×)
    - K_12 closer to K_∞^half than K_1 (monotone convergence)
    """
    mp.mp.dps = 30

    print(f"\n[Matrix-Galerkin K_N → HALF M1, NOT full M1 (refutes round-1 hyp)]")
    print(f"  Test kernel: F(µ) = G(µ) = exp(-aµ)")
    print(f"  K_∞^full = ∫ F G µ T(µ) dµ      (M1 sketch)")
    print(f"  K_∞^half = (1/2) K_∞^full       (corrected, this script)")

    for a_f in (0.5, 1.25, 2.5):
        a_val = mp.mpf(str(a_f))
        K_full = _K_inf_full_mp(a_val)
        K_half = K_full / 2
        K_1 = _K_N_separable_mp(1, a_val)
        K_12 = _K_N_separable_mp(12, a_val)
        rel_full_12 = (K_12 - K_full) / K_full
        rel_half_12 = (K_12 - K_half) / K_half
        rel_half_1 = (K_1 - K_half) / K_half
        print(
            f"  a = {a_f:.2f}: "
            f"K_∞^full={K_full:.6f}, K_∞^half={K_half:.6f}, "
            f"K_1={K_1:.6f} (rel-half={rel_half_1:+.2%}), "
            f"K_12={K_12:.6f} (rel-full={rel_full_12:+.2%}, "
            f"rel-HALF={rel_half_12:+.4%})"
        )

        assert abs(rel_half_12) < 0.05, (
            f"a={a_f}: K_12 should match HALF M1 within 5%, "
            f"got {rel_half_12:+.4%}"
        )
        assert rel_full_12 < -0.45, (
            f"a={a_f}: K_12 should be at ~-50% of FULL M1 "
            f"(proves M1 wrong by 2×), got {rel_full_12:+.4%}"
        )
        assert abs(rel_half_12) < abs(rel_half_1), (
            f"a={a_f}: K_12 should be closer to HALF M1 than K_1. "
            f"|rel_1| = {abs(rel_half_1):.4f}, "
            f"|rel_12| = {abs(rel_half_12):.4f}"
        )


# ---------------------------------------------------------------------
# Test 4 — Sanchez-naive form (no µ in numerator) is divergent
# ---------------------------------------------------------------------


def test_sanchez_naive_form_diverges_at_origin():
    """The Sanchez-A6-style integrand without µ in numerator,

        ∫_0^1 e^{-2aµ} / (1 - e^{-2aµ}) dµ,

    DIVERGES at µ=0 (integrand goes as 1/(2aµ)). Verify by cutoff
    integration: ∫_ε^1 [no-µ form] grows logarithmically as ε → 0.

    This rules out the Sanchez-naive form as the matrix-Galerkin
    continuous limit. The matrix limit MUST have a µ-numerator (which
    we identified as 1/2 · µ/(1-e^{-2aµ}) per test 3).
    """
    mp.mp.dps = 30
    a_val = mp.mpf("1.25")

    cutoffs = [mp.mpf("0.1"), mp.mpf("0.01"),
               mp.mpf("0.001"), mp.mpf("0.0001")]
    int_vals = []
    print(f"\n[Sanchez-naive (no-µ) form: ∫_ε^1 e^{{-2aµ}}/(1-e^{{-2aµ}}) dµ]")
    for eps in cutoffs:
        I_eps = float(mp.quad(
            lambda mu_pt: (
                mp.exp(-a_val * mu_pt) ** 2
                / (1 - mp.exp(-2 * a_val * mu_pt))
            ),
            [eps, 1],
        ))
        int_vals.append(I_eps)
        print(f"  ε = {float(eps):.4e}: integral = {I_eps:.6f}")

    growth = int_vals[-1] - int_vals[0]
    assert growth > 1.0, (
        f"Sanchez naive form should diverge logarithmically. "
        f"Got growth = {growth:.4f} from ε=0.1 to ε=1e-4"
    )


# ---------------------------------------------------------------------
# Test 5 — f_∞(µ) closed form bounded at µ=0
# ---------------------------------------------------------------------


def test_f_infinity_corrected_bounded_at_zero():
    """The corrected continuous-µ multi-bounce factor

        f_∞(µ) = (1/2) · µ / (1 - e^{-2aµ})

    is bounded at µ=0 with limit 1/(4a). This is what allows
    standard GL quadrature (no singularity subtraction needed).

    Compare to:
    - Sanchez naive T(µ) = 1/(1-e^{-2aµ}): unbounded ~1/(2aµ)
    - M1 sketch    µ T(µ) = µ/(1-e^{-2aµ}): bounded → 1/(2a)
    - **Matrix limit** (1/2) µ T(µ): bounded → 1/(4a) ← TRUTH
    """
    mu, a = sp.symbols("mu a", positive=True, real=True)
    f_inf = sp.Rational(1, 2) * mu / (1 - sp.exp(-2 * a * mu))
    lim = sp.limit(f_inf, mu, 0, "+")
    expected = 1 / (4 * a)
    assert sp.simplify(lim - expected) == 0, (
        f"f_∞(0) = {lim}, expected {expected}"
    )


# ---------------------------------------------------------------------
# Test 6 — Cross-check vs ORPHEUS production compute_T_specular_sphere
# ---------------------------------------------------------------------


def test_symbolic_T_matches_orpheus_compute_T_specular_sphere():
    """The symbolic T matrix matches the production
    `compute_T_specular_sphere` numerical output at homogeneous σ to
    GL quadrature precision.

    This validates that our symbolic derivation tracks the same
    object that is shipped, so the limit identified here applies to
    the production matrix-Galerkin form.
    """
    from orpheus.derivations.peierls_geometry import (
        compute_T_specular_sphere,
    )

    mu, a = sp.symbols("mu a", positive=True, real=True)
    a_val_num = 1.25
    a_val = sp.Rational(5, 4)
    R_val = 5.0
    sigma_val = a_val_num / R_val

    N = 3
    _, T_sym, _, _ = _build_matrices_symbolic(N, mu, a)
    T_sym_num = np.array(
        [[float(T_sym[i, j].subs(a, a_val).evalf(30))
          for j in range(N)] for i in range(N)]
    )

    radii = np.array([R_val])
    sig_t = np.array([sigma_val])
    T_orph = compute_T_specular_sphere(radii, sig_t, N, n_quad=64)

    rel_err = np.abs(T_sym_num - T_orph) / np.maximum(
        np.abs(T_sym_num), 1e-12
    )
    print(
        f"\n[Symbolic T vs ORPHEUS at N={N}, σR={a_val_num}]: "
        f"max rel_err = {rel_err.max():.6e}"
    )
    assert rel_err.max() < 1e-10, (
        f"ORPHEUS T should match symbolic to GL precision. "
        f"Got max rel_err = {rel_err.max():.6e}"
    )


# ---------------------------------------------------------------------
# Test 7 — closed form for K_∞^half via dilogarithm
# ---------------------------------------------------------------------


def test_K_inf_closed_form_via_polylog():
    """The corrected continuous-µ K_∞^half for separable F=G=e^{-aµ}::

        K_∞^half = (1/2) ∫_0^1 µ / (e^{2aµ} - 1) dµ
                 = (1/(8a²)) ∫_0^{2a} x / (e^x - 1) dx

    The Bose-Einstein moment ∫_0^t x/(e^x-1) dx has the closed form

        ζ(2) - Li₂(e^{-t}) + t·ln(1 - e^{-t})

    (derived from the series expansion x/(e^x-1) = Σ_n x e^{-nx} for
    n ≥ 1, integrated term-by-term).

    Verify by:
    1. Numerical agreement with the direct quadrature K_inf_half.
    2. Asymptotic limit t → ∞: integral → ζ(2) = π²/6.
    """
    mp.mp.dps = 50

    print(f"\n[K_∞^half closed form via Bose-Einstein polylog]")
    for a_f in (0.5, 1.25, 2.5, 5.0):
        a_val = mp.mpf(str(a_f))
        K_num = _K_inf_half_mp(a_val)

        # ∫_0^t x/(e^x - 1) dx = ζ(2) - Li_2(e^{-t}) + t ln(1 - e^{-t})
        t = 2 * a_val
        be_integral = (
            mp.pi ** 2 / 6
            - mp.polylog(2, mp.exp(-t))
            + t * mp.log(1 - mp.exp(-t))
        )
        K_closed = mp.mpf(1) / (8 * a_val ** 2) * be_integral
        K_closed_f = float(K_closed)
        diff = abs(K_num - K_closed_f)
        print(
            f"  a = {a_f}: direct = {K_num:.15f}, "
            f"closed = {K_closed_f:.15f}, diff = {diff:.3e}"
        )
        assert diff < 1e-12, (
            f"a={a_f}: closed form ↔ direct mismatch ({diff:.3e})"
        )


# ---------------------------------------------------------------------
# Test 8 — convergence rate K_N → K_∞^half (basis truncation order)
# ---------------------------------------------------------------------


def test_KN_convergence_rate_to_K_inf_half():
    """At a fixed σR, log-log fit error(K_N - K_∞^half) vs N should
    show roughly algebraic decay (basis-truncation error in the
    µ-weighted half-range Legendre basis is at best polynomial because
    the integrand is smooth but the basis is non-orthonormal).

    Print rates for the record. PASS: error monotonically decreasing
    in N for N ∈ {1, ..., 10} at all three a values.
    """
    mp.mp.dps = 30

    print(f"\n[K_N convergence rate to K_∞^half]")
    for a_f in (0.5, 1.25, 2.5):
        a_val = mp.mpf(str(a_f))
        K_half = _K_inf_half_mp(a_val)
        errs = []
        for N in range(1, 11):
            K_N = _K_N_separable_mp(N, a_val)
            errs.append(abs(K_N - K_half))
        ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
        print(
            f"  a = {a_f:.2f}: errs[N=1..10] = "
            f"{[f'{e:.4e}' for e in errs]}"
        )
        print(
            f"           ratios = "
            f"{[f'{r:.2f}' for r in ratios]}"
        )

        # Monotone decreasing
        for i in range(len(errs) - 1):
            assert errs[i + 1] < errs[i], (
                f"a={a_f}: error not monotone at N={i+1} -> N={i+2}: "
                f"{errs[i]:.3e} vs {errs[i+1]:.3e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
