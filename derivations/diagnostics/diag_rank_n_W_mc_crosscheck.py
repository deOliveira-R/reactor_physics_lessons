"""Diagnostic: Monte-Carlo cross-check of rank-N transmission matrix W^{mn}.

Created by numerics-investigator on 2026-04-21.
If this test catches a real bug, promote to ``tests/derivations/`` — picking
a name like ``test_peierls_rank_n_transmission_mc.py``.

**What this tests**

``compute_hollow_sph_transmission_rank_n`` defines each off-diagonal block
via an analytical quadrature of the form

    W_io^{mn} = 2 * int_0^theta_c cos(theta) sin(theta)
                  * Ptilde_n(cos theta)  # emission-mode weighting
                  * Ptilde_m(c_in(theta))  # arrival-mode projection
                  * exp(-sigma_t * ell(theta)) dtheta

with c_in(theta) = sqrt(1 - (R sin(theta) / r_0)**2) and
ell(theta) = R cos(theta) - sqrt(r_0**2 - R**2 sin**2(theta)).

This script compares the analytical integral to a direct Monte-Carlo
sample of the SAME integral. A discrepancy here would mean the analytical
integrand is mis-coded (wrong angle, wrong arrival cosine, wrong chord).
Agreement means the analytical formula IS the integral it claims to be —
any closure bug is then downstream, not in W itself.

**Why this matters (Issue #119)**

The rank-N per-face white BC closure gives ~1.4 % k_eff residual at N=2
while the N=1 scalar path gives 0.077 %. One of the prime suspects is
``compute_hollow_sph_transmission_rank_n`` at n >= 1. If the MC check
confirms the analytical W formula is internally consistent, the bug is
in the primitives or closure assembly — not in W.
"""

from __future__ import annotations

import numpy as np
import pytest


# --- Shifted Legendre on [0, 1] (explicit up to n=3) ---------------------


def p_tilde(n: int, mu: np.ndarray | float) -> np.ndarray | float:
    """Shifted Legendre polynomial on [0, 1].

    P~_0(mu) = 1
    P~_1(mu) = 2*mu - 1
    P~_2(mu) = 6*mu**2 - 6*mu + 1
    P~_3(mu) = 20*mu**3 - 30*mu**2 + 12*mu - 1
    """
    mu = np.asarray(mu, dtype=float)
    if n == 0:
        return np.ones_like(mu)
    if n == 1:
        return 2.0 * mu - 1.0
    if n == 2:
        return 6.0 * mu * mu - 6.0 * mu + 1.0
    if n == 3:
        return ((20.0 * mu - 30.0) * mu + 12.0) * mu - 1.0
    raise NotImplementedError(f"p_tilde only implemented for n <= 3, got {n}")


# --- Analytical reference from production code ---------------------------


def analytical_W_io_element(
    m: int,
    n: int,
    r_0: float,
    R: float,
    sig_t: float,
    n_quad: int = 4096,
) -> float:
    """Compute W_io^{mn} by high-resolution Gauss-Legendre quadrature on
    the exact formula in
    :func:`orpheus.derivations.peierls_geometry.compute_hollow_sph_transmission_rank_n`.

    This is NOT the production call — it is a faithful re-implementation
    using numpy. Keeps the diagnostic self-contained and avoids potential
    mpmath precision variations from confusing the MC comparison.
    """
    theta_c = np.arcsin(r_0 / R)
    # Gauss-Legendre on [0, theta_c].
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    theta = 0.5 * theta_c * (nodes + 1.0)
    wts = 0.5 * theta_c * weights
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    h_sq = R * R * sin_th * sin_th
    # Arrival cosine at inner sphere (local outward normal frame).
    arg = 1.0 - h_sq / (r_0 * r_0)
    arg = np.clip(arg, 0.0, 1.0)
    c_in = np.sqrt(arg)
    chord = R * cos_th - np.sqrt(np.maximum(r_0 * r_0 - h_sq, 0.0))
    integrand = (
        cos_th * sin_th
        * p_tilde(n, cos_th)
        * p_tilde(m, c_in)
        * np.exp(-sig_t * chord)
    )
    return 2.0 * float(np.sum(wts * integrand))


def analytical_W_oo_element(
    m: int,
    n: int,
    r_0: float,
    R: float,
    sig_t: float,
    n_quad: int = 4096,
) -> float:
    """Analytical reference for W_oo^{mn} block (outer-to-outer)."""
    theta_c = np.arcsin(r_0 / R)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    theta = 0.5 * (np.pi / 2 - theta_c) * (nodes + 1.0) + theta_c
    wts = 0.5 * (np.pi / 2 - theta_c) * weights
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    chord = 2.0 * R * cos_th
    integrand = (
        cos_th * sin_th
        * p_tilde(n, cos_th)
        * p_tilde(m, cos_th)
        * np.exp(-sig_t * chord)
    )
    return 2.0 * float(np.sum(wts * integrand))


# --- Monte-Carlo estimator of the SAME integral --------------------------


def mc_W_io_element(
    m: int,
    n: int,
    r_0: float,
    R: float,
    sig_t: float,
    n_samples: int = 2_000_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Monte-Carlo estimate of W_io^{mn} by direct integration of the
    analytical formula.

    Variable change u = sin^2(theta):
        cos(theta) sin(theta) d(theta) = 0.5 du
        u in [0, sin^2(theta_c)] = [0, (r_0/R)**2]

    Then
        W_io^{mn} = (r_0/R)**2 * E_{u ~ U(0, (r_0/R)**2)} [
            Ptilde_n(cos theta(u)) * Ptilde_m(c_in(u))
            * exp(-sigma_t * ell(u))
        ]

    Returns (mean, std-error) in the same units as the analytical W.
    """
    rng = np.random.default_rng(seed)
    s_sq_max = (r_0 / R) ** 2  # == sin^2(theta_c)
    u = rng.uniform(0.0, s_sq_max, size=n_samples)
    # theta = arcsin(sqrt(u))
    sin_th = np.sqrt(u)
    cos_th = np.sqrt(1.0 - u)
    # Chord geometry.
    h_sq = R * R * u  # R**2 sin**2(theta)
    # Inner-side arrival cosine: cos(alpha) where sin(alpha) = R sin(theta)/r_0.
    arg = 1.0 - h_sq / (r_0 * r_0)
    arg = np.clip(arg, 0.0, 1.0)
    c_in = np.sqrt(arg)
    chord = R * cos_th - np.sqrt(np.maximum(r_0 * r_0 - h_sq, 0.0))
    sample = (
        p_tilde(n, cos_th) * p_tilde(m, c_in) * np.exp(-sig_t * chord)
    )
    mean = float(np.mean(sample))
    var = float(np.var(sample, ddof=1))
    # Full integral estimate (the "2 *" from the analytical formula and
    # the Jacobian du/(2 sin theta cos theta) cancel out; details:
    #   ∫ 2 cos sin f dtheta = ∫ f du from 0 to sin^2(theta_c)
    #                        = sin^2(theta_c) * E_u[f]
    # So the MC estimator of W_io^{mn} is:
    estimate = s_sq_max * mean
    stderr = s_sq_max * np.sqrt(var / n_samples)
    return estimate, stderr


def mc_W_oo_element(
    m: int,
    n: int,
    r_0: float,
    R: float,
    sig_t: float,
    n_samples: int = 2_000_000,
    seed: int = 4242,
) -> tuple[float, float]:
    """Monte-Carlo estimate of W_oo^{mn}. Same logic as above but over
    theta in [theta_c, pi/2], i.e., u = sin^2(theta) in [s_sq_c, 1].
    """
    rng = np.random.default_rng(seed)
    s_sq_c = (r_0 / R) ** 2
    u = rng.uniform(s_sq_c, 1.0, size=n_samples)
    sin_th = np.sqrt(u)
    cos_th = np.sqrt(np.maximum(1.0 - u, 0.0))
    chord = 2.0 * R * cos_th
    sample = (
        p_tilde(n, cos_th) * p_tilde(m, cos_th) * np.exp(-sig_t * chord)
    )
    mean = float(np.mean(sample))
    var = float(np.var(sample, ddof=1))
    length = 1.0 - s_sq_c
    estimate = length * mean
    stderr = length * np.sqrt(var / n_samples)
    return estimate, stderr


# --- Cross-check against production mpmath implementation ----------------


def production_W(
    r_0: float, R: float, sig_t: float, N: int, dps: int = 25
) -> np.ndarray:
    from orpheus.derivations.peierls_geometry import (
        compute_hollow_sph_transmission_rank_n,
    )
    return compute_hollow_sph_transmission_rank_n(
        r_0, R, np.array([R]), np.array([sig_t]),
        n_bc_modes=N, dps=dps,
    )


# =========================================================================
# Tests
# =========================================================================


# Test geometry: matches the k_eff probe setup.
R_OUT = 5.0
R_IN = 0.3 * R_OUT  # r_0/R = 0.3
SIGMA_T = 1.0
N_MODES = 3  # test up to n, m <= 2
N_MC = 4_000_000
TOL_SIGMA = 5.0  # require |production - MC| < 5 * MC stderr


@pytest.fixture(scope="module")
def W_prod():
    return production_W(R_IN, R_OUT, SIGMA_T, N_MODES, dps=25)


@pytest.mark.parametrize(
    "m,n",
    [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)],
)
def test_W_io_matches_analytical_highres(m, n, W_prod):
    """The analytical re-implementation (numpy GL, 4096 points) must
    agree with the production mpmath quadrature to ~1e-10.

    This is a mundane sanity check: if this fails, the bug is in the
    mpmath routine or the numpy re-implementation — NOT in the
    cross-basis measure issue the investigation cares about.
    """
    analytical = analytical_W_io_element(m, n, R_IN, R_OUT, SIGMA_T, n_quad=4096)
    prod = W_prod[N_MODES + m, n]
    assert abs(analytical - prod) < 1e-8, (
        f"Analytical vs production W_io^[{m},{n}]: "
        f"analytical={analytical:.10e}, prod={prod:.10e}, "
        f"|diff|={abs(analytical - prod):.3e}"
    )


@pytest.mark.parametrize(
    "m,n",
    [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)],
)
def test_W_oo_matches_analytical_highres(m, n, W_prod):
    """Same check for the outer-outer block."""
    analytical = analytical_W_oo_element(m, n, R_IN, R_OUT, SIGMA_T, n_quad=4096)
    prod = W_prod[m, n]
    assert abs(analytical - prod) < 1e-8, (
        f"Analytical vs production W_oo^[{m},{n}]: "
        f"analytical={analytical:.10e}, prod={prod:.10e}, "
        f"|diff|={abs(analytical - prod):.3e}"
    )


@pytest.mark.parametrize(
    "m,n",
    [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)],
)
def test_W_io_matches_monte_carlo(m, n, W_prod):
    """Monte-Carlo cross-check of the analytical W_io^{mn} formula.

    Samples u = sin^2(theta) uniformly over [0, (r_0/R)^2] and
    estimates the same integral. Tolerance is 5 * MC standard error
    for each element.

    This test is the PRIMARY mission deliverable — it isolates whether
    the analytical W_io formula is a faithful quadrature (i.e., the
    bug is NOT in compute_hollow_sph_transmission_rank_n) or not.
    """
    mc_val, mc_err = mc_W_io_element(
        m, n, R_IN, R_OUT, SIGMA_T, n_samples=N_MC, seed=1000 + 10 * m + n,
    )
    prod = W_prod[N_MODES + m, n]
    # Tolerance: max(5 sigma, 5e-5 absolute — MC fluctuation floor).
    tol = max(TOL_SIGMA * mc_err, 5e-5)
    assert abs(mc_val - prod) < tol, (
        f"MC vs production W_io^[{m},{n}]: "
        f"MC={mc_val:.6e} +/- {mc_err:.2e}, prod={prod:.6e}, "
        f"|diff|={abs(mc_val - prod):.2e}, tol={tol:.2e}, "
        f"N_samples={N_MC}"
    )


@pytest.mark.parametrize(
    "m,n",
    [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)],
)
def test_W_oo_matches_monte_carlo(m, n, W_prod):
    """Monte-Carlo cross-check of W_oo^{mn} (outer-to-outer)."""
    mc_val, mc_err = mc_W_oo_element(
        m, n, R_IN, R_OUT, SIGMA_T, n_samples=N_MC, seed=7000 + 10 * m + n,
    )
    prod = W_prod[m, n]
    tol = max(TOL_SIGMA * mc_err, 5e-5)
    assert abs(mc_val - prod) < tol, (
        f"MC vs production W_oo^[{m},{n}]: "
        f"MC={mc_val:.6e} +/- {mc_err:.2e}, prod={prod:.6e}, "
        f"|diff|={abs(mc_val - prod):.2e}, tol={tol:.2e}, "
        f"N_samples={N_MC}"
    )


def test_W_io_sigma_t_zero_closed_form():
    """At sigma_t = 0, W_io^{mn} reduces to a closed form.

    Let s = r_0/R. Then:

        W_io^{mn}|_{sig_t=0} = 2 int_0^{theta_c} cos sin Ptilde_n(cos theta)
                                * Ptilde_m(c_in(theta)) dtheta.

    For n = m = 0: = sin^2(theta_c) = s^2 = 0.09.
    For n = 1, m = 0: this was computed by hand in the investigation
        memo: 1/3 - 4(1-s^2)^{3/2}/3 + (1-s^2). For s = 0.3: ~0.0848.
    For n = 0, m = 1: with c_in(theta) running from 1 at theta=0 to
        0 at theta=theta_c, this is DIFFERENT from W_io^{1,0}.
    """
    prod = production_W(R_IN, R_OUT, 0.0, N_MODES, dps=25)
    s = R_IN / R_OUT
    # W_io^{00} = (r_0/R)**2 — fraction of isotropic that hits the inner.
    assert abs(prod[N_MODES + 0, 0] - s**2) < 1e-10, (
        f"W_io^[0,0] at sigma_t=0: expected {s**2}, got {prod[N_MODES + 0, 0]}"
    )
    # W_io^{01} hand-computed above.
    one_minus = 1.0 - s**2
    expected_01 = 1.0 / 3.0 - 4.0 * one_minus ** 1.5 / 3.0 + one_minus
    assert abs(prod[N_MODES + 0, 1] - expected_01) < 1e-10, (
        f"W_io^[0,1] at sigma_t=0: expected {expected_01}, "
        f"got {prod[N_MODES + 0, 1]}"
    )


def test_W_sanchez_reciprocity_at_nonzero_sigma(W_prod):
    """Cross-check the transposed-index reciprocity at finite sigma_t.

    This is the same identity the production test pins, re-verified
    here to guarantee the MC-checked formula still obeys the canonical
    Sanchez-McCormick form.
    """
    area_ratio = (R_OUT / R_IN) ** 2
    for m in range(N_MODES):
        for n in range(N_MODES):
            # W_oi^{mn} (= W_prod[m, N+n]) = (R/r_0)^2 * W_io^{nm} (= W_prod[N+n, m])
            assert abs(
                W_prod[m, N_MODES + n] - area_ratio * W_prod[N_MODES + n, m]
            ) < 1e-13, f"reciprocity fail at (m,n)=({m},{n})"


def test_W_mode_asymmetry_at_sigma_t_zero():
    """At σ_t = 0, the W_io^{m,n} entries for (m,n) = (0,1) vs (1,0)
    provide a clean probe of the cross-mode asymmetry.

    Index convention: W[N + m, n] = W_io^{m,n} with
        m = arrival mode at inner
        n = emission mode at outer

    Integrand: cos θ · sin θ · Ptilde_n(cos θ) · Ptilde_m(c_in) · e^{-τ}.

    At σ_t = 0:

    W_io^{0,1} = 2 ∫_0^{θ_c} cos θ sin θ · (2 cos θ - 1) · 1 dθ
               = ∫_0^{sin² θ_c} [2 sqrt(1-v(...)) - 1] dv'...

    Let u = sin²θ, u ∈ [0, s²], cos θ = sqrt(1-u).
    W_io^{0,1} = ∫_0^{s²} (2 sqrt(1-u) - 1) du
               = [−(4/3)(1-u)^{3/2} − u]_0^{s²}
               = 4/3 − (4/3)(1-s²)^{3/2} − s²
               For s=0.3: 4/3 - 4/3·0.868 - 0.09
                        = 1.333 - 1.158 - 0.09 = 0.0848

    W_io^{1,0} = 2 ∫_0^{θ_c} cos θ sin θ · 1 · (2 c_in - 1) dθ.
    With c_in = sqrt(1 - (R/r_0)² sin²θ), and v = sin²θ/s², c_in = sqrt(1-v).
    cos θ sin θ dθ = s²/2 dv.
    W_io^{1,0} = s² · ∫_0^1 (2 sqrt(1-v) - 1) dv = s² · (4/3 - 1) = s²/3
               For s=0.3: 0.09/3 = 0.03

    The 0.0848 vs 0.03 DIFFERENCE is the Sanchez-McCormick chord
    asymmetry: emission mode 1 weights the outer μ_e heavily, while
    arrival mode 1 weights the inner c_in. The two differ because the
    angular distribution on the inner sphere is concentrated where
    c_in is small (near-glancing).
    """
    s = R_IN / R_OUT
    one_minus = 1.0 - s**2

    # Closed-form for W_io^{0,1} (emission mode 1 at outer → arrival
    # mode 0 at inner) at σ_t = 0.
    w_io_01_expected = 4.0 / 3.0 - 4.0 * one_minus**1.5 / 3.0 - s**2
    # Closed-form for W_io^{1,0} at σ_t = 0.
    w_io_10_expected = s**2 / 3.0

    prod = production_W(R_IN, R_OUT, 0.0, 3, dps=25)
    w_io_01 = prod[3 + 0, 1]
    w_io_10 = prod[3 + 1, 0]
    print(
        f"\n[diag] σ_t=0 W_io: "
        f"W[0,1]={w_io_01:.8f} (expected {w_io_01_expected:.8f}), "
        f"W[1,0]={w_io_10:.8f} (expected {w_io_10_expected:.8f}), "
        f"asymmetry |W[0,1]-W[1,0]|={abs(w_io_01 - w_io_10):.4f}"
    )
    assert abs(w_io_01 - w_io_01_expected) < 1e-10, (
        f"W_io^[0,1] at σ_t=0 mismatch: got {w_io_01}, expected {w_io_01_expected}"
    )
    assert abs(w_io_10 - w_io_10_expected) < 1e-10, (
        f"W_io^[1,0] at σ_t=0 mismatch: got {w_io_10}, expected {w_io_10_expected}"
    )


def test_W_emission_basis_is_lambert_not_marshak():
    """Diagnose which basis W takes on its EMISSION (column) index.

    Two possible conventions for the emission test function:

    (i) Lambert:   emission angular flux ψ^+(μ) = Ptilde_n(μ) (no μ weight
                    beyond the Jacobian).
    (ii) Marshak:  emission reconstructed from a unit partial-current
                    moment (effective ψ^+ = (B^μ)^{-1} Ptilde_n).

    The "arrival cos θ" factor in the W integrand only supplies the
    partial-current measure dμ dΩ = μ dΩ → dA_s at the emission surface.
    It does NOT make the emission MODE Marshak. The mode function
    Ptilde_n(cos θ) is applied WITHOUT an extra μ weight.

    Here we probe this by comparing the σ_t=0 W_io^{0,1} value to the
    two basis predictions:

    - Lambert: as derived above, 4/3 - 4/3·(1-s²)^{3/2} - s² = 0.0848
      (for s=r_0/R=0.3).
    - Marshak: a modified integrand cos²θ · sin θ · P̃_1(cos θ) ·
      P̃_0(c_in) dθ, which gives a different value.

    The PRODUCTION matches Lambert — so the emission index of W is in
    the Lambert (no-μ-weight on test function) basis.
    """
    s = R_IN / R_OUT
    one_minus = 1.0 - s**2

    # Lambert prediction for W_io^{0,1}:
    # 2 ∫_0^{θ_c} cos θ sin θ · (2 cos θ - 1) dθ
    # = 4/3 · (1 - (1-s²)^{3/2}) - s²
    lambert_01 = 4.0 / 3.0 - 4.0 * one_minus**1.5 / 3.0 - s**2
    # Marshak prediction: ∫ 2 cos²θ sin θ · (2 cos θ - 1) dθ from 0 to θ_c
    # = 2 ∫ [2 cos³θ sin θ - cos²θ sin θ] dθ
    # Let u = cos θ, du = -sin θ dθ.
    # = 2 [0.5 (1 - (1-s²)²) - (1/3)(1 - (1-s²)^{3/2})]
    term1 = 0.5 * (1.0 - one_minus**2)
    term2 = (1.0 / 3.0) * (1.0 - one_minus**1.5)
    marshak_01 = 2.0 * (term1 - term2)

    prod = production_W(R_IN, R_OUT, 0.0, 3, dps=25)
    w_io_01 = prod[3 + 0, 1]

    print(
        f"\n[diag] σ_t=0 basis probe (W_io^[0,1]): "
        f"production={w_io_01:.8f}, "
        f"Lambert={lambert_01:.8f}, "
        f"Marshak={marshak_01:.8f}, "
        f"diff(L)={abs(w_io_01 - lambert_01):.2e}, "
        f"diff(M)={abs(w_io_01 - marshak_01):.2e}"
    )
    assert abs(w_io_01 - lambert_01) < 1e-10, (
        f"W_io^[0,1] does not match Lambert-emission-basis prediction "
        f"({lambert_01}); got {w_io_01}."
    )
    # The two values should differ meaningfully (confirming we can
    # discriminate).
    assert abs(lambert_01 - marshak_01) > 1e-3, (
        f"Lambert and Marshak predictions too close to discriminate "
        f"at s={s}"
    )


def test_W_io_reciprocity_symmetry_detection(W_prod):
    """Detect whether the mode-coupling is truly asymmetric.

    Sanchez-McCormick reciprocity (A_k W_{jk}^{mn} = A_j W_{kj}^{nm})
    relates DIFFERENT integrals. For the SAME block (e.g. W_io),
    there is no reason W_io^{mn} = W_io^{nm} in general unless the
    integrand is symmetric in the two mode indices.

    The W_io integrand uses Ptilde_n(cos theta) for emission at outer
    and Ptilde_m(c_in(theta)) for arrival at inner. These angles
    (cos theta, c_in) differ, so W_io^{mn} != W_io^{nm} generically.

    This test records the asymmetry as a FINDING, not a failure.
    Expected: W_io^{01} and W_io^{10} differ by a factor related to
    the outer/inner emission-cosine distinction.
    """
    w01 = W_prod[N_MODES + 0, 1]
    w10 = W_prod[N_MODES + 1, 0]
    ratio = w01 / w10 if abs(w10) > 1e-15 else float("inf")
    # Non-physical assertion — just log the findings.
    print(
        f"\n[diag] W_io^[0,1] = {w01:.6e}, W_io^[1,0] = {w10:.6e}, "
        f"ratio = {ratio:.4f}"
    )
    # Must not be numerically identical (else there is a bug making
    # the two mode legs fall into the same algebraic expression).
    assert abs(w01 - w10) > 1e-8, (
        f"W_io^[0,1] == W_io^[1,0] to 1e-8 — the emission and arrival "
        f"integrands have accidentally collapsed to identical formulas. "
        f"That would indicate a mis-port (e.g. both using cos theta "
        f"instead of separate cos theta / c_in). Values: "
        f"{w01:.10e}, {w10:.10e}"
    )


if __name__ == "__main__":
    # Summary report when run as a script.
    W = production_W(R_IN, R_OUT, SIGMA_T, N_MODES)
    print(f"Production W at R={R_OUT}, r_0={R_IN}, sigma_t={SIGMA_T}, N={N_MODES}:")
    print(W)
    print("\n--- W_io block (production vs MC) ---")
    for m in range(N_MODES):
        for n in range(N_MODES):
            prod = W[N_MODES + m, n]
            mc_val, mc_err = mc_W_io_element(
                m, n, R_IN, R_OUT, SIGMA_T, n_samples=N_MC,
                seed=1000 + 10 * m + n,
            )
            diff = mc_val - prod
            sigma = diff / mc_err if mc_err > 0 else 0.0
            flag = "" if abs(sigma) < TOL_SIGMA else " *** MISMATCH ***"
            print(
                f"  W_io^[{m},{n}] prod={prod:.6e} MC={mc_val:.6e} "
                f"+/- {mc_err:.2e}  diff={diff:+.2e} ({sigma:+.1f}sigma){flag}"
            )
    print("\n--- W_oo block (production vs MC) ---")
    for m in range(N_MODES):
        for n in range(N_MODES):
            prod = W[m, n]
            mc_val, mc_err = mc_W_oo_element(
                m, n, R_IN, R_OUT, SIGMA_T, n_samples=N_MC,
                seed=7000 + 10 * m + n,
            )
            diff = mc_val - prod
            sigma = diff / mc_err if mc_err > 0 else 0.0
            flag = "" if abs(sigma) < TOL_SIGMA else " *** MISMATCH ***"
            print(
                f"  W_oo^[{m},{n}] prod={prod:.6e} MC={mc_val:.6e} "
                f"+/- {mc_err:.2e}  diff={diff:+.2e} ({sigma:+.1f}sigma){flag}"
            )
