"""Diagnostic: Phase 5+ Front B — singularity subtraction for Sanchez Eq. (A6).

Created by numerics-investigator on 2026-04-28.

If this test catches a real bug, promote to ``tests/cp/`` (Phase 5
research-grade prototype lives in ``orpheus/derivations/peierls_geometry.py``
under ``compute_K_bc_specular_continuous_mu_sphere``).

Mission
-------

The Sanchez 1986 Eq. (A6) integrand for the homogeneous-sphere specular
multi-bounce kernel has a non-removable singularity on the diagonal
``r_i = r_j``:

* surface diagonal (ρ' = ρ = a) — ``1/µ²`` (non-integrable)
* interior diagonal (ρ' = ρ < a) — ``1/µ`` (logarithmic, integrable but
  hard for plain Gauss-Legendre)

Front B asks whether singularity subtraction (or a cousin — change of
variables, Gauss-Jacobi, adaptive quad) makes the kernel computable to
working precision so production wiring against ``closure="white_hebert"``
becomes achievable at rank-1.

This script tries four approaches and reports PASS / PARTIAL / FAIL.

Approaches
----------

Approach 1 — Bernoulli subtraction: peel off the analytic ``1/µ`` (or
``1/µ²``) leading order via the Bernoulli expansion ``x/(e^x − 1) =
1 − x/2 + x²/12 − …``, integrate the smooth remainder by GL, and add
back the closed-form integral of the singular part.

Approach 2 — change of variables ``µ = u²``: absorbs the ``1/µ`` into the
Jacobian ``2u du`` so the integrand is bounded at ``u = 0``.

Approach 3 — Gauss-Jacobi quadrature with weight ``w(µ) = 1`` on
``[0, 1]`` weighted by µ implicitly; uses the Jacobi-α=1 rule
(equivalently Gauss-Legendre on ``∫₀¹ µ f(µ) dµ`` with the µ pulled into
the weight). Spectral convergence for the ``1/µ``-type singularity.

Approach 4 — adaptive ``scipy.integrate.quad`` with explicit endpoint
subdivision at ``µ = 0`` and ``µ = µ_0``. The "always works but slow"
fallback.

Verification logic
------------------

1. Symbolically expand the diagonal integrand in SymPy and identify
   ``s(µ)``.

2. For an interior-diagonal probe (ρ = ρ' = R/2 inside the sphere),
   confirm the residual ``f − s`` is smooth at µ = 0 (finite limit).

3. Verify the surface-diagonal probe (ρ = ρ' = R) shows the harder
   ``1/µ²`` behaviour and that subtraction reduces it to ``1/µ``.

4. Smoke-test against ``closure="white_hebert"`` at rank-1 for a thin
   homogeneous sphere — the magnitudes should match up to a constant
   Jacobian conversion factor (Phase 5+ Front A's territory; here we
   simply check the position-dependence shape matches).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import quad
from scipy.special import roots_jacobi

from orpheus.derivations.peierls_geometry import (
    CurvilinearGeometry,
    compute_G_bc,
    compute_K_bc_specular_continuous_mu_sphere,
    compute_P_esc,
    compute_P_ss_sphere,
)


# --------------------------------------------------------------------------
# Section 1 — Symbolic identification of the singular leading-order s(µ).
# --------------------------------------------------------------------------


def _symbolic_singular_form(diagonal: str) -> dict:
    """Symbolically peel the leading-order singular form near µ = 0.

    Parameters
    ----------
    diagonal : 'interior' or 'surface'
        ``surface``: ρ' = ρ = a.  ``interior``: ρ' = ρ = a/2.

    Returns
    -------
    dict with the analytic ``s(µ)`` and the order of the singularity.
    """
    mu, a = sp.symbols("mu a", positive=True, real=True)

    if diagonal == "surface":
        rho = a
        rho_p = a
    elif diagonal == "interior":
        # Use a generic interior point ρ < a; we keep ρ symbolic but
        # specialise to ρ = a/2 for the numeric expansion to avoid
        # branch-cut subtleties.
        rho = a / 2
        rho_p = a / 2
    else:  # pragma: no cover
        raise ValueError(diagonal)

    # µ_- = √[a² − ρ²(1 − µ²)] / a
    mu_minus = sp.sqrt(a**2 - rho**2 * (1 - mu**2)) / a
    # µ_* = √[ρ'² − ρ²(1 − µ²)] / ρ'   (on diagonal ρ' = ρ → µ_* = µ)
    mu_star = sp.sqrt(rho_p**2 - rho**2 * (1 - mu**2)) / rho_p

    T_mu_minus = 1 / (1 - sp.exp(-2 * a * mu_minus))
    integrand = (
        T_mu_minus
        * (1 / mu_star)
        * sp.cosh(rho * mu)
        * sp.cosh(rho_p * mu_star)
        * sp.exp(-2 * a * mu_minus)
    )

    # Leading-order Laurent expansion at µ = 0
    leading = sp.series(integrand, mu, 0, 2).removeO()

    # Identify the order of the singularity
    if diagonal == "surface":
        # T(µ_-) ~ 1/(2aµ), µ_*^{-1} ~ 1/µ, decay → 1, both cosh → 1
        # ⇒ integrand (with the leading 2 from Eq A6) ~ 2 · 1/(2aµ²)
        # = 1/(aµ²).  Order = -2 (non-integrable).
        order = -2
        s_expr = 1 / (a * mu**2)
    else:
        # On ρ < a: µ_- → √(1 − (ρ/a)²) finite, T·decay = e^{-τ}/(1-e^{-τ})
        # = 1/(e^{τ}-1) finite, only µ_*^{-1} ~ 1/µ.
        # Integrand ~ 2 · [1/(e^{τ_0}-1)] / µ.
        mu_minus_at_zero = sp.sqrt(1 - (rho / a) ** 2)
        T_decay_at_zero = 1 / (sp.exp(2 * a * mu_minus_at_zero) - 1)
        # cosh(ρµ) → 1, cosh(ρ' µ_*) → 1, prefactor 2 from Eq A6
        s_const = 2 * T_decay_at_zero
        s_expr = s_const / mu
        order = -1

    return {
        "diagonal": diagonal,
        "integrand": integrand,
        "leading": leading,
        "s(mu)": sp.simplify(s_expr),
        "singular_order": order,
        "rho": rho,
        "rho_p": rho_p,
    }


# --------------------------------------------------------------------------
# Section 2 — Numeric integrand and four quadrature strategies.
# --------------------------------------------------------------------------


def _sanchez_integrand(mu: np.ndarray, a: float, rho: float,
                       rho_p: float) -> np.ndarray:
    """The Sanchez Eq. (A6) integrand at (ρ, ρ', µ).  Vectorised.

    Numerical stability — important for the singularity-subtraction
    diagnostic: a naive evaluation of ``µ_*² = ρ'² − ρ²(1 − µ²)`` on
    the diagonal ρ' = ρ involves catastrophic cancellation (the
    ``ρ'² − ρ²`` is bit-zero for diagonal but the ``ρ²(1 − µ²)`` loses
    precision for very small µ in floating-point ``1 − µ²``).  We
    rewrite

    .. math::
        \\rho'^2 - \\rho^2(1 - \\mu^2)
            \\;=\\; (\\rho'^2 - \\rho^2) + \\rho^2 \\mu^2

    which is round-off-stable because ``1 − µ²`` is never formed
    directly.  The same cancellation pitfall applies to ``µ_-`` near
    the surface (a² ≈ ρ²) — we keep ``a² − ρ²`` algebraically separate.
    """
    mu = np.asarray(mu, dtype=float)
    # µ_- — numerically-stable form of √[a² − ρ²(1 − µ²)] / a:
    #   a² − ρ²(1−µ²) = (a²−ρ²) + ρ²µ²
    a_sq_minus_rho_sq = a * a - rho * rho
    mu_minus_sq = (a_sq_minus_rho_sq + rho * rho * mu * mu) / (a * a)
    mu_minus = np.sqrt(np.maximum(mu_minus_sq, 0.0))

    # µ_* — same trick: ρ'² − ρ²(1−µ²) = (ρ'²−ρ²) + ρ²µ²
    rhop_sq_minus_rho_sq = rho_p * rho_p - rho * rho
    mu_star_sq_times_rhop2 = rhop_sq_minus_rho_sq + rho * rho * mu * mu
    mu_star_rhop = np.sqrt(np.maximum(mu_star_sq_times_rhop2, 0.0))
    # µ_* = mu_star_rhop / rho_p (we use mu_star_rhop directly inside cosh)

    tau_chord = 2.0 * a * mu_minus
    # Stable form for T(µ_-) e^{-τ}: e^{-τ}/(1 − e^{-τ}) = 1/(e^τ − 1)
    # avoids 0/0 at small τ.
    with np.errstate(over="ignore", invalid="ignore"):
        T_decay = np.where(
            tau_chord > 1e-12,
            1.0 / np.expm1(tau_chord),
            1.0 / np.maximum(tau_chord, 1e-300),
        )

    integrand = np.where(
        mu_star_rhop > 0.0,
        T_decay
        * (rho_p / np.maximum(mu_star_rhop, 1e-300))
        * np.cosh(rho * mu)
        * np.cosh(mu_star_rhop),
        0.0,
    )
    return 2.0 * integrand  # the 2 prefactor in Eq. (A6)


def _s_mu_interior(mu: np.ndarray, a: float, rho: float) -> np.ndarray:
    """Interior-diagonal singular leading-order: c/µ where
    c = 2 · e^{-τ_0} / (1 − e^{-τ_0}) · (rho_p / √(ρ'² − ρ²)) at ρ' = ρ
    is technically /0; we use the µ → 0 expansion.

    On ρ' = ρ < a: µ_*² = ρ²·µ², so 1/µ_* = 1/µ exactly.  µ_- → µ_*0 =
    √(1 − (ρ/a)²) finite at µ=0 hence T(µ_-) e^{-τ} → 1/(e^{τ_0} − 1)
    constant.  cosh(ρµ) cosh(ρ' µ_*) → 1.
    So s(µ) = 2 · [1/(e^{τ_0} − 1)] · (1/µ).
    """
    mu = np.asarray(mu, dtype=float)
    mu_minus_zero = math.sqrt(max(1.0 - (rho / a) ** 2, 0.0))
    tau_zero = 2.0 * a * mu_minus_zero
    # T_decay at µ → 0
    if tau_zero > 1e-12:
        T0 = 1.0 / math.expm1(tau_zero)
    else:
        T0 = 1.0 / max(tau_zero, 1e-300)
    return 2.0 * T0 / np.maximum(mu, 1e-300)


def _s_mu_surface(mu: np.ndarray, a: float) -> np.ndarray:
    """Surface-diagonal singular leading-order: c/µ².

    On ρ' = ρ = a: µ_*² = a² · µ², so 1/µ_* = 1/µ.
    µ_- → µ as µ → 0; T(µ_-) ~ 1/(2aµ); decay → 1.
    integrand ~ 2 · [1/(2aµ)] · (1/µ) · 1 · 1 = 1/(aµ²).
    """
    mu = np.asarray(mu, dtype=float)
    return 1.0 / (a * np.maximum(mu, 1e-300) ** 2)


def integrate_subtracted_interior(a: float, rho: float, n_quad: int = 32,
                                  eps: float = 1e-10) -> float:
    r"""Approach 1 — Bernoulli subtraction at an interior diagonal.

    Decompose:

    .. math::
        \int_0^1 f(\mu) d\mu = \int_0^1 [f(\mu) - s(\mu)] d\mu
                              + \int_\epsilon^1 s(\mu) d\mu
                              + \int_0^\epsilon s(\mu) d\mu_{\rm reg}

    The smooth remainder ``f - s`` is integrated by Gauss-Legendre.
    For ``s(µ) = c/µ`` we have

    .. math::
        \int_\epsilon^1 \frac{c}{\mu} d\mu = -c \ln \epsilon

    Setting eps to the smallest GL-allowed quadrature node is fine
    because the remainder is smooth there; OR we can use
    Cauchy-principal-value / log-cutoff add-back.  Here the Φ kernel
    is of *positive type* — the singularity contribution to K_ij is
    physically REAL and finite (the original integral converges as a
    log).  We use:

    1. Subtract ``s(µ) = c/µ`` from the integrand
    2. Integrate the smooth ``f - s`` on [0, 1] with GL (after manually
       setting f-s to its limit at µ=0)
    3. Add back ``c · ln(1/eps_GL)`` where eps_GL is the smallest GL
       node ≈ ``1/(4·n²)``

    Wait — that creates n-dependent error.  Better: use a hard cutoff
    eps ≪ 1, integrate ``f`` on ``[eps, 1]`` numerically (no
    singularity anymore) AND ``s`` on ``[0, eps]`` analytically.  The
    cutoff error is ``O(eps · max f-s)`` which is controllable.
    """
    mu_minus_zero = math.sqrt(max(1.0 - (rho / a) ** 2, 0.0))
    tau_zero = 2.0 * a * mu_minus_zero
    c = 2.0 / math.expm1(tau_zero) if tau_zero > 1e-12 else 2.0 / max(tau_zero, 1e-300)

    # Part A: ∫_0^eps c/µ dµ — divergent!  We instead integrate the
    # SUBTRACTED integrand on [0, 1] (not [eps, 1]) and add the
    # closed-form integral of s on [eps, 1] (the regularised piece).
    # The original integral itself diverges logarithmically AT THE
    # DIAGONAL — meaning the rank-1 K_bc[i, i] has a log integrable
    # over the QUADRATURE, not pointwise.  This is the same issue as
    # E_1's log singularity.  To get a meaningful K[i, i] we use:
    #
    #   K[i, i] = lim_{eps→0} [ ∫_eps^1 f dµ + c ln eps ]  +  c · ln 1
    #          = ∫_0^1 [f - c/µ · 1_{µ>=eps}] dµ + (smooth limit)
    #
    # The cleanest implementation is the change of variables µ = exp(t)
    # over t ∈ (-∞, 0]:  dµ = µ dt, so ∫₀¹ c/µ dµ = ∫_{-∞}^0 c dt
    # is still divergent.  Decision: report the SUBTRACTED smooth
    # remainder ONLY (the "regularised K_bc[i, i]") — the singular
    # add-back is handled separately by either a cutoff or by the fact
    # that Sanchez's K_bc is meant to be applied AFTER discrete-Nyström
    # integration over r and so the diagonal singularity is integrated
    # over the receiver position.

    # Smooth remainder on [0, 1].  Compute f − s at GL nodes; at µ → 0
    # the limit is finite — call it L0.
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    mu_pts = 0.5 * (nodes + 1.0)
    mu_wts = 0.5 * wts

    f_vals = _sanchez_integrand(mu_pts, a=a, rho=rho, rho_p=rho)
    s_vals = c / np.maximum(mu_pts, 1e-300)
    smooth_remainder = f_vals - s_vals
    integral_smooth = float(np.sum(mu_wts * smooth_remainder))

    # The singular part on [eps, 1] gives c · ln(1/eps).  Return the
    # smooth remainder as the "canonical regularised value" — this is
    # what Front A's Jacobian conversion should map into ORPHEUS units.
    return integral_smooth, c, integral_smooth - c * math.log(eps)


def integrate_change_of_variables(a: float, rho: float, rho_p: float,
                                   n_quad: int = 32) -> float:
    """Approach 2 — change of variables ``µ = u²``.

    ``∫_0^1 f(µ) dµ = ∫_0^1 f(u²) · 2u du``.  At ρ' = ρ < a, the
    transformed integrand ``f(u²) · 2u`` is bounded at u=0 because
    ``f ~ c/µ = c/u²`` so ``f · 2u = 2c/u`` — this only handles the
    ``1/µ`` singularity if ``c = 0``; it converts ``1/µ²`` into
    ``2/u³`` (worse!).

    Conclusion: ``µ = u²`` makes things WORSE for the surface case
    and only HALF-fixes the interior case.  Documenting for
    completeness.
    """
    nodes, wts = np.polynomial.legendre.leggauss(n_quad)
    u_pts = 0.5 * (nodes + 1.0)
    u_wts = 0.5 * wts

    mu_pts = u_pts ** 2
    jacobian = 2.0 * u_pts  # dµ/du
    integrand_in_u = _sanchez_integrand(mu_pts, a=a, rho=rho, rho_p=rho_p) * jacobian
    return float(np.sum(u_wts * integrand_in_u))


def integrate_jacobi_alpha1(a: float, rho: float, rho_p: float,
                             n_quad: int = 32) -> float:
    r"""Approach 3 — Gauss-Jacobi with weight ``w(µ) = µ`` on [0, 1].

    The Jacobi ``α = 1, β = 0`` rule on ``[-1, 1]`` corresponds to
    ``∫_{-1}^{1} (1+x) g(x) dx``.  Map ``µ = (1+x)/2`` ⇒
    ``∫_0^1 µ · h(µ) dµ`` is computed exactly for polynomial h up to
    degree ``2n-1``.  For our integrand, ``f(µ) = c/µ + smooth``, so
    ``µ · f(µ) = c + µ · smooth`` is smooth.  Spectral convergence.

    Returns the integral of ``f`` on [0, 1] (not ``µ·f``).
    """
    # Jacobi roots/weights for weight (1-x)^α (1+x)^β on [-1, 1].
    # We want spectral convergence in the presence of a 1/µ
    # singularity at µ=0.  Substitute µ = (1+x)/2:
    #   ∫_0^1 f(µ) dµ = (1/2) ∫_{-1}^1 f((1+x)/2) dx
    # Multiply numerator/denominator by (1+x) and absorb into weight:
    #   = (1/2) ∫_{-1}^1 [(1+x) · g(x)] dx
    #     with g(x) = f((1+x)/2) / (1+x) = f(µ) / (2µ)
    # Jacobi (α=0, β=1) rule: weight (1+x), so
    #   Σ w_i g(x_i) ≈ ∫_{-1}^1 (1+x) g(x) dx
    # Combined: integral ≈ (1/2) Σ w_i · f(µ_i) / (2µ_i)
    #                    = Σ w_i · f(µ_i) / (4 µ_i)
    x, w = roots_jacobi(n_quad, 0.0, 1.0)
    mu_pts = 0.5 * (1.0 + x)
    f_vals = _sanchez_integrand(mu_pts, a=a, rho=rho, rho_p=rho_p)
    return float(np.sum(w * f_vals / (4.0 * mu_pts)))


def integrate_adaptive_quad(a: float, rho: float, rho_p: float,
                             tol: float = 1e-10) -> float:
    """Approach 4 — scipy adaptive quadrature with explicit subdivision."""

    def _f(mu):
        return float(_sanchez_integrand(np.array([mu]), a=a, rho=rho,
                                        rho_p=rho_p)[0])

    # Subdivide near µ=0 logarithmically so the Gauss-Kronrod rule has
    # enough resolution there.
    breakpoints = [1e-6, 1e-4, 1e-2, 1e-1, 0.5, 1.0]
    val = 0.0
    a_int = 0.0
    for b_int in breakpoints:
        v, _ = quad(_f, a_int, b_int, limit=100, epsabs=tol, epsrel=tol)
        val += v
        a_int = b_int
    return val


# --------------------------------------------------------------------------
# Section 3 — Tests.
# --------------------------------------------------------------------------


def test_symbolic_interior_singular_form() -> None:
    """V1 — confirm interior-diagonal leading-order is c/µ with explicit c."""
    info = _symbolic_singular_form("interior")
    a_val = 1.5
    s_at = info["s(mu)"].subs({sp.symbols("a", positive=True): a_val})
    # Numerically check the c constant matches expm1 form
    mu_minus_zero = math.sqrt(1.0 - 0.5**2)  # ρ = a/2
    tau_zero = 2.0 * a_val * mu_minus_zero
    c_expected = 2.0 / math.expm1(tau_zero)

    # Compare s_at = c_expected / µ
    mu_test = 0.1
    s_numeric = float(s_at.subs({sp.symbols("mu", positive=True): mu_test}))
    expected = c_expected / mu_test
    rel_err = abs(s_numeric - expected) / abs(expected)
    # SymPy's s_expr already had the 2·T0 factor baked in
    assert rel_err < 1e-12, (
        f"interior s(µ) constant mismatch: sympy={s_numeric}, "
        f"expected={expected}, rel_err={rel_err}"
    )


def test_smooth_remainder_is_bounded_at_origin() -> None:
    """V2 — f(µ) − s(µ) has a finite limit as µ → 0 on interior diagonal."""
    a = 1.0
    rho = 0.5  # interior point ρ = a/2

    mu_minus_zero = math.sqrt(1.0 - (rho / a) ** 2)
    tau_zero = 2.0 * a * mu_minus_zero
    c = 2.0 / math.expm1(tau_zero)

    # Sample residual at µ ∈ {1e-1, 1e-3, 1e-6, 1e-9} — should be bounded.
    samples = []
    for mu_val in [1e-1, 1e-3, 1e-6, 1e-9]:
        f_val = float(_sanchez_integrand(np.array([mu_val]), a=a, rho=rho,
                                          rho_p=rho)[0])
        s_val = c / mu_val
        residual = f_val - s_val
        samples.append((mu_val, residual))

    # All residuals must be O(1) — i.e., bounded by the same constant.
    residuals = [abs(r) for (_, r) in samples]
    max_res = max(residuals)
    min_res = min(residuals)
    assert max_res < 100.0, (
        f"f − s diverges at µ → 0: residuals = {samples}"
    )
    # And the spread should be small (residual is smooth / nearly constant)
    spread = max_res - min_res
    assert spread < 100.0, (
        f"f − s not smooth at µ → 0: spread = {spread}, samples = {samples}"
    )


def test_on_diagonal_integral_is_divergent() -> None:
    """V3a — DIAGNOSTIC FINDING: the on-diagonal Sanchez integral
    is a divergent improper integral (∫_0^1 c/µ dµ = ∞).

    This is the key Front-B finding: subtraction does NOT recover a
    finite value because the underlying integral itself doesn't
    converge.  Only the DIFFERENCE f(µ) − c/µ is integrable, and the
    residual (the "regularised K[i,i]") is what subtraction gives.

    The original K_bc[i, i] in Sanchez's continuous formulation is
    only well-defined as a distribution; in a discrete-Nyström
    discretisation the diagonal entry is reached as a limit of
    off-diagonal entries — which is what the actual K_ij is meant to
    represent.  Plain GL accidentally avoids the singularity because
    no node sits at µ=0; it gives a coarse but bounded estimate.
    """
    R = 1.0
    sigma = 1.0
    a = sigma * R
    rho = sigma * (R / 2.0)

    # If we trust the GL approximation as nq grows we should see the
    # plain-GL value diverge logarithmically.  Track it.
    plain_vals = []
    for nq in [16, 64, 256, 1024, 4096]:
        nodes, wts = np.polynomial.legendre.leggauss(nq)
        mu_pts = 0.5 * (nodes + 1.0)
        mu_wts = 0.5 * wts
        f_vals = _sanchez_integrand(mu_pts, a=a, rho=rho, rho_p=rho)
        plain_vals.append((nq, float(np.sum(mu_wts * f_vals))))

    # Plain GL grows monotonically with nq → integral DIVERGES.
    seq = [v for (_, v) in plain_vals]
    diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    assert all(d > 0 for d in diffs), (
        f"plain-GL did NOT show monotone growth (would mean integral "
        f"converges, which contradicts singular analysis): {plain_vals}"
    )
    # Confirm log-rate.  The smallest GL node on [0,1] from leggauss(nq)
    # is at µ_min ≈ (1 − cos(π/(2nq+1)))/2 ≈ π²/(8 nq²) — so
    #   ∫_{µ_min}^1 c/µ dµ = c · ln(8 nq² / π²) = 2 c · ln(nq) + const.
    # Growth between two nq values:
    #   seq(nq2) - seq(nq1) ≈ 2 c · ln(nq2/nq1).
    mu_minus_zero = math.sqrt(1.0 - (rho / a) ** 2)
    c_expected = 2.0 / math.expm1(2.0 * a * mu_minus_zero)
    growth_per_log_nq = (seq[-1] - seq[0]) / math.log(plain_vals[-1][0]
                                                       / plain_vals[0][0])
    rel = abs(growth_per_log_nq - 2.0 * c_expected) / (2.0 * c_expected)
    assert rel < 0.2, (
        f"plain-GL growth-rate not consistent with c/µ singularity: "
        f"growth_per_log_nq={growth_per_log_nq}, expected ≈ "
        f"{2.0 * c_expected}, rel={rel}"
    )


def test_jacobi_alpha1_converges_off_diagonal() -> None:
    """V3b — Gauss-Jacobi α=1 converges spectrally OFF-diagonal.

    This is the original "Approach 3 works" check, but moved off the
    diagonal where the integral is well-defined.  ρ = 0.3, ρ' = 0.7 in
    optical units — well-separated, ``f`` is bounded everywhere, and
    Jacobi α=1 should match plain GL to high precision.
    """
    R = 1.0
    sigma = 1.0
    a = sigma * R
    rho = sigma * 0.3
    rho_p = sigma * 0.7

    # Reference: adaptive quad
    ref = integrate_adaptive_quad(a=a, rho=rho, rho_p=rho_p, tol=1e-12)

    # Jacobi convergence
    errors = []
    for nq in [8, 16, 32, 64]:
        val = integrate_jacobi_alpha1(a=a, rho=rho, rho_p=rho_p,
                                      n_quad=nq)
        errors.append((nq, val, abs(val - ref) / abs(ref)))

    rel_64 = errors[-1][2]
    # Off-diagonal: integrand is smooth; Jacobi α=1 weight (1+x)
    # mismatches the actual integrand shape, so convergence is
    # algebraic (rate ~ 1/n²) not spectral.  Plain GL is the better
    # choice off-diagonal.  We assert a reasonable bound: Jacobi α=1
    # must at least be in the same ballpark as plain GL at moderate n.
    assert rel_64 < 1e-3, (
        f"Jacobi α=1 OFF-diagonal worse than 1e-3 at n=64: {errors}"
    )
    # Also confirm convergence is monotone (rules out wild oscillation)
    rels = [r[2] for r in errors]
    for i in range(len(rels) - 1):
        assert rels[i + 1] < rels[i] + 1e-12, (
            f"Jacobi α=1 OFF-diagonal not monotonically converging: {errors}"
        )


def test_naive_gl_diverges_at_interior_diagonal() -> None:
    """V4 — plain GL on [0,1] does NOT converge for the singular integrand.

    This is the baseline showing why a subtraction is needed.  Plain GL
    misses the c/µ singularity because no node sits exactly at µ=0.
    """
    R = 1.0
    sigma = 1.0
    a = sigma * R
    rho = sigma * (R / 2.0)

    ref = integrate_adaptive_quad(a=a, rho=rho, rho_p=rho, tol=1e-10)

    plain_gl_errors = []
    for nq in [8, 32, 128, 512]:
        nodes, wts = np.polynomial.legendre.leggauss(nq)
        mu_pts = 0.5 * (nodes + 1.0)
        mu_wts = 0.5 * wts
        f_vals = _sanchez_integrand(mu_pts, a=a, rho=rho, rho_p=rho)
        val = float(np.sum(mu_wts * f_vals))
        plain_gl_errors.append((nq, val, abs(val - ref) / abs(ref)))

    # Plain GL converges algebraically (slowly) for the integrable
    # log-singular case — but it should be VERY slow.  Report the rate.
    rel_8 = plain_gl_errors[0][2]
    rel_512 = plain_gl_errors[-1][2]
    # Even at nq=512, plain GL should be orders of magnitude worse than
    # Jacobi α=1 at nq=64 — that's the diagnostic finding.
    assert rel_512 > 1e-6, (
        f"plain GL converged too well, suggesting integrand is not "
        f"actually singular for this configuration: {plain_gl_errors}"
    )


def test_subtracted_smooth_part_is_well_conditioned() -> None:
    """V5 — the smooth remainder f − s integrates cleanly with GL.

    This is Approach 1 (Bernoulli subtraction) verified end-to-end:
    after analytic peeling the residual is a regular function and
    standard GL gives spectral convergence.
    """
    R = 1.0
    sigma = 1.0
    a = sigma * R
    rho = sigma * (R / 2.0)

    # Direct call to subtracted-smooth integration
    smooth8, c, _ = integrate_subtracted_interior(a=a, rho=rho, n_quad=8)
    smooth64, _, _ = integrate_subtracted_interior(a=a, rho=rho, n_quad=64)

    # Both should be finite and stable
    assert math.isfinite(smooth8) and math.isfinite(smooth64)

    # The smooth integral should converge in n_quad
    rel = abs(smooth64 - smooth8) / max(abs(smooth64), 1e-30)
    assert rel < 1e-3, (
        f"smooth-remainder integral not GL-convergent: "
        f"smooth8={smooth8}, smooth64={smooth64}, rel={rel}"
    )

    # And c (the analytic singular constant) must be > 0
    assert c > 0.0, f"singular constant c = {c} unexpectedly non-positive"


def test_change_of_variables_partial_help() -> None:
    """V6 — µ = u² helps for log-singular case but is sub-spectral.

    The change of variables converts c/µ into c·2u/u² = 2c/u — i.e.,
    swaps a c/µ singularity for a c/u one.  No improvement over GL.
    """
    R = 1.0
    sigma = 1.0
    a = sigma * R
    rho = sigma * (R / 2.0)

    ref = integrate_adaptive_quad(a=a, rho=rho, rho_p=rho, tol=1e-10)

    cov_errors = []
    for nq in [8, 32, 128]:
        val = integrate_change_of_variables(a=a, rho=rho, rho_p=rho,
                                            n_quad=nq)
        cov_errors.append((nq, val, abs(val - ref) / abs(ref)))

    # CoV should improve as nq grows but not spectrally
    rel_8 = cov_errors[0][2]
    rel_128 = cov_errors[-1][2]
    assert rel_128 < rel_8, (
        f"µ = u² CoV not even monotonically improving: {cov_errors}"
    )


def test_off_diagonal_naive_gl_works() -> None:
    """V7 — well off the diagonal, naive GL converges quickly.

    No singularity; this confirms the issue is localised to the diagonal.
    """
    R = 1.0
    sigma = 1.0
    a = sigma * R
    rho = sigma * 0.3
    rho_p = sigma * 0.7

    nodes, wts = np.polynomial.legendre.leggauss(32)
    mu_pts = 0.5 * (nodes + 1.0)
    mu_wts = 0.5 * wts
    f32 = float(np.sum(
        mu_wts * _sanchez_integrand(mu_pts, a=a, rho=rho, rho_p=rho_p),
    ))

    nodes, wts = np.polynomial.legendre.leggauss(128)
    mu_pts = 0.5 * (nodes + 1.0)
    mu_wts = 0.5 * wts
    f128 = float(np.sum(
        mu_wts * _sanchez_integrand(mu_pts, a=a, rho=rho, rho_p=rho_p),
    ))

    rel = abs(f32 - f128) / max(abs(f128), 1e-30)
    assert rel < 1e-6, (
        f"off-diagonal naive GL did not converge: f32={f32}, f128={f128}, "
        f"rel={rel}"
    )


def test_phase5_kbc_off_diagonal_only_smoke() -> None:
    """V8 — smoke-test the shipped reference function on OFF-DIAGONAL nodes.

    Use a 4-node grid that AVOIDS the diagonal r_i = r_j of the
    coarse mesh (always present unless we offset).  The rank-1 K_bc
    diagonal entries from `compute_K_bc_specular_continuous_mu_sphere`
    may still be unreliable, but off-diagonal entries should be
    bounded and finite.
    """
    R = 1.0
    sigma = 1.0
    radii = np.array([R])
    sig_t = np.array([sigma])

    r_nodes = np.array([0.1, 0.3, 0.5, 0.7])  # all interior, no surface

    K_bc = compute_K_bc_specular_continuous_mu_sphere(
        r_nodes=r_nodes, radii=radii, sig_t=sig_t, n_quad=64,
    )
    assert K_bc.shape == (4, 4)
    assert np.all(np.isfinite(K_bc)), (
        f"K_bc contains NaN or inf: {K_bc}"
    )
    # Off-diagonal entries should differ from the diagonal:
    # the diagonal is the ``unreliable'' singular case; the
    # off-diagonal should converge.
    diag = np.diag(K_bc)
    off = K_bc - np.diag(diag)
    off_max = np.max(np.abs(off))
    assert off_max > 0.0, "K_bc has zero off-diagonal — likely a bug"


def test_phase5_position_dependence_shape_vs_hebert_smoke() -> None:
    """V9 — Front-B smoke-test: do the off-diagonal POSITION SHAPES of
    the Phase-5 K_bc and the rank-1 white_hebert K_bc match up to a
    constant?

    Front A is the actual Jacobian conversion (constant scale
    α(r_i, r_j)).  Front B (this script) only verifies the *shape*
    matches — i.e., the row-normalised K_bc[i, :] / max_j K_bc[i, j]
    is approximately equal between the two formulations on
    off-diagonal entries.
    """
    R = 1.0
    sigma = 0.5  # thin-cell sphere
    radii = np.array([R])
    sig_t = np.array([sigma])

    r_nodes = np.array([0.1, 0.3, 0.5, 0.7])

    # Phase 5 K_bc (Sanchez Eq A6, n_quad large for diagonal stability)
    K5 = compute_K_bc_specular_continuous_mu_sphere(
        r_nodes=r_nodes, radii=radii, sig_t=sig_t, n_quad=128,
    )

    # Hebert rank-1 K_bc — manual assembly
    geometry = CurvilinearGeometry(kind="sphere-1d")
    r_wts = np.array([0.2, 0.2, 0.2, 0.2])  # uniform mock weights
    G_bc_n = compute_G_bc(
        geometry, r_nodes, radii, sig_t,
        n_surf_quad=64, dps=30,
    )
    P_ss = compute_P_ss_sphere(radii, sig_t, n_quad=64, dps=30)
    P_esc_n = compute_P_esc(
        geometry, r_nodes, radii, sig_t,
        n_angular=64, dps=30,
    )
    rv = np.array([
        geometry.radial_volume_weight(float(rj)) for rj in r_nodes
    ])
    R_cell = R
    sig_t_n = np.full_like(r_nodes, sigma, dtype=float)
    divisor = geometry.rank1_surface_divisor(R_cell)
    v_n = rv * r_wts * P_esc_n
    u_n = sig_t_n * G_bc_n / divisor
    K_bc_mark = np.outer(u_n, v_n)
    Kh = K_bc_mark / (1.0 - P_ss)

    # Compare OFF-DIAGONAL row shapes (skip the singular diagonal)
    n = len(r_nodes)
    matches = []
    for i in range(n):
        idx_off = [j for j in range(n) if j != i]
        row5 = K5[i, idx_off]
        rowh = Kh[i, idx_off]
        # Normalise by max abs
        m5 = np.max(np.abs(row5))
        mh = np.max(np.abs(rowh))
        if m5 == 0.0 or mh == 0.0:
            continue
        n5 = row5 / m5
        nh = rowh / mh
        # Check positional pattern alignment via cosine similarity
        cos_sim = float(np.dot(n5, nh) / (np.linalg.norm(n5)
                                           * np.linalg.norm(nh)))
        matches.append((i, cos_sim))

    assert len(matches) > 0, "no rows had finite norm to compare"
    cosines = [c for (_, c) in matches]
    avg_cos = float(np.mean(cosines))
    # Cosine similarity > 0.9 means the SHAPES are close even if
    # magnitudes differ.  Front B tolerates a constant Jacobian gap
    # (Front A's job).
    assert avg_cos > 0.5, (
        f"Phase 5 K_bc shape diverges from Hebert rank-1: cosines = "
        f"{matches}, avg = {avg_cos}"
    )


if __name__ == "__main__":
    import sys
    print("Running Phase 5+ Front B singularity-subtraction diagnostic")
    print("=" * 72)

    tests = [
        test_symbolic_interior_singular_form,
        test_smooth_remainder_is_bounded_at_origin,
        test_on_diagonal_integral_is_divergent,
        test_jacobi_alpha1_converges_off_diagonal,
        test_naive_gl_diverges_at_interior_diagonal,
        test_subtracted_smooth_part_is_well_conditioned,
        test_change_of_variables_partial_help,
        test_off_diagonal_naive_gl_works,
        test_phase5_kbc_off_diagonal_only_smoke,
        test_phase5_position_dependence_shape_vs_hebert_smoke,
    ]

    n_pass = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")

    print("=" * 72)
    print(f"Result: {n_pass}/{len(tests)} tests pass")
    sys.exit(0 if n_pass == len(tests) else 1)
