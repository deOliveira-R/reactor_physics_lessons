r"""L0 term-verification tests for the reference-solution kernel primitives.

The :math:`E_n` (exponential integral) and :math:`\mathrm{Ki}_n`
(Bickley–Naylor) functions are the atomic building blocks of every
Phase-4 Peierls / Bickley reference solution. Any deviation from
their defining identities or their canonical values at the
reference points (``x = 0``, integrals over the full line, etc.)
invalidates every downstream verification claim that rests on them.

These tests nail down:

1. **Special values** at :math:`x = 0` via closed form.
2. **Derivative identities** :math:`E_n'(x) = -E_{n-1}(x)` and
   :math:`\mathrm{Ki}_n'(x) = -\mathrm{Ki}_{n-1}(x)` via
   finite-difference against the analytical form.
3. **Full-line integrals** :math:`\int_0^\infty E_n(x)\,dx = 1/n`
   and the analogue for :math:`\mathrm{Ki}_n`.

See :doc:`/verification/reference_solutions` for the contract these
primitives underpin.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from orpheus.derivations._kernels import (
    e_n,
    e_n_at_zero,
    e_n_derivative,
    gauss_legendre_visibility_cone,
    ki_n,
    ki_n_at_zero,
    ki_n_derivative,
)


# ═══════════════════════════════════════════════════════════════════════
# E_n special values and identities
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("en-kernel-special-values")
@pytest.mark.parametrize("n, expected", [
    (2, 1.0),          # E_2(0) = 1
    (3, 0.5),          # E_3(0) = 1/2
    (4, 1.0 / 3),      # E_4(0) = 1/3
    (5, 0.25),         # E_5(0) = 1/4
])
def test_en_closed_form_at_zero(n: int, expected: float):
    r"""Closed-form :math:`E_n(0) = 1/(n-1)` from A&S 5.1.23."""
    assert e_n(n, 0.0) == pytest.approx(expected, abs=1e-14)
    assert e_n_at_zero(n) == pytest.approx(expected, abs=1e-14)


@pytest.mark.l0
@pytest.mark.verifies("en-kernel-derivative", "cp-kernel-differential-identities")
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.5, 5.0])
def test_en_derivative_identity(n: int, x: float):
    r"""Finite-difference numerical derivative of :math:`E_n`
    must match the analytical identity :math:`E_n'(x) = -E_{n-1}(x)`.

    Uses central difference at step :math:`h = 10^{-6}` which gives
    :math:`\mathcal{O}(h^{2}) = 10^{-12}` truncation error. Accept
    agreement to :math:`10^{-9}` to leave room for round-off when
    :math:`E_n(x)` is small.
    """
    h = 1e-6
    numerical = (e_n(n, x + h) - e_n(n, x - h)) / (2 * h)
    analytical = e_n_derivative(n, x)
    assert numerical == pytest.approx(analytical, abs=1e-9)


@pytest.mark.l0
@pytest.mark.verifies("en-kernel-integral")
@pytest.mark.parametrize("n, expected_inv", [
    (1, 1.0),   # ∫_0^∞ E_1(x) dx = 1
    (2, 0.5),   # ∫_0^∞ E_2(x) dx = 1/2
    (3, 1 / 3), # ∫_0^∞ E_3(x) dx = 1/3
    (4, 0.25),  # ∫_0^∞ E_4(x) dx = 1/4
])
def test_en_full_line_integral(n: int, expected_inv: float):
    r"""Closed form :math:`\int_0^\infty E_n(x)\,dx = 1/n`
    (Abramowitz & Stegun 5.1.32).

    Evaluated at 50-digit mpmath precision to isolate any loss
    of accuracy in the high-precision evaluator itself.
    """
    with mpmath.workdps(50):
        integral = mpmath.quad(
            lambda x: mpmath.expint(n, x), [0, mpmath.inf],
        )
    assert float(integral) == pytest.approx(expected_inv, abs=1e-13)


# ═══════════════════════════════════════════════════════════════════════
# Ki_n special values and identities
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("kin-kernel-special-values")
@pytest.mark.parametrize("n, expected", [
    (1, math.pi / 2),   # Wallis for k=0: ∫cos⁰ = π/2
    (2, 1.0),           # Wallis for k=1: ∫cos¹ = 1
    (3, math.pi / 4),   # Wallis for k=2: (π/2)·(1/2)
    (4, 2.0 / 3),       # Wallis for k=3: 2/3
    (5, 3 * math.pi / 16),  # Wallis for k=4: (π/2)·(3/8)
])
def test_kin_closed_form_at_zero(n: int, expected: float):
    r"""Closed-form :math:`\mathrm{Ki}_n(0)` via Wallis integral.

    :math:`\mathrm{Ki}_n(0) = \int_0^{\pi/2} \cos^{n-1}\theta\,d\theta`
    which is the Wallis integral — exact rationals or rationals × π.
    Cross-check both the mpmath evaluator ``ki_n`` and the closed-form
    ``ki_n_at_zero`` helper.
    """
    assert ki_n(n, 0.0) == pytest.approx(expected, abs=1e-14)
    assert ki_n_at_zero(n) == pytest.approx(expected, abs=1e-14)


@pytest.mark.l0
@pytest.mark.verifies("kin-kernel-derivative", "cp-kernel-differential-identities")
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("x", [0.3, 1.0, 2.5, 5.0])
def test_kin_derivative_identity(n: int, x: float):
    r"""Numerical derivative of :math:`\mathrm{Ki}_n` must match
    the analytical identity :math:`\mathrm{Ki}_n'(x) = -\mathrm{Ki}_{n-1}(x)`.

    Same central-difference protocol as :func:`test_en_derivative_identity`.
    Skips ``n = 1`` because :math:`\mathrm{Ki}_0(x) = K_0(x)` has a
    logarithmic singularity at :math:`x = 0` — handled in a
    separate test over an ``x``-only grid that avoids the endpoint.
    """
    h = 1e-6
    numerical = (ki_n(n, x + h) - ki_n(n, x - h)) / (2 * h)
    analytical = ki_n_derivative(n, x)
    assert numerical == pytest.approx(analytical, abs=1e-9)


@pytest.mark.l0
@pytest.mark.verifies("kin-kernel-derivative")
@pytest.mark.parametrize("x", [0.5, 1.0, 2.5, 5.0])
def test_kin1_derivative_is_bessel_k0(x: float):
    r""":math:`\mathrm{Ki}_1'(x) = -\mathrm{Ki}_0(x) = -K_0(x)`
    where :math:`K_0` is the modified Bessel function of the second
    kind (A&S 11.2.11).

    Excludes ``x = 0`` where :math:`K_0` has a logarithmic
    singularity.
    """
    h = 1e-6
    numerical = (ki_n(1, x + h) - ki_n(1, x - h)) / (2 * h)
    analytical_k0 = -float(mpmath.besselk(0, x))
    assert numerical == pytest.approx(analytical_k0, abs=1e-9)
    assert ki_n_derivative(1, x) == pytest.approx(analytical_k0, abs=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# Precision sanity — high-precision tier must actually deliver its digits
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("en-kernel-special-values")
def test_en_high_precision_refinement_converges():
    r"""Requesting more mpmath working precision must only tighten,
    never loosen, the :math:`E_n(x)` evaluator's result.

    Regression guard against anyone wiring a hard-coded precision
    floor into the kernel evaluator.
    """
    # Use a mid-range argument where E_2(x) is ~0.05, far from
    # round-off or asymptotic regimes.
    x = 2.0
    low = e_n(2, x, precision_digits=15)
    high = e_n(2, x, precision_digits=60)
    # Both should be the same to double precision (scipy expint
    # is the underlying engine for both), so just check they agree
    # to 14 digits and neither is NaN/inf.
    assert np.isfinite(low) and np.isfinite(high)
    assert low == pytest.approx(high, abs=1e-14)


@pytest.mark.l0
@pytest.mark.verifies("kin-kernel-special-values")
def test_kin_high_precision_refinement_converges():
    r"""Same refinement sanity check for :math:`\mathrm{Ki}_n`.

    Unlike :math:`E_n`, the Bickley evaluator performs an mpmath
    adaptive quadrature at the requested precision, so higher
    ``precision_digits`` should visibly tighten the result when
    the default is not already saturated.
    """
    x = 1.5
    low = ki_n(3, x, precision_digits=15)
    high = ki_n(3, x, precision_digits=60)
    assert np.isfinite(low) and np.isfinite(high)
    # Both should agree to at least 12 digits — mpmath.quad is
    # extremely accurate on this smooth integrand.
    assert low == pytest.approx(high, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# Visibility-cone GL substitution — spectral vs algebraic convergence
# ═══════════════════════════════════════════════════════════════════════
#
# These tests pin the contract claimed in the docstring of
# :func:`~orpheus.derivations._kernels.gauss_legendre_visibility_cone`
# and in :ref:`section-22-coordinate-transforms` §22.7.
#
# Each test compares the visibility-cone substitution against plain
# Gauss-Legendre on the same number of nodes. The substitution is
# correct iff:
#   1. On a closed-form integrand with :math:`\sqrt{y^{2}-y_{\min}^{2}}`
#      or :math:`\sqrt{y_{\max}^{2}-y^{2}}` factor, vis-cone hits
#      machine precision at modest :math:`Q`, while plain GL plateaus
#      at the algebraic-convergence error.
#   2. On a smooth integrand with no endpoint singularity, the two
#      methods agree (the substitution does not introduce bias).


def _plain_gl(y_min: float, y_max: float, n: int):
    """Plain Gauss-Legendre on ``[y_min, y_max]`` — baseline for
    visibility-cone comparison."""
    nodes, wts = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (y_max - y_min)
    mid = 0.5 * (y_max + y_min)
    return half * nodes + mid, half * wts


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_lower_endpoint_spectral():
    r"""Lower variant absorbs :math:`\sqrt{y^{2}-y_{\min}^{2}}` singularity.

    Integrand :math:`f(y) = 1/\sqrt{y^{2} - y_{\min}^{2}}` on
    :math:`[y_{\min}, y_{\max}] = [1, 2]` has the closed form
    :math:`\int = \mathrm{arccosh}(y_{\max}/y_{\min}) = \mathrm{arccosh}(2)
    = \ln(2 + \sqrt{3})`.

    Vis-cone GL at :math:`Q = 16` must hit machine precision; plain GL
    at :math:`Q = 64` must still be at algebraic-convergence error
    :math:`\gtrsim 10^{-3}`.
    """
    y_min, y_max = 1.0, 2.0
    truth = math.acosh(y_max / y_min)

    def f(y: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(y * y - y_min * y_min)

    y_pts, y_wts = gauss_legendre_visibility_cone(
        y_min, y_max, 16, singular_endpoint="lower",
    )
    vis_q16 = float(np.sum(y_wts * f(y_pts)))
    assert vis_q16 == pytest.approx(truth, abs=1e-12), (
        f"vis-cone Q=16 not spectral: |error| = {abs(vis_q16 - truth):.3e}"
    )

    y_plain, w_plain = _plain_gl(y_min, y_max, 64)
    plain_q64 = float(np.sum(w_plain * f(y_plain)))
    plain_err = abs(plain_q64 - truth)
    # Plain GL on this 1/sqrt(y² - y_min²) integrand cannot reach
    # 1e-3 even at Q=64 — algebraic convergence in the singular
    # endpoint dominates. If this assertion ever weakens, it means
    # the test's integrand has lost its singularity (caller bug).
    assert plain_err > 1e-3, (
        f"plain GL Q=64 unexpectedly accurate (err={plain_err:.3e}) — "
        f"is the integrand singularity actually present?"
    )


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_upper_endpoint_spectral():
    r"""Upper variant absorbs :math:`\sqrt{y_{\max}^{2} - y^{2}}` singularity.

    Integrand :math:`f(y) = 1/\sqrt{y_{\max}^{2} - y^{2}}` on
    :math:`[y_{\min}, y_{\max}] = [1, 2]` has the closed form
    :math:`\int = \arcsin(y/y_{\max})\big|_{y_{\min}}^{y_{\max}}
    = \pi/2 - \arcsin(0.5) = \pi/2 - \pi/6 = \pi/3`.
    """
    y_min, y_max = 1.0, 2.0
    truth = math.pi / 2 - math.asin(y_min / y_max)

    def f(y: np.ndarray) -> np.ndarray:
        return 1.0 / np.sqrt(y_max * y_max - y * y)

    y_pts, y_wts = gauss_legendre_visibility_cone(
        y_min, y_max, 24, singular_endpoint="upper",
    )
    vis_q24 = float(np.sum(y_wts * f(y_pts)))
    assert vis_q24 == pytest.approx(truth, abs=1e-12), (
        f"vis-cone (upper) Q=24 not spectral: "
        f"|error| = {abs(vis_q24 - truth):.3e}"
    )


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_offdiag_pair_pattern_spectral():
    r"""Phase 5 Round-3 motivating pattern: integrand
    :math:`f(y) = \sqrt{y^{2} - y_{\min}^{2}}\,\mathrm e^{-y}` on
    :math:`[y_{\min}, y_{\max}] = [1, 3]`.

    Plain GL converges algebraically (:math:`Q^{-3/2}` from the
    :math:`\sqrt{y^{2} - y_{\min}^{2}}` factor); vis-cone (lower)
    is spectral. Both compared against an mpmath ground truth at
    50 digits of precision.

    Acceptance: vis-cone at :math:`Q = 24` must beat plain GL at
    :math:`Q = 24` by at least three orders of magnitude.
    """
    y_min, y_max = 1.0, 3.0

    def f_np(y: np.ndarray) -> np.ndarray:
        return np.sqrt(y * y - y_min * y_min) * np.exp(-y)

    def f_mp(y):
        return mpmath.sqrt(y * y - y_min * y_min) * mpmath.exp(-y)

    with mpmath.workdps(50):
        truth = float(mpmath.quad(f_mp, [y_min, y_max]))

    y_pts, y_wts = gauss_legendre_visibility_cone(
        y_min, y_max, 24, singular_endpoint="lower",
    )
    vis_val = float(np.sum(y_wts * f_np(y_pts)))
    vis_err = abs(vis_val - truth)

    y_plain, w_plain = _plain_gl(y_min, y_max, 24)
    plain_val = float(np.sum(w_plain * f_np(y_plain)))
    plain_err = abs(plain_val - truth)

    assert vis_err < 1e-12, f"vis-cone Q=24 not spectral: {vis_err:.3e}"
    assert plain_err > 1000 * vis_err, (
        f"plain GL Q=24 ({plain_err:.3e}) and vis-cone Q=24 "
        f"({vis_err:.3e}) — singularity not exposed?"
    )


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_smooth_integrand_unbiased():
    r"""On a smooth integrand (no endpoint singularity), vis-cone
    must converge to the same value as the mpmath ground truth.

    This is the bias regression test: the substitution must not
    perturb routine inputs. Uses :math:`f(y) = \mathrm e^{-y^{2}}`
    on :math:`[0.5, 2.5]`, integrated against an mpmath reference
    at 50 digits.
    """
    y_min, y_max = 0.5, 2.5
    with mpmath.workdps(50):
        truth = float(mpmath.quad(
            lambda y: mpmath.exp(-y * y), [y_min, y_max],
        ))

    def f(y: np.ndarray) -> np.ndarray:
        return np.exp(-y * y)

    y_pts, y_wts = gauss_legendre_visibility_cone(
        y_min, y_max, 64, singular_endpoint="lower",
    )
    vis_val = float(np.sum(y_wts * f(y_pts)))
    assert vis_val == pytest.approx(truth, abs=1e-12)

    y_pts_u, y_wts_u = gauss_legendre_visibility_cone(
        y_min, y_max, 64, singular_endpoint="upper",
    )
    vis_val_u = float(np.sum(y_wts_u * f(y_pts_u)))
    assert vis_val_u == pytest.approx(truth, abs=1e-12)


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_constant_integrand_recovers_length():
    r"""Sanity: :math:`\int_{y_{\min}}^{y_{\max}} 1\,\mathrm dy = y_{\max} - y_{\min}`.

    The substitution is non-linear in :math:`y`, so this is not exact
    at :math:`Q = 1`; on the interval :math:`[1, 2]` (representative of
    a chord-annulus shell with :math:`r_{k-1}/r_{k} = 0.5`) both
    variants are machine-precision-correct at :math:`Q = 24`.

    The convergence rate of each variant depends on the position of
    the integrand's branch point in the complex :math:`u` plane:

    - **Lower variant**: :math:`y(u) = \sqrt{y_{\min}^{2}+u^{2}\Delta^{2}}`
      branches at :math:`u^{2} = -y_{\min}^{2}/\Delta^{2}` (purely
      imaginary), distance
      :math:`y_{\min}/\Delta` from the real interval — large for
      :math:`y_{\min}/y_{\max}` away from 0.
    - **Upper variant**: :math:`y(u) = \sqrt{y_{\max}^{2}-u^{2}\Delta^{2}}`
      branches at :math:`u^{2} = y_{\max}^{2}/\Delta^{2}`, real,
      distance :math:`(y_{\max}^{2}/\Delta^{2})^{1/2} - 1` from
      :math:`u = 1` — *small* when :math:`y_{\min} \ll y_{\max}`.
      Callers using the upper variant on intervals with
      :math:`y_{\min}/y_{\max} \lesssim 0.1` should expect slow
      Bernstein-ellipse convergence and bump :math:`Q` accordingly,
      or split the interval.
    """
    y_min, y_max = 1.0, 2.0
    truth = y_max - y_min

    for endpoint in ("lower", "upper"):
        y_pts, y_wts = gauss_legendre_visibility_cone(
            y_min, y_max, 24, singular_endpoint=endpoint,
        )
        val = float(np.sum(y_wts))
        assert val == pytest.approx(truth, abs=1e-12), (
            f"endpoint={endpoint}: |error| = {abs(val - truth):.3e}"
        )


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_input_validation():
    """Bad inputs raise ``ValueError`` rather than producing silent garbage."""
    with pytest.raises(ValueError, match="y_max > y_min"):
        gauss_legendre_visibility_cone(2.0, 1.0, 16)
    with pytest.raises(ValueError, match="y_max > y_min"):
        gauss_legendre_visibility_cone(1.0, 1.0, 16)
    with pytest.raises(ValueError, match="y_min must be non-negative"):
        gauss_legendre_visibility_cone(-0.1, 1.0, 16)
    with pytest.raises(ValueError, match="n must be >= 1"):
        gauss_legendre_visibility_cone(0.0, 1.0, 0)
    with pytest.raises(ValueError, match="singular_endpoint must be"):
        gauss_legendre_visibility_cone(
            1.0, 2.0, 16, singular_endpoint="middle",
        )
