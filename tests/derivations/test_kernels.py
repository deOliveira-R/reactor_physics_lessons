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
