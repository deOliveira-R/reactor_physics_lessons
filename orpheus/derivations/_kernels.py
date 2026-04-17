r"""Special-function kernels for continuous reference solutions.

Two tiers coexist for the verification migration:

- **Legacy tier** (``e3``, ``BickleyTables``): double-precision
  wrappers around ``scipy.special.expn`` and a 20 000-point
  Bickley–Naylor lookup table. Used by the existing ``cp_slab.py``
  / ``cp_cylinder.py`` derivations whose P-matrix constructions
  pre-date the Phase-0 reference-solution contract. They stay
  alive until the Phase-4 Peierls Nyström references replace them.

- **High-precision tier** (``e_n``, ``ki_n``, ``e_n_derivative``,
  ``ki_n_derivative``): mpmath-backed arbitrary-precision
  evaluators for :math:`E_n(x)` and the Bickley–Naylor
  :math:`\mathrm{Ki}_n(x)`. These satisfy the kernel identities

  .. math::

     E_n'(x) = -E_{n-1}(x), \qquad
     \mathrm{Ki}_n'(x) = -\mathrm{Ki}_{n-1}(x),

  and are verified term-by-term in ``tests/derivations/test_kernels.py``
  (L0). They are the primitives the Phase-4 Peierls/Bickley Nyström
  solvers will consume at 50 digits of working precision.

The two tiers must agree to double precision on their common domain;
this is itself a kernel-identity test (``test_legacy_matches_mpmath``).
"""

from __future__ import annotations

import functools
from typing import Callable

import mpmath
import numpy as np
from scipy.integrate import quad
from scipy.special import expn


# ═══════════════════════════════════════════════════════════════════════
# E₃ kernel (slab geometry)
# ═══════════════════════════════════════════════════════════════════════

def e3(x: float) -> float:
    """Third-order exponential integral E₃(x)."""
    return float(expn(3, max(x, 0.0)))


def e3_vec(x: np.ndarray) -> np.ndarray:
    """Vectorised E₃."""
    return expn(3, np.maximum(x, 0.0))


# ═══════════════════════════════════════════════════════════════════════
# Ki₃ / Ki₄ kernels (cylindrical geometry)
# ═══════════════════════════════════════════════════════════════════════

class BickleyTables:
    r"""Legacy tabulated Bickley–Naylor functions.

    .. warning::

       **Naming discrepancy with the Abramowitz & Stegun convention.**
       The :meth:`ki3` method integrates ``sin(t)·exp(-x/sin(t))``,
       which — under the substitution :math:`\theta' = \pi/2 - t` —
       equals the canonical :math:`\mathrm{Ki}_2(x)` of A&S 11.2,
       **not** :math:`\mathrm{Ki}_3(x)`. Similarly :meth:`ki4` returns
       an approximation to the canonical :math:`\mathrm{Ki}_3(x)` via
       a cumulative-sum integration with ~1e-3 absolute error.

       This is kept for backward compatibility with
       :mod:`orpheus.derivations.cp_cylinder` whose P-matrix
       construction was written against these legacy names. The
       Phase-4 Peierls cylinder reference (see the verification
       campaign plan) will replace these with the canonical
       :func:`ki_n` high-precision evaluator and retire this class.

       Tracked in GitHub Issue on the verification campaign. Do
       **not** rename silently — the CP formulas that consume this
       may be numerically self-consistent under the legacy naming
       and would break if the numbering were corrected in place.

    Definitions as implemented:

    .. math::

        \mathtt{ki3}(x) &= \int_0^{\pi/2} \sin(t)\,\exp(-x/\sin t)\,dt
            \;=\; \mathrm{Ki}_2^{\text{A\&S}}(x) \\
        \mathtt{ki4}(x) &\approx \int_x^{\infty} \mathtt{ki3}(t)\,dt
            \;\approx\; \mathrm{Ki}_3^{\text{A\&S}}(x)

    Tables are built once and cached.
    """

    def __init__(self, n_points: int = 20_000, x_max: float = 50.0):
        self.n_points = n_points
        self.x_max = x_max
        self._x = np.linspace(0, x_max, n_points)
        self._dx = self._x[1] - self._x[0]

        # Build Ki₃ table via numerical integration
        ki3 = np.empty(n_points)
        ki3[0] = 1.0
        for i in range(1, n_points):
            ki3[i], _ = quad(
                lambda t, xx=self._x[i]: np.exp(-xx / np.sin(t)) * np.sin(t),
                0, np.pi / 2,
            )
        self._ki3 = ki3

        # Ki₄ = cumulative integral of Ki₃ from x to infinity
        self._ki4 = np.cumsum(ki3[::-1])[::-1] * self._dx
        self._ki4[-1] = 0.0

    def ki3(self, x: float) -> float:
        """Evaluate Ki₃(x) by interpolation."""
        return float(np.interp(x, self._x, self._ki3, right=0.0))

    def ki4(self, x: float) -> float:
        """Evaluate Ki₄(x) by interpolation."""
        return float(np.interp(x, self._x, self._ki4, right=0.0))

    def ki3_vec(self, x: np.ndarray) -> np.ndarray:
        """Vectorised Ki₃ (legacy naming — actually canonical Ki₂)."""
        return np.interp(np.asarray(x, dtype=float), self._x, self._ki3, right=0.0)

    def ki4_vec(self, x: np.ndarray) -> np.ndarray:
        """Vectorised Ki₄ (legacy naming — actually canonical Ki₃)."""
        return np.interp(np.asarray(x, dtype=float), self._x, self._ki4, right=0.0)

    # Canonical A&S-named aliases (added for Phase 4.2, resolves #94)
    def Ki2(self, x: float) -> float:
        """Canonical :math:`\\mathrm{Ki}_2(x)` (= legacy ``ki3``)."""
        return self.ki3(x)

    def Ki3(self, x: float) -> float:
        """Canonical :math:`\\mathrm{Ki}_3(x)` (= legacy ``ki4``, ~1e-3 accuracy)."""
        return self.ki4(x)

    def Ki2_vec(self, x: np.ndarray) -> np.ndarray:
        """Vectorised canonical :math:`\\mathrm{Ki}_2` (= legacy ``ki3_vec``)."""
        return self.ki3_vec(x)

    def Ki3_vec(self, x: np.ndarray) -> np.ndarray:
        """Vectorised canonical :math:`\\mathrm{Ki}_3` (= legacy ``ki4_vec``)."""
        return self.ki4_vec(x)


@functools.lru_cache(maxsize=1)
def bickley_tables(
    n_points: int = 20_000, x_max: float = 50.0,
) -> BickleyTables:
    """Get (or build) the cached Bickley-Naylor lookup tables."""
    return BickleyTables(n_points, x_max)


# ═══════════════════════════════════════════════════════════════════════
# High-precision tier — mpmath-backed E_n and Ki_n
# ═══════════════════════════════════════════════════════════════════════
#
# These wrap :func:`mpmath.expint` (for :math:`E_n`) and
# :func:`mpmath.quad` (for :math:`\mathrm{Ki}_n`) at a caller-chosen
# working precision. They are the primitives consumed by the
# Phase-4 Peierls Nyström references; the Phase-0 kernel-identity
# tests exercise them directly.
#
# The Bickley function is defined as
#
#     Ki_n(x) = ∫_0^{π/2} cos^{n-1}(θ) exp(-x/cos θ) dθ
#
# (Bickley & Naylor 1935, Abramowitz & Stegun 11.2). The integrand
# is smooth on (0, π/2) but develops an exp(-∞) at θ = π/2 when x > 0
# and a mild endpoint behaviour at θ = 0 when x → 0. ``mpmath.quad``
# handles both correctly via adaptive Gauss–Legendre.


def e_n(n: int, x: float, precision_digits: int = 50) -> float:
    r"""Exponential integral :math:`E_n(x)` at arbitrary precision.

    Wraps :func:`mpmath.expint` and returns a Python ``float``
    rounded from the mpmath result at the requested working precision.
    For double-precision callers the default of 50 digits is
    massively over-specified but keeps all consumers consistent;
    Phase-4 Peierls callers request 50 explicitly and store the
    mpmath result unrounded via :func:`e_n_mp`.

    Parameters
    ----------
    n : int
        Order of the exponential integral. ``n >= 1``.
    x : float
        Argument. ``x >= 0``.
    precision_digits : int
        mpmath working precision in decimal digits.
    """
    with mpmath.workdps(precision_digits):
        return float(mpmath.expint(n, x))


def e_n_mp(n: int, x, precision_digits: int = 50):
    """Exponential integral :math:`E_n(x)` returning an mpmath ``mpf``.

    Use this when the caller will chain further mpmath arithmetic
    and must preserve precision.
    """
    with mpmath.workdps(precision_digits):
        return mpmath.expint(n, mpmath.mpf(x))


def e_n_derivative(n: int, x: float, precision_digits: int = 50) -> float:
    r"""Analytical derivative :math:`E_n'(x) = -E_{n-1}(x)`.

    For ``n == 1`` we fall back to the closed form
    :math:`E_1'(x) = -e^{-x}/x` since :math:`E_0(x) = e^{-x}/x`
    diverges at :math:`x = 0`.

    This function exists primarily as the RHS of the kernel-identity
    tests — the identity is *defined* at the mpmath layer, and the
    test verifies that the numerical derivative of ``e_n(n, x)``
    matches ``e_n_derivative(n, x)``.
    """
    if n < 1:
        raise ValueError(f"E_n derivative requires n >= 1, got {n}")
    if n == 1:
        with mpmath.workdps(precision_digits):
            if x == 0:
                raise ValueError("E_1'(0) diverges")
            return float(-mpmath.exp(-mpmath.mpf(x)) / mpmath.mpf(x))
    with mpmath.workdps(precision_digits):
        return float(-mpmath.expint(n - 1, x))


def _ki_integrand(n: int, x):
    r"""Integrand for :math:`\mathrm{Ki}_n(x)` on :math:`[0, \infty)`.

    Uses the substitution :math:`u = \tan\theta`, :math:`du = \sec^{2}\theta\,d\theta`,
    :math:`\cos\theta = 1/\sqrt{1+u^{2}}`, to map
    :math:`\theta \in [0, \pi/2]` to :math:`u \in [0, \infty)` and remove
    the essential singularity at :math:`\theta = \pi/2`. The transformed
    integrand

    .. math::

        \mathrm{Ki}_n(x) \;=\; \int_{0}^{\infty}
            (1+u^{2})^{-(n+1)/2}\,
            \exp\!\bigl(-x\,\sqrt{1+u^{2}}\bigr)\,du

    is smooth and bounded on :math:`[0, \infty)` for all :math:`x \ge 0`
    and decays like :math:`u^{-(n+1)}\,e^{-x u}` at infinity, which
    :func:`mpmath.quad` handles uniformly.
    """
    n_mp = mpmath.mpf(n)
    x_mp = mpmath.mpf(x)
    exponent = -(n_mp + 1)

    def integrand(u):
        s = mpmath.sqrt(1 + u * u)
        return s ** exponent * mpmath.exp(-x_mp * s)

    return integrand


def ki_n(n: int, x: float, precision_digits: int = 50) -> float:
    r"""Bickley–Naylor function :math:`\mathrm{Ki}_n(x)` at arbitrary precision.

    Canonical definition:

    .. math::

        \mathrm{Ki}_n(x) \;=\; \int_{0}^{\pi/2}
            \cos^{n-1}\theta\;
            \exp\!\left(-\tfrac{x}{\cos\theta}\right)\,d\theta.

    Internally evaluated via the substitution :math:`u = \tan\theta`,
    which removes the essential singularity at :math:`\theta = \pi/2`
    and gives a smooth, exponentially decaying integrand on
    :math:`[0, \infty)`. :func:`mpmath.quad` converges to the target
    precision for every :math:`(n, x)` with :math:`n \ge 1` and
    :math:`x \ge 0`.

    Cross-checked against the Wallis closed form
    :math:`\mathrm{Ki}_n(0)` (see :func:`ki_n_at_zero`) and against
    the legacy :class:`BickleyTables` at double precision in the
    L0 kernel-identity tests.

    Parameters
    ----------
    n : int
        Order. Must be :math:`\ge 1`.
    x : float
        Argument. :math:`x \ge 0`.
    precision_digits : int
        mpmath working precision in decimal digits.
    """
    if n < 1:
        raise ValueError(f"Ki_n requires n >= 1, got {n}")
    if x < 0:
        raise ValueError(f"Ki_n requires x >= 0, got {x}")
    with mpmath.workdps(precision_digits):
        val = mpmath.quad(_ki_integrand(n, x), [0, mpmath.inf])
        return float(val)


def ki_n_mp(n: int, x, precision_digits: int = 50):
    """Bickley–Naylor :math:`\\mathrm{Ki}_n` returning an mpmath ``mpf``."""
    if n < 1:
        raise ValueError(f"Ki_n requires n >= 1, got {n}")
    with mpmath.workdps(precision_digits):
        return mpmath.quad(_ki_integrand(n, float(x)), [0, mpmath.inf])


def ki_n_derivative(n: int, x: float, precision_digits: int = 50) -> float:
    r"""Analytical derivative :math:`\mathrm{Ki}_n'(x) = -\mathrm{Ki}_{n-1}(x)`.

    For :math:`n = 1` the limit :math:`\mathrm{Ki}_0(x) = K_0(x)`
    (modified Bessel function) is used via ``mpmath.besselk``.
    Reference: Abramowitz & Stegun 11.2.11.
    """
    if n < 1:
        raise ValueError(f"Ki_n' requires n >= 1, got {n}")
    if n == 1:
        # Ki_1'(x) = -Ki_0(x) = -K_0(x) (modified Bessel)
        with mpmath.workdps(precision_digits):
            if x == 0:
                raise ValueError("Ki_1'(0) diverges (K_0 has log singularity)")
            return float(-mpmath.besselk(0, x))
    return -ki_n(n - 1, x, precision_digits)


# ── Special values (used by kernel-identity tests) ───────────────────

def ki_n_at_zero(n: int) -> float:
    r"""Exact closed form :math:`\mathrm{Ki}_n(0)` for ``n >= 1``.

    Evaluates :math:`\int_0^{\pi/2} \cos^{n-1}\theta\,d\theta` in
    closed form via the Wallis integral. Used by kernel-identity
    tests (and as a sanity initialiser for the Bickley table).

    .. math::

        \mathrm{Ki}_1(0) = \tfrac{\pi}{2},\quad
        \mathrm{Ki}_2(0) = 1,\quad
        \mathrm{Ki}_3(0) = \tfrac{\pi}{4},\quad
        \mathrm{Ki}_4(0) = \tfrac{2}{3}.
    """
    with mpmath.workdps(50):
        # Wallis: ∫_0^{π/2} cos^k θ dθ for k = n-1
        k = n - 1
        if k == 0:
            return float(mpmath.pi / 2)
        if k % 2 == 0:  # even
            # (π/2) * (k-1)!! / k!!
            num = mpmath.mpf(1)
            den = mpmath.mpf(1)
            for i in range(1, k, 2):
                num *= i
            for i in range(2, k + 1, 2):
                den *= i
            return float(mpmath.pi / 2 * num / den)
        # odd k
        num = mpmath.mpf(1)
        den = mpmath.mpf(1)
        for i in range(2, k, 2):
            num *= i
        for i in range(1, k + 1, 2):
            den *= i
        return float(num / den)


def e_n_at_zero(n: int) -> float:
    r"""Exact closed form :math:`E_n(0) = 1/(n-1)` for :math:`n > 1`.

    :math:`E_1(0)` diverges logarithmically — excluded here.
    """
    if n <= 1:
        raise ValueError(f"E_n(0) requires n > 1 (E_1 diverges), got {n}")
    return 1.0 / (n - 1)
