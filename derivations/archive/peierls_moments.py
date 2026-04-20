r"""Closed-form polynomial moments of the Peierls kernel families.

For a polynomial source basis :math:`L_j(x')` of degree :math:`p` on a
panel, the unified Peierls K-matrix entry decomposes as

.. math::

   K[i, j] \;=\; \sum_{\text{ray segments } s}\;
     \big\langle \mathbf{a}_{j}^{(s)},\; \mathbf{M}^{(s)} \big\rangle

where :math:`\mathbf{a}_{j}^{(s)}` are the polynomial coefficients of
:math:`L_j` expressed in the local optical-depth coordinate :math:`u`,
and the moment vector is

.. math::

   \mathbf{M}^{(s)}_k \;=\; \int_{u_a}^{u_b} u^k\, \kappa_d(u)\,\mathrm d u,
   \qquad k = 0, 1, \dots, p,

with :math:`\kappa_d \in \{E_1,\,\mathrm{Ki}_1,\, e^{-u}\}` for slab,
cylinder, sphere respectively. Each moment family admits a **closed-form
recursion** in higher-order kernel values evaluated at panel endpoints —
no inner quadrature required.

This module computes the family of cumulative moments

.. math::

   J_k^{\kappa}(z) \;\equiv\; \int_0^z u^k\, \kappa(u)\,\mathrm d u,
   \qquad k = 0, 1, \dots, p_{\max},

in :func:`e_n_cumulative_moments` (slab :math:`E_1`),
:func:`ki_n_cumulative_moments` (cylinder :math:`\mathrm{Ki}_1`),
:func:`exp_cumulative_moments` (sphere :math:`e^{-u}`).

References
----------
- *Slab moments*: integration by parts using
  :math:`E_1'(u) = -e^{-u}/u`. Pure calculus (Abramowitz & Stegun §5.1.32);
  closed form

  .. math::

     J_k^{E_1}(z) \;=\; \frac{z^{k+1}\,E_1(z)}{k+1}
                       \;+\; \frac{\gamma(k+1, z)}{k+1}.

  Hébert, *Applied Reactor Physics* (3rd ed., 2020) §3.2-3.3 restates the
  polynomial-source slab CP using this recursion.

- *Cylinder moments*: from :math:`\mathrm{Ki}_n'=-\mathrm{Ki}_{n-1}` and
  :math:`\int\!\mathrm{Ki}_n\,\mathrm d u = -\mathrm{Ki}_{n+1}`, repeated
  integration by parts gives

  .. math::

     J_k^{\mathrm{Ki}_1}(z) \;=\; -\sum_{m=0}^{k}
        \frac{k!}{(k-m)!}\, z^{k-m}\, \mathrm{Ki}_{m+2}(z)
        \;+\; k!\,\mathrm{Ki}_{k+2}(0).

  Stamm'ler & Abbate (1983) Ch. 4-6 documents this for higher-order CP
  cylinder spatial expansions; Hébert (2020) §3.4-3.5 restates it.

- *Sphere moments*: closed form via lower incomplete gamma,

  .. math::

     J_k^{e^{-u}}(z) \;=\; \int_0^z u^k\,e^{-u}\,\mathrm d u
                       \;=\; \gamma(k+1, z).

Verification
------------
Each closed form is gated against :func:`mpmath.quad` to 1e-15 relative
in :mod:`tests.derivations.test_peierls_moments` (L0 term verification).
"""

from __future__ import annotations

import mpmath
import numpy as np

from ._kernels import ki_n_mp


def _ki_n_at_zero_mp(n: int):
    r"""Wallis closed form for :math:`\mathrm{Ki}_n(0)` returned as
    :class:`mpmath.mpf` at the current working precision.

    .. math::

       \mathrm{Ki}_n(0) \;=\; \int_0^{\pi/2}\cos^{n-1}\theta\,\mathrm d\theta
       \;=\; \frac{\sqrt{\pi}}{2}\,
              \frac{\Gamma(n/2)}{\Gamma((n+1)/2)}.

    The float-precision counterpart in
    :func:`~_kernels.ki_n_at_zero` is too coarse for the cylindrical
    moment recursion (which exhibits cancellation when ``z`` is small —
    the boundary term :math:`k!\,\mathrm{Ki}_{k+2}(0)` and the sum of
    Bickley evaluations at :math:`z` partially cancel, amplifying any
    boundary-term roundoff). Computed in-place at the caller's
    ``mpmath.workdps`` so cancellation never bites.
    """
    if n < 1:
        raise ValueError(f"Ki_n(0) requires n >= 1, got {n}")
    n_mp = mpmath.mpf(n)
    return mpmath.sqrt(mpmath.pi) / 2 * mpmath.gamma(n_mp / 2) / mpmath.gamma((n_mp + 1) / 2)


# ═══════════════════════════════════════════════════════════════════════
# Slab moments — J_k^{E_1}(z) = ∫_0^z u^k E_1(u) du
# ═══════════════════════════════════════════════════════════════════════

def e_n_cumulative_moments(z: float, k_max: int, dps: int = 30) -> np.ndarray:
    r"""Vector of cumulative slab moments :math:`J_k^{E_1}(z)` for
    :math:`k = 0, 1, \dots, k_{\max}`.

    .. math::

       J_k^{E_1}(z) \;=\; \int_0^z u^k\, E_1(u)\,\mathrm d u
                       \;=\; \frac{z^{k+1} E_1(z) + \gamma(k+1, z)}{k+1}

    where :math:`\gamma(k+1, z)` is the lower incomplete gamma function
    :math:`\int_0^z u^k e^{-u}\,\mathrm d u`.

    Closed form derived by integration by parts using
    :math:`E_1'(u) = -e^{-u}/u`. No quadrature.

    Parameters
    ----------
    z : float
        Upper limit of the moment integral; ``z >= 0``.
    k_max : int
        Highest moment index to return; result has ``k_max + 1`` entries.
    dps : int
        mpmath working precision (decimal digits) for :math:`E_1` and
        :math:`\gamma` evaluations.

    Returns
    -------
    np.ndarray, shape ``(k_max + 1,)``
        ``J[k]`` = :math:`J_k^{E_1}(z)` as a Python float.

    Notes
    -----
    For ``z = 0`` all moments vanish (the integration domain is
    degenerate). For very small ``z`` the closed form is numerically
    stable: :math:`E_1(z) \sim -\ln z - \gamma_E + O(z)` cancels with
    :math:`\gamma(k+1, z) \sim z^{k+1}/(k+1)` term-by-term, producing a
    finite limit :math:`J_k^{E_1}(z) \to z^{k+1} \cdot[-\ln z + O(1)]/(k+1)`.
    """
    if z < 0:
        raise ValueError(f"z must be non-negative, got {z}")
    if k_max < 0:
        raise ValueError(f"k_max must be non-negative, got {k_max}")
    if z == 0.0:
        return np.zeros(k_max + 1)

    out = np.empty(k_max + 1)
    with mpmath.workdps(dps):
        z_mp = mpmath.mpf(z)
        E1_z = mpmath.expint(1, z_mp)
        # Lower incomplete gamma γ(a, z) via mpmath.gammainc(a, 0, z)
        for k in range(k_max + 1):
            gamma_kp1 = mpmath.gammainc(k + 1, 0, z_mp)
            J_k = (z_mp ** (k + 1) * E1_z + gamma_kp1) / (k + 1)
            out[k] = float(J_k)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Cylinder moments — J_k^{Ki_1}(z) = ∫_0^z u^k Ki_1(u) du
# ═══════════════════════════════════════════════════════════════════════

def ki_n_cumulative_moments(z: float, k_max: int, dps: int = 30) -> np.ndarray:
    r"""Vector of cumulative cylinder moments :math:`J_k^{\mathrm{Ki}_1}(z)`
    for :math:`k = 0, 1, \dots, k_{\max}`.

    .. math::

       J_k^{\mathrm{Ki}_1}(z) \;=\; \int_0^z u^k\, \mathrm{Ki}_1(u)\,\mathrm d u
       \;=\; -\sum_{m=0}^{k}\frac{k!}{(k-m)!}\,z^{k-m}\,\mathrm{Ki}_{m+2}(z)
              \;+\; k!\,\mathrm{Ki}_{k+2}(0).

    Closed form from repeated integration by parts using the Bickley
    identities :math:`\mathrm{Ki}_n' = -\mathrm{Ki}_{n-1}` and
    :math:`\int\mathrm{Ki}_n\,\mathrm d u = -\mathrm{Ki}_{n+1}`. No
    quadrature inside the loop — only :math:`\mathrm{Ki}_n` evaluations
    at the single endpoint :math:`z` (plus a fixed table of
    :math:`\mathrm{Ki}_n(0)` constants).

    Parameters
    ----------
    z : float
        Upper limit; ``z >= 0``.
    k_max : int
        Highest moment index; result has ``k_max + 1`` entries.
    dps : int
        mpmath working precision for :math:`\mathrm{Ki}_n(z)` evaluations.

    Returns
    -------
    np.ndarray, shape ``(k_max + 1,)``
        ``J[k]`` = :math:`J_k^{\mathrm{Ki}_1}(z)`.

    Notes
    -----
    Requires :math:`\mathrm{Ki}_n(z)` for :math:`n = 2, 3, \dots, k_{\max}+2`.
    These are computed once via :func:`~_kernels.ki_n_mp` and reused by all
    moment indices, so the per-call cost is dominated by ``k_max + 1``
    Bickley evaluations at a single argument.
    """
    if z < 0:
        raise ValueError(f"z must be non-negative, got {z}")
    if k_max < 0:
        raise ValueError(f"k_max must be non-negative, got {k_max}")
    if z == 0.0:
        return np.zeros(k_max + 1)

    out = np.empty(k_max + 1)
    with mpmath.workdps(dps):
        z_mp = mpmath.mpf(z)
        # Pre-compute Ki_n(z) for n = 2..k_max+2 (one mpmath.quad each).
        ki_at_z = {n: ki_n_mp(n, z_mp, dps) for n in range(2, k_max + 3)}
        # Pre-compute Ki_n(0) constants for n = 2..k_max+2 via Wallis form
        # (mpmath-native at full working precision — float-precision
        # constants would limit cancellation at small z to ~1e-9).
        ki_at_zero = {n: _ki_n_at_zero_mp(n) for n in range(2, k_max + 3)}
        # Evaluate the closed form
        #   J_k = -Σ_{m=0}^{k} [k!/(k-m)!] z^{k-m} Ki_{m+2}(z) + k!·Ki_{k+2}(0).
        # Reindex the sum with j = k - m (z power), so the coefficient
        # k!/(k-m)! = k!/j! and Ki index = m + 2 = (k - j) + 2:
        #   J_k = -Σ_{j=0}^{k} [k!/j!] z^{j} Ki_{k-j+2}(z) + k!·Ki_{k+2}(0).
        # Build coefficient k!/j! incrementally: coeff[0] = k!, coeff[j+1] = coeff[j]/(j+1).
        factorial_k = mpmath.mpf(1)
        for k in range(k_max + 1):
            if k > 0:
                factorial_k *= k  # k! built incrementally
            sum_term = mpmath.mpf(0)
            coeff = factorial_k  # k!/0!
            z_pow_j = mpmath.mpf(1)  # z^0
            for j in range(0, k + 1):
                sum_term += coeff * z_pow_j * ki_at_z[k - j + 2]
                if j < k:
                    coeff = coeff / (j + 1)
                    z_pow_j = z_pow_j * z_mp
            J_k = -sum_term + factorial_k * ki_at_zero[k + 2]
            out[k] = float(J_k)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Sphere moments — J_k^{exp}(z) = ∫_0^z u^k e^{-u} du = γ(k+1, z)
# ═══════════════════════════════════════════════════════════════════════

def exp_cumulative_moments(z: float, k_max: int, dps: int = 30) -> np.ndarray:
    r"""Vector of cumulative sphere moments :math:`J_k^{e^{-u}}(z)` for
    :math:`k = 0, 1, \dots, k_{\max}`.

    .. math::

       J_k^{e^{-u}}(z) \;=\; \int_0^z u^k\, e^{-u}\,\mathrm d u
                       \;=\; \gamma(k+1, z)

    where :math:`\gamma(a, z)` is the lower incomplete gamma function.
    Closed form; no quadrature.
    """
    if z < 0:
        raise ValueError(f"z must be non-negative, got {z}")
    if k_max < 0:
        raise ValueError(f"k_max must be non-negative, got {k_max}")
    if z == 0.0:
        return np.zeros(k_max + 1)
    out = np.empty(k_max + 1)
    with mpmath.workdps(dps):
        z_mp = mpmath.mpf(z)
        for k in range(k_max + 1):
            out[k] = float(mpmath.gammainc(k + 1, 0, z_mp))
    return out


# ═══════════════════════════════════════════════════════════════════════
# Segment moments — used directly by K-matrix assembly
# ═══════════════════════════════════════════════════════════════════════

def slab_segment_moments(
    u_a: float, u_b: float, k_max: int, dps: int = 30,
) -> np.ndarray:
    r"""Slab moments over an arbitrary :math:`u`-segment :math:`[u_a, u_b]`:

    .. math::

       \int_{u_a}^{u_b} u^k\, E_1(u)\,\mathrm d u
       \;=\; J_k^{E_1}(u_b) - J_k^{E_1}(u_a),
       \qquad k = 0, 1, \dots, k_{\max}.

    Returns the difference of two cumulative-moment vectors as floats;
    intended as the per-segment moment vector consumed by the slab
    moment-form K assembly. For downstream linear-solve workflows that
    chain mpmath arithmetic, see :func:`slab_segment_moments_mp`.
    """
    return e_n_cumulative_moments(u_b, k_max, dps) - e_n_cumulative_moments(u_a, k_max, dps)


def slab_segment_moments_mp(
    u_a: float, u_b: float, k_max: int, dps: int = 30,
) -> list:
    r"""mpmath-native variant of :func:`slab_segment_moments`.

    Returns a list of :class:`mpmath.mpf` values, NOT cast to float.
    The K-matrix assembly chains these into an mpmath linear solve to
    avoid Vandermonde conditioning loss when panel nodes are clustered
    in optical depth (high-:math:`u` distant panels).
    """
    if u_b < u_a:
        raise ValueError(f"u_b ({u_b}) must be >= u_a ({u_a})")
    out: list = []
    with mpmath.workdps(dps):
        u_a_mp = mpmath.mpf(u_a)
        u_b_mp = mpmath.mpf(u_b)
        E1_a = mpmath.expint(1, u_a_mp) if u_a > 0.0 else None
        E1_b = mpmath.expint(1, u_b_mp)
        for k in range(k_max + 1):
            gamma_b = mpmath.gammainc(k + 1, 0, u_b_mp)
            J_b = (u_b_mp ** (k + 1) * E1_b + gamma_b) / (k + 1)
            if u_a > 0.0:
                gamma_a = mpmath.gammainc(k + 1, 0, u_a_mp)
                J_a = (u_a_mp ** (k + 1) * E1_a + gamma_a) / (k + 1)
                out.append(J_b - J_a)
            else:
                out.append(J_b)
    return out


def cylinder_segment_moments(
    u_a: float, u_b: float, k_max: int, dps: int = 30,
) -> np.ndarray:
    """Cylinder per-segment moments — see :func:`slab_segment_moments`."""
    return ki_n_cumulative_moments(u_b, k_max, dps) - ki_n_cumulative_moments(u_a, k_max, dps)


def cylinder_segment_moments_mp(
    u_a: float, u_b: float, k_max: int, dps: int = 30,
) -> list:
    r"""mpmath-native variant of :func:`cylinder_segment_moments`.

    Returns a list of :class:`mpmath.mpf` values, NOT cast to float.
    Use this when chaining into an mpmath linear solve to avoid
    Vandermonde conditioning loss in the K-matrix assembly.

    Pre-computes :math:`\mathrm{Ki}_n(u_a)` and :math:`\mathrm{Ki}_n(u_b)`
    once per call and reuses across moment indices.
    """
    if u_b < u_a:
        raise ValueError(f"u_b ({u_b}) must be >= u_a ({u_a})")
    out: list = []
    with mpmath.workdps(dps):
        u_a_mp = mpmath.mpf(u_a)
        u_b_mp = mpmath.mpf(u_b)
        # Ki_n(u_a), Ki_n(u_b) for n = 2..k_max+2.
        ki_at_a = {n: ki_n_mp(n, u_a_mp, dps) if u_a > 0.0 else _ki_n_at_zero_mp(n)
                   for n in range(2, k_max + 3)}
        ki_at_b = {n: ki_n_mp(n, u_b_mp, dps) for n in range(2, k_max + 3)}
        ki_at_zero = {n: _ki_n_at_zero_mp(n) for n in range(2, k_max + 3)}

        factorial_k = mpmath.mpf(1)
        for k in range(k_max + 1):
            if k > 0:
                factorial_k *= k

            def closed_form(z_mp, ki_at_z):
                """Σ-form J_k^{Ki}(z) at single z."""
                s = mpmath.mpf(0)
                coeff = factorial_k  # k!/0!
                z_pow = mpmath.mpf(1)  # z^0
                for j in range(0, k + 1):
                    s += coeff * z_pow * ki_at_z[k - j + 2]
                    if j < k:
                        coeff = coeff / (j + 1)
                        z_pow = z_pow * z_mp
                return -s + factorial_k * ki_at_zero[k + 2]

            J_b = closed_form(u_b_mp, ki_at_b)
            J_a = closed_form(u_a_mp, ki_at_a) if u_a > 0.0 else mpmath.mpf(0)
            out.append(J_b - J_a)
    return out


def sphere_segment_moments(
    u_a: float, u_b: float, k_max: int, dps: int = 30,
) -> np.ndarray:
    """Sphere per-segment moments — see :func:`slab_segment_moments`."""
    return exp_cumulative_moments(u_b, k_max, dps) - exp_cumulative_moments(u_a, k_max, dps)


def sphere_segment_moments_mp(
    u_a: float, u_b: float, k_max: int, dps: int = 30,
) -> list:
    r"""mpmath-native variant of :func:`sphere_segment_moments`.

    Returns a list of :class:`mpmath.mpf` values; closed form via
    lower incomplete gamma differences.
    """
    if u_b < u_a:
        raise ValueError(f"u_b ({u_b}) must be >= u_a ({u_a})")
    out: list = []
    with mpmath.workdps(dps):
        u_a_mp = mpmath.mpf(u_a)
        u_b_mp = mpmath.mpf(u_b)
        for k in range(k_max + 1):
            gamma_b = mpmath.gammainc(k + 1, 0, u_b_mp)
            if u_a > 0.0:
                gamma_a = mpmath.gammainc(k + 1, 0, u_a_mp)
                out.append(gamma_b - gamma_a)
            else:
                out.append(gamma_b)
    return out
