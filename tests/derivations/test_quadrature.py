r"""L0 term-verification tests for the unified Quadrature1D contract.

Pins the contract that
:mod:`orpheus.derivations._quadrature` and
:mod:`orpheus.derivations._quadrature_recipes` ship:

1. The :class:`Quadrature1D` value object (frozen dataclass with
   ``pts``, ``wts``, ``interval``, ``panel_bounds``, ``integrate``,
   ``integrate_array``, and ``|`` composition).
2. Primitive constructors: ``gauss_legendre``,
   ``gauss_legendre_visibility_cone``, ``composite_gauss_legendre``,
   ``gauss_laguerre``.
3. Geometry-aware recipes: ``chord_quadrature``,
   ``observer_angular_quadrature``.

The integral-correctness side of the substitution math
(spectral-vs-algebraic comparisons, Bernstein-ellipse gotchas) is
covered by the visibility-cone L0 tests in ``test_kernels.py``;
those tests now consume the new ``Quadrature1D`` return type but
their assertions on integral values are unchanged. The tests in
this file pin the **contract** itself.
"""

from __future__ import annotations

import math

import mpmath
import numpy as np
import pytest

from orpheus.derivations._quadrature import (
    AdaptiveQuadrature1D,
    Quadrature1D,
    adaptive_mpmath,
    composite_gauss_legendre,
    gauss_laguerre,
    gauss_legendre,
    gauss_legendre_visibility_cone,
)
from orpheus.derivations._quadrature_recipes import (
    chord_quadrature,
    observer_angular_quadrature,
)


# ═══════════════════════════════════════════════════════════════════════
# Quadrature1D contract — invariants of the value object
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_quadrature1d_basic_invariants():
    """Pts and wts are float64 ndarrays, interval validated, panel_bounds
    defaults to the single full interval."""
    q = gauss_legendre(0.0, 1.0, 5)
    assert q.pts.dtype == np.float64
    assert q.wts.dtype == np.float64
    assert q.pts.shape == q.wts.shape == (5,)
    assert q.interval == (0.0, 1.0)
    assert q.panel_bounds == ((0.0, 1.0),)
    assert len(q) == 5
    assert q.n_panels == 1


@pytest.mark.foundation
def test_quadrature1d_iter_yields_pt_wt_pairs():
    """``iter(q)`` yields ``(pt, wt)`` pairs — encourages zip-based
    consumption over indexed-loop access."""
    q = gauss_legendre(0.0, 1.0, 3)
    pairs = list(q)
    assert len(pairs) == 3
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    pts, wts = zip(*pairs)
    assert np.allclose(np.array(pts), q.pts)
    assert np.allclose(np.array(wts), q.wts)


@pytest.mark.foundation
def test_quadrature1d_rejects_invalid_construction():
    """Bad inputs raise ``ValueError`` rather than producing silent garbage."""
    with pytest.raises(ValueError, match=r"shape"):
        Quadrature1D(pts=np.array([1.0, 2.0]), wts=np.array([1.0]),
                     interval=(0.0, 1.0))
    with pytest.raises(ValueError, match=r"1-D"):
        Quadrature1D(pts=np.array([[1.0]]), wts=np.array([[1.0]]),
                     interval=(0.0, 1.0))
    with pytest.raises(ValueError, match=r"interval"):
        Quadrature1D(pts=np.array([0.5]), wts=np.array([1.0]),
                     interval=(1.0, 0.0))


@pytest.mark.foundation
def test_quadrature1d_concatenation_via_or_operator():
    """``q1 | q2`` concatenates abutting panels and accumulates
    ``panel_bounds`` and ``panel_sizes`` in left-to-right order."""
    q1 = gauss_legendre(0.0, 0.5, 4)
    q2 = gauss_legendre(0.5, 1.0, 6)
    q = q1 | q2
    assert len(q) == 10
    assert q.interval == (0.0, 1.0)
    assert q.panel_bounds == ((0.0, 0.5), (0.5, 1.0))
    assert q.panel_sizes == (4, 6)
    # The composite integrates correctly.
    assert q.integrate(lambda x: x) == pytest.approx(0.5, abs=1e-14)


@pytest.mark.foundation
def test_quadrature1d_panel_slice_indexes_nodes_correctly():
    """``q.panel_slice(k)`` selects exactly the nodes belonging to
    panel ``k`` — used by per-panel basis evaluators (Lagrange,
    spectral elements) to map a panel index to the right slice of
    ``q.pts`` / ``q.wts`` without recomputing offsets externally."""
    q1 = gauss_legendre(0.0, 0.5, 4)
    q2 = gauss_legendre(0.5, 1.0, 6)
    q3 = gauss_legendre(1.0, 2.0, 3)
    q = q1 | q2 | q3
    assert q.n_panels == 3
    assert q.panel_sizes == (4, 6, 3)

    # Panel 0: first 4 nodes, all in [0, 0.5].
    s0 = q.panel_slice(0)
    assert s0 == slice(0, 4)
    assert np.all((q.pts[s0] > 0.0) & (q.pts[s0] < 0.5))

    # Panel 1: next 6 nodes, all in [0.5, 1.0].
    s1 = q.panel_slice(1)
    assert s1 == slice(4, 10)
    assert np.all((q.pts[s1] > 0.5) & (q.pts[s1] < 1.0))

    # Panel 2: last 3 nodes, all in [1.0, 2.0].
    s2 = q.panel_slice(2)
    assert s2 == slice(10, 13)
    assert np.all((q.pts[s2] > 1.0) & (q.pts[s2] < 2.0))

    # Out-of-range panel index raises.
    with pytest.raises(IndexError):
        q.panel_slice(3)
    with pytest.raises(IndexError):
        q.panel_slice(-1)


@pytest.mark.foundation
def test_quadrature1d_single_panel_panel_slice_is_full_range():
    """Default single-panel rule has ``panel_slice(0) == slice(0, n)``
    — bridges scalar consumers to the per-panel API uniformly."""
    q = gauss_legendre(0.0, 1.0, 7)
    assert q.n_panels == 1
    assert q.panel_sizes == (7,)
    assert q.panel_slice(0) == slice(0, 7)


@pytest.mark.foundation
def test_quadrature1d_panel_sizes_validation():
    """Construction validates panel_sizes/panel_bounds length and that
    sizes sum to the node count."""
    pts = np.linspace(0.1, 0.9, 5)
    wts = np.ones(5) * 0.1
    # Bad: sum(panel_sizes) != n_pts.
    with pytest.raises(ValueError, match=r"sum\(panel_sizes\)"):
        Quadrature1D(
            pts=pts, wts=wts, interval=(0.0, 1.0),
            panel_bounds=((0.0, 0.5), (0.5, 1.0)),
            panel_sizes=(3, 3),
        )
    # Bad: panel_sizes length != panel_bounds length.
    with pytest.raises(ValueError, match=r"panel_sizes length"):
        Quadrature1D(
            pts=pts, wts=wts, interval=(0.0, 1.0),
            panel_bounds=((0.0, 0.5), (0.5, 1.0)),
            panel_sizes=(5,),
        )


@pytest.mark.foundation
def test_quadrature1d_or_rejects_non_abutting():
    """Non-abutting intervals raise ``ValueError`` — composition is not
    a generic concatenation."""
    q1 = gauss_legendre(0.0, 0.5, 4)
    q2 = gauss_legendre(0.6, 1.0, 4)
    with pytest.raises(ValueError, match=r"do not abut"):
        q1 | q2


@pytest.mark.foundation
def test_quadrature1d_integrate_vs_integrate_array_agree():
    """``q.integrate(f)`` (callable) and ``q.integrate_array(values)``
    (precomputed) must give identical results for the same data."""
    q = gauss_legendre(0.5, 2.5, 16)
    f_callable = lambda y: np.exp(-y * y)  # noqa: E731
    f_array = np.exp(-q.pts ** 2)
    assert q.integrate(f_callable) == pytest.approx(
        q.integrate_array(f_array), abs=0.0,  # bit-equal
    )


@pytest.mark.foundation
def test_quadrature1d_integrate_array_shape_check():
    """Wrong-shape arrays passed to ``integrate_array`` raise."""
    q = gauss_legendre(0.0, 1.0, 5)
    with pytest.raises(ValueError, match=r"shape"):
        q.integrate_array(np.zeros(3))


# ═══════════════════════════════════════════════════════════════════════
# Plain Gauss-Legendre — polynomial exactness
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
@pytest.mark.parametrize("n", [3, 5, 8])
def test_gauss_legendre_polynomial_exactness(n: int):
    r"""GL with ``n`` nodes integrates polynomials of degree
    :math:`\le 2n - 1` exactly (modulo machine precision).

    Tested by integrating :math:`\int_a^b x^k\,\mathrm dx` for
    :math:`k = 0 \ldots 2n-1` against the closed form
    :math:`(b^{k+1} - a^{k+1})/(k+1)`.
    """
    a, b = 1.0, 4.0
    q = gauss_legendre(a, b, n)
    for k in range(2 * n):
        truth = (b ** (k + 1) - a ** (k + 1)) / (k + 1)
        approx = q.integrate(lambda x, k=k: x ** k)
        assert approx == pytest.approx(truth, rel=1e-13, abs=1e-13), (
            f"n={n} failed at degree k={k}: got {approx}, want {truth}"
        )


@pytest.mark.foundation
def test_gauss_legendre_high_dps_consistent_with_float():
    """High-dps and default-dps results agree at machine precision."""
    q53 = gauss_legendre(1.0, 5.0, 16, dps=53)
    q100 = gauss_legendre(1.0, 5.0, 16, dps=100)
    np.testing.assert_allclose(q53.pts, q100.pts, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(q53.wts, q100.wts, rtol=1e-13, atol=1e-15)


# ═══════════════════════════════════════════════════════════════════════
# Composite Gauss-Legendre — subsumes composite_gl_r and CP duplicate
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_composite_gauss_legendre_partitions_correctly():
    """A composite rule's panel_bounds reflect the breakpoint list, and
    the integral splits into a sum over panels."""
    bps = [0.0, 1.0, 2.5, 5.0]
    q = composite_gauss_legendre(bps, 5)
    assert q.n_panels == 3
    assert q.panel_bounds == ((0.0, 1.0), (1.0, 2.5), (2.5, 5.0))
    assert q.interval == (0.0, 5.0)
    assert q.integrate(lambda x: np.ones_like(x)) == pytest.approx(5.0, abs=1e-13)


@pytest.mark.foundation
def test_composite_gauss_legendre_rejects_bad_breakpoints():
    with pytest.raises(ValueError):
        composite_gauss_legendre([1.0], 5)
    with pytest.raises(ValueError, match=r"strictly increasing"):
        composite_gauss_legendre([0.0, 1.0, 0.5, 2.0], 5)


@pytest.mark.foundation
def test_composite_gauss_legendre_matches_or_chain():
    """``composite_gauss_legendre([a,b,c,d], n)`` is bit-equivalent to
    ``gauss_legendre(a,b,n) | gauss_legendre(b,c,n) | gauss_legendre(c,d,n)``."""
    bps = [0.0, 1.0, 2.5, 5.0]
    q_composite = composite_gauss_legendre(bps, 6)
    q_chain = (
        gauss_legendre(0.0, 1.0, 6)
        | gauss_legendre(1.0, 2.5, 6)
        | gauss_legendre(2.5, 5.0, 6)
    )
    np.testing.assert_array_equal(q_composite.pts, q_chain.pts)
    np.testing.assert_array_equal(q_composite.wts, q_chain.wts)
    assert q_composite.panel_bounds == q_chain.panel_bounds


# ═══════════════════════════════════════════════════════════════════════
# Visibility-cone primitive — contract (math is in test_kernels.py)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_returns_quadrature1d():
    """The vis-cone constructor returns ``Quadrature1D`` and its
    integral on the canonical lower-endpoint test
    :math:`\\int_1^2 1/\\sqrt{y^2 - 1}\\,\\mathrm dy = \\mathrm{arccosh}(2)`
    is spectrally accurate at :math:`Q = 16`."""
    q = gauss_legendre_visibility_cone(1.0, 2.0, 16, singular_endpoint="lower")
    assert isinstance(q, Quadrature1D)
    assert q.interval == (1.0, 2.0)
    truth = math.acosh(2.0)
    val = q.integrate(lambda y: 1.0 / np.sqrt(y * y - 1.0))
    assert val == pytest.approx(truth, abs=1e-12)


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_input_validation():
    """Same validation surface the Phase 1A primitive shipped — a
    contract regression check."""
    with pytest.raises(ValueError, match=r"b > a"):
        gauss_legendre_visibility_cone(2.0, 1.0, 16)
    with pytest.raises(ValueError, match=r"non-negative"):
        gauss_legendre_visibility_cone(-0.1, 1.0, 16)
    with pytest.raises(ValueError, match=r"n must be"):
        gauss_legendre_visibility_cone(0.0, 1.0, 0)
    with pytest.raises(ValueError, match=r"singular_endpoint"):
        gauss_legendre_visibility_cone(1.0, 2.0, 16, singular_endpoint="middle")


# ═══════════════════════════════════════════════════════════════════════
# Adaptive (mpmath) quadrature — no fixed nodes, breakpoint-hint API
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_adaptive_mpmath_integrates_polynomial():
    r"""``adaptive_mpmath`` integrates :math:`\int_0^1 x^k\,\mathrm dx
    = 1/(k+1)` to mpmath precision via :func:`mpmath.quad`."""
    q = adaptive_mpmath(0.0, 1.0, dps=30)
    for k in range(6):
        truth = 1.0 / (k + 1)
        approx = q.integrate(lambda x, k=k: x ** k)
        assert approx == pytest.approx(truth, rel=1e-13, abs=1e-13)


@pytest.mark.foundation
def test_adaptive_mpmath_breakpoints_resolve_kink():
    r"""For an integrand with a known kink, supplying breakpoint hints
    lets mpmath.quad converge cleanly. Tests
    :math:`\int_0^2 |x - 1|\,\mathrm dx = 1` with the breakpoint at
    :math:`x = 1` (kink) and without."""
    f = lambda x: abs(float(x) - 1.0)  # noqa: E731
    truth = 1.0
    # With breakpoint: machine-precision convergence — mpmath splits
    # at the kink and integrates each smooth half exactly.
    q_hinted = adaptive_mpmath(0.0, 2.0, breakpoints=(1.0,), dps=20)
    assert q_hinted.integrate(f) == pytest.approx(truth, abs=1e-14)
    # Without breakpoint: mpmath misses the C¹ kink and saturates at
    # ~1e-5, demonstrating *why* the breakpoint-hint API matters.
    # This is the exact failure mode that justifies the existence of
    # the AdaptiveQuadrature1D contract: callers with structural
    # knowledge (panel edges, tangent angles) must pass it through.
    q_naive = adaptive_mpmath(0.0, 2.0, dps=20)
    naive_err = abs(q_naive.integrate(f) - truth)
    assert naive_err > 1e-7, (
        f"naive mpmath.quad on |x-1| converged unexpectedly "
        f"({naive_err:.3e}) — the breakpoint-hint argument is moot"
    )


@pytest.mark.foundation
def test_adaptive_mpmath_high_precision():
    r"""``dps`` truly raises the working precision: a smooth oscillatory
    integrand (Fresnel-like :math:`\int_0^{\sqrt{\pi}}\cos(x^2)\,
    \mathrm dx`) is converged at ``dps=50`` to 1e-15 even though the
    default float64 ``mpmath.quad`` would saturate earlier."""
    truth = float(mpmath.quad(
        lambda x: mpmath.cos(x * x), [0, mpmath.sqrt(mpmath.pi)],
    ))
    q = adaptive_mpmath(0.0, float(np.sqrt(np.pi)), dps=50)
    assert q.integrate(lambda x: mpmath.cos(x * x)) == pytest.approx(
        truth, abs=1e-14,
    )


@pytest.mark.foundation
def test_adaptive_mpmath_input_validation():
    """Bad inputs raise ``ValueError``."""
    with pytest.raises(ValueError, match=r"interval"):
        adaptive_mpmath(2.0, 1.0)
    with pytest.raises(ValueError, match=r"dps"):
        adaptive_mpmath(0.0, 1.0, dps=0)
    with pytest.raises(ValueError, match=r"breakpoint"):
        adaptive_mpmath(0.0, 1.0, breakpoints=(1.5,))  # outside (a, b)
    with pytest.raises(ValueError, match=r"sorted"):
        adaptive_mpmath(0.0, 1.0, breakpoints=(0.7, 0.3))


@pytest.mark.foundation
def test_adaptive_quadrature1d_is_distinct_type():
    """``AdaptiveQuadrature1D`` is a sibling of ``Quadrature1D``, not
    a subclass. Both expose ``integrate(f) -> float`` so a function
    that only needs the scalar can accept either, but they do NOT
    share static node fields (the adaptive type has no ``pts``)."""
    q_static = gauss_legendre(0.0, 1.0, 5)
    q_adaptive = adaptive_mpmath(0.0, 1.0, dps=20)
    assert isinstance(q_static, Quadrature1D)
    assert isinstance(q_adaptive, AdaptiveQuadrature1D)
    assert not isinstance(q_adaptive, Quadrature1D)
    # Adaptive type does not expose pts/wts (would defeat its purpose
    # — the rule chooses nodes at evaluation time).
    assert not hasattr(q_adaptive, "pts")
    assert not hasattr(q_adaptive, "wts")
    # Both integrate the same constant correctly.
    one = lambda x: 1.0  # noqa: E731
    assert q_static.integrate(lambda x: np.ones_like(x)) == pytest.approx(1.0)
    assert q_adaptive.integrate(one) == pytest.approx(1.0, abs=1e-13)


# ═══════════════════════════════════════════════════════════════════════
# Visibility-cone — substitution math (spectral vs algebraic)
# ═══════════════════════════════════════════════════════════════════════
#
# These tests pin the QUALITATIVE claim of the visibility-cone
# substitution: on integrands with a √-endpoint factor the rule is
# spectral, while plain GL plateaus at algebraic-convergence error;
# on smooth integrands it introduces no bias. Math derivation in
# :ref:`section-22-7-visibility-cone` of :doc:`/theory/peierls_unified`.

def _plain_gl_pts_wts(a: float, b: float, n: int):
    """Reference plain-GL nodes/weights — kept private to this file
    as the comparator for the spectral-vs-algebraic tests."""
    nodes, wts = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return half * nodes + mid, half * wts


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_lower_endpoint_spectral():
    r"""Lower variant absorbs :math:`\sqrt{y^{2}-a^{2}}` singularity.

    Integrand :math:`f(y) = 1/\sqrt{y^{2} - a^{2}}` on
    :math:`[a, b] = [1, 2]` has the closed form
    :math:`\int = \mathrm{arccosh}(b/a) = \mathrm{arccosh}(2)
    = \ln(2 + \sqrt{3})`.

    Vis-cone GL at :math:`Q = 16` must hit machine precision; plain
    GL at :math:`Q = 64` must still be at algebraic-convergence
    error :math:`\gtrsim 10^{-3}`.
    """
    a, b = 1.0, 2.0
    truth = math.acosh(b / a)

    q = gauss_legendre_visibility_cone(a, b, 16, singular_endpoint="lower")
    vis_q16 = q.integrate(lambda y: 1.0 / np.sqrt(y * y - a * a))
    assert vis_q16 == pytest.approx(truth, abs=1e-12), (
        f"vis-cone Q=16 not spectral: |error| = {abs(vis_q16 - truth):.3e}"
    )

    y_plain, w_plain = _plain_gl_pts_wts(a, b, 64)
    plain_q64 = float(np.sum(w_plain / np.sqrt(y_plain * y_plain - a * a)))
    plain_err = abs(plain_q64 - truth)
    # Plain GL on this integrand cannot reach 1e-3 even at Q=64 —
    # algebraic convergence at the singular endpoint dominates. If
    # this assertion ever weakens the integrand has lost its
    # singularity (caller bug).
    assert plain_err > 1e-3, (
        f"plain GL Q=64 unexpectedly accurate (err={plain_err:.3e}) — "
        f"is the integrand singularity actually present?"
    )


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_upper_endpoint_spectral():
    r"""Upper variant absorbs :math:`\sqrt{b^{2} - y^{2}}` singularity.

    Integrand :math:`f(y) = 1/\sqrt{b^{2} - y^{2}}` on
    :math:`[a, b] = [1, 2]` has the closed form
    :math:`\int = \arcsin(y/b)\big|_{a}^{b}
    = \pi/2 - \arcsin(0.5) = \pi/3`.
    """
    a, b = 1.0, 2.0
    truth = math.pi / 2 - math.asin(a / b)

    q = gauss_legendre_visibility_cone(a, b, 24, singular_endpoint="upper")
    vis_q24 = q.integrate(lambda y: 1.0 / np.sqrt(b * b - y * y))
    assert vis_q24 == pytest.approx(truth, abs=1e-12), (
        f"vis-cone (upper) Q=24 not spectral: "
        f"|error| = {abs(vis_q24 - truth):.3e}"
    )


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_offdiag_pair_pattern_spectral():
    r"""Phase 5 Round-3 motivating pattern: integrand
    :math:`f(y) = \sqrt{y^{2} - a^{2}}\,\mathrm e^{-y}` on
    :math:`[a, b] = [1, 3]`.

    Plain GL converges algebraically (:math:`Q^{-3/2}` from the
    :math:`\sqrt{y^{2} - a^{2}}` factor); vis-cone (lower) is
    spectral. Both compared against an mpmath ground truth at 50
    digits of precision.

    Acceptance: vis-cone at :math:`Q = 24` must beat plain GL at
    :math:`Q = 24` by at least three orders of magnitude.
    """
    a, b = 1.0, 3.0

    def f_np(y):
        return np.sqrt(y * y - a * a) * np.exp(-y)

    def f_mp(y):
        return mpmath.sqrt(y * y - a * a) * mpmath.exp(-y)

    with mpmath.workdps(50):
        truth = float(mpmath.quad(f_mp, [a, b]))

    q = gauss_legendre_visibility_cone(a, b, 24, singular_endpoint="lower")
    vis_val = q.integrate(f_np)
    vis_err = abs(vis_val - truth)

    y_plain, w_plain = _plain_gl_pts_wts(a, b, 24)
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
    must converge to the same value as the mpmath ground truth
    for both variants. Bias regression: the substitution must not
    perturb routine inputs.
    """
    a, b = 0.5, 2.5
    with mpmath.workdps(50):
        truth = float(mpmath.quad(
            lambda y: mpmath.exp(-y * y), [a, b],
        ))

    f = lambda y: np.exp(-y * y)  # noqa: E731
    q_lower = gauss_legendre_visibility_cone(a, b, 64, singular_endpoint="lower")
    q_upper = gauss_legendre_visibility_cone(a, b, 64, singular_endpoint="upper")
    assert q_lower.integrate(f) == pytest.approx(truth, abs=1e-12)
    assert q_upper.integrate(f) == pytest.approx(truth, abs=1e-12)


@pytest.mark.l0
@pytest.mark.verifies("gauss-legendre-visibility-cone")
def test_visibility_cone_constant_integrand_recovers_length():
    r"""Sanity: :math:`\int_a^b 1\,\mathrm dy = b - a` for both
    variants on a non-pathological interval.

    Uses :math:`[1, 2]` (representative of a chord-annulus shell
    with :math:`r_{k-1}/r_k = 0.5`) and :math:`Q = 24` per the
    Bernstein-ellipse argument in the primitive's docstring.
    """
    a, b = 1.0, 2.0
    truth = b - a
    one = lambda y: np.ones_like(y)  # noqa: E731
    for endpoint in ("lower", "upper"):
        q = gauss_legendre_visibility_cone(
            a, b, 24, singular_endpoint=endpoint,
        )
        val = q.integrate(one)
        assert val == pytest.approx(truth, abs=1e-12), (
            f"endpoint={endpoint}: |error| = {abs(val - truth):.3e}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Gauss-Laguerre — promoting a diagnostic-only primitive
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_gauss_laguerre_integrates_exponential_polynomial():
    r""":math:`\int_0^\infty x^k\,\mathrm e^{-x}\,\mathrm dx = k!`
    via Gauss-Laguerre with the ``e^{-x}`` weight folded in.

    ``gauss_laguerre(n)`` returns nodes/weights such that the user
    passes the *non-exponential* part of the integrand;
    ``q.integrate(lambda x: x**k)`` gives :math:`k!`.
    """
    q = gauss_laguerre(20, scale=1.0)
    for k in range(8):
        truth = math.factorial(k)
        val = q.integrate(lambda x, k=k: x ** k)
        assert val == pytest.approx(truth, rel=1e-12)


@pytest.mark.foundation
def test_gauss_laguerre_scale_dilates_weight():
    r""":math:`\int_0^\infty\mathrm e^{-x/\sigma}\,\mathrm dx = \sigma`
    — the ``scale`` parameter sets the decay length of the absorbed
    weight, demonstrated by integrating the constant ``1``."""
    for scale in (0.5, 1.0, 2.5):
        q = gauss_laguerre(16, scale=scale)
        assert q.integrate(lambda x: np.ones_like(x)) == pytest.approx(
            scale, rel=1e-12,
        )


# ═══════════════════════════════════════════════════════════════════════
# chord_quadrature recipe — h-space subdivision at shell radii
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_chord_quadrature_homogeneous_panel_structure():
    """For ``radii=[R]`` (homogeneous) with split_first_panel=True the
    rule has exactly two panels: ``[0, R/2]`` (plain GL) and
    ``[R/2, R]`` (vis-cone-upper)."""
    R = 5.0
    q = chord_quadrature(np.array([R]), 16, split_first_panel=True)
    assert q.interval == (0.0, R)
    assert q.n_panels == 2
    assert q.panel_bounds == ((0.0, 0.5 * R), (0.5 * R, R))


@pytest.mark.foundation
def test_chord_quadrature_multi_region_panel_structure():
    """For ``radii=[r_1, r_2, ..., R]`` the rule has
    ``len(radii) + 1`` panels with split-first: ``[0, r_1/2]``,
    ``[r_1/2, r_1]``, ``[r_1, r_2]``, ..., ``[r_{N-1}, R]``."""
    radii = np.array([2.5, 5.0])
    q = chord_quadrature(radii, 16)
    assert q.n_panels == 3
    assert q.panel_bounds == ((0.0, 1.25), (1.25, 2.5), (2.5, 5.0))

    radii = np.array([1.0, 2.5, 5.0])
    q = chord_quadrature(radii, 16)
    assert q.n_panels == 4
    assert q.panel_bounds == ((0.0, 0.5), (0.5, 1.0), (1.0, 2.5), (2.5, 5.0))


@pytest.mark.foundation
def test_chord_quadrature_recovers_constant_integral():
    r"""``∫_0^R 1 dh = R`` regardless of subdivision — basic consistency.

    Uses :math:`Q = 32` per panel: the vis-cone-upper rule on
    :math:`[r_{k-1}, r_k]` for the constant integrand has its branch
    point at :math:`u = r_k/\sqrt{r_k^2 - r_{k-1}^2}`, which is close
    to the unit interval for thin shells (e.g. :math:`r_{k-1}/r_k =
    0.5` gives branch point :math:`\approx 1.155`, distance 0.155).
    Bernstein-ellipse error :math:`\sim \rho^{-2N}` reaches machine
    precision around :math:`N = 24-32` on these intervals.
    """
    R = 5.0
    for radii in [
        np.array([R]),
        np.array([2.5, R]),
        np.array([1.0, 2.5, R]),
    ]:
        q = chord_quadrature(radii, 32)
        assert q.integrate(lambda h: np.ones_like(h)) == pytest.approx(
            R, abs=1e-12,
        )


@pytest.mark.foundation
def test_chord_quadrature_rejects_bad_radii():
    with pytest.raises(ValueError, match=r"1-D"):
        chord_quadrature(np.array([[5.0]]), 16)
    with pytest.raises(ValueError, match=r"positive"):
        chord_quadrature(np.array([-1.0, 5.0]), 16)
    with pytest.raises(ValueError, match=r"strictly increasing"):
        chord_quadrature(np.array([5.0, 2.5]), 16)


@pytest.mark.foundation
def test_chord_quadrature_split_disable_keeps_first_panel():
    """``split_first_panel=False`` puts a single panel on
    :math:`[0, r_1]`."""
    radii = np.array([2.5, 5.0])
    q_split = chord_quadrature(radii, 16, split_first_panel=True)
    q_nosplit = chord_quadrature(radii, 16, split_first_panel=False)
    assert q_split.n_panels == 3
    assert q_nosplit.n_panels == 2
    assert q_nosplit.panel_bounds == ((0.0, 2.5), (2.5, 5.0))


# ═══════════════════════════════════════════════════════════════════════
# observer_angular_quadrature recipe — tangent-angle subdivision
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
def test_observer_angular_quadrature_no_interior_shells_is_plain_gl():
    """When no shells are interior to the observer (or none have
    tangent angles in [omega_low, omega_high]), the rule degenerates
    to plain GL on the full interval — i.e. one panel."""
    q = observer_angular_quadrature(
        r_obs=5.0, omega_low=0.0, omega_high=np.pi,
        radii=np.array([5.0]),  # only the outer surface; no interior shells
        n_per_panel=16,
    )
    assert q.n_panels == 1
    assert q.panel_bounds == ((0.0, np.pi),)


@pytest.mark.foundation
def test_observer_angular_quadrature_inserts_tangent_breakpoints():
    """For ``r_obs = 3``, ``radii = [1, 2, 3, 5]``, the tangent angles
    are ``arcsin(1/3)`` and ``arcsin(2/3)`` (forward) plus their
    backward mirrors — four interior breakpoints, five panels on
    :math:`[0, \\pi]`. Shell ``r=3`` itself is the observer's shell
    (strict :math:`<` filter), and ``r=5`` is the outer surface
    (:math:`r_k \\ge r_{\\rm obs}`, smooth crossing, no tangency)."""
    q = observer_angular_quadrature(
        r_obs=3.0, omega_low=0.0, omega_high=np.pi,
        radii=np.array([1.0, 2.0, 3.0, 5.0]),
        n_per_panel=8,
    )
    assert q.n_panels == 5
    expected_inner = sorted([
        np.arcsin(1.0 / 3.0),
        np.arcsin(2.0 / 3.0),
        np.pi - np.arcsin(2.0 / 3.0),
        np.pi - np.arcsin(1.0 / 3.0),
    ])
    actual_inner = [pb[1] for pb in q.panel_bounds[:-1]]
    np.testing.assert_allclose(actual_inner, expected_inner, atol=1e-14)


@pytest.mark.foundation
def test_observer_angular_quadrature_filters_to_window():
    """Tangent angles outside ``[omega_low, omega_high]`` are dropped."""
    # r_obs = 3, shell r=2 has tangent at arcsin(2/3) ≈ 0.7297 forward
    # and π - 0.7297 ≈ 2.412 backward. With omega ∈ [0, 1.0], only the
    # forward tangent is inside.
    q = observer_angular_quadrature(
        r_obs=3.0, omega_low=0.0, omega_high=1.0,
        radii=np.array([2.0, 3.0]),
        n_per_panel=8,
    )
    assert q.n_panels == 2
    assert q.panel_bounds[0][1] == pytest.approx(np.arcsin(2.0 / 3.0))


@pytest.mark.foundation
def test_observer_angular_quadrature_integrates_constant():
    """``∫_a^b 1 dω = b - a`` regardless of subdivision."""
    q = observer_angular_quadrature(
        r_obs=3.0, omega_low=0.5, omega_high=2.5,
        radii=np.array([1.0, 2.0, 3.0]),
        n_per_panel=8,
    )
    assert q.integrate(lambda w: np.ones_like(w)) == pytest.approx(
        2.0, abs=1e-13,
    )


@pytest.mark.foundation
def test_observer_angular_quadrature_input_validation():
    with pytest.raises(ValueError, match=r"omega_high > omega_low"):
        observer_angular_quadrature(
            r_obs=3.0, omega_low=1.0, omega_high=0.5,
            radii=np.array([1.0]), n_per_panel=8,
        )
    with pytest.raises(ValueError, match=r"r_obs must be"):
        observer_angular_quadrature(
            r_obs=0.0, omega_low=0.0, omega_high=np.pi,
            radii=np.array([1.0]), n_per_panel=8,
        )


# ═══════════════════════════════════════════════════════════════════════
# Cross-recipe identity: chord_quadrature gives T_00 = P_ss agreement
# ═══════════════════════════════════════════════════════════════════════
#
# The whole point of the chord_quadrature recipe is that the algebraic
# identity ``T_00 = P_ss`` (sphere or cylinder) becomes trivial — same
# nodes, same weights, identical integrand — instead of holding only
# at finite-Q after two different quadratures converge separately.
# This test verifies the identity at machine precision for both
# homogeneous and multi-region cells, *before* the consumer-side
# migration in Q2 lands.

@pytest.mark.foundation
@pytest.mark.parametrize("radii_sigt", [
    (np.array([5.0]), np.array([0.5])),
    (np.array([2.5, 5.0]), np.array([0.6, 0.4])),
    (np.array([1.5, 5.0]), np.array([0.8, 0.3])),
])
def test_chord_quadrature_sphere_T00_equals_P_ss(radii_sigt):
    r"""For sphere, both
    :math:`T_{00} = 2\!\int_0^1\!\mu\,\mathrm e^{-\tau(\mu)}\,\mathrm d\mu`
    and
    :math:`P_{ss} = 2\!\int_0^1\!\mu\,\mathrm e^{-\tau(\mu)}\,\mathrm d\mu`
    rewrite in :math:`h`-space (with :math:`\mu = \sqrt{1-h^2/R^2}`,
    :math:`\mathrm dh = \cdots`) as
    :math:`(2/R^2)\!\int_0^R h\,\mathrm e^{-\tau(h)}\,\mathrm dh`.
    Same integrand, so same value when computed with the same
    ``chord_quadrature``."""
    from orpheus.derivations._kernels import chord_half_lengths

    radii, sig_t = radii_sigt
    R = float(radii[-1])
    q = chord_quadrature(radii, 16)
    chords = 2.0 * chord_half_lengths(radii, q.pts)  # full antipodal chord
    tau = sig_t @ chords
    integral = q.integrate_array(q.pts * np.exp(-tau))
    T00 = (2.0 / R ** 2) * integral
    P_ss = (2.0 / R ** 2) * integral  # by construction same expression

    assert T00 == P_ss  # bit-equal, not approximately
    # And both are sensible numerically:
    assert 0.0 < P_ss < 1.0
