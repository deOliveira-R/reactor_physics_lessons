r"""Quadrature contract and primitive constructors.

This module ships the single contract every 1-D quadrature in
ORPHEUS speaks:

- :class:`Quadrature1D` — a frozen value object carrying ``pts``,
  ``wts`` ndarrays, the integration interval, and the panel-bounds
  decomposition (for composite rules).
- :func:`gauss_legendre` — plain Gauss-Legendre on :math:`[a, b]`.
- :func:`gauss_legendre_visibility_cone` — GL on :math:`[a, b]`
  with a square-root endpoint singularity absorbed into the
  Jacobian (see :ref:`§22.7 <section-22-7-visibility-cone>`).
- :func:`composite_gauss_legendre` — concatenated plain-GL panels
  on a sorted breakpoint list.
- :func:`gauss_laguerre` — :math:`\int_0^\infty f(x)\,e^{-x/\sigma}
  \,\mathrm dx` rule on :math:`[0, \infty)`.
- :class:`AdaptiveQuadrature1D` + :func:`adaptive_mpmath` — adaptive
  (no-fixed-nodes) rule for verification-tier integrals where the
  consumer only needs the scalar; backed by :func:`mpmath.quad`.

Geometry-aware recipes (``chord_quadrature``,
``observer_angular_quadrature``) live in
:mod:`._quadrature_recipes` and compose the static primitives.

Design notes
------------
The contract is deliberately minimal: a frozen dataclass with an
ndarray-of-nodes, an ndarray-of-weights, the interval, and the
panel-bounds tuple. Consumers integrate by ``q.integrate(f)``
(callable evaluated at nodes, broadcast) or
``q.integrate_array(values)`` (precomputed values at nodes); both
are one-liners that erase the ``for k in range(n_quad)`` smell.

Composition is the ``|`` operator: ``q1 | q2`` concatenates
abutting panels, accumulating ``panel_bounds``. This replaces the
ad-hoc ``np.concatenate`` + parallel-list-of-tuples idiom that
two redundant composite-GL implementations carry today.

Precision dial: every constructor accepts ``dps=53`` (the float64
default). For ``dps > 53`` the nodes are computed via
``mpmath.gauss_quadrature`` at the requested precision and cast
to ``float64`` for the returned arrays — same pattern as the
legacy :func:`~orpheus.derivations.peierls_geometry.gl_float`.

See :doc:`/theory/peierls_unified` § "Quadrature contract"
(§22.0) for the design rationale and § "Coordinate transformations
in Nyström quadrature" (§22.1–§22.7) for the substitution catalogue
that the constructors implement.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import mpmath
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# The contract
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Quadrature1D:
    r"""A 1-D quadrature rule on a (possibly composite) interval.

    The contract: ``pts`` and ``wts`` are 1-D float64 ndarrays of
    equal length such that

    .. math::

        \int_a^b f(y)\,\mathrm dy
        \;\approx\; \sum_i \mathrm{wts}[i]\,f(\mathrm{pts}[i])

    whenever :math:`f` is in the family the rule was built for.
    Compositions of rules concatenate the arrays and the panel-bounds
    tuple, preserving the "an integral is a sum of panel integrals"
    semantics.

    Attributes
    ----------
    pts : np.ndarray, shape ``(n,)``, float64
        Quadrature nodes, strictly inside ``interval``.
    wts : np.ndarray, shape ``(n,)``, float64
        Positive quadrature weights.
    interval : tuple[float, float]
        ``(a, b)`` with ``a < b``. The full integration domain.
    panel_bounds : tuple[tuple[float, float], ...]
        Per-panel ``(a_k, b_k)`` boundaries. For a single-panel rule
        this is ``(interval,)``. Subdivisions push more entries in
        left-to-right order.
    """

    pts: np.ndarray
    wts: np.ndarray
    interval: tuple[float, float]
    panel_bounds: tuple[tuple[float, float], ...] = field(default=())
    panel_sizes: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        # Coerce to float64 ndarrays so callers can pass lists / mpmath.
        object.__setattr__(self, "pts", np.asarray(self.pts, dtype=float))
        object.__setattr__(self, "wts", np.asarray(self.wts, dtype=float))
        if self.pts.shape != self.wts.shape:
            raise ValueError(
                f"Quadrature1D: pts.shape {self.pts.shape} != "
                f"wts.shape {self.wts.shape}"
            )
        if self.pts.ndim != 1:
            raise ValueError(
                f"Quadrature1D: pts must be 1-D, got ndim={self.pts.ndim}"
            )
        a, b = self.interval
        if not a < b:
            raise ValueError(
                f"Quadrature1D: need interval[0] < interval[1], got {self.interval}"
            )
        # Default panel_bounds to the single full interval and place all
        # nodes into that single panel. Composite rules (and ``__or__``)
        # populate ``panel_sizes`` explicitly with one entry per panel.
        if not self.panel_bounds:
            object.__setattr__(self, "panel_bounds", ((float(a), float(b)),))
        if not self.panel_sizes:
            object.__setattr__(self, "panel_sizes", (int(self.pts.size),))
        if len(self.panel_sizes) != len(self.panel_bounds):
            raise ValueError(
                f"Quadrature1D: panel_sizes length "
                f"({len(self.panel_sizes)}) != panel_bounds length "
                f"({len(self.panel_bounds)})"
            )
        if int(np.sum(self.panel_sizes)) != int(self.pts.size):
            raise ValueError(
                f"Quadrature1D: sum(panel_sizes) "
                f"({int(np.sum(self.panel_sizes))}) != pts.size "
                f"({int(self.pts.size)})"
            )

    def __len__(self) -> int:
        return int(self.pts.size)

    def __iter__(self):
        # Iterate as ``(point, weight)`` pairs — never indexed.
        return zip(self.pts.tolist(), self.wts.tolist())

    @property
    def n_panels(self) -> int:
        """Number of sub-panels in this rule."""
        return len(self.panel_bounds)

    def panel_slice(self, k: int) -> slice:
        r"""Index range of the ``self.pts`` / ``self.wts`` entries that
        belong to panel :math:`k`.

        Composite rules concatenate per-panel arrays in left-to-right
        order, so ``self.pts[self.panel_slice(k)]`` returns exactly the
        nodes contributed by panel :math:`k` (whose bounds are
        ``self.panel_bounds[k]``). Consumers that need to evaluate
        per-panel basis functions (e.g.
        :func:`~orpheus.derivations.peierls_geometry.lagrange_basis_on_panels`)
        index through this method rather than recomputing offsets from
        an external per-panel node count.
        """
        if not 0 <= k < self.n_panels:
            raise IndexError(
                f"panel_slice: k={k} out of range [0, {self.n_panels})"
            )
        offsets = np.concatenate([[0], np.cumsum(self.panel_sizes)])
        return slice(int(offsets[k]), int(offsets[k + 1]))

    def integrate(
        self,
        f: Callable[[np.ndarray], np.ndarray],
    ) -> float:
        r"""Vectorised integration: evaluate ``f`` at the nodes and dot
        with the weights.

        ``f`` must accept a 1-D ndarray of length ``len(self)`` and return
        a 1-D ndarray of the same length. Use :meth:`integrate_array` if
        the values are already computed.
        """
        return float(np.dot(self.wts, f(self.pts)))

    def integrate_array(self, f_at_pts: np.ndarray) -> float:
        r"""Sum precomputed values against the weights.

        Use this when the integrand evaluation is non-trivial (e.g.
        per-node mpmath calls inside an outer loop): build the values
        ndarray however you like, then call ``q.integrate_array(values)``.
        """
        f_at_pts = np.asarray(f_at_pts, dtype=float)
        if f_at_pts.shape != self.pts.shape:
            raise ValueError(
                f"integrate_array: shape {f_at_pts.shape} != "
                f"pts.shape {self.pts.shape}"
            )
        return float(np.dot(self.wts, f_at_pts))

    def __or__(self, other: "Quadrature1D") -> "Quadrature1D":
        r"""Concatenate two abutting panels into a composite rule.

        Requires ``self.interval[1] == other.interval[0]`` (modulo
        floating-point slack). The resulting rule's ``panel_bounds``
        is the concatenation of the two operands'.
        """
        if not np.isclose(self.interval[1], other.interval[0], rtol=1e-12, atol=1e-14):
            raise ValueError(
                f"Quadrature1D.__or__: cannot concatenate {self.interval} | "
                f"{other.interval}: intervals do not abut "
                f"(gap = {other.interval[0] - self.interval[1]:.3e})"
            )
        return Quadrature1D(
            pts=np.concatenate([self.pts, other.pts]),
            wts=np.concatenate([self.wts, other.wts]),
            interval=(self.interval[0], other.interval[1]),
            panel_bounds=self.panel_bounds + other.panel_bounds,
            panel_sizes=self.panel_sizes + other.panel_sizes,
        )


# ═══════════════════════════════════════════════════════════════════════
# Primitive constructors
# ═══════════════════════════════════════════════════════════════════════

def _leggauss(n: int, dps: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Gauss-Legendre nodes/weights on :math:`[-1, 1]` at the requested
    decimal precision, returned as float64 ndarrays.

    For ``dps <= 53`` (i.e. double precision) we use
    :func:`numpy.polynomial.legendre.leggauss`. For higher precision
    we use :func:`mpmath.gauss_quadrature` and cast to float — same
    routing as the legacy :func:`peierls_geometry.gl_float`.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if dps <= 53:
        nodes, wts = np.polynomial.legendre.leggauss(n)
        return nodes.astype(float), wts.astype(float)
    with mpmath.workdps(dps):
        mp_nodes, mp_wts = mpmath.gauss_quadrature(n, "legendre")
    return (
        np.fromiter((float(x) for x in mp_nodes), dtype=float, count=n),
        np.fromiter((float(w) for w in mp_wts), dtype=float, count=n),
    )


def gauss_legendre(a: float, b: float, n: int, *, dps: int = 53) -> Quadrature1D:
    r"""Plain Gauss-Legendre on :math:`[a, b]`, exact for polynomials
    of degree :math:`\le 2n - 1`.

    The degenerate case of every other rule in this module: composite
    GL with one panel, vis-cone GL with no singularity, etc. The
    ``dps`` parameter selects the precision at which the nodes are
    computed; the returned arrays are always float64.
    """
    nodes, wts = _leggauss(n, dps)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return Quadrature1D(
        pts=half * nodes + mid,
        wts=half * wts,
        interval=(float(a), float(b)),
    )


def gauss_legendre_visibility_cone(
    a: float,
    b: float,
    n: int,
    *,
    singular_endpoint: str = "lower",
    dps: int = 53,
) -> Quadrature1D:
    r"""Gauss-Legendre on :math:`[a, b]` that absorbs a square-root
    endpoint singularity into the Jacobian.

    For an integrand carrying a single-endpoint factor of the form
    :math:`\sqrt{y^{2} - a^{2}}` (vanishing at the lower endpoint —
    the canonical visibility-cone pattern in :math:`\mu`-integrals
    where :math:`\mu_{\min} = \sqrt{1 - (r/R)^{2}}` bounds the
    cone), the substitution

    .. math::

        u^{2} \;=\; \frac{y^{2} - a^{2}}{b^{2} - a^{2}},
        \qquad
        y(u) \;=\; \sqrt{a^{2} + u^{2}\,(b^{2} - a^{2})}

    yields the global identity
    :math:`\sqrt{y^{2} - a^{2}} = u\,\sqrt{b^{2} - a^{2}}`, so the
    :math:`u`-factor in the Jacobian cancels the endpoint
    singularity and plain GL on :math:`u \in [0, 1]` becomes
    spectral. Pass ``singular_endpoint="upper"`` to instead absorb
    :math:`\sqrt{b^{2} - y^{2}}` (chord half-length pattern at
    :math:`y \to b`); the substitution is symmetric.

    See :ref:`§22.7 <section-22-7-visibility-cone>` in
    :doc:`/theory/peierls_unified` for the full derivation,
    Bernstein-ellipse convergence analysis, and gotchas (notably
    the upper-variant degeneracy at :math:`a = 0` and the lower-
    variant degeneracy when :math:`a \ll b`).
    """
    if b <= a:
        raise ValueError(f"Need b > a, got a={a}, b={b}")
    if a < 0:
        raise ValueError(f"a must be non-negative, got {a}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if singular_endpoint not in ("lower", "upper"):
        raise ValueError(
            f"singular_endpoint must be 'lower' or 'upper', "
            f"got {singular_endpoint!r}"
        )
    nodes, wts = _leggauss(n, dps)
    u = 0.5 * (nodes + 1.0)
    u_w = 0.5 * wts
    delta_sq = b * b - a * a
    if singular_endpoint == "lower":
        pts = np.sqrt(a * a + u * u * delta_sq)
    else:
        pts = np.sqrt(b * b - u * u * delta_sq)
    new_wts = u_w * u * delta_sq / pts
    return Quadrature1D(
        pts=pts,
        wts=new_wts,
        interval=(float(a), float(b)),
    )


def composite_gauss_legendre(
    breakpoints: Sequence[float],
    n_per_panel: int,
    *,
    dps: int = 53,
) -> Quadrature1D:
    r"""Concatenated plain Gauss-Legendre panels on the supplied
    breakpoint list.

    ``breakpoints`` must be a sorted sequence
    ``[a, b_1, b_2, ..., b_{N-1}, b]`` of length :math:`\ge 2`;
    the resulting rule integrates over
    :math:`[a, b] = [\mathrm{breakpoints}[0], \mathrm{breakpoints}[-1]]`
    with ``n_per_panel`` GL nodes per sub-panel. Subsumes
    :func:`peierls_geometry.composite_gl_r` and the private
    ``_composite_gauss_legendre`` in the CP solver — both are now
    thin wrappers (or call sites) of this constructor.

    For non-uniform per-panel ``n``, build each panel with
    :func:`gauss_legendre` directly and concatenate via the ``|``
    operator.
    """
    bps = np.asarray(breakpoints, dtype=float)
    if bps.ndim != 1 or bps.size < 2:
        raise ValueError(
            f"breakpoints must be 1-D with at least 2 entries, "
            f"got shape {bps.shape}"
        )
    if not np.all(np.diff(bps) > 0):
        raise ValueError(f"breakpoints must be strictly increasing, got {bps}")
    rules = [
        gauss_legendre(float(a), float(b), n_per_panel, dps=dps)
        for a, b in zip(bps[:-1], bps[1:])
    ]
    return _concat(rules)


def _concat(rules: Iterable["Quadrature1D"]) -> "Quadrature1D":
    """Reduce an iterable of abutting quadratures via ``|``."""
    rules = list(rules)
    if not rules:
        raise ValueError("Cannot concatenate an empty rule list")
    return functools.reduce(operator.or_, rules)


# ═══════════════════════════════════════════════════════════════════════
# Adaptive 1-D quadrature (no fixed nodes)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AdaptiveQuadrature1D:
    r"""Adaptive 1-D quadrature on :math:`[a, b]` with optional
    breakpoint hints.

    Unlike :class:`Quadrature1D`, an adaptive rule has *no* fixed
    nodes — the rule chooses its own subdivision at evaluation time
    (currently delegated to :func:`mpmath.quad`). The only public
    operation is :meth:`integrate`, which evaluates the integrand at
    rule-chosen points and returns a scalar. Consumers that need
    static nodes/weights for downstream basis evaluation must use
    :class:`Quadrature1D`; adaptive rules are appropriate when the
    integrand has well-known structural breakpoints (panel edges,
    tangent angles, log singularities) and the consumer only needs
    the final scalar.

    Attributes
    ----------
    interval : tuple of (float, float)
        ``(a, b)`` with ``a < b``.
    breakpoints : tuple of float
        Optional sorted breakpoints in :math:`(a, b)` passed as
        subdivision hints to the underlying adaptive engine. Use
        when the integrand has known kinks (e.g. tangent angles for
        chord-traversal integrals, panel boundaries for piecewise-
        polynomial bases).
    dps : int
        mpmath working precision (decimal digits).
    maxdegree : int or None
        Optional cap on the underlying engine's adaptive recursion
        depth. ``None`` defers to the engine's default.

    Notes
    -----
    See :class:`Quadrature1D` for the static-nodes counterpart.
    Both share the consumer-side ``integrate(f) -> float`` ergonomic
    so a function whose integration domain might be either static
    or adaptive can accept either type.
    """

    interval: tuple[float, float]
    breakpoints: tuple[float, ...] = field(default=())
    dps: int = 30
    maxdegree: int | None = None

    def __post_init__(self) -> None:
        a, b = self.interval
        if not a < b:
            raise ValueError(
                f"AdaptiveQuadrature1D: need interval[0] < interval[1], "
                f"got {self.interval}"
            )
        if self.dps < 1:
            raise ValueError(f"dps must be >= 1, got {self.dps}")
        for bp in self.breakpoints:
            if not (a < bp < b):
                raise ValueError(
                    f"AdaptiveQuadrature1D: breakpoint {bp} not in "
                    f"open interval ({a}, {b})"
                )
        if list(self.breakpoints) != sorted(self.breakpoints):
            raise ValueError(
                f"AdaptiveQuadrature1D: breakpoints must be sorted "
                f"ascending, got {self.breakpoints}"
            )

    def integrate(self, f: Callable[[float], float]) -> float:
        r"""Adaptively integrate ``f`` over :math:`[a, b]`.

        ``f`` must accept a *scalar* argument (mpmath number or float)
        and return a scalar — the underlying :func:`mpmath.quad`
        evaluates the integrand point-by-point at rule-chosen
        :math:`x` values. Returns the integral cast to ``float``.
        """
        with mpmath.workdps(self.dps):
            args = [self.interval[0], *self.breakpoints, self.interval[1]]
            kwargs = (
                {} if self.maxdegree is None
                else {"maxdegree": self.maxdegree}
            )
            return float(mpmath.quad(f, args, **kwargs))


def adaptive_mpmath(
    a: float,
    b: float,
    *,
    breakpoints: Sequence[float] = (),
    dps: int = 30,
    maxdegree: int | None = None,
) -> AdaptiveQuadrature1D:
    r"""Construct an :class:`AdaptiveQuadrature1D` backed by
    :func:`mpmath.quad`.

    Use this when:

    - The integrand has known structural breakpoints (panel edges,
      tangent angles, log singularities) but the consumer only needs
      the scalar integral, *not* fixed nodes for downstream basis
      evaluation.
    - High precision (``dps > 53``) is required for verification-tier
      computations where static GL would need too many nodes.
    - The integrand's regularity is unknown a priori — adaptive
      refinement converges where static GL might miss.

    For static-nodes consumers (Lagrange basis evaluation, downstream
    panel-aware operations) use :func:`composite_gauss_legendre` or
    one of the geometry-aware recipes in
    :mod:`._quadrature_recipes` instead — those return
    :class:`Quadrature1D` with full per-panel slicing metadata.
    """
    return AdaptiveQuadrature1D(
        interval=(float(a), float(b)),
        breakpoints=tuple(float(bp) for bp in breakpoints),
        dps=dps,
        maxdegree=maxdegree,
    )


def gauss_laguerre(
    n: int,
    *,
    scale: float = 1.0,
    dps: int = 53,
) -> Quadrature1D:
    r"""Gauss-Laguerre on :math:`[0, \infty)` with weight
    :math:`\mathrm e^{-x/\mathrm{scale}}` folded into the rule.

    For an integrand of the form :math:`f(x)\,\mathrm e^{-x/\sigma}`
    (e.g. exponentially-decaying kernel families like :math:`E_n`,
    Bickley :math:`\mathrm{Ki}_n` after a tanh substitution, or
    optical-depth coordinates), the rule is exact for
    :math:`f` polynomial of degree :math:`\le 2n-1` after the
    substitution :math:`u = x/\sigma`. The returned ``pts`` are in
    :math:`[0, \infty)`; the ``interval`` carries
    ``(0.0, np.inf)`` and ``panel_bounds`` likewise.

    Currently used in diagnostics; promoting to the contract keeps
    consumer code uniform when the optical-depth substitution
    (§22.3) lands.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if dps <= 53:
        nodes, wts = np.polynomial.laguerre.laggauss(n)
        nodes = nodes.astype(float)
        wts = wts.astype(float)
    else:
        with mpmath.workdps(dps):
            mp_nodes, mp_wts = mpmath.gauss_quadrature(n, "laguerre")
        nodes = np.fromiter(
            (float(x) for x in mp_nodes), dtype=float, count=n,
        )
        wts = np.fromiter(
            (float(w) for w in mp_wts), dtype=float, count=n,
        )
    return Quadrature1D(
        pts=scale * nodes,
        wts=scale * wts,
        interval=(0.0, float("inf")),
    )
