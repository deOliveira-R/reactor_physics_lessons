r"""Geometry-aware quadrature recipes.

Two recipes that compose the primitive constructors in
:mod:`._quadrature` into the recurring ORPHEUS chord-and-angular
patterns:

- :func:`chord_quadrature` — surface-to-surface integrals on the
  impact parameter :math:`h \in [0, R]` for concentric annular
  geometries (sphere/cylinder). Subdivides at the interior shell
  radii and applies the visibility-cone substitution per panel; the
  first panel is split at :math:`r_1/2` to dodge the upper-variant
  degeneracy at :math:`h = 0`. This single recipe replaces both of
  the per-coordinate panel builders that an earlier rollout shipped
  (see :ref:`§22.7 <section-22-7-visibility-cone>` and the rollout history
  for the µ/α-space duplication that this consolidates).

- :func:`observer_angular_quadrature` — observer-centred ray sweeps
  in :math:`\omega \in [\omega_{\min}, \omega_{\max}]` from an
  internal observer at radius :math:`r_{\rm obs}`. Subdivides at the
  tangent angles :math:`\omega_{k} = \arcsin(r_{k}/r_{\rm obs})`
  (and their backward mirrors) for each interior shell visible from
  the observer. Plain Gauss-Legendre per sub-panel; this is the
  kink-aware quadrature that today lives only in
  :func:`~orpheus.derivations.peierls_geometry.build_volume_kernel`,
  promoted to a primitive so the per-face mode primitives
  (``compute_P_esc_*_mode``, ``compute_G_bc_*_mode``) can inherit
  it for free instead of each open-coding a smaller version.

- :func:`surface_centred_angular_quadrature` — surface-centred
  :math:`\phi`-sweep with the chord
  :math:`d^{2}(\phi) = r_{\rm obs}^{2} + r_{\rm surf}^{2}
  - 2\,r_{\rm obs}\,r_{\rm surf}\cos\phi`. Subdivides at the tangent
  angles where the chord-quadratic discriminant
  :math:`(r_{\rm obs}^{2} - r_{k}^{2})(r_{\rm surf}^{2} - r_{k}^{2})`
  is positive — i.e., for shells with
  :math:`r_{k} < \min(r_{\rm obs}, r_{\rm surf})`. Used by the legacy
  cylinder :math:`G_{\rm bc}^{\rm cyl}` :math:`\mathrm{Ki}_{1}/d`
  branches retained for backward compatibility with the rank-1 Mark
  closure tests.

All three recipes return a :class:`~._quadrature.Quadrature1D` so
consumers integrate via ``q.integrate(f)`` (or
``q.integrate_array(values)``) without ever indexing the node
array.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ._quadrature import (
    Quadrature1D,
    _concat,
    composite_gauss_legendre,
    gauss_legendre,
    gauss_legendre_visibility_cone,
)


# ═══════════════════════════════════════════════════════════════════════
# Recipe 1: chord-impact-parameter quadrature
# ═══════════════════════════════════════════════════════════════════════

def chord_quadrature(
    radii: np.ndarray,
    n_per_panel: int,
    *,
    split_first_panel: bool = True,
    dps: int = 53,
) -> Quadrature1D:
    r"""Impact-parameter quadrature on :math:`h \in [0, R]` for
    surface-to-surface chord integrals on concentric annular
    geometries.

    The outer radius :math:`R` is taken as ``radii[-1]``. For an
    integrand of the form :math:`g(h)\,\mathrm e^{-\tau(h)}` (or
    :math:`g(h)\,\mathrm{Ki}_{n}(\tau(h))`) where :math:`\tau(h)` is
    the antipodal-chord optical depth carrying
    :math:`\sqrt{r_{k}^{2} - h^{2}}` chord segments per shell, the
    integrand has :math:`\sqrt{r_{k} - h}` derivative singularities
    at every interior shell radius :math:`r_{k}`. The recipe
    subdivides :math:`[0, R]` at those :math:`r_{k}` and applies the
    visibility-cone substitution
    (:func:`~._quadrature.gauss_legendre_visibility_cone`,
    ``singular_endpoint="upper"``) per panel, absorbing the
    :math:`\sqrt{r_{k}^{2}-h^{2}}` factor into the substitution
    Jacobian.

    The leftmost panel :math:`[0, r_{1}]` has :math:`y_{\min} = 0`
    where the upper variant would introduce a spurious
    :math:`1/y(u)` singularity at :math:`u = 1`; with
    ``split_first_panel=True`` (the default) the panel is split at
    :math:`r_{1}/2`, putting plain Gauss-Legendre on the smooth half
    :math:`[0, r_{1}/2]` and vis-cone-upper on
    :math:`[r_{1}/2, r_{1}]` (where :math:`y_{\min} > 0`), recovering
    spectral convergence everywhere.

    Sphere and cylinder T-matrix / :math:`P_{ss}` integrals both
    reduce to a single :func:`chord_quadrature` call after rewriting
    in :math:`h`-space (the angular variable :math:`\mu` for sphere
    or :math:`\alpha` for cylinder is a parametrization, not a
    fundamental coordinate). The algebraic identity
    :math:`T_{00} = P_{ss}` becomes trivial — same nodes, same
    weights, identical integrand — instead of holding only
    approximately at finite :math:`Q`.

    Parameters
    ----------
    radii : np.ndarray, shape ``(N,)``
        Outer radii of the concentric shells, strictly increasing.
        ``radii[0] = r_1`` is the innermost shell, ``radii[-1] = R``
        is the outer surface.
    n_per_panel : int
        Number of Gauss-Legendre nodes per sub-panel. Total nodes
        :math:`= n_{\rm pp} \cdot (\mathrm{len}(\mathrm{radii}) + 1)`
        with ``split_first_panel=True``.
    split_first_panel : bool, keyword-only
        If ``True`` (default), splits the leftmost panel
        :math:`[0, r_{1}]` at :math:`r_{1}/2` to recover spectral
        convergence on the half adjacent to the singular endpoint.
        Set to ``False`` only if the consumer's integrand has a
        compensating :math:`1/h` factor that cancels the upper-variant
        Jacobian singularity at :math:`h = 0`.
    dps : int, keyword-only
        Decimal precision for the underlying GL node computation.
        Default ``53`` (float64); pass higher to use mpmath.

    Returns
    -------
    Quadrature1D
        Composite rule on :math:`[0, R]` with ``panel_bounds``
        recording the subdivision.

    See Also
    --------
    :ref:`§22.7 <section-22-7-visibility-cone>` — the substitution math.
    :func:`~._quadrature.gauss_legendre_visibility_cone` — the primitive
    constructor used per panel.
    """
    radii = np.asarray(radii, dtype=float)
    if radii.ndim != 1 or radii.size < 1:
        raise ValueError(
            f"radii must be 1-D with at least 1 entry, got shape {radii.shape}"
        )
    if not np.all(radii > 0):
        raise ValueError(f"radii must all be positive, got {radii}")
    if not np.all(np.diff(radii) > 0):
        raise ValueError(f"radii must be strictly increasing, got {radii}")

    r_1 = float(radii[0])

    panels: list[Quadrature1D] = []
    if split_first_panel:
        # [0, r_1/2]: smooth integrand (kink at r_1 is at upper boundary
        # of the next panel, not interior here) — plain GL is spectral.
        panels.append(gauss_legendre(0.0, 0.5 * r_1, n_per_panel, dps=dps))
        # [r_1/2, r_1]: y_min > 0 — vis-cone-upper absorbs √(r_1²-h²).
        panels.append(gauss_legendre_visibility_cone(
            0.5 * r_1, r_1, n_per_panel,
            singular_endpoint="upper", dps=dps,
        ))
    else:
        # Single first panel — caller takes responsibility for the
        # 1/h-or-equivalent compensation at h=0.
        panels.append(gauss_legendre_visibility_cone(
            0.0, r_1, n_per_panel,
            singular_endpoint="upper", dps=dps,
        ))

    # Interior + last shells: vis-cone-upper on each [r_{k-1}, r_k].
    panels.extend(
        gauss_legendre_visibility_cone(
            float(r_prev), float(r_next), n_per_panel,
            singular_endpoint="upper", dps=dps,
        )
        for r_prev, r_next in zip(radii[:-1], radii[1:])
    )

    return _concat(panels)


# ═══════════════════════════════════════════════════════════════════════
# Recipe 2: observer-centred angular quadrature
# ═══════════════════════════════════════════════════════════════════════

def observer_angular_quadrature(
    r_obs: float,
    omega_low: float,
    omega_high: float,
    radii: np.ndarray,
    n_per_panel: int,
    *,
    dps: int = 53,
) -> Quadrature1D:
    r"""Observer-centred :math:`\omega`-quadrature on
    :math:`[\omega_{\min}, \omega_{\max}]` with kink-aware subdivision
    at the tangent angles
    :math:`\omega_{k} = \arcsin(r_{k}/r_{\rm obs})` for each interior
    shell radius :math:`r_{k} < r_{\rm obs}` (and their backward
    mirrors :math:`\pi - \omega_{k}` for shells on the far side of
    the observer along the ray). Plain Gauss-Legendre per sub-panel.

    For a ray emitted from an observer at radial position
    :math:`r_{\rm obs}` in direction :math:`\omega` (measured from
    the radial outward), the impact parameter is
    :math:`h = r_{\rm obs}\,|\sin\omega|`. The chord penetrates
    shell :math:`r_{k}` iff :math:`h < r_{k}`; the boundary case
    :math:`h = r_{k}` defines the tangent angle. As :math:`\omega`
    crosses :math:`\omega_{k}` the chord-segment in shell :math:`k`
    appears (or disappears) with a :math:`\sqrt{|\omega - \omega_{k}|}`
    profile, giving :math:`\tau(\omega)` a derivative singularity at
    :math:`\omega_{k}`. Without subdivision, plain GL on
    :math:`[\omega_{\min}, \omega_{\max}]` resolves these kinks only
    algebraically (:math:`\mathcal O(Q^{-3/2})`); subdividing at the
    tangent angles isolates each kink as a panel boundary, recovering
    spectral convergence on each smooth sub-panel modulo the
    derivative-singularity at the panel endpoints (which is itself
    addressable by upgrading each sub-panel to vis-cone-upper, but
    that's deferred until the consumer-side migration shows it's
    needed).

    Today this kink-aware subdivision exists only inside
    :func:`~orpheus.derivations.peierls_geometry.build_volume_kernel`;
    promoting it to a primitive lets the ~20
    ``compute_P_esc_*_mode`` / ``compute_G_bc_*_mode`` /
    ``compute_*_mode_marshak`` primitives — which today reinvent a
    weaker version with no subdivision — inherit it for free.

    Parameters
    ----------
    r_obs : float
        Observer radial position. Must be positive.
    omega_low, omega_high : float
        Integration interval. Must satisfy ``omega_low < omega_high``.
    radii : np.ndarray, shape ``(N,)``
        Shell outer radii. Tangent angles are computed only for those
        :math:`r_{k} < r_{\rm obs}` (shells "behind" the observer along
        the ray); shells with :math:`r_{k} \ge r_{\rm obs}` are crossed
        smoothly without tangency.
    n_per_panel : int
        Plain Gauss-Legendre nodes per sub-panel.
    dps : int, keyword-only
        Decimal precision for the underlying GL nodes. Default ``53``.

    Returns
    -------
    Quadrature1D
        Composite rule on :math:`[\omega_{\min}, \omega_{\max}]` with
        ``panel_bounds`` recording the tangent-angle subdivision.

    Notes
    -----
    Tangent angles strictly outside ``(omega_low, omega_high)`` are
    silently dropped (they don't affect the integral on this interval).
    If no tangent angles fall inside, the rule degenerates to plain
    GL on the full interval — i.e. for a homogeneous cell or when
    the angular range doesn't span any tangency, this recipe is
    bit-equivalent to a single :func:`gauss_legendre` call.
    """
    if omega_high <= omega_low:
        raise ValueError(
            f"Need omega_high > omega_low, got "
            f"omega_low={omega_low}, omega_high={omega_high}"
        )
    if r_obs <= 0:
        raise ValueError(f"r_obs must be positive, got {r_obs}")
    if n_per_panel < 1:
        raise ValueError(f"n_per_panel must be >= 1, got {n_per_panel}")

    radii = np.asarray(radii, dtype=float)
    interior = radii[radii < r_obs]
    forward = np.arcsin(np.clip(interior / r_obs, 0.0, 1.0))
    backward = np.pi - forward
    candidates = np.concatenate([forward, backward])

    inside = (candidates > omega_low) & (candidates < omega_high)
    tangents = np.sort(candidates[inside])

    breakpoints = np.concatenate([[omega_low], tangents, [omega_high]])
    return composite_gauss_legendre(
        breakpoints.tolist(), n_per_panel, dps=dps,
    )


# ═══════════════════════════════════════════════════════════════════════
# Recipe 3: surface-centred angular quadrature
# ═══════════════════════════════════════════════════════════════════════

def surface_centred_angular_quadrature(
    r_obs: float,
    r_surface: float,
    radii: np.ndarray,
    n_per_panel: int,
    *,
    phi_low: float = 0.0,
    phi_high: float = np.pi,
    dps: int = 53,
) -> Quadrature1D:
    r"""Surface-centred :math:`\phi`-quadrature on
    :math:`[\phi_{\min}, \phi_{\max}]` for the legacy cylinder
    :math:`G_{\mathrm{bc}}` form, with kink-aware subdivision at the
    tangent angles where the chord from observer to surface point becomes
    tangent to each interior shell.

    Geometry (2-D polar with origin at the cell axis): observer at
    :math:`P = (r_{\rm obs}, 0)`, surface point at
    :math:`Q(\phi) = (r_{\rm surf}\cos\phi,\ r_{\rm surf}\sin\phi)`.
    The chord length is
    :math:`d(\phi) = \sqrt{r_{\rm obs}^{2} + r_{\rm surf}^{2}
    - 2\,r_{\rm obs}\,r_{\rm surf}\cos\phi}` and the impact parameter
    :math:`b(\phi) = r_{\rm obs}\,r_{\rm surf}\,|\sin\phi|/d(\phi)`.
    The chord crosses shell :math:`r_{k}` iff :math:`b(\phi) < r_{k}`,
    a quadratic condition in :math:`c = \cos\phi` whose discriminant
    factors to
    :math:`\Delta = 4\,r_{\rm obs}^{2}\,r_{\rm surf}^{2}\,
    (r_{k}^{2} - r_{\rm obs}^{2})(r_{k}^{2} - r_{\rm surf}^{2})`.
    The discriminant is **positive** only when
    :math:`r_{k} < \min(r_{\rm obs},\,r_{\rm surf})` (both factors
    negative, product positive); the chord becomes tangent to such a
    shell at the two angles
    :math:`\phi_{\pm} = \arccos\!\left(c_{\pm}\right)`,
    :math:`c_{\pm} = (r_{k}^{2} \pm \sqrt{(r_{\rm obs}^{2} - r_{k}^{2})
    (r_{\rm surf}^{2} - r_{k}^{2})})/(r_{\rm obs}\,r_{\rm surf})`.
    Shells with :math:`r_{k}` between :math:`\min` and :math:`\max` of
    :math:`(r_{\rm obs}, r_{\rm surf})` are crossed unconditionally
    (no tangent); shells with :math:`r_{k} > \max` are not reached.

    Compare with :func:`observer_angular_quadrature`, which has the
    closed-form tangent locations :math:`\arcsin(r_{k}/r_{\rm obs})`
    — the special case where one chord endpoint sits at the origin of
    the angular coordinate. The surface-centred form has the observer
    at finite radius and the surface at finite radius, so the tangent
    angles depend on **both** :math:`r_{\rm obs}` and :math:`r_{\rm surf}`
    via the chord-quadratic, not just on :math:`r_{k}/r_{\rm obs}`.

    Parameters
    ----------
    r_obs : float
        Observer radial position. Must be positive.
    r_surface : float
        Radius of the cylindrical surface the chord terminates on.
        Must be positive. May be smaller than ``r_obs`` (inner-surface
        case) or larger (outer-surface case); the math is symmetric in
        :math:`(r_{\rm obs}, r_{\rm surf})`.
    radii : np.ndarray, shape ``(N,)``
        Outer shell radii. Tangent angles are computed only for those
        shells with :math:`r_{k} < \min(r_{\rm obs}, r_{\rm surf})`
        — the only regime where the chord-quadratic discriminant is
        positive.
    n_per_panel : int
        Plain Gauss-Legendre nodes per sub-panel.
    phi_low, phi_high : float, keyword-only
        Integration interval. Defaults to :math:`[0, \pi]` (the legacy
        cylinder :math:`G_{\rm bc}` integration domain — the other
        half-plane is folded by symmetry).
    dps : int, keyword-only
        Decimal precision for the underlying Gauss-Legendre nodes.
        Default ``53`` (float64); pass higher to use mpmath.

    Returns
    -------
    Quadrature1D
        Composite plain-GL rule on :math:`[\phi_{\min}, \phi_{\max}]`
        with ``panel_bounds`` recording the tangent-angle subdivision.
        For an observer in the innermost shell of an outer-surface
        configuration (no shells with :math:`r_{k} < r_{\rm obs}`) or
        for any inner-surface configuration where every shell satisfies
        :math:`r_{k} \ge r_{\rm surf}` (the typical hollow-cell
        layout), the rule degenerates to plain GL on the full interval
        — bit-equivalent to a single
        ``gauss_legendre(phi_low, phi_high, n_per_panel)`` call.

    Notes
    -----
    Tangent angles strictly outside ``(phi_low, phi_high)`` are silently
    dropped. The discriminant clamp (``np.maximum(disc, 0.0)``) and the
    ``np.clip(c_pm, -1, 1)`` guard handle the degenerate
    :math:`r_{k} = \min(r_{\rm obs}, r_{\rm surf})` case (single tangent
    where :math:`c_{+} = c_{-}`) and floating-point rounding at the
    interval boundaries.
    """
    if r_obs <= 0.0:
        raise ValueError(f"r_obs must be positive, got {r_obs}")
    if r_surface <= 0.0:
        raise ValueError(f"r_surface must be positive, got {r_surface}")
    if phi_high <= phi_low:
        raise ValueError(
            f"Need phi_high > phi_low, got "
            f"phi_low={phi_low}, phi_high={phi_high}"
        )
    if n_per_panel < 1:
        raise ValueError(f"n_per_panel must be >= 1, got {n_per_panel}")

    radii = np.asarray(radii, dtype=float)
    cutoff = min(r_obs, r_surface)
    interior = radii[radii < cutoff]

    if interior.size == 0:
        candidates = np.array([], dtype=float)
    else:
        rk_sq = interior * interior
        disc = (r_obs * r_obs - rk_sq) * (r_surface * r_surface - rk_sq)
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        denom = r_obs * r_surface
        c_plus = np.clip((rk_sq + sqrt_disc) / denom, -1.0, 1.0)
        c_minus = np.clip((rk_sq - sqrt_disc) / denom, -1.0, 1.0)
        candidates = np.concatenate([np.arccos(c_plus), np.arccos(c_minus)])

    inside = (candidates > phi_low) & (candidates < phi_high)
    tangents = np.unique(np.sort(candidates[inside]))

    breakpoints = np.concatenate([[phi_low], tangents, [phi_high]])
    return composite_gauss_legendre(
        breakpoints.tolist(), n_per_panel, dps=dps,
    )
