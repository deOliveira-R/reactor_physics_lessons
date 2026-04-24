r"""Unified polar-form Peierls Nyström infrastructure.

Provides the geometry-abstract machinery shared between the 1-D
radial cylindrical and spherical Peierls integral-equation reference
solvers. The slab geometry is qualitatively different (the 1-D kernel
:math:`E_1` carries a log singularity that the polar reformulation
moves into a :math:`\rho_{\max}\to\infty` grazing-angle divergence)
and is NOT unified here — it remains in :mod:`peierls_slab` with its
native :math:`E_1` Nyström.

See :doc:`/theory/peierls_unified` for the end-to-end mathematical
derivation. The executive summary:

- The 3-D isotropic point kernel is
  :math:`G_{3D}(R) = e^{-\Sigma_t R}/(4\pi R^{2})`.
- Reducing dimensions by the geometry's symmetry gives
  :math:`G_{d}(|r-r'|) = \kappa_d(\Sigma_t|r-r'|) / (S_d\,|r-r'|^{d-1})`
  with :math:`\kappa_d \in \{E_1/2, \mathrm{Ki}_1, e^{-\tau}\}` and
  :math:`S_d \in \{2, 2\pi, 4\pi\}`.
- Writing the Peierls equation in polar coordinates centred at the
  observer, the volume element :math:`\rho^{d-1}\,\mathrm d\Omega\,
  \mathrm d\rho` cancels the :math:`1/|r-r'|^{d-1}` denominator,
  leaving a SMOOTH integrand
  :math:`\kappa_d(\Sigma_t\rho)\,q(r'(\rho,\Omega,r))` for all
  curvilinear geometries:

.. math::

   \Sigma_t(r)\,\varphi(r)
     \;=\; \frac{\Sigma_t(r)}{S_d}\!
       \int_{\Omega_d}\!\mathrm d\Omega\!
       \int_0^{\rho_{\max}(r,\Omega)}\!\!
         \kappa_d(\Sigma_t\rho)\,q\bigl(r'(\rho,\Omega,r)\bigr)\,
       \mathrm d\rho
     + S_{\rm bc}(r).

For the radially-symmetric 1-D problem, cylinder and sphere share
identical ray-geometry formulas — only the angular measure and the
kernel function differ. That is the architectural lever this module
exploits: one body of ray-walking, Lagrange-basis, quadrature, and
power-iteration code serves both.

This module is the Phase-4.2-post refactor deliverable. It does not
change any numerics; it purely consolidates infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import mpmath
import numpy as np

from ._kernels import (  # noqa: F401
    _shifted_legendre_eval,
    chord_half_lengths,
    ki_n_float,
    ki_n_mp,
)
# ═══════════════════════════════════════════════════════════════════════
# Gauss-Legendre helpers (geometry-agnostic)
# ═══════════════════════════════════════════════════════════════════════

def gl_nodes_weights(n: int, dps: int) -> tuple[list, list]:
    """*n*-point Gauss-Legendre on :math:`[-1, 1]` at *dps* precision."""
    with mpmath.workdps(dps):
        nm, wm = mpmath.gauss_quadrature(n, "legendre")
        return [nm[i] for i in range(n)], [wm[i] for i in range(n)]


def map_gl_to(nodes, weights, a, b):
    """Map reference GL nodes/weights from :math:`[-1, 1]` to :math:`[a, b]`."""
    h = (b - a) / 2
    m = (a + b) / 2
    return [m + h * t for t in nodes], [h * w for w in weights]


def gl_float(n: int, a: float, b: float, dps: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """*n*-point GL on :math:`[a, b]` returned as double-precision arrays."""
    ref_nodes, ref_wts = gl_nodes_weights(n, dps)
    h = (b - a) / 2
    m = (a + b) / 2
    nodes = np.array([float(m + h * t) for t in ref_nodes])
    wts = np.array([float(h * w) for w in ref_wts])
    return nodes, wts


def gauss_laguerre_nodes_weights(
    n: int, dps: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    r"""*n*-point Gauss-Laguerre on :math:`[0, \infty)` with weight
    :math:`e^{-\tau}`.

    Optimal for integrands of the form :math:`e^{-\tau}\,g(\tau)` where
    :math:`g` is smooth on :math:`[0, \infty)`. In Peierls polar form
    under the :math:`\tau`-coordinate transform (:doc:`/theory/peierls_unified`
    §5), the ρ integration becomes
    :math:`\int_0^{\tau_{\max}} e^{-\tau}\,q(r'(\tau))/\Sigma_t\,
    \mathrm d\tau`, which Gauss-Laguerre integrates spectrally — the
    grazing-ray stiffness (``τ_max → ∞`` as ``μ → 0``) is absorbed by
    the e^{-τ} weight automatically, since Laguerre nodes concentrate
    where the exponential is non-negligible (:math:`\tau \lesssim n`).

    Returns ``(nodes, weights)`` as :class:`numpy.ndarray`.
    """
    with mpmath.workdps(dps):
        nm, wm = mpmath.gauss_quadrature(n, "laguerre")
        nodes = np.array([float(nm[i]) for i in range(n)])
        weights = np.array([float(wm[i]) for i in range(n)])
    return nodes, weights


def composite_gl_r(
    radii: np.ndarray,
    n_panels_per_region: int,
    p_order: int,
    dps: int = 30,
    *,
    inner_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int, int]]]:
    r"""Composite GL on :math:`[r_0, R]` with panel breakpoints at
    annular radii.

    For solid geometries (``inner_radius == 0``, the default), nodes
    cover the full disk :math:`[0, R]`. For hollow cells
    (``inner_radius > 0``, Phase F.4) nodes cover the annulus
    :math:`[r_0, R]` only — the cavity :math:`[0, r_0]` carries no
    source and is excluded from the radial mesh. The inner endpoint
    :math:`r_0` becomes an additional panel breakpoint.

    Shared by cylindrical and spherical Peierls solvers; the panel
    structure accommodates the :math:`\Sigma_t(r)` discontinuities at
    each :math:`r_k`.

    Returns ``(r_pts, r_wts, panel_bounds)`` where ``panel_bounds`` is
    a list of ``(pa, pb, i_start, i_end)`` tuples describing the
    composite rule's panels.
    """
    radii = np.asarray(radii, dtype=float)
    if inner_radius < 0.0:
        raise ValueError(f"inner_radius must be >= 0, got {inner_radius}")
    if inner_radius >= float(radii[-1]):
        raise ValueError(
            f"inner_radius ({inner_radius}) must be < outer radius "
            f"({float(radii[-1])})"
        )
    gl_ref, gl_wt = gl_nodes_weights(p_order, dps)

    breakpoints = [mpmath.mpf(inner_radius)] + [
        mpmath.mpf(float(r)) for r in radii if float(r) > inner_radius
    ]
    r_all: list = []
    w_all: list = []
    panel_bounds: list[tuple[float, float, int, int]] = []

    with mpmath.workdps(dps):
        for seg in range(len(breakpoints) - 1):
            a_seg = breakpoints[seg]
            b_seg = breakpoints[seg + 1]
            pw = (b_seg - a_seg) / n_panels_per_region
            for pidx in range(n_panels_per_region):
                pa = a_seg + pidx * pw
                pb = pa + pw
                xp, wp = map_gl_to(gl_ref, gl_wt, pa, pb)
                i0 = len(r_all)
                r_all.extend(xp)
                w_all.extend(wp)
                panel_bounds.append((float(pa), float(pb), i0, len(r_all)))

    r_pts = np.array([float(r) for r in r_all])
    r_wts = np.array([float(w) for w in w_all])
    return r_pts, r_wts, panel_bounds


# ═══════════════════════════════════════════════════════════════════════
# Lagrange basis on composite-GL panels (geometry-agnostic)
# ═══════════════════════════════════════════════════════════════════════

def lagrange_basis_on_panels(
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    r_eval: float,
) -> np.ndarray:
    r"""Piecewise Lagrange basis :math:`L_j(r_{\rm eval})`.

    On each panel :math:`[p_a, p_b]` the basis is the Lagrange
    polynomial of the panel's nodes; elsewhere it is zero. Points
    outside :math:`[0, R]` are clamped to the nearest panel.
    """
    N = len(r_nodes)
    L = np.zeros(N)

    panel_idx = None
    for k, (pa, pb, i_start, i_end) in enumerate(panel_bounds):
        if pa <= r_eval <= pb:
            panel_idx = k
            break
    if panel_idx is None:
        panel_idx = 0 if r_eval < panel_bounds[0][0] else len(panel_bounds) - 1

    pa, pb, i_start, i_end = panel_bounds[panel_idx]
    local_nodes = r_nodes[i_start:i_end]
    p = i_end - i_start
    for a in range(p):
        num, den = 1.0, 1.0
        for b in range(p):
            if b == a:
                continue
            num *= (r_eval - local_nodes[b])
            den *= (local_nodes[a] - local_nodes[b])
        L[i_start + a] = num / den
    return L


# ═══════════════════════════════════════════════════════════════════════
# Curvilinear geometry abstraction
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CurvilinearGeometry:
    r"""Unified 1-D radial curvilinear geometry for the Peierls polar form.

    Three concrete specialisations:

    - ``kind = "slab-polar"``: :math:`d = 3` but Cartesian;
      :math:`\kappa = e^{-\tau}`, angular variable
      :math:`\mu \in [-1, 1]` with uniform measure :math:`\mathrm d\mu`.
      Observer-centred form: :math:`\varphi(x) = \tfrac12\!\int_{-1}^{1}\!
      \mathrm d\mu\!\int_0^{\rho_{\max}}\!e^{-\Sigma_t\rho}\,q(x+\rho\mu)
      \,\mathrm d\rho`. NOT the log-E₁ Nyström; see
      :doc:`/theory/peierls_unified` §4 (Chapter 4).
    - ``kind = "cylinder-1d"``: :math:`d = 2`, :math:`S_d = 2\pi`,
      :math:`\kappa_d = \mathrm{Ki}_1`, angular variable
      :math:`\beta \in [0, \pi]` with uniform measure
      :math:`\mathrm d\beta`.
    - ``kind = "sphere-1d"``: :math:`d = 3`, :math:`S_d = 4\pi`,
      :math:`\kappa_d = e^{-\tau}`, angular variable
      :math:`\theta \in [0, \pi]` with measure
      :math:`\sin\theta\,\mathrm d\theta` (azimuthal folded).

    The geometric primitives (:math:`\rho_{\max}`,
    :math:`r'(\rho, \Omega, r)`) share a single closed-form across
    cylinder and sphere; slab has its own linear-ray forms.

    The **direction cosine** :math:`\mu = \cos(\rm angle\ from\ radial/normal)`
    is the unified downstream quantity. For cylinder/sphere
    :math:`\mu = \cos(\omega)`; for slab the angular variable IS
    :math:`\mu`. :meth:`ray_direction_cosine` maps from the
    integration variable to :math:`\mu`.

    The **polar-form prefactor** ``prefactor`` bundles together:

    1. :math:`1/S_d` from the 3-D point-kernel normalisation.
    2. The azimuthal-symmetry fold.
    3. The :math:`\pm\beta` / :math:`\pm\theta` reflection fold.

    For cylinder: :math:`2 \cdot 2\pi / (2\pi) = 2` net numerator
    divided by :math:`2\pi` gives :math:`1/\pi`.
    For sphere: :math:`2\pi / (4\pi) = 1/2` — with :math:`\sin\theta`
    weight.
    For slab: :math:`1/2` — the :math:`(1/(4\pi))\cdot 2\pi`
    azimuthal fold gives :math:`1/2` with uniform :math:`\mathrm d\mu`
    measure after the :math:`\mu = \cos\theta` change of variable.
    """

    kind: str
    inner_radius: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("slab-polar", "cylinder-1d", "sphere-1d"):
            raise ValueError(f"Unsupported geometry kind {self.kind!r}")
        if self.inner_radius < 0.0:
            raise ValueError(
                f"inner_radius must be >= 0, got {self.inner_radius!r}"
            )
        if self.kind == "slab-polar" and self.inner_radius != 0.0:
            raise ValueError(
                "slab-polar does not carry inner_radius (use face_0 / face_L "
                "per-surface BC directly)"
            )

    # ── geometric constants ───────────────────────────────────────────

    @property
    def d(self) -> int:
        """Effective dimension of the dimensionally-reduced kernel (2 or 3)."""
        return {
            "slab-polar": 3,
            "cylinder-1d": 2,
            "sphere-1d": 3,
        }[self.kind]

    @property
    def S_d(self) -> float:
        r"""Total solid angle :math:`S_d` of the unit :math:`(d-1)`-sphere."""
        return {
            "slab-polar": 4.0 * np.pi,
            "cylinder-1d": 2.0 * np.pi,
            "sphere-1d": 4.0 * np.pi,
        }[self.kind]

    @property
    def prefactor(self) -> float:
        """Composite prefactor absorbing :math:`1/S_d` + azimuth fold + :math:`\\pm` fold.

        See the class docstring for the derivation.
        """
        return {
            "slab-polar": 0.5,
            "cylinder-1d": 1.0 / np.pi,
            "sphere-1d": 0.5,
        }[self.kind]

    @property
    def n_surfaces(self) -> int:
        """Number of distinct boundary surfaces carrying re-entry data.

        - ``slab-polar``: 2 (face at :math:`x = 0` and face at :math:`x = L`).
        - ``cylinder-1d`` / ``sphere-1d`` with ``inner_radius == 0``:
          1 (outer boundary only).
        - ``cylinder-1d`` / ``sphere-1d`` with ``inner_radius > 0``:
          2 (outer at :math:`R` and inner at :math:`r_0`).

        Phase F's :class:`BoundaryClosureOperator` uses this to size
        the per-face mode space :math:`A = \\mathbb{R}^{N_{\\rm modes}
        \\times N_{\\rm surfaces}}`.
        """
        if self.kind == "slab-polar":
            return 2
        return 2 if self.inner_radius > 0.0 else 1

    @property
    def topology(self) -> str:
        r"""Topological class for Peierls closure dispatch.

        Returns one of:

        - ``"two_surface"`` — the cell has two boundary surfaces that
          carry re-entry data. Members: slab (two parallel faces),
          hollow annular cylinder (inner + outer ring), hollow sphere
          (inner + outer shell). **F.4 scalar rank-2 per-face closure
          (Stamm'ler Eq. 34 = Hébert 2009 Eq. 3.323) applies to this
          class.** All members share the same L19 stability-protocol
          coverage and the L21 structural residual class.

        - ``"one_surface_compact"`` — the cell has a single boundary
          surface (compact convex body). Members: solid cylinder
          (``inner_radius == 0``), solid sphere (``inner_radius == 0``).
          **F.4 is structurally unavailable** on this class because
          there is no second face to couple to; the only shipped
          closure is rank-1 Mark.

        This is the **primary organizing principle** for Peierls
        reference cases (see
        :ref:`theory-peierls-capabilities` and
        :file:`.claude/plans/topology-based-consolidation.md`). Tests
        and case builders should dispatch on ``topology`` rather than
        on ``kind`` + ``inner_radius`` gymnastics.

        Semantically equivalent to ``n_surfaces`` (2 ↔ ``"two_surface"``,
        1 ↔ ``"one_surface_compact"``), but named for its role rather
        than its arity. Code that cares about the topology *label* —
        dispatching case builders, filtering tests, gating closure
        applicability — should prefer this property.
        """
        return "two_surface" if self.n_surfaces == 2 else "one_surface_compact"

    @property
    def is_planar(self) -> bool:
        """True for Cartesian (slab) geometry; False for curvilinear.

        Flips which branch of the ray-geometry formulas is active
        (linear vs. circular). Use sparingly — prefer the polymorphic
        geometry methods whenever possible.
        """
        return self.kind == "slab-polar"

    @property
    def angular_range(self) -> tuple[float, float]:
        """Integration range of the angular variable.

        Slab: :math:`\\mu \\in [-1, 1]` (direction cosine).
        Cylinder / sphere: :math:`\\omega \\in [0, \\pi]`.
        """
        if self.kind == "slab-polar":
            return (-1.0, 1.0)
        return (0.0, np.pi)

    # ── angular variable ↔ direction cosine ───────────────────────────

    def ray_direction_cosine(self, angular_var: np.ndarray) -> np.ndarray:
        r"""Map the angular integration variable to the ray's direction
        cosine :math:`\mu`.

        Slab: identity (the angular variable IS :math:`\mu`).
        Cylinder / sphere: :math:`\mu = \cos(\omega)`.

        ``build_volume_kernel`` uses this so every downstream
        ray-geometry primitive (rho_max, source_position, optical
        depth, crossings) sees the same :math:`\mu` regardless of
        geometry kind.
        """
        if self.kind == "slab-polar":
            return np.asarray(angular_var, dtype=float)
        return np.cos(angular_var)

    # ── angular measure ───────────────────────────────────────────────

    def angular_weight(self, omega_pts: np.ndarray) -> np.ndarray:
        r"""Weight factor in the angular measure.

        Slab: :math:`\mathrm d\mu` ⇒ weight = 1.
        Cylinder: :math:`\mathrm d\beta` ⇒ weight = 1.
        Sphere:  :math:`\sin\theta\,\mathrm d\theta` ⇒ weight = :math:`\sin\theta`.
        """
        omega_pts = np.asarray(omega_pts, dtype=float)
        if self.kind == "sphere-1d":
            return np.sin(omega_pts)
        return np.ones_like(omega_pts)

    # ── ray geometry ──────────────────────────────────────────────────

    def rho_max(self, r_obs: float, cos_omega: float, R: float) -> float:
        r"""Ray-exit distance along direction :math:`\mu = \cos\Omega`.

        Slab: :math:`(R - x)/\mu` for :math:`\mu > 0`,
        :math:`-x/\mu` for :math:`\mu < 0`. At :math:`\mu = 0` the ray
        is parallel to the slab faces and would have infinite
        :math:`\rho_{\max}` — the caller must avoid sampling exactly
        :math:`\mu = 0` (GL interior nodes naturally skip endpoints).

        Cylinder / sphere: positive root of
        :math:`(r_{\rm obs} + \rho\cos\Omega)^2 + (\rho\sin\Omega)^2 = R^2`.
        """
        if self.kind == "slab-polar":
            if cos_omega > 0.0:
                return (R - r_obs) / cos_omega
            if cos_omega < 0.0:
                return -r_obs / cos_omega  # = r_obs / |cos_omega|
            return float("inf")
        disc = r_obs * r_obs * cos_omega * cos_omega + R * R - r_obs * r_obs
        return -r_obs * cos_omega + np.sqrt(max(disc, 0.0))

    def rho_inner_intersections(
        self, r_obs: float, cos_omega: float,
    ) -> tuple[float | None, float | None]:
        r"""Forward-distances at which a ray hits the **inner** shell
        :math:`r = r_0` (``self.inner_radius``).

        Solves
        :math:`(r_{\rm obs} + \rho\cos\Omega)^2 + (\rho\sin\Omega)^2 = r_0^2`
        for :math:`\rho`, returning the two roots
        :math:`(\rho^-, \rho^+)` with :math:`\rho^- \le \rho^+` if both are
        positive, otherwise ``None`` in the slot of any non-positive root.

        Returns ``(None, None)`` when:

        - :math:`r_0 = 0` (solid geometry — no cavity shell);
        - ``kind == "slab-polar"`` (slab carries its two faces explicitly,
          not via an inner radius);
        - the ray misses the inner shell (negative discriminant) or both
          intersections are behind the observer (:math:`\rho \le 0`).

        Tangent rays to the inner shell produce a double root; both
        slots return the same positive value.
        """
        if self.inner_radius == 0.0 or self.kind == "slab-polar":
            return (None, None)
        r0 = float(self.inner_radius)
        r_obs_sq = r_obs * r_obs
        disc = r_obs_sq * cos_omega * cos_omega - (r_obs_sq - r0 * r0)
        if disc < 0.0:
            return (None, None)
        sqrt_disc = float(np.sqrt(disc))
        rho_minus = -r_obs * cos_omega - sqrt_disc
        rho_plus = -r_obs * cos_omega + sqrt_disc
        return (
            rho_minus if rho_minus > 0.0 else None,
            rho_plus if rho_plus > 0.0 else None,
        )

    def source_position(
        self, r_obs: float, rho: float, cos_omega: float,
    ) -> float:
        r"""Source position along the ray at distance :math:`\rho`.

        Slab: linear, :math:`x' = x + \rho\,\mu`.
        Cylinder / sphere: curvilinear,
        :math:`r' = \sqrt{r_{\rm obs}^2 + 2 r_{\rm obs}\rho\cos\Omega + \rho^2}`.
        The 1-D radial symmetry hides the 3-D direction-of-ray-in-azimuth
        dependence, so only :math:`\cos\Omega` matters.
        """
        if self.kind == "slab-polar":
            return r_obs + rho * cos_omega
        return np.sqrt(
            r_obs * r_obs + 2.0 * r_obs * rho * cos_omega + rho * rho
        )

    # ── ray-integrated optical depth ──────────────────────────────────

    def optical_depth_along_ray(
        self,
        r_obs: float,
        cos_omega: float,
        rho: float,
        radii: np.ndarray,
        sig_t: np.ndarray,
    ) -> float:
        r"""Integrate :math:`\Sigma_t(\cdot)` along the ray from
        :math:`r_{\rm obs}` in direction :math:`\mu = \cos\Omega` for
        distance :math:`\rho`.

        Slab: linear path :math:`x'(s) = x + s\mu` crossing slab
        boundaries at :math:`s_b = (r_b - x)/\mu`. ``radii`` is
        interpreted as :math:`[0, r_1, \dots, r_{N-1} = L]`, i.e. the
        cumulative region boundaries including both endpoints.

        Cylinder / sphere: walks curvilinear crossings of the annular
        shells via :math:`(r_{\rm obs}+\rho\cos\Omega)^2 +
        (\rho\sin\Omega)^2 = r_k^2`.
        """
        radii = np.asarray(radii, dtype=float)
        sig_t = np.asarray(sig_t, dtype=float)
        N = len(radii)

        # Homogeneous solid cell fast path (single annulus, no cavity).
        # Hollow cells must fall through to the crossing walker so the
        # cavity segment can be skipped.
        if N == 1 and self.inner_radius == 0.0:
            return float(sig_t[0]) * rho

        if self.kind == "slab-polar":
            # Linear ray x'(s) = r_obs + s*cos_omega on s ∈ [0, rho].
            # ``radii`` follows the curvilinear convention:
            # ``[r_1, r_2, ..., r_N = L]`` are outer edges of N slabs
            # with implicit r_0 = 0; ``sig_t[k]`` is the XS in
            # :math:`[r_{k-1}, r_k]` (r_{-1} = 0). Walk linear interior
            # crossings.
            crossings = [0.0]
            if cos_omega != 0.0:
                for r_k in radii[:-1]:
                    s = (r_k - r_obs) / cos_omega
                    if 0.0 < s < rho:
                        crossings.append(s)
            crossings.append(rho)
            crossings.sort()

            tau = 0.0
            for i_seg in range(len(crossings) - 1):
                s_lo, s_hi = crossings[i_seg], crossings[i_seg + 1]
                x_mid = r_obs + 0.5 * (s_lo + s_hi) * cos_omega
                k = self.which_annulus(x_mid, radii)
                tau += float(sig_t[k]) * (s_hi - s_lo)
            return tau

        # Curvilinear (cylinder/sphere): ring crossings.
        crossings = [0.0]
        for r_k in radii[:-1]:
            disc = (
                r_obs * r_obs * cos_omega * cos_omega
                - (r_obs * r_obs - r_k * r_k)
            )
            if disc < 0.0:
                continue
            sqrt_disc = np.sqrt(disc)
            s_a = -r_obs * cos_omega - sqrt_disc
            s_b = -r_obs * cos_omega + sqrt_disc
            for s in (s_a, s_b):
                if 0.0 < s < rho:
                    crossings.append(s)

        # Hollow-core cavity: interior to r_0 the medium is void
        # (:math:`\Sigma_t = 0`). Insert the cavity entry/exit ρ as
        # crossings and skip τ accumulation for segments whose midpoint
        # falls inside the cavity (r_mid < r_0).
        r0 = float(self.inner_radius)
        if r0 > 0.0:
            rho_in_minus, rho_in_plus = self.rho_inner_intersections(
                r_obs, cos_omega,
            )
            for s in (rho_in_minus, rho_in_plus):
                if s is not None and 0.0 < s < rho:
                    crossings.append(s)

        crossings.append(rho)
        crossings.sort()

        tau = 0.0
        for i_seg in range(len(crossings) - 1):
            s_lo, s_hi = crossings[i_seg], crossings[i_seg + 1]
            s_mid = 0.5 * (s_lo + s_hi)
            r_mid_sq = (
                r_obs * r_obs + 2.0 * r_obs * s_mid * cos_omega + s_mid * s_mid
            )
            r_mid = np.sqrt(max(r_mid_sq, 0.0))
            # Cavity segment (void): zero-Σ_t, skip contribution.
            if r0 > 0.0 and r_mid < r0:
                continue
            # Outermost annulus is the default (handles floating-point
            # noise at the cylinder boundary).
            k = N - 1
            for kk in range(N):
                if r_mid < radii[kk]:
                    k = kk
                    break
            tau += sig_t[k] * (s_hi - s_lo)
        return tau

    def which_annulus(self, r: float, radii: np.ndarray) -> int:
        """Index of the region containing ``r`` (outer-biased at boundary).

        Shared convention for slab and curvilinear: ``radii`` holds
        outer edges :math:`[r_1, \\ldots, r_N = R]` with implicit
        :math:`r_0 = 0`; ``sig_t[k]`` is the XS in the region
        :math:`[r_{k-1}, r_k]`. ``k`` is the smallest index with
        :math:`r < r_k`, clamped at ``N-1``.
        """
        k = len(radii) - 1
        for kk, r_k in enumerate(radii):
            if r < r_k:
                return kk
        return k

    # ── ray / panel-boundary crossings ────────────────────────────────

    def omega_tangent_angles(
        self,
        r_obs: float,
        panel_boundaries_r: np.ndarray,
        *,
        tol: float = 1e-12,
    ) -> list[float]:
        r"""Angles :math:`\omega` at which the observer-anchored ray is
        tangent to an interior panel-boundary shell.

        For observers strictly outside a panel boundary :math:`r_b`, the
        ray has :math:`r_{\min}(\omega) = r_{\rm obs}|\sin\omega|`; this
        equals :math:`r_b` when :math:`\sin\omega = r_b/r_{\rm obs}`,
        producing a *bifurcation* in the ρ integration structure (two
        crossings just below, zero crossings just above the critical
        angle). The tangent geometry gives :math:`L_j(r'(\rho))` a
        quadratic (C¹-discontinuous) kink at the tangent ρ, which
        translates into a derivative discontinuity of the outer
        :math:`\omega` integrand. Fixed-order GL cannot integrate across
        such kinks.

        Returns the sorted list of critical ω in :math:`(0, \pi)` that
        need to appear as subdivision breakpoints in the outer rule
        (issue #114 — second phase of the curvilinear-ρ/ω fix).

        For observers *inside* a boundary (``r_obs ≤ r_b``), no tangent
        critical angle exists.

        **Slab**: rays are linear, there is no turning-point geometry,
        so no tangent bifurcation. Returns ``[]``.
        """
        if self.kind == "slab-polar":
            return []
        angles: list[float] = []
        for r_b in panel_boundaries_r:
            if r_obs <= r_b + tol:
                continue
            ratio = r_b / r_obs
            if ratio >= 1.0 - tol:
                continue
            omega_c = float(np.arcsin(ratio))
            # Two tangent angles per boundary: symmetric about π/2
            for ang in (omega_c, np.pi - omega_c):
                if tol < ang < np.pi - tol:
                    angles.append(ang)
        return sorted(angles)

    def rho_crossings_for_ray(
        self,
        r_obs: float,
        cos_omega: float,
        rho_max_val: float,
        panel_boundaries_r: np.ndarray,
        *,
        tol: float = 1e-12,
    ) -> list[float]:
        r"""Ray distances :math:`\rho` at which :math:`r'(\rho)` crosses a
        spatial panel boundary :math:`r_b`.

        Solves :math:`r_{\rm obs}^2 + 2 r_{\rm obs}\,\rho\,\cos\Omega +
        \rho^2 = r_b^2` for each panel boundary :math:`r_b` and keeps the
        positive roots strictly inside :math:`(0, \rho_{\max})`. These
        are the points along the ray where the piecewise-polynomial
        Lagrange basis :math:`L_j(r')` has a derivative discontinuity
        (panel kink). Fixed-order Gauss-Legendre cannot integrate across
        such kinks — the quadrature on :math:`\rho` must subdivide at
        every crossing to restore spectral convergence. See :issue:`114`.

        Identical formula for cylinder and sphere; both share
        :meth:`source_position`.

        **Slab**: :math:`x'(\rho) = x + \rho\mu` is linear, so each
        panel boundary :math:`r_b` gives at most one crossing at
        :math:`\rho = (r_b - x)/\mu`. Same kink-in-:math:`L_j` mechanism
        as curvilinear, same subdivision requirement.
        """
        crossings: set[float] = set()
        if self.kind == "slab-polar":
            if cos_omega == 0.0:
                return []
            for r_b in panel_boundaries_r:
                rho = (r_b - r_obs) / cos_omega
                if tol < rho < rho_max_val - tol:
                    crossings.add(rho)
            return sorted(crossings)

        r_obs_sq = r_obs * r_obs
        disc_base = r_obs_sq * cos_omega * cos_omega - r_obs_sq
        for r_b in panel_boundaries_r:
            disc = disc_base + r_b * r_b
            if disc < 0.0:
                continue
            sqrt_disc = float(np.sqrt(disc))
            for sign in (+1.0, -1.0):
                rho = -r_obs * cos_omega + sign * sqrt_disc
                if tol < rho < rho_max_val - tol:
                    crossings.add(rho)
        return sorted(crossings)

    # ── volume kernel :math:`\kappa_d(\tau)` ──────────────────────────

    def volume_kernel_mp(self, tau: float, dps: int = 25) -> float:
        r"""Volume Peierls kernel :math:`\kappa_d(\tau)` returned as a
        Python :class:`float`.

        Slab (observer-centred polar form): :math:`e^{-\tau}` — the
        SAME kernel as the sphere, because both arise from full 3-D
        integration of the isotropic point kernel over the respective
        symmetry. The log-:math:`E_1` that appears in the legacy slab
        Nyström comes from pre-integrating the angular coordinate
        analytically; keeping it explicit (as we do here) gives the
        simpler kernel.

        Cylinder: :math:`\mathrm{Ki}_1(\tau)` (A&S 11.2).
        Sphere:  :math:`e^{-\tau}`.

        For cylinder at ``dps < 30``, uses the fast scipy-based
        :func:`~.._kernels.ki_n_float` since the output is always cast
        to :class:`float`. At ``dps >= 30`` falls back to the
        arbitrary-precision :func:`~.._kernels.ki_n_mp` for
        high-precision reference computations.
        """
        if self.kind == "cylinder-1d":
            if dps >= 30:
                return float(ki_n_mp(1, float(tau), dps))
            return ki_n_float(1, float(tau))
        # slab-polar and sphere-1d both use exp(-τ)
        return math.exp(-float(tau))

    # ── escape kernel (for :math:`P_{\rm esc}` angular integration) ───

    def escape_kernel_mp(self, tau: float, dps: int = 25) -> float:
        r"""Escape-angular-integral kernel :math:`K_{\rm esc}(\tau)`.

        Cylinder: :math:`\mathrm{Ki}_2(\tau)`. The factor of 2 from the
        3-D polar-angle integration
        :math:`\int_0^\pi \sin\theta\,e^{-\tau/\sin\theta}\,\mathrm d\theta =
        2\,\mathrm{Ki}_2(\tau)` is already absorbed into the geometry's
        :attr:`prefactor` :math:`1/\pi` (via the 2π-to-π azimuthal
        fold: :math:`1/(4\pi) \cdot 2 \cdot 2 / S_2^{\rm fold} = 1/\pi`).

        Sphere: :math:`e^{-\tau}` directly (3-D integration is explicit
        with the :math:`\sin\theta` angular weight).

        The unified identity
        :math:`P_{\rm esc}(r_i) = C_d \cdot \int \mathrm d\Omega \cdot
        W_\Omega(\Omega) \cdot K_{\rm esc}(\tau(\Omega))` uses the SAME
        :attr:`prefactor` as the volume kernel for both geometries —
        a consequence of both escape and emergence being angular integrals
        of the same point kernel over the same solid-angle domain.
        """
        if self.kind == "cylinder-1d":
            return float(ki_n_mp(2, float(tau), dps))
        return float(mpmath.exp(-mpmath.mpf(tau)))

    # ── radial integration weight (per-unit-volume element) ───────────

    def radial_volume_weight(self, r: float) -> float:
        r"""Weight :math:`r^{d-1}` from the polar area/volume element
        :math:`\mathrm dV' = r^{d-1}\,\mathrm dr\,\mathrm d\Omega`.

        Slab (1-D radial coordinate :math:`x`, :math:`\mathrm dV = \mathrm dx`):
        returns :math:`1` (no geometric factor).
        Cylinder (:math:`d = 2`): returns :math:`r` (for
        :math:`r\,\mathrm d r\,\mathrm d\beta`).
        Sphere (:math:`d = 3`): returns :math:`r^2`.
        """
        if self.kind == "slab-polar":
            return 1.0
        if self.kind == "cylinder-1d":
            return r
        return r * r

    # ── surface measure at the outer boundary :math:`|r|=R` ───────────

    def surface_area_per_z(self, R: float) -> float:
        """Surface 'area' measure:

        - Cylinder: :math:`2\\pi R` (lateral surface per unit z).
        - Sphere:   :math:`4\\pi R^2` (full spherical surface area).
        """
        if self.kind == "cylinder-1d":
            return 2.0 * np.pi * R
        return 4.0 * np.pi * R * R

    def rank1_surface_divisor(self, R: float) -> float:
        r"""Normalisation divisor for the rank-1 white-BC :math:`u_i` vector.

        The theoretical form :math:`K_{\rm bc}[i, j] = \Sigma_t(r_i)\,
        G_{\rm bc}(r_i) / A_d \cdot A_j\,P_{\rm esc}(r_j)` with
        :math:`A_j` the volume-element area and :math:`A_d` the cell's
        surface area reduces — after the shared azimuthal factor cancels
        between :math:`A_d` and :math:`A_j` — to geometry-specific
        divisors:

        - **Slab** (:math:`A_d = 2` per unit transverse area — two unit-area
          faces at :math:`x = 0` and :math:`x = L`;
          :math:`A_j = w_j` per-panel length): ratio :math:`A_j/A_d = w_j/2`,
          divisor :math:`2`.
        - **Cylinder** (:math:`A_d = 2\pi R`, :math:`A_j = 2\pi r_j w_j`):
          ratio :math:`A_j / A_d = r_j w_j / R`, divisor :math:`R`.
        - **Sphere** (:math:`A_d = 4\pi R^2`, :math:`A_j = 4\pi r_j^2 w_j`):
          ratio :math:`A_j / A_d = r_j^2 w_j / R^{2}`, divisor :math:`R^{2}`.
        """
        if self.kind == "slab-polar":
            return 2.0
        if self.kind == "cylinder-1d":
            return R
        return R * R

    # ── case-builder helpers (Stage 2 of the simplification) ──────────

    def shell_volume_integral(
        self, r_nodes: np.ndarray, r_wts: np.ndarray, phi: np.ndarray,
    ) -> float:
        r"""Compute :math:`\int_{r_0}^{R}\!\varphi(r)\,\mathrm dV(r)` at the
        current geometry's volume element.

        Used by continuous-reference case builders to normalize the flux.
        Replaces the per-geometry duplicated expressions previously
        hand-coded in ``peierls_{cylinder,sphere,slab}.py``:

        - **Slab** (``kind="slab-polar"``): :math:`\int \varphi\,\mathrm dx
          = \sum_j w_j\,\varphi_j`.
        - **Cylinder** (``kind="cylinder-1d"``): :math:`2\pi\int r\,\varphi
          \,\mathrm dr = 2\pi\sum_j r_j\,w_j\,\varphi_j` (per unit z).
        - **Sphere** (``kind="sphere-1d"``): :math:`4\pi\int r^2\,\varphi
          \,\mathrm dr = 4\pi\sum_j r_j^2\,w_j\,\varphi_j`.

        Parameters
        ----------
        r_nodes, r_wts
            Composite GL radial nodes and weights on :math:`[r_0, R]`.
        phi
            Scalar flux sampled at ``r_nodes`` (shape matches).

        Returns
        -------
        float
            The shell-volume integral. Zero or near-zero indicates
            either trivial flux (eigenvector not yet normalised) or a
            quadrature pathology.
        """
        r_nodes = np.asarray(r_nodes, dtype=float)
        r_wts = np.asarray(r_wts, dtype=float)
        phi = np.asarray(phi, dtype=float)
        if self.kind == "slab-polar":
            return float(np.dot(r_wts, phi))
        if self.kind == "cylinder-1d":
            return float(2.0 * np.pi * np.dot(r_nodes * r_wts, phi))
        if self.kind == "sphere-1d":
            return float(
                4.0 * np.pi * np.dot(r_nodes * r_nodes * r_wts, phi)
            )
        raise ValueError(
            f"shell_volume_integral: unknown kind {self.kind!r}"
        )

    def reciprocity_factor(self, R_outer: float, r_inner: float) -> float:
        r"""Return the outer-to-inner area ratio for F.4 reciprocity:

        .. math::

           W_{oi} \;=\; \bigl(A_{\rm outer} / A_{\rm inner}\bigr)\,
           W_{io}.

        - **Cylinder** (``2\pi R`` per unit z): returns :math:`R/r_0`.
        - **Sphere** (``4\pi R^2``): returns :math:`(R/r_0)^2`.
        - **Slab-polar**: raises ``ValueError`` — slab has flat faces
          of equal area with :math:`W_{oi} = W_{io}` (no curvilinear
          cavity reciprocity). Callers should use the slab rank-2
          per-face closure directly.

        This codifies the reciprocity trap noted in the test
        ``test_hollow_cyl_transmission_zero_absorption_conservation``:
        the cylinder form (first power) must not be confused with the
        sphere form (squared).
        """
        if r_inner <= 0.0 or r_inner >= R_outer:
            raise ValueError(
                f"reciprocity_factor requires 0 < r_inner < R_outer; "
                f"got r_inner={r_inner}, R_outer={R_outer}"
            )
        if self.kind == "cylinder-1d":
            return R_outer / r_inner
        if self.kind == "sphere-1d":
            return (R_outer / r_inner) ** 2
        if self.kind == "slab-polar":
            raise ValueError(
                "reciprocity_factor is undefined for slab-polar — "
                "slab has flat equal-area faces. Use the slab rank-2 "
                "per-face closure directly."
            )
        raise ValueError(
            f"reciprocity_factor: unknown kind {self.kind!r}"
        )


# Convenience singletons
SLAB_POLAR_1D = CurvilinearGeometry(kind="slab-polar")
CYLINDER_1D = CurvilinearGeometry(kind="cylinder-1d")
SPHERE_1D = CurvilinearGeometry(kind="sphere-1d")



# ═══════════════════════════════════════════════════════════════════════
# Unified verification: adaptive mpmath.quad over polar form
# ═══════════════════════════════════════════════════════════════════════

def K_vol_element_adaptive(
    geometry: CurvilinearGeometry,
    i: int,
    j: int,
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    dps: int = 50,
) -> "mpmath.mpf":
    r"""Unified K-matrix element via adaptive :func:`mpmath.quad` over
    the polar form. **The single verification primitive** for slab,
    cylinder, and sphere Peierls operators (and, by extension, future
    2-D MoC verification).

    Computes

    .. math::

       K_{ij} \;=\; \Sigma_t(r_i)\,C_d \int_{\Omega_d}\!\!
                    W_\Omega(\Omega)\,\mathrm d\Omega
                    \int_0^{\rho_{\max}(r_i, \Omega)}\!\!
                    \kappa_d(\tau(r_i, \Omega, \rho))\,
                    L_j(r'(\rho, \Omega, r_i))\,\mathrm d\rho

    via nested adaptive :func:`mpmath.quad` with breakpoint hints at:

    - **Outer angular**: tangent-to-interior-boundary critical angles
      for cylinder / sphere; ``μ = 0`` split for slab (grazing-ray
      stiffness).
    - **Inner radial**: panel-boundary crossings of :math:`r'(\rho)`
      along the ray.

    Achieves machine precision uniformly across geometries; the only
    geometry-specific code is the four primitives (``ray_direction_cosine``,
    ``rho_max``, ``source_position``, ``optical_depth_along_ray``,
    ``volume_kernel_mp``) on :class:`CurvilinearGeometry`. Performance
    is intentionally not optimized — this is the verification reference,
    cached by callers as needed.

    Parameters
    ----------
    geometry
        Geometry instance providing the polar primitives.
    i, j
        Observer node index, source basis index.
    r_nodes, panel_bounds, radii, sig_t
        Standard Nyström inputs.
    dps
        mpmath working precision (decimal digits).

    Returns
    -------
    mpmath.mpf
        :math:`K[i, j]` at ``dps`` precision.
    """
    r_nodes_arr = np.asarray(r_nodes, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)
    sig_t_arr = np.asarray(sig_t, dtype=float)
    panel_bounds_f = [
        (float(pa), float(pb), int(i_start), int(i_end))
        for (pa, pb, i_start, i_end) in panel_bounds
    ]
    R = float(radii_arr[-1])

    r_i = float(r_nodes_arr[i])
    ki = geometry.which_annulus(r_i, radii_arr)
    sig_t_i = float(sig_t_arr[ki])

    panel_boundaries_r = sorted(
        {pa for (pa, pb, _, _) in panel_bounds_f}
        | {pb for (pa, pb, _, _) in panel_bounds_f}
    )
    interior_boundaries_r = np.array(
        [r for r in panel_boundaries_r if 0.0 < r < R],
        dtype=float,
    )

    omega_low, omega_high = geometry.angular_range
    pref = mpmath.mpf(geometry.prefactor)

    def integrand_rho(rho_mp, cos_om: float):
        rho_f = float(rho_mp)
        rho_max_val = float(geometry.rho_max(r_i, cos_om, R))
        if rho_f >= rho_max_val or rho_f <= 0:
            return mpmath.mpf(0)
        r_prime = float(geometry.source_position(r_i, rho_f, cos_om))
        if r_prime < 0.0 or r_prime > R:
            return mpmath.mpf(0)
        tau = float(geometry.optical_depth_along_ray(
            r_i, cos_om, rho_f, radii_arr, sig_t_arr,
        ))
        kappa = geometry.volume_kernel_mp(tau, dps)
        L_vals = lagrange_basis_on_panels(
            r_nodes_arr, panel_bounds_f, r_prime,
        )
        return mpmath.mpf(kappa) * mpmath.mpf(float(L_vals[j]))

    def outer_integrand(angular_mp):
        cos_om = float(geometry.ray_direction_cosine(
            np.array([float(angular_mp)]),
        )[0])
        rho_max_val = float(geometry.rho_max(r_i, cos_om, R))
        if rho_max_val <= 0.0:
            return mpmath.mpf(0)
        # Inner ρ subdivision at panel-boundary crossings.
        crossings = geometry.rho_crossings_for_ray(
            r_i, cos_om, rho_max_val, interior_boundaries_r,
        )
        breaks = [mpmath.mpf(0)]
        for rho in crossings:
            breaks.append(mpmath.mpf(float(rho)))
        breaks.append(mpmath.mpf(rho_max_val))
        inner = mpmath.quad(
            lambda rho: integrand_rho(rho, cos_om),
            breaks,
        )
        ang_factor = float(geometry.angular_weight(
            np.array([float(angular_mp)]),
        )[0])
        return mpmath.mpf(ang_factor) * inner

    with mpmath.workdps(dps):
        # Outer angular subdivision:
        #   Slab: split at μ = 0 (grazing-ray stiffness).
        #   Curvilinear: split at tangent angles to interior shells.
        if geometry.kind == "slab-polar":
            outer_breaks = [
                mpmath.mpf(omega_low),
                mpmath.mpf(0),
                mpmath.mpf(omega_high),
            ]
        else:
            tangent_angles = geometry.omega_tangent_angles(
                r_i, interior_boundaries_r,
            )
            outer_breaks = [mpmath.mpf(omega_low)]
            for ang in tangent_angles:
                outer_breaks.append(mpmath.mpf(float(ang)))
            outer_breaks.append(mpmath.mpf(omega_high))

        omega_integral = mpmath.quad(outer_integrand, outer_breaks)

    return mpmath.mpf(sig_t_i) * pref * omega_integral


def build_volume_kernel_adaptive(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    dps: int = 30,
) -> np.ndarray:
    r"""Assemble the K matrix by calling :func:`K_vol_element_adaptive`
    once per (i, j) pair. **The unified verification assembly** for all
    geometries.

    Returns float array (cast from mpmath at end). For mpmath-typed
    output, call :func:`K_vol_element_adaptive` directly per element.

    No ``n_angular`` / ``n_rho`` / ``n_phi`` parameters — adaptive
    quadrature self-determines node counts to reach machine precision.
    Performance scales as :math:`O(N^2 \cdot \text{cost per quad})`;
    not intended as a production hot path.
    """
    r_nodes_arr = np.asarray(r_nodes, dtype=float)
    N = len(r_nodes_arr)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = float(K_vol_element_adaptive(
                geometry, i, j, r_nodes, panel_bounds, radii, sig_t,
                dps=dps,
            ))
    return K




# ═══════════════════════════════════════════════════════════════════════
# Unified volume-kernel assembly
# ═══════════════════════════════════════════════════════════════════════

def build_volume_kernel(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_angular: int,
    n_rho: int,
    dps: int = 30,
    n_phi: int = 16,
) -> np.ndarray:
    r"""Assemble the Nyström volume kernel matrix for a single group.

    The unified operator (identity-:math:`\Sigma_t`-LHS form):

    .. math::

       \Sigma_t(r_i)\,\varphi_i
         \;=\; \sum_j K_{ij}\,q_j + S_{\rm bc}(r_i),
       \quad
       K_{ij} = \Sigma_t(r_i)\,C_d\!\sum_{k,m}
                  w_{\Omega,k}\,W_\Omega(\Omega_k)\,w_{\rho,m}(r_i,\Omega_k)\,
                  \kappa_d\bigl(\Sigma_t\,\rho_m\bigr)\,L_j(r'_{ikm}),

    with :math:`C_d` = :attr:`CurvilinearGeometry.prefactor`,
    :math:`W_\Omega(\Omega) = 1` (cylinder) or :math:`\sin\Omega`
    (sphere), :math:`\kappa_d(\tau) = \mathrm{Ki}_1(\tau)` (cylinder)
    or :math:`e^{-\tau}` (sphere), :math:`r'_{ikm} = r'(\rho_m,
    \Omega_k, r_i)`, and :math:`L_j` the piecewise-panel Lagrange
    basis.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)

    # Slab routes through the unified adaptive primitive — there is no
    # fast slab path in the active code base (the moment-form K assembly
    # has been archived per GitHub Issue #117 for future production
    # discrete-CP application). ``n_angular`` / ``n_rho`` / ``n_phi``
    # are accepted but ignored (mpmath.quad self-determines node count).
    if geometry.kind == "slab-polar":
        return build_volume_kernel_adaptive(
            geometry, r_nodes, panel_bounds, radii, sig_t, dps=dps,
        )

    N = len(r_nodes)
    R = float(radii[-1])

    omega_low, omega_high = geometry.angular_range
    ref_omega_nodes, ref_omega_wts = gl_nodes_weights(n_angular, dps)
    ref_omega_nodes = np.array([float(x) for x in ref_omega_nodes])
    ref_omega_wts = np.array([float(w) for w in ref_omega_wts])

    ref_rho_nodes, ref_rho_wts = gl_nodes_weights(n_rho, dps)
    ref_rho_nodes = np.array([float(x) for x in ref_rho_nodes])
    ref_rho_wts = np.array([float(w) for w in ref_rho_wts])

    # Sorted unique panel-boundary radii (kink locations of the Lagrange
    # basis along each ray). Excludes the outer boundary R because that
    # is always the ρ_max endpoint.
    panel_boundaries_r = np.array(sorted(
        {pa for (pa, pb, _, _) in panel_bounds}
        | {pb for (pa, pb, _, _) in panel_bounds}
    ), dtype=float)
    interior_boundaries_r = panel_boundaries_r[
        (panel_boundaries_r > 0.0) & (panel_boundaries_r < R)
    ]

    # Optical-length cap per ρ sub-interval: beyond τ_cap MFP the
    # exp-decay makes GL nodes waste resolution on the already-
    # negligible tail. Splitting at dyadic τ-breakpoints keeps each
    # sub-interval's integrand within ~1 order of magnitude. Matters
    # most for grazing rays in slab (ρ_max = L/|μ| → ∞ as μ → 0)
    # and for optically-thick curvilinear rays.
    tau_cap_per_subinterval = 4.0
    sig_t_max = float(np.max(sig_t)) if sig_t.size else 1.0
    rho_cap_per_subinterval = (
        tau_cap_per_subinterval / sig_t_max if sig_t_max > 0.0 else float("inf")
    )

    def _insert_tau_breakpoints(subintervals: list[float]) -> list[float]:
        """Split any sub-interval whose length exceeds ``rho_cap_per_subinterval``
        at dyadic τ-breakpoints (rho_cap, 2·rho_cap, 4·rho_cap, ...)
        measured from the left endpoint."""
        if rho_cap_per_subinterval == float("inf"):
            return subintervals
        out = [subintervals[0]]
        for rho_end in subintervals[1:]:
            rho_a = out[-1]
            cap = rho_cap_per_subinterval
            while rho_end - rho_a > cap:
                out.append(rho_a + cap)
                rho_a = out[-1]
                cap *= 2.0
            out.append(rho_end)
        return out

    K = np.zeros((N, N))
    pref = geometry.prefactor

    for i in range(N):
        r_i = r_nodes[i]
        ki = geometry.which_annulus(r_i, radii)
        sig_t_i = sig_t[ki]

        # Subdivide ω at tangent-to-interior-boundary critical angles.
        # For r_i > r_b, sin ω = r_b/r_i is the bifurcation where the
        # ρ-crossing count jumps 2→0; the outer integrand has a C¹
        # discontinuity at ω_c that fixed GL cannot integrate across.
        tangent_angles = geometry.omega_tangent_angles(
            r_i, interior_boundaries_r,
        )
        omega_subintervals = [omega_low, *tangent_angles, omega_high]

        for t_idx in range(len(omega_subintervals) - 1):
            om_a = omega_subintervals[t_idx]
            om_b = omega_subintervals[t_idx + 1]
            if om_b <= om_a:
                continue
            h_om = 0.5 * (om_b - om_a)
            m_om = 0.5 * (om_a + om_b)
            omega_pts = h_om * ref_omega_nodes + m_om
            omega_wts = h_om * ref_omega_wts
            cos_omegas = geometry.ray_direction_cosine(omega_pts)
            angular_factor = geometry.angular_weight(omega_pts)

            for k in range(n_angular):
                cos_om = cos_omegas[k]
                rho_max_val = geometry.rho_max(r_i, cos_om, R)
                if rho_max_val <= 0.0:
                    continue

                # Phase F.4: for a hollow cell, the ray's first-flight
                # stop is the inner shell r = r_0 if it intersects before
                # the outer boundary. Cap ρ accordingly — any contribution
                # beyond that is "after the ray escapes" and must not be
                # counted in the volumetric integral.
                if geometry.inner_radius > 0.0:
                    rho_in_minus, _ = geometry.rho_inner_intersections(
                        r_i, cos_om,
                    )
                    if (
                        rho_in_minus is not None
                        and rho_in_minus < rho_max_val
                    ):
                        rho_max_val = rho_in_minus

                # Subdivide ρ at panel-boundary crossings — without this,
                # the Lagrange-basis kinks along the ray are unresolved
                # and the fixed-order GL rule leaves ~1–5% error per
                # K[i,j]. See issue #114. Then also cap any sub-interval
                # whose optical length exceeds τ_cap (handles slab
                # grazing rays and optically-thick curvilinear rays).
                crossings = geometry.rho_crossings_for_ray(
                    r_i, cos_om, rho_max_val, interior_boundaries_r,
                )
                rho_subintervals = _insert_tau_breakpoints(
                    [0.0, *crossings, rho_max_val]
                )

                outer_weight = (
                    pref * sig_t_i * omega_wts[k] * angular_factor[k]
                )
                for s_idx in range(len(rho_subintervals) - 1):
                    rho_a = rho_subintervals[s_idx]
                    rho_b = rho_subintervals[s_idx + 1]
                    if rho_b <= rho_a:
                        continue
                    h_r = 0.5 * (rho_b - rho_a)
                    m_r = 0.5 * (rho_a + rho_b)
                    rho_pts = h_r * ref_rho_nodes + m_r
                    rho_wts = h_r * ref_rho_wts

                    for m in range(n_rho):
                        rho = rho_pts[m]
                        r_prime = geometry.source_position(r_i, rho, cos_om)
                        # Skip source contributions from inside the
                        # cavity of a hollow cell (Phase F.4): the cavity
                        # is void and carries no source; the Lagrange
                        # basis over annular nodes would otherwise
                        # extrapolate a spurious contribution.
                        if (
                            geometry.inner_radius > 0.0
                            and float(r_prime) < geometry.inner_radius
                        ):
                            continue
                        tau = geometry.optical_depth_along_ray(
                            r_i, cos_om, rho, radii, sig_t,
                        )
                        kappa = geometry.volume_kernel_mp(tau, dps)
                        L_vals = lagrange_basis_on_panels(
                            r_nodes, panel_bounds, float(r_prime),
                        )
                        weight = outer_weight * rho_wts[m] * kappa
                        K[i, :] += weight * L_vals

    return K


# ═══════════════════════════════════════════════════════════════════════
# Unified white-BC rank-1 closure
# ═══════════════════════════════════════════════════════════════════════

def compute_P_esc(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_angular: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Uncollided escape probability :math:`P_{\rm esc}(r_i)` via

    .. math::

       P_{\rm esc}(r_i)
         = \frac{1}{S_d^{\rm reduced}}\,\int_0^\pi W_\Omega(\Omega)\,
           K_{\rm esc}\!\bigl(\tau(r_i,\rho_{\max}(r_i,\Omega),\Omega)\bigr)\,
           \mathrm d\Omega

    where :math:`K_{\rm esc}` is :meth:`CurvilinearGeometry.escape_kernel_mp`
    and the leading factor is :attr:`CurvilinearGeometry.prefactor`
    (which already absorbs the azimuthal fold).
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])

    if geometry.kind == "slab-polar" and len(radii) == 1:
        # Slab, homogeneous single region: P_esc reduces to the
        # closed-form E_2 sum
        #
        #     P_esc_slab(x_i) = (1/2) · [E_2(Σ_t·x_i) + E_2(Σ_t·(L − x_i))]
        #
        # (escape through face 0 at µ < 0, through face L at µ > 0;
        # each half-range contributes an E_2). Closed-form is
        # machine-precision, bypassing the GL grazing-ray stiffness at
        # µ = 0 that limits the unified quadrature branch.
        L = float(radii[-1])
        sig_t_val = float(sig_t[0])
        N = len(r_nodes)
        P_esc = np.zeros(N)
        for i in range(N):
            x_i = float(r_nodes[i])
            tau_left = sig_t_val * x_i
            tau_right = sig_t_val * (L - x_i)
            e2_left = (float(mpmath.expint(2, tau_left))
                       if tau_left > 0 else 1.0)
            e2_right = (float(mpmath.expint(2, tau_right))
                        if tau_right > 0 else 1.0)
            P_esc[i] = 0.5 * (e2_left + e2_right)
        return P_esc

    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    # Use the polymorphic direction-cosine map: identity for slab-polar
    # (angular variable IS µ), cos(Ω) for curvilinear (angular variable
    # is the polar angle Ω).
    cos_omegas = geometry.ray_direction_cosine(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)
    pref = geometry.prefactor

    N = len(r_nodes)
    P_esc = np.zeros(N)
    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_max_val = geometry.rho_max(r_i, cos_om, R)
            if rho_max_val <= 0.0:
                continue
            tau = geometry.optical_depth_along_ray(
                r_i, cos_om, rho_max_val, radii, sig_t,
            )
            K_esc = geometry.escape_kernel_mp(tau, dps)
            total += omega_wts[k] * angular_factor[k] * K_esc
        P_esc[i] = pref * total
    return P_esc


def compute_G_bc(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Surface-to-volume Green's function :math:`G_{\rm bc}(r_i)` for
    unit uniform isotropic-inward re-entry.

    Geometry-specific integral:

    - **Cylinder:** :math:`G_{\rm bc}(r_i) = (2R/\pi)\!\int_0^\pi
      \mathrm{Ki}_1(\tau_{\rm surf}(r_i,\phi))/d(r_i,R,\phi)\,
      \mathrm d\phi` where :math:`d = \sqrt{r_i^2 + R^2 - 2 r_i R
      \cos\phi}`.

    - **Sphere:**  :math:`G_{\rm bc}(r_i) = 2\!\int_0^\pi \sin\theta\,
      e^{-\tau(r_i,\rho_{\max}(r_i,\theta))}\,\mathrm d\theta`.

      Observer-centred ray parametrisation. For a uniform isotropic
      inward partial current :math:`J^{-}` on the sphere, the
      angular flux inside is :math:`\psi_{\rm in} = J^{-}/\pi` (since
      :math:`J^{-} = \pi\,\psi_{\rm in}` for an isotropic inward
      hemisphere). The scalar flux at observer :math:`r_i` is
      :math:`\psi_{\rm in}\,\int_{4\pi}e^{-\tau}\,\mathrm d\Omega
      = (J^{-}/\pi)\,\cdot 2\pi\,\int_0^\pi\sin\theta\,e^{-\tau}\,
      \mathrm d\theta`, dividing by :math:`J^{-}` gives the
      prefactor :math:`2`. No Jacobian :math:`1/d^{2}` appears
      because we integrate over *directions at the observer* rather
      than *area on the surface* (the two forms are equivalent via a
      change of variables; the observer form avoids the
      :math:`\cos\theta'` Lambertian weight).
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])

    N = len(r_nodes)
    G_bc = np.zeros(N)

    if geometry.kind == "slab-polar":
        # Slab has TWO faces (x = 0 and x = L); the scalar flux at
        # interior x_i from UNIT uniform isotropic inward partial
        # currents at *each* face is
        #
        #   G_bc_slab(x_i) = 2·[E_2(Σ_t·x_i) + E_2(Σ_t·(L - x_i))]
        #
        # for a homogeneous slab. The factor 2 comes from
        # :math:`\psi_{\rm in} = 2 J^{-}` for a half-range isotropic
        # partial current. Each E_2 term is one face's contribution;
        # :func:`mpmath.expint` gives machine precision.
        L = R  # for slab, radii[-1] IS L
        if len(radii) == 1:
            sig_t_val = float(sig_t[0])
            for i in range(N):
                x_i = float(r_nodes[i])
                tau_left = sig_t_val * x_i
                tau_right = sig_t_val * (L - x_i)
                e2_left = (float(mpmath.expint(2, tau_left))
                           if tau_left > 0 else 1.0)
                e2_right = (float(mpmath.expint(2, tau_right))
                            if tau_right > 0 else 1.0)
                G_bc[i] = 2.0 * (e2_left + e2_right)
            return G_bc

        # Multi-region slab: GL over the inward half-range µ ∈ (0, 1)
        # for each face, with τ computed by the polymorphic ray walker
        # (handles piecewise Σ_t crossings).
        mu_pts, mu_wts = gl_float(n_surf_quad, 0.0, 1.0, dps)
        for i in range(N):
            x_i = float(r_nodes[i])
            total_0 = 0.0
            total_L = 0.0
            for k in range(n_surf_quad):
                mu = float(mu_pts[k])
                if mu <= 1e-15:
                    continue
                # Face 0: ray enters at (x=0) in direction +µ, observer
                # at x_i means ray length ρ = x_i/µ.
                rho_0 = x_i / mu
                tau_0 = geometry.optical_depth_along_ray(
                    0.0, mu, rho_0, radii, sig_t,
                )
                total_0 += float(mu_wts[k]) * 2.0 * float(np.exp(-tau_0))
                # Face L: ray enters at (x=L) in direction -µ, observer
                # at x_i means ray length ρ = (L - x_i)/µ.
                rho_L = (L - x_i) / mu
                tau_L = geometry.optical_depth_along_ray(
                    L, -mu, rho_L, radii, sig_t,
                )
                total_L += float(mu_wts[k]) * 2.0 * float(np.exp(-tau_L))
            G_bc[i] = total_0 + total_L
        return G_bc

    if geometry.kind == "sphere-1d":
        # Observer-centred angular integral.
        theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
        cos_thetas = np.cos(theta_pts)
        sin_thetas = np.sin(theta_pts)

        for i in range(N):
            r_i = r_nodes[i]
            total = 0.0
            for k in range(n_surf_quad):
                ct = cos_thetas[k]
                st = sin_thetas[k]
                rho_to_surface = geometry.rho_max(r_i, ct, R)
                if rho_to_surface <= 0.0:
                    continue
                if len(radii) == 1:
                    tau = sig_t[0] * rho_to_surface
                else:
                    tau = geometry.optical_depth_along_ray(
                        r_i, ct, rho_to_surface, radii, sig_t,
                    )
                total += theta_wts[k] * st * float(np.exp(-tau))
            G_bc[i] = 2.0 * total
        return G_bc

    # Cylinder: surface-centred form, Ki_1/d kernel.
    phi_pts, phi_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_phis = np.cos(phi_pts)
    sin_phis = np.sin(phi_pts)
    inv_pi = 1.0 / np.pi

    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_surf_quad):
            cf = cos_phis[k]
            d_sq = r_i * r_i + R * R - 2.0 * r_i * R * cf
            d = np.sqrt(max(d_sq, 0.0))
            if d <= 0.0:
                continue
            if len(radii) == 1:
                tau = sig_t[0] * d
            else:
                cb = (R * cf - r_i) / d
                sb = R * sin_phis[k] / d  # noqa: F841 (kept for symmetry)
                tau = geometry.optical_depth_along_ray(
                    r_i, cb, d, radii, sig_t,
                )
            ki1 = float(ki_n_mp(1, float(tau), dps))
            total += phi_wts[k] * ki1 / d
        G_bc[i] = 2.0 * inv_pi * R * total
    return G_bc


# ═══════════════════════════════════════════════════════════════════════
# Per-surface escape / response primitives (Phase F.2)
#
# The single-surface :func:`compute_P_esc` / :func:`compute_G_bc`
# aggregate escape-to-any-boundary / response-from-all-boundaries.
# Phase F splits these into per-face primitives so that
# :class:`BoundaryClosureOperator` can carry a per-face mode space
# ``A = ℝ^(N_modes × N_surfaces)`` and couple independent partial
# currents at each boundary via a block-diagonal reflection ``R``.
#
# For solid cyl/sph (``inner_radius == 0``) the ``_inner`` variants
# return zero arrays — regime-A bit-exact contract. For slab, the
# "outer" surface is face ``x = L`` and "inner" is face ``x = 0`` (the
# faces are positional, not parametrised by inner_radius). For hollow
# cyl/sph (``inner_radius > 0``) the cavity segment contributes zero
# to τ (handled by :meth:`CurvilinearGeometry.optical_depth_along_ray`).
# ═══════════════════════════════════════════════════════════════════════


def _slab_E2(tau: float) -> float:
    """Closed-form :math:`E_2(\\tau)` at machine precision via mpmath."""
    if tau > 0.0:
        return float(mpmath.expint(2, tau))
    return 1.0


def compute_P_esc_outer(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_angular: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Uncollided escape probability to the **outer** boundary.

    For slab, this is the face at :math:`x = L`:
    :math:`P_{\rm esc,L}(x_i) = \tfrac{1}{2}\,E_2(\Sigma_t(L - x_i))`
    (homogeneous single region — closed form at machine precision).

    For cylinder/sphere (solid or hollow), this integrates
    :math:`K_{\rm esc}` over all directions from the observer, with
    τ taken along the full path to :math:`\rho_{\max}(r_i, \Omega, R)`.
    If the ray traverses the cavity (hollow case), the cavity segment
    contributes zero to τ (via
    :meth:`CurvilinearGeometry.optical_depth_along_ray`).
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    N = len(r_nodes)

    if geometry.kind == "slab-polar" and len(radii) == 1:
        # Slab homogeneous: face L closed form.
        L = float(radii[-1])
        sig_t_val = float(sig_t[0])
        P = np.zeros(N)
        for i in range(N):
            x_i = float(r_nodes[i])
            P[i] = 0.5 * _slab_E2(sig_t_val * (L - x_i))
        return P

    # Multi-region slab and curvilinear: GL over angular range.
    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = geometry.ray_direction_cosine(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)
    pref = geometry.prefactor

    P = np.zeros(N)
    is_slab = geometry.kind == "slab-polar"
    is_hollow = (not is_slab) and geometry.inner_radius > 0.0
    for i in range(N):
        r_i = float(r_nodes[i])
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_to_outer = geometry.rho_max(r_i, cos_om, R)
            if rho_to_outer <= 0.0:
                continue
            if is_slab:
                # Slab outer = face at x=L, reached only for µ > 0.
                if cos_om <= 0.0:
                    continue
            if is_hollow:
                # Hollow cyl/sph (Phase F.4): rays that strike the inner
                # shell first exit through the inner boundary and are
                # counted by compute_P_esc_inner; exclude them here so
                # that P_esc_outer + P_esc_inner sum to the total escape
                # probability without double-counting.
                rho_in_minus, _ = geometry.rho_inner_intersections(
                    r_i, cos_om,
                )
                if rho_in_minus is not None and rho_in_minus < rho_to_outer:
                    continue
            tau = geometry.optical_depth_along_ray(
                r_i, cos_om, rho_to_outer, radii, sig_t,
            )
            K_esc = geometry.escape_kernel_mp(tau, dps)
            total += omega_wts[k] * angular_factor[k] * K_esc
        P[i] = pref * total
    return P


def compute_P_esc_inner(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_angular: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Uncollided escape probability to the **inner** boundary.

    Solid geometry (``inner_radius == 0`` and not slab) has no inner
    boundary and returns a zero array — the regime-A sentinel.

    For slab, this is the face at :math:`x = 0`:
    :math:`P_{\rm esc,0}(x_i) = \tfrac{1}{2}\,E_2(\Sigma_t x_i)`.

    For hollow cyl/sph, integrates :math:`K_{\rm esc}` over those
    directions where the ray reaches the inner shell before the outer
    (:math:`\rho^-_{\rm in}` exists and :math:`< \rho_{\max}`), with τ
    taken along the annulus path from :math:`r_i` to :math:`\rho^-_{\rm in}`
    (cavity not yet entered).
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    N = len(r_nodes)

    if geometry.n_surfaces == 1:
        # Solid cyl/sph: regime-A zero-array sentinel.
        return np.zeros(N)

    if geometry.kind == "slab-polar" and len(radii) == 1:
        # Slab homogeneous: face 0 closed form.
        sig_t_val = float(sig_t[0])
        P = np.zeros(N)
        for i in range(N):
            x_i = float(r_nodes[i])
            P[i] = 0.5 * _slab_E2(sig_t_val * x_i)
        return P

    # Multi-region slab: GL over µ ∈ (-1, 0) for face 0.
    if geometry.kind == "slab-polar":
        omega_pts, omega_wts = gl_float(n_angular, -1.0, 1.0, dps)
        cos_omegas = omega_pts
        pref = geometry.prefactor
        P = np.zeros(N)
        for i in range(N):
            x_i = float(r_nodes[i])
            total = 0.0
            for k in range(n_angular):
                mu = cos_omegas[k]
                if mu >= 0.0:
                    continue
                rho = -x_i / mu  # ρ = (0 − x_i)/µ for µ<0
                if rho <= 0.0:
                    continue
                tau = geometry.optical_depth_along_ray(
                    x_i, mu, rho, radii, sig_t,
                )
                K_esc = geometry.escape_kernel_mp(tau, dps)
                total += omega_wts[k] * K_esc
            P[i] = pref * total
        return P

    # Hollow cyl/sph: GL over angular range, restricted to directions
    # that hit the inner shell.
    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = geometry.ray_direction_cosine(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)
    pref = geometry.prefactor

    P = np.zeros(N)
    for i in range(N):
        r_i = float(r_nodes[i])
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, cos_om)
            if rho_in_minus is None:
                continue
            # τ along the ray from r_i to the first inner intersection
            # (annulus path only — cavity not yet entered).
            tau = geometry.optical_depth_along_ray(
                r_i, cos_om, rho_in_minus, radii, sig_t,
            )
            K_esc = geometry.escape_kernel_mp(tau, dps)
            total += omega_wts[k] * angular_factor[k] * K_esc
        P[i] = pref * total
    return P


def compute_P_esc_outer_mode(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_angular: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Mode-:math:`n` outgoing moment at the **outer** surface
    (Phase F.5 / Issue #119).

    For :math:`n = 0` returns the scalar
    :func:`compute_P_esc_outer` (Mark convention, no
    :math:`(\rho/R)^2` Jacobian); for :math:`n \ge 1` uses the
    canonical Gelbard DP\ :sub:`N-1` form

    .. math::

       P_{\rm esc, out}^{(n)}(r_i)
         = C_d \!\int W_\Omega\,\Bigl(\tfrac{\rho_{\rm out}}{R}\Bigr)^2
                 \tilde P_n(\mu_{\rm exit, out})\,
                 K_{\rm esc}(\tau)\,\mathrm d\Omega

    with :math:`\mu_{\rm exit, out} = (\rho_{\rm out} + r_i\cos\Omega)/R`.

    For hollow cells, rays that would hit the inner shell first are
    excluded (Model A — those contribute to :func:`compute_P_esc_inner_mode`).

    Currently implemented for ``kind = "sphere-1d"`` with
    ``inner_radius > 0``; slab and cylinder raise
    :class:`NotImplementedError`.
    """
    if n_mode == 0:
        return compute_P_esc_outer(
            geometry, r_nodes, radii, sig_t,
            n_angular=n_angular, dps=dps,
        )
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"compute_P_esc_outer_mode(n>=1) is implemented for "
            f"hollow sphere only; got kind={geometry.kind!r}, "
            f"inner_radius={geometry.inner_radius}. Slab and cyl "
            f"per-face mode primitives — Issue #119 follow-up."
        )
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = geometry.ray_direction_cosine(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)
    pref = geometry.prefactor
    N = len(r_nodes)
    P = np.zeros(N)
    # Phase F.5 convention: Mark-Lambert angular measure (sin θ dθ) with
    # P̃_n(µ_exit) Legendre factor — NO (ρ/R)² Jacobian, for consistency
    # with the W transmission matrix (which also uses no Jacobian).
    # This means mode-0 is exactly compute_P_esc_outer (above), and
    # mode-n≥1 is the direct Lambert-Legendre moment.
    for i in range(N):
        r_i = float(r_nodes[i])
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_out = geometry.rho_max(r_i, cos_om, R)
            if rho_out <= 0.0:
                continue
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, cos_om)
            if rho_in_minus is not None and rho_in_minus < rho_out:
                continue
            tau = geometry.optical_depth_along_ray(
                r_i, cos_om, rho_out, radii, sig_t,
            )
            K_esc = geometry.escape_kernel_mp(tau, dps)
            mu_exit = (rho_out + r_i * cos_om) / R
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_exit]))[0]
            )
            total += (
                omega_wts[k] * angular_factor[k] * p_tilde * K_esc
            )
        P[i] = pref * total
    return P


def compute_P_esc_inner_mode(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_angular: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Mode-:math:`n` outgoing moment at the **inner** surface.

    For :math:`n = 0` returns the scalar
    :func:`compute_P_esc_inner`; for :math:`n \ge 1`:

    .. math::

       P_{\rm esc, in}^{(n)}(r_i)
         = C_d \!\int W_\Omega\,\Bigl(\tfrac{\rho_{\rm in}^-}{r_0}\Bigr)^2
                 \tilde P_n(\mu_{\rm exit, in})\,
                 K_{\rm esc}(\tau)\,\mathrm d\Omega

    restricted to directions for which the ray hits the inner shell
    before the outer; :math:`\mu_{\rm exit, in} = \sqrt{r_0^2 - h^2}/r_0`
    with :math:`h = r_i|\sin\Omega|`.

    Currently implemented for ``kind = "sphere-1d"`` with
    ``inner_radius > 0``.
    """
    if n_mode == 0:
        return compute_P_esc_inner(
            geometry, r_nodes, radii, sig_t,
            n_angular=n_angular, dps=dps,
        )
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"compute_P_esc_inner_mode(n>=1) for "
            f"kind={geometry.kind!r} not yet implemented."
        )
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    r_0 = float(geometry.inner_radius)
    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = geometry.ray_direction_cosine(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)
    pref = geometry.prefactor
    N = len(r_nodes)
    P = np.zeros(N)
    # Phase F.5 convention: Mark-Lambert weighting (no Jacobian).
    for i in range(N):
        r_i = float(r_nodes[i])
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, cos_om)
            if rho_in_minus is None:
                continue
            tau = geometry.optical_depth_along_ray(
                r_i, cos_om, rho_in_minus, radii, sig_t,
            )
            K_esc = geometry.escape_kernel_mp(tau, dps)
            sin_om = np.sqrt(max(0.0, 1.0 - cos_om * cos_om))
            h_sq = r_i * r_i * sin_om * sin_om
            mu_exit_sq = max(0.0, (r_0 * r_0 - h_sq) / (r_0 * r_0))
            mu_exit = float(np.sqrt(mu_exit_sq))
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_exit]))[0]
            )
            total += (
                omega_wts[k] * angular_factor[k] * p_tilde * K_esc
            )
        P[i] = pref * total
    return P


def compute_G_bc_outer(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Scalar flux at :math:`r_i` from a unit uniform-isotropic inward
    partial current on the **outer** boundary.

    For slab, this is the face at :math:`x = L`:
    :math:`G_{\rm bc,L}(x_i) = 2\,E_2(\Sigma_t(L - x_i))`.

    For cylinder/sphere, the legacy observer-centred / surface-centred
    integrals apply, with τ including cavity-segment zero-attenuation
    for hollow cells.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    N = len(r_nodes)

    if geometry.kind == "slab-polar" and len(radii) == 1:
        L = R
        sig_t_val = float(sig_t[0])
        G = np.zeros(N)
        for i in range(N):
            x_i = float(r_nodes[i])
            G[i] = 2.0 * _slab_E2(sig_t_val * (L - x_i))
        return G

    if geometry.kind == "slab-polar":
        # Multi-region slab: GL over µ ∈ (0, 1) for face L.
        L = R
        mu_pts, mu_wts = gl_float(n_surf_quad, 0.0, 1.0, dps)
        G = np.zeros(N)
        for i in range(N):
            x_i = float(r_nodes[i])
            total = 0.0
            for k in range(n_surf_quad):
                mu = float(mu_pts[k])
                if mu <= 1e-15:
                    continue
                rho_L = (L - x_i) / mu
                tau_L = geometry.optical_depth_along_ray(
                    L, -mu, rho_L, radii, sig_t,
                )
                total += float(mu_wts[k]) * 2.0 * float(np.exp(-tau_L))
            G[i] = total
        return G

    if geometry.kind == "sphere-1d":
        # Observer-centred angular integral. rho_to_surface is the full
        # path to the outer boundary; cavity handling lives in
        # optical_depth_along_ray.
        theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
        cos_thetas = np.cos(theta_pts)
        sin_thetas = np.sin(theta_pts)
        G = np.zeros(N)
        for i in range(N):
            r_i = r_nodes[i]
            total = 0.0
            for k in range(n_surf_quad):
                ct = cos_thetas[k]
                st = sin_thetas[k]
                rho_to_surface = geometry.rho_max(r_i, ct, R)
                if rho_to_surface <= 0.0:
                    continue
                if len(radii) == 1 and geometry.inner_radius == 0.0:
                    tau = sig_t[0] * rho_to_surface
                else:
                    tau = geometry.optical_depth_along_ray(
                        r_i, ct, rho_to_surface, radii, sig_t,
                    )
                total += theta_wts[k] * st * float(np.exp(-tau))
            G[i] = 2.0 * total
        return G

    # Cylinder outer — surface-centred, Ki_1/d kernel.
    phi_pts, phi_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_phis = np.cos(phi_pts)
    sin_phis = np.sin(phi_pts)
    inv_pi = 1.0 / np.pi
    G = np.zeros(N)
    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_surf_quad):
            cf = cos_phis[k]
            d_sq = r_i * r_i + R * R - 2.0 * r_i * R * cf
            d = float(np.sqrt(max(d_sq, 0.0)))
            if d <= 0.0:
                continue
            if len(radii) == 1 and geometry.inner_radius == 0.0:
                tau = sig_t[0] * d
            else:
                cb = (R * cf - r_i) / d
                _ = R * sin_phis[k] / d  # symmetry only
                tau = geometry.optical_depth_along_ray(
                    r_i, cb, d, radii, sig_t,
                )
            ki1 = float(ki_n_mp(1, float(tau), dps))
            total += phi_wts[k] * ki1 / d
        G[i] = 2.0 * inv_pi * R * total
    return G


def compute_G_bc_inner(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Scalar flux at :math:`r_i` from a unit uniform-isotropic outward
    partial current on the **inner** boundary.

    Solid geometry returns a zero array (regime-A sentinel).

    For slab, this is the face at :math:`x = 0`:
    :math:`G_{\rm bc,0}(x_i) = 2\,E_2(\Sigma_t x_i)`.

    For hollow sphere, observer-centred integration restricted to those
    directions where the sightline to :math:`r_0` exists (ray hits the
    inner shell). For hollow cylinder, surface-centred form from
    :math:`r_0`, analogous to the outer ``Ki_1/d`` formula but with the
    chord length and bearing from the inner surface.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    N = len(r_nodes)

    if geometry.n_surfaces == 1:
        return np.zeros(N)

    if geometry.kind == "slab-polar" and len(radii) == 1:
        sig_t_val = float(sig_t[0])
        G = np.zeros(N)
        for i in range(N):
            x_i = float(r_nodes[i])
            G[i] = 2.0 * _slab_E2(sig_t_val * x_i)
        return G

    if geometry.kind == "slab-polar":
        # Multi-region slab: GL over µ ∈ (0, 1) for face 0
        # (ray enters at x=0 with direction +µ, observer at x_i has ρ=x_i/µ).
        mu_pts, mu_wts = gl_float(n_surf_quad, 0.0, 1.0, dps)
        G = np.zeros(N)
        for i in range(N):
            x_i = float(r_nodes[i])
            total = 0.0
            for k in range(n_surf_quad):
                mu = float(mu_pts[k])
                if mu <= 1e-15:
                    continue
                rho_0 = x_i / mu
                tau_0 = geometry.optical_depth_along_ray(
                    0.0, mu, rho_0, radii, sig_t,
                )
                total += float(mu_wts[k]) * 2.0 * float(np.exp(-tau_0))
            G[i] = total
        return G

    r0 = float(geometry.inner_radius)

    if geometry.kind == "sphere-1d":
        # Observer-centred, directions that hit the inner shell.
        theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
        cos_thetas = np.cos(theta_pts)
        sin_thetas = np.sin(theta_pts)
        G = np.zeros(N)
        for i in range(N):
            r_i = r_nodes[i]
            total = 0.0
            for k in range(n_surf_quad):
                ct = cos_thetas[k]
                st = sin_thetas[k]
                rho_in_minus, _ = geometry.rho_inner_intersections(r_i, ct)
                if rho_in_minus is None:
                    continue
                if len(radii) == 1:
                    tau = sig_t[0] * rho_in_minus
                else:
                    tau = geometry.optical_depth_along_ray(
                        r_i, ct, rho_in_minus, radii, sig_t,
                    )
                total += theta_wts[k] * st * float(np.exp(-tau))
            G[i] = 2.0 * total
        return G

    # Cylinder inner — surface-centred on r=r_0, Ki_1/d_inner kernel.
    phi_pts, phi_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_phis = np.cos(phi_pts)
    sin_phis = np.sin(phi_pts)
    inv_pi = 1.0 / np.pi
    G = np.zeros(N)
    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_surf_quad):
            cf = cos_phis[k]
            d_sq = r_i * r_i + r0 * r0 - 2.0 * r_i * r0 * cf
            d = float(np.sqrt(max(d_sq, 0.0)))
            if d <= 0.0:
                continue
            if len(radii) == 1:
                tau = sig_t[0] * d
            else:
                # Direction cosine from observer at r_i to a surface point
                # on r=r_0. Mirror of the outer-surface formula with the
                # inner radius substituted.
                cb = (r0 * cf - r_i) / d
                _ = r0 * sin_phis[k] / d
                tau = geometry.optical_depth_along_ray(
                    r_i, cb, d, radii, sig_t,
                )
            ki1 = float(ki_n_mp(1, float(tau), dps))
            total += phi_wts[k] * ki1 / d
        G[i] = 2.0 * inv_pi * r0 * total
    return G


def compute_G_bc_outer_mode(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Mode-:math:`n` response at observer :math:`r_i` from a unit
    mode-:math:`n` uniform incoming current on the **outer** surface
    (Phase F.5 / Issue #119).

    For :math:`n = 0` returns :func:`compute_G_bc_outer`; for
    :math:`n \ge 1` uses the observer-centred form with the
    :math:`\tilde P_n(\mu_s)` surface-Legendre weight at the surface
    emission point.

    Currently implemented for ``kind = "sphere-1d"`` with
    ``inner_radius > 0``. Rays that would pass through the cavity
    (first-flight hitting inner shell on the way back toward observer
    from outer surface point) are excluded — Model A first-flight.
    """
    if n_mode == 0:
        return compute_G_bc_outer(
            geometry, r_nodes, radii, sig_t,
            n_surf_quad=n_surf_quad, dps=dps,
        )
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"compute_G_bc_outer_mode(n>=1) for kind="
            f"{geometry.kind!r} not yet implemented."
        )
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_thetas = np.cos(theta_pts)
    sin_thetas = np.sin(theta_pts)
    N = len(r_nodes)
    G = np.zeros(N)
    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_surf_quad):
            ct = cos_thetas[k]
            st = sin_thetas[k]
            rho_out = geometry.rho_max(r_i, ct, R)
            if rho_out <= 0.0:
                continue
            # Model A: skip rays blocked by inner shell.
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, ct)
            if rho_in_minus is not None and rho_in_minus < rho_out:
                continue
            # τ along the observer→outer chord.
            if len(radii) == 1:
                tau = sig_t[0] * rho_out
            else:
                tau = geometry.optical_depth_along_ray(
                    r_i, ct, rho_out, radii, sig_t,
                )
            # µ_s at the outer surface point (local inward-normal frame
            # at emission): |µ_s| = (ρ_out + r·cos θ)/R — the same
            # mu_exit formula as P_esc, since chord symmetry at outer
            # maps µ_out (emission) to µ_in (arrival) identically.
            mu_s = (rho_out + r_i * ct) / R
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_s]))[0]
            )
            total += theta_wts[k] * st * p_tilde * float(np.exp(-tau))
        G[i] = 2.0 * total
    return G


def compute_G_bc_inner_mode(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Mode-:math:`n` response at observer :math:`r_i` from a unit
    mode-:math:`n` uniform outward current on the **inner** surface.

    For :math:`n = 0` returns :func:`compute_G_bc_inner`; for
    :math:`n \ge 1`:

    .. math::

       G_{\rm bc, in}^{(n)}(r_i) = 2\!\int_{\rm hit} \sin\theta\,
           \tilde P_n(\mu_s)\,e^{-\tau}\,\mathrm d\theta

    with the sightline-from-observer integration over rays reaching
    the inner shell; :math:`\mu_s = \sqrt{r_0^2 - h^2}/r_0` is the
    local µ at the inner surface emission point, matching the
    :math:`\mu_{\rm exit}` derived for :func:`compute_P_esc_inner_mode`.
    """
    if n_mode == 0:
        return compute_G_bc_inner(
            geometry, r_nodes, radii, sig_t,
            n_surf_quad=n_surf_quad, dps=dps,
        )
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"compute_G_bc_inner_mode(n>=1) for kind="
            f"{geometry.kind!r} not yet implemented."
        )
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    r_0 = float(geometry.inner_radius)
    theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_thetas = np.cos(theta_pts)
    sin_thetas = np.sin(theta_pts)
    N = len(r_nodes)
    G = np.zeros(N)
    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_surf_quad):
            ct = cos_thetas[k]
            st = sin_thetas[k]
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, ct)
            if rho_in_minus is None:
                continue
            if len(radii) == 1:
                tau = sig_t[0] * rho_in_minus
            else:
                tau = geometry.optical_depth_along_ray(
                    r_i, ct, rho_in_minus, radii, sig_t,
                )
            # µ_s at the inner surface emission point.
            sin_om = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
            h_sq = r_i * r_i * sin_om * sin_om
            mu_s_sq = max(0.0, (r_0 * r_0 - h_sq) / (r_0 * r_0))
            mu_s = float(np.sqrt(mu_s_sq))
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_s]))[0]
            )
            total += theta_wts[k] * st * p_tilde * float(np.exp(-tau))
        G[i] = 2.0 * total
    return G


# ═══════════════════════════════════════════════════════════════════════
# Marshak partial-current-moment per-face primitives (Phase F.5, Issue #119)
# ═══════════════════════════════════════════════════════════════════════
#
# These variants add the partial-current weight µ to the integrand of
# both P_esc and G_bc for every mode (including n = 0). They live in
# the same moment basis as :func:`compute_hollow_sph_transmission_rank_n`
# (which has `cos θ · sin θ · P̃_m · P̃_n · e^{-τ}` — i.e. µ dµ with
# P̃_n(µ) test functions). Using a consistent basis across P, G, and W
# is the fix for the Phase F.5 closure failure at N ≥ 2 identified in
# Issue #119: the Lambert / angular-flux primitives have a different
# inner-product Gram matrix than the Marshak / partial-current ones
# that W sits in, so the matrix product `G (I − W)⁻¹ P` couples
# incompatible bases at rank > 1.
#
# At N = 1 these new primitives do NOT reduce bit-exactly to the
# scalar :func:`compute_P_esc_outer` etc. — the scalar primitives are
# Mark-Lambert (no µ weight). The rank-2 Phase F.4 assembly
# (:func:`_build_closure_operator_rank2_white`) keeps the scalar
# primitives, so Phase F.4 regression remains bit-exact; the rank-N
# (N ≥ 2) assembly in
# :func:`_build_closure_operator_rank_n_white` uses these Marshak
# variants instead.


def compute_P_esc_outer_mode_marshak(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_angular: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Marshak partial-current moment :math:`n` of the escape
    probability to the **outer** surface (Phase F.5 / Issue #119).

    .. math::

       P_{\rm esc, out, marshak}^{(n)}(r_i)
         = C_d \!\int_{\rm rays\ reaching\ outer}\!
                 W_\Omega\,\mu_{\rm exit}\,\tilde P_n(\mu_{\rm exit})
                 \,K_{\rm esc}(\tau)\,\mathrm d\Omega

    with :math:`\mu_{\rm exit} = (\rho_{\rm out} + r_i\cos\Omega)/R`,
    :math:`C_d =` :attr:`CurvilinearGeometry.prefactor`, and
    :math:`W_\Omega =` :meth:`CurvilinearGeometry.angular_weight`.

    Unlike :func:`compute_P_esc_outer_mode` (Lambert basis, no
    :math:`\mu` weight), this integrand carries the emission direction
    cosine :math:`\mu_{\rm exit}` explicitly, placing it in the same
    partial-current moment basis as
    :func:`compute_hollow_sph_transmission_rank_n` (which has
    :math:`\cos\theta\,\sin\theta = \mu\,\mathrm d\mu\,/\,\mathrm d\theta`).

    Model A: rays that would hit the inner shell before the outer
    surface are excluded (those contribute to
    :func:`compute_P_esc_inner_mode_marshak`).

    Currently implemented for ``kind = "sphere-1d"`` with
    ``inner_radius > 0``; slab and cylinder raise
    :class:`NotImplementedError`.
    """
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"compute_P_esc_outer_mode_marshak is implemented for "
            f"hollow sphere only; got kind={geometry.kind!r}, "
            f"inner_radius={geometry.inner_radius}. Slab and cyl "
            f"per-face Marshak primitives — Issue #119 follow-up."
        )
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = geometry.ray_direction_cosine(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)
    pref = geometry.prefactor
    N = len(r_nodes)
    P = np.zeros(N)
    for i in range(N):
        r_i = float(r_nodes[i])
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_out = geometry.rho_max(r_i, cos_om, R)
            if rho_out <= 0.0:
                continue
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, cos_om)
            if rho_in_minus is not None and rho_in_minus < rho_out:
                continue
            tau = geometry.optical_depth_along_ray(
                r_i, cos_om, rho_out, radii, sig_t,
            )
            K_esc = geometry.escape_kernel_mp(tau, dps)
            mu_exit = (rho_out + r_i * cos_om) / R
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_exit]))[0]
            )
            total += (
                omega_wts[k] * angular_factor[k]
                * mu_exit * p_tilde * K_esc
            )
        P[i] = pref * total
    return P


def compute_P_esc_inner_mode_marshak(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_angular: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Marshak partial-current moment :math:`n` of the escape
    probability to the **inner** surface (Phase F.5 / Issue #119).

    .. math::

       P_{\rm esc, in, marshak}^{(n)}(r_i)
         = C_d \!\int_{\rm hit}\!
                 W_\Omega\,\mu_{\rm exit, in}\,
                 \tilde P_n(\mu_{\rm exit, in})\,
                 K_{\rm esc}(\tau)\,\mathrm d\Omega

    with :math:`\mu_{\rm exit, in} = \sqrt{r_0^2 - h^2}/r_0` and
    :math:`h = r_i|\sin\Omega|`.

    Integration is restricted to directions for which the ray hits the
    inner shell before exiting via the outer.

    Currently implemented for ``kind = "sphere-1d"`` with
    ``inner_radius > 0``.
    """
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"compute_P_esc_inner_mode_marshak for "
            f"kind={geometry.kind!r} not yet implemented."
        )
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    r_0 = float(geometry.inner_radius)
    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = geometry.ray_direction_cosine(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)
    pref = geometry.prefactor
    N = len(r_nodes)
    P = np.zeros(N)
    for i in range(N):
        r_i = float(r_nodes[i])
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, cos_om)
            if rho_in_minus is None:
                continue
            tau = geometry.optical_depth_along_ray(
                r_i, cos_om, rho_in_minus, radii, sig_t,
            )
            K_esc = geometry.escape_kernel_mp(tau, dps)
            sin_om = np.sqrt(max(0.0, 1.0 - cos_om * cos_om))
            h_sq = r_i * r_i * sin_om * sin_om
            mu_exit_sq = max(0.0, (r_0 * r_0 - h_sq) / (r_0 * r_0))
            mu_exit = float(np.sqrt(mu_exit_sq))
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_exit]))[0]
            )
            total += (
                omega_wts[k] * angular_factor[k]
                * mu_exit * p_tilde * K_esc
            )
        P[i] = pref * total
    return P


def compute_G_bc_outer_mode_marshak(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Marshak partial-current response at observer :math:`r_i` from a
    unit mode-:math:`n` (partial-current-moment) outward-normal incident
    current on the **outer** surface (Phase F.5 / Issue #119).

    .. math::

       G_{\rm bc, out, marshak}^{(n)}(r_i) = 2\!\int_{\rm hit\ outer}
           \sin\theta\,\mu_s\,\tilde P_n(\mu_s)\,e^{-\tau}\,
           \mathrm d\theta

    where :math:`\mu_s = (\rho_{\rm out} + r_i\cos\theta)/R` is the
    cosine at the outer surface emission point (by chord symmetry on
    outer-outer grazing rays).

    The integrand is µ-weighted consistent with
    :func:`compute_P_esc_outer_mode_marshak` and
    :func:`compute_hollow_sph_transmission_rank_n`.

    Model A: rays blocked by the inner shell are excluded.
    """
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"compute_G_bc_outer_mode_marshak for kind="
            f"{geometry.kind!r} not yet implemented."
        )
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_thetas = np.cos(theta_pts)
    sin_thetas = np.sin(theta_pts)
    N = len(r_nodes)
    G = np.zeros(N)
    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_surf_quad):
            ct = cos_thetas[k]
            st = sin_thetas[k]
            rho_out = geometry.rho_max(r_i, ct, R)
            if rho_out <= 0.0:
                continue
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, ct)
            if rho_in_minus is not None and rho_in_minus < rho_out:
                continue
            if len(radii) == 1:
                tau = sig_t[0] * rho_out
            else:
                tau = geometry.optical_depth_along_ray(
                    r_i, ct, rho_out, radii, sig_t,
                )
            mu_s = (rho_out + r_i * ct) / R
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_s]))[0]
            )
            total += (
                theta_wts[k] * st * mu_s * p_tilde
                * float(np.exp(-tau))
            )
        G[i] = 2.0 * total
    return G


def compute_G_bc_inner_mode_marshak(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Marshak partial-current response at observer :math:`r_i` from a
    unit mode-:math:`n` outward current on the **inner** surface
    (Phase F.5 / Issue #119).

    .. math::

       G_{\rm bc, in, marshak}^{(n)}(r_i) = 2\!\int_{\rm hit\ inner}
           \sin\theta\,\mu_s\,\tilde P_n(\mu_s)\,e^{-\tau}\,
           \mathrm d\theta

    with :math:`\mu_s = \sqrt{r_0^2 - h^2}/r_0` the cosine at the
    inner surface emission point.
    """
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"compute_G_bc_inner_mode_marshak for kind="
            f"{geometry.kind!r} not yet implemented."
        )
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    r_0 = float(geometry.inner_radius)
    theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_thetas = np.cos(theta_pts)
    sin_thetas = np.sin(theta_pts)
    N = len(r_nodes)
    G = np.zeros(N)
    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_surf_quad):
            ct = cos_thetas[k]
            st = sin_thetas[k]
            rho_in_minus, _ = geometry.rho_inner_intersections(r_i, ct)
            if rho_in_minus is None:
                continue
            if len(radii) == 1:
                tau = sig_t[0] * rho_in_minus
            else:
                tau = geometry.optical_depth_along_ray(
                    r_i, ct, rho_in_minus, radii, sig_t,
                )
            sin_om = float(np.sqrt(max(0.0, 1.0 - ct * ct)))
            h_sq = r_i * r_i * sin_om * sin_om
            mu_s_sq = max(0.0, (r_0 * r_0 - h_sq) / (r_0 * r_0))
            mu_s = float(np.sqrt(mu_s_sq))
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_s]))[0]
            )
            total += (
                theta_wts[k] * st * mu_s * p_tilde
                * float(np.exp(-tau))
            )
        G[i] = 2.0 * total
    return G


def build_white_bc_correction(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    n_angular: int = 32,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Rank-1 white-BC correction :math:`K_{\rm bc}[i,j] = u[i]\,v[j]`
    for the unified polar-form volume kernel.

    .. math::

       u[i] = \Sigma_t(r_i)\,G_{\rm bc}(r_i) / A_d, \quad
       v[j] = r_j^{d-1}\,w_j\,P_{\rm esc}(r_j).

    Here :math:`A_d` is the cell-surface measure
    (:meth:`CurvilinearGeometry.surface_area_per_z` divided by
    :math:`2\pi` for cylinder to match the
    :math:`\int\,\mathrm dA_s / A` normalisation of the partial
    current :math:`J^+`). The approximation error is discussed at
    length in the companion Sphinx page.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    r_wts = np.asarray(r_wts, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    N = len(r_nodes)

    sig_t_n = np.empty(N)
    for i, ri in enumerate(r_nodes):
        sig_t_n[i] = sig_t[geometry.which_annulus(ri, radii)]

    P_esc = compute_P_esc(
        geometry, r_nodes, radii, sig_t,
        n_angular=n_angular, dps=dps,
    )
    G_bc = compute_G_bc(
        geometry, r_nodes, radii, sig_t,
        n_surf_quad=n_surf_quad, dps=dps,
    )

    # u[i] = Σ_t(r_i) · G_bc(r_i) / A_d_divisor
    #   cylinder: divisor R  (A_d = 2πR, A_j = 2π r_j w_j, ratio → r_j w_j / R)
    #   sphere:   divisor R² (A_d = 4πR², A_j = 4π r_j² w_j, ratio → r_j² w_j / R²)
    u = sig_t_n * G_bc / geometry.rank1_surface_divisor(R)
    # radial_volume_weight(r_j) · w_j captures r_j (cyl) or r_j^2 (sphere)
    rv = np.array([geometry.radial_volume_weight(rj) for rj in r_nodes])
    v = rv * r_wts * P_esc
    return np.outer(u, v)


# ═══════════════════════════════════════════════════════════════════════
# Unified white-BC rank-N (Marshak / DP_N) closure
# ═══════════════════════════════════════════════════════════════════════

def compute_P_esc_mode(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_angular: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Mode-:math:`n` outgoing partial-current moment per unit source.

    For :math:`n = 0`, :math:`\tilde P_0 \equiv 1` and this reduces
    algebraically to :func:`compute_P_esc` (the isotropic-source
    escape probability). For :math:`n \ge 1`, the **canonical DP**\
    :sub:`N` **outgoing partial-current moment** per unit volumetric
    source at :math:`r_i` is

    .. math::
       :label: peierls-rank-n-P-esc-moment

       P_{\rm esc}^{(n)}(r_i)
         \;=\; C_d\!\int_0^\pi W_\Omega(\Omega)\,
                 \Bigl(\tfrac{\rho_{\max}(r_i, \Omega)}{R}\Bigr)^{\!2}\,
                 \tilde P_n\!\bigl(\mu_{\rm exit}(r_i, \Omega, R)\bigr)\,
                 K_{\rm esc}\!\bigl(\tau(r_i, \Omega)\bigr)\,\mathrm d\Omega

    with
    :math:`\mu_{\rm exit}(r, \Omega, R) = (\rho_{\max} + r\cos\Omega)/R`
    the direction cosine of the outgoing ray with the outward surface
    normal at the exit point.

    The :math:`(\rho_{\max}/R)^2` factor is the **surface-to-observer
    Jacobian** :math:`\mathrm d A_s / \mathrm d\Omega_{\mathrm{obs}} =
    d^2 / |\mu_s|` (with :math:`d = \rho_{\max}`), after the
    :math:`1/|\mu_s|` cancels the cosine weighting
    :math:`|\mu_{\rm out}|` of the partial-current moment
    :math:`J^+_n = \int \mu\,\tilde P_n(\mu)\,\psi^+\,\mathrm d\mu`.
    The :math:`R^2` denominator normalises it against the cell's
    characteristic surface area (:math:`A_d = 4\pi R^2` for sphere,
    :math:`A_d = 2\pi R` per unit :math:`z` for cylinder — the
    factor is absorbed into the rank-1 divisor
    :meth:`CurvilinearGeometry.rank1_surface_divisor`, which is
    :math:`R` for cylinder and :math:`R^2` for sphere).

    For :math:`n = 0`, :math:`\tilde P_0 \equiv 1` and
    :math:`(\rho_{\max}/R)^2` is generally **not** :math:`\equiv 1`,
    so this function does not reduce to :func:`compute_P_esc` at
    :math:`n = 0`. The mode-0 path is therefore routed through the
    existing :func:`compute_P_esc` by
    :func:`build_white_bc_correction_rank_n` for bit-exact rank-1
    regression. The function here is the canonical DP\ :sub:`N`
    moment for :math:`n \ge 1`; calling it at :math:`n = 0` returns
    the Jacobian-weighted moment, not :math:`P_{\rm esc}`.
    """
    if geometry.kind == "slab-polar" and n_mode > 0:
        # The current mode-n ≥ 1 formulation uses the curvilinear
        # (ρ_max/R)² surface-to-observer Jacobian and a single-surface
        # mode space A = R^N. For slab this requires per-face mode
        # decomposition (A = R^(2·N)) because the two faces at x = 0
        # and x = L carry independent Legendre moments. Not in scope
        # for Issue #118; tracked separately.
        raise NotImplementedError(
            "Rank-N (n_mode > 0) BC closure for slab-polar requires "
            "per-face mode decomposition; use n_bc_modes = 1 (rank-1 "
            "Mark closure) for slab. See Issue #118 follow-up."
        )

    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    inv_R2 = 1.0 / (R * R)

    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    # Polymorphic direction cosine (identity for slab-polar where the
    # angular variable IS µ; cos(Ω) for curvilinear).
    cos_omegas = geometry.ray_direction_cosine(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)
    pref = geometry.prefactor

    N = len(r_nodes)
    P = np.zeros(N)
    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_max_val = geometry.rho_max(r_i, cos_om, R)
            if rho_max_val <= 0.0:
                continue
            tau = geometry.optical_depth_along_ray(
                r_i, cos_om, rho_max_val, radii, sig_t,
            )
            K_esc = geometry.escape_kernel_mp(tau, dps)
            mu_exit = (rho_max_val + r_i * cos_om) / R
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_exit]))[0]
            )
            jacobian = rho_max_val * rho_max_val * inv_R2
            total += (
                omega_wts[k] * angular_factor[k]
                * jacobian * p_tilde * K_esc
            )
        P[i] = pref * total
    return P


def compute_G_bc_mode(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_mode: int,
    n_surf_quad: int = 32,
    dps: int = 25,
) -> np.ndarray:
    r"""Mode-:math:`n` weighted surface-to-volume Green's function.

    Sphere (observer-centred :math:`\theta`-integral):

    .. math::

       G_{\rm bc}^{(n)}(r_i) = 2\!\int_0^\pi \sin\theta\,
          \tilde P_n(\mu_{\rm exit})\,
          e^{-\tau(r_i, \rho_{\max}(\theta))}\,\mathrm d\theta,
       \qquad
       \mu_{\rm exit} = \frac{\rho_{\max} + r_i\cos\theta}{R}.

    Cylinder (surface-centred :math:`\phi`-integral, matching the
    mode-0 :func:`compute_G_bc` form):

    .. math::

       G_{\rm bc}^{(n)}(r_i) = \frac{2R}{\pi}\!\int_0^\pi
          \tilde P_n(|\mu_s|)\,
          \frac{\mathrm{Ki}_1(\tau_{\rm surf})}{d}\,\mathrm d\phi,
       \qquad
       |\mu_s| = \frac{R - r_i\cos\phi}{d},\quad
       d = \sqrt{r_i^2 + R^2 - 2 r_i R\cos\phi}.

    For :math:`n = 0`, :math:`\tilde P_0 \equiv 1` and this reduces to
    :func:`compute_G_bc`.
    """
    if geometry.kind == "slab-polar" and n_mode > 0:
        # Same scope limit as compute_P_esc_mode: rank-N slab needs
        # per-face modes. Issue #118 follow-up.
        raise NotImplementedError(
            "Rank-N (n_mode > 0) G_bc for slab-polar requires per-face "
            "mode decomposition; use n_bc_modes = 1 (rank-1 Mark closure) "
            "for slab. See Issue #118 follow-up."
        )

    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])

    N = len(r_nodes)
    G = np.zeros(N)

    if geometry.kind == "sphere-1d":
        theta_pts, theta_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
        cos_thetas = np.cos(theta_pts)
        sin_thetas = np.sin(theta_pts)

        for i in range(N):
            r_i = r_nodes[i]
            total = 0.0
            for k in range(n_surf_quad):
                ct = cos_thetas[k]
                st = sin_thetas[k]
                rho_to_surface = geometry.rho_max(r_i, ct, R)
                if rho_to_surface <= 0.0:
                    continue
                if len(radii) == 1:
                    tau = sig_t[0] * rho_to_surface
                else:
                    tau = geometry.optical_depth_along_ray(
                        r_i, ct, rho_to_surface, radii, sig_t,
                    )
                mu_exit = (rho_to_surface + r_i * ct) / R
                p_tilde = float(
                    _shifted_legendre_eval(n_mode, np.array([mu_exit]))[0]
                )
                total += theta_wts[k] * st * p_tilde * float(np.exp(-tau))
            G[i] = 2.0 * total
        return G

    # Cylinder: surface-centred Ki_1/d kernel, weighted by P̃_n(|μ_s_2D|).
    phi_pts, phi_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
    cos_phis = np.cos(phi_pts)
    sin_phis = np.sin(phi_pts)
    inv_pi = 1.0 / np.pi

    for i in range(N):
        r_i = r_nodes[i]
        total = 0.0
        for k in range(n_surf_quad):
            cf = cos_phis[k]
            d_sq = r_i * r_i + R * R - 2.0 * r_i * R * cf
            d = np.sqrt(max(d_sq, 0.0))
            if d <= 0.0:
                continue
            if len(radii) == 1:
                tau = sig_t[0] * d
            else:
                cb = (R * cf - r_i) / d
                sb = R * sin_phis[k] / d  # noqa: F841 (kept for symmetry)
                tau = geometry.optical_depth_along_ray(
                    r_i, cb, d, radii, sig_t,
                )
            mu_s = (R - r_i * cf) / d
            p_tilde = float(
                _shifted_legendre_eval(n_mode, np.array([mu_s]))[0]
            )
            ki1 = float(ki_n_mp(1, float(tau), dps))
            total += phi_wts[k] * p_tilde * ki1 / d
        G[i] = 2.0 * inv_pi * R * total
    return G


# ═══════════════════════════════════════════════════════════════════════
# Factored tensor form of the boundary-closure kernel
# ═══════════════════════════════════════════════════════════════════════
#
# Mathematical background (see :ref:`theory-peierls-unified` Part IV for
# the full derivation). The boundary-closure contribution K_bc is a
# Hilbert-Schmidt integral operator on the radial function space V.
# In the Nyström discretisation V = R^{N_r}, K_bc is a matrix in V ⊗ V*
# (a (1,1) tensor). The white-BC physics factors it through a finite
# mode space A = R^N (the Gelbard shifted-Legendre expansion on the
# inward surface hemisphere) as the tensor network
#
#     K_bc = G · R · P    ∈ V ⊗ V*
#
# with the three operators
#
#     P : V → A    outgoing angular-moment operator ("escape tensor")
#     R : A → A    reflection operator on the mode space
#     G : A → V    surface-to-volume response operator
#
# Every BC flavour (vacuum, reflective, white-Mark, white-Marshak DP_N,
# albedo, interface current) is a CHOICE OF R. The geometry-specific
# integrals live entirely in P and G; the BC physics lives entirely in
# R. The classes and helpers below expose this structure directly.


@dataclass(frozen=True)
class BoundaryClosureOperator:
    r"""Factored tensor-network representation of :math:`K_{\mathrm{bc}}`.

    Stores the three tensors of the factorisation

    .. math::

       K_{\mathrm{bc}} \;=\; G \cdot R \cdot P, \qquad
       (K_{\mathrm{bc}})^i{}_j \;=\; G^i{}_n\,R^n{}_m\,P^m{}_j,

    where :math:`P : V \to A`, :math:`R : A \to A`, :math:`G : A \to V`
    are the escape, reflection, and response operators, with :math:`V`
    the radial Nyström space (dim :math:`N_r`) and :math:`A` the finite
    mode space on the surface (dim :math:`N`). Contraction is over the
    shared mode index.

    Complexity. Storage is :math:`\mathcal O(N_r N + N^2)`; applying
    the operator to a source vector takes :math:`\mathcal O(N_r N +
    N^2)` via :meth:`apply`; materialising the dense
    :math:`N_r \times N_r` matrix costs :math:`\mathcal O(N_r^2 N)`
    and is only needed when a caller demands a dense matrix (e.g. the
    direct-LU eigenvalue iteration in :func:`solve_peierls_1g`).

    Boundary conditions. The choice of BC is entirely encoded in
    :attr:`R`:

    - :func:`reflection_vacuum` — :math:`R = 0` (no reflection)
    - :func:`reflection_mark` — :math:`R = e_0 e_0^{\top}`
      (rank-1 isotropic white closure; only the scalar mode)
    - :func:`reflection_marshak` — diagonal :math:`R` with Gelbard
      :math:`(2n+1)` normalisation (rank-:math:`N` white closure)
    - any other matrix — albedo, partial reflection, or (for
      future lattice extensions) a non-square mode-coupling matrix
      to neighbouring cells

    See :ref:`theory-peierls-unified` Part IV for the Hilbert-Schmidt
    factorisation, the Karhunen-Loève / SVD connection, and the full
    derivation of the factored form.
    """

    P: np.ndarray  # escape tensor,       shape (N_modes, N_nodes)
    G: np.ndarray  # response tensor,     shape (N_nodes, N_modes)
    R: np.ndarray  # reflection operator, shape (N_modes, N_modes)

    def __post_init__(self) -> None:
        if self.P.ndim != 2 or self.G.ndim != 2 or self.R.ndim != 2:
            raise ValueError("P, G, R must all be 2-D arrays")
        N = self.R.shape[0]
        if self.R.shape != (N, N):
            raise ValueError(f"R must be square, got shape {self.R.shape}")
        if self.P.shape[0] != N or self.G.shape[1] != N:
            raise ValueError(
                f"Shape mismatch: P {self.P.shape}, R {self.R.shape}, "
                f"G {self.G.shape} — mode dim (axis 0 of P, both axes "
                f"of R, axis 1 of G) must all equal {N}"
            )
        if self.P.shape[1] != self.G.shape[0]:
            raise ValueError(
                f"Radial node count mismatch: P has {self.P.shape[1]} "
                f"nodes but G has {self.G.shape[0]}"
            )

    @property
    def n_modes(self) -> int:
        """Mode-space dimension :math:`N`."""
        return self.R.shape[0]

    @property
    def n_nodes(self) -> int:
        """Radial-space dimension :math:`N_r`."""
        return self.P.shape[1]

    @property
    def closure_rank(self) -> int:
        """Numerical rank of :math:`K_{\\mathrm{bc}}`, = rank(:math:`R`)
        under the generic assumption that :math:`P` and :math:`G` have
        full mode rank."""
        return int(np.linalg.matrix_rank(self.R))

    def apply(self, q: np.ndarray) -> np.ndarray:
        r"""Matrix-free application :math:`K_{\mathrm{bc}}\,q` via the
        three-step tensor contraction

        .. math::

           K_{\mathrm{bc}}\,q \;=\; G\,(R\,(P\,q)),

        in :math:`\mathcal O(N_r N + N^2)` flops. No intermediate
        :math:`N_r \times N_r` matrix is ever allocated — storage
        stays at :math:`\mathcal O(N_r N + N^2)` throughout.
        """
        outgoing_moments = self.P @ q                 # V → A
        incoming_moments = self.R @ outgoing_moments  # A → A
        return self.G @ incoming_moments              # A → V

    def as_matrix(self) -> np.ndarray:
        r"""Materialise the dense :math:`N_r \times N_r` matrix
        :math:`K_{\mathrm{bc}} = G R P`.

        Cost :math:`\mathcal O(N_r^2 N)`; useful only when a caller
        requires a dense representation (e.g. the direct-solve inner
        iteration of :func:`solve_peierls_1g`, which uses
        :func:`numpy.linalg.solve` on :math:`A = \mathrm{diag}(\Sigma_t)
        - K\,\mathrm{diag}(\Sigma_s)`).
        """
        return self.G @ self.R @ self.P


def reflection_vacuum(n_modes: int) -> np.ndarray:
    r"""Vacuum-BC reflection: :math:`R = 0`.

    All outgoing moments are absorbed; no re-entering flux.
    :math:`K_{\mathrm{bc}} = 0` regardless of :math:`P`, :math:`G`.
    """
    return np.zeros((n_modes, n_modes))


def reflection_mark(n_modes: int) -> np.ndarray:
    r"""Mark / isotropic white-BC reflection: rank-1 projector on the
    scalar mode.

    .. math::

       R \;=\; e_0 e_0^{\top}
             \;=\; \mathrm{diag}(1, 0, 0, \ldots, 0).

    The zeroth outgoing moment :math:`J^{+}_0` is returned as an
    isotropic inward distribution with :math:`J^{-}_0 = J^{+}_0`; all
    higher moments are discarded. This is the classical Mark closure
    and coincides with the pre-rank-N ORPHEUS rank-1 white BC.
    """
    R = np.zeros((n_modes, n_modes))
    R[0, 0] = 1.0
    return R


def reflection_marshak(n_modes: int) -> np.ndarray:
    r"""Marshak / Gelbard DP\ :sub:`N-1` diagonal reflection with the
    :math:`(2n+1)` normalisation built in.

    .. math::

       R \;=\; \mathrm{diag}\bigl(1,\,3,\,5,\,\ldots,\,(2(N-1)+1)\bigr).

    Entry :math:`n = 0` is :math:`1` (preserves bit-exact rank-1
    Mark recovery when the other modes are truncated by
    :math:`n_{\mathrm{modes}} = 1`); entries :math:`n \ge 1` carry the
    Gelbard-shifted-Legendre expansion normalisation :math:`2n+1`
    that appears in :math:`\psi(\mu) = \sum_n (2n+1)\,a_n\,\tilde P_n(\mu)`.
    Off-diagonal entries are zero: under isotropic scattering on a
    rotationally-symmetric cell with white BC, the closure is
    strictly diagonal in mode index (Sanchez & McCormick 1982
    Eq. 167).
    """
    R = np.diag(
        np.array([1.0] + [2.0 * n + 1.0 for n in range(1, n_modes)])
    )
    return R


# ═══════════════════════════════════════════════════════════════════════
# Rank-2 per-face white-BC closure (Phase F.3)
#
# For 2-boundary (Class-A) geometries — slab, hollow cylinder, hollow
# sphere — the rank-1 Mark closure omits the surface-to-surface
# transmission feedback. The acid test is the Wigner-Seitz identity
# :math:`k_{\rm eff} = k_\infty` for a homogeneous cell with white BC on
# both surfaces: rank-1 Mark misses it by 16-40 % at finite L, rank-2
# white closes it to machine precision.
#
# The rank-2 closure decomposes the mode space as
# :math:`A = \mathbb{R}^{N_{\rm modes} \times N_{\rm surfaces}}` and
# stitches the per-face escape (:func:`compute_P_esc_{outer,inner}`),
# per-face response (:func:`compute_G_bc_{outer,inner}`), and a 2x2
# reflection that inverts the coupled partial-current balance
#
# .. math::
#
#    J^-_{\rm out} = J^+_{\rm out} = q \otimes P_{\rm esc,out}
#                                   + T\,J^-_{\rm in}, \\
#    J^-_{\rm in}  = J^+_{\rm in}  = q \otimes P_{\rm esc,in}
#                                   + T\,J^-_{\rm out},
#
# giving
# :math:`R_{\rm white} = (I - W)^{-1}` with :math:`W = T \cdot
# \bigl(\begin{smallmatrix}0 & 1\\1 & 0\end{smallmatrix}\bigr)` for
# slab (:math:`T = 2 E_3(\tau_L)`). For hollow cyl/sph the transmission
# matrix :math:`W` additionally carries self-transmission entries
# (tangent rays grazing the cavity) — see Phase F.4.
# ═══════════════════════════════════════════════════════════════════════


def compute_slab_transmission(
    L: float,
    radii: np.ndarray,
    sig_t: np.ndarray,
    dps: int = 25,
) -> float:
    r"""Slab surface-to-surface transmission coefficient

    .. math::

       T \;=\; 2\,E_3(\Sigma_t\,L)

    for a slab of optical thickness :math:`\Sigma_t\,L`. For a
    multi-region slab :math:`[0, L] = \bigcup_k [r_{k-1}, r_k]` with
    piecewise-constant :math:`\Sigma_t^{(k)}`,
    :math:`T = 2\,E_3(\tau_{\rm total})` with
    :math:`\tau_{\rm total} = \sum_k \Sigma_t^{(k)}(r_k - r_{k-1})`.

    Used by the rank-2 white-BC reflection builder to couple the two
    slab faces.
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    tau_total = 0.0
    r_prev = 0.0
    for k, r_k in enumerate(radii):
        tau_total += float(sig_t[k]) * (float(r_k) - r_prev)
        r_prev = float(r_k)
    return 2.0 * float(mpmath.expint(3, mpmath.mpf(tau_total)))


def compute_hollow_cyl_transmission(
    r_0: float,
    R: float,
    radii: np.ndarray,
    sig_t: np.ndarray,
    dps: int = 25,
) -> np.ndarray:
    r"""Surface-to-surface transmission matrix :math:`W` for a
    homogeneous hollow cylindrical annulus :math:`[r_0, R]` with a
    pure absorber.

    Returns the :math:`2 \times 2` matrix whose entry :math:`W_{k,l}`
    is the probability that a unit uniform-isotropic (Lambertian)
    outgoing current leaving surface :math:`l` re-arrives at surface
    :math:`k` without collision. The mode convention is:
    index 0 = outer surface (:math:`r = R`), index 1 = inner
    (:math:`r = r_0`).

    Closed-form chord decomposition under Lambertian emission
    (3-D angle fold into :math:`\mathrm{Ki}_3`):

    .. math::

       W_{\rm oo} = \frac{4}{\pi}\!\int_{\alpha_c}^{\pi/2}\!
           \cos\alpha \,\mathrm{Ki}_3(2\Sigma_t R\cos\alpha)\,\mathrm d\alpha,
           \qquad \alpha_c = \arcsin(r_0/R),

    (rays from :math:`R` grazing past the cavity, chord
    :math:`2R\cos\alpha`);

    .. math::

       W_{\rm io} = \frac{4}{\pi}\!\int_0^{\alpha_c}\!
           \cos\alpha \,\mathrm{Ki}_3\!\bigl(\Sigma_t(R\cos\alpha
               - \sqrt{r_0^2 - R^2\sin^2\alpha})\bigr)\,\mathrm d\alpha,

    (rays from :math:`R` hitting the inner shell before the opposite
    outer face, chord only the annular leg). The inner-to-outer
    transmission follows by reciprocity (the surface-areas enter via
    :math:`2\pi R \cdot W_{\rm io} = 2\pi r_0 \cdot W_{\rm oi}`):

    .. math::

       W_{\rm oi} = \frac{R}{r_0}\,W_{\rm io}.

    Self-transmission at the inner surface is :math:`W_{\rm ii} = 0`:
    rays leaving :math:`r_0` outward travel through the annulus and
    either exit through :math:`R` or are absorbed; a convex cavity
    cannot route a ray back to :math:`r_0` without a BC reflection.

    For multi-region annuli the integral replaces :math:`\Sigma_t\cdot
    \text{chord}` by the piecewise optical depth
    :math:`\int_{\rm annulus} \Sigma_t(r(s))\,\mathrm ds`; this function
    handles the homogeneous single-region case directly and defers
    multi-region to the general
    :meth:`CurvilinearGeometry.optical_depth_along_ray` walker.

    Parameters
    ----------
    r_0, R
        Inner and outer radii (:math:`0 < r_0 < R`).
    radii
        Outer edges of annular regions (length 1 for homogeneous,
        longer for multi-region). For multi-region the Ki_3 argument
        becomes the actual piecewise-integrated optical depth.
    sig_t
        Region-wise total macroscopic XS.
    dps
        mpmath working precision.

    Returns
    -------
    W : ndarray, shape (2, 2)
        Transmission matrix with rows/cols indexed [outer, inner].
    """
    if not (0.0 < r_0 < R):
        raise ValueError(
            f"Hollow cylinder requires 0 < r_0 < R; got r_0={r_0}, R={R}"
        )
    sig_t = np.asarray(sig_t, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(radii) != 1 or len(sig_t) != 1:
        raise NotImplementedError(
            "Multi-region hollow cylinder transmission is planned but "
            "not yet implemented — pass a single-region annulus "
            "(len(radii) == 1). For multi-region, the Ki_3 chord "
            "integral must be replaced by the piecewise optical depth."
        )
    sig_t_val = float(sig_t[0])

    alpha_c = float(np.arcsin(r_0 / R))

    def _Ki3(x):
        return float(ki_n_mp(3, float(x), dps))

    with mpmath.workdps(dps):
        # W_outer_outer: grazing-cavity rays hit outer again.
        W_oo = float(mpmath.quad(
            lambda a: mpmath.cos(a) * _Ki3(
                2.0 * sig_t_val * R * float(mpmath.cos(a))
            ),
            [mpmath.mpf(alpha_c), mpmath.pi / 2],
        ))
        W_oo *= 4.0 / float(mpmath.pi)

        # W_outer_inner: rays hitting inner shell before opposite
        # outer face. Chord from R to the first inner intersection.
        def _chord_out_to_in(a):
            h_sq = R * R * float(mpmath.sin(a)) ** 2
            return R * float(mpmath.cos(a)) - float(
                mpmath.sqrt(mpmath.mpf(r_0 * r_0 - h_sq))
            )

        W_io = float(mpmath.quad(
            lambda a: mpmath.cos(a) * _Ki3(
                sig_t_val * _chord_out_to_in(a)
            ),
            [mpmath.mpf(0.0), mpmath.mpf(alpha_c)],
        ))
        W_io *= 4.0 / float(mpmath.pi)

    # Reciprocity for outer←inner; convex-cavity inner←inner = 0.
    W_oi = (R / r_0) * W_io
    return np.array([[W_oo, W_oi], [W_io, 0.0]])


def compute_hollow_sph_transmission(
    r_0: float,
    R: float,
    radii: np.ndarray,
    sig_t: np.ndarray,
    dps: int = 25,
) -> np.ndarray:
    r"""Surface-to-surface transmission matrix :math:`W` for a
    homogeneous hollow spherical shell :math:`[r_0, R]` with a pure
    absorber.

    Mirrors :func:`compute_hollow_cyl_transmission` but with the
    bare :math:`e^{-\tau}` kernel — the sphere's 3-D geometry
    integrates directly over solid angle without the
    :math:`\mathrm{Ki}_n` out-of-plane fold:

    .. math::

       W_{\rm oo} &= 2\!\int_{\theta_c}^{\pi/2}\!
           \cos\theta\,\sin\theta\,
           e^{-2\Sigma_t R\cos\theta}\,\mathrm d\theta, \\
       W_{\rm io} &= 2\!\int_0^{\theta_c}\!
           \cos\theta\,\sin\theta\,
           e^{-\Sigma_t(R\cos\theta - \sqrt{r_0^2 - R^2\sin^2\theta})}\,
           \mathrm d\theta,

    :math:`\theta_c = \arcsin(r_0/R)`. Reciprocity on spherical
    surface areas (:math:`A_{\rm out} = 4\pi R^2`,
    :math:`A_{\rm in} = 4\pi r_0^2`) gives

    .. math::

       W_{\rm oi} = (R/r_0)^2\,W_{\rm io},

    and :math:`W_{\rm ii} = 0` by convex-cavity reasoning identical
    to cylinder.

    Parameters
    ----------
    r_0, R
        Inner and outer radii (:math:`0 < r_0 < R`).
    radii, sig_t
        Homogeneous single-region inputs: ``len(radii) == 1``.
    dps
        mpmath working precision.

    Returns
    -------
    W : ndarray, shape (2, 2)
        Transmission matrix [outer, inner] × [outer, inner].
    """
    if not (0.0 < r_0 < R):
        raise ValueError(
            f"Hollow sphere requires 0 < r_0 < R; got r_0={r_0}, R={R}"
        )
    sig_t = np.asarray(sig_t, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(radii) != 1 or len(sig_t) != 1:
        raise NotImplementedError(
            "Multi-region hollow sphere transmission is planned but "
            "not yet implemented — pass a single-region annulus "
            "(len(radii) == 1)."
        )
    sig_t_val = float(sig_t[0])

    theta_c = float(np.arcsin(r_0 / R))

    with mpmath.workdps(dps):
        W_oo = 2.0 * float(mpmath.quad(
            lambda t: (
                mpmath.cos(t) * mpmath.sin(t)
                * mpmath.exp(-2.0 * sig_t_val * R * mpmath.cos(t))
            ),
            [mpmath.mpf(theta_c), mpmath.pi / 2],
        ))

        def _chord(t):
            h_sq = R * R * float(mpmath.sin(t)) ** 2
            return R * float(mpmath.cos(t)) - float(
                mpmath.sqrt(mpmath.mpf(r_0 * r_0 - h_sq))
            )

        W_io = 2.0 * float(mpmath.quad(
            lambda t: (
                mpmath.cos(t) * mpmath.sin(t)
                * mpmath.exp(-sig_t_val * _chord(t))
            ),
            [mpmath.mpf(0.0), mpmath.mpf(theta_c)],
        ))

    # Reciprocity on 4π R² / 4π r_0² — sphere area ratio is (R/r_0)².
    W_oi = (R / r_0) ** 2 * W_io
    return np.array([[W_oo, W_oi], [W_io, 0.0]])


def compute_hollow_sph_transmission_rank_n(
    r_0: float,
    R: float,
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_bc_modes: int,
    dps: int = 25,
) -> np.ndarray:
    r"""Rank-:math:`N` per-face surface-to-surface transmission matrix
    for a homogeneous hollow spherical annulus (Phase F.5 /
    Issue #119).

    Returns a :math:`(2N) \times (2N)` matrix with block structure

    .. math::

       W = \begin{pmatrix}
           W_{\rm oo} & W_{\rm oi} \\
           W_{\rm io} & W_{\rm ii}
       \end{pmatrix},

    each block :math:`(N \times N)` in the per-face Legendre-moment
    basis :math:`\tilde P_n(|\mu|)` on :math:`[0, 1]`. Row/column
    layout is ``[outer_mode_0, ..., outer_mode_{N-1}, inner_mode_0,
    ..., inner_mode_{N-1}]``.

    **Formulas** (Sanchez–McCormick 1982 §III.F, Hébert 2020 §3 —
    with :math:`\mu = \cos\theta` from local inward normal,
    :math:`\theta_c = \arcsin(r_0/R)`):

    .. math::

       W_{\rm oo}^{(m,n)} &= 2\!\!\int_{\theta_c}^{\pi/2}\!\!
           \cos\theta\,\sin\theta\,\tilde P_n(\cos\theta)\,
           \tilde P_m(\cos\theta)\,e^{-2\Sigma_t R\cos\theta}\,
           \mathrm d\theta, \\
       W_{\rm io}^{(m,n)} &= 2\!\!\int_0^{\theta_c}\!\!
           \cos\theta\,\sin\theta\,\tilde P_n(\cos\theta)\,
           \tilde P_m(c_{\rm in})\,e^{-\Sigma_t\ell(\theta)}\,
           \mathrm d\theta,\quad
           c_{\rm in} = \sqrt{1 - (R\sin\theta/r_0)^2}, \\
       W_{\rm oi}^{(m,n)} &= (R/r_0)^2\,W_{\rm io}^{(n,m)} \quad
           \text{(reciprocity, mode indices transposed)}, \\
       W_{\rm ii}^{(m,n)} &= 0 \quad \text{(convex cavity)}.

    For :math:`N = 1` this reduces exactly to
    :func:`compute_hollow_sph_transmission` (the scalar case).

    The Gelbard :math:`(2n+1)` normalisation lives in the reflection
    operator :math:`R`, not in :math:`W` (consistent with the
    existing :func:`reflection_marshak` convention).
    """
    if not (0.0 < r_0 < R):
        raise ValueError(
            f"Hollow sphere requires 0 < r_0 < R; got r_0={r_0}, R={R}"
        )
    if n_bc_modes < 1:
        raise ValueError(f"n_bc_modes must be >= 1, got {n_bc_modes}")
    sig_t = np.asarray(sig_t, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(radii) != 1 or len(sig_t) != 1:
        raise NotImplementedError(
            "Multi-region rank-N hollow sphere transmission not "
            "implemented (single-region homogeneous only)."
        )
    N = int(n_bc_modes)
    sig_t_val = float(sig_t[0])
    theta_c = float(np.arcsin(r_0 / R))
    dim = 2 * N
    W = np.zeros((dim, dim))

    # Precompute P_tilde evaluation at a single µ.
    def _P_tilde(n, mu):
        return float(_shifted_legendre_eval(n, np.array([mu]))[0])

    with mpmath.workdps(dps):
        # --- Outer → outer block (grazing past cavity) ---
        for m in range(N):
            for n in range(N):
                val = 2.0 * float(mpmath.quad(
                    lambda th, m=m, n=n: (
                        mpmath.cos(th) * mpmath.sin(th)
                        * _P_tilde(n, float(mpmath.cos(th)))
                        * _P_tilde(m, float(mpmath.cos(th)))
                        * mpmath.exp(-2.0 * sig_t_val * R * mpmath.cos(th))
                    ),
                    [mpmath.mpf(theta_c), mpmath.pi / 2],
                ))
                W[m, n] = val

        # --- Inner ← outer block (hit inner first) ---
        def _chord(t):
            h_sq = R * R * float(mpmath.sin(t)) ** 2
            return R * float(mpmath.cos(t)) - float(
                mpmath.sqrt(mpmath.mpf(r_0 * r_0 - h_sq))
            )

        for m in range(N):
            for n in range(N):
                def integrand(th, m=m, n=n):
                    cos_th = float(mpmath.cos(th))
                    sin_th = float(mpmath.sin(th))
                    # Arrival µ at inner (local inward-normal frame)
                    c_in_sq = 1.0 - (R * sin_th / r_0) ** 2
                    if c_in_sq < 0.0:
                        c_in_sq = 0.0
                    c_in = float(mpmath.sqrt(mpmath.mpf(c_in_sq)))
                    return (
                        cos_th * sin_th
                        * _P_tilde(n, cos_th)
                        * _P_tilde(m, c_in)
                        * mpmath.exp(-sig_t_val * _chord(th))
                    )

                val = 2.0 * float(mpmath.quad(
                    integrand, [mpmath.mpf(0.0), mpmath.mpf(theta_c)],
                ))
                W[N + m, n] = val

        # --- Outer ← inner block via reciprocity ---
        # A_outer · W_oi^{mn} = A_inner · W_io^{nm}
        # A_outer/A_inner = (R/r_0)² for sphere; indices transposed.
        area_ratio = (R / r_0) ** 2
        for m in range(N):
            for n in range(N):
                W[m, N + n] = area_ratio * W[N + n, m]

        # --- Inner → inner block: zero (convex cavity) ---
        # Already np.zeros.
    return W


def reflection_white_rank2(
    W: np.ndarray,
) -> np.ndarray:
    r"""White-BC reflection with transmission feedback:
    :math:`R = (I - W)^{-1}`.

    ``W`` is the surface-to-surface transmission matrix — entry
    :math:`W_{kl}` is the probability that a unit uniform isotropic
    outgoing current leaving surface :math:`l` re-arrives at surface
    :math:`k` without attenuation on the far side (i.e., having
    traversed the cell interior). For Class-A (:math:`N_{\rm surf} = 2`)
    cells with a pure absorber, :math:`W` is :math:`2 \times 2`.

    For slab with scalar transmission :math:`T`:
    :math:`W = T\,\bigl(\begin{smallmatrix}0 & 1\\1 & 0\end{smallmatrix}\bigr)`,
    giving :math:`R = 1/(1-T^2)\,\bigl(\begin{smallmatrix}1 & T\\T & 1\end{smallmatrix}\bigr)`.

    For hollow cyl/sph :math:`W` additionally has self-transmission
    diagonal entries (tangent rays). See :func:`compute_transmission_matrix`.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square 2-D, got {W.shape}")
    n = W.shape[0]
    I_n = np.eye(n)
    det = np.linalg.det(I_n - W)
    if abs(det) < 1e-30:
        raise ValueError(
            f"(I - W) is singular (det = {det:.3e}) — "
            f"transmission matrix W yields an ill-posed closure. "
            f"This indicates a critical configuration (k_eff = 1 "
            f"without BC) or a bug in W's construction."
        )
    return np.linalg.inv(I_n - W)


def build_closure_operator(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    n_angular: int = 32,
    n_surf_quad: int = 32,
    dps: int = 25,
    n_bc_modes: int = 1,
    reflection: str | np.ndarray = "marshak",
) -> BoundaryClosureOperator:
    r"""Assemble the factored :math:`K_{\mathrm{bc}} = G\,R\,P` operator.

    The escape tensor :math:`P` and response tensor :math:`G` are
    built from geometry-specific Nyström integrals of the mode-0
    (legacy Mark) and mode-:math:`n \ge 1` (Gelbard DP\ :sub:`N-1`)
    primitives. The reflection operator :math:`R` is chosen via
    the ``reflection`` argument (default: Marshak / Gelbard
    DP\ :sub:`N-1` diagonal).

    **Mode-0 convention**. Mode 0 is routed through the existing
    :func:`compute_P_esc` / :func:`compute_G_bc` (the Mark
    escape-probability form that predates the rank-N extension), so
    that ``n_bc_modes = 1`` together with ``reflection = "mark"`` or
    ``"marshak"`` (both have :math:`R_{00} = 1`) yields the rank-1
    :math:`K_{\mathrm{bc}}` bit-exactly matching the legacy
    :func:`build_white_bc_correction`. This is the regression gate.

    **Mode-n (n ≥ 1) convention**. Modes :math:`n \ge 1` use
    :func:`compute_P_esc_mode` (with the canonical
    :math:`(\rho_{\max}/R)^2` surface-to-observer Jacobian
    :eq:`peierls-rank-n-P-esc-moment`) and
    :func:`compute_G_bc_mode`. The Gelbard :math:`(2n+1)` expansion
    normalisation lives in :attr:`BoundaryClosureOperator.R`, not in
    :math:`P` — this exposes the closure structure as the identity
    :math:`R = \mathrm{diag}(2n+1)` on the mode space.

    See :ref:`theory-peierls-unified` Part IV for the operator-level
    derivation and the Hilbert-Schmidt / SVD interpretation.

    Parameters
    ----------
    reflection : str | np.ndarray
        ``"vacuum"``, ``"mark"``, ``"marshak"``, or an explicit
        :math:`(N \times N)` matrix. See
        :func:`reflection_vacuum`, :func:`reflection_mark`,
        :func:`reflection_marshak` for the canonical choices.
    """
    if n_bc_modes < 1:
        raise ValueError(f"n_bc_modes must be >= 1, got {n_bc_modes}")

    r_nodes = np.asarray(r_nodes, dtype=float)
    r_wts = np.asarray(r_wts, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R_cell = float(radii[-1])
    N_r = len(r_nodes)
    N = n_bc_modes

    sig_t_n = np.empty(N_r)
    for i, ri in enumerate(r_nodes):
        sig_t_n[i] = sig_t[geometry.which_annulus(ri, radii)]
    rv = np.array([geometry.radial_volume_weight(rj) for rj in r_nodes])

    # ── Rank-2 per-face layout (Phase F.3, reflection="white") ──────
    # Triggered only for Class-A cells (geometry.n_surfaces == 2) under
    # the "white" string reflection. Solid cyl/sph fall back to the
    # legacy single-surface layout for bit-exact rank-1 regression.
    use_rank2 = (
        isinstance(reflection, str)
        and reflection == "white"
        and geometry.n_surfaces == 2
    )
    if use_rank2:
        if n_bc_modes > 1:
            # Phase F.5 / Issue #119 — rank-N per-face white BC.
            # Infrastructure is present (mode primitives both Lambert
            # and Marshak variants, (2N × 2N) transmission matrix,
            # assembly helper) but the final closure
            # `K_bc = G · (I − W)⁻¹ · P` does NOT close the Wigner-Seitz
            # identity at N ≥ 2 regardless of basis choice. A 16-recipe
            # scan at R=5, r_0/R=0.3, homogeneous Σ_t=1 gives best-case
            # ~1.4 % residual vs the Phase F.4 N=1 scalar result of
            # 0.077 % — adding mode-1 DEGRADES accuracy instead of
            # improving it, indicating a deeper bug beyond the measure
            # mismatch originally diagnosed. Candidates for follow-up
            # investigation: (a) mode-1 primitive sign/normalisation,
            # (b) transmission matrix indexing at cross-face blocks,
            # (c) reciprocity / surface-area factor in W_oi at n ≥ 1.
            # Tracked in Issue #119 follow-up (see the measure-mismatch
            # memo + recipe scan in `derivations/diagnostics/`).
            raise NotImplementedError(
                "Rank-N per-face white BC (n_bc_modes > 1) infrastructure "
                "is present but the final closure DEGRADES k_eff instead "
                "of converging it (best recipe ~1.4 % residual at r_0/R=0.3 "
                "vs 0.077 % at N=1). The measure-mismatch fix from Issue "
                "#119 is necessary but not sufficient — the mode-1 "
                "primitive or transmission-matrix off-diagonal entries "
                "have a second bug. Tracked in Issue #119 follow-up. "
                "Use n_bc_modes=1 for rank-2 per-face (Phase F.3/F.4)."
            )
        return _build_closure_operator_rank2_white(
            geometry, r_nodes, r_wts, radii, sig_t,
            n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
            sig_t_n=sig_t_n, rv=rv,
        )

    # ── Legacy single-surface layout (rank-1 Mark / Marshak, vacuum) ─
    divisor = geometry.rank1_surface_divisor(R_cell)

    P = np.zeros((N, N_r))
    G = np.zeros((N_r, N))

    # Mode 0 — legacy Mark convention for rank-1 bit-exact regression.
    P_esc_0 = compute_P_esc(
        geometry, r_nodes, radii, sig_t,
        n_angular=n_angular, dps=dps,
    )
    G_bc_0 = compute_G_bc(
        geometry, r_nodes, radii, sig_t,
        n_surf_quad=n_surf_quad, dps=dps,
    )
    P[0, :] = rv * r_wts * P_esc_0
    G[:, 0] = sig_t_n * G_bc_0 / divisor

    # Modes n ≥ 1 — canonical Gelbard DP_{N-1} partial-current moments
    # with the (ρ_max/R)² Jacobian baked into compute_P_esc_mode.
    for n in range(1, N):
        P_esc_n = compute_P_esc_mode(
            geometry, r_nodes, radii, sig_t, n,
            n_angular=n_angular, dps=dps,
        )
        G_bc_n = compute_G_bc_mode(
            geometry, r_nodes, radii, sig_t, n,
            n_surf_quad=n_surf_quad, dps=dps,
        )
        P[n, :] = rv * r_wts * P_esc_n
        G[:, n] = sig_t_n * G_bc_n / divisor

    # Reflection operator on the mode space.
    if isinstance(reflection, str):
        reflection_map = {
            "vacuum": reflection_vacuum,
            "mark": reflection_mark,
            "marshak": reflection_marshak,
        }
        if reflection == "white":
            # n_surfaces == 1 (solid cyl/sph): "white" == "mark" (the
            # single-surface form's self-reflection IS white BC for
            # solid geometries — the 2-boundary T-feedback path is
            # inapplicable when only one boundary exists).
            R_matrix = reflection_mark(N)
        elif reflection not in reflection_map:
            raise ValueError(
                f"Unknown reflection = {reflection!r}; expected one of "
                f"{list(reflection_map)} or a {N}×{N} matrix."
            )
        else:
            R_matrix = reflection_map[reflection](N)
    else:
        R_matrix = np.asarray(reflection, dtype=float)

    return BoundaryClosureOperator(P=P, G=G, R=R_matrix)


def _build_closure_operator_rank2_white(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    n_angular: int,
    n_surf_quad: int,
    dps: int,
    sig_t_n: np.ndarray,
    rv: np.ndarray,
) -> BoundaryClosureOperator:
    r"""Rank-2 per-face white-BC closure assembly (Phase F.3).

    Internal helper called by :func:`build_closure_operator` for
    Class-A cells (:math:`N_{\rm surfaces} = 2`) under
    ``reflection="white"``.

    Builds :math:`P \in \mathbb{R}^{2 \times N_r}`,
    :math:`G \in \mathbb{R}^{N_r \times 2}` from the per-face
    escape / response primitives, and
    :math:`R = (I - W)^{-1} \in \mathbb{R}^{2 \times 2}` from the
    surface-to-surface transmission. For slab,
    :math:`W = T\,\bigl(\begin{smallmatrix}0 & 1\\1 & 0\end{smallmatrix}\bigr)`
    with :math:`T = 2\,E_3(\tau_{\rm total})`. Hollow cyl/sph will be
    added in Phase F.4 via geometry-specific transmission integrals.

    Per-face divisor convention: each surface has its own characteristic
    area, so the divisor is the per-face area (slab: 1 per unit
    transverse area per face; cyl: :math:`2\pi R_{\rm out}` / :math:`2\pi r_0`
    per unit z, cancelled against the per-face volume-element ratio;
    sph: :math:`4\pi R_{\rm out}^2` / :math:`4\pi r_0^2`). For slab
    this reduces to divisor_outer = divisor_inner = 1.
    """
    N_r = len(r_nodes)
    P = np.zeros((2, N_r))
    G = np.zeros((N_r, 2))

    P_esc_outer_arr = compute_P_esc_outer(
        geometry, r_nodes, radii, sig_t,
        n_angular=n_angular, dps=dps,
    )
    P_esc_inner_arr = compute_P_esc_inner(
        geometry, r_nodes, radii, sig_t,
        n_angular=n_angular, dps=dps,
    )
    G_bc_outer_arr = compute_G_bc_outer(
        geometry, r_nodes, radii, sig_t,
        n_surf_quad=n_surf_quad, dps=dps,
    )
    G_bc_inner_arr = compute_G_bc_inner(
        geometry, r_nodes, radii, sig_t,
        n_surf_quad=n_surf_quad, dps=dps,
    )

    # Per-surface divisor. For slab both faces have unit area
    # (per unit transverse), divisor = 1 each.
    if geometry.kind == "slab-polar":
        div_outer = 1.0
        div_inner = 1.0
    elif geometry.kind == "cylinder-1d":
        div_outer = float(radii[-1])
        div_inner = float(geometry.inner_radius)
    else:  # sphere-1d
        R_out = float(radii[-1])
        r_in = float(geometry.inner_radius)
        div_outer = R_out * R_out
        div_inner = r_in * r_in

    # Mode layout (rows 0 = outer, 1 = inner) — matches plan §3.2.
    P[0, :] = rv * r_wts * P_esc_outer_arr
    P[1, :] = rv * r_wts * P_esc_inner_arr
    G[:, 0] = sig_t_n * G_bc_outer_arr / div_outer
    G[:, 1] = sig_t_n * G_bc_inner_arr / div_inner

    # Surface-to-surface transmission matrix.
    if geometry.kind == "slab-polar":
        L = float(radii[-1])
        T = compute_slab_transmission(L, radii, sig_t, dps=dps)
        W = T * np.array([[0.0, 1.0], [1.0, 0.0]])
    elif geometry.kind == "cylinder-1d":
        # Hollow cylinder (Phase F.4): Lambert-emission chord transmission
        # with the out-of-plane θ fold into Ki_3. See
        # :func:`compute_hollow_cyl_transmission` docstring for the full
        # derivation; the 2×2 W matrix below drives the partial-current
        # inversion R = (I - W)^{-1}.
        r0 = float(geometry.inner_radius)
        R_out = float(radii[-1])
        W = compute_hollow_cyl_transmission(r0, R_out, radii, sig_t, dps=dps)
    elif geometry.kind == "sphere-1d":
        # Hollow sphere (Phase F.4): same chord decomposition as cylinder
        # but with the bare exp(-τ) kernel (no Ki_3 θ-fold — sphere's
        # 3-D angular integration is explicit). See
        # :func:`compute_hollow_sph_transmission` for the derivation.
        r0 = float(geometry.inner_radius)
        R_out = float(radii[-1])
        W = compute_hollow_sph_transmission(r0, R_out, radii, sig_t, dps=dps)
    else:  # pragma: no cover
        raise NotImplementedError(
            f"Rank-2 white BC for kind={geometry.kind!r} not implemented."
        )

    R_matrix = reflection_white_rank2(W)
    return BoundaryClosureOperator(P=P, G=G, R=R_matrix)


def _build_closure_operator_rank_n_white(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    n_angular: int,
    n_surf_quad: int,
    dps: int,
    n_bc_modes: int,
    sig_t_n: np.ndarray,
    rv: np.ndarray,
) -> BoundaryClosureOperator:
    r"""Rank-:math:`N` per-face Marshak white-BC closure (Phase F.5 /
    Issue #119).

    Implemented for hollow sphere. Mode layout ``[outer_0, ...,
    outer_{N-1}, inner_0, ..., inner_{N-1}]`` gives a :math:`(2N)`-wide
    mode space, all modes in the **Marshak / partial-current-moment**
    basis.

    The reflection

    .. math::

       R_{\rm eff} = (I_{2N} - W_N)^{-1}

    combines the surface-to-surface transmission feedback with the
    mode-wise white BC :math:`J^-_m = J^+_m`. In the Marshak basis
    this BC is diagonal; the transmission matrix
    :func:`compute_hollow_sph_transmission_rank_n` is already in the
    Marshak basis (cos θ · sin θ integrand = µ dµ), so the closure
    requires no basis conversion once the per-face primitives are in
    the same basis.

    The :math:`P` and :math:`G` tensors stack mode-by-mode per-face
    primitives:

    .. math::

       P[kN + n, j] &= r_j^{d-1}\,w_j\,P_{\rm esc, face_k, marshak}^{(n)}(r_j), \\
       G[i, kN + m] &= \Sigma_t(r_i)\,G_{{\rm bc}, {\rm face_k}, {\rm marshak}}^{(m)}(r_i) / A_k,

    where the primitives are the
    :func:`compute_P_esc_{outer,inner}_mode_marshak` /
    :func:`compute_G_bc_{outer,inner}_mode_marshak` variants that
    include the :math:`µ` partial-current weight in the integrand.
    This is the **Phase F.5 fix for Issue #119** — the Lambert-basis
    (no-µ) mode primitives used by
    :func:`_build_closure_operator_rank2_white` mismatch the Marshak
    basis of the transmission matrix at N ≥ 2, producing a :math:`G
    (I−W)^{-1} P` product that couples incompatible inner products.

    Currently implemented: sphere-1d (hollow). Slab and cylinder raise
    :class:`NotImplementedError` from the Marshak mode primitives.
    """
    if geometry.kind != "sphere-1d" or geometry.inner_radius <= 0.0:
        raise NotImplementedError(
            f"Rank-N per-face white BC implemented for hollow sphere "
            f"only (Phase F.5 initial scope); got kind="
            f"{geometry.kind!r}, inner_radius={geometry.inner_radius}. "
            f"Slab and cylinder rank-N per-face are Issue #120 / #121 "
            f"follow-ups."
        )
    N = int(n_bc_modes)
    N_r = len(r_nodes)
    P = np.zeros((2 * N, N_r))
    G = np.zeros((N_r, 2 * N))

    R_out = float(radii[-1])
    r_in = float(geometry.inner_radius)
    div_outer = R_out * R_out
    div_inner = r_in * r_in

    for n in range(N):
        P_out_n = compute_P_esc_outer_mode_marshak(
            geometry, r_nodes, radii, sig_t, n,
            n_angular=n_angular, dps=dps,
        )
        P_in_n = compute_P_esc_inner_mode_marshak(
            geometry, r_nodes, radii, sig_t, n,
            n_angular=n_angular, dps=dps,
        )
        G_out_n = compute_G_bc_outer_mode_marshak(
            geometry, r_nodes, radii, sig_t, n,
            n_surf_quad=n_surf_quad, dps=dps,
        )
        G_in_n = compute_G_bc_inner_mode_marshak(
            geometry, r_nodes, radii, sig_t, n,
            n_surf_quad=n_surf_quad, dps=dps,
        )
        # Mode layout: [outer_0, ..., outer_{N-1}, inner_0, ..., inner_{N-1}]
        P[n, :] = rv * r_wts * P_out_n
        P[N + n, :] = rv * r_wts * P_in_n
        G[:, n] = sig_t_n * G_out_n / div_outer
        G[:, N + n] = sig_t_n * G_in_n / div_inner

    W = compute_hollow_sph_transmission_rank_n(
        r_in, R_out, radii, sig_t, n_bc_modes=N, dps=dps,
    )
    # Rank-N white-BC reflection in the Marshak partial-current-moment
    # basis. The (2n+1) Gelbard factor is NOT applied here: both P, G,
    # and W use the same µ-weighted half-range inner product, so the
    # closure `G · (I − W)^{-1} · P` is basis-consistent. The Gelbard
    # (2n+1) weighting is a property of a DIFFERENT choice (the
    # Lambert / angular-flux-moment basis, used by the legacy
    # single-surface :func:`reflection_marshak` helper) — conflating
    # the two was the Phase F.5 measure-mismatch bug.
    R_matrix = np.linalg.inv(np.eye(2 * N) - W)
    return BoundaryClosureOperator(P=P, G=G, R=R_matrix)


def build_white_bc_correction_rank_n(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    n_angular: int = 32,
    n_surf_quad: int = 32,
    dps: int = 25,
    n_bc_modes: int = 1,
) -> np.ndarray:
    r"""Rank-:math:`N` Marshak / DP\ :sub:`N-1` white-BC correction.

    .. math::

       K_{\rm bc} \;=\; \sum_{n=0}^{N-1} u_n \otimes v_n,
       \quad
       u_n[i] = \frac{\Sigma_t(r_i)\,G_{\rm bc}^{(n)}(r_i)}
                     {A_d^{\rm divisor}(R)},
       \quad
       v_n[j] = (2n+1)\,r_j^{d-1}\,w_j\,P_{\rm esc}^{(n)}(r_j).

    For :math:`n = 0`, :math:`\tilde P_0 \equiv 1` and the
    corresponding :math:`u_0 \otimes v_0` is **identical** to the
    existing rank-1 correction built by
    :func:`build_white_bc_correction` — the :math:`n = 0` contribution
    is in fact routed through that function to preserve bit-exact
    regression. For :math:`n \ge 1`, additional rank-1 outer products
    are accumulated, each capturing one higher Legendre moment of the
    outgoing surface partial-current distribution.

    See :ref:`theory-peierls-unified` §8 for the mathematical
    derivation and the Sanchez & McCormick 1982 §III.F.1 reference.

    .. note::

       **Partial fix landed 2026-04-18.** The
       :math:`(\rho_{\max}/R)^2` surface-to-observer Jacobian
       factor in :func:`compute_P_esc_mode` (see
       :eq:`peierls-rank-n-P-esc-moment`) replaces the old
       plain-weight form. Headline results for the bare homogeneous
       1G 1-region white-BC eigenvalue (:math:`k_\infty = 1.5`):

       .. list-table:: Rank-:math:`N` :math:`k_{\rm eff}` error (fixed)
          :header-rows: 1

          * - Geometry
            - :math:`R` [MFP]
            - N=1
            - N=2
            - N=3 (cyl diverges)
          * - Sphere
            - 1.0
            - 26.9 %
            - **1.22 %**
            - 2.5 %
          * - Sphere
            - 10.0
            - 0.28 %
            - **0.17 %**
            - 0.17 %
          * - Cylinder
            - 1.0
            - 20.9 %
            - **8.3 %**
            - 26.7 %
          * - Cylinder
            - 10.0
            - 1.14 %
            - **1.06 %**
            - 1.04 %

       The rank-1 → rank-2 step is a clean Marshak-ladder
       improvement for both geometries, and rank-N no longer
       degrades thick-cell convergence. Conservation
       (:math:`K\cdot\mathbf 1 = \Sigma_t` for pure absorber) also
       **improves** with rank-N instead of degrading — see
       ``tests/derivations/test_peierls_rank_n_conservation.py``.

       **Remaining work** (tracked in Issue #112):

       - **Cylinder** high-:math:`N` still diverges because
         :func:`compute_G_bc_mode` uses the 2-D projected cosine in
         the surface-centred :math:`\mathrm{Ki}_1/d` integrand. The
         canonical DP\ :sub:`N` closure needs the 3-D
         :math:`\mu_{s,3D} = \sin\theta_p \cdot \mu_{s,2D}` with
         explicit :math:`\theta_p` integration (producing higher-
         order Bickley functions :math:`\mathrm{Ki}_{2+k}`; Knyazev
         1993). Phase C of Issue #112.

       - **Sphere** plateaus at ~2.5 % for :math:`N \ge 3` at
         :math:`R = 1` MFP. Closing the plateau to <1 % at N=8
         likely requires adding a cosine weight on top of the
         Jacobian (the canonical DP\ :sub:`N` partial-current
         moment). Phase A of Issue #112 (slab DP\ :sub:`N`
         calibration) will anchor this.

       For ``n_bc_modes = 1`` (default) the function remains
       bit-exactly equivalent to :func:`build_white_bc_correction`.
       For ``n_bc_modes = 2`` (DP\ :sub:`1` closure) the solver
       converges with the canonical rank-1-to-rank-2 Marshak
       improvement on both geometries. For ``n_bc_modes ≥ 3`` on
       cylinder, results become unreliable until Phase C lands.

    .. note::

       **Thin wrapper** around :func:`build_closure_operator` +
       :meth:`BoundaryClosureOperator.as_matrix`. The factored form
       is the structural representation — assembling the dense
       matrix here is a convenience for the direct-solve eigenvalue
       iteration in :func:`solve_peierls_1g`. Callers that want
       matrix-free application (e.g. large :math:`N_r`, iterative
       solvers) should call :func:`build_closure_operator` and use
       :meth:`BoundaryClosureOperator.apply` directly.
    """
    op = build_closure_operator(
        geometry, r_nodes, r_wts, radii, sig_t,
        n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
        n_bc_modes=n_bc_modes, reflection="marshak",
    )
    return op.as_matrix()


# ═══════════════════════════════════════════════════════════════════════
# Unified solution container + eigenvalue driver
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PeierlsSolution:
    """Result of a polar-form Peierls Nyström solve on a 1-D curvilinear cell.

    Carries both the nodal flux and the piecewise-Lagrange basis
    needed for arbitrary-:math:`r` interpolation via :meth:`phi`.
    """

    r_nodes: np.ndarray
    phi_values: np.ndarray
    k_eff: float | None
    cell_radius: float
    n_groups: int
    geometry_kind: str
    n_quad_r: int
    n_quad_angular: int
    precision_digits: int
    panel_bounds: list[tuple[float, float, int, int]] | None = None

    def phi(self, r: np.ndarray, g: int = 0) -> np.ndarray:
        r"""Evaluate flux at arbitrary radii via the piecewise Lagrange basis."""
        r = np.asarray(r, dtype=float).ravel()
        out = np.empty_like(r)

        if self.panel_bounds is None:
            return np.interp(r, self.r_nodes, self.phi_values[:, g])

        for idx, r_eval in enumerate(r):
            L = lagrange_basis_on_panels(
                self.r_nodes, self.panel_bounds, float(r_eval),
            )
            out[idx] = float(np.dot(L, self.phi_values[:, g]))
        return out


_DEPRECATED_CLOSURE_ALIASES = {
    "white": "white_rank1_mark",
    "white_rank2": "white_f4",
}


def _resolve_closure_name(boundary: str, *, user_stacklevel: int) -> str:
    """Resolve deprecated closure aliases to canonical names.

    Parameters
    ----------
    boundary
        The user-facing closure name (may be a deprecated alias).
    user_stacklevel
        The ``stacklevel`` value to hand to :func:`warnings.warn` so
        the resulting ``DeprecationWarning`` points at the user's
        call site, not at an intermediate wrapper frame.

        Call from :func:`solve_peierls_mg` directly: pass ``3``
        (``warn -> _resolve -> solve_peierls_mg -> user``).
        Call from :func:`solve_peierls_1g` (which itself calls
        :func:`solve_peierls_mg`): pass ``3`` from inside the 1G
        wrapper *before* calling ``solve_peierls_mg``, so the
        MG-side resolution is a no-op on the canonical name.
    """
    if boundary in _DEPRECATED_CLOSURE_ALIASES:
        new_name = _DEPRECATED_CLOSURE_ALIASES[boundary]
        import warnings as _warnings
        _warnings.warn(
            f"boundary={boundary!r} is a deprecated alias; the canonical "
            f"name is {new_name!r}. The old name still works for this "
            f"release. See docs/theory/peierls_unified.rst "
            f"§theory-peierls-naming.",
            DeprecationWarning,
            stacklevel=user_stacklevel,
        )
        return new_name
    return boundary


def _build_full_K_per_group(
    geometry: CurvilinearGeometry,
    r_nodes: np.ndarray,
    r_wts: np.ndarray,
    panels: list,
    radii: np.ndarray,
    sig_t_g: np.ndarray,
    closure: str,
    *,
    n_angular: int,
    n_rho: int,
    n_surf_quad: int,
    n_bc_modes: int,
    dps: int,
) -> np.ndarray:
    """Assemble K = K_vol + K_bc for a single group at per-region Σ_t,g.

    Internal helper shared by the MG driver and the 1G wrapper. The
    closure arg is pre-resolved (no alias handling here).
    """
    K = build_volume_kernel(
        geometry, r_nodes, panels, radii, sig_t_g,
        n_angular=n_angular, n_rho=n_rho, dps=dps,
    )
    if closure == "vacuum":
        return K
    if closure == "white_rank1_mark":
        K_bc = build_white_bc_correction_rank_n(
            geometry, r_nodes, r_wts, radii, sig_t_g,
            n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
            n_bc_modes=n_bc_modes,
        )
        return K + K_bc
    if closure == "white_f4":
        if n_bc_modes > 1:
            raise NotImplementedError(
                "closure='white_f4' with n_bc_modes > 1 is not a "
                "shipped closure. The rank-N Marshak per-face path "
                "was falsified by the 2026-04-22 research program "
                "(see research log L21 and Sphinx §peierls-rank-n-"
                "per-face-closeout). Use n_bc_modes=1 for F.4, or "
                "closure='white_rank1_mark' for rank-1 Mark."
            )
        if getattr(geometry, "n_surfaces", 1) == 1:
            import warnings as _warnings
            _warnings.warn(
                f"closure='white_f4' on a 1-surface (solid) geometry "
                f"(kind={geometry.kind!r}, inner_radius="
                f"{getattr(geometry, 'inner_radius', 0.0)}) silently "
                f"collapses to rank-1 Mark because there is no second-"
                f"face coupling. Use closure='white_rank1_mark' to "
                f"make the intent explicit. This silent-collapse "
                f"behavior will become a ValueError in a future "
                f"release.",
                DeprecationWarning,
                stacklevel=4,
            )
        op = build_closure_operator(
            geometry, r_nodes, r_wts, radii, sig_t_g,
            n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
            n_bc_modes=1, reflection="white",
        )
        return K + op.as_matrix()
    raise ValueError(
        f"closure must be 'vacuum', 'white_rank1_mark', or "
        f"'white_f4' (or the deprecated aliases 'white' / "
        f"'white_rank2'); got {closure!r}"
    )


def solve_peierls_mg(
    geometry: CurvilinearGeometry,
    radii: np.ndarray,
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    chi: np.ndarray,
    *,
    boundary: str = "vacuum",
    n_panels_per_region: int = 2,
    p_order: int = 5,
    n_angular: int = 24,
    n_rho: int = 24,
    n_surf_quad: int = 24,
    dps: int = 25,
    max_iter: int = 300,
    tol: float = 1e-10,
    n_bc_modes: int = 1,
) -> PeierlsSolution:
    r"""Unified multi-group Peierls eigenvalue driver for curvilinear geometry.

    Generalisation of :func:`solve_peierls_1g` to ``ng ≥ 1`` energy
    groups with downscatter / upscatter coupling and :math:`\chi`-weighted
    fission. The per-group Peierls equation is

    .. math::

       \Sigma_{t,g}(r_i)\,\varphi_g(r_i) \;=\;
         \sum_j K^{(g)}_{ij}\!\!
         \sum_{g'}\!\bigl(
           \Sigma_{s,g'\to g}(r_j)\,\varphi_{g'}(r_j)
           + \tfrac{1}{k}\,\chi_g(r_i)\,\nu\Sigma_{f,g'}(r_j)\,\varphi_{g'}(r_j)
         \bigr),

    assembled into a block :math:`(N \cdot n_g)` × :math:`(N \cdot n_g)`
    linear system with row index ``i * ng + g`` (node-major) and solved
    by fission-source power iteration. The volume kernel
    :math:`K^{(g)}` differs across groups only through :math:`\Sigma_{t,g}`;
    the closure (vacuum / white_rank1_mark / white_f4) is rebuilt once
    per group from the per-group :math:`\Sigma_{t,g}` trace.

    The closure primitives (``build_closure_operator``,
    ``build_white_bc_correction_rank_n``) are group-local by
    construction (no cross-group coupling through the reflection
    operator — verified during Issue #104 scoping; see research log
    L21 / Sphinx §peierls-rank-n-per-face-closeout for the F.4 case).

    Parameters
    ----------
    geometry
        Curvilinear geometry instance (slab-polar / cylinder-1d /
        sphere-1d, optionally hollow).
    radii
        Outer radii per region, shape ``(n_regions,)``. The inner
        cavity radius (if any) comes from ``geometry.inner_radius``.
    sig_t
        Total cross section, shape ``(n_regions, ng)``.
    sig_s
        P\ :sub:`0` scattering matrix per region, shape
        ``(n_regions, ng, ng)``. Convention: ``sig_s[r, g_src, g_dst]``
        is the rate of scatter **from** group ``g_src`` **into** group
        ``g_dst`` at region ``r`` — i.e., the first group index is
        the source and the second is the destination. This matches
        the XS library (:mod:`orpheus.derivations._xs_library`) and
        the slab driver
        :func:`~orpheus.derivations.peierls_slab.solve_peierls_eigenvalue`.
        Under this convention downscatter (fast → thermal, i.e.,
        ``g_src < g_dst`` with group 0 = fast) sits in the
        upper-triangular entries.
    nu_sig_f
        :math:`\nu\Sigma_f` per region and group, shape
        ``(n_regions, ng)``.
    chi
        Fission emission spectrum per region and group, shape
        ``(n_regions, ng)``. Must sum to 1 along the group axis
        per region (not checked — caller's responsibility).
    boundary, n_panels_per_region, p_order, n_angular, n_rho,
    n_surf_quad, dps, max_iter, tol, n_bc_modes
        See :func:`solve_peierls_1g`; semantics are unchanged.
        Tolerances apply to the scalar eigenvalue iteration only.

    Returns
    -------
    PeierlsSolution
        ``phi_values`` has shape ``(N, ng)``; ``n_groups == ng``.

    Notes
    -----
    **ng = 1 bit-match guarantee.** For ``ng == 1`` with ``chi == 1``
    the MG path reduces algebraically to the original 1G assembly
    (same K, same A / B, same power iteration). This is enforced by
    the regression test ``test_mg_bitmatch_1g_*`` and is a prerequisite
    for preserving the behaviour of all 1G callers via the
    :func:`solve_peierls_1g` wrapper.
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    sig_s = np.asarray(sig_s, dtype=float)
    nu_sig_f = np.asarray(nu_sig_f, dtype=float)
    chi = np.asarray(chi, dtype=float)

    if sig_t.ndim != 2:
        raise ValueError(
            f"sig_t must be shape (n_regions, ng); got ndim={sig_t.ndim}, "
            f"shape={sig_t.shape}. Use solve_peierls_1g for 1-D Σ_t input."
        )
    n_regions, ng = sig_t.shape
    if sig_s.shape != (n_regions, ng, ng):
        raise ValueError(
            f"sig_s must be shape (n_regions={n_regions}, ng={ng}, "
            f"ng={ng}); got {sig_s.shape}."
        )
    if nu_sig_f.shape != (n_regions, ng):
        raise ValueError(
            f"nu_sig_f must be shape (n_regions={n_regions}, ng={ng}); "
            f"got {nu_sig_f.shape}."
        )
    if chi.shape != (n_regions, ng):
        raise ValueError(
            f"chi must be shape (n_regions={n_regions}, ng={ng}); "
            f"got {chi.shape}."
        )

    # stacklevel=3 resolves:  warn -> _resolve_closure_name ->
    # solve_peierls_mg -> user's call site.
    closure = _resolve_closure_name(boundary, user_stacklevel=3)

    # Forward hollow-cell cavity exclusion to the radial mesh builder
    # (identical to solve_peierls_1g — Issue #119).
    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region, p_order, dps=dps,
        inner_radius=getattr(geometry, "inner_radius", 0.0) or 0.0,
    )
    N = len(r_nodes)

    # Per-group K = K_vol + K_bc. This is the only group-local loop.
    K_per_group = np.empty((ng, N, N))
    for g in range(ng):
        K_per_group[g] = _build_full_K_per_group(
            geometry, r_nodes, r_wts, panels, radii, sig_t[:, g],
            closure,
            n_angular=n_angular, n_rho=n_rho, n_surf_quad=n_surf_quad,
            n_bc_modes=n_bc_modes, dps=dps,
        )

    # Per-node XS (look up the annulus-region index per radial node).
    k_annulus = np.array(
        [geometry.which_annulus(float(r_nodes[i]), radii) for i in range(N)],
        dtype=int,
    )
    sig_t_n = sig_t[k_annulus, :]          # (N, ng)
    sig_s_n = sig_s[k_annulus, :, :]       # (N, ng, ng)
    nu_f_n  = nu_sig_f[k_annulus, :]       # (N, ng)
    chi_n   = chi[k_annulus, :]            # (N, ng)

    # Block (N·ng) × (N·ng) assembly — node-major indexing row = i*ng + g.
    # Convention bit-matches slab's _build_system_matrices in
    # peierls_slab.py:243 (row = i*ng + ge, col = j*ng + gs).
    dim = N * ng
    A = np.zeros((dim, dim))
    B = np.zeros((dim, dim))

    # Diagonal Σ_t term: A[i*ng+g, i*ng+g] = Σ_t,g(r_i).
    for i in range(N):
        for g in range(ng):
            A[i * ng + g, i * ng + g] = sig_t_n[i, g]

    # Off-diagonal scatter / fission operators. Σ_t-LHS convention
    # (matches solve_peierls_1g): A = diag(Σ_t) − K·Σ_s, B = K·χ·νΣ_f.
    # Access sig_s_n[j, gs, ge] as "scatter gs → ge at node j" per the
    # slab reference pattern.
    for ge in range(ng):
        Kg = K_per_group[ge]
        for i in range(N):
            chi_ie = chi_n[i, ge]
            row = i * ng + ge
            for j in range(N):
                kij = Kg[i, j]
                if kij == 0.0:
                    continue
                for gs in range(ng):
                    col = j * ng + gs
                    A[row, col] -= kij * sig_s_n[j, gs, ge]
                    B[row, col] += kij * chi_ie * nu_f_n[j, gs]

    # Fission-source power iteration — bit-identical structure to the
    # 1G path (just acts on a dim-wide vector instead of N-wide).
    phi = np.ones(dim)
    k_val = 1.0
    B_phi = B @ phi
    prod_old = np.abs(B_phi).sum()
    for it in range(max_iter):
        q = B_phi / k_val
        phi_new = np.linalg.solve(A, q)
        B_phi_new = B @ phi_new
        prod_new = np.abs(B_phi_new).sum()
        k_new = k_val * prod_new / prod_old if prod_old > 0 else k_val
        nrm = np.abs(phi_new).sum()
        if nrm > 0:
            phi_new = phi_new / nrm
        B_phi_norm = B @ phi_new
        prod_norm = np.abs(B_phi_norm).sum()
        converged = abs(k_new - k_val) < tol and it > 5
        phi, k_val = phi_new, k_new
        B_phi, prod_old = B_phi_norm, prod_norm
        if converged:
            break

    # Reshape (N*ng,) → (N, ng) for the dataclass; node-major layout
    # means phi[i*ng + g] goes to phi_values[i, g].
    phi_values = phi.reshape(N, ng)

    return PeierlsSolution(
        r_nodes=r_nodes,
        phi_values=phi_values,
        k_eff=float(k_val),
        cell_radius=float(radii[-1]),
        n_groups=ng,
        geometry_kind=geometry.kind,
        n_quad_r=N,
        n_quad_angular=n_angular * n_rho,
        precision_digits=dps,
        panel_bounds=panels,
    )


def solve_peierls_1g(
    geometry: CurvilinearGeometry,
    radii: np.ndarray,
    sig_t: np.ndarray,
    sig_s: np.ndarray,
    nu_sig_f: np.ndarray,
    *,
    boundary: str = "vacuum",
    n_panels_per_region: int = 2,
    p_order: int = 5,
    n_angular: int = 24,
    n_rho: int = 24,
    n_surf_quad: int = 24,
    dps: int = 25,
    max_iter: int = 300,
    tol: float = 1e-10,
    n_bc_modes: int = 1,
) -> PeierlsSolution:
    r"""Unified 1-group Peierls eigenvalue driver for curvilinear geometry.

    The equation

    .. math::

       \Sigma_{t,i}\,\varphi_i \;=\;
         \sum_j K_{ij}\,\bigl(\Sigma_{s,j}\varphi_j + \tfrac1k\,\nu\Sigma_{f,j}\varphi_j\bigr)

    with :math:`K = K_{\rm vol} + K_{\rm bc}\,\delta_{\rm white}` is
    recast as :math:`\tilde A\,\varphi = (1/k)\,\tilde B\,\varphi`
    with :math:`\tilde A = \mathrm{diag}(\Sigma_t) - K\cdot
    \mathrm{diag}(\Sigma_s)` and :math:`\tilde B = K\cdot
    \mathrm{diag}(\nu\Sigma_f)`. Fission-source power iteration.

    As of Issue #104 (2026-04-24) this function is a thin wrapper over
    :func:`solve_peierls_mg` with ``ng = 1`` and a synthesised
    :math:`\chi = 1`. The ng=1 MG path is algebraically identical to the
    original 1G assembly (verified by ``test_mg_bitmatch_1g_*``), so
    all downstream 1G callers continue to work unchanged.
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    sig_s = np.asarray(sig_s, dtype=float)
    nu_sig_f = np.asarray(nu_sig_f, dtype=float)

    # Coerce per-region 1-D arrays to (n_regions, 1) for the MG path.
    if sig_t.ndim != 1:
        raise ValueError(
            f"solve_peierls_1g: sig_t must be a 1-D per-region array; "
            f"got shape={sig_t.shape}. Use solve_peierls_mg for ng > 1."
        )
    n_regions = sig_t.shape[0]
    sig_t_mg = sig_t[:, np.newaxis]                                   # (n_r, 1)
    sig_s_mg = sig_s.reshape(n_regions, 1, 1)                         # (n_r, 1, 1)
    nu_sig_f_mg = nu_sig_f[:, np.newaxis]                             # (n_r, 1)
    chi_mg = np.ones((n_regions, 1))                                  # χ = 1 for 1G

    # Resolve deprecated aliases here so the DeprecationWarning points
    # at the user's call to solve_peierls_1g rather than at the internal
    # solve_peierls_mg call. stacklevel=3 resolves to:
    # warn -> _resolve_closure_name -> solve_peierls_1g -> user.
    canonical_boundary = _resolve_closure_name(boundary, user_stacklevel=3)

    sol = solve_peierls_mg(
        geometry, radii, sig_t_mg, sig_s_mg, nu_sig_f_mg, chi_mg,
        boundary=canonical_boundary,
        n_panels_per_region=n_panels_per_region, p_order=p_order,
        n_angular=n_angular, n_rho=n_rho, n_surf_quad=n_surf_quad,
        dps=dps, max_iter=max_iter, tol=tol, n_bc_modes=n_bc_modes,
    )
    return sol
