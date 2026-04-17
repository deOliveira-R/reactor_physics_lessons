r"""Peierls integral equation reference for cylindrical CP verification.

Solves the 1-D radial Peierls (integral transport) equation on a bare
or annular cylinder via Nyström quadrature at mpmath precision,
producing :class:`ContinuousReferenceSolution` objects whose operator
form is ``"integral-peierls"``.

The cylindrical Peierls equation for a radially-symmetric emission
density :math:`q(r)` on :math:`0 \le r \le R` reads (Sanchez &
McCormick, *Nucl. Sci. Eng.* **80** (1982) Eqs. 47–49; Hébert 2020
§3.5; Stamm'ler & Abbate 1983 §6.2–6.3):

.. math::

   \Sigma_t(r)\,\varphi(r)
     \;=\; \frac{1}{\pi}
       \int_{0}^{\min(r,R)}\!\mathrm{d}y
       \int_{y}^{R}
         \bigl[\mathrm{Ki}_1\bigl(\tau^{+}(r,r',y)\bigr)
             + \mathrm{Ki}_1\bigl(\tau^{-}(r,r',y)\bigr)\bigr]\,
         \frac{q(r')\,r'}{\sqrt{r'^{2}-y^{2}}}\,\mathrm{d}r'
     \;+\; S_{\rm bc}(r)

where for a chord of impact parameter :math:`y`, a point at radius
:math:`r` sits at signed chord position :math:`\pm s_r` with
:math:`s_r = \sqrt{r^{2}-y^{2}}`, and the two optical-path branches
are

.. math::

   \tau^{+}(r,r',y) &= \int_{s_{r}}^{s_{r'}} \Sigma_{t}\bigl(r(s)\bigr)\,\mathrm{d}s
     &&\text{(same-side chord integral)} \\
   \tau^{-}(r,r',y) &= \int_{-s_{r}}^{s_{r'}} \Sigma_{t}\bigl(r(s)\bigr)\,\mathrm{d}s
     &&\text{(through-centre chord integral)}.

Key differences from :mod:`peierls_slab` (documented at length in
``docs/theory/collision_probability.rst``):

- **Kernel**: :math:`\mathrm{Ki}_1` (Bickley–Naylor order 1), not
  :math:`E_1`. The slab's :math:`E_1` comes from integrating the 1-D
  point kernel over polar angle; the cylinder's :math:`\mathrm{Ki}_1`
  comes from integrating the 3-D point kernel over the infinite
  axial direction, leaving the 2-D transverse integral in
  :math:`(y, r')`.
- **Singularity**: the slab integrand has a log singularity
  *in the kernel* (:math:`E_1 \sim -\ln z`) cured by singularity
  subtraction. The cylinder integrand has an inverse-square-root
  singularity :math:`1/\sqrt{r'^{2}-y^{2}}` *in the Jacobian*; it is
  absorbed by the natural Chebyshev-of-second-kind product rule.
- **Prefactor**: :math:`1/\pi`, not :math:`1/2`. Sanchez 1982
  Eq. (47) — the :math:`1/\pi` absorbs the :math:`\mathrm{d}y/2` from
  the half-plane chord sweep and the factor 2 from pairing
  :math:`\pm\mu` directions.
- **White BC closure**: rank-:math:`N_y` dense Schur block
  (continuous lateral surface), not the slab's rank-2 E₂ outer
  product (two discrete faces).

This module is the Phase-4.2 deliverable of the verification campaign.
It is the independent reference against which the flat-source CP
cylinder solver (:mod:`orpheus.cp.solver` on ``cyl1D`` meshes and
:mod:`orpheus.derivations.cp_cylinder`) is verified.

.. note::

   This is the C2 scaffold. The Nyström kernel builder, eigenvalue
   power iteration, and ``continuous_cases()`` registration land in
   subsequent commits (C3–C7 of the Phase-4.2 plan).
"""

from __future__ import annotations

from dataclasses import dataclass

import mpmath
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Composite Gauss–Legendre y-quadrature with breakpoints at annular radii
# ═══════════════════════════════════════════════════════════════════════

def _gl_nodes_weights(n: int, dps: int) -> tuple[list, list]:
    """*n*-point Gauss–Legendre on [-1, 1] at *dps* decimal digits."""
    with mpmath.workdps(dps):
        nm, wm = mpmath.gauss_quadrature(n, "legendre")
        return [nm[i] for i in range(n)], [wm[i] for i in range(n)]


def _map_gl_to(nodes, weights, a, b):
    """Map GL nodes/weights from [-1, 1] to [a, b] at mpmath precision."""
    h = (b - a) / 2
    m = (a + b) / 2
    return [m + h * t for t in nodes], [h * w for w in weights]


def composite_gl_y(
    radii: np.ndarray,
    n_panels_per_region: int,
    p_order: int,
    dps: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int, int]]]:
    r"""Composite Gauss–Legendre quadrature for the y-integration.

    For the cylinder Peierls equation, the outer integration variable
    is the chord impact parameter :math:`y \in [0, R]`. The integrand
    :math:`y \mapsto \mathrm{Ki}_1(\tau^\pm(r, r', y))` has **corners**
    at each annular radius :math:`r_k` (where a chord at :math:`y`
    transitions from crossing annulus :math:`k` to only grazing it),
    so the quadrature uses a composite GL rule with breakpoints at
    each :math:`r_k`.

    Each of the :math:`N` annular segments :math:`[r_{k-1}, r_k]`
    carries ``n_panels_per_region`` panels of order ``p_order``.

    Parameters
    ----------
    radii : np.ndarray, shape (N,)
        Outer radii of the :math:`N` concentric annuli,
        :math:`0 < r_1 < \dots < r_N = R`.
    n_panels_per_region : int
        Number of GL panels per annular segment.
    p_order : int
        GL order per panel.
    dps : int
        mpmath working precision.

    Returns
    -------
    y_pts : np.ndarray, shape (n_y,)
        Quadrature nodes in :math:`[0, R]`, in ascending order, at
        double precision (float).
    y_wts : np.ndarray, shape (n_y,)
        Quadrature weights.
    panel_bounds : list of (pa, pb, i_start, i_end)
        Breakdown of the composite rule: one tuple per panel, with
        panel endpoints and the slice of ``y_pts`` / ``y_wts`` that
        lives on it.
    """
    radii = np.asarray(radii, dtype=float)
    gl_ref, gl_wt = _gl_nodes_weights(p_order, dps)

    breakpoints = [mpmath.mpf(0)] + [mpmath.mpf(float(r)) for r in radii]
    y_all: list = []
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
                xp, wp = _map_gl_to(gl_ref, gl_wt, pa, pb)
                i0 = len(y_all)
                y_all.extend(xp)
                w_all.extend(wp)
                panel_bounds.append((float(pa), float(pb), i0, len(y_all)))

    y_pts = np.array([float(y) for y in y_all])
    y_wts = np.array([float(w) for w in w_all])
    return y_pts, y_wts, panel_bounds


# ═══════════════════════════════════════════════════════════════════════
# τ⁺ / τ⁻ optical-path walker
# ═══════════════════════════════════════════════════════════════════════

def optical_depths_pm(
    r: float,
    r_prime: float,
    y_pts: np.ndarray,
    radii: np.ndarray,
    sig_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Optical paths :math:`\tau^{+}(r,r',y)` and :math:`\tau^{-}(r,r',y)`.

    For a chord of impact parameter :math:`y`, a point at radius
    :math:`\rho \ge y` sits at signed chord position
    :math:`\pm s_\rho` with :math:`s_\rho = \sqrt{\rho^{2}-y^{2}}`.
    The two Peierls-kernel branches integrate
    :math:`\Sigma_t` along:

    - **Same-side (τ⁺)**: from :math:`s_r` to :math:`s_{r'}` on the
      positive branch (both points on the same side of the
      perpendicular foot).
    - **Through-centre (τ⁻)**: from :math:`s_r` to :math:`-s_{r'}`,
      crossing the chord midpoint.

    The integrand :math:`\Sigma_t(\rho(s))` is piecewise constant with
    jumps at the chord crossings of each annular radius
    :math:`r_k`, i.e. at :math:`|s| = s_{r_k}`.

    For :math:`y > r` or :math:`y > r'`, the point is *not* on this
    chord. Following Sanchez's convention we return :math:`\tau = 0`
    for such inaccessible configurations; the Nyström kernel
    multiplies by :math:`\mathrm{Ki}_1(0) = \pi/2` on those, but the
    outer integral in :math:`y` weights them as zero-measure (the
    geometric :math:`y`-range is :math:`[0, \min(r, R)]`).

    Parameters
    ----------
    r, r_prime : float
        Source and target radii (:math:`\ge 0`, :math:`\le R`).
    y_pts : np.ndarray, shape (n_y,)
        Chord impact parameters.
    radii : np.ndarray, shape (N,)
        Outer radii of the :math:`N` concentric annuli.
    sig_t : np.ndarray, shape (N,)
        Total macroscopic cross-section per annulus, for a single
        energy group.

    Returns
    -------
    tau_plus : np.ndarray, shape (n_y,)
    tau_minus : np.ndarray, shape (n_y,)
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    y_pts = np.asarray(y_pts, dtype=float)

    r = float(r)
    r_prime = float(r_prime)

    # Signed-chord positions of the endpoints. For y > r or y > r',
    # the corresponding s_r / s_r' is zero (geometrically: the radius
    # is inside the chord-inaccessible core).
    s_r = np.sqrt(np.maximum(r ** 2 - y_pts ** 2, 0.0))
    s_rp = np.sqrt(np.maximum(r_prime ** 2 - y_pts ** 2, 0.0))

    # Positive-branch annular breakpoints:
    #   s_breaks[0] = 0  (the chord midpoint / perpendicular foot)
    #   s_breaks[k] = sqrt(max(r_k² - y², 0))  for k = 1..N
    # Annulus k (0-indexed in the arrays, 1-indexed in docs) occupies
    # the positive chord interval [s_breaks[k], s_breaks[k+1]].
    N = len(radii)
    s_breaks = np.zeros((N + 1, len(y_pts)))
    for k in range(N):
        s_breaks[k + 1] = np.sqrt(np.maximum(radii[k] ** 2 - y_pts ** 2, 0.0))

    def _optical_on_positive(s_lo: np.ndarray, s_hi: np.ndarray) -> np.ndarray:
        r"""Integral of :math:`\Sigma_t(\rho(s))` over :math:`s \in [s_{lo}, s_{hi}]`
        with both endpoints on the positive branch (:math:`s_{lo} \le s_{hi}`)."""
        tau = np.zeros_like(s_lo)
        for k in range(N):
            lo = np.maximum(s_lo, s_breaks[k])
            hi = np.minimum(s_hi, s_breaks[k + 1])
            overlap = np.maximum(hi - lo, 0.0)
            tau += sig_t[k] * overlap
        return tau

    # Same-side: integrate between s_r and s_rp on the positive branch.
    s_lo = np.minimum(s_r, s_rp)
    s_hi = np.maximum(s_r, s_rp)
    tau_plus = _optical_on_positive(s_lo, s_hi)

    # Through-centre: from +s_r to -s_rp. By symmetry, this equals
    # (integral from 0 to s_r on positive branch) + (integral from 0
    # to s_rp on positive branch).
    zero = np.zeros_like(y_pts)
    tau_minus = _optical_on_positive(zero, s_r) + _optical_on_positive(zero, s_rp)

    return tau_plus, tau_minus


# ═══════════════════════════════════════════════════════════════════════
# Solution container (stub — full implementation lands with C3)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PeierlsCylinderSolution:
    """Result of a Peierls Nyström solve on a 1-D radial cylinder.

    .. note::

       Stub dataclass. The methods ``phi``, ``phi_cell_average`` and
       the full ``solve_peierls_cylinder_eigenvalue`` driver land in
       commit C3 of the Phase-4.2 plan.
    """

    r_nodes: np.ndarray
    """Radial quadrature node positions, shape ``(N,)``."""

    phi_values: np.ndarray
    """Flux at each radial node and group, shape ``(N, ng)``."""

    k_eff: float | None
    """Eigenvalue (None for fixed-source problems)."""

    cell_radius: float
    n_groups: int
    n_quad_r: int
    n_quad_y: int
    precision_digits: int
