r"""Peierls integral equation reference for cylindrical CP verification.

Solves the 1-D radial Peierls (integral transport) equation on a bare
or annular cylinder via Nyström quadrature at mpmath precision,
producing :class:`ContinuousReferenceSolution` objects whose operator
form is ``"integral-peierls"``.

The cylindrical Peierls equation for a radially-symmetric emission
density :math:`q(r)` on :math:`0 \le r \le R` is obtained by
integrating the 3-D point kernel over the infinite z-axis, yielding
the 2-D transverse Green's function
:math:`G_{2D}(|\mathbf{r}-\mathbf{r}'|) = \mathrm{Ki}_1(\Sigma_t\,
|\mathbf{r}-\mathbf{r}'|) / (2\pi\,|\mathbf{r}-\mathbf{r}'|)`:

.. math::

   \varphi(\mathbf{r})
     \;=\; \frac{1}{2\pi}\!\iint_{\rm disc}
       \frac{\mathrm{Ki}_1\!\bigl(\tau(\mathbf{r},\mathbf{r}')\bigr)}
            {|\mathbf{r}-\mathbf{r}'|}\,q(\mathbf{r}')\,\mathrm{d}^{2}r'
     + \varphi_{\rm bc}(\mathbf{r})

For the Nyström discretisation this module expresses the 2-D
integral in **polar coordinates centred at the observer**:

.. math::

   \varphi(r)
     \;=\; \frac{1}{\pi}\!
       \int_{0}^{\pi}\!\mathrm{d}\beta\!
       \int_{0}^{\rho_{\max}(r,\beta)}\!\!
         \mathrm{Ki}_1\!\bigl(\tau(r, \rho, \beta)\bigr)\,
         q\bigl(r'(r, \rho, \beta)\bigr)\,\mathrm{d}\rho
     + \varphi_{\rm bc}(r)

where :math:`\rho` is the distance from the observer along the ray
at angle :math:`\beta` (measured from the outward radial direction
at the observer), :math:`r'(r,\rho,\beta) = \sqrt{r^{2} + 2 r\rho
\cos\beta + \rho^{2}}` is the source radius, :math:`\rho_{\max}
= -r\cos\beta + \sqrt{r^{2}\cos^{2}\beta + R^{2}-r^{2}}` is the
distance to the cylinder boundary, and the prefactor :math:`1/\pi`
absorbs the :math:`1/(2\pi)` of the 2-D kernel plus a factor of 2
from the :math:`y\to-y` (β-reflection) symmetry that folds
:math:`\beta\in[0,2\pi]` to :math:`[0,\pi]`.

Compared to the equivalent chord :math:`(y, r')` form used in
Sanchez-McCormick 1982 §IV.A, the polar form has **no singular
Jacobian** — the 2-D area element :math:`\rho\,\mathrm{d}\rho\,
\mathrm{d}\beta` cancels the :math:`1/\rho` in the Green's function,
leaving a smooth integrand :math:`\mathrm{Ki}_1(\tau)\,q(r')` that
is handled cleanly by ordinary Gauss–Legendre quadrature in both
:math:`\beta` and :math:`\rho`. The chord :math:`(y, r')`
parametrisation picks up the Jacobian
:math:`1/\sqrt{(r^{2}-y^{2})(r'^{2}-y^{2})}` from the two-branch sum
:math:`|\mathrm{d}\alpha_{+}/\mathrm{d}y| + |\mathrm{d}\alpha_{-}/
\mathrm{d}y| = 2/\sqrt{\min(r,r')^{2}-y^{2}}` and carries an extra
singularity at :math:`y=\min(r,r')`. The polar form avoids this.

Key differences from :mod:`peierls_slab`:

- **Kernel**: :math:`\mathrm{Ki}_1` (Bickley–Naylor order 1), not
  :math:`E_1`. The slab's :math:`E_1` comes from integrating the 1-D
  point kernel over polar angle; the cylinder's :math:`\mathrm{Ki}_1`
  comes from integrating the 3-D point kernel over the infinite
  axial direction.
- **Singularity**: none in the polar form (:math:`\rho\,\mathrm{d}\rho\,
  \mathrm{d}\beta` cancels the :math:`1/\rho` of
  :math:`\mathrm{Ki}_1/|\mathbf{r}-\mathbf{r}'|`). Contrast the slab's
  log-singular kernel :math:`E_1(\tau)\sim -\ln\tau`.
- **Prefactor**: :math:`1/\pi`, not :math:`1/2`. Derived above.
- **Source interpolation**: because :math:`r'(\rho,\beta,r_i)` is
  generally not a quadrature node, the Nyström unknown is connected
  to :math:`q(r'_{ikm})` via Lagrange-basis interpolation on the
  radial grid — hence the kernel matrix picks up a
  :math:`L_j(r'_{ikm})` factor.
- **White BC closure**: rank-:math:`N_\rho` dense Schur block
  (continuous lateral surface; C5), not the slab's rank-2 E₂ outer
  product (two discrete faces).

The :math:`\tau^{\pm}` chord walker (:func:`optical_depths_pm`, C2)
remains part of this module: it is the primitive used to evaluate
:math:`\tau(r,\rho,\beta)` for **multi-region** problems (C4),
where the optical depth along the ray from :math:`r_i` in direction
:math:`\beta` to a source at distance :math:`\rho` decomposes into a
same-side / through-centre branch depending on whether the ray
crosses the chord midpoint.

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

from ._kernels import ki_n_mp


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
# Volume kernel assembly (polar (β, ρ) Nyström with Lagrange interpolation)
# ═══════════════════════════════════════════════════════════════════════

def _gl_float(n: int, a: float, b: float, dps: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """*n*-point GL on ``[a, b]``, returned as float arrays."""
    ref_nodes, ref_wts = _gl_nodes_weights(n, dps)
    h = (b - a) / 2
    m = (a + b) / 2
    nodes = np.array([float(m + h * t) for t in ref_nodes])
    wts = np.array([float(h * w) for w in ref_wts])
    return nodes, wts


def _lagrange_basis_on_panels(
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    r_eval: float,
) -> np.ndarray:
    r"""Evaluate the Lagrange basis :math:`L_j(r_{\rm eval})` at an
    arbitrary point, using the panel structure of the r-grid.

    On each panel :math:`[p_a, p_b]`, the Lagrange basis polynomials are
    built over that panel's nodes only; on other panels the basis is
    zero (piecewise polynomial representation). This matches how
    :class:`PeierlsCylinderSolution` would interpolate a discrete
    nodal vector back to a continuous function.

    Parameters
    ----------
    r_nodes : np.ndarray, shape (N,)
        All r-nodes across all panels.
    panel_bounds : list of (pa, pb, i_start, i_end)
        Panel layout from :func:`composite_gl_y` / :func:`composite_gl_r`.
    r_eval : float
        Point at which to evaluate the basis.

    Returns
    -------
    np.ndarray, shape (N,)
        :math:`L_j(r_{\rm eval})` for :math:`j=0,\dots,N-1`. Nonzero
        only on the panel containing :math:`r_{\rm eval}`.
    """
    N = len(r_nodes)
    L = np.zeros(N)

    # Find the panel containing r_eval (boundary right-biased)
    panel_idx = None
    for k, (pa, pb, i_start, i_end) in enumerate(panel_bounds):
        if pa <= r_eval <= pb:
            panel_idx = k
            break
    if panel_idx is None:
        # r_eval out of [0, R]: clamp to nearest panel endpoint
        if r_eval < panel_bounds[0][0]:
            panel_idx = 0
        else:
            panel_idx = len(panel_bounds) - 1

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


def _rho_max(r_obs: float, cos_beta: float, R: float) -> float:
    r"""Distance along ray at angle β from observer at :math:`r_{\rm obs}`
    to the cylinder boundary :math:`|\mathbf{r}| = R`.

    Positive root of :math:`(r_{\rm obs} + \rho\cos\beta)^{2}
    + (\rho\sin\beta)^{2} = R^{2}`.
    """
    disc = r_obs * r_obs * cos_beta * cos_beta + R * R - r_obs * r_obs
    return -r_obs * cos_beta + np.sqrt(max(disc, 0.0))


def _optical_depth_along_ray(
    r_obs: float,
    cos_beta: float,
    sin_beta: float,
    rho: float,
    radii: np.ndarray,
    sig_t: np.ndarray,
) -> float:
    r"""Optical depth from :math:`r_{\rm obs}` along direction
    :math:`(\cos\beta, \sin\beta)` over a distance :math:`\rho`.

    Homogeneous (1-region) short-circuit: :math:`\tau = \Sigma_t\,\rho`.
    Multi-region: the ray crosses annular boundaries
    :math:`r = r_k` at roots of :math:`r_{\rm obs}^{2} + 2r_{\rm obs}
    s\cos\beta + s^{2} = r_k^{2}` (a quadratic in :math:`s`); the
    optical depth is the sum of :math:`\Sigma_{t,k}\cdot\Delta s`
    over each annular segment of the ray. This routine handles
    both cases.
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    N = len(radii)

    # Fast path: homogeneous
    if N == 1:
        return float(sig_t[0]) * rho

    # Multi-region: find all boundary crossings s_k where |r(s)| = r_k
    # r(s)² = r_obs² + 2 r_obs s cos β + s²
    # Crossing r_k: s² + 2 r_obs cos β · s + (r_obs² - r_k²) = 0
    crossings = [0.0]
    for r_k in radii[:-1]:  # outer boundary handled by rho bounds
        disc = r_obs * r_obs * cos_beta * cos_beta - (r_obs * r_obs - r_k * r_k)
        if disc < 0:
            continue  # ray does not intersect this annular boundary
        sqrt_disc = np.sqrt(disc)
        s_a = -r_obs * cos_beta - sqrt_disc
        s_b = -r_obs * cos_beta + sqrt_disc
        for s in (s_a, s_b):
            if 0.0 < s < rho:
                crossings.append(s)
    crossings.append(rho)
    crossings.sort()

    # Between consecutive crossings, the ray is inside a single annulus.
    # Determine annulus index by mid-segment radius.
    tau = 0.0
    for i_seg in range(len(crossings) - 1):
        s_lo, s_hi = crossings[i_seg], crossings[i_seg + 1]
        s_mid = 0.5 * (s_lo + s_hi)
        r_mid_sq = r_obs * r_obs + 2.0 * r_obs * s_mid * cos_beta + s_mid * s_mid
        r_mid = np.sqrt(max(r_mid_sq, 0.0))
        # Annulus k contains r_mid iff r_{k-1} ≤ r_mid < r_k (r_0 = 0).
        # Default: outermost annulus (handles r_mid just outside radii[-1]
        # from floating-point noise at the cylinder boundary).
        k = N - 1
        for kk in range(N):
            if r_mid < radii[kk]:
                k = kk
                break
        tau += sig_t[k] * (s_hi - s_lo)
    return tau


def build_volume_kernel(
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_beta: int,
    n_rho: int,
    dps: int = 30,
) -> np.ndarray:
    r"""Assemble the **volume** Nyström kernel matrix for a single group.

    The cylindrical Peierls equation in polar coordinates centred at
    the observer reads

    .. math::

       \Sigma_t(r_i)\,\varphi(r_i)
         \;=\; \frac{\Sigma_t(r_i)}{\pi}
           \int_{0}^{\pi}\!\mathrm{d}\beta\!
           \int_{0}^{\rho_{\max}(r_i,\beta)}\!
             \mathrm{Ki}_1\!\bigl(\tau(r_i,\rho,\beta)\bigr)\,
             q\!\bigl(r'(r_i,\rho,\beta)\bigr)\,\mathrm{d}\rho
         + S_{\rm bc}(r_i).

    With Lagrange interpolation
    :math:`q(r'_{ikm}) = \sum_j L_j(r'_{ikm})\,q_j`, this discretises to

    .. math::

       \Sigma_t(r_i)\,\varphi_i
         \;=\; \sum_j K_{ij}\,q_j + S_{\rm bc}(r_i),
       \qquad
       K_{ij} = \frac{\Sigma_t(r_i)}{\pi}\sum_{k,m} w_{\beta,k}\,
                  w_{\rho,m}(r_i,\beta_k)\,
                  \mathrm{Ki}_1(\tau_{ikm})\,L_j(r'_{ikm}).

    Row-sum identity (for the **1-group, homogeneous, pure scatterer**
    check): at :math:`\varphi\equiv1,\,q=\Sigma_t`, the infinite-medium
    identity :math:`\sum_j K_{ij} \to \Sigma_t` holds exactly, and the
    finite-cylinder deficit equals :math:`\Sigma_t P_{\rm esc}(r_i)`
    where :math:`P_{\rm esc}` is the uncollided-escape probability
    from :math:`r_i`. The test gate uses :math:`R = 10` mean free
    paths so that :math:`P_{\rm esc}(r_i \lesssim R/2) \ll 10^{-3}`.

    Parameters
    ----------
    r_nodes : np.ndarray, shape (N,)
        Radial quadrature nodes (composite GL on :math:`[0, R]`).
    panel_bounds : list of (pa, pb, i_start, i_end)
        Panel layout of the r-grid, from :func:`composite_gl_r`. Used
        for Lagrange-basis interpolation on the local panel.
    radii : np.ndarray, shape (N_reg,)
        Outer radii of concentric annuli.
    sig_t : np.ndarray, shape (N_reg,)
        Total macroscopic cross-section per annulus.
    n_beta : int
        GL order for the :math:`\beta`-integral on :math:`[0, \pi]`.
    n_rho : int
        GL order for the :math:`\rho`-integral on
        :math:`[0, \rho_{\max}(r_i, \beta_k)]`.
    dps : int
        mpmath working precision for :math:`\mathrm{Ki}_1`.

    Returns
    -------
    K : np.ndarray, shape (N, N)
        Nyström kernel matrix.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    N = len(r_nodes)
    R = float(radii[-1])

    # β-quadrature on [0, π]
    beta_pts, beta_wts = _gl_float(n_beta, 0.0, np.pi, dps)
    cos_betas = np.cos(beta_pts)
    sin_betas = np.sin(beta_pts)

    # Reference ρ-quadrature nodes on [-1, 1] — we map per (i, k)
    ref_rho_nodes, ref_rho_wts = _gl_nodes_weights(n_rho, dps)
    ref_rho_nodes = np.array([float(x) for x in ref_rho_nodes])
    ref_rho_wts = np.array([float(w) for w in ref_rho_wts])

    K = np.zeros((N, N))
    inv_pi = 1.0 / np.pi

    for i in range(N):
        r_i = r_nodes[i]
        # Σ_t(r_i): locate the annulus containing r_i (default to
        # outermost for any r_i that lands exactly on the outer radius
        # due to floating-point noise).
        k_obs = len(radii) - 1
        for kk in range(len(radii)):
            if r_i < radii[kk]:
                k_obs = kk
                break
        sig_t_i = sig_t[k_obs]

        for k in range(n_beta):
            cb = cos_betas[k]
            sb = sin_betas[k]
            rho_max = _rho_max(r_i, cb, R)
            if rho_max <= 0.0:
                continue

            # Map reference ρ-nodes [-1, 1] → [0, ρ_max]
            h = 0.5 * rho_max
            rho_pts = h * ref_rho_nodes + h
            rho_wts = h * ref_rho_wts

            for m in range(n_rho):
                rho = rho_pts[m]
                r_prime = np.sqrt(r_i * r_i + 2.0 * r_i * rho * cb + rho * rho)
                tau = _optical_depth_along_ray(
                    r_i, cb, sb, rho, radii, sig_t,
                )
                ki1 = float(ki_n_mp(1, float(tau), dps))
                L_vals = _lagrange_basis_on_panels(
                    r_nodes, panel_bounds, float(r_prime),
                )
                weight = inv_pi * sig_t_i * beta_wts[k] * rho_wts[m] * ki1
                K[i, :] += weight * L_vals

    return K


def composite_gl_r(
    radii: np.ndarray,
    n_panels_per_region: int,
    p_order: int,
    dps: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int, int]]]:
    """Composite GL quadrature on :math:`[0, R]` with panel breakpoints
    at each annular radius. Same structure as :func:`composite_gl_y`;
    exposed separately because the radial grid and the y-grid may
    have different resolutions in practice.
    """
    return composite_gl_y(radii, n_panels_per_region, p_order, dps=dps)


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
