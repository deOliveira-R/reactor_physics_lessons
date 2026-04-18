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

import mpmath
import numpy as np

from ._kernels import (  # noqa: F401
    _shifted_legendre_eval,
    chord_half_lengths,
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


def composite_gl_r(
    radii: np.ndarray,
    n_panels_per_region: int,
    p_order: int,
    dps: int = 30,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int, int]]]:
    r"""Composite GL on :math:`[0, R]` with panel breakpoints at annular radii.

    Shared by cylindrical and spherical Peierls solvers; the panel
    structure accommodates the :math:`\Sigma_t(r)` discontinuities at
    each :math:`r_k`.

    Returns ``(r_pts, r_wts, panel_bounds)`` where ``panel_bounds`` is
    a list of ``(pa, pb, i_start, i_end)`` tuples describing the
    composite rule's panels.
    """
    radii = np.asarray(radii, dtype=float)
    gl_ref, gl_wt = gl_nodes_weights(p_order, dps)

    breakpoints = [mpmath.mpf(0)] + [mpmath.mpf(float(r)) for r in radii]
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

    Two concrete specialisations:

    - ``kind = "cylinder-1d"``: :math:`d = 2`, :math:`S_d = 2\pi`,
      :math:`\kappa_d = \mathrm{Ki}_1`, angular measure
      :math:`\mathrm d\beta`.
    - ``kind = "sphere-1d"``: :math:`d = 3`, :math:`S_d = 4\pi`,
      :math:`\kappa_d = e^{-\tau}`, angular measure
      :math:`\sin\theta\,\mathrm d\theta` (azimuthal folded).

    The geometric primitives (:math:`\rho_{\max}`,
    :math:`r'(\rho, \Omega, r)`) are IDENTICAL for cylinder and
    sphere because both describe a 1-D radial domain bounded by a
    spherical shell of radius :math:`R` (the cylindrical lateral
    surface in 2-D polar looks geometrically identical to the
    equatorial great-circle of the spherical shell). Only the
    angular measure and the kernel function differ.

    The **polar-form prefactor** ``prefactor`` bundles together:

    1. :math:`1/S_d` from the 3-D point-kernel normalisation.
    2. The azimuthal-symmetry fold (:math:`2\pi` for both cylinder
       and sphere, since :math:`q(r')` is radially symmetric).
    3. The :math:`\pm\beta` / :math:`\pm\theta` reflection fold (:math:`2`).

    For cylinder: :math:`2 \cdot 2\pi / (2\pi) = 2` net numerator
    divided by :math:`2\pi` gives :math:`1/\pi` — plus the
    integration is on :math:`[0, \pi]` after folding.
    For sphere: :math:`2\pi / (4\pi) = 1/2` — integration is on
    :math:`[0, \pi]` with :math:`\sin\theta` weight (no further
    folding since :math:`\sin\theta` is already symmetric).
    """

    kind: str

    def __post_init__(self) -> None:
        if self.kind not in ("cylinder-1d", "sphere-1d"):
            raise ValueError(f"Unsupported geometry kind {self.kind!r}")

    # ── geometric constants ───────────────────────────────────────────

    @property
    def d(self) -> int:
        """Effective dimension of the dimensionally-reduced kernel (2 or 3)."""
        return {"cylinder-1d": 2, "sphere-1d": 3}[self.kind]

    @property
    def S_d(self) -> float:
        r"""Total solid angle :math:`S_d` of the unit :math:`(d-1)`-sphere."""
        return {"cylinder-1d": 2.0 * np.pi, "sphere-1d": 4.0 * np.pi}[self.kind]

    @property
    def prefactor(self) -> float:
        """Composite prefactor absorbing :math:`1/S_d` + azimuth fold + :math:`\\pm` fold.

        See the class docstring for the derivation.
        """
        return {"cylinder-1d": 1.0 / np.pi, "sphere-1d": 0.5}[self.kind]

    @property
    def angular_range(self) -> tuple[float, float]:
        """Integration range of the single remaining angular variable."""
        return (0.0, np.pi)

    # ── angular measure ───────────────────────────────────────────────

    def angular_weight(self, omega_pts: np.ndarray) -> np.ndarray:
        r"""Weight factor in the angular measure :math:`\mathrm d\Omega_d`.

        Cylinder: :math:`\mathrm d\beta` ⇒ weight = 1.
        Sphere:  :math:`\sin\theta\,\mathrm d\theta` ⇒ weight = :math:`\sin\theta`.
        """
        if self.kind == "cylinder-1d":
            return np.ones_like(omega_pts)
        return np.sin(omega_pts)

    # ── ray geometry ──────────────────────────────────────────────────

    def rho_max(self, r_obs: float, cos_omega: float, R: float) -> float:
        r"""Ray-exit distance: positive root of
        :math:`(r_{\rm obs} + \rho\cos\Omega)^2 + (\rho\sin\Omega)^2 = R^2`.
        """
        disc = r_obs * r_obs * cos_omega * cos_omega + R * R - r_obs * r_obs
        return -r_obs * cos_omega + np.sqrt(max(disc, 0.0))

    def source_position(
        self, r_obs: float, rho: float, cos_omega: float,
    ) -> float:
        r"""Source radius :math:`r' = \sqrt{r_{\rm obs}^2 + 2 r_{\rm obs}\rho\cos\Omega + \rho^2}`.

        Identical closed form for cylinder and sphere — the 1-D
        radial symmetry hides the 3-D direction-of-ray-in-azimuth
        dependence, so only :math:`\cos\Omega` matters.
        """
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
        r"""Integrate :math:`\Sigma_t(r(s))` along the ray from
        :math:`r_{\rm obs}` in direction :math:`\Omega` for distance
        :math:`\rho`. Homogeneous short-circuit + annular boundary
        walking (shared between cylinder and sphere because the
        geometry is identical)."""
        radii = np.asarray(radii, dtype=float)
        sig_t = np.asarray(sig_t, dtype=float)
        N = len(radii)

        if N == 1:
            return float(sig_t[0]) * rho

        # Find signed s at each annular crossing by solving
        #   r_obs^2 + 2 r_obs s cos Ω + s^2 = r_k^2
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
        """Index of the annulus containing ``r`` (outer-biased at boundaries)."""
        k = len(radii) - 1
        for kk, r_k in enumerate(radii):
            if r < r_k:
                return kk
        return k

    # ── volume kernel :math:`\kappa_d(\tau)` ──────────────────────────

    def volume_kernel_mp(self, tau: float, dps: int = 25) -> float:
        r"""Volume Peierls kernel :math:`\kappa_d(\tau)` at mpmath precision.

        Cylinder: :math:`\mathrm{Ki}_1(\tau)` (A&S 11.2).
        Sphere:  :math:`e^{-\tau}`.
        """
        if self.kind == "cylinder-1d":
            return float(ki_n_mp(1, float(tau), dps))
        return float(mpmath.exp(-mpmath.mpf(tau)))

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

        Cylinder (:math:`d = 2`): returns :math:`r` (for
        :math:`r\,\mathrm d r\,\mathrm d\beta`).
        Sphere (:math:`d = 3`): returns :math:`r^2`.
        """
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
        surface area reduces — after the shared :math:`2\pi` (cylinder)
        or :math:`4\pi` (sphere) azimuthal factor cancels between
        :math:`A_d` and :math:`A_j` — to a divisor of :math:`R` for the
        cylinder and :math:`R^{2}` for the sphere.

        - Cylinder (:math:`A_d = 2\pi R`, :math:`A_j = 2\pi r_j w_j`):
          ratio :math:`A_j / A_d = r_j w_j / R`, divisor :math:`R`.
        - Sphere (:math:`A_d = 4\pi R^2`, :math:`A_j = 4\pi r_j^2 w_j`):
          ratio :math:`A_j / A_d = r_j^2 w_j / R^{2}`, divisor :math:`R^{2}`.
        """
        if self.kind == "cylinder-1d":
            return R
        return R * R


# Convenience singletons
CYLINDER_1D = CurvilinearGeometry(kind="cylinder-1d")
SPHERE_1D = CurvilinearGeometry(kind="sphere-1d")


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
    N = len(r_nodes)
    R = float(radii[-1])

    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = np.cos(omega_pts)
    angular_factor = geometry.angular_weight(omega_pts)

    ref_rho_nodes, ref_rho_wts = gl_nodes_weights(n_rho, dps)
    ref_rho_nodes = np.array([float(x) for x in ref_rho_nodes])
    ref_rho_wts = np.array([float(w) for w in ref_rho_wts])

    K = np.zeros((N, N))
    pref = geometry.prefactor

    for i in range(N):
        r_i = r_nodes[i]
        ki = geometry.which_annulus(r_i, radii)
        sig_t_i = sig_t[ki]

        for k in range(n_angular):
            cos_om = cos_omegas[k]
            rho_max_val = geometry.rho_max(r_i, cos_om, R)
            if rho_max_val <= 0.0:
                continue

            h = 0.5 * rho_max_val
            rho_pts = h * ref_rho_nodes + h
            rho_wts = h * ref_rho_wts

            for m in range(n_rho):
                rho = rho_pts[m]
                r_prime = geometry.source_position(r_i, rho, cos_om)
                tau = geometry.optical_depth_along_ray(
                    r_i, cos_om, rho, radii, sig_t,
                )
                kappa = geometry.volume_kernel_mp(tau, dps)
                L_vals = lagrange_basis_on_panels(
                    r_nodes, panel_bounds, float(r_prime),
                )
                weight = (
                    pref * sig_t_i
                    * omega_wts[k] * angular_factor[k]
                    * rho_wts[m] * kappa
                )
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

    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = np.cos(omega_pts)
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
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    R = float(radii[-1])
    inv_R2 = 1.0 / (R * R)

    omega_low, omega_high = geometry.angular_range
    omega_pts, omega_wts = gl_float(n_angular, omega_low, omega_high, dps)
    cos_omegas = np.cos(omega_pts)
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
        if reflection not in reflection_map:
            raise ValueError(
                f"Unknown reflection = {reflection!r}; expected one of "
                f"{list(reflection_map)} or a {N}×{N} matrix."
            )
        R_matrix = reflection_map[reflection](N)
    else:
        R_matrix = np.asarray(reflection, dtype=float)

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
    """
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    sig_s = np.asarray(sig_s, dtype=float)
    nu_sig_f = np.asarray(nu_sig_f, dtype=float)

    r_nodes, r_wts, panels = composite_gl_r(
        radii, n_panels_per_region, p_order, dps=dps,
    )
    K = build_volume_kernel(
        geometry, r_nodes, panels, radii, sig_t,
        n_angular=n_angular, n_rho=n_rho, dps=dps,
    )

    if boundary == "white":
        K_bc = build_white_bc_correction_rank_n(
            geometry, r_nodes, r_wts, radii, sig_t,
            n_angular=n_angular, n_surf_quad=n_surf_quad, dps=dps,
            n_bc_modes=n_bc_modes,
        )
        K = K + K_bc
    elif boundary != "vacuum":
        raise ValueError(
            f"boundary must be 'vacuum' or 'white', got {boundary!r}"
        )

    N = len(r_nodes)
    sig_t_n = np.empty(N)
    sig_s_n = np.empty(N)
    nu_sig_f_n = np.empty(N)
    for i, ri in enumerate(r_nodes):
        ki = geometry.which_annulus(ri, radii)
        sig_t_n[i] = sig_t[ki]
        sig_s_n[i] = sig_s[ki]
        nu_sig_f_n[i] = nu_sig_f[ki]

    A = np.diag(sig_t_n) - K * sig_s_n[np.newaxis, :]
    B = K * nu_sig_f_n[np.newaxis, :]

    phi = np.ones(N)
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

    return PeierlsSolution(
        r_nodes=r_nodes,
        phi_values=phi[:, np.newaxis],
        k_eff=float(k_val),
        cell_radius=float(radii[-1]),
        n_groups=1,
        geometry_kind=geometry.kind,
        n_quad_r=N,
        n_quad_angular=n_angular * n_rho,
        precision_digits=dps,
        panel_bounds=panels,
    )
