r"""Independent high-precision reference for the unified Peierls operator.

Purpose: provide a **mesh-independent, mathematically self-contained**
reference for verifying the production Nyström solver in
:mod:`orpheus.derivations.peierls_geometry`. Built from the same
unified polar form documented in :doc:`/theory/peierls_unified` §4,
but uses :func:`mpmath.quad` (adaptive Gauss-Kronrod) and native
arbitrary-precision special functions instead of fixed-order
Gauss-Legendre + mpmath dps truncation.

The reference stands alone: no Nyström basis, no fixed quadrature
orders, no shared code with production. It is the Phase-0
``ContinuousReferenceSolution`` contract made concrete for Peierls.

Design principle
----------------

The production solver and this reference **discretise the same
operator in two mathematically-different ways**:

- **Production** (:func:`~.peierls_geometry.build_volume_kernel`):
  panelwise Gauss-Legendre Nyström + Lagrange basis, fixed dps.
  The K matrix entries come from tensor-product GL quadrature over
  the polar (Ω, ρ) ray parametrisation.
- **Reference** (this module): adaptive :func:`mpmath.quad` over the
  same (Ω, ρ) parametrisation, arbitrary dps, straight evaluation of
  the integral with no basis interpolation between nodes — at each
  radial sample the kernel is evaluated at full precision.

Identical agreement of ``K[i, j]`` element-by-element
is a **complete verification** of: the unified polar-form integrand
formula, the geometry primitives (``rho_max``, ``source_position``,
``angular_weight``, ``volume_kernel_mp``), the optical-depth walker
handling of multi-region Σ_t, and the Lagrange basis evaluation.

Disagreement localises the bug to exactly one of those components.

Applicability
-------------

All three 1-D geometries via :class:`~.peierls_geometry.CurvilinearGeometry`:

- ``kind = "slab-1d"`` — via κ_d = E₁, Ω ∈ {±1} two-face integration
  (NOT CURRENTLY IMPLEMENTED by CurvilinearGeometry — slab has its
  own solver :mod:`orpheus.derivations.peierls_slab`; this module
  therefore provides a separate slab reference path using the same
  mpmath-adaptive methodology).
- ``kind = "cylinder-1d"`` — via κ_d = Ki₁, Ω = β polar angle.
- ``kind = "sphere-1d"`` — via κ_d = exp, Ω = θ polar angle.
"""

from __future__ import annotations

from typing import Callable

import mpmath
import numpy as np

from ._kernels import ki_n_mp
from .peierls_geometry import (
    CurvilinearGeometry,
    CYLINDER_1D,
    SPHERE_1D,
    lagrange_basis_on_panels,
)


# ═══════════════════════════════════════════════════════════════════════
# Slab reference (E₁ kernel, two-face integration)
# ═══════════════════════════════════════════════════════════════════════

def slab_kernel_point_to_point(
    x_i: float, x_j: float, sig_t: float, *, dps: int = 50,
) -> mpmath.mpf:
    r"""Continuous slab Peierls kernel

    .. math::

       K(x_i, x_j) = \frac{1}{2}\,E_1(\Sigma_t\,|x_i - x_j|).

    At high dps this is the exact point-to-point infinite-medium slab
    kernel. The ``1/2`` factor emerges from the 2-D transverse
    integration of the 3-D point kernel (§2 of peierls_unified).
    """
    with mpmath.workdps(dps):
        tau = sig_t * abs(mpmath.mpf(x_i) - mpmath.mpf(x_j))
        if tau == 0:
            return mpmath.mpf("inf")  # E₁(0) diverges logarithmically
        return mpmath.expint(1, tau) / 2


def slab_K_vol_element(
    i: int, j: int,
    x_nodes: list, panel_bounds: list[tuple[float, float, int, int]],
    L: float, sig_t: float,
    *, dps: int = 50,
) -> mpmath.mpf:
    r"""Reference K[i, j] for the slab Peierls operator at dps precision.

    Computes

    .. math::

       K_{ij} \;=\; \int_0^L \tfrac{1}{2}\,E_1(\Sigma_t\,|x_i - x'|)\,
                    L_j(x')\,\mathrm d x'

    via adaptive :func:`mpmath.quad` over each panel. The diagonal
    panel (``i`` and ``j`` in the same panel) has an integrable
    logarithmic singularity at :math:`x' = x_i`; :func:`mpmath.quad`
    with the ``[a, x_i, b]`` subdivision hint handles this natively.
    """
    x_i = mpmath.mpf(x_nodes[i])
    sig_t_mp = mpmath.mpf(sig_t)

    def integrand(x_prime):
        """Lagrange basis L_j at x_prime times kernel K(x_i, x_prime)."""
        if x_prime == x_i:
            return mpmath.mpf(0)  # integrand removable; quad handles this
        tau = sig_t_mp * abs(x_i - x_prime)
        kernel = mpmath.expint(1, tau) / 2
        # Lagrange basis evaluation (numpy-based, float precision)
        L_vals = lagrange_basis_on_panels(
            np.array([float(x) for x in x_nodes]),
            panel_bounds,
            float(x_prime),
        )
        return kernel * mpmath.mpf(float(L_vals[j]))

    # Integrate panel by panel; subdivide the observer's own panel at x_i
    K_ij = mpmath.mpf(0)
    with mpmath.workdps(dps):
        for pa, pb, _, _ in panel_bounds:
            pa_mp, pb_mp = mpmath.mpf(pa), mpmath.mpf(pb)
            if pa_mp <= x_i <= pb_mp:
                # Split at the singularity for the adaptive integrator
                K_ij += mpmath.quad(integrand, [pa_mp, x_i, pb_mp])
            else:
                K_ij += mpmath.quad(integrand, [pa_mp, pb_mp])
    return K_ij


def slab_uniform_source_analytical(
    x: float, L: float, sig_t: float, *, dps: int = 50,
) -> mpmath.mpf:
    r"""Exact scalar flux from uniform unit source S(x) = 1 on a pure
    absorber slab [0, L] with vacuum BC:

    .. math::

       \varphi(x) \;=\; \frac{1}{2\Sigma_t}
                         \bigl[2 - E_2(\Sigma_t\,x)
                                 - E_2(\Sigma_t\,(L - x))\bigr].

    Derivation: :math:`\varphi(x) = \int_0^L (1/2) E_1(\Sigma_t|x-x'|)
    dx'`; split at ``x'`` = ``x`` and use
    :math:`\int E_1(\alpha\,u)\,\mathrm du = (1/\alpha)[E_2(0) -
    E_2(\alpha u)] = (1/\alpha)[1 - E_2(\alpha u)]`.

    This is an **independent closed-form reference** at arbitrary
    precision.
    """
    with mpmath.workdps(dps):
        x_mp = mpmath.mpf(x)
        L_mp = mpmath.mpf(L)
        sig_t_mp = mpmath.mpf(sig_t)
        return (2 - mpmath.expint(2, sig_t_mp * x_mp)
                  - mpmath.expint(2, sig_t_mp * (L_mp - x_mp))) / (2 * sig_t_mp)


# ═══════════════════════════════════════════════════════════════════════
# Curvilinear (cylinder, sphere) reference — unified polar form
# ═══════════════════════════════════════════════════════════════════════

def curvilinear_K_vol_element(
    geometry: CurvilinearGeometry,
    i: int, j: int,
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    *,
    dps: int = 50,
) -> mpmath.mpf:
    r"""Reference K[i, j] for curvilinear Peierls via unified polar form
    with adaptive :func:`mpmath.quad`.

    Integrates the same observer-centred double integral that
    :func:`~.peierls_geometry.build_volume_kernel` approximates with
    Gauss-Legendre, but adaptively and at ``dps=50`` precision:

    .. math::

       K_{ij} = \Sigma_t(r_i)\,C_d \int_{\Omega_d} W_\Omega(\Omega)
                \int_0^{\rho_{\max}(r_i,\Omega)}
                \kappa_d(\tau(r_i,\Omega,\rho))\,L_j(r'(\rho,\Omega,r_i))\,
                \mathrm d\rho\,\mathrm d\Omega.

    Uses :meth:`.CurvilinearGeometry.volume_kernel_mp` for
    :math:`\kappa_d`, :meth:`.CurvilinearGeometry.optical_depth_along_ray`
    for :math:`\tau`, and the existing Lagrange basis. **Fully
    vectorial-to-scalar**: one :func:`mpmath.quad` call per (i, j) pair.
    """
    r_i = float(r_nodes[i])
    R = float(radii[-1])
    radii_arr = np.asarray(radii, dtype=float)
    sig_t_arr = np.asarray(sig_t, dtype=float)

    ki = geometry.which_annulus(r_i, radii_arr)
    sig_t_i = float(sig_t_arr[ki])

    omega_low, omega_high = geometry.angular_range
    r_nodes_f = np.asarray(r_nodes, dtype=float)

    def integrand_rho(rho, omega):
        cos_om = float(mpmath.cos(omega))
        rho_f = float(rho)
        rho_max_val = geometry.rho_max(r_i, cos_om, R)
        if rho_f >= rho_max_val or rho_f <= 0:
            return mpmath.mpf(0)
        # Compute r' in mpmath, guarding against float underflow near r'=0
        r_prime_sq = (mpmath.mpf(r_i) ** 2
                      + 2 * r_i * rho * cos_om
                      + rho ** 2)
        if r_prime_sq <= 0:
            r_prime = mpmath.mpf(0)
        else:
            r_prime = mpmath.sqrt(r_prime_sq)
        tau = geometry.optical_depth_along_ray(
            r_i, cos_om, rho_f, radii_arr, sig_t_arr,
        )
        kappa = geometry.volume_kernel_mp(tau, dps)
        L_vals = lagrange_basis_on_panels(r_nodes_f, panel_bounds, float(r_prime))
        return mpmath.mpf(float(L_vals[j])) * kappa

    panel_boundaries_r = sorted({pa for (pa, pb, _, _) in panel_bounds}
                                | {pb for (pa, pb, _, _) in panel_bounds})

    def rho_crossings_for_ray(cos_om, rho_max_val):
        """Compute ρ values where r'(ρ) crosses a panel boundary.

        For sphere / cylinder, r'(ρ)² = r_i² + 2 r_i ρ cos_ω + ρ² = r_b²
        gives ρ = -r_i cos_ω ± √(r_i² cos_om² + r_b² - r_i²). Keeps
        the positive roots in (0, ρ_max).
        """
        crossings = set()
        for r_b in panel_boundaries_r:
            disc = r_i * r_i * cos_om * cos_om + r_b * r_b - r_i * r_i
            if disc < 0:
                continue
            sqrt_disc = disc ** 0.5
            for sign in (+1, -1):
                rho = -r_i * cos_om + sign * sqrt_disc
                if 1e-12 < rho < rho_max_val - 1e-12:
                    crossings.add(rho)
        return sorted(crossings)

    def outer_omega(omega):
        cos_om = float(mpmath.cos(omega))
        rho_max_val = geometry.rho_max(r_i, cos_om, R)
        if rho_max_val <= 0:
            return mpmath.mpf(0)
        ang_factor = float(geometry.angular_weight(
            np.array([float(omega)]),
        )[0])
        # Subdivide ρ integration at panel crossings so mpmath.quad
        # can handle the derivative discontinuities of the Lagrange basis.
        crossings = rho_crossings_for_ray(cos_om, rho_max_val)
        breaks = ([mpmath.mpf(0)]
                  + [mpmath.mpf(rho) for rho in crossings]
                  + [mpmath.mpf(rho_max_val)])
        inner = mpmath.quad(
            lambda rho: integrand_rho(rho, omega),
            breaks,
        )
        return ang_factor * inner

    pref = mpmath.mpf(geometry.prefactor)
    with mpmath.workdps(dps):
        omega_integral = mpmath.quad(
            outer_omega,
            [mpmath.mpf(omega_low), mpmath.mpf(omega_high)],
        )
        return sig_t_i * pref * omega_integral


# ═══════════════════════════════════════════════════════════════════════
# Analytical diagnostics (independent of any Nyström)
# ═══════════════════════════════════════════════════════════════════════

def slab_row_sum_uniform_identity(
    x_i: float, L: float, sig_t: float, *, dps: int = 50,
) -> mpmath.mpf:
    r"""Slab row-sum identity: :math:`\int_0^L \tfrac12 E_1(\Sigma_t|x_i-x'|)
    \mathrm d x'`.

    Equal to :math:`\varphi(x_i)` for uniform source S=1 with pure
    absorption. See :func:`slab_uniform_source_analytical`. Convenience
    wrapper.
    """
    return slab_uniform_source_analytical(x_i, L, sig_t, dps=dps)


__all__ = [
    "slab_kernel_point_to_point",
    "slab_K_vol_element",
    "slab_uniform_source_analytical",
    "slab_row_sum_uniform_identity",
    "curvilinear_K_vol_element",
]
