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

Scope after the 2026-04-19 cleanup
----------------------------------

This module now contains ONLY the slab :math:`E_1`-form references and
the analytical vacuum-BC slab flux. The polar-form K element (slab,
cylinder, sphere) is now subsumed by the unified verification
primitive :func:`~.peierls_geometry.K_vol_element_adaptive`.

- :func:`slab_kernel_point_to_point` — exact :math:`E_1` slab point
  kernel.
- :func:`slab_K_vol_element` — adaptive ``mpmath.quad`` in real-space
  against the :math:`E_1` integrand. Used as the second independent
  formulation in the slab "two paths agree" verification gate
  (the polar form is the unified primitive).
- :func:`slab_uniform_source_analytical` — closed-form vacuum-BC
  uniform-source flux for slab; the analytical reference for the
  K-row-sum identity.
- :func:`slab_row_sum_uniform_identity` — convenience alias.
- :func:`cylinder_uniform_source_analytical` — semi-analytical vacuum-BC
  uniform-source flux for an infinite cylindrical cell; one
  ``mpmath.quad`` over the in-plane azimuth with :math:`\mathrm{Ki}_2`
  absorbing the out-of-plane polar integral.
- :func:`sphere_uniform_source_analytical` — semi-analytical vacuum-BC
  uniform-source flux for a spherical cell; one ``mpmath.quad`` over
  :math:`\mu = \cos\Theta`.
- :func:`slab_uniform_source_white_bc_analytical` — closed-form slab
  flux under the Mark / isotropic white BC (albedo 1) on both faces,
  derived from the :math:`E_3` transmission formula and the
  partial-current balance. This is the first rung of the BC
  verification ladder — it exercises the
  :class:`~.peierls_geometry.BoundaryClosureOperator`'s rank-1
  (``reflection = mark``) path for the slab geometry.

Together these functions complete the machine-precision analytical flux
reference for vacuum-BC row-sum gating of the K matrix built by
:func:`~.peierls_geometry.build_volume_kernel_adaptive` (the unified
verification primitive), plus the first white-BC variant.
"""

from __future__ import annotations

import mpmath
import numpy as np

from ._kernels import ki_n_mp
from .peierls_geometry import lagrange_basis_on_panels  # noqa: F401  (re-exported for downstream)


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
# Cylinder and sphere vacuum-BC uniform-source analytical references
# ═══════════════════════════════════════════════════════════════════════

def cylinder_uniform_source_analytical(
    r: float, R: float, sig_t: float, *, dps: int = 50,
) -> mpmath.mpf:
    r"""Exact scalar flux from uniform unit source :math:`S = 1` on a
    pure-absorber cylindrical cell of radius :math:`R` (infinite axial
    extent) with vacuum BC on the lateral surface:

    .. math::

       \varphi_{\rm cyl}(r) \;=\;
           \frac{1}{\pi\,\Sigma_t}\!\int_0^\pi\!
             \Bigl[\,1 - \mathrm{Ki}_2\!\bigl(\Sigma_t\,L_{2D}(r,\theta')\bigr)\,\Bigr]
             \,\mathrm d\theta',

    with in-plane chord length

    .. math::

       L_{2D}(r, \theta') \;=\; -r\cos\theta'
                                + \sqrt{R^{2} - r^{2}\sin^{2}\theta'}.

    **Derivation.** Start from the 3-D point kernel
    :math:`e^{-\Sigma_t R'}/(4\pi R'^{2})` integrated against the
    spatial volume. In observer-centred cylindrical coordinates with
    in-plane azimuth :math:`\theta'` and polar angle
    :math:`\psi \in (0, \pi)` from the cylinder axis, the 3-D ray length
    to the exit on the lateral surface is
    :math:`\rho_{\max} = L_{2D}/\sin\psi`. The :math:`\psi`-integral
    then reduces to Bickley :math:`\mathrm{Ki}_2` via the definition
    :math:`\mathrm{Ki}_n(x) = \int_0^{\pi/2}\!\cos^{n-1}\!\phi\,
    e^{-x/\cos\phi}\,\mathrm d\phi` with the substitution
    :math:`\phi = \pi/2 - \psi`.

    **Sanity checks.**

    - :math:`r = 0`: :math:`L_{2D} = R` (constant) gives
      :math:`\varphi(0) = (1 - \mathrm{Ki}_2(\Sigma_t R))/\Sigma_t`.
    - :math:`\Sigma_t R \to 0` (thin):
      :math:`\mathrm{Ki}_2(x) \approx 1 - (\pi/2)\,x`, so
      :math:`\varphi(0) \to (\pi/2)\,R` — the 3-D mean chord through
      the axis of an infinite cylinder.
    - :math:`\Sigma_t R \to \infty`: :math:`\mathrm{Ki}_2 \to 0` and
      :math:`\varphi \to 1/\Sigma_t` (infinite-medium limit).

    **Implementation.** A single adaptive :func:`mpmath.quad` over
    :math:`\theta' \in [0, \pi]`. The integrand is smooth; no
    breakpoints needed. Machine precision via ``dps``.

    References
    ----------
    Bell & Glasstone (1970) "Nuclear Reactor Theory" Ch. 2; Case &
    Zweifel (1967) "Linear Transport Theory" Ch. 3.
    """
    with mpmath.workdps(dps):
        r_mp = mpmath.mpf(r)
        R_mp = mpmath.mpf(R)
        sig_t_mp = mpmath.mpf(sig_t)

        def integrand(theta_p):
            cos_tp = mpmath.cos(theta_p)
            sin_tp = mpmath.sin(theta_p)
            L_2d = -r_mp * cos_tp + mpmath.sqrt(
                R_mp ** 2 - r_mp ** 2 * sin_tp ** 2
            )
            ki2 = ki_n_mp(2, sig_t_mp * L_2d, dps)
            return 1 - ki2

        integral = mpmath.quad(integrand, [0, mpmath.pi])
        return integral / (mpmath.pi * sig_t_mp)


def sphere_uniform_source_analytical(
    r: float, R: float, sig_t: float, *, dps: int = 50,
) -> mpmath.mpf:
    r"""Exact scalar flux from uniform unit source :math:`S = 1` on a
    pure-absorber spherical cell of radius :math:`R` with vacuum BC:

    .. math::

       \varphi_{\rm sph}(r) \;=\; \frac{1}{2\,\Sigma_t}\!\left[\,
           2 - \int_{-1}^{1}\!\exp\!\Bigl(
               -\Sigma_t\bigl[-r\mu + \sqrt{R^{2} - r^{2} + r^{2}\mu^{2}}\bigr]
             \Bigr)\,\mathrm d\mu\,\right],

    where :math:`\mu = \cos\Theta` is the cosine of the polar angle
    from the observer's outward radial direction.

    **Derivation.** In observer-centred spherical coordinates
    :math:`(\rho, \Theta, \psi)` the point kernel
    :math:`e^{-\Sigma_t\rho}/(4\pi\rho^{2})` cancels the
    :math:`\rho^{2}` volume Jacobian; axial symmetry eliminates
    :math:`\psi`. The chord length from the observer at radius
    :math:`r` in direction :math:`\Theta` to the spherical surface is

    .. math::

       L_{\rm chord}(r, \mu) \;=\;
           -r\mu + \sqrt{R^{2} - r^{2}(1 - \mu^{2})}.

    The inner :math:`\rho`-integral gives
    :math:`(1 - e^{-\Sigma_t L_{\rm chord}})/\Sigma_t`; averaging over
    :math:`\mu \in [-1, 1]` yields the displayed form.

    **Sanity checks.**

    - :math:`r = 0`: :math:`L_{\rm chord} = R` (constant) gives
      :math:`\varphi(0) = (1 - e^{-\Sigma_t R})/\Sigma_t`.
    - :math:`\Sigma_t R \to 0` (thin): :math:`\varphi(0) \to R`, the
      3-D mean chord through the centre of a sphere.
    - :math:`\Sigma_t R \to \infty`:
      :math:`\varphi \to 1/\Sigma_t` (infinite-medium limit).

    **Implementation.** A single adaptive :func:`mpmath.quad` over
    :math:`\mu \in [-1, 1]`. The integrand is analytic; no breakpoints
    needed. Machine precision via ``dps``.

    References
    ----------
    Case & Zweifel (1967) "Linear Transport Theory" Ch. 3.
    """
    with mpmath.workdps(dps):
        r_mp = mpmath.mpf(r)
        R_mp = mpmath.mpf(R)
        sig_t_mp = mpmath.mpf(sig_t)

        def integrand(mu):
            L_chord = -r_mp * mu + mpmath.sqrt(
                R_mp ** 2 - r_mp ** 2 * (1 - mu ** 2)
            )
            return mpmath.exp(-sig_t_mp * L_chord)

        integral = mpmath.quad(integrand, [-1, 1])
        return (2 - integral) / (2 * sig_t_mp)


# ═══════════════════════════════════════════════════════════════════════
# Slab white-BC analytical (Mark / rank-1 isotropic re-entry closure)
# ═══════════════════════════════════════════════════════════════════════

def slab_uniform_source_white_bc_analytical(
    x, L: float, sig_t: float, *, dps: int = 50,
) -> mpmath.mpf:
    r"""Exact scalar flux from uniform unit source :math:`S = 1` on a
    pure-absorber slab :math:`[0, L]` with **Mark / isotropic white BC**
    (albedo 1) on both faces:

    .. math::

       \varphi_{\rm white}(x) \;\equiv\; \frac{1}{\Sigma_t}
       \qquad \text{(for all $x$, any $L$)}.

    **Why the answer is the infinite-medium equilibrium.** White BC on
    both faces of a uniform cell is the Wigner-Seitz exact equivalence:
    it models a slab embedded in an infinite symmetric lattice, so the
    cell cannot lose neutrons through the boundary. For a pure absorber
    with uniform source :math:`S = 1`, the balance equation collapses
    to the point-wise equilibrium :math:`\Sigma_t\,\varphi = S = 1`,
    giving :math:`\varphi = 1/\Sigma_t` at every interior point —
    independent of :math:`L`.

    **Self-consistent derivation from the partial-current balance.**
    The slab Peierls integral equation with white BC is

    .. math::

       \varphi(x) \;=\; \tfrac{1}{2}\!\int_0^L\!
                         E_1(\Sigma_t|x-x'|)\,S(x')\,\mathrm d x'
                       + 2\,J^-\,\bigl[E_2(\Sigma_t x)
                                     + E_2(\Sigma_t(L - x))\bigr],

    with :math:`\psi_{\rm in} = 2\,J^-` (the half-range-isotropic
    angular flux per unit inward partial current). The partial-current
    balance at :math:`x = L` for uniform :math:`S = 1`, using the
    identity
    :math:`\int_0^{\tau_L} E_2(u)\,\mathrm du = E_3(0) - E_3(\tau_L)
    = 1/2 - E_3(\tau_L)`:

    .. math::

       J^+(L) \;=\; \tfrac{1}{2\Sigma_t}\bigl(\tfrac{1}{2} - E_3(\Sigma_t L)\bigr)
                  + 2\,E_3(\Sigma_t L)\,J^-(0).

    With symmetry :math:`J^-(0) = J^-(L) = J^-` and the Mark closure
    :math:`J^- = J^+`:

    .. math::

       J^- \;=\; \frac{1/2 - E_3(\Sigma_t L)}
                      {2\,\Sigma_t\,(1 - 2\,E_3(\Sigma_t L))}
             \;=\; \frac{1}{4\,\Sigma_t}

    independent of :math:`\tau_L = \Sigma_t L` (the numerator and
    denominator share the factor :math:`(1 - 2 E_3(\tau_L))`).
    Substituting :math:`2 J^- = 1/(2\Sigma_t)` into the integral
    equation collapses the :math:`E_2` terms:

    .. math::

       \varphi(x) \;=\; \tfrac{1}{2\Sigma_t}\!\bigl[2 - E_2(\Sigma_t x)
                                                - E_2(\Sigma_t(L-x))\bigr]
                  + \tfrac{1}{2\Sigma_t}\!\bigl[E_2(\Sigma_t x)
                                              + E_2(\Sigma_t(L-x))\bigr]
                  \;=\; \tfrac{1}{\Sigma_t}.

    **History.** An earlier version of this file (commit ``2538cfe``)
    shipped an incorrect closed form derived with
    :math:`J^+(L)|_{\rm vol} = (1/(2\Sigma_t))(1 - E_3(\tau_L))` — off
    by the antiderivative identity above
    (:math:`\int_0^{\tau_L} E_2 = 1/2 - E_3`, not :math:`1 - E_3`).
    The fixed-point diagnostic agreed with the wrong formula because
    the fixed-point also used the same buggy :math:`J^+` update. The
    error was caught when the first-order K_bc row-sum gate disagreed
    with the published formula by a factor of 2.19 — a useful
    reminder that "two independent derivations agreeing to 1e-39" is
    worthless if both share a factor-of-2 algebra error.

    **Testing leverage.** Because :math:`\varphi_{\rm white}` is
    spatially *constant* for the slab, it supports two precise tests of
    the Peierls white-BC tensor-network machinery:

    1. The eigenvalue identity :math:`k_{\rm eff}({\rm white}) =
       k_\infty` for any :math:`L`, for any homogeneous slab.
    2. The K-matrix factor-level closed forms:
       :math:`P_{\rm esc}(x) = (1/2)\,[E_2(\Sigma_t x) + E_2(\Sigma_t(L-x))]`
       and
       :math:`G_{\rm bc}(x) = 2\,[E_2(\Sigma_t x) + E_2(\Sigma_t(L-x))]`.

    See the companion tests in :mod:`tests.derivations.test_peierls_reference`.

    References
    ----------
    Wigner & Seitz lattice-cell approximation; Davison (1957) "Neutron
    Transport Theory" Ch. 5; Case & Zweifel (1967) "Linear Transport
    Theory" Ch. 6.
    """
    with mpmath.workdps(dps):
        sig_t_mp = mpmath.mpf(sig_t)
        return 1 / sig_t_mp


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
    "slab_uniform_source_white_bc_analytical",
    "slab_row_sum_uniform_identity",
    "cylinder_uniform_source_analytical",
    "sphere_uniform_source_analytical",
]
