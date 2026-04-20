r"""Archived: cylinder K matrix via the explicit out-of-plane
:math:`\varphi`-quadrature form (the "cylinder-polar" detour).

Extracted verbatim from `orpheus/derivations/peierls_geometry.py`
during the 2026-04-19 cleanup. Originally added in commit ``213278f``
of the ``investigate/peierls-solver-bugs`` branch as part of the
"retire Bickley" sub-thread of issue #116. **Mathematically equivalent**
to the cylinder-1d K assembly (verified element-wise to machine
precision against ``cylinder-1d`` at n_phi=32) — both compute the same
:math:`\mathrm{Ki}_1`-weighted integral; ``cylinder-1d`` evaluates
:math:`\mathrm{Ki}_1` directly via ``ki_n_mp`` / ``ki_n_float``,
whereas this assembly substitutes the defining integral

.. math::

   \mathrm{Ki}_1(\tau) \;=\; \int_0^{\pi/2}\!e^{-\tau/\cos\varphi}\,
       \mathrm d\varphi

and integrates :math:`\varphi` numerically with a 16-node GL on
:math:`[0, \pi/2]`. Useful as an exposition of the cylinder kernel's
dimensional reduction; not a separate physics construct.

**Why archived:** ``cylinder-1d`` is the natural-kernel form (uses
:math:`\mathrm{Ki}_1` directly); cylinder-polar adds an extra layer
for no precision win. After the strategic decision (2026-04-19 user
directive) to embrace the natural kernels as load-bearing
infrastructure, the explicit-:math:`\varphi` route became redundant.

**When to bring back:** If a future analysis needs to expose the
out-of-plane angular distribution explicitly — e.g., for higher-order
angular flux moments at the cell surface, or for a chord-by-chord
out-of-plane reconstruction — this is the assembly to start from.

This module is **not** part of the active import path. To use it:

.. code-block:: python

   import sys
   sys.path.insert(0, "derivations/archive")
   from peierls_cylinder_polar_assembly import _build_volume_kernel_cylinder_phi

Live dependencies (still in ``orpheus.derivations``):

- ``CurvilinearGeometry`` and the ``cylinder-polar`` ``kind``
  (which has been removed from the active package; you would need
  to re-add the ``"cylinder-polar"`` branches to ``__post_init__``,
  ``d``, ``S_d``, ``prefactor``, the ``volume_kernel_mp`` cylinder
  branch, etc., or instantiate with ``kind="cylinder-1d"`` and
  remember that the integrand is defined for the same operator).
- ``gl_nodes_weights``, ``lagrange_basis_on_panels`` from
  ``peierls_geometry``.
"""

from __future__ import annotations

import numpy as np

# Live primitives still in orpheus.derivations
from orpheus.derivations.peierls_geometry import (  # noqa: E402
    gl_nodes_weights,
    lagrange_basis_on_panels,
)


def _build_volume_kernel_cylinder_phi(
    geometry,
    r_nodes: np.ndarray,
    panel_bounds: list[tuple[float, float, int, int]],
    radii: np.ndarray,
    sig_t: np.ndarray,
    n_angular: int,
    n_rho: int,
    n_phi: int = 8,
    dps: int = 30,
) -> np.ndarray:
    r"""Cylinder K-matrix with the out-of-plane :math:`\varphi`
    integration exposed explicitly.

    Mathematically equivalent to the cylinder-1d K assembly via

    .. math::

       \mathrm{Ki}_1(\tau) \;=\; \int_0^{\pi/2}\!
           e^{-\tau/\cos\varphi}\,\mathrm d\varphi.

    A ``n_phi``-point GL on :math:`[0, \pi/2]` replaces one Bickley
    evaluation with ``n_phi`` exponential evaluations.
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    radii = np.asarray(radii, dtype=float)
    sig_t = np.asarray(sig_t, dtype=float)
    panel_bounds = [
        (float(pa), float(pb), int(i_start), int(i_end))
        for (pa, pb, i_start, i_end) in panel_bounds
    ]
    N = len(r_nodes)
    R = float(radii[-1])

    omega_low, omega_high = geometry.angular_range
    ref_omega_nodes, ref_omega_wts = gl_nodes_weights(n_angular, dps)
    ref_omega_nodes = np.array([float(x) for x in ref_omega_nodes])
    ref_omega_wts = np.array([float(w) for w in ref_omega_wts])

    ref_rho_nodes, ref_rho_wts = gl_nodes_weights(n_rho, dps)
    ref_rho_nodes = np.array([float(x) for x in ref_rho_nodes])
    ref_rho_wts = np.array([float(w) for w in ref_rho_wts])

    ref_phi_nodes, ref_phi_wts = gl_nodes_weights(n_phi, dps)
    phi_h = 0.5 * (np.pi / 2)
    phi_mid = 0.5 * (np.pi / 2)
    phi_pts = np.array([phi_h * float(x) + phi_mid for x in ref_phi_nodes])
    phi_wts = np.array([phi_h * float(w) for w in ref_phi_wts])
    cos_phi_pts = np.cos(phi_pts)
    inv_cos_phi_pts = 1.0 / cos_phi_pts

    panel_boundaries_r = np.array(sorted(
        {pa for (pa, pb, _, _) in panel_bounds}
        | {pb for (pa, pb, _, _) in panel_bounds}
    ), dtype=float)
    interior_boundaries_r = panel_boundaries_r[
        (panel_boundaries_r > 0.0) & (panel_boundaries_r < R)
    ]

    K = np.zeros((N, N))
    pref = geometry.prefactor

    for i in range(N):
        r_i = float(r_nodes[i])
        ki = geometry.which_annulus(r_i, radii)
        sig_t_i = float(sig_t[ki])

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

                crossings = geometry.rho_crossings_for_ray(
                    r_i, cos_om, rho_max_val, interior_boundaries_r,
                )
                rho_subintervals = [0.0, *crossings, rho_max_val]

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
                        tau = geometry.optical_depth_along_ray(
                            r_i, cos_om, rho, radii, sig_t,
                        )
                        L_vals = lagrange_basis_on_panels(
                            r_nodes, panel_bounds, float(r_prime),
                        )
                        kappa_phi_sum = float(
                            np.sum(phi_wts * np.exp(-tau * inv_cos_phi_pts))
                        )
                        weight = outer_weight * rho_wts[m] * kappa_phi_sum
                        K[i, :] += weight * L_vals

    return K
