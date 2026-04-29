"""L0 multi-region verification for the cylindrical Peierls Nyström operator.

C4 of the Phase-4.2 campaign. The volume kernel builder from C3 already
handles multi-region via
:meth:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry.optical_depth_along_ray`
(ray annular-boundary walking) and
:func:`~orpheus.derivations.peierls_geometry.lagrange_basis_on_panels`
(piecewise-Lagrange source interpolation across panels). This test
file adds the direct unit-tests on those two helpers and verifies
the ``build_volume_kernel`` output for a thick **inhomogeneous**
cylinder — where the local row-sum identity

.. math::

    \\sum_j K_{ij} \\;\\to\\; \\Sigma_t(r_i)

(LHS-:math:`\\Sigma_t` form) is non-trivial: unlike the homogeneous
case, the row sum depends on **which annulus contains** :math:`r_i`,
and the walker must correctly account for Σ_t variation along each
ray.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_cylinder import GEOMETRY
from orpheus.derivations.peierls_geometry import (
    build_volume_kernel,
    composite_gl_r,
    lagrange_basis_on_panels,
)


# ═══════════════════════════════════════════════════════════════════════
# CurvilinearGeometry.optical_depth_along_ray — multi-region annular walker
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("chord-length")
class TestOpticalDepthAlongRay:
    """Closed-form verification of the ray-integrated optical depth."""

    def test_homogeneous_short_circuit(self):
        """For a single annulus, :math:`\\tau = \\Sigma_t \\cdot \\rho`
        exactly."""
        tau = GEOMETRY.optical_depth_along_ray(
            r_obs=0.5, cos_omega=1.0,
            rho=0.3, radii=np.array([1.0]), sig_t=np.array([2.5]),
        )
        assert tau == pytest.approx(2.5 * 0.3, rel=1e-14)

    def test_ray_stays_in_outer_annulus(self):
        r"""Ray from :math:`r_{\rm obs} = 0.8` outward along the
        positive-:math:`x` axis stays in the outer annulus
        (:math:`r > r_1 = 0.5`) for all :math:`\rho \le 0.2`."""
        radii = np.array([0.5, 1.0])
        sig_t = np.array([1.0, 3.0])
        tau = GEOMETRY.optical_depth_along_ray(
            r_obs=0.8, cos_omega=1.0,
            rho=0.2, radii=radii, sig_t=sig_t,
        )
        # Ray traverses only the outer annulus
        assert tau == pytest.approx(3.0 * 0.2, rel=1e-14)

    def test_ray_crosses_inner_boundary_outward(self):
        r"""Ray from :math:`r_{\rm obs} = 0.3` in the inner annulus
        moving outward crosses :math:`r = r_1 = 0.5` at some
        :math:`\rho^*` and enters the outer annulus.

        For β = 0 (purely radial outward), :math:`\rho^* = r_1 - r_{\rm obs}`.
        """
        r_obs = 0.3
        r_1 = 0.5
        rho = 0.5  # > r_1 - r_obs = 0.2
        radii = np.array([r_1, 1.0])
        sig_t = np.array([1.0, 3.0])
        tau = GEOMETRY.optical_depth_along_ray(
            r_obs=r_obs, cos_omega=1.0,
            rho=rho, radii=radii, sig_t=sig_t,
        )
        rho_star = r_1 - r_obs  # 0.2
        expected = sig_t[0] * rho_star + sig_t[1] * (rho - rho_star)
        assert tau == pytest.approx(expected, rel=1e-12)

    def test_ray_through_center(self):
        r"""Ray from :math:`r_{\rm obs}` in the outer annulus through
        the cylinder axis. For :math:`\beta = \pi` (backward radial),
        the ray passes through :math:`r = 0` at :math:`\rho = r_{\rm obs}`,
        traversing the outer annulus, then the inner annulus, and
        re-entering the outer annulus on the far side."""
        r_obs = 0.8
        r_1 = 0.4
        R = 1.0
        # ρ = R + r_obs = 1.8, covers the full diameter + some extra
        rho = R + r_obs
        radii = np.array([r_1, R])
        sig_t = np.array([1.0, 3.0])
        tau = GEOMETRY.optical_depth_along_ray(
            r_obs=r_obs, cos_omega=-1.0,
            rho=rho, radii=radii, sig_t=sig_t,
        )
        # Path: outer (r_obs - r_1), inner (2·r_1), outer (R + r_obs - (r_obs + r_1))
        # Simplified: outer traversals sum to (r_obs - r_1) + (R - r_1) = r_obs + R - 2·r_1
        expected = (
            sig_t[1] * (r_obs - r_1)     # outer, outbound
            + sig_t[0] * (2.0 * r_1)     # inner, through centre
            + sig_t[1] * (R - r_1)       # outer, outbound on far side
        )
        assert tau == pytest.approx(expected, rel=1e-12)

    def test_ray_tangent_to_inner_boundary(self):
        r"""Ray whose chord has :math:`y > r_1` never enters the inner
        annulus; :math:`\tau = \Sigma_{t,\rm outer} \cdot \rho`."""
        r_obs = 0.8
        r_1 = 0.3
        radii = np.array([r_1, 1.0])
        sig_t = np.array([1.0, 3.0])
        # β = π/2: ray is tangent to the observer's radial direction, so
        # the line has impact parameter y = r_obs · |sin β| = 0.8 > r_1
        tau = GEOMETRY.optical_depth_along_ray(
            r_obs=r_obs, cos_omega=0.0,
            rho=0.4, radii=radii, sig_t=sig_t,
        )
        assert tau == pytest.approx(sig_t[1] * 0.4, rel=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# lagrange_basis_on_panels — piecewise-polynomial basis
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.foundation
class TestLagrangeBasisOnPanels:
    """Standard Lagrange-basis properties (partition of unity, polynomial
    reproduction) — foundation test, not tied to a physics equation."""

    def test_partition_of_unity(self):
        """:math:`\\sum_j L_j(r) = 1` for any :math:`r` in :math:`[0, R]`."""
        radii = np.array([0.4, 1.0])
        r_nodes, _, panels = composite_gl_r(
            radii, n_panels_per_region=2, p_order=4, dps=20,
        )
        for r_eval in np.linspace(0.05, 0.95, 19):
            L = lagrange_basis_on_panels(r_nodes, panels, r_eval)
            assert L.sum() == pytest.approx(1.0, abs=1e-12), (
                f"Partition of unity fails at r = {r_eval}: sum = {L.sum()}"
            )

    def test_reproduces_polynomial(self):
        r"""For any polynomial :math:`p(r)` of degree :math:`< p_{\rm order}`,
        :math:`\sum_j p(r_j)\,L_j(r) = p(r)` on each panel."""
        radii = np.array([1.0])
        p_order = 5
        r_nodes, _, panels = composite_gl_r(
            radii, n_panels_per_region=3, p_order=p_order, dps=20,
        )
        # Test polynomial: p(r) = 1 + 2r - 3r² + r³  (degree 3 < p_order)
        p_coeffs = np.array([1.0, 2.0, -3.0, 1.0])
        p_vals_at_nodes = np.polyval(p_coeffs[::-1], r_nodes)

        # Pick an interior evaluation point in each panel
        for pa, pb, _, _ in panels:
            r_eval = 0.5 * (pa + pb)
            L = lagrange_basis_on_panels(r_nodes, panels, r_eval)
            reproduction = np.dot(L, p_vals_at_nodes)
            truth = np.polyval(p_coeffs[::-1], r_eval)
            assert reproduction == pytest.approx(truth, abs=1e-12), (
                f"Polynomial reproduction fails at r = {r_eval}: "
                f"interpolated = {reproduction}, truth = {truth}"
            )

    def test_nonzero_only_on_owning_panel(self):
        """Lagrange basis evaluated at a point on panel :math:`k` is
        zero on nodes of all other panels (piecewise support)."""
        radii = np.array([1.0])
        r_nodes, _, panels = composite_gl_r(
            radii, n_panels_per_region=3, p_order=4, dps=20,
        )
        # Evaluate at the midpoint of the middle panel
        pa, pb, i_start, i_end = panels[1]
        r_eval = 0.5 * (pa + pb)
        L = lagrange_basis_on_panels(r_nodes, panels, r_eval)
        # L should be zero on nodes outside [i_start:i_end]
        assert np.allclose(L[:i_start], 0.0, atol=1e-14)
        assert np.allclose(L[i_end:], 0.0, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# Multi-region kernel row-sum identity
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-equation")
class TestMultiRegionKernel:
    r"""For a thick inhomogeneous cylinder the correct infinite-medium
    identity for the LHS-\Sigma_t operator is

    .. math::

       \sum_j K_{ij}\,\Sigma_t(r_j) = \Sigma_t(r_i)

    — **apply K to** :math:`q = \Sigma_t`, **not to** :math:`1`. This
    follows from :math:`\int_0^\infty \Sigma_t(r'(\rho))\,
    \mathrm{Ki}_1(\tau(\rho))\,\mathrm{d}\rho = \int_0^\infty
    \mathrm{Ki}_1(u)\,\mathrm{d}u = 1` (change of variables
    :math:`u = \tau(\rho)`), regardless of how :math:`\Sigma_t` varies
    along the ray. The naive "row-sum = Σ_t" identity fails for
    multi-region because it corresponds to :math:`\int \mathrm{Ki}_1(\tau)
    \mathrm{d}\rho = \int \mathrm{Ki}_1(u)/\Sigma_t(r'(u))\,\mathrm{d}u`,
    a weighted path average of :math:`1/\Sigma_t`.
    """

    def test_K_applied_to_sig_t_gives_local_sig_t(self):
        r"""For pure-scatter uniform flux :math:`\varphi \equiv 1`,
        the source :math:`q(r') = \Sigma_t(r')\cdot\varphi(r')` drives
        the Peierls operator back to :math:`\Sigma_t(r_i)` at every
        :math:`r_i`. Tests :math:`\sum_j K_{ij}\,\Sigma_t(r_j) =
        \Sigma_t(r_i)` to within the escape-probability tolerance."""
        r_1 = 3.0
        R = 10.0
        radii = np.array([r_1, R])
        sig_t_inner = 0.8
        sig_t_outer = 1.4
        sig_t = np.array([sig_t_inner, sig_t_outer])

        r_nodes, _, panels = composite_gl_r(
            radii, n_panels_per_region=2, p_order=5, dps=20,
        )
        K = build_volume_kernel(
            GEOMETRY, r_nodes, panels, radii, sig_t,
            n_angular=24, n_rho=24, dps=20,
        )

        # Build q_j = Σ_t(r_j) on the grid using the same panel lookup
        # that build_volume_kernel uses internally.
        q = np.empty_like(r_nodes)
        for j, rj in enumerate(r_nodes):
            # Panel breakpoints happen to sit exactly at annular radii,
            # so r_j < r_1 always lives in the inner annulus panel.
            q[j] = sig_t_inner if rj < r_1 else sig_t_outer

        Kq = K @ q

        for ri, kq in zip(r_nodes, Kq):
            if ri < r_1 - 0.5:  # deep interior of inner annulus
                expected = sig_t_inner
                tol = 5e-3
            elif r_1 + 0.5 < ri < R - 3.0:  # bulk of outer annulus
                expected = sig_t_outer
                tol = 5e-3
            else:
                continue  # boundary layers excluded
            assert kq == pytest.approx(expected, abs=tol), (
                f"(K·q)(r = {ri:.3f}) = {kq:.6f}, expected "
                f"Σ_t = {expected}"
            )

    def test_reduces_to_homogeneous_when_sigma_t_equal(self):
        """If both annuli carry the same Σ_t, the multi-region
        row-sums equal the homogeneous-case row-sums on the
        overlapping grid — to within quadrature error."""
        R = 10.0
        sig_t_val = 1.2

        # Single-region reference
        r_nodes_1, _, panels_1 = composite_gl_r(
            np.array([R]), n_panels_per_region=2, p_order=5, dps=20,
        )
        K_1 = build_volume_kernel(
            GEOMETRY, r_nodes_1, panels_1, np.array([R]), np.array([sig_t_val]),
            n_angular=20, n_rho=20, dps=20,
        )

        # Multi-region: 4 annuli with identical Σ_t
        radii_4 = np.array([2.5, 5.0, 7.5, R])
        r_nodes_4, _, panels_4 = composite_gl_r(
            radii_4, n_panels_per_region=1, p_order=5, dps=20,
        )
        K_4 = build_volume_kernel(
            GEOMETRY, r_nodes_4, panels_4, radii_4, np.array([sig_t_val] * 4),
            n_angular=20, n_rho=20, dps=20,
        )
        rs_1 = K_1.sum(axis=1)
        rs_4 = K_4.sum(axis=1)

        # Interior means should match
        interior_1 = rs_1[(r_nodes_1 > 1.0) & (r_nodes_1 < R - 2.0)]
        interior_4 = rs_4[(r_nodes_4 > 1.0) & (r_nodes_4 < R - 2.0)]
        assert abs(interior_1.mean() - interior_4.mean()) < 5e-4, (
            f"1R mean = {interior_1.mean():.6f}, "
            f"4R (homogeneous) mean = {interior_4.mean():.6f}"
        )
