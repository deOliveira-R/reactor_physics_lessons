"""L0 geometry tests for the spherical Peierls ray primitives.

Mirror of ``test_peierls_cylinder_geometry.py`` — the spherical
specialisation uses the same composite-GL radial quadrature and the
same ray-walker machinery in :class:`CurvilinearGeometry`, but
verified separately so a regression in the sphere branch cannot hide
behind passing cylinder tests.

The walker verified here is
:meth:`~peierls_geometry.CurvilinearGeometry.optical_depth_along_ray`
parameterised by observer-centred polar angle :math:`\\theta`
(:math:`\\cos\\theta` argument), as used by the sphere solver. The
ray-exit distance :math:`\\rho_{\\max}(r_i, \\theta)` is the same
closed form as the cylinder case (the 1-D radial chord algebra is
geometry-agnostic)."""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.peierls_geometry import composite_gl_r
from orpheus.derivations.peierls_sphere import GEOMETRY


# ═══════════════════════════════════════════════════════════════════════
# Geometry constants
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSphereGeometryConstants:
    """Dimensional and angular-measure constants of the sphere branch."""

    def test_effective_dimension_is_three(self):
        assert GEOMETRY.d == 3

    def test_total_solid_angle_is_4pi(self):
        assert GEOMETRY.S_d == pytest.approx(4.0 * np.pi, rel=1e-14)

    def test_prefactor_is_half(self):
        """The sphere prefactor collects :math:`1/(4\\pi)` (3-D point
        kernel) times :math:`2\\pi` (trivial azimuthal) = :math:`1/2`."""
        assert GEOMETRY.prefactor == pytest.approx(0.5, rel=1e-14)

    def test_angular_range_is_zero_to_pi(self):
        assert GEOMETRY.angular_range == (0.0, np.pi)

    def test_angular_weight_is_sin_theta(self):
        """The sphere integrates :math:`\\sin\\theta\\,\\mathrm d\\theta`
        over :math:`[0, \\pi]` (azimuthal already folded)."""
        theta = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        np.testing.assert_allclose(
            GEOMETRY.angular_weight(theta), np.sin(theta), rtol=1e-14,
        )

    def test_radial_volume_weight_is_r_squared(self):
        """dV = 4π r² dr ⇒ the quadrature weight is r² (the 4π is in
        the angular prefactor, not duplicated here)."""
        rs = np.array([0.5, 1.2, 3.7])
        for r in rs:
            assert GEOMETRY.radial_volume_weight(r) == pytest.approx(r * r)

    def test_rank1_surface_divisor_is_R_squared(self):
        """Rank-1 white-BC normalisation uses R² for sphere (vs R for
        cylinder). See derivation in :meth:`rank1_surface_divisor`."""
        for R in (0.5, 1.0, 2.5, 10.0):
            assert GEOMETRY.rank1_surface_divisor(R) == pytest.approx(R * R)


# ═══════════════════════════════════════════════════════════════════════
# Ray-exit distance ρ_max(r_i, θ, R)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSphereRhoMax:
    """Closed-form checks for the ray-exit distance.

    The positive root of :math:`(r_i + \\rho\\cos\\theta)^2 +
    (\\rho\\sin\\theta)^2 = R^2` is identical for cylinder and sphere
    (the 1-D radial chord algebra is geometry-invariant in 3-D)."""

    def test_radial_outward_ray_hits_at_R_minus_r(self):
        """θ = 0 (pure outward radial): ρ_max = R − r_i."""
        R = 2.5
        for r_i in (0.0, 0.5, 1.0, 2.0, 2.4):
            rho = GEOMETRY.rho_max(r_i, cos_omega=1.0, R=R)
            assert rho == pytest.approx(R - r_i, rel=1e-14, abs=1e-14)

    def test_radial_inward_ray_crosses_diameter(self):
        """θ = π (through centre): ρ_max = R + r_i (exits opposite side)."""
        R = 2.5
        for r_i in (0.1, 0.5, 1.0, 2.0):
            rho = GEOMETRY.rho_max(r_i, cos_omega=-1.0, R=R)
            assert rho == pytest.approx(R + r_i, rel=1e-14)

    def test_tangential_ray_from_center_equals_R(self):
        """r_i = 0, any θ: every ray traverses exactly R to reach the surface."""
        R = 4.0
        for theta in (0.0, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi):
            rho = GEOMETRY.rho_max(0.0, cos_omega=np.cos(theta), R=R)
            assert rho == pytest.approx(R, rel=1e-14)

    def test_observer_on_surface_outward_ray_zero(self):
        """r_i = R, θ < π/2: ρ_max = 0 (ray leaves immediately)."""
        R = 1.5
        for theta in (0.0, np.pi / 4, np.pi / 3):
            rho = GEOMETRY.rho_max(R, cos_omega=np.cos(theta), R=R)
            assert rho == pytest.approx(0.0, abs=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# Source position r'(ρ, θ, r_i)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSphereSourcePosition:
    """r'(ρ, θ, r_i) = √(r_i² + 2 r_i ρ cos θ + ρ²)."""

    def test_radial_outward_traversal(self):
        """θ = 0: r' = r_i + ρ."""
        for r_i, rho in ((0.0, 1.0), (0.5, 1.5), (1.0, 0.25)):
            rp = GEOMETRY.source_position(r_i, rho, cos_omega=1.0)
            assert rp == pytest.approx(r_i + rho, rel=1e-14)

    def test_radial_inward_through_centre(self):
        """θ = π: r' = |r_i − ρ|."""
        for r_i, rho in ((1.0, 0.3), (1.0, 1.5), (2.0, 0.5)):
            rp = GEOMETRY.source_position(r_i, rho, cos_omega=-1.0)
            assert rp == pytest.approx(abs(r_i - rho), rel=1e-14)

    def test_perpendicular_ray_from_interior(self):
        """θ = π/2: r' = √(r_i² + ρ²) (perpendicular chord)."""
        r_i, rho = 1.2, 0.7
        rp = GEOMETRY.source_position(r_i, rho, cos_omega=0.0)
        assert rp == pytest.approx(np.hypot(r_i, rho), rel=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# Optical depth along ray
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSphereOpticalDepthAlongRay:
    """Closed-form checks for the directed ray optical-depth walker."""

    def test_homogeneous_1region_linear_in_rho(self):
        """τ(r_i, θ, ρ) = Σ_t · ρ for any direction in a 1-region sphere."""
        radii = np.array([5.0])
        sig_t = np.array([0.7])
        for r_i, ct in ((0.0, 1.0), (1.0, 0.0), (2.0, -0.5), (3.0, 0.3)):
            for rho in (0.5, 1.0, 2.5):
                tau = GEOMETRY.optical_depth_along_ray(
                    r_i, cos_omega=ct, rho=rho, radii=radii, sig_t=sig_t,
                )
                assert tau == pytest.approx(sig_t[0] * rho, rel=1e-12, abs=1e-14)

    def test_scales_linearly_with_sig_t(self):
        """Doubling Σ_t doubles the optical depth (no coupling to geometry)."""
        radii = np.array([0.4, 1.0])
        sig_t_a = np.array([0.6, 1.8])
        sig_t_b = 2.5 * sig_t_a
        r_i, ct, rho = 0.2, 0.3, 0.9

        tau_a = GEOMETRY.optical_depth_along_ray(r_i, ct, rho, radii, sig_t_a)
        tau_b = GEOMETRY.optical_depth_along_ray(r_i, ct, rho, radii, sig_t_b)
        assert tau_b == pytest.approx(2.5 * tau_a, rel=1e-12)

    def test_two_annulus_radial_transit(self):
        """Pure outward radial ray from r_i = 0 to r_i = R in a two-shell
        geometry: τ = Σ_{t,1} · r_1 + Σ_{t,2} · (R − r_1)."""
        r1, R = 0.4, 1.0
        radii = np.array([r1, R])
        sig_t = np.array([0.9, 2.1])
        tau = GEOMETRY.optical_depth_along_ray(
            r_obs=0.0, cos_omega=1.0, rho=R, radii=radii, sig_t=sig_t,
        )
        expected = sig_t[0] * r1 + sig_t[1] * (R - r1)
        assert tau == pytest.approx(expected, rel=1e-12, abs=1e-14)

    def test_two_annulus_through_centre_diameter(self):
        """From r_obs = R going through the centre (cos θ = -1) to the
        far surface (ρ = 2R): the ray traverses the outer shell twice
        and the inner shell once across the full diameter (2 r_1)."""
        r1, R = 0.4, 1.0
        radii = np.array([r1, R])
        sig_t = np.array([0.9, 2.1])
        tau = GEOMETRY.optical_depth_along_ray(
            r_obs=R, cos_omega=-1.0, rho=2 * R, radii=radii, sig_t=sig_t,
        )
        expected = sig_t[0] * (2 * r1) + sig_t[1] * 2 * (R - r1)
        assert tau == pytest.approx(expected, rel=1e-12, abs=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# Composite radial GL sanity checks (shared with cylinder)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.l0
@pytest.mark.verifies("peierls-unified")
class TestSphereCompositeRadialGL:
    """The composite radial rule is geometry-agnostic; re-verify it via
    the sphere import path so a broken re-export is caught."""

    def test_integrates_constant_to_R(self):
        """∫₀ᴿ 1 dr = R."""
        radii = np.array([0.4, 1.0])
        r_pts, r_wts, _ = composite_gl_r(radii, n_panels_per_region=4, p_order=6)
        assert np.isclose(r_wts.sum(), radii[-1], rtol=1e-14)

    def test_integrates_r_squared_weighted_volume(self):
        """∫₀ᴿ r² dr = R³/3 (the 4π is in the angular prefactor)."""
        radii = np.array([0.5, 1.0, 2.0])
        r_pts, r_wts, _ = composite_gl_r(radii, n_panels_per_region=3, p_order=4)
        R = radii[-1]
        integral = np.dot(r_wts, r_pts ** 2)
        assert np.isclose(integral, R ** 3 / 3.0, rtol=1e-14)

    def test_panel_bounds_partition_r_axis(self):
        radii = np.array([0.4, 0.9, 1.5])
        _, _, panel_bounds = composite_gl_r(
            radii, n_panels_per_region=2, p_order=4,
        )
        assert panel_bounds[0][0] == pytest.approx(0.0, abs=1e-14)
        assert panel_bounds[-1][1] == pytest.approx(radii[-1], rel=1e-14)
        for p0, p1 in zip(panel_bounds[:-1], panel_bounds[1:]):
            assert p0[1] == pytest.approx(p1[0], rel=1e-14)
