"""Tests for angular quadrature implementations.

Covers all four quadrature types:
- GaussLegendre1D
- LebedevSphere
- LevelSymmetricSN
- ProductQuadrature

Tests verify mathematical properties that any valid quadrature must satisfy.
"""

import numpy as np
import pytest

from sn_quadrature import (
    GaussLegendre1D,
    LebedevSphere,
    LevelSymmetricSN,
    ProductQuadrature,
)


# ═══════════════════════════════════════════════════════════════════════
# Weight sums
# ═══════════════════════════════════════════════════════════════════════

class TestWeightSums:
    """Quadrature weights must integrate to the correct solid angle."""

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_gl_weights_sum_to_2(self, N):
        quad = GaussLegendre1D.create(N)
        np.testing.assert_allclose(quad.weights.sum(), 2.0, atol=1e-14)

    def test_lebedev_weights_sum_to_4pi(self):
        quad = LebedevSphere.create(order=17)
        np.testing.assert_allclose(quad.weights.sum(), 4 * np.pi, rtol=1e-12)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_level_symmetric_weights_sum_to_4pi(self, order):
        quad = LevelSymmetricSN.create(order)
        np.testing.assert_allclose(quad.weights.sum(), 4 * np.pi, rtol=1e-12)

    @pytest.mark.parametrize("n_mu,n_phi", [(4, 4), (4, 8), (8, 8)])
    def test_product_weights_sum_to_4pi(self, n_mu, n_phi):
        quad = ProductQuadrature.create(n_mu, n_phi)
        np.testing.assert_allclose(quad.weights.sum(), 4 * np.pi, rtol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# Unit sphere condition
# ═══════════════════════════════════════════════════════════════════════

class TestUnitSphere:
    """All ordinates must lie on the unit sphere: η² + ξ² + μ² = 1."""

    def test_lebedev(self):
        quad = LebedevSphere.create(order=17)
        norm = quad.mu_x**2 + quad.mu_y**2 + quad.mu_z**2
        np.testing.assert_allclose(norm, 1.0, atol=1e-14)

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_level_symmetric(self, order):
        quad = LevelSymmetricSN.create(order)
        norm = quad.mu_x**2 + quad.mu_y**2 + quad.mu_z**2
        np.testing.assert_allclose(norm, 1.0, atol=1e-14)

    @pytest.mark.parametrize("n_mu,n_phi", [(4, 8), (8, 8)])
    def test_product(self, n_mu, n_phi):
        quad = ProductQuadrature.create(n_mu, n_phi)
        norm = quad.mu_x**2 + quad.mu_y**2 + quad.mu_z**2
        np.testing.assert_allclose(norm, 1.0, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# Moment conditions
# ═══════════════════════════════════════════════════════════════════════

class TestMomentConditions:
    """Quadratures must integrate low-order polynomials exactly.

    ∫ dΩ = 4π, ∫ μ_i² dΩ = 4π/3 for i ∈ {x, y, z}.
    """

    @pytest.mark.parametrize("order", [4, 6, 8])
    def test_level_symmetric_second_moments(self, order):
        quad = LevelSymmetricSN.create(order)
        target = 4 * np.pi / 3
        for attr in ['mu_x', 'mu_y', 'mu_z']:
            m2 = np.sum(quad.weights * getattr(quad, attr)**2)
            np.testing.assert_allclose(m2, target, rtol=1e-10,
                                       err_msg=f"∫{attr}² dΩ ≠ 4π/3")

    @pytest.mark.parametrize("n_mu,n_phi", [(4, 8), (8, 8)])
    def test_product_second_moments(self, n_mu, n_phi):
        quad = ProductQuadrature.create(n_mu, n_phi)
        target = 4 * np.pi / 3
        for attr in ['mu_x', 'mu_y', 'mu_z']:
            m2 = np.sum(quad.weights * getattr(quad, attr)**2)
            np.testing.assert_allclose(m2, target, rtol=1e-10,
                                       err_msg=f"∫{attr}² dΩ ≠ 4π/3")

    def test_lebedev_second_moments(self):
        quad = LebedevSphere.create(order=17)
        target = 4 * np.pi / 3
        for attr in ['mu_x', 'mu_y', 'mu_z']:
            m2 = np.sum(quad.weights * getattr(quad, attr)**2)
            np.testing.assert_allclose(m2, target, rtol=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# Level structure (for cylindrical sweep)
# ═══════════════════════════════════════════════════════════════════════

class TestLevelStructure:
    """Level-indexed quadratures must have consistent level_indices."""

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_level_sym_indices_partition(self, order):
        """Level indices must cover all ordinates exactly once."""
        quad = LevelSymmetricSN.create(order)
        all_idx = np.sort(np.concatenate(quad.level_indices))
        np.testing.assert_array_equal(all_idx, np.arange(quad.N))

    @pytest.mark.parametrize("n_mu,n_phi", [(4, 4), (8, 8)])
    def test_product_indices_partition(self, n_mu, n_phi):
        quad = ProductQuadrature.create(n_mu, n_phi)
        all_idx = np.sort(np.concatenate(quad.level_indices))
        np.testing.assert_array_equal(all_idx, np.arange(quad.N))

    def test_product_level_mu_match(self):
        """On each level, all ordinates must share the same μ_z value."""
        quad = ProductQuadrature.create(n_mu=4, n_phi=8)
        for p, idx in enumerate(quad.level_indices):
            mu_vals = quad.mu_z[idx]
            np.testing.assert_allclose(mu_vals, quad.level_mu[p], atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# Reflection indices
# ═══════════════════════════════════════════════════════════════════════

class TestReflectionIndices:
    """Reflection partner must have the negated direction cosine."""

    @pytest.mark.parametrize("QuadClass,kwargs", [
        (LevelSymmetricSN, {"sn_order": 4}),
        (ProductQuadrature, {"n_mu": 4, "n_phi": 8}),
    ])
    def test_x_reflection(self, QuadClass, kwargs):
        quad = QuadClass.create(**kwargs)
        ref = quad.reflection_index("x")
        # μ_x of reflected partner should be -μ_x of original
        np.testing.assert_allclose(quad.mu_x[ref], -quad.mu_x, atol=1e-12)

    @pytest.mark.parametrize("QuadClass,kwargs", [
        (LevelSymmetricSN, {"sn_order": 4}),
        (ProductQuadrature, {"n_mu": 4, "n_phi": 8}),
    ])
    def test_reflection_involution(self, QuadClass, kwargs):
        """Reflecting twice must return to the original ordinate."""
        quad = QuadClass.create(**kwargs)
        for axis in ["x", "y", "z"]:
            ref = quad.reflection_index(axis)
            np.testing.assert_array_equal(ref[ref], np.arange(quad.N),
                                          err_msg=f"{axis}-reflection not involution")


# ═══════════════════════════════════════════════════════════════════════
# Alpha redistribution coefficient properties
# ═══════════════════════════════════════════════════════════════════════

class TestAlphaRedistribution:
    """Verify α coefficient properties required for curvilinear SN sweeps.

    The α recursion (Bailey et al. 2009, Eq. 50) uses the radial
    direction cosine η (mu_x): α_{m+1/2} = α_{m-1/2} − w_m · η_m.
    The resulting dome must be non-negative with α[0] = α[M] = 0.
    """

    @pytest.mark.parametrize("QuadClass,kwargs", [
        (ProductQuadrature, {"n_mu": 4, "n_phi": 8}),
        (ProductQuadrature, {"n_mu": 8, "n_phi": 16}),
        (LevelSymmetricSN, {"sn_order": 4}),
        (LevelSymmetricSN, {"sn_order": 6}),
    ])
    def test_alpha_dome_non_negative(self, QuadClass, kwargs):
        """α values must form a non-negative dome on each level."""
        from geometry import CoordSystem, Mesh1D
        from sn_geometry import SNMesh

        quad = QuadClass.create(**kwargs)
        mesh = Mesh1D(
            edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
            coord=CoordSystem.CYLINDRICAL,
        )
        sn_mesh = SNMesh(mesh, quad)

        for p, alpha in enumerate(sn_mesh.alpha_per_level):
            assert np.all(alpha >= -1e-14), (
                f"Level {p}: negative α = {alpha.min():.2e}"
            )

    @pytest.mark.parametrize("QuadClass,kwargs", [
        (ProductQuadrature, {"n_mu": 4, "n_phi": 8}),
        (LevelSymmetricSN, {"sn_order": 4}),
    ])
    def test_alpha_boundary_zero(self, QuadClass, kwargs):
        """α must be zero at both dome boundaries (conservation)."""
        from geometry import CoordSystem, Mesh1D
        from sn_geometry import SNMesh

        quad = QuadClass.create(**kwargs)
        mesh = Mesh1D(
            edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
            coord=CoordSystem.CYLINDRICAL,
        )
        sn_mesh = SNMesh(mesh, quad)

        for p, alpha in enumerate(sn_mesh.alpha_per_level):
            np.testing.assert_allclose(alpha[0], 0.0,
                                       err_msg=f"Level {p}: α[0] ≠ 0")
            np.testing.assert_allclose(alpha[-1], 0.0, atol=1e-13,
                                       err_msg=f"Level {p}: α[-1] ≠ 0")

    def test_spherical_alpha_dome_non_negative(self):
        """Spherical α (cumsum(−w·μ)) must be non-negative for GL quadrature."""
        from geometry import CoordSystem, Mesh1D
        from sn_geometry import SNMesh

        quad = GaussLegendre1D.create(8)
        mesh = Mesh1D(
            edges=np.array([0.0, 1.0]), mat_ids=np.array([0]),
            coord=CoordSystem.SPHERICAL,
        )
        sn_mesh = SNMesh(mesh, quad)

        assert np.all(sn_mesh.alpha_half >= -1e-14), (
            f"Negative spherical α: min = {sn_mesh.alpha_half.min():.2e}"
        )
