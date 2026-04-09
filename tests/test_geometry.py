"""Comprehensive tests for the geometry module.

Tests cover:
- Volume and surface formulas for all coordinate systems (1-D and 2-D)
- Zone subdivision equal-volume property
- Edge position formulas for all coordinate systems
- PWR factory outputs match legacy geometry classes exactly
- Mesh validation (monotonicity, shape, frozen immutability)
- Solver guard patterns
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.geometry import (
    CoordSystem,
    Mesh1D,
    Mesh2D,
    Zone,
    compute_surfaces_1d,
    compute_volumes_1d,
    compute_volumes_2d,
    homogeneous_1d,
    mesh1d_from_zones,
    pwr_pin_2d,
    pwr_pin_equivalent,
    pwr_slab_half_cell,
    slab_fuel_moderator,
)


# ═══════════════════════════════════════════════════════════════════════
# Volume formulas (1-D)
# ═══════════════════════════════════════════════════════════════════════

class TestVolumes1D:
    """Volume formulas for all 1-D coordinate systems."""

    def test_cartesian_single_cell(self):
        edges = np.array([1.0, 3.0])
        vol = compute_volumes_1d(CoordSystem.CARTESIAN, edges)
        np.testing.assert_allclose(vol, [2.0])

    def test_cartesian_multiple_cells(self):
        edges = np.array([0.0, 0.5, 1.5, 4.0])
        vol = compute_volumes_1d(CoordSystem.CARTESIAN, edges)
        np.testing.assert_allclose(vol, [0.5, 1.0, 2.5])

    def test_cylindrical_single_cell(self):
        edges = np.array([1.0, 2.0])
        vol = compute_volumes_1d(CoordSystem.CYLINDRICAL, edges)
        expected = np.pi * (4.0 - 1.0)  # pi * (r2^2 - r1^2)
        np.testing.assert_allclose(vol, [expected])

    def test_cylindrical_zero_inner_radius(self):
        edges = np.array([0.0, 1.0])
        vol = compute_volumes_1d(CoordSystem.CYLINDRICAL, edges)
        np.testing.assert_allclose(vol, [np.pi])

    def test_cylindrical_multiple_cells(self):
        edges = np.array([0.0, 0.5, 1.0, 2.0])
        vol = compute_volumes_1d(CoordSystem.CYLINDRICAL, edges)
        expected = np.pi * np.diff(edges**2)
        np.testing.assert_allclose(vol, expected)

    def test_spherical_single_cell(self):
        edges = np.array([1.0, 2.0])
        vol = compute_volumes_1d(CoordSystem.SPHERICAL, edges)
        expected = (4.0 / 3.0) * np.pi * (8.0 - 1.0)
        np.testing.assert_allclose(vol, [expected])

    def test_spherical_zero_inner_radius(self):
        edges = np.array([0.0, 1.0])
        vol = compute_volumes_1d(CoordSystem.SPHERICAL, edges)
        np.testing.assert_allclose(vol, [(4.0 / 3.0) * np.pi])

    def test_spherical_multiple_cells(self):
        edges = np.array([0.0, 0.5, 1.0, 2.0])
        vol = compute_volumes_1d(CoordSystem.SPHERICAL, edges)
        expected = (4.0 / 3.0) * np.pi * np.diff(edges**3)
        np.testing.assert_allclose(vol, expected)


# ═══════════════════════════════════════════════════════════════════════
# Surface formulas (1-D)
# ═══════════════════════════════════════════════════════════════════════

class TestSurfaces1D:
    """Surface area formulas at each edge."""

    def test_cartesian(self):
        edges = np.array([0.0, 1.0, 3.0])
        surf = compute_surfaces_1d(CoordSystem.CARTESIAN, edges)
        np.testing.assert_allclose(surf, [1.0, 1.0, 1.0])

    def test_cylindrical(self):
        edges = np.array([0.0, 0.5, 1.0])
        surf = compute_surfaces_1d(CoordSystem.CYLINDRICAL, edges)
        expected = 2.0 * np.pi * edges
        np.testing.assert_allclose(surf, expected)

    def test_spherical(self):
        edges = np.array([0.0, 0.5, 1.0])
        surf = compute_surfaces_1d(CoordSystem.SPHERICAL, edges)
        expected = 4.0 * np.pi * edges**2
        np.testing.assert_allclose(surf, expected)


# ═══════════════════════════════════════════════════════════════════════
# Volume formulas (2-D)
# ═══════════════════════════════════════════════════════════════════════

class TestVolumes2D:
    """Volume formulas for 2-D coordinate systems."""

    def test_cartesian_uniform(self):
        edges_x = np.array([0.0, 1.0, 2.0])
        edges_y = np.array([0.0, 0.5, 1.5])
        vol = compute_volumes_2d(CoordSystem.CARTESIAN, edges_x, edges_y)
        expected = np.array([[0.5, 1.0], [0.5, 1.0]])
        np.testing.assert_allclose(vol, expected)

    def test_cylindrical_rz(self):
        # r-z: V = pi*(r_out^2 - r_in^2) * dz
        edges_r = np.array([0.0, 1.0, 2.0])
        edges_z = np.array([0.0, 3.0])
        vol = compute_volumes_2d(CoordSystem.CYLINDRICAL, edges_r, edges_z)
        expected = np.pi * np.array([[1.0], [3.0]]) * 3.0
        np.testing.assert_allclose(vol, expected)

    def test_spherical_raises(self):
        edges = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="2-D volumes not defined"):
            compute_volumes_2d(CoordSystem.SPHERICAL, edges, edges)


# ═══════════════════════════════════════════════════════════════════════
# Zone subdivision — equal-volume property
# ═══════════════════════════════════════════════════════════════════════

class TestZoneSubdivision:
    """Verify that zone subdivision produces equal-volume cells."""

    @pytest.mark.parametrize("coord", list(CoordSystem))
    def test_equal_volume_single_zone(self, coord):
        """All sub-cells within a zone must have equal volume."""
        n = 20
        zones = [Zone(outer_edge=2.0, mat_id=0, n_cells=n)]
        mesh = mesh1d_from_zones(zones, coord=coord)
        vols = mesh.volumes
        np.testing.assert_allclose(vols, vols[0], rtol=1e-14)

    @pytest.mark.parametrize("coord", list(CoordSystem))
    def test_equal_volume_multi_zone(self, coord):
        """Equal-volume within each zone, but zones may differ."""
        zones = [
            Zone(outer_edge=1.0, mat_id=0, n_cells=10),
            Zone(outer_edge=3.0, mat_id=1, n_cells=15),
        ]
        mesh = mesh1d_from_zones(zones, coord=coord)
        vols = mesh.volumes
        # Zone 0: cells 0..9
        np.testing.assert_allclose(vols[:10], vols[0], rtol=1e-14)
        # Zone 1: cells 10..24
        np.testing.assert_allclose(vols[10:], vols[10], rtol=1e-14)

    def test_cartesian_equal_width(self):
        """Cartesian subdivision gives equal-width cells."""
        zones = [Zone(outer_edge=3.0, mat_id=0, n_cells=6)]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.CARTESIAN)
        np.testing.assert_allclose(mesh.widths, 0.5, rtol=1e-14)

    def test_cylindrical_edge_positions(self):
        """Cylindrical: r_k = sqrt(k/N * r_outer^2) for origin=0."""
        n = 5
        r_out = 2.0
        zones = [Zone(outer_edge=r_out, mat_id=0, n_cells=n)]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.CYLINDRICAL)
        expected = r_out * np.sqrt(np.arange(n + 1) / n)
        np.testing.assert_allclose(mesh.edges, expected, rtol=1e-14)

    def test_spherical_edge_positions(self):
        """Spherical: r_k = cbrt(k/N * r_outer^3) for origin=0."""
        n = 5
        r_out = 2.0
        zones = [Zone(outer_edge=r_out, mat_id=0, n_cells=n)]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.SPHERICAL)
        expected = r_out * np.cbrt(np.arange(n + 1) / n)
        np.testing.assert_allclose(mesh.edges, expected, rtol=1e-14)

    def test_cylindrical_nonzero_inner(self):
        """Cylindrical with nonzero inner radius."""
        r_in, r_out, n = 1.0, 2.0, 8
        zones = [Zone(outer_edge=r_out, mat_id=0, n_cells=n)]
        mesh = mesh1d_from_zones(zones, coord=CoordSystem.CYLINDRICAL, origin=r_in)
        fracs = np.linspace(0.0, 1.0, n + 1)
        expected = np.sqrt(r_in**2 + fracs * (r_out**2 - r_in**2))
        np.testing.assert_allclose(mesh.edges, expected, rtol=1e-14)
        np.testing.assert_allclose(mesh.volumes, mesh.volumes[0], rtol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
# PWR factories — match legacy geometry outputs
# ═══════════════════════════════════════════════════════════════════════

class TestPWRFactories:
    """PWR factories produce meshes matching the old geometry classes."""

    def test_slab_half_cell_volumes(self):
        """pwr_slab_half_cell matches SlabGeometry.default_pwr volumes."""
        mesh = pwr_slab_half_cell()
        # SlabGeometry had: n_fuel=10, n_clad=3, n_cool=7
        # thicknesses: fuel=0.9/10, clad=0.2/3, cool=0.7/7
        assert mesh.N == 20
        assert mesh.coord == CoordSystem.CARTESIAN
        # Cartesian: volumes = widths = thicknesses
        np.testing.assert_allclose(mesh.widths[:10], 0.09, rtol=1e-14)
        np.testing.assert_allclose(mesh.widths[10:13], 0.2 / 3, rtol=1e-14)
        np.testing.assert_allclose(mesh.widths[13:], 0.1, rtol=1e-14)

    def test_slab_half_cell_mat_ids(self):
        mesh = pwr_slab_half_cell()
        assert np.all(mesh.mat_ids[:10] == 2)   # fuel
        assert np.all(mesh.mat_ids[10:13] == 1)  # clad
        assert np.all(mesh.mat_ids[13:] == 0)    # cool

    def test_slab_half_cell_total_width(self):
        mesh = pwr_slab_half_cell()
        np.testing.assert_allclose(mesh.total_width, 0.9 + 0.2 + 0.7)

    def test_pin_equivalent_n_cells(self):
        mesh = pwr_pin_equivalent()
        assert mesh.N == 20
        assert mesh.coord == CoordSystem.CYLINDRICAL

    def test_pin_equivalent_r_cell(self):
        """Outer edge = pitch / sqrt(pi)."""
        pitch = 3.6
        mesh = pwr_pin_equivalent(pitch=pitch)
        r_cell = pitch / np.sqrt(np.pi)
        np.testing.assert_allclose(mesh.edges[-1], r_cell, rtol=1e-14)

    def test_pin_equivalent_volumes_match_legacy(self):
        """Volumes must match CPGeometry.default_pwr() exactly."""
        mesh = pwr_pin_equivalent()

        # Reproduce CPGeometry.default_pwr logic:
        r_fuel, r_clad, pitch = 0.9, 1.1, 3.6
        r_cell = pitch / np.sqrt(np.pi)
        n_fuel, n_clad, n_cool = 10, 3, 7
        N = 20
        radii = np.empty(N)
        for k in range(n_fuel):
            radii[k] = r_fuel * np.sqrt((k + 1) / n_fuel)
        for k in range(n_clad):
            radii[n_fuel + k] = np.sqrt(
                r_fuel**2 + (k + 1) / n_clad * (r_clad**2 - r_fuel**2)
            )
        for k in range(n_cool):
            radii[n_fuel + n_clad + k] = np.sqrt(
                r_clad**2 + (k + 1) / n_cool * (r_cell**2 - r_clad**2)
            )
        r_inner = np.zeros(N)
        r_inner[1:] = radii[:-1]
        legacy_volumes = np.pi * (radii**2 - r_inner**2)

        np.testing.assert_allclose(mesh.volumes, legacy_volumes, rtol=1e-13)

    def test_pin_equivalent_mat_ids(self):
        mesh = pwr_pin_equivalent()
        assert np.all(mesh.mat_ids[:10] == 2)   # fuel
        assert np.all(mesh.mat_ids[10:13] == 1)  # clad
        assert np.all(mesh.mat_ids[13:] == 0)    # cool

    def test_pin_equivalent_surfaces(self):
        """Outer surface = 2*pi*r_cell (cylindrical)."""
        mesh = pwr_pin_equivalent()
        r_cell = mesh.edges[-1]
        np.testing.assert_allclose(
            mesh.surfaces[-1], 2.0 * np.pi * r_cell, rtol=1e-14,
        )

    def test_pin_2d_shape(self):
        mesh = pwr_pin_2d(n_cells=10)
        assert mesh.nx == 10
        assert mesh.ny == 10
        assert mesh.mat_map.shape == (10, 10)

    def test_pin_2d_has_all_materials(self):
        mesh = pwr_pin_2d(n_cells=20)
        mats = set(mesh.mat_map.ravel())
        assert mats == {0, 1, 2}

    def test_pin_2d_mat_ids_flat(self):
        """mat_ids returns flat array for assemble_cell_xs."""
        mesh = pwr_pin_2d(n_cells=5)
        assert mesh.mat_ids.shape == (25,)


# ═══════════════════════════════════════════════════════════════════════
# Mesh1D properties and validation
# ═══════════════════════════════════════════════════════════════════════

class TestMesh1D:
    """Mesh1D derived properties and input validation."""

    def test_widths(self):
        mesh = Mesh1D(edges=np.array([0.0, 1.0, 3.0, 6.0]),
                      mat_ids=np.array([0, 1, 2]))
        np.testing.assert_allclose(mesh.widths, [1.0, 2.0, 3.0])

    def test_centers(self):
        mesh = Mesh1D(edges=np.array([0.0, 2.0, 6.0]),
                      mat_ids=np.array([0, 1]))
        np.testing.assert_allclose(mesh.centers, [1.0, 4.0])

    def test_total_width(self):
        mesh = Mesh1D(edges=np.array([1.0, 3.0, 7.0]),
                      mat_ids=np.array([0, 1]))
        assert mesh.total_width == 6.0

    def test_N(self):
        mesh = Mesh1D(edges=np.arange(6.0),
                      mat_ids=np.zeros(5, dtype=int))
        assert mesh.N == 5

    def test_frozen(self):
        mesh = Mesh1D(edges=np.array([0.0, 1.0]),
                      mat_ids=np.array([0]))
        with pytest.raises(AttributeError):
            mesh.edges = np.array([0.0, 2.0])

    def test_non_monotonic_edges_raises(self):
        with pytest.raises(ValueError, match="monotonically increasing"):
            Mesh1D(edges=np.array([0.0, 2.0, 1.0]),
                   mat_ids=np.array([0, 1]))

    def test_equal_edges_raises(self):
        with pytest.raises(ValueError, match="monotonically increasing"):
            Mesh1D(edges=np.array([0.0, 1.0, 1.0]),
                   mat_ids=np.array([0, 1]))

    def test_wrong_mat_ids_length_raises(self):
        with pytest.raises(ValueError, match="len\\(mat_ids\\)"):
            Mesh1D(edges=np.array([0.0, 1.0, 2.0]),
                   mat_ids=np.array([0, 1, 2]))

    def test_too_few_edges_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            Mesh1D(edges=np.array([0.0]), mat_ids=np.array([]))

    def test_coerces_to_float_and_int(self):
        """Accepts lists; coerces edges to float, mat_ids to int."""
        mesh = Mesh1D(edges=[0, 1, 2], mat_ids=[0, 1])
        assert mesh.edges.dtype == float
        assert mesh.mat_ids.dtype == int


# ═══════════════════════════════════════════════════════════════════════
# Mesh2D properties and validation
# ═══════════════════════════════════════════════════════════════════════

class TestMesh2D:
    """Mesh2D derived properties and input validation."""

    def test_dx_dy(self):
        mesh = Mesh2D(
            edges_x=np.array([0.0, 1.0, 3.0]),
            edges_y=np.array([0.0, 0.5]),
            mat_map=np.array([[0], [1]]),
        )
        np.testing.assert_allclose(mesh.dx, [1.0, 2.0])
        np.testing.assert_allclose(mesh.dy, [0.5])

    def test_volumes_cartesian(self):
        mesh = Mesh2D(
            edges_x=np.array([0.0, 2.0]),
            edges_y=np.array([0.0, 3.0]),
            mat_map=np.array([[0]]),
        )
        np.testing.assert_allclose(mesh.volumes, [[6.0]])

    def test_volumes_cylindrical_rz(self):
        mesh = Mesh2D(
            edges_x=np.array([0.0, 1.0]),  # radial
            edges_y=np.array([0.0, 5.0]),   # axial
            mat_map=np.array([[0]]),
            coord=CoordSystem.CYLINDRICAL,
        )
        np.testing.assert_allclose(mesh.volumes, [[np.pi * 5.0]])

    def test_mat_ids_flat(self):
        mat_map = np.array([[0, 1], [2, 3]])
        mesh = Mesh2D(
            edges_x=np.array([0.0, 1.0, 2.0]),
            edges_y=np.array([0.0, 1.0, 2.0]),
            mat_map=mat_map,
        )
        np.testing.assert_array_equal(mesh.mat_ids, [0, 1, 2, 3])

    def test_nx_ny(self):
        mesh = Mesh2D(
            edges_x=np.linspace(0, 1, 4),
            edges_y=np.linspace(0, 1, 6),
            mat_map=np.zeros((3, 5), dtype=int),
        )
        assert mesh.nx == 3
        assert mesh.ny == 5

    def test_frozen(self):
        mesh = Mesh2D(
            edges_x=np.array([0.0, 1.0]),
            edges_y=np.array([0.0, 1.0]),
            mat_map=np.array([[0]]),
        )
        with pytest.raises(AttributeError):
            mesh.edges_x = np.array([0.0, 2.0])

    def test_wrong_mat_map_shape_raises(self):
        with pytest.raises(ValueError, match="mat_map shape"):
            Mesh2D(
                edges_x=np.array([0.0, 1.0, 2.0]),
                edges_y=np.array([0.0, 1.0]),
                mat_map=np.array([[0, 1]]),  # should be (2, 1)
            )

    def test_spherical_2d_raises(self):
        with pytest.raises(ValueError, match="CARTESIAN or CYLINDRICAL"):
            Mesh2D(
                edges_x=np.array([0.0, 1.0]),
                edges_y=np.array([0.0, 1.0]),
                mat_map=np.array([[0]]),
                coord=CoordSystem.SPHERICAL,
            )


# ═══════════════════════════════════════════════════════════════════════
# Factory edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestFactoryEdgeCases:
    """Edge cases for mesh1d_from_zones and PWR factories."""

    def test_empty_zones_raises(self):
        with pytest.raises(ValueError, match="At least one zone"):
            mesh1d_from_zones([])

    def test_single_cell_zone(self):
        mesh = mesh1d_from_zones(
            [Zone(outer_edge=1.0, mat_id=0, n_cells=1)],
            coord=CoordSystem.CARTESIAN,
        )
        assert mesh.N == 1
        np.testing.assert_allclose(mesh.edges, [0.0, 1.0])

    def test_custom_origin(self):
        mesh = mesh1d_from_zones(
            [Zone(outer_edge=5.0, mat_id=0, n_cells=4)],
            coord=CoordSystem.CARTESIAN,
            origin=1.0,
        )
        np.testing.assert_allclose(mesh.edges[0], 1.0)
        np.testing.assert_allclose(mesh.edges[-1], 5.0)

    def test_pwr_slab_custom_params(self):
        mesh = pwr_slab_half_cell(n_fuel=5, n_clad=2, n_cool=3,
                                  fuel_half=1.0, clad_thick=0.1, cool_thick=0.5)
        assert mesh.N == 10
        np.testing.assert_allclose(mesh.total_width, 1.6)

    def test_pin_2d_wrong_mat_ids_length_raises(self):
        with pytest.raises(ValueError, match="len\\(mat_ids\\)"):
            pwr_pin_2d(radii=[1.0], mat_ids=[0, 1, 2])

    def test_homogeneous_1d_basic(self):
        mesh = homogeneous_1d(10, 5.0, mat_id=3)
        assert mesh.N == 10
        np.testing.assert_allclose(mesh.total_width, 5.0)
        assert np.all(mesh.mat_ids == 3)
        np.testing.assert_allclose(mesh.widths, 0.5, rtol=1e-14)

    def test_homogeneous_1d_cylindrical(self):
        mesh = homogeneous_1d(5, 2.0, coord=CoordSystem.CYLINDRICAL)
        # Equal-volume annuli
        np.testing.assert_allclose(mesh.volumes, mesh.volumes[0], rtol=1e-14)

    def test_slab_fuel_moderator(self):
        mesh = slab_fuel_moderator(n_fuel=10, n_mod=10, t_fuel=0.5, t_mod=0.5)
        assert mesh.N == 20
        assert np.all(mesh.mat_ids[:10] == 2)   # fuel
        assert np.all(mesh.mat_ids[10:] == 0)    # moderator
        np.testing.assert_allclose(mesh.total_width, 1.0)
        np.testing.assert_allclose(mesh.widths, 0.05, rtol=1e-14)
