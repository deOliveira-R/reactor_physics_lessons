"""Tests for MOC ray tracing geometry (MOCMesh, tracks, segments)."""

import numpy as np
import pytest

from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.moc.geometry import (
    MOCMesh,
    Track,
    _identify_region,
    _ray_circle_intersections,
    _trace_single_ray,
)
from orpheus.moc.quadrature import MOCQuadrature

pytestmark = pytest.mark.l0  # MOC ray-tracing geometry primitives (MOCMesh, Track, ...)


# ── Helpers ──────────────────────────────────────────────────────────

def _homogeneous_mesh(r_outer: float = 1.0, mat_id: int = 0) -> Mesh1D:
    """Single-region cylindrical mesh (Wigner-Seitz)."""
    return Mesh1D(
        edges=np.array([0.0, r_outer]),
        mat_ids=np.array([mat_id]),
        coord=CoordSystem.CYLINDRICAL,
    )


def _two_region_mesh(r_fuel: float = 0.5, r_cell: float = 1.0) -> Mesh1D:
    """Two-region cylindrical mesh: fuel (id=2) + coolant (id=0)."""
    return Mesh1D(
        edges=np.array([0.0, r_fuel, r_cell]),
        mat_ids=np.array([2, 0]),
        coord=CoordSystem.CYLINDRICAL,
    )


def _three_region_mesh(
    r_fuel: float = 0.4,
    r_clad: float = 0.5,
    r_cell: float = 1.0,
) -> Mesh1D:
    """Three-region mesh: fuel (2) + clad (1) + coolant (0)."""
    return Mesh1D(
        edges=np.array([0.0, r_fuel, r_clad, r_cell]),
        mat_ids=np.array([2, 1, 0]),
        coord=CoordSystem.CYLINDRICAL,
    )


# ── Ray-circle intersection ─────────────────────────────────────────

def test_ray_circle_hit():
    """Horizontal ray through center of a circle: chord = 2R."""
    hits = _ray_circle_intersections(0, 0, 1, 0, 0, 0, 1.0)
    assert len(hits) == 2
    assert hits[0] == pytest.approx(-1.0)
    assert hits[1] == pytest.approx(1.0)


def test_ray_circle_miss():
    """Ray that misses the circle returns empty list."""
    hits = _ray_circle_intersections(0, 5.0, 1, 0, 0, 0, 1.0)
    assert len(hits) == 0


def test_ray_circle_tangent():
    """Ray tangent to circle returns two coincident values."""
    hits = _ray_circle_intersections(0, 1.0, 1, 0, 0, 0, 1.0)
    assert len(hits) == 2
    assert abs(hits[1] - hits[0]) < 1e-10


def test_ray_circle_chord_length():
    """Chord length = 2*sqrt(R^2 - d^2) for impact parameter d."""
    R = 2.0
    d = 1.0  # perpendicular distance from center
    # Ray at y=d, horizontal, circle at origin
    hits = _ray_circle_intersections(0, d, 1, 0, 0, 0, R)
    chord = hits[1] - hits[0]
    expected = 2 * np.sqrt(R**2 - d**2)
    assert chord == pytest.approx(expected, rel=1e-10)


# ── Region identification ───────────────────────────────────────────

def test_identify_region_center():
    """Point at pin center belongs to region 0."""
    radii = np.array([0.5, 1.0])
    assert _identify_region(1.0, 1.0, 1.0, 1.0, radii, 3) == 0


def test_identify_region_outer():
    """Point far from center belongs to outermost region."""
    radii = np.array([0.5])
    assert _identify_region(0.0, 0.0, 1.0, 1.0, radii, 2) == 1


def test_identify_region_boundary():
    """Point on circular boundary belongs to the inner region."""
    radii = np.array([0.5, 1.0])
    # Point at distance exactly 0.5 from center
    assert _identify_region(1.5, 1.0, 1.0, 1.0, radii, 3) == 0


# ── Single-ray tracing ──────────────────────────────────────────────

def test_trace_empty_cell():
    """Ray through a single-region cell gives one segment."""
    mesh = _homogeneous_mesh(r_outer=1.0)
    pitch = 1.0 * np.sqrt(np.pi)
    radii = np.array([], dtype=float)  # no inner circles

    segments, entry, exit_pt, entry_surf, exit_surf = _trace_single_ray(
        x0=0, y0=pitch / 2, cos_phi=1.0, sin_phi=0.0,
        pitch=pitch, cx=pitch / 2, cy=pitch / 2,
        radii=radii, n_regions=1,
    )
    total_length = sum(s.length for s in segments)
    assert total_length == pytest.approx(pitch, rel=1e-10)
    assert all(s.region_id == 0 for s in segments)


def test_trace_horizontal_through_circle():
    """Horizontal ray through center of a 2-region pin cell."""
    mesh = _two_region_mesh(r_fuel=0.5, r_cell=1.0)
    pitch = 1.0 * np.sqrt(np.pi)
    cx = pitch / 2
    radii = np.array([0.5])

    segments, _, _, _, _ = _trace_single_ray(
        x0=0, y0=cx, cos_phi=1.0, sin_phi=0.0,
        pitch=pitch, cx=cx, cy=cx,
        radii=radii, n_regions=2,
    )

    # Should have: coolant | fuel | coolant (3 segments)
    assert len(segments) == 3
    assert segments[0].region_id == 1  # coolant (outside circle)
    assert segments[1].region_id == 0  # fuel (inside circle)
    assert segments[2].region_id == 1  # coolant

    # Fuel chord = 2 * R_fuel = 1.0
    assert segments[1].length == pytest.approx(1.0, rel=1e-6)

    # Total = pitch
    total = sum(s.length for s in segments)
    assert total == pytest.approx(pitch, rel=1e-10)


def test_trace_misses_inner_circle():
    """Ray that misses the inner circle: only outermost region."""
    mesh = _two_region_mesh(r_fuel=0.3, r_cell=1.0)
    pitch = 1.0 * np.sqrt(np.pi)
    cx = pitch / 2
    radii = np.array([0.3])

    # Ray at y very close to boundary (far from center)
    segments, _, _, _, _ = _trace_single_ray(
        x0=0, y0=0.01, cos_phi=1.0, sin_phi=0.0,
        pitch=pitch, cx=cx, cy=cx,
        radii=radii, n_regions=2,
    )

    # All segments should be in the outer region
    assert all(s.region_id == 1 for s in segments)


def test_trace_three_regions():
    """Ray through 3 concentric regions gives up to 5 segments."""
    mesh = _three_region_mesh(r_fuel=0.4, r_clad=0.5, r_cell=1.0)
    pitch = 1.0 * np.sqrt(np.pi)
    cx = pitch / 2
    radii = np.array([0.4, 0.5])

    # Horizontal ray through center
    segments, _, _, _, _ = _trace_single_ray(
        x0=0, y0=cx, cos_phi=1.0, sin_phi=0.0,
        pitch=pitch, cx=cx, cy=cx,
        radii=radii, n_regions=3,
    )

    # Expected: coolant | clad | fuel | clad | coolant = 5 segments
    assert len(segments) == 5
    region_ids = [s.region_id for s in segments]
    assert region_ids == [2, 1, 0, 1, 2]


# ── MOCMesh construction ────────────────────────────────────────────

def test_moc_mesh_pitch_recovery():
    """Pitch is correctly recovered from Wigner-Seitz radius."""
    r_cell = 3.6 / np.sqrt(np.pi)
    mesh = _homogeneous_mesh(r_outer=r_cell)
    quad = MOCQuadrature.create(n_azi=4, n_polar=1)
    moc = MOCMesh(mesh, quad, ray_spacing=0.1)
    assert moc.pitch == pytest.approx(3.6, rel=1e-10)


def test_moc_mesh_region_areas_homogeneous():
    """Single-region: area = pitch^2."""
    r_cell = 2.0 / np.sqrt(np.pi)
    mesh = _homogeneous_mesh(r_outer=r_cell)
    quad = MOCQuadrature.create(n_azi=4, n_polar=1)
    moc = MOCMesh(mesh, quad, ray_spacing=0.1)
    assert moc.region_areas[0] == pytest.approx(2.0**2, rel=1e-10)


def test_moc_mesh_region_areas_two_region():
    """Two-region areas: inner circle + square border."""
    r_fuel = 0.5
    r_cell = 2.0 / np.sqrt(np.pi)
    mesh = _two_region_mesh(r_fuel=r_fuel, r_cell=r_cell)
    quad = MOCQuadrature.create(n_azi=4, n_polar=1)
    moc = MOCMesh(mesh, quad, ray_spacing=0.1)

    assert moc.region_areas[0] == pytest.approx(np.pi * 0.5**2, rel=1e-10)
    assert moc.region_areas[1] == pytest.approx(2.0**2 - np.pi * 0.5**2, rel=1e-10)
    assert moc.region_areas.sum() == pytest.approx(2.0**2, rel=1e-10)


def test_moc_mesh_total_area():
    """Sum of all region areas = pitch^2."""
    r_cell = 3.6 / np.sqrt(np.pi)
    mesh = _three_region_mesh(r_fuel=0.4, r_clad=0.5, r_cell=r_cell)
    quad = MOCQuadrature.create(n_azi=8, n_polar=1)
    moc = MOCMesh(mesh, quad, ray_spacing=0.05)
    assert moc.region_areas.sum() == pytest.approx(3.6**2, rel=1e-10)


def test_moc_mesh_has_tracks():
    """MOCMesh generates a non-empty set of tracks."""
    r_cell = 2.0 / np.sqrt(np.pi)
    mesh = _two_region_mesh(r_fuel=0.5, r_cell=r_cell)
    quad = MOCQuadrature.create(n_azi=8, n_polar=2)
    moc = MOCMesh(mesh, quad, ray_spacing=0.1)

    assert len(moc.tracks) > 0
    assert len(moc.tracks_per_azi) == 8
    for indices in moc.tracks_per_azi:
        assert len(indices) > 0


def test_moc_mesh_mat_ids():
    """Material IDs come from the Mesh1D."""
    mesh = _three_region_mesh()
    quad = MOCQuadrature.create(n_azi=4, n_polar=1)
    moc = MOCMesh(mesh, quad, ray_spacing=0.1)
    np.testing.assert_array_equal(moc.region_mat_ids, [2, 1, 0])


def test_moc_mesh_track_links_valid():
    """All track links point to valid track indices."""
    r_cell = 2.0 / np.sqrt(np.pi)
    mesh = _two_region_mesh(r_fuel=0.5, r_cell=r_cell)
    quad = MOCQuadrature.create(n_azi=8, n_polar=1)
    moc = MOCMesh(mesh, quad, ray_spacing=0.1)

    n_tracks = len(moc.tracks)
    for t in moc.tracks:
        assert 0 <= t.fwd_link < n_tracks
        assert 0 <= t.bwd_link < n_tracks


# ── Volume conservation ──────────────────────────────────────────────

def test_volume_conservation():
    """Sum of (segment_length * ray_spacing) approximates region area.

    This is the fundamental self-consistency check for ray tracing:
    the track-estimated area should converge to the geometric area
    as ray spacing decreases.
    """
    r_fuel = 0.5
    r_cell = 2.0 / np.sqrt(np.pi)
    mesh = _two_region_mesh(r_fuel=r_fuel, r_cell=r_cell)
    quad = MOCQuadrature.create(n_azi=16, n_polar=1)
    moc = MOCMesh(mesh, quad, ray_spacing=0.02)

    # Estimate area per region from tracks
    estimated_areas = np.zeros(moc.n_regions)
    for a_idx in range(moc.quad.n_azi):
        ts = moc.effective_spacing(a_idx)
        omega_a = moc.quad.omega_azi[a_idx]
        for t_idx in moc.tracks_per_azi[a_idx]:
            track = moc.tracks[t_idx]
            for seg in track.segments:
                # Each azimuthal angle covers omega_a fraction of the plane
                # The ray contributes ts * length to the area
                estimated_areas[seg.region_id] += seg.length * ts * omega_a

    # Should approximate geometric areas (within a few percent for
    # ray_spacing=0.02, n_azi=16)
    for k in range(moc.n_regions):
        rel_err = abs(estimated_areas[k] - moc.region_areas[k]) / moc.region_areas[k]
        assert rel_err < 0.05, (
            f"Region {k}: estimated={estimated_areas[k]:.4f}, "
            f"exact={moc.region_areas[k]:.4f}, rel_err={rel_err:.4f}"
        )


def test_track_total_length_equals_cell_diagonal_or_less():
    """Every track's total segment length <= pitch * sqrt(2) (cell diagonal)."""
    r_cell = 2.0 / np.sqrt(np.pi)
    mesh = _homogeneous_mesh(r_outer=r_cell)
    quad = MOCQuadrature.create(n_azi=8, n_polar=1)
    moc = MOCMesh(mesh, quad, ray_spacing=0.1)

    max_len = moc.pitch * np.sqrt(2) + 1e-10
    for track in moc.tracks:
        total = sum(s.length for s in track.segments)
        assert total <= max_len
