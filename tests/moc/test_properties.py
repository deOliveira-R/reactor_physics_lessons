"""Property tests for the Method of Characteristics solver."""

import numpy as np
import pytest

from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.derivations import get
from orpheus.moc.solver import solve_moc

pytestmark = pytest.mark.l0  # MOC property invariants (positivity, balance, symmetry)


def _homogeneous_result():
    """Run a 1G homogeneous MOC calculation for property tests."""
    case = get("moc_cyl1D_1eg_1rg")
    mix = next(iter(case.materials.values()))
    r_cell = 3.6 / np.sqrt(np.pi)
    mesh = Mesh1D(
        edges=np.array([0.0, r_cell]),
        mat_ids=np.array([0]),
        coord=CoordSystem.CYLINDRICAL,
    )
    return solve_moc(
        {0: mix}, mesh,
        n_azi=8, n_polar=3, ray_spacing=0.05,
        max_outer=100,
    ), case


def test_particle_balance():
    """Production / absorption = keff for homogeneous (no leakage)."""
    result, case = _homogeneous_result()
    mix = case.materials[0]

    phi = result.scalar_flux
    areas = result.moc_mesh.region_areas

    p_rate = 0.0
    a_rate = 0.0
    for i in range(result.moc_mesh.n_regions):
        p_rate += mix.SigP @ phi[i, :] * areas[i]
        a_rate += mix.absorption_xs @ phi[i, :] * areas[i]

    assert p_rate / a_rate == pytest.approx(result.keff, rel=1e-4)


def test_flux_positivity():
    """Scalar flux must be non-negative everywhere."""
    result, _ = _homogeneous_result()
    assert np.all(result.scalar_flux >= 0)


def test_flux_per_material_matches_scalar():
    """Volume-averaged flux per material must be consistent with scalar_flux."""
    result, _ = _homogeneous_result()
    phi = result.scalar_flux
    areas = result.moc_mesh.region_areas
    total_area = areas.sum()
    vol_avg = (phi * areas[:, None]).sum(axis=0) / total_area

    np.testing.assert_allclose(
        result.flux_per_material[0], vol_avg, rtol=1e-10
    )


def test_heterogeneous_flux_depression():
    """In a fuel+coolant pin, thermal flux should be higher in coolant."""
    from orpheus.derivations._xs_library import get_mixture
    fuel = get_mixture("A", "2g")
    cool = get_mixture("B", "2g")

    r_fuel = 0.5
    pitch = 2.0
    ws_r = pitch / np.sqrt(np.pi)
    mesh = Mesh1D(
        edges=np.array([0.0, r_fuel, ws_r]),
        mat_ids=np.array([2, 0]),
        coord=CoordSystem.CYLINDRICAL,
    )

    result = solve_moc(
        {2: fuel, 0: cool}, mesh,
        n_azi=16, n_polar=3, ray_spacing=0.03,
        max_outer=200, n_inner_sweeps=15,
    )

    # Thermal group (last) flux should be higher in coolant than fuel
    phi_fuel = result.flux_per_material[2]
    phi_cool = result.flux_per_material[0]
    assert phi_cool[-1] > phi_fuel[-1], (
        f"Thermal flux: fuel={phi_fuel[-1]:.4e}, cool={phi_cool[-1]:.4e}"
    )
