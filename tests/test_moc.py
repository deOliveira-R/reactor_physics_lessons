"""Verify the Method of Characteristics solver against analytical/Richardson references."""

import numpy as np
import pytest

from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.derivations import get
from orpheus.moc.solver import solve_moc


def _build_homogeneous_mesh(mix):
    """Build a single-region Wigner-Seitz mesh for homogeneous tests."""
    r_cell = 3.6 / np.sqrt(np.pi)
    return Mesh1D(
        edges=np.array([0.0, r_cell]),
        mat_ids=np.array([0]),
        coord=CoordSystem.CYLINDRICAL,
    )


@pytest.mark.parametrize("case_name", [
    "moc_cyl1D_1eg_1rg",
    "moc_cyl1D_2eg_1rg",
    "moc_cyl1D_4eg_1rg",
])
def test_moc_homogeneous(case_name):
    """MOC with homogeneous fill must match infinite-medium eigenvalue."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    mesh = _build_homogeneous_mesh(mix)

    result = solve_moc(
        {0: mix}, mesh,
        n_azi=8, n_polar=3, ray_spacing=0.05,
        max_outer=200,
    )

    err = abs(result.keff - case.k_inf)
    assert err < 1e-4, (
        f"{case_name}: solver={result.keff:.6f} "
        f"analytical={case.k_inf:.6f} err={err:.2e}"
    )


def _build_heterogeneous_mesh(radii, mat_ids):
    """Build a Wigner-Seitz mesh for heterogeneous tests."""
    pitch = 2.0 * radii[-1]
    ws_r = pitch / np.sqrt(np.pi)
    edges = [0.0] + list(radii[:-1]) + [ws_r]
    return Mesh1D(
        edges=np.array(edges),
        mat_ids=np.array(mat_ids),
        coord=CoordSystem.CYLINDRICAL,
    )


@pytest.mark.slow
@pytest.mark.parametrize("case_name", [
    "moc_cyl1D_1eg_2rg",
    "moc_cyl1D_2eg_2rg",
    "moc_cyl1D_1eg_4rg",
])
def test_moc_heterogeneous(case_name):
    """MOC heterogeneous must converge near the Richardson reference."""
    case = get(case_name)
    gp = case.geom_params
    mesh = _build_heterogeneous_mesh(gp["radii"], gp["mat_ids"])

    result = solve_moc(
        case.materials, mesh,
        n_azi=16, n_polar=3, ray_spacing=0.03,
        max_outer=300, n_inner_sweeps=20,
    )

    err = abs(result.keff - case.k_inf)
    assert err < 5e-2, (
        f"{case_name}: solver={result.keff:.6f} "
        f"ref={case.k_inf:.6f} err={err:.2e}"
    )
