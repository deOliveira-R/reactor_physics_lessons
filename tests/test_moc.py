"""Verify the Method of Characteristics solver against analytical/Richardson references."""

import numpy as np
import pytest

from derivations import get
from method_of_characteristics import MoCGeometry, solve_moc


@pytest.mark.parametrize("case_name", [
    "moc_cyl1D_1eg_1rg",
    "moc_cyl1D_2eg_1rg",
    "moc_cyl1D_4eg_1rg",
])
def test_moc_homogeneous(case_name):
    """MOC with homogeneous fill must match infinite-medium eigenvalue."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    materials = {0: mix, 1: mix, 2: mix}
    geom = MoCGeometry.default_pwr()
    result = solve_moc(materials, geom, max_outer=200)

    err = abs(result.keff - case.k_inf)
    assert err < 1e-4, (
        f"{case_name}: solver={result.keff:.6f} "
        f"analytical={case.k_inf:.6f} err={err:.2e}"
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
    r_cell = gp["radii"][-1]
    pitch = r_cell * np.sqrt(np.pi) * 2

    geom = MoCGeometry.from_annular(
        gp["radii"], gp["mat_ids"], pitch=pitch, n_cells=12,
    )
    result = solve_moc(case.materials, geom, max_outer=300)

    err = abs(result.keff - case.k_inf)
    assert err < 5e-2, (
        f"{case_name}: solver={result.keff:.6f} "
        f"ref={case.k_inf:.6f} err={err:.2e}"
    )
