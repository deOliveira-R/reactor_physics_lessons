"""Verify the slab collision probability solver against analytical CP eigenvalues."""

import numpy as np
import pytest

from derivations import get
from geometry import CoordSystem, Mesh1D
from collision_probability import solve_cp


@pytest.mark.parametrize("case_name", [
    "cp_slab_1eg_1rg",
    "cp_slab_1eg_2rg",
    "cp_slab_1eg_4rg",
    "cp_slab_2eg_1rg",
    "cp_slab_2eg_2rg",
    "cp_slab_2eg_4rg",
    "cp_slab_4eg_1rg",
    "cp_slab_4eg_2rg",
    "cp_slab_4eg_4rg",
])
def test_slab_cp_eigenvalue(case_name):
    """Slab CP solver must match the analytical CP eigenvalue."""
    case = get(case_name)
    gp = case.geom_params
    thicknesses = np.array(gp["thicknesses"])
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    mesh = Mesh1D(
        edges=edges,
        mat_ids=np.array(gp["mat_ids"]),
        coord=CoordSystem.CARTESIAN,
    )
    result = solve_cp(case.materials, mesh,
                      params=__import__('collision_probability').CPParams(
                          keff_tol=1e-7, flux_tol=1e-6))

    err = abs(result.keff - case.k_inf)
    assert err < 1e-6, (
        f"{case_name}: solver={result.keff:.10f} "
        f"analytical={case.k_inf:.10f} err={err:.2e}"
    )
