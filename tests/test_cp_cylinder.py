"""Verify the cylindrical collision probability solver against analytical CP eigenvalues."""

import numpy as np
import pytest

from orpheus.derivations import get
from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.cp.solver import solve_cp

pytestmark = pytest.mark.verifies(
    "collision-rate",
    "ki3-def",
    "chord-length",
    "self-cyl",
    "second-diff-cyl",
    "wigner-seitz",
    "one-group-kinf",
    "matrix-eigenvalue",
    "mg-balance",
)


@pytest.mark.parametrize("case_name", [
    "cp_cyl1D_1eg_1rg",
    "cp_cyl1D_1eg_2rg",
    "cp_cyl1D_1eg_4rg",
    "cp_cyl1D_2eg_1rg",
    "cp_cyl1D_2eg_2rg",
    "cp_cyl1D_2eg_4rg",
    "cp_cyl1D_4eg_1rg",
    "cp_cyl1D_4eg_2rg",
    "cp_cyl1D_4eg_4rg",
])
def test_cylinder_cp_eigenvalue(case_name):
    """Cylindrical CP solver must match the analytical CP eigenvalue."""
    case = get(case_name)
    gp = case.geom_params
    radii = np.array(gp["radii"])
    edges = np.concatenate([[0.0], radii])
    mesh = Mesh1D(
        edges=edges,
        mat_ids=np.array(gp["mat_ids"]),
        coord=CoordSystem.CYLINDRICAL,
    )
    result = solve_cp(case.materials, mesh)

    err = abs(result.keff - case.k_inf)
    assert err < 1e-5, (
        f"{case_name}: solver={result.keff:.10f} "
        f"analytical={case.k_inf:.10f} err={err:.2e}"
    )
