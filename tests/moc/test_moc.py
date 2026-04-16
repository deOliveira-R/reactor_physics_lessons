"""Verify the Method of Characteristics solver against analytical references."""

import numpy as np
import pytest

from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.derivations import get
from orpheus.moc.solver import solve_moc

pytestmark = pytest.mark.verifies(
    "characteristic-ode",
    "bar-psi",
    "isotropic-source",
    "moc-wigner-seitz",
    "delta-psi",
    "moc-keff-update",
    "boyd-eq-45",
    "one-group-kinf",
    "matrix-eigenvalue",
    "mg-balance",
)


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


