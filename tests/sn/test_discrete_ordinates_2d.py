"""Verify the 2D SN solver (Lebedev quadrature, mesh convergence)."""

import numpy as np
import pytest

from orpheus.derivations._xs_library import get_mixture
from orpheus.geometry import Mesh2D
from orpheus.sn.quadrature import LebedevSphere
from orpheus.sn.solver import solve_sn

# 2D SN mesh convergence — L2 integration check.
pytestmark = [
    pytest.mark.l2,
    pytest.mark.verifies(
        "transport-cartesian-2d",
        "dd-cartesian-2d",
        "multigroup",
    ),
]


@pytest.mark.slow
@pytest.mark.catches("ERR-003")
@pytest.mark.parametrize("ng_key,label", [("1g", "1G"), ("2g", "2G")])
def test_do_mesh_convergence(ng_key, label):
    """2D SN solver must converge with mesh refinement.

    This test catches ERR-003 (octant batching breaking reflective-BC
    ordering on multigroup heterogeneous problems). The original
    catcher was the now-deleted
    ``tests.sn.test_cartesian.test_heterogeneous_convergence``,
    which consumed the T3 Richardson references from
    ``orpheus.derivations.sn._derive_sn_heterogeneous`` — those
    references were deleted in Phase 2.1a of the verification
    campaign. The catcher marker is reassigned here because the
    2G heterogeneous fuel+moderator 2D run with default reflective
    BCs exercises exactly the same ordinate-ordering coupling that
    produced the original bug (see the Phase 2.1a commit for full
    context, and ``tests/l0_error_catalog.md`` ERR-003 for the
    failure-mode description).
    """
    fuel = get_mixture("A", ng_key)
    mod = get_mixture("B", ng_key)
    materials = {2: fuel, 0: mod}

    quad = LebedevSphere.create(order=17)

    keffs = []
    deltas = [0.1, 0.05, 0.02]
    for delta in deltas:
        n_fuel = max(2, round(0.5 / delta))
        n_mod = max(2, round(0.5 / delta))
        nx = n_fuel + n_mod
        ny = 2

        mat = np.zeros((nx, ny), dtype=int)
        mat[:n_fuel, :] = 2
        mat[n_fuel:, :] = 0

        mesh = Mesh2D(
            edges_x=np.linspace(0, nx * delta, nx + 1),
            edges_y=np.linspace(0, ny * delta, ny + 1),
            mat_map=mat,
        )
        result = solve_sn(
            materials, mesh, quad,
            inner_solver="source_iteration",
            max_outer=300, inner_tol=1e-6,
        )
        keffs.append(result.keff)

    diffs = [abs(keffs[i] - keffs[i + 1]) for i in range(len(keffs) - 1)]
    assert diffs[-1] < diffs[0], (
        f"SN not converging: diffs={diffs}"
    )
