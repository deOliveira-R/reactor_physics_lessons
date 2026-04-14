"""Unit tests for collision probability matrix properties.

Tests algebraic properties that the CP matrices must satisfy,
independent of any eigenvalue solver:
- Row sums = 1 (neutron conservation)
- Reciprocity: Σ_t V P_ij = Σ_t V P_ji
- Non-negativity: P ≥ 0
- 1-region limit reproduces homogeneous k

Tests cover all three coordinate systems: Cartesian (slab),
Cylindrical, and Spherical.
"""

import numpy as np
import pytest

from orpheus.geometry import CoordSystem, Mesh1D, Zone, mesh1d_from_zones
from orpheus.cp.solver import CPMesh
from orpheus.derivations._xs_library import get_xs

# CP matrix algebraic invariants (row sums, reciprocity, non-negativity).
# The complementarity and reciprocity labels are named in the theory
# page (docs/theory/collision_probability.rst) as verified by the
# two specific tests in this file — the decorators below make that
# cross-reference load-bearing for the V&V harness audit.
pytestmark = [pytest.mark.l0, pytest.mark.verifies(
    "complementarity",  # test_row_sums: sum_j P_ij + P_out = 1
    "reciprocity",      # test_reciprocity: Sigma_t V P_ij = Sigma_t V P_ji
)]


# ── Fixtures ──────────────────────────────────────────────────────────

def _build_pinf_1g(coord: CoordSystem, r_inner: float = 0.0, r_outer: float = 1.0):
    """Build P_inf for a 1G 2-region problem in any coordinate system.

    Returns (P_inf, sig_t, volumes).
    """
    xs_a = get_xs("A", "1g")
    xs_b = get_xs("B", "1g")
    sig_t_g = np.array([xs_a["sig_t"][0], xs_b["sig_t"][0]])

    mid = 0.5 * (r_inner + r_outer) if coord == CoordSystem.CARTESIAN else None
    if coord == CoordSystem.CARTESIAN:
        mesh = mesh1d_from_zones([
            Zone(outer_edge=0.5, mat_id=0, n_cells=1),
            Zone(outer_edge=1.0, mat_id=1, n_cells=1),
        ], coord=coord)
    else:
        mesh = mesh1d_from_zones([
            Zone(outer_edge=0.5, mat_id=0, n_cells=1),
            Zone(outer_edge=1.0, mat_id=1, n_cells=1),
        ], coord=coord)

    cp_mesh = CPMesh(mesh)
    P_inf = cp_mesh.compute_pinf_group(sig_t_g)
    return P_inf, sig_t_g, mesh.volumes


def _build_pinf_1region(coord: CoordSystem):
    """Build P_inf for a 1G 1-region problem (homogeneous limit)."""
    if coord == CoordSystem.CARTESIAN:
        mesh = mesh1d_from_zones([
            Zone(outer_edge=0.5, mat_id=0, n_cells=1),
        ], coord=coord)
    else:
        mesh = mesh1d_from_zones([
            Zone(outer_edge=1.0, mat_id=0, n_cells=1),
        ], coord=coord)

    sig_t_g = np.array([1.0])
    cp_mesh = CPMesh(mesh)
    return cp_mesh.compute_pinf_group(sig_t_g)


# ── Row sums (neutron conservation) ──────────────────────────────────

@pytest.mark.parametrize("coord", [
    CoordSystem.CARTESIAN,
    CoordSystem.CYLINDRICAL,
    CoordSystem.SPHERICAL,
])
def test_row_sums(coord):
    """P_inf row sums must equal 1 (every neutron collides somewhere)."""
    P_inf, _, _ = _build_pinf_1g(coord)
    row_sums = P_inf.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


# ── Reciprocity ──────────────────────────────────────────────────────

@pytest.mark.parametrize("coord", [
    CoordSystem.CARTESIAN,
    CoordSystem.CYLINDRICAL,
    CoordSystem.SPHERICAL,
])
def test_reciprocity(coord):
    """Σ_t[i] V[i] P[i,j] = Σ_t[j] V[j] P[j,i] for all i≠j."""
    P_inf, sig_t, V = _build_pinf_1g(coord)
    N = P_inf.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            lhs = sig_t[i] * V[i] * P_inf[i, j]
            rhs = sig_t[j] * V[j] * P_inf[j, i]
            assert abs(lhs - rhs) < 1e-10, (
                f"Reciprocity violated for {coord.value}: "
                f"Σ_t[{i}]V[{i}]P[{i},{j}]={lhs:.6e} "
                f"≠ Σ_t[{j}]V[{j}]P[{j},{i}]={rhs:.6e}"
            )


# ── Non-negativity ───────────────────────────────────────────────────

@pytest.mark.parametrize("coord", [
    CoordSystem.CARTESIAN,
    CoordSystem.CYLINDRICAL,
    CoordSystem.SPHERICAL,
])
def test_non_negativity(coord):
    """All collision probabilities must be non-negative."""
    P_inf, _, _ = _build_pinf_1g(coord)
    assert np.all(P_inf >= 0), (
        f"Negative P_inf entry for {coord.value}: min={P_inf.min():.6e}"
    )


# ── 1-region homogeneous limit ───────────────────────────────────────

@pytest.mark.parametrize("coord", [
    CoordSystem.CARTESIAN,
    CoordSystem.CYLINDRICAL,
    CoordSystem.SPHERICAL,
])
def test_1region_homogeneous_limit(coord):
    """1-region CP must give P=1 (all neutrons collide in the only region)."""
    P_inf = _build_pinf_1region(coord)
    np.testing.assert_allclose(P_inf[0, 0], 1.0, atol=1e-10)
