"""Verify the infinite-medium eigenvalue solver against SymPy analytical solutions."""

import pytest

from orpheus.derivations import get
from orpheus.homogeneous.solver import solve_homogeneous_infinite

# File-level verifies marker: every test in this file exercises the
# homogeneous eigenvalue, which is one-group-kinf for 1G and
# matrix-eigenvalue / mg-balance for 2G / 4G. Declared explicitly here
# (rather than inherited from VerificationCase) so the Nexus AST pass
# picks it up via decorator parsing and writes TESTS edges.
pytestmark = pytest.mark.verifies(
    "one-group-kinf",
    "inf-hom-balance",
    "matrix-eigenvalue",
    "removal-matrix",
    "fission-matrix",
    "mg-balance",
)


@pytest.mark.parametrize("case_name", [
    "homo_1eg",
    "homo_2eg",
    "homo_4eg",
])
def test_kinf_exact(case_name):
    """Eigenvalue must match analytical solution to machine precision."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    result = solve_homogeneous_infinite(mix)
    assert abs(result.k_inf - case.k_inf) < 1e-12, (
        f"k_inf mismatch: solver={result.k_inf:.10f} "
        f"analytical={case.k_inf:.10f}"
    )
