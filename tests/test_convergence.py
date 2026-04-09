"""Cross-solver consistency: SN vs CP reference values."""

import pytest

from orpheus.derivations import get
from orpheus.derivations._xs_library import get_mixture
from orpheus.geometry import slab_fuel_moderator
from orpheus.sn.quadrature import GaussLegendre1D
from orpheus.sn.solver import solve_sn


@pytest.mark.slow
def test_sn_approaches_cp_reference():
    """Fine-mesh SN 1D should approach the CP eigenvalue for slab geometry.

    The gap is due to the white-BC approximation in CP (~1% for moderate
    optical thicknesses). The SN value with reflective BCs is more accurate.
    """
    cp_ref = get("cp_slab_1eg_2rg")

    fuel = get_mixture("A", "1g")
    mod = get_mixture("B", "1g")
    materials = {2: fuel, 0: mod}

    mesh = slab_fuel_moderator(
        n_fuel=40, n_mod=40, t_fuel=0.5, t_mod=0.5,
    )
    quad = GaussLegendre1D.create(32)
    result = solve_sn(
        materials, mesh, quad,
        max_outer=300, max_inner=500, inner_tol=1e-10,
    )

    gap = abs(result.keff - cp_ref.k_inf)
    assert gap < 0.02, (
        f"SN-CP gap too large: {gap:.4f} "
        f"(SN={result.keff:.8f}, CP={cp_ref.k_inf:.8f})"
    )
