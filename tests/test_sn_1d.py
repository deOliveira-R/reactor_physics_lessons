"""Verify the 1D SN solver: homogeneous exact, spatial O(h²), angular spectral."""

import numpy as np
import pytest

from orpheus.derivations import get
from orpheus.derivations._xs_library import get_mixture
from orpheus.geometry import homogeneous_1d, slab_fuel_moderator, mesh1d_from_zones, Zone, CoordSystem
from orpheus.sn.quadrature import GaussLegendre1D
from orpheus.sn.solver import solve_sn

pytestmark = pytest.mark.verifies(
    "transport-cartesian",
    "dd-cartesian-1d",
    "dd-solve",
    "dd-recurrence",
    "multigroup",
    "reflective-bc",
    "one-group-kinf",
    "matrix-eigenvalue",
    "mg-balance",
)


# ─── Homogeneous infinite medium (SN with reflective BCs) ────────────

@pytest.mark.parametrize("case_name", [
    "sn_slab_1eg_1rg",
    "sn_slab_2eg_1rg",
    "sn_slab_4eg_1rg",
])
def test_homogeneous_exact(case_name):
    """SN 1D with reflective BCs on a homogeneous slab must match
    the analytical infinite-medium eigenvalue."""
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    materials = {0: mix}
    mesh = homogeneous_1d(20, 2.0, mat_id=0)
    quad = GaussLegendre1D.create(8)
    result = solve_sn(materials, mesh, quad,
                      max_inner=500, inner_tol=1e-10)

    assert abs(result.keff - case.k_inf) < 1e-8, (
        f"keff={result.keff:.8f} vs analytical={case.k_inf:.8f}"
    )


# ─── Heterogeneous: convergence to Richardson reference ──────────────

@pytest.mark.slow
@pytest.mark.parametrize("case_name", [
    "sn_slab_1eg_2rg",
    "sn_slab_2eg_2rg",
    "sn_slab_4eg_2rg",
    "sn_slab_1eg_4rg",
    "sn_slab_2eg_4rg",
    "sn_slab_4eg_4rg",
])
def test_heterogeneous_convergence(case_name):
    """SN on heterogeneous slab must converge to the Richardson reference."""
    case = get(case_name)
    gp = case.geom_params
    zones = []
    edge = 0.0
    for t, mid in zip(gp["thicknesses"], gp["mat_ids"]):
        edge += t
        zones.append(Zone(outer_edge=edge, mat_id=mid, n_cells=20))
    mesh = mesh1d_from_zones(zones, coord=CoordSystem.CARTESIAN)
    quad = GaussLegendre1D.create(16)
    result = solve_sn(
        case.materials, mesh, quad,
        max_outer=500, max_inner=500, inner_tol=1e-10, keff_tol=1e-8,
    )

    err = abs(result.keff - case.k_inf)
    assert err < 1e-3, (
        f"{case_name}: keff={result.keff:.8f} vs ref={case.k_inf:.8f} err={err:.2e}"
    )


# ─── Spatial convergence O(h²) ───────────────────────────────────────

def _convergence_order(values, spacings, reference):
    """Compute observed convergence order between successive refinements."""
    orders = []
    for i in range(1, len(values)):
        err_prev = abs(values[i - 1] - reference)
        err_curr = abs(values[i] - reference)
        if err_prev > 0 and err_curr > 0:
            orders.append(
                np.log(err_prev / err_curr)
                / np.log(spacings[i - 1] / spacings[i])
            )
    return orders


@pytest.mark.l1
def test_spatial_convergence():
    """Diamond-difference scheme must show O(h²) spatial convergence."""
    fuel = get_mixture("A", "1g")
    mod = get_mixture("B", "1g")
    materials = {2: fuel, 0: mod}
    t_fuel, t_mod = 0.5, 0.5

    keffs = []
    dxs = []
    for n_per in [5, 10, 20, 40]:
        mesh = slab_fuel_moderator(
            n_fuel=n_per, n_mod=n_per, t_fuel=t_fuel, t_mod=t_mod,
        )
        quad = GaussLegendre1D.create(16)
        result = solve_sn(
            materials, mesh, quad,
            max_outer=300, max_inner=500, inner_tol=1e-10,
        )
        keffs.append(result.keff)
        dxs.append(t_fuel / n_per)

    # Richardson extrapolation reference
    k_ref = keffs[-1] + (keffs[-1] - keffs[-2]) / 3.0
    orders = _convergence_order(keffs, dxs, k_ref)

    assert orders[-1] > 1.7, (
        f"Expected O(h²) convergence, got order {orders[-1]:.2f}"
    )


# ─── Angular spectral convergence ────────────────────────────────────

@pytest.mark.l1
def test_angular_convergence():
    """Gauss-Legendre quadrature must show spectral convergence in angle."""
    fuel = get_mixture("A", "1g")
    mod = get_mixture("B", "1g")
    materials = {2: fuel, 0: mod}

    keffs = []
    n_ords = [4, 8, 16, 32]
    for N in n_ords:
        mesh = slab_fuel_moderator(
            n_fuel=40, n_mod=40, t_fuel=0.5, t_mod=0.5,
        )
        quad = GaussLegendre1D.create(N)
        result = solve_sn(
            materials, mesh, quad,
            max_outer=300, max_inner=500, inner_tol=1e-10,
        )
        keffs.append(result.keff)

    k_ref = keffs[-1]
    orders = _convergence_order(keffs, [1 / N for N in n_ords], k_ref)
    assert len(orders) >= 2
    assert max(orders[:-1]) > 1.5, (
        f"Expected spectral convergence, got orders {orders}"
    )
