"""Verify the 1D diffusion solver: spatial O(h²) convergence."""

import numpy as np
import pytest

from orpheus.derivations import get
from orpheus.diffusion.solver import CoreGeometry, TwoGroupXS, solve_diffusion_1d


def _make_xs(xs_dict) -> TwoGroupXS:
    return TwoGroupXS(**xs_dict)


def test_spatial_convergence_bare():
    """Finite-difference diffusion must show O(h²) spatial convergence (bare slab)."""
    case = get("dif_slab_2eg_1rg")
    fuel_xs = _make_xs(case.materials)

    keffs = []
    dzs = [5.0, 2.5, 1.25, 0.625]
    for dz in dzs:
        geom = CoreGeometry(
            bot_refl_height=0.0, fuel_height=case.geom_params["fuel_height"],
            top_refl_height=0.0, dz=dz,
        )
        result = solve_diffusion_1d(
            geom=geom, reflector_xs=fuel_xs, fuel_xs=fuel_xs,
        )
        keffs.append(result.keff)

    orders = []
    for i in range(1, len(keffs)):
        err_prev = abs(keffs[i - 1] - case.k_inf)
        err_curr = abs(keffs[i] - case.k_inf)
        if err_prev > 0 and err_curr > 0:
            orders.append(
                np.log(err_prev / err_curr) / np.log(dzs[i - 1] / dzs[i])
            )

    assert orders[-1] > 1.8, (
        f"Expected O(h²) convergence, got order {orders[-1]:.2f}"
    )


@pytest.mark.slow
def test_spatial_convergence_reflected():
    """Fuel + reflector must show O(h²) convergence to analytical interface matching."""
    case = get("dif_slab_2eg_2rg")
    fuel_xs = _make_xs(case.materials["fuel"])
    refl_xs = _make_xs(case.materials["reflector"])
    H_f = case.geom_params["fuel_height"]
    H_r = case.geom_params["refl_height"]

    keffs = []
    dzs = [5.0, 2.5, 1.25, 0.625]
    for dz in dzs:
        geom = CoreGeometry(
            bot_refl_height=0.0,
            fuel_height=H_f,
            top_refl_height=H_r,
            dz=dz,
        )
        result = solve_diffusion_1d(
            geom=geom, reflector_xs=refl_xs, fuel_xs=fuel_xs,
        )
        keffs.append(result.keff)

    orders = []
    for i in range(1, len(keffs)):
        err_prev = abs(keffs[i - 1] - case.k_inf)
        err_curr = abs(keffs[i] - case.k_inf)
        if err_prev > 0 and err_curr > 0:
            orders.append(
                np.log(err_prev / err_curr) / np.log(dzs[i - 1] / dzs[i])
            )

    assert orders[-1] > 1.5, (
        f"Expected O(h²) convergence for reflected slab, got order {orders[-1]:.2f}"
    )
