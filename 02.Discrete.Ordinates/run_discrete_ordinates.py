#!/usr/bin/env python3
"""Run the 2D Discrete Ordinates (SN) transport calculation for a PWR pin cell.

Reference MATLAB result:
    keff = 1.04188  (110 Lebedev ordinates, P0 scattering)
"""

import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path

from data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad
from discrete_ordinates import (
    DOParams,
    PinCellGeometry,
    Quadrature,
    solve_discrete_ordinates,
)
from plotting import (
    plot_do_convergence,
    plot_do_spatial_flux,
    plot_do_spectra,
    plot_mesh_2d,
)

OUTPUT = Path("results")


def main():
    print("=" * 70)
    print("DISCRETE ORDINATES — PWR PIN CELL (2D)")
    print("=" * 70)

    # 1. Build per-material macroscopic cross sections
    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    # 2. Set up geometry and angular quadrature
    geom = PinCellGeometry.default_pwr()
    params = DOParams(L=0)
    quad = Quadrature.lebedev(order=17, L=params.L)

    print(f"\n  Mesh: {geom.nx} x {geom.ny}, delta = {geom.delta} cm")
    print(f"  Ordinates: {quad.N} (Lebedev order 17)")
    print(f"  Scattering anisotropy: P{params.L}")
    print()

    # 3. Solve
    result = solve_discrete_ordinates(materials, geom, quad, params)

    # 4. Report
    print(f"\n  keff = {result.keff:.5f}  (MATLAB reference: 1.04188)")
    match = "YES" if abs(result.keff - 1.04188) < 5e-4 else "NO"
    print(f"  Match: {match}")
    print(f"  Outer iterations: {len(result.keff_history)}")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")

    # 5. Plots
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_mesh_2d(geom, OUTPUT)
    plot_do_convergence(result, OUTPUT)
    plot_do_spectra(result, OUTPUT)
    plot_do_spatial_flux(result, OUTPUT)
    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
