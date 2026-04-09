#!/usr/bin/env python3
"""Run the 2D Method of Characteristics (MoC) transport calculation for a PWR pin cell.

Reference MATLAB result:
    keff = 1.04923  (8 ray directions, P0 scattering, 10x10 mesh)
"""

from pathlib import Path

from orpheus.data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad
from orpheus.moc.solver import solve_moc
from orpheus.moc.geometry import MOCMesh
from plotting import (
    plot_moc_convergence,
    plot_moc_mesh,
    plot_moc_rays,
    plot_moc_spatial_flux,
    plot_moc_spectra,
)

OUTPUT = Path("results")


def main():
    print("=" * 70)
    print("METHOD OF CHARACTERISTICS — PWR PIN CELL (2D)")
    print("=" * 70)

    # 1. Build per-material macroscopic cross sections
    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    # 2. Set up geometry
    geom = MOCMesh.default_pwr()
    print(f"\n  Mesh: {geom.n_cells} x {geom.n_cells}, delta = {geom.delta} cm")
    print(f"  Ray directions: 8 (E, NE, N, NW, W, SW, S, SE)")
    print()

    # 3. Solve
    result = solve_moc(materials, geom, max_outer=200)

    # 4. Report
    print(f"\n  keff = {result.keff:.5f}  (MATLAB reference: 1.04923)")
    match = "YES" if abs(result.keff - 1.04923) < 5e-4 else "NO"
    print(f"  Match: {match}")
    print(f"  Outer iterations: {len(result.keff_history)}")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")

    # 5. Plots
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_moc_rays(geom, OUTPUT)
    plot_moc_mesh(geom, OUTPUT)
    plot_moc_convergence(result, OUTPUT)
    plot_moc_spectra(result, OUTPUT)
    plot_moc_spatial_flux(result, OUTPUT)
    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
