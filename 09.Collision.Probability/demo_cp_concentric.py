#!/usr/bin/env python3
"""Run the collision probability transport calculation for a PWR pin cell.

Uses the same materials as the Discrete Ordinates (SN) exercise to allow
direct comparison of eigenvalues.

Reference results:
    SN (slab geometry):   keff = 1.04188
    MC (slab geometry):   keff = 1.03484 +/- 0.00192
"""

import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path

from data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad
from geometry import pwr_pin_equivalent
from collision_probability import CPParams, solve_cp
from plotting import (
    plot_cp_convergence,
    plot_cp_geometry,
    plot_cp_radial_flux,
    plot_cp_spectra,
)

OUTPUT = Path("results")


def main():
    print("=" * 70)
    print("COLLISION PROBABILITY — PWR PIN CELL (Wigner-Seitz)")
    print("=" * 70)

    # 1. Build per-material macroscopic cross sections (same as SN)
    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    # 2. Set up Wigner-Seitz cylindrical geometry
    mesh = pwr_pin_equivalent(n_fuel=10, n_clad=3, n_cool=7)
    params = CPParams()

    n_fuel = (mesh.mat_ids == 2).sum()
    n_clad = (mesh.mat_ids == 1).sum()
    n_cool = (mesh.mat_ids == 0).sum()

    print(f"\n  Geometry: r_cell = {mesh.edges[-1]:.3f} cm")
    print(f"  Sub-regions: {n_fuel} fuel + {n_clad} clad "
          f"+ {n_cool} cool = {mesh.N} total")
    print()

    # 3. Solve
    result = solve_cp(materials, mesh, params)

    # 4. Report
    print(f"\n  keff = {result.keff:.5f}  (SN slab reference: 1.04188)")
    print(f"  Outer iterations: {len(result.keff_history)}")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")

    # 5. Plots
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_cp_geometry(mesh, OUTPUT)
    plot_cp_convergence(result, OUTPUT)
    plot_cp_spectra(result, OUTPUT)
    plot_cp_radial_flux(result, OUTPUT)
    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
