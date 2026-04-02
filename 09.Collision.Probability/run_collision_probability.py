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
from collision_probability import (
    CPGeometry,
    CPParams,
    solve_collision_probability,
)
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
    geom = CPGeometry.default_pwr(n_fuel=10, n_clad=3, n_cool=7)
    params = CPParams()

    print(f"\n  Geometry: r_fuel = {geom.r_fuel:.3f} cm, "
          f"r_clad = {geom.r_clad:.3f} cm, r_cell = {geom.r_cell:.3f} cm")
    print(f"  Sub-regions: {geom.n_fuel} fuel + {geom.n_clad} clad "
          f"+ {geom.n_cool} cool = {geom.N} total")
    print()

    # 3. Solve
    result = solve_collision_probability(materials, geom, params)

    # 4. Report
    print(f"\n  keff = {result.keff:.5f}  (SN slab reference: 1.04188)")
    print(f"  Outer iterations: {len(result.keff_history)}")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")

    # 5. Plots
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_cp_geometry(geom, OUTPUT)
    plot_cp_convergence(result, OUTPUT)
    plot_cp_spectra(result, OUTPUT)
    plot_cp_radial_flux(result, OUTPUT)
    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
