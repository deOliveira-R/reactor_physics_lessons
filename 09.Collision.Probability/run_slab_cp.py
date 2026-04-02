#!/usr/bin/env python3
"""Run the SLAB collision probability transport calculation.

Uses the same materials and geometry as the Discrete Ordinates (SN)
exercise to allow direct slab-to-slab comparison of eigenvalues.

Reference results:
    SN (slab geometry):   keff = 1.04188
"""

import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path

from data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad
from collision_probability_slab import SlabGeometry, solve_slab_cp

OUTPUT = Path("results")


def main():
    print("=" * 70)
    print("COLLISION PROBABILITY (SLAB) — PWR PIN CELL")
    print("=" * 70)

    # Same materials as SN
    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    # Match SN geometry: fuel 0.9 cm, clad 0.2 cm, coolant 0.7 cm
    geom = SlabGeometry.default_pwr(n_fuel=10, n_clad=3, n_cool=7)

    print(f"\n  Half-cell: {geom.half_cell:.3f} cm")
    print(f"  Fuel: {geom.thicknesses[:geom.n_fuel].sum():.3f} cm")
    print(f"  Clad: {geom.thicknesses[geom.n_fuel:geom.n_fuel+geom.n_clad].sum():.3f} cm")
    print(f"  Cool: {geom.thicknesses[geom.n_fuel+geom.n_clad:].sum():.3f} cm")
    print(f"  Sub-regions: {geom.n_fuel} fuel + {geom.n_clad} clad "
          f"+ {geom.n_cool} cool = {geom.N} total")
    print()

    result = solve_slab_cp(materials, geom)

    print(f"\n  keff = {result.keff:.5f}  (SN slab reference: 1.04188)")
    print(f"  Outer iterations: {len(result.keff_history)}")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
