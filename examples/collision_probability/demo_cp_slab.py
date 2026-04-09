#!/usr/bin/env python3
"""Run the SLAB collision probability transport calculation.

Uses the same materials and geometry as the Discrete Ordinates (SN)
exercise to allow direct slab-to-slab comparison of eigenvalues.

Reference results:
    SN (slab geometry):   keff = 1.04188
"""

from pathlib import Path

from orpheus.data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad
from orpheus.geometry import pwr_slab_half_cell
from orpheus.cp.solver import solve_cp

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
    mesh = pwr_slab_half_cell(n_fuel=10, n_clad=3, n_cool=7)

    n_fuel = (mesh.mat_ids == 2).sum()
    n_clad = (mesh.mat_ids == 1).sum()
    n_cool = (mesh.mat_ids == 0).sum()

    print(f"\n  Half-cell: {mesh.total_width:.3f} cm")
    print(f"  Fuel: {mesh.volumes[mesh.mat_ids == 2].sum():.3f} cm")
    print(f"  Clad: {mesh.volumes[mesh.mat_ids == 1].sum():.3f} cm")
    print(f"  Cool: {mesh.volumes[mesh.mat_ids == 0].sum():.3f} cm")
    print(f"  Sub-regions: {n_fuel} fuel + {n_clad} clad "
          f"+ {n_cool} cool = {mesh.N} total")
    print()

    result = solve_cp(materials, mesh)

    print(f"\n  keff = {result.keff:.5f}  (SN slab reference: 1.04188)")
    print(f"  Outer iterations: {len(result.keff_history)}")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
