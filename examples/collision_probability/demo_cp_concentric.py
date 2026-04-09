#!/usr/bin/env python3
"""Run the collision probability transport calculation for a PWR pin cell.

Uses the same materials as the Discrete Ordinates (SN) exercise to allow
direct comparison of eigenvalues.

Supports both Jacobi (default) and Gauss-Seidel solver modes.
Pass --gauss-seidel to use the GS group sweep with inner iterations.

Reference results:
    SN (slab geometry):   keff = 1.04188
    MC (slab geometry):   keff = 1.03484 +/- 0.00192
"""

import argparse
from pathlib import Path

from orpheus.data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad
from orpheus.geometry import pwr_pin_equivalent
from orpheus.cp.solver import CPParams, solve_cp
from plotting import (
    plot_cp_convergence,
    plot_cp_geometry,
    plot_cp_inner_iterations,
    plot_cp_radial_flux,
    plot_cp_spectra,
)

OUTPUT = Path("results")


def main():
    parser = argparse.ArgumentParser(description="CP solver for PWR pin cell")
    parser.add_argument("--gauss-seidel", action="store_true",
                        help="Use Gauss-Seidel group sweep with inner iterations")
    args = parser.parse_args()

    solver_mode = "gauss_seidel" if args.gauss_seidel else "jacobi"

    print("=" * 70)
    print(f"COLLISION PROBABILITY — PWR PIN CELL (mode: {solver_mode})")
    print("=" * 70)

    # 1. Build per-material macroscopic cross sections (same as SN)
    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    # 2. Set up Wigner-Seitz cylindrical geometry
    mesh = pwr_pin_equivalent(n_fuel=10, n_clad=3, n_cool=7)
    params = CPParams(solver_mode=solver_mode)

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
    if result.residual_history:
        print(f"  Final balance residual: {result.residual_history[-1]:.2e}")
    if result.n_inner is not None:
        n_in = result.n_inner
        print(f"  Inner iterations (last outer): "
              f"max={n_in[-1].max()}, mean={n_in[-1].mean():.1f}")
        # Identify groups needing most inner iterations
        worst_g = n_in[-1].argmax()
        print(f"  Most inner iterations in group {worst_g} "
              f"({n_in[-1, worst_g]} iters)")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")

    # 5. Plots
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_cp_geometry(mesh, OUTPUT)
    plot_cp_convergence(result, OUTPUT)
    plot_cp_spectra(result, OUTPUT)
    plot_cp_radial_flux(result, OUTPUT)
    plot_cp_inner_iterations(result, OUTPUT)
    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()