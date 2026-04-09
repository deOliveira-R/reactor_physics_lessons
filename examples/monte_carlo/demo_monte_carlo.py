#!/usr/bin/env python3
"""Run Monte Carlo neutron transport simulation for a PWR pin cell.

Reference MATLAB result:
    keff = 1.03484 +/- 0.00192  (100 neutrons, 100 inactive + 2000 active cycles)

Note: Results are stochastic and will vary between runs.  The keff should
agree with the MATLAB reference within ~2-3 sigma.
"""

from pathlib import Path

from orpheus.data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad
from orpheus.mc.solver import MCParams, solve_monte_carlo
from plotting import plot_mc_keff, plot_mc_spectrum

OUTPUT = Path("results")


def main():
    print("=" * 70)
    print("MONTE CARLO — PWR PIN CELL (2D)")
    print("=" * 70)

    # 1. Build per-material macroscopic cross sections
    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    # 2. Run MC
    params = MCParams(
        n_neutrons=100,
        n_inactive=100,
        n_active=2000,
        pitch=3.6,
    )
    print(f"\n  Neutrons/cycle: {params.n_neutrons}")
    print(f"  Inactive cycles: {params.n_inactive}")
    print(f"  Active cycles: {params.n_active}")
    print(f"  Cell pitch: {params.pitch} cm")
    print()

    result = solve_monte_carlo(materials, params)

    # 3. Report
    print(f"\n  keff = {result.keff:.5f} +/- {result.sigma:.5f}")
    print(f"  MATLAB reference: 1.03484 +/- 0.00192")
    within = abs(result.keff - 1.03484) / max(result.sigma, 1e-6)
    print(f"  Deviation: {within:.1f} sigma")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")

    # 4. Plots
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_mc_keff(result, OUTPUT)
    plot_mc_spectrum(result, OUTPUT)
    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
