#!/usr/bin/env python3
"""Run the 1D fuel rod thermo-mechanical behaviour calculation.

Simulates 6 years of fuel rod operation with swelling, creep, gap closure.
Reference MATLAB result: gap closure at ~2.8 years.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orpheus.fuel.solver import solve_fuel_behaviour

OUTPUT = Path("results")


def main():
    print("=" * 70)
    print("FUEL BEHAVIOUR — 1D THERMO-MECHANICS (6 YEARS)")
    print("=" * 70)

    result = solve_fuel_behaviour(verbose=True)

    print(f"\n  Time steps: {len(result.time)}")
    print(f"  Fuel center T (EOC): {result.fuel_T[0, -1] - 273:.0f} C")
    # Find gap closure time from gap_open array
    closed_idx = np.where(np.asarray(result.gap_open) == 0)[0]
    if len(closed_idx) > 0:
        print(f"  Gap closure at: {result.time_years[closed_idx[0]]:.2f} years")
    else:
        print("  No gap closure")
    print(f"  Gas pressure (EOC): {result.ingas_p[-1]:.2f} MPa")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Temperature profiles
    fig, ax = plt.subplots()
    ax.plot(result.fuel_r[:, 0] * 1e3, result.fuel_T[:, 0] - 273, "-ob",
            markersize=2, label="BOC")
    ax.plot(result.fuel_r[:, -1] * 1e3, result.fuel_T[:, -1] - 273, "-or",
            markersize=2, label="EOC")
    ax.plot(result.clad_r[:, 0] * 1e3, result.clad_T[:, 0] - 273, "-ob", markersize=2)
    ax.plot(result.clad_r[:, -1] * 1e3, result.clad_T[:, -1] - 273, "-or", markersize=2)
    ax.set_xlabel("Radius (mm)")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "FB_01_temperatures.pdf")
    plt.close(fig)

    # Radii evolution
    fig, ax = plt.subplots()
    ax.plot(result.time_years, result.fuel_r[-1, :] * 1e3, "-r", label="Fuel outer")
    ax.plot(result.time_years, result.clad_r[0, :] * 1e3, "-b", label="Clad inner")
    ax.plot(result.time_years, result.clad_r[-1, :] * 1e3, "--b", label="Clad outer")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Radius (mm)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "FB_02_radii.pdf")
    plt.close(fig)

    # Gap width
    fig, ax = plt.subplots()
    ax.plot(result.time_years, result.gap_dr * 1e6, "-r")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Gap width (um)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "FB_03_gap.pdf")
    plt.close(fig)

    # Gas pressure
    fig, ax = plt.subplots()
    ax.plot(result.time_years, result.ingas_p, "-r")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Inner gas pressure (MPa)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "FB_04_gas_pressure.pdf")
    plt.close(fig)

    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
