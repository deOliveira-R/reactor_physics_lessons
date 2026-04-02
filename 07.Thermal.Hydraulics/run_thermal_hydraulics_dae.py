#!/usr/bin/env python3
"""Run the DAE version of coupled thermal-hydraulics + fuel behaviour LOCA calculation.

Uses pressure as a state variable with stiff relaxation, matching MATLAB's DAE
structure. This avoids NaN from out-of-range pyXSteam calls during post-failure
LOCA blowdown.
"""

import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thermal_hydraulics_dae import solve_thermal_hydraulics

OUTPUT = Path("results_dae")


def main():
    print("=" * 70)
    print("THERMAL HYDRAULICS (DAE) — PWR LOCA TRANSIENT (600s)")
    print("=" * 70)

    result = solve_thermal_hydraulics()

    # fuel_T shape: (nz, nf, nt)
    print(f"\n  Time steps: {len(result.time)}")
    print(f"  Max fuel center T: {result.fuel_T[0, 0, :].max() - 273:.0f} C")
    print(f"  Max clad outer T: {result.clad_T[0, -1, :].max() - 273:.0f} C")
    if hasattr(result, 'clad_fail_time') and result.clad_fail_time is not None and not np.isnan(result.clad_fail_time):
        print(f"  Clad failure at: {result.clad_fail_time:.1f} s")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    nt = len(result.time)

    # Temperature evolution (first axial node)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.time, result.fuel_T[0, 0, :nt] - 273, "-r", label="Fuel center")
    ax.plot(result.time, result.clad_T[0, -1, :nt] - 273, "-b", label="Clad outer")
    ax.plot(result.time, result.cool_T[0, :nt] - 273, "-g", label="Coolant node 1")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "TH_01_temperatures.pdf")
    plt.close(fig)

    # Pressure
    fig, ax = plt.subplots()
    ax.plot(result.time, result.cool_p[0, :nt], "-b", label="Node 1")
    if result.cool_p.shape[0] > 1:
        ax.plot(result.time, result.cool_p[1, :nt], "--b", label="Node 2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (MPa)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "TH_02_pressure.pdf")
    plt.close(fig)

    # Void fraction
    fig, ax = plt.subplots()
    ax.plot(result.time, result.cool_void[0, :nt], "-b", label="Node 1")
    if result.cool_void.shape[0] > 1:
        ax.plot(result.time, result.cool_void[1, :nt], "--b", label="Node 2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Void fraction (-)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "TH_03_void.pdf")
    plt.close(fig)

    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
