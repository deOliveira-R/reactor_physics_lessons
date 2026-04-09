#!/usr/bin/env python3
"""Run the coupled reactor kinetics + thermal-hydraulics RIA calculation.

Simulates a Reactivity Insertion Accident (RIA) in a PWR:
  - 0-100s: steady-state at 69.14 kW
  - 100-120s: transient with Doppler and coolant temperature feedback

Reference: MATLAB Module 10 (Reactor.Kinetics.0D)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orpheus.kinetics.solver import solve_reactor_kinetics

OUTPUT = Path("results")


def main():
    print("=" * 70)
    print("REACTOR KINETICS — PWR RIA TRANSIENT")
    print("=" * 70)

    result = solve_reactor_kinetics()

    pow0 = result.params.pow0
    print(f"\n  Time steps: {len(result.time)}")
    print(f"  Initial power: {result.power[0] * pow0:.0f} W")
    print(f"  Max power: {result.power.max() * pow0:.0f} W ({result.power.max():.1f}x nominal)")
    print(f"  Final power: {result.power[-1] * pow0:.0f} W")

    # Fuel T shape: (nt, nz, nf)
    if result.fuel_T.ndim == 3:
        max_fuel_T = result.fuel_T[0, 0, :].max() - 273  # (nz, nf, nt)
    else:
        max_fuel_T = result.fuel_T.max() - 273
    print(f"  Max fuel center T: {max_fuel_T:.0f} C")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Power
    fig, ax = plt.subplots()
    ax.plot(result.time, result.power * pow0 / 1e3, "-r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (kW)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "RK_01_power.pdf")
    plt.close(fig)

    # Reactivity
    fig, ax = plt.subplots()
    ax.plot(result.time, result.reac_doppler * 1e5, "-r", label="Doppler")
    ax.plot(result.time, result.reac_coolant * 1e5, "-b", label="Coolant")
    ax.plot(result.time, result.reac_total * 1e5, "-k", label="Total")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reactivity (pcm)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "RK_02_reactivity.pdf")
    plt.close(fig)

    # Temperature (first axial node)
    fig, ax = plt.subplots()
    if result.fuel_T.ndim == 3:
        ax.plot(result.time, result.fuel_T[0, 0, :] - 273, "-r", label="Fuel center")
        ax.plot(result.time, result.clad_T[0, -1, :] - 273, "-b", label="Clad outer")
    ax.plot(result.time, result.cool_T[0, :] - 273, "-g", label="Coolant")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "RK_03_temperatures.pdf")
    plt.close(fig)

    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
