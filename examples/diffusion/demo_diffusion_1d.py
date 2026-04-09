#!/usr/bin/env python3
"""Run the 1D two-group neutron diffusion calculation for a PWR subassembly.

Reference MATLAB result:
    keff = 1.022173
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orpheus.diffusion.solver import solve_diffusion_1d

OUTPUT = Path("results")


def main():
    print("=" * 70)
    print("1D TWO-GROUP DIFFUSION — PWR SUBASSEMBLY")
    print("=" * 70)

    result = solve_diffusion_1d()

    print(f"\n  keff = {result.keff:.6f}  (MATLAB reference: 1.022173)")
    match = "YES" if abs(result.keff - 1.022173) < 1e-4 else "NO"
    print(f"  Match: {match}")

    # Plots
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(result.z_cells, result.flux[0, :], "-or", label="Fast", markersize=3)
    ax.plot(result.z_cells, result.flux[1, :], "-ob", label="Thermal", markersize=3)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"Neutron flux (n/cm$^2$s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "DIF_01_flux.pdf")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(result.z_faces, result.current[0, :], "-or", label="Fast", markersize=3)
    ax.plot(result.z_faces, result.current[1, :], "-ob", label="Thermal", markersize=3)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"Neutron net current (n/cm$^2$s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "DIF_02_current.pdf")
    plt.close(fig)

    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
