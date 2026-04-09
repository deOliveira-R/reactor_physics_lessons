#!/usr/bin/env python3
"""Monte Carlo neutron transport solver for a PWR pin cell.

Simulates neutron histories in a 2D unit cell using Woodcock delta tracking,
analog absorption with fission weight adjustment, Russian roulette, and
splitting.  Returns keff with statistical uncertainty.

This script contains the full solver (port of MATLAB monteCarloPWR.m),
all plotting utilities, and a main() that runs the demonstration case.

Reference MATLAB result:
    keff = 1.03484 +/- 0.00192  (100 neutrons, 100 inactive + 2000 active cycles)

Note: Results are stochastic and will vary between runs.  The keff should
agree with the MATLAB reference within ~2-3 sigma.

Dependencies:
    pip install numpy scipy matplotlib
    ORPHEUS data package (orpheus.data)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class MCParams:
    """Monte Carlo simulation parameters."""

    n_neutrons: int = 100        # source neutrons per cycle
    n_inactive: int = 100        # inactive cycles (source convergence)
    n_active: int = 2000         # active cycles (tally accumulation)
    pitch: float = 3.6           # unit cell side length (cm)
    seed: int | None = None      # Rng seed (None = random)


@dataclass
class MCResult:
    """Results of a Monte Carlo calculation."""

    keff: float               # estimated k-effective
    sigma: float              # standard deviation of keff
    keff_history: np.ndarray  # (n_active,) cumulative mean keff
    sigma_history: np.ndarray  # (n_active,) cumulative sigma
    flux_per_lethargy: np.ndarray  # (ng,) cell-averaged flux / du
    eg_mid: np.ndarray        # (ng,) mid-group energies
    elapsed_seconds: float


# ===========================================================================
# Core functions
# ===========================================================================

def _find_material(
    x: float,
    fuel: Mixture,
    clad: Mixture,
    cool: Mixture,
) -> Mixture:
    """Determine material region from x-coordinate.

    Geometry (matching MATLAB):
        fuel:  0.9 < x < 2.7
        clad:  0.7 <= x <= 0.9  or  2.7 <= x <= 2.9
        cool:  x < 0.7  or  x > 2.9
    """
    if 0.9 < x < 2.7:
        return fuel
    elif x < 0.7 or x > 2.9:
        return cool
    else:
        return clad


# ===========================================================================
# Main solver
# ===========================================================================

def solve_monte_carlo(
    materials: dict[int, Mixture],
    params: MCParams | None = None,
) -> MCResult:
    """Run Monte Carlo neutron transport simulation.

    Parameters
    ----------
    materials : dict mapping material ID (0=cool, 1=clad, 2=fuel) to Mixture.
    params : MCParams (default: 100 neutrons, 100 inactive + 2000 active).
    """
    t_start = time.perf_counter()

    if params is None:
        params = MCParams()

    rng = np.random.default_rng(params.seed)
    fuel, clad, cool = materials[2], materials[1], materials[0]
    ng = fuel.ng
    pitch = params.pitch

    eg = fuel.eg
    eg_mid = 0.5 * (eg[:ng] + eg[1:ng + 1])
    du = np.log(eg[1:ng + 1] / eg[:ng])

    # Majorant: maximum total cross section across all materials
    sig_t_max = np.maximum(np.maximum(fuel.SigT, clad.SigT), cool.SigT)

    # Precompute dense scattering rows (from group ig to all groups)
    # sig_s_dense[mat_id][ig] = dense array of shape (ng,)
    sig_s_dense = {}
    for mat_id, mix in materials.items():
        rows = np.array(mix.SigS[0].todense())  # (ng, ng)
        sig_s_dense[mat_id] = rows

    # Cumulative fission spectrum for sampling
    chi_cum = np.cumsum(fuel.chi)

    # Scattering detector
    detect_s = np.zeros(ng)

    # Initialize neutron population
    max_n = params.n_neutrons * 4  # buffer for splitting
    x = rng.random(max_n) * pitch
    y = rng.random(max_n) * pitch
    weight = np.ones(max_n)
    i_group = np.array([np.searchsorted(chi_cum, rng.random()) for _ in range(max_n)], dtype=int)
    i_group = np.clip(i_group, 0, ng - 1)
    n_neutrons = params.n_neutrons

    keff_active = np.zeros(params.n_active)
    keff_history = np.zeros(params.n_active)
    sigma_history = np.zeros(params.n_active)

    total_cycles = params.n_inactive + params.n_active

    for i_cycle in range(1, total_cycles + 1):
        # Normalize weights to n_neutrons_born
        total_weight = weight[:n_neutrons].sum()
        weight[:n_neutrons] *= params.n_neutrons / total_weight
        weight0 = weight[:n_neutrons].copy()

        # Loop over neutrons
        for i_n in range(n_neutrons):
            ig = i_group[i_n]
            nx_, ny_ = x[i_n], y[i_n]
            w = weight[i_n]
            virtual_collision = False

            # Random walk until absorption
            while True:
                # Free path (Woodcock delta tracking)
                free_path = -np.log(rng.random()) / sig_t_max[ig]

                if not virtual_collision:
                    theta = np.pi * rng.random()
                    phi = 2.0 * np.pi * rng.random()
                    dir_x = np.sin(theta) * np.cos(phi)
                    dir_y = np.sin(theta) * np.sin(phi)

                nx_ += free_path * dir_x
                ny_ += free_path * dir_y

                # Periodic boundary conditions
                nx_ = nx_ % pitch
                ny_ = ny_ % pitch

                # Determine material
                mat = _find_material(nx_, fuel, clad, cool)

                # Cross sections for this group
                if mat is fuel:
                    mat_id = 2
                    sig_a = fuel.SigF[ig] + fuel.SigC[ig] + fuel.SigL[ig]
                    sig_p = fuel.SigP[ig]
                elif mat is cool:
                    mat_id = 0
                    sig_a = cool.SigC[ig] + cool.SigL[ig]
                    sig_p = 0.0
                else:
                    mat_id = 1
                    sig_a = clad.SigC[ig] + clad.SigL[ig]
                    sig_p = 0.0

                sig_s_row = sig_s_dense[mat_id][ig, :]
                sig_s_sum = sig_s_row.sum()
                sig_t = sig_a + sig_s_sum
                sig_v = sig_t_max[ig] - sig_t

                # Virtual or real collision?
                if sig_v / sig_t_max[ig] >= rng.random():
                    virtual_collision = True
                else:
                    virtual_collision = False

                    if sig_s_sum / sig_t >= rng.random():
                        # Scattering
                        detect_s[ig] += w / sig_s_sum

                        # Sample outgoing energy group
                        cum_s = np.cumsum(sig_s_row)
                        ig = np.searchsorted(cum_s, rng.random() * sig_s_sum)
                        ig = min(ig, ng - 1)
                    else:
                        # Absorption -> convert to fission neutron
                        if sig_a > 0:
                            w *= sig_p / sig_a
                        else:
                            w = 0.0

                        # Sample new energy group from fission spectrum
                        ig = np.searchsorted(chi_cum, rng.random())
                        ig = min(ig, ng - 1)
                        break

            x[i_n] = nx_
            y[i_n] = ny_
            weight[i_n] = w
            i_group[i_n] = ig

        # Russian roulette
        for i_n in range(n_neutrons):
            if weight0[i_n] > 0:
                terminate_p = 1.0 - weight[i_n] / weight0[i_n]
            else:
                terminate_p = 1.0
            if terminate_p >= rng.random():
                weight[i_n] = 0.0
            elif terminate_p > 0:
                weight[i_n] = weight0[i_n]

        # Remove killed neutrons
        alive = weight[:n_neutrons] > 0
        n_alive = alive.sum()
        x[:n_alive] = x[:n_neutrons][alive]
        y[:n_alive] = y[:n_neutrons][alive]
        weight[:n_alive] = weight[:n_neutrons][alive]
        i_group[:n_alive] = i_group[:n_neutrons][alive]
        n_neutrons = n_alive

        # Split heavy neutrons
        n_new = 0
        for i_n in range(n_neutrons):
            if weight[i_n] > 1.0:
                N = int(np.floor(weight[i_n]))
                if weight[i_n] - N > rng.random():
                    N += 1
                new_w = weight[i_n] / N
                weight[i_n] = new_w
                for _ in range(N - 1):
                    idx = n_neutrons + n_new
                    if idx >= len(x):
                        # Grow arrays
                        grow = max(len(x), 100)
                        x = np.append(x, np.zeros(grow))
                        y = np.append(y, np.zeros(grow))
                        weight = np.append(weight, np.zeros(grow))
                        i_group = np.append(i_group, np.zeros(grow, dtype=int))
                    x[idx] = x[i_n]
                    y[idx] = y[i_n]
                    weight[idx] = new_w
                    i_group[idx] = i_group[i_n]
                    n_new += 1
        n_neutrons += n_new

        # keff for this cycle
        keff_cycle = weight[:n_neutrons].sum() / weight0.sum()

        i_active = i_cycle - params.n_inactive
        if i_active <= 0:
            if i_cycle % 20 == 0 or i_cycle <= 5:
                print(f"  Inactive {i_cycle:3d}/{params.n_inactive}  "
                      f"keff_cycle = {keff_cycle:.5f}  n = {n_neutrons}")
        else:
            ia = i_active - 1
            keff_active[ia] = keff_cycle
            keff_history[ia] = keff_active[:i_active].mean()
            if i_active > 1:
                sigma_history[ia] = np.sqrt(
                    ((keff_active[:i_active] - keff_history[ia])**2).sum()
                    / (i_active - 1) / i_active
                )
            if i_active % 200 == 0 or i_active <= 5:
                print(f"  Active {i_active:4d}/{params.n_active}  "
                      f"keff = {keff_history[ia]:.5f} +/- {sigma_history[ia]:.5f}  "
                      f"n = {n_neutrons}")

    elapsed = time.perf_counter() - t_start
    flux_du = detect_s / du

    print(f"  Elapsed: {elapsed:.1f}s")

    return MCResult(
        keff=keff_history[-1],
        sigma=sigma_history[-1],
        keff_history=keff_history,
        sigma_history=sigma_history,
        flux_per_lethargy=flux_du,
        eg_mid=eg_mid,
        elapsed_seconds=elapsed,
    )


# ===========================================================================
# Plotting
# ===========================================================================

def plot_mc_keff(
    result: MCResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot keff convergence with uncertainty bands."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    iters = np.arange(1, len(result.keff_history) + 1)
    fig, ax = plt.subplots()
    ax.plot(iters, result.keff_history, "-r", label=r"$k_{eff}$")
    ax.plot(iters, result.keff_history + result.sigma_history, "--b",
            label=r"$k_{eff} \pm \sigma$")
    ax.plot(iters, result.keff_history - result.sigma_history, "--b")
    ax.set_xlabel("Active cycle number")
    ax.set_ylabel("k-effective")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "MC_01_keff.pdf")
    plt.close(fig)


def plot_mc_spectrum(
    result: MCResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot cell-averaged neutron flux per unit lethargy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(result.eg_mid, result.flux_per_lethargy)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Neutron flux per unit lethargy (a.u.)")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "MC_02_flux_lethargy.pdf")
    plt.close(fig)


# ===========================================================================
# Demo run
# ===========================================================================

OUTPUT = Path("04_results")


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
