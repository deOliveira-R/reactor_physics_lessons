"""Plotting for discrete ordinates (SN) results.

Module-specific DO plots. Shared functions imported from tools.plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from tools.plotting import plot_2d_field, plot_spectrum  # noqa: F401

if TYPE_CHECKING:
    from sn_geometry import CartesianMesh
    from sn_solver import SNResult


def plot_mesh_2d(
    mesh: CartesianMesh,
    output_dir: Path | str = ".",
    filename: str = "DO_01_mesh.pdf",
) -> None:
    """2D colored rectangle plot of the material map."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_2d_field(mesh.nx, mesh.ny, mesh.dx[0],
                  mesh.mat_map.astype(float), "Unit cell: materials",
                  output_dir / filename)


def plot_do_convergence(
    result: SNResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot keff convergence history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(result.keff_history) + 1), result.keff_history, "-or", markersize=3)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("k-effective")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "DO_02_keff.pdf")
    plt.close(fig)


def plot_do_spectra(
    result: SNResult,
    materials: dict,
    output_dir: Path | str = ".",
) -> None:
    """Plot neutron spectra per unit lethargy per material."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = result.geometry
    sf = result.scalar_flux
    vol = mesh.volume
    eg = result.eg
    eg_mid = 0.5 * (eg[:-1] + eg[1:])
    du = np.log(eg[1:] / eg[:-1])
    ng = sf.shape[2]

    labels = {2: ("Fuel", "r"), 1: ("Cladding", "g"), 0: ("Coolant", "b")}
    fig, ax = plt.subplots(figsize=(10, 6))
    for mat_id, (label, color) in labels.items():
        mask = mesh.mat_map == mat_id
        if not mask.any():
            continue
        vol_mat = vol[mask].sum()
        flux_avg = np.zeros(ng)
        for g in range(ng):
            flux_avg[g] = np.sum(sf[:, :, g][mask] * vol[mask]) / vol_mat
        ax.semilogx(eg_mid, flux_avg / du, f"-{color}", label=label)

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Neutron flux per unit lethargy (a.u.)")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "DO_04_flux_lethargy.pdf")
    plt.close(fig)


def plot_do_spatial_flux(
    result: SNResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot thermal/resonance/fast flux along cell centerline and 2D."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = result.geometry
    x = np.arange(mesh.nx) * mesh.dx[0]

    FI_T = result.scalar_flux[:, 0, :50].sum(axis=1)
    FI_R = result.scalar_flux[:, 0, 50:287].sum(axis=1)
    FI_F = result.scalar_flux[:, 0, 287:].sum(axis=1)

    fig, ax = plt.subplots()
    ax.plot(x, FI_F, "-or", label="Fast", markersize=3)
    ax.plot(x, FI_R, "-og", label="Resonance", markersize=3)
    ax.plot(x, FI_T, "-ob", label="Thermal", markersize=3)
    ax.set_xlabel("Distance from the cell centre (cm)")
    ax.set_ylabel("Neutron flux (a.u.)")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "DO_05_flux_cell.pdf")
    plt.close(fig)

    fun_T = result.scalar_flux[:, :, :50].sum(axis=2)
    fun_R = result.scalar_flux[:, :, 50:355].sum(axis=2)
    fun_F = result.scalar_flux[:, :, 355:].sum(axis=2)

    delta = mesh.dx[0]
    plot_2d_field(mesh.nx, mesh.ny, delta, fun_T,
                  "Thermal flux distribution", output_dir / "DO_06_flux_thermal.pdf")
    plot_2d_field(mesh.nx, mesh.ny, delta, fun_R,
                  "Resonance flux distribution", output_dir / "DO_07_flux_resonance.pdf")
    plot_2d_field(mesh.nx, mesh.ny, delta, fun_F,
                  "Fast flux distribution", output_dir / "DO_08_flux_fast.pdf")
