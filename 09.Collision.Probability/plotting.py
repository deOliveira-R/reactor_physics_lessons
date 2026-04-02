"""Plotting utilities for the collision probability method."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

if TYPE_CHECKING:
    from collision_probability import CPGeometry, CPResult



# Material colors
_MAT_COLORS = {2: "#d62728", 1: "#2ca02c", 0: "#1f77b4"}  # fuel, clad, cool
_MAT_LABELS = {2: "Fuel", 1: "Cladding", 0: "Coolant"}


def plot_cp_geometry(
    geom: CPGeometry,
    output_dir: Path | str = ".",
    filename: str = "CP_01_geometry.pdf",
) -> None:
    """Plot concentric annular sub-regions colored by material."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    r_inner = np.concatenate(([0.0], geom.radii[:-1]))
    plotted_labels = set()

    # Draw from outermost to innermost so inner rings overlap outer
    for k in reversed(range(geom.N)):
        mat_id = geom.mat_ids[k]
        label = _MAT_LABELS[mat_id] if mat_id not in plotted_labels else None
        plotted_labels.add(mat_id)

        circle = Circle(
            (0, 0), geom.radii[k],
            facecolor=_MAT_COLORS[mat_id], edgecolor="k",
            linewidth=0.3, alpha=0.7, label=label,
        )
        ax.add_patch(circle)

    lim = geom.r_cell * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("Wigner-Seitz cell: annular sub-regions")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def plot_cp_convergence(
    result: CPResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot keff convergence history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    iters = range(1, len(result.keff_history) + 1)
    ax.plot(iters, result.keff_history, "-or", markersize=3)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("k-effective")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "CP_02_keff.pdf")
    plt.close(fig)


def plot_cp_spectra(
    result: CPResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot neutron spectra per unit lethargy in fuel, cladding, coolant."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eg = result.eg
    eg_mid = 0.5 * (eg[:-1] + eg[1:])
    du = np.log(eg[1:] / eg[:-1])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(eg_mid, result.flux_fuel / du, "-r", label="Fuel")
    ax.semilogx(eg_mid, result.flux_clad / du, "-g", label="Cladding")
    ax.semilogx(eg_mid, result.flux_cool / du, "-b", label="Coolant")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Neutron flux per unit lethargy (a.u.)")
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "CP_03_flux_lethargy.pdf")
    plt.close(fig)


def plot_cp_radial_flux(
    result: CPResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot radial flux profile for thermal, resonance, and fast groups."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geom = result.geometry

    # Mid-radius of each sub-region
    r_inner = np.concatenate(([0.0], geom.radii[:-1]))
    r_mid = 0.5 * (r_inner + geom.radii)

    # Group ranges (same convention as DO/MoC)
    FI_T = result.flux[:, :50].sum(axis=1)     # thermal < 1 eV
    FI_R = result.flux[:, 50:287].sum(axis=1)   # resonance < 0.1 MeV
    FI_F = result.flux[:, 287:].sum(axis=1)     # fast > 0.1 MeV

    fig, ax = plt.subplots()
    ax.plot(r_mid, FI_F, "-or", label="Fast", markersize=4)
    ax.plot(r_mid, FI_R, "-og", label="Resonance", markersize=4)
    ax.plot(r_mid, FI_T, "-ob", label="Thermal", markersize=4)

    # Mark material boundaries
    for r, lbl in [(geom.r_fuel, "Fuel/Clad"), (geom.r_clad, "Clad/Cool")]:
        ax.axvline(r, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Radius (cm)")
    ax.set_ylabel("Neutron flux (a.u.)")
    ax.set_title("Radial flux distribution")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "CP_04_flux_radial.pdf")
    plt.close(fig)
