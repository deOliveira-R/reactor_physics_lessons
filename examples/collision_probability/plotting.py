"""Plotting utilities for the collision probability method."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

if TYPE_CHECKING:
    from orpheus.geometry import Mesh1D
    from orpheus.cp.solver import CPResult


# Material colors
_MAT_COLORS = {2: "#d62728", 1: "#2ca02c", 0: "#1f77b4"}  # fuel, clad, cool
_MAT_LABELS = {2: "Fuel", 1: "Cladding", 0: "Coolant"}


def _material_boundaries(mesh: Mesh1D) -> list[tuple[float, str]]:
    """Find edges where the material ID changes."""
    boundaries = []
    for i in range(mesh.N - 1):
        if mesh.mat_ids[i] != mesh.mat_ids[i + 1]:
            mid_left = mesh.mat_ids[i]
            mid_right = mesh.mat_ids[i + 1]
            label = f"{_MAT_LABELS.get(mid_left, str(mid_left))}/{_MAT_LABELS.get(mid_right, str(mid_right))}"
            boundaries.append((mesh.edges[i + 1], label))
    return boundaries


def plot_cp_geometry(
    mesh: Mesh1D,
    output_dir: Path | str = ".",
    filename: str = "CP_01_geometry.pdf",
) -> None:
    """Plot concentric annular sub-regions colored by material."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    radii = mesh.edges[1:]  # outer radius of each annulus
    plotted_labels = set()

    # Draw from outermost to innermost so inner rings overlap outer
    for k in reversed(range(mesh.N)):
        mat_id = mesh.mat_ids[k]
        label = _MAT_LABELS[mat_id] if mat_id not in plotted_labels else None
        plotted_labels.add(mat_id)

        circle = Circle(
            (0, 0), radii[k],
            facecolor=_MAT_COLORS[mat_id], edgecolor="k",
            linewidth=0.3, alpha=0.7, label=label,
        )
        ax.add_patch(circle)

    r_cell = mesh.edges[-1]
    lim = r_cell * 1.1
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
    """Plot keff convergence history and neutron balance residual.

    Produces two subplots: keff vs iteration (top) and residual vs
    iteration on a log scale (bottom).  If no residual data is
    available, only the keff plot is shown.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_residual = len(result.residual_history) > 0

    if has_residual:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(8, 4))

    # keff convergence
    iters = range(1, len(result.keff_history) + 1)
    ax1.plot(iters, result.keff_history, "-or", markersize=3)
    ax1.set_ylabel("k-effective")
    ax1.set_title("Power iteration convergence")
    ax1.grid(True)

    # Neutron balance residual
    if has_residual:
        # Residual history starts at iteration 3 (first 2 skipped)
        n_skip = len(result.keff_history) - len(result.residual_history)
        res_iters = range(n_skip + 1, len(result.keff_history) + 1)
        ax2.semilogy(res_iters, result.residual_history, "-sb", markersize=3)
        ax2.set_xlabel("Iteration number")
        ax2.set_ylabel("Neutron balance residual (L₂)")
        ax2.grid(True)
    else:
        ax1.set_xlabel("Iteration number")

    fig.tight_layout()
    fig.savefig(output_dir / "CP_02_keff.pdf")
    plt.close(fig)


def plot_cp_inner_iterations(
    result: CPResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot inner iteration counts per energy group per outer iteration.

    Only applicable when solver_mode = "gauss_seidel".  Produces a
    heatmap where the x-axis is energy group, y-axis is outer iteration,
    and color intensity shows how many inner iterations were needed.

    Groups that need many inner iterations have strong within-group
    scattering (typically thermal groups).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if result.n_inner is None:
        return

    n_inner = result.n_inner  # (n_outer, ng)
    n_outer, ng = n_inner.shape

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Heatmap: inner iterations per (outer, group)
    im = ax1.imshow(
        n_inner, aspect="auto", origin="lower",
        cmap="YlOrRd", interpolation="nearest",
    )
    ax1.set_xlabel("Energy group index")
    ax1.set_ylabel("Outer iteration")
    ax1.set_title("Inner iterations per group per outer iteration")
    fig.colorbar(im, ax=ax1, label="Inner iterations")

    # Summary: max and mean inner iterations per group (last outer)
    ax2.bar(range(ng), n_inner[-1, :], color="#d62728", alpha=0.7,
            label="Last outer iteration")
    ax2.bar(range(ng), n_inner.mean(axis=0), color="#1f77b4", alpha=0.5,
            label="Mean across outers")
    ax2.set_xlabel("Energy group index")
    ax2.set_ylabel("Inner iterations")
    ax2.set_title("Inner iterations per group")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "CP_05_inner_iterations.pdf")
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
    mesh = result.geometry

    # Mid-radius of each sub-region
    r_mid = mesh.centers

    # Group ranges (same convention as DO/MoC)
    FI_T = result.flux[:, :50].sum(axis=1)     # thermal < 1 eV
    FI_R = result.flux[:, 50:287].sum(axis=1)   # resonance < 0.1 MeV
    FI_F = result.flux[:, 287:].sum(axis=1)     # fast > 0.1 MeV

    fig, ax = plt.subplots()
    ax.plot(r_mid, FI_F, "-or", label="Fast", markersize=4)
    ax.plot(r_mid, FI_R, "-og", label="Resonance", markersize=4)
    ax.plot(r_mid, FI_T, "-ob", label="Thermal", markersize=4)

    # Mark material boundaries
    for r, lbl in _material_boundaries(mesh):
        ax.axvline(r, color="gray", linestyle="--", linewidth=0.8)

    ax.set_xlabel("Radius (cm)")
    ax.set_ylabel("Neutron flux (a.u.)")
    ax.set_title("Radial flux distribution")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "CP_04_flux_radial.pdf")
    plt.close(fig)