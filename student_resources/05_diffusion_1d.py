#!/usr/bin/env python3
"""1D two-group neutron diffusion solver for a PWR subassembly.

Physics
-------
This script solves the 1D neutron diffusion eigenvalue problem in two energy
groups (fast and thermal) for an axial slice of a PWR subassembly. The domain
consists of a bottom reflector, fuel region, and top reflector, each with
distinct macroscopic cross sections.

The diffusion equation for group g is:

    -d/dz [D_g(z) d phi_g/dz] + Sigma_a,g phi_g + Sigma_s,g->g' phi_g
        = chi_g / keff * sum_g'(nu Sigma_f,g' phi_g') + Sigma_s,g'->g phi_g'

where D_g = 1/(3 Sigma_tr,g) is the diffusion coefficient.

The discretization uses finite differences with vacuum (phi=0) boundary
conditions at the top and bottom faces. The eigenvalue keff is found by
outer (power) iterations, with BiCGSTAB inner iterations solving the
within-group linear system at each step. The flux is normalised to a
target thermal power after each outer iteration.

Cross sections are hardcoded defaults matching the MATLAB CORE1D.m reference.
Reference MATLAB result: keff = 1.022173.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CoreGeometry:
    """1D geometry for a PWR subassembly."""

    bot_refl_height: float = 50.0   # cm
    fuel_height: float = 300.0      # cm
    top_refl_height: float = 50.0   # cm
    f2f: float = 21.61              # flat-to-flat distance (cm)
    dz: float = 5.0                 # axial node height (cm)

    @property
    def n_refl_bot(self) -> int:
        return int(self.bot_refl_height / self.dz)

    @property
    def n_fuel(self) -> int:
        return int(self.fuel_height / self.dz)

    @property
    def n_refl_top(self) -> int:
        return int(self.top_refl_height / self.dz)

    @property
    def n_nodes(self) -> int:
        return self.n_refl_bot + self.n_fuel + self.n_refl_top

    @property
    def n_edges(self) -> int:
        return self.n_nodes + 1

    @property
    def az(self) -> float:
        """Cross-sectional area of one subassembly (cm^2)."""
        return self.f2f**2

    @property
    def dv(self) -> float:
        """Volume of one node (cm^3)."""
        return self.az * self.dz


@dataclass
class TwoGroupXS:
    """Two-group macroscopic cross sections for a material."""

    transport: np.ndarray    # (2,) transport XS
    absorption: np.ndarray   # (2,) absorption XS
    fission: np.ndarray      # (2,) fission XS
    production: np.ndarray   # (2,) production XS (nu*sigma_f)
    chi: np.ndarray          # (2,) fission spectrum
    scattering: np.ndarray   # (2,) down-scattering (fast->thermal)


@dataclass
class DiffusionResult:
    """Results of a 1D diffusion calculation."""

    keff: float
    flux: np.ndarray          # (2, n_nodes) group fluxes normalized to power
    current: np.ndarray       # (2, n_edges) net currents
    z_nodes: np.ndarray       # (n_nodes,) node center positions
    z_edges: np.ndarray       # (n_edges,) edge positions
    geometry: CoreGeometry


# ---------------------------------------------------------------------------
# Default cross sections
# ---------------------------------------------------------------------------

def _default_xs() -> tuple[TwoGroupXS, TwoGroupXS]:
    """Default cross sections matching MATLAB CORE1D.m."""
    reflector = TwoGroupXS(
        transport=np.array([0.3416, 0.9431]),
        absorption=np.array([0.0029, 0.0933]),
        fission=np.array([0.0, 0.0]),
        production=np.array([0.0, 0.0]),
        chi=np.array([0.0, 0.0]),
        scattering=np.array([2.4673e-04, 0.0]),
    )
    fuel = TwoGroupXS(
        transport=np.array([0.2181, 0.7850]),
        absorption=np.array([0.0096, 0.0959]),
        fission=np.array([0.0024, 0.0489]),
        production=np.array([0.0061, 0.1211]),
        chi=np.array([1.0, 0.0]),
        scattering=np.array([0.0160, 0.0]),
    )
    return reflector, fuel


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_diffusion_1d(
    geom: CoreGeometry | None = None,
    reflector_xs: TwoGroupXS | None = None,
    fuel_xs: TwoGroupXS | None = None,
    power_W: float = 1.76752e7,
    errtol: float = 1e-6,
    maxiter: int = 2000,
) -> DiffusionResult:
    """Solve the 1D two-group diffusion eigenvalue problem.

    Parameters
    ----------
    geom : CoreGeometry (default: standard PWR SA).
    reflector_xs, fuel_xs : TwoGroupXS (default: from MATLAB CORE1D.m).
    power_W : float — target power for flux normalization (W).
    """
    if geom is None:
        geom = CoreGeometry()
    if reflector_xs is None or fuel_xs is None:
        reflector_xs, fuel_xs = _default_xs()

    ng = 2
    nn = geom.n_nodes
    ne = geom.n_edges
    dz = geom.dz
    dv = geom.dv

    # Build node-wise cross sections: (2, n_nodes)
    def _tile(refl_val, fuel_val):
        return np.hstack([
            np.tile(refl_val[:, None], geom.n_refl_bot),
            np.tile(fuel_val[:, None], geom.n_fuel),
            np.tile(refl_val[:, None], geom.n_refl_top),
        ])

    sig_t = _tile(reflector_xs.transport, fuel_xs.transport)
    sig_a = _tile(reflector_xs.absorption, fuel_xs.absorption)
    sig_f = _tile(reflector_xs.fission, fuel_xs.fission)
    sig_p = _tile(reflector_xs.production, fuel_xs.production)
    chi = _tile(reflector_xs.chi, fuel_xs.chi)
    sig_s = _tile(reflector_xs.scattering, fuel_xs.scattering)

    # Edge-interpolated transport XS for diffusion coefficient
    sig_t_edges = np.zeros((ng, ne))
    for ig in range(ng):
        sig_t_edges[ig, 0] = sig_t[ig, 0]
        sig_t_edges[ig, -1] = sig_t[ig, -1]
        sig_t_edges[ig, 1:-1] = 0.5 * (sig_t[ig, :-1] + sig_t[ig, 1:])

    D = 1.0 / (3.0 * sig_t_edges)  # (2, n_edges) diffusion coefficient

    # Energy per fission (J)
    e_per_fission = np.array([0.3213e-10, 0.3206e-10])

    # Linear operator A*x
    def matvec(solution):
        fi = solution.reshape(ng, nn)

        # Gradient with vacuum BCs (phi=0 at boundaries)
        dfidz = np.zeros((ng, ne))
        dfidz[:, 0] = fi[:, 0] / (0.5 * dz)
        dfidz[:, -1] = -fi[:, -1] / (0.5 * dz)
        dfidz[:, 1:-1] = np.diff(fi, axis=1) / dz

        J = -D * dfidz

        # A*fi = div(J)/dz + (SigA + SigS)*fi - SigS_flipped * fi_flipped
        Ax = np.diff(J, axis=1) / dz + (sig_a + sig_s) * fi - sig_s[::-1, :] * fi[::-1, :]

        return Ax.ravel()

    A_op = LinearOperator(shape=(ng * nn, ng * nn), matvec=matvec, dtype=float)

    # Outer (power) iteration
    fi = np.ones((ng, nn))
    solution = fi.ravel()

    n_outer = 0
    while True:
        # Production and absorption rates (volume-integrated per node)
        p_rate = (sig_p * fi * dv).sum(axis=0)  # (n_nodes,)
        a_rate = (sig_a * fi * dv).sum(axis=0)

        # Current and leakage
        dfidz = np.zeros((ng, ne))
        dfidz[:, 0] = fi[:, 0] / (0.5 * dz)
        dfidz[:, -1] = -fi[:, -1] / (0.5 * dz)
        dfidz[:, 1:-1] = np.diff(fi, axis=1) / dz
        J = -D * dfidz
        l_rate = (-J[:, 0] * geom.az + J[:, -1] * geom.az).sum()

        keff = p_rate.sum() / (a_rate.sum() + l_rate)

        # RHS: chi * pRate/keff / keff  (matching MATLAB's double /keff)
        rhs = (chi * np.tile(p_rate / keff, (ng, 1)) / keff).ravel()

        # BiCGSTAB inner solve
        guess = solution.copy()
        solution, info = bicgstab(A_op, rhs, x0=guess, rtol=errtol, maxiter=maxiter)

        fi = solution.reshape(ng, nn)

        # Normalize flux to target power
        f_rate = sig_f * fi * dv
        pow_calc = (f_rate.sum(axis=1) * e_per_fission).sum()
        fi *= power_W / pow_calc
        solution = fi.ravel()

        n_outer += 1

        # Check convergence: solution barely changed from guess
        rel_change = np.linalg.norm(solution - guess) / np.linalg.norm(solution)
        print(f"  keff = {keff:9.6f}  #outer = {n_outer:3d}  rel_change = {rel_change:.2e}")

        if rel_change < 1e-5 and n_outer > 1:
            break
        if n_outer >= 200:
            print("  Warning: max outer iterations reached")
            break

    # Final current
    dfidz = np.zeros((ng, ne))
    dfidz[:, 0] = fi[:, 0] / (0.5 * dz)
    dfidz[:, -1] = -fi[:, -1] / (0.5 * dz)
    dfidz[:, 1:-1] = np.diff(fi, axis=1) / dz
    J = -D * dfidz

    z_nodes = np.arange(dz / 2, dz / 2 + dz * nn, dz)
    z_edges = np.arange(0, dz * ne, dz)

    return DiffusionResult(
        keff=keff,
        flux=fi,
        current=J,
        z_nodes=z_nodes,
        z_edges=z_edges,
        geometry=geom,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_flux(result: DiffusionResult, output_dir: Path) -> None:
    """Plot fast and thermal flux profiles."""
    fig, ax = plt.subplots()
    ax.plot(result.z_nodes, result.flux[0, :], "-or", label="Fast", markersize=3)
    ax.plot(result.z_nodes, result.flux[1, :], "-ob", label="Thermal", markersize=3)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"Neutron flux (n/cm$^2$s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "DIF_01_flux.pdf")
    plt.close(fig)


def plot_current(result: DiffusionResult, output_dir: Path) -> None:
    """Plot fast and thermal net current profiles."""
    fig, ax = plt.subplots()
    ax.plot(result.z_edges, result.current[0, :], "-or", label="Fast", markersize=3)
    ax.plot(result.z_edges, result.current[1, :], "-ob", label="Thermal", markersize=3)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"Neutron net current (n/cm$^2$s)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "DIF_02_current.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("1D TWO-GROUP DIFFUSION — PWR SUBASSEMBLY")
    print("=" * 70)

    result = solve_diffusion_1d()

    print(f"\n  keff = {result.keff:.6f}  (MATLAB reference: 1.022173)")
    match = "YES" if abs(result.keff - 1.022173) < 1e-4 else "NO"
    print(f"  Match: {match}")

    # Plots
    output = Path("05_results")
    output.mkdir(parents=True, exist_ok=True)

    plot_flux(result, output)
    plot_current(result, output)

    print(f"\n  Plots saved to {output.resolve()}/")


if __name__ == "__main__":
    main()
