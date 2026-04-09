#!/usr/bin/env python3
"""2D Discrete Ordinates (SN) neutron transport solver for a PWR pin cell.

Solves the multi-group transport equation on a 2D Cartesian mesh with
reflective boundary conditions using Lebedev angular quadrature and
BiCGSTAB inner iterations.

This script contains the full solver (port of MATLAB discreteOrdinatesPWR.m
+ funDO.m + convert.m), all plotting utilities, and a main() that runs
the demonstration case.

Reference MATLAB result:
    keff = 1.04188  (110 Lebedev ordinates, P0 scattering)

Dependencies:
    pip install numpy scipy matplotlib
    ORPHEUS data package (orpheus.data)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.integrate import lebedev_rule
from scipy.sparse.linalg import LinearOperator, bicgstab

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class PinCellGeometry:
    """2D Cartesian mesh for a PWR pin cell (quarter-symmetry)."""

    nx: int
    ny: int
    delta: float  # mesh step (cm)
    mat_map: np.ndarray  # (nx, ny) int — 0=coolant, 1=clad, 2=fuel
    volume: np.ndarray  # (nx, ny) node volumes (cm^2)

    @classmethod
    def default_pwr(cls) -> PinCellGeometry:
        """Standard 10x2 mesh: 5 fuel + 1 clad + 4 coolant nodes."""
        nx, ny, delta = 10, 2, 0.2

        vol = np.full((nx, ny), delta**2)
        vol[0, :] /= 2
        vol[-1, :] /= 2
        vol[:, 0] /= 2
        vol[:, -1] /= 2

        mat = np.array(
            [[2, 2, 2, 2, 2, 1, 0, 0, 0, 0],
             [2, 2, 2, 2, 2, 1, 0, 0, 0, 0]], dtype=int
        ).T  # shape (10, 2)

        return cls(nx=nx, ny=ny, delta=delta, mat_map=mat, volume=vol)


@dataclass
class Quadrature:
    """Lebedev angular quadrature on the unit sphere."""

    mu_x: np.ndarray   # (N,) direction cosines
    mu_y: np.ndarray
    mu_z: np.ndarray
    weights: np.ndarray  # (N,) summing to 4*pi
    ref_x: np.ndarray   # (N,) int — index of X-reflected partner
    ref_y: np.ndarray   # (N,) int — index of Y-reflected partner
    ref_z: np.ndarray   # (N,) int — index of Z-reflected partner
    R: np.ndarray        # (N, L+1, 2*L+1) spherical harmonics
    N: int = 0
    L: int = 0

    @classmethod
    def lebedev(cls, order: int = 17, L: int = 0) -> Quadrature:
        """Build quadrature from scipy Lebedev rule.

        Parameters
        ----------
        order : int
            Lebedev order (17 gives 110 points, matching MATLAB default).
        L : int
            Scattering anisotropy order (0 = P0 isotropic).
        """
        pts, w = lebedev_rule(order)
        mu_x, mu_y, mu_z = pts[0], pts[1], pts[2]
        n_pts = len(w)

        # Reflective direction indices — vectorised search
        ref_x = _find_reflections(-mu_x, mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_y = _find_reflections(mu_x, -mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_z = _find_reflections(mu_x, mu_y, -mu_z, mu_x, mu_y, mu_z)

        # Spherical harmonics for each ordinate
        R = np.zeros((n_pts, L + 1, 2 * L + 1))
        for n in range(n_pts):
            for j in range(L + 1):
                for m in range(-j, j + 1):
                    col = j + m  # column index in [0, 2*L]
                    if j == 0 and m == 0:
                        R[n, j, col] = 1.0
                    elif j == 1 and m == -1:
                        R[n, j, col] = mu_z[n]
                    elif j == 1 and m == 0:
                        R[n, j, col] = mu_x[n]
                    elif j == 1 and m == 1:
                        R[n, j, col] = mu_y[n]

        return cls(mu_x=mu_x, mu_y=mu_y, mu_z=mu_z, weights=w,
                   ref_x=ref_x, ref_y=ref_y, ref_z=ref_z,
                   R=R, N=n_pts, L=L)


def _find_reflections(
    target_x: np.ndarray, target_y: np.ndarray, target_z: np.ndarray,
    all_x: np.ndarray, all_y: np.ndarray, all_z: np.ndarray,
) -> np.ndarray:
    """Find index of nearest match for each target in the full set."""
    n = len(target_x)
    dist = (
        (target_x[:, None] - all_x[None, :])**2
        + (target_y[:, None] - all_y[None, :])**2
        + (target_z[:, None] - all_z[None, :])**2
    )
    return np.argmin(dist, axis=1)


@dataclass
class EquationMap:
    """Pre-computed mapping between 1D solution vector and 4D angular flux."""

    n_eq: int           # number of angular equations
    n_unknowns: int     # n_eq * NG
    ordinate: np.ndarray  # (n_eq,) int — ordinate index
    ix: np.ndarray        # (n_eq,) int — x-node index
    iy: np.ndarray        # (n_eq,) int — y-node index


@dataclass
class DOParams:
    """Solver parameters for discrete ordinates."""

    max_outer: int = 200
    bicgstab_tol: float = 1e-4
    bicgstab_maxiter: int = 2000
    L: int = 0  # scattering anisotropy (0=P0, 1=P1)


@dataclass
class DOResult:
    """Results of a discrete ordinates calculation."""

    keff: float
    keff_history: list[float]
    residual_history: list[float]
    flux_fuel: np.ndarray     # (NG,) volume-averaged scalar flux in fuel
    flux_clad: np.ndarray     # (NG,) volume-averaged scalar flux in clad
    flux_cool: np.ndarray     # (NG,) volume-averaged scalar flux in coolant
    scalar_flux: np.ndarray   # (nx, ny, NG) scalar flux at each node
    geometry: PinCellGeometry
    eg: np.ndarray            # (NG+1,) energy group boundaries
    elapsed_seconds: float


# ===========================================================================
# Core functions
# ===========================================================================

def build_equation_map(geom: PinCellGeometry, quad: Quadrature, ng: int) -> EquationMap:
    """Identify which (ordinate, ix, iy) combinations are unknowns.

    Filter: mu_z >= 0, and NOT incoming at reflective boundaries.
    """
    ords, ixs, iys = [], [], []
    for iy in range(geom.ny):
        for ix in range(geom.nx):
            for n in range(quad.N):
                if quad.mu_z[n] < 0:
                    continue
                if ix == 0 and quad.mu_x[n] > 0:
                    continue
                if ix == geom.nx - 1 and quad.mu_x[n] < 0:
                    continue
                if iy == 0 and quad.mu_y[n] > 0:
                    continue
                if iy == geom.ny - 1 and quad.mu_y[n] < 0:
                    continue
                ords.append(n)
                ixs.append(ix)
                iys.append(iy)

    n_eq = len(ords)
    return EquationMap(
        n_eq=n_eq,
        n_unknowns=n_eq * ng,
        ordinate=np.array(ords, dtype=int),
        ix=np.array(ixs, dtype=int),
        iy=np.array(iys, dtype=int),
    )


def solution_to_angular_flux(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: Quadrature,
    geom: PinCellGeometry,
    ng: int,
) -> np.ndarray:
    """Convert 1D solution vector to 4D angular flux array.

    Port of MATLAB ``convert.m``.

    Returns
    -------
    fi : (NG, N, nx, ny) array
    """
    fi = np.zeros((ng, quad.N, geom.nx, geom.ny))

    # Scatter solution into the fi array
    flux = solution.reshape(ng, eq_map.n_eq, order='F')
    for k in range(eq_map.n_eq):
        fi[:, eq_map.ordinate[k], eq_map.ix[k], eq_map.iy[k]] = flux[:, k]

    # Z-reflection: directions with mu_z < 0 copy from their reflection
    for n in range(quad.N):
        if quad.mu_z[n] < 0:
            fi[:, n, :, :] = fi[:, quad.ref_z[n], :, :]

    # X reflective boundary conditions
    for n in range(quad.N):
        if quad.mu_x[n] > 0:
            fi[:, n, 0, :] = fi[:, quad.ref_x[n], 0, :]
        if quad.mu_x[n] < 0:
            fi[:, n, -1, :] = fi[:, quad.ref_x[n], -1, :]

    # Y reflective boundary conditions
    for n in range(quad.N):
        if quad.mu_y[n] > 0:
            fi[:, n, :, 0] = fi[:, quad.ref_y[n], :, 0]
        if quad.mu_y[n] < 0:
            fi[:, n, :, -1] = fi[:, quad.ref_y[n], :, -1]

    return fi


def _compute_gradients(
    fi: np.ndarray,
    n: int, ix: int, iy: int,
    quad: Quadrature,
    geom: PinCellGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Diamond-scheme gradients with reflective BCs.

    Port of the ``gradients`` subfunction in ``funDO.m``.
    Returns (dfidx, dfidy), each shape (NG,).
    """
    # X gradient
    if quad.mu_x[n] > 0:
        if ix == 0:
            dfix = fi[:, quad.ref_x[n], ix, iy] - fi[:, quad.ref_x[n], ix + 1, iy]
        else:
            dfix = fi[:, n, ix, iy] - fi[:, n, ix - 1, iy]
    else:
        if ix == geom.nx - 1:
            dfix = fi[:, quad.ref_x[n], ix - 1, iy] - fi[:, quad.ref_x[n], ix, iy]
        else:
            dfix = fi[:, n, ix + 1, iy] - fi[:, n, ix, iy]

    # Y gradient
    if quad.mu_y[n] > 0:
        if iy == 0:
            dfiy = fi[:, quad.ref_y[n], ix, iy] - fi[:, quad.ref_y[n], ix, iy + 1]
        else:
            dfiy = fi[:, n, ix, iy] - fi[:, n, ix, iy - 1]
    else:
        if iy == geom.ny - 1:
            dfiy = fi[:, quad.ref_y[n], ix, iy - 1] - fi[:, quad.ref_y[n], ix, iy]
        else:
            dfiy = fi[:, n, ix, iy + 1] - fi[:, n, ix, iy]

    return dfix / geom.delta, dfiy / geom.delta


def transport_operator(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: Quadrature,
    geom: PinCellGeometry,
    sig_t: np.ndarray,
    ng: int,
) -> np.ndarray:
    """Linear operator A*x for the transport equation.

    Port of MATLAB ``funDO.m``.

    Parameters
    ----------
    sig_t : (nx, ny, NG) total cross section at each node.
    """
    fi = solution_to_angular_flux(solution, eq_map, quad, geom, ng)

    lhs = np.empty((ng, eq_map.n_eq))
    for k in range(eq_map.n_eq):
        n, ix, iy = eq_map.ordinate[k], eq_map.ix[k], eq_map.iy[k]
        dfidx, dfidy = _compute_gradients(fi, n, ix, iy, quad, geom)
        lhs[:, k] = (
            quad.mu_x[n] * dfidx
            + quad.mu_y[n] * dfidy
            + sig_t[ix, iy, :] * fi[:, n, ix, iy]
        )

    return lhs.ravel(order='F')


# ===========================================================================
# Main solver
# ===========================================================================

def solve_discrete_ordinates(
    materials: dict[int, Mixture],
    geom: PinCellGeometry | None = None,
    quad: Quadrature | None = None,
    params: DOParams | None = None,
) -> DOResult:
    """Run the 2D discrete ordinates transport calculation.

    Parameters
    ----------
    materials : dict mapping material ID (0=cool, 1=clad, 2=fuel) to Mixture.
    geom : PinCellGeometry (default: standard 10x2 PWR mesh).
    quad : Quadrature (default: Lebedev order 17, 110 points, P0).
    params : DOParams (default: 200 outer iterations, tol=1e-4).
    """
    t_start = time.perf_counter()

    if geom is None:
        geom = PinCellGeometry.default_pwr()
    if params is None:
        params = DOParams()
    if quad is None:
        quad = Quadrature.lebedev(order=17, L=params.L)

    _any_mat = next(iter(materials.values()))
    eg = _any_mat.eg
    ng = _any_mat.ng

    # --- Pre-compute per-node cross sections ---
    sig_a = np.empty((geom.nx, geom.ny, ng))
    sig_t = np.empty((geom.nx, geom.ny, ng))
    sig_p = np.empty((geom.nx, geom.ny, ng))
    chi_node = np.empty((geom.nx, geom.ny, ng))

    # Store references to sparse matrices per node
    sig_s_node: list[list[list]] = [
        [[] for _ in range(geom.ny)] for _ in range(geom.nx)
    ]
    sig2_node: list[list] = [
        [None for _ in range(geom.ny)] for _ in range(geom.nx)
    ]

    for iy in range(geom.ny):
        for ix in range(geom.nx):
            m = materials[geom.mat_map[ix, iy]]
            sig2_colsum = np.array(m.Sig2.sum(axis=1)).ravel()
            sig_a[ix, iy, :] = m.SigF + m.SigC + m.SigL + sig2_colsum
            sig_t[ix, iy, :] = sig_a[ix, iy, :] + np.array(m.SigS[0].sum(axis=1)).ravel()
            sig_p[ix, iy, :] = m.SigP if m.SigP.ndim > 0 and len(m.SigP) == ng else np.zeros(ng)
            chi_node[ix, iy, :] = m.chi

            # Scattering matrices for each Legendre order
            sig_s_list = []
            for j in range(quad.L + 1):
                if j < len(m.SigS):
                    sig_s_list.append(m.SigS[j])
                else:
                    from scipy.sparse import csr_matrix
                    sig_s_list.append(csr_matrix((ng, ng)))
            sig_s_node[ix][iy] = sig_s_list
            sig2_node[ix][iy] = m.Sig2

    # --- Build equation map ---
    eq_map = build_equation_map(geom, quad, ng)
    print(f"  Equations: {eq_map.n_eq} angular x {ng} groups = {eq_map.n_unknowns} unknowns")

    # --- Build LinearOperator for BiCGSTAB ---
    def matvec(x):
        return transport_operator(x, eq_map, quad, geom, sig_t, ng)

    A_op = LinearOperator(
        shape=(eq_map.n_unknowns, eq_map.n_unknowns),
        matvec=matvec,
        dtype=float,
    )

    # --- Outer iteration loop ---
    keff_history: list[float] = []
    residual_history: list[float] = []
    solution = np.ones(eq_map.n_unknowns)
    four_pi = 4.0 * np.pi

    for n_iter in range(1, params.max_outer + 1):
        guess = solution.copy()

        # Convert to angular flux and compute scalar flux
        fi = solution_to_angular_flux(solution, eq_map, quad, geom, ng)

        # Legendre flux moments at each node
        # fiL[ix][iy] shape (ng, L+1, 2*L+1)
        scalar_flux = np.zeros((geom.nx, geom.ny, ng))
        fiL = [[None for _ in range(geom.ny)] for _ in range(geom.nx)]

        for iy in range(geom.ny):
            for ix in range(geom.nx):
                fl = np.zeros((ng, quad.L + 1, 2 * quad.L + 1))
                for j in range(quad.L + 1):
                    for m_idx in range(-j, j + 1):
                        col = j + m_idx
                        s = np.zeros(ng)
                        for n in range(quad.N):
                            s += fi[:, n, ix, iy] * quad.R[n, j, col] * quad.weights[n]
                        fl[:, j, col] = s
                fiL[ix][iy] = fl
                scalar_flux[ix, iy, :] = fl[:, 0, 0]  # L=0, m=0 moment = scalar flux

        # keff = production / absorption (volume-weighted)
        p_rate = 0.0
        a_rate = 0.0
        for iy in range(geom.ny):
            for ix in range(geom.nx):
                v = geom.volume[ix, iy]
                FI = scalar_flux[ix, iy, :]
                sig2_cs = np.array(sig2_node[ix][iy].sum(axis=1)).ravel()
                p_rate += (sig_p[ix, iy, :] + 2 * sig2_cs) @ FI * v
                a_rate += sig_a[ix, iy, :] @ FI * v

        keff = p_rate / a_rate
        keff_history.append(keff)

        # Build RHS source vector
        rhs = np.zeros((ng, eq_map.n_eq))
        eq_idx = 0
        for iy in range(geom.ny):
            for ix in range(geom.nx):
                FI = scalar_flux[ix, iy, :]

                # Fission source (isotropic)
                qF = chi_node[ix, iy, :] * (sig_p[ix, iy, :] @ FI) / keff / four_pi
                # (n,2n) source (isotropic)
                q2 = 2.0 * (sig2_node[ix][iy].T @ FI) / four_pi

                for n in range(quad.N):
                    if quad.mu_z[n] < 0:
                        continue
                    if ix == 0 and quad.mu_x[n] > 0:
                        continue
                    if ix == geom.nx - 1 and quad.mu_x[n] < 0:
                        continue
                    if iy == 0 and quad.mu_y[n] > 0:
                        continue
                    if iy == geom.ny - 1 and quad.mu_y[n] < 0:
                        continue

                    # Scattering source
                    qS = np.zeros(ng)
                    for j in range(quad.L + 1):
                        s = np.zeros(ng)
                        for m_idx in range(-j, j + 1):
                            col = j + m_idx
                            s += fiL[ix][iy][:, j, col] * quad.R[n, j, col]
                        qS += (2 * j + 1) * (sig_s_node[ix][iy][j].T @ s) / four_pi

                    rhs[:, eq_idx] = qF + q2 + qS
                    eq_idx += 1

        rhs_vec = rhs.ravel(order='F')

        # Inner solve with BiCGSTAB
        solution, info = bicgstab(A_op, rhs_vec, x0=guess,
                                  rtol=params.bicgstab_tol,
                                  maxiter=params.bicgstab_maxiter)

        # Compute residual
        res = np.linalg.norm(A_op @ solution - rhs_vec) / np.linalg.norm(rhs_vec)
        residual_history.append(res)

        print(f"  keff = {keff:9.5f}  #outer = {n_iter:3d}  residual = {res:11.5e}")

        if info == 0 and res < params.bicgstab_tol:
            # Check if BiCGSTAB converged instantly (solution already satisfies)
            if n_iter > 1 and abs(keff_history[-1] - keff_history[-2]) < 1e-7:
                print("  Converged.")
                break

    # --- Post-processing: volume-averaged spectra ---
    fi_final = solution_to_angular_flux(solution, eq_map, quad, geom, ng)
    # Recompute scalar flux from final solution
    for iy in range(geom.ny):
        for ix in range(geom.nx):
            scalar_flux[ix, iy, :] = np.sum(
                fi_final[:, :, ix, iy] * quad.weights[None, :], axis=1
            )

    vol_fuel = geom.volume[geom.mat_map == 2].sum()
    vol_clad = geom.volume[geom.mat_map == 1].sum()
    vol_cool = geom.volume[geom.mat_map == 0].sum()

    flux_fuel = np.zeros(ng)
    flux_clad = np.zeros(ng)
    flux_cool = np.zeros(ng)

    for iy in range(geom.ny):
        for ix in range(geom.nx):
            v = geom.volume[ix, iy]
            FI = scalar_flux[ix, iy, :]
            mat_id = geom.mat_map[ix, iy]
            if mat_id == 2:
                flux_fuel += FI * v / vol_fuel
            elif mat_id == 1:
                flux_clad += FI * v / vol_clad
            else:
                flux_cool += FI * v / vol_cool

    elapsed = time.perf_counter() - t_start
    print(f"  Elapsed: {elapsed:.1f}s")

    return DOResult(
        keff=keff_history[-1],
        keff_history=keff_history,
        residual_history=residual_history,
        flux_fuel=flux_fuel,
        flux_clad=flux_clad,
        flux_cool=flux_cool,
        scalar_flux=scalar_flux,
        geometry=geom,
        eg=eg,
        elapsed_seconds=elapsed,
    )


# ===========================================================================
# Plotting
# ===========================================================================

def _plot_2d_field(
    nx: int, ny: int, delta: float,
    field: np.ndarray,
    title: str,
    filepath: Path,
) -> None:
    """Generic 2D colored-rectangle plot (port of MATLAB plot2D.m)."""
    aspect_ratio = max(ny / nx, 0.3)
    fig, ax = plt.subplots(figsize=(8, max(3, 8 * aspect_ratio)))
    ax.set_aspect("equal")
    ax.set_xlim(0, delta * (nx - 1))
    ax.set_ylim(-delta * (ny - 1), 0)
    ax.set_xlabel("Distance from the cell centre (cm)")
    ax.set_ylabel("Distance from the cell centre (cm)")
    ax.set_title(title)

    fmin, fmax = field.min(), field.max()

    y = 0.0
    for iy in range(ny):
        height = delta - delta / 2 * (iy == 0 or iy == ny - 1)
        y -= height
        x = 0.0
        for ix in range(nx):
            width = delta - delta / 2 * (ix == 0 or ix == nx - 1)
            c = (field[ix, iy] - fmin) / max(fmax - fmin, 1e-30)
            ax.add_patch(Rectangle(
                (x, y), width, height,
                facecolor=(c, 0, 1 - c), edgecolor=(0.5, 0.5, 0.5), linewidth=0.2,
            ))
            x += width

    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def plot_mesh_2d(
    geom: PinCellGeometry,
    output_dir: Path | str = ".",
    filename: str = "DO_01_mesh.pdf",
) -> None:
    """2D colored rectangle plot of the material map (port of plot2D.m)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_2d_field(geom.nx, geom.ny, geom.delta,
                   geom.mat_map.astype(float), "Unit cell: materials",
                   output_dir / filename)


def plot_do_convergence(
    result: DOResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot keff convergence and residual history."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # keff history
    fig, ax = plt.subplots()
    ax.plot(range(1, len(result.keff_history) + 1), result.keff_history, "-or", markersize=3)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("k-effective")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "DO_02_keff.pdf")
    plt.close(fig)

    # Residual history
    fig, ax = plt.subplots()
    ax.semilogy(range(1, len(result.residual_history) + 1), result.residual_history, "-or", markersize=3)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Relative residual error")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "DO_03_residual.pdf")
    plt.close(fig)


def plot_do_spectra(
    result: DOResult,
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
    fig.savefig(output_dir / "DO_04_flux_lethargy.pdf")
    plt.close(fig)


def plot_do_spatial_flux(
    result: DOResult,
    output_dir: Path | str = ".",
) -> None:
    """Plot thermal/resonance/fast flux along cell centerline and 2D."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geom = result.geometry
    x = np.arange(geom.nx) * geom.delta

    # Group ranges for centerline plot
    FI_T = result.scalar_flux[:, 0, :50].sum(axis=1)    # thermal < 1 eV
    FI_R = result.scalar_flux[:, 0, 50:287].sum(axis=1)  # resonance < 0.1 MeV
    FI_F = result.scalar_flux[:, 0, 287:].sum(axis=1)    # fast > 0.1 MeV

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

    # 2D flux distributions (different group ranges)
    fun_T = result.scalar_flux[:, :, :50].sum(axis=2)
    fun_R = result.scalar_flux[:, :, 50:355].sum(axis=2)
    fun_F = result.scalar_flux[:, :, 355:].sum(axis=2)

    _plot_2d_field(geom.nx, geom.ny, geom.delta, fun_T,
                   "Thermal flux distribution", output_dir / "DO_06_flux_thermal.pdf")
    _plot_2d_field(geom.nx, geom.ny, geom.delta, fun_R,
                   "Resonance flux distribution", output_dir / "DO_07_flux_resonance.pdf")
    _plot_2d_field(geom.nx, geom.ny, geom.delta, fun_F,
                   "Fast flux distribution", output_dir / "DO_08_flux_fast.pdf")


# ===========================================================================
# Demo run
# ===========================================================================

OUTPUT = Path("02_results")


def main():
    print("=" * 70)
    print("DISCRETE ORDINATES — PWR PIN CELL (2D)")
    print("=" * 70)

    # 1. Build per-material macroscopic cross sections
    fuel = uo2_fuel(temp_K=900)
    clad = zircaloy_clad(temp_K=600)
    cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
    materials = {2: fuel, 1: clad, 0: cool}

    # 2. Set up geometry and angular quadrature
    geom = PinCellGeometry.default_pwr()
    params = DOParams(L=0)
    quad = Quadrature.lebedev(order=17, L=params.L)

    print(f"\n  Mesh: {geom.nx} x {geom.ny}, delta = {geom.delta} cm")
    print(f"  Ordinates: {quad.N} (Lebedev order 17)")
    print(f"  Scattering anisotropy: P{params.L}")
    print()

    # 3. Solve
    result = solve_discrete_ordinates(materials, geom, quad, params)

    # 4. Report
    print(f"\n  keff = {result.keff:.5f}  (MATLAB reference: 1.04188)")
    match = "YES" if abs(result.keff - 1.04188) < 5e-4 else "NO"
    print(f"  Match: {match}")
    print(f"  Outer iterations: {len(result.keff_history)}")
    print(f"  Wall time: {result.elapsed_seconds:.1f}s")

    # 5. Plots
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_mesh_2d(geom, OUTPUT)
    plot_do_convergence(result, OUTPUT)
    plot_do_spectra(result, OUTPUT)
    plot_do_spatial_flux(result, OUTPUT)
    print(f"\n  Plots saved to {OUTPUT.resolve()}/")


if __name__ == "__main__":
    main()
