"""2D Discrete Ordinates (SN) neutron transport solver for a PWR pin cell.

Solves the multi-group transport equation on a 2D Cartesian mesh with
reflective boundary conditions using Lebedev angular quadrature and
BiCGSTAB inner iterations.

Port of MATLAB ``discreteOrdinatesPWR.m`` + ``funDO.m`` + ``convert.m``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import lebedev_rule
from scipy.sparse.linalg import LinearOperator, bicgstab

from data.macro_xs.mixture import Mixture
from data.micro_xs.isotope import NG


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_equation_map(geom: PinCellGeometry, quad: Quadrature) -> EquationMap:
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
        n_unknowns=n_eq * NG,
        ordinate=np.array(ords, dtype=int),
        ix=np.array(ixs, dtype=int),
        iy=np.array(iys, dtype=int),
    )


def solution_to_angular_flux(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: Quadrature,
    geom: PinCellGeometry,
) -> np.ndarray:
    """Convert 1D solution vector to 4D angular flux array.

    Port of MATLAB ``convert.m``.

    Returns
    -------
    fi : (NG, N, nx, ny) array
    """
    fi = np.zeros((NG, quad.N, geom.nx, geom.ny))

    # Scatter solution into the fi array
    flux = solution.reshape(NG, eq_map.n_eq, order='F')
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
) -> np.ndarray:
    """Linear operator A*x for the transport equation.

    Port of MATLAB ``funDO.m``.

    Parameters
    ----------
    sig_t : (nx, ny, NG) total cross section at each node.
    """
    fi = solution_to_angular_flux(solution, eq_map, quad, geom)

    lhs = np.empty((NG, eq_map.n_eq))
    for k in range(eq_map.n_eq):
        n, ix, iy = eq_map.ordinate[k], eq_map.ix[k], eq_map.iy[k]
        dfidx, dfidy = _compute_gradients(fi, n, ix, iy, quad, geom)
        lhs[:, k] = (
            quad.mu_x[n] * dfidx
            + quad.mu_y[n] * dfidy
            + sig_t[ix, iy, :] * fi[:, n, ix, iy]
        )

    return lhs.ravel(order='F')


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

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

    eg = materials[2].eg  # energy group boundaries from fuel

    # --- Pre-compute per-node cross sections ---
    sig_a = np.empty((geom.nx, geom.ny, NG))
    sig_t = np.empty((geom.nx, geom.ny, NG))
    sig_p = np.empty((geom.nx, geom.ny, NG))
    chi_node = np.empty((geom.nx, geom.ny, NG))

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
            sig_p[ix, iy, :] = m.SigP if m.SigP.ndim > 0 and len(m.SigP) == NG else np.zeros(NG)
            chi_node[ix, iy, :] = m.chi

            # Scattering matrices for each Legendre order
            sig_s_list = []
            for j in range(quad.L + 1):
                if j < len(m.SigS):
                    sig_s_list.append(m.SigS[j])
                else:
                    from scipy.sparse import csr_matrix
                    sig_s_list.append(csr_matrix((NG, NG)))
            sig_s_node[ix][iy] = sig_s_list
            sig2_node[ix][iy] = m.Sig2

    # --- Build equation map ---
    eq_map = build_equation_map(geom, quad)
    print(f"  Equations: {eq_map.n_eq} angular x {NG} groups = {eq_map.n_unknowns} unknowns")

    # --- Build LinearOperator for BiCGSTAB ---
    def matvec(x):
        return transport_operator(x, eq_map, quad, geom, sig_t)

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
        fi = solution_to_angular_flux(solution, eq_map, quad, geom)

        # Legendre flux moments at each node
        # fiL[ix][iy] shape (NG, L+1, 2*L+1)
        scalar_flux = np.zeros((geom.nx, geom.ny, NG))
        fiL = [[None for _ in range(geom.ny)] for _ in range(geom.nx)]

        for iy in range(geom.ny):
            for ix in range(geom.nx):
                fl = np.zeros((NG, quad.L + 1, 2 * quad.L + 1))
                for j in range(quad.L + 1):
                    for m_idx in range(-j, j + 1):
                        col = j + m_idx
                        s = np.zeros(NG)
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
        rhs = np.zeros((NG, eq_map.n_eq))
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
                    qS = np.zeros(NG)
                    for j in range(quad.L + 1):
                        s = np.zeros(NG)
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
    fi_final = solution_to_angular_flux(solution, eq_map, quad, geom)
    # Recompute scalar flux from final solution
    for iy in range(geom.ny):
        for ix in range(geom.nx):
            scalar_flux[ix, iy, :] = np.sum(
                fi_final[:, :, ix, iy] * quad.weights[None, :], axis=1
            )

    vol_fuel = geom.volume[geom.mat_map == 2].sum()
    vol_clad = geom.volume[geom.mat_map == 1].sum()
    vol_cool = geom.volume[geom.mat_map == 0].sum()

    flux_fuel = np.zeros(NG)
    flux_clad = np.zeros(NG)
    flux_cool = np.zeros(NG)

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
