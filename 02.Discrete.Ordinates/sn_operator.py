"""Direct transport operator for Krylov inner solves.

Provides the explicit operator T: ψ → μ·∇ψ + Σ_t·ψ  via finite
differences on a 2D Cartesian mesh with reflective BCs and Lebedev
quadrature.  Used by the ``bicgstab`` inner solver path in SNSolver.

The sweep-based solver (source iteration) inverts T implicitly via
diamond-difference sweeps.  This module forms T explicitly so that
scipy's Krylov solvers (BiCGSTAB, GMRES) can solve  T·ψ = b  directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator

from sn_quadrature import AngularQuadrature


# ═══════════════════════════════════════════════════════════════════════
# Equation map: which (ordinate, cell) pairs are unknowns
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EquationMap:
    """Mapping between 1D solution vector and 4D angular flux."""

    n_eq: int               # number of angular unknowns (per group)
    n_unknowns: int         # n_eq * ng (total scalar unknowns)
    ordinate: np.ndarray    # (n_eq,) ordinate index for each unknown
    ix: np.ndarray          # (n_eq,) x-cell index
    iy: np.ndarray          # (n_eq,) y-cell index


def build_equation_map(
    nx: int, ny: int, quad: AngularQuadrature, ng: int,
) -> EquationMap:
    """Identify which (ordinate, cell) combos are unknowns.

    Filter: mu_z >= 0 (upper hemisphere), and NOT incoming at
    reflective boundaries (those are determined by reflection).
    """
    mu_x, mu_y = quad.mu_x, quad.mu_y
    # Need mu_z — for LebedevSphere it's available, for GL1D it's 0
    mu_z = getattr(quad, 'mu_z', np.zeros(quad.N))

    ords, ixs, iys = [], [], []
    for iy in range(ny):
        for ix in range(nx):
            for n in range(quad.N):
                if mu_z[n] < -1e-15:
                    continue
                if ix == 0 and mu_x[n] > 1e-15:
                    continue
                if ix == nx - 1 and mu_x[n] < -1e-15:
                    continue
                if iy == 0 and mu_y[n] > 1e-15:
                    continue
                if iy == ny - 1 and mu_y[n] < -1e-15:
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


# ═══════════════════════════════════════════════════════════════════════
# Solution ↔ angular flux conversion
# ═══════════════════════════════════════════════════════════════════════

def solution_to_angular_flux(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    nx: int, ny: int, ng: int,
) -> np.ndarray:
    """Convert 1D solution vector to 4D angular flux (ng, N, nx, ny).

    Applies z-reflection and reflective BCs to fill the full array
    from the reduced set of unknowns.
    """
    mu_x, mu_y = quad.mu_x, quad.mu_y
    mu_z = getattr(quad, 'mu_z', np.zeros(quad.N))
    ref_x = quad.reflection_index("x")
    ref_y = quad.reflection_index("y")
    # z-reflection: need ref_z for Lebedev
    ref_z = getattr(quad, '_ref_z', np.arange(quad.N))

    fi = np.zeros((ng, quad.N, nx, ny))

    # Scatter solution into fi
    flux = solution.reshape(ng, eq_map.n_eq, order='F')
    for k in range(eq_map.n_eq):
        fi[:, eq_map.ordinate[k], eq_map.ix[k], eq_map.iy[k]] = flux[:, k]

    # Z-reflection
    for n in range(quad.N):
        if mu_z[n] < -1e-15:
            fi[:, n, :, :] = fi[:, ref_z[n], :, :]

    # X reflective BCs
    for n in range(quad.N):
        if mu_x[n] > 1e-15:
            fi[:, n, 0, :] = fi[:, ref_x[n], 0, :]
        if mu_x[n] < -1e-15:
            fi[:, n, -1, :] = fi[:, ref_x[n], -1, :]

    # Y reflective BCs
    for n in range(quad.N):
        if mu_y[n] > 1e-15:
            fi[:, n, :, 0] = fi[:, ref_y[n], :, 0]
        if mu_y[n] < -1e-15:
            fi[:, n, :, -1] = fi[:, ref_y[n], :, -1]

    return fi


def angular_flux_to_scalar(
    fi: np.ndarray, quad: AngularQuadrature, nx: int, ny: int, ng: int,
) -> np.ndarray:
    """Integrate angular flux to scalar flux: φ = Σ w_n ψ_n."""
    sf = np.zeros((nx, ny, ng))
    for iy in range(ny):
        for ix in range(nx):
            sf[ix, iy, :] = np.sum(
                fi[:, :, ix, iy] * quad.weights[None, :], axis=1,
            )
    return sf


# ═══════════════════════════════════════════════════════════════════════
# Finite-difference gradients (diamond scheme with reflective BCs)
# ═══════════════════════════════════════════════════════════════════════

def _compute_gradients(
    fi: np.ndarray,
    n: int, ix: int, iy: int,
    quad: AngularQuadrature,
    nx: int, ny: int,
    dx: np.ndarray, dy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Diamond-scheme gradients with reflective BCs.

    Returns (dfi/dx, dfi/dy), each shape (ng,).
    """
    ref_x = quad.reflection_index("x")
    ref_y = quad.reflection_index("y")
    mu_x, mu_y = quad.mu_x, quad.mu_y

    # X gradient
    if mu_x[n] > 1e-15:
        if ix == 0:
            dfix = fi[:, ref_x[n], ix, iy] - fi[:, ref_x[n], ix + 1, iy]
        else:
            dfix = fi[:, n, ix, iy] - fi[:, n, ix - 1, iy]
    elif mu_x[n] < -1e-15:
        if ix == nx - 1:
            dfix = fi[:, ref_x[n], ix - 1, iy] - fi[:, ref_x[n], ix, iy]
        else:
            dfix = fi[:, n, ix + 1, iy] - fi[:, n, ix, iy]
    else:
        dfix = np.zeros(fi.shape[0])

    # Y gradient
    if mu_y[n] > 1e-15:
        if iy == 0:
            dfiy = fi[:, ref_y[n], ix, iy] - fi[:, ref_y[n], ix, iy + 1]
        else:
            dfiy = fi[:, n, ix, iy] - fi[:, n, ix, iy - 1]
    elif mu_y[n] < -1e-15:
        if iy == ny - 1:
            dfiy = fi[:, ref_y[n], ix, iy - 1] - fi[:, ref_y[n], ix, iy]
        else:
            dfiy = fi[:, n, ix, iy + 1] - fi[:, n, ix, iy]
    else:
        dfiy = np.zeros(fi.shape[0])

    return dfix / dx[ix], dfiy / dy[iy]


# ═══════════════════════════════════════════════════════════════════════
# Transport operator  T: ψ → μ·∇ψ + Σ_t·ψ
# ═══════════════════════════════════════════════════════════════════════

def transport_operator_matvec(
    solution: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_t: np.ndarray,
    nx: int, ny: int, ng: int,
    dx: np.ndarray, dy: np.ndarray,
) -> np.ndarray:
    """Apply the streaming + collision operator T·ψ.

    Parameters
    ----------
    solution : (n_unknowns,) flattened angular flux vector.
    sig_t : (nx, ny, ng) total cross section.

    Returns
    -------
    (n_unknowns,) result of T applied to the angular flux.
    """
    fi = solution_to_angular_flux(solution, eq_map, quad, nx, ny, ng)

    lhs = np.empty((ng, eq_map.n_eq))
    for k in range(eq_map.n_eq):
        n, ix, iy = eq_map.ordinate[k], eq_map.ix[k], eq_map.iy[k]
        dfidx, dfidy = _compute_gradients(fi, n, ix, iy, quad, nx, ny, dx, dy)
        lhs[:, k] = (
            quad.mu_x[n] * dfidx
            + quad.mu_y[n] * dfidy
            + sig_t[ix, iy, :] * fi[:, n, ix, iy]
        )

    return lhs.ravel(order='F')


def build_transport_linear_operator(
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_t: np.ndarray,
    nx: int, ny: int, ng: int,
    dx: np.ndarray, dy: np.ndarray,
) -> LinearOperator:
    """Build scipy LinearOperator for T = μ·∇ + Σ_t."""
    def matvec(x):
        return transport_operator_matvec(
            x, eq_map, quad, sig_t, nx, ny, ng, dx, dy,
        )

    n = eq_map.n_unknowns
    return LinearOperator((n, n), matvec=matvec, dtype=float)


# ═══════════════════════════════════════════════════════════════════════
# RHS construction (fission + scattering + n2n, per ordinate)
# ═══════════════════════════════════════════════════════════════════════

def build_rhs(
    fission_source: np.ndarray,
    scalar_flux: np.ndarray,
    eq_map: EquationMap,
    quad: AngularQuadrature,
    sig_s0: dict[int, np.ndarray],
    sig2: dict[int, np.ndarray],
    mat_map: np.ndarray,
    nx: int, ny: int, ng: int,
) -> np.ndarray:
    """Build the RHS source vector for T·ψ = b.

    All isotropic sources are divided by 4π (the solid angle normalization
    for the angular flux equation).

    Parameters
    ----------
    fission_source : (nx, ny, ng) — already divided by 4π by the caller.
    scalar_flux : (nx, ny, ng) — current scalar flux for scattering.
    sig_s0 : dict[mat_id → (ng, ng)] P0 scattering matrices.
    sig2 : dict[mat_id → (ng, ng)] (n,2n) matrices.

    Returns
    -------
    (n_unknowns,) RHS vector.
    """
    four_pi = 4.0 * np.pi

    rhs = np.zeros((ng, eq_map.n_eq))
    eq_idx = 0
    for iy in range(ny):
        for ix in range(nx):
            mid = int(mat_map[ix, iy])
            phi_cell = scalar_flux[ix, iy, :]

            # Fission (already normalized by caller)
            qF = fission_source[ix, iy, :]

            # (n,2n) — isotropic, divide by 4π
            q2 = 2.0 * (sig2[mid].T @ phi_cell) / four_pi

            # P0 scattering — isotropic, divide by 4π
            qS = (sig_s0[mid].T @ phi_cell) / four_pi

            for n in range(quad.N):
                mu_z = getattr(quad, 'mu_z', np.zeros(quad.N))
                if mu_z[n] < -1e-15:
                    continue
                if ix == 0 and quad.mu_x[n] > 1e-15:
                    continue
                if ix == nx - 1 and quad.mu_x[n] < -1e-15:
                    continue
                if iy == 0 and quad.mu_y[n] > 1e-15:
                    continue
                if iy == ny - 1 and quad.mu_y[n] < -1e-15:
                    continue

                rhs[:, eq_idx] = qF + q2 + qS
                eq_idx += 1

    return rhs.ravel(order='F')
