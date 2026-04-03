"""2D SN transport solver — backward-compatible wrapper.

Delegates to the unified ``sn_solver.solve_sn()`` with 2D mesh and
Lebedev quadrature. All original names are preserved:
``PinCellGeometry``, ``Quadrature``, ``DOParams``, ``DOResult``,
``solve_discrete_ordinates``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.macro_xs.mixture import Mixture
from sn_geometry import CartesianMesh
from sn_quadrature import LebedevSphere
from sn_solver import solve_sn


# ── Backward-compatible types ─────────────────────────────────────────

@dataclass
class PinCellGeometry:
    """2D Cartesian mesh for a PWR pin cell (quarter-symmetry)."""

    nx: int
    ny: int
    delta: float
    mat_map: np.ndarray
    volume: np.ndarray

    @classmethod
    def default_pwr(cls) -> PinCellGeometry:
        n = 10
        delta = 0.2
        vol = np.full((n, n), delta**2)
        vol[0, :] /= 2
        vol[-1, :] /= 2
        vol[:, 0] /= 2
        vol[:, -1] /= 2
        row = np.array([2, 2, 2, 2, 2, 1, 0, 0, 0, 0], dtype=int)
        mat = np.tile(row, (n, 1)).T
        return cls(nx=n, ny=n, delta=delta, mat_map=mat, volume=vol)

    def _to_mesh(self) -> CartesianMesh:
        return CartesianMesh.uniform_2d(self.nx, self.ny, self.delta, self.mat_map)


@dataclass
class Quadrature:
    """Lebedev angular quadrature on the unit sphere (legacy type)."""

    mu_x: np.ndarray
    mu_y: np.ndarray
    mu_z: np.ndarray
    weights: np.ndarray
    ref_x: np.ndarray
    ref_y: np.ndarray
    ref_z: np.ndarray
    R: np.ndarray
    N: int = 0
    L: int = 0

    @classmethod
    def lebedev(cls, order: int = 17, L: int = 0) -> Quadrature:
        from scipy.integrate import lebedev_rule
        from sn_quadrature import _find_reflections
        pts, w = lebedev_rule(order)
        mu_x, mu_y, mu_z = pts[0], pts[1], pts[2]
        n_pts = len(w)
        ref_x = _find_reflections(-mu_x, mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_y = _find_reflections(mu_x, -mu_y, mu_z, mu_x, mu_y, mu_z)
        ref_z = _find_reflections(mu_x, mu_y, -mu_z, mu_x, mu_y, mu_z)
        R = np.zeros((n_pts, L + 1, 2 * L + 1))
        for n in range(n_pts):
            for j in range(L + 1):
                for m in range(-j, j + 1):
                    col = j + m
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


@dataclass
class DOParams:
    """Solver parameters for discrete ordinates."""

    max_outer: int = 200
    bicgstab_tol: float = 1e-4
    bicgstab_maxiter: int = 2000
    L: int = 0


@dataclass
class DOResult:
    """Results of a discrete ordinates calculation."""

    keff: float
    keff_history: list[float]
    residual_history: list[float]
    flux_fuel: np.ndarray
    flux_clad: np.ndarray
    flux_cool: np.ndarray
    scalar_flux: np.ndarray
    geometry: PinCellGeometry
    eg: np.ndarray
    elapsed_seconds: float


# ── Public API ────────────────────────────────────────────────────────

def solve_discrete_ordinates(
    materials: dict[int, Mixture],
    geom: PinCellGeometry | None = None,
    quad: Quadrature | None = None,
    params: DOParams | None = None,
) -> DOResult:
    """Run the 2D SN transport calculation.

    Delegates to the unified solver with Lebedev quadrature and
    source iteration inner solver.
    """
    if geom is None:
        geom = PinCellGeometry.default_pwr()
    if params is None:
        params = DOParams()

    mesh = geom._to_mesh()

    # Build Lebedev quadrature for unified solver
    leb = LebedevSphere.create(order=17)

    result = solve_sn(
        materials, mesh, leb,
        inner_solver="source_iteration",
        scattering_order=params.L,
        max_outer=params.max_outer,
        keff_tol=1e-7,
        flux_tol=1e-5,
        max_inner=params.bicgstab_maxiter,
        inner_tol=params.bicgstab_tol,
    )

    # Post-processing: per-material volume-averaged fluxes
    sf = result.scalar_flux  # (nx, ny, ng)
    vol = geom.volume
    ng = sf.shape[2]

    flux_per_mat = {}
    for mat_id in [0, 1, 2]:
        mask = geom.mat_map == mat_id
        if mask.any():
            vol_mat = vol[mask].sum()
            flux_per_mat[mat_id] = np.zeros(ng)
            for g in range(ng):
                flux_per_mat[mat_id][g] = np.sum(sf[:, :, g][mask] * vol[mask]) / vol_mat
        else:
            flux_per_mat[mat_id] = np.zeros(ng)

    return DOResult(
        keff=result.keff,
        keff_history=result.keff_history,
        residual_history=[],
        flux_fuel=flux_per_mat.get(2, np.zeros(ng)),
        flux_clad=flux_per_mat.get(1, np.zeros(ng)),
        flux_cool=flux_per_mat.get(0, np.zeros(ng)),
        scalar_flux=sf,
        geometry=geom,
        eg=result.eg,
        elapsed_seconds=result.elapsed_seconds,
    )
