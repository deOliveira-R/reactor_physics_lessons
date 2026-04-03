"""1D SN transport solver — backward-compatible wrapper.

Delegates to the unified ``sn_solver.solve_sn()`` with 1D mesh (ny=1)
and Gauss-Legendre quadrature.

All original names are preserved for backward compatibility:
``GaussLegendreQuadrature``, ``Slab1DGeometry``, ``SN1DResult``, ``solve_sn_1d``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from data.macro_xs.mixture import Mixture
from sn_geometry import CartesianMesh
from sn_quadrature import GaussLegendre1D
from sn_solver import solve_sn


# ── Backward-compatible types ─────────────────────────────────────────

@dataclass
class GaussLegendreQuadrature:
    """Gauss-Legendre angular quadrature on [-1, 1].

    Weights sum to 2.0 (the measure of [-1, 1]).
    """

    mu: np.ndarray
    weights: np.ndarray
    N: int

    @classmethod
    def gauss_legendre(cls, n_ordinates: int) -> GaussLegendreQuadrature:
        mu, w = np.polynomial.legendre.leggauss(n_ordinates)
        return cls(mu=mu, weights=w, N=n_ordinates)


@dataclass
class Slab1DGeometry:
    """1D slab geometry for SN transport."""

    cell_widths: np.ndarray
    mat_ids: np.ndarray
    N: int

    @classmethod
    def from_benchmark(
        cls, n_fuel: int, n_mod: int, t_fuel: float, t_mod: float,
    ) -> Slab1DGeometry:
        dx_fuel = t_fuel / n_fuel
        dx_mod = t_mod / n_mod
        widths = np.concatenate([np.full(n_fuel, dx_fuel), np.full(n_mod, dx_mod)])
        mids = np.concatenate([np.full(n_fuel, 2, dtype=int), np.full(n_mod, 0, dtype=int)])
        return cls(cell_widths=widths, mat_ids=mids, N=n_fuel + n_mod)

    @classmethod
    def from_regions(
        cls,
        thicknesses: list[float],
        mat_ids_per_region: list[int],
        n_cells_per_region: int = 10,
    ) -> Slab1DGeometry:
        widths_list = []
        mids_list = []
        for t, mid in zip(thicknesses, mat_ids_per_region):
            dx = t / n_cells_per_region
            widths_list.append(np.full(n_cells_per_region, dx))
            mids_list.append(np.full(n_cells_per_region, mid, dtype=int))
        return cls(
            cell_widths=np.concatenate(widths_list),
            mat_ids=np.concatenate(mids_list),
            N=sum(n_cells_per_region for _ in thicknesses),
        )

    @classmethod
    def homogeneous(
        cls, n_cells: int, total_width: float, mat_id: int = 2,
    ) -> Slab1DGeometry:
        dx = total_width / n_cells
        return cls(
            cell_widths=np.full(n_cells, dx),
            mat_ids=np.full(n_cells, mat_id, dtype=int),
            N=n_cells,
        )

    def _to_mesh(self) -> CartesianMesh:
        return CartesianMesh.from_slab_1d(self.cell_widths, self.mat_ids)


@dataclass
class SN1DResult:
    """Results of a 1D SN calculation."""

    keff: float
    keff_history: list[float]
    flux: np.ndarray          # (N_cells, ng)
    geometry: Slab1DGeometry
    eg: np.ndarray
    elapsed_seconds: float


# ── Public API ────────────────────────────────────────────────────────

def solve_sn_1d(
    materials: dict[int, Mixture],
    geom: Slab1DGeometry,
    quad: GaussLegendreQuadrature | None = None,
    max_outer: int = 500,
    keff_tol: float = 1e-7,
    flux_tol: float = 1e-6,
    max_inner: int = 200,
    inner_tol: float = 1e-8,
) -> SN1DResult:
    """Solve the 1D multi-group SN eigenvalue problem.

    Delegates to the unified SN solver with 1D mesh and GL quadrature.
    """
    if quad is None:
        quad = GaussLegendreQuadrature.gauss_legendre(16)

    mesh = geom._to_mesh()
    gl = GaussLegendre1D(mu_x=quad.mu, mu_y=np.zeros(quad.N), weights=quad.weights, N=quad.N)

    result = solve_sn(
        materials, mesh, gl,
        inner_solver="source_iteration",
        max_outer=max_outer,
        keff_tol=keff_tol,
        flux_tol=flux_tol,
        max_inner=max_inner,
        inner_tol=inner_tol,
    )

    return SN1DResult(
        keff=result.keff,
        keff_history=result.keff_history,
        flux=result.scalar_flux[:, 0, :],  # squeeze ny=1
        geometry=geom,
        eg=result.eg,
        elapsed_seconds=result.elapsed_seconds,
    )
