"""2D Method of Characteristics (MoC) neutron transport solver for a pin cell.

Solves the multi-group transport equation on a 2-D pin-cell geometry
(concentric annuli inside a square lattice cell) with reflective boundary
conditions.  Uses the flat-source MOC with a product angular quadrature
(azimuthal x Tabuchi-Yamamoto polar) and exact ray tracing through
circular boundaries.

Reference: Boyd et al. (2014) "The OpenMOC Method of Characteristics",
Ann. Nucl. Energy 68, 43-52.

.. seealso:: :ref:`theory-method-of-characteristics` — Key Facts, equations, gotchas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from orpheus.data.macro_xs.mixture import Mixture
from orpheus.geometry import CoordSystem, Mesh1D
from orpheus.numerics.eigenvalue import power_iteration

from .geometry import MOCMesh
from .quadrature import MOCQuadrature
from .core import MOCSolver


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MoCResult:
    """Results of a Method of Characteristics calculation."""

    keff: float
    keff_history: list[float]
    flux_per_material: dict[int, np.ndarray]  # mat_id -> (ng,) volume-averaged flux
    scalar_flux: np.ndarray   # (n_regions, ng) scalar flux per FSR
    moc_mesh: MOCMesh
    eg: np.ndarray            # (ng+1,) energy group boundaries
    elapsed_seconds: float

    @property
    def flux_fuel(self) -> np.ndarray:
        return self.flux_per_material.get(2, np.zeros(len(self.eg) - 1))

    @property
    def flux_clad(self) -> np.ndarray:
        return self.flux_per_material.get(1, np.zeros(len(self.eg) - 1))

    @property
    def flux_cool(self) -> np.ndarray:
        return self.flux_per_material.get(0, np.zeros(len(self.eg) - 1))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_moc(
    materials: dict[int, Mixture],
    mesh: Mesh1D | None = None,
    n_azi: int = 16,
    n_polar: int = 3,
    ray_spacing: float = 0.05,
    max_outer: int = 500,
    keff_tol: float = 1e-6,
    flux_tol: float = 1e-5,
    n_inner_sweeps: int = 15,
) -> MoCResult:
    """Run the 2D Method of Characteristics transport calculation.

    Parameters
    ----------
    materials : dict[int, Mixture]
        Macroscopic cross sections keyed by material ID.
    mesh : Mesh1D, optional
        Cylindrical 1-D Wigner-Seitz mesh.  Defaults to the standard
        PWR pin cell via ``pwr_pin_equivalent()``.
    n_azi : int
        Number of azimuthal angles in [0, pi).
    n_polar : int
        Number of TY polar angles per half-space (1, 2, or 3).
    ray_spacing : float
        Perpendicular spacing between parallel rays (cm).
    max_outer : int
        Maximum number of outer (power) iterations.
    keff_tol, flux_tol : float
        Convergence tolerances.
    n_inner_sweeps : int
        Number of transport sweeps per outer iteration.
    """
    t_start = time.perf_counter()

    if mesh is None:
        from orpheus.geometry import pwr_pin_equivalent
        mesh = pwr_pin_equivalent()

    _any_mat = next(iter(materials.values()))
    eg = _any_mat.eg
    ng = _any_mat.ng

    # Build augmented geometry
    quad = MOCQuadrature.create(n_azi=n_azi, n_polar=n_polar)
    moc_mesh = MOCMesh(mesh, quad, ray_spacing=ray_spacing)

    print(f"  MOC: {moc_mesh.n_regions} regions, {len(moc_mesh.tracks)} tracks, "
          f"{n_azi} azi x {n_polar} polar, spacing={ray_spacing}")

    # Eigenvalue solve
    solver = MOCSolver(
        moc_mesh, materials,
        keff_tol=keff_tol, flux_tol=flux_tol,
        n_inner_sweeps=n_inner_sweeps,
    )
    keff, keff_history, phi = power_iteration(solver, max_iter=max_outer)

    # Post-processing: volume-averaged spectra per material
    nr = moc_mesh.n_regions
    unique_mats = set(int(m) for m in moc_mesh.region_mat_ids)
    vol_per_mat = {m: 0.0 for m in unique_mats}
    flux_per_mat = {m: np.zeros(ng) for m in unique_mats}

    for i in range(nr):
        mat_id = int(moc_mesh.region_mat_ids[i])
        A_i = moc_mesh.region_areas[i]
        vol_per_mat[mat_id] += A_i
        flux_per_mat[mat_id] += phi[i, :] * A_i

    for m in unique_mats:
        if vol_per_mat[m] > 0:
            flux_per_mat[m] /= vol_per_mat[m]

    elapsed = time.perf_counter() - t_start
    print(f"  Elapsed: {elapsed:.1f}s")

    return MoCResult(
        keff=keff_history[-1],
        keff_history=keff_history,
        flux_per_material=flux_per_mat,
        scalar_flux=phi,
        moc_mesh=moc_mesh,
        eg=eg,
        elapsed_seconds=elapsed,
    )
