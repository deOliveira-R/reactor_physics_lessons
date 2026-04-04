"""Cartesian mesh geometry for SN transport.

A 2D Cartesian mesh where 1D is the degenerate case ny=1.
Boundary conditions are reflective on all sides (infinite lattice).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CartesianMesh:
    """1D or 2D Cartesian mesh with per-cell material assignment.

    For 1D slab problems, set ny=1 and dy=[1.0] (arbitrary — the
    y-streaming term vanishes when mu_y=0 for GL quadrature).

    Attributes
    ----------
    nx, ny : mesh dimensions
    dx : (nx,) cell widths in x (cm)
    dy : (ny,) cell widths in y (cm)
    mat_map : (nx, ny) integer material IDs
    """

    nx: int
    ny: int
    dx: np.ndarray
    dy: np.ndarray
    mat_map: np.ndarray

    @property
    def is_1d(self) -> bool:
        """True if this is a 1D mesh (ny=1)."""
        return self.ny == 1

    @property
    def volume(self) -> np.ndarray:
        """(nx, ny) cell volumes (full — no boundary halving).

        With reflective BCs (infinite lattice), boundary cells are
        full-size cells at the symmetry plane, not half-cells.
        """
        return self.dx[:, None] * self.dy[None, :]

    # ── 1D factories ──────────────────────────────────────────────────

    @classmethod
    def from_slab_1d(
        cls,
        cell_widths: np.ndarray,
        mat_ids: np.ndarray,
    ) -> CartesianMesh:
        """Build a 1D mesh (ny=1) from cell widths and material IDs."""
        nx = len(cell_widths)
        return cls(
            nx=nx, ny=1,
            dx=np.asarray(cell_widths, dtype=float),
            dy=np.array([1.0]),
            mat_map=np.asarray(mat_ids, dtype=int).reshape(nx, 1),
        )

    @classmethod
    def from_regions(
        cls,
        thicknesses: list[float],
        mat_ids: list[int],
        n_cells_per_region: int = 10,
    ) -> CartesianMesh:
        """Build a 1D multi-region mesh."""
        dx_list = []
        mid_list = []
        for t, mid in zip(thicknesses, mat_ids):
            d = t / n_cells_per_region
            dx_list.append(np.full(n_cells_per_region, d))
            mid_list.append(np.full(n_cells_per_region, mid, dtype=int))
        dx = np.concatenate(dx_list)
        mids = np.concatenate(mid_list)
        return cls.from_slab_1d(dx, mids)

    @classmethod
    def homogeneous_1d(
        cls,
        n_cells: int,
        total_width: float,
        mat_id: int = 0,
    ) -> CartesianMesh:
        """Build a homogeneous 1D mesh."""
        dx = np.full(n_cells, total_width / n_cells)
        mids = np.full(n_cells, mat_id, dtype=int)
        return cls.from_slab_1d(dx, mids)

    @classmethod
    def from_benchmark(
        cls,
        n_fuel: int,
        n_mod: int,
        t_fuel: float,
        t_mod: float,
    ) -> CartesianMesh:
        """Build a 1D fuel + moderator benchmark slab.

        Material IDs: 2 = fuel, 0 = moderator.
        """
        dx = np.concatenate([
            np.full(n_fuel, t_fuel / n_fuel),
            np.full(n_mod, t_mod / n_mod),
        ])
        mids = np.concatenate([
            np.full(n_fuel, 2, dtype=int),
            np.full(n_mod, 0, dtype=int),
        ])
        return cls.from_slab_1d(dx, mids)

    # ── 2D factories ──────────────────────────────────────────────────

    @classmethod
    def uniform_2d(
        cls,
        nx: int,
        ny: int,
        delta: float,
        mat_map: np.ndarray,
    ) -> CartesianMesh:
        """Build a 2D mesh with uniform cell size."""
        return cls(
            nx=nx, ny=ny,
            dx=np.full(nx, delta),
            dy=np.full(ny, delta),
            mat_map=np.asarray(mat_map, dtype=int),
        )

    @classmethod
    def default_pwr_2d(cls, nx: int = 10, ny: int = 10, delta: float = 0.2) -> CartesianMesh:
        """Standard PWR pin cell: fuel + clad + coolant columns."""
        n_fuel = nx // 2
        n_clad = 1
        n_cool = nx - n_fuel - n_clad
        row = np.array(
            [2] * n_fuel + [1] * n_clad + [0] * n_cool, dtype=int,
        )
        mat = np.tile(row, (ny, 1)).T  # (nx, ny)
        return cls.uniform_2d(nx, ny, delta, mat)
