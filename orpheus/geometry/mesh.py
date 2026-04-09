"""Mesh data structures for 1-D and 2-D geometries.

A mesh is an immutable description of a spatial domain: cell edges,
material assignments, and derived quantities (volumes, surfaces,
cell centres).  Solvers receive a mesh and build mutable,
solver-specific state on top of it.

Both :class:`Mesh1D` and :class:`Mesh2D` are frozen dataclasses --
once created, their fields cannot be reassigned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coord import (
    CoordSystem,
    compute_surfaces_1d,
    compute_volumes_1d,
    compute_volumes_2d,
)


# ═══════════════════════════════════════════════════════════════════════
# Mesh1D
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Mesh1D:
    """One-dimensional mesh in Cartesian, cylindrical, or spherical coordinates.

    Parameters
    ----------
    edges : ndarray, shape (N+1,)
        Monotonically increasing cell boundary positions.
        For cylindrical / spherical meshes these are radii.
    mat_ids : ndarray, shape (N,)
        Integer material ID for each cell.
    coord : CoordSystem
        Coordinate system (default: Cartesian).
    """

    edges: np.ndarray
    mat_ids: np.ndarray
    coord: CoordSystem = CoordSystem.CARTESIAN

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=float)
        mat_ids = np.asarray(self.mat_ids, dtype=int)

        if edges.ndim != 1 or mat_ids.ndim != 1:
            raise ValueError("edges and mat_ids must be 1-D arrays")
        if len(edges) < 2:
            raise ValueError("edges must have at least 2 elements (1 cell)")
        if len(mat_ids) != len(edges) - 1:
            raise ValueError(
                f"len(mat_ids)={len(mat_ids)} must equal "
                f"len(edges)-1={len(edges) - 1}"
            )
        if not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be strictly monotonically increasing")

        # Store validated arrays (frozen bypass via object.__setattr__)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "mat_ids", mat_ids)

    # ── Derived properties ────────────────────────────────────────────

    @property
    def N(self) -> int:
        """Number of cells."""
        return len(self.edges) - 1

    @property
    def widths(self) -> np.ndarray:
        """Cell widths (edge-to-edge distance), shape (N,)."""
        return np.diff(self.edges)

    @property
    def centers(self) -> np.ndarray:
        """Cell centre positions, shape (N,)."""
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def volumes(self) -> np.ndarray:
        """Cell volumes, shape (N,).  Formula depends on *coord*."""
        return compute_volumes_1d(self.coord, self.edges)

    @property
    def surfaces(self) -> np.ndarray:
        """Surface areas at each edge, shape (N+1,).  Formula depends on *coord*."""
        return compute_surfaces_1d(self.coord, self.edges)

    @property
    def total_width(self) -> float:
        """Total extent of the mesh (outer edge minus inner edge)."""
        return float(self.edges[-1] - self.edges[0])


# ═══════════════════════════════════════════════════════════════════════
# Mesh2D
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Mesh2D:
    """Two-dimensional mesh: Cartesian (x, y) or cylindrical (r, z).

    Parameters
    ----------
    edges_x : ndarray, shape (Nx+1,)
        Edge positions in the first direction (x or r).
    edges_y : ndarray, shape (Ny+1,)
        Edge positions in the second direction (y or z).
    mat_map : ndarray, shape (Nx, Ny)
        Integer material ID for each cell.
    coord : CoordSystem
        ``CARTESIAN`` for (x, y) or ``CYLINDRICAL`` for (r, z).
    """

    edges_x: np.ndarray
    edges_y: np.ndarray
    mat_map: np.ndarray
    coord: CoordSystem = CoordSystem.CARTESIAN

    def __post_init__(self) -> None:
        edges_x = np.asarray(self.edges_x, dtype=float)
        edges_y = np.asarray(self.edges_y, dtype=float)
        mat_map = np.asarray(self.mat_map, dtype=int)

        if edges_x.ndim != 1 or edges_y.ndim != 1:
            raise ValueError("edges_x and edges_y must be 1-D arrays")
        if len(edges_x) < 2 or len(edges_y) < 2:
            raise ValueError("edge arrays must have at least 2 elements")
        if not np.all(np.diff(edges_x) > 0):
            raise ValueError("edges_x must be strictly monotonically increasing")
        if not np.all(np.diff(edges_y) > 0):
            raise ValueError("edges_y must be strictly monotonically increasing")

        nx = len(edges_x) - 1
        ny = len(edges_y) - 1
        if mat_map.shape != (nx, ny):
            raise ValueError(
                f"mat_map shape {mat_map.shape} must be ({nx}, {ny})"
            )
        if self.coord not in (CoordSystem.CARTESIAN, CoordSystem.CYLINDRICAL):
            raise ValueError(
                f"Mesh2D supports CARTESIAN or CYLINDRICAL, got {self.coord}"
            )

        object.__setattr__(self, "edges_x", edges_x)
        object.__setattr__(self, "edges_y", edges_y)
        object.__setattr__(self, "mat_map", mat_map)

    # ── Derived properties ────────────────────────────────────────────

    @property
    def nx(self) -> int:
        """Number of cells in x (or r) direction."""
        return len(self.edges_x) - 1

    @property
    def ny(self) -> int:
        """Number of cells in y (or z) direction."""
        return len(self.edges_y) - 1

    @property
    def dx(self) -> np.ndarray:
        """Cell widths in x (or r) direction, shape (Nx,)."""
        return np.diff(self.edges_x)

    @property
    def dy(self) -> np.ndarray:
        """Cell widths in y (or z) direction, shape (Ny,)."""
        return np.diff(self.edges_y)

    @property
    def centers_x(self) -> np.ndarray:
        """Cell centres in x (or r) direction, shape (Nx,)."""
        return 0.5 * (self.edges_x[:-1] + self.edges_x[1:])

    @property
    def centers_y(self) -> np.ndarray:
        """Cell centres in y (or z) direction, shape (Ny,)."""
        return 0.5 * (self.edges_y[:-1] + self.edges_y[1:])

    @property
    def volumes(self) -> np.ndarray:
        """Cell volumes, shape (Nx, Ny).  Formula depends on *coord*."""
        return compute_volumes_2d(self.coord, self.edges_x, self.edges_y)

    @property
    def mat_ids(self) -> np.ndarray:
        """Flat material-ID array, shape (Nx*Ny,).

        Compatible with :func:`data.macro_xs.cell_xs.assemble_cell_xs`.
        """
        return self.mat_map.ravel()
