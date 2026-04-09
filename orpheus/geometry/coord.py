"""Coordinate systems and their volume / surface formulas.

This module is the **single point** where coordinate-system dependence
lives.  All mesh classes delegate to these functions.

Supported coordinate systems
-----------------------------
* **Cartesian** -- flat geometry (slab, plate, box)
* **Cylindrical** -- annular geometry (pin cell, tube)
* **Spherical** -- shell geometry (pebble, sphere)

Volume formulas (1-D)
~~~~~~~~~~~~~~~~~~~~~
=========== ==========================================
Cartesian   :math:`V_i = x_{i+1} - x_i`
Cylindrical :math:`V_i = \\pi (r_{i+1}^2 - r_i^2)`
Spherical   :math:`V_i = \\tfrac{4}{3}\\pi (r_{i+1}^3 - r_i^3)`
=========== ==========================================

Surface formulas (1-D)
~~~~~~~~~~~~~~~~~~~~~~
=========== ==========================================
Cartesian   :math:`S = 1` (per unit transverse area)
Cylindrical :math:`S = 2\\pi r` (per unit height)
Spherical   :math:`S = 4\\pi r^2`
=========== ==========================================
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class CoordSystem(Enum):
    """Coordinate system identifier."""

    CARTESIAN = "cartesian"
    CYLINDRICAL = "cylindrical"
    SPHERICAL = "spherical"


# ── 1-D formulas ─────────────────────────────────────────────────────

def compute_volumes_1d(coord: CoordSystem, edges: np.ndarray) -> np.ndarray:
    """Cell volumes from 1-D edge positions.

    Parameters
    ----------
    coord : CoordSystem
        Coordinate system.
    edges : ndarray, shape (N+1,)
        Monotonically increasing edge positions.

    Returns
    -------
    ndarray, shape (N,)
        Volume of each cell.
    """
    match coord:
        case CoordSystem.CARTESIAN:
            return np.diff(edges)
        case CoordSystem.CYLINDRICAL:
            return np.pi * np.diff(edges**2)
        case CoordSystem.SPHERICAL:
            return (4.0 / 3.0) * np.pi * np.diff(edges**3)
        case _:
            raise ValueError(f"Unknown coordinate system: {coord}")


def compute_surfaces_1d(coord: CoordSystem, edges: np.ndarray) -> np.ndarray:
    """Surface areas at each 1-D edge position.

    Parameters
    ----------
    coord : CoordSystem
        Coordinate system.
    edges : ndarray, shape (N+1,)
        Edge positions.

    Returns
    -------
    ndarray, shape (N+1,)
        Surface area at every edge.
    """
    match coord:
        case CoordSystem.CARTESIAN:
            return np.ones_like(edges)
        case CoordSystem.CYLINDRICAL:
            return 2.0 * np.pi * edges
        case CoordSystem.SPHERICAL:
            return 4.0 * np.pi * edges**2
        case _:
            raise ValueError(f"Unknown coordinate system: {coord}")


# ── 2-D formulas ─────────────────────────────────────────────────────

def compute_volumes_2d(
    coord: CoordSystem,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
) -> np.ndarray:
    """Cell volumes from 2-D edge positions.

    Parameters
    ----------
    coord : CoordSystem
        ``CARTESIAN`` for (x, y) or ``CYLINDRICAL`` for (r, z).
    edges_x : ndarray, shape (Nx+1,)
        Edge positions in x (or radial) direction.
    edges_y : ndarray, shape (Ny+1,)
        Edge positions in y (or axial) direction.

    Returns
    -------
    ndarray, shape (Nx, Ny)
        Volume of each cell.
    """
    match coord:
        case CoordSystem.CARTESIAN:
            dx = np.diff(edges_x)
            dy = np.diff(edges_y)
            return dx[:, np.newaxis] * dy[np.newaxis, :]
        case CoordSystem.CYLINDRICAL:
            # r-z geometry: V = pi * (r_out^2 - r_in^2) * dz
            dr2 = np.diff(edges_x**2)  # (Nr,)
            dz = np.diff(edges_y)       # (Nz,)
            return np.pi * dr2[:, np.newaxis] * dz[np.newaxis, :]
        case _:
            raise ValueError(
                f"2-D volumes not defined for {coord}; "
                f"use CARTESIAN or CYLINDRICAL"
            )
