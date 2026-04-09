"""Mesh construction factories.

Zone-based construction
-----------------------
A *zone* is a material region defined by its outer boundary.  The
:func:`mesh1d_from_zones` function subdivides each zone into cells
with a coordinate-system-aware strategy:

* **Cartesian** -- equal-width cells.
* **Cylindrical** -- equal-volume annuli.
* **Spherical** -- equal-volume shells.

PWR convenience factories
-------------------------
:func:`pwr_slab_half_cell` and :func:`pwr_pin_equivalent` build
standard 3-zone (fuel | clad | coolant) meshes with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coord import CoordSystem
from .mesh import Mesh1D, Mesh2D


@dataclass
class Zone:
    """One material zone for mesh construction.

    Parameters
    ----------
    outer_edge : float
        Absolute position of the outer boundary of this zone.
    mat_id : int
        Material identifier for cells in this zone.
    n_cells : int
        Number of sub-cells to create within the zone.
    """

    outer_edge: float
    mat_id: int
    n_cells: int


# ── Zone-based 1-D construction ──────────────────────────────────────

def _subdivide_zone(
    inner: float,
    outer: float,
    n: int,
    coord: CoordSystem,
) -> np.ndarray:
    """Return *n + 1* edge positions within a zone (inner to outer).

    Subdivision guarantees equal-volume cells in each coordinate system:

    * Cartesian:   ``x_k = inner + k/n * (outer - inner)``
    * Cylindrical: ``r_k = sqrt(inner^2 + k/n * (outer^2 - inner^2))``
    * Spherical:   ``r_k = cbrt(inner^3 + k/n * (outer^3 - inner^3))``
    """
    fracs = np.linspace(0.0, 1.0, n + 1)
    match coord:
        case CoordSystem.CARTESIAN:
            return inner + fracs * (outer - inner)
        case CoordSystem.CYLINDRICAL:
            return np.sqrt(inner**2 + fracs * (outer**2 - inner**2))
        case CoordSystem.SPHERICAL:
            return np.cbrt(inner**3 + fracs * (outer**3 - inner**3))
        case _:
            raise ValueError(f"Unknown coordinate system: {coord}")


def mesh1d_from_zones(
    zones: list[Zone],
    coord: CoordSystem = CoordSystem.CARTESIAN,
    origin: float = 0.0,
) -> Mesh1D:
    """Build a :class:`~geometry.mesh.Mesh1D` from a list of zones.

    Parameters
    ----------
    zones : list[Zone]
        Zones ordered from inner to outer.  Each zone's
        :attr:`~Zone.outer_edge` is the absolute position of its
        outer boundary.
    coord : CoordSystem
        Coordinate system (determines subdivision strategy).
    origin : float
        Position of the inner-most edge (default 0).

    Returns
    -------
    Mesh1D
    """
    if not zones:
        raise ValueError("At least one zone is required")

    edges_list: list[np.ndarray] = []
    mat_ids_list: list[np.ndarray] = []
    inner = origin

    for zone in zones:
        sub_edges = _subdivide_zone(inner, zone.outer_edge, zone.n_cells, coord)
        # Append sub-edges, skipping the first (== previous outer)
        edges_list.append(sub_edges[1:])
        mat_ids_list.append(np.full(zone.n_cells, zone.mat_id, dtype=int))
        inner = zone.outer_edge

    edges = np.concatenate([[origin], *edges_list])
    mat_ids = np.concatenate(mat_ids_list)

    return Mesh1D(edges=edges, mat_ids=mat_ids, coord=coord)


# ── PWR convenience factories ────────────────────────────────────────

def pwr_slab_half_cell(
    n_fuel: int = 10,
    n_clad: int = 3,
    n_cool: int = 7,
    fuel_half: float = 0.9,
    clad_thick: float = 0.2,
    cool_thick: float = 0.7,
) -> Mesh1D:
    """Cartesian 1-D half-cell: fuel | clad | coolant.

    The mesh starts at x = 0 (reflective symmetry plane at the fuel
    centre) and extends to x = fuel_half + clad_thick + cool_thick.

    Material IDs: 2 = fuel, 1 = clad, 0 = coolant.
    """
    x1 = fuel_half
    x2 = x1 + clad_thick
    x3 = x2 + cool_thick

    zones = [
        Zone(outer_edge=x1, mat_id=2, n_cells=n_fuel),
        Zone(outer_edge=x2, mat_id=1, n_cells=n_clad),
        Zone(outer_edge=x3, mat_id=0, n_cells=n_cool),
    ]
    return mesh1d_from_zones(zones, coord=CoordSystem.CARTESIAN)


def pwr_pin_equivalent(
    n_fuel: int = 10,
    n_clad: int = 3,
    n_cool: int = 7,
    r_fuel: float = 0.9,
    r_clad: float = 1.1,
    pitch: float = 3.6,
) -> Mesh1D:
    """Cylindrical 1-D Wigner-Seitz equivalent pin cell.

    The square unit cell (side = *pitch*) is replaced by a cylinder of
    equal area: ``r_cell = pitch / sqrt(pi)``.

    Material IDs: 2 = fuel, 1 = clad, 0 = coolant.
    Sub-cells use equal-volume annuli.
    """
    r_cell = pitch / np.sqrt(np.pi)

    zones = [
        Zone(outer_edge=r_fuel, mat_id=2, n_cells=n_fuel),
        Zone(outer_edge=r_clad, mat_id=1, n_cells=n_clad),
        Zone(outer_edge=r_cell, mat_id=0, n_cells=n_cool),
    ]
    return mesh1d_from_zones(zones, coord=CoordSystem.CYLINDRICAL)


def homogeneous_1d(
    n_cells: int,
    total_width: float,
    mat_id: int = 0,
    coord: CoordSystem = CoordSystem.CARTESIAN,
) -> Mesh1D:
    """Uniform 1-D mesh with a single material.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    total_width : float
        Total extent (thickness for Cartesian, outer radius for
        cylindrical/spherical).
    mat_id : int
        Material identifier for all cells.
    coord : CoordSystem
        Coordinate system (determines subdivision strategy).
    """
    zones = [Zone(outer_edge=total_width, mat_id=mat_id, n_cells=n_cells)]
    return mesh1d_from_zones(zones, coord=coord)


def slab_fuel_moderator(
    n_fuel: int,
    n_mod: int,
    t_fuel: float,
    t_mod: float,
) -> Mesh1D:
    """1-D Cartesian slab benchmark: fuel + moderator.

    Material IDs: 2 = fuel (inner), 0 = moderator (outer).
    """
    zones = [
        Zone(outer_edge=t_fuel, mat_id=2, n_cells=n_fuel),
        Zone(outer_edge=t_fuel + t_mod, mat_id=0, n_cells=n_mod),
    ]
    return mesh1d_from_zones(zones, coord=CoordSystem.CARTESIAN)


def pwr_pin_2d(
    radii: list[float] | None = None,
    mat_ids: list[int] | None = None,
    pitch: float = 3.6,
    n_cells: int = 10,
) -> Mesh2D:
    """2-D Cartesian mesh from concentric annular regions.

    Each cell in the uniform (n_cells x n_cells) grid is assigned a
    material ID based on its distance from the pin centre (pitch / 2).

    Parameters
    ----------
    radii : list[float], optional
        Outer radii of each annular region.  Default: [0.9, 1.1]
        (fuel, clad; everything beyond is coolant).
    mat_ids : list[int], optional
        Material ID for each annulus, plus one for the region beyond
        the outermost radius.  Default: [2, 1, 0].
    pitch : float
        Unit cell side length (cm).
    n_cells : int
        Number of mesh cells per side.
    """
    if radii is None:
        radii = [0.9, 1.1]
    if mat_ids is None:
        mat_ids = [2, 1, 0]

    if len(mat_ids) != len(radii) + 1:
        raise ValueError(
            f"len(mat_ids)={len(mat_ids)} must equal len(radii)+1={len(radii) + 1}"
        )

    delta = pitch / n_cells
    edges = np.linspace(0.0, pitch, n_cells + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cx, cy = np.meshgrid(centres, centres, indexing="ij")
    r = np.sqrt((cx - pitch / 2) ** 2 + (cy - pitch / 2) ** 2)

    mat_map = np.full((n_cells, n_cells), mat_ids[-1], dtype=int)
    for k in range(len(radii) - 1, -1, -1):
        mat_map[r <= radii[k]] = mat_ids[k]

    return Mesh2D(edges_x=edges, edges_y=edges, mat_map=mat_map)
