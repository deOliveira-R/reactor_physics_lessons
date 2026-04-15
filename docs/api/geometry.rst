Geometry Infrastructure
========================

The :mod:`orpheus.geometry` package provides the spatial data
structures that every deterministic solver (SN, CP, MOC, diffusion)
consumes. A *mesh* is an immutable description of a domain — cell
edges, material assignments, and derived quantities such as
volumes and surfaces. Solvers receive a mesh and build mutable,
solver-specific state on top of it.

.. contents::
   :local:
   :depth: 2


Design Principles
-----------------

**Frozen dataclasses.**
Both :class:`~orpheus.geometry.mesh.Mesh1D` and
:class:`~orpheus.geometry.mesh.Mesh2D` are
``@dataclass(frozen=True)``. Once constructed, their fields cannot
be reassigned. This turns every solver entry point into a pure
function of its inputs and prevents whole classes of bugs where a
downstream routine accidentally mutates mesh state shared across
iterations.

**Coordinate-aware volumes and surfaces.**
All geometric quantities route through
:mod:`orpheus.geometry.coord`, which dispatches on a
:class:`~orpheus.geometry.coord.CoordSystem` enum
(``CARTESIAN``, ``CYLINDRICAL``, ``SPHERICAL``). This keeps the
physics solvers coordinate-agnostic — the same
:func:`~orpheus.sn.solver.solve_sn` entry point handles slab,
cylinder, and sphere without branching on geometry.

**Equal-volume subdivision.**
Curvilinear zones (cylindrical, spherical) are subdivided into
**equal-volume** annuli / shells rather than equal-width cells.
This gives uniform statistical weighting across the zone and
avoids skinny inner cells that would dominate the CFL-like step
limits of explicit sweeps.

**Precomputed volumes — the ULP escape hatch.**
:class:`~orpheus.geometry.mesh.Mesh1D` accepts an optional
``precomputed_volumes`` override. The
:func:`~orpheus.geometry.factories.mesh1d_from_zones` factory sets
it by computing the *algebraic* cell volume (e.g.
:math:`V_{\rm cell} = \pi(r_{\rm out}^2 - r_{\rm in}^2)/n` in the
cylindrical case) and broadcasting that scalar to every cell in
the zone. Deriving volumes from the *edges* after the fact via
:func:`~orpheus.geometry.coord.compute_volumes_1d` would pass
through a ``sqrt → **2`` or ``cbrt → **3`` round trip that loses
roughly one ULP per cell and breaks the invariant "every cell in
an equal-volume zone is bit-identical" at ``rtol=1e-14``. Manually
constructed meshes with arbitrary edges still derive volumes from
edges as before — the override only kicks in on the factory path.


Mesh
----

.. automodule:: orpheus.geometry.mesh
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:


Coordinate Systems
------------------

:mod:`orpheus.geometry.coord` defines the
:class:`~orpheus.geometry.coord.CoordSystem` enum and the
coordinate-aware volume / surface primitives:

* ``compute_volumes_1d(coord, edges)``
* ``compute_surfaces_1d(coord, edges)``
* ``compute_volumes_2d(coord, edges_x, edges_y)``

All three dispatch on ``coord`` and return NumPy arrays sized to
match the mesh. The 1-D spherical volume formula,

.. math::

   V_i = \frac{4\pi}{3}\bigl(r_{i+1}^3 - r_i^3\bigr),

and the cylindrical formula,

.. math::

   V_i = \pi\bigl(r_{i+1}^2 - r_i^2\bigr),

are the standard shell / annulus expressions. The surface arrays
return :math:`4\pi r^2` (spherical) or :math:`2\pi r` (cylindrical,
per unit height) at each edge — these drive the :math:`\Delta A /
w_m` redistribution factor in the curvilinear SN sweeps (see
:ref:`theory-discrete-ordinates`).

.. automodule:: orpheus.geometry.coord
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:


Factories
---------

The factory layer is the recommended construction path.
:class:`~orpheus.geometry.factories.Zone` describes one material
region by its outer boundary and cell count;
:func:`~orpheus.geometry.factories.mesh1d_from_zones` builds a
coordinate-aware :class:`~orpheus.geometry.mesh.Mesh1D` from a
list of zones. Three subdivision strategies are baked in:

* **Cartesian** — equal-width cells
  :math:`x_k = x_0 + (k/n)\,(x_n - x_0)`.
* **Cylindrical** — equal-volume annuli
  :math:`r_k = \sqrt{r_0^2 + (k/n)\,(r_n^2 - r_0^2)}`.
* **Spherical** — equal-volume shells
  :math:`r_k = \sqrt[3]{r_0^3 + (k/n)\,(r_n^3 - r_0^3)}`.

Each returns both the edges and the exact per-cell volume (a
broadcast scalar) so the frozen :class:`Mesh1D` can be built with
``precomputed_volumes`` set — see the design principle above.

**Convenience constructors:**

* :func:`~orpheus.geometry.factories.pwr_slab_half_cell` — Cartesian
  3-zone (fuel / clad / coolant) half-cell with a reflective
  symmetry plane at :math:`x = 0`.
* :func:`~orpheus.geometry.factories.pwr_pin_equivalent` — cylindrical
  Wigner--Seitz equivalent pin cell. The square unit cell of side
  *pitch* is replaced by a cylinder of equal cross-sectional area,
  :math:`r_{\rm cell} = {\rm pitch} / \sqrt{\pi}`.
* :func:`~orpheus.geometry.factories.homogeneous_1d` — single-material
  uniform mesh for homogeneous-medium tests and analytical
  benchmarks.
* :func:`~orpheus.geometry.factories.slab_fuel_moderator` — 2-zone
  Cartesian slab (fuel / moderator) for classic L1 verification
  problems.
* :func:`~orpheus.geometry.factories.pwr_pin_2d` — 2-D Cartesian mesh
  with material IDs assigned by radial distance from the pin centre.

**Material ID convention:**
``2 = fuel``, ``1 = clad``, ``0 = coolant / moderator``. This
ordering matches the synthetic cross-section library used by the
L0 / L1 verification suites.

.. automodule:: orpheus.geometry.factories
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
