.. _theory-discrete-ordinates:

==========================================
Discrete Ordinates Method (S\ :sub:`N`)
==========================================

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before modifying the SN solver.**

- Transport equation: :math:`\mu_m \frac{\partial\psi_m}{\partial x} + \Sigma_t \psi_m = Q/2` (1D slab)
- Curvilinear adds angular redistribution: :math:`\alpha` coefficients couple ordinates
- Diamond difference: :math:`\psi^a = (1+\beta)\psi_{\text{out}} - \beta\psi_{\text{in}}`, Morel-Montry sets :math:`\beta = 0`
- Scattering convention: ``SigS[l][g_from, g_to]`` — source uses **transpose**: ``Q = SigS^T @ phi``
- GL weights sum to 2; Lebedev/LS/Product sum to :math:`4\pi`
- **Gotcha**: 1-group tests are degenerate (k = νΣ_f/Σ_a regardless of flux shape)
- **Gotcha**: homogeneous tests are blind to curvilinear redistribution bugs (flat flux → α terms vanish)
- **Gotcha**: conservation holds even with wrong per-ordinate balance (telescoping sum identity)
- The :math:`\alpha` dome must be non-negative; negative → NaN/overflow
- Fixed-source flat-flux diagnostic (Q/Σ_t) is the most powerful curvilinear bug detector
- Key reference: Bailey, Morel & Chang (2009) — Eq. 50 (α recursion), Eq. 74 (M-M weights)
- Verification uses :ref:`synthetic cross sections <synthetic-xs-library>`, not real nuclear data

.. admonition:: Conventions

   - Scattering matrix: :ref:`scattering-matrix-convention` — ``SigS[g_from, g_to]``, source uses transpose
   - Multi-group balance: :eq:`mg-balance` in :ref:`theory-homogeneous`
   - Cross sections: :ref:`theory-cross-section-data`
   - Verification: :ref:`synthetic-xs-library` — regions A/B/C/D
   - Eigenvalue: :ref:`power-iteration-algorithm` shared with all deterministic solvers


Overview
========

The discrete ordinates (S\ :sub:`N`) method solves the
:ref:`multi-group eigenvalue problem <mg-eigenvalue-problem>` in
integro-differential form by discretising the angular variable
:math:`\hat{\Omega}` into a finite set of directions (ordinates).  Unlike the
collision probability method (which works with the integral form and the scalar
flux), S\ :sub:`N` retains the **angular flux**
:math:`\psi(\mathbf{r}, \hat{\Omega}, E)` and resolves directional effects
such as streaming, anisotropic scattering, and angular current at interfaces.

Three coordinate systems are supported:

- **Cartesian** (slab / 2D) --- the simplest case; no angular coupling
  between ordinates.
- **Spherical** (1D radial) --- angular redistribution in :math:`\mu`;
  a single dome of :math:`\alpha` coefficients couples all ordinates.
- **Cylindrical** (1D radial) --- azimuthal redistribution per
  :math:`\mu`-level; independent :math:`\alpha` domes on each level.

All three share a single balance-equation framework with a geometry
factor :math:`\Delta A / w` that guarantees per-ordinate flat-flux
consistency.  The treatment follows [Bailey2009]_ for the curvilinear
formulation, [LewisMiller1984]_ for the general framework, and
[CaseZweifel1967]_ for the angular discretisation.

The solver is implemented in :class:`SNSolver`, which satisfies the
:class:`~numerics.eigenvalue.EigenvalueSolver` protocol.  The convenience
function :func:`solve_sn` runs the full calculation and returns an
:class:`SNResult`.

Because the protocol puts the scattering source *inside*
``solve_fixed_source``, the inner source iteration (which converges the
in-scatter and anisotropic source) stays encapsulated in the SN-specific
sweep — the outer :func:`~numerics.eigenvalue.power_iteration` loop is
identical to the one used by CP, MOC, diffusion, and the homogeneous
solver.  See :doc:`../api/numerics` for the protocol contract.


Architecture
============

Two-Layer Mesh Pattern
----------------------

The S\ :sub:`N` solver follows the same two-layer pattern as the CP
solver.  This pattern (base :class:`~geometry.mesh.Mesh1D` + augmented
mesh) is shared with :ref:`theory-collision-probability` and
:ref:`theory-method-of-characteristics`.

1. **Base geometry** --- :class:`~geometry.mesh.Mesh1D` or
   :class:`~geometry.mesh.Mesh2D` stores cell edges, material IDs, and
   coordinate system.

2. **Augmented geometry** --- :class:`SNMesh` wraps the base mesh and
   an angular quadrature, precomputing the coordinate-specific streaming
   stencil:

   - **Cartesian**: ``streaming_x[n,i] = 2|mu_x|/dx[i]`` and
     ``streaming_y[n,j] = 2|mu_y|/dy[j]`` --- the diamond-difference
     denominator terms, precomputed to avoid per-cell division in the
     sweep hot loop.
   - **Spherical**: ``face_areas`` (:math:`4\pi r^2`), ``delta_A``,
     ``alpha_half`` (angular redistribution dome),
     ``redist_dAw`` (:math:`\Delta A_i / w_n`, shape ``(nx, N)``),
     and ``tau_mm`` (Morel--Montry closure weights).
   - **Cylindrical**: ``face_areas`` (:math:`2\pi r`), ``delta_A``,
     ``alpha_per_level`` (per-level redistribution domes),
     ``redist_dAw_per_level`` (list of ``(nx, M)`` arrays), and
     ``tau_mm_per_level`` (per-level Morel--Montry weights).

3. **Solver** --- :func:`solve_sn` creates an ``SNMesh``, builds the
   ``SNSolver``, and runs power iteration.

.. code-block:: text

   Mesh1D / Mesh2D (base geometry)
       |
       v
   SNMesh (stencil + quadrature + alpha coefficients)
       |
       v
   solve_sn() --> SNResult

Quadrature Dispatch
-------------------

The sweep dispatcher in :func:`transport_sweep` routes based on the
``SNMesh.curvature`` attribute and the quadrature type:

.. code-block:: python

   if sn_mesh.curvature == "spherical":
       return _sweep_1d_spherical(...)
   elif sn_mesh.curvature == "cylindrical":
       return _sweep_1d_cylindrical(...)
   elif is_gl_1d:            # ny=1, mu_y=0, no aniso source
       return _sweep_1d_cumprod(...)
   else:
       return _sweep_2d_wavefront(...)

For 1D meshes (``ny=1``):

- **Gauss--Legendre** quadrature takes the fast cumprod path (all
  :math:`\mu_y = 0`, so no y-streaming).
- **Lebedev** quadrature falls through to the 2D wavefront sweep.
  Ordinates with :math:`\mu_x \neq 0` stream along *x*; the
  *y*-streaming terms cancel via reflective BCs on the single-cell
  *y*-dimension.  Ordinates with :math:`\mu_x = \mu_y = 0`
  (z-directed) reduce to pure collision:
  :math:`\psi = Q \cdot w_{\text{norm}} / \Sigt{}`.

Both quadratures recover the analytical eigenvalue exactly on
homogeneous problems (verified to machine precision for 1G/2G/4G).


The Transport Equation
======================

Cartesian 1D (Slab)
--------------------

The steady-state transport equation in a 1D slab:

.. math::
   :label: transport-cartesian

   \mu \frac{\partial \psi(x, \mu)}{\partial x}
   + \Sigt{} \, \psi(x, \mu)
   = \frac{Q}{W}

where :math:`\mu = \cos\theta` is the direction cosine, :math:`Q` is the
total isotropic source (fission + scattering), and :math:`W = \sum_n w_n`
is the quadrature weight sum.

Cartesian 2D
--------------

In two Cartesian dimensions the angular flux depends on two direction
cosines :math:`\mu_x` and :math:`\mu_y`:

.. math::
   :label: transport-cartesian-2d

   \mu_x \frac{\partial \psi}{\partial x}
   + \mu_y \frac{\partial \psi}{\partial y}
   + \Sigt{} \, \psi
   = \frac{Q}{W}

There is no angular coupling between ordinates --- each direction is
solved independently.  The two streaming terms are the only difference
from the 1D case.

Spherical 1D
-------------

In spherical coordinates the transport equation acquires an **angular
redistribution term** that couples ordinates:

.. math::
   :label: transport-spherical

   \mu \frac{\partial \psi}{\partial r}
   + \frac{1 - \mu^2}{r} \frac{\partial \psi}{\partial \mu}
   + \Sigt{} \psi = \frac{Q}{W}

The curvature term :math:`(1 - \mu^2)/r \cdot \partial\psi/\partial\mu`
arises because a neutron streaming radially at angle :math:`\mu` *rotates*
its direction cosine as it moves to a different radius.  Discretising this
term requires diamond difference in **both space and angle**.

Cylindrical 1D
---------------

For an infinitely long cylinder with azimuthal symmetry, the transport
equation in the radial variable :math:`r` is:

.. math::
   :label: transport-cylindrical

   \frac{\eta}{r} \frac{\partial(r\psi)}{\partial r}
   - \frac{1}{r} \frac{\partial(\xi\psi)}{\partial\varphi}
   + \Sigt{} \psi = \frac{Q}{W}

where the direction cosines are:

- :math:`\eta = \sin\theta\cos\varphi` --- radial projection (streaming)
- :math:`\xi = \sin\theta\sin\varphi` --- azimuthal component
- :math:`\mu = \cos\theta` --- axial component

The constraint :math:`\eta^2 + \xi^2 + \mu^2 = 1` holds.  The azimuthal
redistribution :math:`-\partial(\xi\psi)/\partial\varphi` couples ordinates
on each :math:`\mu`-level.

Multi-Group Extension
---------------------

For :math:`G` energy groups, each transport equation becomes a coupled
system with scattering transfer :math:`\Sigs{g' \to g}` between groups:

.. math::
   :label: multigroup

   \text{streaming} + \Sigt{g} \psi_g
   = \frac{1}{W} \left[
       \sum_{g'} \Sigs{g' \to g} \phi_{g'}
       + \frac{\chi_g}{k} \sum_{g'} \nSigf{g'} \phi_{g'}
   \right]

where the streaming operator depends on the coordinate system and
:math:`\phi_g = \sum_n w_n \psi_{g,n}` is the scalar flux.


.. _quadrature-types:

Angular Quadratures
===================

ORPHEUS provides four angular quadrature types for different geometries.

Gauss--Legendre (1D)
--------------------

For 1D slab and spherical geometry: :math:`N` points on
:math:`\mu \in [-1, 1]`, weights sum to 2.  Optimal for polynomial
integrands (degree :math:`2N-1` exact).  Also used for spherical 1D,
where the single direction cosine :math:`\mu` suffices for the angular
redistribution.

Implemented in :class:`GaussLegendre1D`.

Lebedev (Sphere)
-----------------

For 2D/3D Cartesian geometry: :math:`N` points on the unit sphere with
octahedral symmetry [Lebedev1999]_.  Weights sum to :math:`4\pi`.  On a
1D mesh, z-directed ordinates (:math:`\mu_x = \mu_y = 0`) are handled
as pure collision; all others stream along *x* with *y*-terms cancelling
via reflective BCs.

Implemented in :class:`LebedevSphere`.

Level-Symmetric S\ :sub:`N`
----------------------------

Standard triangular quadrature with :math:`N/2` distinct :math:`\mu_z`
values per hemisphere.  Ordinates on each level are permutations of the
direction cosine set satisfying :math:`\eta^2 + \xi^2 + \mu^2 = 1`.
Equal spacing in :math:`\mu^2` is used with :math:`\mu_1^2 = 4/(N(N+2))`
[CarlsonLathrop1965]_.

Weights sum to :math:`4\pi`.  Provides the ``level_indices`` structure
needed by the cylindrical sweep.  Unlike :class:`ProductQuadrature`
(which has one level per :math:`\mu_z` value), the Level-Symmetric
quadrature groups both :math:`+\mu_z` and :math:`-\mu_z` hemispheres
on the same level (grouped by :math:`|\mu_z|`).  Within each level,
ordinates are sorted by increasing :math:`\eta` for the azimuthal sweep.

Implemented in :class:`LevelSymmetricSN`.

Product Quadrature (GL x equispaced)
-------------------------------------

Tensor product of Gauss--Legendre in :math:`\mu = \cos\theta` (polar)
and equispaced points in :math:`\varphi` (azimuthal).  Each :math:`\mu`
level has the same number of azimuthal points, giving a clean level
structure ideal for the cylindrical sweep.  Weights:

.. math::

   w_{p,m} = w_{\text{GL}}(\mu_p) \cdot \frac{2\pi}{N_\varphi}

Sum to :math:`4\pi`.  Within each level, ordinates are sorted by
increasing :math:`\eta = \sin\theta\cos\varphi` to match the
:math:`\alpha` recursion convention from [Bailey2009]_ Eq. 50.

Implemented in :class:`ProductQuadrature`.

Reflection Index
-----------------

Each quadrature implements a :meth:`reflection_index` method that
returns an index array mapping each ordinate :math:`n` to its
**mirror image** :math:`n'` obtained by negating the direction cosine
along a specified axis.  For example, ``reflection_index("x")``
finds the ordinate whose direction cosines match :math:`(-\mu_x, \mu_y, \mu_z)`.

The implementation in :func:`_find_reflections` computes the
Euclidean distance between the target direction (with one component
negated) and all ordinate directions, then returns the closest match:

.. math::

   n' = \arg\min_j \bigl[
       (\mu_{x,j} - (-\mu_{x,n}))^2
       + (\mu_{y,j} - \mu_{y,n})^2
       + (\mu_{z,j} - \mu_{z,n})^2
   \bigr]

For Gauss--Legendre (1D), the reflection in *x* is simply
:math:`n' = N - 1 - n` because the GL points are symmetric about
zero.  Reflection in *y* is the identity since :math:`\mu_y = 0`.

For multi-dimensional quadratures (Lebedev, Level-Symmetric, Product),
the reflection indices are precomputed at construction time for all
three axes (*x*, *y*, *z*) and stored as ``_ref_x``, ``_ref_y``,
``_ref_z``.  These indices are used by the sweep to implement
reflective boundary conditions (see :ref:`boundary-conditions`).

Comparison Table
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 20

   * - Quadrature
     - Geometry
     - :math:`\sum w`
     - Level structure
     - Best for
   * - Gauss--Legendre
     - Slab, Sphere
     - 2
     - No
     - 1D problems
   * - Lebedev
     - Cartesian 2D
     - :math:`4\pi`
     - No
     - 2D/3D Cartesian
   * - Level-Symmetric
     - Sphere, Cylinder
     - :math:`4\pi`
     - Yes
     - Curvilinear
   * - Product
     - Cylinder
     - :math:`4\pi`
     - Yes
     - Cylindrical 1D


The Discrete Balance Equation
=============================

This is the core of the S\ :sub:`N` method.  The balance equations are
presented from simplest to most complex: Cartesian geometries have no
angular redistribution; curvilinear geometries add :math:`\alpha` coupling
and a geometry factor :math:`\Delta A/w`.

.. _balance-cartesian-1d:

Cartesian 1D Balance Equation
------------------------------

Integrating :eq:`transport-cartesian` over a spatial cell
:math:`[x_{i-1/2}, x_{i+1/2}]` of width :math:`\Delta x_i` and applying
the divergence theorem to the streaming term:

.. math::

   \mu_n \bigl[\psi_{i+\frac12} - \psi_{i-\frac12}\bigr]
   + \Sigt{} \Delta x_i\, \psi_{n,i} = S_i \Delta x_i

where :math:`S_i = Q_i / W` and face areas are unity in slab geometry.
Applying the diamond-difference closure
:math:`\psi_{n,i} = \frac{1}{2}(\psi_{\rm in} + \psi_{\rm out})` and
:math:`\psi_{\rm out} = 2\psi_{n,i} - \psi_{\rm in}`, we solve for the
cell-average angular flux:

.. math::
   :label: dd-cartesian-1d

   \psi_{n,i}
   = \frac{S_i + \dfrac{2|\mu_n|}{\Delta x_i}\, \psi_{\rm in}}
          {\Sigt{} + \dfrac{2|\mu_n|}{\Delta x_i}}

This is the simplest balance equation: no :math:`\alpha` redistribution
and no :math:`\Delta A` factor, because slab geometry has no curvature.
The streaming coefficient :math:`2|\mu|/\Delta x` is precomputed by
:class:`SNMesh` as ``streaming_x[n, i]``.

.. _balance-cartesian-2d:

Cartesian 2D Balance Equation
-------------------------------

Integrating :eq:`transport-cartesian-2d` over a rectangular cell
:math:`\Delta x_i \times \Delta y_j`:

.. math::

   \mu_{x,n}\bigl[\psi_{i+\frac12,j} - \psi_{i-\frac12,j}\bigr] \Delta y_j
   + \mu_{y,n}\bigl[\psi_{i,j+\frac12} - \psi_{i,j-\frac12}\bigr] \Delta x_i
   + \Sigt{} \Delta x_i \Delta y_j\, \psi_{n,i,j}
   = S_{i,j}\, \Delta x_i \Delta y_j

Dividing through by :math:`\Delta x_i \Delta y_j` and applying
diamond-difference closures in **both** directions simultaneously:

.. math::

   \psi_{n,i} &= \tfrac{1}{2}(\psi^x_{\rm in} + \psi^x_{\rm out})
   \qquad\text{(x-closure)} \\
   \psi_{n,i} &= \tfrac{1}{2}(\psi^y_{\rm in} + \psi^y_{\rm out})
   \qquad\text{(y-closure)}

yields the 2D DD equation:

.. math::
   :label: dd-cartesian-2d

   \psi_{n,i,j}
   = \frac{S_{i,j}
     + s_x\, \psi^x_{\rm in}
     + s_y\, \psi^y_{\rm in}}
     {\Sigt{} + s_x + s_y}

where the streaming coefficients are:

.. math::

   s_x = \frac{2|\mu_{x,n}|}{\Delta x_i}, \qquad
   s_y = \frac{2|\mu_{y,n}|}{\Delta y_j}

Both outgoing face fluxes are then updated from the DD closure:

.. math::

   \psi^x_{\rm out} = 2\psi_{n,i,j} - \psi^x_{\rm in}, \qquad
   \psi^y_{\rm out} = 2\psi_{n,i,j} - \psi^y_{\rm in}

These are precomputed by :class:`SNMesh` as ``streaming_x[n, i]`` and
``streaming_y[n, j]``, so the inner loop in
:func:`_sweep_2d_wavefront` reduces to a single vectorised division per
diagonal.

.. _balance-curvilinear:

Curvilinear Balance Equation (Spherical and Cylindrical)
---------------------------------------------------------

Derivation from the Continuous PDE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start with the general 1D curvilinear transport equation.  In
conservative form for a coordinate :math:`r` with face area
:math:`A(r)` and volume element :math:`V`:

.. math::
   :label: conservative-form

   \frac{\mu_n}{V_i}
   \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
   + \frac{1}{V_i}
   \bigl[\alpha_{n+\frac12}\psi_{n+\frac12} - \alpha_{n-\frac12}\psi_{n-\frac12}\bigr]
   + \Sigt{} \psi_{n,i} = S_i

.. vv-status: conservative-form documented

where the streaming cosine is :math:`\mu_n` for spherical and
:math:`\eta_m` for cylindrical, and :math:`S_i = Q_i / W` is the
isotropic source density divided by the quadrature weight sum.

**Step 1: Integrate the PDE over a spatial cell.**

For spherical geometry, integrating :eq:`transport-spherical` over the
shell :math:`[r_{i-1/2}, r_{i+1/2}]` and using the divergence theorem
on the radial streaming gives:

.. math::

   \mu_n \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
   + \int_{V_i} \frac{1-\mu^2}{r} \frac{\partial\psi}{\partial\mu}\, dV
   + \Sigt{} V_i \psi_{n,i} = S_i V_i

For cylindrical geometry, integrating :eq:`transport-cylindrical` over
the annular shell gives:

.. math::

   \eta_m \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
   - \int_{V_i} \frac{1}{r} \frac{\partial(\xi\psi)}{\partial\varphi}\, dV
   + \Sigt{} V_i \psi_{m,i} = S_i V_i

**Step 2: Discretise the angular redistribution.**

The angular integral is discretised as a finite difference in the
ordinate index.  For spherical:

.. math::

   \int_{V_i} \frac{1-\mu^2}{r}\frac{\partial\psi}{\partial\mu}\, dV
   \;\approx\;
   \alpha_{n+\frac12}\psi_{n+\frac12} - \alpha_{n-\frac12}\psi_{n-\frac12}

For cylindrical (per :math:`\mu`-level):

.. math::

   -\int_{V_i} \frac{1}{r}\frac{\partial(\xi\psi)}{\partial\varphi}\, dV
   \;\approx\;
   \alpha_{m+\frac12}\psi_{m+\frac12} - \alpha_{m-\frac12}\psi_{m-\frac12}

**Step 3: Apply the geometry factor** :math:`\Delta A / w`.

The raw discretisation above does NOT preserve per-ordinate flat-flux
consistency.  The correct form from [Bailey2009]_ includes the geometry
factor :math:`\Delta A_i / w_n`:

.. math::
   :label: balance-general

   \mu_n
   \bigl[A_{i+\frac12}\psi_{i+\frac12} - A_{i-\frac12}\psi_{i-\frac12}\bigr]
   + \frac{\Delta A_i}{w_n}
   \bigl[\alpha_{n+\frac12}\psi_{n+\frac12} - \alpha_{n-\frac12}\psi_{n-\frac12}\bigr]
   + \Sigt{} V_i \psi_{n,i} = S_i V_i

where :math:`\Delta A_i = A_{i+1/2} - A_{i-1/2}`.  This is
[Bailey2009]_ Eq. 7--10 for spherical and Eq. 50--55 for cylindrical.

Note why :eq:`dd-cartesian-1d` has no :math:`\alpha` or :math:`\Delta A`
terms: in Cartesian geometry the face area is unity (:math:`A = 1`), so
:math:`\Delta A = 0`, and there is no curvature to redistribute angular
flux.

The Alpha Redistribution Coefficients
--------------------------------------

The :math:`\alpha` coefficients encode how the angular flux redistributes
between neighbouring ordinates due to the geometry curvature.  They are
defined recursively:

.. math::
   :label: alpha-recursion

   \alpha_{n+\frac12} = \alpha_{n-\frac12} - w_n \mu_n

with the boundary condition :math:`\alpha_{1/2} = 0`.

For **spherical** geometry, all :math:`N` ordinates form a single
sequence sorted by :math:`\mu` (most negative to most positive).
The :math:`\alpha` values form a **non-negative dome**: they rise while
:math:`\mu < 0`, peak near :math:`\mu = 0`, and fall back to zero at
:math:`\mu = 1`.  The endpoint condition
:math:`\alpha_{N+1/2} = 0` is guaranteed by Gauss--Legendre
antisymmetry: :math:`\sum_n w_n \mu_n = 0`.

For **cylindrical** geometry, each :math:`\mu`-level has its own
independent :math:`\alpha` sequence.  On level :math:`p`, the ordinates
are sorted by increasing :math:`\eta` (radial direction cosine), and the
recursion uses :math:`\eta` instead of :math:`\mu`:

.. math::
   :label: alpha-cylindrical

   \alpha_{p,m+\frac12} = \alpha_{p,m-\frac12} - w_m \eta_m

This is [Bailey2009]_ Eq. 50.  Each level's :math:`\alpha` values form
an independent dome from :math:`\eta = -\sin\theta` to
:math:`\eta = +\sin\theta`.

**Dome shape properties:**

- :math:`\alpha_{n+1/2} \geq 0` for all :math:`n` (non-negative dome).
- The peak occurs near the ordinate where :math:`\mu_n` (or
  :math:`\eta_m`) crosses zero.
- The dome height scales with the quadrature weight sum: higher-order
  quadratures have narrower but taller domes.
- Non-negativity ensures the denominator of the DD equation is
  unconditionally positive, guaranteeing numerical stability.

The code stores these in ``SNMesh.alpha_half`` (spherical, shape
``(N+1,)``) and ``SNMesh.alpha_per_level`` (cylindrical, list of
``(M+1,)`` arrays).

The Geometry Factor and Why It Is Needed
-----------------------------------------

The geometry factor :math:`\Delta A_i / w_n` in :eq:`balance-general`
is the key to correct curvilinear transport.  Without it, the balance
equation violates **per-ordinate flat-flux consistency**: for a spatially
uniform, isotropic flux :math:`\psi = \text{const}`, the streaming and
redistribution terms should cancel exactly for EACH ordinate
individually.

**Proof of consistency.**

Set :math:`\psi_{n,i} = \psi_{n+1/2} = \psi_{n-1/2} = \psi_0` (flat
in both space and angle) and :math:`\psi_{i+1/2} = \psi_{i-1/2} = \psi_0`
(flat in space).  The streaming term becomes:

.. math::

   \mu_n \bigl[A_{i+\frac12} - A_{i-\frac12}\bigr] \psi_0
   = \mu_n \,\Delta A_i\, \psi_0

The redistribution term with the :math:`\Delta A/w` factor becomes:

.. math::

   \frac{\Delta A_i}{w_n}
   \bigl[\alpha_{n+\frac12} - \alpha_{n-\frac12}\bigr] \psi_0
   = \frac{\Delta A_i}{w_n} (-w_n \mu_n) \psi_0
   = -\mu_n \,\Delta A_i\, \psi_0

where we used the recursion :eq:`alpha-recursion`:
:math:`\alpha_{n+1/2} - \alpha_{n-1/2} = -w_n \mu_n`.  The two terms
cancel exactly, giving :math:`\Sigt{} \psi_0 = S_0`, which is the
correct homogeneous solution.

**Without** the :math:`\Delta A/w` factor (i.e., using
:math:`[\alpha_{n+1/2}\psi_{n+1/2} - \alpha_{n-1/2}\psi_{n-1/2}]`
directly), the redistribution term for flat flux is
:math:`(-w_n \mu_n)\psi_0`, but the streaming term is
:math:`\mu_n \Delta A_i \psi_0`.  These differ by a factor of
:math:`\Delta A_i`, so consistency only holds in the limit
:math:`\Delta A_i \to 0` (i.e., at the origin or on an infinitely fine
mesh).

**Consequence of the missing factor:**  The solver creates artificial
angular anisotropy that *worsens* with mesh refinement near :math:`r = 0`
(where :math:`\Delta A_i` is smallest but non-zero).  This manifests as
a flux spike at the origin in fixed-source problems and as divergent
eigenvalues in heterogeneous eigenvalue problems.

The code precomputes this factor as ``SNMesh.redist_dAw`` (spherical,
shape ``(nx, N)``) and ``SNMesh.redist_dAw_per_level`` (cylindrical,
list of ``(nx, M)`` arrays).

The Morel--Montry Flux Dip
----------------------------

Even with the correct :math:`\Delta A/w` factor, the standard
diamond-difference closure (equal weight :math:`\tau = 0.5`) introduces
a flux error near :math:`r = 0` known as the **Morel--Montry flux dip**
[MorelMontry1984]_.

The standard DD angular closure is:

.. math::

   \psi_{n,i} = \frac{1}{2}(\psi_{n-\frac12} + \psi_{n+\frac12})

This can be rewritten as:

.. math::

   \psi_{n+\frac12} = 2\psi_{n,i} - \psi_{n-\frac12}

The contamination factor :math:`\beta` ([Bailey2009]_ Eq. 41) quantifies
the coupling between the leading-order scalar flux and the first-order
current in the asymptotic diffusion limit.  For spherical geometry:

.. math::

   \beta = \frac{1}{2} \sum_{n=1}^{N} \mu_n
   \bigl[\alpha_{n+\frac12}\, \mu_{n+\frac12}
        - \alpha_{n-\frac12}\, \mu_{n-\frac12}\bigr]

where :math:`\mu_{n\pm 1/2}` are the angular cell-edge cosines.  For
cylindrical, the equivalent is a per-level sum using :math:`\eta` and
:math:`\eta_{m\pm 1/2}`.  When :math:`\beta \neq 0`, the discrete
S\ :sub:`N` equations satisfy a **contaminated** diffusion equation near
:math:`r = 0`, producing the artificial flux dip (or spike).

The module :mod:`derivations.sn_contamination` computes :math:`\beta`
for any quadrature and geometry.  With the correct :math:`\Delta A/w`
factor AND Morel--Montry weights, :math:`\beta \sim 10^{-16}`
(machine zero) for both spherical and cylindrical.

Weighted Diamond Difference (WDD) and Morel--Montry Weights
-------------------------------------------------------------

The Morel--Montry (M-M) angular closure replaces the equal-weight DD
with position-dependent weights :math:`\tau_n` [Bailey2009]_ Eq. 74:

.. math::
   :label: wdd-closure

   \psi_{n,i} = (1 - \tau_n)\,\psi_{n-\frac12} + \tau_n\,\psi_{n+\frac12}

Solving for the angular face flux:

.. math::
   :label: wdd-face

   \psi_{n+\frac12}
   = \frac{\psi_{n,i} - (1 - \tau_n)\,\psi_{n-\frac12}}{\tau_n}

The M-M weights are defined as:

.. math::
   :label: mm-weights

   \tau_n = \frac{\mu_n - \mu_{n-\frac12}}{\mu_{n+\frac12} - \mu_{n-\frac12}}

where :math:`\mu_{n\pm 1/2}` are the angular cell edges.

**Spherical cell edges:**  :math:`\mu_{1/2} = -1`,
:math:`\mu_{N+1/2} = +1`, and interior edges by weight-sum:
:math:`\mu_{n+1/2} = \mu_{n-1/2} + w_n`.  This is exact for
Gauss--Legendre quadrature because the weights correspond to the
:math:`\mu`-space widths of the angular cells.

**Cylindrical cell edges:**  :math:`\eta_{1/2} = -\sin\theta`,
:math:`\eta_{M+1/2} = +\sin\theta`, and interior edges at
**midpoints** of consecutive :math:`\eta` values:
:math:`\eta_{m+1/2} = (\eta_m + \eta_{m+1})/2`.
The weight-sum approach is NOT used for cylindrical because the
quadrature weights are uniform in :math:`\varphi`-space (not
:math:`\eta`-space): the Product quadrature spaces :math:`\varphi`
equally, but :math:`\eta = \sin\theta\cos\varphi` is
cosine-distributed, so equal :math:`\varphi`-widths map to unequal
:math:`\eta`-widths.  The midpoint approach gives a proper partition
of :math:`[-\sin\theta, +\sin\theta]`.

For the Product quadrature with equally-spaced :math:`\varphi`,
ordinates come in **pairs** with the same :math:`|\eta|` but opposite
:math:`\xi` (e.g., :math:`\varphi = \pi/4` and :math:`\varphi = 7\pi/4`
both give :math:`\eta = \sin\theta/\sqrt{2}`).  The midpoint between
paired ordinates equals their shared :math:`\eta`, creating zero-width
angular cells.  The resulting :math:`\tau` alternates between 0.5
(DD, for the left member of each pair) and 1.0 (step, for the right
member).  This alternating pattern is correct but could be smoothed by
using a Gauss-type azimuthal quadrature with distinct :math:`\eta`
values (see ``IMPROVEMENTS.md`` item DO-20260405-002).

The M-M weights force the contamination factor :math:`\beta` to **machine
zero** (verified: :math:`\beta \sim 10^{-16}`), completely eliminating
the Morel--Montry flux dip.

**Clipping:** The code clips :math:`\tau_n` to :math:`[0.5, 1.0]` to
prevent negative angular face fluxes.  This preserves the stability of
the standard DD while gaining the accuracy of the M-M correction.

The code stores these as ``SNMesh.tau_mm`` (spherical, shape ``(N,)``)
and ``SNMesh.tau_mm_per_level`` (cylindrical, list of ``(M,)`` arrays).

Substituting the WDD Closure into the Balance Equation
-------------------------------------------------------

Combining the balance equation :eq:`balance-general` with the WDD
angular closure :eq:`wdd-closure` and the standard spatial DD
(:math:`\psi_{n,i} = \frac{1}{2}(\psi_{\rm in}^s + \psi_{\rm out}^s)`,
:math:`\psi_{\rm out}^s = 2\psi_{n,i} - \psi_{\rm in}^s`), define:

.. math::

   c_{\rm out} &= \frac{\alpha_{n+\frac12}}{\tau_n} \\[6pt]
   c_{\rm in}  &= \frac{(1-\tau_n)}{\tau_n}\,\alpha_{n+\frac12}
                 + \alpha_{n-\frac12}

The cell-average angular flux is then:

.. math::
   :label: dd-solve

   \psi_{n,i} = \frac{
       S_i V_i
       + |\mu_n|(A_{\rm in} + A_{\rm out})\,\psi_{\rm in}^s
       + \dfrac{\Delta A_i}{w_n}\, c_{\rm in}\, \psi_{n-\frac12}
   }{
       2|\mu_n|\, A_{\rm out}^s
       + \dfrac{\Delta A_i}{w_n}\, c_{\rm out}
       + \Sigt{} V_i
   }

where the superscript :math:`s` denotes spatial face fluxes, and
:math:`A_{\rm in}`, :math:`A_{\rm out}` are the cell face areas in the
direction of neutron travel (see :ref:`sweep-algorithm` below for their
definition).  This is the equation solved by both
:func:`_sweep_1d_spherical` and :func:`_sweep_1d_cylindrical`.

Geometry Comparison
--------------------

.. list-table::
   :header-rows: 1
   :widths: 15 28 28 29

   * - Aspect
     - Cartesian
     - Spherical
     - Cylindrical
   * - Streaming cosine
     - :math:`\mu`
     - :math:`\mu`
     - :math:`\eta` (radial)
   * - Face area :math:`A`
     - 1 (per unit area)
     - :math:`4\pi r^2`
     - :math:`2\pi r`
   * - Volume :math:`V`
     - :math:`\Delta x`
     - :math:`\tfrac{4}{3}\pi(r_{\rm out}^3 - r_{\rm in}^3)`
     - :math:`\pi(r_{\rm out}^2 - r_{\rm in}^2)`
   * - :math:`\Delta A`
     - 0
     - :math:`4\pi(r_{\rm out}^2 - r_{\rm in}^2)`
     - :math:`2\pi(r_{\rm out} - r_{\rm in})`
   * - Redistribution
     - None
     - :math:`+(\Delta A/w)\,[\alpha\psi]`
     - :math:`+(\Delta A/w)\,[\alpha\psi]`
   * - :math:`\alpha` scope
     - N/A
     - Global (all :math:`N` ordinates)
     - Per :math:`\mu`-level
   * - :math:`\alpha` recursion variable
     - N/A
     - :math:`\mu`
     - :math:`\eta`
   * - Quadrature required
     - GL or Lebedev
     - GL
     - Product or Level-Sym


.. _sweep-algorithm:

Sweep Algorithm
===============

Because each cell's outgoing flux becomes the next cell's incoming flux,
the equations must be solved in the direction of neutron travel --- this
is called a **transport sweep**.

.. _sweep-cumprod:

Cartesian 1D: Cumprod Recurrence
---------------------------------

For the 1D slab with Gauss--Legendre quadrature, the DD equation
:eq:`dd-cartesian-1d` defines a recurrence for the outgoing face flux:

.. math::
   :label: dd-recurrence

   \psi_{\rm out} = a_i\, \psi_{\rm in} + b_i

where the coefficients for cell :math:`i` are:

.. math::

   a_i = \frac{2|\mu_n|/\Delta x_i - \Sigt{}}
              {2|\mu_n|/\Delta x_i + \Sigt{}},
   \qquad
   b_i = \frac{S_i}
              {2|\mu_n|/\Delta x_i + \Sigt{}}

This arises from substituting the DD closure
:math:`\psi_{\rm out} = 2\psi_{\rm avg} - \psi_{\rm in}` into
:eq:`dd-cartesian-1d`.  The coefficient :math:`a_i` is the
**stream-to-collision ratio**: it controls how much incoming flux
propagates through cell :math:`i`.

Unrolling the recurrence :math:`\psi_{\rm out}^{(i)} = a_i\, \psi_{\rm out}^{(i-1)} + b_i`
gives a linear first-order relation that can be solved analytically
using **cumulative products**.  Define:

.. math::

   C_i = \prod_{k=0}^{i} a_k, \qquad
   R_i = \sum_{k=0}^{i} \frac{b_k}{C_k}

Then the incoming face flux at cell :math:`i+1` is:

.. math::

   \psi_{\rm in}^{(i+1)} = C_i \bigl(\psi_{\rm in}^{(0)} + R_i\bigr)

and the cell-average flux is :math:`\psi_{\rm avg}^{(i)} = \frac{1}{2}(\psi_{\rm in}^{(i)} + \psi_{\rm out}^{(i)})`.

The implementation in :func:`_sweep_1d_cumprod` computes :math:`C` and
:math:`R` via ``np.cumprod`` and ``np.cumsum``, giving an
:math:`O(N \cdot n_x)` **vectorised** sweep --- all spatial cells for a
given ordinate are resolved simultaneously in numpy array operations,
with no Python-level cell loop.  This typically runs in sub-millisecond
time for practical meshes.

Exploiting GL symmetry, only positive-:math:`\mu` ordinates are swept
forward; negative-:math:`\mu` ordinates are obtained by reversing the
cell array and sweeping with the same coefficients.

.. _sweep-wavefront:

Cartesian 2D: Anti-Diagonal Wavefront Sweep
---------------------------------------------

In 2D, the DD equation :eq:`dd-cartesian-2d` creates a data dependency:
cell :math:`(i, j)` requires incoming face fluxes from its upwind
neighbours in both :math:`x` and :math:`y`.  Cells along an
**anti-diagonal** :math:`i + j = k` are mutually independent because
they share no incoming faces, so they can be solved simultaneously.

**The four quadrant sweeps.**  Each ordinate has a sign pair
:math:`(\text{sgn}(\mu_x), \text{sgn}(\mu_y))` that determines the
sweep direction.  The four combinations define four sweep patterns:

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - :math:`\mu_x`
     - :math:`\mu_y`
     - *x*-direction
     - *y*-direction
   * - :math:`+`
     - :math:`+`
     - left :math:`\to` right
     - bottom :math:`\to` top
   * - :math:`-`
     - :math:`+`
     - right :math:`\to` left
     - bottom :math:`\to` top
   * - :math:`+`
     - :math:`-`
     - left :math:`\to` right
     - top :math:`\to` bottom
   * - :math:`-`
     - :math:`-`
     - right :math:`\to` left
     - top :math:`\to` bottom

For each direction pair, the sweep visits anti-diagonals
:math:`k = 0, 1, \ldots, n_x + n_y - 2`.  On diagonal :math:`k`, the
cells :math:`(i, j)` satisfying :math:`i + j = k` (in the swept index
space) are gathered into a numpy batch and solved with a single vectorised
evaluation of :eq:`dd-cartesian-2d`.

**Vectorisation within each diagonal.**  Each diagonal contains up to
:math:`\min(n_x, n_y)` cells.  The incoming face fluxes ``psi_in_x``
and ``psi_in_y`` are gathered by advanced indexing; the DD equation is
evaluated as one numpy operation; and the outgoing face fluxes are
scattered back.  There is no Python-level cell loop within a diagonal.

**Reflective BCs in 2D.**  At each boundary face, the incoming flux for
ordinate :math:`n` is set to the outgoing flux of its reflected partner.
For the left/right boundaries (*x*-reflection), the partner is
``ref_x[n]`` (negating :math:`\mu_x`); for the top/bottom boundaries
(*y*-reflection), the partner is ``ref_y[n]`` (negating :math:`\mu_y`).
The reflection indices are precomputed by the quadrature's
:meth:`reflection_index` method.

Implemented in :func:`_sweep_2d_wavefront`.

Curvilinear 1D: Sequential Ordinate Sweep
------------------------------------------

For spherical and cylindrical geometries, the angular redistribution
couples successive ordinates, preventing vectorisation across the
ordinate dimension.  The sweep proceeds cell-by-cell,
ordinate-by-ordinate:

**Spherical:** Ordinates are processed from most negative :math:`\mu` to
most positive (a single global sequence).  Negative-:math:`\mu` ordinates
sweep **inward** (outer boundary to centre); positive-:math:`\mu`
ordinates sweep **outward** (centre to outer boundary).

**Cylindrical:** For each :math:`\mu`-level, azimuthal ordinates are
processed from most-inward (:math:`\eta = -\sin\theta`) to most-outward
(:math:`\eta = +\sin\theta`).  Negative-:math:`\eta` ordinates sweep
inward; :math:`\eta \approx 0` ordinates have no radial streaming
(pure redistribution); positive-:math:`\eta` ordinates sweep outward.

At each cell, the sweep solves :eq:`dd-solve` for :math:`\psi_{n,i}`,
then updates:

1. **Spatial face flux:**
   :math:`\psi_{\rm out}^s = 2\psi_{n,i} - \psi_{\rm in}^s`
2. **Angular face flux:**
   :math:`\psi_{n+1/2} = (\psi_{n,i} - (1-\tau_n)\psi_{n-1/2})/\tau_n`
   (using M-M weights)
3. **Scalar flux accumulation:**
   :math:`\phi_i \mathrel{+}= w_n \psi_{n,i}`

The spatial face flux propagates to the next cell; the angular face flux
propagates to the next ordinate on the same cell.

Implemented in :func:`_sweep_1d_spherical` and
:func:`_sweep_1d_cylindrical`.

Starting Direction
-------------------

Both curvilinear sweeps initialise the angular face flux
:math:`\psi_{1/2}` to **zero** for each spatial cell.  This is valid
because :math:`\alpha_{1/2} = 0` by construction, so the product
:math:`\alpha_{1/2} \psi_{1/2}` in the balance equation vanishes
regardless of the value of :math:`\psi_{1/2}`.  The starting-direction
treatment of [LewisMiller1984]_ Section 4.5.4 (which tracks
:math:`\alpha\psi` instead of :math:`\psi` alone) is therefore
unnecessary when the :math:`\alpha` recursion is implemented correctly.


BiCGSTAB Alternative
====================

Instead of sweep-based source iteration, the within-group transport
problem can be solved directly using a Krylov method (BiCGSTAB).

Explicit Transport Operator
----------------------------

The finite-difference transport operator :math:`T` is formed explicitly:

.. math::

   T\psi = \mu_n \nabla\psi + \Sigt{}\psi

For Cartesian geometry, this is a banded matrix with finite-difference
gradients.  For curvilinear geometries, the operator includes the same
:math:`\Delta A/w` geometry factor and Morel--Montry angular closure
weights used by the sweep.  The system :math:`T\psi = b` is solved with
``scipy.sparse.linalg.bicgstab``.

Consistency with the Sweep
---------------------------

Both the sweep and BiCGSTAB path must use the **same** spatial
discretisation to produce identical eigenvalues.  In practice:

- The sweep uses diamond-difference (DD): :math:`T^{-1}` is applied
  implicitly.
- BiCGSTAB forms :math:`T` explicitly using finite-difference (FD)
  gradients.

On coarse meshes, DD and FD have different truncation error constants,
so the two paths may give slightly different :math:`\keff` values.  They
converge to the same answer as :math:`h \to 0`.

The curvilinear BiCGSTAB operators read ``redist_dAw`` (spherical) and
``redist_dAw_per_level`` (cylindrical) from :class:`SNMesh`, along with the
M-M weights ``tau_mm`` / ``tau_mm_per_level``.  This ensures both paths
share exactly the same physics.


.. _boundary-conditions:

Boundary Conditions
===================

ORPHEUS uses **reflective boundary conditions** on all faces, representing
an infinite lattice or infinite medium.  The CP solver uses white
(isotropic) BCs instead; see :ref:`white-bc-quality` for a comparison
showing the ~1% gap between the two approaches.

Outer Boundary
--------------

At the outer boundary :math:`r = R` (or :math:`x = L`), the incoming
flux for ordinate :math:`n` is set to the outgoing flux of its reflected
partner:

.. math::
   :label: reflective-bc

   \psi_n^{\rm in} = \psi_{n'}^{\rm out}

where :math:`n'` is the reflected partner ordinate (negating the
appropriate direction cosine).  Reflective partner indices are precomputed
by each quadrature's :meth:`reflection_index` method.

Inner Boundary (Curvilinear)
-----------------------------

At :math:`r = 0`:

- The face area :math:`A(0) = 0`, so **no spatial flux crosses the
  origin**.  The spatial incoming flux for outward-sweeping ordinates is
  zero.
- The **angular redistribution provides the inward-to-outward
  transition**: flux entering as an inward-directed ordinate
  (:math:`\mu < 0` or :math:`\eta < 0`) is redistributed to outward
  ordinates (:math:`\mu > 0` or :math:`\eta > 0`) through the
  :math:`\alpha` coupling.

This means the curvilinear sweep does not need an explicit boundary
condition at :math:`r = 0` --- the geometry handles it naturally.


Scattering
==========

Matrix Convention
-----------------

The ``Mixture.SigS[l]`` matrices use the convention
:math:`\text{SigS}[g_{\rm from}, g_{\rm to}]`:

.. math::

   \text{SigS}[0] = \begin{pmatrix}
       \Sigs{0\to0} & \Sigs{0\to1} \\
       \Sigs{1\to0} & \Sigs{1\to1}
   \end{pmatrix}

For the in-scatter source (total scattering into group :math:`g` from
all groups :math:`g'`):

.. math::

   Q_{\rm scatter}[g]
   = \sum_{g'} \Sigs{g'\to g}\, \phi_{g'}
   = (\text{SigS}^T \cdot \phi)[g]

The vectorised form for batched cells is ``phi @ SigS`` (equivalent to
:math:`(\text{SigS}^T \phi^T)^T` for row-vector :math:`\phi`).

The **analytical eigenvalue problem** uses:

.. math::

   A &= \text{diag}(\Sigt{}) - \text{SigS}^T \quad\text{(removal matrix)} \\
   F &= \chi \otimes \nSigf{} \quad\text{(fission matrix)} \\
   \kinf &= \lambda_{\max}(A^{-1} F)

Note the transpose: :math:`\text{SigS}^T[g, g'] = \Sigs{g'\to g}` gives
the in-scatter contribution, so :math:`\text{diag}(\Sigt{}) - \text{SigS}^T`
is the net removal matrix.  See :ref:`scattering-matrix-convention` for the
full derivation of this convention.

P\ :sub:`0` Isotropic Scattering
----------------------------------

The default mode (``scattering_order=0``).  A direction-independent
source is added to all ordinates equally:

.. math::

   Q_{\rm scatter}(\hat{\Omega}_n)
   = \sum_{g'} \Sigs{g'\to g}^{(0)}\, \phi_{g'} / W

Implemented in :meth:`SNSolver._add_scattering_source`, which performs
``phi @ SigS[0]`` per material.

.. _pn-scattering:

P\ :sub:`N` Anisotropic Scattering
------------------------------------

When ``scattering_order >= 1``, per-ordinate anisotropic sources are
computed from the Legendre moments of the angular flux.  The full
anisotropic scattering source for ordinate :math:`n` and group :math:`g`
is:

.. math::
   :label: pn-scatter

   Q_{\rm scatter}(\hat{\Omega}_n, g)
   = \sum_{\ell=0}^{L} (2\ell+1)
     \sum_{m=-\ell}^{\ell}
     \sum_{g'} \Sigs{g'\to g}^{(\ell)}\,
     f_{\ell,g'}^m \; Y_\ell^m(\hat{\Omega}_n)

where :math:`Y_\ell^m` are real spherical harmonics and the angular flux
moments are computed by quadrature:

.. math::
   :label: flux-moments

   f_{\ell,g}^m = \sum_{n=1}^{N} w_n \, \psi_{n,g} \, Y_\ell^m(\hat{\Omega}_n)

The :math:`(2\ell+1)` factor is the addition theorem normalisation for
real spherical harmonics: it ensures that the P\ :sub:`L` expansion
reproduces the angular flux moments exactly when the angular flux is a
polynomial of degree :math:`\leq L`.

**Implementation in** :meth:`SNSolver._build_aniso_scattering`:

1. **Compute spherical harmonics** at construction time:
   :math:`Y[n, \ell, \ell+m]` for all ordinates, stored as ``self._Y``
   with shape ``(N, L+1, 2L+1)``.  The convention is
   :math:`Y_0^0 = 1`, :math:`Y_1^{-1} = \mu_z`,
   :math:`Y_1^0 = \mu_x`, :math:`Y_1^1 = \mu_y`.

2. **Compute flux moments** via an ``einsum`` contraction over the
   ordinate index:

   .. code-block:: python

      fiL[:, :, :, l, l+m] = np.einsum(
          'n,nxyg->xyg', w * Y[:, l, l+m], angular_flux,
      )

   This contracts :math:`\sum_n w_n Y_\ell^m(\hat{\Omega}_n) \psi_n(x,y,g)`
   into a spatial-energy field of shape ``(nx, ny, ng)``.

3. **Reconstruct per-ordinate source**: for each Legendre order
   :math:`\ell \geq 1` (the :math:`\ell = 0` term is handled by
   :meth:`SNSolver._add_scattering_source`) and each :math:`m`, the
   scattered moment ``moment @ sig_s_l[l]`` is multiplied by
   :math:`(2\ell+1) Y_\ell^m(\hat{\Omega}_n)` and accumulated into
   ``Q_aniso[n, :, :, :]``.

4. The resulting ``Q_aniso`` array of shape ``(N, nx, ny, ng)`` is
   passed to :func:`transport_sweep`, which adds it to the isotropic
   source on a per-ordinate basis.

**Equivalence of the code to the mathematical form.**
Equation :eq:`pn-scatter` writes the sum as
:math:`\sum_\ell \sum_m \sum_{g'} \Sigs{}^{(\ell)} f_\ell^m Y_\ell^m`.
The code separates the :math:`\ell = 0` term (isotropic, handled by
``_add_scattering_source``) from the :math:`\ell \geq 1` terms
(anisotropic, handled by ``_build_aniso_scattering``).  For :math:`\ell = 0`,
:math:`Y_0^0 = 1` and :math:`(2 \cdot 0 + 1) = 1`, so the sum reduces to
:math:`\sum_{g'} \Sigs{g' \to g}^{(0)} f_{0,g'}^0 = \sum_{g'} \Sigs{g' \to g}^{(0)} \phi_{g'}`,
which is exactly the P\ :sub:`0` source.  The split is therefore exact
with no double-counting.

The 421-group cross-section library provides both P0 and P1 matrices.

.. _n2n-reactions:

(n,2n) Reactions
-----------------

The :math:`(n,2n)` reaction is a threshold reaction in which a neutron
is absorbed by a nucleus, which then emits **two** neutrons.  The net
effect is a gain of one neutron per reaction (the incident neutron is
consumed, two are produced).

The :math:`(n,2n)` cross section is stored as a group-to-group transfer
matrix ``Mixture.Sig2`` with the same ``[g_from, g_to]`` convention as
the scattering matrix.  The source contribution is:

.. math::

   Q_{(n,2n)}(g) = 2 \sum_{g'} \Sigma_{2,g'\to g}\, \phi_{g'}

The factor of 2 accounts for the two neutrons produced per reaction.
The implementation in :meth:`SNSolver._add_n2n_source` performs:

.. code-block:: python

   Q[ix, iy, :] += 2.0 * (phi[ix, iy, :] @ self.sig2[mid])

This is added to the isotropic source before the transport sweep, on the
same footing as the P\ :sub:`0` scattering source.  The :math:`(n,2n)`
contribution also enters the :math:`\keff` production term in
:meth:`SNSolver.compute_keff`, where row sums of ``Sig2`` (total
:math:`(n,2n)` removal rate) are used.

Normalization Chain
--------------------

The normalization chain in the code ensures consistent scaling:

1. **Fission source** (:meth:`SNSolver.compute_fission_source`):
   :math:`Q_f = \chi \cdot (\nSigf{} \cdot \phi) / k` --- raw,
   un-normalised.

2. **Scattering source** (:meth:`SNSolver._add_scattering_source`):
   :math:`Q_s = \text{SigS}^T \cdot \phi` --- also un-normalised.

3. **Sweep** (:func:`transport_sweep`): applies
   :math:`Q_{\rm scaled} = Q \cdot w_{\rm norm}` where
   :math:`w_{\rm norm} = 1/\sum w_n`.  This is the :math:`1/W` division
   in the S\ :sub:`N` equation.

4. **Scalar flux** (inside sweep):
   :math:`\phi = \sum_n w_n \psi_n` --- standard quadrature integration.

5. **keff** (:meth:`SNSolver.compute_keff`):
   :math:`k = (\nSigf{} \cdot \phi \cdot V) / (\Sigma_a \cdot \phi \cdot V)`
   --- volume-weighted ratio.

The :math:`1/W` in step 3 and the :math:`W` implicit in step 4 cancel:
:math:`\phi = \sum w_n \cdot Q/(W \Sigt{}) = Q/\Sigt{}` for uniform
isotropic source.

**Convention rule:** Sources passed to the sweep must NOT include
:math:`1/W` --- the sweep applies it.  The BiCGSTAB path (direct
operator) must divide sources by :math:`W` itself, since it solves
:math:`T\psi = b` without the sweep.


The Eigenvalue Problem
======================

The eigenvalue :math:`\keff` is determined by **power iteration**: an
outer loop updates :math:`k` from the production/absorption ratio, with
an inner loop that solves the within-group scattering problem.

Two Inner Solvers
-----------------

**Source iteration (sweep-based):**

- Operator: :math:`T^{-1}` (diamond-difference sweep)
- Solution variable: scalar flux :math:`\phi(x, y, g)`
- Fixed-point: :math:`\phi^{(k+1)} = T^{-1}(S \cdot \phi^{(k)} + Q_f)`
- Convergence rate: spectral radius of :math:`T^{-1}S` (~0.97 for 421
  groups)
- Cost per iteration: one transport sweep
- Works for all geometries

**BiCGSTAB (direct operator):**

- Operator: :math:`T = \mu \nabla + \Sigt{}` (finite-difference
  gradients)
- Solution variable: angular flux :math:`\psi(x, y, n, g)` (much
  larger)
- System: :math:`T\psi = b` where :math:`b` = fission + scattering
- Convergence: ~100 Krylov iterations at ``tol=1e-4`` (always converges)
- Available for all geometries (Cartesian, spherical, cylindrical)

The two architectures use **different spatial discretisations** (DD
sweep vs FD gradient) that converge to different :math:`\keff` on coarse
meshes.  They agree in the limit :math:`h \to 0`.


Verification
============

.. _sn-mms-verification:

Method of Manufactured Solutions (1D slab)
-------------------------------------------

Homogeneous and heterogeneous eigenvalue tests verify :math:`\keff`
--- a scalar. They do not tell us whether the **spatial operator**
itself converges at the design order :math:`\mathcal O(h^{2})` of
diamond difference.  The Method of Manufactured Solutions closes
that gap by constructing a fixed-source problem whose exact angular
flux is known in closed form, so the error against the prescribed
flux is pure spatial-discretisation error.

**Ansatz.**  For a vacuum-BC slab of length :math:`L` in one energy
group, pick an isotropic angular flux

.. math::
   :label: sn-mms-psi

   \psi_n(x) = \frac{1}{W}\,A(x),
   \qquad A(x) = \sin\!\left(\frac{\pi x}{L}\right),

where :math:`W = \sum_n w_n = 2` for Gauss--Legendre.  Because
:math:`A(0) = A(L) = 0`, every ordinate vanishes at both faces ---
the vacuum boundary conditions are satisfied automatically, with no
inflow bookkeeping required on the caller side.  Since :math:`\psi_n`
is independent of ordinate, the scalar flux recovered by any
quadrature order is *exactly* :math:`\phi(x) = A(x)` --- the test
isolates spatial error from angular quadrature error.

**Manufactured source.**  Substituting :eq:`sn-mms-psi` into the
discrete ordinates transport equation :eq:`transport-cartesian`
(with the :math:`1/W` convention ORPHEUS uses),

.. math::

   \mu_n\,\frac{\partial\psi_n}{\partial x} + \Sigma_t\,\psi_n
   = \frac{1}{W}\!\left(\Sigma_s\,\phi + Q^{\text{ext}}_n\right),

and solving algebraically for :math:`Q^{\text{ext}}_n` gives

.. math::
   :label: sn-mms-qext

   Q^{\text{ext}}_n(x)
   = \mu_n\,A'(x) + \bigl(\Sigma_t - \Sigma_s\bigr)\,A(x)
   = \mu_n\,\frac{\pi}{L}\cos\!\left(\frac{\pi x}{L}\right)
     + \bigl(\Sigma_t - \Sigma_s\bigr)\sin\!\left(\frac{\pi x}{L}\right).

The :math:`W` factor cancels cleanly because the ansatz was already
divided by :math:`W`, so what we hand the solver is the full residual
without any additional rescaling.  The expression is per-ordinate and
linear in :math:`\mu_n`: a constant isotropic external source *cannot*
drive a non-trivial manufactured flux because the streaming term
:math:`\mu_n\,\psi'_n` is odd in :math:`\mu`.  That is the fundamental
reason MMS for SN requires the :math:`Q_{\rm aniso}` plumbing path ---
no "cheat" with a cell-by-cell isotropic source exists.

**Why :math:`\sin(\pi x/L)`?**  The ansatz is smooth
(:math:`C^{\infty}`) so all derivatives of the exact solution exist
and DD's :math:`\mathcal O(h^{2})` truncation error dominates.  It
vanishes at both boundaries for free.  Its derivatives do not collapse
to a polynomial --- a cubic ansatz, for instance, has a constant
second derivative so the DD truncation term :math:`\psi'''` would be
zero and the error could disappear for a non-physical reason,
hiding bugs.  Trigonometric or exponential ansätze have bounded
but non-zero derivatives of every order and therefore expose the
leading truncation term cleanly.

**Implementation.**  The case is built by
:func:`orpheus.derivations.sn_mms.build_1d_slab_mms_case` and
consumed by :func:`orpheus.sn.solve_sn_fixed_source`.  The latter
accepts a per-ordinate external source of shape
:math:`(N, n_x, n_y, n_g)` and threads it through the sweep's
:math:`Q_{\rm aniso}` slot --- merging additively with any P1+
scattering contribution the solver itself builds.  The vacuum
boundary condition is a new parameter on
:func:`orpheus.sn.sweep.transport_sweep`; for ``"vacuum"`` the
:math:`2\mathrm D` wavefront path skips the reflective-partner
copy, leaving incoming-face angular fluxes at their zero
initialisation (which is correct because no code path writes the
incoming-face slot of any ordinate except the reflection step
itself).

**Measured convergence.**  With
:math:`\Sigma_t = 1\ \mathrm{cm^{-1}}`,
:math:`\Sigma_s = 0.5\ \mathrm{cm^{-1}}`,
:math:`L = 5\ \mathrm{cm}`, Gauss--Legendre :math:`S_{16}`:

.. list-table::
   :header-rows: 1
   :widths: 10 20 20

   * - :math:`n_{\rm cells}`
     - :math:`\|\phi_h - \phi_{\rm ex}\|_{L^{2}}`
     - measured order
   * - 10
     - :math:`2.17\!\times\!10^{-3}`
     - ---
   * - 20
     - :math:`5.40\!\times\!10^{-4}`
     - 2.01
   * - 40
     - :math:`1.35\!\times\!10^{-4}`
     - 2.00
   * - 80
     - :math:`3.37\!\times\!10^{-5}`
     - 2.00
   * - 160
     - :math:`8.42\!\times\!10^{-6}`
     - 2.00

Successive ratios hit :math:`4.00\pm0.02`, i.e. the measured order
is exactly the design order of diamond difference.  The L1 test
:func:`tests.sn.test_mms.test_sn_1d_slab_mms_converges_second_order`
asserts a slightly loose ``order > 1.9`` bracket to leave room for
round-off at the finest mesh.

**Risk points / things that can go wrong.**

- *Vacuum BC not honoured.*  If the reflective-partner copy is not
  skipped, incoming-face angular flux at the boundary is non-zero
  (the reflected outgoing from the opposite sweep) and the
  manufactured solution no longer satisfies the discrete problem.
  Symptom: :math:`\mathcal O(1)` error at the coarsest mesh; no
  convergence regardless of refinement.
- *Wrong normalisation for* :math:`Q_{\rm ext}`.  The solver's
  :math:`Q_{\rm aniso}` slot is divided by :math:`W` internally;
  the ansatz has a :math:`1/W` prefactor; the two must cancel.
  If the derivation forgets the :math:`W` cancellation, the
  measured flux is a factor of :math:`W` off but still converges at
  order 2 --- sneaky.  Guard: the second test in ``test_mms.py``
  cross-checks the algebraic symmetry of :eq:`sn-mms-qext`.
- *Non-smooth ansatz.*  A discontinuous material or a piecewise
  linear ansatz degrades the observed order to :math:`\mathcal O(h)`.
  The homogeneous sinusoid avoids both.
- *1-group vs multigroup.*  Because the manufactured flux is isotropic
  and there is no fission in the fixed-source problem, 1 group is
  sufficient --- the degeneracy warning about 1-group eigenvalue
  tests does not apply, since no :math:`\keff` enters.  Multigroup
  and heterogeneous MMS extensions are tracked as follow-ups for
  richer operator coverage.

**Follow-ups.**  MMS for :doc:`method_of_characteristics`, diffusion,
and spherical / cylindrical curvilinear SN is tracked in GitHub
Issues (see ``type:feature level:L1``).  The curvilinear sweeps
need their own ansatz because their vacuum BC plumbing is not
yet wired up. **Heterogeneous and multigroup SN MMS is covered
by the next subsection.**


.. _sn-mms-heterogeneous-verification:

Heterogeneous MMS — 2-group continuous-:math:`\Sigma` slab
-----------------------------------------------------------

The homogeneous MMS case above verifies the Cartesian 1D SN
sweep for a *single-material* slab. To verify the multigroup
operator on a **heterogeneous** problem --- where each cell can
have different cross sections and the scatter matrix couples
groups across positions --- the Method of Manufactured Solutions
is extended in Phase 2.1a of the verification campaign with two
deliberate choices:

1. **Continuous (smooth)** :math:`\Sigma_{t,g}(x)` and
   :math:`\Sigma_{s,g\to g'}(x)` instead of piecewise-constant
   material regions. Discontinuous :math:`\Sigma` at interfaces
   that do not coincide with cell faces degrades diamond
   difference from :math:`\mathcal O(h^{2})` to
   :math:`\mathcal O(h)`, which would contaminate the
   spatial-convergence measurement with interface-treatment
   artefacts. With smooth :math:`\Sigma(x)` the diamond-
   difference operator hits its design :math:`\mathcal O(h^{2})`
   order exactly --- the convergence study becomes a clean test
   of the operator itself. This follows Salari & Knupp
   SAND2000-1444 §6, the canonical MMS reference for
   heterogeneous verification.
2. **Per-group amplitudes** :math:`\mathbf c = (c_1, c_2)` in
   the ansatz, so the scalar flux has a non-trivial group
   spectrum and the downscatter source term in the manufactured
   :math:`Q^{\text{ext}}` is non-zero. A bug that transposes
   the scatter matrix or drops a cross-group source term
   produces an incorrect :math:`\phi_2` that the convergence
   test catches immediately.

**Ansatz.**  The homogeneous ansatz carries over, now with a
per-group amplitude:

.. math::
   :label: sn-mms-hetero-psi

   \psi_{n,g}(x) \;=\; \frac{c_g}{W}\,A(x),
   \qquad A(x) \;=\; \sin\!\left(\frac{\pi x}{L}\right),

where :math:`W = \sum_n w_n` is the quadrature weight sum. The
scalar flux in each group is
:math:`\phi_g(x) = c_g\,A(x)`, so the amplitudes
:math:`\mathbf c` literally are the group fluxes at the slab
midpoint (where :math:`A` peaks). With
:math:`\mathbf c = (1.0, 0.3)` the two groups are linearly
independent and the downscatter coupling is visible.

Both groups share the same *spatial* mode :math:`\sin(\pi x/L)`
--- this is the fundamental mode of the bare slab and is exactly
the shape that emerges from separation of variables in the
diffusion limit. The heterogeneous SN problem would in general
have each group living in its own spatial harmonic, but we
*choose* the shared-mode ansatz as the manufactured target and
derive the non-trivial :math:`Q^{\text{ext}}` that makes it
satisfy the transport equation. The test then measures how well
the numerical SN sweep reproduces this prescribed shape.

**Manufactured source.**  Substituting :eq:`sn-mms-hetero-psi`
into the multigroup discrete-ordinates transport equation

.. math::

    \mu_n\,\frac{\partial\psi_{n,g}}{\partial x}
        + \Sigma_{t,g}(x)\,\psi_{n,g}
    \;=\; \frac{1}{W}\!\left(
        \sum_{g'}\Sigma_{s,g'\to g}(x)\,\phi_{g'}(x)
      + Q^{\text{ext}}_{n,g}(x)
    \right)

and solving algebraically for :math:`Q^{\text{ext}}`:

.. math::
   :label: sn-mms-hetero-qext

   Q^{\text{ext}}_{n,g}(x) \;=\;
       \mu_n\,c_g\,A'(x)
     + c_g\,\Sigma_{t,g}(x)\,A(x)
     \;-\; \sum_{g'}\Sigma_{s,g'\to g}(x)\,c_{g'}\,A(x).

The :math:`W` factor cancels between the ansatz's :math:`1/W`
prefactor and the solver's own :math:`1/W` convention on the
isotropic and anisotropic source slots, so :eq:`sn-mms-hetero-qext`
is the residual hand-delivered to the sweep without any
additional rescaling.

**Structure of the source.**  The streaming term
:math:`\mu_n\,c_g\,A'(x)` is odd in :math:`\mu` and carries the
only angular dependence, which is why SN MMS fundamentally
needs the per-ordinate ``Q_aniso`` plumbing path. The removal
term :math:`c_g\,\Sigma_{t,g}(x)\,A(x)` is diagonal in group
index. The **in-scatter** sum
:math:`\sum_{g'}\Sigma_{s,g'\to g}\,c_{g'}\,A(x)` is the only
term that couples groups, and for :math:`g=2` in the default
2-group setup it contributes
:math:`-\Sigma_{s,1\to 2}(x)\,c_1\,A(x)` --- the thermal source
depends on the fast amplitude through the downscatter cross
section, exactly as the multigroup scatter assembly in the
sweep does.

**Canonical cross sections.**  The reference uses smooth
profiles on :math:`[0, L]`:

.. math::

    \Sigma_{t,1}(x) &= 1.0 + 0.2\sin(\pi x/L), \\
    \Sigma_{t,2}(x) &= 2.0 + 0.3\cos(\pi x/L), \\
    \Sigma_{s,1\to 1}(x) &= 0.3 + 0.1\sin(\pi x/L), \\
    \Sigma_{s,1\to 2}(x) &= 0.2 + 0.05\sin(\pi x/L), \\
    \Sigma_{s,2\to 2}(x) &= 1.5 + 0.15\sin(\pi x/L), \\
    \Sigma_{s,2\to 1}(x) &= 0.

These give :math:`\Sigma_{a,1}(x) = 0.5 + 0.05\sin(\pi x/L) > 0`
trivially and
:math:`\Sigma_{a,2}(x) = 0.5 + 0.3\cos(\pi x/L) - 0.15\sin(\pi x/L)`,
bounded below by :math:`0.5 - \sqrt{0.3^{2} + 0.15^{2}} \approx
0.165 > 0`, so the cross sections are physical everywhere. The
scattering ratios :math:`c_g = \Sigma_{s,\text{tot},g}/\Sigma_{t,g}`
stay around :math:`0.5` for both groups, which means source
iteration converges geometrically at rate :math:`\sim 0.5^n`
per sweep.

**Per-cell material construction.**  The solver consumes the
continuous :math:`\Sigma(x)` by creating **one material per cell**
with cross sections evaluated at the cell centre
:math:`x_i = (x_{i-1/2} + x_{i+1/2})/2`. The midpoint rule for
the cell-average cross section is :math:`\mathcal O(h^{2})`-
accurate on smooth :math:`\Sigma`, matching the diamond-
difference design order and not degrading the measured
convergence rate. The number of materials scales with mesh
refinement, so each mesh in the convergence study builds a
fresh materials dictionary via
:meth:`orpheus.derivations.sn_mms.SNSlab2GHeterogeneousMMSCase.build_materials`.

**Measured convergence.**  With default parameters
(:math:`L = 5\,\text{cm}`, :math:`\mathbf c = (1.0, 0.3)`,
Gauss--Legendre :math:`S_{16}`):

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 20

   * - :math:`n_{\rm cells}`
     - :math:`\|\phi_1 - \phi_{1,\rm ex}\|_{L^{2}}`
     - :math:`\|\phi_2 - \phi_{2,\rm ex}\|_{L^{2}}`
     - measured order
   * - 20
     - :math:`3.71\!\times\!10^{-4}`
     - :math:`3.38\!\times\!10^{-4}`
     - ---
   * - 40
     - :math:`9.25\!\times\!10^{-5}`
     - :math:`8.45\!\times\!10^{-5}`
     - 2.00
   * - 80
     - :math:`2.31\!\times\!10^{-5}`
     - :math:`2.11\!\times\!10^{-5}`
     - 2.00
   * - 160
     - :math:`5.78\!\times\!10^{-6}`
     - :math:`5.28\!\times\!10^{-6}`
     - 2.00

Both groups hit the design order independently, confirming
that the multigroup scatter coupling is correctly exercised.
The L1 test
:func:`tests.sn.test_mms_heterogeneous.test_sn_heterogeneous_mms_converges_second_order`
asserts ``> 1.9`` to leave round-off headroom at the finest
mesh.

**What this replaces.** Before Phase 2.1a, the heterogeneous
SN verification was
:func:`orpheus.derivations.sn._derive_sn_heterogeneous`, which
computed the reference :math:`k_{\text{eff}}` by running the
SN solver itself at four mesh refinements and Richardson-
extrapolating the eigenvalue sequence. That is a **T3 circular
self-test** in the verification-campaign taxonomy: the solver
verifies against its own extrapolated output, so any consistent
bug in the SN sweep that affects all mesh refinements the same
way is invisible to the test. The heterogeneous MMS reference
above breaks the circularity: the reference comes from the
manufactured-solution algebra, not from the solver.

**Complementary eigenvalue verification.** The MMS test
verifies the **spatial operator** on a heterogeneous problem
but does not exercise the eigenvalue iteration. Phase 2.1b of
the verification campaign lands a Case singular-eigenfunction
eigenvalue reference that restores eigenvalue-heterogeneous
coverage for the SN solver (T2 semi-analytical, from the
first-order Boltzmann equation itself, no diffusion
approximation). Until Phase 2.1b lands, the eigenvalue
iteration is verified transitively through the homogeneous
:math:`k_\infty = \nu\Sigma_f/\Sigma_a` tests, which are
themselves Phase 1.1 continuous references.


Homogeneous Infinite Medium
----------------------------

For homogeneous geometry with reflective BCs, the flux is spatially flat
and :math:`\keff = \lambda_{\max}(A^{-1}F)`.  This is geometry-independent
--- Cartesian, spherical, and cylindrical must all give the same
:math:`\keff`.

.. list-table::
   :header-rows: 1
   :widths: 10 14 19 19 19 19

   * - Groups
     - :math:`\kinf`
     - Cartesian (GL S8)
     - Spherical (GL S8)
     - Cylindrical (Prod 4x8)
     - Cylindrical (LS S4)
   * - 1
     - 1.5000
     - exact
     - exact
     - exact
     - exact
   * - 2
     - 1.8750
     - exact
     - exact
     - exact
     - exact
   * - 4
     - 1.4878
     - exact
     - exact
     - exact
     - exact

All entries are exact to machine precision.  Spherical 2G/4G results
(previously showing ~1% error) are now exact thanks to the M-M angular
closure weights.

Heterogeneous Convergence
--------------------------

For a cylindrical fuel (r < 0.5) + moderator (r < 1.0) geometry with
Product(4x8) quadrature:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25

   * - Cells/region
     - :math:`\keff` (1G)
     - :math:`\Delta k` from previous
   * - 5
     - 0.9769
     -
   * - 10
     - 0.9842
     - +0.0073
   * - 20
     - 0.9874
     - +0.0032

:math:`\keff` converges monotonically toward the CP reference
(0.9955).  The ~1% residual gap is the white-BC (CP) vs reflective-BC
(SN) approximation difference, consistent with the slab geometry
findings.

For the 2G heterogeneous resolution test, Product(4x8) and Product(8x8)
agree to :math:`< 0.01\%` (keff = 0.7227 for both), confirming
angular convergence.

Why 1-Group Verification Is Degenerate
----------------------------------------

For 1 energy group, the eigenvalue is:

.. math::

   k = \frac{\nSigf{}}{\Sigma_a}

This is a scalar ratio independent of the spatial or angular flux
distribution.  Consequences:

- Angular weight errors scale all flux equally --- cancels in :math:`k`.
- Wrong scattering convention --- no inter-group coupling to distort.
- Wrong flux shape --- does not matter; :math:`k` is a material property.

Only multi-group problems have a flux-shape-dependent eigenvalue:
:math:`k = (\nSigf{} \cdot \phi) / (\Sigma_a \cdot \phi)` where the
dot product weights each group differently.  A wrong group ratio (from
angular errors, normalization errors, or convergence failures) directly
shifts :math:`\keff`.

**Rule:** Every transport solver must be verified on at least 2-group
problems.  1-group success gives false confidence.

Spatial and Angular Convergence
--------------------------------

The diamond-difference scheme converges at :math:`O(h^2)` with mesh
refinement.  Gauss--Legendre quadrature shows spectral convergence in
angle.  Both are verified in ``test_sn_1d.py``.

Property Tests
---------------

For all geometries:

- **Particle balance**: production / absorption :math:`= \keff`
- **Flux non-negativity**: :math:`\phi \geq 0` everywhere
- **Angular flux at** :math:`r = 0` **all positive** (curvilinear only)
- **Multi-group eigenvector not flat**: flux spectrum differs between
  fuel and moderator (catches 1G-degenerate bugs)

Run the full suite::

   pytest tests/sn/ -v -m "not slow"


.. _investigation-history:

Investigation History: Curvilinear Bug
=======================================

This section documents the full history of the cylindrical DD bug and its
resolution.  It is preserved to prevent future sessions from repeating
the same dead ends.

Symptoms
--------

1. Homogeneous eigenvalue problems: exact (1G/2G/4G).
2. Heterogeneous eigenvalue problems: divergent :math:`\keff` with mesh
   refinement (5 cells: 1.15, 10 cells: 0.90, 20 cells: 0.52).
3. Fixed-source flux range: ``[0.59, 5.09]`` (should be near-flat).
4. :math:`\keff` depended strongly on angular quadrature order
   (4x8 vs 8x8 gave a 67% gap on heterogeneous problems).

The Root Cause
--------------

**Two bugs**, both breaking per-ordinate flat-flux consistency:

**Bug 1: Wrong** :math:`\alpha` **recursion.**  The code used
:math:`\alpha = \text{cumsum}(+w\xi)` with the azimuthal cosine
:math:`\xi` (``mu_y``).  The correct recursion ([Bailey2009]_ Eq. 50)
is :math:`\alpha = \text{cumsum}(-w\eta)` with the radial cosine
:math:`\eta` (``mu_x``), and ordinates must be sorted by increasing
:math:`\eta` within each level.

**Bug 2: Missing** :math:`\Delta A/w` **geometry factor.**  The
redistribution term in the balance equation must include
:math:`\Delta A_i / w_m`.  Without this factor, the streaming and
redistribution do NOT cancel per-ordinate for a spatially flat flux,
creating artificial angular anisotropy that worsens with mesh refinement
near :math:`r = 0`.

Six Failed Approaches
----------------------

Before the correct fix was found, six approaches were tested.  All
failed because they addressed symptoms, not the root cause:

1. **Reverse sweep:** Reversed the azimuthal sweep direction.
   No effect --- the dome shape is symmetric.

2. **Step closure:** Replaced DD (:math:`\tau = 0.5`) with step
   (:math:`\tau = 1.0`) in angle.  Reduced the divergence slightly
   but did not eliminate it.

3. **Lewis & Miller starting direction:** Tracked :math:`\alpha\psi`
   instead of :math:`\psi` alone (Section 4.5.4 of [LewisMiller1984]_).
   Unnecessary when the :math:`\alpha` recursion is correct, since
   :math:`\alpha_{1/2} = 0`.

4. **Bidirectional sweep:** Swept azimuthal ordinates in both
   directions and averaged.  Masked the asymmetry but did not fix it.

5. **Scaled** :math:`\alpha`:  Empirically scaled the :math:`\alpha`
   coefficients.  Could not find a consistent scaling.

6. **Zero redistribution:** Set :math:`\alpha = 0` to isolate the
   spatial streaming.  Confirmed the bug was in the redistribution,
   not the spatial DD.

Why the Original Sign Convention Hypothesis Was Wrong
------------------------------------------------------

The initial hypothesis was that the minus sign before the redistribution
in :eq:`transport-cylindrical` required special treatment (tracking
:math:`\alpha\psi`, using :math:`|\alpha|`, etc.).  This was wrong.
The sign convention is absorbed into the :math:`\alpha` definition:
:math:`\text{cumsum}(-\eta w)` with a :math:`+` sign in the balance
equation gives the same physics as :math:`\text{cumsum}(+\xi w)` with
a :math:`-` sign, for symmetric quadratures.

The real issue was the missing :math:`\Delta A/w` geometry factor, which
has nothing to do with signs.

The Fix
-------

Applied the [Bailey2009]_ formulation:

1. Corrected :math:`\alpha` recursion:
   :math:`\alpha_{m+1/2} = \alpha_{m-1/2} - w_m \eta_m`
   with ordinates sorted by increasing :math:`\eta`.

2. Added :math:`\Delta A/w` geometry factor to both the DD sweep and
   the BiCGSTAB operator.

3. Added Morel--Montry angular closure weights (M-M) to eliminate the
   flux dip at :math:`r = 0`.

4. Applied the same :math:`\Delta A/w` fix to the spherical sweep
   (which had the same missing factor).  The fixed-source flux spike
   at :math:`r = 0` dropped from 5.1x to 1.1x.

5. Applied the fix to both the spherical and cylindrical **BiCGSTAB
   operators** (:func:`transport_operator_matvec_spherical`,
   :func:`transport_operator_matvec_cylindrical`).  Multi-group
   spherical BiCGSTAB had been unstable (keff → NaN for 2G+); the
   root cause was the same missing :math:`\Delta A/w` factor in the
   explicit FD operator.  After the fix, 2G and 4G spherical BiCGSTAB
   converge to :math:`< 10^{-6}` of the analytical eigenvalue.

Results After Fix
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 25

   * - Test
     - Before
     - After
   * - Homogeneous 1G/2G/4G
     - Exact
     - Exact
   * - Heterogeneous 1G, 5/10/20 cells
     - 1.15 / 0.90 / 0.52 (diverges)
     - 0.977 / 0.984 / 0.987 (converges)
   * - Heterogeneous 2G, 4x8 vs 8x8
     - 0.54 vs 0.91 (67% gap)
     - 0.723 vs 0.723 (<0.01%)
   * - Fixed-source flux range (40 cells)
     - [0.59, 5.09] (spike)
     - [0.51, 1.12] (bounded)
   * - Contamination :math:`\beta` (cylindrical)
     - ~2.0
     - ~10\ :sup:`-16` (machine zero)


Numerical Sensitivities
========================

:math:`\keff` Sensitivity Table (421-Group Heterogeneous PWR Slab)
-------------------------------------------------------------------

All cases: 10 cells, :math:`\delta = 0.2` cm, material layout
``[fuel x 5, clad x 1, cool x 4]``, P0 scattering, 421 energy groups.

.. list-table::
   :header-rows: 1
   :widths: 50 15 35

   * - Configuration
     - :math:`\keff`
     - Notes
   * - 1D GL S16, BiCGSTAB (FD operator)
     - 1.03882
     - True 1D, 16 ordinates
   * - 1D Lebedev 110, source iteration (DD sweep)
     - 1.04294
     - 1D mesh, 2D quadrature
   * - 2D (10x2) Lebedev 110, source iter (DD sweep)
     - 1.04294
     - Pseudo-2D, full volumes
   * - 2D (10x2) Lebedev 110, BiCGSTAB (FD)
     - 1.04007
     - Pseudo-2D, full volumes
   * - 2D (10x2) Lebedev 110, BiCGSTAB, half-volumes
     - 1.04192
     - MATLAB convention
   * - **MATLAB reference**
     - **1.04188**
     - 2D Lebedev, FD, half-volumes

Sources of Variation
---------------------

1. **Angular quadrature** (GL vs Lebedev): ~0.004 difference.
   GL S16 integrates 1D angular flux with 16 points on :math:`[-1,1]`.
   Lebedev 110 integrates over the unit sphere --- more angular
   resolution but different effective weights per :math:`\mu_x`
   direction.  On a coarse heterogeneous mesh, these give different
   eigenvalues.

2. **Spatial discretisation** (DD sweep vs FD gradient): ~0.003
   difference.  Source iteration uses the DD wavefront sweep
   (:math:`T^{-1}`).  BiCGSTAB uses the explicit FD transport operator
   (:math:`T`).  Both are :math:`O(h)` on this mesh but with different
   truncation error constants.

3. **Boundary volume weighting**: ~0.002 difference (full vs half).
   The MATLAB code halves boundary cell volumes.  With ``ny=2`` and
   materials uniform in *y*, only the *x*-direction halving (fuel edge,
   coolant edge) affects :math:`\keff`.  This is an artifact of the
   pseudo-2D implementation: a true 1D calculation has no *y*-volumes.

4. **Inner convergence**: source iteration with ``max_inner=200``,
   ``inner_tol=1e-8`` does not fully converge for 421 groups (spectral
   radius ~0.97).  BiCGSTAB fully converges the inner solve in ~100
   Krylov iterations.

Matching the MATLAB Result
---------------------------

The MATLAB code uses: 2D Lebedev 110 on a 10x2 mesh, explicit FD
operator with BiCGSTAB, boundary half-volumes, P0 scattering.

The BiCGSTAB path with half-volumes reproduces 1.04192 vs MATLAB's
1.04188 (:math:`4 \times 10^{-5}` agreement).  The residual difference
is from floating-point details in cross-section processing.

The cleanest reference is the **1D GL BiCGSTAB** result (1.03882): no
pseudo-2D artifacts, well-conditioned angular quadrature, fully
converged inner solve.


References
==========

.. [Bailey2009] T.S. Bailey, J.E. Morel, and J.H. Chang,
   "A piecewise-linear finite element discretization of the diffusion
   equation for arbitrary polyhedral grids,"
   *Nuclear Science and Engineering*, 162:3, 2009.
   (Eq. 50: :math:`\alpha` recursion; Eq. 53--54: WDD;
   Eq. 74: Morel--Montry weights.)

.. [MorelMontry1984] J.E. Morel and G.R. Montry,
   "Analysis and elimination of the discrete-ordinates flux dip,"
   *Transport Theory and Statistical Physics*, 13:5, 1984.

.. [LewisMiller1984] E.E. Lewis and W.F. Miller, Jr.,
   *Computational Methods of Neutron Transport*,
   John Wiley & Sons, 1984.

.. [CaseZweifel1967] K.M. Case and P.F. Zweifel,
   *Linear Transport Theory*,
   Addison-Wesley, 1967.

.. [Lebedev1999] V.I. Lebedev and D.N. Laikov,
   "A quadrature formula for the sphere of the 131st algebraic order
   of accuracy," *Doklady Mathematics*, 59(3):477--481, 1999.

.. [CarlsonLathrop1965] B.G. Carlson and K.D. Lathrop,
   "Transport theory -- the method of discrete ordinates,"
   in *Computing Methods in Reactor Physics*,
   Gordon and Breach, 1968.
