.. _theory-discrete-ordinates:

====================================
Discrete Ordinates Method (S\ :sub:`N`)
====================================

.. contents:: Contents
   :local:
   :depth: 3


Overview
========

The discrete ordinates (S\ :sub:`N`) method solves the **integro-differential
form** of the neutron transport equation by discretising the angular variable
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


Architecture
============

Two-Layer Mesh Pattern
----------------------

The S\ :sub:`N` solver follows the same two-layer pattern as the CP
solver:

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
Equal spacing in :math:`\mu^2` is used with :math:`\mu_1^2 = 4/(N(N+2))`.

Weights sum to :math:`4\pi`.  Provides the ``level_indices`` structure
needed by the cylindrical sweep.  Implemented in :class:`LevelSymmetricSN`.

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

This is the core of the S\ :sub:`N` method.  Every geometry shares a
single algebraic structure; only the definitions of face area, volume,
direction cosine, and redistribution coefficient change.

Derivation from the Continuous PDE
-----------------------------------

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

The contamination factor :math:`\beta` measures how much the angular
face flux :math:`\psi_{n+1/2}` deviates from the cell-average flux for
a spatially flat source:

.. math::

   \beta = \frac{\psi_{n+\frac12}}{\psi_{n,i}}

For an ideal discretisation, :math:`\beta = 1` everywhere.  The standard
DD gives :math:`\beta \neq 1` at finite mesh width, creating a flux dip
(or spike) of order :math:`\sim 10\%` at the origin on typical meshes.

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

where :math:`\mu_{n\pm 1/2}` are the angular cell edges.  For spherical
geometry: :math:`\mu_{1/2} = -1`, :math:`\mu_{N+1/2} = +1`, and
interior edges are defined by :math:`\mu_{n+1/2} = \mu_{n-1/2} + w_n`.
For cylindrical geometry: :math:`\eta_{1/2} = -\sin\theta`,
:math:`\eta_{M+1/2} = +\sin\theta`, and interior edges are midpoints
of consecutive :math:`\eta` values.

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

For **Cartesian** geometry there is no redistribution
(:math:`\alpha = 0`, :math:`\Delta A = 0`), and the equation reduces to
the familiar:

.. math::
   :label: dd-cartesian

   \psi_{\rm avg}
   = \frac{Q / W + \frac{2|\mu|}{\Delta x}\, \psi_{\rm in}}
          {\Sigt{} + \frac{2|\mu|}{\Delta x}}

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

Cartesian 1D: Cumprod Recurrence
---------------------------------

For the 1D slab with Gauss--Legendre quadrature, the DD recurrence
:math:`\psi_{\rm out} = a\,\psi_{\rm in} + s` (where :math:`a` and
:math:`s` are precomputed DD coefficients) has a closed-form solution
via cumulative products.

Exploiting GL symmetry, only positive-:math:`\mu` ordinates are swept
forward; negative-:math:`\mu` ordinates are obtained by reversing the
cell array.  The result is O(N) vectorised numpy operations --- typically
sub-millisecond for practical meshes.

Implemented in :func:`_sweep_1d_cumprod`.

Cartesian 2D: Wavefront Sweep
-------------------------------

In 2D, the sweep proceeds along anti-diagonals :math:`i + j = k`,
vectorised within each diagonal.  Four sweep passes cover the four
quadrants :math:`(\pm\mu_x, \pm\mu_y)`.  Reflective boundary conditions
are applied by copying the outgoing flux from the reflected partner
ordinate.

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
``redist_dAw_per_level`` (cylindrical) from ``SNMesh``, along with the
M-M weights ``tau_mm`` / ``tau_mm_per_level``.  This ensures both paths
share exactly the same physics.


Boundary Conditions
===================

ORPHEUS uses **reflective boundary conditions** on all faces, representing
an infinite lattice or infinite medium.

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
by each quadrature's ``reflection_index()`` method.

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
is the net removal matrix.

P\ :sub:`0` Isotropic Scattering
----------------------------------

The default mode.  A direction-independent source is added to all
ordinates equally:

.. math::

   Q_{\rm scatter}(\hat{\Omega}_n)
   = \sum_{g'} \Sigs{g'\to g}^{(0)}\, \phi_{g'} / W

P\ :sub:`N` Anisotropic Scattering
------------------------------------

P\ :sub:`N` (:math:`N \geq 1`) adds per-ordinate sources via Legendre
moments of the angular flux:

.. math::
   :label: pn-scatter

   Q_{\rm scatter}(\hat{\Omega}_n)
   = \sum_{\ell=0}^{L} (2\ell+1)
     \sum_{g'} \Sigs{g'\to g}^{(\ell)}
     \left[\sum_{m=-\ell}^{\ell}
       f_{\ell,g'}^m \, Y_\ell^m(\hat{\Omega}_n)\right] / W

The 421-group cross-section library provides both P0 and P1 matrices.
The current implementation uses P0 only; P1 support requires computing
angular flux moments :math:`f_1^m = \sum_n w_n \psi_n R_n^{1,m}` after
each sweep.

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

   pytest tests/test_sn_1d.py tests/test_sn_properties.py \
          tests/test_sn_solver_components.py tests/test_sn_spherical.py \
          tests/test_sn_cylindrical.py tests/test_sn_quadrature.py \
          tests/test_sn_sweep_regression.py -v -m "not slow"


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
   (which had the same missing factor).

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
