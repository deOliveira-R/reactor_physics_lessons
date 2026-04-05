.. _theory-collision-probability:

====================================
Collision Probability Method
====================================

.. contents:: Contents
   :local:
   :depth: 3

Overview
========

The collision probability (CP) method solves the integral form of the
neutron transport equation for a lattice cell.  Rather than tracking the
angular flux :math:`\psi(\mathbf{r}, \hat{\Omega}, E)` as in discrete
ordinates or method of characteristics, the CP method works directly with
the **scalar flux** :math:`\phi(\mathbf{r}, E)` by integrating out the
angular variable analytically.

The key quantity is the **collision probability** :math:`P_{ij}`: the
probability that a neutron born uniformly and isotropically in region
:math:`i` has its *first* collision in region :math:`j`
[Stamm1965]_.  Once the
:math:`P_{ij}` matrix is known, the transport problem reduces to a
matrix equation in the region-averaged scalar fluxes.

This chapter derives the CP method for three geometries:

- **Slab** (1D Cartesian) — the E\ :sub:`3` exponential-integral kernel
- **Concentric cylinders** (1D radial) — the Ki\ :sub:`3`/Ki\ :sub:`4`
  Bickley–Naylor kernel
- **Concentric spheres** (1D radial) — the exponential kernel with
  :math:`y`-weighted quadrature

and describes how all three share a single eigenvalue solver through the
:class:`CPMesh` augmented-geometry pattern.


Architecture: Base Geometry and Augmented Geometry
===================================================

The CP solver separates geometry description from solver logic through
two layers:

1. **Base geometry** — :class:`~geometry.mesh.Mesh1D` stores cell edges,
   material IDs, and the coordinate system.  It computes volumes and
   surfaces via coordinate-system-aware formulas.

2. **Augmented geometry** — :class:`CPMesh` wraps a ``Mesh1D`` and adds
   the CP-specific kernel, quadrature, and the
   :meth:`~CPMesh.compute_pinf_group` method.  The kernel is selected
   automatically from the mesh's coordinate system.

3. **Solver** — :func:`solve_cp` creates a ``CPMesh``, builds the
   :math:`P^{\infty}` matrices for all energy groups, and runs power
   iteration via :class:`CPSolver` (which satisfies the
   :class:`~numerics.eigenvalue.EigenvalueSolver` protocol).

.. code-block:: text

   Mesh1D (edges, mat_ids, coord)
       │
       ▼
   CPMesh (kernel + quadrature + compute_pinf_group)
       │
       ▼
   solve_cp() → CPResult

This separation means adding a new geometry to the CP method requires
only implementing its kernel in ``CPMesh`` — the eigenvalue solver,
post-processing, and plotting are geometry-agnostic.


The Integral Transport Equation
================================

Starting point: the steady-state, one-speed (or single energy group)
transport equation in integral form.  For a neutron born at
:math:`\mathbf{r}'` travelling in direction :math:`\hat{\Omega}` toward
:math:`\mathbf{r}`, the **uncollided flux** at :math:`\mathbf{r}` is

.. math::
   :label: first-flight-kernel

   \phi^{\text{unc}}(\mathbf{r})
   = \int_{V} \frac{e^{-\tau(\mathbf{r}', \mathbf{r})}}{4\pi |\mathbf{r} - \mathbf{r}'|^2}
     \, Q(\mathbf{r}') \, dV'

where :math:`Q(\mathbf{r}')` is the total source (fission + scattering +
external) and :math:`\tau(\mathbf{r}', \mathbf{r})` is the **optical
path** (number of mean free paths) between the two points:

.. math::
   :label: optical-path

   \tau(\mathbf{r}', \mathbf{r})
   = \int_0^{|\mathbf{r} - \mathbf{r}'|}
     \Sigt{}\bigl(\mathbf{r}' + s\,\hat{\Omega}\bigr) \, ds


Flat-Source Approximation
-------------------------

The CP method assumes the source is **spatially flat** within each
sub-region :math:`i`:

.. math::
   :label: flat-source

   Q(\mathbf{r}) = Q_i \quad \text{for } \mathbf{r} \in V_i

Under this approximation, the collision rate in region :math:`j` due to
sources in region :math:`i` can be written as

.. math::
   :label: collision-rate

   \Sigt{j} \, \phi_j \, V_j
   = \sum_i P_{ji} \, V_i \, Q_i

where :math:`P_{ji}` is the collision probability (probability that a
neutron born in :math:`i` first collides in :math:`j`), and
:math:`V_i` is the volume of region :math:`i`.

.. note::

   **Convention**: in this codebase, :math:`P_{ij}` is indexed as
   :math:`P[\text{birth}_i, \text{collision}_j]`.  The flux update
   uses :math:`P^T`: ``phi = P_inf.T @ source``.

   This is implemented in :class:`CPSolver.solve_fixed_source`.


Definition of Collision Probabilities
======================================

Within-Cell Probabilities
-------------------------

For a cell containing :math:`N` sub-regions, the **within-cell collision
probability** :math:`P_{ij}^{\text{cell}}` is defined as:

.. math::
   :label: p-cell-def

   P_{ij}^{\text{cell}}
   = \frac{\text{Prob(neutron born in } i \text{ first collides in } j
     \text{ without leaving the cell)}}{}

These satisfy the **complementarity relation**:

.. math::
   :label: complementarity

   \sum_{j=1}^{N} P_{ij}^{\text{cell}} + P_{i,\text{out}} = 1

where :math:`P_{i,\text{out}}` is the escape probability (probability
of reaching the cell boundary without collision).


Reciprocity
-----------

From detailed balance, the collision probabilities satisfy
**reciprocity**:

.. math::
   :label: reciprocity

   \Sigt{i} \, V_i \, P_{ij}^{\text{cell}}
   = \Sigt{j} \, V_j \, P_{ji}^{\text{cell}}

This is fundamental: the CP matrix need only be computed for
:math:`j \ge i`, and the lower triangle follows from reciprocity.


Escape and Re-entry (White Boundary Condition)
----------------------------------------------

For an infinite lattice, a neutron escaping one cell immediately enters
an identical neighbouring cell.  The **white boundary condition**
assumes the re-entering angular distribution is **isotropic** — i.e.,
the neutron forgets its direction upon re-entry.

Under this approximation, the probabilities from the cell surface to
region :math:`j` are:

.. math::
   :label: surface-to-region

   P_{\text{in},j} = \frac{\Sigt{j} \, V_j \, P_{j,\text{out}}}{S}

where :math:`S` is the cell surface area, computed by the base geometry:

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - Coordinate system
     - Surface area :math:`S`
   * - Cartesian (slab)
     - :math:`1` (per unit transverse area)
   * - Cylindrical
     - :math:`2\pi R_{\text{cell}}`
   * - Spherical
     - :math:`4\pi R_{\text{cell}}^2`

This is accessed uniformly via ``mesh.surfaces[-1]`` in the code,
making the white-BC closure geometry-agnostic.

The surface-to-surface probability is:

.. math::
   :label: surface-to-surface

   P_{\text{in,out}} = 1 - \sum_j P_{\text{in},j}


Infinite-Lattice CP Matrix
---------------------------

The infinite-lattice collision probability accounts for neutrons that
escape, re-enter, possibly escape again, and so on (geometric series):

.. math::
   :label: p-inf

   P_{ij}^{\infty}
   = P_{ij}^{\text{cell}}
     + \frac{P_{i,\text{out}} \, P_{\text{in},j}}
            {1 - P_{\text{in,out}}}

This formula is **identical for all three geometries** when expressed
in terms of :math:`V_i` and :math:`S`.  It is implemented in
:meth:`CPMesh._apply_white_bc`.

Derivation of the Geometric Series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`P^{\infty}` formula is a geometric series.  Consider a
neutron born in region :math:`i`:

- With probability :math:`P_{ij}^{\text{cell}}` it collides in
  :math:`j` on the first pass.
- With probability :math:`P_{i,\text{out}}` it escapes.
- After escaping, it re-enters with probabilities
  :math:`P_{\text{in},j}` (first collision in :math:`j`) or
  :math:`P_{\text{in,out}}` (traverses and escapes again).
- Each subsequent escape-and-re-entry multiplies by
  :math:`P_{\text{in,out}}`.

Summing the infinite series:

.. math::

   P_{ij}^{\infty} &= P_{ij}^{\text{cell}}
     + P_{i,\text{out}} \, P_{\text{in},j}
     + P_{i,\text{out}} \, P_{\text{in,out}} \, P_{\text{in},j}
     + P_{i,\text{out}} \, P_{\text{in,out}}^2 \, P_{\text{in},j}
     + \cdots \\
   &= P_{ij}^{\text{cell}}
     + P_{i,\text{out}} \, P_{\text{in},j}
       \sum_{k=0}^{\infty} P_{\text{in,out}}^k \\
   &= P_{ij}^{\text{cell}}
     + \frac{P_{i,\text{out}} \, P_{\text{in},j}}
            {1 - P_{\text{in,out}}}

The series converges because :math:`P_{\text{in,out}} < 1` (some
fraction of re-entering neutrons must eventually collide).

**Row sum property:**  :math:`\sum_j P_{ij}^{\infty} = 1` because
every neutron born in region :math:`i` must eventually collide
somewhere in the infinite lattice.  This can be verified algebraically
from the complementarity relation :eq:`complementarity`.

.. warning::

   The white boundary condition is an **approximation**.  It assumes
   the angular distribution at the cell boundary is isotropic.  For
   lattice cells with moderate optical thickness (:math:`\tau \sim
   0.5\text{--}1.0`), this introduces errors of order 1% compared to
   exact transport (e.g., discrete ordinates with reflective BCs).

   This was confirmed numerically: the 1D SN solver with reflective BCs
   gives :math:`\keff \approx 1.261` for the 1G slab benchmark, while
   the CP method with white BCs gives :math:`\keff \approx 1.272` —
   a ~1% discrepancy entirely due to the white-BC approximation.

   The error arises because the true angular distribution at a cell
   boundary is **anisotropic** — neutrons preferentially stream in
   the direction of the flux gradient.  The white BC smears this
   anisotropy into an isotropic re-entry, which overestimates the
   flux in optically thin regions and underestimates it in thick ones.


The Three CP Kernels
=====================

The CP method requires a geometry-specific **kernel function**
:math:`F(\tau)` that encodes the angular averaging appropriate to the
coordinate system.  The dimensionality of the geometry determines how
much angular integration remains:

.. list-table::
   :header-rows: 1
   :widths: 15 20 30 35

   * - Geometry
     - Dimension
     - Kernel :math:`F(\tau)`
     - Why
   * - Slab
     - 1D
     - :math:`E_3(\tau) = \int_0^1 \mu \, e^{-\tau/\mu} \, d\mu`
     - Polar-angle integration over half-space
   * - Cylinder
     - 2D
     - :math:`\text{Ki}_4(\tau) = \int_\tau^\infty \text{Ki}_3(t)\,dt`
     - Azimuthal-angle integration around axis
   * - Sphere
     - 3D
     - :math:`e^{-\tau}`
     - Full symmetry — no residual angular integration

In all three cases, the **reduced collision probability** is built from
a **second-difference** of the kernel:

.. math::
   :label: second-diff-general

   \Delta_2[F](\tau_i, \tau_j, g)
   = F(g) - F(g + \tau_i) - F(g + \tau_j) + F(g + \tau_i + \tau_j)

where :math:`g` is the optical gap between regions and :math:`\tau_i`,
:math:`\tau_j` are the optical thicknesses of the source and target
regions.

This common structure is why all three geometries are handled by a
single :class:`CPMesh` class.


Why the Kernel Differs by Geometry
-----------------------------------

The kernel function :math:`F(\tau)` arises from integrating the
point-to-point first-flight transmission :math:`e^{-\tau}` over
the angular degrees of freedom that are eliminated by the geometry's
symmetry.  The number of remaining angular integrations determines
which special function appears.

**Slab (1D):** A neutron born at position :math:`x'` in a slab travels
at angle :math:`\theta` to the slab normal.  The optical path length is
:math:`\tau / \mu` where :math:`\mu = \cos\theta`.  The angular flux
contribution per unit solid angle involves a factor :math:`\mu` (from
the projected area :math:`dA = \mu\,dA_\perp`).  Averaging the
transmission :math:`e^{-\tau/\mu}` over the forward half-space
:math:`\mu \in [0, 1]` with this flux-weighting factor:

.. math::
   :label: e3-from-integration

   \int_0^1 \mu \, e^{-\tau/\mu} \, d\mu = E_3(\tau)

This is **one** angular integration (over :math:`\mu`), leaving a
function of :math:`\tau` only.  The result is the third exponential
integral :math:`E_3`, computed analytically — no numerical quadrature
needed for the angular variable.

**Cylinder (2D):** For a cylinder, the neutron travels at polar angle
:math:`\theta` to the cylinder axis *and* passes through the annular
cross-section at impact parameter :math:`y`.  The chord length through
each annulus depends on :math:`y` (geometrically) and the optical path
depends on :math:`\theta` (as :math:`\tau / \sin\theta`).  Integrating
the transmission over the polar angle:

.. math::
   :label: ki3-from-integration

   \int_0^{\pi/2} e^{-\tau/\sin\theta} \sin\theta \, d\theta
   = \text{Ki}_3(\tau)

This eliminates one angular dimension, but the :math:`y`-integration
over chord heights **remains as a numerical quadrature**.  It is the
second angular dimension that cylindrical symmetry does NOT eliminate.
The antiderivative :math:`\text{Ki}_4(\tau) = \int_\tau^\infty
\text{Ki}_3(t)\,dt` appears when averaging over the source position
within a region (the double-integral that produces the second-difference
formula; see below).

**Sphere (3D):** Full spherical symmetry means that every direction
through a concentric shell system is equivalent up to the impact
parameter :math:`y`.  A chord at height :math:`y` has transmission
:math:`e^{-\tau(y)}` with no residual angular integration — the
solid-angle measure :math:`d\Omega = 2\pi y\,dy / R^2` is absorbed
into the :math:`y`-quadrature weight.  Hence the kernel is simply:

.. math::

   F(\tau) = e^{-\tau}

No special functions are needed.  This is the simplest kernel, despite
being the highest-dimensional geometry.

.. list-table:: Angular integrations by geometry
   :header-rows: 1
   :widths: 20 30 30

   * - Geometry
     - Integrations eliminated analytically
     - Remaining numerical quadrature
   * - Slab
     - :math:`\mu`-integration :math:`\to E_3`
     - None
   * - Cylinder
     - :math:`\theta`-integration :math:`\to \text{Ki}_3`
     - :math:`y`-quadrature over chord heights
   * - Sphere
     - All (full symmetry :math:`\to e^{-\tau}`)
     - :math:`y`-quadrature (weight includes :math:`y`)


The Second-Difference Formula: Derivation
-------------------------------------------

The four-term second-difference
:math:`F(g) - F(g+\tau_i) - F(g+\tau_j) + F(g+\tau_i+\tau_j)` arises
from averaging the first-flight transmission over **both** the source
position and the collision position within their respective regions.

**Setup.**  Consider two regions :math:`i` and :math:`j` separated by
an optical gap :math:`g`.  A neutron born at optical depth :math:`s`
within region :math:`i` (measured from the left edge,
:math:`0 \le s \le \tau_i`) must traverse:

- the remainder of region :math:`i`: distance :math:`\tau_i - s`
- the gap: distance :math:`g`
- region :math:`j` up to collision point :math:`t`
  (:math:`0 \le t \le \tau_j`): distance :math:`t`

Total optical path: :math:`d = (\tau_i - s) + g + t`.

**Step 1: Source-position average.**  Integrating the kernel over the
source position :math:`s` within region :math:`i`:

.. math::

   I(t) = \int_0^{\tau_i} F\bigl((\tau_i - s) + g + t\bigr) \, ds

Substituting :math:`u = (\tau_i - s) + g + t` (so :math:`du = -ds`):

.. math::

   I(t) = \int_{g+t}^{\tau_i+g+t} F(u) \, du
        = \hat{F}(g + t) - \hat{F}(g + t + \tau_i)

where :math:`\hat{F}(x) = \int_0^x F(u)\,du` is the antiderivative
of :math:`F`.

**Step 2: Collision-position average.**  Integrating over the collision
position :math:`t` within region :math:`j`:

.. math::

   \text{rcp}_{ij} = \int_0^{\tau_j} I(t) \, dt
   = \int_0^{\tau_j} \bigl[\hat{F}(g+t) - \hat{F}(g+t+\tau_i)\bigr] \, dt

Evaluating each integral:

.. math::

   \int_0^{\tau_j} \hat{F}(g+t) \, dt
   = \hat{\hat{F}}(g+\tau_j) - \hat{\hat{F}}(g)

where :math:`\hat{\hat{F}}` is the double antiderivative.  Combining:

.. math::
   :label: second-diff-derivation

   \text{rcp}_{ij} =
   \hat{\hat{F}}(g) - \hat{\hat{F}}(g + \tau_i)
   - \hat{\hat{F}}(g + \tau_j)
   + \hat{\hat{F}}(g + \tau_i + \tau_j)

**Step 3: Identify the kernel.**  For the slab, the point-to-point
transmission kernel is :math:`E_1(\tau)` (first exponential integral).
Its antiderivative is :math:`-E_2(\tau)`, and the double antiderivative
is :math:`E_3(\tau)`.  So :math:`\hat{\hat{F}} = E_3` and the
second-difference uses :math:`E_3` directly.

For the cylinder, the point-to-point kernel (after polar averaging) is
:math:`\text{Ki}_3(\tau)`.  Its antiderivative is
:math:`-\text{Ki}_4(\tau)`, giving the second-difference in
:math:`\text{Ki}_4`.

For the sphere, :math:`F = e^{-\tau}` is its own antiderivative (up to
sign), so the second-difference uses :math:`e^{-\tau}` directly.


Self-Collision: Why It Has a Different Form
--------------------------------------------

The self-collision term (:math:`i = j`) cannot use the gap formula
because source and collision are in the **same region** — there is no
gap, and the source/collision positions overlap.

For the slab, the self-collision reduced CP is:

.. math::
   :label: self-derivation

   r_{ii} = \int_0^{\tau_i} \int_0^{\tau_i}
     E_1(|s - t|) \, dt \, ds

where :math:`E_1(|s-t|)` is the first-flight kernel for optical
distance :math:`|s-t|` within the same region.  Evaluating this
double integral (splitting into :math:`s > t` and :math:`s < t`
halves) gives:

.. math::

   r_{ii} = \Sigt{i} t_i - \bigl(E_3(0) - E_3(\tau_i)\bigr)

The first term :math:`\Sigt{} t_i` represents the total interaction
rate if all neutrons collided within the region.  The subtracted term
:math:`E_3(0) - E_3(\tau_i)` is the fraction that escapes, where
:math:`E_3(0) = 1/2` is the probability of a neutron at the surface
escaping the half-space.

For cylindrical and spherical geometries, the self-collision integral
has the same structure but with :math:`\text{Ki}_4` or :math:`e^{-\tau}`
replacing :math:`E_3`, and with :math:`y`-quadrature.  The code
implements this in the ``self_same`` variable within the sweep loops
of :meth:`CPMesh._compute_slab_rcp` and
:meth:`CPMesh._compute_radial_rcp`.


Slab Geometry: The E\ :sub:`3` Kernel
=======================================

Geometry
--------

The 1D slab half-cell extends from the reflective centre (:math:`x = 0`)
to the cell edge (:math:`x = L`).  It is discretized into :math:`N`
sub-regions, each with constant cross sections.

.. plot::
   :caption: Slab half-cell geometry with fuel, cladding, and coolant regions.

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches

   regions = [
       ('Fuel', 0.9, '#e74c3c'),
       ('Clad', 0.2, '#95a5a6'),
       ('Cool', 0.7, '#3498db'),
   ]

   fig, ax = plt.subplots(figsize=(10, 2.5))
   x = 0
   for name, width, color in regions:
       rect = mpatches.FancyBboxPatch((x, 0), width, 1, boxstyle="square,pad=0",
                                       facecolor=color, edgecolor='black', alpha=0.7)
       ax.add_patch(rect)
       ax.text(x + width/2, 0.5, name, ha='center', va='center', fontsize=12, fontweight='bold')
       x += width

   ax.set_xlim(-0.1, 2.0)
   ax.set_ylim(-0.2, 1.3)
   ax.set_xlabel('x (cm)')
   ax.set_aspect('equal')
   ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Reflective BC')
   ax.axvline(1.8, color='blue', linestyle='--', linewidth=2, label='White BC')
   ax.legend(loc='upper right')
   ax.set_yticks([])
   ax.set_title('Slab Half-Cell Geometry')
   plt.tight_layout()

This geometry is built via :func:`~geometry.factories.pwr_slab_half_cell`
with ``coord = CoordSystem.CARTESIAN``.


The E\ :sub:`3` Function
-------------------------

For slab geometry, the angular integration of the first-flight kernel
:eq:`first-flight-kernel` over the half-space yields the **third
exponential integral**:

.. math::
   :label: e3-def

   E_3(x) = \int_0^1 \mu \, e^{-x/\mu} \, d\mu

where :math:`\mu = \cos\theta` is the direction cosine with respect to
the slab normal.  :math:`E_3(0) = 1/2` and :math:`E_3(x) \to 0`
exponentially as :math:`x \to \infty`.

The :math:`E_3` function is the slab analogue of the Ki\ :sub:`3`
Bickley–Naylor function used in cylindrical geometry.  It is computed
via :func:`scipy.special.expn`.


Second-Difference Formula
-------------------------

The collision probability for a neutron born in region :math:`i` to
first collide in region :math:`j` involves the **second difference** of
:math:`E_3` evaluated at the optical boundaries between the regions.

Let :math:`\tau_k = \Sigt{k} \, t_k` be the optical thickness of region
:math:`k`, and define the cumulative optical path from the cell centre:

.. math::

   x_0 = 0, \quad x_{k} = \sum_{m=0}^{k-1} \tau_m

The **reduced collision probability** (unnormalised) is built from two
path types:

1. **Direct path** (same direction along the slab): for :math:`j > i`,

   .. math::
      :label: dd-slab

      \delta_d = E_3(g) - E_3(g + \tau_i) - E_3(g + \tau_j) + E_3(g + \tau_i + \tau_j)

   where :math:`g = x_j - x_{i+1}` is the optical gap between regions.

2. **Reflected path** (through the reflective centre at :math:`x = 0`):

   .. math::
      :label: dc-slab

      \delta_c = E_3(g_c) - E_3(g_c + \tau_i) - E_3(g_c + \tau_j) + E_3(g_c + \tau_i + \tau_j)

   where :math:`g_c = x_i + x_j` is the optical path via the centre.

3. **Self-collision** (:math:`i = j`):

   .. math::
      :label: self-slab

      r_{ii} = \Sigt{i} t_i - \bigl(E_3(0) - E_3(\tau_i)\bigr)

The total reduced CP for :math:`i \ne j` is:

.. math::

   r_{ij} = \frac{1}{2}(\delta_d + \delta_c)

and the within-cell CP is :math:`P_{ij}^{\text{cell}} = r_{ij} /
(\Sigt{i} \, V_i)`, where :math:`V_i = t_i` for slab geometry.

This is implemented in :meth:`CPMesh._compute_slab_rcp`.


Concentric Cylindrical Geometry: The Ki\ :sub:`3`/Ki\ :sub:`4` Kernel
=======================================================================

Geometry
--------

The Wigner–Seitz cell replaces the square unit cell boundary by a circle
of equal area (:math:`R_{\text{cell}} = p / \sqrt{\pi}` where :math:`p`
is the lattice pitch).  The cell is divided into :math:`N` concentric
annular regions: fuel, cladding, and coolant.

.. plot::
   :caption: Wigner–Seitz cylindrical cell with concentric annular regions.

   import numpy as np
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(6, 6))

   regions = [
       (0.9, '#e74c3c', 'Fuel'),
       (1.1, '#95a5a6', 'Clad'),
       (2.03, '#3498db', 'Cool'),
   ]

   for r, color, label in reversed(regions):
       circle = plt.Circle((0, 0), r, color=color, alpha=0.5, label=label)
       ax.add_patch(circle)

   for r, _, _ in regions:
       circle = plt.Circle((0, 0), r, fill=False, edgecolor='black', linewidth=1)
       ax.add_patch(circle)

   # Chord at height y
   y_chord = 0.6
   x_max = np.sqrt(regions[-1][0]**2 - y_chord**2)
   ax.plot([-x_max, x_max], [y_chord, y_chord], 'k-', linewidth=2)
   ax.annotate('chord at height $y$', xy=(0, y_chord), xytext=(0.5, 1.5),
               fontsize=11, arrowprops=dict(arrowstyle='->', color='black'))
   ax.plot([0, 0], [0, y_chord], 'k--', alpha=0.5)
   ax.text(0.08, y_chord/2, '$y$', fontsize=12)

   ax.set_xlim(-2.5, 2.5)
   ax.set_ylim(-2.5, 2.5)
   ax.set_aspect('equal')
   ax.legend(loc='upper right', fontsize=11)
   ax.set_xlabel('x (cm)')
   ax.set_ylabel('y (cm)')
   ax.set_title('Wigner\u2013Seitz Cell with Chord Integration')
   ax.grid(True, alpha=0.3)
   plt.tight_layout()

This geometry is built via :func:`~geometry.factories.pwr_pin_equivalent`
with ``coord = CoordSystem.CYLINDRICAL``.


The Bickley–Naylor Functions
----------------------------

For cylindrical geometry, the angular integration of the first-flight
kernel involves the **Bickley–Naylor function** of order 3:

.. math::
   :label: ki3-def

   \text{Ki}_3(x) = \int_0^{\pi/2} e^{-x / \sin\theta} \sin\theta \, d\theta

and its antiderivative:

.. math::
   :label: ki4-def

   \text{Ki}_4(x) = \int_x^{\infty} \text{Ki}_3(t) \, dt

The Ki\ :sub:`3` function plays the same role for cylindrical geometry
that :math:`E_3` plays for slab geometry: it represents the probability
of a neutron travelling a certain optical distance in the medium
[Carlvik1966]_.

:math:`\text{Ki}_3(0) = 1` and :math:`\text{Ki}_3(x) \to 0`
exponentially.  The function is tabulated numerically
(:func:`_build_ki_tables`) because no closed-form expression exists.


Chord Integration
-----------------

Unlike the slab case (where the angular integration is analytical), the
cylindrical geometry requires **numerical integration over chord
heights** :math:`y`.

A chord at height :math:`y` above the cell axis intersects a subset of
the annular regions.  For each chord, the half-chord length through
region :math:`k` is:

.. math::
   :label: chord-length

   \ell_k(y) = \sqrt{R_k^2 - y^2} - \sqrt{R_{k-1}^2 - y^2}

where :math:`R_k` is the outer radius of region :math:`k` (with
:math:`R_0 = 0` for the innermost region).  The chord exists only for
:math:`y < R_k`.

The optical half-thickness along the chord is
:math:`\tau_k(y) = \Sigt{k} \, \ell_k(y)`.

This is computed by :func:`_chord_half_lengths`.


Second-Difference Formula (Cylindrical)
-----------------------------------------

The CP matrix is computed by integrating the Ki\ :sub:`4` second
differences over all chord heights:

.. math::
   :label: second-diff-cyl

   r_{ij} = 2 \int_0^{R_{\text{cell}}}
     \bigl[\text{Ki}_4(g) - \text{Ki}_4(g + \tau_i)
           - \text{Ki}_4(g + \tau_j) + \text{Ki}_4(g + \tau_i + \tau_j)\bigr] \, dy

where :math:`g` is the optical gap between regions :math:`i` and
:math:`j` along the chord.  The factor 2 accounts for the two halves
of the chord (by symmetry of the annular geometry, contributions from
the left and right halves are equal).

Two path types contribute (same as the slab):

1. **Same-side path**: source and target on the same side of the chord
   midpoint.
2. **Through-centre path**: the neutron crosses the chord midpoint
   (optical gap :math:`g_c = x_i + x_j` from the cumulative boundary
   positions).

The self-collision term is:

.. math::
   :label: self-cyl

   r_{ii} = 2\Sigt{i} \int_0^{R_{\text{cell}}}
     \left[2\ell_i(y) - \frac{2}{\Sigt{i}}
       \left(\text{Ki}_4(0) - \text{Ki}_4(\tau_i(y))\right)\right] dy

The :math:`y`-integration is performed with composite Gauss–Legendre
quadrature, with breakpoints at each annular boundary to capture the
chord-length discontinuities.

This is implemented in :meth:`CPMesh._compute_radial_rcp` with
``self._kernel = Ki₄``.


Concentric Spherical Geometry: The Exponential Kernel
======================================================

Geometry
--------

The spherical cell consists of :math:`N` concentric spherical shells.
As with the cylindrical Wigner–Seitz approximation, the outer boundary
is a sphere (the natural shape for an isolated fuel particle, pebble, or
TRISO kernel).

The cell is built via :func:`~geometry.factories.mesh1d_from_zones`
with ``coord = CoordSystem.SPHERICAL``.  Volumes are:

.. math::
   :label: sphere-volume

   V_i = \frac{4}{3}\pi\bigl(R_i^3 - R_{i-1}^3\bigr)

and the outer surface area is :math:`S = 4\pi R_{\text{cell}}^2`.


The Exponential Kernel
----------------------

For a sphere, **full 3-D symmetry** means that no residual angular
integration remains after the flat-source average.  The transmission
kernel along a chord at impact parameter :math:`y` is simply:

.. math::
   :label: sphere-kernel

   F(\tau) = e^{-\tau}

Compare with slab (:math:`E_3`) and cylinder (:math:`\text{Ki}_4`),
which encode 1-D and 2-D angular averages respectively.  The sphere
needs no special functions at all.

The values at zero optical thickness are:

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Kernel
     - :math:`F(0)`
   * - :math:`E_3(0)`
     - :math:`1/2`
   * - :math:`\text{Ki}_4(0)`
     - tabulated (~0.4244)
   * - :math:`e^{-0}`
     - 1


Chord Integration with :math:`y`-Weighting
--------------------------------------------

The chord geometry through concentric shells is **identical** to the
cylindrical case — the same formula :eq:`chord-length` gives the
half-chord length :math:`\ell_k(y)` through each shell.

The difference is in the **quadrature weight**.  For a cylinder (2-D),
each chord at height :math:`y` represents a line of sources, giving a
weight proportional to :math:`dy`.  For a sphere (3-D), each chord
represents a ring of sources with circumference :math:`2\pi y`, giving
a weight proportional to :math:`y \, dy`:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Cylinder
     - Sphere
   * - Area element
     - :math:`2\,dy` (line)
     - :math:`2\pi y \, dy` (ring)
   * - Quadrature weight
     - :math:`w_i`
     - :math:`w_i \cdot y_i`

This is why the spherical setup in ``CPMesh._setup_spherical()``
multiplies the Gauss–Legendre weights by the :math:`y`-coordinates.


Second-Difference Formula (Spherical)
--------------------------------------

The reduced collision probability has the same structure as cylindrical,
but with :math:`F = \exp(-\cdot)` and :math:`y`-weighted quadrature:

.. math::
   :label: second-diff-sph

   r_{ij} = 2 \int_0^{R_{\text{cell}}}
     \left[e^{-g} - e^{-(g + \tau_i)}
           - e^{-(g + \tau_j)} + e^{-(g + \tau_i + \tau_j)}\right] y \, dy

The self-collision term follows the same pattern:

.. math::
   :label: self-sph

   r_{ii} = 2\Sigt{i} \int_0^{R_{\text{cell}}}
     \left[2\ell_i(y) - \frac{2}{\Sigt{i}}
       \left(1 - e^{-\tau_i(y)}\right)\right] y \, dy

This is implemented by :meth:`CPMesh._compute_radial_rcp` — the
**same code path as cylindrical**, parameterised by the kernel function
and quadrature weights.


Geometry Comparison
====================

.. list-table::
   :header-rows: 1
   :widths: 18 27 27 28

   * - Aspect
     - Slab (Cartesian)
     - Cylinder (1D radial)
     - Sphere (1D radial)
   * - Kernel :math:`F(\tau)`
     - :math:`E_3(\tau)`
     - :math:`\text{Ki}_4(\tau)` (tabulated)
     - :math:`e^{-\tau}`
   * - :math:`F(0)`
     - :math:`1/2`
     - :math:`\approx 0.4244`
     - :math:`1`
   * - Angular integration
     - analytical (in :math:`E_3`)
     - numerical (:math:`y`-quadrature)
     - numerical (:math:`y`-quadrature)
   * - Quadrature weight
     - none (scalar)
     - :math:`w_i`
     - :math:`w_i \cdot y_i`
   * - Volume :math:`V_i`
     - :math:`t_i`
     - :math:`\pi(R_i^2 - R_{i-1}^2)`
     - :math:`\tfrac{4}{3}\pi(R_i^3 - R_{i-1}^3)`
   * - Surface :math:`S`
     - 1
     - :math:`2\pi R_{\text{cell}}`
     - :math:`4\pi R_{\text{cell}}^2`
   * - Prefactor
     - 1/2 (half-space)
     - 2 (chord halves)
     - 2 (chord halves)
   * - Code path
     - ``_compute_slab_rcp``
     - ``_compute_radial_rcp``
     - ``_compute_radial_rcp``

Despite these differences, the **eigenvalue iteration is identical** for
all geometries once :math:`P_{ij}^{\infty}` is built.  The white-BC
closure is also geometry-agnostic when expressed in terms of
``mesh.volumes`` and ``mesh.surfaces[-1]``.


The Eigenvalue Problem
=======================

Multi-Group Formulation
-----------------------

For :math:`G` energy groups, the neutron balance in region :math:`i`,
group :math:`g` is:

.. math::
   :label: neutron-balance

   \Sigt{i,g} \, V_i \, \phi_{i,g}
   = \sum_{j=1}^{N} P_{ji,g}^{\infty} \, V_j \left[
       \frac{\chi_{j,g}}{\keff} \sum_{g'=1}^{G} \nSigf{j,g'} \phi_{j,g'}
       + \sum_{g'=1}^{G} \Sigs{j,g' \to g} \phi_{j,g'}
     \right]

This is an eigenvalue problem in :math:`\keff`.


Power Iteration
---------------

The eigenvalue problem is solved by **power iteration** (see
:func:`~numerics.eigenvalue.power_iteration`):

1. **Fission source**: compute
   :math:`Q^F_{i,g} = \chi_{i,g} \sum_{g'} \nSigf{i,g'} \phi_{i,g'} / \keff`

2. **Fixed-source solve**: add scattering and (n,2n) sources, then
   apply the CP matrix:

   .. math::

      \phi_{i,g}^{\text{new}}
      = \frac{1}{\Sigt{i,g} \, V_i}
        \sum_j P_{ji,g}^{\infty} \, V_j \, Q_{j,g}^{\text{total}}

3. **Update** :math:`\keff`:

   .. math::

      \keff = \frac{\sum_{i,g} \nSigf{i,g} \phi_{i,g} V_i}
                   {\sum_{i,g} \Siga{i,g} \phi_{i,g} V_i}

   For lattice models with reflective (or white) boundary conditions,
   the leakage term is zero.

4. **Converge** when both :math:`\keff` and :math:`\phi` stop changing.

The power iteration converges to the **dominant eigenvalue**
(:math:`\keff`) and the **fundamental mode** — the unique non-negative
eigenvector (Perron–Frobenius theorem) [Hébert2009]_.

This is implemented in :class:`CPSolver`, which satisfies the
:class:`~numerics.eigenvalue.EigenvalueSolver` protocol.


Verification
============

The CP implementation is verified against semi-analytical eigenvalues
computed from the CP matrix itself.  For each geometry, the derivation
module (e.g., ``derivations/cp_sphere.py``) builds the :math:`P^{\infty}`
matrix independently, assembles the :math:`A` and :math:`B` matrices of
the generalised eigenvalue problem
:math:`A^{-1}B\,\mathbf{v} = k\,\mathbf{v}`, and solves for :math:`k`
via ``numpy.linalg.eigvals``.  The solver's power-iteration result must
match this eigenvalue.

27 verification cases are tested: {1, 2, 4} energy groups × {1, 2, 4}
spatial regions × {slab, cylinder, sphere}.

.. list-table::
   :header-rows: 1
   :widths: 25 10 30 15

   * - Geometry
     - Groups
     - Regions
     - Tolerance
   * - Slab (E\ :sub:`3`)
     - 1, 2, 4
     - 1, 2, 4
     - :math:`< 10^{-6}`
   * - Cylinder (Ki\ :sub:`4`)
     - 1, 2, 4
     - 1, 2, 4
     - :math:`< 10^{-5}`
   * - Sphere (exp)
     - 1, 2, 4
     - 1, 2, 4
     - :math:`< 10^{-5}`

Additionally, **algebraic property tests** are run for all three
coordinate systems (parametrised via ``pytest.mark.parametrize``):

- **Row sums** :math:`= 1` (neutron conservation)
- **Reciprocity**: :math:`\Sigt{i} V_i P_{ij} = \Sigt{j} V_j P_{ji}`
- **Non-negativity**: :math:`P_{ij} \ge 0`
- **Homogeneous limit**: 1-region :math:`P = 1`

Run the verification::

   pytest tests/test_cp_slab.py tests/test_cp_cylinder.py tests/test_cp_sphere.py tests/test_cp_properties.py -v


Implementation Details
=======================

Why P\ :sup:`T` Appears in the Flux Update
--------------------------------------------

The CP convention :math:`P_{ij} = P[\text{birth}_i, \text{collision}_j]`
means the *i*-th row of :math:`P` gives the probabilities of a neutron
born in region :math:`i` colliding in each region :math:`j`.  The
neutron balance :eq:`neutron-balance` sums over **source regions** to
compute the collision rate in the **target region**:

.. math::

   \Sigt{j} V_j \phi_j = \sum_i P_{ji}^{\infty} V_i Q_i

The sum is over the **first** index of :math:`P_{ji}` (birth in
:math:`j`, collision in :math:`i` — wait, that's reversed).  In fact,
:math:`P_{ji}` means "born in :math:`j`, collides in :math:`i`", so
:math:`\sum_j P_{ji} V_j Q_j` computes the collision rate in region
:math:`i` from sources in all regions :math:`j`.  In matrix form:

.. math::

   \Sigt{} V \phi = P^T V Q

Hence the flux update: :math:`\phi = (P^T V Q) / (\Sigt{} V)`.

In code: ``phi[:, g] = P_inf[:, :, g].T @ source`` — the transpose
converts from the birth-indexed convention to the collision-indexed sum.


The Analytical Verification Eigenvalue
---------------------------------------

The semi-analytical eigenvalue used for verification is computed by
assembling the CP neutron balance into a generalised eigenvalue problem.

The multi-group balance :eq:`neutron-balance` can be written as:

.. math::
   :label: cp-eigenvalue-system

   \mathbf{A} \boldsymbol{\Phi} = \frac{1}{k} \mathbf{B} \boldsymbol{\Phi}

where the unknown :math:`\boldsymbol{\Phi}` is the concatenation of all
region-group fluxes :math:`[\phi_{1,1}, \phi_{1,2}, \ldots, \phi_{N,G}]`,
and the matrices are:

.. math::

   A[i \cdot G + g,\; j \cdot G + g']
   &= \delta_{ij} \delta_{gg'} \Sigt{i,g} V_i
     - P_{ji,g}^{\infty} V_j \Sigs{j,g' \to g} \\
   B[i \cdot G + g,\; j \cdot G + g']
   &= P_{ji,g}^{\infty} V_j \chi_{j,g} \nSigf{j,g'}

The eigenvalue :math:`k_{\infty}` is the largest eigenvalue of
:math:`\mathbf{A}^{-1} \mathbf{B}`, computed via
``numpy.linalg.eigvals``.

This is implemented in ``_kinf_from_cp()`` in each derivation module
(``derivations/cp_slab.py``, ``derivations/cp_cylinder.py``,
``derivations/cp_sphere.py``).  The derivation modules compute
:math:`P^{\infty}` independently of the solver, using the same kernel
functions but in a self-contained implementation.  The solver's
power-iteration result must match this independent eigenvalue.


Ki\ :sub:`3`/Ki\ :sub:`4` Table Construction
----------------------------------------------

The Bickley–Naylor functions have no closed-form expression and must be
tabulated numerically.

**Ki₃ table:**  The function :func:`_build_ki_tables` evaluates
:math:`\text{Ki}_3(x)` at :math:`N = 20{,}000` equally spaced points
on :math:`[0, x_{\max}]` where :math:`x_{\max} = 50`.  Each point is
computed by numerical integration via ``scipy.integrate.quad``:

.. math::

   \text{Ki}_3(x_k) = \int_0^{\pi/2}
     e^{-x_k / \sin\theta} \sin\theta \, d\theta

The boundary value :math:`\text{Ki}_3(0) = 1` is set analytically.
For large :math:`x`, :math:`\text{Ki}_3(x) \to 0` exponentially,
so :math:`x_{\max} = 50` is sufficient (residual :math:`< 10^{-20}`).

**Ki₄ table:**  The antiderivative is computed by the cumulative
trapezoid rule, integrating from right to left:

.. math::

   \text{Ki}_4(x_k) = \int_{x_k}^{x_{\max}}
     \text{Ki}_3(t) \, dt
   \approx \Delta x \sum_{j=k}^{N-1} \text{Ki}_3(x_j)

This is implemented as ``np.cumsum(ki3[::-1])[::-1] * dx``.  The
endpoint is set to zero: :math:`\text{Ki}_4(x_{\max}) = 0`.

**Interpolation:**  At runtime, :func:`_ki4_lookup` evaluates
:math:`\text{Ki}_4` by linear interpolation (``np.interp``).  For
arguments beyond :math:`x_{\max}`, the function returns 0.

**Accuracy:**  With 20000 points on [0, 50], the spacing is
:math:`\Delta x = 0.0025`.  Linear interpolation gives
:math:`O(\Delta x^2)` error, which is :math:`\sim 6 \times 10^{-6}`.
This is below the tolerance of the CP eigenvalue tests (:math:`< 10^{-5}`
for cylindrical).  The cached :class:`BickleyTables` in the derivation
module uses the same construction.


Equal-Volume Mesh Subdivision
-------------------------------

The factory functions construct meshes where each sub-cell within a
material zone has **equal volume**.  This ensures uniform spatial
resolution in the volume-weighted sense.

**Cartesian:**  Equal-width cells: :math:`x_k = x_0 + k \Delta x`
where :math:`\Delta x = (x_N - x_0) / N`.

**Cylindrical:**  The annular volume is
:math:`V = \pi(R_k^2 - R_{k-1}^2)`.  Equal volumes require:

.. math::
   :label: cyl-subdivision

   R_k^2 - R_{k-1}^2 = \frac{R_N^2 - R_0^2}{N}

Solving: :math:`R_k = \sqrt{R_0^2 + \frac{k}{N}(R_N^2 - R_0^2)}`.
For :math:`R_0 = 0`: :math:`R_k = R_N \sqrt{k/N}`.

**Spherical:**  The shell volume is
:math:`V = \frac{4}{3}\pi(R_k^3 - R_{k-1}^3)`.  Equal volumes require:

.. math::
   :label: sph-subdivision

   R_k^3 - R_{k-1}^3 = \frac{R_N^3 - R_0^3}{N}

Solving: :math:`R_k = \bigl(R_0^3 + \frac{k}{N}(R_N^3 - R_0^3)\bigr)^{1/3}`.
For :math:`R_0 = 0`: :math:`R_k = R_N \,(k/N)^{1/3}`.

These formulas are implemented in :func:`~geometry.factories._subdivide_zone`,
which is the core of :func:`~geometry.factories.mesh1d_from_zones`.  The
equal-volume property is verified by ``test_equal_volume_single_zone``
and ``test_equal_volume_multi_zone`` in the test suite.


Numerical Evidence: White-BC Approximation Quality
-----------------------------------------------------

The following table compares the CP eigenvalue (white BC) with the SN
eigenvalue (reflective BC) for the same 1G heterogeneous geometries.
The difference quantifies the white-BC approximation error.

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 15

   * - Geometry
     - CP (white)
     - SN (reflective)
     - Difference
   * - Slab 2-region (fuel + mod)
     - 1.272
     - 1.261
     - +0.9%
   * - Cylinder 2-region (10 cells/zone)
     - 0.995
     - 0.997
     - −0.2%

The slab shows ~1% overestimation by CP because the white BC smears
the anisotropic angular flux at the fuel-moderator interface.  The
cylindrical discrepancy is smaller because the Wigner–Seitz cell has
a larger volume-to-surface ratio, making the boundary angular
distribution closer to isotropic.


Design Decisions
==================

Why a Single CPMesh Class
--------------------------

All three geometries share the same eigenvalue iteration — only the
CP matrix construction differs.  Rather than three separate solver
classes, the :class:`CPMesh` class encapsulates the geometry-specific
kernel and provides a single :meth:`~CPMesh.compute_pinf_group` method.
Adding a new geometry (e.g., 2D ray-tracing CP) requires only adding
a new setup method and kernel, not a new solver class.

Why the Per-Group Loop
-----------------------

The CP matrix :math:`P^{\infty}_{ij,g}` is computed independently for
each energy group :math:`g` because the optical thicknesses
:math:`\tau_{i,g} = \Sigt{i,g} \ell_i` depend on the group-dependent
total cross section.  The loop ``for g in range(ng)`` in
:func:`solve_cp` could in principle be vectorised over groups, but the
Ki₄ table lookup and second-difference computation involve
group-specific optical thicknesses that change the evaluation points.
Vectorisation would require restructuring the table lookup, which is
deferred as a performance improvement (see ``CP-20260404-008`` in
``09.Collision.Probability/IMPROVEMENTS.md``).

Why 1-Group Verification Is Degenerate
---------------------------------------

For a single energy group with no scattering coupling,
:math:`\keff = \nSigf{}/\Siga{}` is a material property independent
of the spatial flux distribution.  Any positive flux shape gives the
same :math:`k`.  This means:

- A wrong CP matrix still gives the correct :math:`k` (the numerator
  and denominator of the production/absorption ratio scale identically).
- Weight errors, normalisation bugs, and convention drifts are invisible.

Only multi-group problems have a flux-ratio-dependent eigenvalue.
The verification suite therefore tests 1, 2, AND 4 groups for every
geometry.

.. note::

   This is the same lesson as for the SN solver (see ``gotchas.md``
   and the DO theory chapter).  The 1-group degenerate case masked
   the z-ordinate weight loss, the scattering transpose bug, and the
   BiCGSTAB normalisation error — all caught only by multi-group tests.


References
==========

.. [Carlvik1966] I. Carlvik, "A method for calculating collision
   probabilities in general cylindrical geometry and applications to
   flux distributions and Dancoff factors," *Proc. Third United Nations
   Int. Conf. Peaceful Uses of Atomic Energy*, Vol. 2, 1966.

.. [Stamm1965] R. Stamm'ler and M.J. Abbate, *Methods of Steady-State
   Reactor Physics in Nuclear Design*, Academic Press, 1983.

.. [Hébert2009] A. Hébert, *Applied Reactor Physics*, Presses
   internationales Polytechnique, 2009.
