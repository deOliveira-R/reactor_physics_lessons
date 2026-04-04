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

This chapter derives the CP method for two geometries:

- **Slab** (1D Cartesian) — the E\ :sub:`3` exponential-integral kernel
- **Concentric cylinders** (1D radial) — the Ki\ :sub:`3`/Ki\ :sub:`4`
  Bickley–Naylor kernel

and describes the eigenvalue iteration that uses the CP matrices to
compute :math:`\keff`.


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

where :math:`S` is the cell surface area.  The geometry-dependent
values:

- **Slab**: :math:`S = 1` (unit transverse area)
- **Concentric cylinder**: :math:`S = 2\pi R_{\text{cell}}`

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

This is the CP matrix used in the eigenvalue iteration.  It is
implemented in both :func:`_compute_slab_cp_group` and
:func:`_compute_cp_group`.

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

This geometry is represented by :class:`SlabGeometry`.


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

      r_{ii} = \frac{1}{2} \Sigt{i} \left[
        2 t_i - \frac{2}{\Sigt{i}} \left(\frac{1}{2} - E_3(\tau_i)\right)
      \right]

The factor :math:`1/2` accounts for the direction averaging (neutrons
go left or right with equal probability).  The total reduced CP is:

.. math::

   r_{ij} = \frac{1}{2}(\delta_d + \delta_c)

and the within-cell CP is :math:`P_{ij}^{\text{cell}} = r_{ij} /
(\Sigt{i} \, t_i)`.

This is implemented in :func:`_compute_slab_cp_group`.


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
   ax.set_title('Wigner–Seitz Cell with Chord Integration')
   ax.grid(True, alpha=0.3)
   plt.tight_layout()

This geometry is represented by :class:`CPGeometry`.


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
     \left[\text{Ki}_4(g) - \text{Ki}_4(g + \tau_i)
           - \text{Ki}_4(g + \tau_j) + \text{Ki}_4(g + \tau_i + \tau_j)\right] dy

where :math:`g` is the optical gap between regions :math:`i` and
:math:`j` along the chord.  The factor 2 accounts for the left-half
source symmetry (by symmetry of the annular geometry, contributions from
the left and right halves of the chord are equal).

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

This is implemented in :func:`_compute_cp_group`.


Slab vs. Cylinder: A Comparison
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Slab (1D Cartesian)
     - Concentric Cylinder (1D Radial)
   * - Kernel function
     - :math:`E_3(x)` (exponential integral)
     - :math:`\text{Ki}_3(x)` (Bickley–Naylor)
   * - Antiderivative
     - analytical (:math:`E_3` directly)
     - :math:`\text{Ki}_4(x)` (tabulated)
   * - Angular integration
     - analytical (over :math:`\mu \in [0,1]`)
     - numerical (:math:`y`-quadrature over chords)
   * - Volume :math:`V_i`
     - thickness :math:`t_i` (per unit transverse area)
     - :math:`\pi(R_i^2 - R_{i-1}^2)` (per unit axial length)
   * - Surface area :math:`S`
     - 1 (unit transverse area)
     - :math:`2\pi R_{\text{cell}}`
   * - White BC correction
     - :math:`P_{\text{in},j} = \Sigt{j} t_j P_{j,\text{out}}`
     - :math:`P_{\text{in},j} = \Sigt{j} V_j P_{j,\text{out}} / S`
   * - Implementation
     - :func:`_compute_slab_cp_group`
     - :func:`_compute_cp_group`

Despite these differences, the **eigenvalue iteration is identical** for
both geometries once :math:`P_{ij}^{\infty}` is built.  This is why the
codebase uses a single :class:`CPSolver` class that takes
:math:`P^{\infty}` as input.


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

The CP implementation is verified against analytical eigenvalues
computed from the CP matrix itself (see :doc:`verification`):

.. list-table::
   :header-rows: 1

   * - Benchmark
     - Groups
     - Geometry
     - Error
   * - 1G 2-region slab
     - 1
     - :math:`t_F = 0.5` cm, :math:`t_M = 0.5` cm
     - :math:`< 6 \times 10^{-8}`
   * - 2G 2-region slab
     - 2
     - :math:`t_F = 0.5` cm, :math:`t_M = 0.5` cm
     - :math:`< 2 \times 10^{-7}`
   * - 1G 2-region cylinder
     - 1
     - :math:`R_F = 0.5` cm, :math:`R_C = 1.0` cm
     - :math:`< 5 \times 10^{-7}`
   * - 2G 2-region cylinder
     - 2
     - :math:`R_F = 0.5` cm, :math:`R_C = 1.0` cm
     - :math:`< 3 \times 10^{-6}`

Run the full verification suite::

   python 09.Collision.Probability/run_verification.py


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
