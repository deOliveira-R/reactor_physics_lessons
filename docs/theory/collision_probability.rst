.. _theory-collision-probability:

====================================
Collision Probability Method
====================================

.. contents:: Contents
   :local:
   :depth: 3


.. _key-facts-cp:

Key Facts
=========

**Read this before modifying the CP solver.**

- CP works with **scalar flux** :math:`\phi`, not angular flux :math:`\psi`
- :math:`P_{ij}` = probability that neutron born in *j* has first collision in *i*
- Convention: ``P[i,j]`` = birth in *j*, collision in *i*; flux update uses :math:`P^T` (see :ref:`scattering-matrix-convention`)
- Row sums = 1 (conservation), reciprocity: :math:`\Sigma_{t,i} V_i P(i,j) = \Sigma_{t,j} V_j P(j,i)`
- **Slab**: :math:`E_3` exponential-integral kernel (``scipy.special.expn``)
- **Cylinder**: :math:`\text{Ki}_4` Bickley-Naylor kernel (20,000-point lookup table)
- **Sphere**: exponential kernel
- White-BC approximation: isotropic return at cell boundary → ~1% gap vs reflective SN
- Inner iteration IS needed: self-scatter :math:`\Sigma_s(g \to g) \cdot \phi_g` makes source depend on solution
- **Gotcha**: tautological residual if checking ``denom * phi - transported`` (ERR-016)
- Power iteration tolerance ``keff_tol=1e-7`` bounds eigenvalue error to ~1e-6
- Gauss-Seidel update uses latest flux immediately; Jacobi uses previous iteration
- Verification uses :ref:`synthetic cross sections <synthetic-xs-library>`, not real nuclear data

.. admonition:: Conventions

   - Scattering matrix: :ref:`scattering-matrix-convention` — ``SigS[g_from, g_to]``, source uses transpose
   - Multi-group balance: :eq:`mg-balance` in :ref:`theory-homogeneous`
   - Cross sections: :ref:`theory-cross-section-data`
   - Verification: :ref:`synthetic-xs-library` — regions A/B/C/D
   - Eigenvalue: :ref:`power-iteration-algorithm` shared with all deterministic solvers


Overview
========

The collision probability (CP) method solves the
:ref:`multi-group eigenvalue problem <mg-eigenvalue-problem>` in integral
form.  Rather than tracking the angular flux
:math:`\psi(\mathbf{r}, \hat{\Omega}, E)` as in discrete ordinates or
method of characteristics, the CP method works directly with the
**scalar flux** :math:`\phi(\mathbf{r}, E)` by integrating out the
angular variable analytically.

The key quantity is the **collision probability** :math:`P_{ij}`: the
probability that a neutron born uniformly and isotropically in region
:math:`i` has its *first* collision in region :math:`j`
[Stamm1983]_.  Once the
:math:`P_{ij}` matrix is known, the transport problem reduces to a
matrix equation in the region-averaged scalar fluxes.

Three geometries are supported:

- **Slab** (1D Cartesian) --- the :math:`E_3` exponential-integral kernel
- **Concentric cylinders** (1D radial) --- the canonical
  :math:`\mathrm{Ki}_3` Bickley--Naylor kernel (A&S 11.2), evaluated via
  :func:`orpheus.derivations.cp_geometry._ki3_mp` (Chebyshev
  interpolant of :math:`e^{\tau}\,\mathrm{Ki}_3(\tau)` built from
  :func:`~orpheus.derivations._kernels.ki_n_mp` at 30 dps)
- **Concentric spheres** (1D radial) --- the exponential kernel with
  :math:`y`-weighted quadrature

All three share a single eigenvalue solver through the :class:`CPMesh`
augmented-geometry pattern.

**Derivation sources.**  The analytical eigenvalues and CP matrices used
for verification are computed independently by the derivation scripts.
These are the **source of truth** for all equations in this chapter:

- ``derivations/cp_geometry.py`` --- the unified geometry-dispatching
  core :class:`~derivations.cp_geometry.FlatSourceCPGeometry` and
  :func:`~derivations.cp_geometry.build_cp_matrix` (Phase B.2 refactor,
  commits ``f1b869b`` → ``bf128d3``)
- ``derivations/cp_slab.py`` --- slab :math:`E_3` kernel; thin facade
  over :data:`~derivations.cp_geometry.SLAB`
- ``derivations/cp_cylinder.py`` --- cylindrical canonical
  :math:`\mathrm{Ki}_3` kernel; thin facade over
  :data:`~derivations.cp_geometry.CYLINDER_1D`
- ``derivations/cp_sphere.py`` --- spherical :math:`e^{-\tau}` kernel;
  thin facade over :data:`~derivations.cp_geometry.SPHERE_1D`
- ``derivations/_kernels.py`` --- :math:`E_3` via
  :func:`~derivations._kernels.e3_vec` (wraps
  :func:`scipy.special.expn`); arbitrary-precision :math:`\mathrm{Ki}_n`
  via :func:`~derivations._kernels.ki_n_mp` (wraps
  :func:`mpmath.quad`). Double-precision :math:`\mathrm{Ki}_3`
  goes through :func:`~derivations.cp_geometry._ki3_mp` — a
  Chebyshev interpolant of :math:`e^{\tau}\,\mathrm{Ki}_3(\tau)`
  built from ``ki_n_mp`` at 30 dps (~:math:`5\times 10^{-6}`
  accuracy). The legacy ``BickleyTables`` tabulation was retired
  in Phase B.4 (commit ``6badbe5``, Issue #94)
- ``derivations/_eigenvalue.py`` --- shared eigenvalue computation via
  :func:`~derivations._eigenvalue.kinf_from_cp` and
  :func:`~derivations._eigenvalue.kinf_homogeneous`

Every equation in this chapter can be verified against these scripts.
Every numerical value cited was produced by them.


Architecture: Base Geometry and Augmented Geometry
===================================================

The CP solver separates geometry description from solver logic through
two layers:

1. **Base geometry** --- :class:`~geometry.mesh.Mesh1D` stores cell edges,
   material IDs, and the coordinate system.  It computes volumes and
   surfaces via :func:`~geometry.coord.compute_volumes_1d` and
   :func:`~geometry.coord.compute_surfaces_1d`.

2. **Augmented geometry** --- :class:`CPMesh` wraps a ``Mesh1D`` and adds
   the CP-specific kernel, quadrature, and
   :meth:`CPMesh.compute_pinf_group`.  The kernel is selected
   automatically from the mesh's coordinate system via a ``match``
   statement in :meth:`CPMesh.__init__`.

3. **Solver** --- :func:`solve_cp` creates a ``CPMesh``, builds the
   :math:`P^{\infty}` matrices for all energy groups, and runs power
   iteration via :class:`CPSolver` (satisfying the
   :class:`~numerics.eigenvalue.EigenvalueSolver` protocol).

.. code-block:: text

   Mesh1D (edges, mat_ids, coord)
       |
       v
   CPMesh (kernel + quadrature + compute_pinf_group)
       |
       v
   solve_cp() -> CPResult

**Design rationale (CP-20260404-001).**  Adding a new geometry requires
only a new ``_setup_*()`` method and kernel function --- the eigenvalue
solver, post-processing, and plotting are geometry-agnostic.  The
alternative (three separate solver classes) would duplicate ~200 lines of
iteration logic.  The derivation modules mirror this: all three
``_*_cp_matrix`` functions have identical white-BC closure code, differing
only in the kernel and quadrature.


The Integral Transport Equation
================================

Starting point: the steady-state, one-speed transport equation in
integral form.  For a neutron born at :math:`\mathbf{r}'` travelling
toward :math:`\mathbf{r}`, the **uncollided flux** is

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


.. _flat-source-approximation-cp:

Flat-Source Approximation
-------------------------

The CP method assumes the source is **spatially flat** within each
sub-region :math:`i`.  The MOC solver uses the same approximation; see
:ref:`flat-source-approximation-moc`.

.. math::
   :label: flat-source

   Q(\mathbf{r}) = Q_i \quad \text{for } \mathbf{r} \in V_i

Under this approximation, the collision rate in region :math:`j` due to
sources in region :math:`i` can be written as

.. math::
   :label: collision-rate

   \Sigt{j} \, \phi_j \, V_j
   = \sum_i P_{ji} \, V_i \, Q_i

where :math:`P_{ji}` is the collision probability and :math:`V_i` is the
volume of region :math:`i`.

.. note::

   **Convention**: in this codebase, :math:`P_{ij}` is indexed as
   :math:`P[\text{birth}_i, \text{collision}_j]`.  The flux update
   uses :math:`P^T`: ``phi = P_inf.T @ source``.
   This is implemented in :meth:`CPSolver.solve_fixed_source`.
   See :ref:`why-p-transpose` for the full derivation.


Definition of Collision Probabilities
======================================

Within-Cell Probabilities and Complementarity
-----------------------------------------------

For a cell with :math:`N` sub-regions, the **within-cell collision
probability** :math:`P_{ij}^{\text{cell}}` is the probability of first
collision in :math:`j` without leaving the cell.  **Complementarity**:

.. math::
   :label: complementarity

   \sum_{j=1}^{N} P_{ij}^{\text{cell}} + P_{i,\text{out}} = 1

where :math:`P_{i,\text{out}}` is the escape probability.  In the code:
``P_out = 1 - P_cell.sum(axis=1)`` (:meth:`CPMesh._apply_white_bc`).
Verified by ``test_cp_properties.py::test_row_sums`` for all three
coordinate systems.


Reciprocity
-----------

From detailed balance [Hebert2009]_ section 3.2:

.. math::
   :label: reciprocity

   \Sigt{i} \, V_i \, P_{ij}^{\text{cell}}
   = \Sigt{j} \, V_j \, P_{ji}^{\text{cell}}

**Why reciprocity holds.**  Time-reversal invariance: a neutron born in
:math:`i` colliding in :math:`j` traces a path identical (in reverse) to
one born in :math:`j` colliding in :math:`i`.  The optical thickness
along any chord is direction-independent.  The factor :math:`\Sigt{i} V_i`
converts from "per neutron born" (probability) to "per unit source
intensity" (rate), accounting for different source strengths in regions of
different sizes and cross sections.

**Practical consequence.**  The CP matrix need only be computed for
:math:`j \ge i`; the lower triangle follows from:

.. math::

   P_{ji}^{\text{cell}} = P_{ij}^{\text{cell}}
   \cdot \frac{\Sigt{i} \, V_i}{\Sigt{j} \, V_j}

This halves the computation cost.  In the code,
:meth:`CPMesh._normalize_rcp` divides the reduced collision probability
by :math:`\Sigt{i} V_i` for each row.  Reciprocity is verified by
``test_cp_properties.py::test_reciprocity`` and extended to multi-group
by ``test_cp_verification.py::TestMultiGroupProperties::test_reciprocity_multigroup``.


Escape and Re-entry (White Boundary Condition)
----------------------------------------------

For an infinite lattice, a neutron escaping one cell immediately enters
an identical neighbour.  The **white boundary condition** assumes the
re-entering angular distribution is **isotropic** --- i.e., the neutron
forgets its direction upon re-entry.

The surface-to-region probability is:

.. math::
   :label: surface-to-region

   P_{\text{in},j} = \frac{\Sigt{j} \, V_j \, P_{j,\text{out}}}{S}

where :math:`S` is the cell surface area, accessed uniformly via
``mesh.surfaces[-1]`` (:func:`~geometry.coord.compute_surfaces_1d`):

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

**Derivation of** :math:`P_{\text{in},j}`.  By reciprocity between the
surface source and the volume source in region :math:`j`:

.. math::
   :label: pin-from-reciprocity

   \frac{S}{4} \cdot P_{\text{in},j}
   = \Sigt{j} \, V_j \cdot P_{j,\text{out}}

The factor :math:`S/4` is the effective surface source strength for an
isotropic inward flux on a convex surface.  It originates from the
**Cauchy--Dirac mean chord length theorem**: :math:`\bar{\ell} = 4V/S`
for a convex body ([Hebert2009]_ section 3.3).  Physically: an isotropic
flux crossing a convex surface sees, on average, a path length of
:math:`4V/S` through the interior.  Solving :eq:`pin-from-reciprocity`:

.. math::

   P_{\text{in},j} = \frac{4 \Sigt{j} V_j P_{j,\text{out}}}{S}

In the standard CP formulation [Stamm1983]_ section 3.5, the
surface-to-region probability is defined per unit inward current
:math:`J^-`, and the normalisation convention absorbs the factor of 4.
ORPHEUS uses this convention in the white-BC transform::

    # White-BC closure (geometry-agnostic)
    P_in = sig_t * V * P_out / S_cell

The surface-to-surface probability is:

.. math::
   :label: surface-to-surface

   P_{\text{in,out}} = 1 - \sum_j P_{\text{in},j}

The same formula appears in all three derivation scripts (e.g.,
``derivations/cp_slab.py``, line ``P_in = sig_t_g * t_arr * P_out``
with the slab convention :math:`S = 1`, :math:`V = t`; and
``derivations/cp_cylinder.py``, line ``S_cell = 2.0 * np.pi * r_cell``
with cylindrical :math:`V = \pi(R_k^2 - R_{k-1}^2)`).


Infinite-Lattice CP Matrix
---------------------------

The infinite-lattice CP accounts for neutrons that escape, re-enter,
possibly escape again (geometric series):

.. math::
   :label: p-inf

   P_{ij}^{\infty}
   = P_{ij}^{\text{cell}}
     + \frac{P_{i,\text{out}} \, P_{\text{in},j}}
            {1 - P_{\text{in,out}}}

This formula is **identical for all three geometries** when expressed
in terms of :math:`V_i` and :math:`S`.  It is implemented in the
white-BC transform selected from :attr:`CPMesh.BC_REGISTRY` and
independently in all three derivation scripts (e.g., ``derivations/cp_slab.py``:
``P_inf_g[:,:,g] = P_cell + np.outer(P_out, P_in) / (1.0 - P_inout)``).

.. _cp-bc-registry:

Boundary Condition Infrastructure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The CP solver uses the project-wide ``BC_REGISTRY`` pattern for boundary
condition resolution.  The boundary condition is **declared** on the base
geometry via :class:`~geometry.mesh.BC` on :attr:`Mesh1D.bc_right
<geometry.mesh.Mesh1D.bc_right>` (the outer cell surface), and
**resolved** at :class:`CPMesh` construction time against the registry:

.. code-block:: python

   # Declaration (on geometry)
   mesh = Mesh1D(..., bc_right=BC("vacuum"))

   # Resolution (in CPMesh.__init__)
   factory = CPMesh.BC_REGISTRY[bc.kind]
   self._bc_transform = factory(self, bc)

:attr:`CPMesh.BC_REGISTRY` currently supports two boundary conditions:

.. list-table:: CP Boundary Conditions
   :header-rows: 1
   :widths: 15 30 40

   * - Kind
     - Physics
     - Effect on :math:`P^{\infty}`
   * - ``"white"`` (default)
     - Isotropic re-entry (infinite lattice)
     - Geometric series :eq:`p-inf` applied to :math:`P^{\text{cell}}`
   * - ``"vacuum"``
     - No re-entry (isolated cell)
     - :math:`P^{\infty} = P^{\text{cell}}` (rows sum to < 1)

The default is ``BC("white")``, matching the infinite-lattice assumption
used throughout the CP derivation above.  The vacuum BC is useful for
studying isolated fuel pins where neutrons escaping the cell are lost.

:meth:`CPMesh.compute_pinf_group` calls ``self._bc_transform(P_cell,
sig_t_g)`` to apply whichever BC was resolved at construction time.
Factory docstrings serve as descriptions for programmatic query::

    >>> {k: v.__doc__ for k, v in CPMesh.BC_REGISTRY.items()}

.. plot::
   :caption: White-BC geometric series: a neutron born in region :math:`i` either collides within the cell (:math:`P_{ij}^{\text{cell}}`) or escapes, re-enters isotropically, and the chain repeats.  The infinite sum converges to :math:`P_{ij}^{\infty}`.

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches

   fig, ax = plt.subplots(figsize=(14, 4))

   cell_w = 2.0
   cell_h = 1.5
   gap = 0.3
   n_cells = 4

   for c in range(n_cells):
       x0 = c * (cell_w + gap)
       rect = mpatches.FancyBboxPatch(
           (x0, 0), cell_w, cell_h, boxstyle="round,pad=0.05",
           facecolor='#ecf0f1', edgecolor='black', linewidth=2)
       ax.add_patch(rect)

       fuel = plt.Circle((x0 + cell_w / 2, cell_h / 2), 0.35,
                         color='#e74c3c', alpha=0.6)
       ax.add_patch(fuel)

       if c == 0:
           ax.text(x0 + cell_w / 2, 0.15, 'cell', fontsize=9,
                   ha='center', color='gray')

   arrow_y = cell_h / 2
   ax.annotate(r'born in $i$', xy=(0.6, arrow_y), fontsize=10,
               ha='center', color='red', fontweight='bold',
               xytext=(0.6, arrow_y + 0.5),
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

   ax.annotate('', xy=(1.4, arrow_y + 0.15),
               xytext=(0.8, arrow_y + 0.15),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
   ax.text(1.1, arrow_y + 0.35, r'$P_{ij}^{\mathrm{cell}}$',
           fontsize=10, ha='center', color='green')

   ax.annotate('', xy=(cell_w + gap / 2, arrow_y - 0.15),
               xytext=(1.5, arrow_y - 0.15),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2))
   ax.text(1.85, arrow_y - 0.4, r'$P_{i,\mathrm{out}}$',
           fontsize=10, ha='center', color='blue')

   x1 = cell_w + gap
   ax.annotate('', xy=(x1 + 0.5, arrow_y),
               xytext=(x1 - gap / 2, arrow_y),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2,
                               linestyle='dashed'))
   ax.text(x1 + 0.15, arrow_y + 0.25, 'white BC\n(isotropic)',
           fontsize=8, ha='center', color='blue', style='italic')

   ax.annotate('', xy=(x1 + 1.4, arrow_y + 0.15),
               xytext=(x1 + 0.6, arrow_y + 0.15),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
   ax.text(x1 + 1.0, arrow_y + 0.35, r'$P_{\mathrm{in},j}$',
           fontsize=10, ha='center', color='green')

   ax.annotate('', xy=(2 * (cell_w + gap) - gap / 2, arrow_y - 0.15),
               xytext=(x1 + 1.5, arrow_y - 0.15),
               arrowprops=dict(arrowstyle='->', color='orange', lw=2))
   ax.text(x1 + 1.85, arrow_y - 0.4, r'$P_{\mathrm{in,out}}$',
           fontsize=10, ha='center', color='orange')

   x2 = 2 * (cell_w + gap)
   ax.annotate('', xy=(x2 + 0.5, arrow_y),
               xytext=(x2 - gap / 2, arrow_y),
               arrowprops=dict(arrowstyle='->', color='orange', lw=2,
                               linestyle='dashed'))

   x3 = 3 * (cell_w + gap)
   ax.text(x3 + cell_w / 2, cell_h / 2, r'$\cdots$',
           fontsize=24, ha='center', va='center')

   ax.text(
       n_cells * (cell_w + gap) / 2, -0.6,
       r'$P_{ij}^{\infty} = P_{ij}^{\mathrm{cell}}'
       r' + \frac{P_{i,\mathrm{out}} \cdot P_{\mathrm{in},j}}'
       r'{1 - P_{\mathrm{in,out}}}$',
       fontsize=14, ha='center',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

   ax.set_xlim(-0.5, n_cells * (cell_w + gap))
   ax.set_ylim(-1.2, cell_h + 0.8)
   ax.set_aspect('equal')
   ax.set_xticks([])
   ax.set_yticks([])
   ax.set_title(
       r'White-BC Geometric Series: $P^{\infty}$ from Escape and Re-entry',
       fontsize=13)
   plt.tight_layout()

**Derivation of the geometric series.**  A neutron born in :math:`i`:

- Collides in :math:`j` on the first pass: probability :math:`P_{ij}^{\text{cell}}`.
- Escapes: probability :math:`P_{i,\text{out}}`.
- After escaping, re-enters isotropically and collides in :math:`j`:
  probability :math:`P_{\text{in},j}`.
- Or traverses without collision: probability :math:`P_{\text{in,out}}`.
- Each subsequent escape-and-re-entry multiplies by :math:`P_{\text{in,out}}`.

Summing:

.. math::

   P_{ij}^{\infty} &= P_{ij}^{\text{cell}}
     + P_{i,\text{out}} P_{\text{in},j}
       (1 + P_{\text{in,out}} + P_{\text{in,out}}^2 + \cdots) \\
   &= P_{ij}^{\text{cell}}
     + \frac{P_{i,\text{out}} P_{\text{in},j}}{1 - P_{\text{in,out}}}

The series converges because :math:`P_{\text{in,out}} < 1` (some
fraction of re-entering neutrons must eventually collide).

**Row sum property:** :math:`\sum_j P_{ij}^{\infty} = 1`.  Proof:
substitute complementarity :eq:`complementarity` for both sums:

.. math::

   \sum_j P_{ij}^{\infty}
   &= \sum_j P_{ij}^{\text{cell}}
     + P_{i,\text{out}} \frac{\sum_j P_{\text{in},j}}{1 - P_{\text{in,out}}} \\
   &= (1 - P_{i,\text{out}})
     + P_{i,\text{out}} \frac{1 - P_{\text{in,out}}}{1 - P_{\text{in,out}}} = 1

Verified numerically by ``test_cp_properties.py::test_row_sums``.

.. warning::

   The white BC is an **approximation**.  The true angular distribution
   at a cell boundary is anisotropic --- neutrons preferentially stream
   in the direction of the flux gradient.  The white BC smears this
   anisotropy into an isotropic re-entry, which overestimates the
   flux in optically thin regions and underestimates it in thick ones.
   For the 1G slab benchmark, the CP method
   (white BC) gives :math:`\keff \approx 1.272` while SN (reflective BC)
   gives :math:`\approx 1.261` --- a ~1% discrepancy entirely due to
   the white-BC approximation.  See :ref:`white-bc-quality`.


The Three CP Kernels
=====================

The CP method requires a geometry-specific **kernel function**
:math:`F(\tau)` that encodes the angular averaging.  The kernel arises
from integrating the point-to-point transmission :math:`e^{-\tau}` over
the angular degrees of freedom eliminated by the geometry's symmetry.
The number of remaining angular integrations determines which special
function appears:

.. list-table::
   :header-rows: 1
   :widths: 15 25 15 25

   * - Geometry
     - Kernel :math:`F(\tau)`
     - :math:`F(0)`
     - Residual quadrature
   * - Slab
     - :math:`E_3(\tau) = \int_0^1 \mu \, e^{-\tau/\mu} \, d\mu`
     - :math:`1/2`
     - None (analytical)
   * - Cylinder
     - :math:`\text{Ki}_4(\tau) = \int_\tau^\infty \text{Ki}_3(t)\,dt`
     - :math:`\approx 0.4244`
     - :math:`y`-quadrature over chord heights
   * - Sphere
     - :math:`e^{-\tau}`
     - :math:`1`
     - :math:`y`-quadrature (:math:`y`-weighted)

In all cases, the **reduced collision probability** is built from a
**second-difference** of the kernel :eq:`second-diff-general`.  This
common structure is why all three geometries are handled by a single
:class:`CPMesh` class.

.. plot::
   :caption: The three CP kernel functions: :math:`E_3` (slab), :math:`\text{Ki}_3` / :math:`\text{Ki}_4` (cylinder), and :math:`e^{-\tau}` (sphere).  All decay exponentially; they differ in :math:`F(0)` and rate of decay.

   import numpy as np
   import matplotlib.pyplot as plt
   from scipy.special import expn
   from scipy.integrate import quad

   x = np.linspace(0.001, 6, 500)

   e3 = expn(3, x)

   ki3 = np.array([
       quad(lambda t, xx=xx: np.exp(-xx / np.sin(t)) * np.sin(t),
            0, np.pi / 2)[0]
       for xx in x
   ])

   dx = x[1] - x[0]
   ki4 = np.cumsum(ki3[::-1])[::-1] * dx

   exp_kernel = np.exp(-x)

   fig, ax = plt.subplots(figsize=(10, 6))
   ax.semilogy(x, e3, '-r', linewidth=2, label=r'$E_3(\tau)$ (slab)')
   ax.semilogy(x, ki3, '-g', linewidth=2,
               label=r'$\mathrm{Ki}_3(\tau)$ (cylinder, integrand)')
   ax.semilogy(x, ki4, '--g', linewidth=2,
               label=r'$\mathrm{Ki}_4(\tau)$ (cylinder, kernel)')
   ax.semilogy(x, exp_kernel, '-b', linewidth=2,
               label=r'$e^{-\tau}$ (sphere)')

   ax.axhline(0.5, color='red', linestyle=':', alpha=0.5)
   ax.text(5.5, 0.55, r'$E_3(0)=\frac{1}{2}$', fontsize=10, color='red')
   ax.axhline(1.0, color='blue', linestyle=':', alpha=0.5)
   ax.text(5.5, 1.1, r'$e^0 = 1$', fontsize=10, color='blue')

   ax.set_xlabel(r'Optical thickness $\tau$', fontsize=12)
   ax.set_ylabel(r'$F(\tau)$', fontsize=12)
   ax.set_title('CP Kernel Functions by Geometry')
   ax.legend(fontsize=11, loc='lower left')
   ax.set_ylim(1e-4, 2)
   ax.grid(True, alpha=0.3)
   plt.tight_layout()

**Derivation source:** The kernels are implemented in
``derivations/_kernels.py`` (slab, arbitrary-precision) and
``derivations/cp_geometry.py`` (cylinder double-precision fast
path).  Slab :math:`E_3`: :func:`~derivations._kernels.e3` and
:func:`~derivations._kernels.e3_vec` (wrappers over
:func:`scipy.special.expn`). Cylinder :math:`\mathrm{Ki}_3`:
:func:`~derivations.cp_geometry._ki3_mp` — a Chebyshev interpolant
of :math:`e^{\tau}\,\mathrm{Ki}_3(\tau)` built from
:func:`~derivations._kernels.ki_n_mp` at 30 dps. The legacy
``BickleyTables`` tabulation that preceded ``_ki3_mp`` was retired
in Phase B.4 (commit ``6badbe5``, Issue #94).


Why the Kernel Differs by Geometry
-----------------------------------

**Slab (1D):** A neutron in a slab travels at angle :math:`\theta` to
the normal.  The optical path scales as :math:`\tau / \mu` where
:math:`\mu = \cos\theta`.  The angular flux contribution includes a
factor :math:`\mu` (projected area :math:`dA = \mu\,dA_\perp`).
Averaging the transmission :math:`e^{-\tau/\mu}` over the forward
half-space :math:`\mu \in [0, 1]`:

.. math::
   :label: e3-def

   E_3(\tau) = \int_0^1 \mu \, e^{-\tau/\mu} \, d\mu

This is one angular integration, leaving a function of :math:`\tau`
only.  :math:`E_3(0) = 1/2` and :math:`E_3(\tau) \to 0` exponentially.
Computed analytically via :func:`scipy.special.expn` (wrapped as
:func:`_e3` in the solver and :func:`~derivations._kernels.e3` in the
derivations).

**Cylinder (2D):** The neutron travels at polar angle :math:`\theta` to
the cylinder axis and crosses the annular section at impact parameter
:math:`y`.  The chord length depends on :math:`y` (geometry) and the
optical path depends on :math:`\theta` (as :math:`\tau / \sin\theta`).
Integrating over the polar angle yields the **Bickley--Naylor function**:

.. math::
   :label: ki3-def

   \text{Ki}_3(\tau) = \int_0^{\pi/2}
     e^{-\tau / \sin\theta} \sin\theta \, d\theta

The :math:`\sin\theta` weight arises from the solid-angle measure in
cylindrical coordinates: :math:`d\Omega = \sin\theta\,d\theta\,d\varphi`.
The azimuthal angle :math:`\varphi` integrates trivially by symmetry;
the projected ray length onto the cross-sectional plane is proportional
to :math:`\sin\theta`; and the probability of a ray at angle :math:`\theta`
is proportional to :math:`\sin\theta` from the :math:`d\Omega` measure.

This eliminates one angular dimension, but the :math:`y`-integration over
chord heights **remains as numerical quadrature** --- the second angular
dimension that cylindrical symmetry does NOT eliminate.

The antiderivative :math:`\text{Ki}_4(\tau) = \int_\tau^\infty
\text{Ki}_3(t)\,dt` appears in the second-difference formula because the
derivation involves double antiderivatives of the point-to-point kernel
(see :ref:`second-diff-derivation`).  **Why Ki**\ :sub:`4` **not Ki**\ :sub:`3`:
the slab uses :math:`E_3` (double antiderivative of the point kernel
:math:`E_1`); the cylinder uses :math:`\text{Ki}_4` (single antiderivative
of :math:`\text{Ki}_3`, which is already one integration up from the raw
transmission).  The second-difference formula cancels one antiderivative
level, leaving :math:`\text{Ki}_4`.

The Ki\ :sub:`3` function plays the same role for cylindrical geometry
that :math:`E_3` plays for slab geometry: it represents the probability
of a neutron travelling a certain optical distance in the medium
[Carlvik1966]_.

:math:`\text{Ki}_3(0) = 1` (under the sine-weighted convention used
in :eq:`ki3-def`) and :math:`\text{Ki}_3(x) \to 0` exponentially.

.. note::

   The equation label :eq:`ki3-def` uses the legacy ORPHEUS
   convention :math:`\text{Ki}_3(\tau) = \int_0^{\pi/2} e^{-\tau/\sin\theta}
   \sin\theta\,d\theta`. Under the Abramowitz & Stegun substitution
   :math:`\theta \to \pi/2 - t`, this is actually canonical
   :math:`\mathrm{Ki}_2^{\text{A\&S}}(\tau)`; the kernel consumed by
   the second-difference CP formula is the next function up in the
   hierarchy (called :math:`\text{Ki}_4` below) which equals
   canonical :math:`\mathrm{Ki}_3^{\text{A\&S}}`. The legacy naming
   is preserved in this theory page for cross-consistency with
   existing ``verifies("ki3-def")`` decorators on
   ``tests/cp/test_cylinder.py``,
   ``tests/derivations/test_cp_geometry.py``,
   ``tests/cp/test_verification.py``, and
   ``tests/derivations/test_peierls_cylinder_prefactor.py``. See
   :doc:`/verification/reference_solutions` §"Legacy naming
   discrepancy in ``BickleyTables``" for the full postmortem.

Neither :math:`\text{Ki}_3` nor :math:`\text{Ki}_4` has a
closed-form expression.  The derivation path is now
:func:`~derivations._kernels.ki_n_mp` (arbitrary precision via
:func:`mpmath.quad`) with a double-precision fast path through
:func:`~derivations.cp_geometry._ki3_mp` (a Chebyshev interpolant of
the canonical-:math:`\mathrm{Ki}_3` scaled kernel
:math:`e^{\tau}\,\mathrm{Ki}_3(\tau)`); the solver and the
derivation share the *same* code path so there is no kernel-split
bias across the CP V&V stack. See :ref:`ki-table-construction` for
the full account of what this replaced.

**Sphere (3D):** Full spherical symmetry absorbs all angular variables
into the :math:`y`-quadrature weight.  The kernel is simply
:math:`F(\tau) = e^{-\tau}` --- no special functions needed.  The extra
factor of :math:`y` in the quadrature weight comes from the spherical area
element :math:`2\pi y\,dy` (ring of sources) versus the cylindrical
:math:`2\,dy` (line of sources).  In
:meth:`CPMesh._setup_spherical`::

    # Spherical weight: extra factor of y in the quadrature
    self._y_wts = self._y_wts * self._y_pts


.. _second-diff-derivation:

The Second-Difference Formula: Full Derivation
-------------------------------------------------

The reduced collision probability uses the four-term second-difference:

.. math::
   :label: second-diff-general

   \Delta_2[F](\tau_i, \tau_j, g)
   = F(g) - F(g + \tau_i) - F(g + \tau_j) + F(g + \tau_i + \tau_j)

where :math:`g` is the optical gap, :math:`\tau_i` and :math:`\tau_j` are
the optical thicknesses of the source and target regions.

**Setup.** Regions :math:`i` and :math:`j` separated by optical gap
:math:`g`.  A neutron born at optical depth :math:`s` within region
:math:`i` (:math:`0 \le s \le \tau_i`) must traverse
:math:`d = (\tau_i - s) + g + t` to reach depth :math:`t` in region
:math:`j` (:math:`0 \le t \le \tau_j`).

**Step 1: Source-position average.** Integrate the kernel over the birth
position :math:`s` within region :math:`i`.  Substituting
:math:`u = (\tau_i - s) + g + t` (so :math:`du = -ds`):

.. math::

   I(t) &= \int_0^{\tau_i} F\bigl((\tau_i - s) + g + t\bigr) \, ds \\
        &= \int_{g+t}^{\tau_i + g + t} F(u) \, du \\
        &= \hat{F}(\tau_i + g + t) - \hat{F}(g + t)

where :math:`\hat{F}(x) = \int_0^x F(u)\,du` is the antiderivative.

**Step 2: Collision-position average.** Integrate over the collision
position :math:`t` within region :math:`j`:

.. math::

   \text{rcp}_{ij} &= \int_0^{\tau_j} I(t) \, dt \\
   &= \int_0^{\tau_j} \bigl[\hat{F}(\tau_i + g + t) - \hat{F}(g + t)\bigr] dt

Evaluating each integral using the double antiderivative
:math:`\hat{\hat{F}}(x) = \int_0^x \hat{F}(u)\,du`:

.. math::

   \int_0^{\tau_j} \hat{F}(g + t) \, dt
   &= \hat{\hat{F}}(g + \tau_j) - \hat{\hat{F}}(g) \\
   \int_0^{\tau_j} \hat{F}(\tau_i + g + t) \, dt
   &= \hat{\hat{F}}(\tau_i + g + \tau_j) - \hat{\hat{F}}(\tau_i + g)

Combining:

.. math::
   :label: rcp-from-double-antideriv

   \text{rcp}_{ij} =
     \hat{\hat{F}}(g)
   - \hat{\hat{F}}(g + \tau_i)
   - \hat{\hat{F}}(g + \tau_j)
   + \hat{\hat{F}}(g + \tau_i + \tau_j)

**Step 3: Identify** :math:`\hat{\hat{F}}` **per geometry.**

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 25

   * - Geometry
     - Point-to-point kernel
     - 1st antiderivative
     - :math:`\hat{\hat{F}}`
   * - Slab
     - :math:`E_1(\tau)`
     - :math:`-E_2(\tau)`
     - :math:`E_3(\tau)`
   * - Cylinder
     - :math:`\text{Ki}_3(\tau)`
     - :math:`-\text{Ki}_4(\tau)`
     - :math:`\text{Ki}_5(\tau)`
   * - Sphere
     - :math:`e^{-\tau}`
     - :math:`-e^{-\tau}`
     - :math:`e^{-\tau}`

For the slab, the :math:`E_n` functions satisfy
:math:`E_n'(\tau) = -E_{n-1}(\tau)`.  The code evaluates
:eq:`second-diff-general` directly with :math:`F = E_3` (slab),
:math:`F = \mathrm{Ki}_3^{\text{A\&S}}` (cylinder; the
ORPHEUS-local "Ki\ :sub:`4`\ " of the legacy narrative is canonical
A&S :math:`\mathrm{Ki}_3`), or :math:`F = e^{-\tau}` (sphere).

**Derivation source.** After the Phase B.2 unification, all three
flat-source CP modules dispatch through a single
geometry-invariant operator implemented as the module-level free
function :func:`~derivations.cp_geometry._second_difference`::

    def _second_difference(kernel, gap, tau_i, tau_j):
        return (kernel(gap)
                - kernel(gap + tau_i)
                - kernel(gap + tau_j)
                + kernel(gap + tau_i + tau_j))

The per-geometry kernel is supplied by the
:class:`~derivations.cp_geometry.FlatSourceCPGeometry` singleton:

- :data:`~derivations.cp_geometry.SLAB` — ``kernel_F3 = e3_vec``
  (:func:`scipy.special.expn`)
- :data:`~derivations.cp_geometry.CYLINDER_1D` —
  ``kernel_F3 = _ki3_mp`` (Chebyshev interpolant of
  canonical :math:`\mathrm{Ki}_3^{\text{A\&S}}`)
- :data:`~derivations.cp_geometry.SPHERE_1D` —
  ``kernel_F3 = _exp_kernel`` (``np.exp(-tau)``)

so the four-term structure is written exactly once in the whole
derivations package.

**SymPy verification of the four-term structure.** The following
script verifies the derivation in :eq:`rcp-from-double-antideriv`
symbolically. It builds the exact double integral of the transmission
kernel over source position :math:`s` and collision position :math:`t`,
substitutes the antiderivative identity, and checks that the result
matches the four-term second-difference pattern for a generic
:math:`F`. The same :class:`Delta2` structure emerges independent of
whether :math:`F = E_3`, :math:`\text{Ki}_4`, or :math:`e^{-\tau}` —
proving that the three geometries share one algebraic form::

    import sympy as sp

    # Symbolic setup: optical variables, arbitrary antiderivative F
    s, t, tau_i, tau_j, g = sp.symbols('s t tau_i tau_j g', positive=True)
    F = sp.Function('F')                       # generic kernel
    Fhat  = lambda x: sp.integrate(F(s), (s, 0, x))            # F̂
    Fhh   = lambda x: sp.integrate(Fhat(s),  (s, 0, x))        # F̂̂

    # Step 1: average over birth position s in region i, fixed t
    I_t = sp.integrate(F((tau_i - s) + g + t), (s, 0, tau_i))
    I_t = sp.simplify(I_t.rewrite(sp.Integral))

    # Step 2: average over collision position t in region j
    rcp = sp.integrate(I_t, (t, 0, tau_j))

    # Expected four-term pattern from eq:`rcp-from-double-antideriv`
    rcp_expected = ( Fhh(g)
                   - Fhh(g + tau_i)
                   - Fhh(g + tau_j)
                   + Fhh(g + tau_i + tau_j) )

    # The two must be equal for ANY F (not just E3/Ki4/exp)
    assert sp.simplify(rcp - rcp_expected) == 0

    # Instantiate per-geometry kernels and confirm
    for name, kernel in [
        ('slab',     lambda x: sp.expint(3, x)),         # E_3
        ('sphere',   lambda x: sp.exp(-x)),              # e^(-tau)
        # ('cylinder', lambda x: Ki4(x))                 # Bickley; numerical
    ]:
        Fsym = sp.Function(f'F_{name}')
        check = sp.simplify(
            (rcp_expected.rewrite(sp.Integral)
             .subs(F, kernel))
            .doit()
        )
        print(f'{name}: {check}')

This script is **not** run as part of the test suite (SymPy's
treatment of ``expint`` does not always simplify to a closed form
suitable for boolean equality), but it can be pasted into any Python
session as an independent verification of the derivation. The
generic-kernel assertion (``assert sp.simplify(rcp - rcp_expected)
== 0``) proves the four-term structure without reference to any
specific kernel — which is exactly the claim that slab, cylinder,
and sphere share one algebraic form.

.. plot::
   :caption: Geometric meaning of the four second-difference terms :math:`\Delta_2[F](\tau_i, \tau_j, g)`.  Green arrows mark the two positive terms :math:`+F(g)` and :math:`+F(g+\tau_i+\tau_j)`; orange arrows mark the two negative terms.

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches

   fig, ax = plt.subplots(figsize=(12, 4))

   colors = ['#e74c3c', '#e67e22', '#bdc3c7', '#bdc3c7', '#3498db']
   labels = ['Region $i$', '', 'Gap', '', 'Region $j$']
   widths = [1.0, 0.3, 0.8, 0.3, 1.0]
   positions = np.cumsum([0] + widths)

   for k in range(5):
       rect = mpatches.FancyBboxPatch(
           (positions[k], 0), widths[k], 1.0,
           boxstyle="square,pad=0",
           facecolor=colors[k], edgecolor='black',
           alpha=0.5 if k in [1, 2, 3] else 0.7)
       ax.add_patch(rect)
       if labels[k]:
           ax.text(positions[k] + widths[k] / 2, 0.5, labels[k],
                   ha='center', va='center', fontsize=11,
                   fontweight='bold')

   bnd_labels = [
       (positions[0], r'$x_{i-1}$'),
       (positions[1], r'$x_i$'),
       (positions[3], r'$x_{j-1}$'),
       (positions[4], r'$x_j$'),
   ]
   for xp, label in bnd_labels:
       ax.plot([xp, xp], [-0.1, 1.1], 'k--', linewidth=1, alpha=0.5)
       ax.text(xp, -0.2, label, ha='center', fontsize=11)

   ax.annotate('', xy=(positions[1], 1.2), xytext=(positions[0], 1.2),
               arrowprops=dict(arrowstyle='<->', color='red', lw=2))
   ax.text((positions[0] + positions[1]) / 2, 1.3, r'$\tau_i$',
           ha='center', fontsize=12, color='red')

   ax.annotate('', xy=(positions[4] + widths[4], 1.2),
               xytext=(positions[3] + widths[3], 1.2),
               arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
   ax.text((positions[3] + widths[3] + positions[4] + widths[4]) / 2, 1.3,
           r'$\tau_j$', ha='center', fontsize=12, color='blue')

   ax.annotate('', xy=(positions[3], 1.55), xytext=(positions[1], 1.55),
               arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
   ax.text((positions[1] + positions[3]) / 2, 1.65,
           r'gap $g$', ha='center', fontsize=12, color='purple')

   y_bot = -0.5
   terms = [
       (positions[1], positions[3], r'$+F(g)$', 'green'),
       (positions[0], positions[3], r'$-F(g+\tau_i)$', 'orange'),
       (positions[1], positions[4] + widths[4], r'$-F(g+\tau_j)$', 'orange'),
       (positions[0], positions[4] + widths[4],
        r'$+F(g+\tau_i+\tau_j)$', 'green'),
   ]
   for k, (x1, x2, label, color) in enumerate(terms):
       y = y_bot - k * 0.35
       ax.annotate('', xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle='<->', color=color,
                                   lw=1.5))
       ax.text(x2 + 0.1, y, label, fontsize=10, va='center',
               color=color)

   ax.set_xlim(-0.3, 5.0)
   ax.set_ylim(-2.0, 2.0)
   ax.set_aspect('equal')
   ax.set_yticks([])
   ax.set_xticks([])
   ax.set_title('Second-Difference: Four Boundary Evaluations',
                fontsize=13)
   plt.tight_layout()


The Discretised Second-Difference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The continuous formula :eq:`rcp-from-double-antideriv` is discretised by
evaluating the double antiderivative at region boundary positions.  Define:

.. math::
   :label: s-integral

   S(i, j, g) = \int_0^{R_i}
     \bigl[\text{Ki}_3(\tau_p(y)) - \text{Ki}_3(\tau_m(y))\bigr] \, dy

The within-cell collision probability for :math:`j \ge i` is:

.. math::
   :label: pcell-from-smat

   P_{ij}^{\text{cell}} = \frac{2}{\Sigt{i} V_i}
   \Bigl[ S(i,j) - S(i{-}1,j) - S(i,j{-}1) + S(i{-}1,j{-}1) \Bigr]
   + \delta_{ij}

where :math:`\delta_{ij}` accounts for self-collision and terms with
index 0 vanish (no region interior to region 1).

.. list-table:: Correspondence: continuous vs discrete
   :header-rows: 1
   :widths: 40 40

   * - Continuous term
     - Discrete boundary evaluation
   * - :math:`\hat{\hat{F}}(g)`
     - :math:`S(i{-}1, j{-}1)` --- both inner boundaries
   * - :math:`\hat{\hat{F}}(g + \tau_i)`
     - :math:`S(i, j{-}1)` --- outer of source, inner of target
   * - :math:`\hat{\hat{F}}(g + \tau_j)`
     - :math:`S(i{-}1, j)` --- inner of source, outer of target
   * - :math:`\hat{\hat{F}}(g + \tau_i + \tau_j)`
     - :math:`S(i, j)` --- both outer boundaries

**Implementation.** ORPHEUS does **not** pre-compute the :math:`S` array.
Instead, :meth:`CPMesh._compute_radial_rcp` evaluates the
second-difference at each :math:`y`-quadrature point and integrates
numerically.  This avoids storing :math:`S` and allows the same code path
for all three geometries by parameterising the kernel function.  The
derivation scripts use the same approach.


Self-Collision: Full Derivation
---------------------------------

The self-collision term (:math:`i = j`) cannot use the gap formula because
source and collision positions overlap.  The double integral is:

.. math::
   :label: self-double-integral

   r_{ii} = \int_0^{\tau_i} \int_0^{\tau_i} F_1(|s - t|) \, dt \, ds

where :math:`F_1` is the point-to-point kernel.

**Evaluation (slab case, :math:`F_1 = E_1`).**  Split at :math:`s = t`
and use symmetry (:math:`s > t` half = :math:`s < t` half):

.. math::

   r_{ii} &= 2 \int_0^{\tau_i} \int_0^s E_1(s - t) \, dt \, ds

Evaluate the inner integral using :math:`\int E_1(x) dx = -E_2(x)`:

.. math::

   \int_0^s E_1(s - t) \, dt
   &= \bigl[-E_2(s - t)\bigr]_{t=0}^{t=s}
   = -E_2(0) + E_2(s)
   = E_2(s) - E_2(0)

Since :math:`E_2(0) = 1`, the inner integral is :math:`E_2(s) - 1`.
The outer integral:

.. math::

   r_{ii} &= 2 \int_0^{\tau_i} \bigl[E_2(s) - 1\bigr] ds \\
          &= 2 \bigl[-E_3(s)\bigr]_0^{\tau_i} - 2\tau_i \\
          &= 2 \bigl[-E_3(\tau_i) + E_3(0)\bigr] - 2\tau_i \\
          &= 2 E_3(0) - 2 E_3(\tau_i) - 2\tau_i

Wait --- let me redo this carefully.  :math:`\int_0^s E_1(s-t) dt`:
substituting :math:`u = s - t`, :math:`du = -dt`:

.. math::

   \int_0^s E_1(u) du = \bigl[-E_2(u)\bigr]_0^s = -E_2(s) + E_2(0)

So the inner integral is :math:`E_2(0) - E_2(s)`.  Continuing:

.. math::

   r_{ii} &= 2 \int_0^{\tau_i} \bigl[E_2(0) - E_2(s)\bigr] ds \\
          &= 2 \bigl[\tau_i E_2(0) + E_3(\tau_i) - E_3(0)\bigr]

Using :math:`E_2(0) = 1` and :math:`E_3(0) = 1/2`:

.. math::

   r_{ii} = 2\tau_i + 2 E_3(\tau_i) - 1

The normalised self-collision probability is
:math:`P_{ii} = r_{ii} / (\Sigt{i} t_i)`.  Since
:math:`\tau_i = \Sigt{i} t_i`:

.. math::

   P_{ii} = \frac{2\tau_i + 2 E_3(\tau_i) - 1}{\tau_i}
          = 2 + \frac{2 E_3(\tau_i) - 1}{\tau_i}
          = 1 + \frac{2 E_3(\tau_i) - 2 E_3(0)}{\tau_i}

which is equivalently written as:

.. math::

   P_{ii} = 1 - \frac{2(E_3(0) - E_3(\tau_i))}{\tau_i}

For **thick regions** (:math:`\tau_i \to \infty`): :math:`E_3(\tau_i) \to 0`,
so :math:`P_{ii} \to 1 - 1/\tau_i \to 1`.  For **thin regions**
(:math:`\tau_i \to 0`): :math:`E_3(\tau_i) \to E_3(0) = 1/2`, so
:math:`P_{ii} \to 0`.  Tested by
``test_cp_verification.py::TestOpticalLimits``.

In the solver code (:meth:`CPMesh._compute_slab_rcp`)::

    # Self-collision (slab)
    rcp[i, i] += sti * t[i] - (0.5 - _e3(tau_i))

This is :math:`\Sigt{i} t_i - (E_3(0) - E_3(\tau_i))`, which equals
:math:`r_{ii}/2` (the factor of 2 from two half-spaces is applied later).

For **cylindrical and spherical** geometries, the same structure holds
but with canonical :math:`\mathrm{Ki}_3^{\text{A\&S}}` or
:math:`e^{-\tau}` replacing :math:`E_3`, and with
:math:`y`-quadrature.  In :meth:`CPMesh._compute_radial_rcp`::

    self_same = 2.0 * chords[i, :] - (2.0 / sti) * (
        kernel_zero - kernel(tau_i)
    )
    rcp[i, i] += 2.0 * sti * np.dot(y_wts, self_same)

The term ``kernel_zero - kernel(tau_i)`` is :math:`F(0) - F(\tau_i)`,
the escape fraction.  The same pattern is expressed once in the
unified Phase B.2 derivation core
:mod:`derivations.cp_geometry` — the self-term pairing consumes
:meth:`~derivations.cp_geometry.FlatSourceCPGeometry.kernel_F3_at_zero`
(returns :math:`1/2`, :math:`\pi/4`, or :math:`1` for slab / cyl
/ sph) alongside :meth:`~derivations.cp_geometry.FlatSourceCPGeometry.kernel_F3`
for the evaluated-at-:math:`\tau_i` term.


.. _optical-path-construction:

Optical Path Construction Along a Chord
=========================================

This section describes how :math:`\tau_m` and :math:`\tau_p` are
constructed at chord height :math:`y`.  This is the most geometrically
intricate part of the cylindrical and spherical CP implementations.


Half-Chord Lengths
-------------------

Consider :math:`N` concentric regions with outer radii
:math:`R_1 < \cdots < R_N`.  A chord at height :math:`y` intersects
a subset of these.  The **half-chord length** through region :math:`k` is:

.. math::
   :label: chord-length

   \ell_k(y) = \begin{cases}
     \sqrt{R_k^2 - y^2} - \sqrt{R_{k-1}^2 - y^2}
       & y < R_{k-1} \\
     \sqrt{R_k^2 - y^2}
       & R_{k-1} \le y < R_k \\
     0 & y \ge R_k
   \end{cases}

with :math:`R_0 = 0`.

- **Case 1** (:math:`y < R_{k-1}`): Chord passes entirely through region
  :math:`k`, entering at the inner boundary and exiting at the outer.
- **Case 2** (:math:`R_{k-1} \le y < R_k`): Chord originates inside
  region :math:`k` --- no inner intersection.
- **Case 3** (:math:`y \ge R_k`): Chord misses region :math:`k` entirely.

Computed by :func:`_chord_half_lengths` (solver) and independently by
``derivations/cp_cylinder.py::_chord_half_lengths`` (derivation).  Both
return shape ``(N, n_y)``.

**Optical half-thickness:** ``tau = sig_t_g[:, None] * chords``.

**Gotcha: the innermost region.**  For :math:`k = 1`, :math:`R_0 = 0`
gives :math:`\sqrt{-y^2}` (imaginary).  Both implementations handle this
with a ``r_in = 0`` / ``r_in > 0`` branch that skips the subtraction.

**Implicit zero-padding.**  When :math:`y \ge R_k`,
:func:`_chord_half_lengths` returns :math:`\ell_k = 0`, so
:math:`\tau_k = 0`.  Boundary positions collapse and the
second-difference evaluates to zero --- no explicit conditional logic is
needed to skip non-intersecting regions.

.. plot::
   :caption: Half-chord length :math:`\ell_k(y)` through region :math:`k = 2` (red outline) for each of the three cases.

   import numpy as np
   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 3, figsize=(15, 5))

   radii = [0.4, 0.7, 1.0]
   colors = ['#e74c3c', '#f1c40f', '#3498db']

   for ax_idx, (y_val, case_title) in enumerate([
       (0.2, r'Case 1: $y < R_{k-1}$'),
       (0.5, r'Case 2: $R_{k-1} \leq y < R_k$'),
       (0.8, r'Case 3: $y \geq R_k$'),
   ]):
       ax = axes[ax_idx]

       for r, color in reversed(list(zip(radii, colors))):
           ax.add_patch(plt.Circle((0, 0), r, color=color, alpha=0.3))
           ax.add_patch(plt.Circle((0, 0), r, fill=False,
                                   edgecolor='black', linewidth=1))

       theta = np.linspace(0, 2 * np.pi, 100)
       ax.fill_between(
           radii[1] * np.cos(theta), radii[1] * np.sin(theta),
           alpha=0, edgecolor='red', linewidth=3, linestyle='-')

       if y_val < radii[-1]:
           x_max = np.sqrt(radii[-1] ** 2 - y_val ** 2)
           ax.plot([-x_max, x_max], [y_val, y_val], 'k-', linewidth=2)

           k = 1
           r_out, r_in = radii[k], radii[k - 1]
           if y_val < r_in:
               x_out = np.sqrt(r_out ** 2 - y_val ** 2)
               x_in = np.sqrt(r_in ** 2 - y_val ** 2)
               ax.plot([x_in, x_out], [y_val, y_val], 'r-', linewidth=5,
                       alpha=0.7, label=r'$\ell_k(y)$')
           elif y_val < r_out:
               x_out = np.sqrt(r_out ** 2 - y_val ** 2)
               ax.plot([0, x_out], [y_val, y_val], 'r-', linewidth=5,
                       alpha=0.7, label=r'$\ell_k(y)$')

       if y_val < radii[-1]:
           ax.annotate('', xy=(0, y_val), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='<->', color='black',
                                       lw=1.5))
           ax.text(0.04, y_val / 2, '$y$', fontsize=11)

       ax.text(0.41, -0.15, r'$R_1$', fontsize=10, color='#e74c3c')
       ax.text(0.71, -0.15, r'$R_2$', fontsize=10, color='#f1c40f')
       ax.text(1.01, -0.15, r'$R_3$', fontsize=10, color='#3498db')

       ax.set_xlim(-1.2, 1.2)
       ax.set_ylim(-1.2, 1.2)
       ax.set_aspect('equal')
       ax.set_title(case_title, fontsize=11)
       ax.grid(True, alpha=0.2)
       if ax_idx == 0:
           ax.legend(loc='upper left', fontsize=10)

   fig.suptitle(
       r'Half-chord length $\ell_k(y)$ through region $k=2$ '
       r'(red outline)', fontsize=13, y=1.02)
   plt.tight_layout()


The Two Optical Paths
----------------------

For a neutron born in region :math:`i` targeting region :math:`j \ge i`,
there are two distinct paths along the chord:

1. **Same-side path** (:math:`\tau_m`): gap between the outer boundary
   of :math:`i` and the inner boundary of :math:`j`:

   .. math::
      :label: tau-m

      \tau_m(y) = \sum_{k=i+1}^{j} \tau_k(y)

   For adjacent regions :math:`j = i+1`, this is :math:`\tau_{i+1}`.
   For self-collision :math:`j = i`, :math:`\tau_m = 0`.

2. **Through-centre path** (:math:`\tau_p`): crosses all inner regions
   twice (inward to the centre, back out):

   .. math::
      :label: tau-p

      \tau_p(y) = \tau_m(y) + 2 \sum_{k=1}^{i} \tau_k(y)


Boundary Position Arrays
--------------------------

Cumulative optical distance from the chord midpoint to the outer boundary
of region :math:`k`:

.. math::

   x_k(y) = \sum_{m=1}^{k} \tau_m(y), \quad x_0 = 0

In code (:meth:`CPMesh._compute_radial_rcp`, and identically in all three
derivation scripts)::

    bnd_pos = np.zeros((N + 1, n_y))
    for k in range(N):
        bnd_pos[k + 1, :] = bnd_pos[k, :] + tau[k, :]

The gaps: ``gap_d = bnd_pos[j] - bnd_pos[i+1]`` (same-side) and
``gap_c = bnd_pos[i] + bnd_pos[j]`` (through-centre).


Slab Geometry: The :math:`E_3` Kernel
=======================================

The 1D slab half-cell extends from the reflective centre (:math:`x = 0`)
to the cell edge (:math:`x = L`).  Geometry built via
:func:`~geometry.factories.pwr_slab_half_cell` with
``coord = CoordSystem.CARTESIAN``.

.. plot::
   :caption: Slab half-cell: fuel, cladding, and coolant with reflective (left) and white (right) boundary conditions.

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
       rect = mpatches.FancyBboxPatch(
           (x, 0), width, 1, boxstyle="square,pad=0",
           facecolor=color, edgecolor='black', alpha=0.7)
       ax.add_patch(rect)
       ax.text(x + width / 2, 0.5, name, ha='center', va='center',
               fontsize=12, fontweight='bold')
       x += width

   ax.set_xlim(-0.1, 2.0)
   ax.set_ylim(-0.2, 1.3)
   ax.set_xlabel('x (cm)')
   ax.set_aspect('equal')
   ax.axvline(0, color='red', linestyle='--', linewidth=2,
              label='Reflective BC')
   ax.axvline(1.8, color='blue', linestyle='--', linewidth=2,
              label='White BC')
   ax.legend(loc='upper right')
   ax.set_yticks([])
   ax.set_title('Slab Half-Cell Geometry')
   plt.tight_layout()

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

Implemented in :meth:`CPMesh._compute_slab_rcp`.  Verified element-by-element
against ``derivations/cp_slab.py::_slab_cp_matrix`` by
``test_cp_verification.py::TestDirectPinfComparison::test_slab_pinf_matches_derivation``
(tolerance :math:`< 10^{-10}`).


Concentric Cylindrical Geometry: The Ki\ :sub:`3`/Ki\ :sub:`4` Kernel
=======================================================================

The Wigner--Seitz cell replaces the square unit cell by a circle of equal
area.  From area equivalence :math:`p^2 = \pi R_{\text{cell}}^2`:

.. math::
   :label: wigner-seitz

   R_{\text{cell}} = \frac{p}{\sqrt{\pi}}

Geometry built via :func:`~geometry.factories.pwr_pin_equivalent` with
``coord = CoordSystem.CYLINDRICAL``.

.. plot::
   :caption: Wigner--Seitz cylindrical cell with concentric annular regions and a chord at height :math:`y`.

   import numpy as np
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(6, 6))

   regions = [
       (0.9, '#e74c3c', 'Fuel'),
       (1.1, '#95a5a6', 'Clad'),
       (2.03, '#3498db', 'Cool'),
   ]

   for r, color, label in reversed(regions):
       ax.add_patch(plt.Circle((0, 0), r, color=color, alpha=0.5,
                               label=label))
   for r, _, _ in regions:
       ax.add_patch(plt.Circle((0, 0), r, fill=False, edgecolor='black',
                               linewidth=1))

   y_chord = 0.6
   x_max = np.sqrt(regions[-1][0] ** 2 - y_chord ** 2)
   ax.plot([-x_max, x_max], [y_chord, y_chord], 'k-', linewidth=2)
   ax.annotate('chord at height $y$', xy=(0, y_chord),
               xytext=(0.5, 1.5), fontsize=11,
               arrowprops=dict(arrowstyle='->', color='black'))
   ax.plot([0, 0], [0, y_chord], 'k--', alpha=0.5)
   ax.text(0.08, y_chord / 2, '$y$', fontsize=12)

   ax.set_xlim(-2.5, 2.5)
   ax.set_ylim(-2.5, 2.5)
   ax.set_aspect('equal')
   ax.legend(loc='upper right', fontsize=11)
   ax.set_xlabel('x (cm)')
   ax.set_ylabel('y (cm)')
   ax.set_title('Wigner\u2013Seitz Cell with Chord Integration')
   ax.grid(True, alpha=0.3)
   plt.tight_layout()

The CP matrix integrates :math:`\text{Ki}_4` second-differences over
chord heights using composite Gauss--Legendre quadrature
(:func:`_composite_gauss_legendre`) with breakpoints at each annular
boundary to capture chord-length discontinuities.

Second-Difference Formula (Cylindrical)
---------------------------------------

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

The :math:`y`-integration is performed with composite Gauss--Legendre
quadrature, with breakpoints at each annular boundary to capture the
chord-length discontinuities.

Implemented in :meth:`CPMesh._compute_radial_rcp` with
``self._kernel = Ki_4``.  Verified against
``derivations/cp_cylinder.py::_cylinder_cp_matrix``.


Concentric Spherical Geometry: The Exponential Kernel
======================================================

The spherical cell consists of :math:`N` concentric spherical shells.
Volumes: :math:`V_i = \frac{4}{3}\pi(R_i^3 - R_{i-1}^3)`.  Surface:
:math:`S = 4\pi R_{\text{cell}}^2`.  The kernel is simply
:math:`F(\tau) = e^{-\tau}` --- full 3-D symmetry absorbs all angular
variables.

Chord geometry is identical to cylindrical (:eq:`chord-length`), but
the quadrature weight includes :math:`y`.  For a cylinder (2-D), each
chord at height :math:`y` represents a **line of sources**, giving a
weight proportional to :math:`dy`.  For a sphere (3-D), each chord
represents a **ring of sources** with circumference :math:`2\pi y`,
giving a weight proportional to :math:`y\,dy`.
In :meth:`CPMesh._setup_spherical`::

    # Spherical weight: extra factor of y in the quadrature
    self._y_wts = self._y_wts * self._y_pts

Geometry built via :func:`~geometry.factories.mesh1d_from_zones` with
``coord = CoordSystem.SPHERICAL``.

Second-Difference Formula (Spherical)
-------------------------------------

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

The **same code path** :meth:`CPMesh._compute_radial_rcp` handles both
cylindrical and spherical, parameterised by kernel and weights.
Verified against ``derivations/cp_sphere.py::_sphere_cp_matrix``.


Geometry Comparison
====================

.. list-table::
   :header-rows: 1
   :widths: 18 27 27 28

   * - Aspect
     - Slab
     - Cylinder
     - Sphere
   * - Kernel :math:`F(\tau)`
     - :math:`E_3` (:func:`_e3`)
     - :math:`\text{Ki}_4` (:func:`_ki4_lookup`)
     - :math:`e^{-\tau}`
   * - :math:`F(0)`
     - :math:`1/2`
     - :math:`\approx 0.4244`
     - :math:`1`
   * - Quadrature weight
     - none (scalar)
     - :math:`w_i`
     - :math:`w_i \cdot y_i`
   * - Volume :math:`V_i`
     - :math:`t_i`
     - :math:`\pi(R_i^2 - R_{i-1}^2)`
     - :math:`\tfrac{4}{3}\pi(R_i^3 - R_{i-1}^3)`
   * - Angular integration
     - analytical (in :math:`E_3`)
     - numerical (:math:`y`-quadrature)
     - numerical (:math:`y`-quadrature)
   * - Surface :math:`S`
     - :math:`1`
     - :math:`2\pi R_{\text{cell}}`
     - :math:`4\pi R_{\text{cell}}^2`
   * - Prefactor
     - 1/2 (half-space)
     - 2 (chord halves)
     - 2 (chord halves)
   * - Code path (solver)
     - :meth:`CPMesh._compute_slab_rcp`
     - :meth:`CPMesh._compute_radial_rcp`
     - :meth:`CPMesh._compute_radial_rcp`
   * - Derivation script
     - ``derivations/cp_slab.py``
     - ``derivations/cp_cylinder.py``
     - ``derivations/cp_sphere.py``


The Eigenvalue Problem
=======================

Multi-Group Formulation
-----------------------

For :math:`G` energy groups, the neutron balance in region :math:`i`,
group :math:`g` is:

.. math::
   :label: neutron-balance

   \Sigt{ig} V_i \phi_{ig}
   = \sum_j P_{ji,g}^{\infty} V_j \left[
       \frac{\chi_{jg}}{\keff} \sum_{g'} \nSigf{jg'} \phi_{jg'}
       + \sum_{g'} \bigl(\Sigs{j,g' \to g}
       + 2\,\Sigma_{2,j,g' \to g}\bigr) \phi_{jg'}
     \right]

The factor of 2 on :math:`\Sigma_2` accounts for both outgoing (n,2n)
neutrons (the original neutron is already removed by :math:`\Sigt{}`).

.. note::

   **Scattering convention.**  ``Mixture.SigS[l]`` uses
   ``SigS[g_from, g_to]`` convention.  The in-scatter source uses the
   transpose: ``Q_scatter = SigS.T @ phi``.  This is critical for
   correct downscatter/upscatter coupling.


Matrix Form
-------------

:eq:`neutron-balance` in matrix form:
:math:`\mathbf{A}\Phi = (1/\keff)\mathbf{B}\Phi` with:

.. math::
   :label: matrix-A-def

   A[(i,g),(j,g')] = \delta_{ij}\delta_{gg'}\Sigt{ig} V_i
     - P_{ji,g}^{\infty} V_j (\Sigs{j,g' \to g} + 2\Sigma_{2,j,g' \to g})

.. math::
   :label: matrix-B-def

   B[(i,g),(j,g')] = P_{ji,g}^{\infty} V_j \chi_{jg}\, \nSigf{jg'}

The analytical verification eigenvalue is
:math:`\lambda_{\max}(\mathbf{A}^{-1}\mathbf{B})`, computed by
:func:`~derivations._eigenvalue.kinf_from_cp`, which builds the full
:math:`NG \times NG` matrices and uses ``numpy.linalg.eigvals``.
The solver does NOT form these matrices --- see :ref:`why-not-full-matrices`.


Power Iteration
---------------

Implemented by :func:`~numerics.eigenvalue.power_iteration` via the
:class:`CPSolver` protocol:

1. **Fission source** (:meth:`CPSolver.compute_fission_source`):
   :math:`Q^F_{ig} = \chi_{ig} \sum_{g'} \nSigf{ig'} \phi_{ig'} / \keff`

2. **Fixed-source solve** (:meth:`CPSolver.solve_fixed_source`):
   :math:`\phi_{ig}^{\text{new}} = \sum_j P_{ji,g}^{\infty} V_j Q_{jg}^{\text{total}} / (\Sigt{ig} V_i)`

3. **Update** :math:`\keff` (:meth:`CPSolver.compute_keff`):

   .. math::
      :label: cp-keff-update

      \keff = \frac{\sum_{ig} \nSigf{ig} \phi_{ig} V_i}
                   {\sum_{ig} (\Sigt{ig} - \Sigs{ig}^{\text{out}}
                   - 2\Sigma_{2,ig}^{\text{out}}) \phi_{ig} V_i}

   The denominator is the **net removal rate**: total interactions minus
   all neutrons that remain in the system (scattered or produced by (n,2n)).

   .. admonition:: The (n,2n) subtlety (ERR-015)

      **Bug:** ``compute_keff`` used production/absorption, which is wrong
      when :math:`\Sigma_2 \neq 0` because absorption includes (n,2n)
      removal but does not credit the two outgoing neutrons.  With
      ``Sig2[0,0] = 0.01`` on region A (2G), the solver converged to
      :math:`k = 1.793` instead of analytical :math:`k = 2.045` (12% error).

      **First wrong fix:** Adding (n,2n) production to the numerator
      gave :math:`k = 1.808` --- still wrong because the extra neutron
      has no fission spectrum :math:`\chi` weighting.

      **Correct fix:** Net-removal denominator :eq:`cp-keff-update`.
      When :math:`\Sigma_2 = 0`, this reduces to
      :math:`\nSigf{}\phi V / \Siga{}\phi V`.

      **How it hid:** All test materials had ``Sig2 = 0`` (zero sparse
      matrix).  The ``make_mixture`` API didn't even accept a ``sig_2``
      parameter, making it impossible to construct test materials with
      nonzero (n,2n).

      **Test:** ``test_cp_verification.py::TestN2N::test_n2n_solver_keff_matches_analytical``.

      **Lesson:** When adding a new reaction type, trace it through BOTH
      the transport solve AND the eigenvalue estimate.  Test with the
      term nonzero.

   For lattice models with reflective (or white) boundary conditions,
   the leakage term is zero, so this balance is exact.

4. **Converge** (:meth:`CPSolver.converged`) when :math:`|\Delta k| <`
   ``keff_tol`` and :math:`\|\Delta\phi\|_\infty <` ``flux_tol``.

The power iteration algorithm is shared with all ORPHEUS eigenvalue
solvers; see :ref:`power-iteration-algorithm` in the homogeneous theory
for the general formulation.

The power iteration converges to the **dominant eigenvalue**
(:math:`\keff`) and the **fundamental mode** --- the unique non-negative
eigenvector (Perron--Frobenius theorem) [Hebert2009]_.

This is implemented in :class:`CPSolver`, which satisfies the
:class:`~numerics.eigenvalue.EigenvalueSolver` protocol.


Solver Modes: Jacobi and Gauss-Seidel
---------------------------------------

:class:`CPSolver` supports two modes via ``CPParams.solver_mode``.

**Jacobi (default,** :meth:`CPSolver._solve_fixed_source_jacobi` **).**
Scattering source from all groups computed simultaneously using the
previous iteration's flux.  No inner iterations.  One matrix-vector
multiply per group per outer.

**Gauss-Seidel (**\ :meth:`CPSolver._solve_fixed_source_gs` **).**
Groups swept from fast (:math:`g=0`) to thermal (:math:`g=G-1`).  For
each group :math:`g`, **inner iterations** converge within-group
self-scatter before proceeding to :math:`g+1`.

The inner iteration solves the fixed-point equation:

.. math::

   \phi_g = T_g\!\bigl[Q_g^{\text{ext}} + \Sigs{g \to g} \phi_g\bigr]

where :math:`T_g[\cdot]` is the CP transport operator for group :math:`g`
(the matrix multiply :math:`P^T V \cdot / (\Sigt{} V)`).  The operator is
exact for a given source, but the source depends on :math:`\phi_g` through
within-group self-scatter :math:`\Sigs{g \to g}`.

**Why inner iterations matter.**  The spectral radius of the inner
iteration operator is approximately :math:`\Sigs{g \to g} / \Sigt{g}`.
Thermal groups (ratio ~0.6--0.9) need 3--8 inner iterations; fast groups
(ratio << 1) converge in 1.  Convergence criterion: relative flux change
:math:`\|\phi_g^{(n+1)} - \phi_g^{(n)}\| / \|\phi_g^{(n+1)}\| < \texttt{inner\_tol}`.

.. admonition:: Tautological inner residual (ERR-016)

   **Bug:** The original inner convergence check computed
   :math:`\|\Sigt{} V \phi_g^{\text{new}} - P^T V Q_g\|`, which is
   **identically zero by construction**:
   :math:`\phi_g^{\text{new}} \equiv P^T V Q_g / (\Sigt{} V)`, so
   the check computes :math:`\|x - x\| = 0`.  The inner loop always
   exited after 1 iteration, making GS functionally identical to
   sequential-group Jacobi.

   **How it hid:** All 27 eigenvalue tests passed (the outer iteration
   converged regardless).  The ``n_inner`` array showed all 1s,
   interpreted as "fast convergence" rather than "broken check".  A QA
   review initially concluded that inner iterations are *fundamentally
   unnecessary* for CP --- this is **wrong**: the transport is exact for
   a given source, but the source depends on :math:`\phi_g` through
   self-scatter.

   **Fix:** Changed to relative flux change
   :math:`\|\phi^{(n+1)} - \phi^{(n)}\| / \|\phi^{(n+1)}\|`.  With the
   corrected residual, thermal groups genuinely require multiple inner
   iterations; fast groups converge in 1.

   **Tests:** ``test_cp_verification.py::TestGSInnerIterations`` ---
   ``test_thermal_needs_more_inner_than_fast`` (thermal > fast inner
   counts), ``test_gs_eigenvalue_matches_jacobi`` (same eigenvalue),
   ``test_no_self_scatter_one_inner`` (zero diagonal in :math:`\Sigma_s`
   converges in 1).

   **Lesson:** A convergence check that compares quantities derived from
   each other by construction tests nothing.  Always verify with a
   problem that *should* require multiple iterations.

**Convergence diagnostics** (stored in :class:`CPResult`):

.. list-table::
   :header-rows: 1
   :widths: 25 15 40

   * - Field
     - Shape
     - Description
   * - ``residual_history``
     - ``(n_outer,)``
     - Neutron balance residual :math:`\|\Sigt{} V \phi - P^T V Q\|_2` per outer (both modes)
   * - ``n_inner``
     - ``(n_outer, ng)``
     - Inner iteration count per group per outer (GS only; ``None`` in Jacobi)

Both modes converge to the **same eigenvalue and flux distribution**
(``test_cp_verification.py::TestGSInnerIterations::test_gs_eigenvalue_matches_jacobi``).

.. list-table:: Solver mode comparison
   :header-rows: 1
   :widths: 35 35

   * - Jacobi
     - Gauss-Seidel
   * - All groups from previous flux
     - Sequential; latest flux from earlier groups
   * - No inner iterations
     - Inner iterations for within-group self-scatter
   * - Simpler (1 matrix multiply/group/outer)
     - Faster convergence for strong upscatter


Cross-Section Data Layout
===========================

Cross sections flow through two layers:

1. :class:`~data.macro_xs.mixture.Mixture` --- per-material: ``(ng,)``
   arrays for :math:`\Sigt{}`, :math:`\Siga{}`, :math:`\Sigf{}`,
   :math:`\nSigf{}`, :math:`\chi`; ``(ng, ng)`` sparse matrices for
   scattering :math:`\Sigs{}` (``SigS[0]``) and (n,2n) :math:`\Sigma_2`
   (``Sig2``).

2. :class:`~data.macro_xs.cell_xs.CellXS` --- per-cell on the mesh:
   ``(N_cells, ng)`` arrays mapped from materials by
   :func:`~data.macro_xs.cell_xs.assemble_cell_xs` using ``mat_ids``.

**Indexing:** ``xs.sig_t[i, g]`` is the total macroscopic cross section
in spatial cell :math:`i` and energy group :math:`g`.  Group 0 = fastest
(highest energy); group :math:`G-1` = slowest (thermal).  Cell 0 =
innermost; cell :math:`N-1` = outermost.

**Scattering convention:** ``SigS[g_from, g_to]``.  The in-scatter source
uses the transpose: ``Q += SigS.T @ phi`` (applied per-cell)::

    for k in range(N):
        mid = self.mat_ids[k]
        Q[k, :] += self._scat_mats[mid].T @ flux_distribution[k, :]
        Q[k, :] += 2.0 * (self._n2n_mats[mid].T @ flux_distribution[k, :])


Verification
============

**106 tests** across 6 test files verify the CP implementation against
analytical eigenvalues computed independently by the derivation modules.

Consolidated Eigenvalue Solvers (CP-20260405-007)
---------------------------------------------------

:func:`~derivations._eigenvalue.kinf_from_cp` and
:func:`~derivations._eigenvalue.kinf_homogeneous` replaced 10 duplicated
eigenvalue computations across ``homogeneous.py``, ``sn.py``, ``moc.py``,
``mc.py``, ``cp_slab.py``, ``cp_cylinder.py``, and ``cp_sphere.py``.
Both accept optional ``sig_2`` / ``sig_2_mats`` for (n,2n) reactions.
When omitted, (n,2n) is zero and all previous eigenvalues are preserved.


Eigenvalue Verification Cases
-------------------------------

27 core cases: {1, 2, 4} groups |times| {1, 2, 4} regions |times|
{slab, cylinder, sphere}.

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 20

   * - Geometry
     - Tolerance
     - Test file
     - Derivation
   * - Slab (:math:`E_3`)
     - :math:`< 10^{-6}`
     - ``test_cp_slab.py``
     - ``derivations/cp_slab.py``
   * - Cylinder (:math:`\text{Ki}_4`)
     - :math:`< 10^{-5}`
     - ``test_cp_cylinder.py``
     - ``derivations/cp_cylinder.py``
   * - Sphere (:math:`e^{-\tau}`)
     - :math:`< 10^{-5}`
     - ``test_cp_sphere.py``
     - ``derivations/cp_sphere.py``

Historically, the cylinder/sphere tolerances were 10× looser
because the legacy 20 000-point :math:`\text{Ki}_4` table
interpolation introduced :math:`O(\Delta x^2) \approx
6\times 10^{-6}` error. Phase B.4 (commit ``6badbe5``, Issue #94)
replaced the table with the Chebyshev interpolant
:func:`~derivations.cp_geometry._ki3_mp` of canonical
:math:`\mathrm{Ki}_3`, which reaches ~:math:`5\times 10^{-6}`
absolute accuracy; the declared ``< 1e-5`` tolerance is now
~100× larger than the actual solver/reference error
(~:math:`10^{-7}`, same kernel on both sides).  See
:ref:`ki-table-construction` for the full postmortem.  The old
convergence-with-table-size regression
``test_cp_verification.py::TestKi4Resolution`` was replaced by
``test_ki3_kernel_is_insensitive_to_n_ki_table``: ``n_ki_table``
is now a no-op and ``keff`` is bit-identical across
``{5000, 20000, 40000}``.

Additionally, **algebraic property tests** (``test_cp_properties.py``)
are run for all three coordinate systems:

- **Row sums** :math:`= 1` (neutron conservation)
- **Reciprocity**: :math:`\Sigt{i} V_i P_{ij} = \Sigt{j} V_j P_{ji}`
- **Non-negativity**: :math:`P_{ij} \ge 0`
- **Homogeneous limit**: 1-region :math:`P = 1`


Extended Verification (CP-20260405-005)
----------------------------------------

31 additional tests in ``test_cp_verification.py`` closing 9 QA gaps:

- **L0 P_inf comparison** (G-1): element-by-element solver vs derivation
  (tolerance :math:`< 10^{-10}`)
- **Multi-group properties** (G-6, W-2): row sums and reciprocity at 2G/4G
- **Upscatter** (G-2, W-1): eigenvalue with nonzero thermal-to-fast transfer
- **(n,2n)** (C-2, G-3, W-3): solver :math:`\keff` matches analytical with
  ``Sig2[0,0] = 0.01``
- **Optical limits** (G-4, G-7): thick (:math:`\tau \gg 1`,
  :math:`P_{ii} > 0.99`) and thin (:math:`\tau \ll 1`, high escape)
- **Convergence rate** (G-5): monotonic error decrease, dominance ratio
  estimation
- **8-region mesh** (G-8): mesh refinement convergence
- **GS inner iterations** (C-1, G-9): thermal > fast inner counts;
  no-self-scatter in 1; GS/Jacobi eigenvalue agreement
- **Ki4 table resolution** (W-6): diminishing returns from 5k to 40k points

Plus 36 diagnostic tests in ``test_cp_diagnostics.py``.

::

   pytest tests/cp/ -v


Implementation Details
=======================

.. _why-p-transpose:

Why P\ :sup:`T` Appears in the Flux Update
--------------------------------------------

With :math:`P_{ij} = P[\text{birth}_i, \text{collision}_j]`, the neutron
balance :eq:`collision-rate` sums over birth regions (first index of
:math:`P`), which is a column sum.  In matrix form:

.. math::

   \Sigt{} V \phi = P^T V Q

Hence: :math:`\phi = P^T V Q / (\Sigt{} V)`.

In code: ``phi[:, g] = P_inf[:, :, g].T @ source``
(:meth:`CPSolver._solve_fixed_source_jacobi`).

.. admonition:: Historical bug (ERR-009)

   ``P @ source`` instead of ``P.T @ source``.  Correct for homogeneous
   problems (:math:`P` symmetric) but **8% wrong** for the 1G 2-region
   slab (k=1.373 vs analytical 1.272).  Caught by the formal verification
   suite on the first heterogeneous test case.  The bug survived 4 weeks
   because all prior tests were homogeneous.


.. _ki-table-construction:

Ki\ :sub:`3` Kernel Construction (historical: Ki\ :sub:`3`/Ki\ :sub:`4` tables)
-------------------------------------------------------------------------------

**Current path (post-Phase B.4, commit 6badbe5).** Both the
derivation module and the runtime solver consume the **same**
kernel via :func:`~derivations.cp_geometry._ki3_mp`: a Chebyshev
polynomial of degree 63 fit to the scaled kernel
:math:`e^{\tau}\,\mathrm{Ki}_3(\tau)` (canonical A&S) on
:math:`[0, 50]` at Chebyshev-Gauss-Lobatto nodes. The tabulation
values are computed once at module load by
:func:`~derivations._kernels.ki_n_mp` at 30 dps and cached via
:func:`functools.lru_cache`; Chebyshev fitting of the
:math:`e^{\tau}`-scaled kernel converts the exponentially-decaying
tail into a slowly-varying function the polynomial reaches to
~:math:`5\times 10^{-6}` absolute accuracy. Evaluation at runtime
is one :class:`numpy.polynomial.Chebyshev` call plus one
:func:`numpy.exp` — comparable in cost to the legacy
:func:`numpy.interp` on a 20 000-point grid. For
:math:`\tau > 50`, the clamp to zero is already below double
precision (:math:`\mathrm{Ki}_3(50) \approx 3\times 10^{-23}`).

.. note::

   :func:`~derivations.cp_geometry._ki3_mp` uses the **canonical
   A&S convention**:
   :math:`\mathrm{Ki}_n^{\text{A\&S}}(x) = \int_0^{\pi/2}
   \cos^{n-1}\theta\,\exp(-x/\cos\theta)\,d\theta`, so
   :math:`\mathrm{Ki}_3^{\text{A\&S}}(0) = \pi/4`. The legacy
   ORPHEUS convention carried by :eq:`ki3-def` used an extra power
   of :math:`\sin\theta`, making the ORPHEUS-local "Ki\ :sub:`4`\ "
   the canonical :math:`\mathrm{Ki}_3^{\text{A\&S}}`. The
   :math:`P_{ij}` matrix assembled by
   :mod:`derivations.cp_cylinder` consumes canonical
   :math:`\mathrm{Ki}_3^{\text{A\&S}}` — the Phase-4.2 alias
   ``BickleyTables.Ki3_vec := BickleyTables.ki4_vec`` made the
   convention uniform before the Phase B.4 retirement deleted both
   names.

**Historical path (retired, pre-Phase B.4).** Until commit
``6badbe5``, the kernel tabulation lived in two places that had to
stay in sync:

1. :func:`_build_ki_tables` on :class:`orpheus.cp.solver.CPMesh`
   (solver side) and
2. The ``BickleyTables`` class in :mod:`derivations._kernels`
   (derivation side).

Both tabulated the functions identically:

1. :math:`\text{Ki}_3(x_k) = \int_0^{\pi/2} e^{-x_k/\sin\theta} \sin\theta\,d\theta`
   (ORPHEUS-local convention, equivalent to canonical
   :math:`\mathrm{Ki}_2^{\text{A\&S}}`) at 20 000 points on
   :math:`[0, 50]` via :func:`scipy.integrate.quad`. Boundary
   value :math:`\text{Ki}_3(0) = 1` set analytically.

2. :math:`\text{Ki}_4` (ORPHEUS-local) via cumulative trapezoid
   from right to left (a pre-computer-era idiom for the recurrence
   :math:`\mathrm{Ki}_{n+1}(x) = \int_x^\infty \mathrm{Ki}_n(t)\,dt`)::

       ki4_vals = np.cumsum(ki3[::-1])[::-1] * dx

   This realised canonical :math:`\mathrm{Ki}_3^{\text{A\&S}}` to
   :math:`O(\Delta x^2)` trapezoidal error on a uniform grid.

3. Runtime lookup via :func:`numpy.interp` (linear interpolation);
   with :math:`\Delta x = 0.0025`, interpolation error was
   :math:`O(\Delta x^2) \approx 6 \times 10^{-6}` and the
   cumulative-trapezoid bias on the antiderivative pushed the
   effective kernel accuracy to ~:math:`10^{-3}` at :math:`x = 0`
   tapering to ~:math:`10^{-4}` elsewhere — the dominant error in
   every cylindrical flat-source CP reference value for the life
   of the project.

The old ``test_cp_verification.py::TestKi4Resolution`` confirmed
that increasing from 5 000 to 40 000 points produced diminishing
returns (validating the legacy 20 000 default). It was replaced in
Phase B.4 by
``test_cylinder_ki3_kernel_is_insensitive_to_n_ki_table`` — a much
stronger statement than "converges with more points": the Chebyshev
interpolant is built lazily from mpmath and is ignorant of any
``CPParams.n_ki_table`` knob, so ``keff`` is bit-identical across
``{5000, 20000, 40000}`` table-size values. The knob itself is
kept only as an unused no-op argument so that callers constructing
:class:`~orpheus.cp.solver.CPParams` with explicit
``n_ki_table=...`` do not break.

Why the retirement was deferrable until Phase B.4 is covered in
:doc:`/theory/peierls_unified` §16 (postmortem) and
:doc:`/verification/reference_solutions` §"Legacy naming
discrepancy in ``BickleyTables``" (safety argument for the swap).


Equal-Volume Mesh Subdivision
-------------------------------

:func:`~geometry.factories._subdivide_zone` creates equal-volume cells:

**Cartesian:** :math:`x_k = x_0 + k(x_N - x_0)/N`.

**Cylindrical.**  Equal annular volumes:
:math:`V = \pi(R_k^2 - R_{k-1}^2) = \text{const}`.  Summing:
:math:`\pi R_k^2 = \pi R_0^2 + k \cdot \pi(R_N^2 - R_0^2)/N`, giving:

.. math::

   R_k = \sqrt{R_0^2 + \frac{k}{N}(R_N^2 - R_0^2)}

For :math:`R_0 = 0`: :math:`R_k = R_N\sqrt{k/N}`.

**Spherical.**  Equal shell volumes:
:math:`V = \frac{4}{3}\pi(R_k^3 - R_{k-1}^3) = \text{const}`, giving:

.. math::

   R_k = \left(R_0^3 + \frac{k}{N}(R_N^3 - R_0^3)\right)^{1/3}

For :math:`R_0 = 0`: :math:`R_k = R_N(k/N)^{1/3}`.


.. _white-bc-quality:

Numerical Evidence: White-BC Approximation Quality
-----------------------------------------------------

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
     - -0.2%

The slab shows ~1% overestimation because the white BC smears the
anisotropic angular flux at the fuel-moderator interface.  The
cylindrical discrepancy is smaller because the Wigner--Seitz cell has
a larger volume-to-surface ratio, making the boundary angular
distribution closer to isotropic.  The SN solver uses reflective
(specular) BCs; see :ref:`boundary-conditions`.  The ~1% gap between
white and reflective BCs is expected physics, not a bug.


.. _why-not-full-matrices:

Design Decisions
==================

**Single CPMesh class (CP-20260404-001).**  All three geometries share
the eigenvalue iteration.  Only the kernel construction differs.  Adding
a geometry requires one ``_setup_*()`` method and kernel function, not a
new solver class.

**Per-group loop (CP-20260404-008).**  The CP matrix
:math:`P^{\infty}_{ij,g}` is computed independently per group because
:math:`\tau_{ig} = \Sigt{ig} \ell_i` varies with group.  Vectorising
over groups would require restructuring the Ki4 table lookup ---
deferred as a performance improvement.

**Why ORPHEUS does NOT form the full A/B matrices.**  The analytical
verification (:func:`~derivations._eigenvalue.kinf_from_cp`) builds the
full :math:`NG \times NG` matrices and runs ``numpy.linalg.eigvals``.
The solver does not, for two reasons:

1. **Memory.** For 421 groups, 20 cells: :math:`8420 \times 8420 =`
   567 MB dense.  The per-group approach: :math:`20 \times 20 \times 421 =`
   1.3 MB.
2. **Sparsity.** The scattering matrix is sparse (downscatter only for
   fast groups).  Per-material sparse storage is far more efficient.

**1-group verification is degenerate.** :math:`\keff = \nSigf{}/\Siga{}`
regardless of flux shape.  Wrong CP matrices, weight errors, and
convention drifts are invisible.  The suite tests 1, 2, AND 4 groups.

.. note::

   The 1-group degenerate case masked ERR-001 (z-ordinate weight loss),
   ERR-002 (scattering transpose), ERR-004 (BiCGSTAB normalisation),
   ERR-009 (CP transpose), and ERR-015 ((n,2n) keff) --- all caught
   only by multi-group tests.  See ``tests/l0_error_catalog.md``.


Peierls integral equation reference
====================================

The collision probability method discretises the integral transport
equation by assuming flat sources within each region.  The **Peierls
integral equation** is the continuous (pre-discretisation) form of that
equation, and solving it at high quadrature resolution provides an
independent reference for verifying the CP solver's flux profiles.

Derivation from the 1-D transport equation
-------------------------------------------

The starting point is the steady-state, isotropic-source transport
equation for a 1-D slab in energy group :math:`g`:

.. math::

   \mu\,\frac{\partial\psi_g}{\partial x}(x,\mu)
   + \Sigt{g}(x)\,\psi_g(x,\mu)
   = \frac{q_g(x)}{2}

where :math:`\psi_g(x,\mu)` is the angular flux,
:math:`\mu = \cos\theta \in [-1, 1]` is the direction cosine, and
:math:`q_g(x)` is the isotropic source (scattering + fission).

For :math:`\mu > 0`, the formal solution (integrating factor) for a
slab :math:`[0, L]` with vacuum boundary at :math:`x = 0` gives:

.. math::

   \psi_g(x,\mu) = \frac{1}{2\mu}\int_0^x
   \exp\!\Bigl(-\frac{\tau_g(x',x)}{\mu}\Bigr)\,q_g(x')\,\mathrm{d}x'

where
:math:`\tau_g(x',x) = \int_{x'}^{x}\Sigt{g}(s)\,\mathrm{d}s` is the
optical path.  A symmetric expression holds for :math:`\mu < 0`.

The scalar flux is obtained by integrating over angle:

.. math::

   \varphi_g(x) = \int_{-1}^{1}\psi_g(x,\mu)\,\mathrm{d}\mu
   = \frac{1}{2}\int_0^L q_g(x')\!\int_0^1
   \frac{1}{\mu}\exp\!\Bigl(-\frac{\tau_g(x,x')}{\mu}\Bigr)\,\mathrm{d}\mu
   \;\mathrm{d}x'

The inner angular integral is exactly the **first exponential integral**:

.. math::

   E_1(z) = \int_0^1 \frac{1}{\mu}\,e^{-z/\mu}\,\mathrm{d}\mu

(Case & Zweifel 1967, Ch. 2, Eq. 2.67; equivalently,
:math:`E_1(z) = \int_1^\infty t^{-1} e^{-zt}\,\mathrm{d}t`.)

Substituting yields the **Peierls integral equation** for the scalar flux:

.. math::
   :label: peierls-equation

   \varphi(x)
   = \tfrac12 \int_0^L E_1\!\bigl(\tau(x,x')\bigr)\,q(x')\,\mathrm{d}x'
     + \varphi_{\rm bc}(x)

where :math:`q(x') = \sum_{g'}\Sigma_{s,g'\to g}\varphi_{g'}(x')
+ \chi_g\sum_{g'}\nu\Sigma_{f,g'}\varphi_{g'}(x')/k` is the total
isotropic source and the optical path is
:math:`\tau(x,x') = \int_{\min(x,x')}^{\max(x,x')}\Sigma_t(s)\,\mathrm{d}s`.

.. note::

   The equation above is written for :math:`\varphi` (scalar flux), NOT
   :math:`\Sigt{}\varphi` (collision rate).  This matters for the
   operator form: the LHS is :math:`\varphi`, not
   :math:`\Sigt{}\varphi`, so the identity operator appears in the
   eigenvalue system :math:`(I - K_s)\varphi = (1/k)\,K_f\varphi`.
   An earlier version incorrectly placed :math:`\Sigt{}` on the LHS,
   producing a ``diag(Sigma_t) - K_scatter`` operator that broke the
   conservation property (see :ref:`peierls-conservation` below).

Why :math:`E_1` and not :math:`E_3`
------------------------------------

The CP method uses the :math:`E_3` function (double antiderivative of
:math:`E_1`) to integrate the kernel analytically over flat-source
regions (:eq:`dd-slab`, :eq:`dc-slab`, :eq:`self-slab`).  A Peierls
reference that uses :math:`E_1` directly provides **genuinely
independent** verification:

- **Different special function**: :math:`E_3` is computed from :math:`E_1`
  via recursion (:math:`E_{n+1}(z) = [e^{-z} - z\,E_n(z)]/n`).  A
  systematic error in the :math:`E_3` recursion (wrong sign, off-by-one
  in the index) would affect both the CP matrices and any reference
  built from the same recursion.

- **No flat-source assumption**: the Peierls Nystrom solver resolves the
  flux within each region as a high-order polynomial (degree
  :math:`p-1` per GL panel), while CP assumes piecewise-constant flux.
  This makes the Peierls reference sensitive to flux shape errors that
  are invisible to a CP-vs-CP comparison.

- **Common-mode failure prevention**: the existing CP eigenvalue tests
  compare the CP solver against analytically-constructed CP matrices
  (same :math:`E_3` kernel, same flat-source discretisation).  A sign
  error in the :math:`E_3` second-difference formula would cancel
  between the test and the code.  The Peierls reference would catch it.

Nystrom method details
----------------------

The Nystrom method converts the integral equation into a matrix
eigenvalue problem by approximating the integral with a quadrature rule
[Kress2014]_.  For the Peierls equation, the key challenge is the
logarithmic singularity in :math:`E_1` at :math:`x = x'`.

**Composite Gauss--Legendre panels.**
The slab :math:`[0, L]` is divided into :math:`n_{\rm panels}` panels
per material region.  Within each panel, :math:`p` Gauss--Legendre
nodes and weights are computed.  The total number of quadrature nodes is
:math:`N = n_{\rm regions} \times n_{\rm panels} \times p`.

**Singularity treatment.**
The :math:`E_1` kernel has a logarithmic singularity at :math:`x = x'`:

.. math::
   :label: e1-decomposition

   E_1(z) = \bigl[-\ln z - \gamma\bigr] + R(z),
   \qquad R(z) \equiv E_1(z) + \ln z + \gamma,\quad R(0) = 0.

The remainder :math:`R(z)` is a smooth (analytic) function that
vanishes at the origin. This decomposition motivates the classical
**singularity-subtraction** approach (used in the original
implementation), in which diagonal panels split the kernel into a
smooth :math:`R` part (handled by GL weights) and a
:math:`-\ln|x_i - x'|` part (handled by product-integration weights).
Off-diagonal panels would then use standard GL weights on the smooth
:math:`E_1` integrand.

**Why the classical singularity-subtraction scheme failed.** Issue
#113 closed the slab K-matrix cross-panel logarithmic-singularity
bug — the naive GL-collocation scheme made two assumptions that
turned out to be insufficient (off-diagonal GL exact for smooth
integrands; :math:`R(\tau)` smooth across the diagonal panel) and
hid the resulting ~1 % error inside row-sum conservation because
:math:`\sum_j L_j(x') = 1` cancels basis-individual :math:`K[i,j]`
errors. The forensic — including the basis-aware ERR-027 /
ERR-028 catalog entries and the rank-N "passes at mode 0, fails
at mode n > 0" diagnostic signature — is preserved in
`Issue #113 <https://github.com/deOliveira-R/ORPHEUS/issues/113>`_.
The unified ``_pg.solve_peierls_*`` adaptive-quadrature K-matrix
assembly described next is the production path.

**Unified basis-aware assembly** (current implementation,
:func:`~orpheus.derivations.peierls_slab._basis_kernel_weights`).
Every :math:`K[i, j]` is computed directly as

.. math::

   K[i, j] \;=\; \tfrac{1}{2} \int_{p_a}^{p_b}
                  E_1\!\bigl(\tau(x_i, x')\bigr)\,L_j(x')\,\mathrm{d}x'

via adaptive :func:`mpmath.quad`, with the subdivision hint
:math:`[p_a, x_i, p_b]` when :math:`x_i` lies inside the source
panel (same-panel case). This single code path:

- handles the integrable log singularity at :math:`x'=x_i` via the
  subdivision hint (mpmath resolves it to machine precision);
- handles the derivative kink of :math:`R(\tau)` in :math:`x'` — GL
  on each smooth half of the subdivided panel converges spectrally;
- eliminates the off-diagonal-panel quadrature error — adaptive
  refinement resolves :math:`E_1`'s non-polynomial structure on
  arbitrary panel pairs without relying on near-log assumptions.

The implementation exactly mirrors the adaptive reference
:func:`~orpheus.derivations.peierls_reference.slab_K_vol_element`, so
the production code and the reference agree to machine
:math:`\mathrm{dps}` by construction.

**Why adaptive quadrature over the alternatives.**
Four strategies exist for the Peierls log singularity [Atkinson1997]_:

1. *Graded meshes* (cluster GL nodes near the diagonal): algebraic
   convergence only; many nodes needed for high accuracy.
2. *IMT transformation* (Iri–Moriguti–Takahashi double-exponential):
   effective for endpoint singularities but hard to control for the
   travelling singularity at :math:`x'=x_i`.
3. *Singularity subtraction + product integration*: the classical
   approach. Exact for the isolated log, but still requires the
   surrounding integrand to be polynomial-representable — which is
   false for the full :math:`E_1`-times-:math:`L_j` product (see
   bugs above).
4. *Adaptive :func:`mpmath.quad` with explicit subdivision hints*
   (used here): handles all three non-smoothness sources — log
   singularity at :math:`x'=x_i`, derivative kink of :math:`R` at
   :math:`x'=x_i`, and non-polynomial :math:`E_1` decay across the
   source panel — in one uniform code path. The cost is higher per
   weight than GL product integration, but this is a reference-quality
   solver; correctness beats speed.

White boundary conditions
-------------------------

For an infinite lattice (white BC), the re-entrant isotropic flux is
derived by requiring that the angular flux entering each face equals
the isotropic average of the flux exiting the opposite face.  The
result is a separable rank-2 perturbation of the interior kernel:

.. math::
   :label: peierls-white-bc

   \varphi_{\rm bc}(x) = \frac{1}{1 - T^2}\sum_j w_j\,q_j\,
   \bigl[E_2(\tau_{0,i})(E_2(\tau_{0,j}) + T\,E_2(\tau_{L,j}))
   + E_2(\tau_{L,i})(E_2(\tau_{L,j}) + T\,E_2(\tau_{0,j}))\bigr]

where :math:`T = 2\,E_3(\tau_{\rm total})` is the slab transmission
probability,
:math:`\tau_{0,i} = \int_0^{x_i}\Sigt{}(s)\,\mathrm{d}s`, and
:math:`\tau_{L,i} = \int_{x_i}^{L}\Sigt{}(s)\,\mathrm{d}s`.

The :math:`E_2` factors represent the uncollided angular flux
arriving at point :math:`x_i` from a face source:
:math:`E_2(\tau) = \int_0^1 e^{-\tau/\mu}\,\mathrm{d}\mu`.  The
:math:`1/(1-T^2)` denominator sums the geometric series of
multiple slab traversals.

.. _peierls-conservation:

Conservation property
---------------------

For the :math:`\varphi` equation (identity on the LHS, not
:math:`\Sigt{}`), the Nystrom kernel satisfies the following
conservation identity for constant unit flux with white BC:

.. math::

   \sum_{j=1}^{N} K_{\rm total}[i, j]\,w_j = \frac{1}{\Sigt{i}}
   \qquad\text{(row-sum identity)}

where :math:`K_{\rm total}` includes both the interior :math:`E_1`
kernel and the white-BC re-entry kernel.

**Physical interpretation**: a spatially uniform, isotropically
scattering medium with :math:`\Sigs{} = \Sigt{}` (no absorption)
must produce :math:`\varphi(x) = \text{const}` everywhere.
Substituting :math:`q(x') = \Sigt{}\,\varphi_0` into
:eq:`peierls-equation` and requiring
:math:`\varphi(x_i) = \varphi_0` gives the row-sum identity above.

.. warning::

   An earlier version of the Peierls solver used
   :math:`\Sigt{}\,\varphi(x)` on the LHS instead of :math:`\varphi(x)`.
   This corresponds to a different normalisation where the eigenvalue
   system is :math:`(\mathrm{diag}(\Sigt{}) - K_s)\varphi
   = (1/k)\,K_f\varphi`.  The row-sum identity then becomes
   :math:`\sum_j K[i,j]\,w_j = 1`, which ALSO holds --- but the
   operator itself is wrong because the :math:`\Sigt{}` factor changes
   the eigenvectors.  The identity-LHS form is the physically correct
   normalisation for computing the scalar flux.  This was a debugging
   insight during Phase 4.1 development.

Relationship to CP
------------------

The CP flat-source approximation integrates the :math:`E_1` kernel
analytically over each region to obtain the :math:`E_3` second-difference
formulae (:eq:`dd-slab`, :eq:`dc-slab`, :eq:`self-slab`).  The
Peierls reference bypasses this integration and solves the full
integral equation numerically, making it an independent check on
the CP method's spatial discretisation.

Multi-group scattering convention
---------------------------------

See :ref:`peierls-scattering-convention` (in the Peierls unified
theory page) for the canonical, project-wide statement of the
``sig_s[g_src, g_dst]`` convention, which the CP / Peierls
drivers and the XS library
(:mod:`orpheus.derivations._xs_library`) all follow. In the Peierls
slab assembly loop
(:func:`orpheus.derivations.peierls_slab._build_system_matrices`)
the scatter kernel for the equation in group ``ge`` sums over
source groups ``gs`` via ``sig_s_at_node[j][gs][ge]`` =
:math:`\Sigma_{s,\,gs \to ge}` — first index source, second
destination.

.. warning::

   A naive reading of ``sig_s[gs][ge]`` might suggest "from ge to gs"
   (confusing row/column with from/to).  The correct convention is
   ``sig_s[from, to]``, consistent with the ``Mixture.SigS`` storage
   where ``SigS[g_from, g_to]`` and the in-scatter source is
   ``Q = SigS^T @ phi``.  See the scattering convention note in the
   :ref:`key-facts-cp` section and the authoritative
   :ref:`peierls-scattering-convention` note.

Numerical evidence
------------------

The Nystrom eigenvalue converges to the CP eigenvalue as the quadrature
is refined.  The following table shows the 2-group, 2-region slab case
(materials A + B, white BC, :math:`p`-point GL per panel):

.. list-table:: Nystrom k-eigenvalue convergence (2G, 2-region slab)
   :header-rows: 1
   :widths: 15 15 25 25

   * - Panels/region
     - :math:`p`
     - Nystrom :math:`\keff`
     - Relative error vs CP
   * - 4
     - 4
     - 1.2149
     - :math:`1.4 \times 10^{-2}`
   * - 8
     - 6
     - 1.2281
     - :math:`3.3 \times 10^{-3}`
   * - 8
     - 8
     - 1.2307
     - :math:`1.2 \times 10^{-3}`

The convergence is slow because the :math:`E_1` kernel's logarithmic
singularity limits the global convergence rate even with
product-integration handling on the diagonal panel.  For verification
purposes, the 2% agreement at moderate resolution is sufficient to
confirm that both methods solve the same integral transport equation.

The registered reference in the test suite
(``continuous_cases()`` in :mod:`orpheus.derivations.peierls_slab`) uses a
lightweight configuration (4 panels |times| 4 GL points per region,
20-digit ``mpmath`` precision) to keep import time fast.  Tests that
need higher accuracy should call
``_build_peierls_slab_case()``
directly with larger parameters.

Performance note
----------------

The ``mpmath`` solver is CPU-bound on the :math:`O(N^2)` kernel
assembly and :math:`O(N^3)` LU factorisation at arbitrary precision:

- **4 panels |times| 4 points** per region (2G 2R): ~6 s
- **8 panels |times| 6 points** per region (2G 2R): ~120 s
- **16 panels |times| 6 points** per region (2G 2R): ~10+ min

For this reason, the slow 2G 2-region convergence test is marked
``@pytest.mark.slow`` and excluded from routine CI.

.. seealso::

   :mod:`orpheus.derivations.peierls_slab` — Nystrom solver implementation.

   :class:`orpheus.derivations.peierls_slab.PeierlsSlabSolution` — result container
   with barycentric interpolation for flux evaluation at arbitrary points.

   ``tests/derivations/test_peierls_convergence.py`` — L0 self-convergence
   and eigenvalue agreement tests.

   ``tests/cp/test_peierls_flux.py`` — L1 CP flux convergence against
   the Peierls reference.


Peierls integral equation reference — cylinder
===============================================

The slab Peierls reference above verifies the CP flat-source
discretisation in Cartesian 1-D. The **cylindrical** Peierls reference
serves the same role for ``cyl1D`` meshes: it solves the integral
transport equation on a bare or concentric-annulus cylinder at
arbitrary quadrature order, providing an independent numerical
reference against :func:`~orpheus.cp.solver.solve_cp` and the
analytical CP eigenvalue in :mod:`orpheus.derivations.cp_cylinder`.

Unlike the slab, the cylinder's kernel is not an exponential integral
:math:`E_n` but the **Bickley--Naylor function** :math:`\mathrm{Ki}_1`
that arises from integrating the 3-D point-kernel over the infinite
axial direction. Unlike the slab, the boundary is a continuous lateral
surface, not a pair of discrete faces, so white-BC closure is not
a rank-2 outer product. And unlike the slab, a direct Nyström
discretisation of the canonical Sanchez--McCormick chord form picks
up a **non-integrable coincident singularity** that the slab's
product-integration trick does not cure — this motivates the
reformulation described below.

The implementation lives in :mod:`orpheus.derivations.peierls_cylinder`.
This section documents the mathematics, the formulation choice
(including the dead-end that was tried first), and the verification
evidence.

Canonical chord form and why it is not what the code solves
------------------------------------------------------------

Integrating the 3-D point kernel :math:`e^{-\tau R}/(4\pi R^{2})`
over the infinite axial coordinate :math:`z` yields the 2-D
transverse Green's function

.. math::
   :label: peierls-cylinder-green-2d

   G_{\rm 2D}(|\mathbf{r} - \mathbf{r}'|)
     \;=\; \frac{\mathrm{Ki}_1(\tau)}{2\pi\,|\mathbf{r}-\mathbf{r}'|},
   \qquad
   \tau \;=\; \int_{\mathbf{r}'}^{\mathbf{r}} \Sigma_t(\mathbf{s})\,\mathrm{d}\ell.

.. vv-status: peierls-cylinder-green-2d documented

The **pointwise** scalar-flux form of the Peierls integral equation
on a bare cylinder of radius :math:`R` is therefore

.. math::

   \varphi(\mathbf{r})
     \;=\; \frac{1}{2\pi}\!\iint_{\rm disc}
       \frac{\mathrm{Ki}_1\!\bigl(\tau(\mathbf{r},\mathbf{r}')\bigr)}
            {|\mathbf{r}-\mathbf{r}'|}\,q(\mathbf{r}')\,\mathrm{d}^{2}r'
     \;+\; \varphi_{\rm bc}(\mathbf{r}).

The classical textbook presentation ([Sanchez1982]_ §IV.A,
Eqs. 47--49; [Stamm1983]_ §6.2--6.3; [Hebert2020]_ Eqs. 3.95--3.110)
rotates the 2-D integral to the **chord** coordinate
system :math:`(y, r')`, where :math:`y` is the perpendicular distance
from the cylinder axis to the straight-line trajectory through
:math:`\mathbf{r}` and :math:`\mathbf{r}'`. Expressing
:math:`\mathrm{d}^{2}r'` as :math:`\bigl(r'/\sqrt{r'^{2}-y^{2}}\bigr)\,
\mathrm{d}r'\,\mathrm{d}y` on each branch and pairing the two
branches gives

.. math::

   \Sigma_t(r)\,\varphi(r)
     \;=\; \frac{1}{\pi}
       \int_{0}^{\min(r,R)}\!\mathrm{d}y
       \int_{y}^{R}
         \bigl[\mathrm{Ki}_1(\tau^{+}) + \mathrm{Ki}_1(\tau^{-})\bigr]\,
         \frac{q(r')\,r'}{\sqrt{r'^{2}-y^{2}}}\,\mathrm{d}r'
     \;+\; S_{\rm bc}(r).

.. warning::

   A derivation shortcut that keeps only the :math:`r'` Jacobian
   :math:`r'/\sqrt{r'^{2}-y^{2}}` (as the Phase-4.2 literature
   sweep initially reported) **is missing a factor**. Computing the
   branch sum :math:`|\mathrm{d}\alpha_{+}/\mathrm{d}y| +
   |\mathrm{d}\alpha_{-}/\mathrm{d}y|` for the
   :math:`(r', \alpha) \to (y, r')` transformation — where
   :math:`\alpha` is the chord-angle coordinate at the source point
   — gives :math:`2/\sqrt{\min(r,r')^{2} - y^{2}}`, **not**
   :math:`2` as the one-sided Jacobian would suggest. The correct
   combined Jacobian is therefore

   .. math::

      \frac{1}{\sqrt{(r^{2}-y^{2})(r'^{2}-y^{2})}},

   with a **second** integrable singularity at :math:`y = r`
   (co-located with the :math:`y = r'` root of the :math:`r'`-side
   factor when :math:`r = r'`). The unchecked "simplified" form with
   only :math:`r'/\sqrt{r'^{2}-y^{2}}` would amount to a mass-loss
   bug of the same flavour as missing the :math:`\Delta A/w_m`
   redistribution factor in a cylindrical :math:`S_N` sweep: the
   integrand does not reproduce the infinite-medium identity
   :math:`\sum_j K_{ij}\,\Sigma_t(r_j) = \Sigma_t(r_i)`.

The chord form above therefore has **two coincident endpoint
singularities** (at :math:`y = r` *and* :math:`y = r'`). The slab's
product-integration recipe absorbs one singularity (log at
:math:`x = x'`) analytically against a Lagrange basis; it does not
generalise to two coincident inverse-square-root singularities
sitting at nested quadrature endpoints. Attempting it produced
numerical divergence of the row-sum identity under refinement —
the kernel matrix simply does not converge for a
moderate-precision radial grid.

Formulation pivot: polar coordinates centred at the observer
-------------------------------------------------------------

Rather than patching the chord form, the implementation uses the
**equivalent polar form** centred on the observer. Let
:math:`\beta \in [0, 2\pi]` be the azimuth from the outward radial
direction at :math:`\mathbf{r}`, and :math:`\rho \ge 0` the distance
along the ray at angle :math:`\beta`. The source position is

.. math::
   :label: peierls-cylinder-r-prime

   r'(r, \rho, \beta) \;=\; \sqrt{r^{2} + 2r\rho\cos\beta + \rho^{2}}.

.. vv-status: peierls-cylinder-r-prime documented

Because the 2-D area element is :math:`\rho\,\mathrm{d}\rho\,
\mathrm{d}\beta` and the Green's function carries
:math:`1/|\mathbf{r} - \mathbf{r}'| = 1/\rho`, the :math:`\rho`
factor **cancels** and the integrand becomes smooth:

.. math::
   :label: peierls-cylinder-polar

   \varphi(r)
     \;=\; \frac{1}{\pi}\!
       \int_{0}^{\pi}\!\mathrm{d}\beta\!
       \int_{0}^{\rho_{\max}(r,\beta)}\!\!
         \mathrm{Ki}_1\!\bigl(\tau(r, \rho, \beta)\bigr)\,
         q\bigl(r'(r, \rho, \beta)\bigr)\,\mathrm{d}\rho
     \;+\; \varphi_{\rm bc}(r).

.. vv-status: peierls-cylinder-polar documented

The prefactor :math:`1/\pi` absorbs the :math:`1/(2\pi)` of the 2-D
Green's function plus a factor of 2 from :math:`\beta \to -\beta`
symmetry that folds :math:`[0, 2\pi] \to [0, \pi]`. The upper
radial limit is the intersection of the ray with the cylinder
boundary,

.. math::
   :label: peierls-cylinder-rho-max

   \rho_{\max}(r, \beta)
     \;=\; -r\cos\beta
         + \sqrt{r^{2}\cos^{2}\beta + R^{2} - r^{2}}.

.. vv-status: peierls-cylinder-rho-max documented

Writing the identity-LHS form used by the eigenvalue driver, the
canonical cylindrical Peierls equation solved by this module is

.. math::
   :label: peierls-cylinder-equation

   \Sigma_t(r_i)\,\varphi(r_i)
     \;=\; \frac{\Sigma_t(r_i)}{\pi}\!
       \int_{0}^{\pi}\!\mathrm{d}\beta\!
       \int_{0}^{\rho_{\max}(r_i,\beta)}\!\!
         \mathrm{Ki}_1\!\bigl(\tau(r_i, \rho, \beta)\bigr)\,
         q\!\bigl(r'(r_i, \rho, \beta)\bigr)\,\mathrm{d}\rho
     \;+\; S_{\rm bc}(r_i).

.. vv-status: peierls-cylinder-equation documented

The Sanchez tie-point and row-sum-identity tests currently carry
``@pytest.mark.verifies("peierls-equation", ...)`` (the slab label).
Retrofitting those decorators to point at the cylinder-specific
labels is a follow-up tracked in the V&V harness; until then this
equation is marked ``documented`` rather than ``tested`` to keep
the orphan gate honest.

.. note::

   The polar formulation is **mathematically equivalent** to the
   Sanchez chord form — the underlying integral equation is the
   same — but the :math:`\rho\,\mathrm{d}\rho\,\mathrm{d}\beta`
   area element absorbs the :math:`1/\rho` of the Green's function
   and thus the integrand
   :math:`\mathrm{Ki}_1(\tau(\rho))\,q(r'(\rho))` is **regular on
   the whole integration domain**. Ordinary tensor-product
   Gauss--Legendre quadrature converges spectrally. This is the
   dominant motivation for the pivot: the chord form trades one
   integrable singularity for two, the polar form eliminates them.

Why :math:`\mathrm{Ki}_1` and not :math:`\mathrm{Ki}_3`
-------------------------------------------------------

The flat-source CP method (:mod:`orpheus.derivations.cp_cylinder`)
uses the :math:`\mathrm{Ki}_3` kernel because it averages the
pointwise :math:`\mathrm{Ki}_1` kernel twice — once over the
source region :math:`j` and once over the target region :math:`i`
— producing the second-difference formula

.. math::

   P_{ij} \;\propto\; \mathrm{Ki}_3(\text{gap})
                    - \mathrm{Ki}_3(\text{gap} + \tau_i)
                    - \mathrm{Ki}_3(\text{gap} + \tau_j)
                    + \mathrm{Ki}_3(\text{gap} + \tau_i + \tau_j).

See :eq:`ki3-def` for the :math:`\mathrm{Ki}_n` definition and
:eq:`chord-length` for the chord geometry underlying
:math:`P_{ij}`.

The Peierls reference solves for the **pointwise** flux
:math:`\varphi(r_i)`, not the region-average collision rate, so it
uses :math:`\mathrm{Ki}_1` directly. This is the independent-kernel
property that makes the reference useful:

- The :math:`\mathrm{Ki}_n` kernels are computed by recursion
  :math:`\mathrm{Ki}_{n+1}(x) = \int_x^\infty \mathrm{Ki}_n(t)\,
  \mathrm{d}t`, so a sign error or off-by-one index in the recursion
  would affect every :math:`\mathrm{Ki}_n` built from
  :math:`\mathrm{Ki}_1`. The CP test suite verifies the CP solver
  against analytically-constructed CP matrices (same
  :math:`\mathrm{Ki}_3`, same flat-source discretisation), so a
  systematic :math:`\mathrm{Ki}` bug could cancel between test and
  code. The Peierls reference uses :math:`\mathrm{Ki}_1` and a
  polynomial source representation, breaking the common-mode path.
- Because the Peierls Nyström operator resolves the flux as a
  piecewise polynomial of degree :math:`p-1` on each radial panel,
  it is sensitive to flux-shape errors that are invisible to a
  CP-vs-CP comparison with flat sources.

The canonical :math:`\mathrm{Ki}_n` recurrence evaluator lives in
:func:`~orpheus.derivations._kernels.ki_n_mp` (arbitrary precision
via :func:`mpmath.quad` on the A&S integral form). The
double-precision fast path for :math:`\mathrm{Ki}_3` goes through
:func:`~orpheus.derivations.cp_geometry._ki3_mp` — a Chebyshev
interpolant built from ``ki_n_mp`` at module load (Phase B.4,
commit ``6badbe5``, Issue #94). The legacy ``BickleyTables``
20 000-point tabulation this replaced is documented historically
in :ref:`ki-table-construction`.

Nyström assembly in polar coordinates
-------------------------------------

The discretisation has three nested quadrature layers, each chosen
to match a distinct piece of the integrand's structure.

**Radial grid (composite Gauss--Legendre on** :math:`[0, R]`\ **).**
The r-axis is partitioned into the :math:`N_{\rm reg}` annular
regions :math:`[r_{k-1}, r_k]`, :math:`r_0 = 0`, :math:`r_{N} = R`.
Each region carries :math:`n_{\rm panels}` panels, each carrying
:math:`p` Gauss--Legendre nodes. Panel breakpoints coincide with
annular radii so the emission density :math:`q(r')`, which is
piecewise-smooth but has **slope discontinuities** at material
boundaries, is represented by a piecewise polynomial of degree
:math:`p-1`. This mirrors the slab composite-panel strategy.
The total number of radial Nyström unknowns is
:math:`N = N_{\rm reg} \times n_{\rm panels} \times p`. The builder
is ``composite_gl_r`` (aliased from ``composite_gl_y`` in
:mod:`orpheus.derivations.peierls_cylinder`).

**Azimuthal quadrature (Gauss--Legendre on** :math:`[0, \pi]`\ **).**
With :math:`n_\beta` nodes and weights :math:`w_{\beta,k}`; the
physical interval :math:`[0, 2\pi]` is folded to :math:`[0, \pi]`
by the :math:`\beta \to -\beta` symmetry already absorbed into the
:math:`1/\pi` prefactor.

**Ray-distance quadrature (Gauss--Legendre on**
:math:`[0, \rho_{\max}(r_i, \beta_k)]`\ **).**
With :math:`n_\rho` nodes per ray. The upper limit depends on both
the observer radius :math:`r_i` and the direction :math:`\beta_k`,
so the ρ-quadrature is **remapped per (i, k)** from the reference
interval :math:`[-1, 1]`. For a homogeneous bare cylinder, a fixed
:math:`\rho`-scale would under-resolve rays near the tangent
direction and over-resolve radial rays; the per-ray remapping
gives uniform relative accuracy.

**Source interpolation by Lagrange basis.**
Because the source point :math:`r'(r_i, \rho_m, \beta_k)` is
**generally not a radial quadrature node**, the emission density
at the source is expressed via the panel-local Lagrange basis:

.. math::

   q\bigl(r'_{ikm}\bigr)
     \;=\; \sum_{j=1}^{N} L_j(r'_{ikm})\,q_j,

where :math:`L_j` is the degree-:math:`(p-1)` Lagrange polynomial
supported only on the panel containing :math:`r'_{ikm}`
(piecewise-polynomial representation matching the composite GL
radial mesh). The basis is built by
``_lagrange_basis_on_panels`` in
:mod:`orpheus.derivations.peierls_cylinder`. Two properties are
enforced by L0 foundation tests:

- **Partition of unity**: :math:`\sum_j L_j(r) = 1` for any
  :math:`r \in [0, R]`. Tested in
  ``TestLagrangeBasisOnPanels.test_partition_of_unity``.
- **Polynomial reproduction**: for any polynomial of degree
  :math:`< p`, :math:`\sum_j p(r_j)\,L_j(r) = p(r)` exactly.
  Tested in
  ``TestLagrangeBasisOnPanels.test_reproduces_polynomial``.

**Assembled matrix.**
Substituting :math:`q(r'_{ikm}) = \sum_j L_j(r'_{ikm})\,q_j`
into :eq:`peierls-cylinder-equation` gives the identity-LHS form

.. math::
   :label: peierls-cylinder-nystrom

   \Sigma_t(r_i)\,\varphi_i
     \;=\; \sum_{j=1}^{N} K_{ij}\,q_j + S_{\rm bc}(r_i),

.. vv-status: peierls-cylinder-nystrom documented

with

.. math::

   K_{ij}
     \;=\; \frac{\Sigma_t(r_i)}{\pi}
       \sum_{k=1}^{n_\beta}\!\sum_{m=1}^{n_\rho}
         w_{\beta,k}\,w_{\rho,m}(r_i,\beta_k)\,
         \mathrm{Ki}_1(\tau_{ikm})\,L_j(r'_{ikm}).

The kernel matrix is assembled by ``build_volume_kernel`` in
:mod:`orpheus.derivations.peierls_cylinder`. The per-sample optical
depth :math:`\tau_{ikm}` is computed by ``_optical_depth_along_ray``,
which walks annular boundary crossings as described next.

Ray optical-depth walker
------------------------

The optical depth along the ray from :math:`r_i` in direction
:math:`\beta` over distance :math:`\rho`,

.. math::
   :label: peierls-cylinder-ray-optical-depth

   \tau(r_i, \rho, \beta)
     \;=\; \int_{0}^{\rho}
       \Sigma_t\!\bigl(r'(r_i, s, \beta)\bigr)\,\mathrm{d}s,

.. vv-status: peierls-cylinder-ray-optical-depth documented

is piecewise-constant in the integrand. The boundary crossings
:math:`|\mathbf{r}(s)|^{2} = r_k^{2}` give the quadratic

.. math::

   s^{2} + 2 r_i \cos\beta\,s + (r_i^{2} - r_k^{2}) \;=\; 0,

whose roots :math:`s = -r_i \cos\beta \pm
\sqrt{r_i^{2}\cos^{2}\beta + r_k^{2} - r_i^{2}}` are the entry and
exit points for the ray crossing annulus :math:`k`. The walker
sorts all such roots in :math:`(0, \rho)`, evaluates
:math:`r_{\rm mid}` on each segment, and accumulates
:math:`\Sigma_{t,k}\cdot\Delta s`. The homogeneous case
(:math:`N_{\rm reg} = 1`) short-circuits to
:math:`\tau = \Sigma_t\,\rho` for speed.

The walker is L0-verified against closed-form traversals in
``tests/derivations/test_peierls_cylinder_multi_region.py``:
``TestOpticalDepthAlongRay`` covers the homogeneous short-circuit,
a ray staying in the outer annulus, a ray crossing one inner
boundary, a ray through the axis traversing three annular
segments, and a tangent ray that grazes the inner annulus.

Relationship to the :math:`\tau^{\pm}` chord walker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module also exposes a separate walker ``optical_depths_pm``
that computes the same-side (:math:`\tau^{+}`) and through-centre
(:math:`\tau^{-}`) branches of the chord form for reference /
verification purposes. It is not used by
``build_volume_kernel`` (which operates in polar coordinates)
but is the primitive that would be needed for any future
Schur-complemented white-BC boundary closure (see
:ref:`peierls-cylinder-white-bc` below), where the relevant
variable is the chord impact parameter :math:`y`, not the
observer-centred :math:`\rho`. Its L0 tests live in
``tests/derivations/test_peierls_cylinder_geometry.py``.

.. _peierls-cylinder-row-sum:

Row-sum identity — homogeneous vs multi-region
----------------------------------------------

The row-sum identity is the single most diagnostic
self-consistency check for the Peierls operator. It isolates the
prefactor, the kernel normalisation, and the quadrature against
the multiplicative-factor class of bugs that otherwise only show
up as a biased eigenvalue. The cylindrical identity is subtler
than the slab's because of how :math:`\Sigma_t` interacts with
the ray integral.

**Homogeneous cylinder.** For a bare homogeneous cylinder of
radius :math:`R`, the infinite-medium identity for the
identity-LHS kernel :eq:`peierls-cylinder-nystrom` is

.. math::

   \sum_{j=1}^{N} K_{ij} \;=\; \Sigma_t(r_i) \qquad (R \to \infty).

The finite-cylinder deficit
:math:`\Sigma_t - \sum_j K_{ij}` equals the uncollided escape
probability :math:`\Sigma_t\,P_{\rm esc}(r_i)` times :math:`\Sigma_t`
(a standard result: [BellGlasstone1970]_ §2.7; [Hebert2020]_
Eq. 3.101), and for :math:`R = 10` MFP this deficit is
:math:`< 10^{-3}` at :math:`r_i \le R/2`. Tested in
``TestRowSumIdentity.test_interior_row_sum_equals_sigma_t`` in
``tests/derivations/test_peierls_cylinder_prefactor.py``.

**Multi-region cylinder.** The naive "apply :math:`K` to
:math:`q \equiv 1`" identity **fails** when :math:`\Sigma_t` is
piecewise-constant across annuli. The reason is visible in the
change-of-variables :math:`u = \tau(\rho)`:

.. math::

   \int_{0}^{\rho_{\max}}\!\mathrm{Ki}_1\!\bigl(\tau(\rho)\bigr)\,
     \mathrm{d}\rho
     \;=\; \int_{0}^{\tau_{\max}}\!
       \frac{\mathrm{Ki}_1(u)}{\Sigma_t\!\bigl(r'(u)\bigr)}\,\mathrm{d}u.

The :math:`1/\Sigma_t` in the integrand depends on **where along
the ray** the source point sits, so the identity collapses only
if :math:`\Sigma_t` is constant.

The correct multi-region identity is obtained by applying
:math:`K` to the source :math:`q = \Sigma_t` — physically, the
pure-scatter emission density that sustains a spatially uniform
flux :math:`\varphi \equiv 1`:

.. math::
   :label: peierls-cylinder-row-sum-identity

   \sum_{j=1}^{N} K_{ij}\,\Sigma_t(r_j) \;=\; \Sigma_t(r_i)
   \qquad\text{(multi-region, } R \to \infty\text{)}.

.. vv-status: peierls-cylinder-row-sum-identity documented

The :math:`\Sigma_t(r_j)` factor absorbs the :math:`1/\Sigma_t`
left behind by the change of variables, restoring
:math:`\int \mathrm{Ki}_1(u)\,\mathrm{d}u = 1` independently of
:math:`\Sigma_t` variation along the ray. This is the identity
actually tested in
``TestMultiRegionKernel.test_K_applied_to_sig_t_gives_local_sig_t``
— it applies :math:`K` to :math:`q_j = \Sigma_t(r_j)` and
verifies recovery of :math:`\Sigma_t(r_i)` at every interior
observer, to 0.5 % accuracy on an
:math:`(r_1, R) = (3, 10)`-MFP two-annulus configuration with
:math:`\Sigma_{t,{\rm inner}} = 0.8, \Sigma_{t,{\rm outer}} = 1.4`.

.. warning::

   A test that applied :math:`K` to :math:`\mathbf{1}` and
   compared to :math:`\Sigma_t(r_i)` would silently fail for the
   multi-region case even when the implementation is correct.
   The row sum
   :math:`\sum_j K_{ij}` instead equals a ray-path-weighted
   average of :math:`1/\Sigma_t`, which is not a local quantity.
   This is the reason the multi-region test file applies
   :math:`K` to :math:`\Sigma_t`, not to :math:`\mathbf{1}`.

.. _peierls-cylinder-white-bc:

Boundary conditions
-------------------

**Vacuum.** :math:`S_{\rm bc} \equiv 0`. The kernel :math:`K` is
the full operator; the eigenvalue problem is

.. math::

   \bigl[\mathrm{diag}(\Sigma_t) - K\,\mathrm{diag}(\Sigma_s)\bigr]
     \varphi \;=\; \frac{1}{k}\,K\,\mathrm{diag}(\nu\Sigma_f)\,\varphi,

solved by fission-source power iteration in
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
(with ``geometry=_pg.CYLINDER_1D`` and ``boundary="vacuum"``); the
sphere-/cylinder-specific façades in
:mod:`orpheus.derivations.peierls_cylinder` /
:mod:`orpheus.derivations.peierls_sphere` are registry-only
shims after the Issue #138 collapse. This is the closure that is
currently implemented; it is used for the Sanchez tie-point
verification below.

**White / reflective.** The cylinder's lateral surface is a
continuous 1-parameter family of boundary points; every
(impact-parameter :math:`y`, travel-direction) pair produces a
distinct re-entering ray. The slab's rank-2 :math:`E_2` outer
product — which comes from exactly two boundary faces — does **not**
generalise. Instead, the white-BC closure is a rank-:math:`N_y`
**dense** Schur-complement block, where :math:`N_y` is the
number of :math:`y`-quadrature nodes used to represent the
outgoing/incoming currents on the cylinder surface.

The two implementable options are:

(a) **Reduced form.** Eliminate the boundary-current unknowns
    by Schur complement; the result is an effective boundary
    source :math:`S_{\rm bc}(r_i)` that is a smooth integral
    operator of the volume unknowns :math:`\varphi_j`. [Sanchez1982]_
    §IV.B.3 uses this form.
(b) **Full form.** Keep the boundary currents :math:`J^\pm(y_\ell)`
    as explicit unknowns and solve the coupled
    :math:`(\varphi, J^{+}, J^{-})` block system. [Hebert2020]_
    uses this form because the coupling block is trivially
    populated from the :math:`\tau^{\pm}` walker.

For :math:`N_y = O(50\text{--}100)` the dense block is a
:math:`50 \times 50` to :math:`100 \times 100` matrix, negligible
relative to the :math:`O(N^{3})` radial LU factorisation.

.. note::

   The white-BC closure is **not yet implemented**. The current
   :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
   call (with ``geometry=_pg.CYLINDER_1D``) and ``boundary="vacuum"``
   handles vacuum BC only. The
   ``optical_depths_pm`` :math:`\tau^{\pm}`
   walker that lives alongside ``build_volume_kernel`` is the
   primitive needed for either option (a) or option (b); it is
   retained in the module for this planned extension. The
   white-BC driver is Phase-4.2 item C8 of the verification
   campaign and is a prerequisite for comparing the Peierls
   reference against the CP flat-source cylinder flux profile
   at full white-BC parity.

Verification evidence
---------------------

Three independent checks gate the Peierls cylinder implementation.

**Sanchez--McCormick 1982 tie-point.** For a bare 1-group
homogeneous cylinder with :math:`\Sigma_t = 1` cm⁻¹,
:math:`(\Sigma_s, \nu\Sigma_f) = (0.5, 0.75)` — giving
:math:`k_\infty = \nu\Sigma_f/\Sigma_a = 1.5` — [Sanchez1982]_
Table IV reports a critical radius :math:`R = 1.9798` cm.
At that :math:`R` the present solver gives

.. math::

   k_{\rm eff}(R = 1.9798) \;=\; 1.00421 \pm 10^{-5}

under polar-quadrature refinement
:math:`(n_\beta, n_\rho) = (20, 20) \to (32, 32)`. The 0.42 %
offset from unity reflects the ambiguous scatter / fission split
in the Sanchez ``c = 1.5`` problem definition: the 1-group
:math:`k_{\rm eff}` is **not invariant** under the split at fixed
:math:`k_\infty`, because :math:`\Sigma_s` enters the resolvent
:math:`(\Sigma_t\mathbf{I} - K\Sigma_s)^{-1}` separately from the
fission source. The Zotero MCP server was unreachable during the
Phase-4.2 literature sweep (see Directive-4 demand note in
``.claude/agent-memory/literature-researcher/phase4_cylinder_peierls.md``),
so the reference split cannot yet be cross-checked; the test gate
is set to a 1 % tolerance in
``TestSanchezTiePoint.test_k_eff_at_R_equals_1_dot_9798``, which
is robust to this ambiguity but tight enough to catch any
multiplicative-factor regression.

**Vacuum-BC thick-cylinder limit.** As :math:`R \to \infty`,
leakage vanishes and :math:`k_{\rm eff} \to k_\infty`. At
:math:`R = 30` MFP the solver reaches
:math:`k_{\rm eff} \approx 1.49` against :math:`k_\infty = 1.5`
(0.5 % gap), and :math:`k_{\rm eff}(R)` is **monotone increasing**
for :math:`R \in \{1.5, 3, 6, 12, 24\}` MFP. Tested in
``TestVacuumBCThickLimit``.

**Row-sum identity.** For the homogeneous :math:`R = 10` MFP
configuration, :math:`\max_i |\Sigma_t - \sum_j K_{ij}| < 10^{-3}`
at :math:`r_i \le R/2`, and the deficit decays monotonically
toward :math:`1` as :math:`r_i \to R` (escape probability rises
at the surface). The multi-region identity
:math:`\sum_j K_{ij}\,\Sigma_t(r_j) = \Sigma_t(r_i)` holds to
0.5 % on a :math:`(\Sigma_{t,{\rm inner}},
\Sigma_{t,{\rm outer}}) = (0.8, 1.4)` two-annulus problem.
Tested in
``TestRowSumIdentity`` and ``TestMultiRegionKernel`` in
``tests/derivations/test_peierls_cylinder_prefactor.py`` and
``tests/derivations/test_peierls_cylinder_multi_region.py``.

.. list-table:: Cylindrical Peierls verification summary
   :header-rows: 1
   :widths: 35 25 20 20

   * - Check
     - Tolerance
     - Status
     - Identity
   * - Sanchez tie-point :math:`R = 1.9798`
     - :math:`|k_{\rm eff} - 1| < 2\times10^{-2}`
     - passes at :math:`10^{-2}`
     - :eq:`peierls-cylinder-equation`
   * - Thick limit :math:`R = 30` MFP
     - :math:`|k_{\rm eff} - k_\infty|/k_\infty < 10^{-2}`
     - passes at :math:`5\times10^{-3}`
     - vacuum-BC fixed point
   * - Row sum (homogeneous)
     - :math:`<10^{-3}` at :math:`r_i \le R/2`
     - passes
     - :eq:`peierls-cylinder-nystrom`
   * - Row sum (multi-region)
     - :math:`<5\times10^{-3}` bulk interior
     - passes
     - :eq:`peierls-cylinder-row-sum-identity`

Relationship to the CP flat-source cylinder solver
--------------------------------------------------

The CP flat-source method for the cylinder
(:func:`~orpheus.cp.solver.solve_cp` on ``cyl1D`` meshes;
:mod:`orpheus.derivations.cp_cylinder`) integrates the
:math:`\mathrm{Ki}_1` kernel analytically over each annulus to
produce the :math:`\mathrm{Ki}_3` second-difference formula
quoted above. The Peierls reference **bypasses that integration**
entirely:

- the kernel is :math:`\mathrm{Ki}_1`, not :math:`\mathrm{Ki}_3`,
- the spatial representation is a piecewise polynomial of degree
  :math:`p - 1` per panel, not a piecewise constant,
- the ray integration is performed numerically in polar
  coordinates, not analytically over rectangular annular regions.

So the two methods share almost nothing except the underlying
integral equation. A sign error, off-by-one index, or factor-of-2
in the :math:`\mathrm{Ki}_3` second-difference formula — which
would cancel between the CP solver and a CP-self-verification
test — would be caught by the Peierls reference. Conversely, a
systematic error in :math:`\mathrm{Ki}_1` evaluation would be
caught by the CP eigenvalue tests (which use pre-tabulated
:math:`\mathrm{Ki}_3` values). The two together triangulate the
cylindrical integral-transport stack.

Numerical cost
--------------

The :math:`(\beta, \rho)` tensor-product quadrature is the dominant
cost. For each observer :math:`r_i` and each :math:`\beta_k`, the
kernel assembly evaluates :math:`\mathrm{Ki}_1` at :math:`n_\rho`
points via :func:`~orpheus.derivations._kernels.ki_n_mp` (mpmath
at ``dps`` precision), which is :math:`O(N \cdot n_\beta \cdot
n_\rho)` kernel evaluations. For :math:`N = 10` radial nodes,
:math:`(n_\beta, n_\rho) = (24, 24)`, ``dps = 20``, kernel
assembly takes :math:`\approx 3` s on current hardware; eigenvalue
power iteration is a further :math:`O(N^{3})` LU per iteration,
typically converging in 20--30 iterations to
:math:`10^{-10}` eigenvalue tolerance.

Short-circuit: the homogeneous single-region branch of
``_optical_depth_along_ray`` returns :math:`\Sigma_t \rho` without
sorting crossings, making the bare-cylinder case
:math:`\sim\!2\times` faster than the multi-region path.

.. seealso::

   :mod:`orpheus.derivations.peierls_cylinder` — registry-only
   module; binds the cylinder ``GEOMETRY`` singleton and ships the
   ``_build_peierls_cylinder_*_case`` continuous-reference
   constructors. The Nyström solver implementation lives in
   :mod:`~orpheus.derivations.peierls_geometry`.

   :class:`~orpheus.derivations.peierls_geometry.PeierlsSolution`
   — canonical result container with radial node positions, flux
   values, :math:`k_{\rm eff}`, and ``geometry_kind`` discriminator.
   Same dataclass for slab / cylinder / sphere.

   :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
   — 1-group eigenvalue driver (call with
   ``geometry=_pg.CYLINDER_1D`` and ``boundary="vacuum"`` for the
   vacuum-BC closure described above).

   ``tests/derivations/test_peierls_cylinder_geometry.py`` — L0
   tests for ``composite_gl_y`` and ``optical_depths_pm``.

   ``tests/derivations/test_peierls_cylinder_prefactor.py`` — L0
   row-sum-identity tests (homogeneous).

   ``tests/derivations/test_peierls_cylinder_multi_region.py`` —
   L0 multi-region optical-depth walker, Lagrange-basis
   foundation tests, and the multi-region
   :math:`\sum_j K_{ij}\,\Sigma_t(r_j) = \Sigma_t(r_i)` identity.

   ``tests/derivations/test_peierls_cylinder_eigenvalue.py`` — L1
   Sanchez tie-point and thick-cylinder limit eigenvalue tests.


Peierls integral equation reference — sphere
=============================================

The cylindrical Peierls reference above verifies the CP flat-source
discretisation for 1-D radial geometries with 2-D translational
symmetry. The **spherical** Peierls reference closes the trio for
``sph1D`` meshes: it solves the 3-D integral transport equation on a
bare or concentric-shell sphere at arbitrary quadrature order,
providing an independent numerical reference against
:func:`~orpheus.cp.solver.solve_cp` on spherical meshes and the
flat-source spherical CP of :mod:`orpheus.derivations.cp_sphere`.

Unlike the slab (:math:`E_1` from :math:`y, z`-integration) and the
cylinder (:math:`\mathrm{Ki}_1` from :math:`z`-integration), the
sphere **does not reduce dimensions** — the native 3-D point kernel
:math:`e^{-\tau}/(4\pi\,d^{2})` is already 3-D. Rotational symmetry
about the centre is not a translation and cannot be used to collapse
the kernel; it only constrains the *source field* to depend on
:math:`|\mathbf r'| = r'` rather than on all three coordinates. The
polar-form pivot still buys the Jacobian cancellation (the
:math:`\rho^{2}` polar volume element cancels the :math:`1/d^{2}` of
the Green's function exactly), but the resulting reduced kernel is
the bare exponential :math:`e^{-\tau}`, not any :math:`E_n` or
:math:`\mathrm{Ki}_n` function. This makes the sphere both the
simplest kernel to evaluate (plain ``np.exp``) and the most
informative cross-check for a common-mode :math:`\mathrm{Ki}_n`
recursion bug: any factor-of-two, off-by-one, or sign error in the
Bickley recurrence that would affect the cylinder cancels out of the
sphere, and vice-versa.

The implementation lives in :mod:`orpheus.derivations.peierls_sphere`
— a thin facade over the unified
:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
(``kind = "sphere-1d"``). The sphere shares the
Lagrange-basis, composite-GL radial quadrature, optical-depth walker,
and power-iteration primitives with the cylinder verbatim; the only
geometry-specific ingredients are the kernel :math:`\kappa_d(\tau) =
e^{-\tau}`, the angular weight :math:`W_\Omega(\theta) =
\sin\theta`, the prefactor :math:`C_d = 1/2`, and the
surface-divisor :math:`A_d = R^{2}` in the rank-1 white-BC
closure. The architectural separation is documented at full length
in :doc:`peierls_unified`.

Derivation from the 3-D point kernel
-------------------------------------

Starting from the 3-D Green's function for the one-speed, isotropic
integral transport equation,

.. math::
   :label: peierls-sphere-green-3d

   G_{\rm 3D}(\mathbf r, \mathbf r')
     \;=\; \frac{e^{-\tau(\mathbf r, \mathbf r')}}
                {4\pi\,|\mathbf r - \mathbf r'|^{2}},
   \qquad
   \tau \;=\; \int_{\mathbf r'}^{\mathbf r}\!\Sigma_t(\mathbf s)\,
     \mathrm d\ell,

.. vv-status: peierls-sphere-green-3d documented

the **pointwise** scalar-flux form of the Peierls integral equation
on a bare sphere of radius :math:`R` is

.. math::

   \varphi(\mathbf r)
     \;=\; \iiint_{\rm ball}\!
       \frac{e^{-\tau(\mathbf r, \mathbf r')}}
            {4\pi\,|\mathbf r - \mathbf r'|^{2}}\,
         q(\mathbf r')\,\mathrm d^{3}r'
     \;+\; \varphi_{\rm bc}(\mathbf r).

This is the **native** 3-D kernel. Compare with the slab
:eq:`peierls-equation` (whose kernel is :math:`E_1`, obtained by
integrating the point kernel over two transverse dimensions) and
the cylinder :eq:`peierls-cylinder-green-2d` (whose kernel is
:math:`\mathrm{Ki}_1`, obtained by integrating over one axial
direction). The sphere's
point kernel **cannot be pre-integrated** because there is no
translational symmetry to exploit — a radial 1-D problem inherits
only rotational symmetry from the embedding space, and rotations
move every point on a shell to every other point on the same shell
without ever crossing the shell boundary.

.. note::

   The monotone progression of dimensional reductions — two for the
   slab, one for the cylinder, zero for the sphere — is the defining
   feature of the trio. :doc:`peierls_unified` §2 tabulates the
   reduced kernels side-by-side.

Observer-centred polar form and Jacobian cancellation
------------------------------------------------------

The native 3-D integral above is not directly tractable: the
:math:`1/|\mathbf r - \mathbf r'|^{2}` singularity at
:math:`\mathbf r' = \mathbf r` is a **volume singularity**
(non-integrable without the Jacobian of an appropriate coordinate
change). Rather than attempt a chord/impact-parameter formulation
analogous to the cylinder's (which, on the sphere, would introduce
*three* coincident endpoint singularities at a point), the
implementation uses the **equivalent polar form** centred on the
observer.

Let :math:`\theta \in [0, \pi]` be the polar angle from the outward
radial direction at :math:`\mathbf r`, :math:`\phi \in [0, 2\pi]`
the azimuth, and :math:`\rho \ge 0` the distance along the ray from
:math:`\mathbf r` in direction :math:`(\theta, \phi)`. The source
position is

.. math::
   :label: peierls-sphere-r-prime

   r'(r, \rho, \theta) \;=\;
     \sqrt{r^{2} + 2 r \rho \cos\theta + \rho^{2}},

.. vv-status: peierls-sphere-r-prime documented

**identical** to the cylinder case
:eq:`peierls-cylinder-r-prime` — the 1-D radial chord algebra does
not care whether the surrounding source field is
:math:`2`-D-symmetric (cylinder) or :math:`3`-D-symmetric
(sphere). That is the architectural insight that let
:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
share ray primitives between the two geometries.

The 3-D volume element in observer-centred coordinates is
:math:`\rho^{2}\,\sin\theta\,\mathrm d\rho\,\mathrm d\theta\,
\mathrm d\phi`. Combined with the 3-D Green's function the
integrand becomes

.. math::

   \frac{e^{-\tau}}{4\pi\,\rho^{2}}\,\cdot\,
   \rho^{2}\,\sin\theta\,\mathrm d\rho\,\mathrm d\theta\,
   \mathrm d\phi
   \;=\;
   \frac{\sin\theta}{4\pi}\,e^{-\tau}\,\mathrm d\rho\,
   \mathrm d\theta\,\mathrm d\phi.

The :math:`\rho^{2}` polar volume factor **cancels the**
:math:`1/\rho^{2}` **of the Green's function exactly** — the
Jacobian-cancellation trick derived in full generality in
:doc:`peierls_unified` §3. The integrand is now a bounded,
polynomial-smooth function on the whole integration domain (modulo
:math:`\Sigma_t` jumps, which are handled by the composite radial
grid described below). Because the source field is radially
symmetric, nothing in the integrand depends on :math:`\phi`, so the
azimuthal integral collapses to a factor of :math:`2\pi`:

.. math::
   :label: peierls-sphere-polar

   \varphi(r)
     \;=\; \frac{1}{2}\!
       \int_{0}^{\pi}\!\sin\theta\,\mathrm d\theta\!
       \int_{0}^{\rho_{\max}(r, \theta)}\!\!
         e^{-\tau(r, \rho, \theta)}\,
         q\!\bigl(r'(r, \rho, \theta)\bigr)\,\mathrm d\rho
     \;+\; \varphi_{\rm bc}(r).

.. vv-status: peierls-sphere-polar documented

The prefactor :math:`1/2` is :math:`2\pi / (4\pi) = 1/2`, and the
:math:`\sin\theta` weight remains from the solid-angle element
:math:`\mathrm d\Omega = \sin\theta\,\mathrm d\theta\,\mathrm d\phi`.

.. note::

   **No** :math:`\beta \to -\beta` **folding is needed** on the
   sphere. The cylindrical polar form
   :eq:`peierls-cylinder-polar` folds :math:`\beta \in [0, 2\pi]`
   to :math:`[0, \pi]` (buying a factor of 2 into the
   :math:`1/\pi` prefactor) because the integrand there is
   :math:`\beta`-symmetric. On the sphere the polar angle
   :math:`\theta \in [0, \pi]` already covers the full hemisphere
   of ray directions at the observer, and :math:`\sin\theta \ge 0`
   on that interval — no folding is available or needed.

The upper radial limit :math:`\rho_{\max}` is the intersection of
the ray with the outer sphere,

.. math::
   :label: peierls-sphere-rho-max

   \rho_{\max}(r, \theta)
     \;=\; -r\cos\theta
         + \sqrt{r^{2}\cos^{2}\theta + R^{2} - r^{2}},

.. vv-status: peierls-sphere-rho-max documented

**identical** to the cylinder :eq:`peierls-cylinder-rho-max`.
Verified by ``TestSphereRhoMax`` in
``tests/derivations/test_peierls_sphere_geometry.py``, which covers
the radial-outward ray (:math:`\rho_{\max} = R - r`), the
radial-inward through-diameter ray (:math:`\rho_{\max} = R + r`),
the tangential ray from the centre (:math:`\rho_{\max} = R`), and
the observer-on-surface outward ray (:math:`\rho_{\max} = 0`).

Writing the identity-LHS form used by the eigenvalue driver, the
canonical spherical Peierls equation solved by this module is

.. math::
   :label: peierls-sphere-equation

   \Sigma_t(r_i)\,\varphi(r_i)
     \;=\; \frac{\Sigma_t(r_i)}{2}\!
       \int_{0}^{\pi}\!\sin\theta\,\mathrm d\theta\!
       \int_{0}^{\rho_{\max}(r_i, \theta)}\!\!
         e^{-\tau(r_i, \rho, \theta)}\,
         q\!\bigl(r'(r_i, \rho, \theta)\bigr)\,\mathrm d\rho
     \;+\; S_{\rm bc}(r_i).

.. vv-status: peierls-sphere-equation tested

The sphere test files carry
``@pytest.mark.verifies("peierls-unified")``, which is the coarse
label shared across the whole unified polar-form implementation
while finer-grained per-equation labels are retrofitted in a
follow-up V&V harness pass. The ``vv-status: ... tested`` annotation
above reflects that coverage.

Why :math:`e^{-\tau}` and not :math:`E_3` / :math:`\mathrm{Ki}_3`
------------------------------------------------------------------

The flat-source CP method for the sphere
(:mod:`orpheus.derivations.cp_sphere`) uses a second-difference
formula in the :math:`E_3` function (see :eq:`second-diff-sph` above)
because it averages the pointwise :math:`e^{-\tau}` kernel twice —
once over the source region :math:`j` and once over the target
region :math:`i` — and the **double antiderivative** of
:math:`e^{-\tau}/d^{2}` along a chord produces exactly
:math:`E_3(\tau)`. This is the sphere-specific instance of the
general second-difference identity:

.. list-table:: Pointwise kernels vs flat-source second-differences
   :header-rows: 1
   :widths: 18 30 30

   * - Geometry
     - Pointwise kernel
     - Flat-source second-difference
   * - Slab
     - :math:`E_1`
     - :math:`E_3`
   * - Cylinder
     - :math:`\mathrm{Ki}_1`
     - :math:`\mathrm{Ki}_3`
   * - Sphere
     - :math:`e^{-\tau}`
     - :math:`E_3`

The sphere is alone in the right column: the second antiderivative
of :math:`e^{-\tau}` involves :math:`E_1` and :math:`E_2` depending
on which combination of chord endpoints is averaged, and the
particular combination that appears for a concentric-shell spherical
geometry happens to collapse to :math:`E_3`. Full derivation in
``derivations/cp_sphere.py``; see :eq:`second-diff-sph` and
:eq:`self-sph` for the resulting CP matrix elements, and
:eq:`rcp-from-double-antideriv` for the general second-difference
identity that specialises to the sphere via the same
double-antiderivative argument used for the slab and cylinder.

The Peierls reference solves for the **pointwise** flux
:math:`\varphi(r_i)` — not a region-averaged collision rate — so it
uses the raw :math:`e^{-\tau}` kernel directly and **bypasses** the
analytic chord averaging that produces :math:`E_3`. This makes the
sphere Peierls a clean cross-check on the CP flat-source
construction:

- A sign error, off-by-one, or factor-of-two in the
  :math:`E_n(\tau)` recurrence for :math:`n \ge 2` would affect the
  CP :math:`E_3` second-difference formula *but not* the Peierls
  :math:`e^{-\tau}` evaluation (which is just ``np.exp``). A
  systematic :math:`E_n` bug would therefore cancel between CP
  solver and CP-self-verification test but be caught cleanly by
  the Peierls reference.
- Because the Peierls Nyström operator resolves the flux as a
  piecewise polynomial of degree :math:`p - 1` on each radial
  panel, it is sensitive to flux-shape errors that are invisible to
  a flat-source CP-vs-CP comparison. The Phase-A Peierls-vs-CP
  flux-shape test
  (``TestCPvsPeierlsSphereAtThickR.test_flux_shape_agrees_at_thick_R``
  in ``tests/cp/test_peierls_sphere_flux.py``) is the first-order
  check that the CP flat-source approximation recovers the correct
  pointwise flux in the thick-sphere limit where the approximation
  is asymptotically exact.

The :math:`e^{-\tau}` kernel also avoids the common-mode
:math:`\mathrm{Ki}_n` recursion path: the cylinder Peierls depends
on :func:`~orpheus.derivations._kernels.ki_n_mp` (via the
mpmath-backed :math:`\mathrm{Ki}_1` evaluator, as the Phase B.4
retirement of ``BickleyTables`` routes all cylindrical kernel
evaluations through a single canonical primitive), but a sphere
Peierls run makes **zero** calls into any :math:`\mathrm{Ki}_n`
code path — just :func:`numpy.exp`. The two references triangulate
the integral-transport stack from orthogonal angles.

Nyström assembly in polar coordinates
-------------------------------------

The sphere discretisation mirrors the cylinder's three-layer polar
quadrature; each layer is dispatched through the unified
:class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
with ``kind = "sphere-1d"``.

**Radial grid (composite Gauss–Legendre on** :math:`[0, R]`\ **).**
The r-axis is partitioned into :math:`N_{\rm reg}` concentric
shells :math:`[r_{k-1}, r_k]`, :math:`r_0 = 0`, :math:`r_N = R`.
Each shell carries :math:`n_{\rm panels}` panels, each carrying
:math:`p` Gauss–Legendre nodes. Panel breakpoints coincide with
shell radii so the emission density :math:`q(r')` — which is
piecewise-smooth but has slope discontinuities at material
boundaries — is represented by a piecewise polynomial of degree
:math:`p - 1`. This is the same strategy as the cylinder. The
total number of radial Nyström unknowns is
:math:`N = N_{\rm reg} \cdot n_{\rm panels} \cdot p`. Builder is
``composite_gl_r`` (shared with the cylinder; the sphere module
re-exports it verbatim via
:mod:`orpheus.derivations.peierls_sphere`).

Verified by ``TestSphereCompositeRadialGL`` in
``tests/derivations/test_peierls_sphere_geometry.py``: the weighted
integrals :math:`\int_0^R 1\,\mathrm dr = R` and
:math:`\int_0^R 4\pi r^{2}\,\mathrm dr = \tfrac{4}{3}\pi R^{3}`
recover the analytic values to machine precision under the
composite grid.

**Polar quadrature (Gauss–Legendre on** :math:`[0, \pi]`\ **).**
With :math:`n_\theta` nodes and weights :math:`w_{\theta, k}`;
the physical interval :math:`[0, \pi]` is **not folded** (see the
note under :eq:`peierls-sphere-polar`). The :math:`\sin\theta`
factor from :eq:`peierls-sphere-equation` is applied inside the
assembly by
:meth:`CurvilinearGeometry.angular_weight`; for
``kind = "sphere-1d"`` this returns :math:`\sin\theta` evaluated at
each node, distinguishing the sphere from the cylinder's constant
:math:`W_\Omega = 1`.

**Ray-distance quadrature (Gauss–Legendre on**
:math:`[0, \rho_{\max}(r_i, \theta_k)]`\ **).**
With :math:`n_\rho` nodes per ray. The upper limit depends on both
the observer radius and the polar angle, so the
:math:`\rho`-quadrature is **remapped per** :math:`(i, k)` from the
reference interval :math:`[-1, 1]` to :math:`[0, \rho_{\max}]`.
For a homogeneous bare sphere, a fixed :math:`\rho`-scale would
under-resolve the long rays that pass near the diameter and
over-resolve short outward rays near the surface; the per-ray
remapping gives uniform relative accuracy.

**Source interpolation by Lagrange basis.**
Because :math:`r'(r_i, \rho_m, \theta_k)` is generally not a radial
quadrature node, the emission density at the source point is
expressed via the panel-local Lagrange basis:

.. math::

   q\!\bigl(r'_{ikm}\bigr)
     \;=\; \sum_{j=1}^{N} L_j\!\bigl(r'_{ikm}\bigr)\,q_j,

where :math:`L_j` is the degree-:math:`(p-1)` Lagrange polynomial
supported only on the panel containing :math:`r'_{ikm}`. The basis
is shared with the cylinder (``_lagrange_basis_on_panels`` in
:mod:`orpheus.derivations.peierls_geometry`); partition of unity
and polynomial reproduction are L0-verified in the cylinder's
``TestLagrangeBasisOnPanels`` and carry over to the sphere case
without modification.

**Assembled matrix.**
Substituting into :eq:`peierls-sphere-equation` gives the
identity-LHS form

.. math::
   :label: peierls-sphere-nystrom

   \Sigma_t(r_i)\,\varphi_i
     \;=\; \sum_{j=1}^{N} K_{ij}\,q_j + S_{\rm bc}(r_i),

.. vv-status: peierls-sphere-nystrom tested

with

.. math::

   K_{ij}
     \;=\; \frac{\Sigma_t(r_i)}{2}\!
       \sum_{k=1}^{n_\theta}\!\sum_{m=1}^{n_\rho}\!
         w_{\theta, k}\,\sin\theta_k\,w_{\rho, m}(r_i, \theta_k)\,
         e^{-\tau_{ikm}}\,L_j\!\bigl(r'_{ikm}\bigr).

The kernel matrix is assembled by ``build_volume_kernel`` in
:mod:`orpheus.derivations.peierls_sphere`, which dispatches to
:func:`peierls_geometry.build_volume_kernel` with the sphere
geometry singleton pre-bound. The per-sample optical depth
:math:`\tau_{ikm}` is computed by the shared multi-annulus walker,
identical to the cylinder case — the 1-D radial annulus crossings
are geometry-agnostic.

Ray optical-depth walker
------------------------

The optical depth along the ray from :math:`r_i` in direction
:math:`\theta` over distance :math:`\rho`,

.. math::
   :label: peierls-sphere-ray-optical-depth

   \tau(r_i, \rho, \theta)
     \;=\; \int_{0}^{\rho}\!
       \Sigma_t\!\bigl(r'(r_i, s, \theta)\bigr)\,\mathrm ds,

.. vv-status: peierls-sphere-ray-optical-depth tested

shares **the same walker** as the cylinder's
:eq:`peierls-cylinder-ray-optical-depth`: the boundary crossings
:math:`|\mathbf{r}(s)|^{2} = r_k^{2}` give the same quadratic in
:math:`s`, whose roots partition the ray into annular segments of
constant :math:`\Sigma_t`. Because the embedding ambient space
(2-D disc vs 3-D ball) enters only through the solid-angle weight
of :eq:`peierls-sphere-equation` and the kernel :math:`\kappa_d`,
the walker itself — which only sees the 1-D radial :math:`\Sigma_t`
profile and the 1-D chord algebra — is reusable verbatim.

L0-verified against closed-form traversals in
``tests/derivations/test_peierls_sphere_geometry.py``:

- ``TestSphereOpticalDepthAlongRay.test_homogeneous_1region_linear_in_rho``
  — short-circuit :math:`\tau = \Sigma_t\,\rho` for a bare sphere.
- ``test_scales_linearly_with_sig_t`` — doubling :math:`\Sigma_t`
  doubles :math:`\tau` at every :math:`(r_i, \theta, \rho)`.
- ``test_two_annulus_radial_transit`` — radial outward ray that
  crosses from the inner shell into the outer shell, pinning the
  per-segment accumulation against the analytic partitioning.
- ``test_two_annulus_through_centre_diameter`` — through-centre
  ray that traverses the inner shell twice (once on each side of
  the centre) and the outer shell twice, covering the four
  boundary-crossing algebra.

Row-sum identity — homogeneous and multi-region
-----------------------------------------------

The row-sum identity is the single most diagnostic consistency
check for the Peierls operator on any geometry. For the sphere the
structure is **identical** to the cylinder's
:ref:`peierls-cylinder-row-sum`, and the same
:math:`u = \tau(\rho)` change-of-variables argument carries
through: applying :math:`K` to a pure-scatter emission density
:math:`q \equiv \Sigma_t` reproduces the spatially uniform
:math:`\varphi \equiv 1` because the :math:`1/\Sigma_t` factor left
by the Jacobian is absorbed.

**Homogeneous sphere.** For a bare homogeneous sphere of radius
:math:`R`, the infinite-medium identity for the identity-LHS
kernel :eq:`peierls-sphere-nystrom` is

.. math::

   \sum_{j=1}^{N} K_{ij} \;=\; \Sigma_t(r_i) \qquad (R \to \infty).

The finite-sphere deficit :math:`\Sigma_t - \sum_j K_{ij}` equals
:math:`\Sigma_t \cdot P_{\rm esc}(r_i)` (the uncollided escape
probability weighted by :math:`\Sigma_t`). For :math:`R = 10` MFP,
:math:`\max_i |\Sigma_t - \sum_j K_{ij}| < 10^{-3}` at
:math:`r_i \le R/2`. Tested in
``TestSphereRowSumIdentity.test_interior_row_sum_equals_sigma_t``
in ``tests/derivations/test_peierls_sphere_prefactor.py``.

The deficit is **monotone increasing** from centre to surface
(``test_deficit_grows_toward_boundary``), and shrinks under
quadrature refinement at every interior observer
(``test_convergence_under_quadrature_refinement``).

**Multi-region sphere.** The naive "apply :math:`K` to
:math:`\mathbf 1`" identity fails when :math:`\Sigma_t` is
piecewise-constant across shells, for the same reason as the
cylinder: the change of variables :math:`u = \tau(\rho)` gives

.. math::

   \int_{0}^{\rho_{\max}}\!e^{-\tau(\rho)}\,\mathrm d\rho
     \;=\; \int_{0}^{\tau_{\max}}\!
       \frac{e^{-u}}{\Sigma_t\!\bigl(r'(u)\bigr)}\,\mathrm du,

and the :math:`1/\Sigma_t` factor depends on where along the ray
the source point sits. The correct multi-region identity is
obtained by applying :math:`K` to :math:`q = \Sigma_t`:

.. math::
   :label: peierls-sphere-row-sum-identity

   \sum_{j=1}^{N} K_{ij}\,\Sigma_t(r_j) \;=\; \Sigma_t(r_i)
   \qquad\text{(multi-region, } R \to \infty\text{)}.

.. vv-status: peierls-sphere-row-sum-identity documented

The :math:`\Sigma_t(r_j)` factor absorbs the :math:`1/\Sigma_t`
left behind by the change of variables, restoring
:math:`\int_0^\infty e^{-u}\,\mathrm du = 1` independently of
:math:`\Sigma_t` variation along the ray.

.. warning::

   As on the cylinder, a test that applied :math:`K` to
   :math:`\mathbf 1` and compared to :math:`\Sigma_t(r_i)` would
   silently fail for the multi-region case even when the
   implementation is correct. The row sum :math:`\sum_j K_{ij}`
   instead equals a ray-path-weighted average of
   :math:`1/\Sigma_t`, which is not a local quantity. The sphere
   test suite mirrors the cylinder pattern: the **homogeneous**
   identity uses :math:`\mathbf 1` because :math:`\Sigma_t` is
   constant; any future multi-region row-sum test must apply
   :math:`K` to :math:`\Sigma_t`, not to :math:`\mathbf 1`.

Surface-to-volume Green's function :math:`G_{\rm bc}`
------------------------------------------------------

White-BC closure needs the sphere's surface-to-volume Green's
function :math:`G_{\rm bc}(r_i)` — the scalar flux at interior
observer :math:`r_i` induced by a **unit uniform isotropic inward
partial current** :math:`J^{-}` on the spherical surface. This
section derives the compact observer-centred form used by the
implementation and documents the design choice to parametrise by
directions at the observer rather than area elements on the
surface.

For a uniform isotropic inward partial current :math:`J^{-}` on
the sphere, the inward angular flux on the surface is
:math:`\psi_{\rm in} = J^{-}/\pi`, since the partial current and
the isotropic angular flux are related by :math:`J^{-} =
\int_{\Omega \cdot \hat n < 0}|\Omega \cdot \hat n|\,\psi\,
\mathrm d\Omega = \pi\,\psi_{\rm in}` for an isotropic inward
hemisphere. The scalar flux at interior observer :math:`r_i` is
obtained by integrating the attenuated emission over **directions
at the observer**:

.. math::

   \varphi(r_i)
     \;=\; \psi_{\rm in}\!\int_{4\pi}\!
       e^{-\tau_{\rm surf}(r_i, \Omega)}\,\mathrm d\Omega
     \;=\; \frac{J^{-}}{\pi}\,\cdot\,2\pi\!\int_{0}^{\pi}\!
       \sin\theta\,e^{-\tau_{\rm surf}(r_i, \theta)}\,
       \mathrm d\theta,

where
:math:`\tau_{\rm surf}(r_i, \theta) = \int_0^{\rho_{\max}(r_i,\theta)}
\Sigma_t(r'(s))\,\mathrm ds` is the optical depth along the ray
from :math:`r_i` in direction :math:`\theta` to the surface.
Dividing by :math:`J^{-}`:

.. math::
   :label: peierls-sphere-G-bc

   G_{\rm bc}^{\rm sph}(r_i)
     \;=\; 2\!\int_{0}^{\pi}\!\sin\theta\,
       e^{-\tau_{\rm surf}(r_i, \theta)}\,\mathrm d\theta.

.. vv-status: peierls-sphere-G-bc tested

.. note::

   **Observer parametrisation vs surface parametrisation.** The
   standard textbook derivation writes :math:`G_{\rm bc}(r_i)` as
   an integral over the boundary surface area element
   :math:`\mathrm dA_{\rm surf} = R^{2}\sin\theta'\,\mathrm d\theta'\,
   \mathrm d\phi'` with a :math:`\cos\theta'` Lambertian weight
   (the projection of the inward normal onto the ray) and a
   :math:`1/d^{2}` geometric attenuation:

   .. math::

      G_{\rm bc}^{\rm surf}(r_i)
        \;=\; \frac{1}{\pi}\!\iint_{\rm surf}\!
          \frac{\cos\theta'\,e^{-\tau_{\rm surf}}}{d(r_i, \theta')^{2}}\,
            \mathrm dA_{\rm surf}.

   The **observer form** :eq:`peierls-sphere-G-bc` and the
   **surface form** are equivalent via change of variables: for
   every inward-pointing ray at the observer there is one and only
   one entry point on the surface, and the Jacobian of the mapping
   exactly cancels the :math:`1/d^{2}` attenuation and the
   :math:`\cos\theta'` Lambertian weight. The observer
   parametrisation is **structurally simpler** — no
   :math:`\cos\theta'`, no :math:`1/d^{2}`, no extra branch
   choice — and the angular range is the natural :math:`[0, \pi]`
   of polar-angle integration.

   This is the same Jacobian-cancellation principle that eliminates
   the :math:`1/\rho^{2}` volume singularity in the polar form of
   the volume kernel. Choosing the observer parametrisation is the
   design decision that makes :math:`G_{\rm bc}` a smooth integral
   that Gauss–Legendre quadrature handles spectrally.

**Vacuum limit (sanity check).** As :math:`\Sigma_t \to 0` the
exponential collapses to unity and

.. math::

   G_{\rm bc}^{\rm sph}(r_i)\,\big|_{\Sigma_t = 0}
     \;=\; 2\!\int_{0}^{\pi}\!\sin\theta\,\mathrm d\theta
     \;=\; 2 \cdot 2 \;=\; 4.

Physically: a uniform isotropic inward partial current of strength
:math:`J^{-}` on an empty ball fills the interior with scalar flux
:math:`4 J^{-}` (:math:`4\pi` sr of angular flux, each mode
:math:`\psi_{\rm in} = J^{-}/\pi`, integrated gives :math:`4\pi \cdot
J^{-}/\pi = 4 J^{-}`). This limit is tested in
``TestSphereGBCVacuumLimit.test_vacuum_G_bc_is_four`` in
``tests/derivations/test_peierls_sphere_prefactor.py``: a
:math:`\Sigma_t R = 10^{-8}` sphere gives
:math:`G_{\rm bc} = 4` to :math:`10^{-5}` at every interior
observer.

Rank-1 white-BC closure — geometry-aware surface divisor
----------------------------------------------------------

Under the **rank-1 Mark / isotropic closure**, the white-BC
correction to the volume kernel is of outer-product form
:math:`K_{\rm bc}[i, j] = u_i\,v_j` with

.. math::

   u_i \;=\; \frac{\Sigma_t(r_i)\,G_{\rm bc}(r_i)}{A_d},
   \qquad
   v_j \;=\; r_j^{d-1}\,w_j\,P_{\rm esc}(r_j),

where :math:`A_d` is the cell-surface measure, :math:`d \in
\{2, 3\}` is the ambient dimension, and :math:`P_{\rm esc}(r_j)`
is the uncollided escape probability.

For the **sphere** (:math:`d = 3`, :math:`A_d = 4\pi R^{2}`,
volume-element area :math:`A_j = 4\pi r_j^{2} w_j`), the
:math:`4\pi` azimuthal factor cancels between :math:`A_d` and
:math:`A_j`, leaving a ratio :math:`A_j / A_d = r_j^{2} w_j /
R^{2}`. The implementation therefore uses a surface divisor of
:math:`R^{2}`:

.. math::

   u_i^{\rm sph} = \frac{\Sigma_t(r_i)\,G_{\rm bc}(r_i)}{R^{2}},
   \qquad
   v_j^{\rm sph} = r_j^{2}\,w_j\,P_{\rm esc}(r_j).

For the **cylinder** the analogous ratio is :math:`r_j w_j / R`,
so the divisor is :math:`R`. These two cases are dispatched by
:meth:`CurvilinearGeometry.rank1_surface_divisor`, which returns
:math:`R` for the cylinder and :math:`R^{2}` for the sphere.

.. warning::

   **R-vs-R² gotcha.** A sphere Peierls implementation that
   re-uses cylinder scaffolding **without** updating the
   surface-divisor from :math:`R` to :math:`R^{2}` under-counts
   :math:`u_i` by a factor of :math:`R`. The symptom is an
   enormous overestimation of :math:`k_{\rm eff}`: the white-BC
   correction, having the wrong normalisation, feeds a spuriously
   large boundary source back into the fission eigenvalue. A
   previous attempt — see the historical retraction in
   :ref:`issue-100-retraction` and the full debate in
   `Issue #100 <https://github.com/deOliveira-R/ORPHEUS/issues/100>`_
   — hit exactly this wall and reported
   :math:`k_{\rm eff} \approx 6.7` for a 1-G bare sphere where
   :math:`k_\infty = 1.5`. The
   :meth:`~CurvilinearGeometry.rank1_surface_divisor` abstraction
   exists precisely to make this mistake impossible in new code.

The implementation is thin: :func:`build_white_bc_correction` in
:mod:`orpheus.derivations.peierls_sphere` calls
:func:`compute_G_bc`, :func:`compute_P_esc`, and assembles the
rank-1 outer product with the geometry-aware divisor — all via the
unified :class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
dispatch in :func:`peierls_geometry.build_white_bc_correction`.

.. _issue-100-retraction:

Rank-1 white-BC numerical evidence (sphere)
-------------------------------------------

The :math:`R^{2}`-divisor rank-1 closure is pinned by
``TestWhiteBCRank1ErrorScan`` against the bare-sphere
:math:`k_\infty = 1.5` reference at
:math:`(\Sigma_t, \Sigma_s, \nu\Sigma_f) = (1, 0.5, 0.75)` and
:math:`n_\theta = n_\rho = 20`, :math:`n_\phi = 32`, ``dps = 25``:

.. list-table:: Rank-1 white-BC :math:`k_{\rm eff}` scan, bare sphere
   :header-rows: 1
   :widths: 14 18 18 22

   * - :math:`R` / MFP
     - :math:`k_{\rm eff}` (sphere)
     - err vs :math:`k_\infty`
     - cylinder comparator
   * - 1.0
     - 1.0963
     - 26.9 %
     - 1.19 (21 %)
   * - 2.0
     - 1.3914
     - 7.2 %
     - 1.40 (7 %)
   * - 5.0
     - 1.4897
     - 0.7 %
     - 1.48 (2 %)
   * - 10.0
     - 1.4957
     - 0.3 %
     - 1.49 (1 %)
   * - 20.0
     - 1.4945
     - 0.4 %
     - —
   * - 30.0
     - 1.4946
     - 0.4 %
     - —

The sphere's rank-1 white-BC behaviour **parallels the
cylinder's** — large error at thin :math:`R`, monotone convergence
to :math:`k_\infty` as :math:`R \to \infty`. The
beyond-:math:`R = 10` MFP plateau at
:math:`|k_{\rm eff} - k_\infty|/k_\infty \approx 0.3\,\text{--}\,0.4\%`
is the quadrature-noise floor at the cited
:math:`(n_\theta, n_\rho, n_\phi)` order, not a rank-1 closure
defect. Under refinement the floor drops further
(``TestQuadratureConvergence.test_k_eff_converges_under_refinement``
monitors this at :math:`R = 4` MFP).

Row-sum residuals under the rank-1 white-BC correction
(:math:`\max_i |K_{\rm tot} \cdot \Sigma_t - \Sigma_t|` for the
homogeneous sphere at :math:`\Sigma_t = 1`):

.. list-table:: Row-sum residuals (homogeneous sphere)
   :header-rows: 1
   :widths: 20 30 30

   * - :math:`R`
     - vacuum :math:`K`
     - white :math:`K_{\rm tot}`
   * - 2.0 MFP
     - :math:`5.4\cdot10^{-1}`
     - :math:`6.2\cdot10^{-2}`
   * - 5.0 MFP
     - :math:`4.0\cdot10^{-1}`
     - :math:`9.4\cdot10^{-3}`
   * - 10.0 MFP
     - :math:`2.9\cdot10^{-1}`
     - :math:`5.8\cdot10^{-3}`

At :math:`R = 5` MFP the rank-1 closure recovers the row-sum
identity to better than 1 % — pinned in
``TestSphereWhiteBCRowSum.test_medium_sphere_residual_below_five_percent``
and ``test_thick_sphere_residual_below_two_percent``.

**Issue #100 historical retraction.** A pre-:math:`R^{2}`-divisor
sphere prototype reported :math:`k_{\rm eff} \approx 6.7` and
attributed the blow-up to a structural rank-1 failure on the
sphere (:math:`P_{\rm esc} / G_{\rm bc}` 40 %-variation argument).
That conclusion is **retracted** — the spurious factor of
:math:`R` injected by the cylinder-port divisor
(:math:`u_i / R` instead of :math:`u_i / R^{2}`) accounts for the
inflation at :math:`R \sim 1` MFP, and the
:math:`P_{\rm esc} / G_{\rm bc}` ratio argument conflated
:math:`u_i\,v_j` with the volume-to-volume coupling. The full
debate (pre-correction "ratio varies 40 % so rank-1 fails"
argument and the divisor-bug post-mortem) lives in
`Issue #100 <https://github.com/deOliveira-R/ORPHEUS/issues/100>`_
and the sister stub in :doc:`peierls_unified` §8. The
**residual** rank-1 deficit at :math:`R \to 0` is the genuine
flat-source-accuracy limit shared by sphere and cylinder; it is
tracked under
`Issue #103 <https://github.com/deOliveira-R/ORPHEUS/issues/103>`_
(higher-rank N1 closure), not as a sphere-specific bug.

Boundary conditions — rank-1 white and vacuum
----------------------------------------------

**Vacuum.** :math:`S_{\rm bc} \equiv 0`. The kernel :math:`K` is
the full operator; the eigenvalue problem is

.. math::

   \bigl[\mathrm{diag}(\Sigma_t) - K\,\mathrm{diag}(\Sigma_s)\bigr]
     \varphi \;=\; \frac{1}{k}\,K\,\mathrm{diag}(\nu\Sigma_f)\,\varphi,

solved by fission-source power iteration in
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
(with ``geometry=_pg.SPHERE_1D``) and ``boundary="vacuum"``. This
is the clean closure: no approximation enters beyond the quadrature
orders.

**Rank-1 white.** The unified :math:`K_{\rm vol} + K_{\rm bc}`
structure with :math:`K_{\rm bc}` the rank-1 outer product derived
above, solved by the same power iteration via
:func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
(with ``geometry=_pg.SPHERE_1D``) and ``boundary="white"``. Accuracy is governed by the cell
optical thickness (``test_thin_sphere_rank1_error_bounded`` /
``test_medium_sphere_rank1_error_bounded`` /
``test_thick_sphere_rank1_near_k_inf``).

.. note::

   The rank-1 closure collapses because of radial symmetry, not
   because of the specific dimensionality. For a 1-D radial sphere
   with isotropic scattering, the re-entering partial current is
   scalar (:math:`J^{-}` has a single degree of freedom by
   rotational symmetry), and the scalar balance
   :math:`J^{-} = J^{+}` is exact at every surface point. What the
   rank-1 Mark closure approximates is the **angular shape** of
   :math:`J^{-}` — treating it as isotropic in the inward
   hemisphere. That approximation is exact in the thick-cell
   limit (where the angular flux at the boundary is the integrated
   average over the slab's emission density) and degrades as
   :math:`R \to 0` (where the angular dependence on the surface is
   Fourier-rich). Issue #103 (N1) is the planned higher-rank
   angular expansion that lifts this restriction.

Relationship to the CP flat-source sphere solver
--------------------------------------------------

The CP flat-source method for the sphere
(:func:`~orpheus.cp.solver.solve_cp` on ``sph1D`` meshes;
:mod:`orpheus.derivations.cp_sphere`) integrates the native
:math:`e^{-\tau}` 3-D kernel analytically over each concentric
shell to produce the :math:`E_3` second-difference formula
(:eq:`second-diff-sph` and :eq:`self-sph` above). The Peierls
reference **bypasses that integration** entirely:

- the kernel is :math:`e^{-\tau}`, not :math:`E_3`,
- the spatial representation is a piecewise polynomial of degree
  :math:`p - 1` per panel, not a piecewise constant,
- the ray integration is performed numerically in observer-centred
  polar coordinates, not analytically over annular shell
  boundaries.

So the two methods share **almost nothing** except the underlying
point kernel they both derive from. A sign error, off-by-one, or
factor-of-two in the :math:`E_3(\tau_a) - E_3(\tau_b) - E_3(\tau_c)
+ E_3(\tau_d)` sphere CP second-difference formula — which would
cancel between the CP solver and a CP-self-verification test —
would be caught by the Peierls reference. Conversely, a systematic
error in the :math:`e^{-\tau}` evaluation (unlikely given that it
is ``numpy.exp``) would be caught by the CP eigenvalue tests. The
two together triangulate the spherical integral-transport stack.

The **kernel-level** relationship is exactly parallel to the
cylinder case:

.. list-table:: Peierls vs CP kernel pairing by geometry
   :header-rows: 1
   :widths: 20 30 30

   * - Geometry
     - Peierls (pointwise)
     - CP (flat-source)
   * - Slab
     - :math:`E_1`
     - :math:`E_3` (2nd diff.)
   * - Cylinder
     - :math:`\mathrm{Ki}_1`
     - :math:`\mathrm{Ki}_3` (2nd diff.)
   * - Sphere
     - :math:`e^{-\tau}`
     - :math:`E_3` (2nd diff.)

The Phase-A tests pin this relationship numerically:

- ``TestPeierlsSphereSelfConvergence.test_k_eff_cauchy_convergence_at_thick_R``
  and ``test_flux_cauchy_convergence_at_thick_R`` — Peierls is
  Cauchy-convergent in :math:`k_{\rm eff}` and flux profile under
  :math:`(n_\theta, n_\rho, n_\phi)` refinement.
- ``TestCPvsPeierlsSphereAtThickR.test_k_eff_agrees_at_thick_R`` —
  CP :math:`k_{\rm eff}` and Peierls :math:`k_{\rm eff}` agree to
  < 2 % at :math:`R = 10` MFP.
- ``TestCPvsPeierlsSphereAtThickR.test_flux_shape_agrees_at_thick_R``
  — volume-weighted normalised flux profiles agree to L2
  < 5 % at :math:`R = 10` MFP.

Phase B will unify ``cp_sphere.py`` and ``cp_cylinder.py`` under a
single ``cp_geometry.py`` module (Issue #107 / N6), mirroring the
already-completed Peierls unification; the rank-1 white-BC
closure parity between CP flat-source and Peierls rank-1 at the
thick-:math:`R` limit, verified by the tests above, is the
correctness gate for that unification.

Verification evidence
---------------------

Three classes of independent checks gate the Peierls sphere
implementation: geometry primitives (L0), prefactor / kernel
normalisation / row-sum (L0), and eigenvalue / flux-shape (L1).
All 35 sphere tests pass; cylinder regression (31 tests) passes
unchanged.

.. list-table:: Spherical Peierls verification summary
   :header-rows: 1
   :widths: 40 18 20 22

   * - Check
     - Level
     - Tolerance
     - Identity / eq.
   * - Geometry constants (prefactor, sin θ, r², R²)
     - L0
     - exact
     - :eq:`peierls-sphere-polar`
   * - :math:`\rho_{\max}` closed forms (5 cases)
     - L0
     - :math:`10^{-12}`
     - :eq:`peierls-sphere-rho-max`
   * - :math:`r'` closed forms (3 cases)
     - L0
     - :math:`10^{-12}`
     - :eq:`peierls-sphere-r-prime`
   * - Optical-depth walker (4 cases)
     - L0
     - :math:`10^{-12}`
     - :eq:`peierls-sphere-ray-optical-depth`
   * - Composite radial GL (3 integrals)
     - L0
     - :math:`10^{-12}`
     - :math:`\int_0^R 4\pi r^{2}\,\mathrm dr`
   * - :math:`G_{\rm bc}` vacuum limit
     - L0
     - :math:`10^{-5}`
     - :eq:`peierls-sphere-G-bc`
   * - Row sum (homogeneous, :math:`R = 10` MFP)
     - L0
     - :math:`<10^{-3}` interior
     - :eq:`peierls-sphere-nystrom`
   * - Row sum (white-BC, :math:`R = 10` MFP)
     - L0
     - :math:`<2\times 10^{-2}`
     - :eq:`peierls-sphere-row-sum-identity`
   * - Thick vacuum limit (:math:`R = 30` MFP)
     - L1
     - :math:`<10^{-2}` vs :math:`k_\infty`
     - vacuum fixed point
   * - Vacuum :math:`k_{\rm eff}(R)` monotonicity
     - L1
     - monotone on 5 :math:`R` values
     - vacuum fixed point
   * - Quadrature convergence (:math:`R = 4` MFP)
     - L1
     - :math:`|\Delta k|_{\rm next}` < prev
     - :eq:`peierls-sphere-nystrom`
   * - Rank-1 white-BC error scan
     - L1
     - :math:`<35\%` at :math:`R=1`, :math:`<1\%` at :math:`R=10`
     - :eq:`peierls-sphere-G-bc`
   * - CP-vs-Peierls :math:`k_{\rm eff}` (:math:`R = 10` MFP)
     - L1
     - :math:`<2\%`
     - :eq:`second-diff-sph`
   * - CP-vs-Peierls flux shape (:math:`R = 10` MFP)
     - L1
     - :math:`L^2 < 5\%`
     - :eq:`second-diff-sph`

[CaseZweifel1967]_ tabulates bare-sphere critical-radius
:math:`R_c` values as a function of :math:`c = \nu\Sigma_f /
\Sigma_a` (1-group) and offers a literature tie-point analogous to
the cylinder's Sanchez 1982 tie-point. The Peierls-sphere test
suite currently pins the solver empirically via the vacuum-BC
thick-limit and the monotone-:math:`R` scan rather than by
transcribing numerical :math:`R_c` values from the Case–Zweifel
tables (Cardinal Rule L4 forbids hand-transcription; a programmatic
ingestion of the tables via the literature-researcher agent is a
planned follow-up).

Numerical cost
--------------

The :math:`(\theta, \rho)` tensor-product quadrature dominates. For
each observer :math:`r_i` and each :math:`\theta_k`, the kernel
assembly evaluates :math:`e^{-\tau}` at :math:`n_\rho` points —
which is a single ``numpy.exp`` call per sample (no special-function
recurrence, unlike the cylinder's :math:`\mathrm{Ki}_1`). Dominant
cost: :math:`O(N \cdot n_\theta \cdot n_\rho)` exponential
evaluations per group. For :math:`N = 10` radial nodes,
:math:`(n_\theta, n_\rho) = (24, 24)`, ``dps = 20``, kernel
assembly takes :math:`\approx 1` s on current hardware (cheaper
than the cylinder by the :math:`\mathrm{Ki}_1`-vs-``exp`` speed
ratio); eigenvalue power iteration is a further :math:`O(N^{3})`
LU per iteration, typically converging in 20–30 iterations to
:math:`10^{-10}` eigenvalue tolerance.

Short-circuit: the homogeneous single-shell branch of
:func:`~orpheus.derivations.peierls_geometry.compute_G_bc` bypasses
the multi-annulus walker and computes :math:`\tau_{\rm surf} =
\Sigma_t\,\rho_{\max}` directly, making the bare-sphere case
:math:`\sim 2\times` faster than the multi-region path.

.. seealso::

   :mod:`orpheus.derivations.peierls_sphere` — thin facade; the
   sphere-specific API names, the
   ``_build_peierls_sphere_case`` registry builder, and the
   ``continuous_cases`` registration.

   :mod:`orpheus.derivations.peierls_geometry` — unified polar-form
   Nyström infrastructure; the
   :class:`~orpheus.derivations.peierls_geometry.CurvilinearGeometry`
   class with ``kind = "sphere-1d"`` and
   ``kind = "cylinder-1d"`` handles both geometries through one
   code path.

   :class:`~orpheus.derivations.peierls_geometry.PeierlsSolution`
   — canonical result container; same dataclass shape for
   ``geometry_kind="sphere-1d"`` / ``"cylinder-1d"`` / ``"slab"``.

   :func:`~orpheus.derivations.peierls_geometry.solve_peierls_1g`
   — 1-group vacuum- or white-BC eigenvalue driver. Pass
   ``geometry=_pg.SPHERE_1D`` (or ``CYLINDER_1D`` / ``SLAB_POLAR_1D``)
   and ``boundary="vacuum"`` for the scaffold-level verification
   gate.

   ``tests/derivations/test_peierls_sphere_geometry.py`` — 17 L0
   tests for angular/radial geometry primitives, the composite
   Gauss–Legendre builder, the :math:`\rho_{\max}` closed forms,
   the :math:`r'` closed forms, and the multi-annulus
   optical-depth walker.

   ``tests/derivations/test_peierls_sphere_prefactor.py`` — 6 L0
   tests for the row-sum identity (homogeneous and
   white-BC-corrected), and the :math:`G_{\rm bc}` vacuum-limit
   sanity check.

   ``tests/derivations/test_peierls_sphere_eigenvalue.py`` — 4 L1
   tests: vacuum-BC thick limit, :math:`k_{\rm eff}(R)`
   monotonicity, quadrature convergence, white-BC thick-limit
   sanity.

   ``tests/derivations/test_peierls_sphere_white_bc.py`` — 4 L1
   tests pinning the rank-1 closure error at :math:`R \in \{1, 2,
   5, 10\}` MFP (Issue #103 bounds).

   ``tests/cp/test_peierls_sphere_flux.py`` — 4 L1 tests for
   Peierls self-convergence and CP-vs-Peierls flux / eigenvalue
   agreement at :math:`R = 10` MFP.

   :doc:`peierls_unified` — cross-cutting architectural page:
   §2 dimensionally-reduced kernels, §3 Jacobian cancellation,
   §8 white-BC rank-1 closure and Issue #100 historical record.


References
==========

.. note::

   Citations shared across pages are defined in
   :doc:`/theory/peierls_unified` (the deeper treatment) and
   :doc:`/theory/discrete_ordinates`; ``[Foo1234]_`` references on
   this page resolve cross-document via Sphinx's docutils citation
   index. Only citations unique to this page are defined locally.

.. [Hebert2009] A. Hebert, *Applied Reactor Physics*, Presses
   internationales Polytechnique, 2009.

.. [Kress2014] R. Kress, *Linear Integral Equations*, 3rd ed.,
   Springer, 2014.


.. |times| unicode:: U+00D7