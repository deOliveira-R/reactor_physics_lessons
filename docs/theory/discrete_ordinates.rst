.. _theory-discrete-ordinates:

====================================
Discrete Ordinates Method (SN)
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
flux), the S\ :sub:`N` method retains the **angular flux**
:math:`\psi(\mathbf{r}, \hat{\Omega}, E)` and resolves directional effects
such as streaming, anisotropic scattering, and angular current at interfaces.

Compared to the :ref:`homogeneous model <theory-homogeneous>`, which
collapses spatial and angular dependence entirely, the S\ :sub:`N` method
produces the full spatial distribution of the neutron flux across a
heterogeneous geometry — fuel, cladding, and coolant zones with distinct
cross sections.

The treatment follows [LewisMiller1984]_, with cross-section data and
prototypical geometries. This chapter covers:

- The 1D slab transport equation and its multi-group generalisation
- Angular quadrature: Gauss–Legendre (1D) and Lebedev (sphere)
- Spatial discretisation: the diamond-difference scheme and sweep algorithm
- Scattering anisotropy: P\ :sub:`N` Legendre expansion
- Two inner solver strategies: sweep-based source iteration and
  direct-operator BiCGSTAB
- Verification against analytical eigenvalues

The solver is implemented in :class:`SNSolver`, which satisfies the
:class:`~numerics.eigenvalue.EigenvalueSolver` protocol.  The convenience
function :func:`solve_sn` runs the full calculation and returns an
:class:`SNResult`.


The Neutron Transport Equation
===============================

Starting Point
--------------

The steady-state neutron transport equation [CaseZweifel1967]_ for a single
energy group in a 1D slab geometry is (see the full 3D form in
:ref:`theory-homogeneous`):

.. math::
   :label: sn-transport-1d

   \mu \frac{\partial \psi(x, \mu)}{\partial x}
   + \Sigt{} \, \psi(x, \mu)
   = \frac{1}{2} \int_{-1}^{1}
     \Sigs{} \, \psi(x, \mu') \, d\mu'
   + \frac{1}{2} \frac{\chi \, \nSigf{}}{k} \int_{-1}^{1} \psi(x, \mu') \, d\mu'

where :math:`\mu = \cos\theta` is the direction cosine with respect to the
:math:`x`-axis, :math:`\Sigt{}` is the total cross section,
:math:`\Sigs{}` the scattering cross section, :math:`\nSigf{}` the fission
production cross section, :math:`\chi` the fission spectrum, and :math:`k`
the eigenvalue to be determined.

The **scalar flux** is the zeroth angular moment:

.. math::
   :label: scalar-flux-def

   \phi(x) = \int_{-1}^{1} \psi(x, \mu) \, d\mu

Multi-Group Extension
---------------------

For :math:`G` energy groups, the transport equation becomes a coupled system.
In group :math:`g`:

.. math::
   :label: sn-multigroup

   \mu \frac{\partial \psi_g}{\partial x}
   + \Sigt{g} \, \psi_g
   = \frac{1}{2} \sum_{g'=1}^{G}
     \Sigs{g' \to g} \, \phi_{g'}
   + \frac{1}{2} \frac{\chi_g}{k} \sum_{g'=1}^{G} \nSigf{g'} \, \phi_{g'}

The scattering transfer matrix :math:`\Sigs{g' \to g}` couples the groups.
ORPHEUS stores this as ``Mixture.SigS[0]`` with convention
:math:`\text{SigS}[g_\text{from}, g_\text{to}]`, so the in-scatter sum
:math:`\sum_{g'} \Sigs{g' \to g} \phi_{g'}` is computed as
:math:`(\mathbf{\Sigma}_s^T \cdot \boldsymbol{\phi})_g`.


Angular Discretisation
=======================

The S\ :sub:`N` approximation replaces the continuous angular integral with a
**quadrature rule**: a set of :math:`N` discrete directions
:math:`\{\hat{\Omega}_n\}` with weights :math:`\{w_n\}` such that

.. math::
   :label: angular-quadrature

   \int f(\hat{\Omega}) \, d\Omega
   \approx \sum_{n=1}^{N} w_n \, f(\hat{\Omega}_n)

The scalar flux is then approximated as

.. math::
   :label: scalar-flux-quad

   \phi(x) = \sum_{n=1}^{N} w_n \, \psi_n(x)

and the transport equation is solved **independently for each ordinate** :math:`n`:

.. math::
   :label: sn-ordinate

   \mu_n \frac{d\psi_n}{dx} + \Sigt{} \, \psi_n
   = \frac{Q}{W}

where :math:`Q` is the total isotropic source (fission + scattering + (n,2n))
and :math:`W = \sum_n w_n` is the weight sum.  The factor :math:`1/W` converts
the volumetric source rate to the per-steradian angular source density.

Gauss–Legendre Quadrature (1D)
-------------------------------

For 1D slab geometry, the angular variable is the single cosine
:math:`\mu \in [-1, 1]`.  The optimal quadrature is **Gauss–Legendre**,
which exactly integrates polynomials up to degree :math:`2N-1`:

.. math::
   :label: gauss-legendre

   \int_{-1}^{1} f(\mu) \, d\mu
   = \sum_{n=1}^{N} w_n \, f(\mu_n) + O(f^{(2N)})

The GL points are the roots of the Legendre polynomial :math:`P_N(\mu)`,
and the weights sum to 2 (the measure of :math:`[-1, 1]`).  The points
are symmetric: :math:`\mu_n = -\mu_{N+1-n}`, which ensures that the
angular flux is consistently represented for both forward and backward
directions.

Implemented in :class:`GaussLegendre1D`.  The :attr:`mu` property returns
the direction cosines (alias for ``mu_x``).

Lebedev Quadrature (2D/3D)
---------------------------

For multi-dimensional geometries, the angular variable is a direction on
the unit sphere :math:`\hat{\Omega} = (\mu_x, \mu_y, \mu_z)`.  ORPHEUS
uses **Lebedev quadrature** [Lebedev1999]_, which distributes :math:`N`
points on the sphere with octahedral symmetry.  For order 17, :math:`N = 110`
and the weights sum to :math:`4\pi`.

Implemented in :class:`LebedevSphere`.  Each ordinate has direction cosines
``(mu_x, mu_y, mu_z)`` and a reflective partner index for each coordinate
axis, computed at construction.

.. note::

   Ordinates with :math:`\mu_x = \mu_y = 0` (pointing purely in the
   :math:`z`-direction) have no streaming component in the :math:`(x,y)`
   plane.  These are handled as **pure-collision** ordinates:
   :math:`\psi_n = Q / (W \cdot \Sigt{})`.  Omitting them would lose
   0.77% of the quadrature weight, causing a systematic eigenvalue error
   visible only in multi-group problems (see ``gotchas.md`` in the
   repository root).


Spatial Discretisation: Diamond Difference
===========================================

The Diamond-Difference Equation
--------------------------------

Consider ordinate :math:`n` with :math:`\mu_n > 0` traversing cell :math:`i`
of width :math:`\Delta x_i`.  Integrating the transport equation over the
cell volume and applying the **diamond-difference approximation**
:math:`\psi_{\text{avg}} = \frac{1}{2}(\psi_{\text{in}} + \psi_{\text{out}})`
gives:

.. math::
   :label: diamond-difference

   \psi_{\text{avg},i}
   = \frac{Q_i / W + \frac{2|\mu_n|}{\Delta x_i} \psi_{\text{in}}}
          {\Sigt{i} + \frac{2|\mu_n|}{\Delta x_i}}

The **outgoing** face flux is obtained from the diamond relation:

.. math::
   :label: dd-outgoing

   \psi_{\text{out}} = 2 \psi_{\text{avg}} - \psi_{\text{in}}

Sweep Algorithm
---------------

Because each cell's outgoing flux becomes the next cell's incoming flux,
the equations must be solved in the direction of neutron travel:

- For :math:`\mu_n > 0`: sweep left → right (cells :math:`1, 2, \ldots, N_x`)
- For :math:`\mu_n < 0`: sweep right → left (cells :math:`N_x, \ldots, 2, 1`)

This directional sweep is the defining feature of the S\ :sub:`N` method.
A single sweep over all ordinates and all cells is one **transport sweep**,
implemented in :func:`transport_sweep`.

In 2D, cells on the same anti-diagonal :math:`i + j = k` are independent
for a given sweep direction, enabling **wavefront parallelism**.  The
implementation in :func:`_sweep_2d_wavefront` precomputes diagonal indices
for each of the four sweep directions.

Convergence Order
-----------------

The diamond-difference scheme is **second-order accurate** in the mesh
spacing :math:`h`:

.. math::

   k_{\text{eff}}(h) = k_{\text{exact}} + C h^2 + O(h^4)

This is verified by the spatial convergence test in the test suite, which
demonstrates the expected :math:`O(h^2)` rate on a 1G heterogeneous slab.


Boundary Conditions
====================

ORPHEUS uses **reflective boundary conditions** on all faces, representing
an infinite lattice of identical unit cells.  At a reflective boundary,
the incoming angular flux equals the outgoing flux of the **reflected
partner** ordinate:

.. math::
   :label: reflective-bc

   \psi_n^{\text{in}} = \psi_{n'}^{\text{out}}

where :math:`n'` is the ordinate obtained by reflecting :math:`\hat{\Omega}_n`
about the boundary normal.  For example, at the left boundary (:math:`x = 0`),
an ordinate with :math:`\mu_x > 0` (incoming) receives the outgoing flux of
the ordinate with :math:`-\mu_x` (same :math:`\mu_y, \mu_z`).

The reflection indices are precomputed by :meth:`~LebedevSphere.reflection_index`
and stored per axis.  Boundary fluxes are **persistent** between outer
iterations (stored in the ``psi_bc`` cache), which improves convergence by
carrying information from previous sweeps.


Scattering Anisotropy: P\ :sub:`N` Expansion
===============================================

P\ :sub:`0` Isotropic Scattering
----------------------------------

In the simplest approximation (P\ :sub:`0`), scattering is isotropic: a
neutron emerging from a collision has no preferred direction.  The
scattering source for group :math:`g` is:

.. math::
   :label: p0-scatter

   Q_{\text{scatter},g}
   = \sum_{g'} \Sigs{g' \to g}^{(0)} \, \phi_{g'}

This is independent of the ordinate direction :math:`\hat{\Omega}_n` and
is added to the isotropic source :math:`Q` before the sweep.

P\ :sub:`N` Anisotropic Scattering
------------------------------------

Real scattering is not isotropic — neutrons scattered by light nuclei
(hydrogen) are strongly forward-peaked.  The scattering cross section is
expanded in Legendre polynomials:

.. math::
   :label: pn-scatter-kernel

   \Sigs{}(\hat{\Omega}' \to \hat{\Omega})
   = \sum_{\ell=0}^{L} \frac{2\ell + 1}{W}
     \Sigs{}^{(\ell)} \sum_{m=-\ell}^{\ell}
     Y_\ell^m(\hat{\Omega}) \, Y_\ell^m(\hat{\Omega}')

where :math:`Y_\ell^m` are the real spherical harmonics and
:math:`\Sigs{}^{(\ell)}` is the :math:`\ell`-th Legendre moment of the
differential scattering cross section.

The anisotropic scattering source for ordinate :math:`n` is:

.. math::
   :label: pn-scatter-source

   Q_{\text{scatter},g}(\hat{\Omega}_n)
   = \sum_{\ell=0}^{L} (2\ell+1)
     \sum_{g'} \Sigs{g' \to g}^{(\ell)}
     \left[ \sum_{m=-\ell}^{\ell}
       f_{\ell,g'}^m \, Y_\ell^m(\hat{\Omega}_n) \right]
     \Big/ W

where the **Legendre moments** of the angular flux are:

.. math::
   :label: legendre-moments

   f_{\ell,g}^m = \sum_{n=1}^{N} w_n \, \psi_{n,g} \, Y_\ell^m(\hat{\Omega}_n)

Spherical Harmonic Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ORPHEUS uses the following real spherical harmonics, matching the MATLAB
reference implementation:

.. math::

   Y_0^0 = 1, \qquad
   Y_1^{-1} = \mu_z, \quad
   Y_1^{0} = \mu_x, \quad
   Y_1^{+1} = \mu_y

These are computed by :meth:`~LebedevSphere.spherical_harmonics` and
satisfy the discrete orthogonality relation:

.. math::

   \sum_{n=1}^{N} w_n \, Y_\ell^m(\hat{\Omega}_n) \,
   Y_{\ell'}^{m'}(\hat{\Omega}_n)
   = \frac{W}{2\ell+1} \, \delta_{\ell\ell'} \, \delta_{mm'}

Implementation
~~~~~~~~~~~~~~

For :math:`L = 0`, the anisotropic source is zero and the code path is
identical to the P\ :sub:`0`-only formulation.  For :math:`L \geq 1`, the
:math:`\ell \geq 1` terms produce a **per-ordinate** source
:math:`Q_{\text{aniso}}(n, x, y, g)` that is added to the isotropic source
inside the sweep.  See :meth:`SNSolver._build_aniso_scattering`.

.. note::

   The 421-group cross-section library provides P\ :sub:`1` data
   (``Mixture.SigS[1]``).  The synthetic verification library provides
   P\ :sub:`1` data for all four regions (A–D) with physically motivated
   anisotropy ratios: fuel :math:`\bar{\mu} = 0.05`, moderator
   :math:`\bar{\mu} = 0.60`, cladding :math:`\bar{\mu} = 0.10`,
   gap :math:`\bar{\mu} = 0.30`.


The Eigenvalue Problem
=======================

The eigenvalue :math:`\keff` is determined by **power iteration**: an outer
loop that updates :math:`k` from the production/absorption ratio, with an
inner loop that solves the within-group scattering problem for a fixed
fission source.

.. math::
   :label: power-iteration

   k^{(m+1)} = k^{(m)} \cdot
   \frac{\sum_g \int \nSigf{g} \, \phi_g^{(m+1)} \, dV}
        {\sum_g \int \nSigf{g} \, \phi_g^{(m)} \, dV}

The outer iteration is implemented by :func:`~numerics.eigenvalue.power_iteration`,
which calls the :class:`SNSolver` methods at each step.

Source Iteration (Sweep-Based)
-------------------------------

The default inner solver performs **scattering source iteration**: a
fixed-point iteration that sweeps the transport equation with the
scattering source lagged from the previous iterate:

.. math::
   :label: source-iteration

   \boldsymbol{\phi}^{(k+1)} = T^{-1}\bigl(
     S \boldsymbol{\phi}^{(k)} + \mathbf{Q}_f
   \bigr)

where :math:`T^{-1}` denotes one transport sweep (source → scalar flux),
:math:`S` is the scattering operator, and :math:`\mathbf{Q}_f` the fission
source.  The convergence rate is governed by the **spectral radius** of
:math:`T^{-1}S`, which is close to 1 for optically thick, highly scattering
media (e.g., :math:`\rho \approx 0.97` for 421-group water).

Implemented in :meth:`SNSolver._solve_source_iteration`.

BiCGSTAB (Direct Operator)
----------------------------

For problems where source iteration converges slowly, the alternative
inner solver forms the transport operator :math:`T` explicitly:

.. math::
   :label: transport-operator

   (T \boldsymbol{\psi})_n
   = \mu_n \frac{\partial \psi_n}{\partial x}
   + \Sigt{} \, \psi_n

using finite-difference gradients (the same diamond-scheme stencil as the
sweep, but applied as a matrix-vector product).  The full system
:math:`T \boldsymbol{\psi} = \mathbf{b}` is then solved by scipy's BiCGSTAB
Krylov solver, where :math:`\mathbf{b}` includes fission, scattering, and
(n,2n) sources normalised by :math:`1/W`.

This approach works on the **angular flux** :math:`\psi(x, n, g)` rather
than the scalar flux, and converges in :math:`\sim 100` Krylov iterations
regardless of the scattering spectral radius.

Implemented in :meth:`SNSolver._solve_bicgstab` using the operator from
:mod:`sn_operator` (:func:`build_transport_linear_operator`,
:func:`build_rhs`).

.. warning::

   The two inner solvers use **different spatial discretisations**
   (wavefront sweep vs. finite-difference gradient) that produce
   slightly different :math:`\keff` on coarse meshes.  They converge to
   the same answer as :math:`h \to 0`.


Multi-Group Conventions
========================

Cross-Section Storage
---------------------

The macroscopic cross sections are stored per material in the
:class:`~data.macro_xs.mixture.Mixture` dataclass:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Field
     - Meaning
     - Shape
   * - ``SigT``
     - Total cross section :math:`\Sigt{g}`
     - ``(ng,)``
   * - ``SigS[l]``
     - P\ :sub:`l` scattering matrix :math:`\Sigs{g' \to g}^{(\ell)}`
     - ``(ng, ng)`` sparse
   * - ``SigP``
     - Production :math:`\nSigf{g}`
     - ``(ng,)``
   * - ``chi``
     - Fission spectrum :math:`\chi_g`
     - ``(ng,)``

The scattering matrix convention is :math:`\text{SigS}[\ell][g_\text{from}, g_\text{to}]`.
The in-scatter sum is therefore :math:`(\mathbf{\Sigma}_s^T \cdot \boldsymbol{\phi})_g`.

The ``absorption_xs`` property includes fission:
:math:`\Siga{} = \Sigf{} + \Sigma_c + \Sigma_L + \Sigma_{n,2n}`, which
equals the removal rate :math:`\Sigt{} - \sum_{g'} \Sigs{g \to g'}`.

The keff Formula
-----------------

For an infinite lattice (reflective BCs, no leakage):

.. math::
   :label: keff-formula

   \keff
   = \frac{\sum_g \int \nSigf{g} \, \phi_g \, dV}
          {\sum_g \int \Siga{g} \, \phi_g \, dV}

The volume integration uses the cell volumes from
:attr:`CartesianMesh.volume`.

.. note::

   For a single energy group, :math:`\kinf = \nSigf{}/\Siga{}` is
   independent of the flux shape.  Only multi-group problems have a
   flux-ratio-dependent eigenvalue — see the discussion in
   ``gotchas.md``.


Example: 1D PWR Pin Cell Slab
===============================

The demonstration problem is a 1D slab representing one quarter of a PWR
unit cell with reflective boundary conditions:

- **Fuel** (5 cells): UO\ :sub:`2` at 900 K, 3% enrichment
- **Cladding** (1 cell): Zircaloy at 600 K
- **Coolant** (4 cells): borated water at 600 K, 16 MPa, 4000 ppm boron
- Cell width: :math:`\delta = 0.2` cm
- 421 energy groups, P\ :sub:`0` scattering
- S\ :sub:`16` Gauss–Legendre quadrature, BiCGSTAB inner solver

.. plot::
   :caption: Neutron flux per unit lethargy in fuel, cladding, and coolant.

   import sys, numpy as np
   sys.path.insert(0, '02.Discrete.Ordinates')
   sys.path.insert(0, '.')
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   from data.macro_xs.recipes import borated_water, uo2_fuel, zircaloy_clad
   from sn_geometry import CartesianMesh
   from sn_quadrature import GaussLegendre1D
   from sn_solver import solve_sn

   fuel = uo2_fuel(temp_K=900)
   clad = zircaloy_clad(temp_K=600)
   cool = borated_water(temp_K=600, pressure_MPa=16.0, boron_ppm=4000)
   materials = {2: fuel, 1: clad, 0: cool}

   widths = np.full(10, 0.2)
   mat_ids = np.array([2,2,2,2,2,1,0,0,0,0], dtype=int)
   mesh = CartesianMesh.from_slab_1d(widths, mat_ids)
   quad = GaussLegendre1D.create(16)

   result = solve_sn(materials, mesh, quad, inner_solver='bicgstab',
                     max_outer=500, max_inner=2000, inner_tol=1e-4)

   sf = result.scalar_flux[:, 0, :]
   eg = result.eg
   eg_mid = 0.5 * (eg[:-1] + eg[1:])
   du = np.log(eg[1:] / eg[:-1])

   vol = mesh.volume[:, 0]
   labels = {2: ('Fuel', 'r'), 1: ('Cladding', 'g'), 0: ('Coolant', 'b')}
   fig, ax = plt.subplots(figsize=(10, 6))
   for mid, (label, color) in labels.items():
       mask = mat_ids == mid
       vol_mat = vol[mask].sum()
       flux_avg = np.sum(sf[mask] * vol[mask, None], axis=0) / vol_mat
       ax.semilogx(eg_mid, flux_avg / du, f'-{color}', label=label, linewidth=1)
   ax.set_xlabel('Energy (eV)')
   ax.set_ylabel('Neutron flux per unit lethargy (a.u.)')
   ax.legend(loc='upper left')
   ax.grid(True, alpha=0.3)
   fig.tight_layout()

Result: :math:`\keff = 1.03882`, 117 outer iterations, 11 s wall time.


Verification
=============

Homogeneous Infinite Medium
----------------------------

For a spatially homogeneous slab with reflective BCs, the S\ :sub:`N`
equation has the exact solution :math:`\phi_g = \text{const}` and
:math:`\keff = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})` where
:math:`\mathbf{A} = \text{diag}(\Sigt{}) - \mathbf{\Sigma}_s^T` and
:math:`\mathbf{F} = \boldsymbol{\chi} \otimes \boldsymbol{\nu\Sigma_f}`.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Groups
     - Analytical :math:`\kinf`
     - S\ :sub:`N` :math:`\keff`
     - Error
   * - 1
     - 1.5000000000
     - 1.5000000000
     - :math:`< 10^{-15}`
   * - 2
     - 1.8750000000
     - 1.8750000000
     - :math:`< 10^{-14}`
   * - 4
     - 1.4877619048
     - 1.4877619048
     - :math:`< 10^{-14}`

Spatial and Angular Convergence
--------------------------------

The diamond-difference scheme converges at :math:`O(h^2)` with mesh
refinement.  The Gauss–Legendre quadrature shows **spectral** convergence
with increasing :math:`N` (number of ordinates): errors decrease faster
than any polynomial order.  Both are verified in the test suite
(``test_sn_1d.py``).

Analytical Derivations
-----------------------

.. include:: ../_generated/sn_derivation.rst


References
==========

.. [LewisMiller1984] E.E. Lewis and W.F. Miller, Jr.,
   *Computational Methods of Neutron Transport*,
   John Wiley & Sons, 1984.

.. [CaseZweifel1967] K.M. Case and P.F. Zweifel,
   *Linear Transport Theory*,
   Addison-Wesley, 1967.

.. [Lebedev1999] V.I. Lebedev and D.N. Laikov,
   "A quadrature formula for the sphere of the 131st algebraic order
   of accuracy," *Doklady Mathematics*, 59(3):477–481, 1999.
