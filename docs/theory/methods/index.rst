.. _theory-transport-methods:

=================
Transport Methods
=================

The realization part: each chapter takes the one transport problem and
commits to a concrete way of solving it.  The frame the methods
*share* — the one object they all discretize, the invariant that poses
it, the three axes on which they differ, and where each method lands —
is the corpus root, :doc:`/theory/foundations/path_integral`
(:ref:`path-integral-method-map`); this page deliberately does **not**
duplicate that comparative map.  What lives here: the local equation
the deterministic methods start from, the cast of chapters, and the
reading tracks.

The differential transport equation
===================================

The deterministic methods discretize the **local
(integro-differential) form** of the steady-state transport equation:
for each energy group :math:`g` and direction :math:`\hat\Omega`,

.. math::
   :label: methods-local-transport-equation

   \underbrace{\hat\Omega\cdot\nabla\,
      \psi_g(\mathbf r,\hat\Omega)}_{\text{streaming}}
   \;+\;
   \underbrace{\Sigt{g}(\mathbf r)\,
      \psi_g(\mathbf r,\hat\Omega)}_{\text{collision}}
   \;=\;
   \underbrace{\sum_{g'}\int_{4\pi}
      \Sigma_{s,\,g'\to g}(\hat\Omega'\!\cdot\hat\Omega)\,
      \psi_{g'}(\mathbf r,\hat\Omega')\,d\Omega'}_{\text{in-scatter}}
   \;+\;
   \underbrace{\frac{\chi_g}{4\pi\,k}\sum_{g'}
      \nu\Sigma_{f,\,g'}\,\phi_{g'}(\mathbf r)}_{\text{fission}}
   \;+\; q_g(\mathbf r,\hat\Omega),

.. (vv-status rationale) definition: the part-opener statement of the
.. governing local (integro-differential) transport equation the
.. deterministic methods discretize — a pedagogical restatement of the
.. governing PDE, not a solver claim. Each method's discretization of it is
.. verified on its own page.
.. vv-status: methods-local-transport-equation documented

closed by the **inflow boundary condition** on the domain surface
(vacuum: zero inflow; reflective: inflow = reflected outflow — the
law catalog is :doc:`/theory/foundations/boundary_conditions`).  The
:math:`1/k` scaling on fission is the criticality posing — the
eigenvalue is introduced *before* any discretization
(:ref:`path-integral-eigenvalue`).

Streaming reads most naturally in its **Lagrangian** form: along a
flight at direction :math:`\hat\Omega`, parametrised by arc length
:math:`s`, the :term:`angular flux` changes only by collision and emission —

.. math::
   :label: methods-lagrangian-streaming

   \frac{d\psi}{ds} \;=\; \hat\Omega\cdot\nabla\psi ,

.. (vv-status rationale) definition: the Lagrangian (along-the-flight) form of
.. the streaming term — a notation/identity introducing the arc-length view,
.. not a solver claim.
.. vv-status: methods-lagrangian-streaming documented

so the streaming term is *leakage along the flight*.  Every
deterministic method is a strategy for this one term: S\ :sub:`N`
upwinds it on a mesh, MoC integrates it exactly along traced rays, CP
integrates its resolvent into collision probabilities.  The
**Eulerian** face of the same term — the 0th angular moment
:math:`\int(\hat\Omega\cdot\nabla\psi)\,d\Omega = \nabla\cdot\mathbf J`,
the divergence of the current — is what diffusion closes with Fick's
law (:doc:`diffusion_1d`, the P\ :sub:`1` limit's own chapter).

Discretized on a deterministic angular–spatial grid, the equation
becomes the honest operator algebra
:math:`A = L + C - S - N_{2n} - B` posed as
:math:`A\psi = \tfrac{1}{k}F\psi`
(:doc:`/theory/foundations/operator_algebra`,
:ref:`eigenvalue-posing`).  *How* each method realizes the swept /
traced / integrated resolvent inside :math:`A^{-1}` is exactly the
first of the three axes on which the methods differ
(:ref:`path-integral-axes`).

The methods
===========

One line each — what the chapter *is*, not how the methods compare
(the comparative placement is the root's map,
:ref:`path-integral-method-map`):

- :ref:`theory-discrete-ordinates` — differential transport via
  angular :term:`quadrature` and spatial sweeps; the corpus's most developed
  method (a sub-book: Cartesian 1-D/2-D, curvilinear 1-D, adjoint,
  representations).
- :ref:`theory-collision-probability` — integral transport via the
  :math:`P_{ij}` matrix (slab, cylindrical, spherical kernels).
- :ref:`theory-method-of-characteristics` — characteristic ray
  tracing with flat-source approximation (2-D pin cell).
- :ref:`theory-monte-carlo` — stochastic transport via Woodcock
  delta-tracking with analog absorption and weight-based population
  control.
- :ref:`theory-diffusion-1d` — the P\ :sub:`1` (lowest-order) angular
  limit; not a transport solver in the strict sense, but the workhorse
  of reactor design and the target of the DSA acceleration seam.

Reading tracks
==============

Several orders through this part serve different jobs
(tracks, not one linear sequence):

* **Newcomer** — read the S\ :sub:`N` base chapter first:
  :doc:`sn/slab_one_group` shows the *whole machine* (posing →
  balance → operator → :term:`sweep` → iteration) at its simplest, then
  :doc:`sn/slab_multigroup` adds energy and the eigenvalue, then
  :doc:`diffusion_1d` shows the P\ :sub:`1` limit the accelerators
  lean on.  Broaden afterwards: :doc:`collision_probability` and
  :doc:`method_of_characteristics` re-solve the same problem with a
  different streaming realization.
* **Choosing a method** — start UP at the root: the three axes and
  the method map (:ref:`path-integral-axes`,
  :ref:`path-integral-method-map`).
* **Debugging a wrong answer** — each part carries its diagnostics
  where the machinery lives: the S\ :sub:`N` gotcha index
  (:ref:`sn-gotchas`) and the verification part
  (:doc:`/theory/verification/index`) with its per-equation V&V matrix.

.. toctree::
   :maxdepth: 2

   sn/index
   collision_probability
   method_of_characteristics
   monte_carlo
   diffusion_1d
