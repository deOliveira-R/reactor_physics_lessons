.. _theory-transport-methods:

=================
Transport Methods
=================

Deterministic transport solvers for heterogeneous lattice cells.
Each method discretises the angular variable differently and uses a
geometry-specific kernel or sweep algorithm:

- **Collision Probabilities** — integral transport via the
  :math:`P_{ij}` matrix (slab, cylindrical, spherical kernels).
- **Discrete Ordinates** — differential transport via angular
  quadrature and spatial sweeps (Cartesian 1-D / 2-D).
- **Monte Carlo** — stochastic transport via Woodcock delta-tracking
  with analog absorption and weight-based population control.

.. toctree::
   :maxdepth: 2

   collision_probability
   discrete_ordinates
   monte_carlo
