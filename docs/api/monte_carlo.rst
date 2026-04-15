Monte Carlo Neutron Transport
==============================

The :mod:`orpheus.mc` package is a 2-D Monte Carlo neutron
transport solver for PWR-style pin cells. It is designed as the
stochastic counterpart to the deterministic solvers (SN, CP, MOC,
diffusion) — same cross-section library, same material layouts,
same k-effective output — so students can compare deterministic
and stochastic answers on identical problems.

.. contents::
   :local:
   :depth: 2

.. seealso::

   :ref:`theory-monte-carlo` — Woodcock delta tracking, analog
   absorption with fission weight, population control, and the
   variance / bias tradeoffs.


Algorithmic Choices
-------------------

**Woodcock delta tracking.**
The random walk uses Woodcock (1965) delta tracking rather than
ray-casting to surfaces. Distance-to-collision is sampled from
the *majorant* :math:`\Sigma_{\rm maj} = \max_m \Sigma_{t,m}`
over all materials; the collision point is then rejected with
probability :math:`1 - \Sigma_t(\mathbf r)/\Sigma_{\rm maj}`. This
lets the geometry interface expose only ``material_id_at(x, y)``
and skip distance-to-surface computation entirely — a huge win
for complex CSG and the key reason the
:class:`~orpheus.mc.solver.MCGeometry` protocol is so small.

**Analog absorption with fission weight.**
Collisions are handled analog (a sampled reaction type decides
scatter vs absorb vs fission), but fission produces *weight*
rather than immediate daughter neutrons. The banked weight is
carried to the next cycle, giving unbiased keff estimates
without implicit capture variance inflation.

**Collision flux tally.**
Flux-per-lethargy is tallied with a collision estimator
(:math:`\phi \approx \sum_c w/\Sigma_t`) rather than a track-length
estimator. This was the refactor in commit ``44fa7bb``: the
track-length estimator had a subtle miscount when delta-tracking
virtual collisions were excluded from the summation, and the
collision estimator is simpler to implement correctly in a
delta-tracking transport loop.

**(n,2n) in the scattering kernel.**
The scattering kernel samples from
:math:`\Sigma_{s0} + 2\,\Sigma_{\rm n,2n}` and treats the (n,2n)
reaction as a pure scatter with double weight. Without the
factor of two the Monte Carlo answer was low-biased by the total
(n,2n) yield — fixed in commit ``b60518e``. The deterministic
homogeneous solver applies the same 2× factor in its removal
matrix (see :doc:`homogeneous`), which is why the two codes
now agree on test problems with strong (n,2n) content.


Data Structures
---------------

:class:`~orpheus.mc.solver.Particle` is a minimal per-particle
state record (position, direction, weight, energy group). It is
intentionally transport-neutral — the
:class:`~orpheus.mc.solver.Neutron` subclass adds nothing today
but leaves room for a future gamma transport mode.

:class:`~orpheus.mc.solver.NeutronBank` is the population
container. It uses array-backed storage (NumPy arrays sized to
the current population, growable) rather than a Python list of
``Particle`` objects, which gives a roughly order-of-magnitude
speed-up on the hot random-walk loop. The bank exposes three
population-control primitives:

* ``normalize_weights(n_target)`` — rescale the weights so the
  total equals ``n_target`` (the source-iteration normalisation).
* ``save_start_weights()`` — snapshot the pre-transport weights
  so the cycle keff can be computed as
  :math:`k_c = W_{\rm after}/W_{\rm before}`.
* ``compact()`` — drop dead particles and shrink the arrays.

The population-control functions
:func:`~orpheus.mc.solver._russian_roulette` and
:func:`~orpheus.mc.solver._split_heavy` operate on a bank in place.


Geometry
--------

Two concrete :class:`~orpheus.mc.solver.MCGeometry` implementations
ship with the module:

* :class:`~orpheus.mc.solver.ConcentricPinCell` — nested annuli
  inside a square lattice, material looked up by radial distance
  from the cell centre.
* :class:`~orpheus.mc.solver.SlabPinCell` — Cartesian slab layout
  (the ``default_pwr`` constructor matches the
  :func:`~orpheus.geometry.factories.pwr_slab_half_cell` geometry
  used by the deterministic solvers).

Both implement the runtime-checkable
:class:`~orpheus.mc.solver.MCGeometry` protocol, so user code can
pass in any duck-typed geometry that exposes ``pitch`` and
``material_id_at(x, y)``.


Source Iteration / Cycle Loop
-----------------------------

:func:`~orpheus.mc.solver.solve_monte_carlo` is the orchestrator:

1. Build the cross-section packs via
   :func:`~orpheus.mc.solver._precompute_xs` — cumulative
   distribution tables for energy group sampling, per-material
   total cross sections, and the Woodcock majorant.
2. Initialise the bank with ``n_neutrons`` source particles drawn
   from the fission spectrum.
3. Loop over ``n_inactive + n_active`` cycles:

   a. Normalise weights so :math:`\sum w = n_{\rm neutrons}`.
   b. Transport every particle via
      :func:`~orpheus.mc.solver._random_walk`.
   c. Apply Russian roulette, compact the bank, split heavy
      particles.
   d. Compute the cycle keff from the weight ratio.
   e. If in an active cycle, accumulate the cumulative mean and
      standard deviation of :math:`k_{\rm eff}`.

4. Return an :class:`~orpheus.mc.solver.MCResult` containing the
   final keff with its uncertainty, the full
   :math:`(k_{\rm eff}, \sigma)` history, the flux-per-lethargy
   spectrum, and the mid-group energies.

**Inactive vs active cycles.**
The first ``n_inactive`` cycles are **source convergence** — the
spatial and energy distribution of fission sites has not yet
settled onto the fundamental mode, so their keff estimates are
biased. They are discarded. Only the ``n_active`` cycles that
follow contribute to the final keff and its variance estimate.


API Reference
-------------

.. automodule:: orpheus.mc.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
