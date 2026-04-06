.. _theory-monte-carlo:

====================================
Monte Carlo Neutron Transport
====================================

.. contents:: Contents
   :local:
   :depth: 3

Overview
========

The Monte Carlo (MC) method solves the neutron transport equation by
**simulating individual neutron histories** stochastically.  Rather than
discretising the phase space (angle, energy, space) as deterministic methods
do, MC tracks particles through a geometry, sampling collision events from
probability distributions derived from the underlying cross sections.

ORPHEUS implements a **power iteration Monte Carlo** solver for the
:math:`k`-eigenvalue problem in a 2-D unit cell with periodic boundary
conditions.  The key algorithmic features are:

- **Woodcock delta-tracking** --- virtual-collision rejection that avoids
  distance-to-surface calculations, enabling simple geometry interfaces.
- **Analog absorption with fission weight adjustment** --- on absorption,
  the neutron weight is multiplied by :math:`\nu\Sigma_f / \Sigma_a` and
  reborn from the fission spectrum.
- **Russian roulette and splitting** --- population control to maintain
  a stable neutron count across power iteration cycles.
- **Batch statistics** --- the :math:`k`-effective is estimated from the
  cycle-to-cycle weight ratio with Central Limit Theorem uncertainty.

**Derivation sources.**  The analytical eigenvalues used for verification
are computed independently by the derivation scripts.  These are the
**source of truth** for all equations in this chapter:

- ``derivations/mc.py`` --- homogeneous eigenvalues from random walk
  probability (:func:`~derivations._eigenvalue.kinf_homogeneous`),
  heterogeneous eigenvalues from CP cylinder reference
  (:func:`~derivations._eigenvalue.kinf_from_cp`).
- ``derivations/_xs_library.py`` --- synthetic cross-section library
  (regions A/B/C/D in {1G, 2G, 4G}).

Every numerical value cited in this chapter was produced by these scripts.


Architecture: Base Geometry and Augmented Geometry
===================================================

The MC solver follows the same three-layer geometry pattern as the CP
(:class:`CPMesh`) and SN (:class:`SNMesh`) solvers:

1. **Base geometry** --- :class:`~geometry.mesh.Mesh1D` stores cell edges,
   material IDs, and the coordinate system.  It computes volumes and
   surfaces via :func:`~geometry.coord.compute_volumes_1d`.

2. **Augmented geometry** --- :class:`MCMesh` wraps a ``Mesh1D`` and adds
   the MC-specific **point-wise material lookup** needed by delta-tracking.
   The lookup is selected automatically from the mesh's coordinate system:

   - **Cartesian** --- ``x`` position mapped to cell index via
     ``np.searchsorted(edges, x, side="right") - 1``.
   - **Cylindrical** --- radial distance
     :math:`r = \sqrt{(x - x_c)^2 + (y - y_c)^2}` from the cell centre
     (:math:`x_c = y_c = \text{pitch}/2`), then mapped to cell index.

3. **Solver** --- :func:`solve_monte_carlo` receives an :class:`MCGeometry`
   protocol object (satisfied by :class:`MCMesh`, :class:`ConcentricPinCell`,
   or :class:`SlabPinCell`) and runs the power iteration.

.. code-block:: text

   Mesh1D (edges, mat_ids, coord)
       |
       v
   MCMesh (point-wise material_id_at + pitch)
       |
       v
   solve_monte_carlo() -> MCResult

**Design rationale (MT-20260406-001).**  Delta-tracking only needs to know
the material at the collision point --- no distance-to-surface calculation.
This makes the geometry interface minimal:

.. code-block:: python

   @runtime_checkable
   class MCGeometry(Protocol):
       pitch: float
       def material_id_at(self, x: float, y: float) -> int: ...

The :class:`MCGeometry` protocol is runtime-checkable, enabling both
:class:`MCMesh` (wrapping ``Mesh1D``) and direct geometry classes
(:class:`ConcentricPinCell`, :class:`SlabPinCell`) to be used
interchangeably.  Verified by
``test_mc_properties.py::test_mcmesh_satisfies_protocol``.

**MCMesh vs standalone geometry classes.**  The standalone classes
predate the unified geometry module.  :class:`MCMesh` integrates with the
same ``Mesh1D`` factories used by CP and SN (e.g.,
:func:`~geometry.factories.pwr_pin_equivalent`), enabling cross-method
comparisons on identical base geometry.  Verified:
``test_mc_properties.py::test_mcmesh_cylindrical_matches_concentric``
shows zero mismatches over 10,000 random sample points.


.. _mc-geometry-mismatch:

Geometry Mismatch: Square Cell vs Wigner-Seitz Cylinder (ERR-017)
------------------------------------------------------------------

When comparing MC (square cell, periodic BCs) against CP (Wigner-Seitz
cylinder, white BCs), the cell areas must match:

.. math::
   :label: ws-pitch

   p^2 = \pi R_{\text{cell}}^2
   \quad \Longrightarrow \quad
   p = R_{\text{cell}} \sqrt{\pi}

This is the convention used by :func:`~geometry.factories.pwr_pin_equivalent`
(``r_cell = pitch / sqrt(pi)``).

**ERR-017 investigation history.**  The pre-existing heterogeneous tests
used the formula ``pitch = r_cell * sqrt(pi) * 2``, which gives **four
times** the correct cell area.  The extra area was all moderator, which:

- For 1G 2-region: :math:`k_{\text{MC}} = 0.757` vs
  :math:`k_{\text{ref}} = 0.990` (24% systematic error).
- For 2G 2-region: the neutron population collapsed to zero (NaN) because
  the subcritical system with 4× moderator could not sustain 200
  neutrons/cycle.

**How it hid.**  All homogeneous tests passed (single material --- pitch
is irrelevant).  The heterogeneous tests were marked ``@pytest.mark.slow``
and may not have been run regularly.  The NaN z-scores failed the
``< 5.0`` assertion without indicating the error direction.

**Fix.**  Corrected to ``pitch = r_cell * sqrt(pi)``.  Even with matched
areas, the square corners contain extra moderator not present in the
circle, introducing a ~3--6% systematic bias.  The heterogeneous test
tolerance accounts for this:

.. math::
   :label: hetero-tolerance

   |k_{\text{MC}} - k_{\text{ref}}|
   < 5\sigma_{\text{MC}} + 0.06 \, k_{\text{ref}}

**Lesson (meta-lesson 8):**  When constructing geometry for cross-method
comparison, verify the cell area/volume matches between the two methods.
A factor-of-2 in a linear dimension is a factor-of-4 in area --- large
enough to change supercritical to subcritical.


The Monte Carlo Random Walk
============================

The random walk is the heart of the MC solver.  Each neutron undergoes a
sequence of free flights and collisions until it is absorbed.  The
implementation in :func:`solve_monte_carlo` (lines 269--328 of
``monte_carlo.py``) is annotated below.


Free-Flight Distance (Delta-Tracking)
--------------------------------------

Standard Monte Carlo tracks a neutron to the nearest material boundary,
then samples a collision within the current region.  **Woodcock
delta-tracking** [Woodcock1965]_ eliminates the distance-to-surface
calculation by introducing a fictitious **majorant** cross section:

.. math::
   :label: majorant

   \Sigma_{\text{maj},g} = \max_m \Sigma_{t,m,g}

where the maximum is over all materials :math:`m` for energy group
:math:`g`.  In the code::

    sig_t_max = np.zeros(ng)
    for mix in materials.values():
        sig_t_max = np.maximum(sig_t_max, mix.SigT)

Verified by ``test_mc_properties.py::test_majorant_computation``:
for 2G materials A (:math:`\Sigma_t = [0.50, 1.00]`) and B
(:math:`\Sigma_t = [0.60, 2.00]`), the majorant is :math:`[0.60, 2.00]`.

The free-flight distance is sampled from an exponential distribution with
rate :math:`\Sigma_{\text{maj},g}`:

.. math::
   :label: free-flight

   s = -\frac{\ln \xi}{\Sigma_{\text{maj},g}},
   \qquad \xi \sim U(0,1)

This gives :math:`E[s] = 1/\Sigma_{\text{maj},g}` and
:math:`\text{Var}[s] = 1/\Sigma_{\text{maj},g}^2`.  Both moments are
verified by ``test_mc_gaps.py::test_free_path_exponential``.

In the code::

    free_path = -np.log(rng.random()) / sig_t_max[ig]

Delta-Tracking Equivalence Proof
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Claim:**  Sampling free flights from the majorant and rejecting virtual
collisions recovers the correct transport kernel.

**Proof.**  Decompose the total cross section at each point as real +
virtual:

.. math::
   :label: decompose

   \Sigma_{\text{maj},g}
   = \Sigma_{t,g}(\mathbf{r}) + \Sigma_{\text{virtual},g}(\mathbf{r})

The probability of a neutron at :math:`\mathbf{r}_0` reaching distance
:math:`s` without any (real or virtual) collision is:

.. math::

   P(\text{no collision up to } s)
   = e^{-\Sigma_{\text{maj},g} \, s}

At each collision site :math:`\mathbf{r}_0 + s\hat{\Omega}`, the collision
is real with probability:

.. math::

   P(\text{real} | \text{collision at } s)
   = \frac{\Sigma_{t,g}(\mathbf{r}_0 + s\hat{\Omega})}{\Sigma_{\text{maj},g}}

The joint probability of first real collision at distance :math:`s` is:

.. math::

   p(s) \, ds
   = \Sigma_{\text{maj},g} \, e^{-\Sigma_{\text{maj},g} \, s}
     \cdot \frac{\Sigma_{t,g}(\mathbf{r}(s))}{\Sigma_{\text{maj},g}} \, ds
   = \Sigma_{t,g}(\mathbf{r}(s)) \, e^{-\Sigma_{\text{maj},g} \, s} \, ds

For a homogeneous medium (:math:`\Sigma_{t,g}` constant along the path),
this simplifies to the familiar exponential attenuation
:math:`\Sigma_t e^{-\Sigma_t s}` since the virtual collisions (which
preserve direction and continue the flight) effectively reduce the rate
from :math:`\Sigma_{\text{maj}}` to :math:`\Sigma_t`.

**Key property:**  Virtual collisions preserve the flight direction.  This
is essential --- if direction were resampled on virtual collisions, the
path would become a random walk instead of a straight line, and the
exponential attenuation would be violated.  Verified by
``test_mc_gaps.py::test_virtual_collision_preserves_direction``.

In the code::

    # Virtual or real collision?
    if sig_v / sig_t_max[ig] >= rng.random():
        virtual_collision = True    # direction unchanged
    else:
        virtual_collision = False   # process real collision

The virtual collision probability at the collision site is:

.. math::

   P_{\text{virtual}} = \frac{\Sigma_{\text{maj},g} - \Sigma_{t,g}}
                              {\Sigma_{\text{maj},g}}

Verified by ``test_mc_properties.py::test_delta_tracking_virtual_probability``
and ``test_mc_properties.py::test_delta_tracking_homogeneous_no_virtual``
(homogeneous medium → zero virtual collisions).

**Efficiency.**  The virtual collision fraction in material :math:`m` is
:math:`1 - \Sigma_{t,m,g}/\Sigma_{\text{maj},g}`.  For the 2G library
with materials A and B: in the fast group, B has
:math:`\Sigma_t = 0.60 = \Sigma_{\text{maj}}` (0% virtual), while A has
:math:`\Sigma_t = 0.50` (17% virtual).  In the thermal group, A has
:math:`\Sigma_t = 1.00` (50% virtual).  High virtual rates waste
computation; see MT-20260403-006.


Direction Sampling
------------------

At each **real** collision that results in scattering, a new flight
direction is sampled.  The solver uses a 2-D projection:

.. math::
   :label: direction-sampling

   \theta &= \pi \, \xi_1, \qquad
   \phi = 2\pi \, \xi_2 \\
   \Omega_x &= \sin\theta \, \cos\phi, \qquad
   \Omega_y = \sin\theta \, \sin\phi

where :math:`\xi_1, \xi_2 \sim U(0,1)`.  In the code::

    theta = np.pi * rng.random()
    phi = 2.0 * np.pi * rng.random()
    dir_x = np.sin(theta) * np.cos(phi)
    dir_y = np.sin(theta) * np.sin(phi)

.. _mc-direction-sampling-err018:

Direction Sampling Limitation (ERR-018)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**This is NOT isotropic sampling on the unit sphere.**  True isotropic
sampling requires :math:`\theta = \arccos(1 - 2\xi_1)` to account for the
solid-angle Jacobian :math:`\sin\theta \, d\theta \, d\phi`.

**Mathematical analysis.**  For uniform :math:`\theta`:

.. math::

   E[\sin^2\theta]
   = \frac{1}{\pi} \int_0^{\pi} \sin^2\theta \, d\theta
   = \frac{1}{2}

For true isotropic sampling:

.. math::

   E[\sin^2\theta]_{\text{iso}}
   = \frac{1}{2} \int_0^{\pi} \sin^3\theta \, d\theta
   = \frac{2}{3}

Therefore:

.. math::

   E[\Omega_x^2]
   = E[\sin^2\theta] \cdot E[\cos^2\phi]
   = \frac{1}{2} \cdot \frac{1}{2}
   = \frac{1}{4}
   \qquad\text{(uniform } \theta\text{)}

vs :math:`1/3` for isotropic.  The uniform-:math:`\theta` formula
shortens the average 2-D step length by :math:`\sqrt{1/4} / \sqrt{1/3}
= \sqrt{3/4} \approx 0.87`, i.e., ~13% shorter.

**Why it's retained.**  The formula matches the original MATLAB
``monteCarloPWR.m``.  Since the solver tracks only the 2-D projection
:math:`(\Omega_x, \Omega_y)` and uses periodic BCs on a square cell,
the non-isotropic sampling changes the effective mean free path but does
not invalidate the eigenvalue --- the bias is absorbed into the geometry
scaling.  Fixing would break MATLAB compatibility and require
re-establishing all reference values (MT-20260406-006).

Verified by ``test_mc_properties.py::test_direction_sampling``:
:math:`E[\Omega_x^2] = 0.25 \pm z < 5`.


Periodic Boundary Conditions
-----------------------------

The unit cell is a square of side length :math:`p` (``pitch``).  When a
neutron exits the cell, its position wraps via the modulo operation:

.. math::
   :label: periodic-bc

   x \leftarrow x \bmod p, \qquad y \leftarrow y \bmod p

In the code::

    nx_ = nx_ % pitch
    ny_ = ny_ % pitch

Python's modulo returns non-negative for positive ``pitch``.  Verified by
``test_mc_properties.py::test_periodic_bc_wrapping`` with edge cases
including negative positions and exact-boundary positions.


Collision Physics
==================

At each **real** collision site, two outcomes are possible: scattering or
absorption.  The decision is based on the branching ratio:

.. math::
   :label: branching

   P(\text{scatter}) = \frac{\Sigma_{s,g}}{\Sigma_{t,g}}, \qquad
   P(\text{absorb}) = \frac{\Sigma_{a,g}}{\Sigma_{t,g}}

where :math:`\Sigma_{a,g} = \Sigma_{f,g} + \Sigma_{c,g} + \Sigma_{L,g}`.
In the code::

    sig_a = mat.SigF[ig] + mat.SigC[ig] + mat.SigL[ig]
    sig_s_row = sig_s_dense[mat_id][ig, :]
    sig_s_sum = sig_s_row.sum()
    sig_t = sig_a + sig_s_sum

Verified by ``test_mc_properties.py::test_scattering_branching_ratio``:
for region A (1G), :math:`P(\text{scatter}) = 0.5/1.0 = 0.5`, confirmed
by 100k samples.  XS consistency (:math:`\Sigma_t = \Sigma_a + \Sigma_s`)
verified for all 12 materials by ``test_mc_gaps.py::test_xs_consistency_in_solver``.

**Cross-section preprocessing.**  The solver precomputes dense scattering
rows for all materials::

    sig_s_dense = {}
    for mat_id, mix in materials.items():
        rows = np.array(mix.SigS[0].todense())  # (ng, ng)
        sig_s_dense[mat_id] = rows

The ``SigS[0]`` index selects the **isotropic (P0) scattering matrix**.
Higher-order anisotropy (P1, etc.) is not used by the MC solver.  The
``todense()`` conversion is necessary because the source storage is sparse
(:class:`scipy.sparse.csr_matrix`).


Scattering: Group Transfer
--------------------------

On scattering, the outgoing energy group :math:`g'` is sampled from the
cumulative distribution of the scattering row:

.. math::
   :label: scattering-cdf

   P(g' \le G \mid \text{scatter from } g)
   = \frac{\sum_{g'=0}^{G} \Sigma_{s,g \to g'}}
          {\sum_{g'} \Sigma_{s,g \to g'}}

In the code::

    cum_s = np.cumsum(sig_s_row)
    ig = np.searchsorted(cum_s, rng.random() * sig_s_sum)
    ig = min(ig, ng - 1)

.. note::

   **Convention (anti-ERR-002):**  The code uses ``sig_s_dense[mat_id][ig, :]``
   which is **row** ``ig`` of the scattering matrix.  The codebase convention
   is :math:`\Sigma_s[\text{from}, \text{to}]`, so row ``ig`` gives transfer
   *from* group ``ig`` to all groups.

   For an asymmetric 2G matrix with zero upscatter
   (:math:`\Sigma_s[1,0] = 0`), a neutron in group 1 can only scatter to
   group 1.  If the code used the column (transpose), upscatter would appear.
   Verified by ``test_mc_properties.py::test_scattering_convention_no_upscatter``:
   10,000 samples from group 1, zero upscatter events.

   Scattering fractions verified by ``test_mc_properties.py::test_scattering_cdf_sampling``:
   for region A (2G), the expected fast-to-fast fraction is
   :math:`0.38/0.48 = 79.2\%`, confirmed by 100k samples within :math:`z < 5`.


Absorption: Fission Weight Adjustment
--------------------------------------

On absorption, rather than killing the neutron and sampling a new fission
site, the solver uses **analog absorption with weight adjustment**.  The
neutron's weight is multiplied by the expected number of fission neutrons
per absorption:

.. math::
   :label: fission-weight

   w \leftarrow w \cdot \frac{\nu\Sigma_{f,g}}{\Sigma_{a,g}}

In the code::

    sig_p = mat.SigP[ig]           # = nu * Sigma_f
    sig_a = mat.SigF[ig] + mat.SigC[ig] + mat.SigL[ig]
    w *= sig_p / sig_a

**1-group derivation from random walk probability.**  Consider a neutron
undergoing a random walk in a homogeneous 1G infinite medium.  At each
collision:

1. The neutron scatters with probability :math:`c = \Sigma_s / \Sigma_t`.
2. After :math:`n` scattering events, it is absorbed with probability
   :math:`(1-c)`.
3. On absorption, it produces :math:`\nu\Sigma_f / \Sigma_a` fission
   neutrons.

The probability of reaching the :math:`n`-th collision is :math:`c^n`.
The expected multiplication per generation is:

.. math::

   k = (1-c) \cdot \frac{\nu\Sigma_f}{\Sigma_a} \cdot \sum_{n=0}^{\infty} c^n
     = (1-c) \cdot \frac{\nu\Sigma_f}{\Sigma_a} \cdot \frac{1}{1-c}
     = \frac{\nu\Sigma_f}{\Sigma_a}

This is exact for 1G and equals the weight factor applied at each
absorption.  For region A (1G): :math:`\Sigma_a = 0.2 + 0.3 = 0.5`,
:math:`\nu\Sigma_f = 2.5 \times 0.3 = 0.75`, so :math:`k = 1.5`.
The numerical value is produced by ``derivations/mc.py`` via
:func:`~derivations._eigenvalue.kinf_homogeneous`; the derivation above
is presented for pedagogical context.

Verified by ``test_mc_properties.py::test_fission_weight_adjustment``
(hand-calculated weight factor = 1.5) and
``test_mc_properties.py::test_1g_homogeneous_deterministic``
(:math:`\sigma = 0` because every neutron sees identical XS).

**Non-fissile materials.**  When :math:`\nu\Sigma_f = 0` (materials B, C,
D), the weight goes to zero: :math:`w \leftarrow 0`.  Russian roulette
then terminates the neutron.  Verified by
``test_mc_properties.py::test_fission_weight_non_fissile`` and
``test_mc_gaps.py::test_absorption_nonfissile_zeroes_weight``.

**Fission spectrum resampling.**  After weight adjustment, the neutron is
reborn in a new energy group from the fission spectrum CDF:

.. math::
   :label: chi-sampling

   g_{\text{new}} = \text{searchsorted}\!\bigl(
     \text{cumsum}(\chi), \, \xi \bigr),
   \qquad \xi \sim U(0,1)

For 4G region A: :math:`\chi = [0.60, 0.35, 0.05, 0.00]`.  Verified by
``test_mc_properties.py::test_chi_spectrum_sampling``: 100k samples match
expected group fractions within :math:`z < 5` per group.  Group 3 gets
exactly zero samples.

**Limitation:** The fission spectrum :math:`\chi` is taken from a single
material (``_any_mat``), not the material at the absorption site.  For
heterogeneous problems with different :math:`\chi` vectors, this is a
simplification.  All test materials use the same :math:`\chi`.


Population Control
===================

Russian Roulette
-----------------

After a complete random walk (free flights until absorption), neutrons with
reduced weight are subjected to Russian roulette.

.. math::
   :label: roulette-prob

   P_{\text{kill}} = 1 - \frac{w}{w_0}

where :math:`w` is the post-walk weight and :math:`w_0` is the weight at
cycle start.  The outcome is:

.. math::
   :label: roulette-restore

   w_{\text{after}} = \begin{cases}
   0   & \text{with probability } P_{\text{kill}} \\
   w_0 & \text{with probability } 1 - P_{\text{kill}}
   \end{cases}

In the code::

    terminate_p = 1.0 - weight[i_n] / weight0[i_n]
    if terminate_p >= rng.random():
        weight[i_n] = 0.0
    elif terminate_p > 0:
        weight[i_n] = weight0[i_n]

Weight Conservation Proof
~~~~~~~~~~~~~~~~~~~~~~~~~~

The expected weight after roulette equals the weight before:

.. math::
   :label: roulette-conservation

   E[w_{\text{after}}]
   = (1 - P_{\text{kill}}) \cdot w_0 + P_{\text{kill}} \cdot 0
   = \frac{w}{w_0} \cdot w_0
   = w
   = w_{\text{before}}

Verified statistically by
``test_mc_properties.py::test_roulette_weight_conservation`` (100k neutrons
with :math:`w/w_0 = 0.3`, mean weight after roulette matches :math:`0.3`
within :math:`z < 5`) and by
``test_mc_properties.py::test_roulette_restore_weight`` (surviving neutrons
have weight exactly :math:`w_0`).

**Supercritical edge case.**  When :math:`w > w_0` (fission weight
adjustment in supercritical system, e.g., :math:`\nu\Sigma_f/\Sigma_a =
1.5`), then :math:`P_{\text{kill}} = 1 - w/w_0 < 0`.  The ``terminate_p
>= rng.random()`` condition is never true (rng returns :math:`[0,1)`), and
``terminate_p > 0`` is also false, so the weight is **unchanged** at
:math:`w`.  This is correct --- supercritical neutrons should not be
rouletted.  Verified by
``test_mc_gaps.py::test_roulette_supercritical_preserves_weight``.


Splitting
---------

Neutrons with weight :math:`w > 1` are split into :math:`N` copies, each
with weight :math:`w/N`.  The number of copies is:

.. math::
   :label: splitting

   N = \begin{cases}
   \lfloor w \rfloor + 1 & \text{with probability } w - \lfloor w \rfloor \\
   \lfloor w \rfloor     & \text{with probability }
                           \lfloor w \rfloor + 1 - w
   \end{cases}

Splitting Weight Conservation Proof
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Exact conservation:**  Each copy gets weight :math:`w/N`, so total
weight = :math:`N \cdot (w/N) = w`.  No stochastic noise.

**Expected copies:**  The stochastic rounding gives:

.. math::

   E[N] = \lfloor w \rfloor \cdot (1 - (w - \lfloor w \rfloor))
        + (\lfloor w \rfloor + 1) \cdot (w - \lfloor w \rfloor)

Let :math:`f = w - \lfloor w \rfloor` and :math:`n = \lfloor w \rfloor`:

.. math::

   E[N] = n(1-f) + (n+1)f = n + f = w

So :math:`E[N] = w`, and total weight per copy is :math:`w/N`, meaning
the expected number of particles times weight per particle equals :math:`w`.

Verified by ``test_mc_properties.py::test_splitting_weight_conservation``
(exact total weight) and ``test_mc_gaps.py::test_splitting_copy_count``
(statistical: :math:`P(N=4) = 0.7` for :math:`w = 3.7`).


Eigenvalue Estimation
======================

keff from Weight Ratio
-----------------------

The :math:`k`-effective for each cycle is estimated as:

.. math::
   :label: keff-cycle

   k_{\text{cycle}} = \frac{\sum_{n} w_n^{\text{end}}}{\sum_{n} w_n^{0}}

where :math:`w_n^{0}` are the normalised starting weights and
:math:`w_n^{\text{end}}` are the weights after the random walk, roulette,
and splitting.  In the code::

    keff_cycle = weight[:n_neutrons].sum() / weight0.sum()

Verified by ``test_mc_gaps.py::test_keff_cycle_estimator``:
:math:`[1.5, 0.0, 1.2, 0.8, 2.0] / [1,1,1,1,1] = 5.5/5.0 = 1.1`.

**Weight normalisation.**  At the start of each cycle, weights are
rescaled so that :math:`\sum w_n = N_{\text{source}}`::

    total_weight = weight[:n_neutrons].sum()
    weight[:n_neutrons] *= params.n_neutrons / total_weight

Verified by ``test_mc_gaps.py::test_weight_normalization_consistency``.


Batch Statistics
----------------

The final :math:`k`-effective is the **cumulative mean** of
:math:`M` active cycle values:

.. math::
   :label: keff-mean

   \bar{k}_M = \frac{1}{M} \sum_{m=1}^{M} k_m

The standard deviation of the mean:

.. math::
   :label: sigma-keff

   \sigma_M = \sqrt{\frac{1}{M(M-1)} \sum_{m=1}^{M} (k_m - \bar{k}_M)^2}

By the **Central Limit Theorem**, :math:`\sigma_M \sim 1/\sqrt{M}`.
Verified by ``test_mc_convergence.py::test_sigma_scales_with_sqrt_n``:
the ratio :math:`\sigma(400)/\sigma(100)` is within [0.25, 0.75] of the
theoretical 0.5.

Verified term-by-term by
``test_mc_properties.py::test_batch_statistics_formula``: for
:math:`k_{\text{active}} = [1.0, 1.1, 0.9, 1.05]`, the hand-calculated
cumulative means and sigmas match the solver's formula to machine precision.

**Source convergence.**  The first :math:`N_{\text{inactive}}` cycles are
discarded to allow the fission source to converge from the initial
(uniform) guess.  Only :math:`N_{\text{active}}` cycles contribute.
Verified by ``test_mc_convergence.py::test_inactive_cycles_reduce_bias``.


Flux Tally (Known Limitation)
------------------------------

The solver accumulates a scattering detector during the random walk::

    detect_s[ig] += w / sig_s_sum

at each **scattering** event in group :math:`g` (line 307).

.. warning::

   **This is not a proper flux estimator** (MT-20260406-007).  A correct
   collision estimator would accumulate :math:`w / \Sigma_{t,g}` at every
   **real** collision (both scattering and absorption).  The current tally:

   - Misses absorption events entirely
   - Weights by :math:`1/\Sigma_s` instead of :math:`1/\Sigma_t`
   - Produces negative ``flux_per_lethargy`` values because the energy grid
     is high-to-low (:math:`\Delta u = \ln(E_{g+1}/E_g) < 0`)

   The eigenvalue :math:`k_{\text{eff}}` is computed from the weight ratio
   :eq:`keff-cycle` and is **not affected** by this tally.


Solver Parameters and Results
===============================

.. list-table:: :class:`MCParams` fields
   :header-rows: 1
   :widths: 20 15 45

   * - Field
     - Default
     - Description
   * - ``n_neutrons``
     - 100
     - Source neutrons per cycle
   * - ``n_inactive``
     - 100
     - Inactive cycles (source convergence, not tallied)
   * - ``n_active``
     - 2000
     - Active cycles (tallied for :math:`k_{\text{eff}}`)
   * - ``pitch``
     - 3.6
     - Unit cell side length (cm)
   * - ``seed``
     - None
     - RNG seed (None = random)
   * - ``geometry``
     - None
     - :class:`MCGeometry` (None → default slab)

.. list-table:: :class:`MCResult` fields
   :header-rows: 1
   :widths: 20 20 40

   * - Field
     - Shape
     - Description
   * - ``keff``
     - scalar
     - Final cumulative mean :math:`\bar{k}_M`
   * - ``sigma``
     - scalar
     - Standard deviation of the mean :math:`\sigma_M`
   * - ``keff_history``
     - ``(n_active,)``
     - Cumulative mean at each active cycle
   * - ``sigma_history``
     - ``(n_active,)``
     - Cumulative sigma at each active cycle
   * - ``flux_per_lethargy``
     - ``(ng,)``
     - Scattering detector / :math:`\Delta u` (see limitation above)
   * - ``eg_mid``
     - ``(ng,)``
     - Mid-group energies (eV)
   * - ``elapsed_seconds``
     - scalar
     - Wall-clock time


Analytical Verification
========================

Homogeneous Infinite Medium
----------------------------

For a single material filling the entire cell, the eigenvalue is derived
from the collision kernel.

**1-group:**  From the random walk analysis above (:eq:`fission-weight`):

.. math::
   :label: kinf-1g

   k_\infty = \frac{\nu\Sigma_f}{\Sigma_a}

For region A (1G): :math:`k = 0.75/0.5 = 1.5`.

**Multi-group:**  The eigenvalue is the dominant eigenvalue of

.. math::
   :label: kinf-mg

   k_\infty = \lambda_{\max}\!\left(\mathbf{A}^{-1} \mathbf{F}\right)

where :math:`\mathbf{A} = \text{diag}(\Sigma_t) - \Sigma_s^T` (the
transpose converts from the ``[from, to]`` storage convention to the
in-scatter form needed by the balance equation; see scattering convention
note in :ref:`theory-monte-carlo`) and
:math:`\mathbf{F} = \chi \otimes (\nu\Sigma_f)`.  Computed by
:func:`~derivations._eigenvalue.kinf_homogeneous`.

.. list-table:: Homogeneous eigenvalues
   :header-rows: 1
   :widths: 30 25 25

   * - Case
     - :math:`k_{\text{ref}}`
     - Tolerance
   * - 1G (``mc_cyl1D_1eg_1rg``)
     - 1.500000
     - :math:`\sigma = 0` (exact)
   * - 2G (``mc_cyl1D_2eg_1rg``)
     - 1.875000
     - :math:`z < 5`
   * - 4G (``mc_cyl1D_4eg_1rg``)
     - 1.487762
     - :math:`z < 5`


Heterogeneous Pin Cell
-----------------------

For multi-region problems, the reference eigenvalue comes from the **CP
cylinder derivation** (:func:`~derivations._eigenvalue.kinf_from_cp`).

.. list-table:: MC heterogeneous verification cases
   :header-rows: 1
   :widths: 28 8 8 18 18

   * - Case
     - G
     - Reg
     - :math:`k_{\text{ref}}`
     - Tolerance
   * - ``mc_cyl1D_1eg_2rg``
     - 1
     - 2
     - 0.989750
     - :math:`5\sigma + 0.06 k`
   * - ``mc_cyl1D_2eg_2rg``
     - 2
     - 2
     - 0.739887
     - :math:`5\sigma + 0.06 k`
   * - ``mc_cyl1D_1eg_4rg``
     - 1
     - 4
     - 0.806751
     - :math:`5\sigma + 0.06 k`
   * - ``mc_cyl1D_2eg_4rg``
     - 2
     - 4
     - 0.502725
     - :math:`5\sigma + 0.06 k`
   * - ``mc_cyl1D_4eg_2rg``
     - 4
     - 2
     - 0.648637
     - :math:`5\sigma + 0.06 k`
   * - ``mc_cyl1D_4eg_4rg``
     - 4
     - 4
     - 0.446472
     - :math:`5\sigma + 0.06 k`


Verification Suite
===================

The MC solver is verified by **55 tests** across four levels:

.. list-table:: Verification test summary
   :header-rows: 1
   :widths: 10 8 52

   * - Level
     - Count
     - Description
   * - L0
     - 31
     - Term-level isolation of each algorithmic component
   * - L1
     - 18
     - Eigenvalue: {1,2,4}G × {1,2,4}-region
   * - L2
     - 3
     - Convergence: :math:`\sigma \sim 1/\sqrt{N}`, bias, inactive cycles
   * - XV
     - 2
     - Cross-verification: MC vs CP (cylinder, slab)

**Determinism.**  Verified by ``test_mc_gaps.py::test_seed_reproducibility``
(same seed → identical results) and ``test_different_seeds_differ`` (different
seeds → different histories).


Failure Mode Coverage
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45

   * - Failure mode
     - Tests
   * - Sign flip
     - L0-MC-004 (delta-tracking), L0-MC-016 (collision fraction)
   * - Variable swap
     - L0-MC-005, L0-MC-014 (SigS^T)
   * - Missing factor
     - L0-MC-006 (fission weight), L0-MC-015 (free-path)
   * - Factor error
     - L0-MC-012 (direction moments)
   * - Index error
     - L0-MC-005, L0-MC-007 (searchsorted)
   * - Convention drift
     - L0-MC-013 (batch stats), XS consistency
   * - 1G degeneracy
     - L1: 2G/4G high-stats, 4G fast guard
   * - Homogeneous degeneracy
     - L1: heterogeneous {2,4}-region
   * - Geometry mismatch
     - ERR-017 fixed, pitch tested


Design Decisions
=================

**Delta-tracking vs surface-tracking (MT-20260403-001).**  Delta-tracking
eliminates distance-to-surface calculations, simplifying the geometry
interface to ``material_id_at(x, y)``.  The cost is virtual collisions in
low-density regions.

**Analog absorption vs implicit capture (MT-20260403-002).**  Implicit
capture reduces weight at every collision (:math:`w \leftarrow w \cdot
\Sigma_s/\Sigma_t`).  Analog absorption terminates the walk at each
absorption event, producing exactly one surviving particle with weight
:math:`w \cdot \nu\Sigma_f/\Sigma_a`.  Combined with roulette and
splitting, this maintains a stable population.

**Uniform theta sampling (ERR-018, MT-20260406-006).**  Retained from
MATLAB original.  See :ref:`mc-direction-sampling-err018`.

**1-group is degenerate.** :math:`k = \nu\Sigma_f/\Sigma_a` regardless of
code correctness.  The suite tests at 1, 2, AND 4 groups.


Limitations and Future Work
=============================

.. list-table::
   :header-rows: 1
   :widths: 20 50

   * - Tracking ID
     - Description
   * - MT-20260403-004
     - Python neutron loop performance (inner loop not vectorised)
   * - MT-20260403-005
     - Heterogeneous independent reference (high-statistics MC)
   * - MT-20260403-006
     - Per-region majorant for efficiency
   * - MT-20260406-005
     - Solver ignores Sig2 (n,2n) reactions
   * - MT-20260406-006
     - Direction sampling not isotropic (ERR-018)
   * - MT-20260406-007
     - Flux tally is not a proper estimator


References
==========

.. [Woodcock1965] E.R. Woodcock, T. Murphy, P.J. Hemmings, and
   T.C. Longworth, "Techniques used in the GEM code for Monte Carlo
   neutronics calculations in reactors and other systems of complex
   geometry," *Proc. Conf. Applications of Computing Methods to Reactor
   Problems*, ANL-7050, 1965.

.. [Lux1991] I. Lux and L. Koblinger, *Monte Carlo Particle Transport
   Methods: Neutron and Photon Calculations*, CRC Press, 1991.

.. [Brown2005] F.B. Brown, "Fundamentals of Monte Carlo Particle
   Transport," LA-UR-05-4983, Los Alamos National Laboratory, 2005.
