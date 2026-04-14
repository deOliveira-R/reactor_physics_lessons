.. _theory-verification:

==================
Verification Suite
==================

.. contents:: Contents
   :local:
   :depth: 3


Overview
========

ORPHEUS verifies every transport solver against analytical reference solutions
derived from each method's own mathematical equations using SymPy.  Each
derivation is **self-contained**: it starts from the solver's formulation and
derives the expected eigenvalue independently.  No cross-verification (comparing
one solver against another) is used — each solver's verification stands on its
own, as if every other solver were deleted.

The same code that produces the LaTeX equations in this chapter also produces
the reference values consumed by the ``pytest`` test suite.  This is the
**single source of truth**: equations in the documentation cannot drift from
the values in the tests because both come from the same ``derivations/``
package.


Architecture
============

Three interlinked systems flow from one source:

.. code-block:: text

   derivations/                  SymPy derivations (single source of truth)
        │
        ├──→ tests/              pytest imports reference values, runs solvers
        │
        └──→ docs/_generated/    RST fragments with LaTeX + results tables


.. _reference-values-lazy-registry:

Reference-Values Registry (Eager vs Lazy)
------------------------------------------

``orpheus/derivations/reference_values.py`` is the central registry
that every test and every Sphinx page reads from. Its loader is
split into two tiers to avoid a circular import that would otherwise
make ``import orpheus.derivations`` pull in the entire solver
package:

**Eager tier — analytical cases**
   Loaded at import time by walking
   ``homogeneous.py``, ``cp_slab.py``, ``cp_cylinder.py``,
   ``cp_sphere.py``, and ``diffusion.py``. These modules use only
   NumPy, SciPy, and SymPy — no solver imports — so loading them
   is free and fast. The 9-case CP grids (:ref:`nine-case-cp-grid`),
   the homogeneous matrix eigenvalues, and the diffusion buckling
   cases all land here.

**Lazy tier — Richardson-extrapolated cases**
   Deferred until first ``get("case_name")`` call. These are the
   SN heterogeneous slab, the MOC heterogeneous pin cell, and the
   diffusion fuel + reflector reference — each of which requires
   running the actual solver at four mesh levels to estimate a
   converged :math:`k` (see :ref:`richardson-extrapolation`).
   Loading them eagerly would force
   ``import orpheus.derivations`` to transitively import
   ``orpheus.sn``, ``orpheus.moc``, and ``orpheus.diffusion_1d``,
   creating a circular dependency (because those modules, in turn,
   import from ``orpheus.derivations`` to read their reference
   values).

The mechanism is a module-level dict ``_LAZY_LOADERS`` keyed by
case name. ``get("case_name")`` checks the eager table first; if
absent, it looks up a loader in ``_LAZY_LOADERS``, calls it once,
caches the result, and returns it. Subsequent calls hit the cache.

The practical effect is that **every test pays only for the
reference values it actually uses**. Running ``pytest -m l0``
(which uses only analytical cases) avoids the 15-minute Richardson
recompute entirely; the lazy tier only activates when a test
parametrizes over an SN / MOC / diffusion heterogeneous case.

A JSON-backed cache on disk
(``orpheus/derivations/_richardson_cache.json``) persists the
lazy-tier results across sessions so the Richardson run happens
at most once per clean checkout, not once per pytest invocation.

.. _synthetic-xs-library:

Cross-Section Library
---------------------

All verification cases use **abstract synthetic cross sections** from
``derivations/_xs_library.py``.  Four regions are defined, each with
{1G, 2G, 4G} variants:

- **Region A** (fissile): fuel-like, with fission and moderate scattering
- **Region B** (scatterer): moderator-like, strong scattering, no fission
- **Region C** (absorber): cladding-like, moderate absorption
- **Region D** (gap): thin, very low opacity

All cross sections satisfy the consistency relation
:math:`\Sigma_t = \Sigma_c + \Sigma_f + \sum_{g'} \Sigma_{s,g \to g'}`.

P1 scattering anisotropy
~~~~~~~~~~~~~~~~~~~~~~~~

Every region carries a P1 scattering matrix in addition to the P0
isotropic form. The P1 matrix is built as
:math:`\Sigma_{s,1}(g \to g') = \bar\mu \cdot \Sigma_{s,0}(g \to g')`,
where :math:`\bar\mu` is the mean lab-frame scattering cosine set
per region to reflect the target nuclide's mass:

.. csv-table::
   :header: Region, Role, Nuclide analogue, :math:`\bar\mu`, Rationale
   :widths: 10, 22, 22, 10, 36

   A, fissile, "heavy uranium/plutonium", 0.05, "A ~ 235 gives :math:`\bar\mu_{cm \to lab} \approx 2/(3A) \approx 0.003`; bumped to 0.05 to expose the anisotropic-correction code path without making it dominant."
   B, moderator, "light H / H\ :sub:`2`\ O", 0.60, "A = 1 is the textbook forward-peaked limit, :math:`\bar\mu_{lab} = 2/3` for s-wave elastic on H. Rounded to 0.60."
   C, cladding, "intermediate Zr-90", 0.10, "A = 90 places it between fuel and moderator; 0.10 gives a mild forward peak consistent with Zr's :math:`2/(3A) \approx 0.007` lab-frame, again slightly amplified for test coverage."
   D, gap, "light He / void", 0.30, "Low-density gas with light nuclei. 0.30 picks a middle value that is neither fully isotropic nor dominated by the streaming peak."

These values are physically motivated but NOT tuned to any specific
isotope. The point is to give every transport solver a non-trivial
:math:`\Sigma_{s,1}` matrix so that the P1 correction term is
exercised by the verification suite; the absolute values are not
intended to match real nuclear data. Production runs always use the
real library from ``orpheus/data/micro_xs/``.

The definitive values and per-group arrays live in
``orpheus/derivations/_xs_library.py`` (see the ``_MU_BAR`` dict at
the top of the file). Changing them there automatically propagates
to every verification case because all derivation modules import
``make_mixture`` from the same module.

Region layouts:

- **1 region**: A only (homogeneous fissile medium)
- **2 regions**: A + B (fuel + moderator)
- **4 regions**: A + D + C + B (fuel + gap + cladding + moderator)


Verification Methodology
=========================

Each verification case defines three things:

1. **Cross sections** — synthetic macroscopic data for abstract regions
   (not specific materials).  This isolates the numerical method from
   the cross-section processing pipeline.

2. **Analytical eigenvalue** — derived from the solver's own equations
   using SymPy (symbolic where possible, numerical for special-function
   kernels like E₃ and Ki₄).

3. **Tolerance** — principled, not arbitrary.  The tolerance is bounded
   by the dominant error source:

.. list-table:: Tolerance Rationale
   :header-rows: 1
   :widths: 20 15 20 45

   * - Method
     - Tolerance
     - Error source
     - Rationale
   * - Homogeneous
     - < 10⁻¹²
     - FP arithmetic
     - Direct ``numpy.eigvals`` of small dense matrix
   * - CP slab
     - < 10⁻⁶
     - Power iteration
     - Solver keff_tol=10⁻⁷ bounds the error
   * - CP cylinder
     - < 10⁻⁵
     - Power iteration + Ki₄ interpolation
     - Solver keff_tol=10⁻⁶ plus Ki₄ table resolution
   * - SN (homogeneous)
     - < 10⁻⁸
     - Power iteration
     - Flat flux is exact in DD; only iteration error
   * - SN (heterogeneous)
     - O(h²)
     - Spatial discretisation
     - Richardson extrapolation from own mesh convergence
   * - MOC (homogeneous)
     - < 10⁻⁴
     - Ray spacing + iteration
     - Flat-source exact; convergence limited by ray density
   * - MC
     - z < 5σ
     - Statistical
     - Central Limit Theorem; σ ~ 1/√N_active
   * - Diffusion
     - O(h²)
     - FD spatial discretisation
     - Analytical buckling eigenvalue (bare) or Richardson (reflected)


Reference Case Types
====================

Analytical
----------

The eigenvalue is computed in closed form or as the eigenvalue of a
finite-dimensional matrix derived symbolically by SymPy.  No numerical
solver is involved.

**Examples**: homogeneous infinite medium (matrix eigenvalue), diffusion
bare slab (buckling eigenvalue), SN/MOC/MC homogeneous (each derived from
the solver's own equations showing that the infinite-medium eigenvalue
is the exact solution).

Semi-Analytical
---------------

The reference involves a special function (E₃, Ki₃/Ki₄) evaluated to
integrator precision, followed by a finite matrix eigenvalue.  The only
approximation is the numerical quadrature for the special function, which
is controlled to machine precision via ``scipy.special.expn`` (E₃) or
high-resolution lookup tables (Ki₃/Ki₄ with 20,000 points).

**Examples**: all CP slab and CP cylinder cases.  The CP matrix is
exact for the collision probability formulation; the eigenvalue is
a finite matrix problem.

.. _richardson-extrapolation:

Richardson-Extrapolated
-----------------------

For discretised solvers on heterogeneous problems, the reference is the
converged limit of the solver itself, estimated by running at 4 mesh
refinement levels and extrapolating assuming O(h²) convergence:

.. math::

   k_{\rm ref} \approx k_h + \frac{k_h - k_{2h}}{2^p - 1}

where :math:`p = 2` for diamond-difference and finite-difference schemes.

This is legitimate formal verification of the **convergence rate** — it
tests that the implementation converges at the theoretically expected order,
even though the reference value is self-generated.

**Examples**: SN heterogeneous slab, MOC heterogeneous pin cell, diffusion
fuel+reflector.


Reference Cases
===============

.. include:: ../_generated/verification_table.rst


Homogeneous Infinite Medium
============================

For an infinite homogeneous medium, there is no spatial variation and no
leakage.  The neutron balance reduces to a matrix eigenvalue problem:

.. math::

   \mathbf{A} \phi = \frac{1}{k} \mathbf{F} \phi

where :math:`\mathbf{A} = \text{diag}(\Sigma_t) - \Sigma_s^T` is the
removal matrix and :math:`\mathbf{F} = \chi \otimes (\nu\Sigma_f)` is
the fission production matrix.

For 1 group: :math:`k = \nu\Sigma_f / \Sigma_a`.

For multi-group: :math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})`.

.. include:: ../_generated/homogeneous_derivation.rst


Discrete Ordinates (S\ :sub:`N`)
=================================

The S\ :sub:`N` method discretises the angular variable into a finite set of
directions.  For a homogeneous medium with reflective boundary conditions,
the derivation starts from the 1D S\ :sub:`N` transport equation:

.. math::

   \mu_m \frac{\partial\psi_m}{\partial x} + \Sigma_t \psi_m = \frac{Q}{2}

For a homogeneous medium, :math:`\partial\psi_m/\partial x = 0` (spatially
flat flux), so :math:`\psi_m = Q/(2\Sigma_t)` for every direction.  Integrating
with Gauss-Legendre weights (:math:`\sum w_m = 2`):

.. math::

   \phi = \sum_m w_m \psi_m = \frac{Q}{\Sigma_t}

Substituting the source :math:`Q = \Sigma_s \phi + (1/k)\nu\Sigma_f \phi`
and cancelling :math:`\phi` yields the same eigenvalue as the homogeneous
problem.  This is an exact result — the GL quadrature integrates a constant
exactly, and diamond-difference is exact for flat flux.

For heterogeneous problems, the reference comes from Richardson extrapolation
of the O(h²) diamond-difference scheme.

.. include:: ../_generated/sn_derivation.rst


Slab Collision Probability
==========================

The slab CP method uses the E₃ exponential integral kernel to compute
first-collision probabilities in a 1D half-cell with reflective centre and
white boundary condition at the cell edge.

The CP matrix :math:`P_{\infty}(i,j,g)` gives the probability that a neutron
born uniformly in region *j* has its first collision in region *i*, for energy
group *g*.  It is computed from the E₃ second-difference formula and the
white-BC closure.

The eigenvalue problem dimension is (N_regions × N_groups), solved as a
dense matrix eigenvalue.  This is semi-analytical: E₃ is computed to machine
precision via ``scipy.special.expn``.

.. include:: ../_generated/cp_slab_derivation.rst


Cylindrical Collision Probability
=================================

The cylindrical CP method uses the Ki₃/Ki₄ Bickley-Naylor kernel for a
Wigner-Seitz cell with annular regions and white boundary condition.

The Ki₄ function is the integrated Bickley-Naylor function:

.. math::

   \text{Ki}_3(x) = \int_0^{\pi/2} e^{-x/\sin\theta} \sin\theta \, d\theta, \qquad
   \text{Ki}_4(x) = \int_x^\infty \text{Ki}_3(t) \, dt

The CP matrix is computed via y-quadrature (Gauss-Legendre with breakpoints
at each radial boundary) and the Ki₄ second-difference formula.

.. include:: ../_generated/cp_cylinder_derivation.rst


.. _nine-case-cp-grid:

The 9-Case Grid (per geometry)
------------------------------

Both slab and cylinder derivation modules populate a **3 × 3 grid**
of verification cases, indexed by the number of energy groups and
the number of spatial regions:

.. csv-table::
   :header: Case name, N_groups, N_regions, Layout
   :widths: 25, 12, 12, 51

   ``cp_slab_1eg_1rg``, 1, 1, A (homogeneous fissile)
   ``cp_slab_1eg_2rg``, 1, 2, A + B (fuel + moderator)
   ``cp_slab_1eg_4rg``, 1, 4, A + D + C + B (fuel + gap + clad + moderator)
   ``cp_slab_2eg_1rg``, 2, 1, A
   ``cp_slab_2eg_2rg``, 2, 2, A + B
   ``cp_slab_2eg_4rg``, 2, 4, A + D + C + B
   ``cp_slab_4eg_1rg``, 4, 1, A
   ``cp_slab_4eg_2rg``, 4, 2, A + B
   ``cp_slab_4eg_4rg``, 4, 4, A + D + C + B

The cylinder set has the same structure with ``cp_slab`` replaced by
``cp_cyl1D``. This is **9 + 9 = 18** semi-analytical cases per
Sphinx build.

The grid is deliberate, not incidental:

- **N_groups × N_regions sweep** exposes the two independent failure
  modes — matrix-eigenvalue assembly (groups) and collision-probability
  quadrature (regions). A bug in one is uncovered by the other being
  correct.
- **1/2/4 chosen at each axis** spans the meaningful regimes: 1-group
  is the degenerate :math:`k = \nu\Sigma_f/\Sigma_a` limit, 2-group is the smallest genuine multi-group
  problem, and 4-group exercises the full :math:`\chi` spectrum
  distribution across fast/resonance/thermal ranges.
- **4-region layout A + D + C + B** is the minimal non-trivial
  heterogeneous pin cell: it puts fuel (A), gap (D), clad (C), and
  moderator (B) in the physically correct radial order and forces
  every region-coupling term in the CP matrix to be non-zero.
  Reducing to 2 regions (A + B) removes the gap and cladding; to
  1 region removes heterogeneity entirely.

The per-case ``VerificationCase`` records are populated by SymPy
evaluation of the E\ :sub:`3` (slab) or Ki\ :sub:`4` (cylinder)
second-difference formulas at build time — see
``orpheus/derivations/cp_slab.py`` and
``orpheus/derivations/cp_cylinder.py``. Each record carries:
analytical :math:`k`, materials, geometry, matrix-eigenvalue
context, and (since PR-2) the V&V level and equation-label list
used by the test harness (see ``docs/testing/architecture.rst``).


Method of Characteristics (MOC)
================================

The MOC method solves the transport equation along characteristic rays.  The
characteristic ODE for angular flux along a ray with direction :math:`\hat\Omega`:

.. math::

   \frac{d\psi}{ds} + \Sigma_t \psi = \frac{Q}{4\pi}

The segment-average solution:

.. math::

   \bar\psi = \psi_{\rm in} \frac{1 - e^{-\Sigma_t \ell}}{\Sigma_t \ell}
   + \frac{Q}{4\pi\Sigma_t} \left(1 - \frac{1 - e^{-\Sigma_t \ell}}{\Sigma_t \ell}\right)

For a homogeneous medium with isotropic incoming flux
(:math:`\psi_{\rm in} = Q/(4\pi\Sigma_t)`), this simplifies to
:math:`\bar\psi = Q/(4\pi\Sigma_t)` regardless of segment length — the
flat-source solution is exact.  The eigenvalue is then :math:`k = \nu\Sigma_f/\Sigma_a`.

.. include:: ../_generated/moc_derivation.rst


Monte Carlo
===========

The Monte Carlo method simulates neutron random walks.  For a homogeneous
infinite medium, the expected multiplication factor can be derived from
collision probabilities:

1. Probability of fission per collision: :math:`\Sigma_f / \Sigma_t`
2. Expected secondaries per collision: :math:`\nu \Sigma_f / \Sigma_t`
3. Mean collisions before absorption: :math:`\Sigma_t / \Sigma_a`
4. Therefore: :math:`k = \nu\Sigma_f / \Sigma_a`

The MC estimator :math:`\hat{k}` converges to this with standard error
:math:`\sigma \sim 1/\sqrt{N_{\rm active}}` by the Central Limit Theorem.

.. include:: ../_generated/mc_derivation.rst


Diffusion (Buckling Eigenvalue)
================================

The 1D finite-difference diffusion solver is verified against the analytical
buckling eigenvalue for a bare homogeneous slab with vacuum boundary conditions:

.. math::

   B^2 = \left(\frac{\pi}{H}\right)^2

The 2-group eigenvalue is :math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})`
where :math:`\mathbf{A}` includes the leakage term :math:`D B^2`.

.. _diffusion-2region-richardson:

2-Region Fuel + Reflector: Richardson-Extrapolated Reference
------------------------------------------------------------

The bare-slab buckling eigenvalue tests a homogeneous medium but
leaves the **interface-matching** logic of the diffusion solver
unexercised. The 2-region ``dif_slab_2eg_2rg`` case closes that gap
with a fuel + reflector geometry:

.. csv-table::
   :header: Region, Extent, Role, Cross sections
   :widths: 18, 25, 25, 32

   fuel, 0 – 50 cm, fissile core, 2-group fuel set (``_FUEL_XS``)
   reflector, 50 – 80 cm, non-fissile scatterer, 2-group reflector set (``_REFL_XS``)

Boundary conditions are **vacuum on both ends**. The analytical
2-group coupled transcendental equation (cos in fuel, sinh in
reflector, flux and current continuity at the interface) has no
closed-form :math:`k`, so the reference is generated by running
the diffusion solver itself at four mesh refinements and
extrapolating:

.. math::
   :label: richardson-diffusion

   k_{\text{ref}} \approx k_h + \frac{k_h - k_{2h}}{2^p - 1}
   \qquad (p = 2)

where :math:`p = 2` reflects the O(h²) central-difference
discretization of the diffusion equation. The four mesh widths
used are :math:`dz \in \{2.5, 1.25, 0.625, 0.3125\}` cm — ratio 2,
spanning one doubling more than strictly necessary so that the
second-finest pair can be used as a sanity check.

**Reference value**: :math:`k_{\text{eff}} \approx 0.870` —
measurably higher than the :math:`k_{\infty} \approx 0.821` that
the fuel composition alone would produce, quantifying the
**reflector savings** from returning leaked neutrons back into the
fuel region.

The derivation lives in ``orpheus/derivations/diffusion.py::derive_2rg``.
Because running the solver four times costs ~15 minutes, the
result is cached on disk in
``orpheus/derivations/_richardson_cache.json`` (see
:ref:`reference-values-lazy-registry`). The cache key includes
``(fuel_height, refl_height, dzs, fuel_xs, refl_xs)`` so any
parameter change invalidates it automatically.

This case is an L1 equation-verification per the V&V ladder —
it proves that the solver converges at the expected order on a
non-trivial heterogeneous problem, even though the "reference"
is self-generated. The test carrying it is
``tests/test_diffusion.py::test_spatial_convergence_reflected``,
which consumes the Richardson-extrapolated reference via
``case.k_inf`` from the derivation registry and asserts O(h²)
convergence of the diffusion solver across four mesh widths.

.. include:: ../_generated/diffusion_derivation.rst


Unit Tests
==========

Beyond eigenvalue verification, the test suite verifies structural properties
of each solver's intermediate quantities.

CP Matrix Properties
--------------------

For every CP case (slab and cylinder), the collision probability matrix
:math:`P_{\infty}` must satisfy:

- **Row sums = 1** (neutron conservation): every neutron born in region *i*
  must have its first collision somewhere.
  :math:`\sum_j P_{\infty}(i,j,g) = 1 \; \forall \, i, g`.
  Tolerance: < 10⁻¹⁰.

- **Reciprocity**: :math:`\Sigma_{t,i} V_i P(i,j) = \Sigma_{t,j} V_j P(j,i)`.
  This is detailed balance — a consequence of time-reversal symmetry of the
  transport equation.  Tolerance: < 10⁻¹⁰.

- **Non-negativity**: :math:`P(i,j,g) \geq 0 \; \forall \, i, j, g`.

- **Homogeneous limit**: a 1-region cell must give :math:`P(0,0) = 1`.

SN Properties
-------------

- **GL quadrature weights**: must sum to 2 (measure of [-1,1]).
- **GL symmetry**: :math:`\mu_i = -\mu_{N-1-i}`.
- **Flux symmetry**: homogeneous slab must have spatially flat flux.
- **Particle balance**: with reflective BCs (no leakage),
  production / absorption = keff.

Diffusion Properties
--------------------

- **Vacuum BC**: flux at boundary cells is small compared to peak.
- **Flux positivity**: all flux values positive in fundamental mode.
- **Flux symmetry**: bare slab flux is symmetric about the center.

MOC Properties
--------------

- **Particle balance**: production / absorption = keff.
- **Flux positivity**: scalar flux > 0 everywhere.
- **from_annular geometry**: material assignment matches radial distances.

MC Properties
-------------

- **Geometry protocol**: ``ConcentricPinCell`` and ``SlabPinCell`` return
  correct material IDs at known positions.
- **1G deterministic**: homogeneous 1-group MC has σ = 0 (all neutrons
  see identical cross sections).


Convergence Studies
===================

Beyond point-value verification, the test suite checks that discretisation
errors decrease at the expected rate:

**SN 1D spatial convergence** (diamond-difference):
  The observed convergence order should approach 2.0 as the mesh is refined,
  confirming the O(h²) truncation error of the diamond-difference scheme.

**SN 1D angular convergence** (Gauss-Legendre):
  The eigenvalue error should decrease faster than any polynomial in
  1/N, confirming spectral convergence of the GL quadrature.

**Diffusion spatial convergence**:
  The three-point finite-difference stencil gives O(h²) convergence,
  verified against the analytical buckling eigenvalue.


.. seealso::

   Verified solvers and their theory pages:

   - :ref:`theory-homogeneous` — matrix eigenvalue (analytical, exact)
   - :ref:`theory-collision-probability` — E₃/Ki₄ semi-analytical eigenvalue
   - :ref:`theory-discrete-ordinates` — homogeneous exact + :ref:`Richardson <richardson-extrapolation>` heterogeneous
   - :ref:`theory-method-of-characteristics` — homogeneous exact + Richardson heterogeneous
   - :ref:`theory-monte-carlo` — z-score against analytical + CP reference

   Cross sections: :ref:`theory-cross-section-data` (real nuclear data pipeline).


Running the Tests
=================

.. code-block:: bash

   # Install test dependencies
   pip install -e ".[test]"

   # Run non-slow tests (~90s, 56 tests)
   pytest

   # Run slow tests (~7min, 17 tests including Richardson + MC high-stats)
   pytest -m slow

   # Run all 73 tests
   pytest -v

   # Run a specific solver's tests
   pytest tests/test_homogeneous.py -v
   pytest tests/test_cp_properties.py -v
