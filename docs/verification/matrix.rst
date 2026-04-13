Verification Matrix
===================

.. note::

   Auto-generated from ``tests._harness.registry.TEST_REGISTRY``
   by ``tools/verification/generate_matrix.py``. Do not edit by
   hand — changes will be overwritten on the next rebuild.

Total tests collected: **501**

V&V level distribution
----------------------

.. csv-table::
   :header: Level, Count, Share
   :widths: 15, 10, 10

   L0, 239, 47.7%
   L1, 162, 32.3%
   L2, 45, 9.0%
   L3, 0, 0.0%
   unmarked, 55, 11.0%

Tagging source
--------------

How each test acquired its V&V level (see ``tests/conftest.py`` for the precedence chain).

.. csv-table::
   :header: Source, Count
   :widths: 20, 10

   explicit, 318
   verify, 0
   class-name, 46
   func-name, 0
   case, 82
   unmarked, 55

Module × level grid
-------------------

.. csv-table::
   :header: Module, L0, L1, L2, L3, ??
   :widths: 40, 6, 6, 6, 6, 6

   test_convergence, 0, 0, 1, 0, 0
   test_cp_cylinder, 0, 9, 0, 0, 0
   test_cp_diagnostics, 8, 28, 0, 0, 0
   test_cp_properties, 12, 0, 0, 0, 0
   test_cp_slab, 0, 9, 0, 0, 0
   test_cp_sphere, 0, 9, 0, 0, 0
   test_cp_verification, 1, 25, 5, 0, 0
   test_diffusion, 0, 2, 0, 0, 0
   test_diffusion_properties, 3, 0, 0, 0, 0
   test_discrete_ordinates, 0, 0, 2, 0, 0
   test_geometry, 6, 0, 0, 0, 55
   test_homogeneous, 0, 3, 0, 0, 0
   test_mc_convergence, 0, 0, 3, 0, 0
   test_mc_cross_verification, 0, 0, 2, 0, 0
   test_mc_gaps, 6, 8, 0, 0, 0
   test_mc_properties, 24, 0, 0, 0, 0
   test_mixture, 4, 0, 0, 0, 0
   test_moc, 0, 3, 3, 0, 0
   test_moc_properties, 4, 0, 0, 0, 0
   test_moc_quadrature, 24, 0, 0, 0, 0
   test_moc_ray_tracing, 20, 0, 0, 0, 0
   test_moc_verification, 27, 15, 6, 0, 0
   test_monte_carlo, 0, 12, 0, 0, 0
   test_sn_1d, 0, 5, 6, 0, 0
   test_sn_cylindrical, 0, 14, 11, 0, 0
   test_sn_properties, 4, 0, 0, 0, 0
   test_sn_quadrature, 49, 0, 0, 0, 0
   test_sn_solver_components, 35, 0, 0, 0, 0
   test_sn_spherical, 0, 20, 6, 0, 0
   test_sn_sweep_regression, 12, 0, 0, 0, 0

Equation coverage
-----------------

Every Sphinx ``.. math:: :label:`` block declared in ``docs/theory/*.rst`` and the number of tests carrying ``@pytest.mark.verifies("label")`` that reference it.

.. csv-table::
   :header: Equation label, Tests
   :widths: 50, 10

   ``matrix-eigenvalue``, 159
   ``mg-balance``, 159
   ``one-group-kinf``, 113
   ``reflective-bc``, 112
   ``alpha-recursion``, 100
   ``wdd-closure``, 100
   ``wdd-face``, 100
   ``collision-rate``, 88
   ``alpha-cylindrical``, 74
   ``mm-weights``, 74
   ``multigroup``, 64
   ``self-slab``, 52
   ``self-cyl``, 51
   ``e3-def``, 49
   ``flux-moments``, 49
   ``ki3-def``, 49
   ``p-inf``, 49
   ``self-sph``, 49
   ``wigner-seitz``, 46
   ``chord-length``, 36
   ``cp-keff-update``, 31
   ``keff-mean``, 31
   ``matrix-A-def``, 31
   ``matrix-B-def``, 31
   ``neutron-balance``, 31
   ``sigma-keff``, 31
   ``free-flight``, 29
   ``chi-sampling``, 26
   ``decompose``, 26
   ``scattering-cdf``, 26
   ``transport-spherical``, 26
   ``transport-cylindrical``, 25
   ``dc-slab``, 15
   ``dd-slab``, 15
   ``second-diff-cyl``, 15
   ``second-diff-general``, 15
   ``second-diff-sph``, 15
   ``direction-sampling``, 14
   ``fission-weight``, 14
   ``keff-cycle``, 14
   ``roulette-conservation``, 14
   ``roulette-prob``, 14
   ``dd-cartesian-1d``, 12
   ``kinf-1g``, 12
   ``kinf-mg``, 12
   ``periodic-bc``, 12
   ``transport-cartesian``, 12
   ``ws-pitch``, 12
   ``dd-recurrence``, 11
   ``dd-solve``, 11
   ``bar-psi``, 6
   ``boyd-eq-45``, 6
   ``characteristic-ode``, 6
   ``delta-psi``, 6
   ``isotropic-source``, 6
   ``moc-keff-update``, 6
   ``moc-wigner-seitz``, 6
   ``macro-sum``, 4
   ``fission-matrix``, 3
   ``inf-hom-balance``, 3
   ``removal-matrix``, 3
   ``dd-cartesian-2d``, 2
   ``transport-cartesian-2d``, 2
   ``two-group-A``, 1
   ``two-group-Ainv``, 1
   ``two-group-F``, 1
   ``two-group-M``, 1
   ``two-group-charpoly``, 1
   ``two-group-roots``, 1

Orphan equations
----------------

Equations with zero tests carrying ``@pytest.mark.verifies("label")``. **66** of the equations found on theory pages are orphan.

- ``absorption-xs``
- ``attenuation``
- ``azimuthal-angles``
- ``balance-general``
- ``bickley-integral``
- ``boltzmann``
- ``branching``
- ``burst-criterion``
- ``clad-heat``
- ``complementarity``
- ``conservative-form``
- ``convergence-rate``
- ``coolant-energy``
- ``coolant-feedback``
- ``coolant-rate``
- ``creep-rate``
- ``doppler-feedback``
- ``effective-spacing``
- ``fb-bc4-displacement``
- ``fb-clad-strain``
- ``fb-fuel-heat``
- ``fb-fuel-strain``
- ``fb-swelling``
- ``first-flight-kernel``
- ``fission-source``
- ``fixed-source-solve``
- ``flat-source``
- ``fuel-heat``
- ``fuel-rate``
- ``gap-closure-event``
- ``gap-conductance``
- ``gas-pressure``
- ``group-flux``
- ``group-xs``
- ``hetero-tolerance``
- ``keff-update``
- ``majorant``
- ``maxwellian``
- ``normalisation``
- ``number-density``
- ``one-over-E``
- ``optical-path``
- ``optical-thickness``
- ``pcell-from-smat``
- ``pin-from-reciprocity``
- ``pitch-recovery``
- ``pn-scatter``
- ``power-equation``
- ``precursor-equation``
- ``ray-circle``
- ``rcp-from-double-antideriv``
- ``reciprocity``
- ``roulette-restore``
- ``s-integral``
- ``scalar-flux-integral``
- ``self-double-integral``
- ``sigT-computed``
- ``sigma-zero``
- ``sigs-convention``
- ``splitting``
- ``surface-to-region``
- ``surface-to-surface``
- ``tau-m``
- ``tau-p``
- ``transport-equation``
- ``xs-interp``

L0 error-catalog coverage
-------------------------

Every ``ERR-NNN`` entry in ``tests/l0_error_catalog.md`` and the tests that carry ``@pytest.mark.catches("ERR-NNN")`` to guard it. A missing catcher is a publication-blocker for the error catalog.

.. csv-table::
   :header: Error tag, Catching tests
   :widths: 15, 10

   ``ERR-001``, **0 (MISSING)**
   ``ERR-002``, **0 (MISSING)**
   ``ERR-003``, **0 (MISSING)**
   ``ERR-004``, **0 (MISSING)**
   ``ERR-005``, **0 (MISSING)**
   ``ERR-006``, **0 (MISSING)**
   ``ERR-007``, **0 (MISSING)**
   ``ERR-008``, **0 (MISSING)**
   ``ERR-009``, **0 (MISSING)**
   ``ERR-010``, **0 (MISSING)**
   ``ERR-011``, **0 (MISSING)**
   ``ERR-012``, **0 (MISSING)**
   ``ERR-013``, **0 (MISSING)**
   ``ERR-014``, **0 (MISSING)**
   ``ERR-015``, **0 (MISSING)**
   ``ERR-016``, **0 (MISSING)**
   ``ERR-017``, **0 (MISSING)**
   ``ERR-018``, **0 (MISSING)**
   ``ERR-019``, **0 (MISSING)**
   ``ERR-020``, 6

Unmarked tests
--------------

**55 tests** have no V&V level marker.
Unmarked is acceptable for tests that exercise infrastructure
(mesh construction, dataclass immutability, CLI behaviour) and
do not verify a physics equation.

.. csv-table::
   :header: File, Unmarked tests
   :widths: 60, 10

   ``tests/test_geometry.py``, 55

