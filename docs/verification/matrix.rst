Verification Matrix
===================

.. note::

   Auto-generated from ``tests._harness.registry.TEST_REGISTRY``
   by ``tools/verification/generate_matrix.py``. Do not edit by
   hand — changes will be overwritten on the next rebuild.

Total tests collected: **607**

V&V level distribution
----------------------

.. csv-table::
   :header: Level, Count, Share
   :widths: 15, 10, 10

   L0, 345, 56.8%
   L1, 156, 25.7%
   L2, 45, 7.4%
   L3, 0, 0.0%
   foundation, 61, 10.0%
   unmarked, 0, 0.0%

Tagging source
--------------

How each test acquired its V&V level (see ``tests/conftest.py`` for the precedence chain).

.. csv-table::
   :header: Source, Count
   :widths: 20, 10

   explicit, 519
   verify, 0
   class-name, 46
   func-name, 0
   case, 42
   unmarked, 0

Module × level grid
-------------------

.. csv-table::
   :header: Module, L0, L1, L2, L3, FD, ??
   :widths: 40, 6, 6, 6, 6, 6, 6

   cp/test_cylinder, 0, 9, 0, 0, 0, 0
   cp/test_diagnostics, 8, 28, 0, 0, 0, 0
   cp/test_properties, 12, 0, 0, 0, 0, 0
   cp/test_slab, 0, 9, 0, 0, 0, 0
   cp/test_sphere, 0, 9, 0, 0, 0, 0
   cp/test_verification, 1, 25, 5, 0, 0, 0
   data/test_cross_section_data, 11, 0, 0, 0, 0, 0
   data/test_mixture, 4, 0, 0, 0, 0, 0
   derivations/test_kernels, 67, 0, 0, 0, 0, 0
   diffusion/test_diffusion, 0, 2, 0, 0, 0, 0
   diffusion/test_properties, 3, 0, 0, 0, 0, 0
   geometry/test_geometry, 0, 0, 0, 0, 61, 0
   homogeneous/test_continuous_reference, 0, 7, 0, 0, 0, 0
   homogeneous/test_homogeneous, 0, 4, 0, 0, 0, 0
   mc/test_convergence, 0, 0, 3, 0, 0, 0
   mc/test_cross_verification, 0, 0, 2, 0, 0, 0
   mc/test_gaps, 7, 9, 0, 0, 0, 0
   mc/test_monte_carlo, 0, 12, 0, 0, 0, 0
   mc/test_properties, 24, 0, 0, 0, 0, 0
   moc/test_moc, 0, 3, 3, 0, 0, 0
   moc/test_properties, 4, 0, 0, 0, 0, 0
   moc/test_quadrature, 24, 0, 0, 0, 0, 0
   moc/test_ray_tracing, 22, 0, 0, 0, 0, 0
   moc/test_verification, 27, 15, 6, 0, 0, 0
   sn/test_cartesian, 0, 5, 6, 0, 0, 0
   sn/test_cylindrical, 4, 10, 11, 0, 0, 0
   sn/test_discrete_ordinates_2d, 0, 0, 2, 0, 0, 0
   sn/test_mms, 0, 2, 0, 0, 0, 0
   sn/test_properties, 4, 0, 0, 0, 0, 0
   sn/test_quadrature, 49, 0, 0, 0, 0, 0
   sn/test_solver_components, 35, 0, 0, 0, 0, 0
   sn/test_spherical, 13, 7, 6, 0, 0, 0
   sn/test_sweep_regression, 12, 0, 0, 0, 0, 0
   test_convergence, 0, 0, 1, 0, 0, 0
   test_pending_ports, 5, 0, 0, 0, 0, 0
   test_vv_harness_audit, 9, 0, 0, 0, 0, 0

Equation coverage
-----------------

Every Sphinx ``.. math:: :label:`` block declared in ``docs/theory/*.rst`` and the number of tests carrying ``@pytest.mark.verifies("label")`` that reference it.

.. csv-table::
   :header: Equation label, Tests
   :widths: 50, 10

   ``matrix-eigenvalue``, 167
   ``mg-balance``, 167
   ``one-group-kinf``, 121
   ``reflective-bc``, 112
   ``alpha-recursion``, 100
   ``wdd-closure``, 100
   ``wdd-face``, 100
   ``collision-rate``, 88
   ``alpha-cylindrical``, 74
   ``mm-weights``, 74
   ``multigroup``, 64
   ``self-slab``, 52
   ``balance-general``, 51
   ``self-cyl``, 51
   ``e3-def``, 49
   ``flux-moments``, 49
   ``ki3-def``, 49
   ``p-inf``, 49
   ``self-sph``, 49
   ``attenuation``, 48
   ``optical-thickness``, 48
   ``scalar-flux-integral``, 48
   ``wigner-seitz``, 46
   ``chord-length``, 36
   ``keff-mean``, 33
   ``sigma-keff``, 33
   ``cp-keff-update``, 31
   ``first-flight-kernel``, 31
   ``flat-source``, 31
   ``free-flight``, 31
   ``matrix-A-def``, 31
   ``matrix-B-def``, 31
   ``neutron-balance``, 31
   ``optical-path``, 31
   ``pcell-from-smat``, 31
   ``pin-from-reciprocity``, 31
   ``rcp-from-double-antideriv``, 31
   ``s-integral``, 31
   ``self-double-integral``, 31
   ``surface-to-region``, 31
   ``surface-to-surface``, 31
   ``chi-sampling``, 28
   ``decompose``, 28
   ``scattering-cdf``, 28
   ``transport-spherical``, 26
   ``transport-cylindrical``, 25
   ``azimuthal-angles``, 24
   ``effective-spacing``, 22
   ``pitch-recovery``, 22
   ``ray-circle``, 22
   ``en-kernel-derivative``, 20
   ``kin-kernel-derivative``, 20
   ``dd-slab``, 16
   ``direction-sampling``, 16
   ``fission-weight``, 16
   ``keff-cycle``, 16
   ``roulette-conservation``, 16
   ``roulette-prob``, 16
   ``dc-slab``, 15
   ``second-diff-cyl``, 15
   ``second-diff-general``, 15
   ``second-diff-sph``, 15
   ``dd-cartesian-1d``, 14
   ``transport-cartesian``, 13
   ``complementarity``, 12
   ``kin-bickley-legacy-convention``, 12
   ``kinf-1g``, 12
   ``kinf-mg``, 12
   ``periodic-bc``, 12
   ``reciprocity``, 12
   ``ws-pitch``, 12
   ``dd-recurrence``, 11
   ``dd-solve``, 11
   ``fission-matrix``, 11
   ``inf-hom-balance``, 11
   ``removal-matrix``, 11
   ``two-group-A``, 11
   ``two-group-Ainv``, 11
   ``two-group-F``, 11
   ``two-group-M``, 11
   ``pn-scatter``, 9
   ``tau-m``, 9
   ``tau-p``, 9
   ``bar-psi``, 6
   ``boyd-eq-45``, 6
   ``characteristic-ode``, 6
   ``delta-psi``, 6
   ``isotropic-source``, 6
   ``kin-kernel-special-values``, 6
   ``moc-keff-update``, 6
   ``moc-wigner-seitz``, 6
   ``en-kernel-special-values``, 5
   ``xs-interp``, 5
   ``absorption-xs``, 4
   ``en-kernel-integral``, 4
   ``fission-source``, 4
   ``fixed-source-solve``, 4
   ``keff-update``, 4
   ``macro-sum``, 4
   ``two-group-charpoly``, 4
   ``two-group-roots``, 4
   ``hetero-tolerance``, 3
   ``number-density``, 3
   ``sigma-zero``, 3
   ``dd-cartesian-2d``, 2
   ``richardson-diffusion``, 2
   ``roulette-restore``, 2
   ``transport-cartesian-2d``, 2
   ``branching``, 1
   ``collision-estimator``, 1
   ``majorant``, 1
   ``normalisation``, 1
   ``sigT-computed``, 1
   ``sn-mms-psi``, 1
   ``sn-mms-qext``, 1
   ``splitting``, 1

Orphan equations
----------------

Equations with zero tests carrying ``@pytest.mark.verifies("label")``, excluding labels explicitly marked ``:vv-status: documented``. **0** of the testable equations found on theory pages are orphan.

*(none — every testable theory equation has at least one verifying test)*

Documented-only equations
-------------------------

Theory labels marked ``.. vv-status: <label> documented`` in their RST source. These are excluded from the orphan-equation gate because they are either definitional (no single implementing function — e.g. ``boltzmann``), describe a module whose Python port does not yet exist (e.g. the thermal-hydraulics / fuel-behaviour / reactor-kinetics equations), or have a deliberately deferred test paired with a tracking issue. **29** labels carry the directive. See ``docs/testing/architecture.rst``:ref:`vv-status-documented` for the full taxonomy.

- ``bickley-integral``
- ``boltzmann``
- ``burst-criterion``
- ``clad-heat``
- ``conservative-form``
- ``convergence-rate``
- ``coolant-energy``
- ``coolant-feedback``
- ``coolant-rate``
- ``creep-rate``
- ``doppler-feedback``
- ``fb-bc4-displacement``
- ``fb-clad-strain``
- ``fb-fuel-heat``
- ``fb-fuel-strain``
- ``fb-swelling``
- ``fuel-heat``
- ``fuel-rate``
- ``gap-closure-event``
- ``gap-conductance``
- ``gas-pressure``
- ``group-flux``
- ``group-xs``
- ``maxwellian``
- ``one-over-E``
- ``power-equation``
- ``precursor-equation``
- ``sigs-convention``
- ``transport-equation``

L0 error-catalog coverage
-------------------------

Every ``ERR-NNN`` entry in ``tests/l0_error_catalog.md`` and the tests that carry ``@pytest.mark.catches("ERR-NNN")`` to guard it. A missing catcher is a publication-blocker for the error catalog.

.. csv-table::
   :header: Error tag, Catching tests
   :widths: 15, 10

   ``ERR-001``, 1
   ``ERR-002``, 1
   ``ERR-003``, 6
   ``ERR-004``, 1
   ``ERR-005``, 1
   ``ERR-006``, 2
   ``ERR-007``, 3
   ``ERR-008``, 1
   ``ERR-009``, 9
   ``ERR-010``, 1
   ``ERR-011``, 1
   ``ERR-012``, 1
   ``ERR-013``, 1
   ``ERR-014``, 1
   ``ERR-015``, 1
   ``ERR-016``, 2
   ``ERR-017``, 3
   ``ERR-018``, 1
   ``ERR-019``, 1
   ``ERR-020``, 6
   ``ERR-021``, 2
   ``ERR-022``, 1
   ``ERR-023``, 1
   ``ERR-024``, 1

Unmarked tests
--------------

*(none — every test carries an L0/L1/L2/L3 or foundation marker)*

