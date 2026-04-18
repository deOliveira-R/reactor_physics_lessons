Verification Matrix
===================

.. note::

   Auto-generated from ``tests._harness.registry.TEST_REGISTRY``
   by ``tools/verification/generate_matrix.py``. Do not edit by
   hand — changes will be overwritten on the next rebuild.

Total tests collected: **777**

V&V level distribution
----------------------

.. csv-table::
   :header: Level, Count, Share
   :widths: 15, 10, 10

   L0, 442, 56.9%
   L1, 210, 27.0%
   L2, 36, 4.6%
   L3, 0, 0.0%
   foundation, 78, 10.0%
   unmarked, 11, 1.4%

Tagging source
--------------

How each test acquired its V&V level (see ``tests/conftest.py`` for the precedence chain).

.. csv-table::
   :header: Source, Count
   :widths: 20, 10

   explicit, 687
   verify, 0
   class-name, 46
   func-name, 0
   case, 33
   unmarked, 11

Module × level grid
-------------------

.. csv-table::
   :header: Module, L0, L1, L2, L3, FD, ??
   :widths: 40, 6, 6, 6, 6, 6, 6

   cp/test_cylinder, 0, 9, 0, 0, 0, 0
   cp/test_diagnostics, 8, 28, 0, 0, 0, 0
   cp/test_peierls_cylinder_flux, 0, 4, 0, 0, 0, 0
   cp/test_peierls_flux, 0, 1, 0, 0, 0, 0
   cp/test_peierls_sphere_flux, 0, 4, 0, 0, 0, 0
   cp/test_properties, 12, 0, 0, 0, 0, 0
   cp/test_slab, 0, 9, 0, 0, 0, 0
   cp/test_sphere, 0, 9, 0, 0, 0, 0
   cp/test_verification, 1, 25, 5, 0, 0, 0
   data/test_cross_section_data, 11, 0, 0, 0, 0, 0
   data/test_mixture, 4, 0, 0, 0, 0, 0
   derivations/test_cp_geometry, 48, 0, 0, 0, 0, 0
   derivations/test_kernels, 55, 0, 0, 0, 0, 0
   derivations/test_peierls_convergence, 5, 0, 0, 0, 0, 0
   derivations/test_peierls_cylinder_eigenvalue, 3, 5, 0, 0, 0, 0
   derivations/test_peierls_cylinder_geometry, 10, 0, 0, 0, 0, 0
   derivations/test_peierls_cylinder_multi_region, 7, 0, 0, 0, 3, 0
   derivations/test_peierls_cylinder_prefactor, 4, 0, 0, 0, 0, 0
   derivations/test_peierls_cylinder_white_bc, 4, 3, 0, 0, 0, 0
   derivations/test_peierls_sphere_eigenvalue, 0, 4, 0, 0, 0, 0
   derivations/test_peierls_sphere_geometry, 21, 0, 0, 0, 0, 0
   derivations/test_peierls_sphere_prefactor, 6, 0, 0, 0, 0, 0
   derivations/test_peierls_sphere_white_bc, 0, 4, 0, 0, 0, 0
   diffusion/test_continuous_reference, 0, 8, 0, 0, 0, 0
   diffusion/test_diffusion, 0, 2, 0, 0, 0, 0
   diffusion/test_properties, 3, 0, 0, 0, 0, 0
   geometry/test_geometry, 0, 0, 0, 0, 75, 0
   homogeneous/test_continuous_reference, 0, 7, 0, 0, 0, 0
   homogeneous/test_homogeneous, 0, 4, 0, 0, 0, 0
   mc/test_convergence, 0, 0, 3, 0, 0, 0
   mc/test_cross_verification, 0, 0, 2, 0, 0, 0
   mc/test_gaps, 7, 9, 0, 0, 0, 0
   mc/test_monte_carlo, 0, 12, 0, 0, 0, 0
   mc/test_properties, 24, 0, 0, 0, 0, 0
   moc/test_mms, 0, 3, 0, 0, 0, 0
   moc/test_moc, 0, 3, 0, 0, 0, 0
   moc/test_properties, 4, 0, 0, 0, 0, 0
   moc/test_quadrature, 24, 0, 0, 0, 0, 0
   moc/test_ray_tracing, 22, 0, 0, 0, 0, 0
   moc/test_verification, 27, 15, 6, 0, 0, 0
   sn/test_boundary_conditions, 0, 0, 0, 0, 0, 11
   sn/test_cartesian, 1, 6, 0, 0, 0, 0
   sn/test_cylindrical, 4, 10, 11, 0, 0, 0
   sn/test_discrete_ordinates_2d, 0, 0, 2, 0, 0, 0
   sn/test_heterogeneous_transport, 0, 2, 0, 0, 0, 0
   sn/test_mms, 0, 2, 0, 0, 0, 0
   sn/test_mms_2d, 0, 3, 0, 0, 0, 0
   sn/test_mms_aniso, 0, 2, 0, 0, 0, 0
   sn/test_mms_curvilinear, 0, 2, 0, 0, 0, 0
   sn/test_mms_heterogeneous, 0, 4, 0, 0, 0, 0
   sn/test_properties, 4, 0, 0, 0, 0, 0
   sn/test_quadrature, 49, 0, 0, 0, 0, 0
   sn/test_solver_components, 35, 0, 0, 0, 0, 0
   sn/test_spherical, 13, 7, 6, 0, 0, 0
   sn/test_sweep_operator_inconsistency, 0, 4, 0, 0, 0, 0
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

   ``mg-balance``, 165
   ``matrix-eigenvalue``, 160
   ``one-group-kinf``, 132
   ``reflective-bc``, 110
   ``alpha-recursion``, 100
   ``wdd-closure``, 100
   ``wdd-face``, 100
   ``collision-rate``, 91
   ``alpha-cylindrical``, 74
   ``mm-weights``, 74
   ``multigroup``, 65
   ``ki3-def``, 61
   ``e3-def``, 58
   ``self-slab``, 52
   ``balance-general``, 51
   ``chord-length``, 51
   ``self-cyl``, 51
   ``p-inf``, 50
   ``flux-moments``, 49
   ``self-sph``, 49
   ``attenuation``, 48
   ``optical-thickness``, 48
   ``scalar-flux-integral``, 48
   ``wigner-seitz``, 46
   ``peierls-unified``, 39
   ``cp-kernel-differential-identities``, 36
   ``keff-mean``, 33
   ``sigma-keff``, 33
   ``flat-source``, 32
   ``cp-keff-update``, 31
   ``first-flight-kernel``, 31
   ``free-flight``, 31
   ``matrix-A-def``, 31
   ``matrix-B-def``, 31
   ``neutron-balance``, 31
   ``optical-path``, 31
   ``pcell-from-smat``, 31
   ``peierls-equation``, 31
   ``pin-from-reciprocity``, 31
   ``rcp-from-double-antideriv``, 31
   ``s-integral``, 31
   ``self-double-integral``, 31
   ``surface-to-region``, 31
   ``surface-to-surface``, 31
   ``dd-slab``, 30
   ``chi-sampling``, 28
   ``decompose``, 28
   ``scattering-cdf``, 28
   ``cp-flat-source-derivation``, 27
   ``cp-flat-source-double-integral``, 27
   ``cp-unified-outer-integration``, 27
   ``transport-spherical``, 27
   ``transport-cylindrical``, 26
   ``azimuthal-angles``, 24
   ``dc-slab``, 24
   ``second-diff-cyl``, 24
   ``second-diff-sph``, 24
   ``effective-spacing``, 22
   ``pitch-recovery``, 22
   ``ray-circle``, 22
   ``en-kernel-derivative``, 20
   ``kin-kernel-derivative``, 20
   ``dd-cartesian-1d``, 17
   ``direction-sampling``, 16
   ``fission-weight``, 16
   ``keff-cycle``, 16
   ``roulette-conservation``, 16
   ``roulette-prob``, 16
   ``transport-cartesian``, 16
   ``second-diff-general``, 15
   ``complementarity``, 12
   ``kinf-1g``, 12
   ``kinf-mg``, 12
   ``periodic-bc``, 12
   ``reciprocity``, 12
   ``ws-pitch``, 12
   ``fission-matrix``, 11
   ``inf-hom-balance``, 11
   ``pn-scatter``, 11
   ``removal-matrix``, 11
   ``two-group-A``, 11
   ``two-group-Ainv``, 11
   ``two-group-F``, 11
   ``two-group-M``, 11
   ``dd-recurrence``, 9
   ``tau-m``, 9
   ``tau-p``, 9
   ``bare-slab-buckling``, 8
   ``bare-slab-critical-equation``, 8
   ``bare-slab-eigenfunction``, 8
   ``cp-inner-integral-antiderivative``, 8
   ``diffusion-M-matrix``, 8
   ``diffusion-back-substitution``, 8
   ``diffusion-coefficient``, 8
   ``diffusion-exponential-branch``, 8
   ``diffusion-interface-matching``, 8
   ``diffusion-matching-matrix``, 8
   ``diffusion-mode-decomposition``, 8
   ``diffusion-operator``, 8
   ``diffusion-region-ode``, 8
   ``diffusion-spurious-root-validation``, 8
   ``diffusion-transcendental``, 8
   ``diffusion-trigonometric-branch``, 8
   ``dd-solve``, 7
   ``bar-psi``, 6
   ``boyd-eq-45``, 6
   ``characteristic-ode``, 6
   ``kin-kernel-special-values``, 6
   ``cp-outer-integral-antiderivative``, 5
   ``en-kernel-special-values``, 5
   ``xs-interp``, 5
   ``absorption-xs``, 4
   ``dd-cartesian-2d``, 4
   ``en-kernel-integral``, 4
   ``fission-source``, 4
   ``fixed-source-solve``, 4
   ``keff-update``, 4
   ``macro-sum``, 4
   ``sn-mms-hetero-psi``, 4
   ``sn-mms-hetero-qext``, 4
   ``transport-cartesian-2d``, 4
   ``two-group-charpoly``, 4
   ``two-group-roots``, 4
   ``cp-escape-from-p-cell``, 3
   ``delta-psi``, 3
   ``hetero-tolerance``, 3
   ``isotropic-source``, 3
   ``moc-keff-update``, 3
   ``moc-mms-psi-ref``, 3
   ``moc-mms-qext``, 3
   ``moc-wigner-seitz``, 3
   ``number-density``, 3
   ``sigma-zero``, 3
   ``cp-second-difference-operator``, 2
   ``richardson-diffusion``, 2
   ``roulette-restore``, 2
   ``sn-case-back-substitution``, 2
   ``sn-case-matching-matrix``, 2
   ``sn-case-per-ordinate``, 2
   ``sn-case-physical-validation``, 2
   ``sn-case-real-basis``, 2
   ``sn-case-slope-matrix``, 2
   ``sn-case-spatial-modes``, 2
   ``sn-mms-2d-2g-psi``, 2
   ``sn-mms-p1-qext``, 2
   ``branching``, 1
   ``collision-estimator``, 1
   ``majorant``, 1
   ``normalisation``, 1
   ``sigT-computed``, 1
   ``sn-mms-2d-2g-qext``, 1
   ``sn-mms-2d-psi``, 1
   ``sn-mms-2d-qext``, 1
   ``sn-mms-cylindrical-psi``, 1
   ``sn-mms-cylindrical-qext``, 1
   ``sn-mms-p1-psi``, 1
   ``sn-mms-psi``, 1
   ``sn-mms-qext``, 1
   ``sn-mms-spherical-psi``, 1
   ``sn-mms-spherical-qext``, 1
   ``splitting``, 1

Orphan equations
----------------

Equations with zero tests carrying ``@pytest.mark.verifies("label")``, excluding labels explicitly marked ``:vv-status: documented``. **7** of the testable equations found on theory pages are orphan.

- ``e1-decomposition``
- ``peierls-sphere-G-bc``
- ``peierls-sphere-equation``
- ``peierls-sphere-nystrom``
- ``peierls-sphere-ray-optical-depth``
- ``peierls-white-bc``
- ``vacuum-bc``

Documented-only equations
-------------------------

Theory labels marked ``.. vv-status: <label> documented`` in their RST source. These are excluded from the orphan-equation gate because they are either definitional (no single implementing function — e.g. ``boltzmann``), describe a module whose Python port does not yet exist (e.g. the thermal-hydraulics / fuel-behaviour / reactor-kinetics equations), or have a deliberately deferred test paired with a tracking issue. **47** labels carry the directive. See ``docs/testing/architecture.rst``:ref:`vv-status-documented` for the full taxonomy.

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
- ``peierls-cylinder-equation``
- ``peierls-cylinder-green-2d``
- ``peierls-cylinder-nystrom``
- ``peierls-cylinder-polar``
- ``peierls-cylinder-r-prime``
- ``peierls-cylinder-ray-optical-depth``
- ``peierls-cylinder-rho-max``
- ``peierls-cylinder-row-sum-identity``
- ``peierls-e1-derivation``
- ``peierls-ki1-derivation``
- ``peierls-point-kernel-3d``
- ``peierls-polar-jacobian-cancellation``
- ``peierls-sphere-green-3d``
- ``peierls-sphere-polar``
- ``peierls-sphere-r-prime``
- ``peierls-sphere-rho-max``
- ``peierls-sphere-row-sum-identity``
- ``peierls-unified``
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
   ``ERR-003``, 2
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
   ``ERR-025``, 4
   ``ERR-026``, 4

Unmarked tests
--------------

**11 tests** have no V&V level marker.
This is a gap — every test in the tree should carry either
a physics-ladder marker (``l0``..``l3``) or the orthogonal
``foundation`` marker (``@pytest.mark.foundation``) for
tests that verify software invariants rather than physics
equations. See ``docs/testing/architecture.rst``
:ref:`vv-foundation-tests` for the taxonomy.

.. csv-table::
   :header: File, Unmarked tests
   :widths: 60, 10

   ``tests/sn/test_boundary_conditions.py``, 11

