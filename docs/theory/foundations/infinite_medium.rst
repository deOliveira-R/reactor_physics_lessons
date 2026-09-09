.. _theory-homogeneous:

=============================================
Homogeneous Infinite-Medium Reactor
=============================================

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before modifying the homogeneous solver.**

- Balance: :math:`\mathbf{A}\phi = \frac{1}{k}\mathbf{F}\phi` where the loss
  matrix is :math:`\mathbf{A} = \text{diag}(\Sigma_t) - \Sigma_{s0}^T - 2\Sigma_2^T`
  and the production dyad is :math:`\mathbf{F} = \chi \otimes (\nu\Sigma_f)`
- **(n,2n) convention**: the :math:`(n,2n)` reaction is a **loss-side
  multiplicity-2 transfer** — it lives ONLY in :math:`\mathbf{A}` (as
  :math:`-2\Sigma_2^T`), NEVER in the production :math:`\mathbf{F}`. The two
  emitted neutrons are redistributed by :math:`2\Sigma_2`; they are not
  produced with the fission spectrum :math:`\chi`. Production is
  :math:`\nu\Sigma_f` only. (Double-counting :math:`2\,\text{colsum}(\Sigma_2)`
  into production — the retired bespoke bug — moves :math:`\kinf` by
  :math:`\sim 0.43` on the asymmetric-:math:`\Sigma_2` ``homo_2eg_n2n`` case.)
- 1-group: :math:`k = \nu\Sigma_f / \Sigma_a` (exact, no iteration)
- Multi-group: :math:`\kinf = \lambda_{\max}(\mathbf{K})`, the dominant
  eigenpair of the multiplication operator
  :math:`\mathbf{K} = \mathbf{A}^{-1}\mathbf{F}` **spelled in the operator
  algebra** — ``K = MatrixInverseOperator(loss) @ production`` — and extracted
  from the materialized :math:`[\mathbf{K}]` by the shared Perron--Frobenius
  primitive :func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`. There is
  **no power iteration**: the 0-D spectrum is an *exact* eigenproblem, so the
  direct dense inverse is used, not the iterative
  :func:`~orpheus.numerics.eigenvalue.power_iteration` the spatially-coupled
  solvers use (see :ref:`direct-eigensolve-solve`,
  :ref:`three-eigenvalue-engines`)
- **Homogeneous is the FIRST production consumer of**
  :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
  (taxonomy step 5b). Constructing the matrix inverse *explicitly* — rather
  than calling the structure-keyed ``loss.inverse()``, which would return the
  **iterative** :class:`~orpheus.numerics.green_operator.GreenOperator`
  splitting — **is** the direct-realization strategy choice, encoded as a type
  rather than a flag. :func:`~orpheus.numerics.eigenvalue.direct_eigenvalue`
  (the ``(A, F)``-posed sibling engine) is **no longer on the homogeneous call
  path**
- **A is assembled from the transport operators**, not a bespoke matrix:
  :math:`\mathbf{A} = C - K_\mathrm{iso}`, posed on
  :math:`V_E \otimes V_{\rm pt}` (the energy axis tensored with the
  quotient point) and reading its cross sections off a *meshless*
  single-cell
  :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh`. **The
  carrier supplies data; the problem poses its own space** — since
  campaign 1 CS4a (K2) the space is minted from the MIXTURE by
  :func:`~orpheus.homogeneous.solver._pose_space` and threaded into all
  three arms, not read off the carrier (which mints an ``==`` space and
  is now a reference, not the production source). With
  :math:`C = \text{diag}(\Sigma_t)` and
  :math:`K_\mathrm{iso} = \Sigma_{s0}^T + 2\Sigma_2^T` supplied by
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
  and :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`.
  Streaming :math:`L` is identically zero in an infinite medium and is dropped,
  so the whole spectrum runs through the SAME operator algebra the meshed SN
  solver uses (cross-model single source, Cardinal Rule 2; campaign #276)
- This is the reference eigenvalue for ALL solvers on homogeneous problems
- Tolerance: < 1e-12 (limited only by FP arithmetic on small dense matrices)
- **Gotcha**: this eigenvalue is flux-shape independent — it tests nothing
  about spatial or angular discretization


Overview
========

The infinite homogeneous medium is the simplest model in reactor physics.
All spatial dependence vanishes (infinite geometry), all angular
dependence integrates out (isotropic medium), and the neutron transport
equation reduces to a pure **energy balance**.  The only unknowns are the
**neutron energy spectrum** :math:`\phi(E)` and the **infinite
multiplication factor** :math:`\kinf`.

Despite its simplicity, the homogeneous model is the foundation on which
all other solvers build:

- It is the **first module** students encounter in the ORPHEUS
  curriculum, introducing the multi-group eigenvalue problem and its
  direct dense solution.
- The **cross-section preparation pipeline** — isotope loading,
  sigma-zero self-shielding, interpolation, macroscopic summation — is
  exercised here and reused unchanged by every subsequent solver (SN,
  MoC, CP, Monte Carlo, diffusion).
- Analytical eigenvalues for 1-, 2-, and 4-group homogeneous media
  serve as **verification benchmarks** for all deterministic solvers.

This chapter derives the infinite-medium eigenvalue problem from first
principles, describes the cross-section preparation pipeline, and
presents the direct dense eigensolve used to compute :math:`\kinf`
and :math:`\phi(E)`.

The solver is the single function
:func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite`, which
assembles the loss matrix from the model-shared transport operators
(see :ref:`direct-eigensolve`), takes the dominant eigenpair of
:math:`\mathbf{A}^{-1}\mathbf{F}`, and returns a
:class:`~orpheus.homogeneous.solver.HomogeneousResult`.


From the Boltzmann Equation to the Infinite Medium
====================================================

The Boltzmann Transport Equation
---------------------------------

The starting point is the steady-state neutron transport equation in its
integro-differential form :cite:`Duderstadt1976`:

.. math::
   :label: boltzmann

   \hat{\Omega} \cdot \nabla \psi(\mathbf{r}, \hat{\Omega}, E)
   + \Sigma_\mathrm{t}(\mathbf{r}, E) \, \psi(\mathbf{r}, \hat{\Omega}, E)
   = \int_0^\infty \!\!\int_{4\pi}
     \Sigma_\mathrm{s}(\mathbf{r}, E' \!\to\! E, \hat{\Omega}' \!\to\! \hat{\Omega})
     \, \psi(\mathbf{r}, \hat{\Omega}', E') \, d\Omega' \, dE'
   + \frac{\chi(E)}{4\pi \, k}
     \int_0^\infty \nu\Sigma_\mathrm{f}(\mathbf{r}, E')
     \, \phi(\mathbf{r}, E') \, dE'

.. vv-status: boltzmann documented

Here :math:`\psi(\mathbf{r}, \hat{\Omega}, E)` is the :term:`angular flux`,
:math:`\phi(\mathbf{r}, E) = \int_{4\pi} \psi \, d\Omega` is the scalar
flux, :math:`\chi(E)` is the fission spectrum, and :math:`k` is the
multiplication factor eigenvalue.


Simplification for the Infinite Homogeneous Medium
----------------------------------------------------

Three physical conditions dramatically simplify Eq. :eq:`boltzmann`:

1. **Infinite geometry** — no boundaries, so the flux is spatially
   uniform: :math:`\nabla \psi = 0`.  The streaming term vanishes
   entirely, and with it all leakage.

2. **Homogeneous medium** — all cross sections are independent of
   position: :math:`\Sigma_x(\mathbf{r}, E) = \Sigma_x(E)`.

3. **Isotropy** — in an infinite homogeneous medium with isotropic
   sources, the angular flux is isotropic:
   :math:`\psi(\hat{\Omega}, E) = \phi(E) / 4\pi`.  The scattering
   kernel reduces to its :math:`P_0` (isotropic) component.

After integrating over all directions, the transport equation collapses
to a **one-dimensional energy balance**:

.. math::
   :label: inf-hom-balance

   \Sigt{} \phi(E)
   = \int_0^\infty \Sigma_{\mathrm{s},0}(E' \!\to\! E) \, \phi(E') \, dE'
     + \frac{\chi(E)}{k}
       \int_0^\infty \nu\Sigma_\mathrm{f}(E') \, \phi(E') \, dE'


.. implements:: inf-hom-balance
   :by: orpheus.homogeneous.solver.HomogeneousProblem.loss

   **Implemented by** 7 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: inf-hom-balance
   :by: orpheus.homogeneous.solver.solve_homogeneous_infinite

.. implements:: inf-hom-balance
   :by: orpheus.derivations.common.eigenvalue._infinite_medium_matrices

.. implements:: inf-hom-balance
   :by: orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous

.. implements:: inf-hom-balance
   :by: orpheus.derivations.common.eigenvalue.kinf_homogeneous

.. implements:: inf-hom-balance
   :by: orpheus.derivations.continuous.analytical.homogeneous.derive_1g

.. implements:: inf-hom-balance
   :by: orpheus.derivations.continuous.analytical.homogeneous.derive_1g_continuous

where :math:`\Sigma_{\mathrm{s},0}` is the isotropic scattering kernel.

.. note::

   In the infinite homogeneous medium, scattering merely redistributes
   neutrons in energy.  It does not change the total production-to-loss
   ratio, so :math:`\kinf` depends only on the fission and absorption
   cross sections.  Scattering does, however, determine the **shape** of
   the neutron spectrum :math:`\phi(E)` — specifically the 1/E
   slowing-down region and the thermal Maxwellian peak.


Multi-Group Energy Discretisation
==================================

Group-Averaged Cross Sections
------------------------------

The continuous energy variable is discretised into :math:`G` groups.
The fastest group carries the highest energies, the last group the
lowest (thermal neutrons): the descending boundary array
:math:`E_0 > E_1 > \cdots > E_G` defines the grid stored in
:attr:`Mixture.eg` for production cases (XS computed from ENDF
:class:`~orpheus.data.micro_xs.isotope.Isotope` data via
:func:`compute_macro_xs`). This is the :ref:`canonical fast-first
energy-group convention <canonical-group-convention>`; the code index
runs :math:`g = 0` (fastest) to :math:`g = G-1` (thermal), while the
1-based labels used in the equations below (group 1 = fastest) are a
presentation choice for the slowing-down algebra. For synthetic verification cases (Sood-style
abstract XS, MMS test mixtures), :attr:`Mixture.eg` is ``None`` —
there is no real grid, only a discrete set of group cross-sections.
Per-energy diagnostics (:term:`lethargy` widths, flux-per-energy plots,
spectrum-weighted condensation) require the grid to be populated and
gracefully skip the synthetic-XS path.

The group flux is the integral over the group's energy interval:

.. math::
   :label: group-flux

   \phi_g = \int_{E_g}^{E_{g-1}} \phi(E) \, dE

.. vv-status: group-flux documented

Group-averaged cross sections are **flux-weighted** averages:

.. math::
   :label: group-xs

   \Sigt{g} = \frac{1}{\phi_g} \int_{E_g}^{E_{g-1}} \Sigt{}(E) \, \phi(E) \, dE

.. vv-status: group-xs documented

In practice, these averages are pre-computed and stored in the
421-group HELIOS library that ships with ORPHEUS.  The library provides
cross sections tabulated at several background cross section
(:math:`\sigma_0`) values; the sigma-zero iteration (see
:ref:`sigma-zero-iteration`) selects the appropriate value for each
isotope and group.


.. _mg-eigenvalue-problem:

The Multi-Group Neutron Balance
--------------------------------

Substituting group-averaged quantities into Eq. :eq:`inf-hom-balance`
gives the **multi-group neutron balance** for group :math:`g`:

.. math::
   :label: mg-balance

   \Sigt{g} \, \phi_g
   = \sum_{g'=1}^{G} \Sigs{g' \to g} \, \phi_{g'}
     + \frac{\chi_g}{k} \sum_{g'=1}^{G} \nSigf{g'} \, \phi_{g'}


.. implements:: mg-balance
   :by: orpheus.cp.solver.CPSolver._compute_balance_residual

   **Implemented by** 12 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

   Four of the twelve were re-pointed on 2026-09-04 (#426 step 2), when
   the two collision-gain channels collapsed onto one family: the
   per-material P\ :sub:`0` verb and the scalar energy binding's
   ``apply`` moved to the shared
   :class:`~orpheus.transport.material_field.TransferMaterialField` /
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicTransfer`
   cores, ``LegendreMomentScattering`` became
   :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`
   — the ONE :math:`\Lambda` both channels use — and the retired
   ``N2NMomentOperator`` row is now the :math:`(n,2n)` term itself,
   :class:`~orpheus.transport.operators.n2n.N2NOperator`, since its
   moment factor is no longer a class of its own.

.. implements:: mg-balance
   :by: orpheus.homogeneous.solver.HomogeneousProblem.loss

.. implements:: mg-balance
   :by: orpheus.homogeneous.solver.solve_homogeneous_infinite

.. implements:: mg-balance
   :by: orpheus.moc.core.MOCSolver.solve_fixed_source

.. implements:: mg-balance
   :by: orpheus.transport.material_field.TransferMaterialField.add_p0_source

.. implements:: mg-balance
   :by: orpheus.transport.operators.isotropic_transfer.IsotropicFission.apply

.. implements:: mg-balance
   :by: orpheus.transport.operators.isotropic_transfer.IsotropicTransfer.apply

.. implements:: mg-balance
   :by: orpheus.transport.operators.transfer.LegendreMomentTransfer

.. implements:: mg-balance
   :by: orpheus.transport.operators.n2n.N2NOperator

.. implements:: mg-balance
   :by: orpheus.transport.operators.scattering.ScatteringOperator

.. implements:: mg-balance
   :by: orpheus.derivations.common.eigenvalue._infinite_medium_matrices

.. implements:: mg-balance
   :by: orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous

The first term on the right is in-scattering from all groups
(including self-scattering :math:`g' = g`), and the second is the
fission source weighted by the fission spectrum :math:`\chi_g`.


Matrix Form
------------

Collecting all :math:`G` group equations into vectors and matrices:

.. math::
   :label: matrix-eigenvalue

   \mathbf{A} \, \boldsymbol{\phi}
   = \frac{1}{k} \, \mathbf{F} \, \boldsymbol{\phi}

where the **removal matrix** and **fission matrix** are:

.. math::
   :label: removal-matrix

   \mathbf{A} = \mathrm{diag}(\Sigt{g})
                - \boldsymbol{\Sigma}_{\mathrm{s}}^T
                - 2 \, \boldsymbol{\Sigma}_2^T

.. math::
   :label: fission-matrix

   \mathbf{F} = \boldsymbol{\chi} \otimes \nu\boldsymbol{\Sigma}_\mathrm{f}

Here :math:`\boldsymbol{\Sigma}_{\mathrm{s}}` is the :math:`G \times G`
scattering transfer matrix (:math:`P_0` component) and
:math:`\boldsymbol{\Sigma}_2` is the :math:`(n,2n)` transfer matrix. The
production matrix :math:`\mathbf{F}` is the **rank-1 dyad**
:math:`\boldsymbol{\chi} \otimes \nu\boldsymbol{\Sigma}_\mathrm{f}`
embodied by
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
— a group contraction onto the production rate followed by a broadcast
across the emission spectrum :math:`\boldsymbol{\chi}`.

.. note::

   **The** :math:`(n,2n)` **reaction appears ONLY in the loss matrix**
   :math:`\mathbf{A}` (as :math:`-2\boldsymbol{\Sigma}_2^T`), never in
   the production :math:`\mathbf{F}`.  The :math:`(n,2n)` event is a
   **loss-side multiplicity-2 transfer**: one neutron of group
   :math:`g'` is removed and **two** neutrons are deposited into the
   scattering system with the :math:`(n,2n)` energy-transfer kernel
   :math:`2\boldsymbol{\Sigma}_2(g' \!\to\! g)`.  The factor of two is
   the emission multiplicity; the transpose puts the *source* group on
   the row exactly as for :math:`\boldsymbol{\Sigma}_{\mathrm{s}}^T`
   (see :ref:`scattering-matrix-convention`).

   The two emitted neutrons are **not** produced with the fission
   spectrum :math:`\boldsymbol{\chi}` — they carry the :math:`(n,2n)`
   transfer kernel, not :math:`\boldsymbol{\chi}`.  Production is
   :math:`\nu\boldsymbol{\Sigma}_\mathrm{f}` only.  This matches the
   analytical oracle
   :func:`~orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous`
   (:math:`\mathbf{A} = \text{diag}(\Sigma_t) - (\Sigma_s + 2\Sigma_2)^T`,
   :math:`\mathbf{F} = \chi \otimes \nu\Sigma_f`) and the collision-probability
   oracle :func:`~orpheus.derivations.common.eigenvalue.kinf_from_cp`.

   .. warning::

      A retired bespoke formulation put :math:`(n,2n)` in **both**
      matrices — :math:`+2\,\text{colsum}(\boldsymbol{\Sigma}_2)` in the
      production numerator as well as :math:`-2\boldsymbol{\Sigma}_2^T`
      in the loss.  That double-counts the :math:`(n,2n)` neutrons.  On
      the asymmetric-:math:`\boldsymbol{\Sigma}_2` ``homo_2eg_n2n`` case
      it moves :math:`\kinf` from the correct ``1.6532`` to ``2.08`` — a
      :math:`\sim 0.43` error, far above the FP floor.  See the
      :ref:`direct-eigensolve` section for the live assembly and the
      ``homo_2eg_n2n`` de-vacuum case.

   The loss matrix :math:`\mathbf{A} = C - K_\mathrm{iso}` is assembled
   from the transport operators
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
   (:math:`\Sigma_{s0}^T`) and
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`
   (:math:`2\Sigma_2^T`); the production dyad :math:`\mathbf{F}` is the
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
   rank-1 form (materialised densely as
   :math:`\chi \otimes \nu\Sigma_f`).  See
   :func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite`.

The eigenvalue :math:`k = \kinf` is the largest eigenvalue of the
generalised problem :eq:`matrix-eigenvalue`.  By the Perron–Frobenius
theorem :cite:`Hebert2009`, the dominant eigenvector :math:`\boldsymbol{\phi}`
is the unique non-negative solution — the **fundamental mode** — which
is the physically meaningful neutron spectrum.


.. _scattering-matrix-convention:

Scattering Matrix Convention
-----------------------------

The scattering transfer matrix :math:`\boldsymbol{\Sigma}_{\mathrm{s}}`
is stored in the **from-row, to-column** convention:

.. math::
   :label: sigs-convention

   (\boldsymbol{\Sigma}_{\mathrm{s}})_{g',g}
   = \Sigs{g' \to g}

.. vv-status: sigs-convention documented

That is, row :math:`g'` gives the **source group** and column :math:`g`
gives the **destination group**.  A downscatter-only matrix is therefore
**upper-triangular**: non-zero entries only on or above the diagonal,
because :math:`\Sigs{g' \to g} = 0` when :math:`g < g'` (no neutrons
scatter from thermal to fast).  Its transpose — the form that acts on
:math:`\boldsymbol{\phi}` in the in-scattering sum below — is
correspondingly lower-triangular, which is why the two-group operator
:eq:`two-group-A` carries its off-diagonal entry
:math:`-\Sigs{1 \to 2}` below the diagonal.

The neutron balance :eq:`mg-balance` requires the **in-scattering** into
group :math:`g` from all groups :math:`g'`:

.. math::
   :label: sigs-in-scatter-transpose

   \sum_{g'} \Sigs{g' \to g} \phi_{g'}
   = \bigl(\boldsymbol{\Sigma}_{\mathrm{s}}^T \cdot \boldsymbol{\phi}\bigr)_g

.. vv-status: sigs-in-scatter-transpose documented
.. Representational convention identity: the in-scatter sum equals the transpose
.. matvec Sig_s^T phi (the from-row / to-column convention). Its terminal use is
.. the removal matrix (removal-matrix), verified end-to-end by the multi-group
.. homogeneous chain (tests/homogeneous/test_homogeneous.py verifies
.. "removal-matrix", >=2 groups per the ERR-002 warning). A convention identity,
.. not a separate solver claim.

This is why the removal matrix :eq:`removal-matrix` uses the
**transpose** :math:`\boldsymbol{\Sigma}_{\mathrm{s}}^T`: the transpose
converts column :math:`g` (destination) to row :math:`g`, making the
matrix-vector product give the in-scattering rate per group.

.. warning::

   Getting the transpose wrong is a common source of bugs (see
   ERR-002 in the error catalog).  For symmetric scattering matrices
   (e.g., 1-group self-scatter), the transpose is invisible, and the
   bug only manifests in multi-group problems with asymmetric
   down-scatter.  This is why verification must always include
   :math:`\geq 2` groups.

The :class:`~data.macro_xs.mixture.Mixture` stores ``SigS`` as a
list of :math:`G \times G` sparse matrices, one per Legendre order.
The :math:`P_0` component ``SigS[0]`` is used by the homogeneous
solver; the higher orders are used by transport solvers with
anisotropic scattering (SN, MoC).


Analytical Solutions
=====================

One-Group Theory
-----------------

For a single energy group, the matrices reduce to scalars.  The
scattering terms cancel (a neutron scattered in group 1 remains in
group 1), and the eigenvalue problem gives immediately:

.. math::
   :label: one-group-kinf

   \kinf = \frac{\nu \Sigf{}}{\Siga{}}

.. verifies:: one-group-kinf
   :by: orpheus.derivations.continuous.analytical.homogeneous.derive_1g

   Verified analytically (exact closed-form ratio) against the
   ``homo_1eg`` :class:`~orpheus.derivations.common.verification_case.VerificationCase`.

This is the most fundamental result in reactor physics.  It states that
:math:`\kinf` is the ratio of neutron production to neutron absorption,
which is the definition of the infinite multiplication factor :cite:`Stacey2007`.

For the connection to the **four-factor formula**: in a single-material
homogeneous medium the thermal utilisation :math:`f = 1`, the resonance
escape probability :math:`p = 1` (no spatial heterogeneity), and the
fast fission factor :math:`\varepsilon = 1`, so :math:`\kinf = \eta \cdot f
\cdot p \cdot \varepsilon = \eta`.

**Numerical example** (from :func:`orpheus.derivations.continuous.analytical.homogeneous.derive_1g`):
:math:`\Sigt{} = 1.0`, :math:`\Sigma_\mathrm{c} = 0.2`,
:math:`\Sigf{} = 0.3`, :math:`\nu = 2.5`,
:math:`\Sigs{} = 0.5` cm\ :sup:`-1`:

.. math::

   \kinf = \frac{2.5 \times 0.3}{0.2 + 0.3} = 1.500000


Two-Group Theory
-----------------

For two energy groups (fast and thermal) with downscatter only
(:math:`\chi = [1, 0]`, no upscatter from thermal to fast), the
matrices are:

.. math::
   :label: two-group-A

   \mathbf{A} = \begin{pmatrix}
     \Sigt{1} - \Sigs{1 \to 1} & 0 \\
     -\Sigs{1 \to 2} & \Sigt{2} - \Sigs{2 \to 2}
   \end{pmatrix}


.. implements:: two-group-A
   :by: orpheus.homogeneous.solver.HomogeneousProblem.loss

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: two-group-A
   :by: orpheus.derivations.common.eigenvalue._infinite_medium_matrices

.. implements:: two-group-A
   :by: orpheus.derivations.continuous.analytical.homogeneous.derive_2g

.. math::
   :label: two-group-F

   \mathbf{F} = \begin{pmatrix}
     \nu_1 \Sigf{1} & \nu_2 \Sigf{2} \\
     0 & 0
   \end{pmatrix}


.. implements:: two-group-F
   :by: orpheus.transport.operators.isotropic_transfer.IsotropicFission

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others. (The transport-side site was ``FissionOperator`` until CS4c
   step 4 rebound the channel — the infinite-medium problem carries no
   angular axis, so the *energy* binding is the one that executes this
   dyad; :ref:`fission-as-dyad`.)

.. implements:: two-group-F
   :by: orpheus.derivations.common.eigenvalue._infinite_medium_matrices

.. implements:: two-group-F
   :by: orpheus.derivations.continuous.analytical.homogeneous.derive_2g

Note that :math:`\mathbf{A}` is lower-triangular because there is no
upscatter (:math:`\Sigs{2 \to 1} = 0`).  This makes the inverse
analytical:

.. math::
   :label: two-group-Ainv

   \mathbf{A}^{-1} = \begin{pmatrix}
     \dfrac{1}{\Sigma_{\mathrm{r},1}} & 0 \\[8pt]
     \dfrac{\Sigs{1 \to 2}}{\Sigma_{\mathrm{r},1} \, \Sigma_{\mathrm{r},2}}
     & \dfrac{1}{\Sigma_{\mathrm{r},2}}
   \end{pmatrix}


.. implements:: two-group-Ainv
   :by: orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: two-group-Ainv
   :by: orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator.as_matrix

.. implements:: two-group-Ainv
   :by: orpheus.derivations.continuous.fn_method.origins.k_inf_derivations.derive_kinf_mg_matrix_form

where :math:`\Sigma_{\mathrm{r},g} = \Sigt{g} - \Sigs{g \to g}` is
the **removal cross section** for group :math:`g` (total minus
in-group scattering = absorption + out-scattering).

The eigenvalue matrix :math:`\mathbf{M} = \mathbf{A}^{-1}\mathbf{F}`
is:

.. math::
   :label: two-group-M

   \mathbf{M} = \begin{pmatrix}
     \dfrac{\nu_1 \Sigf{1}}{\Sigma_{\mathrm{r},1}}
     & \dfrac{\nu_2 \Sigf{2}}{\Sigma_{\mathrm{r},1}} \\[8pt]
     \dfrac{\Sigs{1 \to 2}\, \nu_1 \Sigf{1}}
           {\Sigma_{\mathrm{r},1}\,\Sigma_{\mathrm{r},2}}
     & \dfrac{\Sigs{1 \to 2}\, \nu_2 \Sigf{2}}
             {\Sigma_{\mathrm{r},1}\,\Sigma_{\mathrm{r},2}}
   \end{pmatrix}


.. implements:: two-group-M
   :by: orpheus.homogeneous.solver.solve_homogeneous_infinite

   **Implemented by** 3 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: two-group-M
   :by: orpheus.numerics.eigenvalue.direct_eigenvalue

.. implements:: two-group-M
   :by: orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous

Because the fission source enters only group 1 (:math:`\chi = [1, 0]`,
so the second row of :math:`\mathbf{F}` is zero), the term
:math:`\nu_2\Sigf{2}/\Sigma_{\mathrm{r},2}` does **not** appear in
:math:`M_{22}`: group 2 absorbs and down-scatters but produces no
fission emission of its own.  Consequently the two rows of
:math:`\mathbf{M}` are proportional — :math:`\mathbf{M}` is **rank 1**
(as it must be, since :math:`\mathbf{F} = \boldsymbol{\chi}\otimes\nu\Sigf{}`
is a rank-1 dyad) — and its only non-zero eigenvalue is its trace.

The characteristic equation :math:`\det(\mathbf{M} - \lambda\mathbf{I}) = 0`
gives a quadratic in :math:`\lambda`:

.. math::
   :label: two-group-charpoly

   \lambda^2 - \bigl(M_{11} + M_{22}\bigr)\lambda
   + \bigl(M_{11}M_{22} - M_{12}M_{21}\bigr) = 0


.. implements:: two-group-charpoly
   :by: orpheus.derivations.continuous.fn_method.origins.k_inf_derivations.derive_kinf_mg_matrix_form

   **Implemented by** the one site in the tree that executes this
   equation's arithmetic.

whose roots are:

.. math::
   :label: two-group-roots

   \lambda_{\pm} = \frac{(M_{11} + M_{22})
                   \pm \sqrt{(M_{11} - M_{22})^2 + 4 M_{12} M_{21}}}{2}


.. implements:: two-group-roots
   :by: orpheus.numerics.eigenvalue.dominant_eigenpair

   **Implemented by** 2 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: two-group-roots
   :by: orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous

The dominant root :math:`\lambda_+` is :math:`\kinf`.

**Worked numerical example** (from :func:`orpheus.derivations.continuous.analytical.homogeneous.derive_2g`):

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 15 15 15

   * - :math:`g`
     - :math:`\Sigt{}`
     - :math:`\Sigma_\mathrm{c}`
     - :math:`\Sigf{}`
     - :math:`\nu`
     - :math:`\Sigs{g \to g}`
     - :math:`\Sigs{1 \to 2}`
   * - 1
     - 0.50
     - 0.01
     - 0.01
     - 2.50
     - 0.38
     - 0.10
   * - 2
     - 1.00
     - 0.02
     - 0.08
     - 2.50
     - 0.90
     - ---

The removal cross sections are :math:`\Sigma_{\mathrm{r},1} = 0.50 - 0.38 = 0.12`
and :math:`\Sigma_{\mathrm{r},2} = 1.00 - 0.90 = 0.10`.

The eigenvalue matrix entries are:

.. math::

   M_{11} &= \frac{2.50 \times 0.01}{0.12} = 0.208\overline{3} \\[4pt]
   M_{12} &= \frac{2.50 \times 0.08}{0.12} = 1.6\overline{6} \\[4pt]
   M_{21} &= \frac{0.10 \times 2.50 \times 0.01}{0.12 \times 0.10} = 0.208\overline{3} \\[4pt]
   M_{22} &= \frac{0.10 \times 2.50 \times 0.08}{0.12 \times 0.10} = 1.6\overline{6}

The two rows are identical (rank-1 :math:`\mathbf{M}`), so the
characteristic polynomial :eq:`two-group-charpoly` factors as
:math:`\lambda\,(\lambda - \operatorname{tr}\mathbf{M}) = 0` and the
dominant root :eq:`two-group-roots` is simply the trace:

.. math::

   \kinf = \lambda_+ = M_{11} + M_{22}
         = 0.208\overline{3} + 1.6\overline{6} = 1.8750000000

The second eigenvalue is :math:`\lambda_- = 0`.  This is exact and
structural, not a coincidence of these cross sections: the production
matrix :math:`\mathbf{F} = \boldsymbol{\chi} \otimes \nu\Sigma_f` is a
**rank-1 dyad** (fission emits with the single spectrum
:math:`\boldsymbol{\chi}`), so :math:`\mathbf{M} = \mathbf{A}^{-1}\mathbf{F}`
is also rank 1 — it has exactly one non-zero eigenvalue,
:math:`\kinf`, regardless of the group count.  The direct dense
eigensolve (see :ref:`direct-eigensolve`) returns this dominant
eigenpair immediately; there is no iteration whose convergence rate
would depend on a dominance ratio.

.. note::

   The large :math:`\kinf` in these analytical benchmarks reflects the
   synthetic cross sections chosen for verification, not a physical
   reactor.  The cross sections are deliberately simple to enable exact
   symbolic solutions.


Four-Group Theory
------------------

For four groups (fast, epithermal, thermal-1, thermal-2) with a full
downscatter cascade and fission in all groups, the characteristic
polynomial is degree 4 and has no convenient closed form.  The
analytical eigenvalue is computed numerically by SymPy's symbolic
eigenvalue solver applied to the :math:`4 \times 4` matrix
:math:`\mathbf{A}^{-1} \mathbf{F}`.

**Result** (from :func:`orpheus.derivations.continuous.analytical.homogeneous.derive_4g`):

.. math::

   \kinf = 1.4877619048

.. warning::

   The analytical eigenvalues are computed from the **same matrix
   structure** as the numerical solver.  This is a code-verification
   test (does the code correctly implement the matrix algebra?), not a
   physics-validation test.  Independent validation requires comparison
   to a different code or to experimental data (see the MATLAB
   reference values in the demo scripts).


.. _xs-preparation:

Cross-Section Preparation
==========================

Before any solver can run, the macroscopic cross sections must be
assembled from isotopic data.  This section describes the pipeline
implemented in :mod:`data.macro_xs`, which is exercised for the first
time in the homogeneous module and reused by all subsequent solvers.


.. _xs-pipeline-overview:

Pipeline Overview
------------------

The cross-section preparation follows five steps:

1. **Load isotopes** — read the 421-group microscopic cross-section
   library for each nuclide at the desired temperature.
2. **Compute number densities** — convert mass densities and
   compositions to number densities in the library's unit system.
3. **Sigma-zero iteration** — find the self-consistent background cross
   section for each isotope and group (self-shielding).
4. **Interpolate** — evaluate microscopic cross sections at the
   converged sigma-zero values.
5. **Sum to macroscopic** — weight by number densities and sum over
   isotopes to obtain the :class:`~data.macro_xs.mixture.Mixture`.

This pipeline is encapsulated in
:func:`~data.macro_xs.mixture.compute_macro_xs`.


Number Densities
-----------------

The atomic number density of species :math:`i` (in :math:`1/(\text{barn}
\cdot \text{cm})`) is:

.. math::
   :label: number-density

   N_i = \frac{\rho_i}{m_u \, A_i}


.. implements:: number-density
   :by: orpheus.data.macro_xs.recipes._number_density

   **Implemented by** the one site in the tree that executes this
   equation's arithmetic.

where :math:`\rho_i` is the partial mass density in
:math:`\text{g}/\text{cm}^3`, :math:`m_u = 1.660538 \times 10^{-24}` g
is the atomic mass unit, and :math:`A_i` is the atomic weight.  The
factor :math:`10^{-24}` converts the natural units
(:math:`\text{cm}^{-3}`) to the library units
(:math:`1/(\text{barn} \cdot \text{cm})`).

For aqueous solutions, the water density is obtained from the IAPWS-IF97
steam tables via ``pyXSteam``.  See
:func:`~data.macro_xs.recipes.aqueous_uranium` and
:func:`~data.macro_xs.recipes.pwr_like_mix`.


.. _sigma-zero-iteration:

Sigma-Zero Self-Shielding
---------------------------

Cross sections in the resonance region depend strongly on the
**background cross section** :math:`\sigma_{0,i,g}` — a measure of how
"dilute" isotope :math:`i` is relative to its neighbours.  The
background cross section is defined as :cite:`Bondarenko1964`:

.. math::
   :label: sigma-zero

   \sigma_{0,i,g}
   = \frac{\Sigma_\mathrm{escape} + \displaystyle\sum_{j \ne i}
           N_j \, \sigma_{\mathrm{t},j,g}}{N_i}

where :math:`\Sigma_\mathrm{escape}` is the escape cross section
(zero for an infinite homogeneous medium) and the sum runs over all
other isotopes in the mixture.

**Physical meaning**: when :math:`\sigma_0` is large (dilute limit or
strong moderator), the resonance peaks are fully resolved and the
effective cross section is close to the infinite-dilution value.  When
:math:`\sigma_0` is small (concentrated heavy absorber), the neutron
flux is depressed at resonance energies — **self-shielding** — and the
effective cross section is reduced.

The definition :eq:`sigma-zero` is implicit: :math:`\sigma_{\mathrm{t},j,g}`
itself depends on :math:`\sigma_{0,j,g}` through the library
interpolation tables.  The solution is obtained by **fixed-point
iteration**:

1. Initialise :math:`\sigma_0` to a large value (:math:`10^{10}` barns,
   the infinite-dilution limit).
2. Interpolate :math:`\sigma_{\mathrm{t},j,g}` from the library at the
   current :math:`\sigma_0`.
3. Recompute :math:`\sigma_0` from Eq. :eq:`sigma-zero`.
4. Repeat until :math:`\|\sigma_0^{(n)} - \sigma_0^{(n-1)}\| < 10^{-6}`.

Convergence is fast (typically 3--5 iterations) because the dependence
of :math:`\sigma_\mathrm{t}` on :math:`\sigma_0` is weak and monotonic.
This is implemented in :func:`~data.macro_xs.sigma_zeros.solve_sigma_zeros`.

.. note::

   For an **infinite homogeneous** medium, :math:`\Sigma_\mathrm{escape}
   = 0`.  The sigma-zero depends only on the other isotopes in the
   mixture.  For **heterogeneous** cells (fuel pins), the escape cross
   section :math:`\Sigma_e = \Sigma_\mathrm{pot} / \bar{\ell}
   \approx S/(4V)` accounts for spatial self-shielding via the
   equivalence theory of Bondarenko :cite:`Bondarenko1964`.


Cross-Section Interpolation
-----------------------------

The 421-group library tabulates microscopic cross sections at discrete
:math:`\sigma_0` base points (e.g., :math:`10^0, 10^1, \ldots, 10^{10}`
barns).  Once the sigma-zero iteration converges, the cross section at
the converged :math:`\sigma_0` is obtained by **log-linear interpolation**
in :math:`\log_{10}(\sigma_0)` space:

.. math::
   :label: xs-interp

   \sigma_{x,g}(\sigma_0) \approx \sigma_{x,g}(\sigma_0^{(a)})
   + \frac{\log_{10} \sigma_0 - \log_{10} \sigma_0^{(a)}}
          {\log_{10} \sigma_0^{(b)} - \log_{10} \sigma_0^{(a)}}
     \bigl[\sigma_{x,g}(\sigma_0^{(b)}) - \sigma_{x,g}(\sigma_0^{(a)})\bigr]

where :math:`\sigma_0^{(a)}` and :math:`\sigma_0^{(b)}` are the
bracketing base points.  This is performed by
:func:`~data.macro_xs.interpolation.interp_xs_field` for scalar
cross sections and
:func:`~data.macro_xs.interpolation.interp_sig_s` for scattering
matrices.


Macroscopic Summation
----------------------

The macroscopic cross section for reaction :math:`x` in group :math:`g`
is the density-weighted sum over all isotopes:

.. math::
   :label: macro-sum

   \Sigma_{x,g} = \sum_{i=1}^{I} N_i \, \sigma_{x,i,g}

The following reaction types are assembled:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Attribute
     - Reaction
     - Notes
   * - ``SigC``
     - :math:`(n,\gamma)` capture
     - Radiative capture
   * - ``SigL``
     - :math:`(n,\alpha)` loss
     - Charged-particle emission
   * - ``SigF``
     - :math:`(n,f)` fission
     - Fission cross section
   * - ``SigP``
     - Production
     - :math:`\nu\Sigf{}`, summed over **fissile** isotopes only
   * - ``SigS``
     - Scattering matrices
     - One :math:`G \times G` sparse matrix per Legendre order
   * - ``Sig2``
     - :math:`(n,2n)` matrix
     - :math:`G \times G` sparse transfer matrix
   * - ``SigT``
     - Total
     - :math:`\Sigma_\mathrm{c} + \Sigma_\mathrm{L} + \Sigma_\mathrm{f}
       + \text{rowsum}(\Sigma_\mathrm{s}^{P_0})
       + \text{rowsum}(\Sigma_2)`
   * - ``chi``
     - Fission spectrum
     - Taken from first fissile isotope (simplification)

The **absorption cross section** — the diagnostic one-group balance
ratio reported alongside :math:`\kinf` (and the denominator of the
classical production/absorption form of the eigenvalue, Eq.
:eq:`keff-update`) — is not stored directly but computed as a derived
property (:attr:`~data.macro_xs.mixture.Mixture.absorption_xs`):

.. math::
   :label: absorption-xs

   \Siga{g} = \Sigf{g} + \Sigma_{\mathrm{c},g}
            + \Sigma_{\mathrm{L},g}
            + \text{rowsum}(\boldsymbol{\Sigma}_{2,g})

This includes fission (neutron is absorbed to produce fission
fragments), radiative capture :math:`(n,\gamma)`, charged-particle
emission :math:`(n,\alpha)`, and the :math:`(n,2n)` reaction (where
one neutron is "absorbed" and two are emitted, for a net gain of one).

The result is stored in a :class:`~data.macro_xs.mixture.Mixture`
dataclass, which is the universal input to all ORPHEUS solvers.


Neutron Spectrum Physics
=========================

The shape of the neutron energy spectrum :math:`\phi(E)` in a
homogeneous medium is controlled by the competition between
moderation (slowing-down) and absorption.  Three distinct energy
regions are visible in the spectrum plots:

Fast Region (:math:`E > 0.1` MeV)
-----------------------------------

Neutrons are born in fission with a spectrum peaked around 2 MeV.
At these energies, scattering is nearly isotropic in the
centre-of-mass frame and the mean logarithmic energy loss per
collision with hydrogen is :math:`\xi = 1`.  The fission source
produces the characteristic fast peak.

For heavy nuclei like :sup:`238`\ U, the elastic energy loss
per collision is very small (:math:`\xi \approx 2/A`), so
the spectrum in the fast range is close to the fission spectrum
:math:`\chi(E)`.

Slowing-Down Region (:math:`1\;\text{eV} < E < 0.1\;\text{MeV}`)
-------------------------------------------------------------------

In this intermediate range, neutrons are slowed by elastic
scattering (primarily with hydrogen).  In the absence of absorption,
the slowing-down equation yields the well-known **1/E flux** law:

.. math::
   :label: one-over-E

   \phi(E) = \frac{S}{\xi \Sigt{}} \cdot \frac{1}{E}

.. vv-status: one-over-E documented

where :math:`S` is the slowing-down source (neutrons entering from
above) and :math:`\xi` is the mean logarithmic energy decrement.
On a **flux-per-lethargy** plot, the 1/E region appears as a
horizontal plateau:

.. math::
   :label: flux-per-lethargy-plateau

   \frac{\phi}{du} = \frac{\phi(E) \cdot E}{\Delta u}
   \propto \frac{1}{E} \cdot E = \text{const}

.. vv-status: flux-per-lethargy-plateau documented
.. Definitional physics identity: the 1/E slowing-down flux appears flat on a
.. per-lethargy plot (phi/du ~ (1/E)*E = const), the plotting-convention sibling
.. of the 1/E law (one-over-E). A spectral-physics teaching identity, not a
.. solver claim.

This is why flux-per-lethargy is the standard representation: it
makes the slowing-down region flat, and deviations (resonance dips
from :sup:`238`\ U, thermal peak) are immediately visible.

Resonance absorption (:sup:`238`\ U capture resonances) creates
**dips** in the spectrum throughout this range.  The sigma-zero
self-shielding (see :ref:`sigma-zero-iteration`) accounts for the
flux depression in the resonance peaks.

Thermal Region (:math:`E < 1` eV)
------------------------------------

Below about 1 eV, neutrons reach thermal equilibrium with the
moderator atoms.  The thermal flux approaches a **Maxwell–Boltzmann
distribution** at the moderator temperature :math:`T`:

.. math::
   :label: maxwellian

   \phi_\mathrm{th}(E) \propto E \, \exp\!\left(-\frac{E}{k_B T}\right)

.. vv-status: maxwellian documented

which peaks at :math:`E_\mathrm{peak} = k_B T`.  At room temperature
(294 K), :math:`k_B T = 0.0253` eV, producing the characteristic
thermal peak near 0.025 eV.

At higher moderator temperatures (e.g., 600 K in a PWR), the peak
shifts to higher energies and broadens — this is Doppler broadening
of the moderator distribution, which affects the thermal spectrum
shape and hence the thermal cross sections.

Absorber poisons (e.g., boron) selectively remove thermal neutrons,
depressing the thermal peak.  This is clearly visible comparing the
aqueous spectrum (no boron, strong thermal peak) with the PWR-like
spectrum (4000 ppm B, suppressed thermal peak) in the
:ref:`example-problems` section below.


.. _power-iteration-algorithm:
.. _direct-eigensolve:

The Eigenvalue Solution
=======================

The eigenvalue problem :eq:`matrix-eigenvalue`,
:math:`\mathbf{A}\boldsymbol{\phi} = \tfrac{1}{k}\mathbf{F}\boldsymbol{\phi}`,
asks for the dominant eigenpair :math:`(\kinf, \boldsymbol{\phi})`.  How
it is solved depends on whether the problem couples space:

- **Spatially-coupled solvers** (SN, CP, MoC, diffusion) cannot afford a
  dense inverse of the full loss operator, so they sweep/solve the loss
  once per outer step and drive :math:`k` up the dominant mode by
  **power iteration** on the fission source :cite:`Hebert2009`.  Those
  realisations live in the spatial theory pages (e.g.
  :eq:`cp-keff-update`, :eq:`moc-keff-update`); this section is the
  shared conceptual hub they cross-reference.
- **The infinite homogeneous medium** has no spatial coupling — the loss
  matrix :math:`\mathbf{A}` is a single :math:`G \times G` dense block —
  so the eigenpair is taken **directly**, with no iteration, by
  :func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite`.

The remainder of this section describes the direct dense eigensolve.


.. _direct-eigensolve-assembly:

Assembling the loss matrix from the transport operators
-------------------------------------------------------

The defining design decision of campaign **#276** is that the
infinite-medium loss matrix is **not** a bespoke energy matrix — it is
the meshed SN solver's own loss operator
:math:`\mathbf{A} = C - K_\mathrm{iso}` evaluated on the degenerate
phase space :math:`V_E \otimes V_{\rm pt}`, with the cross sections
read off a *meshless* single-cell carrier.  This is Cardinal Rule 2
(cross-model single source) applied to the simplest model in the
curriculum: there is exactly one place in ORPHEUS where the isotropic
in-scatter source :math:`\Sigma_{s0}^T\phi + 2\Sigma_2^T\phi` is
assembled, and the homogeneous solver reuses it rather than
re-implementing the same algebra.

The construction proceeds in five steps inside
:func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite`, and the
first two are deliberately separate — **the problem poses its own
space; the carrier only supplies data**:

1. **Pose the space.**
   :func:`~orpheus.homogeneous.solver._pose_space` mints
   :math:`V_E \otimes V_{\rm pt}` from the **mixture** — the energy axis
   through the one energy-arm rule
   (:meth:`EnergyAxis.from_materials
   <orpheus.numerics.axis.EnergyAxis.from_materials>`, the same rule
   :attr:`MaterialMesh.bulk_space
   <orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>`
   routes through, so the two spellings cannot diverge) tensored with
   the explicit **quotient point**, a one-element spatial axis carrying
   the COUNTING weight. That weight *is* the normalized "per unit
   volume" density convention of clause 1
   (:ref:`spaces-quotient-family`), and it is what the post-processing
   reaction-rate pairings consume
   (:ref:`homogeneous-rates-and-normalisation`).

   ⚠ This is the campaign-1 CS4a **K2** correction, and it is a
   separation of concerns rather than a change of value: the degenerate
   carrier's :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space`
   mints an ``==`` space (the identity-bridge gate pins it) but is no
   longer what production consumes. Read "the carrier supplies cross
   sections, the problem poses its space" as the rule; a page that says
   the space comes off the carrier is describing the pre-K2 tree.

2. **Supply the cross sections.**  A single-cell, single-region
   :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` is built
   from the mixture via
   :meth:`~orpheus.transport.mesh.material_mesh.MaterialMesh.from_materials`,
   and its
   :meth:`~orpheus.transport.mesh.material_mesh.MaterialMesh.material_xs_field`
   exposes the per-cell macroscopic cross sections
   (:math:`\Sigma_t`, :math:`\chi`, :math:`\nu\Sigma_f`, and the
   per-material transfer matrices) as the
   :class:`~orpheus.transport.mesh.material_xs_field.MaterialXSField`
   every transport operator consumes.  This carrier is **mesh-less** —
   it has no spatial mesh and no boundary faces at all — and since
   CS4b S7 the tree says so with typed refusals rather than by
   accident: promoting it to an :math:`S_N` phase space raises a named
   :class:`ValueError` (there is no boundary trace to sweep), and
   asking it for :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.areas`
   raises naming *its* case rather than a 2-D mesh's.

3. **Collision diagonal** :math:`C = \mathrm{diag}(\Sigma_t)`, read from
   :attr:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.total_cross_section`.

4. **Isotropic energy transfer**
   :math:`K_\mathrm{iso} = \Sigma_{s0}^T + 2\Sigma_2^T`, the action of the
   two model-shared operators

   .. math::
      :label: fission-source

      K_\mathrm{iso} \;=\;
      \underbrace{\Sigma_{s0}^{T}}_{\text{\scriptsize :class:`IsotropicScattering`}}
      \;+\;
      \underbrace{2\,\Sigma_2^{T}}_{\text{\scriptsize :class:`IsotropicN2N`}}

   where :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
   realises :math:`\Sigma_{s0}^T` (the in-scatter source matrix, the stored
   ``[g_from, g_to]`` transfer transposed) and
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`
   realises :math:`2\Sigma_2^T` (the loss-side multiplicity-2 transfer).
   The composed loss operator :math:`\mathbf A = C - K_\mathrm{iso}` is
   returned **un-materialized** (an
   :class:`~orpheus.numerics.operator.OperatorSum`) by the private
   ``_assemble_loss_operator`` helper — the consumer chooses the
   realization (taxonomy step 5b). Its dense :math:`(n_g, n_g)` form is
   **not** assembled term-by-term from per-material blocks; it is produced
   one layer later, by the operator's own
   :meth:`~orpheus.numerics.operator.LinearOperator.as_matrix` apply-to-basis
   (:ref:`matrix-inverse-operator`) on the meshless single cell — the
   ``(ng, 1)`` basis shape **derived from the operators' threaded domain**
   (the mixture-minted pose of step 1; campaign 1 CS1 gave the meshless
   operators a real space to derive it from, and CS4a K2 made that space
   the caller's rather than the carrier's — before CS1, every consumer
   passed ``basis_shape=(ng, 1)`` by hand) — **inside the**
   :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
   **constructor** (one eager materialization + LU factorization; see
   :ref:`direct-eigensolve-solve`). (The operators'
   :meth:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering.dense_per_material`
   accessor — the transpose read straight off the stored cross sections — is
   a storage-side *oracle* used by the verification gates as a
   structurally-independent cross-check, **not** a production assembly path.)

5. **Drop streaming.**  In an infinite medium the streaming operator
   :math:`L` is identically zero (:math:`\nabla\psi = 0`), so it is
   omitted from the sum.  What remains,
   :math:`\mathbf{A} = C - K_\mathrm{iso} = \mathrm{diag}(\Sigma_t)
   - \Sigma_{s0}^T - 2\Sigma_2^T`, is exactly the removal matrix
   :eq:`removal-matrix`.

.. note::

   The label :eq:`fission-source` historically named the per-iteration
   fission source of the retired power iteration,
   :math:`\mathbf{Q}_f = (\boldsymbol{\chi}/k)\,\nu\Sigma_f\cdot\boldsymbol{\phi}`.
   Under the direct method there is no iterate :math:`k^{(n)}` and no
   reassembled source; the production source is the single application of
   the dyad :math:`\mathbf{F}\boldsymbol{\phi}` (see
   :ref:`direct-eigensolve-solve`).  The label is retained on the
   isotropic-transfer assembly :math:`K_\mathrm{iso}` — the energy
   redistribution that *was* the in-scatter half of the old source — so
   the verification edge from the homogeneous test suite continues to pin
   the operator algebra that produces the source.


.. _direct-eigensolve-solve:

The fission dyad and the dense eigensolve
-----------------------------------------

The production matrix is the rank-1 dyad
:math:`\mathbf{F} = \boldsymbol{\chi} \otimes \nu\Sigma_f`
:eq:`fission-matrix`, assembled by

.. math::
   :label: fixed-source-solve

   \mathbf{M} \;=\; \mathbf{A}^{-1}\mathbf{F}
   \;=\; \mathbf{A}^{-1}\,\bigl(\boldsymbol{\chi}\otimes\nu\Sigma_f\bigr)

i.e. the loss matrix is **solved out** of the production once (rather than
inverted explicitly), giving the :math:`G \times G` eigenvalue matrix
:math:`\mathbf{M}`.  The eigenpair follows directly:

.. math::
   :label: keff-update

   \kinf \;=\; \lambda_{\max}(\mathbf{M}),
   \qquad
   \boldsymbol{\phi} \;=\; \text{the dominant right eigenvector of }\mathbf{M},


.. implements:: keff-update
   :by: orpheus.homogeneous.solver.solve_homogeneous_infinite

   **Implemented by** 4 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: keff-update
   :by: orpheus.numerics.eigenvalue.direct_eigenvalue

.. implements:: keff-update
   :by: orpheus.numerics.eigenvalue.dominant_eigenpair

.. implements:: keff-update
   :by: orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous

selected as the eigenpair with the largest real eigenvalue.  By the
Perron–Frobenius theorem :cite:`Hebert2009` this dominant eigenvector is the
unique non-negative solution — the **fundamental mode** — so the spectrum
is sign-normalised to non-negative components.

Both steps are spelled in the **operator algebra** rather than posed as a
dense ``(A, F)`` pair.  The multiplication operator
:math:`\mathbf{K} = \mathbf{A}^{-1}\mathbf{F}` :eq:`fixed-source-solve` is
constructed, and its dominant eigenpair taken, in four lines:

.. code-block:: python

   space = _pose_space(mix)                        # Energy ⊗ point, minted from the mixture
   loss = _assemble_loss_operator(mat_xs, space)   # A = C − K_iso, un-materialized
   production = IsotropicFission.from_material_xs(  # F = χ ⊗ νΣ_f
       mat_xs, space=space,
   )
   K = MatrixInverseOperator(loss) @ production
   k_inf, phi = dominant_eigenpair(K.as_matrix())

(Since CS4a K2 the operators pose on the MIXTURE-MINTED Energy ⊗ point
space — the problem's own physics names its space, :doc:`spaces`; the
degenerate carrier's ``bulk_space`` mints an ``==`` space but is a
reference, no longer the production source. ``MatrixInverseOperator``
and ``as_matrix`` **derive** the ``(ng, 1)`` basis shape from the
threaded domain; the pre-CS1 idiom passed ``basis_shape=(ng, 1)``
explicitly at both sites because the meshless operators carried no
space to derive it from.)

.. note::

   **Which fission binding the production factor is (CS4c step 4,
   2026-08-30).**  The line above named
   :class:`~orpheus.transport.operators.fission.FissionOperator` until
   step 4 rebound the channel as **two bindings of one datum**
   (:ref:`fission-as-dyad`): the *energy* binding
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
   — the rank-1 dyad on the scalar flux, and the one this solver, the
   S\ :sub:`N` k-outer and the 1-D diffusion solver all consume — and
   the *angular* binding ``FissionOperator``, the frame's
   :math:`\ell = 0` conjugation of the same dyad on a posed angular
   composite, which only S\ :sub:`N`'s eigen-:math:`M` posing needs.
   The infinite-medium problem has no angular axis, so the energy
   binding is not merely sufficient here — it is the honest one, and
   the angular binding now **refuses** a scalar carrier at construction
   rather than silently accepting it.  The arithmetic is unchanged: the
   dyad, its ``outer`` reduction order, and therefore :math:`k_\infty`
   are the same object under both names.

:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
materializes and LU-factors the loss operator **once** at construction; the
``@`` composes it with the fission dyad into the multiplication operator
:math:`\mathbf{K}`.  Its
:meth:`~orpheus.numerics.operator.LinearOperator.as_matrix` then walks the
:math:`G` basis columns — each column is one dyad apply
:math:`\mathbf{F}\mathbf{e}_j` followed by one LU backsolve
:math:`\mathbf{A}^{-1}(\cdot)` against the held factors — producing the dense
:math:`G \times G` resolvent :math:`[\mathbf{K}] = \mathbf{A}^{-1}\mathbf{F}`
(the loss matrix is still **solved out** of the production, never inverted
into an explicit :math:`[\mathbf{A}^{-1}]`).

The dominant eigenpair of that materialized resolvent is then taken by
:func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`, the **shared
Perron–Frobenius extraction primitive**: it runs :func:`numpy.linalg.eig`,
selects the largest-real eigenpair, sign-normalises :math:`\boldsymbol{\phi}`
so its components sum to a non-negative value, and **rejects a complex
dominant eigenvalue** (:class:`ValueError`) — the resolvent
:math:`\mathbf{A}^{-1}\mathbf{F}` of a well-posed criticality problem has a
real, positive dominant by Perron–Frobenius, so a complex one signals a
malformed :math:`(\mathbf{A}, \mathbf{F})` and is failed loud rather than
silently truncated (Cardinal Rule 1).  This validation has **one home**
(:func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`); every direct
spelling delegates to it.

**Explicit direct realization — the strategy choice as a type.**  The
homogeneous solver is the **first production consumer** of
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
(taxonomy step 5b).  It constructs the matrix inverse *explicitly* rather
than calling the structure-keyed ``loss.inverse()`` — which, reading the
operand tree :math:`C - K_\mathrm{iso}` (a sum with an invertible leading
collision diagonal :math:`C`), would return the **iterative**
:class:`~orpheus.numerics.green_operator.GreenOperator` preconditioned
splitting.  For a 0-D loss operator that is a single small dense block the
iterative splitting is the wrong realization and the exact dense inverse is
right; encoding that decision as the *type* ``MatrixInverseOperator`` — not a
``strategy=`` flag on ``.inverse()`` — is the taxonomy §3 strategy-override
seam realized honestly (the type **is** the choice).

The ``(A, F)``-posed convenience engine
:func:`~orpheus.numerics.eigenvalue.direct_eigenvalue` is the **sibling
spelling** of this same exact-dense extraction: it forms the resolvent from a
dense ``(A, F)`` pair via :func:`numpy.linalg.solve` and then delegates to the
identical :func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`.  Both routes
terminate in the one extraction home; they differ only in how the resolvent is
*posed* — the homogeneous path builds it through the operator algebra and so
**no longer calls** ``direct_eigenvalue``, which now has **zero production
consumers**, retained as the ``(A, F)``-posed engine of the three-engine
family (:ref:`three-eigenvalue-engines`) and the Rayleigh-quotient test oracle.

.. note::

   **Principled-equivalence re-baseline (step 5b).**  Re-spelling the
   resolvent through the operator algebra changed the LAPACK call sequence:
   the previous :func:`numpy.linalg.solve` formed
   :math:`\mathbf{A}^{-1}\mathbf{F}` in one batched ``gesv``; the operator
   path holds a single :func:`scipy.linalg.lu_factor` of :math:`\mathbf{A}`
   and issues one ``lu_solve`` backsolve per basis column.  Because
   floating-point addition is not associative, the two sequences may differ
   at the ULP level, so the cross-engine regression contract widened from
   byte-identity to ``rtol=1e-12`` (:math:`\kappa(\mathbf{A})\cdot`\ ULP
   portable across BLAS builds; measured bit-identical on the reference host,
   numpy 2.4 / scipy 1.17 sharing one LAPACK).  This satisfies all three
   re-baseline criteria (:ref:`operator-algebra`): the resolvent is formed
   from **named** operators (``MatrixInverseOperator`` is
   :math:`\mathbf{A}^{-1}`, the fission dyad is :math:`\mathbf{F}`); the value
   is anchored on a **structurally-independent** reference — the closed-form
   SymPy :math:`\kinf` of ``test_kinf_exact`` (1e-12), into which
   ``dominant_eigenpair`` is never wired; and the admissible drift
   (:math:`\sim\kappa(\mathbf{A})\cdot`\ ULP) is orders of magnitude below any
   rewire bug (a factor swap or dropped term moves :math:`k` by
   :math:`O(10^{-3})` or more).

.. note::

   The labels :eq:`fixed-source-solve` and :eq:`keff-update` historically
   named the per-iteration fixed-source solve
   (:math:`\mathbf{A}\boldsymbol{\phi}^{(n)} = \mathbf{Q}_f^{(n)}`) and
   the production/absorption eigenvalue ratio of the retired power
   iteration.  They are retained on the **direct** analogues: the
   loss-matrix solve :math:`\mathbf{M} = \mathbf{A}^{-1}\mathbf{F}` (the
   single dense solve that replaces the per-iteration sequence) and the
   eigenvalue extraction :math:`\kinf = \lambda_{\max}(\mathbf{M})` (the
   converged limit the iteration ratio approached).  The classical
   production/absorption form
   :math:`k = (\nu\Sigma_f\cdot\phi)/(\Sigma_a\cdot\phi)` remains a valid
   one-group balance identity — it is the per-group balance the
   :class:`~data.macro_xs.mixture.Mixture.absorption_xs` property reports
   alongside :math:`\kinf` — but it is no longer the computational path.

.. note::

   Because :math:`\mathbf{A}` is a single small dense block, the whole
   solve is one :func:`scipy.linalg.lu_factor` of :math:`\mathbf{A}`,
   :math:`G` ``lu_solve`` backsolves to form the resolvent
   :math:`[\mathbf{K}]`, and one :func:`numpy.linalg.eig`.  There is **no
   inner iteration and no outer iteration** — the homogeneous solver is the
   one deterministic solver in ORPHEUS with no iteration at all.  This is what
   makes it the instantaneous reference eigenvalue for every other solver on a
   homogeneous problem.


.. _spectral-invisibility:

Spectral invisibility: what the eigenvalue gate cannot see
----------------------------------------------------------

Two natural mistakes in the operator spelling — swapping the factor **order**
of the resolvent (:math:`\mathbf{F}\mathbf{A}^{-1}` instead of
:math:`\mathbf{A}^{-1}\mathbf{F}`), and **transposing** the materialized
resolvent — are *spectrally invisible*: they move :math:`\kinf` by **exactly
zero**.  Every :math:`k`-level gate (the cross-engine equivalence, the
closed-form SymPy anchor) is therefore structurally **blind** to them.  The
reason is a pair of standard linear-algebra identities, and understanding them
dictates *which* gate must catch the bug.

**Factor-order swap is a similarity transform.**  For any invertible
:math:`\mathbf{A}`,

.. math::
   :label: resolvent-similarity

   \mathbf{A}\,\bigl(\mathbf{A}^{-1}\mathbf{F}\bigr)\,\mathbf{A}^{-1}
   \;=\; \mathbf{F}\mathbf{A}^{-1},

.. vv-status: resolvent-similarity documented
.. Structural linear-algebra identity (similarity of the swapped resolvent),
.. NOT a solver claim; explains why the factor-order/transpose mutations are
.. spectrally invisible. Verifiable content is the object-level matrix gate
.. ``test_K_operator_as_matrix_is_the_resolvent`` (rtol=1e-12) named below.

so :math:`\mathbf{F}\mathbf{A}^{-1} = \mathbf{A}\,\mathbf{M}\,\mathbf{A}^{-1}`
is **similar** to :math:`\mathbf{M} = \mathbf{A}^{-1}\mathbf{F}` with
similarity matrix :math:`\mathbf{A}`.  Similar matrices share their entire
spectrum, so :math:`\lambda_{\max}(\mathbf{F}\mathbf{A}^{-1}) =
\lambda_{\max}(\mathbf{A}^{-1}\mathbf{F}) = \kinf` **identically**.  The
eigenvector is *not* invariant — if
:math:`\mathbf{M}\boldsymbol{\phi} = \kinf\boldsymbol{\phi}` then
:math:`(\mathbf{F}\mathbf{A}^{-1})(\mathbf{A}\boldsymbol{\phi}) =
\kinf(\mathbf{A}\boldsymbol{\phi})`, i.e. the mode maps
:math:`\boldsymbol{\phi} \mapsto \mathbf{A}\boldsymbol{\phi}` — but the
*eigenvalue* is untouched.

**Transpose preserves the spectrum.**  A matrix and its transpose share a
characteristic polynomial,
:math:`\det(\mathbf{M}^{\mathsf T} - \lambda\mathbf{I}) =
\det\bigl((\mathbf{M} - \lambda\mathbf{I})^{\mathsf T}\bigr) =
\det(\mathbf{M} - \lambda\mathbf{I})`, so
:math:`\lambda_{\max}(\mathbf{M}^{\mathsf T}) = \lambda_{\max}(\mathbf{M}) =
\kinf` (the eigenvectors of :math:`\mathbf{M}^{\mathsf T}` being the *left*
eigenvectors of :math:`\mathbf{M}`).

Both mutations were verified to give :math:`|\Delta\kinf| = 0.0` exactly.  A
value gate that reads only the eigenvalue — or even the sign-normalised
eigenvector, whose *shape* is fixed only up to the
:math:`\boldsymbol{\phi} \mapsto \mathbf{A}\boldsymbol{\phi}` remapping above —
cannot see either bug.  The committed catcher is therefore an **object-level**
gate, not a spectral one: ``test_K_operator_as_matrix_is_the_resolvent``
asserts the *materialized matrix itself* equals the reference resolvent,

.. math::
   :label: resolvent-object-gate

   [\mathbf{K}] \;=\; \texttt{np.linalg.solve}(\mathbf{A},\,\mathbf{F})
   \qquad (\text{rtol} = 10^{-12}),


.. implements:: resolvent-object-gate
   :by: orpheus.numerics.eigenvalue.direct_eigenvalue

   **Implemented by** 4 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: resolvent-object-gate
   :by: orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator

.. implements:: resolvent-object-gate
   :by: orpheus.numerics.operator.LinearOperator.as_matrix

.. implements:: resolvent-object-gate
   :by: orpheus.numerics.operator.OperatorProduct

and both mutations move :math:`[\mathbf{K}]` by :math:`O(1)`
(:math:`\mathbf{F}\mathbf{A}^{-1} \neq \mathbf{A}^{-1}\mathbf{F}` unless
:math:`\mathbf{A}` and :math:`\mathbf{F}` commute;
:math:`\mathbf{M}^{\mathsf T} \neq \mathbf{M}` unless :math:`\mathbf{M}` is
symmetric).  The general lesson is **pin the object, not just its spectrum**: a
value gate can be blind to an entire mutation class for structural (here
spectral-similarity) reasons, so a resolvent-forming operator earns an
intrinsic gate on the matrix it produces, above and beyond the eigenvalue it
feeds.


.. _three-eigenvalue-engines:

Why a direct engine: the three eigenvalue realisations
------------------------------------------------------

The dominant eigenpair of :math:`\mathbf{A}^{-1}\mathbf{F}` is what *every*
deterministic ORPHEUS solver ultimately wants; what differs is the
**realisation**.  :mod:`orpheus.numerics.eigenvalue` ships three siblings of
the same generalised eigenproblem
:math:`\mathbf{A}\boldsymbol{\phi} = \tfrac{1}{k}\mathbf{F}\boldsymbol{\phi}`:

.. list-table:: The three eigenvalue engines (:mod:`orpheus.numerics.eigenvalue`)
   :header-rows: 1
   :widths: 26 22 52

   * - Engine
     - Convergence
     - When it is the right realisation
   * - :func:`~orpheus.numerics.eigenvalue.power_iteration`
     - iterative, **linear** (rate :math:`|k_1/k_0|`)
     - Large, **sweep-only** loss operators that are never densely formed
       (SN, CP, MoC, diffusion).  These only *apply* :math:`\mathbf{A}^{-1}`
       — a :term:`sweep` or Krylov inner solve — and drive :math:`k` up the dominant
       mode through the
       :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` Protocol, which
       sees only a normalised-source fixed point, never a dense matrix.
   * - :func:`~orpheus.numerics.eigenvalue.direct_eigenvalue`
     - **exact** (one LAPACK shot)
     - **Small, densifiable** operators posed as a dense
       :math:`(\mathbf{A}, \mathbf{F})` pair — few-group / few-region
       problems.  Forms the dense resolvent
       :math:`\mathbf{A}^{-1}\mathbf{F}` via :func:`numpy.linalg.solve` and
       delegates the extraction to
       :func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`.  The direct
       (non-iterative) sibling of ``power_iteration``; the 0-D homogeneous
       medium reaches the *same* exact extraction through the operator-algebra
       spelling (:ref:`direct-eigensolve-solve`) rather than this ``(A, F)``
       entry point.
   * - :func:`~orpheus.numerics.eigenvalue.rayleigh_quotient_iteration`
     - iterative, **superlinear** (locally quadratic)
     - Polishing an eigenpair *estimate* to the eigenpair NEAREST its
       Rayleigh quotient — **not** necessarily the dominant one (warm-start
       near the mode you want).  The bordered / augmented-Newton form, in
       which the previous iterate enters as the normalisation **row**.  Not
       yet wired into a meshed solver — that integration, and its use as the
       adjoint-:math:`\phi^*` vehicle, is
       `#277 <https://github.com/deOliveira-R/ORPHEUS/issues/277>`_.

The infinite-medium :math:`\kinf` takes the **exact dense** route — the
operator-algebra spelling ``MatrixInverseOperator(loss) @ production``
extracted by :func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`
(:ref:`direct-eigensolve-solve`) — because the 0-D loss matrix
:math:`\mathbf{A}` is a single :math:`G \times G` block, so the spectrum of
:math:`\mathbf{A}^{-1}\mathbf{F}` is **exactly solvable**: an iterative engine
would only approximate, at a convergence tolerance, an answer the dense solve
gives to machine precision in one shot.  That exactness matters here
specifically: the homogeneous solver is verified against a :math:`10^{-12}`
**closed-form** analytical eigenvalue (the ``homo_1eg`` / ``homo_2eg`` /
``homo_4eg`` benchmarks below), and the cost an iterative engine pays to reach
:math:`10^{-12}` is coupled to the dominance ratio :math:`|k_1/k_0|`, a
dependence the direct dense inverse does not have.

.. note::

   **The rank-1 subtlety.**  For the *pure* fission dyad
   :math:`\mathbf{F} = \boldsymbol{\chi}\otimes\nu\Sigma_f` the resolvent
   :math:`\mathbf{A}^{-1}\mathbf{F} = (\mathbf{A}^{-1}\boldsymbol{\chi})\,
   (\nu\Sigma_f)^{\mathsf T}` is **rank-1**: it has a single nonzero
   eigenvalue and :math:`G-1` exact zeros, so the dominance ratio is
   :math:`0` and :func:`~orpheus.numerics.eigenvalue.power_iteration` would
   in fact converge in *one* step on this problem.  That one-step property
   is a fragile consequence of :math:`\mathbf{F}`'s rank-1 structure, not a
   general guarantee — it does not survive a future multi-spectrum
   production term.  The **exact dense inverse** is exact for **any**
   :math:`\mathbf{F}`, so it is the robust as well as the exact choice.

**The pure-math verification of the engines.**  All three engines — and the
shared :func:`~orpheus.numerics.eigenvalue.dominant_eigenpair` extraction they
delegate to — are verified against a **transport-unrelated, hand-derived
closed-form eigenproblem** — :math:`\mathbf{M} =
V\operatorname{diag}(\lambda)V^{-1}` with chosen eigenpairs, and the rank-1
closed form :math:`k = v^{\mathsf T} A^{-1} u` — in the pure-math gate
``tests/numerics/test_eigenvalue.py`` (the closed-form eigenproblem, the direct
``dominant_eigenpair`` surface with its one-home relocation proofs, and the RQI
gates).  This is a **closed-form** reference: V&V pillar 1, the *only* pillar
that proves an eigenvalue (MMS is source-driven and cannot).  No reference
value is produced by calling the same :func:`numpy.linalg.eig` the engine uses,
so the cross-check is structurally independent by construction.

Once an engine is pinned against domain-independent ground truth it is
**trusted machinery**, and a production solver AND its verification oracle
may BOTH call it without contamination.  The structural independence does
NOT live in the eigensolver — it lives in how :math:`(\mathbf{A},
\mathbf{F})` are *assembled*.  The homogeneous solver builds
:math:`\mathbf{A} = C - K_\mathrm{iso}` through the transport operator
algebra (:ref:`direct-eigensolve-assembly`); an oracle may build the same
:math:`\mathbf{A} = \operatorname{diag}(\Sigma_t) - \Sigma_{s0}^{T} -
2\Sigma_2^{T}` by the fused route.  Two different structural paths to the
same matrix, cross-checked at the eigenvalue — the shared, pre-verified
extraction (:func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`) is not a
contamination because the independence was never asked of it.


.. _homogeneous-rates-and-normalisation:

Reaction rates, flux normalisation, and the one-group condensation
-------------------------------------------------------------------

The eigenvector :math:`\boldsymbol{\phi}` is determined only up to a
scalar multiple.  After the eigensolve,
:func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite` normalises
the flux so that the **fission** production rate is 100 n/cm\ :sup:`3`/s:

.. math::
   :label: normalisation

   \boldsymbol{\phi} \leftarrow \boldsymbol{\phi} \times
   \frac{100}{\nu\boldsymbol{\Sigma}_\mathrm{f} \cdot \boldsymbol{\phi}}


.. implements:: normalisation
   :by: orpheus.homogeneous.solver.solve_homogeneous_infinite

   **Implemented by** 2 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: normalisation
   :by: orpheus.homogeneous.solver.solve_homogeneous_infinite

The normalisation denominator is the **fission** production rate
:math:`\nu\Sigma_f\cdot\boldsymbol{\phi}` only — consistent with the
production matrix :math:`\mathbf{F} = \boldsymbol{\chi}\otimes\nu\Sigma_f`.
The :math:`(n,2n)` neutrons are **not** in this denominator: they are a
loss-side transfer folded into :math:`\mathbf{A}` as
:math:`2\Sigma_2^T`, not a production channel (see
:ref:`scattering-matrix-convention` and the note under the production
matrix :eq:`fission-matrix` above).

The rate is a typed integrated co-vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since campaign 1 CS4b (step S7, EE-1) that denominator — and the two
post-processing rates beside it — are not a hand-written contraction.
All three evaluations go through
:class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`,
the volume-integrated reaction rate

.. math::

   R_x(\varphi)
   \;=\; \int_V \sum_g \Sigma_{x,g}(\vec r)\,\varphi_g(\vec r)\,\mathrm{d}V
   \;=\; \sum_{\text{cells}} V_{\rm cell}\,
          \langle \Sigma_x, \varphi\rangle(\text{cell}),

which is the :math:`\varphi^\dagger = 1` **degenerate** of the
adjoint-weighted homogenization bilinear
:math:`\langle \varphi^\dagger, M[\Sigma_x]\varphi\rangle` (theorem
T1 of the algebra of record,
:mod:`orpheus.derivations.common.homogenization`; the weighted form is
live — pass ``adjoint=``). On the infinite-medium pose the spatial sum
has one term and :math:`V_{\rm cell}` is the quotient point's unit
weight, so the object degenerates to the bare group contraction — but
it is the *same* object the meshed solvers use, which is the point:
one functional, one contraction, no 0-D special case.

.. important::

   **The solver RE-POSES the cross-section fields onto its own pose,
   and that is a correctness requirement, not tidiness.** The
   functional's measure authority is its cross section's space (the
   :math:`\sigma`\ ↔geometry pairing tier), while the total-flux leg
   below is the *pose's* pairing. Bind the rates to the
   carrier-minted space and the two legs would read **different**
   measures — so re-weighting the pose would move one and not the
   other, and the condensed cross sections would stop being ratios of
   commensurable quantities. The solver therefore rebinds
   (``replace(field, space=space)``) before wrapping, and ``replace``
   re-runs the field's own construction validation on the way.

   In production this is content-neutral: the pose content-equals the
   carrier's mint (the identity-bridge gate pins it), and `[M]` the
   rewiring is **bit-identical** with the pre-EE-1 raw
   ``space.inner_product`` spelling on
   :math:`\nu\Sigma_f` and :math:`\Sigma_a` across the 2-group and
   4-group fixtures (4 of 4 probed; the spelling-equivalence gate pins
   it, and the D5 byte-stability suite is the end-to-end witness).
   Neutral *today* is not the same claim as *correct under the
   mutation*: the CS4a-R G2.5 gate scales the pose's point weight by
   two and requires the rates to move with it, and that gate is what
   the rebinding exists to satisfy.

**The total flux is NOT a reaction rate.** The denominator of the
one-group condensation is

.. math::

   \langle 1, \varphi\rangle \;=\; \int_V \sum_g \varphi_g\,\mathrm{d}V ,

which carries no cross section: it is the pose's own **integration
co-vector**, so it stays
:meth:`FunctionSpace.inner_product
<orpheus.numerics.space.FunctionSpace.inner_product>` against a field
of ones rather than being dressed as an
:class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
with a unit cross section. `[M]` on the counting-weighted quotient
point it is bit-identical to ``float(phi.sum())``, which is what it
replaced.

The two condensed one-group cross sections
:attr:`~orpheus.homogeneous.solver.HomogeneousResult.sig_prod` and
:attr:`~orpheus.homogeneous.solver.HomogeneousResult.sig_abs` are then
**same-pairing ratios**

.. math::

   \bar\sigma_x \;=\;
   \frac{\langle \Sigma_x, \varphi\rangle}{\langle 1, \varphi\rangle},

and being ratios of two integrals against the *same* measure they are
**measure-invariant**: a future pose carrying a different point weight
moves numerator and denominator together and leaves
:math:`\bar\sigma_x` fixed. That is the CS4a-R **XD-6** ruling — a
quantity documented as a cross section must not scale with the point
weight — and its gate is the same ×2 pose mutation, asserting
*rates move, ratio stays*. The counting weight of step 1 is a
convention the posing function states; it is not a contract these
ratios depend on.

Post-processing reads three energy-grid diagnostics off the mixture's
:class:`~orpheus.data.energy_grid.EnergyGrid` value object (campaign #276
P4-F — the group geometry lives on the grid, not re-derived in the solver)
and stores them on
:class:`~orpheus.homogeneous.solver.HomogeneousResult`:

- **Representative energy** — the plot abscissa:
  :attr:`~orpheus.homogeneous.solver.HomogeneousResult.representative_energy`
  :math:`= \bar E_g = \sqrt{E_g^{\mathrm{up}}\,E_g^{\mathrm{lo}}}`, the
  **geometric** group centre.
- **Flux per unit energy**: :math:`\phi_g / \Delta E_g`
  (:attr:`~orpheus.homogeneous.solver.HomogeneousResult.flux_per_energy`),
  with :math:`\Delta E_g = E_g^{\mathrm{up}} - E_g^{\mathrm{lo}}`.
- **Flux per unit lethargy**: :math:`\phi_g / \Delta u_g`
  (:attr:`~orpheus.homogeneous.solver.HomogeneousResult.flux_per_lethargy`),
  with :math:`\Delta u_g = \ln\!\bigl(E_g^{\mathrm{up}} / E_g^{\mathrm{lo}}\bigr)`.

Here :math:`E_g^{\mathrm{up}} =` ``edges[g]`` and :math:`E_g^{\mathrm{lo}} =`
``edges[g+1]`` are the upper / lower bounds of group :math:`g` under the
**fast-first descending** convention (group :math:`0` is the highest-energy
group, boundaries strictly decreasing; see
:ref:`canonical-group-convention`).

.. note::

   **Why the geometric centre (the P4-F correction).**  The spectrum is
   plotted as flux-per-lethargy on a **logarithmic** energy abscissa
   (``semilogx``).  The natural centre of a group on a log axis is the
   **geometric** mean :math:`\sqrt{E^{\mathrm{up}}E^{\mathrm{lo}}}`, which
   sits at the midpoint of the group's *lethargy* interval — exactly where a
   flux-per-lethargy value belongs.  The arithmetic midpoint
   :math:`\tfrac{1}{2}(E^{\mathrm{up}} + E^{\mathrm{lo}})` is biased
   **high** by the AM–GM inequality
   (:math:`\tfrac{1}{2}(a+b) \ge \sqrt{ab}`, the gap widening as a group
   spans more decades), so it plots each point to the right of the lethargy
   centre.  Before P4-F the result carried the (wrong) arithmetic midpoint;
   P4-F renamed the field ``eg_mid`` → ``representative_energy`` and moved it
   to the geometric centre.  The change is **purely the abscissa** — the
   flux *values* are unchanged, only the energy each group is plotted *at*
   moved.  At a thermal floor (:math:`E_g^{\mathrm{lo}} = 0` eV) the
   geometric mean degenerates;
   :attr:`~orpheus.data.energy_grid.EnergyGrid.representative_energy` falls
   back to half the upper edge there (still strictly inside the group), while
   the lethargy width :math:`\Delta u_g \to +\infty` is genuinely unbounded.

For synthetic verification mixtures with no physical energy grid
(:attr:`Mixture.eg` is ``None``) all three diagnostics are ``None`` and the
:attr:`~orpheus.homogeneous.solver.HomogeneousResult.flux_per_energy` /
:attr:`~orpheus.homogeneous.solver.HomogeneousResult.flux_per_lethargy`
properties raise — :math:`\kinf` and the flux spectrum are still
well-defined, only the per-energy plotting path is unavailable.


.. _example-problems:

Example Problems
=================

Aqueous Uranium Solution Reactor
----------------------------------

The simplest physical problem: water with dissolved uranium-235
(1000 ppm) at room temperature (294 K) and atmospheric pressure.  This
models a bare, infinite, aqueous homogeneous reactor — a configuration
historically important for early criticality experiments.

The mixture contains only three isotopes: :sup:`1`\ H, :sup:`16`\ O,
and :sup:`235`\ U.  Water provides the moderation (hydrogen
down-scatter) and :sup:`235`\ U the fission source.  The water density
is obtained from the IAPWS-IF97 steam tables.

See :func:`~data.macro_xs.recipes.aqueous_uranium`.

.. plot::
   :caption: Neutron spectrum for an aqueous uranium solution reactor
             (:math:`k_\infty \approx 1.036`).  The thermal Maxwellian
             peak near 0.025 eV, the 1/E slowing-down region, and the
             fast fission peak above 1 MeV are clearly visible.

   import numpy as np
   import matplotlib.pyplot as plt
   import warnings
   warnings.filterwarnings('ignore')

   from orpheus.data.macro_xs.recipes import aqueous_uranium
   from orpheus.homogeneous import solve_homogeneous_infinite

   mix = aqueous_uranium(temp_K=294, pressure_MPa=0.1, u_conc_ppm=1000.0)
   result = solve_homogeneous_infinite(mix)

   fig, ax = plt.subplots()
   ax.semilogx(result.representative_energy, result.flux_per_lethargy, 'b-', linewidth=1.2)
   ax.set_xlabel('Energy (eV)')
   ax.set_ylabel(r'Flux per unit lethargy $\phi / \Delta u$')
   ax.set_title(
       rf'Aqueous U Solution — $k_\infty$ = {result.k_inf:.5f}'
   )
   ax.set_xlim(1e-3, 1e7)
   ax.grid(True, alpha=0.3)
   plt.tight_layout()


PWR-Like Homogenised Cell
---------------------------

A more realistic problem: a PWR unit cell (UO\ :sub:`2` fuel, Zircaloy
cladding, borated water) **volume-homogenised** into a single mixture.
This is not a physically realisable configuration, but it exercises the
full cross-section pipeline with 12 isotopes, self-shielding of
:sup:`238`\ U resonances, and boron absorption.

The geometric homogenisation uses volume fractions from the pin-cell
geometry:

.. math::
   :label: pin-cell-volume-fractions

   f_\mathrm{fuel} = \frac{r_\mathrm{fuel}^2}{r_\mathrm{cell}^2}, \quad
   f_\mathrm{clad} = \frac{r_\mathrm{clad,out}^2 - r_\mathrm{clad,in}^2}
                           {r_\mathrm{cell}^2}, \quad
   f_\mathrm{cool} = \frac{r_\mathrm{cell}^2 - r_\mathrm{clad,out}^2}
                           {r_\mathrm{cell}^2}

.. vv-status: pin-cell-volume-fractions documented
.. Definitional geometric formula: the Wigner-Seitz pin-cell volume fractions
.. consumed by data.macro_xs.recipes.pwr_like_mix. A textbook geometry
.. definition, not a solver claim.

where :math:`r_\mathrm{cell} = p / \sqrt{\pi}` is the Wigner–Seitz
equivalent radius for a square lattice of pitch :math:`p`.

The mixture includes: :sup:`235`\ U, :sup:`238`\ U, :sup:`16`\ O (fuel),
five Zr isotopes (:sup:`90,91,92,94,96`\ Zr), :sup:`1`\ H,
:sup:`16`\ O (coolant), :sup:`10`\ B, :sup:`11`\ B.

See :func:`~data.macro_xs.recipes.pwr_like_mix`.

.. plot::
   :caption: Neutron spectrum for the PWR-like homogenised mixture
             (:math:`k_\infty \approx 1.014`).  Compared to the
             aqueous solution, the thermal peak is suppressed by boron
             absorption and the :sup:`238`\ U resonance self-shielding
             is visible in the epithermal range.

   import numpy as np
   import matplotlib.pyplot as plt
   import warnings
   warnings.filterwarnings('ignore')

   from orpheus.data.macro_xs.recipes import pwr_like_mix
   from orpheus.homogeneous import solve_homogeneous_infinite

   mix = pwr_like_mix()
   result = solve_homogeneous_infinite(mix)

   fig, ax = plt.subplots()
   ax.semilogx(result.representative_energy, result.flux_per_lethargy, 'r-', linewidth=1.2)
   ax.set_xlabel('Energy (eV)')
   ax.set_ylabel(r'Flux per unit lethargy $\phi / \Delta u$')
   ax.set_title(
       rf'PWR-Like Homogenised Cell — $k_\infty$ = {result.k_inf:.5f}'
   )
   ax.set_xlim(1e-3, 1e7)
   ax.grid(True, alpha=0.3)
   plt.tight_layout()


Comparison
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - Aqueous U Solution
     - PWR-Like Mixture
   * - :math:`\kinf`
     - 1.03596
     - 1.01357
   * - Fuel
     - Dissolved :sup:`235`\ U (1000 ppm)
     - UO\ :sub:`2` (3% enrichment)
   * - Moderator
     - Light water (294 K)
     - Borated water (600 K, 4000 ppm B)
   * - Isotopes
     - 3
     - 12
   * - Self-shielding
     - Negligible (:sup:`235`\ U dilute)
     - Significant (:sup:`238`\ U resonances)
   * - MATLAB reference
     - 1.03596
     - 1.01357


.. _infinite-medium-verification-pins:

Verification — what pins this chapter
=====================================

The homogeneous solver's verification evidence — the SymPy-derived analytical
:math:`\kinf` eigenvalues
(:mod:`orpheus.derivations.continuous.analytical.homogeneous`), the multi-group
matrix-eigenvalue chain, and the two 421-group industrial cross-checks against
the legacy MATLAB implementation — lives in the verification part:
:doc:`/theory/verification/homogeneous`. The same
:class:`~orpheus.derivations.common.verification_case.VerificationCase` objects
serve both that chapter's LaTeX equations and the test suite. The
auto-generated :doc:`/theory/verification/matrix` reports per-equation test
coverage; :ref:`theory-verification` carries the part-wide principles and
harness contracts.


Comparison with Spatially-Dependent Solvers
============================================

The homogeneous infinite-medium solver sits at the simplest end of the
solver hierarchy.  The following table compares it with the
spatially-dependent solvers available in ORPHEUS:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Aspect
     - Homogeneous
     - Collision Probability
     - Discrete Ordinates
     - Diffusion
   * - Spatial dependence
     - None
     - Region-averaged
     - Mesh-resolved
     - Mesh-resolved
   * - Angular dependence
     - None (isotropic)
     - Integrated out
     - Discrete ordinates
     - Fick's law
   * - Transport operator
     - :math:`\mathbf{A}^{-1}` (direct)
     - :math:`P_\infty` matrix
     - Diamond-difference sweep
     - Implicit solve
   * - Inner iterations
     - None
     - None
     - Scattering source
     - None
   * - Typical convergence
     - Direct (no iteration)
     - 10--20 outer
     - 20--50 outer
     - 100+ outer
   * - Eigenvalue computed
     - :math:`\kinf`
     - :math:`\kinf` (lattice)
     - :math:`\kinf` (lattice)
     - :math:`\keff` (core)
   * - Implementation
     - :func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite`
     - :class:`CPSolver`
     - :class:`SNSolver`
     - :class:`DiffusionSolver`
