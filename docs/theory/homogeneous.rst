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

- Balance: :math:`\mathbf{A}\phi = \frac{1}{k}\mathbf{F}\phi` where :math:`\mathbf{A} = \text{diag}(\Sigma_t) - \Sigma_s^T`, :math:`\mathbf{F} = \chi \otimes (\nu\Sigma_f)`
- 1-group: :math:`k = \nu\Sigma_f / \Sigma_a` (exact, no iteration)
- Multi-group: :math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})` via ``numpy.eigvals``
- This is the reference eigenvalue for ALL solvers on homogeneous problems
- Tolerance: < 1e-12 (limited only by FP arithmetic on small dense matrices)
- **Gotcha**: this eigenvalue is flux-shape independent — it tests nothing about spatial or angular discretization


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
  curriculum, introducing the multi-group eigenvalue problem and the
  power iteration algorithm.
- The **cross-section preparation pipeline** — isotope loading,
  sigma-zero self-shielding, interpolation, macroscopic summation — is
  exercised here and reused unchanged by every subsequent solver (SN,
  MoC, CP, Monte Carlo, diffusion).
- Analytical eigenvalues for 1-, 2-, and 4-group homogeneous media
  serve as **verification benchmarks** for all deterministic solvers.

This chapter derives the infinite-medium eigenvalue problem from first
principles, describes the cross-section preparation pipeline, and
presents the power iteration algorithm used to compute :math:`\kinf`
and :math:`\phi(E)`.

The solver is implemented in :class:`HomogeneousSolver`, which satisfies
the :class:`~numerics.eigenvalue.EigenvalueSolver` protocol.  The
convenience wrapper :func:`solve_homogeneous_infinite` runs the full
calculation and returns a :class:`HomogeneousResult`.


From the Boltzmann Equation to the Infinite Medium
====================================================

The Boltzmann Transport Equation
---------------------------------

The starting point is the steady-state neutron transport equation in its
integro-differential form [Duderstadt1976]_:

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

Here :math:`\psi(\mathbf{r}, \hat{\Omega}, E)` is the angular flux,
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
Group 1 carries the highest energies (fast neutrons), group :math:`G`
the lowest (thermal neutrons).  The energy boundaries
:math:`E_0 > E_1 > \cdots > E_G` define the grid stored in
:attr:`Mixture.eg`.

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

   \mathbf{F} = \boldsymbol{\chi} \otimes
                \bigl(\nu\boldsymbol{\Sigma}_\mathrm{f}
                      + 2 \, \text{colsum}(\boldsymbol{\Sigma}_2)\bigr)

Here :math:`\boldsymbol{\Sigma}_{\mathrm{s}}` is the :math:`G \times G`
scattering transfer matrix (:math:`P_0` component) and
:math:`\boldsymbol{\Sigma}_2` is the :math:`(n,2n)` transfer matrix.

.. note::

   The :math:`(n,2n)` reaction appears in **both** matrices.  In the
   removal matrix, :math:`-2\boldsymbol{\Sigma}_2^T` removes the
   incident neutron and accounts for the two emitted neutrons entering
   the scattering system.  In the fission matrix, the column-sum of
   :math:`\boldsymbol{\Sigma}_2` adds the net production of one extra
   neutron per :math:`(n,2n)` event.

   See :class:`HomogeneousSolver.__init__` for the construction of
   :math:`\mathbf{A}` and :meth:`HomogeneousSolver.compute_fission_source`
   for the production term.

The eigenvalue :math:`k = \kinf` is the largest eigenvalue of the
generalised problem :eq:`matrix-eigenvalue`.  By the Perron–Frobenius
theorem [Hebert2009]_, the dominant eigenvector :math:`\boldsymbol{\phi}`
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
**lower-triangular**: non-zero entries only below (or on) the diagonal,
because :math:`\Sigs{g' \to g} = 0` when :math:`g < g'` (no neutrons
scatter from thermal to fast).

The neutron balance :eq:`mg-balance` requires the **in-scattering** into
group :math:`g` from all groups :math:`g'`:

.. math::

   \sum_{g'} \Sigs{g' \to g} \phi_{g'}
   = \bigl(\boldsymbol{\Sigma}_{\mathrm{s}}^T \cdot \boldsymbol{\phi}\bigr)_g

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
   :by: orpheus.derivations.homogeneous.derive_1g

   Verified analytically (exact closed-form ratio) against the
   ``homo_1eg`` :class:`~orpheus.derivations._types.VerificationCase`.

This is the most fundamental result in reactor physics.  It states that
:math:`\kinf` is the ratio of neutron production to neutron absorption,
which is the definition of the infinite multiplication factor [Stacey2007]_.

For the connection to the **four-factor formula**: in a single-material
homogeneous medium the thermal utilisation :math:`f = 1`, the resonance
escape probability :math:`p = 1` (no spatial heterogeneity), and the
fast fission factor :math:`\varepsilon = 1`, so :math:`\kinf = \eta \cdot f
\cdot p \cdot \varepsilon = \eta`.

**Numerical example** (from :func:`derivations.homogeneous.derive_1g`):
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

.. math::
   :label: two-group-F

   \mathbf{F} = \begin{pmatrix}
     \nu_1 \Sigf{1} & \nu_2 \Sigf{2} \\
     0 & 0
   \end{pmatrix}

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
     + 0
     & \dfrac{\Sigs{1 \to 2}\, \nu_2 \Sigf{2}}
             {\Sigma_{\mathrm{r},1}\,\Sigma_{\mathrm{r},2}}
       + \dfrac{\nu_2 \Sigf{2}}{\Sigma_{\mathrm{r},2}}
   \end{pmatrix}

The characteristic equation :math:`\det(\mathbf{M} - \lambda\mathbf{I}) = 0`
gives a quadratic in :math:`\lambda`:

.. math::
   :label: two-group-charpoly

   \lambda^2 - \bigl(M_{11} + M_{22}\bigr)\lambda
   + \bigl(M_{11}M_{22} - M_{12}M_{21}\bigr) = 0

whose roots are:

.. math::
   :label: two-group-roots

   \lambda_{\pm} = \frac{(M_{11} + M_{22})
                   \pm \sqrt{(M_{11} - M_{22})^2 + 4 M_{12} M_{21}}}{2}

The dominant root :math:`\lambda_+` is :math:`\kinf`.

**Worked numerical example** (from :func:`derivations.homogeneous.derive_2g`):

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
   M_{21} &= \frac{0.10 \times 2.50 \times 0.01}{0.12 \times 0.10} = 2.083\overline{3} \\[4pt]
   M_{22} &= \frac{0.10 \times 2.50 \times 0.08}{0.12 \times 0.10}
            + \frac{2.50 \times 0.08}{0.10} = 1.6\overline{6} + 2.0 = 3.6\overline{6}

Substituting into the quadratic formula :eq:`two-group-roots`:

.. math::

   \kinf = \lambda_+ = 1.8750000000

The second eigenvalue :math:`\lambda_- = 2.0833\overline{3}` is also
positive but smaller.  The **dominance ratio** is
:math:`|\lambda_-/\lambda_+| = 1.11`, which for this particular set of
cross sections exceeds unity — indicating that the spectrum is
initially far from the fundamental mode but converges rapidly because
the power iteration eigenvalue still converges monotonically.

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

**Result** (from :func:`derivations.homogeneous.derive_4g`):

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
background cross section is defined as [Bondarenko1964]_:

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
   equivalence theory of Bondarenko [Bondarenko1964]_.


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

The **absorption cross section** used in the eigenvalue update
:eq:`keff-update` is not stored directly but computed as a derived
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

   \frac{\phi}{du} = \frac{\phi(E) \cdot E}{\Delta u}
   \propto \frac{1}{E} \cdot E = \text{const}

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

The Power Iteration Algorithm
==============================

Algorithm
----------

The eigenvalue problem :eq:`matrix-eigenvalue` is solved by **power
iteration** [Hebert2009]_, converging to the dominant eigenvalue
:math:`\kinf` and the fundamental mode :math:`\boldsymbol{\phi}`.
The algorithm implements the
:class:`~numerics.eigenvalue.EigenvalueSolver` protocol:

1. **Initialise**: :math:`\boldsymbol{\phi}^{(0)} = [1, 1, \ldots, 1]^T`,
   :math:`k^{(0)} = 1.0`.

2. **Fission source** (:meth:`HomogeneousSolver.compute_fission_source`):

   .. math::
      :label: fission-source

      \mathbf{Q}_f^{(n)} = \frac{\boldsymbol{\chi}}{k^{(n-1)}}
        \bigl(\boldsymbol{\Sigma}_\mathrm{p}
              + 2 \, \text{colsum}(\boldsymbol{\Sigma}_2)\bigr)
        \cdot \boldsymbol{\phi}^{(n-1)}

3. **Fixed-source solve** (:meth:`HomogeneousSolver.solve_fixed_source`):

   .. math::
      :label: fixed-source-solve

      \mathbf{A} \, \boldsymbol{\phi}^{(n)} = \mathbf{Q}_f^{(n)}

   This is a single sparse direct solve via
   :func:`scipy.sparse.linalg.spsolve`.  The removal matrix
   :math:`\mathbf{A}` is **constant** across iterations (pre-built in
   :meth:`HomogeneousSolver.__init__`).

4. **Eigenvalue update** (:meth:`HomogeneousSolver.compute_keff`):

   .. math::
      :label: keff-update

      k^{(n)} = \frac{\text{production}}{\text{absorption}}
      = \frac{\bigl(\boldsymbol{\Sigma}_\mathrm{p}
              + 2 \, \text{colsum}(\boldsymbol{\Sigma}_2)\bigr)
              \cdot \boldsymbol{\phi}^{(n)}}
             {\boldsymbol{\Sigma}_\mathrm{a}
              \cdot \boldsymbol{\phi}^{(n)}}

   There is no leakage term in an infinite medium.

5. **Convergence** (:meth:`HomogeneousSolver.converged`):
   stop when :math:`|k^{(n)} - k^{(n-1)}| < 10^{-10}` after at least
   3 iterations.

The generic loop is implemented in
:func:`~numerics.eigenvalue.power_iteration`.

.. note::

   Unlike spatially-dependent solvers (SN, CP, diffusion), the
   homogeneous solver has **no inner iteration**: the removal matrix
   :math:`\mathbf{A}` is constant and inverted directly.  The only
   iteration is the outer power iteration on :math:`k`.  This makes
   the homogeneous solver extremely fast — convergence in 3--5
   iterations is typical.


Convergence Properties
-----------------------

Power iteration converges to the dominant eigenvalue at a rate governed
by the **dominance ratio** :math:`\rho = |k_1 / k_0|`, where
:math:`k_0` and :math:`k_1` are the two largest eigenvalues of
:math:`\mathbf{A}^{-1}\mathbf{F}`.  After :math:`n` iterations, the
eigenvalue error decays as:

.. math::
   :label: convergence-rate

   |k^{(n)} - k_0| \sim \rho^n

.. vv-status: convergence-rate documented

For the 421-group industrial problems, the dominance ratio is very small
(the spectrum is dominated by a single fundamental mode), so
convergence is rapid.  The following table shows the iteration history
for the aqueous uranium case:

.. list-table::
   :header-rows: 1
   :widths: 15 25 25

   * - Iteration
     - :math:`k^{(n)}`
     - :math:`|k^{(n)} - k^{(n-1)}|`
   * - 1
     - 1.03596
     - ---
   * - 2
     - 1.03596
     - :math:`< 10^{-5}`
   * - 3
     - 1.03596
     - :math:`< 10^{-10}`

The Perron–Frobenius theorem guarantees that the converged eigenvector
is the unique non-negative solution.  This is critical for physical
interpretability: the neutron flux must be non-negative everywhere in
energy space, and the fundamental mode is the only eigenvector with
this property.

**Why homogeneous converges faster than spatially-dependent solvers:**

In spatially-dependent problems (SN, diffusion), the dominance ratio
approaches unity as the mesh is refined and the optical thickness
increases, requiring hundreds of outer iterations.  The homogeneous
problem has no spatial mesh — the only degrees of freedom are the
:math:`G` energy groups — so the eigenvalue spectrum of
:math:`\mathbf{A}^{-1}\mathbf{F}` is typically well-separated, giving
:math:`\rho \ll 1`.


Flux Normalisation
-------------------

The eigenvector :math:`\boldsymbol{\phi}` is determined only up to a
scalar multiple.  After convergence,
:func:`solve_homogeneous_infinite` normalises the flux so that the
total neutron production rate is 100 n/cm\ :sup:`3`/s:

.. math::
   :label: normalisation

   \boldsymbol{\phi} \leftarrow \boldsymbol{\phi} \times
   \frac{100}{\bigl(\boldsymbol{\Sigma}_\mathrm{p}
         + 2\,\text{colsum}(\boldsymbol{\Sigma}_2)\bigr)
         \cdot \boldsymbol{\phi}}

Post-processing computes two spectral representations stored in
:class:`HomogeneousResult`:

- **Flux per unit energy**: :math:`\phi_g / \Delta E_g`
  (:attr:`HomogeneousResult.flux_per_energy`)
- **Flux per unit lethargy**: :math:`\phi_g / \Delta u_g`
  (:attr:`HomogeneousResult.flux_per_lethargy`)

where :math:`\Delta E_g = E_{g-1} - E_g` and
:math:`\Delta u_g = \ln(E_{g-1} / E_g)`.


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

   from data.macro_xs.recipes import aqueous_uranium
   from homogeneous import solve_homogeneous_infinite

   mix = aqueous_uranium(temp_K=294, pressure_MPa=0.1, u_conc_ppm=1000.0)
   result = solve_homogeneous_infinite(mix)

   fig, ax = plt.subplots()
   ax.semilogx(result.eg_mid, result.flux_per_lethargy, 'b-', linewidth=1.2)
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

   f_\mathrm{fuel} = \frac{r_\mathrm{fuel}^2}{r_\mathrm{cell}^2}, \quad
   f_\mathrm{clad} = \frac{r_\mathrm{clad,out}^2 - r_\mathrm{clad,in}^2}
                           {r_\mathrm{cell}^2}, \quad
   f_\mathrm{cool} = \frac{r_\mathrm{cell}^2 - r_\mathrm{clad,out}^2}
                           {r_\mathrm{cell}^2}

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

   from data.macro_xs.recipes import pwr_like_mix
   from homogeneous import solve_homogeneous_infinite

   mix = pwr_like_mix()
   result = solve_homogeneous_infinite(mix)

   fig, ax = plt.subplots()
   ax.semilogx(result.eg_mid, result.flux_per_lethargy, 'r-', linewidth=1.2)
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


Verification
=============

The homogeneous solver is verified against **analytical eigenvalues**
derived symbolically with SymPy (see :mod:`derivations.homogeneous`).
The same :class:`~derivations._types.VerificationCase` objects serve
both the documentation (LaTeX equations in :doc:`verification`) and the
test suite.

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 25 15

   * - Benchmark
     - Groups
     - Regions
     - :math:`\kinf` (analytical)
     - Error
   * - ``homo_1eg``
     - 1
     - 1
     - 1.5000000000
     - :math:`< 10^{-12}`
   * - ``homo_2eg``
     - 2
     - 1
     - 1.8750000000
     - :math:`< 10^{-12}`
   * - ``homo_4eg``
     - 4
     - 1
     - 1.4877619048
     - :math:`< 10^{-12}`

All three benchmarks achieve **machine-precision** agreement
(:math:`< 10^{-12}`), confirming that the solver correctly implements
the matrix eigenvalue algebra.

Additionally, the two 421-group industrial problems (aqueous uranium
and PWR-like mixture) are verified against the original MATLAB
implementation to 5 significant digits.

Run the verification suite::

   pytest tests/test_homogeneous.py -v


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
     - 3--5 outer
     - 10--20 outer
     - 20--50 outer
     - 100+ outer
   * - Eigenvalue computed
     - :math:`\kinf`
     - :math:`\kinf` (lattice)
     - :math:`\kinf` (lattice)
     - :math:`\keff` (core)
   * - Implementation
     - :class:`HomogeneousSolver`
     - :class:`CPSolver`
     - :class:`SNSolver`
     - :class:`DiffusionSolver`


References
==========

.. [Duderstadt1976] J.J. Duderstadt and L.J. Hamilton, *Nuclear Reactor
   Analysis*, Wiley, 1976.

.. [Bondarenko1964] I.I. Bondarenko et al., *Group Constants for Nuclear
   Reactor Calculations*, Consultants Bureau, 1964.

.. [Hebert2009] A. Hebert, *Applied Reactor Physics*, Presses
   internationales Polytechnique, 2009.

.. [Stacey2007] W.M. Stacey, *Nuclear Reactor Physics*, 2nd ed.,
   Wiley-VCH, 2007.
