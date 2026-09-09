Homogeneous Infinite-Medium Solver
====================================

The :mod:`orpheus.homogeneous` package solves the multi-group
eigenvalue problem in an infinite homogeneous medium — the
simplest reactor physics configuration and the foundation for
every spatial solver in ORPHEUS. Because the flux is uniform and
all streaming terms vanish, the transport equation collapses to a
single :math:`G \times G` dense eigenvalue problem solved directly
(no iteration), with an exact analytical structure that makes it the
go-to harness for L0 / L1 verification of cross-section libraries and
scattering-matrix conventions.

.. contents::
   :local:
   :depth: 2

.. seealso::

   :ref:`theory-homogeneous` — full derivation, scattering
   convention, and worked examples.


Eigenvalue Problem
------------------

With no spatial dependence, the multi-group transport equation
reduces to

.. math::

   \underbrace{\bigl(\operatorname{diag}(\Sigma_t)
   - \Sigma_{s0}^{\mathsf T}
   - 2\,\Sigma_{2}^{\mathsf T}\bigr)}_{\mathbf{A}}\,\phi
   \;=\; \frac{1}{k_\infty}\,
   \underbrace{\bigl(\chi \otimes \nu\Sigma_f\bigr)}_{\mathbf{F}}\,\phi,

where :math:`\Sigma_{s0}` is the :math:`P_0` (isotropic) scattering
matrix, :math:`\Sigma_2` is the (n,2n) cross-section matrix (stored
separately because each collision produces two neutrons), and
:math:`\chi` is the prompt fission spectrum. The :math:`(n,2n)`
reaction is a **loss-side multiplicity-2 transfer**: the
:math:`-2\,\Sigma_2^{\mathsf T}` term in the loss matrix
:math:`\mathbf{A}` removes the incident neutron and redistributes the
two emitted neutrons by the :math:`(n,2n)` transfer kernel. It does
**not** enter the production matrix :math:`\mathbf{F}` — the two
neutrons are not produced with the fission spectrum :math:`\chi`, so
production is :math:`\nu\Sigma_f` only. (A retired bespoke formulation
added :math:`2\,\mathrm{colsum}(\Sigma_2)` to the production numerator,
double-counting the :math:`(n,2n)` neutrons; see
:ref:`theory-homogeneous` for the de-vacuum case.)

**Scattering convention.**
:attr:`~orpheus.data.macro_xs.mixture.Mixture.SigS` stores matrices
in ``SigS[g_from, g_to]`` order — **the source uses the transpose**,
:math:`Q_{\rm scatter} = \Sigma_{s}^{\mathsf T}\phi`. The same
transpose appears in the removal matrix
:math:`\Sigma_{s0}^{\mathsf T}` above. This is the single convention
every ORPHEUS solver follows; mis-transposing is the most common
bug when porting from other codes and is caught by L0 spectrum
tests on asymmetric scattering matrices.


Implementation
--------------

The package ships two objects.
:class:`~orpheus.homogeneous.solver.HomogeneousProblem` is the
**hub** — a frozen dataclass over one
:class:`~orpheus.data.macro_xs.mixture.Mixture` that owns, as
per-instance ``cached_property`` state minted from that mixture and from
nothing else, every object the calculation consumes: the pose space, the
one-cell material layout, the kernel-tier material fields, the three
cross-section fields born on the pose, the bound operators, and the two
typed reaction-rate co-vectors.
:func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite` is the
**solver** — it reads the hub and computes, constructing no data of its
own.  Nothing is fabricated on the path: the infinite medium has no mesh,
so no carrier is built (CS4c coda, 2026-09-08; until then a meshless
single-cell ``MaterialMesh`` supplied the cross sections and its
``from_materials`` factory has since retired — see
:ref:`homogeneous-development-history`).

The hub assembles the loss operator from the **model-shared transport
operators** (campaign #276, Cardinal Rule 2 — the infinite-medium
spectrum runs through the same operator algebra as the meshed SN solver,
not a bespoke matrix), and the solver then takes the dominant eigenpair
directly:

* **Loss matrix** :math:`\mathbf{A} = C - K_\mathrm{iso} =
  \operatorname{diag}(\Sigma_t) - \Sigma_{s0}^{\mathsf T} -
  2\Sigma_2^{\mathsf T}`, with :math:`C = \operatorname{diag}(\Sigma_t)`
  the collision diagonal and :math:`K_\mathrm{iso}` the sum of the
  model-shared
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`
  (:math:`\Sigma_{s0}^{\mathsf T}`) and
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`
  (:math:`2\Sigma_2^{\mathsf T}`) operators. The composed operator
  :math:`C - K_\mathrm{iso}` is materialised densely via its own
  :meth:`~orpheus.numerics.operator.LinearOperator.as_matrix`
  apply-to-basis — the ``(ng, 1)`` basis shape derived from the threaded
  domain (the mixture-minted Energy ⊗ point space
  :attr:`HomogeneousProblem.space
  <orpheus.homogeneous.solver.HomogeneousProblem.space>`; CS4a K2 — until
  then a carrier's axis-built ``bulk_space``, which a genuine one-cell
  carrier still mints ``==`` to as a *reference*, never as the source).
  Streaming :math:`L \equiv 0` in an infinite medium and is dropped.
* **Production dyad** :math:`\mathbf{F} = \chi \otimes \nu\Sigma_f`,
  the rank-1 form of the fission energy binding
  :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
  (CS4c step 4 — the same class diffusion and the S\ :sub:`N` k-outer
  consume; the angular binding ``FissionOperator`` has no role in an
  infinite medium), likewise materialised densely via its own
  :meth:`~orpheus.numerics.operator.LinearOperator.as_matrix`.
* **Eigenpair** :math:`k_\infty = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})`
  and the dominant right eigenvector: one ``scipy.linalg.lu_factor`` of
  :math:`\mathbf{A}`, taken eagerly when
  :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
  is constructed and reused for every basis column, plus one
  :func:`numpy.linalg.eig` on the materialised
  :math:`[\mathbf{K}] = \mathbf{A}^{-1}\mathbf{F}`. There is **no inner
  or outer iteration**.

The function normalises the flux so that the **fission** production
rate :math:`\nu\Sigma_f\cdot\phi` equals :math:`100\ {\rm n/cm^3/s}`
(production is :math:`\nu\Sigma_f` only — :math:`(n,2n)` lives in
:math:`\mathbf{A}`, not :math:`\mathbf{F}`), computes the one-group
collapsed production and absorption cross sections, and packages
everything into a
:class:`~orpheus.homogeneous.solver.HomogeneousResult`.

**Why no iteration is needed.**
In an infinite homogeneous medium there is no spatial eigenmode
spectrum to filter and no spatial coupling to invert iteratively —
the loss operator is a single :math:`G \times G` dense block, so the
fundamental eigenpair is taken in closed form by the dense
eigensolver. This makes the homogeneous solver the one deterministic
solver in ORPHEUS with no iteration at all, and the instantaneous
reference eigenvalue for every other solver on a homogeneous problem.


API Reference
-------------

.. automodule:: orpheus.homogeneous.solver
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
