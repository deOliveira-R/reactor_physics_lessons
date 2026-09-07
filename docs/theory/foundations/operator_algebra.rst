.. _operator-algebra:

=============================
Operator Algebra Architecture
=============================

ORPHEUS' transport, eigenvalue, and Krylov solvers all act on a flux
distribution :math:`\psi` by composing a small set of linear operators,
each with a **distinct intrinsic mathematical type**:

- **collision** :math:`C = M[\sigma_t]` — a *multiplication operator*
  (diagonal; pointwise multiplication by the total cross section);
- **scattering** :math:`S = R\circ\Lambda\circ M` and **fission**
  :math:`F = |\chi\rangle\langle\nu\Sigma_f|` — *integral kernels*
  (nonlocal: scattering redistributes in angle, fission is the rank-1
  emission dyad);
- **streaming** :math:`L = \hat\Omega\cdot\nabla` and the **boundary
  law** :math:`B` — the leakage and its trace closure.

They compose into the **within-group transport operator**

.. (vv-status rationale) The governing within-group composition
   A = L+C−S−B — the loss operator. Definitional identity; the
   assembled composite is exercised by build_within_group_system and
   the fixed-source / eigenvalue suites, matching the sentineled
   operator-fixed-source / operator-eigenvalue siblings.
.. vv-status: operator-within-group-composition documented

.. math::
   :label: operator-within-group-composition

   A \;=\; L + C - S - B ,

posed either as an eigenvalue problem :math:`A\psi = \tfrac{1}{k}F\psi`
or a fixed-source problem :math:`A\psi = q`. Here :math:`A` is the
**loss operator** — removal and leakage net of the within-group gains —
against the fission **gain** :math:`F`. The boundary law :math:`B` is a
**first-class sibling** (not folded into :math:`L`), and :math:`F` sits
on the **right-hand side** — never inside :math:`A`. The invertible
sub-composite :math:`L+C` — streaming leakage plus total collision — has
the transport **sweep** as its exact inverse; :math:`S` and :math:`B`
are the within-group gains the outer iteration lags.

.. implements:: operator-within-group-composition
   :by: orpheus.sn.coupled_system.build_within_group_system

   **Implemented by** the one production spelling of the composition.
   The assembled System-A diagonal block is ``A_AA = LC - S - B_a`` over
   ``LC = build_streaming_collision(sn_mesh, mat_xs)`` — i.e. :math:`A =
   L+C-S-B` written as operator arithmetic, with :math:`B` a first-class
   sibling rather than something folded into :math:`L`. The same function
   returns the :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record
   carrying the named splitting :math:`A = M - N`
   (:ref:`coupled-block-operator`), so every SN within-group solve — SI and
   Krylov, fixed-source and eigenvalue — reads :math:`A` from here and there
   is no second assembly to drift against.

The :mod:`orpheus.numerics.operator` module installs these as a uniform
*matrix-free* algebra, so the eigenvalue, fixed-source, and
preconditioned-Krylov code consumes any method (S\ :sub:`N` / MoC / CP /
diffusion) without knowing which transport discretisation lives
underneath. This page is that algebra's reference: the intrinsic type of
each operator, the composition laws, the invertible :math:`L+C` and its
:term:`sweep`, and how a method extends the shared operators (S\ :sub:`N`
expands :math:`S` for anisotropy).

.. contents::
   :local:
   :depth: 2


Key Facts
=========

- **The realized boundary law** :math:`B` **is a first-class sibling
  operator** (Issue #208): the within-group transport
  operator is :math:`A = L + C - S - B` on the **two-block** transport
  state :math:`V = V_{\rm bulk} \oplus V_{\rm boundary}` (the bulk
  :term:`angular flux` :math:`\oplus` a single boundary trace — inflow and
  outflow fold into one :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`),
  posed :math:`A\psi = \tfrac{1}{k}F\psi` or :math:`A\psi = q`. The
  fission gain :math:`F` **never enters** :math:`A` — it is applied on
  the right-hand side and divided by :math:`k` at the eigenvalue layer.
  The boundary reflection is **no longer re-applied inside the streaming
  sweep** (the deleted "keystone"); it is delivered as the off-diagonal
  :math:`-B` coupling, and the outer Krylov / SI loop drives the
  boundary-consistency residual to zero. See :ref:`bc-extraction`.

- **Every operator's** ``.apply`` **output boundary is a**
  :class:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink`
  (Issue #208), completing
  the boundary half of the operator-output "dimensional-sin" carve (the
  bulk half — ``.apply.bulk`` →
  :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`).
  The governing principle: an operator output
  is :math:`A\psi` — a **source/sink**, NOT a residual; the residual
  arises ONLY from an explicit :meth:`~orpheus.transport.residuals.angular_boundary_residual.AngularBoundaryResidual.from_balance`
  of the output against a source. The completed boundary role grid
  mirrors the bulk:
  :math:`\texttt{.apply} \to \texttt{AngularBoundarySourceSink}`,
  :math:`\texttt{.solve} \to \texttt{AngularBoundaryFlux}`,
  :math:`\texttt{from\_balance} \to \texttt{AngularBoundaryResidual}`. See
  :ref:`bc-extraction-operator-output-typing`.

- **Flux lives in the positive cone** :math:`K` **of an ordered vector
  space** :math:`V`\ **; a flux difference is the same type, signed, and**
  ``flux + flux`` **is legal** (Issue #331; campaign-1 CS3). The flux
  states :class:`~orpheus.transport.fields.angular_flux.AngularFlux` /
  :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux` /
  :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
  / :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux` live in
  the **positive cone** :math:`K` of an ordered **vector space**
  :math:`V`. ``flux + flux`` is legal (superposition is a theorem of the
  linear operator), the source-iteration increment
  :math:`\Delta\psi = \psi^{(i)} - \psi^{(i-1)}` is the **same** leaf
  type carrying a signed value, and cone membership is an element
  **predicate** (:meth:`Field.cone_violations
  <orpheus.numerics.field.Field.cone_violations>`), never a constructor
  invariant — diamond difference does not preserve :math:`K`, so a
  :math:`\psi\ge0` type would refuse production output. What "fluxes of
  different problems don't mix" really enforces is the **fiber** — class
  identity plus space CONTENT equality on :meth:`Field._check_partner
  <orpheus.numerics.field.Field._check_partner>` (the mesh-object tier
  it carried at CS3 retired at CS4b S3;
  :ref:`cone-fiber-discipline`) — and the iterate
  diagnostics that a state structurally cannot carry live on the
  **iteration record**
  (:attr:`IterationRecord.increment_norms
  <orpheus.numerics.convergence.IterationRecord.increment_norms>`, from
  which :math:`\rho` and the :math:`c\to1` true-error estimate derive).
  The equation residual :math:`r = (L+C-S-B)\psi - q` is typed via
  :func:`~orpheus.sn.solver.evaluate_residual` (the box-7 consumer of
  the previously-unconsumed ``from_balance`` mint). See
  :ref:`cone-typed-field-algebra`.

  ⛔ Until 2026-08-19 this bullet read *"flux states are an* **affine
  space** :math:`\mathbb{A}` *over a difference vector space* :math:`V`
  *… so* ``flux + flux`` *raises"*. That ontology was overturned at
  campaign-1 CS3; the six-argument adjudication and the retired design
  are at :ref:`cone-overturn-adjudication` and
  :ref:`cone-the-overturned-affine-design`.

- **The carriers form a** :math:`(\text{Representation} \times
  \text{Role})` **double category, and the operator algebra traverses
  it**. A carrier is a cell
  :math:`(\text{Representation}, \text{Role})`: **Representation**
  :math:`\in \{\text{Angular}, \text{Moment}, \text{Scalar},
  \text{Trace}\}` sets the array shape and carries the change-of-basis
  (the Frame); **Role** :math:`\in \{\text{Flux}, \text{Source},
  \text{Residual}\}` sets the arithmetic interface (the units the
  class-identity gate reads, and the cross-class containment injection
  the Source role owns). The **horizontal** 1-morphisms are the
  representation-changing frame faces :math:`M`/:math:`R` (role-generic —
  a base change that fixes the fiber); the **vertical** 1-morphisms are
  the role-changing cross sections :math:`C`/:math:`\Lambda`/:math:`F`
  (representation-generic — the role change *is* the cross-section
  physics); **scattering** :math:`S = \tfrac{1}{W}(R\circ\Lambda\circ M)
  = \texttt{frame.conjugate}(\Lambda)` is the **2-cell**, the vertical
  :math:`\Lambda` conjugated by the horizontal adjoint pair, and the
  bit-identical windowed-vs-full crosscheck is its **interchange-law
  coherence witness**. **A grid cell IS an operator's** ``(Domain,
  Codomain)``:
  :class:`~orpheus.numerics.operator.LinearOperator` ``[Domain,
  Codomain]`` is the typed grid traversal — the parametrization belongs
  on the *operator*, not the carrier, because a fully-typed
  ``Carrier[Representation, Role]`` is **structurally impossible** (Role
  changes ``__add__`` ⟹ Role must be a class; Representation changes
  shape ⟹ Representation must be a class; a parameterized carrier would
  break the runtime units gate via generic erasure). The flat
  multiple-inheritance leaves ``AngularFlux(AngularField)`` /
  ``CrossSectionField(CoefficientRole, ScalarField)`` are
  therefore the **unique principled normal form**, not a compromise. See
  :ref:`carrier-grid-double-category` and
  :ref:`carrier-grid-flat-leaf-normal-form`.

- **The interior cell-face angular fluxes are a 1-cochain**
  :math:`C^1_{\rm int}` (Issue #208): the 2-D wavefront sweep and matvec no
  longer carry raw ephemeral ``psi_x`` / ``psi_y`` numpy arrays. The
  interior 1-cochain :math:`C^1_{\rm int}` and the boundary trace
  :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
  (the boundary 1-cochain :math:`C^1_\partial`) **biproduct-decompose
  the full face cochain** :math:`C^1 = C^1_{\rm int} \oplus
  C^1_\partial` — the :math:`V_{\rm bulk} \oplus V_{\rm boundary}`
  shape of :eq:`bc-extraction-direct-sum-state` one locus down, at the
  *face* level. The seed/absorb the sweep applies are the typed trace
  operators :math:`\iota_*` / :math:`\iota^*`, with the "absorption =
  identity" fact the provable biproduct law :math:`\iota^* \circ
  \iota_* = \mathrm{id}`. The dedicated ``WavefrontFlux`` carrier was
  **retired** (#222): the cochain now lives in the rolling
  front (``_MovingFrontier``) and the full-cochain oracle history
  (``_octant_face_cochain``). See :ref:`wavefront-flux-cochain` for the
  succession.

- **The 2-D Cartesian SI iterate lives in moment space** — the
  within-group source-iteration fixed point
  :math:`\psi_{k+1} = (L{+}C)^{-1}(S\psi_k + B\psi_k + q)` consumes
  :math:`\psi` only through its flux moments :math:`\phi_\ell^m =
  (M\psi)_\ell^m`, so the *persistent* iterate is held as the moment
  tensor :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
  (:math:`N \to (L{+}1)(2L{+}1)`, measured **18.3×** shrink at
  :math:`N=110`, :math:`L=1`) rather than the full :term:`per-ordinate <ordinate>`
  :class:`~orpheus.transport.fields.angular_flux.AngularFlux`. The
  source is **bit-identical** (the moment arm of
  :meth:`ScatteringOperator.apply <orpheus.transport.operators.scattering.ScatteringOperator.apply>`
  shares the :math:`R\,\Lambda` reconstruction with the full-angular
  arm); only the SI convergence test moves to the moment :math:`L^2`
  (principled-equivalence). 2-D Cartesian only (curvilinear's
  Morel–Montry Carlson seed reads the per-ordinate iterate; Krylov
  iterates the full bulk). Interior-bulk only — the trace
  :math:`C^1_\partial` stays un-reduced. See :ref:`sn-angular-windowing`.

- **2-D Cartesian eigenvalue problems solve via BOTH inner solvers**
  (Issue #208): the
  source-iteration inner
  (:meth:`SNSolver._solve_source_iteration <orpheus.sn.solver.SNSolver._solve_source_iteration>`,
  the :func:`~orpheus.sn.solver.solve_sn` default for *every* geometry)
  AND the Krylov inner
  (:meth:`SNSolver._solve_krylov <orpheus.sn.solver.SNSolver._solve_krylov>`).
  The SI inner is the **geometry-agnostic structural twin** of Krylov:
  identical composite RHS, identical loss decomposition (the invertible
  resolvent :math:`L + C` plus the two lagged coupling gains — the bulk
  scattering :math:`S` and the trace boundary reflection :math:`B` —
  delivered to the **variadic** driver per :ref:`bc-extraction-variadic-driver`;
  zero within-group fission), identical angular reduction — differing
  **only** in the iteration driver. The reflective coupling rides the **bare** 2-D
  sweep via the sibling :math:`-B` on the natively four-face
  (``xmin`` / ``xmax`` / ``ymin`` / ``ymax``)
  :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator` — the same
  operator the 2-D Krylov path uses. The legacy "B1'' face block"
  (never a code symbol; a 1-D boundary-closure *name*) is retired. See
  :ref:`bc-extraction-2d-si-krylov-twin`.

- The transport eigenvalue and fixed-source problems share **one**
  operator algebra: they differ only in what sits on the right of the
  within-group transport operator :math:`A = L + C - S - B`.

  .. (vv-status rationale) The within-group fixed-source form
     A ψ = q, A = L+C-S-B. Verified by the operator-algebra assembly
     (build_within_group_system) and the fixed-source solver suites.
  .. vv-status: operator-fixed-source documented

  .. math::
     :label: operator-fixed-source

     A\,\psi \;=\; q
     \qquad \text{(fixed source)}

  .. implements:: operator-fixed-source
     :by: orpheus.sn.solver.SNSolver._solve_source_iteration

     **Implemented by** the source-iteration within-group arm — one of
     the two realizations that *pose* :math:`A\psi = q`. Its own docstring
     carries the posing verbatim: :math:`A = L+C`, :math:`S = \tfrac{1}{W}\,
     \text{full multi-group scatter}`, :math:`F = 0_{\rm wg}`, with the
     external source on the right as ``q_ext_composite``. A posing equation is
     implemented by the code that **assembles the two sides**, so the
     method-agnostic drivers
     (:meth:`SourceIteration.solve <orpheus.numerics.iteration.SourceIteration.solve>`,
     the Krylov acceleration) and the three-line dispatcher
     :meth:`~orpheus.sn.solver.SNSolver.solve_fixed_source` are consumers, not
     implementers — they solve :math:`(A - \sum\text{gains})\psi = q` without
     knowing what the leaves are.

  .. implements:: operator-fixed-source
     :by: orpheus.sn.solver.SNSolver._solve_krylov

     **Implemented by** the Krylov within-group arm, the structural twin
     of the SI arm: the same
     :func:`~orpheus.sn.coupled_system.build_within_group_system` call, the
     same implicit operator plus explicit lagged gains, the same
     ``q_ext_composite`` on the right — differing **only** in the iteration
     driver. If the posing were wrong, both arms would assemble the wrong
     problem directly, which is exactly the test for an implementer.

  .. (vv-status rationale) The k-eigenvalue form A ψ = (1/k) F ψ,
     A = L+C-S-B, with F the right-hand-side fission gain. Verified by
     the eigenvalue engines (power iteration and K = A⁻¹F) against the
     closed-form k∞ oracle.
  .. vv-status: operator-eigenvalue documented

  .. math::
     :label: operator-eigenvalue

     A\,\psi \;=\; \tfrac{1}{k}\,F\,\psi
     \qquad \text{(eigenvalue)}

  .. implements:: operator-eigenvalue
     :by: orpheus.sn.solver.SNSolver.compute_fission_source

     **Implemented by** the :math:`\tfrac{1}{k}F\psi` right-hand side —
     which is the *only* thing distinguishing this equation from
     :eq:`operator-fixed-source` ("they differ only in what sits on the
     right"). The body is literally ``self.fission_op.apply(φ) / keff``: the
     :math:`1/k` division stays at the solver level precisely because
     :math:`F` is a **linear** operator and the eigenvalue scaling is not part
     of it.

  .. implements:: operator-eigenvalue
     :by: orpheus.numerics.iteration.KEigenvalue.compute_fission_source

     **Implemented by** the same right-hand side at the operator-triple
     layer, ``self.F.apply(flux_distribution) / keff`` — the k-posing's
     eigen-operator :math:`M = F`. Two postings of one equation, one per
     tier.

  Both are built from operator addition, subtraction, scalar
  multiplication, and composition (``+``, ``-``, ``*``, ``@``) acting on
  :class:`~orpheus.numerics.operator.LinearOperator` instances; the
  fission gain :math:`F` is applied on the right and never enters
  :math:`A`.

- **The eigenvalue problem is layered into four tiers — leaves,
  posing, resolvent, algorithm**. The canonical **standard form** is the generalized
  eigenproblem :math:`A_{\rm loss}\,\psi = \lambda\,M\,\psi`, whose
  power-method realization is the dominant eigenpair of the
  **resolvent** :math:`A_{\rm loss}^{-1} M`. The **k-eigenvalue** row
  is :math:`A_{\rm loss} = L+C-S-B`, :math:`M = F`, :math:`k = \mu`;
  the :math:`\alpha`-eigenvalue, adjoint, and transient rows are
  **documented future seams**.
  :func:`~orpheus.numerics.eigenvalue.power_iteration` is the
  **canonical Layer-4 algorithm** (NOT deprecated — it is the *more
  general* layer, binding the resolvent late through the opaque
  :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` Protocol so it
  admits monolithic-matrix resolvents that have no :math:`(A,S,F)`
  triple);
  :class:`~orpheus.numerics.iteration.KEigenvalue` is the
  operator-triple realization that *delegates* its loop to it (one
  loop in the codebase, Cardinal Rule 2). See
  :ref:`eigenvalue-posing`.

- **The Hilbert adjoint** ``op.H`` **is the metric-correct G-adjoint**
  :math:`A^{\dagger} = G^{-1} A^{\mathsf T} G`, NOT the Euclidean transpose. For the SN composite the
  metric :math:`G` is **block-diagonal** on :math:`V_{\rm
  bulk}\oplus V_{\rm trace}`: bulk :math:`V_{\rm cell}\,w_n` (phase-space
  measure) :math:`\oplus` trace :math:`|\Omega\cdot\hat n_f|\,w_n`
  (partial-current surface measure, pseudo-inverted on the singular
  tangential ordinates). The carrier is
  :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`;
  ``L``/``C``/``S``/``F``/``B`` all carry it so the
  within-group :class:`~orpheus.numerics.operator.OperatorSum` guard
  VALIDATES the composition, and — because every loss leaf carries the
  composite metric — the adjoint is applied **once at the op level**
  (the ``AdjointOperator`` wrapper reads :math:`G` off the composite
  domain) and is never metric-blind. Any non-adjointable operand still
  makes its composite non-adjointable, so the recursive
  :attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable`
  reports ``False`` and ``.H`` *raises*
  :class:`~orpheus.numerics.operator.MissingAdjoint` **eagerly at
  construction**, never silently goes Euclidean. See
  :ref:`g-adjoint`.

- The :class:`~orpheus.numerics.operator.LinearOperator` Protocol
  carries one mandatory method (``apply``) and, per optional axis
  (inverse, adjoint, and **assembly**,
  :ref:`operator-algebra-assembly-axis`), a **three-layer structural
  surface** (#226): a runtime **predicate**
  (:attr:`~orpheus.numerics.operator.LinearOperator.is_invertible` /
  :attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable` /
  :attr:`~orpheus.numerics.operator.LinearOperator.is_assemblable`), an
  **operator-returning method** (``inverse()`` per-class /
  :attr:`~orpheus.numerics.operator.LinearOperator.H` on the base /
  ``assemble()`` →
  :class:`~orpheus.numerics.assembled_operator.SparseAssembledOperator`),
  and a **realization verb** (``solve`` / ``apply_transpose``) present
  only where a native realization exists — each axis refusing eagerly via
  its own ``TypeError`` sibling
  (:class:`~orpheus.numerics.operator.NotInvertible` /
  :class:`~orpheus.numerics.operator.MissingAdjoint` /
  :class:`~orpheus.numerics.operator.MissingAssembly`). The stringly-typed
  ``capabilities: frozenset[str]`` advertisement it replaced (``CAP_*``
  tags + ``MissingCapability``) is **retired** — the surface itself is
  now the single source of truth (:ref:`capability-set-semantics`).

- **SN array-storage convention** for every operator leaf
  (:class:`~orpheus.sn.operators.streaming.StreamingOperator`, the collision
  multiplier :math:`C = M[\sigma_t]`
  (:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
  :class:`~orpheus.transport.operators.scattering.ScatteringOperator`,
  :class:`~orpheus.transport.operators.fission.FissionOperator`): the
  ``apply(psi) -> psi'`` contract consumes and returns
  ``psi.shape == (N, ng, nx, ny)`` for angular flux,
  ``phi.shape == (ng, nx, ny)`` for :term:`scalar flux`.  The canonical
  statement with derivation and migration history lives at
  :ref:`theory-sn-index-convention`.

- The **operator surface itself** is the single source of truth for
  what an operator can do — a method exists (or is refused) exactly
  where the ability does, with no parallel string registry that could
  drift. Composition primitives compute their own
  :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible` /
  :attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable`
  recursively from the operands; mismatches fail at composition time,
  NEVER mid-iteration — an ``apply`` mismatch raises ``TypeError``
  eagerly, a non-adjointable ``.H`` raises
  :class:`~orpheus.numerics.operator.MissingAdjoint` at construction,
  and a value-dependent ``inverse()`` raises
  :class:`~orpheus.numerics.operator.NotInvertible` before any inverse
  object exists.

- **The curvilinear ψ½ ray is System B of a 2×2 coupled block operator**
  (GH #280/#282). The within-group
  augmented S\ :sub:`N` problem is posed as
  :math:`\bigl[\begin{smallmatrix} A_{AA} & A_{AB} \\ A_{BA} & A_{BB}
  \end{smallmatrix}\bigr]\bigl[\begin{smallmatrix} \psi_A \\ \psi_B
  \end{smallmatrix}\bigr] = \bigl[\begin{smallmatrix} q_A \\ q_B
  \end{smallmatrix}\bigr]` over **System A** (the transport
  :class:`~orpheus.transport.full_field.FullField`) and **System B** (the
  ψ½ radial-characteristic ray, its own
  :class:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField`
  closed by a two-point BVP). The four blocks are named operators in
  :mod:`orpheus.sn.operators.radial_characteristic` — the seed
  :math:`A_{AB}` (σ-independent), the emission
  :math:`A_{BA} = \mathrm{Fold}\circ K_{\rm iso}\circ\int d\mu`, the
  radial march :math:`A_{BB}` (its direct Carlson solve IS the inverse) —
  assembled by the N-general
  :class:`~orpheus.numerics.coupled_system.CoupledOperator` machinery. The
  one production spelling is
  :func:`~orpheus.sn.coupled_system.build_within_group_system`, which
  returns the :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record
  carrying the named splitting :math:`A = M - N`: the resolvent
  :math:`M = \bigl[\begin{smallmatrix} L+C & \text{Seeding} \\ \mathbf 0
  & A_{BB}\end{smallmatrix}\bigr]` solves block-triangular (System B
  first), the emission gain rides :math:`N` (lagged). Presence is
  **structural** (R12a — System B exists iff the mesh carries a ray; a
  mismatched composite is unconstructable). The stop is the ρ-honest
  free-identity residual with a driver-level lag-death
  :class:`~orpheus.sn.solver.ConvergenceCertificateError`. See
  :ref:`coupled-block-operator`.


Definitions
===========

The three primitive actions on a flux vector :math:`\psi`:

.. (vv-status rationale) Phase 0 stub label for the ``apply`` primitive
   action. Verified at the protocol level by
   ``tests/numerics/test_operator.py`` (foundation-tagged software
   invariants); a per-solver Phase-1 test will check that each
   solver's ``L.apply(x)`` matches its legacy path bit-for-bit.
.. vv-status: operator-apply documented

.. math::
   :label: operator-apply

   \texttt{apply} \;:\; x \;\mapsto\; L\,x

.. implements:: operator-apply
   :by: orpheus.numerics.operator.LinearOperator.apply

   **Implemented by** the Protocol's own declaration of the verb, whose
   docstring *is* this equation. ``apply`` is the one **mandatory** member
   of :class:`~orpheus.numerics.operator.LinearOperator` — by the
   base-hosting rule below, a method lives on the base exactly when a
   universal realization exists — so the declaration site and the equation
   coincide. The concrete overrides implement their **own** equations
   (:eq:`diagonal-operator-action`, :eq:`multiplication-operator-action`,
   :eq:`tensor-product-action`, :eq:`inverse-as-operator`), not this one.

.. (vv-status rationale) Phase 0 stub label for the ``solve``
   primitive action — the algorithmic dual of ``apply``, NOT the
   matrix inverse. Verified at the protocol level by
   ``tests/numerics/test_operator.py``; per-solver verification is
   queued for the BiCGSTAB-consumer migration (Issue 15).
.. vv-status: operator-solve documented

.. math::
   :label: operator-solve

   \texttt{solve} \;:\; b \;\mapsto\; L^{-1}\,b
   \quad \text{(the algorithmic dual of } \texttt{apply}\text{)}


.. no-implementation:: operator-solve
   :kind: definition

   **Nothing implements this**, and the page already says why 20 lines up:
   *"no universal realization exists — so the declaration site and the
   equation coincide. The concrete overrides implement their own equations
   … not this one."* ``solve`` is a verb the Protocol declares; every
   concrete solver implements its OWN equation
   (:eq:`diagonal-operator-action`, :eq:`inverse-as-operator`, …). A
   declaration here would have to name one of them, which would assert that
   that solver *is* the definition of the verb.

.. (vv-status rationale) Phase 0 stub label for the
   ``apply_transpose`` primitive action. Verified at the protocol
   level by ``tests/numerics/test_operator.py``; per-solver adjoint
   sensitivity tests are queued for the sensitivity track (Issue 17).
.. vv-status: operator-apply-transpose documented

.. math::
   :label: operator-apply-transpose

   \texttt{apply\_transpose} \;:\; x \;\mapsto\; L^{T}\,x

.. implements:: operator-apply-transpose
   :by: orpheus.numerics.operator.SupportsAdjoint.apply_transpose

   **Implemented by** the algebra's single declaration of the raw
   **Euclidean** transpose verb. Unlike ``apply`` it is *not* base-hosted —
   no universal realization exists — so it lives on the narrowing Protocol
   :class:`~orpheus.numerics.operator.SupportsAdjoint`, and that member is
   this equation's declaration site.

   ⚠ Read the equation as written: this is :math:`L^{\mathsf T}`, the
   Euclidean transpose. The **metric Hilbert adjoint** reached through
   ``op.H`` is the different object :math:`A^{\dagger} = G^{-1}A^{\mathsf
   T}G` (:ref:`g-adjoint`), and ``apply_transpose`` is the raw ingredient
   the ``AdjointOperator`` wrapper conjugates — never the adjoint itself.

The dual relationship in :eq:`operator-solve` is **algorithmic**, not
matrix-theoretic: ``solve(L, b)`` returns whatever vector the
operator's solve algorithm produces such that
``apply(solve(b)) ≈ b`` to working precision. For a sparse direct
factorisation this is the matrix inverse; for an iterative
preconditioned solver it is an approximate inverse; for a
matrix-vector product wrapping BiCGSTAB it is the Krylov approximate
solution. The protocol does NOT require ``apply ∘ solve = I`` exactly
— only that ``solve`` realizes a meaningful approximation. Operators
without an efficient inverse action simply report
:attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`
``= False`` and carry no ``solve`` verb, and downstream code declines
them at composition time (:ref:`capability-set-semantics`).


.. _heteromorphic-apply-typing:

Typing the heteromorphic ``apply`` — the ends select the body
--------------------------------------------------------------

The :eq:`operator-apply` contract is nominally an **endomorphism**
``apply(x: V) -> V`` — flux in, flux of the same type out. That is the
honest signature for the streaming, collision, and boundary leaves. The
collision gains are not endomorphisms: a gain reads a flux and emits a
*source/sink* (the §B.5.2 truth), and until 2026-09-04 it could be handed
several different carriers for the same physics — a per-ordinate flux, a
harmonic-moment iterate, a scalar flux, the composite — so its ``apply``
mapped *each input carrier to a distinct output carrier*.

**The current answer is that the question does not arise: the carrier is
a consequence of the binding, decided once at construction.** A bound
operator is an arrow between two declared spaces (:eq:`operator-apply`),
and those two ends already say what its operand is — so an operator
constructed with composite ends admits exactly the
:class:`~orpheus.transport.full_field.FullField` riding its bound
interior, one bound on plain spaces admits exactly the bare array of its
end's shape, and *which body runs* is fixed at construction rather than
re-decided per call. The two admissions are the family's ONLY carrier
parses (:func:`!orpheus.transport.operators.lift.admit_composite` /
:func:`!admit_array`); everything else is a typed refusal naming the
operator and both spaces.

For the angular gains the selection has a second, sharper coordinate:
the retained flux-analysis face :math:`M \otimes I` has **two ends**, and
the binding's domain interior is one of them — the per-ordinate angular
space it reads, or the harmonic-moment space it writes. Which one it is
selects the interior body (:math:`\phi` by angular integration vs
:math:`\phi` as the :math:`\ell = 0` slot; ``frame.conjugate`` vs
``frame.reconstruct_after``; a per-ordinate cotangent vs a moment one).
A third interior is refused. The base that owns this is
:class:`!orpheus.transport.operators.angular_lift.AngularLift`, and the
re-binding verb is ``on_moment_domain()`` — see
:ref:`cs4c-ends-select-the-body`.

`[M]` 2026-09-04, by AST over ``orpheus/transport/operators/``: **0**
``singledispatchmethod`` decorators (was 3, 13 arms), **0** occurrences of
``_apply_impl`` anywhere in ``orpheus/``, and **2** carrier ``isinstance``
tests inside an ``apply`` / ``apply_transpose`` / ``solve`` body (was 12) —
both of them
:meth:`LegendreMomentTransfer.apply <orpheus.transport.operators.transfer.LegendreMomentTransfer.apply>`
and its transpose, the **declared exemption** of ruling R-5 (the typed
moment hatch the minted source-reconstruction face exists for).

``@overload`` survives, and it is now honest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What retired is the *dispatcher*, not the typed surface. Three verbs on
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
(``apply`` / ``solve`` / ``apply_transpose``) still carry
:func:`typing.overload` stubs, and so does
:meth:`LegendreMomentTransfer.apply <orpheus.transport.operators.transfer.LegendreMomentTransfer.apply>`.
The difference is what sits underneath them: a **real method with a real
body**, not a ``TYPE_CHECKING`` alias of a dispatcher. The multiplier's
overloads name its two *bindings* — ``FullField -> FullField`` on a
composite binding, ``ndarray -> ndarray`` on a plain one — and the body
branches on a flag frozen in ``__post_init__`` from the DOMAIN, never on
the operand's class. That is the same information the overloads promise,
made a property of the object instead of a property of the call.

.. _pattern-m-history:

⛔ Pattern M — retired 2026-09-04 (CS4c step 5), kept as the record
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything below describes the shape the collision gains carried from
#257 S8c until CS4c step 5. It is preserved because the *reasons* the two
obvious spellings fail are still true of any future multi-carrier verb,
and because the question this section parked is now answered.

Python has no native multiple dispatch, and the two stdlib tools each
fell short alone:

* **A raising endomorphism base poisons the type.** The natural
  ``@singledispatchmethod`` shape has a *base* method that raises
  :class:`TypeError` for unregistered carriers. If that base is named
  ``apply`` and inherits the mixin's nominal ``apply(x: V) -> V``, the
  raising body makes pyright infer ``singledispatchmethod[NoReturn]`` —
  so every caller of ``apply`` statically sees ``NoReturn`` (a poison
  type that contaminates downstream inference), and every
  ``@apply.register`` arm errors against the inherited endomorphism
  signature.

* **An** :func:`~typing.overload` **stub cannot carry the body.**
  ``@overload`` is a type-checker fiction erased at runtime: an overload
  signature is *only* a signature, so it can never hold the
  source-assembly math (≈150 lines for scattering). And
  ``@singledispatchmethod`` *does* carry per-carrier bodies with
  automatic routing — its only failing was the homomorphic
  ``singledispatchmethod[_T]`` typing.

**Pattern M** kept ``@singledispatchmethod`` for its routing and bodies
and bolted a typed surface on top: (1) rename the dispatcher ``apply`` →
``_apply_impl`` with an ``-> Any`` base, so each ``@_apply_impl.register``
arm is a bodied, *real-typed* function (e.g. ``def _(self, phi:
ScalarFlux) -> ScalarSourceSink``) that ``.register`` accepts at its
natural indentation; (2) add the typed surface only for the type checker:

.. code-block:: python

   # RETIRED 2026-09-04 — the shape until CS4c step 5.
   if TYPE_CHECKING:
       @overload
       def apply(self, phi: ScalarFlux, /) -> ScalarSourceSink: ...
       @overload
       def apply(self, psi: FullField, /) -> FullField: ...
       # ... one overload per carrier ...
       def apply(self, x: Any, /) -> Any: ...
   else:
       apply = _apply_impl

At runtime the ``else`` branch made the public ``apply`` the **same
object** as the dispatcher — ``Type.__dict__['apply'] is
Type.__dict__['_apply_impl']`` was ``True`` — so runtime was
byte-identical to the untyped version while pyright saw the per-carrier
overloads. Pattern M was chosen over a ``TYPE_CHECKING`` / ``else``
*split* of the whole method (Pattern C — a fully-typed ``apply`` under
``if TYPE_CHECKING`` and the dispatcher under ``else``) on the
master-standard rule ordering: "code reveals intention" (Beck rule 2)
outranks "fewest elements" (rule 4), and Pattern C buried the bulk of the
source-assembly math inside an ``else:`` block.

✅ **The parked question is answered, and neither candidate won.** This
section used to close by parking the deeper *spelling* question — Pattern
M versus a thin ``@overload`` + :keyword:`match` router over shared
primitives — on #261, "to be settled together with the C / F / S core
relocation, because the sharing should dictate the form". The relocation
happened (CS4c steps 3–5: the transfer core, the two role subclasses, the
shared lift base), and it dictated a **third** form. A ``match`` router
is still a per-call parse of the operand; what the sharing actually
revealed is that the operand's kind is not free — it is *implied by the
ends the operator was constructed with*. So the router was not written
thinner; it was deleted, and its decision moved to ``__post_init__``.
The ``psi.bulk`` cast #262 tracked goes with it: the interior is reached
through the admission, which has already proved the space.

⚠ **What this cost, honestly.** The C6 gate
(``tests/sn/operators/test_operators_apply_typed.py``) pinned the alias
identity and a runtime dispatch-parity check on the public ``apply``;
those rows pin a mechanism that no longer exists. Their successor is the
AST no-dispatch census (``tests/transport/test_no_carrier_dispatch.py``,
which states its predicate and declares its carve-outs) plus the
ends-to-body fence in ``tests/transport/test_kernels.py`` — a
per-``(operator, binding kind, carrier)`` matrix whose off-binding cells
are typed refusals. The lexical AST census cannot see a carrier parse one
frame out in a helper; that limitation is stated in the gate itself
rather than left to the reader.


.. _cs4c-ends-select-the-body:

Each binding acts through the body its ends select
----------------------------------------------------

The section above says the carrier follows from the binding. This one
says what that buys, because the payoff is larger than a tidier
signature: **a per-call parse of the operand is a decision the operator
is entitled to have already made**, and making it at construction turns a
class of silent mis-feeds into a construction-time refusal.

The two admissions, and what they refuse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: The transport family's two carrier parses
   :header-rows: 1
   :widths: 20 32 48

   * - Binding
     - Carrier
     - What is refused, naming the operator
   * - **composite** — both ends are
       :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
     - the :class:`~orpheus.transport.full_field.FullField` whose
       interior rides the bound end's interior space (content equality,
       never object identity)
     - a bare array (*"a bare array is the PLAIN binding's carrier"*); a
       typed bulk field outside its composite (*"a typed bulk field
       rides inside a FullField"*); a composite on **another** interior
       — the moment iterate handed to an angular-bound gain, with the
       message naming ``on_moment_domain()``
   * - **plain** — the ends are ordinary
       :class:`~orpheus.numerics.space.FunctionSpace`\ s
     - the bare :class:`numpy.ndarray` of the end's shape (the
       model-portable contract every solver family already feeds)
     - a composite (*"lift the binding onto the composite with
       BulkLift(...)"*); a typed field (*"pass ``.values``"*); a wrong
       shape, with both shapes printed

The plain binding is the energy operators' and the multiplier's
(:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicScattering`,
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicN2N`,
:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`,
and a mesh-free
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`);
the composite binding is the angular gains' and of the composite
multiplier. A consumer that holds a composite and wants an energy
binding's action does not get a ``FullField`` arm on the energy binding —
it *lifts* (next subsection). That is ruling **R-4** of the step-5 design
round, and it is what keeps the array carriers where the numerics tier
(:meth:`LinearOperator.as_matrix
<orpheus.numerics.operator.LinearOperator.as_matrix>`,
:class:`~orpheus.numerics.operator.OperatorSum`, ``power_iteration``'s
protocol vector) can reach them without a typed wrapper.

The angular gains' second coordinate: which END of the analysis face
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An angular gain retains the minted flux-analysis face :math:`M \otimes I`
(:class:`~orpheus.transport.frames.harmonic_frame.HarmonicAnalysisOperator`).
That face is an arrow, so it has two ends, and the binding's domain
interior is required to be one of them:

.. list-table:: Domain interior → interior body, fixed in ``__post_init__``
   :header-rows: 1
   :widths: 22 30 24 24

   * - Domain interior is …
     - the operand IS
     - :math:`\ell = 0` half
     - :math:`\ell \ge 1` half
   * - ``flux_analysis.domain`` — the **angular** end
     - the per-ordinate flux :math:`\psi`
     - :math:`\phi = \int \psi\,d\Omega` (the reaction-rate fast path)
     - the cached kernel :math:`R\,\Lambda_{\ell\ge1}\,M`
   * - ``flux_analysis.codomain`` — the **moment** end
     - already :math:`M\psi` (the 2-D Cartesian windowed iterate)
     - :math:`\phi` = the :math:`\ell = 0` slot (:math:`Y_0^0 = 1`)
     - :math:`R\,\Lambda_{\ell\ge1}` on the typed grid path
       (:math:`M` skipped — re-projecting would double-project)
   * - anything else
     - —
     - :class:`TypeError` at construction
     - :class:`TypeError` at construction

The transposes' cotangents follow the same selection: on the angular end
the cotangent is an
:class:`~orpheus.transport.source_sinks.AngularSourceSink`, on the moment
end a
:class:`~orpheus.transport.source_sinks.HarmonicMomentSourceSink`.

**Why this is not a refactor of a working thing.** Before the step the
windowed SI driver handed a *moment* composite to an operator bound
``(angular, angular)``, which then dispatched on the carrier's class per
call — `[M]` **143 such feeds per windowed solve** on a bit-exact frozen
snapshot. That is a shipped non-endomorphism: the operator's declared
domain and the operand's actual space disagreed, and nothing could say
so, because the arm that handled it was registered on the operator that
did not own that domain. The driver now binds its gains where the
iterate lives (``S.on_moment_domain()``, ``N2N.on_moment_domain()``,
built through :func:`dataclasses.replace` so every admission re-runs),
and the mismatch it used to absorb is a refusal with both spaces in the
message.

⚠ The moment binding is deliberately **not** an endomorphism — its
domain is the moment composite and its codomain the angular composite —
and that is legal here because the windowed loop consumes the gains one
by one; the ``OperatorSum`` ends guard on the within-group
:math:`(L + C) - S - N_{2n} - B` never sees it (that composition is built
from the angular-bound siblings, which are endomorphic on the posed
composite).

`[M]` on a GL8 slab, 2 groups, 20 cells, :math:`P_1` scattering,
200 seeds (``default_rng(s)``, ``s ∈ 0…199``, standard-normal
:math:`\psi`): the angular-bound gain's ``apply`` and its
``on_moment_domain()`` sibling's ``apply`` on :math:`M\psi` agree
**200 / 200 under** ``np.array_equal``, ``max |Δ| = 0.0`` — the two
routes genuinely share :math:`\Lambda` and the frame's :math:`R`. The
committed gate is
``tests/sn/operators/test_scattering_kernel_crosscheck.py``, which runs
the same comparison on a 2-D :math:`P_1` heterogeneous multigroup
fixture and records its own 200-seed sweep.

A bulk action enters the composite by extension by zero
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every volumetric operator of the algebra — the gains :math:`S`,
:math:`N_{2n}`, :math:`F`, the multiplier :math:`M[f]`, a diffusion
energy binding on its scalar composite — acts on the bulk block alone and
emits **nothing** on the trace. `[M]` that one fact was spelled **nine
times** across four modules before this step (``transfer.py`` 2,
``fission.py`` 2, ``isotropic_transfer.py`` 1,
``multiplication_operator.py`` 4), once per operator per carrier family,
each spelling naming the trace's zero class by hand.

It is now one verb,
:func:`!orpheus.transport.operators.lift.lift_bulk_action`: run the
interior body, emit the zero field of *the operand's own boundary class*
in the requested ROLE. The verb names a role, never a leaf; the leaf
comes from the operand (next subsection). Its assembly twin,
:func:`!embed_bulk_assembly`, embeds a bulk operator's sparse emission in
the composite flat layout ``[bulk C-ravel | trace]`` — index-identity on
the leading block, zero on every trace row and column, entries carried
verbatim — so ``assemble`` composes exactly the way ``apply`` does. The
operator that packages both is
:class:`!orpheus.transport.operators.lift.BulkLift`, and the 1-D
diffusion solver is its consumer: `[M]` **2** production construction
sites, both in ``orpheus/diffusion/solver.py``, lifting
``IsotropicScattering + IsotropicN2N`` and ``IsotropicFission`` from the
mesh's scalar bulk onto the scalar composite so the loss
:math:`L + C - \mathrm{lift}(K_{\rm iso}) - B` composes under the
``OperatorSum`` ends guard and assembles for the exact LU resolvent.

⭐ This was #306 item 2, and the ruling (**R-2**) is that the zero-trace
emission is **blessed**, not transitional. It is the honest composite
action of a bulk-role operator — the extension-by-zero half of the
restriction/extension pair — and the reason it looked like a shim was
that it had nine spellings, not that the mathematics was provisional.
Single-sourcing it is what closed the item.

`[M]` by AST over ``orpheus/transport/operators/``: **0** occurrences of
``AngularBoundarySourceSink.zeros`` / ``…BoundaryFlux.zeros`` — the
boundary leaf is not named in the package at all any more; it is read off
the operand's declaration.

.. _role-partner-declaration:

The carriers declare their role partners — the grid's vertical edge, once
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The verb above says *"emit the zero of the operand's family in the SOURCE
role"* and needs no ``isinstance`` to do it, because the leaves know their
own partner. On the :ref:`(Representation × Role) grid <carrier-grid-census>`
this is the **vertical edge** — flux :math:`\leftrightarrow` source/sink on
one carrier — declared once per pair rather than re-parsed at each operator
that crosses it.

The declaration is a class-statement keyword on the *source/sink* half::

    class AngularSourceSink(AngularField, flux=AngularFlux): ...

and :meth:`!__init_subclass__` registers **both** directions into one
shared mapping, so the map is a bijection *by construction*
(``coding-elegance`` Pattern 4 — the illegal state is unspellable, not
validated): a second source/sink naming an already-partnered flux is a
:class:`TypeError` at **import** time, and neither half can be re-pointed
afterwards. The source/sink side declares because the dependency already
runs that way — the source/sink leaves import the flux leaves for their
named compositions (``from_isotropic``, ``from_balance``) and the flux
leaves import no source/sink — and :mod:`orpheus.transport.fields`
completes the registration with a bare ``import`` of
:mod:`orpheus.transport.source_sinks` at its tail.

Three consumers, and the third is the one that removes the parses:

* ``role_partner(role)`` — the leaf CLASS playing ``role`` on this leaf's
  carrier. Asking for a leaf's **own** role returns that leaf (the identity
  half), so an operator can name its output role without branching on its
  input's.
* ``role()`` — which half a leaf is.
* ``into_role(role, values, space=…)`` — *"same space, same family fields,
  the other role's class"*, spelled once. The family fields (``L``,
  ``spatial_moments``, …) ride across via :func:`dataclasses.fields`, so a
  new family field is carried without this verb learning its name. `[M]`
  this is the parse the step-5 census counted **12 times** across three
  verbs.

`[M]` 2026-09-04, measured at runtime over the loaded
:mod:`orpheus.transport.fields` + :mod:`orpheus.transport.source_sinks` +
:mod:`orpheus.transport.residuals` (30 classes on the mixin): **7 declared
pairs**, and **16** carriers with no partner.

.. list-table:: The 7 role pairs (flux ↔ source/sink), enumerated
   :header-rows: 1
   :widths: 50 50

   * - Flux leaf
     - Source/sink leaf
   * - ``AngularFlux``
     - ``AngularSourceSink``
   * - ``ScalarFlux``
     - ``ScalarSourceSink``
   * - ``HarmonicMomentFlux``
     - ``HarmonicMomentSourceSink``
   * - ``AngularBoundaryFlux``
     - ``AngularBoundarySourceSink``
   * - ``ScalarBoundaryFlux``
     - ``ScalarBoundarySourceSink``
   * - ``RadialCharacteristicInteriorFlux``
     - ``RadialCharacteristicInteriorSourceSink``
   * - ``RadialCharacteristicBoundaryFlux``
     - ``RadialCharacteristicBoundarySourceSink``

⚠ **Seven, not five.** The design round enumerated the pairs the operator
tier was known to cross and counted **five**; the roster is a property of
the *carrier* package, not of the operators that traverse it, and the two
ψ½ (radial-characteristic) pairs are real members that no operator in this
step touches. This is the finite-roster rule (``vv-principles`` #31): when
the population is an enumerable shipped set, enumerate it — a ladder or a
"the ones we use" list is a sample wearing a roster's clothes.

The **16 unpaired** carriers are exactly the classes for which a partner
is meaningless, and asking raises a :class:`TypeError` naming them:
the 10 family/locus ABCs (``BulkField``, ``FaceField``, ``AngularField``,
``ScalarField``, ``MomentField``, ``BoundaryField``, and the four typed
boundary/interior bases), the **5 residual leaves** (a residual is the
*defect of a balance* — it is neither half of the flux/source pair, which
is the same argument the grid's ``(Moment, Residual)`` hole rests on), and
``CrossSectionField`` (a coefficient, not a state).

Pure-L streaming + the affine collision split
==============================================

The streaming leaf :math:`L` (:class:`~orpheus.sn.operators.streaming.StreamingOperator`)
computes **pure** :math:`\sigma`-free streaming directly: its ``apply``
is the named :math:`\sigma`-free
:meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action`
leaf, the spatial streaming :math:`\Omega\cdot\nabla\psi` plus the
curvilinear angular redistribution, with NO collision diagonal.  The
collision diagonal :math:`C = M[\sigma_t]`
(a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`,
the §5.7 multiplier promotion — now literally :math:`C = M[\sigma_t]`) is
the separate shared leaf, and the composition
:math:`L + C` recovers the full within-group loss.

The discrete within-group WDD matvec is **affine in** :math:`\sigma` in
the forward direction:

.. (vv-status rationale) #257 S8b — the σ-free streaming primitive. The
   intrinsic σ-freedom of pure L (its apply reads no σ) is a SOFTWARE
   invariant pinned by the foundation catcher C1
   (``tests/sn/operators/test_pure_L_sigma_free.py``, with a Mode-11
   σ-leak mutation that reddens C1); the affine relation
   ``M(σ)ψ = streaming_action(ψ) + σ⊙ψ`` and the byte-identical (L+C)
   recovery are pinned by ``test_streaming_operator_decomposition.py``
   and ``test_loss_action_convention.py``. The streaming discretization
   is single-sourced through ``loss_action`` at σ = 0.
.. vv-status: streaming-action-pure-l documented

.. math::
   :label: streaming-action-pure-l

   M(\sigma)\,\psi \;=\; \underbrace{\text{streaming\_action}(\psi)}_{L\,\psi,
       \;\sigma\text{-free}} \;+\; \sigma_t \odot \psi
   \qquad\Longleftrightarrow\qquad
   \text{streaming\_action}(\psi) \;=\; \texttt{loss\_action}(0, \psi)

so :math:`L` reads no :math:`\sigma`: the curvilinear Carlson
coupled-pole seed's :math:`\sigma`-dependence is exactly the collision
diagonal it injects, which cancels into :math:`\sigma_t\odot\psi` and
belongs to :math:`C` (ERR-058 / #195 made the seed σ-independent, which
is what licenses the carve).  The streaming discretization lives ONCE in
``loss_action``; ``streaming_action`` is single-sourced from it at
:math:`\sigma = 0` (``coding-elegance`` Pattern 2), so there is no twin
σ-free walk.

.. implements:: streaming-action-pure-l
   :by: orpheus.sn.loss_representation._LossRepresentation.streaming_action

   **Implemented by** one line — ``return
   self.loss_action(self._zero_sigma_for(psi), psi)`` — which is the
   equation's right-hand half, :math:`\texttt{streaming\_action}(\psi) =
   \texttt{loss\_action}(0, \psi)`, taken as the **definition** of the
   :math:`\sigma`-free leaf rather than as a claim about it. That is what
   makes the pure-:math:`L` walk single-sourced from the full loss walk
   instead of a twin.

   ⛔ Deliberately **not** declared:
   :meth:`StreamingOperator.apply <orpheus.sn.operators.streaming.StreamingOperator.apply>`.
   It is a pure delegation to this method — a consumer of the identity, not
   a second realization of it.

Why the matvec is affine in :math:`\sigma`
------------------------------------------

The discrete within-group cell balance is the single source of the
affine structure. In the geometry-agnostic 1-D scan
(:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.affine_scan_coefficients`)
and in the curvilinear cell update
(:func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`) the
WDD cell-average solves

.. (vv-status rationale) The WDD cell-balance identity S·ψ̄ = source +
   streaming numerator, S = S_stream + σ_t V — the source of the
   σ-affine split. Definitional; the discrete sweep it feeds is verified
   downstream (dd-slab-scalar / dd-curvilinear-scalar).
.. vv-status: streaming-action-cell-balance documented

.. math::
   :label: streaming-action-cell-balance

   S\,\bar\psi \;=\; \underbrace{Q\,V \;+\; \text{(upstream-face inflow)}}_{\text{source} + \text{streaming numerator}},
   \qquad
   S \;=\; \underbrace{S_{\rm stream}}_{\text{geometric}}
       \;+\; \underbrace{\sigma_t\,V}_{\text{collision}},

where the cell-balance diagonal :math:`S` is the **sum** of a purely
geometric streaming term :math:`S_{\rm stream}` and the collision
volume term :math:`\sigma_t\,V`. In the production code
(:func:`~orpheus.transport.spatial.diamond.DiamondDifference._cartesian_streaming_diagonal`)
the Cartesian scan denominator is literally
``denom = reaction_xs + Σ_axes (2 g_axis)`` with ``reaction_xs`` the
collision term and ``2 g_axis = 2|μ_axis|/Δ_axis`` the geometric
streaming term; the curvilinear form
(:func:`~orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients`)
is ``denom = geometric_streaming_term + collision_volume_term`` with
``geometric_streaming_term = 2|μ|·face_area_downstream + (ΔA/w)·c_out``. The
collision cross section enters the diagonal **purely additively** —
:math:`S` is affine in :math:`\sigma_t` with unit slope :math:`V`. That
is exactly :eq:`streaming-action-pure-l`: the forward matvec
:math:`M(\sigma)\psi` is the *application* of this diagonal (not its
inverse), so the collision contribution is the clean additive term
:math:`\sigma_t\odot\psi` that the pure-L leaf simply does not carry.

.. implements:: streaming-action-cell-balance
   :by: orpheus.transport.spatial.cell_balance.cell_balance_for_streaming

   **Implemented by** the curvilinear assembly, which returns
   ``denom, numer_upstream``: the cell-balance diagonal :math:`S` and the
   upstream-face numerator, as one named pair rather than two loose
   arrays.  Since P4.9a (2026-08-28) it is the **only** cell-balance
   assembly: the term-resolved sibling ``cell_balance_terms``, which
   returned ``denom = 2·|μ|·A_downstream + (ΔA/w)·c_out + Σ_t·V`` inside a
   ``CellBalanceTerms`` record so the geometric and collision halves
   stayed individually nameable, was retired onto this function and its
   declaration removed here.  The nameability it bought is preserved
   without the twin — the three summands of :math:`S` are
   :eq:`tensor-network-cell-balance-three-terms`, and the angular one now
   arrives as its own argument.

.. implements:: streaming-action-cell-balance
   :by: orpheus.transport.spatial.diamond.DiamondDifference._cartesian_streaming_diagonal

   **Implemented by** the Cartesian diagonal: ``denom = Σ_t + Σ_a 2 g_a``
   with ``g_a = |μ_a|/Δ_a`` the raw down-face streaming and the ``2`` the
   diamond factor. This is the equation's :math:`S = S_{\rm stream} +
   \sigma_t V` with the collision term entering **purely additively** — the
   unit-slope-in-:math:`\sigma_t` fact that :eq:`streaming-action-pure-l`
   rests on.

.. implements:: streaming-action-cell-balance
   :by: orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients

   **Implemented by** the curvilinear scan's ``denom =
   geometric_streaming_term + collision_volume_term``, with
   ``geometric_streaming_term = streaming_face_term +
   angular_denom_term`` — the same additive split, with the
   angular redistribution :math:`(\Delta A/w)\,c_{\rm out}` folded into the
   geometric half where it belongs.  (The second summand was a locally
   recomputed ``curvature_redistribution_term`` until P4.9a re-posed the
   signature to take one already-assembled ``angular_denom_term``; the
   value and the association order are unchanged, and the method now
   carries no Morel--Montry name.)

.. implements:: streaming-action-cell-balance
   :by: orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch

   **Implemented by** the residual form of the same balance,
   :math:`r = S\bar\psi - \bigl(Q + \sum_a 2 g_a\,\psi_{\rm in}\bigr)` — the
   equation moved to one side. It shares
   ``_cartesian_streaming_diagonal`` with the sites above, which is why it
   belongs to this equation and not to :eq:`apply-solve-cell-resolvent`: it
   never divides.

The subtle part is the **curvilinear angular closure**. For sphere /
cylinder the Carlson coupled-pole seed
(``precompute_psi_state``) builds a half-angle starting flux from the
flat moment :math:`\bar\phi_0` via a recursion whose coefficients —
:math:`\bar Q = \sigma_t\,\bar\phi_0` and a per-pole denominator
:math:`\Delta r\,\sigma_t + 2` — are themselves :math:`\sigma_t`-bearing.
At face value this looks like a :math:`\sigma_t`-dependence that lives
in the *streaming* (angular-redistribution) term, which would break the
clean additive split. It does not. The seed's :math:`\sigma_t`-dependence
is **exactly the collision diagonal it injects** into the half-angle
balance; when the cell update assembles
:math:`m_{\rm full} = (S\,\bar\psi - \text{numerator})/V`, that injected
collision exactly cancels the seed's :math:`\sigma_t`-bearing
contribution, leaving the redistribution term :math:`(1-\mu^2)/r\,
\partial\psi/\partial\mu` (the genuine angular streaming) **independent of
:math:`\sigma_t`**. The net :math:`\sigma_t` that survives is the single
:math:`\sigma_t\,V` collision term in :math:`S` — and nothing else.

ERR-058 / #195 is what *licenses* writing this down as a software
invariant. Before that fix, the curvilinear Carlson seed used a
:math:`\sigma_t`-coupled angular-edge extrapolation that left a residual
:math:`\sigma_t`-dependence in the redistribution; the decomposition
test's top docstring still carried the now-stale claim that
``matvec(σ=0)`` was "3–13 % wrong for curvilinear". ERR-058 replaced the
seed with a :math:`\sigma_t`-independent ``AngularEdgeExtrapolation``,
which made the curvilinear matvec genuinely affine in :math:`\sigma_t` —
and that is the precondition the pure-L carve depends on. (Issue #282
route (a) later retired that strategy *class*, but the
:math:`\sigma_t`-affinity invariant survives — the direct
starting-direction march and the inlined ``edge_extrapolated_seed`` are
both affine in :math:`\sigma_t` exactly like the bulk walk; see
:ref:`sn-direct-seed-solve`.) The lesson
catalogued in :mod:`orpheus.sn.loss_representation` is to probe the live
behaviour rather than trust the prose: the affinity is an *empirical*
fact that must be re-verified, not a transcribed claim.

Probe evidence and the retired fold
------------------------------------

The carve does **not** duplicate the ~400-line discretization walk to
produce a separate :math:`\sigma`-free streaming kernel. ``loss_action``
is monolithic in :math:`\sigma` (the cross section is threaded into the
Cartesian ``residual_kernel_batch``, the curvilinear
``cell_balance_for_streaming``, *and* the Carlson seed
``precompute_psi_state``), so a hand-separated streaming walk would be a
twin path (Cardinal Rule 2 violation). Instead
:meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action`
is **single-sourced** from the same walk at :math:`\sigma = 0`
(``coding-elegance`` Pattern 2 — name the primitive, do not clone the
algebra). Two in-process probes establish that this is value-correct,
not merely convenient:

.. list-table:: Pure-L σ-freedom — measured drift (#257 S8b)
   :header-rows: 1
   :widths: 46 18 18 18

   * - Probe
     - slab / Cartesian
     - sphere
     - cylinder
   * - Affine relation
       :math:`\texttt{loss\_action}(0,\psi) = \texttt{loss\_action}(\sigma,\psi)-\sigma\odot\psi`
       (bulk; boundary strict 0 ULP)
     - ≤ 32 ULP
     - ≤ 2 ULP
     - ≤ 72 ULP
   * - σ-leak test
       :math:`\texttt{streaming\_action}(\,\cdot\,;\,\sigma_a) = \texttt{streaming\_action}(\,\cdot\,;\,\sigma_b)`
       for two **wildly different** :math:`\sigma` fields
     - .. centered:: ≤ 64 ULP (rel ~1e-16), all geometries
     -
     -
   * - Pure-L leaf vs retired ``(L+C)−C`` fold
       (the matvec :math:`L\psi` the carve replaced)
     - ≤ 16 ULP
     - ≤ 16 ULP
     - ≤ 16 ULP

The σ-leak test is the decisive one: applying the streaming leaf to the
same flux under two completely different cross-section fields produces
byte-stable results (relative drift at the floating-point floor), so the
leaf demonstrably reads no :math:`\sigma`. The first row is the
quantitative form of the cancellation argument above — the difference
between the full loss and the pure stream is *exactly*
:math:`\sigma\odot\psi` to the floating-point floor, with the curvilinear
geometries (where the cancellation runs through the angular closure) the
tightest of all (sphere ≤ 2 ULP). The third row records that the carve
re-associates the floating-point reduction tree relative to the retired
``(L+C)−C`` fold (the old way of obtaining :math:`L\psi`: build the full
loss, subtract the collision diagonal) — the values agree to ≤ 16 ULP,
well inside the dimensionally-explainable single-step bound (per
``vv-principles`` § "Bit-identity vs principled-equivalence", criterion 3).

The carve is therefore **principled-equivalent, not bit-identical**, on
the *streaming-leaf* matvec — and **byte-identical** on the
:math:`(L+C)` composite matvec and the WDD sweep, which were not touched
(:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply` still computes
:math:`M(\sigma_t)\psi` through the same ``loss_action`` call). The
software invariant "pure :math:`L` reads no :math:`\sigma`" is pinned by
the foundation catcher
:func:`tests.sn.operators.test_pure_L_sigma_free.test_c1_pure_L_apply_is_sigma_free`,
which carries a Mode-11 σ-leak mutation (monkeypatch a σ-leaking
``streaming_action`` stub) that reddens the gate — so the gate is
verified to be *able* to see a regression, not merely green. The affine
relation and the byte-identical :math:`(L+C)` recovery are pinned by
``test_streaming_operator_decomposition.py`` and
``test_loss_action_convention.py``.

This affine structure is the algebraic foundation of the next subsection:
because the *forward* application is affine in :math:`\sigma` (additive,
distributive), the leaves compose additively — but the *inverse* is not,
which is precisely why ``solve`` cannot live on the leaves.


.. _apply-solve-asymmetry:

apply is linear in the operator; solve is not
=============================================

The single most important structural fact about the
:math:`L+C` algebra is an **asymmetry between the two primitive actions**
of :ref:`Definitions <operator-algebra>`: forward application
(:eq:`operator-apply`) is *linear in the operator*, but inversion
(:eq:`operator-solve`) is *not*. This is why ``apply`` and ``solve`` are
**two faithful views of the same operator only for the bundled**
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` (:math:`= L+C`), and
**never** for the individual streaming / collision leaves. It is the
mathematical content behind the :ref:`three-layer operator surface
<capability-set-semantics>`: ``apply`` lives on the leaves; ``solve``
lives on the bundle.

The asymmetry
-------------

Forward application **distributes over the operator sum**, because
applying a sum of linear maps is the sum of the applications:

.. math::
   :label: apply-distributes

   (L + C)\,\psi \;=\; L\,\psi \;+\; C\,\psi.

.. implements:: apply-distributes
   :by: orpheus.numerics.operator.OperatorSum.apply

   **Implemented by** ``return self.a.apply(x) + self.b.apply(x)`` —
   "applying a sum of linear maps is the sum of the applications" as a body,
   not as a claim.

   ⚠ The declaration deliberately names the **generic** sum, and only it. On
   the very instance this equation is written for, :math:`(L+C)`, the
   shipped path does **not** execute :math:`L\psi + C\psi`:
   :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`
   *overrides* ``apply`` and computes ``loss_representation.loss_action(σ,
   ψ)`` in a single walk. The inherited leaf-sum is value-equal to it only
   through the forward-direction affine-in-:math:`\sigma` coincidence of
   :eq:`streaming-action-pure-l` — so on the composite, distributivity holds
   by **theorem**, not by code path, and extending this declaration to
   ``StreamingCollisionOperator.apply`` would assert a route the tree does
   not take.

Inversion does **not** distribute:

.. math::
   :label: solve-does-not-distribute

   (L + C)^{-1} \;\neq\; L^{-1} \;+\; C^{-1}.


.. no-implementation:: solve-does-not-distribute
   :kind: law

   **Nothing implements this**, because it is enforced by an ABSENCE. The
   forbidden expression is not merely wrong — it is unspellable:
   :math:`[M]` 2026-08-18, neither
   :class:`~orpheus.sn.operators.streaming.StreamingOperator` nor
   :class:`~orpheus.numerics.operator.OperatorSum` has a ``solve`` member
   (not in ``__dict__``, not inherited), so ``L.solve(q) + C.solve(q)``
   raises rather than returning a meaningless answer. What *does* ship is
   the fused :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve`
   — the sweep, i.e. the coupled inverse — which implements
   :eq:`loss-rep-scanmarch-solve-affine`, not this inequality. An absence is
   not an implementation; declaring one here would name the very path the
   law forbids.

The inverse is a :math:`1/x`-shaped functional of the operator, and
:math:`1/x` does not distribute over :math:`+`. The crispest sanity
anchor is the scalar case: :math:`(3+5)^{-1} = 1/8 = 0.125`, whereas
:math:`3^{-1} + 5^{-1} = 0.533`. The two are not equal, not even close.
``L.solve(q) + C.solve(q)`` would be the operator version of
:math:`q/L + q/C`, which is emphatically **not** :math:`q/(L+C)`. This is
the same fact that the :ref:`composition algebra
<composition-algebra>` table encodes as "``OperatorSum`` does not
propagate ``solve``": no general algorithm exists for :math:`(A+B)^{-1}`
from :math:`A^{-1}` and :math:`B^{-1}`.

.. _smw-low-rank-exception:

The lone systematic exception is **low-rank structure**:
Sherman–Morrison–Woodbury rebuilds :math:`(A + U V^{\mathsf T})^{-1}`
from :math:`A^{-1}` at the price of one dense solve of rank size — and
that exception must be scoped honestly **per block**, not read as a
claim about the whole algebra (Issue #299). The dense collision
diagonal :math:`C` and the streaming operator :math:`L` carry no
low-rank split, so SMW buys nothing for the bulk pair. But the
boundary operator :math:`B` **is** exactly that structure: rank-1 per
face for the isotropically re-entering white/albedo laws (one
re-entry mode fed by a cosine-weighted outflow average), and an
ordinate permutation — rank :math:`N/2` on a slab face, trace-sized
rather than bulk-sized — for specular reflection (see the
:ref:`boundary-law census <bc-law-layer>` and
:ref:`bc-rank-n-algebra`). CP already exploits precisely this: its
white-boundary re-entry closes in **closed form** as the rank-1
update :math:`P_\infty = P_{\rm cell} + P_{\rm out} \otimes
P_{\rm in} / (1 - P_{\rm inout})` (``orpheus/cp/solver.py``), and
borrowing the same Woodbury closure for SN's boundary cycle — which
source iteration currently *iterates* — is Issue #300.

What each separate inverse would mean physically
------------------------------------------------

The within-group transport balance is the operator equation
:math:`(L+C)\,\psi = q`, i.e.

.. (vv-status rationale) The within-group transport balance
   Ω·∇ψ + Σ_t ψ = q. Governing equation (definitional).
.. vv-status: apply-solve-within-group-balance documented

.. math::
   :label: apply-solve-within-group-balance

   \Omega\cdot\nabla\psi \;+\; \Sigt{}\,\psi \;=\; q,

.. implements:: apply-solve-within-group-balance
   :by: orpheus.sn.operators.streaming.StreamingCollisionOperator

   **Implemented by** the class that *is* the discretised
   :math:`\hat\Omega\cdot\nabla + \Sigma_t`. Its ``apply`` computes this
   balance's left-hand side and its ``solve`` inverts that same left-hand
   side against :math:`q` by the WDD sweep — the two faithful views the
   asymmetry above says exist **only** for the bundle, never for the
   separate :math:`L` and :math:`C` leaves.

with :math:`L = \Omega\cdot\nabla` the streaming (advection) operator and
:math:`C = M[\Sigt{}]` the collision diagonal. Each *separate* inverse
solves a *different, decoupled* problem:

.. list-table:: The three inverses solve three different problems
   :header-rows: 1
   :widths: 16 30 54

   * - Inverse
     - Equation it solves
     - Physical meaning
   * - :math:`L^{-1}`
     - :math:`\Omega\cdot\nabla\psi = q`
     - **Pure advection, no absorption** — the flux if neutrons streamed
       freely and never collided. A formal inverse only: pure streaming
       is rank-deficient (see below).
   * - :math:`C^{-1}`
     - :math:`\Sigt{}\,\psi = q \;\Rightarrow\; \psi = q/\Sigt{}`
     - **Infinite-medium / no-leakage flux** — purely local, the flux at
       a point set entirely by the local collision rate, with no spatial
       coupling whatsoever.
   * - :math:`(L+C)^{-1}`
     - :math:`\Omega\cdot\nabla\psi + \Sigt{}\,\psi = q`
     - **The coupled balance** — the flux that satisfies *both* loss
       mechanisms simultaneously, everywhere. This is the true transport
       solution, and it is **not** the sum of the two decoupled problems.

The true solution :math:`(L+C)^{-1}q` is genuinely coupled: streaming
moves a neutron from cell to cell while collision removes it, and the
two compete at every point. Solving them separately and adding throws
away the competition. That is *why* a Neumann-style series is the only
honest way to express the coupled inverse through the parts (below).

.. note::

   ``L.solve`` is **not a live call** in ORPHEUS. The streaming leaf
   :class:`~orpheus.sn.operators.streaming.StreamingOperator` is
   adjointable but **not invertible** (it exposes ``apply`` and
   ``apply_transpose``, reports
   :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`
   ``= False``, and declares **no** ``inverse()`` / ``solve`` — the
   *structural* arm of the two-kinds split, Design C) — precisely
   because pure streaming is rank-deficient (without a collision term
   the within-group cell balance is singular;
   :eq:`streaming-action-cell-balance` has :math:`\sigma_t\,V = 0`, so
   :math:`S` degenerates to the geometric streaming term alone, which has
   a zero-mode the inflow boundary condition cannot pin in general). The
   :math:`L^{-1}` row above is therefore the **mathematical** advection
   inverse — "even if :math:`L` alone were inverted, it would solve pure
   advection" — not a method you can invoke (asking for it is a *static*
   error, not a runtime raise). The collision leaf
   :math:`C = M[\sigma_t]`
   (a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`)
   *does* report
   :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`
   ``= True`` and carry ``solve``, but **only** when
   :math:`\min|\sigma| > 0` (the multiplier spectrum law
   :math:`\mathrm{spec}(M[\sigma]) = \mathrm{ess\,range}(\sigma)`) — it
   is the *value-dependent* arm: it always declares ``inverse()`` /
   ``solve`` and refuses eagerly with
   :class:`~orpheus.numerics.operator.NotInvertible` on a zero
   coefficient. When invertible, ``C.solve(q) = q/\sigma``
   is the infinite-medium flux of the :math:`C^{-1}` row, computed as an
   element-wise division.

The crispest proof — the WDD cell denominator
---------------------------------------------

The non-separability is visible inside a **single cell**, before any
global coupling enters. The :term:`diamond-difference <diamond difference>` cell update solves the
balance :eq:`streaming-action-cell-balance` for the cell-average flux:

.. (vv-status rationale) The per-cell WDD resolvent
   ψ̄ = (Q V + inflow)/S — the single-cell shadow of the non-distributing
   inverse. Definitional restatement; the sweep it expresses is verified
   downstream via dd-slab-scalar.
.. vv-status: apply-solve-cell-resolvent documented

.. math::
   :label: apply-solve-cell-resolvent

   \bar\psi
   \;=\;
   \frac{Q\,V \;+\; \text{(upstream-face inflow)}}{S},
   \qquad
   S \;=\; S_{\rm stream} \;+\; \sigma_t\,V,

so the resolvent (the per-cell ``solve``) **divides by the SUM**
:math:`S = S_{\rm stream} + \sigma_t\,V`. In the production code this is
literally ``inverse_denom = 1.0 / denom`` with ``denom = streaming_term
+ collision_term``
(:func:`~orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients`,
:func:`~orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients`).
Now compare the three inverses on this single cell:

.. implements:: apply-solve-cell-resolvent
   :by: orpheus.transport.spatial.diamond.DiamondDifference.update

   **Implemented by** the per-visit cell resolvent, ``psi_avg = (source
   + terms.numer_upstream) / terms.denom`` — the division by the *summed*
   denominator, one cell at a time.

   The discriminator against :eq:`streaming-action-cell-balance` is which
   half of the fraction a site owns: that equation is the **assembly** of
   :math:`S`, this one is the **division** by it.

.. implements:: apply-solve-cell-resolvent
   :by: orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch

   **Implemented by** the vectorised form of the same division —
   ``numer / denom`` over a whole anti-hyperplane level at once, with
   ``denom`` from the shared ``_cartesian_streaming_diagonal``.

.. implements:: apply-solve-cell-resolvent
   :by: orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients

   **Implemented by** the curvilinear scan form, which materialises the
   reciprocal once — ``inverse_denom = 1.0 / denom`` — so the whole scan
   multiplies by :math:`1/S` instead of dividing repeatedly.

.. implements:: apply-solve-cell-resolvent
   :by: orpheus.transport.spatial.diamond.DiamondDifference.cartesian_scan_coefficients

   **Implemented by** the Cartesian row-march scan form, the same
   ``inverse_denom = 1.0 / denom`` on the axis-aligned denominator (the
   ``×inverse_denom`` convention, deliberately not the legacy
   ``÷S``).

- :math:`L^{-1}` would divide by :math:`S_{\rm stream}` **alone**
  (set :math:`\sigma_t = 0`) — the pure-streaming denominator.
- :math:`C^{-1}` would divide by :math:`\sigma_t\,V` **alone** (no
  upstream-face coupling at all) — the local denominator.
- The coupled resolvent divides by :math:`S_{\rm stream} + \sigma_t\,V`,
  and

  .. (vv-status rationale) The single-cell non-separability
     1/(S_stream + σ_t V) ≠ 1/S_stream + 1/(σ_t V). Mathematical
     identity, matching the sentineled apply-distributes /
     solve-does-not-distribute siblings.
  .. vv-status: apply-solve-denominator-inequality documented

  .. math::
     :label: apply-solve-denominator-inequality

     \frac{1}{S_{\rm stream} + \sigma_t\,V}
     \;\neq\;
     \frac{1}{S_{\rm stream}} \;+\; \frac{1}{\sigma_t\,V}.


  .. no-implementation:: apply-solve-denominator-inequality
     :kind: identity

     **Nothing implements this** — it is the single-cell shadow of
     :eq:`solve-does-not-distribute`, and the page's own vv-status rationale
     calls it *"Mathematical identity"*. Unlike its global sibling this one is
     not enforced by an absence — both leaf denominators are perfectly
     spellable — which is exactly why it is an identity and the sibling is a
     law.

The two losses are **added before the division** (the additive structure
of :eq:`streaming-action-pure-l`); you must invert the *sum*. This is the
single-cell shadow of the global inequality
:eq:`solve-does-not-distribute`. The forward matvec, by contrast,
*multiplies* by the cell-balance diagonal :math:`S` — and multiplication
by a sum distributes (:eq:`apply-distributes`), which is exactly why
``apply`` survives on the leaves while ``solve`` does not.

The Neumann series — the only honest way through the parts
----------------------------------------------------------

If one insists on expressing :math:`(L+C)^{-1}` through the individual
inverses, the only correct expression is an **infinite operator-splitting
(Neumann) series**, not a finite sum. Splitting around the collision
diagonal :math:`C` (which is the cheap-to-invert leaf, ``C.solve(q) =
q/\sigma``):

.. (vv-status rationale) The operator-splitting (Neumann) series for
   (L+C)⁻¹ around the collision diagonal C. Mathematical identity
   (Lewis & Miller §3.2).
.. vv-status: apply-solve-neumann-series documented

.. math::
   :label: apply-solve-neumann-series

   (L+C)^{-1}
   \;=\;
   \bigl[C\,(I + C^{-1}L)\bigr]^{-1}
   \;=\;
   (I + C^{-1}L)^{-1}\,C^{-1}
   \;=\;
   \sum_{k=0}^{\infty} (-1)^{k}\,(C^{-1}L)^{k}\,C^{-1},


.. no-implementation:: apply-solve-neumann-series
   :kind: identity

   **Nothing implements this**: the series is exhibited to show what
   production does NOT do. Splitting around :math:`C` is never run — the
   sweep inverts the coupled operator directly. The page's own vv-status
   rationale: *"Mathematical identity (Lewis & Miller §3.2)."* The
   transport-native Neumann series that IS run is the source iteration
   around :math:`S`, which is a different equation on this page.

i.e.

.. (vv-status rationale) The term-by-term Neumann expansion
   C⁻¹ − C⁻¹LC⁻¹ + …. Mathematical identity (the expanded form of
   apply-solve-neumann-series).
.. vv-status: apply-solve-neumann-expansion documented

.. math::
   :label: apply-solve-neumann-expansion

   (L+C)^{-1}
   \;=\;
   C^{-1} \;-\; C^{-1}L\,C^{-1} \;+\; C^{-1}L\,C^{-1}L\,C^{-1} \;-\;\cdots,


.. no-implementation:: apply-solve-neumann-expansion
   :kind: identity

   **Nothing implements this** — it is :eq:`apply-solve-neumann-series`
   written out term by term, and the page's own vv-status rationale says
   *"Mathematical identity (the expanded form of
   apply-solve-neumann-series)."* No production path forms
   :math:`C^{-1}LC^{-1}`.

which converges when the spectral radius
:math:`\rho(C^{-1}L) < 1`. The leading term :math:`C^{-1}` is the
infinite-medium flux; every subsequent term is a streaming correction.
A *finite* truncation — and in particular the one-term sum
:math:`L^{-1} + C^{-1}` — is **never** the coupled inverse. The closest
clean closed form involving both inverses is the **parallel** (resistors
-in-parallel) identity

.. (vv-status rationale) The parallel / harmonic identity
   (L⁻¹ + C⁻¹)⁻¹ = L(L+C)⁻¹C. Mathematical identity (still not the
   coupled inverse).
.. vv-status: apply-solve-parallel-identity documented

.. math::
   :label: apply-solve-parallel-identity

   \bigl(L^{-1} + C^{-1}\bigr)^{-1}
   \;=\;
   L\,(L+C)^{-1}\,C,


.. no-implementation:: apply-solve-parallel-identity
   :kind: identity

   **Nothing implements this**, and it is here precisely because it is
   *not* the coupled inverse. The page's own vv-status rationale:
   *"Mathematical identity (still not the coupled inverse)."* It is shown so
   that the harmonic combination cannot be mistaken for
   :math:`(L+C)^{-1}`.

which is still **not** :math:`(L+C)^{-1}` — it is the harmonic
combination, related to but distinct from the coupled inverse.

This is more than an algebraic curiosity: the transport-native Neumann
series is the **source-iteration / collision-number expansion** itself.
The full within-group-plus-scattering problem
:math:`(L+C-S)\psi = q` is solved by splitting off the scattering source
:math:`S` and summing the series

.. (vv-status rationale) The source-iteration / Peierls
   collision-number series ψ = Σ_k [(L+C)⁻¹S]^k (L+C)⁻¹ q. Mathematical
   identity; its ρ = c convergence (green-scattering-ratio-bound) and the
   SI fixed point are exercised downstream by the source-iteration suites.
.. vv-status: apply-solve-source-iteration-series documented

.. math::
   :label: apply-solve-source-iteration-series

   \psi
   \;=\;
   \sum_{k=0}^{\infty}
     \bigl[(L+C)^{-1}S\bigr]^{k}\,(L+C)^{-1}\,q,

where the **sweep** :math:`(L+C)^{-1}` is the per-term inverter and the
outer source iteration sums the series. This is the Peierls
collision-number expansion (each term :math:`k` is the flux of neutrons
that have scattered exactly :math:`k` times). The series converges with
spectral radius :math:`\rho\bigl[(L+C)^{-1}S\bigr] \le \max_g
\Sigma_{s,g}/\Sigma_{t,g} = c` (the :term:`scattering ratio`;
:ref:`cone-iterate-diagnostics` documents the matching contraction
:math:`M = (L+C)^{-1}(S+B)`, whose measured factor :math:`\rho` is
carried as a derived diagnostic on the SI iteration record). The sweep :math:`(L+C)^{-1}` being a *single bundled* inverse —
not :math:`L^{-1} + C^{-1}` — is exactly the point: it is the WDD
forward-substitution on :eq:`apply-solve-cell-resolvent`, dividing by the
summed denominator cell-by-cell in inflow-to-outflow order. See Lewis &
Miller, *Computational Methods of Neutron Transport* (:cite:`LewisMiller1984`,
§3.2 for the sweep as the discrete-ordinates resolvent and §4 for the
source-iteration / Neumann scattering series), and Adams & Larsen 2002
(:cite:`AdamsLarsen2002`, §II for the spectral radius :math:`\rho = c`).

.. implements:: apply-solve-source-iteration-series
   :by: orpheus.numerics.iteration.SourceIteration.solve

   **Implemented by** the loop, whose iterates *are* the partial sums.
   Each pass builds ``rhs = q_ext`` then ``rhs = rhs + g.apply(psi)`` over
   the gains, and applies ``psi = self.A_inv.apply(rhs, ...)`` — one more
   factor of :math:`(L+C)^{-1}S` on top of everything accumulated, so
   iterate :math:`k` is the flux of neutrons that have scattered exactly
   :math:`k` times. The driver never sees :math:`L` or :math:`C`
   separately; it sees one bundled inverse, which is the architectural point
   of the series.

Why this is the right architecture, not a limitation
----------------------------------------------------

Invertibility is a property of the **sum**, not of the parts. That is
exactly why :math:`L + C` is packaged as one
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`: the
:class:`~orpheus.numerics.operator.OperatorSum` that *carries* the
WDD sweep as its ``.solve``. The asymmetry maps cleanly onto the two
sides of the algebra:

- **apply lives on the faithful separate leaves.** Pure streaming
  :math:`L` (:class:`~orpheus.sn.operators.streaming.StreamingOperator`, the
  :math:`\sigma`-free :eq:`streaming-action-pure-l` leaf) and collision
  :math:`C = M[\sigma_t]`
  (a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`)
  each advertise
  ``apply``, and their applications compose additively
  (:eq:`apply-distributes`). The forward direction is affine in
  :math:`\sigma` (the previous subsection), so the leaf decomposition is
  *faithful*: :math:`(L+C)\psi = L\psi + C\psi` holds exactly.
- **solve belongs to the bundled unit.** Only
  :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` reports
  :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`
  ``= True`` and carries a direct-sweep ``solve``; the leaves do not
  (streaming has no ``solve`` at all; collision's ``solve`` is the
  *local* :math:`q/\sigma`, which is the
  :math:`C^{-1}` of a *different* problem, never the coupled inverse).
  The :class:`~orpheus.numerics.operator.OperatorSum` deliberately
  **does not propagate** ``solve`` (:ref:`composition-algebra`); the
  :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` *adds it back* via the
  SN-specific algebraic identity "WDD sweep :math:`\approx (L+C)^{-1}`"
  (:cite:`LewisMiller1984` §3.2). The composite owns ``apply``, ``solve``, and
  ``apply_transpose`` as three actions of **one** operator on a single
  shared
  :class:`~orpheus.sn.loss_representation.LossRepresentation` (L21 —
  "matvec ≡ sweep"); :meth:`StreamingCollisionOperator.apply` and
  :meth:`StreamingCollisionOperator.solve` single-source :math:`\sigma` from the
  collision diagonal, so they cannot disagree on which loss they invert.

The :ref:`three-layer operator surface <capability-set-semantics>` is
what makes this architecture *enforced* rather than merely *intended*: a
downstream Krylov consumer that asks for the inverse of a bare
:class:`~orpheus.sn.operators.streaming.StreamingOperator` cannot even
spell ``L.inverse()`` (the streaming leaf declares no such method — a
*static* error), and a generic sum that has not been promoted to
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` returns the
iterative splitting :class:`~orpheus.numerics.green_operator.GreenOperator`
rather than the direct sweep — never silently handed
:math:`L^{-1} + C^{-1}` (a meaningless answer to a problem nobody
posed). The asymmetry between :eq:`apply-distributes` and
:eq:`solve-does-not-distribute` is, in this sense, the *reason
invertibility is a per-instance predicate on the operator, not a flag in
a parallel string registry*.

.. vv-status: apply-distributes documented
.. vv-status: solve-does-not-distribute documented


.. _capability-set-semantics:

The three-layer operator surface
================================

Many transport operators have **no** efficient inverse action, and this
is a load-bearing fact, not an inconvenience:

- The scattering source operator :math:`S` has rank in the thousands
  of unknowns and is never inverted directly (the source iteration
  scheme exists precisely to avoid that inversion).
- The fission source operator :math:`F` is rank-deficient — it
  projects onto the fission spectrum :math:`\chi`. There is no
  inverse.
- A Jacobi-preconditioned matvec wrapped for BiCGSTAB has ``apply``
  but no ``solve`` (the solve is the iterative scheme itself, not
  this operator).

Subclassing or abstract methods would force these classes to provide
``solve`` stubs that raise :class:`NotImplementedError`. That is the
**harmful-stub anti-pattern**: downstream Krylov consumers would only
discover the failure mid-iteration. The honest surface for a missing
ability is **method absence**, not an advertising flag — but *how* that
honesty is enforced is the subject of this section.

.. note:: **The retired capability frozenset (#226 taxonomy step 6).**

   Through step 5 the advertisement was a stringly-typed
   ``capabilities: frozenset[str]`` class property listing the ``CAP_APPLY``
   / ``CAP_SOLVE`` / ``CAP_APPLY_TRANSPOSE`` tags an operator supported,
   plus a ``MissingCapability`` exception raised when a composition asked
   for an absent tag. **Carve P4 retired it from every operator** — leaves,
   composers, aggregators, and shims. The frozenset was a **parallel
   registry that could silently drift from the actual method surface**: a
   class could advertise ``CAP_SOLVE`` yet have a broken ``solve`` (or, as
   the collision leaf once did, a ``solve`` that produced a silent NaN), and
   nothing forced the string set to track the code. The replacement makes
   the **surface itself the single source of truth**: an ability exists
   exactly where its method exists, and a runtime predicate reads the live
   structure and values rather than a cached string. The design below is
   the user-locked "Design C + B" (2026-07-02).

The three layers, per axis
--------------------------

Each optional axis — **inverse** and **adjoint** — now has exactly three
layers, and each layer carries the truth it *alone* can express:

.. list-table::
   :header-rows: 1
   :widths: 20 34 46

   * - Layer
     - Inverse axis
     - Adjoint axis
   * - **Predicate** (runtime truth)
     - :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`
     - :attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable`
   * - **Operator-returning method**
     - ``inverse()`` — declared *per-class*
     - :attr:`~orpheus.numerics.operator.LinearOperator.H` — hosted on the base
   * - **Realization verb**
     - ``solve``
     - ``apply_transpose``

- The **predicate** is the runtime, instance-accurate truth. Unlike an
  ``isinstance`` check — which sees only class-level method presence — it
  reads the operator's actual structure AND values: a zero-coefficient
  :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
  reports ``is_invertible = False``; a sum reports its *leading* term
  (the left-spine head — :ref:`green-operator`); a composite derives its
  value recursively from the operands, never a cached registry. The
  default is ``False`` — an operator is invertible or adjointable only by
  explicit override.
- The **operator-returning method** is the canonical act: it returns an
  *operator*, not a vector. ``inverse()`` returns the inverse operator
  (a member of the inverse family — the sweep, the
  :class:`~orpheus.numerics.green_operator.GreenOperator`, the
  :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`);
  :attr:`~orpheus.numerics.operator.LinearOperator.H` returns the
  metric-correct Hilbert adjoint wrapper.
- The **realization verb** is the raw numerical act — ``solve`` maps a
  right-hand side to a solution vector, ``apply_transpose`` applies the
  Euclidean transpose — present **exactly where a native realization
  exists**, never as an exists-but-raises stub. The wrap-delegate
  inverse family delegates through ``solve``; the composer transpose
  laws recurse through ``apply_transpose``.

.. _design-c-structural-value-split:

Design C — the structural-vs-value split
----------------------------------------

The load-bearing insight is that **two mathematically distinct kinds of
non-invertibility deserve two honest surfaces**, and conflating them was
a false dichotomy (see :ref:`design-c-false-dichotomy` below). The two
kinds:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Kind
     - The claim
     - Surface
   * - **Structural**
     - *This TYPE has no inverse* — the map is mathematically
       non-invertible for every instance.
     - ``inverse()`` is **not declared at all**. Asking for it is a
       *static* error (pyright reports ``reportAttributeAccessIssue`` at
       the call site); at runtime the attribute is simply absent.
   * - **Value-dependent**
     - *This TYPE supports inversion, this INSTANCE refuses* — a
       zero-coefficient diagonal, a sum with a non-invertible leading
       term, a product with a singular factor.
     - ``inverse()`` **is declared** and raises
       :class:`~orpheus.numerics.operator.NotInvertible` **eagerly**, at
       construction of the inverse and never mid-iteration.

The structural leaves are
:class:`~orpheus.numerics.operator.ZeroOperator`, the incoming-ordinate
and periodic masks, the source dyads
(:class:`~orpheus.numerics.operator.RankOneOperator`), and the
transport source leaves —
:class:`~orpheus.sn.operators.streaming.StreamingOperator` (:math:`L`),
:class:`~orpheus.transport.operators.scattering.ScatteringOperator`
(:math:`S`),
:class:`~orpheus.transport.operators.fission.FissionOperator`
(:math:`F`), and the boundary operator
:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator` (:math:`B`).
For each, ``op.inverse()`` does not type-check — the absence *is* the
honest surface, and the compiler enforces it.

The value-dependent operators are
:class:`~orpheus.numerics.operator.DiagonalOperator`,
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`,
and the composers when their invertibility fails a value test
(:class:`~orpheus.numerics.operator.OperatorSum` with a non-invertible
head, :class:`~orpheus.numerics.operator.OperatorProduct` with a
singular factor,
:class:`~orpheus.numerics.operator.ScaledOperator` of a non-invertible
operand). They *always* declare ``inverse()`` (the type *can* invert),
and refuse loudly and eagerly when the specific values do not permit it.

.. _typeguard-bridge:

The checked bridge — a PEP-647 ``TypeGuard``
--------------------------------------------

The predicate is a *runtime* fact; ``inverse()`` living only on some
classes is a *static* fact. The bridge that converts one into the other
is a pair of free functions,
:func:`~orpheus.numerics.operator.invertible` and
:func:`~orpheus.numerics.operator.adjointable`, each a PEP-647
``TypeGuard`` narrowing a
:class:`~orpheus.numerics.operator.LinearOperator` to the
:class:`~orpheus.numerics.operator.SupportsInverse` /
:class:`~orpheus.numerics.operator.SupportsAdjoint` narrowing target:

.. code-block:: python

   if invertible(op):          # runtime check on op.is_invertible
       y = op.inverse().apply(b)   # pyright now permits .inverse() — no cast

**The runtime check IS the static permission.** You cannot obtain the
permission without executing the check, and deleting a guard un-narrows
the call so CLI pyright REDs — the guards are *type-load-bearing* (this
is verified as a static tooth: spec §39.1). This finally fulfils the
original charter of the
:class:`~orpheus.numerics.operator.SupportsInverse` /
:class:`~orpheus.numerics.operator.SupportsAdjoint` Protocols (a static
contract backed by a runtime property) through a *checked* bridge — the
carve deleted all four ``cast(SupportsInverse, …)`` sites and all ten
``solve`` / ``apply_transpose`` ``# type: ignore`` comments they used to
require.

Two subtleties fix the exact construct:

- **``TypeGuard``, deliberately NOT ``TypeIs``.** The predicate is
  value-dependent: a zero-coefficient multiplier structurally *has*
  ``inverse()`` while reporting ``is_invertible = False``. Only the
  *one-directional* promise is honest — ``True`` licenses the call;
  ``False`` makes no static claim (a ``TypeIs`` would wrongly narrow the
  negative branch too, asserting the operand is *not* a
  ``SupportsInverse`` when structurally it may still be one).
- **A free function, not a method.** PEP 647 narrowing applies only
  through a call expression, and a method form narrows its first
  *explicit* argument, never ``self`` — so there is no
  ``op.is_invertible``-style property spelling that could narrow the
  operand. The narrowing target
  :class:`~orpheus.numerics.operator.SupportsInverse` was *promoted* to
  **extend** :class:`~orpheus.numerics.operator.LinearOperator`, so a
  guarded branch keeps the whole algebra (``apply``,
  :attr:`~orpheus.numerics.operator.LinearOperator.H`, the composition
  dunders) alongside the licensed ``inverse()``.

.. _base-hosting-rule:

The base-hosting rule
---------------------

**A method lives on the base** :class:`~orpheus.numerics.operator.LinearOperator`
**iff a universal realization exists.**

- :attr:`~orpheus.numerics.operator.LinearOperator.H` **is base-hosted**:
  the Hilbert adjoint has one generic realization — the
  ``AdjointOperator`` wrapper that applies the metric once and delegates
  to ``apply_transpose`` — valid for *any* adjointable operator, so
  ``.H`` is defined once on the base with an eager
  :class:`~orpheus.numerics.operator.MissingAdjoint` gate.
- ``inverse()``, ``solve``, and ``apply_transpose`` are **NOT
  base-hosted**: there is no universal inverse (each structure inverts
  differently — a diagonal by reciprocal, a triangular sweep by
  forward-substitution, a full loss by a Neumann splitting), no universal
  transpose realization, and no universal solve. Each lives per-class,
  exactly where its realization exists.

This is why the structural non-invertibility surface *works*: because
``inverse()`` is not on the base, a class that omits it genuinely has no
such attribute, and ``ZeroOperator().inverse()`` is a static error
rather than a stub that raises.

.. _design-b-native-solve:

Design B — ``solve`` pruned to native realizations
--------------------------------------------------

The realization verb ``solve`` is now present **only where a native
realization exists** — never as an exists-but-raises stub, and never
duplicating what ``.inverse().apply`` already does. This executes the
"one public surface = predicate + operator-returning method" ruling
(taxonomy §11): *solving with an operator IS applying its inverse
object*, ``A.inverse().apply(b)``.

.. (V&V scope note) The inverse-as-operator keystone (#226 Phase 2):
   applying an operator's inverse OBJECT equals invoking its native
   realization verb ``solve``. A foundation software invariant (no
   eigenvalue / flux claim); the label is wired to the bit-identity
   gate ``tests/sn/operators/test_inverse_operator_equivalence.py``
   (``(L+C).inverse().apply(b) == (L+C).solve(b)`` for the sweep-invertible
   loss operator, plus the seed-drop and returned-surface-type checks).

.. math::
   :label: inverse-as-operator

   A^{-1} b \;=\; \texttt{A.inverse().apply(b)} \;=\; \texttt{A.solve(b)}

.. implements:: inverse-as-operator
   :by: orpheus.numerics.operator.InverseOperator.apply

   **Implemented by** the keystone as a body rather than a claim:
   ``return self.inner.solve(x)``. There is no separate inverse machinery to
   drift — applying the inverse *object* runs the operator's own ``solve``,
   by construction.

.. implements:: inverse-as-operator
   :by: orpheus.sn.operators.sweep_operator.SweepOperator.apply

   **Implemented by** the same delegation on the concrete
   :math:`(L+C)` instance this equation is written for: ``return
   self.inner.solve(rhs)``, i.e. the WDD sweep. Reached through the inverse
   object or through ``solve``, the sweep is bit-identically one call.

   ⛔ Deliberately **not** declared:
   :meth:`GreenOperator.apply <orpheus.numerics.green_operator.GreenOperator.apply>`.
   Its forward :class:`~orpheus.numerics.operator.OperatorSum` carries no
   ``solve`` at all, so it is precisely the case this identity does **not**
   cover — a driver-realized inverse *action*, not a native inverse.

For the sweep-invertible loss operator :math:`(L+C)` the native ``solve``
IS the WDD sweep, so this keystone reads: applying the inverse object runs
the same sweep the operator's own ``solve`` runs — no separate inverse
machinery.

**Deleted** (the algebra-closed and driver-realized kinds):

- :class:`~orpheus.numerics.operator.OperatorSum` — a generic sum's
  inverse action is driver-realized by the
  :class:`~orpheus.numerics.green_operator.GreenOperator` (the
  preconditioned splitting), not a substrate verb. (The
  sweep-invertible ``(L+C)``
  :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` subclass
  keeps its own direct-sweep ``solve`` — that IS a native realization.)
- :class:`~orpheus.numerics.operator.IdentityOperator`,
  :class:`~orpheus.numerics.operator.PermutationOperator`,
  :class:`~orpheus.numerics.operator.ScaledOperator`,
  :class:`~orpheus.numerics.operator.TensorProductOperator` — the
  **algebra-closed** kinds, whose inverse is itself a first-class
  *forward* operator (a permutation's inverse is a permutation, a
  scaling's is a scaling): solving is just ``.inverse().apply``.
- the reflective boundary shim's forward.

**Kept** (the native-realization face, what the wrap-delegate inverse
family wraps): :class:`~orpheus.numerics.operator.DiagonalOperator`
(with its value guard, now raising
:class:`~orpheus.numerics.operator.NotInvertible`),
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`,
the sweep composites, the mixin's un-invert, and
:class:`~orpheus.numerics.operator.OperatorProduct` — whose ``solve``
**re-routes** through each factor's canonical surface:

.. (vv-status rationale) Representational identity — the factor-wise
   ``OperatorProduct.solve`` re-route (Design B), NOT a solver claim
   (no eigenvalue / flux). The verifiable content is the §40 re-route
   gate battery in ``tests/numerics/test_operator.py`` (the dense
   ``np.linalg.solve`` anchor + the five-row factor-kind matrix vs the
   pre-carve baseline + the Mode-11 execution sentinel) and the §13 I2
   functoriality gate ``(AB)^{-1} = B^{-1}A^{-1}``.
.. vv-status: product-solve-reroute documented

.. math::
   :label: product-solve-reroute

   (A\,B)^{-1} b
   \;=\;
   B^{-1}\bigl(A^{-1} b\bigr)
   \;=\;
   \texttt{self.b.inverse().apply(self.a.inverse().apply(b))} .

.. implements:: product-solve-reroute
   :by: orpheus.numerics.operator.OperatorProduct.solve

   **Implemented by** the equation transcribed: ``return
   self.b.inverse().apply(self.a.inverse().apply(b_vec))``. The reversal of
   factor order *is* the whole content, and it is executed rather than
   asserted. Note the recursion goes through each factor's canonical
   ``.inverse().apply`` surface and not through a factor ``solve`` — which
   is what makes the re-route total over the kinds whose own ``solve``
   Design B retired.

The re-route is **bit-identical per factor kind** — each inverse object
delegates to the same realization the factor's own ``solve`` used to —
AND **total over the solve-retired kinds**: a permutation / scaling /
Green-invertible-sum factor, whose own ``solve`` Design B deleted, now
composes through its ``.inverse().apply`` (a permutation's inverse is a
first-class forward; a sum-in-a-product works via its
:class:`~orpheus.numerics.green_operator.GreenOperator`). The re-route is
gated at spec §40 by a structurally-independent dense
``np.linalg.solve(as_matrix, q)`` anchor plus a five-row factor-kind
matrix against a pre-carve baseline snapshot — with a Mode-11
counter-sentinel proving ``OperatorProduct.solve`` actually *executes*
under ``(A@B).inverse().apply`` (a value gate spelled ``b.inverse().apply
(a.inverse().apply(q))`` on both sides would be tautological).

The exception successors
------------------------

``MissingCapability`` split into **two** ``TypeError`` successors, one
per axis:

- :class:`~orpheus.numerics.operator.NotInvertible` — the inverse-axis
  refusal, raised eagerly by ``inverse()`` overrides (the
  value-dependent arm).
- :class:`~orpheus.numerics.operator.MissingAdjoint` — the adjoint-axis
  refusal, raised eagerly by
  :meth:`~orpheus.numerics.operator.LinearOperator.adjoint` /
  :attr:`~orpheus.numerics.operator.LinearOperator.H` and by the
  composer ``apply_transpose`` law bodies.

Both parent to :class:`TypeError`, carrying the retired
``MissingCapability``'s public contract forward: **no ``except`` clause
written against the old gate changes meaning**. (The migration was
staged: at wave W1 ``NotInvertible`` was born as a ``MissingCapability``
subclass so every landed ``pytest.raises(MissingCapability)`` stayed
green by inheritance while the new keystone went live; wave W2
re-parented both to ``TypeError`` and deleted the old class.)

.. _eager-adjoint-behavior-change:

The one behavior change — eager ``.H``
--------------------------------------

The carve is otherwise behavior-preserving; it has **exactly one**
observable behavior change (spec §38). Before, ``A.H`` on a
non-adjointable ``A`` *succeeded* — it constructed the wrapper
unconditionally — and the refusal was **lazy**, deferred to the
wrapper's first ``.apply``. Now
:meth:`~orpheus.numerics.operator.LinearOperator.adjoint` raises
:class:`~orpheus.numerics.operator.MissingAdjoint` **eagerly at
construction**:

.. code-block:: python

   def adjoint(self):
       if not adjointable(self):
           raise MissingAdjoint(...)   # eager — was lazy at .apply
       return AdjointOperator(self)

A wrapper that could only fail at its first ``.apply`` is precisely the
broken-stub anti-pattern this module refuses; the
:func:`~orpheus.numerics.operator.adjointable` guard doubles as the
static bridge (the wrapper's constructor consumes the narrowed
:class:`~orpheus.numerics.operator.SupportsAdjoint`).

.. _design-c-false-dichotomy:

What was tried and rejected — the per-class-casts vs base-declaration false dichotomy
-------------------------------------------------------------------------------------

The design that Design C *replaced* framed the choice as a dichotomy
between two unappealing options, and the resolution was to recognise the
dichotomy as false (taxonomy §16 record):

- **Option A — declare ``inverse()`` on the base.** Then every call site
  type-checks, but a
  :class:`~orpheus.numerics.operator.ZeroOperator` (which mathematically
  has no inverse) would inherit a method it cannot honour — demoting the
  compiler's ability to catch ``Zero.inverse()`` misuse from a *static
  error* to a *runtime raise*. The honest "this type has no inverse"
  signal is lost.
- **Option B — keep ``inverse()`` per-class and ``cast`` at every call
  site.** Then structural absence is preserved (a static error on
  ``Zero.inverse()``), but every composer body that calls ``op.inverse()``
  on a ``LinearOperator``-typed operand needs a ``cast(SupportsInverse,
  op)`` and a ``# type: ignore`` — an *unchecked* assertion that could
  drift from the runtime truth exactly as the frozenset did.

The false premise was that *structural* absence and *value-dependent*
refusal are the same phenomenon. They are not: the
:class:`~orpheus.numerics.operator.ZeroOperator` case is structural
(Option A's failure), the zero-coefficient
:class:`~orpheus.numerics.operator.DiagonalOperator` case is
value-dependent (a method that *should* exist and refuse). **Design C
splits them** — structural leaves omit the method (Option B's win,
static error), value-dependent operators declare it and raise
``NotInvertible`` (Option A's ergonomics, honest runtime refusal) — and
replaces the *unchecked* ``cast`` (Option B's cost) with the *checked*
``TypeGuard`` bridge, which certifies the narrowing at runtime. The two
false horns dissolve because the two phenomena were never one.

.. note:: **What the static layer can and cannot certify.**

   Do NOT annotate a parameter with
   :class:`~orpheus.numerics.operator.SupportsInverse` to *demand*
   invertibility: the static layer can certify only **spelling** (the
   method exists on the class), never **solvability** (the value-level
   predicate). A zero-coefficient
   :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
   satisfies ``SupportsInverse`` structurally yet is not invertible.
   Guard with :func:`~orpheus.numerics.operator.invertible` at the
   ``LinearOperator``-typed call site instead — and only there, since a
   ``TypeGuard`` **replaces** (does not intersect) the declared type, so
   guarding an already-concrete operand would widen it.


.. _composition-algebra:

Composition algebra
===================

The composers in :mod:`orpheus.numerics.operator` implement the
following closure laws:

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Composer
     - ``apply``
     - ``solve``
     - ``apply_transpose``
   * - :class:`~orpheus.numerics.operator.OperatorSum`
       (:math:`A + B`)
     - Both must have ``apply``.
     - **Does not propagate.** No general algorithm exists for
       :math:`(A + B)^{-1}` from :math:`A^{-1}, B^{-1}`
       (Sherman-Morrison-Woodbury applies only under low-rank
       structure — which the boundary block :math:`B` HAS, unlike
       the bulk :math:`C`, :math:`L`;
       :ref:`the scoped statement <smw-low-rank-exception>`).
     - Both must have ``apply_transpose``;
       :math:`(A + B)^T = A^T + B^T`.
   * - :class:`~orpheus.numerics.operator.OperatorProduct`
       (:math:`A\,B`)
     - Both must have ``apply`` (function composition).
     - Both must have ``solve``, **applied in REVERSE order**:
       :math:`(A\,B)^{-1} = B^{-1}\,A^{-1}`.
     - Both must have ``apply_transpose``, applied in REVERSE order:
       :math:`(A\,B)^T = B^T\,A^T`.
   * - :class:`~orpheus.numerics.operator.ScaledOperator`
       (:math:`\alpha\,L`)
     - Always preserved.
     - Preserved with division:
       :math:`(\alpha L)^{-1} = (1/\alpha)\,L^{-1}`. Zero scalar is
       rejected — use :class:`ZeroOperator` instead.
     - Preserved (scalars commute with transpose).
   * - :class:`~orpheus.numerics.operator.IdentityOperator`
       (:math:`I`)
     - Trivially yes.
     - ``solve`` is the same code path as ``apply``:
       :math:`I^{-1} = I`.
     - Trivially yes: :math:`I^T = I`.
   * - :class:`~orpheus.numerics.operator.ZeroOperator`
       (:math:`0`)
     - Trivially yes (returns ``np.zeros_like(x)``).
     - **Does not propagate.** The zero operator is not invertible.
     - Trivially yes (returns zero).

The closure rules above are advertised by the recursive
:attr:`~orpheus.numerics.operator.LinearOperator.is_invertible` /
:attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable`
predicates and enforced by **eager guards** at composition time (an
``apply`` mismatch raises ``TypeError``, a non-adjointable ``.H`` raises
:class:`~orpheus.numerics.operator.MissingAdjoint`, a value-dependent
``inverse()`` raises :class:`~orpheus.numerics.operator.NotInvertible`),
never mid-iteration. A generic
:class:`~orpheus.numerics.operator.OperatorSum` carries no ``solve``
verb at all (Design B): solving with it IS applying its inverse object,
``.inverse().apply`` (the
:class:`~orpheus.numerics.green_operator.GreenOperator` splitting), so a
downstream consumer that spells :math:`L = A + 0` and asks for its
inverse action cannot silently receive :math:`A^{-1} + 0^{-1}`.

When a composed within-group operator meets the Krylov accelerator's
:mod:`scipy.sparse.linalg` interface, the ORPHEUS↔scipy boundary is
crossed at a **single** site internal to
:class:`~orpheus.numerics.iteration.KrylovAcceleration`: one ravel-aware
adapter ``_as_scipy_linop(carrier_matvec, template, n)`` lifts the flat
scipy vector to the typed carrier (via ``from_flat``), runs the
carrier-space matvec, and ravels the result back — so both the system
matvec and the preconditioner route through one source of truth (#257
S7; the per-operator ``build_transport_linear_operator`` scipy wrappers
it replaced are retired). The system matvec is the named carrier-space
closure ``loss_minus_gains`` — the invertible resolvent minus the lagged
coupling gains, :math:`(L{+}C)\,\psi - \sum_{\rm gains} g\,\psi`, i.e.
the honest full within-group operator :math:`A = L + C - S - B` applied
(the gains are the scattering :math:`S` and the boundary reflection
:math:`B`) — which reads like the operator rather than ravel plumbing.
Because it is expressed purely through the operator algebra, it is the
discretisation-agnostic form a unified cross-solver iteration driver
(SN / MoC / CP / diffusion, Issue #14) consumes without knowing which
transport discretisation produced the triple. A future consumer needing
a standalone operator→scipy adapter (e.g. a DSA preconditioner built as
an operator, Issue #2) should **generalise** ``_as_scipy_linop`` to
accept an ``op.apply`` callable, **not** resurrect the retired flat-only
``as_scipy_linop`` twin.


.. _intrinsic-operator-types:

The intrinsic operator types
============================

The maps of the algebra fall into a three-way partition by what their
action *returns* — the grand report §5.6 **suffix law**. An
**Operator** carries a field to a field: the local multiplication
:math:`C = M[\sigma_t]` (the collision diagonal, §5.7) and the
streaming leaf :math:`L` are its transport instances. A **Kernel** is
the nonlocal refinement — an Operator whose output at a point
*integrates* the input across an axis — realized by the two Boltzmann
emission kernels, scattering :math:`S = R\circ\Lambda\circ M` and
fission :math:`F = |\chi\rangle\langle\nu\Sigma_f|`. A **Functional**
is the disjoint sibling that maps a field to a **scalar** — the
reaction-rate integrals behind the :math:`k`-eigenvalue. The
discriminator between the local Operator and the nonlocal Kernel is
**locality**: multiplication is the diagonal sub-algebra, whose output
at a phase-space point reads the input only there, and the kernels are
everything off-diagonal. The three type sections below develop this
partition in order — the diagonal / multiplication Operator, then the
Functional, then the two integral Kernels — and the full codomain
partition with its type-system table is set out in
:ref:`functional-category`.

.. note::

   **Where the axis collapse pair sits.** Campaign 1 CS4b added a third
   kind of nonlocal map: the **axis marginal** —
   :class:`~orpheus.numerics.operator.AxisRetractionOperator` (fiber
   integration over one named axis) and its
   :class:`~orpheus.numerics.operator.AxisSectionOperator`. By the
   locality discriminator these are **nonlocal**: the retraction's
   output at a point reads the input at every index of the collapsed
   axis. `[M]` they do NOT conform to the
   :class:`~orpheus.transport.operators.integral_kernel_operator.IntegralKernelOperator`
   Protocol (no ``kernel`` member) — deliberately, because the
   "kernel" they would expose is the axis measure itself, which the
   bound spaces already carry, and no consumer wants a second copy of
   it. They are plain
   :class:`~orpheus.numerics.operator.LinearOperator`\ s born bound,
   minted by the SPACE rather than by a mesh or a materials record.
   Their admission is not an operator-algebra question at all — it is
   the collapse doctrine's — which is why the pair is developed on
   :doc:`/theory/foundations/spaces` (:ref:`spaces-collapse-pair`) and
   only pointed at from here.


.. _diagonal-operator:

Diagonal operator on a tagged axis
==================================

The simplest non-trivial operator beyond the composition primitives
is the **diagonal-on-an-axis** operator
:class:`~orpheus.numerics.operator.DiagonalOperator`. For a 1-D
weight vector :math:`w \in \mathbb{R}^N` and target axis ``axis``,
its action on a multi-axis tensor :math:`x` is elementwise
multiplication along ``axis``:

.. (vv-status rationale) Verified by
   ``tests/numerics/test_diagonal_operator.py`` — apply against
   ``np.einsum`` reference on randomised tensors, self-adjointness,
   and round-trip ``apply ↔ solve`` bit-identity.
.. vv-status: diagonal-operator-action documented

.. math::
   :label: diagonal-operator-action

   (D x)_{\ldots,\,n,\,\ldots}
   \;=\; w_n \, x_{\ldots,\,n,\,\ldots}.

.. implements:: diagonal-operator-action
   :by: orpheus.numerics.operator.DiagonalOperator.apply

   **Implemented by** ``return self._broadcast(x_arr.ndim) * x_arr``:
   the tagged axis carries the weights and every other axis broadcasts
   through unchanged. The broadcast helper is what makes the equation's
   ``…, n, …`` index placement **structural** rather than a shape convention
   each caller has to honour.

All other axes broadcast through unchanged. This is the canonical
"diagonal in some basis" operator — the abstraction Grand Report v3
§9 names :math:`W` (``AngularWeightMatrix``) when the basis is the
discrete-ordinate set of an angular cubature, and the same primitive
any method needs for "multiply-by-weights along one axis":

.. list-table:: Cross-method consumers of DiagonalOperator
   :header-rows: 1
   :widths: 24 38 38

   * - Consumer
     - Role of the diagonal
     - Source
   * - **SN** (:math:`Y^* W` projection)
     - :math:`W` = the angular cubature weights
       :math:`w_n`; the operator multiplies the angular axis of
       :math:`\psi`.
     - This work, Wave 1 (the :math:`W` inside the SH frame's
       analysis face ``frame.analysis``).
   * - **MoC**
     - Track-weight diagonal :math:`w_t` on the track axis of an
       angular flux defined per-track.
     - Future MoC consumer.
   * - **CP**
     - Region-volume diagonal :math:`V_i` on the cell axis of a
       collision-probability matrix.
     - Future CP consumer.
   * - **MC**
     - Importance-weighting diagonal on the source / track axis of
       a tally.
     - Future MC consumer.

Self-adjointness is automatic for real-valued weights:
:meth:`apply_transpose` is the same code path as :meth:`apply`.
Invertibility is by-element: if every weight is non-zero, the
operator reports ``is_invertible = True`` and its :meth:`solve`
divides by :math:`w_n` along ``axis``. If any weight is zero,
``is_invertible`` reports ``False`` and ``inverse()`` / ``solve`` raise
:class:`~orpheus.numerics.operator.NotInvertible` **eagerly** (the
value-dependent arm of Design C) — a zero weight has no inverse, and the
harmful-stub anti-pattern the three-layer surface exists to prevent (a
downstream Krylov consumer silently dividing by zero) is caught upfront.

The implementation does NOT eagerly materialise an
:math:`N \times N` diagonal matrix. The action is a single
broadcast-multiply
(``w.reshape((1, ..., -1, ..., 1)) * x``) so memory cost is
:math:`O(N)` regardless of the input tensor's shape. For the SN
angular axis this matters: an :math:`(N, n_x, n_y, n_g)` field with
:math:`N = O(10^2)` and :math:`n_x \cdot n_y \cdot n_g = O(10^7)`
does NOT need a :math:`(10^7, 10^7)` materialised diagonal.


.. _multiplication-operator-promotion:

The multiplication operator — a coefficient field promoted (§5.7)
=================================================================

The grand report §5.7 closes a loop that the rest of the operator
algebra leaves open. Throughout this page, an operator is a
:class:`~orpheus.numerics.operator.LinearOperator` — an opaque
``Generic[V]`` whose ``apply`` carries a flux to a flux, with nothing in
its type that says *what physics it carries*. For the collision
operator that opacity is a missed opportunity: the collision term
:math:`C\psi = \Sigma_t\,\psi` is **nothing but a cross section acting
by pointwise multiplication**. The cross section is not an input to the
operator; the cross section **is** the operator. §5.7 makes that
identity literal: a :class:`~orpheus.transport.fields.cross_section_field.CrossSectionField`
:math:`f` is *promoted* to the multiplication operator :math:`M[f]`, and
:math:`C = M[\sigma_t]` becomes a named instance of that promotion
rather than an anonymous broadcast-multiply buried in the operator's
``apply`` body (the action now lives once, in
:meth:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator.apply`).

The multiplier-algebra embedding
--------------------------------

The promotion is the **multiplier-algebra embedding** of measure theory:
for the Hilbert space :math:`L^2` of square-integrable flux distributions
over phase space, every bounded measurable coefficient
:math:`f \in L^\infty` defines a bounded operator on :math:`L^2` by
pointwise multiplication,

.. (vv-status rationale) Structural / definitional identity — the
   multiplier-algebra embedding M: L^∞ → B(L²). Not a solver claim; the
   verifiable content is the multiplier-algebra law-suite
   (``tests/transport/test_multiplication_operator.py``) which pins the
   discrete realization :eq:`multiplication-operator-action`.
.. vv-status: multiplication-operator-embedding documented

.. math::
   :label: multiplication-operator-embedding

   M \;:\; L^\infty \;\longrightarrow\; B(L^2),
   \qquad
   (M[f]\,\psi)(\xi) \;=\; f(\xi)\,\psi(\xi),

where :math:`\xi = (\hat\Omega, g, \vec r)` ranges over the discrete
phase space (ordinate, group, cell). The map :math:`M` is the canonical
faithful unital ``*``-homomorphism of :math:`L^\infty` *onto the
diagonal subalgebra* of :math:`B(L^2)` — "diagonal" because
multiplication by :math:`f` couples no two phase-space points: the
output at :math:`\xi` reads the input only *there*. This is the algebraic
content of the §5.6 *locality* criterion that separates the §5.7
multiplication operator from the nonlocal integral kernels
(:ref:`integral-kernel-category`): :math:`M[f]` is the diagonal
sub-algebra, the kernels are everything off-diagonal.

.. implements:: multiplication-operator-embedding
   :by: orpheus.transport.operators.multiplication_operator.MultiplicationOperator

   **Implemented by** the class itself: it *is* the map :math:`f \mapsto
   M[f]`. It stores **only** a
   :class:`~orpheus.transport.fields.cross_section_field.CrossSectionField`,
   so "the cross section IS the operator" is a fact about the type and not a
   comment on it — there is no second piece of state for the embedding to
   lose or contradict.

   The division of labour with the next equation is deliberate and the two
   declarations are disjoint: the **class** realizes the embedding, its
   ``apply`` realizes the discrete **action**
   (:eq:`multiplication-operator-action`).

For the leading-ordinate broadcast on the SN per-ordinate carrier
:math:`\psi(\hat\Omega_n, g, \vec r)`, the discrete embedding is the
group-and-space-indexed broadcast over the ordinate axis:

.. (vv-status rationale) #257 S3b — the §5.7 multiplier promotion. The
   broadcast action is verified at the VALUES level against the legacy
   ``σ[None]·ψ`` (0 ULP, ``assert_array_equal`` — the generalized engine
   reduces to the same broadcast-multiply) and the multiplier-algebra
   laws are verified as intrinsic properties (``tests/transport/
   test_multiplication_operator.py``). The structurally-independent
   physics backing is the k_∞ = νΣf/Σa analytical limit and the
   streaming-equilibrium ψ = Q/σ_t reference, both of which route σ_t
   through the promoted C.
.. vv-status: multiplication-operator-action documented

.. math::
   :label: multiplication-operator-action

   (M[f]\,\psi)_{n,g,\vec r} \;=\; f_{g,\vec r}\,\psi_{n,g,\vec r}.

.. implements:: multiplication-operator-action
   :by: orpheus.transport.operators.multiplication_operator.MultiplicationOperator.apply

   **Implemented by** the public verb, through the ONE multiply
   ``_run`` — :math:`f_{g,\vec r}\,\psi_{n,g,\vec r}` evaluated by the
   :class:`~orpheus.numerics.operator.DiagonalOperator` broadcast engine
   built once over the immutable coefficient, on either bulk rank. What
   the binding selects is only *how the result is packaged*: a composite
   binding lifts (the multiply on the bulk block, the implicit zero
   source/sink on the trace), a plain binding returns the bare array.
   The arithmetic is single-sourced, so the two bindings agree by
   construction and not by a copied predicate.

   ⚠ **This declaration named** ``_apply_impl`` **until CS4c step 5
   (2026-09-04)**, because the public ``apply`` then existed only under
   ``TYPE_CHECKING`` — at runtime it was an alias of a
   :func:`functools.singledispatchmethod` dispatcher with no symbol of
   its own, and the per-carrier ``.register`` arms were anonymous ``_``
   functions, so the private dispatcher was the finest **addressable**
   grain (the same held for the scattering and fission operators). That
   shape is retired: ``apply`` is a real method with a real body again,
   its :func:`typing.overload` stubs name the two *bindings* rather than
   a carrier zoo, and the retired dispatcher's rationale is preserved at
   :ref:`pattern-m-history`. The equation's finest grain is now the
   verb the caller actually names.

The action is delegated to the N-D
:class:`~orpheus.numerics.operator.DiagonalOperator` broadcast engine
(:eq:`diagonal-operator-action`, #257 S3a) as
``DiagonalOperator(f.values, broadcast_axes=(0,))``: the coefficient is
an :math:`(n_g, *\text{spatial})` field, broadcast over the leading
ordinate axis. The transport-level
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
does **not** re-derive the broadcast — it *builds the engine once* over
its immutable coefficient and forwards (``coding-elegance`` Pattern 2,
single source of truth), so the transport operator and the numerics
engine agree by construction rather than by a copied predicate.

The cross section IS the operator
---------------------------------

The promotion is the moment the opaque ``Generic[V]`` becomes an honest
*carrier*. Before §5.7, the collision operator stored a raw
:math:`\sigma_t` array (an ``ndarray`` with no type-level meaning) and
re-implemented the broadcast in its ``apply`` body — the type
``CollisionOperator`` said nothing, and the meaning lived in a comment.
After §5.7,
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
stores **only** a :class:`~orpheus.transport.fields.cross_section_field.CrossSectionField`
(the cone-typed coefficient of #257 S1), sourced through the typed
:meth:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.total_cross_section_field`
accessor (#257 S2). The operator's *identity* is its coefficient field;
the action follows from the embedding. The collision leaf is now a
**plain** :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
carrying :math:`\sigma_t` as its coefficient — :math:`C = M[\sigma_t]` is
literally true, with no SN-specific subtype at all. #261 retired the
former ``CollisionOperator`` thin subclass: once the transport base gained
an optional ``space`` binding (for the W-D composition guard — a field
CS4c step 2 superseded with the mandatory kw-only write-once
``domain``/``codomain`` ends of
:class:`~orpheus.transport.operators.bound_operator.BoundOperator`) and
the bare-array
:meth:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator.from_mesh`
constructor, the subclass added nothing the base lacked. The ``L + C``
dispatch that assembles the bundled
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` lives **one-directionally**
on :meth:`~orpheus.sn.operators.streaming.StreamingOperator.__add__` (keyed on the
``MultiplicationOperator`` base type): ``L + C`` is the canonical (and
only) ordering, because the ``numerics ↛ transport ↛ sn`` layer order
forbids a transport-level multiplier from dispatching back onto an ``sn``
operator (that would be a ``transport → sn`` upward import).

The multiplier-algebra laws
---------------------------

That :math:`M` is a faithful unital ``*``-homomorphism is not decoration
— it is the set of *intrinsic properties* the promotion must satisfy, and
each is pinned as a law-suite test (the user directive: every
math-bearing type ships a test of its defining laws). The laws, verified
in ``tests/transport/test_multiplication_operator.py`` on a discriminating
``nx=5 ≠ ny=3, ng=2`` carrier with asymmetric heterogeneous coefficients:

.. list-table:: Multiplier-algebra laws (faithful unital ``*``-homomorphism)
   :header-rows: 1
   :widths: 26 30 44

   * - Law
     - Statement
     - Meaning / verification
   * - **Unit**
     - :math:`M[1] = I`
     - The constant-one coefficient is the identity operator (the
       embedding is *unital*).
   * - **Zero (codomain-aware)**
     - :math:`M[0] = 0`
     - The zero coefficient is the zero operator — but its codomain is a
       **source**, not a flux: collision turns flux into a reaction rate,
       so ``M[0].apply`` emits a zeroed
       :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`.
   * - **Linearity**
     - :math:`M[af + bg] = a\,M[f] + b\,M[g]`
     - :math:`M` is linear in the coefficient (the embedding is a vector-
       space map). Tested on ≥2-group asymmetric heterogeneous fields.
   * - **Homomorphism**
     - :math:`M[f]\,M[g] = M[f g]`
     - Composing two multiplications multiplies the coefficients. Tested
       at the VALUES level on the raw :math:`\sigma\cdot\sigma'` product
       (which has units :math:`\mathrm{cm}^{-2}` — the units grading is
       exactly why coefficient-field ``*`` is deferred to the values
       layer rather than the typed field layer).
   * - **Self-adjointness**
     - :math:`M[f]^* = M[f]` for real :math:`f`
     - A real-valued multiplication is self-adjoint;
       ``apply_transpose`` is the same code path as ``apply``.
   * - **Spectrum**
     - :math:`\mathrm{spec}(M[f]) = \mathrm{ess\,range}(f)`
     - :math:`M[f]` is invertible **iff** :math:`f` is bounded away from
       zero; the inverse is :math:`M[1/f]`. This is the honest
       invertibility gate (below).

The spectrum law and the honest invertibility gate
--------------------------------------------------

The spectrum of a multiplication operator is the essential range of its
coefficient: :math:`M[f]` has a bounded inverse iff
:math:`\mathrm{ess\,inf}\,|f| > 0`, in which case
:math:`M[f]^{-1} = M[1/f]`. The promotion enforces this at the value
level —
:attr:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator.is_invertible`
reports ``True`` **iff** :math:`\min|f| > 0`, single-sourced from the
broadcast engine (the operator reads the engine's ``is_invertible``, so
the transport operator and the numerics engine agree by construction).
The multiplier is the *value-dependent* arm of Design C
(:ref:`design-c-structural-value-split`): it always **declares**
``inverse()`` / ``solve`` (the type *can* invert), and refuses eagerly.

This is a **behavioral strengthening**, an illegal-states-unrepresentable
hardening (``coding-elegance`` Pattern 4). The legacy ``CollisionOperator``
advertised a ``solve``
*unconditionally*; on a :math:`\sigma = 0` entry its ``solve`` divided
:math:`q/\sigma` and produced a **silent IEEE NaN** that propagated into
the iterate. The promotion refuses the inverse it does not have: a zero
coefficient has no inverse, so
:attr:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator.is_invertible`
reports ``False`` and any request for the inverse
(``inverse()`` / ``solve``) raises
:class:`~orpheus.numerics.operator.NotInvertible` **eagerly** rather than
producing a NaN at *call* time. Construction still succeeds (the gate
governs only the inverse, never blocks the object); ``apply`` is
unaffected.

.. note::

   The gate change is purely additive honesty — an audit (#257 S3b)
   confirmed **no production path** relies on a :math:`\sigma=0` collision
   ``solve``. Since B.2d d3 the within-group :math:`L + C` composition is
   spelled in **one** place — the fused-factor primitive
   :func:`~orpheus.sn.coupled_system.build_streaming_collision`, called by
   BOTH :func:`~orpheus.sn.coupled_system.build_within_group_system` (whose
   :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record the
   solver's within-group inners consume) and the solver's own ``L``
   binding legs (the former per-solver ``Streaming + Collision``
   spellings — the third copy was the L-002 collapse trigger). That builder
   sources
   :math:`\sigma_t > 0` (total cross sections are bounded away from zero),
   and the removal cross section :math:`\sigma_r` ``solve`` appears only in
   a docstring, with no live caller. The bundled
   :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` has its own
   stricter construction-time ``min σ > 0`` guard, consistent with the new
   gate.


.. _functional-category:

The functional category (the §5.6 suffix law)
=============================================

The grand report §5.6 *suffix law* partitions the maps of the algebra
by what they return: an **Operator** maps a field to a field
(:eq:`operator-apply`), a **Kernel** integrates a field against a
measure (:eq:`scattering-as-tensor-product-sum`), and a **Functional**
maps a field to a **scalar** — or, fiberwise over space, to a
scalar-*field*. The functional surface is the single method
``evaluate(x) -> R``; it deliberately carries **none** of the
:class:`~orpheus.numerics.operator.LinearOperator` surface (no
``apply``, no ``is_invertible`` / ``is_adjointable``), and that
disjointness is the category's defining property (#257 S5,
:class:`orpheus.numerics.functional.Functional`).

The category is seated at **both layers**. The generic numerics floor is
:class:`~orpheus.numerics.functional.InnerProductFunctional`\ ``(weight,
axis)`` — the co-vector :math:`\langle w, \cdot\rangle` whose ``evaluate``
is the single primitive ``(w * x).sum(axis, keepdims=True)``. The
transport leaf
:class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
**specialises** it (carrying the weight as a domain-typed
:class:`~orpheus.transport.fields.cross_section_field.CrossSectionField`,
which brings ``.mesh`` and the ``1/cm`` units), exactly as
:class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame` specialises
:class:`~orpheus.numerics.frame.GalerkinFrame`. The canonical instance is
the per-cell reaction-rate **density**, contracting a reaction cross
section against the flux over the source groups:

.. math::
   :label: production-rate-functional

   r_x(\vec r) \;=\; \langle \Sigma_x, \phi\rangle
   \;=\; \sum_{g'} \Sigma_{x,g'}(\vec r)\,\phi_{g'}(\vec r),

with the group axis collapsed and **no** volume measure folded in (it
is the density, not the integral :math:`\int r_x\,dV`). The two named
instances are the **production rate** (:math:`\Sigma_x = \nu\Sigma_f`)
and the **absorption rate** (:math:`\Sigma_x = \Sigma_a`); the
Rayleigh-quotient eigenvalue is their ratio
:math:`k = \langle\nu\Sigma_f,\phi\rangle / \langle\Sigma_a,\phi\rangle`.

.. implements:: production-rate-functional
   :by: orpheus.transport.reaction_rate_functional.ReactionRateFunctional

   **Implemented by** the typed co-vector :math:`\langle\Sigma_x,\cdot
   \rangle` itself: the constructor binds ``weight = cross_section.values``
   and ``axis = 0``, so the group axis is the contracted one and the spatial
   axes survive as the per-cell **density**.

   ⛔ Deliberately **not**
   :meth:`InnerProductFunctional.evaluate <orpheus.numerics.functional.InnerProductFunctional.evaluate>`.
   That is the *generic* :math:`\langle w, \cdot\rangle` and would remain
   correct even if this equation were wrong; the reaction-rate claim — which
   weight, which axis, no volume measure — lives only in the transport
   specialisation.

This functional is not a parallel *description* of a contraction the
operator algebra performs elsewhere — it **is** the contraction. It is
the row-factor :math:`\langle\nu\Sigma_f|` of the fission rank-1 dyad
:math:`F = |\chi\rangle\langle\nu\Sigma_f|`
(:ref:`fission-as-dyad`): :meth:`RankOneOperator.apply
<orpheus.numerics.operator.RankOneOperator.apply>` routes the fission
matvec *through* ``functional.evaluate``, so there is no separate "fused"
realization to drift from the named factor (the procedural twin the
earlier S5 design carried is **dissolved** — :ref:`fission-as-dyad`
records the upgrade). Naming the contraction turns the most physically
central diagnostic in criticality into a typed, inspectable Functional.

The volume-integrated companion
:class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
returns the **scalar** :math:`R_x = \int_V \langle\Sigma_x,\phi\rangle\,dV`
— the per-cell density above, integrated against the mesh's canonical
``volume_measure`` (single source: it reuses
:eq:`production-rate-functional`'s density, no independent re-derivation
of either reduction). It is the typed object behind the
:math:`k`-eigenvalue numerator and denominator
(:ref:`integrated-reaction-rate-keff`), and the :math:`\phi^\dagger=1`
degenerate of the homogenisation Petrov–Galerkin campaign's
adjoint-weighted bilinear :math:`\langle\phi^\dagger, M[\Sigma_x]\,\phi\rangle`
(a future adjoint flux replaces the implicit :math:`\phi^\dagger = 1`).

.. vv-status: production-rate-functional documented

The §5.6 suffix law as a type-system fact
-----------------------------------------

The §5.6 *suffix law* is not a taxonomy imposed on the prose — it is a
structural fact the type system enforces by what each category's surface
*is*. Three categories partition the maps of the algebra by their
codomain:

.. list-table:: The §5.6 suffix law (the codomain partition)
   :header-rows: 1
   :widths: 18 24 24 34

   * - Category
     - Signature
     - Type relationship
     - ORPHEUS realization
   * - **Operator**
     - field :math:`\to` field
     - the base —
       :class:`~orpheus.numerics.operator.LinearOperator`
     - streaming :math:`L`, collision :math:`C = M[\sigma_t]`
       (:eq:`multiplication-operator-action`)
   * - **Kernel**
     - field :math:`\to` field, *nonlocal*
     - a **refinement** of Operator (adds ``kernel``)
     - scattering :math:`S`, fission :math:`F`
       (:ref:`integral-kernel-category`)
   * - **Functional**
     - field :math:`\to` scalar
     - a **disjoint sibling** of Operator (shares no member)
     - the reaction-rate density :math:`r_x(\vec r)`
       (:eq:`production-rate-functional`)

The asymmetry between the two non-Operator categories is the point. A
**Kernel** *is* a :class:`~orpheus.numerics.operator.LinearOperator` (it
still maps a field to a field, it still has ``apply`` and the two-axis
predicates) and merely *adds* the ``kernel`` member — so it is a
strict refinement (:ref:`integral-kernel-category`). A **Functional** is
*not* a :class:`~orpheus.numerics.operator.LinearOperator` at all: its
sole surface is ``evaluate(x) -> R``, and it deliberately carries
**none** of the operator surface — no ``apply``, no ``is_invertible`` /
``is_adjointable``, no ``solve``, no ``apply_transpose``, no ``.H``, no
``domain`` / ``codomain``.
That *disjointness* is the category's defining property, not an
omission, and it is checkable: ``isinstance(p, Functional)`` is true while
``isinstance(p, LinearOperator)`` is false, and the discriminator foils
both directions (a bare operator is not a Functional; the production-rate
functional is not an operator). The category is the
:class:`orpheus.numerics.functional.Functional` Protocol (#257 S5, the L1
numerics floor — the **co-vector companion of**
:class:`~orpheus.numerics.vector.Vector`: a ``Vector`` is what an
operator acts *on*, a ``Functional`` is what acts *on* a vector to
produce a scalar).

Variance: a Functional is contravariant in, covariant out
---------------------------------------------------------

The Functional's typevars carry a *different* variance profile than the
operator's, and the difference is structural. An operator's
``apply(x: V) -> V`` uses :math:`V` **both ways** (it consumes a flux and
produces a flux), so :math:`V` is *invariant by dual use* — and pyright
emits no variance warning. A functional's ``evaluate(x: V_contra) ->
R_co`` uses its input **only** as a consumer (contravariant) and its
result **only** as a producer (covariant). Declaring the Functional over
the operator's invariant :math:`V` would be a type error of variance;
the correct declaration is a contravariant input typevar and a
covariant, *unbounded* result typevar (the result is unbounded because
the production rate's "scalar" is fiberwise a scalar-**field** over space,
not a Python :class:`float` — a ``float | V`` union would mistype it).
This is the co-vector mirror of the invariant :math:`V` and is the
load-bearing typing lesson of S5: the variance is not cosmetic, it is the
formal statement that a functional is a co-vector.

The reaction rate is a density, not an integral
-----------------------------------------------

The :class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
of :eq:`production-rate-functional` returns the per-cell reaction-rate
**density** :math:`r_x(\vec r)` — the cross section contracted against the
flux over the source groups, group axis collapsed, **no volume measure
folded in**. The "no measure" choice is a deliberate Mode-3
(missing-factor) guard turned into a *named contract*: :math:`r_x` is the
density :math:`\Sigma_x\,\phi`, not the integral :math:`\int r_x\,dV`. A
consumer that needs the *integrated* rate uses the dedicated
:class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
(which applies the mesh ``volume_measure`` to this exact density); folding
the measure into :math:`r_x` itself would silently double-count it the
moment a second consumer integrated again. The split is the structural
division between the two functionals — the density carries the group
contraction, the integrated rate adds the spatial measure on top of it.
The coefficient side is the cone-typed
:class:`~orpheus.transport.fields.cross_section_field.CrossSectionField`
(#257 S1) carrying :math:`\Sigma_x` through the typed
:meth:`~orpheus.transport.mesh.material_xs_field.MaterialXSField.fission_production_field`
accessor (#257 S2) for production, or the absorption accessor for
:math:`\Sigma_a`.

.. _integrated-reaction-rate-keff:

The eigenvalue numerator and denominator are integrated reaction rates
----------------------------------------------------------------------

The SN :math:`k`-eigenvalue is routed through
:class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
directly:

.. math::
   :label: keff-as-integrated-rates

   k \;=\; \frac{R_{\nu\Sigma_f}(\phi)}
                {R_{\Sigma_a}(\phi) \;+\; L_{\rm leak} \;-\; E_{2n}(\phi)} ,
   \qquad
   R_x(\phi) \;=\; \int_V \langle\Sigma_x,\phi\rangle\,dV .

.. implements:: keff-as-integrated-rates
   :by: orpheus.sn.solver.SNSolver.compute_keff

   **Implemented by** the only site that computes :math:`k` as a ratio of
   :class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
   values. Its denominator is assembled term-by-term — ``production /
   (absorption + leakage - emission_n2n.sum())`` — precisely so that no
   term *can* be silently dropped, which is the structural answer to the
   #291 omission the equation itself used to reproduce.

**The numerator is fission ONLY; the denominator is net removal.**
:math:`R_{\nu\Sigma_f}` and :math:`R_{\Sigma_a}` are the
volume-integrated reaction rates of :eq:`production-rate-functional`,
both routed through :class:`IntegratedReactionRate
<orpheus.transport.reaction_rate_functional.IntegratedReactionRate>`,
which is what this section is about. The other two denominator terms are
**not** :math:`\langle\Sigma_x,\phi\rangle` contractions and are added
explicitly: :math:`L_{\rm leak}` is the net vacuum-boundary outflow (a
**structural zero** on an all-reflective problem, which is what keeps a
lattice case bit-identical to the historical ``production /
absorption``), and :math:`E_{2n}` is the :math:`(n,2n)` emission,
*subtracted* because a gain reduces net removal.

.. important::

   **The canonical derivation is not here.** :eq:`sn-keff-update` in
   :ref:`sn-keff-estimator` is the single source of truth for this
   estimator: the divergence-telescoping cell balance it falls out of,
   the per-term definitions, the leakage functional
   :eq:`sn-leakage-functional`, and the wiring to the cross-engine
   consistency gate all live there. This page restates the formula only
   because its own label must not state a false one — the *claim* being
   made here is the **typing** claim (both ends of the ratio are the
   same typed functional), not the estimator's correctness. Any future
   change to the estimator is edited **there** and mirrored here.

.. note::

   The leakage term is spelled :math:`L_{\rm leak}` on this page **only**
   to avoid colliding with the streaming operator
   :math:`L = \hat\Omega\cdot\nabla` used everywhere else here;
   :eq:`sn-keff-update` and
   :meth:`~orpheus.sn.solver.SNSolver.compute_keff`'s own docstring both
   write it bare as :math:`L`. Same quantity, disambiguated locally.

Why the two extra terms sit where they do is a fact about the **posing**,
not about bookkeeping taste: every inner solve poses the eigenproblem
with **only** fission scaled by :math:`1/k` (:eq:`operator-eigenvalue`),
while scattering and the :math:`(n,2n)` emission are plain gains inside
the within-group problem, so the reported :math:`k` must be the
eigenvalue of exactly *that* problem.

.. note:: **Correction (2026-08-17).** Until this revision the labelled
   equation read :math:`k = \bigl(\int_V\langle\nu\Sigma_f,\phi\rangle\,dV
   + (n,2n)\bigr) \big/ \int_V\langle\Sigma_a,\phi\rangle\,dV` — the
   :math:`(n,2n)` channel in the **numerator**, and **no leakage term at
   all**. That is the pre-#291, pre-R7 convention; both were superseded on
   2026-07-03 and :meth:`~orpheus.sn.solver.SNSolver.compute_keff` has
   computed the form above ever since — so the equation had been
   present-tense false against the very method the surrounding prose named
   as its implementer. The stale numerator is not a stale *spelling* of
   :math:`k`'s numerator: :math:`\int_V\langle\nu\Sigma_f,\phi\rangle\,dV
   + (n,2n)` is exactly
   :meth:`~orpheus.sn.solver.SNSolver.compute_production_rate`, the
   **total physical production** that
   :func:`~orpheus.numerics.eigenvalue.power_iteration` uses as the
   ERR-052 renormalisation scale anchor. Two different quantities, and the
   method's own docstring contrasts them explicitly ("the k numerator …
   is fission-only, because the posed eigenproblem scales only fission by
   :math:`1/k`").

Reusing
:class:`IntegratedReactionRate
<orpheus.transport.reaction_rate_functional.IntegratedReactionRate>` for
both production and absorption gives the eigenvalue numerator and
denominator a single typed source for the :math:`\langle\Sigma_x,\phi
\rangle` contraction and its volume integral, the
:math:`\phi^\dagger\!=\!1` degenerate of the homogenisation
Petrov–Galerkin bilinear :math:`\langle\phi^\dagger, M[\Sigma_x]\,\phi
\rangle`.

.. note::

   **Scope.** The SN :math:`k`-eigenvalue routes through
   :class:`IntegratedReactionRate
   <orpheus.transport.reaction_rate_functional.IntegratedReactionRate>`
   for **both** its numerator and denominator. Diffusion (#290) routes
   its **production** rate through the same functional (the #270
   diffusion arm — see :doc:`/theory/methods/diffusion_1d`), but poses its
   denominator as the integrated loss-operator action
   :math:`\langle 1, (A\psi)_{\rm bulk}\rangle_V` (= absorption +
   leakage by the column-sum theorem), not a second reaction-rate
   contraction. The CP and MoC eigenvalues are **not yet** routed;
   the homogeneous 0-D case has no spatial integral to fold, so it
   does not participate. Do not read this section as a claim that every
   solver's :math:`k` flows through the typed functional in the same
   way — it is the SN path, with diffusion's production numerator the
   one other current consumer.

.. (vv-status rationale) Structural / decomposition label: the SN
   eigenvalue expressed as a ratio of integrated reaction rates. Not a new
   solver claim (the SN keff value is verified by the existing eigenvalue
   gates); the verifiable content of the *routing* is the closed-form
   k∞-as-ratio gate ``tests/transport/test_integrated_reaction_rate.py``.
.. vv-status: keff-as-integrated-rates documented

.. _reaction-rate-kinf-oracle-section:

Verification: the closed-form :math:`k_\infty` oracle, not a procedural twin
----------------------------------------------------------------------------

The earlier S5 design verified the functional **byte-for-byte (0 ULP)
against the rank-1 ``inner`` reduction** :math:`\chi\cdot\mathrm{evaluate}
\equiv F.\mathrm{apply}`. That cross-check was *procedurally* independent
(two code paths) but **not** *structurally* independent (both sides ran
the same NumPy ``(w * x).sum(axis, keepdims)`` primitive on the same
arrays) — by ``vv-principles`` L11 a procedural twin proves the two
spellings agree, not that either is **correct**. With the matvec now
routed *through* ``functional.evaluate`` (:ref:`fission-as-dyad`) the twin
no longer even exists to compare against: there is one contraction, not
two.

The correctness floor is therefore re-anchored on a **structurally
independent closed-form ground** — the infinite-medium decomposition

.. math::
   :label: reaction-rate-kinf-oracle

   k_\infty \;=\; \lambda_{\max}\!\bigl(\mathbf{A}^{-1}\mathbf{F}\bigr),
   \qquad
   \mathbf{A} \;=\; \mathrm{diag}(\Sigt{})
       \;-\; \bigl(\Sigma_s + 2\,\Sigma_2\bigr)^{\mathsf T},
   \qquad
   \mathbf{F} \;=\; |\chi\rangle\langle\nu\Sigma_f| ,

whose dominant eigenpair :math:`(k_\infty, \phi^*)` comes from a
:func:`numpy.linalg.eig` of the transfer matrix — a path that shares **no
primitive** with the rank-1 functional. The
:class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
is pinned by evaluating production **and** absorption *independently* at
the converged spectrum :math:`\phi^*` and checking
:math:`\langle\nu\Sigma_f,\phi^*\rangle / \langle\Sigma_a,\phi^*\rangle =
k_\infty` (``tests/transport/test_reaction_rate_functional.py``). Pinning
the two functionals *separately* (not just their ratio) is what gives the
gate teeth a ratio test lacks: a shared-factor error that scales both the
numerator and the denominator — a mis-scaled accessor, a spurious volume
fold on both — cancels in :math:`k` but is caught term-by-term.

This is a legal **eigenvalue claim** paired with the **closed-form**
pillar (``vv-principles``: eigenvalue claims need a closed-form or
semi-analytical reference, never MMS). It carries genuine **flux-shape
teeth** because the test uses a **4-group** mixture whose converged
:math:`\phi^*` is genuinely non-flat — the 2-group case is degenerate (its
:math:`\phi^*` is coincidentally flat ``[0.707, 0.707]``, so its flat-flux
ratio equals :math:`k_\infty` and is flux-shape-blind), so the 4-group leg
is **mandatory** (``vv-principles`` anti-pattern #3: a 1-group, or any
flat-spectrum, eigenvalue test cannot detect a flux-shape error). A
second leg pins the per-cell density against an explicit Python
double-loop (``hand_derived_production_density`` — an accumulation with no
NumPy reduction, the L11 structurally-independent reference), and a third
asserts the **no-volume-measure** contract (Mode-3). Together they replace
the retired procedural twin with a reference at the right level for each
claim.

Why :math:`\mathbf{A}` is spelled this way
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pair above is transcribed from the algebra of record,
:func:`~orpheus.derivations.common.eigenvalue._infinite_medium_matrices`.
Two details of that spelling are load-bearing, and both are lost if
:math:`\mathbf{A}` is written the textbook way as
:math:`\mathrm{diag}(\Sigma_a) - \Sigma_s`:

* **The transpose is forced by the storage convention.** ``Mixture.SigS``
  is stored ``SigS[g_from, g_to]``
  (:ref:`scattering-matrix-convention`), so the in-scatter *into* group
  :math:`g` is :math:`\sum_{g'}\Sigma_{s,g'\to g}\,\phi_{g'} =
  (\Sigma_s^{\mathsf T}\phi)_g`. Dropping the transpose is ERR-002 — a
  Mode-2 variable swap that is **invisible** on a symmetric (1-group,
  self-scatter-only) matrix and wrong for every asymmetric down-scatter
  case. It is one of the reasons the gate that consumes this oracle is
  mandatorily multi-group.
* **The removal diagonal is** :math:`\Sigt{}`, **not** :math:`\Sigma_a`,
  because the **whole** scattering matrix — self-scatter included — is
  subtracted, rather than only the group-changing part. The
  :math:`2\,\Sigma_2` term folds the :math:`(n,2n)` transfer in as a
  doubled gain, so the oracle stays exact on a multiplying-scatter
  mixture; ``sig_2=None`` reduces it away.

.. implements:: reaction-rate-kinf-oracle
   :by: orpheus.derivations.common.eigenvalue._infinite_medium_matrices

   **Implemented by** the :math:`(\mathbf{A}, \mathbf{F})` assembly
   itself — and it is the *single* site the forward and the adjoint
   references share, because two references that disagreed about the
   operator pair would produce incomparable :math:`k`.

.. implements:: reaction-rate-kinf-oracle
   :by: orpheus.derivations.common.eigenvalue.kinf_homogeneous

   **Implemented by** the eigenvalue itself, at any group count:
   ``np.linalg.solve(A, F)`` and then the dominant real eigenvalue of
   :func:`numpy.linalg.eig`.

.. implements:: reaction-rate-kinf-oracle
   :by: orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous

   **Implemented by** the same eigenvalue **plus** its dominant
   eigenvector — which is what gives the consuming gate its flux-shape
   teeth. The 4-group :math:`\phi^*` is genuinely non-flat, where the
   2-group one is coincidentally flat (``[0.707, 0.707]``) and therefore
   flux-shape-blind (``vv-principles`` anti-pattern #3).

   All three implementers live under ``orpheus/derivations/`` — **inside**
   the package, not under ``tests/`` — and the tests *import* them. That
   is what makes this a structurally-independent closed-form reference
   rather than a test-local twin.

.. (vv-status rationale) Structural / decomposition label: the closed-form
   k∞ = λ_max(A⁻¹F) identity that grounds the reaction-rate functional's
   correctness. Not itself a solver claim; the verifiable content is the
   eigenvalue/closed-form gate ``tests/transport/test_reaction_rate_functional.py``
   (production AND absorption pinned independently at φ*) and the
   ``tests/transport/test_integrated_reaction_rate.py`` k∞-as-ratio gate.
.. vv-status: reaction-rate-kinf-oracle documented

Why the estimators are NOT Functionals
---------------------------------------

The criticality eigenvalue and production-rate **estimators**
(:meth:`~orpheus.numerics.iteration.KEigenvalue.compute_keff` and
:meth:`~orpheus.numerics.iteration.KEigenvalue.compute_production_rate`)
are the obvious candidates to wrap as ``Functional`` objects — and they
are deliberately **not**. The eigenvalue estimator is a **ratio** of two
triple-dependent contractions,
:math:`\sum(F\psi)\,/\,(\sum(A\psi) - \sum(S\psi))` — it consumes the
whole operator triple :math:`(A, S, F)` (carried on the ``KEigenvalue``
instance) together with the iterate :math:`\psi`, not a lone field acted
on by a single co-vector. That ratio-of-triple-contractions shape is not
the ``evaluate(x) -> R`` shape of a
:class:`~orpheus.numerics.functional.Functional` (a linear co-vector on
one vector space). The category simply now *names* what their
field-to-scalar **core** is (the production-rate contraction
:math:`\sum(F\psi)`), without forcing the estimators into a wrapper that
would misrepresent their arity. They stay bare (hardwired) methods,
arithmetic bit-identical to the pre-R8 module-level defaults they
replaced (pinned by ``tests/numerics/test_estimators_as_functionals.py``),
and the honesty of *not* wrapping them is itself a category-correctness
claim.

.. note:: **Injection seam retired (R8, #259 P1, 2026-07-03).** Before
   this, these estimators were injectable *callables* — the
   ``_default_keff_estimator`` / ``_default_production_estimator`` module
   functions were the **defaults** of ``KEigenvalue``'s ``keff_estimator``
   / ``production_estimator`` kwargs, which a caller could override.  The
   kwargs, the ``KeffEstimator`` / ``ProductionEstimator`` aliases, and
   the ``_default_*`` functions are gone; the spellings moved verbatim
   onto the hardwired methods above.  The seam was dead **by design** —
   at a converged eigenpair every estimator consistent with the posed
   problem agrees, so an injected *different* estimator could only be
   inconsistent (illegal states unrepresentable).  See the full R8
   rationale — the consistency theorem and the honest-``A.apply``
   contract — in the *KEigenvalue: outer power iteration* section of
   :doc:`/theory/methods/sn/index`.

.. note::

   **Source map.** Category Protocol:
   :class:`orpheus.numerics.functional.Functional` (L1). Generic numerics
   leaf: :class:`~orpheus.numerics.functional.InnerProductFunctional`
   ``⟨w, ·⟩`` (L1). Transport leaves:
   :class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
   (the per-cell density, specialising ``InnerProductFunctional``) and
   :class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
   (the volume-integrated scalar) (L2). Rank-1 constructor:
   :func:`~orpheus.numerics.operator.outer`. Intrinsic-category gate:
   ``tests/transport/test_functional_category.py`` (Functional ≠
   LinearOperator, both directions + discriminator foils). Correctness:
   ``tests/transport/test_reaction_rate_functional.py`` (the closed-form
   :math:`k_\infty = \lambda_{\max}(A^{-1}F)` per-term oracle —
   production AND absorption pinned independently — + the hand-derived
   double-loop density reference + the no-measure guard);
   ``tests/transport/test_integrated_reaction_rate.py``
   (:math:`k_\infty` as the ratio of integrated rates, incl. the
   ``(n,2n)`` term). Dyad laws: ``tests/numerics/test_outer_dyad.py``
   (action / rank-1 / adjointability / linearity). Estimator honesty:
   ``tests/numerics/test_estimators_as_functionals.py``. The
   field-to-scalar contraction is coded in three places today
   (SN / CP / numerics) — unifying that fragmentation is tracked on #259.


.. _integral-kernel-category:

The integral-kernel category (the §5.6 Kernel suffix)
=====================================================

The §5.6 suffix law's middle term is the **Kernel**: a
:class:`~orpheus.numerics.operator.LinearOperator` whose action is
**nonlocal** — it integrates the carrier field against a measure on one
or more axes — distinct from a LOCAL / diagonal
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
(:eq:`multiplication-operator-action`, the §5.7 Operator) and from a
field→scalar :class:`~orpheus.numerics.functional.Functional`
(:eq:`production-rate-functional`). The single discriminator is
**locality** (Frame 3): a multiplication operator's output at a point is
a pointwise function of the input *there*, while a Kernel's output reads
the input across an integrated axis. #257 S6 names this category as the
:class:`orpheus.transport.operators.integral_kernel_operator.IntegralKernelOperator`
Protocol — a **refinement of** LinearOperator (it still has ``apply`` +
the two-axis predicates, UNLIKE the disjoint Functional) that adds a
single ``kernel`` member exposing the integral structure as a
:class:`~orpheus.numerics.operator.LinearOperator`:

.. math::
   :label: integral-kernel-category

   (A\,\psi)(x) \;=\; \int K(x, x')\,\psi(x')\,d\mu(x') ,
   \qquad K \;=\; A.\mathrm{kernel}.

.. implements:: integral-kernel-category
   :by: orpheus.transport.operators.integral_kernel_operator.IntegralKernelOperator.kernel

   **Implemented by** the Protocol member whose declaration *is* this
   equation, verbatim in its docstring: a kernel operator is one that can
   hand you its :math:`K`, returned as the common
   :class:`~orpheus.numerics.operator.LinearOperator` supertype. This is the
   declaration-site reading again — the same one taken for
   :eq:`operator-apply`.

   The two concrete kernels carry their **own** labels
   (:eq:`fission-as-dyad` for the rank-1
   :class:`~orpheus.numerics.operator.TensorProductOperator`,
   :eq:`scattering-aniso-composite` for the :math:`R\circ\Lambda\circ M`
   :class:`~orpheus.numerics.operator.OperatorProduct`); this declaration is
   the **category**, not an instance of it.

The two named transport instances are the Boltzmann emission kernels.
**Fission** exposes the rank-1 **dyad** :math:`F = |\chi\rangle\langle
\nu\Sigma_f| = \texttt{outer}(\chi,\,\mathrm{ReactionRateFunctional}(\nu
\Sigma_f))` (a :class:`~orpheus.numerics.operator.RankOneOperator`, lifted
to a :class:`~orpheus.numerics.operator.TensorProductOperator` only to
advertise the spatial-axis broadcast). Its row co-vector
:math:`\langle\nu\Sigma_f|` is the S5 :eq:`production-rate-functional`
density, and the matvec routes *through* that functional's ``evaluate``
(:ref:`fission-as-dyad`) — there is no fused procedural twin. **Scattering**
exposes the anisotropic Legendre redistribution :math:`R \circ \Lambda
\circ M` (an :class:`~orpheus.numerics.operator.OperatorProduct` whose
middle factor is the moment-space
:eq:`scattering-as-tensor-product-sum`); the isotropic :math:`P_0`
in-scatter is the local component of the full scattering ``apply``.
(The :math:`(n,2n)` doubling stood beside it in this list until
2026-08-30; it is now a separate first-class operator entirely —
:ref:`scattering-binding-cs4c`.) Fission is the **rank-1
(single-mode) degenerate** of scattering's multi-mode spectral sum — the
polyadic/block-term view of :ref:`emission-kernels-btd` makes the
relationship precise. The matvec arms of both operators are UNCHANGED in
S6 (additive, bit-identical).

The scattering kernel :math:`R\circ\Lambda\circ M` is not an arbitrary
nonlocal operator: it is the **spectral theorem** :math:`A = U\Sigma U^*`
of a rotationally-invariant kernel. The scattering kernel
:math:`\Sigma_s(\hat\Omega\cdot\hat\Omega')` depends only on the
direction cosine — a *zonal* kernel — so by the Funk–Hecke theorem the
spherical harmonics are its eigenfunctions, with eigenvalues the
Legendre moments :math:`\Sigma_{s,\ell}` (the diagonal of
:math:`\Lambda` =
:class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`). Reading
:math:`M` = :math:`U^*` (change of basis *into* the eigenbasis),
:math:`\Lambda` = :math:`\Sigma` (the diagonal spectrum), and
:math:`R` = :math:`U` (synthesis *out of* it) is what makes the
conjugation
:math:`S = \tfrac{1}{W}\,\texttt{frame.conjugate}(\Lambda)` — the
scattering **2-cell** of the carrier-grid double category
(:ref:`carrier-grid-double-category`) — a *spectral* statement: the
horizontal adjoint pair :math:`(M, R)` is the unitary diagonalising the
vertical :math:`\Lambda`. This is also why the harmonic frame is
**scattering's** frame: the spherical harmonics are *scattering's*
eigenbasis, and the streaming operator — the :math:`\ell=1` direction
irrep — is the one transport operator the basis does **not**
diagonalise (it couples :math:`\ell\!\leftrightarrow\!\ell\pm1`, the Pℓ
recurrence). The full Funk–Hecke / Schur derivation, the literature
corroboration, and the unifying principle *"an operator owns its frame
iff the frame is its eigenbasis"* (which also explains why energy
condensation and spatial homogenisation are Petrov-Galerkin, not
Galerkin) are in :ref:`frame-eigenbasis-ownership`
(:doc:`/theory/foundations/frame`).  **Ownership in that sentence is
mathematical, not constructional** — which basis diagonalises which
operator — and the two came apart at CS4c step 3, where the frame
stopped being minted inside :math:`S`'s constructor and became a shared
object handed in (:ref:`scattering-binding-cs4c`).

.. vv-status: integral-kernel-category documented

The locality criterion completes the partition
-----------------------------------------------

With the Kernel category named, the §5.6 suffix-law partition is
**complete and exhaustive**: every map in the transport algebra is an
Operator, a Kernel, or a Functional, and the boundary between Operator
and Kernel is a single mathematical criterion — **locality**. A
multiplication operator (:eq:`multiplication-operator-action`) is the
**diagonal** sub-algebra: its output at a phase-space point reads the
input only *there*. An integral kernel is everything **off-diagonal**:
its output at :math:`x` integrates the input across an axis,
:math:`(A\psi)(x) = \int K(x,x')\,\psi(x')\,d\mu(x')`. The Kernel's
``kernel`` member is exactly that integrating object :math:`K`, exposed
as a :class:`~orpheus.numerics.operator.LinearOperator` in its own right.

This is why the category is a **refinement** of
:class:`~orpheus.numerics.operator.LinearOperator` and not a disjoint
sibling like the Functional. The Protocol is written
``IntegralKernelOperator(LinearOperator[V], Protocol[V])`` — a Kernel
*is-a* LinearOperator (it keeps the inherited surface: ``apply``, the
``is_invertible`` / ``is_adjointable`` predicates, ``domain`` /
``codomain``) and merely *adds* the ``kernel`` property, which the
``@runtime_checkable`` ``isinstance`` sees on top of the inherited
members. The
refinement is **strict**, which the intrinsic gate verifies in both
directions: a kernel-less LinearOperator
(:class:`~orpheus.numerics.operator.IdentityOperator`, the local
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`)
is **not** a Kernel, and a :class:`~orpheus.numerics.functional.Functional`
(no ``apply``, no ``kernel``) is **not** a Kernel either. Variance is
inherited verbatim from LinearOperator — a Kernel's ``apply(x: V) -> V``
is invariant by dual use (no variance warning), *unlike* the co-vector
Functional that needed contravariant input + covariant output.

The two emission kernels: fission and scattering
------------------------------------------------

The two named transport instances are the Boltzmann **emission**
kernels, living in :mod:`orpheus.transport.operators`. Each gains a
``kernel`` member that satisfies the Protocol. **Scattering** is reframed
*in place* and additively: its ``kernel`` is the semantic reading of an
existing matvec that stays byte-for-byte unchanged, a cross-check rather
than a rewrite. **Fission**, by contrast, was genuinely *re-realized* as
the dyad :math:`|\chi\rangle\langle\nu\Sigma_f|` whose ``apply`` routes
through the production-rate functional (:ref:`fission-as-dyad`) — the
earlier "fused realization vs. semantic decomposition" split is gone,
because the dyad *is* the realization. The matvec **value** is unchanged
(0 ULP, :ref:`reaction-rate-kinf-oracle-section`); the realization is no
longer a procedural twin of a separately-named decomposition.

.. list-table:: The two §5.6 emission kernels
   :header-rows: 1
   :widths: 16 38 46

   * - Operator
     - ``kernel`` shape
     - Why this shape
   * - **Fission** :math:`F`
     - rank-1 dyad
       :math:`|\chi\rangle\langle\nu\Sigma_f|`
       (:class:`~orpheus.numerics.operator.RankOneOperator` via
       :func:`~orpheus.numerics.operator.outer`, lifted to a
       :class:`~orpheus.numerics.operator.TensorProductOperator` for the
       spatial-axis broadcast)
     - Fission emits an isotropic source whose group spectrum
       :math:`\chi` is the same everywhere fission occurs — a rank-1
       outer product of the emission-spectrum column with the
       production-rate row co-vector.
   * - **Scattering** :math:`S`
     - :math:`R \circ \Lambda \circ M`
       (an :class:`~orpheus.numerics.operator.OperatorProduct`)
     - The anisotropic :math:`\ell \ge 1` Legendre redistribution: project
       the angular flux onto moments (:math:`M`), apply the per-:math:`\ell`
       group-to-group transfer (:math:`\Lambda`), reconstruct to
       ordinates (:math:`R`). Genuinely nonlocal in angle.

.. _fission-as-dyad:

Fission as the rank-1 dyad :math:`|\chi\rangle\langle\nu\Sigma_f|`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fission is the rank-1 dyad — a reconstruction **column** :math:`|\chi\rangle`
(the emission spectrum) tensored with a functional **row**
:math:`\langle\nu\Sigma_f|` (the production-rate co-vector):

.. (vv-status rationale) Structural / decomposition of the fission
   operator — the rank-1 dyad reading. Not a solver claim; the verifiable
   content is the closed-form k∞ = λ_max(A⁻¹F) correctness of the row
   co-vector (``tests/transport/test_reaction_rate_functional.py``) and the
   0-ULP equivalence of the dyad apply to the matvec arm
   (``tests/sn/operators/test_fission_kernel_crosscheck.py``).
.. vv-status: fission-as-dyad documented

.. math::
   :label: fission-as-dyad

   F \;=\; |\chi\rangle\langle\nu\Sigma_f|
     \;=\; \texttt{outer}\bigl(\chi,\;
           \mathrm{ReactionRateFunctional}(\nu\Sigma_f)\bigr),

read right-to-left as the dyad action :math:`F\,\phi = \chi \cdot
\langle\nu\Sigma_f,\phi\rangle`: the row co-vector
:math:`\langle\nu\Sigma_f|` contracts the flux over groups to the scalar
emission **density** (the S5 :eq:`production-rate-functional`, exposed as
:attr:`IsotropicFission.production_rate <orpheus.transport.operators.isotropic_transfer.IsotropicFission.production_rate>`),
and the column :math:`\chi` broadcasts it back across the emission groups.
The realization is literally that dyad — ``outer(chi,
self.production_rate) & IdentityOperator()`` — and the matvec **routes
through** the row's ``evaluate`` (``coding-elegance`` Pattern 5: build the
*right primitive*; here the right primitive *is* the contraction, not a
parallel description of one). Because there is one contraction and not
two, the "fused vs. unfolded" distinction the earlier S5 design carried
**no longer exists** — :ref:`the verification section <reaction-rate-kinf-oracle-section>`
records the resulting upgrade from the dissolved procedural twin to the
closed-form :math:`k_\infty = \lambda_{\max}(A^{-1}F)` oracle.

.. note::

   **Where the dyad lives, since CS4c step 4 (2026-08-30).**  The
   arithmetic home of this equation moved from the angular operator to
   the **energy binding**
   :class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`,
   which is one half of the step's ruling that the fission channel is
   **two bindings of one datum**: a representation-free
   :class:`~orpheus.transport.kernels.FissionKernel` pair
   :math:`(\chi, \nu\Sigma_f)` per material, bound once on the *scalar*
   space (this dyad) and once on the *angular composite*
   (:class:`~orpheus.transport.operators.fission.FissionOperator`, the
   frame's :math:`\ell = 0` conjugation of the same dyad —
   :ref:`sn-fission-binding-adjoint`).
   :attr:`FissionOperator.kernel
   <orpheus.transport.operators.fission.FissionOperator.kernel>` and
   :attr:`FissionOperator.production_rate
   <orpheus.transport.operators.fission.FissionOperator.production_rate>`
   survive as *delegations* to that one home, so both spellings still
   read the identical object — which is why both are declared below.
   Two consequences for a reader of the transcriptions:

   * the column :math:`|\chi\rangle` is no longer "an array on the
     operator" — it is ``self.fission.gather_chi(bulk.shape[1:])``, a
     **gather** of the validated per-material kernels over the binding's
     own bulk shape (SPACE FIRST: the ends size the data);
   * the row is built from the gathered
     :math:`\nu\Sigma_f` rather than read off a facade field, because
     the operator no longer retains a ``MaterialXSField`` at all —
     ``FissionOperator.chi`` / ``.sig_p`` / ``.mat_xs`` are **retired**.

.. implements:: fission-as-dyad
   :by: orpheus.transport.operators.isotropic_transfer.IsotropicFission.kernel

   **Implemented by** the equation transcribed, in its one arithmetic
   home: ``chi = self.fission.gather_chi(tuple(bulk.shape[1:]))`` then
   ``return outer(chi, self.production_rate) & IdentityOperator()``. Both
   ``apply`` arms route the matvec **through** this kernel, and
   :meth:`RankOneOperator.apply <orpheus.numerics.operator.RankOneOperator.apply>`
   is ``recon * functional.evaluate(x)`` — so there is one contraction, not
   a fused realization sitting beside a named one.

.. implements:: fission-as-dyad
   :by: orpheus.transport.operators.isotropic_transfer.IsotropicFission.production_rate

   **Implemented by** the dyad's row co-vector
   :math:`\langle\nu\Sigma_f|`, exposed as the binding's own member:
   ``ReactionRateFunctional(CrossSectionField(values=nu_sig_f, space=bulk))``
   over the gathered ``nu_sig_f``.

.. implements:: fission-as-dyad
   :by: orpheus.transport.operators.fission.FissionOperator.kernel

   **Implemented by** the angular binding's *delegation* — ``return
   self.isotropic_energy.kernel``. It is declared because the §5.6
   integral-kernel
   Protocol and
   :class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicEmission`
   reach the dyad through this name; the arithmetic is the energy
   binding's above, so the two can never disagree.

.. implements:: fission-as-dyad
   :by: orpheus.transport.operators.fission.FissionOperator.production_rate

   **Implemented by** the same delegation on the row factor — ``return
   self.isotropic_energy.production_rate``.

.. note::

   **What was tried and rejected — the three-factor composition** :math:`F
   = M_\chi \circ \mathrm{ProductionRate} \circ M_{\nu\Sigma_f}`. An
   earlier reading framed fission as a literal three-operator product:
   multiply by :math:`\nu\Sigma_f`, contract to the density, re-broadcast
   through :math:`\chi`. It was abandoned for two reasons. First, it is
   **dimensionally inconsistent as a composition of operators**: the
   middle factor :math:`\mathrm{ProductionRate}` is a *Functional* (field
   :math:`\to` scalar-field), not a field-:math:`\to`-field operator, so
   the three pieces do not chain as a clean
   :class:`~orpheus.numerics.operator.OperatorProduct` of
   :class:`~orpheus.numerics.operator.LinearOperator`\ s — the
   "composition" silently changed category mid-chain. Second,
   :math:`M_\chi` is **rank-changing** (it takes one scalar density per
   cell and expands it across the :math:`n_g` emission groups), not a
   diagonal
   :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`,
   so even reading it generously it is not the multiplication operator the
   :math:`M_\bullet` notation implies. The honest form is the **two-factor
   dyad** :math:`|\chi\rangle\langle\nu\Sigma_f|`: the column and the row
   are the only two objects, and the rank-1
   :class:`~orpheus.numerics.operator.RankOneOperator` is the principled
   primitive that *is* the dyad's effect — its ``apply`` routes the matvec
   through the row functional, with no intermediate operator stages to
   materialise.

Transfer as the nonlocal-in-angle kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transfer :attr:`kernel <orpheus.transport.operators.transfer.TransferOperator.kernel>`
is the :class:`~orpheus.numerics.operator.OperatorProduct`
:math:`R \circ \Lambda \circ M` built from the SH frame's analysis
face ``frame.analysis`` (:math:`M`),
:class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer` (:math:`\Lambda`),
and the frame's reconstruction face ``frame.reconstruction``
(:math:`R`). It is the strictly-anisotropic :math:`\ell \ge 1` part of the
full ``apply``: the isotropic :math:`P_0` in-transfer and the
per-ordinate :math:`1/W` normalisation
are the **local / separate** components that live *outside* the kernel
(a strict sub-component, pinned). The kernel reproduces the existing
anisotropic moment path :math:`R(\Lambda(M\psi))` byte-for-byte (0 ULP);
its physics L1 backing is the existing anisotropic MMS gate
``tests/sn/verification/mms/test_curvilinear_aniso_scattering_p1.py``,
not a new reference.

**Both collision gains are this kernel, since #426 step 2** (2026-09-04).
The :math:`(n,2n)` doubling was an entry in the *local / separate* list
above until 2026-08-30, when the extraction gave the channel its own
operator (:ref:`scattering-binding-cs4c`) — and that operator then
carried a :math:`\Lambda` of one block, because its kernel held one
matrix and its frame was minted at :math:`L = 0`.  It now holds the
channel's whole Legendre stack at the solve's order, so
:math:`N_{2n} = \tfrac1W R\,\Lambda_{2n}\,M` is the same product as
:math:`S` over a different middle factor.  What distinguishes the two is
the yield inside :math:`\Lambda` — :math:`\Lambda_c = y_c \sum_\ell
P_\ell \otimes \Sigma_{c,\ell}` with :math:`y_S = 1`,
:math:`y_{2n} = 2` — and nothing else in this section.

.. note::

   The moment-space :math:`\ell`-sum :math:`\Lambda = \sum_\ell
   P_\ell \otimes \Sigma_{s,\ell}`
   (:eq:`scattering-as-tensor-product-sum`) is real and realized — it is
   what :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`
   *is*. ⚠ Since 2026-09-02 (#429 / ERR-080) its per-:math:`\ell` block
   contraction is selected by the RANK of the operator's angular head
   (:class:`~orpheus.numerics.spaces.moment_head.MomentHead`), because a
   1-D rule's head is FLAT and has no :math:`m` axis: the projector
   :math:`P_\ell` is ``head.degree_block(l)``, not a hard-coded
   ``[l, :2l+1]`` (:ref:`spaces-moment-head`). Lifting that shape up to the **full** kernel
   :math:`R\circ\Lambda\circ M` was **considered and rejected**. A
   :class:`~orpheus.numerics.operator.SumOfTensorProductsOperator` would
   require :math:`R` and :math:`M` to be tensor-product *factors* on
   independent axes, but they are rank-changing einsums that mix the
   ordinate and harmonic-coefficient axes — they are *not* valid
   tensor-product factors. Expressing the kernel as a
   ``SumOfTensorProductsOperator`` would be a re-derivation of the
   moment redistribution, not a re-presentation of the existing one;
   that path is tracked as the un-orphaning of
   :class:`~orpheus.numerics.operator.SumOfTensorProductsOperator` on
   #260. The :math:`1/W` per-ordinate normalisation lives *outside* the
   kernel (the kernel is the redistribution, not the :term:`quadrature`
   weighting).

.. _scattering-binding-cs4c:

The CS4c binding — what S retains, and what it is handed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above is about what :math:`S` *is*.  This subsection is
about what :math:`S` **holds** — the CS4c step-3 rebind (design record
``.claude/plans/cs4c_binding_design.md`` §§2, 4, 14), which changed the
constructor from *"a cross-section facade plus a quadrature plus an
optional space"* to *"representation-free data plus two minted arrows
plus two mandatory ends"*.  The arithmetic did not move; what moved is
which object is allowed to know what.

**The exact constructor retains exactly what the instance uses.**

.. list-table:: What ``ScatteringOperator(...)`` takes, and why
   :header-rows: 1
   :widths: 22 34 44

   * - Field
     - What it is
     - Why the operator retains it
   * - ``transfer``
       (positional, first)
     - a :class:`~orpheus.transport.material_field.TransferMaterialField`
       — the frozen per-material
       :class:`~orpheus.transport.kernels.TransferKernel` map (a Legendre
       stack **and its yield**) paired with the mesh's own material
       layout — **already brought to** this binding's Legendre order
     - the representation-free datum: what each material does, and
       where each material sits.  The order **is** the field's:
       :attr:`~orpheus.transport.operators.transfer.TransferOperator.legendre_order`
       is a derived read (``self.transfer.order``), so there is no
       second place an order could be stored and disagree.  (The
       accessor was ``scattering_order`` until #426 step 2 — a channel
       name on a channel-generic core; the solver *parameter* keeps that
       name, because it is the SCATTERING stack the clamp reads.)  Since #426
       step 2 (2026-09-04) the field is also what makes the operator
       channel-generic: the same constructor takes the scattering
       channel (:math:`y = 1`) or the :math:`(n,2n)` channel
       (:math:`y = 2`), and nothing else in the class knows which.
   * - ``flux_analysis``
       (kw-only)
     - the minted :math:`M\otimes I` face,
       ``AngularFlux → HarmonicMomentFlux``
     - the windowed bulk projection and the adjoint gates apply it
       directly; it also carries the frame (hence the measure, hence
       :math:`W`) without the operator storing either.
   * - ``source_reconstruction``
       (kw-only)
     - the minted :math:`R\otimes I` face,
       ``HarmonicMomentSourceSink → AngularSourceSink``
     - the windowed in-scatter arm's typed :math:`R`: it synthesises
       the per-ordinate source on its own bound codomain.
   * - ``domain``, ``codomain``
       (kw-only, **mandatory**, write-once)
     - the composite full-field space, both, for the shipped
       endomorphic binding
     - the
       :class:`~orpheus.transport.operators.bound_operator.BoundOperator`
       base's contract: a bound operator is an arrow and an arrow has
       two ends.  Naming both at every exact-ctor site is what makes
       the domain/codomain **swap** — the ERR-002 / ERR-076
       transposition family, which type-checks and yields a well-formed
       *reversed* arrow — unspellable-silently.

What is **gone** from that list is the point: no ``mat_xs`` facade, no
``quadrature``, no ``Optional`` space.  The facade was a whole
cross-section library where a scattering operator needs one channel;
the quadrature was a *second* source of the angular measure, sitting
beside the frame's own; and an optional space made "which space is this
operator on?" answerable with ``None``, which is how the energy-extent
guard came to be inert on the majority path elsewhere in this algebra.

**The frame is constructed outside and handed in.**  The tier-2
classmethod
:meth:`~orpheus.transport.operators.transfer.TransferOperator.from_solver_data`
takes ``(mat_xs, scattering_order, space)``, extracts **its role's**
channel, reaches the frame, mints the two faces, and **forgets the
frame**.  Since #426 step 2 that classmethod lives on the shared core
and the ROLE supplies only ``channel`` — the one-line answer to *which*
``Mixture`` channel to read — so :math:`S` and :math:`N_{2n}` mint
through one body, at one order, into one interned frame.  The blessed chain is one spelling, used by every consumer:

.. code-block:: text

   composite space → .interior_space → axes[0]
                   → axis.generator_as(Quadrature)      (the CS5 channel)
                   → Quadrature.angular_frame(L)        (interned per (rule, L))
                   → HarmonicFrame.from_galerkin(...)   (interned on the upstream frame)

wrapped as
:meth:`HarmonicFrame.for_space
<orpheus.transport.frames.harmonic_frame.HarmonicFrame.for_space>`.
Both hops intern, so "one frame per (axis content, :math:`L`)" is an
**object identity**, not a convention: :math:`S`, :math:`F`, and the
angular-windowing method minting from the same posed space receive the
same frame object and therefore the same cached projection table.  The
same-metric guarantee is likewise structural rather than checked — the
:class:`~orpheus.numerics.quadrature.Quadrature` is the single source of
the weights, and no copy exists that could drift from it.  This is why
the frame could not stay a constructor-owned cached property: a frame
minted *inside* :math:`S` is unreachable by :math:`F` and by the
windowing method, so sharing would have to be re-established by passing
:math:`S` around, which is exactly the coupling the shared factory
exists to remove.

**Why the faces sit on the constructor and the frame at tier 2** —
two independent reasons, and the second is the sharper one:

1. *The operator only needs the products.*  A frame is a factory; the
   operator's instance state is what the factory made.  Retaining the
   factory as well would be the stage-2 generator anti-pattern (keeping
   the generator beside the generated), and the accessor
   :attr:`~orpheus.transport.operators.scattering.ScatteringOperator.frame`
   costs no state at all — it reads ``flux_analysis.frame``, i.e. the
   provenance already riding on a retained product.  It is documented
   as provenance, kept for prototyping, and retirement-tracked.
2. *A frame-welded constructor forbids the negative controls its own
   correctness gate needs.*  The tightness gate below reds by
   constructing faces with a **deliberately wrong embedding**.  If the
   constructor took a frame and minted the faces itself, spelling that
   control would require doctoring a frame — a much larger and less
   honest fixture than handing in a hand-built pair of faces carrying
   the right spaces and the wrong measure.  The exact ctor is the
   cheap-fixture surface *because* it takes the products.

The tier-2/exact-ctor split then owes one gate, and gets it: every
extract-and-mint classmethod is pinned against the exact constructor on
the same inputs (``tests/transport/test_tier2_equivalence_s_family.py``),
so the hand-built fixtures the rest of the suite uses provably stand for
what production builds — the standing hazard being that a *convenience*
factory populates a field the *composite* factory forgets, and the
test-side fixture then exercises a guard production never reaches.

**The fused composites are now built once per operator, not once per
apply.**  :attr:`~orpheus.transport.operators.transfer.TransferOperator.kernel`
and
:attr:`~orpheus.transport.operators.transfer.TransferOperator.full_transfer_kernel`
are ``cached_property``: the bound field is immutable, so the cache
cannot go stale.  The step-0 execution census measured the pre-rebind
rate at up to **911** satellite :math:`\Lambda` instances minted inside
a single Krylov :math:`k`-solve (once per ``apply``); after the rebind
it is one per construction.  The measurement is worth keeping because
the *reason* is architectural rather than a missing ``lru_cache``:
before the rebind the operator held a facade and re-derived the moment
operator from it on every call, because there was no immutable datum it
could have cached against.

**The datum's shape (the O-6 landing, and what #426 step 2 did to
it).**  :class:`~orpheus.transport.material_field.TransferMaterialField`
and its :class:`~orpheus.transport.material_field.FissionMaterialField`
sibling specialise one generic base, ``MaterialField[K]``, which owns
the pairing (per-material kernel map × mesh layout), the single
per-material dispatch loop, and the ONE gathered ``(ng, ng)``
contraction primitive — with the trailing ``...`` subscript that lets a
linear-discontinuous :math:`2^d` spatial-moment axis ride through as a
broadcast spectator.  The subclasses add the *domain* vocabulary
(:meth:`~orpheus.transport.material_field.TransferMaterialField.add_p0_source`,
:meth:`~orpheus.transport.material_field.TransferMaterialField.moment_source`,
:meth:`~orpheus.transport.material_field.TransferMaterialField.add_to_group_rate`),
and the cell-index partition itself belongs to
:class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` — mesh owns
machinery, so every field over one mesh shares one ``np.where``.  This
is where the eight ``apply_*`` arms of the old cross-section facade
went: not onto the operators (which would have re-scattered them) and
not onto the kernels (which know no layout), but onto the *pairing* of
kernel and layout, which is the only object that has both.

O-6 landed **three** transfer-side subclasses where there are now one.
``ScatteringMaterialField`` and ``N2NMaterialField`` were member-for-member
twins whose only difference was that the :math:`(n,2n)` verbs multiplied
their contraction by 2 and existed for :math:`\ell = 0` alone
(``add_emission`` / ``moment_emission``).  #426 step 2 read that
difference for what it is — a **datum**, the channel's yield :math:`y`,
which the tape itself stores folded into every Legendre order
(ENDF-102 Eq. 6.1/6.3) — and collapsed both onto
:class:`~orpheus.transport.material_field.TransferMaterialField`, whose
every verb carries ``scale = self.multiplicity``.  The
:math:`y = 1` path is bit-identical by construction (the scale branch is
skipped), and the arithmetic a :math:`y > 1` channel adds on top —
production accounting :math:`(y-1)\,\Sigma_{c,0}^{T}\phi` — vanishes
for scattering by arithmetic rather than by a branch, so no verb names a
channel.  This is the type-vs-property rule applied to a *pair of
fields*: two isomorphic realizations under one morphism (scale by
:math:`y`) are one type with a property, not two types.

The tightness gate — what makes the faces the RIGHT faces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handing faces in raises a question a self-minting constructor could
pretend not to have: *what stops a caller binding a pair of faces that
are not adjoint to each other?*  The failure this guards against is
catalogued — **ERR-039**, a claimed :math:`\Pi^{*} = R` that was in
fact the addition-theorem reconstruction, i.e. an adjoint claim that
held under one embedding and not under the one that shipped.  The gate
(``tests/transport/frames/test_binding_tightness.py``) runs three legs,
and the third is a recorded blindness rather than a claim:

.. list-table:: The three legs, and what each can see
   :header-rows: 1
   :widths: 14 42 44

   * - Leg
     - Property
     - Measured (:math:`L = 1, 2, 3`)
   * - (i) Galerkin
     - the shipped analysis face's Hilbert adjoint **is** the
       reconstruction over the measure's total weight,
       :math:`M^{\dagger} = R/W`, swept as matrices under the frame's
       OWN space metrics — no hand-derived Gram
     - shipped embedding :math:`\le 1.1\times10^{-15}`; a face re-minted
       with a **constant** embedding :math:`\Sigma w/N` reads
       :math:`3.3\times10^{-1}`, an **unweighted** one :math:`8.7` —
       the G6.3 87 %-class margin.  This leg is the ERR-039 catcher.
   * - (ii) multiplicativity
     - :math:`\mathrm{bind}(K_1 K_2) = \mathrm{bind}(K_1)\,
       \mathrm{bind}(K_2)` on Funk–Hecke eigenstacks — true iff the
       frame is **tight** on the spanned harmonics
     - tight rule :math:`\le 10^{-13}`; the equispaced-equal-weight
       control :math:`\ge 10^{-3}`.  Composition of zonal kernels is
       the elementwise product of their :math:`\ell`-stacks, so the
       product operand is built gate-side from the stacks — no kernel
       product API is minted for one consumer.
   * - (iii) :math:`\ell = 0`
     - **blindness, asserted**: at :math:`L = 0` both a tight and a
       non-tight rule read clean on both legs
     - so an :math:`\ell = 0` gate discriminates nothing; the
       :math:`\ell \ge 1` rows are the coverage, and writing the
       blindness as an assertion is what stops it silently ceasing to
       be true

.. note::

   **Two negative controls were chosen by measurement, and one earlier
   choice was refuted by it.**  The pre-carve plan warned that
   ``gauss_legendre(L)`` is maximally non-tight
   (:math:`\lVert MR - I\rVert = 1`) yet **blind** on zonal
   multiplicativity — measured :math:`\le 5.9\times10^{-16}` over 200
   draws.  Re-measured *through the shipped faces* (F-0 Parseval
   metrics, producer-side :math:`1/W`) it is **not** blind in that
   spelling (relative :math:`3.0` / :math:`4.3` at :math:`L = 2, 3`),
   so the recorded hazard was a property of the probe's raw-table
   binding rather than of the gate's construction.  The equispaced rule
   remains the shipped control, on its own measured bite.  The
   disagreement is recorded rather than quietly resolved, because the
   two numbers are answers to two different questions.

.. warning::

   **There is no kernel dagger, and there will not be one absent a
   consumer that can state its hypotheses.**  The obvious spelling for
   an adjoint leg — compare ``bind(K).H`` against an independently
   assembled ``M†K†R†`` — was **refuted at design time**:
   :math:`(RKM)^{\dagger} = M^{\dagger}K^{\dagger}R^{\dagger}` is an
   algebraic identity of the metric adjoint for *any* three maps and
   *any* nondegenerate metrics, so a wrong embedding enters both sides
   and cancels.  Measured :math:`\le 2.24\times10^{-16}` under the
   correct, the constant, AND the unweighted embedding — a reading that
   cannot change, which is not evidence.  The identity is kept as a
   documented **theorem** in the gate module, never as an assertion.

   The deeper reason is worth stating, because it decides where an
   adjoint may live at all: an adjoint is defined by
   :math:`\langle A\psi,\varphi\rangle_{G_W} = \langle\psi,
   A^{\dagger}\varphi\rangle_{G_V}`, which reads **both** metrics — so
   it is a property of the *bound operator*, not of the kernel data.  A
   metric-free involution on the data does exist (swap the transfer
   matrix's indices), and it is genuinely well-defined as data; what it
   is not is an adjoint.  ``bind(swap(K)) = bind(K)†`` is a
   *conditional* theorem of the binding whose hypotheses only the
   binding can check (counting measure on energy, a tight frame on
   angle, nondegenerate metrics).  Putting a ``.dagger`` on the kernel
   would place an adjointness claim where its hypotheses are
   unknowable.  The type system already says so on the sibling channel:
   :class:`~orpheus.transport.kernels.FissionKernel` **refuses**
   ``FissionKernel(chi=νΣf, nu_sig_f=χ)`` by its own :math:`\chi`
   simplex guard — the adjoint image of a fission datum is not a
   forward fission datum, because the simplex invariant is directional
   physics.

.. _emission-kernels-btd:

The two emission kernels as a tensor decomposition (a lens, not a type)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Seen through the vocabulary of tensor decompositions, fission and
scattering are the **same object at two ranks**, and naming the relationship
sharpens why fission is the simpler one. This subsection is a *reading* —
a lens that organises the two kernels — **not** a new type to build. The
right-grained primitives already exist (the dyad
:func:`~orpheus.numerics.operator.outer`, the
:class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame`, and the
``⊗`` / :class:`~orpheus.numerics.operator.TensorProductOperator`); the
decomposition vocabulary names what they already *are*.

.. list-table:: The emission kernels in tensor-decomposition vocabulary
   :header-rows: 1
   :widths: 18 30 52

   * - Kernel
     - Decomposition
     - Reading
   * - **Fission** :math:`F`
     - rank-1 (CP atom)
       :math:`|\chi\rangle\langle\nu\Sigma_f|`
     - A single canonical-polyadic term: one emission column
       :math:`\chi` against one production row :math:`\nu\Sigma_f`. The
       *rank-1 degenerate* of the scattering sum below — fission is the
       :math:`\ell=0`, one-mode case.
   * - **Scattering** :math:`S_{\rm aniso}`
     - orthogonal-CP / spectral sum
       :math:`\sum_\ell |Y_\ell\rangle\,\sigma_\ell\,\langle Y_\ell|`
     - A *sum* of rank-1 dyads in the spherical-harmonic eigenbasis
       (Funk–Hecke): the modes :math:`|Y_\ell\rangle` are orthogonal and
       the weights :math:`\sigma_\ell = \Sigma_{s,\ell}` are the Legendre
       moments — the spectral theorem :math:`U\Sigma U^*`, managed by the
       Frame (:math:`R\circ\Lambda\circ M`).
   * - **Full scattering** :math:`S`
     - block-term decomposition (BTD)
     - rank-1 in **angle** :math:`\otimes` a *full-rank* **energy**
       transfer :math:`\otimes` diagonal in **space**. The energy block
       is genuinely dense (group-to-group transfer is not low-rank);
       fission is its CP-rank-1 degenerate (one energy mode, one angle
       mode).

The collision operator :math:`C = M[\sigma_t]` completes the picture as
the **diagonal** term (:eq:`multiplication-operator-action`, the §5.7
multiplication operator): no decomposition, pointwise in every axis. So
the transport algebra's emission/loss structure reads as one ladder —
diagonal (collision) :math:`\to` rank-1 CP atom (fission) :math:`\to`
orthogonal-CP spectral sum (anisotropic scattering) :math:`\to` block-term
decomposition (full scattering).

.. note::

   **Since CS4c step 4 the "degenerate" reading is no longer only a
   lens — the two kernels are bound through literally the same
   faces.**  When this subsection was written, "fission is scattering's
   rank-1 degenerate" organised two operators that shared *vocabulary*
   and no code.  The step-4 rebind made the sentence structural: on the
   angular composite the two bindings are

   .. math::

      S \;=\; \tfrac{1}{W}\,R\,\Lambda_{\ell\le L}\,M ,
      \qquad
      F \;=\; \tfrac{1}{W}\,R_0\,
              \bigl(|\chi\rangle\langle\nu\Sigma_f|\bigr)\,M_0 ,

   — the **same** analysis/reconstruction faces, minted from the **same
   hub-interned** :class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame`
   (so an :math:`S` and an :math:`F` posed on one space share one
   metric by construction, not by convention), with fission taking one
   moment instead of :math:`(L{+}1)^2` and a rank-1 energy factor
   instead of a dense transfer stack.  :math:`N_{2n}` is the third
   member of the same shape, differing from :math:`F` only in that its
   :math:`\ell = 0` energy factor is a full transfer matrix
   :math:`\nu_{2n}\Sigma_{2n}^{\mathsf T}` rather than a dyad.  The
   consequence that matters is the adjoint: all three transposes are
   **one factor reversal** of the conjugated product, with no
   channel-specific :math:`w`-arithmetic anywhere
   (:ref:`sn-fission-binding-adjoint`).

   The guardrails below are unaffected — they forbid *fitting* the
   factors and *minting an umbrella type*, and neither is what the
   shared faces do.  What changed is that the "two realisations, named
   honestly" the last guardrail appeals to are now demonstrably one
   realisation used at two ranks.

.. warning::

   **The decomposition is a lens; the guardrails are load-bearing.** This
   framing must NOT be turned into machinery:

   * **Keep the energy block dense.** The group-to-group transfer is
     genuinely full-rank; do not attempt a low-rank energy factorisation.
   * **Do not import general-CP / tensor-fitting.** ALS / CP-decomposition
     *fitting* (the numerical recovery of unknown factors) is foreign to
     this algebra — the factors here are *known by physics* (:math:`\chi`,
     :math:`\nu\Sigma_f`, the :math:`Y_\ell` eigenbasis), not fitted.
   * **Do not mint a ``CPOperator`` or a polyadic umbrella type.** The dyad
     :func:`~orpheus.numerics.operator.outer`, the
     :class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame`, and
     ``⊗`` are already the right-grained primitives; an umbrella type would
     add ceremony and a conversion seam without making any illegal state
     unrepresentable (``coding-elegance`` type-vs-property: a separate type
     earns its existence only with :math:`\ge 2` non-isomorphic realisations
     under a non-identity morphism — the dyad and the Frame already *are*
     those two realisations, named honestly).

.. (vv-status rationale) Conceptual lens organising the two emission
   kernels in tensor-decomposition vocabulary; not a type and not a solver
   claim. The verifiable content is the fission dyad's closed-form k∞ gate
   and the scattering kernel's anisotropic MMS gate, both cited above.

.. _scattering-carrier-grid:

The completed carrier grid — the four leaves and the three edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`R \circ \Lambda \circ M` kernel above reads cleanly as an
:class:`~orpheus.numerics.operator.OperatorProduct` of three ``np.ndarray``
maps, but each factor crosses between two **typed transport carriers** —
the per-ordinate flux and its harmonic moments, in their flux and
source/sink roles. The Frame campaign's P4 phase closed those carriers
into a complete :math:`(\text{angular} \otimes \text{moment}) \times
(\text{flux} \otimes \text{source})` grid: four leaf types and the three
edges that map between them.

.. math::
   :label: scattering-carrier-grid

   \begin{array}{ccc}
     \texttt{AngularFlux}
       & \xrightarrow{\;\;M\;\;}
       & \texttt{HarmonicMomentFlux} \\[2pt]
     & & \big\downarrow{\scriptstyle\;\Lambda} \\[2pt]
     \texttt{AngularSourceSink}
       & \xleftarrow{\;\;R\;\;}
       & \texttt{HarmonicMomentSourceSink}
   \end{array}

.. implements:: scattering-carrier-grid
   :by: orpheus.transport.frames.harmonic_frame.HarmonicAnalysisOperator.apply

   **Implemented by** the diagram's top edge :math:`M` — since F-1 the
   MINTED analysis face's ``apply`` (carrier-generic over the flux/source
   columns; the flux instance is the one minted today). Role-**preserving**
   and typed as such by the face's own generics: flux in, flux out,
   representation changed.

.. implements:: scattering-carrier-grid
   :by: orpheus.transport.operators.transfer.LegendreMomentTransfer.apply

   **Implemented by** the diagram's vertical edge :math:`\Lambda` — and
   the only role-**changing** one:
   :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
   in, ``HarmonicMomentSourceSink`` out. "Scattering turns flux into source"
   is a signature here, not a comment.

.. implements:: scattering-carrier-grid
   :by: orpheus.transport.frames.harmonic_frame.HarmonicReconstructionOperator.apply

   **Implemented by** the bottom edge :math:`R` — since F-1 the MINTED
   reconstruction face's ``apply`` (the source instance is the one minted
   today), role-preserving like :math:`M`, carrying the source leg back to
   the per-ordinate representation.

   The diagram's four **nodes** are types, not code — only the three
   **edges** have implementers, which is exactly the split
   :eq:`carrier-grid-cell` records as un-implementable.

.. (vv-status rationale) The carrier-grid layout: a named-field-typing
   identity (which carrier type sits at each node, and the role/axis
   semantics of each edge). Not a solver claim; the verifiable content is
   the role/class-identity algebra of the four leaves (the foundation
   tests ``tests/sn/primitives/test_typed_source_sinks.py`` ::
   ``TestHarmonicMomentSourceSink`` and ``tests/transport/frames/
   test_harmonic_frame.py``) and the 0-ULP equivalence of the typed and
   ndarray scattering arms (``test_scattering_kernel_crosscheck.py``).
.. vv-status: scattering-carrier-grid documented

The **two vertical axes** of the grid are the representation (angular ↔
moment) and the role (flux ↔ source). The four leaves and three edges:

.. list-table:: The four carriers and the three edges of the scattering grid
   :header-rows: 1
   :widths: 22 30 48

   * - Carrier / edge
     - Type
     - What it is
   * - top-left leaf
     - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
     - The per-ordinate flux :math:`\psi_n(\vec r, g)` — an element of
       the flux vector space :math:`V`, physically in its positive cone
       :math:`K` (:ref:`cone-ordered-vector-space`).
   * - top-right leaf
     - :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
     - The flux moments :math:`\phi_\ell^m(\vec r, g)` — the same flux
       state in moment space (``(<angular head>, ng, *spatial)``:
       ``(L+1, 2L+1)`` on a rule that binds the harmonics, ``(L+1,)`` on
       a 1-D rule since 2026-09-02 — :ref:`spaces-moment-head`). A
       ``MomentField`` carrier; ``flux units`` =
       :data:`~orpheus.numerics.units.SCALAR_FLUX_UNITS` (a moment is
       angle-integrated, so the :math:`\ell=0` block **is** the scalar
       flux exactly).
   * - bottom-right leaf
     - :class:`~orpheus.transport.source_sinks.harmonic_moment_source_sink.HarmonicMomentSourceSink`
       (bare ``MomentField``)
     - The **scattered in-scatter source** moments — the output of
       :math:`\Lambda`. A *rate density*, so it adds vectorially
       (``source + source`` is CLOSED);
       :data:`~orpheus.numerics.units.SCALAR_RATE_UNITS`. The P4 leaf that
       gave the flux→source role change a *home* (before it, the role
       change leaked to the scattering consumer as a raw ``np.ndarray``).
   * - bottom-left leaf
     - :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`
       (bare)
     - The per-ordinate in-scatter source :math:`Q_n(\vec r, g)` the
       sweep consumes — the bottom of the chain.
   * - :math:`M` (left edge)
     - :class:`HarmonicAnalysisOperator
       <orpheus.transport.frames.harmonic_frame.HarmonicAnalysisOperator>`
       (minted: ``S.flux_analysis``)
     - **Role-preserving, axis-changing.** Projects per-ordinate →
       moment, flux→flux *and* source→source (the minted, bound face over
       the Galerkin frame's analysis — the canonical pure-Galerkin
       :math:`\Pi`).
   * - :math:`\Lambda` (right edge)
     - :meth:`LegendreMomentTransfer.apply <orpheus.transport.operators.transfer.LegendreMomentTransfer.apply>`
     - **Role-changing, axis-preserving.** The *sole* role-changing edge:
       the per-:math:`\ell` group-transfer :math:`\Sigma_{s,\ell}` maps a
       :class:`HarmonicMomentFlux` to the
       :class:`HarmonicMomentSourceSink` it emits — flux → source, both in
       moment space.
   * - :math:`R` (bottom edge)
     - :class:`HarmonicReconstructionOperator
       <orpheus.transport.frames.harmonic_frame.HarmonicReconstructionOperator>`
       (minted: ``S.source_reconstruction``)
     - **Role-preserving, axis-changing.** Reconstructs moment →
       per-ordinate, flux→flux *and* source→source (the reconstruction
       face; the addition-theorem :math:`R`, not the Hilbert adjoint
       :math:`M^*` — under the frame's Parseval metric the two differ by
       exactly the total weight, :math:`R = W\,M^*`; see
       :ref:`frame-parseval-metric`).

The kernel of the previous subsection is the composite of these three
edges plus the producer-side normalisation,

.. (vv-status rationale) The carrier-grid composite
   S_aniso = (1/W)(R∘Λ∘M). Representational (named-field-typing)
   identity, matching the sentineled scattering-carrier-grid /
   carrier-grid-* siblings; the bit-identical composed kernel is pinned
   by test_scattering_kernel_crosscheck.
.. vv-status: scattering-aniso-composite documented

.. math::
   :label: scattering-aniso-composite

   S_{\rm aniso} \;=\; \tfrac{1}{W}\,(R \circ \Lambda \circ M)
   \;:\; \texttt{AngularFlux} \longrightarrow \texttt{AngularSourceSink},

a flux at the top-left mapped all the way round the grid to the source
at the bottom-left. The role only ever changes **once**, at :math:`\Lambda`
— that is the physical content of "scattering turns flux into source", now
visible in the type signatures rather than buried inside an ndarray chain.
The *same* :math:`M` and :math:`R` carry both the flux leg (top edge /
bottom edge, flux side) and the source leg the moment-end binding uses
below: the faces are minted per role from one interned frame, and the
role transition itself is the carriers' declared partnership
(:ref:`role-partner-declaration`) rather than a polymorphic verb.

.. implements:: scattering-aniso-composite
   :by: orpheus.transport.operators.transfer.TransferOperator._redistribute_ordinates

   **Implemented by** literally ``self.kernel.apply(bulk.values)
   / self.total_weight`` — i.e.
   :math:`\tfrac{1}{W}(R\circ\Lambda\circ M)`, the equation with its
   producer-side normalisation applied at the ``apply`` boundary.
   :math:`W` is read off the retained faces' frame measure
   (``flux_analysis.frame.measure``), not off a stored weight vector:
   the measure IS the binding's metric, so there is one place it can
   come from (:ref:`scattering-binding-cs4c`).

.. implements:: scattering-aniso-composite
   :by: orpheus.transport.operators.transfer.TransferOperator.kernel

   **Implemented by** the ``frame.conjugate(Λ)`` composite — the
   :math:`R\circ\Lambda\circ M` product **before** the :math:`1/W`, because
   the kernel is the redistribution and the :term:`quadrature` weighting
   lives outside it by design.

.. implements:: scattering-aniso-composite
   :by: orpheus.transport.operators.transfer.TransferOperator._redistribute_moments

   **Implemented by** the second, deliberately-kept realization — the
   **moment-end** binding's interior body, selected at construction
   (:ref:`cs4c-ends-select-the-body`). It spells the composite explicitly
   on the typed carriers: ``self._moment_transfer(skip_l0=True).apply(...)``
   maps flux moments to a typed
   :class:`~orpheus.transport.source_sinks.HarmonicMomentSourceSink` (the
   role-changing edge, in the signature), then
   ``self.source_reconstruction.apply(emitted) / self.total_weight``
   synthesises the per-ordinate source on the face's own bound codomain.
   Both factors are the operator's own retained objects since the CS4c
   rebind: :math:`\Lambda` is minted on the bound kernel field, and
   :math:`R` is the **retained face**, applied on its own bound
   codomain rather than through a frame verb. Its angular-end sibling
   ``_redistribute_ordinates`` runs the fused
   :attr:`kernel <orpheus.transport.operators.transfer.TransferOperator.kernel>`
   declared above; which one exists on a given operator is fixed in
   ``__post_init__``, and neither is spelled when the binding
   :attr:`~orpheus.transport.operators.transfer.TransferOperator.is_isotropic`.

   ⛔ **This declaration named** ``TransferOperator._apply_impl`` **until
   CS4c step 5 (2026-09-04)** — the same body, then reached as a
   registered ``singledispatchmethod`` arm on the *angular*-bound
   operator when a caller happened to hand it a moment carrier. It is now
   the production body of an operator BOUND on the moment end, which is
   why the declaration can name a method instead of a dispatcher
   (:ref:`pattern-m-history`).

   Both routes are production **by design** — the moment-end binding and
   the angular-end fused kernel — and their 0-ULP agreement is the
   interchange-law coherence witness of
   :ref:`carrier-grid-double-category`, not a redundancy waiting to be
   retired.

This 2×2 scattering square is one face of a larger structure. The next
three subsections lift it to the full :math:`(\text{Representation} \times
\text{Role})` carrier grid, identify that grid as a **double category**
whose 2-cell IS :math:`\texttt{frame.conjugate}(\Lambda)`, and then derive
the load-bearing architectural consequence: why the flat
multiple-inheritance leaves the grid is built from are not a workaround but
the **unique principled normal form**, and why the genericity the grid
expresses belongs on the *operator* (:math:`[\textsf{Domain},
\textsf{Codomain}]`), never on the carrier.


.. _carrier-grid-double-category:

The carrier grid is a double category — two kinds of morphism, one 2-cell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scattering square above is not a special diagram — it is one cell of a
grid that every transport carrier and every transport operator inhabits.
A carrier is a pair

.. math::
   :label: carrier-grid-cell

   \texttt{Carrier} \;=\; (\,\text{Representation},\ \text{Role}\,),


.. no-implementation:: carrier-grid-cell
   :kind: definition

   **Nothing implements this** — it DEFINES what a carrier is. Code inhabits
   the grid; nothing computes the pair. ⚠ Note the discrimination against
   its own siblings: :eq:`carrier-grid-operator-typing` and
   :eq:`scattering-carrier-grid` carry the same *"not a solver claim"*
   rationale and ARE declarable, because a typing RULE has a materialized
   carrier (a class, a parameter list, typed methods) where a definition of
   the vocabulary does not.

and the two coordinates are **independent and orthogonal**, each governing
a different facet of the object:

* **Representation** :math:`\in \{\text{Angular},\ \text{Moment},\
  \text{Scalar},\ \text{Trace}\}` sets the **array shape** and carries the
  change-of-basis. Angular is the per-ordinate :math:`(N, n_g, *\text{spatial})`
  layout; Moment is the harmonic-coefficient
  :math:`(L{+}1, 2L{+}1, n_g, *\text{spatial})` layout; Scalar is the
  angle-integrated :math:`(n_g, *\text{spatial})` layout; Trace is the
  boundary face-cochain (the flat :math:`(\,\text{layout.total\_size}\,)`
  buffer on the :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`).
  Changing representation is a change of basis between two *realisations of
  the same physical quantity* — the addition theorem :math:`M`/:math:`R`
  between per-ordinate and moment angular space, the angular integral
  between Angular and Scalar. This axis is the storage-family ABC layer of
  :mod:`orpheus.transport.fields._bases`
  (:class:`~orpheus.transport.fields._bases.AngularField` /
  :class:`~orpheus.transport.fields._bases.MomentField` /
  :class:`~orpheus.transport.fields._bases.ScalarField` /
  :class:`~orpheus.transport.fields._bases.AngularBoundaryField`).

* **Role** :math:`\in \{\text{Flux},\ \text{Source},\
  \text{Residual}\}` sets the **arithmetic interface** — the field
  algebra of :ref:`cone-typed-field-algebra`. All three are *vectors*:
  ``flux + flux``, ``source + source`` and ``residual + residual`` are
  each closed (:eq:`flux-vector-algebra`), and a flux difference is the
  flux type carrying a signed value. What the role decides is **which
  additions mean something**: the class identity *is* the units identity
  (one ``UNITS`` constant per leaf), so the runtime gate
  :meth:`Field._check_partner
  <orpheus.numerics.field.Field._check_partner>` refuses every
  cross-role pair; and the Source role additionally owns a
  *cross-representation* ``__add__`` (the isotropic→per-ordinate
  containment injection). This axis is the role-**leaf** layer — the
  concrete classes themselves, since only
  :class:`~orpheus.transport.fields._coefficient_role.CoefficientRole`
  still carries a role mixin.

  ⛔ Until 2026-08-19 this bullet read *"Flux is an affine* **point**
  *(*``flux − flux → Displacement``*, and* ``flux + flux`` *is a*
  :class:`TypeError`*)"* over a four-role set. The affine ontology was
  overturned at campaign-1 CS3 — see
  :ref:`cone-the-overturned-affine-design`.

The two axes are genuinely orthogonal: a representation change preserves
role (the addition theorem maps a flux to a flux and a source to a source
— it never turns a flux into a source), and a role change preserves
representation (scattering a moment flux to a moment source stays in
moment space). That orthogonality is exactly the structure of a **double
category**, and naming it that way is not decoration — it tells you which
generic each morphism is, and it identifies a coherence theorem the code
already pins to 0 ULP.

.. list-table:: The carrier grid as a double category
   :header-rows: 1
   :widths: 24 30 46

   * - Categorical part
     - In the carrier grid
     - Consequence for the code
   * - **Objects** (0-cells)
     - The grid cells :math:`(\text{Representation}, \text{Role})` — the
       :math:`\approx 4 \times 4` leaf types
       (:class:`~orpheus.transport.fields.angular_flux.AngularFlux`,
       :class:`~orpheus.transport.source_sinks.harmonic_moment_source_sink.HarmonicMomentSourceSink`,
       …), each a 2-line MI binding ``Leaf(RoleMixin, RepBase)``.
     - There is **no cell-by-cell duplication** — each leaf is the
       intersection of one role mixin and one representation base. The
       grid is already its own normal form (see
       :ref:`carrier-grid-flat-leaf-normal-form`).
   * - **Horizontal 1-morphisms**
     - **Representation-changes** — the frame faces :math:`M` (analysis)
       and :math:`R` (reconstruction), built from the
       :math:`(\text{basis}, \text{measure})` pair that *is* the Frame
       (:ref:`scattering-carrier-grid`). A horizontal arrow fixes the
       Role coordinate.
     - A base change that **fixes the fiber** ⟹ :math:`M`/:math:`R` are
       **role-generic**: the *same* analysis face projects a flux to a
       flux and a source to a source. This is why the frame verbs are
       role-polymorphic, and why the role-changing edge is deliberately
       **not** a frame verb.
   * - **Vertical 1-morphisms**
     - **Role-changes** — the cross sections :math:`C = \sigma_t`
       (collision), :math:`\Lambda = \Sigma_{s,\ell}` (the per-:math:`\ell`
       group transfer), :math:`F = \chi \otimes \nu\Sigma_f` (fission). A
       vertical arrow fixes the Representation coordinate.
     - A fiber morphism **identical over every base** ⟹ the cross sections
       are **representation-generic**: the role change *is* the
       cross-section physics (flux → emitted source), and it carries the
       same meaning whether applied in angular, moment, or scalar
       representation. "Scattering turns flux into source" is a vertical
       arrow.
   * - **The 2-cell**
     - **Scattering** :math:`S_{\rm aniso} = \tfrac{1}{W}\,(R \circ
       \Lambda \circ M)` — the vertical 1-morphism :math:`\Lambda`
       **conjugated by the horizontal adjoint pair** :math:`M`/:math:`R`.
       Realised as :meth:`frame.conjugate(Λ) <orpheus.numerics.frame.FrameBase.conjugate>`
       :math:`=` ``OperatorProduct(R, OperatorProduct(Λ, M))``.
     - The 2-cell fills the square: it is the canonical conjugation of a
       vertical morphism by the horizontal frame. The role change stays
       **localized at** :math:`\Lambda` (:math:`M`/:math:`R` preserve
       role), so :math:`S` is honestly an operator from the
       :math:`(\text{Angular}, \text{Flux})` cell to the
       :math:`(\text{Angular}, \text{Source})` cell.

.. (vv-status rationale) The double-category reading of the carrier grid:
   a structural / categorical identity naming which morphism class each
   operator belongs to (horizontal = representation-change, vertical =
   role-change, scattering = the 2-cell). Not a solver claim; the
   verifiable content is the 0-ULP interchange-coherence identity
   (``tests/sn/operators/test_frame_conjugate_carve.py`` ::
   ``TestFrameConjugateEqualsRLambdaM`` +
   ``test_kernel_property_is_frame_conjugate_of_lambda``) plus the
   role/class-identity algebra of the leaves (the foundation tests cited
   at :eq:`scattering-carrier-grid`).
.. vv-status: carrier-grid-cell documented

The interchange law is a theorem the code already pins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A double category's **interchange law** states that the 2-cell may be
read either way round the square and the two readings agree. For the
scattering 2-cell this is the statement that the *single composed*
operator :math:`R \circ \Lambda \circ M` and the *step-by-step* typed
evaluation (project with :math:`M`, scatter with :math:`\Lambda`,
reconstruct with :math:`R`) compute the **same** per-ordinate source —
not approximately, but to the last bit. That is exactly what
``tests/sn/operators/test_scattering_kernel_crosscheck.py`` and
``tests/sn/operators/test_frame_conjugate_carve.py`` assert with
``np.array_equal`` (0 ULP, **not** ``allclose``):

.. math::
   :label: carrier-grid-interchange-witness

   \underbrace{\big(\texttt{frame.conjugate}(\Lambda)\big).\texttt{apply}(\psi)}_{\text{single composed 2-cell}}
   \;\;\equiv\;\;
   \underbrace{R\big(\Lambda\,(M\,\psi)\big)}_{\text{step-by-step horizontal·vertical·horizontal}}
   \qquad(0\ \text{ULP}).

.. (vv-status rationale) The interchange-coherence identity: the composed
   2-cell equals the step-by-step horizontal/vertical/horizontal reading
   bit-for-bit. The verifiable content is the 0-ULP ``np.array_equal``
   crosscheck (``test_scattering_kernel_crosscheck.py``, the definitional
   identity of :attr:`ScatteringOperator.kernel`, and
   ``test_frame_conjugate_carve.py`` ::
   ``test_conjugate_equals_manual_R_A_M_nesting``). Not a solver claim —
   a structural equivalence between two evaluations of one operator.
.. vv-status: carrier-grid-interchange-witness documented

The bit-identity is the point. The two sides of
:eq:`carrier-grid-interchange-witness` share the *same* :math:`\Lambda`
kernel and the *same* frame :math:`R` face, so their agreement is a
**coherence theorem of the double category** — it holds by construction,
not by numerical coincidence — and the 0-ULP gate is its
**interchange-law coherence witness**. This is why the crosscheck is a
``np.array_equal`` *definitional* identity rather than an ``allclose``
*regression* tolerance: a tolerance would admit two genuinely different
reduction trees agreeing only to round-off, which would mean the square
does not actually commute. The 0 ULP says it commutes exactly — the mark
of a real 2-cell.

.. note::

   **An equivalent reading: a category fibered over Representation.** The
   same structure can be stated as a *fibered category* (a Grothendieck
   fibration) :math:`p : E \to B` with base :math:`B =` Representation and
   the **Role as the fiber coordinate** (:ref:`cone-typed-field-algebra`;
   until 2026-08-19 this read "carrying a **torsor on the Flux fiber**",
   which the CS3 overturn retired — the fiber is a vector space). In this
   reading a
   role-change (:math:`\Lambda`, :math:`C`, :math:`F`) is a *cartesian
   morphism within a fiber* (fixed representation), and a
   representation-change (:math:`M`, :math:`R`) is a *base change* lifting
   to the total space. :math:`M` is role-generic **because a base change
   fixes the fiber coordinate and lifts uniformly**; :math:`\Lambda` is
   representation-generic **because the fiber morphism is the same over
   every base point**. The double-category and fibration pictures describe
   the same code; the double category makes the 2-cell (scattering)
   explicit, the fibration makes the role-preservation a *theorem* (base
   change fixes the fiber) rather than a per-operator assertion.


.. _carrier-grid-domain-codomain-identity:

The key identity — a grid cell IS an operator's ``(Domain, Codomain)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The grid has two coordinates on the *carrier*; an operator has two
coordinates too — its **domain** and its **codomain**. These are the same
two coordinates. An operator is an arrow from one grid cell to another,
and naming its endpoints names the morphism completely:

.. math::
   :label: carrier-grid-operator-typing

   \texttt{LinearOperator[Domain, Codomain]}
   \;\;\text{IS the typed traversal of the grid:}\quad
   \begin{aligned}
     M &: \texttt{LinearOperator[AngularFlux,\ HarmonicMomentFlux]}
        &&\text{(horizontal)}\\
     \Lambda &: \texttt{LinearOperator[HarmonicMomentFlux,\ HarmonicMomentSourceSink]}
        &&\text{(vertical)}\\
     S &: \texttt{LinearOperator[AngularFlux,\ AngularSourceSink]}
        &&\text{(the 2-cell)}.
   \end{aligned}

.. implements:: carrier-grid-operator-typing
   :by: orpheus.numerics.operator.LinearOperator

   **Implemented by** ``class LinearOperator(Protocol[Domain,
   Codomain])`` with ``apply(x: Domain) -> Codomain``: the typing rule
   written in code. The **operator** carries the two grid coordinates and
   the carrier does not — see :ref:`carrier-grid-flat-leaf-normal-form` for
   why a fully-typed ``Carrier[Representation, Role]`` is structurally
   impossible.

.. implements:: carrier-grid-operator-typing
   :by: py:data:orpheus.numerics.operator.Domain

   **Implemented by** the type variable that names the **domain**
   coordinate: ``Domain = TypeVar("Domain", bound=Vector)``, the operator's
   input cell.

.. implements:: carrier-grid-operator-typing
   :by: py:data:orpheus.numerics.operator.Codomain

   **Implemented by** the type variable that names the **codomain**
   coordinate: ``Codomain = TypeVar("Codomain", bound=Vector,
   default=Domain)``. The PEP-696 default is what keeps
   ``LinearOperator[V] ≡ LinearOperator[V, V]`` for the endomorphic
   majority, so making the grid's second coordinate explicit costs the
   endomorphic leaves nothing.

   ⚠ Both type variables are ``py:data`` nodes in the knowledge graph, not
   classes — hence the explicit ``py:data:`` node-id prefix on the two
   ``:by:`` targets above. A bare dotted name is resolved only against the
   function / method / class prefixes and would silently fail to bind.

.. (vv-status rationale) The operator-typing identity: an operator's two
   type parameters ARE the two grid cells it maps between. A
   representational/structural statement about where the parametrization
   lives (on the operator, not the carrier). The verifiable content is the
   static ``assert_type`` pins on the heteromorphic ``apply``
   (``tests/sn/operators/test_operators_apply_typed.py``) and the
   composition-guard tests; not a solver claim.
.. vv-status: carrier-grid-operator-typing documented

The two-parameter operator type
:class:`~orpheus.numerics.operator.LinearOperator` (``Protocol[Domain,
Codomain]``, :ref:`heteromorphic-apply-typing`) is therefore the **right
and complete machinery for traversing the grid**. Its ``apply`` maps an
input carrier :data:`~orpheus.numerics.operator.Domain` to a
(possibly distinct) output carrier
:data:`~orpheus.numerics.operator.Codomain` — exactly an arrow
:math:`(\text{Rep}_{\rm in}, \text{Role}_{\rm in}) \to (\text{Rep}_{\rm
out}, \text{Role}_{\rm out})`. The endomorphic majority (collision
:math:`C`, identity, the ``np.ndarray`` serialization boundary) is the
special case :math:`\textsf{Codomain} = \textsf{Domain}`, recovered for
free by the PEP-696 default ``Codomain = TypeVar("Codomain",
default=Domain)`` so that ``LinearOperator[V]`` :math:`\equiv`
``LinearOperator[V, V]`` and the endomorphic call sites need no change.

The names are spelled in full — :data:`~orpheus.numerics.operator.Domain`
and :data:`~orpheus.numerics.operator.Codomain`, not abbreviations —
because ``Domain`` already reads as "the input" and ``Codomain`` as "the
output"; the morphism vocabulary is the domain vocabulary here.

.. important::

   **The parametrization belongs on the operator, NOT on the carrier.**
   This is the load-bearing architectural decision the grid forces. One
   could imagine pushing the :math:`(\text{Representation}, \text{Role})`
   pair *onto the carrier* as type parameters —
   ``Carrier[Representation, Role]`` — and writing operators generic in
   both axes. That design is **structurally impossible in Python** and
   would break a runtime safety gate even where it type-checks; the next
   subsection is the full argument. The realised design puts the two
   coordinates where they are expressible and where they belong: on the
   **operator's** ``[Domain, Codomain]``, with the carriers as the flat
   intersection leaves the grid cells already are.


The HarmonicFrame typed seam — and why it lives in transport, not numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The casting between the generic ``np.ndarray`` frame faces and the typed
:class:`Field` carriers is the load-bearing design decision of P4, and it
is forced to live **one layer up from the frame itself**. The argument is
a layering constraint, not a preference:

* The generic angular spherical-harmonic frame is the numerics
  :class:`~orpheus.numerics.frame.GalerkinFrame` built by
  :meth:`~orpheus.numerics.quadrature.Quadrature.angular_frame` — its
  :attr:`~orpheus.numerics.frame.FrameBase.analysis` /
  :attr:`~orpheus.numerics.frame.FrameBase.reconstruction` faces are
  **carrier-agnostic** ``np.ndarray → np.ndarray`` maps. This is by
  design: the same generic faces are shared with the P3
  indicator-homogenisation frame, and keeping them ndarray-valued is what
  makes that sharing 0-ULP-safe.
* The two carriers the angular frame maps between
  (:class:`AngularFlux` ↔ :class:`HarmonicMomentFlux`, and their
  source/sink siblings) share their *deepest* primitive,
  :class:`~orpheus.numerics.field.Field`, in **numerics**.
* But the part that makes them **castable** — the concrete leaf CLASSES
  themselves, plus the moment family's keyed
  :meth:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux.from_mesh_and_L`
  factory, which is what builds the typed carrier from a raw array —
  lives in the transport
  :class:`~orpheus.transport.fields._bases.BulkField` hierarchy,
  **above** numerics. (Until CS4b S5 that clause also named a ``mesh``
  binding and a ``from_mesh`` sugar factory on the base; both retired —
  a leaf now carries ``values`` and a numerics
  :class:`~orpheus.numerics.space.FunctionSpace` and nothing else. The
  layer argument is unaffected, because it is the leaf *classes* that
  are transport-level, not the key their factories take.)
  And :meth:`Quadrature.angular_frame <orpheus.numerics.quadrature.Quadrature.angular_frame>`
  is in numerics, which *cannot* import the transport carriers without
  inverting the layer order.

So a generic numerics face **cannot** return a typed
:class:`HarmonicMomentFlux`: the cast needs the transport-layer factories,
and numerics is below transport. The clean home for the casting is
therefore the transport layer, in
:class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame`:

.. (vv-status rationale) The Liskov typing identity HarmonicFrame IS-A
   GalerkinFrame (the angular SH projection is the canonical
   pure-Galerkin frame). Structural / representational identity; the
   from_galerkin faces are bit-identical to the generic numerics frame's.
.. vv-status: harmonic-frame-is-galerkin documented

.. math::
   :label: harmonic-frame-is-galerkin

   \texttt{HarmonicFrame} \;\text{IS-A}\; \texttt{GalerkinFrame}
   \qquad(\text{Liskov — the angular SH projection is the canonical
   pure-Galerkin frame}),

constructed from the generic frame's basis + measure via
:meth:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame.from_galerkin`
(``cls(frame.basis, frame.measure)`` — **zero** rebuild of the basis /
measure / projection table, so the inherited ndarray faces and the §5.6
:attr:`kernel <orpheus.transport.operators.scattering.ScatteringOperator.kernel>` stay
bit-identical), adding **only** the MINT verbs
:meth:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame.flux_analysis_on`
(:math:`M`, minted) and
:meth:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame.source_reconstruction_on`
(:math:`R`, minted) — since F-1 the carrier typing lives on the minted
FACES, not on frame verbs. The generic numerics faces are untouched. This is the
"shared primitive is :class:`Field` in numerics, but castability is
:class:`BulkField` in transport, so the typed seam lives in transport"
layering — the same reason a typed wrapper, not the generic frame, owns
the carrier verbs.

.. implements:: harmonic-frame-is-galerkin
   :by: orpheus.transport.frames.harmonic_frame.HarmonicFrame

   **Implemented by** ``class HarmonicFrame(GalerkinFrame)`` — the
   subtyping relation *is* the Liskov claim; there is nothing else to
   write.

.. implements:: harmonic-frame-is-galerkin
   :by: orpheus.transport.frames.harmonic_frame.HarmonicFrame.from_galerkin

   **Implemented by** the operative content — ``return cls(basis,
   frame.measure)``, after an upgrade-boundary guard that rejects a trial
   basis carrying **no truncation order** *there* rather than later, when
   a mint first reads :math:`L`. The claim is worth stating only
   because the construction rebuilds **nothing**: basis, measure and
   projection table are carried over, so the inherited ndarray faces and the
   §5.6 kernel stay bit-identical and the IS-A is substitutability in fact,
   not merely in the type checker.

   ⚠ **This clause named ONE CLASS until 2026-09-02** — it read *"rejects
   a non-*\ ``SphericalHarmonicBasis``\ * trial basis … the SH-only
   truncation order* :math:`L`\ *"*, which was an accurate description of
   the guard and a false description of the concept. #429 tracker 2.5
   replaced both ``isinstance`` doors with a demand for the
   :class:`~orpheus.numerics.basis.base.TruncatedBasis` **surface**
   (``L`` + ``space``): :math:`L` is not SH-only — it is the harmonic
   *family's* truncation order, shared by the σ-even restriction a folded
   rule binds and by the Legendre basis on :math:`S^2/O(2)_a` that
   tracker 3.4 bound on a 1-D one the same day. The same step made every operator
   end and every moment-field head READ the bound basis's coefficient
   space instead of re-minting it from the integer, so the family the
   quadrature chooses propagates by construction. See
   :ref:`frame-moment-space-single-home` for the eight-homes census, the
   ``basis.space``-vs-``basis_space`` fork with its measurements, and the
   gates.

.. note::

   The role-changing edge :math:`\Lambda` is deliberately **not** a frame
   verb. A frame's faces are role-*preserving* changes of representation
   (flux↔flux, source↔source); the flux→source role change is *physics*
   (scattering emits a source), so it lives on the scattering operator
   (:meth:`LegendreMomentTransfer.apply <orpheus.transport.operators.transfer.LegendreMomentTransfer.apply>`),
   where the cross sections are. Putting :math:`\Lambda` on the frame would
   conflate "change of basis" with "change of physical kind".

Explicit typed path vs the fused composed kernel — one per END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The algebra deliberately keeps **two** realisations of the same
:math:`R\Lambda M` math, each chosen for what its consumer needs to see.
Since CS4c step 5 they are not two arms of one operator — they are the
interior bodies of the **two bindings**, selected at construction from
which end of the retained analysis face the domain's interior is
(:ref:`cs4c-ends-select-the-body`):

* The **angular-end** binding (every 1-D, curvilinear, Krylov and
  un-windowed iterate; :ref:`sn-angular-windowing-factoring`, the body
  ``TransferOperator._redistribute_ordinates`` — a public
  ``build_aniso_source`` wrapper stood in front of it until #448)
  consumes the §5.6 :attr:`kernel <orpheus.transport.operators.transfer.TransferOperator.kernel>`
  ``= frame.conjugate(Λ)`` — the **single composed** ``np.ndarray``
  operator. This is the 0-ULP canary: one composition, one reduction
  tree. The role change is *implicit* inside the ndarray chain; that is
  correct here, because the consumer is a tight numerical loop that never
  names the intermediate moment source.
* The **moment-end** binding (the 2-D Cartesian windowed SI driver's
  gain, built by ``S.on_moment_domain()``) takes the **explicit typed**
  edges: :math:`\Lambda` scatters the flux moments to a typed
  :class:`HarmonicMomentSourceSink` (the role-changing edge made visible
  in the signature), then the MINTED source-reconstruction face
  (:class:`~orpheus.transport.frames.harmonic_frame.HarmonicReconstructionOperator`,
  bound to the posed interior) synthesises the per-ordinate
  :class:`AngularSourceSink`, then the producer-side :math:`1/W`. Here
  :math:`M` is *already done*, so conjugating would double-project.

Both routes go through the *same* :math:`\Lambda` kernel and the *same*
frame :math:`R` face, so they agree numerically. The choice is one of
**legibility at the call site**, not of math (ruling **R-5** keeps both):
the windowed consumer holds the moments as a typed iterate and the
explicit edges make the flux→source→angular role flow read off the
signatures; the angular consumer holds a raw ndarray and the fused
operator keeps the reduction tree single and the 0-ULP canary meaningful.

⛔ **The crosscheck's second side moved UP a tier, and the private oracle
is gone.** Until CS4c step 5 the comparison was against
``ScatteringOperator._aniso_source_from_moment_values``, a private
``frame.reconstruct_after(Λ)`` chain reached by handing a moment iterate
to the ANGULAR-bound operator (`[M]` 143 such feeds per windowed solve).
That helper is **retired**: the moment operand now rides an operator
bound on the moment end, so the two sides of the crosscheck are the two
BOUND operators' own public actions rather than a private helper against
a private chain. The claim did not weaken — the second side is now the
production route instead of a fragment of it. The gate is
``tests/sn/operators/test_scattering_kernel_crosscheck.py``; it records
its own 200-seed ``array_equal`` sweep, and an independent reproduction
on a 1-D GL8 :math:`P_1` slab (2 groups, 20 cells, seeds 0…199) reads
**200 / 200, max |Δ| = 0.0**.


.. _carrier-grid-census:

The full (Representation × Role) census — and its two principled holes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The scattering square uses four cells of the grid; the whole grid is the
:math:`4 \times 4` product of the four representations against the four
roles. Most cells are realised as a concrete leaf; two are *deliberately*
empty, for reasons that are themselves part of the architecture (an empty
cell that is principled tells a future session "do not mint this", which is
as load-bearing as a populated one). The census below is the live state of
:mod:`orpheus.transport.fields`,
:mod:`orpheus.transport.source_sinks`,
:mod:`orpheus.transport.residuals` (the fourth package,
``transport/displacements/``, retired at campaign-1 CS3 —
:ref:`cone-role-grid`).

.. list-table:: The (Representation × Role) carrier grid — leaf census
   :header-rows: 1
   :stub-columns: 1
   :widths: 22 26 26 26

   * -
     - **Flux**
     - **Source**
     - **Residual**
   * - **Angular**
     - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
     - :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`
     - :class:`~orpheus.transport.residuals.angular_residual.AngularResidual`
   * - **Moment**
     - :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
     - :class:`~orpheus.transport.source_sinks.harmonic_moment_source_sink.HarmonicMomentSourceSink`
     - — *(principled hole, below)*
   * - **Scalar**
     - :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux`
     - :class:`~orpheus.transport.source_sinks.scalar_source_sink.ScalarSourceSink`
     - :class:`~orpheus.transport.residuals.scalar_residual.ScalarResidual`
   * - **Trace**
     - :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
     - :class:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink`
     - :class:`~orpheus.transport.residuals.angular_boundary_residual.AngularBoundaryResidual`

⭐ **The flux → source/sink edge of each row is now DECLARED, once, on the
row's source/sink leaf** — the bijection the operator tier reads instead of
parsing carriers (:ref:`role-partner-declaration`, CS4c step 5). `[M]` 7
such pairs ship; the residual column is deliberately outside the
partnership, for the same reason its ``(Moment, Residual)`` cell is empty.

Reading the columns confirms the two-axis structure of
:ref:`cone-typed-field-algebra`: every column is a *plain vector role*
whose arithmetic is the inherited
:class:`~orpheus.numerics.field.Field` algebra, and what separates the
columns is **class identity**, which the runtime gate reads as units
identity. The Trace (boundary) row mirrors the bulk rows exactly — the
parallel the boundary role grid completes at
:ref:`bc-extraction-operator-output-typing`.

⛔ **A fourth column, Displacement, stood here from 2026-06-08 to
2026-08-19** — ``AngularDisplacement`` / ``MomentDisplacement`` /
``ScalarDisplacement`` / ``AngularBoundaryDisplacement``, the affine
increment leaves. It retired whole with the cone overturn: a flux
difference is the flux type carrying a signed value, so the column had
nothing left to hold (:ref:`cone-role-grid`).

The role axis carries no behaviour, and that is the type-vs-property rule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`[M]` **no flux leaf defines** ``__add__`` **or** ``__sub__``:
:class:`~orpheus.transport.fields.angular_flux.AngularFlux`'s MRO is
``AngularField → BulkField → Field → ABC``, with the algebra inherited
whole. The only leaf on the grid that overrides an additive dunder is
:class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`,
and what it overrides is *cross-representation*, not cross-role: it
accepts a
:class:`~orpheus.transport.source_sinks.scalar_source_sink.ScalarSourceSink`
partner and applies the canonical subspace-containment injection before
adding (the refined #207 exception; Issue #288 records that this is
statically unspellable against Field's ``(T, T) -> T``).

This is the project's type-vs-property rule (CLAUDE.md Cardinal Rule 2;
the ``coding-elegance`` "build the primitive" pattern) working: mint a
distinct **role object** only where a **non-identity morphism** lives on
that role. Under the cone ontology no role has one, so **no role carries
a mixin** — the roles are distinguished by being different *classes*,
which is exactly what the runtime units gate reads. A residual is born
only from a :meth:`~orpheus.numerics.field.Field._from_balance`, a source
from an operator ``apply``, a flux from a solve; class identity gates
every cross-role addition even where the units coincide. Giving any of
them a marker mixin "for symmetry" would be **ceremony** — a class with
no behaviour, the theatrics the rule forbids.

⛔ **Until 2026-08-19 this subsection argued the opposite asymmetry** —
*"Flux and Displacement are* **mixins** *(they add behaviour — the torsor
algebra, the contraction diagnostics), while Source and Residual carry*
**no mixin**\ *"*, with "**Flux** earns a class because ``flux + flux``
must *raise*". The CS3 overturn deleted both mixins, so the axis that
"changes the arithmetic interface" is now the *Source* axis. The
**conclusion** is unchanged and is re-derived above; only its worked
example moved. Note the type-vs-property rule is *vindicated* by the
overturn: the affine mixin was behaviour minted for a morphism that
turned out not to exist.

The two principled holes
^^^^^^^^^^^^^^^^^^^^^^^^^

Two cells are deliberately empty, and both absences are designed.

* **(Moment, Residual) —** :class:`!HarmonicMomentResidual` **is absent.**
  A residual is born only from a balance equation
  (:meth:`~orpheus.numerics.field.Field._from_balance`), and **moment space
  is never the subject of a balance**. The transport balances are the
  bulk-angular :math:`(L + C - S - F)\psi - q` and the boundary consistency
  :math:`\psi.\text{inflow} - B\,\psi.\text{outflow} - q.\text{inflow}`
  (:ref:`bc-extraction`); neither is posed in moment coordinates, so there
  is no ``from_balance`` consumer that would produce a moment residual.
  Minting :class:`!HarmonicMomentResidual` would create a leaf no producer
  fills — an illegal state by the "build the primitive only when it has a
  consumer" rule. The hole is the rule, not an oversight. (Until 2026-08-19
  this bullet contrasted the hole with a *populated*
  **(Moment, Displacement)** cell — ``MomentDisplacement`` existed because
  the angular-windowed SI iterate *does* hold its state as a moment flux
  and *does* difference it, :ref:`sn-angular-windowing`. Under the cone
  ontology that difference **is** a
  :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`,
  so the contrast is gone and the residual hole stands alone.)

* **The** ``iso + aniso → AngularSourceSink`` **source injection is a
  hand-rolled Representation traversal inside a dunder, and it is
  endorsed.** :meth:`AngularSourceSink.__add__ <orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink.__add__>`
  accepts a :class:`~orpheus.transport.source_sinks.scalar_source_sink.ScalarSourceSink`
  partner and applies the **canonical subspace-containment injection**: a
  scalar (isotropic) source lives in the subspace of the per-ordinate
  (angular) source where every ordinate carries the same value, and the
  injection :math:`\text{iso} \to \mathbf 1 \otimes \text{iso}` (broadcast
  across the :math:`\Omega` axis) maps it in before the add. This is a
  *Representation* change (Scalar → Angular) performed inside a *Role*
  operation (source + source), so it sits slightly outside the clean
  "horizontal morphisms are frame faces" story — it is a one-off
  representation embedding baked into a leaf's arithmetic. It is kept
  because the embedding is the genuine mathematical relation between an
  isotropic and an anisotropic source (the :math:`\ell=0` block *is* the
  isotropic part), and it was **endorsed at creation** as the right home
  for the iso/aniso combine. It is documented here as a recognised,
  deliberate exception so a future reader does not "tidy it away".

  Since CS4c step 5 the production ``iso + aniso`` combine has ONE
  home — ``AngularLift._combine``, the shared lift base's producer-side
  method, which performs ``(iso / W) + aniso`` through this injection and
  is where the producer-side :math:`1/W` convention lives.  Every
  isotropically-lifted producer routes through it: :math:`S`'s
  :math:`P_0` half (combined with its :math:`\ell\ge1` part), the
  :math:`(n,2n)` gain's, and fission's whole action
  (:ref:`cs4c-ends-select-the-body`).

  ⛔ **It was a free function until 2026-09-04.**  ``(iso / W) + aniso``
  was hoisted out of :math:`S` as ``assemble_per_ordinate_isotropic`` in
  ``orpheus/transport/operators/_per_ordinate.py`` at the moment the
  second consumer appeared — the defer-until-two rule taken at its word —
  and that module's own docstring named its retirement trigger: *"when a
  third isotropic lifted channel appears, this is the seed of the generic
  lift operator"*.  The third channel is fission (CS4c step 4), so the
  trigger fired, the seed grew into the ``AngularLift`` base, and the
  module was deleted into it.  ⭐ The record is worth keeping precisely
  because the prediction was *written down with its trigger* and then
  honoured: the function did not drift into a utility module, it was
  promoted on the condition it had declared.

  ⚠ **This dunder is the PULLBACK, not the section — and this bullet
  said otherwise until 2026-08-24.** It read *"it is single-sourced
  through*
  :meth:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink.from_isotropic`\ *"*,
  which is false at HEAD and is the exact confusion the two-arrow
  design exists to prevent: `[M]` the dunder's body is
  ``self.values[None] + other.values`` — the **plain** broadcast
  :math:`\pi^{*}`, with no division — while ``from_isotropic`` applies
  the producer-side :math:`1/\Sigma w` and is therefore the **section**
  :math:`E`. The two differ by exactly the axis's total weight
  (:ref:`spaces-collapse-pair-two-arrows` on
  :doc:`/theory/foundations/spaces`), so they are not one another's
  single source and never were interchangeable. Use the dunder when the
  caller has ALREADY divided; use ``from_isotropic`` when it has not.
  What the two DO share since CS4b S3 is their coherence check — both
  demand that the iso operand's space BE the angular operand's
  non-angular marginal, compared axis-wise by CONTENT.


.. _carrier-grid-flat-leaf-normal-form:

Why a fully-typed ``Carrier[Representation, Role]`` is impossible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The grid *invites* an obvious design: make the carrier itself generic in
both axes — ``Carrier[Representation, Role]`` — and write operators that
are generic in both, so that the whole grid is one parametrized type and
the leaf zoo collapses. This is the design the architecture **explored and
rejected**, because it is structurally impossible in Python and would break
a runtime safety gate even in the fragments where it type-checks. Recording
*why* it cannot work is the point of this subsection: it stops a future
session from re-attempting the "obvious" collapse and discovering the wall
the hard way. The flat multiple-inheritance leaves are not a compromise
forced by a weak type system — they are the **unique principled normal
form**, and the genericity the grid expresses has a correct home (the
operator's ``[Domain, Codomain]``) that the carrier does not.

The argument is five obstructions, each fatal on its own.

.. list-table:: Why ``Carrier[Representation, Role]`` cannot be built
   :header-rows: 1
   :widths: 8 30 62

   * - #
     - Obstruction
     - Why it is fatal
   * - **(a)**
     - **Role changes the arithmetic interface ⟹ Role MUST be a class.**
     - A phantom type parameter (``Generic[Role]``) is **erased at
       runtime**, so it cannot specialize a dunder — and it cannot be read
       by a runtime gate either. Two facts make Role's interface
       role-dependent. First, `[M]` the Source role owns a
       *cross-representation* ``__add__``
       (:meth:`AngularSourceSink.__add__
       <orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink.__add__>`
       accepts a ``ScalarSourceSink`` partner and injects it by
       subspace containment) that no other role has; a single ``__add__``
       body shared across ``Carrier[Rep, Flux]`` and
       ``Carrier[Rep, Source]`` under one phantom ``Role`` **cannot** admit
       that partner for one and refuse it for the other, because the
       parameter that would select between them does not exist at the
       moment ``__add__`` runs. Second, role identity **is** units identity
       (one ``UNITS`` class constant per leaf), and
       :meth:`Field._check_partner
       <orpheus.numerics.field.Field._check_partner>` reads it as
       ``type(self) is type(other)`` — which under erasure would compare
       equal for *every* role. A *runtime* ``role`` field that ``__add__``
       branches on is the stringly-typed anti-pattern (an illegal state is
       representable — ``replace(f, role=Source)`` would relabel a flux as
       a source without changing its units), so that escape is closed too.

       ⛔ Until 2026-08-19 this cell argued from the affine gate — *"the
       Flux role must make* ``flux + flux`` **raise** *while the Source
       role must make* ``source + source`` **succeed**\ *"*. That gate was
       retired at campaign-1 CS3 (:ref:`cone-the-overturned-affine-design`)
       and the obstruction is re-derived above from the two facts that
       survive it. The conclusion is unchanged.
   * - **(b)**
     - **Representation changes the array shape ⟹ Representation MUST be a
       class.**
     - The representation sets the ``values`` / ``space`` shape
       (:math:`(N, n_g, *\text{spatial})` vs
       :math:`(L{+}1, 2L{+}1, n_g, *\text{spatial})` vs the flat trace
       buffer) and the shape validation against it (since the S4-amendment
       the direct ``values.shape == space.shape`` guard in
       :class:`~orpheus.numerics.field.Field`; the pre-S4
       ``_phase_space_shape`` hook it replaced carried the same fact).
       A phantom ``Representation`` parameter, erased, carries none of that
       — the shape check has nothing to read. This refutes the role-outer
       phantom ``Flux[Rep]``, which obstruction (a) would have spared —
       Role is the outer class there — but the shape obstacle kills it
       instead.
   * - **(c)**
     - **The only both-classes form with role-arithmetic-once and
       rep-shape-once is the flat MI leaf — which already exists.**
     - (a) forces Role to be a class; (b) forces Representation to be a
       class. The form that has *both* as classes, with the role algebra
       written once (per role mixin) and the representation shape written
       once (per storage base), and no per-cell duplication, is the
       multiple-inheritance normal form the grid is **already** built
       from — ``AngularFlux(AngularField)`` where the role needs no
       behaviour, ``CrossSectionField(CoefficientRole, ScalarField)`` where
       it does. There is no novel encoding to discover; the normal form is
       the current code.
   * - **(d)**
     - **A parameterized carrier would break the runtime units gate via
       erasure.**
     - The field algebra enforces units/meaning at runtime with a
       **class-identity** check — ``type(self) is type(other)`` in
       :meth:`~orpheus.numerics.field.Field._check_partner` (extended by
       :meth:`BulkField._check_partner <orpheus.transport.fields._bases.BulkField._check_partner>`
       for the mesh binding). Under a generic ``Carrier[Rep, Role]`` the
       runtime class is the *erased* ``Carrier`` for **every** cell, so the
       identity check would read all cells as the same type and **admit**
       cross-representation, cross-role addition that is physically
       meaningless (adding a moment flux to an angular source). The generic
       carrier does not merely fail to *help* — it actively **disables** a
       working safety gate. The flat leaves keep one concrete class per
       cell, so the identity check stays sharp.
   * - **(e)**
     - **Both-axes-generic operators would need higher-kinded types, which
       Python lacks.**
     - Even granting a generic carrier, an operator generic in *both* axes
       — "for any Representation :math:`X` and any Role :math:`Y`,
       :math:`\Lambda` maps ``Carrier[X, Flux]`` to ``Carrier[X,
       Source]``" — quantifies over a *type constructor* applied to a
       parameter, i.e. a higher-kinded type. Python's type system has no
       higher-kinded type variables, so this signature is unspellable.

**Conclusion.** The flat multiple-inheritance leaves are the **unique,
principled normal form** — not a workaround the language imposes, but the
one design that (i) keeps a role's own dunder (the Source role's
containment injection) expressible, (ii) keeps the per-representation
shape check expressible, (iii) keeps the runtime units gate sharp, and
(iv) avoids per-cell duplication. The
genericity the grid genuinely has — "an operator maps one cell to another"
— lives where Python *can* express it and where it belongs: on the
**operator's** :math:`[\textsf{Domain}, \textsf{Codomain}]`
(:eq:`carrier-grid-operator-typing`), as the typed traversal of the grid.
The carrier stays a flat leaf; the operator carries the two coordinates.

.. note::

   **This is a closed exploration, recorded so it is not re-opened.** The
   four candidate carrier encodings — a phantom ``Field[Representation,
   Role]``, a role-outer ``Flux[Representation]``, a representation-outer
   ``Angular[Role]``, and the current flat MI leaves — were weighed against
   the obstructions above. Only the flat MI leaves survive: the
   phantom-``Role`` forms die on (a), the role-outer form dies on (b), and
   any parameterized carrier dies on (d). The full structural verdict (the
   double-category / fibration / torsor frame attack that produced this
   conclusion) is recorded in
   ``.claude/agent-memory/cross-domain-attacker/rep_role_grid_double_category_frames.md``.
   A future session reaching for ``Carrier[Rep, Role]`` should read this
   subsection first.


Deferred relocation
-------------------

The fission and scattering operators are reframed *in place* in ``sn/``;
their carrier-agnostic, cross-section-only **cores** (the rank-1
production core, the :math:`\Lambda` group-transfer, the iso / :math:`(n,2n)`
fast paths) are the natural residents of the shared ``transport/`` layer
(L2), with only the quadrature-coupled per-ordinate angular adapters
staying in ``sn/`` (L3). That relocation — together with the CP / MoC
carrier unification that would let those solvers *consume* the shared
cores instead of reimplementing C / F / S inline on bare scalar arrays —
is tracked on #261. ✅ **The dispatch-spelling half of #261 was settled on
2026-09-04** (CS4c step 5): the relocation it waited on landed, and the
answer was neither Pattern M nor an ``@overload`` + ``match`` router — the
per-call parse is gone entirely, because the operand's kind is implied by
the ends the operator was constructed with
(:ref:`cs4c-ends-select-the-body`; the retired shape and why the two
obvious spellings failed are at :ref:`pattern-m-history`). What #261 still
tracks is the *relocation* itself and the CP / MoC carrier unification.

.. note::

   **Source map.** Category Protocol:
   :class:`orpheus.transport.operators.integral_kernel_operator.IntegralKernelOperator`
   (L2). Named kernels:
   :attr:`orpheus.transport.operators.isotropic_transfer.IsotropicFission.kernel`
   +
   :attr:`orpheus.transport.operators.isotropic_transfer.IsotropicFission.production_rate`
   (the arithmetic home since CS4c step 4; the angular binding's
   :attr:`FissionOperator.kernel
   <orpheus.transport.operators.fission.FissionOperator.kernel>` /
   :attr:`~orpheus.transport.operators.fission.FissionOperator.production_rate`
   delegate to it, and are what the Protocol gate reaches);
   :attr:`orpheus.transport.operators.scattering.ScatteringOperator.kernel`.
   Category-refinement gate:
   ``tests/transport/test_integral_kernel_category.py``. Fission
   cross-check: ``tests/sn/operators/test_fission_kernel_crosscheck.py``
   (hand-derived correctness reference + the
   :math:`\chi \cdot \mathrm{production\_rate} \equiv F.\mathrm{apply}`
   0-ULP de-risk). Scattering cross-check:
   ``tests/sn/operators/test_scattering_kernel_crosscheck.py`` (the
   :math:`S.\mathrm{kernel}.\mathrm{apply} \equiv R\circ\Lambda\circ M`
   0-ULP equivalence). Deferred follow-up: #260
   (:class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`
   un-orphaning).


.. _tensorial-framing:

Tensor product algebra
======================

Streaming, scattering, and any operator on a multi-axis flux
factor naturally as a **tensor product** of per-axis operators:

* **Streaming** (Grand Report v3 §15.1, line 2044):

  .. math::
     :label: streaming-as-tensor-product-sum

     L \;=\; D_x \otimes \Omega_x \otimes I_g
            + D_y \otimes \Omega_y \otimes I_g.


  .. no-implementation:: streaming-as-tensor-product-sum
     :kind: canonical-form

     **Nothing implements this** — the factorization is exhibited to show the
     structure, and no production path takes it. :math:`[M]` 2026-08-18:
     :class:`~orpheus.numerics.operator.SumOfTensorProductsOperator` has
     **zero** consumers outside its own definition module and the package
     re-export, so nothing in ``orpheus/`` ever assembles
     :math:`L = D_x \otimes \Omega_x \otimes I_g + D_y \otimes \Omega_y
     \otimes I_g`. Production streams by marching
     (:eq:`loss-rep-scanmarch-solve-affine`). ⚠ Contrast the sibling
     :eq:`scattering-as-tensor-product-sum`, which IS declared: that one is a
     statement about **moment** space, where :math:`\Lambda` genuinely is a
     sum of tensor products.

* **Pℓ moment scattering** (§15.2) — on **moment space**:

  .. math::
     :label: scattering-as-tensor-product-sum

     \Lambda \;=\; \sum_{\ell} P_\ell \otimes \Sigma_{s,\ell},

  where :math:`P_\ell` selects the :math:`\ell`-block on the
  harmonic-coefficient axis and :math:`\Sigma_{s,\ell}` is the
  per-:math:`\ell` group-to-group transfer matrix.

  ⭐ **Since #426 step 2 (2026-09-04) this** :math:`\Lambda` **is the
  TRANSFER FAMILY's, not scattering's**, and the generalisation is one
  scalar: the class carries the channel's **yield** :math:`y_c`, so the
  shipped form is

  .. math::

     \Lambda_c \;=\; y_c \sum_{\ell} P_\ell \otimes \Sigma_{c,\ell} ,
     \qquad y_S = 1, \quad y_{2n} = 2 ,

  and the :math:`(n,2n)` gain is the SAME
  :math:`\tfrac1W R\,\Lambda_c\,M` conjugation as :math:`S`
  (:eq:`sn-n2n-transfer-binding`).  The tensor-product structure is
  untouched — a scalar multiple of a sum of tensor products is a sum of
  tensor products — which is why the label, the equation and both
  declarations below survived the collapse unchanged.  The label keeps
  its ``scattering-`` prefix because it is a citation target with
  citers; the object it names is wider than the name.

  .. implements:: scattering-as-tensor-product-sum
     :by: orpheus.transport.operators.transfer.LegendreMomentTransfer

     **Implemented by** the class — whose own docstring cites this label
     for itself. It is block-diagonal on the harmonic-coefficient axis by
     construction, and that is the structural content of the
     :math:`P_\ell\,\otimes` factorisation: it is why
     :math:`\Lambda`, unlike the full :math:`S`, genuinely *is* a sum of
     tensor products.

  .. implements:: scattering-as-tensor-product-sum
     :by: orpheus.transport.operators.transfer.LegendreMomentTransfer.apply

     **Implemented by** the action — per-:math:`\ell` block, per-group
     transfer, :math:`(\Lambda\phi)_\ell^m\big|_g = \sum_{g'}
     \Sigma_{s,\ell}(g'\!\to\!g)\,\phi_\ell^m\big|_{g'}`, dispatched
     per-material through the cell axis. Production's default
     ``skip_l0=True`` omits the :math:`\ell = 0` block, which the separate
     :math:`P_0` in-transfer fast path carries — the energy binding
     :attr:`~orpheus.transport.operators.transfer.TransferOperator.isotropic_energy`
     over
     :meth:`~orpheus.transport.material_field.TransferMaterialField.add_p0_source`
     (a ``TransferOperator.add_iso_source`` delegator fronted it until
     #448); ``skip_l0=False`` restores the full sum for the
     :math:`R\Lambda M\psi` composition.

  .. warning:: **The left side is** :math:`\Lambda`, **not** :math:`S`.

     This equation is a statement about **moment space**, and as such it
     *is* realized. The full per-ordinate scattering operator
     :math:`S = \tfrac{1}{W}(R\circ\Lambda\circ M)` has **no** such form:
     a :class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`
     would need :math:`R` and :math:`M` to be tensor-product factors on
     independent axes, and they are rank-changing einsums that *mix* the
     ordinate and harmonic-coefficient axes. That lift was considered and
     rejected — see the note in :ref:`integral-kernel-category` and #260.
     ⛔ Until 2026-08-17 this equation wrote :math:`S`, which put it in
     direct contradiction with that rejection ~900 lines further up the
     same page. :math:`[M]` it was the only site that did: every other
     page in the corpus that names this :math:`\ell`-sum already called
     it :math:`\Lambda` — :doc:`/theory/foundations/frame` (which cites
     this very label while writing :math:`\Lambda`),
     :doc:`/theory/methods/sn/slab_multigroup` and
     :doc:`/theory/methods/sn/cartesian_multid` — as does
     :class:`~orpheus.transport.operators.transfer.LegendreMomentTransfer`'s
     own docstring.

.. vv-status: streaming-as-tensor-product-sum documented
.. vv-status: scattering-as-tensor-product-sum documented

These canonical forms motivate the
:class:`~orpheus.numerics.operator.TensorProductOperator` and
:class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`
types. The user's architectural correction that drove their
introduction:

  *"Is there any tensorial machinery to be brought in support of
  this?"*

The answer is yes, and the cost of NOT having it is that
:math:`M`, :math:`R`, :math:`\Lambda` would be three named
operators that *happen* to be expressible via
:func:`numpy.einsum`, with the axis structure implicit inside each
op. With the tensor-product types, the §9 literal statement
:math:`S_{\text{SN}} = R \Lambda M` becomes the operator-algebra
type signature, and the multi-axis structure is **structural**
rather than buried in einsum subscripts.


Definition
----------

For operators :math:`A_1, A_2, \ldots, A_k` acting on
**independent** tensor axes (each carries an ``axis`` attribute and
broadcasts on the rest), the tensor-product operator's action is
the sequential per-axis application

.. (vv-status rationale) Verified by
   ``tests/numerics/test_tensor_product_operator.py`` — Kronecker-
   product reference on small concrete factors, the
   ``is_invertible`` / ``is_adjointable`` predicate meet, and the
   algebraic laws below.
.. vv-status: tensor-product-action documented

.. math::
   :label: tensor-product-action

   (A_1 \otimes A_2 \otimes \cdots \otimes A_k)\,x
   \;=\; A_k\bigl(\cdots A_2(A_1\,x) \cdots\bigr).

.. implements:: tensor-product-action
   :by: orpheus.numerics.operator.TensorProductOperator.apply

   **Implemented by** the sequential per-factor loop, ``out = x`` then
   ``for op in self.ops: out = op.apply(out)``. Because the factors act on
   disjoint axes the order does not matter — which is why a *sequential*
   loop is a faithful realization of a *commutative* product.

Because the constituents act on disjoint axes, the order does not
matter — the operators commute on the joint tensor. The two-axis
predicates are the **meet** (recursive ``and``) over the
constituents: the tensor product is invertible iff every factor is
invertible, adjointable iff every factor is adjointable (``apply`` is
universal).


Algebraic laws
--------------

The
:class:`~orpheus.numerics.operator.TensorProductOperator` carries
three algebraic laws verified by tests
(:file:`tests/numerics/test_tensor_product_operator.py`):

.. math::
   :label: tensor-product-adjoint-distributivity

   (A \otimes B)^*
   \;=\; A^* \otimes B^*.

.. implements:: tensor-product-adjoint-distributivity
   :by: orpheus.numerics.operator.TensorProductOperator.apply_transpose

   **Implemented by** the law executed rather than asserted: the body
   loops ``op.apply_transpose`` over the factors, so
   :math:`(A\otimes B)^{*} = A^{*}\otimes B^{*}` is the *shape of the loop*.
   Each iteration guard-narrows on
   :attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable` and
   raises :class:`~orpheus.numerics.operator.MissingAdjoint` naming the
   offending factor, so the law cannot be applied where it does not
   hold.

.. math::
   :label: tensor-product-axis-wise-composition

   (A \otimes B) \circ (C \otimes D)
   \;=\; (A \circ C) \otimes (B \circ D)
   \quad
   \text{when } A, C \text{ share an axis and } B, D \text{ share
   an axis}.

.. math::
   :label: tensor-product-inverse

   (A \otimes B)^{-1}
   \;=\; A^{-1} \otimes B^{-1}
   \quad
   \text{when both factors are invertible}.

.. implements:: tensor-product-inverse
   :by: orpheus.numerics.operator.TensorProductOperator.inverse

   **Implemented by** the law constructed: ``return
   TensorProductOperator(tuple(factor_inverses))``, factor order preserved.
   The per-factor
   :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible` check
   raises :class:`~orpheus.numerics.operator.NotInvertible` before any
   inverse object exists, so the composite cannot be built unless every
   factor's inverse does.

.. vv-status: tensor-product-adjoint-distributivity documented
.. vv-status: tensor-product-axis-wise-composition documented
.. vv-status: tensor-product-inverse documented

.. _tensor-product-spaces:

The spaces: a commutative composite resolves them by AGREEMENT
---------------------------------------------------------------

The three laws above are about the *action*. There is a fourth, about
the *typing*, and it is forced by the first sentence of the definition
above — **the factors commute**.

A composite has to answer ``domain`` and ``codomain`` from its
operands, and there are only two ways to do it. The non-commutative
composite :math:`A \circ B` answers **by position**: the input space is
:math:`B`'s, the output space is :math:`A`'s, and that is exactly right
because swapping the operands makes a different operator. Apply the
same rule to :math:`A \otimes B` and it produces a contradiction —
:math:`A \otimes I` would be bound at the domain and unbound at the
codomain while :math:`I \otimes A` is the reverse, so an
order-INDEPENDENT operator would carry order-DEPENDENT spaces.

So the commutative composites — the sum and the tensor product —
answer **by agreement** instead:

.. math::
   :label: tensor-product-space-agreement

   \operatorname{dom}(A_1 \otimes \cdots \otimes A_k)
   \;=\;
   \begin{cases}
     V & \text{if every factor that declares a domain declares } V \\
     \text{undeclared} & \text{if no factor declares one} \\
     \text{REFUSED} & \text{otherwise,}
   \end{cases}

and identically for the codomain. Silence is not disagreement: a factor
that declares nothing contributes nothing, which is how
:math:`K_\omega \otimes I` — every shipped SN boundary law, with the
group factor an identity — is bound exactly where :math:`K_\omega` is.

.. implements:: tensor-product-space-agreement
   :by: orpheus.numerics.operator._agreed_space

   **Implemented by** the three-way law itself, written **once**: agree
   ⟹ that space, all silent ⟹ ``None`` (silence is not disagreement), any
   disagreement ⟹
   :class:`~orpheus.numerics.operator.IncompatibleOperatorComposition`. It
   is shared with :class:`~orpheus.numerics.operator.OperatorSum`, which is
   what stops the two commutative composites from drifting into two
   spellings of one rule.

.. implements:: tensor-product-space-agreement
   :by: orpheus.numerics.operator.TensorProductOperator.domain

   **Implemented by** the equation's named subject on the left-hand
   side: the composite's ``domain``, answered by delegating to the shared
   law instead of by position.

.. implements:: tensor-product-space-agreement
   :by: orpheus.numerics.operator.TensorProductOperator.codomain

   **Implemented by** the "and identically for the codomain" half —
   the same delegation, so the two ends cannot be resolved by two different
   rules.

.. vv-status: tensor-product-space-agreement documented

**Why agreement suffices, and where it would stop.** A factor's binding
in this module is a WHOLE-space binding, not a per-leg one: a
:class:`~orpheus.numerics.operator.PermutationOperator` tagged
``axis=0`` acting on a :math:`(4, 3)` trace declares
``domain.shape == (4, 3)`` — both axes — because it broadcasts on the
rest. The factors are therefore not describing separate legs to be
multiplied together; each bound factor describes the WHOLE space, and
at most one of them can be non-trivial. Should genuine per-leg bindings
ever arrive (an energy-dependent group kernel bound on its own axis),
agreement becomes the wrong law and a **product-space constructor** is
what has to be built — which is what the refusal says, rather than
silently electing one leg.

.. admonition:: Development history — G6.3 step 8.0, 2026-08-06
   :class: note

   Until 2026-08-06 the tensor product derived **nothing** from its
   factors, inheriting the base class's ``None``. Two consequences, and
   the second was live:

   1. A binding real at the inner factor was **invisible at the object a
      realizer hands out**, so the composability check
      :math:`A.\mathrm{domain} = B.\mathrm{codomain}` — which skips
      whenever either side is ``None`` — could not fire on it. ``[M]``
      4941 bindings measured across the suite with **zero** failures:
      a green that meant nothing.
   2. ``[M]`` because
      :meth:`AdjointOperator.apply <orpheus.numerics.operator.AdjointOperator.apply>`
      reads the spaces to decide whether to apply the metrics,
      :math:`(K_\omega \otimes I)^{*}` silently degraded to the
      **Euclidean transpose** — not a weaker adjoint, a different
      operator. On a 3-group
      ``gauss_legendre(8)`` xmin face it was **87 % relative** away
      from the partial-current Hilbert adjoint for the Lambertian.

   ⭐ **The specular mirror is exactly blind to (2)**, and that is why no
   gate saw it: :math:`G_{\Gamma_-} = G_{\Gamma_+} \circ \pi`
   bit-exactly, because a mirror preserves
   :math:`|\Omega\cdot\hat n|\,w_n`, so the two metrics cancel and the
   weighted and unweighted adjoints agree to **0.0**. A
   reflective-only fixture certifies a Lambertian defect. The gate that
   closed it therefore parametrizes both laws AND asserts that the
   Lambertian fixture can still tell a weighted adjoint from a bare one
   (:ref:`verification-anti-patterns`, Mode 12 — the measured
   functional's invariance group contained the error class).

   Gated by ``TestTensorProductSpaces`` /
   ``TestSumOfTensorProductsSpaces`` in
   :file:`tests/numerics/test_tensor_product_operator.py` (the law) and
   ``TestTheRealizedLawIsMETRICCorrect`` in
   :file:`tests/sn/operators/test_sn_boundary_realizer.py` (the
   consequence, at the tier the physics lives in). ``[M]`` the mutation
   battery: dropping the derivation reddens **18**, swapping
   ``domain`` ↔ ``codomain`` reddens **8**, replacing agreement with the
   position rule reddens **7**, and removing the disagreement refusal
   reddens exactly the **3** refusal gates.


Relation to numpy primitives
-----------------------------

This is the load-bearing distinction the user explicitly asked
about:
:func:`numpy.kron`, :func:`numpy.tensordot`, and
:func:`numpy.einsum` are **array primitives** — the
*implementation* layer.
:class:`TensorProductOperator` is the **operator-algebra type** —
a layer above.

.. list-table:: Two abstraction layers
   :header-rows: 1
   :widths: 26 36 38

   * - Layer
     - Carrier
     - Examples
   * - **Implementation** (numpy)
     - Untyped multi-axis arrays; subscript strings encode axis
       structure; algebra is in the programmer's head.
     - ``np.einsum("nlm,n,n...->lm...", Y, w, psi)`` — the axes
       ``n``, ``l``, ``m``, and ``...`` are conventions only;
       nothing in the type system prevents passing a wrong-shaped
       ``Y`` or a wrong-axis ``w``.
   * - **Operator algebra** (this module)
     - Typed operators; ``axis`` attribute on each factor; the two-axis
       predicate meet; algebraic laws checked at composition.
     - ``quad.angular_frame(L).analysis & IdentityOperator()`` —
       the type signal carries the axis structure; mismatched axes
       raise at composition; the :meth:`apply` routes through
       ``np.einsum`` internally with the correct subscripts.

The two layers are **complementary**, not competitive. The
operator-algebra layer routes through the array layer for
performance — every :meth:`apply` call eventually calls
``np.einsum`` or a broadcast-multiply, because numpy's einsum
backend is more optimised than anything Python-level the project
could write. The advantage of the operator-algebra layer is in the
**composition language**:

* ``A & B & C`` reads as :math:`A \otimes B \otimes C`.
* ``(A & B) @ (C & D)`` automatically reduces to
  ``(A @ C) & (B @ D)`` (the axis-wise composition law).
* ``(A & B).H`` is ``A.H & B.H`` (adjoint distributivity).
* The ``is_invertible`` / ``is_adjointable`` predicates are computed
  automatically; a composition mismatch raises at composition, not at
  the first :func:`numpy.einsum` call mid-iteration.

.. note::

   :func:`numpy.kron` constructs a **dense** Kronecker product
   matrix — the explicit :math:`(N_A N_B) \times (N_A N_B)` entry
   table. This is the wrong abstraction for transport: an SN
   sweep's effective tensor-product operator
   :math:`L = D_x \otimes \Omega_x \otimes I_g` would be a
   :math:`(N N_x N_y N_g)^2` matrix — never materialised in
   practice because the action is matrix-free. The
   :class:`TensorProductOperator` carries the algebra without
   the materialisation, just as
   :class:`scipy.sparse.linalg.LinearOperator` carries dense-matrix
   algebra without the dense matrix.


SumOfTensorProductsOperator
---------------------------

When several tensor products are summed,
:class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`
provides the §15.2 canonical scattering / streaming form's named
type:

.. math::
   :label: sum-of-tensor-products

   T \;=\; \sum_{k=1}^{K} A_k \otimes B_k \otimes C_k.

.. vv-status: sum-of-tensor-products documented

Algebraically just :class:`OperatorSum` over
:class:`TensorProductOperator` summands, but exposed as a named
class so the §15.2 invariants
(:meth:`assert_separable` — every summand is a tensor product;
shared-axis-factor refactoring; future TT-compression entry point
per §15.3) carry a load-bearing type signal. The
:meth:`assert_separable` method is currently a contract-validator
(separability is enforced at construction); it is a hook for
future invariant checks.


Tensor product as the inverse of partition
-------------------------------------------

The :class:`~orpheus.numerics.measure.DiscreteMeasure`'s
:meth:`partition_by` method (see :ref:`discrete-measures`) realises
the **direct sum**
:math:`\mu_{S^2} = \bigoplus_\lambda \mu_\lambda`. When the
predicate is the octant-sign label
:math:`\lambda(\hat\Omega) =
(\mathrm{sign}\,\mu_x, \mathrm{sign}\,\mu_y, \mathrm{sign}\,\mu_z)`,
the partition recovers the eight octants of :math:`S^2` (or four
in 2-D where :math:`\mu_z = 0` is a degenerate case).

For per-octant operators (e.g. the SN sweep's :math:`A_{oct}^{-1}`
acting on
:math:`(\text{octant\_ordinates} \times \text{cells} \times
\text{groups})`), the **tensor product** factors the per-axis
structure within an octant while the **direct sum** assembles the
octants into the global angular cubature:

.. math::
   :label: octant-direct-sum-tensor-product

   A^{-1} \;=\; \bigoplus_{\text{oct}}\,
                A_{oct}^{-1}, \qquad
   A_{oct}^{-1} \;=\; \text{(per-axis tensor product within an octant)}.

.. vv-status: octant-direct-sum-tensor-product documented

This is the operator-algebra type signature behind Wave 2 of the
SN performance plan: the 2-D Cartesian sweep iterates
:math:`\bigoplus_{oct}` (4 octants — structural) and per-octant
anti-diagonals (sweep-DAG topology — structural), with no Python
loop over the ordinate axis (which is **internal** to every
:meth:`apply` call within an octant, vectorised by the tensor-
product structure).


.. _field-type-vs-property-criterion:

When a moment representation earns a type (#263)
================================================

The tensor-product algebra above raises a recurring design question whenever
a new moment representation appears: should it be a first-class **field type**
(a sibling of
:class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`),
or merely a **property** — a moment axis riding on an existing field?  The
question surfaced sharpest in the SN linear-discontinuous (LD) boundary work
(:ref:`ld-cartesian-2d-coherent-promise`, Issue #257 S9), which had to decide
whether the transverse boundary moment deserved a ``BoundaryMomentField`` type.
This section records the criterion that answers it, because the answer is a
durable design invariant, not a one-off call.

The criterion: a non-canonical dual must coexist
------------------------------------------------

   A representation earns a distinct first-class **type** if and only if there
   exist **two bases that are NOT canonically isomorphic** (the isomorphism
   depends on a quadrature or node choice), connected by a **change-of-basis
   operator that is itself modelled and applied** — it carries truncation
   error, has an adjoint, and participates in the operator algebra.

All three clauses must hold.  This is the sharp form of "a dual must coexist
and not mix", and it is decidable by inspection: count the within-axis
representations and count the applied, non-identity morphisms between them.  If
there is one representation, or the only morphism is the identity, the
representation is a **property**; a type would add no behaviour beyond class
identity — type-theatrics by the project's own standard (a type hint that does
not prevent a bug by construction earns nothing).

The criterion is the field-type analogue of the tensor-product algebra's own
discipline (a typed operator is justified when it carries an ``axis`` attribute,
the two-axis inverse/adjoint predicates, and algebraic laws checked at
composition — not merely a name):
a typed field is justified when it carries a *dual* whose mixing must be
forbidden, not merely a name.

Angular order PASSES; spatial order FAILS (today)
-------------------------------------------------

**Angular order is correctly TWO types.** The ordinate basis
(:class:`~orpheus.transport.fields.angular_flux.AngularFlux`, :math:`N`
collocation directions on :math:`S^2`) and the harmonic-modal basis
(:class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`,
:math:`(L+1)(2L+1)` real-spherical-harmonic coefficients) are NOT canonically
isomorphic — the isomorphism depends on the quadrature
:math:`Y_\ell^m(\hat\Omega_n)`.  They are bridged by the APPLIED
projection / reconstruction pair :math:`M` / :math:`R`
(:eq:`harmonic-moment-projection` and its reconstruction inverse), which carry
truncation content and have adjoints and live in the operator algebra.  All
three clauses hold, so the two field types are load-bearing: a ``flux +
moments`` addition is type-rejected by construction (the field-layer partner
gate), exactly as it should be — the ordinate and modal representations must
not silently mix.

**Spatial order is correctly a PROPERTY (today).** There is ONE within-cell
spatial basis — the tensor-Legendre DG tower
(:class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`,
``per_axis**ndim`` coefficients).  The only change-of-basis within it is the
identity (and ``truncate`` / inclusion, which stay within the same family and
return the same tower).  Clause 1 fails: no non-canonical dual coexists.  So
the spatial moment rides as a property — a trailing
:class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`
factor on the bulk leaf's SPACE (minted by
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space`, the
scheme-widened sibling of
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space`; before
CS4b S5 the same factor was composed on by an explicit
``spatial_moments=`` factory argument through
:meth:`BulkField._compose_spatial_moments <orpheus.transport.fields._bases.BulkField._compose_spatial_moments>`,
which survives only as the private derivation behind the admission
re-mint), and the flat face-buffer moment tail minted by
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout` on
the boundary — rather than as its own field type.  A ``BoundaryMomentField`` leaf whose
partner-check added nothing beyond class identity would be the vacuous naming
leaf the criterion warns against; the transverse boundary moment is therefore a
PROPERTY of the boundary field, the call S9 made.

The defer-with-trigger decision (#263)
--------------------------------------

The first-class spatial ``SpatialMomentField`` type is DEFERRED, with an
explicit trigger, under Issue #263.  The trigger is the arrival of a
**non-canonical spatial dual**: a nodal / point-value within-cell (or
face-current) representation enters production AND a modelled, applied
nodal↔modal morphism is written between it and the existing
:class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`.  Two
concrete arrivals would supply it:

* **Nodal discontinuous Galerkin SN** — nodal Lagrange point-values coexist
  with the modal Legendre coefficients, bridged by the applied Vandermonde
  matrix (truncation content, adjoint, in the algebra).  The
  Hesthaven–Warburton nodal-DG construction is the canonical instance.
* **Nodal diffusion (NEM / SANM / ANM)** — transverse-Legendre moments coexist
  with face partial currents, bridged by the coupling coefficients (a modelled
  morphism).  This is the strongest spatial for-case, though it is not on the
  current roadmap.

Until such a dual exists, every order-expansion in the codebase is a
PARAMETER within one tower (the ``per_axis = 2 → 3`` widening for higher-order
SN, hierarchical-Legendre / hp-FEM degree :math:`p`, p-multigrid — all single
hierarchical towers where prolong is inclusion and restrict is its adjoint,
WITHIN one representation), not a new type.  When the dual arrives, the right
move is to lift ``SpatialMomentField`` and its nodal dual into the
:class:`~orpheus.transport.fields._bases.MomentField` family ABC, mirroring the
:math:`M` / :math:`R` pair — the ABC already exists as a thin family marker
anticipating exactly this second instance.  p-adaptivity that needs modelled
``prolong`` / ``restrict`` operators flips toward typed OPERATORS within one
family (like ``truncate``), not a new field type, because those morphisms are
canonical within one Legendre tower.


Boundary conditions as operators
================================

Every *realised* boundary condition IS a Wave-0
:class:`~orpheus.numerics.operator.LinearOperator`, composable with the
streaming / collision / scattering / fission operators through the same
algebra dunders — which is why the boundary law :math:`B` enters
:math:`A = L + C - S - B` as a first-class sibling rather than being
folded into streaming. A boundary *law* is a pure descriptor with **no**
``apply``; only its *realisation* through
:class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` is a Wave-0
operator.

The full treatment — the §15.2 :math:`G_\alpha` geometric realisation
primitives (permutation, mask, average, wrap, source; the law→primitive
map), the rank-N Marshak / partial-current **descriptor-tree algebra**
(``LawSum`` / ``LawScaled`` realised by
:func:`~orpheus.geometry.boundary.realize_recursively`), and the
load-bearing **descriptor-tree vs operator-tree** type separation
("mixing the two algebras is a type error") — is documented on the
boundary-condition page: see the :ref:`extraction narrative
<bc-extraction>`, the :ref:`law→primitive realisation map
<bc-tensor-primitives>`, and the :ref:`recursive realiser
<bc-realize-recursively>` in
:doc:`/theory/foundations/boundary_conditions`.

Tensor-network decomposition
============================

The factored / tensor-network shape decomposition of the S\ :sub:`N`
operators — *which* algebraic shape (tensor product, sum-of-tensor-products,
or an irreducible sequential walk) each operator leaf takes, the MA-Q1
admissibility condition, and why streaming resists a clean factorisation —
is developed in :doc:`/theory/foundations/operator_tensor_network`.


The cone-typed field algebra
============================

The S\ :sub:`N` **field** algebra — the recognition that
flux lives in the **positive cone** :math:`K` of an ordered vector
space :math:`V`, with membership an element predicate and preservation a
property of the realization — is developed in full on its own page,
:doc:`/theory/foundations/field_algebra`. That page types the **fields**
(flux / source / residual) that the operators of this page act on; it
also wires the typed equation residual
:math:`r = (L + C - S - B)\,\psi - q` to its ``from_balance`` consumer,
and it carries the six-argument adjudication that overturned the 2026-06
affine ontology (⛔ until 2026-08-19 this paragraph read *"flux* **states**
*form an affine space* :math:`\mathbb{A}` *(points, no origin) over a
difference vector space* :math:`V`, *with the field-role grid completed
by a* **displacement** *column"*).

The composite metric adjoint
============================

The **metric-correct Hilbert adjoint** ``op.H`` of the composed loss
operator :math:`A = L + C - S - B` — the **G-adjoint**
:math:`A^{\dagger} = G^{-1} A^{\mathsf T} G` over the block-diagonal
:class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace` metric,
with the singular trace block inverted by a Moore–Penrose pseudo-inverse
— is developed on its own page, :doc:`/theory/foundations/operator_adjoint`.
That page derives it from the reciprocity identity :math:`\langle A\psi,
\varphi\rangle_G = \langle \psi, A^{\dagger}\varphi\rangle_G`, shows why
the metric applies once at the operator level, and pins reciprocity to
round-off against a structurally-independent dense-probe oracle. It is the
adjoint face of the operator algebra on this page — distinct from the
frame's Petrov–Galerkin test-space adjoint in
:doc:`/theory/foundations/frame`.


The interior face-flux cochain
==============================

The **interior face-flux cochain** :math:`C^1_{\rm int}` — the
sweep-internal record of angular flux on interior faces, its biproduct
decomposition :math:`C^1 = C^1_{\rm int} \oplus C^1_\partial` and trace
algebra, and the succession note on why the typed ``WavefrontFlux``
carrier retired (the concept survives in its two native realizations) —
is developed in full on its own page,
:doc:`/theory/foundations/wavefront_cochain`.

The inverse family
==================

How the composed operator's inverse is realized and materialized — the
driver-applied sweep, the Green preconditioned inverse, the dense
materialising inverse, and the sparse assembly axis — is developed in
:doc:`/theory/foundations/operator_inverse_family`.

The coupled block operator
==========================

The **coupled block operator** — the 2×2 block system in which the
curvilinear starting-direction flux :math:`\psi_{1/2}` (the ψ½ ray)
becomes a first-class **System B** alongside the angular-flux System A,
its four named blocks (:math:`A_{AA}` the within-group loss composite,
plus the seed / emission / march couplings), the N-general block
machinery, and the structure-keyed block solve — is developed in full
on its own page, :doc:`/theory/foundations/coupled_block_operator`.

.. _trace-spaces-doc:

Trace spaces — :math:`\Gamma_-` and :math:`\Gamma_+`
====================================================

Boundary conditions act on the **directional half** of the
transport equation's boundary trace. Per Grand Report v3 §5.3 and
§16A.5, the trace splits into two pieces by the sign of
:math:`\Omega \cdot \hat n`:

.. math::
   :label: trace-half-decomposition

   \Gamma_- \;=\; \{(\mathbf{r}, \Omega) \in \partial\Omega \times S^d
                  : \Omega \cdot \hat n(\mathbf{r}) < 0\},
   \qquad
   \Gamma_+ \;=\; \{(\mathbf{r}, \Omega) : \Omega \cdot \hat n > 0\}.

.. implements:: trace-half-decomposition
   :by: orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face

   **Implemented by** the :math:`\Gamma_-` selector — the ordinate
   indices with :math:`\Omega\cdot\hat n_f < -\epsilon`, i.e. the strictly
   **inward** half.

.. implements:: trace-half-decomposition
   :by: orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_indices_for_face

   **Implemented by** the :math:`\Gamma_+` selector, :math:`\Omega\cdot
   \hat n_f > +\epsilon`.

   Between them the two selectors realize exactly the property the
   surrounding prose insists on: **disjoint but not exhaustive**. Each
   excludes the tangential band independently, so "not inflow" is never
   "outflow" — a fact no single mask could carry.

.. vv-status: trace-half-decomposition documented

The two halves are directional selectors on the single unified
:class:`~orpheus.numerics.space.FunctionSpace` subclass
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`
(the per-face ``InflowTraceSpace`` / ``OutflowTraceSpace`` classes were
consolidated into inflow/outflow selectors on it — the
:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face`
/ :meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_indices_for_face`
methods), which carry a **per-face directional mask**:

.. math::
   :label: per-face-inflow-mask

   \mathrm{inflow\_mask}[f, n]
   \;=\;
   \bigl(\Omega_n \cdot \hat n_f < -\epsilon\bigr).

.. implements:: per-face-inflow-mask
   :by: orpheus.numerics.spaces.angular_trace_space.build_omega_dot_n

   **Implemented by** the builder of the
   ``(n_faces, n_ordinates)`` table of :math:`\Omega_n\cdot\hat n_f` that
   the mask is a predicate **on**.

.. implements:: per-face-inflow-mask
   :by: orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face

   **Implemented by** the predicate itself — ``np.flatnonzero(row <
   -TANGENTIAL_EPS)`` on that table's face row, where ``TANGENTIAL_EPS = 4 *
   np.finfo(np.float64).eps`` is the :math:`\epsilon` of the equation
   (:math:`\approx 8.88\times 10^{-16}`).

.. vv-status: per-face-inflow-mask documented

The mask has shape ``(n_faces, n_ordinates)`` boolean; the
tangential band :math:`|\Omega_n \cdot \hat n_f| \leq \epsilon`
(``TANGENTIAL_EPS = 4 * np.finfo(np.float64).eps``, i.e.
:math:`\approx 8.9\times 10^{-16}`) is in neither half — so **"not
inflow" is never "outflow"**, and the two index sets are disjoint but
NOT exhaustive.

Three consumers read the mask today:

* The SN realizer consumes **both** selectors: ``inflow_indices_for_face(face)``
  gives a realized law's codomain :math:`\Gamma_-`, and
  ``outflow_indices_for_face(face)`` its **domain** :math:`\Gamma_+` —
  restricted through the
  :class:`~orpheus.numerics.operator.TraceRestrictionOperator` the same
  table caches (campaign phase B3.2; see
  :ref:`bc-domain-narrowing`).
* The universal invariant
  :meth:`~orpheus.geometry.boundary.BoundaryTraceLaw.assert_source_is_placeable`
  reads the inflow selector as a **structural** check on a law carrying an
  :class:`~orpheus.geometry.boundary.InflowSourceSpec` (ERR-047): a law with a
  source, realized against a space that cannot name :math:`\Gamma_-`, is
  refused. (Until campaign phase **P6** it was a *presence* check that first
  probed the source's values — and a source could decline it by answering the
  probe with zeros. A spec now receives :math:`\Gamma_-(f)` itself, so
  :math:`q \in \Gamma_-` holds by construction and only the structural
  precondition remains.) Where the face CAN name it, the
  delivery is structural — since campaign phase B3.4a the realizer
  sizes the source's block from :math:`|\Gamma_-|`, so
  :math:`q \in \Gamma_-` holds by typing rather than by masking a
  full-face evaluation.
* The SN curvilinear sweep (1-D spherical / cylindrical) consumes
  the same realizer-routed mask as the slab and 2-D Cartesian
  paths (Issue #188 + #176, closed 2026-05-11).

Construction goes through the classmethod factory
:meth:`AngularTraceSpace.from_quadrature_and_layout
<orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.from_quadrature_and_layout>`,
which (since Issue #225 / C5.3) is **geometry-blind**: it builds the
per-face :math:`\Omega\cdot\hat n` masks from the angular quadrature's
direction-cosine arrays and a :class:`~orpheus.numerics.face_layout.FaceLayout`
whose ``"{axis}{min|max}"`` face names imply axis-aligned outward
normals — no spatial mesh is consulted. Every :class:`Mesh1D` coord
system (``CARTESIAN`` / ``SPHERICAL`` / ``CYLINDRICAL``) shares the
same ``("xmin", "xmax")`` radial-axis face structure —
:meth:`~orpheus.numerics.quadrature.Quadrature.gauss_legendre` is the
shared quadrature, with ``mu_x`` as the
direction cosine along that axis — and 2-D / 3-D Cartesian add the
``y`` / ``z`` faces from the same convention. The 2-D cylindrical
(axisymmetric :math:`(r, z)`) case never reaches the factory: such a
:class:`Mesh2D` cannot become an :class:`SNMesh` (no 2-D cylindrical SN
sweep exists), so the refusal lives at the :class:`SNMesh` construction
surface, not the trace factory. See :ref:`sn-c5-geometry-blind-trace`.

The two trace spaces and the
:class:`~orpheus.geometry.boundary.InflowSourceSpec` Protocol
together close the §16A.1 affine boundary form
:math:`\gamma_- \psi = R\,G\,\gamma_+ \psi + q` documented in
detail at :ref:`affine-bc-form`.


.. _eigenvalue-posing:

Eigenvalue posing and the power-iteration algorithm
===================================================

The operator leaves :math:`L, C, S, F, B` documented above are the raw
material; this section is the **assembly instruction** that turns them
into a criticality eigenproblem and runs it. The architecture is
**layered into four tiers**, and the layering is the load-bearing
design decision: it is what makes the :math:`\alpha`-eigenvalue,
adjoint, and transient problems land later as *pure additions* (new
posing data) rather than new solver engines. The corrected layering
was confirmed by an independent structural analysis (the
``cross-domain-attacker`` ``eigenvalue_posing_layering_frames`` and
``power_iteration_vs_keigenvalue_morphism`` memos, 2026-06-04) and
realized in commits ``650032e`` / ``7603c8e`` (2026-06-05).

.. admonition:: Key Facts (eigenvalue posing)
   :class: tip

   - **Standard form:** the generalized eigenproblem
     :math:`A_{\rm loss}\,\psi = \lambda\,M\,\psi`. Its power-method
     realization is the dominant eigenpair of the **resolvent**
     :math:`A_{\rm loss}^{-1} M`.
   - **Krein–Rutman / Perron–Frobenius:** for a compact, positive
     :math:`A_{\rm loss}^{-1} M` the fundamental mode is the *unique*
     non-negative eigenvector and the dominant eigenvalue is real and
     positive — the only physically meaningful steady state. All
     higher harmonics change sign in space.
   - **k-eigenvalue (LIVE):** :math:`A_{\rm loss} = L+C-S-B`,
     :math:`M = F`, :math:`k = \mu`. The dominant eigenvalue of
     :math:`A_{\rm loss}^{-1} F` is :math:`k_{\rm eff}`.
   - **Four layers:** operator leaves (method-specific) → problem
     posing (bifurcated 2a/2b) → resolvent :math:`A_{\rm loss}^{-1}`
     (method-specific) → solution algorithm (general over the standard
     form).
   - **The invariant:** the Layer-4 algorithm sees ONLY a
     normalized-source fixed-point procedure — *apply* :math:`M`,
     *solve* :math:`A_{\rm loss}^{-1}`, *normalize*, *estimate* the
     dominant :math:`\mu`. It never touches the method's operators or
     sweeps.
   - :func:`~orpheus.numerics.eigenvalue.power_iteration` is the
     **canonical** Layer-4 algorithm (the *more general* layer);
     :class:`~orpheus.numerics.iteration.KEigenvalue` is one Layer-2b
     implementer that delegates its loop to it. **One loop.**


The standard form and its resolvent
------------------------------------

Discretizing the steady-state transport (or diffusion) equation
produces a balance between **loss** (streaming + collision − in-group
scattering − boundary in-scatter) and **production** (fission). Group
every loss term into a single operator :math:`A_{\rm loss}` and every
production term into a single **eigen-operator** :math:`M`. The
criticality condition — that a self-sustaining flux distribution
exists when production is scaled by :math:`1/\lambda` — is the
**generalized eigenproblem**

.. (vv-status rationale) The generalized-eigenproblem standard form.
   The verifiable claim — that the dominant eigenvalue of
   A_loss⁻¹ M equals k_eff for the k-posing — is anchored against the
   homogeneous closed-form algebra-of-record
   (orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous,
   k = λ_max(A⁻¹F)); the production iteration is the L1/L2 chain of the
   five family solvers.
.. vv-status: eigen-standard-form documented

.. math::
   :label: eigen-standard-form

   A_{\rm loss}\,\psi \;=\; \lambda\,M\,\psi .

Inverting the loss operator turns the generalized eigenproblem into a
standard one for the **resolvent operator**
:math:`K_{\rm pm} \equiv A_{\rm loss}^{-1} M`:

.. (vv-status rationale) The resolvent standard form
   K_pm ≡ A_loss⁻¹ M with K_pm ψ = (1/λ)ψ. Definitional / posing
   identity, matching the sentineled eigen-standard-form; the
   k = λ_max(A⁻¹F) claim is anchored by the homogeneous closed-form
   algebra-of-record (kinf_and_spectrum_homogeneous).
.. vv-status: eigen-resolvent documented

.. math::
   :label: eigen-resolvent

   K_{\rm pm}\,\psi \;=\; \tfrac{1}{\lambda}\,\psi ,
   \qquad
   K_{\rm pm} \;\equiv\; A_{\rm loss}^{-1} M .

This is the form every power-method realization actually iterates: one
outer step is *apply* :math:`M`, then *invert* :math:`A_{\rm loss}`
(the fixed-source solve), then *renormalize*, then *estimate*
:math:`\lambda`. The reason the dominant eigenpair is the one the
iteration converges to — and the reason it is the *physical* one — is
the **Krein–Rutman theorem** (the infinite-dimensional Perron–Frobenius
statement): for a compact, positive :math:`K_{\rm pm}`,

* the dominant eigenvalue :math:`\rho(K_{\rm pm})` is real, positive,
  and simple;
* its eigenvector is the *unique* eigenvector with no sign changes —
  i.e. a physically realizable non-negative flux distribution;
* every higher harmonic changes sign in space and is therefore not a
  steady reactor state.

Power iteration converges to exactly this fundamental mode, at a rate
governed by the **dominance ratio** :math:`|\lambda_1/\lambda_0| =
|k_1/k_0|` (Trefethen & Bau 1997, §27). The dominant K and
:math:`\alpha` eigenvalues are *extreme* eigenvalues of
:math:`A_{\rm loss}^{-1} M`, reachable by plain power iteration;
**shift-invert** :math:`(A_{\rm loss} - \sigma M)^{-1}` is the strict
generalization needed only for *interior* eigenvalues (higher
harmonics, FEAST/Arnoldi–Schur), and is a documented future seam, not
a present need.

.. note::

   The standard form :eq:`eigen-standard-form` is the discrete twin of
   the algebra-of-record in
   :func:`~orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous`,
   which solves :math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})`
   with :math:`\mathbf{A} = \mathrm{diag}(\Sigma_t) - (\Sigma_s +
   2\Sigma_2)^{T}` and :math:`\mathbf{F} = \chi \otimes \nu\Sigma_f`
   for the homogeneous infinite medium. That closed-form reference is
   the **structurally-independent** ground (a closed-form analytical
   pillar, exact in the homogeneous limit; reducing to the 1-group
   :math:`k = \nu\Sigma_f/\Sigma_a`) against which the production
   power iteration's converged eigenvalue is verified. The agreement
   of two ORPHEUS solvers is cross-implementation agreement, not
   correctness evidence; the closed-form ratio is the anchor.


The four layers
---------------

.. list-table:: The four-tier eigenvalue architecture
   :header-rows: 1
   :widths: 6 22 16 56

   * - Layer
     - Name
     - Specificity
     - What lives here
   * - 1
     - Operator leaves
     - method-specific
     - :math:`L, C, S, F, B` (+ future :math:`T = 1/v`). The
       block-diagonal :math:`G`-metric (codomain inner product of
       :math:`L`'s composite :math:`V_{\rm bulk}\oplus V_{\rm trace}`
       space: bulk :math:`V\,w_n` :math:`\oplus` trace
       :math:`|\Omega\cdot\hat n|\,w_n`) lives HERE and is reused by
       every posing (step O.2b R5, :ref:`g-adjoint`).
   * - 2
     - Problem posing
     - **bifurcated**
     - **2a** (method-agnostic): role assignment + the :math:`\mu \to`
       physical-eigenvalue map — which leaves play :math:`A_{\rm loss}`
       vs :math:`M`, and how :math:`\mu` maps to :math:`k` / :math:`\alpha`.
       **2b** (method-specific): how the method assembles and inverts
       the concrete :math:`A_{\rm loss}` object.
   * - 3
     - Resolvent :math:`A_{\rm loss}^{-1}`
     - method-specific
     - the fixed-source inner solve. SN:
       :class:`~orpheus.numerics.iteration.SourceIteration` /
       :class:`~orpheus.numerics.iteration.KrylovAcceleration`. CP:
       BiCGSTAB. Diffusion: direct FD inverse (exact LU of the fused
       :math:`A`, no inner iteration — #290). Inverts *whatever*
       :math:`A_{\rm loss}` the posing produced; independent of problem
       type.
   * - 4
     - Solution algorithm
     - general over the standard form
     - eigenvalue-finders (power iteration | full-spectrum Arnoldi /
       Krylov–Schur | shift-invert / FEAST) over
       :math:`(A_{\rm loss}^{-1}, M)`; time-integrators (transient)
       over :math:`(A_{\rm loss}, T, q(t))`.

**Why posing bifurcates (2a vs 2b).** The first-draft architecture
treated posing as wholly method-agnostic — "just arrange the leaves."
That is false: the *role assignment* is agnostic, but the *loss-operator
realization* is method-specific. SN realises its loss operator as the
invertible resolvent :math:`L + C` (the WDD sweep) plus the lagged
coupling gains :math:`S` (bulk scattering) and :math:`B` (boundary
reflection), handed to the **variadic** within-group driver as
:math:`(L+C,\,S,\,B)` (:ref:`bc-extraction-variadic-driver`). CP has
**no** :math:`(A, S, F)` split at all — its
:meth:`solve_fixed_source <orpheus.numerics.eigenvalue.EigenvalueSolver.solve_fixed_source>`
is one BiCGSTAB on a *monolithic* collision-probability matrix; the
factor :math:`(A-S)^{-1}` does not exist as a separable object.
Splitting the posing into

* **2a — role assignment + μ-map** (pure data: a posing-table row), and
* **2b — loss-operator realization** (the method's concrete assembly),

lets :math:`2a \circ 2b \circ 3 \circ 4` compose cleanly across every
family. The key consequence:
:class:`~orpheus.numerics.iteration.KEigenvalue` (built from the
``(A, S, F)`` triple) is the
operator-triple **2b realization** — NOT a problem-type layer. Treating
the operator triple as a "problem type" was the conflation the
bifurcation removes.

**The variadic driver IS the posing/resolvent boundary made explicit.**
The Layer-3 SN resolvent
(:class:`~orpheus.numerics.iteration.SourceIteration` /
:class:`~orpheus.numerics.iteration.KrylovAcceleration`) is now
**problem-type-agnostic**: it consumes
:math:`\text{Driver}(A_{\rm resolvent},\,*\text{gains})` and never asks
which gain plays which posing role. *Which* leaves are gains is the
2a decision — for the SN k-row the gains are exactly the
:math:`A_{\rm loss}` couplings :math:`S` and :math:`B` (fission
:math:`F` is the eigen-operator :math:`M`, not a within-group gain; it
enters as :math:`q_{\rm ext}`). The retired fixed :math:`(A, S, F)`
triple had baked a 2a role distinction into the Layer-3 resolvent,
where it does not belong — the variadic generalisation pushes the
distinction back up to the posing layer (:ref:`bc-extraction-variadic-driver`).

**The invariant (Layer-4 sees only a fixed point).** Layer 4 consumes
the method-agnostic
:class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` boundary — a
normalized-source fixed-point procedure with exactly these moves:
build the eigen-source from the current flux, solve the resolvent,
renormalize to unit production rate, estimate the dominant eigenvalue,
test convergence. It never sees SN sweeps, CP matrices,
:class:`~orpheus.transport.timed_full_field.TimedFullField` carriers,
or angular state — those live at Layers 1–3, *below* the boundary. This
is the abstraction that makes a new problem type a posing-row addition
rather than an engine rewrite.


The posing table
-----------------

Each row assigns the leaves to roles, gives the eigen-operator, and
gives the :math:`\mu \to` physical-eigenvalue map. **The k-row is LIVE;
the rest are documented future seams** — recorded here so a future
session lands them as pure additions.

.. list-table:: Eigenvalue posing rows (2a — method-agnostic)
   :header-rows: 1
   :widths: 16 30 14 26 14

   * - Problem type
     - :math:`A_{\rm loss}`
     - eigen-operator :math:`M`
     - :math:`\mu \to` physical map
     - Status
   * - **k-eigenvalue**
     - :math:`L + C - S - B`
     - :math:`F`
     - :math:`k = \mu`
     - **LIVE**
   * - :math:`\alpha`-eigenvalue
     - :math:`L + C - S - F - B`
     - :math:`T = 1/v`
     - :math:`\alpha = -1/\mu`
     - future seam
   * - adjoint
     - :math:`A_{\rm loss}^{\dagger}` (daggered row)
     - :math:`M^{\dagger}`
     - same :math:`\lambda` as the forward row
     - future seam
   * - transient
     - :math:`A_{\rm loss} + T/\Delta t` (per implicit step)
     - — (source-driven, not eigen)
     - — (time integration)
     - future seam

**The k-row, in full.** For k-eigenvalue criticality the fission
production is the eigen-operator and *everything else* is loss. The
resolvent's dominant eigenvalue is :math:`k_{\rm eff}` directly
(:math:`k = \mu`):

.. (vv-status rationale) The k-row posing (L+C−S−B)ψ = (1/k)Fψ ⟺
   [(L+C−S−B)⁻¹F]ψ = kψ — eigen-standard-form specialized to the LIVE
   k-row. Definitional / posing identity; k_eff is anchored by the
   homogeneous closed-form oracle (kinf_and_spectrum_homogeneous).
.. vv-status: eigen-k-posing documented

.. math::
   :label: eigen-k-posing

   (L + C - S - B)\,\psi \;=\; \tfrac{1}{k}\,F\,\psi
   \qquad\Longleftrightarrow\qquad
   \bigl[(L+C-S-B)^{-1} F\bigr]\,\psi \;=\; k\,\psi .

This is exactly :eq:`operator-eigenvalue` with the boundary in-scatter
:math:`B` made explicit (Wave O step O.4a.2 promoted :math:`B` to a
first-class sibling leaf; see :ref:`bc-extraction`). In production the
within-group loss :math:`L+C-S-B` is realised honestly: :math:`S` and
:math:`B` are two separate coupling gains handed to the variadic driver
(Wave O step O.2a — :ref:`bc-extraction-variadic-driver`), so the
matvec is :math:`(L+C).\text{apply} - S.\text{apply} - B.\text{apply}`.
The transitional :math:`S + B` driver fold is retired.

**The α-row (future seam).** The :math:`\alpha`-eigenvalue (the
time-eigenvalue, governing the asymptotic exponential time behaviour)
follows from the ansatz :math:`\psi(\mathbf r, \Omega, t) \propto
e^{\alpha t}\,\psi(\mathbf r, \Omega)`. Substituting into the
time-dependent transport equation, the time derivative
:math:`\frac1v \partial_t \psi` becomes :math:`\frac{\alpha}{v}\psi`,
so the steady balance reads

.. (vv-status rationale) The α-eigenvalue derivation (the e^{αt} ansatz
   → (L+C−S−F−B)ψ = −α T ψ). Definitional derivation of a documented
   future seam — the α-row is Not built (only the k-row exists;
   unify-after-two).
.. vv-status: eigen-alpha-derivation documented

.. math::
   :label: eigen-alpha-derivation

   \tfrac{\alpha}{v}\,\psi + (L + C - S - F)\,\psi \;=\; 0
   \qquad\Longleftrightarrow\qquad
   (L + C - S - F - B)\,\psi \;=\; -\alpha\,T\,\psi ,
   \quad T \equiv \tfrac1v .

Matching to the standard form :eq:`eigen-standard-form` gives
:math:`A_{\rm loss} = L+C-S-F-B` (fission now joins the *loss* side,
because it is no longer the eigen-operator), :math:`M = T = 1/v`, and
:math:`\mu = -1/\alpha`, i.e. :math:`\alpha = -1/\mu`. The only new
machinery the :math:`\alpha`-row needs is a sixth leaf — a
:class:`~orpheus.numerics.operator.DiagonalOperator` realizing
:math:`T = 1/v` — joining :math:`L, C, S, F, B`. The posing, resolvent,
and algorithm layers are unchanged; this is the cleanest possible fit
and is why the layering was designed this way. **Not built** (only K
exists; *unify-after-two*).

**The adjoint row (LIVE — #276 A4/A5).** The adjoint eigenproblem
:math:`A_{\rm loss}^{\dagger}\,\psi^{\dagger} = \lambda\,M^{\dagger}\,
\psi^{\dagger}` is **just another posing row** whose role-operators are
the daggers of the forward leaves — and it now RUNS in production:
``KEigenvalue((L+C).H, (S+B).H, F.H)`` through the unchanged
:func:`~orpheus.numerics.eigenvalue.power_iteration`
(:func:`~orpheus.sn.solver.solve_sn_adjoint`; the full chapter is
:ref:`sn-adjoint`). The dagger is *free* from the
dagger-biproduct category already documented on this page (the ``.H``
adjoint propagates through ``+`` / ``&`` / ``@`` — see
:eq:`tensor-product-adjoint-distributivity`) and, crucially, from the
**composite metric-correct G-adjoint** that step O.2b R5 made concrete:
``op.H`` is the metric-correct :math:`A^{\dagger} = G^{-1} A^{\mathsf T}
G` over the :math:`V_{\rm bulk}\oplus V_{\rm trace}` carrier
(:ref:`g-adjoint`), dense-oracle-verified to round-off.
Because forward and adjoint share the spectrum, :math:`\lambda` — and
therefore the :math:`\mu \to` physical map — is **unchanged**; only the
operators are daggered. The adjoint slots in at 2a with **zero new
engine machinery**: it is a row, not a layer, and its loss-operator
dagger is *already built and verified* by R5. (The first-draft
architecture's instinct to make adjoint a separate "mode" is the same
conflation the 2a/2b split removes.)

**The transient row (future seam).** Backward-Euler time stepping
:math:`(T/\Delta t + A_{\rm loss})\,\psi^{n+1} = (T/\Delta t)\,\psi^{n}
+ q^{n+1}` is a fixed-source solve with a **shifted** loss operator
:math:`A_{\rm loss} + T/\Delta t`. It **shares the Layer-3 resolvent**:
:meth:`solve_fixed_source <orpheus.numerics.eigenvalue.EigenvalueSolver.solve_fixed_source>`
inverts whatever loss operator the posing hands it, and the shifted
operator is still a streaming-plus-collision-like invertible object the
same sweep / BiCGSTAB handles. Transient therefore needs only (a) a
transient posing row and (b) a *time-integrator* Layer-4 sibling of
:func:`~orpheus.numerics.eigenvalue.power_iteration` that loops the
fixed-source solve in time and advances delayed-neutron precursors.
**No new resolvent, no new leaves** beyond the :math:`T` leaf the
:math:`\alpha`-row already introduces.


Why ``power_iteration`` is canonical, not deprecated
----------------------------------------------------

An earlier framing carried a :class:`DeprecationWarning` on
:func:`~orpheus.numerics.eigenvalue.power_iteration`, intending to
migrate all five solver families onto
:class:`~orpheus.numerics.iteration.KEigenvalue`. **The deprecation
arrow pointed the wrong way** and was removed in ``650032e`` /
``7603c8e``. The two are the **same fixed-point combinator** — one
power-method loop (the five-step *build source → solve resolvent →
renormalize → estimate → converge?* body) — instantiated at two
different layers:

* :func:`~orpheus.numerics.eigenvalue.power_iteration` exposes the
  inner resolvent **late**, behind the opaque
  :meth:`EigenvalueSolver.solve_fixed_source <orpheus.numerics.eigenvalue.EigenvalueSolver.solve_fixed_source>`
  Protocol method — a morphism the solver owns.
* :class:`~orpheus.numerics.iteration.KEigenvalue` binds the resolvent
  **early**, building it as :math:`(A-S)^{-1}` from the operator triple
  via an inner :class:`~orpheus.numerics.iteration.SourceIteration`.

The late-bound layer is **strictly more general**: it admits *both* the
sweep-posed operator-triple resolvent (SN, MoC — where the inner
:math:`(A-S)^{-1}` is a source iteration over the sweep :math:`A^{-1}`)
*and* the **monolithic-matrix resolvent** (CP, diffusion, homogeneous —
a single direct inverse). The early-bound layer can only express
methods whose resolvent factors as :math:`(A-S)^{-1}` from a triple via
that inner sweep — strictly narrower. Diffusion is the instructive case:
it *does* now carry an in-algebra :math:`(L, C, S, F)` family (#290;
:doc:`/theory/methods/diffusion_1d`), yet it still belongs to the monolithic
camp, because it has **no sweep** — its resolvent is the explicit
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
of the *fused* :math:`A = L + C - S - B` (the scattering already
subtracted in), not an :math:`(A-S)^{-1}` iterated over a within-group
solve. Forcing CP or diffusion into the narrow layer would mean
manufacturing a fictitious sweep they do not have. Therefore the
**Protocol layer is canonical** and the **triple layer is a
specialization that adapts into it**.

.. list-table:: ``power_iteration`` vs ``KEigenvalue`` — same morphism, two layers
   :header-rows: 1
   :widths: 22 39 39

   * -
     - :func:`~orpheus.numerics.eigenvalue.power_iteration`
     - :class:`~orpheus.numerics.iteration.KEigenvalue`
   * - Layer
     - 4 (algorithm) over the 2-boundary
     - 2b (operator-triple posing realization)
   * - Resolvent binding
     - **late** — opaque ``solve_fixed_source``
     - **early** — :math:`(A-S)^{-1}` from the triple
   * - Admits
     - SN, MoC, CP, diffusion, homogeneous (any Protocol implementer)
     - SN / MoC only (needs an :math:`(A,S,F)` triple)
   * - The loop
     - **owns** the single power-iteration loop body
     - **delegates** to ``power_iteration``
   * - Role
     - the canonical engine
     - one implementer of the boundary

After the fix, the loop body lives in **one place**
(:func:`~orpheus.numerics.eigenvalue.power_iteration`).
:class:`~orpheus.numerics.iteration.KEigenvalue` realizes the
:class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` boundary from its
triple — :meth:`compute_fission_source <orpheus.numerics.iteration.KEigenvalue.compute_fission_source>`
:math:`= F\psi/k`,
:meth:`solve_fixed_source <orpheus.numerics.iteration.KEigenvalue.solve_fixed_source>`
:math:`= (A-S)^{-1} q` via the warm-started inner
:class:`~orpheus.numerics.iteration.SourceIteration`, the **hardwired**
:math:`k`- and production-estimators
(:meth:`~orpheus.numerics.iteration.KEigenvalue.compute_keff` /
:meth:`~orpheus.numerics.iteration.KEigenvalue.compute_production_rate`;
the pre-R8 injection kwargs retired at #259 P1), and the
:math:`\ge 3`-iteration :math:`dk`/:math:`d\phi` convergence test — then
:meth:`solve <orpheus.numerics.iteration.KEigenvalue.solve>` simply
calls ``power_iteration(self, max_iter=self.max_outer)``. SN production
(:func:`~orpheus.sn.solver.solve_sn`), CP, diffusion, MoC, and
homogeneous all drive the same loop directly via the Protocol; the
``KEigenvalue`` adapter is for callers who *have* a natural
:math:`(A,S,F)` triple and want to skip writing a full solver class.

.. note::

   The "``KEigenvalue`` regresses :math:`P_\ell` (anisotropic
   scattering)" objection from the migration era dissolves under this
   framing: :class:`~orpheus.numerics.iteration.SourceIteration` is
   type-agnostic and **angular-capable** — it routes the RHS through
   the ravellable protocol, so a typed
   :class:`~orpheus.transport.operators.scattering.ScatteringOperator` acting on a
   :class:`~orpheus.transport.full_field.FullField` (the history-blind
   operator carrier; the driver feeds its
   :class:`~orpheus.transport.timed_full_field.TimedFullField` iterate, which
   reaches the arm via MRO and carries the full angular flux on its bulk)
   carries :math:`P_\ell` correctly. The observed regression was a property of
   an L1 *test adapter* that collapsed angular flux to scalar between
   outer iterations (dropping the angular moments), not of
   ``KEigenvalue``. The decisive — and sufficient — reason
   ``KEigenvalue`` cannot be the universal engine is the
   CP/diffusion/homogeneous **monolithic-resolvent (no-sweep)** fact
   alone.


The metric lives at the leaf, not the posing
---------------------------------------------

The :math:`G`-metric is the *codomain inner product* of :math:`L`'s
composite :math:`V_{\rm bulk}\oplus V_{\rm trace}` space (step O.2b R5,
:ref:`g-adjoint`): a block-diagonal Hilbert metric with a
bulk phase-space block :math:`V_{\rm cell}\,w_n` and a partial-current
trace block :math:`|\Omega\cdot\hat n|\,w_n`. It is **intrinsic to the
streaming leaf** — :math:`L` carries its ``domain`` / ``codomain``
:class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace` with
the per-block ``inner_product_weights``, and the ``.H`` adjoint reads
them via the unchanged
:class:`~orpheus.numerics.operator.AdjointOperator` wrapper. It is
**NOT a posing-layer concern**: posing *arranges* leaves; the leaves
already *know* their metric through their composite space. The
:math:`G`-weighting is applied **once at the op level** — the
:class:`~orpheus.numerics.operator.AdjointOperator` wrapper reads
:math:`G` off the composite ``domain`` / ``codomain`` of the *summed*
operator and conjugates the whole sum
:math:`G^{-1}(\cdot)^{\mathsf T} G`, never re-applying it per leaf
(:ref:`g-adjoint`). Since P4.5 W-D the bulk leaves
:math:`C, S, F` advertise the same composite ``full_field_space`` as
:math:`L` / :math:`B` (so the *forward* within-group guard validates),
but their own ``apply_transpose``, where defined, is the metric-blind
Euclidean transpose — the metric is layered on once by the sum's
adjoint wrapper, so a leaf carrying the composite domain is no
double-application risk. Consequently the adjoint posing row gets the
correct :math:`G`-weighted transpose for free — the dagger functor
applied to a composite that already carries the metric — which is
precisely why the adjoint row adds no new machinery.


Honest scope — what is agnostic today, and what is not
------------------------------------------------------

The architecture is realized **minimally**: only the k-row and
:func:`~orpheus.numerics.eigenvalue.power_iteration` exist. Two honest
caveats record where the present code stops short of the ideal so a
future session does not mistake intent for fact.

* **The Layer-4 loop is not yet *literally* K/α-agnostic.** Today the
  eigenvalue scaling lives in the K-specific
  :meth:`compute_keff <orpheus.numerics.eigenvalue.EigenvalueSolver.compute_keff>`
  (production/absorption ratio) and the :math:`/k` placement in
  :meth:`compute_fission_source <orpheus.numerics.eigenvalue.EigenvalueSolver.compute_fission_source>`.
  :func:`~orpheus.numerics.eigenvalue.power_iteration` is agnostic only
  by *delegating* the estimate to the problem's ``compute_keff``.
  Making the loop literally agnostic (relocating the scaling into the
  algorithm as a Rayleigh-quotient update on
  :math:`A_{\rm loss}^{-1} M`, adding an ``apply_loss`` method, and
  renaming the K-flavoured Protocol methods to ``eigen_operator`` /
  ``mu_to_eigenvalue``) touches all five families' Protocol surface and
  is **the first step of the α-wave**, snapshot-bit-identity-gated. It
  is deferred because only K exists (premature to unify;
  *unify-after-two*).

* **The full-spectrum / shift-invert seam is reserved, not built.**
  The ``eigenvalue_method`` constructor selector on
  :class:`~orpheus.numerics.iteration.KEigenvalue`
  picks the Layer-4 algorithm. Only ``"power"`` is implemented; any
  other value raises :class:`NotImplementedError` at construction time.
  Full-spectrum Arnoldi / Krylov–Schur and shift-invert / FEAST (for
  interior eigenvalues — higher spatial harmonics) slot in at this
  exact dispatch point, consuming the same
  :math:`(A_{\rm loss}^{-1}, M)` boundary.

.. warning::

   The :math:`\alpha`-eigenvalue, adjoint, transient, and full-spectrum
   rows are **documented future seams, not implemented features**.
   There is zero :math:`\alpha` / transient / Arnoldi / shift-invert
   scaffolding in production transport today (the
   :mod:`orpheus.kinetics` solver is 0-D point kinetics, not a
   deterministic-transport :math:`\alpha`/transient solver). The
   layering exists so each lands as a pure addition — a new posing row
   (α / adjoint), a new leaf (:math:`T = 1/v`), and at most a new
   Layer-4 sibling (the transient time-integrator) — never a rewrite of
   the engine, the resolvent, or the existing leaves.


Verification status
--------------------

The discriminating gate for the canonical-loop refactor is
``tests/numerics/test_iteration.py::test_keigenvalue_matches_solve_sn_2g_slab``:
it stays green after
:class:`~orpheus.numerics.iteration.KEigenvalue` delegates to
:func:`~orpheus.numerics.eigenvalue.power_iteration`. This is the
**same-morphism evidence** — if the two had been different algorithms,
routing ``KEigenvalue``'s loop through ``power_iteration`` would change
the converged answer; bit-stable agreement on a **2-group** slab (≥2
groups is mandatory — a 1-group eigenvalue is the flux-shape-independent
ratio :math:`k = \nu\Sigma_f/\Sigma_a` and detects no operator error)
confirms they are one combinator at two layers. Because the
:class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` Protocol and the
five family solvers are untouched by the refactor, **every** family's
eigenvalue snapshot is trivially bit-identical across the change.

The converged *eigenvalue* (a solver-level claim) is verified against
the closed-form algebra-of-record
:func:`~orpheus.derivations.common.eigenvalue.kinf_and_spectrum_homogeneous`
(:math:`k = \lambda_{\max}(\mathbf{A}^{-1}\mathbf{F})`) for the
homogeneous reflective limit — a structurally-independent closed-form
pillar, not a code-to-code comparison.

.. _operator-algebra-declaration-contract:

The declaration contract — which equations can have implementers at all
=======================================================================

Most of this page's labelled equations now carry one or more
``.. implements::`` blocks naming the production symbol that realizes
them. Eight do **not**, and their emptiness is a *finding*, not a
backlog: each states something **about** the algebra rather than
computing something **in** it. This section records the distinction,
because it is not recoverable from a label and nothing in the toolchain
can currently see it.

Why the declarations exist
--------------------------

Nexus links code to equations two ways. A ``.. implements::`` directive
writes a **declared** edge at ``confidence = 1.0``; absent any
declaration, an inference heuristic mints **guessed** edges wherever a
symbol's name shares a token with an equation's label. Declaring *any*
implementer of an equation stands the guessing down for **that whole
equation** — so one directive silences every guess pointing at that
label.

The page's own graph shows the trade, and shows it as a *measurement*
rather than a projection. The graph happened to be rebuilt while this
pass was in flight, with three equations already declared and the other
37 not — so one snapshot holds both sides of the comparison:

* the **37 undeclared** equations carried **771** inferred
  ``implements`` edges between them — a median of **11** guesses each
  and a maximum of **58**, on :eq:`operator-apply-transpose`;
* the **3 declared** ones carried **zero**.

After the pass, on the rebuilt graph: the **32** declared equations
carry **57** directive edges and **zero** inferred ones — the stand-down
is total, not partial. **166** inferred edges remain, and every one of
them lands on the **8** equations that cannot be declared at all. That
residue is the subject of the next two subsections.

How coarse the guessing is deserves one concrete look.
:eq:`operator-solve` alone attracts **60** of the 166 — and only **5**
of those 60 symbols are named ``solve`` at all. The rest are matched by
the label's *other* token: five ``apply`` methods, three
``solve_fixed_source``, ``is_invertible``, ``is_adjointable``,
:func:`~orpheus.numerics.operator.outer`, and whole classes such as
:class:`~orpheus.numerics.operator.ZeroOperator` and
:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`.
A two-token label over a module named ``operator`` matches most of the
module.

(:math:`[M]` 2026-08-17, counting ``implements`` edges into
``math:equation:*`` for this page's labels and splitting them by the
edge's ``source`` attribute.)

⚠ The corollary is a contract on whoever edits this page: **declare
every implementer of an equation, or none of them.** A single directive
on an equation implemented in two places leaves the second one unlinked,
because the guess that used to cover it has stood down. **15 of the 32**
declared equations here have more than one implementer —
:eq:`streaming-action-cell-balance` has five,
:eq:`apply-solve-cell-resolvent` four, five more have three each — and
every one is declared exhaustively rather than representatively. In the
other direction, three symbols legitimately implement **two** equations
each
(:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.affine_scan_coefficients`
assembles the balance diagonal *and* divides by it;
:meth:`LegendreMomentTransfer.apply <orpheus.transport.operators.transfer.LegendreMomentTransfer.apply>`
is both the :math:`\ell`-sum and the carrier grid's role-changing edge;
:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face`
is both the :math:`\Gamma_-` half and the inflow predicate) — so the
directive count exceeds the symbol count by design.

.. note::

   Two of the ``:by:`` targets on this page carry an explicit
   ``py:data:`` node-id prefix (the ``Domain`` / ``Codomain`` type
   variables of :eq:`carrier-grid-operator-typing`). A bare dotted name
   is resolved against the function / method / class prefixes only, so a
   type variable — a ``py:data`` node — has to be named by its full node
   id, or the directive binds nothing and says so only in the build log.

The eight equations with no implementer, by kind
------------------------------------------------

No symbol can be pointed at for any of these without asserting a
falsehood at ``confidence = 1.0``.

.. list-table:: Equations that no code implements, and why
   :header-rows: 1
   :widths: 26 18 56

   * - Equation
     - Kind
     - Why nothing can implement it
   * - :eq:`apply-solve-parallel-identity`
     - Identity
     - The harmonic combination :math:`(L^{-1}+C^{-1})^{-1} =
       L(L+C)^{-1}C`, stated precisely to exhibit what is **not** the
       coupled inverse. It is unspellable in production:
       :class:`~orpheus.sn.operators.streaming.StreamingOperator`
       declares no ``inverse()`` and no ``solve``, so :math:`L^{-1}`
       does not exist as an object to compose.
   * - :eq:`apply-solve-neumann-series`
     - Identity
     - The splitting around :math:`C`, with :math:`C^{-1}L`, is never
       run. :class:`~orpheus.numerics.green_operator.GreenOperator` *is*
       a Neumann/splitting iteration — but around the sum's **leading
       term**, not around the collision diagonal; production always
       spells :math:`L + C` and promotes it to the direct-sweep
       :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`.
       Declaring ``GreenOperator`` here would be a *wrong* declaration,
       not a generous one.
   * - :eq:`apply-solve-neumann-expansion`
     - Identity
     - The term-by-term expansion of the row above; same reason.
   * - :eq:`apply-solve-denominator-inequality`
     - Identity (an inequality)
     - Nothing computes a non-equality.
   * - :eq:`solve-does-not-distribute`
     - Law, enforced by **absence**
     - Enforced structurally rather than checked:
       :class:`~orpheus.numerics.operator.OperatorSum` carries no
       ``solve`` verb, and
       :class:`~orpheus.sn.operators.streaming.StreamingOperator`
       declares no ``inverse()``. An absence is not an implementation —
       there is no symbol to point at, which is precisely why the
       guessing engine attaches unrelated ones.
   * - :eq:`streaming-as-tensor-product-sum`
     - Canonical form, deliberately not realized
     - :math:`[M]` its named type
       :class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`
       has **zero** production consumers: grepping ``orpheus/`` returns
       only its own class definition and guard messages, two
       ``numerics/__init__`` export lines, and one docstring mention.
       Streaming is realized as a sequential walk, and
       :doc:`/theory/foundations/operator_tensor_network` records why it
       resists a clean factorisation.
   * - :eq:`operator-solve`
     - Definition — a verb with no declaration site
     - The base-hosting rule forbids one. :math:`[M]`
       :class:`~orpheus.numerics.operator.LinearOperator` declares only
       ``apply``, and the three narrowing Protocols declare
       ``inverse()``
       (:class:`~orpheus.numerics.operator.SupportsInverse`),
       ``apply_transpose``
       (:class:`~orpheus.numerics.operator.SupportsAdjoint`) and
       ``assemble``
       (:class:`~orpheus.numerics.operator.SupportsAssembly`) — never
       ``solve``; no ``SupportsSolve`` exists. Contrast
       :eq:`operator-apply` and :eq:`operator-apply-transpose`, whose
       declaration sites *do* exist and are declared. The asymmetry
       among the three verb equations is **predicted** by the page's own
       base-hosting rule, not arbitrary.
   * - :eq:`carrier-grid-cell`
     - Definition — a taxonomy
     - :math:`\texttt{Carrier} = (\text{Representation}, \text{Role})`
       neither computes a quantity nor performs an operation. Its code
       counterpart is the flat multiple-inheritance leaf grid
       (:ref:`carrier-grid-flat-leaf-normal-form`) — a *structure*.
       Related but **not** implementing:
       :mod:`orpheus.transport.fields._bases` (the Representation ABCs)
       and the concrete role leaves themselves (the two Role mixins this
       entry once named, ``FluxRole`` and ``Displacement``, retired at
       campaign-1 CS3 — :ref:`cone-role-grid`). By contrast the *three
       edges* of
       :eq:`scattering-carrier-grid` are materialized methods and are
       declared — a diagram's arrows can have implementers where its
       nodes cannot.

What the kind predicts, and what the page's own prose does not
--------------------------------------------------------------

The eight kinds above sort cleanly, and the sort is the transferable
output of this audit:

.. list-table:: The kind of a statement predicts whether code can implement it
   :header-rows: 1
   :widths: 34 20 46

   * - Kind of statement
     - Implementable?
     - Reason
   * - Identity · Law · Canonical form
     - **No**
     - An identity between *quantities* has no carrier; a law enforced
       by absence has no symbol; a canonical form the tree declines to
       realize has no realization. There is nothing to point at.
   * - Typing rule · Definition
     - **Look for a declaration site**
     - A typing rule *can* have a materialized carrier — a class
       declaration, a Protocol parameter list, a set of typed methods —
       and where it does, that carrier is the implementer. A definition
       may or may not: :eq:`operator-apply` has a declaration site,
       :eq:`operator-solve` does not.

.. warning:: **The page's existing rationale prose is not a classifier —
   measured.**

   It is tempting to read the machine-readable ``.. (vv-status
   rationale)`` comment above each equation as already carrying this
   distinction, because the un-implementable rows do tend to say
   *"Mathematical identity"* while implementable ones tend to name a
   verb, a value, or a test file. :math:`[M]` across the 40 equations
   audited here, that reading does **not** survive contact:

   * only **28 of 40** carry a rationale block at all (**6** of the 8
     un-implementable rows, **22** of the 32 implementable ones), so a
     third of the page is silent either way;
   * the word *identity* appears in **5 of 6** rationale-bearing
     un-implementable rows — and in **11 of 22** implementable ones,
     i.e. half of them;
   * *"not a solver claim"* appears in **1 of 6** un-implementable rows
     and **5 of 22** implementable ones — pointing the wrong way;
   * a reference to a ``tests/`` file — the supposed implementable
     signal — appears in **2 of 6** un-implementable rows, including
     :eq:`operator-solve`, which has no implementer at all.

   The reason is a genuine ambiguity in the word, not sloppy writing: an
   **identity between quantities** (:eq:`apply-solve-parallel-identity`)
   cannot have a carrier, while an **identity between types**
   (:eq:`carrier-grid-operator-typing`,
   :eq:`harmonic-frame-is-galerkin`, :eq:`product-solve-reroute`) is
   *exactly* a claim about a class declaration and therefore can. Both
   are honestly called identities. So the useful narrowing feature for
   an inference engine is the **kind**, stated as a kind — not a
   keyword mined out of the rationale prose.

A last asymmetry, because it bounds what any amount of authoring can
achieve here: an equation that legitimately has **no** implementer still
attracts guesses, since the stand-down is triggered by a *declaration*
and there is no way to declare an absence. The eight rows above are the
part of this page's guessing load that cannot be retired by writing
directives — which is the case the kind taxonomy exists to solve.

Coverage of this pass
---------------------

The audit behind these declarations covered **40** of this page's
labelled equations: **32** declarable, realized by **57**
``.. implements::`` directives over **54** distinct symbols, plus the
**8** above.

Eight further labelled equations on this page were **outside** that
audit and carry no declaration yet:
``carrier-grid-interchange-witness``,
``tensor-product-axis-wise-composition``, ``sum-of-tensor-products``,
``octant-direct-sum-tensor-product``, ``eigen-standard-form``,
``eigen-resolvent``, ``eigen-k-posing`` and ``eigen-alpha-derivation``.
:math:`[M]` all eight attract **zero** edges of either kind today — no
guess and no declaration — so nothing is currently mis-attributed to
them; they are unfinished rather than wrong. The four ``eigen-*`` labels are the proper home of
:func:`~orpheus.numerics.eigenvalue.power_iteration` and the eigenvalue
tiers of :ref:`eigenvalue-posing` — declaring ``power_iteration``
against :eq:`operator-eigenvalue` instead would poach them, which is why
:eq:`operator-eigenvalue` names only the two ``compute_fission_source``
postings that build its right-hand side.



Development history
===================

This is a reverse-chronological (latest first) changelog of the major
**architectural** milestones in the operator algebra. Iteration-rate
work, gate counts, and intermediate replans are deliberately omitted —
see the GitHub issues and the per-phase plan files for that granularity.
Entries marked *(in development)* live on an unmerged feature branch and
have no landed merge-to-``main`` hash yet; trust ``git`` over this table
for merge status.

.. list-table::
   :header-rows: 1
   :widths: 10 50 12 28

   * - When
     - Architectural milestone
     - Issue
     - Where
   * - in dev
       (2026-08-28)
     - **A discretization scheme closes ONE axis — the angular closure
       owns its march, and the operator composes the two** (un-weld
       campaign, phase P4.9a).  The Morel--Montry angular march had a
       second, inline spelling inside
       :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.update`,
       and the twin was **forced**: ``transport`` is an L2 package and
       ``sn`` an L3 one, so ``FORBIDDEN_EDGES["transport"]`` (gated per
       module in ``tests/test_layer_imports.py``) made it impossible for
       the scheme to *call* the closure that owns the relation — it could
       only re-spell it.  P4.9a moves the responsibility up to the site
       that already sees both packages: the march has one production
       spelling
       (:func:`~orpheus.sn.angular.closure.march_psi_half_step`, with the
       batch kernel delegating to it and
       :meth:`~orpheus.sn.angular.closure.AngularClosureBase.advance_psi_half`
       as the per-cell entry), and the SN walk applies it.  The scheme
       now receives the angular axis's effect on the balance as two
       **assembled** contributions, ``angular_denom_term`` and
       ``angular_numer_upstream``, whose slab values are the neutral
       elements of the sums they enter — so the L2 layer names no
       Morel--Montry quantity at all.  Four consequences: the scalar
       twin ``cell_balance_terms`` retires onto
       :func:`~orpheus.transport.spatial.cell_balance.cell_balance_for_streaming`;
       the visit family goes purely spatial (``UpstreamState`` loses
       ``angular_upstream``, ``CellResult`` loses
       ``outgoing_angular_state``, ``CellVisit`` loses the closure stamp
       and with it ``SNMesh._make_cell_visit``); the closure **mints**
       its own scan constants instead of the cache deriving them; and
       Linear-Discontinuous's curvilinear refusal is re-keyed from a
       retired field's presence onto two value signals, which is a
       guard reachable without a mesh.  The two arithmetic forms of the
       march are welded by gate rather than unified by spelling — `[M]`
       they agree bitwise on only :math:`46`–:math:`51\,\%` of inputs
       (:math:`\max|\Delta| = 1.776\times10^{-15}`), and on
       :math:`100\,\%` exactly where :math:`\tau` is bitwise
       :math:`\tfrac12`.  See
       :ref:`sn-p49a-closure-owns-the-march`.  ⭐ Its sequel **P4.9b**
       (2026-08-28) finishes the arc one level up: the streaming operator
       is **posed** with both closures — three required fields, no
       defaults and no guards, with a ``pose`` classmethod reading the
       hub — so the walk consumes what it was handed instead of reaching
       back into the mesh, and the fused scan table becomes the solution
       strategy's lazily-resolved artifact rather than the operator's or
       the mesh's.  See :ref:`sn-p49b-operator-poses-with-closures` and
       the S\ :sub:`N` :doc:`/theory/methods/sn/history` entry.
     - #407
     - branch ``refactor/unweld-p49a-closure-owns-march``
   * - 2026-07-12
     - **The curvilinear ψ½ ray is System B of a 2×2 coupled block
       operator** — the augmented S\ :sub:`N` within-group problem is
       posed as a 2×2 block system over **System A** (the transport
       :class:`~orpheus.transport.full_field.FullField`) and **System B**
       (the ψ½ radial-characteristic ray). The four blocks — the seed
       :math:`A_{AB}`, the emission :math:`A_{BA}`, the radial march
       :math:`A_{BB}` (its direct Carlson solve IS the inverse) — are
       named operators assembled by the N-general
       :class:`~orpheus.numerics.coupled_system.CoupledOperator`
       machinery; :func:`~orpheus.sn.coupled_system.build_within_group_system`
       is the one production spelling, and System B's presence is
       **structural** (it exists iff the mesh carries a ray). See
       :ref:`coupled-block-operator`.
     - #280 / #282
     - ``main`` (``6732778a``)
   * - in dev
       (2026-07-04)
     - **The assembly axis — structural sparse emission as the third
       Mat-functor realization** (stencil-assembly campaign, Phase 2b).
       ``as_matrix`` gains a second realization beside apply-to-basis
       probing: leaves that know their stencil emit a
       :class:`~orpheus.numerics.assembled_operator.SparseAssembledOperator`
       (a :mod:`scipy.sparse` serialization of the operator, not a new
       COO-builder algebra), and the axis carries the full three-layer
       surface minted like inverse/adjoint —
       :attr:`~orpheus.numerics.operator.LinearOperator.is_assemblable`
       predicate, :class:`~orpheus.numerics.operator.SupportsAssembly`
       narrowing Protocol, :func:`~orpheus.numerics.operator.assemblable`
       ``TypeGuard`` bridge, and the eager
       :class:`~orpheus.numerics.operator.MissingAssembly` refusal. The
       composers recurse through the homomorphism laws (Sum → ``+``,
       Product → ``@``, Scaled → scalar ``·``; TensorProduct → ``kron``
       deferred, no consumer), and ``as_matrix`` **delegates** to the
       densified emission when :func:`~orpheus.numerics.operator.assemblable`
       (ruling R2), with the probing loop retained as
       ``_as_matrix_by_probing`` — the fallback AND the anti-tautology
       oracle the probed≡assembled gates force. First production consumer:
       the diffusion resolvent
       ``MatrixInverseOperator(FlattenedOperator(A, template))`` now
       LU-factors the **assembled** matrix automatically (probed↔assembled
       measured bit-identical, max :math:`|\Delta| = 0`; a Mode-11 sentinel
       proves the delegation executes). See
       :ref:`operator-algebra-assembly-axis`.
     - #272
     - merged to ``main`` (branch ``refactor/spatial-promotion-assembly``,
       deleted post-merge; first assembly commit ``83a0db7b``)
   * - in dev
       (2026-06-25)
     - **The carrier grid recognised as a double category, and the
       operator type made two-parameter** (Frame-projection campaign,
       P4.5). The transport carriers are identified as the cells of a
       :math:`(\text{Representation} \times \text{Role})` **double
       category** — horizontal 1-morphisms are the representation-changing
       frame faces :math:`M`/:math:`R` (role-generic), vertical
       1-morphisms are the role-changing cross sections :math:`C`,
       :math:`\Lambda`, :math:`F` (representation-generic), and scattering
       :math:`S = \tfrac{1}{W}(R\circ\Lambda\circ M) =
       \texttt{frame.conjugate}(\Lambda)` is the **2-cell**, with the
       existing 0-ULP windowed-vs-full crosscheck recognised as its
       interchange-law coherence witness (:ref:`carrier-grid-double-category`).
       The operator Protocol is widened to the honest two-parameter
       :class:`~orpheus.numerics.operator.LinearOperator` ``Protocol[Domain,
       Codomain]`` (``apply(x: Domain) -> Codomain``; the PEP-696 default
       ``Codomain = Domain`` keeps ``[V] ≡ [V, V]`` for the endomorphic
       majority; requires-python raised to ``>=3.13``) — **a grid cell IS an
       operator's** ``(Domain, Codomain)`` (:eq:`carrier-grid-operator-typing`).
       The accompanying structural finding — that a fully-typed
       ``Carrier[Representation, Role]`` is impossible (Role-arithmetic and
       Representation-shape each force a class; a parameterized carrier
       breaks the runtime units gate via erasure), so the flat MI leaves
       are the unique normal form — is documented at
       :ref:`carrier-grid-flat-leaf-normal-form`. **Realisation status:**
       the two-parameter operator type and the double-category framing have
       **landed on the branch**, and W-C/W-F have **confined the composite
       operator carrier to the timeless** :class:`~orpheus.transport.full_field.FullField`:
       the ``apply`` dispatch arms register on ``FullField`` (a driver's
       :class:`~orpheus.transport.timed_full_field.TimedFullField` iterate
       reaches them via MRO), and W-F realigned the
       heteromorphic-``apply`` ``@overload`` stubs
       (:ref:`heteromorphic-apply-typing`) for scattering/fission from
       ``TimedFullField`` to ``FullField`` to match that registration
       (typing-only; runtime byte-identical). The ``@overload`` confessions
       themselves are **kept** — they are the honest per-carrier surface, not
       a wart to retire; their deeper *dissolution* into a thin ``match``
       router over single-sourced primitives is parked on #261 (the
       Pattern M-vs-``match`` spelling question, :ref:`heteromorphic-apply-typing`).
       The secondary-carrier ``ScalarFlux`` arm is likewise **kept** as the
       typed entry-point for the #205 cross-method scalar consumers; the bare-
       ``ndarray`` arm is K-eigenvalue-live. The deeper *secondary-carrier-arm*
       collapse couples to the C/F/S core relocation and CP / MoC carrier
       unification (#261).

       ⛔ **Both "kept" verdicts were overturned on 2026-09-04 (CS4c step
       5), and the #261 spelling question was answered by neither
       candidate.** The relocation those verdicts waited on happened, and
       what it revealed is that the operand's kind is *implied by the ends
       the operator was constructed with* — so the router was not written
       thinner, it was deleted, and its decision moved to
       ``__post_init__`` (:ref:`cs4c-ends-select-the-body`). The
       ``ScalarFlux`` arm retired with it (ruling **R-3**): the typed
       scalar entry point #205 asked for already exists as the **energy
       binding** ``S.isotropic_energy``, so the arm was a Pattern-2 twin
       of it at one hop. The ``@overload`` confessions survive, but over
       real methods naming the two *bindings* rather than a carrier zoo
       (:ref:`pattern-m-history`).
     - #65 / #268 / #261
     - merged ``574cff81`` (branch ``refactor/operator-inverse-algebra`` —
       ``[M]`` 2026-08-19 an ancestor of ``main``)
   * - 2026-08-19
     - **Flux lives in the positive cone** :math:`K \subset V` — the
       affine field algebra is overturned. ``flux + flux`` becomes legal,
       a flux difference is the **same** leaf type carrying a signed
       value, cone membership becomes an element predicate and cone
       preservation stays the realization's flag; the ``FluxRole`` mixin
       and the whole ``transport/displacements/`` package retire, and the
       iterate diagnostics move onto
       :class:`~orpheus.numerics.convergence.IterationRecord`. Closes #331
       (operators are linear on :math:`V`). Full treatment, including the
       six-argument adjudication:
       :ref:`cone-overturn-adjudication`.
     - #331
     - merged ``f9d571b5`` (branch ``refactor/cone-field-algebra``)
   * - 2026-06-08
     - **Flux states typed as an affine space; the iterate increment is a
       typed displacement.** ``flux − flux`` minted a ``Displacement``,
       ``flux ⊕ displacement`` was the torsor update, and ``flux + flux``
       was a :class:`TypeError` — the #201 dimensional gate as a *type*
       consequence. The Role axis of the carrier grid.
       ⛔ **OVERTURNED 2026-08-19** — see the row above and
       :ref:`cone-the-overturned-affine-design`.
     - #208 / #201
     - ``main`` (Wave O step O.2)
   * - 2026-06-07
     - **The 2-D Cartesian SI iterate lives in moment space** — the
       within-group source-iteration fixed point consumes :math:`\psi`
       only through its flux moments, so the *persistent* iterate is held
       as the moment tensor
       :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
       rather than the full per-ordinate
       :class:`~orpheus.transport.fields.angular_flux.AngularFlux`; the
       source stays bit-identical (shared :math:`R\,\Lambda`
       reconstruction), only the SI convergence test moves to the moment
       :math:`L^2` (principled-equivalence). 2-D Cartesian, interior-bulk
       only. See :ref:`sn-angular-windowing`.
     - #205
     - ``main`` (Phase 5a, ``93807aa`` / ``b97d4f9`` / ``13ca001``)
   * - 2026-06-05
     - **The eigenvalue problem is layered into four tiers** (leaves /
       posing / resolvent / algorithm): the generalized eigenproblem
       :math:`A_{\rm loss}\,\psi = \lambda\,M\,\psi` (k-row
       :math:`A_{\rm loss} = L+C-S-B`, :math:`M = F`) has its power-method
       realization as the dominant eigenpair of the resolvent
       :math:`A_{\rm loss}^{-1} M`;
       :func:`~orpheus.numerics.eigenvalue.power_iteration` is the
       canonical Layer-4 algorithm and
       :class:`~orpheus.numerics.iteration.KEigenvalue` delegates its loop
       to it (one loop, Cardinal Rule 2). See :ref:`eigenvalue-posing`.
     - —
     - ``main`` (``650032e`` / ``7603c8e``)
   * - 2026-06-05
     - **The Hilbert adjoint** ``op.H`` **is the metric-correct
       G-adjoint** :math:`A^{\dagger} = G^{-1} A^{\mathsf T} G` over the
       block-diagonal
       :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
       metric (:math:`V_{\rm bulk} \oplus V_{\rm trace}`; singular trace
       block pseudo-inverted) — NOT the Euclidean transpose; applied
       **once at the operator level** and raising
       :class:`~orpheus.numerics.operator.MissingAdjoint` eagerly rather
       than silently going Euclidean. See :ref:`g-adjoint`.
     - —
     - ``main`` (Wave O step O.2b R5, ``5c06196``)
   * - 2026-06-04
     - **2-D Cartesian eigenvalue problems solve via BOTH inner solvers**
       — the source-iteration inner is the geometry-agnostic structural
       twin of the Krylov inner: identical composite RHS, identical loss
       decomposition (the invertible resolvent :math:`L + C` plus the two
       lagged coupling gains — scattering :math:`S`, boundary reflection
       :math:`B` — on the variadic driver), differing **only** in the
       iteration driver. See :ref:`bc-extraction-2d-si-krylov-twin`.
     - #208
     - ``main`` (Wave O 2-D SI Phase A)
   * - 2026-06-04
     - **The interior cell-face angular fluxes are a 1-cochain**
       :math:`C^1_{\rm int}` — with the boundary trace
       (:math:`C^1_\partial`) it biproduct-decomposes the full face
       cochain :math:`C^1 = C^1_{\rm int} \oplus C^1_\partial`; the
       sweep's seed/absorb are the typed trace operators :math:`\iota_*`
       / :math:`\iota^*` (``absorption = identity`` is the biproduct law
       :math:`\iota^* \circ \iota_* = \mathrm{id}`). The ``WavefrontFlux``
       carrier retired at S6.4(f). See :ref:`wavefront-flux-cochain`.
     - #208 / #222
     - ``main`` (Wave O #205 Phase 5)
   * - 2026-06-03
     - **The boundary law** :math:`B` **becomes a first-class sibling
       operator** and every operator's ``.apply`` output is typed as a
       *source/sink* (the bulk → :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`,
       the boundary → :class:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink`),
       completing the operator-output role typing
       (:ref:`bc-extraction-operator-output-typing`).
     - #208
     - ``main`` (Wave O steps O.4a.2 / B.5.2)

