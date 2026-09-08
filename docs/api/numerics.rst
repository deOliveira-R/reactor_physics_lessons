Numerical Methods (``numerics``)
================================

The :mod:`orpheus.numerics` package holds the algorithm-agnostic
numerical primitives that every deterministic solver shares. Its
job is to keep one copy of "how to converge an eigenvalue
problem" in the codebase rather than replicating the loop in each
of the SN, CP, MOC, diffusion, and homogeneous drivers.

.. contents::
   :local:
   :depth: 2


Power Iteration
---------------

The criticality eigenvalue problem

.. math::

   A\,\phi \;=\; \frac{1}{k}\,F\,\phi

has a spectrum of eigenvalues
:math:`k_0 > k_1 > k_2 > \dots`. Only the **dominant eigenvalue**
:math:`k_0 = k_{\rm eff}` and its eigenvector :math:`\phi_0` are
physically meaningful: by the Perron–Frobenius theorem
:math:`\phi_0` is the unique non-negative eigenvector, while all
higher harmonics change sign in space.

:func:`~orpheus.numerics.eigenvalue.power_iteration` converges to
:math:`(k_0, \phi_0)` by repeatedly applying the transport
operator to an estimate of :math:`\phi`:

.. math::

   \phi^{(n+1)} \;=\; A^{-1}\,\frac{1}{k^{(n)}}\,F\,\phi^{(n)},
   \qquad
   k^{(n+1)} \;=\; \frac{\lVert F\,\phi^{(n+1)}\rVert}
                         {\lVert L\,\phi^{(n+1)}\rVert}.

The convergence rate is governed by the **dominance ratio**
:math:`|k_1/k_0|`; problems with a narrow spectral gap (large
lattices, near-critical systems with weakly coupled regions)
converge slowly and may benefit from Chebyshev or Wielandt
acceleration — not currently implemented in ORPHEUS.

**Normalisation.**
The returned eigenvector has *arbitrary* absolute scale. Power
iteration preserves shape but not magnitude — callers that need
absolute flux (e.g. for power calibration, dose calculations) must
post-normalise, typically by fixing the total integral fission
source or the total power deposition.

**Direct and Rayleigh-quotient siblings.**
:func:`~orpheus.numerics.eigenvalue.power_iteration` is the iterative engine
for large, sweep-only operators that are never densely formed.  Two siblings
in :mod:`orpheus.numerics.eigenvalue` solve the same generalised
eigenproblem by different realisations:
:func:`~orpheus.numerics.eigenvalue.direct_eigenvalue` forms the dense
resolvent :math:`\mathbf{A}^{-1}\mathbf{F}` from a posed
:math:`(\mathbf{A}, \mathbf{F})` pair via :func:`numpy.linalg.solve` and
returns the EXACT dominant eigenpair — the right tool for small, densifiable
operators — and
:func:`~orpheus.numerics.eigenvalue.rayleigh_quotient_iteration` polishes an
eigenpair *estimate* to the NEAREST eigenpair superlinearly (bordered /
augmented-Newton).  Both direct engines terminate in one shared
Perron–Frobenius extraction primitive,
:func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`, which takes a
materialized resolvent :math:`\mathbf{M}` — *however* it was formed, whether
from the ``(A, F)`` pair or through the operator algebra as in the homogeneous
:math:`k_\infty` solve — selects the dominant eigenpair, sign-normalises the
mode, and rejects a complex dominant as a malformed problem.  The full
three-engine comparison, and the pure-math verification principle that lets a
production solver and its oracle share an engine without contamination, is at
:ref:`three-eigenvalue-engines`.


The EigenvalueSolver Protocol
-----------------------------

Every deterministic solver plugs into the same power iteration
loop by implementing the
:class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` protocol.
The protocol has five methods and one structural contract:

* ``initial_flux_distribution`` — return a flux guess. Most
  solvers use a flat unit array; MOC uses a cell-averaged flat
  angular flux.
* ``compute_fission_source`` — build
  :math:`Q_f = \chi\,(\nu\Sigma_f\,\phi)/k`. Pure function of the
  current flux and eigenvalue.
* ``solve_fixed_source`` — apply :math:`A^{-1}` to the fission
  source. **Scattering and (n,2n) sources are assembled *inside*
  this method** because they need to be updated between inner
  iterations (source iteration in SN, Gauss–Seidel in CP, etc.).
  This is the single most important structural decision in the
  protocol: it lets each solver manage its own inner iteration
  strategy without leaking through to the outer loop.
* ``compute_keff`` — update the eigenvalue from the current
  :math:`\phi`. For reflective lattices the leakage term is zero;
  for whole-core diffusion it is not.
* ``converged`` — stopping test. Typical tolerance
  :math:`10^{-6}` on :math:`|\Delta k|`; richer tests on flux
  L2 norm are also used.

**Reference implementations** (each satisfies the protocol and is
tested against the power-iteration loop without any solver-specific
glue):

* :class:`orpheus.cp.solver.CPSolver` — collision probability.
* :class:`orpheus.sn.solver.SNSolver` — discrete ordinates.
* :class:`orpheus.moc.solver.MOCSolver` — method of characteristics.
* :class:`orpheus.diffusion.solver.DiffusionSolver` — 1-D two-group
  diffusion.

The infinite **homogeneous** solver is the one deterministic solver
that does **not** implement this protocol: with no spatial coupling
its loss operator is a single :math:`G \times G` dense block, so
:func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite` takes
the dominant eigenpair of :math:`\mathbf{A}^{-1}\mathbf{F}` **directly**,
with no power iteration.  It spells the resolvent in the operator algebra —
``K = MatrixInverseOperator(loss) @ production`` (the first production
consumer of
:class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`) —
and extracts the eigenpair from the materialized :math:`[\mathbf{K}]` via the
shared :func:`~orpheus.numerics.eigenvalue.dominant_eigenpair`.  See
:ref:`theory-homogeneous` and :ref:`three-eigenvalue-engines`.


Operator Algebra (Wave A)
-------------------------

The :mod:`orpheus.numerics.operator` module installs the matrix-free
operator-algebra primitives consumed by every solver. See
:ref:`operator-algebra` for the design rationale, the three-layer
operator surface (predicate / operator-returning method / realization
verb), and tensor-product algebra.

Tensor-product primitives (Wave 0 of SN performance plan, Wave T
consumers landed May 2026):

* :class:`~orpheus.numerics.operator.DiagonalOperator` — diagonal
  multiplication on a tagged tensor axis. The ``AngularWeightMatrix``
  :math:`W` of Grand Report v3 §9 is
  ``DiagonalOperator.from_measure(quad.measure, axis=0)``.
* :class:`~orpheus.numerics.operator.TensorProductOperator` —
  per-axis tensor product :math:`A \otimes B \otimes \cdots`. Built
  via the ``&`` dunder; carries axis tags and the closure laws
  :math:`(A \otimes B)^* = A^* \otimes B^*`,
  :math:`(A \otimes B) \circ (C \otimes D) = (A \circ C) \otimes
  (B \circ D)`. See :ref:`tensorial-framing`.
* :class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`
  — :math:`\sum_k A_k \otimes B_k \otimes \cdots`; the §15.2
  *aspirational* canonical scattering / streaming form. **Zero
  production consumers** today; see :ref:`tensor-network-decomposition`
  for why (the MA-Q1 master condition: coupled physics falls back to
  :class:`OperatorSum` over bespoke leaves, not SOTP).
* :class:`~orpheus.numerics.operator.OperatorSum` — the additive
  composer :math:`A + B`. Wave T promoted this to load-bearing
  status (T.3 scattering kernel; the T.4 per-direction streaming split
  was retired in #238 — the fused matvec walks both directions in one
  pass); see :ref:`tensor-network-shape-table`.
* :class:`~orpheus.numerics.operator.RankOneOperator` — the rank-1 dyad
  :math:`|v\rangle\langle w|`: a reconstruction **column** ``v`` and a
  :class:`~orpheus.numerics.functional.Functional` **row** ``⟨w|``, with
  ``apply(x) = v * functional.evaluate(x)`` (the matvec routes *through*
  the functional, not a parallel reduction). Built by the free function
  :func:`~orpheus.numerics.operator.outer`\ ``(reconstruction,
  functional)``; native to the multigroup fission emission
  :math:`F = |\chi\rangle\langle\nu\Sigma_f|` (:attr:`IsotropicFission.kernel`,
  the fission energy binding's one arithmetic home since CS4c step 4;
  ``FissionOperator.kernel`` delegates to it).
  A genuine :math:`M\times K` rank-1 operator is legal (no same-shape
  constraint between column and row).
* :func:`~orpheus.numerics.operator.outer` — the universal rank-1
  constructor ``outer(reconstruction, functional)``, the readable verb
  for :math:`|v\rangle\langle w|`. Exported from
  :mod:`orpheus.numerics`.
* :class:`~orpheus.numerics.functional.InnerProductFunctional` — the
  generic co-vector :math:`\langle w, \cdot\rangle` (``evaluate(x) = (w *
  x).sum(axis, keepdims=True)``), the row-factor of a rank-1 operator.
  The transport
  :class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
  specialises it. Exported from :mod:`orpheus.numerics`.


Consumer matrix for tensor-product primitives (post Wave T)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The table below lists the production consumers as of Wave T close-out
(May 2026). The full architectural narrative — including the MA-Q1
master condition that decides between :class:`TensorProductOperator`,
:class:`SumOfTensorProductsOperator`, and
:class:`OperatorSum`-of-bespoke-leaves — is at
:ref:`tensor-network-decomposition`.

.. list-table:: Production consumers of the tensor-product primitives
   :header-rows: 1
   :widths: 28 14 28 30

   * - Primitive
     - Consumers (count)
     - Examples
     - Notes
   * - :class:`~orpheus.numerics.operator.TensorProductOperator`
     - 6
     - 5 BC realizers (vacuum / specular / white / albedo / periodic
       via Wave T T.1 ``& IdentityOperator()`` wrap); fission kernel
       (Wave T T.2, ``IsotropicFission.kernel = outer(χ,
       ReactionRateFunctional(νΣ_f)) & IdentityOperator()`` — the
       energy binding since CS4c step 4)
     - Six clean-TP production instances. The MA-Q1 master condition
       is satisfied: each consumer factors as disjoint per-axis
       operations.
   * - :class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`
     - 0
     - (no production consumers)
     - The §15.2 SOTP form is aspirational; Wave T T.3 (scattering)
       and T.4 (streaming) both fell back to :class:`OperatorSum`
       over bespoke leaves per the MA-Q1 master condition. See
       :ref:`tensor-network-decomposition` for the per-substep rationale.
   * - :class:`~orpheus.numerics.operator.OperatorSum`
     - many (the load-bearing composer)
     - The within-group loss ``A_AA = (L+C) - S - N2N - B_a``
       (:func:`~orpheus.sn.coupled_system.build_within_group_system`
       — the ``- N2N`` term is explicit since the CS4c step-3
       extraction, :eq:`sn-within-group-with-n2n`);
       :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`
       (a subclass, pinning the ``L + C`` legs);
       the diffusion loss ``leakage + collision - scattering - boundary``
       (:mod:`orpheus.diffusion.solver`); the shared isotropic energy
       kernel ``IsotropicScattering + IsotropicN2N``
     - Every ``+``/``-`` in the operator algebra lands here (the
       :class:`LinearOperator` dunder defaults), so the honest
       :math:`A = L + C - S - N_{2n} - B` composition IS a left-nested
       ``OperatorSum``.  It is also where the *context-dependent*
       groupings live: diffusion's ``S`` IS
       ``IsotropicScattering + IsotropicN2N``, and the S\ :sub:`N`
       builder's ``K_iso`` IS
       ``S.isotropic_energy + N2N.isotropic_energy`` — two different
       bundlings of one pair of leaves, each written at its own
       composition site rather than fixed inside an operator.  (The
       :math:`N_{2n}` accessor was ``.energy`` until #426 step 2,
       2026-09-04, when the two gains became roles of ONE binding and
       the P0 energy leaf became a member of the shared core; the
       spelling is now the same on both.)
       The T.4 streaming per-direction split
       (``M_spatial`` as an :class:`OperatorSum` of two per-direction
       summands) was retired in #238 — it had no production consumer;
       the fused matvec
       (:meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`)
       walks both directions in ONE bidirectional pass.
       See :ref:`tensor-network-orchestrated-apply`.
   * - :class:`~orpheus.numerics.operator.RankOneOperator`
     - 1
     - Wave T T.2 fission kernel
       (:attr:`IsotropicFission.kernel` first factor of the
       :class:`TensorProductOperator`); built via
       :func:`~orpheus.numerics.operator.outer`
     - The dyad :math:`|\chi\rangle\langle\nu\Sigma_f|`: encodes the
       group-axis contraction-then-broadcast
       :math:`(F\,\phi)_g = \chi_g\,\sum_{g'}\nu\Sigma_{f,g'}\,\phi_{g'}`
       as a typed primitive, with the contraction owned by the
       :class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
       row-factor.

.. note::

   Wave T T.3 originally shipped a private per-ℓ leaf
   (``_PerLegendreOrderScattering``) summed into
   :attr:`ScatteringOperator.kernel` via an :class:`OperatorSum`, exposed
   as ``ScatteringOperator.kernel_summands``.  **Both were retired** in
   ``93807aa7``, which factored the anisotropic path onto the shared
   :math:`R\circ\Lambda` moment→source primitive.  Today the kernel is a
   single :class:`~orpheus.numerics.operator.OperatorProduct` —
   ``frame.conjugate(LegendreMomentTransfer(..., skip_l0=True))``, i.e.
   :math:`R \circ \Lambda_{\ell\ge1} \circ M` with **one** shared
   :math:`\Lambda` rather than a per-ℓ summand family.  (The class was
   ``LegendreMomentScattering`` until #426 step 2, 2026-09-04; it is now
   the transfer family's, over either channel's field.)

.. note::

   Wave T also shipped a per-direction streaming split (``M_spatial`` /
   ``M_angular_redist`` over ``_SpatialSweepDirection`` /
   ``_MSpatialOperatorSum`` / ``AngularRedistributionOperator``) as
   separately-applicable typed leaves to anticipate Wave-O adjoint /
   DSA consumers. **#238 retired that split**: no production code ever
   applied the leaves separately (the #240 adjoint uses the fused
   ``loss_action_transpose``), so the streaming + curvilinear
   Morel–Montry angular redistribution is computed IN-SWEEP inside the
   fused matvec
   (:meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`),
   verified end-to-end by the anisotropic curvilinear MMS
   (:ref:`sn-mms-curvilinear-aniso-verification`).

The leading-underscore primitives are intentionally private (the
public surface is via the
:attr:`~orpheus.transport.operators.transfer.TransferOperator.kernel`
and
:attr:`~orpheus.transport.operators.transfer.TransferOperator.full_transfer_kernel`
properties on the operator classes — since #426 step 2 on the shared
transfer core, so :math:`S` and :math:`N_{2n}` expose one surface). Wave O (`Issue #208
<https://github.com/deOliveira-R/ORPHEUS/issues/208>`_) will introduce
``BulkOperator`` / ``FullOperator`` / ``BoundaryOperator`` Protocols
that may promote some of these to public status if a downstream
consumer surfaces.

Discrete measures and partition (Wave 0 of SN performance plan):

* :class:`~orpheus.numerics.measure.DiscreteMeasure` — atomic
  measure :math:`\mu = \sum_i w_i\,\delta_{x_i}` on a measurable
  space; carries integration, tensor product, direct sum,
  pushforward, restriction, and the new
  :meth:`~orpheus.numerics.measure.DiscreteMeasure.partition_by`
  primitive (the inverse of direct sum). See
  :ref:`discrete-measure-partition`.
* :class:`~orpheus.numerics.measure.DiscreteMeasurePartition` —
  a partition entry returned by
  :meth:`partition_by`. Carries label, indices into the parent,
  and the restricted measure.

Discrete frame, basis, and harmonic projection (Frame/Basis carve;
discipline-type hierarchy, GitHub #268):

* :class:`~orpheus.numerics.frame.FrameBase` — the abstract discrete
  frame: binds a :class:`~orpheus.numerics.basis.Basis` to a
  :class:`~orpheus.numerics.measure.DiscreteMeasure` and emits the
  ``analysis`` (:math:`M = T`) and ``reconstruction`` (:math:`R`)
  faces as :class:`~orpheus.numerics.operator.LinearOperator` views.
  Carries the **discipline-free** mechanics — the trial table, the two
  spaces, the reconstruction face, the analysis-face wiring; the single
  abstract hook is the test basis. The coefficient-extraction verb
  :meth:`~orpheus.numerics.frame.FrameBase.project` (:math:`G^{-1}M`, the
  homogenise / condense verb) normalises by a **row-sum** Gram probe,
  valid only for a row-sum-collapsible trial; it **refuses**
  (:class:`~orpheus.numerics.operator.NotInvertible`) a trial whose
  :attr:`~orpheus.numerics.basis.base.Basis.gram_structure` is
  :attr:`~orpheus.numerics.basis.base.GramStructure.DENSE` (the dense
  :math:`(MR)^{-1}M` least-squares solve is unbuilt — `GitHub #275
  <https://github.com/deOliveira-R/ORPHEUS/issues/275>`_).
* :class:`~orpheus.numerics.frame.PetrovGalerkinFrame` — the general
  discipline: an explicit ``test_basis`` distinct from the trial basis
  (test ≠ trial), so :math:`M^* \ne R`. Flux-weighted spatial
  homogenisation and spectrum-weighted energy condensation are the
  headline consumers.
* :class:`~orpheus.numerics.frame.GalerkinFrame` — the Galerkin
  specialisation (``test is trial``), which *strengthens* the base
  promise to :math:`\Pi^* = R`. The angular spherical-harmonic
  projection (``quadrature.angular_frame(L)``) is the canonical
  pure-Galerkin frame; its SH case is a 4π-tight frame.
* :class:`~orpheus.numerics.basis.Basis` — the synthesis (trial)
  side ABC: tabulate, naked synthesis :math:`S_0`, the three
  weighted contractions, the discrete Gram, and the
  :attr:`~orpheus.numerics.basis.Basis.gram_structure` declaration
  (below). Defaults to
  :attr:`~orpheus.numerics.basis.base.GramStructure.DENSE` — the safe
  refusal (a new basis must consciously declare it row-sum-collapsible).
* :class:`~orpheus.numerics.basis.GramStructure` — the trial basis's
  **projection-validity declaration** (the precondition
  :meth:`~orpheus.numerics.frame.FrameBase.project`'s row-sum probe needs,
  carried by the *type* not a docstring):
  :attr:`~orpheus.numerics.basis.base.GramStructure.DIAGONAL`
  (disjoint-support — orthogonal harmonics, nested indicators),
  :attr:`~orpheus.numerics.basis.base.GramStructure.PARTITION_OF_UNITY`
  (overlapping rows summing to 1 — the fractional
  :class:`~orpheus.numerics.basis.OverlapBasis`; ``MR`` not diagonal but
  :math:`R\mathbf 1=\mathbf 1` still collapses the probe), or
  :attr:`~orpheus.numerics.basis.base.GramStructure.DENSE` (neither — the
  row-sum probe is wrong; ``project`` refuses it).
* :class:`~orpheus.numerics.basis.IndicatorBasis` — the
  piecewise-constant (P0) cell/group-indicator basis: a one-hot
  membership table built by ``searchsorted`` (declares
  :attr:`~orpheus.numerics.basis.base.GramStructure.DIAGONAL`). The trial
  side of spatial homogenisation and the *nested* energy-condensation
  degenerate.
* :class:`~orpheus.numerics.basis.OverlapBasis` — the
  partition-of-unity **fractional**-membership generalisation of
  :class:`~orpheus.numerics.basis.IndicatorBasis` (a straddling fine cell
  belongs *fractionally* to each coarse cell it overlaps). The trial
  side of *non-nested* energy condensation
  (:ref:`sn-condensation-fractional-overlap`); the one-hot
  :class:`~orpheus.numerics.basis.IndicatorBasis` is its nested
  degenerate. Overrides exactly one method (``evaluate``, returning the
  precomputed overlap table) — a no-op extension through the inherited
  contractions — and declares
  :attr:`~orpheus.numerics.basis.base.GramStructure.PARTITION_OF_UNITY`.
  Two table diagnostics carry the re-binning provenance:
  :attr:`~orpheus.numerics.basis.overlap_basis.OverlapBasis.dominant_column`
  (the ``argmax`` containing-coarse map) and
  :attr:`~orpheus.numerics.basis.overlap_basis.OverlapBasis.fractional_columns`
  (the coarse columns that received a strictly-fractional contribution —
  empty for a nested table).
* :class:`~orpheus.numerics.basis.WeightedIndicatorBasis` — the
  Petrov-Galerkin **test**-side basis: a weight (flux / spectrum /
  production) carried as an *analysis* weight on the cell/group
  indicator. The flux-weighted test side of homogenisation and the
  spectrum-weighted test side of condensation.
* :class:`~orpheus.numerics.basis.SphericalHarmonicBasis` — the
  first concrete basis: real spherical harmonics on :math:`S^2`,
  carrying the no-:math:`4\pi/(2\ell+1)`-prefactor convention (the
  addition theorem reads
  :math:`\sum_m Y_\ell^m Y_\ell^m = P_\ell(\Omega \cdot \Omega')`),
  the :attr:`~orpheus.numerics.basis.SphericalHarmonicBasis.addition_theorem_factor`
  :math:`(2\ell+1)`, and the
  :meth:`~orpheus.numerics.basis.SphericalHarmonicBasis.evaluate`
  :math:`Y_\ell^m(\hat\Omega_n)` evaluator.

The abstract analysis / reconstruction operator **roles**
(:mod:`orpheus.numerics.projection`). These carry the operator role
only; the **discipline** (Galerkin vs Petrov-Galerkin) is the frame's
TYPE, never a marker on the role (GitHub #268 retired the
``GalerkinProjection`` / ``PetrovGalerkinProjection`` marker ABCs):

* :class:`~orpheus.numerics.projection.AnalysisOperator` — the abstract
  fine→coarse (measured) role :math:`M : V \to W`. The concrete
  realisation is a frame's ``analysis`` face.
* :class:`~orpheus.numerics.projection.ReconstructionOperator` —
  the abstract coarse→fine role :math:`R : W \to V`. The concrete
  realisation is a frame's ``reconstruction`` face.

See :ref:`galerkin-projection` for the discrete-frame narrative,
the discipline-type hierarchy, and the cross-method consumer table;
:ref:`spherical-harmonics` for the SH convention and addition theorem.


Field algebra (Depth B, step D-A)
---------------------------------

.. _field-algebra:

The :class:`~orpheus.numerics.field.Field` ABC is the L1 algebraic
base of every typed transport field — angular flux, scalar flux,
spherical-harmonic moments, boundary traces, sources, residuals. It
codifies the Grand Report v3 §5.5 / §32.5 prescription:

   *Every typed transport field is the pair ``(values, space)`` with
   closed same-CLASS, same-SPACE arithmetic.*

The ABC carries the dunder algebra (``+``, ``-``, ``-`` unary, scalar
``*``, scalar ``/``) and the diagnostics (``linf``, ``l2``,
``inner_product``) inherited unchanged by every concrete subclass.
Subclasses add domain-specific fields (``mesh``, ``boundary``,
``history``) on top of the ``(values, space)`` base; the algebra is
inherited verbatim via :func:`dataclasses.replace`. The same
hand-coded dunder skeleton that previously lived in six separate
classes (``AngularFlux``, ``ScalarFlux``, ``HarmonicMomentFlux``,
``AngularBoundaryFlux``, ``ScalarSourceSink``, ``AngularSourceSink``) is
consolidated here — Cardinal Rule 2 (single source of truth).

**Three-layer dimensional enforcement.** Dimensional consistency is
gated at three layers, each with a different cost / coverage trade-off:

* **Layer 1 — class identity.**
  :meth:`~orpheus.numerics.field.Field._check_partner` rejects
  ``type(self) is not type(other)`` before any value comparison. This
  is the *primary* gate: even when units match (an ``AngularSourceSink``
  and an ``AngularResidual`` may both carry
  :math:`1/(\mathrm{cm^2 \cdot s \cdot sr \cdot eV})`),
  cross-class arithmetic raises by construction. Same units gives
  PERMISSION to add in linear algebra; it does not give MEANING.
* **Layer 2 — construction-time dimensional check.** Solvers like
  :class:`~orpheus.numerics.iteration.SourceIteration` do a single
  :math:`O(1)` ``pint.Unit.dimensionality`` comparison per operator at
  ``__init__`` to verify the operator algebra is dimensionally sound
  before any iteration runs. Cost: microseconds per build. ALWAYS
  runs (both default ``python -O -m pytest`` and ``pytest``).
* **Layer 3 — defensive assert in dunders.** Inside
  ``_check_partner``, ``assert self.space.units == other.space.units``
  catches the rare class/units misdesign (two instances of the same
  class whose spaces nonetheless carry inconsistent unit STRINGS — e.g.
  one in :math:`1/\mathrm{cm}^2`, one in :math:`1/\mathrm{m}^2` — same
  dimensionality, different scaling). Stripped in ``-O`` mode; defense
  in depth during development.

Together these layers make dimensional-mismatch bugs unrepresentable
without paying the cost of full ``pint.Quantity`` arithmetic on every
ndarray operation.

**Class identity for cross-class same-units operations.** Two mechanisms
work together when distinct Field subclasses share a dimensional
signature (e.g. ``AngularResidual`` and ``AngularSourceSink`` both carry
``1/(cm³·s·sr)``):

* **The class gate forbids cross-class arithmetic** even when units
  match — ``angular_residual - angular_source`` RAISES. Same units grant
  permission to add in linear algebra, not meaning.
* **A role transition goes through a named composition.** The transport
  balance :math:`r = A\psi - q` differences two *same-class*
  ``AngularSourceSink`` operands (the operator output :math:`A\psi` and
  the external source :math:`q`), but the *result* is a residual (a
  defect), not a source. Bare same-class subtraction typechecks yet
  MIS-TYPES the defect; each residual leaf's ``from_balance`` factory
  makes the transition explicit and lands the result in the correct
  residual class:

  .. code-block:: python

      # same-class subtraction typechecks but MIS-TYPES the defect:
      wrong = operator_output - q_ext           # AngularSourceSink (!)

      # named composition — correctly typed as the residual role:
      residual = AngularResidual.from_balance(lhs=operator_output, rhs=q_ext)

The named-composition discipline IS what makes the Field algebra
sound under physical interpretation — ``coding-elegance`` Pattern 4
(illegal states unrepresentable) takes its strictest form here.

See the Depth B plan
(``.claude/plans/depth_b_field_on_function_space.md``) §3.2 for the
ABC spec, §3.7 for the singledispatch policy that consumes ``Field``
in operator apply, §5 for the Layer 2 construction-time check, and
§7.5 for the full CI matrix.


API Reference
-------------

.. automodule:: orpheus.numerics.axis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: orpheus.numerics.convergence
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: orpheus.numerics.coupled_system
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: orpheus.numerics.eigenvalue
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: orpheus.numerics.field
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: orpheus.numerics.functional
   :members:
   :undoc-members:
   :show-inheritance:

The :class:`~orpheus.numerics.operator.LinearOperator` Protocol,
its three-layer inverse/adjoint surface, and the composition /
tensor-product primitives are documented at :ref:`operator-algebra`
(theory page).
The :class:`~orpheus.numerics.functional.Functional` Protocol (the §5.6
suffix-law co-vector), its concrete
:class:`~orpheus.numerics.functional.InnerProductFunctional`, and the
rank-1 dyad constructor :func:`~orpheus.numerics.operator.outer`
(:math:`|v\rangle\langle w|`) are documented at
:ref:`functional-category`.
The discrete :class:`~orpheus.numerics.frame.FrameBase` hierarchy
(:class:`~orpheus.numerics.frame.PetrovGalerkinFrame` →
:class:`~orpheus.numerics.frame.GalerkinFrame`), the
:class:`~orpheus.numerics.basis.Basis` hierarchy, and the
discipline-as-type rationale are documented at
:ref:`galerkin-projection`. The :math:`Y_\ell^m` evaluator and the
no-:math:`4\pi/(2\ell+1)`-prefactor convention are documented at
:ref:`spherical-harmonics`. The
:meth:`partition_by` primitive on
:class:`~orpheus.numerics.measure.DiscreteMeasure` is documented at
:ref:`discrete-measure-partition`. The
:class:`~orpheus.numerics.metric.HilbertMetric` family
(:class:`~orpheus.numerics.metric.DiagonalMetric` /
:class:`~orpheus.numerics.metric.DenseMetric` /
:class:`~orpheus.numerics.metric.FactoredMetric`) — a space's inner
product as an object that is *applied* rather than an array that is
multiplied — is documented at :ref:`spaces-metric-object`. The theory
pages contain the full
mathematical narrative; per-symbol API docstrings live in the
modules themselves and are accessible via the standard
``orpheus.numerics`` import path.

Function spaces — trace and interior face cochains
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~orpheus.numerics.space.FunctionSpace` base class carries
the identity of every typed field's space, and since the identity flip
(structural ``__eq__``, CS4c step 6, 2026-09-07) that identity is **per
class**: an axis-built space compares and hashes by its ``axes`` tuple,
an axes-less one by ``(name, shape)``, and one of each is never the same
space (:ref:`spaces-identity-bridge`).
⛔ This paragraph said *"carries the (name, shape, inner_product_weights)
identity"* until 2026-09-07 — a third spelling that was already wrong
before the flip: ``inner_product_weights`` is declared ``compare=False``
and has never entered ``==``.  On an axis-built space the measure DOES
reach identity, but through the axes, which is the doctrine *metric
differences imply space differences*. Two **face** spaces specialise it
for the SN sweep's
codim-1 (face) quantities, both re-homing their
:class:`~orpheus.numerics.face_layout.FaceLayout` onto the space as
``compare=False`` leaf-data (the A.5 "layout-on-space" pattern):

* :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` — the
  **boundary** trace cochain :math:`C^1_\partial`, carrying
  ``omega_dot_n`` for the inflow / outflow directional partition. The
  space of :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`.
* the **interior** face cochain :math:`C^1_{\rm int}` — through S6.4
  the typed ``InteriorFaceSpace`` (the trace space *minus*
  ``omega_dot_n``; flux-only, axis-parametric) carrying
  ``WavefrontFlux``; RETIRED at S6.4(f) when the walk re-layering
  dissolved the type's boundary algebra into the shared octant frame.
  The cochain concept's live realizations are the rolling front
  (``_MovingFrontier``) and the per-octant full-cochain buffers
  (``FullFieldWavefront._octant_face_cochain``).

Together the boundary and interior cochains biproduct-decompose the
full face cochain
:math:`C^1 = C^1_{\rm int} \oplus C^1_\partial` (Issue #205 Phase 5).
The full cochain frame, the :math:`\iota_*` / :math:`\iota^*` trace
operators, the flux-only-single-role rationale, and the succession
history are documented at :ref:`wavefront-flux-cochain` (theory page).
