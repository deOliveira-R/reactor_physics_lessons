.. _cone-field-algebra:

============================================
The Field Algebra: Flux in the Positive Cone
============================================

.. contents:: Contents
   :local:
   :depth: 2


.. Machine header — the ``nexus-meta`` schema for this page (PROVISIONAL).
.. Extracted from ``operator_algebra.rst`` (#231 Phase 3); rewritten as the
.. cone chapter at campaign-1 CS3 (2026-08-19). The schema is provisional
.. pending a full re-audit of the corpus.

.. dropdown:: Machine header — ``nexus-meta`` schema (PROVISIONAL)
   :color: muted

   .. code-block:: yaml

      module: transport
      concept: field_algebra
      role: "the SN flux field algebra — flux as an element of the positive cone K of an ordered vector space V, cone membership as an element predicate, cone preservation as a realization property, and the state / residual / source role grid"
      depends_on: [operator_algebra]
      related: [operator_adjoint, boundary_conditions]
      status: "rewritten at campaign-1 CS3 (the cone overturn); provisional header"


This page develops the **field algebra** of the S\ :sub:`N` transport
solve — the recognition that a flux is an element of the **positive
cone** :math:`K` of an ordered **vector** space :math:`V`, that cone
membership is a **predicate on elements** rather than an invariant of a
type, and that cone **preservation** is a property of the *realization*
that produced the field. It is the field-algebra companion to the
operator algebra developed in
:doc:`/theory/foundations/operator_algebra`: that page types the
**operators** — the within-group loss composite
:math:`A = L + C - S - N_{2n} - B`
and its invertible sub-composite :math:`L + C`, whose inverse
:math:`(L+C)^{-1}` is the transport :term:`sweep` — while this page types
the **fields** those operators act on.

.. attention::

   **This page was rewritten on 2026-08-19.** From 2026-06-08 (Issue
   #208) until that date it taught the opposite ontology: flux states as
   an **affine space** :math:`\mathbb{A}` over a difference space
   :math:`V`, with ``flux + flux`` a :class:`TypeError` and a separate
   ``Displacement`` type for the iterate increment. That design was
   **overturned by user ruling** on correctness grounds. The argument
   that overturned it, and the argument that *motivated* it, are both
   preserved below — see :ref:`cone-overturn-adjudication` for the six
   reasons and :ref:`cone-the-overturned-affine-design` for the retired
   design kept as dated history.

.. note::

   **Three symbols that look alike are kept distinct throughout this
   page.**

   - :math:`V` (italic) is the **flux vector space** — the ordered
     vector space every flux leaf realizes. It is *not* the cell volume
     :math:`V_{\rm cell}` that appears in the space metric elsewhere in
     the corpus; where both are in play this page writes
     :math:`V_{\rm cell}` in full.
   - :math:`K \subset V` (upright roman) is the **positive cone**. It is
     *not* the collision-probability kernel matrix :math:`\mathbf{K}` of
     :doc:`/theory/methods/collision_probability`, and it is not the
     multiplication factor :math:`k`.
   - :math:`A = L + C - S - N_{2n} - B` (italic) is the **within-group
     loss operator** (the two collision gains :math:`S` and
     :math:`N_{2n}` are two instances of one binding;
     :ref:`the two collision gains <operator-algebra-two-gains>`). The **sweep** is the inverse of its invertible
     sub-composite, :math:`(L+C)^{-1}` — the *inner kernel* of the full
     within-group solve :math:`A^{-1}`, never :math:`A^{-1}` itself.

.. warning::

   **"Affine" survives elsewhere in the corpus and means something
   different there — do not read this page's overturn onto it.** The
   boundary law :eq:`affine-bc-form`,
   :math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q`, is an **affine
   map** (a linear operator plus a constant), which is a statement about
   an *arrow*. What was overturned here is an **affine space** (a set of
   points with no origin), which is a statement about an *object*. The
   two are independent: an affine map from :math:`V` to :math:`V` is
   perfectly ordinary linear algebra, and
   :doc:`/theory/foundations/boundary_conditions` is untouched by this
   page's ruling. The same word, two unrelated senses.


.. _cone-typed-field-algebra:

The cone-typed SN field algebra — state, residual, source
=========================================================

Campaign-1 phase **CS3** (2026-08-19, plan
``.claude/plans/space_and_kernel_binding_campaign.md`` §4; the ruling is
decision **D1** of ``.claude/plans/orpheus-operator-machinery-report-v2.md``
Part VI.5) replaced the affine field algebra with the cone ontology. The
user's criterion was stated once and governs the whole carve: *"What
matters is correctness. We should not be bound by past mistakes."*

.. admonition:: Key Facts (the cone triad)
   :class: tip

   - **Flux lives in the positive cone** :math:`K` **of an ordered
     vector space** :math:`V`. :math:`V` is a genuine vector space: it
     has an origin (vacuum), it is closed under :math:`+` and under
     scalar multiplication of either sign, and every flux leaf realizes
     it through the inherited
     :class:`~orpheus.numerics.field.Field` dunders
     (:eq:`flux-vector-algebra`).
   - **Cone membership is a PREDICATE on an element, never a constructor
     invariant.** :meth:`Field.cone_violations
     <orpheus.numerics.field.Field.cone_violations>` returns the offending
     index tuples, most-negative first; emptiness *is* membership. A
     :math:`\psi \ge 0` type would **refuse legitimate production
     output**: `[M]` a converged
     :func:`~orpheus.sn.solver.solve_sn_fixed_source` ships
     :math:`\min\psi = -6.399383\times10^{-1}`
     (:ref:`cone-membership-is-a-predicate`). On an axis-built space the
     predicate first asks the SPACE whether the sign test is meaningful
     at all, and REFUSES on a modal factor
     (:ref:`spaces-nodal-modal`).
   - **Cone preservation is a property of the REALIZATION**, carried by
     the ``is_positivity_preserving`` class attribute of
     :class:`~orpheus.transport.spatial.scheme.DiscretizationScheme`.
     Both shipped schemes declare it **honestly** ``False``
     (:class:`~orpheus.transport.spatial.diamond.DiamondDifference`,
     :class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`).
   - **A difference of fluxes is the SAME type, signed.**
     :math:`\psi_2 - \psi_1 \in V` leaves :math:`K` but stays in
     :math:`V` — it is a perturbation / error / iterate-increment field,
     the same tensor type without the positivity predicate. There is no
     separate displacement type.
   - **The fiber discipline survives, and it was never the torsor's.**
     "Fluxes of different problems don't mix" is enforced by the
     retained :meth:`Field._check_partner
     <orpheus.numerics.field.Field._check_partner>` chain — **class
     identity** (which *is* units identity, one ``UNITS`` constant per
     leaf) + **space CONTENT equality**. ⚠ At CS3 the chain's third
     tier was mesh-object identity; campaign 1 CS4b (S3) retired it —
     **the fiber IS the space now**
     (:ref:`cone-fiber-discipline`).
   - **The iterate diagnostics live on the ITERATION layer.** The
     contraction factor :math:`\rho` (:eq:`iterate-contraction-ratio`)
     and the :math:`c \to 1` true-error estimate
     (:eq:`iterate-true-error`) derive from
     :attr:`IterationRecord.increment_norms
     <orpheus.numerics.convergence.IterationRecord.increment_norms>` —
     the single recorded trajectory. A *state* cannot carry them; that
     observation was right, and its home was the iteration, not a field
     type.
   - **The typed residual is unchanged by the overturn.** It was always
     a plain vector role — see :ref:`affine-typed-residual` (the
     ``affine-`` in that anchor is a historical artefact of this page's
     former title, not a claim; the note there says so).


The two dimensional universes — flux and rate-density
-----------------------------------------------------

Every typed SN field is a *quantity* with a *dimension* (the View-G
decision, Issues #205 / #207: units live on the field, not the
:class:`~orpheus.numerics.space.FunctionSpace`; see
:class:`~orpheus.numerics.field.Field`). The transport solve moves
between exactly two dimensional universes, connected by the loss
operator :math:`A = L + C - S - N_{2n} - B`:

.. list-table:: The two universes connected by :math:`A` / :math:`A^{-1}`
   :header-rows: 1
   :widths: 22 26 26 26

   * - Universe
     - Units
     - State role
     - Defect / source role
   * - **Flux**
     - :math:`1/(\mathrm{cm^2 \cdot s \cdot sr})`
     - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
       :math:`\psi` — and its own differences
       :math:`\Delta\psi = \psi_2 - \psi_1`, the *same* type, signed
     - —
   * - **Rate-density**
     - :math:`1/(\mathrm{cm^3 \cdot s \cdot sr})`
     - —
     - :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`
       :math:`q`, :math:`A\psi`; and
       :class:`~orpheus.transport.residuals.angular_residual.AngularResidual`
       :math:`r`

The map between them is dimensional: applying :math:`A` (the full
within-group loss — streaming :math:`\hat\Omega\cdot\nabla` + collision
:math:`\Sigma_t` − scattering − boundary, each carrying a
:math:`1/\mathrm{cm}` factor) sends a flux to a rate-density (``A.apply`` →
:class:`AngularSourceSink`, documented at
:ref:`bc-extraction-operator-output-typing`); inverting it
(``A.solve`` :math:`= A^{-1}`, the full within-group solve) sends a
rate-density back to a flux (``.solve`` → :class:`AngularFlux`). The
invertible sub-composite :math:`L + C` has the sweep :math:`(L+C)^{-1}`
as its inverse — the inner kernel that ``A.solve`` drives to its fixed
point, never :math:`A^{-1}` itself. The source side (the B.5.2
operator-output typing) and the residual side are BOTH rate-density;
states and their differences are BOTH flux. This is the load-bearing
distinction the role grid encodes: *same universe* grants permission to
add in linear algebra, but the *class* still gates meaning (a residual
and a source share rate-density units yet are different classes — see
:class:`~orpheus.numerics.field.Field`).

.. note::

   **What the overturn changed in this table is a deletion, not a
   retyping.** Until 2026-08-19 the Flux column named a second type —
   ``AngularDisplacement``, the tangent — and the grid had a fourth
   column for it. The difference of two fluxes is now the flux type
   itself, so the column has nothing left to hold and the type family
   retired (:ref:`cone-role-grid`).


.. _cone-ordered-vector-space:

The ordered vector space and its positive cone
==============================================

Let :math:`V` be the space of discrete flux fields of one leaf shape on
one mesh — for the per-ordinate bulk leaf,
:math:`V \cong \mathbb{R}^{N \times n_g \times n_{\rm spatial}}` carrying
the space-induced inner product
:math:`\langle \psi, \varphi\rangle_V` of the leaf's
:class:`~orpheus.numerics.space.FunctionSpace`. :math:`V` is a vector
space over :math:`\mathbb{R}`, with no qualification: the flux algebra
is exactly the inherited
:class:`~orpheus.numerics.field.Field` algebra,

.. math::
   :label: flux-vector-algebra

   \psi_1 + \psi_2 &\;\to\; \psi \in V
       \qquad&&\text{(superposition — legal, and it is physics)}\\
   \psi_2 - \psi_1 &\;\to\; \Delta\psi \in V
       \qquad&&\text{(the SAME leaf type, signed)}\\
   \lambda\,\psi &\;\to\; \psi \in V,\quad \lambda\in\mathbb{R}
       \qquad&&\text{(scalar action, either sign; } \psi/k
       \text{ is the } \lambda = 1/k \text{ case)}\\
   0 &\;\in\; V
       \qquad&&\text{(vacuum — the additive identity, not a chosen base point)}

.. vv-status: flux-vector-algebra documented

.. Rationale: this is the STRUCTURAL typing identity of a flux leaf, not a
.. solver claim — there is no eigenvalue, flux shape, or convergence order
.. in it. Its verifiable content is pinned bit-exactly by the foundation
.. battery tests/numerics/test_flux_vector_algebra.py across the four
.. parameterized leaves (legality + exactness of the sum, commutativity,
.. zero-is-the-identity with the copy contract, the signed difference, the
.. round-trip and telescoping to 8 ULP, and scalar algebra), plus the
.. per-leaf modules named in that battery's docstring for the other three
.. flux leaves. Those are `foundation` gates by doctrine, so they carry no
.. `verifies(...)` marker and this label stays `documented`.

The **positive cone** is the subset on which the physical interpretation
lives:

.. math::
   :label: positive-cone-definition

   K \;=\; \bigl\{\, \psi \in V \;:\; \psi_i \ge 0
   \ \ \text{for every index } i \,\bigr\}

.. vv-status: positive-cone-definition documented

.. Rationale: a definitional set identity — the coordinate cone of the
.. nodal axes — with no single implementing function to point an
.. `implements` edge at. Its verifiable content is the closure algebra and
.. the predicate that decides membership: the foundation unit legs of
.. tests/sn/solve/test_cone_membership_witness.py's siblings (apex, closure
.. under + and under nonnegative scaling, NON-closure under difference and
.. under negative scaling, the exact-index report, and the IEEE edges) plus
.. the two production witness rows in that module. `foundation` gates, so no
.. `verifies(...)` marker and the label stays `documented`.

:math:`K` is a **convex cone with apex at the origin**: it contains
:math:`0`, it is closed under addition
(:math:`\psi, \varphi \in K \Rightarrow \psi + \varphi \in K`) and under
scaling by :math:`\lambda \ge 0`, and it is **pointed** —
:math:`K \cap (-K) = \{0\}`, so it induces a genuine partial order
:math:`\psi \le \varphi \iff \varphi - \psi \in K`. That order is what
makes :math:`(V, K)` an *ordered* vector space, and it is what the
transport theorems need.

Three things :math:`K` is **not** closed under, and each one matters:

- **Differences.** :math:`\psi_2 - \psi_1` is generically outside
  :math:`K` even when both operands are inside it. Iterate increments,
  perturbation fields, MMS error fields and Krylov directions all live
  here, in :math:`V \setminus K`.
- **Negative scaling.** :math:`-\psi \notin K` for any
  :math:`\psi \ne 0` in :math:`K`, by pointedness.
- **The shipped spatial realizations.** Neither
  :class:`~orpheus.transport.spatial.diamond.DiamondDifference` nor
  :class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`
  maps :math:`K \to K` (:ref:`cone-preservation-is-a-realization-property`).

Why the cone is load-bearing rather than decorative
----------------------------------------------------

:math:`K` is not a book-keeping nicety attached to a vector space — it
is the object the fundamental theorems of the eigenvalue problem are
stated over.

- **Krein–Rutman.** Existence, uniqueness, simplicity and strict
  positivity of the fundamental mode of a positive compact operator are
  cone theorems, not vector-space theorems. The statement *"the*
  :math:`k`-*eigenfunction is positive and simple"* has no formulation
  without :math:`K`.
- **Birkhoff–Hopf.** Power-iteration convergence is a **contraction in
  the Hilbert projective metric on the cone**,

  .. math::

     d_H(\psi, \varphi) \;=\;
     \log\Bigl[\bigl(\sup_i \psi_i/\varphi_i\bigr)
                \bigl(\sup_i \varphi_i/\psi_i\bigr)\Bigr],

  and the dominance-ratio bound :math:`d \le \tanh(\Delta/4)` is a
  function of the projective **diameter** :math:`\Delta` of
  :math:`F(K)`. Neither quantity is definable in :math:`V` alone.
- **The eigenfunction is a RAY, not a point.**
  :math:`\psi` and :math:`\lambda\psi` (:math:`\lambda > 0`) are the
  same physical mode; the fundamental mode is a point of the projective
  cone :math:`P(K)`, and its magnitude is fixed only by a normalization
  convention. This already ships:
  :func:`~orpheus.numerics.eigenvalue.power_iteration` rescales every
  iterate to unit production rate
  (:math:`\int \nu\Sigma_f\,\phi\,\mathrm{d}V = 1`) by the *scalar*
  division ``flux_distribution / p`` — which is a :math:`V` operation
  that an affine space does not admit at all (see argument **3** of
  :ref:`cone-overturn-adjudication`).

.. note::

   **Cone representability is a property of the AXIS, not of the
   field.** A *nodal* axis — the per-ordinate discrete-ordinates axis,
   the spatial cell axis, the group axis — carries a coordinate cone:
   pointwise nonnegativity of the coefficients *is* nonnegativity of the
   function. A *modal* axis does not: for a harmonic-moment field,
   coefficient positivity is neither necessary nor sufficient for the
   reconstructed function to be positive. This dichotomy is the
   ray-effect / negative-source dichotomy seen from the algebra side —
   positivity is native to the quadrature axis, rotational equivariance
   is native to the harmonic axis, and no angular basis has both. It is
   why :meth:`Field.cone_violations
   <orpheus.numerics.field.Field.cone_violations>` is a statement about
   ``values`` in *this leaf's* layout, and why a violation reported on
   :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
   is a statement about its coefficients rather than about the angular
   function they synthesize. Contraction against a positive weight never
   degrades cone representability, so the :math:`\ell = 0` retract of
   moment space regains the coordinate cone exactly — which is why
   :meth:`AngularFlux.integrate_angular
   <orpheus.transport.fields.angular_flux.AngularFlux.integrate_angular>`
   carries a violation of :math:`\psi` into a violation of :math:`\phi`
   and the witness gate asserts precisely that.

   **Since campaign-1 CS1 (2026-08-20) the dichotomy is
   machine-readable**, not only prose: an axis declares its
   :class:`~orpheus.numerics.axis.BasisKind` (``NODAL`` / ``MODAL``) at
   construction, a space answers
   :attr:`has_coordinate_cone
   <orpheus.numerics.space.FunctionSpace.has_coordinate_cone>` from its
   factors, and :meth:`Field.cone_violations
   <orpheus.numerics.field.Field.cone_violations>` consults that answer
   before doing any arithmetic — see :ref:`spaces-nodal-modal` for the
   space-layer statement. ⛔ **The harmonic sentence above still holds;
   the paragraph that used to follow it does NOT.** Until 2026-09-08 this
   read *"the property is three-valued, and no harmonic-moment space in
   the tree is axis-built — `[M]` the only axis mint inside* ``orpheus/``
   *is* :attr:`MaterialMesh.bulk_space
   <orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>` *— so
   every one of them takes the* ``None`` *arm … The refusal arm becomes
   production-reachable when CS2 mints the harmonic axis."* CS4c step 6
   item 6.2c-ii made both moment heads axis-built
   (:ref:`spaces-moment-head-axis-built`), so the moment space now takes
   the ``False`` arm and :meth:`Field.cone_violations
   <orpheus.numerics.field.Field.cone_violations>` REFUSES on it instead
   of answering about coefficients — which is the stronger behaviour the
   dichotomy always argued for, arriving from a different phase than
   predicted. `[M]` 2026-09-08: ``has_coordinate_cone`` is ``False`` on
   all 33 shipped (rule, :math:`L`) frames' moment heads, where it read
   ``None`` before; and an AST census of ``orpheus/`` finds **11** axis
   mints (7 ``of_axes`` + 4 ``for_basis``), not one.


.. _cone-overturn-adjudication:

Why the affine ontology was overturned — the six arguments
==========================================================

The affine design was not a slip; it was argued for by three
independent expert frames and it shipped enforced arithmetic across 16
production modules for ten weeks. Overturning it therefore needs an
argument, not a preference. Six were adjudicated on 2026-08-19, in
dependency order; each is sufficient on its own for the clause it
supports, and together they close every escape.

.. list-table:: The six-argument adjudication (2026-08-19, decision D1)
   :header-rows: 1
   :widths: 6 30 64

   * - #
     - The argument
     - Why it is decisive
   * - **1**
     - **Superposition is physics, and the transport operator is
       linear.**
     - For fixed cross sections the within-group operator
       :math:`A = L + C - S - N_{2n} - B` is linear in :math:`\psi`, so if
       :math:`A\psi_1 = q_1` and :math:`A\psi_2 = q_2` in the *same
       medium* then :math:`A(\psi_1 + \psi_2) = q_1 + q_2`. The sum of
       two fluxes from independent sources in one medium is the flux of
       the combined source — a *theorem*, and the affine gate made it
       unspellable. What genuinely does not add is fluxes of
       **different problems**, and that is a statement about the
       **fiber** (which mesh, which space, which leaf class), not about
       affine structure. Making a theorem of the model unrepresentable
       to enforce a bookkeeping rule is the wrong trade at any price.
   * - **2**
     - **Vacuum is a canonical zero.**
     - The affine argument was *"flux states have no natural zero — the
       zero flux is a chosen base point, not an identity."* That is
       false as physics: the absence of neutrons is not a convention.
       :math:`\psi \equiv 0` is the unique solution of the homogeneous
       problem with no source and no inflow, it is the value every
       boundary operator returns at zero input
       (:math:`B(0) = 0` — a shipped gate), and it is the cold start of
       every iteration. A space with a distinguished, physically
       meaningful, operation-preserved zero **is a vector space**; that
       is what "having an origin" means.
   * - **3**
     - **The shipped code had already conceded** :math:`V`.
     - `[M]` under the affine doctrine, scalar scaling was explicitly
       kept legal ("``__mul__`` / ``__truediv__`` / ``__neg__`` are
       inherited, so eigenvalue normalisation :math:`\psi/k` survives");
       zero fluxes were constructed freely by ``zeros_on`` /
       ``from_mesh(0)`` for initial iterates and gates; and
       :func:`~orpheus.numerics.eigenvalue.power_iteration` divided
       iterates by a scalar every outer. **A literal affine space admits
       no scalar action at all** — :math:`\lambda \cdot p` is undefined
       on a point. So the shipped object was never an affine space: it
       was precisely *a vector space with binary same-class* ``+``
       *disabled and* ``-`` *retyped*. The doctrine described one object
       and the code implemented another, and the code's object was the
       one with the theorems.
   * - **4**
     - :class:`~orpheus.numerics.operator.LinearOperator` **requires**
       :math:`V`, **so Issue #331 dissolves.**
     - #331 recorded that the three leaves of the *one* sum
       :math:`A = (L+C) - S - B` disagreed about their own domain:
       ``L.apply(displacement)`` worked, ``S.apply`` and ``B.apply``
       raised :class:`TypeError`, and therefore
       ``build_within_group_system(...).loss`` refused too — while a
       Krylov method iterates on :math:`V` by construction and had to
       drop to ``to_flat`` to do it. Under the affine ontology the
       repair was to give **every** operator a second, parallel tangent
       map :math:`B_V`, doubling every domain contract in the algebra.
       In :math:`V` the disagreement is **unspellable**:
       ``S.apply(Δψ)`` type-checks because there is only one type. The
       issue closed with the algebra flip (``993fa280``), not with a
       tangent-map layer.
   * - **5**
     - **Diamond difference does not preserve** :math:`K`, **so
       membership must be a predicate.**
     - This is the argument that decides the *form* of the cone, not
       just its existence, and it is the one measured against production
       output. If cone membership were a constructor invariant, the type
       would refuse a converged solve that the solver legitimately
       produces. See :ref:`cone-membership-is-a-predicate` for the
       witness and the illegal-states-unrepresentable boundary it
       establishes.
   * - **6**
     - **The one genuinely affine object in the tree is handled by
       gauge-fixing IN** :math:`V`.
     - There *is* real affine structure in this codebase, and finding
       where it lives is what settles the question. On any
       :math:`d \ge 2` Cartesian diamond-difference mesh with
       :math:`\ge 2` reflective axis pairs the loss operator is exactly
       singular, so :math:`A\psi = q` has a solution **manifold**
       :math:`\psi_0 + \ker A` — a genuine affine subspace of
       :math:`V`, over the genuine vector space :math:`\ker A`
       (:ref:`sn-loss-kernel-gauge`, Issue #344). The tree resolves it
       by **gauge-fixing**: projecting onto the canonical
       minimum-:math:`\lVert\cdot\rVert_G` member, an operation defined
       entirely in :math:`V`. So the affine structure that really exists
       is a property of a **solution set**, resolved by a projection —
       not a property of the **state space**, and not something a field
       *type* could have expressed. The affine instinct was pointing at
       a real object, one layer away.

.. important::

   **Read arguments 1–4 together as one shape.** Each of them says the
   affine typing was mis-*layered* rather than mis-*motivated*. The
   three concerns the torsor was carrying are all real, and all three
   have a correct home:

   - **iterate hygiene** ("an increment is not a state; it knows
     *previous*") belongs to the **iteration**, and now lives on
     :class:`~orpheus.numerics.convergence.IterationRecord`
     (:ref:`cone-iterate-diagnostics`);
   - **fiber discipline** ("fluxes of different problems don't mix")
     belongs to the **partner check**, and always did — class + space,
     untouched by the carve; the mesh-object tier it also carried at
     CS3 was retired one campaign later
     (:ref:`cone-fiber-discipline`);
   - **positivity** ("a flux is nonnegative") belongs to the
     **element**, as a predicate, and to the **realization**, as a flag
     (:ref:`cone-membership-is-a-predicate`).

   None of the three needed an affine space. Distributing them to the
   layers that own them is what the carve did.


.. _cone-membership-is-a-predicate:

Membership is a predicate — and why illegal-states-unrepresentable fails here
=============================================================================

The project's standing preference is to make illegal states
unrepresentable (Cardinal Rule 2; the ``coding-elegance``
"illegal-states-unrepresentable" pattern). A :math:`\psi \ge 0`
constructor invariant is the obvious application. **It is wrong here**,
and the reason is worth stating as a rule because the failure mode
generalizes.

.. admonition:: The rule the DD witness establishes
   :class: important

   Make a state unrepresentable **iff both** halves hold:

   1. every value the type would admit is legal, **and**
   2. every legal value is admitted.

   Half 2 is the one that is skipped, because it is a claim about the
   *producers* rather than about the concept. When a production path
   legitimately emits a value the invariant would reject, a constructor
   invariant does not prevent a bug — **it refuses correct output**, and
   the pressure that follows is to weaken the invariant, silence it, or
   route around the type. All three are worse than not having it.

For flux, half 2 fails twice over and independently:

- **Algebraically.** :math:`K` is not closed under difference or
  negative scaling, and both operations are load-bearing: an iterate
  increment, an MMS error field, a perturbation and a Krylov direction
  are all outside :math:`K` by construction. A :math:`\psi \ge 0` type
  could not host its own subtraction — which is exactly why the affine
  design needed a *second* type in the first place.
- **Numerically.** The shipped spatial realizations do not map
  :math:`K \to K`, so a converged, correct, physically-interpreted
  production solve can carry negative entries.

The DD witness — measured production output outside the cone
-------------------------------------------------------------

The second half is the one that has to be *measured* rather than
argued, so it is. The gate is
``tests/sn/solve/test_cone_membership_witness.py``, and the numbers
below were re-derived for this page through the public entry
:func:`~orpheus.sn.solver.solve_sn_fixed_source`.

The fixture is a one-group homogeneous pure-slab problem:
:math:`\Sigma_t = 10`, :math:`c = \Sigma_s/\Sigma_t = 0.5`, vacuum on
both faces, :math:`S_2` Gauss–Legendre, and a per-ordinate source of
:math:`100` in the first cell only (the asymmetry that drives the
diamond dome). Both rows converge, and both report ``converged``.

.. list-table:: The witness pair — the cone violation is the DISCRETIZATION
   :header-rows: 1
   :widths: 12 12 16 20 20 20

   * - :math:`n_x`
     - width
     - :math:`\Delta x \cdot \Sigma_t`
     - :math:`\min \psi`
     - negative entries
     - :math:`\min \phi`
   * - 2
     - 20
     - 100
     - :math:`+2.181405\times 10^{-1}`
     - 0 of 4
     - :math:`+8.826374\times 10^{-1}`
   * - 4
     - 40
     - 100
     - :math:`-6.399383\times 10^{-1}`
     - **2 of 8**
     - :math:`-8.438399\times 10^{-1}`

The pair holds the **materials, the quadrature, the boundary
conditions, the source magnitude and the optical cell size** all fixed —
:math:`\Delta x \cdot \Sigma_t = 100` in *both* rows — and moves only
the cell count, hence the slab thickness. No material or angular
explanation survives that. What changes is how far the diamond closure's
sign-alternating overshoot propagates: with two cells the excursion
never reaches a cell centre negative; from three cells on it does, and
the scalar flux inherits the violation through the angular integral.

.. list-table:: Cell-count scan at fixed :math:`\Delta x \cdot \Sigma_t = 100`
   :header-rows: 1
   :widths: 20 40 40

   * - :math:`n_x`
     - :math:`\min \psi`
     - negative entries
   * - 2
     - :math:`+2.181405\times 10^{-1}`
     - 0 of 4
   * - 3
     - :math:`-6.420666\times 10^{-1}`
     - 2 of 6
   * - 4
     - :math:`-6.399383\times 10^{-1}`
     - 2 of 8
   * - 5
     - :math:`-6.379506\times 10^{-1}`
     - 4 of 10
   * - 6
     - :math:`-6.360939\times 10^{-1}`
     - 4 of 12

The mechanism is the classical one, and refining the cell exposes it
rather than hiding it, because the onset is governed by the optical
cell size and not by the mesh count:

.. list-table:: Cell-size scan at :math:`n_x = 4` — the classical DD positivity limit
   :header-rows: 1
   :widths: 25 25 25 25

   * - :math:`\Delta x`
     - :math:`\Delta x \cdot \Sigma_t`
     - :math:`\min \psi`
     - negative entries
   * - 0.1
     - 1
     - :math:`+5.807774\times 10^{-2}`
     - 0 of 8
   * - 0.2
     - 2
     - :math:`-8.682834\times 10^{-1}`
     - 2 of 8
   * - 0.5
     - 5
     - :math:`-3.510366\times 10^{0}`
     - 2 of 8
   * - 1.0
     - 10
     - :math:`-3.372516\times 10^{0}`
     - 2 of 8
   * - 2.0
     - 20
     - :math:`-2.378338\times 10^{0}`
     - 2 of 8
   * - 5.0
     - 50
     - :math:`-1.185141\times 10^{0}`
     - 2 of 8
   * - 10.0
     - 100
     - :math:`-6.399383\times 10^{-1}`
     - 2 of 8

The transition sits exactly where the textbook puts it. The
weighted-diamond closure with weight :math:`\tfrac12` reads
:math:`\psi_{\rm out} = 2\bar\psi - \psi_{\rm in}`, so a cell whose
optical thickness exceeds :math:`\approx 2` can drive
:math:`\psi_{\rm out} < 0` from strictly positive ``source`` and
``upstream_state`` (:cite:`LewisMiller1984` §5.3 exhibits the canonical
counter-example, and the
:attr:`DiamondDifference.is_positivity_preserving
<orpheus.transport.spatial.diamond.DiamondDifference.is_positivity_preserving>`
docstring cites it). `[M]` :math:`\Delta x \cdot \Sigma_t = 1` is in
:math:`K`; :math:`\Delta x \cdot \Sigma_t = 2` already is not.

.. warning::

   **A green cone reading claims nothing about enforcement.** The
   predicate *observes*. Production does **not** keep fields in
   :math:`K`: there is no clipping, no fixup, no warning on the SN
   spatial path; a violating solve is not refused; DD is not repaired.
   The same disclaimer is carried verbatim in the
   :meth:`Field.cone_violations
   <orpheus.numerics.field.Field.cone_violations>` docstring, deliberately, so
   that no audit reading either surface alone can mistake the predicate
   for a guarantee.

The predicate's shape
----------------------

:meth:`Field.cone_violations
<orpheus.numerics.field.Field.cone_violations>` returns the **index tuples**
of the offending entries, most-negative first, into this leaf's own
``values`` layout — not a :class:`bool`. Returning the structure rather
than a verdict is the ``vv-principles`` anti-pattern #14 discipline
(*"prefer returning the structure to returning a* ``bool`` *about it,
because a returned structure makes its own correctness assertable"*): a
:class:`bool` cannot say **where**, so nothing can check that the
predicate found the right entries. The witness gate asserts set equality
against ``np.argwhere(values < 0)`` and that the worst violation is
reported first — assertions a boolean return could not carry.

Emptiness *is* membership, and the emptiness answer is exact under any
``k`` cap. Two IEEE edges are pinned rather than left to accident:
:math:`-0.0` **is** a member (``-0.0 >= 0.0`` is ``True``), and
``nan`` **is** a violation (an unordered entry belongs to no cone),
while :math:`+\infty` is admitted. The predicate is spelled
``not (value >= 0.0)`` precisely so ``nan`` falls on the violation side
without a separate branch.

.. _cone-predicate-basis-kind-consult:

The basis-kind consult — the predicate's first question is asked of the SPACE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before any arithmetic, the predicate asks whether a per-component sign
test is **meaningful on this space at all** — because it is meaningful
only on a *coordinate* cone, and whether the space has one is the
space's own structural answer, not the field's (campaign-1 CS1 step 4,
2026-08-20). The consult reads
:attr:`FunctionSpace.has_coordinate_cone
<orpheus.numerics.space.FunctionSpace.has_coordinate_cone>`, which is
three-valued, and each value has a different obligation:

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Answer
     - When
     - What ``cone_violations`` does
   * - ``False``
     - axis-built, ANY factor ``MODAL``
     - **REFUSES**, with a typed error that names the space and states
       the reason. Components are expansion COEFFICIENTS; a positive
       function may have negative coefficients, so answering would
       *manufacture violations out of a basis choice*. Refusal is the
       only honest answer — the question is malformed, not merely hard.
   * - ``True``
     - axis-built, ALL factors ``NODAL``
     - **Answers**, with exactly the arithmetic described above. The
       consult gates the modal case only; it does not change the
       predicate.
   * - ``None``
     - legacy space (``axes is None``)
     - **Pre-CS1 behavior, unchanged** — deliberately. The question
       cannot be answered structurally, and collapsing ``None`` into
       ``False`` would fire the refusal on every legacy space in the
       tree, which is most of them.

This is the element-level face of a space-layer distinction; the
space-layer statement, and why ``None`` is a third value rather than a
defaulted ``False``, is :ref:`spaces-nodal-modal`.

✅ **The refusal arm HAS a production witness, since 2026-09-08.** The
paragraph here read, until then: *"the only axis-built space minted
inside* ``orpheus/`` *today is* :attr:`MaterialMesh.bulk_space
<orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>`\ *, whose
factors are both* ``NODAL``\ *, and no harmonic-moment space is
axis-built — so production reaches only the* ``True`` *and* ``None``
*arms. … It becomes production-reachable when CS2 mints the harmonic
axis."* CS4c step 6 item 6.2c-ii minted it instead, for the metric's sake
rather than CS2's: both moment heads carry ONE ``MODAL`` head axis
(:class:`~orpheus.numerics.axis.HarmonicAxis` /
:class:`~orpheus.numerics.axis.LegendreAxis`), so every moment space
answers ``False`` and every moment field's ``cone_violations`` is
refused. `[M]` 2026-09-08, on the 33 shipped (rule, :math:`L`) frames:
``has_coordinate_cone`` is ``False`` on 33 of 33 (previously ``None`` on
33 of 33). ⚠ The ``None`` arm has NOT retired — it is still the honest
answer for every legacy space, and for a WIDENED moment product, whose
axes-less ``SpatialMomentSpace`` tail keeps the whole product axes-less
until item 6.2c-iii lands.

The ``False`` arm's test-constructed gate stays, paired with its positive
leg (the same values on an all-nodal space answering exactly what the
legacy path answers) as ``vv-principles`` anti-pattern #11 requires of
any contract-validation method; what it gained is a production input it
can be run against.


.. _cone-preservation-is-a-realization-property:

Cone preservation is a property of the realization
==================================================

A sweep is a map :math:`V \to V` that *approximates* a map
:math:`K \to K`. Whether the approximation keeps the cone is a fact
about the **discretization**, not about the continuous operator and not
about the field type — so it is recorded where the discretization lives,
as the class attribute
``is_positivity_preserving`` on
:class:`~orpheus.transport.spatial.scheme.DiscretizationScheme`.

.. list-table:: The three tiers of guarantee, and what falsifies each
   :header-rows: 1
   :widths: 18 30 26 26

   * - Tier
     - What it guarantees
     - Carrier
     - Falsified by
   * - **Type**
     - units, representation shape, and the fiber (which problem this
       field belongs to)
     - class identity + :meth:`Field._check_partner
       <orpheus.numerics.field.Field._check_partner>`, whose second
       tier is **space CONTENT equality** (the CS4b S3 re-key; the
       :class:`~orpheus.transport.fields._bases.BulkField` mesh
       override it replaced is retired)
     - a cross-class binary operation that succeeds, or a binary
       operation between fields whose spaces differ in content
   * - **Realization**
     - whether nonnegative inputs give nonnegative outputs
     - ``DiscretizationScheme.is_positivity_preserving``
     - a scheme declaring ``True`` whose sweep leaves :math:`K`
   * - **Element**
     - whether *this* field is in :math:`K` right now
     - :meth:`Field.cone_violations
       <orpheus.numerics.field.Field.cone_violations>`
     - a converged solve outside :math:`K` that the predicate calls
       clean, or vice versa

`[M]` **both shipped schemes declare the flag** ``False``, and both are
honest:
:attr:`DiamondDifference.is_positivity_preserving
<orpheus.transport.spatial.diamond.DiamondDifference.is_positivity_preserving>`
(the witness above is its counter-example) and
:class:`~orpheus.transport.spatial.linear_discontinuous.LinearDiscontinuous`.

.. warning::

   **There is currently no shipped realization for which this flag is**
   ``True``, **so the flag's** ``True`` **branch has no production
   witness.** The registry holds exactly two schemes
   (``diamond_difference``, ``linear_discontinuous``), both ``False``;
   the only ``True`` values in the tree are on test doubles. Step and
   step-characteristic closures *are* positivity-preserving and are the
   flag's intended second occupant, but they are **not built** — the
   ``Step`` class that appears in the
   :class:`~orpheus.transport.spatial.scheme.DiscretizationScheme`
   docstring is a registration *example*, not a realization. A future
   gate written as *"the flag partitions the schemes"* would therefore
   be a gate with no witness on one side (``plan-authoring`` §6c) until
   such a scheme lands. State the flag's meaning; do not credit it with
   coverage it cannot yet have.


.. _cone-fiber-discipline:

The fiber discipline — what "fluxes of different problems don't mix" really was
===============================================================================

The affine gate's most persuasive motivation was that adding two fluxes
is *meaningless*. Half of that intuition is right, and the carve had to
show which half — because if the fiber discipline had been living inside
the torsor machinery, retiring the machinery would have dropped it.

`[M]` it was not. The guard chain that refuses an ill-formed partner is
:meth:`Field._check_partner
<orpheus.numerics.field.Field._check_partner>`, and it tests two things
in order:

1. **class identity** — ``type(self) is type(other)``. Because each role
   leaf carries its units as a class constant ``UNITS``, class identity
   *is* units identity; and because the leaf class also names the
   representation, it is representation identity too. Same units never
   grant meaning: an
   :class:`~orpheus.transport.fields.angular_flux.AngularFlux` and an
   :class:`~orpheus.transport.residuals.angular_residual.AngularResidual`
   do not share a class even where they share a shape.
2. **space CONTENT equality** — the two operands'
   :class:`~orpheus.numerics.space.FunctionSpace` must compare equal,
   and since campaign 1 that comparison is a statement about
   **content**: an axis-built space derives its name deterministically
   and injectively from its axes (shape, basis kind, measure bytes), and
   a trace space's name folds a digest of its layout, quadrature and
   face geometry (:ref:`spaces-identity-bridge`).

The base dunders route every same-class pair through this chain before
touching ``values``, so deleting the flux role mixin dropped ``+`` and
``-`` straight onto it.

.. note::

   **The chain had a THIRD tier at CS3, and campaign 1 retired it.**
   Until CS4b step S3 a
   :class:`~orpheus.transport.fields._bases.BulkField` override added
   **mesh-object identity**: two fields of the same class and the same
   nominal space, bound to different mesh objects, refused. That was
   the right guard for a *nominal* space identity — it stood in for a
   content check the spaces could not then perform. Once the carrier's
   cached spaces became axis-built and content-keyed, the override
   became a strictly stronger predicate than the property it was
   protecting, and it was retired on that ground (the **F2 doctrine**:
   operator and field admission compare space CONTENT, never
   provenance). Nothing loosened that should have stayed tight — the
   *fiber* was never the mesh object; it was always the geometry,
   the group structure and the quadrature, and all three are now axis
   content.

**What actually changed, measured.** Build two carriers from *equal*
edge arrays, and a third whose second cell edge moved:

.. list-table:: The fiber, after the S3 re-key (`[M]` 2026-08-24, 1-D slab, GL4, 2 groups)
   :header-rows: 1
   :widths: 34 22 22 22

   * - the partner differs by…
     - ``angular_bulk_space ==``
     - ``angular_trace ==``
     - ``psi_a + psi_b``
   * - nothing (a *twin* carrier — a distinct object built from equal
       arrays)
     - ``True``
     - ``True``
     - **ADDS**
   * - the boundary CONDITION only (vacuum vs reflective)
     - ``True``
     - ``True``
     - **ADDS**
   * - one moved cell edge
     - ``False``
     - ``True``
     - **REFUSES** (``ValueError``)
   * - the quadrature order
     - ``False``
     - ``False``
     - refuses
   * - the group count
     - ``False``
     - ``False``
     - refuses

Three readings, and each is a design decision made visible:

- **A twin carrier is the same fiber.** Two ``SNMesh`` objects built
  from equal inputs describe one problem, and their fields now mix.
  Under the old rule they refused — a false negative that forced
  callers to thread one carrier object through code that only needed
  one *geometry*.
- **A boundary LAW is not part of the fiber.** Changing vacuum to
  reflective changes neither the degrees of freedom nor the Gram, so
  the spaces are equal and the fields mix. Boundary laws are operator
  data (:doc:`/theory/foundations/boundary_conditions`), not field
  data — and that is the sharpest evidence that the retired tier was
  over-tight rather than merely redundant.
- **A moved edge is a different problem, and it still refuses** — on
  the space arm, because cell volumes are the spatial axis's measure
  and the measure is part of axis identity. The refusal that matters
  survived the re-key; only its *carrier* moved from provenance to
  content.

.. note::

   **CS5 (2026-08-29) put a distinct object INSIDE the twin row's
   spaces, and the row did not move — by design.** Since campaign-1
   phase CS5 the angular axis of
   :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space`
   carries its own :class:`~orpheus.numerics.quadrature.directional.Quadrature`
   as an :attr:`~orpheus.numerics.axis.Axis.generator`, and two twin
   carriers hold **different rule instances** (`[M]` ``a.quad is not
   b.quad``). Had provenance been admitted to the axis identity key,
   the top row would have flipped from ``True`` to — literally — a
   ``ValueError``, since the generator objects are un-``==``-able. The
   exclusion is what keeps the row true, and it is the F2 doctrine
   above enforced one layer down: *compare space CONTENT, never
   provenance*. `[M]` re-measured after CS5 on the same fixture, the
   twin row still reads ``True`` on both space columns with equal
   hashes, and the moved-edge row still reads ``False``. See
   :ref:`spaces-generator-identity-exclusion`.

⚠ Note the third column: the **trace** spaces compare equal even for
the moved-edge pair, because on a 1-D slab the face areas, the layout
and the quadrature are all unchanged — the boundary really is the same
boundary. Boundary-trace fields from two carriers with different
interiors are therefore contractible, which is correct and is what the
partial-current metric asserts.

The carve owed this a **negative control**, and it has one, re-derived
in place: ``test_fiber_guard_cross_mesh_refuses`` in
``tests/numerics/test_flux_vector_algebra.py`` now carries the
correctly-blind leg (twin carriers ADD, with an activation guard
asserting their spaces really do compare equal — so the leg cannot
silently become vacuous), the refusal leg (a stretched carrier reds on
``"equal space"``), and the same-fiber positive leg.
**Mutation-verified:** deleting the space arm of
``Field._check_partner`` reds the refusal leg while both positive legs
stay green. The row's CS3 docstring had *instructed* this re-derivation
— "if a later phase gives spaces mesh-dependent identity, re-derive
this row" — and the later phase did the opposite, giving spaces
**content** identity; either way the instruction fired, which is what a
discriminator guard is for.

.. note::

   **The retired mixin's own mesh check retired with it, correctly.**
   The affine machinery carried a second partner check whose job was the
   fiber test on a **cross-class** partner — the case Layer 1 refuses
   outright once both operands are the same type. With the cross-class
   partner gone from the flux algebra, that check had no case left to
   decide. Nothing was lost: the measured refusal above lands on the
   retained chain.


.. _cone-role-grid:

The role grid after the carve
=============================

The field vocabulary is a grid of three orthogonal axes — **locus**
{Bulk, Boundary} × **family** {Angular, Scalar, Moment} × **role**
{Flux, SourceSink, Residual} — realized as flat multiple-inheritance
leaves named ``<Family><Role>`` (bulk) or ``<Family>Boundary<Role>``
(boundary). The role axis lost its fourth column at CS3.

.. list-table:: The field-role grid (Issues #205 / #201 / #208, re-scoped at CS3)
   :header-rows: 1
   :widths: 22 26 26 26

   * - Block
     - Flux (state, and its differences)
     - Source/Sink
     - Residual (defect)
   * - **Angular**
     - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
     - :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`
     - :class:`~orpheus.transport.residuals.angular_residual.AngularResidual`
   * - **Scalar**
     - :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux`
     - ``ScalarSourceSink``
     - ``ScalarResidual``
   * - **Moment**
     - :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
     - ``HarmonicMomentSourceSink``
     - — *(principled hole — a residual is born only from a balance, and
       moment space is never the subject of one)*
   * - **Boundary**
     - :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
     - :class:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink`
     - :class:`~orpheus.transport.residuals.angular_boundary_residual.AngularBoundaryResidual`

.. note::

   **A fourth role, Displacement, existed from 2026-06-08 until
   2026-08-19.** ``transport/displacements/`` held eight modules — seven
   leaves plus a representation-keyed sibling registry that minted a
   displacement per flux leaf on ``-``. It retired whole at CS3 step 3
   (``5efd2178``): after the algebra flip it had zero production
   consumers, zero test imports, and — measured — an empty
   ``catches`` / ``verifies`` marker set, so the retirement carried no
   coverage away with it. The family tree in
   :mod:`orpheus.transport.fields._bases` records the same retirement
   note on the code side.

The role-axis asymmetry, re-derived
------------------------------------

Before the carve, the role axis was read as *"Flux and Displacement are
mixins because they add behaviour; Source and Residual are bare."*
`[M]` **that reading has inverted, and the inversion is worth stating
because it is easy to carry the old sentence forward.** As of CS3:

- :class:`~orpheus.transport.fields.angular_flux.AngularFlux` has **no
  role mixin at all** — its MRO is
  ``AngularField → BulkField → Field → ABC``, and it defines neither
  ``__add__`` nor ``__sub__``. The flux role's arithmetic *is* the
  inherited vector algebra.
- :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`
  **does** define its own ``__add__`` — the canonical
  subspace-containment injection that accepts a
  :class:`~orpheus.transport.source_sinks.scalar_source_sink.ScalarSourceSink`
  partner and broadcasts the isotropic operand across the
  :math:`\hat\Omega` axis before adding (the refined #207 exception;
  statically unspellable against Field's ``(T, T) -> T``, which is
  Issue #288).

So the axis that "changes the arithmetic interface" is now the *source*
axis, not the flux axis. The **conclusion** the old reading supported
survives untouched — Role must be a class, Representation must be a
class, and the flat multiple-inheritance leaf is the unique normal form
— but the *example* that carried it has moved. The full argument, with
its five obstructions re-derived under the cone, lives at
:ref:`carrier-grid-flat-leaf-normal-form`.

.. note:: The interior-face cochain :math:`C^1_{\rm int}`
   (:ref:`wavefront-flux-cochain`) is deliberately **not** a role leaf —
   it is a sweep-internal cochain, not an iterate state. After the
   carrier retired at S6.4(f) it is plain numpy arrays
   (``_MovingFrontier`` / ``_octant_face_cochain``), so the "not a role
   leaf" conclusion holds *a fortiori*: there is no field type at all to
   mis-place in the grid.


.. _cone-iterate-diagnostics:

The iterate diagnostics live on the iteration record
====================================================

The affine design's strongest genuine observation was that **a state
cannot carry "previous"**: a flux is a snapshot, with nowhere to put a
contraction factor, a previous increment, or an a-posteriori error
bound. That observation is correct. What it argues for is a home on the
**iteration**, not a second field type — because the thing that knows
"previous" is the *loop*, and the loop already has a durable record.

The single recorded trajectory
-------------------------------

:attr:`IterationRecord.increment_norms
<orpheus.numerics.convergence.IterationRecord.increment_norms>` is the
iterate-increment trajectory
:math:`\bigl(\lVert\Delta\psi^{(i)}\rVert\bigr)_i`, one entry per pass
that produced a typed iterate. It is the **single source**; the two
diagnostics are *derived views* of it, so they cannot drift from it.

The norm is the **space-induced norm of the principal bulk leaf** — not
the flat norm of the whole composite. :func:`!_principal_bulk_leaf`
(in :mod:`orpheus.numerics.iteration`) walks structure rather than
types: it descends ``interior`` for an
:class:`~orpheus.transport.full_field.FullField` composite and
``systems[0]`` for a
:class:`~orpheus.numerics.coupled_system.CoupledField` block iterate
(the convention: the coupling's first member is the principal field),
and duck-types only ``l2``. A bare-ndarray (L0) iterate yields no leaf
and records nothing, as before.

.. important::

   **The convention is load-bearing and it is measured.** `[M]` the
   interior-leaf *space* norm and the interior-leaf *flat* norm agree to
   :math:`2.29\times 10^{-16}` (0–1 ULP) on the pin fixture, because the
   ``angular_flux`` space carries ``inner_product_weights is None``
   *today*; but the interior-leaf norm and the **whole-composite** flat
   norm — which additionally ravels the boundary trace block — differ by
   :math:`4.71\times 10^{-3}`. That nine-order gap is what the capture
   gate ``tests/numerics/test_si_diagnostic_trajectory.py`` exists to
   catch, at ``rtol = 1e-12``.

**Why** :math:`\rho` **is deliberately not a stopping criterion.** The
record's :attr:`converged
<orpheus.numerics.convergence.IterationRecord.converged>` is
``all(criterion.cleared ...)`` over
:attr:`criteria <orpheus.numerics.convergence.IterationRecord.criteria>`,
so adding a :math:`\rho` entry there would put an *observation* into a
conjunction of *verdicts* and flip every producer's convergence answer.
:math:`\rho` is observed, never driven below a tolerance. The two
trajectories are also on different cadences — increments are recorded
per pass, criteria at the stop evaluation — and are deliberately not
co-indexed.

The contraction ratio
----------------------

:attr:`IterationRecord.contraction_ratios
<orpheus.numerics.convergence.IterationRecord.contraction_ratios>`
derives the Banach factor

.. math::
   :label: iterate-contraction-ratio

   \rho^{(i)} \;\approx\;
   \frac{\lVert \Delta\psi^{(i)} \rVert}{\lVert \Delta\psi^{(i-1)} \rVert}

.. vv-status: iterate-contraction-ratio documented

.. Rationale: this is the DEFINITION of the recorded diagnostic (a ratio
.. of two entries of increment_norms), not a solver claim in its own
.. right — the physics claim it enables (rho -> c) is the separate L1
.. gate named below. Its computed content is pinned by the frozen
.. trajectory in tests/numerics/test_si_diagnostic_trajectory.py (11
.. ratios at rtol=1e-12, with a measured mutation battery) and its
.. physical calibration by tests/sn/solve/test_si_convergence_diagnostics.py
.. (marked `l1`, pillar = closed-form). Neither carries a verifies(...)
.. marker today, so the label stays `documented`.

A value :math:`\rho > 1` diverges (wrong fixed point / unstable scheme);
:math:`\rho \approx 1` is stalled (the :math:`c \to 1` slow mode; the
curvilinear and reflective slow modes); :math:`\rho < 1` is healthy. It
turns the :math:`\rho`-blind :math:`\lVert\Delta\psi\rVert` stopping
reading honest. A pair whose *predecessor* norm is exactly ``0.0``
contributes no ratio — the iteration was already at the fixed point and
the ratio is undefined.

The true-error estimate — the :math:`c \to 1` false-convergence fix
--------------------------------------------------------------------

This is the load-bearing numerical content. For a geometric contraction
the distance from the current iterate to the fixed point is the tail sum

.. math::
   :label: iterate-true-error

   e^{(i)} \;=\; \sum_{j\ge 0} \rho^{j}\,\lVert\Delta\psi^{(i)}\rVert
          \;=\; \frac{\lVert\Delta\psi^{(i)}\rVert}{1-\rho} ,

.. vv-status: iterate-true-error documented

.. Rationale: the closed form of a geometric tail — a literature identity
.. (Adams & Larsen 2002; the standard Banach a-posteriori bound), not an
.. ORPHEUS solver claim. Its implementation is pinned by
.. tests/numerics/test_si_diagnostic_trajectory.py (the frozen value at
.. rtol=1e-12) and its identity against the recorded norm and ratio by
.. tests/sn/solve/test_si_convergence_diagnostics.py. Both are foundation /
.. l1 gates without a verifies(...) marker, so the label stays
.. `documented`.

so the bare increment :math:`\lVert\Delta\psi\rVert` **understates** the
true error by a factor :math:`1/(1-\rho)`. As the
:term:`scattering ratio` :math:`c \to 1` (optically thick,
near-pure-scatter), :math:`\rho \to 1` and the understatement blows up:
at :math:`c = 0.99`, :math:`\rho \approx 0.99` and the true error is
:math:`\sim 100\times` the increment — so a solve that "converges" at
:math:`\lVert\Delta\psi\rVert < \text{tol}` is actually
:math:`\sim 100 \cdot \text{tol}` from the solution. This is the
canonical source-iteration stall-masking-as-convergence trap
(:cite:`AdamsLarsen2002`).
:meth:`IterationRecord.true_error_estimate
<orpheus.numerics.convergence.IterationRecord.true_error_estimate>`
surfaces it, and **raises** when :math:`\rho \notin [0, 1)` (a
non-contracting iteration has no finite geometric-tail estimate) or when
fewer than two increments were recorded (no :math:`\rho` to estimate
with).

The convergence map
--------------------

:meth:`Field.where_largest <orpheus.numerics.field.Field.where_largest>`
returns the :math:`k` index tuples with the largest ``|values|``,
largest first, into this leaf's own layout — the per-cell /
per-group / :term:`per-ordinate <ordinate>` map of which entries
dominate. Applied to an increment it is the convergence map (a pole-cell
resonance, a material-interface slow mode, a lagging group); applied to
a residual it says where the equation defect concentrates.

It was **promoted from the retired displacement surface to**
:class:`~orpheus.numerics.field.Field` at CS3 step 1, and the reason is
the general shape of this whole section: the map reads only ``values``,
so it was never a property of difference-ness. Everything the retired
type carried that needed *only* the array moved down to
:class:`Field`; everything that needed *"previous"* moved up to the
:class:`~orpheus.numerics.convergence.IterationRecord`. Nothing needed a
type in between — which, in retrospect, is the whole ruling in one
sentence.

.. warning::

   :math:`\rho` **is defined on the SPACE norm** (user ruling,
   2026-08-19), and that is a forward commitment, not a description of
   today's arithmetic. Today the space norm and the Euclidean norm agree
   to 0–1 ULP because the angular-flux space carries
   ``inner_product_weights is None``. When a later phase installs the
   physical :math:`V_{\rm cell} \times w_n` metric on the spaces, `[M]`
   the recorded trajectory moves by up to
   :math:`1.12\times 10^{-3}` relative — nine orders above the capture
   gate's ``rtol = 1e-12``, so **that phase will legitimately RED the
   capture gate**. That is correct behaviour: the ruling makes the
   diagnostic follow the space, so the phase that changes the space owns
   re-deriving the frozen numbers, with a regeneration note. Do not
   "fix" that red by pinning the diagnostic to the Euclidean norm.


.. _affine-typed-residual:

The typed equation residual — the box-7 ``from_balance`` consumer
=================================================================

.. note::

   **The** ``affine-`` **in this section's anchor and equation label is
   a historical artefact of this page's former title, and is
   deliberately kept.** The residual role was *never* affine — it is a
   plain vector role, closed under ``+`` and ``-``, and nothing in this
   section changed at the CS3 overturn. The anchor is
   ``:ref:``-cited from eight sites across
   :doc:`/theory/foundations/boundary_conditions` and
   :doc:`/theory/foundations/coupled_block_operator`, and the label is
   an ``:eq:`` API; renaming them would buy a cosmetic gain and risk a
   silent dangling cross-document reference, which renders as plain text
   with no build warning at any severity. The name is stale; the claim
   is not.

The residual column of the role grid (B.5.2 left it
*minted-but-consumerless*) is wired by
:func:`~orpheus.sn.solver.evaluate_residual`, the **#208 box-7
consumer**. It evaluates the within-group balance defect

.. math::
   :label: affine-typed-residual-eq

   r \;=\; (L + C - S - N_{2n} - B)\,\psi \;-\; q

.. vv-status: affine-typed-residual-eq documented

.. Rationale: the DEFINITION of the typed residual (a named composition,
.. not a computed physical law) — structural, so `documented`. Its
.. verifiable content is pinned by tests/.../test_typed_residual_evaluation.py
.. (l0 + foundation): balance_map ~ 0 at the SI fixed point, detectably
.. localised under a 10% cell perturbation, the type/units/space of the
.. mint, the boundary/interior split identity, and relative_to.

as the typed composite
``FullField(bulk=AngularResidual, boundary=AngularBoundaryResidual)`` — the
**timeless** carrier (a residual is a one-shot balance defect, history-free;
P4.5 W-C confines the timed type to the driver iterate) —
minted via the named composition
:meth:`AngularResidual.from_balance <orpheus.transport.residuals.angular_residual.AngularResidual.from_balance>`
/
:meth:`AngularBoundaryResidual.from_balance <orpheus.transport.residuals.angular_boundary_residual.AngularBoundaryResidual.from_balance>`
— **NOT** a bare cross-class ``-`` (which would mis-type the defect as a
source). The operator output :math:`(L+C-S-N_{2n}-B)\psi` is a source/sink
composite (the B.5.2 typing); subtracting the source :math:`q` is a
**role** transition (source operands → residual result), so it must go
through ``from_balance``.

.. important::

   **The role-transition argument survives the overturn intact, and it
   is worth seeing why.** The cone ruling made *same-class* flux
   addition legal; it did not make *cross-class* arithmetic legal, and
   the residual mint is cross-class. A bare
   ``AngularSourceSink - AngularSourceSink`` type-checks and produces an
   :class:`AngularSourceSink`, mis-typing a **defect** as a **source** —
   a role transition hidden inside a same-shape ``-`` that the type
   system cannot see. That was Issue #201's original sin, it is
   orthogonal to the affine question, and ``from_balance`` is still the
   answer.

The residual carries three diagnostics (a typed defect a flux cannot
be):

- :meth:`AngularResidual.balance_map <orpheus.transport.residuals.angular_residual.AngularResidual.balance_map>`
  — the per-cell / per-group transport-balance violation
  :math:`\max_n |r_n(\vec r, g)|`. This is the **typed form of the
  per-ordinate flat-flux residual probe** (``vv-principles``
  Signature 1): it exposes WHERE the discrete balance is violated per
  cell, the localised defect that *global* conservation (telescoping
  particle balance) HIDES — for example the curvilinear pole-cell spike
  (the ERR-026 failure-mode-7 class, where the SI gives a large
  pole-cell error while global balance looks fine).
- :func:`~orpheus.sn.solver.boundary_vs_interior_split` — splits the
  composite into :math:`(\lVert r_\partial \rVert, \lVert r_{\rm bulk}
  \rVert)` (with :math:`\sqrt{b^2 + i^2} = \lVert r\rVert`, the same
  flat metric the SI test uses), discriminating a BC-realizer /
  reflective-trace defect from an interior-streaming defect — free from
  the typed composite.
- :meth:`AngularResidual.relative_to <orpheus.transport.residuals.angular_residual.AngularResidual.relative_to>`
  — the tolerance-portable relative residual :math:`\lVert r\rVert /
  \lVert q\rVert` (the bare residual has rate-density units, so its
  magnitude scales with :math:`\Sigma_t \cdot V_{\rm cell}`; dividing by
  the drive makes it problem-portable).

The residual is **additive / diagnostic** — never in the convergence
path. The SI stopping test stays the relative residual; the GMRES defect
stays the *flat* :math:`b - A\psi` on the raveled vector (never typed as
a field). The **consistent DSA** correction (Issue #2) landed on the
increment, not on this residual: the correction operator consumes the
iterate increment
:math:`\Delta\psi = \psi^{l+1/2} - \psi^{l}` (whose moment-0 is
Larsen's :math:`d_0` source — for the exact inner sweep the residual is
:math:`S\,\Delta\psi`, so the two carry the same information up to the
:math:`\hat\sigma_S h` weighting the proven :math:`G` map owns), and
its low-order operator is the **derived edge-centered SN-side
consistent system** — NOT the standalone diffusion loss
:math:`A_{\rm diff}`, which the R4 characterization measured
**divergent** as an accelerator for :math:`\sigma_t h \gtrsim 2`
(:mod:`orpheus.sn.acceleration.dsa`; the promised ``as_dsa_source``
was therefore never minted — a third moment-0 spelling — and the one
restriction is the canonical
:meth:`~orpheus.transport.fields.angular_flux.AngularFlux.integrate_angular`).

.. note::

   **DSA re-typed at the carve, and the shape of the change is the
   ruling in miniature.**
   :meth:`DSACorrection.apply <orpheus.sn.acceleration.dsa.DSACorrection.apply>`
   used to mint displacement-typed correction blocks; it now admits
   **one** interior type
   (:class:`~orpheus.transport.fields.angular_flux.AngularFlux`) and
   returns flux-typed blocks, so the update step
   :math:`\psi + \mathcal{C}\Delta\psi` is the plain vector add. The SI
   sweep increment :math:`\psi^{l+1/2} - \psi^{l}` and the Krylov swept
   vector — GMRES directions rebuilt flux-typed at the scipy
   ``from_flat`` boundary — are now the *same type*, which is exactly
   the #331 disagreement dissolving one layer down. A moment-windowed
   carrier still refuses loudly: that is an arm-1 admission boundary,
   not an ontology claim.


.. _cone-the-overturned-affine-design:

The overturned design (2026-06-08 → 2026-08-19)
===============================================

This section preserves the affine design and the argument that produced
it. It is kept, in the past tense, for three reasons: the reasoning was
serious and partly right; a future session that re-derives "an increment
is not a state" needs to find the ruling rather than re-implement the
type; and the shape of the mistake — a real concern placed one layer too
low — is the reusable lesson.

What the affine algebra was
----------------------------

Flux states were typed as an **affine space** :math:`\mathbb{A}` over a
difference vector space :math:`V` — a torsor. The four legal operations
were:

.. math::

   \psi_2 \ominus \psi_1 &\;\to\; \Delta\psi \in V
       \qquad&&\text{(the ONLY mint of a displacement)}\\
   \psi \oplus \Delta\psi &\;\to\; \psi' \in \mathbb{A}
       \qquad&&\text{(the torsor action } \mathbb{A}\times V\to\mathbb{A})\\
   \psi_1 \oplus \psi_2 &\;\to\; \bot
       \qquad&&\text{(no origin — } \textstyle\sum\lambda_i = 2
       \text{ lands off } \mathbb{A})\\
   \textstyle\sum_i \lambda_i \psi_i,\ \textstyle\sum_i\lambda_i=1
       &\;\to\; \psi \in \mathbb{A}
       \qquad&&\text{(the ``affine\_combination`` blend — the only multi-flux one)}

(Displayed **without a** ``:label:`` **deliberately.** A labelled
equation is an API: anything may ``:eq:``-cite it, and a citer inherits
whatever it states. These four lines state a retired claim, so they must
not be citable. The live algebra is :eq:`flux-vector-algebra`.)

The enforcement was a ``FluxRole`` mixin on the seven flux leaves, a
``Displacement`` marker with a representation-keyed sibling registry,
and seven displacement leaf classes. The whole apparatus spanned 16
production modules and 16 test modules.

The argument that was made for it
----------------------------------

The pre-carve algebra typed ``AngularFlux + AngularFlux →
AngularFlux``, treating :math:`\psi = 0` as an additive origin; the same
``__sub__`` mis-typed the increment as a *state*. This was read as the
**same sin** Issue #201 fixes for the residual: the bare subtraction
:math:`A\psi - q` typechecks as ``AngularSourceSink −
AngularSourceSink`` yet mis-types the defect as a source. In both cases
a role transition was hidden inside a same-shape ``-`` the type system
could not see.

Three expert frames converged on the affine fix. The table below is the
original, with the CS3 verdict added as a fourth column.

.. list-table:: The three frames that converged on the displacement type — and what became of each
   :header-rows: 1
   :widths: 18 30 26 26

   * - Frame
     - The structure it saw
     - What it contributed
     - Verdict at CS3
   * - **Affine geometry / torsor**
     - flux states as points in :math:`\mathbb{A}`;
       :math:`\Delta\psi` a vector in :math:`V`; the three ops as the
       torsor axioms
     - ``state + state`` becomes unrepresentable by construction; the
       #201 gate is a type consequence, not a runtime check
     - ⛔ **REFUTED.** The premise is false: flux has a canonical zero
       and superposition is a theorem
       (:ref:`cone-overturn-adjudication`, arguments 1–3). The frame was
       applied to the wrong object; the genuinely affine object is the
       singular system's solution manifold (argument 6).
   * - **Banach fixed-point / contraction**
     - :math:`\Delta\psi^{(i+1)} = M\,\Delta\psi^{(i)}`,
       :math:`M = (L+C)^{-1}(S+N_{2n}+B)` — an exact linear recurrence on the
       increments
     - the increment is the natural carrier of the contraction factor
       :math:`\rho`, the a-posteriori bound, the Aitken extrapolation —
       data a state has nowhere to put
     - ✅ **UPHELD, RE-HOMED.** The diagnostics are real and are kept;
       the carrier is the *iteration record*, not a field type
       (:ref:`cone-iterate-diagnostics`). Note the frame's own equation
       is a **linear** recurrence on :math:`V` — it never needed
       :math:`\mathbb{A}`.
   * - **Krylov / residual dual**
     - :math:`\Delta\psi = (L+C)^{-1}\,\tilde r` — the increment and the
       residual are the SAME defect, mapped between the two universes by
       the sweep
     - the increment (flux, primal, SI-native) and the residual
       (rate-density, dual, Krylov/DSA-native) are duals, not
       competitors; a stopping criterion is a *frame* choice
     - ✅ **UPHELD, and it was the tell.** A Krylov method iterates on
       :math:`V`; under the affine typing it could only do so by
       erasing the types at ``to_flat``. That standing exception was
       Issue #331 waiting to be read (argument 4).

.. list-table:: The specific claims that were falsified, and what replaced each
   :header-rows: 1
   :widths: 44 56

   * - The affine-era claim
     - What is true now
   * - "flux states have no natural zero — the zero flux is a *chosen
       base point*, not an identity"
     - **False.** Vacuum is canonical, it is the unique no-source
       no-inflow solution, and it is the additive identity — pinned per
       leaf by ``test_zero_flux_is_the_additive_identity``.
   * - "``flux + flux`` is meaningless: two points cannot be added"
     - **False for one medium** (superposition is a theorem of the
       linear operator) and **true for different problems** — which is
       the fiber, enforced by class identity + space CONTENT equality
       (:ref:`cone-fiber-discipline`).
   * - "``affine_combination`` (:math:`\sum\lambda_i = 1`) is the only
       legal multi-flux blend"
     - **Dissolved.** It had zero production callers. In :math:`V` the
       relaxation blend
       :math:`\omega\psi_{\rm new} + (1-\omega)\psi_{\rm old}` is
       ordinary arithmetic, spellable everywhere and still pinned by
       ``test_relaxation_blend_is_plain_algebra`` — the ceremony went,
       the content stayed.
   * - "the displacement column is the dual of the residual column"
     - **Half true, and the half that fails is the operative one.** A
       residual crosses into rate-density — a role *and* a universe
       change, so it needs a class. A difference of fluxes changes
       neither units nor space nor fiber; it changes only whether the
       cone predicate holds, and a predicate needs no type.
   * - "the torsor round-trip
       :math:`\psi_1 \oplus (\psi_2 \ominus \psi_1) = \psi_2` is exact
       up to 8 ULP, not bit-exact"
     - **Still true, and still pinned** — now as a statement about one
       type: ``a + (b - a) != b`` bit-for-bit under IEEE-754 because the
       subtraction rounds. ``test_difference_round_trip_and_telescoping``
       asserts it, and the telescoping identity, to ``nulp=8``.
   * - "mint the full displacement type at one consumer, because the
       benefit is established rather than speculative"
     - **The exception was correctly *reasoned* and wrongly *taken*.**
       The rule it invoked (build a genuine primitive at the first
       consumer) is sound; what failed was the premise that the
       primitive was genuine. The lesson is not "defer harder" — it is
       that *"this makes an illegal state unrepresentable"* must be
       checked against **both** halves of the rule in
       :ref:`cone-membership-is-a-predicate`, and the affine type failed
       half 2 the moment a Krylov method needed :math:`V`.


Numerical evidence
==================

.. list-table:: Cone-era verification gates
   :header-rows: 1
   :widths: 30 20 50

   * - Gate (test)
     - Level / pillar
     - What it proves
   * - ``tests/numerics/test_flux_vector_algebra.py``
     - ``foundation``
     - The :math:`V` algebra of :eq:`flux-vector-algebra`, across four
       parameterized flux leaves (angular / scalar / moment /
       angular-boundary): ``ψ₁+ψ₂`` returns the leaf type carrying the
       bit-exact numpy sum and commutes bit-exactly; ``ψ + 0 == ψ`` with
       a *copy* contract (a ``return self`` short-circuit would pass a
       value-only assertion and is refused); ``ψ₂−ψ₁`` is the same leaf
       type and is **asserted to carry a negative entry** — the
       activation guard for the signed claim; round-trip and telescoping
       to ``nulp=8``; scalar algebra exact. **Mutation-verified at the
       carve:** ``Field.__add__`` → subtraction reds 12 value legs while
       every type leg stays green, which is why the type legs alone are
       not the gate.
   * - ``tests/numerics/test_flux_vector_algebra.py``
       (``test_fiber_guard_cross_mesh_refuses``)
     - ``foundation``, negative control
     - The fiber discipline lands on the **retained**
       ``_check_partner`` chain after the mixin retired. ⚠ The row was
       **re-derived at CS4b S3**, as its own docstring had instructed:
       the mesh tier is gone, so the correctly-blind leg is now a pair
       of *twin* carriers (distinct objects, equal space content) that
       legitimately ADD, and the refusal leg is a carrier whose cell
       edges moved, which reds on the base gate's space-content arm.
       The same-fiber positive leg is unchanged (``vv-principles`` #11
       pairing). **Mutation-verified:** deleting the SPACE arm of
       ``Field._check_partner`` reds the refusal leg while both
       positive legs stay green.
   * - ``tests/sn/solve/test_cone_membership_witness.py``
     - ``foundation``, production witness
     - :meth:`Field.cone_violations
       <orpheus.numerics.field.Field.cone_violations>` against converged
       output of the public entry, both ways: the benign row is wholly
       in :math:`K`; the thick row's report is **exactly** the
       negative-entry set with the worst violation first, and
       :math:`\phi` inherits the violation through the angular integral.
       Both legs carry activation guards, so a fixture that stopped
       discriminating fails loudly rather than passing vacuously. The
       measured tables are reproduced above.
   * - ``tests/numerics/test_si_diagnostic_trajectory.py``
     - ``foundation``, capture gate
     - **Value-neutrality of the diagnostics' relocation.** The
       :math:`\rho` trajectory (11 ratios), :math:`\lVert\Delta\psi\rVert`,
       :math:`\lVert\Delta\psi\rVert/(1-\rho)` and the
       ``where_largest`` map were frozen *before* the carve and
       reproduce *after* it at ``rtol = 1e-12``. Its claim kind is
       **RECORD** — on its own it says *something changed*, never
       *which side is right* — so it is deliberately anchored by the
       independent :math:`\rho \approx c` gate below and carries an
       in-file control proving it can RED.
   * - ``tests/sn/solve/test_si_convergence_diagnostics.py``
     - ``l1``, pillar = closed-form (:math:`\rho = c`)
     - ``record.contraction_ratios`` :math:`\to \rho \approx c` on a
       homogeneous slab: `[M]` :math:`\rho \in [0.40, 0.56]` at
       :math:`c = 0.5` and :math:`\rho \in [0.80, 0.92]` at
       :math:`c = 0.9`, with :math:`\rho(0.9) > \rho(0.5) + 0.2` — the
       discriminating cross-check that :math:`\rho` *tracks* :math:`c`
       rather than sitting at a constant. It also pins the identity
       ``true_error_estimate() == increment_norms[-1]/(1-ρ)``.
       **1-group is acceptable HERE** because :math:`\rho = c` is a
       flux-shape-independent *rate* claim, not an eigenvalue or
       flux-shape claim; the :math:`\ge 2`-group mandate guards the
       latter two, which this makes neither.
   * - ``tests/.../test_typed_residual_evaluation.py``
     - ``l0`` + ``foundation``
     - ``balance_map`` :math:`\approx 0` at the SI fixed point (relative
       defect :math:`< 10^{-7}`), detectably :math:`\ne 0` and localised
       when one interior cell is perturbed by 10 %
       (:math:`> 100\times` the converged peak, at the perturbed cell);
       ``from_balance`` mints the correct type / units / space and
       raises on a flux operand;
       :math:`\sqrt{b^2+i^2} = \lVert r\rVert`;
       ``relative_to(q)`` :math:`= \lVert r\rVert/\lVert q\rVert`.

.. admonition:: The mutation battery behind the capture gate
   :class: tip

   A gate is not evidence until a named mutation reddens it. At the
   freeze commit the capture gate was probed with five in-process
   mutations, and two of the results are worth carrying because they
   define the gate's blind spots:

   - :math:`\rho` **from the whole composite's flat norm** (the
     relocation error the module exists for) → **2 failed** (the
     :math:`\rho` pin and the :math:`\lVert\Delta\psi\rVert` pin).
   - ``Field.l2`` **→** ``np.linalg.norm`` (metric → flat) → **5
     passed**: the declared :math:`\le 1` ULP blindness, *measured*
     rather than assumed.
   - **A spurious leading ratio** (recording-cadence drift) → **1
     failed** (the length leg).
   - ``where_largest`` **ravelling before locating** → **1 failed** (the
     map leg only).
   - **Every norm** :math:`\times (1 + 10^{-9})` — the positive control
     → **1 failed, the** :math:`\lVert\Delta\psi\rVert` **leg only.**

   ⛔ That last row is a **stabiliser fact and it must not be
   over-read**: :math:`\rho` is a *ratio*, so any factor applied
   uniformly to the norm cancels **exactly** and the :math:`\rho`
   trajectory is invariant under the whole class
   (``vv-principles`` Mode 12). That error class is caught only by
   pinning :math:`\lVert\Delta\psi\rVert` in its own right, which is why
   the two must never be folded into one test.

.. note::

   **What this page can no longer claim, and why the loss is honest.**
   The #208 carve advertised "zero numerical change — the converged flux
   is bit-identical to a frozen ``sha256``". That claim was verified at
   the pre-carve commit and is **historical**: #333 retired the digests
   (a hash cannot report a *magnitude*, so 1 ULP and a catastrophic
   error were the same red once four verified quadrature commits
   legitimately moved the values). The CS3 carve makes the narrower and
   checkable claim instead — the **diagnostics** reproduce bit-tight
   through the relocation (``rtol = 1e-12``), the DD regression wall and
   the bit-identity wall stay green under
   ``-W error::DriftWarning``, and the algebra itself is pinned
   structurally rather than by a digest.


Development history
===================

Reverse-chronological (latest first) changelog of the architectural
milestones of *this page's* subject — the field algebra. The
S\ :sub:`N` solver's own milestone changelog is
:ref:`sn-development-history`, and the operator algebra's is on
:doc:`/theory/foundations/operator_algebra`. Entries marked *(in
development)* live on an unmerged feature branch and have no landed
merge-to-``main`` hash yet; trust ``git`` over this table for merge
status.

.. list-table::
   :header-rows: 1
   :widths: 10 50 12 28

   * - When
     - Architectural milestone
     - Issue
     - Where
   * - 2026-08-24
     - **A field is an element of a SPACE, and the fiber is space
       CONTENT** (campaign 1, phase CS4b). The leaves' space source
       flips to the carrier's cached, axis-built mints (S1/S2), and the
       partner gate's third tier — mesh-object identity — **retires**
       in favour of the base gate's space-content equality (S3, the
       **F2 doctrine**): twin carriers and BC-only-differing carriers
       now legitimately mix, while a moved cell edge, a different group
       structure or a different quadrature refuses exactly as before
       (:ref:`cone-fiber-discipline`). Trace and ray spaces gain
       content-digest names so ``(name, shape)`` equality IS content
       equality. S4 then retires the ``mesh`` binding itself and
       collapses the per-family ``_phase_space_shape`` hook into
       :class:`~orpheus.numerics.field.Field`'s own
       ``values.shape == space.shape`` check — a twin of the space's
       own content that died with the binding. S5 makes construction
       **space-primary** (the leaf sugar tier is deleted; call sites
       read a carrier mint), and S6.2 gives the canonical angular
       reduction and the isotropic-source projection ONE realization
       each — the space's frame-induced collapse pair
       (:ref:`spaces-collapse-pair` on
       :doc:`/theory/foundations/spaces`) — while the per-face packing
       loop re-homes to its layout's own
       :meth:`FaceLayout.pack
       <orpheus.numerics.face_layout.FaceLayout.pack>` (native place),
       leaving
       :meth:`BoundaryField.from_face_arrays
       <orpheus.transport.fields._bases.BoundaryField.from_face_arrays>`
       as the typed entry over it.
     - —
     - merged @ ``55bb47b9`` —
       ``4069155b`` / ``07e0fe77`` / ``8a205cbf`` (the carrier mints),
       ``9138b3c3`` … ``a82d31e4`` (the S3 content re-key),
       ``554ff10b`` / ``1333135e`` (the mesh binding retires),
       ``b00bf2d7`` … ``2690a434`` (space-primary construction),
       ``78925753`` / ``53e7d207`` (the reduction and the packer)
   * - 2026-08-19
     - **Flux lives in the positive cone** :math:`K \subset V`; the
       affine field algebra is overturned. Cone membership becomes an
       element **predicate**
       (:meth:`Field.cone_violations
       <orpheus.numerics.field.Field.cone_violations>`, returning offending
       indices) rather than a constructor invariant, because diamond
       difference does not preserve :math:`K` and a
       :math:`\psi \ge 0` type would refuse legitimate production
       output; cone **preservation** stays the realization's
       ``is_positivity_preserving`` flag. ``flux + flux`` becomes legal
       (superposition is a theorem of the linear operator);
       ``flux - flux`` returns the **same** leaf type, signed;
       ``affine_combination`` dissolves (zero production callers). The
       ``FluxRole`` mixin and the whole ``transport/displacements/``
       package retire; the iterate diagnostics (:math:`\rho`, the
       :math:`c\to1` true-error estimate) relocate onto
       :class:`~orpheus.numerics.convergence.IterationRecord` as views
       of one recorded ``increment_norms`` trajectory, and
       ``where_largest`` is promoted to
       :class:`~orpheus.numerics.field.Field`. The fiber discipline
       survives untouched on the retained ``_check_partner`` chain
       (class + space + mesh). Closes #331 — operators are linear on
       :math:`V`, so the "does my domain include the difference space?"
       disagreement between :math:`L`, :math:`S` and :math:`B` becomes
       unspellable.
     - #331
     - merged ``f9d571b5`` (branch ``refactor/cone-field-algebra``) —
       ``c3e66b18`` (diagnostics → the record), ``993fa280`` (the
       algebra flip + DSA re-typing), ``5efd2178`` (the displacement
       package retires), ``3b9e8651`` (the cone predicate + its DD
       witness)
   * - 2026-06-08
     - **Flux states typed as an affine space; the iterate increment is
       a typed displacement.** ``flux − flux`` minted a
       ``Displacement``, ``flux ⊕ displacement`` was the torsor update,
       and ``flux + flux`` was a :class:`TypeError` — the #201
       dimensional gate as a *type* consequence. The Role axis of the
       carrier grid. ⛔ **OVERTURNED 2026-08-19** — see the row above
       and :ref:`cone-the-overturned-affine-design`.
     - #208 / #201
     - ``main`` (Wave O step O.2)
