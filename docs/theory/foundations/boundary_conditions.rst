.. _theory-boundary-conditions:

============================================
Boundary Conditions — Trace-Law Architecture
============================================

.. contents:: Contents
   :local:
   :depth: 3


Key Facts
=========

**Read this before touching anything in** :mod:`orpheus.geometry.boundary`
**or any** ``BoundaryRealizer``.

- A boundary condition is a **method-agnostic affine map** on the
  transport equation's boundary trace:
  :math:`\gamma_- \psi = R\,G\,\gamma_+ \psi + q`, where
  :math:`\gamma_\pm` are the inflow / outflow trace operators,
  :math:`G` is the **deck transformation** (a specular mirror, a
  spatial wrap, a rotation), :math:`R` is the **constitutive response
  kernel** (an :term:`albedo` amplitude, or a rank-one angular kernel
  for diffuse re-emission), and :math:`q` is an optional prescribed
  inflow source. **Membership in** :math:`G` **is decidable by
  multiplicativity** — :math:`G(\psi\varphi) = (G\psi)(G\varphi)`
  holds for a relabeling and never for an average, so an angular
  average is an :math:`R`, not a :math:`G`. See :eq:`affine-bc-form`
  and :ref:`bc-factor-roles`; campaign phase B3.0 corrected the
  assignment (the Lambertian average had shipped in the geometry
  slot).
- **The** :math:`R\,G` **product is a TAXONOMY, not a computational
  factorization** (:ref:`bc-taxonomy-vs-factorization`). It answers
  *"is this law's content geometry or physics?"*; it does NOT say how
  the law is evaluated. **Whichever factor is non-trivial carries the
  crossing** :math:`\Gamma_+ \to \Gamma_-`, which is well defined
  because exactly one of them ever is: for a **quotient** law the
  crossing is geometric (:math:`G`), for a **constitutive** law the
  response does it by integrating the outgoing flux and re-emitting an
  incoming one — there is no ambient isometry at a wall to provide it.
  This bullet read ":math:`G : \Gamma_+ \to \Gamma_-` … it carries the
  crossing, because the mirror … is an ambient isometry" and
  ":math:`R : \Gamma_- \to \Gamma_-`" until 2026-08-04; the first was
  proven only for the mirror and the second is contradicted by every
  realized response, all of which type :math:`\Gamma_+ \to \Gamma_-`.
- The architecture has **three concrete layers**, connected by the
  kind-keyed law registry (#290 P7b dissolved the Wave-5 realizer
  registry — realizers are owned by their method-meshes):

  +-------+-----------------------+-----------------------------------------------------+
  | Layer | What                  | Where                                               |
  +=======+=======================+=====================================================+
  | 1     | Trace structure       | :mod:`orpheus.numerics.spaces.angular_trace_space`  |
  |       | (Γ\_-, Γ\_+ + mask)   | (all Mesh1D coord systems + 2-D Cartesian;          |
  |       |                       | 2-D cylindrical Mesh2D deferred)                    |
  +-------+-----------------------+-----------------------------------------------------+
  | 2     | Boundary law          | :mod:`orpheus.geometry.boundary` (ABC +             |
  |       | (method-agnostic)     | 7 concrete laws, kind-keyed law registry)           |
  +-------+-----------------------+-----------------------------------------------------+
  | 3     | Method realizer       | per-method packages (SN + diffusion,                |
  |       | (per-method strategy) | #290 P3), each owned by its method-mesh             |
  |       |                       | via the ``TransportMethod`` hook (P7b)              |
  +-------+-----------------------+-----------------------------------------------------+

- Rank-N boundary conditions (Marshak, partial-current mixes) are
  expressed via a **descriptor-tree algebra** on the unrealised laws
  themselves. The :class:`~orpheus.geometry.boundary.BoundaryTraceLaw`
  algebra dunders (``+``, ``-``, ``*``, ``/``, unary ``-``) return
  :class:`~orpheus.geometry.boundary.LawSum` /
  :class:`~orpheus.geometry.boundary.LawScaled` nodes — a closed
  algebra over ``BoundaryTraceLaw | LawSum | LawScaled``. The tree is
  a **pure descriptor** with no ``apply`` method; the
  :func:`~orpheus.geometry.boundary.realize_recursively` type
  transformer (method-blind, in ``geometry/boundary/`` since #290
  P7b; the leaf realizer is a required argument) walks it once per
  face and emits an operator tree of
  :class:`~orpheus.numerics.operator.OperatorSum` /
  :class:`~orpheus.numerics.operator.ScaledOperator` composers around
  realised 1-arg leaves. See :ref:`bc-trace-law-descriptor-model` and
  :ref:`bc-rank-n-algebra`. There is no dedicated
  ``MixedBoundaryOperator`` class (retired Wave 11); there is also no
  ``apply`` method on the raw law (retired Issue #186, B3 + β2).
- TWO functional realizers exist:
  :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` and
  :class:`~orpheus.diffusion.boundary_realizer.DiffusionBoundaryRealizer`
  (#290 P3 — every diffusion law collapses to the albedo-family
  scalar :math:`\mathcal{A}` in :math:`J^- = \mathcal{A} J^+`). Each
  is OWNED by its method-mesh: ``realize_boundary_law`` — the
  per-method arm of the
  :class:`~orpheus.transport.method.TransportMethod` Protocol —
  instantiates it directly, and the shared
  :func:`~orpheus.transport.method.resolve_boundary_conditions` body
  drives the per-face resolution for every method. The Wave-5
  ``BoundaryRealizerRegistry`` + the MoC/MC/CP
  ``NotImplementedError`` stub realizers were **dissolved at #290
  P7b**: no consumer ever resolved a realizer by name (you hold the
  method-mesh → you have its realizer), and the string indirection
  carried registration-timing hazards for zero payoff. A future
  MoC / MC / CP modernization mints its method-mesh + realizer pair
  directly — no central registration step.
- **Whether a configuration sweeps in one pass is a property of the
  whole face configuration, never of a single law.** No
  :class:`~orpheus.geometry.boundary.BoundaryTraceLaw` subclass
  carries a sweep-cycle flag. The per-law ``ClassVar`` that once
  claimed to — ``True`` on
  :class:`~orpheus.geometry.boundary.ReflectiveBoundary` and
  :class:`~orpheus.geometry.boundary.PeriodicBoundary` — was
  **retired 2026-07-30**: a boolean on the boundary *kind* cannot
  express a configuration-dependent property, since
  ``reflective|vacuum`` is acyclic while ``reflective|reflective``
  is not. The honest criterion is a strongly-connected-component
  decomposition of the :math:`(\text{face}, \text{ordinate})` trace
  digraph, computed by
  :mod:`orpheus.derivations.discrete.sn.sweep_acyclicity` and gated
  by ``tests/sn/sweep/test_sweep_acyclicity.py``. See
  :ref:`bc-sweep-cycle`.
- The eight typed errors :class:`~orpheus.geometry.boundary.IncomingOutgoingTraceClassificationError`
  through :class:`~orpheus.geometry.boundary.BoundarySourceNotOnIncomingTraceError`
  (ERR-040..ERR-047 in the V&V error catalog at
  ``docs/theory/verification/error_catalog.rst``) replace the
  pre-refactor generic :class:`ValueError` raises; every one is
  pinned by a ``@pytest.mark.catches("ERR-NNN")`` decorator on the
  test that fires it.
- **A realized SN law's DOMAIN is** :math:`\Gamma_+` **(campaign phase
  B3.2).** It consumes the outflow half-trace and produces the inflow
  half-trace — exactly the shape :eq:`affine-bc-form` states and the
  diffusion arm always had. The consumer composes
  :math:`B_{\rm face} = \iota_- \circ \text{law} \circ \gamma_+`
  (transpose :math:`\iota_+ \circ \text{law}^{\mathsf T} \circ
  \gamma_-`) out of the trace restrictions
  :class:`~orpheus.numerics.operator.TraceRestrictionOperator`; nothing
  is computed and then discarded. Pre-B3.2 the law was handed the
  **whole face slot** and the consumer threw the outflow rows away with
  a slice-write — see :ref:`bc-domain-narrowing`, and
  :ref:`bc-vacuum-semantic-correction` for what that removed.
- **Vacuum realizes to the ZERO MAP** :math:`\Gamma_+ \to \Gamma_-`
  (a :class:`~orpheus.numerics.operator.ZeroOperator` carrying both
  space hooks). Vacuum's whole content is :math:`R = 0`; with the
  domain narrowed there is nothing else to represent. Pre-B3.2 it
  realized to an ``IncomingOrdinateMaskTensor``
  — a full-face projector onto the *outflow* subspace whose preserved
  rows the consumer then discarded. That mask, and the "which rows does
  it preserve?" question two campaign phases had documented as having
  "no consumer today", are gone from the vacuum path:
  :ref:`bc-vacuum-semantic-correction`. The realizer path is uniform
  across every supported mesh (1-D Cartesian / spherical / cylindrical
  + 2-D Cartesian) since Issue #188 lifted the curvilinear deferral on
  the boundary trace space (then named ``InflowTraceSpace``; since
  #205 / #201 the unified
  :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`).
- **So does PRESCRIBED INFLOW — the same zero map, the same object**
  (campaign phase **P3**, 2026-08-05). The realizer tier realizes the
  law's LINEAR factor :math:`L = R\,G` and only that
  (:eq:`bc-affine-linear-factor`); for prescribed inflow :math:`L = 0`,
  so **vacuum and prescribed inflow differ only in** :math:`q`
  (:eq:`bc-prescribed-zero-linear-factor`). The source travels the
  typed boundary-source channel —
  :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`,
  the boundary leaf of the composite :math:`q = q_{\rm bulk} \oplus
  q_\partial` — and :func:`~orpheus.sn.solver._build_fixed_source_rhs`
  is its single construction point for both inner solvers. Until P3 this
  law realized to an ``IncomingSourceOperator`` whose ``apply`` ignored
  its input and returned :math:`q`: an **affine** map in a linear slot,
  which the missing :attr:`~orpheus.numerics.operator.BlockRole.BOUNDARY`
  stamp was believed to fence out of :math:`B` and did not
  (``SNBoundaryOperator._face_laws`` has no role filter — measured
  :math:`\lVert B(0) \rVert_\infty = q`, a doubled delivery on source
  iteration, and a raised
  :class:`~orpheus.sn.solver.ConvergenceCertificateError` on Krylov).
  Full derivation, measurements and gotchas:
  :ref:`bc-affine-source-channel`.
- **The face ordinate partition is THREE-way**, not two:
  :math:`\{1..N\} = I_f \sqcup O_f \sqcup T_f` with
  :math:`T_f = \{|\Omega\cdot\hat n| \le \texttt{TANGENTIAL\_EPS}\}`
  (:eq:`ordinate-partition-inflow-outflow`). **"Not inflow" is NOT
  "outflow".** Measured: a cylinder under ``product(n_mu=2, n_phi=4)``
  carries **4 of 8** ordinates tangential at ``xmax``; ``gauss_legendre(5)``
  carries 1; every ``lebedev`` carries 4–8; only ``gauss_legendre(4)``
  is the clean two-way case — **the slab is the unrepresentative
  fixture.** Measured too: :math:`|\Gamma_+| = |\Gamma_-|` on every
  quadrature × face in the tree, so **a shape assertion cannot
  distinguish** :math:`\Gamma_+ \to \Gamma_+` **from**
  :math:`\Gamma_+ \to \Gamma_-` (a vv Mode-12 invariance-group
  blindness — the measured functional has the error class in its
  stabiliser).
- **The realized boundary law is a first-class sibling operator**
  :math:`B` **in the SN algebra (Wave O steps O.4a.2 + O.4b, Issue
  #208).** For **every** SN geometry (1-D slab / sphere / cylinder and
  2-D Cartesian), the realized per-face law is assembled into the
  whole-trace
  :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator` — the
  :math:`A_{ss}` boundary block of the canonical loss operator
  :math:`(L_{\rm full} + C - S - F - B)`. The reflection
  :math:`\psi.\text{inflow} = B\,\psi.\text{outflow}` is **no longer
  re-applied inside the streaming sweep** for any geometry (O.4a.2
  made the 1-D sweep bare; O.4b made the 2-D wavefront sweep + matvec
  bare); it is delivered as the off-diagonal :math:`-B` source term and
  the outer Krylov / SI loop drives the boundary consistency residual
  to zero. The full block-matrix derivation and design rationale live
  at :ref:`bc-extraction`.

.. admonition:: V&V status

   This page is **L4-informational** with respect to correctness.
   The architecture documented here is structural — it makes the
   code understandable and composable but does not by itself verify
   any equation. The verification load is carried by:

   - L0 foundation tests on individual primitives
     (:mod:`tests.numerics`,
     :mod:`tests.geometry.test_boundary_trace_law`,
     :mod:`tests.geometry.test_bc_errors`).
   - Foundation reference-image tests
     (:mod:`tests.geometry.test_bc_equivalence_snapshot`), which
     compare each realised operator against a **frozen,
     independently-derived** reference image — see
     :ref:`bc-numerical-evidence`. Re-anchored on 2026-08-01: the
     committed ``.npz`` artefacts used to be recordings of production
     output (a drift lock, worth what the recorded code was right);
     they now carry an image computed from the law's own equation, so
     the gate states correctness rather than stability.
   - L1 descriptor-tree algebra tests
     (:mod:`tests.geometry.test_law_composition`) pinning the
     :class:`LawSum` / :class:`LawScaled` closed-algebra contract
     (foundation + L1 coverage).
   - L1 universal-invariant tests
     (:mod:`tests.geometry.test_bc_universal_invariants`) that fire
     ERR-043 / ERR-044 / ERR-046 under fault-injection.

   No equation on this page makes a claim that requires a closed-form
   or MMS reference; all equations are **definitional** or
   **structural-architecture** statements drawn from Grand Report v3
   §16, §16A, §16A.10.


.. _bc-overview-three-layers:

The §16A three-layer decomposition
==================================

A boundary condition in transport-theory codes is, in the
discrete-form-typical mathematical sense, a **single linear operator**
that takes the outgoing :term:`angular flux` at a face and returns the
incoming angular flux. In ORPHEUS we explicitly factor that single
operator into three layers because each layer carries different
mathematical, physical, and architectural responsibilities. The split
is taken verbatim from Grand Report v3 §16A.3 and the source-of-record
plan ``.claude/plans/transient-giggling-cake.md``.

.. _affine-bc-form:

Layer 2 — the affine law on the boundary trace
----------------------------------------------

The full mathematical form of every boundary law in this codebase is
the **affine map**

.. math::
   :label: affine-bc-form

   \gamma_- \psi \;=\; R\,G\,\gamma_+ \psi \;+\; q,

.. (vv-status rationale) Structural/definitional framing: the master affine
   form of every boundary law (Grand Report §16A.3). Per this page's own note,
   its equations are definitional / structural-architecture statements; the
   concrete rank-0 / rank-n realisations (:eq:`bc-rank-n-tensor-decomposition`,
   PrescribedInflow) are the tested forms. Not a solver claim.
.. vv-status: affine-bc-form documented

where:

* :math:`\gamma_\pm` are the **trace operators** that restrict the
  angular flux :math:`\psi(\mathbf{r}, \Omega)` from the volumetric
  function space to the inflow / outflow boundary trace spaces
  :math:`\Gamma_\pm` (see :ref:`bc-trace-structure` below for the
  formal definition).
* :math:`G` is the **deck transformation** — a measure-preserving
  permutation, pushforward, or spatial wrap-around. It carries pure
  geometry (it changes nothing about the physical interaction at the
  boundary; it just relabels the angular fluxes that meet there). When
  it is the non-trivial factor it is a map
  :math:`\Gamma_+ \to \Gamma_-` and it carries the **crossing**,
  because the mirror that exchanges the two hemispheres is an ambient
  isometry — see :ref:`bc-factor-roles` below.
* :math:`R` is the **response kernel** — the constitutive law. A scalar
  amplitude in :math:`[0, 1]` for the standard sub-Markov BCs (albedo,
  partial-current), or a rank-one angular kernel for diffuse
  (Lambertian) re-emission. When it is the non-trivial factor it is
  the one that crosses, and every realized response in the tree is
  typed :math:`\Gamma_+ \to \Gamma_-` accordingly. The general
  weak-form angular kernel remains deferred; see the
  :class:`~orpheus.geometry.boundary.BoundaryError` catalog and
  Issue #175 close-out follow-ups.

  .. note::

     This bullet typed :math:`R` as an endomorphism
     :math:`\Gamma_- \to \Gamma_-` until 2026-08-04, and the
     :math:`G` bullet above claimed the crossing for :math:`G`
     unconditionally. As a **classification** the pair
     :math:`G : \Gamma_+ \to \Gamma_-`,
     :math:`R : \Gamma_- \to \Gamma_-` is coherent and it answers the
     question the taxonomy exists to answer; it is *not* the realized
     typing of either factor, and reading it as such is what the new
     :ref:`bc-taxonomy-vs-factorization` separates out.

* :math:`q \in \Gamma_-` is the **prescribed inflow source** — a
  vector-valued quantity on :math:`\Gamma_-` only. The empty case
  :math:`q \equiv 0` is the homogeneous BC; the inhomogeneous case
  :math:`q \neq 0` is the rank-0 affine BC
  :class:`~orpheus.geometry.boundary.PrescribedInflow`. It is **not**
  realized as an operator: :math:`q` is the one factor the realizer
  tier does not carry, travelling instead as the boundary leaf of the
  composite source — see :ref:`bc-affine-source-channel`, which is also
  where the consequence of that split (vacuum and prescribed inflow
  realize to the *same* operator) is derived and measured.

Three remarks make this form load-bearing:

1. **Method-agnostic.** Nothing in :eq:`affine-bc-form` is SN-specific.
   The same affine map describes how MoC track-bundles, MC particle
   histories, CP boundary-to-region coupling matrices, and diffusion
   bilinear-form weak BCs all interact with the geometry. Each
   method's *realization* of the operators :math:`G`, :math:`R`,
   :math:`q` differs (see :ref:`bc-realizer-layer`); the algebraic
   shape of the law itself is universal.
2. **Affine, not linear.** The :math:`q` term is what makes the map
   affine. Most published transport-theory references treat the
   homogeneous case (:math:`q \equiv 0`) and never give the affine
   form an explicit name; ORPHEUS does because two distinct rank-0
   cases (:class:`~orpheus.geometry.boundary.VacuumInflow` with
   :math:`R = q = 0` and
   :class:`~orpheus.geometry.boundary.PrescribedInflow` with
   :math:`R = 0` but :math:`q \neq 0`) need a single uniform
   contract. Note that it is :math:`R` alone that vanishes in both:
   :math:`G` is the **identity** deck element, not zero — the zero map
   is not a bijection and so cannot be a geometry map at all. Writing
   ":math:`R = G = 0`" spelled the same vanishing twice, once in the
   wrong tier; campaign phase B3 corrected it.
3. **The three operators are first-class.** The
   :class:`~orpheus.geometry.boundary.BoundaryTraceLaw` ABC exposes
   :attr:`~orpheus.geometry.boundary.BoundaryTraceLaw.geometry_map`,
   :attr:`~orpheus.geometry.boundary.BoundaryTraceLaw.response_kernel`,
   and :attr:`~orpheus.geometry.boundary.BoundaryTraceLaw.source`
   as Python properties on every concrete subclass. The properties
   default to ``None / None / NoSource()``; concrete laws override
   when applicable. The split lets cross-method realizers introspect
   the law's geometric and response components separately — the SN
   realizer dispatches on the law's class today, but a future
   weak-form realizer might dispatch on the geometry / response /
   source independently.

.. _bc-factor-roles:

Which factor owns what — the decidable criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two factors are not "the first thing that happens" and "the second
thing that happens". They are **different kinds of mathematical
object**, and membership is decidable — but it takes **two** tests, and
the first published form of this section shipped only one.

**The necessary test — multiplicativity.**

    :math:`G` is the **composition (Koopman) operator of a
    measure-preserving bijection of the boundary phase space** —
    :math:`(G\psi)(x) = \psi(g^{-1}x)` for some :math:`g` acting on
    :math:`\partial\Omega \times S^d`.

Such operators are invertible, preserve the trace measure
:math:`|\Omega\cdot\hat n|\,d\Omega\,dA`, form a **group**, and are
**multiplicative**: :math:`G(\psi\varphi) = (G\psi)(G\varphi)`. A
relabeling satisfies that identity; **an averaging operator never
does**. Anything *failing* multiplicativity is a kernel, and kernels
are :math:`R` — the argument that moved the Lambertian out of the
geometry slot at campaign phase B3.

**Why that test is not sufficient.** A *specular kernel* is a
permutation, hence multiplicative too. So multiplicativity alone cannot
separate

* a polished wall that returns :math:`\alpha` of the flux specularly,
  from
* a symmetry plane, across which the domain genuinely continues.

Both relabel; only one is geometry. Reading multiplicativity as *the*
criterion puts a surface's re-emission law in :math:`G`, and the two
objects have nothing in common but their matrix.

**The sufficient test — is it a quotient?**

    :math:`G` is the deck transformation **of an actual quotient of the
    physical domain** (:ref:`bc-factor-quotients`).

A physical surface is not a quotient: the domain does *not* continue on
the other side of it, so nothing is identified with anything and there
is no deck group to be an element of. A surface's specular pairing is
therefore **constitutive** — it is :math:`R`.

.. admonition:: The law this yields
   :class: important

   **Exactly one of** :math:`G`, :math:`R` **is non-trivial.**

   It is the contrapositive of *"*:math:`R = I` *exactly when the BC is
   a pure symmetry statement adding no physics"* — a law that asserts
   any physics at all has :math:`G = \mathrm{id}`. See
   :ref:`bc-factor-quotients` for the table, and for the one shipped
   row that violates it deliberately.

Physically: a change of direction caused by the **geometry** is
:math:`G`; a change of direction caused by the **constitutive
assumption of the BC** is :math:`R`. Absorption, accommodation and
diffusivity are :math:`R`. Mirrors, translations and rotations are
:math:`G` — but only when they are mirrors *of the domain*, not of a
wall standing in it.

**Whichever factor is non-trivial carries the crossing.** For a
**quotient** law the crossing is geometric, and the argument is exact:
the specular mirror
:math:`\Omega \mapsto \Omega - 2(\Omega\cdot\hat n)\hat n` is the unique
ambient isometry fixing the face; it exchanges the hemispheres and
preserves :math:`|\Omega\cdot\hat n|`. So across a symmetry plane the
passage from :math:`\Gamma_+` to :math:`\Gamma_-` is not something the
*physics* does — it is something the *geometry* provides, and
:math:`G`, not :math:`R`, carries it.

For a **constitutive** law there is no such isometry to lean on. A wall
is not a quotient — that is the sufficient test above — so nothing
identifies an outgoing direction with an incoming one geometrically,
and the crossing is performed by the *physics*: the response integrates
the outgoing flux and re-emits an incoming one. The realization is the
witness. The response :math:`R_{\text{diff}}` of the white /
diffuse-albedo law realizes as a **two-link chain** whose links state
the crossing out loud:
:class:`~orpheus.sn.boundary.angular.PartialCurrentOperator` collapses
the *outflow* angle axis, and
:class:`~orpheus.sn.boundary.angular.IsotropicEmissionOperator`
broadcasts the result over the *inflow* one, so the composite types
itself :math:`\Gamma_+ \to \Gamma_-` (:ref:`bc-response-adjoint`); the
specular kernel behind
:class:`~orpheus.geometry.boundary.SpecularReemission` is likewise a
narrowed :math:`\Gamma_+ \to \Gamma_-` permutation. `[M]` **no realized
response is an endomorphism of** :math:`\Gamma_-`.

Since exactly one of :math:`G`, :math:`R` is non-trivial, "the
non-trivial factor crosses" is well defined. Two boundary cases sharpen
it rather than break it:

* **Rank-0 laws** (vacuum, prescribed inflow). :math:`R = 0`, so the
  composite is the zero map on :math:`\Gamma_+` landing in
  :math:`\Gamma_-`: it *does* cross, vacuously, because no pairing
  information is needed to emit nothing.
* **A bare scalar response on an ANGULAR trace.** :math:`R = \alpha I`
  is non-trivial in *magnitude* and trivial in *angular structure* — it
  commutes with everything and therefore cannot pair directions — while
  :math:`G = \mathrm{id}` supplies no crossing either. So neither
  factor crosses, and the law is genuinely under-determined: this is
  exactly the closure-free ``AlbedoBoundary(α)`` spelling the SN
  realizer **refuses**, and the same law is *complete* on a scalar
  trace where :math:`J^- = \alpha J^+` exhausts the single degree of
  freedom. See :ref:`bc-method-realizability` for that scalar-vs-angular
  axis, which is the same distinction one level up.

.. note::

   This paragraph read *"The crossing is geometric … which is why*
   :math:`G` *and not* :math:`R` *carries it"* until 2026-08-04. The
   mirror argument it gives is correct and is retained above; what was
   wrong is that it was stated for **every** law, and a law with no
   isometry has no geometric crossing to inherit.

**Why a misassignment can hide — a theorem.** If :math:`R` is rank-one,
:math:`R = u \otimes v`, then

.. math::

   R \circ G \;=\; u \otimes \bigl(G^{\mathsf T} v\bigr),

a :math:`\Gamma_+ \to \Gamma_-` operator — the same type as the boundary
law's composite action, with :math:`u \in \Gamma_-` and
:math:`G^{\mathsf T} v \in \Gamma_+^{*}`. Now :math:`G` is the
composition operator of a measure-preserving bijection :math:`g`, and
that is precisely the hypothesis under which the three candidate
"transposes" coincide: :math:`G^{-1} = G_{g^{-1}}` because :math:`g` is
a bijection, and preserving the trace measure makes that inverse both
the Euclidean transpose and the metric adjoint (the same theorem the
deck-transformation row of
:ref:`the evaluation table <bc-taxonomy-vs-factorization>` cites). So
:math:`G^{\mathsf T} v = v \circ g`, and the Lambertian's
:math:`v = |\Omega\cdot\hat n|` is **preserved** by both the mirror and
the periodic translation — so :math:`v \circ g` is *the same function*
:math:`|\Omega\cdot\hat n|`, merely read on :math:`\Gamma_+` rather than
on :math:`\Gamma_-`. Hence :math:`R \circ G` comes out the SAME operator
whichever admissible :math:`G` was used:

    :math:`G` is **unobservable exactly when** :math:`R` is rank-one.

Read "unobservable" precisely: it means the composite does not depend on
*which* :math:`G` it was given. The hypothesis doing the work is that
:math:`v` is :math:`G`-invariant — true of every response this codebase
ships, because the only rank-one response is the Lambertian and its
:math:`v` *is* the trace measure's own weight. A hypothetical rank-one
response with a direction-selective :math:`v` (say
:math:`v = \delta_{\Omega_0}`) would make :math:`G` fully observable, so
do not read the slogan as "rank-one alone suffices".

The white BC is precisely the invariant case. Its :math:`G` slot
therefore had no observable consequence, and the physics drifted into it
— the cosine-weighted Lambertian average shipped as a geometry map until
campaign phase B3 moved it to the response tier where it belongs. The
same theorem is why that correction leaves the composite :math:`R\,G`
unchanged by construction.

.. note::

   **This step concluded** ":math:`R \circ G = R`" **until 2026-08-04.**
   That spelling type-checked only while the two half-traces were the
   same space in practice: with :math:`G : \Gamma_+ \to \Gamma_-` and
   the classifying :math:`R : \Gamma_- \to \Gamma_-`, the left side is
   :math:`\Gamma_+ \to \Gamma_-` and the right side is
   :math:`\Gamma_- \to \Gamma_-`, so they cannot be equal as operators.
   The step silently identified :math:`G^{\mathsf T} v` with :math:`v`
   by treating :math:`v` as the *function* :math:`|\Omega\cdot\hat n|`
   without tracking which half-trace it is restricted to — harmless
   before campaign phase **B3.2** (:ref:`bc-domain-narrowing`) narrowed
   the SN law onto :math:`\gamma_\pm` and made the two halves genuinely
   distinct spaces, a type abuse afterwards. **The theorem itself is
   untouched**: its content is the :math:`G`-independence of the
   composite, which is what makes :math:`G` unobservable and what makes
   the B3 correction safe. Nothing numerical changes.

.. _bc-factor-quotients:

The quotient picture — which BC is the orbifold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`G` is the **deck transformation of the quotient by which the
physical domain is represented**:

.. list-table::
   :header-rows: 1
   :widths: 18 30 20 32

   * - BC
     - quotient
     - fixed points
     - what it is
   * - periodic
     - :math:`\mathbb{R}^d/\Lambda` by a translation
     - none — **free**
     - a torus; a genuine **covering space**, a manifold
   * - reflective
     - by a reflection
     - the mirror plane
     - an **orbifold** (Thurston *reflector* boundary)
   * - rotational (⅛-core)
     - by a finite rotation
     - the axis
     - an **orbifold** (cone points)

The orbifold label therefore attaches to **reflective**, not to
periodic — periodic is the free / covering-space case. And
:math:`R = I` **exactly when** the BC is a pure symmetry statement
adding no physics. Vacuum, white and albedo are not symmetry statements
at all, which is why their :math:`G` is the identity deck element and
all their content sits in :math:`R`.

Read as a *test* rather than as a remark, that sentence is the
sufficient criterion of :ref:`bc-factor-roles`, and it partitions every
implemented law:

.. list-table:: Which tier owns each law's content
   :header-rows: 1
   :widths: 30 22 24 24

   * - law
     - :math:`G`
     - :math:`R`
     - what it asserts
   * - ``ReflectiveBoundary(axis)``
     - ``SelfPairedDeck.mirror(axis)``
     - :math:`I`
     - a symmetry plane — a quotient, **zero physics**
   * - ``PeriodicBoundary(axis)``
     - ``PairedDeck.wrap(axis)``
     - :math:`I`
     - a torus — a quotient
   * - ``AlbedoBoundary(α, SpecularReturn)``
     - ``SelfPairedDeck.identity()``
     - ``SpecularReemission``
     - a **surface** returning :math:`\alpha` specularly
   * - ``AlbedoBoundary(α, IsotropicReturn)``
     - ``SelfPairedDeck.identity()``
     - ``LambertianReemission``
     - a surface returning :math:`\alpha` diffusely
   * - ``WhiteBoundary(axis, sign, α)``
     - ``SelfPairedDeck.identity()``
     - ``LambertianReemission``
     - the same diffuse surface, under its traditional name
   * - ``VacuumInflow``
     - ``SelfPairedDeck.identity()``
     - :math:`0`
     - a surface returning nothing

Two entries realize to the *same matrix* as a geometry-tier law and are
nonetheless different objects:
``AlbedoBoundary(α, SpecularReturn(a))`` is
``ReflectiveBoundary(a, α)``'s matrix, and
``AlbedoBoundary(α, IsotropicReturn(a, s))`` is
``WhiteBoundary(a, s, α)``'s. Keeping the *types* distinct is what
makes "put a wall's response in the geometry slot" unspellable — the
exact error this section's earlier form permitted. In the code the two
routes share one realization body, so the equivalences hold by
construction rather than by two transcriptions agreeing.

.. warning::

   **One shipped row violates the law, deliberately.**
   :class:`~orpheus.geometry.boundary.ReflectiveBoundary` still accepts
   an ``albedo`` parameter, so ``ReflectiveBoundary(axis, 0.7)`` has
   BOTH factors non-trivial. A symmetry plane cannot absorb — that
   object is ``AlbedoBoundary(0.7, SpecularReturn(axis))`` wearing the
   geometry costume. It is unreachable from a ``BC(...)`` tag (the tag
   parser hard-codes :math:`\alpha = 1`), so nothing production-facing
   rides on it; retiring the parameter is campaign phase **B5**.
.. note::

   **SN apply matvec honours the affine BC contract (Issue #168
   Phase C, 2026-05-12).** The then-production per-geometry matvecs
   (``transport_operator_matvec_spherical`` and ``_cylindrical`` —
   since deleted in the typed-field campaign (#197), their successor
   ``_transport_operator_matvec_unified`` in turn retired at the walk
   unification (#280 campaigns)) were rewritten as one sweep
   iteration semantically: the BC trace law is applied **at least
   once** per matvec at the boundary edge on the WDD-propagated
   outflow face values (:math:`\gamma_+ \psi`), not on cell-centre
   approximations.  The live forward action is now
   :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply`
   through the loss-representation walk. The pre-Phase-C cell-centre-as-face-value
   contamination — and the Phase A ``BoundaryFaceFlux`` Protocol
   that patched it — both retire in Phase C. See
   :ref:`bc-trace-contract-respected-by-matvec` for the
   verification gate that pins this contract, and
   :ref:`bc-two-bc-applies-per-matvec` for the Phase D
   strengthening that audits **two** BC apply calls per matvec
   (Phase D Carlson context + Phase C trace law).

.. _bc-taxonomy-vs-factorization:

The taxonomy and the factorization are different questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything the two sections above say about :math:`R \circ G` is a
**taxonomy**. Two successive designs in this campaign were built on the
premise that it is also a *recipe for evaluation*, and both were refuted
by the tree's own realization — which is why the distinction gets its
own section rather than a parenthesis.

**What** :math:`R \circ G` **is.** A classification of the law's
*content*: the multiplicativity test plus the quotient test decide
whether a boundary law asserts geometry or physics
(:ref:`bc-factor-roles`, :ref:`bc-factor-quotients`). In that register
the typing :math:`G : \Gamma_+ \to \Gamma_-`,
:math:`R : \Gamma_- \to \Gamma_-` is perfectly coherent and it answers
the only question the taxonomy exists to answer. What makes the
classification load-bearing is that it is **decidable** and that it
makes "put a wall's response in the geometry slot" unspellable.

**What it is NOT.** How a law is *evaluated* — and, decisively, how its
adjoint is obtained — follows the law's **kind**, not the two-factor
product:

.. list-table:: Evaluation follows the KIND, not the classifying product
   :header-rows: 1
   :widths: 16 40 44

   * - kind
     - structure
     - adjoint
   * - **deck transformation**
     - **Atomic.** A measure-preserving bijection does not factor into
       two meaningful pieces. Pure geometry — a law imposed by theorem
       and transport-method-agnostic apart from needing to know *which
       space* it acts on. It is :math:`\Gamma_+(f) \to \Gamma_-(f)` for a
       self-paired face and :math:`\Gamma_+(f') \to \Gamma_-(f)` through
       a genuine face pair.
     - A **theorem**: the composition operator of a bijection :math:`g`
       is invertible with :math:`G^{-1} = G_{g^{-1}}`, and
       measure-preservation makes that inverse the transpose. Nothing to
       verify per law.
   * - **response**
     - :math:`N` **composed** operations
       :math:`\Gamma_+(f) \to \cdots \to \Gamma_-(f)`. The Lambertian is
       an outflow **angle contraction** :math:`C` followed by an
       **isotropic broadcast** :math:`B`, so :math:`R = B \circ C`, with
       an intermediate state in the angle-integrated per-face
       **scalar-current space** :math:`S(f)`. Constitutive — an
       assumption about a real surface.
     - **Conditional** on :math:`S(f)` carrying a non-degenerate metric
       — see :ref:`bc-response-adjoint`.

:math:`S(f)` is not bookkeeping: it is the **outgoing partial current**
:math:`J^+_g = \sum_{\Gamma_+} w\,\lvert\Omega\cdot\hat n\rvert\,\psi`,
a real physical quantity with units, and it is exactly what the
Lambertian's realization used to compute and immediately throw away as
the anonymous local ``psi_avg`` inside the welded
``AngularAverageOperator``'s apply. Both halves shipped at **G6.3**
(2026-08-04): the per-face accessor is
:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.current_space`
(step 2) and the factored :math:`B \circ C` spelling is
:class:`~orpheus.sn.boundary.angular.IsotropicEmissionOperator`
:math:`\circ`
:class:`~orpheus.sn.boundary.angular.PartialCurrentOperator` (step 3).

.. note::

   **The host is a new tier of the** :math:`\Gamma` **ladder, not**
   :class:`~orpheus.numerics.spaces.scalar_trace_space.ScalarTraceSpace`.
   This paragraph predicted the latter — "the type that will host it
   exists" — and the prediction did **not** hold. ``S(f)`` is not a
   subset of ordinates but the *collapse of the ordinate axis*, one
   value per group per boundary cell, carrying a deliberately **unit**
   metric because the angular measure
   :math:`\lvert\Omega\cdot\hat n\rvert \odot w` has already been
   consumed by the contraction that lands there.
   :class:`~orpheus.numerics.spaces.scalar_trace_space.ScalarTraceSpace`
   carries the :math:`(J^+, J^-)` **pair** for the whole boundary under
   the face-**area** metric — diffusion's P1 Cauchy data. Same physical
   name, different object, different metric, different scope; hosting
   one in the other would have double-counted the area weight.

.. important::

   **Why the conflation was invisible, and what it cost.** Reading the
   classifying product as the computational one is what made
   ":math:`R \circ G = R`" look sound (it is a type error the moment
   :math:`\Gamma_\pm` are distinct spaces —
   :ref:`see the theorem <bc-factor-roles>`) and what made "the crossing
   is geometric" look general (it is proven for the mirror only). Both
   claims shipped — on this page and in the
   :mod:`orpheus.geometry.boundary` Protocols — and both were corrected
   on 2026-08-04.
   The transferable lesson is the campaign's own: **a declaration tier's
   typing is a declaration, not a measurement — check it against the
   realization before designing on it.** In this case one read of the
   then-welded ``AngularAverageOperator``'s first line, which typed
   itself :math:`\Gamma_+ \to \Gamma_-`, refuted both — and the typing
   survived the operator: the chain that replaced it composes
   :math:`\Gamma_+ \to S(f) \to \Gamma_-`, so the refutation now reads
   off the two links' bound spaces instead of one docstring.

.. _bc-response-adjoint:

When a response has an adjoint — the intermediate metric cancels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A deck transformation's adjoint is free (the theorem above). A
**response**'s is not, because factoring it introduces an intermediate
space :math:`S(f)` whose metric appears in each factor's Hilbert
adjoint. Write :math:`G_+`, :math:`G_-`, :math:`G_S` for the metrics on
:math:`\Gamma_+`, :math:`\Gamma_-`, :math:`S(f)`, and recall the Hilbert
adjoint of a map between metric spaces,
:math:`A^{*} = G_{\mathrm{dom}}^{-1} A^{\mathsf T} G_{\mathrm{cod}}`.
For :math:`R = B \circ C` with :math:`C : \Gamma_+ \to S` and
:math:`B : S \to \Gamma_-`,

.. math::
   :label: bc-response-factored-adjoint

   R^{*} \;=\; C^{*} B^{*}
   \;=\; \bigl(G_+^{-1} C^{\mathsf T} G_S\bigr)
         \bigl(G_S^{-1} B^{\mathsf T} G_-\bigr)
   \;=\; G_+^{-1} C^{\mathsf T} B^{\mathsf T} G_-
   \;=\; G_+^{-1} R^{\mathsf T} G_- .

.. (vv-status history) This equation carried
   ``.. vv-status: bc-response-factored-adjoint documented`` until G6.3 step 3b
   (2026-08-04). The sentinel's rationale was: "The FACTORED spelling does not
   exist in code yet — the Lambertian ships as one operator with
   ``is_adjointable=False``, so there is no function for a ``verifies`` marker
   to point at." Both halves shipped — ``S(f)`` is
   ``AngularTraceSpace.current_space``, the factorization is
   ``IsotropicEmissionOperator @ PartialCurrentOperator`` — so the precondition
   expired and the directive is removed rather than left as a standing claim
   that the equation is documented-only. It is now GATED on the shipped chain:
   ``tests/sn/operators/test_lambertian_chain.py::
   TestReciprocityAgainstTheMirrorFace::test_H_is_pointwise_the_mirror_face_kernel``
   carries ``verifies("bc-response-factored-adjoint")``, alongside six
   abstract-matrix gates in ``tests/numerics/test_factored_adjoint_identity.py``.
   Its PRECONDITION is separately gated:
   ``tests/numerics/test_angular_face_trace_space.py::
   test_the_half_trace_metric_is_strictly_positive`` pins the non-degeneracy
   this equation requires of the intermediate, and
   ``test_the_metric_is_not_euclidean`` pins that the metric is load-bearing.

**The intermediate metric cancels.** :math:`G_S` appears once as a
factor and once as its inverse, so it drops out of the composite
entirely — which means the requirement on :math:`S(f)` is exactly **one
binary condition: the metric must be non-degenerate.** Not "the
physically correct metric". Measured with :math:`|\Gamma_+| = 7 \neq
|\Gamma_-| = 5` deliberately (so no accidental index pairing can mask
an error), sweeping :math:`G_S` over eleven orders of magnitude:

.. list-table:: :eq:`bc-response-factored-adjoint`, measured 2026-08-04
   :header-rows: 1
   :widths: 34 30 36

   * - :math:`G_S`
     - :math:`\max\lvert R^{*}_{\text{factored}} - R^{*}_{\text{direct}}\rvert`
     - reading
   * - identity (Euclidean)
     - ``0.000e+00``
     - exact
   * - ``1e-6``
     - ``1.110e-16``
     - one ULP — cancels
   * - ``3.7e5``
     - ``1.110e-16``
     - one ULP — cancels
   * - **``0`` (degenerate)**
     - **``7.628e-01``**
     - ⚠ **BROKEN** — no adjoint exists

and the weighted adjoint law
:math:`\langle Rx, y\rangle_{G_-} = \langle x, R^{*}y\rangle_{G_+}`
holding at exactly ``0.0``.

The probe is a G6.3 design-session artefact and is reproducible in a
dozen lines, so it is described rather than pinned to a path: build
:math:`C = (\cos\!w/\operatorname{norm})` as a :math:`(1, 7)` row and
:math:`B = \mathbf{1}` as a :math:`(5, 1)` column (so
:math:`R = BC` is the rank-one Lambertian shape; the *shipped* split
puts the :math:`1/\operatorname{norm}` in :math:`B` instead, since
:math:`C` must produce a current and :math:`B` an intensity — the
composite, and therefore this measurement, is unaffected), take
:math:`G_+`,
:math:`G_-` diagonal with entries drawn from :math:`[0.5, 2]`, and
compare :math:`C^{*}B^{*}` against :math:`G_+^{-1}R^{\mathsf T}G_-`
for each :math:`G_S` in the table. The asymmetric sizes are the
load-bearing part of the design: at :math:`|\Gamma_+| = |\Gamma_-|` an
index-pairing error would be invisible, and equal sizes are `[M]` what
**every** production quadrature happens to give — measured
:math:`|\Gamma_+| = |\Gamma_-|` on every face of
``gauss_legendre(4)`` (2/2), ``gauss_legendre(8)`` (4/4),
``product(2,4)`` (2/2), ``product(4,4)`` (4/4),
``level_symmetric(6)`` (24/24) and ``lebedev(17)`` (49/49). That
equality is an **accident, not a contract**, which is why
``_narrowed_zero_operator`` refuses to lean on it and why a factored
response — routed through :math:`S(f)` — never pairs by index at all.

Two consequences to carry:

1. ⚠ **Only the COMPOSITE is metric-free.** :math:`C^{*}` and
   :math:`B^{*}` *individually* depend on :math:`G_S`, and factoring
   exists precisely so that the factors become usable — so
   :math:`S(f)`'s metric must still be chosen deliberately. What the
   cancellation guarantees is that no admissible choice can make the
   composite wrong.
2. **The full per-face tier can never be the intermediate.**
   :math:`\Gamma(f)`'s metric :math:`|\Omega\cdot\hat n| \odot w_n`
   **vanishes** on the tangential ordinates
   (:math:`|\Omega\cdot\hat n| \le \varepsilon_{\text{tan}}`), so it is
   only semi-definite; the two half-traces are strictly positive
   precisely because they exclude those rows
   (:ref:`bc-trace-structure`). That retroactively gives the G6.1 gate
   ``test_the_half_trace_metric_is_strictly_positive`` its real reason:
   it reads as hygiene and is in fact the precondition for
   :eq:`bc-response-factored-adjoint` to hold.

**The payoff — the deferred transpose fell out, and shipped.** The
welded ``AngularAverageOperator`` reported ``is_adjointable = False``
and deferred its transpose to campaign phase **B5**. Factored at
**G6.3 step 3b** there was nothing left to defer — each link has ONE
honest transpose:
:math:`C^{\mathsf T}(s) = \cos\!w \otimes s` (the outer product, on
:meth:`~orpheus.sn.boundary.angular.PartialCurrentOperator.apply_transpose`)
and :math:`B^{\mathsf T}(\varphi) = \bigl(\sum_{\Gamma_-} \varphi\bigr)
/ \operatorname{norm}` (the sum over inflow, on
:meth:`~orpheus.sn.boundary.angular.IsotropicEmissionOperator.apply_transpose`),
so :math:`R^{\mathsf T} = C^{\mathsf T} B^{\mathsf T}` is

.. math::

   R^{\mathsf T}(\varphi) \;=\;
   \frac{\cos\!w}{\operatorname{norm}} \sum_{\Gamma_-} \varphi ,

verified `[M]` bit-exactly (``max|Rᵀφ − (cos_w/norm)·Σφ| = 0.0`` on
``product(2,4)``, ``xmax``) against the dense transpose of the chain's
own ``apply``, and :math:`\le` 1 ULP on ``gauss_legendre(8)``,
``level_symmetric(6)`` and ``lebedev(17)``. Note where the
normalisation sits: in :math:`B`, not :math:`C`, because :math:`C`
produces a *current* and :math:`B` must produce an *intensity* — which
is what leaves :math:`S(f)` carrying an honest :math:`J^+` and lets an
albedo enter as the pure scalar law :math:`J^- = \alpha J^+`.
:attr:`~orpheus.geometry.boundary.LambertianReemission.is_adjointable`
flipped to ``True`` in the same step.

.. warning::

   That one-liner is the **Euclidean transpose** :math:`R^{\mathsf T}`,
   not the Hilbert adjoint. The two differ by exactly the re-weighting
   in :eq:`bc-response-factored-adjoint`:
   :math:`R^{*} = G_+^{-1} R^{\mathsf T} G_-`, and on a trace the metric
   is **never** Euclidean (it is :math:`|\Omega\cdot\hat n| \odot w_n`
   — the ERR-067 family is what happens when a half-trace pairing drops
   it). Advertising :math:`R^{\mathsf T}` under a name that reads as
   "the adjoint" is exactly the two-``.T``-semantics ambiguity the
   welded operator declined to resolve — and note how the factoring
   *dissolved* rather than settled it: each link's ``apply_transpose``
   is unambiguous on its own, so no link ever had to choose a reading,
   and the composite's Hilbert adjoint follows from its bound spaces.
   The honest channel is therefore still ``.H``, carrying the metrics
   of the
   :class:`~orpheus.numerics.spaces.angular_trace_space.AngularFaceTraceSpace`
   the chain is bound to — measured `[M]` equal to
   :math:`G_+^{-1} R^{\mathsf T} G_-` to :math:`\le` 1 ULP, with
   :math:`\langle Rx, y\rangle_{G_-} - \langle x, R^{*}y\rangle_{G_+}`
   at :math:`\le` 1 ULP, on ``gauss_legendre(8)``, ``product(2,4)``,
   ``level_symmetric(6)`` and ``lebedev(17)`` at ``xmax``.

That is the campaign's thesis in one line: **the adjoint falls out of
well-posedness instead of being hand-rolled.**

.. _bc-deck-length-one-chain:

The deck transformation is the same chain, one link long
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-link chain above invites a taxonomy — *composed* responses
against *atomic* deck transformations — and that taxonomy is a trap. It
would make "atomic" a **kind**, and a kind needs its own code arm. It is
not a kind; it is a **degenerate length**. A boundary law is a sequence
whose first link has domain :math:`\Gamma_+(f)` and whose last has
codomain :math:`\Gamma_-(f)`, with the interior determined by whatever
the physics needs; a measure-preserving bijection has nothing to factor,
so its sequence has one link. Specular reflection is therefore built by
the same body as the Lambertian, bound to the same two spaces, and
composed by the same ``@`` — there is no second path (G6.3 step 5, issue
**#330**).

.. _bc-narrowed-involution:

The involution that isn't — a flag retired in favour of the algebra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A mirror is an involution — :math:`\sigma^2 = \mathrm{id}` is the whole
content of a symmetry plane, and ERR-044 guards it. It is therefore
natural to expect the *realized* specular operator to report itself an
involution, and until G6.3 step 5
:class:`~orpheus.numerics.operator.PermutationOperator` carried an
``is_involution`` attribute — set at construction from
``perm[perm] == np.arange(N)`` — to say so. **It could not, and the reason
is structural rather than numerical.**

The realizer does not build the full reflection table; it builds the
table **narrowed to** :math:`\Gamma_+`-local indices, because the law was
narrowed to :math:`\Gamma_+ \to \Gamma_-` in B3.4a. Whether that narrowed
array satisfies ``perm[perm] == arange`` depends on how ``to_local``'s
``searchsorted`` happens to order the locals — which is a property of the
quadrature, not of the mirror. `[M]` for **one** law, a mirror about
``x`` installed on ``xmin``:

.. list-table::
   :header-rows: 1
   :widths: 34 12 24 30

   * - quadrature
     - :math:`|\Gamma_+|`
     - full-space table
     - **narrowed** ``perm[perm] == arange``
   * - ``gauss_legendre(4)``
     - 2
     - involution
     - ``True``
   * - ``gauss_legendre(8)``
     - 4
     - involution
     - ``True``
   * - ``product(4, 4)``
     - 4
     - involution
     - ``True``
   * - ``level_symmetric(6)``
     - 24
     - involution
     - ``True``
   * - ``lebedev(17)``
     - 49
     - involution
     - ``False``

The physics does not vary with the quadrature, so the raw index answer
carries no invariant meaning — and Lebedev is only the row that makes
that *visible*. The deeper point holds at all five: the flag's documented
purpose is **self-adjointness in the unweighted inner product**, and
self-adjointness is undefined for a map between two different spaces. The
four ``True`` rows were never right; they were unfalsifiable.

Binding the operator is what converts this from a wrong value into a
**refused question**. Once the ends are :math:`\Gamma_+(f)` and
:math:`\Gamma_-(f)`, the composition :math:`P \circ P` is not an
expression at all, and ``@`` says so at construction:

.. code-block:: text

   P @ P            -> IncompatibleOperatorComposition:
                       A.domain=angular_trace[xmin:outflow]
                       B.codomain=angular_trace[xmin:inflow]
   P_xmin @ P_xmax  -> IncompatibleOperatorComposition   (cross-face)
   P @ P.inverse()  -> composes: Γ₋ → Γ₊ → Γ₋

⭐ So the flag was **retired rather than corrected**, and that is the
load-bearing choice. A refined flag would have answered ``False`` for a
bound cross-space permutation — the right value, still stored, still
obliged to answer for every future caller in every future binding state.
Asking ``P @ P`` instead replaces a value that *can* be wrong with a
composition that *cannot be formed*: the same claim, delivered by the
algebra, with no second clause to keep in step. The involution that IS
real lives one tier up, on the full-space mirror pairing
(:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`'s
derived :math:`\pi`), where domain and codomain coincide — asserted there
by the ERR-044 certification. **Two different claims had been sharing one
predicate name across two tiers**, and the fix is to make the tier
explicit rather than to tune the value.

.. note::

   The retirement had zero production consumers and two test assertions,
   both on unbound abstract permutations. Those were *rewired, not
   deleted* — from "the flag matches the index test" to "the square is the
   identity", asked of the algebra
   (``tests/numerics/test_permutation_operator.py``). The behavioural
   content survived the attribute; only the caching of it did not.

One consequence worth stating, because dropping it would have been the
quieter bug. :math:`P : \Gamma_+ \to \Gamma_-` has
:math:`P^{-1} : \Gamma_- \to \Gamma_+`, so
:meth:`~orpheus.numerics.operator.PermutationOperator.inverse` **inverts
the binding** rather than carrying or discarding it. Discarding it would
have left the return leg composable with anything while the forward leg
was fully typed — an asymmetry no gate on the forward leg could see.

.. _bc-method-realizability:

When a law realizes in a method — the three tiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A transport **method** is a *discretization of the trace* — a
projection

.. math::

   \Pi \;:\; \Gamma \longrightarrow \Gamma_h

onto whatever that method can represent on the boundary. SN keeps
per-ordinate values on the quadrature; a P1 / diffusion closure keeps
the two half-range moments :math:`(J^+, J^-)`; MoC keeps track angles;
MC keeps nothing at all (it resolves the trace by sampling).

Whether a boundary law **realizes** in a method is therefore not a
question about the law's complexity. It is the question of whether the
naturality square commutes:

.. math::
   :label: bc-realizability-square

   \Pi \circ (R\,G) \;=\; (R_h\,G_h) \circ \Pi ,

.. (vv-status rationale) Structural/definitional: the commuting square that
   DEFINES "this law realizes in this method". It is the framing statement the
   two shipped realizers' dispatch and refusals instantiate; the verifiable
   content is per-realizer (the diffusion 𝒜-table pinned law-by-law in
   ``tests/geometry/test_boundary_factors.py``, the SN narrowing pinned
   bit-identical in ``tests/sn/operators/test_b3_domain_narrowing.py``), not a
   solver claim of this equation's own.
.. vv-status: bc-realizability-square documented

i.e. *does discretizing the law's action agree with acting with the
discretized law?* Three tiers follow, and the middle one is the one
that surprises readers.

**Tier 1 — exact and faithful.** :math:`R = \alpha I` for **any**
:math:`\alpha`, not merely :math:`0` or :math:`1`. A scalar commutes
with every linear projection, so :eq:`bc-realizability-square` holds
identically, and :math:`\alpha` is *recoverable downstairs* — the
projected law still carries the number that distinguishes it. This is
the whole reason
:class:`~orpheus.diffusion.boundary_realizer.DiffusionBoundaryRealizer`
is one line: :math:`\mathcal{A} = \texttt{law.response\_kernel.amplitude}`.

**Tier 2 — exact but NOT faithful.** The square commutes — the
realization is *correct*, nothing is approximated — and the projection
nonetheless **identifies laws that differ upstairs**. At P1, specular
and Lambertian both give :math:`J^- = \alpha J^+`; diffusion simply
cannot tell them apart. The diffusion realizer's own module docstring
states it:

    *"White coincides with reflective at P1 … specular and Lambertian
    return differ only in the ANGULAR redistribution of the returned
    particles, which the half-range* :math:`\ell = 0` *moments
    integrate out — both preserve the returned current,*
    :math:`J^- = \alpha J^+`. *The distinction is real in transport (SN
    realizes them as a permutation vs a cosine-weighted average) and
    vanishes in any P1-closed method by construction."*

This is the same fact as :ref:`the rank-one theorem <bc-factor-roles>`
read from the other side. Where the response destroys directional
information, the composite :math:`R \circ G` comes out the **same
operator** for any admissible measure-preserving :math:`G` — upstairs,
in the continuum. (That is the theorem's honest conclusion; the equality
":math:`R \circ G = R`" this sentence carried until 2026-08-04 does not
type-check once the two half-traces are distinct spaces.) Tier 2 is the *method's*
version of the same collapse: the projection, not the response, is what
destroys the distinction. A reader who conflates "the realization is
exact" with "the realization is a faithful record of the law" will
mis-read the diffusion table's four identical-looking rows as a
coincidence. They are not: they are the image of four distinct laws
under a non-injective :math:`\Pi`.

**Tier 3 — not exact.** The law's action depends on structure the
method does not represent: an anisotropic response kernel below P1 is
the canonical case. Here :eq:`bc-realizability-square` genuinely fails
and no amount of care in the realizer recovers it.

So the dividing line is **scalar vs angular**, NOT trivial vs
non-trivial. An :math:`\alpha = 0.37` albedo is "non-trivial" and lands
in tier 1; a Lambertian is "simple" and lands in tier 2.

.. admonition:: Tier 2, measured — the re-emission closure is INVISIBLE to
                diffusion
   :class: tip

   Campaign phase B3.4b gave
   :class:`~orpheus.geometry.boundary.AlbedoBoundary` an explicit
   re-emission closure, which turns tier 2 from an argument into a
   measurement. All three spellings realize identically on the scalar
   trace `[M]`:

   .. code-block:: text

      AlbedoBoundary(0.4)                        -> ScaledOperator(0.4)
      AlbedoBoundary(0.4, SpecularReturn("x"))   -> ScaledOperator(0.4)
      AlbedoBoundary(0.4, IsotropicReturn(...))  -> ScaledOperator(0.4)

   Three distinct laws, one image. The closure is not *ignored* by the
   diffusion realizer — it is **annihilated by** :math:`\Pi`, exactly as
   :eq:`bc-realizability-square` permits. The same three laws are
   pairwise distinguishable under SN, where the closure selects a
   permutation, a cosine-weighted average, or a refusal.

   Read the refusal in this frame too: the SN realizer rejects the
   closure-free spelling not because :math:`\alpha\,I` is *wrong* but
   because it is **under-determined** at SN's resolution. Tier 1's
   guarantee — "a scalar commutes with every projection" — is a statement
   about laws that are *complete* upstairs. A bare :math:`\alpha` is not a
   complete angular law; it is a complete *scalar* one, and the tier
   argument applies only where its own hypothesis holds.

.. _bc-equivariance:

Equivariance — when the deck transformation has a realization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`G` is **method-independent as a geometric object** — the mirror
:math:`\Omega \mapsto \Omega - 2(\Omega\cdot\hat n)\hat n` and the
translation :math:`x \mapsto x + L` are ambient facts about the domain,
not about the solver. Its **realization** :math:`G_h` is not. A
discrete :math:`G_h` exists exactly when the discretization is
**equivariant** under :math:`g`, i.e. when :math:`\Pi` intertwines the
two:

.. math::

   \Pi \circ G \;=\; G_h \circ \Pi .

And that condition splits cleanly by **which coordinate** :math:`g`
touches:

* **Specular acts on the ANGULAR coordinate.** So :math:`G_h` exists
  only if the *quadrature* is symmetric under the reflection: there
  must be an index map :math:`\pi` with
  :math:`\Omega_{\pi(n)} = \Omega_n - 2(\Omega_n\cdot\hat n)\hat n` and
  matching weights. That is precisely what the three reflective
  invariants check —
  :meth:`~orpheus.geometry.boundary.ReflectiveBoundary.assert_geometry_map_measure_preserving`
  (**ERR-042**: the pushforward of the discrete face measure
  :math:`m_n = w_n |\mu_{a,n}|` under :math:`\pi` is that measure,
  :math:`m_{\pi(n)} = m_n`),
  :meth:`~orpheus.geometry.boundary.ReflectiveBoundary.assert_is_involutive`
  (**ERR-044**: :math:`\pi \circ \pi = \mathrm{id}`), and
  :meth:`~orpheus.geometry.boundary.ReflectiveBoundary.assert_reflection_maps_inflow_to_outflow`
  (**ERR-045**: every non-tangential ordinate's partner is
  non-tangential with the opposite sign on the law's axis). Read them
  correctly: they are **discretization-admits-the-symmetry** checks,
  not physics checks. The physics — that specular reflection is an
  isometry — is not in question; what is in question is whether *this
  quadrature* can express it. All three are failure mode **#5 (index
  error)** in the V&V taxonomy, and their independence is **measured,
  not assumed**: the GL-8 neighbour-pair table passes involution while
  redding the measure, and the identity table passes involution *and*
  measure while redding only the inflow → outflow check. All three
  fire at realization through
  :meth:`~orpheus.geometry.boundary.BoundaryTraceLaw.assert_realizable`,
  so every :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` construction
  certifies them.
* **A spatial wrap acts only on the SPATIAL coordinate.** Ordinate
  :math:`n` at face :math:`f'` feeds ordinate :math:`n` at face
  :math:`f`, untouched — which is why the wrap's
  :class:`~orpheus.geometry.boundary.PairedDeck` answers
  ``permutes_ordinates = False``, DERIVED from its motion's identity
  linear part rather than declared. **Every angular discretization is
  therefore trivially equivariant under it.** Periodic is the more
  method-agnostic of the two deck transformations, and for a sharper
  reason than "it is a trace connection": there is no angular symmetry
  requirement to fail.

.. _bc-refusal-axes:

Three independent axes of method-dependence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every realizer refusal in the codebase lies on exactly one of three
axes. They are **independent** — a method can sit anywhere on each —
and reading them as one axis ("this method is too coarse") is what kept
the taxonomy invisible. Each shipped guard now names its axis in-place.

.. list-table:: The three axes of method-dependence
   :header-rows: 1
   :widths: 20 38 42

   * - axis
     - question
     - the refusal that shows it
   * - **angular resolution**
     - can the method represent :math:`R`'s angular structure?
     - tier 3 above — an anisotropic :math:`R` below P1. (No shipped
       law is anisotropic yet; albedo's missing re-emission closure is
       the near case — see the census note under
       :ref:`bc-law-layer`.)
   * - **spatial / topological**
     - can the method's operator express **cross-face** coupling?
     - :class:`~orpheus.diffusion.boundary_realizer.DiffusionBoundaryRealizer`
       **refuses periodic** — "the one geometry P1 cannot integrate
       away into a per-face albedo :math:`J^- = \mathcal{A} J^+`". Its
       codomain is a per-face scalar, a block-diagonal object with no
       slot for a face *pair*. **Nothing to do with angle**: as above,
       a translation is trivially equivariant for every angular
       discretization. A method could resolve angle perfectly and still
       refuse this.
   * - **state-cone / sign**
     - is the value representable in the method's **state cone**?
     - :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer`
       **refuses zero-flux** — :math:`\mathcal{A} = -1` needs a
       *signed* current, and the SN state is an angular flux with
       :math:`\psi \ge 0`, which admits no negative angular inflow
       (:math:`\psi \ge 0 \Rightarrow J^\pm \ge 0`). Diffusion realizes
       the very same law without difficulty, as
       ``ScaledOperator(-1.0, IdentityOperator())``.

Note the shape of that table: on the topological axis diffusion refuses
what SN accepts, and on the state-cone axis SN refuses what diffusion
accepts. Neither method is "coarser". They are incomparable.

:math:`q` **is a fourth thing, and not on any of these axes.** It is a
**vector in** :math:`\Gamma_-`, not an operator, so the only question
it asks of a method is whether :math:`\Gamma_-` is represented at the
fidelity the source demands. Diffusion's refusal of
:class:`~orpheus.geometry.boundary.PrescribedInflow` is therefore a
**plumbing** refusal, not a representability one — :math:`\Gamma_-` on
a scalar trace is one number per face, which the trace carries
perfectly well. The realizer's guard says so in place: the refusal
disappears the day the diffusion fixed-source arm is wired (#290 P5),
**with no theory changing**. Stated explicitly so a future reader does
not mis-file it beside the two structural refusals above.


Layer 1 — trace structure
-------------------------

The trace operators :math:`\gamma_\pm` carry their domain
information on **one** typed
:class:`~orpheus.numerics.space.FunctionSpace` subclass,
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`, which stores
the whole boundary :math:`\Gamma = \partial\Omega \times S^d` once and
exposes inflow / outflow as two directional **selectors** over it:

* :meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face`
  selects :math:`\Gamma_- = \{(\mathbf{r}, \Omega) \in \partial\Omega
  \times S^2 : \Omega \cdot \hat n(\mathbf{r}) < 0\}` — the per-face
  directional half of the boundary on which the incoming angular flux
  is constrained by the law.
* :meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_indices_for_face`
  selects :math:`\Gamma_+` symmetrically — the boundary half on which
  the outgoing flux is *not* constrained by the BC but is *consumed* by
  it (as :math:`\gamma_+ \psi`).

.. note:: **One space, two selectors (Issues #205 / #201).** The
   pre-#188 design carried two separate typed spaces,
   ``InflowTraceSpace`` and ``OutflowTraceSpace``, one per direction.
   The View-G field-vocabulary refactor (#205 / #201) collapsed them
   into the single :class:`AngularTraceSpace
   <orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace>` on the observation
   that **inflow and outflow are operations on one space, not two
   spaces**: whether an :term:`ordinate` is incoming or outgoing at a face is a
   *predicate* — :math:`\mathrm{sign}(\Omega \cdot \hat n_f)` —
   evaluated against the same signed-projection data, not a property of
   the space's identity. :class:`AngularTraceSpace` stores the signed
   projection :math:`\Omega \cdot \hat n_f` once per face; the two
   ``*_indices_for_face`` methods are selectors over it (see
   :ref:`bc-trace-structure`).

The signed-projection table is what the
:class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` reads to name
each face's two half-traces: the codomain :math:`\Gamma_-` it certifies
the law against (ERR-041 / ERR-047) and — since campaign phase B3.2 —
the **domain** :math:`\Gamma_+` it restricts the law to.


.. _bc-domain-narrowing:

The trace maps as operators — the narrowed domain
-------------------------------------------------

Until campaign phase **B3.2** the trace operators :math:`\gamma_\pm`
existed on this page and in the affine form :eq:`affine-bc-form` but
had **no type in the code**. They were spelled three different ways and
typed as none of them, and the SN boundary law was consequently handed
the *whole face slot* — all ``quad.N`` ordinate rows — with the outflow
rows thrown away afterwards by a slice-write at the consumer. B3.1
minted the missing primitive and B3.2 narrowed the law onto it.

:class:`~orpheus.numerics.operator.TraceRestrictionOperator` is the
gather / scatter pair. Given a **sorted, unique** index set
:math:`S \subset \{0,\dots,N-1\}` of size :math:`m < N`:

.. math::
   :label: bc-trace-restriction-pair

   \gamma_S : \mathbb{R}^N \to \mathbb{R}^m,\quad
   (\gamma_S x)_j = x_{S(j)}
   \qquad\text{and}\qquad
   \iota_S = \gamma_S^{\mathsf T} : \mathbb{R}^m \to \mathbb{R}^N,\quad
   (\iota_S y)_i =
   \begin{cases} y_j & i = S(j) \\ 0 & i \notin S \end{cases}

.. (vv-status rationale) Definitional: the gather/scatter pair that types the
   affine form's γ±. Its DEFINING LAWS (γι = I, ιγ idempotent and symmetric,
   ι materialised against the dense γᵀ, γ₋∘ι₊ = 0, and the three-way partition
   resolving I) are pinned by the nine foundation tests opening
   ``tests/numerics/test_trace_restriction_operator.py``; this equation states
   the definition those tests verify, not a solver claim.
.. vv-status: bc-trace-restriction-pair documented

with :math:`\gamma_S \iota_S = I` on the restricted space and
:math:`\iota_S \gamma_S = P_S` the orthogonal projector onto it. It is
a **sibling of** :class:`~orpheus.numerics.operator.PermutationOperator`
and deliberately **not a subclass**: same ``np.take`` mechanism with a
non-square index array, but different algebra in kind — a permutation
is a bijection (invertible, with an algebra-closed
:meth:`~orpheus.numerics.operator.PermutationOperator.inverse`) while a
restriction is rank-deficient by construction and has a *scatter*
transpose rather than an inverse. Inheriting would promise what the
type cannot honour.

The two named instances live on the trace space, cached per face
(:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_restriction`
:math:`= \gamma_+` and
:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_restriction`
:math:`= \gamma_-`), and the SN face action is their composition with
the law:

.. math::
   :label: bc-face-action-narrowed

   B_{\rm face} \;=\; \iota_- \circ \text{law} \circ \gamma_+ ,
   \qquad
   B_{\rm face}^{\mathsf T}
   \;=\; \iota_+ \circ \text{law}^{\mathsf T} \circ \gamma_- .

.. (vv-status rationale) Structural: the composition ``_reflect_trace`` spells
   for every face. Its verifiable content is the B3.2 bit-identity gate —
   the composition reproduces the retired full-face-then-slice expression
   exactly (``np.array_equal``, against a numpy reference materialised off the
   law DESCRIPTOR, over slab-asym / slab-sym / sphere / cyl ``product(2,4)`` /
   2-D Cartesian LS4), in ``tests/sn/operators/test_b3_domain_narrowing.py``.
   Not an independent solver claim.
.. vv-status: bc-face-action-narrowed documented

**Nothing is computed and then discarded.** The rows the pre-B3.2
slice-write dropped are simply not in the operator's domain, so a
non-zero outflow emission — which would corrupt the outflow-definition
residual :math:`r_{\rm outflow}`, a quantity that carries no :math:`B`
term at all — is now **unrepresentable** rather than projected away.
That is the ``coding-elegance`` Pattern-4 form of what
:ref:`bc-extraction-design-corrections` correction 2 previously
achieved by projection.

.. _bc-narrowing-what-it-removed:

What the narrowing removed — three spellings, one pair
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every restriction-shaped expression the boundary review catalogued is a
composition of :eq:`bc-trace-restriction-pair`, not a primitive of its
own:

.. list-table::
   :header-rows: 1
   :widths: 46 24 30

   * - spelling found in the subsystem
     - is
     - status
   * - the slice-write ``out[sel] = full[sel]`` in ``_reflect_trace``
     - :math:`\iota_- \circ \gamma_-` (:math:`= P_{\rm in}`)
     - **dissolved** at B3.2 — the law's codomain *is*
       :math:`\Gamma_-`, so :math:`\iota_-` is the honest scatter, not
       a projection of a wider image
   * - ``IncomingSourceOperator``'s dense inflow-mask multiply
     - :math:`\iota_- \circ \gamma_-` (measured bit-identical to the
       slice-write)
     - **dissolved** at B3.4a — the operator was thereafter asked to
       fill :math:`|\Gamma_-|` rows directly, so nothing off
       :math:`\Gamma_-` was left to erase. That operator itself retired
       at **P3**, and the spec is now evaluated at the same
       :math:`|\Gamma_-|` shape one tier out, by
       :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_specs`
       — see :ref:`bc-affine-source-channel`
   * - ``IncomingOrdinateMaskTensor`` (**retired**)
     - :math:`I - \iota_- \circ \gamma_-`
     - taken **off the vacuum path** at B3.2 and **deleted at B3.3**.
       Of its thirteen tests, twelve asserted either the masking
       semantics the narrowing removed or a law
       :class:`~orpheus.numerics.operator.TraceRestrictionOperator`'s
       own battery already gates; only the non-aliasing claim was
       genuinely uncovered, and it migrated

and ``P_in ∘ P_out = 0`` stops being a curiosity: it is
:math:`\gamma_- \circ \iota_+ = 0`, true because two disjoint index
sets have nothing to hand each other.

.. warning::

   **Two traps the narrowing exposes, both measured.**

   1. **The index remap is** ``searchsorted``, **not** ``arange``.
      Mapping a subset of *global* rows into positions inside a
      restricted space is
      :meth:`~orpheus.numerics.spaces.angular_trace_space.AngularFaceTraceSpace.to_local`
      (owned by the half-trace SPACE since G6.5 — the embedding data,
      ``ordinate_indices``, is the space's), which needs a sorted index
      set — which is why sortedness is a **construction guard** on the
      space, not tidiness. The naive
      ``arange(sel.size)`` is right only when the subset is a *prefix*
      of the index set, and the two sites where the remap appears are
      discriminated by **different, complementary fixtures**: the
      reflective narrowing (``perm[inflow]`` into :math:`\Gamma_+`) is
      wrong on the **slab**, where the mirror reverses order —
      ``gauss_legendre(4)`` at ``xmax`` gives ``perm[inflow] = [3, 2]``
      → local ``[1, 0]`` where ``arange`` says ``[0, 1]`` — and right
      on the cylinder; the schedule split (a rows-subset into
      :math:`\Gamma_-`) is wrong in **2-D**, where the lower-half rows
      are not a prefix, and right in 1-D. **1-D coverage is not
      sufficient and neither is 2-D.**
   2. **A shape assertion cannot detect a mis-typed domain.** Measured:
      :math:`|\Gamma_+| = |\Gamma_-|` on every quadrature × face in the
      tree, so an un-narrowed endomorphism :math:`\Gamma_+ \to \Gamma_+`
      has *exactly the right output shape*. This is a textbook vv
      **Mode 12** blindness — the measured functional (shape) has the
      error class in its invariance group — and it is what let three
      un-narrowed realizer arms survive B3.2's first pass. Only the
      anti-Mode-12 leg, which interrogates the emitted operator's
      declared spaces rather than its output shape, found them.

**Four of the seven laws are narrowed today; two remain.** B3.2
narrowed the two laws SN reaches from a mesh — ``vacuum`` and
``reflective`` — and measured the remainder then as *six* realizer rows
across four law kinds. **B3.4a** took two of those kinds:

* ``white``, whose Lambertian kernel now contracts over
  :math:`\Gamma_+` and re-emits on :math:`\Gamma_-`
  (:ref:`bc-narrowing-b34a`); and
* ``prescribed_inflow``, whose rank-0 source was thereafter asked for
  :math:`|\Gamma_-|` rows directly. **P3** then collapsed that arm
  further — onto the zero morphism vacuum already returned, with the
  source itself moved to the boundary-source channel
  (:ref:`bc-affine-source-channel`), so the narrowing survives while the
  operator that carried it does not.

What remains is **four rows across two law kinds** — ``albedo`` at
*three* rows (:math:`\alpha = 0` and :math:`\alpha = 1` take fast paths
returning a bare :class:`~orpheus.numerics.operator.ZeroOperator` /
:class:`~orpheus.numerics.operator.IdentityOperator`, which are
**endomorphisms**, and :math:`\alpha \notin \{0,1\}` scales an
identity), plus ``periodic``. Both are blocked on a **design ruling**,
not on plumbing: albedo is under-determined on an angular trace
(:math:`R = \alpha\,I` is a :math:`\Gamma_+ \to \Gamma_+` endomorphism
and its :math:`G = \mathrm{id}` supplies no crossing), so
**B3.4b** must give it an explicit re-emission closure carried in
:math:`R`; periodic's :math:`G` reads the PARTNER face's
:math:`\Gamma_+`, which **B3.4c** builds (#183, #189). Measured on both:
they silently accept a :math:`\Gamma_+` input and echo it back — i.e.
:math:`\Gamma_+ \to \Gamma_+`, the wrong codomain, invisible to a shape
check (vv Mode 12 again). All four are unreachable in production — the
SN registry admits only ``{vacuum, reflective}`` — so the tree is
green; they are pinned by strict xfails, each ``--runxfail``-verified
to red for *its own* documented reason.

Two consequences to carry meanwhile. First, a narrowed law **cannot
honestly compose with an un-narrowed one**: the sum of a
:math:`\Gamma_+ \to \Gamma_-` leaf and a :math:`\Gamma_+ \to \Gamma_+`
one has no well-typed codomain. Note that this **no longer announces
itself as a raise** — since B3.4a, ``0.3·specular + 0.7·white`` is a
sum of two narrowed leaves and is simply correct, while
``0.3·specular + 0.7·albedo`` *runs* and returns
:math:`|\Gamma_-|`-shaped output that is silently wrong, because
:math:`|\Gamma_+| = |\Gamma_-|` swallows the mismatch. The Mode-12
lesson applies to the *algebra* as well as to the leaves.

Second, ``SNMethodSpace.minimal`` is now a **partial constructor**: a
quadrature alone cannot name a *face's* :math:`\Gamma_+`, so it no
longer suffices for any of the four narrowed laws — only ``albedo`` and
``periodic`` still realize from it, and precisely because they have not
yet been narrowed. Face orientation is a structural demand of
realization, not an implementation detail.


.. _bc-narrowing-b34a:

B3.4a — narrowing white and prescribed inflow, and what dissolved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two laws B3.4a narrowed are the campaign's cleanest illustration of
its thesis: **a too-wide codomain does not merely cost rows, it buys
bugs, and narrowing does not fix them — it makes them unspellable.**
Each law arrived carrying a correction for a defect that the narrowing
then removed the possibility of, so in both cases the shipped diff
*deletes* the correction rather than repairing it.

**White — the outflow classifier that had a twin.** Pre-B3.4a the
then-welded ``AngularAverageOperator`` was a
full-face endomorphism whose ``cos_w`` carried :math:`N` entries zeroed
off the outgoing hemisphere by its OWN test,
``(outward_sign * mu_n) > 0.0``. That is a **second outflow
classifier**, and it disagreed with the trace space's wherever a
quadrature carries tangential ordinates, because the trace space
classifies against ``TANGENTIAL_EPS`` while a strict ``> 0.0`` compare
does not. Measured at ``xmax``:

.. list-table:: Tangential ordinates, and where the two classifiers diverge
   :header-rows: 1
   :widths: 30 14 22 34

   * - Quadrature
     - :math:`N`
     - tangential at a face
     - rows the ``> 0.0`` test claims that :math:`\Gamma_+` does not
   * - ``gauss_legendre(8)``
     - 8
     - 0
     - 0 — which is why the twin never surfaced in slab tests
   * - ``product(2, 4)``
     - 8
     - **4**
     - **2** — the two whose :math:`\Omega\cdot\hat n` round-off is
       *positive*; the composite diverged by ``6.1e-05`` on a cylinder
   * - ``lebedev(9)``, ``lebedev(17)``
     - 38, 110
     - 12 on *every* face
     - **0 on every face** — the band's round-off falls on the
       non-positive side, so the two classifiers happen to agree
   * - ``level_symmetric(4)``, ``level_symmetric(6)``
     - 24, 48
     - 0
     - 0

Read that last column carefully: a quadrature can carry many tangential
ordinates and still expose no disagreement, because what matters is not
how many sit in the band but **which side of zero their round-off falls
on**. Measured over the whole production inventory
(``gauss_legendre`` 4/8, ``product`` 2×4 / 3×4 / 4×8, ``lebedev`` 9/17,
``level_symmetric`` 4/6) across all six face names, the disagreement
occurs **only for the** ``product`` **family, and there only on**
``xmax`` / ``xmin`` / ``ymax`` — ``ymin`` carries the same tangential
count with zero mis-admissions, because the sign flip moves the
round-off across zero. ``lebedev`` has twelve tangential ordinates per
face and mis-admits none anywhere; ``level_symmetric`` has none at all.

Two consequences worth carrying. A tangential-count audit is **not** a
sufficient screen for this bug class — only using one classifier is.
And the exposure is *face-asymmetric within a single quadrature*, so a
fixture that exercises one face of a ``product`` rule can be green
while its opposite face is wrong.

B3.4a's fix was to route the classification through the codebase's
**single** face-name :math:`\to` signed-projection primitive:
``(axis, outward_sign)`` is rendered as the face NAME and handed to
:func:`~orpheus.numerics.spaces.angular_trace_space.build_omega_dot_n`,
which classifies against ``TANGENTIAL_EPS`` exactly as the trace space
does. It landed on the welded operator's own ``from_quadrature``
constructor, and since **G6.3 step 3b** retired that operator the same
lines live one layer out, in the realizer's ``_checked_angular_average``
factory. With the domain narrowed the kernel classifies **nothing**: it
is *handed* :math:`\Gamma_+`. The twin is not fixed, it is unspellable,
and every ``cos_w`` entry is strictly positive by construction rather
than mostly zero — which is why the guard tightened from ``>= 0`` to
``> 0``, and why it is
:class:`~orpheus.sn.boundary.angular.PartialCurrentOperator`'s
constructor that today refuses a full-face weight vector by name.

**Prescribed inflow — the mask that dissolved.** An
:class:`~orpheus.geometry.boundary.InflowSourceSpec` fills whatever
block *shape* it is handed; it carries no trace knowledge. So pre-B3.4a
``IncomingSourceOperator`` emitted a
full-face block and then zeroed every non-inflow ordinate — outflow AND
tangential — to make :math:`q \in \Gamma_-` hold. With the codomain
narrowed it simply asked the spec to fill
``(|Γ₋|,) + psi_out.shape[1:]``:
the rows the mask used to zero were not in the codomain to be emitted
on, so **ERR-047 was closed by the TYPE rather than by an erasure**.
That is precisely the dissolution vacuum's projector underwent at B3.2
— the mask was never load-bearing physics, only the cost of a codomain
that was too big. Its companion, an *unmasked* fallback branch legal
only for :math:`q \equiv 0`, retired with it: post-B3.2 every
realization needs :math:`\gamma_+` too, so a method space with no face
data could not reach the operator at all and the branch was already
unreachable on the realize path.

.. note::

   **That operator no longer exists** (campaign phase **P3**,
   2026-08-05), so this paragraph is history — but the *typing* half of
   it survives verbatim one tier out. The spec is still evaluated at
   exactly ``(|Γ₋|,) + trailing``, now by
   :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_specs`
   on the way into the boundary-source channel, and ERR-047 is still
   closed by the type rather than by an erasure. What B3.4a could not
   see is that the operator was ALSO affine — it returned :math:`q`
   whatever it was handed — and that this put an affine term inside the
   linear :math:`B` block, which the missing ``BlockRole.BOUNDARY``
   stamp did **not** prevent. See :ref:`bc-affine-source-channel` for
   the collapse, the measurement, and why vacuum and prescribed inflow
   realize to the same object.

**Equivalence — and the reason two DIFFERENT effects both measure at
1 ULP.** White's operator was measured against a reconstruction of the
pre-B3.4a body over an :math:`\mathcal{O}(1)` random probe, six seeds:

.. list-table:: Old-vs-new, and which effect produces the difference
   :header-rows: 1
   :widths: 26 10 16 22 26

   * - Quadrature
     - face
     - rows the ``> 0.0`` test mis-admits
     - old vs new on an :math:`\mathcal{O}(1)` probe
     - cause
   * - ``gauss_legendre(8)``
     - ``xmax``
     - 0
     - **bit-identical**, every seed
     - — (padding with exact zeros is bit-neutral here)
   * - ``level_symmetric(6)``
     - ``xmax``
     - 0
     - **bit-identical**, every seed
     - —
   * - ``level_symmetric(6)``
     - ``ymax``
     - 0
     - ``1.11e-16`` / ``5.55e-17`` (:math:`\le` 1 ULP)
     - **reduction order only**
   * - ``lebedev(17)``
     - ``xmax``, ``ymax``
     - 0
     - ``1.11e-16`` / ``5.55e-17`` (:math:`\le` 1 ULP)
     - **reduction order only**
   * - ``product(2, 4)``
     - ``xmax``, ``ymax``
     - **2**
     - :math:`\le` 1 ULP — *and this is the trap*
     - **the classifier twin**, masquerading as noise

Two mechanically different things are being measured, and on a
well-scaled probe they are **indistinguishable**.

Where the classifiers agree (every quadrature but ``product``), the
only change is that the sum now runs over the restricted
:math:`|\Gamma_+|`-entry array instead of a zero-padded
full-:math:`N` one, so the floating-point reduction ORDER changed and
addition is not associative. That is *not* an error and *not* a
regression — it is the ``vv-principles`` **principled-equivalence**
case: a named intermediate (the cosine-weighted current over
:math:`\Gamma_+`) replacing an unnamed zero-padded one, drift bounded
by reduction depth :math:`\times` ULP.

Where the classifiers **disagree**, the change is a genuine VALUE
correction that merely *looks* like noise. At ``product(2, 4)``,
``xmax``, the four tangential ordinates carry
:math:`\Omega\cdot\hat n` of :math:`+5.0\times10^{-17}` (twice) and
:math:`-1.5\times10^{-16}` (twice) — round-off, not exact zero — so a
strict ``> 0.0`` test admits the two positive ones into
:math:`\Gamma_+` while the trace space calls all four tangential. The
resulting spurious weights are :math:`7.85\times10^{-17}` against a
normalization of :math:`2.5651`, so the **denominator is unchanged to
the last bit** (measured :math:`\Delta\text{norm} = 0.0` exactly) and
the whole discrepancy lives in the NUMERATOR — where it is
:math:`\psi`-weighted. It therefore scales with the flux carried on
the mis-admitted rows and is **not bounded by floating point**:
measured **6.1e-05** relative when those rows carry :math:`10^{12}`
times the genuine outflow flux, which is the regime the cylinder
composite hit.

.. warning::

   **Do not read the 1-ULP row for** ``product(2, 4)`` **as evidence
   of equivalence.** An :math:`\mathcal{O}(1)`-scaled probe cannot
   separate a reduction-order artefact from a mis-classified-ordinate
   bug, because the offending weight is itself :math:`\mathcal{O}(\epsilon)`
   — the error only becomes visible once the *flux ratio* between the
   tangential and outflow rows is large. This is why the twin survived
   in-tree: every equivalence measurement was taken on a well-scaled
   probe, and ``gauss_legendre`` (the slab default) carries no
   tangential ordinates at all, so it could not see the disagreement
   under any probe. The narrowing is justified structurally — one
   classifier, not two — not by the ULP table.

**And one guard was added, not removed.**
:class:`~orpheus.geometry.boundary.WhiteBoundary` declares its own
``axis`` / ``outward_sign`` while the method space independently names
the face — two encodings of ONE orientation, and until B3.4a nothing
compared them. A white law declared for ``x`` and installed on
``ymin`` averaged over the wrong hemisphere and reported nothing.
``SNBoundaryRealizer``'s ``_checked_angular_average`` now cross-checks
them, raising a :class:`~orpheus.geometry.boundary.BoundaryError` with
``law="white"`` on a mismatch. It is the same shape as the vacuum arm's
ERR-041 guard, and — critically — it compares index **SETS**, not
sizes: :math:`|\Gamma_+| = |\Gamma_-|` on every quadrature in the tree
would make a size comparison Mode-12 blind. On the canonical
:meth:`SNMesh.realize_boundary_law
<orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law>` path both
encodings derive from the same face label, so the guard is green by
construction; it bites on hand-built method spaces and on a
mis-declared law.

**Two things B3.4a deliberately did NOT do.** It did not type the
Lambertian kernel as the rank-one :math:`u \otimes v` it now visibly is
(with :math:`u = \mathbf{1}_{\Gamma_-}` and :math:`v =
\cos w / \mathrm{norm}` on :math:`\Gamma_+`) — deferred to phase
**B5** as what would make its adjoint structurally available, carrying
the Euclidean-vs-cosine-metric transpose question with it. In the event
that deferral was discharged at **G6.3 step 3b**, and by a different
mechanism: the kernel was *factored* into
:math:`B \circ C` rather than typed as :math:`u \otimes v`, which
dissolved the transpose question instead of answering it
(:ref:`bc-response-adjoint`). And it did
not give the narrowed laws a domain **validator**: white happens to
refuse a full-face input (because
:meth:`PartialCurrentOperator.apply
<orpheus.sn.boundary.angular.PartialCurrentOperator.apply>`, the
chain's first link, checks
``psi.shape[0]``), while prescribed inflow ignores its input entirely
and so has nothing to validate against. Refusal and non-endomorphism
are separate properties; see the warning at :ref:`bc-worked-example`.


.. _bc-affine-source-channel:

The affine source channel — which tier carries :math:`q`
--------------------------------------------------------

:eq:`affine-bc-form` is an **affine** map, and an affine map has two
halves. The realizer of :ref:`bc-realizer-layer` realizes the
**linear** half and only the linear half; the inhomogeneous half
travels a separate, typed channel and is added by the solve. Campaign
phase **P3** (2026-08-05) is the carve that made that sentence true of
the code as well as of the design; before it, one law realized both
halves into a single object that was declared linear and was not.

Write the law with the linear factor named:

.. math::
   :label: bc-affine-linear-factor

   \gamma_-\psi\big|_f
   \;=\; \underbrace{R\,G}_{\textstyle L_f}\;\gamma_+\psi\big|_f
   \;+\; q_f ,
   \qquad
   \operatorname{realize}(\text{law}_f) \;=\; L_f ,
   \qquad
   q_f \in \Gamma_-(f).

.. (vv-status rationale) Structural/tier-assignment: the same affine form as
   :eq:`affine-bc-form`, with the realizer's output NAMED as its linear factor
   L. It is a statement about which tier carries which half, not a solver
   claim; its verifiable content is the linearity of every realized leaf
   (``B(0) = 0``, ``B(2x) = 2B(x)``) plus the source-tier claim
   :eq:`bc-single-delivery`. Per this page's V&V-status note, its equations are
   definitional / structural-architecture statements.
.. vv-status: bc-affine-linear-factor documented

Then every law the SN realizer admits splits cleanly, and the split is
the whole content of the carve (``zero_flux``, the seventh registered
kind, has no SN ``isinstance`` arm at all and is REFUSED — a negative
angular inflow is unrepresentable, so an angular method says ``vacuum``
instead; it is realizable only on a scalar trace):

.. list-table:: Every SN-realizable law as a linear factor plus a source
   :header-rows: 1
   :widths: 26 32 20 22

   * - law
     - :math:`L = R\,G`
     - :math:`q`
     - realized leaf
   * - :class:`~orpheus.geometry.boundary.VacuumInflow`
     - :math:`0` — the zero morphism
       :math:`\Gamma_+ \to \Gamma_-`
     - :math:`0`
     - :class:`~orpheus.numerics.operator.ZeroOperator`, both spaces
       bound
   * - :class:`~orpheus.geometry.boundary.PrescribedInflow`
     - :math:`0` — **the same zero morphism**
     - :math:`\neq 0`
     - :class:`~orpheus.numerics.operator.ZeroOperator`, both spaces
       bound
   * - :class:`~orpheus.geometry.boundary.ReflectiveBoundary`
     - :math:`P` — a permutation (scaled by :math:`\alpha < 1`)
     - :math:`0`
     - ``PermutationOperator & IdentityOperator``
   * - :class:`~orpheus.geometry.boundary.WhiteBoundary` /
       diffuse :class:`~orpheus.geometry.boundary.AlbedoBoundary`
     - :math:`B \circ C` — emit :math:`\circ` contract, the factored
       Lambertian
     - :math:`0`
     - ``(IsotropicEmissionOperator @ PartialCurrentOperator)
       & IdentityOperator``
   * - specular :class:`~orpheus.geometry.boundary.AlbedoBoundary`
     - :math:`\alpha P` — the reflective body at its own
       :math:`\alpha`
     - :math:`0`
     - ``ScaledOperator(α, <that TP>)``
   * - :class:`~orpheus.geometry.boundary.PeriodicBoundary`
     - the identity on the local index, fed the **partner** face's
       :math:`\Gamma_+`
     - :math:`0`
     - ``IdentityOperator & IdentityOperator``

**Vacuum and prescribed inflow differ ONLY in** :math:`q`. That is not
a tidiness observation, it is what makes a separate affine operator
**unnecessary** rather than merely untidy: there is no linear content
to distinguish them by, so a type that existed to carry prescribed
inflow's realization was carrying an *empty* linear factor and a
source that belongs to another tier. Measured on one
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh`, one face, both laws
realized through :meth:`SNMesh.realize_boundary_law
<orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law>`:

.. math::
   :label: bc-prescribed-zero-linear-factor

   \operatorname{realize}\bigl(\texttt{PrescribedInflow}(q)\bigr)
   \;=\;
   \operatorname{realize}\bigl(\texttt{VacuumInflow}()\bigr)
   \;=\;
   0_{\,\Gamma_+(f) \to \Gamma_-(f)}
   \qquad\text{for every } q .

.. (vv-status rationale) Structural/typing identity, not a solver claim: the
   two laws realize to the SAME expression, so it is verified by object- and
   space-identity assertions, not by a value. Gated at the production tier by
   ``tests/sn/operators/test_operator_block_role.py``
   (``test_prescribed_inflow_realizes_the_same_object_vacuum_does``:
   ``type(...) is type(...)`` plus ``domain is`` / ``codomain is``) and by
   ``tests/sn/operators/test_capability_survival.py``'s re-posed capability
   rows; the ``ZeroOperator`` two-space contract itself is pinned by the
   foundation suite ``tests/numerics/test_zero_operator_spaces.py``. NOTE the
   Mode-12 hole recorded in the gotchas below: an identity row proves the two
   AGREE, not that either names the right END — see G5 of the P3 verification
   plan.
.. vv-status: bc-prescribed-zero-linear-factor documented

.. admonition:: The fixture every ``[M]`` on this page's section uses
   :class: note

   A **2-group, heterogeneous, scattering-active slab**, chosen so that
   no result below rides on a degeneracy: two regions of mixtures
   ``A`` | ``D`` from the cross-section library at ``2g``
   (:math:`c \approx 0.90`–:math:`0.96`, with a non-symmetric
   :math:`\Sigma_{s0}` so the group transfer is transpose-active), 1.0
   and 2.0 cm thick, 6 + 6 cells, vacuum ``BC`` tags on both faces,
   ``Quadrature.gauss_legendre(n_ordinates=8)`` — hence
   :math:`|\Gamma_+| = |\Gamma_-| = 4` per face and a 32-value trace
   (2 faces :math:`\times` 8 ordinates :math:`\times` 2 groups). The
   prescribed inflow is installed by constructing the law and calling
   :meth:`SNMesh.realize_boundary_law
   <orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law>` for
   the face, which is what the mesh's own resolve body does — a ``BC``
   tag cannot express this law (see the warning below). The bulk source
   is a spatially uniform isotropic
   :math:`Q = 1` through
   :meth:`AngularSourceSink.from_isotropic
   <orpheus.transport.source_sinks.AngularSourceSink.from_isotropic>`,
   identical on every leg of every comparison (gotcha 1 below explains
   why that matters), and ``inner_tol = 1e-13`` unless a row says
   otherwise. **Not** the campaign's ``placeholder_materials`` slab:
   :math:`\Sigma_s \equiv 0` there, so a delivered inflow never feeds a
   scattering source and the SI solve is a single sweep — the
   delivery-count claim survives that degeneracy but a flux-level claim
   on it would be configuration-blind.

``[M]`` on that fixture, ``xmin`` declaring
``PrescribedInflow(ConstantInflowSource(2.5))``:
both leaves are a ``ZeroOperator``; ``type(prescribed) is
type(vacuum)`` is ``True``; and on a single mesh instance
``prescribed.domain is vacuum.domain`` and ``prescribed.codomain is
vacuum.codomain`` are both ``True``, with
``domain = AngularFaceTraceSpace('angular_trace[xmin:outflow]',
shape=(4, 2))`` and ``codomain = …[xmin:inflow]…`` — i.e. exactly
:math:`\Gamma_+(f)` and :math:`\Gamma_-(f)`. The default-source
spelling ``PrescribedInflow()`` realizes to the same object as the
sourced one, which is the honest reading of ":math:`L` does not depend
on :math:`q`".

One consequence is worth stating before the mechanics, because it is
the reason the collapse is *safe*: since the realized operator no
longer distinguishes the two laws, **anything that must still
distinguish them has to read the LAW**, not the operator. Two
consumers do, and both were written that way before P3 — see
:ref:`bc-affine-channel-law-tier-consumers`.


.. _bc-affine-channel-where-q-travels:

Where :math:`q` travels instead — the composite source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The typed :math:`q` is
:class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink` —
the *eager, whole-boundary, mesh-bound* snapshot of the inflow,
packed into the unified
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`
flat layout (one flat vector of ``layout.total_size`` values, with
per-face views). It carries angular-flux units, because :math:`q` is
added to :math:`\gamma_-\psi`, not to a volumetric rate.

It is not a second, parallel source object.
:class:`~orpheus.transport.full_field.FullField` **is**
``Composite[BulkField, BoundaryField]``, so the bulk and boundary
source are **one** object

.. math::
   :label: bc-composite-source

   q \;=\; q_{\rm bulk} \,\oplus\, q_\partial ,
   \qquad
   q_{\rm bulk} \in \text{BulkField},\quad
   q_\partial \in \text{BoundaryField},

.. (vv-status rationale) Representational: names the composite the solve's RHS
   already is (the ``TimedFullField`` pair). Its verifiable content is the
   class gate on every operator-output boundary plus the RHS construction
   rows in ``tests/sn/solve/test_declared_inflow_reaches_the_rhs.py``
   (``TestTheDeclarationIsNotInert``,
   ``TestTheSourceCannotBeSpecifiedTwice``). Not a solver claim.
.. vv-status: bc-composite-source documented

and the single construction point for it is
:func:`~orpheus.sn.solver._build_fixed_source_rhs`. **Both** inner
paths consume what that helper returns — the source-iteration driver
and the Krylov driver — so there is exactly one place where a boundary
source can be attached and exactly one place where a bug in the
attachment could live. A caller may hand
:func:`~orpheus.sn.solver.solve_sn_fixed_source` a bare
``(N, ng, *spatial)`` bulk array or the full composite
:class:`~orpheus.transport.timed_full_field.TimedFullField`; either
way the helper normalises to :eq:`bc-composite-source`.

The path from a declared law to a packed snapshot is a **ladder**, each
rung delegating to the next so the packing rule is stated exactly once:

.. list-table:: The recipe → snapshot ladder on :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`
   :header-rows: 1
   :widths: 30 44 26

   * - rung
     - takes the source in the form …
     - delegates to
   * - :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_mesh_laws`
     - **the form the problem holds** — a boundary condition someone
       declared. Reads every face's ``mesh.bc[face].law`` and
       materialises the
       :class:`~orpheus.geometry.boundary.PrescribedInflow` sources
       among them; every other law contributes nothing because its
       whole content is :math:`L`
     - ``from_specs``
   * - :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_specs`
     - **a lazy per-face recipe** — a mapping ``{face:
       InflowSourceSpec}``, each evaluated at that face's
       :math:`(|\Gamma_-(f)|,) + \text{trailing}` shape
     - ``prescribed_inflow``
   * - :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
     - **known per-face arrays** — writes the **inflow** ordinate slots
       of the named faces and leaves every other slot zero (a
       prescribed source's outflow rows are physically meaningless, so
       they are unrepresentable rather than overwritten)
     - ``zeros``
   * - :meth:`AngularBoundarySourceSink.zeros(mesh.angular_trace) <orpheus.numerics.field.Field.zeros>`
     - **nothing but the trace SPACE** — the all-zero source every
       sourceless law needs, and the base every rung above allocates
       before filling. Space-keyed since CS4b S5 (it was
       ``zeros_on(mesh)`` until then); the shared
       :meth:`Field.zeros <orpheus.numerics.field.Field.zeros>` body
       allocates ``np.zeros(space.shape)`` and nothing else
     - — (the bottom)
   * - *(sibling, not a rung)*
       :meth:`~orpheus.transport.fields._bases.AngularBoundaryField.from_face_arrays`
     - **every** face's **full** slot, outflow rows included — the
       general inherited constructor for non-inflow uses; nothing in the
       ladder delegates to it, and a prescribed inflow must not use it
     - —

Only the top rung starts from a **declaration**, and that is why it is
the top rung. A driver or a test that reaches for ``prescribed_inflow``
directly is supplying by hand what the declaration should have
produced, and is therefore exercising a path no user travels — see
:ref:`verification-user-path`, whose worked example this boundary
source is.

``[M]`` the ladder is not decorative:
``from_mesh_laws(mesh)`` and ``from_specs(mesh, {face: spec})`` return
**bit-identical** arrays for a declared ``ConstantInflowSource(2.5)``
(``np.array_equal`` on the flat values), and ``from_mesh_laws`` returns
exactly the zero trace field
``AngularBoundarySourceSink.zeros(mesh.angular_trace)`` —
``q.linf == 0.0`` — for ``vacuum``,
``reflective``, ``white``, and for ``PrescribedInflow(NoSource())``.
Only ``PrescribedInflow(ConstantInflowSource(2.5))`` gives
``q.linf == 2.5``. On the documented fixture the packed :math:`q` has 8
non-zeros in 32 values: :math:`|\Gamma_-(\texttt{xmin})| = 4` inflow
ordinates :math:`\times` 2 groups, and nothing anywhere else on the
trace.

The claim the channel exists to support is the affine law **evaluated
on the answer**. For prescribed inflow :math:`L = 0`, so at
convergence

.. math::
   :label: bc-single-delivery

   \gamma_-\psi\big|_f \;=\; q_f
   \qquad\text{exactly, and once.}


.. implements:: bc-single-delivery
   :by: orpheus.geometry.boundary.prescribed_inflow.PrescribedInflow

   **Implemented by** 7 sites. Every symbol that executes this
   equation's arithmetic is declared, not only the canonical one: a
   test is adjudicated against the transcription it actually ran, so
   declaring a single site would refute the tests that exercise the
   others.

.. implements:: bc-single-delivery
   :by: orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face

.. implements:: bc-single-delivery
   :by: orpheus.sn.solver._build_fixed_source_rhs

.. implements:: bc-single-delivery
   :by: orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink

.. implements:: bc-single-delivery
   :by: orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink.from_mesh_laws

.. implements:: bc-single-delivery
   :by: orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink.from_specs

.. implements:: bc-single-delivery
   :by: orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink.prescribed_inflow

Three failure directions are distinguishable in one reading of that
equation, with no reference solver and no discretization dependence:
a doubled delivery reads :math:`2q`, a lost channel reads :math:`0`,
and a wrong linear factor reads :math:`q + L\gamma_+\psi \neq q`.

.. warning::

   ⚠ **That reading is bit-exact on source iteration and is NOT
   bit-exact on Krylov** — and the difference is a property of the
   inner solver, not of the delivery. It is the one measurement on this
   page most likely to be gated wrongly, in either direction.

   The sweep *writes* the inflow seed into :math:`\gamma_-\psi`, so on
   SI the converged trace is :math:`q` **as a copy**: ``[M]``
   ``0x1.4000000000000p+1``, :math:`\lVert \gamma_-\psi - q
   \rVert_\infty = 0.0` exactly, on every fixture below and at every
   tolerance. GMRES instead *solves* for the trace rows along with
   everything else, so the converged trace carries the iteration
   residual:

   .. list-table:: :math:`\lVert \gamma_-\psi - q \rVert_\infty` at :math:`q = 2.5`, post-P3
      :header-rows: 1
      :widths: 34 12 18 10 12 14

      * - fixture
        - inner
        - :math:`\lVert \gamma_-\psi - q \rVert_\infty`
        - ULP
        - exact?
        - :math:`/\,(q\cdot\texttt{tol})`
      * - GL-8, ``vac|vac``, ``inner_tol = 1e-13``
        - SI
        - ``0.0``
        - 0
        - **yes**
        - ``0.0``
      * - GL-8, ``vac|vac``, ``inner_tol = 1e-13``
        - Krylov
        - ``7.994e-15``
        - 18
        - no
        - ``0.032``
      * - GL-16, ``vac|vac``, ``inner_tol = 1e-13``
        - Krylov
        - ``3.109e-15``
        - 7
        - no
        - ``0.012``
      * - GL-8, ``vac|refl``, ``inner_tol = 1e-13``
        - Krylov
        - ``4.441e-16``
        - 1
        - no
        - ``0.002``
      * - GL-8, 12+12 cells, ``inner_tol = 1e-13``
        - Krylov
        - ``1.021e-14``
        - 23
        - no
        - ``0.041``
      * - GL-8, ``vac|vac``, ``inner_tol = 1e-10``
        - Krylov
        - ``1.225e-11``
        - **27 580**
        - no
        - ``0.049``

   Read the last row: loosening ``inner_tol`` by :math:`10^{3}` moves
   the deviation by :math:`10^{3}`, so it is the **iteration residual**
   and not floating-point noise — the ULP count is not a stable
   quantity, while the ratio to :math:`q \cdot \texttt{inner\_tol}` is
   (measured ``0.002``–``0.049`` across the scan). ``[M]`` the
   hand-supplied channel shows the *same* deviation to the last bit, so
   this is not an artefact of the declaration route.

   **What the gate ships, and why it is TWO legs.** A single
   ``assert_array_equal`` parameterized over both inner solvers — the
   shape P3's verification plan first specified, on the strength of a
   pre-carve measurement printed at 12 decimal places, which cannot
   resolve :math:`8\times10^{-15}` at :math:`2.5` — is **red on
   Krylov** for a reason that has nothing to do with the delivery
   count. The keystone
   ``test_the_declared_boundary_law_holds_on_the_answer`` therefore
   splits the claim:

   * **the delivery COUNT**, ``assert_allclose(rtol=0, atol=1e-9)`` on
     both legs — eleven orders of margin over readings of ``0.0`` /
     ``2.5`` / ``5.0``, so the three cases stay exactly
     distinguishable while the failure message stays readable; and
   * **the exactness**, per path: ``assert_array_equal`` on SI, and
     ``assert_array_almost_equal_nulp(nulp=64)`` on Krylov
     (:math:`\approx 3.5\times` the measured 18 ULP).

   Splitting is what keeps the two claims from eroding each other. Do
   **not** relax the SI leg to a tolerance to "match" Krylov: that
   discards the one exact reading in the whole claim. And note what the
   scan above implies about the ULP budget — a **fixed** ``nulp`` is a
   floating-point claim about a quantity that is really an *iteration
   residual*, so it is stable only at the fixture's ``inner_tol``.
   ``nulp=64`` has ample headroom at ``1e-13`` and would be exceeded by
   three orders at ``1e-10``. That is a deliberate trade: a ULP budget
   cannot be quietly loosened the way an ``rtol`` can, so if someone
   changes the fixture's tolerance the row goes red and asks to be
   re-derived, which is the intended conversation.

The pre-P3 double delivery is likewise exact on SI —
``5.000000000000``, i.e. :math:`2.5 + 2.5` in IEEE — and on Krylov it
did not produce a reading at all, because the solve raised (see
:ref:`bc-affine-channel-the-defect`). See
:ref:`bc-affine-channel-two-channels` for the *flux*-level comparison
and which assertion belongs there.

.. note::

   :eq:`bc-single-delivery` carries **no** ``vv-status`` sentinel
   because it needs none: it is a genuine L1 equation claim with a
   committed gate. ``tests/sn/solve/test_declared_inflow_reaches_the_rhs.py``'s
   ``test_the_declared_boundary_law_holds_on_the_answer`` carries
   ``@pytest.mark.verifies("bc-single-delivery")`` on both inner-solver
   parameters, plus ``@pytest.mark.catches("ERR-075")`` — the error
   catalog's entry for the double delivery described in
   :ref:`bc-affine-channel-the-defect`. Its negative control
   (``test_an_undeclared_face_has_a_zero_inflow_trace``) pins the
   complement: an all-vacuum mesh under the identical bulk source must
   read :math:`\gamma_-\psi = 0` on **both** faces, without which the
   keystone would pass against an implementation that wrote :math:`q`
   onto every inflow trace unconditionally.


.. _bc-affine-channel-the-defect:

The defect the collapse closed — an affine map in a linear slot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Until P3 the SN realizer answered ``PrescribedInflow`` with an
``IncomingSourceOperator`` whose ``apply`` **ignored its input** and
returned :math:`q`. That is an affine map declared as a
:class:`~orpheus.numerics.operator.LinearOperator`. It carried no
:attr:`~orpheus.numerics.operator.BlockRole.BOUNDARY` stamp, and the
stamp's absence was *believed* to fence it out of the :math:`B` block.

**It did not.** The property that assembles :math:`B` is
``SNBoundaryOperator._face_laws``
(:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`), and it
reads

.. code-block:: python

   return {
       face: self.sn_mesh.bc[face]
       for face in self.sn_mesh.angular_trace.layout.faces
   }

— every face in the trace layout, with **no** ``block_role`` filter.
:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator` carries the
stamp itself, applies every law it collected, and never asks a leaf
what its role is. So the affine leaf reached the block, and :math:`B`
stopped being linear.

``[M]`` in a worktree pinned at the pre-carve commit (``ef4c3537``,
which already carried the P2′ source wiring), same 2-group
heterogeneous slab, ``xmin`` declaring
``PrescribedInflow(ConstantInflowSource(2.5))``:

.. list-table:: The operator tier, before and after the collapse
   :header-rows: 1
   :widths: 42 29 29

   * - measured on :math:`B`
     - pre-P3
     - post-P3
   * - faces collected by ``_face_laws``
     - ``['xmax', 'xmin']``
     - ``['xmax', 'xmin']`` — **unchanged; there was never a filter**
   * - ``xmin`` leaf's ``block_role``
     - ``None``
     - ``BlockRole.BOUNDARY``
   * - realized ``xmin`` leaf
     - ``IncomingSourceOperator``
     - ``ZeroOperator``
   * - :math:`\lVert B(0) \rVert_\infty`, bulk block
     - ``0.0``
     - ``0.0``
   * - :math:`\lVert B(0) \rVert_\infty`, **boundary block**
     - ``2.5`` — a linear operator has :math:`B(0) = 0`
     - ``0.0``
   * - :math:`\lVert B(2x) - 2B(x) \rVert_\infty`
     - ``2.5``
     - ``0.0``
   * - ``B.is_adjointable``
     - ``False`` — the affine leaf declined a transpose, and the
       per-face intersection rule made the **whole** block
       non-adjointable
     - ``True``

Two end-to-end consequences followed, and they are different on the
two inner solvers.

**On source iteration the inflow was delivered TWICE.** :math:`q_\partial`
enters the SI right-hand side and :math:`B\psi` enters as a coupling
gain, so once P2′ wired the declared law into the source channel while
the affine operator was still in the block, both carried it. ``[M]``
on the documented fixture, comparing a declared inflow (``D``) against
the same inflow hand-supplied on an all-vacuum mesh (``C``) against a
no-inflow control (``V``):

.. list-table:: Converged inflow trace :math:`\gamma_-\psi` at ``xmin``, and the delivery ratio
   :header-rows: 1
   :widths: 34 22 22 22

   * - configuration
     - pre-P3, SI
     - pre-P3, Krylov
     - post-P3, both
   * - ``D`` — declared ``PrescribedInflow(2.5)``
     - ``5.0`` (bit-exact: ``0x1.4000000000000p+2``)
     - raises (below)
     - ``2.5``
   * - ``D0`` — declared, source channel disabled (the pre-P2′
       behaviour, reached at the time by monkeypatching
       ``from_mesh_laws`` back to the then-current mesh-keyed
       ``zeros_on``; today's spelling of the same probe is
       ``zeros(mesh.angular_trace)``)
     - ``2.5``
     - raises (below)
     - —
   * - ``C`` — all-vacuum mesh, :math:`q_\partial` supplied directly
     - ``2.5``
     - ``2.5``
     - ``2.5``
   * - ``V`` — all-vacuum, no inflow (control)
     - ``0.0``
     - ``0.0``
     - ``0.0``
   * - :math:`\lVert \varphi_D - \varphi_V \rVert_\infty /
       \lVert \varphi_C - \varphi_V \rVert_\infty`
     - ``2.0000000000``
     - —
     - ``1.0000000000``

Every entry in that table is a trace value read at the SI convergence
point, where it is bit-exact; the ``post-P3, both`` column reads
``2.5`` on SI bit-exactly and on Krylov to the iteration residual (the
warning at :eq:`bc-single-delivery` carries that scan). The ratio is
**exactly** two pre-P3 and **exactly** one post-P3, on
every fixture measured. Note what row ``D0`` says about the history:
pre-P2′ a declared inflow was *not* inert — it was delivered, through
the affine operator inside :math:`B`. P2′ then added the channel it
should have travelled all along, and the two deliveries summed. P3
removes the wrong one.

**On the Krylov path it was worse, and had been for longer.** GMRES's
residual bookkeeping rests on the Arnoldi relation
:math:`A V_k = V_{k+1} H_k`, which requires :math:`A` to be linear. An
affine :math:`A(x) = A_{\rm lin}(x) + c` breaks it, so the residual
SciPy tracks internally becomes meaningless: it reports convergence
while the iterate does not solve the equation.
:func:`~orpheus.sn.solver._certify_within_group_exit` catches exactly
that — it recomputes the honest equation residual
:math:`\lVert A\psi - q \rVert / \lVert q \rVert` after any convergence
claim and raises
:class:`~orpheus.sn.solver.ConvergenceCertificateError`:

.. code-block:: text

   solve_sn_fixed_source[krylov]: the within-group solve claimed
   convergence (running residual 0.000e+00 < tol 1.0e-13) but the
   honest equation residual is ‖Aψ − q‖/‖q‖ = 2.313e+00 — the
   iteration's fixed point does not solve the equation (the #282
   lag-death class: a stale/lagged block inside M; the free-identity
   stop is structurally blind to it).

``[M]`` pre-P3 **both** rows raise: ``D`` at a defect of ``2.313e+00``
and ``D0`` at ``2.504e+00`` on the documented fixture (P3's
verification plan measured ``1.718e+00`` on its own probe — the
magnitude is fixture-dependent, the :math:`\mathcal{O}(1)` scale is
not). So **a declared prescribed inflow combined with the Krylov inner
solver was simply UNUSABLE**, at the pre-carve commit *and* before the
source channel existed. P3 is a bug **fix**, not prophylaxis; ``[M]``
post-P3 the same configuration converges on both inner solvers and the
certificate is silent.

.. warning::

   **Only one fence was ever real, and it is the reason a green suite
   never saw any of this.** ``prescribed_inflow`` is **not a registered
   ``BC`` kind** — :class:`~orpheus.geometry.mesh.BC`'s ``params`` is
   ``dict[str, float]``, so a tag can carry ``{"albedo": 0.7}`` but
   never an inflow distribution, and the SN tag registry admits only
   ``{vacuum, reflective}``. A non-trivial prescribed inflow is
   reachable **only** by constructing the law object and installing it,
   which no production driver did. Issue **#189** tracks registering
   the kind for the constant case; it is a separate, weaker question.

   The two defects protected each other. The bypass — every existing
   non-vacuum test supplied :math:`q_\partial` by hand — is why no test
   could observe the law tier at all; and the law tier's inertness (as
   far as any driver was concerned) is why the bypass looked harmless.
   That is the :ref:`verification-user-path` doctrine's canonical
   instance: the gates were operator-tier or right-hand-side-tier, and
   never a *solve* started from a *declaration*.


.. _bc-affine-channel-two-channels:

The two channels, and what "equivalent" means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two user-visible ways to put flux into :math:`\Gamma_-`:

1. **Declare the law** — install
   :class:`~orpheus.geometry.boundary.PrescribedInflow` on a face and
   let :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_mesh_laws`
   find it. This is the user path.
2. **Supply** :math:`q_\partial` **directly** — build the source with
   :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
   and pass the composite. This is what the existing non-vacuum
   manufactured-solution coverage does, deliberately bypassing the law
   tier.

Post-P3 the two reach the same fixed point, and the *strength* of that
agreement is the sharpest single statement about what the carve did:

.. list-table:: Declared (``D``) versus hand-supplied (``C``), post-P3
   :header-rows: 1
   :widths: 34 17 15 17 17

   * - fixture
     - inner
     - :math:`\lVert \varphi_D - \varphi_C \rVert_\infty`
     - ``array_equal``
     - delivery ratio
   * - slab 2G het, ``xmin`` only
     - SI
     - ``0.000e+00``
     - ``True``
     - ``1.0000000``
   * - slab 2G het, ``xmin`` only
     - Krylov
     - ``0.000e+00``
     - ``True``
     - ``1.0000000``
   * - slab 2G het, **both** faces declaring
     - SI / Krylov
     - ``0.000e+00``
     - ``True``
     - ``1.0000000``
   * - slab 2G het, ``xmax`` reflective
     - SI / Krylov
     - ``0.000e+00``
     - ``True``
     - ``1.0000000``
   * - slab 2G het, ``gauss_legendre(16)``
     - SI / Krylov
     - ``0.000e+00``
     - ``True``
     - ``1.0000000``

**Bit-identical, on every fixture and both inner solvers — and that is
structural, not luck.** Post-P3 the two configurations are the *same
floating-point program*: the realized :math:`B` is a ``ZeroOperator``
in both (by :eq:`bc-prescribed-zero-linear-factor`), every other
operator is built from the same ``(mesh, quadrature, materials)``, and
the two right-hand sides carry bit-equal :math:`q_\partial` arrays. The
only precondition is that last one — the declared spec must *evaluate*
to the same floats the hand-built slot was filled with, which holds by
construction for a constant source compared against a constant slot,
and is exactly the property
:meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_specs`
was built to guarantee (it evaluates at the same shape the inline path
asked for).

.. warning::

   ⚠ **The pre-P3 number for this same comparison is 1.998e-13 — and
   a tolerance-based two-channel flux gate is therefore BLIND to the
   defect P3 removed.** ``[M]`` at ``ef4c3537``, rows ``D0`` versus
   ``C``: :math:`\lVert \varphi_{D0} - \varphi_C \rVert_\infty =
   1.9984\times10^{-13}` with ``array_equal = False``, against
   :math:`\lVert \varphi \rVert_\infty = 6.838442` — a relative
   :math:`2.92\times10^{-14}`, i.e. :math:`0.3 \times
   \texttt{inner\_tol}` at ``inner_tol = 1e-13``.

   That non-zero difference was never an error. It was the
   **signature** of the two channels being two different computations
   reaching the same fixed point: the delivery travelled through
   :math:`B` as an off-diagonal gain rather than through
   :math:`q_{\rm ext}`, so the reduction tree differed. Post-P3 it
   collapses to ``0.0`` because there is one computation.

   The consequence for gate design is concrete. A two-channel flux
   comparison at ``rtol = 10 × inner_tol`` — a defensible tolerance
   derived from the iteration floor — **passes** on a
   :math:`2.9\times10^{-14}` relative difference. It can see the
   :math:`2\times` double delivery and it cannot see a re-introduced
   *structural* difference between the channels. Assert
   ``np.array_equal`` on the flux for the constant-source case, and
   state the precondition; if a future non-constant spec makes the two
   :math:`q_\partial` arrays differ in their last bit, pin **that**
   (a source-tier ``array_equal`` on the two :math:`q_\partial`
   arrays) and let the flux comparison relax — do not silently swap
   an exact gate for a tolerance and call it the same claim.

**Which gate asserts what.** The distinction matters in both
directions and is easy to get backwards:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - quantity
     - assertion
     - why
   * - converged inflow trace
       :math:`\gamma_-\psi\big|_f` against the declared :math:`q_f`
       (:eq:`bc-single-delivery`) — the **delivery count**
     - ``assert_allclose(rtol=0, atol=1e-9)``, both inner solvers
     - eleven orders of margin over ``0.0`` / ``2.5`` / ``5.0``, so the
       three readings stay exactly distinguishable. This is the leg
       that IS the claim; the exactness legs below are about *how* the
       trace is reached, not *how much* arrived
   * - the same trace — the **exactness**, SI leg
     - ``assert_array_equal`` — bit-exact
     - the sweep *writes* the seed, so :math:`q` arrives as a copy;
       measured ``0.0`` deviation on every fixture and every tolerance.
       If a future moment-resolved trace makes it inexact, descend to
       ``assert_array_almost_equal_nulp`` and say why — **not** to
       ``rtol``
   * - the same trace — the **exactness**, Krylov leg
     - ``assert_array_almost_equal_nulp(nulp=64)``
     - GMRES *solves* for the trace rows, so the reading carries the
       iteration residual (18 ULP measured, and tolerance-dependent —
       see the warning above). A fixed ULP budget is chosen over an
       ``rtol`` precisely because it cannot be quietly loosened
   * - converged **flux**, declared versus hand-supplied
     - ``np.array_equal`` post-P3 (measured), with the bit-equal
       :math:`q_\partial` precondition stated
     - the two solves are the same float program; a tolerance here is
       blind to the ``1.998e-13`` structural-difference signature
       above
   * - the operator tier: :math:`B(0) = 0`,
       :math:`B(2x) = 2B(x)`, :math:`B(x{+}y) = B(x){+}B(y)` over the
       whole shipped law set
     - exact, and reachable **without a solve**
     - it catches an affine term anywhere in the block at
       *construction* time — the day the next affine operator is
       written, before it reaches a solver. It is blind to the
       delivery count (a source doubled in :math:`q_\partial` leaves
       :math:`B` perfectly linear) and blind to a wrong-but-linear
       :math:`L`

.. note::

   A **superposition** gate — "the flux is affine in the inflow
   amplitude with slope :math:`s`" — was considered for this claim and
   **rejected**. The double delivery *is* :math:`q \mapsto 2q`, which
   is still exactly affine in :math:`q`: :math:`\varphi(a) =
   \varphi(0) + a\,s` holds for every :math:`s`, including
   :math:`s = 2 s_{\rm true}`. The measured functional's invariance
   group contains the scale factor, so the gate is designed-green on
   the entire defect class at every tolerance and every refinement —
   a textbook ``vv-principles`` **Mode 12** non-catcher. Only the
   *coefficient*, pinned against an independent reference, has teeth,
   and that is what :eq:`bc-single-delivery` does directly.

   A frozen snapshot of the converged flux was rejected for a
   different reason: it asserts that production equals a recording of
   production, cannot distinguish "delivered twice" from "the spec
   changed", and would have to be re-baselined by the very carve it is
   meant to gate.


.. _bc-affine-channel-gotchas:

Five gotchas, all measured
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. The two bulk-source arms take different types, and mixing them
reads as a boundary difference.**
:func:`~orpheus.sn.solver._build_fixed_source_rhs` accepts either a
per-ordinate ``(N, ng, *spatial)`` array *or* a composite whose bulk
leaf is an
:class:`~orpheus.transport.source_sinks.AngularSourceSink`. The two are
**not** interchangeable spellings of one source. The array form is
already a per-ordinate density :math:`Q_n`, while
:meth:`AngularSourceSink.from_isotropic
<orpheus.transport.source_sinks.AngularSourceSink.from_isotropic>`
takes a *scalar* :math:`Q(\vec r, g)` and applies the
:math:`1/\sum_n w_n` projection at the producer boundary (Pattern 7).
So the same-looking pair of spellings differs by exactly the
quadrature's total weight. ``[M]`` on the fixture above,
``np.ones((N, ng, …))`` against
``AngularSourceSink.from_isotropic(np.ones((ng, …)))``:
:math:`\varphi[0,0] = 4.524718` versus :math:`2.262359`, ratio
**2.000000** exactly — which is :math:`\sum_n w_n = 2` for a
Gauss–Legendre rule on :math:`\mu \in [-1, 1]` (it would be
:math:`4\pi` for a full-sphere rule).

A factor of :math:`\sum_n w_n` in the **bulk** presents as a
disagreement of the *boundary* channel, because that is the only thing
the comparison was varying. Any two-channel comparison must therefore
supply the *identical* bulk spelling on both legs; building both legs'
source through one helper makes the trap unspellable rather than
avoided by care.

**2. Double specification is REFUSED, not resolved.** If the mesh
declares a ``PrescribedInflow`` **and** the composite
``external_source`` carries a non-zero boundary leaf, there are two
answers to one question.
:func:`~orpheus.sn.solver._build_fixed_source_rhs` raises. Neither
alternative is acceptable: adding double-counts (the exact defect this
section documents), and overriding makes one of the two inputs a silent
no-op. **Both wrong answers are quiet**, which is why the refusal is
loud. A composite whose boundary leaf is all-zero alongside a
declaration is fine — that is the normal declared-law path.

**3. The** ``BlockRole.BOUNDARY`` **stamp is honest metadata; it is
NOT a fence.** ``[M]`` and stated plainly because a reader will
otherwise assume it is load-bearing: ``_face_laws`` never filtered on
``block_role``, so the stamp's presence or absence changes **nothing**
about :math:`B`'s behaviour. Making
:func:`~orpheus.geometry.boundary.stamp_boundary_role` a no-op for
prescribed inflow reddens the direct stamp assertions and no numerical
gate — the stamp claim needs its own assertion, and a value gate
structurally cannot supply one. What the stamp *does* do is describe a
leaf's role for a reader and for the block-role marker classes; what it
never did is keep anything out of the block. Post-P3 the stamp is also
no longer an exception: every realizable law carries it, prescribed
inflow included, because the leaf is now genuinely linear.

**4. A declared prescribed inflow is REFUSED on a carrying
(1-D curvilinear) mesh — and the hand-supplied channel is not.** This
is the one place where the two channels of
:ref:`bc-affine-channel-two-channels` are *not* interchangeable, and
it is a deliberate loud deferral rather than a defect. ``[M]`` on a
2-group heterogeneous sphere:

.. code-block:: text

   declared law      : NotImplementedError
       RadialCharacteristicBoundaryOperator: the outer-face law
       PrescribedInflow (G=SelfPairedDeck, R=0.0) has no ruled corner
       action yet (white / albedo / periodic / a prescribed source at
       the off-quadrature μ = ±1 ray — loud-deferred, 2.5d
       plan-of-record).
   hand-supplied q_∂ : OK   γ₋(xmax) = [2.5]

A seed-carrying mesh closes its ray boundary at the off-quadrature
:math:`\mu = \pm 1` corner, and
:class:`~orpheus.sn.operators.boundary.RadialCharacteristicBoundaryOperator`'s
corner block is a **linear** operator on the ray alone. A prescribed
inflow's corner value is a free parameter, not a function of the
outflow ray, so a linear corner block structurally cannot carry it —
whereas the *source* side has a ruled three-arm inflow-corner law and
delivers it fine. The predicate is a **type** test on the
prescribed-inflow family, by design: ``[M]``
``_has_ruled_corner_action`` is ``False`` for both
``PrescribedInflow(ConstantInflowSource(2.5))`` **and** the default
``PrescribedInflow()`` at its zero source, and ``True`` for vacuum.
Testing the source *value* instead would quietly admit the default
spelling and then silently drop the source the day one was set.

**5. The identity row proves the two laws AGREE — not that either names
the right END.** :eq:`bc-prescribed-zero-linear-factor` is gated by
``test_prescribed_inflow_realizes_the_same_object_vacuum_does``, which
asserts ``prescribed.domain is vacuum.domain`` and
``prescribed.codomain is vacuum.codomain``. Read what that row cannot
see: if **both** leaves were bound swapped —
:math:`\text{domain} := \Gamma_-`,
:math:`\text{codomain} := \Gamma_+` — the row stays **green**, because
it compares the two operators against each other and never against the
trace. And the construction-time guard cannot catch it either:
:math:`|\Gamma_+| = |\Gamma_-|` on every quadrature :math:`\times` face
in the tree (see the warning at :ref:`bc-narrowing-what-it-removed`), so
a size check passes both ways. This is a textbook ``vv-principles``
**Mode 12** blindness — the measured functional (mutual agreement) has
the error class in its invariance group.

``[M]`` the binding is correct today: ``prescribed.domain`` **is**
``trace.outflow_space("xmin")`` and ``prescribed.codomain`` **is**
``trace.inflow_space("xmin")``, verified by object identity, and the
same holds for the default-source spelling. But that measurement is not
a gate. The gate that would close the hole is one ``is``-identity row
per bound realized law naming *which* space is *which end* — asserting
against the **trace's** ``outflow_space`` / ``inflow_space``, not
against a sibling operator. It exists for the hand-built type
(``tests/numerics/test_zero_operator_spaces.py`` carries a
``test_the_ends_are_not_swapped`` row on an intentionally *unequal*
:math:`3 \times 7` fixture, which is what makes a swap observable
there) and for the specular chain
(``tests/sn/operators/test_specular_deck_chain.py``); at the production
tier it exists for **neither** vacuum nor prescribed inflow. It is item
**G5** of P3's verification plan, and it is the reason the
``vv-status`` rationale on
:eq:`bc-prescribed-zero-linear-factor` records a known hole rather than
claiming full coverage.


.. _bc-affine-channel-law-tier-consumers:

The two law-tier consumers that still discriminate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collapsing the operator raises an obvious question: what breaks in the
consumers that needed to tell prescribed inflow apart? Nothing, and
the reason is that **both of them read the LAW, not the realized
operator** — and both said so, in their own comments, before P3 was
planned. This was a prediction to check, not a reassurance to accept;
``[M]`` it holds, and both refusals fire on the *family*, whatever
:math:`q` currently holds:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - consumer
     - what it asks
     - post-P3
   * - :class:`~orpheus.sn.operators.boundary.RadialCharacteristicBoundaryOperator`'s
       corner block, via ``_has_ruled_corner_action``
     - is the law outside the prescribed-inflow family, **and** either
       :math:`R = 0` or the composite a specular pairing? The corner
       block is linear on the off-quadrature ray, so a free-parameter
       inflow has no expressible action there
     - unchanged — ``isinstance(law, PrescribedInflow)`` disqualifies,
       loud-deferred (gotcha 4 above)
   * - :class:`~orpheus.sn.acceleration.dsa.DSALowOrderSystem`
     - is the law's low-order edge row proven? A zero response gives
       the Marshak row and an ordinate-permuting geometry gives the
       zero-net-current row; a free-parameter inflow has **no row** in
       the low-order edge system
     - unchanged — excluded by family, because admitting it on a zero
       default :math:`q` would build a Marshak row and silently drop
       the source the day one was set

Both are the same discipline in two places: **a type test on the family
is the honest predicate when the disqualifying property belongs to the
family whatever the current value is.** A response-factor test alone
would admit prescribed inflow (its :math:`R` *is* zero) — measured
while writing the DSA guard, and the reason the ``isinstance`` arm is
there. Post-P3 that discipline is what carries the discrimination the
operator tier deliberately gave up, and it is the reason the collapse
does not leak into the accelerator or the ray corner.

.. _bc-realizer-layer:

Layer 3 — the method realizer
-----------------------------

A single :class:`~orpheus.geometry.boundary.BoundaryTraceLaw`
describes the physics at the boundary but is **not** by itself
ready for consumption by a transport sweep. The conversion from
method-agnostic law to method-specific
:class:`~orpheus.numerics.operator.LinearOperator` is the job of a
:class:`~orpheus.geometry.boundary.BoundaryRealizer`. Each transport
method that has adopted the unified BC architecture (SN; diffusion
since #290 P3) ships one realizer class, owned by its method-mesh:
the mesh's ``realize_boundary_law`` arm — the per-method hook of the
:class:`~orpheus.transport.method.TransportMethod` Protocol (#290
P7b) — instantiates it directly.

The realizer's :meth:`realize` method takes the law plus a
**method space** — a method-specific container holding the
:term:`quadrature`, mesh, trace masks, and any other discretization
metadata the realizer needs — and returns a 1-arg
:class:`~orpheus.numerics.operator.LinearOperator` whose
:meth:`apply` carries the method-specific realization of the
affine BC :eq:`affine-bc-form`.

Why this third layer? Because the same affine law is realized by
**structurally different** linear operators in each transport
method:

* SN realizes vacuum as the **zero map**
  :math:`\Gamma_+ \to \Gamma_-` on that face's two half-traces (a
  :class:`~orpheus.numerics.operator.ZeroOperator` carrying both space
  hooks, so the forward emits the zero of :math:`\Gamma_-` and the
  transpose the zero of :math:`\Gamma_+` — see
  :ref:`bc-domain-narrowing`).
* Diffusion realizes vacuum as the albedo-family
  :math:`\mathcal{A} = 0`, i.e. the Marshak zero-incoming-current
  condition :math:`J^- = 0` on the scalar partial-current trace — the
  *same* law, a structurally different operator.
* MoC realizes vacuum by zeroing the entering boundary fluxes of
  every track that intersects the face.
* MC realizes vacuum by killing particle histories at the face.
* CP realizes vacuum as zero rows in the boundary-to-region
  coupling matrix.

Splitting the realizer out of the law makes each piece independently
testable and gives every method a single bolt-in point for its BC
treatment — see :ref:`bc-cross-method-stubs` for how a future method
adopts the architecture (and for the Wave-5 stub scaffolding that was
dissolved at #290 P7b).


.. _bc-extraction:

Boundary-condition extraction — :math:`B` as a sibling operator
===============================================================

The composite metric-correct G-adjoint that closes this extraction
narrative is documented at :ref:`g-adjoint` in
:doc:`/theory/foundations/operator_adjoint`.

Wave O step O.4a.2 (`Issue #208
<https://github.com/deOliveira-R/ORPHEUS/issues/208>`_, three commits
``d7e1316`` / ``4c0ff96`` / ``2bdc66d``, 2026-06-03) made the realized
boundary law :math:`B` a **first-class sibling** of the streaming +
collision operator :math:`L + C`. The canonical SN transport algebra
became

.. (vv-status rationale) Structural framing of the post-extraction SN
   loss operator. The verifiable claim — that the matvec/SI driver
   path with the realized ``B`` folded in agrees with the analytical
   infinite-medium balance and the homogeneous closed-form :math:`k_\infty`
   — is verified by the reflective convergence-equivalence gates
   catalogued below, not by this label directly.
.. vv-status: bc-extraction-loss-operator documented

.. math::
   :label: bc-extraction-loss-operator

   (L_{\rm full} + C - S - F - B)\,\psi \;=\; q,

acting on the **direct-sum transport state**

.. math::
   :label: bc-extraction-direct-sum-state

   V \;=\; V_{\rm bulk} \;\oplus\; V_{\rm inflow} \;\oplus\;
           V_{\rm outflow},

where :math:`V_{\rm bulk}` is the cell-centre angular flux
(:class:`~orpheus.transport.fields.angular_flux.AngularFlux`) and
:math:`V_{\rm inflow} \oplus V_{\rm outflow}` is the boundary trace
(:class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`),
partitioned per face by the sign of :math:`\Omega\cdot\hat n` into the
inflow (:math:`\Omega\cdot\hat n < 0`) and outflow
(:math:`\Omega\cdot\hat n > 0`) ordinate slots (the
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` selectors
:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face`
/ :meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_indices_for_face`,
single source of truth — see :ref:`trace-spaces-doc`).

.. vv-status: bc-extraction-direct-sum-state documented

This is the realisation, for the boundary block, of the Wave T
prediction (:ref:`tensor-network-decomposition`): "Wave O typing must accept
non-SOTP summands." :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`
is the :math:`A_{ss}` leaf — a bespoke
:class:`~orpheus.numerics.operator.LinearOperator` carrying
:attr:`~orpheus.numerics.operator.BlockRole.BOUNDARY`, NOT a
:class:`~orpheus.numerics.operator.TensorProductOperator`.


The block matrix
----------------

On :math:`V = V_{\rm bulk} \oplus V_{\rm boundary}` the operator families
occupy disjoint blocks of the :math:`2\times 2` block matrix, grouped by
the three :class:`~orpheus.numerics.operator.BlockRole` values
(:math:`b` = bulk row/column, :math:`s` = surface/trace row/column):

.. math::
   :label: bc-extraction-block-matrix

   \underbrace{
   \begin{bmatrix} A_{bb} & 0 \\ 0 & 0 \end{bmatrix}
   }_{C,\,S,\,N_{2n},\,F\;(\text{BULK})}
   \;+\;
   \underbrace{
   \begin{bmatrix} A_{bb} & A_{bs} \\ A_{sb} & A_{ss} \end{bmatrix}
   }_{L_{\rm full}\;(\text{FULL})}
   \;-\;
   \underbrace{
   \begin{bmatrix} 0 & 0 \\ 0 & A_{ss} \end{bmatrix}
   }_{B\;(\text{BOUNDARY})}

.. vv-status: bc-extraction-block-matrix documented

The position :math:`A_{ss}` is occupied by **both** :math:`L_{\rm full}`
and :math:`B` — with complementary triangle structure on the trace
splitting :math:`V_{\rm boundary} = V_{\rm inflow} \oplus V_{\rm outflow}`:

.. math::
   :label: bc-extraction-trace-blocks

   \underbrace{
   \begin{bmatrix} I & 0 \\ -T & I \end{bmatrix}
   }_{A_{ss}\ \text{of}\ L_{\rm full}}
   \qquad\text{vs.}\qquad
   \underbrace{
   \begin{bmatrix} 0 & R \\ 0 & 0 \end{bmatrix}
   }_{A_{ss}\ \text{of}\ B}
   \qquad\text{on}\quad
   \begin{bmatrix} \psi.\text{inflow} \\ \psi.\text{outflow} \end{bmatrix}.

.. vv-status: bc-extraction-trace-blocks documented

:math:`L_{\rm full}`'s trace–trace block is **unit-lower-triangular**,
with the identity on BOTH diagonal sub-blocks: the inflow row is the
carried identity :math:`I\cdot\psi.\text{inflow}` (it reads nothing
else), and the outflow row is the self-consistency defect's
stored-unknown identity :math:`I\cdot\psi.\text{outflow}` (design
correction 1 below; the per-row sign is free because
:math:`q.\text{outflow} \equiv 0` — the 2-D path spells the outflow
row with the opposite sign, same diagonal-bearing structure).
:math:`T` is the closure's **direct inflow→outflow transmission**: the
strictly-sub-diagonal coefficient the sweep chain hands the outflow
face in terms of the inflow face that seeded it — nonzero whenever a
direction's chain runs face-to-face (e.g. diamond difference on a
slab), zero when the chain terminates on the :math:`r = 0`
pole-regularity seed instead of a face, or under a pure-upwind
closure whose faces read cells only.

That identity diagonal is **load-bearing twice over** (Issue #298):

* It is what makes every trace row of :math:`L_{\rm full} + C`
  **diagonal-bearing**, so the augmented within-group operator is
  block lower-triangular in the ordering inflow → bulk → outflow —
  and fully triangular once the bulk is ordered by the sweep
  schedule. Its direct solve IS the sweep, literally **forward
  substitution**: read the given inflow, sweep the bulk, define the
  outflow.
* It is the diagonal the sibling :math:`-B` leans on: :math:`B`'s
  :math:`A_{ss}` (:math:`R` the realized per-face law,
  :ref:`bc-law-layer`) sits **strictly upper** on the same ordering —
  the inflow row reading the outflow column — the ONE up-edge that
  closes the boundary cycle. Vacuum kills it (:math:`B = 0`, pure
  forward substitution); SI/Krylov iterates it
  (:eq:`bc-extraction-two-residuals`); and :math:`B`'s low rank is
  what lets a Woodbury closure solve it in closed form instead
  (Issue #300).

(The block matrix above wrote :math:`L_{\rm full}`'s :math:`(s,s)`
entry as :math:`0` until Issue #298 — silently contradicting the role
table below, whose inflow-identity and outflow-defect rows are both
:math:`(s,s)` content. The :math:`0` was never load-bearing — no
downstream derivation consumed it — but the root-page triangularity
argument was about to.)

The three operator roles (Wave O block-role typing, Issue #208) read
off the block structure directly:

.. list-table:: Block occupancy by operator role
   :header-rows: 1
   :widths: 16 16 30 38

   * - Operator(s)
     - Role
     - Reads / writes
     - Block content
   * - :math:`L_{\rm full}`
       (:class:`~orpheus.sn.operators.streaming.StreamingOperator`,
       via :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` ``L+C``)
     - ``FULL``
     - Reads :math:`\psi.\text{bulk}` **and** the *given*
       :math:`\psi.\text{inflow}` trace; writes :math:`\psi.\text{bulk}`
       and the :math:`\psi.\text{outflow}` trace.
     - :math:`A_{bb}` (streaming) + :math:`A_{bs}` (inflow seeds the
       sweep) + :math:`A_{sb}` (sweep produces outflow) +
       :math:`A_{ss}` (the trace rows' unit-triangular structure,
       :eq:`bc-extraction-trace-blocks`). The
       **outflow row keeps the self-consistency defect**
       :math:`\psi.\text{outflow} - \text{streamed}`; the **inflow
       row carries the identity** :math:`I\cdot\psi.\text{inflow}`.
       **No BC logic.**
   * - :math:`C, S, F`
       (the collision multiplier :math:`C = M[\sigma_t]`
       — :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`,
       :class:`~orpheus.transport.operators.scattering.ScatteringOperator`,
       :class:`~orpheus.transport.operators.fission.FissionOperator`)
     - ``BULK``
     - Bulk → bulk only.
     - :math:`A_{bb}` only; the boundary block is zero.
   * - :math:`B`
       (:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`)
     - ``BOUNDARY``
     - Maps :math:`V_{\rm outflow} \to V_{\rm inflow}` via the
       realized per-face law :math:`\psi.\text{inflow} =
       B\,\psi.\text{outflow}`.
     - :math:`A_{ss}` only — strictly upper on the trace splitting
       (:eq:`bc-extraction-trace-blocks`); emits on the **inflow row
       ONLY** (see design correction 2 below).

The outer Krylov / SI loop drives **two residuals to zero**
simultaneously:

.. math::
   :label: bc-extraction-two-residuals

   \begin{aligned}
   r_{\rm inflow} &\;=\; \psi.\text{inflow}
                          \;-\; B\,\psi.\text{outflow}
                          \;-\; q.\text{inflow}
                    \;\longrightarrow\; 0
                    \quad (\text{boundary consistency}), \\
   r_{\rm outflow} &\;=\; \psi.\text{outflow}
                           \;-\; \text{streamed}(\psi.\text{bulk},
                           \psi.\text{inflow})
                    \;\longrightarrow\; 0
                    \quad (\text{outflow definition}).
   \end{aligned}

.. vv-status: bc-extraction-two-residuals documented

For vacuum :math:`B = 0` (no inflow); for reflective/white/albedo
:math:`B` is the realized :math:`R\,G` reflection that the
pre-extraction sweep applied at the boundary edge.


The deleted keystone
--------------------

Before O.4a.2, the streaming sweep re-applied the boundary law
**inside one matvec call**: the backward (inward) sweep seeded its
boundary-face inflow from the forward sweep's own reflected outflow,

.. code-block:: python

   # PRE-O.4a.2 — the "keystone" (operator.py _compute_LpC, now DELETED):
   outflow_at_boundary = _sweep_direction(+1, pole_face_seed)  # forward
   inflow_full = bc_outer.apply(outflow_at_boundary.T)         # ← KEYSTONE
   outflow_at_inner = _sweep_direction(-1, inflow_full[...])   # backward

This single line **coupled bulk ↔ boundary within the matvec**: the
operator :math:`L` secretly contained the boundary reflection, so
:math:`L` was not a pure streaming operator and :math:`B` had no
independent existence. O.4a.2 **deletes the keystone**: the backward
sweep now reads the *given* outer inflow trace
(:math:`\text{face\_outer}`'s :math:`\mu < 0` ordinates) directly, so
one matvec call computes a pure-streaming residual with no BC
reflection. The reflective coupling moves entirely to the sibling
:math:`-B`:

.. code-block:: python

   # POST-O.4a.2 — bare, keystone-free (operator.py _compute_LpC):
   outflow_at_boundary = _sweep_direction(+1, pole_face_seed)  # forward
   outflow_at_inner   = _sweep_direction(-1, face_outer)       # GIVEN inflow

The curvilinear pole seed (``psi_view[:, :, 0, 0]``) survives the
deletion because it is the **r = 0 regularity condition**, NOT a
boundary condition — it reads the innermost cell flux, a geometric
case-split on ``curvature != "cartesian"``, never a ``bc.apply``.


.. _bc-extraction-design-corrections:

Three design corrections (what was tried and corrected)
-------------------------------------------------------

The extraction surfaced three subtle traps. All three are preserved
here per Cardinal Rule 3 so a future session re-deriving the block
matrix does not re-make them.

**1. Keep the outflow defect — NOT the raw outflow.**
The in-flight plan prose said :math:`L_{\rm full}` should emit the
*raw* streamed outflow on the outflow row. This is **wrong**.
:math:`\psi.\text{outflow}` is a *stored unknown* that the sibling
:math:`-B` reads as its input (:math:`B\,\psi.\text{outflow}`).
Emitting the raw outflow would make the outflow row
:math:`-\,\text{streamed}` (an off-diagonal-only row with no diagonal
on :math:`\psi.\text{outflow}`), which **singularises** that row: the
:math:`A_{ss}` outflow-column diagonal disappears and :math:`-B` is no
longer a well-posed sibling. The fix is to keep the row as the
self-consistency defect :math:`\psi.\text{outflow} - \text{streamed}`
— the identity :math:`I\cdot\psi.\text{outflow}` diagonal stays on the
outflow row, and the outflow-definition residual
:math:`r_{\rm outflow}` of :eq:`bc-extraction-two-residuals` is the
quantity the outer loop drives to zero. Keeping the row as
``computed − stored`` also makes the vacuum path **bit-identical** to
the pre-extraction matvec (the per-row sign is free because
:math:`q.\text{outflow} \equiv 0` — the outflow trace is a pure
definition with no source).

**2.** :math:`B` **must not emit on the outflow row** — first solved by
projection, since B3.2 solved by **typing**.
At the time of the extraction the realized per-face law was a
**full-face operator**: a specular
:class:`~orpheus.numerics.operator.PermutationOperator` for reflective,
the then-welded ``AngularAverageOperator`` for
white. Its permutation mapped the input's *inflow* slots onto the
*output's outflow* slots (a spurious :math:`R\cdot\psi.\text{inflow}`),
because the permutation was defined on the whole face, not just the
:math:`A_{ss}` :math:`V_{\rm outflow} \to V_{\rm inflow}` map. In the
legacy sweep this was harmless — the sweep only ever read the
inflow slots of ``bc.apply(face)``, discarding the outflow output.
But as a sibling :math:`-B` on the direct-sum state, a non-zero
outflow emission corrupts the outflow-definition residual
:math:`r_{\rm outflow}` (which must carry **no** :math:`B` term).
*Empirically confirmed before the fix*: the outflow slots carried
nonzero :math:`R\cdot\psi.\text{inflow}`.

The extraction's fix was an **output projection** —
``_reflect_trace`` wrote only the ``inflow_indices_for_face`` slots of
the law's full-face image. Campaign phase **B3.2** replaced that with
the honest domain: the law is typed :math:`\Gamma_+ \to \Gamma_-`, so
the composition is :math:`\iota_-\circ\text{law}\circ\gamma_+` and the
spurious emission is **unrepresentable** rather than projected away
(:ref:`bc-domain-narrowing`, :eq:`bc-face-action-narrowed`). The
diagnosis in this correction is what B3.2 acted on; only the remedy
changed, from a mask at the consumer to a domain at the producer.

.. warning::

   **The transpose scatters over** :math:`\Gamma_+` **— never over**
   :math:`\Gamma_-`. This trap **survives the narrowing** and is the
   one piece of correction 2 that is still live discipline rather than
   history. Because :math:`(\iota_-\circ\text{law}\circ\gamma_+)^{\mathsf T}
   = \iota_+\circ\text{law}^{\mathsf T}\circ\gamma_-`, the transpose's
   *input* is masked to the forward's codomain :math:`\Gamma_-` and its
   *output* lands on :math:`\Gamma_+`. Scattering that output over
   :math:`\Gamma_-` instead — "projecting :math:`\text{law}^{\mathsf T}`
   onto the law's own codomain", which reads like the natural mirror of
   the forward — extracts the operator's **diagonal block**. For vacuum
   that spells a spurious :math:`+1` where the forward is the zero map.
   It was caught **only** by the A2a grid-reciprocity arm on the
   heterogeneous-**vacuum** sphere: off-diagonal permutation laws are
   bit-identical under either spelling, so every reflective fixture
   stayed green over the wrong one. (This Euclidean ``apply_transpose``
   is the un-weighted shadow of the metric-correct Hilbert adjoint
   ``B.H`` under :math:`|\Omega\cdot\hat n|\,w`; the two are separate —
   see :ref:`g-adjoint`.)

**3. The bare sweep seeds inflow from** :math:`\text{rhs.boundary}`,
**not** :math:`\text{initial\_guess.boundary}`.
Under the extraction the WDD sweep ``(L+C).solve`` is **bare** (see
:ref:`bare-sweep-extraction` in
:doc:`/theory/methods/sn/curvilinear_one_group`): it reads
the seeded inflow trace directly instead of re-applying ``bc``.
:meth:`StreamingCollisionOperator._solve_timed_full_field <orpheus.sn.operators.streaming.StreamingCollisionOperator._solve_timed_full_field>`
must therefore seed the sweep's boundary buffer from
:math:`\text{rhs.boundary}` — the *boundary source*
:math:`q.\text{boundary} + B\,\psi.\text{outflow}` — **not** from the
iterate ``initial_guess.boundary`` (the retired partner-flux carrier).
The ``initial_guess`` still threads the bulk Carlson warm start;
only the boundary seed moved.


.. _bc-extraction-variadic-driver:

The honest :math:`L+C-S-N_{2n}-B` driver via variadic couplings
---------------------------------------------------------------

The within-group inner solve no longer hands the drivers a fixed
:math:`(A, S, F)` operator *triple*. Wave O step O.2a generalised both
:class:`~orpheus.numerics.iteration.SourceIteration` and
:class:`~orpheus.numerics.iteration.KrylovAcceleration` to the
**variadic** shape :math:`\text{Driver}(A_{\rm resolvent},\,*\text{gains})`:
one invertible resolvent :math:`A` plus a homogeneous bag of lagged
coupling operators :math:`g_i`. The two consume the gains identically —

.. math::
   :label: bc-extraction-variadic-matvec

   \text{matvec} \;=\; A.\text{apply} \;-\; \sum_i g_i.\text{apply}
   \,,\qquad
   \text{rhs} \;=\; q_{\rm ext} \;+\; \sum_i g_i.\text{apply}\,.

.. vv-status: bc-extraction-variadic-matvec documented

The driver is now **problem-type-agnostic**: it sees only the resolvent
operator and a bag of operators it must lag. (Since #226 taxonomy step 3,
:class:`~orpheus.numerics.iteration.SourceIteration` receives that
resolvent **already inverted** — the pre-built inverse operator it
*applies* — while :class:`~orpheus.numerics.iteration.KrylovAcceleration`
keeps the *forward* resolvent for its GMRES matvec; the
:eq:`bc-extraction-variadic-matvec` form above is the Krylov matvec, and
the rhs term is the shared source assembly. See
:ref:`inverse-application-driver`.) *Which* leaves are gains is a
**posing-layer** decision, not an iteration-layer one (see
:ref:`eigenvalue-posing`) — the gains are exactly the posing's coupling
terms.

For the SN **k-eigenvalue** within-group inner the posing's couplings
are the bulk scattering :math:`S` and the boundary reflection
:math:`B`; the fission :math:`F` is zero within-group (it enters as the
external source :math:`q_{\rm ext}` per the eigenvalue
outer / within-group split, Lewis & Miller §6.4). So the within-group
loss decomposition is the honest

.. math::
   :label: bc-extraction-within-group-decomposition

   (L+C,\; S,\; N_{2n},\; B)
   \quad\Longrightarrow\quad
   \underbrace{(L+C).\text{apply} - S.\text{apply} - N_{2n}.\text{apply}
               - B.\text{apply}}_{\equiv\,(L+C-S-N_{2n}-B)\,\psi}
   \,,\qquad
   \text{rhs} = q_{\rm ext} + S.\text{apply}(\psi)
              + N_{2n}.\text{apply}(\psi) + B.\text{apply}(\psi)

.. vv-status: bc-extraction-within-group-decomposition documented

assembled once by the single-source-of-truth builder
:func:`~orpheus.sn.coupled_system.build_within_group_system`, which returns
the frozen :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record —
the loss grid together with its **named splitting**
:math:`A = M - N` (Hackbusch 2016 §11 — block partitionings; a
*splitting*, **not** a *regular* splitting in Varga's sense, so the
comparison theorem does **not** bound the boundary Gauss-Seidel rate
against Jacobi's — :ref:`sn-boundary-gs-not-regular`). On a seedless
(slab / cylinder /
Cartesian) mesh the record degrades to exactly this triple: its ``implicit_operator``
is :math:`M = (L+C)` — the invertible resolvent
(:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`, ``.solve`` = the WDD
sweep) — and its ``explicit_gains`` are :math:`N = (S,\ N_{2n},\ B_a)`,
the three lagged
couplings the driver applies: the bulk scattering gain
(:class:`~orpheus.transport.operators.scattering.ScatteringOperator`,
:attr:`block_role <orpheus.numerics.operator.BlockRole>` ``BULK``), the
bulk :math:`(n,2n)` emission gain
(:class:`~orpheus.transport.operators.n2n.N2NOperator`, ``block_role``
``BULK`` — a third member since CS4c step 3;
:ref:`the two collision gains <operator-algebra-two-gains>`) and the
boundary reflection gain
(:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`,
``block_role`` ``BOUNDARY``). On a *carrying* sphere the record poses the
ψ½ System B as the second row of a coupled :math:`M - N` block grid — the
full 2×2 coupled block operator documented in
:ref:`coupled-block-operator` (its starting-direction physics in
:ref:`sn-direct-seed-solve` in
:doc:`/theory/methods/sn/curvilinear_one_group`);
the former ``_within_group_triple`` / ``_lagged_gains`` construction pair
retired into this one builder at B.2d. The :math:`B\,\psi.\text{outflow}`
term lands on :math:`\text{rhs.boundary}`, which the bare ``(L+C).solve``
sweep reads as the inflow seed (:ref:`bare-sweep-extraction` in
:doc:`/theory/methods/sn/curvilinear_one_group`).

**This retires the transitional** :math:`S + B` **fold.** The
predecessor packed the boundary reflection into the *middle slot* of
the fixed triple by returning a summed operator
:math:`S + B` — the now-deleted ``SNSolver._scattering_with_boundary_op``
property. The honest composition keeps :math:`S` and :math:`B` as two
**separate first-class gains**.

**Why variadic — the fixed triple encoded a false posing distinction.**
:math:`S`, :math:`F` and :math:`B` are *homogeneous* in the driver:
each is subtracted in the matvec and summed in the rhs, exactly as
:eq:`bc-extraction-variadic-matvec` shows. The fixed :math:`(A, S, F)`
triple gave :math:`S` and :math:`F` named slots the *resolvent layer*
never uses — it was encoding a posing-layer role assignment (which
operator is loss, which is the eigen-operator) at the iteration layer,
where it does not belong. Collapsing the triple to a homogeneous
:math:`*\text{gains}` bag moves the role distinction back to the
posing layer (its proper home) and lets a fourth gain (a future
:math:`B`-trace term, an :math:`\alpha`-time term) slot in as a data
addition rather than a new named slot. Existing positional
:math:`(A, S, F)` callers stay source-compatible — ``gains = (S, F)``.

**Why** :math:`B` **is a SEPARATE gain, not folded into** :math:`S`.
Two structural reasons forbid the old fold:

#. **The adjoint metric lives on the trace.** :math:`B` lives on the
   boundary trace (:attr:`domain <orpheus.sn.operators.boundary.SNBoundaryOperator.domain>`
   ``= sn_mesh.angular_trace``), and the cosine-weighted
   :math:`|\Omega\cdot\hat n|\,w` adjoint metric (Wave O step O.2 — the
   codomain inner product of :math:`L`'s boundary-trace block) lives on
   that **trace** domain, not the bulk. Folding :math:`B` into the
   bulk :math:`S` would erase the trace typing the future adjoint
   ``.H`` needs.
#. :math:`B` **cannot join the** :math:`L+C` **preconditioner.** A
   generic :class:`~orpheus.numerics.operator.OperatorSum` carries **no**
   direct-sweep ``solve`` — its inverse action is the *iterative*
   :class:`~orpheus.numerics.green_operator.GreenOperator` splitting
   (:ref:`green-operator`), not the ``(L+C)``
   :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` subclass's
   :math:`O(N\cdot N_{\rm cells})` forward-substitution sweep. So folding
   :math:`B` into an :math:`L + C - B` sum would **demote** the cheap
   direct sweep the SI step and the Krylov preconditioner depend on to an
   iterated solve. :math:`B` must stay a *gain* (lagged, applied) — never
   a summand of the resolvent.

The old fold type-checked at the time **only because**
:attr:`ScatteringOperator.domain` *was* ``None`` (the pre-W-D bulk
operators inherited the
:class:`~orpheus.numerics.operator.LinearOperator` default). The
:class:`~orpheus.numerics.operator.OperatorSum` domain-compatibility
check fires only when both operands declare non-``None`` domains that
differ; with :math:`S` untagged the check skipped, so the
trace-typed :math:`B` summed silently with the bulk :math:`S`. The
structural reason the fold stays gone is the **variadic-driver
redesign** below — :math:`B` is a lagged *gain*, never a resolvent
summand — not a typing tripwire: P4.5 W-D gave :math:`S` the
**composite** :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
(the *same* instance :math:`L`/:math:`C`/:math:`B` carry, so the
within-group ``(L+C) - S`` guard *validates* the ``- S`` arm), NOT a
bulk-only :math:`V_{\rm bulk}` domain, so an ``OperatorSum`` of the
composite-typed :math:`S` and the composite-typed :math:`B` would
still compose — the once-envisioned "bulk-S ≠ trace-B" rejection seam
(see :ref:`bc-extraction-scope-future`) was **not** the shape W-D
landed.

.. note::

   The drivers' :class:`~orpheus.numerics.iteration.KrylovAcceleration`
   matvec :eq:`bc-extraction-variadic-matvec` and the
   :class:`~orpheus.numerics.iteration.SourceIteration` rhs are now the
   *honest* :math:`(L+C-S-N_{2n}-B)\,\psi` and
   :math:`q_{\rm ext}+S\psi+N_{2n}\psi+B\psi`
   — the reassociation :math:`A-(S+B)\to(A-S)-B` is documented as a
   **principled-equivalence** change in
   :ref:`bc-extraction-numerical-evidence` (criterion 3 of the
   ``vv-principles`` bit-identity-vs-principled-equivalence gate), not
   a bug.


.. _bc-extraction-two-routes:

The two :math:`-B` delivery routes
----------------------------------

The same :math:`-B` coupling reaches the sweep two ways, both calling
the **identical**
:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator` (single
source of truth, Cardinal Rule 2):

.. list-table:: The two delivery routes for :math:`-B`
   :header-rows: 1
   :widths: 22 40 38

   * - Route
     - Mechanism
     - Used by
   * - **Variadic gain**
       (:func:`~orpheus.sn.coupled_system.build_within_group_system`
       returns :math:`B` as a gain)
     - :math:`B` is one of the ``*gains`` the variadic driver lags:
       the matvec subtracts :math:`B.\text{apply}`, the SI rhs adds it
       (:eq:`bc-extraction-variadic-matvec`).
       :math:`B\,\psi.\text{outflow}` lands on
       :math:`\text{rhs.boundary}`, which the bare ``(L+C).solve``
       sweep reads as the inflow seed.
     - The eigenvalue SI inner driver
       (:meth:`SNSolver._solve_source_iteration <orpheus.sn.solver.SNSolver._solve_source_iteration>`),
       the eigenvalue Krylov inner
       (:meth:`SNSolver._solve_krylov <orpheus.sn.solver.SNSolver._solve_krylov>`),
       and both fixed-source paths
       (:func:`_solve_fixed_source_si <orpheus.sn.solver._solve_fixed_source_si>` /
       :func:`_solve_fixed_source_krylov <orpheus.sn.solver._solve_fixed_source_krylov>`)
       — every solve that routes through a driver.
   * - **Masked additive reflect**
       (:meth:`SNMaskedBoundaryOperator.reflect_rows_inplace
       <orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace>`,
       on the mask's rows)
     - **Accumulates** :math:`B\,\psi.\text{outflow}` onto the mask's
       inflow rows, in place on the boundary buffer
       (``bf[f][rows] += (B·bf)[f][rows]``), through the same
       :class:`SNBoundaryOperator` the mask wraps.  Additive, **not** a
       whole-face assignment — the row it completes is the
       inhomogeneous forward-substitution row :math:`z_{\rm in} =
       y_{\rm row} + (Bz)_{\rm row}`, whose seed :math:`y_{\rm row}`
       an overwrite would drop (:ref:`the rejected whole-face
       overwrite <gs-whole-face-overwrite-rejected>`).
     - The octant-group Gauss–Seidel forward substitution: the reified
       :math:`M = (L+C-B_{\rm lower})` passes
       ``reflect=self.lower.reflect_rows_inplace`` into its scheduled
       walk (:class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`),
       so each just-swept octant group's reflective faces feed the next
       one — the inter-group reflect of :ref:`the reified splitting
       matrix <si-gauss-seidel-reification>`.
       Through the **Jacobi** split's ``upper`` half — which IS the
       full-inflow mask, because a Jacobi schedule reflects no face
       in-sweep (``[M]`` 4/4 geometries: ``lower`` is empty and
       ``upper`` carries every inflow row of every face) — the same
       verb is also the sweep-tier gates' *inter-sweep* reflect
       (``tests/sn/_test_helpers.py::reflect_outflow_into_inflow``).

       ⛔ **Until 2026-09-07 this row named a whole-trace ASSIGNMENT
       verb instead** — ``SNBoundaryOperator.reflect_inflow_inplace``,
       which filled each face's inflow slots with
       :math:`B\,\psi.\text{outflow}` outright before a bare sweep.  It
       had lost its last production caller at #448 (2026-09-06): the
       direct fixed-source SI loop moved onto the variadic driver at
       Wave O O.2a, and the eigenvalue reconstruction sweep in
       :func:`solve_sn <orpheus.sn.solver.solve_sn>` became one step of
       that same driven map (:ref:`sn-finalize-one-step`).  CS4c step 6
       item 6.5 retired it, with its trace-only leaf
       ``reflect_into_inflow``, rather than keeping a second verb for a
       test-tree consumer: *zero the inflow rows, then reflect
       additively on the full-inflow mask* reproduces the assignment
       **bit-for-bit** — ``[M]`` ``np.array_equal`` on 40/40 seeds ×
       4/4 geometries against a NON-zero-inflow buffer, and dropping
       the zeroing moves the answer by :math:`O(1)` (the positive
       control that the reading is not a zero-inflow artefact).  So the
       semantics survived without the surface; the gate is
       ``tests/sn/operators/test_reflect_helper_reexpression.py``.

The masked reflect is **not** a fold of :math:`B` into :math:`S`: it is
the trace-only :math:`A_{ss}` action of the *same* :math:`B`, restricted
to a subset of that operator's block ROWS and expressed on the boundary
trace alone. Both routes therefore deliver the identical :math:`-B`
coupling, and cannot drift, because both descend from
:meth:`SNBoundaryOperator._reflect_trace <orpheus.sn.operators.boundary.SNBoundaryOperator>`
(:ref:`bc-extraction-reflect-trace`).  Under the Gauss-Seidel schedule
they are not even alternatives but the two halves of ONE splitting
:math:`(L+C-B) = M - B_{\rm upper}`: the masked reflect carries
:math:`B_{\rm lower}` *inside* the reified forward :math:`M`, the
variadic gain carries :math:`B_{\rm upper}` lagged
(:ref:`si-gauss-seidel-reification`).


.. _bc-extraction-reflect-trace:

The trace-only :math:`A_{ss}` core — ``_reflect_trace``
-------------------------------------------------------

:math:`B` is the :math:`A_{ss}` block :math:`V_{\rm outflow} \to
V_{\rm inflow}`: it maps the *outflow* trace to the *inflow* trace.
Both delivery routes ultimately need the same per-face action —
restrict the face slot to :math:`\Gamma_+`, apply that face's realized
law (a specular
:class:`~orpheus.numerics.operator.PermutationOperator` on the
**reduced** ordinate axis for reflective, the zero map for vacuum), and
scatter the image back over :math:`\Gamma_-`. To guarantee
they cannot drift, that action is the single
:meth:`SNBoundaryOperator._reflect_trace <orpheus.sn.operators.boundary.SNBoundaryOperator>`
core, and **every** public route descends from it — there are exactly
two families.  The ``_apply_faces`` lift onto a zero-bulk
:class:`~orpheus.transport.full_field.FullField` carrier serves
:meth:`B.apply <orpheus.sn.operators.boundary.SNBoundaryOperator.apply>`,
its Euclidean transpose, and the masked operator's row-restricted
``apply``; the in-place additive
:meth:`SNMaskedBoundaryOperator.reflect_rows_inplace
<orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace>`
calls the core directly, because it writes onto a caller's trace buffer
rather than returning a field.

⛔ **Until 2026-09-07 the sentence above named a different second
consumer** — the trace-only *leaf* ``B.reflect_into_inflow``, added at
Wave O step O.2a (commit ``8563f4b``) with its mutating façade
``reflect_inflow_inplace``, and retired together with it at CS4c step 6
item 6.5.  The no-drift argument is unchanged in kind and stronger in
reach: the second consumer is now the verb production actually binds
(the Gauss-Seidel resolvent's row update), not a leaf whose only
remaining caller was a test-tree helper.

**Why the leaf existed, and why that reason expired.**
:meth:`B.apply <orpheus.sn.operators.boundary.SNBoundaryOperator.apply>` operates on a :class:`~orpheus.transport.full_field.FullField`
(zero bulk, trace populated) — the timeless, history-blind operator carrier
(:meth:`SNBoundaryOperator.apply <orpheus.sn.operators.boundary.SNBoundaryOperator.apply>`
is the base arrow ``FullField -> FullField``; the comonad lives on the
driver), the bulk only a carrier to reach the :math:`A_{ss}` boundary block. The pre-extraction direct helper
fabricated a throwaway zero-bulk field purely to call ``B.apply`` and
then discarded the (zero) bulk output.  ``reflect_into_inflow`` removed
that probe: it took a bare
:class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
trace and returned the boundary-only
:class:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink`
directly.  The saving was real for a *whole-trace* caller — and by 2026-09-07
there was none: the only trace-shaped consumer left was the sweep-tier gates'
inter-sweep reflect, which the masked verb serves through the Jacobi split's
``upper`` half.  A verb whose whole justification is an ergonomic saving for
callers that no longer exist is a second spelling of one action, so item 6.5
retired it rather than keeping the two in step.

**The full-field route's input contract** (CS4c step 6 item 6.3,
2026-09-07).  Because ``B.apply`` is reached through a composite carrier,
``_apply_faces`` parses that carrier before touching it, and it does so
through the SAME body its :math:`L` and :math:`L+C` siblings use —
:meth:`FullField.require_member
<orpheus.transport.full_field.FullField.require_member>`: a ``TypeError``
naming the refusing surface and the carrier it wanted for a foreign object
(until this item ``_apply_faces`` read ``psi.interior`` unguarded and leaked
a raw ``AttributeError``), a ``ValueError`` carrying the greppable
*space-content invariant* vocabulary for a content mismatch.  The parse is
keyed on the operator's **carrier mesh**, not on its bound end, and
deliberately so: :math:`B_a` is fed the windowed *moment* iterate as well as
the angular composite its ends name (``[M]`` 59 / 58 / 47
``HarmonicMomentFlux``-interior composites per 2-D windowed solve), and a
bound-end comparison would refuse every one of them.  ⚠ The guard is
tagged ``ELEGANCE-DEBT[guard]`` (`#457
<https://github.com/deOliveira-R/ORPHEUS/issues/457>`_) — it is a
protection, not the target state.  It retires when every leaf is bound on
the end it acts on, after which an alien carrier cannot be typed at all and
the admission is the ordinary composability guard.

Keeping :math:`B`'s emission off the outflow row is load-bearing: as
the sibling :math:`-B` reading the *whole* boundary block, a non-zero
outflow emission corrupts the outflow-definition residual
:math:`r_{\rm outflow}` (which must carry **no** :math:`B` term —
:ref:`bc-extraction-design-corrections`). The extraction achieved that
by *projecting* a full-face law's image onto the inflow rows; since
campaign phase **B3.2** it is achieved by *typing* — the realized law's
domain is :math:`\Gamma_+` and its codomain is :math:`\Gamma_-`, so
:meth:`_reflect_trace <orpheus.sn.operators.boundary.SNBoundaryOperator>`
composes :math:`\iota_-\circ\text{law}\circ\gamma_+` and there is no
wider image to project (:ref:`bc-domain-narrowing`). The transpose
composes :math:`\iota_+\circ\text{law}^{\mathsf T}\circ\gamma_-`, so it
masks its INPUT to :math:`\Gamma_-` and scatters its OUTPUT over
:math:`\Gamma_+` — the asymmetry that
:ref:`correction 2's warning <bc-extraction-design-corrections>`
explains and that no reflective fixture can detect.


.. _bc-extraction-scope:

Scope — both 1-D and 2-D are now bare (O.4b complete)
-----------------------------------------------------

O.4a.2 made the **1-D** sweep bare (slab / sphere / cylinder). Step
**O.4b** then made the **2-D Cartesian wavefront sweep bare as well**
(both :func:`~orpheus.sn.loss_representation._sweep_jacobi` and the 2-D matvec
:meth:`StreamingOperator._apply_2d_cartesian <orpheus.sn.operators.streaming.StreamingOperator>`):
the intra-sweep ``bc.apply`` is **gone** for every geometry. The
octant-incoming face edge is seeded from the *given* inflow trace and
the reflective coupling :math:`\psi.\text{inflow} = B\,\psi.\text{outflow}`
is delivered by the sibling :math:`-B`
(:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`) for the
2-D trace exactly as for the 1-D trace. The 2-D matvec emits the
boundary block as an active-trace residual (outflow slots carry the
self-consistency defect ``streamed − ψ.outflow``; inflow slots carry
the identity ``ψ.inflow``), wired into the composed Krylov matvec as
the boundary gain :math:`B` of
:func:`~orpheus.sn.coupled_system.build_within_group_system`.
The interior face fluxes the bare 2-D sweep + matvec propagate are the
interior 1-cochain :math:`C^1_{\rm int}` — carried since S6.4(f) by the
rolling ``_MovingFrontier`` front (the ``WavefrontFlux`` type that named
it through #205 Phase 5 is retired; see :ref:`wavefront-flux-cochain`).

The dispatch is still guarded by a **single predicate** so the two
geometry paths cannot drift: ``sn_mesh.reduced is not None`` is the
**same** predicate the representation dispatch
(:func:`~orpheus.sn.loss_representation.default_for`, via each
representation's ``supports``) reads to select the 1-D scan
(:class:`~orpheus.sn.loss_representation.CumprodScan`) vs the 2-D
wavefront body, and the **same**
predicate the two entries' reconstructions read.  ⛔ This clause named
*"the direct-helper guards … before calling
``_reflect_outflow_into_inflow``"* until 2026-09-06: neither entry calls a
direct helper any more (the fixed-source SI loop stopped at Wave O O.2a,
the eigenvalue finalize at #448 — :ref:`sn-finalize-one-step`), so the
predicate now selects a *representation*, which was always what it was
about.  Both branches are now bare-sweep + sibling :math:`-B`; the predicate
selects the *fold shape* (1-D parallel-prefix scan vs 2-D wavefront
DAG), **not** a bare-vs-bc-in-sweep distinction.


.. _bc-extraction-scope-future:

Closed typing-completion seam — :attr:`ScatteringOperator.domain` (P4.5 W-D)
----------------------------------------------------------------------------

This Wave-O typing-completion **landed in P4.5 W-D** (commit
``0610b39``); it is recorded here because it was a documented seam at
the time of Wave O's close-out. The Wave-O framing envisioned giving
:class:`~orpheus.transport.operators.scattering.ScatteringOperator`
(and the other bulk leaves) a **bulk** :math:`V_{\rm bulk}` domain so
that :class:`~orpheus.numerics.operator.OperatorSum` would **reject** a
re-introduced :math:`S + B` fold (the domain-compatibility check
throwing ``IncompatibleOperatorComposition`` on a bulk :math:`S`
summed with a trace :math:`B`).

W-D closed the seam, but with a **different and stronger** choice: the
bulk leaves :math:`C`/:math:`S`/:math:`F` carry the **composite**
:class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace` — the
*same* instance :math:`L`/:math:`B` advertise — not a bulk-only
:math:`V_{\rm bulk}` space. The motivation shifted from the negative
"reject :math:`S + B`" tripwire to the positive **within-group
composition guard**: every operand of the within-group loss
:math:`(L + C) - S` now reports the same composite domain, so the
:class:`~orpheus.numerics.operator.OperatorSum` guard **validates** the
build (equal domains AND codomains) on every solve instead of silently
skipping a ``None``-spaced summand (W-D — see
:ref:`g-adjoint`). A consequence is that the original
"bulk-:math:`S` ≠ trace-:math:`B`" rejection no longer applies: a
composite-typed :math:`S` and a composite-typed :math:`B` speak the
same space, so an ``OperatorSum`` of the two would compose. The reason
the :math:`S + B` fold stays gone is the **variadic-driver redesign**
(:ref:`bc-extraction-variadic-driver`) — :math:`B` is a lagged gain,
never a resolvent summand — not a typing rejection.

The space-anonymous ``domain = None`` survives only for the **bare /
test constructor** (a :class:`ScatteringOperator` built without
``from_solver_data``'s ``full_field_space=`` thread): then the guard
skips that operand, preserving the legacy backward-compatible contract
for direct callers.


.. _bc-extraction-2d-si-krylov-twin:

The 2-D Cartesian eigenvalue SI inner is the geometry-agnostic twin of Krylov
-----------------------------------------------------------------------------

Because the variadic :math:`-S - N_{2n} - B` gains ride the **bare** sweep
for
every geometry (above), the two within-group eigenvalue inner solvers
are **structural twins** — they share every operator and every
reduction, differing only in the iteration driver. This holds for 2-D
Cartesian exactly as it does for slab / sphere / cylinder, so a 2-D
Cartesian eigenvalue problem solves through **both** inner solvers:

- :meth:`SNSolver._solve_source_iteration <orpheus.sn.solver.SNSolver._solve_source_iteration>`
  — the source-iteration inner, the :func:`~orpheus.sn.solver.solve_sn`
  **default** ``inner_solver="source_iteration"`` for *every* geometry,
  driven by :class:`~orpheus.numerics.iteration.SourceIteration`;
- :meth:`SNSolver._solve_krylov <orpheus.sn.solver.SNSolver._solve_krylov>`
  — the Krylov inner, opt-in ``inner_solver="krylov"``, driven by
  :class:`~orpheus.numerics.iteration.KrylovAcceleration`.

The two inners are identical except for that driver. Both build the
same composite right-hand side
(:meth:`AngularSourceSink.from_isotropic <orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink.from_isotropic>`
bulk + ``AngularBoundarySourceSink.zeros(sn_mesh.angular_trace)``
boundary inside a
:class:`~orpheus.transport.timed_full_field.TimedFullField`), the same
loss decomposition (the resolvent :math:`L + C` from
:class:`~orpheus.sn.operators.streaming.StreamingOperator` + the collision
multiplier :math:`C = M[\sigma_t]`
(:class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
plus the scattering
gain :math:`S` and the boundary reflection gain :math:`B` from
:func:`~orpheus.sn.coupled_system.build_within_group_system`;
zero within-group fission), and the
same :meth:`integrate_angular <orpheus.transport.fields.angular_flux.AngularFlux.integrate_angular>`
angular reduction. Neither driver carries any geometry dependence.

The reflective coupling reaches both drivers on the **bare** 2-D
wavefront sweep through the sibling :math:`-B` (the **variadic-gain**
route of :ref:`bc-extraction-two-routes`), never through an in-sweep
``bc.apply``. The :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`
is natively **four-face** (``xmin`` / ``xmax`` / ``ymin`` / ``ymax``)
and is the *same* operator the 2-D Krylov path uses — there is no
separate per-geometry boundary closure.

.. admonition:: The "B1'' face block" is retired legacy
   :class: note

   The 2-D Cartesian eigenvalue path was historically described as
   needing a distinct "B1'' face block" that was "1-D-only", which is
   why the source-iteration inner was once guarded against 2-D meshes.
   That guard is **removed**. "B1''" was never a code symbol — it was a
   1-D boundary-closure *name* in docstrings and comments, fully
   superseded by the L2
   :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux` +
   :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`
   bare-boundary architecture (O.4a.2 / O.4b above), which realises the
   boundary handling for *all* geometries. The 2-D path never required
   a separate 1-D-only block. Because
   :func:`~orpheus.sn.solver.solve_sn` defaults to the source-iteration
   inner for every geometry, the now-removed guard had meant the
   **default** 2-D Cartesian eigenvalue entry point raised; the carve
   restores it.

**Numerical evidence (SI ≡ Krylov ≡ closed-form** :math:`k_\infty`
**).** The twin is pinned at the production
:func:`~orpheus.sn.solver.solve_sn` entry (not a hand-rolled power
loop) by ``tests/sn/eigenvalue/test_keff_2d.py::TestSIKrylov2DEquivalence``:

.. list-table:: 2-D Cartesian SI/Krylov verification (Wave O step #208)
   :header-rows: 1
   :widths: 38 34 28

   * - Leg
     - Reference (pillar)
     - Result
   * - Default-entry homogeneous (1G / 2G / 4G)
       (:func:`test_default_entry_hits_kinf <tests.sn.eigenvalue.test_keff_2d.TestSIKrylov2DEquivalence.test_default_entry_hits_kinf>`)
     - Closed-form :math:`k_\infty = \lambda_{\max}(A^{-1}F)`
       (1g → 1.5, 2g → 1.875, 4g → 1.4878)
     - SI hits :math:`k_\infty` to :math:`< 10^{-8}`
   * - Heterogeneous 2G fuel\|moderator, **non-flat** flux
       (:func:`test_si_krylov_heterogeneous_2g_nonflat_flux <tests.sn.eigenvalue.test_keff_2d.TestSIKrylov2DEquivalence.test_si_krylov_heterogeneous_2g_nonflat_flux>`)
     - SI vs Krylov flux **shape** + eigenvalue
     - flux shape agrees to :math:`\sim 10^{-9}`
   * - 2-D SI :math:`k_{\rm eff}` Cauchy under refinement
       (:func:`test_si_2d_keff_converges_under_refinement <tests.sn.eigenvalue.test_keff_2d.TestSIKrylov2DEquivalence.test_si_2d_keff_converges_under_refinement>`)
     - Self-convergence (consistency regression catcher)
     - monotone, single fixed point

The structural-independence discipline (``vv-principles`` L11 /
anti-pattern #1) applies: SI ≡ Krylov *alone* is twin-path agreement —
necessary but **not** sufficient, since both could share a defect. It
becomes correctness evidence only because the homogeneous leg
independently anchors the same production path to the **closed-form**
:math:`k_\infty` eigenvalue (the closed-form pillar; per ``vv-principles``,
twin-implementation agreement is L4-class on its own and MMS does not
prove eigenvalues). The heterogeneous leg carries a genuinely non-flat
(≥2G, fuel\|moderator) flux so the angular / wavefront redistribution
terms are active rather than nulled (``vv-principles`` anti-patterns
#3 / #4), and the un-xfailed L2 mesh-convergence pin
(``tests/sn/sweep/cartesian_2d/test_discrete_ordinates_2d.py::test_do_mesh_convergence``,
the ERR-003 catcher) plus the ``2d_2g_LS4_dd_8x4_het_si`` regression
snapshot round out the catch surface.


.. _bc-extraction-numerical-evidence:

Numerical evidence
------------------

The extraction is verified by three independent grounds (per the
``vv-principles`` skill's three pillars and the bit-identity
vs principled-equivalence gate).

**1. Vacuum bit-identity.** With :math:`B = 0` the boundary gain
contributes nothing (:math:`B.\text{apply} \equiv 0`), so the variadic
matvec :math:`(L+C).\text{apply} - S.\text{apply} - B.\text{apply}`
reduces exactly to :math:`(L+C).\text{apply} - S.\text{apply}` and the
vacuum path is **bit-identical** to the pre-extraction matvec.  (The
two-gain spelling here is Wave O's, which is what the cited captures
were taken against; the :math:`(n,2n)` gain :math:`N_{2n}` joined the
tuple at CS4c step 3 and rides this argument unchanged, contributing
exactly zero on the :math:`\Sigma_{2n} \equiv 0` fixtures the captures
use.) Verified
by:

- the matvec 18-baseline snapshot
  (:func:`numpy.array_equal` against the pre-O.4a.2 captures across
  slab / sphere / cylinder × 1G / 2G / asymmetric :math:`\Sigma_s` ×
  vacuum / white / specular), and
- the end-to-end regression snapshots.

This is the bit-identity-by-inheritance gate: vacuum keeps the
verified pre-extraction value for free (``vv-principles``
§"Bit-identity vs principled-equivalence", criterion: implementation
unchanged on the vacuum path because the bare sweep reads a zero
inflow seed).

**2. Reflective convergence-equivalence (closed-form pillar).** The
reflective path relocates the reflection from inside the sweep to the
sibling :math:`-B`, so it is **not** bit-identical but
*convergence-equivalent* to a structurally-independent analytical
reference:

.. list-table:: Reflective convergence-equivalence gates
   :header-rows: 1
   :widths: 40 30 30

   * - Test
     - Reference (pillar)
     - Both solvers?
   * - Curvilinear streaming-equilibrium
       (``tests/sn/sweep/curvilinear/test_streaming_equilibrium_curvilinear.py``)
     - Analytical infinite-medium balance
       :math:`\phi = q/\Sigma_t` (closed-form)
     - ``source_iteration`` AND ``krylov``
   * - Reflective :math:`k_\infty` homogeneous
       (``tests/sn/verification/analytical/test_kinf_homogeneous.py``)
     - :math:`k_\infty = \nu\Sigma_f / \Sigma_a` (closed-form
       eigenvalue — MMS does NOT prove eigenvalues)
     - both
   * - ``test_si_carve_recovers_analytical_kinf``
       (``tests/sn/operators/test_invertible_operator.py``)
     - Analytical :math:`k_\infty` via the SI path with :math:`B`
       folded (closed-form)
     - SI path
   * - Invertible-operator :math:`Q/\Sigma_t` recovery
       (``tests/sn/operators/test_invertible_operator.py``)
     - Flat-flux fixed-source balance (closed-form)
     - direct ``−B`` drive

**3. Reflective eigenvalue regression (principled-equivalence ULP).**
The reflective cylinder eigenvalue regression snapshot **drifts within
tolerance**: :math:`4\times 10^{-13}` on :math:`k_{\rm eff}` and
:math:`7\times 10^{-12}` relative on the scalar flux. This is **not** a
bug — it is FP-non-associativity from relocating the reflection
(``vv-principles`` § criterion 3: the reduction-tree changes because
the reflection now happens in :math:`-B` rather than fused into the
sweep, so additions occur in a different IEEE-754 order). The drift is
bounded by ``iteration_count × condition_number × ULP``, well under the
existing ``rtol`` regression tolerance. The new value is
convergence-equivalent to the analytical references above (criterion
2), so the regression contract is satisfied without relaxation beyond
the snapshot tolerance.

**The O.2a variadic reassociation is a second principled-equivalence
instance.** Splitting the matvec from :math:`(L+C) - (S+B)` (the
retired fold) to :math:`(L+C) - S - B` (the two separate gains
:ref:`bc-extraction-variadic-driver` had at the time — :math:`N_{2n}`
became a third at CS4c step 3), and the rhs symmetrically,
re-associates the same additions into a different IEEE-754 order. The
regression snapshots drift at FP-noise level — reflective cylinder
:math:`4.2\times 10^{-13}` on :math:`k_{\rm eff}` and :math:`6.8\times
10^{-12}` relative on the scalar flux, anisotropic 3–5 ULP — all within
the existing tolerances (:math:`10^{-11}` / :math:`10^{-9}` /
:math:`10^{-12}`). Per ``vv-principles`` criterion 2 the new value is
verified against **structurally-independent** references, not merely
shown close to the old value: the NEW-1 closed-form :math:`Q/\Sigma_t`
flat-flux balance, the SI ≡ Krylov twin (:ref:`bc-extraction-2d-si-krylov-twin`),
and the ``keff_2d`` closed-form :math:`k_\infty`. The reassociation
satisfies all three criteria (named intermediates — each gain's output
is a principled source/sink; structurally-independent reference;
dimensionally-explainable drift), so no contract relaxation is needed.


.. _bc-extraction-operator-output-typing:

Operator-output role typing — :math:`A\psi` is a source/sink
------------------------------------------------------------

Wave O step B.5.2 (`Issue #208
<https://github.com/deOliveira-R/ORPHEUS/issues/208>`_, commit
``6ef5063``, 2026-06-03) retyped every SN operator's ``.apply`` output
``.boundary`` from
:class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux` to
:class:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink`
— the *source/sink* role leaf. This completes the **boundary** half of
the B.5 operator-output "dimensional-sin" carve; the **bulk** half
(``.apply.bulk`` →
:class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`)
landed earlier in commit ``f400743``. The two halves make the boundary
role grid a clean parallel of the bulk.

.. list-table:: The completed role grid (bulk ‖ boundary)
   :header-rows: 1
   :widths: 18 28 28 26

   * - Block
     - ``.apply`` (operator output :math:`A\psi`)
     - ``.solve`` (swept solution trace)
     - ``from_balance`` (the defect)
   * - **bulk** (:math:`V_{\rm bulk}`)
     - :class:`~orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink`
       (``f400743``)
     - :class:`~orpheus.transport.fields.angular_flux.AngularFlux`
     - :class:`~orpheus.transport.residuals.angular_residual.AngularResidual`
       (consumed by :func:`~orpheus.sn.solver.evaluate_residual`,
       O.2 — :ref:`affine-typed-residual`)
   * - **boundary** (:math:`V_{\rm inflow} \oplus V_{\rm outflow}`)
     - :class:`~orpheus.transport.source_sinks.angular_boundary_source_sink.AngularBoundarySourceSink`
       (``6ef5063``, B.5.2)
     - :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
     - :class:`~orpheus.transport.residuals.angular_boundary_residual.AngularBoundaryResidual`
       (consumed by :func:`~orpheus.sn.solver.evaluate_residual`,
       O.2 — :ref:`affine-typed-residual`)

The governing principle (the load-bearing rationale)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   *A residual only arises after we compare an operator output against
   something else and get a defect (a balance). The output of an
   operator is NOT a residual straightaway.*

Each operator's ``.apply`` emits :math:`A\psi` — a **source/sink**
(a signed reaction-rate / flux density: a *source* when produced, a
*sink* when it is an operator-loss output such as :math:`L\psi`; the
single role leaf holds both, hence ``SourceSink``). The residual is
**only** the named composition
:meth:`AngularBoundaryResidual.from_balance(lhs, rhs) <orpheus.transport.residuals.angular_boundary_residual.AngularBoundaryResidual.from_balance>`
of the affine boundary balance
:math:`r_\Gamma = \gamma_-\psi - (R\,G\,\gamma_+\psi + q)`. The GMRES
*flat* residual :math:`b - A\psi` is formed internally on the **raveled
vector** (via :meth:`TimedFullField.to_flat <orpheus.transport.timed_full_field.TimedFullField.to_flat>`)
and is **never typed as a field** — so at B.5.2
:class:`AngularBoundaryResidual` had no operator-output consumer; its first
consumer is the honest :math:`L+C-S-N_{2n}-F-B` driver of Wave O step
**O.2** (see :ref:`bc-extraction-operator-output-o2`). That consumer
has since landed:
:func:`~orpheus.sn.solver.evaluate_residual` types the balance defect
:math:`(L+C-S-N_{2n}-B)\psi - q` via ``from_balance`` (Wave O step O.2
close-out, :ref:`affine-typed-residual`), so
:class:`AngularBoundaryResidual` and its bulk sibling
:class:`~orpheus.transport.residuals.angular_residual.AngularResidual`
are now consumed, not merely minted.


The two-hat tension and why ``AngularBoundarySourceSink`` dissolves it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The earlier in-flight plan
(``.claude/plans/b52_boundary_residual_retype.md``) proposed typing the
matvec output boundary as :class:`AngularBoundaryResidual`. That choice was
**rejected** for two reasons:

1. **It breaks consistency with the already-landed bulk.** The bulk
   ``.apply.bulk`` uses the source/sink leaf
   (:class:`AngularSourceSink`), **not** a residual, for operator
   outputs. Typing the boundary output as a residual would make the two
   halves of the same carve disagree on what an ``.apply`` output *is*.

2. **It creates a "two-hat" tension that the class gate cannot
   satisfy.** The realized boundary law
   :class:`~orpheus.sn.operators.boundary.SNBoundaryOperator` (:math:`B`)
   emits :math:`B\,\psi.\text{outflow}`, and that **same** emission is
   consumed two ways:

   .. list-table:: :math:`B`'s two consumers — the "two hats"
      :header-rows: 1
      :widths: 24 38 38

      * - Consumer
        - Composition
        - The hat :math:`B\,\psi.\text{outflow}` would wear
      * - Krylov matvec
        - :math:`(L+C).\text{apply} - S.\text{apply} -
          N_{2n}.\text{apply} - B.\text{apply}`
        - a **residual** term (subtracted from the diagonal)
      * - SI rhs
        - :math:`q_{\rm ext} + S.\text{apply} + N_{2n}.\text{apply} +
          B.\text{apply}`
        - a **source** term (the inflow seed the bare sweep reads)

   One operator cannot emit :class:`AngularBoundaryResidual` for the matvec
   **and** :class:`AngularBoundarySourceSink` for the SI rhs — the
   :class:`TimedFullField <orpheus.transport.timed_full_field.TimedFullField>`
   class gate (strict class identity:
   ``type(self.boundary) is not type(other.boundary)`` ⟹ ``TypeError``)
   throws on ``AngularBoundaryResidual + AngularBoundarySourceSink`` the moment the SI
   rhs tries to add :math:`B.\text{apply}` (a residual, under OPT-BR)
   to :math:`S.\text{apply}` and :math:`q_{\rm ext}` (sources). The
   variadic driver (:ref:`bc-extraction-variadic-driver`) makes this
   sharper than the retired fold: each gain's output is summed
   *individually*, so :math:`B`'s lone hat must be a source/sink for the
   rhs sum :eq:`bc-extraction-variadic-matvec` to close.

Choosing :class:`AngularBoundarySourceSink` for **all** operator outputs
dissolves the two-hat: :math:`B` wears **one** hat (it always emits a
source/sink), and **both** sums close as homogeneous
:class:`AngularBoundarySourceSink` sums —

.. math::
   :label: bc-extraction-two-hat-closed-sums

   \underbrace{(L+C).\text{apply} - S.\text{apply}
               - B.\text{apply}}_{\text{Krylov matvec}}
   \quad\text{and}\quad
   \underbrace{q_{\rm ext} + S.\text{apply}
               + B.\text{apply}}_{\text{SI rhs}}

both stay within the single :class:`AngularBoundarySourceSink` class. This
needs **no SI-driver restructure** and **no partial-O.2**:
:class:`AngularBoundaryResidual` stays reserved for the named
``from_balance`` composition exactly as
:class:`AngularResidual` waits on the bulk.

.. vv-status: bc-extraction-two-hat-closed-sums documented

A throwaway **decision instrument**
(``derivations/diagnostics/diag_b52_boundary_typing_decision.py``, the
B0 de-risk) proved on a 1-D reflective slab **and** a 2-D reflective
box that the OPT-BSS choice (``AngularBoundarySourceSink`` for the matvec
output) closes both sums, while the OPT-BR choice
(:class:`AngularBoundaryResidual` for the matvec output) throws the two-hat
``TypeError`` on the SI rhs.


Why the Krylov path is safe with a ``AngularBoundarySourceSink`` matvec output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The matvec output never escapes scipy as a :class:`AngularBoundarySourceSink`,
so the *solution* side stays :class:`AngularBoundaryFlux`. The mechanism is the
flat round-trip:

* :meth:`TimedFullField.to_flat <orpheus.transport.timed_full_field.TimedFullField.to_flat>`
  ravels the composite to ``[bulk.values.ravel(), boundary.values]`` —
  a **type-agnostic** 1-D vector (the class of ``.boundary`` is erased).
* scipy's GMRES iterate is reconstructed via
  :meth:`TimedFullField.from_flat <orpheus.transport.timed_full_field.TimedFullField.from_flat>`,
  which rebuilds the boundary with
  ``replace(template.boundary, values=...)`` off the flux
  ``solution_template``. Because the template's boundary is a
  :class:`AngularBoundaryFlux`, the reconstructed iterate's boundary is a
  :class:`AngularBoundaryFlux`.

So the matvec's *internal* :class:`AngularBoundarySourceSink` boundary class
lives only inside one ``op.apply`` call; the moment the result is
raveled and handed to scipy, the class is gone, and the iterate scipy
hands back is reconstructed as the flux type. The solve/iterate/trace
sites are therefore correctly **kept** :class:`AngularBoundaryFlux`:
:meth:`MultiplicationOperator.solve <orpheus.transport.operators.multiplication_operator.MultiplicationOperator.solve>`
(the collision multiplier :math:`C = M[\sigma_t]`),
the boundary buffer of
:meth:`StreamingCollisionOperator._solve_timed_full_field <orpheus.sn.operators.streaming.StreamingCollisionOperator._solve_timed_full_field>`,
the cold-start ``initial_guess`` iterates
(``TimedFullField.zeros(..., boundary=AngularBoundaryFlux, ...)``), the
converged traces, and the sweep's persistent boundary buffer.


The 13 retyped sites
~~~~~~~~~~~~~~~~~~~~

Thirteen sites (operator outputs + ``q_ext`` sources) flipped from
:class:`AngularBoundaryFlux` to :class:`AngularBoundarySourceSink`:

.. list-table:: B.5.2 retyped sites
   :header-rows: 1
   :widths: 34 38 28

   * - Module / symbol
     - Site
     - Emission
   * - :mod:`orpheus.sn.loss_representation`
     - ``_OneDimScanWalk._apply_walk`` (``m_boundary``) — the fused
       1-D matvec body; #206 relocated it here, #238 folded the former
       ``_compute_LpC`` / ``_compute_decomposition`` /
       ``_SpatialSweepDirection`` sites into this single walk
     - :math:`L+C` boundary block
   * - :mod:`orpheus.sn.operators.streaming`
     - :meth:`StreamingOperator._apply_2d_cartesian <orpheus.sn.operators.streaming.StreamingOperator>`
     - 2-D boundary block
   * - :mod:`orpheus.transport.operators.multiplication_operator`
     - :meth:`MultiplicationOperator.apply <orpheus.transport.operators.multiplication_operator.MultiplicationOperator.apply>`
       (the collision multiplier :math:`C = M[\sigma_t]`)
     - bulk → bulk; boundary zero
   * - :mod:`orpheus.transport.operators.scattering`
     - :meth:`ScatteringOperator.apply <orpheus.transport.operators.scattering.ScatteringOperator>`
     - boundary zero
   * - :mod:`orpheus.transport.operators.n2n`
     - :meth:`N2NOperator.apply <orpheus.transport.operators.n2n.N2NOperator>`
     - boundary zero
   * - :mod:`orpheus.transport.operators.fission`
     - :meth:`FissionOperator.apply <orpheus.transport.operators.fission.FissionOperator>`
     - boundary zero
   * - :mod:`orpheus.sn.operators.boundary`
     - :meth:`SNBoundaryOperator._apply_faces <orpheus.sn.operators.boundary.SNBoundaryOperator>`
       (``apply`` **and** ``apply_transpose``)
     - :math:`B\,\psi.\text{outflow}` on the inflow slots
   * - :mod:`orpheus.sn.solver`
     - ``q_ext.boundary`` at the **3 source builds** (eigenvalue SI,
       eigenvalue Krylov, fixed-source SI / reconstruction)
     - prescribed inflow (zero for vacuum / reflective)

(The B.5.2 ``_zero_within_group_fission`` slot — a ``ZeroOperator``
``codomain_zero`` emitting the boundary zero for an explicit ``F = 0``
within-group operator — was designed here but NEVER wired: the Wave-O
within-group decomposition (:func:`~orpheus.sn.coupled_system.build_within_group_system`)
routes within-group fission through ``q_ext`` instead, so no zero-fission
operator is ever constructed. The dead helper retired 2026-07-03 (C4).)

The change is **type-only**: the zero-trace allocation (then spelled
``AngularBoundarySourceSink.zeros_on(mesh)``, since CS4b S5
``.zeros(mesh.angular_trace)``)
and the per-face-view writes produce **bit-identical** ``.values`` —
only the wrapping role-type differs. The dead :class:`AngularBoundaryFlux`
runtime imports were retired from the retyped sites.


.. _bc-extraction-operator-output-o2:

The extraction close-out — where the remaining items landed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wave O step **O.2a has landed the honest** :math:`L+C-S-B` **driver**
via the variadic couplings of :ref:`bc-extraction-variadic-driver`:
the transitional :math:`S + B` fold is **retired** and :math:`B` is now
a first-class coupling gain. Of the items B.5.2 left for the rest of
O.2, **the adjoint metric and its gate landed in step O.2b
R5** (:ref:`g-adjoint`), and **the residual column landed
in the O.2 close-out** (:ref:`affine-typed-residual`):

* :meth:`AngularResidual.from_balance <orpheus.transport.residuals.angular_residual.AngularResidual.from_balance>`
  and
  :meth:`AngularBoundaryResidual.from_balance <orpheus.transport.residuals.angular_boundary_residual.AngularBoundaryResidual.from_balance>`
  now have their first production-reachable consumer:
  :func:`~orpheus.sn.solver.evaluate_residual` types the within-group
  balance defect :math:`r = (L+C-S-N_{2n}-B)\psi - q` as the composite
  ``FullField(bulk=AngularResidual, boundary=AngularBoundaryResidual)`` — the
  **timeless** carrier, because a residual is a one-shot balance defect
  carrying no iteration history (the ``history_depth = 0`` degenerate;
  P4.5 W-C confines the timed type to the driver iterate)
  (see :ref:`affine-typed-residual`). The honest variadic driver still
  emits each gain's output as a :class:`AngularBoundarySourceSink` and the
  GMRES defect is still the *flat* :math:`b - A\psi` on the raveled
  vector — the typed residual is an **additive diagnostic + DSA
  substrate**, never in the convergence path (so the converged flux
  stays bit-identical).
* the :math:`|\Omega\cdot\hat n|\,w` :math:`G`-metric adjoint ``.H``
  (the boundary-weighted inner product for the transpose) **landed in
  R5** — ``op.H`` is now the metric-correct G-adjoint :math:`G^{-1}
  A^{\mathsf T} G` over the composite :math:`V_{\rm bulk}\oplus
  V_{\rm trace}` (:ref:`g-adjoint`). This is exactly why
  :math:`B` stays trace-typed as a separate gain
  (:ref:`bc-extraction-variadic-driver`): a bulk-folded :math:`B` would
  erase the trace metric the adjoint needs.
* **Gate-1.3** (the O.2 adjoint verification gate) **landed in R5** —
  the dense-probe oracle + the L11 wrong-metric control
  (:ref:`g-adjoint`).

The direct-helper survived O.2a (the driver stopped routing through it,
but the final eigenvalue reconstruction sweep still did) and has **no
production caller at all since #448**, when that sweep became one step of
the driven map (:ref:`sn-finalize-one-step`); it now lives with the
sweep-tier gates (:ref:`bc-extraction-two-routes`).  The
:attr:`ScatteringOperator.domain` typing completion that was once a
documented seam **landed in P4.5 W-D** — :math:`S` now carries the
composite full-field space (:ref:`bc-extraction-scope-future`).

The residual column is now wired: :class:`AngularBoundaryResidual` and
:class:`~orpheus.transport.residuals.angular_residual.AngularResidual`
are consumed by :func:`~orpheus.sn.solver.evaluate_residual` (Wave O
step O.2 close-out — :ref:`affine-typed-residual`). ⛔ That close-out
also completed the *affine* flux algebra — the iterate increment was a
typed ``AngularDisplacement`` and ``flux + flux`` was a
:class:`TypeError` — and **that half was overturned on 2026-08-19**:
flux lives in the positive cone of a vector space, so ``flux + flux`` is
legal and the increment is the flux type carrying a signed value
(:ref:`cone-typed-field-algebra`). The residual half above is unaffected
— the residual role was always a plain vector role. The canonical home
for the field algebra is now :ref:`cone-typed-field-algebra`.


.. vv-status note: the operator-output role typing of B.5.2 is a
   type-only refactor (bit-identical ``.values``); its correctness is
   verified by the same gates that verify the extraction
   (:ref:`bc-extraction-numerical-evidence`) plus the type-residual
   gates catalogued below. B.5.2's verification ground:

* the **B0 decision instrument** (``diag_b52_boundary_typing_decision.py``)
  proving OPT-BSS closes both sums while OPT-BR throws the two-hat;
* the core operator / boundary / 2-D suite (324 passed);
* SI eigenvalue slab / sphere / cylinder × 1 / 2 / 4-group — the
  **two-hat exerciser** (the SI rhs sum that OPT-BR would throw on);
* Krylov :math:`k_\infty` (14 cases);
* the type-residual gates (``test_native_matvec`` boundary-output type
  assert migrated; positive type asserts added for the 2-D matvec
  ``test_bc_extraction_2d`` and for :math:`B` in
  ``test_sn_boundary_operator``);
* the dimensional-check sentinel suite (36 / 36, run without ``-O``);
* MMS L1 1-D + 2-D + curvilinear (8 passed, 6 xfail — flux-shape /
  convergence-order pillar; MMS does **not** prove the eigenvalue).

The change was reviewed by the ``elegance-enforcer`` (PASS, no
conditions).


.. _bc-trace-structure:

Trace structure (Γ\_-, Γ\_+)
============================

The transport equation lives on a phase space :math:`\Omega \times
S^d` where :math:`\Omega \subset \mathbb{R}^d` is the spatial domain
and :math:`S^d` is the unit sphere of directions. The boundary
:math:`\partial\Omega` carries an outward unit normal :math:`\hat
n(\mathbf{r})` at every regular point. For an angular flux
:math:`\psi(\mathbf{r}, \Omega)` defined on the full phase space, the
**boundary trace** splits naturally into two pieces by the sign of
:math:`\Omega \cdot \hat n`:

.. math::
   :label: trace-sign-predicate

   \Gamma_- \;=\; \{(\mathbf{r}, \Omega) \in \partial\Omega \times S^d
                  : \Omega \cdot \hat n(\mathbf{r}) < 0\},
   \qquad
   \Gamma_+ \;=\; \{(\mathbf{r}, \Omega) \in \partial\Omega \times S^d
                  : \Omega \cdot \hat n(\mathbf{r}) > 0\}.

.. (vv-status rationale) Notation definition: the continuous inflow / outflow
   trace half-spaces Γ_± by the sign of Ω·n. Its discrete realisation
   :eq:`inflow-mask-discrete` is the tested form
   (``tests/numerics/test_angular_trace_space.py`` selector gates). A
   definitional predicate, not a solver claim.
.. vv-status: trace-sign-predicate documented

Points with :math:`\Omega \cdot \hat n = 0` are **tangential** —
they belong to neither half. For axis-aligned ordinates on
axis-aligned faces these arise exactly (no round-off) for face
normals perpendicular to the ordinate's direction cosine; for
general curvilinear faces or generic ordinates they arise only
at a measure-zero subset that the discrete representation
identifies via a small tolerance
(``TANGENTIAL_EPS = 4 * np.finfo(np.float64).eps``, i.e.
:math:`\approx 8.9\times 10^{-16}`, in
:mod:`orpheus.numerics.spaces.angular_trace_space`). That constant is
the codebase's ONE outflow/inflow classifier: campaign phase **B3.4a**
retired the white operator's private ``> 0.0`` twin precisely because a
strict compare disagrees with it wherever a quadrature carries
tangential ordinates (:ref:`bc-narrowing-b34a`).

In the discrete setting, the spatial boundary is a union of finite
faces :math:`\{f_1, \ldots, f_F\}` and the angular variable is a
finite ordinate set :math:`\{\Omega_n : n = 1, \ldots, N\}`. The
sign predicate :eq:`trace-sign-predicate` then collapses to a
**per-face boolean mask** of shape :math:`(F, N)`:

.. math::
   :label: inflow-mask-discrete

   \mathrm{inflow\_mask}[f, n]
   \;=\; \bigl(\Omega_n \cdot \hat n_f < -\epsilon\bigr),
   \qquad
   \mathrm{outflow\_mask}[f, n]
   \;=\; \bigl(\Omega_n \cdot \hat n_f > +\epsilon\bigr).

This mask is the discrete realization of :math:`\Gamma_\pm`. It is
the load-bearing primitive that downstream consumers need:

* **The SN realizer reads BOTH masks**, one per half-trace: the
  inflow indices
  (:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face`)
  give a law's **codomain** :math:`\Gamma_-` and are cross-checked
  against the face-name geometry (ERR-041); the outflow indices
  (:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_indices_for_face`)
  give its **domain** :math:`\Gamma_+`, restricted through the
  :math:`\gamma_+` operator the same table builds
  (:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_restriction`).
  Both are carried on the method space as the ``inflow_indices`` /
  ``outflow_indices`` **fields**, so a hand-built space can name its
  own half-traces without a whole trace space
  (:ref:`bc-domain-narrowing`).
* The universal invariant
  :meth:`~orpheus.geometry.boundary.BoundaryTraceLaw.assert_source_is_placeable`
  reads the inflow set as a **structural** check — it asks whether the
  realization can NAME :math:`\Gamma_-`, and raises
  :class:`~orpheus.geometry.boundary.BoundarySourceNotOnIncomingTraceError`
  (ERR-047) when a source-carrying law is realized against a space that
  cannot. ⭐ Until campaign phase **P6** it probed the source's *support* on
  the per-ordinate shape first, and a source that answered that probe with
  zeros **skipped the certification entirely** while still delivering. Since a
  spec now receives :math:`\Gamma_-(f)` itself, :math:`q \in \Gamma_-` holds
  by construction, the probe is retired, and the discriminator is whether the
  law carries a source at all (:class:`NoSource` or not) rather than what that
  source currently evaluates to. When the face CAN name it, the
  delivery guarantee is structural: since **B3.4a** the realizer sizes
  the source's block from :math:`|\Gamma_-|`, so :math:`q \in \Gamma_-`
  holds by typing (pre-B3.4a it held because the realized operator
  masked a full-face evaluation — see :ref:`bc-narrowing-b34a`).
* The SN curvilinear sweep (1-D spherical / cylindrical) consumes
  the same realizer-routed mask as the Cartesian path — Issue #188
  (C188.1+C188.2 in :mod:`orpheus.numerics.spaces.angular_trace_space`, C188.3 in
  :mod:`orpheus.sn.mesh.augmented_mesh`) lifted the curvilinear deferral and
  Issue #176 then dropped the legacy 2-arg shim that existed only
  to bridge that deferral.

The class :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`
carries the per-face :math:`\Omega\cdot\hat n` masks as
``Optional[np.ndarray]`` fields excluded from
:meth:`__eq__` and :meth:`__hash__` — preserving the
:class:`~orpheus.numerics.space.FunctionSpace` identity convention
``(name, shape)``, which for this axes-less class IS content identity
(its factory folds the quadrature and layout into an
``angular_trace#<digest>`` name) and which the 2026-09-07 identity flip
therefore leaves unchanged: the flip made identity structural only for
spaces that declare their ``axes``. Construction goes through the
classmethod factory
:meth:`AngularTraceSpace.from_quadrature_and_layout
<orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.from_quadrature_and_layout>`;
the bare dataclass constructor is reserved for trace spaces whose mask
will be populated later (or never).

**Geometry-blind, layout-driven (post Issue #225 / C5.3).** The factory
is **geometry-blind**: it takes only the angular quadrature and a
:class:`~orpheus.numerics.face_layout.FaceLayout` (canonically
:attr:`SNMesh.boundary_face_layout <orpheus.sn.mesh.augmented_mesh.SNMesh>`), and
reads every datum from those two — the layout's ``"{axis}{min|max}"``
face names imply axis-aligned outward normals, so the
:math:`\Omega\cdot\hat n` row for an axis-:math:`a` face is
:math:`\pm\mu_a` (sign from ``min`` / ``max``). It works for every
constructible :class:`~orpheus.geometry.mesh.Mesh1D` coord system
(``CARTESIAN`` / ``SPHERICAL`` / ``CYLINDRICAL`` — all share the
``("xmin", "xmax")`` radial-axis face structure, with the
:math:`\mu_x` of a :meth:`Quadrature.gauss_legendre
<orpheus.numerics.quadrature.Quadrature.gauss_legendre>` quadrature as
the direction cosine along that axis), for 2-D Cartesian, and — since
C5.5 — for 3-axis Cartesian. The factory has **no** geometry refusal:
the former ``mesh`` parameter (on the retired
``from_mesh_and_quadrature``) was gate-only and is gone (see
:ref:`sn-c5-geometry-blind-trace`). The 2-D cylindrical
(axisymmetric :math:`(r, z)`) case never reaches the factory because
such a :class:`~orpheus.geometry.mesh.Mesh2D` cannot become an
:class:`SNMesh` (no 2-D cylindrical SN sweep exists); the refusal lives
at the :class:`SNMesh` construction surface, not in the trace factory.


.. _bc-law-layer:

Boundary law (``BoundaryTraceLaw`` ABC + concretes)
===================================================

The base class :class:`~orpheus.geometry.boundary.BoundaryTraceLaw`
is an ``abc.ABC`` whose MRO is exactly ``[BoundaryTraceLaw,
RegistryMixin, ABC, object]``: it mixes in
:class:`~orpheus.numerics.registry.RegistryMixin` (so each concrete
subclass self-registers under its ``key=`` class-creation kwarg) and
nothing else. It does **not** inherit
:class:`~orpheus.numerics.operator.LinearOperator` — that inheritance
was dropped at Issue #186 / B3 + β2 (see
:ref:`bc-trace-law-descriptor-model`), and there is no ``@`` /
``__matmul__`` on the class. The dunders it does carry (``+``, ``-``,
``*``, ``/``, unary ``-``) are the **descriptor-tree** algebra: they
return :class:`LawSum` / :class:`LawScaled` nodes, never operators.
The ABC ships:

1. Three properties named for the :eq:`affine-bc-form` factors:
   ``geometry_map``, ``response_kernel``, ``source``. The ABC's
   defaults are ``None``, ``None``,
   :class:`~orpheus.geometry.boundary.NoSource`, but **all seven
   concrete laws populate all three** since campaign phase B1 — the
   two factor tiers are typed specifications
   (:class:`~orpheus.geometry.boundary.SelfPairedDeck` /
   :class:`~orpheus.geometry.boundary.PairedDeck` for :math:`G`;
   :class:`~orpheus.geometry.boundary.ScalarResponse` /
   :class:`~orpheus.geometry.boundary.LambertianReemission` for
   :math:`R`), never realized matrices. Measured on the live tree:

   .. list-table::
      :header-rows: 1
      :widths: 30 34 36

      * - law
        - ``geometry_map``
        - ``response_kernel``
      * - ``VacuumInflow``
        - ``SelfPairedDeck.identity()``
        - ``ScalarResponse(alpha=0.0)``
      * - ``ReflectiveBoundary``
        - ``SelfPairedDeck.mirror(axis)``
        - ``ScalarResponse(alpha=albedo)``
      * - ``WhiteBoundary``
        - ``SelfPairedDeck.identity()``
        - ``LambertianReemission(alpha, axis, outward_sign)``
      * - ``AlbedoBoundary``
        - ``SelfPairedDeck.identity()``
        - by closure (**B3.4b**): ``SpecularReemission(alpha, axis)`` /
          ``LambertianReemission(alpha, axis, outward_sign)`` /
          ``ScalarResponse(alpha=albedo)`` with none
      * - ``PeriodicBoundary``
        - ``PairedDeck.wrap(axis)``
        - ``ScalarResponse(alpha=1.0)``
      * - ``ZeroFluxBoundary``
        - ``SelfPairedDeck.identity()``
        - ``ScalarResponse(alpha=-1.0)``
      * - ``PrescribedInflow``
        - ``SelfPairedDeck.identity()``
        - ``ScalarResponse(alpha=0.0)`` (plus ``source``)

   They are **read by production**: phase B2.2 repointed five sites
   from string comparison to
   ``law_permutes_ordinates(law)`` (a function over BOTH tiers since
   **B3.4b**, when a specular pairing became legal in :math:`R`) /
   ``law.response_kernel.is_zero`` /
   ``law.response_kernel.amplitude``, and the diffusion realizer's
   whole law → :math:`\mathcal{A}` table collapsed to the single read
   ``law.response_kernel.amplitude``. The ABC keeps the ``None``
   defaults, so the diffusion realizer guards against a subclass that
   never populates them rather than dying on an ``AttributeError``.
2. Five **universal** ``assert_*`` invariants and three **specific**
   invariants on the BCs that need them — but four of the five
   universals are empty and two of those are overridden by nobody.
   See :ref:`bc-universal-invariants` for the measured inventory.
3. A :meth:`realize` hook that raises with guidance (route through a
   method realizer) — see :ref:`bc-realizer-layer-detail`.
4. **No ``apply`` method at all** (Issue #186 / B3 + β2,
   2026-05-11). The descriptor model that survived the C176.3
   Option A interim was retired in favour of a pure-descriptor
   contract: :class:`BoundaryTraceLaw` is no longer a
   :class:`~orpheus.numerics.operator.LinearOperator` subclass,
   no concrete law carries ``apply`` / ``apply_transpose``
   methods, and none reports the operator predicates
   ``is_invertible`` / ``is_adjointable``.
   The §16A.3 three-layer split (descriptor / realizer / operator)
   is now enforced by the **type system**, not by convention —
   ``law.apply(psi)`` on a raw law is an ``AttributeError`` at
   runtime and a static-type error at the linter level. The
   :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` is the
   sole bridge from descriptor to callable; see
   :ref:`bc-trace-law-descriptor-model` for the design rationale
   and the predecessor approaches that were tried and rejected.

Seven concrete laws ship under :mod:`orpheus.geometry.boundary`,
one per submodule. The Grand Report v3 vocabulary is used verbatim
for the class names; the pre-refactor names were deprecated aliases
only until Wave O step O.4a.1 retired them, and none is importable
now (see :ref:`bc-naming-audit` for the historical index).

The table below covers the **six affine trace laws** — those that
decompose into the :eq:`affine-bc-form` factors
:math:`(G_\alpha, R_\alpha, q)`. The seventh,
:class:`~orpheus.geometry.boundary.ZeroFluxBoundary`, is deliberately
absent: :math:`\phi_\Gamma = 0` is a *relation* between the two
traces, :math:`A_-\gamma_- + A_+\gamma_+ = 0`, not a map from one to
the other, so it has no :math:`(G, R)` pair to tabulate. It rides the
same ABC as a pragmatic placement — the diffusion realizer collapses
it to the albedo-family :math:`\mathcal{A} = -1` (outside the
sub-Markov range by construction) and the SN realizer refuses it
outright, with a hand-written ``isinstance`` guard rather than a type
distinction. Giving the relation tier its own type is issue **#177**.

.. list-table:: Concrete ``BoundaryTraceLaw`` subclasses
   :header-rows: 1
   :widths: 17 14 18 11 11 29

   * - Class
     - Registry key
     - :math:`G_\alpha`
     - :math:`R_\alpha`
     - :math:`q`
     - Trace-edge family
   * - :class:`~orpheus.geometry.boundary.VacuumInflow`
     - ``"vacuum"``
     - identity (fixes no geometry)
     - 0
     - 0
     - **none** — the inflow is data
   * - :class:`~orpheus.geometry.boundary.ReflectiveBoundary`
     - ``"reflective"``
     - axis-reflection permutation
     - albedo
     - 0
     - same-face back-edge, mirror-partner map
   * - :class:`~orpheus.geometry.boundary.WhiteBoundary`
     - ``"white"``
     - identity (fixes no geometry)
     - albedo × cosine-weighted hemispheric average (Lambertian) —
       **rank-one**
     - 0
     - same-face back-edge, all-to-all on the face
   * - :class:`~orpheus.geometry.boundary.PeriodicBoundary`
     - ``"periodic"``
     - spatial wrap along ``axis``; the realizer derives the partner
       face from the installation face
     - 1
     - 0
     - opposite-face **pair**, mutually feeding
   * - :class:`~orpheus.geometry.boundary.AlbedoBoundary`
     - ``"albedo"``
     - identity (fixes no geometry)
     - albedo — **magnitude only**; the angular re-emission closure is
       the gap (see the census note below)
     - 0
     - same-face back-edge for :math:`\alpha \neq 0` (degenerates
       to none at :math:`\alpha = 0`); not yet modelled in
       ``sweep_acyclicity``
   * - :class:`~orpheus.geometry.boundary.PrescribedInflow`
     - ``"prescribed_inflow"``
     - identity (fixes no geometry)
     - 0
     - :math:`q \in \Gamma_-`
     - **none** — the inflow is data

The last column is the **structure** each law contributes to the
:math:`(\text{face}, \text{ordinate})` trace digraph, which *is*
intrinsic to the law. Whether the resulting digraph carries a
**cycle** is not: that depends on the other faces, and only the
strongly-connected-component decomposition described under
:ref:`bc-sweep-cycle` can answer it. A same-face back-edge is a
*forward* edge when it is the only one; it closes a loop only when
a second law feeds back the other way. The opposite-face pair is
the exception — periodic closes a loop from a single law.

Per-law rank census: vacuum and prescribed inflow carry no response
kernel (rank-0) — the identity in their :math:`G_\alpha` column is the
*deck* identity, i.e. "this law fixes no geometry", and the vanishing
is :math:`R`'s. White is **rank-1 in angle** — one isotropic re-entry
mode, fed by the cosine-weighted (Lambertian) outflow average, written
to every inflow ordinate; the whole of it lives in :math:`R_\alpha`.
Reflective is an **angular permutation** (the axis reflection, scaled
by the albedo) — rank :math:`N/2` per slab face: structured,
trace-sized, NOT rank-1.

.. note:: **Albedo's angular closure was the one genuine gap** — the
   *closure* landed at campaign phase **B3.4b**; wiring the law into
   the SN law-admission registry is still open (issue **#189**; the
   registry admits ``{vacuum, reflective}`` today). Its
   :math:`R_\alpha` stated the *magnitude* with which flux returns but
   not the *distribution*, and on an angular trace those are
   independent — the scalar is complete only on a scalar
   (partial-current) trace, where the distribution has no degrees of
   freedom. The closure makes
   :math:`\text{albedo}(\alpha, \text{isotropic}) \equiv
   \text{white}(\alpha)` and
   :math:`\text{albedo}(\alpha, \text{specular}) \equiv
   \text{reflective}(\alpha)` **theorems** rather than coincidences —
   in the code both routes execute *one* realization body rather than
   two transcriptions agreeing.

   **Retraction (2026-08-04).** This note used to forecast that the
   specular closure would move its content across into
   :math:`G_\alpha`, "which is exactly what the
   :ref:`membership criterion <bc-factor-roles>` predicts". That was
   **backwards**: it read the criterion's *necessary* half
   (multiplicativity — which a permutation passes) as the whole test.
   The *sufficient* half is the quotient test, and a polished wall is
   not a quotient of the domain, so its specular return is
   **constitutive**. The 2026-08-01 ruling and the shipped code both
   put it in :math:`R_\alpha`
   (:class:`~orpheus.geometry.boundary.SpecularReemission`), with
   ``AlbedoBoundary.geometry_map`` staying
   :meth:`~orpheus.geometry.boundary.SelfPairedDeck.identity`
   **unconditionally** — whatever the closure.

   What blocked albedo's **domain narrowing** is unchanged and still
   reads correctly: without a closure :math:`R = \alpha\,I` is a
   :math:`\Gamma_+ \to \Gamma_+` endomorphism and albedo's
   :math:`G = \mathrm{id}` supplies no crossing, so there is nothing
   to narrow *onto* — which is why the SN realizer **refuses** the
   closure-free spelling on an angular trace.

Periodic is a **spatial pushforward** pairing opposite faces. Marshak /
partial-current boundaries are rank-N via the
**descriptor-tree algebra** on the unrealised laws (:class:`LawSum`
/ :class:`LawScaled` over :class:`BoundaryTraceLaw` leaves) —
realised once per face by
:func:`~orpheus.geometry.boundary.realize_recursively`; see
:ref:`bc-rank-n-algebra` below. Every one of these kernels is
**trace-sized** — tiny against the bulk dimension — which is
exactly the low-rank/structured shape a Woodbury boundary closure
exploits (:ref:`the scoped SMW statement <smw-low-rank-exception>`,
Issue #300).


.. _bc-naming-audit:

Naming audit: pre-refactor vs Grand Report v3 vocabulary
--------------------------------------------------------

Wave 7 of the refactor renamed every concrete BC to match the
Grand Report v3 vocabulary verbatim. During the deprecation window
the pre-refactor names were re-exported as deprecated aliases from
:mod:`orpheus.geometry.boundary.__init__` so existing import sites
kept working unchanged. Those aliases were **retired in Wave O step
O.4a.1** once every code and test consumer had migrated; the
canonical names in the middle column are now the sole importable
symbols. The table is retained as the historical naming index for
readers tracing pre-Wave-O commits:

.. list-table:: Wave 7 BC renames (pre-refactor name → canonical name)
   :header-rows: 1
   :widths: 35 35 30

   * - Pre-refactor name (retired Wave O O.4a.1)
     - Canonical name
     - Why renamed
   * - ``VacuumBoundaryOperator``
     - :class:`~orpheus.geometry.boundary.VacuumInflow`
     - Emphasizes "inflow set to zero", not "operator that vacuums";
       distinguishes from the rank-N case
       :class:`PrescribedInflow` which also writes only the inflow
       trace.
   * - ``SpecularBoundaryOperator``
     - :class:`~orpheus.geometry.boundary.ReflectiveBoundary`
     - "Specular" is one specific axis-aligned reflection;
       "Reflective" is the family name that the Grand Report uses.
       A future ``SymmetryBoundary`` (deferred) will share the
       reflective-family base but apply on non-physical octant
       boundaries.
   * - ``WhiteBoundaryOperator``
     - :class:`~orpheus.geometry.boundary.WhiteBoundary`
     - Drops the redundant "Operator" suffix that pre-dated the
       law / realizer split. The law is no longer "the operator";
       it's the abstract physical statement that gets *realized*
       to an operator.
   * - ``PeriodicBoundaryOperator``
     - :class:`~orpheus.geometry.boundary.PeriodicBoundary`
     - Same rationale: pre-refactor "Operator" suffix is
       structurally misleading.
   * - ``AlbedoBoundaryOperator``
     - :class:`~orpheus.geometry.boundary.AlbedoBoundary`
     - Same rationale.
   * - ``MixedBoundaryOperator``
     - **retired Wave 11** — see :ref:`bc-rank-n-algebra`.
     - Replaced by the descriptor-tree algebra
       (:class:`LawSum` / :class:`LawScaled` over
       :class:`BoundaryTraceLaw` leaves; Issue #186 / B3 + β2);
       the dedicated class added no value over the inherited
       algebra dunders.


.. _bc-sweep-cycle:

Sweep cycles: a configuration property, not a per-law flag
-----------------------------------------------------------

The SN sweep visits cells in a **topological order** of the directed
cell-visit graph, whose edges are oriented by
:math:`\mathrm{sign}(\Omega_n \cdot \hat n_f)`. (The production code
does not *run* a topological sort to find that order — it is closed-form
index arithmetic, ``level_of = local.sum(axis=0)`` in
:mod:`orpheus.sn.loss_representation.sweep_graph` — which is precisely
why nothing in the solver ever builds a digraph and nothing in the
solver can detect a cycle.) For most BCs the boundary is the *root*
of that order — inflow values come from the BC, get propagated
through the cells, and exit as outflow values that the BC consumes
but doesn't feed back. For two BC families this is no longer true:

* **Reflective.** The outflow flux at a face is mapped to an inflow
  flux at the **same** face (under the reflection permutation), so
  the law adds a *trace back-edge*: an outflow slot feeds an inflow
  slot. Whether that edge closes a **cycle** depends on the rest of
  the configuration — see the warning below.
* **Periodic.** The edge spans two different faces (outflow at face
  A maps to inflow at face B, **and vice versa**). Because the pair
  of edges is mutually feeding, periodic closes a cycle from a
  *single* law — which is why an :class:`SNMesh` refuses it outright
  (:attr:`SNMesh.BOUNDARY_OPERATOR_REGISTRY` admits only ``vacuum``
  and ``reflective``).

.. warning::

   **A reflecting face does not, by itself, create a sweep cycle.**
   The proposition "reflective ⟹ the sweep DAG acquires a cycle" is
   *false as stated*, and it is worth stating precisely because the
   opposite reading would forbid something ORPHEUS actually does: the
   curvilinear :math:`r = 0` pole mirror is a specular reflection kept
   **inside** the walk, certified lower-triangular, because the sweep
   visits :math:`\mu < 0` before :math:`\mu > 0` and the mirror feeds
   strictly downstream.

   A cycle requires a closed **loop** — e.g. *both* faces of a slab
   reflecting, so the left mirror feeds the :math:`\mu>0` sweep and
   the right mirror feeds it back. The honest criterion is therefore a
   **strongly-connected-component decomposition** of the
   :math:`(\text{face}, \text{ordinate})` trace digraph, not a boolean
   on the boundary *kind*.

   This is executable, not editorial:
   :mod:`orpheus.derivations.discrete.sn.sweep_acyclicity` is the
   algebra of record, and
   ``tests/sn/sweep/test_sweep_acyclicity.py`` gates it. Measured on
   an S\ :sub:`4` slab:

   .. csv-table::
      :header: left, right, acyclic?, why
      :widths: 16, 16, 12, 56

      vacuum, vacuum, **yes**, no boundary edge at all
      reflective, vacuum, **yes**, one mirror — a forward edge
      vacuum, reflective, **yes**, the mirror image of the above
      reflective, reflective, **no**, "2 SCCs, one per mirror pair"

   A second distinction the same module makes, easy to conflate with
   the first: acyclicity says *some* one-pass order exists, not that a
   **given** one does. A left-reflecting slab is one-pass in the
   :math:`\mu<0`-first order; a right-reflecting slab is equally
   acyclic but needs :math:`\mu>0` first. Triangularity is a property
   of an (operator, **order**) pair.

What the trace digraph replaced: the retired per-law flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   **Retired 2026-07-30.** Until then
   :class:`~orpheus.geometry.boundary.BoundaryTraceLaw` carried a
   ``creates_sweep_cycle`` ``ClassVar``, defaulting to ``False`` and
   declared explicitly on three laws: ``True`` on
   :class:`ReflectiveBoundary` and :class:`PeriodicBoundary`,
   ``False`` (redundantly) on
   :class:`~orpheus.geometry.boundary.PrescribedInflow`; the other
   four laws simply inherited the default. It has been removed from
   the ABC, from those three laws, and from the tests that asserted
   its values. The archaeology is kept here because the *shape* of
   the mistake recurs, not because the flag might come back.

Three findings retired it, in increasing order of importance.

1. **It had zero production readers.** The §15A.2 sweep-cycle
   detector it was designed to feed was never built, and the
   invariant named for that job, ``assert_cycles_are_declared``, was
   likewise never implemented. The flag was purely declarative for
   its whole life: a value asserted by tests and read by nothing.

2. **The claim attached to it was false.** Its docstring asserted
   "reflective ⟹ the sweep DAG acquires a cycle" — the proposition
   the warning above refutes. A reflecting face adds a trace
   *back-edge*; a **cycle** needs a closed *loop*.

3. **It could not have worked even in principle** — and this is the
   part worth carrying forward. Whether a face's back-edge closes a
   cycle depends on the **whole face configuration**, not on the
   kind of any one law. The measured truth table above makes the
   point in one line: ``reflective|vacuum`` is acyclic,
   ``reflective|reflective`` is not, and the flag reads ``True`` on
   *reflective* in both. Meanwhile ``periodic|vacuum`` **is** cyclic
   from a single law. So the one value ``True`` was carrying two
   structurally different facts — "this law can take part in a loop
   that *other* faces may close" (reflective) and "this law closes a
   loop by itself" (periodic) — which is the tell that the property
   does not live on the law at all. A ``ClassVar`` on the law
   *class* is evaluated once, at class-creation time, with no
   knowledge of the mesh or of the opposite face; there is no value
   it could have held that would be correct.

The general design rule, which is the reason to keep this record:

.. admonition:: A law may carry only what is intrinsic to it
   :class: tip

   A boundary law is a **descriptor of one face**. It can honestly
   declare its own algebraic content — its :math:`G_\alpha`,
   :math:`R_\alpha`, :math:`q`, and the *family* of trace edge it
   contributes (see the
   :ref:`concrete-subclass table <bc-law-layer>`). It cannot declare
   a property of the *configuration* it will be placed in.
   Sweepability, triangularity, and cycle-freedom are all
   configuration properties — the same relinquishment recorded for
   operator names in
   :mod:`orpheus.derivations.discrete.sn.sweep_acyclicity`: an
   operator's name may state what the object **is**, never that it
   sweeps, because a new mesh or a new opposite face can falsify the
   latter without touching the object.

The honest replacement is not a flag but a computation, and it
already exists: build the trace digraph for the *configuration* and
decompose it. Vacuum and prescribed inflow are the only laws that
are unconditionally cycle-free, and for a structural reason — they
supply the inflow as **data**, contributing no trace edge at all, so
they cannot participate in any loop. Reflective, white, and albedo
(at :math:`\alpha \neq 0`) each add a same-face back-edge: harmless
alone, loop-closing when a second such law faces it (``white|white``
is cyclic for exactly the reason ``reflective|reflective`` is —
the coupling is all-to-all rather than mirror-partnered, but it
still feeds an inflow slot from an outflow slot on the same face).
Periodic pairs opposite faces mutually and so is the one law that is
cyclic on its own — which is why :class:`SNMesh` refuses it outright.

The gates on that computation are ``@pytest.mark.foundation``: the
claim is a software/structural invariant of a discrete construction,
not an equation claim, so the tests carry no ``verifies(...)`` marker
(the ``verifies`` ⊥ level doctrine). What they pin is the SCC
decomposition itself, the acyclic ⟹ lower-triangular certificate, and
— as mutation teeth — that dropping the boundary edge *falsely*
certifies a cyclic configuration as acyclic.


.. _bc-realizer-layer-detail:

Realizer (``BoundaryRealizer`` Protocol + ``SNBoundaryRealizer``)
=================================================================

The :class:`~orpheus.geometry.boundary.BoundaryRealizer` Protocol is
``@runtime_checkable`` and lives at
:mod:`orpheus.geometry.boundary._realizer`. Its contract is one
attribute and one method:

.. code-block:: python

   @runtime_checkable
   class BoundaryRealizer(Protocol):
       method_name: str
       def realize(
           self,
           law: BoundaryTraceLaw,
           method_space: Any,
       ) -> LinearOperator: ...

The Protocol intentionally does *not* prescribe how
:meth:`realize` dispatches over law types — different methods will
have different optimal dispatch strategies. The SN realizer uses
``isinstance`` because the law set is small and stable; a future
realizer that needs runtime extension might use the
:class:`~orpheus.numerics.registry.RegistryMixin` machinery instead.

The Wave 5 SN dispatch table is the documented standard — the §15.2
:math:`G_\alpha` geometric primitives, one per boundary law:

.. _bc-tensor-primitives:

.. list-table:: SN realization map (law → Wave-0 / Wave-1 primitive)
   :header-rows: 1
   :widths: 24 38 38

   * - Law
     - Realized representation (α = 1 fast path)
     - Realized representation (α ∉ {0, 1})
   * - :class:`VacuumInflow` — **narrowed**
     - the **zero map** :math:`\Gamma_+ \to \Gamma_-`: a
       :class:`~orpheus.numerics.operator.ZeroOperator` whose
       ``codomain_zero`` hook emits :math:`|\Gamma_-|` rows and whose
       ``transpose_zero`` hook emits :math:`|\Gamma_+|` rows. The
       symmetric space hooks are load-bearing: relying on the
       endomorphic ``0.0 * x`` echo would be right only by accident
       (:math:`|\Gamma_+| = |\Gamma_-|` on every reachable fixture — a
       coincidence, not a contract).
     - n/a (vacuum has no α parameter)
   * - :class:`ReflectiveBoundary(axis, α)` — **narrowed** (B3.4a),
       **bound** (G6.3 step 5)
     - ``PermutationOperator(local_perm, axis=0,
       domain=Γ₊(f), codomain=Γ₋(f)) & IdentityOperator()``
       — a 2-factor
       :class:`~orpheus.numerics.operator.TensorProductOperator` on the
       **reduced** ordinate axis, with
       ``local_perm = Γ₊(f).to_local(π⁻¹[inflow])`` where :math:`\pi` is
       the mirror's derived ordinate permutation
       (:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`;
       :math:`\pi^{-1} = \pi` for a mirror) and the remap is the
       half-trace SPACE's own (G6.5 — the space owns its row order, and
       the deck arm consequently REQUIRES the bound spaces, so its
       output is always fully bound).
       Row :math:`j` reads the mirror of the :math:`j`-th inflow
       ordinate *at that ordinate's position inside* :math:`\Gamma_+`;
       ``to_local`` is mandatory because a slab mirror **reverses**
       order. The ERR-045 inflow → outflow invariant is *consumed*
       here rather than re-assumed — ``to_local`` raises if the mirror
       sent an inflow ordinate anywhere but :math:`\Gamma_+`. The
       binding makes this the **one-link** case of the white arm's chain
       (:ref:`bc-deck-length-one-chain`) and costs zero arithmetic
       — `[M]` ``apply`` and ``apply_transpose`` bit-identical to the
       unbound build on all five shipped quadratures. ⚠ **Read that as
       the two verbs it names, not as "inert".** ``.H`` is a THIRD verb
       and the binding does change it — from the Euclidean transpose to
       the metric-weighted Hilbert adjoint
       (:ref:`tensor-product-spaces`). It happens to be bit-identical
       *for the mirror*, whose metric cancels, and is **87 %** away for
       the Lambertian row below.
     - ``ScaledOperator(α, <that TP>)``
   * - :class:`WhiteBoundary(axis, outward_sign, α)` — **narrowed**
       (B3.4a), **factored** (G6.3 step 3b)
     - ``(IsotropicEmissionOperator(...) @ PartialCurrentOperator(...))
       & IdentityOperator()`` — the Lambertian kernel as a two-link
       chain :math:`\Gamma_+ \to S(f) \to \Gamma_-`: the first link
       contracts over :math:`\Gamma_+` to the outgoing partial current,
       the second broadcasts it over
       :math:`\Gamma_-`. Both half-traces come from the single
       face-name :math:`\to` signed-projection primitive classified
       against ``TANGENTIAL_EPS``, so the pre-B3.4a private
       ``> 0.0`` outflow test — a second classifier that disagreed with
       the trace space on any quadrature carrying tangential ordinates
       — is gone. The law's declared ``axis`` / ``outward_sign`` is
       cross-checked against the installation face's :math:`\Gamma_+`
       (index SETS, not sizes) before construction.
     - ``ScaledOperator(α, <that TP>)``
   * - :class:`AlbedoBoundary(α)` with α=0 — **not yet narrowed**
     - :class:`~orpheus.numerics.operator.ZeroOperator` (bare, so an
       endomorphism — it carries no space hooks)
     -
   * - :class:`AlbedoBoundary(α)` with α=1 — **not yet narrowed**
     - :class:`~orpheus.numerics.operator.IdentityOperator` (an
       endomorphism by definition)
     -
   * - :class:`AlbedoBoundary(α)` with α ∉ {0, 1} — **not yet
       narrowed**
     -
     - ``ScaledOperator(α, IdentityOperator() & IdentityOperator())``
   * - :class:`PeriodicBoundary` — **narrowed** (B3.4c); arrow derived
       (G6.3 step 7)
     - ``PermutationOperator(arange) & IdentityOperator()``, bound
       :math:`\Gamma_+(f') \to \Gamma_-(f)` and consuming the
       PARTNER face's :math:`\Gamma_+`. This is the one law whose domain
       is a different face, and that IS what makes it a quotient rather
       than a wall: :meth:`PairedDeck.domain_face <orpheus.geometry.boundary.PairedDeck.domain_face>` names the partner and
       :attr:`SNBoundaryOperator._face_domains` supplies it, so the
       pushforward the spec names lives in the CHANNEL and the action on
       the trace is the identity relabelling between two DISTINCT index
       sets — the ordinate permutation the wrap MOTION induces, through
       the same kernel as the mirror's. Earned, not assumed — the
       kernel's membership and size checks certify
       :math:`\Gamma_+(f') \equiv \Gamma_-(f)` (opposite outward
       normals) by construction. Issue **#183** was exactly this gap and
       closes here; the ``PeriodicWrapOperator`` type retired with it,
       and step 7 retired the unbound-identity stand-in (an endomorphism
       :math:`V \to V` can never be an isomorphism between two different
       spaces — the one link the composability check could not police)
     - n/a (periodic has no α parameter)
   * - :class:`PrescribedInflow(source)` — the rank-0 **affine** law;
       **narrowed** (B3.4a), **collapsed** (P3)
     - the **zero map** :math:`\Gamma_+ \to \Gamma_-` — literally the
       same :class:`~orpheus.numerics.operator.ZeroOperator` expression
       the vacuum row above builds, from the same
       ``_narrowed_zero_operator`` body, stamped
       :attr:`~orpheus.numerics.operator.BlockRole.BOUNDARY` like every
       other law. The law stays affine; this tier realizes its LINEAR
       factor, and for prescribed inflow that factor is zero, so
       **vacuum and prescribed inflow differ only in** :math:`q` — which
       travels the boundary-source channel instead
       (:ref:`bc-affine-source-channel`). Until **P3** this arm returned
       an ``IncomingSourceOperator`` whose ``apply`` ignored the
       outgoing flux and asked the source spec to fill
       ``(|Γ₋|,) + psi_out.shape[1:]`` — an AFFINE map in a linear slot,
       measured :math:`\lVert B(0) \rVert_\infty = 2.5` at
       :math:`q = 2.5`; the spec is now evaluated at that same shape by
       :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.from_specs`
       one tier out. The dense inflow **mask** had already dissolved
       with the codomain at B3.4a, so :math:`q \in \Gamma_-` holds by
       TYPING rather than by an erasure (ERR-047), exactly as vacuum's
       projector dissolved at B3.2. Still unreachable from a ``BC`` tag
       in production (the SN registry admits only
       ``{vacuum, reflective}``, #189) — the law is declarable only by
       constructing it directly.
     - n/a

.. note::

   **Four of the seven laws are narrowed; the rows flagged not yet
   narrowed are albedo and periodic.** Since campaign phase **B3.2** a
   realized SN law is typed :math:`\Gamma_+ \to \Gamma_-`
   (:ref:`bc-domain-narrowing`). B3.2 landed ``vacuum`` and
   ``reflective``; **B3.4a** landed ``white`` and ``prescribed_inflow``
   (:ref:`bc-narrowing-b34a`). The remaining rows still emit
   full-:math:`N` endomorphisms, are unreachable in production, and are
   pinned by strict xfails; each is blocked on a **design ruling** —
   **B3.4b** must give albedo an explicit re-emission closure in
   :math:`R` (its :math:`G` supplies no crossing), and **B3.4c** must
   build periodic's partner-face :math:`G` (#183, #189). Note that a
   shape assertion cannot tell the two typings apart —
   :math:`|\Gamma_+| = |\Gamma_-|` on every quadrature × face in the
   tree — so read the *declared spaces*, not the output shape.

The α = 1.0 fast paths return the **bare** primitive (no
``ScaledOperator`` wrap). This is load-bearing for bit-identity:
without it, the "perfect reflection" case
:class:`~orpheus.geometry.boundary.ReflectiveBoundary` (pre-refactor
``SpecularBoundaryOperator(axis="x", albedo=1.0)``) would shift by
one ULP under the realizer relative to its pre-refactor
``np.take(psi_out, reflection_index, axis=0)`` body — see
:ref:`bc-numerical-evidence` for the bit-equivalence pin (whose
``specular_x_lebedev17`` row is still ``assert_array_equal``, now
against the mirror isometry rather than a recording). The narrowing
preserved that bit-identity and was **gated against the retired
expression, not against the new code called twice**: the reference is
materialised in numpy off the law *descriptor*
(``α·np.take(ψ, mirror_partner, 0)[inflow]``, the partner map from the
independent geometric reference in ``tests/_harness/references.py``,
transpose via ``argsort``), compared with
``np.array_equal`` over slab-asymmetric, slab-symmetric, sphere,
cylinder ``product(2,4)`` and 2-D Cartesian ``level_symmetric(4)``.
The gate was falsified independently — forcing the naive ``arange``
remap reds 6 rows across slab and sphere in both directions, with a
positive control confirming 26 interceptions.

The :class:`~orpheus.sn.mesh.method_space.SNMethodSpace` dataclass is the
realizer's second argument. It carries:

* :attr:`~orpheus.sn.mesh.method_space.SNMethodSpace.quadrature` — the
  angular quadrature (mandatory).
* :attr:`~orpheus.sn.mesh.method_space.SNMethodSpace.face` — the
  face-name label (``"xmin"`` … ``"zmax"``) so the vacuum branch can
  look up the right inflow indices. (The pre-C4 ``"left"`` / ``"right"``
  spellings were aliases of ``"xmin"`` / ``"xmax"``; since the C4
  face-name carve — :ref:`bc-face-name-carve` — every face is keyed by
  its canonical ``"{axis}{min|max}"`` name.)
* :attr:`~orpheus.sn.mesh.method_space.SNMethodSpace.inflow_indices` —
  the per-face inflow ordinate indices: a realized law's **codomain**
  :math:`\Gamma_-` (derived from the held trace at :meth:`for_face`
  time).
* :attr:`~orpheus.sn.mesh.method_space.SNMethodSpace.outflow_indices` —
  its **domain** :math:`\Gamma_+`, added at campaign phase **B3.2** as
  the sibling of ``inflow_indices``. The codomain has been a *field*
  precisely so a hand-built space can name its own trace without a
  whole trace space; a law's domain deserves the same. The realizer
  raises a :class:`~orpheus.geometry.boundary.BoundaryError` naming
  :meth:`for_face` when a narrowed law is realized without it.
* :attr:`~orpheus.sn.mesh.method_space.SNMethodSpace.mesh`,
  :attr:`~orpheus.sn.mesh.method_space.SNMethodSpace.trace` — the
  (optional) spatial mesh and the single unified
  :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` for any
  realizer branch that needs more than the per-face index list. Since
  the #205 / #201 unification this is **one** ``trace`` attribute, not a
  separate ``inflow_trace`` / ``outflow_trace`` pair — inflow and
  outflow are selectors over the same trace space (see the
  one-space-two-selectors note above). The ``mesh`` slot is optional
  metadata (C5.3, #225): nothing in the realizer chain reads it (the
  inflow indices come from the trace), and an axis-native ``SNMesh``
  with no legacy mesh adapter passes ``None``.

The :meth:`SNMethodSpace.for_face` factory is the standard
construction site inside ``SNMesh.realize_boundary_law`` (per face,
driven by the shared
:func:`~orpheus.transport.method.resolve_boundary_conditions` body,
#290 P7b); the :meth:`SNMethodSpace.minimal` factory returns a
quadrature-only method space for unit tests that don't need mesh +
face metadata. **Since B3.2** ``minimal`` is a *partial* constructor:
a quadrature alone cannot name a particular face's :math:`\Gamma_+`,
so it no longer suffices for a narrowed law. B3.4a widened that from
two laws to four — only ``albedo`` and ``periodic`` still realize from
a ``minimal`` space, and precisely because they are the two laws still
awaiting narrowing. After B3.4b / B3.4c it will realize nothing at all
— a retirement candidate, not a fixture.


.. _bc-dual-registry:

The law registry (and the realizer registry that was dissolved)
===============================================================

ONE registry connects the tag layer to the law layer:

**Law registry** — keyed by ``BC.kind`` string
(``"vacuum"``, ``"reflective"``, ``"white"``, ``"periodic"``,
``"albedo"``, ``"prescribed_inflow"``, ``"zero_flux"``). The
registry IS :attr:`BoundaryTraceLaw.registry` (a class-level dict
maintained by :class:`~orpheus.numerics.registry.RegistryMixin`).
Concrete laws self-register at module import time via the
``key=`` class-creation kwarg:

.. code-block:: python

   class VacuumInflow(BoundaryTraceLaw, key="vacuum"):
       ...

Lookup is :meth:`BoundaryTraceLaw.create("vacuum")` or direct
dictionary access ``BoundaryTraceLaw.registry["vacuum"]``. Each
method-mesh additionally carries its own **admission table**
(``BOUNDARY_OPERATOR_REGISTRY: dict[str, type[BoundaryTraceLaw]]``
— the subset of laws its realizer can honestly realize, e.g.
``zero_flux`` is diffusion-only), and the shared
:func:`~orpheus.transport.method.resolve_boundary_conditions` body
uses THAT table to recover the law class from a mesh-declared
:class:`~orpheus.geometry.mesh.BC` — an unsupported tag refuses at
phase-space construction with the method's supported list.

**The realizer registry was dissolved at #290 P7b.** The Grand
Report §16A.11 design (lines 3252–3257) paired the law registry
with a second, method-name-keyed ``BoundaryRealizerRegistry``
(``"SN"``, ``"MoC"``, …; realizers self-registered via a decorator
at import time). It shipped in Wave 5 and was retired when the
second functional realizer (diffusion, #290 P3) made the real
consumption pattern visible: **no consumer ever resolved a realizer
by method name**. Production holds a method-mesh, and the mesh's
``realize_boundary_law`` arm — the per-method hook of the
:class:`~orpheus.transport.method.TransportMethod` Protocol —
instantiates its own realizer directly; the rank-N walker takes the
realizer as an explicit argument. The string indirection carried a
real hazard class for zero payoff: a registry populated by import
side-effects is EMPTY in a fresh process until the right module
happens to be imported, a timing miss invisible to in-suite tests
(process-global state masks it). Dissolving the registry deleted
the hazard class instead of gating it.

The two extension axes the dual-registry design named survive
without it:

* Adding a new BC type means adding one
  :class:`BoundaryTraceLaw` subclass with a ``key=`` registration
  (+ the admission-table entries of the methods that support it).
  Each existing realizer adds a dispatch branch; no existing law
  changes.
* Adding a new transport method means minting one method-mesh
  (structurally conforming to ``TransportMethod``), one
  :class:`BoundaryRealizer`, and one admission table. **M** realizer
  branches need to be implemented (one per admitted BC), but no
  existing law changes — and no central registration step exists.


.. _bc-cross-method-stubs:

Adopting the architecture in a new method (and the retired stubs)
-----------------------------------------------------------------

A method adopts the unified BC architecture by shipping three
pieces (the diffusion adoption at #290 P3–P7b is the worked
example):

1. a **method-mesh** (``DiffusionMesh(MaterialMesh)``) carrying the
   method's trace + an admission table + the
   ``realize_boundary_law`` arm — it conforms structurally to
   :class:`~orpheus.transport.method.TransportMethod` and gets the
   whole tag → law → realized-``bc``-dict pipeline from the shared
   :func:`~orpheus.transport.method.resolve_boundary_conditions`
   body;
2. a **realizer** (``DiffusionBoundaryRealizer``) mapping each
   admitted law onto the method's operators;
3. a **method space** (``DiffusionMethodSpace``) carrying whatever
   discretization metadata the realizer reads.

.. note::

   **Historical: the Wave-5 stub scaffolding (retired #290 P7b).**
   Between Wave 5 and #290, ``MoCBoundaryRealizer`` /
   ``MCBoundaryRealizer`` / ``CPBoundaryRealizer`` (and, until #290
   P3, ``DiffusionBoundaryRealizer``) existed as
   ``NotImplementedError`` stubs auto-registered by each method's
   ``__init__.py``, "holding the dispatch architecture in place."
   With the registry dissolved there is no dispatch table to hold a
   place in — a stub realizer that cannot realize anything serves no
   consumer — so the three stub modules, their auto-import lines,
   and their stub-invariant tests were deleted. MoC / MC / CP keep
   their legacy solver-side BC validation until each adopts the
   architecture per the recipe above.

The SN realizer is **not** auto-imported by ``orpheus.sn.__init__``
(it's a heavy module that every SN consumer pays for); the
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` construction imports
it explicitly when it needs it.


.. _bc-worked-example:

Worked example — end to end
===========================

The following walks the
``BC("vacuum") → VacuumInflow → SNBoundaryRealizer.realize →
ZeroOperator(Γ₊ → Γ₋)`` chain that
:meth:`orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law`
performs per face (driven by the shared
:func:`~orpheus.transport.method.resolve_boundary_conditions` body) at
SNMesh construction time. The example uses a 1-D Cartesian slab; the same
chain runs on Mesh2D with face labels ``xmin`` / ``xmax`` /
``ymin`` / ``ymax``.

Step 1 — declaration on the mesh
--------------------------------

The user declares the vacuum BC on the mesh's left face:

.. code-block:: python

   from orpheus.geometry.mesh import Mesh1D, BC

   mesh = Mesh1D(
       edges=np.linspace(0.0, 1.0, 11),
       mat_ids=np.zeros(10, dtype=int),
       coord=CoordSystem.CARTESIAN,
       bc_left=BC("vacuum"),
       bc_right=BC("reflective"),
   )

The :class:`~orpheus.geometry.mesh.BC` dataclass is a thin wrapper
``BC(kind: str, params: dict)`` with no SN-specific knowledge. The
mesh is method-agnostic.

Step 2 — law resolution (in ``SNMesh.__init__``)
------------------------------------------------

When :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` is constructed against the
mesh, the shared
:func:`~orpheus.transport.method.resolve_boundary_conditions` body
walks the four (1-D: two) faces, calling
:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law`
per face:

.. code-block:: python

   law_cls = SNMesh.BOUNDARY_OPERATOR_REGISTRY["vacuum"]
   # law_cls is VacuumInflow  (registry key -> law class lookup)
   law = law_cls()
   # law is a zero-arg instance: VacuumInflow has no parameters

The :attr:`SNMesh.BOUNDARY_OPERATOR_REGISTRY` is the SN-side view
of the law registry; today it carries only ``"vacuum"`` and
``"reflective"`` because those are the only kinds the SN sweep
pipeline has been wired for in production (the other three —
white, periodic, albedo — are realizable but require sweep-side
plumbing tracked in separate issues). Adding a new kind is one
dict-entry edit.

Step 3 — method space construction
----------------------------------

The per-face
:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law`
calls share **one** unified
:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` for the whole
mesh, built once and stored on ``self._trace``. The factory is the
geometry-blind :meth:`AngularTraceSpace.from_quadrature_and_layout
<orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.from_quadrature_and_layout>`
— it takes the angular quadrature and the mesh's
:attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout` (a
:class:`~orpheus.numerics.face_layout.FaceLayout`, the single source of
truth for which faces exist and how they pack into one flat buffer), and
nothing else:

.. code-block:: python

   from orpheus.numerics.spaces.angular_trace_space import AngularTraceSpace

   self._trace = AngularTraceSpace.from_quadrature_and_layout(
       self.quad, self.boundary_face_layout,
   )

The trace stores the **signed projection** :math:`\Omega \cdot \hat
n_f` once per face as a ``(n_faces, N)`` float array — *not* two
direction-specific boolean masks. Inflow and outflow are
**selectors** over this one table, derived on demand by the sign of
:math:`\Omega \cdot \hat n_f` (the one-space-two-selectors design of
Issues #205 / #201; see :ref:`bc-trace-structure`). For the ``xmin``
face (``axis=0``, outward normal :math:`-\hat x`, so
:math:`\Omega \cdot \hat n = -\mu_x`), the inflow predicate
:math:`\Omega \cdot \hat n < -\epsilon` becomes :math:`-\mu_x[n] <
-\epsilon`, i.e. :math:`\mu_x[n] > \epsilon` — the rightward-pointing
ordinates are inflow at the left boundary, as expected.

The :meth:`SNMethodSpace.for_face` factory takes the precomputed trace
through a **single** ``trace=`` argument and extracts **both** per-face
half-trace index sets for the requested face:

.. code-block:: python

   from orpheus.sn.mesh.method_space import SNMethodSpace

   method_space = SNMethodSpace.for_face(
       mesh=self.mesh,           # optional metadata; None at d≥3
       quadrature=self.quad,
       face="xmin",
       trace=self._trace,
   )
   # for_face derives BOTH half-traces from the trace for this one face:
   #   inflow_indices  = trace.inflow_indices_for_face("xmin")   # Γ₋, codomain
   #   outflow_indices = trace.outflow_indices_for_face("xmin")  # Γ₊, domain
   # i.e. the 1-D int arrays [n for n in range(N) if omega_dot_n[n] < -eps]
   # and [... > +eps].  Neither is the complement of the other: the
   # tangential band |omega_dot_n| <= eps belongs to neither.

There is **one** trace object and **one** ``trace=`` parameter, not an
``inflow_trace`` / ``outflow_trace`` pair. The directional split lives
in the two selector methods
(:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.inflow_indices_for_face`
/
:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_indices_for_face`),
not in the space's identity. Why one space? Because whether an ordinate
is incoming or outgoing at a face is a *predicate* evaluated against the
same boundary data, not a property of two distinct domains — folding the
two former spaces into one removes a class of bugs where the inflow and
outflow descriptions of the *same* boundary could drift out of sync, and
gives the Wave-O adjoint work (#208) a single
:math:`|\Omega\cdot\hat n|`-weighted boundary inner product to install
(see :ref:`bc-trace-structure`). The ``mesh=`` argument is optional
metadata: nothing in the realizer chain reads it (the inflow indices
come from the trace), so an axis-native ``SNMesh`` with no legacy mesh
adapter passes ``None`` (C5.3, #225).

.. note:: **Historical — the pre-Issue-#188 split trace.** Before #188
   and the #205 / #201 unification, this step built a per-face
   ``InflowTraceSpace`` / ``OutflowTraceSpace`` *pair* via
   ``InflowTraceSpace.from_mesh_and_quadrature(mesh, quad,
   faces=("left", "right"))``, and :meth:`SNMethodSpace.for_face` took
   *two* arguments, ``inflow_trace=`` and ``outflow_trace=``. That
   machinery is retired: there is now one geometry-blind
   :class:`AngularTraceSpace <orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace>`
   with two directional selectors. This note records the older API only
   so that references to ``_inflow_trace`` / ``_outflow_trace`` in
   pre-#188 commit history are legible; it is **not** the current code.

Step 4 — realization
--------------------

The :class:`SNBoundaryRealizer` is now invoked — directly, by the
method-mesh that owns it (``SNMesh.realize_boundary_law``, the SN
arm of the :class:`~orpheus.transport.method.TransportMethod` hook;
#290 P7b removed the registry lookup that used to sit here).
Instantiation is stateless:

.. code-block:: python

   from orpheus.sn.boundary.realizer import SNBoundaryRealizer

   realized = SNBoundaryRealizer().realize(law, method_space)

The realizer's vacuum branch fires — after the ERR-041 guard
cross-checks the claimed ``inflow_indices`` against the signed
projection the face NAME alone implies:

.. code-block:: python

   # Inside SNBoundaryRealizer.realize, VacuumInflow arm (B3.2):
   gamma_out = _outflow_restriction(method_space, "vacuum")   # γ₊
   return stamp_boundary_role(
       ZeroOperator(
           codomain_zero=_zero_rows(method_space.inflow_indices.size),
           transpose_zero=_zero_rows(gamma_out.n_restricted),
       )
   )

The returned ``realized`` is the **zero map**
:math:`\Gamma_+ \to \Gamma_-`: a
:class:`~orpheus.numerics.operator.ZeroOperator` carrying **both**
space hooks, so the forward emits the zero of :math:`\Gamma_-` and the
transpose the zero of :math:`\Gamma_+`. It reports
``is_adjointable = True`` and ``is_invertible = False``. Vacuum's whole
content is :math:`R = 0`; with the domain narrowed there is nothing
else to represent.

.. note::

   **What this replaced, and why the replacement is not a
   simplification.** Pre-B3.2 this arm returned
   ``IncomingOrdinateMaskTensor(inflow_indices=…, n_ordinates=quad.N)
   & IdentityOperator()`` — a **full-face** operator that zeroed the
   inflow rows and *preserved* the outflow rows, which the consumer
   then discarded with a slice-write. Two campaign phases had
   documented that survival as having "no consumer today". The
   narrowing does not answer that question, it **removes** it: those
   rows are no longer in the operator's domain.

   The two space hooks are load-bearing, not ceremony. A
   :class:`~orpheus.numerics.operator.ZeroOperator` with no hooks
   returns ``0.0 * x`` — an *endomorphic echo* of its input's shape.
   That would be right here only by accident, because
   :math:`|\Gamma_+| = |\Gamma_-|` on every reachable fixture; the
   hooks make the map between two genuinely different spaces
   structural rather than lucky.

.. _bc-step5-pair-with-law:

Step 5 — pair the realized operator back with its law
-----------------------------------------------------

Every ``SNMesh.bc[<face>]`` entry carries a uniform 1-arg
``apply(psi)`` contract (Wave 9 migrated 13 production sites from
2-arg to 1-arg; C4 / #220 re-keyed the per-attribute ``bc_<face>``
surface to the face-name-keyed :attr:`~SNMesh.bc` dict — see
:ref:`bc-face-name-carve`). Post Issue #186 / C-B3.4 the
:class:`~orpheus.geometry.boundary._bound_compat._BoundBoundaryOperator`
shim is a **strict 1-arg passthrough**; campaign phase **B2.0** made
what it passes through *both* faces of the realization:

* the **law** the operator was realized from. The three-layer
  architecture (:ref:`bc-overview-three-layers`) is descriptor →
  realizer → operator, and until B2.0 this step *dropped the
  descriptor*: the shim kept a copy of ``law.key`` — a **string** — and
  nothing else. So ``sn_mesh.bc[face]`` could answer *"what were you
  declared as?"* but not *"what does your law DO?"*, and the five
  production sites needing the latter — ``sweep_schedule``'s reflective
  set, the two ruled-corner gates on the radial-characteristic boundary,
  ``solver``'s leakage face list, and the DSA low-order admission check
  — had no choice but to re-derive it by comparing that string against
  literals. Handing ``law`` through makes the structural questions
  answerable at the object, which is what phase **B2.2** then did:
  ``law_permutes_ordinates(bc[face].law)`` — spelled
  ``bc[face].law.geometry_map.permutes_ordinates`` inline at each of the
  four sites until **B3.4b** collapsed them into one function that asks
  both factor tiers (a polished wall's specular return lives in
  :math:`R`, so the geometry-tier read gave two identical operators
  different answers) —
  ``bc[face].law.response_kernel.is_zero``, and — collapsing diffusion's
  five-arm ``isinstance`` ladder —
  ``bc[face].law.response_kernel.amplitude`` (named ``.scalar`` until
  B3.0 minted the kernel tier: with a rank-one
  :class:`~orpheus.geometry.boundary.LambertianReemission` in play
  "scalar" is actively wrong, while "amplitude" is true of both
  realizations and is exactly the dimension-reduced view the diffusion
  arm reads). The tag frozensets those
  sites keyed on (``_RULED_CORNER_KINDS``, ``_SUPPORTED_BC``, both
  ``{"vacuum", "reflective"}``) are retired;
* a ``kind`` string tag, now a **read-through** of the law's registry
  key rather than a stored copy — load-bearing for the
  ``sn_mesh.bc["xmin"] == "vacuum"`` string-equality surface that
  several SN tests rely on, until phase B2.2 retires it;
* :attr:`~orpheus.numerics.operator.LinearOperator.is_invertible` /
  :attr:`~orpheus.numerics.operator.LinearOperator.is_adjointable`
  delegation, and the ``domain`` / ``codomain`` function-space tags,
  forwarded to the wrapped inner operator so consumers composing the
  shim with other Wave-0 primitives inherit the right surface.

The shim's ``apply`` / ``apply_transpose`` signatures are strict
1-arg ``(self, psi)`` — extra positional or keyword arguments
raise :class:`TypeError`. The pre-Issue-#186 affordance that
swallowed ``*_extra, **_kw`` was the last remnant of the 2-arg
legacy era; it was dropped in C-B3.4 alongside the descriptor
cleanup because every production and test call site is now
strict 1-arg.

.. code-block:: python

   from orpheus.geometry.boundary._bound_compat import _BoundBoundaryOperator

   return _BoundBoundaryOperator(realized, law)

``law`` is **required**: a realized boundary law that cannot say which
law it realizes is precisely the state phase B2 exists to delete, so it
is not constructible.

.. warning::

   ``kind`` reads ``type(law).key``, deliberately **not** ``law.kind``.
   The two agree for six of the seven laws and diverge for exactly one:
   a partially-reflecting
   :class:`~orpheus.geometry.boundary.reflective.ReflectiveBoundary`
   reports ``kind == "partial"`` — mirroring the ``BC("partial",
   albedo=…)`` *declaration* vocabulary that ``BC.to_alpha`` accepts —
   while its ``key`` stays ``"reflective"`` for every albedo. The key
   is what the pre-B2.0 shim stored, so it is the behaviour-preserving
   choice; sourcing the more obvious ``law.kind`` here would silently
   drop partially-reflecting faces out of
   ``sweep_schedule.reflective_faces``' ``== "reflective"`` set. That
   is a semantic change wearing a refactor's clothes, and
   ``tests/geometry/test_bound_compat.py`` reddens on it.

The shim is **internal** to the package (not in :attr:`__all__`)
— a test pins its private status.

**Historical note (pre Issue #176 / #186).** The Wave-8/9
implementation carried an optional ``quadrature=`` kwarg that,
when non-``None``, bound an
``AngularQuadrature`` and forwarded
``inner.apply(psi, bound_quad)`` to a legacy 2-arg
:class:`BoundaryTraceLaw` body. That dual-mode existed ONLY
because the trace factory (then named ``InflowTraceSpace.from_mesh_and_quadrature``,
since C5.3 the geometry-blind
:meth:`AngularTraceSpace.from_quadrature_and_layout
<orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.from_quadrature_and_layout>`)
raised :class:`NotImplementedError` for curvilinear ``Mesh1D``, which
forced the per-face resolution (then ``SNMesh._resolve_one``, since
#290 P7b ``SNMesh.realize_boundary_law``) to bypass the realizer for
spherical / cylindrical meshes. Issue #188 lifted that
deferral; Issue #176 (C176.1) dropped the bound-quadrature mode
here because no production-issued shim carried
``_quadrature is not None`` after C188.3. Issue #186 (C-B3.4)
then dropped the residual ``*_extra, **_kw`` argument-swallow
because, with concrete-BC ``apply`` methods retired and all
production / test sites strict 1-arg, the defensive net was
dead code.

Step 6 — consumption by the sweep
---------------------------------

The resolved operator keeps the uniform 1-arg
``apply(psi)`` interface, but **what it consumes changed twice**, and
both changes matter to anyone reading old call sites:

.. code-block:: python

   # Wave-8 era — the sweep called it, on the WHOLE face slot:
   psi_in = sn_mesh.bc["xmin"].apply(psi_out_full)      # (N, ng)

   # Today — the sweep is BARE. The sole consumer is B's per-face
   # composition, and the law's domain is Γ₊:
   gamma_out = trace.outflow_restriction("xmin")        # γ₊
   gamma_in  = trace.inflow_restriction("xmin")         # γ₋
   image = sn_mesh.bc["xmin"].apply(gamma_out.apply(face_in))   # (|Γ₊|,…) → (|Γ₋|,…)
   out_boundary.face_view("xmin")[...] = gamma_in.apply_transpose(image)

Wave O step O.4a.2 / O.4b removed ``bc.apply`` from the sweep entirely
for every geometry — the reflective coupling is delivered by the
sibling :math:`-B` (:ref:`bc-extraction`) — and campaign phase B3.2
narrowed the law's domain to :math:`\Gamma_+`
(:ref:`bc-domain-narrowing`). The public ``sn_mesh.bc[face].apply``
surface survives, so a caller can still reach a realized law directly;
it must now hand it a :math:`\Gamma_+`-shaped argument.

.. warning::

   **Being narrowed and validating one's own domain are separate
   properties.** Measured, and still true: fed a full-face input, both
   vacuum's :class:`~orpheus.numerics.operator.ZeroOperator` and
   reflective's :class:`~orpheus.numerics.operator.TensorProductOperator`
   return :math:`|\Gamma_-|` rows of **wrong values with no raise** —
   they are correctly typed :math:`\Gamma_+ \to \Gamma_-` and still
   silent about a wrong-shaped argument. The construction guard lives
   on :class:`~orpheus.numerics.operator.TraceRestrictionOperator` and
   does not travel to the operator the realizer *emits*. This is
   unreachable through
   :meth:`_reflect_trace <orpheus.sn.operators.boundary.SNBoundaryOperator>`
   — which always feeds a guarded ``γ₊.apply(...)`` — but **reachable
   through** ``sn_mesh.bc[face].apply``.

   The B3.4a arms split on exactly this axis, which is why the two
   properties must be named separately. White **refuses** the
   full-face input (:meth:`PartialCurrentOperator.apply
   <orpheus.sn.boundary.angular.PartialCurrentOperator.apply>`, the
   chain's first link, checks
   ``psi.shape[0]`` against :math:`|\Gamma_+|`) — the strictly stronger
   answer. Prescribed inflow does **not**: it ignores its input
   entirely, so it has nothing to validate against, and merely returns
   :math:`|\Gamma_-|` rows. Both are narrowed; only one validates.
   Never read "narrowed" as "guarded", and never credit the weaker
   property as the stronger one.


.. _bc-universal-invariants:

Universal ``assert_*`` invariants
=================================

The :class:`~orpheus.geometry.boundary.BoundaryTraceLaw` ABC declares
**five universal** assertion methods plus **three specific**
assertions on the BCs that need them. Together they form the
structural verification surface that the
:mod:`tests.geometry.test_bc_universal_invariants` suite exercises.

.. warning::

   **The declared surface is wider than the implemented one, and the
   tables below now say so per row.** Measured on the live tree:
   four of the five universal methods have an **empty** base body,
   and two of those four (``assert_inflow_outflow_classification``,
   ``assert_outgoing_leakage_unconstrained``) are overridden by
   **nobody** — they are permanent no-ops that assert nothing about
   any law. Four of the seven concrete laws
   (:class:`~orpheus.geometry.boundary.VacuumInflow`,
   :class:`~orpheus.geometry.boundary.PeriodicBoundary`,
   :class:`~orpheus.geometry.boundary.PrescribedInflow`,
   :class:`~orpheus.geometry.boundary.ZeroFluxBoundary`) override no
   universal invariant at all. Read the "What it asserts" column as a
   statement of *intent* wherever it says "Default: no-op"; only
   ``assert_source_is_placeable`` fires for every law.

   A second gap, orthogonal to the empty bodies: the aggregate
   :meth:`~orpheus.geometry.boundary.BoundaryTraceLaw.assert_realizable`
   that fires these at realize time has exactly **one** production
   caller, the SN realizer. The diffusion realizer never calls it, so
   the claim that "every law arrives at its primitive construction
   already certified" holds on the SN arm only.

The split between "universal" and "specific" follows the
Grand Report v3 §16A.12 + §27.6 catalog. Universal invariants are
properties the affine law :eq:`affine-bc-form` should satisfy for
**any** physically meaningful BC; specific invariants pin properties
that only a subset of laws claim (e.g. involution is meaningful for
reflective but not for white).

.. list-table:: Universal invariants
   :header-rows: 1
   :widths: 30 25 45

   * - Method
     - Pinned error
     - What it asserts, and what is actually implemented
   * - ``assert_inflow_outflow_classification``
     - :class:`~orpheus.geometry.boundary.IncomingOutgoingTraceClassificationError`
       (ERR-040) — declared, but **never raised** anywhere in
       ``orpheus/``
     - *Intended*: every ordinate at the face is either inflow or
       outflow (no tangential ordinates allowed by the law's
       contract). *Implemented*: **nothing** — empty base body,
       overridden by no law. A permanent no-op.
   * - ``assert_outgoing_leakage_unconstrained``
     - n/a (architectural contract)
     - *Intended*: the outgoing trace flux is not constrained by the
       BC. *Implemented*: **nothing** — empty base body, overridden
       by no law. A future Dirichlet-outflow / prescribed cell-edge
       interface law would be the first to give it content.
   * - ``assert_geometry_map_measure_preserving``
     - :class:`~orpheus.geometry.boundary.BoundaryGeometryMapNotMeasurePreservingError`
       (ERR-042)
     - The geometric map :math:`G` preserves the angular measure
       :math:`w(\Omega)\,|\Omega \cdot \hat n|`. Empty base body;
       **overridden by** :class:`ReflectiveBoundary` **only**, which
       compares :math:`m_{\pi(n)}` against :math:`m_n` **directly**.
       It does *not* delegate to the involution check: an involutive
       table that pairs ordinates from different weight classes
       passes involution while breaking the measure, and that is
       precisely the hole the pre-#52 delegation left open. The
       partition, the involution and the measure are INDEPENDENT
       invariants.
   * - ``assert_response_positive_if_declared``
     - :class:`~orpheus.geometry.boundary.BoundaryResponseNotPositiveError`
       (ERR-043)
     - If a response kernel is declared, it produces non-negative
       output on the inflow trace. Empty base body; **overridden by**
       :class:`WhiteBoundary` and :class:`AlbedoBoundary`.
   * - ``assert_source_is_placeable``
     - :class:`~orpheus.geometry.boundary.BoundarySourceNotOnIncomingTraceError`
       (ERR-047)
     - A law carrying a source has a NAMEABLE :math:`\Gamma_-` to deliver
       into. **The only universal with a real base body** — it raises when a
       source-carrying law is realized without an inflow-index set. (P6
       re-pose: it asserted "the source :math:`q` is nonzero only on
       :math:`\Gamma_-`" and probed the source's values to decide; that claim
       is now structural, so there is nothing left to probe.) **No law overrides
       it**; every law (including
       :class:`~orpheus.geometry.boundary.PrescribedInflow`) is
       certified by that one body, which is why the
       :class:`NoSource` default passes trivially.

.. list-table:: Specific invariants
   :header-rows: 1
   :widths: 40 25 35

   * - Method
     - Pinned error
     - Where it's defined
   * - :meth:`ReflectiveBoundary.assert_is_involutive`
     - :class:`~orpheus.geometry.boundary.ReflectionNotInvolutiveError`
       (ERR-044)
     - :class:`ReflectiveBoundary` (the derived specular pairing must
       satisfy ``perm[perm] == arange``).
   * - :meth:`ReflectiveBoundary.assert_reflection_maps_inflow_to_outflow`
     - :class:`~orpheus.geometry.boundary.ReflectionDidNotMapInflowToOutflowError`
       (ERR-045)
     - :class:`ReflectiveBoundary` (every inflow ordinate maps to an
       outflow ordinate under the reflection).
   * - :meth:`WhiteBoundary.assert_submarkov` /
       :meth:`AlbedoBoundary.assert_submarkov`
     - :class:`~orpheus.geometry.boundary.SubmarkovViolationError`
       (ERR-046)
     - :class:`WhiteBoundary` + :class:`AlbedoBoundary` (the sub-Markov
       kernel constraint :math:`\int R\,\mathrm{d}y \leq 1` per row;
       albedo :math:`> 1` violates this physically).


.. _bc-named-error-catalog:

Named-error catalog (ERR-040..ERR-047)
======================================

Per Grand Report v3 §26A.4 and the `vv-principles` skill's
"Log every caught bug" directive, every typed error has a matching
``@pytest.mark.catches("ERR-NNN")`` decorator on the test that
proves it fires under the right fault-injection. The eight errors
shipped under :mod:`orpheus.geometry.boundary._errors` are:

.. list-table::
   :header-rows: 1
   :widths: 18 26 12 44

   * - Error class
     - Trigger condition
     - Mode
     - Mechanism
   * - :class:`~orpheus.geometry.boundary.IncomingOutgoingTraceClassificationError`
       (ERR-040)
     - Tangential ordinate at a face that requires strict partition.
     - #5 (index)
     - ``assert_inflow_outflow_classification`` finds
       ``|Ω · n| ≤ ε`` on a face where the law's contract
       forbids it.
   * - :class:`~orpheus.geometry.boundary.VacuumAppliedToOutgoingTraceError`
       (ERR-041)
     - Vacuum law applied to an outgoing trace.
     - #6 (convention)
     - Vacuum sets only :math:`\gamma_- \psi = 0`; applying it on
       :math:`\Gamma_+` is geometrically meaningless and typically
       indicates a wrong face annotation.
   * - :class:`~orpheus.geometry.boundary.BoundaryGeometryMapNotMeasurePreservingError`
       (ERR-042)
     - Geometric map :math:`G` does not preserve
       :math:`w(\Omega) |\Omega \cdot \hat n|`.
     - #5 + #6
     - Wrong specular pairing or inconsistent quadrature
       :math:`\mu_n` / weights.
   * - :class:`~orpheus.geometry.boundary.BoundaryResponseNotPositiveError`
       (ERR-043)
     - Response kernel produces negative output.
     - #1 (sign)
     - Sign-flipped kernel construction (e.g. accidental
       :math:`-\alpha`).
   * - :class:`~orpheus.geometry.boundary.ReflectionNotInvolutiveError`
       (ERR-044)
     - Reflection permutation is not its own inverse:
       :math:`\pi \circ \pi \neq \mathrm{id}`.
     - #5 (index)
     - Wrong reflection axis or a non-involutive derived pairing
       (:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`).
   * - :class:`~orpheus.geometry.boundary.ReflectionDidNotMapInflowToOutflowError`
       (ERR-045)
     - Reflection maps an inflow ordinate to itself.
     - #5 (index)
     - Non-axis-aligned reflection mislabeled as ``ReflectiveBoundary``
       (the right BC family would be a future ``SymmetryBoundary``).
   * - :class:`~orpheus.geometry.boundary.SubmarkovViolationError`
       (ERR-046)
     - Sub-Markov BC with :math:`\alpha > 1`.
     - #4 (factor)
     - Albedo / white kernel scalar exceeds 1.0 — physically this
       would imply a source on the boundary surface.
   * - :class:`~orpheus.geometry.boundary.BoundarySourceNotOnIncomingTraceError`
       (ERR-047)
     - Boundary source :math:`q` has nonzero outflow entries.
     - #6 (convention)
     - User-supplied :class:`InflowSourceSpec` has nonzero entries on
       :math:`\Gamma_+`; geometrically meaningless and indicates a
       wrong source-shape contract.

All eight extend :class:`ValueError` (via the
:class:`~orpheus.geometry.boundary.BoundaryError` base) so existing
``except ValueError`` consumers from the pre-refactor code keep
working. Every catch site can additionally pattern-match on the
typed subclass to recover the offending law name from
:attr:`BoundaryError.law`.


.. _bc-rank-n-algebra:

.. _bc-descriptor-tree-vs-operator-tree:

Descriptor-tree algebra for rank-N boundaries
=============================================

Rank-N (Marshak, partial-current) boundary conditions are
**not** a special class. They are expressed via a closed
**descriptor-tree algebra** over
``BoundaryTraceLaw | LawSum | LawScaled`` nodes — pure
declarative structure with **no** ``apply`` method on any node.
The :class:`~orpheus.geometry.boundary.BoundaryTraceLaw` algebra
dunders (``+``, ``-``, ``*``, ``/``, unary ``-``) return
:class:`~orpheus.geometry.boundary.LawSum` /
:class:`~orpheus.geometry.boundary.LawScaled` instances, never
operators. The :func:`~orpheus.geometry.boundary.realize_recursively`
type transformer (method-blind since #290 P7b — the leaf realizer is
an explicit argument) is the **sole** path from descriptor tree to
operator tree.

The §15.2 sum-of-tensor-products form

.. math::
   :label: bc-rank-n-tensor-decomposition

   B \;=\; \sum_{\alpha} c_{\alpha}\, G_{\alpha},
   \qquad c_{\alpha} \in \mathbb{R},
   \quad G_{\alpha} \in
   \{\text{permutation, average, mask, wrap, identity, source}\},

.. vv-status: bc-rank-n-tensor-decomposition documented

maps onto the LawXxx algebra as ``c1 * law_1 + c2 * law_2 + ...``,
where each ``c_i * law_i`` term is a :class:`LawScaled` node and
the sum is a :class:`LawSum` node.

The standard Marshak boundary (Bell & Glasstone 1970 §1.5) — a
mix of specular reflection (weight :math:`c_1`) and diffuse
white reflection (weight :math:`c_2`) — is:

.. code-block:: python

   from orpheus.geometry.boundary import (
       LawScaled, LawSum,
       ReflectiveBoundary, WhiteBoundary,
       realize_recursively,
   )
   from orpheus.sn.boundary.realizer import SNBoundaryRealizer
   from orpheus.sn.mesh.method_space import SNMethodSpace

   # Build the descriptor tree — no realization yet.
   spec = ReflectiveBoundary(axis="x", albedo=1.0)
   white = WhiteBoundary(axis="x", outward_sign=+1, albedo=1.0)
   marshak_law = 0.3 * spec + 0.7 * white
   # marshak_law is:
   #   LawSum(
   #       LawScaled(0.3, ReflectiveBoundary(axis="x", albedo=1.0)),
   #       LawScaled(0.7, WhiteBoundary(axis="x", outward_sign=+1,
   #                                    albedo=1.0)),
   #   )
   # NOT callable — no .apply method on LawSum or its children.
   assert not hasattr(marshak_law, "apply")

   # Realize the tree at one face. realize_recursively walks
   # LawSum / LawScaled / leaf-law nodes and emits the matching
   # Wave-0 operator-tree composers around realized 1-arg leaves.
   # The walker is method-blind — pass the method's own realizer.
   ms = SNMethodSpace.for_face(...)
   marshak_op = realize_recursively(marshak_law, ms, SNBoundaryRealizer())
   # marshak_op is (MEASURED — each leaf is the 2-factor tensor
   # product the realizer emits, not the bare angular primitive;
   # white's angular factor is the G6.3 two-link chain):
   #   OperatorSum(
   #       ScaledOperator(0.3, PermutationOperator(...) & IdentityOperator()),
   #       ScaledOperator(0.7, (IsotropicEmissionOperator(...)
   #                            @ PartialCurrentOperator(...))
   #                           & IdentityOperator()),
   #   )
   psi_in = marshak_op.apply(psi_out)   # 1-arg; psi_out is Γ₊-shaped

The output is a Wave-0
:class:`~orpheus.numerics.operator.OperatorSum` of
:class:`~orpheus.numerics.operator.ScaledOperator`-wrapped Wave-0
primitives, consumable by the SN sweep / Krylov path via the
uniform 1-arg :meth:`apply`. The descriptor-tree algebra and the
operator-tree algebra are **separate type families**: the
descriptor tree is built with ``LawXxx`` nodes that have **no**
``apply``; the operator tree is built with ``OperatorXxx`` nodes
that **do** have ``apply``. The two families never inter-compose —
mixing a :class:`LawNode` with an already-realized
:class:`~orpheus.numerics.operator.LinearOperator` via ``+`` is a
type error; the user MUST call :func:`realize_recursively` first.

Closed-algebra guarantees
-------------------------

* **Constant folding on scalars.**
  ``LawScaled(α, LawScaled(β, x))`` collapses to
  ``LawScaled(α * β, x)`` at construction time. The intermediate
  ``LawScaled`` nesting never appears at rest, which keeps the tree
  shallow under repeated scalar multiplication.
* **No associativity flattening on sums.** ``(a + b) + c`` is
  :class:`LawSum(LawSum(a, b), c)`, distinct from
  :class:`LawSum(a, LawSum(b, c))`. The walker treats both shapes
  identically — the realized output is the same Wave-0 operator
  algebra value up to floating-point non-associativity in the
  final sum.
* **Subtraction rewrites via :class:`LawScaled(-1, ...)`.** The
  unary ``-`` operator and the binary ``-`` operator both produce
  trees containing only :class:`LawSum` / :class:`LawScaled`
  nodes — there is no dedicated ``LawDifference`` type.
* **Division rewrites via :class:`LawScaled(1/α, ...)`.** Pure
  syntactic sugar for ``LawScaled(α, ...).__truediv__``.

The pre-refactor implementations
--------------------------------

Two prior approaches converged on the present descriptor-tree
design through empirical falsification:

**Wave 11 (~2026-03)** — ``MixedBoundaryOperator(components:
list[tuple[float, BoundaryOperator]])`` class whose
:meth:`apply` body looped over ``components`` and summed
``coeff * primitive.apply(psi, quad)``. The SN realizer
dispatched on it via an ``isinstance(law, MixedBoundaryOperator)``
branch that ran the same loop with
``coeff * realize(primitive, ms)`` summed via
:class:`OperatorSum`. Wave 11 deleted this code because the
delayed-realization-by-container pattern broke down once vacuum
needed per-face inflow indices that the bare-law container had
no access to.

**β1 interim landing (Issue #186 / B3, ~2026-04)** — every
:class:`BoundaryTraceLaw` inherited the Wave-0 operator-algebra
dunders from :class:`~orpheus.numerics.operator.LinearOperator`,
so writing ``0.3 * spec + 0.7 * white`` directly produced an
:class:`OperatorSum` of :class:`ScaledOperator`-wrapped raw
:class:`BoundaryTraceLaw` leaves (NOT realized). The
:func:`realize_recursively` walker then traversed the Wave-0
composer tree, realized each leaf, and emitted a parallel
operator tree. This achieved the right algebraic shape but
**violated the type system**: the resulting tree was an
:class:`OperatorSum` instance (a :class:`LinearOperator`!) whose
:meth:`apply` could not actually be called before realization —
calling it raised :class:`BoundaryError` at apply-time because the
leaves were laws, not operators. The convention "you must realize
this OperatorSum before calling apply" was a runtime contract that
the type system did nothing to enforce. β1 retained the
ergonomic of "the same ``+`` operator before and after
realization" but at the cost of conflating two type families.

**β2 (this scope, Issue #186 / B3 + β2)** — separates the two
type families explicitly: :class:`LawSum` / :class:`LawScaled` for
the descriptor tree (no :meth:`apply`); :class:`OperatorSum` /
:class:`ScaledOperator` for the operator tree (with
:meth:`apply`). The static type system enforces "you cannot call
this until it's realized" — :class:`LawSum` has no :meth:`apply`
method on the class, so the linter flags ``tree.apply(...)``
without running the program. The ergonomic of "the same ``+``"
survives because both the law-tree and the operator-tree use the
same Python ``+`` syntax; the runtime dispatch on type tells the
reader (and the type checker) which algebra is in effect.

The reference-image harness verifies that the Marshak case
``0.3 * spec + 0.7 * white`` survived the β2 transition: the realized
``OperatorSum(ScaledOperator, ScaledOperator)`` reduction tree
reproduces the derived convex combination
:math:`0.3\,R_{\text{spec}} + 0.7\,R_{\text{diff}}` inside the
reduction-order bound, because the operator-tree shape after
:func:`realize_recursively` is algebraically identical. (Until the
2026-08-01 re-anchoring the claim was the narrower one that the
realized output matched the β1-era *recorded* output; see
:ref:`bc-snapshot-reanchoring` for why the reference moved.)


.. _bc-realize-recursively:

The ``realize_recursively`` walker — a descriptor → operator type transformer
=============================================================================

:func:`~orpheus.geometry.boundary.realize_recursively` is the
**type transformer** from the descriptor-tree algebra
(``BoundaryTraceLaw | LawSum | LawScaled``) to the operator-tree
algebra (``LinearOperator`` with
:class:`~orpheus.numerics.operator.OperatorSum` /
:class:`~orpheus.numerics.operator.ScaledOperator` composers
around realized 1-arg leaves). Calling it is the **only** path
from a non-callable descriptor to a callable operator.

The dispatch is exhaustive on the descriptor-tree node types:

.. code-block:: python

   def realize_recursively(
       node: BoundaryTraceLaw | LawSum | LawScaled,
       method_space: MethodSpaceT,
       realizer: BoundaryRealizer[MethodSpaceT],
   ) -> LinearOperator:
       if isinstance(node, BoundaryTraceLaw):
           # Leaf: dispatch through the CALLER's realizer — the walker
           # is method-blind (#290 P7b; it names no method's realizer).
           return realizer.realize(node, method_space)
       if isinstance(node, LawScaled):
           # Scalar-times-law: wrap the realized inner in ScaledOperator.
           inner_op = realize_recursively(node.inner, method_space, realizer)
           return ScaledOperator(node.scalar, inner_op)
       if isinstance(node, LawSum):
           # Sum: realize each side, wrap in OperatorSum.
           a_op = realize_recursively(node.a, method_space, realizer)
           b_op = realize_recursively(node.b, method_space, realizer)
           return OperatorSum(a_op, b_op)
       raise TypeError(
           f"realize_recursively expected BoundaryTraceLaw | LawSum | "
           f"LawScaled, got {type(node).__name__}."
       )

Usage on the descriptor tree:

.. code-block:: python

   from orpheus.geometry.boundary import (
       ReflectiveBoundary, WhiteBoundary, realize_recursively,
   )
   from orpheus.sn.boundary.realizer import SNBoundaryRealizer

   # Build the descriptor tree.
   law = (
       0.3 * ReflectiveBoundary(axis="x")
       + 0.7 * WhiteBoundary(axis="x", outward_sign=+1)
   )
   # law is LawSum(LawScaled(0.3, ...), LawScaled(0.7, ...)).

   # Realize once, at face resolution time — the walker is
   # method-blind, so the method's realizer is passed explicitly.
   realized = realize_recursively(law, method_space, SNBoundaryRealizer())
   # realized is (MEASURED — each leaf is the 2-factor tensor product
   # the realizer emits, not the bare angular primitive; white's
   # angular factor is the G6.3 two-link chain):
   #   OperatorSum(
   #       ScaledOperator(0.3, PermutationOperator(...) & IdentityOperator()),
   #       ScaledOperator(0.7, (IsotropicEmissionOperator(...)
   #                            @ PartialCurrentOperator(...))
   #                           & IdentityOperator()),
   #   )
   psi_in = realized.apply(psi_out)   # 1-arg; psi_out is Γ₊-shaped

Type-system contract
--------------------

The walker's input is intentionally narrow:

* :class:`BoundaryTraceLaw` instances and the two descriptor-tree
  composer dataclasses (:class:`LawSum`, :class:`LawScaled`) are
  the only valid node shapes.
* Wave-0 operator-tree composers
  (:class:`~orpheus.numerics.operator.OperatorProduct`,
  :class:`~orpheus.numerics.operator.TensorProductOperator`,
  :class:`~orpheus.numerics.operator.SumOfTensorProductsOperator`,
  :class:`~orpheus.numerics.operator.OperatorSum`,
  :class:`~orpheus.numerics.operator.ScaledOperator`) are **not**
  recognized — they belong to the operator tree, not the
  descriptor tree, so they should never appear in the realizer's
  input.
* Unknown nodes raise :class:`TypeError` (not
  :class:`BoundaryError`) with the offending type name in the
  message, because this is a **typing** failure (caller passed
  the wrong kind of object), not a BC-domain failure.

The β1 → β2 transition (see :ref:`bc-rank-n-algebra`) sharpened
the walker's type signature from "any Wave-0 composer tree with
:class:`BoundaryTraceLaw` leaves" to "any descriptor-tree node".
The dispatch table shrank from five Wave-0 composers + leaf to
three descriptor types + leaf (counting the leaf as the same
:class:`BoundaryTraceLaw` branch). The eliminated branches
(:class:`OperatorProduct`, :class:`TensorProductOperator`,
:class:`SumOfTensorProductsOperator`) handle operator composition
patterns that have no descriptor-tree analog — they were dead
dispatch paths once :class:`LawSum` / :class:`LawScaled` replaced
the in-tree Wave-0 algebra. Removing them clarified the walker's
role: it is **exactly** the type transformer between the two
algebras, nothing more.

Placement — the deferral that fired at the second method
--------------------------------------------------------

The walker lives in :mod:`orpheus.geometry.boundary` (the
``_realizer`` module — the realization seam, next to the
:class:`~orpheus.geometry.boundary.BoundaryRealizer` Protocol it
dispatches through), and it is **method-blind**: the leaf realizer
is a REQUIRED argument, and ``method_space`` is threaded verbatim to
that realizer without inspection. It is *not* on the production
single-BC path — production realizes one BC directly (each
method-mesh's ``realize_boundary_law`` arm →
``<Method>BoundaryRealizer().realize``). The walker is the **rank-N
composition entry point**: the only thing that realizes a
*descriptor tree* (the Marshak ``0.3 * Reflective + 0.7 * White``
partial-current BC of :eq:`affine-bc-form`) rather than a single
leaf law. Production does not yet wire a rank-N BC, so the walker
runs only from the rank-N tests.

**How it got here is a worked example of the
defer-until-two-instances rule.** Until #290 the walker lived in
:mod:`orpheus.sn.boundary.realizer`, honestly SN-specific: it
threaded an :class:`~orpheus.sn.mesh.method_space.SNMethodSpace` and
hardcoded ``SNBoundaryRealizer`` at the leaf. (Before that it sat in
a separate ``boundary_realize`` module — the near-twin filename next
to ``boundary_realizer`` was a standing legibility hazard; merging
the two retired it.) The cross-method generalization was explicitly
**deferred until the second functional realizer ships**, with the
recorded insight that the deferral trigger was not local to boundary
realization: the same event — a second transport method arriving —
would also mint the ``TransportMethod`` Protocol flagged in
:mod:`orpheus.transport.mesh.material_mesh`, because the
**boundary-realizer seam** and the **homogenization method-layer**
were two independent witnesses to one missing type, to be typed
*together* at method #2 rather than as string-keyed half-steps.

That trigger fired when diffusion adopted the architecture (#290
P3), and the landing (#290 P7b) matched the prophecy on every point
but one: the deferral-era sketch had the walker "resolving its leaf
realizer through ``BoundaryRealizerRegistry``" — the actual carve
**dissolved the registry instead**. With a real second method in
hand, the consumption pattern was visible: production holds a
method-mesh and therefore holds its realizer; nobody resolves
realizers by name. The
:class:`~orpheus.transport.method.TransportMethod` Protocol landed
on the **method-mesh layer** (``SNMesh`` / ``DiffusionMesh`` — the
user ruling: the method-mesh IS the method's behavior carrier, not a
stateless singleton), the twin per-mesh ``_resolve_bcs`` loops
collapsed into the ONE shared
:func:`~orpheus.transport.method.resolve_boundary_conditions` body,
and the walker moved here with an explicit-realizer signature. Only
the **leaf** dispatch was ever method-bound; the **composer**
dispatch (:class:`LawSum` / :class:`LawScaled` → operator composers)
survived the move byte-identically, exactly as recorded.


.. _bc-vacuum-semantic-correction:

The vacuum semantic correction (§16A.5) — and its dissolution at B3.2
=====================================================================

.. important::

   **Status (campaign phase B3.2, 2026-07-31).** This whole section is
   now **history**. Vacuum realizes to the **zero map**
   :math:`\Gamma_+ \to \Gamma_-`
   (:ref:`bc-domain-narrowing`); the inflow-only mask, the outflow rows
   it preserved, and the question of what preserving them was *for* are
   all off the vacuum path. The section is kept because the reasoning
   below is the reasoning a future session would otherwise re-derive —
   and because the way it was resolved (by narrowing the domain rather
   than by answering the question) is the more general lesson.

The most subtle design decision of the trace-law refactor concerned
vacuum. Pre-Wave-7 the legacy ``VacuumBoundaryOperator.apply`` body
was:

.. code-block:: python

   def apply(self, psi_out: np.ndarray, quadrature) -> np.ndarray:
       return np.zeros_like(psi_out)

This returns **all zeros**, including the outflow ordinates that
the BC has no physical interpretation for (vacuum says nothing
about :math:`\gamma_+ \psi`; it only sets :math:`\gamma_- \psi = 0`).

The post-Wave-8 SN realizer's vacuum branch therefore returned an
``IncomingOrdinateMaskTensor`` — a
sparse mask that zeroed **only the inflow ordinates** and preserved the
outflow trace. That was read as the §16A.10 *trace-correct*
representation: a projector onto the outflow ordinate subspace, which
looked like the right algebraic object for the affine law
:eq:`affine-bc-form` to read.

Why it seemed to matter — and why each reason dissolved
--------------------------------------------------------

.. note::

   **Retraction (2026-07-31, campaign phase B3.2).** The three
   consequences below were the published justification for preserving
   the outflow rows. **All three are now moot, and the first was
   measurably false.** They are preserved verbatim in substance,
   past-tensed, with each disposition stated — deleting them would
   destroy the record of *why* a wrong contract looked right for two
   campaign phases.

Three downstream consequences were argued to make the inflow-only mask
the right contract:

1. **Sensitivity adjoints.** *The argument was:* a future adjoint
   sensitivity path needs the outflow trace preserved to compute the
   response of an outgoing-current functional to the inflow BC; the
   zeros-all body would silently lose the gradient at the boundary.

   **Disposition: the consumer was measured not to exist**, and B3.2
   removed the preservation entirely. The rows are outside the
   operator's domain now, so there is nothing to preserve. This was the
   load-bearing justification and it was a *promise about a future
   consumer* — precisely the "declared capability, no consumer" pattern
   the boundary review catalogued five times in this subsystem. The
   adjoint work that did land (#276, :ref:`g-adjoint`) reaches the
   boundary through :math:`B^{\mathsf T}` and the metric-correct
   ``B.H``, neither of which needs a vacuum law to echo its input.
2. **Compositional clarity.** *The argument was:* the realized vacuum
   operator is self-adjoint and idempotent — a projector — whereas the
   zeros-all operator is the ``ZeroOperator`` projector, the wrong type
   tag for "inflow-mask only".

   **Disposition: inverted.** With the domain narrowed to
   :math:`\Gamma_+`, the vacuum law is not an endomorphism at all, so
   "idempotent" is not even a well-typed thing to ask of it — the
   composite :math:`\text{law} \circ \text{law}` does not exist.
   :class:`~orpheus.numerics.operator.ZeroOperator` is now exactly the
   right type tag, carrying two space hooks rather than an echo.
3. **Algebraic uniformity.** *The argument was:* every other rank-1 law
   acts on the inflow ordinates only and leaves the outflow rows
   untouched, so the legacy vacuum was the asymmetric special case.

   **Disposition: the uniformity is real and B3.2 made it structural
   rather than conventional.** Every law now *maps*
   :math:`\Gamma_+ \to \Gamma_-`; "leaves the outflow rows untouched"
   is no longer a behaviour a law could get wrong, because those rows
   are not in its codomain. The observation was right; the mechanism
   was one layer too shallow.

The algebra: where the two legacy semantics agreed and where they diverged
--------------------------------------------------------------------------

Decompose the angular ordinate set at a given face :math:`f` as

.. math::
   :label: ordinate-partition-inflow-outflow

   \{1, \ldots, N\} \;=\; I_f \,\sqcup\, O_f \,\sqcup\, T_f,

with :math:`I_f = \{n : \Omega_n \cdot \hat n_f < -\epsilon\}` the
inflow set, :math:`O_f = \{n : \Omega_n \cdot \hat n_f > +\epsilon\}`
the outflow set, and :math:`T_f` the (measure-zero in the continuum,
:math:`\epsilon`-band in the discretisation) tangential set. For
:math:`\psi_{\text{out}} \in \mathbb{R}^N` representing the trace
ordinate values at face :math:`f`, the two vacuum representations
are:

.. math::
   :label: vacuum-legacy-vs-trace-correct

   \mathrm{zeros\_all}(\psi_{\text{out}})[n] &= 0
       \qquad \forall\, n \in \{1, \ldots, N\}, \\[2pt]
   \mathrm{inflow\_mask}(\psi_{\text{out}})[n] &=
       \begin{cases}
           0 & n \in I_f, \\[2pt]
           \psi_{\text{out}}[n] & n \in O_f \cup T_f.
       \end{cases}

.. (vv-status rationale) HISTORICAL structural/explanatory identity: contrasts
   the two pre-B3.2 vacuum spellings (zeros-all vs the inflow-only mask) —
   they agree on the inflow set and diverge on the outflow set. NEITHER is the
   live realization: since campaign phase B3.2 vacuum realizes to the zero map
   Γ₊ → Γ₋, and the live gate is the re-posed vacuum snapshot case in
   ``tests/geometry/test_bc_equivalence_snapshot.py`` (asserts Γ₋ shape + all
   zero). Kept as the record of a contract that looked right for two campaign
   phases. An explanatory comparison, never a solver claim.
.. vv-status: vacuum-legacy-vs-trace-correct documented

The two functions agree **on** :math:`I_f` (both give 0) and
**diverge** on :math:`O_f` (legacy gives 0, trace-correct gives
:math:`\psi_{\text{out}}[n]`). They diverge on :math:`T_f` too.

.. warning::

   **The published mitigation for** :math:`T_f` **was an
   initialisation claim, not a structural guarantee — and the two
   spellings of "outflow" genuinely differ.** This page previously
   argued that ORPHEUS's quadrature adapters carry every tangential
   ordinate at :math:`\mu = 0`, so :math:`\psi_{\text{out}}[n] = 0` on
   :math:`T_f` "for a properly-initialised flux", making the divergence
   physically restricted to :math:`O_f`. Two corrections, both
   **measured** during the B3 crosswalk:

   1. :math:`\mu_{\rm axis} \approx 0` is what *defines* a tangential
      ordinate; it does not make :math:`\psi` vanish there. What is
      true is weaker and conditional — the boundary operators write
      **only** the inflow and outflow slots, so a zero-initialised
      trace carrier keeps its tangential rows at zero. That is a
      property of the *carrier's initialisation*, not of the algebra.
      (The tangential rows do carry zero *weight* in the
      :math:`|\Omega\cdot\hat n|\,w` trace metric, which is a separate
      and genuinely structural fact — see :ref:`g-adjoint`.)
   2. The mask preserved :math:`O_f \cup T_f` while
      :meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_indices_for_face`
      selects **strict** :math:`O_f`. So
      :math:`I - P_{\rm in} \neq P_{\rm out}`, and the mask was never a
      projector onto :math:`\Gamma_+` — it is
      :math:`I - \iota_-\gamma_-`, a different map whenever the
      quadrature carries tangential ordinates, which is **every**
      production quadrature except ``gauss_legendre(4)``. Measured on a
      cylinder under ``product(n_mu=2, n_phi=4)``: **4 of 8** ordinates
      at ``xmax`` are tangential.

   Both corrections are moot for vacuum since B3.2 — the map is the
   zero map on :math:`\Gamma_+` — but the second is *live discipline*
   everywhere else: **never spell "not inflow" as "outflow"**.

The §16A.5 production-relevant subset was **the inflow rows**.
Every SN sweep call site read :math:`\psi_{\text{in}}[n]` only for
:math:`n \in I_f` — outflow rows were never consumed downstream.
The Wave 8 close-out audited all 13 ``bc.apply(...)`` sites in
``orpheus.sn.loss_representation`` (the dissolved ``sweep.py``) and :mod:`orpheus.sn.operators.streaming`
(the file:line references below are frozen at that audit and do not
resolve against the live tree — Wave O later removed ``bc.apply`` from
the sweep entirely, see :ref:`bc-extraction`):

* ``sweep.py:334,351`` (1-D slab) read
  ``psi_face_left_in[n_half + n]`` for positive-μ ordinates
  only.
* ``sweep.py:508,654`` (spherical / cylindrical) gate the read
  by ``mu_n < 0`` / ``eta_n < 0`` (inflow at the outer face).
* ``sweep.py:843-854`` (2-D wavefront) reads
  ``full_face_x[oct_idx]`` only, where ``oct_idx ↔ inflow
  ordinates`` by construction.
* ``operator.py:230-256`` (2-D FD matvec) has explicit
  ``mu_x[n] > 1e-15`` / ``mu_y[n] < -1e-15`` gates per face.
* ``operator.py:530`` (spherical FD matvec) gates on
  ``quad.mu_x[n] < -1e-15``.

No call site read :math:`\psi_{\text{in}}[n]` for
:math:`n \in O_f`, so the two semantics produced **bit-identical
observable output** for every production consumer of that era.
That is what made the §16A.5 change a safe semantic upgrade — and,
read a second time, it is exactly the evidence that the preserved
outflow rows had **no consumer at all**. B3.2 acted on the second
reading.

Post Issue #188 + #176 (2026-05-11) the **realizer path is
uniform** across every supported mesh — 1-D Cartesian, 1-D
spherical, 1-D cylindrical, and 2-D Cartesian; that uniformity
survives B3.2 unchanged (every geometry's vacuum face realizes to the
same zero map, every geometry's reflective face to a permutation on
its own reduced axis). **Empirical confirmation at the time**:
spherical 26/26 + cylindrical 25/25 + MMS curvilinear 2/2 xfail
(pre-existing ERR-026) green on C188.3 — the curvilinear sweeps had
been consuming the zeros-all legacy body via the bound-quadrature shim
and moved onto the realizer output, bit-identical on inflow rows (the
only rows the sweep read), confirming the Wave 8 call-site audit
empirically.

The Wave 6 snapshot harness gates the vacuum case explicitly, and its
assertion was **re-posed, not weakened, at B3.2**: the
``vacuum_lebedev17`` case now cross-checks the live ``inflow_indices``
against the frozen snapshot's index set, feeds the realized law
``psi_out[outflow_indices]``, and asserts *both* that the emission has
:math:`\Gamma_-` shape and that it is identically zero — "the narrowed
vacuum law is the ZERO map :math:`\Gamma_+ \to \Gamma_-`". The old
inflow-rows-only comparison, which documented the intentional semantic
divergence on the outflow rows, has nothing left to compare.

"Option a" (Wave 7) — historical context, retired Issue #186
-------------------------------------------------------------

The Wave 7 brief considered three migration strategies for the
2-arg legacy path. **Option (a)** ("vacuum-stays-legacy") landed:
:class:`VacuumInflow` carried a standalone
:meth:`apply(psi_out, quad)` whose body was
``np.zeros_like(psi_out)`` (the pre-§16A.5 zeros-all form), and
the realizer path produced the inflow-only-mask form via
``IncomingOrdinateMaskTensor``.
The two paths agreed on inflow rows (the production-relevant
subset) and diverged on outflow rows.

Options (b) ("face-aware BC" — add ``face`` to the constructor)
and (c) ("combined ABC merge") were rejected because they would
have distorted the law's interface to better serve a transitional
path the refactor was retiring anyway.

**Status after Issue #186 (2026-05-11):** Option (a)'s standalone
``apply`` body has been **deleted**. :class:`VacuumInflow` (like
every other concrete BC) is a pure descriptor with no ``apply``
method; the only path to vacuum action is
:func:`realize_recursively` or :class:`SNBoundaryRealizer`. The
**two-paths-divergence is therefore eliminated by design** — there
is no longer a "second path" that could disagree with the realizer
path. What that single path *produces* has since changed again: the
§16A.5 inflow-only mask was the unique vacuum semantics from #186
until campaign phase **B3.2**, which replaced it with the zero map
:math:`\Gamma_+ \to \Gamma_-` (:ref:`bc-vacuum-semantic-correction`).
The uniqueness claim is what survived; the operator behind it did not.

This is the load-bearing architectural payoff of B3 + β2: the
documentation no longer needs to caveat which path you're on,
because there is only one path. See
:ref:`bc-trace-law-descriptor-model` for the design rationale.


.. _bc-trace-law-descriptor-model:

The trace-law descriptor model
==============================

Issue #186 / Scope B3 + β2 (landed 2026-05-11 on branch
``feature/bc-curvilinear-realizer-cleanup``) is the architectural
**closure** of the BC trace-law refactor. It collapses the
remaining 2-arg ``apply`` affordance from the Wave-8/9 era into a
**pure-descriptor** contract:

* :class:`BoundaryTraceLaw` no longer inherits
  :class:`~orpheus.numerics.operator.LinearOperator`.
* The abstract :meth:`apply` method that the mixin used to provide
  is gone. So is ``apply_transpose``. So is any operator-surface
  advertisement — the two-axis ``is_invertible`` / ``is_adjointable``
  predicates (before the #226 carve P4, the retired
  ``capabilities: ClassVar[frozenset[str]]`` frozenset).
* Every concrete BC (vacuum / reflective / white / albedo /
  periodic / prescribed-inflow / zero-flux) is now a **frozen
  dataclass** carrying only its parameters (axis, albedo, source,
  ...) and the relevant :meth:`assert_*` invariants. **No**
  ``apply`` method on any concrete BC. All three affine-factor
  properties are overridden on every concrete law since campaign
  phase B1 — see the measured table under :ref:`bc-law-layer`; at the
  time of Issue #186 only :attr:`source` was (by prescribed-inflow).
  :attr:`kind` is derived once on the ABC from the registry key,
  with :class:`~orpheus.geometry.boundary.ReflectiveBoundary` the
  single legitimate override (it reports ``"partial"`` at
  :math:`\alpha \neq 1`, matching the declaration vocabulary).
* The base class :class:`BoundaryTraceLaw` carries a **minimal
  algebra** that returns :class:`LawSum` / :class:`LawScaled`
  nodes — the descriptor-tree composition algebra documented at
  :ref:`bc-rank-n-algebra`. The dunders are: ``+``, ``-``, ``*``,
  ``/``, unary ``-``, plus their reflected variants. Each returns
  a new descriptor-tree node; none returns an operator.
* :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` is the
  **sole** bridge from descriptor to callable. There is no
  alternative path. Calling ``law.apply(psi)`` raises
  :class:`AttributeError` at runtime; a static type checker flags
  the call without running it.

The §16A.3 three-layer architecture (descriptor / realizer /
operator) is now **enforced by the type system**, not by convention.

What was tried and rejected before B3 + β2 landed
-------------------------------------------------

This subsection preserves the design history because the rejected
paths are the load-bearing intellectual content of the close-out
— future sessions asking "why does the realizer exist?" need to
see why every alternative failed.

**Option A** (Issue #176 / C176.3, ~2026-04). Concrete BC
``apply`` methods kept a keyword-optional
``quadrature: AngularQuadrature | None = None`` parameter with
defensive errors. Every ``apply`` named in this subsection is written
as a literal, not a role: the descriptor classes carry no ``apply``
today — that is the whole point of the bullet above — so a live
``:meth:`` role here would advertise a link to a method the type
system deliberately removed.

* ``ReflectiveBoundary.apply`` / ``WhiteBoundary.apply``
  raised :class:`BoundaryError` when ``quadrature is None``
  because their geometric / response operators needed the
  quadrature to construct themselves.
* :class:`VacuumInflow` / :class:`AlbedoBoundary` /
  :class:`PeriodicBoundary` / :class:`PrescribedInflow` accepted
  and ignored the ``quadrature`` parameter.

Option A was the **interim** landing — it preserved the
direct-call convenience pattern ("sketching code can write
``ReflectiveBoundary(axis='x').apply(psi, quad)``") while routing
production through the realizer. The C176.3 audit identified
three architectural costs that made Option A unsustainable:

1. **Asymmetric semantics on ``quadrature=None``.** Two BCs
   raised; four accepted-and-ignored. The behaviour was
   inconsistent across the law family and required per-BC
   documentation of "when is this method usable".
2. **Vacuum two-paths-divergence.** Direct ``VacuumInflow.apply(psi)``
   returned ``np.zeros_like(psi)`` (the pre-§16A.5 zeros-all body).
   The realizer-routed path returned
   ``IncomingOrdinateMaskTensor``
   output (the §16A.5 inflow-only mask). The two paths agreed on
   inflow rows (the production-relevant subset; see
   :ref:`bc-vacuum-semantic-correction`) but diverged on outflow
   rows. The divergence was harmless at every existing production
   call site; the cost argued at the time was a documentation-burden
   landmine for *future adjoint-sensitivity consumers that read
   outflow rows*. **That consumer was later measured not to exist**,
   and B3.2 removed the outflow rows from the law's domain
   altogether — so the cost was real (two paths, one contract) but the
   reason given for it was not. The retirement of Option A stands on
   costs 1 and 3.
3. **Liskov violation.** The abstract
   :meth:`BoundaryTraceLaw.apply(self, psi_out)` was strict 1-arg
   (post Issue #176 / C176.4). The concrete
   :meth:`apply(self, psi_out, quadrature=None)` was technically
   Liskov-substitutable (the optional parameter has a default), but
   calling ``bc.apply(psi)`` polymorphically on a
   :class:`BoundaryTraceLaw`-typed parameter could fail at runtime
   for Reflective/White — the type signature said "this works" but
   the runtime behaviour said "this raises". The static type
   system could not catch the failure because the contract was
   carried only in the docstring.

The B3 audit cataloged every remaining 2-arg ``bc.apply(psi,
quad)`` call site in production AND tests; none was load-bearing
for correctness. The Wave-6 snapshot regression carried legacy
halves (regenerated through the realizer path), the
:class:`PrescribedInflow` invariant tests ignored the
``quadrature`` argument anyway, and the realizer-vs-legacy
equivalence assertions could be replaced by hand-computed
expressions strictly stronger than legacy-agreement. The B3 sweep
rewrote every such site in one commit cycle (C-B3.7 through
C-B3.12) and deleted the ``apply`` methods.

**β1 interim** (sub-option within Issue #186 B3, considered but
not landed as a final state). Keep
:class:`LinearOperator` inheritance on
:class:`BoundaryTraceLaw` and drop only the abstract
:meth:`apply`. Rank-N composition would then build an
:class:`OperatorSum` of :class:`ScaledOperator`-wrapped *unrealized*
laws, and :func:`realize_recursively` would walk the resulting
Wave-0 composer tree. This β1 form was **algebraically equivalent**
to β2 but conflated two type families: the
:class:`OperatorSum` instance representing a not-yet-realized
descriptor tree was structurally identical to the
:class:`OperatorSum` representing an actual operator composition,
and only runtime inspection of the leaves could tell which was
which. β2 was preferred because the type system can then enforce
"this is a law, that is an operator" by static inspection — a
type checker rejects ``law_tree.apply(...)`` at the linter level
without ever running the code. β2 is more verbose (two new
dataclasses) but is the architecturally-checkable form.

**The vacuum two-paths-divergence is eliminated by design.** Under
B3 + β2 there is no "direct path" any more — every consumer must
realize the law before applying it, so there is exactly one vacuum
semantics at any moment (the inflow-only mask until campaign phase
B3.2, the zero map :math:`\Gamma_+ \to \Gamma_-` since). The Wave-6
snapshot harness's ``vacuum_lebedev17`` case tracks whichever it is —
it now pins the zero-map emission and its :math:`\Gamma_-` shape — and
no caller can accidentally invoke the pre-§16A.5 zeros-all path
because that path no longer exists in the code.

Empirical justification
-----------------------

The 18-test :mod:`tests.geometry.test_law_composition` suite pins
the descriptor-tree contract (foundation + L1 tests):

* Algebra closure on every dunder for every node-type pairing
  (laws × LawSum × LawScaled).
* :class:`LawScaled` constant folding
  (``2 * (3 * spec) == LawScaled(6, spec)``).
* The walker's exhaustive dispatch on the three node types and
  its :class:`TypeError` on unknown nodes.
* Walker value-correctness against hand-composed expectation
  (``realize_recursively(law_tree, ms, realizer).apply(psi)``
  ``== 0.3 * realize(spec).apply(psi) + 0.7 * realize(white).apply(psi)``).
* The absence of ``apply`` on any descriptor-tree node
  (``not hasattr(tree, "apply")``).

The reference-image harness verifies that the realized output matches
the independently-derived image on every case — bit-identically for
vacuum and the two specular rows, and inside the reduction-order bound
for the Lambertian and Marshak rows (the operator tree is structurally
the same; only the route to it changed). Until the 2026-08-01
re-anchoring the comparison was against the pre-B3 realizer-path
*output*; :ref:`bc-snapshot-reanchoring` records why that was a drift
lock rather than a reference, and what replaced it.

Call-site contract
------------------

There is **one** way to call a boundary law's ``apply``:

.. code-block:: python

   from orpheus.geometry.boundary import ReflectiveBoundary
   from orpheus.sn.boundary.realizer import (
       SNBoundaryRealizer, SNMethodSpace,
   )

   law = ReflectiveBoundary(axis="x", albedo=0.5)
   # A NARROWED law needs a FACE: since B3.2 its domain is that face's
   # Γ₊, which a quadrature alone cannot name. ``SNMethodSpace.minimal``
   # raises here — it survives only for the two laws still un-narrowed.
   ms = SNMethodSpace.for_face(quadrature=quad, face="xmax", trace=trace)
   op = SNBoundaryRealizer().realize(law, ms)
   psi_in = op.apply(gamma_out.apply(psi_face))   # Γ₊-shaped argument

For descriptor-tree composition:

.. code-block:: python

   from orpheus.geometry.boundary import realize_recursively

   tree = 0.3 * ReflectiveBoundary(axis="x") + 0.7 * WhiteBoundary(
       axis="x", outward_sign=+1,
   )
   op_tree = realize_recursively(tree, ms, SNBoundaryRealizer())
   psi_in = op_tree.apply(psi_out)

No call site routes through a putative ``law.apply(psi)`` — that
method does not exist.


.. _bc-numerical-evidence:

Numerical evidence
==================

The reference-image harness
(:mod:`tests.geometry.test_bc_equivalence_snapshot`) is the widest
mutation net in the boundary subsystem. Seven cases compare the
realized operator against a frozen ``.npz`` image at a per-case
tolerance.

.. _bc-snapshot-reanchoring:

The 2026-08-01 re-anchoring — from recording to reference
---------------------------------------------------------

Until campaign phase **B3.4b** these artefacts were **recordings of
production output**: the generator called
``SNBoundaryRealizer().realize(...)`` and froze ``op.apply(psi_out)``.
Two consequences followed mechanically. The assertion was
``production == a recording of production`` — a regression LOCK, worth
exactly what the recorded code was right, and unable to say the value
is *correct* (``vv-principles`` §bit-identity criterion 2). And it
broke on every signature change, which it did twice: at **B3.2**, when
the SN law narrowed to :math:`\Gamma_+ \to \Gamma_-`, and at
**B3.4a**.

The interim repair kept the frozen full-face artefacts and *restricted*
them, arguing that an artefact frozen before the narrowing is an
independent statement. That is only **procedural** independence: it
certifies *"the new path agrees with the old path"*, and the premise of
the whole narrowing campaign is that the old path read the wrong
half-trace.

So every case was **re-anchored against an expression derived from the
mathematics** — :eq:`affine-bc-form`, the mirror isometry
:math:`\Omega \mapsto \Omega - 2(\Omega\cdot\hat n)\hat n`, and the
Lambertian partial-current balance :math:`J^- = \alpha J^+`. The
generator now imports nothing from
:mod:`orpheus.sn.boundary.realizer`, which makes it refactor-proof by
construction, and the artefact **IS** the reference rather than a
recording.

The file stays frozen for one reason: a reference the harness
recomputed would let the generator's expression and production drift
*together*. The committed artefact is the barrier — an expression
change moves the gate only through a regeneration, which is a
reviewable diff.

**What the migration measured.** Against the retired pre-B3.2
recordings restricted to :math:`\Gamma_-`: ``vacuum``, ``specular_x``
and ``specular_y`` are **bit-identical**; ``white_xmax``,
``white_xmin`` and ``mixed`` differ by **1–2 ULP** (reduction order
only — the reference contracts with :func:`numpy.tensordot` where
production runs a broadcast-multiply-then-``sum``); ``periodic``
differs by **98 %**, by design. Six of seven claims did not move; what
changed is where they come from.

.. list-table:: BC reference-image cases
   :header-rows: 1
   :widths: 24 26 16 34

   * - Case
     - BC
     - Quadrature / face
     - Reference, and its tolerance
   * - ``vacuum_lebedev17``
     - ``VacuumInflow()``
     - Lebedev 17, ``xmin``
     - :math:`R = q = 0`, so the **zero map**: an image with
       :math:`\Gamma_-` shape, identically zero.
       ``assert_array_equal`` — no arithmetic is performed.
   * - ``specular_x_lebedev17``
     - ``ReflectiveBoundary(axis="x", albedo=1.0)``
     - Lebedev 17, ``xmax``
     - the mirror gather :math:`\psi^-(\Omega) = \psi^+(\Omega')`.
       ``assert_array_equal`` — reduction depth 0.
   * - ``specular_y_partial_07_LS6``
     - ``ReflectiveBoundary(axis="y", albedo=0.7)``
     - LevelSymmetricSN(6), ``ymax``
     - the same gather scaled by α — the α-fold row, and the only
       one on a non-``x`` axis. ``assert_array_equal``.
   * - ``white_xmax_LS4``
     - ``WhiteBoundary(axis="x", outward_sign=+1, albedo=1.0)``
     - LevelSymmetricSN(4), ``xmax``
     - isotropic re-emission with :math:`J^- = J^+` in the
       :math:`w_n|\Omega_n\cdot\hat n|` measure.
       ``rtol = |Γ₊|·ε``.
   * - ``white_xmin_partial_03_GL``
     - ``WhiteBoundary(axis="x", outward_sign=-1, albedo=0.3)``
     - GaussLegendre1D(8), ``xmin``
     - the same law at α = 0.3 on a quadrature whose :math:`\sum w`
       is 2, not :math:`4\pi` — the canary against a hard-coded
       normalisation. ``rtol = |Γ₊|·ε``.
   * - ``mixed_30spec_70white_LS4``
     - ``0.3 * spec + 0.7 * white`` (Wave-0 algebra)
     - LevelSymmetricSN(4), ``xmax``
     - the pointwise convex combination of the two images above.
       ``rtol = (|Γ₊| + 2)·ε``.
   * - ``periodic_lebedev17``
     - ``PeriodicBoundary()``
     - Lebedev 17, ``xmin`` ← ``xmax``
     - the PARTNER face's outflow at the same ordinates.
       ``assert_array_equal``, inside ``xfail(strict=True)``.

Every tolerance above is **derived, not measured-and-rounded-up**. The
old ``nulp = 4`` / ``nulp = 64`` constants encoded no claim; the bound
now states one. Recursive summation of :math:`n` terms carries a
relative error :math:`\le (n-1)\,u\,\kappa` with
:math:`u = \varepsilon/2`; every summand in the Lambertian is positive
(the probe is :math:`U(0,2)`, and :math:`w_n|\Omega_n\cdot\hat n| > 0`
on :math:`\Gamma_+`), so :math:`\kappa = 1` and :math:`|\Gamma_+|\cdot
\varepsilon` is that bound with room for the trailing division. The
mixed row adds one rounding for its scaling and one for the sum.
Measured against those bounds: 1.9e-16 (2.7e-15), 1.2e-16 (8.9e-16),
3.2e-16 (3.1e-15).

.. note::

   **The periodic row is a deliberate strict xfail, not a passing
   gate.** A translation identifies the two faces without touching
   direction, so a particle leaving through ``xmax`` in direction
   :math:`\Omega` re-enters at ``xmin`` in the SAME :math:`\Omega`:
   :math:`\Gamma_-(\text{xmin})` and :math:`\Gamma_+(\text{xmax})` are
   the same ordinate SET, and the law is a **two-face coupling**.
   Production realizes it as a per-face angular identity, so the
   narrowed composition
   :math:`\iota_- \circ \text{law} \circ \gamma_+` hands it this
   face's own :math:`\Gamma_+ = \{\mu_x < 0\}` and it feeds
   outgoing-left flux back in as incoming-right flux — measured 98 %
   relative error, an :math:`\mathcal{O}(1)` wrong answer.

   The artefact therefore carries TWO probes, drawn from independent
   seeds: with one shared draw a per-face endomorphism would look
   correct on the rows that coincide. **B3.4c** builds the partner-face
   wrap; the marker's deletion is forced by the XPASS(strict) failure,
   which is this campaign's standing technique — the xfail set IS the
   todo list. The body itself is already correct: a companion test
   shows the identity reproduces the reference exactly when fed the
   PARTNER's :math:`\Gamma_+`, so the defect is entirely in which
   half-trace the composition supplies.

.. note::

   **A narrowed row is re-posed onto** :math:`\Gamma_+`\ **, never
   weakened.** The B3.2 / B3.4a recipe was to feed the realized law
   ``snapshot["psi_out"][space.outflow_indices]`` and compare against
   the frozen pre-narrowing image restricted to :math:`\Gamma_-`. The
   2026-08-01 re-anchoring made that restriction unnecessary — the
   artefacts are now stored on the narrowed spaces directly, with the
   face and BOTH index sets alongside — and it kept the probe: each
   stored ``psi_out`` is the same deterministic full-face draw the
   retired artefacts used, restricted to :math:`\Gamma_+`. The
   migration changed the schema and the reference; it never changed
   what the operators are probed with.

   **The Marshak row stopped being an honest red at B3.4a.** While
   white was un-narrowed, ``0.3 * spec + 0.7 * white`` mixed a narrowed
   leaf with a full-\ :math:`N` one, so the sum's factors disagreed and
   its ``apply`` raised on a :math:`\Gamma_+` probe —
   ``AngularAverageOperator.apply: psi.shape[0] = |Γ₊|, expected N``.
   The row was pinned by ``xfail(strict=True)`` precisely so that the
   narrowing would force the marker's deletion. Both leaves are now
   narrowed and the composition is measured correct: it returns
   :math:`|\Gamma_-|` rows equal to
   ``0.3 * spec.apply(ψ₊) + 0.7 * white.apply(ψ₊)``. So the rank-N
   composition claim, which was **unchanged and un-weakened** but
   *unstateable*, is stateable again for this pair.

   .. warning::

      **A mix with an un-narrowed leaf no longer announces itself.**
      Do NOT generalize the old raise into a safety property. Measured
      post-B3.4a: ``0.3 * spec + 0.7 * albedo(0.5)`` **runs** and
      returns :math:`|\Gamma_-|`-shaped output that is silently
      wrong — the albedo leaf is a :math:`\Gamma_+ \to \Gamma_+` echo
      and :math:`|\Gamma_+| = |\Gamma_-|` swallows the mismatch. The
      raise was an accident of white's shape check, not a guarantee of
      the algebra; until B3.4b / B3.4c land, a mixed tree containing
      albedo or periodic is unsound and shape-invisible (vv Mode 12).

The specular rows are ``assert_array_equal`` and the choice is
structural, not optimistic: a gather introduces no re-association and
an α-fold is one multiplication of the same two floats, so the
predicted drift is EXACTLY zero — a tolerance there would admit the
failure mode the row exists to catch. (Before the re-anchoring these
rows carried ``nulp ≤ 4``, on a reasoning about the ``ScaledOperator``
wrapper's extra rounding step. That reasoning is right about the extra
rounding and wrong about its effect: the reference performs the same
multiplication, so both sides round identically.)

The ``.npz`` files live at
``tests/geometry/snapshots/bc_equivalence_*.npz`` and are committed to
the repository — the artefacts ARE the verification reference.
``tests/geometry/_generate_bc_equivalence_snapshots.py`` regenerates
them; since the re-anchoring, regeneration is legitimate only when the
reference EXPRESSION or the probe changes, never to make a red go
away — that would re-anchor the gate on the very code it gates, which
is the shape the inversion removed.

.. note::

   **The** ``albedo_05_lebedev17`` **case was retired at B3.4b.** It
   pinned ``psi_in == 0.5 * psi_out`` on the WHOLE FACE, an artefact of
   a law SN no longer realizes: a bare ``AlbedoBoundary(α)`` is refused
   because its :math:`R = \alpha I` is a :math:`\Gamma_+ \to \Gamma_+`
   endomorphism and its :math:`G` supplies no crossing (see
   :ref:`bc-method-realizability`). Its successor is the second method
   on the ``specular_x_lebedev17`` case, which pins the same α-fold on
   the same quadrature through the ``≡`` theorem
   ``AlbedoBoundary(α, SpecularReturn(a)) ≡ ReflectiveBoundary(a, α)``
   — and, since the re-anchoring, against the mirror isometry rather
   than against a sibling implementation's recorded output.


.. _bc-two-bc-applies-per-matvec:

Two BC apply calls per curvilinear matvec
=========================================

.. important::

   **Historical — this describes the Phase-D-era curvilinear matvec,
   which no longer exists.** Wave O step O.4a.2 deleted the keystone
   re-apply and made the sweep read ``ψ.boundary.inflow`` as *given*, so
   the matvec calls ``bc.apply`` **zero** times; the reflective coupling
   moved to the sibling :math:`-B` (:ref:`bc-extraction`). The Gate-1.5
   test named below was re-posed accordingly — it now pins the
   0-call extraction *and* the emitted WDD outflow, and asserting both
   together is what prevents a silent regression that re-absorbs the BC
   into the matvec. The vacuum realization named in the two call
   descriptions is likewise the pre-B3.2 one; see
   :ref:`bc-vacuum-semantic-correction`. The section is kept because
   the two-call decomposition it analyses is the reason the extraction
   needed a capture-and-compare gate rather than a round-trip check.

Phase C (:ref:`bc-trace-contract-respected-by-matvec`) established
that the SN curvilinear matvec applies the BC trace law **once per
matvec** at the boundary edge, consuming the WDD-propagated outflow
trace :math:`\gamma_+\psi`.  Phase D (Issue #168 Phase D, 2026-05-12,
landed on ``refactor/sn-operator-algebra``) extends the matvec with
a **second** BC apply call, used to build the Carlson coupled-pole
seed's ``bc_outer_value`` (see
:ref:`sn-phase-d-carlson-coupled-pole-sweep` in
:doc:`/theory/methods/sn/curvilinear_numerics` for the math).  The §16A.3 affine trace
law contract is therefore exercised **twice per matvec** in the
post-Phase-D code path:

.. list-table:: BC apply call sequence inside one curvilinear matvec
   :header-rows: 1
   :widths: 12 28 30 30

   * - Order
     - Caller / purpose
     - Input shape & meaning
     - Output use
   * - **#1**
     - Phase D Carlson context build
       (the then-production ``transport_operator_matvec_spherical``
       / ``_cylindrical`` matvec — the whole per-geometry family since
       deleted in the typed-field campaign (#197), successor retired at
       the walk unification (#280 campaigns) — early in the call)
     - ``(N, ng)`` — outer-cell cell-centred :math:`\psi` (NOT the
       face trace; a first-order proxy used only to construct the
       linear-in-:math:`\psi` ``bc_outer_value`` scalar at
       :math:`\mu = -1`)
     - Extract the most-inward-ordinate row; scalar feeds into
       ``CarlsonSweepContext.bc_outer_value`` (this Phase-D context
       object was later retired by Issue #282 route (a) — the
       starting-direction inflow corner is now a typed carrier slot; see
       :ref:`sn-direct-seed-solve`)
   * - **#2**
     - Phase C BC trace law application (at the boundary edge after
       the WDD sweep completes)
     - ``(N, ng)`` — WDD-propagated outflow face values
       :math:`\gamma_+\psi` (the §16A.3 contract input)
     - Fill the inflow rows used as
       :math:`\psi^{\text{face}}_{\text{in}}` for the inward sweep
       phase

The two calls are **structurally distinct**:

* **Call #1** is a Phase D-specific use of the BC operator as a
  *linear-in-ψ* construction of the inward-zero-weight ordinate's
  outer-face flux.  For vacuum BC the then-realized
  ``IncomingOrdinateMaskTensor`` zeroed the inflow ordinate,
  giving ``bc_outer_value = 0``; for reflective BC the
  :class:`PermutationOperator` mirrors outgoing :math:`\leftrightarrow`
  incoming, giving ``bc_outer_value = ψ_cell[N-1]`` (i.e. the
  cell-centred outer-cell value).  Both behaviours preserve operator
  linearity in the input :math:`\psi`.

  The input shape ``(N, ng)`` for cell-centres is a structural
  proxy: the BC trace law expects a trace ``(N, ng)`` shape, and
  feeding it the outer cell-centre array IS the right shape for
  Call #1's linear extraction.  The §16A.3 contract is not
  literally honoured here in the *interpretation* sense (the input
  is not a face trace), but the resulting scalar is linear in the
  matvec's input :math:`\psi` and gives the correct value at
  :math:`\mu = -1` on the only configurations the apply matvec
  cares about (reflective + flat :math:`\psi` :math:`\rightarrow` C,
  vacuum :math:`\rightarrow` 0).  This is the **principled
  shortcut**: a linearly-compatible scalar extraction whose values
  match the canonical inward-zero-weight ordinate's flux on the
  load-bearing test configurations.

* **Call #2** is the canonical Phase C use — the BC trace law
  consumes the **actual** WDD-propagated outflow face trace and
  produces the inflow trace per the §16A.3 affine-bc-form
  contract.  This is the call the Phase C
  :ref:`Gate 1.5 <bc-trace-contract-respected-by-matvec>` test
  pins.

Capture-and-compare Gate 1.5 strengthening
------------------------------------------

The pre-Phase-D Gate 1.5 test was a "round-trip" check: invoke
``bc.realize().apply(...)`` independently and compare against the
matvec's observable output.  Phase D **strengthens** the gate to a
capture-and-compare check that audits the *exact* value the matvec
passes into the BC trace law — necessary because the matvec now
calls ``bc.apply`` twice and the test must locate Call #2 (the
§16A.3 call) unambiguously.

The Phase D test
:func:`tests.sn.sweep.core.test_phase_c_gates.test_bc_trace_contract_capture_and_compare_sphere`
(parametrised over ``vacuum`` and ``reflective``):

#. Monkey-patches ``sn_mesh.bc["xmax"].apply`` (the outer radial
   face — a sphere's ``"outer"`` endpoint renders as ``"xmax"``)
   with a recorder wrapper that appends every input array passed to
   it during one matvec call.
#. Independently reconstructs the WDD-propagated outflow trace via
   a reference implementation
   (``_outflow_at_boundary_for_sphere(sn_mesh, sig_t, psi_input)``).
#. **Locates Call #2** by matching shape ``(N, ng)`` AND content
   (the captured input that bit-matches the independent reference
   IS the Phase C call).
#. Asserts the located input matches the reference to
   ``rtol=1e-14`` — bit-equal up to FP non-associativity.

The test is **foundation-tagged** (``@pytest.mark.foundation``)
because it pins a software invariant about the matvec's
two-application sequence, not a math claim about the BC operator.
The matching strategy by *both* shape and content protects against a
future regression that adds a third BC apply with the right shape
but wrong content — the test would still locate the canonical Phase
C call provided its content matches the WDD reference.

Both ``vacuum`` and ``reflective`` parametrised cases pass.  The
``vacuum`` case was the load-bearing check because Call #1 produced
non-trivial output under vacuum (the then-realized
``IncomingOrdinateMaskTensor`` zeroed inflow ordinates, so the
extracted ``bc_outer_value`` was zero — but the **input** to Call #1
was the outer cell-centre value, which is **not** zero on a
non-trivial :math:`\psi`).  Locating Call #2 unambiguously required
content matching, not just shape matching.


.. _bc-three-bc-applies-per-sweep-iteration:

BC applies in the SI sweep path
===============================

Phase D (Issue #168 Phase D, :ref:`bc-two-bc-applies-per-matvec`
above) instituted the *two BC apply calls per curvilinear matvec*
contract on the apply-matvec path (the within-group operator,
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` post-Depth-B; the
matvec then lived at ``_transport_operator_matvec_unified`` — since
deleted at the walk unification (#280 campaigns); the live forward action
is now :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply`
through the loss-representation walk).  Phase F
(Issue #168 Phase F, 2026-05-12, also landed on
``refactor/sn-operator-algebra``) propagates the same pattern to the
**SI/sweep path** (the then-production ``transport_sweep`` entry — since
retired at the coupled-block campaign step 6 (R-6.1, 2026-07-12) —
dispatching to ``_sweep_1d_spherical`` (the dissolved ``sweep.py``) /
``_sweep_1d_cylindrical``).  See
:ref:`sn-phase-f-carlson-sweep-path-backport` in
:doc:`/theory/methods/sn/curvilinear_numerics` for the math and the
twin-path-fix-incompleteness anti-pattern that motivated the
backport.

The post-Phase-F SI sweep iteration applies the BC operator in
**three** distinct invocations per sweep call — one for the Phase F
Carlson seed, plus :math:`N_{\text{inward ordinates}}` legacy inflow
applications inside the per-ordinate loop (which are not
fundamentally new — they predated Phase D and Phase F).  The
load-bearing addition is the **Phase F seed call**:

.. list-table:: BC apply call sequence inside one SI sphere sweep
   :header-rows: 1
   :widths: 12 28 30 30

   * - Order
     - Caller / purpose
     - Input shape & meaning
     - Output use
   * - **#1**
     - Phase F Carlson seed
       (``_sweep_1d_spherical`` (the dissolved ``sweep.py``) early in
       the call, before the per-ordinate loop)
     - ``(N, ng)`` — persistent outer-face outflow buffer
       ``bc_outer`` carrying the previous outward sweep's
       outgoing flux per ordinate (zero on the first SI
       iteration)
     - Extract the most-inward-ordinate row; scalar feeds into
       :func:`carlson_inward_sweep_from_source` as
       ``bc_outer_value`` (the seed for Hébert (3.434)–(3.435)
       at the outer face)
   * - **#2 … #N₋**
     - Per-inward-ordinate inflow read inside the
       per-ordinate sweep loop
     - ``(N, ng)`` — same ``bc_outer`` buffer (re-read each
       iteration, since intervening outward ordinates may have
       updated it)
     - Read the inflow row ``psi_in_full[n]`` for the current
       inward ordinate :math:`\mu_n < 0` as the spatial
       sweep's incoming flux

**Comparison with the apply-matvec twin** (per
:ref:`bc-two-bc-applies-per-matvec`):

* The apply matvec consolidates its inflow logic into the **single
  Phase C trace law call** at the boundary edge — the BC operator
  is invoked once on the WDD-propagated outflow trace, producing
  the inflow trace per the §16A.3 affine-bc-form contract for ALL
  ordinates simultaneously.
* The SI sweep, by contrast, processes ordinates **sequentially**;
  each inward ordinate independently reads its inflow row from
  the persistent ``bc_outer`` buffer.  The per-ordinate apply
  calls are **not** a §16A.3 trace-law application of the same
  semantic kind — they are *consumer reads* of an already-updated
  buffer.  The Phase F seed call (Call #1) is the only call that
  semantically mirrors Phase D's apply-matvec Call #1 (a
  Phase-specific use of the BC operator as a linear-in-:math:`\psi`
  construction of the inward-zero-weight ordinate's outer-face
  flux).

The Phase F seed call's role is exactly analogous to the
apply-matvec's Phase D Call #1 (per the
:ref:`bc-two-bc-applies-per-matvec` table): a
linear-in-:math:`\psi` extraction of the inward-zero-weight
ordinate's outer-face flux.  Vacuum BC zeros it; reflective BC
yields the most-inward ordinate's mirrored outflow.  In both
contexts the BC operator's **linearity** is the load-bearing
contract — the Phase F helper
:func:`~orpheus.sn.sweep.psi_half_angle_seed.carlson_inward_sweep_from_source`
must remain linear in the input to preserve the SI loop's
fixed-point convergence properties.

The cylindrical path has the analogous structure with a
**per-:math:`\mu`-level** seed call: one BC apply per level,
each invocation extracting the level-specific most-inward
ordinate's row from the same persistent ``bc_outer_cyl``
buffer.  The
``_sweep_1d_cylindrical`` (the dissolved ``sweep.py``) body invokes
the BC operator ``len(quad.level_indices) + N_inward`` times
total per sweep — once per :math:`\mu`-level for the Phase F
seed, plus once per inward ordinate inside each level's
azimuthal loop.

Phase F leaves the §16A.3 BC trace contract semantics
**unchanged**: the SI sweep path does not call the §16A.3 trace
law application that the apply matvec uses at the boundary
edge — the SI sweep updates ``bc_outer`` directly from each
outward ordinate's last visit (line ~593 of
:file:`orpheus/sn/sweep.py`), then the next inward ordinate
reads it via ``apply``.  The semantic contract is the same
(BC operator maps outflow trace :math:`\gamma_+\psi` to inflow
trace :math:`\gamma_-\psi`), but the *invocation pattern* is
per-ordinate sequential rather than once-per-matvec
collective.

No new Gate 1.5 test variant is needed for the Phase F seed
call.  The Phase F bit-identity test module
:mod:`tests.sn.sweep.core.test_sweep_vs_apply_consistency`
(57 foundation tests) pins that the sweep-path's
``bc_outer_value`` extraction matches the apply-path's Phase D
Call #1 result on every test configuration — the structural
invariant that the two paths' Carlson seeds agree on matching
inputs subsumes the BC-apply-input pinning.


.. _bc-curvilinear-realizer-unification:

Curvilinear realizer unification
================================

The pre-cleanup architecture carried a **Cartesian / curvilinear
split** at the mesh-side resolver then named ``SNMesh._resolve_one``
(retired at #290 P7b — the shipped hook is
:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law`):
the slab and 2-D Cartesian
paths constructed a trace space (then named ``InflowTraceSpace``,
unified into :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`
post-#188 and made geometry-blind in C5.3) and routed the BC through
:class:`SNBoundaryRealizer`, while spherical and cylindrical ``Mesh1D``
bypassed the realizer entirely because that factory
(``from_mesh_and_quadrature``, since C5.3
:meth:`AngularTraceSpace.from_quadrature_and_layout
<orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.from_quadrature_and_layout>`)
raised :class:`NotImplementedError` on those coord systems. The bypass
wrapped the bare law instance in
:class:`_BoundBoundaryOperator(law, quadrature=self.quad)`, a
dual-mode shim whose ``apply(psi)`` forwarded
``law.apply(psi, bound_quad)`` to the legacy 2-arg body.

**Why the split existed.** The Wave 2 ``InflowTraceSpace``
factory deferred curvilinear support because no curvilinear-Krylov
consumer at the time needed the per-face mask — the curvilinear
sweep paths computed the inflow / outflow predicate on the fly
inside the inner loop. The deferral was load-bearing for the
shim's dual-mode design: ``quadrature=None`` (Cartesian, wraps a
realized 1-arg op) and ``quadrature=<set>`` (curvilinear, wraps a
legacy 2-arg BC).

**Why the split has been removed.** Issue #188 / C188.1+C188.2
discovered that all three :class:`Mesh1D` coord systems
(``CARTESIAN`` / ``SPHERICAL`` / ``CYLINDRICAL``) share the same
``("left", "right")`` face structure with the radial axis as the
outward normal and the ``GaussLegendre1D``
adapter's :math:`\mu_x` as the direction cosine along that axis —
identically to the slab case. The mask predicate
:math:`\Omega \cdot \hat n < -\epsilon` therefore applies
unchanged. The factory's curvilinear guard was lifted; only 2-D
cylindrical :class:`Mesh2D` (which has no SN sweep in ORPHEUS
today) continues to raise :class:`NotImplementedError`.

Issue #188 / C188.3 then collapsed ``SNMesh._resolve_one``
to a single path: every supported mesh (1-D Cartesian / spherical
/ cylindrical + 2-D Cartesian) builds an
:class:`SNMethodSpace.for_face` and routes through
:class:`SNBoundaryRealizer`. Issue #176 / C176.1 then dropped the
``quadrature=`` kwarg from the shim because no
production-issued shim carried ``_quadrature is not None`` after
C188.3. Issue #176 / C176.3+C176.4 then trimmed the concrete BC
``apply`` signatures to the Option-A interim (keyword-optional
``quadrature=None`` with defensive errors). Issue #186 / B3 + β2
then retired Option A entirely — every concrete BC ``apply``
method was deleted; see
:ref:`bc-trace-law-descriptor-model` for the retrospective.

The architectural sequence is therefore:

* **Issue #188 unblocks Issue #176.** The shim's dual mode existed
  ONLY because curvilinear ``InflowTraceSpace`` could not be
  constructed. Once #188 lifted that, #176's "drop the 2-arg form"
  cleanup became possible without breaking curvilinear sweeps.
* **Issue #176 unblocks Issue #186.** The Option-A interim was
  necessary because dropping ``apply`` outright before the
  curvilinear sweeps consumed realizer output (#188) and the test
  fleet migrated to the realizer-routed contract (#176 / C176.5
  cleanup commits) would have broken curvilinear regression.
  Once those landed, the descriptor cleanup (#186 / B3 + β2)
  became the next step on the architectural ladder.

.. _bc-face-name-carve:

The face-name carve — one crosswalk, one face-keyed BC dict
===========================================================

Wave 8 through Issue #186 settled *how a single boundary law is
realized* (the law / realizer / shim split of
:ref:`bc-overview-three-layers`). What they did **not** settle is
*how the set of resolved laws is keyed and stored on the* ``SNMesh``.
Pre-C4 that storage was a hand-listed per-geometry construction with
named attributes — ``bc_xmin`` / ``bc_xmax`` / ``bc_ymin`` /
``bc_ymax`` (2-D), ``bc_left`` / ``bc_right`` aliases (1-D), plus a
pair of degenerate ``bc_ymin`` / ``bc_ymax`` placeholders on a slab
that no production code ever read. Three separate hand-lists carried
the same ``(axis, endpoint) → "{axis}{min|max}"`` knowledge, and a
fourth hand-list mapped a face name back to a reflection axis. C4
(part of the N-D layout campaign, Issue #220) collapses all four to
**one crosswalk function and one dict-comprehension loop**, keyed by
the same :class:`~orpheus.transport.mesh.axis.FaceLabel` inventory the trace
layout already derives from.

This is the storage-layer counterpart of the realizer unification
in :ref:`bc-curvilinear-realizer-unification`: that section made the
*realization path* uniform across geometries; this one makes the
*storage and keying* uniform across **dimensions**. After C4 a
3-axis mesh yields six face slots and six BC entries with **no edit**
to either producer — the pre-C4 3-branch body would have been
silently wrong the day it was reached. C5 (:ref:`sn-axis-primary-c5`)
admits exactly that 3-axis mesh — not via a hypothetical ``Mesh3D``
dataclass but via the axes tuple directly — and the face-name keying
built here carries it through unchanged.

The three string-named layers and the single crosswalk
-------------------------------------------------------

Three SN-side structures key on the same boundary-face string names
``"xmin"`` / ``"xmax"`` / ``"ymin"`` / ``"ymax"`` (and, in C5,
``"zmin"`` / ``"zmax"``):

.. list-table:: The three face-name-keyed layers and their shared crosswalk
   :header-rows: 1
   :widths: 26 30 44

   * - Layer
     - Structure
     - Role
   * - **Face layout**
     - :attr:`SNMesh.boundary_face_layout`
       (:class:`~orpheus.numerics.face_layout.FaceLayout`)
     - The flat-buffer descriptor: which faces exist, each face's
       per-face shape ``(N, ng, *codim-1 cells)``, and the offsets
       that pack them into one backing array.
   * - **Trace space**
     - :attr:`SNMesh._trace`
       (:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`)
     - The inner-product geometry on the boundary: per-face
       inflow / outflow ordinate masks over the signed
       :math:`\Omega\cdot\hat n` it carries (``trace.layout.faces``
       reproduces the same names).
   * - **BC dict**
     - :attr:`SNMesh.bc`
       (``dict[str, _BoundBoundaryOperator]``)
     - The resolved boundary operator per face — the realized
       1-arg law that maps an outgoing face trace to its incoming
       partner.

Pre-C4 each of these grew its key set from its own per-geometry
hand-list. The crosswalk knowledge — "axis 0 is ``x``; the
``min`` / ``max`` endpoints suffix the axis name; a solid radial
axis's single ``outer`` endpoint renders as the ``max`` face"
— was duplicated at the layout builder, the BC resolver, and the
reflective-axis dispatch. C4 lifts that knowledge into **one**
single-sourced rendering on the structural face key:

.. code-block:: python

   # orpheus/sn/axis.py
   AXIS_NAMES = ("x", "y", "z")
   _ENDPOINT_SUFFIX = {"min": "min", "max": "max", "outer": "max"}

   @dataclass(frozen=True, slots=True)
   class FaceLabel:
       axis_index: int
       endpoint: str   # "min" / "max" / "outer"

       @property
       def face_name(self) -> str:
           suffix = _ENDPOINT_SUFFIX.get(self.endpoint)
           if suffix is None:                       # fail loud
               raise ValueError(...)
           return f"{AXIS_NAMES[self.axis_index]}{suffix}"

:attr:`FaceLabel.face_name <orpheus.transport.mesh.axis.FaceLabel.face_name>`
is THE rendering of the structural identity ``(axis_index,
endpoint)`` into the ``"{axis}{min|max}"`` string world. Both
producers — :attr:`SNMesh.boundary_face_layout` and
:meth:`SNMesh.realize_boundary_law` — call it, so a key drift between the
face layout and the BC dict is **unrepresentable by construction**:
they cannot disagree because they read the same function over the
same :func:`~orpheus.transport.mesh.axis.face_labels` inventory.

.. note::

   ``AXIS_NAMES`` moved **down** from
   :mod:`orpheus.sn.loss_representation.sweep_graph` to :mod:`orpheus.transport.mesh.axis` in C4 —
   to the bottom of the SN dependency graph, next to the axis
   primitives it names. ``sweep_graph`` re-exported it only outward;
   ``sweep_schedule`` and ``loss_representation`` now import it
   downward. This puts the single source of the axis↔name crosswalk
   in the same module as :class:`~orpheus.transport.mesh.axis.FaceLabel`, the
   walk's in/outflow-face derivation, and the schedule's
   outgoing-face derivation — no consumer hand-lists
   ``("x", ...), ("y", ...)`` pairs any longer.

The ``"outer" → "max"`` convention and fail-loud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A solid sphere or cylinder is a :class:`~orpheus.transport.mesh.axis.RadialAxisMesh`
with a single ``"outer"`` endpoint (the pole at :math:`r=0` is
**not** an endpoint — see :ref:`bc-pole-structural-absence`). The
crosswalk renders ``"outer"`` as the ``max`` face of its axis, so a
sphere's outer radius is keyed ``"xmax"``. This is the **historical
curvilinear convention** — every curvilinear boundary operator,
trace face name, and sweep schedule already keys the outer radius on
``"xmax"`` — preserved verbatim by ``_ENDPOINT_SUFFIX["outer"] =
"max"`` rather than re-derived.

Any endpoint label that is **not** one of the three canonical
strings (``"min"`` / ``"max"`` / ``"outer"``) raises
:class:`ValueError`. An :class:`~orpheus.transport.mesh.axis.AxisMesh` exposes
user-overridable ``label_low`` / ``label_high`` fields (a slab user
may rename them ``"left"`` / ``"right"`` for convention); such a
renamed endpoint has **no face name** and must fail loud rather than
silently desynchronize from the ``"{axis}{min|max}"`` world that the
three layers key on. The failure surfaces at L0 — at the crosswalk
itself — not three layers up as a mis-keyed boundary operator or a
``KeyError`` deep inside a sweep.

The derivation chain — face_labels → {layout, bc}
-------------------------------------------------

The SN phase space factors as a tensor product of per-axis 1-D
meshes (grand report §15.1). The axis tuple
:attr:`SNMesh.axes <orpheus.sn.mesh.augmented_mesh.SNMesh.axes>` is therefore the root of every boundary-keyed
structure:

.. code-block:: text

   SNMesh.axes  (tuple[Axis1D, ...])
        │
        │  face_labels(axes) — one FaceLabel per (axis, endpoint),
        │  iterated axis-ascending then endpoint-in-axis-order
        ▼
   SNMesh.face_labels  (tuple[FaceLabel, ...])
        │
        ├─── boundary_face_layout : one slot per label,
        │       named  label.face_name,  shaped (N, ng, *face_shape(label))
        │
        └─── bc : one entry per label,
                keyed  label.face_name,  resolved from
                axes[label.axis_index].bc[label.endpoint]

The BC **declaration** source for each face is the per-axis
inventory ``axes[label.axis_index].bc[label.endpoint]`` — the
*same* axes that ``face_labels`` derives the labels from. The face
inventory **is** the BC inventory: a face that exists has exactly
one declaration; a face that does not (the pole) has no label and no
entry. The resolution loop is one comprehension:

.. code-block:: python

   # orpheus/sn/geometry.py — _resolve_bcs (post-C4)
   default = BC("reflective")
   self.bc: dict[str, _BoundBoundaryOperator] = {
       label.face_name: self._resolve_one(
           self.axes[label.axis_index].bc[label.endpoint] or default,
           label,
       )
       for label in self.face_labels
   }

``None`` on an axis defaults to ``BC("reflective")`` (the
infinite-lattice / eigenvalue convention). Each declaration is
realized by
:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law`
— the SN arm of the :class:`~orpheus.transport.method.TransportMethod`
hook, driven per face by the shared
:func:`~orpheus.transport.method.resolve_boundary_conditions` body
(the C4-era spelling was the mesh-private ``SNMesh._resolve_one``,
which owned the per-face loop itself; #290 P7b moved the loop up and
renamed the hook). Its realizer plumbing
(registry → ``SNMethodSpace.for_face`` →
:class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` →
``_BoundBoundaryOperator``) is
**unchanged** from the pre-C4 path — C4 changed only the *keying and
storage*, not the *realization*. Hence the resolved operators are
bit-identical objects to the pre-C4 ones (see
:ref:`bc-face-name-carve-verification`).

The :attr:`boundary_face_layout` producer is the dual comprehension:

.. code-block:: python

   # orpheus/sn/geometry.py — boundary_face_layout (post-C4)
   return FaceLayout.from_named_shapes([
       (label.face_name, (N, self.ng, *self.face_shape(label)))
       for label in self.face_labels
   ])

The pre-C4 body of each producer was a 3-branch ``isinstance`` /
coord-system split (1-D slab / 1-D curvilinear / 2-D Cartesian) that
hand-listed the face names. The dict-comprehension slot order
reproduces the historical hand-listed order **byte-for-byte** —
``face_labels`` iterates axis-ascending then endpoint-in-axis-order,
which is exactly the order the hand-lists used. The affine
``sha256`` goldens stayed byte-identical across the carve.

.. _bc-pole-structural-absence:

The pole is structurally absent, not null (Pattern 4 sharpened)
---------------------------------------------------------------

A :class:`~orpheus.transport.mesh.axis.RadialAxisMesh` has
``endpoints = ("outer",)`` — exactly one BC-bearing endpoint. A
solid sphere or cylinder therefore has a ``bc`` dict with **exactly
one entry** (``"xmax"``) and **no pole entry**. The geometric pole
at :math:`r=0` is the angular closure's regularity condition (the
:math:`1-\mu^2` redistribution coefficient vanishes at
:math:`\mu=\pm 1`; the inward sweep seeds from a moment-folded
source at :math:`\mu=-1` — see
:ref:`sn-pole-angular-closure-protocol` in
:doc:`/theory/methods/sn/curvilinear_one_group`), not a BC trace law.

C4 **sharpens** the pre-existing Pattern-4 treatment of the pole.
Pre-C4 the pole-as-non-BC was spelled by an explicit null:
``bc_left = bc_xmin = None`` — a named attribute that *exists* and
*holds* ``None``. Post-C4 the pole-BC is **structurally absent**: a
dict key that does not exist. Asking for it is a :class:`KeyError`,
not a ``None`` that a consumer might forget to guard:

.. list-table:: Pole-BC representation before and after C4
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Pre-C4 (``None`` placeholder)
     - Post-C4 (structural absence)
   * - Sphere ``bc`` surface
     - ``bc_xmin = None``, ``bc_xmax = <op>``
     - ``bc = {"xmax": <op>}``
   * - Asking for the pole
     - ``sn_mesh.bc_xmin`` → ``None``
     - ``sn_mesh.bc["xmin"]`` → :class:`KeyError`
   * - Failure mode of a buggy consumer
     - silent ``None.apply`` → ``AttributeError`` deep
       in a sweep, or a guard that *should* exist but
       doesn't
     - immediate ``KeyError`` at the access site — the
       illegal access is unrepresentable rather than null

This is the illegal-states-unrepresentable principle (Pattern 4)
applied to the dict: a pole-BC is not "a BC that is ``None``", it is
"not a face at all". The :func:`~orpheus.transport.mesh.axis.face_labels`
inventory simply does not emit a label for the pole, so neither
producer writes a slot or entry for it.

.. _bc-face-name-latent-d3-bug:

The latent d=3 axis-dispatch bug, closed by construction
--------------------------------------------------------

Before C4, ``SNMesh._resolve_one`` derived a reflective law's
reflection axis from a hand-listed membership test::

    axis = "y" if face in ("ymin", "ymax") else "x"

This is correct at :math:`d \le 2` by string coincidence — every
non-``y`` face is on the ``x`` axis. But a ``"zmin"`` / ``"zmax"``
face (a 3-axis mesh) would have fallen into the ``else`` branch and
silently built the **X-axis** reflection permutation — the *wrong
reflection partner*. A reflective law that reflects across the wrong
axis is a ``vv-principles`` **Mode-9 class** error (a law that is
wrong on a configuration the degenerate lower-dimensional test never
reaches): it would produce a plausible-but-wrong converged flux on a
3-D reflective problem, invisible to any :math:`d \le 2` test.

C4 derives the axis from the label's own
``AXIS_NAMES[label.axis_index]``, so the reflection partner is
correct at **any** dimension by construction::

    law = ReflectiveBoundary(axis=AXIS_NAMES[label.axis_index], albedo=1.0)

This is the boundary-resolution sibling of the C3.6 finding that a
z-face never sheds in the in-plane projection — both are latent
3-D correctness traps closed *before* ``Mesh3D`` exists, so that the
N-D layout campaign reaches C5 with the boundary keying already
3-D-correct.

.. note::

   **No ERR entry was filed for the latent d=3 axis bug.** When C4
   landed, no 3-axis mesh was constructible (the axis-native d=3
   admission did not arrive until C5 — :ref:`sn-axis-primary-c5`), so
   no 3-axis mesh had ever reached ``_resolve_one`` and **no production
   bug ever shipped**: the hand-listed dispatch was retired *before*
   any caller could exercise its wrong branch. The d=2 observable proxy
   (``test_2d_reflective_y_face_builds_y_axis_permutation``, with a
   non-vacuity guard that the x- and y-reflection maps differ under
   Lebedev) pins the *structural* correctness of the per-label axis
   derivation, which is what makes the d=3 extension correct by
   construction. The error catalog records *shipped* L0-caught bugs;
   a defect closed by construction before its triggering type exists
   is documented here, not in ``error_catalog.rst``.

Why string-keyed, not FaceLabel-keyed
-------------------------------------

Issue #220 allowed either a ``dict[str, ...]`` keyed by face name or
a ``dict[FaceLabel, ...]`` keyed by the structural label directly.
C4 chose **string-keyed**, isomorphic to
:attr:`FaceLayout.faces <orpheus.numerics.face_layout.FaceLayout>`,
for one reason: **every consumer iterates ``trace.layout.faces``
(strings)**. The within-group operator's
:meth:`SNBoundaryOperator._face_laws` and the schedule's
``reflective_faces`` both walk the trace layout's face-name strings
and index the BC by that string. A ``FaceLabel``-keyed dict would
force a reverse ``name → label`` lookup at *every* consumer, re-deriving
the very crosswalk C4 single-sources.

:class:`~orpheus.transport.mesh.axis.FaceLabel` remains the **structural source**
— it is the load-bearing key for the dim-agnostic face inventory,
the outflow-ordinate mask cache, and the sweep DAG's face-trace
state. ``face_name`` is its *single rendering* into the string world
the consumers already speak. One crosswalk function, called by both
producers, makes the keys of the layout and the BC dict identical by
construction:

.. math::
   :label: bc-face-name-key-identity

   \operatorname{set}(\texttt{sn\_mesh.bc})
   = \operatorname{set}(\texttt{boundary\_face\_layout.faces})
   = \{\, \ell.\texttt{face\_name} : \ell \in \texttt{face\_labels} \,\}

.. (vv-status rationale) Structural by-construction identity: the BC-dict keys,
   the boundary-face-layout faces, and the FaceLabel.face_name renderings are
   the same set because both producers call one crosswalk. The crosswalk is
   pinned by the foundation gate ``tests/sn/primitives/test_face_name_crosswalk.py``
   (the exhaustive (axis,endpoint)→face_name table + fail-loud negatives). A
   single-source-of-truth structural identity, not a solver claim.
.. vv-status: bc-face-name-key-identity documented

— a set equality that *cannot drift* because both sides are the same
comprehension over the same inventory.

.. _bc-face-name-carve-what-retired:

What C4 retired
---------------

C4 retires every pre-C4 named-attribute spelling of the resolved BC
surface, with no deprecation shim (aggressive retirement — a
read-through ``@property`` outliving its merge cycle would be the
very desync the carve removes):

* **The named instance attributes** ``bc_xmin`` / ``bc_xmax`` /
  ``bc_ymin`` / ``bc_ymax`` (2-D) and the ``bc_left`` / ``bc_right``
  aliases (1-D). Consumers now key into :attr:`SNMesh.bc` by face
  name. Accessing a retired attribute is an :class:`AttributeError`.
* **The degenerate 1-D y-face placeholders.** Pre-C4, a slab
  :class:`SNMesh` carried a pair of realized no-op
  ``ReflectiveBoundary(axis="y")`` operators at ``bc_ymin`` /
  ``bc_ymax``, routed through :meth:`SNMethodSpace.minimal` so
  cross-dimensional code could read them without coord-system
  gating. **No production code ever read them**: a 1-D mesh's
  ``trace.layout.faces`` is ``("xmin", "xmax")``, so the generic
  consumers (which iterate the trace layout) never asked for a
  y-face. The placeholders were a uniformity affordance with no
  consumer — exactly the kind of dead realized state the
  face-labels-derived dict makes unrepresentable: a slab has no
  y-axis in its :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.axes` tuple, so
  :func:`~orpheus.transport.mesh.axis.face_labels` emits no y-label, so
  :attr:`bc` has no y-entry, so ``slab.bc["ymin"]`` is a
  :class:`KeyError`. (Pre-C4 design rationale for *why the
  placeholders were once safe* is preserved in the
  :ref:`bc-curvilinear-realizer-unification` history above; C4
  removes the need for the rationale by removing the placeholders.)
* **The hand-listed reflective-axis dispatch**
  (``"y" if face in (...) else "x"``) — see
  :ref:`bc-face-name-latent-d3-bug`.

Consumers migrated in C4: :meth:`SNBoundaryOperator._face_laws`
(within-group operator) and ``sweep_schedule.reflective_faces``
(the schedule) both changed from ``getattr(mesh, f"bc_{face}")`` to
``mesh.bc[face]``, iterating over ``trace.layout.faces`` exactly as
before.

.. _bc-face-name-carve-verification:

Verification — bit-identity by inheritance + new L0 pins
--------------------------------------------------------

C4 is a **structural** carve: it changes keying and storage, not
the realized operators or any numerical value. The verification
strategy reflects that:

* **Bit-identity by inheritance.** A BC realization is object
  construction; the realizer plumbing (registry →
  ``SNMethodSpace.for_face`` →
  :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` →
  ``_BoundBoundaryOperator``) is unchanged — it is reached today
  through
  :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.realize_boundary_law`
  rather than the C4-era ``_resolve_one``. The resolved
  operators are the same objects as before, so every solver test
  that exercises them inherits its prior verification. The affine
  ``sha256`` goldens stayed byte-identical; the broad sweep /
  operators / primitives / solve suite is green.
* **L0 crosswalk pins**
  (:mod:`tests.sn.primitives.test_face_name_crosswalk`,
  foundation-tagged). An exhaustive **hand-transcribed**
  ``(axis, endpoint) → face-name`` table for :math:`d \in \{1,2,3\}`
  (mirror-not-import, so the test is not tautological against the
  production derivation it verifies), the d=3 ``z``-face admission
  (the crosswalk is a pure function, so the 3-axis rendering is
  verifiable **now** with no ``Mesh3D``), and both fail-loud
  negatives (a non-canonical endpoint → :class:`ValueError`; an
  axis beyond the named inventory → :class:`IndexError`).
* **L0 bc-dict / face-layout inventory pins**
  (:mod:`tests.sn.operators.test_snmesh_realizer_wiring`,
  foundation-tagged):

  - ``test_bc_inventory_equals_face_layout_across_geometries`` —
    ``set(sn.bc) == set(boundary_face_layout.faces)`` across slab
    (2 faces), 2-D Cartesian (4), sphere (1), cylinder (1), the
    Issue #220 acceptance set.
  - ``test_2d_reflective_y_face_builds_y_axis_permutation`` — the
    d=2 observable proxy for the latent d=3 axis-dispatch bug, with
    a non-vacuity guard asserting the x- and y-reflection maps
    differ under Lebedev (else the pin would be vacuous).
  - ``test_bc_dict_misses_and_retired_attributes_fail_loud`` — a
    face that does not exist is a :class:`KeyError` (plain dict, no
    masking default); every retired named attribute is an
    :class:`AttributeError` (no silent ``None``-shim survives).

The test-architect verification design memo is
``.claude/agent-memory/test-architect/c4_snmesh_bc_dict_verification.md``.

C4 closure
----------

The face-name carve lands in two parts under Issue #220 (the SN N-D
layout campaign): C4.1 (the :attr:`FaceLabel.face_name` crosswalk +
the :attr:`boundary_face_layout` loop collapse) and the bc-dict
resolution loop + named-attribute retirement. The carve is byte-identical
on every numerical output (affine ``sha256`` goldens unchanged) and
leaves the SN boundary keying **3-D-correct by construction** ahead of
``Mesh3D`` (C5). The realizer plumbing it sits on top of was unified by
the predecessor close-out below.

Predecessor closure — curvilinear realizer unification (Issue #188 + #176 + #186)
---------------------------------------------------------------------------------

The realizer path that C4 keys into was made uniform across geometries
by the curvilinear-realizer-unification arc
(:ref:`bc-curvilinear-realizer-unification`), closed by branch
``feature/bc-curvilinear-realizer-cleanup``
(2026-05-11). Three GitHub issues converged on that branch:

* **Issue #188** — curvilinear ``InflowTraceSpace`` support
  (commits ``9cf2b0a`` + ``17067d5``). Lifted the
  :class:`NotImplementedError` guard on spherical / cylindrical
  Mesh1D so every supported mesh can build a per-face inflow mask.
* **Issue #176** — drop 2-arg ``apply`` + simplify shim (commits
  ``cf29ce4`` + ``a4a43c2`` + ``913e501`` + ``188bf9a``). Collapsed
  the dual-mode shim into a strict 1-arg passthrough; landed the
  Option-A interim with keyword-optional ``quadrature=None`` on
  the concrete laws.
* **Issue #186 (B3 + β2)** — pure-descriptor cleanup (commits
  ``f71a32c`` + ``da414eb`` + ``89d09a4`` + ``633cc69`` +
  ``bb674da`` + the test-migration trail). Retired the Option-A
  ``apply`` methods, dropped :class:`LinearOperator`
  inheritance from :class:`BoundaryTraceLaw`, and formalised the
  descriptor-tree composition algebra via the new
  :class:`LawSum` / :class:`LawScaled` types. The architectural
  sequence is therefore Issue #188 → #176 → #186: each step
  unblocked the next.

The :class:`_BoundBoundaryOperator` shim survives because a resolved BC
must carry **both faces of its realization** — what it does
(:attr:`inner`) and what it means (:attr:`law`, since phase B2.0). Its
``kind``-string tag is the older, weaker reason: load-bearing for the
BC-resolution diagnostic and several ``sn_mesh.bc["xmin"] ==
"vacuum"``-style test sites (the dict-keyed spelling since C4 / #220;
this was ``sn_mesh.bc_left == "vacuum"`` pre-C4), and now a read-through
of ``law`` rather than a stored copy — phase B2.2 retires it, and the
shim survives that retirement on the descriptor alone. The dual-mode
bound-quadrature
backing is gone (#176), and the ``*_extra, **_kw`` swallow is
gone (#186 / C-B3.4). Every supported mesh consumes a strict
1-arg :class:`LinearOperator` produced by
:class:`SNBoundaryRealizer` for single BCs, or by
:func:`realize_recursively` for rank-N descriptor trees.

Plan documents:

* ``.claude/plans/transient-giggling-cake.md`` — the foundational
  12-wave BC trace-law refactor plan (Waves 0–12 close-out
  documented at :ref:`theory-boundary-conditions`).
* ``.claude/plans/curvilinear-realizer-and-2arg-cleanup.md`` —
  the #188 + #176 cleanup plan (Option-A landing).
* ``.claude/plans/bc-trace-law-descriptor-cleanup.md`` — the
  Issue #186 B3 + β2 cleanup plan (descriptor-model landing).

Grand Report v3 §16A.3 (the three-layer architecture) is now
**enforced by the type system** — descriptors have no ``apply``,
operators do. Grand Report v3 §16A.5 (the trace-correct vacuum
representation) is uniform across coord systems and the legacy
zeros-all path no longer exists.


.. _sn-axis-primary-c5:

The axis-primary inversion and 3-D admission
============================================

C4 (:ref:`bc-face-name-carve`) made the *boundary keying*
dimension-agnostic. C5 makes the **whole mesh** dimension-agnostic and
then admits the first 3-axis Cartesian :class:`SNMesh` — *without* a
``Mesh3D`` dataclass. The design fork (resolved by the user,
2026-06-11) is **axis-native**: a 3-D problem enters ORPHEUS only
through :meth:`SNMesh.from_axes` with a 3-tuple of
:class:`~orpheus.transport.mesh.axis.AxisMesh`. :class:`~orpheus.geometry.mesh.Mesh1D`
and :class:`~orpheus.geometry.mesh.Mesh2D` stay the :math:`d \le 2`
user-facing surface, bit-identical to before
(``sha256`` affine goldens unchanged, no regeneration). A ``Mesh3D``
would have had **exactly one consumer** (SN — ``cp`` / ``mc`` / ``moc`` /
``diffusion`` consume zero ``Mesh2D``); the "Unify after two instances"
discipline forbids minting a base type for a single consumer.

The campaign's keystone insight, surfaced by the C5 elegance audit, was
that the d=3 admission could not be a clean *extension* until a
pre-existing **data-flow inversion** in the constructor was repaired. C5
is therefore sequenced *clean before extend*: C5.1–C5.4 invert and
de-phantom the mesh layer, and only then C5.5 admits d=3 as a
one-line gate removal.

.. _sn-c5-lossy-roundtrip:

Pre-C5: the lossy axes → mesh → axes round-trip
-----------------------------------------------

The SN phase space factors as a tensor product of per-axis 1-D meshes
(grand report §15.1); the natural primary representation of an
:class:`SNMesh` is therefore its **axes tuple**
:attr:`SNMesh.axes <orpheus.sn.mesh.augmented_mesh.SNMesh.axes>`. Pre-C5.1 the constructor did not treat it that
way. :meth:`SNMesh.from_axes` *synthesized a legacy*
:class:`~orpheus.geometry.mesh.Mesh1D` / :class:`~orpheus.geometry.mesh.Mesh2D`
from the caller's axes (via ``legacy_mesh_from_axes``), handed that
mesh to ``__init__``, and ``__init__`` then **discarded the caller's
tuple and re-derived the axes from the synthesized mesh**:

.. code-block:: text

   from_axes(axes)                     __init__(mesh, ...)
        │                                    │
        │  legacy_mesh_from_axes(axes)       │  axes = axes_from(mesh)   ← re-derived
        ▼                                    ▼
   Mesh1D / Mesh2D  ───────────────────►  self.axes   (NOT the caller's tuple)

This ``axes → mesh → axes`` round-trip is **lossy** in two ways, and
its existence was the structural reason d=3 appeared to need a "third
construction arm":

1. **Custom endpoint labels were silently reset.** An
   :class:`~orpheus.transport.mesh.axis.AxisMesh` carries user-overridable
   ``label_low`` / ``label_high`` fields (a slab user may name them
   ``"left"`` / ``"right"``). The legacy mesh has no slot for those
   labels, so the round-trip dropped them and the re-derived axes came
   back with default labels — a silent desync of exactly the kind C4's
   :attr:`FaceLabel.face_name <orpheus.transport.mesh.axis.FaceLabel.face_name>`
   crosswalk relies on never happening.
2. **d=3 had nowhere to round-trip *through*.** A 3-axis tuple cannot
   synthesize a ``Mesh1D`` or ``Mesh2D``, so the inverted flow
   *mandated* a legacy mesh at every dimension — which is exactly the
   ``d \ge 3`` blocker the user directive named ("clean before
   extending").

C5.1 inverts the flow: the axes tuple becomes primary, stored verbatim.

.. _sn-c5-axis-primary-construction:

The axis-primary construction — one body, verbatim axes
-------------------------------------------------------

Post-C5.1, **both** entry surfaces funnel into one private body,
``_init_core``, which stores the axes tuple as-is:

.. code-block:: text

   SNMesh(mesh, ...)   ──►  axes = axes_from_legacy_mesh(mesh)   (convert ONCE,
                                                                  at the inbound
                                                                  boundary)
   from_axes(axes, ...) ─►  axes  (stored verbatim — the caller's tuple)
                                            │
                                            ▼
                                       _init_core(axes, ...)

The legacy ``SNMesh(mesh, ...)`` surface converts via
``axes_from_legacy_mesh`` **once**, at the inbound boundary
(parse-don't-validate); :meth:`SNMesh.from_axes` stores the caller's
tuple directly. There is no longer an ``axes → mesh → axes``
round-trip — the conversion is one-directional, ``mesh → axes``, and
only on the legacy surface.

The pre-C5.1 constructor branched on ``isinstance(mesh, Mesh1D)`` vs
``isinstance(mesh, Mesh2D)`` to compute per-dimension metadata (cell
widths, spatial shape). That **isinstance metadata branch dissolves**
into axis-derived properties. The single load-bearing identity is that
per-axis cell widths come from the axis edges:

.. math::
   :label: sn-axis-widths

   \texttt{axis\_widths}[a] = \operatorname{np.diff}(\texttt{axes}[a].\texttt{edges})

.. (vv-status rationale) Representational bit-identity: per-axis cell widths are
   np.diff of the axis edges, byte-identical to the retired Mesh1D.widths /
   Mesh2D.dx / Mesh2D.dy spellings. Pinned by the bit-identity gates
   ``tests/sn/primitives/test_axis_native_construction.py``
   (``test_d2_metadata_byte_identical_axis_vs_legacy`` /
   ``test_1d_slab_metadata_byte_identical_axis_vs_legacy``). A carve
   representational identity, not a solver claim.
.. vv-status: sn-axis-widths documented

This is **bitwise identical** to the legacy per-dataclass spellings it
replaces — :attr:`Mesh1D.widths <orpheus.geometry.mesh.Mesh1D>`,
:attr:`Mesh2D.dx <orpheus.geometry.mesh.Mesh2D>`, and
:attr:`Mesh2D.dy <orpheus.geometry.mesh.Mesh2D>` are each
``np.diff(edges)`` over the same edge arrays (``mesh.py:287`` /
``:567`` / ``:572``), so the carve produces the same floating-point
bytes. The whole-mesh coordinate system is likewise derived from the
per-axis coordinates by a new pure primitive
:func:`~orpheus.transport.mesh.axis.coord_system` (a multi-axis mesh must be
all-Cartesian); the constructor's reduced-operator dispatch and the
angular-closure default now read the **axis-derived** :attr:`SNMesh.coord <orpheus.sn.mesh.augmented_mesh.SNMesh.coord>`,
not ``mesh.coord``.

After C5.1, the ``mesh`` attribute of
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` is **inbound provenance
only** — it records *which legacy mesh the caller passed, if any*. It is
``None`` when the mesh was built from axes at :math:`d \ge 3` (no legacy
mesh exists to record; ``augmented_mesh.py`` spells the branch
``legacy_mesh_from_axes(axes, mat_map=mat_map) if len(axes) <= 2 else
None``). It is written here as a literal rather than an ``:attr:`` role
because the base ``MaterialMesh`` sets it on the *instance* with no
class-level annotation, so autodoc mints no target for it and a role
would render as unlinked text. A handful of :math:`d \le 2` consumers (the 1-D
reduced streaming constructors, the trace build, realizer metadata)
still read ``self.mesh`` at C5.1; each dissolves across C5.2–C5.5 as
its datum is repointed to an axis-native source. ``legacy_mesh_from_axes``
narrows from a round-trip *source* to a :math:`d \le 2` **adapter**
synthesis for those remaining consumers.

Custom endpoint labels now fail loud (C4 doctrine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the axes stored verbatim, a custom endpoint label survives
construction — and therefore reaches the
:attr:`FaceLabel.face_name <orpheus.transport.mesh.axis.FaceLabel.face_name>`
crosswalk. A label that is **not** one of the canonical strings
(``"min"`` / ``"max"`` / ``"outer"``) now raises :class:`ValueError`
**at construction** (the crosswalk's fail-loud — see
:ref:`bc-face-name-carve`), rather than being silently normalized away
by the round-trip. This is the C4 doctrine made operative by C5.1:
**overridable labels cannot silently desync the crosswalk**, so a label
the three face-keyed layers cannot key on must fail at L0, at the
construction site, not three layers up as a mis-keyed boundary
operator or a ``KeyError`` deep inside a sweep. (Pre-C5.1 the
round-trip *masked* this class of user error by discarding the label;
that masking is gone.)

.. _sn-c5-phantom-retirement:

The phantom shims retire (ny, dy, dx)
-------------------------------------

With per-axis widths and the rank-generic :attr:`SNMesh.spatial_shape <orpheus.sn.mesh.augmented_mesh.SNMesh.spatial_shape>`
now native, the legacy phantom-bearing metadata retires. Every spatial
read becomes rank-honest:

.. list-table:: The retired spellings and their rank-honest replacements
   :header-rows: 1
   :widths: 24 30 46

   * - Retired spelling
     - The phantom it carried
     - Replacement
   * - ``SNMesh.ny`` / ``SNMesh.dy``
     - At :math:`d = 1` these **lied** — ``ny`` returned a phantom
       ``1`` and ``dy`` a phantom ``[1.0]`` (the Issue #214 phantom
       class), and at :math:`d \ge 3` they underspecify the mesh.
     - :attr:`SNMesh.spatial_shape <orpheus.sn.mesh.augmented_mesh.SNMesh.spatial_shape>` (the per-axis cell counts) and
       :attr:`SNMesh.axis_widths <orpheus.sn.mesh.augmented_mesh.SNMesh.axis_widths>` (per-axis widths). ``AttributeError``
       on the retired names.
   * - ``SNMesh.dx``
     - A duplicate spelling of the per-axis widths.
     - :attr:`SNMesh.axis_widths <orpheus.sn.mesh.augmented_mesh.SNMesh.axis_widths>` — promoted from the private
       ``_axis_widths`` to **the** single public spelling of per-axis
       cell widths.
   * - ``SNMesh.nx``
     - (kept) Documented :attr:`spatial_shape[0] <orpheus.sn.mesh.augmented_mesh.SNMesh.spatial_shape>`
       sugar — honest at any :math:`d`, with a broad legitimate 1-D
       consumer base.
     - unchanged.

The phantom ``ny`` / ``dy`` at :math:`d = 1` is the **same defect
class** the N-D layout campaign closed at C2 / Issue #214: a trailing
singleton that masquerades as a real axis. The C5.2 retirement removes
the masquerade at the metadata source.

The two production ``dr`` consumers (the
:mod:`~orpheus.sn.loss_representation` 1-D bare sweep and the
:mod:`~orpheus.sn.angular.closure` Carlson preamble) repoint
from ``.dx`` to :attr:`SNMesh.axis_widths <orpheus.sn.mesh.augmented_mesh.SNMesh.axis_widths>`. The
field / cross-section / scattering read-through chains collapse to the
rank-generic :attr:`spatial_shape <orpheus.sn.mesh.augmented_mesh.SNMesh.spatial_shape>`:

* :class:`~orpheus.transport.fields.angular_flux.AngularFlux` (and the
  ``BulkField`` base) **retire** their ``nx`` / ``ny`` read-throughs.
  This is a live :math:`d = 3` landmine, not cosmetic: an
  ``(nx, ny)``-keyed field read **silently truncates** a 3-D tensor to
  its first two axes (a ``vv-principles`` Mode-2 / Mode-5 class
  index error that the degenerate :math:`d \le 2` test never reaches).
* :class:`~orpheus.transport.mesh.material_xs_field.MaterialXSField` and
  :class:`~orpheus.transport.operators.scattering.ScatteringOperator` collapse their
  ``nx`` / ``ny`` reads to **one** rank-generic
  :attr:`spatial_shape <orpheus.sn.mesh.augmented_mesh.SNMesh.spatial_shape>` read-through each.

Finally, a new :attr:`SNMesh.volume_measure <orpheus.sn.mesh.augmented_mesh.SNMesh.volume_measure>` property gives the
SN-side ``keff`` rate consumers (the production / absorption rates in
:mod:`~orpheus.sn.solver`) a native source: they read it instead of
reaching through ``sn_mesh.mesh.volume_measure``. While the
:math:`d \le 2` adapter is present it delegates to the dataclass
measure (bit-identical, including the ``precomputed_volumes`` hatch);
the axis-native arm lands with C5.5.

.. _sn-c5-geometry-blind-trace:

The geometry-blind trace space (z faces admitted)
-------------------------------------------------

The trace layer (:ref:`bc-trace-structure`) carried two C5-blockers:
a hand-listed face-normal table that **silently lacked the z faces**,
and a gate-only ``mesh`` parameter on the trace factory.

**The face-normal source collapses onto** ``AXIS_NAMES``. Pre-C5.3,
``trace_space._FACE_NORMALS`` was a hand-listed **four-entry**
transcription (``xmin`` / ``xmax`` / ``ymin`` / ``ymax``) — it had no
``zmin`` / ``zmax`` rows, the silent :math:`d = 3` blocker. C5.3
derives the table from :data:`~orpheus.numerics.face_layout.AXIS_NAMES`
so every axis-aligned face (all **six** at :math:`d = 3`) is present by
construction: face ``"{axis}min"`` has outward normal
:math:`-\hat e_{\text{axis}}`, face ``"{axis}max"`` has :math:`+\hat
e_{\text{axis}}`. The :math:`\Omega\cdot\hat n` row for ``zmax`` is
then exactly :math:`+\mu_z` and for ``zmin`` exactly :math:`-\mu_z`.

To share the ``"{axis}{min|max}"`` crosswalk without an ``sn``-ward
import from the ``numerics`` layer, ``AXIS_NAMES`` **moved down** to
:mod:`orpheus.numerics.face_layout` — the home of
:class:`~orpheus.numerics.face_layout.FaceLayout`, keeper of the face
string-name world. :mod:`orpheus.transport.mesh.axis` re-exports it, so SN consumers
are unchanged; the trace space (a ``numerics`` leaf) now reads it
without depending on ``sn``.

**The trace factory is geometry-blind.** The mesh parameter on the old
``from_mesh_and_quadrature`` factory was **gate-only** — its single use
was an ``isinstance`` check that refused a curvilinear ``Mesh2D``. That
refusal is **unreachable**: a curvilinear ``Mesh2D`` cannot become an
:class:`SNMesh` in the first place (2-D cylindrical SN has no sweep), so
no such mesh ever reached the factory. The ``isinstance`` check carried
no data the factory used. C5.3 therefore renames the factory to
:meth:`AngularTraceSpace.from_quadrature_and_layout
<orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.from_quadrature_and_layout>`
and **retires the dead mesh parameter** (aggressive retirement; callers
and the bare-constructor error message migrated). Every datum now comes
from the quadrature (the :math:`\mu_x` / :math:`\mu_y` / :math:`\mu_z`
cosines) and the layout's face names (the axis-aligned normals implied
by the ``"{axis}{min|max}"`` convention).

With the gate gone, :meth:`SNMesh.realize_boundary_law` builds the trace
**unconditionally** (the pre-C5.3 ``isinstance`` gate excluded only the
unconstructible 2-D cylindrical mesh), and :attr:`SNMesh.angular_trace` is typed
and documented as **always non-None**.
:meth:`SNMethodSpace.for_face <orpheus.sn.mesh.method_space.SNMethodSpace.for_face>`'s
``mesh`` parameter becomes **optional metadata** — nothing in the
realizer chain reads it, and an axis-native :class:`SNMesh` passes
``None``.

.. note::

   **The fail-loud rank-mismatch raise (the C5.3 elegance-review
   carry, landed in C5.5).** A trace built for an axis-:math:`k` face
   on a quadrature that lacks ``mu_k`` previously **zero-padded** the
   :math:`\Omega\cdot\hat n` row to all-tangential — silently
   producing a face on which *no* ordinate is inflow or outflow. C5.5
   makes ``_build_omega_dot_n`` raise loudly on that rank mismatch:
   asking for a ``"zmax"`` face from a 2-D quadrature is a
   construction error, not a silent all-tangential face.

.. _sn-c5-windowing-gs-gate:

Windowing and Gauss–Seidel gate on genuine dimensionality (vv Mode 9)
---------------------------------------------------------------------

C5.4 is the **highest-risk edit of the campaign** — a textbook
``vv-principles`` **Mode 9** case (a splitting / optimization verified
only in a regime where the wrong gate is *accidentally* satisfied). Two
gates inside the SN source-iteration driver keyed on
``sn_mesh.reduced is None``:

* the **moment-windowing** gate (:meth:`_maybe_window
  <orpheus.sn.solver>`), which decides whether the SI iterate is held
  as compact harmonic moments rather than the full angular flux; and
* the **Gauss–Seidel splitting** selector
  (:func:`~orpheus.sn.solver._select_si_splitting`),
  which decides whether the boundary-G-S accelerator is used.

``reduced is None`` is a **coincidence proxy**: it is ``None`` for
*every* multi-D Cartesian mesh, including a :math:`d = 3` one. The
moment-windowing path's in-sweep moment-emission kernel is **2-D
only** (it indexes a ``(N_oct, ng, nx, ny)`` block; see Issue #227).
So at :math:`d = 3` the old proxy would have **silently
moment-windowed the SI iterate** — the
:class:`~orpheus.sn.loss_representation.FullFieldWavefront` spine
refuses moment mode on the Jacobi path and ``None``-subscripts on the
G-S path: a **corrupted iterate, not a principled refusal**. This is
precisely the Mode-9 failure: a :math:`d \le 2` test cannot observe it
because at :math:`d \le 2` ``reduced is None`` *and* the 2-D kernel is
correct, so the proxy and the truth coincide.

C5.4 retargets both gates to the **genuine** dimensionality predicate:

.. list-table:: The Mode-9 gate retarget
   :header-rows: 1
   :widths: 34 30 36

   * - Gate
     - Pre-C5.4 (coincidence proxy)
     - Post-C5.4 (genuine condition)
   * - Moment windowing (``_maybe_window``)
     - ``reduced is None``
     - ``is_cartesian and ndim == 2`` — the genuine
       windowing-eligibility condition (the 2-D moment kernel's exact
       domain).
   * - Boundary-G-S (``_select_si_splitting``)
     - ``reduced is None``
     - ``is_cartesian and not is_1d`` — multi-D Cartesian.

The G-S resolvent's old ``"2-D Cartesian ONLY"`` docstring was **stale
Phase-3 narration**: :attr:`SweepSchedule.gauss_seidel
<orpheus.sn.loss_representation.sweep_schedule>` and the scheduled sweep
(``_sweep_scheduled``) have been **d-generic since C3**, so the
resolvent is constructible at :math:`d = 3`. The narration is corrected
in C5.4; the actual :math:`d = 3` boundary-G-S *fixed-point invariance*
(that G-S and Jacobi converge to the **same bulk** flux) is
**value-gated**
by the C5.5 Mode-9 mixed-BC box (:ref:`sn-c5-value-gates`) before any
:math:`d = 3` G-S solve is trusted — the Mode-9 discipline made
operative: never gate a splitting's FP-invariance on a degenerate box.

.. note::

   ⚠ **The word "bulk" above is load-bearing, added 2026-08-15 (#344),
   and this gate does not need re-scoping — but its measurands do need
   naming.**  The C5.5 box is x-reflective / y-vacuum / z-reflective,
   i.e. it closes **two** reflective axis pairs, and ``[M]`` that makes
   :math:`A = L+C-S-N_{2n}-B` **exactly singular** there:
   :math:`\dim\ker A =
   36` at cells :math:`(5,3,4)`, ``level_symmetric`` :math:`S_4`,
   :math:`n_g = 2` (:math:`= n_g (N/4)\, n_y`).  So G-S and Jacobi do
   **not** return the same boundary trace on it.  The gate is sound
   because what it asserts — :math:`k_{\rm eff}` and the normalized
   flux *shape* — is mirror-even, and every mirror-even functional
   annihilates :math:`\ker A` by theorem
   (:eq:`sn-kernel-mirror-blindness`).  ⟹ a future strengthening of
   this gate must not reach for the raw trace without gauging it
   first.  Derivation: :ref:`sn-loss-kernel-gauge`.

The one ``reduced is not None`` branch that **stays** is the 1-D
sweep-cache: there the predicate keys on the *availability of the data
it reads* (the reduced-operator cache), not on dimensionality. That is
not a proxy — it is the genuine guard.

.. _sn-c5-3d-admission:

3-D Cartesian admission: the axes tuple is the only 3-D entry
-------------------------------------------------------------

After the C5.1–C5.4 cleanup, the :math:`d = 3` admission is an
**extension, not a new arm**. A 3-axis Cartesian :class:`SNMesh` now
constructs and **solves** through the same generic body as
:math:`d \le 2`, **mesh-adapter-free from birth** (``self.mesh is
None``) on the d-generic
:class:`~orpheus.sn.loss_representation.FullFieldWavefront` spine.

* **The gate retires.** :meth:`SNMesh.from_axes` drops the
  ``d \ge 3`` admission guard; :math:`d \le 2` still synthesizes the
  legacy adapter for its remaining consumers.
* **Axis-native arms.** The cell-volume array is the iterated outer
  product of the per-axis widths,
  :math:`V[i,j,k] = \mathrm{d}x_i\,\mathrm{d}y_j\,\mathrm{d}z_k`; the
  :attr:`volume_measure <orpheus.sn.mesh.augmented_mesh.SNMesh.volume_measure>` is the rank-:math:`d`
  meshgrid-of-centers
  :class:`~orpheus.numerics.measure.DiscreteMeasure` (the natural
  rank-:math:`d` generalization of the ``Mesh2D`` analogue).
* **Entry surface.** :func:`~orpheus.sn.solver.solve_sn` and
  :func:`~orpheus.sn.solver.solve_sn_fixed_source` accept the **axes
  tuple** — the *only* 3-D entry — through one inbound seam
  (``_as_sn_mesh``). A new ``mat_map`` keyword is the axes-entry
  material channel (it raises if combined with a legacy mesh, which
  carries its own material map). Default-BC semantics are handled per
  surface (``_apply_default_bcs`` accepts both declaration styles —
  per-face dataclass fields *or* per-endpoint axis slots — with the
  same all-or-nothing semantics).

.. note::

   **Two default-BC conventions, by design.** The *solver* entry
   defaults un-declared faces to **vacuum** (the fixed-source
   convention — an un-specified boundary leaks); a freshly constructed
   :class:`SNMesh` with no BC declarations defaults to **reflective**
   (the infinite-lattice / eigenvalue convention — see
   :ref:`bc-face-name-carve`). The d=3 admission preserves **both**
   conventions on their respective surfaces; the value gates below
   exercise the reflective (eigenvalue) convention for the headline
   :math:`k_\infty` identity and a mixed convention for the Mode-9 box.

.. _sn-c5-value-gates:

Numerical evidence — the d=3 value gates
----------------------------------------

C5.5's admission is gated by four value tests
(:mod:`tests.sn.solve.test_d3_admission`), **all driven through the
production entry points** (``np.testing.assert_*`` only — Mode-8 safe
under ``python -O``, where bare ``assert`` is stripped). Each probes a
distinct failure class:

.. list-table:: The d=3 admission value gates
   :header-rows: 1
   :widths: 30 16 54

   * - Gate
     - V&V level
     - Evidence
   * - **k_inf 3-D ≡ 2-D ≡ 1-D**
     - L1 (closed-form eigenvalue)
     - Homogeneous all-reflective boxes at :math:`d = 1, 2, 3`. The
       reference is the closed-form matrix eigenvalue
       :math:`k_\infty = \lambda_{\max}(A^{-1}F)`,
       :math:`A = \operatorname{diag}(\Sigma_t) - \Sigma_{s0}^{\mathsf T}`
       — **never** the sweep. Each dimension matches ``case.k_inf`` to
       ``atol=1e-8``; the d=3 box solved
       :math:`1.8750000050` against the closed form :math:`1.875`
       (2g). Run at **2 groups and 4 groups, never 1 group** — a 1-G
       eigenvalue is the flux-shape-independent ratio
       :math:`\nu\Sigma_f/\Sigma_a` and is degenerate.
   * - **Per-ordinate ψ = Q/(W·Σₜ)**
     - L1 (closed-form flux)
     - Pure absorber (:math:`c = 0`), all-reflective. DD is flat-flux
       *exact* and :math:`c = 0` needs no iteration, so **every**
       ordinate must carry the closed-form value
       :math:`\psi_{n,g} = Q_g/(W\,\Sigma_{t,g})` to ``rtol=1e-10``.
       Per-group distinct :math:`Q` and :math:`\Sigma_t` make a group
       swap (Mode-2) observable; the per-ordinate residual is the
       sharpest Mode-1 / Mode-3 / Mode-4 probe.
   * - **Scattering multigroup balance**
     - L1 (closed-form flux)
     - Scattering medium, all-reflective:
       :math:`\phi = (\operatorname{diag}(\Sigma_t) -
       \Sigma_{s0}^{\mathsf T})^{-1} Q`. The group-coupling companion —
       a **Mode-6 convention-drift catcher** because mixture C's
       scattering matrix is **asymmetric**, so :math:`\Sigma_s` vs
       :math:`\Sigma_s^{\mathsf T}` (the ``SigS`` / ``SigS^T``
       convention, see :ref:`theory-discrete-ordinates`) is observable.
       Measured max relative error :math:`2.6\times 10^{-9}` — this is
       **SI-convergence-limited, not a discretization error** (DD is
       flat-flux exact on a homogeneous box).
   * - **Mode-9 G-S ≡ Jacobi FP-invariance** (of the **bulk** — the
       box is kernel-BEARING, see the note above)
     - L2 (integration)
     - Boundary-Gauss–Seidel and Jacobi converge to the **same** d=3
       bulk on a box that **breaks every degenerate
       coincidence**: mixed BCs (x-reflective / y-vacuum / z-reflective
       — axis-asymmetric, so a wrong reflection partner shifts the
       answer), ``nx ≠ ny ≠ nz`` (5, 3, 4), a heterogeneous 2-G split
       across x (a non-flat-flux guard), and a **diagonal**
       level-symmetric cubature (ERR-056 shared-face discipline —
       diagonal cubatures share faces between octants, the regime where
       a wrong G-S shared-face reflect is observable). :math:`k_{\rm
       eff}` agrees to ``atol=1e-8`` *and* the normalized flux shape to
       ``rtol=1e-6``.

The four gates together cover the verification ladder for the new
capability: a closed-form eigenvalue (L1, the only pillar that can
verify :math:`k`), two closed-form flux identities (L1, isolating the
streaming / collision / scattering operators per-term), and a
splitting FP-invariance on the degenerate-breaking box (L2 / Mode-9).
The eigenvalue gate's reference is **structurally independent** of the
sweep (a matrix eigensolve, not a transport solve), satisfying the
``vv-principles`` requirement that an eigenvalue claim rest on a
closed-form or semi-analytical reference rather than MMS.

What runs the d=3 path, and what is deferred
--------------------------------------------

The :math:`d = 3` admission runs on the **d-generic
FullFieldWavefront ORACLE spine** — the never-stuck full-field
representation that is correct from day one (the four value gates), but
**not** the optimized sweep kernels. Two kernel widenings are deferred
to Issue #227, gated on *measurement* against the spine (the C3.6
principle: "construct general, select narrow, specialize only on
measured cost"):

* **ScanMarch** :math:`d \ge 3` — the row-march kernels currently
  unpack 2-D pairs; the
  :math:`\text{scan}(x)\circ\text{march}(y, z)` generalization widens
  the predicate **only with** the kernel and a profile showing it beats
  the spine.
* **MovingFrontierWindow** :math:`d \ge 3` — the rolling-frontier
  window is built ``frontier_dim = d-1`` and its ``supports`` is
  conservatively ``is_cartesian and ndim == 2``; the :math:`d = 3`
  windowed *walk* is graph-layer-pinned but the window kernels need
  their own profile (the 2-D window was a ~0.71–0.80× **speedup** plus
  a peak-memory win — the :math:`d = 3` economics need separate
  numbers).

Separately, the **multi-D adjoint** (``loss_action_transpose`` raises
:class:`NotImplementedError` at any multi-D) is a **pre-existing
deferral** orthogonal to C5 (G-adjoint campaign territory), noted here
only so the :math:`d = 3` capability map is complete.

C5 closure
----------

C5 lands in six substeps under Issue #225 (the SN N-D layout campaign):
C5.1 (axis-primary inversion), C5.2 (phantom-shim retirement +
native ``coord`` / ``volume_measure``), C5.3 (geometry-blind trace),
C5.4 (Mode-9 windowing / G-S gate retarget), C5.5 (3-D admission), and
C5.6 (structure-pin flips to the now-constructible mesh). The
:math:`d \le 2` path is **byte-identical** on every numerical output
(the affine ``sha256`` goldens are unchanged across the whole carve);
the :math:`d = 3` path is correct from birth on the FullFieldWavefront
spine, value-gated by the four tests above. The campaign reaches its
3-D admission **without** a ``Mesh3D`` dataclass — the axes tuple,
made primary by C5.1, *is* the N-D entry.


Anti-pattern catalog
====================

A short list of patterns the refactor's authors considered and
rejected, with the reasoning preserved so future sessions don't
re-attempt them:

1. **Single ``BoundaryOperator`` ABC carrying both law and
   realizer responsibilities.** This is the pre-refactor shape.
   Rejected because the law / realizer split is the architectural
   point of the refactor: laws are method-agnostic, realizers are
   method-specific. Keeping them in one class would force every
   law to know about the SN sweep's half-trace plumbing (an inflow
   mask at the time; the :math:`\gamma_\pm` restrictions since B3.2),
   which is precisely what Wave 0 / Wave 1 / Wave 2 was supposed to
   abstract away.
2. **Dedicated ``MixedBoundaryOperator`` class.** Pre-Wave-11
   shape. Rejected (deleted Wave 11) — the Wave-0 algebra dunders
   on every :class:`BoundaryTraceLaw` already produce
   :class:`OperatorSum` shapes; the dedicated class added a
   second realization path with no semantic difference. See
   :ref:`bc-rank-n-algebra`.
3. **Shared ``BoundaryRealizerBase`` ABC for cross-method
   realizers.** Considered Wave 5, when only one functional
   realizer existed. Rejected per the "Unify after two instances"
   architectural discipline: building the abstraction on a single
   instance would force a particular shape on every future method
   based on SN's dispatch idiom (``isinstance``). Vindicated at
   method #2: the diffusion realizer (#290 P3) chose a DIFFERENT
   shape (law → albedo scalar → structure-keyed collapse, not an
   isinstance ladder over per-law primitives), and the structural
   :class:`~orpheus.geometry.boundary.BoundaryRealizer` Protocol
   remains the only shared contract — no ABC was ever needed.
4. **Adding ``face`` to ``VacuumInflow``'s constructor for
   semantic correctness on the standalone-apply path.** Option (b)
   in :ref:`bc-vacuum-semantic-correction`. Rejected because it
   would have inflated every test signature for one wave to fix a
   path that the refactor was retiring anyway. The transitional
   legacy-vacuum body is the right cost.
5. **Auto-importing every cross-method realizer at
   :mod:`orpheus.geometry.boundary` import time.** Considered in
   the registry era to make ``BoundaryRealizerRegistry.get("MoC")``
   work without the caller having to ``import orpheus.moc`` first.
   Rejected because :mod:`orpheus.sn` is a heavy module that every
   consumer of the boundary package would then pay for. #290 P7b
   made the whole question moot by dissolving the registry: each
   method-mesh imports its own realizer explicitly, so there is no
   name-lookup to keep populated and no import-side-effect timing
   to defend.
6. **Cartesian-vs-curvilinear bypass in
   ``SNMesh._resolve_one`` + dual-mode shim.** Pre Issue #188
   shape: curvilinear ``Mesh1D`` bypassed the realizer and wrapped
   the bare 2-arg law in
   ``_BoundBoundaryOperator(law, quadrature=self.quad)``, while
   Cartesian routed through the realizer with
   ``_BoundBoundaryOperator(realized)``. **Retired Issue #188 +
   #176**; documented here because the seductive trap is to
   "preserve flexibility" by keeping the dual mode after the
   curvilinear deferral lifts. The right move is to delete the
   bypass and consolidate on one path — see
   :ref:`bc-curvilinear-realizer-unification`.
7. **Option A (keyword-optional ``quadrature=None`` on the
   concrete laws).** Landed Issue #176 / C176.3 as the interim
   form; **retired Issue #186 / B3 + β2** in favour of the
   pure-descriptor model (no ``apply`` on any law). The
   architectural costs of Option A (asymmetric semantics on
   ``quadrature=None``, vacuum two-paths-divergence, Liskov
   violation under polymorphic typing) made it unsustainable as
   the long-term contract; the interim was kept only long enough
   to land curvilinear realizer unification (Issue #188 first)
   before the descriptor cleanup could ship. See
   :ref:`bc-trace-law-descriptor-model` for the full retrospective.
8. **Calling ``apply`` on a raw BC descriptor.** Under the
   pre-#186 contract this either worked (with surprising
   semantics — see Option A entry above) or raised
   :class:`BoundaryError`. Under post-#186 it's a **static type
   error** — :class:`BoundaryTraceLaw` has no :meth:`apply`
   method on the class, and neither do :class:`LawSum` /
   :class:`LawScaled`. The correct contract is
   ``SNBoundaryRealizer().realize(law, ms).apply(psi)`` for a
   single BC, or
   ``realize_recursively(tree, ms, SNBoundaryRealizer()).apply(psi)``
   for a descriptor tree. The realizer is the **sole** bridge; the
   §16A.3 three-layer split is enforced by the type system.
9. **In-tree Wave-0 operator algebra over unrealized
   :class:`BoundaryTraceLaw` instances (β1 form).** Considered as
   the rank-N composition mechanism during Issue #186 B3
   exploration. Rejected in favour of the separate-type-family
   approach (β2 / :class:`LawSum` / :class:`LawScaled`). β1
   produced :class:`OperatorSum` trees whose leaves were laws,
   not operators — the type checker could not distinguish a
   not-yet-realized "operator" from a real operator, and calling
   :meth:`apply` on the β1 tree raised at the leaf realization
   step. β2 makes the law-vs-operator distinction inspectable
   statically: :class:`LawSum` has no :meth:`apply` method, full
   stop. See :ref:`bc-rank-n-algebra` for the detailed
   comparison.


References
==========

* Grand Report v3 §16, §16A.1–5 (affine boundary form + trace
  structure), §16A.10 (sparse trace primitives), §16A.11 (dual
  registry), §16A.12 + §27.6 (universal invariants), §26A.4
  (named-error catalog). Source: ``.claude/plans/neutron_transport_grand_report_v3.md``.
* The 12-wave refactor plan:
  ``.claude/plans/transient-giggling-cake.md``.
* The post-Wave-12 cleanup plan (Issue #188 + #176):
  ``.claude/plans/curvilinear-realizer-and-2arg-cleanup.md``.
* Lewis, E. E. & Miller, W. F. (1984). *Computational Methods of
  Neutron Transport*. American Nuclear Society. §3.4 (boundary
  conditions in transport).
* Bell, G. I. & Glasstone, S. (1970). *Nuclear Reactor Theory*.
  Van Nostrand Reinhold. §1.5 (albedo, white, and Marshak
  boundary conditions).
* The tensor decomposition equation :eq:`bc-tensor-decomposition`
  at :ref:`bc-tensor-decompositions` (in
  :doc:`/theory/methods/sn/boundary_conditions`) shows the algebra
  :math:`B = \sum_\alpha G_\alpha \otimes A_\alpha` that this page
  refines into the affine form :eq:`affine-bc-form`.
* :ref:`operator-algebra` for the Wave-0 primitives the realized
  BCs decompose into.
* The V&V error catalog in the ``vv-principles`` skill
  (``docs/theory/verification/error_catalog.rst``) carries
  the ERR-040..ERR-047 entries in canonical form.
