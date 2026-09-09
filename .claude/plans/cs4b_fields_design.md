# CS4b design record — fields are space elements

**Status: ROUND 2 RECORDED** (round 0 = grounding; round 1 = user rulings +
redirections; round 2 = kernel re-engagement + F1/F2/F5 RULED — all
2026-08-21; investigation memos: `scratch/cs4b_field_taxonomy_census.md`,
`scratch/cs4b_kernel_as_frame_stress.md`, both adjudicated + probe-verified).
⚠ Read §"Round 2" LAST-SECTION-FIRST on pickup (§3's own hazard: earlier
fork text carries proposals that round 2 supersedes in place). Charter:
`space_and_kernel_binding_campaign.md` §5 "### CS4b" (+ the CS4a-R amendments
EE-1/XD-10/EE-5 recorded there). Grounding census (every number below with its
predicate + file:line): `scratch/cs4b_grounding_census.md` @ `466e6756` — cite
it, do not re-copy it (plan-authoring §9).

**Goal (domain terms).** A field is a pair (values, space) whose space answers
every structural question the field has — sizing, measure, membership identity.
The mesh is the space's *provenance*, not the field's *attribute*. (XD-10
bound: SIZING and — per the census, MEASURE — derive from the space; ROLE does
not, role is class identity.)

## ⭐ Meta-ruling (user, 2026-08-21, round 1) — effort is NOT a criterion

Verbatim: *"we never decide on something because it is easier. The difficulty
of something is merely a question of how many sessions it will take. … our
objective is to create the best code. Correct, elegant, ergonomic, efficient.
It will take as long as it takes."* Migration size (the 632-vs-22 framing) is
retracted as a justification everywhere in this file; a convenience surface is
justified only on the four criteria, or accepted solely as a **labeled
intermediate state before full migration**. Recorded durably in
[[feedback-build-the-machinery-operator-algebra]].

## Grounding verdict (corrections to the charter, census §9)

- "10 leaves" → **2 declaration roots** (`BulkField._bases.py:170`,
  `FaceField:782`), 10 ABCs, **20 concrete leaves**. Migration is written
  against the 2 roots.
- "≥11 output-mint sites" → **~107 production construction sites** (~65
  operator-arm / ~28 solver / ~14 promotion; §2). The dunder algebra rides
  free (`replace`-based); the factories and `_from_balance` do not.
- "16 test sites" → **15 direct-ctor sites; 632 factory calls in 86 test
  files** if factory signatures change (§5) — the F4 fork.
- The fabrication's ENERGY half already landed (CS4a K2 `_pose_space`);
  `from_materials` survives as XS-data supplier with ONE production consumer
  (`homogeneous/solver.py:229`); the rebind dissolving it is chartered CS4c —
  the F5 fork.
- EE-1's co-vector partially exists (`ReactionRateFunctional` fiberwise,
  `InnerProductFunctional` generic axis-contraction) — extend, don't mint
  (existence-check done, census §9.6).
- `cross_section_field.py:89-91` false `mesh : SNMesh` docstring — **✅ FIXED
  on sight 2026-08-21** (this session, pre-design).

## The forks

### F1 — WHICH space family do bulk fields get?

Census facts: three disjoint bulk-space mint families coexist (§6c: per-leaf
name+shape TAGS / axis-built `MaterialMesh.bulk_space` / `full_field_space`'s
"sn_bulk" Gram interior); the bulk MEASURE (`mesh.quad.weights` reads at
`_bases.py:409`, `angular_source_sink.py:186`) is not derivable from the tag
family; `has_coordinate_cone`: axis-built → True, tags → None (§10); face
families already read CACHED rich spaces off the mesh — the pattern that works.

- **(A) Axis-built per-role cached spaces on the carrier** *(proposed,
  2026-08-21, unratified)*: `ScalarField.space := mesh.bulk_space` (exists,
  cached); `AngularField.space := mesh.angular_bulk_space` (new
  cached_property: `of_axes(angular axis from quad ⊗ EnergyAxis ⊗ spatial
  axis)` [+ LD moment tail]); the moment family's cell-group factor re-points
  at the cached scalar space. Unifies mint family (i)→(ii); (iii)'s interior
  can re-point at the same object (Gram equivalence V·w — bit-vs-principled
  adjudicated by the byte gates at test-architect time). Measure/ng/N become
  axis reads; cone answerable; the `_space_for_mesh` twin mint RETIRES.
- **(B) Re-point at the existing rich spaces without unification**: angular →
  `full_field_space.interior`, scalar → `bulk_space`. Less construction; keeps
  three families; ng stays positional; measure stays fused inside G.

Why (A): it is the campaign's own direction (CS2 = axes on composites — CS4b
lays the field-layer slice of that substrate), and it makes the census's
sharpened XD-10 (measure not derivable today) actually false afterward.

**Round 1 (user) — F1 WIDENED to a review of the field taxonomy itself.**
Verbatim: *"with space formalized, there is the possibility that we don't need
an angular flux field and a scalar flux field as separate classes, or at least
there is some opportunity for simplification. This is especially true if the
angular space has all the information for a retract operation leading to its
collapse."* I.e. the FAMILY axis (Angular/Scalar/Moment/…) may collapse into
space structure (which axes the space has), leaving classes to carry ROLE only.
Member census dispatched (T1–T5). The adjudication tension already visible:

- **For collapse**: the moment family is the in-tree precedent (L lives in the
  SPACE); the dunder algebra is family-uniform; face families' distinctive
  methods already read `self.space`; per-family `_phase_space_shape` becomes a
  space-axes predicate; units may decompose along axes (the /sr is the angular
  factor's density — units compositional along the tensor structure).
- **Against full class-collapse**: Pattern 4 / the coding-standards decision
  lattice — *"axis changes the SHAPE ⇒ class"*: ψ+φ is today a STATIC type
  error; one class with shape-as-data demotes it to runtime-only. The census's
  T3 (static-typing surface) measures what that costs.
- **The likely synthesis to pressure-test**: family classes survive as thin
  TYPED VIEWS (static ψ/φ distinction kept), while the family MACHINERY
  (validation, integration, factories) collapses into space-driven generic
  bodies + the F3 retract/embed operators. To be adjudicated against the
  census, with the user.

**F1-sub (surfaced by `_from_balance`): per-FAMILY spaces, not per-LEAF.**
Today each leaf tags its own `_SPACE_NAME`, so space identity duplicates ROLE
— exactly what XD-10 says the space must not carry. Under per-family cached
spaces, `_from_balance` collapses to `cls(values=lhs.values − rhs.values,
space=lhs.space)` — no factory, no mesh, no navigation; role transition is
purely class transition (Layer 1 already owns it). *(proposed)*

### F2 — identity re-homing (the §8 surface)

Census §7, measured: space-`==` is a DEMOTION at every family (tags blind to
volumes+BCs; axis-hash BC-blind); two `from_mesh` calls today give `space is`
→ False (uncached mint). ⟹ the honest replacement for the `mesh is` gate is
**cached-space `is`** — which REQUIRES F1's move (the mint lives on the
carrier, cached; factories source it, never mint). *(proposed; near-forced)*

**Round 1 sharpening (mine, after the F3 axis reads — for the user's eye):**
on axis-built spaces, `==` is CONTENT equality (axes' structural bytes incl.
measure — `space.py:161-175`), so the "demotion" verdict was measured on the
TAG family and does not transfer. The BC-blindness of bulk-space `==` may be
mathematically CORRECT rather than a defect: by the DOF-set+Gram criterion the
bulk space genuinely does not depend on BCs — BCs enter the TRACE spaces
(which ARE BC-sensitive), so trace-field partner gates carry that
discrimination. ⛔ REFUTED same day (round 2): `[M]` the trace space is
law-blind TOO — `angular_trace` is built from quadrature + face layout only
(augmented_mesh.py:766-774). The doctrine survives STRONGER: no space sees
the law, correctly, because a law changes neither DOFs nor Gram — laws are
operator data. The round-2 F2 ruling carries the final form. ⟹ the candidate doctrine: partner identity = space CONTENT
equality (axis-built `==`), with cached-`is` as the fast path — provenance
(which mesh instance) stops being an arithmetic gate, content does the work.
⚠ This shifts "different problems don't mix" from provenance-identity to
content-identity — a doctrinal call the user must ratify (CS3 ruling 1 kept
fiber discipline as "space/mesh identity"; this says the SPACE half suffices).

Same-step obligations (§6b — the gate's full call-site set):
- `BulkField._check_partner` mesh-`is` → space-`is` (`_bases.py:196`); the
  `FaceField` copy (`:832`); `ScalarSourceSink.__add__`'s private spelling
  (`scalar_source_sink.py:155`).
- The **17 operator/solver-side `field.mesh is not …` gates** (census §7 list)
  re-keyed per site.
- `FullField.__post_init__`'s `getattr(x,"mesh",None)`-tolerant gate
  (`full_field.py:265-274`) — ⚠ becomes a SILENT NO-OP for migrated leaves;
  re-key in the same step. Elegant form: **FullField becomes an element of
  `FullFieldSpace`** — the composite carries the composite space and the slot
  gate reads `space.interior is interior.space` (+ trace) — the composite-level
  spelling of "fields are space elements". Also re-homes the `FullField.mesh`
  property (`:279`).
- Test pins of the gate messages (~10 files, census §7) migrate with it.

### F3 — sibling-space navigation for cross-family derivations

The problem: `AngularFlux.integrate_angular` mints a ScalarFlux **from
self.mesh** (`angular_flux.py:125`); same shape at
`HarmonicMomentFlux.scalar_flux` (:240), `ScalarSourceSink.as_per_ordinate`
(:198), `truncate/extend` (L→L′). Once fields hold no mesh, the method cannot
reach its sibling space. No third way exists: either spaces are navigable or
the caller/owner supplies the codomain.

- **(a)** The carrier mints a navigable space FAMILY (spaces know their
  marginals) — re-introduces the weld one level down; rejected unless the
  funnel ruling later creates a natural family object.
- **(b)** The maps become bound OPERATORS at the carrier (angular integral,
  moment truncation) and field methods retire/delegate — realizes the algebra
  (the standing lens), but pulls operator-binding machinery into a field-layer
  phase; the binding BASE is chartered CS4c (EE-6).
- **(c)** ~~*(proposed bridge)*: the method takes its codomain space as an
  argument~~ ⛔ SUPERSEDED same day by the user's reframe + the measured
  answer below.

**Round 1 (user reframe + `[M]` resolution): retract/embed, owned by the
PRODUCT space.** User: *"this seems like a retract and embed question. I think
(but I'm not sure) that the space has all the information to retract. I'm not
sure about all the information to embed. But even if it doesn't, if we left an
accessor on space to access the original Discrete Measure, then the
information exists."* Measured against the axis layer (2026-08-21):

- **Retract: YES, fully.** An axis-built space stores `axes: tuple[Axis, ...]`
  (`space.py:196`), each axis carrying its FACTOR MEASURE (`axis.py:16,114-121`;
  `None` ≡ counting, canonicalized; per-factor storage, never the outer
  product). Dropping the angular axis: the remaining axes ARE the marginal
  space (`of_axes(*rest)` — name derived injectively from content, so the
  reconstruction is canonical-by-`==`), and the dropped axis's `weights` IS
  the integration kernel. Conditional on F1-(A): TAG spaces have `axes=None`
  and cannot retract — a further argument for (A).
- **Embed: owned by the CODOMAIN product, so the marginal never navigates.**
  The product knows its factors: which axes the operand's space must match,
  and which axis to broadcast along (with `weights` available for the
  isotropic ÷Σw normalization — today's `from_isotropic` math). Callers
  (solvers/operators) hold the richer space via the carrier.
- **The categorical structure is real**: with the isotropic convention,
  `R ∘ E = id` on the marginal (a genuine retraction pair); `E ∘ R` is the
  isotropic projector on the product — the K_iso family (#276). Condensation
  (XD-9's `T·bind(K)·T⁺`) is retract/embed along the ENERGY axis;
  homogenization along the SPATIAL axes — one primitive, three campaign
  consumers ("build primitives, not products").
- **Realization *(proposed)*: as OPERATORS minted by the product space** —
  `space.retraction(axis_label) → LinearOperator` /
  `space.embedding(axis_label) → LinearOperator` — so `integrate_angular`
  becomes (sugar over) `space.retraction("angular") @ psi`, realizing the
  algebra instead of the welded einsum (`_bases.py:409`). The DiscreteMeasure
  accessor (user's fallback) is noted as the escape hatch if a genuinely
  non-axis measure ever needs to ride along — not needed for the shipped
  families.

### F4 — factory survival

~~632 factory calls vs 15 direct-ctor sites ⟹ factories survive as
conveniences *(near-forced)*~~ ⛔ RETRACTED same day — the framing was
effort-based, which the meta-ruling forbids ("can be accepted only as an
intermediate state before full migration", user).

**Re-derived on the four criteria (round 1, proposed):** the factories
DECOMPOSE — they are not one kind of thing.

- **Pure sugar** (`from_mesh`, `zeros_on`, `from_ndarray`): each saves exactly
  one property read (`space=mesh.<role>_space`) over the primary constructor /
  `Field.zeros(space)`; a second construction idiom beside the primary is a
  Pattern-2 seam with no ergonomic payload ⟹ **retire, full migration**; any
  mesh-delegating stage they pass through en route is a LABELED intermediate
  state, never the destination.
- **Math-bearing** (`from_isotropic` = the isotropic EMBED ÷Σw;
  `from_mesh_and_L` = SH(L) space construction; `from_face_arrays` =
  slot-layout assembly): these are retract/embed/space-construction machinery
  wearing factory clothes ⟹ **re-home to the space/operator layer** (F3's
  operators; space builders), then retire the factory spelling.
- `_from_balance` stops needing any factory under F1-sub (per-family spaces):
  `cls(values=lhs.values − rhs.values, space=lhs.space)`.

### F5 — the homogeneous rebind slice (the "and THEN" tension)

> ✅ REMEDIED 2026-09-08 by the CS4c coda — C1 `5caad3d6` / C2 `39e7f32f`. The ruled branch (defer; the SN path is the first kernel
> consumer, the homogeneous path re-points LAST as the degenerate coda)
> was executed at the coda: `HomogeneousProblem` supplies the data (C1)
> and `MaterialMesh.from_materials` — with the three arms and the
> witnesses that served only it — is deleted (C2). `[M]` bit-identical:
> the D5 byte gate 8 of 8, the operator-tier anchor (A and F) 8 of 8
> `array_equal`. O1's tell completes. Record:
> `docs/theory/foundations/infinite_medium.rst`, "Development history".
> The rows below are PRESERVED as the reasoning that produced the
> ruling — read them as history, not as a description of the tree.

Census §9.5: CS4a K1's kernels have ZERO production consumers; the rebind that
dissolves `from_materials`' last consumer is chartered CS4c; the CS4b charter
says "and THEN the fabricated path retires". Either:

- **(pull, proposed)**: a homogeneous-only rebind slice lands in CS4b — the
  O9 ~10 operator-construction sites in `homogeneous/solver.py` re-point at
  the kernels, `from_materials`' last production consumer dissolves, the
  fabricated path retires as chartered, O1's byte-gate tell completes (D5 8/8
  is the wall). Kernels get their first production consumers (closes the
  unconsumed-machinery gap early).
- **(defer)**: fabricated path survives into CS4c; CS4b's done-when must NOT
  claim the O1 tell (charter edit required either way — §3, edit in place).

**Round 1 (user): REDIRECTED — neither branch taken yet.** The kernel design
itself is re-opened first: *"right now, kernel look like a VERY thin class.
But I think there is some inspiration to be taken from Frame. Frame assembles
the frame object and generated the analysis and synthesis operators. It seems
to me like Kernel could be a more robust class that assembles the linear
operators. You might want to stress test that perspective."* Stress test
dispatched (cross-domain-attacker, P1–P8: generation set, layering inversion,
cross-method reuse, the XD-1 analysis-verb home, the C asymmetry, EE-6
compatibility, the thinness diagnosis, foreign frames) →
`scratch/cs4b_kernel_as_frame_stress.md`.

**Stress-test VERDICT (2026-08-21, sharpest claims probe-verified):** the
Frame analogy is off by one layer — `[M]` `FrameBase` is a 2-field frozen
dataclass (basis, measure) implementing ZERO math (`frame.py:113-132`; both
apply faces one-line delegations); the RICH object is `Basis` (6
representation-free verbs, zero runtime imports). `FrameBase(basis, measure)`
IS `bind(kernel, space)` — **the precedent argues FOR the chartered external
binder**, and kernel-generates-operators (design b) is refuted on two
structural grounds (import direction — a kernels↔scattering cycle once CS4c
re-points; binding is BINARY, neither operand owns it — the third object is
EE-6, lifted to `BoundOperator(datum, space)`). The deciding rule: **a data
object's verbs return ARRAYS; only the binder returns OPERATORS.**

**The user's thinness diagnosis SURVIVES, re-aimed**: kernels are thin
against `Basis` (the 6-verb data analogue), not against Frame — so the
surviving design (c) = thin-data kernels ENRICHED with representation-free
array/kernel-returning verbs (`truncated` ships; dagger-where-typed,
`condensed` per XD-9's ruled pair, channel algebra) + the external binder +
C's deliberate 3+1 (its `Id`-frame absence is `IntegralKernelOperator`'s sole
discriminator — uniformising blinds a working gate).

**Probe results (executed this session, `scratchpad/probe_kernel_stress.py`):**
P-1 no adjoint-family member on any kernel — the chartered `bind(K)† =
bind(K†)` gate's operand DOES NOT EXIST; P-10 the fission factor swap is
REFUSED by the χ-simplex guard — K† of a FissionKernel is a DIFFERENT TYPE
(recorded as the XD-1 sharpening in the campaign plan CS4c block); P-2 fused
vs split channel binding differ at 5.6e-17 (FP association — the 0-ULP pin
moves under any split); P-3 all four ℓ=0 spellings agree bit-exact (agreeing
Pattern-2 copies, no live bug); P-7 both kernel docstring equalities hold;
P-4 over-order refused via accidental bare `IndexError` (untyped — the
kernel's derived `order` subsumes the operator's int at the rebind).

**F5 resolution *(proposed, now safe)*:** with (b) refuted, the chartered
external-binder design STANDS, so the homogeneous rebind slice can pull into
CS4b without double-bind risk: the O9 ~10 sites re-point at kernels through
the binder, `from_materials`' last consumer dissolves, O1's tell completes,
D5 8/8 walls it. The kernel VERB enrichment (the re-aimed thinness) lands at
CS4c with the binding base. Open probe for the test-architect: P-6 (is any
SHIPPED rule a non-tight witness, or is the XD-1 gate's negative leg
custom-rule-only — a §6c question).

### F6 — EE-1's integrated reaction-rate co-vector (obligation, not a fork)

Extend against the shipped pair (`ReactionRateFunctional` fiberwise /
`InnerProductFunctional` axis-contraction) — the integrated pairing ⟨Σ,φ⟩_G
with codomain scalar; re-point the three homogeneous rate lines + the
`.. implements:: normalisation`. Existence-check done (census §9.6); the
extend-vs-new adjudication happens at the execution step with both files open.

## Non-fork obligations (carried from charter + census, all same-phase)

1. The bare assert → typed refusal: `sn/mesh/augmented_mesh.py:322` (probed:
   messageless plain / deep `AttributeError` under `-O`); model on
   `diffusion/augmented_mesh.py:211-218` (the honest O8 half).
2. `.areas` wrong-message (`material_mesh.py:517-523` — 3 arms share
   `_areas = None`, message true for 1).
3. The `mesh is None` two-meanings sentinel (`material_mesh.py:207,492`) —
   the un-weld names the states apart.
4. Docs: `infinite_medium.rst` 4-step narrative re-write (HALF-STALE already
   — `_pose_space` undocumented there; sites census §4f), `spaces.rst:1037`;
   `dead_references` baseline **0** at HEAD — exit must hold it.
5. `_from_balance` flips WITH the factories (verified weld,
   `numerics/field.py:356-358`).
6. Charter number corrections edited in place at the rulings commit
   (plan-authoring §3): the 10/≥11/16 rows.
7. EE-5 grep obligation: every data-flow removal swept over `orpheus/` +
   `docs/` + `tests/`, by CONCEPT as well as symbol.

## Protocol tail (per the ratified per-phase protocol)

Rulings on F1–F5 (user) → fold into the campaign plan §5 CS4b → dispatch
**test-architect** (proactive trigger: the carve crosses
numerics/transport/sn/homogeneous; brief = this file + census) → compact →
execute, surgical posture (main agent writes, user steers).

## Round 2 (2026-08-21) — kernel re-engagement + F1/F2/F5 rulings

### The KernelBasis proposal, reconciled with the stress test

The user's counter: *"a KernelBasis either ABC or Protocol that does the
common part of all operators (binding to space, defining domain and codomain),
and a specialization such as ScatteringKernel (brings the frame to obtain
analysis, reconstruct and synthesis operators, forgets the frame, leaves an
accessor to the frame)."* Reconciliation — the two views AGREE once the
common part is named precisely:

- **The common-part abstraction IS the binding base** (EE-6 → `BoundOperator
  (datum, space)`): space admission (the ONE `__post_init__` guard),
  domain/codomain derivation, and the forget-with-accessor storage contract.
  **ABC (dataclass mixin), not Protocol** — it carries shared BEHAVIOR and
  fields, which a Protocol cannot; Protocols stay for capability
  discrimination (`IntegralKernelOperator`).
- **The per-channel binding RECIPE lives on the bound-operator subclass's
  constructor** (`ScatteringOperator.from_kernel(kernel, space)` mints the
  frame from kernel eigenbasis × space measure — ruling 2 — uses it, then
  FORGETS it, accessor kept). Import direction preserved: operators import
  kernels (C8), never the reverse. Kernels stay array-verb data.
- ⭐ **The frame accessor resolves P4**: the bound operator's retained
  `frame` accessor + a declared analysis-verb field is WHERE XD-1's
  "the binding must declare its verb" lives. Mint-and-forget with a
  retained accessor is the tournament's forgetful principle applied to
  binding ("the arrows plus the laws they satisfy").

### Streaming: same BINDING SHAPE, not a kernel — ruled by the shipped Protocol

User's doubt confirmed mathematically and by the tree: `[M]`
`IntegralKernelOperator` (integral_kernel_operator.py:164-183) defines kernel-
hood as NONLOCAL — integrating the carrier against a measure on ≥1 axis —
and explicitly excludes local/diagonal operators. Ω·∇ is local-differential:
NOT a kernel. But L shares the binding SHAPE exactly (space + the scheme's
closure minted at binding, forgotten, accessor left — the tournament's
scheme finding). ⟹ the abstraction that unifies is the BINDING BASE, not
kernel-hood; the datum KINDS stay three: integral kernel (S/N2N/F),
multiplier (C), differential-stencil (L). P5's "3+1" becomes "3+1+1 under
one base"; `IntegralKernelOperator` remains the strict kernel-hood
discriminator. "Kernel" is never spelled onto L.

### Restriction — the third verb, chartered with its precision

User: *"The BC lives in a restricted space of the bulk, right?"* — right at
the MEASURE level, not the DOF level, and the distinction is the design:
- The trace's DOFs are NOT a subset of the bulk's (faces vs cell centres) —
  discrete restriction bulk→trace is not a subselection for cell-centred
  schemes.
- The trace MEASURE is exactly the restricted bulk measure (dV→dA,
  w→|Ω·n̂|w) — which is already how `angular_trace` is built (`[M]`
  augmented_mesh.py:766-774: quadrature + boundary_face_layout, nothing
  else). The formal seat the earlier session derived, now vocabulary.
- Where restriction IS a true subselection: (i) COMPOSITE block projections
  (FullField → interior/boundary), (ii) the half-range Γ± splits on the
  trace (support subselection by sign(Ω·n̂)) — today hand-spelled selectors.
Payoffs chartered: Γ± as algebra; restriction† = extension-by-zero (residual
assembly); R∘G's G gets its formal seat (the OPERATOR rewiring stays on the
boundary thread, #367). The complete forgetful family: **retract**
(integrate out against the axis measure), **embed** (the section),
**restrict** (subselect support). All three minted by the richer space.

### Rulings

- **F1 RULED (user):** the machinery collapse is COMMITTED scope — "can be
  done now or later... it needs to be done at some point. Just a matter of
  when." Sequenced: NOW, in CS4b (it is what CS4b is). The class-merge
  question becomes a NAMED decision point immediately after CS4b's landing
  (recorded so it cannot silently drop), decided on the then-pure
  static-typing + units-decomposition evidence.
- **F2 RATIFIED (user):** space content-equality is the identity mechanism —
  *"the equality becomes very well defined, and since everyone uses space,
  the equality of a lot of things becomes well-defined at once."* The BC
  question resolved by the DOF-set+Gram criterion: a BC LAW changes neither
  the DOF set nor any Gram — `[M]` even the trace space is law-blind by
  construction — so law-blindness of EVERY space is CORRECT; laws are
  operator data (the realized R table). Sync: composite gates compare
  per-block spaces; cross-problem element arithmetic that equality permits
  is well-defined; problem-identity discipline lives at the iteration layer
  (CS3's own relocation).
- **F5 RULED (user), ⛔ REVERSES the round-1 "pull" proposal:** the **SN
  path is the FIRST kernel consumer** — "using the homogeneous path to
  design this is akin to a greedy optimization heuristic... a local
  optimization." No homogeneous pull-in; CS4b's done-when DROPS the O1
  fabricated-path tell (charter edited in place); at CS4c the SN rebind
  drives the binding design (all axes, retract/embed/restrict, all kernels
  live) and the homogeneous path re-points LAST as the degenerate coda —
  where `from_materials` then dissolves. The ruled phase order (CS4b →
  CS1.5′ → CS2 → CS4c) already places the SN binding after every axis
  exists, so the order satisfies the ruling as-is.
  — ✅ REMEDIED 2026-09-08 by the CS4c coda — C1 `5caad3d6` / C2 `39e7f32f`: the coda ran exactly as ruled. C1 gave the problem a HUB
  (`HomogeneousProblem`, ruling R-c1) that supplies its own data, and C2
  deleted `from_materials` with its three now-input-less arms. `[M]`
  bit-identical (D5 8/8; the operator anchor A/F 8/8), so the O1 tell
  the CS4b done-when had DROPPED is now completed by the coda instead.

## Round 3 (2026-08-22) — Q3 ruled; Q5/Q6 explained

- **Q3 RULED (user): NO GATE for the scalar quadrature-blindness permission** —
  "we can't protect a bad user from every bad choice… too hypothetical to gate
  now" unless a concrete in-code site exists. `[M]` checked: none does —
  production solvers are single-mesh by construction (`solve_sn` takes one
  `sn_mesh`); the one cross-method seam (DSA, `dsa.py:638-686`) crosses at the
  raw `.values` level and rebuilds on the SN mesh, so no field-level arithmetic
  ever meets two quadratures. Disposition: the permission gets ONE sentence in
  the partner-gate rationale (articulation of the doctrine's consequence), no
  invariance gate. The verification plan's §4-consequence-2 gate row is
  DROPPED.
- ⭐ Q5 strengthened while checking Q3: `dsa.py:661` hand-spells the section
  (`delta_phi0[None] / self._sum_w` broadcast) — a FOURTH live embed spelling —
  and the same file carries TWO retract spellings (`integrate_angular` :638 +
  a hand `einsum` with `w_mu` :649). The verb consolidation has one more
  call-site cluster than the plan counted.
- Q6 blast radius sharpened: `[M]` `cone_violations` has ZERO production
  callers (5 hits, all prose) — the LD answering→refusing flip is a
  diagnostic/test-surface change; the refusal branch + message ALREADY SHIP
  (CS1 step 4, `field.py:471-479`); CS4b only changes which fields ROUTE into
  it (LD moves from the `None`-legacy arm to the honest `False` arm).

### Round 3b (2026-08-22) — Q5 resolved in structure; Q6's theorem family

**Q5 — the split is ONE MISSING CORNER of a square the frame already ships
3 corners of.** `[M]` `numerics/frame.py` + `projection.py`: the frame emits
**analysis** M = ⟨χ,·⟩_W (primal down, :195/:434), **reconstruction** R = plain
synthesis (primal up, :200/:475), and **project** = G⁻¹M (:310-333 — "the
homogenise / condense verb", the DUAL down, with `gram` the row-sum probe
:254). The **dual reconstruction R·G⁻¹ — the dual UP — does not exist as a
face.** At ℓ=0: G₀ = ⟨1,1⟩_W = Σw, so the section E = broadcast/Σw IS the
missing corner's moment-0 instance, and `R∘E = id` is biorthogonality
(M·R·G⁻¹ = G·G⁻¹ = id). `from_isotropic`, `as_per_ordinate`, `dsa.py:661` are
its hand-rolled shadows. ⟹ **F3's verbs re-specified in frame vocabulary**:
`retraction(axis)` = the analysis face of the constant-basis frame on that
axis; `embedding(axis)` = the DUAL reconstruction (composable from shipped
parts: `reconstruction ∘ gram.apply_inverse_metric` — the /Σw is G₀⁻¹, a
DERIVED quantity, never hand-spelled again); the frame itself gains its
fourth named face (`dual_reconstruction`) so the square closes at every L,
not only ℓ=0. The adjoint identity "R.H = the synthesis" is the
GalerkinFrame's own strengthened promise (Π* = R, projection.py). The G6.x
gate family re-specifies accordingly: the `R.H == Σw·E` row becomes the
biorthogonality/gram gate (M∘dual_reconstruction = id; gram probe = Σw at
ℓ=0). And the user's "reconstruct vs synthesis" — `[M]` a binding-level
distinction, not mathematical: `reconstruction.apply` delegates to the naked
`Basis` verb bound to the frame's table; the genuine second up-map is the
dual one, which is exactly the corner the tree lacks.

**Q6 — yes, a theorem family, and the transport literature has a name for
the modal cone: the REALIZABLE SET.**
(a) The shipped LD basis is tensor degree-1 (multilinear per cell) ⟹ the
**vertex theorem**: a multilinear function attains its extrema at the 2^d
cell vertices ⟹ p ≥ 0 on the cell ⟺ p(v) ≥ 0 at every vertex ⟺ 2^d LINEAR
inequalities on the modal coefficients (1-D: c₀ ≥ |c₁|). Exact, finite,
polyhedral — the cone is the pullback of the vertex-nodal positive orthant
under the invertible vertex-evaluation map. The shipped refusal message
("evaluate the field in a nodal realization first") is therefore EXACT for
LD, not an approximation. (b) Higher-degree 1-D: Markov–Lukács / SOS — the
cone is SPECTRAHEDRAL (a PSD condition on a Hankel-type coefficient matrix);
decidable exactly, never a sign test. (c) Multi-d higher degree: nonneg ⊋
SOS (Motzkin); deciding positivity is NP-hard in general; Bernstein
coefficients give a one-sided certificate. (d) The angular twin: moment
REALIZABILITY (Levermore closure hierarchy; Eddington |f⃗₁| ≤ f₀ — the same
ice-cream-cone shape as LD's c₀ ≥ |c₁|: both are the degree-1 truncated
moment problem). Disposition *(proposed, consistent with the Q3 posture)*:
the refusal stands (`has_coordinate_cone` is precisely named — the cone
exists but is not coordinate-wise in the modal basis); the vertex test is
the NAMED future capability with its natural first consumer (a
positivity-preserving limiter, Zhang–Shu family); the theorem family is
recorded in the corpus with the refusal.

### Round 3c (2026-08-22) — Q6 disposed; the eight-ruling status table; DESIGN COMPLETE

- **Q6 disposed (user)**: the refusal + routing flip stand ("nice that
  something is already inside the code, but this requires an extension");
  the exact-modal-cone extension is **#400** (vertex test / realizability —
  user-directed filing: the corpus record alone is not a robust recovery
  surface for an improvement). The two-sided gate (DD answers / LD refuses)
  + the `field.py:463-475` and `scheme.py:530` docstring re-words ride CS4b
  S6/S7 as planned.

**Status of the verification plan's eight open rulings at design close**
(epistemic markers per plan-authoring §2 — RULED = the user's; RECOMMENDED =
mine, presented and unvetoed, operative unless redirected at execution):

| Q | status |
|---|---|
| Q1 `MomentResidual` | RECOMMENDED: record-as-choice at `_from_balance` (the flip dissolves the 2-arg blocker; `L` recoverable from the space); MINT only with a consumer |
| Q2 re-point `full_field_space.interior` | RECOMMENDED YES (architect concurs): `[M]` scalars bit-identical, vector ≤1 ULP on DD AND LD — cheapest it will ever be; DriftWarning wall runs at S2 end under the vv re-baseline criteria |
| Q3 quadrature-blindness gate | ⛔ RULED (user, round 3): NO gate — `[M]` no concrete production site; one docstring sentence, articulation not protection |
| Q4 trajectory-gate ownership → CS4b | FORCED consequence of the round-1 F1 ruling (machinery collapse now); the §3 edit-in-place is authorized by that chain; re-derivation licence = the ρ≈c Adams–Larsen anchor, never old-vs-new |
| Q5 verb naming split | ⛔ SUPERSEDED by round 3b: **complete the frame square** — the frame gains `dual_reconstruction`; verbs specified in frame vocabulary; the /Σw is G₀⁻¹, DERIVED. The plan's G6.8 Σw-swap battery arm re-specifies as the biorthogonality/gram gate |
| Q6 LD cone refusal | RATIFIED (user, via the #400 filing); extension tracked at **#400** |
| Q7 `_StubMesh` seam | DEFERRED to execution with the user steering (surgical posture); default proposal: `MaterialXSField` takes the space NARROWLY (the honest move, pulled forward minimally) |
| Q8 battery amendment | ADOPT (mechanical): `[M]` +175 rows / +2.90 s; without it the battery misses its own headline regression |

**⏹ DESIGN COMPLETE.** Execution not started; it opens by writing the L17
convention crosswalk from this record + the verification plan, then S1.
