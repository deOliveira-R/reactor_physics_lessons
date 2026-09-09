# Cross-Domain Attacker — Lessons

Behavioral/process lessons only: "what detection mistake did I make, or what
recurring insight sharpened my frame-attacks?" The frame-trigger CATALOG lives
in the `cross-domain-frames` skill (Part A/B/C) — do NOT duplicate it here.
Durable frame-matches that became real architecture are DESIGN POINTERS in
`MEMORY.md`, not lessons.

The cross-cutting meta-pattern behind most of these: **a frame-attack's value
is not "I named an exotic frame" — it is "I produced a concrete reformulation
with a fail-able first test, OR I crisply refuted the frame with a reason that
survives into the trigger table as a non-entry."** Speculation that names a
frame without a payoff degrades the table's signal. Every lesson is one face of
that standard.

---

## L-001 -- Refuted frames are first-class output; record the REASON, not just the rejection

→ The DIRECTIVE is now in AGENT.md (Required Output Shape — "A refuted frame is
first-class output"). It is identity-level (every attack writes an UNEXPLORED
block). What stays HERE is the forensic catalog: the specific high-prior frames
that keep getting correctly refuted on transport work, with their structural
reasons — recalled when a fresh attack is tempted by one of them.

The recurring high-prior frames that keep getting (correctly) refuted on
transport work, with their reasons:

- **Wiener-Hopf factorization** — wrong solver FAMILY. It is native to the
  Chandrasekhar/H-function half-space line, structurally incompatible with a
  bouncing-Peierls or sweep formulation. Keeping the two families structurally
  independent is itself a V&V requirement (independent references).
- **Homology / chain complex** — tempting via the word "boundary," but
  `∂∘∂ ≠ 0` in transport (two reflections compose to a non-trivial map; the
  boundary trace + its extension are a dagger adjoint PAIR, not a differential).
  No `∂²=0` ⇒ no homology payoff.
- **Category theory / operad / PROP** — almost always LOW-SIGNAL: the concrete
  win it gestures at (role-parameterization, compositional structure) is already
  captured by a nameable concrete frame (biproduct, affine torsor, forgetful
  functor with explicit laws). Name the concrete frame; list category theory
  UNEXPLORED unless a specific functor/law produces a test.
- **Tensor networks / MPO** — fires ONLY on a genuine bond-dimension trigger
  (a rank-N chain where N is a real truncation knob). A rank-1 or rank-2 fixed
  structure (a biproduct, a 2-surface BIE) is bond-dimension-1/2 DEGENERATE —
  not a network. Do not promote it to MPO until N≥3 actually ships.
- **Differential geometry / Christoffel** — needs a CURVATURE term to
  redistribute. Straight Euclidean chord segments or a Cartesian cell have none.
  It fires for the curvilinear streaming redistribution `(1−µ²)/r ∂_µ`, NOT for
  geometry-of-the-domain questions.

How to apply: when a high-prior exotic frame does not fire, write the one-line
STRUCTURAL reason (wrong family / no `∂²=0` / degenerate rank / no concrete law)
into UNEXPLORED. A bare "category theory — no trigger" is weaker than "category
theory — role-parameterization win already captured concretely by affine+Krylov;
no abstract-nonsense lever needed."

⭐ **SHARPENING (2026-08-27, the α cross-method attack): a refutation must record
the QUESTION it was refuted FOR, because the same frame can be decisive on a
different question.** See L-021 — the symplectic frame was correctly refuted for
a DESIGN question ("does it buy a better scheme?" — no) and is the load-bearing
answer to a CLASSIFICATION question ("which families share this object?"). A
banner reading "⛔ symplectic — no independent lever" would have suppressed the
best available answer. Write "refuted FOR <question>; the fact it establishes is
<fact>" — the *fact* survives even when the *lever* does not.

---

## L-002 -- A first test that cannot fail is rejected output — make it DISCRIMINATE

→ The DIRECTIVE is now in AGENT.md (Required Output Shape — "A first test that
cannot fail is rejected output"). It is identity-level (every frame candidate and
pollination emits a first test). What stays HERE is the forensic detail: the
specific discriminator constructions that have worked, recalled when building a
first test for a new claim.

A real first test discriminates the reformulation from the status quo by being
able to RED. The discriminator constructions that have worked:

- Frame the test as a property the NATIVE frame predicts and a wrong/naive
  implementation VIOLATES. Multiplier-algebra: `M_f @ M_g == M_{f·g}`
  bit-identical (a "wrapper that just stores an array" fails). Transpose-of-a-
  sweep: build the dense `L`, take `L.T`, assert the reverse-walk recurrence
  reproduces it (a spatial-only reverse fails on the nested angular block).
- For a refactor claimed bit-identical, the discriminator is `array_equal`
  (0 ULP), NOT `allclose` — only bit-identity distinguishes a genuine
  single-source from a value-correct-by-coincidence twin.
- For a typing claim, the discriminator is a NEGATIVE test: `np.ndarray`
  satisfies `Vector` but NOT `TransportState`; `tff_flux + tff_moment` RAISES.
  A test where everything passes proves the bound is theatrics.

How to apply: before emitting a first test, ask "what implementation would this
test PASS that I am claiming is wrong?" If the answer is "none," the test cannot
fail — rewrite it to target the specific divergence (the dropped term, the wrong
metric, the un-transposed nested recurrence).

---

## L-003 -- Smell #16 (distinct paths/reps to one operator) is the dominant transport tell — fire all four shapes

The single most-recurring native-frame-not-found signal in this project's
SN/transport work. The four-shape CATALOG is NOT re-copied here: it lives in the
skill (reference.md Part C, Smell #16) and in the AGENT.md promoted kernel, both
preloaded on every dispatch. The LESSON is the detection discipline, not the
catalog.

How to apply: every shape resolves to the SAME elegance move — collapse the
distinct paths onto ONE primary object (faces / a named trace-or-multiplication
operator / a displacement type / a shared primitive), which turns a correctness
COINCIDENCE into a theorem and usually deletes a marshalling shim. When you spot
the smell, name WHICH shape — the fix differs per shape.

---

## L-004 -- Property-vs-TYPE is decidable, not a taste call: demand a coexisting dual + an APPLIED morphism

A recurring design question is "should X earn its own field/state TYPE (vs a
PROPERTY/parameter on an existing type)?" The durable criterion that resolves it
without an unbounded taste argument:

> A representation earns a distinct TYPE iff there exist **≥2 bases that are NOT
> canonically isomorphic** (the iso depends on a quadrature/node choice),
> connected by a **change-of-basis operator that is itself MODELED and APPLIED**
> — it carries truncation error, has an adjoint, and participates in the
> operator algebra. All three clauses must hold.

Worked: angular order PASSES (ordinate `AngularFlux` ↔ harmonic moment field,
bridged by the applied projection/reconstruction Vandermonde-like pair) → two
types, correct. Spatial order FAILS clause 1 (one tensor-Legendre basis; the
only morphism is identity) → a PROPERTY, correct. If the modeled morphism would
be `id`, the type's sole payoff (forbid-mixing-a-dual) has no referent →
type-theatrics. Decidable by grep: count the within-X representations and the
applied non-identity morphisms between them.

Corollary — defer-with-an-EXPLICIT-trigger: when no method supplies the dual
TODAY but one plausibly arrives (here: nodal-DG / nodal-diffusion would supply a
nodal↔modal morphism), record the precise condition that flips the verdict.
"No current consumer" is not "never" — name the latent consumer and the trigger.

How to apply: when asked property-vs-type, do not argue aesthetics — count the
coexisting non-iso reps and the applied morphisms. Zero applied non-id morphisms
⇒ PROPERTY. Pairs with the project's unify-after-two-instances rule.

Corollary — an axis that changes the ARITHMETIC INTERFACE cannot be a phantom
type PARAMETER; it MUST be a distinct CLASS. A `Generic[Tag]` parameter is erased
at runtime and does NOT specialize dunders — so two instantiations
(`Field[Rep,Flux]` vs `Field[Rep,Source]`) share ONE `__add__` body. If the two
"values" of the axis have DIFFERENT `__add__` signatures (a torsor `A×V→A` that
forbids `A×A`, vs a vector `V×V→V`), no shared body satisfies both — the only
encoding is a distinct class per value (a mixin). This is a HARD refutation, not
a taste call: it killed the phantom `Field[Rep,Role]` carrier outright (the
(Rep×Role) grid attack). The decision lattice for a 2-axis carrier grid:
axis-changes-arithmetic ⇒ class; axis-changes-SHAPE ⇒ class; only an axis that
changes NEITHER (a true index/tag) can be a phantom param. When BOTH axes change
arithmetic-or-shape, the unique elegant form is the orthogonal-factor MULTIPLE
INHERITANCE `Leaf(RoleMixin, RepBase)` — and that parametrization, if wanted on a
type at all, belongs on the OPERATOR contract `[Din,Cout]` (where the axis values
are leaf TYPES, the genericity is APPLIED, and role-preservation is a fibration
theorem), NOT duplicated onto the carrier. The NEGATIVE discriminator: a
phantom-param impl that "passes" only by branching on a stored `role` field at
runtime is the stringly-typed anti-pattern — `replace(f, role=Other)` type-checks
and bypasses the gate; that bypass test REDs it.

---

## L-005 -- Read the WORKTREE, distrust Nexus on a feature branch

Frame-attacks on active branches repeatedly grounded on STALE facts because
Nexus answers from the main checkout's graph and the live code is on the branch.
Every design memo that grounded its frames on Nexus `context`/`query` while the
work lived on a worktree risked citing a superseded file:line.

How to apply: on a feature branch, ground frame-attacks on file:line read DIRECTLY
from the worktree (Read/Grep on the absolute branch path), and say so in the memo
("branch-verified, NOT Nexus — stale"). Reserve Nexus structural queries for
main-checkout questions or after `use_workspace(<worktree root>)` against a graph
built inside the worktree. A frame whose trigger is a code fact is only as good as
the freshness of that fact.

---

## L-006 -- "Frame-leak naming" — a model-agnostic interface named after ONE consumer is a latent lie

Detection-adjacent vocabulary insight that recurs on shared interfaces: a slot on
a model-AGNOSTIC layer named after one consumer's physics (`total_xs` on an
advection–reaction closure a diffusion solver will also consume;
`is_scan_march_compatible` — a SCHEME property named after a sweep STRATEGY)
becomes a lie the moment the second consumer passes a different realization
through the same slot. TELL: a docstring that says "generic in X" while the
parameter is named after a specific X₁. FIX: name the ROLE in the INTERSECTION of
all consumers' domains (`reaction_xs`; `transverse_coupling_is_facewise`), not the
realization. Distinct from Smell #16 (that is two reps of one quantity; this is
one slot whose NAME over-commits to one of N consumers).

How to apply: when assessing a name on a multi-consumer interface, ask "what does
the SECOND consumer call this?" The decisive first test is a second consumer that
reads the property with NO first consumer in scope — if it can't, the name is
strategy-entangled. SKILL-PROMOTION STATUS (re-checked 2026-06-22): HELD for a
THIRD sighting. Current count = TWO independent (`total_xs` on an
advection–reaction closure a diffusion solver also consumes; `is_scan_march_compatible`
= a SCHEME property named after the ScanMarch STRATEGY, the #240 D5 trait). The
project floor already covers the GENERIC vice via `coding-elegance` ("frame-leak
parameter naming"); a Part C SMELL earns its slot only when the cross-domain
detection angle (a 2nd-consumer-with-no-1st-in-scope first test) has a third
sighting distinct from the two naming cases. Until then, fire it inline, do not
promote.

---

## L-007 -- The transport-resolvent backbone predicts cross-method layering AND its exceptions — reach for it first

The spine itself is the AGENT.md kernel ("Cross-method backbone: the transport
resolvent") — preloaded, so it is NOT restated here. The LESSON is how to DEPLOY it:

- When a "find-the-special-value" family (k, α, time-step, fixed-source) looks like
  distinct solvers, check whether they are POSINGS of one generalized eigenproblem
  `Aψ=λMψ` sharing the resolvent backbone. If so, the ONLY genuinely per-method
  layer is the loss-operator REALIZATION; role-assignment, the µ→physical map,
  adjoint, and transient-shift are method-agnostic data over a shared engine.
- Generality flows toward the OPAQUE resolvent interface, never away from it. When
  two iterative drivers share a loop body, the one exposing its resolvent BEHIND a
  Protocol is the general engine; the one built from a concrete `(L,S,F)`
  factorization is the specialization that adapts INTO it. A "retire the opaque
  loop, migrate everyone to the concrete loop" plan points the deprecation arrow
  the wrong way (a CP/diffusion matrix has no `(L−S)⁻¹` to factor out).

How to apply: open any transport solve/adjoint/eigenvalue/layering question with
the resolvent backbone — it predicts both the layering split and the diffusion
exception from ONE principle, and it tells you which layer is the shared engine
before you read a line of the specific drivers.

Corollary — the backbone tells you WHERE a foreign frame fires. A cross-domain
frame keyed to an operator's ALGEBRAIC SHAPE fires only on the members whose
shape matches. Worked (DSA ↔ mixed-FEM/CFD saddle-point,
[[dsa-saddle-point-mixed-fem-frames]]): the **saddle-point / inf-sup / mixed-FEM**
frame fires on the **diffusion / low-order member ONLY** — because the mixed
`[[A,Bᵀ],[B,C]]` structure IS the elliptic exception (diffusion is
self-adjoint ⇒ mixed-when-first-ordered), while SN/MoC/CP are
characteristic-triangular SWEEPS with no saddle to stabilize (the primary
transport operator is either `L⁻¹` triangular or the Peierls `I−PL⁻¹Σs`
compact-perturbation-of-identity — neither a saddle point). So "consistent DSA"
= "the low-order is the **Schur complement of a compatible pairing**," and the
whole mixed-FEM/CFD apparatus (inf-sup, Darcy-vs-Stokes, Rhie–Chow, block
preconditioners) attaches to the acceleration subproblem, never to the sweep.
Before pointing a foreign frame at "transport," ask the backbone WHICH member
has the matching shape — pointing inf-sup theory at the sweep is a category
error the backbone catches for free.

---

## L-008 -- A "fully probes" claim is about operator LINEARITY, not input polynomial degree

A recurring MMS/verification reformulation trap: assuming a richer (higher-degree)
input is needed to "more fully" exercise an operator. For a LINEAR operator, an
input that is merely NON-CONSTANT in the operator's active variable already probes
the FULL map; a higher-degree input only changes WHICH point in the already-fully-
probed range you land on. Worked: the curvilinear angular redistribution
`(1−µ²)/r ∂_µ` is linear in ψ, so a linear-in-µ ansatz `(A+µB)/W` (the native
truncated-P1 Legendre element) fully activates it — no P2 needed. Enrich the
ansatz degree ONLY to satisfy a QUADRATURE-exactness requirement (e.g. `Σ wₙµₙ²`),
never to "more fully" probe a linear closure.

Paired HAZARD (the larger correctness risk when lifting a Cartesian reference to
curvilinear): a redistribution/curvature term carries a `1/r`, so a curvilinear
slope driver MUST vanish at the origin (`B(0)=0`) for pole-regularity; the slab has
no such constraint, so a slab-derived ansatz silently drops BOTH the term itself
AND its regularity constraint. The geometry MEASURE also enters the L2 error norm,
not just the source — an unweighted norm mis-measures the convergence order.

How to apply: before enriching an MMS ansatz, check the operator's linearity. If
linear, a non-constant input suffices — spend the degree budget on quadrature
exactness, and on a curvilinear geometry check the `1/r` pole-regularity of every
redistribution term and the measure-weighting of the error norm.

---

## L-009 -- A change-of-basis frame's OWNER and its Galerkin-vs-PG discipline are predicted by the operator's SYMMETRY (commutant membership / Funk–Hecke), not by which subsystem calls it first

A recurring architectural question on this project: when a method projects to
coefficients, acts there, reconstructs (`R∘A∘M`), WHO owns the frame `(M,R)` and is
it Galerkin or Petrov-Galerkin? The durable detection kernel — distinct from the
resolvent backbone (L-007, which is about solve/iteration LAYERING; this is about
projection-frame OWNERSHIP and DISCIPLINE):

> A frame `(M,R)` is OWNED by the operator `A` whose EIGENBASIS it is, and it is
> GALERKIN iff that eigenbasis is ORTHOGONAL — both decided by `A`'s symmetry. If
> `A` commutes with a group action (is in the commutant), Schur's lemma forces it
> block-diagonal-per-irrep in the isotypic basis, that basis IS `M`'s codomain, and
> a SELF-ADJOINT `A` (real kernel) diagonalizes ORTHOGONALLY ⟹ M*=R up to the
> Plancherel metric ⟹ Galerkin. No symmetry ⟹ no eigenbasis ⟹ the frame is a
> SOLUTION-WEIGHTED projection (test≠trial) ⟹ Petrov-Galerkin, owned by no operator.

Worked (the angular SH frame): Σ_s(Ω·Ω') is SO(3)-zonal ⟹ Funk–Hecke diagonalizes it
in {Y_ℓ^m} with eigenvalues = the Legendre moments (= the diagonal of the in-code
`Λ`); so M is LITERALLY the change-of-basis into scattering's eigenspace, the frame
is scattering-OWNED, and it is a `GalerkinFrame` BECAUSE Σ_s is self-adjoint-zonal
(orthogonal eigenbasis). Streaming `Ω·∇` is the ℓ=1 tensor operator (Clebsch–Gordan
⟹ ℓ↔ℓ±1 PN recurrence), does NOT diagonalize, so it does NOT own the basis. The
DISANALOGY that confirms the rule: energy condensation's G×G group-transfer matrix
has NO symmetry / no Funk–Hecke ⟹ its frame is a flux-weighted `PetrovGalerkinFrame`,
owned by no operator. ONE principle (operator symmetry) thus explains an entire
campaign's Galerkin-vs-PG split that prior memos had ASSERTED axis-by-axis.

Two corollaries that drop out:
- **Falsifiability of "subsystem X owns the frame":** the claim is structurally
  CONFIRMED (not non-falsifiable) when X's operator is the one whose eigenbasis the
  frame is. The genuine falsifier is a SECOND consumer whose TRUNCATION ORDER is set
  independently of X's operator (an output detector-functional of order L_d, or a
  flux expansion L_flux ≠ X's order) — that consumer makes the frame a general
  L²-tool with ≥2 independent consumers, flipping ownership. "Any function is
  X-basis-expandable" is NOT such a falsifier: the INFINITE expansion is basis-
  agnostic, but the TRUNCATED frame the code actually has is dimensioned by X's
  spectrum support (the operator's moments vanish above its order).
- **Placement:** the eigenbasis-owner is the canonical CONSTRUCTOR + the L-binding,
  NOT a private field — the generic frame machinery (analysis/reconstruct/conjugate)
  stays in the neutral layer (shared with the no-symmetry PG consumers); only the
  CONSTRUCTOR `owner.frame = neutral_factory(owner_order)` records ownership, and it
  relocates to the neutral factory the instant a second independent-L consumer lands.

How to apply: for any `R∘A∘M` ownership/discipline question, ask "what symmetry does
A have?" before reading call sites. Rotationally-invariant/zonal/convolution kernel
⟹ Funk–Hecke/Schur eigenbasis ⟹ A owns a GALERKIN frame. No symmetry ⟹ solution-
weighted PETROV-GALERKIN, owned by none. The first test that discriminates: assert
the owned frame is `GalerkinFrame` (M*=R up to the Plancherel/Gram metric) while a
no-symmetry sibling is a genuine `PetrovGalerkinFrame` (M*≠R, test=solution·trial).

Corollary — the SYMMETRY-SUB-BLOCK + multigrid-coarse-operator face (DSA #2, the ℓ=0
frame; [[dsa-rp-angular-frame]]): the Galerkin verdict descends to an IRREP SUB-BLOCK.
An acceleration/coarsening projecting onto ONE symmetry sub-block (DSA → the ℓ=0 / V₀
trivial-SO(3)-irrep constant on S²) inherits Galerkin from the parent symmetry-owned
frame — it is `angular_frame(0)`, NOT a new `ConstantBasis`, and the R/P pair is that
sub-frame's two faces (Π=P∘M W-self-adjoint under the PLAIN measure; a solution weight
would be the ONLY thing making it PG, and DSA has none). The multigrid connection this
adds: a "consistent low-order operator" IS the **Galerkin coarse operator** `R A_high P`
of the sub-block frame, post-composed with a **Schur complement** of the
retained-but-closed moments (Fick = odd-block Schur; Marshak = incoming-partial-current
Schur — the SAME move interior vs boundary), and "consistent" means that triple product
is taken on the DISCRETE (assembled) high-order operator (reduce-discrete ≠
discretize-reduced). One symmetry principle now predicts frame OWNERSHIP, the
Galerkin-vs-PG DISCIPLINE, AND the multigrid CONSISTENCY condition.

Corollary — the THIRD outcome: **ALL owners ⟹ NO owner, and it is still GALERKIN**
(#326 symmetry quotient, [[quadrature-symmetry-quotient-frames]]). The rule as stated
has two outcomes (one operator's eigenbasis ⟹ owned+Galerkin; no symmetry ⟹ owned by
none+PG). A **symmetry quotient** is the third: the group sits in the commutant of
**every** equivariant operator at once (streaming, collision, scattering, fission AND
the BCs are all `C_{2v}`-equivariant for a 1-D cylinder), so by Schur they are all
simultaneously block-diagonal in the isotypic decomposition. A frame owned by
everything is owned by nothing — it belongs to the **PROBLEM's symmetry**, not to an
operator. Crucially this does NOT make it Petrov-Galerkin: PG in this project means a
**solution weighting**, and a symmetry fold carries none (its Gram stays diagonal at
exactly the parent value, because invariant functions are constant on orbits and the
orbit weights sum to the parent's). So the verdict is **Galerkin on a SMALLER SPACE**
— second sighting of that exact verdict after the DSA ℓ=0 sub-block. Detection rule:
before typing an `R∘A∘M` as PG, ask *"is the test≠trial gap a SOLUTION weight, or a
GROUP identification?"* A group identification is Galerkin-on-a-sub-block; typing it
PG repeats the category error B3.0 fixed when it moved the Lambertian out of the
geometry slot.

SKILL-PROMOTION STATUS: a STRONG candidate for skill Part C (a new smell:
"eigenbasis-blind frame placement" / "operational-pipeline vocabulary for a spectral
decomposition") — the `harmonic_moment_flux.py:6` "natural data carrier of the
Galerkin pipeline" is the tell that the native Funk–Hecke frame is unnamed. The DSA
ℓ=0 case is a second CONSUMER of the SAME angular frame (not the independent
non-angular sighting the bar wants), but it strengthens the rule with the sub-block +
multigrid face above; the #326 symmetry-quotient case adds the third outcome and a
SECOND instance of the Galerkin-on-a-sub-block verdict. Still held for a genuinely
non-angular eigenbasis frame before promotion; until then fire it inline.

---

## L-010 -- A conserved-quantity COLLAPSE splits by WHAT is conserved (rate vs probability/mass), which fixes the MORPHISM (average vs marginalize) — NOT by a weight

When two coarsening/reduction operations look like "the same projection with vs
without a weight" (a 1-frame-vs-2-frame asymmetry, a "bare sum vs weighted
average" asymmetry), DO NOT accept the weight framing. Ask first: **what
functional does each collapse preserve?** A reaction RATE `⟨T·w, Σ⟩` is preserved
by an AVERAGE = `G⁻¹·M` (the projection `frame.project`, normalize=True). A
PROBABILITY or MASS (`Σχ=1`; a particle count) is preserved by a MARGINALIZE =
`M` alone (the un-normalized analysis `frame.analysis` / a bare `@T` against a
partition-of-unity table, normalize=False). These are DIFFERENT MORPHISMS that
differ by the `G⁻¹` factor — a weight=1 `project` would divide by the bin COUNT
and BREAK `Σχ=1`, so the "weight=1 degenerate of project" framing is provably
wrong. The honest unification is ONE machinery `(test_weight, normalize?)`:
`average = analysis ∘ G⁻¹` vs `marginalize = analysis`. Exposing both DISSOLVES
the frame-count asymmetry — it was never about how many frames, it was about
whether each channel's collapse axis carries a conserved RATE or a conserved
MASS.

Two corollaries that recurred in the same attack (XS coarsening: spatial
homogenize ∥ energy condense):
- **A "same slot ± weight" comparison can be hiding an AXIS category error.** χ
  in spatial homogenization collapses the SPATIAL axis (average); χ in energy
  condensation collapses the BIRTH-ENERGY axis (marginalize). Comparing them as
  one slot conflates two operations on orthogonal axes. Before unifying two
  collapses, confirm they act on the SAME axis; if not, the "asymmetry" is just
  two different reductions wearing the same channel name.
- **A precondition spelled as a 30-line docstring caveat on a 3-line method body
  wants to be a TYPE.** `FrameBase.gram` hardcodes a row-sum probe valid only for
  disjoint (diagonal Gram) or partition-of-unity (`R·1=1`) bases, then documents
  the third (tapered/dense) case it silently gets wrong. The Gram structure is a
  property of the BASIS (declare DIAGONAL / POU / DENSE), and `project` should
  dispatch on the declaration and RAISE on the unhandled case — the same
  "no-consumer ⟹ raise, don't silently delegate a half-consistent op" discipline
  a test-only basis already uses for its unbuilt synthesis side. A silent wrong
  number is the landmine; the declared-type + negative-test (a DENSE-declaring
  stub makes `.project` RAISE) closes it.

How to apply: at any "reduce/coarsen/collapse a container, preserving X"
question, name X (rate? probability? mass? current?) for EACH channel before
reaching for a frame. Rate → average (`G⁻¹M`); probability/mass → marginalize
(`M`); a second functional (surface current, leakage — GET/Smith) → a second
test space. The morphism follows from the conserved functional, and the
discriminating first test is the order-non-commutativity of a multi-axis channel
(`project(Σ@T) ≠ (project Σ)@T` because the normalization is keyed on one axis).
SKILL-PROMOTION STATUS: strong Part C candidate ("collapse-morphism-blind:
treating a marginalization as a weight=1 average"). Held for a SECOND sighting
(a non-XS conserved-collapse — e.g. a probability/measure reduction in MC tally
binning or a flux-to-current marginalization) before promotion; fire inline
until then.

---

## L-011 -- A "coupled / nested block system" proposal is a FREE RE-ASSOCIATION of an existing biproduct, not a new object — and "N instances justify the machinery" fails unless the N share a coupling KIND

Two reusable detection moves fired together on the augmented-SN "coupled 2×2
[[A_AA,A_AB],[A_BA,A_BB]], system=field+BC" adjudication
([[coupled-system-field-bc-frames]]):

- **Mat∘Mat≅Mat: a coupled 2×2-of-subsystems over a direct-sum carrier that
  ALREADY carries a biproduct block algebra is that biproduct RE-PARTITIONED,
  not a new categorical object.** The biproduct `⊕` is coherently associative, so
  grouping a flat N-block composite into 2 subsystems (`Mat₂(Mat₂(𝒞))≅Mat₄(𝒞)`)
  is free; the off-diagonals were always there (the seed + the −B boundary block).
  The G-adjoint composes block-wise for free when G is block-diagonal per subsystem
  (`A†` reads `G⁻¹AᵀG` at ANY partition granularity). TELL: a proposal says a new
  type "sits above" the existing block algebra — analogy-language for a theorem
  (same shape as issue_208's "natural 2×2 / adjoint for free"). DO NOT mint the
  `CoupledOperator`/nested type: it is a VIEW (redundant) or a twin (Smell #16). The
  discriminating first test is a CHALLENGE with a definite structural answer: "exhibit
  a LINEAR coupled system expressible nested but NOT flat" — impossible; every
  candidate is flat-re-expressible (⇒ view) or nonlinear (⇒ not a LinearOperator, a
  DIFFERENT abstraction). Before accepting a "coupled/nested/N-way" type, check whether
  the base algebra's block INDEX is merely FROZEN (here `BlockRole` = a 3-value enum
  while `_join_block_roles` already treats a role as a set-of-touched-blocks) — if so,
  the minimal object is "lift the freeze to N-way," not "a new layer above."

- **Defer-until-≥2 counts KINDS of the structure, not INSTANCES of the word.** A
  build-now case citing N coupled-system instances (ψ½ / DSA / multiphysics) collapses
  the moment you classify their coupling STRUCTURE: ψ½ = linear/triangular/metric-adjoint
  off-diagonals; DSA = linear/two-way-iterative/R⊣P-Galerkin off-diagonals; multiphysics
  = NONLINEAR/fixed-point (not a Mat(𝒞) block matrix at all). Three different kinds ⇒ no
  two pair up ⇒ the general machinery has no second instance to generalize FROM, and the
  nonlinear one UNDER-reaches a linear block abstraction (drop it from the count). The
  over-reach dual: a metric/triangularity/PSD assumption baked from the FIRST kind
  (ψ½ triangular biproduct, PSD block-diag metric) EXCLUDES the others (DSA two-way; RQI
  KKT-indefinite-zero-corner). Each kind gets its own home + trigger (ψ½=biproduct-exists;
  DSA=coupled-iterative-defer, the R⊣P shape DEFINES it when it lands; RQI=saddle-point-defer).

How to apply: at any "unify these coupled/composite subsystems under one new type"
request, (1) ask if the carrier already has a biproduct — if yes, the coupling is a
re-association, name the off-diagonals + lift the block-index freeze, do NOT add a layer;
(2) tabulate each cited instance's (off-diagonal structure, metric definiteness, solve
kind, linear?) — build only where ≥2 rows MATCH, defer the rest with the row that will
define them. Pairs with L-004 (property-vs-type by applied morphisms) and L-007 (the
resolvent backbone predicts which layer is shared).

---

## L-012 -- A NAMING task is a frame-detection task: a family word can be FORBIDDEN by a theorem (⇒ species + genus-ABC), and a name's first test is the invariant the name PROMISES run against the object that VIOLATES it

Naming requests ("find the faithful name for these N sibling objects") are
trigger-table work, not taste work. Three durable moves, from the reaction-term
attack ([[reaction-term-naming-species-split]]):

- **Check the refinement invariant BEFORE looking for a family word.** A uniform
  word is *forbidden* when a theorem splits the siblings. Reactions: locality
  *within the fiber* splits 1 multiplier (collision: continuum object is
  `Σ_t δ`, a DISTRIBUTION) from 3 kernels (Fredholm functions) — so "kernel" as
  the family word is false, not merely imprecise. The honest output is
  **species words on the leaves + a genus word on the ABC**, and the genus stays
  greppable by ONE token precisely because the leaves are not uniform. Bonus
  elegance check: make the layer-1 species BIJECT with the already-landed
  layer-3 species (multiplier↔`MultiplicationOperator`, kernel↔
  `IntegralKernelOperator`) — then the seam is *species-preserving*, a real
  property that is strictly weaker than the functor the seam does NOT have.
- **The discriminating test for a NAME is the invariant the name promises, run
  against the object that violates it.** A name is a claim; test the claim.
  Worked: `ReactionTerm`'s genus invariant is decomposability ⟹ assert
  `A(m⊙x) == m⊙A(x)` for a **cell-varying** mask (a constant mask cannot fail),
  where all four reactions PASS and **streaming FAILS**. Species: the multiplier
  additionally commutes with a **group-wise** mask, the kernels must FAIL that —
  and the `Σ_s` used must be genuinely off-diagonal, because a diagonal-only
  `Σ_s` passes the multiplier gate and a design that picks the species from the
  data's *accidental* diagonality is exactly the bug. Corollary: a name that
  cannot carry its own invariant (an invented weakest-true word) OWES a test
  that does.
- **A word already spent elsewhere in the codebase gets a delete-it-and-ask-
  what-breaks check, not an analogy argument.** `Law` (from `BoundaryTraceLaw`)
  for reactions: delete the BC law ⟹ **ill-posed** (it is a CLOSURE, enforced);
  delete scattering ⟹ a *different, still well-posed* problem (it is a
  GENERATOR TERM, applied). Two independent confirmations followed for free —
  the law is *affine* (carries `q`) while reaction terms are purely linear, and
  the domain has a live false friend (ENDF File 7 "thermal scattering law"
  `S(α,β)`). When reusing a precedent, split it: the **realizer** half
  transferred (Kalman realization is layer-agnostic), the **descriptor word**
  did not.

**New smell (candidate, 1st sighting): "the name states a contract the content
violates."** `MaterialXSField` is named as an apply-free datum (Dixmier field)
and carries NINE `apply_*` verbs consumed by two operator modules — the
fiber-operator ACTION living inside the datum. Distinct from Smell #16 shape 1
(two paths to one operator): here there is ONE path, hosted on the wrong LAYER.
TELL: a class whose docstring says "data/descriptor/field" while its method list
says `apply`/`add_`. FIX is relocation (it is a shared primitive, not a twin) —
and the *name* is usually the correct one, so resist the rename reflex.

**Two moves added 2026-08-21 (the `SNMesh` residual tournament,
[[container-ownership-dof-criterion]]) — both produce DECISIVE kills where a
taste argument produces a preference:**

- **The sharpest kill is a word already SPENT IN THIS REPO on an ORTHOGONAL
  AXIS of the same object — and it is one grep.** L-012's existing clause tests
  a borrowed word by deleting the OTHER user and asking what breaks; this is the
  cheaper, harder case where the word is not borrowed but COLLIDES. `[M]`
  `SNPose`/`SNPosing` for a discretization container: *posing* already means
  "the arrangement of leaves into `(A_loss, M)` + the eigenvalue role"
  (`numerics/eigenvalue.py:20-26`, `iteration.py:24-29`,
  `homogeneous/solver.py:225`, `coupled_system.py:118`, and "zero-inflow posing"
  at `loss_representation/assembly.py:401`) — a DIFFERENT axis of the same
  solve, about to get worse when Campaign 2 lands `GeneralizedEigenPencil`.
  `SNRealization` dies twice over: the word is spent on `realize_boundary_law` /
  `SNBoundaryRealizer` / `realize_recursively`, AND the **direction is
  inverted** (realization binds an abstract law TO a space; the candidate object
  exists BEFORE any space and is realization's INPUT). ⟹ before ranking any
  name, grep its stem across `orpheus/` and read what it already MEANS; a
  same-object-different-axis hit is a kill, not a cost.
- **For a CONTAINER, prefer the ROLE name over the CONTENTS name — and the
  incumbent is usually the evidence.** A contents name is falsified by every
  content move a live campaign has already chartered; a role name is falsified
  only if the role changes. `[M]` `SNMesh` is a contents name that is already
  false (it holds a quadrature, a cell closure, a realized-BC table, a projector
  and a sweep schedule), and its module is `augmented_mesh.py` — a contents name
  patched with an adjective, which is the tell. The winning candidate named the
  invariant ROLE (`SNDiscretization` = "the choice of how the continuous problem
  is made finite, refusing inadmissible combinations"), survived all four
  chartered content moves, and degenerated correctly across the method family
  (`DiffusionDiscretization` collapses to mesh+BCs — and its smallness becomes
  informative rather than an unexplained asymmetry).
- **A settings-bag name is a DIAGNOSTIC, not a candidate.** If `XOptions` /
  `XConfiguration` / `XSetup` reads *right* to a reader, that is evidence the
  object carries no invariant and should not be a class at all. Put the bag
  names in the tournament table explicitly, labelled as the diagnostic — they
  are how the no-class arm announces itself.

How to apply: at any "what should this be called" dispatch, (1) hunt the
refinement theorem first — a forbidden family word is the highest-value finding
and it must be said PLAINLY; (2) report the math-faithful name AND the domain
(NE) name, with the routing rule *types get the faithful name, accessors/docs/
equations get the domain name* — a type name is read once per design decision, an
accessor on every line; (3) emit a per-name buys/costs line, and for any name
that would promise an unsupported operation, say which operation and cite the
measurement that refutes it (`Functor` ⟹ `reduce-discrete ≠ discretize-reduce`,
measured). Do NOT invent a name where none exists — say "no faithful name
exists", give the least-bad invented one, and flag it as invented. (4) ⭐ For a
container, settle the ONTOLOGY first and let it pick the name — a name argued
before the contents are assigned is a preference; a name argued after is a
consequence.

---

## L-013 -- Before accepting "the machinery lacks X", check whether a PREDICATE already computes X and throws it away — a `bool` return is the commonest way a primitive stays missing

Recurring shape on "what is missing in the machinery" dispatches. A proposal says a
capability is absent; the truth is that a **verification predicate already computes
the exact object** and destroys it at the `return` statement, because the return type
was chosen as `bool`. The capability is not missing — its **witness** is.

Worked (#326, [[quadrature-symmetry-quotient-frames]]): `SubgroupOfO3.is_invariant`
was offered as "the checking face; the quotient is what's missing". But
`_orbit_closure` (`symmetry.py:904-954`) computes the matched partner index `j` for
every `(node i, group element M)` — i.e. **the permutation representation, which IS
the quotient's only hard input** — and returns `bool`. Two OTHER modules then
re-implement the same permutation independently (`_compute_sphere_reflection_partners`;
MoC's `_reflected_azi_index`). So the honest finding is not "add a quotient class"
(the proposal's scale) but "**return the witness**", after which the quotient is one
further verb (`consolidate`). That collapses the estimated work by an order of
magnitude and is a strictly better answer for the requester.

TELL (grep-able): a function named `is_*` / `check_*` / `verify_*` / `*_closure`
whose BODY builds an index map, a permutation, a matching, a partition, a
factorisation, or a certificate — and whose signature says `-> bool`. Same family as
Smell #16 shape 1 (the re-implementations downstream are the confirmation), but the
CAUSE is different and so is the fix: shape 1 says "collapse two paths"; this says
"one path exists and is throwing away its output — widen the return type first, and
the twin paths delete themselves."

**Sub-shape (2nd sighting, #336): the predicate is COMPLETE, CORRECT and TESTED —
and wired to the ADVISORY path while the CONSTRUCTIVE path bypasses it.** Here the
`bool` throws nothing away; the gap is purely the call site. `AngularSymmetry.
admits_domain` answers "may this rule serve this geometry?" correctly, has a
committed test asserting the exact #336 case (`not slab.admits_domain(lebedev)`),
and its only production caller is `select_quadrature` — the *recommender*. The
object's own CONSTRUCTOR (`SNMesh`) never asks, so the ill-posedness is discovered
several layers down as an out-of-range float. TELL: a guard living in a
`select_*` / `recommend_*` / `default_*` helper, with the type it guards
constructible directly. CHECK: grep the predicate's callers and sort them into
*advisory* vs *constructive*; zero constructive callers means the gate is a
suggestion. The fix is a call site, not machinery — and saying so collapses a
"build the reduction machinery" proposal by an order of magnitude, exactly as the
witness-returning fix did.

Discriminating first test: assert the consumer's hand-rolled artefact is
`array_equal` (0 ULP, L-002) to the predicate's now-returned witness. A
re-implementation with a different tolerance or tie-break diverges exactly on the
degenerate elements (self-partners / fixed points), which is where every bug in that
family lives.

How to apply: at any "map what is missing in the machinery" brief, before enumerating
new types, grep the predicates in the neighbourhood and read their BODIES. Ask "does
this `bool` know something?" A found witness reframes the whole deliverable — and a
proposal that over-scoped the gap is a finding worth reporting plainly, because it is
good news.

---

## L-014 -- An UNSATISFIABLE predicate is a wrong-ARGUMENT diagnosis, not a wrong-predicate one: check each argument's KIND before redesigning the relation

Recurring shape on "derive the correct formulation of predicate P" dispatches. A
gate is found to reject everything (or accept everything) once a broken checker is
fixed, and the brief offers a menu of alternative RELATIONS. Nearly always the
relation is right and one of its ARGUMENTS is of the wrong KIND — a
cardinality/topology mismatch that no amount of re-shaping the relation repairs.

The diagnostic: **compare the CARDINALITY or TOPOLOGY the predicate demands against
what the object can carry.** A finite object cannot satisfy a containment against a
continuous group; a band-limited claim cannot constrain a non-band-limited unknown;
a static table cannot carry a configuration-dependent generator.

Worked (#326/Q2, [[quadrature-symmetry-quotient-frames]]): `G_geom ⊆ G_rule` was
found unsatisfiable because `GEOMETRY_GROUPS` supplies `SO(2)`. Two one-line
theorems settle it without touching the relation: **(A)** `Sym(Q)` of a finite node
set is FINITE (an orthogonal map fixing a spanning set is `id` ⟹ `Sym(Q) ↪ S_N`),
so a continuous `G` is unsatisfiable *by any discretisation*; **(D)** the
correctly-derived requirement — the DISCRETE residual `Γ = G/G⁰` acting on the
fiber — is ALWAYS finite (discrete subgroup of a compact group). The predicate was
never wrong; it was being handed the half of the symmetry group that the
dimensional reduction had already CONSUMED.

Two generalisable corollaries, both cheap to check:

- **A symmetry group that reduces a problem's dimension is SPENT; it cannot also be
  a requirement on the reduced problem.** Its continuous isotropy becomes the
  angular/fiber QUOTIENT (which domain the rule lives on), its free part becomes
  the CONNECTION/redistribution term, and only the DISCRETE residual is still owed
  as a constraint. Three parts, three fates, none discarded — one decomposition
  predicts "why does the slab use a 1-D rule and the cylinder a 2-D one" AND "why
  does the cylinder have an α term and the slab not" from the same split.
- **A "vacuous" candidate framing usually has a non-vacuous sibling one theorem
  away.** "Exactness space is `G`-invariant for every rule ⟹ the test says nothing"
  is true; the sibling is `E = Q∘(Id − Π_V)` (the error functional IS the
  quadrature on the aliased-out part), from which `G ⊆ Sym(Q)` ⟹ `E` annihilates
  every NON-trivial isotypic component at EVERY degree (average `E[f]` over `G`).
  Before discarding a framing as vacuous, apply the group-average to its error
  functional.

Discriminating first test for this family: a case where the candidates DIVERGE on a
parameter the current predicate cannot see (odd vs even `n_φ`), and where the
derived answer REPRODUCES an independently-established in-tree guard (ERR-042) as a
consequence. **A derived predicate that re-derives an existing hand-written guard is
strongly confirmed; one that contradicts it owes an explanation.**

How to apply: at any "the gate rejects/accepts everything, pick a new formulation"
brief, tabulate `(argument, kind it has, kind the relation needs)` FIRST. If a kind
mismatches, fix the argument and stop — the menu of alternative relations is a
distraction. Then hunt for an existing hand-written guard the corrected predicate
should reproduce; that reproduction is the cheapest available confirmation.

---

## L-015 -- Before proposing a foreign basis, check whether the SHIPPED NODES already ARE its collocation points; and a VANISHING FLUX FUNCTION means there is no boundary condition to supply

Two moves from the Q68 cylinder-angular attack
([[cylinder-angular-march-jacobi-ladder]]). Both turned a "here is an exotic
frame with a cost" proposal into "here is a RECOGNITION with a free transform",
which is a strictly better deliverable.

**(a) Identify the node set before designing the basis.** A discretisation
built from a trigonometric / roots-of-unity / equispaced construction is very
often a CLASSICAL GAUSSIAN RULE in a transformed variable — and if it is, the
matching orthogonal family comes with a free transform, a diagonal Gram, a
known exactness degree, and a truncation that is EXACT at the nodes (the
(n+1)-th orthogonal polynomial vanishes at its own n Gauss nodes). Worked:
`folded_product`'s STAGGERED-plus-σ_y arc nodes are `ω_k = (k+½)π/M`, i.e.
`cos ω_k` = the roots of `T_M` — **exactly Chebyshev–Gauss**, with equal weights
= the Chebyshev–Gauss weights. The sphere's GL nodes are the roots of `P_N`. So
"should we go spectral in angle?" was never a proposal: the code was already
sitting on the optimal collocation points and interpolating between them
piecewise-linearly. TELL: a node set generated by `cos` of an equispaced grid,
or by roots of unity, described anywhere as "equispaced" / "uniform" / "the
trapezoid rule". CHECK: apply the geometric map and compare against the Gauss
nodes of `jacobi(α,β)`; ORPHEUS ships the whole zoo in
`numerics/generating_measure.py`, so the check is a 3-line probe.

**(b) A first-order operator whose FLUX FUNCTION vanishes at a domain endpoint
admits NO boundary condition there.** `∫∂_x(f ψ)φ = [fψφ] − ∫fψφ'`, and
`f(endpoint) = 0` kills the boundary term for EVERY ψ, φ. Fichera function zero
⟹ the endpoint is characteristic: neither inflow nor outflow. So any seed /
starting value / admission gate at that endpoint is a **discretisation
artifact**, not physics — and what the endpoint really supplies is a
COMPATIBILITY CONDITION (set `f=0` in the balance and read off the reduced
equation). Worked: the curvilinear SN angular endpoints — `(1−μ²)` on the
sphere, `ξ = sinθ sinω` per cylinder level. This is the promoted form of the
"metric-invisible-yet-active DOF" candidate smell (2nd sighting; the 1st,
[[psi-half-seed-angular-trace-frames]], had the symptom — zero quadrature
weight — without the cause). Corollary that pays immediately: **count the
discrete system.** Here `2M+1` unknowns vs `2M` equations = one condition short,
while the continuous problem needs none — and the code independently computes
TWO endpoint data and discards one. *One short and two long* is a far sharper
finding than "the seed is inaccurate", and the discarded datum is a free
a-posteriori estimator.

How to apply: on any "is this the right discretisation of this operator"
dispatch, run BOTH checks before reaching for a frame — (a) map the nodes,
(b) evaluate the flux function at the domain endpoints. Each is a one-line
computation, and each can convert the whole attack from proposal to recognition.
SKILL-PROMOTION STATUS: (b) is a Part C smell candidate at 2 sightings, both
curvilinear-SN. Held for a NON-transport sighting (a degenerate-drift
Fokker–Planck / Sturm–Liouville / population-balance endpoint elsewhere) before
promotion; fire inline until then.

---

## L-016 -- A recurrence with UNIT end-to-end gain is REVERSIBLE, so "march the stable direction" (Miller/Gautschi) buys nothing — check `G(M)` before proposing it

A high-prior trap that a measured interior amplification actively baits, caught
in-reasoning on the Q68 attack and recorded because the next attacker will meet
the same bait.

The setup that fires it: a linear recurrence with a measured transient
amplification (`A(M) = max_m Π_{k≤m}|g_k| = 2.41…9.44` for `M = 2…32`) whose
step gains `|g_k|` are monotone through 1. The reflex is Miller's algorithm /
Gautschi's minimal-solution principle: *march in the direction the unwanted
homogeneous solution DECAYS*. **It does not apply when the END-TO-END gain is
exactly 1.** `[D]` forward seed error → `G(m) = Π_{k≤m}|g_k|`; backward seed
error → `G(M)/(G(M)/G(m)) = G(m)` — the IDENTICAL profile, because `G(M)=1`.
Unit end-to-end gain means the recurrence is reversible: there is no
dominant/minimal solution PAIR, so there is no stable direction to pick, and
reversing the march is a pure no-op for conditioning.

Here `G(M)=1` was not luck — it is forced by an exact anti-symmetry of the
scheme (`τ(π−ω) = 1−τ(ω)`, itself the discrete shadow of the operator's
`R A R = −A`). ⟹ the generalisation: **a symmetry that makes a scheme
neutrally stable end-to-end also makes it immune to direction-reversal
remedies.** The energy that WAS in the symmetry goes somewhere better — into an
EQUIVARIANCE gate (`R A_h R + A_h == 0`), which is a genuine discriminator where
the scalar shadow of it (`Π(1−τ)/τ = 1`, which the tree already gates) is blind:
`[M-cite]` all four τ variants including "shuffled" pass the scalar and only the
operator identity can separate them (and only at `M ≥ 4` — at `M = 2` reversal
preserves the sum).

How to apply: before proposing any march-direction / minimal-solution /
backward-recurrence remedy, compute the END-TO-END gain. `= 1` ⟹ refuted, say so
with this reason. Then look for the symmetry that forced it and ask whether the
tree gates the SCALAR shadow of an OPERATOR identity — that gap is usually where
the missing discriminator lives.

---

## L-017 -- Before designing a REDUCTION, ask what the OPTIMAL object on the TARGET space is — an optimality theorem there bounds every possible reduction and can refute the whole design

The strongest refutation available against a "derive the induced object" design,
and it is a THEOREM rather than a measurement. A reduction/collapse/marginalization
lands its output in a target space that usually has its own well-developed
optimality theory. Look it up FIRST: if the target admits a known optimum at the
output's size, then **every** reduction is bounded by it, and the design's entire
image is "the optimum" (in the aligned case) or "strictly worse than the optimum"
(otherwise) — so the reduction can never produce something the user could not have
asked for directly and better.

Worked (#336, [[sphere-mu-line-reduction-frames]]): reducing a 3-D `S²` cubature to
the 1-D spherical arm's μ-line lands on `[-1,1]` with a **Lebesgue** reference
(Archimedes' hat-box theorem: the pushforward of uniform-`S²` under `Ω↦Ω·r̂` is
`2π·Leb`, no Jacobian). There the maximal-degree theorem is absolute — `n` nodes ⟹
degree ≤ `2n−1`, attained UNIQUELY by Gauss–Legendre. `[D]` reduce(LS S4) = 4
nodes, degree **5**; `gauss_legendre(4)` = 4 nodes, degree **7**. Equality holds
exactly when the parent is a tensor `product` rule whose μ-factor already IS GL, in
which case the reduce is the tensor projection and returns `gauss_legendre(n)`
verbatim. Two cases, no third, and the design collapses to a DIAGNOSTIC (compute
the reduction to write a refusal message that names the induced rule) plus an
ORACLE, never a value path.

Two corollaries worth carrying:
- **State the metric the domination is measured in, and bound it.** "Dominated in
  the degree metric the tree's own `ExactnessClaim` records; a rule tuned for a
  non-polynomial class is outside that claim." Without the bound the theorem
  over-reaches.
- **Optimality also tells you the ALIGNMENT condition, which is where the real bug
  lives.** Equality-iff-Gauss forces the question "when is the reduced rule Gauss?"
  — answer: iff the parent factors with the μ-line as a factor **about the axis you
  reduce along**. That surfaced the actual hazard: the shipped 3-D rules carry
  their symmetry axis on node column 2 while the 1-D arms read column 0, and the
  arm's radial cosine is *defined* as column 0. Reducing along the arm's axis
  instead of the rule's turns a 4-node GL into an 8-or-9-node non-standard rule
  (`[D]`, offset-dependent). The free repair is a THEOREM too: rotating a cubature
  is exactness-neutral because the reference is `SO(3)`-invariant, so align the
  rule's own maximal rotation axis with `r̂` first — do not push forward along a
  direction the rule has no symmetry about.

How to apply: at any "derive/induce/auto-convert X into Y" dispatch, name Y's space
and its reference measure, then ask whether an optimality theorem exists there
(Gauss for polynomial exactness on an interval; Chebyshev for uniform approximation;
Kolmogorov n-widths; Cramér–Rao for estimators). If one does, the refutation or
confirmation is one paragraph and needs no code. Pairs with L-013: a design refuted
this way usually leaves a much smaller true deliverable (a gate, a message, an
oracle), and reporting THAT is the better answer.

---

## L-018 -- A chartered gate of the form `PROPERTY ⟺ HYPOTHESIS` is the highest-value thing to re-derive: the recurring defect is the RIGHT hypothesis on the WRONG law, and the shipped witness satisfies BOTH so it lands designed-green

A design that carries an equivalence as its acceptance gate reads as maximally
rigorous — an `⟺` looks self-checking. It is not, and it fails in a specific,
repeatable way: someone identifies a real hypothesis (tightness, positivity,
exactness, invertibility), correctly senses it is load-bearing SOMEWHERE, and
attaches it to the law that happens to be in view. Then:

- the **⟸** half is usually TRUE but VACUOUS (the property holds unconditionally
  under the shipped convention, so the hypothesis buys nothing), and
- the **⟹** half is FALSE (a counterexample satisfies the property without the
  hypothesis), and
- **the only witness the campaign ships satisfies both**, so the gate cannot red
  either way — plan-authoring §6c with an equivalence in place of a call site.

The procedure, and it needs no code: **derive the necessary-and-sufficient
condition for the stated property yourself, then check the two directions
SEPARATELY, then ask what the named hypothesis IS n&s for.** It is almost always
n&s for a *different, adjacent* law, and naming that law is the deliverable —
because that law is usually the one certifying the design's central claim.

Worked (CS4a-R, [[cs4a-kernel-binding-representation-frames]]): the charter's
`bind(K)† = bind(K†) ⟺ the frame is TIGHT`. `[T]` the adjoint law is n&s for
`M† ∝ R` (GALERKIN) — free under `FrameBase.analysis`; tightness (`MR = I`) is
n&s for **multiplicativity** `bind(K₁K₂) = bind(K₁)bind(K₂)`, which is exactly
the law that makes "the kernel is representation-free data" honest (a
representation must preserve products). And at the ℓ=0 witness the campaign
ships, `MR = Σw/W = 1` for EVERY rule, so both laws hold for both a tight and a
deliberately non-tight quadrature. Three findings from one re-derivation: the
gate cannot red, the negative control will pass, and the missing gate is the one
certifying the campaign's headline claim.

Two sharpenings worth carrying:
- **An equivalence's truth-value can DEPEND ON A CONVENTION the charter never
  states.** Here it flips on whether the binding's analysis is `analysis` (`M =
  R†`, Galerkin ⟹ vacuous) or `project` (`M = G⁻¹R†`, canonical dual ⟹ the ⟺ is a
  theorem). Both verbs ship. ⟹ before adjudicating an `⟺`, ask *which spelling of
  each face does the binding use?* — an equivalence over an undeclared convention
  is not yet a claim.
- **A normalisation constant that the design guards ("Σw = 4π owned once")
  usually CANCELS in the law it is guarding.** `[T]` `E = (1/W)R†` ⟹ `E† =
  (1/W)R` ⟹ the two 1/W's cancel in the adjoint identity. The constant is a
  VALUE contract; do not let it be read as a law hypothesis, or the gate inherits
  a hypothesis that cannot fail.

**Sub-shape (2nd sighting, CS4b 2026-08-21) — the gate's OPERAND does not
exist.** The shape above is a wrong HYPOTHESIS on a right property. This one
needs no hypothesis at all: the law's right-hand side **names a morphism the
datum does not have**, so the gate cannot be written, let alone red. `[M-by-read]`
CS4c charters `bind(K)† = bind(K†)` while none of `ScatteringKernel` /
`N2NKernel` / `FissionKernel` carries any of `{T, transpose, adjoint, dagger}` —
their whole surface is `{__post_init__, from_mixture, ng, order, p0, truncated,
emission_matrix, dyad}`. The plan records the gate as re-specified and
sharpened (XD-1) without anyone noticing its operand is unspellable.

TELL, and it is one grep: **take every symbol on a chartered law's RHS and check
it against the datum's method list.** Cheaper than re-deriving the law, and it
runs before you have understood the math. The payoff is usually good news —
the missing morphisms are one-liners (here: transpose the moments; swap the
fission factors, which the datum's own docstring already calls a theorem), and
naming them makes the gate spellable. ⭐ And the missing morphism often carries
a design fork nobody has met: the fission swap **violates the χ-simplex
invariant its own `__post_init__` enforces**, i.e. "the adjoint of an emission
kernel is not an emission kernel" — a typing decision the charter never
reached, surfaced for free by asking whether the operand exists.

How to apply: any brief containing `⟺`, "iff", or "holds exactly when" about an
operator construction ⟹ re-derive both directions before reading the
justification, and ask what the shipped witness can distinguish. **Then check
that both sides are even spellable.** Pairs with L-002 (a first test that cannot
fail is rejected output) — this is its equivalence-shaped face, where the
un-failability hides inside a true-but-vacuous half, or inside an operand that
is not there.

---

## L-019 -- A God object is usually a FIXED POINT of a weak identity relation, not a discipline failure — find the identity-scarce owner before arguing about contents

**New elegance smell (1st sighting, Part C candidate): "identity-scarcity
accretion."** A derived object (a cache, a resolved table, a projector) lands on
a CONTAINER rather than on its natural owner **because the natural owner's
`==`/`is` cannot separate the inputs the derived object depends on**. The
container then wins every future placement decision by default, and the growth is
a fixed point rather than a series of lapses.

TELL, and it is often written down by the code itself: a docstring that ARGUES
for the container as owner by disqualifying the alternatives on identity grounds.
`[M]` 2026-08-21, `sn/mesh/augmented_mesh.py:986-994` — the loss-kernel gauge
sits on the mesh "rather than `AngularTraceSpace` (which is geometry-blind …) or
`FullFieldSpace` (whose `__eq__` is `(name, shape)`, so two meshes with different
BCs and the same DOF count compare equal — a cache keyed there would be keyed on
a size)". Both disqualifications are true; both are statements about identity,
not about ownership.

Why this changes the attack: a contents-based argument ("this class holds too
much") prescribes relocation, and relocation **will not hold** while the identity
stays weak — the next cache comes back to the container for the same correct
reason. The load-bearing prescription is the ORDER: strengthen the owner's
identity FIRST (here, the campaign's own `of_axes` derived-name flip, which
digests "label, shape, kind, measure bytes, subclass identity" —
`numerics/space.py:248-256`), THEN relocate. Reporting the sequencing is worth
more than reporting the smell.

Discriminating first test: build two containers differing ONLY in a datum the
derived object depends on but the candidate owner's identity ignores (two meshes
differing only in a boundary declaration), and assert the candidate owners
compare UNEQUAL. It REDs today, and its RED is the same sentence the docstring
already wrote in prose — which is the confirmation that the smell, not the
placement, is the finding.

How to apply: on any God-object / "should this hold everything" dispatch, before
tabulating contents, ask **"which object in this stack has usable identity?"** If
the answer is "only the container", say so — that is the mechanism, and every
contents-level recommendation is downstream of fixing it. Pairs with L-013 (the
proposal over-scopes the gap; the true deliverable is smaller and elsewhere).

---

## L-020 -- An analogy to an in-repo precedent can be ADJECTIVE-accurate and LAYER-wrong; map onto BOTH layers and keep the mapping that preserves ARITY

`plan-authoring` §1's PRECEDENT clause says a precedent is cited from memory of
its shape, so read it and check each adjective. This is the failure the adjective
check does **not** catch: every adjective can be roughly right while the analogy
lands on the wrong OBJECT in the precedent's stack — and a layer error inverts
the conclusion instead of degrading it.

The tell is a **two-layer precedent** (data + binder, model + view, kernel +
driver) cited by ONE of its layer names. Ask which layer holds the DATA and which
holds the PAIRING, map the new problem's objects onto **both**, and keep the
mapping that preserves ARITY — field count and verb count. A "thin data class"
maps onto the precedent's data class, never onto its binder.

Worked (CS4b, [[kernel-as-frame-layer-inversion]]): *"Kernel is a VERY thin
class; take inspiration from Frame — Frame assembles the operators."* `[M-by-read]`
`FrameBase` is a frozen dataclass with **2 fields** that implements **zero math**
(both faces' `apply` are one-line delegations); the rich class is **`Basis`**, 6
representation-free verbs. So the thin datum's analogue is `Basis`, and
`FrameBase(basis, measure)` **IS** the external binder the proposal was offered as
an alternative to. Net: the precedent, read at the right layer, **argues for the
design the proposal opposed**, while the proposal's *diagnosis* (thin) survives
re-aimed at the data class. Reporting both halves is the deliverable — a bare
"the analogy is wrong" would have thrown away a correct observation.

⭐ **The discriminator that falls out, and it is grep-checkable:**

> **In a data/binder split, the DATA object's verbs return ARRAYS; only the
> BINDER returns OPERATORS.**

That single rule decides the layering question outright, because returning arrays
is what keeps the data module's imports empty (`basis/base.py`: zero runtime
imports beyond stdlib + numpy; its two domain types are `TYPE_CHECKING`-only),
and empty imports are what make it reusable from any layer. `datum.bind(space) ->
LinearOperator` inverts exactly that property, and no dispatch mechanism
(registry, double dispatch, `singledispatchmethod`) avoids it — the
method-agnostic module must NAME the method's types however the dispatch is
spelled. Use the import direction as the refutation; an `isinstance` chain is
only its symptom.

Two corollaries worth carrying:
- **Binding is a BINARY operation, so neither operand owns it** — the precedent's
  answer is a THIRD OBJECT, and the third object is where the CACHING lives,
  which neither operand can host *because neither knows the other*. When a
  campaign has already chartered a "binding base", check its arity against the
  precedent's: one field short usually means an abstract hook standing in for the
  operand that was left out.
- **A 3-of-4 uniformity gap is INFORMATIVE, not a smell, when the fourth member
  is the DEGENERATE case of the same construction** (here: the collision
  multiplier's frame is `Id`, because a diagonal operator's eigenbasis is the
  nodal basis — L-009). Criterion, not taste: *different construction ⟹ smell;
  degenerate case ⟹ unifying DELETES content*. And check whether the asymmetry is
  already load-bearing as a TYPE before proposing to erase it — here
  `IntegralKernelOperator`'s sole discriminator is the member the fourth object
  lacks, so uniformity would blind a working gate (§6c, inflicted deliberately).

How to apply: when a brief hands you an analogy to an in-repo object, read the
object AND its collaborators before reasoning, and write the two-layer table
(role / object / field count / verb count) as the first section of the reply. If
the mapping inverts the brief's conclusion, say so plainly and then say which
half of the brief's diagnosis survives — that half is usually the real
deliverable.

---

## L-021 -- On a CROSS-METHOD "who consumes X" brief, find the GENERAL-GEOMETRY (or general-case) derivation FIRST: one equation decides every row, and a refutation must carry the QUESTION it was refuted for

Two moves from the α-dome cross-method attack
([[alpha-dome-chart-vs-measure-cross-method]]). Both converted a per-method
survey — eight separate arguments, each arguable — into consequences of one
citable equation.

**(a) Hunt the general-case derivation before answering ANY row.** A brief that
asks "which of these N families needs X?" invites N independent arguments, and
N independent arguments is N chances to be plausibly wrong. The literature
almost always contains a derivation of X for the GENERAL case, and reading it
collapses the whole table. Worked: **Pomraning 1989, NSE 101:330, "The Transport
Equation in General Geometry"** derives the angular-derivative coefficient for an
ARBITRARY chart — his Eq. (68) expresses `dμ/ds` purely in **principal radii of
curvature + metric coefficients**, i.e. the second fundamental form. That single
equation (i) settled chart-vs-measure outright (no quadrature appears on the RHS
at all), and (ii) **DERIVED three rulings the project had asserted**: slab
neutrality (`ρ=∞`), the sphere's azimuthal decoupling (**umbilic**: `ρ_u=ρ_v` ⟹
`cos²φ+sin²φ=1` ⟹ φ cancels), and the cylinder's exact `μ_z` conservation (the
axial principal curvature is **zero**, so the only channel that could move `μ_z`
is identically absent). TELL that you need this move: the brief lists ≥4 families
and you are about to reason about each separately. CHECK: grep the local
literature folder for "general geometry" / "arbitrary geometry" / "general
formulation" BEFORE the first per-family argument.

Corollary that pays on its own: **the general derivation also yields the
3-clause discriminator that GENERATES the table** rather than summarising it.
Here: a family needs α iff it has (1) an angular UNKNOWN, (2) indexed by a LOCAL
rotating frame, (3) whose derivative is discretised by COLLOCATION. Every "no"
row then fails a *nameable* clause instead of carrying a bespoke excuse — and
the clause is what a future session can re-apply to a family nobody asked about.

**(b) A refutation must record the QUESTION it was refuted FOR.** Same frame,
different question, opposite verdict — and because §3-style banners are permanent
(the plan-authoring "refuted premise stays" rule), the stale verdict is the one a
summariser carries forward. Worked: the **symplectic / momentum-map** frame is
recorded ⛔ REFUTED in [[cylinder-angular-march-jacobi-ladder]] — correctly, for a
DESIGN question ("does it buy a better scheme?" — no, because the chart that
conserves `p` by construction IS MoC). For a CLASSIFICATION question it is the
decisive answer: `[D]` `p = r sinω` (cylinder) and `p² = r²(1−μ²)` (sphere) are
exactly conserved along every characteristic, so *"a method that uses `p` as a
COORDINATE generates no redistribution term; a method that uses `(r,x)` must carry
the connection"* classifies MoC/CP/MC against SN/Pn in one sentence. ⟹ write
refutations as **"refuted FOR <question>; the FACT it establishes is <fact>"** —
the fact outlives the lever. This is L-001's sharpening, applied.

**(c) The elegance-verdict shape this produces, worth reusing verbatim.** When a
primitive is a general object composed with discretisation choices, do not answer
"is it general?" — answer with the LAYER TABLE (continuous object / evaluation
points / reconstruction rule) plus **two tests that point opposite ways**: hold
the measure fixed and change the chart (does it move? ⟹ not a measure invariant);
hold the chart and refine the measure (does it have a continuum limit as a
function of the coordinate? ⟹ it IS a chart object). Two opposed tests is what
makes it a verdict rather than a preference, and neither alone suffices.

How to apply: on any cross-method consumption brief — (1) find and read the
general-case derivation before arguing a single row; (2) extract the discriminator
clauses and generate the table from them; (3) check every in-memory ⛔ REFUTED
banner for whether it was refuted for THIS question; (4) close with the layer
table + the two opposed tests. Pairs with L-017 (an optimality theorem on the
target space bounds every design) — same family: **look for the theorem that
makes the survey unnecessary.**

---

## L-022 -- A FACTORY/dispatcher proposal is refuted or confirmed by THREE counts taken before any frame: the branches it would collapse, the members of its fiber, and who actually consumes the thing it claims to serve

Recurring shape: a design asks for "an object that returns the appropriate X
depending on the problem". It reads as consolidation, so the reflex is to argue
about scope. Three counts settle it faster than any argument, and each is one
grep. Take all three BEFORE reaching for a frame ([[iso-family-factory-refutation]]).

- **(a) Count the runtime BRANCHES the factory would collapse.** A dispatcher's
  only honest job is to move a repeated conditional to one site (the
  `discriminations` "a repeated conditional is a missing type" move). `[M]` if the
  count is **zero** — every construction site knows statically which member it
  wants, because the CONSUMER's identity decides — then the factory **manufactures
  the discrimination and then names itself the fix**, converting a
  statically-known fact into a runtime dispatch. That runs
  illegal-states-unrepresentable backwards. Worked: 4 construction sites of the
  SN/diffusion iso pair, 0 branches.
- **(b) Model it as a SECTION of a fibration and count the FIBER.** `Op → Space`
  with the factory as `s : Space → Op` is the literal shape of every "problem →
  object" proposal. A section needs a canonical choice per fiber. Enumerate the
  fiber by grep. `[M]` four legitimate members of ONE channel over ONE space
  (composite binding, energy satellite, and the two within-group split siblings)
  ⟹ no section ⟹ the key must gain a second coordinate, and that coordinate is
  almost always a **ROLE** — i.e. the consumer/posing concept, which is usually
  deferred to another campaign. **The factory's missing key is the finding**, not
  its size.
- **(c) Grep who CONSUMES the thing the justification names.** A justification
  ("subsystem Z needs both flavours") is a claim about the tree. `[M]` DSA
  consumes **zero** channel operators — it reads `diag(K_ℓ)` as a per-cell scalar
  COEFFICIENT. A diagonal is not a member of the transfer-kernel family, so no
  factory over that family can produce it. ⟹ the justification is refuted AND it
  converts into a smaller true deliverable (a kernel verb `within_group()`
  returning the per-ℓ diagonals — the multigrid Galerkin restriction
  `e_gᵀ K e_g`), which is L-013's shape.

⭐ **The strongest counter-evidence to any co-sourcing factory: an existing
SATELLITE property.** A factory co-sources two objects **by CONVENTION** (same
inputs, two calls, agreement by discipline). A `cached_property` returning a
binding of the parent's OWN datum co-sources them **by CONSTRUCTION** (one datum
instance, one object graph — agreement is a theorem). When a proposal asks for a
factory "so the pair stays consistent", grep for a satellite first; if one exists,
the request is already satisfied by the stronger mechanism and saying so is the
whole answer.

⭐⭐ **And the datum that reframes the request when you find it: shipped
machinery with ZERO production callers.** Worked: `S.foldable_part()` /
`residual_part()` already implement operator-tier "two flavours of one channel",
gated at `rtol=1e-14`, with `[M]` **zero** call sites in `orpheus/` (every
invocation is in `tests/`). The proposal's premise was implemented twice over.
⟹ when the capability exists and is unconsumed, the productive question is NOT
"how do we produce it" but "why does nothing call it" — which relocates the whole
brief to the consumer side. TELL: a name grep whose `orpheus/` hits are all
definitions and docstring cross-references.

**Two riders, both cheap:**
- **The steelman usually survives at a smaller scale — say what it is.** Here the
  real duplication was the tier-2 EXTRACTION chain (two guards with two messages,
  two scalar-sub-space derivations, two tier names), whose native fix is two verbs
  on the space type, not a factory. Currying tells you which: `bind(·, space)` is
  the only partial application with a REUSED operand.
- **A grep for a stale-owner docstring pays while you are there.** Measuring (c)
  found `material_xs_field.py:760,780` still naming an owner that stopped
  consuming those accessors two commits earlier — a present-tense-false doc bug
  found for free by asking *who consumes this*.

How to apply: at any "should we add a Factory / Manager / Builder / dispatcher"
dispatch, run (a), (b), (c) first and report the three counts. Zero branches +
non-singleton fiber ⟹ refuted as a dispatcher; then look for the INVERSE object
(a functor GENERATING the sibling, not a dispatcher CHOOSING it) — a shared
primitive whose own docstring states a defer-until-2 trigger is where to look,
because the trigger has usually already fired.

---

## L-023 -- On a SPLIT brief, do two things before frames: sort the FUSION POINT's attributes, and run the brief's own discriminating rule over the tree's straddlers counting INVERSIONS

A "split X into A and B" brief (Problem/Solution, model/run, data/behaviour,
kernel/driver) invites an argument about where the boundary goes. Two
measurements settle more than any argument, and both are greps
([[problem-solution-split-frames]]).

**(a) The boundary almost never runs where the brief names it — find the object
where the two halves are currently FUSED and sort its attributes.** A brief that
says "split the hub from the result" is naming the two objects that already
EXIST; the fusion lives in the third object nobody named, because that is where
the split's absence is doing its work. Sort every attribute that object's
`__init__` sets and report the counts.

Worked (CS4c §22.5): the brief named `SNMesh` and `Solution`; `[M]` the boundary
runs through `SNSolver.__init__` (`sn/solver.py:1418-1589`), which is **12
Problem-side / 6 Strategy / 2 Solution-in-progress** — 60 % of the "solver" is
the Problem. That one count (i) sizes the carve, (ii) shows the campaign owes
THREE types not two (the fiber coordinate has no home either), and (iii) names
the five Problem-side data that exist nowhere else, so a returned Solution cannot
be re-solved. It also converts a design argument into a checklist.

**(b) A chartered discriminating rule is a hypothesis about the tree — run it
over the straddlers and COUNT the inversions before proposing anything.** A rule
phrased on *what the code constructs* inverts on every case where an object is
manufactured without the mathematics moving. `[M]` 3 of 12 shipped straddlers
invert (a schedule kwarg that builds an operator yet leaves the fixed point
fixed; a splitting; and — the other way — a strategy object that builds nothing
at solve time yet deletes a term from `A`). The repair is one clause, and it is
the same clause every time: **phrase the rule on the SOLUTION SET, not on the
operator** — Problem-side iff changing it moves `{ψ : Aψ = q}`; Solution-side iff
the set is fixed and only the path to it changes. That phrasing also IS the
acceptance gate (solve one problem at two values of each Solution-side
coordinate; the limits must agree to a tolerance that SHRINKS with the
tolerances).

⭐ **The corollary that pays on its own: report the row you got WRONG.** My first
pass ruled `inner_schedule` Problem-side by following the chartered rule
literally, and the tree's own measurement inverted it. A rule that a careful
reader follows into the wrong answer is a finding about the rule, not a slip —
say so in the deliverable, with the row.

**(c) Two riders, each a one-grep check worth running on any split brief.**
- *Which side has identity?* L-019 says a God object is a fixed point of a weak
  identity relation. The INVERTED case is now sighted: `[M]` everything the hub
  induces has structural `__eq__`; the hub has none. A "the container's identity
  IS the identity of what it induces" charter is then **false in the only
  checkable direction**, and the fix is to DERIVE the container's `__eq__` from
  its induced objects rather than hand-write it.
- *Does the proposed A-side get MUTATED by the B-side today?* Grep
  `<b_object>\.<a_object>\._` and `setattr`. `[M]` four sites stamped σ-dependent
  memos onto the save-state candidate, and the reader never validated σ. A
  save-state charter is refuted by any such site, and enumerating them is the
  minimal enforcement (freezing is the cheapest ENFORCEMENT, not the
  REQUIREMENT — the requirement is only that solving does not mutate).

How to apply: on any split/carve/ontology brief — (1) find the fused object and
publish its attribute sort; (2) run the brief's rule over the straddlers and
publish the inversion count with the re-phrasing that fixes it; (3) check the
identity direction and the mutation sites. Only then reach for frames. Pairs with
L-013 and L-017 (the true deliverable is usually smaller than the proposal and
sits somewhere else) and with L-022 (the fiber count — here the fiber is the
STRATEGY, and a non-singleton fiber means the campaign owes a third type).
