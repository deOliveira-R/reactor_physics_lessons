# CS4c design record — every operator receives its two spaces and its minted data

**STATUS: LIVING — original round CLOSED 2026-08-30 (§1–§12); step-3 round §14; step-4 round §16 (2026-08-30, EXECUTING). Live resume surface: §15 COMPACTION POINT #2. No deferred forks remain open except the §16.3 factory probe (ruled separately).**
This is the design-round record for the CS-ladder remainder (Campaign 2's
opening act; plan of record `.claude/plans/cs_ladder_remainder.md`; charter
`.claude/plans/space_and_kernel_binding_campaign.md` §CS4c/§CS2). It is a
LIVING document under `plan-authoring.md`: rulings are edited in as they land,
refuted text stays with its refutation, every claim carries its epistemic
marker. Nothing here is executed until the round closes and test-architect
runs (the proactive trigger).

Ground memos (both at `fc60ea64`, 2026-08-30, read-only explorer dispatches):
- `scratch/cs4c_opener_structural_ground.md` — ledger rows, frame-independence
  verdict, leaf surfaces, CS2 residue, #359 remains, the 10-item charter
  corrections list.
- `scratch/cs4c_opener_count_census.md` — the four count re-censuses with
  predicates (S4/S5 DISSOLVED; C-migration 7-vs-133 nuance; R-A roster intact).

Marker key: **[R]** = ruled by the user (dated, quoted where load-bearing);
**[P]** = main-agent proposal, presented and awaiting ruling; **[M]** =
measured (with source); *hypothesis* = unverified means-sketch.

---

## 0. Ground facts the design stands on (all [M], 2026-08-30)

1. **The pull-forward's license HOLDS — via PROVENANCE.** The frame mint needs
   only what the space can answer through the CS5 generator channel
   (`axis.generator_as(Quadrature, consumer=…)`); the S² nodes live on the
   generator, not on the axis; the production bulk space is generator-wired
   axis-built (`augmented_mesh.py:1183`). The live S-side mint to re-route is
   `ScatteringOperator.frame` (`transport/operators/scattering.py:485-512`),
   which today reads the operator's own mandatory `quadrature` field.
2. **The acceptance instrument**: 3 marker constants / 14 strict-xfail
   param-rows in `tests/sn/architecture/test_monomorphic_leaves.py` —
   R1{L,C,S,B}, R2{C,S}, R6{B×8}. CS4c owns R1-C/R2-C/R1-S/R2-S; the CS2
   residue owns R1-L/R1-B/R6.
3. **Dissolved charter inputs**: S5 (909-site sugar) landed `2690a434`;
   S4 (21 field-`.mesh` reads) at 0/0/0; #359's three-spellings premise
   dissolved by P4.9a (one canonical body, `sn/angular/closure.py:1405-1435`);
   CS2 witness 2 (moment mass) remedied by construction.
4. **Binding-relevant landed substrate**: collapse pair
   `FunctionSpace.retraction`/`section` (frame-induced, born bound, S6.0/b);
   `DualSpace` with axes+metric threaded (P7 S2); `HilbertMetric` family
   (P7) whose docstring reserves the Riesz-leg composition; `ScatteringKernel`
   + `FissionKernel` minted with **0 production consumers** — this ladder is
   their first (⚠ second-pass correction 2026-08-30: the family is THREE —
   `N2NKernel` also ships, `kernels.py:222-246`, already carrying
   `multiplicity: ClassVar[int] = 2` as the datum home XD-2 names); L's binding ruled + P4.9b landed (transitional
   `(sn_mesh, spatial_closure, angular_closure)` ctor; end state = the
   4-tuple `(domain, codomain, spatial_closure, angular_closure)` riding O-3).
5. **Dispatch surface**: exactly 3 `singledispatchmethod` sites —
   S (4 arms, no ndarray), C (FullField/ndarray), F (FullField/ScalarFlux/
   ndarray). ⚠ Step-0 census correction (2026-08-30): S's `FullField` arm is
   a RE-DISPATCHER (`scattering.py:1189` re-enters the dispatcher on
   `psi.interior`), so "4 arms" over-counts bodies and under-counts
   branching — construction-time selection relocates that branch one frame
   in; the assembly-owned tier-3 lift is what removes it. The ndarray k-path is alive and protocol-wide
   (`numerics/eigenvalue.py:420` → ndarray-typed realizers in SN/diffusion/
   cp/moc). The 8 `MaterialXSField.apply_*` arms (:743-1023) are consumed
   ONLY by the S family (14 sites) — O-6's move rides the S rebind.
6. **Posing-filtration rulings bind this design** (R13–R23, 2026-08-25):
   CS1.5′ superseded in shape (`Materials` landed; `InfiniteMedium` is
   post-ladder, R23); `MaterialXSField` dissolution ratified (R13);
   B's reshape waits for the crystallized operator concept (R18).

---

## 1. The binding base — [R] corrected: EVERY bound operator carries its TWO spaces; `(datum, space)` was a guide, not the signature

**[R] (user, 2026-08-30, verbatim in substance):** *"space is not enough to
determine domain and codomain. It would be enough naively… but once we have
Riesz operators implemented, even the naive version will die. At the very
least every BoundOperator will need its domain and codomain, and different
amounts of data (the collision here is the easy one, but the scattering will
require kernel and frame for example)."*

**Main-agent corroboration — the tree already proves this four ways:**
- **EE-8's own recorded limit** (charter, CS4a-R): the `(space, ng)` guard
  "structurally assumes an ENDOMORPHISM — a 47g→2g condensation binding has
  no single ng conforming to both ends." The charter knew the one-space
  spelling was the degenerate case.
- **ERR-076** is the founding measurement: an operator that derived NO spaces
  from its factors silently degraded `(K_ω ⊗ I).H` from the Hilbert adjoint
  to the Euclidean transpose (87 % on a production law).
- **XD-7**: `_agreed_space`'s stated expiry — per-leg bindings make agreement
  the wrong law; resolution must become BY LEG. Per-leg = each leg knows its
  own two spaces.
- ⭐ **A SHIPPED witness (step-0 census, [M] 2026-08-30)**: on 2-D
  Cartesian, S at `sn/solver.py:1406` receives a composite whose interior is
  the MOMENT space while its bound `space` names the ANGULAR-interior
  composite — `domain ≠ codomain` live in production, the single `space`
  field naming only the codomain (`scratch/cs4c_feeding_census.md` §3).
  ERR-076 is the historical version of exactly this.
- **L's ruled end state is already the corrected shape**: the O-3 4-tuple
  `(domain, codomain, spatial_closure, angular_closure)`. The user's
  correction makes S/C/F converge to the SAME discipline — one shape across
  the 3+1+1 datum kinds. And the moment route's internal factors (analysis
  `V → moments`, reconstruction `moments → V`) are non-endomorphisms even
  when the composite is one; with `DualSpace` shipped, `riesz_lower: V → V*`
  is non-endo by nature.

✅ **[R] RULED (user, 2026-08-30, the step-1+2 checkpoint): kw-only
mandatory `domain=`/`codomain=` fields on the base** — the datum stays
positional-first; the swap-transposition family (ERR-002/ERR-076 habitat)
is unspellable-silently at every exact-ctor site; channels evolve their
field lists independently of the base. And the analysis-verb declaration
lives on the FRAME-CONSUMING channels only, never the base — the user's
articulation: *"StreamingOperator is also a BoundOperator, just bound to
other data"* — the base carries only what every binding shares.

**Consequence for the base [P]:** `BoundOperator` is a dataclass ABC carrying
`domain` + `codomain` (both mandatory) + the shared behavior: per-END
admission (the energy-conformity guard runs against the domain's AND the
codomain's energy axis — one check collapses the two for endomorphisms), the
mint-and-forget storage contract, and the declared analysis-verb field
(XD-1's home). Each channel declares its own datum fields — C: coefficient;
S: kernel + minted faces; F: kernel (+ faces on angular spaces); L (later,
O-3): the closure pair. The abstract `data_ng` stays deleted (XD-1 stress
verdict). Endomorphism sugar (`codomain` defaulting to `domain`) may live on
the CLASSMETHOD tier, never on the base.

---

## 2. The frame is CONSTRUCTED OUTSIDE and handed in — [R]; the operator mints its faces from it and forgets it (accessor stays for provenance during prototyping)

**[R] (user, 2026-08-30):** minting the frame inside `ScatteringOperator`
construction *"prevents us from using the same Frame for the Fission Operator
and for the Windowing solution method. In principle the Frame needs to be
constructed and given as an argument, together with domain and codomain. The
Frame should act as an Analysis and Reconstruct Operator factory: during
ScatteringOperator construction it uses the frame factory to produce the
operators it needs, then forgets the frame (an accessor can stay behind for
now for provenance and during prototyping, until we're sure it can be
completely gone)."* Same concept applies to `FissionOperator`.

**This is the posing-filtration §5c interning doctrine, now applied to S/F
[M]:** *"sharing justifies interning — one frame per axis pair, many arrows
per consumer (Windowing minting its own differently-bound arrows from the
same frame is the use case). The cache triad: kernel per cross-section set,
frame per axis pair, arrows per space."* It is also the stage-2 generator
discipline verbatim (Frame/Scheme induce; forgetting = retaining the induced
parts; accessors are provenance). F-1's commit subject already says it:
*"the frame is the shared factory; faces are the bound operators."*

**[P] — the synthesis with the ctor/classmethod discipline (§3):** the EXACT
constructor takes the minted faces (what the operator retains); the
extract-and-mint CLASSMETHOD takes `(kernel, frame, domain, codomain)`,
mints the faces, and feeds the ctor, attaching the frame reference as a
`compare=False` provenance field (the accessor, retirement-tracked). Two
reasons the faces — not the frame — sit on the exact ctor:
(a) the XD-1 gate's negative controls REQUIRE constructing
deliberately-wrong pairs (the wrong-embedding control), which a
frame-welded ctor forbids without doctoring the frame — the exact defect
the readiness assessment flagged ("two shipped gates require a doctored
closure to get its own values");
(b) vv#28's fixture trap closes by gating classmethod-built ≡ ctor-built.
✅ **[R] CONFIRMED (user, 2026-08-30):** *"Your read and my intentions
match."* The frame arrives at the classmethod tier; the faces sit on the
exact ctor; the provenance field rides until proven removable.

✅ **[R] RULED (user, 2026-08-30): the HUB holds the shared frame** — by
symmetry with the P4.9b scheme ruling (the hub keeps the generator; the
operator holds the minted products). One frame per (axis content, L),
shared by S, F, and the Windowing method; consumers that hold only the
space still reach the generator through `generator_as` (the CS5 channel).

---

## 3. The construction discipline for ALL operator families — ✅ [R] RULED (user, 2026-08-30): the three-tier discipline is ENDORSED

**The question [R-posed]:** establish ONE discipline in wide context — do
Bulk/Full/Boundary operators each get exactly their space (bulk/full/trace),
or does everyone get the full space and extract? The user's stated habit:
*"the class constructor takes just what it needs, and a class method takes
more than it strictly needs and extracts exactly what it needs to feed the
construction."*

**[P] The three-tier discipline** (the habit, promoted to law, with the
extraction placed once):

- **Tier 1 — the exact constructor.** Takes precisely what the instance
  RETAINS: its two spaces + its operative data (coefficient / kernel +
  minted faces / closure pair / realized law table). Nothing richer. This is
  Pattern 4 (illegal states unrepresentable), the cheap-fixture surface, and
  what the Riesz legs read (`A* = domain.riesz_raise ∘ A.dual() ∘
  codomain.riesz_lower` needs both ends' metrics — exact spaces make it a
  theorem).
- **Tier 2 — extract-and-mint classmethods.** Take richer objects (full
  space, frame, scheme, kernel, mesh, mixture) and extract/mint exactly,
  then call the ctor. `MultiplicationOperator.from_mesh` is the shipped
  precedent; P4.9b's `.pose(sn_mesh)` is another. R1 (posing arc) is
  honoured: operators stop REQUIRING the mesh (ctor tier) while transitive
  reach at the factory tier stays legal. **Every classmethod owes an
  equivalence gate** (classmethod-built ≡ ctor-built on the same inputs) so
  the exact-ctor fixtures provably represent production (closes vv#28's
  simple-ctor-vs-composite-factory blindness).
- **Tier 3 — lifts between space kinds are explicit verbs, never arms.**
  A bulk-born operator enters a full-space composition through the
  restriction/extension family (`embed ∘ A_bulk ∘ restrict` — extension by
  zero on the trace), owned by the ASSEMBLY at one site, not by a FullField
  arm on every class. The repeated "which block do I act on" conditional is
  the missing type; the verb is the type.

**The per-family answer this yields:** which space a binding takes is a
property of the BINDING, not of the class — recorded per row in the
binding-arity table. The within-group members bind the composite (that is
the space the sum lives on); the k-outer F binds the bulk space (its operand
is a flux distribution, never a trace — ⛔ ASPIRATIONAL: [M] step-0 census
2026-08-30, the tree binds `sn_mesh.full_field_space` at `sn/solver.py:1412`
while feeding bare `(ng,*spatial)`; step 4 makes the bulk binding true); law objects bind trace spaces
exactly (Γ₊ → Γ₋), with the boundary realization applying the crystallized
concept later (R18). At construction, the declared domain selects the ONE
action body — construction-time selection is exactly what the R-A census
forbade **until feeding is normalized**, which is step 0/step 5's job; after
normalization it is the legal mechanism O2 names.

✅ **[R] (user, 2026-08-30), the load-bearing articulation:** *"As long as we
get the essence right (the ctor), then we have a lot of flexibility to change
class methods going forward as the code evolves. Maybe we change our mind on
how the class method should work based on experience enforcing V&V discipline
… but if we get the design right, the core survives."* ⟹ the exact ctor is
the DESIGN COMMITMENT; the classmethod tier is deliberately evolvable without
touching the core. Endorsed as proposed, tier 3's lift-verbs included.

**Angles weighed** (for the record): testing (exact ctors = minimal
fixtures + hand-built negative controls); illegal states (a bulk S cannot
hold a trace it never touches); sharing/interning (factories at tier 2 keep
one mint site per consumer); sum agreement (lifting keeps the within-group
sum single-space today, deferring XD-7's per-leg law to the consumer
campaign); vv#28 (guards keyed on what the object always carries — the
exact spaces); #394's JAX lowering (exact retained arrays are the static
boundary).

---

## 4. Kernels COLLAPSE to one datum per interaction — [R]; operators ask for the truncation they need at construction

**[R] (user, 2026-08-30):** *"The kernels at least should collapse. When the
operators are constructed, they ask for what they need — SN asks for
anisotropy up to whatever order it wants; diffusion asks for just the
isotropic part at construction of its operator."*

[M] `ScatteringKernel` already carries exactly this surface (`p0`,
`truncated(L)`, the ℓ-stack over one `Mixture`). The O7 twin heal follows:
`LegendreMomentScattering`, the iso pair, and `N2NMomentOperator` become
bindings/truncations of the one datum; the 8 `MaterialXSField.apply_*` arms
(S-family-only consumers, 14 sites) become the kernel's array verbs (O-6,
R13's ratified dissolution advancing as a by-product).

⭐ **Step-0 census addenda ([M] 2026-08-30, `scratch/cs4c_feeding_census.md`):**
(a) the satellite mint rate is 1-per-APPLY, not 1-per-solve (up to 911 LMS
instances in one Krylov k-solve) — §4's collapse converts it to
1-per-construction, a measurable cost win with its baseline now recorded;
(b) forward and adjoint S take structurally DIFFERENT internal routes
(`N2NMomentOperator.apply` has zero forward traffic; the forward path builds
the (n,2n) source through `add_n2n_source`) — a twin-path fact the O-6 arm
absorption must price; (c) the iso pair is a measured ASYMMETRIC arrow
(`ScalarFlux → bare ndarray`, 4816 applies, chart-dependent bare-ndarray leg
on sphere/cylinder).

⚠ **AMENDED 2026-08-30 (step-3 design round, §14.1):** the n2n half of this section's O7 heal is re-homed — N2N is its OWN first-class operator, NOT a field of S; the kernel collapse stands.

**Open residue [P]:** the operator-CLASS fate. The kernel collapse is ruled;
whether `IsotropicScattering`/`IsotropicN2N` survive as named ℓ=0 binding
recipes (diffusion's spelling) or dissolve into truncated
`ScatteringOperator` bindings is a step-3/4 design decision — decided with
the census's per-binding carrier table in hand.

---

## 5. The solution path is DEFERRED to its own campaign — [R]

**[R] (user, 2026-08-30, verbatim in substance):** *"I don't want to put much
focus on the solution path yet (which consumes what we're doing) — the
solution path will have a detailed campaign, and it will drastically reshape
the machinery of the consumers, because now we understand the ontology of the
consumers much better than when we first designed it."*

Consequences for this ladder: step 5 (arm deletion + #205/#276) takes the
**solver-side adapter** route — operators end whole, the ndarray hatches
close via explicit adapters at the solver seams; typing the k-outer iterate
and every consumer reshape rides the consumers campaign. The within-group
assembly keeps its current shape (tier-3 lift at the existing single site);
nothing in this ladder redesigns iteration/posing machinery.

---

## 6. The tightness gate's adjoint leg — [P] operator-side, explained

The binding's chartered correctness gate exists to catch the ERR-039 family
(a claimed `Π* = R` that wasn't). CS4a-R's XD-1 probe REFUTED the original
spelling — `bind(K)† = bind(K†)` holds at 2.2e-16 even for a deliberately
non-tight rule under the shipped un-normalized analysis verb, so its negative
control would have PASSED. The re-specified gate: (i) the Galerkin leg with
the wrong-EMBEDDING negative control (w-weighted vs constant — the measured
G6.3 87 % class); (ii) the MULTIPLICATIVITY leg
(`bind(K₁K₂) = bind(K₁)·bind(K₂)` — breaks at 1.58 non-tight, 1e-14 tight)
at ℓ≥1; (iii) the recorded ℓ=0 blindness (both laws hold for every rule).

The unresolved half: any adjoint-LAW leg needs the operand `bind(K†)`, and
**the adjoint kernel does not exist as a type** — [M] no kernel carries any
dagger API, and `FissionKernel(chi=νΣf, nu_sig_f=χ)` is REFUSED by its own
χ-simplex guard (the swap breaks the normalization invariant), so F's
adjoint kernel is a DIFFERENT type, not a re-parametrization. Two routes:

- **(a) kernel-side dagger machinery**: mint typed adjoint images
  (`AdjointFissionKernel` etc.). Cost: a type family with no other consumer
  until Campaign 2's pencil possibly wants F† as a first-class datum.
- **(b) operator-side [P, lean]**: realize S/F as composites with
  space-supplied faces (the F4 addendum's mandate), land the Riesz legs
  (accepted suggestion 4), and then `.H` is the reversed composite of the
  factors' adjoints — a THEOREM of the factors. ~~The gate compares `bind(K).H`
  against an independently-ASSEMBLED adjoint (built from daggered factors,
  never through `.H`)~~, plus legs (i)–(iii). Kernel daggers are minted at the
  pencil, with their consumer, if Campaign 2 demands the datum.

  ⛔ **REFUTED 2026-08-30 (test-architect pre-carve round,
  `scratch/cs4c_verification_plan.md` §1.1): the struck comparison is a
  TAUTOLOGY and cannot red under ANY embedding.** [M]
  `(RKM)† = M†K†R†` is an algebraic identity for any three maps and any
  nondegenerate metrics — a wrong embedding enters BOTH sides and cancels
  (measured ≤ 2.24e-16 under correct, constant, AND unweighted embeddings).
  The RULING's substance stands untouched (no kernel dagger; adjointness is
  operator-side, by theorem — the theorem-ness is exactly what the probe
  confirmed). The ERR-039 CATCHER is leg (i), the Galerkin defect on the
  FACES (`‖M† − R‖/‖R‖`: [M] 0.0 correct vs 2.0e-1…3.2 wrong). The identity
  is kept as a documented structural theorem in the gate module's docstring,
  not as an assertion (vv#19: a reading that cannot change is not evidence).

  ⛔ **And leg (ii)'s negative control must be chosen by MEASURED redness,
  not by non-tightness magnitude** (plan §1.2): [M] `gauss_legendre(L)` is
  maximally non-tight (`‖MR−I‖ = 1.0`) yet bit-clean on multiplicativity for
  zonal operands (≤ 5.9e-16, 200 seeds × L∈{1,2,3}); `equispaced_equal` reds
  at ≥ 1.2e-2 and is the control that ships.

✅ **[R] RULED (user, 2026-08-30): route (b) — and the kernel-dagger
question is RESOLVED: no kernel dagger, in this ladder or later, absent a
consumer that can state the missing hypotheses.** The user's argument,
confirmed with one refinement:

- **The user (correct):** an adjoint only makes sense once the kernel is
  realized into an operator — domain and codomain with their Hilbert
  metrics are necessary to define it (⟨Aψ,φ⟩_{G_W} = ⟨ψ,A†φ⟩_{G_V} reads
  both), which is why the operator level pins it by theorem; *"if it cannot
  be pinned by theorem the math is probably incomplete to define the
  meaningful dagger."*
- **The refinement:** a metric-free DATA involution does exist (argument
  swap / matrix transpose / factor swap) — well-defined as pure data. What
  it is NOT is an adjoint: `bind(swap(K)) = bind(K)†` is a CONDITIONAL
  theorem of the binding, with hypotheses only the binding can check
  (counting measure on energy — the CS4a gate that "CANNOT RED, [M] 0.0
  under defect and fix" is this theorem's triviality; tight frame on
  angular — XD-1's multiplicativity finding; nondegenerate metrics). A
  kernel-level `.dagger` claiming adjointness would place the claim where
  its hypotheses are unknowable — vv#28's shape, the F4 addendum's
  Σw-placement warning, and ERR-002's transpose-convention drift habitat
  reopened. Where the swapped data is genuinely needed (inside the
  independently-assembled adjoint), it is array-verb consumption of the one
  datum, never a type.
- **Type-level corroboration [M]:** `FissionKernel(chi=νΣf, nu_sig_f=χ)`
  is refused by its own χ-simplex guard — the type CORRECTLY says the
  adjoint image is not a forward fission datum (the simplex invariant is
  directional physics). And L23(b) already rules "the adjoint is a daggered
  POSING row" — even Campaign 2's pencil consumes bound adjoints, not
  kernel daggers.

---

## 7. Accepted suggestions — [R] all six (user, 2026-08-30)

1. **CS2 witness 1 (BC) re-ruled correct-by-doctrine**: spaces are law-blind
   (F2's DOF-set+Gram criterion); the done-when's BC aliasing probe becomes a
   POSITIVE gate (law-blindness by design); BC-keyed state stays on the mesh.
   Charter edit owed at the CS2 section (in place, per plan-authoring §3).
2. **The S3 flip is a bridge-retirement**: digest names already carry axis
   content through `(name, shape)` equality; the flip makes identity
   structural directly and retires the bridge. Not blocking steps 0–5.
3. **The densifier retirement goes native**: `_tensor_product_factored_metric`
   (P7's new consumer, `space.py:901-905`) reads per-axis diagonals natively;
   `_dense_axes_weights` retires with both clients re-pointed.
4. **The Riesz-leg debt rides this ladder**: mint `riesz_lower`/`riesz_raise`
   on the space wrapping the `HilbertMetric` faces; retire `_AdjointOperator`
   into `A* = domain.riesz_raise ∘ A.dual() ∘ codomain.riesz_lower`
   (`.H` consumers unchanged); #375 expected to dissolve; Λ's factor
   collection rides step 3's frame work. The RESTRICTION verb stays parked
   for the boundary thread unless step 4's composite genuinely needs it.

   ⚠ **Three pre-carve measurements bound this step (plan §1.5/§1.6/§2,
   2026-08-30):** (a) the honest round-trip law is
   `riesz_raise ∘ riesz_lower = P_range(G)` — identity on the bulk,
   the tangential-zeroing projector on the trace ([M] a legal
   `product(4,4)` 2-D mesh has 32/64 tangential slots, trace defect 2.87;
   all four ledger fixtures carry 0 tangential slots and are blind) — a
   `== id` gate is both blind on the corpus and a false red in production;
   a singular-metric fixture SHIPS with the legs (R3 resolved: yes).
   (b) the two verbs live on the PRIMAL space only — `DualSpace.of(V)`
   deliberately carries the primal's metric, so a generic
   `apply_metric`-as-lower on a DualSpace yields G² ([M] `[0.25,4,16]` for
   `w=[.5,2,4]`); never write the double-Riesz involution gate.
   (c) ⭐ the retirement is a measured COVERAGE UPGRADE — splitting the
   paired metric mutation into single legs takes the ledger 9/20 → 20/20
   rows red — with the dual obligation that the flat-metric BLINDNESS
   control keeps the PAIRED mutation ([M] a single leg reads `|1−c|`
   exactly on the flat slab: honest arithmetic, false red); a stale
   `_METRIC_CONSTRAINED` after the split is silent coverage loss.
5. **#359 hygiene now**: comment posted re-pointing the site map (canonical
   body + deliberate gated ULP-twin; premise dissolved; residue =
   close-verify). No code.
6. **`_interior_space`'s unwitnessed refusal**: skipped iff step 3 lands this
   campaign; otherwise owes one witness.

---

## 8. The step ladder (updated hypothesis — re-shaped by §§1–7; step design per phase)

| step | content | ledger rows | notes |
|---|---|---|---|
| 0 | ✅ DONE 2026-08-30 — the feeding census (vv#29), `scratch/cs4c_feeding_census.md` (11 entries × 13 sites × 23 verbs; per-arm activation controls; 11/11 headline numbers non-perturbed) | — | HOMO traffic did NOT move (only the space's provenance did); SN C is minted per-outer and NEVER applied (fused override reads its data — vv#29 mode (d)); #205 ScalarFlux arm corroborated at 0 traffic with an AST-closed reference set |
| 1 | ✅ EXECUTED on `refactor/cs4c-binding-base` (2026-08-30, `68a9c9f3` + `733d96f3`) — Riesz legs + `dual()` + the AdjointOperator re-expression/promotion (§7.4 as amended by R2); #375 closed; per-leg ledger battery (9/20→20/20) | — | the retirement became a RE-EXPRESSION (R2 ruling — see §12-bis); [M] tests/numerics 2538 + tests/sn/operators 1249 green; sphinx -W clean; dead_references 0/52 |
| 2 | ✅ EXECUTED on the branch (2026-08-30) — `BoundOperator` base (kw-only write-once ends, mechanic E: post-class property injection over dataclass fields — [M] the InitVar route is pyright-hostile, the bare-field route re-abstracts under `abc.update_abstractmethods`); C rebound; `from_mesh` = tier-2 sugar + bottoming-out refusal; G-C1 gates | R1-C, R2-C ✅ flipped (14→12 xfails) | [M] tests/sn 3352 + transport/diffusion/homogeneous 730 green; 26 direct + 7 anonymous sites re-keyed; ⚠ residue for a later checkpoint: C.H-as-object-identity elegance fork (kept the uniform leg route for battery coverage); the M.H==M pin re-keyed to nulp=4 (the metric roundtrip now EXECUTES) |
| 3 | ✅ MERGED 2026-08-30 @ `600c5c80` (ff-only; 3a `c0e904ea`, 3b-A `8f376135`, 3b `b435431c`, 3c `81e9e7e1`, 3d `92dcc30f`, docs `600c5c80`) — S rebind + the §14.1 N2N extraction + O-6 + XD-1/XD-2 + the corpus pass; ledger R1[S]/R2[S] flipped, `_R2_XFAIL` deleted; both pins re-keyed; #306 item 4 closed-in-place (issue stays open for items 1/3/5) | R1-S, R2-S ✅ | gate [M] **10079/0/19sk/66xf, 13 trees rc=0** (`scratch/_cs4c3_fast_gate.log`; per-tree reconciliation in `scratch/cs4c_step3_predicted_deltas.md` — every delta named: transport +61, sn +9, root +3 corpus-coupled, ten trees 0) |
| 4 | F rebind: `FissionKernel` first consumer; composite realization with space faces (F4 addendum; collapse pair ships); XD-9 condensation gate; XD-2 (n,2n) count gate | — | |
| 5 | arm deletion + #205/#276 re-litigation with census evidence; solver-side adapters (§5) | — | construction-time body selection becomes legal here |
| 6 | CS2 residue: S3 bridge-retirement + densifier-native (§7.2/.3); L/B minimal annotation flips; R6 carrier guard at `boundary.py:714` | R1-L, R1-B, R6 | minimal by design — O-3's 4-tuple and R18's B reshape NOT pre-empted; ⚠ three boundary classes carry Optional |
| coda | homogeneous path re-points last (F5); `from_materials` consumer dissolves; O1 completes | — | `InfiniteMedium` (R23) stays post-ladder |

Standing: surgical posture; test-architect BEFORE step 1's first edit; L17
crosswalk first; per-step batteries; predicted-then-measured deltas vs
9950/0 (13 trees rc=0); branch + ff-merge.

⭐ **STEPS 1+2 MERGED 2026-08-30 @ `9fc8bf04` (ff-only).** Merge gate:
[M] 13 trees rc=0, **10006 / 0 / 19 sk / 227 des / 68 xf** (driver
`scratch/_cs4c12_gate_driver.sh`, log `scratch/_cs4c12_fast_gate.log`,
64:42 wall). Predicted 10005/68 — the +1 reconciled exactly: the layer
gate's per-module parametrization gained `bound_operator.py`'s row (the
P7 corpus-coupled-count mechanism, this time caught by the discipline).
Baseline for step 3 is therefore **10006/0/19/227/68**.

⚠ **Second-pass review 2026-08-30 (§12) amends this table**: steps 1+2 fuse
to ONE merge unit (F2); step 2's §6b population is the direct-ctor re-key
set, not the 7 anonymous sites (F3); the R1 reader must be extended in the
same commit as the first field-tier flip (F4); step 3's test-side set is
[M] 99 calls / 18 files (F5); XD-2's count gate moves to step 3 and carries
a user fork on its denominator (F6); XD-1's multiplicativity operand needs a
spelling decision (F7).

---

## 9. Open forks

| id | fork | status |
|---|---|---|
| F-A | the three-tier construction discipline (§3) | ✅ [R] 2026-08-30 — endorsed; the ctor is the surviving core, classmethods deliberately evolvable |
| F-B | frame at ctor vs classmethod tier (§2 synthesis) | ✅ [R] 2026-08-30 — confirmed (classmethod + provenance field) |
| F-C | adjoint-leg route (§6) | ✅ [R] 2026-08-30 — operator-side (b); NO kernel dagger (resolution recorded in §6) |
| F-D | per-binding space table (within-group composite / k-outer bulk / …) | census LANDED 2026-08-30 — observed table at `scratch/cs4c_feeding_census.md` §3 ([M]: bound space is a faithful domain in HOMO only, 1 of 4 families; DIFF needs {composite, scalar-bulk} per binding; SN C's space is a label on data). Ruling at step design. |
| F-E | iso-pair operator-CLASS fate (§4 residue) | ✅ [R] RESOLVED 2026-08-30 (step-3 design round, §14.2): both survive by construction as the energy-space bindings the composite operators lift |
| F-F | CS2-residue placement | standing lean (late, step 6) — presented, unobjected |
| F-G | frame interning site (§2) | ✅ [R] 2026-08-30 — the hub |

## 10. Supersessions this round creates (owed edits, executed when the round closes)

- ✅ Charter §CS4c: the `(datum, space)` EE-6 spelling annotated as
  guide-superseded by §1's two-space correction (in place, 2026-08-30).
- ✅ Charter §CS2: witness 1/2 banner at the ontology-tournament bullet
  (in place, 2026-08-30).
- ✅ `.claude/plans/cs_ladder_remainder.md`: §3 staleness list resolved from
  the two censuses; §4 opening-protocol status stamped (2026-08-30).
- ✅ #359: comment posted 2026-08-30
  (github.com/deOliveira-R/ORPHEUS/issues/359#issuecomment-5469489901).

## 11. ▶ NEXT ACT (ruled 2026-08-30): a fresh-context SECOND-PASS REVIEW of this record, then the dispatch

**[R] (user, 2026-08-30):** *"I will give you the opportunity to compact
context and review the plan with fresh context to see if there is something
you'd tighten on a second pass. After you review it, we will do the proper
agent dispatch."*

The review's method (the CS3-R/CS4a-R clear-context precedent, scaled to a
PLAN): read the tree-facing claims fresh — this record top-to-bottom, the two
ground memos, the charter banners — and attack before balancing:
(a) **plan-authoring defects in the §8 ladder**: §6b step boundaries vs
call-site sets (does any step leave a call site speaking the old signature?);
§6c gates landing with their witnesses (what shipped input does each new gate
REJECT on landing day?); §10 done-when reachability (run every tell's grep at
its stated denominator, intersect with the declared untouched set);
(b) **contradictions between the recorded rulings and the tree** (re-derive,
don't trust this file's tense — §7 discipline);
(c) **unstated denominators/predicates** in any ladder row or ruling;
(d) anything the design round missed that a builder would trip on
(the review may propose additions, not only cuts).
Findings tighten THIS document in place ([M]-marked, dated). THEN:
test-architect dispatch (the pre-carve MUST trigger) + step 0, per §8's
standing block — both on the user's go after the review is presented.

## 12. Second-pass review findings (2026-08-30, fresh context per §11) — all [M] at `14201b48`

Method: §11's attack list, run against the tree with fresh context. Every
claim below was measured this session (commands: AST censuses via
`.venv/bin/python`, greps stated inline). Findings are edits-in-place
obligations on the steps they name; none reopens a ruled fork except F6
(one small user fork).

### F1 — step 1's `_AdjointOperator` retirement is underpriced, and its §6b set includes the ACCEPTANCE INSTRUMENT itself

- **(a) `A.dual()` does not exist on operators.** [M] `grep -rn "def dual"`
  → only `FunctionSpace.dual` (`space.py:804`) + `CoupledField.dual_zeros`.
  The §7.4 formula (verbatim reserved at `metric.py:84` ✓) therefore mints
  THREE new objects: an operator-level dual wrapper (apply =
  `inner.apply_transpose`, spaces = the duals) + the two riesz-leg
  operators. The retirement is a re-expression, not a re-plumbing.
- **(b) Four test files import/monkeypatch the private class directly**
  ([M] grep `_AdjointOperator` over `tests/`):
  `test_operator_capability_predicates.py` (:25/:177/:186/:280 — direct
  construction of the wrapper), `test_coupled_operator.py` (:78/:866 —
  monkeypatches `.apply`), `test_inverse_adjoint_coherence.py`
  (:92/:340/:377 — the M-ADJ mutation legs), and ⭐
  **`test_monomorphic_leaves.py` — THE LEDGER — at :608 module-scope
  `_TRUE_ADJOINT_APPLY = _operator_module._AdjointOperator.apply` + three
  monkeypatch sites (:939/:987/:1030, the M-10 family)**. Deleting the
  class kills the ledger file at COLLECTION (vv#17's third pipeline
  failure — reads as 0 failed). ⟹ the retirement commit migrates the
  mutation instruments to mutate the riesz legs / dual wrapper instead
  (coding-standards: retirement = test+marker migration), applied to the
  campaign's own acceptance instrument FIRST.
  ⚠ Pre-carve round extensions (plan §1.7/§1.8, 2026-08-30): [M] there are
  TWO collection-killers, not one — `test_operator_capability_predicates.py`
  constructs the wrapper inside a module-level parametrize list (`:280`);
  and the retirement retires FOUR surfaces, not one: `apply`, `inverse()`
  (the #280 swap law), `is_invertible`, and `apply_transpose`'s refusal —
  [M] the last has ZERO witnesses tree-wide, so its successor spelling owes
  one in the same commit.
- **(c) ~40 docstring references** to `_AdjointOperator` across `orpheus/`
  (`:class:` Python-domain refs — silent at every Sphinx severity) —
  `dead_references` sweep owed at the step, magnitude now stated.
- Day-1 witnesses exist ✓: the adjoint-coherence + factored-adjoint-identity
  gates exercise the new composition on landing.
- Nuance recorded: the unbound-`.H` refusal ALREADY ships
  (`operator.py:1313`, S4-amendment ruling 2026-08-22, with the pointwise
  self-adjoint exemption at `multiplication_operator.py:264-265`) — the
  retirement changes the MECHANISM of `.H`, not its admission.

### F2 — steps 1+2 are one MERGE UNIT (§6c)

The base ABC's per-END admission has no concrete subclass until step 2 —
no shipped input it can reject on landing day. Fuse: land base + Riesz
legs + C rebind on one branch (commits may stay separate inside it); the
R1-C/R2-C flips are the base's first witnesses.

### F3 — step 2's "7 sites" is the WRONG-PREDICATE count (§2 filter)

The 7 counts space-ANONYMOUS direct ctors. The step as ruled (§1: base
carries `domain`+`codomain` as mandatory fields; endomorphism sugar
classmethod-only) re-keys EVERY direct ctor call: [M] **26 direct C sites**
(3 production + 23 test, census 1's own numbers re-bucketed), while the
127 `from_mesh` spellings survive ONLY IF tier-2 keeps a one-space
`space=` parameter — a design decision the record did not state. ⟹ stated
now: **tier-2 classmethods keep the endomorphism one-space spelling** (the
sugar §1 licenses); the §6b population per step is the DIRECT-ctor re-key
set + attribute-read set (`.space` reads: [M] 161 Load-context sites in
`orpheus/`, mixed operator/field receivers — the per-step census
discriminates receivers at design time).

### F4 — §10: the R1 acceptance reader CANNOT SEE the success shape

[M] `_domain_annotation` (`test_monomorphic_leaves.py:680-705`) walks the
MRO for a PROPERTY (`vars(klass).get(prop_name)` → `fget.__annotations__`).
A mandatory dataclass FIELD never appears in `vars(klass)` → the reader
returns `("<not found>", "<not found>")`, which contains neither "None"
nor "Optional" → **the strict xfail xpasses VACUOUSLY** — identically for
a correct flip, for a field still annotated `FunctionSpace | None`, and
for `domain` being deleted outright. ⟹ the first flip commit EXTENDS the
reader (also read dataclass `__annotations__` through the MRO; treat
`<not found>` as FAIL), in the same commit as the SUT change, so the
instrument keeps discriminating. (§10's question — "what does the
instrument print on success?" — answered: PASS, but also PASS on two
failure shapes.) Pre-carve round: [M] independently confirmed and worse —
the reader also passes on `field(default=None)`, kw-only fields, and a
deleted attribute (plan §3.1). R6 RESOLVED from §1's own ruling: the base
is a dataclass ABC carrying domain+codomain as FIELDS, so the reader
extension (plan §3.2 G-B1 + meta-test G-B2) is MANDATORY.

### F5 — step 3's test-side denominator, now measured

[M] AST census over `tests/` (call nodes, direct + `from_*`):
**99 construction calls in 18 files** — ScatteringOperator 7 direct +
8 `from_solver_data`, LegendreMomentScattering 19, IsotropicScattering 23,
IsotropicN2N 18, N2NMomentOperator 4, FissionOperator 2 + 18
`from_solver_data`. Step 3 (≈71 S-family calls) and step 4 (≈20 F calls)
now carry their populations. ⚠ census 1's variable-call/registry-loop
check (the 2026-08-29 §6b spelling inventory) was run for C ONLY — step
3's opener census owes the same check for the S family.

### F6 — XD-2's count gate is §10-third-shape DESIGNED-RED at its stated denominator ⟶ one small user fork

[M] the multiplicity-2 literals at HEAD: `material_xs_field.py`
:1020/:1036/:1067, `isotropic_scattering.py:448`, `cp/solver.py`
:568/:629/:697/:784, `moc/core.py` :184/:316/:366 (+1 MC site per the
charter's own census) — **7–8 of ~12 live in cp/moc/mc**, solver families
in this ladder's untouched set. As chartered ("no production literal
outside the channel datum") the gate cannot reach green from the
SN/diffusion rebind alone. Fork:
- **(i) [lean]** the step includes the mechanical ~8-line re-point of
  cp/moc/mc literals to `N2NKernel.multiplicity` — datum CONSUMPTION
  (recycling the vocabulary, not improving those methods on their own
  terms — consistent with the sharpening-order law);
- **(ii)** the gate's denominator shrinks to SN+diffusion with the
  cp/moc/mc residue recorded in the gate's own docstring.

✅ **[R] RULED (user, 2026-08-30): option (i)** — the step includes the
mechanical re-point of every multiplicity literal to
`N2NKernel.multiplicity`; the gate's done-when is 0 production literals
outside `kernels.py`, at the full ~12-site denominator.

⚠ Pre-carve round corrections to this finding (plan §1.3/§1.4/§6.3):
- [M] the denominator is **14**, not ~12 — `material_xs_field.py:809/:862`
  were missing from every prior list; and two members evade a `2.0 *`
  filter (`moc/core.py:316` is an INTEGER `2 *`; `mc/solver.py:447` is
  `w *= 2.0`, an AugAssign) — the count gate's predicate is validated
  against all four spellings (plan §6.1).
- [M] the XD-9 pair's path: `orpheus/data/macro_xs/mixture.py:437-450`
  (the charter's `transport/mixture.py` path never existed at HEAD), and
  `condense` has a second (bilinear) branch obeying a different law.
- [M] §6d run on the re-point: cp/moc/mc → transport is layer-LEGAL
  (L3 → L2) but all three edges are 0 today and cost ~254 ms cold import
  each (eager `transport/__init__`); MC's spelling hoists to a module
  constant (single source — a literal with an exclusion would be the
  thirteenth home again) unless the user objects.
- R5 RESOLVED: the two new sites sit inside the arms O-6 absorbs at step
  3 — they die by absorption; the other 12 re-point; the count gate lands
  at END of step 3 and is green from landing day.
Also: XD-2's SUBJECT (the N2N truncations) lands at step 3; the record
placed the gate at step 4 — ⟹ **the count gate moves to step 3** so gate
and witness land together (§6c).

### F7 — XD-1's multiplicativity operand has NO SPELLING

[M] `kernels.py` API: `p0` + `truncated` only — no product/compose/
`__matmul__`. The gate's `bind(K₁K₂)` operand needs a decision:
a kernel composition verb (Funk–Hecke: composition of zonal kernels =
the ℓ-stacks' elementwise product — domain-true) vs gate-side array
construction from the moment stacks. Lean (defer-until-2): the GATE
builds the product kernel from the stacks; mint the verb only when a
second consumer appears. Named here as a test-architect design input.

### F8 — small corrections and step-note additions

- The kernel family is **three** (§0.4 corrected in place): `N2NKernel`
  ships with `multiplicity: ClassVar[int] = 2` (`kernels.py:224`) — the
  n2n datum is ALREADY collapsed; §4's O7 heal consumes it rather than
  minting it.
- Step 5's notes gain the [M] monkeypatch pair
  `tests/homogeneous/test_homogeneous.py:166/:174` (patches
  `MultiplicationOperator.apply` — an arm-deletion §6b member, census 1's
  own warning).
- Step 6's notes gain: (a) the `_R6_XFAIL` marker text cites
  `boundary.py:343` verbatim — re-point at the flip; (b) the L/B
  annotation flips owe a constructor-site census (who passes
  `space=None` to `StreamingOperator` + the three boundary classes)
  before any annotation turns non-Optional.
- §7.4's "(`.H` consumers unchanged)" — TRUE for the public spelling
  ([M] 149 production / 381 test `.H` mentions, no signature change);
  FALSE for the four private-importing test files — covered by F1(b).

### §12-bis — the pre-carve verification round (2026-08-30): plan, census, and ruling status

Both mandated pre-carve artifacts landed 2026-08-30 (both at HEAD
`2f44ed4e`, numbers re-measured there — cite the artifacts, never copy):
**`scratch/cs4c_verification_plan.md`** (test-architect; 17 gate rows,
5 batteries all ≤ 24 s, cumulative prediction ≈ 10 000/0/19sk/227des/56xf
at the coda) and **`scratch/cs4c_feeding_census.md`** (step 0; folded into
§§0–4, 8–9 above).

Further pre-carve facts the step designs consume (plan § refs):
- **Ledger coverage gap (plan §1.10):** the R1 instrument covers [M]
  4 of 8 classes carrying an Optional space annotation, in two spellings —
  `IsotropicScattering`, `IsotropicN2N`,
  `RadialCharacteristicBoundaryOperator`, `SNMaskedBoundaryOperator` flip
  UN-GATED; steps 3/6 either extend the ledger rows or record the gap in
  the flip commit.
- **`_R2_XFAIL`'s stated mechanism is present-tense-false for S** (plan
  §1.9): an anonymous `ScatteringOperator.H` now RAISES `MissingAdjoint`;
  only C degrades (via `is_metric_free_adjoint=True`) — the marker's
  evidence text rides the flip.
- **§6b members no AST call census can see** (plan §8.4): 3
  variable-mediated constructions (`test_isotropic_scattering.py`
  :124/:183/:220) + 2 monkeypatch surrogates (`test_homogeneous.py`
  :166/:174) + the set-literal registry pins (`test_kernels.py:660-671`).
- **The tier-2 equivalence-gate roster is larger than three** (plan §4):
  `StreamingOperator.pose` + the three kernel `from_mixture`s also owe
  G-C1 rows. R7 RESOLVED: yes — §3's ruled "every classmethod" discipline
  admits no datum-tier exemption.
- **dead_references baseline: [M] 0 dead / 52 checked** — every step's
  exit re-runs it against this zero.

**Ruling status:** R3 ✅ (singular-metric fixture ships — §7.4 rider),
R4 ✅ (legal; MC hoists to a module constant), R5 ✅ (absorption),
R6 ✅ (fields ⟹ reader extension), R7 ✅ (from_mixture owes gates).
**R1 ✅ [R] CONFIRMED (user, 2026-08-30: "Corrections proven by math are
always welcome")** — the tautological leg is dropped; Galerkin-on-faces is
the catcher; the identity stays as documented theorem.

**R2 ✅ [R] RULED (user, 2026-08-30): the NAMED-COMPOSITE direction — the
AdjointOperator survives as a FIRST-CLASS operator realizing the dagger
arrow.** ~~My own earlier proposal — `.H` returns the composed product,
the swap law a theorem of reversed factor inverses~~ ⛔ REFUTED at design
time: [M] `OperatorProduct.inverse()` is a wrap-delegate
(`InverseOperator`), demoting the #280 swap law from object identity; and
role passthrough dies (`test_psi_half_coupling:2895` pins
`a_ab.H.system_role`). The ruled design, with the user's own articulation:
*"during the construction of the AdjointOperator we literally construct an
operator with domain and codomain swapped — the domain of A becomes the
codomain of A†, and the codomain of A the domain of A† — and this exchange
is realized in code."* Maximum-leverage consequences, all step 1:
- apply routes through three first-class legs built at construction
  (`codomain.riesz_lower`, `A.dual()`, `domain.riesz_raise`) —
  bit-identical (G-A1);
- `A.H.H is A` becomes an OBJECT IDENTITY (adjoint() on the adjoint
  returns `inner`);
- `apply_transpose` becomes a THEOREM of the legs
  (`(A*)ᵀ = ♭_W ∘ A ∘ ♯_V`, metrics symmetric by admission) — closing
  **#375** (both defects: the dead-end AND the 0-witness raising stub);
- the swap law stays #280's object identity; `dual(dual(A)) = A` object
  identity too;
- the legs are public space verbs (`V.riesz_lower : V → V*`,
  `V.riesz_raise : V* → V`, PRIMAL-only — `DualSpace` REFUSES them, making
  the G² trap unspellable);
- the class is PROMOTED to public `AdjointOperator` in its own mechanical
  follow-up commit (3-search audit + dead_references before/after).

⚠ §7.4's "retire `_AdjointOperator`" is hereby SUPERSEDED-IN-MECHANISM:
the retirement target was the INLINE metric recipe in its apply, never the
arrow object; the chartered outcome (adjoint = algebra, legs
single-sourced and individually mutable, DualSpace consumed) lands whole,
plus the dagger-law identities a deletion could never have carried.

## 13. ⏸ COMPACTION POINT #1 (2026-08-30, written pre-compaction with full context; every anchor re-verified at HEAD `f167f3e5`)

### Phase → commit table

| act | commit(s) | state |
|---|---|---|
| design round + reviews (§§1–12) | `14201b48` → `2f44ed4e` | closed; F6 ruled (i); R1/R2 ruled |
| step 1: legs + dual + AdjointOperator re-expression | `68a9c9f3` | ✅ merged |
| step 1b: the public promotion (141 occ / 43 files) | `733d96f3` | ✅ merged |
| step 2: BoundOperator base + C rebind | `9fc8bf04` (the ff-merge tip) | ✅ merged |
| merge gate + reconciliation | `f167f3e5` (docs) | ✅ 13 trees rc=0 |

### The measured baseline every later step diffs against

**[M] 10006 / 0 / 19 sk / 227 des / 68 xf** at `9fc8bf04`, 13 trees rc=0
(driver `scratch/_cs4c12_gate_driver.sh`, ~65 min wall; log
`scratch/_cs4c12_fast_gate.log`). Tree costs: sn ~15 min, derivations
~38 min, everything else < 6 min. The 68 xf = 70 − R1-C − R2-C.

### Durable lessons of the execution phase (beyond §12's)

1. **Mechanic E** — a dataclass field CANNOT realize an abstract property:
   `dataclass()` deletes the no-default sentinel and re-runs
   `abc.update_abstractmethods`, re-abstracting it ([M] prototyped); the
   InitVar route is pyright-hostile (poisons every downstream read). The
   working spelling: fields for `__init__` generation + POST-CLASS
   write-once property injection + a second `update_abstractmethods`
   (`bound_operator.py` — reuse it verbatim for any later leaf family).
2. **Whole-file `str.replace(old, new, 1)` is NOT call-site surgery** —
   with the same `space=<expr>` text at neighboring calls, count-1
   replaces land on the FIRST occurrence, not the intended one ([M] 5 of
   16 landed wrong; caught by the red loop). Use position-anchored AST
   spans, edited in reverse source order.
3. **A prototype that "passed" may never have run** — the runner's
   relative path broke after `cd` and the pyright half of the output read
   as the whole result (the L61 family). Assert the prototype PRINTED its
   positive control before believing it.
4. **Never `git stash` under a running gate** — one stash/pop mid-suite
   risks false reds; baseline checks use `git show HEAD:file > tmp`.
5. `dead_references` caught the retirement's ONE doc casualty
   (`:attr:` → the retired field) that `-W` structurally cannot; run it
   before and after every step (baseline 0 dead / 52 checked).
6. The predicted-then-measured +1 was the layer gate's per-module
   parametrization gaining `bound_operator.py` — count corpus-coupled
   parametrizations (per-module, per-registry gates) in every step's
   predicted delta.

### ▶ RESUMES AT — step 3: S speaks kernel and frame

> ✅ **SUPERSEDED 2026-08-30 — step 3 MERGED @ `600c5c80`.** This block
> executed in full (§14 holds the design round's rulings, §14.8 the
> close-out). The LIVE resume surface is **§15 COMPACTION POINT #2**
> below (the step-4 block). Read this §13 block only as history.

**Goal (outcome, not mechanism):** the scattering binding is expressible
from representation-free data — a `ScatteringKernel` truncated to the
operator's order, faces minted from the HUB-interned frame, two mandatory
ends — with no `quadrature` field, no `mat_xs`-reading twin paths, and
the (n,2n) multiplicity owned once.

**Opening protocol (surgical posture): a DESIGN ROUND with the user
before any edit.** Its inputs, all verified at `f167f3e5`:

- **Ruled design (this record):** §2 (hub interns the frame; the
  classmethod tier mints the faces; `compare=False` provenance field
  rides), §4 (+ its census addenda — the satellite mint rate, the
  forward/adjoint twin routes, the iso asymmetric arrow), §6 as amended
  (Galerkin-on-faces is the ERR-039 catcher; the reverse-composite
  comparison is a THEOREM, never a gate; leg-(ii) control =
  `equispaced_equal`, chosen by measured bite), §12 F5/F6(i)/F7,
  §12-bis (analysis-verb on S/F only; ledger covers 4 of 8
  Optional-space classes — iso pair flips un-gated unless extended).
- **The verification plan:** `scratch/cs4c_verification_plan.md` §5
  (XD-1 legs), §6 (XD-2 at the 14-site denominator, option (i) ruled;
  MC hoists to a module constant), §8.4 (the S/F §6b population WITH its
  predicate limitation), §11.4 (battery B-3, ~24 s).
- **[M] anchors (re-verified this compaction):** the frame mint
  `ScatteringOperator.frame` `scattering.py:486+` (reads the mandatory
  `quadrature` field — the mint step 3 re-routes through
  `axis.generator_as(Quadrature, consumer=…)`); the space-anonymous iso
  mint `scattering.py:755` (`isotropic_kernel` at `:727`); the
  space-less refusal `_interior_space` `:515+` (dissolves at the flip);
  kernels at `kernels.py:101` (Scattering), `:200` (N2N,
  `multiplicity` ClassVar), `:259` (Fission); the two pins that MUST be
  re-keyed in-step: `tests/transport/test_kernels.py:698` (`domain is
  None` on the iso pair) and `tests/numerics/test_axis_generator.py:379`
  (the AST call-site set gains the frame-mint member).
- **Census obligations before designing (§12 F5):** the S-family
  variable-call/registry-loop check (census 1 ran it for C ONLY); the
  `.space`/attribute-read sweep for S receivers; [M] the known test
  population is 99 S/F calls in 18 files (~71 S-side).
- **Fork F-E is DECIDED IN THIS STEP (or step 4), by ruling:** iso-pair
  operator-CLASS fate — named ℓ=0 binding recipes vs truncated
  `ScatteringOperator` bindings — now with the census's evidence (the
  iso pair is the hottest operator, 4816 applies, an asymmetric arrow).
- **Parked forks:** C.H-as-object-identity (§8 row 2 note); F-F (CS2
  residue stays step 6).

**Standing constraints (unchanged):** Host → `.venv/bin/python`;
canonical runner `-O -m pytest -p no:randomly -m "not slow" -q` SERIAL;
branch + ff-merge + the 13-tree driver before merging; predicted-then-
measured vs 10006/0/19/227/68; main agent writes, user steers
(AskUserQuestion checkpoints); test-architect plan EXISTS (do not
re-dispatch for step 3 — extend by SendMessage if needed); never
`git add -A`; commit messages via heredoc `-F`; no source edits under
running gates; sphinx `-W` + `dead_references` at every step exit.

---

## 14. Step-3 design round (2026-08-30) — the S rebind's rulings

Held per §13's opening protocol. The user reshaped the presented design at
four points; all evidence re-measured at `c69c2856` before the rulings.

### 14.1 ✅ [R] N2N is its OWN first-class bound operator — extracted from S in step 3

**The user's argument (the ruling's substance):** (n,2n) is scattering-like
(it carries its own anisotropy in principle) AND production-like (it carries
multiplicity) — so its bundling is CONTEXT-dependent (with S for anisotropy
studies, with F for production accounting), and a context-dependent bundling
must not be decided at the operator level. A bundled operator may be minted
later if attractive enough to justify it.

**Structural echo [M]:** `N2NMomentOperator`'s own docstring already insists
the channel is "a *multiplication* channel, NOT scattering … its own named
operator, summed with Λ" — the tree half-believed it (summed in moment
space, hidden inside S). **The within-group algebra becomes literal:
`(L + C) − S − N2N`**, and bundling is a solver-side `OperatorSum` grouping.

**Production blast radius [M] (small):** the solver's legacy helpers already
call `add_iso_source`/`add_n2n_source` SEPARATELY
(`sn/solver.py:2284/:2321/:2535`); `coupled_system.py:575` shares
`S.isotropic_kernel` (re-points to the solver-assembled K_iso); the solve
paths get n2n only via `_assemble_per_ordinate_source`'s K_iso fold; the
adjoint via `full_scatter_kernel = conjugate(Λ + N2N)`.

**The shared primitive this creates (defer-until-2 satisfied on day 1):**
S's P0 per-ordinate path and N2N's composite binding are BOTH "an isotropic
energy endomorphism lifted to the angular composite" —
`(1/W)·E ∘ K ∘ ∫dΩ`, mathematically the frame's ℓ=0 conjugation. ONE lift
primitive, two consumers (S internally for P0; the solver for N2N); its
`apply` keeps the reaction-rate fast path (algebra eager, performance lazy)
with a gate pinning fast ≡ conjugated. Exact class-vs-frame-mint spelling
decided at execution.

**Priced consequences (accepted in the ruling):** (a) every within-group
composition site gains the explicit `− N2N` term (solver, coupled system,
adjoint — certification suite re-pins); (b) the summation order changes
(`(isoS+isoN2N)/W + aniso` → separately-applied arrays summed), so
bit-identity may break at ULP level on `Sig2 ≠ 0` fixtures — re-baseline
per the principled-equivalence criteria, measured at execution.

⟹ **§4 is AMENDED in place:** S's exact ctor retains ONLY the scattering
kernel field + faces; no `n2n` field. `residual_part` loses its n2n
zeroing (the foldable/residual split becomes a pure scattering-kernel
split). XD-2 unchanged (denominator 14, gate end of step 3) — the
multiplicity's operator-side owner is now the N2N family.

### 14.2 ✅ [R] F-E RESOLVED BY STRUCTURE (§9 row closed)

Under 14.1, `IsotropicScattering`/`IsotropicN2N` survive **by
construction** as the energy-space bindings that the composite operators
lift and diffusion/homogeneous consume directly (zero angular content —
no frame, no faces, scalar ends). The "dissolve into truncated
ScatteringOperator bindings" option is dead: the collapse is at the
KERNEL (one datum); the named recipes are the scalar bindings of it.

### 14.3 ✅ [R] The material-field shape: generic base + channel subclasses

`MaterialField[K]` (frozen, generic) owns the pairing — per-material
kernel map + the cell-to-material layout — and the ONE gathered per-cell
`(ng,ng)`-apply einsum primitive. `ScatteringMaterialField` /
`N2NMaterialField` subclass it with domain-named verbs (the 8 O-6 arms +
`add_n2n_to_group_rate`, einsums ported VERBATIM for per-arm
bit-identity). The `cells_by_material` where-cache moves to
`MaterialMesh` (mesh owns machinery), shared by every field — no
per-field recomputation. Step 4 adds `FissionMaterialField`.
⚠ Recorded for the record: the charter's O-6 phrasing ("arms → the bound
operators") and §4's ("the kernel's array verbs") named different homes;
this ruling settles it kernel-field-side.

### 14.4 ✅ [R] The frame outputs the SPACE; creation is standardized through the CS5 channel

- `_moment_space_on` → **public `moment_space_on(angular_space)`** — the
  single source of the moment-space derivation; fields are minted OUTSIDE
  from it (a Frame induces spaces + operators per the stage-2 generator
  discipline; a field is data and mints where values exist). The F-2
  mesh-keyed twin (`_space_for_mesh_and_L`, deferred into O-3) is checked
  in-step: pure re-point ⟹ take it now; scheme-read load-bearing ⟹ stays
  with O-3, reason recorded.
- **The blessed frame chain, one spelling for every consumer** (S tier-2,
  F, windowing, gates): `domain space → interior → angular axis →
  axis.generator_as(Quadrature, consumer=…) → hub → frame`. The
  same-metric guarantee is structural (the Quadrature is the single
  source; no copy exists to drift).
- **Hub mechanics [M]:** `frame.table` is a per-instance `cached_property`
  (`numerics/frame.py:166`), so "one frame per (axis content, L)" must
  hand out the SAME object. numerics cannot mint transport's
  `HarmonicFrame` (layer contract) ⟹ the interning home is
  transport-side: one classmethod (working name
  `HarmonicFrame.for_space(angular_space, L)`) running the chain,
  interned keyed `(quadrature, L)`. `Quadrature.angular_frame(L)` stays
  the pure numerics mint.

### 14.5 R13 scope split — step 3 advances O-6 ONLY

Recovered reasoning (posing-filtration charter §5, ratified 2026-08-25):
`MaterialXSField` is `MaterialMesh`'s XS **facade** (`from_mesh` adds no
data); dissolution map: content → `Materials`; expansion → field-minting
path; typed mints → `CrossSectionField`s (STAY); coarsening →
Materials/Mixture morphisms; **the 8 `apply_*` arms (~400 lines) → CS4c
(O-6, "may phase with or before F-1")**. Full F-1 blast radius [M] 18
production + 32 test files. **Evaluated 2026-08-30: direction confirmed;
step 3 takes the arms + dense caches; the facade's other four halves stay
with F-1.** `mat_xs` leaves the CTOR now but remains the legal tier-2
argument (`from_solver_data(mat_xs=…)`) until F-1 — the posing arc's R1
shape exactly.

### 14.6 The step-3 opener census (obligations §13/§12 F5 — RUN, positive control passed)

Script `scratchpad/s_family_census.py` (session-local; predicate
limitation stated: B1 is file-local single-assignment dataflow):
- **A3 internal S→S constructions: 9** — Λ ×4, N2NMoment ×1, iso pair ×1
  (`scattering.py:755`), and SELF at `foldable_part:1051` /
  `residual_part:1092` (the 2026-08-29 "own internal call" member class);
- **A1 registry family:** confirmed = the `factory` parametrize family in
  `test_isotropic_scattering.py` (3 sites) only;
- **A4 monkeypatch: 1** — `test_sn_adjoint_certification.py:255` patches
  `ScatteringOperator.apply_transpose` (signature unchanged — survives);
- **B1 attribute pins:** `S.sig_s` ×25 / `S.sig2` ×10 / `S.Y` ×2, all in
  `test_scattering_operator.py` (re-key to kernel-field reads);
- **Production external reads on solver-held S: exactly 3** —
  `flux_analysis` (`sn/solver.py:898`), `scattering_order` (`:919`),
  `isotropic_kernel` (`coupled_system.py:575`); string-form reads: 0.

### 14.7 The execution shape (sub-steps; one merge unit)

- **3a machinery (additive):** `MaterialMesh.cells_by_material`;
  `transport/material_field.py` (base + 2 subclasses, verbatim einsums +
  per-arm bit-identity gates); public `moment_space_on`;
  `HarmonicFrame.for_space` hub; the iso-lift primitive.
- **3b rebind (the core):** S ctor flip (mechanic E; retained = scattering
  field + 2 faces + ends; `scattering_order` derived from the field;
  composites cached at construction — the 911/solve satellite fix;
  provenance frame via `flux_analysis.frame`, zero extra state;
  `total_weight` named scalar) + Λ/N2NMoment/iso-pair rebinds (mandatory
  ends, per-END admission) + the N2N extraction (solver compositions,
  K_iso re-point, adjoint) + all call sites + ledger flips R1[S]/R2[S]
  (+ `_R2_XFAIL` constant deletion in the flip commit) + §8.3 pin re-keys.
- **3c retirements:** O-6 arms + dense caches + the `sig_s`/`sig2`/
  `sig_s0`/`cells_by_mat` transients (**#306 closes**) +
  `_interior_space` refusal dissolves; `dead_references` sweep.
- **3d gates:** XD-1 (G-D1/2/3 per plan §5), XD-2 (14 re-points + count
  gate), G-C1 equivalence rows, battery B-3.
- Exit: 13-tree driver vs **10006/0/19/227/68**, sphinx `-W`,
  `dead_references` 0/52, predicted-then-measured delta.


### §14.8 — step-3 CLOSE-OUT (2026-08-30, merged @ `600c5c80`)

**The measured baseline every later step diffs against:**
[M] **10079 / 0 / 19 sk / 66 xf**, 13 trees rc=0
(driver `scratch/_cs4c3_gate_driver.sh`, log `scratch/_cs4c3_fast_gate.log`,
~66 min wall; sn 14:13, derivations 36:01). Per-tree reconciliation
PRE-REGISTERED before readout (`scratch/cs4c_step3_predicted_deltas.md`)
and closed with zero unexplained ±1: transport +61 (material_field 28
net of the retired transitional battery, tightness 17, tier-2 G-C1 7,
census 2, hub gates 7), sn +9 (ledger flips +2, test_n2n_operator +10,
in-place rewrites −3), root +3 (the layer gate's per-module rows for
material_field/n2n/_per_ordinate — the P7 corpus-coupled mechanism,
caught by pre-registration this time), all other trees 0. xf 68 → 66
(the two ledger flips). sphinx -W clean; dead_references [M] 0 / 52;
`npx pyright orpheus/` 0; verification matrix +3 documented labels
(568 → 571, exactly the archivist's prediction).

**Durable lessons of the step (beyond §13's):**
1. ⭐ **A labelled equation is an API and survives every symbol sweep**
   (coding-standards' clause, fired again): `sn-scattering-adjoint-kernel`
   / `-transpose` still STATED the (n,2n) summand after the in-commit
   `dead_references` sweep came back clean — no symbol appears in a
   `.. math::` body. The archivist's corpus pass is the instrument that
   catches it; budget one per step that moves an equation-level claim.
2. **B3.5's premise refuted through the shipped faces** ([M] recorded in
   `test_binding_tightness.py`'s docstring): gauss_legendre(L) is NOT
   blind on zonal multiplicativity in the shipped-face spelling (rel
   3.0/4.3 at L=2/3) — the plan §1.2 blindness (≤5.9e-16) was the raw-
   table probe's property. §4's verify rule worked: the disagreement is
   recorded, not resolved by picking a side.
3. **The Mode-11 sentinels caught every re-route** (three of them: the
   LD cells-index mutation, the diffusion stencil transpose swap, the
   homogeneous K_iso identity) by reddening when their patched surface
   went dead — the §8.4 monkeypatch-surrogate member class, working as
   the enumeration instrument the 2026-08-24 §6b row said it is.
4. **A bundled issue must be read before its title's fragment is acted
   on**: #306 was closed on its `sig_s/sig2` title fragment and
   reopened within minutes — items 1/3/5 live on. The docstring's
   "#306 tracks this" was a pointer to a 5-item estate issue.
5. **The seedless gains unpack was load-bearing in 9 test files** —
   `S, B = system.explicit_gains` and `explicit_gains[1]` both broke
   LOUD (the gauge test's own comment predicted it: "a splat would
   silently mis-bind if the splitting ever grew a third gain"); the one
   SILENT candidate (`[1]` as "the boundary") was caught by reading,
   not by a red.
6. **My own `git stash -- <path>` reflex stashed an uncommitted edit**
   under no gate at all (recovered by `pop` immediately) — the L37
   family extends to pathspec'd stash used as a "temporary revert";
   `git show HEAD:file` was already the rule and the reflex bypassed it.

**Steps 4–6 + coda remain per §8; step 4 (F rebind) opens with its own
design round; its census obligations: the ~20 F test calls (F5), the
XD-9 condensation gate (plan §7) with its activation precondition, the
FissionKernel first-consumer wiring, and F-E's N2N-side residue is
CLOSED (the extraction resolved it).**


---

## 15. ⏸ COMPACTION POINT #2 (2026-08-30, written pre-compaction with full context; every anchor re-verified at merged HEAD `5b9681e9`)

### Phase → commit table (step 3, all ✅ MERGED to main, ff-only)

| act | commit | state |
|---|---|---|
| step-3 design round (§14 rulings: N2N first-class; MaterialField[K]; frame→space + hub) | `b2ad262a` | closed |
| 3a machinery (kernel fields, hub interning, `for_space`, `moment_space_on`) | `c0e904ea` | ✅ |
| 3b-A (Λ + N2NMoment rebind) | `8f376135` | ✅ |
| 3b (S ctor flip + N2N extraction + solver composition + ~30 test files + ledger flips) | `b435431c` | ✅ |
| 3c (9 facade arms retire; XD-2 re-points + census green; dead_references 9→0) | `81e9e7e1` | ✅ |
| 3d (battery B-3; B3.5 refuted through the shipped faces) | `92dcc30f` | ✅ |
| corpus pass (archivist 14 pages) + code-side prose | `600c5c80` (the merge tip) | ✅ |
| close-out (§14.8) | `5b9681e9` | ✅ |

### The measured baseline every later step diffs against

[M] **10079 / 0 / 19 sk / 66 xf**, 13 trees rc=0
(driver `scratch/_cs4c3_gate_driver.sh`, log `scratch/_cs4c3_fast_gate.log`,
~66 min; sn 14:13, derivations 36:01, all else < 6 min). Per-tree deltas
were PRE-REGISTERED before readout (`scratch/cs4c_step3_predicted_deltas.md`)
and reconciled with zero unexplained ±1. sphinx `-W` 0; `dead_references`
[M] 0/52; `npx pyright orpheus/` 0; verification matrix 571 documented
labels (the +3 exactly as predicted).

### Durable lessons — §13's six + §14.8's six stand; nothing new since.

### ▶ RESUMES AT — step 4: F speaks kernel and faces

> ✅ **EXECUTED 2026-08-31 — this block was the resume surface step 4 ran on;
> §16 holds the design round and §16.8 the close-out + the NEW baseline
> (10106/0/19/66). Superseded in place per §3.**

**Goal (outcome, not mechanism):** the fission binding is expressible
from representation-free data — the `FissionKernel` factor pair
(χ, νΣf) gains its FIRST production consumer; the composite realization
carries space-supplied faces (the F4-addendum composite); the k-outer F
binds the BULK space honestly (§3's ⛔ ASPIRATIONAL row becomes true);
the χ↔νΣf-coupled condensation is gated (XD-9).

**Opening protocol (surgical posture): a DESIGN ROUND with the user
before any edit.** Its inputs, all verified at `5b9681e9`:

- **Ruled design (this record):** §3 (the three-tier discipline; the
  per-family row: *"the k-outer F binds the bulk space — ⛔ ASPIRATIONAL:
  [M] the tree binds `sn_mesh.full_field_space` while feeding bare
  `(ng,*spatial)`; step 4 makes the bulk binding true"* — the solver
  mint is now at `sn/solver.py:1424`), §4 (kernel collapse; F-E is
  CLOSED by the §14.1 extraction — no iso-pair residue remains), §6
  (NO kernel dagger — F's adjoint is the operator-level factor swap by
  theorem; `FissionKernel(chi=νΣf, nu_sig_f=χ)` is REFUSED by its own
  simplex guard, re-verified by the archivist), §14 (the N2N extraction
  is the fresh PRECEDENT for an isotropic emission operator: the shared
  combine `_per_ordinate.assemble_per_ordinate_isotropic`, the
  carrier-arm shape, the tier-2 space-derivation idiom — read
  `orpheus/transport/operators/n2n.py` FIRST as the model, per §1's
  precedent rule: check each adjective against the file).
- **The verification plan** (`scratch/cs4c_verification_plan.md`): §7
  (XD-9 condensation gate G-F1 — the law + THREE measured negative
  controls: average-vs-marginalize 6.42e-1 / 1.69e0 / 7.09e-2, and the
  ACTIVATION PRECONDITION asserted per B4.5 — a 1-fine-per-coarse
  target must red the precondition, §10's designed-green hazard); §4
  G-C1 (F's `from_solver_data` + `FissionKernel.from_mixture` owe
  rows — the kernel row EXISTS since step 3,
  `tests/transport/test_tier2_equivalence_s_family.py`); §11.5 battery
  B-4 (6 arms, measured bands).
- **[M] anchors (re-verified this compaction):** `fission.py:149`
  (class; fields `mat_xs` + `space: FunctionSpace` — ⚠ ALREADY
  mandatory since CS4a K2, so step 4's flip is the BASE swap to
  `BoundOperator` kw-only ends via mechanic E, not an Optional flip;
  the ledger F rows flipped long ago), `:259` `from_solver_data`,
  `:281` `kernel` (a `TensorProductOperator` dyad); `kernels.py:259`
  `FissionKernel` (`:315` `from_mixture`, `:324` `dyad`; consumer-status
  paragraph says "the first production consumer is CS4c's rebind" —
  step 4 makes that true and MUST update that paragraph);
  `mixture.py:313` `condense` (χ↔νΣf coupling; the BILINEAR second
  branch at ~`:363-386` obeys a different law — XD-9's corrected path,
  plan §1.4).
- **Census obligations before designing (§12 F5 + the 2026-08-29 §6b
  inventory):** the F-family variable-call/registry-loop check (the
  step-3 opener script `s_family_census.py` covered the FIVE S classes
  only — extend it to `FissionOperator`/`FissionKernel`); the
  `.space`/attribute-read sweep for F receivers; [M] the known test
  population is 2 direct + 18 `from_solver_data` F calls (plan §8.4);
  production: 4 `from_solver_data` sites.
- **Known step-4 obligations from the plan:** XD-9's gate lands WITH
  its activation precondition asserted; G-F2 (the dyad factor-order
  pin); battery B-4; the `fission.py` docstring's K2/R2 narrative
  updates to the BoundOperator story; the per-END admission migrates to
  the base helper (F currently calls `assert_energy_extent_conforms`
  directly at `fission.py:205`).
- **Parked forks (unchanged):** C.H-as-object-identity (§8 row 2 note);
  F-F (CS2 residue stays step 6). **#306 items 1/3/5 remain open**
  (item 2's bless-vs-retire now spans THREE zero-boundary emitters:
  F, S, N2N — a step-4-adjacent adjudication candidate for the design
  round). **#424** (RST nested-markup corpus sweep) is independent.

**Standing constraints (unchanged from §13 except the baseline):**
Host → `.venv/bin/python`; canonical runner
`-O -m pytest -p no:randomly -m "not slow" -q` SERIAL; branch +
ff-merge + the 13-tree driver before merging; predicted-then-measured
vs **10079/0/19/66** with per-tree PRE-REGISTRATION (the §14.8
discipline that closed step 3's readout in minutes); main agent writes,
user steers (AskUserQuestion checkpoints); test-architect plan EXISTS
(do not re-dispatch — extend by SendMessage); never `git add -A`; never
`git stash` — including `git stash -- <path>` (§14.8 lesson 6; use
`git show HEAD:file`); commit messages via heredoc `-F`; no source
edits under running gates; sphinx `-W` + `dead_references` at every
step exit; an archivist corpus pass whenever a step moves an
equation-level claim (§14.8 lesson 1).

---

## 16. Step-4 design round — rulings (2026-08-30, user + main agent; all [R] unless marked)

### 16.1 [R] Machinery-first: F is designed for the pencil/resolvent/adjoint triple, never for today's consumers

**Goal (domain terms).** The fission binding must simultaneously satisfy,
with NO hand-rolling: (a) the `GeneralizedEigenPencil` — (A, F) as two
peers of one Bound-operator discipline on one space; (b) resolvent
shifts — the α-eigenvalue moves F to the loss side, so F must be able to
enter an `OperatorSum` with `(L+C−S−N₂ₙ−B)` under the ends guard; (c) the
adjoint falling out of the machinery by theorem. **Today's consumer
architecture is transitory** (the k-outer's bare-array power iteration will
be redesigned later): consumers get ADAPTERS or minimal changes; they get
no design authority over F. (User, verbatim in substance: "The
FissionOperator must be thought of as part of the Bound operator machinery
first… design it thinking with forward thinking.")

### 16.2 [R] The three-piece family, with Riesz maximal use

Fission is the **ℓ=0, rank-1-in-energy degenerate of the scattering
binding**: S = R·Λ·M/W over ℓ ≤ L; F = R₀·(χ⊗νΣf)·M₀/W. Same faces, same
interned hub, same metric.

1. **`FissionMaterialField`** (`MaterialField[FissionKernel]`): per-material
   VALIDATED kernels × layout (χ-simplex per material by construction —
   Pattern 4) + gathered cellwise (χ, νΣf). `FissionKernel`'s first
   production consumer (kernels.py:259's consumer-status paragraph updates).
2. **`IsotropicFission`** (energy binding, iso-family shape beside
   `IsotropicScattering`/`IsotropicN2N`): retains the material field; the
   rank-1 dyad `kernel` (TensorProductOperator) + `production_rate` (the
   ReactionRateFunctional diagnostic) cached at construction (the S
   satellite-fix ruling — read-through/depletion semantics deliberately
   dropped; docstrings update). The ONE arithmetic home of χ(νΣf·φ).
   Honest scalar ends.
3. **`FissionOperator`** (composite binding): retains `energy:
   IsotropicFission` + the two ℓ=0 faces (`flux_analysis`,
   `source_reconstruction` from the blessed interned chain) + mandatory
   composite ends (mechanic E; per-END admission via the base helper —
   the direct `assert_energy_extent_conforms` call at fission.py:205
   migrates). Forward = the S-idiom ℓ=0 route (analysis → scalar accessor
   [bit-exact = integrate_angular] → dyad → the shared
   `assemble_per_ordinate_isotropic` combine). **Euclidean transpose =
   factor reversal of the cached face product** — fission.py's inline
   `np.multiply.outer(w,·)/W` transpose arithmetic dies.

**Riesz maximal use (the user's directive, resolved):** the family
hand-rolls NOTHING on the adjoint path. `.H` composes
`♯_V ∘ Fᵀ ∘ ♭_W` from the spaces' own first-class legs (CS4c R2,
`operator.py:1556` — the transport operators become the legs' first
production consumers through the daggered posing); M† = R/W stays XD-1's
gated theorem, never spelled arithmetic; `production_rate` stays a typed
Functional (a DualSpace-typed co-vector is defer-until-2).

**Consumer re-points (minimal, no design authority):** SN k-outer
(`solver.py:1424` instance + `:1536` apply), homogeneous (`:251`),
diffusion (`:247`) → `IsotropicFission` with honest scalar ends;
`RadialCharacteristicEmission` (`solver.py:2939`) consumes the energy
binding's dyad; the eigen-M posing (`:2901`) keeps `FissionOperator`
(composite). [M] step-0 census: the k-outer instance's ONLY carrier ever
is the bare `(ng,*spatial)` ndarray (118 calls), so the re-point changes
no production arithmetic.

### 16.3 [R] Naming now; the factory rethink runs in parallel

`IsotropicFission` picked (family greppability; the docstring states the
redundancy honestly — fission has no anisotropic sibling; the prefix names
the FAMILY role). **AND the user chartered an adversarial rethink** of the
whole iso-trio toward "a Factory object that returns the appropriate
operators depending on the problem (domain, codomain, anisotropy order…)"
— justifications offered as tentative: SN+DSA needs iso+aniso flavours of
one channel; per-level sensors. Explicitly requested: "skepticism and
adversarial consideration to create tension to cause the right
architecture to emerge." A cross-domain-attacker probe is running (memo →
`scratch/cs4c_step4_factory_probe.md`); its finding is RULED SEPARATELY —
it governs future SELECTION machinery, not this step's exact ctors (§3's
ruling: the classmethod tier is deliberately evolvable; the exact ctor is
the commitment).

### 16.4 [R] N2N harmonized in step 4

The moment F's faces shape lands, `N2NOperator`'s bare-`weights` +
hand-spelled transpose formula is a twin spelling of the iso lift
(Cardinal Rule 2). In-step: faces replace the `weights` field
(`total_weight` derives from `flux_analysis.frame.measure.weights.sum()`
as S does), transpose by reversal, the 10-row gate re-keys.

### 16.5 [R] #306 item 2 deferred to step 5

The zero-boundary emission is a property of the composite ARMS; step 5
owns the arms. Dated note goes on the issue.

### 16.6 [M] Opener census (f_family_census.py, positive control PASS)

Population matches plan §8.4 exactly: 4 production `from_solver_data`
(sn :1424/:2901, homogeneous :251, diffusion :247), 18 test
`from_solver_data`, 2 direct-ctor tests + 1 direct FissionKernel ctor,
5 `from_mixture` test sites. **Zero registry/variable-call members, zero
patch surrogates** (predicate limitation: file-local single-assignment
dataflow, as step 3). B1 attribute pins: `.chi` ×3 / `.nu_sig_f` ×5 (all
test_kernels.py — kernel fields, survive), `.kernel` ×6 (1 production =
:2939). String-form: `hasattr(·,'kernel')` in the integral-kernel
category tests (protocol probes — survive). ⚠ Staleness found:
fission.py's head algebra still spells `(L+C−S−B)` — zero N₂ₙ mentions in
the file (step 3's corpus sweep missed it; dies in this step's rewrite).

### 16.7 Execution shape (one merge unit, sub-steps in order)

- **4a machinery (additive):** `FissionMaterialField` + `IsotropicFission`
  + transitional bit-identity gates (old-F-scalar-arms ≡ new, per arm).
- **4b the rebind:** `FissionOperator` → composite binding (ends, faces,
  cached product, reversal transpose); consumer re-points; docstring
  estate (head algebra, K2/R2 narrative, kernels.py consumer paragraph).
- **4c N2N harmonization** (§16.4).
- **4d gates:** XD-9/G-F1 (+3 measured controls + asserted activation
  precondition per B4.5), G-F2 collapse row, G-C1 rows, battery B-4;
  ledger check (F rows long since flipped — no marker action expected).
- Exit: 13-tree gate vs **10079/0/19/66** with per-tree PRE-REGISTRATION;
  sphinx `-W` + `dead_references`; archivist corpus pass (this step moves
  equation-level claims: the fission adjoint/binding story on the theory
  pages).

### 16.8 Step-4 close-out (2026-08-31) — ✅ ALL SUB-STEPS LANDED, gate reconciled exact

| act | commit |
|---|---|
| step-4 design round (§16.1–§16.7) | `8e26b8f2` (branch open) |
| 4a machinery (FissionMaterialField + IsotropicFission; on-sight foldable-prose fixes + `foldable_sig_s` retired) | `f4caf04a` |
| 4b the rebind (F = the frame's ℓ=0 conjugation; 4 consumers honest; ~17 test files re-keyed) | `75500cd9` |
| 4c N2N harmonization (weights → frame; product transpose) | `9061637b` |
| 4d gates (G-F1 + controls + asserted precondition; G-F2; battery B-4; G-C1 angular-F row) | `fadad026` + `1b083b06` |
| corpus pass (archivist, 19 pages; labels 571→574 documented; n2n docstring upgraded to its measured claim) + #425 filed | `4e46dbb9` |
| exit-gate census catch fixed | `b68e0f56` |

**The measured baseline every later step diffs against:** [M]
**10106 / 0 / 19 sk / 66 xf, 13 trees rc=0** (driver
`scratch/_cs4c4_gate_driver.sh`, log `scratch/_cs4c4_fast_gate.log`,
~61 min; sn 15:29, derivations 37:53). Per-tree deltas PRE-REGISTERED
(`scratch/cs4c_step4_predicted_deltas.md`) and reconciled with ZERO
unexplained rows — the single first-run red WAS the 10106th predicted
row: the declared-consumer census catching 4c's dissolved
`generator_as` site (a registry-tier §6b member the sweep missed; the
gate caught it at the exit, cost one 6-min tree re-run). sphinx `-W` 0;
`dead_references` 0/52; `npx pyright orpheus/` 0; verification matrix
574 documented labels ([M] +3 exactly as the archivist declared, zero
test delta — a documented label removes itself from the orphan gate).

**Durable lessons (step 4):**

1. **A Mode-11 sentinel's red on a re-route is the sentinel WORKING** —
   the crosscheck registry-wrap reddened the moment the k-outer left
   `FissionOperator`, and was re-keyed onto the new live arm in the
   same commit with the catch recorded in its docstring. Design
   sentinels so the re-route reddens them; then re-key, never delete.
2. **A TRUE claim can still be a defect: the under-claim.** n2n.py's
   "principled-equivalent, gated at tolerance" was honest and
   UNDERSTATED its own result — [M] the N₂ₙ harmonization is
   bit-identical (1000/1000 draws, a THEOREM of ℓ=0), where F's is
   genuinely 4–5 ULP. Two docstrings that read alike for different-kind
   changes invite a gate relaxation on the stronger one. The archivist
   measured both; the docstrings now differ.
3. **A rebind can WIDEN a guard's reach as a side effect** — the
   wrong-ng bind on an axes-less SN composite was "declared inert"
   (no EnergyAxis to check) until from_solver_data began deriving the
   scalar ends from the interior's axes; the inertness row FLIPPED from
   constructs-to-refuses (test_kernels row 3, banner in place). After a
   re-wire, re-derive what every adjacent guard now reaches — the
   mirror of the silent-demotion clause, in the favorable direction,
   and it still costs a test re-key.
4. **The §6b registry tier includes DECLARED-CONSUMER sets in tests** —
   4c dissolved a call site and the census gate's declaration survived
   it (caught at the exit gate). A census/registry test that ENUMERATES
   call sites is a consumer of every one of them.
5. **The B1 receiver-name predicate missed `op.chi`-style generic
   locals** (the fission-flavored receiver filter; ~20 sites in two
   adjoint/crosscheck modules) — caught by the red loop, loud, zero
   silent damage. A retired attribute's test consumers spell the
   receiver as `op`/`self`, not by the domain name.

**Parked for the user's ruling (design round, next checkpoint):** the
factory probe's verdict (`scratch/cs4c_step4_factory_probe.md` +
cross-domain-attacker memory `iso_family_factory_refutation.md`):
factory-as-dispatcher REFUTED on three structural grounds (no section
over Space alone — the key must be Space × Role, and Role is the §5
posing concept; zero runtime branches exist to collapse; rank splits
the trio 2+1). The honest redirect: (a) collapse
IsotropicScattering+IsotropicN2N into ONE energy operator whose channel
lives in its field's type (fission held out by rank), (b) the lift
functor energy→angular as the named object. Both are SELECTION/step-5+
machinery, deliberately not ruled inside step 4.

**Residue:** #425 (37 four-term algebra sites, declared at the chapter
root); #306 items 1/3/5 + item 2 deferred to step 5 (dated note on the
issue); the parked forks C.H-as-object-identity and F-F unchanged.

---

## 17. ⏸ COMPACTION POINT #3 (2026-08-31, written pre-compaction with full context; every anchor re-verified at merged HEAD `b91246e8`)

**Steps 1–4 are ✅ MERGED; the phase→commit tables live in §15 (steps
1–3) and §16.8 (step 4) — cited, not copied (§9). The measured baseline
every later step diffs against:** [M] **10106 / 0 / 19 sk / 66 xf, 13
trees rc=0** (§16.8; driver `scratch/_cs4c4_gate_driver.sh`, log
`scratch/_cs4c4_fast_gate.log`, ~61 min; per-tree pre-registration
reconciled with zero unexplained rows).

**Durable lessons — §13's six + §14.8's six + §16.8's five stand;
nothing new since §16.8.**

### ▶ RESUMES AT — step 5: each binding's action is selected by its construction

**Goal (outcome, not mechanism).** Per-call carrier dispatch stops
being the bindings' shape: an operator constructed with its ends acts
through the ONE body those ends select, dead arms die, and the
bare-ndarray hatches close via explicit **solver-side adapters at the
seams** (§5's ruling — operators end whole; typing the k-outer iterate
and every consumer reshape is the deferred consumers campaign, NOT this
step). The #205/#276 keep-rulings are re-litigated with FRESH census
evidence, not inherited.

**Opening protocol (surgical posture): a DESIGN ROUND with the user
before any edit.** The round OWES three rulings §16.8 parked:

1. **The factory redirect** (`scratch/cs4c_step4_factory_probe.md` +
   the cross-domain-attacker memory `iso_family_factory_refutation.md`):
   factory-as-dispatcher REFUTED (the key must be Space × ROLE — the §5
   posing concept; zero runtime branches exist to collapse; rank splits
   the trio 2+1). The redirect on the table: (a) collapse
   IsotropicScattering + IsotropicN2N into ONE energy operator whose
   channel lives in its field's type, fission held out by rank; (b) the
   lift functor energy→angular as the named object. Both are SELECTION
   machinery — exactly step 5's subject.
2. **#306 item 2** — bless-vs-retire of the implicit-zero boundary
   emissions on the THREE composite arms (F, S, N2N; dated deferral
   note on the issue): the arms' fate IS this step's subject.
3. **#205's ScalarFlux keep-ruling** — S's arm docstring
   (`scattering.py:1120`, "Deliberately retained — a named-future-
   consumer surface") records the keep call; re-adjudicate against the
   fresh census.

**⚠ THE OPENER OBLIGATION (the §2 shelf-life rule, already bitten
once):** the step-0 feeding census (`scratch/cs4c_feeding_census.md`)
is a **2026-08-30 PRE-step-3 capture** and its step-5 "consequences"
section is now partially FALSE — its item 2 (*"FissionOperator.apply
[ndarray] is the SN k-outer's only F arm; deleting it breaks every SN
eigenvalue solve"*) died at step 4 ([M] `FissionOperator` carries NO
ndarray or ScalarFlux arm; the k-outer feeds `IsotropicFission`), its
F-D space-table F rows are superseded, and every `file:line` in it is
stale. **Re-run the per-arm traffic capture at the opener** (the
census's own §0 records the method) before designing the arm list; do
the same for battery B-5's sibling-realizer anchors
(`diffusion/solver.py:289`, `cp/solver.py:544`, `moc/core.py:108` —
census-time lines, unverified at HEAD). Still-credible census facts to
RE-CHECK rather than trust: the MultiplicationOperator[ndarray]
audience (2 calls, homogeneous-only) and the iso pair's CHART-DEPENDENT
bare leg (sphere/cylinder feed both carriers from one instance — any
arm decision needs the chart in its denominator).

**[M] anchors (re-verified this compaction):** the four "until step 5"
markers — `scattering.py:420` (S's dispatch note), `fission.py:378` +
`n2n.py:56/:202` (the mirrored carrier-arm banners); the k-outer seam
`numerics/eigenvalue.py:374` (`power_iteration`) →
`sn/solver.py:1527` (`compute_fission_source`); the verification plan's
**B-5** (§11.6: adapter-identity positive control; adapter drops the
conversion; a stale `_registry_type_names` set) with its ⚠ that the
battery scope must include `tests/sn/solve/` — measure the cost first.

**§6b members of ANY arm deletion (named now so no sweep misses
them):** the G2.8 arm-survival matrix + the registry-keyset gate
(`test_kernels.py` — B5.3's own catchers, they re-key WITH the arms);
the C6 static pins + dispatch parity
(`test_operators_apply_typed.py`); declared-consumer/census sets in
tests (§16.8 lesson 4 — a census gate is a consumer of every site it
enumerates); duck-typed stubs and `getattr`-string reads per the
2026-08-24/29 §6b inventory.

**Precedents (§1's check-each-adjective rule applies):** the step-4
`FissionOperator`/`N2NOperator` shapes are the models for
construction-selected bodies (ends → faces → one action); read the
FILES, not this record's description of them.

**Ledger:** §8.1 schedules NO flip rows for step 5 (steps 1+2/3/6
only); the 66 xf carries through unless an arm deletion touches a
documented row — reconcile any xf delta, never absorb it.

**Parked forks (unchanged):** C.H-as-object-identity (§8 row 2 note);
F-F (CS2 residue, step 6). **Open issues:** #306 items 1/3/5 (+ item 2
= this round's ruling 2); #425 (the 37 four-term algebra doc sites,
declared at the chapter root); #424 (RST nested-markup sweep,
independent).

**Standing constraints (unchanged from §15 except the baseline):**
Host → `.venv/bin/python`; canonical runner
`-O -m pytest -p no:randomly -m "not slow" -q` SERIAL; branch +
ff-merge + the 13-tree driver before merging; predicted-then-measured
vs **10106/0/19/66** with per-tree PRE-REGISTRATION; main agent writes,
user steers (AskUserQuestion checkpoints); test-architect plan EXISTS
(extend, don't re-dispatch); never `git add -A`; never `git stash` —
including `git stash -- <path>`; commit messages via heredoc `-F`; no
source edits under running gates; sphinx `-W` + `dead_references` at
every step exit; an archivist corpus pass whenever a step moves an
equation-level claim; in-process mutation batteries run as
subprocess-scoped pytest plugins (the 4d pattern — crash-safe by
construction).

---

## 18. ⏸ COMPACTION POINT #4 (2026-08-31, written pre-compaction with full context; every anchor re-verified at merged HEAD `01ed1d79`)

**Steps 1–4 remain ✅ MERGED (tables at §15 and §16.8, cited not copied).
Step 5 OPENED this session and did NOT execute: its design round ran, its
first ruling was REFUTED BY DATA, and the session was redirected by the
user into the data investigation that produced the refutation.** Nothing
in §17's ▶ block was invalidated except its ruling 1, which is now
ANSWERED — see §18.3.

### 18.1 What landed (all on `main`, pushed)

| commit | what |
|---|---|
| `ea06fbbd` | `BE009.GXS` joins `micro_xs/` via LFS (133-byte pointer / 22 558 446 B) |
| `6906f2a2` | the (n,2n) isotropy claim corrected across 11 files — docs, production docstrings, test rationales |
| `0a13b0ea`, `131dc08f`, `01ed1d79` | agent memory (archivist L-078, literature-researcher + the new ENDF/GENDF format reference) |
| `7433507d` | **the (n,2n) yield double-count fixed at ingest — `Closes #427`** |

### 18.2 The step-5 OPENER ran; its two products are on disk

⚠ **Do not re-run either. Read them.**

* **The per-arm traffic census** (§17's mandated opener obligation, discharged):
  `scratch/cs4c_step5_feeding_census.md`, `[M]` at HEAD `c6dc40c3`, all four
  controls passing, denominator **34 surfaces** (18 dispatch arms + 16 plain
  verbs), 34/34 hand-activated, 13/13 headline numbers bit-identical
  instrumented vs control. **FIRED 20 / NOT-RUN 14.** Two ⭐⭐ refutations of
  the step-0 census that a step-5 arm deletion MUST honour:
  - ⛔ **`IsotropicScattering.apply_transpose` and `IsotropicN2N.apply_transpose`
    are NOT dead.** Step 0 called them dead-with-no-consumer; the consumer is
    the ray-system adjoint (`radial_characteristic.py:1536`), invisible to
    step 0 because its workload carried only *slab* adjoints. One added
    scenario takes them **0 → 985 calls each**. Deleting on the old evidence
    would have removed a live path.
  - ⛔ **All five `FissionOperator` forward arms are at ZERO**; step 4 took it
    off the forward path entirely (the k-outer now feeds `IsotropicFission`,
    139 calls, bare `ndarray`, bound to `mesh.bulk_space`). Step 0's *"deleting
    the ndarray arm breaks every SN eigenvalue solve"* is **void** — that arm
    no longer exists.
  - `S.apply[ScalarFlux]` still measures **0** (#205's keep-ruling
    corroborated); SN's `C` is **140 instances, 140 silent**.
* **The owed edits the census could not apply** (its brief forbade touching
  tracked files): `scratch/cs4c_step5_OWED_skill_and_memory.md` — two proposed
  `vv-principles` #29 sharpenings, a third report-hygiene candidate, and the
  `qa` memory delta. **NOT APPLIED. Adjudicate them before they rot.**

### 18.3 ⛔ RULING 1 IS ANSWERED — the F ≡ N2N collapse is REFUTED, by data

The design round put three options to the user. The user declined to rule and
directed a **data stress test** first (*"we need to figure out from ENDF and
GENDF specifications if there is anisotropy data for N2N or if it is genuinely
isotropic at the data level… stress-test the N2N design based on data"*), and
added `BE009.GXS` to make the test sharp. The test settled it.

**The structural finding that raised the question** (`[M]` by AST at
`c6dc40c3`, still true): `FissionOperator` and `N2NOperator` share 14 members,
**8 identical modulo the class/energy names**; `apply_transpose` and
`total_weight` are **identical in code** (docstrings differ); all four dispatch
arms are identical — the entire difference across the dispatch surface is **two
string literals**. `IsotropicScattering` vs `IsotropicN2N`: 5 of 9 shared
members identical, the other 4 differing by exactly one expression each, every
one a **verb name on the datum** (`add_p0_source`/`add_emission`,
`…_transpose`, `kernel.p0`/`emission_matrix()`, `.truncated(0)`).

**The data refutes the collapse.** Full memo: `scratch/n2n_data_stress_test.md`;
format half: `scratch/n2n_data_format_spec.md`; issue: **#426**.

| | scattering | **(n,2n)** | fission |
|---|---|---|---|
| Legendre orders stored (GENDF NL) | 7 | **7** | **1** |
| transfer form | full `g'→g` | **full `g'→g`** | **separable χ(g)·νΣf(g')** |
| rank-1 separable | no | **no** (rank 50, 58 % err) | **yes**, by construction |
| σ₀ self-shielding (NZ) | 6–10 | 1 | 1 |

⟹ **`FissionOperator ≡ N2NOperator` is a coincidence of today's ISOTROPIC
MODEL, not a fact about the physics.** Collapsing them would freeze a
truncation into the type system — anti-pattern #18's inversion (a type that
cannot express correct physics), worst exactly on the material the test was
run against. ⛔ **Do not collapse F with N2N.**

✅ **The redirect's OTHER half is CORROBORATED**, and for a deeper reason than
the factory probe's rank argument: `IsotropicScattering` + `IsotropicN2N`
folding into one energy operator (channel in the field's type) matches the
data — both are anisotropic, non-separable, matrix-valued channels; fission is
the outlier. ⟹ **the honest long-run structure is N2N alongside SCATTERING**,
with `Isotropic…` in the N2N names recording a TRUNCATION rather than a
property. If the anisotropy is restored (#426), N2N wants exactly scattering's
machinery — which `N2NMomentOperator` already half-anticipates.

⭐ And the `_per_ordinate.py` docstring's own **defer-until-2 trigger has
FIRED**: it enumerates two consumers and says *"when a third isotropic lifted
channel appears, this is the seed of the generic lift operator"* — `[M]` step 4
made fission the third (`fission.py:414`, `:436`). The lift functor is live as
a design candidate; the *collapse of the composite classes* is not.

### 18.4 The open issues this investigation derived

* **#426 — the P0 truncation** (OPEN). GENDF stores NL=7 for MT=16; `gendf.py`
  keeps `sig2_data[(0, 0)]`. Be-9: all 8195 transfer entries non-zero at every
  ℓ=1…6, μ̄ = +0.278 **over its own 50 open groups**, and (n,2n) supplies a
  median **62 % of the P0** and **45 % of the P1** emission source there
  (Be-9 has no inelastic MF=6 section, so `elastic + 2·(n,2n)` is its COMPLETE
  fast emission source). Body corrected in place 2026-08-31 — see §18.5.
  ⚠ **The missing experiment, and it should precede any carve:** no transport
  solve was run, so those are **cross-section shares, not an error in any `k`
  or flux**. A Be-reflected fixture solved with and without the ℓ≥1 source is
  what decides whether #426 is a carve or a documented approximation.
  `BE009` is now in the store, so the fixture is buildable.
  ⚠ **The generalisation limit** (`scratch/n2n_data_format_spec.md`): NJOY
  builds LAB moments with `P_ℓ` at the lab cosine against a CM distribution, so
  ℓ≥1 is non-zero even for an isotropic CM emission. **8 of 9 (n,2n)-bearing
  nuclides are LCT=2**; **⁹Be alone is LCT=1**, the one clean case. This does
  not change the truncation verdict (lab moments are real and the transport
  equation is solved in the lab frame) but it bounds *"the evaluations carry
  anisotropy data"* to Be-9.
* **#427 — the yield double-count** (✅ CLOSED @ `7433507d`). GENDF MF=6 holds
  `σ·y·f`; MT=16's yield is 2; every consumer assumed the reaction matrix and
  applied the ×2 itself, so removal was doubled and emission quadrupled. Fixed
  by renormalising the transfer rows onto MF=3 — **never a literal `/ 2`**, so
  `N2NKernel.multiplicity` keeps its single home in a package `orpheus.data`
  sits below. ⚠ **The HDF5 store is GITIGNORED and `load_isotope` reads it**, so
  the fix reaches a checkout only when `convert_gxs_to_hdf5.py` is re-run
  there; a stale store serves the old convention with no signal. Regenerated
  locally (13 files, all reading `1.000000000` against MF=3).
* **#428 — `cross_section_data.rst`'s "Reactions Not Included"** (OPEN). Says
  ORPHEUS does not extract (n,2n); `[M]` `gendf.py` does, and
  `sn/solver.py:1713` carries `- emission_n2n.sum()` in the k balance. The page
  contradicts itself at `:172` and `:1001`. Small doc edit **gated behind a
  four-solver check** — ERR-023 says MC ignores `Sig2`, so the corrected text
  must be per-solver.

### 18.5 ⛔ A number I published and had to retract — read before quoting §18.4

I reported (n,2n) as *"~3× more forward-peaked than elastic"*, citing
μ̄ = **+0.278** against **+0.094**. **The two were summed over DIFFERENT
WINDOWS** — `+0.094` is elastic over all 421 live groups (dominated by the
low-energy s-wave region, per-group `0.0746`); `+0.278` is (n,2n) over its 50.
`[M]` over the SAME 50 groups elastic reads **+0.4264**, i.e. `1.53×` the
(n,2n) value: **the comparison inverts.** Caught by the archivist reproducing
the figure before publishing it, so the wrong contrast never reached the
corpus; #426's body and the memo carry the correction in place.

⭐ The transferable half, and why it is not merely §2's quantifier clause
again: **I had run a rigorous positive control** — elastic σ₁/σ₀ reproduces the
analytic s-wave `2/(3A) = 0.074615` to six significant figures — and that
control validated the **INSTRUMENT**, saying nothing about whether two readings
from it are **COMMENSURABLE**. A control on the instrument lends unearned
authority to a ratio formed from its outputs. ⟹ when a claim is a RATIO of two
measurements, the check that matters is *do these share a population?*, and it
is a different check from validating the instrument. The same caution now
applies to the `‖P1‖∞/‖P0‖∞` figures (0.690 vs 0.075): honest per channel,
**not usable as a cross-channel ratio**.

### 18.6 ▶ RESUMES AT — user's ruling, 2026-08-31

*"merge it and let's fix the issues that this investigation derived before
doing any additional design work"*, then *"we will take care of those after
compaction"*. ⟹ **#426 and #428 come BEFORE step 5 resumes.** Recommended
order, with the reason:

1. **#426's missing measurement first** — the Be-reflected with/without-ℓ≥1
   solve. It is a MEASUREMENT, not design work, and it is what chooses #426's
   scope (carve vs documented approximation). Everything else about #426 is
   unanswerable until it exists.
2. **#428's four-solver check** — independent of 1, smaller, dispatchable.
3. **Then step 5**, whose ▶ block at **§17 remains the resume surface** with
   ruling 1 struck (§18.3) and the census obligation discharged (§18.2).
   Rulings 2 (#306 item 2) and 3 (#205's ScalarFlux keep) are still OWED and
   now have fresh census evidence to be adjudicated against.

### 18.7 Standing constraints (unchanged except the baseline)

Host → `.venv/bin/python`; canonical runner
`-O -m pytest -p no:randomly -m "not slow" -q` SERIAL; branch + ff-merge + the
13-tree driver before merging; per-tree PRE-REGISTRATION with
predicted-then-measured; main agent writes, user steers (AskUserQuestion
checkpoints); test-architect plan EXISTS (extend, don't re-dispatch); never
`git add -A`; never `git stash`; commit messages via `git commit -F -` with a
**quoted** heredoc; no source edits under running gates; sphinx `-W` +
`dead_references` at every step exit; archivist corpus pass whenever a step
moves an equation-level claim; mutation batteries as subprocess-scoped pytest
plugins.

⚠ **The 13-tree baseline is UNMEASURED at this HEAD.** The last full gate is
step 4's **10106 / 0 / 19 sk / 66 xf, 13 trees rc=0** (§16.8). This session
added **10** rows (`tests/data/test_n2n_yield_convention.py`) and reworded
prose only elsewhere, so the prediction is **10116** — *predicted, not run*.
Targeted evidence only: `tests/data` + both recipe consumers **279 passed /
0 failed**; the two archivist-touched test files **53 passed**; `sphinx -W`
clean; `dead_references` 0 dead / 52 checked. **Run the full gate before the
next merge that claims a baseline.**

### 18.8 ⏸ Re-anchored 2026-09-03 — the baseline is MEASURED; the symmetry review landed in between

Between §18.7's `01ed1d79` and this note, the user chartered and closed a fresh-context
review of the symmetry & quotient machine — **#434, CLOSED**, four carves R1 `f9d3b15b`
→ R4 `13423a59` → R2 `27703297` → R3 `fe219888`, all ff-merged to `main` at `de3cba4d`
(plan + compaction record: `.claude/plans/symmetry_machine_reads_like_the_math.md`).
✅ **The 13-tree baseline at `main` `de3cba4d` is `[M]` 11007 collected, 13 of 13 rc=0**
(R3's gate, `scratch/_r3_full_gate.log`; per tree: numerics 3255, transport 707,
geometry 732, data 237, homogeneous 50, diffusion 113, cp 141, moc 121, mc 41,
cross_method 81, sn 3439, derivations 1661, root+harness 429). §18.7's "UNMEASURED"
warning is repealed by this measurement; its 10116 prediction was never run as such
(the +10 rows are inside the 11007).

**What #434 changed that step 5 / #426 / #428 may touch** (§1 existence-checks at
resume): `SubgroupOfO3.is_invariant` is GONE — invariance is asked of the measure
(`measure.is_invariant_under(G)` and four sibling verbs; `orpheus/numerics/invariance.py`);
the registry's `AngularSymmetry` is `(spent, unspent, owed)`; a `Quotient` carries its lift
as `lift_coordinates` / `lift_codomain`; `manifold.spent_group` is retired. Nothing in
`orpheus/transport/` or the CS ladder's operators was touched.

**Resume order unchanged** — §18.6: #426's Be-reflected with/without-ℓ≥1 measurement →
#428's four-solver check → step 5 at §17's ▶ block.

✅ **2026-09-03, step 1 of §18.6 DONE and it chose the carve**: `[M]` restoring the ℓ = 1 (n,2n) moment
moves k on a Be-reflected U-235 slab by −414 (Δk·10⁵) / −346 (Δρ·10⁵) — reproduced independently;
#426 is a DEFECT. Its remedy has its own plan of record, **`.claude/plans/n2n_anisotropy_kept.md`**
(three forks ruled the same day: lossless ingest for every channel; one transfer family with the yield
a kernel datum; `Transfer*` names the kernel tier, role names stay on the operator tier). §18.6 step 2
(#428's four-solver check) is `[M]` done the same day — all six families handle (n,2n), MC included;
#428 CLOSED @ `f1422f24`. **#426 step 1 (the lossless data layer) LANDED @ `f96de34c`** and merged;
step 2 (the transfer family) is next on a fresh branch — its design and compaction record are the plan's §4/§5.
**Step 5 waits behind the #426 carve** (its N2N/S design is exactly what the carve settles). The two rulings owed from before the
review (#429 umbrella-or-close; resume Campaign 2) still stand as owed.

✅ **2026-09-04 — #426 steps 2 + 3 LANDED (branch `feature/n2n-transfer-family`: `7b44ee68` → `1a3b78ec` → `f52877db` →
`9e6adf3c`, `Closes #426`; merge hash = git).** The transfer family ships: `TransferKernel(moments, multiplicity)`,
`TransferMaterialField`, `LegendreMomentTransfer`, the cores `IsotropicTransfer` / `TransferOperator`, and the four roles as
TWO CLASS CONSTANTS each (`channel`, `isotropic_binding`) with no code — the AST gate refuses any method. `N2NOperator` is
minted at the solve's order on the SAME interned frame as `S`; the Be-reflected slab reads 1.0911996566537725 with the
shipped library (ERR-082's catcher). **Step 5 is UNBLOCKED**: its N2N/S question is settled — `S` and `N₂ₙ` are one binding,
`TransferOperator`, and the §18 ⛔ ({S, N2N} | {F}) stands realised in the tree (`FissionOperator` keeps its own shape; #450
proposes it adopt the transfer family's derived-energy-binding form). ⚠ Step 5's design text in §17 predates the family:
re-derive its N2N/S rows against `orpheus/transport/operators/transfer.py` at the opener (plan-authoring §7), and note the
core's order property is `legendre_order`, the fold family lives on the core generic in the yield, and `is_isotropic` is the
predicate the anisotropic arms branch on.

---

## 19. Step-5 design round (2026-09-04) — each binding acts through the body its ends select

**Opener (§7 reconciliation, every anchor `[M]` at `main` `f90f7914`, tree clean;
the 13-tree baseline is #426's exit gate, `[M]` **11142 collected, 13 of 13 rc=0**,
`scratch/_426_step2_full_gate.log` + the sn re-run).**

- §17's four "until step 5" markers: `scattering.py:420` → **`transfer.py:384`**
  (*"the carrier dispatch serves it until step 5's arm deletion"*);
  `fission.py:378` → **`fission.py:398`** (*"carrier arms mirror N2N's, until
  step 5"*); `n2n.py:56/:202` are **GONE** — `n2n.py` is a 112-line role since
  #426 step 2 (`1a3b78ec`); the banners died with the carve. The k-outer seam:
  `numerics/eigenvalue.py:374` (`power_iteration`) → `:420` →
  `sn/solver.py:1528` (`compute_fission_source`) → `IsotropicFission` bound at
  `sn/solver.py:1431` on `mesh.bulk_space`, bare `(ng, *spatial)` in and out.
- §17's ruling 1 is **DONE by #426 step 2**: `IsotropicScattering` +
  `IsotropicN2N` are thin roles of ONE `IsotropicTransfer`; `{S, N2N} | {F}`
  is realised as two families (§18.3 ⛔ stands in the tree). What survives of
  the item is its half (b) — *the lift energy→angular as the named object* —
  and it is this round's R-1 (§19.3).
- §18.8's "two rulings owed": **#429 is CLOSED** (`[M]` `gh`: 2026-09-03T17:30Z,
  COMPLETED, with the #434 review) — the umbrella-or-close question is moot;
  the memory index carried the stale OPEN and was corrected. ONE ruling set is
  owed: this round's.
- §18.2's OWED edits: A1 + A2 **LANDED** at `8707c53a` (vv-principles #29's
  two sharpenings); A3 (a finding names ONE `file:line` ⟹ it is repaired at
  that `file:line` only) is NOT landed — adjudicated: general, lands in this
  step's docs commit as a report-hygiene sharpening. The qa memory delta is the
  qa agent's own (it consolidates its memory at each dispatch; `[M]` its
  `lessons.md` A15/L-077 slots were taken by a different lesson — the delta's
  numbering is stale, its content is in #29).
- ⚠ The step-5 census `scratch/cs4c_step5_feeding_census.md` (§18.2) is
  `[M]` at `c6dc40c3` — **five carves old** (#434 R1/R4/R2/R3 + #426 steps
  1–3), and #426 step 2 re-shaped exactly the surfaces it counts (S and N2N
  now share ONE dispatcher; `.energy` → `.isotropic_energy`;
  `N2NMomentOperator` retired; ~~N2N's ℓ ≥ 1 arms fire at `scattering_order ≥ 1`~~ — ⛔ REFUTED 2026-09-04 by the re-run (§19.11 D3/D3b): the ℓ ≥ 1 BODY runs iff the bound (n,2n) DATUM carries a nonzero moment above ℓ = 0; on every P0-only `Sig2` fixture the pad supplies exact zeros, `is_isotropic` reads values, and the body is skipped at `scattering_order` 1 AND 2 — a property of the datum, not of the order).
  Its tree half is void per §2's shelf-life rule. **Re-run dispatched
  2026-09-04 (`qa`, read-only, `scratch/cs4c_step5b_feeding_census.md`)**,
  keeping the 13 scenarios verbatim + 3 new (P2 slab, P1 slab adjoint with a
  live (n,2n) stack, the Be-reflected fixture if cheap). Its delta table is
  §19.11 (appended when it lands). Nothing below is priced on the old numbers
  except where marked *[M c6dc40c3]*.

### 19.1 The goal, in the domain's terms (§5), and its done-when

A bound operator is an ARROW between two declared spaces (§1); its action is a
function of its ends. Today the action is resolved per CALL on the operand's
Python type. `[M]` by AST over `orpheus/transport/operators/` at `f90f7914`
(`scratch`-free, in-process; positive control: `TransferOperator._apply_impl`
found):

| surface | count | where |
|---|---:|---|
| `singledispatchmethod` tables | **3** (13 arms) | `transfer.py:1080` (5: object/FullField/ScalarFlux/AngularFlux/HarmonicMomentFlux — shared by S and N2N), `fission.py:401` (5), `multiplication_operator.py:410` (3) |
| carrier `isinstance` inside `apply`/`apply_transpose`/`solve` | **12** | `isotropic_transfer.py:328/:342/:620/:637`, `fission.py:508`, `transfer.py:1306` (all `FullField`); `multiplication_operator.py:454/:463/:530/:537` (the angular/scalar FAMILY parse ×2 verbs); `transfer.py:307/:346` (`LegendreMomentTransfer`'s typed moment arm) |
| `getattr(x, "values", x)` fall-throughs (typed in, bare out — the F10 asymmetric arrow the G2.8 matrix RECORDS as shipped) | **7 sites, 3 spellings** | `isotropic_transfer.py:120` (`_values_of`, used ×4), `fission.py:516`, `transfer.py:1318` |
| "a BULK operator emits the zero trace" | **9 spellings** | `fission.py:421/:514`, `isotropic_transfer.py:157`, `multiplication_operator.py:461/:472/:535/:544`, `transfer.py:1145/:1316` |
| a registered arm that RE-ENTERS the dispatcher on `psi.interior` (the census's bodies-not-arms tell) | **2** | `fission.py:416`, `transfer.py:1138` |

**Goal.** An operator constructed with its ends acts through the ONE body
those ends select: composite-bound → `FullField → FullField`; plain-bound →
the space's array; the interior body of an angular binding is fixed at
construction by WHICH END of its retained analysis face the domain's interior
is. Dead arms die; the typed/bare fall-throughs die; the zero-trace emission
is spelled once.

**Done-when (checkable):**
1. an AST gate over `orpheus/transport/operators/` finds **0**
   `singledispatchmethod` and **0** carrier `isinstance` inside any
   `apply`/`apply_transpose`/`solve` (first red = the tree today: 3 + 12;
   the LMT exemption is R-5's ruling);
2. every public verb's signature names exactly the carrier its ends name —
   the G2.8 fence re-keyed from *"12 cells, no arm dies"* to **18 cells,
   `(operator, binding kind, carrier) → outcome`**, where every off-binding
   carrier is a typed refusal naming the operator (G1.3's contract kept);
3. `grep -c "BoundarySourceSink.zeros\|BoundaryFlux.zeros"` over the package
   reads **1** (the base's spelling);
4. bit-identical: the 13-scenario census's headline numbers; the ERR-082
   record (`test_be_reflected_n2n_anisotropy.py`, 1.0911996566537725 at
   rtol 1e-8 / the two bit-identical controls); the ledger's 66 xf; the
   adjoint records (the transpose is still ONE conjugated product).

### 19.2 The mechanism — selection from the retained FACE's two ends

For an angular binding the composite domain's interior is one of exactly two
spaces, both already retained on the operator: the flux-analysis face's
**domain** (the per-ordinate angular space — every 1-D, curvilinear, Krylov
and un-windowed iterate) or its **codomain** (the harmonic-moment space — the
2-D Cartesian windowed SI iterate; `[M]` `sn/solver.py:737` *"un-windowed SI,
both Krylov paths"*: windowing is SI-only on 2-D Cartesian, minted by
`_maybe_window` at `:874–900`). `__post_init__` today admits the interior only
as the face's domain; it will admit EITHER end and **select the interior body
by which one it is**:

| interior is … | ℓ = 0 half | ℓ ≥ 1 half (absent when `is_isotropic`) |
|---|---|---|
| `flux_analysis.domain` (angular) | `integrate_angular()` — the reaction-rate fast path | `kernel = R Λ M` (unchanged) |
| `flux_analysis.codomain` (moments) | `scalar_flux(space=…)` — the ℓ = 0 slot | `source_reconstruction ∘ Λ` — M already applied |

The `is_isotropic` binding selects NO ℓ ≥ 1 summand at construction (today the
predicate is evaluated per call at three sites: `build_aniso_source`, the
windowed arm, `kernel`). The public verbs are `apply(FullField) → FullField`
and `apply_transpose(FullField) → FullField`; the interior bodies are private.
The re-entrant `FullField` arm (`transfer.py:1138`, `fission.py:416`) dies —
the composite action IS the lifted interior body (§19.5).

**The 2-D windowed path becomes a DECLARED non-endomorphism.** The census
measured S and N2N bound `(angular, angular)` and fed a moment composite
(*[M c6dc40c3]* §3: operand `FullFieldSpace#fb9da0f7…` ≠ bound `#c65fb053…`).
The windowed driver already mints that operand space — `BulkAnalysisOperator`
poses `FullFieldSpace.from_blocks(S.flux_analysis.codomain, trace)` as its
codomain (`sn/operators/windowing.py:97–108`). Step 5 binds the windowed
siblings on it: a tier-2 verb on the core, `on_moment_domain()` (name open),
= `type(self)(transfer, faces, domain=<that composite>, codomain=self.codomain)`;
`_windowed_cold_start` / `_maybe_window` hand the windowed SI its
moment-domain `S_w`, `N2N_w`. The `A_AA` OperatorSum guard is untouched:
`[M]` the windowed loop consumes `explicit_gains` (`numerics/iteration.py:698`),
never `A_AA`. The moment-domain sibling's transpose is `(R Λ)ᵀ = Λᵀ Rᵀ` —
built whole and reciprocity-gated although no 2-D adjoint exists today
(machinery-first, §16.1).

### 19.3 R-1 — the lift as the shared BASE: `{S, N2N} | {F}` without spelling 19.2 twice

`[M]` at HEAD `FissionOperator` and the transfer core still share, member for
member: `_interior_space`, `total_weight`, `frame`-via-faces vs `frame` held
(the one asymmetry #450 names), the composite wrap, the two ℓ = 0 interior
arms (`fission.py:424–457` ≡ `transfer.py:1174–1264` with Λ removed), the
`apply_transpose` composite branch. Writing 19.2's selection into both is
Cardinal Rule 2's stop sign — the census's *"8 of 14 members identical"*
(§18.3) is STILL the shape, one class over.

**Proposed [P]:** a base **`AngularLift(BoundOperator["FullField"])`** — *the
angular binding of an energy channel: the frame's ℓ = 0 conjugation
`R₀ E A₀ / W` on the composite, with its interior body selected by the ends* —
retaining `(flux_analysis, source_reconstruction, ends)` and DERIVING
`isotropic_energy` from its datum through the role's `isotropic_binding`
(#450's shape, applied to F as well: `FissionOperator` holds
`fission: FissionMaterialField`, derives `IsotropicFission`; its runtime shape
guard and its held `frame` field die — the frame rides the faces, as on the
transfer core). Then:

* `FissionOperator(AngularLift)` = the lift alone (+ the `kernel` /
  `production_rate` accessors and `full_fission_kernel`);
* `TransferOperator(AngularLift)` = the lift + the ℓ ≥ 1 redistribution
  (`kernel`, `full_transfer_kernel`, `build_aniso_source`, the fold family);
  the roles `ScatteringOperator` / `N2NOperator` stay two class constants and
  no code (the AST role gate is unchanged: `_all_subclasses(TransferOperator)`).

This IS §17 item 1(b) — *"the lift functor energy→angular as the named
object"* — realised as a base rather than a free function:
`_per_ordinate.py`'s own *"when a third isotropic lifted channel appears, this
is the seed of the generic lift operator"* fired at step 4 (`[M]` its
docstring lists F as the third), and `assemble_per_ordinate_isotropic` is the
seed it named; it folds into the base. **NOT a collapse of F with N2N**
(§18.3 ⛔): the datum types, the moment factors (`LegendreMomentTransfer` vs
`FissionMomentOperator`) and every ℓ ≥ 1 arithmetic stay on their own
subclass; the base carries only what the DATA says the two share — the ℓ = 0
conjugation. **Name [P] `AngularLift`**; alternatives: `IsotropicLift`
(refuted — a base whose transfer subclass is anisotropic), `LiftedEnergyOperator`.

**Refuted candidate — composition** (`TransferOperator` HAS a lift and a
redistribution summand; `apply = lift + aniso`): the transpose is ONE
conjugated product `frame.conjugate(Λ_{ℓ≥0})ᵀ` whose factor reversal the
adjoint records pin — a sum of two transposes changes the summation order on
the adjoint path; and both summands need the same faces and interior, so the
sharing would be by hand-off rather than by structure.

### 19.4 R-4 — the energy bindings and the multiplier: the ends select the CARRIER

**Plain-bound** (a scalar `FunctionSpace`: the SN k-outer's `IsotropicFission`
on `bulk_space`; the two `isotropic_energy` bindings K_iso sums; the
homogeneous solver's C / IsoS / IsoN2N / IsoF on Energy ⊗ point): the carrier
is the space's ARRAY — `apply(ndarray) → ndarray`, `apply_transpose` likewise
— the numerics tier's contract (`as_matrix`, `OperatorSum`,
`power_iteration`'s protocol vector, `RadialCharacteristicEmission`) and
nothing else. A `ScalarFlux` is REFUSED there: the typed wrap is the business
of the consumer that holds the space — the LIFT (which is exactly what
`_assemble_per_ordinate_source` does today at its one site:
`ScalarSourceSink(values=isotropic_energy.apply(phi), space=phi.space)`).

**Composite-bound** (the diffusion solver's four on the scalar composite): see
R-2 — under §3's tier-3 ruling the energy bindings stop binding the composite
at all; diffusion binds them on the scalar bulk and LIFTS at the assembly.
`[M]` `_assemble_iso_energy_operator` feeds bare interior-shaped impulses to a
composite-bound op today (`isotropic_transfer.py:236`) — with the lift the
assembly mode moves to where the composite layout is known (the lift embeds
the bulk matrix). ⚠ `[M]` the assembly mode HAS production consumers at the NUMERICS tier —
`numerics/flat_operator.py:168` (`FlattenedOperator` assembles its inner when
`is_assemblable`: the diffusion resolvent's path), `numerics/operator.py:1135`
(`as_matrix` takes the assembled fast path) and `coupled_system.py:1246` — a
first census that excluded `orpheus/numerics/` read **0** and was wrong by
exactly the tier the filter dropped (§2's FILTER clause, live); so the lift
verb carries `assemble()` too: the plain-bound energy binding assembles its
bulk block-diagonal, the lift embeds it into the composite layout (assembly
composes like apply). `[M]` the diffusion resolvent IS on that path in
production: `LeakageOperator.is_assemblable` and
`DiffusionBoundaryOperator.is_assemblable` are both `True`
(`diffusion/operators.py:440/:625`), so `MatrixInverseOperator(FlattenedOperator(loss))`
assembles `L + C − (IsoS + IsoN2N) − B` at every `DiffusionSolver` construction —
the census's DIFF bare-ndarray calls on the iso pair are the assembly's impulse
probes. `[M]` 2 test files exercise the family's assembly directly
(`tests/diffusion/test_operators.py`, `tests/homogeneous/test_operator_spaces.py`).

This retires `_values_of` and both inline `getattr` fall-throughs, the F10
asymmetric arrow, and every carrier `isinstance` on the four energy verbs.
`MultiplicationOperator`: composite-bound → the lift of its engine;
plain-bound → the engine on the array; its angular/scalar family parse
(`:454/:463/:530/:537`) dies with R-2's role partners. Its ndarray arm is the
#276 keep-ruling honoured structurally (the plain binding's carrier).

⚠ The **refuted alternative** is typed carriers on plain spaces everywhere
(`ScalarFlux` in, `ScalarSourceSink` out, `as_matrix` minting typed basis
vectors through the space's zeros seam) — §5's deferral: *typing the k-outer
iterate and every consumer reshape is the consumers campaign*. Not this step.

### 19.5 R-2 — #306 item 2: BLESS the zero-trace emission, spelled ONCE as the tier-3 lift verb

`[M]` 9 spellings (§19.1's table). §3 tier 3 (✅ [R] 2026-08-30): *"a
bulk-born operator enters a full-space composition through the
restriction/extension family — extension by zero on the trace — owned by the
ASSEMBLY at one site, not by a FullField arm on every class."* Read with §3's
own per-family answer — the within-group members BIND the composite — the
honest shape is one **verb** with two placements: a bulk-space operator
`B: V_int → W_int` lifted to the composite by `FullField(interior=B(ψ.interior),
boundary=0)` —

* INSIDE the composite-bound bulk-role bindings (S, N2N, F, SN's C): their
  composite action IS the lift of their construction-selected interior body
  (the base's one spelling, keyed on `block_role == BULK`);
* AT the assembly for bulk-born operators entering a composite sum: the
  diffusion loss `L + C − lift(IsoS + IsoN2N) − B` (three sites,
  `diffusion/solver.py:236–250`), so the energy bindings never see a
  `FullField` (§19.4).

The zero trace's CLASS (angular vs scalar; source vs flux for `solve`) is
today the reason for the family parse. It dissolves if the carriers name
their role partner: each flux leaf declares its source-sink class and each
source-sink its flux (`AngularFlux ↔ AngularSourceSink`, `ScalarFlux ↔
ScalarSourceSink`, `HarmonicMomentFlux ↔ HarmonicMomentSourceSink`, the two
boundary pairs) — one `ClassVar` per leaf, Pattern 4 at the carrier tier.
`[M]` no such link exists (`grep SOURCE_SINK|role_partner|as_source` over
`transport/fields` + `source_sinks` = 0; the pairs are already sibling
subclasses of one family base: `AngularFlux(AngularField)`,
`AngularSourceSink(AngularField)`, …). The "transitional shim" framing in the
docstrings dies with the bless; #306 item 2 closes.

**Refuted candidate** — retire the composite arms and lift ONLY at the
assembly (tier 3 read literally, S/N2N/F re-bound on the bulk): the
within-group members bind the composite by ruling (§3), `A_AA`'s OperatorSum
lives on it, and the ledger's R1/R2 rows pin those ends.

### 19.6 R-3 — #205's ScalarFlux arm on the transfer core: RETIRE

*[M c6dc40c3]* 0 traffic in two censuses with an AST-closed reference set
(the third, §19.11, pending). Its body (`transfer.py:1149–1171`) is
`isotropic_energy.apply` wrapped in a `ScalarSourceSink` — a Pattern-2 twin
of the energy binding at one hop. The typed scalar entry point #205 wanted
EXISTS as the energy binding (`n2n.py:65–67` already routes a scalar consumer
there). The C6 static pin `S.apply(phi) → ScalarSourceSink`
(`tests/sn/operators/test_operators_apply_typed.py:410/:463`) re-keys to the
energy binding; F's ScalarFlux REFUSAL arm dies with the dispatcher (the one
body's admission refuses any non-`FullField` with a message naming the
operator — G1.3's contract, `test_monomorphic_leaves.py:793`, kept).

### 19.7 Declared non-changes, and one re-home the census found

* SN's C stays 140/140 silent — the fused sweep reads its coefficient
  (#29(d)); not this step's subject.
* `LegendreMomentTransfer` / `FissionMomentOperator` keep the moment-space
  array contract. **R-5:** LMT's typed `HarmonicMomentFlux` arm
  (`transfer.py:307/:346`) is consumed only by the windowed body; keep it as a
  declared exemption of done-when (1) — the minted `source_reconstruction`
  face exists FOR that route — or retire it by routing the windowed body
  through `frame.reconstruct_after(Λ)` on values. Lean: **keep**.
* `_aniso_source_from_moment_values` has `[M]` **0 production callers**; its
  docstring's *"sole caller is the windowed moment-iterate arm"* is
  present-tense-FALSE (the windowed arm took the typed faces at F-1). It
  survives as the 0-ULP crosscheck ORACLE
  (`tests/sn/operators/test_scattering_kernel_crosscheck.py`) — re-described,
  not deleted (`coding-standards`' fuller-view exception).
* `kernel` (raises on `is_isotropic`), `full_transfer_kernel`, the fold family,
  `is_foldable_into_sigma_r`, `foldable_sigma`: unchanged.
* The eigenvalue protocol stays array-valued (§5).
* **One slot, one level** (the census's §5/R10, #29's fifth way): the adjoint
  posing feeds `RadialCharacteristicEmission` **`F.kernel`** (`sn/solver.py:2951`)
  where the within-group site feeds the OPERATORS `S.isotropic_energy +
  N2N.isotropic_energy` (`coupled_system.py:594`). With `isotropic_energy`
  derived on F (R-1), the posing feeds `F.isotropic_energy` — the same level —
  and `IsotropicFission.apply_transpose` stops being stepped over.

### 19.8 The rulings — ✅ [R] user, 2026-09-04 (AskUserQuestion, all four forks ruled as recommended; R-5/R-6 by default)

| id | fork | lean | ruled |
|---|---|---|---|
| **R-1** | the lift as a shared base `AngularLift` (F = lift; Transfer = lift + Λ≥1; #450 folded in) vs spelling 19.2 twice | base (Rule 2) | ✅ `AngularLift` base |
| **R-2** | #306 item 2: BLESS the zero-trace emission as the tier-3 lift verb, spelled once, carriers declare their role partner | bless | ✅ bless + one verb + role partners |
| **R-3** | #205: retire the transfer core's ScalarFlux arm | retire | ✅ retire |
| **R-4** | plain-bound energy bindings / multiplier carry the ARRAY; `ScalarFlux` refused there; diffusion lifts at the assembly | array | ✅ array on plain; diffusion lifts |
| **R-5** | LMT's typed moment arm: declared exemption vs retired | keep | ✅ keep (default) |
| **R-6** | the name `AngularLift` | as proposed | ✅ `AngularLift` (default) |

### 19.9 §6b set — predicate-stated; the test-side half is a RUNTIME census, not a grep

Production (complete by reading, at `f90f7914`): the 3 dispatchers, the 4
energy verbs (+ transposes), `MultiplicationOperator.apply/solve/apply_transpose`,
`_per_ordinate.py` (absorbed into the base), `bound_operator.py` (the lift
spelling), the windowed driver (`sn/solver.py:874–925`), the diffusion assembly
(`diffusion/solver.py:236–250`), the adjoint posing's kernel slot
(`sn/solver.py:2951`), the assembly mode (`isotropic_transfer.py:161–247`).

Tests: `[M]` **86 files** name a family member (grep union of the 10 names);
the members that matter are the ARM-DIRECT callers (`S.apply(psi: AngularFlux)`,
`F.apply(moments)`, `IsoS.apply(phi: ScalarFlux)`, …) — spelled without any
symbol a static census can key on (the operand is a variable) ⟹ **enumerate
by running the re-keyed spy over the test trees**, per test node, after the
production census lands. Named now: the G2.8 matrix + registry gate
(`tests/transport/test_kernels.py:565–705` — re-keyed, not deleted), the C6
pins + dispatch parity (`test_operators_apply_typed.py:392–475`), the ledger's
`_leaf_set` (ctor spellings unchanged), `test_scattering_kernel_crosscheck.py`
(the oracle), `test_fission_operator.py`, `test_isotropic_fission.py`,
`test_tier2_equivalence_s_family.py` (F's exact-ctor row re-keys with #450),
`test_transfer_roles.py` (unchanged: the roles stay thin over the same cores).
Duck-typed stubs / `getattr`-string reads per the 2026-08-24/29 inventory:
found by the red loop, listed at close-out.

### 19.10 Battery B-5 — the §11.6 skeleton, re-shaped to this design

| arm | mutation | must RED |
|---|---|---|
| B5.1 **positive control** | the selected interior body swapped (angular body on a moment-domain binding) | the cart2d headline + windowed gates |
| B5.2 | the lift wraps `.values` without the space | typed-sink shape/space gates |
| B5.3 | one role-partner pair swapped (`AngularFlux → ScalarSourceSink`) | C/S/F composite gates, the ledger |
| B5.4 | the AST no-dispatch gate loosened to admit one `isinstance` | its own row |
| B5.5 | the windowed sibling bound on the ANGULAR domain | the moment-domain admission + cart2d |
| B5.6 | the zero-trace spelling emits a non-zero trace | within-group composition, ledger, adjoint records |
| B5.7 | the adjoint posing fed `F.kernel` again | the one-level witness (new) |

Scope: `tests/transport`, `tests/sn/operators`, `tests/sn/solve`,
`tests/diffusion`, `tests/homogeneous` + the ERR-082 record; cost measured
before the arm list is fixed (§11.6's ⚠). The test-architect EXTENDS §11.6
after the rulings (plan exists; do not re-dispatch from scratch).

### 19.11 The step-5b census delta — LANDED 2026-09-04

`[M]` `scratch/cs4c_step5b_feeding_census.md` (qa, read-only, HEAD `f90f7914`;
instruments `scratch/cs4c_step5b_*`; 16 scenarios = the 13 legacy verbatim + slab-P2
+ slab-P1 adjoint with a live (n,2n) stack + the **Be-reflected 421-group fixture**;
all four controls pass; all 16 headline numbers bit-identical instrumented vs
control; the 13 shared rows bit-identical to the 2026-08-31 census — #434 and
#426 steps 1–3 moved no production value on them).

**The denominator FORKED with the role carve, and both halves close:** **24
distinct BODIES** (13 dispatch arms + 11 plain verbs) / **32 ROLE × surface
rows**. `34 − 5` (S+N2N registries fused) `− 1` (their transposes) `− 2` (the
iso pair's verbs fused onto `IsotropicTransfer`) `− 2` (`N2NMomentOperator`
retired, `[M]` 0 AST refs over 347 files) `= 24`. **FIRED 19 / NOT-RUN 13** of
the 32, every NOT-RUN row carrying its gating branch (both sides measured) and
its kernel-bypass check.

| row | what it means for §19 |
|---|---|
| ⭐ **D3 — arm-identical, body-different.** S and N2N fire the same arms with identical counts (`4687 / 4422 / 265`) and on **12 of 13** legacy scenarios `N2NOperator.is_isotropic` is `True` on 100 % of its calls (the fixtures' `Sig2` is P0-only → the pad supplies exact zeros → the ℓ ≥ 1 body is skipped). A synthetic ℓ = 1 `Sig2` flips it (k moves **−36.9 Δk·10⁵**); the Be fixture flips it (`SOURCE ×748`). | The (n,2n) anisotropic path is LIVE and load-bearing only on data that carries it — §19.2's construction-time absence of the ℓ ≥ 1 summand is keyed on exactly that datum fact, so it is right by construction. ⚠ B-5's positive control must use the Be fixture or the synthetic ℓ = 1 `Sig2`, not the 2-group legacy fixture (§19.10 amended). |
| **D3b** — at `scattering_order = 2` on a P1-elastic fixture the clamp (O-1, `SigS` alone) runs L = 1, bit-identical to P1; the (n,2n) PAD fires (2 events, exact zeros) at every L ≥ 1. | Nothing to change; the pad + `is_isotropic` are one mechanism. |
| **D5 REFUTED** — the 2026-08-31 *"IsoN2N typed 0 of 7, bare at 2× its sibling"* is gone: the P0 pair is fed byte-for-byte identically, **9 of 9** (`transfer.py:668` / `:842` — one shared body). | The iso-pair asymmetry §17 warned about no longer exists. |
| **D6 REFUTED** — `N2NOperator.apply[ScalarFlux]` was a refusal and now ACTS (shared body): "2 of 2 act, F refuses". | R-3 retires the arm on the core → both roles at once. |
| **D15** — `S.apply[ScalarFlux]` **0**, third census (row 6: a *coarse-entry surface whose arithmetic ships through the finer binding*, `isotropic_energy`, 4687 typed calls; **not** "no consumer"). | R-3's evidence: the arm is a twin at one hop, measured. |
| **D8** — `LegendreMomentTransfer.apply/apply_transpose` are TYPED hatches (`HarmonicMomentFlux ↔ HarmonicMomentSourceSink`, `transfer.py:307/:321`), a §2 surface the old census recorded as ndarray-only. | R-5. |
| **D11** — `IsotropicFission.apply_transpose` 0, **bypassed on TWO routes**: the posing feeds `F.kernel` (`sn/solver.py:2951`) and `FissionMomentOperator.apply_transpose` reads `energy.kernel` (`fission.py:219`). | §19.7's one-slot-one-level item covers BOTH routes: with `isotropic_energy` derived on F (R-1), the posing and the moment factor feed the OPERATOR. |
| **D12** — `FissionOperator`'s 5 forward arms **0 of 5**, both sides of its one gating branch exercised (slab early-return ×3, sphere stack ×1). | F's forward body stays as the pencil's F (§16.1); it becomes ONE body under R-1. |
| **D1** — SN's C **174 (174) silent**, override re-verified at identical lines. **D14** — `C.apply[ndarray]` 2 calls, one site (homogeneous). | §19.7 declared. |
| **D16** — the 2-D non-endomorphism unchanged (`#c65fb053…` bound / `#fb9da0f7…` operand), now on ONE body serving two roles. | §19.2's windowed sibling. |
| **D17** — ctor census 15 external / **3** internal + **11 POLYMORPHIC mints** (`cls(...)`, `type(self).isotropic_binding(...)`) a name-keyed census cannot return (`grep "ScatteringOperator("` over `orpheus/` = 0). | §19.9's §6b set uses the polymorphic predicate; the 5b OWED A2 lands in plan-authoring §6b. |
| **D20** — `isotropic_transfer.py:3-6/:56-58`: *"CP / MoC / diffusion feed raw scalar-flux arrays; SN passes `phi.values`"* — `[M]` **0 of 4** clauses hold (CP/MoC reference 0 roster classes; diffusion feeds a `FullField`; SN passes a typed `ScalarFlux`). | Present-tense-false prose: corrected in step 5's docs pass with the measured denominator. |

**§6 — construction-time selectability, measured:** `TransferOperator` (both
roles) is **NOT a function of the bound kind today** — 3 arms on ONE bound
kind — because (i) the `FullField` arm re-enters on `psi.interior` (an arm
counted twice), and (ii) the windowed 2-D iterate arrives on an instance bound
to the ANGULAR composite. Externally every feed is a `FullField` (4687 =
4422 angular interiors + 265 moment interiors). ⟹ §19.2 answers it exactly: the
re-entry dies (the composite action is the lifted interior body) and the
moment-interior operand gets its own binding on the moment domain — after the
step, bound kind → body IS a function, by construction. The `IsotropicTransfer`
/ `IsotropicFission` hatches are NOT functions on the DIFF composite (`FullField`
×28 from the realizer + bare ×28–30 from the assembly's impulse probes) ⟹
§19.4/§19.5: plain-bound + lifted at the assembly, and the assembly's impulses
ride the plain binding. `MultiplicationOperator` IS a function on the two
kinds with traffic (`FullField` on DIFF's composite, `ndarray` on HOMO's bulk);
its SN binding (174 instances) has no traffic at all.

**§7 — the B-5 anchors moved** (`diffusion/solver.py:289 → :292`,
`cp/solver.py:544 → :546`, `moc/core.py:108 → :110`, each now
`compute_fission_source`), and the battery's premise was wrong: the
`EigenvalueSolver` protocol is generic over `Carrier`, CP and MoC reference
**0** roster classes, and diffusion's realizer is TYPED inside (it unflattens
to a `FullField` and rides the iso hatch — the 28 composite calls). §19.10's
arms are keyed on the operator family, not on "the sibling realizers are
ndarray-typed".

**OWED tracked-file edits from 5b** (`scratch/cs4c_step5b_OWED_skill_and_memory.md`,
all three checked ABSENT at HEAD with a positive control): A1 → vv-principles
#29 (*an arm counter is one frame too coarse: arm-identical, body-different*);
A2 → plan-authoring §6b (*a core/role carve makes a name-keyed construction
census return zero — the mint is `cls(...)` in the core*); A3 → plan-authoring
§2 (*a shared-body carve forks the denominator; state both halves and close
the Δ*). Adjudicated: all three general — they land, with the earlier
report-hygiene A3, in step 5's docs commit. The qa memory delta is the agent's
own.

**§19.10 amended by the census:** B5.1's positive control uses the Be-reflected
fixture (`SOURCE ×748`) or the synthetic ℓ = 1 `Sig2` (−36.9 Δk·10⁵), never the
2-group legacy fixture (D3: on it every (n,2n) ℓ ≥ 1 body is a no-op, so a
mutation there is blind by construction). B5.7 gains the second route: the
moment factor fed `energy.kernel` again (`fission.py:219`).

### 19.12 The test-side carrier census — LANDED 2026-09-04 (the §6b set is MEASURED, not grepped)

`[M]` `scratch/cs4c_step5b_test_consumers.md` (explorer; pytest plugin
`scratch/cs4c_step5b_test_spy.py`; 16 trees = `tests/transport`, `diffusion`,
`homogeneous`, every `tests/sn` subtree + top-level files, `cross_method`;
4447 items, 3 809 742 recorded calls, 1050 s serial under the canonical flags;
**16 of 16 trees rc=0 at HEAD**; the `REQUIRE_ALL` gate tripped once on
`tests/transport` because `FissionOperator.apply` has 0 calls there — re-run
without it, 737 passed / 1 skipped).

* **The re-key population:** direct off-design feeds (a)–(g) = **55 nodes in
  14 files** (58 caller sites); with the refusal pins (h) and the helper feeds
  (i) = **102 nodes in 15 files**. Top buckets: (a) S/N2N `.apply` fed a typed
  flux directly **35 nodes / 8 files** (`test_scattering_operator.py` 15,
  `test_n2n_operator.py` 8, `test_scattering_adjoint.py` 5); (h) refusal pins
  32 / 7 (`test_monomorphic_leaves.py` 24); (i) the transfer helpers
  (`add_iso_source`, `build_aniso_source`, `_aniso_source_from_moment_values`)
  16 / 4; (f′) `MultiplicationOperator` fed neither ndarray nor FullField 10 / 3;
  (e) iso fed `FullField`@plain or ndarray@composite 7 / 3.
* **610 of 975** family-touching nodes reach it only through production —
  the bit-identity coverage the step rides.
* **7 monkeypatch-surrogate nodes** (3 trees) override a spied verb in the test
  body (`MultiplicationOperator.apply`, `IsotropicFission.apply`,
  `ScatteringOperator.apply_transpose` ×2, `FissionOperator.apply_transpose`
  ×3) — the §6b members no symbol census returns (2026-08-24/29 inventory);
  7 is a floor.
* ⭐ **The largest off-design feed in the suite is PRODUCTION:**
  `transfer.py:842` hands a `ScalarFlux` to the PLAIN-bound `isotropic_energy`
  — 683 527 calls over 608 nodes in 11 trees. It is §19.4's one named site;
  R-4 makes the lift wrap `.values` there.
* **The typed-flux arms of S/N2N/F have zero external production callers**
  (production feeds `FullField`/`TimedFullField` only; the sub-carrier arms are
  reached re-entrantly or by tests) — R-3's evidence, from the test side.
* ⚠ **The moment-domain sibling has NO test binding its ends today:** every
  composite-bound S/N2N/F binding in the whole suite has an ANGULAR interior;
  the windowed moment composite reaches 64 nodes in 20 files from exactly two
  production sites (`iteration.py:699`, `sn/solver.py:4013`), 0 direct. §19.2's
  `on_moment_domain()` sibling lands with its own binding + reciprocity gate
  (§6c: the gate's first red is the current tree's angular-bound instance fed a
  moment composite).
* 3 trees carry zero family traffic: `cross_method` (81), `sn/mesh` (178),
  `sn/angular` (20).

---

## 20. ⏸ COMPACTION POINT #5 (2026-09-04, written pre-compaction with full context; step 5 RULED, execution NOT started)

**Where the tree is.** Branch **`refactor/cs4c-step5-construction-selected-bodies`** =
`main` `f90f7914` (the #426 exit; tree clean; 13-tree baseline `[M]` **11142 collected,
13 of 13 rc=0**) + `86c17898` (§19, the design round) + this record's commit. `main ==
origin/main == f90f7914`. **No production file has been edited.** The HDF5 store is
format 2 (regenerated here; every other checkout regenerates).

**What this session did (all read-only on `orpheus/`/`tests/`/`docs/`):** the §7
reconciliation (§19 opener); the production census re-run at HEAD (§19.11,
`scratch/cs4c_step5b_feeding_census.md`, 16 scenarios, 24 bodies / 32 role rows, 19
fired / 13 not-run — D3 is the finding to carry: S and N2N are arm-identical and
BODY-different; the (n,2n) ℓ ≥ 1 body runs only on data with a moment above ℓ = 0);
the test-side carrier census (§19.12, `scratch/cs4c_step5b_test_consumers.md`, 55
direct-feed nodes / 14 files; 102/15 with refusal pins + helper feeds; 7 monkeypatch
surrogates; the moment-domain sibling has NO test binding its ends); the design round
(§19.1–§19.7) and its **six rulings — ✅ [R] user 2026-09-04, all as recommended**
(§19.8: R-1 `AngularLift` base; R-2 bless the zero trace as ONE tier-3 lift verb +
carriers declare role partners; R-3 retire the transfer core's ScalarFlux arm; R-4
plain-bound energy bindings carry the ARRAY, diffusion lifts at the assembly; R-5 keep
LMT's typed moment arm; R-6 the name `AngularLift`). #429 is CLOSED (the umbrella
ruling is moot; the memory index corrected).

**⏳ IN FLIGHT at compaction: the test-architect** (dispatched 2026-09-04, brief =
§19 + both censuses + `scratch/cs4c_verification_plan.md` §11.6/§8) writing
**`scratch/cs4c_step5_verification_plan.md`** — gates G5.1–G5.9 (below), battery B-5
re-shaped with its measured scope cost, predicted per-tree deltas vs 11142, and any
§19 claim it refutes at the TOP of its memo. ▶ At resume: if the file exists, READ IT
FIRST and fold its refutations into §19 in place (§3); if it does not, check `/tasks`
(the agent may still be running) and only then re-dispatch with the same brief. Its
report is not shown to the user — relay what matters.

### 20.1 The execution shape — file by file (settled this session after reading every implementing class; `[R]` where ruled, `[P]` the mechanism)

One merge unit; commit order (i)→(v); every `file:line` at `f90f7914`.

1. **Carriers declare their role partner** (`[R]` R-2) — `orpheus/transport/fields/_bases.py`:
   a `FieldRole` enum (`FLUX`, `SOURCE_SINK`) + a role-pair mixin on `BulkField` and
   `FaceField` whose `__init_subclass__(cls, *, flux=None, **kw)` registers BOTH
   directions once (`cls.FLUX = flux; cls.SOURCE_SINK = cls; flux.FLUX = flux;
   flux.SOURCE_SINK = cls`; a second declaration for one flux refuses), plus
   `role_partner(role) -> type` and `into_role(role, values)` (same space and family
   fields, new class — via `dataclasses.fields`, so `L`/`spatial_moments` ride). Each
   SOURCE-SINK leaf declares `flux=<its flux>` in its class statement (`Angular`,
   `Scalar`, `HarmonicMoment` bulk; `AngularBoundary`, `ScalarBoundary`; the two
   radial-characteristic pairs). ⚠ Import direction: sinks import flux LEAVES
   (`[M]` today sinks import only `fields._bases`; flux leaves import no sink) — legal,
   cycle-free; `fields/__init__.py` ends with a bare `import
   orpheus.transport.source_sinks` so the registration completes whenever `fields`
   loads (a NAMED import there would cycle on the partially-initialised package).
2. **`orpheus/transport/operators/lift.py` (NEW; `_per_ordinate.py` folds in and
   retires):** `admit_composite(op, x, *, space) -> FullField` and
   `admit_array(op, x) -> ndarray` — the family's ONLY two carrier parses (typed
   refusals naming the operator; G5.1 exempts these two by name);
   `lift_bulk_action(psi, interior_action, *, trace_role)` — THE one spelling of
   "a bulk action enters the composite by extension-by-zero on the trace" (`[M]` 9
   spellings today, §19.1); `embed_bulk_assembly(bulk_assembled, composite_space)`
   (the flat layout is `[bulk C-ravel | trace]`, so the embedding is index-identity
   on the bulk block); `BulkLift(inner, *, domain, codomain)` — a bulk-born operator on
   the composite (`apply`/`apply_transpose` through the verb with `into_role`;
   `assemble` = the embed; `is_assemblable`/`is_adjointable` = the inner's;
   `block_role = BULK`; the inner's ends must content-equal the composite's interior).
3. **`orpheus/transport/operators/angular_lift.py` (NEW):** `AngularLift(BoundOperator["FullField"])`
   with kw-only `flux_analysis`, `source_reconstruction` + ends; `block_role = BULK`;
   abstract `data_ng`, `_bind_energy()`, `_frame_form()`; `__post_init__` = energy
   extent, face admission against the **codomain** interior (the analysis face is
   minted on the angular space the binding EMITS on), then the SELECTION: domain
   interior `== flux_analysis.domain` → angular (`_scalar_flux_of = integrate_angular`,
   `_conjugate = frame.conjugate`, domain cotangent = `AngularSourceSink`); `==
   flux_analysis.codomain` → moment (`scalar_flux(space=scalar_interior)`,
   `frame.reconstruct_after`, cotangent = `HarmonicMomentSourceSink(values, space,
   L=frame.truncation_order, spatial_moments=BulkField._spatial_moments_per_axis_of(dom))`);
   anything else refused. `isotropic_energy` = cached `_bind_energy()` (ONE cache home);
   `total_weight`, `frame` (off the faces), `_moment_space`, `_domain_interior`,
   `_codomain_interior`, `_scalar_interior_space` as today. `apply(FullField)` =
   `lift_bulk_action(admit_composite(…), self._interior_action, trace_role=SOURCE_SINK)`;
   `_interior_action(bulk)` = `_combine(_isotropic_source(bulk), None)`;
   `_isotropic_source` = `ScalarSourceSink(values=isotropic_energy.apply(phi.values),
   space=phi.space)`; `_combine(iso, aniso)` = `(iso / W) + (aniso or
   AngularSourceSink.zeros(codomain_interior))` — the former
   `assemble_per_ordinate_isotropic`, bit-identical (it allocated the zeros too).
   `apply_transpose(FullField)` = the same verb with `domain_cotangent(
   _frame_form().apply_transpose(bulk.values) / W)`. `is_adjointable = True`.
4. **`transfer.py`:** `TransferOperator(AngularLift)` — field `transfer`; ClassVars
   `isotropic_binding` (default the core), `channel` (no default); `_bind_energy` =
   `type(self).isotropic_binding(self.transfer.at_order(0), domain=s, codomain=s)`;
   `__post_init__` → `super()` then `_redistribution` selected at CONSTRUCTION: `None`
   iff `is_isotropic` (the per-call predicate at three sites dies); angular →
   `AngularSourceSink(values=kernel.apply(bulk.values) / W, space=codomain_interior)`;
   moment → `source_reconstruction.apply(_moment_transfer(skip_l0=True).apply(bulk)) / W`
   (`[R]` R-5: the typed faces route keeps LMT's typed arm live); `_interior_action`
   override = `_combine(iso, self._redistribution(bulk))`; `kernel` =
   `self._conjugate(Λ_{ℓ≥1})` on THIS binding's domain (raises on isotropic, as
   ruled at #426); `full_transfer_kernel` = `self._conjugate(Λ_{ℓ≥0})`, `_frame_form`
   returns it (the term-named products STAY — `[M]` 32 + 11 sites; the base calls the
   abstract); NEW tier-2 verb **`on_moment_domain()`** → `type(self)(transfer, faces,
   domain=FullFieldSpace.from_blocks(flux_analysis.codomain, self.domain.trace_space),
   codomain=self.codomain)`; the ScalarFlux arm RETIRED (`[R]` R-3); `add_iso_source`
   becomes ndarray-only (its typed arm has `[M]` 1 test consumer, 0 production);
   `build_aniso_source(AngularFlux | None)` kept (production `sn/solver.py:2336`);
   `_aniso_source_from_moment_values` RETIRED (`[M]` 0 production callers, its
   docstring false — the crosscheck oracle becomes `S.on_moment_domain().kernel` fed
   `M·ψ` against the typed route, 0 ULP); `_sibling`/fold family require an angular
   endomorphism (refuse otherwise; production-dead anyway); the `singledispatchmethod`,
   the `@overload` surface and the FullField re-entry die.
5. **`fission.py`:** `FissionOperator(AngularLift)` — field `fission:
   FissionMaterialField` (REPLACES `energy` + `frame`; #450); `_bind_energy` =
   `IsotropicFission(self.fission, domain=s, codomain=s)`; `kernel` /
   `production_rate` delegate to `isotropic_energy`; `full_fission_kernel` =
   `self._conjugate(FissionMomentOperator(self.isotropic_energy, ends))`;
   `from_solver_data` mints the L = 0 faces through `HarmonicFrame.for_space(interior,
   0)`; the five arms and the runtime shape guard die. `FissionMomentOperator.apply /
   apply_transpose` call `self.energy.apply(…)` / `.apply_transpose(…)` — the energy
   OPERATOR, not `energy.kernel` (D11's second bypass route).
6. **`isotropic_transfer.py`:** `IsotropicTransfer` / `IsotropicFission` are
   plain-bound ONLY (`[R]` R-4; a `FullFieldSpace` domain refused with a message naming
   `BulkLift`); `apply(ndarray) -> ndarray` via `admit_array`, `apply_transpose`
   likewise (the composite refusal + its "#281" note die — `[M]`
   `tests/diffusion/test_operators.py::test_k_iso_composite_refuses_angular_bulk_and_transpose`
   pins `match="#281"`, re-key); `_values_of`, `_scalar_composite_source`,
   `_iso_is_assemblable`, `_assemble_iso_energy_operator` retire; `assemble()` emits
   the BULK cell-block-diagonal on the plain ends (ng impulses through `apply`),
   `is_assemblable = True` on the transfer binding (⚠ §8: `as_matrix` BRANCHES on it —
   the homogeneous `C − K_iso` then takes the assembly path instead of probing; the
   entries are the same `apply` columns, so k∞ should be bit-identical — MEASURE, the
   homogeneous records + `tests/homogeneous/test_operator_spaces.py` are the pins);
   `IsotropicFission` keeps NO assembly (its docstring's deliberate omission stands);
   the module header's D20 falsehoods corrected with the measured denominators.
7. **`multiplication_operator.py`:** the body selected in `__post_init__` from the
   DOMAIN (a `FullFieldSpace` parse at construction is legal — it is the space, not
   the carrier): composite → `lift_bulk_action` with `bulk.into_role(SOURCE_SINK,
   engine(values))`, the scalar-interior degenerate `[None]…[0]` lift chosen by
   `len(interior.shape) == coefficient.values.ndim` (`[M]` `DiagonalOperator._check_shape`
   pins the rank under `broadcast_axes`), never by carrier class; plain → the engine on
   `admit_array`; `solve` the same with `FLUX`; the dispatcher and the four family
   parses die; `assemble` composite = `embed_bulk_assembly(bulk diagonal)`, plain =
   the bulk diagonal (new: a plain binding is assemblable).
8. **`diffusion/solver.py:236–250`:** `bulk = mesh.bulk_space`; `scattering =
   BulkLift(IsotropicScattering.from_material_xs(mat_xs, space=bulk) +
   IsotropicN2N.from_material_xs(mat_xs, space=bulk), domain=space, codomain=space)`;
   `self.fission = BulkLift(IsotropicFission.from_material_xs(mat_xs, space=bulk), …)`;
   `compute_fission_source` unchanged (feeds a `FullField`). `[M]` the resolvent is on
   the assembly path (§19.4) — G5.6 pins the assembled loss bit-identical.
9. **`sn/solver.py`:** `_within_group_si` (`:1174–1237`) — when `windowed`, the gains
   become `S.on_moment_domain()`, `n2n.on_moment_domain()` (`B_a` unchanged; the
   Krylov paths never window, `:737`); `_adjoint_posing_parts` (`:2951`) feeds
   `RadialCharacteristicEmission(F.isotropic_energy, …)` — was `F.kernel` (D11's first
   route; `coupled_system.py:594` already feeds the operators). `sn/coupled_system.py`
   unchanged.
10. **Tests** — the re-key set is `scratch/cs4c_step5b_test_consumers.md` §2 (buckets
    (a)–(i), listed per node); the gates from the verification plan (G5.1 the AST
    no-dispatch gate with a synthetic positive control; G5.2 the 18-cell ends→body
    fence replacing G2.8's 12 cells + the registry gate inverted; G5.3 the
    moment-domain sibling: ends, bit-identity vs the angular-bound instance fed the
    moment composite, transpose reciprocity, third-interior refusal; G5.4 the
    role-partner bijection over `__subclasses__`; G5.5 the one zero-trace spelling;
    G5.6 the diffusion assembly bit-identity; G5.7 the one-level witness
    (`IsotropicFission.apply_transpose` REACHED on the sphere adjoint); G5.8
    `AngularLift`'s own laws; G5.9 the production headlines bit-identical: the 16
    census values, the ERR-082 record, the diffusion witness, the adjoint records);
    `test_transfer_roles.py` gains the row `AngularLift.__subclasses__ == {TransferOperator,
    FissionOperator}` (the roles' gate is untouched: `AngularLift` sits ABOVE the cores);
    the tier-2 F row re-keys to the exact ctor `FissionOperator(field, flux_analysis=,
    source_reconstruction=, domain=, codomain=)`; the C6 pins collapse to
    `apply(FullField) -> FullField` (+ `S.isotropic_energy.apply(ndarray)`); the 7
    monkeypatch surrogates re-key (`grep "monkeypatch.setattr(.*(Operator|Transfer|Fission)"`
    for the ceiling); battery B-5 per the verification plan (positive control on the
    Be fixture or the synthetic ℓ = 1 `Sig2`, NEVER the 2-group legacy fixture — D3).
11. **Docs (archivist pass; a step that moves an equation-level claim):**
    `isotropic_transfer.py` header (D20); every page naming
    `assemble_per_ordinate_isotropic` / `_per_ordinate` (`[M]` `operator_algebra.rst`,
    `error_catalog.rst`, `sn/history.rst`, `sn/index.rst`, `curvilinear_*.rst`),
    `_aniso_source_from_moment_values` (`operator_algebra.rst`, `cartesian_multid.rst`),
    `F.energy` (`coupled_block_operator.rst`, `operator_algebra.rst`, `adjoint.rst`,
    `sn/history.rst`, `api/numerics.rst`), the "transitional shim" framing (#306 item 2
    BLESSED), `api/transport.rst` (two new modules), `operator_algebra.rst
    §integral-kernel-category` (the windowed route = the moment-domain binding),
    `adjoint.rst` (the transposes' `FullField`-only surface). Issues: `Closes #450`;
    #306 item 2 closed by comment (items 1/3/5 stay); #205 comment (the typed scalar
    entry is the energy binding). Plan-authoring surprise-log row OWED: a consumer
    census that EXCLUDES a tier (`orpheus/numerics/`) read **0** and was wrong by
    exactly that tier (§19.4's correction) — the 2026-09-02 file-exclusion row's
    sibling. The 5b OWED edits (A1/A2/A3 in `scratch/cs4c_step5b_OWED_skill_and_memory.md`)
    and the earlier report-hygiene A3 land in the docs commit.
12. **Exit:** the 13-tree driver (model `scratch/_426_step2_gate_driver.sh`) against
    11142 with the verification plan's predicted per-tree deltas, reconciled per file
    against a HEAD worktree collect-only; `sphinx -E -W -q docs docs/_build/html` from
    the repo root; `mcp__nexus__dead_references` (the retirements: `_per_ordinate`,
    `assemble_per_ordinate_isotropic`, `_aniso_source_from_moment_values`,
    `_scalar_composite_source`, `_values_of`, `FissionOperator.energy/.frame`); `npx
    pyright` 0 on `orpheus/`; elegance-enforcer review after the carve; ff-merge; branch
    deleted; §20.x close-out + the compaction protocol.

**Landing order (§6b — the interface change is one commit):** (i) item 1 + item 2
(additive, bit-identical, green alone) · (ii) items 3–9 + the test re-keys (item 10's
re-keys) in ONE commit — the tree does not compile between them · (iii) the new gates +
the battery · (iv) docs · (v) plans/memory. Predicted-then-measured per tree; the 66 xf
carry through (§8.1 schedules no flip rows).

### 20.2 Corrections that supersede older text (read these over anything above them)

* §17's *"S's arm docstring at `scattering.py:1120`"* → `transfer.py:1157–1170`
  (moved with the body at #426 step 2); §17's `n2n.py:56/:202` banners are GONE.
* §18.2's *"Do not re-run either. Read them."* is SUPERSEDED by §19.11/§19.12 — the
  2026-08-31 census's tree half is void (five carves old); read the 5b memos.
* §18.8's *"two rulings owed"* → ONE (#429 CLOSED 2026-09-03); step 5's own is now
  RULED (§19.8).
* §19's opener sentence *"N2N's ℓ ≥ 1 arms fire at `scattering_order ≥ 1`"* is
  ⛔ REFUTED in place (D3/D3b): the ℓ ≥ 1 body runs iff the DATUM carries a moment
  above ℓ = 0.
* §19.4's first draft claimed the assembly mode had 0 production consumers — corrected
  in place: `numerics/flat_operator.py:168` (the diffusion resolvent), `operator.py:1135`,
  `coupled_system.py:1246`.
* §19.10's positive control moves off the 2-group legacy fixture (D3).

### 20.3 Standing constraints (unchanged)

Host → `.venv/bin/python`; canonical runner `-O -m pytest -p no:randomly -m "not slow"
-q` SERIAL; `main` always green; branch + `git merge --ff-only`, never squash; never `git
add -A` (stage by path; read `git status --porcelain` first — `scratch/` is untracked
and large); never `git stash`; never `git checkout`/`restore` on a tracked path; no
source edits under a running gate (L37); no production edit while a sub-agent holds the
tree (L38 — the test-architect is in flight); long gates detached + LOG + the 13-tree
driver + Monitor; batteries as subprocess-scoped pytest plugins, crash-safe, one
process per arm; `sphinx -E -W` + `dead_references` at every exit; a bare `assert` in
`orpheus/` is inert under `-O`; commit via `git commit -F <file>` with the two trailers;
streamed `<new-diagnostics>` are the advisory LSP artefact — trust `npx pyright`; main
agent writes production, sub-agents review/design (test-architect before — in flight;
elegance-enforcer after; archivist for docs); agent memories and the generated
`matrix.rst` / `error_index.md` are committed with the carve.

### ▶ RESUMES AT — step 5 EXECUTION, item (i) of §20.1's landing order

Open with §7: `git log --oneline -3` (expect `86c17898` + this record on the branch);
read `scratch/cs4c_step5_verification_plan.md` (or `/tasks`); re-read
`orpheus/transport/operators/transfer.py:360–560` and `fission.py:224–300` before the
first edit (a scope read from THIS record gets refuted by the realization — §7.2); then
write `lift.py` and the role-pair mixin first (green alone), then the one-commit
interface change. The user steers at the AskUserQuestion checkpoints the surgical
posture prescribes; every ruling this step needed is already taken (§19.8).

## 21. ⏸ COMPACTION POINT #6 (2026-09-05) — step 5 EXECUTED through the qa round; the elegance round STAGED, not applied

Every anchor below re-verified at branch tip `b68d4bf0`
(`refactor/cs4c-step5-construction-selected-bodies`; `main` is `f90f7914`).

### 21.1 What landed — the commit table

| commit | content | gate |
|---|---|---|
| `b915cb90` | item (i): `FieldRole` + `RolePair` on the two storage bases, the 7 source/sink leaves declare `flux=`, `fields/__init__` tail import; `lift.py` (`admit_composite`, `admit_array`, `lift_bulk_action`, `embed_bulk_assembly`, `BulkLift`); G5.4 + the lift's laws (41 rows) | scope C + layer contract 2733 passed / 16 xf rc=0 |
| `8d432a5d` | item (ii): `angular_lift.py` (`AngularLift[EnergyT]`, the two ends, `on_moment_domain`), `TransferOperator`/`FissionOperator` on the base (F's field `fission` replaces `energy`+`frame`, #450), plain-bound energy bindings with `assemble()`, the multiplier's domain-selected body, `diffusion/solver.py` lifts, `sn/solver.py` windowed siblings + `F.isotropic_energy` in the posing, `_per_ordinate.py` deleted; the §6b re-keys (18 files) + G5.1, G5.2 (33 cells), G5.3, G5.7, G5.8, G5.4b. 37 files | targeted scope C+W+A 2910 passed / 16 xf rc=0 |
| `6171e3ea` | item (iv): the docs pass (8 pages; `cs4c-ends-select-the-body` + `role-partner-declaration` in `operator_algebra.rst`; the changelog row; two `implements` re-points), the owed skill/rule edits (vv #21/#29, plan-authoring ×4 rows + §2/§6b), lessons L63 | `sphinx -E -W` rc=0; `dead_references` 0 / 66 |
| `caeb995d` | the **qa review round** (report `scratch/cs4c_step5_qa_report.md`): F-1 a real negative leg (group-flipped transpose), F-2 an (n,2n)-carrying datum for the N2N sibling row, F-4 the displaced guard arm's witness + the false docstring, F-5 static pins adjudicated by hand (#452 filed), F-6..F-10; vv #22 widened, #36's non-pytest instance | touched files green |
| `b68d4bf0` | the 13-tree gate's ONE red re-keyed (`test_mms_ld_2d.py` M4 → `AngularLift._combine` via `monkeypatch`); three residual spellings dated | the test green in 5 s |
| `772af012` | the **elegance review round** (report `scratch/cs4c_step5_elegance_report.md`): S1–S11, N1, N3–N8 applied; N2 dispositioned in the docstring (two tiers guard two different defects). **N4 COMPLETED beyond the staged half** — `_select_si_splitting(LC, B, mesh, schedule) -> (implicit, boundary_gain)`: the staged patch had kept `S`/`n2n` as DEAD parameters and asserted "no other caller" inside `sn/solver.py` alone; `[M]` the selector's §6b set is 7 test files (22 sites) + the `cartesian_multid` paragraph (already drifted to a two-gain `(S, parts.upper)`), all re-keyed. Five witnesses (S1/S2/S3/S4/S6) in `test_angular_lift.py`; `docs/api/transport.rst` follows | `npx pyright orpheus/` 0; touched scope 2914/0 (5 min 06 s, `scratch/_cs4c_step5_rr_scope.log`); after N3 2260/0 (`_rr_n3.log`); `sphinx -E -W` rc=0; `dead_references` 0 dead / 66 checked |

### 21.2 Measured — the gate and the battery (on `6171e3ea`, before the qa round)

**13-tree gate** (`scratch/_cs4c_step5_full_gate.log`, canonical flags): **12 of 13 rc=0**; the one red is `b68d4bf0`'s fix. Per tree (passed; baseline passed): numerics 3255/3255 · transport **830**/737 · geometry 727/727 (+4 sk +1 xf unchanged) · data 314/314 · homogeneous 50/50 · diffusion 115/115 · cp 141/141 · moc 121/121 · mc 40/40 (+2 xf unchanged) · cross_method 81/81 · sn **3434+1**/3415 (47 xf unchanged) · derivations 1636/1636 · root **426**/425. Collected 11256 vs 11142 = **+114**: transport +93 (the new gates — `test_role_partners` 24, `test_bulk_lift` 22, `test_no_carrier_dispatch` 15, `test_angular_lift` 13, the 33-cell fence net +21 over G2.8's 12, minus the retired registry row; ⚠ **itemise per file at close-out** — `pytest --collect-only -q` per new file), sn +20 (`test_moment_domain_binding` 14, `test_fission_adjoint_route` 4, re-key deltas +2), **root +1 = the corpus-coupled term the plan predicted** (the docs pass added labels; the harness gate parametrized over the documented-label registry gained a param). The qa round (`caeb995d`) adds +6 (the arm-2 witness) + 0 net elsewhere — **the final gate must be re-run on the final tree** (21.4).

**Battery B-5** (`scratch/_cs4c_step5_battery.status`, per-arm logs `_cs4c_step5_battery_<arm>.log`, summariser `scratch/_cs4c_step5_battery_summary.py`; plugin `scratch/_cs4c_step5_battery.py`, driver `.sh`): honest 0 · B5.0 147 (positive control) · **B5.0b 25** (the plan predicted NOTHING — refuted honestly: scope C+W carries fixtures WITH Σ₂ — `test_n2n_operator`, `homo_2eg_n2n`, `test_material_field`, `diffusion/test_n2n_witness`; D3's blindness is a property of the LEGACY 2g fixture, not of the scope) · B5.1 19 (G5.3 ×6, crosscheck ×3, the 3 windowed files, the 2 two-D regression cases; the 1-D cases green ✓) · B5.1b 10 (the ERR-082 record ×4 ✓) · B5.2 62 · B5.3 20 (G5.4 + the composite gates ✓) · **B5.3b 40** (the plan said G5.4's bijection leg ONLY — refuted: `ScalarFlux→AngularSourceSink` breaks every scalar-family lift, 17 in `diffusion/test_solver.py`) · B5.5 23 · B5.5b 26 · **B5.6 killed at 67 % after 48 min** (a non-zero trace makes the 2-D solves non-convergent; 125 reds by then — the arm bites; rc=143 recorded) · B5.6b 55 (wrong trace CLASS — the scalar family reds, the SN gates never see it: the asymmetry the plan predicted) · ⛔ **B5.7 and B5.7b read 0 — a DRIVER SCOPE ERROR, not blindness**: scope A listed `tests/sn/operators/test_fission_adjoint.py` (the plan's provisional name) while the landed G5.7 gate is `tests/sn/operators/test_fission_adjoint_route.py`, so the arm ran without the gate it exists to redden; the qa report §2 independently shows both routes RED, per-route attributable. Re-run both arms with the gate in scope before the table is final (the driver is corrected) · B5.8, B5.9 **were still running at compaction** (the Monitor `bue0kn89i` fires on `DONE`). Two plan predictions refuted, none in the direction of a blind arm.

### 21.3 The elegance review — STAGED, not applied (`scratch/cs4c_step5_elegance_report.md`)

No blocker; 11 should-fix (S1–S11), 8 nits; approval conditions S1–S6, S10, S11. All prepared under `scratch/_rr/`: **`lift.py`** (S2 both blocks admitted + `carrier=` role admission; S7 `CompositeBound` mixin; S8 `lift_pointwise` — the one spelling behind `BulkLift`'s two verbs and the multiplier's two; S11 the exemptions named; N8), **`angular_lift.py`** (S1 `AngularEnd`/`MomentEnd` exported with `operand` class; S3 faces-meet-in-the-middle; S4 `apply` admits the end's flux leaf, `apply_transpose` the angular family; S5 `data_ng` and the inert per-end call GONE — the eager bind IS the ng admission; S6 `_scalar_interior_space = codomain.retraction("angular").codomain`, identity with `integrate_angular`'s space; S10 `isotropic_energy: EnergyT = field(init=False)` assigned in `__post_init__`; N1 `spatial_moments_per_axis_of` public), **`apply_review_round.py`** (transfer S1 dict `{AngularEnd: …, MomentEnd: …}[self._end]` + N6; fission/isotropic `data_ng` retired; multiplication S9 one `_interior` parse + N5 `_scalar_operand`; `_bases`/`harmonic_frame` the public static; `sn/solver.py` N4 `_select_si_splitting` returns the boundary gain and the caller names `(S, n2n, boundary)`; `__init__` N7; the battery plugin's end names), **`apply_review_round_tests.py`** (five witnesses appended to `tests/transport/test_angular_lift.py`: S1 route map keyed on the exported ends, S4 role refusal, S2 foreign-trace refusal, S3 mixed-frame refusal, S6 identity). ⛔ Refuted by me: the review's "opportunity 4" (`N2NMomentOperator` prose in `adjoint.rst`/`frame.rst`) — `[M]` both sites are dated history already.

✅ **APPLIED 2026-09-05 @ `772af012`** (the header above is history — the round is no longer staged). Two corrections to the staged material, found at pick-up by the §6b census the resume block did not mandate: **(1) the staged N4 was HALF a signature change** — it re-typed the selector's return and left `S`/`n2n` as parameters no longer read, and its self-check (`assert s.count("_select_si_splitting(") == 2`) ranged over `sn/solver.py`'s text only, so it certified "no other caller" against a population of ONE FILE; `[M]` `grep -rn _select_si_splitting tests/ docs/` → 12 test files (7 calling it at 22 sites, incl. the `_singular_loss_box` helper whose two consumers unpack the pair) + 2 theory pages. Landed as the FULL change (selector takes `(LC, B, mesh, schedule)`); every caller now names the triple `(S, N₂ₙ, boundary_gain)` itself — see the surprise-log row dated 2026-09-05. **(2) the apply script's own guard tripped on a DOCSTRING** (`assert "_composite" not in s` matched `:func:\`…admit_composite\`` in the module docstring) — the over-broad assertion was right to stop the script and wrong about why; part 2 (`scratch/_rr/apply_review_round_part2.py`) carries the narrowed check and the N4 completion. **N3** (`cast(getattr(op, end))` ×2 → one typed `_end_space(op, end)` on `BoundOperator`, whose ends are non-Optional) was NOT in the staged files and landed by `scratch/_rr/apply_n3.py`.

### 21.4 ▶ RESUMES AT — apply the elegance round, then the exit

**▶ PROGRESS 2026-09-05 (after the review round):** steps 1–3 DONE (`772af012`). Step 4 RUNNING on the final tree, launched concurrently at ~19:40 local: the 13-tree gate (`scratch/_cs4c_step5_gate_driver.sh` → `scratch/_cs4c_step5_full_gate.log`, ⚠ the driver TRUNCATES that log, so the `6171e3ea` gate's per-tree numbers survive only in §21.2), the battery (`scratch/_cs4c_step5_battery.sh` → `.status` + per-arm logs; the driver now carries `--timeout=240` per test via `pytest-timeout`, so the two arms that crawled — B5.6 at 67 %, B5.9 at 45 %, both SI runs whose mutated source diverges — are CAPPED and count their timeouts as reds; B5.7/B5.7b now scope `test_fission_adjoint_route.py` + `test_angular_lift.py`), and ONE Sphinx build — DONE: `-E -W -q` rc=0, `mcp__nexus__dead_references` 0 dead / 66 checked. B5.9 on the pre-round tree: killed at 45 % after 15 min CPU with **40** partial reds (`_cs4c_step5_battery_B5_9.log`). **The per-file itemisation of the test-count delta** — `[M]` collect-only, canonical `-m "not slow"`, `f90f7914` (a throwaway worktree, `orpheus` imported from IT) vs `772af012`: **11 142 → 11 267 rows (+125)**, 488 → 494 files, closing exactly over 10 files — transport **+104**: `test_kernels.py` 60→88 (+28, the 33-cell fence minus the cells that share a row), `fields/test_role_partners.py` +22 (new), `test_bulk_lift.py` +20 (new), `test_angular_lift.py` +18 (new; 13 at the qa round + 5 review-round witnesses), `test_no_carrier_dispatch.py` +15 (new), `test_transfer_roles.py` 7→8 (+1, G5.4b); sn **+20**: `test_moment_domain_binding.py` +14 (new, G5.3), `test_fission_adjoint_route.py` +4 (new, G5.7), `test_scattering_kernel_crosscheck.py` 4→6 (+2); root **+1**: `test_layer_imports.py` 359→360 — the per-module layer gate is parametrized over the package's modules, and step 5 added two (`lift.py`, `angular_lift.py`) and retired one (`_per_ordinate.py`): the corpus-coupled +1 the arithmetic never modelled, now itemised. (The §21.2 figures `+93 transport / +20 sn / +1` were measured at `6171e3ea`; the qa round added 6 transport rows and the review round 5.) Remaining: steps 5–6 (close-out numbers here, memory, ff-merge, push, the #306/#205 comments; the surprise-log rows are WRITTEN — see plan-authoring 2026-09-05).

**✅ 13-tree gate on `772af012` — `[M]` 13 of 13 rc=0** (`scratch/_cs4c_step5_full_gate.log`, canonical flags, run CONCURRENTLY with the battery so the timings are loaded): **11 182 passed / 19 skipped / 66 xfailed / 0 failed**, 227 deselected (the `slow` tier) — collected 11 267, which is EXACTLY the collect-only census above (the two instruments agree row for row). Per tree (passed): numerics 3255 · transport **841** · geometry 727 (+4 sk, 1 xf) · data 314 · homogeneous 50 · diffusion 115 · cp 141 · moc 121 · mc 40 (2 xf) · cross_method 81 · sn **3435** (1 sk, 47 xf, 16 min 26 s) · derivations 1636 (13 sk, 11 xf, 37 min 07 s loaded) · root+harness **426** (5 xf). Against the step-5 baseline in the memory topic file (10 106 passed / 19 sk / 66 xf at `f90f7914`… ⚠ that figure predates #426's own additions; the honest comparison is the collect-only census: 11 142 → 11 267): every tree that step 5 touched moved by exactly its itemised files, and no other tree moved at all.

**✅ Battery B-5 on `772af012` — `[M]` 16 of 16 arms complete, EVERY mutation bites** (`scratch/_cs4c_step5_battery.status`, per-arm logs, `--timeout=240` per test): honest **0** · B5.0 **148** (P0 source zeroed, 30 files) · B5.0b **26** (moment sibling only — the plan's predicted NULL, refuted) · B5.1 **20** (moment end reads the angular body) · B5.1b **10** · B5.2 **62** (`into_role` wrong shape) · B5.3 **20** · B5.3b **40** (the plan's second predicted NULL, refuted) · B5.5 **24** (`on_moment_domain` → identity) · B5.5b **27** · B5.6 **167** (non-zero trace, 35 files; 8 of them the cap's `Timeout` reds — the arm that crawled 30+ min at 67 % on the pre-round tree now finishes in 21 min 48 s) · B5.6b **55** · **B5.7 2 / B5.7b 3** (both in `test_fission_adjoint_route.py` — the gate the driver's provisional filename had left out of scope; the earlier zeros were the SCOPE ERROR §21.2 diagnosed, not blindness) · B5.8 **40** · B5.9 **108** (`_combine` drops `/W`; 1 h 02 min under the cap, 0 timeouts — its crawlers finished on their own budgets; 40 partial at the 45 % kill on the pre-round tree). Compared with the pre-round battery (§21.2): B5.0 147→148, B5.0b 25→26, B5.1 19→20, B5.5 23→24, B5.5b 26→27 — each **+1** is the review round's five witnesses entering the arms' scope; every other arm identical.

1. `cat scratch/_cs4c_step5_battery.status` — if not `DONE`, wait for the Monitor (or check `ps` for a crawling arm and kill it as B5.6 was; record partial reds from the arm's log).
2. `.venv/bin/python scratch/_rr/apply_review_round.py && .venv/bin/python scratch/_rr/apply_review_round_tests.py`; then `npx pyright orpheus/` (expect 0) and the touched scope: `tests/transport tests/sn/operators tests/sn/architecture tests/diffusion tests/homogeneous tests/sn/regression tests/sn/solve/test_2d_anisotropic_windowing.py tests/sn/solve/test_si_gate_dispatch.py tests/sn/sweep/core/test_sweep_graph_window_equivalence.py tests/sn/verification/mms/test_mms_ld_2d.py tests/test_layer_imports.py` (background, ~5 min). Expect the S4 role admission to red nothing in production (the spy `[M]` every production `apply` feed is the flux leaf; transposes carry `AngularFlux` cotangents — admitted as the family).
3. Commit: `refactor(transport): the elegance review round of step 5` (S1..S11, N1, N4..N8 + the five witnesses; cite the report).
4. Re-run the exit on the FINAL tree: `nohup scratch/_cs4c_step5_gate_driver.sh &` (13 trees, ~60 min unloaded) and `nohup scratch/_cs4c_step5_battery.sh &` (sequentially after, or concurrently accepting the slowdown; B5.6 will crawl again — cap it); `sphinx -E -W -q docs docs/_build/html` (ONE build at a time — L63) + `mcp__nexus__dead_references`.
5. Close-out: §21 gains the final numbers (per-file itemisation of the +93/+20/+1); the memory topic file; ff-merge to `main`, delete the branch, push (`Closes #450` closes on push); post `scratch/_cs4c_step5_gh_comments.md` to #306 (item 2) and #205; #452 stays open (the C6 static pins).
6. The plan-authoring surprise-log rows this step still OWES (not yet written): (a) the qa review's F-1/F-2 — a new gate's NEGATIVE leg can be tautological by construction (a signed pairing vs a squared norm) and a "same weight as S's" row can compare 0 with 0 on a datum without the channel: **a gate's fixture must ACTIVATE the thing the gate varies** (vv L40c at gate-authoring time); (b) the battery predictions refuted (B5.0b, B5.3b): a "declared null" arm is a hypothesis until run — already the plan's own rule (L58b), so a log row not a clause.

### 21.5 Standing constraints (unchanged) + two learned this step

Everything in §20.3, plus: **one Sphinx build at a time** (L63 — two builds on one `_build`/graph DB deadlocked; a 22 h 47 min hung build); **a sub-agent stall is a wait** — read the tree, not the transcript (the test-architect and archivist both stalled; their partial work was complete enough to finish by hand); **a foreground pytest over more than ~2 000 rows must run in the background** (the 2-min Bash cap killed one measurement).

---

## 22. ⏸ COMPACTION POINT #7 (2026-09-05, written after the step-5 merge, tree clean, no gate running) — the ladder's next acts, in the RULED order

**State `[M]`:** `main` @ `950f7c9b` = `origin/main`; the step-5 branch deleted; #450 CLOSED; #306 item 2 and #205 commented; working tree clean (only `scratch/` untracked + the foreign `.claude/plans/harness_context_budget.md`). Step 5's whole record is §21 (commit table §21.1, the exit on the final tree in §21.4's ✅ blocks — 13 of 13 rc=0, 11 182 / 19 sk / 66 xf, battery 16 of 16 bite, sphinx `-W` rc=0, `dead_references` 0/66). The Nexus graph was built on `772af012`; `950f7c9b` touched no indexed source.

**[R] (user, 2026-09-05): "tackle those steps in order"** — the order is the one proposed in the same turn, recorded here as the ruling. Each act opens with the standing protocol (reground against the TREE per plan-authoring §7 — never this file's tense; explorer re-census; design round with the user; **test-architect BEFORE any carve** that moves a production path; surgical main-agent writes; branch + ff-merge; 13-tree gate + battery + ONE sphinx build + `dead_references` at exit).

### 22.1 ▶ NEXT ACT — #448: the eigenvalue finalize returns a flux reconstructed from a P0-only source at every `scattering_order`

**✅ LANDED `6379e9ab` (2026-09-07, `main` = `origin/main`; #448 CLOSED; branch deleted).** Exit on the final tree: 13 of 13 trees rc=0 — **11 281 passed / 19 sk / 66 xf / 0 failed** (= step 5's 11 182 + 103 new rows − 4 retired API-smoke tests); the gate module 95 + 8 primitive rows green (10 red → 0 on the pre-carve rows); battery 12 of 12 arms bite (honest 0; positive control 60; `scratch/_448/battery_r2*.status`); pyright 0 on `orpheus/`; `sphinx -E -W` rc=0; `dead_references` 0 / 68. Log copied aside as `scratch/_448/_448_full_gate_FINAL.log`. The record of the act is the rest of this section (ground facts → refutations → rulings → the landed state → the review round); **the next act is §22.2 (#425)**.

**Goal (domain terms):** the angular flux `solve_sn` returns is the converged solve's OWN flux — reconstructed from the source the converged iteration actually used, at every scattering order — never from a weaker source than the one the eigenvalue was computed with.

**Ground facts `[M]` 2026-09-05 at `950f7c9b`:**
* The finalize (`orpheus/sn/solver.py:2577-2579`) builds `Q_final = compute_fission_source(φ, k)`, then `solver._add_scattering_source(Q_final, φ)` and `solver._add_n2n_source(Q_final, φ)` — P0 in both channels, lifted by `AngularSourceSink.from_isotropic`; the ℓ ≥ 1 part of BOTH channels is dropped whenever `scattering_order ≥ 1`. `_build_aniso_scattering` (`:2328`) has **0** production callers tree-wide (tests only). Pre-existing (filed 2026-09-04 by the #426 elegance review; not a regression of any carve), widened by #426 to two channels.
* **Step 5 made the fix a RETIREMENT, not a patch.** The three helpers (`_add_scattering_source :2320`, `_build_aniso_scattering :2328`, `_add_n2n_source :2357`) are now twin paths of what the bound `ScatteringOperator` / `N2NOperator` (`AngularLift`s, one body each, every order) already do on every inner solve. Proposed means (hypothesis, not yet designed): the finalize applies the record's bound `S` and `N₂ₙ` to the converged angular iterate (or their `isotropic_energy` bindings to `φ` where the solve was isotropic) and the three helpers retire with their tests migrated — `coding-standards` "retire as you go"; the test migration set is `[M]` **6 direct call sites** in `tests/sn/operators/test_solver_components.py` (`:159`, `:170`, `:191`, `:630`, `:694`, `:701`; the two at 694/701 are a timing harness). Open design question for the round: the finalize's operand — the converged solve holds the ANGULAR iterate (`solver._psi_typed`, System-A member on a carrying mesh), so the ℓ ≥ 1 source can be formed from it directly, which is the honest fix; the P0 legs of S/N2N on `φ` alone reproduce today's (wrong) source exactly and are the natural bit-identity CONTROL for the carve.
* **Behaviour change, by design:** `Solution.angular_flux` MOVES for every `scattering_order ≥ 1` case (a correctness fix ⟹ principled re-baseline per `vv-principles`' three criteria, never a tolerance relax). Snapshot artefacts that may pin it (`[R]` — check which KEYS each pins before the carve; `test_dd_regression.py` names `angular` once at `:34`, the octant generator `_generate_2d_octant_snapshots.py` writes an `angular_flux` key): `2d_2g_p1_aniso_dd_8x4_het_si.npz`, `slab_2g_p1_aniso_dd_n20.npz`, `sphere_2g_p1_aniso_dd_n20.npz`, `2d_octant_equivalence_05_qaniso_mixedBC_2g_het_LS4.npz`. `k_eff` and `φ` must NOT move (they come from the converged iteration, which this block does not touch) — that invariance is the carve's first gate; the second is that the returned ψ's own angular moments reproduce the converged φ (today they cannot at L ≥ 1); the third is an L ≥ 1 MMS/closed-form case whose returned ψ is currently wrong.
* This is an operator-algebra carve crossing solver ↔ operator (the MUST test-architect trigger). Test-architect first, then a `numerics-investigator`-free carve (the defect is structural, not a wrong number).

**`[M]` 2026-09-05, the issue's first act — the SIZE, `scratch/_448_probe.py` → `_448_probe.log`** (the Be-reflected U slab of `test_be_reflected_n2n_anisotropy.py` §0: 421 groups, GL-8, 40 cells, vacuum; `keff_tol 1e-9 / flux_tol 1e-8 / inner_tol 1e-10`; SI inner, not windowed, not coupled). Three fluxes compared cell-average, `max|Δ|/max|ref|`:

| `scattering_order` | returned vs converged iterate | honest one-step `M⁻¹(q_F(φ,k) + Σ gains·ψ_conv)` vs iterate | `∫returned dΩ` vs reported φ | `∫honest dΩ` vs φ |
|---|---|---|---|---|
| 0 (control) | 3.3e-10 | 1.7e-10 | 1.6e-10 | 3.5e-10 |
| 2 | **8.8e-2** (fast groups up to 1.8e-1; per ordinate 4–9 %) | **1.2e-10** | **3.4e-2** | 3.2e-10 |

⟹ the finding is REAL and large, not a dead code path: today's returned ψ is not the converged solve's flux at L ≥ 1 and its own moments do not reproduce the φ it ships beside. The honest reconstruction — the fixed-source windowed arm's spelling at `solver.py:4016-4032`, one source-iteration step through the driven `gains` — reproduces the converged iterate to inner tolerance. `k` is untouched by construction (`1.091199656654` = the fixture's `_K_POST_CARVE_L2` to 9 decimals).

⛔ **Three claims above, written 2026-09-05 at `950f7c9b`, REFUTED at pick-up the same day (plan-authoring §2 — each is a census over the wrong population):**
* *"`[M]` 6 direct call sites in `tests/sn/operators/test_solver_components.py`"* — `[M]` **7** in that file (`:743` is a second profiling harness), AND the census was ONE FILE: the migration set spans `test_scattering_operator.py` (14 mentions incl. the `p1_build_aniso_source` pre-T3 snapshot pin at `:1554-1604`), `test_mms_aniso.py:5`, `test_isotropic_scattering.py:9-10`, `test_frame_conjugate_carve.py:414-416`, `tests/numerics/test_iteration.py:731`, `test_harmonic_moment_flux.py:593`, `test_legendre_moment_scattering.py:14/:228`, `_capture_pre_t3_snapshots.py`, the MMS message string `orpheus/derivations/continuous/mms/sn.py:2421`, **5 theory pages** (`slab_multigroup.rst` ×13, `slab_one_group.rst:543`, `solver.rst:30/:145`, `verification/sn.rst:680/:1795`, `error_catalog.rst:59/:1058`) and the `.. implements:: scattering-aniso-composite :by: TransferOperator.build_aniso_source` declaration at `operator_algebra.rst:4043`. The exclusion family's FOURTH member (2026-09-02 file / 09-04 package / 09-05 definition-file-only / now ONE test file).
* *"`test_dd_regression.py` names `angular` once at `:34`"* — `[M]` **0** hits at HEAD.
* *"`2d_2g_p1_aniso_dd_8x4_het_si.npz`, `slab_2g_p1_aniso_dd_n20.npz`, `sphere_2g_p1_aniso_dd_n20.npz` may pin `angular_flux`"* — `[M]` none carries an angular key; over all **36** `.npz` under `tests/`, **14** carry one, and every one is a bare-sweep or operator artefact (the 6 `2d_octant_equivalence_*` from `run_sweeps`, `pre_t3_snapshots.npz`'s `p1_apply_angular_flux`, 7 `bc_equivalence_*` traces). **Zero frozen artefacts pin the eigenvalue finalize's ψ** — the re-baseline half of the done-when is void; the red half must come from a NEW gate.
* And two present-tense-false docstrings the carve owes: `TransferOperator.add_iso_source` (*"the legacy in-place seam the SI driver feeds (`[M]` 345 production calls from `sn/solver.py`)"* — `[M]` 1 production caller, the finalize helper) and `_reflect_outflow_into_inflow` (*"the DIRECT fixed-source SI loop … and the final eigenvalue reconstruction sweep … call this helper"* — `[M]` 1 production call site, the finalize at `:2636`; the fixed-source path routes through the drivers).

**Precedent found at pick-up (§1's PRECEDENT clause, read before designing):** `_within_group_si`'s own docstring returns `base_implicit` and `gains` *"needed to rebuild the converged source for that reconstruction"*, and `_solve_fixed_source_si` DOES it (`:4016-4032`: `rhs = q + Σ gain.apply(ψ)`, one `base_implicit.inverse().apply(rhs, initial_guess=ψ)`). The eigenvalue finalize is the twin that hand-rolls the source instead. The eigenvalue solver discards the `(base_implicit, gains, windowed)` triple its inner solve built (`_gains` unused at `:2041`).

**Done when:** the three helpers are gone, `test_solver_components.py`'s six sites are re-keyed to the bound operators, and a gate at the solve tier shows the returned ψ at `scattering_order = 1` agreeing with an independently-reconstructed ψ from the converged iterate (the pre-fix value being the gate's first red). `Closes #448`.

**⛔ The done-when above is superseded as written** (2026-09-05, at the design round): "six sites" was one file's census (see the refutation block), and the bound operators are not what the pins re-key to — the surviving vocabulary is the channel FIELD's `transfer.add_p0_source` (P0, in place), the angular end's `_redistribute_ordinates` (ℓ ≥ 1, `(1/W)·kernel`) and `apply`. **Ruled done-when:** the five symbols are gone (`_add_scattering_source`, `_build_aniso_scattering`, `_add_n2n_source`, `TransferOperator.add_iso_source`, `build_aniso_source`), every test pin in the memo's §6.2 table is re-keyed or retired by claim class, `tests/sn/solve/test_eigenvalue_finalize_reconstruction.py` is green on all 45 rows (10 red pre-carve), the 13-tree gate + the battery pass, `sphinx -W` + `dead_references` clean, the docs' finalize prose re-tensed and ERR-083 minted. `Closes #448`.

**[R] (user, 2026-09-05) — three rulings at the design round:**
1. **Option B — the finalize IS one source-iteration step**: `ψ_final = M⁻¹(q_F(φ_conv, k_conv) + Σ Nᵢ·ψ_conv)` through the splitting the LAST inner solve drove; the hand reflect / `prescribed_inflow` / corner-seed block retires (B is a gain, as in every inner solve). Recommended over A (swap the source only) because A keeps the finalize a hand-rolled twin of the SI step.
2. **Retire BOTH operator verbs** (`add_iso_source`, `build_aniso_source`) — twins of `isotropic_energy.apply` / `_redistribute_ordinates`; the pins re-key; the `.. implements:: scattering-aniso-composite :by:` at `operator_algebra.rst:4043` RE-POINTS to `TransferOperator._redistribute_ordinates` (the block's prose is a character-for-character description of it).
3. **`_reflect_outflow_into_inflow` MOVES to `tests/sn/_test_helpers.py::reflect_outflow_into_inflow`** — 0 production callers under B; the sweep-tier gates' own inter-sweep −B idiom (7 files, 14 sites); production keeps only the two boundary cores it wraps.

**Landed on the branch `fix/sn-eigenvalue-finalize-448` (2026-09-06, uncommitted at this writing — trust git):**
* `orpheus/numerics/iteration.py` — `lagged_source(q, gains, ψ)` + `fixed_point_step(inverse, gains, q, ψ)`: the splitting map `G(ψ) = M⁻¹(q + Nψ)` named once; `SourceIteration.solve`'s loop assembles its rhs through `lagged_source` (the inverse apply stays in the loop: the ρ-honest stop reads the rhs BEFORE the sweep it would trigger — no wasted sweep).
* `orpheus/sn/solver.py` — `DrivenSplitting(system, implicit, gains)` recorded by BOTH inner solves beside `_psi_typed` (`self._driven`; recorded rather than re-selected because `_within_group_si` would hand a Krylov inner on a 2-D Cartesian mesh moment-bound gains for a full-angular iterate); `_eigenvalue_driver_source(fission_source, sn_mesh, *, coupled, context)` — the ONE eigenvalue-rhs construction (SI inner, Krylov inner, finalize; the coupled arm folds the FISSION source alone into the ψ½ seed, as the inner does — `[M]` the test-architect's constructibility probe read 5.2e-11 – 6.6e-11 on sphere/cylinder with that fold); the finalize = `fixed_point_step(driven.implicit.inverse(), driven.gains, _eigenvalue_driver_source(Fφ/k, …), converged)`; the three helpers gone; the reflect helper gone (moved).
* `orpheus/transport/operators/transfer.py` — the two verbs gone; `_redistribute_ordinates` carries the `:eq:pn-scatter` derivation pointers.
* Tests: the memo's §6.2 migration applied (28 code sites, 3 API-smoke tests deleted, `test_returns_none_for_p0` re-posed onto `is_isotropic` + `_redistribution is None`, the pre-T3 snapshot pin re-keyed to `_redistribute_ordinates` with its `nulp` bound unchanged, `TestAddScatteringSource/TestAddN2NSource` → `TestP0ScatteringEmission/TestP0N2NEmission`, two test names de-verbed); the 7 reflect consumers re-import from the test helper.
* `[M]` after the carve: the gate module **45 of 45** (10 red → 0); the 18 migrated/consumer files **412 passed / 0 failed** (94 s); the probe: returned ψ **bit-identical** to the honest step at both orders, **1.2e-10** from the converged iterate at L=2, k `1.091199656654` / `1.158712037137` unchanged to 12 digits.

**`[M]` the test-architect's memo (`scratch/_448_verification_plan.md`) refuted two premises of the brief and found two hazards worth carrying:**
* ⛔ *"`history.balance_defect` already reads large at L ≥ 1"* — REFUTED: `_exit_balance_defect` returns `None` whenever `record.fully_converged`, so the diagnostic is structurally silent on every converged solve (`[M]` `bal=None nwarn=0`, 12/12 rows) — the hole #448 lived in; on a TRUNCATED exit it is CORRUPTED instead (pinned at a #448 floor: 7.4865e-02 → 7.4847e-02 over `max_outer` 3 → 12 at L=1, vs 1.25e-05 → 8.6e-12 at L=0). Two guards whose complements did not cover the exit.
* ⛔ A manufactured Σ₂ₙ NOT balanced into Σ_t makes φ differ from `∫ψ_conv dΩ` by an EXACT global 1.100212 — the L=0 control reds too; every `xs_library` mixture ships `Sig2 = 0`. Any fast (n,2n) fixture must balance `SigT` (the module's `_balanced_n2n`).
* Battery arm **M5** (skip the finalize's reflect) is a MEASURED NULL — 2.0e-13 / 2.3e-15 / bit-identical on vacuum; the catcher for the boundary path is **M5b** (a WRONG B: G1 5.2e-12 → 2.8e-01). **M9** (ℓ ≥ 1 removed tree-wide) reddens 0 G1 rows ⟹ G1 measures CONSISTENCY, never PRESENCE (`TestTheLGe1TermIsLive` measures presence).
* **R3 — `[R]` (user, 2026-09-06): KEEP G3c as an INSTRUMENT test.** `TestTheShippedDiagnostic::test_the_balance_defect_responds_to_the_outer_budget` asserts the diagnostic TRACKS truncation (a response claim about the instrument), never that a solve converged — outside the #340 N5 prohibition's intent ("do not threshold it; do not assert on its magnitude"), and its docstring says so. **R1's owed trace gate** (B changes the returned trace's provenance; G1 is bulk-blind — N1 measured) and the **P1…P4 post-carve battery arms** re-keyed onto the fix's surface (`fixed_point_step` / `driven.gains`) — the test-architect resumes for both after the review round.

**The review round (2026-09-06), all landed on the branch:**
* **elegance-enforcer** — approved the core ("`fixed_point_step` is the master standard met"); 2 blocking + 6 should-fix, all applied: `_solve_fixed_source_si`'s windowed arm open-coded the new step (now `fixed_point_step` — its second consumer); `DrivenSplitting` + `_psi_typed` → ONE record `InnerSolve(system, implicit, gains, iterate)` declared `None` in `__init__` (the string-keyed `getattr` reads gone; the write-only `_boundary_flux` retired); `_eigenvalue_driver_source` derives the coupled partition from `sn_mesh.radial_characteristic_field_space` (no `coupled` flag — the partition `_build_fixed_source_rhs` reads); `final_ray = _system_b_member(final_state)` (the reader spells the partition); the moved helper shed `faces=`/`radial_characteristic=` (`[M]` 0 of 14 sites passed either); 13 duplicate/mangled imports (one a column-0 multi-line import my blanket rewrite injected inside a method body) collapsed; `Sequence` from `collections.abc`. Filed **#453** (a `SupportsSeededInverse` Protocol — the `inner.implicit.inverse()` union is not a `SupportsSeededApply` of the iterate union; a `cast` naming #453 stands at the finalize).
* **qa** — the algebra complete and correct on all FIVE arms incl. the un-briefed boundary-G-S (`ScheduledInvertibleOperator` + `SNMaskedBoundaryOperator` gains; G1 1.93e-11); the returned trace `inflow == B·outflow` to 2.0e-11 and the ψ½ ray to 2.5e-11 (both declared-blind surfaces measured right); the retired admission guard SUPERSEDED by `admit_composite` (stronger, on the only production route); six findings: F-1 no G-S self-consistency witness (`solve_sn`'s default schedule is **jacobi**, not G-S — the gate module never produced a G-S splitting: 0 of 161 inner solves); F-2 the owed trace gate; F-3 ERR-083 undeclared ⟹ `-W` FAILS (per-marker nexus warning; refuted its own "silently dropped" hypothesis by reading `merge.py`); F-4 the docs; F-5 import debris; F-6 no direct test of the two public primitives. Corrected the memo's band MECHANISM (`[M]` `flux_tol` 1e-6…1e-9 all give `n_outer = 10`; at 1e-11 the polish falls 49× — the OUTER term dominates at `inner_tol = 1e-11`; the polish is a step function of `n_outer`) — landed as `vv-principles` #13's fourth disguise.
* **archivist** — ERR-083 minted with before/after tables it MEASURED itself (a `git worktree add … HEAD --detach` IS the pre-carve tree while the carve is uncommitted); a new theory section `sn-finalize-one-step` in `solver.rst` (+ `sn-finalize-map`, `sn-exit-balance-projection`); the `:by:` re-pointed; **30** retired-symbol doc sites re-pointed/tensed (0 xref roles survive; past-tense history kept); the #448 SN changelog row; `sphinx -E -W` rc=0 (baseline was rc=1: the dead `:by:` + 4 phantom `catches`); `dead_references` **0 dead / 67 checked**. Its §6 flagged 4 production docstrings still lying — applied: `reflect_inflow_inplace` has **0 production callers** (the G-S resolvent binds `SNMaskedBoundaryOperator.reflect_rows_inplace`, never it — my own new sentence was false), so it stays as the operator's mutating verb for the sweep-tier gates with a truthful docstring; `reflect_corner_inplace` had **0 callers anywhere** → RETIRED; `_solve_fixed_source_si`'s docstring gains `(S, N₂ₙ, B)` and the N₂ₙ channel attributed to `N2NOperator`. Filed #454 (19 pre-existing nested-markup hazards on 8 pages) — ⛔ a DUPLICATE of #422/#424, closed 2026-09-07 with its per-file census moved to #422 (filed without checking the open list; Cardinal Rule 4 hygiene).
* `[M]` after the round: gate module 45/45 (pre-test-architect-resume); 20 migrated/consumer files **441 passed**; boundary neighbourhood 187 passed; probe bit-identical; pyright **0** on `orpheus/` (touched test files: 0 errors on this carve's hunks; the ~1 400 pre-existing `tests/` errors are cp/cross_method); a bare `npx pyright` OOMs node (memory `reference_pyright_lsp_rooting`).
* In flight: the test-architect's resume (F-1 `cart2d_gs` arm, F-2 trace gate, F-6 primitive tests, the module header re-tense, the band-mechanism correction, the P1…P4 battery arms re-keyed onto `fixed_point_step` / `solver._inner`). Then: the 13-tree gate (`scratch/_448/gate_driver.sh`), the battery, ONE more sphinx build + `dead_references` (my post-archivist edits touched `cartesian_multid.rst` and retired a method), commit, ff-merge.

### 22.2 Then — #425: 37 SN-chapter sites still spell `A = L + C − S − B` (pre-§14.1)

A docs pass, per-page numerics adjudication (which pages state the GENERAL algebra vs a Σ₂ₙ-free special case) — an **archivist** task with the issue's census (`[M]` 2026-08-31 regex over `docs/theory/methods/sn/*.rst`, 37 sites) re-run FIRST at HEAD with a positive control (the count is 5 days and one merged step old). A present-tense-false algebra is a bug (the articulation lens's enforcement half); it stays a separate act so its adjudication is not smuggled into #448's carve. ONE sphinx build; `dead_references` after.

### 22.3 Then — step 6 of the ladder (§8): the CS2 residue

S3 bridge retirement + densifier-native (§7.2/§7.3); the L/B minimal annotation flips; the R6 carrier guard at `boundary.py:714`; ledger rows R1-L, R1-B, R6 flip in `tests/sn/architecture/test_monomorphic_leaves.py`. ⚠ from §8's own row: three boundary classes carry Optional; O-3's 4-tuple and R18's B reshape are NOT pre-empted. Opens with a design round (the F-F lean "late, step 6" is now due) and the explorer's re-census of the boundary leaf surfaces — every count in the CS2 charter dates 2026-08-20/21.

### 22.4 Then — the coda (§8): the homogeneous path re-points last (F5), the `from_materials` consumer dissolves, O1 completes; `InfiniteMedium` (R23) stays post-ladder.

### 22.5 Then — the consumers / solution-path campaign (§5, deferred by the user's 2026-08-30 ruling): typing the k-outer iterate, the solver-seam adapters, every consumer reshape; the §5b fork + the O-3 split at the pencil boundary. Chartered AFTER the ladder closes; not before.

### 22.6 Standing constraints (all of §20.3 + §21.5), plus three learned at step 5's close

* **A patch script's self-check is a CENSUS and owes its population** — `count == 2 in solver.py` is not `count == 2 in the tree` (the exclusion family's third member, plan-authoring row 2026-09-05).
* **A battery arm whose mutation breaks a solve's convergence needs a per-test cap** — the driver carries `pytest-timeout --timeout=240`; uncapped, B5.6 and B5.9 crawled to `max_inner` (30+ min at one row); capped, they finished (167 and 108 reds, 8 timeouts counted as reds).
* **`pgrep -fl sphinx-build` before ANY Sphinx build** (L63), and the 13-tree driver TRUNCATES its log — copy a gate log aside before re-running the driver if its numbers are still owed anywhere.
* Residue issues from step 5 stay open and are NOT in this order: #452 (C6 static pins adjudicated by nothing), #451 ((n,2n) Legendre closed form), #449, #446, #444.

## 23. ⏸ COMPACTION POINT #8 (2026-09-07, written after the #448 merge, tree clean, no gate running) — #448 landed; the next act is #425

**State `[M]`:** `main` @ `54125133` = `origin/main` (`6379e9ab` the carve, `54125133` the plan close-out); #448 CLOSED; branches deleted; working tree clean (only `scratch/` untracked + the foreign `.claude/plans/harness_context_budget.md`); no pytest / sphinx running. **#448's whole record is §22.1** (ground facts → refutations → the size measurement → the three rulings → the landed state → the review round → the ✅ LANDED banner with the exit: 13 of 13 rc=0, 11 281 / 19 / 66 / 0; battery 12 of 12; pyright 0; sphinx `-W` rc=0; `dead_references` 0/68). The Nexus graph was built on the final tree by the exit build (`docs/_build/html`); `54125133` touched no indexed source. The ruled order (§22, `[R]` user 2026-09-05) stands: **#425 → step 6 → coda → consumers campaign.**

### 23.1 ▶ NEXT ACT — #425: the SN chapter states the within-group algebra as the tree spells it

**Goal (domain terms):** every page of the SN chapter that states the GENERAL within-group algebra spells it as the tree has posed it since §14.1 — `A = L + C − S − N₂ₙ − B`, the (n,2n) gain a first-class term — and every page that keeps the 4-term form does so as a DECLARED Σ₂ₙ-free special case, never as the general claim. (Articulation lens: a present-tense-false algebra on a theory page is a bug.)

**Ground facts `[M]` 2026-09-07 at `54125133`** (`re` over the 15 pages of `docs/theory/methods/sn/*.rst`; positive control: the 5-term form found 11×):

| page | 4-term, ASCII `-` (the issue's regex) | 4-term, unicode `−`/`–` (INVISIBLE to the issue's regex) | 5-term (control) |
|---|---|---|---|
| `slab_multigroup.rst` | 8 | 0 | 4 |
| `slab_one_group.rst` | 6 | 1 | 0 |
| `cartesian_multid.rst` | 5 | 0 | 0 |
| `curvilinear_multigroup.rst` | 5 | 0 | 0 |
| `solver.rst` | 5 | 1 | 0 |
| `curvilinear_one_group.rst` | 3 | 0 | 0 |
| `index.rst` | 3 | 0 | 2 |
| `history.rst` | 2 | 1 | 1 |
| `acceleration.rst` / `adjoint.rst` / `boundary_conditions.rst` | 1 / 1 / 1 | 0 | 0 / 3 / 0 |
| `curvilinear_numerics.rst` | 0 | 1 | 0 |
| **total** | **40** | **4** | 11 |

* ⚠ The issue's own count (`[M]` 2026-08-31: **37**) is stale in BOTH directions: the ASCII population GREW to 40 (three merged passes — step 5, the #448 docs pass — wrote fresh 4-term spellings; the archivist's #448 pass touched `solver.rst`/`slab_multigroup.rst`/`cartesian_multid.rst`, and `history.rst`'s new #448 row may be one), and the issue's regex `L\s*\+\s*C\s*-\s*S\s*-\s*B` cannot see the 4 unicode-minus spellings at all (plan-authoring §2's FILTER clause: a filter is validated against a known member of every SHAPE — the corpus spells minus three ways). The honest population is **44 sites on 12 of 15 pages**; the done-when regex must carry `[-−–]`.
* The scope rule the issue states and this act executes: per site, does the page state the GENERAL algebra (→ rewrite to the 5-term form) or a deliberately Σ₂ₙ-free fixture / special case (→ the one-line scope note; the 4-term form is then numerically exact for the case at hand)? The chapter root (`index.rst` machine header + `.. note::`) and the multigroup Key Facts already DECLARE the simplification, so no reader inherits the 4-term form as the general claim today — this act makes each page self-consistent.
* `history.rst`'s and `error_catalog`-style past-tense mentions ("the algebra was `L+C−S−B` until §14.1") are HISTORY and stay (coding-standards: discriminate by tense) — the census cannot tell tense; the adjudication is by reading.
* The step-4 census memo `scratch/cs4c_step4_corpus_pass.md:150` carries the original command (untracked `scratch/`; the table above supersedes it).

**Proposed means (archivist task — ruled a separate act so its adjudication is not smuggled into a carve):** dispatch the archivist with the table above as the population, the tense/scope rule, the validated regex (`L\s*\+\s*C\s*[-−–]\s*S\s*[-−–]\s*B` + the 5-term positive control), and the instruction to re-run the census FIRST and report its own count before editing; ONE `sphinx -E -W -q` build from the repo root (`pgrep -fl sphinx-build` first) + `mcp__nexus__dead_references` at exit; the regenerated `matrix.rst` committed with the pass, never hand-edited. Branch `docs/425-within-group-algebra`, ff-merge, `Closes #425`.

**Done when:** the validated regex over the 15 pages returns zero sites OR every remaining site is on the same line as / within 3 lines of a scope annotation the archivist names (state the annotation's spelling so the census can grep it); the 5-term control still ≥ 11; `sphinx -W` rc=0; `dead_references` 0 dead. Predicted delta to the test count: **0** (docs only) — a harness gate parametrised over documented labels could move it by the number of NEW equation labels the pass adds (the 2026-08-30 +1 row); reconcile, do not shrug.

### 23.2 Then — step 6 of the ladder (§8, the CS2 residue) — as §22.3; opens with a design round + the explorer's re-census of the boundary leaf surfaces (every CS2 count dates 2026-08-20/21; #448 retired `reflect_corner_inplace` and moved the reflect helper, so the boundary surface has MOVED since).

### 23.3 Then — the coda (§22.4) and the consumers / solution-path campaign (§22.5), unchanged.

### 23.4 Standing constraints (all of §22.6), plus five learned at #448

* **A bare `npx pyright` OOMs node here** (no `include` in the generated config; ~12 k `.py` incl. the nexus worktree's own `.venv`) — the ratchet is `npx pyright orpheus` (0) plus the touched files explicitly, with test-tree errors attributed to your hunks by `git diff -U0` (memory `reference_pyright_lsp_rooting`).
* **A battery mutation that makes the PROBLEM divergent is unreadable** — `_reflect_trace × 2` on a reflective eigenvalue problem reddened everything as "did not converge"; only `× 1.001` read the WIRING. Scale a shared body, never double it; and cap every arm (`--timeout=240`) — P5a crawled 30+ min uncapped.
* **A sub-agent watchdog stall mid-correction leaves a FALSE claim on disk** (the `catches("ERR-083")` marker on the ψ½ class, caught by the agent's own mutation and cut off before the fix) — on a stall, resume the SAME agent on exactly the sentence it was correcting; do not re-run the gate and read green as done. Filed **#455** (a `catches` marker has no executed adjudication; `_phantom_verifies` exists, `_phantom_catches` does not).
* **File an issue only after `gh issue list --search`** — #454 duplicated #422/#424 (closed, census moved).
* **The `[R]`-then-verify discipline paid twice in one act** (plan-authoring row 2026-09-05): the `[R]` claims in §22.1 were checked first at zero cost; the `[M]` one-file census was the one that lied. Write `[R]` when you have not run it.
* Residue issues stay open and are NOT in the order: #455, #453, #452, #451, #449, #446, #444.
