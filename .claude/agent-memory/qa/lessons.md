# QA Lessons — hot digest

Read every dispatch. **Behavioral rules only** — imperative, standalone.

- **War stories, evidence, `file:line`, measured tables** live in
  `lessons_archive.md` (`## L-0NN`, ascending, L-001..L-072). Open it only for
  the exact `L-0NN` a rule points at. Never read it whole.
- **Doctrine is NOT restated here.** `vv-principles` (preloaded) owns
  anti-patterns #1–#17, Modes 7–12, the bit-identity criteria, 1-group
  degeneracy, the `catches` directive; `numerical-bug-signatures` owns Sig 1–10
  + H1–H5; `qa/AGENT.md` owns make-it-RED (#11) and the field-role contract
  (#10). A `[skill]` tag means the principle is already there — what is kept is
  the ORPHEUS mechanic or the procedural trap. §I is the map.
- New lesson ⟹ append `L-0NN` to the archive, land a 2–5 line rule here.
  Sharpen in place; never let this file grow narrative.

---

## A. Making a gate RED — mutation mechanics

**A1. Disable the OVERRIDE, not the value, when two paths are value-equal.** A
specialised `apply` overriding an inherited leaf-sum can agree to ≤2 ULP, so only
`array_equal` discriminates; rename the override away and every tooth must red. → L-024

**A2. Revert PRODUCTION ONLY (keep the new tests) to prove a fix's negatives
could have failed.** `git stash push -- <production files only>`; the red message
must NAME the original bug, not just `AttributeError`. → L-027

**A3. Mutate in-process (throwaway pytest plugin / monkeypatch); revert by
RE-EDITING** — never `git stash`/`checkout` a path with uncommitted state.
Untracked files make `git diff` empty, so the revert proof is gate-green-again +
zero mutation markers. A `-p <module>` plugin needs `PYTHONPATH`. → L-039, L-043, L-052

**A4. Your OWN mutation needs a bite check.** [skill: Mode-8 METHOD WARNING]
Residue: a capability REFUSAL is TWO-part — adding `apply_transpose` does not
lift `is_adjointable`/`is_invertible` (predicates defaulting False). And a
**0-call counter is a FINDING** (path unreachable), not an inert mutation. → L-061, L-062

**A5. When `simplify` is pathologically slow on the MUTATED expression, don't
wait on pytest** — call the `derive_*` builder with concrete Rationals and read
the residual directly. Seconds, and decisive. → L-029

**A6. Cripple a GENERATOR, not a value, for the sharpest coverage verdict** —
replacing `O_h`'s 48 ops with its 8 diagonal sign-flips (= `D_2h`) left a
182-test suite green. → L-062

**A8. Mutate INSIDE the object's algebraic class, or the reds are catching the
LAW you broke.** [skill: #18] A CONSTANT written into a linear operator's output
makes it affine → 60 Krylov/SI gates red; the realistic LINEAR bug (same 94k
rows) red exactly 1 of 5076. Ask of every red: *by what mechanism does THIS gate
see THIS property?* Over-power lies "richly caught" — the flattering direction. → L-063

**A7. ONE mutation direction is almost never enough — enumerate the leaks and
red each.** (a) *Capability default-OFF*: the factory AUTO-SELECTS the wider
shape, OR appends a PHANTOM length-1 axis (control:
`not hasattr(space,"factors")`). (b) *`xfail`→live flip*: red against the
re-introduced bug AND the EMULATED PRE-change behaviour — only the latter rules
out a gate already green at HEAD. (c) *Polymorphic hook*: override returns the
base type, AND override DROPPED (base `replace()` keeps state, so only the
empty-state tooth reds — and only if the test ADVANCES first). → L-032, L-038, L-041

**A10. Run the mutation over the WHOLE module tree in BOTH arms — the
symmetric difference turns "does an external pin exist?" from an argument into
a LIST.** `[M]` old-τ vs HEAD over `tests/sn`: 7 red only at HEAD, **32 red
only under old-τ**, 9 red in both (another agent's scope). The 32 named the
analytic-closed-form pins I had just concluded did not exist — my own draft,
refuted by my own measurement. Bite check first (the target gates must FLIP,
with a non-zero call count). → L-069

**A11. Two DUALS of A9, both flattering.** (a) **A normalized fingerprint reused as a
CHANGE detector inherits its deliberate BLINDNESSES** — `[M]` nexus `body_shingles` is
bit-identical under `rtol 1e-12→1e-6`, `max_inner 1000→50`, a re-baselined expected
value and a fixture-arg swap (it normalizes `Constant→"C"` for clone robustness), i.e.
blind to every Mode-8-class-7 decay cause; a ledger on it reports every decayed marker
FRESH. Intersect the fingerprint's invariance group with the change class — Mode 12,
asked of an INSTRUMENT. (b) **A recall counter placed DOWNSTREAM of a filter cannot
count what the filter dropped** — `[M]` `nexus runtime-ingest` printed
`nodes: 0 / unresolved: 0`, exit 0, on a real report, because all 339 file keys were
dropped by a path filter before reaching the resolver (absolute-vs-relative); the same
artifact joined **2892** nodes once normalized. Demand a per-REASON drop breakdown. → L-070

**A9. An AUDIT instrument needs one control PER STATE its predicate accepts.**
[skill: Mode-8 METHOD WARNING] A production predicate reused as a detector
inherits its OTHER meanings — `_claims_convergence` is also False for an EMPTY
history (= GMRES exited in 0 iterations = CONVERGED), so 44 of 90 census rows
were invented, in the flattering direction, while the positive control passed
(it only exercised the genuine branch). → L-067

**A12. A guard HOISTED to one shared home has as many arms as it has CALL
SITES — mutate per site, not per branch.** [skill: vv#17 granularity, whose
worked example is in-body arms only] Pattern 2 single-sources the guard BODY;
it does not single-source the WIRING, and each site passes its own operands
(`self.mat_xs.ng` at three sites, `self.coefficient.values.shape[0]` at the
fourth). The hoist therefore CREATES the blind spot the elegance rule is
rewarded for. `[M]` CS4a `assert_energy_extent_conforms`: disabled per site
over `tests/{transport,homogeneous}` + the ledger — F **1** red, C **0**,
IsoS+IsoN2N **0**; the fragment `"energy extent"` occurs in exactly ONE
assertion tree-wide. Also do the DUAL grep: count call sites, then count the
distinct expressions they pass. → L-074

**A13. Read the red set by IDENTITY, not by SIZE — reds == the NAMING set ⟹ no
consumer, and the pins are a mirror.** [skill: landed as a ⭐ on vv#17]
#18 does NOT catch it: *"by what mechanism does this gate see this property?"*
answers fine ("it reads the field directly") — the pin is not blind, there is
nothing DOWNSTREAM of the value. Check = set-diff red set vs
`grep -rln "<sym>" tests/`. `[M]` flipping BOTH dead
`ReducedStreamingOperator` fields on **997** operators over 2591 rows → **6
red, and they are the 6 assertions that name them**. Two mechanics: patch
EVERY rebinding site (a package `__init__` re-export kept the ORIGINAL and
halved my reds), and carry a call counter or *no consumer* ≡ *no bite*.
⭐ Companions: a field's test-hit count can be all WRITES (5 of 5 hits on
`StreamingTerms.mu_start` are constructor kwargs — split READS from WRITES);
and a zero-reader field naming a REAL contract is usually RESPELLED, not dead —
ask "how does production answer this question today?" before wiring it.
→ L-075

**A14. A runtime traffic census counts BODIES EXECUTED, not arms dispatched —
and "0 applies" is A13 with the polarity FLIPPED: not dead, load-bearing as
DATA.** [skill: landed as (d) + the re-dispatcher clause on vv#29]
Two ORPHEUS mechanics. (i) A **fused parent** can override `OperatorSum.apply`
and read its operand's *field* instead of calling its *body*: `[M]` SN's C at
`coupled_system.py:446` is minted 20–25× per k-solve (once per outer) and is
**silent every time, under BOTH inner solvers** — `StreamingCollisionOperator`
overrides `apply` to use `self.diagonal.coefficient.values`. Retiring it from
the traffic verdict would be exactly backwards. (ii) A registered **arm can
re-dispatch** (`self.apply(psi.interior)`), so one call scores twice — tell:
two arms with EQUAL counts on every row; construction-time selection then
relocates the branch, it does not remove it.
⭐ Mechanics: patch the singledispatch **registry** (`dispatcher.register(typ,
w)`, cache auto-clears); `apply is _apply_impl` is usually ONE object — check
the identity, don't assume it; attribute per SITE by wrapping `__init__` +
`extract_stack`; keep instances alive so `id()` can't recycle. Controls are a
LADDER: instrument → installation-marker → ⭐ **per-ARM activation** (fire every
arm, else 8 of 23 zeros are unreadable) → headline bit-identity.
⭐⭐ Bound the workload with a STATIC reference census — `[M]` 6 files reference
the 7 roster classes, `cp`/`moc`/`mc` = 0 — so "measures its workload only"
stops being a ritual disclaimer and names the real residual (another *config*
of a driven family, not another family). → L-076

**A15. When a guard's refusal is DUPLICATED one frame down, the outer one has
no witness and mutating it "away" reads GREEN.** [companion to A12: A12 is one
body / many sites; this is one refusal / two frames] `[M]` #429: deleting
`_invariance_on_orbit_space`'s step-1 normaliser check reddened **0 of 670**
because `Quotient.induced_action` refuses the same motions — and deleting it
WITHOUT swallowing that raise reddened **16**, so the outer guard's real job
is converting a raise into a `False`, which is not what its gate asserts.
⟹ run BOTH arms (swallow / let-it-raise); the difference names the guard's
actual contribution. Same run: step 2 (`H ⊇ G ⟹ True`) is 0/670 too, and
there it is CORRECT — `orbit_coordinates` is H-invariant, so the fall-through
returns the identity permutation bit-exactly. A 0-red arm is *guard with a
twin* or *provable optimisation*; decide which by algebra, not by red count.
⭐ And the per-ARM discipline pays here: `_identity_component_normalises`
whole-body → **1** red, but its five arms → 7 / 27 / **1** / **1** / **1**,
with arm (c) invoked **once** in the whole suite. → L-077

**A16. A "brute-force CONTROL" can be the production expression α-renamed —
prove it by AST, not by reading.** [skill: #22 is the shared-INPUT case; this
is the shared-EXPRESSION case, and it is mechanically decidable]
`ast.unparse` the production statement and the test's reference expression,
α-normalise the bound variables, compare strings. `[M]` #429's
`test_the_criterion_agrees_with_a_brute_conjugation_control`: **character-
identical**, and the test's element list is production's own
`_group_elements`. Its real content is the neighbouring hand-derived
`assert brute is (axis == mirror)`; the docstring credits the tautology.
⟹ for every gate whose docstring says "control" / "brute force" /
"independent", run the two-line AST diff before crediting it. → L-077

**A17. A TWO-STAGE census filter (name-net THEN pattern) needs a positive
control per STAGE — a synthetic fixture you author passes stage 1 by its own
NAME and certifies only stage 2.** [skill: #17's positive-control clause, one
level finer] `[M]` #428: `test_n2n_multiplicity_census`'s net
`("n2n","sig2","sig_2n","_2n")` misses `sig_2` — the spelling `derivations/`
uses — so its claim *"a thirteenth literal home is unspellable"* is false;
widening the net by that ONE token finds 2 literals (`derivations/common/
eigenvalue.py:61,:290`). Its control (`:91`) is a synthetic source whose
FUNCTION is named `n2n_source_assembly`, so every arm clears the net for free
and the control validates the four literal spellings, never the net.
⟹ when a census filters in two stages, the control must include a member that
passes stage 2 and is only reachable through a stage-1 token you did **not**
think of — i.e. name the fixture with the spelling you are least sure about.
→ L-079

**A18. Scope a mutation to ONE PHASE by rewriting the RECORD that phase reads,
never by flagging the shared verb.** [companion to A12/A14; solves vv#18's
over-power problem structurally] When a finalize/reconstruction re-uses verbs
the driver also calls, a global mutation moves the CONVERGED answer instead of
the reconstruction and reports false coverage. If the phase reads a recorded
object (`SNSolver._driven`, a system record, a cached splitting), wrap the
producer and rewrite the RECORD after it returns: the driver already holds its
own real operators, so only the later consumer is mutated — no phase flag, no
call-stack sniffing, no production edit. `[M]` #448: rewriting `_driven.gains`
to drop the boundary gain reddened **8 of 14** G1 rows (exactly the 4 arms with
a live `B`) while every inner solve stayed honest.
⭐ And the same wrapper is a free CENSUS: `[M]` 161 inner solves / **0**
`ScheduledInvertibleOperator` splittings said, in one number, that the new gate
module never reaches the boundary-Gauss-Seidel arm — a `-k`-free denominator no
grep could produce. → L-080

**A19. A TOLERANCE sweep needs its ITERATION COUNT beside every row.** [skill:
`vv-principles` #13, fourth disguise — landed 2026-09-06] A tolerance is a
discrete knob: rows with equal counts are one measurement, and "this tolerance
does not move the error" usually means "none of my values changed the count".
`[M]` #448: `flux_tol 1e-6…1e-9` all `n_outer = 10`; at `1e-11` (`n_outer = 12`)
the deviation falls 49×, so the term the memo called non-binding dominates at
the shipped config. → L-080

---

## B. Where a gate is structurally blind (ORPHEUS shapes)

**B1. Mutate the SHARED source, not the dead-for-this-path method.** SWEEP and
MATVEC share only precomputed coefficients: three apply-path mutations gave
call-count 0 and identical error ladders (GREEN-BLIND on dead code). Instrument a
call-count FIRST to find which twin the gate runs. → L-036 [skill: Mode 11]

**B2. A round-trip / self-consistency test cannot pin a CONVENTION** — both arms
carry the stale input. Recurs at every scale (1-D `s_axes`, d≥2 `|μ_axis|`). → L-018, L-023, L-031

**B3. "The matvec is tested" needs an instrumented call-count, not a
round-trip.** SI sweeps never touch `loss_action`; the matvec runs only under
`inner_solver="krylov"` (measured 1600 / 0 kernel calls on an MMS solve). → L-018, L-021, L-033

**B4. A `@pytest.mark.slow` catcher is DESELECTED under the canonical
`-m "not slow"`** — a sibling of Mode-8. Never trust a plan's "test_X must red";
simulate the regression under the ACTUAL invocation. → L-053

**B5. A fixture SYMMETRIC in the axis under test cannot see that axis — make it
asymmetric.** Instances: a SQUARE `nx==ny` mesh hides axis-ORDERING, and the
algebra-law suite is swap-invariant anyway (linearity/homomorphism survive a
CONSISTENT transpose of all operands), so the catcher is a broadcast oracle on
`nx≠ny` → L-040; a UNIFORM fixture makes a per-cell and a global-mean check
indistinguishable, so one fixture must VARY along the non-reduced axis → L-030;
two SAME-AXIS faces make `|Ω·n|` bit-identical, annihilating the packing gate's
only knob-reader (`slot.slice_view(metric_flat)`) — `[M]` 0/10 red,
`changed=False` on every call; a y-face in the layout moves it 0.963 → L-065.

**B6. An A-vs-B INVARIANCE gate's coverage is the set of production lines that
READ the knob — grep them; it is usually ONE.** [skill: #23] Two runs of the same
code ⟹ blind to every non-knob-dependent mutation, so the CATASTROPHIC positive
control is INVALID (an identity kernel leaves it correctly 10/10 green); the
control must be knob-dependent — neuter the knob and the ACTIVATION leg must red.
Name the rows that structurally cannot see it (`vacuum`/`lambertian` never reach
the deck kernel) so their green is not counted. → L-065

**B7. A transpose/adjoint RECIPROCITY gate pins the transpose RELATIONSHIP, not
correctness** — green for ANY genuine `(S,Sᵀ)` pair, so Mode-12 blind to a
SYMMETRIC drop in both halves. Mutate BOTH ways and require the one-sided
`A∘A⁻¹≡I` companion; never let it be deleted on "reciprocity covers it". → L-060

**B8. The branch you are crediting may be dead under the shipped config.** A
"fires under quadrature Q" claim is a 3-line probe (`count_nonzero(|mu_x|<1e-15)`
→ zero at every LS order); likewise an instrument whose reflective arm never
runs. → L-016, L-059

**B9. A "the matrix says the operator is healthy" argument must cite a
certificate for the EXACT gated BC** — a sibling-BC certificate plus "same
mechanism" is inference. BUILD the fully-coupled matrix: `ρ_prod > ρ_matrix` ⟹
splitting/wall lag (honest); `ρ_matrix ≈ 1` ⟹ real consistency failure. → L-059

**B10. The headline category gate is usually the WEAK one.** `runtime_checkable`
checks member PRESENCE only — `isinstance` flips True on a monkeypatched attr and
stays False under a realistic PARTIAL leak. The direct `not hasattr(...)`
negatives are the defense; credit them, not the headline. → L-039, L-042, L-047

**B11. The MMS refinement ladder is BLIND to the diffusion limit — probe
`σ_t·h ≫ 1` on a COARSE mesh** (refinement drives `σ_t·h` thin, where flat-source
is fine; the failure lives where users run). Probe vs DD with an ε-scaled
diffusive material. A reflective `c≈1` probe is a TRAP — both schemes read ~82 %
wrong from non-convergence. → L-017

**B12. A CONVERGENCE flag at an eigenvalue entry is the OUTER fact only —
inner truncation under a power iteration is invisible BY CONSTRUCTION.**
`solve_sn`/`solve_sn_adjoint` warn on `max_outer`/`keff_tol`; a within-group
solve hitting `max_inner` never reaches the warning. Before crediting a
convergence sweep as complete, ask WHICH LOOP the flag belongs to, and wrap
`_certify_within_group_exit` for the other one. Companion holes: a suppressed
warning, an `xfail`-absorbed one, and `-m "not slow"` deselection. → L-067, L-053

**B14. The STATIC call graph cannot answer "did this test exercise it" — measure the
EXERCISED set with `coverage`, never with `calls`.** `[M]` 0 of 21 claiming tests reach
`Quadrature.ordinate_permutation` statically; **7** do at runtime — 100 % false-DEAD,
because every production call site is annotation-mediated (`quadrature.x()`,
`self.mesh.quad.x()`), which nexus #16 does not resolve. `nexus callers` → 0 and
`dead-functions` FLAGS the method. Recipe: `dynamic_context = test_function` +
`[json] show_contexts = True`, then join `contexts` to
`sphinxcontrib.nexus.runtime.build_node_index` spans (~15 lines; `[M]` 23/23 contexts,
1353 pairs, 1.45× runtime overhead). ⚠ It measures CO-EXECUTION, not co-constraint — a
candidate list to mutation-verify, never a licence. The rung ladder is
CLAIMED 21 → EXERCISED 7 → ASSERTED ≤2 → MUTATION-VERIFIED 0, and **no edge quality
separates rungs 2 and 3.** → L-070

**B15. A retired type's WORKAROUND IDIOM outlives the type, and it is a COVERAGE
question, not a style one — ask what error class the detour's functional
annihilates.** [skill: Mode 12, asked of the IDIOM rather than the fixture] `[M]`
CS3-R: 5 operator gates still verified linearity through the affine detour
`op(ψ₁+λ(ψ₂−ψ₁)) = (1−λ)op(ψ₁)+λop(ψ₂)` — a workaround for a restriction that no
longer exists. Affine maps PRESERVE affine combinations, so it is **exactly**
blind to an affine regression `A(x)=Lx+q`: 10-line pure-numpy probe, retired form
`4.440892e-16` at `q≠0` — *bit-identical to its own `q=0` control* — vs direct
`A(ψ₁+ψ₂)` at `1.288361e+00`. ⟹ two moves: **(a)** when the SUT tree is under
CONCURRENT edit, a pure-numpy model of the two functionals settles blindness
decisively **touching no file** — safer than a mutation battery and here equally
decisive; **(b)** verify a "we fixed it" sweep moved the **assertion**, not the
prose — `git show <old>:<f>` vs working tree. A prose-only fix leaves the
blindness wearing a corrected comment. → L-071

**B13. A published COMMAND is a separate claim from the API it wraps — gate the
STRING.** [skill: Mode-8 EIGHTH class] `-W error::ConvergenceWarning` (4 doc
sites incl. the runtime message) does NOT parse — an undotted `-W` category
resolves against `builtins`; pytest exits ERROR, 0 collected. The file's own
"it is escalatable" test passes because it installs the filter
programmatically. Gate: `_pytest.config.parse_warning_filter(s, escape=False)`.
Also grep `pytest\.xfail(` — the CALL form can never XPASS, so its deferral is
immortal and its reason string rots unfalsifiably. → L-067

---

## C. Reference contamination & structural independence

**C1. The circularity test is: does the bug live on BOTH sides?** A re-encoded
production formula vs an INDEPENDENTLY-ASSEMBLED primitive is legitimate;
re-encoded-vs-re-encoded is circular. Two ORPHEUS tells that a *corroboration*
is only PROCEDURAL: (a) the two sides ride one **antiderivative identity** — a
discrete recursion summing `f` "confirms" a claim about the exact face value
`F=∫f` only because `F'=f`, so it is true whatever the claim is; (b) the
corroborating GATE's own docstring cites, as ITS reference, the very claim being
corroborated. `[M]` Q5.6.4: `α = κ·w_gl·ξ(e_arc)` to 1e-15 was offered as
evidence for "the exact face coefficient is `ξ`", and `test_alpha_closed_form`
names "Hebert 3.399 — α IS the tangential cosine at a half-angle boundary" as
its ground. → L-029, L-068

**C2. For a WEIGHTED value-pin, L11-independence is not enough.** The hand-ref
must carry EVERY weight factor AND the fixture must make a factor-BLIND formula
give a different answer. Prove both: hand-compute the blind number, then mutate
production blind and confirm only that gate reds. → L-046

**C3. Pin a re-baselined `.npy` to a STRUCTURALLY-INDEPENDENT reference** —
never to "whatever the changed leaf emits" (circular), never to old-vs-new ULP.
When the composite is byte-identical, the anchor is `composite − collision`. → L-049, L-034

**C4. Two independent implementations IS independence;
producer-vs-its-own-projector is not.** Compare production's emitted slot against
a TEST-SIDE `leggauss` ref, then separately pin the two projectors' agreement. → L-044

**C5. Recompute the OLD contraction on a structurally-independent table to
verify an API-migration's bit-identity** — a brief's "0 ULP" is a CLAIM. → L-051, L-052

**C6. For an adjoint, the independent reference is a DENSE matrix built by
LOOPS, transposed directly, composed with metrics by hand.** Re-derive the
inner-product identity first, then prove `(A⁻¹)ᵀ=(Aᵀ)⁻¹`. An ASYMMETRIC transpose
pair is CORRECT when each transpose mirrors its OWN forward. → L-052, L-060

**C7. A two-paths oracle's analytical anchor is often TRANSITIVE and in another
file** — confirm that file is green before crediting analytical grounding. → L-028

**C8. Independence has TWO axes — DERIVATION and INPUT RESOLUTION; a
single-sourcing retirement closes the second silently.** [skill: #22] Tell: the
test builds ONE domain object and hands it to the SUT *and* the reference, where
the SUT used to resolve it from its own tag. Prove per-axis by mutating the
shared RESOLUTION (`[M]` an axis-letter x↔y swap left the "genuinely independent
routes" file 15/15 GREEN and reddened 78 siblings). Prove a helper's independence
mechanically, not by reading: `dis.Bytecode(f).codeobj.co_names` — the forbidden
names usually appear only in the DOCSTRING. → L-064

---

## D. Re-baseline & bit-identity integrity

**D12. A retirement's CONCEPT grep must cover the retired FIELD's hyphenation,
not only the retired SYMBOL's — and the gutted package's OWN sibling docstrings
are where the survivors live.** `[M]` a grep for "reflection-index table" caught
3 sites and missed 4 "precomputes the reflection-partner map at construction"
claims in the very `numerics/quadrature/` files the commit rewrote. No Sphinx
severity can see them (no `automodule` for that package) and the xref checker
tests TARGETS, not prose truth — grep is the ONLY gate, so its vocabulary is the
whole audit. → L-064

**D15. A type/concept retirement's blast radius includes `.claude/agents/*/AGENT.md`,
`.claude/skills/*/`, and `.claude/agent-memory/*/` — and AGENT.md outranks a
production docstring.** AGENT.md loads FRESH per dispatch, so a stale brief is
re-injected as CURRENT FACT into every future sub-agent and its output is
indistinguishable from a correct one. `[M]` CS3-R: 3 of 12 survivors were agent
briefs — `explorer` still teaching a 4-role grid with `FluxDisplacement` and
"`flux+flux` is a TypeError" (triple-false), `cross-domain-attacker` carrying the
**imperative** "FIX: a torsor displacement type" as the un-migrated twin of a
skill its own source had already ⛔-corrected, and `elegance-enforcer` ruling that
a deleted mixin must not be flagged. Memory is the biggest and least-swept slice
(**182 lines / ~20 files** vs **75** for skills+agents+rules combined).
⭐ **And re-audit the CORRECTED file FIRST, not last** — a partial pass fixes the
site it was looking at and leaves a SELF-CONTRADICTING file where the stale line
comes FIRST (docstring above body, module header above class body), so the file
can now be cited for either reading [vv #21 aggravator]. `[M]` `_bases.py:18`
and `:1134` record the retirement while `:1160`/`:1220` still name the deleted
classes as live role leaves. → L-071

**D13. MOVING a method to a sibling object can convert a SELF-consistency into a
cross-object coupling guarded only by EXTENT.** Ask what array the old owner's
callers relied on; if the new owner carries a COPY cross-checked by
shape/length only, a same-size-different-values pair is now ACCEPTED where it
used to RAISE. `[M]` `to_local` moved operator→space: the gather reads
`op.indices`, the remap now reads `space.ordinate_indices`. And a round-trip
gate that was harmless while ONE array existed (`searchsorted(op.indices,
op.indices[perm]) == perm`) becomes the gap the moment there are two. → L-065

**D14. Before judging a re-baseline's LEGITIMACY, `git log` the snapshot's own
directory for a commit that ALREADY made the decision** — the reds may be its
REMAINDER, and then the question is completeness. `[M]` `39b46a31`
re-baselined 2 cylinder artifacts with a sha256 sweep + a per-artefact
`τ:=0.7` screen, and its universal *"all 23 snapshots … the only two that
changed"* was scoped to ONE directory while 7 more references had moved.
⭐ And check its per-artefact NULL reasons against EVERY mechanism the commit
bundled: one conjunction in the subject = two mechanisms = two checks owed.
`[M]` *"at M=2 the partition is BIT-IDENTICAL, so this case's τ did not
change at all"* — partition true (`5e-17`), τ moved **2.071e-01**, because the
same commit also retired the `[½,1]` absorber. Right conclusion, void
argument, durable certificate of blindness. [skill: #25] → L-069

**D1. Grep the WHOLE tree for the OLD literal, not the diff's touched files.**
A cross-check against a derived value WILL break (genuine miss); a
self-consistency round-trip survives while feeding wrong physics (latent stale). → L-023, L-025

**D2. Run the MASKING-CHECK on any loosened gate or regenerated baseline.**
Loosened → re-run the untouched arms, confirm they STILL hard-fail ≫ the bound.
Regen → OLD-snapshot-vs-NEW-code must hard-fail (load-bearing) AND
NEW-snapshot-vs-OLD-code must hard-fail (live gate). → L-022, L-028

**D3. Characterize drift from the BINARY:** `git show <c>~1:x.npy` vs
`git show <c>:x.npy`, then ULP-diff. Live-code vs regenerated-snapshot is
necessarily 0 ULP and characterizes nothing. → L-022

**D4. Large ULP at small magnitude is a metric artifact** — inspect the worst
element's magnitude before calling 256 ULP a violation. Conversely ~1e15 ULP on
ONE geometry with siblings green is a stale snapshot. → L-022, L-024, L-028 [Sig-10]

**D5. A HARD nULP floor and a STRICT bit-identity floor are different gates —
verify WHICH invocation ran.** Strict = `-W error::DriftWarning` layered on top;
`tests/sn/regression/conftest.py` downgrades it for its own dir but does NOT leak
to siblings (measured — assume neither way). Prove a strict floor live by
perturbing the baseline 1 ULP (`np.nextafter`). → L-014, L-015

**D6. Settle a byte-identity dispute with the IEEE micro-fact + `git status
--short '**/*.npy'` — NEVER a docstring.** `0.5*(a+b)==0.5*a+0.5*b` bit-for-bit
for all doubles (a `w=½` affine closure IS byte-identical to DD, contra its own
docstring); `2*X/D ≠ 2*X*(1/D)` at 1 ULP (the real re-baseline trigger); an
einsum spectator lift `fc->gc` ⇒ `fc...->gc...` is `array_equal` at rank-2. → L-020, L-028, L-032

**D7. When a carve preserves the COMPOSITE and not the leaf, prove byte-identity
on the composite DIRECTLY** (both emitted against a read-only baseline worktree).
A brief's "≤16 ULP" can understate leaf drift ~7× — say which object is pinned. → L-049

**D8. Prove a "verbatim relocation" by NORMALIZED AST-diff, not by re-running
gates** — substitute into the old body, strip docstrings/imports/blanks,
`difflib`. A true move reduces to the signature line plus the declared fork. → L-013

**D9. For an ADDITIVE-only change, grep the tree for ANY importer of the new
module (excluding its own tests). Empty ⟹ it cannot perturb a pre-existing
outcome** — stronger than re-running the baseline reds. → L-042

**D10. Prove a `singledispatch` alias rename via
`Cls.__dict__['apply'] is Cls.__dict__['_apply_impl']`** — `Cls.apply is
Cls._apply_impl` is False (fresh descriptor per access), a red herring. → L-050

**D11. Do NOT trust "byte-identical EXCEPT one LATENT collision" — PROBE it.**
Compute BOTH branches every call and `array_equal` them across the FULL gate
suite (plugin reassigns the symbol in EVERY importing module; attribute by
`item.nodeid`; read under `-s`). Measured 48 divergences at 70 % ⟹ REACHED. The
two-paths gate that found them is Mode-11 blind (shared callee). "Latent via the
public entry" can be TRUE while "latent everywhere" is FALSE. → L-035

---

## E. Markers, levels, and the ORPHEUS audit surface

**E7. NEVER read a Nexus V&V number as a coverage claim — the surface is a SEARCH
relation wearing a PROOF relation's name.** `[M]` all 2748 `tests` edges are
`test→equation` (there is **no** test→code edge); all 16624 `implements` edges are
`source="inferred"`, **81 % on ONE shared token** (`operator`/`method`/`case`/`cell`);
`verified` is set iff `len(tests)>0` with **no** confidence floor, so **351 of 692**
"verified" equations have no declared test and a **CP** test "verifies" an SN
cell-flatten identity via the token `"cell"`. `nexus provenance` compounds it —
`implemented_by` is really *"documented on the same page"* (10 printed / 1 real).
Ask of any status: **what predicate sets it, and what is its weakest admissible
evidence?** → L-070

**E8. The graph's marker surface is PARTIAL — census it before designing a
marker-driven query.** `[M]` `foundation` (1515 usages / 308 files) and `regression`
(10 files) have **no node attribute at all**; only `verifies`/`vv_level`/`catches`/`slow`
are lifted. `catches` is an ATTRIBUTE, not an edge, and no `ERR-NNN` node exists, so
its claims cannot be joined to the catalog by traversal. No `.npy`/`.npz` snapshot is a
node either — a frozen reference cannot even be NAMED. → L-070

**E1. `@foundation` stacked with `@verifies("<physics-eq>")` is silent level
conflation** — the harness records both, so Nexus credits a physics equation with
a foundation test's parametrizations. Tell: a `documented` equation whose ONLY
coverage is a foundation test. → L-007

**E2. A `catches("ERR-NNN")` marker DECAYS** [skill: Mode-8 class 7] — re-verify
on every review of the file, not once at authoring. Same-area misattribution is
constant (an `A==A` matrix pin credited with an INFLOW-factor bug). → L-031, L-054, L-061

**E3. An audit-MISSING `catches` has FOUR outcomes — grep the production RAISE
SITE first.** (1) genuine catcher → tag, mutation-verified; (2) the catalog's L0
test was RETIRED and the marker did not migrate → re-tag the successor; (3) the
typed error is exported but NEVER raised → dead scaffolding, NO CATCHER, do not
invent a marker; (4) `assert_X` delegates to a WEAKER sibling → NO CATCHER. → L-054

**E4. `xfail(strict=True)` is satisfied by ANY failure — verify the REASON**
(`--runxfail`, match the documented `reason=`, then re-run without it). A stale
array index made a gate a FALSE xfail. → L-008

**E5. Two more level-marker tells, distinct from E1.** `foundation` under a
module `pytestmark=l1` emits `conflicting V&V level markers` and the intended
level is SILENTLY DROPPED. A self-generated regression baseline wearing `l1` is
conflation — fix the marker, not the file; its `_load_or_skip` should HARD-FAIL. → L-058, L-061

**E6. Audit mechanics.** Orphan triage order D→B→A→C (class D = existing test
needs only the label; ~25 % of orphans). `matrix.rst` LAGS a label rename — re-run
`python -O -m tests._harness.audit --gaps` for live spelling. `vv-status`
rationale comments use (parens), never [brackets] (docutils reads citations). → L-002, L-003, L-004

**E7. A `catches("ERR-NNN")` marker on a `@pytest.mark.slow` test is a coverage
claim the CANONICAL gate never adjudicates.** [NOT in the skill — Mode 8's nine
classes are all about a gate that cannot FAIL; this is a gate that cannot RUN]
The catalogue counts the ERR as caught, the test genuinely reds when the defect
is re-introduced, and `-m "not slow"` — the project's canonical invocation —
deselects it, so a regression lands green in every gate that decides a merge.
`[M]` #428: ERR-023's ONLY catcher is `tests/mc/test_gaps.py:718`
(`slow` + `catches("ERR-023")`); under ν₂ₙ: 2→1 the MC tree reads **39 passed /
0 red** at `-m "not slow"` and the same test **FAILS in 84 s** run alone.
⟹ when auditing an ERR's coverage, read the catcher's MARKERS, not just its
existence; a `slow`-only catcher is an ERR whose real not-slow coverage is
**zero** and should be said so in the audit. → L-079

---

## F. Claim-scope — the claim is broader than the evidence

**F1. A "behavior-neutral" claim holds only for the ONE contract it was proven
against.** [skill: #12 / ERR-063] Residue: the proxies that fooled the closeout
were "no guard errors" and "DD snapshots didn't move" — DD is the SN-only
consumer where the field IS inert. **Run the slow accuracy-band suites.** → L-045

**F2. Exercised ≠ constrained.** [skill: Mode 10] Three states — nulled /
exercised-but-unconstrained / verified; never collapse the middle. With NO
isolating regime the STRUCTURAL pair (machine-precision threading + sign-flip ≫
tol + a no-op control) is COMPLETE; never manufacture a value-improvement leg.
Calibrate the tol live — a deterministic SI re-solve floors at 0.0. → L-026, L-037, L-038

**F3. NEVER accept a "designed-green / blind to this mutation class" narrative by
inspection — RUN the mutation.** [skill: Mode 12] Both directions are real:
over-claiming a catcher, and UNDER-claiming blindness (a leaf-transpose DROP is a
NON-transpose operator; its k SHIFTS). When the pushback lands on the skill's own
example, flag the skill edit as a finding. → L-058

**F4. A cited mutation MAGNITUDE for a metric-adjoint SOLVE must be the
full-solve value — RUN it, never the angular-collapsed 0-D proxy.** Metric
conjugation of a MUTATED operator is not spectrum-preserving. A never-asserted
cited number is still a plausible-substitution error. → L-058

**F5. For a "no missed site" dedup claim, the PLAN is the scope authority, not
the closeout.** A residual hit is a defect only if (a) a direct reconstruction,
(b) not transitively routed one level deeper, AND (c) in declared scope. → L-025

**F6. A stress-ansatz mandated by the test-architect memo is a binding
contract** — shipping the canonical `sin(πx/L)` 1G homogeneous case instead is a
gate DOWNGRADE. Flag it even when all tests pass. → L-019

**F7. "BC X is load-bearing because k = k_∞" is TRUE only for HOMOGENEOUS.** On a
heterogeneous reflective sphere the flux is non-flat, so the term DOES move k
(measured larger than vacuum). Check the config. → L-012

**F8. Check what the test HELPER tolerates before crediting an enforcement
claim** — a `squeeze_density` helper made the suite agnostic to `keepdims`, so
the bit-identity claim held only up to a squeeze. → L-042

**F9. "Matvec twin verified" is KERNEL-level; end-to-end Krylov≡SI is a separate
claim.** A loud `NotImplementedError` on the deferred half is the CORRECT interim
— but say so, and don't let a spec's wording credit the un-shipped half. → L-031, L-033

**F10. Every brief-named symbol/file is a CLAIM — confirm with `find`/grep before
editing** (two phantoms in one brief). Byte-compile no-test generator SCRIPTS
after a rewire: a broken import there is a breakage no test run surfaces. → L-051

**F12. A retired/tombstoned claim is a CONJUNCTION — enumerate its legs and check
each; one leg routinely survives its siblings' death.** Legs come two ways: per
SUCCESSOR (a tombstone naming N gates may name one PER LEG — a dropped `codomain`
binding reddened only the periodic gate, the split gate's `a.codomain is
b.codomain` being `None is None`-satisfiable beside an `x is <concrete>` domain
leg) and per PARTITION CLASS (SN faces are THREE-way, so "residual zero at
non-outflow" = inflow [INVERTED by #208] ⊔ tangential [still exactly true]).
Mutation-check per leg, and check the fixtures can EXPRESS the survivor at all
(all 3 there carry 0 tangential ordinates). → L-063, L-064

**F14. FILL a plan's ⏳PENDING decisive row yourself — and use the plan's own
anchors as the probe's positive control.** A "decide nothing until X is
measured" row is the highest-value thing you can produce; the plan usually
states what X must reproduce, and that IS the control (`[M]` Q5.6.4: my live
probe hit both anchors `6.5934e-02` / `1.2676e-01` to the printed digits, which
is what licensed reading its NEW row `τ≡½ = 1.0181e-01`). Cache the expensive
reference to disk — the arm sweep is then cheap. Grep `Solution.__dataclass_fields__`
before assuming an attribute name (`keff`, not `k_eff`). → L-068

**F15. An "honest cost" is a COMPARISON — measure it against the candidates on
the table, never in the abstract.** A caveat "X has the usual exposure to Y" is
inverted if every candidate has Y and X has the LEAST. [skill: #24(c)] → L-068

**F11. ACCEPT a floor-CHARACTER gate; do not demand a floor-REMOVAL gate.** When
a fix cleans a RATE but leaves a floor, the honest claim is a falsifiable scaling
pin (`err(S32) < err(S16)/2`): a closure-BUG floor is quadrature-independent
(ratio ≈ 1 → fails). Here the pushback is against over-demanding. → L-009

**F16. An inherited blast-radius number counts a NAME, not a TYPE — re-measure it
with an in-process wrap before it sizes (or DEFERS) the work.** `[M]` "~87 reads"
was `grep '\.converged' tests/derivations/`, of which **72** belonged to an
unrelated result family sharing the attribute name; wrapping the actual class's
`__init__` + `__getattribute__` gave **33 constructions / 0 without the field / 2
reads** — 43× over, in the direction that defers a zero-churn fix. Route:
**Nexus for producers** (an attribute read is not an edge — `degree: 1` is the
graph being right), **dynamic wrap for readers**, grep only to enumerate
candidates. Pair the dynamic `0` with a static no-other-path proof (no `**`
splat / `asdict` / `replace` on that class) or it is "not observed", not "none".
→ L-066

**F18. A design's TELL / done-when is a GATE — run its own predicate before
crediting it, and intersect the hits with the design's declared UNTOUCHED set.** A
tell scoped tree-wide inside a design that declares part of the tree out of scope
is **DESIGNED-RED** (the mirror of #17's designed-green harness): pinned at
failure however well the work lands, and it reads as *work remaining*, so a later
session chases an unreachable target. `[M]` "grep `SigS` finds the datum owned
once and viewed once" → **70 hits**, direct `Mixture.SigS` readers in
cp/moc/mc/sn + derivations, ALL inside the assembly's own "Untouched" list.
⚠ Naming an instrument makes a done-when read as MORE rigorous — which is exactly
why nobody re-runs it. Third shape for `plan-authoring` §10. → L-072

**F19. "X is not data of this operation" is decided at the CODOMAIN constructor,
not in the arithmetic.** A pure / local / diagonal kernel can still need X to
BUILD its result. `[M]` all four SN energy operators ARE spatially diagonal and
all still read `mesh` off the OPERAND to stamp `…SourceSink.from_mesh(v, mesh)` /
`zeros_on(mesh)` (≥11 sites; the bound space carries no mesh on any block), and a
production guard asserts the thesis's opposite (`streaming.py:589`: "its mesh is
carried by its CrossSectionField coefficient"). ⚠ Beware one word, two referents —
"carrier" = the FLUX operand in docstrings, `MaterialMesh` in the plans; a
prose-sourced `[M]` inherits the wrong one silently. → L-072

**F20. In a multi-assembly review, read every RIVAL's self-attacks as a checklist
against your target, then push one level past the argument each answers.** A
self-attack marks the SEAM, not the depth — the prepared defence is the tell the
author stopped there. `[M]` the parsimony rival published as its OWN weakness the
conformity-guard denominator the target asserts away (axis-structural on **4 of
13** bindings), and the target's Attack 2 argued instance-vs-class monomorphism
while the factual half of that same seam went unstated. → L-072

**F17. A hardcoded status constant is a defect only if the producer ITERATES —
triage one hop UP before a grep-driven sweep.** `[M]` 7 hardcoded
`converged=True`; 3 sat on direct `scipy.linalg.eig` / `np.linalg.solve`
producers where `True` is honest. A "fix every hardcode" pass mints **false
honesty** at those. The lies and the facts are grep-identical — which is how the
lies hid. (Same shape for any `success`/`valid`/`exact` flag.) → L-066

**F21. A reproduction can agree to EVERY PUBLISHED DIGIT and still "disagree" —
check the UNIT before the number, because an overloaded unit name can invert the
study's own conclusion.** `pcm` = 10⁻⁵ and says NOTHING about what was divided by
what: `Δk·1e5` / `Δk/k₀·1e5` / `Δρ·1e5` are three different numbers, differing by
`k₀`. `[M]` #426: my −377.56 vs the claim's −413.55 was purely this, with all
three k's bit-equal to 9 dp. The bite is the fixture SET — at k₀ = 1.0953 vs
1.5262 the effect ranks 414 < 529 in absolute and 346 > 228 in reactivity, so
*"a thicker reflector makes the truncation worse"* is TRUE in one convention and
FALSE in the other (2.3× spread). ⟹ a derived comparison quantity carries its
DEFINITION, not its unit name; the tell that it matters is a fixture set
spanning a range of the normalising quantity. → L-078

**F22. Two probes over the same production code cannot see a SHARED convention —
test the shared premises against PHYSICS, not against each other.** [skill: #7,
one layer down: shared *code* rather than shared *identity*] The two cheap ones
that closed #426: an energy-losing channel's transfer matrix must be strictly
upper-triangular in canonical order (`[M]` 8195/8195 and 6067/6067 nonzeros,
lower-triangle mass 0.0 — closes the (from,to) convention); and a single
transfer's `Σ_ℓ/Σ_0 = ⟨P_ℓ(μ)⟩ ∈ [−1,1]` is a HARD entrywise bound (`[M]` max
0.9603, **0** entries > 1 — a stray `(2ℓ+1)=3` would have read ≈2.9). Both are one
`.todense()` and a comparison. Then STATE the premises you did not close. → L-078

**F23. A CALIBRATION of your cross-check is itself a ratio and needs the
share-a-population test — an uncalibrated corroboration beats a mis-calibrated
refutation.** `[M]` #426: a transport-correction second route read 1.5× the direct
answer; calibrating it on the elastic channel gave `ΔTR/ΔP1 = 0.60`, which would
have "shown" the direct route 2.6× too small — but `[M]` **327 of 421** groups run
a negative corrected diagonal in the elastic leg against **6** in the (n,2n) leg,
so the factor is not transferable. ⟹ report the corroboration at its measured
accuracy class ("sign decisive, magnitude within a factor the approximation is
itself measured to span; cannot adjudicate a factor of 2") and say so. And when a
second route's convention risk exceeds the claim's, DON'T run it — a wrong
reproduction of yours impeaches a correct result. → L-078

---

## G. Doc / prose correctness (Cardinal Rule 3 findings, not V&V)

**G1. Campaign-narration staleness: the FIX bar is "provably lies about CURRENT
code", VERIFIED before ruling** — grep the named symbol/wiring tree-wide,
`gh issue view N`. Default = KEEP. Guards: a stale line inside a RUNTIME STRING
is behavioral → KEEP; a "failure here HALTs Phase X" banner is a record → KEEP. → L-055

**G2. Reviewing a skill→Sphinx distillation: verify code-anchored specifics
against CODE, never the skill twin** (the source's stale specifics propagate
verbatim). Python-domain roles are NOT `-W`-gated — a dead `:mod:` renders as
plain text. Grep the corpus: the OUTLIER spelling count is the bug. → L-056

**G3. Reviewing a results-compilation page:** a count DE-FREEZE is CERTIFIABLE
(live `--collect-only` proves the old literal lied); a doc RETITLE can beat the
test's own stale name/docstring — verify against the live `assert` body; a
run-book delegating detail to a config file may point at a contradicting note. → L-057

**G5. A CAMPAIGN-STEP NAME in a forward-looking claim is a self-expiring token —
when the step lands, grep the step's own name.** `grep -rn 'G6\.5'` minus the
retrospective forms (`since|at|\(|—`) found the 2 survivors out of 33 hits in one
command; both said "G6.5 retires the lengths" and G6.5 deliberately did not — one
of them 146 lines from the SAME FILE's corrected twin (vv #21 aggravator: the
file can now be cited for either). Also: a brief declaring "the known baseline
reds" declares the reds of the batteries IT ran — widen scope, reconcile the new
ones against the PARENT commit in a worktree before attributing. → L-065

**G6. DERIVATIVE staleness — a correction's own TODO note outlives the
correction.** A sentence of the form *"file X is stale and owes a dated fix"* is
a claim ABOUT ANOTHER FILE, and nothing in X's repair prompts anyone to retire
it; it then instructs readers to distrust a file that is now correct. ⟹ after
fixing X, **grep for pointers AT X**, not only inside it. `[M]`
`coding-elegance/SKILL.md:390` still says two named files "still cite" the
retired type — both were corrected since. Same shape for a retirement TOMBSTONE
naming its own successor test: `[M]` a "successors carrying the surviving
claims" comment named `test_subtraction_mints_a_displacement_composite_per_block`,
which exists NOWHERE in the tree (renamed by the same carve) — a dead pointer
inside the artefact whose job is keeping coverage traceable. → L-071

**G4. A test's own prose is the least reliable thing in the file.** A "frozen /
bit-identical to the pre-carve path" docstring stales SILENTLY when its `.npy` is
regenerated and the test file is untouched — on any regen, grep consumers for
"frozen". A cited issue number can be wrong (trust git-archaeology). A prose "the
ERR-NNN class" citation is a nit; the same string in `catches()` is a defect. → L-020, L-028, L-030, L-034

---

## H. Mechanics, environment, probe hygiene

**H1. NEVER use a bare `assert` in your OWN `python -O` probe script** — it is
stripped; a throwaway printed "PASSED" while the values were visibly unequal.
Run teeth-checks through pytest, `np.testing.*`, or an explicit `raise`. → L-052

**H2. Settle a brief's Mode-8 hypothesis about a `tests/` subtree in 2 min, then
PIVOT.** Synthetic control + a falsified COPY of a real file, both modes — the
premise is usually REFUTED (0/676 inert); the real surface is `orpheus/` plus
NON-COLLECTED helpers. Pivot to an AST census of *what the asserts assert*: only
~29 % of bare asserts pinned a VALUE. → L-006, L-010

**H3. Replicate the test's OWN solve helper before judging a value claim** — a
naive `solve_sn_fixed_source(...)` defaults to vacuum; a divergent
hand-replication usually means YOU dropped the BC. → L-011

**H4. Baseline via a READ-ONLY worktree + `PYTHONPATH`, and verify it took**
(`git worktree add -d <ref> /tmp/x`; the editable `.venv` otherwise resolves to
the MAIN tree — confirm via `orpheus.__file__` / `inspect.getsource`). A worktree
pyright count needs the main `.venv` symlinked into its root. → L-041, L-045, L-049

**H5. pyright deltas are apples-to-apples only after line-stripping and per-file
reconciliation** — a `(file, rule, msg)` diff gives FALSE net-new when a
type-RENDERING string shifts. Rule out a masked offset with: full-tree total
exactly the baseline, SUT isolation 0/0, EMPTY diff on any reverted seam file. → L-027, L-039, L-050

**H6. Removing a `NoReturn`-poisoned return UNMASKS every latent error downstream
of the first poisoned call** (pyright suppresses after a `Never`-returning call).
Expect net-new ≠ per-file delta; classify each as latent vs regression. → L-050

**H7. In a SHARED working tree, diff ONLY your own touched files** — other
agents' edits make every dirty file look like yours. → L-054, L-055

**H8. Locating slow/timeout tests: batch into runs that COMPLETE** (a SIGTERM'd
run writes no junit-xml and loses the `-rfE` reasons). Mark slow PARAMS with
`pytest.param(..., marks=...)`, not the function; verify with `--collect-only`. → L-005

**H9. zsh does NOT word-split an unquoted `$VAR` — it has now bitten TWICE, in
two different tools, and the second time it manufactured a clean bill.** Use an
ARRAY + `"${VAR[@]}"`, always. (a) `pytest $SUITE -p $M` passed the whole list as
ONE argument → 0 collected. (b) `grep -rn "$pat" $TREES 2>/dev/null || echo "(0
hits)"` searched ONE nonexistent path across six trees and reported **all-clean**
— `2>/dev/null` ate the error and the `|| echo` laundered rc≠0 into a *finding*.
⟹ on any census: never `2>/dev/null`, never `|| echo "(0 hits)"`, and **run a
positive control per tree BEFORE every sweep** (`grep -rl <ubiquitous-token>
<tree>` → a file count). A dropped tree is otherwise indistinguishable from a
clean one, in the flattering direction. → L-062, L-071

**H14. THREE greps, three answers — always run ≥2 and reconcile numerically.**
`grep` here is a shell FUNCTION wrapping `ugrep`; **`command grep`** is real BSD
grep; **`git grep`** is tracked-only. ⛔ **CORRECTED 2026-08-20 — the wrapper does
NOT honour `.gitignore`** (this file claimed it did, which would have certified an
inflated count as clean). `[M]` same query, `docs/`: wrapper **514** · `command
grep` **793** · `git grep` **11**. Use `git grep` as the SOURCE-truth filter and
`command grep` as the ignore-blind upper bound. ⚠ `git grep -- .claude` sweeps
`plans/` + `agent-memory/` too — restrict the path list to the SAME trees or the
mismatch reads as a discrepancy when it is a denominator (`[M]` `75 = 75` once
scoped). → L-071

**H15. A `grep -rl` FILE COUNT over a repo with build trees is an artifact count
until you exclude them — and the inflation always argues for the conclusion its
author already reached.** `--include=*.py` does NOT save you; `_build/` holds
`.rst`/`.html` sources. ⟹ on ANY file-count claim: `--exclude-dir=_build
--exclude-dir=__pycache__ --exclude-dir=.nexus`, then confirm with `git grep -l`
(a second, independently-chosen filter), then `git check-ignore` + `git ls-files
<tree> | wc -l` to prove the excluded tree is untracked. `[M]` 2026-08-20, a
design assembly's sole quantitative blocker for a structural non-mint read
*"529 files reference the name"*; **503 of the 529 sit in eleven stale
`docs/_build/html_*` trees — gitignored, 0 tracked files.** True radius **26**
(15 `.py` + 11 `.rst`), independently confirmed `11 + 15` by `git grep`. The
number was 20× high in the direction that made "don't rename" look forced.
Read with F-section: a count with no stated exclusion is an unmarked claim.

**H12. The SUBJECT of your review can move while you review it — re-`wc -l` and
`git log -1` the reviewed document before writing the verdict.** `[M]` the Q5.6.4
memo grew 721→879 lines mid-dispatch (`8db88596` added a §9bis.9 whose literature
argument was the strongest defence of the link I was refuting). Also re-check the
BRANCH: the harness's session-start git snapshot said `main`; git said
`refactor/operator-strategy-layers`. ⭐ And for a CENSUS in a shared tree the
document is the TREE: stamp `git rev-parse --short HEAD` + `date` at the start
AND the end, and re-run every finding as a PREDICATE at the end. `[M]` CS3-R:
HEAD moved `f43758d8 → 755f99b5 → a740d7ba` in 8 min while a parallel agent
fixed 5 of my flagged files — caught only by a `sed` reading "increments" where
my own grep 4 min earlier read "displacements" at the same `file:line`. Report
the REMEDIATED set separately; a finding someone else silently fixed, left in
the list, reads as a false positive and discredits the rest. → L-068, L-071

**H13. `grep "^FAILED"` on COLOURED pytest output matches NOTHING — a false
all-green, in the flattering direction.** ANSI escapes precede the `F`. `[M]`
my extraction reported zero failures beside a `41 failed` summary; only the
warnings lines leaked, because `-W error::…` contains `error`. And **never
pipe a BACKGROUND command through `grep`** — the task file then holds only the
filtered output and the evidence cannot be re-extracted (17 min lost). Always
`--color=no`, redirect FULL output to a file, filter afterwards. → L-069

**H10. Run `-rs` and READ skip reasons on any suite with a non-zero skip count**
— a skip reason containing an exception message is a permanently-dead gate. → L-061

**H11. `full_output=True` does NOT make a scipy status readable — `disp=False` is
the load-bearing half.** `[M]` with `disp` defaulted `True`, a non-converged
`brentq`/`root_scalar` **raises** instead of returning `converged=False`, so the
`False` leg is an unreachable branch wearing an honest name. On any "we read
scipy's flag" claim, check `disp` first. (Omitting `full_output` entirely is
*honest-by-raising* — fine, but not *readable*, so it cannot serve a
warn-don't-raise contract.) → L-066

---

## I. Already in the skills — point, don't restate

Homes are `vv-principles` SKILL.md unless named otherwise.

| Doctrine | Home | Archive |
|---|---|---|
| Test count ≠ coverage; het + multi-group + refinement | #3/#4; `bug-signatures` H1–H5; AGENT.md #5 | L-001 |
| Mode-8 `-O` strip: rewriter boundary, `testpaths`, the 6 fires-but-cannot-fail classes, the bite-check METHOD WARNING | Mode 8 | L-006, L-010, L-061 |
| `catches` = coverage CLAIM; mutation-verify the EXACT bug; markers decay | §Log every caught bug | L-007, L-031, L-054 |
| Mode-10 activated-but-unconstrained, incl. no-isolating-regime | Mode 10 | L-026, L-037, L-038 |
| Mode-11 gate-never-executes-the-rewired-path + plugin sentinel | Mode 11 | L-018, L-031, L-033, L-043 |
| Mode-12 invariant-functional; metric repair; commutator criterion; k-tooth 0.171 | Mode 12 | L-058, L-060 |
| Behavior-neutral zeroing/retype needs per-consumer VALUE proof (ERR-063) | #12 | L-045, L-048 |
| Sample-generates-the-group (ERR-072); partner ≠ bijection (ERR-073); monotonicity law; positive control | #13/#14/#15/#17 | L-061, L-062 |
| ⭐ A **refinement ladder** `8/16/32/64` is ONE congruence class, not "every order" | #13 (the ladder sharpening) | L-068 |
| ⭐⭐ Validating an **ADJUDICATING instrument** (ranks designs, nothing to mutate): the BASIS check (probe modes vs the problem's symmetry + what the rule can represent — the WEIGHTING can be provably robust while the basis is wrong), the RANK-CORRELATION check (≥3 candidates: which mechanism is the metric ordered by?), the cost-against-alternatives check | #24 | L-068 |
| Bit-identity vs principled-equivalence; the 3 criteria; AMPLIFY | §Bit-identity | L-022, L-049 |
| Stale-snapshot huge-ULP triage; splitting verified in a degenerate regime | `bug-signatures` Sig-10; Mode 9 | L-034, L-036, L-041, L-053 |
| ⭐ A BUNDLED change's per-artefact NULL reason must be checked against EVERY mechanism it retired (else a durable false blindness certificate); a re-baseline's radius is the frozen REFERENCES, not one directory's `.npz` | #25 (added by L-069) | L-069 |
| ⚠ Sig-10's sibling-pass discriminator is VOID when the changed code is single-geometry — SLB/SPH green carries NO information about a cylinder-only carve; bisect instead | `bug-signatures` Sig-10 | L-069 |
| ⭐⭐ A guard keyed on an operand's **OPTIONAL METADATA** (`space.axes`, a `record`, any `X \| None` slot) is INERT wherever the field is `None` — and because a *convenience* factory populates it while a *composite* factory forgets, the inert region is systematically PRODUCTION while every hand-built fixture reddens on demand. Signature-checking passes; only a runtime read of a **production-built instance** answers it. Write the live/inert FRACTION into the guard's docstring. `[M]` `FullFieldSpace.from_blocks` passes no `axes` ⟹ an `EnergyAxis`-based conformity refusal is inert on **7 of 13** SN/diffusion bindings (⛔ the item's own "8 of 13" is off by one — `homogeneous/solver.py:152` falls through `from_mesh`'s chain to `bulk_space`; `MaterialMesh` has no `full_field_space`) | **#28** | L-073 §2 |
| ⭐⭐ Collapsing runtime dispatch onto a construction-time **KEY** is a claim about **TRAFFIC**; an inventory of **ARMS** is not that claim. Run the per-INSTANCE census before believing it — wrap `cls.__dict__["apply"]` *through the descriptor protocol* (a naive `cls.apply` re-bind breaks `singledispatchmethod`), log `(bound key, type(operand))` per `id(instance)`, positive-control on a bit-identical headline number. Three silent failures: **wrong arm** (bound composite, fed only arrays), **non-determination** (one key, two families, one solve), **asymmetric arrow** (typed in, bare out). `[M]` **6 of 12** ORPHEUS production instances refute it. ⟹ And the meta-rule: **when a document flags a missing measurement as its OWN weakest point, run that one first** — the author has localised the defect for you | **OWED as #29** (drop-in text: `scratch/cs4a_attack_algebra.md` §5.2 — not landed, charter was change-nothing) | L-073 |
| ⭐ The metric is **INERT** on a spatially-diagonal operator over any bulk space (`[G,Aᵀ]=0`), and multigroup energy is counting **by theorem** (`axis.py:226-239` refuses a weighted `EnergyAxis`) ⟹ **no `.H`-vs-`apply_transpose` gate can witness** an Optional→mandatory space flip on C/IsoS/IsoN2N/F. `[M]` `0.0 / 0.0 / 4.4e-16 / 2.2e-16` at `V_cell` spread 3358×. The honest witness is a construction refusal | #19 + Mode 12 (commutator) | L-073 §2 |
| ⭐ A **mandatory-parameter flip**'s §6b unit is production ∪ **TEST** constructions, and the test half runs 10–20× larger. `[M]` 10 production vs **165** space-less test constructions in ~50 files — all three assemblies counted only the 10 | `plan-authoring` §6b + `coding-standards` (retirement = test migration) | L-073 |
| Green gate = nothing until RED; SN `.apply`/`.solve` role contract | `qa/AGENT.md` #11/#10 + the role memo | §A |
