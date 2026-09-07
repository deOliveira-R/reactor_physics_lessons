# Explorer — Lessons

Behavioral corrections only: "what mistake did I make exploring, and what
did I learn that improved my behaviour?" The HOW of each Nexus tool lives
in the preloaded skills (`nexus-exploring`, `nexus-guide`, and the wider
`nexus-impact` / `nexus-debugging` / `nexus-verification` / `nexus-refactoring`
family) — point there, never duplicate the workflow here. Per-campaign
`file:line` maps are archaeology; they go stale in days. A lesson earns its
place only if it changes how the NEXT exploration is run.

The cross-cutting spine: **an exploration answer is not "I found the symbol"
— it is "I found EVERY consumer the next action will touch, I verified the
premise against the current tree (not the issue text, not a frozen memory),
and I separated the durable subsystem-shape from the line numbers that will
drift."** This spine is now codified as standing directives in `AGENT.md`
Operating Principles 4–7 (blast-radius, premise-verification, git-merge-status,
durable-vs-line). L-001/L-002/L-003/L-005 below are RETAINED for forensic
value (the war-story behind each directive); the directive itself, not the
incident, governs behaviour. L-004 and L-006 stay lesson-only — they fire on
narrower question shapes (carve-verdict / probe-collapse), not every task.

---

## L-001 -- A retirement/rename blast radius = graph callers AND text grep AND direct constructors AND doc nodes

→ **Now AGENT.md Operating Principle 4** (the four-search discipline). War-story kept below for the specific misses each search catches.

`mcp__nexus__callers` / `impact` find the *graph* consumers, but a retirement
audit that stops there under-scopes — and under-scoping a retirement forces a
mid-session re-plan (the documented ~2× cost behind the proactive-explorer
trigger). The consumers a single `callers` query misses:

- **Property-reached leaves.** A method only reachable by reading a
  `cached_property` and calling `.apply` shows `callers() == 0` while still
  being live through the property. Audit the property's readers, not just the
  method's callers.
- **Bypass-trick / class-name consumers.** A test that uses an orphan via its
  CLASS NAME for a side purpose (a validation-bypass) is invisible to a
  method-level `callers`. A repo-wide grep for the `_ClassName` surfaces it.
- **Direct constructors of a guarded type.** A guard-at-the-data-source has
  blast radius = EVERY direct `Foo(...)` call, not just the factory path.
- **Doc nodes that will dangle.** A retired symbol referenced from a theory
  page leaves a broken `:ref:`. `graph_query` for the doc→symbol edge (or the
  symbol name in `docs/`) catches it so the archivist hand-off is complete.

How to apply: for ANY retirement/rename audit, run BOTH graph (`callers`/
`impact`) AND a text grep of the symbol/class name AND a constructor audit (if
a guarded type) AND a doc-node scan. Four searches, not one. (Reinforces the
proactive-explorer-before-retirement trigger; sibling to method-implementer
L-004.)

**Sharpening (two graph-blinding patterns endemic to the operator algebra).**
Two constructs make `callers()` systematically lie in this codebase, and BOTH
appeared in the W-F scope audit — when an `apply`-dispatch arm's liveness is the
question, grep is not a cross-check, it is the *primary* evidence:
(a) **runtime-aliased dispatch** — `apply = _apply_impl` with `@singledispatchmethod`
arms means the graph attributes calls to the alias, not to the per-type arm; a
`@_apply_impl.register` leaf shows near-zero callers while a whole solver feeds it.
(b) **Protocol-typed receivers** — a call `solver.compute_fission_source(...)` where
`solver: EigenvalueSolver` (a Protocol) is unresolvable to the concrete `SNSolver`
method, so the concrete method reads `callers==0` even though `power_iteration`
drives it every outer step. The liveness of a dispatch arm is decided by the
ACTUAL input TYPE at the production call site — trace `power_iteration →
solver.compute_X → op.apply(<what type?>)` by READING the chain, and let grep, not
the graph, enumerate the `op.apply` sites.

---

## L-002 -- The issue text is a stale premise; verify it against the current tree FIRST

→ **Now AGENT.md Operating Principle 5.** War-story kept below for the worked examples (diamond-coefficients / 2-D-matvec premises already landed).

Repeatedly, an audit's first deliverable was "the premise the issue describes
is STALE — that work already landed." An issue body is written at one moment
and the natural trigger for its work (a related carve) often lands it early
under a *different* campaign. Examples of the same shape: a "lift the inline
diamond coefficients" issue was already folded onto the scheme; a "2-D matvec
recomputes inline" concern was resolved by an earlier phase.

How to apply: before mapping HOW to do an issue, spend one query confirming it
still NEEDS doing. Grep the named symbol / read the current body of the named
function. If the premise is stale, the deliverable flips from "implement" to
"CLOSE-VERIFY (regression-pin + issue hygiene)" — say so up front. This pairs
with the git-authoritative discipline (L-005): code state, not the issue's
prose, is ground truth.

---

## L-003 -- Separate the DURABLE subsystem-shape from the line numbers that will drift

→ **Now AGENT.md Operating Principle 7.** War-story kept below for the home-placement detail (durable → AGENT.md durable-shape section; transient → topic file flagged with the HEAD it was current at).

Every audit I wrote mixed two things with opposite shelf-lives: the durable
STRUCTURE (what couples to what, which seam is polymorphic, which path is
canonical) and the perishable `file:line` map. The structure survives years of
churn; the line map is wrong within a sprint. A memory that fronts the line map
reads as authoritative long after it has rotted, and a future session trusts a
dead address.

How to apply: lead every finding with the durable claim ("the within-group
operator is the variadic `(L+C, S, B)`; the sweep reads `ψ.boundary.inflow` and
does NOT re-apply `R·G` internally"), then mark line numbers as
re-derive-via-Nexus, never as the headline. The durable subsystem-shape belongs
in `AGENT.md` (its "SN operator-algebra subsystem — durable shape" section is
the canonical home); transient maps belong in a topic file flagged
"line numbers current at HEAD X, re-derive if drifted" — and are deletable once
the campaign merges.

---

## L-004 -- A clean carve verdict names BOTH the retire case and the keep-as-anchor case, with the discriminator

The strongest audit verdicts were not "retire it" or "keep it" but "RETIRE-eligible
BY <discriminator>, AND here is the defensible documented-KEEP, AND the call is
the user's because it turns on a judgment the explorer can surface but not make."
The discriminator that decides it is the `coding-standards` aggressive-retirement
rule's own test: same-math-available-via-the-surviving-helper ⟹ retire (genuine
redundancy); genuine-independent-consumer-need (even a future one a named typed
leaf would serve) ⟹ keep-as-anchor is defensible. The honest counter-weight is
Cardinal Rule 2 cutting both ways: a clean typed surface a future preconditioner/
DSA would consume is an architectural asset, not just dead weight.

How to apply: when asked "does X earn its keep?", deliver the dependency surface
+ the retire-with-rewire map + the keep-as-anchor counter-weight + the explicit
discriminator, and hand the value judgment to the user. Do not pre-decide a
retirement that turns on "will a future open issue consume this."

---

## L-005 -- Git is authoritative for merge-status; a memory's "in-flight / NOT pushed" freezes mid-flight

→ **Now AGENT.md Operating Principle 6** (and the always-on `process-discipline.md` rule). War-story kept below for the SN-campaign pattern that motivated it.

Memory notes captured a campaign as "uncommitted on branch X / NOT pushed," but
nearly every SN campaign merged in a later session — the note froze the moment it
was written. A future dispatch that trusts the frozen "in-flight" wastes effort
re-deriving landed work or, worse, treats merged code as still-pending.

How to apply: NEVER trust a memory's merge-status. Reconcile every "resume X /
in-flight X" against `git merge-base --is-ancestor <hash> HEAD` (or
`... <branch> main`) before acting. Active-state in MEMORY.md should say only
what git confirms; when in doubt, the answer is "check git." (Now an always-on
rule: `.claude/rules/process-discipline.md` §"Trust git for merge-status".)

---

## L-007 -- On a branch under ACTIVE edit, re-run the census immediately before reporting

During the F2 cast-family recon (2026-07-02, `refactor/pyright-burndown`), a cast
site moved 1532 → 1552 BETWEEN two of my greps: the main session was editing the
same files concurrently (uncommitted C3 carve in flight), and part of my brief
(the scattering `apply_transpose` item) was being FIXED while I explored. This is
intra-session drift — a different failure shape from L-002's stale-issue drift.

How to apply: when `git status` shows uncommitted edits in the subsystem being
audited, (1) re-run the position census as the LAST step before writing the
report, (2) diff the uncommitted hunks against the brief's items — an item may
already be mid-fix, flipping that deliverable to "confirm the in-flight fix +
report the alternative," and (3) timestamp reported line numbers as "at final
read; tree moving."

---

## L-008 -- zsh: an unquoted separator starting with `=` aborts the WHOLE compound command

`echo ===` (or any unquoted word starting with `=`) triggers zsh's `=cmd`
expansion; the lookup fails ("== not found") and the ENTIRE command line is
aborted — including greps sequenced after the echo, silently costing a
round-trip. Quote separators (`printf 'NAME\n'` or `echo "---"`), never bare
`===`, when batching multiple searches into one Bash call.

---

## L-009 -- A dataclass-FIELD rename audit is a grep problem, not a graph problem — and the field name may be a substring of an English word

Two independent findings from the `WithinGroupSystem.resolvent`/`.gains` rename
audit, both of which change how the NEXT field-rename audit is run:

**(a) Nexus does not model dataclass fields as nodes.** `context` on
`py:class:…WithinGroupSystem` returns class-level edges (doc pages, implemented
equations, referencing functions) but the only `py:attribute:` node was
`…WithinGroupSystem.loss` (degree 2) — `.resolvent` and `.gains` had NO nodes at
all, despite ~75 consumer lines. The graph surface contributed **0 of the 75**
sites. This is L-001's "graph alone under-scopes" at its extreme: for a FIELD
(as opposed to a function/class/method) rename, **text-grep is the primary
evidence and the graph is at best a way to find the owning class's doc pages.**
Don't spend a round-trip on `impact`/`callers` for a field; spend it on
`grep -rn "\.<field>\b|<field>=|<field>:"` plus a `replace(obj, <field>=` /
`getattr` / `asdict` sweep for dynamic access.

**(b) Check whether the OLD token is a substring of a common word before
proposing any replace strategy.** `gains` is a substring of **`against`**
(a-**gains**-t) — 679 occurrences in `orpheus/`+`tests/` `.py` alone. A bare
`sed s/gains/…/g` or an unanchored `replace_all` corrupts every one. The same
class of trap: `loss` ⊂ `lossless`, `space` ⊂ `namespace`, `role` ⊂ `payroll`,
`gain` ⊂ `bargain`. It also poisons the CENSUS, not just the edit — my first
grep of the owning file reported 21 `gains` hits where 2 were `against` in
prose.

How to apply: for any rename audit, (1) skip the graph for fields and go
straight to anchored greps; (2) run one `grep -c "<newword-containing-old>"`
sanity probe — or simply grep the old token with `-w`/`\b` anchors AND without,
and report the delta as a hazard line in the deliverable. Report the anchoring
requirement as an explicit instruction to the implementer, since a rename is
usually executed by a different agent than the one that audited it.

---

## L-010 -- "Complement" ≠ "the named sibling": a two-way selector split by a signed predicate has a THIRD bucket, and whether it is populated is DATA-dependent

Asked to verify a measured claim that projector `M` "is the projector onto the outflow
subspace" (i.e. `M == P_out`), the measurement reproduced on a slab and **failed on the
production cylinder**. `M` is `I − P_in` *by construction*; `I − P_in == P_out` only when
`inflow ∪ outflow` exhausts the index set. The trace's selectors are `< -eps` / `> +eps`
over a signed projection, so ordinates with `|·| <= eps` (tangential/grazing) fall in
NEITHER — and the CYL production quadrature has 4 of 8 there (Lebedev: always). Rank 18
vs rank 6 on the same face. The original claim was true only on the geometry it was
measured on.

How to apply: when an audit hands you "X is the complement of Y, hence X == Z", (1) find
the PREDICATE that defines the split and check whether it is exhaustive (a strict `<`/`>`
pair with an epsilon never is), and (2) re-run the measurement across the **production
data** that reaches it — enumerate the real quadratures / meshes / grids, not the one the
first fixture used. The slab is the degenerate case for nearly every SN index question;
CYL (`Quadrature.product`) and Lebedev are the discriminating ones. Same family as L-006
(split the probe KINDS before proposing the collapse), one level up: split the SET
partition before accepting an identity between two spellings.

---

## L-006 -- A "shape probe" is not always a missing predicate — split boolean-presence from integer-width before proposing a typed swap

Asked to collapse N value-based `arr.shape[-1] > 1`-style probes into one typed
predicate, the load-bearing finding was that the probes split into two KINDS with
opposite fates: Kind-A pure-presence ("does this axis exist?", boolean → swap to
the typed predicate) vs Kind-B width/count ("the actual `2^d`, needed for buffer
ALLOCATION → these are honest counts, KEEP them). Proposing to delete the width
derivations would have broken allocation. A second constraint that governs such
work: a typed factor may live on the FIELD, but the inner-walk sites that do the
probing often see only a bare ndarray + `mesh.scheme` — so the "clean predicate
swap" is really a small-plumbing change, not a one-line rename.

How to apply: before recommending "collapse these probes into one predicate,"
classify each probe boolean-presence vs integer-width, and check whether the
probe site even HAS the typed object in scope. Report the verdict as
"(B) small plumbing," not "(A) clean swap," when the factor isn't reachable at
the site.

---

## L-012 -- On a "blast radius ahead of a carve" brief, run `git diff --stat` as the FIRST tool call — the carve may already be underway, and that flips the deliverable's shape

During the B3.4c periodic/face-name audit (2026-08-01, `refactor/operator-strategy-layers`)
the main session executed the carve **while I was auditing it**: 8 production
files went clean → modified (+511/−76) across the dispatch, including the
"primitive to be minted" (already minted), all five named transcription sites
(already repointed), and the `apply_transpose` defect I had just written up as
the top risk (already fixed). I discovered this by accident — an import probe
caught `__all__` listing three names the module did not yet define, a transient
mid-write state.

This is L-007 (intra-session drift) one step earlier and with a bigger
consequence. L-007 says *re-run the census before reporting*; L-012 says **run
the diff BEFORE the census**, because on a pre-carve brief the premise "X is not
yet built" is the thing most likely to be hours stale, and when it is, the
deliverable is no longer a blast radius — it is a **done-vs-remaining
reconciliation**, which is a different document.

**Sharpening (2026-08-01, #326 half-range map).** `git status --short` at OPEN is
not enough to call a file "tracked at HEAD". Three test modules I cited as
"exists, tracked" were **untracked** — the main session created them mid-dispatch,
and a `?? ` line is easy to miss in a 38-line status. **Run
`git ls-files --error-unmatch <path>` on every file you are about to describe as
landed**, and re-run `git diff --name-only` at close (a helper went clean → `+118`
under me). Then tag each reference *(untracked, in-flight)* vs *(at HEAD)* in the
deliverable — the distinction changes what the reader may build on.

**Sharpening 2 (2026-08-03, non-SN geometry census).** The drift can invalidate a
**VERDICT**, not just a line number. I wrote the gap "`roots_of_unity` has zero
production consumers" — true at the opening HEAD, **false by the closing one**:
the main session landed `rules_circle.periodic_trapezoid` mid-dispatch, and that
new rule turned out to BE the thing MoC hand-rolls (its upper half, measured
identical to 5e-16). The deliverable flipped from "extract a missing primitive"
to "consume the rule that now exists" — a different recommendation. What caught
it was **re-running the census greps verbatim at close and following one
surprising hit**, not reading the diff. So: at close, re-run the *searches whose
EMPTINESS is a finding* (a "zero consumers" / "zero hits" claim is the most
drift-fragile kind of claim there is), and when `git log <open>..<close>` shows
commits in the neighbourhood you audited, read the NEW code — it may be your
answer.

How to apply: for any brief phrased "ahead of a surgical carve / before we change
X", open with `git status --short` + `git diff --stat` (and re-run both at the
end). If the carve is underway: (1) keep the audit sections as taken — they are
the record of what the carve was walking into — but (2) add a terminal
reconciliation section that verifies each finding **by runtime probe against the
final tree, not by reading the diff** (probes caught that `SpatialWrap.is_adjointable`
flipped `False→True` while `permutes_ordinates` correctly did NOT — a distinction
the diff hunk alone rendered ambiguous), and (3) lead the report with the
still-open items, since the closed ones are now archaeology. The highest-value
finding in such an audit is the item the carve did NOT reach.

---

## L-011 -- A docstring that DELEGATES ("the sweep handles it", "whoever orchestrates it") is the highest-yield falsity shape — grep the named MECHANISM, never the named symbol

Investigating the periodic-BC claim "*the SN sweep handles the spatial wrap via
its own face-pair indexing*", the mechanism (`face-pair indexing`) did not exist
anywhere in `orpheus/sn/` — no face→face map, no `partner_face`, nothing. Three
more falsities sat in the same file family, all the same shape: prose that hands
a responsibility to an unnamed other layer. Delegation claims are unfalsifiable
by the reader (the work is "over there"), so they survive refactors that would
have deleted an ordinary wrong sentence.

Two techniques that made it cheap:

- **Grep the MECHANISM NOUN, not the symbol.** `grep "PeriodicBoundary"` returns
  40 live hits and looks healthy; `grep "face-pair"` returns 2 — both of them the
  claim itself, restated. The noun the claim invents is the fastest disproof.
- **The SIBLING METHOD that REFUSES the same thing is a free oracle.** The
  diffusion realizer raises `BoundaryError` on the identical law with the exact
  structural reason ("*couples a face to its OPPOSITE face … no slot for
  cross-face coupling*"), while SN silently realizes an identity. When one method
  in a polymorphic family refuses and another accepts, the accepting one is the
  suspect — read its acceptance, not its docstring.

Also: the claim may be **half-stale**. The brief attributed it to two files; one
had been rewritten already and now carried a DIFFERENT unsatisfied claim. Always
re-locate the quoted prose before judging it (Operating Principle 5 applied to
prose, not just issues), and report where it actually lives.

How to apply: for any "is this doc claim true?" task, (1) extract the invented
noun and grep THAT; (2) check whether a sibling implementation refuses the same
input and why; (3) check the strict-`xfail` set — a `pytest.mark.xfail(strict=True)`
row naming the gap is the project's own admission that the claim is false, and is
better evidence than any prose.

---

## L-013 -- For "what breaks if this numeric primitive changes?", SWAP IT AND RUN — a grep-classification of exact assertions is guesswork

Auditing #325 (trig-evaluated → algebraically-generated quadrature nodes), a grep
for `assert_array_equal` across the 50 consuming test files returned ~200 hits.
Classifying those by reading would have been slow AND wrong. Instead: a **pytest
plugin that swaps the primitive at `pytest_configure`** and a run of the consuming
surface. 3024 tests, ~9 min, answer = **exactly 1 failure + 1 DriftWarning**. The
audit's central number went from "~200 candidates to triage" to "2 items", measured.

Three sub-lessons that generalize:

- **The dangerous class is a FROZEN right-hand side, not an exact comparison.**
  `assert_array_equal(route_A, route_B)` computed in the SAME process from the
  same inputs is *immune* to an input perturbation — both sides move together.
  Only a comparison against a stored `.npz`/`.npy`, a hardcoded literal, or a
  hash can move. Classify by "is the RHS frozen?", never by "is the comparison
  exact?". Nearly all of the ~200 hits were route-equivalence and immune.
- **Patch every re-export, not just the definition.** `from X import f` binds a
  name; patching `X.f` misses `directional.f`, `registry.f`, and both
  `__init__.f`, plus any dataclass field that CAPTURED the function object (the
  registry's `QuadratureSpec.factory` needed `object.__setattr__`). Patch the
  module list AND the captured references, then print a confirmation line from
  `pytest_configure` so a silent no-op swap is impossible.
- **The consuming-file list from grep is INCOMPLETE — run the sibling suites
  too.** `test_dd_regression.py` never spells `.product(`; it reaches it through
  `_generate_snapshots.CASES`. That file held the ONLY moving snapshot. A
  second batch over the whole owning directories (`tests/sn/regression`,
  `tests/moc`, …) is what found it.

Two more findings from the same audit, both re-usable question shapes:

- **A guard test's FIXTURE ENUMERATION is where vacuity hides.** A test asserting
  "no shipped quadrature has a cosine in the round-off band" built its list from
  `gauss_legendre` + `lebedev` only — excluding `product`, the one family that
  violates it. The assertion was strong and the *sample* was empty of violators
  (vv Mode 7). When asked "does this guard bite?", read the parametrize/fixture
  LIST first, then the assertion.
- **Check whether the "new" hazard already exists on the sibling.** #325's ties
  looked like a new reproducibility hazard — until measuring `level_symmetric`
  (already algebraic, already exact) showed 18–216 ties per rule TODAY, with
  `np.argsort` kinds already disagreeing at LS6+. Exact symmetry CREATES ties;
  the already-exact family is the free oracle for "is this consequence new or
  pre-existing?" (Same oracle move as L-011's sibling-that-refuses.)

---

## L-014 -- To adjudicate an ALGORITHM against the literature, read the source's DERIVATION and its INDEX-DOMAIN sentence — the equation alone is permutation- and domain-agnostic

Asked whether the cylindrical α-recursion's ordinate ordering is "correct or merely a
convention" (#326), the theory page and the code agreed with each other and both
matched the published *equation* — so an equation-level check said "fine". The answer
was in two places the equation is not:

- **The source's DERIVATION names what the quantity IS.** Hébert doesn't just state
  `α_{q+1/2} = α_{q−1/2} + 𝒲μ`; two lines earlier he *defines* `α ≡ 𝒲_p·η_{q+1/2}` —
  the tangential cosine at a real boundary. That single definition converts "which
  ordering is conventional?" into "which ordering reproduces the closed form?", i.e.
  from a taste question into a decidable one. **Always read the paragraph that
  PRODUCES the equation, not the equation.**
- **The load-bearing sentence is PROSE that bounds the index range.** "Each axial
  level contains 2ℱ(p) base points in interval `0 ≤ ω ≤ π`" and "the weights are
  normalized on each level to sum to `2√(1−ξ²)`" are the two sentences that decide
  the whole issue (the level is a HALF range; ORPHEUS spans the full circle). Neither
  is an equation; a grep for `\alpha` finds neither. Grep the OCR sidecar for the
  *domain* words — `interval`, `octant`, `normalized to`, `range`, `for m = 1 … M` —
  right after you find the equation.

Two corollaries that generalize past this issue:

- **A recursion is only as meaningful as its enumeration.** Any cumulative recursion
  (`x_{k+1} = x_k + f_k`) telescopes under EVERY permutation, so its closure test, its
  sum rule, and its "step" identity are all permutation-invariant — structurally blind
  to the very ordering they appear to certify (`vv-principles` Mode 12). When a brief
  asks "does this gate adjudicate X?", check whether the gate's quantity is a
  telescoping sum before reading its assertion.
- **Find the closed form first; it is a cheaper oracle than an MMS.** The whole
  question collapsed to a pointwise identity (`α == −W·ξ` at the boundary, exact via a
  Dirichlet kernel) — a millisecond quadrature-only check, versus the proposed
  "run the L1 MMS suite under each candidate ordering". And the MMS turned out to be
  Mode-7 blind anyway (both ansatzes lived inside the symmetric sector). **When a plan
  proposes an expensive reference to settle a discretization choice, spend one query
  asking whether the coefficient has a closed form.**

How to apply: for any "is this convention or is it determined?" brief, (1) pull the
source's derivation paragraph, not its equation; (2) grep the sidecar for the index-domain
prose; (3) classify each candidate gate as telescoping-invariant / fixture-restricted /
frozen-RHS before believing it bites; (4) hunt the closed form before endorsing a
reference solve.

---

## L-015 -- To test a "this DOF is redundant, fold it" hypothesis, enumerate the FUNCTIONALS, not the algorithm — and first ask whether the fold is of the ALGORITHM or of the STATE

Asked to map a half-range azimuthal level (#326) and to try to REFUTE the framing
"the redundancy IS the bug", every attempted refutation aimed at the marching
algorithm (does alpha still close? does the redistribution term survive doubled
weights? do the specular BCs still pair?) came back CLEAN. The one real break was
somewhere the algorithm never looks: a **functional of the state**. A fold with
doubled weights reproduces every *even*-parity spherical-harmonic moment to 5e-16
and turns the *odd*-parity one from its structural `-1.3e-16` into `+2.94`.

The reusable move, in two parts:

- **The sweep is parity-blind; the analysis face is not.** A quotient by a
  symmetry group G is exact on the G-invariant sector and meaningless on its
  complement. So enumerate every INTEGRAL the code takes of the state — moments,
  currents, leakage, inner products, the adjoint metric — and classify each
  integrand's parity under G. Even-parity functionals survive the fold with
  reweighting; odd-parity ones are *out of the space*, not "inaccurate". That
  reframing is what turns a defect into a typed obligation (restrict the trial
  space / a Petrov-Galerkin analysis face) instead of a patch.
- **Split "fold the algorithm" from "fold the state" BEFORE scoping.** They read
  as the same proposal and have opposite blast radii. Folding only the MARCH and
  lifting the partners back into the full state buffer makes the symmetry hold by
  construction, kills the ordering ambiguity, AND makes the functional break
  vanish (the integrals see the lifted, exactly-symmetric state) — at the price of
  no memory saving. Folding the STATE is the memory win and pulls in the trial-
  space restriction, the partition-by-sign consumers, and every `n_dof`/Krylov
  resize. The brief usually means the first; the headline number ("2x fewer
  unknowns") advertises the second. Name the split in the deliverable.

A third, cheaper corollary from the same audit: when two candidate constructions
both satisfy the criterion the issue argues from (here, both half-range rules
reproduce the alpha closed form with the SAME constant), **the criterion does not
discriminate them — go find the predicate that does.** It was a one-line structural
predicate elsewhere in the tree (`0 < tau_raw[0] < 1`) that flipped an entire
solver route on for one candidate and not the other. Sweep the STRUCTURAL
PREDICATES the codebase already keys behaviour on, and evaluate each candidate
against them; that is where the real cost difference lives.

---

## L-016 -- A stored NUMERIC PROPERTY (an exactness/order/rank tag) is a claim: sweep it, and first establish what the SYMMETRY gives for free

Surveying the quadrature landscape (2026-08-01), the single highest-value finding
came from ~40 lines of probe: brute-force sweep the monomials and ask "what is the
LARGEST degree at which EVERY monomial is exact?", per rule, per parameter. Result:
`level_symmetric_sn` tags `degree_of_exactness = N-1` and measures **3 for every
N** — a 12-degree over-claim at S16, with a live consumer (the registry selector
returns it when you ask for degree 15). Two moves made it decisive:

- **Establish the FREE baseline before crediting the construction.** I built a
  RANDOM `O_h` orbit with equal weights summing to 4π and measured it: degree-3
  exact, fails at 4 — identical to the real rule. So the rule's entire measured
  exactness is a consequence of the `invariance_group` tag it already carries, and
  the level construction contributes nothing. Without that control I would have
  reported "degree 3, not N−1" instead of "the number is not just wrong, it is
  redundant with another field". **For any "does this property hold?" audit, first
  measure what a structurally-trivial object with the same declared symmetry gives.**
- **Read the WEIGHTS, not the nodes, when a moment claim fails.** `n_distinct_weights == 1`
  for every N immediately named the root cause (equal-weight, not the cited
  Carlson-Lathrop moment-matched construction). A node-level diff would not have.

Why no test caught it (the reusable diagnosis): every test asserted the **tag**
(`assert m.degree_of_exactness == sn_order - 1`) and every *property* test stopped
at degree 2 — inside the sector the symmetry makes free (vv Mode 7,
fixture-restricted). **A tag-pinning assert is not a property test; when a brief
says "audit claim X", grep the tests for `== <the tag>` and treat every such line
as evidence that the property is UNTESTED.**

Also from the same survey, two cheap discriminators worth reusing:
- **When a docstring says a `min()` is "conservative" over two incommensurable
  units, look for the unstated MAPPING before calling it a bug.** The product
  rule's `min(2n_mu-1, n_phi-1)` measured CORRECT and SHARP on both branches,
  because for `x^a y^b z^c` on `S²` the max azimuthal frequency is exactly `a+b`.
  The defect was the missing mapping, not the arithmetic — a different (and much
  cheaper) deliverable than "the formula is wrong".
- **A structural FLAG's docstring and its registry ENTRIES can disagree; measure
  to decide which is wrong.** `half_range_clean`'s attribute docstring said
  "Lebedev and level-symmetric are not"; the entries said LS=True, Lebedev=False.
  Measured `w(z>0)/w_tot`: LS/product exactly 0.5, Lebedev 0.33-0.43 (equator
  nodes). The ENTRIES were right. Never assume the prose is the ground truth just
  because it is longer.

---

## L-017 -- Before counting a retirement's blast radius, check whether a NON-target sibling shares the target's name; and never accept a test's self-description of what it pins

Auditing four Gauss-rule retirements, a bare `grep gauss_legendre tests/` returned
~570 lines. The true blast radius of the target was **2 files**. The other **450
lines** were `Quadrature.gauss_legendre(...)` — a *classmethod* in a sibling module
of the SAME package, spelled identically to the module-level function being retired,
and emphatically not retiring. Reporting the unanchored number would have inflated
the scope ~200x and buried the one finding that mattered.

This is L-009's substring trap one level up: not `gains` ⊂ `against` (a lexical
accident) but a genuine **namespace collision inside one package** — the factory
classmethod named after the rule it wraps. It is the NORM in a
`Facade.rule()` → `rules_x.rule_on_y()` layering, not an exception.

- **How to apply:** the FIRST action of a retirement audit is a two-number probe —
  `grep -c '<name>'` vs `grep -c '<anchored form>'`. Report the delta as a
  named hazard, and hand the implementer the anchored pattern explicitly (they
  are usually a different agent). Anchor on the call shape (`[^.]name(`) or on
  import lines, not on the bare token.

**Second half — a test's docstring is not evidence of what the test pins.** The
audited `test_..._bit_identical_to_legacy_adapter` self-declared as "the
**load-bearing contract** for the refactor: if the nodes drift even at the last
bit, the regression snapshots will silently mis-compare", and used
`np.array_equal` for emphasis. Reading the RHS's call chain showed it was
`Quadrature.gauss_legendre(n)` — which *calls the LHS function*. Same process,
same source: pure route-equivalence, immune to the exact drift it advertised
(L-013's frozen-RHS rule, but disguised by a confident docstring and by the two
sides having DIFFERENT SPELLINGS). The real characterization surface was a set of
`.npz`/`.npy` snapshots that never name the symbol.

- **How to apply:** for every test you classify as CHARACTERIZATION, resolve the
  right-hand side's call chain one hop and ask "does this move when the SUT
  moves?" Two differently-spelled call routes that converge on one implementation
  read as independent and are not. Then go find the real frozen baselines by
  `find tests -name '*.npz' -o -name '*.npy'` — a grep of the symbol will never
  surface them.

A third, cheap corroboration from the same audit: a **captured function object in
a dataclass field** (`QuadratureSpec(factory=gauss_legendre_on_mu)`) is a live
production consumer with ZERO graph edges — `callers()` reported it nowhere. L-013
already flags this for *patching*; it applies identically to *auditing*.

---

## L-018 -- Before scoping a change to a STATIC TABLE, measure which rows are still consulted; and when one tag routes through two dispatch branches, the discriminating fixture is the one that FAILS

Mapping the blast radius of parameterizing `SubgroupOfO3.Z2` by its mirror plane,
the two findings that reshaped the answer both came from measurement, not reading:

- **Half the table was dead code.** `_NAMED_LATTICE` looks like 5 load-bearing
  `Z2` edges. But `_contains` decides **finite × finite by computed matrix
  containment** and only falls back to the table when one side is *continuous* —
  so `(Trivial,Z2)`, `(Z2,Oh)`, `(Z2,Ih)` are never read. The change needed **2**
  relations, not 5. Generalizes: **a hand-written lookup table that sits behind a
  computed fast path has dead rows; enumerate the table and ask, per row, "which
  branch answers this?" before costing a change.** (Corollary for a NEW tag type:
  both `_contains` and `_check_invariance_1d` end in a bare `return False`, so an
  unhandled tag gets a *wrong-but-silent* answer — measured `O3.contains(Mirror('x'))
  = False`. Check the fallthrough, not just the arms.)
- **A tag with two dispatch branches is invisible on the fixture both accept.**
  `Z2` means "σ_z" on 3-D nodes and plane-free "`x → −x`" on 1-D nodes. Every
  shipped gate uses Gauss-Legendre / Lebedev — sets closed under **all three**
  coordinate mirrors — so σ_x and σ_z agree and the overload cannot be seen. The
  discriminator was a set that *breaks* the symmetry: embed an asymmetric μ two
  ways and the answers split (`True` vs `False`), exposing a false certification.
  **When a brief asks "is tag X consistent across its consumers?", build the input
  that FAILS the property and check whether the two routes still agree** — an
  input that passes proves nothing about which question was asked. (Sibling to
  L-016's free-baseline control and vv Mode 7.) The cheapest version: hunt a
  SHIPPED datum that already discriminates — `product(4,3)` is closed under σ_z
  and not σ_x, so no synthetic fixture was needed at all.

---

## L-020 -- The BRIEF's own timeline is a claim; and a prior audit's "structurally cannot express X" EXPIRES the moment a substrate lands

Re-auditing a 1-day-old boundary-layer audit (2026-08-03, `refactor/operator-strategy-layers`),
the two findings that reshaped the deliverable were both about **inherited claims
that had a timestamp**, and neither was in the list of things I was asked to check:

- **The dispatch brief said "what landed AFTER that audit: B3.4a/b/c".** Measured:
  B3.4a/b/c landed **2026-08-01**, the audit was written **2026-08-03 00:43**, and
  the audit *cites B3.4a and B3.4c by name in its own body*. The brief's framing
  was backwards, and had I trusted it I would have gone hunting for changes that
  could not exist, and mis-attributed `SpatialWrap`'s `is_adjointable` flip to a
  post-audit event. **One `git log -1 --format="%h %ad" --date=iso <hash>` per
  named commit, plus the target document's mtime (`stat -f "%Sm"` on macOS —
  `ls --time-style` is GNU-only), settles it in one call.** Operating Principle 5
  says verify the ISSUE's premise; this extends it: **verify the BRIEF's own
  timeline the same way.** A brief is written from the dispatcher's memory, which
  freezes exactly like an issue body does.
- **The audit's negative capability claim was half-stale in 6 hours.** It wrote
  "`symmetry.py` **CANNOT express** the periodic translation — `SubgroupOfO3` is
  origin-fixing; explicitly out of reach today." True at write time. Six hours
  later a *different* campaign step (G3) landed `RigidMotion`, an E(d) element
  carrying a translation part with a `translation_by` constructor. The tag layer
  still can't name it — so the claim is now true of `SubgroupOfO3` and false of
  the substrate, which is a materially different recommendation.

The reusable rule: **an "X cannot be expressed / X does not exist / X has zero
consumers" verdict is the most perishable kind of finding, and the thing that
expires it is usually a sibling campaign, not a change to the audited code.** So
when re-auditing, don't only re-run the emptiness greps (L-012 Sharpening 2) —
**read the NEW module any intervening commit added**, and ask the negative claim
against it directly. Here that was ~10 lines of probe (`RigidMotion.translation_by`,
`on_points` vs `on_directions`, and `_orbit_closure` fed the translation) and it
turned "out of reach" into a precise three-tier answer: the ELEMENT exists, the
TAG cannot name it, and the certifier correctly REJECTS it because a translation
is the identity on directions.

Corollary worth reusing: **when an audit reports N spellings of one concept, check
whether they differ in DOMAIN/CODOMAIN before endorsing a unification.** The
"four vocabularies of σ_a" turned out to be four different categories (deck
transformation `Γ₊→Γ₋` / constitutive kernel `Γ₋→Γ₋` / a curried factory /
a subgroup tag), and three of the four are a deliberate, documented,
sweep-schedule-load-bearing split. The genuine duplication was somewhere the
audit never looked: **three live spellings of the axis-letter→index table, beside
a docstring asserting it has "ONE home"**. Count the tiers first; the real twin is
usually the boring shared primitive, not the named types.

---

## L-019 -- Hunting a hidden TRANSFORMATION: read the chart-defining ASSIGNMENT and COUNT the partition's parts — the matrix and the docstring are the two places it isn't

Auditing "where does the angular layer rotate/reflect without naming a group
element?", the two highest-value findings were invisible to every grep I would
naturally run (`rotat`, `mirror`, `np.eye(3)`, `wigner` — all ran, all missed
them), and both were found by a different move:

- **A coordinate CONVENTION applied by variable choice is a group element with
  no matrix.** `cos_theta = mu_x` (one assignment, in a basis module) makes every
  `Y_ℓ^m` in the project the textbook harmonic composed with the 120° rotation
  about `(1,1,1)` — a real `O_h` element, in the checker's own `_octahedral_ops()`,
  and *not expressible as a tag* (`Cn(3)` is about z and measurably excludes it).
  Nothing can test it: there is no matrix for an invariance check, no adjoint,
  and a rename breaks nothing. **How to apply:** for any "find the hidden
  transformation" brief, grep the *chart-defining assignments* — `cos_theta =`,
  `= arctan2(`, `polar axis`, `_ = nodes[:, k]` — and reconstruct the implied
  frame matrix by hand, then ask `_group_elements(tag)` whether the machinery can
  NAME it. Constructing the 3×3 and testing membership is ~10 lines and decides it.
- **A partition predicate's LABEL SET is an orbit-type stratification — count the
  parts before believing the name.** `Quadrature.octants` is documented as the
  8-way sign decomposition; measured it returns **26** parts on `lebedev(17)`
  (8 chambers + 18 walls) and 2 on a slab. The `0`-component labels are exactly
  `Fix(σ_a)` — the singular set the same package computes EXACTLY elsewhere. One
  `len(...)` per shipped rule turned "ad-hoc sign classification?" into a table.

Two corollaries that generalize past this audit:

- **The tolerance-family census is a cheap, high-yield side product.** Three
  epsilons for one question (`1e-15`, `8.88e-16`, `1e-14`), all provably idle
  (measured min genuine `|cos|` = `1.57e-1`), and the one comment defending the
  first points at `_DEGENERATE_ABS_MU_THRESHOLD` — a symbol that exists **nowhere**.
  That is L-011's delegation-shaped falsity in a `#:` comment rather than a
  docstring: whenever a constant is justified by "keep in lockstep with X", grep X.
- **Measuring a docstring's claim on the DEGENERATE input finds the bug the test
  fixture can't.** "For slab GL1D only the `m=0` harmonics are non-zero" is false
  at `ℓ≥2` (measured ~0.83 in the `m>0` slots; a 4.4× reconstruction difference),
  because the slab's `(μ,0,0)` embedding makes `(cos φ, sin φ) = (0,0)` — not a
  point of `S¹` — and the on-axis guard never fires. The only `P≥2` test in the
  tree uses a 3-D Lebedev rule, where the chart is fine. **When a docstring says
  "for the degenerate/1-D case, X vanishes", evaluate X on that case.** It costs
  one probe and it is exactly where the fixtures aren't.

---

## L-023 -- "N spellings of one concept" is a SYMPTOM: find the primitive that DISCARDED the information. And a brief's named EXEMPLAR consumer is a claim to verify

Auditing "convergence has three spellings in `sn/solver.py`" (2026-08-08), three
moves reshaped the deliverable, and all three generalize to any
"one concept, many transcriptions" brief:

- **Count the spellings, then go UP one hop and ask who threw the answer away.**
  There were FIVE, not three (the brief's own count was low: a 4th differed only
  by `<=` vs `<`, a 5th was the Krylov arm). But the finding that changed the
  scope was that `power_iteration` *knows* whether it broke out of its loop or
  exhausted `max_iter` and **returns a 3-tuple with no flag**. Four production
  solvers consume that tuple; one re-derives the fact correctly, one hardcodes
  `True`, three don't try. **The N transcriptions are not N independent bugs —
  they are N callers re-deriving a fact the callee already had.** So: for any
  "why is this spelled N ways?" brief, read the CALLEE's `return` statement
  before mapping the callers. If the callee computes and drops the quantity,
  the fix is one hop up and the deliverable flips from "unify the N sites" to
  "the primitive's return type is wrong". (Corollary that decided the
  local-vs-shared scope: the eigenvalue path records only a COUNT, never the
  inner residuals — so a derived flag there is structurally outer-only until
  new plumbing lands. That constraint is invisible from the construction site.)
- **A default value can be the mechanism behind the hardcoded lie.** The type's
  own `converged: bool = True`, plus a delegate returning `True` when the
  history is `None`, means "forgot the kwarg" and "no diagnostics" both read as
  success. Grep the *defaults* of any boolean claim field before writing up its
  hardcoded sites — the literal `=True` at one call site is usually the type's
  posture made visible, and the type-level fix reds the unit tests that PIN the
  default (find those; they are the contract change made visible, and they get
  rewritten, not deleted).
- **The brief's named EXEMPLAR is a claim, like its timeline (L-020).** The
  brief said "`test_dsa_rate.py` caps `max_inner=50` on purpose" — measured, its
  helper defaults to **4000** and the `=50` sites are HEADROOM on tests asserting
  landing in 2–3 iterations. Had I inherited it, the opt-out population would
  have been sized wrong in the direction that decides the default policy. One
  `sed` of the helper's signature settled it. Same move as verifying an issue's
  premise, applied to the one datum a brief states with most confidence.

Cheapest high-leverage probe of the whole dispatch: **grep the pytest config for
`filterwarnings`**. Zero occurrences ⟹ a `warn`-by-default policy costs *zero*
test churn, which resolves the raise-vs-warn question that the rest of the audit
could only frame. When a brief asks "should this raise or warn?", one grep of the
consumer's escalation config often decides it outright.

---

## L-022 -- On a "what REMAINS of issue X" recon, the highest-yield staleness class is the campaign's own MID-FLIGHT PROSE

Mapping the #325/#326 remainders (2026-08-08) after a campaign had landed the
fixes in steps, the four falsified-prose finds all had the same shape: an
in-code note the campaign ITSELF wrote at an intermediate step, describing the
then-true remainder — falsified by a LATER step of the same campaign. Found in
one pass: a "See Also" saying the consumer is "today still welded in as
`linspace`" (the repoint landed later); a `.. caution::` saying "the remaining
half is the checker's own C_n/σ_v operators, which still [evaluate trig]" (the
checker was repointed to the shared generator two steps later); a docstring
citing a gate by a test name that never existed; a theory page still saying
"reflection partners … cached at construction" (retired). L-020 says claims
expire; this names the SEARCH STRATEGY for remainder recon: don't only re-run
emptiness greps — grep the campaign's own interim honesty vocabulary ("today
still", "remaining half", "until then", "not yet", the caution/note directives
in the touched modules) and re-verify each hit against the tree. The notes
written to be honest mid-flight are precisely the ones nobody returns to.

Also from the same dispatch, two cheap reusable probes: (a) a "did the
acceptance gate land?" check must read the gate's FIXTURE ENUMERATION, not its
assertion — the eps-gap gate existed and still enumerated only GL+Lebedev,
skipping the family (`product`) that motivated the issue (L-013's finding,
still unfixed 7 days later — report it as the remainder, citing the earlier
find); (b) when a brief hands you six "the campaign landed X @ hash" claims,
verifying them costs one read each and all six held — but the SAME session's
plan had already existence-checked its own NEXT pointer (per plan-authoring
§1), which is why; trust rises with the tree's own hygiene, never with the
brief's confidence.

---

## L-021 -- A brief's TYPE table is a claim about MATERIALIZED objects; and an all-green "does it break?" run may have measured INERT, not SAFE

Mapping the G6.3 boundary-operator binding sites (2026-08-04,
`refactor/operator-strategy-layers`), the two findings that reshaped the deliverable
were both about the *shape of the question*, not about the code:

- **The brief handed me a four-row typing table** (`γ₊ : Γ→Γ₊`, `G : Γ₊→Γ₋`,
  `R : Γ₋→Γ₋`, …) and asked "where is each constructed?". The table was correct
  MATHEMATICS and had **no code counterpart**: `law.geometry_map` returns a
  `SelfPairedDeck`/`SpatialWrap` and `law.response_kernel` a `SpecularReemission` —
  *descriptors*, not `LinearOperator`s. The realizer emits ONE operator per law with
  `G` and `R` already collapsed, so two of the four rows had **zero** construction
  sites and the "endomorphism of `Γ₋`" row was un-bindable in principle. Had I filled
  the table by finding the nearest plausible object per row, I'd have shipped a map
  that told the implementer to bind a `PermutationOperator` as `G` — which is the
  SAME body two different laws reach while declaring the mirror lives in different
  tiers, i.e. structurally unfillable.
  **How to apply:** when a brief (or a theory page) names an arrow `A --f--> B`, the
  FIRST query is *"is `f` a materialized object?"* — one grep of its accessor's return
  type. Only then map construction sites. This is Operating Principle 5 applied to a
  TYPING premise: OP5 catches a premise that went stale; this catches one that was
  never true in code, only on paper. The deliverable's headline flips from "here are
  the sites" to "two of your four rows do not exist, and here is why they can't".

- **I did the L-013 move (install the change, run the suite) and it came back
  ALL GREEN — 4 941 bindings, ~5 100 tests, zero new failures. That result was
  nearly a false reassurance.** The composition G6.3 types is spelled as three raw
  `.apply` calls, never `@`/`+`, so **no composability gate exists on that path at
  all**. Green did not mean "the binding is right"; it meant "the binding is inert".
  Those are opposite messages to an implementer: the first says ship it, the second
  says ship it AND schedule the step that makes it bite.
  **How to apply:** after any all-green "what breaks?" measurement, spend one query on
  *"could a gate have fired?"* — find the gate (here: `OperatorSum`/`OperatorProduct`
  `__init__`), and check whether the audited path routes through it. Report
  `inert` vs `verified` explicitly. A measurement that CANNOT fail is Mode-7 blind at
  the whole-suite scale, and an audit that reports it as "safe" has laundered a
  no-op into evidence.

Two cheaper corollaries from the same dispatch, both reusable:

- **A predecessor survey from the SAME campaign expires like any other audit.** Two of
  the G6.0 survey's Tier-1 "pure omission, both types exist" calls were written against
  the pre-B3.4a operator shapes and were wrong by G6.1 (`AngularAverageOperator` is
  `Γ₊→Γ₋`, not angular→scalar; the boundary `TensorProductOperator`'s space is NOT
  `TensorProductSpace(a.domain, b.domain)` because the face space ALREADY is the
  product). L-020 says a sibling campaign expires a claim; this says **an earlier PHASE
  of your own campaign does too** — re-measure the predecessor's per-item calls, don't
  inherit them.
- **The cheapest cross-face threading is usually already computed and thrown away.**
  The periodic partner face — the one genuinely hard-looking argument in the whole map —
  is *returned* by the guard the realizer already calls and *discarded* at the call
  site. Before writing "X would have to be threaded", read the helper's return
  statement.

---

## L-024 -- A solver's NESTING SHAPE is a per-ENTRY-POINT fact, never a per-module one — trace each entry, and trace it by RUNNING

Surveying the #340 consumer landscape (2026-08-09, `refactor/operator-strategy-layers`),
the brief asked for "the level tree for an SN k-eigenvalue solve" as if a module had
ONE. It does not, and the difference is exactly what a recursive record has to model:

- `solve_sn` → `power_iteration(SNSolver)` → `SNSolver.solve_fixed_source` →
  `_solve_source_iteration`/`_solve_krylov` → `SourceIteration.solve`. The per-inner
  residual list reaches SN code, which sums `len(...)` and drops the rest.
- `solve_sn_adjoint` → `KEigenvalue.solve` → `power_iteration(KEigenvalue)` →
  `KEigenvalue.solve_fixed_source` → `SourceIteration.solve`. **One frame deeper, and
  the inner history is discarded INSIDE the shared numerics primitive** (`psi,
  _inner_residuals = self._inner.solve(...)`), so it never reaches SN code at all.

Two sibling public entries of the same solver, same math, structurally different
trees — and a design that models "the" tree from either one alone is wrong for the
other. Same shape one module over: CP nests outer → **per-group** inner (a loop the
SN path does not have, because SN's "within-group" `S` is full-multigroup with no
group loop); MoC's inner is a FIXED sweep count with no tolerance at all; diffusion's
inner is one LU back-substitution, i.e. no inner level.

**And the tree is not reliably readable from the static graph.** `SNSolver.converged`
has **0 `callers`** in Nexus — it is reached polymorphically through the
`EigenvalueSolver` Protocol inside `power_iteration`. A `callees`/`callers` walk of a
Protocol-dispatched driver understates the tree by exactly the levels that matter.

**How to apply.** For any "what is the iteration/level/stage structure?" question:
(1) enumerate the PUBLIC ENTRIES first and expect them to differ; (2) get the tree by
**running** — a ~30-line `sys.setprofile` probe filtered to 3 files and a `WATCH` name
set prints the nesting, the per-level call counts, and the truncation evidence in one
shot (cheaper and more honest than reading five call sites); (3) read the count against
the budget — my probe's `total_inner_iterations=1470` over 30 outers at `max_inner=50`
meant **30 of 30 inners hit the cap**, which no static read would have told me. Nexus
`runtime_ingest` is the durable version of this when a `cProfile` artifact exists;
`runtime_runs` returning `[]` means you write the probe yourself.

Corollary, cheap and reusable: **when a status is derived from a LENGTH, check whether
the length is injective.** `SourceIteration` appends one residual per iteration from
the second onward, so `max_iter=50` yields `len(residuals)=49` both for "exhausted" and
for "converged on the last possible check". A consumer reconstructing convergence from
the count is guessing on the boundary case — which is precisely why the status has to
be recorded, not recomputed.

---

## L-026 -- "Is this degraded condition DELIBERATE?" is decided by a COUNTERFACTUAL, never by the budget literal or the docstring — and the unit of the question is the SOLVE, not the test

Adjudicating the 20 tests #340's guard flip would newly warn (2026-08-10), the
three findings that decided the verdicts were all invisible to reading:

- **Run the test at the healthy setting and re-evaluate its OWN assertion.**
  A test whose docstring never mentions truncation can still be *load-bearing*
  on it. `test_collision_cache_invariance_under_source_iteration` starves its
  inner (7/7 at 50/50); at a converging budget the power iteration finishes in
  **3** outers and the test's own `len(keff_history) >= 5` non-degeneracy floor
  **FAILS**. Conversely a test that *looks* pinned can be free: the W1 iso gate
  carries two frozen in-module literals and survives the raise with 20–170×
  headroom. **Neither verdict is readable from the source.** One counterfactual
  solve per suspicious row settles it; both directions were surprises.
- **A per-TEST filter cannot separate two solves in one test body — so ask the
  question per SOLVE.** `test_inner_tol_bias_collapses_at_1e_12` runs a LOOSE
  leg (the studied bias — genuinely deliberate) and a TIGHT leg (the reference
  the assertion is made against). Measured, the *tight* leg also truncated in 3
  of 4 rows: the reference half of a bias measurement was itself biased. A
  marker "declaring" the row would have silenced that too. **The free
  discriminator was already in the census**: the per-test CALL COUNT column
  (2 vs 1) predicted exactly which rows had a second bad solve — read the
  counts before reading any code.
- **Never cost a budget raise by the budget ratio.** Converging the inner
  removes the increment suppression that was inflating the OUTER count, so
  wall time rises far less than the knob does — `[M]` a 3.5× budget raise cost
  **1.6×** (outers 4 → 3). Two rows I had pre-labelled "too expensive to fix"
  were affordable. Same mechanism, opposite sign, is what makes the fixture
  break above.

Two corollaries: (a) a fitted "projected iterations" advisory is exact at
ρ ≲ 0.98 (788 → 789 measured) and a **lower bound** at ρ ≳ 0.99 (2066 → 2601,
26 % low) — and the *first* failure is rarely the *worst*, so take the max over
the tree. (b) Adjudicating a population like this yields **stale-docstring finds
for free**, because the prose states the mechanism the measurement contradicts —
four here, incl. a documented drift magnitude wrong by four orders and a
"converged X" claim on a solve that never converged. Harvest them; they are the
same defect class the campaign exists to remove.

---

## L-025 -- "What is in scope at call site X?" is a FRAME-LOCALS probe, never a read; and a `[M]`-marked NEGATIVE claim is the most perishable marker in the project

Mapping the #340 N6b exit-residual scope (2026-08-10, `refactor/operator-strategy-layers`),
the brief asked "what is in local scope at each of five call sites". Reading the five
sites would have produced a plausible, incomplete, and in one place *wrong* answer. A
~20-line spy on the CALLEE — patch the module-global helper, read
`sys._getframe(1).f_locals`, print `{name: type}`, then call through — answered it
exhaustively in one run and found three things reading could not:

- **Names bound only inside a conditional branch.** On a carrying (sphere) mesh
  `final_state`/`final_ray`/`corner_state` are live at the call; on a slab they do not
  exist. A read of the function body shows the `if` and leaves you guessing which
  fixture reaches it. The probe just lists them, per geometry.
- **Whether the object in scope actually WORKS.** Having the name is not having the
  capability. Composing the residual *from the probed names* is one extra line in the
  spy and it found the two real holes: an LD (`spatial_basis_per_axis=2`) iterate makes
  `AngularResidual.from_balance` raise on the trailing moment axis, and the windowed 2-D
  SI arm holds a `HarmonicMomentFlux` moment iterate at the call — the full-angular
  reconstruction that fixes it is bound **28 lines later**. Both are invisible to a read;
  both are one measured `ValueError` in the probe.
- **The reach of a hazard.** One `inspect.signature(fn).parameters` sweep showed the two
  eigenvalue entries have no `scheme=` at all, so the LD hole cannot reach them — turning
  "the number is sometimes unavailable" into "unavailable on exactly 2 of 5 sites, under
  one named argument".

**Two sharpenings that generalize past this dispatch:**

- ⭐ **A `[M]` marker on a NEGATIVE existence claim is not stronger evidence — it is a
  claim that some measurement, answering some question, returned nothing.** The plan
  carried `[M] "solve_sn discards the solver it builds"`. Measured false: `solver` is
  bound and live at the warning call. The original measurement was almost certainly
  honest — it asked whether a returned `Solution` exposes the operators (it does not).
  The plan then reused that answer for a *different* question (what is in scope at the
  call site) and the polarity survived the change of question while the scope did not.
  ⟹ when a brief or plan hands you `[M] X is absent / X is discarded / X has no
  consumers`, re-measure it against YOUR question before building on it; the marker
  raises confidence, and for negative claims that is precisely the danger. (OP5 and
  L-020 say premises expire; this says the strongest-marked ones expire too, and the
  mechanism is question-drift, not time.)
- ⭐ **To adjudicate "can A be reused for B?", hunt the input that makes them DIVERGE
  most, never the fixture where both work.** Asked whether the within-group certificate's
  residual could stand in for the outer balance projection, the SI fixture showed the two
  defects within 2× of each other — reuse would have looked defensible. The **Krylov**
  inner (which converges its lagged equation to 1e-9 while the outer is truncated) put
  them **10⁶ apart** at outer #1 and 173× apart at the truncation point. Same shape as
  L-018: the discriminating fixture is the one that fails, and here it was a *shipped*
  configuration (`inner_solver="krylov"`), not a synthetic one. Corollary specific to
  residual/defect reuse questions: **read the rhs's PROVENANCE chain to the loop that
  produced it** — `q_driver` traced back through three frames to
  `power_iteration`'s `fission_source = solver.compute_fission_source(flux_distribution,
  keff)` taken BEFORE `solve_fixed_source`, i.e. lagged by one outer. That single read
  decided the question; the numbers only sized it.

**How to apply.** For any "what does the code have available at point P" /
"could P compute Q" brief: (1) write the frame-locals spy first — it costs one file and
one run and it is exhaustive where a read is a sample; (2) put the *attempted
computation* inside the spy, with a `try/except` that prints the exception type, so
"has the name" and "can do the job" are separated in the output; (3) run it on the
DISCRIMINATING configurations, not just the first one that imports (here: slab vs
sphere-carrying vs 2-D-windowed vs LD — four geometries, four different answers);
(4) sweep `inspect.signature` to bound the reach of any hole you find.

---

## L-027 -- Reconciling a survey: the highest-value finding is usually in NONE of the claims you were handed — diff the PRIMITIVE the survey is about, not only the code it audited

Re-verifying a 1-day-old survey of how CP/MoC/diffusion would adopt
`IterationRecord` (2026-08-10, `refactor/operator-strategy-layers`), all ten
handed claims (A–J) came back CONFIRMED or line-drifted, and the line drift was
mechanical and harmless. The finding that actually changed the design landed in
a commit the brief never mentioned: a sibling step added a **new field with a
`__post_init__` guard** (`IterationRecord.budget_name`, empty string raises) and
a matching `power_iteration(..., budget_name=)` keyword. SN passes it; the three
audited families do not — so all three are already emitting an outer record
whose advice names a knob their public entry points do not have. **No claim in
the survey could have caught this, because every claim was about the AUDITED
code and the change was in the PRIMITIVE being adopted.**

- **How to apply.** On any "reconcile this survey/plan against the tree" brief,
  spend one call on `git diff --stat <baseline>..HEAD -- <the primitive's
  module>` *before* walking the claim list, and READ the diff of the type the
  survey proposes to adopt. A new required field / new guard / new constructor
  keyword is a new obligation on every future producer, and it is invisible to a
  claim-by-claim re-check. (L-020 says a negative claim expires when a sibling
  lands a substrate; this is the positive twin — a *new requirement* appears
  and nothing in the document is even wrong.)
- **Corollary, cheap:** the `__post_init__` of the primitive is the fastest read
  of "what does N4 now owe per record" — three lines of validation named the
  whole new obligation.

**Second half — a handed COUNT that reproduces under no convention was
estimated, so report the convention, not just the number.** The survey said "37
`solve_cp(` call sites, all in `tests/`". Measured: 33 executable calls in
`tests/` (+2 in `examples/`, 4 in `.rst`), 35 grep-LINES under `tests/` for
`solve_cp(`, 56 bare-token lines under `tests/`, 88 tree-wide. No convention
gives 37, and the "all in `tests/`" half was simply false. Two moves make this
cheap and non-arguable: (1) print **several** counting conventions side by side
rather than picking one — the spread itself is the evidence the inherited number
was a guess; (2) treat the qualifier ("all in tests/", "zero production
consumers") as a **separate claim from the count** and check it independently —
here it was the qualifier, not the number, that hid two live `examples/` readers
of the exact two fields the campaign was about. Same family as L-023's
"the brief's named EXEMPLAR is a claim", applied to its arithmetic.

---

## L-028 -- On a UNITS/representation change to a shared producer, sort consumers by WHICH GUARD they sit behind — the loud path is the safe one, and the consumer with zero tests is where the garbage lands

**The measurement (2026-08-11, `angular_cell_edges_per_level` cosine→radians).** A
producer returns per-level angular cell edges; the proposal changed the cylinder
branch from the radial cosine to the march angle. Swapping it in-process
(L-013) split the five consumers into three sharply different classes, and the
class boundary was *not* "does it want a cosine" — every one of them did:

| class | consumer | outcome under the swap |
|---|---|---|
| **guarded / loud** | `morel_montry_tau_per_level` (+ everything downstream: the closure `__init__`, every cylindrical solve) | **RAISES** — its own P3 `τ ∈ [0,1]` guard fires (`τ₀ = 4.598`) |
| **unguarded but GATED** | `contamination_beta` | silently returns `−0.44 / −2.85` where baseline is `~1e-18`; two L0 tests assert `< 1e-14` ⟹ RED |
| **unguarded and UNGATED** | `alpha_defect_beta` (`1 − e²` with `e = π` ⟹ `π²−1`), `nu_closure_residual` (seeds `ν = e[0] = π`, divides by `e[-1] = 0.0` ⟹ `inf`) | silently garbage, **zero test consumers** — nothing anywhere reddens |

- **Why it matters.** The instinct is to fear the production path, because that is
  where the blast radius looks biggest. It was the *safest* consumer: it carries
  a validity predicate on the very quantity the change perturbs, so the change
  cannot pass it quietly. The danger sat in the sibling **analysis/derivations
  module** that consumes the same producer *deliberately, to avoid a twin* — a
  Cardinal-Rule-2 win that hands the diagnostic the producer's units without
  inheriting the producer's guard. And of its three functionals, the two with no
  test consumer are precisely the two that go to garbage undetected.
- **How to apply.** For any change to a shared producer's UNITS, RANGE, SIGN or
  ORDER: (1) enumerate consumers; (2) for each, name the guard it sits behind
  (`assert`, a range check, a raising predicate) and whether that guard reads the
  changed quantity; (3) grep each unguarded consumer for its own test consumers.
  Report the table in that order. A consumer with **no guard and no test** is the
  finding; a consumer with a guard on the changed quantity is a non-event that
  merely looks alarming.
- **Two cheap sharpeners measured here.** (a) **Check the ORDER, not only the
  units** — the cosine edges ascend (`−sinθ → +sinθ`), the angle edges *descend*
  (`π → 0`), so `np.diff(edges)` flips sign and every `(x − e[m])/(e[m+1] − e[m])`
  barycentric silently negates. A units change is often secretly an ordering
  change, and the docstring's "ascending in the radial cosine" is the contract
  that breaks first. (b) **Check for a ZERO in the new range** — the old range
  never contained an endpoint of 0 (`±sinθ`), the new one closes at exactly
  `ω = 0`, which turned a normalising division into `inf`. Ask "does any
  denominator read `edges[0]` or `edges[-1]`?" before costing the change.
- **Corollary for the range-assertion class the brief asked about.** No test
  asserted on the returned edge VALUES at all — not `[-1,1]`, not `Σ Δμ == 2`,
  not monotonicity. The two tests that call the producer directly only assert its
  *refusal* (`pytest.raises`). So the "a `[-1,1]` check would silently pass or
  fail wrongly" hazard did not exist, and its absence is the real gap: a
  single-source partition producer with no direct value gate.

---

## L-029 -- On a "should we build this CAPABILITY?" fork, MEASURE the capability against the cheaper alternative it competes with — a capability can be negative-value; and "does the primitive exist?" is answered per TIER, not per repo

Adjudicating #336's REFUSE-vs-REDUCE fork (2026-08-13), three moves decided it,
and all three generalize to any "add the capability or refuse the input" question:

- **Price the capability in the units the user would compare.** REDUCE
  (marginalize a 3-D rule onto the μ-line) is mathematically correct, well-posed
  on the flagship rule, and *strictly worse than the one-line alternative*:
  `[M]` the μ-marginal of `level_symmetric(N)` is a degree-`N+1` rule on `N`
  nodes where `gauss_legendre(N)` is degree `2N−1`, and on a smooth integrand the
  error gap reaches **7 orders of magnitude** at n=10 (1.8e-8 vs 2.7e-16), after
  constructing 120 ordinates to get 10. `product(8,8)`'s marginal is 17 nodes at
  degree **7** (GL(17) is 33). So the "larger capability" branch delivers a rule
  the user could beat by typing a different factory name. **Whenever a fork reads
  "cheap refusal vs larger capability", spend one probe measuring the capability's
  OUTPUT against the incumbent at equal cost** — the capability framing carries an
  unexamined assumption that more is better.
- **"Does the primitive already exist?" has a different answer at each TIER, and
  the reusability test is what the existing machinery MATCHES ON.**
  `DiscreteMeasure.pushforward` + `.consolidate()` is *exactly* the marginal
  (measure tier, mass-preserving) — it exists and is generic. One tier up,
  `LevelStructure.quotient` looked reusable ("descend a per-ordinate field along
  a reduction") and is not: it indexes the parent by `nodes[i].tobytes()` and
  REFUSES anything that is not a bit-for-bit selection, because *a quotient never
  moves a node* and a marginalization moves every node. ⟹ read the candidate
  primitive's **matching/precondition code**, not its docstring's verb; "reduces a
  measure" and "selects orbit representatives" read alike and compose oppositely.
- **A "guard exists on this arm" claim needs the guard's QUANTITY checked against
  the symmetry of the input class.** The sphere arm's `_assert_alpha_dome_closes`
  looks like an admission guard on the quadrature; `α_{M+1/2} = −Σ w μ` is the
  rule's FIRST MOMENT, which every `O_h` cubature has *bit-exactly zero* (`[M]`
  `0.000e+00` on LS4/6/8, product, folded_product). So it passes on all five
  wrong-domain rules. Same shape as L-014's telescoping blindness: before
  crediting a guard, ask what functional it computes and whether the input class's
  symmetry annihilates it.

Cheap corollary that sharpened the verdict: **sweep every shipped rule through
the real constructor and read the traceback FRAME, not just the exception.** All
five 3-D rules were refused at the identical line — so REFUSE would not change
the accepted set by one element, only the layer and the wording. That reframes
the deliverable from "which behaviour do we want" to "this is a diagnostics/
vocabulary fix", which is a much smaller and differently-owned piece of work.

---

## L-030 -- When you INSTRUMENT a seam, count the seam's `return`s first — a "this is the ONE construction site" docstring is a claim, and trusting it silently under-scopes the census

Scoping a change to "what the SN solve returns" (2026-08-14), I patched
`_package_solution` — whose docstring reads *"The ONE `SolutionBase`
construction convention … the single boundary where converged iterates become
the typed return"* — and ran the suite. It reported **10** affected tests. The
docstring is present-tense FALSE: `_solve_fixed_source_si` and
`_solve_fixed_source_krylov` each `return Solution(...)` directly, so the whole
FORWARD fixed-source family bypassed the probe. Re-running with all three sites
patched gave **24**. The undercount was 2.4×, and it was silent — the probe was
green, the plugin printed "INSTALLED", and every number it produced was real.

What exposed it was not reading the docstring's neighbourhood but an
**arithmetic inconsistency in the census itself**: two tests showed a target
MESH construction (`SNMesh.__init__`, an exhaustive patch) and *no* exit, while
their bodies plainly called `solve_sn_fixed_source`. A counter that cannot
explain its own zeros is the tell.

- **How to apply.** Before instrumenting any seam, run
  `grep -n "return <Type>(\|return _<shared_tail>(" <module>` and patch **every**
  site. Cost: one grep. Then build the census with **two counters at different
  depths** — one at a constructor you can prove is exhaustive (`__init__` of the
  config object) and one at the seam under study — so that "config built but seam
  never reached" is a visible row you must explain, not an absence you never see.
  A single-counter census cannot detect its own blindness.
- **Sibling to L-017's second half** (a test's self-description is not evidence of
  what it pins) and to L-011 (delegation-shaped prose): here the false prose is a
  *single-source-of-truth claim on a shared tail*, which is the highest-yield
  variety, because the whole point of such a tail is that readers stop looking
  for siblings. Expect it wherever a refactor introduced a shared packaging
  function and one arm was left behind — and note the leftover arms are usually
  the ones with a DIFFERENT carrier shape (here: raw iterate vs cell-average
  view, and a Krylov arm whose boundary block is a residual, not a flux), which
  is precisely why they were not folded in.

---

## L-031 -- Cite a doc/code anchor by GREPPING its string, never from the `sed` range you happened to read — and spot-check before shipping, because the error is silent and uniform

Producing the MC `implements` declaration map (2026-08-18) I cited ~19 page
anchors as `monte_carlo.rst:NNN-MMM`, each inferred from the offsets of the
`sed -n 'A,Bp'` window the text appeared in. Spot-checking 10 of them before
delivery showed **every single one was off by 3-10 lines**, and I corrected
**18 of 19**. The content of each citation was right; the address was wrong.

Why it is worth its own lesson rather than folding into L-003 (which says
*don't front the line map*): here the line map was the DELIVERABLE — the parent
was going to land declarations from it, so a reader following
`monte_carlo.rst:415` lands on an unrelated `**Claim:**` paragraph and cannot
tell whether my analysis or their tree is wrong. The failure is also **uniform
and silent**: reading a 190-line window makes every offset inside it drift the
same way, so nothing looks anomalous and no single citation "feels" suspicious.

- **How to apply.** When a deliverable carries `file:line`, spend ONE grep per
  anchor on its distinctive string (`grep -n "This is the convention used by" <file>`),
  batched — 19 anchors is one `grep -n "A\|B\|C..."` call, cheaper than the
  re-read that discovers the drift. And prefer the ANCHOR STRING over the number
  where the reader can grep it themselves.
- **Corollary — spot-check by SAMPLING before shipping.** Because the error is
  uniform, 3 samples detect it as reliably as 19. Print
  `sed -n "${a},${b}p"` for a handful of your own citations and read what comes
  back; if one is off, they all are, and the batched grep is the repair.

---

## L-032 -- Hunting an equation's IMPLEMENTER: the authored rationale usually sits on a NEIGHBOURING label, and "nothing implements it" is the verdict that fails flatteringly

Finding implementers for 17 equations across 9 theory pages (the `implements`-declaration
campaign, 2026-08-18) — all 17 came back DECLARABLE, none `NOTHING:<kind>`. Three moves
decided it, and each generalises to any "what code realises this documented claim?" task:

- **The `.. (vv-status rationale)` that names the answer is usually attached to a
  DIFFERENT label.** Measured: 8 of 17. `bare-slab-keff`'s rationale is a comment inside
  the *test's* `pytestmark`; `moc-mms-psi-ref`/`-qext` are named inside
  `moc-mms-reference-equilibrium`'s sentinel ("the MOC operators the MMS convergence test
  verifies are …"); `sn-homogenization-balance-preservation`'s is inside its sibling
  `sn-homogenization-balance`'s; `en-kernel-derivative`'s is inside `en-definition`'s;
  `sigT-computed`'s lives on two *other pages*. ⟹ **set the search radius to the SECTION,
  and grep the label name across the whole `docs/` tree + `tests/`**, not the ±20 lines
  around the `.. math::`. The authors sentinel the *definitional* member of a cluster and
  name the *verified* members inside that sentinel — so the rationale on the neighbour is
  where the verb, the value, and the file are.
- **The strongest evidence often reads as ordinary prose, because the author wrote a
  NEGATIVE sentinel.** Three labels carry an explicit note explaining why they have NO
  `vv-status` — e.g. "`bc-single-delivery` carries **no** ``vv-status`` sentinel because
  it needs none: it is a genuine L1 equation claim with a committed gate", and
  `sn-homogenization-bilinear`'s "(Wired P6, #281 — no vv-status sentinel.) … is now a
  VERIFIED solver claim, not documented-only. `Solution.homogenize` / `Solution.condense`
  build the collapse". A grep for `vv-status` MISSES both. Grep the label itself.
- **An ORTHOGONALITY / ADJOINT-EQUALS / BALANCE-PRESERVED statement is the trap, and the
  discriminator is "is the LHS a shipped computation?"** `real-sh-discrete-orthogonality`
  looks like a pure identity; `SphericalHarmonicBasis.mass_matrix` computes its LHS
  verbatim (`einsum("n,nlm,nLM->lmLM", w, Y, Y)`) and is the SUT of both verifying tests.
  Same for `sn-homogenization-balance-preservation`, whose LHS *is*
  `Mixture.balance_residual`, plus a `raise`-backed `assert_balanced` and a symbolic
  `derive_balance_tradeoff`. ⟹ before writing `NOTHING:identity`, ask **what object the
  equation's left-hand side is** and grep for a routine returning it.

Two cheap corroborations reusable on the next batch: (a) `equation_labels=(...)` tuples
on `VerificationCase` / `ContinuousReferenceSolution` / MMS case dataclasses are the
tree's *existing* declarations — grep the label there FIRST, it is a one-hop answer for
the derivations-backed pages; (b) MMS references really are implemented in
`orpheus/derivations/continuous/mms/`, and the source function (`mms_sweep`) implements
the `q_ext` label while the ansatz method (`phi_ref`) implements the `psi_ref` one —
they are two labels, two symbols, not one.

**Ontology gotcha that cost a wrong answer if unchecked:** a dataclass FIELD resolves as
`py:attribute:`, which is NOT `py:data:`. `Mixture.SigT` is illegal as an `implements`
source even though a module-level constant (`_GAMMA_EULER` → `py:data:`) is legal. When a
brief hands you a worked example, run it through the resolver before trusting the pattern
— this one was wrong in the brief itself. Escalate to the owning CLASS instead.

---

## L-033 -- For an equation's implementer, the CODE may already declare the label, and the `verifies()` CLAIMANT names the SUT — read both before the prose. And there IS a principled NOTHING: the word is *independent*

Second batch of the same `implements`-declaration campaign (2026-08-18, 17 labels /
9 SN pages, 130 claims). L-032 covers the rationale-on-a-neighbour move; these are the
four things it does not, each of which decided a verdict my batch would otherwise have
guessed at.

- **⭐ The highest-yield pointer is a `:label:` INSIDE `orpheus/`, and a page will
  sometimes tell you it is there.** `curvilinear_numerics.rst`'s `.. note::` reads:
  *"The three labels `hebert-3-432-source`, `hebert-3-434`, `hebert-3-435` are **also
  declared in the `orpheus.sn.sweep.psi_half_angle_seed` module docstring** (the
  canonical algebra-of-record) … the Sphinx page is the **presentation layer** for the
  equations the code module owns as source-of-truth."* That one sentence resolved three
  labels with zero searching. The sibling shape is a **derivation module header that
  names labels term-by-term**: `orpheus/derivations/discrete/sn/balance.py:1-28` declares
  itself *"the **source of truth** for the balance equations … If an equation in the RST
  cannot be derived from this script, it must be added here first"* and then lists
  *"6. Cumprod recurrence coefficients (Eq. **dd-recurrence**)"*, *"5. WDD substitution →
  solved form (Eq. **dd-solve**)"*. ⟹ **two greps, both cheap, both before the page:**
  `grep -rn ":label: <name>" orpheus/` and `grep -rn "(Eq. <name>)" orpheus/derivations/`.
  Corollary: when the code declares it, the page may carry a DUPLICATE label for the same
  identity (`addition-theorem` in `slab_multigroup.rst` vs `real-sh-addition-theorem` in
  `foundations/` **and** in `spherical_harmonic_basis.py`) — report the duplication, do
  not silently declare both.

- **The `@pytest.mark.verifies("<label>")` claimants are primary evidence, and their
  BODY names the SUT.** One `grep -rn '"<label>"' tests/` per label, then read the
  claiming test. It settled two of mine outright: `normalization-dd-source-coefficient`
  (the gate builds `source = Q*dx/W` — the comment even says *"the contract source is
  Q · V · weight_norm"* — and calls `DiamondDifference.update` against
  `derive_cumprod_recurrence`'s symbolic `b`), and `addition-theorem` (the gate calls
  `quad.spherical_harmonics(1)` and sums `Y[i,l,:]*Y[j,l,:]`, pointing straight at the
  basis whose normalisation *is* the theorem). This is the dual of the usual direction:
  the ledger asks code→equation, and the test already wrote equation→code.

- **⭐ The principled NOTHING exists, and its tell is the word *INDEPENDENT*.** L-032
  warns that `NOTHING` fails flatteringly — true, and the counterweight is that a
  **hand-reference / oracle** equation must NOT be declared, because declaring it is a
  correctness error, not a style choice. `sn-p1-cylinder-hand-ref`'s page says
  *"explicit `Y_1^m` moment-sum, **independent of** the production `R Λ M` einsum"* and
  its claiming test says *"**NOT** the production frame analysis/reconstruction faces /
  `LegendreMomentScattering` einsums — so a transposed einsum in the production path is
  detectable"*. Pointing it at the production symbols would make the gate a
  self-comparison on paper (the `coding-standards` demotion). Kind: `canonical-form` —
  *a form exhibited to show structure that no production path takes*. ⟹ **before writing
  `NOTHING`, look for the word "independent"/"hand-derived"/"NOT the production" in the
  page or the gate; finding it turns a weak absence-of-hits into positive evidence.**
  (The same page uses a `reference:` rationale KIND for these — grep that too.)

- **An equation can be HALF-retired, and the page will not say so.** `phase-f-q-bar-twin-forms`
  asserts two equal-on-the-fixed-point expressions; the apply-path twin (`Q̄ = ½Σ_tφ₀`)
  was retired as an O(1)-wrong proxy (ERR-058b / #282 route (a)) and only the sweep-path
  twin survives. ⟹ **when an equation asserts `A ≡ B`, existence-check BOTH sides.**
  Same family: a page describing N *branches* of one function can be stale after a
  branch collapse — `index.rst` still documents `DiamondDifference.update` as three
  geometry branches (`alpha_in is None`, `abs_mu < 1e-15`), and the tree's own comment
  reads *"One body — no geometry dispatch"*, so three labels legitimately share
  implementers and one of them (`dd-cylindrical-degenerate`) is now realised by
  **data** (`A_downstream = 0.0`), not by the threshold the page names. Declaring is
  still right; the doc-drift is a separate finding and belongs in the deliverable.

Two mechanical notes worth reusing: (a) a `w`-GENERIC primitive
(`outgoing_face_from_average(ψ̄, ψ_in, w)`) legitimately implements several labelled
specialisations at once (`w=½` is Hébert 3-435 AND the DD slab closure AND `wdd-face`) —
say so and let the declarer rule on breadth, rather than picking one silently;
(b) tier the answer (arithmetic / generic-primitive-at-a-constant / factory-that-makes-
the-collapse-exact / symbolic-algebra-of-record) — "complete enumeration" and
"minimal honest set" are different asks and the tiering serves both from one pass.

---

## L-034 -- A "doctrine X is overturned" brief needs a TWO-SIDED inventory: the challenger ontology usually already ships in fragments

Measuring the flux torsor→cone overturn blast radius (2026-08-19, plan
orpheus-operator-machinery-report-v2 §I.7), the incumbent inventory was the
expected half (16 production files, 7×7 leaf classes, 2 displacement-type
consumers). The finds that most reshape the CAMPAIGN came from the other half —
inventorying the CHALLENGER's existing footholds, which the plan (scoping the fix
as "sweep the prose") had not asked about:

- `is_positivity_preserving` already ships on the scheme Protocol/ABC with DD
  honestly `False` AND a numerical witness test — but has **0 production readers**
  and a "gates negative-flux diagnostics" docstring claim nothing implements.
- A production cone-membership REFUSAL already exists (`realizer.py` refuses
  ZeroFluxBoundary because "a negative inflow is outside that cone").
- The COEFFICIENT family already states cone-as-predicate doctrine verbatim
  ("nonnegativity is the cone, a property — not a type invariant") WITH its own
  cone test battery (`TestCrossSectionConeAlgebra`).
- `power_iteration` already implements the challenger's ray normalization
  (unit production rate, `flux / p`).
- The incumbent's own implementation contains the challenger's concessions:
  scalar scaling kept legal (a literal affine space has none), zero fluxes
  constructed freely, a docstring saying "the swept vector IS the displacement
  from zero".

How to apply: when a brief says "ontology/doctrine A replaces B", run THREE
inventories, not one — (1) B's implementation (the expected blast radius), (2) A's
existing fragments (grep A's vocabulary: the flag, the refusal, the sibling
family's battery, the normalization — these flip campaign items from BUILD to
UNIFY/CONSUME), and (3) B's internal concessions to A (the operations B's own
implementation left legal — they measure how much of A was always true). The
sharpest deliverable rows came from (2) and (3). Sibling of L-016 (a stored flag
is a claim: check who READS it) and L-020 (capability claims expire) — here the
unread flag was the challenger's, not the incumbent's.

---

## L-035 -- A plan-vs-HEAD reconciliation OPENS with the scope-restricted `git log --since=<audit date>`; and a claim confirmed at its cited site is unconfirmed until the site's CONSUMERS are counted

(Renumbered from a duplicate "L-034" 2026-08-19 — two lessons landed under one
number from two parallel dispatches; content untouched.)

Reconciling the operator-machinery plan's space-layer audit (2026-08-08 epoch)
against HEAD eleven days later (2026-08-19), two moves structured the whole
dispatch and both generalize:

- **Bound what COULD have changed before re-verifying anything.** `git log
  --oneline --since=<audit date> -- <the audited files>` returned 2 commits,
  both provably docstring-only (one said so in its own body via AST
  comparison; the other was a 1-line pointer repoint). That single command
  flipped the deliverable's shape: not "re-verify every claim against a moved
  tree" but (a) verify the audit's *epoch* reads still hold (they must,
  nothing moved) and (b) **hunt what the audit never READ** — which is where
  all the real deltas were (3 of 6 space files unmentioned by the plan, all
  PRE-dating the audit: unread, not new; `git log --follow ... | tail -1`
  dates each). L-012 opens an in-flight carve with the diff; this is the
  same move for a COLD reconciliation, with the date bound taken from the
  plan's own "read from the tree <date>" stamp.
- **A cited site can be dead while its doctrine is live.** The plan's F3
  claim quoted `scalar_flux_space` / the F2 axis-order claim quoted
  `angular_flux_space` — both real, both stating exactly the doctrine
  claimed, and `[M]` both have ZERO production consumers (test_space.py
  only); the LIVE sites minting those spaces are a bare inline
  `FunctionSpace("sn_bulk", …)` in the mesh and per-`_SPACE_NAME` mints in
  the field layer, with a DIFFERENT axis order than the dead factory. A
  verdict of CONFIRMED at the cited site would have steered the repair
  campaign at dead code — green, done-looking, defect intact. So: before
  confirming any claim "X does Y at site S", grep S's consumers; the verdict
  language that carries both facts is "CONFIRMED (substance) / REFUTED
  (site)". Sibling of L-017 (namespace collision inflates a census) and
  L-023 (the brief's exemplar is a claim) — here the exemplar was the
  PLAN's, and it deflated silently rather than inflating loudly.

Also reusable: when the reconciliation target is governed by plan-authoring
§2, the deliverable's unit is *claim → verdict → command that produced it*,
and live probes beat re-reading — 10 lines of `.venv/bin/python` measured
five identity-aliasing claims (`==` across quadratures/decompositions/
factorisations) in one shot, turning "the docstring says they compare equal"
into `[M] True` at HEAD.

---

## L-036 -- A versioned plan's RETAINED section keeps its ORIGINAL vintage: the `--since=<plan date>` opener is exactly the check a "(v1 audit, retained)" section defeats

Reconciling the operator-machinery plan v2's §I.3 + Phase 1 (plan dated
2026-08-08) against HEAD 2026-08-19, the L-035 opener came back maximally
reassuring: `git log --since=2026-08-08 -- orpheus/transport/operators/` →
**0 commits**. Read naively, "nothing moved since the plan ⟹ the plan's audit
still holds." Every §I.3 premise about S/F/C was nonetheless false **at plan
write time**: the section was marked "(v1 audit, retained)", and its true
baseline was the v1 epoch — the tree had landed the declared Optional
`(domain, codomain)` pair, the `.kernel` exposure + `IntegralKernelOperator`
Protocol, `frame.conjugate(Λ)` as production, and the frame-eigenbasis-ownership
ruling in a 4-day burst ending `bbe8a51d` 2026-06-26 — **six weeks before the
plan's own date stamp**. The plan even named a class (`CollisionOperator`)
that had not existed for six weeks.

How to apply, on any plan/report that carries version scars ("retained",
"v1 audit", "unchanged from …", an "amendments" table):

- **Date each SECTION, not the document.** The `--since` bound for L-035's
  opener must be the section's OWN vintage. If the section does not state it,
  bound it from the plan's amendment table (a v2 correcting v1 implies v1's
  epoch) or treat the section as undated and verify its claims against the
  tree directly.
- **The tell is a cheap grep:** "retained", "as in v1", "audit kept" in the
  plan's own text. Every §I.3 refutation in this dispatch traced to one
  retained block; the v2-fresh sections (III-S, "read from the tree
  2026-08-08") were sibling-verified as accurate on their epoch reads.
- **0 commits since the plan's date is then a DIFFERENT finding**: it proves
  the staleness is *inherited*, not drift — which changes the deliverable's
  framing from "the tree moved" to "the plan was written against a memory",
  and points the fix at the plan's baseline, not at recent campaigns.
- Sibling of L-020 (the brief's timeline is a claim) and plan-authoring §7
  (reconcile before resuming): here the un-reconciled reader was the plan's
  own v2 AUTHOR, and the marker that would have caught it is the section's
  missing date stamp.

Also from the same dispatch, a verdict-shape worth reusing: when a plan
PRESCRIBES a design ("split kernel from operator; bind via `.on(V)`") and the
tree carries a landed, gated, documented COUNTER-design (the Kernel-REFINES-
LinearOperator Protocol + operator-owned frames), the honest verdict is
neither NOT-LANDED nor REFUTED — it is **"conflict to adjudicate"**: landing
the plan's item as written would now mean REVERSING a ratified decision, and
the deliverable must say so, with both artifacts' dates.

---

## L-034 -- Reconciling a PLAN against HEAD includes reconciling it against the OTHER PLANS — and `git log -S <name>` discriminates a code-symbol premise from a paper-concept one

Reconciling the operator-machinery report v2 (2026-08-19), the two findings that
outranked every claim-by-claim verdict came from checks the brief never asked for:

- **The audited plan had an older TRACKED sibling covering the same ground, with
  user rulings that refute the plan's Phase-3 mechanism — and the plan never cites
  it.** `ls .claude/plans/ | grep -i <topic>` (one call) surfaced
  `operator_strategy_realization_campaign.md` (2026-07-28, tracked, P0 landed with
  in-tree gates), whose P4 carries a user constraint ("the pencil must NOT displace
  `power_iteration`'s late binding") that inverts the report's 3.1 ("split
  KEigenvalue → CriticalityProblem + PowerIteration"). Also check `git ls-files
  --error-unmatch` on the plan ITSELF — the report was untracked, the sibling
  tracked, which is itself evidence of which document carries authority. A
  plan-reconciliation deliverable that only diffs plan-vs-tree misses the
  plan-vs-plan conflict, which is where a campaign inherits a refuted mechanism.
- **A "promote X" / "reify X" row is a claim that X exists as CODE.** One
  `git log -S "X" --oneline` answers it: `TrackMonodromy`'s only hit was the commit
  that committed the grand-report *document* — the name was an MoC sheaf CONCEPT on
  paper, never a symbol, and the plan row reads as promoting existing machinery.
  Cheaper and stronger than grepping HEAD alone (which cannot distinguish
  "retired" from "never existed").

Corollary that pays for itself: **when a campaign ships its own gate suite, RUN it
— the xfail rows are the premise oracle.** The strategy campaign's marker said
"verify P1-not-started from the Optional leaf domains"; 1.7 s of
`pytest tests/sn/architecture/ -q -rx` returned the exact 21-row todo list
(5 marker sites × parametrize), bit-matching the campaign's own 2026-08-13
checkpoint — simultaneously verifying the campaign plan's honesty and the
report's staleness in one measurement.

---

## L-037 -- The AST route has its own viewport: a node-TYPE predicate clips like a `| head`, and a same-named field/method pair splits a census into two populations

Two mechanisms from the P4-remainder ground re-measure (2026-08-29), both in
censuses that were "done by AST" and therefore read as exempt from the filter
family:

- **`ast.Assign` excludes `ast.AnnAssign`.** Walking for `FORBIDDEN_EDGES = ...`
  returned EMPTY on a module where the table is spelled
  `FORBIDDEN_EDGES: dict[...] = {...}` — an annotated assignment, a different
  node type. The zero read exactly like "no such table". Caught in one step only
  because the table was KNOWN present (a grep had located it first) — i.e. the
  positive control was accidental, not designed. Same family for walks that
  match `FunctionDef` and miss `AsyncFunctionDef`, or `Name` and miss the
  `Attribute` spelling. **When an AST walk over a known-populated file returns
  zero, suspect the node-type predicate before the file** — and when the answer
  matters, skip the walk and `exec`/import the module to read the object itself.
- **A method on class A and a FIELD on class B can share a name, and an
  attribute-census then returns one merged population.** `RSO.streaming_terms`
  (a method) and `CellVisit.streaming_terms` (a dataclass field holding that
  method's RESULT) co-exist by design — the packet's producer and its carrier.
  A bare `.streaming_terms` census mixes callers with packet consumers; the
  RECEIVER text (collected per hit) is what separates them, and `\.name(` vs
  `\.name\b` cross-checks the split. L-009 said fields are grep-problems;
  this adds: when the same spelling is a method THERE and a field HERE, every
  member census needs the receiver column or it answers a different question.

---

## L-038 -- A "consumers" census must split EXECUTABLE sites from PROSE citations in its FIRST pass — a line-grep flatters a public function by its own docstring fame

Mapping the `symmetry.py`/`manifold.py` public surface (2026-09-02), the
word-bounded line census reported `maximal_invariance_groups` at "5 orpheus/
lines / 4 files" and `singular_set` at "1 / 1" — and an AST census over
`Call`/`Name`-loads in files that IMPORT the name found **0** executable
production callers for both (every hit was a `:func:` role inside another
module's docstring or a comment). 3 of the 6 public `symmetry.py` functions
were in this state; `barycentre` likewise (4 prose lines, 0 calls). Nexus
`impact` inherits the same flattery, because a docstring `:func:` role mints a
`references` edge that reads like a caller. The direction is the dangerous
one: a well-documented function looks well-consumed, so a retirement audit
under-prices it as "live" and a boundary discussion over-weights it.

How to apply: for any "who consumes X" deliverable, run the AST pass FIRST
(`Call`/`Name` loads restricted to files whose import census binds the name;
string-annotation `"X | None"` counted separately) and report the grep-line
count only as the PROSE column. Three cheap corollaries from the same run:
(a) Nexus `dead_functions` on these two modules returned 8 private candidates —
**8 of 8 false positives** (callers were `@property` bodies, a function object
captured in a module-level catalogue dict, or a class-body install), so
confirm every candidate by AST before listing it; (b) the CLI `nexus impact`
prints its depth-1 list then dies with a traceback — use the MCP tool, the
CLI `callers` is fine; (c) `__all__` vs the AST public set is a one-line diff
worth printing — it surfaced 3 production-consumed names missing from
`__all__` and 1 alias with zero references tree-wide.

## L-039 -- Nexus `callers` on a METHOD returns empty + an `unresolved` block; only module-level functions resolve — read the unresolved COUNT as the census, and cross-check it against an AST call count

`[M]` 2026-09-05 (#448 census). `callers` on four `SNSolver.*` methods returned `nodes: []`
each, with `unresolved.count` = 6 / 2 / 4 / 9 (receiver-spelled phantoms `solver._x`,
`op.add_iso_source`). The module-level `_reflect_outflow_into_inflow` resolved to 8 callers
exactly. An AST call census over orpheus+tests gave 6 / 2 / 4 / **10** — the unresolved counts
matched to within one (a duck-typed `apply` body dropped). ⟹ for a METHOD, the `unresolved`
count IS the graph's answer; treat `nodes: []` as "not resolvable", never as "uncalled", and
always pair it with an AST `Attribute.attr` call count. Also from the same census: an
"entry-result consumer" question (`.angular_flux` off `solve_sn(`) needs a TWO-LEVEL resolver —
direct Assign-from-call catches 7 of 62 reads; helper-return and fixture-param resolution
catches the rest — and the L ≥ 1 population turned out to have ZERO ψ readers, i.e. the fix's
witness had to be named as a deliverable (§6c) rather than found.
