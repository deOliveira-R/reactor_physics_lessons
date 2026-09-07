# Test Architect — Lessons

Read at the START of every invocation. Behavioral corrections only:
"what test-design mistake did I make / catch, and what verification
discipline fixed it?" The failure-mode TAXONOMY (Modes 1–11, the three
pillars, the anti-patterns) lives in the preloaded `vv-principles`
skill — NEVER restate it here; cite it (`Mode N`, `anti-#N`, `§pillar`).
The reference inventory + cross-section mixtures live in `AGENT.md`.
Campaign play-by-play is gone (merged archaeology).

The spine behind nearly all of these: **a verification plan is not done
when "the tests pass" — it is done when (a) each gate is provably ABLE
to red (a mutation reddens it, under the canonical `-O`), (b) the
reference is structurally INDEPENDENT of the SUT, and (c) the test's
regime ACTIVATES the term the bug lives in.** Every lesson is one face
of that standard. → This spine is now a STANDING directive in `AGENT.md`
§0.5 (gate-liveness) + §0.6 (config-blindness) + §1.5 (structural
independence) — it governs every plan by default. The L1/L4 entries
below are kept for the forensic war-stories that grounded it.

---

## L1 — Homogeneous / 1-group / flat-flux tests are blind to the hardest math

→ **Promoted to `AGENT.md` §0.6 as a standing directive.** Kept here for
the forensic war-story (the cyl DD suite that documented the `Q/Σ_t`
diagnostic and STILL omitted it, so the bug hid).

The recurring root failure: the convenient test config nulls the exact
term the solver is most likely to get wrong. The blindnesses compound —
know all of them and pick a config that breaks every one that matters:

- **Flat flux** nulls every redistribution / α-recursion / weight-cancel
  term (curvilinear closure, angular redistribution). `vv §H2`.
- **1-group** makes `k = νΣ_f/Σ_a` flux-shape-independent — degenerate.
  Always ≥2G for an eigenvalue claim (`vv §1-group`, anti-#3).
- **Homogeneous** nulls redistribution AND spatial-distribution bugs.
- **Slab geometry** is the degenerate SOTP case (angular redistribution
  is a ZeroOperator) — a slab-only separability/closure test PROVES
  NOTHING about the coupled curvilinear case.
- **Isotropic-source 2-D snapshots** are blind to a dropped φ_ℓ≥1: a
  windowing that silently drops higher moments PASSES an all-isotropic
  suite. Before carving a moment-reduction path, AUDIT whether the
  existing snapshots exercise the moments being reduced; if all are
  isotropic, manufacture an anisotropic case FIRST.
- **A SYNTHETIC fixture can null a property the REAL data exercises — making
  a synthetic-only assertion a brittle trap that FALSE-REDs on real data.**
  (#274 F7.) An old leg condensed a *synthetic* 421-group χ (a literal step
  function in the fastest 10 % of groups) → WIMS-69 and asserted
  `argmax(χ_coarse)==0`. The REAL `pwr_like_mix()` χ peaks at group 61
  (~1.15 MeV) — the 2e7-eV grid ceiling carries ~60 groups of
  small-but-nonzero high-energy tail ABOVE the fission peak, so the real
  χ-peak lands in coarse group ~4, NOT 0; `argmax==0` would FALSE-RED on the
  production library. The synthetic fixture had nulled the high-energy tail
  the real data has, so the brittle assertion only "worked" on the synthetic
  case. **Fix = assert the physically-correct CUMULATIVE-MASS property
  (`χ[:n//2].sum() > 0.5`, the bulk is in the fast half), never a brittle
  argmax/exact-index on a value whose real distribution differs from the
  fixture's.** Trap signature: a `np.argmax(...)==K` / exact-index assertion
  whose K was read off a *synthetic* fixture. Before pinning an index/peak on
  synthetic data, ask "does the REAL data put the peak HERE?" — if the real
  distribution is broader/shifted, pin a cumulative/inequality property
  instead. (Same family as the config-blindness rule, one level up: the
  *fixture*, not the *config*, is unrepresentative — verify the gate against
  REAL data when it exists, which is exactly why F7 replaced the
  synthesize-and-skip leg with `pwr_like_mix()`.)
- **`make_mixture` SILENTLY NULLS two channels — `Sig2` AND `SigL`.**
  (a) `sig_2` defaults to `csr_matrix((ng,ng))` = all-zero, AND it is zero
  on EVERY `xs_library` A/B/C/D fixture; (b) **`make_mixture` has NO
  `sig_l` parameter — it hardcodes `SigL = np.zeros(ng)`
  (`xs_library.py:56`)**, so any `sig_l` you "pass" is dropped on the
  floor. So any (n,2n) term (SN `compute_group_production_rate`'s
  `+2·∫Σ₂·φ`, the CP/MoC/homog net-removal `−2·Σ₂`) and any (n,α)/SigL term
  of the balance identity are identically NULLED (Mode-10) on a
  `make_mixture` fixture; regression tests that touch them go VACUOUSLY
  green. **Trap signature**: a "balanced" fixture built as `sig_t = sig_c +
  sig_l + sig_f + rowsum` THROUGH `make_mixture` is IMBALANCED by exactly
  `sig_l` (the SigT carries it, the Mixture zeros it → `assert_balanced`
  raises). To exercise SigL/Sig2 you MUST build the `Mixture(...)`
  constructor DIRECTLY (the `test_mixture_xs_balance.py` / P5.0
  `test_mixture_condense.py::_balanced_fissile_4g` hand-built pattern), or
  `make_mixture(..., sig_2≠0)` for n2n only (reuse
  `tests/cp/test_verification.py::_make_mixture_with_n2n`). Same shape as
  the anisotropic-case rule above: the convenient builder nulls the
  channel; manufacture the activating fixture FIRST.

How to apply: minimum catch for a curvilinear solver = heterogeneous
spatial-convergence (keff differences shrink under refinement) +
fixed-source flat-flux `Q/Σ_t` (the single most powerful curvilinear
diagnostic — it was documented and STILL omitted from the cyl DD suite,
and the bug hid). Print the analytical-references checklist from
`AGENT.md` and check each item off against a test; do not rely on memory.

---

## L2 — Match the reference to the CLAIM LAYER, and terminate in a structurally-independent ground

The two recurring reference mistakes: using a reference that cannot
prove the claim's layer, and using one that shares structure with the SUT.

- **MMS does NOT prove eigenvalues** (source-driven by construction).
  An eigenvalue claim needs a closed-form (homogeneous k_inf, transfer
  matrix) or semi-analytical reference. MMS reaches only flux-shape +
  convergence-order. Declare each gate's claim layer (convergence-order
  / flux-shape / eigenvalue) and confirm the pillar can prove it
  (`vv §pillars`).
- **Convergence rate is necessary, NEVER sufficient.** O(h²) to the
  wrong limit is still O(h²). A self-referencing Richardson ladder (ref
  built from the same buggy code) converges cleanly to the wrong value.
  ALWAYS pair a rate assertion with a converged-VALUE assertion against
  a structurally-independent reference (anti-#5).
- **No mesh-independent transport eigenvalue reference exists** in this
  codebase (heterogeneous refs are diffusion-based ~0.3% gap or
  self-referencing). Use diffusion-eigenvalue as a cross-check with an
  explicit tolerance, NEVER a precision target. Tracked: Issue #8.
- **Two-anchor template for a pure-refactor matvec carve**: a committed
  snapshot ("didn't move" = bit-id inheritance) is necessary-NOT-
  sufficient (ULP-distance can't tell you the pre-carve value was right)
  — pair it with a closed-form value anchor (`Q/Σ_t`, k_inf). The Krylov
  path is a GENUINE cross-check of the SI path (different representation,
  same fixed point) only when one side is the UN-windowed/UN-carved path.

How to apply: write the (claim-layer, pillar, truth-source) triple per
gate BEFORE drafting it. Force the structural-independence cross-check
from a DIFFERENT angle (a kernel/row-sum check AND a closed-form check),
never two derivations of the same closed form (`vv §structural-indep`).

---

## L3 — Override the MMS simplification bias at write-time; declare what the ansatz ACTIVATES vs NULLS

The "simplest trig that satisfies the BCs" is the default human choice
because hand-derivation is error-prone — and it is exactly the ansatz
that tests nothing (`vv Mode 7`). Two concrete instances that bit:

- **Every prior SN MMS ansatz VANISHES at the boundary** → the
  prescribed-inflow `q.boundary≠0` path was NEVER exercised. The fix
  needed a NON-vanishing-at-face ansatz: `ψ=(A(x)+μ·B(x))/W` with
  `A=a0+a1·sin(kx)`, **a0>0 load-bearing** (makes the incoming current
  γ₋ψ≠0). A sin-only ansatz re-creates the very gap the MMS exists to
  close.
- **An isotropic-in-μ ansatz nulls the angular redistribution** — the
  sweep's hardest math. A curvilinear/Pℓ MMS MUST pair with an
  angularly-non-trivial companion (`ψ=(A(r)+B(r)μ)/W`, B≠0 so ∂ψ/∂μ≠0).
  NEVER ship slab-only or isotropic-only.

For every MMS gate, declare in the docstring which terms the ansatz
**activates** AND which it **nulls** (`vv Mode 7`). If the nulled set
includes a term covered by an active ERR-NNN, redesign the ansatz. When
the test-architect spec mandates a stress ansatz (mixed-scale k, ≥2G
het, a0>0), that is a BINDING CONTRACT — a shipped test that swaps in
the canonical sin/1G/homogeneous case is a gate downgrade, not a style
choice (caught downstream by qa).

How to apply: with SymPy the source is derived programmatically, so
there is NO derivation-cost excuse for the simple ansatz. Pick ψ_chosen
for stress value (high frequency, mixed scales, group coupling,
non-vanishing boundary) and write the activate/null declaration first.

---

## L4 — A gate that cannot red is worse than no gate: prove every gate's teeth bite

→ **Promoted to `AGENT.md` §0.5 as a standing directive** (the gate must
be provably able to red, under `-O`). Kept here for the three concrete
mechanisms + the forensic detail.

The deepest recurring test-design failure: shipping a green gate that is
structurally incapable of catching the bug it claims to catch. Three
distinct mechanisms, all requiring an explicit liveness proof:

- **Mode-8 (`-O` strips bare `assert`)**: the canonical invocation is
  `python -O`. A bare `assert` in a PRODUCTION/helper module (or any
  non-collected module) is a NO-OP under `-O` — an always-on canary that
  cannot die is a FALSE green. Use `np.testing.assert_*` / `pytest.fail`
  / explicit `raise` (function calls, fire under `-O`). Bare asserts in
  COLLECTED `tests/` modules ARE rewritten by pytest and DO fire under
  `-O` (verified) — reserve the Mode-8 flag for `orpheus/` paths and
  always-on sentinels. DEMONSTRATE: a deliberately-wrong value must still
  FAIL under `-O`.
- **The xfail-strict false-positive**: a `strict=True` xfail is
  "satisfied" by failing for ANY reason — a stale array index makes it a
  FALSE xfail (green suite, wrong reason). For features-not-yet-landed
  use `strict=False` + a `reason=` pointing at the unlocking API/label,
  so it naturally un-xfails (flips to xpass) when the feature lands.
  Prove an xfail→live FLIP is non-vacuous: confirm it is RED pre-fix for
  the DOCUMENTED reason and GREEN post-fix.
- **Mutation-validate the catcher**: a `catches(ERR-NNN)` is a COVERAGE
  CLAIM — re-introduce the EXACT documented bug and confirm THIS test
  (not merely some test in the run) reddens. Same-area/same-module is
  NOT coverage. For a self-consistency round-trip (`residual(kernel(q))≈0`,
  both arms share the kernel), instrument a call-COUNT to prove the path
  is even exercised end-to-end — a round-trip pins neither correctness
  nor convention (a matvec runs only under `inner_solver='krylov'`; SI
  sweeps never touch it).
- **Retiring a runtime guard that had NO negative test → its
  replacement's teeth are NET-NEW, not migrated.** Before crediting a
  guard→invariant mechanism-swap as behavior-identical, grep whether ANY
  test asserts the OLD guard's raise (`pytest.raises(match=<guard msg>)`).
  If none — common for internal "belt-and-suspenders" consistency
  asserts (the SN `_require`/`_refuse_starting_direction` biconditional
  had ZERO negative tests across 7 sites) — the retirement silently
  drops a REAL-but-unpinned safety net. The plan MUST WRITE the negative
  test the guard never had, pinned to the NEW consolidated check, or the
  swap loses teeth invisibly. A `pytest.raises` teeth-gate for such a
  check MUST `match=` the SPECIFIC consolidated-check message: a
  downstream crash on the same malformed input (e.g. `precompute_psi_
  state(None)` → AttributeError) satisfies a bare `pytest.raises(...)`
  and gives a FALSE-green teeth claim (the xfail-strict cousin). Pair
  with the positive CONTROL (the correctly-built composite solves) so
  the raise is attributable to the mismatch, not a broken path.

How to apply: for every gate named as evidence, ask "what mutation
reddens this, and does it fire under `-O`?" before crediting it.
Mutation-by-monkeypatch in-process (NEVER `git checkout` a file you have
uncommitted edits in). See qa L-006/L-010 (Mode-8 classification),
`vv Mode 8/11`.

---

## L5 — A characterization gate pins what is TRUE + the regression floor WITHOUT calcifying the limitation

When pinning a documented WONTFIX limitation (a first-order-at-pole
closure, an interpolation floor, an approximation plateau), the gate
must protect the floor without blocking a legitimate future fix:

- **GUARANTEE tests** carry `verifies(...)` and assert the property that
  IS correct (global L2 O(h²)).
- **CHARACTERIZATION tests** carry NO `verifies(...)` and bound the
  limitation ONE-SIDED (pole L∞ order `> 0.8` = "≥ first-order, does not
  regress", NO upper bound → a future O(h²) fix keeps the gate GREEN).
- When a fix claims to "remove a floor", do NOT assert removal — measure
  the floor's SCALING with the OTHER axis (quadrature). If it scales the
  same pre+post, the fix did NOT touch that floor; pin the floor as a
  verified scaling quantity (`err(S32) < err(S16)/2` is FALSIFIABLE — a
  fixed closure-bug floor is quadrature-independent → ratio ~1 → fails).
  `vv anti-#5/#17`.

How to apply: separate the guarantee from the characterization; bound a
limitation below-only; pin a floor by its scaling, never assert its
absence. Promote a diagnostic as a NEGATIVE-regression foundation gate
(no `verifies`) where its FAILURE is the signal to update the gate.

---

## L6 — A Mode-10 sub-floor term is closed by STRUCTURAL teeth, not a tightened value band

When a term's code path RUNS but its error is O(h²)-small (below the
convergence floor — a slope source, a boundary-trace transverse slope),
tightening the converged-flux value band does NOT pin it (the error is
sub-floor by construction). The complete closure is TWO O(1) structural
teeth + a control:

1. The production producer threads the projected moment through at
   MACHINE PRECISION (catches a regression to zeroing), with a
   structurally-independent (leggauss-only) reference.
2. A CONSUMED source-row sign flip moves the converged answer ≫ solver
   tol (catches sign-blindness), PAIRED with the no-op control (scalar /
   zeroed input → byte-identical) that pins the asymmetry.

A SHARPER case has NO value-improvement leg AT ALL: a boundary-trace
slope is sub-floor for ANY value claim everywhere (no regime makes it
the dominant forcing — a correctly-consumed slope can make the converged
value slightly WORSE). Do NOT manufacture a value-improvement gate there
— it would falsely RED a correct term.

How to apply: when a term is localized / higher-order-small, ship
producer-threading-at-machine-precision + consumed-flip-≫-tol + a no-op
control; accept the absence of a value-band gate as the CORRECT
signature. Mode-7's activated/nulled declaration is necessary but not
sufficient — mutation-verify an activated term is also CONSTRAINED.
`vv Mode 10`; mirrors method-implementer L-007 / qa L-037.

---

## L7 — Regression-snapshot tolerance is a CLAIM about what the solver promised, not a magic floor

Hand-picked floors (`rtol=1e-12`, `rtol=5e-6`) encode no claim — they
were either too tight for FP-non-associativity or papered over a real
bug. Replace with a single-source-of-truth `assert_regression`:

- **Iterative** result (k_eff/flux from power/source iteration) →
  `SAFETY × conv_tol`, where conv_tol is the solver's OWN stopping
  criterion read OFF the run config (the SoT shared by generator AND
  test — never hardcoded), and `SAFETY=10` is the `ρ/(1−ρ)`
  amplification headroom (principled, not a fudge).
- **Direct** result (single sweep, no outer iteration) →
  `assert_array_almost_equal_nulp(nulp=reduction_depth)`.
- **Bit-identity** for a pure-refactor is enforced by `-W
  error::DriftWarning` LAYERED on top — "the gate passes" ≠
  "bit-identical"; verify WHICH invocation ran.

Two traps that recur:

- **An ITERATED end-to-end snapshot CANNOT be the bit-identity gate** for
  a zero-numerical-change refactor — committed iterated DD snapshots
  already drift 1000s–100000s ULP under `-W error::DriftWarning`
  (cross-run/cross-machine FP jitter). The bit-id PROOF must descend to a
  single-step DIRECT snapshot on a fixed-seed RANDOM het ≥2G ψ with
  non-zero inflow (flat ψ nulls redistribution), captured pre-carve via
  the root-conftest `--capture-baseline` flag.
- **A non-fissile mixture has NO eigenvalue.** A `solve_sn` (eigen)
  snapshot on a moderator mixture is `k=0/abs→nan` — a silent dead test.
  To exercise a scattering/streaming operator path on a non-fissile
  mixture, reformulate as FIXED-SOURCE (`solve_sn_fixed_source` + uniform
  Q); the same path runs and is well-posed, corroborated by the
  closed-form `φ=(diagΣ_t−Σ_s0ᵀ)⁻¹Q` flat infinite-medium solution.

How to apply: the gate IS the claim — map quantity→config-key explicitly,
make the helper `-O`-safe, and demonstrate both DriftWarning modes
(informational + escalated) with a deliberate ±ULP perturbation. The
principled-equivalence ladder (bit-id ↔ nulp ↔ SAFETY×conv_tol ↔
sha256-golden) is chosen per the `vv` 3-criteria; a re-association that
names per-axis intermediates is principled-equivalent, gated at
`nulp(reduction_depth)`, anchored at k_inf + MMS — never at old-vs-new
ULP alone. Distinguish exact-cancellation from IEEE-754: a "clamp silent
on flat" / "w=½ byte-identical" premise from a τ-bearing or
`a+b`-summed reduction is exact-arithmetic-only — verify the IEEE
micro-fact at the prompt, not the docstring.

---

## L8 — Reuse the registry schema for cross-method infra; tag pairwise agreement L1, gate it max(tol)

When building cross-solver regression infra over an existing case
registry, a thin wrapper (pointer back to the registry case + per-solver
tolerances + pillar/claim-layer/truth-source tags) beats a richer
parallel schema (which loads the migration cost and becomes a second
drifting source of truth). Cross-method agreement tolerance MUST be
`max(tol_a, tol_b)` — tighter is reference contamination (`vv §6`). Tag
both truth gates and pairwise-agreement gates **L1** (the codebase
convention — each method's individual L1 backing makes the agreement
L1-strength when structural independence is genuine), NOT L4. Truth
values MUST trace to PRIMARY citations — transcribing from memory made up
two values the tests caught; verify against the literature memo first.

How to apply: reuse, don't reinvent; declare one-sided coverage honestly
(don't pretend coverage that doesn't exist); the harness knows
L0–L3+foundation only.

---

## L9 — V&V tagging idioms (the registry contract)

The audit is driven by `tests/_harness/registry.py` parsing markers —
mis-tagged tests show as orphans in every `session_briefing`:

- Module `pytestmark=[pytest.mark.verifies("label")]` for a default
  Sphinx label; override per-test for an extra equation. (No module
  `pytestmark` when it would clobber a foundation tag — tag explicitly.)
- `foundation` = a SOFTWARE invariant (data-structure / factory /
  algebraic-reduction); it MUST NOT carry `verifies(<physics-eq>)` —
  stacking them is silent level conflation (the audit credits the
  physics eq with a flat-flux foundation test). `math-origin` promotes
  carry no `verifies` (no `:label:` to point at).
- `@pytest.mark.slow` for >5s; the standard fast gate is `-m "l1 and not
  slow"`. Mark just the slow PARAMS (`pytest.param(..., marks=slow)`),
  not the whole function, when only some params are slow.

How to apply: see qa L-007 (foundation+verifies conflation) for the
removal recipe. `vv-status` rationale comments use (parens) not
[brackets] (docutils reads [xxx] as a citation → duplicate-citation
warnings under `-W`).

---

## L10 — The proactive-test-architect payoff: refute the plan's premises at design time

Dispatched proactively before an operator-algebra carve, the highest-value
output is REFUTING the plan's optimistic premises before the ink dries —
the plan repeatedly claims "bit-identical" / "clean O(h²)" / "improves on
flat" / "this arm is dead" / "N errors clear" and these are measurably
false:

- "iso-MMS bit-identical" was exact-arithmetic-only (≤1 ULP in IEEE).
- "aniso clean O(h²) at S16" was a #229-interpolation floor, untouched by
  the carve (needs S32).
- "improves-on-flat at the boundary" was unachievable (the slope is
  sub-floor; the correctly-consumed slope made the value slightly WORSE).
- **"the bare-`np.ndarray` arm is DEAD" — REFUTED by a runtime trace** (a
  signature-only carve). A production call site whose ARGUMENT is annotated
  `np.ndarray` is NOT evidence the typed arm superseded it — the annotation
  is the strongest reason to suspect the bare arm is LIVE. Trace the actual
  runtime object, do not read the type hint: `SNSolver.initial_flux_
  distribution()->np.ndarray` flowed bare into `F.apply(...)` (dispatched to
  the ndarray arm, returning `ndarray` not `ScalarSourceSink`), and `solve_sn`
  fed a bare `Q` to `add_iso/n2n_source` (in-place mutate, `None` return) —
  both load-bearing, neither dead. The retirement-audit's "stale annotation
  lies forward" cousin: a `T`-typed arm can be live precisely because the
  caller's `T` annotation is honest. The `assert_type(F.apply(arr),
  np.ndarray)` row is the static gate that REDS (→ `reportCallIssue`) if you
  wrongly retire it.
- **"N CLI-pyright errors clear" — never trust the count; assert it.** The
  N errors rarely map 1:1 onto the carve parts; some belong to an adjacent
  under-typing the carve doesn't touch (a `FullField`-arm `psi.bulk.
  integrate_angular()` gap is NOT cleared by retiring the *ndarray* arm).
  Gate with a verbatim post-carve `npx pyright <module>` and assert the
  EXACT residual.
- **"the same fold applies UNIFORMLY across the N solvers" — REFUTED by
  reading all N bodies.** A carve that routes a shared quantity (keff =
  prod/absn) "through one functional across 5 solvers" assumes the 5 sites
  compute the same thing; they rarely do. Reading every `compute_keff`:
  SN denom = `∫Σa·φ` (Σa already folds Σ₂.sum); **CP denom = net-removal
  `∫Σt·φ − ∫Σs·φ − 2·n2n`, NOT absorption** (equal value, different FP path
  AND n2n on the DENOMINATOR not numerator); diffusion denom = `∫Σa·φ +
  leakage`; homogeneous is **0-D (no `Σ_cells` — the volume fold is a
  category error, leave it)**. The "uniform `IRR(νΣf)/IRR(Σa)` fold" is
  clean only at the NUMERATOR; denominators are a different-physics family
  per solver. Gate each family separately; flag the CP net-removal rewire
  and diffusion leakage as per-solver MATH decisions (re-baseline per
  vv §bit-identity), not a uniform mechanical fold.

How to apply: measure the premise (temp-edit + revert, matched-quadrature
ladders, probe the regime the term dominates, RUNTIME-trace the consumer,
verbatim-count the pyright delta) rather than encoding the plan's hope into
a gate. A gate built on a false premise either calcifies a limitation,
falsely reds a correct term, or BREAKS a live path a "signature-only" carve
swore it wouldn't. For a type-contract deliverable the gate is a pyright-
checked `assert_type` block (extend the existing one — one home per
apply-contract); prove its teeth by mutating an overload return and
confirming `reportAssertTypeFailure` fires under CLI-pyright. State the
refutation in the plan so the implementer ships the achievable carve, not
the hoped-for one.

---

## L11 — An "axis-transpose of an existing reduction" reuses the machinery but ONE collapse rule flips — gate the DIFFERING rule, not the shared one

When a new operation is the axis-transpose of one that already shipped
(energy condensation = the energy-axis transpose of spatial homogenization;
both reuse the SAME `PetrovGalerkinFrame.project` = `G⁻¹M`), the implementer
will (correctly) copy the template — and the ONE place the two genuinely
DIFFER is exactly where the copy goes wrong. The gate's whole job is to
isolate the differing rule with a structurally-independent oracle; the
shared machinery is already verified by the original's gate.

- **The differing rule for condensation (vs homogenization): the MATRIX
  2-axis collapse.** Homogenize flux-weights BOTH axes of `Σ_s[g_from,g_to]`
  (`material_xs_field.py:403`, `project(_gather_legendre(l))`). Condense
  SUMS the sink (`Σ @ T`) then flux-AVERAGES the source (`project`). The
  vector channels, the `G⁻¹M` mechanics, the Gram pseudo-inverse, the chi
  birth-group-sum are all IDENTICAL or trivially analogous — only the matrix
  sink-vs-source asymmetry is new. So the load-bearing gate is the
  scattering 2-axis probe with three value-RED mutations: (i) swap
  sink/source, (ii) sum BOTH (drop source flux-weight), (iii) **project
  BOTH** — and (iii) IS the homogenize behavior, so it directly catches
  "implementer copied homogenize verbatim". Verified: only the
  sink-summed/source-averaged collapse preserves the hand-summed in-scatter
  rate `Σ_{g∈G}Σ_{g'∈G'} φ_g Σ_s[g,g']`; project-both BREAKS it (the rate
  gate reds on the copy-bug). Matrix collapse is ONE-ULP (the `@T` matmul
  reduction tree ≠ explicit sum) → `assert_allclose(atol=1e-12)`, never `==`
  (vv §bit-identity); vector channels ARE 0-ULP.
- **chi is the OTHER differing rule.** Homogenize production-weights χ ACROSS
  CELLS (a real average via `emission_frame`); condense is single-material,
  so χ collapses by PURE birth-group SUM `χ @ T` — NOT frame-projected (a
  projection would average → `Σχ ≈ 0.38 ≠ 1`, destroying the simplex). Gate
  the sum + `Σχ` preservation SEPARATELY from the projected channels.
- **The orientation trap is silent and ORDER-not-partition.** The canonical
  `eg` is DESCENDING; `IndicatorBasis.evaluate` buckets with
  `searchsorted(side="right")` = ASCENDING-only. Feeding descending coarse
  edges does NOT break the partition (row-sums stay 1) — it REVERSES the
  coarse-group ORDER (fast fine groups → thermal coarse column). So the
  intrinsic-property test must pin the coarse-group ORDER (fastest fine →
  coarse 0), not just partition completeness; demonstrate the reversal
  directly against the live `IndicatorBasis` (needs no SUT). Mode-6
  convention-drift with a green-partition disguise.

How to apply: for any "transpose/mirror of a shipped reduction" gate, (1)
diff the two production bodies and enumerate every rule that FLIPS; (2) make
the load-bearing gate a mutation-probe whose mutation IS the un-flipped
(copied-from-original) rule, and confirm it REDS against a hand-summed
rate/invariant oracle; (3) keep the shared-machinery checks light (the
original's gate covers them); (4) re-audit the convenient builder for a
silently-nulled channel (L1: `make_mixture` zeros SigL AND Sig2) before
trusting any "balance survives" leg. The skeleton goes GREEN-on-contact
(`@_needs` skipif + oracle math in-test); the convention traps + oracles run
UNCONDITIONALLY so they bite at design time.

---

## L12 — Folding a "fast path" INTO a composed form = principled-equiv, NOT 0-ULP; the UNCHANGED sibling stays 0-ULP; "transpose falls out free" is gated on the missing leaf(es) + a Mode-11 wrap. SPLIT variant (perf-forced) adds: no-tensor sentinel + OperatorSum heterogeneous-space guard + LD-monkeypatch re-pin

When a clean-before-extend carve re-expresses a bespoke "fast path" as an
instance of an already-composed operator — justified by a NORMALIZATION
IDENTITY (`Y₀⁰=1` makes `integrate_angular`=ℓ=0 analysis `M₀`, the iso `/W`
broadcast=ℓ=0 reconstruction `R₀`, so the iso P0 in-scatter =
`(1/W)·frame.conjugate(Λ)` restricted to ℓ=0) — three test-design facts
that recur and are easy to get wrong:

- **The fold is principled-EQUIV, never 0-ULP.** Same math (the identity
  guarantees it), but a DIFFERENT reduction tree (per-material matmul
  `einsum("fg,fc...->gc...")` vs the chained `R₀(Λ₀(M₀·ψ))` + the
  addition-theorem `(2ℓ+1)` broadcast). Gate the equivalence at
  `assert_allclose(rtol≈1e-14)` (reduction-depth ULP, `vv §bit-identity`
  crit-3), NEVER `array_equal`. Pair with a CLOSED-FORM iso anchor
  (`φ=(diagΣ_t−Σ_{s0}ᵀ)⁻¹Q` flat infinite-medium) — new-vs-legacy ULP is
  necessary-NOT-sufficient (L2/crit-2; can't tell you the pre-carve value
  was right).
- **The UNCHANGED sibling kernel MUST STAY 0-ULP.** If the fold adds a
  SEPARATE full-ℓ conjugation (skip_l0=False) for the iso path and leaves
  the ℓ≥1 `kernel` (skip_l0=True) untouched, the existing 0-ULP crosscheck
  on that kernel STAYS `array_equal` — do NOT relax it. Its RED is the
  CORRECT signal that the implementer accidentally re-routed the aniso
  path through the new fold (a scope violation). Audit disposition
  EXPLICITLY: which existing pin stays 0-ULP, which re-baselines to
  principled-equiv, which is net-new (the §1.5 re-baseline table).
- **"S† falls out free" rests on the ONE missing leaf, not the
  propagation.** `OperatorProduct.apply_transpose` propagates iff BOTH
  operands have `CAP_APPLY_TRANSPOSE`; the frame faces M/R already do, so
  the ENTIRE free-transpose claim reduces to adding `apply_transpose` (the
  per-ℓ group-flip `Σ_{s,ℓ}ᵀ`, index `[g,g']` vs forward `[g',g]`) +
  `CAP_APPLY_TRANSPOSE` to the leaf (`LegendreMomentScattering`, which was
  `{CAP_APPLY}`-only). Load-bearing gate = the LEAF transpose vs a hand
  loop (≥2G+4G asymmetric `Σ_{s,ℓ}`, the group-flip mutation reds). A
  forgotten `CAP_APPLY_TRANSPOSE` on the frozenset SILENTLY drops the
  capability through the product (no transpose, raises downstream) — gate
  the capability set directly (and FLIP the old "deferred" negative pin
  `CAP_APPLY_TRANSPOSE not in caps` → positive). The capability-set/impl
  mismatch is a distinct catcher from the value gate.
- **Euclidean transpose ≠ the metric Hilbert adjoint `.H`.** The carve
  wires `apply_transpose = (1/W)·kernel.apply_transpose` (plain matvec Sᵀ);
  the reciprocity gate is Euclidean `⟨Sφ,ψ*⟩=⟨φ,Sᵀψ*⟩` (`.sum()`, per-group,
  L27). Do NOT use `.H` (= `G_V⁺⊙apply_transpose(G_W⊙·)`, carries the SH
  Gram + `1/W` — a DIFFERENT identity that false-greens/reds). The dense
  `Sᵀ` oracle (build the matrix from FORWARD applies, transpose, apply)
  pins the VALUE structurally-independently but does NOT call
  `apply_transpose` → it is Mode-11-blind to the path; PAIR it with an
  in-process WRAP of the leaf transpose (counter>0) that pins the PATH.
- **A channel OUTSIDE the fold needs its own transpose gate.** When (n,2n)
  stays a parallel ℓ=0 term (separate `Σ_2n`, factor 2 — NOT folded into
  the frame), the highest-value mutation is "omit n2n from `S.apply_transpose`
  entirely" (transpose only the frame conjugation). Gate it with a `Sig2≠0`
  fixture (lessons L1: `make_mixture` AND every A/B/C/D library mixture zero
  `Sig2` → build it on the Mixture directly; #269 non-vacuity) + reciprocity
  that includes the n2n term on both sides.

- **The SPLIT variant (#276 A2-Option2): when the full-frame fold REGRESSES
  PERF (materializes the ℓ=0 moment tensor), the production keeps the iso
  SCALAR fast-path and SPLITS `S = (1/W)·[S_iso + S_aniso]` as an
  OperatorSum of two composed operators (S_iso = `R₀∘K_iso∘M₀` scalar faces,
  NO tensor; S_aniso = the UNCHANGED `kernel`). The transpose still falls out
  free — but now via `(A+B)ᵀ=Aᵀ+Bᵀ` (`OperatorSum.apply_transpose` propagates
  iff BOTH summands carry `CAP_APPLY_TRANSPOSE`), so the "free transpose"
  reduces to S_iso's leaves (M₀ᵀ/K_isoᵀ/R₀ᵀ), not just Λᵀ. THREE Option-2-only
  gates the full-frame fold did NOT need: (1) **the no-moment-tensor SENTINEL**
  — an in-process wrap on the tensor verbs asserting **0 calls** at L=0 is the
  LOAD-BEARING PERF invariant (it IS the reason the full-frame reverted); route
  P0 through the frame → counter>0 → RED. (2) **the OperatorSum heterogeneous-
  space guard** — `OperatorSum.__init__` raises iff `a.domain!=b.domain` (by
  `FunctionSpace (name,shape)`), SKIPPED when a side is `None`. S_aniso's outer
  space is the per-ordinate `L2[S^2]/(N,)` (conjugate wraps M/R); S_iso's outer
  must EQUAL it (reuse `frame.measure.space`, Pattern-2) OR be `None`. A
  freshly-MINTED per-ordinate space name (same `(N,)` shape, diff name) RAISES;
  a `None`-skip makes the sum's shape-guard VACUOUS (then the LD-multi-moment
  equivalence gate is the SOLE shape catch). Gate it with a positive-construct
  + negative-mismatch-RAISES pair (anti-#11). (3) **the LD-monkeypatch RE-PIN**
  — a pre-existing slope-source mutation gate that monkeypatches the OLD shared
  combine (`_assemble_per_ordinate_source`) goes VACUOUS when the split retires/
  reshapes it (Mode-11); re-pin to the NEW production line (`K_iso.apply` /
  `apply_p0_in_scatter`, which carries the `...` spatial-moment spectator) +
  sentinel-wrap that the gate EXECUTES it. The LD config is mandatory for the
  oracle gate (the full-frame regression was LD-only). The oracle
  (`full_scatter_kernel` = the moment-tensor form, KEPT per fuller-view-oracle)
  pins forward AND transpose VALUE structurally-independently.

How to apply: for an iso/fast-path→composed fold, the §0.5 named risk is
THE equivalence (principled-equiv 1e-14 + closed-form anchor + UNCHANGED-
sibling-stays-0-ULP). For the "transpose falls out free" extend, gate the
ONE leaf transpose (hand loop, group-flip mutation) + capability-set FLIP +
Euclidean reciprocity (asymmetric SigS, per-group, +discriminator) + the
Mode-11 leaf-WRAP (dense oracle pins value, wrap pins path) + the outside-
the-fold channel's own transpose. For the SPLIT variant add the no-tensor
sentinel + the OperatorSum space-guard construct/mismatch pair + the
LD-monkeypatch re-pin (above). Re-audit the Mode-7 trap: a P0-only
fixture NEVER builds the ℓ≥1 blocks and is BLIND to the unified ℓ≥0 fold (or
the iso/aniso SPLIT) — run every equivalence/transpose gate on an
ANISOTROPIC+het+Sig2≠0 fixture, reserve P0-only for the perf P0-path gate.
`vv §bit-identity / Mode-7 / Mode-11`; mirrors qa/method-implementer
fission-adjoint (A1) template.

- **The CROSS-MODEL energy-operator EXTRACTION variant (#276 #119, the 3rd
  design — SUPERSEDES the SPLIT):** the iso/aniso SPLIT is dropped; the iso
  ℓ=0 projection becomes a standalone `K_iso = IsotropicScattering +
  IsotropicN2N` OperatorSum on the SCALAR flux, routing through the EXISTING
  `mat_xs` scalar verbs (`apply_p0_in_scatter`/`apply_n2n`,
  material_xs_field.py:613/652) + NEW scalar transpose twins
  (`apply_p0_in_scatter_transpose`/`apply_n2n_transpose`, mirror the in-tree
  moment twins at :751/:840), extracted to `transport/operators/` (sn-free),
  consumed at THREE sites incl. a CROSS-MODEL homogeneous LHS-fold. This is
  SIMPLER to gate than the SPLIT: same `mat_xs` verbs + same `+=` order ⇒ the
  forward `K_iso≡fast-path` AND the whole-`_assemble_per_ordinate_source`
  bit-id are **0-ULP `array_equal`** (NO reduction-tree change — the SPLIT's
  frame faces are gone, so it is NOT principled-equiv). The SPLIT's two
  headline risks DISSOLVE: OperatorSum-space is LOW (both summands share
  `mat_xs`+scalar space) and the no-tensor sentinel is moot (scalar path never
  touches the tensor). The risk MOVES to ONE new place: **the cross-model
  `as_dense` consumer holds a bare `Mixture`, NOT a `MaterialXSField`** —
  `homogeneous/solver.py:93` builds `A=diags(SigT)−SigS0_T−2·Sig2_T` from
  `mix.SigS[0].T` with NO mesh / NO `cells_by_material`, so `K_iso.as_dense()`
  REQUIRES the operator be constructible from bare ENERGY data (a `from_mixture`
  ctor → `{0: mix.SigS[0].toarray()}`); else homogeneous must import an SN mesh
  and the "sn-free/model-independent" claim is FALSE (DEFER P4 to an
  AskUserQuestion if not). VERIFY-IN-TREE payoff (L10): the brief's
  "diffusion/homogeneous build A=diags−K_iso" is FALSE for diffusion
  (`diffusion/solver.py:240` is matrix-FREE + 2G-hardcoded
  `(sig_a+sig_s)*fi−sig_s[::-1]*fi[::-1]`, builds NO dense A) ⇒ `as_dense` has
  ONE real consumer, shape = per-material `dict[int,(ng,ng)]` (mirror
  `n2n_matrix`), NOT a block-diagonal `(N·ng)²` (YAGNI). Add a VACUUM bit-id
  correctness gate (Sig2=0 ⇒ `as_dense()≡` legacy `SigS0_T` exactly — differ ⇒
  STOP it's a bug; `snapshot_migration_when_production_goes_bare.md`) + the
  homogeneous keff invariance vs the closed-form analytical k_inf (2eg+4eg,
  `<1e-12`, NOT old-vs-new). Production S† rides the `full_scatter_kernel`
  ORACLE (frame form, not K_iso) ⇒ the S†-oracle-equiv leg is a near-TAUTOLOGY
  (prod IS that expression); the LOAD-BEARING S† pin is Euclidean reciprocity
  `⟨S.apply,χ⟩=⟨ψ,S.apply_transpose⟩` (fwd fast-path vs adjoint frame-form =
  two structurally-different reps). The LD-monkeypatch on
  `_assemble_per_ordinate_source` SURVIVES (it wraps the OUTER combiner the
  carve keeps) — but ADD a Mode-11 wrap on `IsotropicScattering.apply`
  (counter>0) AND a direct-on-`IsotropicScattering` mutation (the old
  monkeypatch is now one layer ABOVE the changed reader, so a stamp bug it
  faithfully passes through could be masked). Plan
  `.claude/plans/archive/a2_kiso_verification.md`; SUPERSEDES `a2_option2_verification.md`.

- **PHASE C continuation (#276 — homogeneous USES `C=MultiplicationOperator`
  + `F=FissionOperator` as operators via APPLY-TO-BASIS, dropping
  `np.diag`/`np.outer`):** A=`C−K_iso` (OperatorSum), F=fission dyad; the dense
  `(ng,ng)` realized by applying each op to the ng single-cell `(ng,*sp)` columns
  `e_i`. The GAP: the 206-caller FullField-only `MultiplicationOperator.apply`
  gains a `singledispatchmethod` bare-`ndarray` arm (`coeff.values*x` = the
  per-ordinate engine action; FullField arm UNCHANGED = the conversion's
  bit-id-inheritance guard ⇒ existing `test_multiplication_operator.py` +
  SN forward/adjoint suite stay 0-ULP green). **The LOAD-BEARING gate is
  CROSS-ARM CONSISTENCY, NOT bare-vs-its-own-`coeff*x` (tautological):**
  `apply(φ_bare (ng,*sp))` ≡ `apply(φ BROADCAST to (N,ng,*sp) FullField)
  .bulk.values[n]` ∀ ordinate n — the FullField-engine path
  (`DiagonalOperator(coeff,broadcast_axes=(0,))` = `coeff[None]*x_full`) is the
  structurally-independent reference, 0-ULP `assert_array_equal` (same two floats
  multiplied). Build the broadcast input: `np.broadcast_to(φ[None],(N,ng,*sp))
  .copy()` into `TimedFullField.zeros(...).bulk.values`. ADD an EXPLICIT
  `_require(bare.shape==(ng,*sp))` — `assert_array_equal` BROADCASTS so it ALONE
  misses the extra-leading-axis mutation (`coeff[None]*x`→`(1,ng,*sp)`). The
  single-cell meshless homogeneous consumer is AXIS-BLIND (spatial=`(1,)` nulls a
  group↔spatial transpose) ⇒ the intrinsic gate MUST use the existing
  `_cartesian_2d_mesh(nx=5,ny=3,ng=2)` + asymmetric σ (config-blindness L1);
  homogeneous k_inf catches only the COMPOSITION (sign-flip `C+K_iso` /
  C-omission `−K_iso` / K_iso-omission `=C` all move k O(1) on the all-channel
  `homo_2eg_n2n` — the existing `test_kinf_exact` covers it, NO new homogeneous
  CORRECTNESS test needed). NEW homogeneous gate worth adding = a SHARP A-level
  procedural pin: expose `_assemble_loss_matrix`→ `assert_allclose(A_applytobasis,
  diag(σt)−(σs0+2σ2)ᵀ, atol=1e-12)` (localizes a sign/term bug faster than the
  end-to-end eig; PROCEDURAL — shares `mat_xs` data ⇒ PAIR with the structural
  `case.k_inf` anchor, L2 two-anchor). Mutations: (i) axis-transpose + (ii)
  group-drop → intrinsic only (homog axis-blind); (iii) extra-leading-axis →
  intrinsic shape-assert (downstream crash/mis-assemble is unreliable).
  `apply_transpose` delegates to `apply` (self-adjoint real coeff ⇒ free bare
  arm; gate pins the delegation survives); `solve` STAYS FullField-only (no
  consumer). Dispatch gate: an unregistered type (ScalarFlux / `object`) → base
  `TypeError` (mirror F). Mode-11 wrap on the ndarray arm (counter>0 during
  `solve_homogeneous_infinite`) confirms the k_inf gate EXECUTES the new reader.
  **Scope = ndarray-ONLY** (F has +ScalarFlux ∵ a live SN scalar-fission
  consumer; C has NO scalar-flux collision consumer yet — Pattern-6 defer;
  trigger = the first diffusion/CP/MoC refounding on C). K_iso's input surface
  is duck-typed `_values_of` (model-portable primary), C's is FullField-primary
  + bare-secondary — the three operators' surfaces differ ∵ their consumer sets
  differ (honest, not inconsistent). `vv §bit-identity / Mode-11`; mirrors the
  K_iso A2 plan above.

---

## L13 — Verifying a "capability-string → typed-operator" carve (`.solve`/`CAP_SOLVE` → `.inverse()`; `CAP_APPLY_TRANSPOSE` → `.H`): the equivalence keystone is bit-id INHERITANCE; the WHOLE frozenset (BOTH axes) retires SAFELY because the "iff both" law lives in the METHOD BODY — licensed by a FAITHFULNESS keystone (derived-predicate ≡ old-frozenset ∀ operator, before deletion); a half-modernized axis is itself the twin-path HAZARD

When a carve replaces a stringly-typed capability + gated method
(`CAP_SOLVE` + `.solve`) with a typed operator (`.inverse()` returns a
`SweepInverseOperator`/`KrylovInverseOperator`), three reusable
test-design facts (#226 B4 Phases 2–5, plan
`issue_226_inverse_operator_verification.md`):

- **The keystone equivalence gate is bit-identity INHERITANCE, not a
  fresh value claim.** `A.inverse().apply(b) == A.solve(b)`
  `assert_array_equal` is bit-exact BY CONSTRUCTION (the inverse
  operator WRAPS the legacy method) — it proves the wrapping changed
  NOTHING, necessary-NOT-sufficient (L2). It RIDES the EXISTING
  closed-form anchors (`test_removal_form_kinf_independent_reference_2g`,
  keff k_inf) for "was right"; do NOT mint a new reference. The
  curvilinear seed (`initial_guess`) MUST be threaded on a
  sphere/cyl seeded variant (slab is BLIND — §0.6); PAIR the value gate
  with a Mode-11 in-process wrap capturing `initial_guess is seed` (the
  converged solve may be seed-INSENSITIVE, nulling the value gate — only
  the wrap pins the threading unconditionally, L4/L12).
  - **The DELIVER-ALL nine advertisers (`is_invertible` → a working
    `.inverse()`): do NOT trust a brief's assumed inverse REALIZATION —
    read the code.** The elegant choice is DELEGATION-to-`solve` (a
    generic `InverseOperator` whose `apply` IS `leaf.solve`, the division
    `b/c`), NOT a reciprocal-coefficient twin — because (a) `(1/c)·b ≠
    b/c` drifts and (b) `1/Σ` is a units-dishonest mean-free-path
    (coding-elegance Pattern 3). This SUPERSEDES the "reciprocal Diagonal
    ~1-ULP" premise and splits into TWO DISTINCT tolerance claims:
    (i) `inverse().apply == solve` is **BIT-IDENTICAL** for ALL nine
    (delegation + bit-id-preserving structural composites — `Perm→take(π⁻¹)`,
    `(AB)⁻¹=B⁻¹A⁻¹`, `(αL)⁻¹=(1/α)L⁻¹`, `Id→x`); (ii) the round-trip
    `inv.apply(A.apply(x))≈x` is still per-type **nulp** (the two
    directions use ×-then-÷, DIFFERENT arithmetic — nulp even when
    equivalence-to-solve is bit-id). Tolerance = the round-trip's FP
    reduction depth (count ops: `array_equal` for gather/passthrough
    Perm/Id; `nulp(2)` for Diagonal ×÷; sweep `rtol=1e-10`, **sphere
    excluded** #282) — NEVER a blanket loose tol. And `(A⁻¹)⁻¹ = A`
    holds by **OBJECT IDENTITY** for the `InverseOperator`-backed leaves
    (`inverse()` returns `self.inner`) — assert `is A`, NOT the brief's
    "action-equality at ~ULP" (that degrades to action-nulp ONLY for the
    recomputed-scalar `ScaledOperator`, `1/(1/α)`; NOT testable on
    `InvertibleOperator` — `SweepOperator.inverse` is #280-deferred).
    The universal round-trip (I1) is the STRONGEST Step-1 gate: it needs
    NO legacy `.solve` (survives the frozenset/solve retirement) and pins
    the inverse against its OWN forward + a closed-form value anchor.
    Every mutation names its ACTIVATING config (§0.6 applies to
    MUTATIONS): non-commuting `Diag@Perm` for the `(AB)⁻¹` order-swap,
    NON-involution 3-cycle for the perm-inverse, α≠1 for the scale,
    non-unit `c` for the delegation. Negative controls RAISE eagerly at
    `.inverse()` (not `assert` — `-O`); α=0 is a ctor `ValueError`, a
    DIFFERENT edge from the value-dependent singular (`min|f|=0`).

- **A LAW-gating capability CAN be retired — the law is in the METHOD
  BODY, not the frozenset — and leaving it as a string while a sibling
  axis modernizes is ITSELF the hazard (twin-path, Cardinal Rule 2).**
  My first pass recommended KEEP `CAP_APPLY_TRANSPOSE` (retire `CAP_SOLVE`
  only) because it gates `OperatorSum/Product.apply_transpose` "propagates
  iff BOTH" (the #276 "S† falls out free" law). The user OVERRULED it: a
  modernized solve axis (`.inverse()`/`SupportsInverse`) beside a
  stringly-typed transpose axis (`CAP_APPLY_TRANSPOSE`) is TWO mechanisms
  for one structural question — forbidden. The resolution: retire the
  WHOLE `capabilities` frozenset (`CAP_APPLY`+`CAP_SOLVE`+
  `CAP_APPLY_TRANSPOSE`) via ONE mechanism per axis (`.inverse()`/`.H`
  operator-returning methods + `SupportsInverse`/`SupportsAdjoint` STATIC
  Protocols + recursive derived `@property is_invertible`/`is_adjointable`
  + eager `NotInvertible`/`MissingAdjoint`). SAFE because the "iff both"
  LAW already lives in the method body (`OperatorSum.apply_transpose =
  a.apply_transpose(x)+b.apply_transpose(x)`; `OperatorProduct.solve =
  b.solve(a.solve(·))`) — the frozenset is only a redundant
  ADVERTISEMENT. The retirement does NOT touch S† (the reciprocity LAW
  `a.H+b.H` is untouched); only the advertisement moves.
  **The FAITHFULNESS keystone is what licenses the deletion:** a SCAFFOLD
  gate asserting `is_invertible == (CAP_SOLVE in caps)` AND `is_adjointable
  == (CAP_APPLY_TRANSPOSE in caps)` for EVERY operator (leaves + every
  composite, the value-dep zero-coeff + half-adjointable asymmetry
  fixtures MANDATORY — else blind to the axis separation), proved GREEN
  during coexistence, THEN deleted WITH the frozenset. Its PERMANENT
  successors are the recursive composition pins (`(a+b).is_adjointable ==
  a.is_adjointable and b.is_adjointable`, etc. — they never reference the
  frozenset) + the migrated closure-law tests + the S† method canaries.
  Mutations M-ADJ-PROP (break the recursive `and`) and M-ADJ-FORGE (force
  `is_adjointable=True` on a half-adjointable sum) both RED the keystone;
  M-ADJ-FORGE downstream is a RAISE not a wrong-value (a half-adjointable
  summand has no transpose to forge — the forge bypasses the eager `.H`
  raise → AttributeError).
- **Runtime-query form = derived `@property`, NEVER `isinstance(Supports*)`.**
  `runtime_checkable` Protocol `isinstance` is CLASS-uniform (checks method
  PRESENCE) → blind on composites (every `OperatorSum` HAS
  `apply_transpose` on the class, so a half-adjointable sum passes
  `isinstance(SupportsAdjoint)`) AND blind to the value-dep singular edge
  (`isinstance(M_zero, SupportsInverse)` True even at `min|f|==0`). Only
  the recursive/data-reading `@property` is instance-accurate. The
  Protocols earn their place as the STATIC contract ONLY (pyright +
  annotation target). The genuine pre-query consumer forcing the property
  is KrylovAccel preconditioner-SELECTION (solve axis); the ADJOINT axis
  has NO external pre-query consumer (read only by composers + the eager
  `.H` raise) → smaller migration than solve.
- **The LITERAL-STRING precondition trap.** Two landed S† canaries
  (`test_g_adjoint_reciprocity:210`, `test_removal_form_matvec_sweep:308`)
  guard with `if "apply_transpose" not in A.capabilities` — a BARE string,
  so a `CAP_APPLY_TRANSPOSE`-CONSTANT grep MISSES them, and they
  `AttributeError` the instant the frozenset deletes. "S† stays green" is
  NOT automatic — the deletion REQUIRES rewiring these preconditions to
  `not A.is_adjointable`. Before any capability retirement, grep BOTH the
  constant AND the literal string `"solve"`/`"apply_transpose"` membership
  reads; the consumer surface is WIDER than the carve plan's sketch
  (CAP_SOLVE wove through DiagonalOperator's value-dep spectrum law, the
  sn/solver resolvents, every composer, KrylovAccel's preconditioner
  SELECTION — not just SourceIteration; the adjoint cap is read by every
  composer + the boundary-op caps-derivation).

- **The adjoint-of-inverse `(A.H)⁻¹ = (A⁻¹).H` is verified by the
  RECIPROCITY identity, not a round-trip tautology.** For `x =
  A.H.inverse().apply(b)`, assert `⟨A.apply(ψ), x⟩_G = ⟨ψ, b⟩_G` for
  random ψ — the FORWARD matvec + G-metric is structurally independent
  of how `x` was computed (transpose-solve via `loss_action_transpose`),
  and reuses the EXACT metric `test_g_adjoint_reciprocity` validates.
  The round-trip `A.H(A.H.inverse(b)) ≈ b` is necessary-not-sufficient
  (both arms share the metric) — keep it but PAIR with reciprocity. The
  metric-vs-Euclidean discriminator (L12: `.H` ≠ Euclidean `Aᵀ`)
  ACTIVATES on sphere/cyl only (non-trivial trace metric) — the
  drop-the-G-metric mutation reds curvilinear, stays green on slab.
  `A.H.inverse()` rides the LANDED transpose matvec ⇒ this gate
  pre-builds #280's substrate; if the carve defers the transpose-solve,
  the gate is `xfail(strict=False, reason="#280 …")` that flips to
  xpass when #280 lands (NEVER strict=True — a stale failure satisfies
  it for the wrong reason, L4).

How to apply: for a capability→operator carve, (1) keystone =
array_equal inheritance + existing closed-form anchor + seed Mode-11
wrap; (2) classify EVERY retired capability string structural-vs-law-
gating and mutation-prove the law-gating one's retirement ≠ no-op;
(3) the value-dependent singular edge (DiagonalOperator zero-coeff,
`CAP_SOLVE iff min|f|>0`) survives as a runtime RAISE inside `.inverse()`
(positive+negative pair, anti-#11; RAISE not bare-assert — `-O`), NOT a
frozenset; the STATIC `ScaledOperator(0.0)` edge is already a ctor
`ValueError`. The `# type: ignore` deletions are pinned by EXTENDING the
`assert_type` C6 block (`test_operators_apply_typed.py`) + CLI-pyright
`reportAssertTypeFailure` teeth (L10). `vv §bit-identity / Mode-8/11 /
anti-#2,#11,#19`; mirrors L12 (the scattering-fold sibling) + L2/L4/L10.

---

## L14 — Verifying the FIRST *iterative* inverse of a family (`GreenOperator` = `OperatorSum.inverse()` wrapping the SI driver) + the `is_invertible=a.is_invertible` contract that FLIPS frozen pins + a wrap-delegate MIXIN extraction

When a carve adds an ITERATIVE inverse-as-operator (the inverse WRAPS a
convergent driver, not a legacy `.solve`) and changes a composite's
invertibility contract to route to it, six reusable test-design facts
(#226 taxonomy §12 step 4, spec `issue_226_inverse_operator_verification.md`
PART III):

- **The first iterative inverse has NO legacy `.solve` to inherit — so its
  keystone is STRUCTURAL-INDEPENDENCE, not bit-id inheritance (contrast
  L13).** Its correctness rides a closed-form dense-LU anchor
  (`np.linalg.solve` of the materialized sum) + the distinguishing
  name-earning invariant, NEVER old-vs-new ULP. And once the composite
  gains `solve(b) := self.inverse().apply(b)` (the conditional-CAP_SOLVE
  backing method), the `inverse().apply ≡ solve` gate is a **TAUTOLOGY**
  (solve IS defined as inverse().apply) → it proves NOTHING; do NOT add it
  as coverage (it was bit-id evidence in L13 only because the inverse
  wrapped a PRE-EXISTING solve). Round-trip tol is DRIVER-TOL
  (`SAFETY×tol`, L7 iterative), NOT nulp — the one row that breaks the
  §12.0 "count arithmetic ops → nulp" rule.
- **A raise-on-non-converge gate (`ConvergenceFailure`) MUST test the TRUE
  residual `‖A·ψ−q‖/‖q‖`, NOT the driver's iterate-INCREMENT.** Signature 9
  (ρ-blind stopping): the increment under-reports the true error by
  `1/(1−ρ)`, so an increment-check passes a non-converged iterate SILENTLY
  as `c→1` — defeating the exception's Cardinal-Rule-1 purpose. Falsifier =
  a `c=0.99` near-critical split where increment<tol but true residual≫tol.
  This is a FIRM design objection to raise if the brief says "final
  residual" ambiguously (the driver returns `(psi, history)` and does NOT
  raise on max_iter — so the wrapping op owns the check). The divergent
  edge PAIRS with a convergent CONTROL (a bare `pytest.raises` with no
  no-raise control is Mode-10-blind); the raise is a `raise` not a bare
  assert (`-O`).
- **`is_invertible = self.a.is_invertible` (left-spine leading term) FLIPS
  frozen `is_invertible is False` pins False→True BY DESIGN** — a plain sum
  with an invertible LEADING term now HAS a general inverse (the Green), so
  "no general (A+B)⁻¹" is retired. Retirement-audit: `grep -rn
  "is_invertible is False" tests/` WITHOUT an adjoint filter (a `grep -iv
  adjoint` HIDES the `is_adjointable…is_invertible is False` compound-line
  hit) and classify EACH: leaf / product (`a and b`) / scaled STAY False;
  plain-sum-with-invertible-leading FLIPS. The predicate is
  "leading-term-preconditionable AT THIS SPELLING", NOT "mathematically
  invertible" (`(−S)+A` is math-invertible yet reports False) — acceptable
  ONLY given the #261 canonical-ordering precedent + zero current consumers
  + a LOUD refusal; doc it + pin the refusal message.
- **The faithfulness keystone (`is_invertible ≡ CAP_SOLVE∈caps`, L13) stays
  green ONLY if the conditional `CAP_SOLVE` add is LOCKSTEP** — both read
  `self.a.is_invertible` (single source). The `a`-recursion needs BOTH
  asymmetric fixtures `inv+applyonly` AND `applyonly+inv`: a both-invertible
  sum is BLIND to a/b/and/or (§0.6); only the `(True,False)` pattern
  uniquely IDs the `a`-rule (b→(F,T), and→(F,F), or→(T,T)). Mutation
  M-SUM-CAPDRIFT (add CAP_SOLVE under `_has(a) and _has(b)` ≠ the property)
  reds faithfulness.
- **The order-dependent-strategy trap has THREE edges, each needing teeth**
  (#261): `L+C` → `SweepOperator` (MRO shadows the generic sum inverse —
  assert `type(...) is SweepOperator`, NOT Green); `C+L` → a Green whose
  collision-preconditioned Richardson DIVERGES → loud `ConvergenceFailure`
  at apply (not silent wrong answer); `(−S)+A` (leading non-invertible) →
  REFUSES at construction with the canonical-ordering message. The Green
  auto-split must FLATTEN the left spine of EXACT-`OperatorSum` nodes ONLY
  (subclasses like `InvertibleOperator` are leaf-terms) — mutation
  M-GRN-FLATTEN (flatten THROUGH the subclass) makes `A_loss.inverse()`
  refuse (leading `L` non-invertible) where it should build a Green.
  Green-vs-hand-`SourceIteration` equivalence (auto-split ≡ canonical SI) is
  bit-id-inheritance (necessary-not-sufficient) → PAIR with the dense-LU
  anchor.
- **A NEW iterative inverse's seed-threading needs its OWN Mode-11 pin — a
  landed per-iterate spy ONE LEVEL BELOW is BLIND to it.** `Green.apply(q,
  initial_guess=x0)` threads x0 as the driver's START; the landed spy
  (wraps the inner `InvertibleOperator.solve`, pins driver→inner per-iterate
  threading) stays GREEN if Green drops x0 (each iterate still seeds the
  next), AND the drop is value-invisible (warm-start changes RATE not the
  converged value) → wrap the INTERNAL `SourceIteration.solve`, assert the
  first call receives x0 (values-equal, not `is` — a defensive copy
  false-REDs `is`). Name-earning invariant triage (taxonomy §13 bar):
  G-Neumann (partial sums `Σ_k(P⁻¹N)^k P⁻¹q`, ratio≈c) + forward dense-LU
  anchor = LOAD-BEARING; G-reciprocity = INCLUDE-lighter (the ONLY thing
  pinning the transposed-operand split without a 2nd oracle — needs a
  NON-symmetric A; Euclidean transpose, NOT `.H` = #280-adjacent); G-kernel
  (`G·δ_j`=col j of A⁻¹) = FOLD into the anchor (δ_j inputs). G-Neumann-L1
  Mode-9 config = het ≥2G VACUUM slab, NOT isotropic reflective box (flat
  flux nulls the multiple-scattering the series expands).
- **A wrap-delegate MIXIN extraction (shared back-half of ≥2 inverse twins)
  is behaviour-preserving → the no-op proof is "existing suites green
  UNCHANGED"; the NEW gate is the pyright ABSTRACT-signature enforcement**
  (`apply(x,/,*,initial_guess=None)` on the mixin → `reportIncompatible
  MethodOverride` if a sibling drops the kwarg — the #285 structural-vs-
  convention decision resolves to STRUCTURAL) + a mixin mutation that reds
  ALL siblings (M-MIXIN-CODOM/INV — single-source proof). `SupportsSeeded
  Apply` conformance is a STATIC (annotation) gate, not runtime (not
  runtime_checkable).

How to apply: for an iterative-inverse-as-operator carve, (1) keystone =
dense-LU anchor + G-Neumann, NOT bit-id (no legacy solve) — and flag the
`inverse().apply≡solve` tautology; (2) demand the raise checks the TRUE
residual (Signature 9 falsifier); (3) grep-audit the `is_invertible is
False` flip WITHOUT an adjoint filter, classify leaf/composite, migrate the
premise ("no general (A+B)⁻¹" retires); (4) faithfulness lockstep +
both-asymmetric-fixtures; (5) three ordering edges (MRO-shadow / divergent-
loud / leading-non-invertible-refuse) + flatten-exactness; (6) the new op's
OWN seed Mode-11 pin; (7) mixin no-op = existing-suites-unchanged + pyright
abstract-sig teeth. `vv §bit-identity / Mode-8/9/11 / anti-#2,#11 /
Signature-9`; the iterative sibling of L13 (which handled the DIRECT/exact
inverses); mirrors L2/L4/L7/L10/L12.

---

## L15 — Verifying the RETIREMENT of a coexisting mechanism (the frozenset→typed-predicate TERMINAL step): keystone-v2 is a structural/value CONTRACT (not a comparison), migration is a mechanical-rule + completeness-regrep (not a table), land-order is ATOMIC, and the pruned-body re-route rides inheritance+independent-anchor+sentinel with a direct/iterative TOLERANCE SPLIT

When a carve RETIRES a coexisting advertisement (the `capabilities`
frozenset, BOTH axes) that a typed successor already replaced (the ADDITIVE
steps = L13/L14 — `is_invertible`/`is_adjointable` + `.inverse()`/`.H` +
`SupportsInverse`/`SupportsAdjoint`), four terminal-step disciplines (#226
§12 step 6, spec PART V §36–§44):

- **The coexistence-era faithfulness scaffold (`is_X ≡ CAP_X∈caps`) DELETES
  with the frozenset — its PERMANENT successor is a structural/value-split
  CONTRACT, not a comparison (keystone v2).** Per operator: `is_invertible
  True ⟹ .inverse() returns`; False ⟹ EITHER `_STRUCTURAL_ABSENT`
  (`not hasattr` — Zero/masks/RankOne/streaming-L/energy S·F leaves/B) OR
  `_VALUE_RAISE` (declares `.inverse()`, raises the eager `NotInvertible` —
  zero-coeff Diag/Mult, non-invertible-head sum/product/scaled). PIN WHICH
  PER TYPE — the split IS the contract. Mirror the adjoint axis
  (`is_adjointable ⟺ .H returns vs raises MissingAdjoint EAGERLY`; uniform,
  no structural-absent — `.H` is on the base). Add bridge-consistency
  (`invertible(op)==op.is_invertible` pins the one-line TypeGuard body vs
  drift — M-BRIDGE `return True` reds it). The recursive-composition pins
  survive VERBATIM (never read the frozenset); only the scaffold CALLS strip
  out. Rewrite the ONE shared helper (Cardinal Rule 2), `-O`-safe
  explicit-`raise` (un-collected module). The two asymmetry fixtures (Zero =
  adjointable-not-invertible; TRUE-zero-coeff Mult; half-adjointable
  `full+apply_only`) stay MANDATORY (blind to the axis SEPARATION otherwise).
- **Migration is a MECHANICAL RULE + a completeness RE-GREP gate, NEVER a
  fixed table.** The test surface is WIDER than any map (127 reflective
  `.capabilities` reads / 33 files; steps keep ADDING files a stale map
  predates — RE-GREP yourself, don't trust the plan's inventory). Rule:
  `CAP_SOLVE∈caps→is_invertible`, `CAP_APPLY_TRANSPOSE∈caps→is_adjointable`,
  strict `==frozenset({...})`→the axis-predicate CONJUNCTION,
  `pytest.raises(MissingCapability)`→by-AXIS (`NotInvertible`/`MissingAdjoint`/
  plain `TypeError match="apply"`). Bare-STRING `"apply_transpose" not in
  op.capabilities` PRECONDITIONS (§1d.2) are MISSED by a CAP-constant grep —
  grep `.capabilities` too; leaving ONE unrewired `AttributeError`s
  post-deletion (a false RED masking whether S† held — M-LITERAL-STRING).
  DONE only when `grep -rn ".capabilities\|MissingCapability\|CAP_*\|_has("
  tests/ orpheus/` returns ZERO.
- **Land-order is W1-additive / W2-ATOMIC / W3-teeth.** The frozenset
  deletion breaks ALL reflective reads at ONCE → W2 is atomic (delete +
  full test-migration in one commit). The exception REPLACE
  (`MissingCapability`→`NotInvertible`/`MissingAdjoint`) must ALSO be
  W2-atomic — a W1 flip breaks every `pytest.raises(MissingCapability)`
  mid-flight (sibling TypeError subclasses don't cross-catch). The
  eager-`.H` flip (raise site `.apply`→`.H`, a REAL behavior change) is the
  ONE W1 change, with its own `.H`-raise-site test migration (M-ADJ-EAGER;
  lazy `wrapper=A.H; raises: wrapper.apply` → `raises(MissingAdjoint): A.H`).
  Re-baseline the pyright ratchet DOWN in-commit (deleting N `# type:
  ignore` + `cast(SupportsInverse)` clears reds; assert the EXACT new floor,
  never trust the count — L10); guard-narrow deletions are CLI-pyright teeth
  (`reportAttributeAccessIssue` — M-GUARD-DELETE-PYRIGHT).
- **A "solve pruned to native realizations" body-rewrite (the composed
  `OperatorProduct.solve` re-route `b.inverse().apply(a.inverse().apply(q))`,
  which FIXES the deleted-solve factors) needs bit-id-INHERITANCE (pre-carve
  `--capture-baseline`) + a structurally-INDEPENDENT dense anchor
  (`np.linalg.solve(as_matrix,q)`) + a Mode-11 SENTINEL.** The re-route
  EXECUTES (`(A@B).inverse().apply` → `InverseOperator.apply=inner.solve` →
  the re-routed body) — but a value-ref spelled the SAME way is tautological,
  so wrap `OperatorProduct.solve` (counter>0). The factor-kind tolerance
  SPLITS: DIRECT factors (Diag/Perm/Scaled/Identity — integer gather /
  identical float path) are `array_equal`; the ITERATIVE sum-factor row
  (Green-vs-Green — math-bit-id but driver-iterated) is
  `assert_regression(SAFETY×driver_tol)` NOT `array_equal` (L7 iterated-drift
  trap). Config-blindness: NON-COMMUTING factors (Diag@3-cycle-Perm) — the
  commuting-D1@D2 designed-green control proves the order-swap mutation is
  BLIND without them.

How to apply: for a coexisting-mechanism retirement, the scaffold gate is
TEMPORARY (deletes with the mechanism) — design its PERMANENT
structural-contract successor FIRST; migrate by rule+regrep not a table
(re-grep the surface yourself — it is wider than any map); retire
atomically (frozenset AND its exception class together); a pruned-body
rewrite rides inheritance + independent-anchor + Mode-11-sentinel with a
direct/iterative tolerance split. The terminal sibling of L13 (the ADDITIVE
design this EXECUTES) / L14 (the iterative Green) / L7 (regression
tolerance) / L10 (assert the pyright delta). `vv §Mode-8/11 / bit-identity /
anti-#2,#11,#19`.

---

## L16 — Verifying a NEW *consumption mode* of a shipped per-cell closure algebra (`assemble()` = the 3rd mode beside `solve`/`apply`) + the relocation-is-behavior-free gate

When a carve adds a THIRD consumption mode of the ONE closure algebra
(SN/diffusion `solve`=sweep + `apply`=matvec already share the per-cell
coefficient source; the new `assemble()` emits the SAME coefficients as
(row,col,value) into a global scipy-sparse `LinearOperator`), and relocates
its module DOWN a layer (`sn/spatial`→`transport/spatial`), six reusable
test-design facts (Phase-2 spatial-substrate + assembly campaign,
`.claude/plans/archive/assembly_mode_crosswalk.md`):

- **The ONE-SOURCE PROOF is the campaign's whole point (the Phase-F twin-path
  guardrail).** `assemble` MUST consume the same coefficient source `solve`/
  `apply` do (DD: `cell_balance.cell_balance_for_streaming`; LD:
  `_ubld.assemble_ubld`+`d1_closed_form`; diffusion:
  `operators._interior_conductance`/`_boundary_closure`). The proof =
  a monkeypatch **sign-flip in the SHARED source** must red BOTH the new
  equivalence gates AND the EXISTING sweep/matvec suites. If ONLY the new gate
  reds → a parallel stencil (twin path) exists → **STOP, fix, log ERR-NNN**.
  Classify every mutation shared-source (reds both) vs assembly-local (reds
  only new: DOF-transposition, wrong-neighbor, walk-order-P). **DD is the
  HIGHER twin-path risk than LD**: DD has NO dense block (its matvec is the
  fused scalar `residual_kernel_batch` → a fresh stencil emission is tempting);
  LD's apply already does `einsum(assemble_ubld["A"], psi)` so LD-assembly is
  scatter-of-EXISTING-blocks. The M1 sign-flip on the DD shared source is the
  single highest-value mutation.
- **Sparse-order ≠ apply-order ⇒ NO gate is 0-ULP; the independence is a
  FEATURE.** `assembled@x` sums each row in CSR order; `apply(x)` in the
  native `np.diff`/`einsum`/left-fold order → G1 (`assembled@x≡apply(x)`) is
  **nulp/`rtol=1e-11`** (bound = row-bandwidth×eps), NEVER `array_equal`
  (vv §bit-identity crit-3). `solve_triangular` (scipy→LAPACK `dtrtrs`) is a
  structurally-INDEPENDENT forward substitution vs the ORPHEUS scalar sweep
  recurrence → G2 (`solve_triangular(P·L·Pᵀ,q)≡sweep.solve(q)`) is **rtol**,
  and that independence EARNS G2 its L2-cross-check status + **discharges the
  sweep-inverse contract question object-level** (here #284). G2's
  triangularity leg (`triu(P·L·Pᵀ,k=1)==0`) is a structural EXACT zero.
- **The `as_matrix`-DELEGATES-to-assembly TAUTOLOGY trap (proactive refute,
  L10).** Once `as_matrix` delegates to densified assembly (the R2 design), the
  naive `op.as_matrix()≡op.assemble().to_dense()` is VACUOUS (assembly vs
  itself). The fuller-view-oracle gate MUST force the **RETAINED probing
  fallback** (`_as_matrix_by_probing`, the pre-carve apply-to-basis loop) ≡
  `assemble().to_dense()`. Keep EXACTLY ONE probed≡assembled pin per family as
  the permanent oracle (an explicit keep decision, coding-standards fuller-view
  exception) — the diffusion loss (already has the `as_matrix`≡hand-posed-FD
  stencil l0 gate) + one DD slab block. Probing `A[i,j]=apply(e_j)[i]` extracts
  ONE coefficient (no summation) but via apply-arithmetic ≠ direct emission →
  still nulp, not 0-ULP.
- **Config: the DOF-transposition + face-pairing mutations are transpose-BLIND
  on a SYMMETRIC operator (Mode-12).** A row/col swap on `A=Aᵀ` is invisible →
  the fixture MUST be het + **asymmetric SigS** + **non-uniform h** so `A≠Aᵀ`
  and the transposition/face-swap is observable (reuse the exact
  `tests/diffusion/test_operators.py` fixture — it already pulls every lever
  with the comment "asymmetric Σ_s so a transpose is observable"). G1's `x`
  MUST be **non-flat fixed-seed random** with non-zero inflow (flat x nulls the
  DD streaming coupling, §0.6/H2). ≥2G always (anti-#3); exclude 1G explicitly,
  ship NO 1G "structural smoke".
- **G1 is object-level ONLY as a matvec, NEVER a derived scalar.** `Δx=0` for a
  FIXED implementation bug `Δ=A_asm−A_apply` is caught by a single GENERIC
  random x w.p.1 (kernel measure-zero); K≈8 fixed-seed random = redundancy;
  G3 (probed≡assembly on the full matrix) is the exhaustive object pin on the
  anchor member. NEVER gate `keff(assembled)≡keff(apply)` — a similarity/
  transpose Δ is spectrally invisible (Mode-12, the #226 step-5b overclaim).
- **The out-of-scope-defect CHARACTERIZATION gate flips LOUD, opposite to
  xfail.** Curvilinear assembly is OUT (blocked on the #282 lagged-ψ½ pole
  seed); characterize it by assembling the spherical walk and POSITIVELY
  asserting `np.any(triu(P·L_sph·Pᵀ,k=1)≠0)` (the defect EXISTS) with an
  actionable message. NOT xfail: xfail-`strict=False` flips silently to xpass
  for a not-yet-landed FEATURE; here we want a LOUD RED forcing a rewrite when
  the defect is FIXED (promote to a G2 triangular gate). Check `error_catalog`
  for a #282 ERR → `catches()`; else cite the issue.
- **RELOCATION is behavior-free BY CONSTRUCTION → lean on existing gates, argue
  AGAINST a new snapshot/hash.** A pure `git mv`+import-rewire changes no
  numerical-code byte, so a NEW output-hash/snapshot is redundant with the
  EXISTING `-W error::DriftWarning` snapshot suites (which already pin byte-id;
  an iterated snapshot is the WRONG bit-id gate anyway, L7). The walls are:
  existing DriftWarning suites green + Sphinx `-W` clean (a `:mod:`/`:class:`
  xref at the old path renders as plain text with NO `-W` warning → ALSO)
  grep-completeness `grep -rn "<old.path>" orpheus/ tests/ docs/` returns ZERO
  (L15 regrep) + importer-suite nodeid-set identical modulo the moved files'
  own paths. Tests travel WITH their modules (relocation ≠ retirement — move
  behavioral AND import-path smokes; markers key on nodeid, `registry.clear()`
  per-collection so no surgery).

How to apply: for a new-consumption-mode-of-a-shipped-algebra carve, the
§0.5 named risk is THE one-source proof (shared-source sign-flip reds BOTH new
AND existing; DD is the twin-path-risk cell). Tolerances: G1 nulp / G2
rtol-via-independent-scipy / structural-EXACT triangularity; NEVER `array_equal`
(sparse-order), NEVER a scalar functional (Mode-12). Force the retained probing
path for the fuller-view-oracle (as_matrix-delegates tautology). Mandatory
asymmetric-SigS+non-uniform-h (transpose observability) + non-flat random x.
Characterize an out-of-scope defect with a positive assert-defect + loud-flip
message, not xfail. Relocation: existing DriftWarning + Sphinx-W + regrep-ZERO,
no new hash. The consumption-mode sibling of L11 (axis-transpose of a shipped
reduction) / L12 (fast-path→composed fold); `vv §bit-identity / Mode-8/12 /
anti-#2,#3,#4,#11`; mirrors L7/L10/L15.

## L17 — Verifying the reverse-SCAN transpose-solve (#280) + its apply-loop unification: the assembled-Mᵀ oracle, the self-referential-relocation trap, and the retired-vocab reconciliation

The #280 carve builds the 4th face of the `(L+C)` 2×2 — the reverse-DAG
transpose-solve — as a reverse-SCAN coherent with the forward Blelloch
`_run` (NOT the reverse-LOOP the a3 spec assumed), and first unifies the
apply-loop family `{_apply_walk (fwd matvec), loss_action_transpose (adj
matvec)}`. Full recipe `a3_reverse_scan_transpose_verification.md`; the
durable test-design facts:

- **The operator VOCAB is stale — reconcile against the #226 typed-predicate
  world FIRST (L10 refute-the-premise).** The a3 spec's
  `InvertibleOperator.solve_transpose`/`CAP_SOLVE_TRANSPOSE`/`_AdjointOperator.solve`/
  `CAP_SOLVE` is RETIRED (L13/L15 frozenset→predicate landed). CURRENT
  primitive = **`SweepOperator.apply_transpose`** (`SweepOperator=(L+C).inverse()`;
  the reverse-scan IS the inverse operator's transpose matvec), gated by
  **`SweepOperator.is_adjointable`→True**. The metric transpose-solve
  `A.inverse().H.apply(b)=G⁻¹·apply_transpose(G·b)` falls out of the EXISTING
  `_AdjointOperator.apply` FOR FREE — NO `_AdjointOperator.solve` is minted;
  `_AdjointOperator.inverse()`+`is_invertible` close the coherence
  `A.H.inverse()≡A.inverse().H`. Design gates against the LIVE surface, never
  the retired string tags (`InverseWrapMixin` docstring already names #280).
- **The existing 0-ULP `array_equal` matvec canaries are SELF-REFERENTIAL →
  necessary-not-sufficient for a RELOCATION.** `test_removal_form_matvec_sweep`'s
  fwd/adj `array_equal` gates compare `op.apply[_transpose]` against the SAME
  `loss_action[_transpose]` on a FRESH `StreamingOperator(σ_r)` — a
  reduction-tree "override-owns-it vs leaf-sum-leak" discriminator, NOT a value
  oracle. A relocation moving BOTH the SUT path AND the reference path (same
  relocated code) leaves them GREEN even if values shifted (the twin/Mode-11
  hazard). The genuine 0-ULP relocation proof is a **FROZEN pre-carve baseline
  snapshot** (`assert_regression --capture-baseline`, both orientations,
  slab/sphere/cyl 2G) — structurally independent because captured BEFORE the code.
- **The new shared 1-D loop frame needs its OWN structural pin (there is NONE
  today).** Multi-D has `test_one_octant_walk` (runtime `_interior_walk` spy +
  AST tripwire on `{is_solve,is_apply,is_matvec}`); the 1-D `_OneDimScanWalk`
  has no spy (its `_apply_walk`/`loss_action_transpose` were verbatim
  relocations, three separate skeletons). 2.5a CREATES the shared frame → mint
  the ORIENTATION sibling: a Mode-11-WRAP runtime spy (both orientations route
  through the ONE method, count>0) + an AST tripwire forbidding
  `is_adjoint`/`is_forward`/`is_transpose`/`is_reverse` as identifiers, demanding
  an orientation-carrying OBJECT (like `_ApplyOperands`/`_SolveOperands`).
- **The assembly mode gives a NEW structurally-independent transpose oracle —
  Cartesian ONLY.** `permuted=block.as_matrix()[ix_(order,order)]` is
  walk-order lower-tri; the reverse-solve oracle is
  `solve_triangular(permuted.T, b[order], lower=False)` (LAPACK back-sub,
  shares NO code with the ORPHEUS scan). CATCHES the wrong TRANSPOSED-SCAN-
  COEFFICIENT (a'/b' of the reversed affine recurrence — scan≠loop) + gives the
  exact triangularity certificate + discharges "reverse-substitution IS
  sweep_transpose" object-level (transpose analog of #284). But assembly is
  Cartesian-only (`SNMesh.streaming` gate) → **the SPHERE μ-reversal/mirror bug
  is INVISIBLE to it; the dense-apply G2 `np.linalg.solve(M.T,b)` (M from fwd
  apply) stays the KEYSTONE on the mandatory sphere leg.** LD caveat: if
  `loss_action_transpose` is DD-only (no moment tail), the LD assembled-Mᵀ
  oracle is moot — verify LD-adjoint scope first.
- **Reverse-SCAN's NEW modes over the reverse-LOOP:** (i) wrong transposed scan
  coeffs → G2 dense-Mᵀ + assembled-Mᵀ; (ii) two-denom seam (÷V apply vs ×V
  `_run` scan, #242) → G1 round-trip + G2; (iii) curvilinear μ-level-coupled scan
  reversal + `angular_adjoint` + Carlson mirror → G1/G2 SPHERE leg (slab NULLS);
  all object-level — NEVER credit `sweep_transpose` on `k*`/norm (`eig(Kᵀ)=eig(K)`
  is adjoint-blind, Mode-12; memory "NEVER keff(asm)≡keff(apply)").
- **Scaffold: G3 full-loss `(L+C−S−B)` reciprocity lands PRE-carve** (S† at
  `15185e5`; extend `test_g_adjoint_reciprocity` L+C−B→L+C−S−B, asymmetric SigS
  ≥2G, per-group one-hot φ vv L27) — it hardens the `apply_transpose` surface
  2.5a rebuilds, giving the adjoint bit-identity a composite-level canary.
  #282-fix (CONDITIONAL, unruled): if the augmented ψ(·,μ=−1) state lands in
  2.5, the `test_282_characterization` loud-flip fires (L16) and its successor
  is a spherical G2 gate on the augmented system (seed rows first, back edge →
  forward edge; teeth = swap the seed coupling direction).

How to apply: reverse-DAG transpose-solve or apply-loop-unification carve →
FIRST reconcile the retired `CAP_*`/`solve_transpose` vocab to the live
`SweepOperator.apply_transpose`/`is_adjointable`; frozen-baseline the
relocation (the array_equal canaries are self-referential); mint the
orientation-OBJECT AST tripwire + 1-D spy (none exists); the assembled-Mᵀ
oracle is a real Cartesian L2 cross-check but the SPHERE dense-apply G2 is the
keystone (assembly is Cartesian-only); object-level gates ONLY (Mode-12);
G3 reciprocity is pre-carve. The transpose/inverse-axis sibling of L13/L14/L15
(the operator-taxonomy family) + L16 (the assembly consumption mode);
`vv §bit-identity / Mode-8/11/12 / anti-#2,#3`.

## L18 — Verifying a CARRIER AUGMENTATION's ψ½ block (#282 route (a): the direct μ=−1 starting-direction seed as a typed per-level block). ⚠ SHARPENED 2026-07-06: the "zero-G-weight DOF → reciprocity IDENTICALLY blind → encode the blindness as a positive pin" claim was REFUTED by the derivation-of-record — the STATE metric is G_sd=V_cell (SPD), NOT the ghost zero, so reciprocity CLOSES Mode-12 (a seed-row flip REDS). Still-valid: closed-form-direct-solver pillar, prove-the-block-is-CONSUMED, cold-residual acceptance number, "cylinder unmoved" is a TESTED assertion

The #282 fix adds a typed per-starting-direction-level ψ½ block to the
curvilinear composite (sphere 1 level, cyl n_polar; slab/cart ABSENT) and makes
`solve` march the seed rows FIRST via `carlson_inward_sweep_from_source` on the
TRUE source's q½ block (the lag dies; the sphere solve becomes a genuine
triangular inverse). Full spec `a3_solve_transpose_verification.md` §16. Durable
test-design facts (the sub-phase sibling of L16/L17):

- **⚠ CORRECTED: a carrier DOF's Hilbert STATE metric is set by its OPERATOR
  ROLE, NOT by its angular-INTEGRATION weight — conflating them gives the WRONG
  (ghost) metric and a Mode-12 FALSE-GREEN** (derivation-of-record
  `starting_direction_metric_gauge_derivation.md`, 2026-07-06; this REPLACES my
  earlier "zero-weight DOF is reciprocity-invisible / encode the blindness as a
  positive pin" claim). The Carlson α₁ᐟ₂=0 is the angular THROUGH-FLUX weight
  (correctly excludes ψ½ from `φ=Σw_n ψ_n` — an OPERATOR coefficient); the STATE
  metric is set by ψ½'s role as a first-class RADIAL field (its self-block A_ss
  is a BANDED radial transport operator `μ∂_r+σ`, not a grazing angular face —
  diag_gsd_05). So **G_sd = V_cell (SPD)**, mirroring `G_bulk=V·w`; since
  `apply_transpose≡Aᵀ` EXACTLY, reciprocity is GAUGE-FREE among ALL SPD G_sd.
  **G_sd=0 is the single FORBIDDEN value**: it puts the seed in null(G), severing
  the seed→bulk A_bs coupling from `A.H=G⁻¹AᵀG` — a WRONG adjoint the instant the
  seed carries data (production random-seed defect 1.3e-2), green ONLY because the
  old gate fed a present-but-ZERO seed. **With SPD G_sd, reciprocity DOES catch a
  seed-row transpose error (Mode-12 CLOSED)** — INVERT any gate built on the old
  blindness (rename `_blind_to_` → `_catches_`; assert flip REDS not stays-green).
- **A Mode-12-CLOSURE reciprocity gate needs BOTH legs — control-holds AND
  mutated-reds; the mutated leg ALONE is fooled by a broken baseline.** Flip
  `_seed_rows_forward` (forward A_ss) but NOT `_seed_rows_transpose` (A.H's
  independent reverse mode) → an apply-vs-transpose inconsistency the SPD metric
  sees. Assert (1) UNMUTATED nonzero-seed reciprocity HOLDS `<1e-12` (baseline is
  the honest V_cell adjoint) AND (2) the flip REDS `>1e-6`. WITHOUT leg (1),
  reverting G_sd→0 gives flip-defect 0.107>tol TOO (baseline broken, flip
  invisible — a broken baseline MIMICKING "caught") → false green. Measured dense
  sweep {0/id/V_cell/V·w}×{zero/random/flip}: g_sd=0 → random 0.107 / flip 0.107
  (BLIND, unchanged); V_cell → random 1e-16 / flip 1.000 (CLOSED); faithfulness
  `‖G⁺TG−A.H‖`=2.8e-14. The gauge is INERT (control leg c): corner V[-1]→2·V[-1]
  keeps reciprocity green AND the bulk solve bit-identical (observable-level).
  Feed the independent `_g_inner` seed term from PRODUCTION's
  `starting_direction_space.inner_product_weights` (the gauge must MATCH A.H's,
  else false-red — the seed metric verifies no VALUE, only SPD-ness, so this is
  NOT reference contamination; the load-bearing content is the TRANSPOSE, caught
  because the error is in Aᵀ not G). Canonical Mode-12 gate = the promoted
  `diag_gsd_02` dense sweep; production-path corroboration = the inverted
  `test_282::test_mode12_g_reciprocity_catches_a_seed_row_flip`.
- **The direct-solver structural pillar is the closed-form exponential ODE, NOT
  the recurrence hand-trace.** The μ=−1 equation `−dφ/dr+σφ=q` has
  `φ(r)=q/σ+(φ_R−q/σ)e^{−σ(R−r)}`; DD converges O(Δr²) to it. The EXISTING
  Carlson pins hand-trace the SAME DD recurrence (procedural, NOT structural —
  the ERR-032 shared-upstream hazard) and route through the PROXY `Q̄=σφ₀/Σw`,
  never the true-source direct entry. Add the convergence pin on an ARBITRARY
  `Q_bar`; the GRADED-mesh row is MANDATORY (uniform is blind to a `dr[k]` index
  drift, Mode 5). The q½ source fold needs the anisotropic 2-term
  `P_ℓ(−1)=(−1)^ℓ` hand-check manufactured FIRST (an all-isotropic suite is
  silently blind to a dropped/mis-signed φ_ℓ≥1, §0.6).
- **Prove the new carrier block is CONSUMED before crediting any gate (term
  activation, Mode-11-adjacent).** Zero the source's q½ block → the solve MUST
  move (`solve(b_with_q½) ≠ solve(b, q½=0)`); if equal, the whole augmentation
  is INERT (silent coverage loss). Pair with a Mode-11 WRAP sentinel on
  `carlson_inward_sweep_from_source` (count>0 on sphere solve, ==0 on slab).
- **The cold-residual is THE acceptance number, promoted from a diagnostic —
  a LAG-DEATH classifier, distinct from seed correctness.** `‖A·solve(b)−b‖/‖b‖`
  over the FULL augmented field, COLD: sphere 5.18e5 → `<1e-11` (slab-level, no
  iterate-threading, no sphere slack — the fix EARNS the tight tol); slab/cyl
  controls already ~7e-16 must STAY. It certifies solve↔apply CONSISTENCY (lag
  dead), NOT coefficients (the direct-solver pin + Euclidean `Mᵀ` own those).
  Companions: seed-INSENSITIVITY (Δ 4.57e-2 → bitwise), the decisive classifier
  (`b=A.apply(psi0)`; cold solve recovers `psi0` → the fixed point did not move),
  the coarse `level_symmetric` S4 end-to-end (SI-NaN/Krylov-negative → finite+
  positive; a physicality gate, NOT precision NOT keff), the c=0 no-outer-
  iteration degenerate.
- **"Curvilinear re-baselines" is too coarse for the CYLINDER — assert-unmoved
  FIRST.** The cyl seed is weightless (α=0 AND the per-level α-dome telescopes,
  the #282 0.0-bit row) → the frozen `walk_matvec_cyl_2g` MUST stay `array_equal`
  (a DIFFERENT ψ½ carrier value gives the SAME physical output iff the seed is
  truly annihilated). A cyl move HALTS the carve (FP-reorder → re-capture WITH a
  3-criteria record + "weightless was too strong"; or a leak-bug). Only the
  SPHERE re-captures (principled, [[feedback-principled-over-bit-identical]]);
  the slab stays bitwise. Sharper than the roadmap's blanket "curvilinear
  re-captures" — it pins the seed-weightless INVARIANT a silent re-capture loses.
- **Sequencing (R10: 2.5d BEFORE 2.5b) sets the gate split.** The fix lands
  first → the 2.5b sphere G1/G2 `Mᵀ` chain is written EXACT single-pass (rtol
  1e-11, NO iterate-threading — SUPERSEDES the a3 §0.3 loose-1e-9). 2.5d's own
  successor to the loud-flipped `test_282_characterization` is triangularity
  ONLY (`triu(PᵀM_augP,1)==0` on the matvec-probed AUGMENTED matrix, seed-rows-
  first order — NO assembly-emitter extension); the LAPACK-≡-sweep leg waits for
  the 2.5b reverse-scan.
- **Framing-agnostic by construction.** The ψ½ block's storage (F1 third block /
  F2 angular-boundary-trace / F3 zero-weight extra ordinate) is under concurrent
  design — write gates against the carrier's PUBLIC contract (presence keyed by
  curvature both-ways, to_flat round-trip, add/scalar closure, zero-metric-weight,
  FULL-field residual) so they hold in ALL three; `# FRAMING:` sub-notes ONLY on
  storage-layout assertions. All three give the seed zero G-weight → the
  reciprocity-blindness note is framing-independent.
- **The ℓ=0-ONLY q½ source fold SILENTLY BREAKS the MMS fixed-source — the
  eigenvalue wall is BLIND (migration wave 3, 2026-07-04, concrete war-story of
  the anisotropic-`P_ℓ(−1)`-first warning above).** An MMS fixed-source is
  INHERENTLY anisotropic: the streaming operator `Ω·∇ψ` manufactures a ℓ=1
  source even for an ISOTROPIC trial `ψ=A(r)` (`Q = μA'(r) + σ_t A(r)`). A
  `from_angular_source` fold that folds only ℓ=0 (`q½=½q₀`, its documented
  "honest scope: production sources are isotropic") seeds the pole march with a
  WRONG q½ → the WHOLE inward march corrupts (interior AND pole wrong; sphere L2
  stuck ~6.7e-2 vs the frozen O(h²) 3.7e-3; NEGATIVE convergence order). The
  EIGENVALUE path is BLIND (isotropic fission/scatter → ℓ=0 fold EXACT), so a
  green k_inf / SI≡Krylov wall MISSES it; ONLY the anisotropic MMS FIXED-SOURCE
  pole-cell gate reds. Two corollaries for the migration test-architect: (1)
  before trusting an MMS fixed-source as an ABSORB-gate under a ψ½-source carve,
  verify the fold's MOMENT REACH ≥ the MMS source's anisotropy — a "production
  sources are isotropic" scope note is FALSIFIED by the MMS's own streaming
  source; (2) a pole-cell CONVERGENCE gate cannot be re-baselined (no stored
  value) — a negative order under the carve is a STOP-and-report, never a
  re-capture. The isotropic-source solve (reflective `q/Σ_t`) stays green, which
  is exactly why the break hides until the anisotropic gate runs.

How to apply: a carve that ADDS a zero-measure/zero-weight DOF to a
metric-carrying composite (a starting-direction seed, a ghost cell, any
α=0 phase-space DOF) → the G-adjoint reciprocity gate is IDENTICALLY blind
(Mode 12); constrain the DOF by the EUCLIDEAN transpose oracle + a closed-form
direct-solver pin + the full-field round-trip, encode the blindness as a
positive pin, and PROVE the DOF is consumed (zero its source → the solve moves)
before crediting anything. Promote the seed-lag diagnostics (cold-residual,
seed-insensitivity, decisive classifier) to acceptance gates. Assert-unmoved
FIRST on the config where the DOF is weightless (a silent re-capture discards the
invariance). The direct-solver's structural reference is the closed-form ODE, not
the recurrence hand-trace (ERR-032). Sub-phase sibling of L16/L17;
`vv §Mode-10/11/12 / anti-#8 / §bit-identity`.

## L19 — Verifying the adjoint-inverse SWAP-LAW wiring (#280 2.5c, `A.H.inverse() ≡ A.inverse().H`): Gate 1 is a FORWARD-only reciprocity structurally independent of the reverse-scan, `b` MUST be bulk-only source-carried, Gate 2 is a bit-identical OBJECT identity, the `.H`≠Euclidean "slab stays GREEN" discriminator is FALSE, and the predicate flip MUST propagate to the capability-survival CONTRACT

The 2.5c terminal of the #280 reverse-scan sub-campaign wires the swap law
`(A*)⁻¹ = (A⁻¹)*` as an OBJECT IDENTITY: `_AdjointOperator.inverse() =
inner.inverse().H`, `SweepOperator.apply_transpose = inner.solve_transpose` (the
2.5b reverse-scan), `SweepOperator.is_adjointable` flips True over the
`InvertibleOperator` arm; the metric adjoint-solve `A.H.inverse().apply(b) =
G⁺·solve_transpose(G·b)` falls out of the EXISTING `_AdjointOperator.apply` FOR
FREE. Gate file `tests/sn/operators/test_inverse_adjoint_coherence.py` (helpers
reused from `test_loss_transpose_solve.py`). Durable gate-design facts (the
wiring-proof sibling of L13–L18):

- **Gate 1 (forward-matvec G-reciprocity `⟨Aψ,x⟩_G=⟨ψ,b⟩_G`, `x=A.H.inverse().apply(b)`)
  is structurally independent of the reverse-scan BY CONSTRUCTION.** The
  verification arithmetic is FORWARD `A.apply` + the metric inner product ONLY —
  it NEVER calls `apply_transpose`/`solve_transpose`; the reverse-scan lives
  solely inside the SUT `x`. `⟨Aψ,(A*)⁻¹b⟩_G = ⟨ψ,A*(A*)⁻¹b⟩_G = ⟨ψ,b⟩_G` holds
  iff `x` genuinely solves `A*x=b`, paired against the independently-verified
  forward operator (the strongest available ground: forward apply is the SoT, the
  metric is pinned by `test_g_adjoint_reciprocity`). This is the elegant way to
  gate a reverse solve WITHOUT the reference sharing its code.
- **`b` MUST be bulk-only source-carried; a random FULL `b` falsely reds Gate 1/3
  (~1e-1) even UNMUTATED.** solve/apply (and transposes) are genuine inverses only
  OFF the boundary/seed outflow FREE-DOF slots (#284) AND off the metric null
  space (tangential trace |Ω·n|=0 ⊕ the all-zero-ghost-metric seed). A bulk-only
  `b` (zero boundary/seed) lies in the range of `A*` → Gate 1/3 clean to ~1e-16 on
  slab/sphere/cyl. ψ (the test vector) stays FULL (bulk+trace random) so the trace
  coupling + trace metric ARE exercised in the pairing; seed present-but-ZERO
  (Mode-12 regime). The well-posed round-trip is `(A*)⁻¹∘A*` (=I on bulk);
  Gate 3's `A*∘(A*)⁻¹` is =I only on the source-carried bulk — compare BULK, never
  the full field.
- **Gate 2 (swap law) is a BIT-IDENTICAL object identity, NOT an rtol check.**
  `A.H.inverse()` returns LITERALLY `A.inverse().H`, so both spellings build+run
  the SAME object graph on the same `b` → `assert_array_equal` (0 ULP), stronger
  than the spec's allowed rtol=1e-12. Compare bulk ⊕ boundary ⊕ seed.
- **The M-ADJ-metric "slab stays GREEN" discriminator is FALSE — flag it, don't
  encode it.** The spec predicted skipping the `G⁺`/`G` wrap in
  `_AdjointOperator.apply` reds Gate 1 only on sphere/cyl (slab "metric trivial").
  EMPIRICALLY the composite metric `G = V·w_n` (bulk) ⊕ `|Ω·n|·w_n` (trace) is
  non-trivial on EVERY geometry (non-uniform mesh + the bulk-vs-trace weight
  mismatch that streaming couples), so `A*=G⁻¹AᵀG≠Aᵀ` everywhere and Gate 1 reds on
  slab too (slab 0.33 / sphere 0.19 / cyl 0.13). The `.H`≠Euclidean claim is TRUE
  universally (the metric wrap is load-bearing on every geometry — STRONGER
  coverage than the intended discriminator), just NOT a slab-vs-curvilinear split.
  NEVER design a `.H`≠Euclidean gate as a slab-green discriminator on a composite
  bulk⊕trace metric — the trace weight `|Ω·n|` ≠ the bulk weight `V` on ANY
  geometry (sibling of L1: verify the CONCRETE metric before assuming slab nulls a
  term). The mutation still has teeth (reds all geoms); assert RED on all three.
- **`A.H.inverse()` needs the Design-C `invertible(A.H)` TypeGuard bridge for
  pyright.** `.inverse()` lives on `SupportsInverse`; `.H` returns the base
  `LinearOperator` (which erases it). Wrap: `ah=A.H; if invertible(ah): return
  ah.inverse()` (`invertible` is a TypeGuard — POSITIVE branch only; a trailing
  `pytest.fail` is NoReturn). The bridge doubles as a live
  `_AdjointOperator.is_invertible` pin and reads predicates only (no solve) so it
  does NOT perturb the Mode-11 counters. `A.inverse().H` needs NO bridge (`A` is a
  concrete `InvertibleOperator`).
- **Mode-11 wrap sentinel: wrap `InvertibleOperator.solve_transpose` (>0) AND
  `.solve` (==0).** `A.H.inverse().apply(b)` MUST enter the reverse-scan and NEVER
  the forward solve; the routing mutation (`SweepOperator.apply_transpose →
  inner.solve` forward) reds BOTH counters. All geometries give st=1/solve=0
  (sphere seed is single-call exact post-2.5d — no iterate-threading).
- **The predicate flip is SURGICAL and MUST propagate to the capability-survival
  CONTRACT in the SAME landing.** `SweepOperator.is_adjointable` flips True over
  `InvertibleOperator`; the sibling wrap-delegates `InverseOperator`/`GreenOperator`
  STAY False (no override — keyed on `isinstance(inner, InvertibleOperator) and
  inner.is_adjointable`). **The Part-A landing that flipped it BROKE TWO existing
  forward canaries (both the same flip; blast radius WIDER than any one contract
  table — the L14/L15 grep-audit lesson):** (a)
  `test_capability_survival.py::test_inverse_adjoint_contract_keystone_v2_sn` row
  `((L+C).inverse(), True, False, INVERTIBLE)` (`adj=False`→`True`); (b)
  `test_inverse_operator_equivalence.py::test_inverse_returns_sweep_operator_surface`
  `assert inv.is_invertible is True and inv.is_adjointable is False` (→`is True`).
  A predicate-flip carve's retirement audit MUST grep EVERY `is_adjointable is
  False`/`adj=False` assertion on the flipped type (NOT just the contract-table
  rows) and update them in the same commit; a single-file gate-authoring session
  downstream cannot fix them (scope) — report as required Part-A completion items.

How to apply: verifying an adjoint-inverse / swap-law wiring (`A.H.inverse()` /
`A.inverse().H`) → Gate 1 is the FORWARD-only G-reciprocity (structurally
independent of the reverse-scan, the keystone) on a bulk-only source-carried `b`
with a FULL ψ; Gate 2 is `array_equal` (object identity); route `A.H.inverse()`
through the `invertible()` TypeGuard bridge; Mode-11-wrap the reverse+forward
inner readers. NEVER design a `.H`≠Euclidean gate as a slab-green discriminator —
the composite bulk⊕trace metric is non-trivial on every geometry. A predicate flip
MUST propagate to the capability-survival contract table. Terminal wiring-proof of
the #280 reverse-scan chain (L13→L17 vocab, L18 carrier);
`vv §Mode-11/12 / L11 structural-independence / L1 config-blindness`.

---

## L20 — Verifying a STRUCTURE-ONLY ABC re-parent + stringly-key→typed-key `[K]` generalization (intended bit-identical): the crux is a MULTI-index synthetic fixture (the shared layout is exercised in production at ONE key-index → the generalized index term is never activated), the metric is a REGRESSION-GUARD (not a new gate) with an ERR-067 "NO metric on the ABC" tripwire, and the load-bearing MRO check is sibling-NOT-child

The C2 `FaceField(Field)` carve (design `facefield_codim1_design.md` §5): introduce a
parent ABC that owns a shared flat-buffer discipline ONCE, re-parent three sibling
field types under it (`AngularBoundaryField`/`ScalarBoundaryField`/
`StartingDirectionField`), and generalize `FaceLayout`/`FaceSlot` from `str` keys to a
typed key `[K]` so the pole space (`(level,sign)` keys) stops re-implementing
`_leg_offset`/`cells_slice`/`corner_slice` and shares the ONE impl with the trace
(`str` keys). Intended bit-identical; nothing numeric moves. Five reusable facts:

- **The crux is a MULTI-INDEX SYNTHETIC fixture — the shared layout is exercised in
  PRODUCTION at exactly ONE key-index, so the generalized index term is never
  activated (config-blindness, sharpening of L1/L11).** `StartingDirectionSpace`'s
  offset formula is `(2·pos + sign_pos)·per_sign` where `pos = levels.index(level)`,
  but production (sphere-GL) carries a SINGLE seed level → `pos ≡ 0` always → the
  level-index term is DEAD in every existing pin (`test_starting_direction_carrier.py`
  builds only `_sphere`). A typed-key `FaceLayout[(level,sign)]` bug in the
  level-position mapping is INVISIBLE to the whole suite. Fix = a direct
  `StartingDirectionSpace.for_levels((0,2,5), ng, nx, cell_volumes=…)` synthetic
  (valid — the ctor guard only rejects empty/non-ascending), forcing `pos ∈ {0,1,2}`.
  The LOAD-BEARING mutation is `2*pos → 1*pos`: it REDs the multi-level golden at
  `pos≥1` while the single-level sphere STAYS green — the asymmetry IS the
  config-blindness evidence (same shape as L11's product-RED/LS-GREEN). The golden is
  HAND-COMPUTED (`(2·pos+sign_pos)·per_sign`), NOT SUT-captured, so it holds
  independently of the carve; pin `.offset`+`cells_slice`/`corner_slice` start/stop AND
  slice CONTENTS via an `np.arange` fingerprint fill (a transposed/mis-offset slot
  diverges; a bare shape check is blind to a transpose). The `str` regime is already
  golden'd by `test_face_layout.py::TestFromNamedShapes` (hardcoded `.offset==0/16/32/56`).

- **The metric is a REGRESSION-GUARD, not a new gate — with an ERR-067 tripwire that
  the ABC stays metric-free.** "Structure-only, metric descends per-leaf" means the
  carve MUST NOT touch the per-leaf metrics (trace `|Ω·n|·w`, pole `V_cell`). Lean on
  the EXISTING object-level metric gates kept green + unmodified
  (`test_starting_direction_metric.py::test_shipped_metric_block_values` `atol=0.0`;
  `TestA4SeedStateMetricVcell`; the trace 0-ULP `test_one_kernel_reproduces_trace_metric_at_0ulp`).
  ADD ONE new tripwire: a mutation that gives the ABC a UNIFORM face metric
  (`(1−µ²)·w ≡ 0` at the pole — the retired ghost!) must RED the byte-exact `V_cell`
  gate. That is the ERR-067 guardrail: "NO metric on the `FaceField` ABC" is precisely
  what stops `G_sd=0` from returning. Pole gates are Mode-12-SAFE only because they
  assert the weight ARRAY object (`assert_array_equal` on `cells_view(weights)`), never
  a scalar functional; every pole gate feeds a NONZERO seed + a control leg (the twin
  bulk/trace-only composite) — a zero seed sits in the ghost's invariance group (L18).

- **The load-bearing MRO check is SIBLING-NOT-CHILD.** After `X` and `Y` both become
  children of the new parent `P`, the dispatch that discriminates them
  (`isinstance(self.boundary, BoundaryField)` / `isinstance(self.starting_direction,
  StartingDirectionField)` in `full_field.py:274/280`) survives ONLY if the re-parent
  keeps them SIBLINGS: `isinstance(sd_field, BoundaryField) is False` and
  `isinstance(boundary_field, StartingDirectionField) is False` MUST stay — assert both
  (the cross-slot contamination guard). The break mode to gate against: collapsing the
  INTERMEDIATE name (`BoundaryField`) INTO the new parent `P` breaks
  `isinstance(x, BoundaryField)` at the consumer. Also gate `P in cls.__mro__` (between
  child and `Field`), `P` exported from `__all__`, `isinstance(x, Field)` stays True for
  every leaf (the `functional.py` consumer), and — the subtle seam — the flat-buffer
  validation (moving UP from `BoundaryField.__post_init__` onto `P`) fires EXACTLY ONCE
  through the `super().__post_init__()` chain (a moved-up check must not double-run or
  skip; pin the layout-less-space raise still fires with the right leaf type-name).

- **`to_flat`/`from_flat` round-trip is bit-identical BY CONSTRUCTION iff the flat
  offsets are unchanged** — it reads `.values` (the flat buffer), never `FaceLayout`
  directly. So the offset gate DISCHARGES the round-trip; no new serialization
  mechanism, just keep the existing `TestA2FlatRoundTrip` green (incl. its dropped-seed
  `_two_block_to_flat` mutation tooth) and extend the flat-total assertion to the
  multi-level synthetic.

- **Retest = the offset/metric/MRO seams, not the whole solver.** A structure-only
  relocation is behavior-free by construction; the proof is (offsets byte-identical
  both regimes) ⊕ (metric regression-guards green + the ERR-067 tripwire) ⊕ (MRO
  sibling-not-child + post_init-once) ⊕ (round-trip inherited from offsets). Deliver
  TWO new files (`test_face_layout_typed_key.py`, `test_facefield_hierarchy.py`); keep
  the commit-1 metric suite + `test_face_layout.py` + `test_face_streaming_normal.py`
  UNMODIFIED as guards.

How to apply: for any "introduce a parent ABC + re-parent siblings + generalize a
stringly-keyed layout to `[K]`" bit-identical carve — (1) find the production
key-index cardinality; if the shared layout is exercised at ONE index, MANUFACTURE a
multi-index synthetic and make the load-bearing mutation the generalized index term
(`n*pos→pos`), demonstrated RED on multi-index / GREEN on single-index; (2) hand-compute
the offset golden so it is carve-independent, pin slice CONTENTS not just shape; (3)
treat per-leaf metrics as regression-guards + add ONE "ABC stays metric-free" tripwire
(the ERR-067/`G_sd=0` guardrail); (4) the MRO gate is sibling-not-child +
`post_init`-fires-once, and the break mode is collapsing an intermediate class name;
(5) round-trip is inherited from the offset gate. `vv §Mode-12 / L11 / L1 config-blindness`.

---

## L21 — Verifying a WRAP resolvent operator that DELEGATES to an already-verified free-function engine (`A_BB.solve` WRAPS `carlson_inward_sweep_from_source`; coupled-block campaign step 1c): the equivalence keystone is bit-id INHERITANCE + a Mode-11 call-COUNT spy; structural independence MUST come from the ENGINE's OWN closed-form (NOT its recurrence); and an operator-internal transform line you CANNOT monkeypatch (a `[:, ::-1]` data reversal) is pinned by a spy on the call ARGS + a graded-mesh non-vacuity check

When an operator's `.solve`/`.apply` is a thin WRAP of a standalone engine
already carrying its own L0/L1 gates (the #282 `carlson_inward_sweep_from_source`
had a full closed-form convergence gate BEFORE `RadialCharacteristicOperator`
wrapped it), the operator is the SUT — its gates add the WRAP layer (space
cells/corner views, source-block reading, the two-leg composition) the engine
gate never touches. Five reusable test-design facts, recurring across every
"WRAP→bit-id" step of a block-operator campaign:

- **The equivalence keystone is bit-id INHERITANCE + a Mode-11 call-COUNT
  sentinel, not a fresh value claim.** Monkeypatch the engine in the OPERATOR's
  module namespace (`_rc_mod.carlson_inward_sweep_from_source`) with a spy that
  DELEGATES to the real engine and records `(args, result)`; assert
  `len(calls) == 2·n_levels` (2 legs/level) AND `solve.values ==`
  (`array_equal`) an independent two-leg reference built with the REAL engine
  (imported at test top, UNPATCHED). The count proves the path is EXECUTED (a
  divergent inlined copy leaves the counter at 0 — Mode-11); the bit-id proves
  the WRAP changed nothing. Necessary-NOT-sufficient (L2) → pair with a
  structurally-independent value anchor.
- **Structural independence is from the ENGINE's OWN gate, not just the SUT
  (ERR-032).** The operator's convergence anchor MUST be the analytic
  closed-form ODE (`φ = q/σ + (φ_R − q/σ)e^{−σ(R−r)}` for the μ=−1 ray — a pure
  exponential, no marching), the SAME reference the engine gate uses — NOT a
  hand-re-execution of the DD recurrence (procedural, not structural). Anchor
  the INWARD leg's cells (its entry face φ_R = corner_in is FREELY settable, so
  it maps 1:1 to the closed form); the outward leg's entry is the derived
  pole_face (harder to anchor cleanly → cover it via the spy, below). R=1.0
  keeps σR≤1.3 so the O(Δr²) regime is reached by nx=64 (R=4.0 gave ratios
  3.16→3.75, still short of the [3.4,4.6] band at nx=64); DISTINCT per-group
  (σ,q,φ_R) + max-over-groups error makes it genuinely ≥2G and catches a
  group-axis view transpose.
- **An operator-INTERNAL transform line you CANNOT monkeypatch (the `[:, ::-1]`
  outward-leg reversal + `[::-1]` un-reversal) is pinned by the spy on the call
  ARGS + a graded-mesh non-vacuity check.** You may not `git checkout` the
  committed operator, and monkeypatching a bare slice inside `solve` is
  impossible — so assert the outward call received `dr[::-1]`, `q_plus[:,::-1]`,
  `σ[:,::-1]` (the WRAP's data-carried-orientation contract, the "2.5a
  discipline"). The `dr`-WIDTH reversal is vv Mode-5 BLIND on a uniform mesh
  (`dr[::-1]==dr`), so the gate MUST run on a GRADED mesh and assert
  `dr[::-1] ≠ dr` (non-vacuity) — else a dropped width-reversal hides. (The
  `q`/`σ` reversals are caught on ANY spatially-varying-data mesh; only the
  width reversal needs graded. Measured dropped-reversal delta: outward cells
  move O(1) — 5.3e-1.)
- **Pole continuation / leg-chaining is pinned by the SAME spy:** outward call's
  `bc_outer_value` == inward call's returned `phi_face_final` (the exit IS the
  entry, R13) + assert the exit face is non-trivial (≠0) so the equality is not
  vacuously satisfied by zeros.
- **The ISOLATED resolvent adjoint (`solve_transpose`)'s consistency partner is
  the EUCLIDEAN inner product (plain dot), NOT the `V_cell` Hilbert metric.**
  The pure ray-block transpose is the Euclidean reverse-mode adjoint; the
  metric `.H` is realized ONCE at the composite (L19). Do NOT reuse the sibling
  `_g_recip` (it carries `G_sd`). Gate `⟨solve u,v⟩=⟨u,solve_transpose v⟩`
  (< 1e-11, het σ, ≥2 draws) + assert the source μ=+1 corner cotangent is
  EXACTLY 0 (the q½ fold never writes that slot). Teeth: a sign flip in the
  engine transpose (`−f_bar → +f_bar`) → defect 5.9e-17 → 1.0.

How to apply: for a WRAP operator over a verified engine, ship (1) the count-spy
+ bit-id-reference keystone; (2) an operator-level convergence gate re-anchored
on the engine's OWN closed form (graded MANDATORY, ≥2G distinct-group,
φ_R≠q/σ), teethed by monkeypatching the engine with DD-coefficient mutants
(closure/denom sign, diamond factor RED both meshes; `dr[k]→dr[k−1]` uniform-
BLIND/graded-RED = the Mode-5 keystone); (3) the spy-on-args reversal +
pole-continuation gates on a GRADED mesh with non-vacuity checks (the escape
hatch for un-monkeypatchable internal transforms); (4) Euclidean (not metric)
adjoint reciprocity + the unused-slot-is-zero pin; (5) constructor negative
gates for the non-carrying CONTROL (cylinder needs a level-structured
quadrature — GL is slab-only; `match=` the specific message, + a positive
build control — L4 net-new teeth). `vv §Mode-11/8 / L1 config-blindness / L2
structural-independence / L4 gate-liveness`.

## L22 — Verifying a WELDED-fold UN-WELD (a per-cell/per-level closure hand-rolled at N sites → ONE single source; `A_BA` the ψ½ Schur fold, Coupled Block Operator Step 2): the single-source routing proof is a Mode-11 wrap-COUNTER on the object the sites construct (EXACT `2·n_levels`, not `>0`), bit-identity INHERITS from the manufactured-anisotropic fold contract, the transpose must single-source too (a hand-rolled `0.5` = `coeff[0]`), and the S/F arms feed ℓ=0 ONLY so the contract MUST manufacture ℓ≥1

Rule: to gate an un-weld that extracts an inlined closure to one source, ship
SEVEN gate types — and design against the fold CONTRACT (shape-agnostic), never
a specific call surface (isolate "how the extracted source is invoked" behind a
one-line `# BIND:` helper the implementer flips; run the correctness gates
THROUGH the real consumer operators, which are shape-stable).

Why: (1) the CENTERPIECE is single-source ROUTING — a Mode-11 wrap-counter on
the SAME object the sites construct, asserting BOTH consumers (S-fwd AND F-fwd)
ENTER it; a green-but-unrouted arm = a divergent inlined copy = Mode 11. Count
EXACT `2·n_levels` (not `>0`). The seed arms `local-import` the fold INSIDE
`apply`, so `monkeypatch.setattr(source_module, fold, spy)` IS picked up. (2)
Bit-identity `np.array_equal(out, old_loop_ref)` INHERITS verification from the
fold contract — but PAIR it with a structurally-INDEPENDENT closed form
(`½·emission`, independent of the fold fn) so it pins the coefficient, not just
the routing. (3) The convenient config nulls the hardest term: the production
arms feed **ℓ=0 ONLY**, so an S/F-only gate is STRUCTURALLY blind to `P_ℓ(±1)`
for ℓ≥1 — MANUFACTURE ℓ=0+ℓ=1 (≥2G distinct-group) and assert
`Q̄(±1)=½Q₀±(3/2)Q₁`; the tooth (drop `sign^ℓ`) reds anisotropic (`3·|Q₁|`) but
the SAME mutation on an iso-only input stays 0.0 — the iso-null asymmetry IS the
necessity proof (L1/L11). (4) A TRANSPOSE hand-coded as a bare constant (the
S-adjoint's `0.5` = `coeff[0]`, bypassing the fold) needs its OWN single-source:
gate the helper-level Euclidean contract `⟨fold(m),y⟩=⟨m,fold_T(y)⟩` (tooth: a
`0.6` ℓ=0 coeff reds) AND the operator-level fwd↔adj CONSISTENCY
`⟨A_BA φ,χ̄⟩=⟨φ,A_BAᵀ χ̄⟩` (tooth: patch the FORWARD fold→0.6 → the adjoint's
still-`0.5` DISAGREES → reds). The consistency gate is BLIND to a SHARED
coefficient (both legs scale together) — it catches the fwd↔adj MISMATCH the
hand-rolled constant risks; the VALUE is pinned by the fwd contract. If the
chosen shape is factory-only (no transpose surface), those two carry the whole
contract — do NOT force a `.apply_transpose` gate; flag the binding.

How to apply: (a) refutation #4 — split configs: the fissile arm (F: `χ·νΣf·φ`)
is IDENTICALLY zero on a non-fissile mixture → a VACUOUS 0==0 gate; build a
fissile mixture + a `max|emission|>1e-6` NON-VACUITY guard before the assertion.
(b) refutation #6 — cylinder+slab (`space is None`) are the non-carrying CONTROL
(emit `None` ray + counter 0), NOT "other geometries". (c) exercise the arm on a
HAND-BUILT seed-carrying composite even when the production DRIVER is "dormant"
(the arm fires on any non-None-ray composite). (d) BIND-isolate the direct
extracted surface behind `# BIND:`; those rows are `xfail(strict=False,reason=)`
→ flip to xpass on landing (L4). `vv §Mode-11/8 / L1 config-blindness (ℓ=0
iso-null) / L2 structural-independence / anti-#3 ≥2G / anti-#4 fissile`. Full
recipe → §3 [aba-schur-fold-unweld-verification].

## L23 — Verifying a SYNTHETIC ACCELERATOR (DSA #2, pre-carve spec): the `correction→0` property PARTITIONS the failure surface, and FP-invariance is STRUCTURALLY BLIND to the majority of it — the object+rate gates are the load-bearing catchers, not supplements

The deepest test-design fact for any accelerator whose correction vanishes
at the fixed point (DSA, and the TSA/#5 sibling): the correction→0 property
makes the accelerator correctness-safe BY CONSTRUCTION for EVERY error in
{R (restriction), P (prolongation), the low-order operator `A_diff`,
correction sign/scale} — those leave the converged FP **identically**
unchanged (not sub-floor; identically, because the correction source is zero
at the FP regardless of how wrong the machinery is). This PARTITIONS the
plausible-error surface into two disjoint classes with disjoint catchers:

- **Value-changing class = corruption of the WITHIN-GROUP transport operator
  itself** (`A=L+C−S−B`): the #215 σ_r-fold (`A_wg.solve(S_residual)` with
  the σ_r-sweep = `Σ_s0·P_iso` only for isotropic flux → 46–56% wrong on
  anisotropic), a sweep sign flip. Caught by **FP-invariance** (converged
  flux ≡ plain-SI FP) — but ONLY on an ANISOTROPIC config (vv Mode 9; the
  isotropic-reflective box is the designed-green degeneracy where the fold
  is exact). This is the class FP-invariance exists for — and it is exactly
  ONE of the eight canonical errors.
- **Rate-only class = the accelerator machinery** {R, P, `A_diff`,
  correction sign, low-order D/removal, boundary rows}: FP-invariance is
  **structurally BLIND** to all of it (correction→0). Caught ONLY by
  **object gates** (R conservation `⟨1,Rr⟩=⟨1,r⟩` with NON-uniform weights +
  the anti-#8 per-cell delta-source distribution pairing; P≡Rᵀ adjoint as
  the full MATRIX not a sampled bilinear form; the low-order matrix ≡
  hand-posed stencil, element-wise, Mode-12-safe object-level) + **rate
  gates** (ρ vs 0.2247c Fourier; the c→1 × optical-thickness iteration-count
  table; reflective-BC divergence — the inconsistent-low-order operator
  diverges on reflective, is forgiving on vacuum).

The battery-completeness corollary: a plan that gates a correction→0
accelerator's correctness ONLY on FP-invariance — even a *correctly*
anisotropic FP-invariance — ships SEVEN of eight canonical errors silently.
FP-invariance and the rate gates have DISJOINT blind spots (ρ can't see the
value, the flux can't see the rate) — which is WHY both exist. Three
gate-design closures recur: (1) the **correction→0 gate's** own blind spot
is a DEAD R (δφ≡0 passes "→0" trivially) → pair with a first-iterate
NON-triviality lower bound. (2) the **no-masking** converse: mutate the
transport OPERATOR (not the machinery) → DSA must converge to the SAME wrong
answer as SI (proves it changes the RATE not the VALUE in both directions).
(3) the **#215 routing guard** is a Mode-11 wrap-counter / AST tripwire (the
within-group solve must NOT route through the σ_r-fold), catching at
design-time what FP-invariance catches numerically. Assemble-consistency
(3a) is object-level matrix diff (interior/boundary/scaling decomposition —
"≡ up to a boundary row" = interior block element-wise-equal AND the
nonzero-diff support ⊆ {0,N−1}); its independence rides a hand-posed anchor,
never the two ORPHEUS assemblies agreeing (L16 one-source proof). ρ estimator
= the RESIDUAL ratio (Signature 9: the increment under-reports as c→1);
1G is LEGITIMATE for the ρ claim (rate is flux-shape-independent — declare
the claim layer, AGENT.md §5). Anisotropic FP configs MUST build the
`Mixture` constructor DIRECTLY (`make_mixture` nulls SigL AND every A/B/C/D
mixture has Sig2=0 — lessons L1; a `make_mixture` ℓ≥1 config silently nulls
the moment DSA must leave untouched).

How to apply: for any correction→0 accelerator, FIRST draw the value/rate
partition table (§0 of the spec) — for each of the 8 canonical errors name
the gate that reds and confirm FP-invariance is blind. The object+rate gates
are LOAD-BEARING (the only catchers for 7/8), not supplements; a
FP-invariance-only plan is Mode-9-complete for the σ_r-fold and Mode-9-EMPTY
for the machinery. Sharpens vv Mode 9 (the config degeneracy) with the
gate-completeness corollary. Full spec → `.claude/plans/archive/dsa_verification_spec.md`.
`vv §Mode-9/12 / anti-#8 / Signature-9 / bit-identity`.

---

## L24 — Verifying a THREE-DOF SEPARATION campaign (operator ∥ splitting ∥ realization): MEASURE the proposed acceptance criterion BEFORE the plan is anchored on it — a signature-invariance AC is permanently GREEN, and the real red is the WELD (one record carrying two stages, its stale half silently re-derived downstream)

A campaign that separates degrees of freedom the code conflates arrives
with an acceptance criterion of the shape *"changing X must not touch Y"*.
That shape is a trap: it is usually **already true by SIGNATURE**, so the
gate is unfalsifiable from the first commit. The highest-value output of a
proactive dispatch is REFUTING it with a measurement (L10), then supplying
the criterion that IS red.

- **Run the AC as a probe before writing any gate.** Measured on `main`
  @ `b0a003b4`: `build_within_group_system(sn_mesh, mat_xs, *,
  scattering_op, scattering_order)` takes **no** strategy parameter ⟹ the
  posed `A` is strategy-invariant by construction; and `A x == (M−ΣN) x`
  reads **EXACTLY 0.0** (2-D Cartesian seedless, both schedules) /
  **3.5e-17** (sphere carrying — the coupled grid re-associates, so
  `array_equal` is the WRONG contract there from day one). Falsifiers
  (drop `B` → 5.15e-2; sign-flip `N` → 8.57e-3) prove the probe
  non-vacuous — it is simply GREEN. → the new **vv Mode-8 SIGNATURE-
  tautological** sub-case: a falsifier check PASSES on such a gate, so
  "does it red?" gives FALSE confidence; the honest gate is on the
  SIGNATURE (adding the knob must red it), and the value row demotes to a
  regression floor.
- **The red is the WELD, and it is findable by object IDENTITY, not
  value.** `WithinGroupSystem` carries stage-2 (`loss`) AND stage-3
  (`implicit_operator`/`explicit_gains`) in one record, so there is no
  boundary at which "strategy may enter" is assertable — and because there
  is none, a SECOND splitting grew beside the first: measured
  `_within_group_si(..., inner_schedule='gauss_seidel')` returns
  `base is record.implicit_operator` → **False** (driver runs
  `ScheduledInvertibleOperator`+`SNMaskedBoundaryOperator`; the record
  still advertises `StreamingCollisionOperator`+`SNBoundaryOperator`),
  while `jacobi` → **True**. The `is`-identity gate across the strategy
  ladder catches this; no value gate can (both splittings reconstruct the
  same `A` — that is what makes them both *valid* and the twin *silent*).
  Ship the green arm as the CONTROL leg so the asymmetry is pinned.
- **`A − M` may not even be SPELLABLE.** Measured:
  `IncompatibleOperatorComposition — OperatorSum requires equal domains;
  got CoupledSpace('coupled(full_field)', shape=(128,)) and
  FullFieldSpace('full_field', shape=(128,))`. Same shape, different NAME
  ⟹ `FunctionSpace.__eq__` refuses. So the splitting identity can only be
  checked by APPLYING to a state with a manual carrier unwrap — a
  structural red worth naming in the plan, because it is invisible to
  every value probe.
- **Monomorphism reds are cheap to measure and belong in the plan, not
  the implementation.** `F.domain is None` on the production
  `from_solver_data` build (every other leaf reports a concrete space);
  `F.H` then BUILDS and silently returns a bare Euclidean transpose
  (`_AdjointOperator.apply` applies the metric only when both inner spaces
  are non-`None`). Also: the carrier guard is NON-UNIFORM — one leaf
  raises a typed `TypeError` with a remediation message, another a raw
  `AttributeError`. Gate the CONTRACT (one carrier accepted, all others
  refused with a typed message), never the dispatch SPELLING when a
  corpus ruling parks it (#261) — the plan must respect the parked
  ordering or it will be rejected on process, not on math.
- **The rate DOF needs its own closed form, and the degenerate regime
  kills the RATE gate too.** ρ(M⁻¹N) is the one constraint structure
  cannot supply. Its anchor is hand algebra on the spatially-flat /
  angularly-anisotropic mode (`L=0` ⟹ `ρ = c` standard SI, `ρ = c/(1−c)`
  for the σ_r fold, diverging at `c ≥ ½`); an ISOTROPIC power-iteration
  seed makes `N ≡ 0` and ρ reads **identically 0** — ship that as a
  permanent CONTROL leg, not a one-off mutation. 1G is LEGITIMATE for a ρ
  claim (declare the claim layer). The measurand already exists and is
  read by nothing: `SourceIteration.contraction_ratios`
  (`numerics/iteration.py:742`) — cross-check the operator-side power
  iteration against it (`rtol=5e-2`); disagreement means the driver is not
  running the advertised splitting (the weld regression, above).
- **A composition-over-fusion carve owes a PERF gate whose catcher is
  STRUCTURAL, not wall-clock.** `A_ij = Rᵢ A Jⱼ` is the exact shape that
  cost 10–20× on slab / ~6× on the suite with every correctness gate green.
  Wall-clock is machine-dependent and CI-flaky (normalise against an
  in-process same-FLOP calibration if used at all); the deterministic
  catcher is a **leaf-kernel call-COUNT that must not scale with `n_cells`**
  (refine nx 20→160, assert the count is unchanged) plus a `tracemalloc`
  peak-per-DOF ceiling. Capture the baseline on the commit BEFORE the
  composition phase — a perf gate baselined after the regression is worthless
  — and hard-block that phase's merge on it.

How to apply: for any N-DOF separation campaign, (1) EXECUTE the proposed
AC as a probe and report GREEN loudly if it is — a plan anchored on a false
red ships unverifiable phases; (2) hunt the WELD (one record spanning two
stages) and gate it by object `is`-identity across the strategy ladder,
with the already-green arm as the control; (3) enumerate the *unspellable*
states (cross-carrier composition, absent pencil type) separately from the
red ones — unspellable is worse than red and needs a phase-ordering
constraint, not a tolerance; (4) order the phases so no gate is unwritable
at its phase's merge (write the red gate BEFORE the fix that flips it);
(5) the rate DOF gets a closed form + a degenerate-seed control; (6) the
composition DOF gets a call-count perf gate baselined pre-carve. Plan of
record → `.claude/plans/campaign_verification_plan.md`. Sharpens vv Mode 8
(signature-tautological) + Mode 9 (rate-blind degeneracy); mirrors L10
(refute the premise) and L16 (one-source proof / object-not-scalar gates).

---

## L26 — A metric-adjoint (`.H`) reciprocity gate is blind exactly when `[G, A] = 0`: the decidable criterion is the COMMUTATOR, not "uniform h" — and a leaf whose action commutes with the metric ALGEBRAICALLY needs a second, metric-agnostic mutation or its row is a dead gate

The received G1.4-style recipe — *"make the metric non-degenerate and
non-uniform (non-uniform `h`, curvilinear so `V_cell` spans an order of
magnitude); ship the uniform-`h` leg as the config-blindness control"* —
is **two-thirds wrong, and both errors are measurable in minutes.** The
mutation is the ERR-067/Mode-12 one: drop the metric inside
`_AdjointOperator.apply` so `.H` degrades to the bare Euclidean
transpose. Its residual is the **commutator** `[G, A]`: with diagonal
`G`, `⟨Ax,y⟩_G = yᵀGAx` and `⟨x,Aᵀy⟩_G = yᵀAGx`, so the mutation is
**identically silent iff `GA = AG`**. Three consequences, all MEASURED
on the SN leaf set (P0, `main` @ `b0a003b4`):

- **"Uniform `h`" is NOT the blind config.** A uniform-`h` slab under
  `gauss_legendre(4)` still REDs at 1.3e-1 (`L`) / 4.0e-1 (`S`) / 2.7e-1
  (`F`) — because `G = V_cell·w_n` and the *quadrature weights* vary even
  when `V_cell` does not. Shipping it as the "blind control" would assert
  a falsehood and the control would RED. The honest blind config is a
  **globally CONSTANT** metric: `G = c·𝟙 ⟹ G⁻¹AᵀG = Aᵀ` exactly. Build it
  deliberately — Cartesian slab (constant `V`) × `gauss_legendre(2)` (the
  only SN rule with equal weights, both exactly 1) × `h = 1/√3` so the
  bulk constant `V·w` EQUALS the trace constant `|Ω·n|·w = 1/√3` across
  the bulk⊕trace composite. MEASURED: one unique metric entry, every leaf
  blind at ≤ 4.3e-16 (vs 1.40/1.01/1.46 on the sphere). ALWAYS pin the
  constancy as a precondition (`np.unique(...).size == 1`) or the leg
  silently stops being the blindness proof — verified reddenable by
  swapping the fixture for the sphere.
- **The CURVILINEAR lever only constrains the leaves the metric fails to
  commute with, and the quadrature FAMILY decides which.** A
  `level_symmetric` rule has **constant weights** (MEASURED: one unique
  value), so on a non-uniform sphere `G = V(space) ⊗ c` — which commutes
  with every SPACE-DIAGONAL, angle-mixing operator. Result: `S` and `F`
  measure **exactly 0.0** under the mutation; only `L` (bulk↔trace
  coupled) reds. Swap to `gauss_legendre(4)` and `L`/`S`/`F` all red at
  O(1). So "curvilinear + non-uniform `h`" is NOT sufficient — the metric
  must vary along **both** the spatial and the angular axis. Check the
  weight vector's unique count before trusting the config.
- **Some leaves are metric-blind by ALGEBRA, in every reachable config —
  say so, and close them with a DIFFERENT mutation.** A
  `MultiplicationOperator` (`C`) is diagonal, so `Cᵀ = C` and `G⁻¹CG = C`
  for ANY diagonal `G`: no configuration exists. A specular boundary
  (`B`) is a signed PERMUTATION mapping `μ→−μ`, preserving both `|Ω·n|`
  and `w_n`, so it commutes with the trace metric; the only
  non-permutation face laws (white/albedo) are **refused by `SNMesh`**
  (MEASURED: `ValueError`, only `reflective`/`vacuum` accepted), so there
  is no reachable config either. Those rows are Mode-10
  *exercised-but-unconstrained*. Do NOT assert their silence (it
  calcifies a fact a legitimate change could move) and do NOT claim teeth
  they lack. Close them with a **second, metric-agnostic mutation** —
  scale the adjoint by 2, which reds EVERY leaf at exactly 0.5 relative —
  so each row still provably pins `apply_transpose`'s structure, and
  point at the gate that DOES pin their metric (for `B`: the existing
  L11 drop-`|Ω·n|`-from-the-REFERENCE control in
  `tests/sn/operators/test_g_adjoint_reciprocity.py`).

**Two companion traps caught in the same file, both worth the habit:**

- **The xfail-strict FALSE POSITIVE bit on first write (L4, live).** The
  G1.5 row "an anonymous leaf must refuse construction" is written as
  *construct → if it succeeded, demonstrate the degradation → `pytest.fail`*.
  The demonstration (`.H` vs `apply_transpose` on a meshless `(ng,1)`
  probe) blew up inside `np.einsum` for `ScatteringOperator` — so that
  row "xfailed" while asserting **nothing** about the claim. Fix: the
  demonstration is BEST-EFFORT, wrapped, and its outcome (including its
  own failure) is reported as **evidence text inside the `pytest.fail`
  message, never as the verdict**. Rule: in a strict-xfail body, there
  must be exactly ONE statement that can fail, and it must be the
  documented one — run `--runxfail` and READ every message before
  crediting the gate.
- **Prove the XPASS flip, don't assume it.** Simulate the landing phase
  with a throwaway pytest **plugin** (`-p mysim` on `PYTHONPATH`) that
  monkeypatches the production surface the phase will change
  (`from_solver_data` raises without a space; `_apply_faces` raises a
  typed `TypeError`) — the strict xfails then turn into
  `[XPASS(strict)] …` hard failures, which is the proof the marker
  forces acknowledgment. Cheap, in-process, and NEVER a `git checkout`
  (this tree carries uncommitted-by-policy skill files with no history).

Also durable from the same pass: **a "reuse the existing fixture" brief is
a hypothesis, not an instruction.** The neighbouring SN operator fixtures
carry `placeholder_materials` (SigS/χ/νΣf all zero) — MEASURED: `F` is
then the ZERO operator and its reciprocity row is the tautology `0 == 0`
(the non-vacuity guard reds with `|⟨Ax,y⟩_G| = 0.0` on all four
geometries). Build the fissile `Mixture` DIRECTLY (L1: `make_mixture`
nulls `SigL` AND `Sig2` and has no P1 channel), and state in the file
WHY each neighbour was rejected — with the measurement.

How to apply: for any `.H`/adjoint/metric gate, (1) write `G` explicitly
and ask *which leaves commute with it* BEFORE choosing the fixture —
that, not "non-uniform mesh", is the config criterion; (2) partition the
leaves into metric-CONSTRAINED and metric-INVARIANT by running the drop
mutation, and give the invariant ones a metric-agnostic non-vacuity
mutation; (3) build the globally-constant-metric leg deliberately and
guard its constancy; (4) `--runxfail` every strict xfail and read the
message; (5) prove the flip with a plugin-based landing simulation.
Sharpens vv Mode 12 (the ERR-067 metric-repair closure) with the
commutator criterion, and vv Mode 10 (the third state:
exercised-but-unconstrained *by algebra*, where no isolating regime can
exist). Gate file → `tests/sn/architecture/test_monomorphic_leaves.py`.

---

## L25 — Building the call-count PERF gate (P0 of the three-DOF campaign, EXECUTED): "the count must not scale with `n_cells`" is TOO COARSE — the invariant is PER-AXIS, the regression it catches is EXACTLY value-identical, and a perf baseline is a (number, FIXTURE) pair

L24 §6 specified the composition-cost gate as "a leaf-kernel call-COUNT
that must not scale with `n_cells` (refine nx 20→160, assert the count is
unchanged)". Building it against the tree corrected that in four ways.
Every number below is MEASURED (branch `refactor/operator-strategy-layers`,
`orpheus/` @ `360a8087` ≡ `361471ec`, host `.venv` Py 3.14.3, `python -O`).

**1. The spec's leaf name was STALE — prove the wrap fires BEFORE asserting.**
`DiamondDifference.update_batch` does not exist (renamed at S6.4(e) to
`cell_kernel_batch` / `residual_kernel_batch`). The genuine apply-direction
leaf on `A.apply(x)` is **`residual_kernel_batch`**. The non-vacuity proof
is two-part and the SECOND part is what makes it a proof: (a) the counted
method fires (>0) on a real apply on every geometry; (b) the plausible
SIBLINGS (`cell_kernel_batch`, per-cell `residual`, per-cell `update`)
stay at **exactly 0** through the SAME harness — which simultaneously
demonstrates the harness reports 0 for an off-path name, i.e. that the
vacuity branch is real (vv Mode 11). Never assert a count before both.

**2. The invariant is PER-AXIS. Measure which axes are hoisted FIRST.**
MEASURED arity of ONE `A.apply(x)`:

| path | count | scales with |
| --- | --- | --- |
| 2-D Cartesian `ScanMarch` (production default) | `8 · ny` = 80/80/80/80 at nx=8/16/32/64 | **ny only** — invariant in nx, in `sn_order` (S2/S4/S6 → 80/80/80 at N=8/24/48) and in ng (1/2/4 → 80/80/80) |
| 1-D slab `CumprodScan` **apply** | `2 · nx` = 40/80/160/320 at nx=20/40/80/160 | **n_cells — already a per-cell Python fold** |
| 1-D slab `CumprodScan` **solve** (`ordinate_scan`) | `2` at every nx | nothing — fully hoisted |

So a blanket "count must not scale with `n_cells`" would have been FALSE
on `main` for 1-D and would have shipped either a permanently-red gate or
(worse) a quietly-weakened one. The honest gate is four legs:
(a) strict `==` on the **scanned** axis (the catcher); (b) calls/cell must
DECAY under isotropic refinement (1.00→0.50→0.25 MEASURED — admits the
honest per-row linearity of the marched axis while still forbidding the
per-cell regime); (c) an **exact-arity regression FLOOR** on the
already-folded path (pin `2`, red on BOTH sides — an improvement must
re-baseline consciously, L7); (d) the hoisted sibling as a **CONTROL leg**
proving the instrument can see invariance, so leg (c)'s linearity is a
property of the code and not of the harness.

**3. The regression class is EXACTLY value-identical — that IS the argument.**
Re-dispatching the batched kernel once per column of its level slice
(the faithful L16 fold) gives `np.array_equal` **True**, `max|diff|`
**0.0e+00** — the DD residual kernel is cell-local, so the split changes
the arity and nothing else. **No value gate, at any tolerance, against any
reference, can ever see this.** State that measurement in the module; it is
what demotes wall-clock to corroboration and promotes the count to catcher.
Corollary for the mutation battery: **check the batch SHAPE the leaf
receives before choosing the fold axis.** A cell-axis fold is INERT in 1-D
(`psi_bar` is already `(4, 2, 1)`; 2-D gets `(3, 2, nx)`) — mutate the
ordinate axis there instead (`{20:40,…}` → `{20:200,…}` → RED). A mutation
that fails to red because it is inapplicable is NOT "the gate has no teeth".

**4. A perf baseline is a (number, FIXTURE) PAIR — own the fixture, gate it.**
Do NOT source a perf fixture from the package's shared `_config`: the
correctness gates are free to retune it (a Mode-9 fix bumping S4→S6, a
third region) and every committed constant silently starts describing a
different problem. Import the shared *conventions* (the direct `Mixture`
builder, the one posing site, the carrier-generic probe state), own the
*sizes*, and ship a **fixture-fingerprint gate** (`n_cells`, `n_ordinates`,
`n_groups`, `n_regions`, `n_dof`) whose failure message says RE-CAPTURE.
Read `n_regions` off `material_xs_field().cells_by_material` (typed,
dimension-agnostic, and it counts materials as the OPERATORS resolve them)
— never off `mesh.mat_map`, which is 2-D-only and `Optional`.

**Tolerance discipline, split by contention-immunity.** MEASURED: P-2
(wall clock, ratio to an in-process dense contraction of the same nominal
FLOP count) = 22.45–24.41 across runs → the constant is PROVISIONAL on a
contended host; set it at ~3× and say so. P-3 (`tracemalloc` peak /
`n_dof·8`) = 3.0299–3.0328, spread **0.03 %** → contention-IMMUNE, so
TIGHTEN it rather than padding it: MEASURED `+1/+2/+3` held full-field
temporaries → `4.035 / 5.033 / 6.033`, so `4.0` catches the FIRST extra
field where the plan's suggested `6.0` needs three. Note a PEAK metric is
blind to transients below the existing peak (a 0.6 MB per-call scratch
moved the reading by 0.0006); the mutation that proves its teeth must
HOLD the temporary, which is also the mechanism the gate exists to catch.

How to apply: for any composition-over-fusion carve, (1) find the real
batched leaf by wrapping candidates and reading the counts — never trust a
spec's illustrative method name; (2) tabulate the arity against EVERY axis
(cell, ordinate, group, and each spatial axis separately) before writing a
single assertion; (3) prove the fold mutation is value-identical and put
that number in the docstring; (4) split the legs by contention-immunity —
pad the wall clock, tighten the allocation; (5) fingerprint the fixture.
Delivered: `tests/sn/architecture/test_composition_cost.py` (9 gates,
0.9 s, pyright-clean). Refines L24 §6; sharpens L16 (the perf-regression
precedent) and L4 (prove every gate's teeth).

---

## L28 — REPAIRING blind gates: re-pose onto a regime-INDEPENDENT mechanism, and measure "before" with `git show`, never `git checkout`

Context: boundary-machinery review phase B0.3 (2026-07-30), seven measured
coverage defects in a suite that was already strong (30/31 mutations caught).
Repairing blind gates is a different design problem from writing them, and the
four rules below are what the job actually turned on. Result: **31/31** on the
auditor's own set, plus the historical miss closed.

**1. A DECAYED `catches` marker is repaired by moving to a REGIME-INDEPENDENT
mechanism invariant — NOT by driving the fixture back into the regime.** The
reflex is "the config converges in 6 outers but the bug needs 30–60, so tighten
the tolerance". MEASURED: that is impossible here — with ERR-052 re-introduced
the un-normalised iterate stabilises at `|φ|max = 5.6552e-01` at 6, 24 AND 32
outers, because `F·φ/k` is scale-neutral once `k` converges, so the catalogued
denormal catastrophe is **unreachable in its own fixture at any depth**. The
repair is to assert the *output convention the fix establishes* — here
`P(φ) = ∫νΣ_f φ dV = 1`, which holds at EVERY outer count, so the marker cannot
decay again. Mutation: 1.0 → 0.0739 (13.5×) on the subcritical leg, 1.0 →
0.84375 on the supercritical one; `rtol=1e-12`, no margin. **Always check
whether the catalogued mechanism is reachable before trying to reach it** — and
record the answer, because the next reader will otherwise chase it.
**Corollary:** the mechanism reference must NOT be computed by the routine that
ESTABLISHES it (`compute_production_rate` is what the normaliser divides by) —
hand-assemble it from the mixture + mesh + solution, or the gate is a tautology.

**2. A Mode-12-blind "hand-computed" test needs BOTH halves: break the
invariance AND source the reference independently.** Breaking the invariance
alone (non-constant input) leaves you re-transcribing the production weight
formula — which is what the *existing* catcher already did (vv L11 procedural,
not structural). Type the PUBLISHED table (Abramowitz & Stegun GL nodes/weights)
as literals, assert the quadrature matches it as a precondition, and build the
expected number from the literals. Bonus available cheaply: the closed-form
continuum anchor (`∫|Ω·n̂|μ dΩ / ∫|Ω·n̂| dΩ = 2/3`) at an honestly-stated
discretization tolerance — the audit had flagged its absence.

**3. A tautological-`raises` file is repaired guard-by-guard with the guard
named at `file:line` in each docstring, and the acceptance measurement is the
PER-GUARD red table.** 0/14 → 12/14. Every residual miss must be
*categorically* out of scope and said so (the 2 here are composition-walker
mutations, not error guards) — an unexplained residual is indistinguishable
from an incomplete repair. Two production defects fell out of the exercise that
no amount of reading had surfaced: an error class with **zero** raise sites
(its trigger is a no-op ABC default nobody overrides) and a one-site `law=` tag
convention drift. **Pointing tests at production entry points is itself an
audit of production.** When a drift is found and you may not fix production,
assert the *property* (case-insensitively) rather than the drifted *value* —
pins the contract without calcifying the bug.

**4. Measure "before" with `git show HEAD:<file> > <tmp copy beside it>`.** The
tree carried uncommitted work in `.claude/` and `orpheus/`, so `git checkout` /
`restore` / `stash` were forbidden. Writing the pre-repair content to a
throwaway sibling file runs it under the SAME mutation, in the SAME collection,
against the SAME fixtures — a true A/B, then `rm`. This is how every
before/after row in the report was obtained (phantom re-confirmed as `11
passed`; sentinels re-confirmed as `3 SKIPPED`; registry probe RED).

**5. HYGIENE — for the "committed" side of any doc/generated-artifact diff, use
`git show HEAD:<path>`, never the working-tree file.** A concurrent Sphinx
build regenerated `docs/theory/verification/matrix.rst` mid-session (its
`builder-inited` hook), between two of my edits. Diffing a regeneration against
the already-regenerated working-tree copy manufactured a confident,
circumstantial, and entirely WRONG "the committed V&V matrix has always
disagreed with the code" finding, complete with a git-log narrative. Caught
only by checking `git show HEAD:` and the file mtimes. **On a shared tree,
`ls -la`/mtime is part of the measurement.** Retract loudly when it happens —
a wrong finding costs more than no finding.

**6. Never let a repair reduce the catch rate — prove it, don't assume it.**
Re-run the auditor's own harness rather than a re-implementation
(`$CLAUDE_JOB_DIR/tmp/mutplugin{,2,3,4}.py` survived the session), so the
before/after is apples-to-apples. The per-mutation red COUNTS are the real
signal: `white_nocos` 11→13 and exactly +1 on each of the twelve error guards
proved the repairs landed where designed, and every other count being unchanged
proved nothing was disturbed.

---

## L29 — Verifying a DOMAIN NARROWING (an operator's domain shrinks from a superset to the honest half; SN boundary B3): the old gate does NOT go tautological, it stops RUNNING — and the narrowing's own new index remap is activated only in a regime the existing gate does not use

The campaign shape: production hands a realized law the WHOLE slot and discards
the unwanted rows on the way out (`full = law.apply(face_in); out[sel] =
full[sel]`); the honest physics is `Γ₊ → Γ₋`, so the carve narrows the law's
domain AND codomain. A sibling arm (diffusion) already does it right, so the
carve is a domain narrowing against a worked in-repo precedent. Five reusable
test-design facts, all MEASURED (boundary review §B3, 2026-07-31):

- **The "will this gate go tautological?" question usually has a THIRD answer:
  the gate BREAKS.** The reflex worry (`vv` Mode-8 signature-tautological) is
  that "the discarded rows are zero" becomes unfalsifiable once the codomain no
  longer contains them. MEASURED: it stays reddenable (a pass-through mutation
  reds it with its original message) — but the gate never gets there, because
  its *reference expression* (`bc.apply(whole_slot)`) feeds the narrowed
  operator the wrong shape. **Always run the gate under a full-fidelity
  simulation of the carve** (patch the REALIZER, not only the call site) — a
  call-site-only simulation leaves the reference working and hides the break.
  Diagnostic value: `realizer` mode reds 16 items, `narrow` mode reds 0.
- **The gate's TEETH change even when the assertion survives.** After the
  narrowing the original bug (the law's discarded image leaking) is
  *unspellable* — strictly better than red (L24) — and what remains reddenable
  is only the WRITE-TARGET family (wrong target / extra write). So the honest
  re-pose is on the mechanism: `got[complement(codomain_rows)] == 0` plus a
  SHAPE gate on the realized operator (`apply` maps `(n_dom,…)→(n_cod,…)`) plus
  a `pytest.raises` NEGATIVE for the domain guard. Ship the shape gates as the
  post-carve contract: they are RED pre-carve and GREEN post-carve, which is
  the flip-proof the carve is measuring the right thing.
- **A three-way index split hides in a two-way vocabulary.** The face slot is
  `inflow ⊔ outflow ⊔ TANGENT` (`|Ω·n| ≤ ε` belongs to neither) — 4 of 8
  ordinates on the product-quadrature fixture. The shipped gate asserted
  `got[outflow] == 0` and was MEASURED BLIND to a tangent leak that the
  widened `got[complement(inflow)] == 0` catches. Whenever a gate says "the
  other rows are zero", check whether "other" is really the complement.
- **The narrowing can introduce a NEW index remap whose activating regime the
  existing gate does not use — and the SAME remap appears at ≥2 sites whose
  discriminating fixtures are COMPLEMENTARY.** `out[sel] = full[sel]` needs no
  remap; after narrowing, the image is indexed by position WITHIN the codomain,
  so both (a) the row subset of a schedule split and (b) the law's own table
  need `searchsorted(codomain_rows, ·)`. MEASURED at landing, one mutation per
  site over the whole boundary suite: the **split** remap's naive
  `arange(sel.size)` is EXACTLY correct in 1-D and reds exactly ONE gate (the
  new 2-D one) — the sphere-based split gate stays green; the **law's** remap
  is wrong on the SLAB (the mirror reverses order: `perm[inflow] = [3,2]` →
  local `[1,0]`) and reds 16 items, but the CYLINDER and the 2-D
  level-symmetric fixtures stay green (there `perm[inflow]` is already the
  codomain set in order). So 1-D covers one site and 2-D the other, and
  **neither fixture covers both**. Ship both gates, each with an ACTIVATION
  guard asserting its fixture still discriminates (`local != arange`), or one
  silently decays into the other's blind spot (Mode-8 class 7). **Before the
  carve, enumerate the arithmetic the narrowing ADDS, then find a
  discriminating fixture PER SITE** (§0.6, one level in from config-blindness:
  the blind axis is the *dimension*, and it points opposite ways at the two
  sites).
- **The re-posed WIRING gate is structurally blind to the law's MATH — say so
  and name the other catcher.** After the re-pose, the composite gate's
  reference is `law.apply(γ₊ψ)`, i.e. the SUT's own law object; it therefore
  pins which domain / which rows and takes the law's table as given. MEASURED:
  collapsing the law's permutation to the identity, or rolling it by one,
  reddens NOTHING in the re-posed composite file while changing production's
  output hash. The catcher is the bit-identity gate whose reference is the
  RETIRED expression materialised in numpy from the law DESCRIPTOR (there:
  6 and 10 items red). Two gate families, two threat models — record the split
  in the file's docstring or a future reader reads the wiring gate's green
  under a math mutation as coverage.
- **Bit-identity here is `array_equal`, and a SECOND, independently-written
  narrowed implementation is the strongest evidence.** The narrowing removes
  rows from a gather — no reduction tree changes — so any nulp tolerance would
  hide the bug class. Prove it twice: once with a wrapper
  (`full.apply(embed(γ₊))[cod]`) and once with a genuinely reduced kernel (a
  reduced permutation). sha256-identical on both, forward AND transpose, over
  every reachable fixture. Also measure the laws that DON'T narrow neutrally
  (SN's albedo/periodic realize as full-face identities → their narrowed action
  is a DECISION, not a refactor) and ship those as `xfail(strict=True)` so the
  decision cannot be forgotten when a later phase admits them.
- **The "which laws are still un-narrowed?" gate needs an anti-Mode-12 leg, and
  the answer is usually BIGGER than the plan says.** A shape assertion
  (`apply(|dom|) → |cod|`) CANNOT distinguish `Γ₊→Γ₋` from `Γ₊→Γ₊` whenever
  `|Γ₊| == |Γ₋|` — MEASURED true for EVERY quadrature × face pair in the tree,
  so the error class sits inside the shape functional's invariance group. The
  discriminator that escapes it: feed the law a FULL-face probe and require it
  NOT to emit `N` rows (a narrowed law structurally cannot; an endomorphism
  always does). That single leg is what caught the laws the plan had NOT listed
  — the brief predicted "3 un-narrowed laws"; measurement found **6 rows across
  4 kinds**, because the α=0 / α=1 *fast paths* (a bare `ZeroOperator` /
  `IdentityOperator`) are endomorphisms too, and a rank-0 source law raised on
  the narrowed domain. Enumerate from the REGISTRY and add a
  completeness test (`covered == set(registry)`) so a law added later cannot
  escape the gate.
- **A narrowed law cannot compose with an un-narrowed sibling — budget for the
  mixed-composition gates going xfail, and RE-POSE their probe first.** Every
  `α·narrowed + β·full_face` gate breaks the moment the first law narrows.
  Marking them `xfail(strict=True)` is right, but a probe left on the OLD
  domain fails at the *reference* line with a broadcast error, not at the
  documented composition refusal — a misattributed xfail (Mode-8 class 4) that
  will XPASS for the wrong reason. Move the probe onto the narrowed domain
  FIRST, then `--runxfail` and read the message.
- **Prove the strict-xfail set FLIPS, and watch for a Mode-8 class-4 trap
  inside that very proof.** A landing-simulation plugin that narrows ONE law
  must turn exactly that law's rows `XPASS(strict)` and leave the others
  `XFAIL`. First attempt simulated the α=0 arm as `ScaledOperator(0.0, …)`; the
  constructor REFUSES a zero scalar ("degenerate; use ZeroOperator
  explicitly"), the xfail swallowed the `ValueError`, and the row read as "did
  not flip". The simulation must reproduce the shape production would actually
  emit — otherwise the class-4 defence is itself class-4.
- **Method note that paid for itself twice.** (a) The reduced operator does NOT
  validate its domain — `PermutationOperator(reduced_perm).apply(full_slot)`
  silently truncates and returns wrong numbers (MEASURED), so "hand it the full
  slot" stays spellable unless the carve adds a guard; the negative test for
  that guard is a deliverable, not a nicety. (b) A first tangent-leak mutation
  was VACUOUS (a permutation's image on the tangent rows is structurally zero
  once the domain is `Γ₊`) — caught by the sha256 positive-control table before
  any colour was read. Keep a per-variant hash table; it is cheap and it is the
  only thing standing between you and a confident false "no gate catches this".

---

## L30 — MIGRATING the inherited test surface of a domain narrowing (B3.4a, the phase AFTER L29's carve): a rectangular operator's self-adjointness gate is not lost, it is RECIPROCITY against a sibling the tree already builds; and the guard the narrowing dissolves is replaced by a guard on the CLASSIFIER, whose discriminating fixture is usually ONE quadrature

L29 is the carve; this is the test-migration phase that follows it — the 28
inherited reds of the white/prescribed-inflow narrowing (`AngularAverageOperator`
+ `IncomingSourceOperator` typed `Γ₊ → Γ₋`). Every finding below is MEASURED
(2026-08-01, `scratch/b3_4a_test_migration.md`, harness `scratch/b3_4a_mutations.py`).

- **A "cannot be posed on the narrowed operator" gate is usually a RECIPROCITY
  gate in disguise — look for the mirror object the tree already builds before
  reaching for `xfail`.** The brief pre-authorized a strict xfail for
  `⟨Ax,y⟩_w == ⟨x,Ay⟩_w` on a now-rectangular `A`. It was NOT needed. Read WHY
  the old endomorphic version held: `A = 1_N ⊗ (cos_w/norm)` with the SAME
  masked `cos_w` on both the broadcast and the contraction, so `W·A` was
  symmetric. Narrowed, the weighted adjoint `A* = W₊⁻¹AᵀW₋ = 1_{Γ₊} ⊗
  (cos_w₋/norm₊)` is *itself a Lambertian average*, i.e. **the SAME production
  constructor at the OPPOSITE face** (`from_quadrature(q, axis, -sign)`) —
  because that face's `Γ₊` IS this face's `Γ₋`. So the gate becomes
  `⟨Aψ,φ⟩_{W₋} == ⟨ψ,A*φ⟩_{W₊}` with BOTH sides production objects and nothing
  hand-built. It rides on `Σ_{Γ₊} w|Ω·n| == Σ_{Γ₋} w|Ω·n|`, which is
  BIT-IDENTICAL on every quadrature × face in the tree — assert it INLINE as an
  activation guard (a future asymmetric rule must red HERE, loudly, not be
  absorbed into a tolerance). Teeth are REAL and distinct from conservation: a
  face-sign-dependent `norm` mutation reds both reciprocity rows while the
  `+1`-face conservation rows stay green.
- **When a narrowing retires a private CLASSIFIER, the replacement gate is on
  the classifier, and exactly ONE fixture in the tree discriminates it.** The
  operator used to carry its own `(outward_sign*mu) > 0.0` outflow test, a twin
  of the trace space's `> TANGENTIAL_EPS`. MEASURED across 8 quadrature × face
  pairs, the two sets agree on SEVEN — `product(n_mu=2, n_phi=4)` on **xmax**
  is the only disagreement (|Γ₊|=2 vs strict-set 4; 4 tangential ordinates),
  and the SAME quadrature on **ymin** agrees. So the gate must name that exact
  fixture AND carry an activation guard (`assert not array_equal(gamma_plus,
  retired_strict)`), or it silently decays into a restatement of its siblings.
  Mutation W1 (restore `> 0.0`) reds that ONE test and nothing else in 110.
- **`|Γ₊| == |Γ₋|` on EVERY reachable face makes the codomain claim
  Mode-12 blind — and part of it is IRREDUCIBLE.** Sizing the codomain from
  `Γ₊` inside `from_quadrature` reddened **0 of 110** tests and CANNOT be
  caught: no quadrature × face pair in the tree has unequal half-traces. The
  *operator-level* twin (an `apply` that broadcasts over the INPUT's leading
  axis) IS catchable — but only via a HAND-CONSTRUCTED operator with unequal
  sizes (`n_outflow=3, n_inflow=5`), never from `from_quadrature`. So: for any
  narrowed operator, ship one hand-built unequal-size row as the anti-Mode-12
  leg, and report the constructor-level half as an honest residual blind spot
  (closing it is a TYPING job, not a test job).
- **A `catches(ERR-NNN)` on a re-posed leg silently decays through the SAME
  Mode-12 hole — re-run the mutation, don't reason about it.** The ERR-047
  delivered-`q` leg, re-posed to "the output has exactly |Γ₋| rows", was fed a
  `Γ₊`-sized probe and stayed GREEN under the exact catalogued bug (`q` sized
  from its input). Adding a **FULL-FACE probe** leg (N rows in ⟹ |Γ₋| rows out)
  restored the red. The `catches` marker was a phantom for one edit cycle and
  only the mutation run showed it.
- **A guard the narrowing NEWLY reaches ships with no negative test — write
  it, and write the positive control with it.** Three guards were net-new at
  this phase (white's `_outflow_restriction`, white's orientation cross-check,
  prescribed-inflow's `_outflow_restriction`); each got a `pytest.raises` with
  a SPECIFIC `match=` plus, for the orientation guard, a four-face positive
  control — without which a realize that raised for ANY reason satisfies the
  negatives (L4's attributability rule).
- **The orientation cross-check FINDS the mismatches the old silence hid —
  expect them in the FIXTURES, and check the production tag-parser too.**
  MEASURED: a sibling file built `face="xmin"` while its law declared
  `axis="x", outward_sign=+1` (= `xmax`) — pre-narrowing it averaged the
  installation face's INFLOW hemisphere and reported nothing. And
  `_law_from_tag` builds every parameter-free law as `law_cls()`, so
  `WhiteBoundary` always gets the default `xmax` orientation whatever face is
  being resolved (reflective correctly threads `AXIS_NAMES[label.axis_index]`)
  — LATENT only because the method's admission registry does not yet list
  `white`. When a law carries its own copy of a datum the installation site
  also names, grep the tag-parser for the parameter-free fallback.
- **Method: the migration's own regression proof is the DELTA, not the green.**
  Run the wide suite before and after: `40 failed` → `12 failed` on
  `tests/{geometry,sn/operators}`, and `12 failed, 4225 passed` on
  `tests/{sn,geometry,numerics,transport} -m "not slow"` — the same 12
  out-of-scope rows, so nothing else moved. Pair it with `-rs` (no
  skip-swallowed sentinel: 110 passed, 0 skipped, 0 xfailed).

---

## L31 — Verifying an `A ≡ B` theorem that holds BY SHARED BODY (B3.4b): the equivalence gate is designed-GREEN under a body bug, and two sibling routes through one assembly have DISJOINT discriminating fixtures

B3.4b re-posed `AlbedoBoundary`'s specular pairing into `R` and routed the two
completions through the bodies that already realize `ReflectiveBoundary` /
`WhiteBoundary`, making `albedo(α, SpecularReturn(a)) ≡ reflective(a, α)` and
`albedo(α, IsotropicReturn(a,s)) ≡ white(a,s,α)` **theorems by construction**.
Every finding MEASURED 2026-08-01 (`scratch/b34b_verification_plan.md`; new gate
`tests/geometry/test_reemission_closure.py`; harness selected by `ORPHEUS_B34B`).

- **The design's own justification is the reason the gate cannot verify.** "The
  two routes execute the same code, so the theorem holds" ⟹ a shared body
  carries a SHARED bug and moves both sides identically. MEASURED: writing the
  `to_local` remap as a bare `arange` left `TestEquivalenceTheorems` at
  **60 passed / 0 failed** while the operator was wrong on 3 of 5 quadratures.
  The `≡` gate catches ARGUMENT drift only (wrong axis / dropped α / wrong
  `law_key`); the catcher is an **independent-expression anchor** written from
  raw quadrature data (`reflection_index`, `weights × omega_dot_n`) importing
  nothing from the realizer above the `np.take`/`np.tensordot` line. Ship BOTH
  and say in the closeout which one is evidence.
- **⭐ Two sibling routes through one assembly need COMPLEMENTARY fixtures —
  neither list covers the other, and a "representative quadrature" serves
  neither.** MEASURED: the specular remap mutation (`to_local`→`arange`)
  DISCRIMINATES on `gauss_legendre(4/8)` + `lebedev(17)` and is BLIND on
  `product(2,4)` + `level_symmetric(6)` (there `perm[sorted(inflow)] ==
  sorted(outflow)`, so the local permutation IS `arange`); the diffuse
  classification mutation (`TANGENTIAL_EPS`→`> 0.0`) DISCRIMINATES on
  **`product(2,4)` ONLY** (mis-admits 2 ordinates) and is blind on all three
  others. Parametrise both routes over the FULL list and keep the blind rows as
  documented controls with an asserted fixture-table test (`if _id in
  discriminating: assert not array_equal(local, arange)`) — so the day a
  quadrature change makes the anchor blind, THAT reds instead of the anchor
  silently becoming theatre.
- **Folding N routes into one assembly EXPOSES latent crashes that were never
  gated — the fold's own gate is net-new, not migrated.** MEASURED:
  `ScaledOperator` refuses a zero scalar, and the pre-fold reflective/white arms
  had only an `α == 1` fast path ⟹ `ReflectiveBoundary(axis, 0.0)` and
  `WhiteBoundary(..., 0.0)` were LEGAL laws (α=0 passes every invariant) that
  died in the numerics layer. Gate the fixed behaviour AND pin the asymmetry
  with a **negative control** (`ScaledOperator(0.0, …)` still raises), else the
  gate would also pass if someone had widened the numerics primitive instead.
- **A refusal added to a law with per-amplitude fast paths MUST be parametrised
  over α.** MEASURED: guarding the refusal with `and law.albedo not in (0.0,
  1.0)` reds ONLY the α∈{0,1} rows — the exact hole that kept albedo un-narrowed
  at both fast paths through two campaign phases. Same shape as L29's
  "narrowing is per-branch".
- **A `pytest.raises` on a refusal is teeth-less without MESSAGE legs.**
  MEASURED: replacing the specific message with a generic one keeps
  `exc.value.law == "albedo"` TRUE, so a bare `raises(BoundaryError)` + `law`
  check stays green. Assert the substrings that name the COMPLETIONS and the
  DEFECT; pair with a positive control (the completed law realizes), else an arm
  that refused everything also passes (`vv` Mode-8 class 5).
- **An "exactly one of X, Y is non-trivial" invariant: pose it over the whole
  registry; carry a violator with a LANDING PHASE as `xfail(strict=True)` and a
  PERMANENT one as a named positive assertion.** MEASURED two violators, not the
  one the design named: `ReflectiveBoundary(α<1)` = 2 non-trivial (→ B5, xfail —
  the XPASS forces the marker's deletion) and `AlbedoBoundary(1.0)` bare = ZERO
  non-trivial (permanent — the degenerate factorisation a SCALAR trace forces;
  an xfail with no unlock is a red that never flips, so assert it positively and
  gate the union with an inventory-completeness leg). NEVER scope such a gate to
  "tag-reachable laws": measured, `_law_from_tag` hard-codes `albedo=1.0`, so
  that scoping silently drops the exact row the design wanted flagged.
- **When a method REFUSES a spelling another method realizes, the second arm
  needs an INERTNESS gate.** The sibling claim "on a scalar trace the closure has
  nothing to fix" is a claim about the OTHER arm and nothing pins it: gate
  `diffusion.realize(law(α, C)) ≡ diffusion.realize(law(α))` ∀C. MEASURED to red
  under a `type(law.response_kernel)` dispatch — the one mutation that would
  otherwise silently change every `BC("albedo", …)` answer.
- **Method traps.** (1) A mutation can red a gate by RAISING rather than
  comparing (a shape guard fires first) — that red is Mode-8-class-4-shaped
  *inside the mutation*; name the attributable catcher separately. (2) NEVER
  baseline against a tree being written: the same command read `32 failed`
  mid-edit and `9 failed` once settled. (3) A snapshot GENERATOR silently rots
  one phase behind its harness — measured, `_generate_bc_equivalence_snapshots`
  raises on 6 of 8 cases while the harness's own failure message instructs the
  reader to run it.
- **⭐ The `≡` must ALSO hold where production reads the FACTORS, and THERE the
  gate has teeth.** A realized-output `≡` gate is blind to a shared-body bug;
  but the same two laws are also interrogated by predicates that read their
  FACTORS, and those two interrogations share NOTHING. MEASURED: with a
  specular pairing legal in `R`, four SN sites spelling
  `law.geometry_map.permutes_ordinates` inline returned different answers for
  two laws that realize to the SAME matrix (sweep-schedule `B_lower`/`B_upper`
  split, DSA admission, ruled-corner, `_reflect_corner`) — a wrong fixed point,
  not a slow one. Gate it as the EQUIVALENCE (not a hard-coded expected table,
  which drifts with the design) + a NON-VACUITY leg pinning both answers
  (`False`-for-everything is exactly the bug's shape) + a leg asserting the
  RETIRED single-tier read DIVERGES, so a "simplification" back to it reds.
- **Constructing a break-exactly-ONE-invariant mutant is a design problem, and
  the obvious candidate usually breaks several.** For a specular-pairing table:
  `np.roll(arange(N),1)` breaks measure AND sign AND involution, so the
  earliest check fires and the row proves nothing about independence. And **no
  ODD cycle can ever isolate the involution**: `a→b→c→a` forces
  `sign(b)=−sign(a)`, `sign(c)=sign(a)`, `sign(a)=−sign(a)`. The working
  construction is `π ∘ σ` (true mirror ∘ a swap of two SAME-sign SAME-measure
  ordinates) — and it needs a DEGENERATE measure class, which
  `gauss_legendre(8)` does not have. Carry the QUADRATURE per row, not shared.
- **A refusal that stands on two different arguments must have its message
  assertions KEYED to the argument.** `AlbedoBoundary(α)` bare is refused at
  every α, but for α≠0 the defect is a pairing-by-array-position and for α=0 it
  is a `VacuumInflow` twin (nothing returns ⟹ no pairing is missing). A blanket
  "the message names both completions and the under-determination" pins a FALSE
  reason on the α=0 row. Key each leg, and give the negative ("α=0 must NOT say
  under-determined") a CONTROL that α≠0 does — else it is vacuously true.
- **A generator can rot one phase behind its harness, and "fix the generator"
  can be a SCHEMA question wearing a bug's clothes.** MEASURED: the BC-snapshot
  generator raised on 6 of 8 cases (still `SNMethodSpace.minimal` for narrowed
  laws) while the harness's own failure message told the reader to run it.
  Giving it face-ful spaces only moves the failure to the SCHEMA — a narrowed
  law cannot emit the FULL-FACE `psi_in` the frozen artefacts store, and
  migrating the schema DESTROYS the property that makes those artefacts
  valuable (frozen pre-narrowing ⟹ old-vs-new is independent). STOP and hand
  the decision back; fix the false instruction in the error path meanwhile.
- **Snapshot re-pose when a completion supersedes a retired spelling: inherit a
  FROZEN artefact generated by the SIBLING law, do not regenerate.** MEASURED
  `0.5 × specular_x_lebedev17["psi_in"][inflow]` is bit-identical (maxdiff 0.0)
  to the albedo-with-specular-closure image, so the `≡` theorem gets pinned
  against an artefact predating every line under test — criterion 2 of the
  re-baseline ladder satisfied by construction, and no self-generated baseline.
  The retired law's OWN artefact is unusable (its Γ₋ rows read ordinates the
  narrowed input never sees) ⇒ delete artefact + generator case together.
  (SUPERSEDED for this harness by **L32** — the user ruled the whole file over
  to derived references. The inherit-from-a-sibling recipe stays correct
  whenever a derived expression is NOT available.)

---

## L32 — INVERT a snapshot generator from RECORDER to REFERENCE: the artefact stops being production's echo and becomes the math; the frozen FILE is what stops generator+production drifting together

A snapshot generator that calls production and freezes its output is
**self-referential** — the gate says `production == a recording of production`,
which detects change and certifies nothing, and it BREAKS on every production
signature change (this one broke twice: B3.2, then B3.4a). Restricting a
pre-change artefact rescues *procedural* independence only (`vv` L11): it
certifies "the new path agrees with the old path", worth exactly what the old
path was right — and a narrowing campaign's whole premise is that the old path
was wrong. The repair is to **compute the reference from the law's equation and
freeze THAT**; the generator then imports nothing from the realization layer and
cannot be broken by it again.

- **The precondition on the recipe: the expression must be TOTAL.** A recording
  pins every bit of the output; a derived reference pins only what the
  expression determines. Check per case that the expression fixes the whole
  array (all 7 did: `0`, a gather, a broadcast average, their sum, a partner
  gather). If an expression is partial ("the image is isotropic", magnitude
  unfixed), inverting TRADES a total drift lock for a partial correctness
  claim — then keep the recording, or pair the two.
- **Derive from the EQUATION, never by transcribing the implementation.** The
  specular pairing came from the isometry `Ω ↦ Ω − 2(Ω·n̂)n̂`, realized by
  matching flipped direction cosines back onto the ordinate table with EXACT
  float equality (MEASURED exact + involutive on gl4/gl8/lebedev17/ls4/ls6/
  product24) — never `quadrature.reflection_index`, which is production's own
  table. Demanding an exact unique hit is what keeps it independent: a tolerant
  nearest-match would agree with a WRONG table. The Lambertian came from
  `J⁻ = αJ⁺` + isotropy (normalising over Γ₋), where production normalises over
  Γ₊; assert at generation time that `Σ_Γ₊ w|Ω·n̂| == Σ_Γ₋ w|Ω·n̂|` within
  reduction-order ULP so the choice cannot silently matter — and if it ever
  fails, the derived one is the correct one.
- **⭐ Once the reference is computable, the FROZEN FILE is the only thing left
  standing between a wrong expression and a green gate.** If the harness
  recomputed it, generator and production would drift TOGETHER through one
  shared expression — a different self-reference. So the harness reads
  `psi_in` off disk and imports ONLY the case registry; make that
  **structural, not documentary**: an AST gate asserting (a) the generator
  imports nothing matching `realizer|method_space|orpheus.sn`, (b) the harness
  pulls only `{CASES, case_by_id, BCEquivalenceCase}`, (c) artefacts on disk ==
  registered cases (no orphan). All three proven-red by feeding the predicates
  a mutated source copy — never by editing a tracked file.
- **Fold the duplicated case identity into the registry while you are there.**
  Pre-migration the law + quadrature of every case were spelled TWICE
  (generator and harness) with nothing keeping them in step. The clean field is
  `compose: Callable[[Realize], Operator]` — a function OF the harness's
  realization step, so `lambda realize: realize(Vacuum())` covers rank-1 and
  `0.3*realize(a) + 0.7*realize(b)` covers rank-N, and the generator still
  imports no realization code. NOT a stringly-typed `("WhiteBoundary", {...})`.
- **Re-derive every tolerance from structure; the migration MEASURES its own
  quality.** Gathers and α-folds are reduction-depth 0 ⟹ `array_equal` (a
  tolerance there would admit the bug); an `n`-term positive-summand contraction
  vs a `tensordot` is `κ=1`, so `|Γ₊|·ε` (the probe being `U(0,2)`, hence
  non-negative, is what makes κ=1 — say so). Retire inherited `nulp=4`/`nulp=64`
  folklore. Then compare the DERIVED reference against the retired recording:
  MEASURED **bit-identical on 3 of 7**, **1–2 ULP on 3**, **98 % on 1** — the
  outlier being exactly the case whose law production gets wrong. That table is
  the migration's own evidence that six claims did not move, only their source.
- **Keep the PROBE bit-identical across the migration** (draw full-face with the
  same seed, then restrict to Γ₊). Free traceability: the stored rows are a
  literal subset of what the retired artefacts probed with, so the change is
  provably schema+reference and never the input.
- **A two-face law cannot be probed with a one-face draw.** Periodic's reference
  is the PARTNER face's outflow (`Γ₋(xmin) == Γ₊(xmax)` as ordinate sets —
  assert it, do not assume). With ONE draw shared by both faces the defect is
  unobservable/arguable (the faces ARE identified under the quotient, so the
  identity looks defensible); use a SECOND independent seed and ship a LIVE test
  asserting the two probes differ. Config-blindness, one level up: the
  convenient single-draw fixture nulls the term.
- **A strict xfail for a not-yet-built phase needs a FLIP-PROOF, because it will
  not flip by itself when the fix changes the CALL rather than the body.**
  Production's periodic op is an angular identity fed the wrong half-trace, so
  after B3.4c the test's `apply` line changes and the marker is deleted by hand.
  Ship a LIVE companion showing the body already reproduces the reference when
  fed the partner's Γ₊ — it is simultaneously the flip-proof (it names exactly
  what the phase changes) and a body pin (a scaling/permuting/averaging operator
  fails it), and it is NOT `x == x`. Keep every premise (space cross-check,
  pairing, probe activation) in LIVE tests so the xfail body has exactly ONE
  failable statement; verify with `--runxfail` that it reds on THAT line.
- **Do not add cases that duplicate an existing anchor.** The two B3.4b closures
  were rejected as new cases: `test_reemission_closure.py` already anchors both
  against hand-written expressions over 5 quadratures × 3 α, and the only thing
  a case adds is the frozen artefact — already delivered for the specular
  closure by one extra method on an existing case.
- **Store the face and BOTH index sets on EVERY artefact** (pre-migration only
  the vacuum case carried one). It makes each file self-describing AND turns the
  Layer-1 classification into a frozen independent statement the harness
  cross-checks. Honest scope: MEASURED, that cross-check is BLIND to the
  `> 0.0` tangential twin on this case set (lebedev/LS/GL carry EXACT-zero
  tangential cosines; only `product(2,4)` has round-off ones) — name the blind
  spot and point at the file that does cover it, rather than adding a case.
- **Mutation table (8 in-process monkeypatches, control 43 passed/1 xfailed):**
  local-remap→`arange` (2 red), α-fold dropped (3), cosine weight dropped (3),
  zero-map→echo (1), domain built from Γ₋ (6), local-perm reversed (4),
  normal flipped (16), `TANGENTIAL_EPS→0` (0 — the documented blind). The
  `arange` mutation is quadrature-coincidence-blind on ls4/ls6; a REVERSED
  local perm is the coincidence-proof specular mutation — use it to give a
  rank-N row's specular leaf its own red.

---

## L33 — When the phase's PRODUCTION lands while you plan: the deferred gate whose FLIP IS A NO-OP, and the green gate whose REASON decayed (B3.4c, periodic → the partner face)

Two failure shapes, both measured, both new. The context: B3.4c narrows the SN
periodic law so its domain is the PARTNER face's Γ₊ (`γ₋ψ|_f = γ₊ψ|_{f'}`), and
its plan-of-record shipped a strict-xfail "todo list" naming the phase.

### 1. ⭐ The MARKER WHOSE FLIP IS A NO-OP — a fifth Mode-8 class-4 shape

L32 taught "an xfail for a not-yet-built phase needs a FLIP-PROOF because it
will not flip by itself." That is right and it is not enough. The prescribed
flip — *"when it lands, hand the operator the PARTNER's Γ₊ and delete the
marker"* — turned the xfail into a **character-for-character duplicate of the
LIVE flip-proof sibling next to it**, a row green BEFORE and AFTER the phase.
Nothing about the edit is sensitive to whether the production change landed, so
deleting the marker would signal "phase complete" while asserting nothing new.
**MEASURED, decisively:** the production step landed mid-session and the row
still red byte-identically — same 735/735, same first-mismatch values. So a
"B3.4c gate" survived the entire B3.4c production change untouched.

**The discriminator, and it is cheap:** *diff the xfail body against the live
companion you shipped as its flip-proof.* If applying the documented edit makes
them textually equal, the flip is ceremony. **Rule: an xfail's flip-edit must
touch a statement whose VALUE the production change determines.** The repair
here was to state the claim against the production ANSWER rather than a
hard-coded face — `source = law.geometry_map.domain_face(face)`, then index the
probe dict BY that answer, so a regression makes the gate select the wrong probe
and red. And the composite-level claim (the thing the phase actually builds) is
a NEW test in the operator's own file, never an edit of the law-level row: a
harness that only ever realizes a law against a hand-built method space cannot
state a claim about the COMPOSITION.

### 2. ⭐ The gate that stays GREEN while its REASON becomes false

`B` was documented block-DIAGONAL over faces; a wrap makes it block-STRUCTURED.
Three test rows assert it — and `[M]` **all three stay green**, because every
one is posed on a reflective-only fixture. What decays is the *justification*
("`B` is block-diagonal over faces, **so** the face-subset reflect MUST be the
exact restriction" — the Gauss-Seidel schedule's exactness argument). Nothing
in the test, the tag, or CI notices; it is Mode-8 class 7 applied to a REASON
rather than a fixture. **Rule: when a phase falsifies a structural claim, grep
the claim's WORDS in tests/ as well as its symbols — the rows that assert it on
a now-special-case fixture must be re-scoped from "`B`" to "these laws" IN THE
SAME CHANGE, and the new structure gets its own positive gate.**

### 3. The plan's OWN premises had already moved — steps 1/2/3/5/6 were on disk

Five of seven plan steps were implemented, uncommitted, when the plan began, and
step 4 landed mid-session. **Re-run `git status --porcelain -- orpheus/` at the
START and again before every claim; stamp the plan with the tree state.** A
verification plan written against a stale premise gates the wrong thing.

### 4. Recipes worth keeping

- **Ship the plan's gates as a RUNNABLE dry-run module**, not only as fenced
  code. Transcribing them exposed two errors no amount of reading would: the
  perturbation form `B(base+e) − B(base)` is NOT bit-exact (`[M]` 1 ULP, the
  `(x+1)−x` cancellation — exploit LINEARITY with a zero base instead), and one
  negative's refusal had moved from the realizer to the composite.
- **When a tag registry refuses the law, the gate still exists one layer down.**
  `BC("periodic")` raises (`SNMesh.BOUNDARY_OPERATOR_REGISTRY == {"vacuum",
  "reflective"}`), but `sn.realize_boundary_law(law, face)` installs it — that
  is the ONLY layer at which an end-to-end claim is stateable pre-#189, and it
  must be named as such in the fixture's docstring.
- **A predicate flip on a FACTOR may have zero downstream observable.** `[M]`
  `SpatialWrap.is_adjointable: False → True` changed nothing: the composite
  reads each REALIZED operator (identity body ⇒ already `True`). Gate the
  factor, state the scope, and put the teeth on a transpose SUPPORT gate.
- **Reciprocity `⟨Bx,y⟩ = ⟨x,Bᵀy⟩` is a CONSISTENCY check, not a correctness
  one.** `[M]` the pre-fix operator reciprocated at rel 1.15e-16 because forward
  and transpose were wrong the SAME way; reverting the fix does not red it. It
  catches only the forward/transpose MISMATCH (`[M]` rel 1.5e-2…2.3e-1). Pair it
  with an object-level SUPPORT gate (feed a state on Γ₋(f), assert the deposit
  lands on Γ₊(f′) and the own face stays zero).
- **A slab is the degenerate two-face case for any partner map** — "the
  opposite face" and "the other face" coincide, so a 2-D companion is mandatory.
  `[M]` a cross-axis partner is SHAPE-LEGAL (`|Γ₋(xmin)| = |Γ₊(ymax)| = 12` on
  LS4) while the index sets differ, and a face-NAME permutation certification
  passes it.
- **Mutation battery (7 monkeypatches, control 95/95 green):** `domain_face →
  face` (25 red), `face_opposite → identity` (18), suffix↔sign swapped (12),
  revert the composition's source face (7), wrap body ×2 (6), factor
  `is_adjointable → False` (2). **Three predicted blindnesses CONFIRMED:** the
  revert does NOT red reciprocity; `face_opposite → identity` does NOT red the
  involution row (the identity IS an involution — the NO-FIXED-POINT row is the
  catcher); the suffix swap does NOT red either round-trip row (a bijection's
  round-trip is invariant under relabelling both directions — a transcription of
  the RETIRED expression is the catcher). A convention's round-trip is never
  sufficient; pin the ABSOLUTE table.
- **A same-sized-wrong-SET probe is the only way to gate "compare sets, not
  sizes".** `[M]` `replace(space, inflow_indices=space.outflow_indices)` → the
  real guard raises, a size comparison passes. No mesh produces that input, so
  the discipline is otherwise an unverified claim however well its docstring
  argues for it.

---

## L34 — Making a SYMMETRY EXACT is not a ≤1-ULP change: exactness manufactures TIES, and a downstream sort that was accidentally total becomes under-determined (#325, roots_of_unity)

The framing that arrives with such a change is always *"the node values move
by ≤1 ULP, so anything pinned bit-exactly re-baselines."* That sentence is
true about the **values** and can be catastrophically false about the
**results**, because the whole point of the change is to make previously-
distinct floats **bit-equal** — and anything downstream that ORDERS, GROUPS,
DEDUPES, or takes an `argmin`/`argsort` over them was relying on the noise to
break ties.

`[M]` #325: `roots_of_unity` makes azimuthal mirror partners share `η`
bit-exactly. `rules_product.py` sorts each μ-level by `η` with a default
`np.argsort`. Distinct-η per level at `n_phi=64` drops **60 → 33**; the
per-level ordering changed in **36 of 36** `(n_mu, n_phi)` configurations,
including those with no axis node. End-to-end on a heterogeneous 2-G
cylindrical fixed-source at `n_phi=32`, with the node values held
**bit-identical** and only the ordering varied: **1.008 % flux change**; and
`kind="quicksort"` vs `kind="stable"` — a numpy implementation detail —
differ by **1.775 %**. The node-value drift in the same run was **1.06e-14**.
Twelve orders apart, in the same commit, under one justification sentence.

**The checklist when a change makes something exact:**
1. Grep the consumers for `argsort` / `argmin` / `sort` / `unique` /
   `set(` / dict-keying **on the quantity being made exact**. Each is a
   tie-break that was being decided by the noise you just removed.
2. **Is the sort key INJECTIVE?** `η = sinθ·cos φ` is 2-to-1 over
   `φ ∈ [0,2π)`. A non-injective key means the ordering was never determined
   *by the physics* — exactness only reveals it. That is a live latent defect
   to be ruled on, not a side effect of your change.
3. **Does the ambiguity converge away?** `[M]` it did not: the gap sat flat at
   ~1.0–1.8 % across `n_phi ∈ {32, 64, 128}`. A flat-in-refinement gap is a
   defect; a shrinking one is discretization.
4. **Split the commits.** The ordering ruling lands FIRST, alone, with its own
   reference. Otherwise the value-change commit's DriftWarning run is
   uninterpretable and its justification record is a lie.

**The Mode-12 trap specific to ordering:** *"the level is sorted by η"* is the
wrong functional — sortedness is invariant under permuting equal elements, so
the two orderings that differ by 1.8 % are BOTH sorted. Gate the **full index
tuple** against an independently-constructed one, plus a
**`kind=`-invariance** row (`quicksort`/`stable`/`heapsort`/`mergesort` must
give bit-identical tuples), which is the operational proof the key is
injective. And the canonical curvilinear diagnostic is no help: `[M]` the
homogeneous flat-flux `Q/Σ_t` leg gave the SAME value on all four ordering
legs — the config that AGENT.md §0.6 calls the single most powerful
curvilinear diagnostic is designed-green here, because flat flux nulls exactly
the redistribution the ordering perturbs.

### L34b — the two blind involutions, and the "is it a live bug?" discipline

`[M]` Two measurements from the same session, both worth reusing:

- **A nearest-neighbour partner search is not automatically a bug.** The
  hypothesis "NN matching over a noisy set must be mis-pairing" was
  **REFUTED**: the map is a valid permutation AND involution up to
  `n_phi=1024`, because the mis-pair margin is the node separation
  (`5.0e-3`) versus a `1e-16` perturbation — 13 orders. Measure the margin
  before filing it as a bug; and **KEEP the search** rather than replacing it
  with an index formula, because the sibling family (`lebedev`) has no such
  formula and the replacement would mint a twin path AND delete the only
  detector of the next item.
- **`ref[ref] == id` passes on a residual-0.94 garbage map.** `[M]` for odd
  `n_phi` the x-mirror partner does not EXIST; the unguarded `argmin` returns
  the nearest node anyway and the result is still a permutation and still an
  involution. The only functional outside that stabiliser is the
  **RESIDUAL** `max‖R·x_n − x_{ref[n]}‖`. Same shape as the `face_opposite`
  lesson (L33): an involution law alone is blind because the identity is an
  involution — here, because *any* self-inverse pairing is.

### L34c — "≤1 ULP agreement with `np.cos`" is the wrong reference, not just a wrong number

`[M]` The shipped construction is `3.75 ulp(1.0)` from `np.cos(2πp/q)` — the
stated acceptance criterion FAILS. But `np.cos(2πp/q)` is not the true value:
`fl(2πp/q)` already carries argument-reduction error. Against 100-digit
mpmath the new code is `0.57 ulp(1.0)` and the legacy is `3.72 ulp(1.0)`.
**Gating a more-accurate implementation against the less-accurate one it
replaces gates it against the error it exists to remove.** State the criterion
against the arbitrary-precision value, and state it honestly: `[M]` the legacy
is strictly closer on `30/1040 = 2.9 %` of components (by ≤ `1.11e-16`), so
the claim is *"within 0.57 ulp everywhere"*, never *"always closer"*.

### L34d — the harness that says "N gates are blind" is the one to distrust first

`[M]` My first teeth harness reported **18 of 18 gates never red**. The
mutation had never bound — `pytest.main` imports its own copy of the test
module, so patching a privately-loaded copy does nothing. The corrected
harness patches the SOURCE module, purges the test module from `sys.modules`,
and carries a **positive-control probe** ("did the freshly-imported test
module bind the mutant?") printed on every row. With it: 14 of 17 gates red
from the body battery, the remaining 3 from targeted mutations. This is the
`vv-principles` Mode-8 METHOD WARNING and it cost a full false verdict — an
all-blind result is far more likely to be a broken harness than a broken
suite.

Corollary, and it caught me: the same harness flagged a gate **I had just
written** as never-reddening — an "activation guard" asserting odd `q` has no
node at `π` (`not any(2p == q)`), which is a **theorem about parity**, true
for every input. It survived authoring and a full green run. Run the teeth
harness over your OWN new module before delivering it.

### L34e — a "bit-identical to the legacy adapter" pin whose legacy adapter is gone

`[M]` `test_product_bit_identical_to_legacy_adapter` compares
`product_mu_phi(...)` to `Quadrature.product(...)` — and `Quadrature.product`
CALLS `product_mu_phi`. The named legacy adapter
(`orpheus.sn.quadrature.ProductQuadrature`) no longer exists. Replacing
`product_mu_phi` with **random nodes** leaves the test **green**, while its
docstring claims *"Pins the cylindrical regression snapshots"*. A snapshot
whose only "independent" pin is that test has **no anchor at all**, and
re-baselining it against its own old value is the contaminated re-baseline
L32 is about. **Whenever a pin names a "legacy"/"reference"/"adapter"
counterpart, grep that the counterpart still EXISTS and is not the SUT
reached by another name** — a delegation added later silently converts the
pin into `X == X`.

## L35 — verifying a pure-math PRIMITIVE (a group/algebra type) against pure math

From the G2 gate spec for `orpheus/geometry/transformation.py` (rigid motions
`E(d)`), written PRE-carve — and the module LANDED mid-plan, which is half the
lesson. Full plan: `scratch/g2_verification_plan.md`.

### L35a — the reference pillars for an ALGEBRA are different from a solver's

There is no MMS row and no semi-analytical row: a group-theory type has no
differential operator to manufacture a solution for and nothing reduces to a
quadrature. Every row is **closed-form**, and the structurally-independent
grounds available are: (1) **SymPy under an EXPLICIT unit parameterisation**
(`n̂ = (sinθcosφ, sinθsinφ, cosθ)` — `[M]` imposing `Σnᵢ²=1` by `subs` after
expansion **does not fire** and leaves `4n₀²(Σnᵢ²−1)` residuals); (2) an
**external implementation with a different ALGORITHM** (`scipy…Rotation.
from_rotvec` is quaternion-based, so it is not a re-spelling of Rodrigues);
(3) the **Lie definition** `R = expm(θ(vuᵀ−uvᵀ))` — Padé, structurally unlike
every closed form, and the only **dimension-generic** reference (`[M]` 3.9e-14
vs the shipped constructor, so gate at 1e-12, not 1e-15); (4) **published
tables** (group orders; the cube's Wyckoff site symmetries); (5) **exact
integer arithmetic** (a permutation's bijectivity, `π(gh)=π(g)∘π(h)`,
`ord(g^k)=n/gcd(k,n)`) — this last one needs no reference at all and is the
strongest class available.

### L35b — the group-action HOMOMORPHISM is the deepest cheap gate

`π(g∘h) = π(g)∘π(h)` on an induced permutation cross-checks **two independent
layers** — the element algebra and the nearest-neighbour action on points —
using only integers. It simultaneously pins the composition order, the
row-vs-column action convention, and `π` vs `π⁻¹`. `[M]` the wrong composition
order violates it on **102 of 144** cube/`O_h` pairs. It is **VACUOUS on an
abelian group** (`C_n`: 0 violations either way) — the fixture must be
non-abelian. A checker that computes `π` and returns a `bool` (ERR-073's
shape) makes this law unaskable; returning the permutation makes it free.

### L35c — an involution/order law is Mode-12 BLIND to the AFFINE part

`(σ, t)² = (I, σt + t)`, and `σt = −t` for **every** `t ∈ span(n̂)`. `[M]` all
of `t ∈ {d n̂, 2d n̂, 4d n̂, −2d n̂}` are involutions, while the true fixed
plane moves by `0.37 / 0.74 / 1.48`. Likewise a rotation's ORDER is blind to
its centre. **The only catcher is "fixes its named fixed set POINTWISE"** —
gate the FIXED SET, never the order, for anything affine. Generalisation: for
any type carrying a decomposition `(linear, affine)`, enumerate which laws
factor through the linear part alone — those are designed-green on the affine
half.

### L35d — the SEAT is a theorem, not a convention (and it is one gate)

A `G`-preserved weighted point set has a `G`-FIXED centroid (3-line proof from
`w_{π(i)} = w_i` + linearity). `[M]` a cube shifted to `(2.5,−1.25,0.75)`:
**48/48** `O_h` elements *seated at the centroid* preserve it; **1/48**
unseated do. That single row converts "where do we put the origin?" from a
modelling choice into a computed fact, and it is the abstract form of a live
defect (`Mirror('x').is_invariant(mesh)` reads False for production meshes
solely because they start at `origin = 0.0`).

### L35e — bijectivity and the match WINDOW are INDEPENDENT failure modes

The ERR-073 injectivity guard does **not** make a nearest-neighbour certifier
sound. `[M]` a two-point set off-symmetry by **1e-9** certifies under a
**1e-7** window with a *perfectly injective* π. So the window is a
first-class correctness parameter: it must be an **explicit argument** (a
module constant makes the "the window bites" gate SIGNATURE-tautological), and
it should default relative to the point set's **minimum pairwise separation**
— a computable, set-intrinsic quantity — rather than to an absolute constant.

### L35f — a `-> T | None` collapses N guards into one value; isolate from the INPUT side

`permutes` returns `None` for "image not in the set" OR "not injective" (and
`preserves` adds "weights disagree"). A negative gate asserting `is None`
proves only that *some* guard fired. Where the API cannot be changed, prove
the isolation **from the input**, in the test, with plain numpy: `[M]` the
ERR-073 duplicate-node witness has guard1 passing **48/48**, guard3 **48/48**,
guard2 **0/48** — a clean isolator; and an unequal-weight antipodal pair makes
`permutes` SUCCEED while `preserves` FAILS, isolating the weight guard (which
is **VACUOUS on any equal-weight fixture**, i.e. on every shipped quadrature).

### L35g — ⛔ do NOT assert TIGHTER than the type's own invariant

`RigidMotion.__post_init__` accepts `max|QᵀQ−I| ≤ 1e-10`, so an element
carrying a 1e-11 shear is a **legal value of the type**. A gate asserting
`1e-14` on *any* element asserts a property the type does not promise and
would red on a legal instance. Split it: one row for **the type's invariant**
(and its rejection threshold, via the production constructor), a separate,
stronger row for **the named constructors' actual quality** (`[M]` ≤1e-14, and
`signed_permutation` **exactly 0** — integer entries). Conflating them leaves
the constructors ungated at their real quality while over-claiming the type's.

### L35h — bit-exactness is EARNED PER LAW; measure before choosing the assertion

`[M]` on the same type: identity `e∘g=g∘e=g` **500/500 bit-exact**;
associativity **500/500** on signed permutations but **0/500** on general
rotations; `σ²=I` bit-exact for a COORDINATE normal but **0/300** for a
general one; `g∘g⁻¹` **0/500**; seating fixes `c` only **58/500**. And a
seated reflection of a cell lattice lands its images **5.6e-17–2.2e-16** off
their partners while the induced PERMUTATION is exactly `arange(n)[::-1]` —
integers exact, coordinates not. Pick `array_equal` vs `atol` per row from
measurement; a uniform choice is either a false red or a thrown-away gate.
The coordinate-vs-general normal split is also a **size-blindness** trap in the
argsort-n family: a coordinate-only fixture tests the degenerate diagonal
sign-flip and is structurally blind to the general Householder path.

### L35i — the module can LAND while you are planning it (and that IS the deliverable)

`[M]` the brief said `transformation.py` did not exist; it was written during
the session (untracked, mtime mid-plan). Re-checking the plan against the
shipped code re-scored **10 of 15 API findings as DISCHARGED** and surfaced
**3 new gates the landed design earns** (two spelled actions
`on_points`/`on_directions`; a circle-point rotation primitive that preserves
exactness where an angle cannot; `element_order() -> int | None` where `None`
is the glide/screw law). **Before delivering any pre-carve plan, `ls` the
target path** — the reconciliation is higher-value than the original matrix,
and an unreconciled plan silently specifies a dead API. Same family as the
B3.4c finding (L33) where all seven production steps landed mid-plan.

### L35j — WRITING the gates found three defects, and every one was in the FIXTURE

G2 shipped as `tests/geometry/test_transformation.py` (42 gates / 96 cases,
8.9 s, 32/32 mutations caught, 0 blind). The three reds during authoring were
all mine, and each is a reusable trap:

1. **A single Gram–Schmidt pass is not orthonormal enough for a 1e-12 gate.**
   `[M]` `max|GGᵀ − I| = 2.1e-12` on near-parallel random draws, so the
   production constructor *correctly refused my fixture*. When a shipped guard
   rejects your test input, suspect the fixture first. Build orthonormal
   frames with QR.
2. **An ABSOLUTE tolerance is the wrong contract for a translation residual.**
   It scales `O(ops × ‖t‖ × eps)`, so a fixed `atol` reds on correct code for
   large-‖t‖ draws (`[M]` 2.1e-14 against a 1e-14 bound) — while being too
   loose for small ones. Normalise by `max(1, ‖desired‖_∞)` and MEASURE the
   normalised worst case (`[M]` ≤6.9e-15 over 3000–4000 draws per law).
3. **State the law in the direction that is a FLOAT theorem.**
   `on_points − on_directions == t` is NOT exact (`fl(a+t) − a ≠ t`; `[M]`
   2.2e-16) but `on_points == on_directions + t` IS bit-exact `[M]` 6000/6000,
   because it recomputes the same expression. The true direction is also the
   stronger assertion — check both before settling for a tolerance.

### L35k — the SELF-CONSISTENT gate is blind to a transposed action; the homomorphism is not

`[M]` The mutation `x @ Q` for `x @ Qᵀ` reddens **O2 (the homomorphism) but
NOT O1 (the positive `permutes` gate)**. O1 asserts `g.on_points(P) == P[π]`
where `π` was *built by the same mutated action* — and a transposed action
still maps a group-invariant set onto itself (`Qᵀ` is in the group), so O1 is
self-consistent and green while returning the WRONG permutation. Only
`π(g∘h) = π(g)∘π(h)` catches it, because a transposed action is an
ANTI-homomorphism. Generalisation: **a gate that builds its own reference
through the mutated path can only see errors that break self-consistency** —
pair every "the map is correct" row with a COMPOSITION law it must satisfy.

### L35l — the harness lied first, again (L34d recurrence, and the control caught it)

`[M]` The G2 battery's first run reported **32/32 BLIND** while the summary
lines plainly read `23 failed` / `63 failed` / `38 failed`: the parser wanted
`FAILED` lines that `-q --tb=no` never emits (needs `-rf`), and ANSI colour
codes broke the match. The POSITIVE CONTROL is what exposed it — a mutation
making `reflection` return `+I` cannot leave 42 gates green, so "0 gates" was
a *harness* verdict. Cost: one run. **Never ship a battery without a mutation
whose expected blast radius is large enough that a zero reading is absurd.**

---

## L36 — TYPE-COLLAPSE carve (N types → 1 parameterised type): the class-level gates decay silently

Source: G5 verification plan (`scratch/g5_verification_plan.md`), PRE-carve,
`refactor/operator-strategy-layers` @ `af99f064`. The carve: collapse
`IdentityMap` + `SpecularMirror` (`orpheus/geometry/boundary/_factors.py:443,474`)
into one `SelfPairedDeck(motion: RigidMotion)`.

### L36a — the archetype's signature failure: class-level assertions become tautologies

A type-collapse moves the discriminating information **from the TYPE to a
FIELD**. Every existing gate that asserts on the type keeps passing and stops
discriminating. Two measured instances in ONE file
(`tests/geometry/test_boundary_factors.py`):

* `test_every_production_law_states_both_factors` asserts
  `isinstance(law.geometry_map, geom_cls)` per law. Today: `IdentityMap × 5`,
  `SpecularMirror × 2`, `SpatialWrap × 1`. Post-collapse **7 of 8 rows expect
  the same class**, so "a vacuum law that mistakenly declared a mirror" is no
  longer distinguishable.
* `test_specular_mirror_is_the_only_ordinate_permuting_geometry` asserts
  `permuting_geoms == {SpecularMirror}`. Post-collapse the permuting set and the
  non-permuting set are the SAME class — the gate reduces to "the class is among
  the classes that permute", satisfied by any non-empty subset.

**RULE: for any N→1 type collapse, inventory every `isinstance` / `type(...)` /
class-set assertion over the collapsing family FIRST, and re-pose each onto the
PARAMETER in the same commit that lands the collapse.** A commit that lands the
collapse alone has deleted those gates while the suite stayed green. The
re-posed form asserts the parameter's VALUE (here: the exact signed-diagonal
matrix per law, `array_equal` — bit-exactness is earned, `[M]` 12/12 exact).

Cheap companion the archetype earns for free: `type(A().factor) is
type(B().factor)` — `is`, not `isinstance`. It is the collapse's headline claim
as one line, and it is the ONLY gate that catches "collapsed" into a base class
plus two subclasses.

### L36b — the naming invariant is necessary-not-sufficient, and the missing clause has a concrete inhabitant that does the DEFERRED type's job

The brief's insight — "a face paired with ITSELF has an involutive pairing" — is
true forward and was used as the guard `element_order() in (1,2)`. The CONVERSE
is false. `[M]` (probe p1) the involutions of `E(3)` are FOUR families:

```
identity        order=1 det=+1 fix=3      self-paired  ✔
reflection      order=2 det=-1 fix=2      self-paired  ✔
half-turn       order=2 det=+1 fix=1      maps a face to its OPPOSITE  ✘
inversion (-I)  order=2 det=-1 fix=0      maps a face to its OPPOSITE  ✘
```

So the sketched guard admits exactly the elements that do the **deferred**
type's job (`SpatialWrap`), re-opening the illegal state the carve exists to
close. Correct guard is a conjunction; the second clause (`fixed_subspace_
dimension ≥ dimension − 1`) is the load-bearing one, and it is dimension-generic
(`[M]` d=1: reflection fix=0=d−1 admitted, inversion(1) IS the mirror; d=2:
inversion(2) is a half-turn, det=+1 fix=0, rejected).

**RULE: when a carve's guard is ONE algebraic property named after the concept,
enumerate that property's full conjugacy classes and check each against the
SEMANTIC claim.** Then build the two-way witness pair — here a glide
(`T[0,1,0] ∘ refl_x`: `fix=2` ✔, `order=None` ✘) and `inversion(3)`
(`order=2` ✔, `fix=0` ✘) — so each clause has a mutant that only IT catches,
with DISTINCT `match=` strings. A single generic message lets one clause be
deleted with the other's message still matching.

### L36c — a `RigidMotion`-carrying type's TRANSLATION is bit-identically invisible to every angular functional

`RigidMotion.on_directions` **drops the translation by construction**
(`transformation.py:515-518`). `[M]` `reflection(normal=e_x, offset=off)` gives
an image bit-identical to `offset=0` at `off ∈ {0, 2.5, −17.0}` on
`lebedev(17)`. So a wrong mirror-plane POSITION is designed-green at the
permutation level, the realized-image level, the frozen-snapshot level and any
scalar — no tolerance, refinement or regime can expose it.

Closure is the ERR-067 pattern: **repair the TYPE, not the gate.** Requiring
`motion.is_linear` at construction makes the offset unspellable and turns the
blindness into a real gate. `[M]` the stronger guard `is_linear ∧ fix ≥ d−1`
also *implies* involution (a linear orthogonal `Q` fixing a hyperplane pointwise
is `I` or a reflection ⟹ `Q²=I`), so it needs no `element_order` and carries no
`atol` — the involution property demotes to an asserted theorem. Prefer it.

Sibling permanent blindness worth recording: for ANY involution
`perm == argsort(perm)` identically, so forward-vs-inverse permutation is
unfalsifiable by theorem. Good news for `is_adjointable`; it means the day a
NON-involutive deck map lands (#178), the transpose direction becomes falsifiable
for the first time and needs its own gate.

### L36d — the level lattice: bool / matrix / permutation / certified-table are pairwise incomparable

The recurring question "is a permutation-level gate blind to what a matrix-level
gate catches, and vice versa?" has a clean answer for this shape — **neither
subsumes the other, so all four levels are required**:

| level | sees | blind to |
|---|---|---|
| bool (`permutes_ordinates`) | that *some* relabeling happens | which axis; the offset; whether the quadrature admits it |
| matrix (`motion.linear`) | which mirror, bit-for-bit | whether the quadrature is CLOSED under it |
| permutation (`reflection_index`) | the right relabeling of THIS quadrature | the offset; forward-vs-inverse; a local remap unless the fixture REVERSES order |
| certified table (ERR-042/044/045) | involutive ∧ measure-preserving ∧ inflow→outflow | which mirror was declared (its canonical mutant IS the identity table) |

Two operational corollaries. (i) `domain_face`-style accessors that don't read
the parameter have the WHOLE group as their stabiliser — such a gate can never
be evidence about the parameter; its value is entirely in the cross-type
partition (self-paired ⟺ not-a-wrap), so it MUST ship with the sibling type's
no-fixed-point control. (ii) The certified-table row whose canonical mutant is
the identity table (`ERR-045`,
`tests/geometry/test_bc_universal_invariants.py::TestReflectiveInflowToOutflowInvariant`)
is the row a merge-identity-with-mirror carve must not let decay — it is the gate
saying the two are still distinguishable where it matters.

### L36e — measure the brief's cost estimate before rationing the battery

`[M]` The brief budgeted "`tests/numerics` + `tests/geometry` ≈ 5.5 min". The
subset the carve can actually reach —
`tests/geometry tests/numerics/test_quadrature_directional.py tests/numerics/test_face_layout.py`
— runs in **9.40 s** (`1 failed, 743 passed`, the 1 being the pre-declared
task-#33 red). Off by ~35×, in the helpful direction: a 16-mutation battery goes
from "ration it" to ≈6 min. **Measure the reachable subset, not the directory
the brief names.** Same discipline as refuting an optimistic premise — an
over-stated cost silently shrinks the battery, which is the same loss as a blind
gate but harder to see.

### L36f — a `pytest.raises(ValueError)` written against a `ValueError` SUBCLASS is a false gate that reads correct

`class BoundaryError(ValueError)` (`orpheus/geometry/boundary/_errors.py:44`).
The defect being fixed is that a certification path raises a *bare* `ValueError`
instead of `BoundaryError` — so `[M]` `except ValueError` catches it today and
`except BoundaryError` does not. A gate spelled `pytest.raises(ValueError)` is
therefore GREEN BEFORE THE FIX and stays green after: total loss, one word, and
it looks right in review. **When the fix is "raise the project's typed error
instead of the builtin", the gate MUST name the subclass, and the pre-fix state
of that gate MUST be RED** — ship it `xfail(strict=True)` so the fix's landing
is what deletes the marker.

---

## L37 — OPTIONAL-BINDING carve (a `None` default that silently disables a derivation): the gate you cannot write, and the sentinel that means two things

**Campaign:** ORPHEUS `geometric_transformation_consolidation` G6 — plan
`scratch/g6_verification_plan.md`, written PRE-carve 2026-08-04 at `2ca7dd86`.
Scope: `LinearOperator.domain`/`.codomain` are `Optional[FunctionSpace] = None`;
when `None`, `_AdjointOperator.apply` (`numerics/operator.py:1223-1226`) **skips
both metric applications**, so `A† = G_V⁻¹AᵀG_W` degrades silently to the bare
Euclidean transpose. G6 mints `Γ±(face)` spaces, binds the boundary tier, makes
binding mandatory, moves traversal onto the space.

### a. The carve archetype and where its keystone lives

An **optional-binding** carve is not a value carve. Nothing numerical changes on
the forward path (`apply` never reads `domain`), so a bit-identity keystone is
available and CHEAP — and it is worthless alone, because the whole point of the
carve is the ONE path where numbers DO change (`.H`). The keystone is a
**metric-sensitive reciprocity gate**, and it must be written BEFORE the metric
is installed so it goes RED→GREEN. Written after, it can only ever be green, and
a green metric gate proves nothing about whether the metric does anything.

Corollary the plan states as non-negotiable ordering: **the red→green transition
IS the evidence the metric step is not decorative.** Nothing else produces it.

### b. ⭐ The Mode-12 stabiliser was FOUR cases wide — measure the whole tier, not the one operator

`vv` Mode 12 says compute the commutator `[G, Aᵀ]` at design time. Measured on
the ORPHEUS SN boundary tier, EVERY shipped law except one is in the stabiliser:

| realized law | operator | `[G, Aᵀ] = 0`? |
|---|---|---|
| vacuum / any re-emission at α=0 | `ZeroOperator` | yes — `G⁻¹0ᵀG = 0` |
| reflective, albedo+specular | `PermutationOperator` | yes — ERR-042 weight preservation, **`G₋ == G₊[local]` bit-identical** on GL(4)/product(4,4)/lebedev(17) |
| periodic | index-identity wrap | yes — opposite normals ⟹ same `\|Ω·n\|` |
| prescribed inflow | `IncomingSourceOperator` | rank-0, apply-only |
| **white / albedo+isotropic** | `AngularAverageOperator` | **NO** — `‖A†−Aᵀ‖/‖Aᵀ‖` = 0.209 / 0.684 / 0.612 |

⟹ the adjoint is "right by accident of the one case anyone checked", and the
accident is four cases wide. **A design-time stabiliser audit must enumerate the
WHOLE family the gate could be posed on, not just the operator in front of you** —
otherwise "test the reflective BC harder" reads like a cheaper alternative when
it is provably a non-catcher.

Measured margins for the one catcher (`probe_rank1.py`, lebedev 17, spaces bound,
`outer(1_{Γ₋}, InnerProductFunctional(cos_w/norm))` carrying PRODUCTION `cos_w`):

```
⟨Ax,y⟩_G−  vs  ⟨x, A.H y⟩_G+   rel err 2.2e-16   <- the LAW
⟨Ax,y⟩_G−  vs  ⟨x, Aᵀ  y⟩_G+   rel err 5.5e-1    <- the DEGRADATION
mutation domain:=None  ⟹  A.H y  array_equal  Aᵀ y   (exactly)
activation: trace metric max/min = 1.351 / 3.468 / 5.601  (GL4 / product / lebedev)
```

### c. ⭐ The MUTATION found what code-reading could only assert

Positive control PC1 (`FunctionSpace.apply_inverse_metric := identity`, a
linear/shape-preserving mutation — anti-#18 clean) over
`tests/sn/operators + tests/geometry`:

```
baseline    3 failed, 1597 passed   23.2 s
PC1        33 failed, 1567 passed   23.7 s      ⟹ 30 NET REDS
  19 test_g_adjoint_reciprocity.py · 7 test_inverse_adjoint_coherence.py
   3 test_radial_characteristic_metric.py · 1 test_psi_half_coupling.py
```

**All 30 are bulk / SN-leaf / radial-characteristic gates. NOT ONE narrowed
boundary-law operator reddened** — because `domain is None` short-circuits the
call, so the metric mutation is *unreachable* there. That is empirical proof of
the blindness, obtained by mutation rather than by reading `if x is not None`.

**Reusable technique: run the positive control BEFORE writing the gate, and read
WHICH files it misses.** The absent reds map the blind region exactly, and they
are far more persuasive in a plan than a code citation.

### d. ⭐ A sentinel that encodes TWO states makes the discriminating gate
UNWRITABLE — the type must fix it, not a test

`domain = None` means BOTH "this operator is space-GENERIC by mathematics"
(`IdentityOperator`, `ZeroOperator`, `DiagonalOperator`, `PermutationOperator` —
an identity is the identity on every space) AND "nobody bothered to bind it".
**No gate can distinguish them**, because the two states are the same value.
That is *why* the degradation is silent, and it is why "an unbound operator
cannot be constructed" is the wrong acceptance line for the generic leaves.

The verification verdict is a TYPE demand, not a test: mint an explicit
space-generic marker so the escape is DECLARED. Generalisation — **when an
acceptance criterion asks a gate to separate two states that share one runtime
value, the honest deliverable is "this gate cannot exist; here is the type that
makes it exist", not a cleverer assertion.**

### e. The survey that sizes a "turn on a check that never ran" risk

Instrument the composer constructors and COUNT the skips before binding
anything. Plugin `skipcount2.py`, `tests/geometry + tests/sn/operators`, 23 s:

```
OperatorSum 172 SKIP / 1694 CHECK (9.2 %) · OperatorProduct 6 / 1568 (0.4 %)
178 skipping compositions, 10 distinct operand shapes
  148  Sum(IsotropicScattering, IsotropicN2N)          both sides fully anonymous
    5  Product(BulkAnalysisOperator, SweepOperator)    LEFT CODOMAIN ONLY  <- the dangerous one
```

**The half-bound composition is the highest-risk row**, not the biggest one: the
bound side already commits to a space and nothing has ever compared it to the
other, so a genuine mismatch can be hiding there. A fully-anonymous pair usually
just needs the same space on both sides.

### f. Two acceptance lines that were arithmetically / structurally impossible

1. **"permute the layout and assert BIT-IDENTICAL output."** Measured: a
   49-term reduction under 200 random permutations is bit-identical **50/200
   (25 %)**, worst drift 4 ULP; even `sum(cos_w)` alone is bit-stable 146/200.
   FP addition is not associative — **no architecture can make a REDUCTION
   permutation bit-identical.** The honest replacement is a permutation that
   reorders no addition: the **face packing order** in the flat-buffer layout.
   Measured bit-identical 4/4 per face (slices + index sets) while the flat
   buffer moves — a stronger discriminator, because it reds exactly the
   consumer that reads a self-computed offset instead of a name.
   **Rule: before accepting a bit-identity acceptance line, ask which
   REDUCTIONS the proposed change reorders. Zero ⟹ `array_equal` is honest.
   Any ⟹ the line is impossible and must be re-scoped, not loosened.**
2. **"ZERO diff in any operator file."** Not a pytest gate — a permanent test
   asserting a property of a diff that no longer exists is signature-tautological
   (`vv` Mode 8 class 3). Split into (a) a one-shot commit-scoped
   `git diff --name-only` check quoted in the commit message, and (b) a
   PERMANENT **AST vocabulary gate** forbidding the book-keeping the carve
   relocates (`axis=<int>` signature defaults, `.to_local(`, bare
   `n_restricted`/`n_total`) inside operator modules — with a negative leg
   feeding the walker a fixture string that MUST be flagged, else it is a grep
   that passes because it matched nothing.

### g. The decay list has THREE flavours, not one (the G5 lesson, sharpened)

G5 taught: a type-collapse makes class-level gates DECAY to tautologies, and the
re-pose must land in the same commit. An **optional→mandatory** carve is sharper,
because it changes what is CONSTRUCTIBLE:

- **DIE** — the gate can no longer build its subject (`test_space_anonymous_by_default`
  asserting `M.domain is None`; `test_space_anonymous_leaves_refuse` constructing
  three unbound operators). It reds; it must be deleted or re-posed.
- **DECAY** — still constructs, still passes, assertion is now a theorem of the
  type system (`test_leaf_declares_both_function_spaces`). **Invisible in a run.**
  Re-pose onto what stays contingent: not "a space is bound" but "the space is the
  RIGHT one" — `op.domain is mesh.full_field_space`, **identity not equality**,
  since `FunctionSpace.__eq__` is `(name, shape)` and accepts a look-alike.
- **INVERT** — the gate now pins the degradation as the contract. Canonical
  instance: `test_self_adjoint_M_H_equals_M`, whose docstring literally says
  *"the metric-blind Euclidean `.H` (domain None) reduces to the representation
  transpose"* and asserts `array_equal(M.H.apply(ψ), forward)`. Both halves break:
  the docstring's REASON goes present-tense-false, and the assertion itself reds —
  **measured, binding a metric changes a self-adjoint diagonal's `.H` by 2 nulp**
  (`G⁻¹(f·(G x))` vs `f·x` — `(g·x)/g` is not an IEEE identity).
  **Corollary: `assert_array_equal` on ANY `.H` result of a newly-bound operator
  breaks at ~2 nulp. Grep them all before the carve.**

### h. The strict-xfail set was ALREADY the todo list — and it must be read by ARM

`tests/sn/architecture/test_monomorphic_leaves.py` already carried G6.4's
acceptance gates as `strict=True` xfails whose reasons say *"WHEN THIS XPASSES:
P1 has landed — delete this marker."* Measured `105 passed, 21 xfailed` in 1.31 s,
of which **exactly 12** flip (4 `test_model_generic_leaf_declares_a_space` + 5
`test_leaf_space_annotation_is_not_optional` + 3
`test_leaf_without_a_space_refuses_construction`). The other **8** (`_R6_XFAIL`,
the carrier-guard non-uniformity) are a DIFFERENT campaign phase and must NOT be
deleted. **Deleting markers because "the xfails flipped" is exactly the Mode-8
class-4 misattribution.** Read the xfail set BY ARM, never by file.

### i. The scope alarm a decay-list sweep surfaces for free

The committed xfail `test_model_generic_leaf_declares_a_space` mirrors
`orpheus/homogeneous/solver.py` line for line and MEASURES `C.domain is None` /
`F.domain is None` there — a PRODUCTION path. ⟹ **a "boundary-tier" carve that
retires the `None` default on the BASE class silently reaches the homogeneous
solver, 12 test-local `LinearOperator` subclasses across 6 files, and a
`_multiplier` helper feeding 21 call sites.** The decay list is ~20 tests if the
mandate is boundary-scoped and ~150 if it is tree-wide, and *the plan must choose
before the first line of code*, because everything downstream is sized by it.

**Generalisable: when a campaign step says "make X mandatory", locate WHERE the
optional default is declared. If it is on a shared base, the step is not scoped
to the campaign's tier no matter what the plan's heading says.**

### j. A harness lie reproduced, in the exact anti-#17 shape

`grep "^FAILED"` on a `-q --tb=no` run returned **zero rows** on a run whose own
summary line read `33 failed` — the `FAILED` lines carry ANSI colour codes.
Identical to the 2026-08-03 battery that reported 32/32 BLIND. **Always
`sed -e 's/\x1b\[[0-9;]*m//g'` before parsing, and always cross-check the
extracted count against the summary line.** Cost here: one confusing minute,
because a positive control was already in hand to contradict it.

---

## L38 — BINDING carve, step 5 (bind the deck permutation `Γ₊→Γ₋`): the binding is ungated by construction, and it evaporates at the tensor product

**Context.** ORPHEUS `#330` G6.3 step 5, `refactor/operator-strategy-layers`,
2026-08-04. `PermutationOperator` gains optional `domain`/`codomain`;
`_specular_kernel` binds them; `is_involution` refuses across two different
spaces; `inverse()` swaps them. I was dispatched to design the gates.

### 1. The tree moved TWICE mid-dispatch, and it demoted one of my measurements

The brief said "I am writing the code directly; you design the gates" — i.e.
pre-carve. `[M]` `git status` at 22:39: `operator.py` modified (items 1/3/4
already landed). At 22:41: `realizer.py` modified (item 2 landed). The plan
went from PRE-carve to POST-carve while I was reading context.

The damage: my first bit-identity probe compared the production kernel against a
hand-bound copy of itself. Between probe run 1 (22:39, production UNBOUND) and
run 2 (22:41, production BOUND) the implementer shipped the binding, so both
sides of my `array_equal` were bound and the leg was true for free — the
lesson-#4 demotion arriving through the clock rather than through a rewire. The
tell was a *sign inversion in the data*: run 1 reported
`is_involution=True` on `gauss_legendre(4)`, run 2 reported `False` for the same
fixture. **A measurement that contradicts your own earlier measurement of the
same quantity is a tree-moved signal, not a flaky test.**

Defences adopted: build controls EXPLICITLY (`np.asarray(P.perm).copy()`),
assert `control.domain is None` INSIDE the gate, re-`git diff` before every
claim, and read "I am about to write X" as "X may already be on disk".

### 2. The binding is gated by NOTHING — measured, not argued

In-process pytest plugin rebinding `realizer._specular_kernel`; the plugin
raises if the rebind does not take and prints its entry count in the terminal
summary. Baseline `tests/geometry tests/sn/operators -m "not slow"` =
`3 failed / 1668 passed` in 24 s, `_specular_kernel entered 1252 times`.

| mutation | new reds |
|---|---|
| **POSITIVE CONTROL** — return the IDENTITY permutation, binding intact | **+23** (26 failed) |
| drop the binding (`domain=None, codomain=None`) | **0** |
| ⭐ **SWAP** the binding (`domain=Γ₋, codomain=Γ₊`) | **0** |
| bind both ends to `Γ₊` | **0** |

The control is inside the algebraic class (the identity IS a permutation:
linear, orthogonal, right shape, right spaces), so anti-#18 holds and its 23
reds mean "wrong bijection", not "stopped being an operator".

⭐ **The SWAP is the mutation to design for.** It is a one-word implementer slip;
it survives the extent guard because `|Γ₊| == |Γ₋|` on every shipped quadrature;
it leaves `is_involution` correctly `False` (the two spaces still differ), so
the refusal gate cannot see it; it changes no arithmetic. The ONLY catcher that
can exist is an `is`-identity row asserting WHICH space is WHICH end. And it is
not benign: once the composition is routed through `@`, a swapped binding makes
the LEGAL composition raise and the ILLEGAL one pass.

### 3. The binding evaporates at the tensor product — the step's own goal unmet

`[M]` through `SNBoundaryRealizer().realize(...)`:

| law | returned type | `.domain` |
|---|---|---|
| reflective α=1 | `TensorProductOperator` | **None** |
| reflective α=0.5 | `ScaledOperator` (forwards a `None`) | **None** |
| white | `TensorProductOperator` | **None** |
| — its inner `OperatorProduct` (the committed step-3 chain) | `OperatorProduct` | `Γ₊(xmin)` ✅ |

`OperatorProduct` derives its spaces from its factors; `TensorProductOperator`
does not. So a leaf bound `Γ₊→Γ₋` reaches the realizer's OUTPUT as `None`, and
the campaign's next step ("route `_reflect_trace` through `@` so the
composability check FIRES") **cannot fire** — `[M]` one `None` short-circuits
the check (`bound @ unbound` is ACCEPTED). The already-COMMITTED step 3 has the
identical hole.

⟹ **Measure the binding at the tier the CONSUMER sees.** Rule: before crediting
"the check now fires", compose the object production HANDS OUT, not the leaf you
just bound. And when a sibling step landed the same pattern earlier, check it too
— then ship the gap as a `strict=True` xfail naming the later step, rather than
widening scope.

### 4. `is_involution` — the fixture-dependent attribute, and its four legs

One physical law (mirror about `x` on `xmin`), narrowed to `Γ₊`-local indices:

| quadrature | raw `perm[perm]==arange` | UNBOUND | SAME-SPACE | HALF-BOUND | bound `Γ₊→Γ₋` |
|---|---|---|---|---|---|
| gl(4) / gl(8) / product(4,4) / ls(6) | True | True | True | True | **False** |
| lebedev(17) | **False** | False | False | False | **False** |

Three gate-design consequences:

* **A refusal gate written only on `lebedev(17)` is green on a NO-OP** — its
  `False` comes from the index test, not the space test. Every refusal row needs
  an ACTIVATION GUARD asserting `perm[perm] == arange` is True, and lebedev
  belongs in a separately-named control row.
* **Two positive legs, not one, and they are not interchangeable.** UNBOUND
  (`None`/`None`) and SAME-SPACE (`Γ₊→Γ₊`) both report True; only SAME-SPACE
  proves the clause discriminates on *space inequality* rather than on *presence
  of binding*. An implementation spelled `not (domain is not None)` passes
  UNBOUND and fails SAME-SPACE.
* **HALF-BOUND reports True** (the conjunction needs both ends) — a deliberate
  asymmetry pinned by nothing; it is the sole catcher for an `and`→`or` slip.

### 5. `.H` is metric-blind, and the criterion is EXACT (not a tolerance question)

`[M]` `G_{Γ₋} == G_{Γ₊} ∘ π` **bit-exactly on all five quadratures** — a specular
mirror preserves `|Ω·n|·w_n`, so `G₊⁻¹PᵀG₋ = Pᵀ` in exact arithmetic. This is
`vv` Mode 12's own named example. `.H` vs `apply_transpose`: `array_equal` True
on gl(4) and product(4,4), **1 nulp** on gl(8), ls(6), lebedev(17) — the
`(g·y)/g` round-trip, and note the 0-vs-1 split lands on DIFFERENT fixtures than
step 1's did. Fixtures agree by luck; `assert_array_equal` is a false red on 3
of 5. Tolerance 2 nulp from round-trip DEPTH.

⭐ **Pin the blindness CRITERION as its own gate**, not just the near-equality:
`wm == wp[perm]` is a statement about the physics, it explains why the `.H` row
is allowed to be approximate, and it reds FIRST if a future quadrature breaks
it. Negative legs, measured and quadrature-independent: scaling `G_{Γ₊}` by 3
moves `.H` by exactly `2/3`; scaling `G_{Γ₋}` by 3 moves it by exactly `2`.
Assert the NUMBER, not "it moved".

### 6. Cost — the reachable-subset rule again

The brief's cost ladder said `+ tests/numerics` ≈5 m 45 s. `[M]` the positive
control reddens **zero** files under `tests/numerics`, and no `tests/numerics`
module imports `sn.boundary.realizer`. The whole realizer-side battery lives in
a **24 s** slice. Budgeting off the directory rather than the reachable subset
would have made a 7-mutation battery 40 minutes instead of 3.

⚠ **The ANSI trap fired again**: `grep "^FAILED"` returned nothing against a run
that had plainly failed. `--color=no` when parsing. Third instance.

### 7. A retirement question the plan surfaced but did not decide

`is_involution` has **zero production consumers**, its one would-be consumer now
reports `False` by design, and the new docstring concedes *"nothing in the tree
does"*. Zero consumers + a self-conceding docstring is `coding-elegance`
Pattern-6's trim signal verbatim. Raised as an explicit user ruling (refine vs
retire) rather than decided in a test plan — because if it is retired, three of
the ten gates collapse into "ask the algebra" (`P @ P` raises), which is the
better claim anyway.

---

## L39 — P3 affine-boundary-source carve: the CONSUMPTION-TIER readout, and a "prophylactic" carve that was a live bug

Dispatch 2026-08-05, `refactor/operator-strategy-layers` @ `ef4c3537`. Brief:
plan the verification for retiring `IncomingSourceOperator` and collapsing
`realize(PrescribedInflow)` onto the zero morphism. Brief already carried a
measured double-delivery finding; my job was the gate design.

### 1. ⭐⭐ The keystone gate was NOT one of the three candidates offered

The brief offered (a) `B(0)==0` linearity, (b) two-user-path flux agreement,
(c) superposition in `q`. The winner was a FOURTH: **evaluate the boundary law
on the converged answer.** For prescribed inflow `L = 0`, so
`γ₋ψ|_f == q_f` — read from `sol.angular_flux.boundary.face_view(f)[inflow]`.

`[M]` het-2G scattering slab, identical composite bulk on every row:

| config | SI `γ₋(xmin)` | Krylov |
|---|---|---|
| declared, HEAD (both channels) | `5.000000000000` | RAISES |
| declared, pre-P2′ (operator channel only) | `2.500000000000` | RAISES |
| all-vacuum + composite `q_∂` (source channel only) | `2.500000000000` | `2.500000000000` |
| vacuum control | `0.000000000000` | `0.000000000000` |

**Three configurations, three bit-exactly distinguishable readings, ONE
assertion, NO reference solver.** Double = `2q`, loss = `0`, wrong `L` =
`q + Lγ₊ψ`. Closed-form pillar (the BC *is* the definition), discretization-,
mesh-, quadrature- and materials-independent. `assert_array_equal` is EARNED
(`5.0` is `2.5+2.5`, exact; `2.5` is a copy).

⭐ **Generalisable rule: when a carve is about "is this term delivered, and how
many times?", look for a tier where the governing EQUATION can be evaluated on
the converged answer.** A trace/boundary/interface DOF is usually such a tier,
because the posed system's rows there ARE the constitutive law. That beats a
snapshot (self-referential), beats a flux comparison (needs tolerance + a
non-vacuity leg), and beats linearity (blind to the source channel).

### 2. ⛔ Candidate (c) — superposition in `q` — is a Mode-12 NON-CATCHER

The double delivery IS `q → 2q`, **which is still exactly affine in `q`**.
`φ(a) = φ(0) + a·s` holds for every `s`, including `s = 2 s_true`. The scale
factor is in the measured functional's invariance group ⟹ designed-green on the
whole defect class at every tolerance and refinement. Only the *coefficient*
pinned against an independent reference has teeth. **A "linearity in the
parameter" gate can never detect a wrong CONSTANT multiplying that parameter** —
check this before adopting any superposition row.

### 3. ⭐⭐ The "prophylactic architecture" framing was FALSE — measure the fence

The plan's §1 asserted TWO fences and concluded the carve was prophylactic. Both
failed: (i) the `block_role=None` stamp fences nothing because
`SNBoundaryOperator._face_laws` collects EVERY face's law with no role filter
(`|B(0)|_inf = 2.5`); (ii) the consequence is a **hard raise** on Krylov —
`ConvergenceCertificateError: ‖Aψ−q‖/‖q‖ = 1.718` — because an affine
`A(x) = A_lin(x) − c` breaks GMRES's Arnoldi relation.

⭐ And the raise fired on **BOTH sides of P2′**, so the Krylov path was already
dead pre-P2′: P2′ was a double-delivery regression on **SI only**. **A "not a
live bug, fenced twice" claim is a HYPOTHESIS — apply `B(0)` (or the fence's own
predicate) to the object the consumer actually holds.** A stamp only fences if
somebody reads it; grep the consumer for the filter before believing the fence.

### 4. ⚠ anti-#18 in its sharpest form: the OBVIOUS mutation is out-of-class

The natural mutation ("put the affine operator back") **breaks linearity**, so
every solve-level red comes from the operator ceasing to be linear, not from the
delivery count. Reds under it must NOT be credited as single-delivery coverage.

⭐ **The in-class mutation for a delivery-COUNT claim is `q + q` in the SOURCE
channel** — everything stays linear, so every red is attributable to the count
alone. Test: *a gate that reds under the affine reinstatement but NOT under
`q + q` is not a single-delivery catcher.* Second in-class mutation, for the
carve's own content: `L := IdentityOperator` (perfectly linear) — gate A
(`B(0)==0`) is BLIND to it; only the trace gate sees it. That asymmetry is why
one gate cannot replace the other.

### 5. ⭐ The tree was being written UNDER me — three times, and each time the
### reconciliation outvalued the forecast

`realizer.py` had lost the `IncomingSourceOperator` import while the call site
still referenced it (`NameError`) — so I measured "before" from
`git worktree add <tmp> <HEAD-hash>` (NEVER `git checkout`; the tree carries
irrecoverable uncommitted state). Then, mid-plan: four test files migrated, a
brand-new `tests/numerics/test_zero_operator_spaces.py` appeared, and the 8-red
debt I had measured went to **`555 passed`** before I finished. So §2 became an
AUDIT of what landed and §8 the residual-gap list.

⭐ **Re-measure the red count at the END of a planning dispatch, not just the
start.** A migration table whose status column is stale by minutes is worse than
no table — add an explicit "READ §8 FIRST" banner rather than silently shipping
it. And the deliverable that survived was **the gap list**: measured absent by
grep at the end — Gate D, Gate A, the two-channel gate, the Krylov row, the
production SWAP row, the `checked_space_extent` negative, the `is_adjointable`
widening, the un-mutation-verified `catches` re-home, `DEAD TARGETS 2`, a
label-less `:ref:`, and the whole battery.

### 6. ⭐ A measured number recorded in a COMMENT is not a gate

`|B(0)| = 2.5` — the campaign's central measurement — appears in the tree ONLY
as prose at `tests/sn/operators/test_operator_block_role.py:203`. Excellent
documentation; zero teeth. **When auditing a landed carve, grep for the
measurement's NUMBER and ask whether any `assert` consumes it.** The
"honest metadata that gates nothing" shape.

### 7. Did the retired oracle's independence survive? Audit, don't assume

`test_boundary_source_from_specs.py` used `IncomingSourceOperator.apply` as its
shape oracle. **Verdict: no demotion** — that operator's entire body was
`source.evaluate((n_inflow,) + psi_out.shape[1:])`, a ONE-LINE ADAPTER, so
inlining it to `spec.evaluate((|Γ₋|, ng))` preserves the independence, which
always lived in *who produces each side* (the bridge vs the user's own spec
object), never in the adapter.

⭐ **The discriminator for "did this rewire demote the gate?": is the retired
symbol a SOURCE of the expected value, or a FORWARDER of it?** A forwarder can
be inlined for free. Corollary trap recorded as a refuted candidate: making the
new oracle dimension-generic by reading `slot_shape[1:]` — the SUT's own
derivation — WOULD be the demotion.

### 8. The `|Γ₊| ≠ |Γ₋|` Mode-12 fixture migrated STRICTLY STRONGER

`TestIncomingSourceOperator` was the only place unequal half-trace extents were
reachable (equal on every production face), which is what discriminates "fills
the codomain" from "echoes the input". `ZeroOperator` is equally hand-buildable
AND adds a transpose direction and two space identities the apply-only affine
operator never had. ⭐ **When a Mode-12 fixture's subject is retired, check
whether the successor is also hand-constructible — if it is, the claim migrates
and usually GAINS legs.** Two production docstrings had independently written
the same argument (`_narrowed_zero_operator`: *"wrong in principle and merely
lucky in practice"*), which is how I knew the claim was load-bearing.

### 9. The production SWAP hole that a same-object gate cannot see

The landed `test_prescribed_inflow_realizes_the_same_object_vacuum_does` asserts
`prescribed.domain is vacuum.domain` — that the two AGREE, never that **either
is the RIGHT space**. Both bound swapped ⟹ green, and `|Γ₊| == |Γ₋|` on every
production face ⟹ `checked_space_extent` passes too. `L37` measured that three
wrong bindings each produced ZERO new reds over 1668 tests. ⭐ **An
agreement-between-two-siblings row is NOT an is-it-correct row** — a collapse
carve naturally produces the former and it feels like coverage.

### 10. Doc-debt instrumentation: two instruments, and a hole between them

`[M]` `tools/check_docstring_xrefs.py orpheus tests docs` → `DEAD TARGETS 2
across 9 sites`, with file:line and role — far better than a hand grep. BUT it
gates only **fully-qualified Python-domain** targets (7073 of 14541 roles), and
`sphinx -W`/`-n` cannot see a docstring in an un-`automodule`'d module (verified:
`orpheus.sn.boundary.*` and `orpheus.numerics.operator` have none). ⟹ a
`:ref:`bc-affine-source-channel`` cited twice by the new code with **no target
anywhere** is caught by NEITHER. ⭐ **After a carve adds a `:ref:` from a
docstring, grep the label in `docs/` by hand — no instrument reports it.** And
raw grep found 13 files vs the checker's 9 sites: prose mentions are invisible to
the checker, so run BOTH and triage BY TENSE (past-tense provenance STAYS).

### 11. Fixture facts (add to the config-blindness inventory)

* `_build_fixed_source_rhs`'s two arms take **different bulk sources** —
  per-ordinate `(N, ng, *spatial)` array vs an `AngularSourceSink`. Feeding one
  to each leg of a two-channel comparison gave `φ[0] = 3.083` vs `2.480`: a bulk
  difference misread as a channel difference. **Use ONE spelling on every leg.**
* Two channels reaching the same fixed point are **NOT bit-identical**:
  `|φ_D0 − φ_C|_inf = 1.998e-13` at `inner_tol = 1e-13` (rel ≈ 0.3 × tol),
  `array_equal = False`. Gate at `SAFETY(10) × inner_tol`.
* `tests/sn/solve` naming trap: `test_affine_carve_bit_identity.py` is **#208's**
  `FluxDisplacement` carve, NOT the affine BOUNDARY carve — all-vacuum configs,
  `sha256`-frozen, and 3 of its params are pre-existing reds. `[M]` its signature
  is unchanged by P3, which is the useful fact: **a `sha256` gate that MOVES
  during a carve that should not touch its path is a real signal.**
* Costs `[M]`: the P3 blast-radius slice **24.5 s** (555 passed);
  `tests/sn/solve + tests/numerics` **7 m 27 s** (2016 passed, the 3 known reds).

---

## L40 — P4 non-trivial MMS through the DECLARED inflow channel: the brief's central risk was a LANDED fix, and the phase collapsed from "build a reference" to "re-route one"

Dispatch 2026-08-05, branch `refactor/operator-strategy-layers` @ `8d552395`
(P3 landed). Deliverable `scratch/p4_mms_design.md`. Probes
`/Users/rodrigo/.claude/jobs/c30e4f25/tmp/m{1..9}_*.py`.

### L40a — the enumeration refutation, and why the phase changed shape

The brief's framing block was headed *"⛔ THE CENTRAL RISK — vv Mode 7, and it is
already realised in the tree"*, asserting **"Every existing SN MMS ansatz is
chosen so `ψ ≡ 0` on both faces"** and asking me to *"verify this claim across all
four case families"*.

`[M]` M1 — `dir(orpheus.derivations.continuous.mms.sn)` holds **12** `SN*MMSCase`
classes and **13** builders. Boundary trace at the faces:

| builder | `|φ|` @ faces | `prescribed_inflow()` |
|---|---|---|
| slab 1G / slab-2G-het / P1-aniso / 2-D / 2-D-het / sphere / cyl / sphere-aniso / cyl-aniso | `0` … `1.2e-16` | no |
| **`build_slab_nonvacuum_mms_case`** | **0.5, 0.5** | **yes** |
| **`build_slab_2g_nonvacuum_mms_case`** | **0.5, 0.5** | **yes** |
| **`build_sphere_nonvacuum_mms_case`** | **0.5 pole, 0.75 @R** | **yes** |
| **`build_2d_cartesian_ld_stress_mms_case`** | **1.0 on x-edges** | **yes** |

And the module's own §4.6 header states the brief's claim **as the thing 4.6 was
written to fix** ("That is the entire novelty"). This is the SAME lesson the
campaign already recorded twice (`L39` / plan lesson #2 — *grep the corpus before
calling a thing missing*), now at three-for-three, and this time it fired on my
OWN earlier lesson `L3`, which said the gap existed. **`L3` was true when written
and the fix landed; the digest entry had no landing note.** ⟹ when a digest entry
records a GAP, it needs a landing note the moment the gap closes, or it becomes a
brief-generator for a phase that no longer exists.

Consequence: a fresh anisotropic non-vacuum ansatz would have been a Pattern-2
twin of `SNSlabNonVacuumMMSCase` — same `(A_g + μ_n B_g)/W` form, second
manufactured source, second SymPy derivation, second place for the `1/W`
convention to drift. **The phase's real content is ONE line of delivery:**
`case.prescribed_inflow(sn)` (channel tier) → `sn.bc[face] =
sn.realize_boundary_law(PrescribedInflow(source=spec), face)` (law tier), letting
`from_mesh_laws` assemble `q_∂`. `[M]` M5b the two routes are **bit-identical**:
`array_equal(declared[Γ₋], supplied[Γ₋]) = True`, whole-slot `max|diff| = 0.0`,
off-inflow `0.0`, both `linf = 0.39841014024874755`.

`[M]` the ansatz's shipped parameters and the trace they produce (GL-16, `W = 2`):
`a₀=0.5, a₁=0.25, b₀=0.3, k=2π·1.5/L, L=5, c=(1.0,0.4), Σ_t=(1.0,1.5)`,
`SigS=[[0.3,0.2],[0.0,0.6]]`. `xmin`: `A=+0.5c_g, B=+0.3c_g`; `xmax`:
`A=+0.5c_g, B=−0.3c_g` (sign flip — `cos 3π = −1`). Over Γ₋(xmin) g=0 the trace
reads `[0.264252, 0.398410]`, i.e. **`(max−min)/mean = 0.390`** — a 39 % angular
swing, which is exactly what `ConstantInflowSource` cannot express.

### L40b — ⛔ the vv frequency-strengthening default is WRONG for a channel claim

`vv` §MMS-operational-rules says override the simplification bias with high
frequency / mixed scales. `[M]` `n_wavelengths` sweep on the §4.6 2G slab, SI,
declared route, ladder `[20,40,80,160]`:

| `n_wl` | `L2(20)` | `L2(80)` | orders | order gate reds at `ε_q = 1e-4`? |
|---|---|---|---|---|
| **1.5 (shipped)** | `1.70e-3` | `1.02e-4` | `2.041, 2.010, 2.003` | no |
| 3.0 | `1.61e-2` | `9.42e-4` | `2.077, 2.019, 2.005` | no |
| 4.5 | `4.84e-2` | `2.72e-3` | `2.125, 2.031, 2.008` | no |

Raising `k` multiplies the **bulk truncation error ×16** while the
boundary-source error is untouched, so the gate's S/N for the *boundary* claim
gets strictly worse. **Generalisation: the "hardest trial function" rule is
scoped to the claim it was written for (spatial discretization). Identify which
term is the NUMERATOR of your gate's signal-to-noise ratio and strengthen THAT
axis; everything else is denominator.**

### L40c — ⛔ the brief's matvec row is tautological on the phase's own fixture

`[M]` M8b — the P4 MMS slab declares `PrescribedInflow` on BOTH faces, and P3
collapsed that law onto `_narrowed_zero_operator`, so:

```
B.block_role = BlockRole.BOUNDARY ; per-face block_role = BOUNDARY (both)
per-face domain/codomain = Γ₊(f)/Γ₋(f) shape (8,2) ; is_adjointable = True
|B(x)|_inf for a random x = 0.0     ← the whole B is the zero morphism
|B(0)|_inf = 0.0 ; |B(c·x) − c·B(x)|_inf = 0.0
```

So `B(0)=0` / `B(2x)=2B(x)` / additivity hold with both sides structurally zero —
`vv` Mode 8's tautological-companion class, and impossible to red for any input.

`[M]` M9a — the fixture on which linearity IS a claim: **prescribed(`xmin`) +
REFLECTIVE(`xmax`)** (het 2G slab, GL-8): `|B(x1)| = 1.3201645939238549`,
`|B(0)| = 0.0`, `|B(c·x) − c·B(x)| = 0.0` exactly. `[M]` M9b both drivers
converge on it (`γ₋(xmin) = 2.500000000000`; SI 0.05 s, Krylov 0.24 s).

⚠ The MMS ansatz does NOT satisfy a reflective law, so the two claims need two
fixtures — this is why the split is P4 = {the zero-morphism STRUCTURAL row, the
Krylov-reproduces-the-MMS row} and P5 = {every linearity row, on the mixed
fixture, with a `|B(x)| > 0` activation leg}.

Also `vv` anti-#18 applies to the whole-matvec version: on the MMS fixture the
prescribed law's contribution to `A = (L+C) − S − B` is structurally zero, so any
red of an `A`-linearity gate comes from `L+C−S`, not from the boundary law.

### L40d — the sensitivity band where BOTH the rate AND the value gate are blind

`[M]` M8a + a narrowing run. Scale the declared `q` by `(1+ε)`, SI, `[20,40,80]`:

| `ε` | `L2(80)`/honest (g0, g1) | orders g0 | order gate reds? |
|---|---|---|---|
| `1e-4` | `0.8`, `0.6` | `2.089, 2.187` | no |
| `2e-4` | `0.76`, `0.44` | `2.135, 2.286` | no |
| `3e-4` | `0.76`, `0.60` | `2.178, 2.229` | no |
| `5e-4` | `1.00`, `1.36` | `2.254, 1.727` | **yes** |
| `1e-3` | `2.1`, `3.5` | `2.314, 0.503` | yes |
| `1e-2` | `25.9`, `42.9` | `−0.274, −0.106` | yes |

The trap is that below `~3e-4` the error **decreases** (partial cancellation
against the `O(h²)` truncation), so a value gate at ANY `rtol` is blind there too
— "value + rate" is not a floor. Contrast the trace-level keystone, same
mutation: `max|γ₋ψ − manufactured|` reads `0` / `3.985e-13` / `3.984e-10` /
`3.984e-07` at `ε = 0 / 1e-12 / 1e-9 / 1e-6` ⟹ **reds from `ε ≈ 3e-12`, eight
orders sharper.**

### L40e — the keystone's ORACLE decides whether it catches anything

Two spellings of `γ₋ψ|_f = q_f`:
* ❌ `γ₋ψ == spec.evaluate(...)` — self-consistency; under a magnitude mutation
  both sides move ⟹ green for every `ε`. Catches delivery-COUNT only.
* ✅ `γ₋ψ == (case.A(x_f,g) + μ_n·case.B(x_f,g))/W` — the manufactured trace
  recomputed from the reference OBJECT ⟹ catches everything in the L40f battery.

The shipped channel-tier sibling
(`tests/sn/verification/analytical/test_mms_prescribed_inflow.py`, assertion 3)
already uses ✅. **General: for any "the answer satisfies the declared condition"
gate, name which side is under test. If the answer is "both", it is not a gate.**

`[M]` exactness by path (the `nulp` budget differs from P3's): SI is bit-exact
(`array_equal = True`, `0.000e+00`, every face/group/mesh) because it copies `q`
into the inflow slot and sweeps; Krylov SOLVES those rows and carries the
iteration residual — smallest passing `nulp` = **256 / 16 / 256** at
`nc = 20/40/80`, `inner_tol = 1e-13`, trace values `~0.34`. Non-monotone in mesh
⟹ it is an `inner_tol` budget, not an `h`-law. (P3 measured 64 at `_VALUE=2.5`;
same absolute residual, 7× smaller values ⟹ more ULP.)

### L40f — the Mode-7 activation battery (ratios = `L2(80)_mut / L2(80)_honest`)

`[M]` M7, 2G case, SI, `[20,40,80]`, all mutations in-process on the spec:

| mutation | models | g0 / g1 ratio | orders g0 |
|---|---|---|---|
| `q0` (declare `NoSource`) | the silent-vacuum defect | **×2653 / ×4376** | `0.004, 0.001` |
| `q2` (double) | the P2′ double-delivery regression | **×2652 / ×4374** | `−0.005, −0.001` |
| `W`-drop (omit `1/W`) | vv mode #3 missing factor | **×2652** | `−0.005, −0.001` |
| `qscale` (`×1.02`) | 2 % magnitude | **×52 / ×87** | `−0.203, −0.054` |
| `qiso` (drop `μ_n B_g`) | **the isotropic-ansatz bug** | **×725 / ×1216** | `0.017, 0.004` |
| `qrev` (reverse Γ₋ rows) | mis-ordered inflow | **×156 / ×213** | `0.011, 0.001` |
| `qswap` (swap group axis) | group-axis bug | **×1592 / ×4097** | `0.007, −0.004` |
| `qface` (xmax gets xmin's trace) | per-face confusion | **×107 / ×147** | `0.020, 0.002` |

Three things this settles:
1. **`qiso` is the decisive row** — it keeps `q ≠ 0` and the correct scalar
   content, changing ONLY the `μ`-dependence, and reds ×725. So the anisotropy is
   *constrained* (`vv` Mode 10 discharged), not merely activated. **That row does
   not exist under an isotropic ansatz** — which is the whole argument for the
   anisotropic one.
2. **Row order is gated by the SOLVE**, not only by an array compare (`qrev`
   ×156). A constant-valued spec would be blind; the angular variation is
   load-bearing, not decorative.
3. **The orders collapse to ~0 under every mutation** because a boundary-source
   error is MESH-INDEPENDENT — it becomes an error FLOOR, so `orders > 1.9` is
   unusually a genuine catcher here. Do not generalise (vv anti-#5 still stands).
4. ⚠ **`W = 2` on Gauss–Legendre ⟹ "drop the `1/W`" and "deliver twice" are the
   SAME mutation** — non-attributable on a 1-D slab (needs `W ≠ 2`, i.e. a
   full-sphere rule). The error message must name both candidates.

`vv` anti-#18 check: no mutation breaks a structural law (each mutated `q` is a
legal Γ₋ element, `B` stays the zero morphism, the matvec stays linear, every
solve converges) ⟹ no red is credited to a broken law.

### L40g — the `InflowSourceSpec` Protocol is evaluated at TWO shapes

`[M]` M2/M3/M5a, instrumented `evaluate` per face:

```
realize_boundary_law(PrescribedInflow(spec), face) → spec.evaluate((N,))        rank 1
AngularBoundarySourceSink.from_mesh_laws(sn)       → spec.evaluate((|Γ₋|, ng))  rank 2
```

| fixture | `N` | `|Γ₋|` | slot | realize | materialise |
|---|---|---|---|---|---|
| slab 1G GL-16 | 16 | 8 | `(16,1)` | `(16,)` | `(8,1)` |
| slab 2G GL-16 | 16 | 8 | `(16,2)` | `(16,)` | `(8,2)` |
| 2-D LD 6×6 N=24 2G | 24 | 12 | `(24,2,6)` | `(24,)` | `(12,2,6)` |
| plain 2-D 4×6, `product(4,8)` | 38 | 13 | x-faces `(38,2,6)`, y-faces `(38,2,4)` | `(38,)` | `(13,2,6)` / `(13,2,4)` |

The rank-1 call is `BoundaryTraceLaw.assert_source_lives_on_incoming_trace`
(`orpheus/geometry/boundary/_base.py:440`), the ERR-047 PRESENCE probe: a
per-ordinate axis over ALL ordinates, no group axis. A spec written only for
`(|Γ₋|, ng)` dies `IndexError: tuple index out of range` **before the MMS runs** —
my first probe did exactly that. Row order: `q[i, g]` lands on trace ordinate
`inflow_indices_for_face(face)[i]`; `xmin → [8..15]`, `mu ∈ [+0.095, +0.989]`;
`xmax → [0..7]`, `mu ∈ [−0.989, −0.095]`, both ascending. Pinned by a per-row
ramp oracle (2-D: `i + 100g + 10000j`, `array_equal = True`, off-inflow `0.0`).

⛔ **And the presence guard is opt-out-able.** Its body opens
`probe = source.evaluate((N,)); if not np.any(probe): return`. `[M]` M5c — a spec
returning zeros at rank 1 and `7.0` at rank 2 realizes cleanly and materialises
`linf = 7.0`; the ERR-047 certification never runs. **A presence predicate a
source can decline is not a guard** (a P6 defect, and a hazard for anyone writing
a spec: return a non-zero rank-1 probe deliberately).

### L40h — ⛔ constraint refuted #2: the sphere is NOT refused

The brief said a declared `PrescribedInflow` is refused on a carrying mesh because
`RadialCharacteristicBoundaryOperator._reflect_corner` raises. `[M]` M4 on
`build_sphere_nonvacuum_mms_case` (`coord=SPHERICAL`,
`radial_characteristic_field_space is not None` — genuinely seed-carrying):
`realize_boundary_law` OK; `from_mesh_laws` OK (`linf = 0.5`);
`solve_sn_fixed_source` OK — while
`_has_ruled_corner_action(PrescribedInflow) = False` and
`RadialCharacteristicBoundaryOperator(sn).is_adjointable = False`.

So the refusal machinery is real but **fires only when the ray-corner forward or
transpose action is INVOKED**, which the fixed-source drivers here do not do. The
genuine restriction is narrower: **the transpose is unavailable**, so an
adjoint/reciprocity row on a declared-prescribed sphere is blocked. **Lesson: "X
raises" read off a producer is a claim about a CONSUMER — measure it at the
consumer.** (The exact shape of the campaign's own `⛔⛔ REFUTED 2026-08-05`
block, one phase later, in the opposite direction.)

### L40i — ⛔ and a probe that "proved the public path works" was discarding my declaration

M4c printed `solve: OK` and I nearly believed it proved end-to-end sphere
support. `[M]` `solve_sn_fixed_source(materials, mesh, quadrature, q, …)`
(`solver.py:2974`) takes a **raw** `Mesh1D|Mesh2D|tuple[Axis1D,...]` and at
`:3095` calls `_as_sn_mesh(...)`, which **constructs its own `SNMesh`** — there
is no `SNMesh` pass-through and no `sn_mesh=` kwarg, so my `sn3.bc[...] = ...`
was silently dropped and the solve ran ALL-VACUUM.

⟹ **there is NO public entry point that can be handed a declared boundary law.**
`SNSolver` is public and takes an `SNMesh`, but its `solve_fixed_source(
fission_source, flux_distribution)` is the eigenvalue-inner interface (no
external composite source). `BC.params : dict[str, float]` cannot carry a
manufactured trace. So a declared law reaches a solve only via the private
`_build_fixed_source_rhs` + `_solve_fixed_source_{si,krylov}` triple — which is
what P3's keystone assembles by hand. That is `:ref:`verification-user-path``'s
SECOND half (test-owned machinery = a GAP SIGNAL), and P4 must DECLARE it in the
module docstring rather than silently reuse the privates. **Method lesson: when a
probe on a public entry point succeeds unexpectedly, check whether the entry
point REBUILT the object you configured.**

### L40j — costs and baselines `[M]`

| item | cost |
|---|---|
| SI 2G declared ladder `[20,40,80]` | **0.064 s** |
| **Krylov** 2G declared ladder `[20,40,80]` | **46.0 s** ⟹ one mesh only (`≈15 s`) |
| Krylov 1G ladder | 7.4 s |
| G7 2-D layout ramp, plain `Mesh2D` + `placeholder_materials(ng=2)` | **0.67 s** (do NOT reach for the LD stress case) |
| the 3 existing gate modules together | `29 passed, 1 deselected in 0.74 s` |
| medium slice `tests/{transport,geometry,sn/operators} -m "not slow"` | **`3 failed, 2205 passed, 5 skipped, 9 xfailed in 42.57 s`** — RE-VERIFIED at `8d552395`, identical to the pre-P3 figure ⟹ P3 added no reds |
| wide slice (campaign figure, not re-run) | `7 failed`, ≈17 m 35 s — run `tests/sn` WHOLE |

Labels: **no new one is needed.** `bc-single-delivery`
(`foundations/boundary_conditions.rst:2037`) is literally `γ₋ψ|_f = q_f`
"exactly, and once" — the keystone's own equation; `bc-composite-source` (`:1947`)
for the channel row; `sn-mms-nonvacuum-{psi,qext,qext-mg}`
(`verification/sn.rst:3610/3711/3749`) for the convergence rows.
⚠ **No second `catches("ERR-075")`** — it lives on P3's
`test_declared_inflow_reaches_the_rhs.py`; a duplicate is an unverified coverage
claim unless the new row is separately mutation-proven against the exact
documented double delivery.

---

## L41 — G6.3 step 7, the DECK-TRANSFORMATION UPLIFT: a type that spells two actions apart still has one method that conflates them

**Carve shape.** `SpatialWrap(axis: str)` + `SelfPairedDeck(motion)` → one
`PairedDeck(motion)` beside `SelfPairedDeck(motion)` (7a), and
`_specular_kernel` (keyed on `quadrature.reflection_index(axis)`) + the periodic
`IdentityOperator() & IdentityOperator()` arm → one `_deck_kernel` keyed on the
rigid motion (7b). Delivered `tests/geometry/test_paired_deck.py` (63 rows) and
`tests/sn/operators/test_deck_kernel.py` (78 rows); spec
`scratch/step7_deck_uplift_verification_plan.md`. Both production halves landed
DURING the dispatch (7a at `e13313a8`), so the plan is a POST-carve
reconciliation with the gate matrix as the audit of record.

### a. ⛔⛔ `RigidMotion.permutes` is the AFFINE action — the briefed contract could not run

`RigidMotion` spells `on_points(x) = Qx + t` and `on_directions(v) = Qv`
separately *precisely so that applying an affine map to a direction is
unwriteable* — its own docstring says so. But `permutes(points)` matches
`on_points(x)` against the set, and the deck kernel is its first consumer whose
motions carry a **translation**. `[M]` 2026-08-07 on `gauss_legendre(8)` and
`product(4,8)`:

| motion | `motion.permutes` | `RigidMotion(motion.linear).permutes` |
|---|---|---|
| mirror-x | `perm` | `perm` (equal) |
| mirror-x seated at 2.5 | ⛔ `None` | `perm` |
| **wrap `t = ê_x`** | ⛔ **`None`** | `perm` = `arange` |
| rotation 90° about z | `perm` | `perm` (equal) |
| rotation 90° **seated** | ⛔ `None` | `perm` |
| glide | ⛔ `None` | `perm` |

So the brief's `pi = motion.permutes(embedded_nodes, atol=1e-13)` is correct for
the specular arm — every element `SelfPairedDeck` admits is origin-fixing,
because its guard FORBIDS the affine part — and **hard-fails on the one arm the
step exists to build**. The blindness is structural: the sibling type's guard
had been hiding the defect by never producing an input that could expose it.

⭐ The main agent found it independently and landed
`Quadrature.ordinate_permutation(motion)` →
`motion3.linear_part.preserves(_embedded_nodes(measure), weights, …)`. The gate
therefore had to be posed on the **OBSERVABLE**, not the spelling: two motions
with the same linear part must produce **bit-identical** kernels (seated mirror
≡ origin mirror; wrap `ê_x` ≡ wrap `17 ê_x`; seated rotation ≡ origin rotation).
That holds for an inline `RigidMotion(m.linear)`, for `linear_part`, or for a
future `permutes_directions`. `[M]` reverting to the affine spelling reds **26
of 78** rows.

### b. ⛔ The brief's nominated keystone fixture was the DEGENERATE one

The brief said "on `product(4,4)`". `[M]` the rotation deck's **local**
permutation there is exactly `[0,1,2,3]` — `arange`, the same shape the periodic
wrap has and the shape a wrong implementation hard-codes (the retired periodic
arm's own comment read *"the body is the IDENTITY on the local index"*).

| fixture | \|Γ₋(xmin)\| | local perm | `== arange`? |
|---|---|---|---|
| `product(2,4)` | 2 | `[0,1]` | ⛔ yes |
| `product(4,4)` | 4 | `[0,1,2,3]` | ⛔ yes |
| **`product(4,8)`** | 12 | `[1,2,0,4,5,3,7,8,6,10,11,9]` | ✅ no |
| `product(4,12)` | 20 | `[2,3,4,0,1,…]` | ✅ no |
| `level_symmetric(6)` | 24 | `[8,9,10,11,4,5,6,7,0,1,2,3,…]` | ✅ no |
| `lebedev(17)` | 49 | `[0..8,14,15,16,13,9,…]` | ✅ no |

`[M]` `local := np.arange(n)` reds 23 rows on the shipped set and would have
reddened only the equivalence rows had the keystone stayed on `product(4,4)`.
The degenerate fixtures were KEPT as a labelled control whose docstring says it
proves nothing about the remap.

### c. ⭐ An error class that manifests only as a REFUSAL needs its own positive gate

π-vs-π⁻¹ is out of range **by a theorem**: `h(Γ₋(f)) = h²(Γ₊(f'))`, which equals
`Γ₊(f')` iff `h² = e` — i.e. only for the involutions, where the two conventions
coincide anyway. `[M]` on all six admissible rotation fixtures the forward
gather lands in `Γ₋(f')` and `to_local` refuses; on `product(4,4)`,
`π⁻¹[inflow] = [3,7,11,15] = Γ₊(ymin)` exactly while `π[inflow] = [1,5,9,13]`.

Consequence: the `π`-for-`π⁻¹` mutation reds 18 rows **entirely by RAISING**, so
per the standing rule it has attributed nothing to the value rows. The repair is
a SECOND, in-range in-class mutation — **reverse the local assignment** (right
set, wrong assignment: the most dangerous silent bug) — under which the same
rows red **by comparing**, 40 of them. Ship both or the value rows' catcher
status is unproven. And the refusal itself earns a positive row
(`test_the_forward_gather_is_OUT_of_range`), which doubles as the keystone's
activation guard.

### d. ⭐ When a defect is closed STRUCTURALLY, the obvious mutation reds nothing

`[M]` replacing `ordinate_permutation` with a bare `argmin` (no injectivity
check) — the textbook ERR-073 mutation — reds **0 of 78**. Reason: the
`Permutation` TYPE asserts bijectivity at construction, so a non-injective match
cannot be returned at all; a bare-argmin body still refuses, one frame in.

⟹ the only mutation that can red an ERR-073 gate deletes
`Permutation.__post_init__`'s clause, and `[M]` it then reds **exactly 1 row**.
Rule: when a defect class was closed by making the illegal state
unrepresentable, **the mutation must target the TYPE's invariant, not the
consumer** — and the marker's docstring must say that, because "earned by
deleting the type's guard" is a stronger and different claim from "the consumer
checks".

Fixture note for the ERR-073 row: duplicate a node **and its mirror partner**
(GL(8) → 10 nodes) so `|Γ₊| = |Γ₋| = 5` and the kernel's extent guard cannot
fire first — otherwise the refusal is attributed to the wrong clause.

### e. ⭐ An introspecting test adapter INFLATES a battery once the signature lands

The SN module opened with `_KERNEL_PARAMS = inspect.signature(_deck_kernel)`,
written when the signature was unknown, filtering kwargs. `_KERNEL_PARAMS` is
computed at **test-module import**, which happens AFTER the plugin installs a
mutation — so every mutation whose wrapper had a `**kw` signature made the
adapter `pytest.fail` and unrelated rows "red". `[M]` M2/M4/M5 first read
**55 / 60 / 55**; true values **23 / 11 / 27**. `vv` anti-#17 in the
**inflating** direction, which reads as *well covered* rather than as a broken
instrument. Retiring the adapter for a direct call (the signature had landed)
was both the elegance fix and the harness fix.

### f. ⭐ Grep where a PARAMETER is READ before designing a mutation around it

The landed kernel returns `PermutationOperator(..., domain=gamma_out_domain.codomain, ...)`
— one source for the restriction's binding — so the `domain_face` argument
reaches only an f-string. `[M]` overriding it to the wrong face changed the
returned binding **not at all** (0 reds) and the wrong-partner defect had to be
injected at the **restriction** instead (27 reds). A parameter that only reaches
a message is not a lever.

### g. ⭐ Bound an exclusivity claim by MEASURING both modules, not by reasoning

The partition law "every rigid motion is admitted by exactly one deck type" was
first written as *"the only row in the tree that can see the two guards drift"*.
`[M]` run over `test_paired_deck.py` + `test_self_paired_deck.py`:

| mutation | reds in the new module | reds in the sibling |
|---|---|---|
| `PairedDeck` threshold `fix >= d` (OVERLAP on the mirror) | P1 (2) + partition (4) | **0** |
| `SelfPairedDeck` guard fully relaxed | **partition only (9)** | 5 |
| `SelfPairedDeck` guard → "is it an involution?" | **partition only (3)** | 2 |

So the honest claim is two-part: exclusive for the **overlap/gap** class
tree-wide, and within THIS module the only row that sees a sibling drift at all.
The stronger reading is false. The docstring now carries the table.

### h. Exclusive catcher relationships, the strongest evidence in the battery

Three single-row verdicts, each a claim no other row can make:

| mutation | reds | the gate |
|---|---|---|
| `domain` ↔ `codomain` **swap** | 11 (all binding rows) | the `is`-identity rows — arithmetic-free, extent-legal |
| periodic realizes through a **numerically identical private duplicate** | **1** | the Mode-11 wrap counter (`== 2`, never `> 0`) |
| delete `Permutation`'s bijectivity clause | **1** | the ERR-073 row |

### i. Tolerances, measured per law

`[M]` witness residual `max|nodes[g] − h⁻¹·nodes[j]|`: **exactly 0.0** for the
mirror (a signed permutation is IEEE-exact) and for the wrap (`on_directions` is
the identity) ⟹ `array_equal`; **≤ 1.11e-16** for the 90° rotation
(`cos(π/2)` round-off) ⟹ `atol=1e-15`, deliberately **not** the kernel's own
`1e-11` node window, which is four orders looser than the code achieves.

### j. Measured costs

`tests/geometry/test_paired_deck.py` 63 rows / 0.34 s;
`tests/sn/operators/test_deck_kernel.py` 78 rows / 0.95 s; pyright 0.
Slice `tests/geometry tests/sn/operators -m "not slow"`: **1921 passed /
15 failed / 28 s** — 3 pre-existing baseline reds + 12 from the main agent's
in-flight `test_b3_domain_narrowing.py` edit; none mine.

---

## L42 — #337, the level-symmetric moment-matched node seed: a "root-find on (0, 1/3)" whose root is not unique, and a keystone that is blind at a third of its own rows

**Dispatch (2026-08-08).** Design the PRE-implementation verification plan for
upgrading `level_symmetric`'s node seed from the project convention
`μ₁² = 4/(N(N+2))` to the moment-matched μ₁ (the published Carlson–Lathrop
LQ_N family). Deliverable `scratch/issue_337_verification_plan.md`. Probes:
`scratch/_probe_337_mu1.py`, `_probe_337_roots.py`, `_probe_337_precision.py`,
`_probe_337_prec2.py` (all read-only on production).

### L42a — ⛔⛔ "root-find over the interval (a, b)" is a construction only if the root is UNIQUE there. Count the roots before accepting the design.

The design of record read: *"μ₁² becomes the root of the FIRST generically-
unsatisfied ladder condition's residual (outer scalar root-find over
μ₁² ∈ (0, 1/3))"*. That reads complete. It is not.

`[M]` a 2500-point sign-change scan of the residual over the whole stated
bracket: **two roots at N = 6, 10, 14, 18** (one at N = 4, 8, 12, 16, 20).

| N | root 1 (μ₁) | min w | root 2 (μ₁) | min w |
|---|---|---|---|---|
| 6 | 0.2666354015 | +0.2469 | 0.4225186538 | **−0.5965** |
| 10 | 0.1893213265 | +0.0708 | 0.3471231056 | **−1.0866** |
| 14 | 0.1519858615 | +0.0130 | 0.3008785706 | **−3.4280** |
| 18 | 0.1293445045 | +0.000175 | 0.2689383416 | **−7.6693** |

Two roots ⟹ `f` has the **same sign at both endpoints** ⟹ `brentq(f, 0, 1/3)`
does not converge to the wrong answer, it **raises before it starts**. So the
briefed construction is literally unrunnable at 4 of 9 orders, and the failure
is loud rather than silent — but only if someone tries it. A plan that shipped
the sentence as written would have sent the implementer to debug scipy.

**Rule.** For any plan sentence of the form *"solve for X on interval I"*,
run a sign-change scan over I and report the root COUNT before accepting it.
If > 1, the plan owes a **selection rule** and that rule owes its own gate — a
two-legged one: (a) the shipped value IS the selected root, (b) the discarded
root is genuinely bad (here: exhibit it and measure its weight negative). Leg
(b) is what makes the rule a *reason* rather than a coincidence.

⚠ And the obvious mutation for such a gate reds **by raising** (taking the
larger root makes the whole construction refuse), which per L31/L25 has
attributed nothing to the consumers it reddens. Only leg (b) is attributable.

### L42b — ⛔⛔ The tolerance for a root-find gate is `noise / slope`, and the slope can collapse 4 orders across a parameter family. A single rtol is then a false red at one end and a dead gate at the other.

The brief proposed `~1e-7`. My own first draft proposed `rtol = 1e-12`.
**Both were wrong, in opposite directions**, and I only found out by measuring
the float64 root-find against a 50-digit mpmath re-solve of the same equations:

| N | float64 vs 50-digit | ULP | residual slope `d(res/want)/d(rel μ₁)` |
|---|---|---|---|
| 4 | 1.59e-16 | **1.0** | 1.03 |
| 6 | 7.08e-15 | 34 | 7.3e-2 |
| 8 | 1.88e-14 | 148 | 7.3e-2 |
| 10 | 2.46e-13 | 1 678 | 9.7e-3 |
| 12 | 3.82e-13 | 2 300 | 7.7e-3 |
| 14 | 4.72e-12 | 25 838 | 1.3e-3 |
| 16 | 3.57e-12 | 17 860 | 9.0e-4 |
| 18 | **8.72e-12** | **40 653** | **1.7e-4** |

`1e-12` is a **false red** at S14/S16/S18. `1e-7` is 4 orders too loose at S18.
The spread is 5 orders of magnitude across ONE family, and it is **derivable**:
`Δμ₁ ≈ (residual evaluation noise) / |dres/dμ₁|`; the noise is the FP floor of
an `N(N+2)`-term positive sum (`κ=1`, measured 3.5e-16…7.7e-15 relative) and
the slope collapses because the constraining condition flattens with N. The
prediction (~4.5e-11 at S18) is within 5× of the measurement.

**Rules.**
1. **Per-order tolerance, tabulated from measurement × 10, decade-rounded** —
   the memory §5 rule ("bit-exactness is EARNED PER LAW") extends to *earned
   per ROW of a parameter family*.
2. **The literal is the arbitrary-precision value, never the implementation's
   own output.** Gating a float64 root-find against its own answer is the
   value-compared-with-itself demotion; gating it against 50-digit mpmath is
   the L34c rule ("state the criterion against the arbitrary-precision value").
3. **Say what the floor leaves UNGATED.** At S16/S18 a sub-1e-10 μ₁ error is
   invisible to G1 (rtol 1e-10), to the residual gate (floor 5.8e-10) and to
   the degree table (L42c) — three independent gates, all blind. That is the
   arithmetic floor of the formulation, not a plan defect, but it must be
   *stated* or it gets discovered later as false coverage.

### L42c — ⛔⛔ The obvious keystone (a derived table) can be IDENTICAL before and after the change at a subset of parameter values. Measure old-vs-new DISTINGUISHABILITY per row before nominating it.

The change's headline consequence is "the achieved polynomial degree rises".
So the degree table looks like the keystone. `[M]` it is not:

| N | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|
| deg NEW | 5 | 7 | 9 | 11 | **11** | 15 | **15** | **17** |
| deg OLD | 3 | 5 | 7 | 9 | **11** | 13 | **15** | **17** |
| distinguishes? | ✅ | ✅ | ✅ | ✅ | ⛔ | ✅ | ⛔ | ⛔ |

**At S12, S16 and S18 the two families reach the same degree.** So the degree
gate is structurally blind to the entire seed change at 3 of 8 orders — and
those are exactly the orders where the value gate's tolerance is loosest
(L42b). The keystone had to move to the μ₁ value plus the *residual of the one
condition the seed is chosen to satisfy*.

**Rule.** Before nominating any derived quantity as a carve's keystone, build
BOTH configurations and tabulate the derived quantity per parameter row. A row
where old == new is a `vv` anti-#20 non-catcher, and a closeout claiming "the
degree table gates this at all eight orders" is false in a way that survives
review because the table *looks* complete.

### L42d — ⛔ I shipped a guessed integer sequence in my own first draft. The obvious continuation of an integer table is not a measurement.

Draft §5 carried orbit counts `{14:6, 16:7, 18:8}` — extrapolated from the
committed `{2:1, 4:1, 6:2, 8:3, 10:4, 12:5}` by continuing "…, 6, 7, 8".
`[M]` the truth is **`{14:7, 16:8, 18:10}`**: the count is `p₃(N/2 − 1)`,
partitions into at most three parts — `1,1,2,3,4,5,7,8,10`. The sequence looks
linear for exactly as long as the committed table shows it.

**Rule.** A numeric table in a plan is a `[M]` claim (`plan-authoring` §2). If
the continuation was not computed, it is not a row — it is a placeholder, and
it must be marked as one or omitted. Cheapest fix: compute the extension in the
same probe that produced the committed part.

### L42e — Answering a "is this margin real?" question with arbitrary precision is cheap, and it converts a blocking user ruling into a closed one.

The new positivity frontier is decided by min-weight `+1.750e-4` at S18 vs
`−2.191e-4` at S20 — a `4e-4` swing in a solve whose greedy row-selection is
`[M]` rank-deficient over **31 %** of the S20 bracket. That is the exact shape
of a float64 conditioning artifact, and I opened it as a blocking user ruling
(U1: "do not ship a sign-of-a-tiny-number frontier without this answer").

Then I ran the whole construction in mpmath at 50 and 60 dps — exact gamma
targets, exact level recursion, exact Gram–Schmidt row selection, exact LU.
`[M]` S18 `+0.0001750014795257`, S20 `−0.0002191020803263`, S22 `−1.5987e-2`,
S24 `−2.2381e-2` — agreeing with float64 to every printed digit, and the
float64 root's margin is `2.3e-11` from the exact one, i.e. the sign has ~7
orders of headroom. **The frontier is a property of the family.** U1 closed,
in one probe.

**Rule.** When a plan's blocker is *"is this small number real or is it
arithmetic?"*, the answer is usually one mpmath re-implementation away. Prefer
answering it to escalating it — and keep the reasoning in the plan (marked
✅ ANSWERED, not deleted) because it is why the check was worth running
(`plan-authoring` §3). Corollary for the shipped gate: pin the MARGIN VALUE
alongside the sign, so a conditioning regression reds before the frontier
silently sheds an order.

### L42f — Grepping the CONCEPT of the change found two production docstrings already present-tense-false from the PREVIOUS issue.

Scoping #337's prose blast radius, I grepped the *concept* ("degree 3",
"doe=3") rather than only the symbol. That surfaced 10 test-comment sites
stating `level_symmetric(4) (O_h N=24 doe=3)` — which #337 makes false (S4
becomes 5) — and, unexpectedly, **two production docstrings already false from
#327, landed two days earlier**:
`orpheus/numerics/measure.py:226` and
`orpheus/numerics/generating_measure.py:126`, both stating in the present tense
that `level_symmetric_sn` "assigns one weight to every ordinate by hand" and
"**is** degree-3 at every order while claiming N−1".

⚠ And one of the ten (`test_keff_2d.py:270`) is not a stale number but a
load-bearing ARGUMENT — *"doe=3 EXACTLY integrates the degree-2 Y₁·Y₁ moment"*
— which survives (5 ≥ 3) but must be re-stated, not find-replaced.

**Rule.** A test-design dispatch's prose grep is also a free audit of the
PREVIOUS change's retirement pass. Grep the concept, sort the hits by tense,
and read the justifying sentence — not just the number in it.

### L42g — A pre-existing red must be characterised, not merely counted, or it masks the change's own reds.

`[M]` baselining `tests/geometry/test_bc_equivalence_snapshot.py` found
`1 failed, 109 passed in 2.10 s`. The failure is
`TestWhiteXminPartial03GLSnapshot::test_matches_the_frozen_scaled_lambertian`
— 8/60 elements, max abs diff `2.22e-16` (**1 ULP**). Critically it is the
**GL** case; both `LS4` and `LS6` cases in that same file PASS. So the one red
in the file #337 will re-baseline is on the ONE rule #337 does not touch.

Recording "1 pre-existing red" would have been useless. Recording *which* row
and *that it is the non-LS rule* is what lets the implementer attribute later
reds without triage, and it produced a concrete instruction: run the file with
`-k "LS4 or LS6"` when attributing, and do NOT absorb the GL red into the
re-baseline (that would hide an unrelated regression inside a legitimate one).

### L42h — Cost: measure the reachable subset; the directory estimate was useless.

`[M]` the four fast gate files are **1.45 s** and **2.10 s**. `tests/numerics/`
as a WHOLE did not finish in 120 s. The battery's per-mutation slice is ~3 s,
so an 11-mutation battery is under a minute — there is no reason to ration it.
A directory-level estimate would have over-stated the relevant cost by
orders of magnitude (memory §2, measured again).

## L43 — Q5.6 "6.3 flip" (cylindrical quadrature ADMISSION): a provenance field that invites the wrong guard, a ∀ nobody can falsify, and a palindrome that eats the obvious gate

**Dispatch.** 2026-08-08, branch `refactor/operator-strategy-layers`. Design
three gate sets for `SNMesh(CYLINDRICAL)` refusing every non-*carrying*
quadrature (the Q5.4 R12a predicate: carrying ⟺ neither `on_edge_node` nor
`degenerate`). Deliverable `scratch/q5_6_3_gate_design.md`. Substrate: the
call-site map `scratch/q5_6_3_flip_call_sites.md`.

### L43a — The tree moved TWICE mid-dispatch, and one whole ask had already landed

Brief pinned HEAD `143e6e2a`. `[M]` at delivery HEAD was `ce6607f5` (two
commits later) with 12 further test files uncommitted-modified. `ce6607f5`
("6.3 leg 1 — the cylindrical MMS builders ride the fold") had **already
shipped ask (b)'s σ_y-parity gate**, re-posed exactly as I was deriving it
(evaluate on the fold's PARENT rule via `dataclasses.replace`, both builders
parametrized, plus a restriction leg). Ask (b) collapsed from *design a gate*
to *audit the landed one* — six residual gaps, which is a far better
deliverable than the gate would have been.

The transient state was itself informative: at 03:0x the module's SECTION
HEADER comment already read "the case builders default to `folded_product`"
while `build_cylindrical_mms_case` still returned `Quadrature.product`. A
prose claim landed ahead of its code. `[M]` no gate in the tree could see that
— which is exactly residual gap **G-b6** (assert `case.quadrature` nodes
`array_equal` `folded_product(n_mu, n_phi)`).

**Rule (reinforces `plan-authoring` §1 / L40a).** Re-`git log` + `git status`
at the START and the END of a design dispatch, and run an existence check per
promised DELIVERABLE, not only per named symbol.

### L43b — ⭐⭐ A type carrying a PROVENANCE field makes provenance-keyed admission the cheapest wrong guard; kill it with a two-sided pincer

`Quadrature.quotient()` stamps `folded_by=group` (`directional.py:443`). So
"admit the fold" has an obvious, wrong one-liner: `if quad.folded_by is None:
raise`. A brief asking only for "a hand-built quotient also constructs" does
NOT kill it — a hand-built quotient is tagged too.

`[M]` the pincer that does:

| fixture | `folded_by` | carrying? | must |
|---|---|---|---|
| `Quadrature(measure=folded_m, level_structure=folded_s)` (hand-assembled, arrays `array_equal` to `folded_product(4,8)`) | `None` | all 4 | **construct** |
| `Quadrature(GL(4)×trapezoid(8, NODE_ALIGNED)).quotient(Mirror("y"))` | `Mirror('y')` | none (`on_edge_node` ×4, N=20) | **refuse** |

Either row alone leaves the tag-keyed guard green; together they are the only
mutation-verified spelling of "admission by STRUCTURE, not provenance".
⚠ Each row must assert its own precondition in-test (`assert quad.folded_by is
None` / `is not None`) or a future tagging change silently turns the control
into a duplicate of its positive (the L2 "build the control explicitly"
discipline).

**Rule.** Before designing a "by structure, not provenance" gate, grep the type
for a field that RECORDS the provenance. If one exists, the gate is a two-sided
pincer, never a single positive.

### L43c — A guard wired AFTER an existing guard: that input is a DISCRIMINATION row, not a negative row

The brief listed slab GL among the families the new refusal rejects. `[M]`:

```
GL level_structure: None
GL level_indices  : [array([0, 1, 2, 3])]     # a FAKE single level
GL march-start facts: (MarchStart(on_edge_node=True, degenerate=False),)   # does NOT raise
GL SNMesh RAISES: ValueError | Cylindrical streaming requires a quadrature with level structure ...
```

Two facts collide: the classifier happily classifies GL (so the predicate would
own it if wired one line earlier), but the structure-less guard at
`reduced_operator.py:826` fires first. So a GL row asserting the NEW message is
a false red, and one asserting the OLD message twins the two committed gates.

The honest row asserts, on the same input, **old fragment PRESENT + new
fragment ABSENT** — i.e. it pins the WIRING ORDER, which is a real
mutation-testable claim (move the helper before `cylindrical_streaming` → reds).
Corollary: the two messages must share no substring, asserted once.

### L43d — ⛔ A ∀-quantifier is ungate-able when no factory produces a MIXED input — factor the pure predicate so the mixed case is synthesizable

`[M]` every refused family has ALL levels non-carrying; every admitted family
has ALL carrying (product/`quotient(NODE_ALIGNED)` → `(True,False)` on every
level; staggered-unfolded/`level_symmetric` → `(False,True)` on every level;
`folded_product` → `(False,False)` on every level). No shipped factory yields
a mixed rule ⟹ `all(...)` vs `any(...)` in the admission agree on every
constructible input. Mode-12 at the FIXTURE, not the functional.

Fix is architectural, and it is the deliverable's strongest instruction: split

```python
def non_carrying_levels(starts: tuple[MarchStart, ...]) -> tuple[int, ...]   # pure; returns POSITIONS (vv anti-#14)
def assert_carrying_quadrature(quad, coord) -> None                          # raises, single-sources the reporting
```

so a synthetic `(carrying, on_edge, carrying, degenerate)` tuple is a two-line
unit test. Without the split the quantifier ships **provably ungated** and the
implementer would be credited for it.

### L43e — ⛔⛔ A symmetric polar rule makes EVERY per-level geometric datum bit-palindromic — the level-REVERSAL is invisible to the obvious coordination gate

The natural gate for "slot key `p` addresses level `p`" is the per-level march
ray cosine the seed march consumes (`psi_half_angle_seed.py:314`:
`s_p = start_cosines[p]`, `two_over_dr = 2.0/(dr/s_p)`). `[M]` on
`folded_product(4,8)`:

```
_march_start_cosines : {0: 0.5083741268536304, 1: 0.9404322888985427,
                        2: 0.9404322888985427, 3: 0.5083741268536304}
mu_start_per_level bit-palindromic: True     alpha_per_level: True
redist_dAw:  True     per-level eta: True    per-level w: True
mu_z per level: [-0.8611, -0.3400, +0.3400, +0.8611]  palindromic: False
level_indices : [[3,2,1,0],[7,6,5,4],[11,10,9,8],[15,14,13,12]]
```

Because GL nodes are ±symmetric, levels `p` and `n−1−p` share **bit-identical**
η, w, α, ΔA/w AND start cosine. So `p → n−1−p` is designed-green for the
cosine functional at every mesh, order and tolerance; only the SIGNED axial
cosine and the ordinate index sets escape the stabiliser. The cosine gate is
still worth having — it catches every NON-reversal permutation (`[M]` `s_p`
swings 0.5084 ↔ 0.9404, 1.85×) — but it needs a partner row on `mu_z`
ordering, with the blindness declared.

**Rule.** For a "the index addresses the right object" gate, list the
per-index data the consumer reads and test each for invariance under the
plausible permutations *before* nominating one as the functional. A symmetric
generating rule makes half the permutation group invisible.

### L43f — ⭐ Count the rows that REACH the assertion, not the rows that exist

`test_gate_source_is_the_space_key_source` opens with
`if not seed_levels: _require(space is None, ...); return`. `[M]` every
cylinder rule yields `radial_characteristic_levels == ()` ⟹ **10 of its 20
cylinder rows return after one check**, and the invariant the test is NAMED for
has only ever run on the 6 sphere rows, each with a ONE-element key set. The
flip does not damage this test — it gives the invariant its first multi-element
key sets. An early-return branch is anti-#20 row inflation wearing a guard
clause.

### L43g — ⛔⛔ An ADMISSION change can make a whole production BRANCH unreachable — grep the predicate's consumers for `if X: continue`

`loss_representation` `:4195-4212` (+ transpose twin `:4661-4678`) is the #280
2.5b direct-seed fold: `if not is_sphere:` … `for p_fold, _level in
enumerate(levels): if p_fold in seed_levels: continue` … else fold, else
`raise NotImplementedError`. `not is_sphere` + curvilinear = cylinder;
post-flip every admitted cylinder has ALL levels carrying (`[M]`
`radial_characteristic_levels == (0,1,2,3)` on `folded_product(4,8)`), so the
`continue` fires every iteration and the branch — refusal included — is
unreachable on every constructible mesh (L43d: a mixed rule cannot be built).

Consequence for test design: `tests/sn/sweep/test_cyl_direct_seed_fold.py`
(6 `foundation` tests) does not need re-quadraturing, its SUBJECT is gone —
4 retire with the fold, 2 (`cold_solve == matvec inverse` / `== SI fixed
point`) are geometry-general and re-pose. Route the production-retirement
decision to the user; the tests cannot stay either way.

### L43h — The refused-family enumeration in a brief is a per-family CLAIM to measure

Brief: "full NODE_ALIGNED product (edge-node + degenerate)". `[M]` it is
`on_edge_node=True, degenerate=False` — ONE fact. Transcribed, that would have
pinned a FALSE reason on every product row (L31's blanket-reason trap). The
measured table (product & folded-NODE_ALIGNED → edge-node only; staggered-full
& level_symmetric → degenerate only) is what makes the "assert WHICH fact
fired" design possible AND makes the two per-conjunct mutations (predicate
reads only `degenerate` / only `on_edge_node`) the pair that proves both
conjuncts load-bearing — exactly half the refused families red under each.

### L43i — Two MMS measurements worth keeping

`[M]` volume-weighted L2(φ), `n_cells=[10,20,40]`, same case object with only
`quadrature` swapped:

```
ANISO  folded_product(4,8)  N=16  [0.009511 0.004418 0.003651] orders [1.106 0.275]
       product(4,8)         N=32  [0.022096 0.019504 0.019116] orders [0.180 0.029]
       folded_product(4,16) N=32  [0.008520 0.002477 0.001007] orders [1.782 1.299]
       folded_product(4,32) N=64  [0.008206 0.002140 0.000619] orders [1.939 1.789]
ISO    every rule                 [0.00885  0.00219  0.000547] orders [2.013 2.003]
```

(a) The fold is a **19× improvement at equal ordinate count** on the #229
azimuthal floor — but it can NEVER become a permanent gate, because post-flip
the comparison arm (`product(4,8)`) refuses at construction. A one-time
migration measurement belongs in the commit, not in a test.
(b) The ISOTROPIC case is a perfect Mode-7 NULL for the ξ channel — identical
to 3 s.f. across every rule, folded or not. An isotropic-only migration check
proves nothing about the fold; the anisotropic row is the only catcher for the
ξ-parity class (`[M]` parity deltas: honest `0.0` exactly; `ξ²→ξ` `0.0254`
(rel 0.36); `η→ξ` `0.0852` (rel 1.21–1.46)).

### L43j — The landed parity gate's six residual gaps (the audit shape)

Worth keeping as a template for auditing a gate somebody else just landed:
(1) tolerance not earned (`atol=1e-15` where `[M]` the delta is **exactly 0.0**
and bit-exactness is a float THEOREM: `eta[π]==eta`, `xi[π]==-xi`, and IEEE
`(-a)*(-a)==a*a`); (2) the node-level preconditions ungated, which also leaves
a Mode-12 hole — `[M]` the manufactured source contains **no `mu_z` term**, so
`Q[π]==Q` holds for the σ_z pairing too, and the gate's instrument legs
(permutation + no fixed points) cannot discriminate; (3) the docstring claims
parity for `psi_ref`, which is **signature-tautological** (`psi_exact(r,
eta_n)` cannot receive ξ) and doubly false for the isotropic case, which has no
`psi_exact` at all; (4) the negative control runs on a DIFFERENT fixture
(NODE_ALIGNED) than the positive leg (STAGGERED); (5) parent and case coupled
by coincident defaults, so a default change reds by RAISING (`IndexError`),
attributing nothing; (6) nothing asserts the builder ships the fold.

---

## L44 — #340 N6 commit 1 (the truncation warning re-sourced onto `first_failure`): a "move three parameters" brief that is actually "re-point eight reads", and a keystone whose leaf fixtures make it a theorem

Dispatch 2026-08-10, HEAD `52650a86`. Design-only (a full-tree pytest run was
live; project lesson L37 — never edit `.py` under a live run). Deliverable
`scratch/n6_verification_plan.md`.

**The defect.** `orpheus/sn/solver.py:455` — `if history.converged: return`
inside `_warn_if_unconverged`, the single `ConvergenceWarning` emission point.
`IterationHistory.converged` is the OUTERMOST level; `fully_converged` is the
tree. A converged outer standing on a truncated inner returns in silence, from
a function whose own docstring (`:403-406`) says a truncated solve "announces
itself once, in one voice, wherever it came from". Not a regression — the guard
has read `converged` since the campaign's first commit, and N2b-ii minted
`fully_converged` without re-pointing its one consumer.

### L44a — ⛔⛔ A re-sourcing carve's blast radius is the READS, not the PARAMETERS — and a PARTIAL re-point is worse than none

The brief named **three** facts moving from hand-passed arguments to
`record.first_failure`: `budget`, `tolerance`, `knob`. `[M]` the consumer body
reads the top-level record at **five more sites**:
`:473 binding_criterion` (which feeds `distance`, `rate`, `projected` and the
`cleared` branch), `:493 min_iterations`, `:496/:498/:501 n_iterations`.

Implement the brief literally and the emitted message welds the INNER's knob
and budget to the OUTER's criterion, rate and projection:

> `hit max_inner=20 without reaching tol=1.000e-10 (last dphi 6.308e-02) …
> rho=0.153338 … set max_inner=17`

— every number real, every pairing wrong. That is strictly worse than today's
coherent-but-mis-levelled message, because it *looks* level-correct.

⟹ **Before designing gates for "source X from Y instead of Z", `awk` the
consumer's body for every read of Z.** The elegant carve is ONE named
intermediate (`failing = history.record.first_failure or history.record`,
Pattern 3 ∩ 2); the gate set then needs the two PARTIAL mutations (re-point
budget/tol/knob only; re-point criterion only) as separate battery rows,
because a single "did it re-point" mutation cannot distinguish them.

Corollary for the plan's shape: the deliverable opened with a `⛔ READ FIRST`
correction table (`line | read | today's source | must become`) rather than
burying it — the brief's three-fact framing is the thing a fresh implementer
would code to.

### L44b — ⛔ A LEAF fixture makes a re-sourcing gate an `X == X` THEOREM by OBJECT IDENTITY, not by coincidence

`[M]` all four committed audibility fixtures route through
`solve_sn_fixed_source`, a 1-LEVEL tree: `first_failure is record` returns
**`True`** (object identity, `convergence.py:856` `return None if self.converged
else self`). So after the carve, six of the seven message facts are the same
object's attributes read twice — **no input exists** that could make the two
sourcings differ. The leaf rows are not "under-tested" and not "sub-floor";
they are annihilated (`vv` anti-#22 / the `coding-standards` single-sourcing
demotion, arriving through `is` rather than through a call).

The ONE fact a leaf row can still see is whatever genuinely CHANGED SOURCE
KIND — here the knob name, which moved from a caller literal to a record field.
`[M]` two committed leaf gates (`:378` `"max_inner=50" in msg`, `:393`
`f"set max_inner={projected}"`) therefore become free catchers for a forgotten
producer stamp, and that is the entirety of their new coverage. Say so in the
docstring or an audit counts them as coverage of the re-sourcing.

### L44c — The two-tier fixture split, and the synthetic tier is the KEYSTONE

- **Tier A, synthetic (`~0 s`)**: hand-build a two-level `IterationRecord` and
  call the private emitter directly (three committed gates already do this).
  Choose EXACT geometric trajectories (`3.7e-2·0.9^k`, `1e-2·0.5^k`) so `rho`
  and `projected_iterations()` are analytically determined — `[M]` inner
  `rho=0.900000 / proj 189`, outer `rho=0.500000 / proj 25`. All seven facts
  pairwise distinct by construction.
- **Tier B, end-to-end (`[M]` 0.18 s)**: `solve_sn` 2-D `(2.0,1.5)/(4,3)`
  all-reflective, `get_mixture("A","2g")`, LS-S4, `max_outer=3, keff_tol=
  flux_tol=1e-12, max_inner=20, inner_tol=1e-10`. `[M]` outer TRUNCATED 3/3
  binding `dphi` (proj 17, rho 0.153338); all three children TRUNCATED 20/20,
  `first_failure = children[0]` binding `residual` (proj 586, rho 0.962217).
  Tier A cannot see that production NESTS or that the SN layer stamps the
  child (`vv` Mode 11).

⭐ **Leg 0 of the keystone is an ACTIVATION table**, asserting each inner fact
`!=` its outer counterpart. Without it a future fixture edit that collapses one
pair silently disarms that leg forever (Mode 12 at the fixture). And
**recompute** `rho`/`projected` from the record inside the test — hard-coding
`189` is a false red waiting on `_RATE_FIT_TAIL_FRACTION`; the literals belong
in leg 0, not in the assertion.

⛔ The brief's own d=3 fixture (`(1.0,2.0,3.0)/(3,4,5)`, `max_inner=200`) is
equally discriminating and `[M]` ~200× more expensive. Measure a cheaper
equivalent before adopting the brief's.

### L44d — ⛔ `hasattr`-check the campaign's OWN mutation harness before planning a battery on it

`[M]` `scratch/mutate_convergence_contract.py` is a TRACKED file and **cannot
import at HEAD**: line 24 binds `REAL_CLAIMS = sv._claims_convergence` at
module scope, and the campaign retired `_claims_convergence` on 2026-08-09
(`hasattr(...) is False`). Two further staleness modes in the same file, both
now anti-#18 over-powered: `ma1` constructs `PowerIterationOutcome(converged=
True)` — a keyword a sibling gate asserts cannot exist — and `ma4`/`ma5`
declare replacements with the pre-carve signature, so after the carve their
reds are `TypeError` crashes, not property reds.

⟹ a campaign that retires symbols **breaks its own instruments by module-scope
binding**, silently, because nobody runs the harness between carves. Repairing
it is part of the commit, not cleanup: `lessons` §2 — no negative verdict is
trustworthy until the control passes, and here the control cannot even load.

### L44e — Run the guard-flip mutation BEFORE writing the gate that catches it

`[M]` predicted (and stated as the gate's justification): flipping `:455` to
`fully_converged` reds **nothing** on today's tree. The ≥11 solves with
`converged=True, fully_converged=False` would all begin emitting, and
`pyproject.toml` sets no `filterwarnings`, so they stay green; the only
silence gate (`:370`) rides a leaf where the two properties coincide. The
measurement, not an argument, is what earns the scope-boundary gate — and it
must be paired with an `xfail(strict=True)` naming the commit-2 end state, so
the deliberate flip is self-announcing while the accidental one is caught now
(the positive gate is INVERTED by commit 2; say so in its first docstring line).

### L44f — `vv` anti-#16 for a new field: enumerate the CONSUMERS' knob spellings, not the producers'

The new field names the caller-facing budget knob. The obvious guard —
whitelist `{max_inner, max_outer}` — breaks a real caller: `[M]`
`GreenOperator` drives `SourceIteration` and its own knob is `max_iter`
(`green_operator.py:277`). Same family as the `GreenOperator(tol=0)` refusal
this campaign already shipped and reverted (`convergence.py:398-407`). The
field takes a free `str`; the ONLY thing it may refuse is `""`, mirroring the
existing `label` invariant. The "is it a real knob?" claim is expressible only
as a TEST over the public entries (`inspect.signature(entry).parameters`) —
three committed gates pass `where="probe"`, so a production check would red
them.

⚠ And the membership check alone is blind to a SWAP: `[M]` `solve_sn` exposes
BOTH `max_outer` and `max_inner`, so SI-stamps-`max_outer` passes a membership
test. Ship the exact-expectation leg too (`lessons` §6, OPTIONAL→MANDATORY
BINDING: design the battery around the swap).

### L44g — Side-finding: `exhausted_budget` is in the WRONG UNITS on the Krylov path

`[M]` `solve_sn_fixed_source(inner_solver="krylov", max_inner=5)` on the d=3
absorber returns `budget=5, n_iterations=732, status=CONVERGED` — so
`exhausted_budget` is `True` on a converged solve. `budget` is scipy's
`maxiter` = restart CYCLES (`iteration.py:1055`, `restart=n_dof`), while the
trajectory counts CALLBACKS (≈ cycles × n_dof). `truncated` is saved only by
`converged`. If a Krylov inner ever truncates, the advice reads
`hit max_inner=5 … set max_inner=<hundreds>` — the projected count and the
budget are in different units. Found by measuring a candidate fixture, not by
reading; a plan dispatch that measures its own fixture cost gets these for free.

### L44h — ⛔⛔ ANOTHER AGENT'S MUTATION IS INDISTINGUISHABLE FROM A PRODUCTION BUG

The implementation half of this dispatch, 2026-08-10. I read
`_warn_if_unconverged` at the start and saw `if history.converged:`. Twenty
minutes later a fixture sweep produced `converged=True` **with a warning
emitted** — which is impossible under that guard. I was one step from
reporting "the landed commit has a live guard bug" when I ran
`inspect.getsource` and found:

```python
if history.fully_converged:  # M8 PROBE — REVERT ME
```

The coordinator was running MY OWN battery's M8 pre-measurement, in the
shared tree, at `[M]` mtime 05:59:29 — after my read, before my sweep.

Three durable rules:

1. **When a behavioural anomaly contradicts source you have READ, re-read
   the source with `inspect.getsource` before believing the anomaly.** A
   `sed`/`Read` snapshot is a point-in-time claim exactly like a plan's; on a
   shared tree it rots in minutes. The failure mode is *reporting someone
   else's deliberate mutation as a defect*, which is worse than a missed
   finding because it burns the reviewer's trust.
2. **Bracket every measurement with the state of the thing being mutated.**
   `echo GUARD BEFORE …; pytest …; echo GUARD AFTER …` costs one line and
   converts an ambiguous number into an attributable one. That is how the
   M8 datum below became usable rather than discardable.
3. ⭐ **A collision can be a GIFT — take the measurement it hands you.** The
   probe was live during a baseline run I had already taken, so re-running
   the suite bracketed gave M8 for free: `[M]` **`130 passed, 1 warning in
   23.36s`** with the guard flipped, against the coordinator's `130 passed
   in 25.11 s` un-flipped ⟹ **M8 reds 0 of 130**, exactly the prediction,
   measured rather than argued. That measurement is the entire justification
   for the commit-1 scope gate.

### L44i — The battery, measured: predictions vs reality (14 rows, 2 runs)

`[M]` 2026-08-10, `scratch/mutate_convergence_contract.py` (repaired), suite
= the contract file + both numerics record files, `150 passed, 1 xfailed`
baseline. Identical results before and after a mid-flight refactor of the
gate code, which is itself the reproducibility check.

| mutation | predicted | measured | verdict |
|---|---|---|---|
| M0 control (unmutated) | 0 | **0** | instrument alive |
| MC warning suppressed | ≥5 | **12** | positive control |
| M1 no navigation | keystone + nested | **2** | as predicted |
| M2 criterion left on top | the tol/rate legs | **2** | ⚠ same 2 TESTS as M1 |
| M3 knob+budget left on top | the knob legs | **2** | ⚠ same 2 TESTS as M1 |
| M7 `first_failure` self-first | keystone + nested + record gates | **7** | as predicted |
| M4 SI+Krylov forget the knob | leaf msgs + knob sweep | **15** | as predicted |
| M5 **SWAP** the two knobs | exact-leg only | **10** | ⭐ see below |
| M6 `power_iteration` forgets | eigenvalue rows only | **6** | leaf untouched ✓ |
| M8 guard → `fully_converged` | scope row + xpass | **2** | + the xfail XPASSes |
| M9 drop the empty-knob guard | the field's negative leg | **1** | exact |
| MR re-add a defaulted `tol=` | the signature row | **1** | exact |
| MN drop the `or record` fallback | **0** (unreachable) | **0** | labelled non-catcher |
| MX `first_failure := None` | many, via `AttributeError` | **13** | anti-#18 example |

Two findings worth carrying:

- ⚠ **M1/M2/M3 are indistinguishable at TEST granularity** — all three red
  the same two rows, because both gates assert every leg. The *leg-level*
  predictions were right; the *test-level* ones were over-specified. The
  discriminating evidence is the emitted MESSAGE, captured in the smoke test:
  M2 prints `inner(within-group) hit max_inner=30 … tol=1.000e-09 (last dk
  3.125e-04)` — the inner's level and knob welded to the OUTER's criterion.
  When a plan predicts distinct red SETS for partial mutations of one
  function, check whether any single gate asserts all the legs; if it does,
  the sets collapse and the honest prediction is "each partial reds, and the
  message distinguishes them".
- ⭐⭐ **The SWAP prediction landed EXACTLY, including its blind spot.** Under
  SI↔power_iteration knob swap the *membership* leg (`budget_name in
  inspect.signature(entry).parameters`) reddened **only** the two
  fixed-source rows — whose entries have no `max_outer` to be wrong about —
  while all four eigenvalue rows (`solve_sn`, `solve_sn_adjoint`, which
  expose BOTH knobs) were caught by the exact-expectation leg and by nothing
  else. That is the L37 "design the battery around the swap" rule paying out
  in a measured table, and it is the reason a membership-only gate would have
  shipped a false coverage claim on 4 of 6 rows.

### L44j — Two production additions my PRE-carve spec did not predict, both gate-worthy

A pre-carve spec enumerates the facts it knows; the implementer legitimately
adds more. **Diff the landed function against your own spec's fact list
before writing the gates** — I found two additions, each needing a row my
plan never contemplated:

1. **A level PREFIX** (`level = "" if failing is history.record else
   f"{failing.label} "`) — the message now names the failing level when it is
   not the top. The discriminator is an `is` identity, so no value comparison
   substitutes for it, and without a LEAF row asserting the prefix is ABSENT
   the prefix could be emitted unconditionally with the whole suite green
   (the tree row asserts only presence). Classic positive-without-negative.
2. **A dropped `without reaching tol=` clause** when the level has no
   criteria at all (MoC's fixed sweep count) — `[M]` `probe: hit max_inner=4
   (no criterion recorded) … Raise max_inner`. This is also the only cheap
   way to reach the `rate is None` advice branch, which my spec had listed as
   a blind spot (B5) reachable "only synthetically" — correct, and the
   synthetic row costs ~0 s, so the blind spot closes.

### L44k — pyright caught MY OWN stringly-typed dispatch; the principled fix removed it

`[M]` my first cut of the knob-stamping sweep parametrized over `(entry_name:
str, inner_solver: str | None, expected)` and dispatched with an
`if entry == "…"` chain plus a `**kw` splat built from an untyped dict.
**24 pyright errors**, all from the splat: pyright cannot tell which
parameter an untyped `dict[str, str]` lands on, and it guessed `scheme`.

The `# type: ignore` reflex would have hidden a REAL elegance defect
(anti-#4: a tag plus a branch chain is a missing type). The principled
spelling — a table of `(id, thunk, entry_function, expected)` where the row
IS the call and every argument is explicit — took 24 → 0 AND made the
membership reference read the signature of the very callable the row invoked,
instead of a `getattr` by name. Test code earns the same standard as
production; run pyright on new test modules before delivering them.

Side-findings from the same run: the COMMITTED file had **10** pyright
errors, 9 of which the N6 migration cleared for free (the retired kwargs);
the 10th is a **mis-placed `# type: ignore[call-arg]`** — the suppression sat
on the `PowerIterationOutcome(` line while pyright reports at the
`converged=True` argument line, so the file had been carrying an unsuppressed
error. One-line move, fixed.

### L44l — Attributing an out-of-scope red: audit the DIFF for arithmetic, don't guess

The wider run (`tests/numerics` + `tests/sn/solve`, `-m "not slow"`) came
back `[M]` **3 failed, 2375 passed in 514.69s** — three golden-sha
bit-identity rows in `test_affine_carve_bit_identity.py`. The cheap, rigorous
attribution is **not** "run it before and after" (forbidden here — the tree
carries uncommitted state and `git checkout` is banned); it is:

`git diff -U0 orpheus/ | <strip comments> | grep -E '=|def |raise |return '`

— the complete list of added/removed CODE lines. For N6 that is a string
field, its emptiness guard, three `budget_name=` plumbing parameters, and the
warning's message assembly. **Not one line touches a flux, a matrix, a
residual, or any arithmetic on the solve path** ⟹ provably numerics-neutral
⟹ the reds are pre-existing. Corroborating: both 2-D rows use
`Quadrature.level_symmetric(sn_order=4)`, the family #337 re-seeded, and the
goldens are inline sha256 literals last touched by a rename commit — the
Signature-10 stale-snapshot shape. Do NOT re-baseline them from inside an
unrelated campaign; that is #337's re-baseline sweep, and re-freezing from
here would hide an unrelated regression inside a legitimate one.

---

## L45 — #340 R5 (retiring the imperative `pytest.xfail`): four of five exclusions had ALREADY healed, and not one of them by the fix it cited

**Dispatch.** #340's convergence-honesty campaign, step R5: convert the 5
imperative `pytest.xfail(reason)` call sites in `tests/` to the declarative
`@pytest.mark.xfail(strict=True)`. The defect (now `vv` Mode 8, NINTH class) is
that the function-call form raises `XFailed` at that line, so the row reports `x`
unconditionally and **can never report `XPASS`** — the day the blocker is fixed,
nothing says so. The brief's governing instruction was the right one: **measure
each row with the imperative call neutralised BEFORE converting**, because a row
that now passes must be *un-excluded*, not wrapped in a strict marker that reds.

### a. The instrument, and why it needed to assert its own installation

`scratch/_r5_noop_xfail.py` — a `-p` plugin doing `pytest.xfail = _noop` in
`pytest_configure` (safe: the test modules do `import pytest` and call
`pytest.xfail(...)`, a late attribute lookup; the skipping plugin's own internal
`xfail` is `from _pytest.outcomes import xfail`, a different binding, so it is
untouched). Per §2's harness rule it does not print a banner and hope: it
accumulates every neutralised reason and at `pytest_sessionfinish` prints the
COUNT plus each reason, with an explicit *"ZERO neutralisations — this run's
verdict is NOT trustworthy"* line. Every run below reported the expected count
(2, 1, 1, then 4 in the combined run) with the expected reason strings, so no
verdict rests on an instrument that silently failed to bite.

### b. The verdicts — 4 of 5 healed

| row | site | condition | verdict | margin |
|---|---|---|---|---|
| 1 | `test_apply_full_field_codomain.py:229` | `sphere ∧ krylov` (case **2eg**) | **PASSES** | `rel = 1.184e-14`, gate `rtol=1e-10` |
| 2 | `test_removal_form_matvec_sweep.py:483` | unconditional stub, **empty body** | still blocked | n/a |
| 3 | `test_kinf_homogeneous_tolerance.py:144` | `sphere ∧ 4eg` | **PASSES** | `rel = 4.462e-13`, gate `< 1e-8` |
| 4 | `test_kinf_homogeneous.py:163` | `sphere ∧ 4eg ∧ krylov` | **PASSES** | `rel = 3.582e-15`, gate `rtol=1e-10` |
| 5 | `test_kinf_homogeneous.py:229` | same (spectrum) | **PASSES** | eigenvector, gate `rtol=1e-9` |

### c. ⛔⛔ The cited fix was NOT the fix — and it is still open

All four healed rows said the same thing: *"unpreconditioned GMRES on sphere-4g
exceeds the `max_inner=N` budget without converging. Issue #200 tracks the
block-inverse face preconditioner that re-enables Krylov."* **#200 is still
OPEN** and `_within_group_krylov` still ships `preconditioner = lambda q: q`. So
the natural cheap audit — "has the cited issue landed? no ⟹ still blocked" —
returns the WRONG answer on 4 of 4 rows. Only running the row answers it.

What healed them is the GMRES `restart`-sizing lineage:

* **ERR-053** (2026-05-28) removed `restart=min(50, full_size)`, which truncated
  the Krylov subspace on any mesh above 50 unknowns. Its catalog entry does the
  arithmetic on *this* fixture family (sphere, N=8, 2g → 328 unknowns vs 50).
* **#282 / #280 route (a)** (2026-07-04) sized `restart` from the FULL augmented
  ravel (bulk ⊕ trace ⊕ ψ½-seed). Its gate
  `test_krylov_restart_covers_augmented_composite` states the symptom in words:
  the bulk-only count left *"the poorly-conditioned curvilinear-eigenvalue
  inner"* stalling and *"at a realistic outer cap it returns a WRONG keff"*.

I could not discriminate which was decisive (that needs a production revert; the
dispatch forbade editing `orpheus/`), and **said so** rather than picking the
prettier story. The attributable claim — the one that changes behaviour next
time — is that the cure lives in the restart sizing, not the preconditioner.
Exclusions stamped `R-1 Step D (2026-05-19)`; first cure nine days later; they
survived **eleven weeks** past it.

### d. The reason strings were false a SECOND way, and this one is generalisable

`max_inner` is not an iteration count on the krylov path. It becomes scipy's
`maxiter`, which counts **restart CYCLES**, and `restart = n_dof` (`[M]`
`n_dof = 112`, sphere-2eg). One cycle therefore spans the whole Krylov space, so
**no value of `max_inner ≥ 1` can truncate this fixture** — `[M]` the same
sphere-4eg krylov solve at `max_inner=2` returns the `max_inner=1000` answer,
`rel = 3.582e-15` both. The budget named in the exclusion was never a live knob
on the path being excluded. (Second, independent confirmation of the units
side-finding already recorded for N6.) Generalisation: **when a deferral's reason
names a NUMBER, check that the number is a knob on the path it names** — a
budget in the wrong units makes the reason unfalsifiable, which is the same
defect class as the immortal xfail.

### e. The corroboration needed its own control, and the obvious control was impossible

The campaign's own failure mode would be a row that passes because a truncated
inner solve was never reported. Check: all four pass under `-W
error::orpheus.numerics.convergence.ConvergenceWarning` (the DOTTED spelling —
the bare name does not resolve, Mode 8 EIGHTH class). But a positive reading
alone carries zero information about whether the instrument can bite (`vv`
anti-#19), and **the natural control could not be built on the path under test**
— per (d), the krylov path cannot be truncated by `max_inner` at all, so
`max_inner=2` produced no warning *and no error*, which reads exactly like a dead
instrument. Resolution: move the control to the SIBLING path where the knob IS
iterations, and say which path it covers.

| leg | config | `rel` | warnings |
|---|---|---|---|
| positive control | **SI**, `max_inner=2`, `inner_tol=1e-12` | `1.139e-02` | `['ConvergenceWarning']` |
| negative leg | **SI**, `max_inner=1000`, `inner_tol=1e-12` | `6.492e-14` | `[]` |

So the warning machinery and the capture are live and a truncated inner moves
`keff` by `1.1e-2` — the krylov rows' silence is informative. Stated honestly:
the control covers the SI path, not the krylov path, because on the krylov path
the truncation state is unreachable.

### f. ⛔ An unconditional stub with an EMPTY body cannot be converted literally

Row 2's whole body was the `pytest.xfail(...)` call. A strict marker over an
empty body **XPASSes ⟹ FAILS** immediately, so the brief's "unconditional stub →
`@pytest.mark.xfail(strict=True)`" is unrunnable as written. The conversion has
to SUPPLY a body, and the design constraint is Mode 8 FOURTH class: exactly one
statement may fail, and it must be the documented reason.

Its blocker is genuinely live, verified against the TREE and not the reason
string: `orpheus/sn/solver.py` exposes `solve_sn`, `solve_sn_adjoint`,
`solve_sn_adjoint_fixed_source`, `solve_sn_fixed_source` and none takes a
removal-form / σ_r-fold parameter; `grep -rn "removal_form" orpheus/` → 0. (The
`ScatteringOperator.foldable_part` / `foldable_sigma` machinery exists at the
OPERATOR tier; no solver consumes it.) #200's body does stack *"the σ_r foldable
preconditioner (Adams & Larsen 2002 §III)"* after the block-inverse, so the
citation is legitimate — checked with `gh issue view`, per the
never-resolve-an-issue-from-a-bare-`#N` rule.

Body: a **concept-level capability probe** — any public `orpheus.sn.solver`
`solve_*` name or `solve_sn` parameter whose spelling contains `removal` or
`fold`. Grep the WORDS the issue itself uses rather than guessing one symbol
name; state the residual fragility in the helper's docstring (if #200 lands under
neither word the row will not flip) with a grep pointer. `None` ⟹ `pytest.fail`
with the documented reason; otherwise **fall through and pass**, so `strict=True`
converts the landing into a FAILURE that forces the stub to be written. A comment
marks the fall-through as deliberate so nobody "fixes" it with an assertion.

Both mandatory checks run: `--runxfail` reds with *"no removal-form solver entry
exists yet (#200)…"* and nothing else; and a flip control
(`scratch/_r5_flip_control.py`, injecting `solve_sn_removal_form` **in-process** —
never editing `orpheus/`) produces `FAILED … [XPASS(strict)] removal-form k_inf
recovery needs the #200 solver entry…`. The marker is a proven self-retiring todo,
not a hope.

### g. Blast radius: the healed row's DOC claims go present-tense-false

R5 was briefed "entirely a `tests/` change". It could not be. The retirement
audit's third search (grep the CONCEPT across code, tests AND docs) found
`docs/theory/methods/sn/curvilinear_multigroup.rst` §"Gotcha" calling sphere-4G
*"the xfail'd cell of the coordinates × groups × drivers grid"* **and naming the
file I had just changed** — plus the same page's bullet list and its "Two real …
interactions exist" intro. Leaving them is exactly `vv` anti-#21's aggravated
state: the stale claim and its correction coexisting in one tree, each citable.
Rewritten as past-tense history carrying the measurement and the correct
attribution; the live interaction reordered first, the retired one kept *because
the reason it outlived its cure is the transferable lesson*.
`loss_representation.rst:574` ("it is `xfail` until the #200 solver entry exists")
was still TRUE — sharpened to name the strict form rather than deleted.

### h. Two audit hygiene notes worth reusing

* **`grep -rn "<concept>" docs/` hits `docs/_build/`** — 6.4 MB of built HTML and
  a `searchindex.js` that matches everything. Always `--include="*.rst"` (or
  exclude `_build`) or the audit output is unreadable and the real hits are
  buried.
* **pyright over the changed test modules, measured on BOTH sides**: `29 errors`
  now, `29 errors` on the HEAD versions placed as siblings via
  `git show HEAD:<f> > <dir>/_r5head_<f>` (same directory ⟹ identical import
  resolution; never `git checkout`). All pre-existing (`keff: float | None`
  arithmetic). Name the copies so pytest does NOT collect them (`_r5head_test_*`
  does not match `test_*.py`) and delete them after.
* **A generated artifact can be stale at HEAD and your build "changes" it.**
  `docs/theory/verification/matrix.rst` regenerated `9491 → 9492` under my `-W`
  build; `git show --stat` showed commit `3f76d651` added 59 lines of tests
  without regenerating the matrix. My four modules collect **75** before and
  after. Report it as a pre-existing staleness the build corrected — do not
  silently own the delta, and do not revert a correct regeneration.

## L46 — #340 N4.7 (one warning helper, four families): `stacklevel` is a claim about EVERY call site's DEPTH, and the obvious gate is blind to half of it

**Dispatch, 2026-08-11.** Brief: design the verification plan for a carve
"I am implementing right now" — move `sn/solver.py:_warn_if_unconverged` into
`numerics/convergence.py` as a public family-agnostic
`warn_if_unconverged(record, *, where, balance_defect=None)`, re-point 5 SN
sites, add 3 new ones (`solve_cp`, `solve_moc`, `solve_diffusion_1d`).
Deliverable file `scratch/n4_7_verification_plan.md` (659 lines).

### a. The carve landed DURING the dispatch — again, and it moved twice

`[M]` `git status` + `grep -rn warn_if_unconverged orpheus/` at ~04:20 and
04:34: the first read showed the helper in `convergence.py` but the SN sites
still on the old private name and ZERO cp/moc/diffusion sites; the second
showed all eight sites in place and `_warn_if_unconverged` gone (grep rc=1).
By 04:50 the SN TEST surface had migrated too
(`tests/sn/solve/test_convergence_contract.py`, +61/−36). ⛔ **One of my first
three measurements — a `tests/diffusion -W error` baseline reading
`113 passed` — was taken PRE-carve and looked exactly like a post-carve
baseline.** Only the bracketing (`grep -c` before AND after each run) made it
attributable. Re-measured post-carve it is `113 passed` again, but that was
luck, not method. This is the third campaign in a row where the ask moved
mid-design (L39, L43a, now L46).

### b. ⛔⛔ THE FINDING — `stacklevel=3` was already false at 2 of 8 sites

`stacklevel=3` means frame 1 = the helper, 2 = the public entry, 3 = user
code. `[M]` per-entry, calling each from a probe file and reading
`warnings.catch_warnings(record=True)[0].filename`:

| entry | enclosing `def` of the call | attributed to |
|---|---|---|
| `solve_sn` / `solve_sn_adjoint` / `solve_sn_adjoint_fixed_source` | the public entry | the probe file ✅ |
| **`solve_sn_fixed_source` [SI]** | **`_solve_fixed_source_si` (private)** | **`orpheus/sn/solver.py:3541`** ⛔ |
| **`solve_sn_fixed_source` [krylov]** | **`_solve_fixed_source_krylov`** | (same depth ⟹ `solver.py:3552`) ⛔ |
| `solve_cp` / `solve_moc` / `solve_diffusion_1d` | the public entry | the probe file ✅ |

`solver.py:3541` is the entry's own `return _solve_fixed_source_si(` dispatch
line. Reproduced at `max_inner ∈ {1,2,5,50}` on two fixtures — not
config-dependent. **Pre-existing** (the old helper had the same
`stacklevel=3` and the same two private call sites), so the carve inherits it
rather than causing it — but the carve's stated goal is "audible at EVERY
public entry", so it is in scope.

⭐ **The whole class is invisible to every gate that reads the MESSAGE.** The
message is a pure function of `(record, where, balance_defect)`; the
attribution is a second observable that no substring assertion can see. `[M]`
`grep -rn stacklevel tests/` = **1 hit**, an unrelated `warnings.warn` in a
regression helper. The production docstring even says the precondition is
"Gated, per family, rather than left to convention" — and the gate did not
exist.

### c. ⭐⭐ And the OBVIOUS gate is blind to half the error class (Mode 12)

The natural gate is *"the attributed filename is not under `orpheus/`"*.
`[M]` mutating `stacklevel=3 → 2` blames `cp/solver.py:994` /
`diffusion/solver.py:471` ⟹ that gate reds. `[M]` mutating `3 → 4` blames
`mutate_n47.py:109/111/113/120` — the caller's CALLER — which is **still
outside `orpheus/`**, so the gate stays **GREEN** while the warning points at
the wrong line. The functional "is this path inside the package?" has an
invariance group containing every over-deep stacklevel. The second leg —
`w.filename == __file__ and w.lineno == (inspect.currentframe().f_lineno + 1
recorded immediately above the call)` — is the only thing that sees it, and
it is only writable where the entry is called DIRECTLY in the test body.

### d. ⭐⭐ A guard-WIDENING needs a CHILD-starved fixture per family, or the battery is blind to the whole widening

N6b widened the emission guard from `record.converged` (top level) to
`record.fully_converged` (the tree). `[M]` re-installing the OLD guard as a
mutation produces an exact partition: the two rows whose failing level is a
CHILD go silent (`cp-gauss-seidel max_inner=2`; `cp_sph1D_4eg_4rg` at the
default), the two OUTER-starved rows still warn (`cp-jacobi max_outer=3`,
`diffusion max_outer=3, keff_tol=0.0`). So a starvation battery built only
from outer-starved fixtures is a provable non-catcher for the widening it is
credited with. Diffusion structurally CANNOT supply a child-starved row (its
inner is one LU back-substitution, `budget == 0`) — say so rather than
counting its row as coverage (vv anti-#20).

### e. The census instrument — two failures, both in the flattering direction

An entry-wrapping pytest plugin reading the PRODUCTION predicate
(`result.record.fully_converged`) is the right instrument for "which tests
will newly warn". Two ways it lied:

1. ⛔ It reported **`0 entry calls` on `tests/diffusion`** — it wrapped only
   `orpheus.diffusion.solver.solve_diffusion_1d` while
   `tests/diffusion/test_solver.py:34` imports from the **PACKAGE**
   (`from orpheus.diffusion import solve_diffusion_1d`). Fix: rebind every
   module in `sys.modules` whose attribute `is` the original object (6
   bindings, printed and asserted `>= 3` at configure time). A "0 found" from
   a half-installed instrument reads as good news.
2. ⛔ Its `(*args, **kwargs)` wrapper **BREAKS a committed gate**:
   `tests/_harness/predicates.py:reachable_knobs` reads
   `inspect.signature(entry)`, so under the plugin the knob rows fail for
   instrument reasons (4 extra reds). The L41e lesson in the other direction —
   there an introspecting TEST adapter inflated a battery; here an
   introspecting instrument breaks an introspecting REFERENCE.

**Decoder cross-check that made the census trustworthy:** on `tests/moc` its
DISTINCT node-id count is **24** and the `-W error` run is **24 failed**.

### f. The adjudication — 53 candidate reds, all INCIDENTAL, and the CP fix is free

`[M]` under `-W error::orpheus.numerics.convergence.ConvergenceWarning`:
`tests/diffusion` **113 passed** (zero), `tests/moc` **24 failed / 100 passed**
(100 s), `tests/cp` (peierls ignored) **29 tests** by census,
`test_family_convergence_contract.py` **5 failed**.

- **CP: one cause — `CPParams.max_inner = 100` (production default) against
  its own `inner_tol = 1e-8`.** Sufficiency measured 132 / 203 / 280 / **403**;
  `|k − k_inf|` IMPROVES (`9.8e-9 → 1.5e-9`) and the wall clock does **not**
  grow (`0.24 s → 0.22 s` on the sphere — a converged inner buys fewer
  outers). A shipped default that cannot meet its own tolerance is a
  production finding, not a fixture one.
- **MoC: one cause — the test harness's `_quick_solve(n_inner=15)`** against a
  production default of 200. ⛔ **200 is not enough for the heterogeneous
  rows**: outer-1's inner from a cold boundary flux needs **~493** sweeps
  (`rate 0.971`; converges in 491 at 600), and the record's own
  `projected_iterations()` predicted 493 — 0.4 % accurate. Cost is real here:
  the 3-spacing L2 het test goes **18.8 s → 88.9 s (4.7×)** at 200 alone.
- ⭐ **The highest-stakes row was measured, not assumed.** An L2
  convergence-RATE claim whose every refinement level is truncated at a fixed
  sweep cap is Signature 8 (the cap, not the stencil, produces the rate). `[M]`
  it is NOT that here: `k` moves `5.9e-9` between `n_inner=20` and `200` at all
  three spacings and `diff_1`/`diff_2` are identical to 4 s.f.

### g. H3 (message bit-identity) — how to capture a "before" on an uncommitted tree, and why it cannot be a gate

`git show HEAD:orpheus/sn/solver.py` → `ast.parse` → extract the
`FunctionDef` → `exec` into a namespace holding only the four names it closes
over → call BOTH on the SAME input. **No `git checkout` anywhere.** `[M]` over
five message shapes (leaf-with-balance, leaf-without, nested-starved-inner,
non-contracting, no-criteria-at-all) `difflib` reports **exactly one edit
region** in every case — the advice clause. ⚠ It cannot become a permanent
gate: HEAD moves, and one commit later it compares the new text with itself.
Run once as ACCEPTANCE; the permanent successors are the per-clause message
rows already committed.

### h. Design critique that survived

`balance_defect: float | None = None` is the right SHAPE (it belongs to the
returned ITERATE, not the iteration, so it cannot be a record field without
breaking "nothing is stored"), and a `BalanceDefect` NewType would be
over-minting (one realization, only `f"{x:.3e}"` applied). But ⛔ **the DEFAULT
makes "this family has none" indistinguishable from "this entry forgot"** —
five SN sites pass it, three omit it, and only ONE of the eight has a
behavioural catcher. Four AST lines close it. The free-function-vs-method
rejection is right, and the strongest argument is neither purity nor I/O: **a
method would need `where` and `balance_defect` as arguments anyway**, so it
buys nothing — and it would put the verb on every object in `numerics/` that
holds a record, dissolving the "two emission points, enumerable" invariant.
The decorator shape (`@announces_truncation`) is the ONLY one that fixes the
stacklevel by construction, and it is rejected because extracting the record
would have to branch on the return TYPE (elegance anti-#4).

## L47 — the SN curvilinear ANGULAR-CLOSURE seam: a single-source producer with no value gate, and a τ gate set whose whole symmetry group contains the error class

Dispatch 2026-08-11, branch `refactor/operator-strategy-layers`, PRE-numerics
experiment. Ask: gate `orpheus/sn/sweep/pole_angular_closure.py` —
`angular_cell_edges_per_level` (the one partition producer),
`morel_montry_tau_per_level` (P2, BMC Eq. 43), the march-orientation sign, and
half-angle-flux (ψ̂) positivity. Delivered: `tests/sn/sweep/test_angular_cell_partition.py`
(NEW, 9 gates / 56 rows), `tests/sn/sweep/curvilinear/test_psi_half_positivity.py`
(NEW, 4 gates / 19 rows), and a widen+re-derive of
`tests/sn/sweep/curvilinear/test_tau_producer_equivalence.py` (5 → 14 rows).
13-mutation battery over a 298-row scope, un-mutated control clean.

### a. ⭐⭐ THE HEADLINE — three committed τ properties, one symmetry group, and the error class inside it

The tree gated τ with exactly three properties, all on the fold:

* P3 membership `τ ∈ [0, 1]` (`_assert_tau_within_unit_interval`);
* the box `τ ∈ [¼, ¾]` (`test_tau_arc_wellposedness::test_the_folded_tau_is_bounded_with_the_reversal_identity`);
* the reversal identity `τ_m + τ_{M−1−m} = 1` (same test).

**All three are invariant under `τ → 1−τ`.** The box is symmetric about ½; the
membership interval is symmetric about ½; and `(1−τ_m) + (1−τ_{M−1−m}) = 2 − 1 = 1`.
And `τ → 1−τ` is EXACTLY the march-orientation flip — measuring the ordinate's
barycentric position from the DOWNSTREAM edge instead of the upstream one, i.e.
the plausible index-drift `(mu[m] − mu_edge[m])` → `(mu_edge[m+1] − mu[m])`.

`[M]` mutation M8 (`τ → 1−τ`) reddened **0 of the 4** box/reversal rows and
6 of 298 rows overall — all six τ-VALUE rows (the closed form is signed).
A design-time Mode-12 stabiliser computation would have found this before any
mutation: the gate set's invariance group is `{id, τ↦1−τ}` and the threat is in it.

⭐ **The catcher is a SIGNED law and it is bit-exact on both arms:**
`(τ_m − ½)·μ_m ≥ 0`, with equality **exactly** where the radial cosine is zero.
Measured over sphere GL N ∈ {2,3,4,5,6,8,10,12,16,20,24,32,48,64} and cylinder
`folded_product(4, n_φ)` for n_φ ∈ {4,6,8,10,16,18,32}: `min((τ−½)·μ)` is
strictly positive at even N/M and **exactly `0.0` or `−0.0`** at odd N/M (where
one node sits at μ = 0 / ω = π/2), so `>= 0.0` passes with NO tolerance
(`-0.0 >= 0.0` is True in IEEE-754). At the zero cosine, `|τ − ½| ≤ 1.11e-16`.
Under M8 the gate reds **12 of 12** rows. It needs an ACTIVATION leg
(`max((τ−½)·μ) > 0`) because `τ ≡ ½` makes every product zero.

⚠ **Do NOT use `np.sign` equality.** At odd M the middle ordinate has
`np.cos(np.pi/2) = 6.12e-17 > 0` while `τ − ½` is `0.0` or `∓1.11e-16`, so
`array_equal(sign(τ−½), sign(cot ω))` is FALSE at n_φ = 6 and 10 and TRUE at
8 and 16 — a parity artefact that reads as a real disagreement.

### b. ⭐⭐ The brief's headline NUMBER was regime-conditional, and the regime it named is not the production path

Brief: *"`[M]` the angular recurrence produces `min ψ̂ ≈ −77` on a normalised
cylinder problem under the shipped convention."* Measured on a heterogeneous
2-region 2-group cylinder (`get_mixture("A"/"B","2g")`, 12 cells, vacuum outer,
`folded_product(4, n_φ)`, converged fixed source, `inner_tol=1e-12`):

| seed regime | n_φ=6 | n_φ=8 | n_φ=16 | n_φ=32 |
|---|---|---|---|---|
| converged ψ + **MARCHED** ψ½ state (the production VALUE path) | **+0.1337** | **+0.1286** | **+0.1287** | **+0.1286** |
| converged ψ + ZERO seed (the ψ-independent COEFFICIENT path) | −12.09 | −16.35 | −25.89 | −39.47 |
| random \|ψ\| + ZERO seed | −48.2 | **−76.9** | −132.6 | −133.7 |

⟹ the `−77` is the random-ψ + zero-seed row at n_φ = 8. **On the path every
value-carrying matvec and sweep actually takes, ψ̂ is strictly POSITIVE**, and
within 12 % of `min ψ` itself. The sign of ψ̂ is a property of the SEED's
consistency, not of the scheme. A positivity claim that does not name its seed
is not a claim.

The mechanism is exact: the recurrence `ψ̂_{m+1/2} = (ψ_m − (1−τ_m)ψ̂_{m−1/2})/τ_m`
has amplification `−(1−τ_m)/τ_m` per step, and its fixed point at flat ψ is
`ψ̂ = ψ`, so a consistent seed starts near the fixed point.

### c. ⭐ Pin the MECHANISM, not the observation — `A(M) = max_m ∏_{k≤m}(1−τ_k)/τ_k`

Solve-free, a pure function of the τ chart, identical on every level (τ is
sinθ-independent), with a structurally independent reference (build it from the
analytic closed form instead of production τ). `[M]`

| M | 2 | 3 | 4 | 5 | 8 | 9 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|
| A | 2.414214 | 2.732051 | 3.359161 | 3.656876 | 4.728870 | 4.976120 | 6.679689 | **9.443672** |

The `9.44` at M = 32 is the same number `vv-principles` #24b quotes as the
ω-midpoint chart's "recurrence error-amplification" — it had lived only in a
skill and a plan. Sphere: 1.639121 / 2.015931 / 2.510677 / 3.145681 / 3.952582
at N = 4/8/16/32/64.

⭐ **Companion identity, exact and explanatory:** `∏_k (1−τ_k)/τ_k = 1` (`[M]`
residual ≤ 8.6e-14), because the reversal identity makes `(1−τ_k) = τ_{M−1−k}`
— the numerators are the denominators re-ordered. So the recurrence is
*neutrally stable across a whole level* while amplifying `A(M)×` in the middle.
⛔ That leg is INVARIANT under `τ → 1−τ` (every ratio inverts), so it is a
non-catcher for the orientation flip and must say so.

### d. ⛔ A parametrize argument list runs at COLLECTION — a raising mutation then reports ZERO failures

My first draft built `(label, cosines, taus)` tuples in the `parametrize`
argument list, calling `morel_montry_tau_per_level` at module import. Six of
thirteen mutations make the P3 guard raise one frame downstream of the
partition, so the module failed to COLLECT and pytest reported
`Interrupted: 1 error during collection` — `FAILED=0`, `rc=2`, in 2 s. Read
off a summary line that is **"0 caught"**, in the flattering direction, for
`MC` (the positive control!), `M1`, `M2`, `M3`, `M5`, `M6`.

Two rules: **(1)** never call production in a `parametrize` argument list —
parametrize by a LABEL and build inside the body; **(2)** put
`--continue-on-collection-errors` in the battery and count `^ERROR` lines
separately from `^FAILED`, so a collection kill can never be read as a green.
Same family as L41e (an introspecting adapter that runs at import, after the
plugin installs the mutation).

### e. ⛔ A palindromic-weight mutation on Gauss–Legendre is a PROVABLE non-catcher

`mu_edge[n+1] = mu_edge[n] + w[n]` → `+ w[N-1-n]`. GL weights are symmetric, so
the cumulative ladder is bit-identical: `[M]` **0 of 298** rows red, `d(edges) = 0.0`.
No gate in the tree can catch it and none should claim to — it is a Mode-12
stabiliser of the RULE's own symmetry (the same shape as L43e's bit-palindromic
per-level data). Record it as a refuted candidate; do not "fix" it with a gate.

### f. ⛔ A flat `atol` is wrong in BOTH directions — derive it from the amplifier

`test_tau_producer_equivalence`'s sphere row carried `atol=1e-13` with a
docstring warning *"a new row at N ≥ 64 must widen it"*. `[M]` at N = 64 the
measured gap is **2.247e-13** — the row was simultaneously ~450× too loose at
N = 8 and a **false red** at the very order its own docstring predicted.

Derive instead, from what the quantity divides by:

* **sphere** — τ divides an `O(ε)` edge discrepancy (pairwise `cumsum` vs the
  producer's sequential loop, which IS the independence) by the cell width, and
  the narrowest GL cell narrows with N ⟹ `atol = 16·ε/w_min`. `[M]`
  `gap ÷ (ε/w_min)` = 0.00 / 0.81 / 0.80 / 0.19 / 1.81 / 3.81 at
  N = 4/8/16/32/64/128 ⟹ ≥ 4× margin everywhere.
* **cylinder** — `τ − ½ = ½cot(ω)tan(Δω/4)` and near the arc endpoints `cot`
  has condition `csc²ω ≈ (2M/π)²` while `tan(Δω/4) ≈ π/4M`, so an `O(ε)`
  `arctan2` error reaches τ times `≈ M/2π` ⟹ `atol = 40·M·ε`. `[M]`
  `gap ÷ (M·ε)` = 0.25 / 0.17 / 0.25 / 0.40 / 0.44 / 0.28 / 1.27 / 2.09 / 3.20 / 7.99
  at M = 2/3/4/5/8/9/13/16/32/64.
* **and the EDGE agreement is M-INDEPENDENT at ≤ 1.5 ULP** (both sides are one
  `cos` of an O(1) angle) — so the partition gate takes `atol = 8ε` while its
  own τ consumer needs `40Mε`. Two quantities, two mechanisms, two tolerances.

### g. ⛔ "Bit-identical at the degenerate fixture" is usually 1 ULP — and `assert_array_equal` on it is a self-inflicted red

The brief (correctly in substance) warned that `folded_product(·, 4)` is M = 2
per level and the ω-midpoint and chord partitions are bit-identical there.
`[M]` they are **3.11e-17 / 5.76e-17** apart in edge space and **1.11e-16** in
τ space — because `np.cos(np.pi/2)` is `6.12e-17`, not `0.0`, while the chord's
midpoint of `±η₀` is exactly `0.0`. My first draft asserted `array_equal` and
went red on my own control. The honest control asserts the separation is
**15 orders below** the `1e-1`-scale signal, not that the bits match.

⭐ The control EARNS its place: it reds under M5/M6/M7 (partition mutations that
move the M=2 edge) and stays green under M4 (which makes production BECOME the
chord) — exactly the semantics "one of the two conventions moved".

### h. The negative controls, both arms, and the derived floors

* **sphere** — the naive `linspace(-1, 1, N+1)` equal-width partition. `[M]`
  edge gap 0.1521 / 0.1764 / 0.1908 / 0.2020 at N = 4/8/16/32; τ gap 0.159 /
  0.646 / 1.525 / 3.206. Without it the cumulative-weight row certifies the
  arithmetic and carries **zero** information about the convention (vv #19).
  The sphere row had no such control before this dispatch.
* **cylinder** — the retired chord partition. In EDGE space the gap SHRINKS
  like `M⁻²` (`[M]` measured ÷ predicted `sinθ(1−cos(Δω/2))` = 0.500 / 0.707 /
  0.809 / 0.924 / 0.940 / 0.981 / 0.995 / 0.999 at M = 3…64) ⟹ any fixed floor
  becomes a false red; assert `0.4 ×` the predicted floor. In **τ** space the
  gap GROWS (0.0317 → 0.0832), so the committed `> 1e-3` is safe there — two
  opposite scalings for the same convention difference, one per space.

### i. Measured coverage delta (the deliverable's own acceptance number)

Scope: 298 rows (this module + `test_tau_arc_wellposedness` +
`test_march_start_structure` + `tests/sn/sweep/curvilinear/` +
`test_mms_ordering_blindness` + `test_cylindrical_quadrature_admission`, minus
three files costing 244/70/53 s). Un-mutated: **298 passed, 0 failed, 25–34 s**.
(The same scope WITH `slow` and the three heavy files is **24 m 53 s** —
measure before rationing.)

| mutation | pre-existing catchers | new catchers |
|---|---|---|
| MC uniform partition (POSITIVE CONTROL) | 29 | 46 |
| M1 sphere seed `-1 → 0` | 33 | 30 |
| M2 sphere half weight | 33 | 30 |
| M3 sphere accumulation dropped | 21 | 21 |
| M3b sphere palindromic weights | **0** | **0** (refuted candidate) |
| M4 chord partition (the Q5.6.4 regression) | **6** | 27 |
| M5 `sinθ` dropped | 22 | 41 |
| M6 arc endpoints swapped | 23 | 42 |
| M7 interior edge at ¾ω_m + ¼ω_{m+1} | **6** | 32 |
| M8 `τ → 1−τ` (orientation flip) | **6** | 34 |
| M9 `τ ≡ ½` (the plain angular diamond) | 7 | 37 |
| M10 recurrence `/(1−τ)` | 24 | 6 |
| M11 recurrence drops `(1−τ)` | 25 | 3 |

The three partition-specific mutations (M4/M7/M8) each had **6 pre-existing
catchers of 298**, and for M4/M7 four of the six are the SAME parametrized box
row. Nothing asserted the partition; every catcher read τ, which is P2 applied
to the partition.

### j. Fixture facts worth carrying

* `Quadrature.folded_product(n_mu, n_phi)` gives `M = n_phi/2` per level, so
  **odd M is reachable and legal**: n_φ = 6/10/14/18 ⟹ M = 3/5/7/9. The
  committed τ ladder was n_φ ∈ {8,16} — a single congruence class (vv #13's
  refinement-ladder sharpening).
* `Quadrature.gauss_legendre(N)` accepts **odd N** (3, 5), which puts a node at
  μ = 0 exactly — the equality case of the orientation law on the sphere arm.
* `Solution.radial_characteristic` is `Optional` and is the marched ψ½ member;
  the closure wants its `.interior` (`RadialCharacteristicInteriorFlux`). Guard
  the `None` with `pytest.fail`, not a `# type: ignore` — pyright flags it and
  a silent `None` collapses the marched-seed row onto the zero-seed one.
* `precompute_psi_state` takes `(N, ng, nx)`, NOT the `(N, ng, nx, 1)` its own
  comment claims (`psi_g_first[:, level_idx, :]` would be 4-D).
* A heterogeneous 2G cylinder fixed-source solve at n_φ ≤ 16, nx = 12 costs
  **0.3–0.4 s** — cheap enough for a per-row solve.

---

## L48 — #235, designing the instrument SUITE for a curvilinear angular-differencing scheme: six dead instruments, and the seventh death mode is "it scores the incumbent zero"

`[M]` 2026-08-12, branch `refactor/operator-strategy-layers` @ `bea6a367`.
Deliverable `scratch/q68_angular_instrument_design.md` (746 lines). Probes
`$CLAUDE_JOB_DIR/tmp/{harness,fixture,p1_basis,p2_spectral,p4_family,p5_psi_metric,p6_scan,p7_affine,p8_final,p9_recommend}.py`.
Prior art: `.claude/plans/archive/q64_tau_partition_memo.md`,
`scratch/q65_endpoint_defect_findings.md`.

**The brief.** Five instruments were already dead (τ-blind ×3,
chart-relative, reference-limited) and a sixth (the endpoint defect `D`)
had just been killed for being uncorrelated with accuracy. The ask was an
instrument SUITE, a mechanical acceptance protocol, a discriminating
fixture, and cost.

### L48a ⛔⛔ A SEVENTH death mode, and it is the flattering one: the instrument whose ZERO SET is the incumbent

`[M]` the closure `ψ_m = τψ̂_+ + (1−τ)ψ̂_−` is EXACT on `span{1, μ}` —
`4.44e-16` cylinder `folded_product(4, 8/16/32)`, `8.88e-16` sphere
`gauss_legendre(8/16/32)`, and **`0` exactly** for the isotropic field —
*because* τ is DEFINED as the barycentric coordinate in the radial cosine
(BMC Eq. 43 / R&L Eq. 13c). So any instrument whose trial field is affine
in the radial cosine scores the incumbent **zero** and every alternative
positive, at every order, on every material.

Three live proposals share that zero set:
* **#319's diffusion-limit instrument** — the diffusion limit's angular
  content IS `span{1, μ}` (`J_φ ≡ 0` under the σ_y fold), so it is a
  provable non-catcher for the ANGULAR closure. (It remains sound for the
  SPATIAL closure — scope the rejection.)
* **the shipped `build_cylindrical_anisotropic_mms_case` ansatz**
  `ψ = A(r) + B(r)η` — affine in η at every `r` ⟹ `[M]` closure residual
  `6e-17` ⟹ **the fixture is in the incumbent's kernel and cannot rank it.**
* **§9bis.6's η-weighted P1 closure defect** — same zero set; its own
  published row already reads `6.94e-16` and was read as a win.

And a fourth, **R&L Eq. 15/16 `|τ−½|/w`**, has zero set `τ ≡ ½` — i.e. it
is one *candidate* wearing a criterion's clothes, not a measurement over
all of them.

⟹ **the check is one line of algebra and no run: solve
`instrument(candidate) = 0` for the candidate.** Promoted into
`vv-principles` anti-#24 as clause **(d)**, the ZERO-SET check.

### L48b ⛔⛔ The graded FUNCTIONAL is a design choice with its own stabiliser — an INTEGRATED one can rank garbage above production

Every gate in `tests/sn/verification/mms/` grades `‖φ_h − φ_exact‖`, and
`φ = Σ_n w_n ψ_n`. `[M]` same solves, fixture `harmonics=(1,2,3)`,
`σ_t=5`, `folded_product(4,64)`, `nx=320`, all `converged=True`:

| scheme | scalar-flux L2 | vs shipped | angular-flux L2 | vs shipped |
|---|---|---|---|---|
| shipped | `4.047e-5` | — | `1.561e-4` | — |
| τ jitter 2 % | `3.909e-5` | **0.97× (blind)** | `3.249e-4` | 2.08× worse ✓ |
| reversed (`τ→1−τ`) | `1.417e-5` | **0.35× ("better")** | `1.781e-4` | 1.14× worse ✓ |
| shuffled/A (GARBAGE) | `2.533e-5` | **0.63× ("better")** | `2.651e-3` | 17.0× worse ✓ |
| shuffled/B (GARBAGE) | `2.055e-5` | **0.51× ("better")** | `1.251e-3` | 8.0× worse ✓ |

Dynamic range over a 12-member ensemble: **`2.1×` (φ) vs `40×` (ψ)**.

**The mechanism is exact, not statistical.** The per-cell closure defect
for a reversal-compatible τ is `∝ cos(mω_m)`, and the discrete azimuthal
moment `Σ_n w_n cos(mω_n)` vanishes for every `0 < m < n_φ` — **the very
identity that makes the manufactured `φ_exact = A(r)` closed-form and
quadrature-independent is the identity that annihilates the defect in the
graded quantity.** The convenience that buys the analytic reference is
what buys the blindness.

⟹ **grade the un-integrated field when one exists.** `[M]` measured
signed/absolute `w`-weighted defects, `folded_product(4,16)` level 0:

| variant | `m=0` | `m=1` | `m=2` | `m=3` | `m=4` |
|---|---|---|---|---|---|
| shipped | `0`/`0` | `7.6e-17`/**`1.7e-16`** | `+8.24e-2`/`8.24e-2` | `1.4e-16`/`2.09e-1` | `+1.52e-1`/`3.89e-1` |
| `τ≡½` | `0`/`0` | `1.1e-16`/`2.69e-2` | **`−7.3e-17`**/`1.09e-1` | `6.9e-18`/`2.36e-1` | **`2.2e-16`**/`4.53e-1` |
| reversed | `0`/`0` | `2.1e-17`/`5.38e-2` | **`−8.24e-2`**/`1.63e-1` | `9.7e-17`/`3.29e-1` | **`−1.52e-1`**/`5.60e-1` |

Three stabilisers in one table: `m=0` is zero for EVERY τ (a convex
combination reproduces constants — the mechanism behind L43i); every ODD
`m` has a machine-zero SIGNED sum for every reversal-compatible τ; and
`τ≡½` has a machine-zero signed sum at EVERY `m`.

### L48c ⛔ Closure-EXACT is NOT accuracy-optimal — a closure-residual instrument is ANTI-correlated with accuracy here

`[M]` shipped ansatz, `nx=320` (angular-isolated), `n_φ=64`: the
closure-exact scheme reads `5.71e-5` (ψ) / `7.17e-5` (φ) and `τ≡½` —
nonzero defect at every ordinate — reads `1.76e-5` / `9.15e-6`, i.e.
**3.2× / 7.8× better**, monotonically along the whole
`blend(w) = (1−w)τ_ship + w·½` homotopy. The closure error partially
CANCELS the α-redistribution truncation. ⟹ the whole family of
"how big is the local closure residual" instruments is a DIAGNOSTIC, never
a ranker.

### L48d ⭐ "Make the fixture harder" is the WRONG axis — the τ-independent floor can grow faster than the signal

L40b again, in a new form. `[M]` ψ-metric reversal resolution
(`shipped/reversed`), `n_φ=64`, `nx=320`, over an 8-fixture scan:
`m1` **`1.00×` (BLIND)** → `m1+m2` `1.33×` → `m2` only `1.32×` →
`m3` only `1.01×` (blind again) → `m1+m2+m3` **`1.05×`** (diluted back).
Adding the third harmonic makes it WORSE. Amplitude saturates too:
`c₂ = 0.25/0.5/1/2/4` gives `1.15/1.32/1.44/1.50/1.52×`.
⟹ **the axis is PARITY, not frequency**: add ONE even harmonic, stop.

### L48e ⛔ The brief's headline number carried a confounded `nx` — half the reported non-discrimination was configuration

The brief's table was at `nx = 80`, where `[M]` the spatial floor is
`≈1.3e-4` for every scheme (the memo's own §4.3 measured this). Spread
over 12 schemes: **`2.7×` at `nx=80, n_φ=32` vs `9.2×` at `nx=320`**;
`1.8×` vs **`18×`** at `n_φ=64`. The `τ↔1−τ` blindness is real and
survives `nx=320`; the compression on top of it was not.
⟹ apply the confound ladder (refine every OTHER axis) before quoting a
"the fixture stopped discriminating" number — including one inherited
from a plan that measured the confound itself two sections earlier.
⭐ And the check is also a COST optimisation: `[M]` the ψ metric on the
recommended fixture is spatially clean already at `nx = 80` for
`n_φ ≤ 32` (`+0.99 %` / `+4.76 %` vs `nx=320`), needing `nx=160` only at
`n_φ=64` (`nx=80` is `+26 %`) — 8× cheaper than the φ metric's `nx=320`.

### L48f ⭐ A CONTINUOUS homotopy beats "rank agreement" as the acceptance criterion

The natural protocol ("correlate against an independent accuracy measure
over a spread of schemes, require rank agreement") is sound and is the
LAST, most expensive gate. Cheaper and strictly stronger: build a
one-parameter family `blend(w)`, `w ∈ {0,¼,½,¾,1}`, and require the
instrument to be **MONOTONE in `w`** — falsified by a single non-monotone
triple, 5 solves. `[M]` it was decisive: the scalar-flux metric is
monotone in `w` on every fixture tested and the angular-flux metric is
NOT on the even-harmonic ones, which says the two are not measuring the
same thing *before* any correlation is computed.
Companion sharpenings: **stratify** the ensemble (NEAR = a 2 % jitter,
MID = the genuine rival, FAR = garbage) and require the threshold on
NEAR∪MID alone — a ρ over all three is dominated by the garbage split,
which is exactly how `D` scored `r = +0.75` at `n_φ=8` and `+0.06` at 32;
and the ensemble MUST contain the pair inside the stabiliser you fear
(`{τ, 1−τ}`), reported as its own line.

### L48g ⭐⭐ The suite's most valuable output can be "this comparison is BELOW MY RESOLUTION" — with the number

`[M]` best fixture, best functional (`harmonics=(1,2)`, `σ_t=5`, ψ-L2,
`nx=80`), resolution against the shipped scheme:
garbage **17–40×** ✓ · a 2 % τ jitter **2–4×** ✓ · the march-orientation
reversal **1.25–1.6×** ✓ (up from `1.00×`) · ⛔ **`shipped` vs `diamond
τ≡½` NOT resolved** — `1.34×` at `n_φ=16`, `1.10×` at 32, `0.95×` at 64,
**the sign flips with `n_φ`**, and the φ metric on the same solves says
diamond wins at every order.
⟹ the two candidates the campaign spent two attempts arguing about sit
INSIDE the instrument's resolution, so Q5.6.4's resolution — decide on
Tier-0 constraints plus the primary source — was the only sound route,
not a fallback. Say this with the number; it is a finding, not a caveat.

### L48h — the suite's shape, and where the load actually sits

Four tiers, and only ONE ranks:
* **Tier 0 CONSTRAINTS** (solve-free, closed-form, DISQUALIFYING):
  admissibility `τ∈[0,1]`; non-amplification `A(M)=max∏(1−τ)/τ`; the
  signed orientation law `(τ−½)μ ≥ 0` (the ONLY committed law outside the
  `τ→1−τ` stabiliser, L47a); the α-dome closure; ψ̂ positivity.
* **Tier 1 DIAGNOSTICS** (τ-loaded, forbidden from voting): the spectral
  closure-defect matrix, the endpoint defect `D`, R&L `|τ−½|/w`.
* **Tier 2 THE RANKER** (one): angular-flux L2 vs a harmonic-rich analytic
  MMS, angular-isolated mesh, stratified ≥8 ensemble, reporting the error
  CONSTANT `K = ‖e‖M²` (not the order — `[M]` every reversal-compatible
  scheme is asymptotically 2nd order, so order discriminates nothing).
* **Tier 3 CORROBORATION**: the independent method, admissible only with a
  published reference self-convergence ladder whose acceptance test is the
  L49 sign flip.
Cost: whole suite ≈ **2 min** per candidate; the pre-flight that kills
most bad instruments is **< 1 s** — `[M]` 4 of the 6 dead instruments die
at A1/A2/A6 without a single solve.

### L48i — ⛔ conservation/balance is τ-blind BY CONSTRUCTION, and now with the mechanism

`Σ_m [α_{m+½}ψ̂_{m+½} − α_{m−½}ψ̂_{m−½}]` telescopes to
`α_{M+½}ψ̂_{M+½} − α_{½}ψ̂_{½}`, and `[M]` `α_{1/2} = 0.000e+00`,
`α_{M+1/2} = 2.78e-17`. **τ never appears** — balance holds identically
for a random τ, any ψ̂. It is a constraint on the α recursion and the
quadrature, worth nothing about the closure. (`vv` anti-#8 with the
mechanism spelled out for this seam.)

### Method notes worth keeping

* **τ override for a full solve**: monkeypatch
  `orpheus.sn.sweep.pole_angular_closure.morel_montry_tau_per_level`
  (single production consumer at `pole_angular_closure.py:1360`, read by
  module-global name). Re-solve every variant to its own fixed point.
* **A garbage τ can raise `ConvergenceCertificateError`** (`τ ≡ 0.30/0.40`
  at some `(n_φ, nx)`): *"the within-group solve claimed convergence …
  but the honest equation residual is 2.851e-10"*. Catch it in a scan
  harness; it is a Tier-0 robustness signal, not a harness bug.
* An MMS fixed-source solve costs `0.35 s` at `nx=80`, `0.83 s` at
  `nx=320` on this host — cheap enough for a 12×3 ensemble in ~1 min.
* The general harmonic ansatz is `h_m = Re[(η+iξ)^m]`,
  `∂h_m/∂ω = −m·Im[(η+iξ)^m]`, giving
  `Q = ηA' + Σ_m [η h_m B_m' + m ξ k_m B_m/r + Σ_t h_m B_m] + (Σ_t−Σ_s)A`,
  and `φ_exact = A(r)` for every `0 < m < n_φ`.

---

## L49 — #344 step 7 (promoting the reflective-box `ker A` characterization): the diagnostic's own "blindness" gate was a THEOREM about its SVD, and three of the ten mutations I designed were provable non-catchers

Dispatch 2026-08-15, branch `refactor/track-b-remainder`. Task: promote
`derivations/diagnostics/diag_344_reflective_box_loss_nullspace.py` (13 test
functions, 24 cases, `[M]` **166.58 s**, reproduced) into `tests/`, where
`pyproject.toml`'s `testpaths = ["tests"]` means it currently runs never.
Shipped: `tests/sn/_singular_loss_box.py` (shared, NON-collected builders),
`tests/sn/operators/test_loss_nullspace_reflective_box.py` (12 rows) and
`tests/sn/solve/test_boundary_gs_is_a_coherent_splitting.py` (13 rows).
`[M]` **25 passed in 64.69 s**, pyright 0, 13-arm battery.

### a. ⛔⛔ A "functional X is blind to `ker A`" gate built on a DENSE-SVD null basis is a TAUTOLOGY

The diagnostic's flagship blindness gate took `Vn` from `svd(A)`, formed
`ψ = ψ_exact + v` with `v = Vn·c`, and asserted `‖Aψ − q‖ ≈ 0` and
`‖R_g(Aψ − q)‖ ≈ 0`. Both are `A·(a null vector of A) = 0` — true of the
FACTORISATION, measurable without a solver, unfalsifiable by any production
change to the stopping rule. Only two legs had teeth (that the analytic uniform
state IS the discrete solution, and that a non-null control moves both
functionals), and the docstring credited the tautological ones.

⭐ **The re-pose that makes it a measurement: run the production driver TWICE
from cold starts differing only inside `ker A`.** `v ∈ ker A = ker(M−N)` is a
FIXED direction of the iteration operator `G = M⁻¹N` (`Gv = v`), so SI preserves
it exactly and lands on `ψ* + v`. `[M]` jacobi, `(2,3)` LS4 absorber: iteration
count **344 both**, final residual `9.028098e-14` vs `9.022488e-14` (`6.2e-04`
apart), the production balance projection `2.795085` **bit-identically** on each,
bulk `8.4e-16` — and the traces **11.26 %** apart, the difference equal to the
injected `v` to `2.3e-14`. Control: the same experiment on a kernel-free box
gives `2.6e-15`. Now a stopping rule that became kernel-sensitive reds.

### b. ⛔ A face-SUMMED mirror-odd functional is inert when the TRANSVERSE cell count is EVEN — a SECOND parity rule, independent of the source-excitation one

The campaign already carries "an even first axis + a symmetric source leaves the
kernel unexcited". That is about the SOLVE. This one is about the FUNCTIONAL and
bites even on a pure kernel mode: the face sum runs the transverse cells against
the checkerboard `(−1)^Σi`, which cancels for an even count. `[M]` max face
tangential current on a unit-norm `ker A` mode: `(2,2)` **2.5650e-15** (inert),
`(2,3)` `1.9134e-01`, `(3,2)` `3.2691e-01`, `(3,3)` `1.1985e-01`, `(3,4)`
`2.4252e-01`. So `(2,2)` — the obvious "smallest box" — silently kills the
activation leg of any mirror-odd witness. Two independent parity rules on one
campaign; name which one a fixture is chosen for.

### c. ⛔⛔ A mutation on ONE copy of a Pattern-2 twin predicate is INERT — the survivor guards the gate

`SNMesh.reflective_axis_pairs` and `loss_kernel_gauge._reflective_axes` are a
line-for-line twin: same `by_axis` loop, same `len(faces) == 2 and all(faces)`,
one returning the count and the other the axes (the count IS
`len(_reflective_axes(mesh))`). `[M]` widening only the mesh property → **0 of
25 red**; widening BOTH → **exactly 1 red**, the `xmin reflective / xmax vacuum`
row, in either module. Read a null verdict as "my mutation was insufficient"
before "the gate is blind", and grep for a second implementation of the
predicate you broke.

### d. ⛔⛔ Compute the gate's stabiliser EMPIRICALLY when the claim is a CHARACTER identity — five plausible metric errors are designed-green

The canonicality leg (`ψ_exact ⊥_G ker A`, the theorem that makes minimum-norm
CANONICAL rather than conventional) survives every metric that is constant
across an orbit's cells, because the vanishing quantity is a character sum and
the weight only has to be shared by the ordinates being summed. `[M]` `frac`:
shipped `1.5514e-15`; `×2` everywhere `1.5137e-15`; `×(1+½ sign Ω·n)`
`1.5135e-15`; `×(1+½ sign of the PARTNER face's)` `1.7500e-15`; one whole face
`×3` `1.8743e-15`; random per-DOF **`1.9377e-02`**. I wrote two metric mutations
believing each would red it; both were annihilated. The honest scope is *a gate
on the kernel's PARITY, not on the metric's values* — say so in the docstring
rather than claiming a sensitivity the row does not have.

### e. ⭐ A positive control can be pathologically SLOW when the mutation is on the wrong side of a stability threshold

`_DD_W = 0.45` (diamond weight below ½) makes the face transmission
`−(1−w)/w = −1.22`, i.e. AMPLIFYING, so the SI grinds: killed at **25 min**,
18 of 25 rows reached. `_DD_W = 0.55` gives `−0.818` — genuinely DAMPED, the
regime the control is meant to model — and the whole arm runs in **46.21 s**,
reddening **13 of 25**. One sign of algebra separates a usable control from an
unusable one; do it before launching.

### f. Battery table (25 rows, `python -O`, `-p no:randomly`)

| arm | mutation | red |
|---|---|---|
| baseline | — | **0** (`25 passed in 64.69s`) |
| A1 | `reflective_axis_pairs` widened (one copy) | 0 — INERT, see (c) |
| A1b | BOTH copies widened | 1 — the mixed-BC row |
| A2 | `_TANGENTIAL_MU = 0.5` | 4 |
| A3 | metric × `(1+½ sign Ω·n)` | 0 — non-catcher, see (d) |
| A3b | metric × partner face's sign | 0 — non-catcher, see (d) |
| A4 | `net_current` reads the wrong face row | 1 — the kernel-mode current row |
| A5 | mesh ignores `scheme=` | 2 — both LD rows |
| B1 | G-S arm drops `B_upper` | 12 (2 intended + 10 collateral) |
| B2 | ERR-056 first-group reflect | 11 — incl. all 4 coherence rows |
| B3 | split direction flipped | 11 |
| B4 | `_balance_projection := identity` | 1 — the cold-start certificate row |
| **PC** | `_DD_W = 0.55` (damped closure) | **13** |

⭐ The result worth keeping: **B2 reds all four coherence rows and NEITHER
`M − N == A` row.** That is the module's whole thesis measured — ERR-056 is an
INCOHERENT schedule, not a non-splitting — and no single gate could have shown
it.

### g. Two documentation defects found in passing

* `tests/sn/operators/test_loss_kernel_gauge.py`'s module-docstring SVD table
  records `mine`/`law` = `224 / 464 / 242` for `product(4,4)`, `product(8,8)`
  and `lebedev(11)`. `[M]` the true values are **0 / 16 / 18** — the columns
  recorded `T + R` where they claim `R`. The table contradicts its own prose
  four lines below (*"`product(4,4)` … is pure T (224 tangential, R = 0)"*) and
  the theory page, which is correct. **3 of 3 T-bearing rows.**
* `dd-null-counting-law` ends in `+ #{tangential trace DOFs}` and
  `predicted_kernel_dimension` counts `R` only — every `_SINGULAR` fixture in
  the gauge suite is `level_symmetric`, where `T = 0`, so the equation's `T`
  term had **no** executed gate. The promoted `T + R` row is it.

---

## L55 — The static call graph is blind to ORPHEUS's operator algebra; a graph-derived "test count" measures the RESOLVER, not the suite

**Context.** 2026-08-15, `main` @ `a1c90aac`, nexus 0.16.1, graph
`docs/_build/html/_nexus/graph.db` (24530 nodes / 217667 edges). Dispatched to
design the forward half of a graph-grounded test workflow for issue **#358**
("a red should INVALIDATE its dependent cone"). Deliverable memo:
`scratchpad/nexus_demand_test_architect.md` (this session's scratchpad; NOT a
tracked artefact).

### a. ⛔⛔ THE HEADLINE — 94.8 % of the tests that execute an operator are invisible to `callers`

`[M]` For `py:class:orpheus.numerics.operator.OperatorSum`, restricted to the
`tests/numerics` slice that I traced under cProfile:

```
STATIC selection      : 13
RUNTIME-observed      : 229
observed but NOT selected (UNSAFE MISS): 217      # 94.8 %
selected but not observed (over-select):   1
```

Conservative direct-edge-only recount (rules out a transitive-closure artefact):
`runtime DIRECT test→OperatorSum callers = 29`, `static DIRECT = 14`, **17
direct misses**.

**The mechanism, measured exactly:**

```
OperatorSum members observed FIRING     : {__init__ 16, apply 17, inverse 11,
   is_invertible 12, assemble 8, b 7, domain 7, apply_transpose 6, codomain 4}
OperatorSum members with STATIC in-edges: {'OperatorSum': 39}
```

The **only** `OperatorSum` node carrying any static incoming call edge is the
CLASS (39 constructor calls). `.apply`, `.apply_transpose`, `.inverse`,
`.assemble`, `.domain`, `.codomain`, `.is_invertible` all have **ZERO** — while
all nine members fire at runtime.

⟹ **The static graph sees the operator algebra only at CONSTRUCTION, never at
USE.** `A_loss = L + C - S` resolves; `A_loss.apply(ψ)`, `A_loss.H`,
`(L+C).solve(q)` do not. Whole-graph confirmation: `[M]` **25889 of 121788
`calls` edges (21.3 %) point into `unresolved`**, 5389 unresolved nodes, and the
top named unresolved targets are `op.apply` (265), `A.apply` (114), `L.apply`
(100), `case.build_mesh` (118), `quad.weights.sum` (101). Same family:
`[M]` `SNSolver.solve_fixed_source` has **in-calls = 0** while the free function
`solve_sn_fixed_source` has **233**; `SNMesh.__init__` has **0** (constructor
calls attach to the class node).

⭐ **The perverse part, and the reason this is a standing lesson rather than a
tool bug:** this is nexus issue #16's annotation-mediated dispatch gap, and its
severity here is a direct consequence of Cardinal Rule 2. `coding-elegance`
Pattern 1 says *spell the domain's operations as dunders on domain types* — and
every such call is exactly what the resolver cannot follow. **The better
ORPHEUS's architecture gets, the blinder the static call graph becomes.**

### b. ⛔ The number I nearly shipped as a coverage finding

`[M]` `production fn/method (excl. derivations) = 2050`; **`production nodes
with ZERO reaching test : 1646 (80.3 %)`**; median tests-per-node = 0. The
resulting "what to test next" queue is topped by:

```
   fanout= 37 tests= 0  method:orpheus.sn.solver.SNSolver.solve_fixed_source
   fanout= 32 tests= 0  method:orpheus.numerics.measure.DiscreteMeasure.quotient
   fanout= 30 tests= 0  method:orpheus.sn.solver.SNSolver._solve_source_iteration
   fanout= 24 tests= 0  method:orpheus.sn.mesh.augmented_mesh.SNMesh.__init__
```

Every one of those is heavily tested. **The 80.3 % is a measurement of the
resolver, not of the suite.** The tell that saved it was implausibility — I know
`solve_fixed_source` is exercised — not any property of the instrument. The
runtime overlay repairs it: `[M]` `DiscreteMeasure.quotient` goes **0 → 9**
test-callers, `SNMesh.__init__` **0 → 1** (8 any-callers), and distinct orpheus
nodes reached by the slice's tests goes **163 → 404** (313 recovered, 2.48×).

⟹ **RULE: never quote a graph-derived per-node test count as coverage in this
codebase without the runtime union.** A `tests=0` on a method is the null
hypothesis "the resolver could not see it", not "nobody tests it".

### c. `tests` edges are test→EQUATION. There is no test→code edge type.

`[M]` `graph-query "* -tests-> function"` → `[]`. Denominator from the DB:
**all 2748 `tests` edges are `{function,method} → equation`** (method 1430,
function 1318); sources are `tests.*` 2745 / `derivations.diagnostics` 2 /
`orpheus.*` 1. It is the `verifies()` V&V-matrix edge and must never be read as
a test→SUT relation. The equation-join alternative (`test -tests-> eq
<-implements- code`) is silent on 75 % of production: `[M]` only **771 of 3043**
orpheus fn/method nodes carry any `implements` edge.

### d. ⭐ The enabling fact — graph test nodes ≡ pytest collected defs, EXACTLY

`[M]` Full suite, `pytest tests -q --collect-only` (**7.0 s**, 9861 items) joined
to the graph on `(path relative to root, dotted name tail after the module
part)`:

```
collected items 9861 · distinct (file,def) keys 5273 · parametrization fan-out 1.87x
graph is_test fn/method nodes 5273 · distinct keys 5273
COLLECTED-but-NOT-IN-GRAPH : 0 keys / 0 items
IN-GRAPH-but-NOT-COLLECTED : 0 keys
JOIN RATE by ITEM = 9861/9861 = 100.0%
```

So `py:method:tests.a.b.C.d` → `tests/a/b.py::C::d` is mechanical and lossless,
and a graph-derived selection is directly runnable: `[M]` the 25-row
`OperatorSum` selection ran **25 passed in 0.46 s** with every ID resolving.
⛔ But granularity is the `def`: **1.87× fan-out**, so a red *param row* cannot
be named from the graph.

### e. ⛔ The L54 lesson fired on ME — `node_attrs.value` is JSON-encoded

My first join reported **`JOIN RATE = 0.0%`, 1113 keys missing, 0 phantom** —
a clean, confident, completely wrong "nexus cannot do this". The graph was fine;
`node_attrs` stores values **JSON-encoded**, so `file_path` arrives as
`"/Users/…"` *with the quote characters* and `os.path.relpath` silently produced
garbage. I caught it only because I *knew* a node existed in that directory.
The corrected script raises `AssertionError` when the graph side comes back
empty; the first one did not. ⟹ **a two-sided join needs a denominator
assertion on BOTH sides** — I had asserted the collected count and not the graph
count, and the unasserted side is the one that failed.

### f. `retest` truncates at a hard-coded depth 3, and divides by an inflated denominator

`[M]` `query.py:2330-2368` calls `self.impact(..., max_depth=3)` with the 3
literal. Sweeping the depth on `orpheus.numerics.convergence.warn_if_unconverged`:

```
  max-depth 3 : pytest_selectable= 333      <-- retest stops here
  max-depth 4 : pytest_selectable= 351      <-- fixed point (stable to depth 8)
```

**18 tests (5.1 %) reach the symbol and are reported `safe_to_skip`.** Symbol-
dependent: `solve_sn` reaches its fixed point at depth 2, so the obvious
spot-check shows nothing. Separately `[M]` `is_test` means *"lives under
`tests/`"* — 7298 nodes = fn 2604 + method 2669 + **data 933 + class 877 +
attribute 215**; only 5273 are collectable. `retest`'s `total_tests` and
`safe_to_skip` are inflated by **2025 (27.7 %)** non-runnable nodes.

### g. ⛔ The naive cone is 31–36 % of the suite, and it is not filterable

`[M]` Two seeds, two-stage cone (red test → downstream `calls` → orpheus nodes →
upstream `calls` → tests), denominator 7298:

```
test_keigenvalue_matches_solve_sn_2g_slab : 2338 (32.0 %) | external BLOCKED 2294 (31.4 %)
test_keff_curvilinear.test_homogeneous_exact: 2589 (35.5 %) | external BLOCKED 2589 (35.5 %)
```

The external-blocking control moves it **0.6 pp / 0.0 pp** — the bloat is
genuine shared ORPHEUS infrastructure (`SNMesh`, `Quadrature`, `FunctionSpace`),
not numpy noise. **Structural reason: `calls` gives "A and B both touch P",
which is SYMMETRIC, hence not a partial order. A cone partition needs an
ANTISYMMETRIC "B rests on A".** No edge filter converts one into the other — the
information (which node a test is *about*) is simply absent. Feasibility of
supplying it: `[M]` the median test has exactly **one** direct production callee
(histogram `0:3625, 1:2088, 2:803, 3:371, …`), and the production `calls` DAG
has depth 0–14 with **2030 of 3367 nodes at depth 0**, so a subject-depth
ordering exists but is coarse (60 % ties).

### h. Runtime overlay — cost, and the `--source-prefix` trap

`[M]` `tests/numerics`: baseline **331.21 s** (2344 passed) → cProfile
**529.97 s** = **1.60×**; `.prof` 2.49 MB; `runtime-ingest` **2.3 s**; sidecar
1.47 MB. `runtime-edges`: `fired` 1643 edges (tests→orpheus 575),
`dynamic_only` **3671** (tests→orpheus **1924**), `dead` 99.
`--substantive-only` drops `dynamic_only` 3671 → 2190.

⛔ **`--source-prefix` takes ONE prefix (`runtime.py:263`,
`filename.startswith(source_prefix)`) and the obvious choice is the wrong one**,
because `.venv/` lives *inside* the repo:

| prefix | nodes | unresolved | join | has tests→orpheus edges? |
|---|---|---|---|---|
| repo root | 2417 | 8591 | **21.9 %** | yes, but keeps every `.venv` frame |
| `<repo>/orpheus` | 1109 | 541 | **67.2 %** | **no** (1883 orpheus-internal) |
| `<repo>/tests` | 1308 | 169 | **88.6 %** | **no** (773 tests-internal) |

No setting yields both trees while excluding `.venv`.

### i. ⭐ The claim-kind taxonomy I proposed — THEOREM / REFERENCE / RECORD

#358 asked whether `vv` L51's CONSTRAINT/RANKER/DIAGNOSTIC fits *tests*. It does
not: **every collected pytest test is a CONSTRAINT by construction** (pass/fail
is the only outcome), so that partition has one non-empty cell. L51 classifies
*instruments by role*; the question here is *assertions by evidence provenance*.
The axis that partitions is **where the expected value comes from**:

* **THEOREM** — entailed by a law holding for every admissible input (identity,
  adjointness, involution, conservation, `M − N ≡ A`). Red ⟹ the object violates
  its own definition; every REFERENCE and RECORD claim on that subject is void.
* **REFERENCE** — a structurally-independent external route (`vv`'s three
  pillars). Red ⟹ implementation disagrees with the mathematics *here*.
* **RECORD** — whatever the code produced on a chosen day (frozen `.npy`, hash,
  pinned literal). Red ⟹ *something changed*; **zero** information about which
  side is right (`numerical-bug-signatures` Signature 10).

`[M]` It must be DECLARED, not derived: `assert_allclose` appears in **218
files** and is used identically for closed-form, MMS and frozen baselines;
RECORD's syntactic tell (`np.load`, 14 files) under-counts against 40 `.npy`
artefacts and 55 "snapshot" files. Declaring a node ATTRIBUTE does not violate
#358's ban on hand-maintained EDGES — a missing attribute is loud and
enumerable, a missing edge is silent.
⭐ The audit it unlocks, and the one task #51 had to run by hand twice:
**every RECORD subject must also carry a THEOREM or REFERENCE test.** Honest
limit, which must ship inside the audit's own output: it finds subjects with
*no* independent pin, never a *blind* one — that is mutation's job (#358
anti-rec 5).

### j. The marker axes are already in the graph — and are a 28.9 % sample

`[M]` `node_attrs` keys: `decorators` 3981, `vv_level` 1530, `verifies` 896,
`catches` 239, `slow` 143. So a `claim` marker costs one `elif` in
`ast_analyzer.py:194-262`. ⛔ **But `vv_level` resolves on only 1524 of 5273
selectable tests (L1 798 / L0 677 / L2 49; 3749 `<none>`)** because
`[M]` **254 test files carry a module-level `pytestmark`** the AST cannot
attribute to a `def` — the graph sees the `regression` marker on **1** test
against 14 source sites — and because `tests/conftest.py` resolves the level by
a **five-rule precedence** whose class-name / func-name / case-inherited rules
are computed at collection time and are not in the AST at all.

### k. Side finding, unrelated to #358 but large

`[M]` `runtime-hotspots --by ncalls` on the 331 s `tests/numerics` slice:
`RigidMotion.determinant` **55 753 738 calls**, `RigidMotion.__post_init__`
**27 720 955**, `__matmul__` **27 305 170**, `close_group` **27 255 285**. The
frozen dataclass's construction invariant re-runs 27.7 M times inside
`close_group`. Performance, not correctness; not investigated. Route as
`module:geometry`, `type:improvement`.

---

## L58 — CS3 (the cone overturn, PRE-carve): "compute it from flat norms" names TWO conventions 0.47 % apart, and a ratio-valued pin cannot see a uniform scale

Dispatched 2026-08-19 to plan phase CS3 of the space-and-kernel campaign (retire the
affine/torsor flux algebra; flux lives in the positive cone K ⊂ V) on branch
`refactor/cone-field-algebra`, HEAD `000cf144`. Deliverables: a value-neutrality
harness, a ρ-trajectory capture gate implemented and green, a test-migration map, the
new-algebra gates, and the cone-predicate spec. Plan:
`scratch/cs3_verification_plan.md`; gate: `tests/numerics/test_si_diagnostic_trajectory.py`.

### L58a — the brief's relocation phrase was ambiguous, and the two readings are 4 orders apart

The brief's step 1 read *"SI computes the diagnostic trajectory from flat norms"*.
Today ρ is `Displacement.contraction_ratio` = `self.l2 / previous.l2` where
`Field.l2 = space.norm(values)`, evaluated on the **interior leaf** the finder
`_flux_displacement_leaf` extracts from the composite. "Flat norms" therefore names
at least two different computations, and I measured both on the same 12-pass c→1
fixture:

| rival spelling | `[M]` max relative difference from the shipped ρ |
|---|---|
| interior leaf, `np.linalg.norm(values)` instead of `space.norm(values)` | **2.29e-16** (0–1 ULP per step) |
| whole composite, `_l2_norm(displacement)` — the spelling the SI loop ALREADY has in hand — which additionally ravels the boundary trace block | **4.71e-3** |

⭐ The second one is the natural relocation: `_l2_norm` is a module-level helper in
the very file the diagnostics move into, it accepts the composite directly, and
using it deletes the finder. It is also the one that silently moves the number.

⟹ **When a brief says a relocated computation uses "plain / flat / simple"
arithmetic, enumerate the candidate spellings and measure the spread BEFORE writing
the pin.** The spread is not a footnote — it IS the pin's discriminating power, it
sets the tolerance, and it is the control leg the gate needs. Here it gave
`rtol=1e-12`: 4 orders above the ULP floor of the harmless rival, 9 orders below the
error of the dangerous one.

⚠ And the first rival is only harmless *today*: `[M]` the `angular_flux` space has
`inner_product_weights is None` (Euclidean). Recomputing the same trajectory under a
physical `V_cell × w_n` metric moves ρ by **1.12e-3** — so the LATER phase of the same
campaign (CS2, which relocates metrics onto spaces) will legitimately RED this pin.
That is a plan-level question, not a test-level one, and the plan carries it as a
blocking user ruling: is the relocated diagnostic defined on the SPACE norm (CS2 owns
re-deriving the numbers) or the EUCLIDEAN norm (permanent, but no longer a
Hilbert-space quantity)?

### L58b — a RATIO-valued pin is Mode-12 blind to a uniform scale, and only the battery says so

The 5-mutation battery on my own gate:

| # | mutation | `[M]` result |
|---|---|---|
| M1 | ρ from the whole composite's flat norm | 2 failed — the ρ pin AND the ‖Δψ‖ pin |
| M2 | `Field.l2` → `np.linalg.norm(values)` | **5 passed** — the declared ≤1 ULP blindness, measured rather than asserted |
| M3 | one spurious leading ratio (cadence drift) | 1 failed — the length leg |
| M4 | `where_largest` ravels first (index-layout loss) | 1 failed — the map leg only |
| M5 | every norm × (1 + 1e-9) — the positive control | **1 failed — the ‖Δψ‖ leg ONLY** |

⛔ M5 is the finding. A 1e-9 relative perturbation of EVERY norm leaves the whole ρ
trajectory GREEN, because ρ is a ratio and a common factor cancels exactly. So a
plan that pins "the ρ trajectory" and nothing else has a gate that is blind to every
uniform mis-scaling of the norm — including the whole class "the relocation forgot a
weight". The catcher is the SEPARATE magnitude pin (‖Δψ‖ and ‖Δψ‖/(1−ρ)), which is
why the two must never be folded into one row "for tidiness".

⟹ **Whenever a gate's measurand is a RATIO, write down its stabiliser before writing
the row** (`vv` #12's design-time question): ratios annihilate common factors,
normalised shapes annihilate global scaling, spectra annihilate similarity. Then pin
one un-normalised quantity from the same object.

⭐ M2 is the mirror discipline and equally worth running: a mutation you EXPECT to be
green, run to prove the declared blindness is real. Writing "≤1 ULP, so the pin
cannot see it" without running it is an unmeasured claim wearing a measurement's
clothes.

### L58c — before minting a bit-identity instrument, grep for a WARNING you can escalate

`tests/sn/solve/test_affine_carve_bit_identity.py` (the #208 carve's own gate, re-posed
at #333 from `sha256` onto stored `.npy` + a `DriftWarning` tripwire) is a
`SAFETY × conv_tol = 1e-11` value wall by default — and `-W
error::tests.sn.regression._regression_assert.DriftWarning` turns it into a **1-ULP
bit-identity wall on three drivers** (2-D windowed SI, 2-D Krylov, 1-D slab SI).
`[M]` 3 passed in **1.60 s**; positive control (a plugin advancing the FIRST element
of every loaded baseline by one ULP, `raise`ing at `sessionfinish` if it perturbed
nothing) → **3 failed**, and unescalated → 3 passed with 7 warnings reporting
`drifted 1 ULP / 1.90e-16 rel`.

The same escalation on the broad `test_dd_regression.py` suite: `[M]` **11 of 13
cases are bit-exact at HEAD**; the 2 that are not are exactly
`cyl_1g_homogeneous_folded_{2x4,4x8}_dd_n20` (`scalar_flux` 40 679 ULP / 5.32e-12 and
549 721 ULP / 7.19e-11). Characterising them by NAME rather than counting them makes
every post-carve red attributable with zero triage.

⟹ **A project that ships a "drift is audible" warning class has already built the
bit-identity gate; the escalation flag is the instrument.** Check for one before
proposing a new snapshot — and verify BOTH that the `-W` string parses (`vv` Mode-8
EIGHTH class) and that it bites, with a control.

### L58d — a "the fiber discipline survives" claim is simulable pre-carve in four lines

The campaign plan asserted that retiring the flux mixin drops `+`/`−` through to the
base `Field` dunders and thence to the mesh-binding guard, so cross-problem fluxes
still refuse. That is checkable WITHOUT the carve: call the base dunder directly.

`[M]` `Field.__add__(a, b)` on two cross-mesh same-shape fluxes REFUSES —
`ValueError: <Leaf> arithmetic across distinct SNMesh instances is forbidden — the
field is mesh-bound` — on `AngularFlux`, `ScalarFlux`, `AngularBoundaryFlux` and
`HarmonicMomentFlux`; same-mesh returns the leaf type. ⚠ And the discriminator the
gate must assert: `a.space == b.space` is **True** (`FunctionSpace.__eq__` is
`(name, shape)`) while `a.space is b.space` is False — so the SPACE gate does not
catch it and only the mesh arm does. A gate that omits that in-test precondition
silently degrades the day space equality changes.

⟹ **A "the fall-through lands on X" claim is a runnable experiment: invoke the base
implementation the carve will expose, on the inputs the guard must still refuse.**
Cheaper than the carve, and it converts the plan's owed negative control from a
promise into a specified test with its message fragment already measured.

### L58e — two overlapping grep predicates in a brief are not two work items

The brief listed "the 7 files pinning the flux+flux TypeError" and "the ~16 affine
raise-sites" as separate migration buckets. Measured with the predicates written out:
`[M]` 16 raise-sites in **10** files; my flux+flux predicate matches **8** files
(the memo's narrower one, 7); the **union is 12**, of which 1
(`test_ordinate_scan.py`, whose "affine" is the DD scan recurrence) is a false
positive ⟹ **11 real files**. Four are raise-site-only and two are flux+flux-only, so
a plan built on either list alone misses 2–4 files.

⟹ **Compute the union and print the two set differences.** And triage a concept grep
by MEANING before calling any hit a site — `affine` in this tree names at least three
unrelated things (the flux torsor, the DD recurrence's affine-in-(b, ψ₀) structure,
and the affine BOUNDARY law `affine-bc-form`, which alone has ~18 `:eq:` citers and
must not be touched).

### L58f — the brief's named witness for a new predicate was at the wrong TIER, and a better one was two probes away

The brief said the cone predicate would be "exercised by DD's existing negative-flux
witness (`TestPositivityFailure`)". `[M]` that test lives at
`tests/sn/sweep/core/test_diamond.py:793-855` and asserts on
`strat.update(...).outgoing_spatial_flux` — a bare ndarray from a single CELL VISIT.
There is no field, so an element predicate on a `Field` cannot be exercised by it
without wrapping the number back up.

Three probe runs found a strictly better witness through the PUBLIC entry: `[M]`
`solve_sn_fixed_source` on a 4-cell / 40 cm / Σ_t = 10 slab (Δx·Σ_t = 100, S2, unit
source in the first cell only) CONVERGES to `min ψ = −6.399383e-01` with 2 of 8
entries negative and `min φ = −8.438399e-01`; the benign sibling at `nx = 2`, same
materials, gives `min ψ = +2.181405e-01`. One parameter apart, same entry, both
converged — the positive and negative legs of `vv` #11 with nothing else varying.

⟹ **A witness cited for a NEW predicate must be checked at the TIER the predicate
takes.** A cell-level demonstration and a field-level predicate are different
objects, and the gap is invisible in a plan because both are "the negative-flux
witness".

### L58g — a relocation that leaves the OLD methods alive turns three committed tests into pins of dead code

The brief's step 1 relocates the three diagnostics onto the iteration layer while
"the Displacement TYPE still exists after this step". If the METHODS survive too,
the tree carries two implementations and the foundation battery's three diagnostic
rows assert the copy production no longer calls — a transient Pattern-2 twin whose
tests point at the dead half.

`[M]` the fix costs nothing: outside its defining module, `true_error_estimate` and
`where_largest` have **0 production call sites** (grep returns prose only) and
`contraction_ratio` has exactly **one** (`iteration.py:792`). So step 1 deletes the
three methods in the same commit that lands their replacements.

⚠ Second §6b-class finding in the same step: the retiring finder
`_flux_displacement_leaf` does TWO jobs — WALK the composite (`.interior`, else
`systems[0]`) and DECIDE it carries diagnostics (`hasattr(..., "contraction_ratio")`).
The relocation destroys the second, and the natural repair (delete the finder)
destroys the first — which is exactly the failure a committed gate already exists
for (`test_psi_half_coupling.py::test_g_d1_7_…`, written because the coupled walk
once went silent).

⟹ **When a step "moves" a capability, ask what ELSE the moved code was doing.** A
helper named for one job routinely carries a second, and the second is the one with
a scar-tissue test.

---

## L59 — CS1 (the Energy axis / axis-composed spaces, PRE-carve): the design's own `.H` control was a PROVABLE non-catcher, and the fixture library made one production arm witness-less

**Campaign.** Campaign 1 CS1 — "the homogeneous solver poses its problem on a real
Energy space". Dispatch: §T of `.claude/plans/cs1_energy_space_design.md`.
PRE-implementation; deliverable `scratch/cs1_verification_plan.md` (1255 lines, 40
gates A1–A12 / B1–B11 / C1–C2 / D1–D12 / E1–E3, a 23-mutation battery with 3 positive
controls and a MUST-STAY-GREEN column). Tree grounded at HEAD `4e11731b`; the design
record's own §P grounding pass had run at `273d431a` and was itself thorough — every
finding below is something a careful grounding pass still missed, which is the point.

### L59a ⭐⭐ — a rank-1 axis makes EVERY metric SCALAR, so no production space on that carrier can EVER load `.H`. This is the SPACE-side dual of `vv` #12's commutator clause, and it kills the obvious control.

`vv` #12 already says a metric-adjoint gate is blind iff `[G, Aᵀ] = 0`, and gives the
OPERATOR-side instances (a diagonal `C` commutes with every diagonal `G`; a
measure-preserving permutation `B` likewise). The dual nobody had written down: the
**SPACE's own shape** can force `G` into the centre. The homogeneous `bulk_space` is
`Energy ⊗ point` with `spatial_shape == (1,)`, so its spatial weight is a
**one-element array = a scalar**, and the Energy weight is counting **by the
counting-measure theorem**. ⟹ every metric that space can carry commutes with
everything.

`[M]` 2026-08-20, `loss = C − (IsoS + IsoN2N)` on `get_mixture("A","2g")`,
`x = [[1.],[2.]]`, `loss.H.apply(x)`:

| space | `.H.apply(x)` | bit-identical to the `None` path |
|---|---|---|
| `None` (today's production) | `[-0.08, 0.2]` | baseline |
| `weights = None` | `[-0.08, 0.2]` | **True** |
| `weights = [1., 1.]` (the quotient point) | `[-0.08, 0.2]` | **True** |
| `weights = [2., 2.]` (a one-cell mesh, `V = 2`) | `[-0.08, 0.2]` | **True** ⛔ |
| `weights = [2., 5.]` (per-GROUP) | `[-0.38, 0.2]` | False ⭐ |

Row 4 is the finding. The design record proposed distinguishing "the quotient point,
weight 1.0" from "a genuine one-cell mesh, `V ≠ 1`" and treated that difference as the
thing that makes the metric non-trivial. It is **bit-identical** — a provable
non-catcher, not a weak one, and no tolerance / refinement / regime change can expose
it. The control had to become a deliberately **non-physical** toy carrying a
per-GROUP weighted axis — a metric the counting theorem FORBIDS on a real
`EnergyAxis` — and its docstring has to say so, or a later reader "fixes" the toy into
uselessness.

⟹ **The rule: before designing a vv#19 loaded/blind PAIR, MEASURE the candidate
control on the pre-carve tree.** It costs one probe (this one was ~20 lines and 2 s)
and it is the only thing that can tell "the control is loaded" from "the control is in
the stabiliser". Both readings are small residuals; only the measurement separates
them. ⚠ And the second half of the same probe: only component **0** moves; component 1
reads `0.2` under both metrics — a control asserting a single component, or the wrong
one, is blind even when the space IS loaded. Assert the whole vector.

⟹ **The corollary that generalises past `.H`: if a distinction is invisible to every
VALUE functional a carrier admits, then IDENTITY is the only instrument.** Here the
quotient-vs-one-cell distinction survives only as space equality (the derived NAME must
encode the weights), so the gate is `space_a != space_b`, and the mutation that proves
it (`bulk_space` weights → `np.ones(...)`) must be verified to leave the byte gate,
`D4a`, and the floor **GREEN** — otherwise the closeout credits a value gate with
measure coverage it cannot have.

### L59b ⭐ — a fixture library that is uniform in the discriminating field makes a production ARM witness-less, and `vv` #17's granularity trap then fires at the FIXTURE tier

CS1's `bulk_space` energy arm branches: all materials carry an energy grid ⟹
`EnergyAxis.from_grid`; else ⟹ `EnergyAxis.synthetic(ng)`.

`[M]` 2026-08-20: `get_mixture(region, ng_key).eg is **None**` for **all 12** shipped
`(region, ng)` pairs (`{A,B,C,D} × {1g,2g,4g}`), and `get("homo_2eg_n2n")`'s mixture
too. ⟹ **every** `bulk_space` mint reachable from the fixture library takes the
synthetic arm; the `from_grid` arm has **no witness**, so a whole-arm mutation reddens
only the synthetic side and the battery reports "the energy arm is gated".

This is `vv` #17's multi-arm-guard trap, but the arm is unreachable because of the
**FIXTURES**, not because of the code — so the usual remedy ("mutate each arm
separately") produces a mutation with an empty red set and no obvious diagnosis.

⭐ The witness existed one line away, already in the tree, and only a grep for the
FIELD (not for the arm) found it — `tests/homogeneous/test_homogeneous.py:415-417`
builds the only `eg`-bearing homogeneous mixture in the repository via
`dataclasses.replace(base, eg=np.array([1.0e7, 1.0e3, 1.0e-3]))`.

⟹ **The rule: when a production branch discriminates on a FIELD, census that field
across the fixture corpus before writing the gate** — one comprehension, and it turns
"parametrize over both arms" from an assumption into a measurement. The census is also
what tells you whether the witness must be manufactured or merely located.

### L59c ⛔ — "prove it does not allocate" cannot be gated by asking a densifier to `MemoryError`: it gets OOM-KILLED, which fails the RUN, not the TEST

CS1's `of_axes` must keep the factor measures per-axis (never materialise the outer
product). The obvious memory-shaped gate is "build a space so large the dense form
cannot fit, and assert construction succeeds".

`[M]` `np.multiply.outer(np.ones((4096,4096)), np.ones(4096))` (≈550 GB) — **exit
code 137**, SIGKILL from the OOM killer. It never raised. A pytest run gated that way
dies without a verdict, nondeterministically by machine.

⟹ **The rule: size a no-allocation gate for SEPARATION, not for failure.** `[M]` the
safe form is `(2000,) ⊗ (2000,)`: dense `32 000 000` bytes vs per-axis `32 000` bytes
— a **1000×** separation, built in **4 ms**, asserted on *reachable* `ndarray.nbytes`
walked off the object. Pair it with an EXACT structural leg (the dense slot is `None`;
no reachable array has `size == prod(shape)`) and a BEHAVIOURAL leg (the metric still
applies correctly at that shape) — because "never densify" implemented by *dropping
the metric* passes the first two. Rejected `tracemalloc` as the instrument: the NumPy
allocator domain makes the reading version-fragile, where `nbytes` is exact and free.

### L59d — a TYPE-ANNOTATION widen has NO runtime witness; its gate is pyright, and a pytest row for it is a Signature-8 tautology

CS1 step 3a widens `FissionOperator`'s space field `FullFieldSpace | None →
FunctionSpace | None`. The natural gate — "hand it a plain `FunctionSpace`, assert it
constructs" — is **green before and after**: Python does not enforce annotations, so
the row asserts a capability the tree already had. It reads like coverage of the widen
and is `vv` Mode 8's signature-tautological class wearing a type hint.

⟹ **The rule: a widen/narrow of a static type is gated by the type checker, never by
a runtime row.** In this tree that is `tests/test_pyright_ratchet.py`; record the
ratchet delta in the commit body. Say so explicitly in the plan, or the step ships with
a green row and an uncovered change. Same family, measured in the same pass:
`homogeneous/solver.py` keeping a leftover `basis_shape=(ng, 1)` after the derivation
lands is **value-identical** (`_resolve_basis_shape` lets explicit win, and the explicit
value equals the derived one) ⟹ the done-when item "both production spellings gone" has
**no runtime witness either** and is a `grep` obligation on the commit, not a gate.
State which done-when items are grep obligations; an unstated one reads as covered.

### L59e — a `repr`-derived identity is NOT injective: ndarray `repr` TRUNCATES, and no small-toy gate can see it

CS1's `of_axes` derives the space NAME from the axes' structural content, and — because
space identity is `(name, shape)` until a later phase — a name COLLISION between
different axis tuples makes two different spaces compare EQUAL and compose silently.
The cheap implementation is a name built from `repr(axes)`.

⛔ `repr` of an ndarray elides with `...` above the print threshold, so two distinct
LONG weight vectors render identically. The collision is invisible to every gate built
on small toys — which is every injectivity gate anyone writes.

⟹ **The rule: an identity derived from a rendering is only as injective as the
renderer.** Derive from `.tobytes()` through a digest, and state the float caveats
where they exist (`-0.0` vs `+0.0` have different bytes and compare equal; `nan` bytes
can be equal while the values compare unequal). And when an injectivity gate is
written, include at least one pair whose SHAPES are identical — otherwise `shape`
carries the discrimination and the NAME is never tested. CS1's two such pairs:
`synthetic(2)` vs `from_grid(<2-group edges>)`, and a spatial point with weight `1.0`
vs weight `2.0`.

### L59f — the CS1 grounding-pass residue: what a thorough §P pass still missed

The design record's §P ran 10 grounding items and closed all 10. Four call-site facts
still slipped, and all four share a shape — **the pass checked the SITES it enumerated
and not the CONSEQUENCES at those sites**:

1. `tests/numerics/test_matrix_inverse_operator.py:206/:265` were classified "stays on
   the legal None path". `[M]` both call
   `MultiplicationOperator.from_mesh(…, mat_xs.mesh)` with the **degenerate carrier**,
   so they pick up the new default. They stay green (explicit `basis_shape` wins) but
   are no longer None-path witnesses — a later citation of them as such would be false.
2. `tests/homogeneous/test_homogeneous.py:284-286/:359-365` were classified "mirror
   tests — migrate to domain-derivation". `[M]` **half of each cannot**: both build `F`
   BARE, and no-default-derivation is RULED, so `F.domain is None` and `as_matrix()`
   raises. ⭐ And `:359-365` is `@verifies("resolvent-object-gate")`, written as the
   line-for-line mirror of the production `K = M⁻¹ @ F` — once production threads
   `space=V`, a mirror that keeps the bare `F` **pins a retired idiom**, which is the
   identical warrant on which the campaign deletes two strict-xfail rows.
   ⚠ Note the product guard SKIPS a `None` codomain, so `M⁻¹(bound) @ F(None)`
   constructs happily and `K.domain` is silently `None` — the breakage surfaces one
   call later, at `as_matrix`.
3. `_R1_XFAIL` decorates **two** tests, not one; the constant survives the deletion
   carrying a reason string the carve makes present-tense FALSE.
4. Two production refusal messages (`"this multiplier is space-anonymous"`) become
   present-tense false — the PREDICATE stays right, the stated REASON dies.

⟹ **The rule for a PRE-carve dispatch that follows a grounding pass: re-derive the
CONSEQUENCE at every enumerated site, do not re-verify the enumeration.** The
enumeration is what a grounding pass is good at and what it will have done well; the
per-site consequence is what it skips, because that requires simulating the carve. The
cheapest form: for each site, write the ONE sentence "after the carve this site
{keeps / gains / loses} X" — the sentences that will not write are the findings.

---

## L60 — CS1.5 (the Medium un-weld, PRE-carve): a `getattr` default swallows a typed refusal, and "exact by construction" was 1 ULP off in exactly one sub-family

**Dispatch** 2026-08-20, branch `feature/cs1-energy-space` @ `a5d59425`,
PRE-implementation. Deliverable `scratch/cs15_verification_plan.md` (47 gates
across 4 steps, a 34-arm battery with 3 positive controls and a MUST-STAY-GREEN
column, 11 findings, 8 open rulings, 7 declared untestables). Design note
`.claude/plans/cs15_medium_unweld_design.md`; census
`scratch/cs15_grounding_census.md`; campaign plan
`.claude/plans/space_and_kernel_binding_campaign.md` §2.5. Nexus absent from the
tool list (expected per the brief); everything grep/`ast`/probe-derived.

### L60a ⛔⛔ `getattr(obj, "attr", default)` swallows an `AttributeError` raised INSIDE the property — so a "type-absent attribute" design degrades SILENTLY at every duck-typed consumer

The carve's centrepiece is a typed union (`MeshBackedRegions | QuotientPoint`)
whose quotient member makes five geometry attributes **type-absent** — a
`@property` that raises `AttributeError` with an honest reason. That is the right
shape, and it has one failure mode nobody looks for: **a consumer that reads the
attribute through `getattr(obj, name, default)` cannot tell "absent" from
"raised", because `getattr`'s default catches the exception the property raises.**

`[M]` `orpheus/transport/operators/multiplication_operator.py:344-346`:

```python
if space is None: space = getattr(mesh, "full_field_space", None)
if space is None: space = getattr(mesh, "bulk_space", None)
```

If `bulk_space` had joined the type-absent set (which one reading of the design's
§4.2 and the census's promise-1 wording both instruct), the homogeneous `C`
operator would go space-anonymous again, the `OperatorSum` guard would go back to
SKIPPING, and the whole CS1 D1 floor would un-do — **with no exception anywhere
in production**. The only reason a test notices is that CS1's D1 computes its own
reference through the same property and therefore ERRORs; a future D1 with a
tolerant reference would be green.

⟹ **Before designing type-absence for an attribute, grep
`getattr(<receiver>, "<name>"` and `hasattr(` across the tree and read what each
hit DECIDES.** `[M]` here: 6 `getattr`-with-default hits over mesh-family
receivers, 4 of them method-mesh trace spaces, 2 the chain above. The attribute
partitions into "may be absent" and "must stay legal", and the second class is
decided by the duck-typed readers, not by the concept.

⭐ And the companion, because the two documents disagreed: the census's promise
ledger said *"`bulk_space` migrates carrier → Medium; the property body rides
unchanged"* while the design's §4.3 said *"the carrier's own `bulk_space` …
STAYS"*. An implementer discharging the ledger literally deletes the property.
**A promise-ledger item is a work INSTRUCTION; when it contradicts the design
body, the ledger is the dangerous one, because it is the thing someone ticks
off.**

### L60b ⛔ Measure the float agreement of BOTH sides of a conformity guard before choosing `==` — "exact by construction" can be false in exactly one sub-family

The guard compares region interface positions (cumulative thickness sums) against
mesh edges. Both sides *look* exact: `from_geometry` accumulates the same
thicknesses in the same order.

`[M]` probe over **4902** random interfaces (400 trials × 2–6 regions × 3
geometries × 2 subdivision methods, `default_rng(20260820)`, thicknesses
`U(0.05,3.0)`, 1–8 cells/region), with the interface side computed BOTH by a
Python accumulate loop and by `np.cumsum` (identical):

| geometry | method | worst ULP | worst abs |
|---|---|---|---|
| SLB | uniform / equal-volume | **0.00** | 0.0 |
| CYL | uniform | **0.00** | 0.0 |
| CYL | **equal-volume** | **1.00** | `4.441e-16` |
| SPH | uniform | **0.00** | 0.0 |
| SPH | **equal-volume** | **1.00** | `8.882e-16` |

**10 of 4902** non-bit-exact, ALL curvilinear equal-volume. Mechanism:
`np.linspace` pins its endpoint to `stop` exactly, so the `uniform` arm is exact;
the equal-volume arm computes `sqrt(inner² + 1·(outer²−inner²))` /
`cbrt(inner³ + 1·(outer³−inner³))`, and the round-trip is the very thing
`_subdivide_zone`'s docstring warns about (`cbrt(x)**3 != x`).

⟹ `==` is a **latent** false red: green for as long as nobody meshes a
curvilinear multi-region geometry by equal volume, then a legitimate production
mesh is refused. The discipline is a derived ULP band (`4 × np.spacing(|x|)`,
4× the measured worst) whose discrimination margin is stated beside it — the
nearest WRONG edge is one cell away (`≈6e-3` on the plan's own fixtures), i.e.
**13 orders** above the band. And the battery carries the arm that PROVES the
band rather than asserting it: set the guard to `==` and exactly the two
CYL/SPH-equal-volume acceptance rows red while the four SLB/uniform rows stay
green.

⭐ The generalisable form: **when a guard compares two independently-accumulated
float quantities, the tolerance is a MEASUREMENT of their agreement over the
constructible population, per sub-family — never a judgement about whether the
construction "should" be exact.** A sub-family that goes through a transcendental
round-trip is invisible to the reasoning and obvious to the probe.

### L60c ⛔ An attribute→property conversion kills every committed `hasattr(Class, …)` PREMISE — and the replacement witness is usually one grep away

`[M]` `tests/test_docstring_xrefs.py:391` asserts
`assert not hasattr(SNMesh, "mesh"), "premise: 'mesh' is per-instance"`, using
`SNMesh.mesh` as the witness for the *unannotated instance attribute* resolution
shape. Making `mesh` a forwarding property on the base makes
`hasattr(SNMesh, "mesh")` **True** and the row reds on its own premise line —
not on the thing it verifies. Neither the design note nor a 600-line census
named it: the census swept `.mesh` READERS, and a `hasattr` premise is not a
read.

`[M]` the replacement was found with one `ast` scan of the hierarchy: the SNMesh
family has exactly **two** bare (unannotated) public `self.X = …` attributes —
`mesh` and `quad` — and `quad` satisfies all three of the row's assertions today
(`in _self_attributes`, `not hasattr(class)`, `resolve(...) == (True, None)`) and
is untouched by the carve.

⟹ **Before any attribute→property conversion, `grep -rn 'hasattr(' tests/` and
filter to the names being converted.** `[M]` in this tree that is one hit for ten
names — cheap, and it is the difference between a planned witness re-point and a
mystery red at execution.

### L60d ⛔ A "type X is `eq=False` for identity semantics" ruling can be FORCED by the field types — measure it on a toy before recording it as a design choice

The design note called `Medium`'s `eq=False` "identity semantics, the
`EnergyGrid` precedent"; the census called `EnergyGrid` "the anti-precedent … do
NOT copy its `eq=False`". Both were arguing style. `[M]` neither reading matters:

* `Mixture.__eq__` **raises** `ValueError: truth value of an array … is
  ambiguous` (a plain `@dataclass` over numpy fields); `hash(Mixture)` raises
  `TypeError: unhashable type: 'Mixture'`.
* a `@dataclass(frozen=True)` (⟹ `eq=True`) holding `materials: dict`:
  `hash()` raises `TypeError: unhashable type: 'dict'` **always**, and `==`
  returns `True` only when the values are the SAME objects (dict comparison
  short-circuits on identity) and **raises** otherwise.

⟹ `eq=True` would ship an object whose `__eq__` raises on some inputs and whose
`__hash__` never works. Three lines of probe settled a two-document disagreement
— and the *test-design* consequence is the real output: **content identity must
then be gated on what the type MINTS (its spaces), never on the type's own
equality**, which is where the campaign's existing B11 gate already lives.

### L60e ⛔ An invariant arm's witness can be unreachable on ONE MEMBER of a union — check reachability per member, not per corpus

L59b recorded the corpus-tier form (every shipped mixture has `eg is None`, so
the `from_grid` arm had no witness). The member-tier form is sharper and it bit
here: the eg-coherence invariant ("assigned materials must agree on their energy
grid") is **unspellable on the infinite member**, which has exactly ONE assigned
material. Its only constructible witness is a *structured* medium with ≥2 regions
naming ≥2 materials — i.e. the arm the design itself ships production-unreached.
Left unsaid, a whole-invariant mutation reddens via the id-coverage arm and the
run reports "the invariant is gated" (vv #17's granularity trap, at the member
tier).

⟹ **For each arm of a multi-arm construction invariant, name the member AND the
input that witnesses it.** An arm with no witness on any member is either
deleted or declared unfalsifiable in its own docstring.

### L60f ⛔ A guard placed after the caller's own attribute reads is UNREACHABLE — and the error TYPE is the placement gate

`SNMesh.from_material_mesh` reads `material_mesh.axes` and `.mesh` in its own
body before calling `_init_core`. Under the union those are type-absent on the
quotient member, so a typed refusal placed inside `_init_core` never runs: the
forwarding property's `AttributeError` fires one frame earlier.

⟹ the guard goes at the TOP of the promoting classmethod, and — the useful part —
**`pytest.raises(ValueError, match=…)` IS the placement gate**, because
`AttributeError` is not a `ValueError`. One assertion covers both "it refuses"
and "it refuses in the right place", with no source inspection.

⭐ Companion, already in the digest but re-confirmed: `[M]` today that promotion
dies as a bare `AssertionError` (empty message) at `augmented_mesh.py:322` under
plain `python` and as `AttributeError: 'NoneType' object has no attribute
'coord'` at `reduced_operator.py:858` under `-O` — the canonical runner strips
the assert, so the "guard" is an accident (`coding-standards`' bare-assert
discriminator). A mutation that deletes the new guard still reds the gate, **by
raising** — so only the `match=` leg attributes it (L31).

### L60g ⭐⭐ The byte-gate blindness partition, MEASURED end-to-end — and it is the plan's granularity pair

Same fixture (`homo_2eg_n2n`), same entry (`solve_homogeneous_infinite`), two
mutations of "the quotient's measure":

| mutation | `k_inf` | `flux` | `sig_prod` | `sig_abs` | D5 |
|---|---|---|---|---|---|
| baseline | `1.6532258064516119` | `[397.94608472, 359.4351733]` | `0.13203389830508477` | `0.11613559322033898` | — |
| **space weight ×2** (`bulk_space`'s `weights=`) | same | **same** | **same** | **same** | ⭐ **GREEN** |
| **volume ×2** (`volumes` ⟹ `volume_measure`) | same | `[198.97304236, 179.71758665]` | `0.26406779661016955` | `0.23227118644067796` | **RED** |

Two facts worth carrying: (i) L59a's space-side Mode-12 dual holds END-TO-END, not
just at `.H` — a rank-1 point axis makes the space measure invisible to every
value the solve produces, so **space identity is the only instrument**; (ii)
`k_inf` is blind to BOTH, because it is a production/absorption RATIO — its
stabiliser contains the whole measure, so **no k-level row may ever be credited
for a measure claim**.

⟹ "mutate the measure" is ambiguous until you say WHICH measure. In this carve
`volumes` feeds both `bulk_space`'s weights and `volume_measure`'s, so the
single-mutation instinct conflates them; the honest battery has one arm per
consumer and the pair IS the vv #17 proof.

### L60h ⭐ Sizing: measure the reachable subset, and let the excluded numbers justify themselves

`[M]` `tests/numerics` whole = **329.66 s**; the four files this carve can reach
= **2.72 s** (a 122× tax). `[M]` `tests/sn` whole ≈ **80 min** (extrapolated from
~6 % in ~5 min, 3329 collected) ⟹ it belongs to the pre-merge ≥90-min gate under
the campaign's BRANCH-HOLD ruling, never to a per-arm battery. The scope that
remains — homogeneous + transport + diffusion + sn/{mesh,primitives,architecture}
+ 4 numerics files — is `[M]` **1258 passed / 1 skipped / 17 xfailed in 25.30 s**,
so a 34-arm battery is ~15 min instead of days. Also `[M]` the pyright ratchet's
single remaining error is `orpheus/transport/operators/scattering.py:761` — in the
edited PACKAGE but not in any edited FILE, and the ratchet reds in BOTH directions,
so "new code adds none" is necessary but a carve that accidentally BURNS it must
re-baseline in the same commit.

---

## L61 — CS4a (the kernel core, PRE-carve): the charter's flagship gate cannot red, and its construction refusal DESTROYS 250 of 845 rows

Dispatch 2026-08-20, branch `feature/cs1-energy-space` @ `54bc6165`. Deliverable
`scratch/cs4a_verification_plan.md` (24 gates, 25-arm battery, 10 findings,
3 open rulings). Tree clean at start AND close; every probe in the session
scratchpad, every plugin loaded via `PYTHONPATH`, nothing written into `orpheus/`
or `tests/`.

### L61a ⛔⛔ A campaign's flagship NUMERICAL gate can be a theorem with no reachable falsifier — and the charter said so in its own subordinate clause

The CS4a done-when required *"the counting-measure adjoint THEOREM gate lands"*,
justified as: *"on Energy ⊗ point the metric is counting ⊗ counting, so
`.H == apply_transpose` becomes a THEOREM of the posed space instead of the R2
silent degradation **that today produces the same equality for the wrong
reason**."* The italic clause **states the blindness** and the gate was chartered
anyway.

`[M]` `max|.H.apply(x) − apply_transpose(x)|`, `python -O`, `get_mixture("A","2g")`:

| configuration | C | IsoS | IsoN2N | F |
|---|---|---|---|---|
| `space=None` (the R2 DEFECT) | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `space=` quotient E⊗pt (the FIX) | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `space=` spherical meshed bulk, volume spread **56 000×** | `5.551e-17` | `2.220e-16` | `0.000e+00` | `5.551e-17` |

A peer had measured the same at 3358×; re-running at 56 000× (edges
`[0,.05,.2,.5,1,2]`, spherical) confirms it is not a spread problem. The closed
form: `A† = G⁻¹AᵀG`, so `A† = Aᵀ` iff `[G, Aᵀ] = 0`; all four leaves are
**spatially diagonal** and every reachable bulk metric is `V_cell ⊗ counting`,
which commutes exactly. The only loading axis is a **per-group energy weight**,
and `EnergyAxis` **refuses weights at construction** — so the loading is
unreachable *by a construction invariant the same campaign shipped one phase
earlier*.

⟹ Two repairs, both cheap, and the plan ships both: (i) gate the theorem's
**PREMISE** instead (`space.apply_metric(x)` is `array_equal` to `x`;
`inner_product == np.sum(x*y)`; both axes `weights is None`) — red-capable by
mutating the mint; (ii) keep ONE corollary row, labelled claim-kind **THEOREM**,
whose docstring carries the blindness table and names the pre-existing D4b
(`test_H_MOVES_under_a_per_group_weighted_axis`) as the only loaded partner.
⛔ Do NOT manufacture a wrong-metric control on the production mint: `EnergyAxis`
refuses it, so the control is unconstructible, and D4b already does it with a
deliberately non-physical generic `Axis`.

⭐ The transferable tell: **a gate's justification containing "…for the wrong
reason" / "…which today produces the same result" is the author having already
computed the stabiliser and not acted on it.** Grep a charter for those phrases.

### L61b ⛔⛔ "Key the construction refusal on the space's SHAPE" — measured, it destroys 29.6 % of the suite; the axis-keyed alternative is inert on 81.2 % of constructions

Three independent design assemblies converged on a ng-conformity construction
refusal, and the ruling that survived selection said to key it on **NON-OPTIONAL**
content — *"materials' ng vs the space's shape"* — precisely to escape the vv#28
hazard (an `Optional` axes field is `None` on the composite). `[M]` the escape
does not exist:

| space handed to a production binding | `axes` | `shape` | where is ng? |
|---|---|---|---|
| homogeneous `MaterialMesh.bulk_space` | `(EnergyAxis, Axis)` | `(2, 1)` | index 0 |
| `SNMesh.full_field_space` | **None** | **`(64,)`** | **nowhere — FLAT** |
| `SNMesh…interior_space` | None | `(4, 2, 6)` | index **1** (ordinate-first!) |
| `DiffusionMesh.full_field_space` | **None** | **`(20,)`** | **nowhere — FLAT** |
| `DiffusionMesh…interior_space` | None | `(2, 6)` | index 0 |

and `[M]` `FunctionSpace`'s whole public surface is `apply_inverse_metric,
apply_metric, axes, dual, has_coordinate_cone, inner_product,
inner_product_weights, norm, of_axes` — **no ng, no energy accessor**, and
`FullFieldSpace` adds none.

Then the two candidate guards were **installed as pytest plugins and run**
(`vv` #29's per-INSTANCE census, not a static site count; each plugin raises
`RuntimeError` unless it binds 4 of 4 classes, `lessons` §2). Sub-scope baseline
`[M]` **845 passed / 1 skipped / 17 xfailed, 23.89 s**:

* **`space.shape[0] != kernel.ng ⟹ raise`** → `[M]` **182 failed, 595 passed,
  68 errors** — **250 of 845 rows destroyed (29.6 %)**. Unrunnable, not weak.
* **axis-keyed** (fire only when the space carries an `EnergyAxis`) → `[M]`
  `{'checked': 192, 'skipped_axesless': 578, 'skipped_nospace': 252,
  'raised': 0}`, **845 passed**. Over **1022 instrumented constructions** it is
  live on **192 (18.8 %)** and inert on **830 (81.2 %)** — and it **raised 0
  times**, so it has *no witness anywhere in the shipped suite*.

⭐ Two numbers, two different lessons. The **site-level** count (4 of 13 bindings
axis-bearing) is what the design records carried; the **instance-level** count
(192 of 1022) is what production actually does, and it is 4× worse as a fraction.
⟹ **a guard's inertness fraction belongs in its docstring, measured at the
INSTANCE tier, not the site tier** — a site census counts call lines, a running
suite counts what those lines execute.

⭐⭐ And the third candidate, which looks like the clever escape and is a provable
non-catcher: **divisibility** (`total_dof % ng == 0`). `[M]` SN total `= 32·ng`,
diffusion `= 10·ng`, so a **4g** kernel on a **2g** space is ADMITTED at both —
and 2g↔4g is the *only* mismatch pair the tree ships fixtures for (the D2/D3
witnesses). A guard that admits the only witness it will ever meet is worse than
no guard.

### L61c ⛔ A re-pose can INVERT a migration gate's sensitivity partition — the inherited `[M]` row dies by being FIXED

CS1's byte gate D5 was characterised (and the characterisation was carried into
the objectives file as a salvaged row, and into a sibling plan's §2(e)) as:
*"BLIND to space weights, LOADED on cell volume; `k_inf` blind to both."*
`[M]` reproduced end-to-end through `solve_homogeneous_infinite("homo_2eg_n2n")`:
volumes ×2 ⟹ `flux 397.94608472 → 198.97304236`, both rates double, `k_inf`
unchanged; space weight ×2 ⟹ **every field bit-identical**.

CS4a re-poses the two homogeneous `IntegratedReactionRate` sites to
`space.inner_product`. `[M]` the values are **bit-identical** (0 ULP, 6 of 6
rows over ng ∈ {1,2,4} × {production, absorption}) — so the re-pose is a value
no-op and D5 stays 8/8. But the *sensitivity* swaps sides: the old spelling ends
in `mesh.volume_measure(...)`, the new one in the space's per-axis weights. After
K2, D5 is **LOADED on the space weight** and **BLIND to the carrier volume**.

⟹ CS1's anti-claim arm ("space weight ×2 ⟹ D5 GREEN") becomes a **must-RED**
arm, and a brand-new must-stay-GREEN arm appears ("volumes ×2 ⟹ D5 GREEN") that
is the *un-wiring proof* — a claim that could not be stated before the change.
Ship BOTH, at BOTH HEADs, and put the 2×2 table in the gate's docstring.

⭐ This is `plan-authoring` §3's "a fact can die by being FIXED" at the
**verification-instrument** tier: nobody refutes the row, the campaign simply
repeals it, and the row keeps reading as current because it was true.

### L61d ⛔ A doc paragraph can carry an honest `[M]` whose LOAD-BEARING half is false — and the carve is what makes it true

`docs/theory/foundations/spaces.rst:926-937` argues *"the quotient point's weight
is genuinely consumed, not decorative"*, offering as `[M]` evidence that
`IntegratedReactionRate.evaluate` *"contracts against `mesh.volume_measure`"*,
with `0.225` (quotient, weights `[1.0]`) vs `0.450` (a one-cell slab of width 2,
weights `[2.0]`). `[M]` reproduced today, exactly, both spellings.

But the experiment varies the carrier volume and the space weight **together** —
vv#17's granularity trap, at the doc tier — and the separated probe shows the
**space** weight is bit-identically inert on the value path. So the measured half
is true and the inference is false: `plan-authoring` §2's "so/therefore" defect,
living in the corpus rather than in a plan. K2's re-pose is precisely what makes
the conclusion true, and simultaneously makes the *mechanism* clause
present-tense-false.

⭐ And the sharpest residue: the same page carries
`.. implements:: normalisation :by: …IntegratedReactionRate.evaluate` — a
**declared** graph edge onto a symbol the homogeneous path will stop calling, on
a page whose own prose says *"a test is adjudicated against the transcription it
actually ran"*. A re-pose's doc blast radius therefore includes the **provenance
directives**, not only the prose; and `dead_references` cannot see it (the symbol
still exists — it is the *caller* that changed).

### L61e ⛔ An accessor that returns its own CACHE, writeable, is a production-reachable mutation channel — and a "frozen kernel view" inherits it

`[M]` `MaterialXSField.sig_s_legendre(mid)` returns the cache **object itself**
(`is` True across calls, for the list AND its elements), `.flags.writeable` True,
`owndata` False, and the element **is** `mx._sig_s_dense[mid][0]`. Reach:
`stack[0][0,0] += 1.0` then re-assembling gives `loss[0,0] 0.152 → −0.848`, delta
exactly `−1.0`. `n2n_matrix(mid)` behaves identically. (`dense_per_material()`
returns a fresh copy per call — it is the storage ORACLE and is unaffected.)

⟹ three consequences for a carve that mints a `@dataclass(frozen=True)` kernel
described as "a view over `Mixture`, absorbing/delegating the dense caches":
(i) frozen-ness is a **name only** unless the kernel copies and sets
`writeable=False` (CS1's `Axis` already ruled this shape — the `+0.0` that forces
a defensive copy); (ii) the equivalence gate must be **bit-identity
(`array_equal`), never view-identity (`is`)** — an `is` gate would assert the
hazard as the contract; (iii) the honest REFERENCE is the `Mixture`'s **sparse**
data (`[M]` `sig_s_legendre(0)[0]` is `array_equal` to
`np.asarray(mix.SigS[0].todense())`), because it is independent of the
absorb-vs-delegate choice — whichever the design picks, a cache-vs-cache gate
goes tautological (`coding-standards`' single-sourcing clause) while the
sparse-source gate survives.

### L61f ⭐⭐ "NO arm deleted" is a BEHAVIOURAL matrix, not a grep — and a registry-keyset gate is blind to exactly the operators that need it

The phase's headline fence (from `vv` #29's refutation of space-keyed arm
selection) is *"NO apply arm is deleted"*. `[M]` the cheap instruments:

* `singledispatchmethod` registry keysets, readable at runtime —
  `MultiplicationOperator._apply_impl` → `{FullField, ndarray, object}`;
  `FissionOperator` → `{FullField, ScalarFlux, ndarray, object}`;
  `ScatteringOperator` → `{AngularFlux, FullField, HarmonicMomentFlux,
  ScalarFlux, object}`; ⛔ **`IsotropicScattering` / `IsotropicN2N` → `{}`** —
  they have NO singledispatch at all; their arms are `isinstance` branches.
* the **behavioural matrix**, 4 operators × 3 carriers, `[M]` every cell
  distinct: `C×ScalarFlux → TypeError`; `IsoS/IsoN2N × ScalarFlux → bare
  ndarray` (an untyped fall-through through `getattr(phi,"values",phi)` — `vv`
  #29's *asymmetric arrow*, typed in / bare out); `F×ScalarFlux →
  ScalarSourceSink`; all four × `FullField → FullField`; all four × `ndarray →
  ndarray`.

⟹ ship the matrix as the gate and the keysets as a companion, and say WHY: a
registry-only gate certifies 3 of 5 classes and is structurally blind to the 2
whose dispatch is an `isinstance` chain — which are also the 2 the carve is most
likely to "tidy". A source grep is weaker still (dies on reformatting, on the
`singledispatchmethod` → explicit-dispatch rewrite a later phase may do).

### L61g ⭐ The silent failure of a marker SPLIT is `strict` loss, and both obvious gates are blind to it

A step that converts function-level `@xfail` decorators into per-row
`pytest.param(..., marks=[...])` is chartered as ledger-preserving. `[M]` the two
natural gates — `--collect-only -q` node-id identity (98 lines) and `-rx`
status+reason identity (16 `XFAIL` lines) — are both **blind** to spelling the
new mark `pytest.mark.xfail(reason=…)` **without `strict=True`**: the row still
reports `x`, the ids are unchanged, the reasons are unchanged, the suite is
green. And `[M]` `pyproject.toml` carries **no `xfail_strict`** ini key, so the
default is non-strict.

The gate that catches it is 5 lines and permanent, because `[M]`
`pytest.param(...).marks` is introspectable at import (`_G13_ROWS`' B row reads
`[('xfail', True)]`, its L row `[]`):

```python
for row in (*_R1_ROWS, *_R2_ROWS, *_G13_ROWS):
    for m in row.marks:
        if m.name == "xfail" and m.kwargs.get("strict") is not True: fail(...)
```

Losing `strict` costs the campaign its self-retiring todo list — the entire
reason the ledger exists (`vv` Mode 8, FOURTH class). ⭐ Also worth knowing: a
*dropped* mark is a **visible RED** for both families here (an unmarked
annotation row fails on the Optional; an unmarked refusal row fails because the
constructor does not raise), so only the `strict` half is silent.

### L61h ⭐ A brief's "flip X if free" is a claim to MEASURE, and one gate DIES rather than flipping

`[M]` the R1 family asserts the **return annotation** of the `domain` property on
the owning class (`L Optional['FunctionSpace']`, `C/S/F 'FunctionSpace | None'`,
`B Optional['FunctionSpace']`). CS4a flips **F only** — a sibling ruling defers
C's mandatory flip to a later phase with its `[M]` 131-of-145 space-less test
migration — so **R1-C does NOT flip free**; the charter's parenthetical resolves
to NO and the ledger goes **16 → 14** (R1 5→4, R2 3→2, R6 stays 8), which is a
number a later reader will otherwise hunt as a regression.

And the flip **kills a gate outright**:
`test_from_solver_data_does_NOT_default_derive_a_space` constructs the operator
space-less as its whole subject, and its own docstring already carries the
trigger (*"WHEN CS4's Optional→mandatory flip lands: delete this gate"*). ⛔ It
must be DELETED, not repaired — adding `space=` to its body turns a real pin into
`X == X` while keeping its authoritative name.

⭐ Sizing the flip honestly: `[M]` **15** `FissionOperator` constructions in
`tests/`, **10** space-less, of which **1 is a message STRING** (found by reading
it, not by the regex) ⟹ **9 real sites in 5 files**, two of them the decay items
above. A regex census over a 7-line window is right for finding candidates and
wrong for counting them.

## L62 — CS4b (fields are space elements, PRE-carve): 8 of 22 identity guards catch nothing, and the "space re-point" moves a shipped diagnostic 41 %

**Dispatch.** Design the CS4b verification plan before implementation, from
`.claude/plans/cs4b_fields_design.md` (rounds 0–2, F1/F2/F5 ruled) + two censuses.
Deliverable `scratch/cs4b_verification_plan.md` (987 lines, 16 sections).
Branch `feature/cs1-energy-space` @ `34df88cb`.

### a. ⛔⛔ The hoisted-guard witness table, measured BEFORE the plan shipped

F2 re-keys partner identity from `mesh is` to space content equality. `vv` #17
says such a guard has as many arms as CALL SITES. I enumerated **21** (later 22 —
see (b)) and MEASURED the witness table: disable each guard (`if <cond>:` →
`if False:`), run a 548-row / 3.31 s attribution scope, count reds.

`[M]` **7 arms redden NOTHING** over a **3936-row** denominator (14-path battery
1436 + `tests/sn/operators` 1230 + `tests/sn/solve` 202 + a partial 1068-row
sweep/moc/cp leg): `diffusion/operators.py:603`, `sn/solver.py:3091`,
`sn/solution.py:475`, `radial_characteristic.py:1176`, `windowing.py:114`,
`boundary.py:715`, `streaming.py:1065`.

⭐ **The shape worth carrying: two of the seven are the `apply_transpose` and
second-`solve`-arm TWINS of witnessed forward arms** (`radial_characteristic.py:944`
→ 1 red; `:1176` → 0. `streaming.py:906` → 2 reds; `:1065` → 0). A whole-guard
mutation would have reddened on the forward arm and certified both.

⛔ **This is not "pre-existing debt to file".** A rewrite of an unwitnessed guard
is indistinguishable from a DELETION of it, and it reads green.

**Instrument**: `/tmp/cs4b_mutate.py` — copy-aside outside the repo, per-arm
byte-`diff`, positive control (all-21 → 22 reds in the battery + 7 in
`sn/operators`). The 548-row scope was constructed FROM the positive control's
red set, which is why the 21-arm loop cost 21 × 3.3 s instead of 21 × 25 s.

### b. ⭐⭐ A 22nd site the design record missed — and the general rule it yields

The record named ONE `getattr(x,"mesh",None)`-tolerant gate. `[M]` there are
**four** such spellings tree-wide, and the fourth is a live guard:
`sn/solver.py:338-345` refuses a bare System-A residual on a carrying mesh, its
own comment naming the Mode-12(b) blindness its removal re-opens. `[M]` **no
test witness** (3 fragments, 0 hits).

⟹ promoted into `vv-principles` #28 as the **TEMPORAL twin**: *a guard whose
predicate reads an attribute through a DEFAULTED `getattr` goes silently inert
the day that attribute retires* — the defensive spelling is what makes the death
silent instead of an `AttributeError`. The retirement audit's FOURTH search:
grep the retiring name inside `getattr(`/`hasattr(`.

### c. ⛔⛔ "Re-point the space" INSTALLS a metric — 41 %, not ULP

Bulk field spaces today are bare `FunctionSpace(name, shape)` with **no metric**,
so `Field.l2` is flat Euclidean. `[M]` on a NON-uniform 5-cell slab
(`V=[0.2,0.3,0.4,0.7,1.4]`, GL(4), ng=2, `rng(0)`):
`AngularFlux.l2` **4.99286387678374 → 2.9593324544042217** (ratio **0.5927**);
`HarmonicMomentFlux.l2` ratio **0.8214**.

Consumers: `numerics/iteration.py:740` (`increment_norms` → ρ, `true_error_estimate`)
and `residuals/angular_residual.py:206` (`relative_to`). ⭐ The SI **STOP is
unaffected** — `[M]` it rides `_l2_norm(rhs_prev − rhs)/q_norm`, a bare-ndarray
norm — which is why the change is invisible to a "does the solve still converge"
check.

⭐ **The gate that predicted it names the WRONG PHASE.**
`tests/numerics/test_si_diagnostic_trajectory.py` (CS3) carries in its own
docstring *"CS2 will legitimately RED this gate … CS2 owns re-deriving these
frozen numbers"* and an `[M]` battery row `M2 Field.l2 → np.linalg.norm` = **5
passed** (the declared blindness). F1-(A) makes the metric arrive at the FIELD
layer in CS4b, so **CS4b reds it, not CS2**, and the M2 blindness prose becomes
false. Lesson: *a phase-ordering hazard written into a gate is itself a claim
with a shelf life* — re-check WHICH phase owns it whenever an earlier phase
adopts the mechanism.

### d. ⭐⭐ Two embeds `Σw` apart are the ADJOINT and the SECTION

The design's F3 says "`E ∘ R` is the isotropic projector". `[M]` the tree ships
**two** embeds and they differ by exactly `Σw`:
`HarmonicFrame.reconstruct` at `L=0` vs `AngularSourceSink.from_isotropic` —
`max|Δ| = 6.77e-01`, ratio `min = max = 2.0` = `Σw`, and `reconstruct/Σw == E`
**bit-exactly**. A single `space.embedding(axis)` verb would have been a
factor-`Σw` landmine (the ERR-051 class).

⭐ Resolved constructively, not by choosing: `[M]` `⟨Rψ,φ⟩_M = ⟨ψ, R†φ⟩_P` at
**nulp 1.0** where `R†φ = broadcast(φ)` — i.e. **`reconstruct` IS the metric
adjoint of the retraction**, and `E = R.H/Σw` is the SECTION defined by
`R∘E = id` (`[M]` **BIT-EXACT**, `array_equal`). Name them apart and the missing
factor becomes unspellable.
Also `[M]` bit-exact and gate-ready: `E∘R` idempotent; `HarmonicFrame.analyse`
at `ℓ=0` ≡ `integrate_angular` (`max|Δ| = 0.0`) — a THIRD spelling of `R` that
already agrees.

### e. The census's own denominator was 1.4× low

`[M]` AST over `tests/`: the migration surface is **909 sites in 122 files**, not
"632 in 86". The three uncounted buckets: composite `zeros(…, mesh=)` **121**
(`TimedFullField` 114 / `FullField` 7 — a SECOND factory tier), non-`self`
`.mesh` READS **116** (not a signature migration at all — each is a consumer that
must find the mesh elsewhere), BC factories **25**. Production mirror ≈182.

⟹ the step-decomposition rule that falls out (§6b, and it is an argument about
call-site sets, not effort — the user's meta-ruling bars the latter):
**bodies → consumers → signatures.** Change what a factory BUILDS while its
signature stays; re-point every consumer while the attribute still EXISTS; delete
the field last, in a 2-line diff whose done-when is a grep predicate.

### f. Measured facts that shaped gates, one line each

- `[M]` F2 discrimination table: axis-built angular space `==` is **False** on
  {volumes, quad order, ng} and **True** on {twin instance, BC-only}. But the
  SCALAR `bulk_space` is **True** on quad order too ⟹ **F2 silently permits
  `φ_S4 + φ_S8`** — a permission strictly wider than the BC one the record
  argued, and unrecorded.
- `[M]` after F1-sub the three angular role leaves share ONE space, so the space
  arm of `_check_partner` stops discriminating ROLE — the class arm becomes sole
  enforcement. Today a class-arm-deletion mutation is MASKED by the space arm
  (the three space names differ), so that gate is genuinely NEW coverage.
- `[M]` the axis-built Gram reproduces `full_field_space.interior_space`:
  `inner_product` **bit-identical** (DD and LD), `norm` bit-identical (DD) / 1 ULP
  (LD), `apply_metric` `max|Δ|` **2.78e-17** (DD) / **1.11e-16** (LD) ⟹ scalars
  `array_equal`, vectors `nulp ≤ 4`.
- `[M]` the LD arm WORKS (`moment_mass_diagonal = [1, 1/3]` IS an axis weight) —
  but the moment axis is MODAL ⟹ `has_coordinate_cone` **False** ⟹
  `Field.cone_violations` flips from ANSWERING to **REFUSING** on every LD bulk
  field. Correct physics, unnamed behaviour change.
- `[M]` `FullFieldSpace.__eq__` is `(name, shape)` with `name="full_field"` a
  LITERAL and the blocks `compare=False` ⟹ **block-blind**. A composite-level
  `space ==` gate would be designed-green; the cross-slot re-key must compare
  BLOCKS by `is`.
- `[M]` #399's red witnesses ALREADY EXIST: on a `spatial_moments=2` moment field,
  `truncate` / `isotropic_part` / `anisotropic_part` all raise `ValueError` today;
  `scalar_flux` / `l_block` are correct. §6c is satisfiable with no manufacturing.
- `[M]` the repairs' witnesses all construct: SN promotion from the meshless
  carrier raises a **messageless** `AssertionError` under plain python and
  **`AttributeError: 'NoneType' object has no attribute 'coord'` under `-O`** (the
  canonical runner); all three `.areas` arms are reachable (2-D legacy → message
  TRUE; `SNMesh.from_axes` d=3 and the quotient carrier → message FALSE), with
  **0** test pins tree-wide.
- `[M]` `Axis` identity is per-SUBCLASS and `of_axes` derives the space NAME from
  the axes' structural bytes ⟹ **CS2's `QuadratureAxis`/`SpatialAxis` will change
  every axis-built space's name**. No CS4b gate may pin the derived name literal.
- `[M]` the battery-scope amendment costs **+175 rows / +2.90 s** — the four files
  carrying 7 of the 29 identity reds AND the gate (c) predicts will red are NOT in
  the inherited 14-path scope. A CS4b run of the inherited battery would miss its
  own headline regression.
- ⚠ Method note: the `parametrize`-collection-kill worry (`vv` #17) was CHECKED,
  not assumed — `[M]` AST over the 12 attribution paths: **0** module-scope or
  parametrize-list field constructions. I had written the hazard as live; the
  measurement refuted my own sentence.

## L63 — P4.9a (the per-cell angular un-weld, PRE-carve): the phase's named canary is a provable non-catcher, and a THIRD spelling of the relation is 204 ULP away

**Context.** ORPHEUS `.claude/plans/streaming_path_says_what_it_is.md` P4.9a —
`DiamondDifference.update` deletes its inline Morel–Montry twin,
`cell_balance_terms` retires, the L2 spatial protocol sheds its angular members,
and the closure hands the cache its march constants. Plan:
`scratch/p4_9a_verification_plan.md`. Measured at `5c4f56d7`, all probes
in-process monkeypatch (no production file edited on disk).

### L63a ⛔⛔ The whole carve has ONE production execution route, and it is a CONGRUENCE CLASS nothing frozen samples

`[M]` counting spy on `DiamondDifference.update` over full 2-G heterogeneous
eigenvalue solves: **0** calls on SLAB `gauss_legendre(4|8)`, **0** on SPHERE
`gauss_legendre(4|8|9)`, **0** on CYLINDER `folded_product(4,4)` and `(4,8)`;
**13 760** on `folded_product(4,6)` and **13 536** on `(4,10)` — every one
through the M-M branch.

The mechanism, `[M]` exhaustively over `n_phi = 4…34` even: the staggered
azimuthal circle hits `ω = π/2` **iff `n_phi ≡ 2 (mod 4)`**, placing one
bit-exact `η = 0` ordinate per μ-level (`deg = n_mu`); `n_phi ≡ 0 (mod 4)` ⟹
`deg = 0`, 16 of 16. Sphere is `deg = 0` at every order **including odd
`gauss_legendre` where `min|μ| = 0.0` exactly** — that ordinate still carries a
downstream face, so it is not degenerate.

⟹ **Every frozen artifact in the tree is blind**: the regression snapshots
(`_generate_snapshots.py` cyl at `n_phi ∈ {4,8}`), `walk_matvec_cyl_2g.npz`
(via `_make_cyl` = `fp(4,8)`), the affine-carve baseline's `CYL` row
(`n_phi = 2·_N_ORD = 8`), and — the finding that matters — **the plan's own
named done-when, the aniso curvilinear canary** (`n_phi ∈ {8,16}`). It will be
bit-identical unconditionally, because it never executes the changed line.

⭐ The rule: **when a code path is gated by a parameter's CONGRUENCE CLASS,
the frozen corpus almost certainly samples one class only** — `4, 8, 16, 32`
looks like a refinement ladder and is a single residue (`vv` #13's ladder trap
at fixture scale). Before crediting any snapshot as a carve's anchor, **run a
counting spy and confirm the changed line executes.** `[M]` here the spy cost
0.6 s per config and refuted a four-artifact done-when.

⭐⭐ And the aggravator that made it survive: the tree ALREADY knew. Authored
comments at `test_apply_matvec_cylinder_invariants.py:70-76` and
`test_g_adjoint_reciprocity.py:189-210` state the `n_phi ≡ 2 (mod 4)` rule
exactly, and ship the right fixture pair (`_make_cyl` blind / `_make_cyl_product`
activating). The knowledge was authored and the frozen corpus never adopted it —
`nexus-tools`' "knowledge can be AUTHORED and INERT", at fixture level.

### L63b ⛔⛔ A "delete the twin" carve found a THIRD spelling, in a DIFFERENT arithmetic form

The brief described one owner (`closure.py:1329`) and one twin
(`diamond.py:229`), both `(ψ̄ − (1−τ)ψᵃ)/τ` — **Form A**. `[M]` a third
production site exists outside the done-when's `transport/` scope:
`cache.py:377`'s `mm_a_in_coeff = (1.0-tau)/tau`, consumed at
`loss_representation:4348` as `τ⁻¹ψ̄ − ((1−τ)/τ)ψᵃ` — **Form B**.

Algebraically identical, **not** bitwise: `[M]` on the real τ of `fp(4,6)`,
bit-equal **59.13 %** of 2400 evaluations, max **204 ULP**; on random
τ ∈ [¼,¾], 49.5 % and **37 559 ULP**.

⭐ The sub-family that hides it: bit-equality is **100 %** exactly when
`τ == 0.5` bitwise (`1/τ = 2.0`, `(1−τ)/τ = 1.0` are exact) and 54–57 % at
`τ = ½ ± 1 ULP`. `[M]` on the degenerate ordinates τ is ½ *up to 1 ULP* and
exactly ½ only sometimes — `fp(4,6)` 2 of 4, `fp(4,10)` 0 of 4, `fp(6,6)` 0 of
6. A bit-identity gate validated on one ordinate of one config is reading a
coin (`vv` #31 + #13).

⟹ **RULING extracted: when a carve re-routes a relation, ask which ARITHMETIC
FORM the destination spells, not just which module owns it.** `[M]` routing the
degenerate branch through Form B costs `Δkeff = 2.776e-17` (1–2 ULP, harmless)
**and breaks `array_equal` on 3 of 4 configs** — a re-baseline for zero
architectural gain. Route through the owner's form; bit-identity is then free.

⭐ Second-order consequence worth carrying: **the `is`-identity gate the charter
asks for is only BUILDABLE if the branch calls a closure METHOD.** If the
implementation inlines cache constants instead, no closure object is reached
and the gate cannot be written. Two independent arguments converge on one
design ruling — say so, because a gate's buildability is a legitimate design
constraint and reads as one only when stated.

### L63c ⛔ Shedding a protocol field DESTROYS every guard that keys on it — here, the #158 curvilinear refusal

`[M]` `LinearDiscontinuous._require_slab` keys its curvilinear refusal on
**`upstream_state.angular_upstream is not None`** — the exact field the carve
removes — and its sole witness (`test_curvilinear_visit_raises`) constructs
`UpstreamState(angular_upstream=…)` directly, so after the shed it is
unwritable. `vv` #17's displaced-guard clause and #28's temporal twin at once.

⟹ **A protocol-shedding step owes, in the SAME commit: the re-keyed guard AND
its re-written witness.** `[M]` the replacement is a VALUE signal already on the
visit — `streaming_terms.delta_A_over_w` is exactly `0.0` on CARTESIAN (4/4
cells) and non-zero on SPHERE (`62.9…9.3`) and CYLINDER (`4.29`) — and `vv` #17
prefers it precisely because a value-keyed guard is reachable by calling the
scheme directly, needing no mesh.

⟹ **The generalisable audit step:** for every field a carve removes, grep it as
a GUARD PREDICATE (`is None` / `is not None` / `getattr(…, default)`), not only
as a read. `plan-authoring` §8 says a field a consumer BRANCHES on is an input;
the mirror is that REMOVING such a field silently disarms the branch.

### L63d ⭐ "Nothing gates them against each other" ≠ "nothing gates it" — read the claim's quantifier before believing the gap

The memo said nothing gates the two spellings. True, and narrower than it reads.
`[M]` in-process mutation of the twin (`outgoing_angular_state *= 1.05`) reds
**13 rows**: 3 unit (`test_diamond.py`) + 10 end-to-end (the `n_phi=6`
`three_way_standoff`, `psi_half_positivity`, `sweep_inverse_identity`,
`loss_transpose_solve` rows). So the twin has a real, if narrow, net.

`[M]` mutating `cell_balance_terms.denom` reds **55** over the four
helper-naming files (`test_diamond` 33 / `test_cache` 19 /
`test_cell_balance_for_streaming` 3 / `test_ordinate_scan` **0**).

⟹ Two corrections a memo's gate list will not give you, both from ONE mutation:
* **The load-bearing equivalence family was not on the list.** 25 of the 33 are
  `TestResidual` rows, which red because `update` (helper A) and `residual`
  (helper B) disagree — they ARE the cross-helper gate, more than the two rows
  actually named `…_matches_cell_balance`.
* **A listed gate reds ZERO.** `test_ordinate_scan.py` uses the helper to BUILD
  the `(a,b)` it feeds the SUT, so both sides move together (`vv` #22's
  shared-input blindness). It is a fixture generator, not a reference — the
  rewire costs no claim, and saying so in its docstring is the deliverable.

⟹ **Run the mutation before writing the rewire prescriptions.** A per-gate
claim-class verdict guessed from reading is wrong in both directions: it invents
demotions that are not there and misses the family that carries the claim.

### L63e ⛔ Retiring a helper breaks DECLARED provenance edges, and one equation loses its ONLY catchers

`[M]` `mcp__nexus__provenance_chain`: `cell_balance_terms` declares `implements`
on four equations; the survivor `cell_balance_for_streaming` already declares
three — **`dd-slab-scalar` is unique to the retiring one** and is orphaned
unless migrated. `[M]` `dd-mm-closure-constants` has **zero declared**
implementers (all inferred by name) and exactly **three** claiming tests — and
all three are the twin's catchers, so the carve drops the equation from
`verified` to `implemented` unless a closure-side replacement lands in the same
commit. `[M]` `DiamondDifference.update` itself declares
`dd-cylindrical-degenerate`; after the carve it closes only the spatial axis, so
that claim becomes present-tense false.

⟹ **`provenance_chain` on BOTH sides of a retirement is a mandatory step**, and
the question is set DIFFERENCE, not presence: *which declared edges does the
dying symbol carry that the survivor does not?*

### L63f ⭐ The cheapest anchor is usually a PARAMETRIZE ROW on the harness that already has the right regime

The gap was a bit-identity anchor that executes the degenerate branch. Rather
than a new module, `[M]` `test_affine_carve_baseline.py` already runs
`sweep_once` with heterogeneous σ_t + fixed-seed random per-ordinate source
(`vv` §H2 activated) under a `--capture-baseline` snapshot harness — and is
blind only because `_GEOMS_1D`'s `CYL` row computes `n_phi = 2·_N_ORD = 8`.

`[M]` adding a `CYL_DEG` row at `folded_product(4,6)`: `N=12`, `deg=4`,
`sweep_once` **2.1 ms**, `DD.update` **32 calls** all M-M. Two snapshots,
~2 ms, and the carve acquires its anchor.

⚠ **And it must land BEFORE the carve**, on unmodified production, or the
snapshot inherits the new code and pins nothing
(`snapshot_migration_when_production_goes_bare` rule 4).

⚠ Do NOT retune `_N_ORD` to reach the class — that silently re-baselines the
existing row. Add a member; never move a shared literal.

### L63g ⭐ The realistic mutation for a "hand it the constant" move is 1–2 ULP, and exactly one gate sees it

`[M]` the cache derives `mm_a_in_coeff = (1.0-tau)/tau` and `tau_inv = 1.0/tau`
from `closure.tau_per_ordinate` — moving those two lines into the closure is
bit-identical **by construction**, so a before/after gate is tautological. The
defect that will actually happen is the cleaner spelling
`mm_a_in_coeff = tau_inv - 1.0`, `[M]` **1–2 ULP** away on 4/4 configs.

`[M]` that mutation reds **exactly 2 rows** over 463 tests
(`TestAffineCarveSweepBaseline::test_sweep_angular_and_scalar_unmoved[SPH|CYL]`)
— so an external pin EXISTS and MOVES under the old value
(`coding-standards`' re-baseline licence requirement), and it is the only one.
⟹ gate with `array_equal`; any tolerance ≥ `1e-15` makes it a non-catcher.

### L63h ⭐ Harness note — a monkeypatch-only mutation battery is crash-safe BY CONSTRUCTION

Every arm here was a pytest plugin installing in-process wrappers, so no
production file was ever modified on disk and a `SIGTERM` at the harness timeout
could not leave the tree mutated. This is strictly stronger than
`process-discipline`'s copy-aside + `diff -q` (there is nothing to restore) and
it made 6 arms cheap to iterate. `[M]` scoping: `sweep/core` + `transport/spatial`
= 541 tests / 74 s; the `n_phi=6` slice = 108 / 19 s; the four helper files
alone = 140 / **0.8 s**.

⚠ Two environment traps hit on the way: **zsh does not word-split unquoted
`$VAR`**, so `pytest $SLICE` collected **0 tests** and read as a clean run —
use a driver script with `"$@"`. And a long background pytest writes a
**block-buffered** log that reads as empty for minutes; run a narrow slice in
the foreground instead of polling an empty file.

---

## L64 — P4.9b (the operator is posed with its two closures, PRE-carve): the phase's real §6c witness is a ROUTE gate, and the design's own cost claim was 17–25 % wrong

**Dispatch** 2026-08-28, at `10314dfa`. Design the verification plan for the
un-weld campaign's P4.9b BEFORE implementation. Ruled design
(`scratch/p4_9b_design.md` §§7–8): `StreamingOperator(sn_mesh, scheme,
pole_angular_closure)` — three required fields, no defaults, **no guards**;
`StreamingOperator.pose(sn_mesh)` reads the hub's two objects; 136 ctor sites
migrate; the representation receives the closure pair; the fused scan table
becomes the STRATEGY's lazily resolved artifact. Deliverable
`scratch/p4_9b_verification_plan.md` (12 findings, 4 §6c witnesses, an 11-arm
battery with the M1 denominator measured).

### L64a ⭐⭐ When the phase's claim is "consumer X now reads from OWNER A instead of OWNER B", the gate is a ROUTE gate: pose, SWAP the old owner's object, and require the answer UNMOVED

Every §6c witness the brief offered was post-hoc: the ctor-arity witness is
trivially red, the pose-identity gate is green the moment `.pose` is written,
the lazy-table and memo-retirement gates are consequences. The phase's actual
claim is a ROUTE claim (`vv` #26), and the instrument is:

```
L = StreamingOperator.pose(sn); _ = L.loss_representation
base = drive(sn, L)                      # (L + C).solve(rhs)
swap sn.scheme / sn.pole_angular_closure for a MUTANT subclass
drop the mesh-attr memos
assert np.array_equal(drive(sn, L), base)          # <- post-carve
```

`[M]` pre-carve it MOVES on every geometry: SLAB `GL8` scheme swap rel
**5.000e-02**; CYL `fp(4,6)` closure swap **4.596e-02**; CYL `fp(4,8)`
**5.313e-02**; SPHERE `GL8` **1.196e-01**; `array_equal` False in all four.

⚠ **Three traps, all measured, each of which makes it silently green for the
wrong reason:**

1. ⛔ **The insufficient mutation.** Mutating `MorelMontryAngularSweep
   .cell_contribution` alone reads `array_equal = True` on ALL THREE
   curvilinear rows — P4.9a's Q1 ruling split the surfaces (`.solve` consumes
   `advance_psi_half` + the minted scan constants; `cell_contribution` is the
   MATVEC's arm). One surface = one route certified. (L49c, in a new dress.)
2. ⛔ **The driver that re-poses.** `tests/sn/_test_helpers.sweep_once` builds
   `StreamingOperator(sn_mesh)` **internally** at `:814`, i.e. AFTER the swap —
   so post-carve it would still read the mutant and the gate would stay red for
   a reason unrelated to the carve. Drive `(L + C).solve` on the operator the
   test posed.
3. ⛔ **The cache that masks the swap.** Without dropping `_geom_cache` /
   `_coll_cache` the frozen table survives and the gate passes *because of the
   cache*. Post-carve the memo moves — keep the drop and add an ACTIVATION leg
   (a freshly posed operator over the mutant hub MUST move), else the gate is
   `X == X`.

⟹ the transferable rule: **a route gate needs the ACTIVATION leg, the whole
consumed surface, and a driver that does not re-derive the thing under test.**

### L64b ⛔⛔ A design memo's "measured-cheap, time it at execution" is an UNMEASURED cost claim — and this one was 17–25 % of a solve

The memo priced the operator-held table as "numpy assembly, measured-cheap; if
contested, time it at execution". Timed:

* `[M]` `StreamingOperator` is constructed **6** times per slab eigenvalue solve
  and **10** on the sphere — **independent of `nx`, `ng` and inner solver**; and
  `default_for` fires once per operator (`cached_property`), so a per-operator
  memo is one table build per operator.
* `[M]` today the count is **exactly 1** per solve on every config (the eager
  `SNSolver.__init__` build + the mesh-attr memo).
* `[M]` build cost / solve wall: cartesian `GL16` nx=200 **8.78 ms / 284.8 ms**
  ⟹ 6 builds = **16.8 %** (8 = 24.65 %); sphere `GL16` nx=200 8.81 / 1471.9;
  cylinder `fp(8,16)` nx=200 39.10 / 10 246.

⟹ **the gate is a COUNT, never a wall clock** (L24/L25): pin
`StreamingCoefficientCache.from_mesh_and_quad` calls per solve, and write the
ruled number into the message. A memo-lifetime regression is then one red away
instead of a silent 17 %.

### L64c ⛔ A "the reads re-plumb" done-when is DESIGNED-RED when the ruling puts a third of them out of scope — partition by ATTRIBUTE, not by count

The design memo says step 2 re-plumbs "the 43 (ii) + 17 (iii) + 5 (i) reads";
the ruling says the hub KEEPS the scheme *because* it induces the space and
supplies cross-consumers (DSA). `[M]` by attribute over `orpheus/`:
`scheme.spatial_basis_per_axis` **15** and `scheme.is_multi_moment` **6** are
SPACE/layout facts — 21 of the reads — and re-plumbing them through the operator
inverts the ruling. Meanwhile the per-cell kernel surface
(`residual_kernel_batch` 2, `source_emission` 2, `cell_average` 2,
`cell_kernel_batch` 1, `residual_kernel_batch_transpose` 1) is **9 reads**.

⟹ ship the partition as an **executable read-set gate**: wrap the hub's two
objects in a recording descriptor after the pose, run one sweep + one matvec,
and assert the recorded attribute set ⊆ a declared allowlist. That converts
`plan-authoring` §10's third shape (a tell whose denominator the design is
forbidden to touch) into a checkable predicate — and it is the ONLY possible
witness for the two reads that are base **staticmethods** (below).

### L64d ⭐ Resolve a class-level mutation through the MRO — and two of the surfaces are base STATICMETHODS, which makes their re-plumb value-inert BY CONSTRUCTION

`[M]` of the nine surfaces the walk consumes, **four do not live on the concrete
class**: `source_emission` and `cell_average` are `staticmethod`s on
`DiscretizationSchemeBase`; `advance_psi_half` is a function and
`c_out_per_ordinate` a `property` on `PoleAngularClosureBase`. A battery
patching only `DiamondDifference` / `MorelMontryAngularSweep` binds **5 of 9**
and reports a confident partial zero (my plugin's own installation assertion is
what caught it — `vv` #17).

⭐ And the design consequence, which is not about the battery: because those two
are base staticmethods, `mesh.scheme.source_emission` and
`op.spatial_closure.source_emission` resolve to the **same function object**.
Re-plumbing them **cannot change any value at any tolerance on any fixture** —
Mode 12 at the dispatch — so their only possible witness is the structural
read-set gate. A value gate credited with covering them is a false coverage
claim.

### L64e ⭐ The M1 superset denominator, and its geometry PARTITION (the frozen corpus, 27 tests)

`[M]` in-process, MRO-resolved, self-asserting plugin over
`tests/sn/regression` + `test_affine_carve_baseline.py`:

| arm | mutated ×1.05 | bound | wall | reds |
|---|---|---:|---:|---:|
| baseline | — | 0 | 48 s | 0 |
| `m1_scheme` | `residual_kernel_batch`, `source_emission`, `cell_average` | 3 | 122 s | **20** |
| `m1_closure` | `advance_psi_half`, `cell_contribution` | 2 | 60 s | **16** |

Partition: **10 both · 10 scheme-only · 6 closure-only · 1 never-red** (union
26 of 27). Scheme-only = every SLAB + 2-D row; closure-only = every curvilinear
MATVEC row. This is an independent corroboration of the activation census's
finding that **the two halves of the step have DISJOINT activating configs**
(`[M]` per-cell scheme dispatch: slab **80**, curvilinear **0**; closure
dispatch: slab **0**, curvilinear **3 192 – 14 496**).

⚠ The single survivor (`2d_1g_LS4_dd_15x15`) is my ARM's composition — it is
2-D wavefront and its surface is `cell_kernel_batch`, which the arm omits — not
a blind gate. Record the gap; do not bank 26/27 as a coverage number.
⛔ And compare **per arm**, never the union: a union comparison hides a
scheme-side regression behind closure-side reds.

### L64f ⭐ Ask the P4.9a activation question again — the answer can INVERT between phases of one campaign

P4.9a's §F1 found every frozen artifact blind (a congruence-class gate). The
reflex is to carry that forward. `[M]` for P4.9b it is the opposite:
`StreamingOperator.__init__` fires in **26 of 27** frozen artifacts, so the
corpus pins step 1 universally and **nothing new needs to be built** — while the
scan-cache re-pose is activated by only **15 of 27** (the matvec/walk baselines
and the 2-D rows build zero). Two different denominators inside one phase.

⟹ re-run the census per PHASE and per CLAIM; a previous phase's blindness
verdict is about a different line.

### L64g ⛔ A design memo's hazard prose is a claim — the "silent, plausible-wrong k" was LOUD on every geometry that matters

The ruling's no-guard position rests on an enumeration of what the expert seam
does not check. `[M]` attack 3 (`IdentityAngularClosure` on a curvilinear mesh)
CONSTRUCTS and then **raises at the first sweep**: `TypeError: … requires the
Morel-Montry closure` on the sphere, `IndexError: tuple index out of range` on
the cylinder (`fp(4,6)` and `fp(4,8)`), and is bit-identically inert on the slab
(`0.0000e+00` — `IdentityAngularClosure` IS the Cartesian default). The same
`isinstance(closure, MorelMontryAngularSweep)` residual that refuses a recording
PROXY refuses the wrong family.

⟹ the ruling survives; **the sentence must not be transcribed into the ctor
docstring** — a hazard note saying "silent" about something that raises teaches
the next reader the opposite and reads as licence to add the forbidden guard.
And the characterization test that freezes such a ruling asserts
**constructibility only** (one positive leg, no negative), with the reason in
its docstring so the missing negative is not read as an oversight (`vv` #11).

### L64h ⛔ A retirement's memo/slot inventory is a claim to COUNT — three memos, and the contract's only witness dies with the slots

`[M]` the design named `_geom_cache`; the tree carries **three** mesh-attr
memos of the same idiom (`_geom_cache`, `_coll_cache`, `_pole_mirror_cache`),
and `_coll_cache` is re-stamped by `SNSolver.rebind_cross_sections` — which is
the ONLY reason a depletion/thermal rebind does not serve a stale σ
(`_ensure_coll_cache` reads the memo and never validates σ).

⛔ `tests/sn/sweep/core/test_cache.py:295-340` is the **only** witness of the
two-stratum rebind contract and it asserts on `solver.geom_cache` /
`solver.coll_cache` — the slots being retired, so it DIES (L61h's flavour).
Its re-pose owes a THIRD leg that does not exist today ("the post-rebind answer
reflects the new σ"), i.e. **net-new teeth created by the retirement itself**.
Same shape, same phase: `[M]` `grep "StreamingCoefficientCache requires" tests/`
→ **0**, so converting that bare `assert` (inert under `-O`) to a `raise` is
new coverage, not a migration.

### L64i ⚠ Other measurements worth keeping

* `[M]` `StreamingOperator.__hash__` is **`None`** (plain `@dataclass`), so
  nothing in the tree can key a dict/set/`lru_cache` on it, and `__eq__` today
  is *mesh identity* (`SNMesh.__eq__` is `object.__eq__`). An AST sweep of every
  `Compare` in the 61 touching files finds **no operator `==` anywhere**, and
  `repr(op)` appears only inside failure MESSAGES. ⟹ widening a dataclass's
  field list is inert here — but that is a measurement, not a default.
* `[M]` **a second `DiamondDifference` instance exists in every solve**, born in
  the `@cache`d, type-keyed `_face_transmission_spectrum` probe
  (`transport/spatial/scheme.py:566`, reached from `loss_kernel_gauge:375`), and
  `cell_kernel_batch` dispatches on IT. Any "every consumer reaches ONE object"
  gate must carve it out by name.
* `[M]` step 2's §6b set is **79 sites, not ~28**: 22 `default_for` calls (1
  production) **plus 58 direct `LossRepresentation`-subclass ctors in 11 test
  files**, every one passing a single positional `mesh`. Derive the class-name
  set from the `LOSS_REPRESENTATIONS` registry at census time, never by hand.
* `[M]` the obvious `match=` fragment for a ctor-arity witness is already taken
  AND ambiguous: `test_streaming_collision_operator.py:262` reads
  `pytest.raises(TypeError, match="StreamingOperator")` against an `isinstance`
  refusal, and Python's own arity error contains that same substring. Match the
  argument NAMES.
* `[M]` scope costs at `10314dfa` (`-O -p no:randomly -q -m "not slow"`,
  serial): the 40 ctor-site files **663 p / 56.2 s**; `tests/sn/operators`
  **1240 p / 74.0 s**; `tests/sn/sweep` **911 p / 282.2 s**; `tests/transport`
  **566 p / 22.3 s**; `tests/sn/{solve,regression,architecture}`
  **327 p / 311.9 s**. `dead_references` baseline **0 dead / 52 checked**.

---

## L65 — CS5 "an axis can name the generator that made it" (plan delivered 2026-08-29)

Deliverable `scratch/cs5_verification_plan.md`. Opener
`scratch/cs5_ground_measure.md`. Campaign §5.5 of
`.claude/plans/space_and_kernel_binding_campaign.md`. All `[M]` below measured
in-process on 2026-08-29; the mutation battery ran as **pytest plugins
monkeypatching at `pytest_configure`** — no file was edited, so
`process-discipline`'s crash-safety clause is satisfied by construction
(nothing to restore, and the working tree carried uncommitted production
edits throughout, making `git checkout` forbidden anyway).

### L65a ⛔⛔ The landed surface had ZERO genuine catchers — and the ONE red was a harness artefact

Anchor set: `tests/numerics/{test_axis, test_space_of_axes, test_measure,
test_quadrature_directional, test_axis_marginal}.py` +
`tests/sn/mesh/test_angular_bulk_space.py` = **184 passed / 1.31 s**.

| mutation | reds |
|---|---:|
| `generator` **inside** `Axis._identity_key` | **1** — `test_of_axes_name_is_deterministic_across_processes`, whose SUBPROCESS leg runs an unmutated interpreter ⟹ it reds for ANY in-process digest change. A cross-process differential, **not** a provenance catcher (`vv-principles` #17). |
| `DiscreteMeasure.axis` mints `BasisKind.MODAL` | **0** |
| `Quadrature.axis` drops the `replace(..., generator=self)` upgrade | **0** |

⟹ **the transferable move**: when a phase's step-1 code is already in the tree,
run the battery against the EXISTING suite FIRST. The number that matters is
not "will my new gates redden" but "what does the tree already catch" — and
here the honest answer made three gates mandatory rather than confirmatory.
⚠ And read the single red's IDENTITY, not its count: a cross-process
determinism gate is a *universal* digest tripwire and will red for every
in-process mutation, so banking it as a catcher inflates coverage by one and
hides that the real count is zero.

### L65b ⭐⭐ An identity-key inclusion whose operand is un-hashable RAISES — it does not merely disagree

The plan's predicted mutation outcome was *"the derived space name changes"*.
`[M]` it does that **plus two louder things**:

```
CORRECT   a1 == a2 -> True    hash equal -> True   names equal -> True
MUTATED   a1 == a2 -> ValueError "truth value of an array ... is ambiguous"
          hash(a1) -> TypeError "unhashable type: 'Quadrature'"
          of_axes(a1).name  angular(4,)#a1259d874905e50e -> #a56a82a93fac074b
```

Root cause, both measured: `DiscreteMeasure` is `@dataclass(frozen=True,
eq=True)` over ndarrays (`m1 == m2` raises `ValueError`; `hash(m1)` raises
`TypeError`), and `Quadrature` is `@dataclass(frozen=False, eq=True)`, which
makes Python set `__hash__ = None`.

⟹ **the design ruling ("provenance is not identity") was justified on taste
and is true for a structural reason nobody had written down.** The gate should
pin the REASON — two `pytest.raises` legs on the generator types themselves —
so a future "tidy the field into the key" is refuted by a gate instead of
discovered by a traceback. Generalisation: *before excluding a field from an
identity key on doctrinal grounds, check whether the exclusion is also
MANDATORY; if it is, that is the stronger and more durable gate.*

### L65c ⭐⭐ When a re-point makes `new_path is old_path`, every value gate is a tautology — build a DECOY

Step 2 re-points `sn_mesh.quad.mu_x` → `space.axis("angular").generator.mu_x`,
and `[M]` `generator is sn_mesh.quad`. So the "re-pointed read agrees with the
quadrature" assertion is green before the re-point, after it, and under a
PARTIAL re-point. Only a **route** gate discriminates (`vv-principles` #26).

The instrument, `[M]` constructible: a **weight-preserving decoy**
— same `weights`, `np.roll(nodes, 1)`:

| property | reading |
|---|---|
| weights true vs decoy | **identical** |
| `of_axes(...).name`, `spaces ==` | **EQUAL** (invisible to identity) |
| `mu_x` `[-0.861,-0.340,0.340,0.861]` vs `[0.861,-0.861,-0.340,0.340]` | **DIFFERENT** |

Feed a consumer a space carrying the decoy: read-through ⟹ the answer MOVES;
reach-past ⟹ UNMOVED, and unmoved is the failure. Plus the anti-dud control
(true generator reproduces the baseline) and the instrument's own precondition
(the decoy must NOT move the space name, else the gate measures identity).

⚠ **Refuted decoy**: `-nodes[::-1]` on Gauss–Legendre is the **identity** (the
rule is symmetric), so the keystone read green vacuously. Always print the
decoy's discriminating array before trusting the gate.

⟹ this is P4.9b's keystone shape run in REVERSE — there, *swap the hub's object
and require the answer UNMOVED*; here, *swap the space's generator and require
the answer to MOVE*. Same family: the claim is about the READ PATH, so the
instrument must perturb the path, not the values.

### L65d ⛔⛔ A measure-generated axis is RANK-1 by construction — and d=1 hides it

`DiscreteMeasure.axis` must mint `shape=(self.n_points,)`; a measure is a flat
atom list. `[M]` against the shipped spatial axis:

```
d=1  minted (5,)   shipped (5,)    ==  True      (identical, digest unmoved)
d=2  minted (12,)  shipped (3, 4)  ==  False
     spatial(12,)#37129e801028f040   vs   spatial(3, 4)#1dcbbbc62beddd24
```

Space identity is `(name, shape)` until the S3 flip, so a d≥2 re-mint moves
`bulk_space`, `angular_bulk_space`, `full_field_space` and every operator keyed
on them. `[M]` d≥2 is reachable and exercised — **8 test files** call
`SNMesh.from_axes` / `MaterialMesh.from_axes`.

⚠ **The congruence here is RANK** (`vv-principles` #13): every axis-suite
fixture is 1-D, so the natural gate certifies a change that is wrong for d≥2.
⟹ **for any mint that consumes a flat collection, ask what the CONSUMER's rank
is before writing the gate.**

The tree took a third option: `isinstance(mesh, Mesh1D)` branches the mint,
1-D through the measure and rank-d keeping the literal (`generator=None`,
"the CS2 rank-d seam"). That is fine and creates a **P4.9a-shaped blindness** —
the generator path executes on 1-D carriers ONLY — so the gate must parametrize
over the **BRANCH**, with the d≥2 row asserting the OPPOSITE claim (`generator
is None`, digest unmoved). Two arms, two claims; a one-arm gate lets the other
drift unobserved.

### L65e ⭐ The intrinsic law of a provenance accessor is the SECTION law

An axis is a *forgetful map* of its generator (keeps weights, drops nodes). The
forgetting is recoverable iff the mint is a **section**:

> for every generator-ful axis `a`:  `a.generator.axis(a.label) == a`

`[M]` holds 4 of 4 shipped angular rules and at d=1 spatially; `[M]` it is
**exactly what fails at d≥2** — so writing it converts L65d from "someone might
notice" into "a red the step cannot land past". ⚠ State the law over
*generator-ful* axes only, or the three shipped generator-less sites read as
violations.

### L65f ⭐ §6c witnesses: look for the state that SHIPS before manufacturing one

The brief's §6c answer was *"a hand-built generator-less axis"*. `[M]` three
production sites are generator-less at landing and stay so: the homogeneous
counting point (`homogeneous/solver.py:148`), **every** `EnergyAxis` (all three
constructors mint none), and the MODAL `spatial_moment` axis
(`transport/spatial/scheme.py:1714`). ⟹ the refusal guards a live state, not a
hypothetical — a strictly stronger §6c answer, found by *reading the generator
off real objects* rather than by reading the design.
⚠ And record the witness's SHELF LIFE: when item 3 gives the MODAL axis a
`Basis` generator, one of the three retires.

### L65g ⚠ Two comparisons in this plan are wrapper tautologies — say which half is real

* **Angular (G3)**: `q.weights` **is** `q.measure.weights`, which is what
  `DiscreteMeasure.axis` reads. The gate pins THREADING (label, shape
  *spelling*, `kind`, that `weights=` is wired to weights, that `replace`
  preserves the canonicalized bytes) — never the weight VALUES.
* **Spatial (G6)**: `[M]` `mesh.volume_measure.weights` **is** `carrier.volumes`
  — the same array object.
* `[M]` the only independent literal-volume pin in the axis suites is
  `test_space_of_axes.py:393` (`[2.0]`, n=1). The non-uniform vector
  `[0.2, 0.3, 0.4, 0.7, 1.4]` appears at `test_angular_bulk_space.py:53` **as a
  comment**, while `:94` asserts against `sn.volumes` ⟹ the mesh→measure→axis
  chain has **no independent anchor at any rank**. One literal row closes it.

⟹ the honest G6 is the one that rebuilds the **pre-CS5 literal space in the
TEST** and compares digests. `[M]` `energy(2,)*spatial(5,)#f6838f4e6474b7c4` on
both sides post-landing — a real gate, where `carrier.name == carrier.name`
would have been nothing.
⚠ And `Mesh1D.volume_measure` is a plain `@property` ⟹ a FRESH `DiscreteMeasure`
per access, so `axis.generator is mesh.volume_measure` is a latent false red.
Assert type + content. (Same fresh-object-per-access trap `EnergyAxis`'s
docstring records for `Mixture.energy_grid`.)

### L65h ⚠ Other measurements worth keeping

* `[M]` `dataclasses.replace` round-trips an `Axis` **bit-identically** —
  weight bytes equal, read-only flag preserved, `eq`+`hash` preserved, the
  all-ones→`None` canonicalization idempotent, and `EnergyAxis`'s kw-only
  `edges` survives. So a `replace`-based provenance upgrade is safe on every
  subclass (Pattern 4∩2: the invariant re-fires rather than being restated).
* `[M]` `test_space_of_axes.py`'s `_reachable_arrays` walker (`:43-55`) descends
  only into `ndarray`/`tuple`/`list`/`Axis`, so a `Quadrature`-valued field is
  DROPPED and the no-densification gate is unaffected. **A near miss**: had it
  descended into arbitrary objects, the quadrature's `nodes`/`weights` would be
  swept into the reachable set and the gate would red for an unrelated reason.
  ⟹ after adding a field to a type, grep the tests for reflection walkers
  (`vars(`, `asdict`, `fields(`) — `[M]` exactly **one** in this suite.
* `[M]` `AngularMeasure` (`sn/angular/redistribution.py:220`) declares **6**
  members; `level_structure` is absent while `sn/mesh/reduced_operator.py:708`
  probes it as `getattr(x, "level_structure", None)` — a **string-form** read no
  symbol grep sees, and `vv-principles` #28's temporal twin (a defaulted
  `getattr` in a guard condition goes silently inert on rename). `[M]` the
  Protocol widening is §6b-clean: `grep -rln "AngularMeasure|angular_measure"
  tests/` → **0 files**, so there are no duck-typed surrogates to break.
* `[M]` the solve-time / mint-time split by enclosing function (AST): mint-time
  = `AngularClosure.__init__` 7, `angular_redistribution` 5,
  `SweepConstants.from_mesh_and_quad` 4, `angular_bulk_space` 2,
  `streaming_terms` 2 (all STAY per the P4.9b hub ruling); solve-time = **18 of
  19** reads in `sn/loss_representation/__init__.py` (`_run` 5,
  `_run_transpose` 4, `loss_action_transpose` 3, `_apply_walk` 2,
  `_sweep_scheduled` 2, `loss_action` 1, `_dag_legs` 1). One keystone row per
  re-pointed entry (`vv-principles` #17's per-call-site clause) — a decoy that
  moves `_run` says nothing about `_run_transpose`.
* `[M]` `eta` has **no solve-time consumer at all** — its only reads tree-wide
  are `augmented_mesh.py:1620` and `:1751`, both mint-time on `self.quad`. Its
  gate rows are CONTRACT rows, not catchers of a re-pointed read.
* `[M]` health at delivery: `pyright` **0/0/0** over the 8 touched modules;
  fresh-interpreter import of 6 packages rc=0 (§6d clean — the new
  `measure → axis` runtime edge closes no cycle); anchor set **215 passed /
  2.06 s** with steps 1–3 all landed.

---

## L66 — P4-remainder step 2 (the producer binds the angular axis; contract-change half, PRE-carve, plan delivered 2026-08-29)

**Deliverable** `scratch/p4rem_step2_verification.md`. Branch
`feature/p4rem-producer-binds-axis`, HEAD `1a1aa60f` (one docs-only commit past
the ground memo's `7d1129cc` ⟹ every `file:line` in
`scratch/p4rem_ground_measure.md` re-verified). Ruled design: (1) RSO gains
`angular_axis: Axis`, factories mint `quadrature.axis("angular")` internally;
(2) closure family ctor takes the axis third, `AngularRedistribution` sheds
`quadrature`; (3) `_weight_of` retires; (4) G5/G7 land per the CS5 specs.

### L66a — the re-point is a THREE-WAY `is`-identity, so the keystone is a route gate

`[M]` runtime probe, real `SNMesh`:
`sn.angular_bulk_space.axis("angular").generator is sn.quad` **True**;
`sn.reduced.angular.quadrature is sn.quad` **True**; hence the axis's generator
`is` the courier's quadrature. The ruled design mints the axis inside the
factory from the SAME `quadrature` parameter that feeds
`angular_redistribution`, so the identity SURVIVES the change ⟹ every value
gate is `X == X`, green before, after, and under a PARTIAL re-point. Same shape
as CS5's `L65c`, one tier down.

⭐ The sharpening CS5 did not have: **the courier's REMOVAL is what makes the
partial re-point unspellable.** Once `AngularRedistribution.quadrature` is
deleted there is no second route to reach past to, so "did every read move?"
stops being a test question and becomes structural — which promotes the
`dataclasses.fields` name-set row from nicety to load-bearing.

### L66b — ⛔⛔ the CS5-recommended decoy is REFUSED by a production admission guard

CS5 §3.7 prescribes a weight-preserving `np.roll`ed-node decoy. `[M]` it is
weight-preserving and space-identity-invisible as advertised — and
`angular_redistribution` REFUSES it on every curvilinear chart, because the
α-dome closes only if `Σ_m w_m µ_m = 0` and a roll breaks the (µ, w) pairing
(`gauss_legendre(4)` roll → `ValueError: the alpha dome does not close:
alpha[M+1/2] = 0.3654883720235732`). At the closure-mint tier it is refused
again (τ ∉ [0,1] on the sphere; ω-arc non-monotone on the cylinder).

⟹ **a decoy must survive the PRODUCTION admission guards of the arm the gate
lives on**, which is one level past CS5's "print the decoy's discriminating
array before trusting the gate". Measured catalogue, all at HEAD:

| decoy | axis `==` | \|µ\| moved | M-M mint | verdict |
|---|---|---|---|---|
| `nodes × 0.9` | True | 4/4 GL4, 8/8 GL8, 8/12 fp(4,6) | **ADMITTED**, τ + µ_x MOVED, dAw same | ⭐ the only decoy admissible at BOTH tiers |
| `level_indices` roll-1 | True | µ_x array unchanged; 8/12 packets | n/a | ⭐ isolates the cylinder index read from the cosine read |
| `np.roll(nodes,1)` | True | 2/4 GL4, 10/12 fp(4,6) | REFUSED | packet tier only; slab-admissible (2/4, no dome check) |
| `nodes[::-1]`, `-nodes` | True | **0/N on every GL rule** | REFUSED | ⛔ provable non-catchers — `abs_mu` takes the modulus and GL nodes are antisymmetric |
| `level_indices` **reversal** | True | **0/12** fp(4,6) | n/a | ⛔ the `L43e` palindrome, live |
| different N | False | n/a | ADMITTED | the ONLY discriminator for `IdentityAngularClosure` |

⚠⚠ **`gauss_legendre(2)` moves 0 of 2 under the roll decoy — and
`make_tiny_spherical_sn_mesh()` is N = 2.** The tree's own default tiny
spherical fixture is the blind one.

⛔ **`IdentityAngularClosure` is a structural non-catcher for any same-N
decoy**: `[M]` its only courier read is the ordinate COUNT
(`angular.quadrature.mu_x.size`), so a same-N decoy mints bit-identical neutral
constants. Its discriminator is a different-N axis.

### L66c — ⛔⛔ the §6b set was 8; a subclass-aware census finds 10, and #9 is the campaign's own keystone

The banked set (ground §G.4) is a class-NAME census. Two members it is
structurally blind to (the P4.9b lesson, one campaign later):

* **`_MutantMM(sn.reduced.angular, sn.reduced.redistribution_pairing)`**,
  `tests/sn/operators/test_operator_feeds_the_walk.py:161` — a SUBCLASS
  inheriting `__init__`. It is **the P4.9b keystone's mutant factory**, so
  missing it does not weaken a gate, it removes that gate from the run at
  COLLECTION.
* **`AngularClosureBase.create("morel_montry_angular_sweep", angular=…,
  pairing=…)`**, `tests/sn/sweep/curvilinear/test_angular_closure.py:129` — a
  keyword-forwarding registry call, which also makes the third parameter's
  NAME an API surface.

`[M]` also corrected: production prose sites are **6**, not 2 (`closure.py:258,
:263, :298, :384, :386` + `augmented_mesh.py:407`). Recovery method:
transitive base resolution over every `ClassDef` + `super().__init__` +
registry `create`. `[M]` calls inside a `parametrize` ARGUMENT LIST: **0**
tree-wide for this blast radius, so `L47d`'s collection-kill does not apply.

### L66d — ⛔⛔ a SECOND mint site nobody named: the d ≥ 2 branch has no producer to mint from

`augmented_mesh.py:409-427` has two arms. The `reduced is None` (d ≥ 2
Cartesian) arm builds its closure operands from `angular_redistribution(
self.quad, self.coord)` with **no `ReducedStreamingOperator` in existence**
(`[M]` `sn2d.reduced is None`, already asserted at `test_cache.py:961`) — so
the third ctor argument needs its OWN mint there. A Pattern-2 twin one label
typo apart, on the exact branch M1's 27-row corpus **cannot redden** (its only
2-D row is the structural survivor `m1_scheme` omits). ⟹ the coherence gate
must be parametrized over the BRANCH (CS5 `L65d`), and the two arms assert
different things.

### L66e — the silent failure mode is the WRONG LABEL, and no existing anchor sees it

Because the RSO holds the axis OBJECT, `quadrature.axis("ordinate")` produces
bit-identical values and reds nothing anywhere — until O-3 hands the axis off a
space. `[M]` `sn.quad.axis("angular") == sn.angular_bulk_space.axis("angular")`
is **True** (hashes equal) while `is` is **False** (`Quadrature.axis` is a
METHOD minting fresh) ⟹ the coherence gate must be `==`, never `is`.

Blind-anchor audit, per candidate: the escalatable `DriftWarning` stored-value
gates witness only the LOUD modes (dropped `generator=` ⟹ `AttributeError` per
packet; wrong-N ⟹ shape error); `test_tau_producer_equivalence` is blind (both
sides read one quadrature through the closure); `CollisionCache._build_count`
is blind (different cache); `test_snmesh_consumes_reduced` is blind (its reads
come off the untouched AR). ⟹ **the route rows are the only discriminators for
the re-point itself** — say exactly that.

### L66f — the obvious oracle migration DEMOTES ~10 committed comparisons

`_dAw_of` (`test_diamond.py:85`, 10 call sites) and
`_c_surrogate.mm_constants_for_ordinate` are reference oracles that read
`op.angular.quadrature`. Re-pointing them to `op.angular_axis.generator` makes
the oracle read through the very accessor production now uses, so a wrong-axis
mint moves both sides together (`coding-standards` rewire-demotion; `vv` #22's
shared-input axis). `[M]` free to avoid: `quad` is already in scope at every
call site — pass it explicitly. The migration changes NO value today
(`op.angular.quadrature is quad`); the entire gain is that only the explicit
form survives as a wrong-mint catcher.

### L66g — the generator UNION cannot answer the reads, so the refusal must NARROW

`[M]` `Axis.generator: DiscreteMeasure | Basis | Quadrature | None`, and
`DiscreteMeasure` has **none** of `mu_x`, `eta`, `mu_z`, `level_indices`,
`axis_cosines`, `N` (only `weights`, `nodes`). ⟹ a bare
`axis.generator.mu_x` is a pyright error at EVERY re-pointed read, and the G5
refusal accessor should return a narrowed `Quadrature` — which makes G5
load-bearing rather than decorative and forecloses the `# type: ignore` reflex
(`coding-elegance` #19). ⚠ `tests/test_pyright_ratchet.py` gates BOTH
directions (an INCREASE and a DECREASE both red), so any legitimate movement is
re-baselined in the same commit.

### L66h — price the field-vs-property choice at plan time; the COUNT gate is blind to it

`[M]` `Quadrature.axis("angular")` mint = **9.61 µs** (N=16);
`ReducedStreamingOperator.streaming_terms` calls per solve = **320**
(`solve_sn_fixed_source`, cyl `folded_product(4,8)`, nx=20, ng=2; 320 = nx × N
exactly ⟹ the intern is built once); solve wall **0.123 s** ⟹ a
fresh-minting `@property` would cost **2.49 %** of the solve AND make
`op.angular_axis is op.angular_axis` **False** (a latent false red for any
identity gate). `CollisionCache._build_count == 1` counts a DIFFERENT cache and
cannot see it. ⟹ owed row: the field-identity leg, the idiom already in the
tree at `test_snmesh_consumes_reduced.py:182` for `delta_A`.

### L66i — `_weight_of` retirement: 0 test surface, TWO prose falsehoods

`[M]` `tests/` **0** (symbol + string-form), `docs/` **0**, `orpheus/` **2** —
the `def` at `reduced_operator.py:439` and a **body comment at `:536`** that
repeats the docstring's false claim verbatim (the ground memo caught only the
docstring). Both assert the scan cache calls it; `[M]` the cache reads
`quad.weights` directly (`cache.py:341-344`). No promoted guard, so **no
net-new gate is owed** — predicted delta exactly 0.

⭐ And the doc blast radius is ONE line Sphinx cannot see:
`docs/theory/methods/sn/curvilinear_one_group.rst:490` cites
`:attr:`…AngularRedistribution.quadrature``, and `orpheus.sn.angular` is not
`automodule`'d ⟹ `-W` is silent at every severity. `[M]`
`mcp__nexus__dead_references` baseline **0 dead / 52 checked** — it is the only
instrument.

### L66j — measured scope costs (serial, `-O`, `-m "not slow"`, `-p no:randomly`)

`tests/sn/mesh` 166 / **2.2 s** · `tests/sn/primitives` 336 / **3.2 s** ·
the numerics axis trio 82 / **1.9 s** · `tests/transport/spatial` 80 /
**21.9 s** · `tests/sn/sweep/core` 474 / **60.4 s** · `tests/sn/operators`
1255 / **71.2 s** · `tests/sn/sweep/curvilinear` 308 / **209.9 s** ⟹ union
**2701 / ≈ 371 s**. Baseline reconciliation: `[M]` `pytest tests/ -m "not slow"
--collect-only` = **9988 / 10215, 227 deselected**, and 9896 + 22 + 70 = 9988,
so the campaign's headline baseline IS a `-m "not slow"` run.

### L66k — ⛔ G7 cannot be a separate step

The ruled order schedules the route gates as step 3, after the binding (step
2). K1–K3 are UNWRITABLE before step 2 (their subject is the third ctor
parameter step 2 creates), so a step-3 landing leaves an interval with no route
gate AND makes G7's §6c red-before reading permanently untakeable. Under
`plan-authoring` §6b/§6c they are one step. The pre-carve evidence that IS
available is the SIMULATION — substituting a decoy quadrature for the courier
on a true-dome AR, which is the exact pre-carve analogue of "true AR + decoy
axis"; the §5.2 catalogue is that record.

---

## L67 — CS4c binding ladder (pre-carve verification plan, 2026-08-30, HEAD `2f44ed4e`)

Plan: `scratch/cs4c_verification_plan.md`. Design record:
`.claude/plans/cs4c_binding_design.md` (§12 = its own second-pass review).
Every number below re-measured this session with `.venv/bin/python`
(`ast` censuses, in-process runtime probes, in-process monkeypatch batteries).
Nexus WAS available (used for `dead_references` + `query`).

### L67a — the reverse-composite adjoint law is a THEOREM, so it cannot gate the embedding

The record's ruled §6 route-(b) leg: *"compare `bind(K).H` against an
independently-ASSEMBLED adjoint built from daggered factors."* `[M]` on the
shipped frame algebra `bind(K) = R∘K∘M` (orthonormal Legendre, `G_V = w`,
`G_M = 1`, `L ∈ {1,2,3}`, `gauss_legendre(L+2)`):

| analysis-face embedding | Galerkin defect `‖M†−R‖/‖R‖` | `bind(K).H` vs `M†K†R†` |
|---|---|---|
| `w`-weighted (CORRECT) | 0.0 / 5.3e-17 / 0.0 | ≤ 2.24e-16 |
| **constant** (the wrong-embedding control) | 2.00e-1 / 4.37e-1 / 6.88e-1 | **≤ 1.02e-16** |
| **unweighted** | 8.00e-1 / 1.875 / 3.221 | **≤ 2.00e-16** |

`A† = G_V⁻¹(RKM)ᵀG_V = (G_V⁻¹MᵀG_M)(G_M⁻¹KᵀG_M)(G_M⁻¹RᵀG_V) = M†K†R†` for ANY
three maps and ANY nondegenerate metrics — the wrong embedding is applied to
BOTH sides and cancels identically. `vv` #24(d) in pure form. The
discriminator is the **Galerkin defect on the FACES**, which needs no random
operand at all.

### L67b — ⭐⭐ the MAXIMALLY non-tight control was the BLIND one

Seed-swept, 200 draws, **diagonal (zonal / Funk–Hecke) operands** — the shape a
`ScatteringKernel` binding actually produces:

| rule | `‖MR − I‖` | multiplicativity rel. (min / med / max) |
|---|---|---|
| `gauss_legendre(L+1)` TIGHT | ≤ 1.1e-15 | 0.0 / 2.0e-16 / **1.74e-15** |
| `equispaced_equal(L+1)` NON-TIGHT | 2.5e-1…5.9e-1 | **1.20e-2** / 1.4e-1 / 4.61e-1 |
| `equispaced_equal(2L+2)` | — | **6.35e-3** / 7.1e-2 / 1.83e-1 |
| ⛔ **`gauss_legendre(L)` too-few-nodes** | **1.000e+00** | 0.0 / 2.0e-16 / **5.9e-16** |

The rule that is *maximally* non-tight by the property's own norm gives a
**bit-clean** defect at every `L` and every one of 200 draws. Ranking controls
by "how badly does it violate the property" picks the blind one. The fix is
one battery arm — a **positive-control-of-the-control**: the rejected control
must be shown NOT to red, alongside the chosen one reddening.

Good news for the design: diagonal operands DO discriminate ⟹ no kernel
product API is needed; the gate builds `K₁K₂` from the ℓ-stacks. ⚠ convention
trap: `moments[l]` is `[g_from, g_to]`, so `bind(K₁)·bind(K₂)` is
`moments₂[ℓ] @ moments₁[ℓ]`.

`[M]` ℓ=0 blindness reconfirmed: tight 8.99e-17, non-tight **0.000e+00**.

### L67c — a pseudo-inverse round trip is `P_range(G)`, and the WHOLE corpus dodges it

`riesz_raise ∘ riesz_lower` on the SN composite `FullFieldSpace`. `[M]` all
four ledger fixtures (slab / sphere / cylinder / cart2d) read ≤ 4.44e-16 —
and every one carries **0 tangential (`|Ω·n| = 0`) trace slots**; `_cart2d`'s
own docstring says `level_symmetric` was chosen to *"avoid the `mu_y == 0`"*
case. `[M]` the same 2-D mesh under `Quadrature.product(4,4)`: **32/64
tangential**, trace round-trip **2.871e+00** (`product(4,8)`: 32/128,
**2.666e+00**). So `raise∘lower == id` is (a) blind on the whole corpus and
(b) a FALSE RED at 2.87 on a legal production quadrature. `plan-authoring`
§2's "the denominator held FIXED across every row", at corpus scale.

### L67d — ⭐⭐ splitting a PAIRED metric mutation per LEG went 9/20 → 20/20

Today's ledger M-10 drops both metric sandwiches at once. `[M]` in-process, 4
geometries × 5 leaves, clean ≤ 3.0e-15, floor 1e-3:

| mutation | rows RED / 20 | min RED |
|---|---|---|
| M-10 both (today) | **9** — C 0/4, B 0/4, S+F silent on cart2d | 5.56e-2 |
| **drop `riesz_lower` only** | **20** | 1.28e-1 |
| **drop `riesz_raise` only** | **20** | 4.45e-1 |

`C` and `B` commute with `G`, so `G⁻¹AᵀG = Aᵀ` and dropping BOTH is invisible
(the ledger's own `_METRIC_CONSTRAINED = ("L","S","F")`). Dropping ONE leaves
`G⁻¹Aᵀ` — not a similarity — so commutation buys nothing. **Un-welding a
composite into named legs is a COVERAGE opportunity, not just a re-spelling.**

⛔ And the false-red it creates: the config-blindness control
(`G = c·1 ⟹ G⁻¹AᵀG = Aᵀ`) MUST keep the PAIRED mutation. `[M]` on
`_flat_metric_slab` (`c = 0.57735026919`): paired ≤ 7.09e-16 (blind ✓),
either single leg **4.226e-01 = |1−c| exactly**, on every leaf. Correct
arithmetic, wrong gate.

### L67e — `V.dual()` carries the PRIMAL's metric ⟹ a generic Riesz leg gives `G²`

`[M]` `DualSpace.of(V)` threads `metric=primal.metric` deliberately
(`space.py:1144-1157`; the P7 comment records that dropping it made a
dense-metric dual read the Euclidean pairing, 4.5 vs 23.3). On
`w = [0.5, 2, 4]`, `x = [1,1,1]`: `raise_V(lower_V(x)) = [1,1,1]` ✓ but
`lower_{V*}(lower_V(x)) = [0.25, 4.0, 16.0] = G²x`. ⟹ the Riesz legs are a
**two-verb pair on the PRIMAL** (which is what `metric.py:84`'s reserved
formula already spells — both reads are `A.domain.` / `A.codomain.`), and the
natural double-Riesz involution gate is a false red.

### L67f — an AST call census is the wrong FILTER for a physics constant

XD-2's ruled denominator was "~12 production multiplicity-2 literals".
`[M]` AST over `orpheus/`: `Constant(2|2.0)` in `BinOp(Mult)` or
`AugAssign(Mult)` whose enclosing function mentions `n2n|sig2|Sig2|sig_2n|_2n`
→ **15 rows = 14 genuine + 1 FP**. Two genuine sites were missing from every
prior list (`material_xs_field.py:809` `apply_n2n`, `:862`
`apply_n2n_transpose`), and **two of the fourteen evade a `2.0 *` regex**:
`moc/core.py:316` is the INTEGER `2 *`, and `mc/solver.py:447` is
`w *= 2.0` (an `AugAssign` with no `2.0 *` prefix and no n2n token on the
line). The one FP is `mc/solver.py:411` `2.0 * np.pi` — a function-SCOPE token
match. ⟹ a physics-constant census needs (a) AST, (b) both the `BinOp` and
`AugAssign` shapes, (c) a NAMED exclusion set, (d) a positive control on a
synthetic source string.

### L67g — the acceptance instrument was a 4-of-8 SAMPLE, and its reader is shape-blind

`[M]` AST census of `domain`/`codomain` property return annotations: **8
classes carry an Optional-shaped annotation** in two spellings
(`Optional['FunctionSpace']` and `'FunctionSpace | None'`). The ledger's
`_LEAVES = (L,C,S,F,B)` sees **4**; `IsotropicScattering`, `IsotropicN2N`,
`RadialCharacteristicBoundaryOperator` and `SNMaskedBoundaryOperator` flip
un-gated — and the first two are exactly the pair the campaign rebinds.

And the reader itself: `[M]` `_domain_annotation` returns
`("<not found>","<not found>")` — hence the strict xfail **PASSES** —
identically for a mandatory dataclass field, an Optional one, a
`field(default=None)` one, a kw-only one, and **a deleted attribute**. Five
shapes, one output. Its `if prop is None: continue` is a VALUE check
masquerading as a presence check (a `= None` default puts `None` in
`vars(cls)`). ⚠ It bites only if the flip converts properties to FIELDS —
`[M]` all 11 operator classes expose properties today, so the shipped reader
works at HEAD and the extension's necessity is conditional on the base's
ruled shape.

### L67h — `condense`'s asymmetric morphism pair, and its activation precondition

`[M]` on the shipped `tests/data/test_mixture_condense.py` 4g→2g fixture
(`_PHI = [1,4,2,0.5]`), against `dyad(FissionKernel.from_mixture(condensed))`:

| morphism pair | max-rel |
|---|---|
| (χ marginalize, νΣf average) — RULED | **2.876e-17** ✓ |
| (χ average, νΣf average) | **6.421e-01** |
| (χ marginalize, νΣf marginalize) | **1.685e+00** |
| (χ average, νΣf marginalize) | **7.087e-02** |

All three controls O(1)-red. ⚠ **Activation precondition, assertable:** every
coarse group must hold ≥ 2 fine groups *and* φ must vary within one — with one
fine group per coarse, `average ≡ marginalize` and two of three controls go
silent. Path correction: the ruled pair is at
`orpheus/data/macro_xs/mixture.py:437-450`, NOT `orpheus/transport/mixture.py`
(which does not exist); and `condense` has a SECOND branch
(`adjoint_spectrum=`) obeying a different law — the gate must say which it pins.

### L67i — a class deletion has MORE collection-killers than the audit names

`[M]` AST census of CODE-level `_AdjointOperator` references: **5 files** (4
test + the producer). TWO of them die at COLLECTION, not at run time:
`test_monomorphic_leaves.py:608` (module-scope attribute read) **and**
`test_operator_capability_predicates.py:280` (inside the module-level
`_CONTRACT_ROWS` list consumed by `@parametrize`). The design record named
one. `vv` #17's third pipeline failure: pytest reports `FAILED = 0`, `rc=2`.

And the retirement retires **four** surfaces, not one: `apply`, `inverse()`
(the #280 swap law), `is_invertible`, and `apply_transpose`'s
`NotImplementedError` — `[M]` the last has **0 witnesses tree-wide** (grep of
its message fragment over `tests/` = 0).

### L67j — `plan-authoring` §6d run on a permitted edge: legal ≠ free

`[M]` `FORBIDDEN_EDGES['cp'|'moc'|'mc'] = L3_PACKAGES − {self}` and
`transport` is L2 ⟹ the XD-2 re-point's three new edges are **PERMITTED**. But
`[M]` all three are **0 today**, and the marginal import cost is **254 ms**
(cumulative `orpheus.transport`, entirely its eager `__init__`; the module
itself is 0.8 ms). MC currently imports only `data` + `geometry` — the leanest
package in the tree. A "mechanical ~8-line re-point" that adds a package
dependency to three solver families is larger than its line count
(`lessons` L57).

### L67k — measured battery scopes (all cheap)

| scope | rows | wall |
|---|---|---|
| the 4 `_AdjointOperator` files + the ledger | 208 P + 14 xf | **0.71 s** |
| `tests/transport/` + axis-generator + the 3 S suites | 712 P + 1 sk | **23.6 s** |
| `tests/homogeneous/` + `tests/diffusion/` + mixture-condense | 204 P | **3.1 s** |

`[M]` `--runxfail` over the ledger: all **14** strict-xfail rows red for their
own documented reason (the `vv` Mode-8 FOURTH-class audit passes). ⚠ but the
`_R2_XFAIL` REASON STRING is present-tense-false for S: `[M]` an anonymous
`ScatteringOperator.H` now RAISES `MissingAdjoint` (the S4 amendment,
`operator.py:1320-1330`) rather than degrading; only `C` still degrades, via
`is_metric_free_adjoint = True`. The test BODY is honest (it reports the
`MissingAdjoint` as evidence text, never as the verdict); the marker is not.

---

## L68 — 2.1-W, the σ-even quotient sub-basis (gate SHIPPED 2026-08-31, HEAD `fc71a84e`)

**Task.** #429 tracker 2.1-W: make falsifiable the choice
`Quadrature._harmonic_basis` (`orpheus/numerics/quadrature/directional.py:505-530`)
makes — a FOLDED rule binds `MirrorEvenSphericalHarmonicBasis` (the σ-even
sub-basis of the quotient), an unfolded one the plain `SphericalHarmonicBasis`.
The defining mutation is the silent rebind `return SphericalHarmonicBasis(L=L)`.

**Shipped.** `tests/sn/eigenvalue/test_keff_curvilinear.py`,
`TestFoldedCylinderP1BindsTheQuotientBasis` (2 tests / 3 rows, `[M]` 18.16 s,
3 passed at HEAD; pyright delta **0** — 21 errors at HEAD, 21 after).

### L68a ⛔⛔ The plan's founding measurement was a pytest-NEVER-RAN artifact — and it is worse than the collection-ERROR class

The plan carried `[M] "deleting MirrorEvenSphericalHarmonicBasis outright reds
0 of 1913"` and built 2.1-W's whole premise on it ("zero committed witnesses").

`[M]` **REFUTED, two ways.**

1. **The IN-CLASS mutation reddens two committed gates already.** Rebinding to
   the parent basis via a `pytest_configure` monkeypatch:
   `tests/sn/primitives/test_quadrature_fold.py` goes **14 passed → 2 failed,
   12 passed**, the two being
   `TestFoldedHarmonics::test_flat_moments_are_the_isotropic_moment_alone` and
   `::test_the_folded_frame_analysis_is_isotropic_on_a_flat_flux`. Census:
   `calls=2 folded=2 mutated_at_L={2: 1, 1: 1}` — the whole file makes exactly
   two harmonic-basis calls and both are catchers.
2. **The DELETION mutation cannot produce a red at all.**
   `directional.py:83-87` imports the class at MODULE SCOPE, and
   `tests/sn/primitives/conftest.py:7` reaches it transitively:
   `conftest → tests.sn._test_helpers:30 → orpheus.transport:77 →
   transport.fields:49 → cross_section_field.py:68 → orpheus.numerics:37 →
   numerics.quadrature:71 → .directional:83 → ImportError`. Reproduced with a
   `sitecustomize` meta-path hook that strips the attribute before any import:
   scoping to `tests/sn/primitives/` gives **rc=4, 0 collected, 0 `^FAILED`
   lines AND 0 `^ERROR` lines** — pytest aborts on the conftest.

⟹ **the rule, and it is one notch past `vv` Mode-8's third pipeline class.**
That class says a collection kill is reported as `ERROR` and misread as 0 by a
`^FAILED` scanner. Here the CONFTEST dies, so pytest never reaches collection:
*both* scanners read zero, and `--continue-on-collection-errors` does not help.
**Before believing any "deleting X reds N", check whether X is imported at
module scope anywhere on a conftest's import chain** — one `grep -rn "import
X" orpheus/` answers it. If it is, the deletion measures the import graph, not
the test suite, and the honest instrument is the IN-CLASS mutation (`vv` #18),
which here is a legal `SphericalHarmonicBasis` and breaks no structural law.

⭐ The transferable tell: a deletion mutation on a symbol with a module-scope
import is **never** in-class, so `vv` #18 already forbids it — the two rules
meet.

### L68b ⛔ A per-instance memo masks the mutation, and the masked reading is a plausible bit-identical GREEN

`Quadrature.angular_frame` memoises into `self._angular_frames[L]`
(`directional.py:614-627`); `_harmonic_basis` itself is uncached.

`[M]` same fixture, three orderings:
* rule built AFTER the mutation installs → `keff = 0.4159228684117852`
* rule built before, cache warmed by an unmutated solve → **`0.9726641733732218`
  — bit-identical to honest**, `array_equal=True`, `cached frame Ls = [0, 1]`
* rule built before, cache NOT warmed → `0.4159228684117852`

The masked reading is exactly the shape of a "this gate has no teeth" verdict.
⟹ **install the mutation at `pytest_configure` (before any object is built),
and give every arm plugin a BITE CHECK**: build the honest and mutant basis on
the canonical rule and `raise` unless `max|ΔY| > 0` at `L = 1`. Mine printed
`ARM …: BIT (max|dY|@L=1 = 8.688461e-01) calls=8 folded=8` on every arm, so no
green was ever ambiguous. L64a trap (c), now with a measured instance.

### L68c ⭐ A GREEN arm can be a GEOMETRY THEOREM — a third category beside "insufficient mutation" and "blind gate"

Battery over the shipped gate, all six arms bite-checked, `python -O`:

| arm | keystone 4x8 / 8x4 (`\|k−k_inf\|`) | companion (Δ = k_P0 − k_P1) | pytest |
|---|---|---|---|
| M0 control | 1.4699e-11 / 1.5106e-11 | 2.4873e-02 | 3 passed / 18.16 s |
| **M1 rebind → `SphericalHarmonicBasis(L)`** | 4.3231e-02 / 4.9992e-02 | 5.8161e-01 | **3 failed** |
| M3 `mirror_axis` 1→0 | 4.3231e-02 / 4.9992e-02 | 6.0566e-01 | **3 failed** |
| M4 `mirror_axis` 1→2 | 4.3231e-02 / 4.9992e-02 | 5.8161e-01 | **3 failed** |
| M5 over-mask EVEN slot [1,1] (μ_x) | 1.4788e-11 / 1.4055e-11 | **0.0000e+00** | 1 failed, 2 passed |
| M6 over-mask EVEN slot [1,0] (μ_z) | 1.4699e-11 / 1.5106e-11 | 2.4873e-02 | **3 passed** |

M6 bit (`max|dY|@L=1 = 8.611363e-01`) and moved **nothing**. `[M]` the reason
is a correlation probe against the direction cosines: slot `[1,0]` has
`corr(Y[:,1,0], μ_z) = +1.000`, `[1,1]` has `corr(·, μ_x) = +1.000`, `[1,2]`
has `std = 0.0` (the masked σ-odd ξ carrier). A 1-D cylinder is symmetric under
`μ_z → −μ_z`, so the axial current is identically zero and that basis function
is annihilated by the CHART, at any refinement, on any 1-D fixture.

⟹ **read a null arm through three hypotheses, not two**: (i) the mutation was
insufficient (L49c's Pattern-2 twin), (ii) the gate is blind, (iii) **the DOF
the mutation broke is annihilated by the geometry**. (iii) is a declared
blindness that belongs in the gate's docstring with its witness-location
("only a 2-D/3-D fixture could carry one"), not a defect. The discriminator is
cheap and design-time: **correlate each basis slot against the direction
cosines and ask which of them the chart can excite.**

⭐ And M5 is why the companion row exists at all: it separates the two rows
exactly (keystone green at 1.4788e-11, companion red), so the non-flat-flux row
is justified by a MEASUREMENT rather than by §0.6's general warning.

### L68d ⭐ REUSABLE REFERENCE — the infinite medium is a Pℓ-ORDER-INVARIANT closed form

For any claim about the ANGULAR BASIS / moment machinery, the strongest
structurally-independent anchor available in ORPHEUS:

> In an infinite homogeneous medium the solution is spatially flat and
> angularly isotropic ⟹ `φ_ℓ ≡ 0` for every `ℓ ≥ 1` ⟹ the anisotropic
> scattering source is inert ⟹ `k = k_inf` **at every Pℓ truncation order**.

`derivations.get("sn_slab_2eg_1rg").k_inf = 1.8750000000000009` is a 2-group
transfer-matrix eigenvalue with **no solver, no quadrature and no basis** in its
chain — the closed-form pillar, the only one that may carry an eigenvalue claim.
`[M]` honest `|k(P1) − k_inf| = 1.4699e-11` against a `1e-6` gate: nine orders
of headroom, and the mutation lands at `4.3e-2`.

⚠ **Two activation obligations, both mandatory:**
* `SigS[1] ≠ 0`, asserted IN the test — a zero ℓ=1 cross-section multiplies the
  contaminated moment by zero and the row is vacuous. (`[M]` mixture A 2g:
  `SigS[1]` max `0.045`.)
* the claim must be posed at `scattering_order ≥ 1`. `[M]` at `L = 0` the
  σ-even table is **bit-identical** to the parent's (`max|ΔY| = 0.000000e+00`);
  it first diverges at `L = 1` (`8.688461e-01`, flat-flux moment `+6.486547`).
  Every folded eigenvalue row in the module ran at the default `P0` — which is
  precisely why the binding had no solve-path witness.

### L68e ⛔ The folded-vs-UNFOLDED equivalence gate is UNWRITABLE on a cylinder

The obvious A-vs-B design (a quotient must reproduce its parent's eigenvalue)
dies at the mesh: `[M]` `Quadrature.product(4, 8)` on a cylindrical `SNMesh`
raises *"A cylindrical SNMesh admits only a quadrature whose every mu-level is
CARRYING (the R12a march-start predicate): level 0 is non-carrying"*. The Q5.6.3
admission flip means the unfolded parent cannot be posed on the chart its child
is designed for. Check the ADMISSION guard before designing any fold/unfold
invariance gate.

### L68f ⚠ zsh does not word-split an unquoted parameter expansion

`SCOPE="a b c"; pytest $SCOPE` passes **one** argument in zsh (unlike bash), so
the run silently selects nothing and the summary greps return empty — which
reads as "the battery found nothing" rather than "the battery ran nothing".
`[M]` cost one confusing cycle where a control run printed no summary at all.
Use `${=SCOPE}` or an array, and always print the collected count.

### L68g Population filter — two predicates, positive-controlled, and the second one was the one I nearly missed

`_harmonic_basis` returns the fold basis only when `folded_by is not None`, set
only by `Quadrature.folded_product` / `.quotient`. Filter 1 (direct
construction) gives **74** test files. But `[M]` production ALSO mints folded
rules: `orpheus/derivations/continuous/mms/sn.py:2104` and `:3873`
(`build_cylindrical_mms_case`, `build_cylindrical_anisotropic_mms_case` — "the
case builders default to folded_product"), so filter 2 adds **6 indirect-only**
files, one of them outside `tests/sn` and `tests/numerics` entirely
(`tests/derivations/test_sn_mms_anisotropic_symbolic.py`). Union **80**.
Both filters carry an in-script positive control asserting a known member.
⟹ **a "who can reach this seam" census needs the PRODUCTION defaults, not just
the test-side constructor grep** — the plan-authoring §2 FILTER clause, at the
call-graph tier.

---

## L69 — the FUSED step (Landing A + B) for #429 / ERR-080 (verification plan delivered 2026-09-02, HEAD `17e0eb0e`)

Two landings ruled by the user: **A** — "the angular moment space is READ off the frame,
never minted from `L`" (bit-identical pre-step, 7 `SphericalHarmonicSpace.from_L(L)`
producers re-pointed + the `HarmonicFrame` SH narrowing widened); **B** — the ONE fused
commit `0.1b + 0.6 + 2.2 + 3.4 + 3.4b` that mints a true `LegendreBasis` on
`S²/SO(2)_a`, arms G0 at the frame, deletes the 1-D `(μ,0,0)` forgery, and makes
`_evaluate_real_sh` refuse off-sphere points. Plan: `scratch/_fused_verification_plan.md`.
Probes: `scratch/_fused_va_{dense,bits,lpmv,einsum,metricfork,hfork2,isotypic,misc,operands,census,theorem,testcount}.py`.

### (a) ⛔⛔ NO single library routine reproduces a hand-BRANCHED production table

The brief asserted `Y[:, ℓ, ℓ] = lpmv(0, ℓ, μ_x) = P_ℓ(μ_x)` **bit-exactly**, and the
whole "slab L ≤ 1 flux is bit-identical across the repair" exit predicate rests on it.
`[M]` `_evaluate_real_sh` hardcodes `Y[:,0,0] = 1.0` and `Y[:,1,1] = mu_x` (the INPUT
array) and only calls `lpmv` from ℓ ≥ 2. So:

```
l=0   == lpmv ✓   == eval_legendre ✓
l=1   == lpmv ✗ (8.327e-17 GL8, 1.110e-16 GL16/LS4/lebedev)   == eval_legendre ✓
l>=2  == lpmv ✓   == eval_legendre ✗ (3.3e-16 … 8.0e-16)
```

The consequence is not cosmetic. Simulating the replacement contraction on the REAL slab
tables (`ng=2`, `nx=4`), the **matched** spelling (`1.0` / `μ` / `lpmv`) gives
`array_equal` on `analyze`, `analyze_transpose` AND `reconstruct` at L = 0, 1, 2 on GL8
and GL16 — **6 of 6 rows, `max|Δ| = 0.00e+00`** — while the pure-`lpmv` spelling breaks
every one at `4.44e-16` (GL8) / `8.88e-16` (GL16). ⭐ And the same matched spelling makes
the `Descent` isomorphism `array_equal` (`max|Δ| = 0.000e+00`) on **7 of 7** full-sphere
and folded rules at L = 4, so the gate can be stated at the BIT tier rather than "to
machine precision".

⟹ **when a new type must reproduce an existing table bit-exactly, diff the existing
producer's BRANCHES, not its name.** A hardcoded low-order block is a second spelling of
the same function and it is invisible to a spot-check at one ℓ.

### (b) ⛔⛔ A space and its metric-dressed twin are `==`-EQUAL and metrically DIFFERENT

Landing A's charter offered `frame.basis.space` **/** `frame.basis_space` as one thing.
`[M]` over 12 (rule, L) rows: `basis.space` is metric-IDENTICAL to
`SphericalHarmonicSpace.from_L(L)` (12/12, `array_equal` on the weights) and
`basis_space` is metric-DIFFERENT (12/12) — `[12.566…]` vs `[0.5]` on the slab, vs
`[0.0796]` on a sphere rule, and a `DenseMetric` (no weights at all) on the slab at L=2.
The observable: `apply_metric` moves **96.0 %–161.3 % relative**.

`FunctionSpace.__eq__` is `(name, shape)`, so BOTH compare equal, no `==`-based gate can
see the fork, and the bit-identity charter would have been met on paper while every
moment operator's `.H` moved. This is `plan-authoring` §8 in its exact G6.3-step-8.0
shape (87 % on a boundary law) — a *metric* is an input a consumer branches on.

⟹ **when a charter names two spellings of "the space", measure `apply_metric` on both
before writing one line.** `==` is the wrong instrument by construction here.

### (c) ⛔ A vv#13 negative control can be BLIND below a threshold parameter value

The SO(2) isotypic probe's whole point is that right angles generate `C₄` and falsely
admit slots. `[M]` about the SH polar axis (x), incommensurate and right angles agree
EXACTLY at L = 1, 2, 3 — **zero** false positives — and the first divergence is at L = 4
(`(4,0)`, `(4,8)`, i.e. m = ±4), growing at L = 5 (`+ (5,1)`, `(5,9)`). So the mutation
arm "swap incommensurate for right angles" reddens **only** at L ≥ 4, and a probe gate
parametrized over L ∈ {1,2,3} ships with an unfalsifiable control.

⭐ Companion measurement, needed to read the number: the honest answer at L = 4 is
**25 of 45 table slots**, of which 20 are `|m| > ℓ` padding (invariant under everything)
⟹ **5 real of 25**, which is what §III.8 recorded. A gate counting table slots reports 25
and reads as "all of them".

⟹ **a vv#13 control needs its own activation threshold measured**, and a probe over a
padded layout needs its denominator stated in REAL slots.

### (d) ⭐⭐ A random-draw separation statistic has an exact, draw-free replacement

`test_frame.py`'s D3/D4 pin "no diagonal metric can satisfy Parseval on a dense frame"
with a floor (`1.5`) on a randomly drawn coefficient vector. `[M]` 400 seeds on the very
frame it pins: the ratio ranges **0.2327 … 1.9975** (median 1.827) — the floor pins a
SEED. The ratio is `(Gc)ᵀD(Gc)/(cᵀGc)` with `D = diag(1/G_kk)`, i.e. a generalized
Rayleigh quotient, so its exact range is the generalized eigenvalue range of `(G D G, G)`
on `range(G)` — closed form, one `eigvalsh`, no draw. Measured that way today's flagship
reads `[0.065, 2.000]`.

⟹ **before pinning a floor on a random probe, ask whether the statistic is a Rayleigh
quotient.** If it is, the range is exact and the gate stops being seed-fragile.

### (e) ⭐⭐ The Gauss–Legendre DEAD-SLOT theorem (a project config fact)

`[M]` 12 of 12 rows (`n ∈ {2,4,8,16}` × `L ∈ {n−1, n, n+1}`): a `GL_n` rule's Legendre
Gram is **DIAGONAL and exact** (`max|diag − 2/(2ℓ+1)| ≤ 4.7e-16`, offdiag ≤ `3.7e-15`)
for `L ≤ n−1`, and acquires a **structurally dead slot exactly at ℓ = n** — because the
`GL_n` nodes ARE `P_n`'s roots. Hence **no 1-D Gauss–Legendre frame is dense AND
full-rank.** Consequences for gate design: (i) the slab GL8/L=2 frame's Gram becomes
DIAGONAL under a Legendre basis (offdiag `1.418e-16`), so the campaign-1 flagship DENSE
witness dies; (ii) `solve_sn(gauss_legendre(2), scattering_order ≥ 2)` reaches a
near-singular `DenseMetric` (`cond` up to `1.9e32`) that the Penrose guard currently
ACCEPTS; (iii) the full-rank dense Legendre witness must come from a **non-Gauss** 1-D
measure — `[M]` equispaced `n = 8, L = 3`: offdiag `6.107e-01`, 0 dead slots, cond `21.0`.

⭐ And the elegant coincidence: that equispaced measure IS
`tests/transport/frames/test_binding_tightness.py::_equispaced_frame`, the one existing
fixture G0 would refuse (it declares `support = COSINE_INTERVAL`). Re-declaring its
support as `SPHERE.quotient(SO2("x"))` makes it pass G0 **and** become the post-carve
dense flagship — one edit, two problems.

⭐ The B-unaffected replacement that needs no new fixture at all:
`folded_product(2,4).angular_frame(2)` reads range `[1.000, 2.707]` and `(…,3)` reads
`[1.000, 3.707]` — **1.7×/2.7× today's flagship separation**, and never below 1 (today's
goes to 0.065).

### (f) ⚠ A DERIVED `invariance_group` is a LOWER bound, so a lattice gate can refuse a CORRECT pairing

`Basis.invariance_group` is derived `@final` as `domain.by`. For `{P_ℓ(Ω·ê_x)}` that is
`SO2('x')` — but `[R]` the true stabiliser is `O(2)_x`: `σ_y` does not move `μ_x`, so the
functions ARE σ_y-invariant. `[M]` `SO2('x').contains(Mirror('y')) = False`, so a lattice
G0 refuses Legendre-on-a-σ_y-folded rule, which is mathematically admissible. The ABC's
own docstring predicts this ("the reading is a LOWER BOUND… the remedy is to declare the
finer domain") — and `[M]` there is no axis-parameterised `O2` to declare
(`SubgroupOfO3` ships `Trivial/Dinfh/OctahedralOh/IcosahedralIh/SO3/O3` + `Cn/Dnh/Mirror/SO2`;
`Dinfh` is parameter-free). Inert today (the dispatch never selects that pairing) and a
§6c landmine the day a cylindrical `P_L` expansion is wanted.

### (g) other measured hazards worth carrying

* **A value gate cannot discriminate `axis_cosines(0)` from `mean_axis_cosine(0)`** —
  `[M]` bit-identical on 5 of 5 GL rules. The DSA μ-read's principled choice has only the
  REFUSAL leg as a witness (`axis_cosines(1)` raises; `mean_axis_cosine(1)` returns zeros).
* **G0 lands with ZERO shipped production refusals** — `[M]` 7 of 7 pairings admitted
  today, and post-carve the dispatch selects the admitted basis for every shipped support.
  Its one tree witness (`_equispaced_frame`) hits THREE guards in sequence, and `[M]`
  `dom.realization == COSINE_INTERVAL` is **True** while `dom == COSINE_INTERVAL` is
  False, so an equality-G0 and a realization-comparing G0 give OPPOSITE verdicts on it.
* **A pin can be two-thirds honest.** `test_frame.py:906` pins `[0.4, 0.8, 0.8]` as "the
  fabricated ℓ=2 Gram diagonal"; `[M]` the row is `[0, 0, 0.4, 0.8, 0.8]` and `0.4 = 2/5`
  is the CORRECT m=0 entry — only the two `0.8`s are fabricated. A re-pin that deletes all
  three loses a real datum.
* **The carrier's two-axis head fails SILENTLY on a rank-1 head.** `scalar_flux`,
  `isotropic_part`, `anisotropic_part`, `moment(ℓ)` and `truncate` all index
  `values[0, 0]`; on a `(L+1, ng, nx)` array that is group 0's spatial slice, no exception.
  And the L = 0 chain (`fission.py:305`, `n2n.py:200`) puts the **isotropic** solve inside
  the blast radius, not only `scattering_order ≥ 1`.
* **Exit predicate 3's "two coarse-rule rows" is up to FOUR.** `[M]` post-carve the
  isotropic-flux ℓ≥1 moment is ≤ `8.8e-17` on GL4/8/16/32 through L=5 but reads
  `7.778e-01` on `gauss_legendre(2)` at L ≥ 4; and if the Legendre-on-a-full-sphere
  pairing is exercised, `product(4,4)` L=4 reads `3.665e+00` and `folded_product(2,4)`
  L=4 reads `4.887e+00`. `[R]` the mechanism for the last two is the tree's TWO POLES
  (SH about x, the product family about z), not a defect.

---

## L70 — #429 tracker 1.9 / #432 `SubgroupOfO3.O2(axis)` (gates SHIPPED 2026-09-02, HEAD `e590623f` + uncommitted)

**The change.** An orbit space is named by the LARGEST group with its orbits. The
constant-μ circles on `S²` are the orbits of `SO(2)_a` **and** of its stabiliser
`O(2)_a = {g : g ê_a = ê_a}` (a vertical mirror carries each circle onto itself), so
`R[x]^{SO2_a} = R[x]^{O2_a}` and the catalogue entry `S^2/O2_a` records the bigger
group. `SPHERE.quotient(SO2(a))` is REFUSED with that theorem. The consequence:
`LegendreBasis.invariance_group` derives the FULL group, and `GalerkinFrame`'s G0
admits the Legendre basis on a σ_b-fold for `b ≠ a` (previously over-refused —
`L69f`).

**Deliverable.** 43 new tests across 5 files (`tests/numerics/test_symmetry.py` +23,
`test_manifold.py` +8, `test_frame.py` +5, `test_registry.py` +5,
`test_legendre_basis.py` +2), a 17-arm in-process battery (`scratch/_p19_mut/`), and
one present-tense-false docstring repaired.

### L70a ⛔ A frozen dataclass's generated `__hash__` hashes the FIELD TUPLE, not the class

`[M]` `hash(SubgroupOfO3.SO2('x')) == hash(SubgroupOfO3.O2('x')) ==
hash(SubgroupOfO3.Mirror('x'))` — all three `_tag`s are frozen dataclasses whose only
field is `axis`, and `dataclasses`' generated `__hash__` is `hash((self.axis,))`. Same
for `hash(Cn(2)) == hash(Dnh(2))`. Equality still discriminates (the generated
`__eq__` opens with `other.__class__ is self.__class__`), so dicts and sets are
CORRECT — `len({Mirror('x'), SO2('x'), O2('x')}) == 3`. It is a bucket collision, not
a defect.

⟹ **`hash(a) != hash(b)` is NOT a legal "these are different groups" leg.** It is the
leg that comes to mind beside `a != b`, it reds on correct code, and it reads in
review as extra rigour. I wrote it into three parametrized rows and it failed on all
three. The property that matters — and the one to assert — is that a SET keeps them
apart. Generalisation: for any family of same-signature frozen value types, the hash
is a function of the field values ALONE; test identity through the container, never
through the digest.

### L70b ⛔⛔ When two groups share an INVARIANT RING, no function can separate them — so a brief's "σ_v-odd control" is unwritable, and the honest deliverable is the measured inertness

The brief asked for: *"the mirrored half genuinely differs from the rotations (a
control: a σ_v-ODD function is NOT constant across them while it IS across the SO(2)
half)"*. **No such function exists.** If `f` is constant on every `SO(2)_a` orbit and
σ_v carries each constant-μ circle onto itself, then `f(σ_v x) = f(x)` pointwise —
which is the same theorem `_sphere_mod_o2` records as its reason for existing.

`[M]` over **3 axes × L = 1..6 = 18 rows**, the descending-slot mask computed from
`generic_images(O2(a))` is `array_equal` to the one from `generic_images(SO2(a))`. So
dropping the mirrored half is **production-INERT at the shipped incommensurate
angles**, and no `descending_slots` row can catch it.

⭐ The regime where the mirrored half IS observable is the **degenerate sample**: four
right angles generate `C_4` without it and `C_4v` with it, so the existing
`test_a_right_angle_sample_falsely_admits_slots_from_L_four` reads 7 → 6 live slots at
`L = 4` and 10 → 8 at `L = 5`. Battery arm `c` (drop the mirrored half) reddens **7**
rows: my 6 structural `generic_images` rows plus that ONE pre-existing degenerate-sample
control — i.e. before the structural rows, the component split had exactly one witness,
and it lives in the *blindness* control rather than in any positive gate.

⟹ two transferable moves. (1) **Before writing a control that separates two groups,
ask whether their invariant RINGS differ** — one line of algebra, and if they coincide
the control is unwritable at every fixture. (2) Ship the impossibility as a
**measured, named blindness row** (`test_the_MIRRORED_half_is_inert_…_that_is_a_THEOREM`)
pointing at the regime that does discriminate; otherwise a later battery arm reddens
nothing there and reads as a coverage gap instead of a theorem.

### L70c ⭐ The stabiliser-maximality gate — "named by the LARGEST group with these orbits", as a runnable test

Reusable shape for any orbit-space / quotient catalogue:

> for every group `G` the lattice can spell:
> `G ⊆ entry.by`  ⟺  every generic image of generic base points leaves
> `entry.orbit_coordinates` unchanged.

The RHS is what *"these are the orbits"* means; the LHS is the lattice's answer, so
neither half can be wrong alone (`vv` #15). It is also the MAXIMALITY claim: a `by`
set too small leaves a group outside it still fixing the invariants (⟸ fails); too big
and an element moves them (⟹ fails).

`[M]` 2026-09-02: **140** (entry × group) pairs — 7 shipped sphere entries × 20 groups
whose generic images can be sampled — **0** mismatches, **33** inside / **107**
outside, 0.74 s. ⚠ The denominator EXCLUDES `D_∞h`, `SO(3)`, `O(3)`: they are
continuous with no axis to sample about and `generic_images` refuses them by design —
asserted in the same row so the exclusion cannot silently widen.

Non-vacuity companion, and it is the row that shows the rename bought something:
`[M]` **7** of the 20 groups sit inside `O(2)_x` (`Trivial, sigma_y, sigma_z, SO2_x,
O2_x, C_1, D_1h`) against **3** inside `SO(2)_x` (`Trivial, SO2_x, C_1`) — the four
edges the rotation-half naming would have lost, two of which (`sigma_y`, `sigma_z`) are
exactly what admits the Legendre basis on a σ_b-fold.

### L70d ⛔ A refusal that makes an OLD SPELLING unconstructible turns "revert the spelling" into a crash arm — it attributes nothing, and its attributable twin must be shipped beside it

Arm `h` (`LegendreBasis.domain` back to `SPHERE.quotient(SO2(axis))`) now RAISES at the
property, because the catalogue refuses the rotation half. `[M]` **144 reds / 5
collection errors across 19 files** — loud, wide, and about the *raise*, not about the
domain claim (`L31`/`L25`: a mutation that reds by raising has attributed nothing).
The attributable twin is arm `h2` (`invariance_group` returns the lower bound while the
domain stays legal): `[M]` **4 reds**, all pre-existing, in 3 files. Ship both, and
read only the second as evidence about the claim.

### L70e ⭐ The two arms with ZERO pre-existing catchers are the report's headline — compute new-vs-pre-existing per arm, not a total

`[M]` per-arm split over `tests/numerics + tests/transport` (3558 passed baseline,
17 arms, ~52 s each, every arm carrying a BITE assertion that printed its own defect):

| arm | reds | of which NEW | pre-existing |
|---|---:|---:|---:|
| `f` `_axial_contains` puts `O2 ⊆ SO3` | 5 | 5 | **0** |
| `i` `LegendreSpace.from_L` hard-codes the old name | 2 | 2 | **0** |
| `c` `generic_images` drops the mirrored half | 7 | 6 | 1 |
| `g` `candidate_groups` without the O2 rows | 2 | 1 | 1 |
| `d` the orbit space records the group it was asked for | 4 | 3 | 1 |
| `h2` `invariance_group` is the lower bound | 4 | 0 | 4 |

Two arms were **witness-less before this dispatch** — the properness of the axial
lattice's `SO(3)` edge, and the space name being READ off the basis's domain. A
whole-battery total (e.g. "17 arms, all caught") would have hidden both. `h2` is the
honest opposite: the invariance-group claim was already fully gated and my rows add
nothing there — worth saying, because a plan that credits every new gate equally is
overstating.

⭐ The two POSITIVE CONTROLS were chosen at different tiers and the wide one is the
useful one: `P1` (`_is_axis_supported` always True) reds **24** in 3 files;
`P2` (`O2.name` drops the axis) reds **624** in **57** files, because the entry's name
is a `FunctionSpace` identity component and every consumer keyed on it moves. The
`rotation_axis`-returns-None arm (`m`) reads the same shape (617 / 57) — those two are
the tree's real load-bearing surfaces for this member.

### L70f ⚠ A design delta arriving MID-DISPATCH is a re-key list, not a rewrite

The coordinator landed an accepted elegance review while the battery ran (mint
`orbit_stabiliser`; move the refusal into `Quotient.__post_init__` + a
`_catalogued_quotient` door check; drop the three `SO2_a` catalogue keys;
`_fixes_axis` loses `proper_only`). The message-fragment gate was the one at risk:
I had pinned **six** fragments and only the head *"is the orbit space S^2/O2_a"* was
promised verbatim. Re-posed to that head PLUS a **placement** leg (`ValueError`
carrying it, and NOT the catalogue's `NotImplementedError` "no catalogue entry") — the
second leg is the one that gains teeth after the move, since a check placed AFTER the
lookup, or a key merely deleted, reddens it.

⟹ when a refusal's ENFORCEMENT SITE is about to move, pin the sentence the refusal
exists to say (which the move preserves) and the ERROR TYPE / ordering (which the move
can break), never the diagnosis wording.

## L71 — #429 tracker 2.2b, the Γ-slot (plan delivered 2026-09-02, HEAD `4b7d24c3`, carve landing concurrently)

**Subject.** A geometry demands `(G⁰, Γ)`; a rule may live on a further quotient
`X = D/H` (a FOLD). Stage 0 becomes *"the descent arrow `quotient_onto(D, X)` exists
AND `Γ ⊇ X.by`"*; `SubgroupOfO3.is_invariant` routes a quotient-supported measure
through `Quotient.induced_action`, so a group acting trivially on the orbit space is
no longer read as "not closed" on the representatives.
Deliverables: `scratch/_22b_verification_plan.md`, `scratch/_22b_gates_draft.py`
(14 classes), `scratch/_22b_mut/mutplugin_22b.py` (14 arms, not run),
`scratch/_22b/*.py` (probes + a shadow of the design).

### L71a — ⭐⭐ THE METHOD: when the tree is held by a concurrent carve, snapshot it and SHADOW the design

The coordinator held `orpheus/` and the package went **unimportable mid-dispatch**
(`@dataclass` left decorating a newly inserted `def`). Two moves recovered the whole
dispatch and are the transferable half:

1. `git archive <HEAD> orpheus | tar -x -C scratch/<n>/pristine`, then neutralise the
   editable install's MetaPathFinder (`sys.meta_path = [f for f in sys.meta_path if
   "editable" not in …]`) and `sys.path.insert(0, PRISTINE)`. **The `.pth` finder wins
   over `PYTHONPATH`**, so a bare `PYTHONPATH=` does NOT re-point the package —
   measured, and it is the step everyone will skip.
2. Write the DESIGN as a shadow module outside `orpheus/` and diff BEFORE vs AFTER on
   real production objects. Its validity control is that it reproduces the shipped
   answers **exactly** on every input the design says it does not touch (`[M]` 395 of
   415 rows identical) — that is what makes the 20 changed rows evidence rather than
   a guess. This is a pre-carve verification plan's strongest instrument and it needs
   NO production edit, so it is legal under a "do not touch `orpheus/`" constraint.

`[M]` every prediction the shadow made reproduced verbatim against the landed carve
(4 flips, the walk, both stages, the slab literals, the embedding).

### L71b — ⭐⭐ CENSUS BY EVALUATION beats census by reading, and it answers the §6b question directly

A grep census over `is_invariant(` returns 61 test sites and CANNOT say which verdict
moves, because the measure is built by a fixture. A `-p` plugin wrapping the method,
returning the honest answer and recording `(test id, support kind, before,
after-shadow)`, ran a real suite and answered it: **1636 calls, 74 tests, exactly 1
verdict moves**. Supports seen: `S^2` 1402, `[-1,1]` 93, `S^2/O2_x` 88, `S^2/O2_z` 29,
`S^2/O2_y` 20, **`S^2/sigma_y` 4**. Same trick on the two registry stages: 107 + 59
calls, **0 move, and not one ever sees a fold**.
⟹ the §6b table becomes a measurement with a denominator instead of a reading.
⚠ Mechanics: the wrapper must reproduce the KEYWORD-ONLY signature (`*, atol`) or 72
tests fail for the harness's reason and the census reads as a catastrophe (`vv` #17,
the harness lies first); take a no-wrapper baseline in the same command.

### L71c — ⭐⭐ THE HEADLINE FLIP CAN BE THE LEAST FALSIFIABLE ONE

The design's named consequence was `Mirror('y').is_invariant(fold)` `False → True`.
`[M]` that verdict is produced by the step-2 short circuit (`H.contains(G)` — G acts
trivially on `M/H`) and **never reads a node**: it stays `True` on a fold with a node
DELETED and on a fold with a weight scaled 1.5×, while `D_2h`/`C_2`/`D_1h`/`σ_x` all
go `False`. So the flip a plan advertises gates the WIRING; the discriminating gate is
a different group. ⟹ for every advertised consequence, ask *which arm produced it* and
whether that arm can see the data — `vv-principles` #19 applied to a design's own
headline. Say it in the gate's docstring rather than hiding it.

### L71d — ⛔ A CONSEQUENCE LIST IS A UNIVERSAL AND OWES ITS DENOMINATOR

The design's §8 named **2** flips (`Mirror('y')`, `Dnh(2)`). `[M]` over each rule's own
`candidate_groups` set — 21 shipped rules, **415 rows** — the carve moves **20**, i.e.
**4 per fold**: `σ_y`, `C_2`, `D_1h`, `D_2h`. My own first pass under-counted to 12
because I used a HAND-WRITTEN group list (which had `Dnh(2)/(3)/(4)` but not `Dnh(1)`).
⟹ the denominator for a group-lattice claim is `candidate_groups(measure)`, never a
list you typed; `plan-authoring` §2, and it bit the person applying it.

### L71e — ⭐ A RETIREMENT'S BLAST RADIUS IS ITS SECOND CONSUMER, and a 1-of-3-axes agreement hides it

The design ruled *"the axial arm of `_embedded_nodes` RETIRES"*. `[M]` that function
has TWO production readers and only one is the invariance kernel: the other is
`Quadrature.ordinate_permutation`. Deleted, an `S^2/O2_y`/`O2_z` rule embeds as
`(μ,0,0)` (max |Δ| **9.603e-01**) and `ordinate_permutation(σ_x)` returns the μ→−μ
permutation where the identity is right — a silent wrong answer in a public accessor
on **2 of 3 axes**, invisible because the slab's own `S^2/O2_x` is the axis where the
two spellings agree. The landed carve took the re-point route
(`support.section_coordinates(nodes)`), so the hazard did not fire — and `[M]`
`tests/numerics` is **2857 passed / 0 failed** with no gate added, i.e. reverting that
half would still be invisible. ⟹ a "retire this internal" ruling is a §6b question:
enumerate the readers, and when the families agree on ONE parameter value, that value
is the blind spot.

### L71f — ⛔ A NEW GUARD CAN BE INERT AT THE TIER ITS END-TO-END TEST LIVES ON

Stage 0's new Γ-containment leg is load-bearing at the `admits_domain` API tier
(`[M]` 4 of 28 geometry×rule rows; without it a 1-D polar rule is admitted for a 2-D
geometry). At the `select_quadrature` tier it is **INERT**: with the leg removed —
a control strictly stronger than the change — the selector picks the identical spec,
parameters and point count on **16 of 16** rows, because stage 2's V conjunct refuses
the rule first. ⟹ price every new guard with a stronger-than-the-change control AT
EACH TIER, and write the inert tier into the gate's docstring, or a future green
end-to-end row gets credited to it. (`plan-authoring` §8's sharpening, arriving from
the guard side.)

### L71g — ⭐ THE EQUALITY SHORT CIRCUIT IN A LATTICE PREDICATE IS ITSELF A GATE

`admits_domain` reads "arrow exists AND (support == domain OR Γ ⊇ support.by)". The
`==` arm is not an optimisation: `[M]` `sigma_x.contains(O2_x)` is **False**, so asking
the Γ leg unconditionally makes the SLAB refuse its own Gauss-Legendre rule, and the
cylinder refuse every sphere rule — **10 of 28 rows**, the widest non-control arm of
the battery. ⟹ when a predicate carves out an identity case, gate the carve-out;
"for X == D there is nothing to contain" is a claim, and its counterexample ships.

### L71h — bit-exactness as a THEOREM, and its shelf life

`[M]` every law of the new machinery is exact: `π∘lift = id` `array_equal` on 3
families, `ind(g)∘ind(h) = ind(g·h)` **0.000e+00** over 128 pairs, the chart-closure
match residual **0.000e+00** on 3 folds × 8 elements, the axial chart path exact at 5
GL orders. The reason is a theorem, not a draw (`vv` #31): every shipped
`orbit_coordinates` is a COLUMN SELECTION and every `D_2h` element is a SIGNED
PERMUTATION, so no arithmetic touches a surviving coordinate. ⚠ **Write the shelf
life beside it**: the argument dies at the first group with a genuine rotation
(`C_3`/`C_6`, a hex lattice), and an `array_equal` pin then becomes a false red.

### L71i — ⛔ A NEW INVARIANCE NOTION SPLITS AN OLD ONE, and the sibling's PROSE goes false

`orbit_certificate` reads the AMBIENT nodes. After the carve `[M]` `σ_y`/`C_2`/`D_2h`
on a fold are `is_invariant=True` with `orbit_certificate=None` (while `σ_x` has both)
— correct, and it makes two production claims present-tense-false: the certificate's
docstring gives `None` exactly two causes (there is now a third), and
`DiscreteMeasure.quotient`'s refusal says *"this measure is not σ_y-invariant"*. The
coordinator repaired the second in flight (`"lies in the spent group sigma_y"`); the
first is still open. ⟹ when a predicate is re-posed on a new space, grep the module
for every OTHER function whose docstring enumerates that predicate's outcomes.

### L71j — the intrinsic-law gate a group-theory spec forgets

The ruled `normalises` spec (`H ⊆ {e, −I}` for `SO(3)`), read literally, answers
`SO3.normalises(SO3) = False` and `SO3.normalises(O3) = False`. Both are theorems the
other way. **Latent** (a `Quotient.by` is only ever `O2_a`/`σ_a`/`Trivial`), which is
exactly why it needs a gate rather than a note. ⭐ And the sharpest row the same gate
carries: `[M]` **`O2_x` CONTAINS `σ_y` but does not NORMALISE it** — the counterexample
to the intuition a reviewer supplies for free, and the reason step 1 must precede
step 2.

## L72 — #434 R1, "every question about a group is computed from its realization" (plan + gates delivered 2026-09-03; carve LANDED mid-dispatch, symmetry.py `d541cbe6…`→`3c83caf9…`)

**Deliverables.** `scratch/_r1_verification_plan.md` (12 measured corrections,
28 gates in 9 classes, the dim / identity_component / orbit_stabiliser tables,
a 20-arm battery table, 5 open rulings), `scratch/_r1_gates_draft.py`
(importable, `[M]` 28 passed / 10.46 s cold / 3.80 s warm, pyright 0 errors, 0
top-level name collisions with `tests/numerics/test_symmetry.py`),
`scratch/_r1_test_migration.md`, `scratch/_r1_mut.py` + `_r1_battery.sh`.

### L72a — the carve LANDED mid-dispatch, and that inverted the deliverable's evidentiary status

The brief said "BEFORE the main agent implements it". `[M]` the file was
already modified at first import (`dataclasses.is_dataclass(SubgroupOfO3)` True
and `g._tag = 'X'` raising, while the source I had read 12 minutes earlier
showed `__slots__` + a plain `__init__`). The hash moved **four** times during
the dispatch; `tests/numerics/test_{symmetry,manifold,registry}.py`,
`registry.py` and four `.rst` pages were migrated in parallel.

⟹ **Every gate became MEASURED rather than predicted**, and the §6b list became
a RESIDUAL. Discipline that made it safe: bracket every measurement with
`shasum -a 256 <file> | cut -c1-16` BEFORE and AFTER, take a `cp -a` pristine
copy before the first mutation, and `diff -q` after every battery batch (the
battery is pure monkeypatch, so the only file changes were the main agent's —
which `diff` then attributes correctly instead of alarming).

⭐ The headline the collision produced for free: `[M]` `tests/numerics/` =
**3004 passed / 0 failed / 46.02 s** under the full carve — 13 helpers retired,
`_NAMED_LATTICE` gone, `Cn(1) → Trivial`, `identity_component` corrected — and
**nothing reds**. The migration is a quality obligation, not a red list.

### L72b — ⛔⛔ a done-when spelled with `is` on a value type is a FALSE RED waiting

The plan's done-when read `SubgroupOfO3.Cn(1) is SubgroupOfO3.Trivial`. `[M]`
`is` is **False**, `==` is True: `__post_init__` normalises the TAG, so the
constructor returns a fresh instance carrying the normalised tag rather than
the pre-instantiated singleton. Landing that leg would have reddened a correct
implementation. ⟹ for a merge on a **value** type gate `==`, `hash`, `name`,
`repr`, container-dedup and the downstream door (`SPHERE.quotient(Cn(1)) ==
SPHERE.quotient(Trivial)`) — and say in the docstring that `is` is deliberately
NOT asserted, or the next reader adds it back. Pairs with L70a: the SEPARATION
half is asserted through the container, never through `hash(a) != hash(b)`.

### L72c — ⭐⭐ an independent construction of a finite group's ELEMENT SET is cheap, and it is the keystone

`[M]` all 22 finite realizations rebuilt in plain numpy from each group's
DEFINITION — `C_n` = Rodrigues `exp(θ[ê_z]ₓ)` (not the module's circle-point /
roots-of-unity form), `D_nh` = rotations ∪ rotations·σ_h ∪ half-turns at
`kπ/n` ∪ mirrors (the `D_n × {e,σ_h}` definition, not a closure), `σ_a` =
`I − 2n̂n̂ᵀ`, `O_h` = the 48 signed permutations by index assignment, and **`I_h`
by the FLAG construction** (the unique rotation carrying one
(vertex, neighbour) pair of the icosahedron onto every other, ×{±I}; production
runs a BFS closure of `{R_5, R_3, −I}`). Build cost **2.9 ms**; agreement with
production **22/22, worst per-element residual 1.166e-15**, six orders inside
`_ELEMENT_ATOL = 1e-9`.

⟹ **State where the independence STOPS.** The reference shares production's
standard SETTING (principal axis z, vertex on x) and must, because containment
in that module is *literal subgroup containment in that setting*; a conjugated
reference answers a different question. What is independent is the ALGORITHM.
Writing that sentence is what turns "an independent construction" from a slogan
into a claim.

### L72d — ⛔ the obvious element-set mutation is UNCONSTRUCTIBLE, and the obvious dim-2 mutation is a CRASH ARM

Two arms failed for structural reasons that are themselves findings:

* *"perturb one `O_h` matrix by 1e-6"* → `RigidMotion.__post_init__` refuses a
  linear part off orthogonality by more than **1e-12**
  (`ValueError: linear part is not orthogonal`). L70d's shape. The in-class
  mutation is a **SUBSTITUTION** — a legal rigid motion that is simply not an
  element of the group (5 reds). Corollary: `_ELEMENT_ATOL` can never be
  exercised by a non-orthogonal element, only by a wrong-but-legal one, which
  is why a "the window bites" gate would be signature-tautological (L35e).
* *"give `O(2)_a` a second independent generator"* (to gate "so(3) has no
  2-dimensional subalgebra") reds **14** rows — but `[M]` **10 of them by
  RAISING** (`_in_span`: `too many values to unpack`; `axis`: `only a torus has
  an axis`), which attributes nothing (L31/L25). The theorem is a CONSTRUCTION
  INVARIANT of `_in_span`, not an assumption. The attributable catcher is the
  STRUCTURAL row; the honest in-class torus mutation is a **rotated axis**
  (13 reds, all assertions).

⟹ before designing a numerical-perturbation arm, check the type's own
construction guard; if it refuses the perturbation, the in-class move is a
substitution inside the guard's admissible set.

### L72e — ⭐ a genuine MAXIMUM search is affordable as a gate once it is vectorised

The `orbit_stabiliser` claim is *"the LARGEST subgroup of O(3) with this group's
orbits"*. qa's probe searched it in Python loops. Vectorised —
`einsum` the candidate set `H·O(2)_{p₀}` (the finiteness argument: `g p₀ ∈ H p₀`
⟹ `h⁻¹g ∈ Stab(p₀)`), then filter by probe point with an analytic orbit
predicate — `[M]` **0.61 s over all 31 members**, 180-point stabiliser sample,
13 probes.

⭐ It needs **both halves**, and the second is the one that witnesses anything:
(a) MAXIMALITY — every surviving candidate lies inside the reported stabiliser;
(b) CORRECTNESS — every element of the reported stabiliser preserves every
orbit, **and where it GREW the growth is witnessed** (some sampled element of
`Stab(H)` is outside `H`). Without (b) the `SO(2)_a → O(2)_a` and `SO(3) → O(3)`
rows are unearned. `[M]` the growth census is exactly
`{SO2_x→O2_x, SO2_y→O2_y, SO2_z→O2_z, SO3→O3}`.

### L72f — the two rows NO arm reds are DECLARED reference-side controls, and saying so is the deliverable

`[M]` 20 arms, 20 bit, 20 red; **26 of 28** rows reddened. The 2 that are not —
`test_the_reference_construction_is_ITSELF_a_group` and
`test_a_WRONG_stabiliser_claim_fails_the_orbit_check` — call **no production
predicate** by design: one validates the reference (vv #17), one validates that
a wrong claim would fail the two rows above it. Their green is the *licence* to
read the neighbouring rows as coverage, not coverage itself. Both docstrings now
say so, because an audit that counts them is counting nothing.

⭐ And read the two **1-red** arms as the headline, not as weakness (L70e):
`identity_component_returns_self` re-installs exactly what shipped until
2026-09-02 and `[M]` **one** row in the whole tree notices — the one this plan
adds. Before it, that property had **zero** readers, zero tests and zero doc
roles, and was wrong on **21 of 22** finite members.

### L72g — ⛔ a MERGE collapses denominators silently: count DISTINCT, not entries

`Cn(1) == Trivial` made two roster constants hold duplicates, and both gates
still pass:

* `test_computed_containment_obeys_the_order_relation_laws`: `finite` list
  `[M]` **10 entries / 9 distinct**, and `order = {g: …}` is a dict, so it holds
  **9** keys while the loops iterate the list.
* `_SPELLABLE`: `[M]` **23 entries / 22 distinct**, and the gate asserts
  `n == 23` — a LIST length. Its pinned edge counts (`strict == 131`,
  `touching == 23`) COUNT the duplicate (`Trivial ⊇ C_1` and `C_1 ⊇ Trivial` are
  two "strict" edges for one group). Its docstring's *"they DO contain each
  other **while comparing unequal**"* went present-tense-FALSE.

⟹ after any value-merge, grep the roster constants and assert
`len(set(roster)) == len(roster) == N`; a length assertion on a list is blind to
exactly the thing the merge changed.

### L72h — cost is a DESIGN INPUT for a lattice gate: ration the denominator, name the exclusion

`[M]` the behaviour contract's full tables cost **17.79 s** (`is_normalised_by`,
31 × 336 motions) and **22.78 s** (`normalises`, 31²) — 40.6 s of pytest for two
rows. Per-member cost concentrates savagely: `I_h` 1.98 ms/call, `O_h` 0.86,
everything else ≤ 0.57, and for `normalises` the 44 pairs with `O_h`/`I_h` as
the OTHER argument are `[M]` **19.9 s of the 23.0 s**.

⟹ the gates ship **1426** (31 × 46 stratified motions) and **931 of 961** pairs,
each naming its exclusion in its own docstring, and the full tables stay in the
main agent's `scratch/_r1_behaviour.py`. A stratified motion set is chosen to
SEPARATE (mirrors, half-turns, quarter-turns, incommensurate rotations, improper
partners, diagonal and polyhedral axes), and its census is asserted: `[M]` **578
True / 848 False**, with the **4 CONSTANT columns** (`Trivial`, `C_1`, `SO3`,
`O3` — normal in `O(3)`, a theorem) named in the assertion so a criterion that
made one separable, or made a separable member constant, reds.

### L72i — ⭐ the two rows I got wrong were MINE, and chasing them produced the best gate in the file

First draft asserted `O2("x").contains(Mirror("x"))` and
`SO2("z").normalises(Dnh(n))`. Both are **False** and production is right
(`σ_x` has normal `ê_x`, so it FLIPS the axis and is not in the stabiliser; a
rotation about z carries `D_nh`'s in-plane `C₂'` axes to axes the group does not
contain). L35f's rule — *when a shipped predicate rejects your test input,
suspect the FIXTURE first* — held.

⭐ The repair is the sharpest row in the plan: on ONE group and ONE family,
containment and normalisation are **independent in both directions** —
`[M]` `O2_x`: `σ_x` not-contained/normalised, `σ_y` and `σ_z`
contained/not-normalised. That is the row that forces step 1 (normalise) before
step 2 (contains) in the invariance kernel, and it is strictly stronger than the
committed one-directional version.

### L72j — a name can be FREE in code and TAKEN in the prose corpus

`[M]` `class Realization` : **0** occurrences in `orpheus/` + `tests/`. And
`Realization` (capital R) appears **32** times in `.claude/`, in every case
naming the third axis of the operator campaign — *Operator · Splitting ·
Realization*, "how an operator is computed". `plan-authoring` §1's name clause
run FORWARD: the name is not refuted, it is *taken*, and a fresh session reading
it in `numerics/symmetry.py` arrives with the other meaning primed. Raised as an
open ruling rather than a blocker (the gates import whatever lands).

---

## L73 — #434 R4, "the lift is a derivation output, and an orbit space's dimension is a theorem" (plan + gates DELIVERED 2026-09-03, pre-carve; R1's 13-tree gate running concurrently)

**Route.** Bash + `.venv/bin/python` probes only (`scratch/_r4_probe{1..13}.py`,
`_r4_census.py`, `_r4_shim.py`). READ-ONLY on `orpheus/`, `tests/`, `docs/`,
`.claude/` throughout — a 13-tree gate was mid-run on the working tree
(`scratch/_r1_full_gate.log`, R1 uncommitted). Nexus tools were present and unused: every
question was a completeness census over spellings, which `plan-authoring` §2's FILTER
clause routes to Python `re`, not to a graph walk.

Deliverables: `scratch/_r4_verification_plan.md`, `scratch/_r4_gates_draft.py`
(31 functions / **102 rows**), `scratch/_r4_test_migration.md`.

### L73a — ⛔⛔ The carve's whole subject was Mode-12 invisible to the machinery it lives in, and the DECISIVE measurement cost one in-process monkeypatch

R4 replaces a mirror orbit space's lift (a hemisphere SECTION onto the fundamental
domain) with the orbit BARYCENTRE `P_H p` — a change of `max |section − projector| =
9.943e-01 / 9.735e-01 / 9.778e-01` in the ambient space. The invariance kernel
(`_orbit_space_closure`, `is_invariant`, `maximal_invariance_groups`,
`ordinate_permutation`, `induced_permutation`) reads `orbit_coordinates(...)` of whatever
the lift returns — and `orbit_coordinates` is exactly the column selection `P_H` re-writes,
so `π(g · P_H p) = π(g · p)` for every `g` in the normaliser and the kernel cannot see it.

`[M]` the decisive form — install R4's lift semantics over
`Quotient.ambient_representatives` in-process and re-run the campaign's OWN behaviour
battery (`scratch/_r1_behaviour.py capture`, 31 groups × the containment/normaliser/
invariance/walk tables): **0 of 9925 answers moved**, twice 9.1 s. That single number
replaced every argument in the plan and produced three design constraints:
(a) no kernel-tier row may be credited as a catcher; (b) every gate asserts at the AMBIENT
tier; (c) `chart ∘ lift == id` is a DECLARED BLIND leg (`[M]` true of the retired section
too), so it ships labelled with the teeth in `lift ∘ chart == P_H`.

⟹ **When a carve moves an intermediate that a chart later projects away, model the new
semantics as a monkeypatch and re-run the campaign's behaviour capture BEFORE designing
gates.** It is the cheapest possible Mode-12 stabiliser computation and it is empirical.

### L73b — The bit-vs-near split of a "block-diagonal ⟹ commutes" claim is TWO populations, and one `np.allclose` would have hidden the exact half

The brief asked: the closure now acts on barycentres, and every normalising `g` is
block-diagonal, so `chart(g·P p) == chart(g·p)` — bit-exact, or within the node window?

`[M]` over 3 folds × every representative of the 31 members that normalises `⟨σ_y⟩` (261
motions): **243 `array_equal`, 18 not**, `max |Δ| = 4.996e-16`, and the 18 are ALL `I_h`
elements. Re-partitioned by the right predicate:

| population | rows | `array_equal` | max\|Δ\| |
|---|---|---|---|
| signed-permutation normalisers | **100** | **100** | `0.000e+00` |
| the rest (all `I_h`) | **7** | **0** | `4.996e-16` |

Mechanism: `is_normalised_by` admits at `_ELEMENT_ATOL = 1e-9`, so an icosahedral element
fixing `ê_y` only to ~1e-16 carries non-zero off-axis entries and does not commute with
`P_H` exactly. The closure's node window is `_NODE_WINDOW_FACTOR × atol = 1e-7` — NINE
orders above — which is why L73a's 0 of 9925 holds.

⟹ **ship the row as TWO legs**: `array_equal` on the signed-permutation population (a bit
wall a future non-signed motion cannot silently weaken) and `atol=1e-13` on the rest with
the window quoted. A single `allclose` over the union is green either way and loses the
statement that the shipped traffic is EXACT.

### L73c — ⭐ A DRY-RUN SHIM turns a gate draft into a measurement, and it caught a mixed-subject assertion on the first run

`lessons` L31/L33 says ship a plan's gates as a runnable module. Here the API does not
exist yet, so the runnable form is a **shim**: `scratch/_r4_shim.py` installs
`_coordinate_chart`, `lift_coordinates`/`lift_codomain` (as ctor kwargs AND properties),
`orbit_barycentres`, `is_trivial`, the dimension law wrapped around `__post_init__`, the
`(M/H)/{e}` short-circuit and the widened `barycentre`, then
`pytest -p _r4_shim scratch/_r4_gates_draft.py`.

`[M]` **100 passed / 2 failed in 0.91 s**, and the 2 are exactly the claims a shim cannot
provide (real `dataclasses.fields` entries; a retired symbol's absence).

The find: my `test_the_five_call_sites_answer_identically` (subject: `is_trivial`) ended
with `assert barycentre(entry).domain is entry` — item 4's WIDENING, a different carve
item. A red there would have been unattributable between two unrelated subjects. It is now
its own row with a positive leg over all 8 entries and the narrowed refusal, which is also
the ONLY gate item 4 has.

`[M]` pyright over the draft: 28 errors, all 28 naming the not-yet-landed API, 0 others —
after re-spelling `sum(...)/len(...)` (types as `NDArray | float`) as the stacked `mean`
it is (`coding-elegance` anti-#19; `[M]` `array_equal` either way).

### L73d — ⛔ A filter's own POSITIVE CONTROL caught it, in the flattering direction, on the §6b constructor set

The §6b set for two new REQUIRED dataclass fields is "every direct construction". First
spelling: `^\s*Quotient\s*\(|=\s*Quotient\s*\(` → **1 hit**, and the control
`    return Quotient(` did NOT match (the line begins with `return`). Corrected
`\bQuotient\s*\(` → **10 hits**, including production's own three builders.
The wrong answer said *"only one construction site to fix"*.

Same session, same shape, on my OWN prose: I wrote *"a `getattr(…, 'name', …)` sweep over
`orpheus/` returns 0"* from memory; `[M]` it returns **3** — all three inside error-message
f-strings, none in a condition, so the CONCLUSION survived and the sentence did not.
Both corrections are recorded in place in the migration doc.

### L73e — The §6b members a spelling census cannot return, measured for this carve

- **`match` PATTERNS on the changed type** — `[M]` 3 sites (`measure.py:490`,
  `manifold.py:1947`, `basis/base.py:402`). All KEYWORD patterns, so inserting fields is
  safe; a POSITIONAL one would silently re-bind. One reading; its absence is invisible.
- **Docstring cross-references** — `numerics/manifold.py` and `symmetry.py` carry no
  `automodule`, so a dead `:meth:` renders as plain text at EVERY severity.
  `mcp__nexus__dead_references` is the only instrument that reads that surface.
- **A test HARNESS on every module's import chain** — `tests/_harness/references.py:31`
  imports `symmetry._embedded_nodes`; break it and the suite dies at COLLECTION with
  `0 collected` and both `^FAILED` and `^ERROR` reading zero (L68a).

### L73f — The opener question, answered in three measurements instead of one

*"Does projecting a fold's representatives to barycentres change what
`tests/_harness/references.py` asserts?"* The tempting single measurement is the function's
output. Three were needed, and only the third settles it:

1. `_embedded_nodes` output, 11 rules → `array_equal` on **9 of 11**; the 2 folds move.
2. the harness's ANSWER (partner map + refusal), 11 × 3 axes → **31 of 33 unchanged**; the
   2 that move are `axis="y"` **on a fold** (RAISES today at residual 1.15–1.19; identity
   permutation after).
3. **which axis each folded call site passes** → `[M]` `"x"`, both of them (a literal at
   `test_snmesh_realizer_wiring.py:439,450`; the CYL fold's only realized face carries
   `ReflectiveBoundary(axis='x')`, measured by building the `SNMesh`).

⟹ no committed assertion moves. ⭐ But the change is a real CAPABILITY — the harness stops
contradicting `induced_permutation`, which has answered the identity since #429 2.2b — and
nothing else in the tree states it, so it needs a gate of its own or it lands unrecorded.

⚠ And a quantified coverage note the census would have missed:
`tests/sn/sweep/curvilinear/test_coupled_pole_mu_level_invariant.py`'s `_CUBATURES`
includes 2 folded rows. `[M]` the gate passes on 8 of 8 before AND after, but on those 2
rows `|μ_y|max` collapses `8.688e-01 → 0.000e+00` and `9.082e-01 → 0.000e+00`, so its
docstring's *"μ_y/μ_z must be held"* goes vacuous in y on 2 of 8 (`vv` #20). The remedy is
the docstring note, not a re-key: feeding raw nodes would re-mint the Pattern-2 duplicate
that file records as removed.

### L73g — §6c witnesses that EXIST pre-carve, found by construction rather than by hope

Three of R4's claims land with a case the tree already carries, each measured:

- **the dimension law** — `[M]` BOTH forgeries construct today
  (`S^2/O2_z` on `Ball(2)`; `S^2/sigma_x` on `[-1,1]`), and both carry
  `fundamental_domain=None` so the pre-existing fd clause returns early and provably
  cannot see them. They are ITS witnesses and no other gate's.
- **retiring the three-arm `lift` branch** — `[M]` a hand-built
  `Quotient(base=SPHERE, by=OctahedralOh, realization=Ball(2), …)` CONSTRUCTS today
  (`O_h` is its own `orbit_stabiliser`; the law reads `2−0 = 2` ✓) and `entry.lift` raises
  `NotImplementedError: no lift is spelled for S^2/Oh`.
- **`(M/H)/{e}`** — `[M]` today it builds a second object `S^2/sigma_y/Trivial`, and
  `[M]` **no test pins that string** (Python `re` over `tests/`: 0); the only carrier is
  `manifolds.rst:7563`. So the change would otherwise land unwitnessed.

⚠ And the L43c companion: the dimension law and the fd gate are two clauses in one
`__post_init__`, and `[M]` the fd gate's HISTORICAL witness violates BOTH after R4. Each
owes a DISCRIMINATING input — measured and both constructible — and the ORDER owes its own
row (present fragment / absent fragment on the both-violating input).

### L73h — Instrument choice measured, not assumed: MORE trapezoid points is WORSE

The axial lift is the orbit CIRCLE's mean; the reference is a trapezoid of `R_θ p`. `[M]`
residual vs `n`: `8 → 2.220e-16`, `16 → 3.331e-16`, `32 → 5.551e-16`, `64 → 1.110e-15`,
`1024 → 2.587e-14`. The rule integrates `cos θ`/`sin θ` exactly for `n ≥ 3`; everything
past that is summation error. Ship `n = 16, atol = 1e-14` and SAY in the docstring that
raising `n` degrades it — otherwise a later session "strengthens" the gate into a false red.

### L73i — the DECLARED-BLIND control arm reddened, and its red set partitioned the suite

`chart_pair_reverses_BOTH_halves` was shipped as a null arm on a theorem:
`embed ∘ select` is the orthogonal projector onto the span of the selected columns,
and a projector does not know the ORDER its columns are written in, so reversing a
list in BOTH halves is exactly `P_H`. `[M]` 2026-09-03 it reddened **6 of 4597**:

| red | why |
|---|---|
| `TestTheEntryCarriesItsQuotientMapAndItsReference::test_the_numeric_map_IS_the_recorded_symbol` ×3 | the entry's own symbolic-vs-numeric chart pin reads the column ORDER |
| `TestTheInducedActionIsWellDefinedAndActsAsTheTheoremSays::test_sigma_x_on_the_sigma_y_fold_is_the_disks_x_flip` | a positional disk-coordinate read |
| `TestR4…::test_select_is_bit_identical_to_the_entrys_own_orbit_coordinates[S2/Trivial, R3/Trivial]` ×2 | the TRIVIAL builder uses `_all_coordinates` and never routes through `_coordinate_chart`, so the test's reversed `select` legitimately disagrees with the entry's identity chart |

Every projector row (A1, A3, A4, B, C, E) stayed GREEN — as designed. ⟹ the arm
did what no green arm could: it PARTITIONED the committed set into *"rows blind to
column order, by theorem"* and *"rows that gate the order"*, and it showed that A2
is loaded on the two trivial entries against a helper/builder disagreement.

### L73j — the bite check read its own mutant (LATE BINDING)

`dim_law_probes_a_POLE` reported `bit=0` in the smoke test. Cause:
`_generic_orbit_dimension` resolves `_generic_point` through the MODULE GLOBALS at
call time, so the captured `honest_orbit_dim` function object returns the MUTATED
rank too — the check compared mutant-vs-mutant. Repair: take the honest value
BEFORE the patch. `[M]` 18 of 19 arms bit on the first smoke pass; this was the
19th, and it is exactly the failure the smoke pass exists to find.

⟹ **a bite check that dereferences the patched name through a captured callable is
not a control.** Capture VALUES, not callables, on the honest side.

### L73k — a law that lands as PREVENTION-BY-CONSTRUCTION has no value-tier witness

`[M]` the arm replacing the dimension law's `rank[X p : X ∈ 𝔤]` with `group.dim`
reddens **0 of 4597**. Every shipped entry has `rank == dim H` — axial `1 = 1`,
mirror and trivial `0 = 0` — so the law's entire reason for being stated on the
ORBIT rather than on the GROUP is un-witnessed by the catalogue; and the gate I had
written (`test_dim_H_alone_would_be_WRONG_and_the_rank_helper_says_why`) asserts the
TEST's own rank helper, which a production mutation cannot touch.

**Repair, measured.** Two entries that are constructible on the landed tree and
refused the moment the law reads `dim H`:

| entry | honest law | `dim H` law |
|---|---|---|
| `S^2/O(3)` on `IndexSet(dim 0)` | `2 − 2 = 0` ✓ | `2 − 3 = −1` ⛔ |
| `R^3/O(3)` on `[0, ∞)` (`dim 1`) | `3 − 2 = 1` ✓ | `3 − 3 = 0` ⛔ |

`scratch/_r4_dimH_witness.py` — **2 passed honest / 2 failed under the arm**.
(⚠ `R^3/SO(3)` is NOT usable: `_assert_named_by_stabiliser` refuses it first,
naming `O(3)`. The stabiliser law preempts the dimension law on that spelling.)

⭐ **And the mirror finding: three natural mutants are UNINSTALLABLE, each refused
by a NAMED guard** — which is a finding, not a failure:

| mutant | refused by |
|---|---|
| a mirror builder shipping a half-meridian `fundamental_domain` against `Ball(2)` | the fd clause of `Quotient.__post_init__` |
| an axial builder realizing `S^2/O(2)_x` on `Ball(2)` | **the dimension law itself** — the forgery is unspellable at the BUILDER tier, so only D2's direct `Quotient(...)` can witness it |
| an axial builder re-naming the entry by its rotation half `SO(2)_a` | `_assert_named_by_stabiliser` (#432) |

A fourth candidate is inert **by TYPE, not by a guard**: reversing an `int` column
is the identity, so the axial arm cannot express the both-halves reversal at all.

### L73l — an R4-only red set is not `vv`#17's "mirror"; the discriminator is the consumer census

`orbit_barycentres_passes_ambient_width_through` (the pre-carve pass-through
restored) reddens **9 R4 rows and 0 of the 4588 others** — including the five
`_embedded_nodes` consumers and every `tests/geometry` mirror gate.

`vv`#17's red-set-by-IDENTITY clause would read that as *"the symbol has no
consumer; its pins are a mirror"*. It has two consumers (`_act_through`,
`_embedded_nodes`) and they are blind for a **measured** reason: the chart drops
exactly the column the projector rewrites, `[M]` 0 of 9925 kernel answers move.

⟹ the R4-only red set is the EVIDENCE that E2/E5 are net-new coverage. The two
readings — *no consumer* and *consumers blind by theorem* — are told apart by the
PRE-CARVE consumer census, never by the red count.

### L73m — the battery's measured table (19 + 3 arms, 4597-row denominator)

Scope `tests/numerics tests/geometry tests/transport` + the two SN
`_embedded_nodes` consumers; `-m "not slow"`, serial, `python -O`.
Baseline **4597 passed / 5 skipped / 1 xfailed / 0 failed in 51.9 s**; 22 arms
≈ 20 min. Pristine copies of the three carved modules taken before arm 1 and
`diff -q` clean after the last. Every arm carried a BITE CHECK; **22 of 22 bit**.

Every gate class A–I has at least one arm that reddens it. The two arms that
redden most widely are `embed_writes_ones` (243) and `dim_law_probes_a_POLE` (723
+ 39 ERRORs — a CRASH arm: every axial entry refuses at construction, so it reds
by RAISING and attributes the generic-point premise, nothing at the value tier).
The positive control `orbit_coordinates_returns_column_zero` reddens **99** (37 R4
+ 62 non-R4 across `test_manifold`, `test_symmetry`, `test_slab_orbit_space`,
`test_quadrature_directional`, `test_reemission_closure`) — chosen OUTSIDE every
entry's symmetry group, per `vv`#17's own trap.

## L74 — #434 R2, "invariance is the measure's question; groups import geometry only" (plan + gates DELIVERED 2026-09-03, pre-carve; branch `fix/angular-phantom-support`, HEAD `caad5bf3`, R4 uncommitted in the tree)

Deliverables: `scratch/_r2_verification_plan.md` (10 sections, 33 gate rows in 9 groups,
19-arm battery, 5 open rulings), `scratch/_r2_gates_draft.py` (**121** rows, dry-run
**18 failed / 103 passed in 5.92 s** through `scratch/_r2_shim.py`; pyright 18 errors, all
18 naming the unlanded API, 0 others), `scratch/_r2_test_migration.md` (the §6b set),
plus two frozen references captured at HEAD (`_r2_perm_baseline.npz` 45 rows,
`_r2_selection_baseline.json` 48 rows).

### L74a — ⛔⛔ The blocking obstacle to reversing an import edge was a CONSTANT, not a type; and `import <package>` alone stays rc=0 while 6 of 9 entry points die

R2's stated means: *"`manifold.py` imports `symmetry` at module scope"*. The plan
enumerated the KERNEL names leaving `symmetry` (`Quotient`, `RealSpace`,
`DiscreteMeasure`) and never asked what else `symmetry` imports from `manifold`. `[M]`
`symmetry.py:105` reads `AXIS_INDEX, AXIS_LETTER` too, at **6** sites
(`:560, 667, 1059, 1073, 1440, 1707`) inside `rotation_axis`, `mirror_axis`,
`_letter_of`, `_perpendicular_letter`, `_realize`, `_axis_vector` — none of which moves.

`[M]` `scratch/_r2_import_probe.py` — a shadow copy of `orpheus/`, four injected variants,
one `python -c "import X"` SUBPROCESS per (variant, entry):

| variant | clean |
|---|---|
| V0 baseline | 9 of 9 |
| V1 = the plan's literal means | **3 of 9** — `ImportError: cannot import name 'AXIS_INDEX' from partially initialized module 'orpheus.numerics.manifold'` |
| V2 = V1 + `symmetry` stops importing `manifold` (AXIS constants local; the rest under `TYPE_CHECKING`) | 9 of 9 |
| V3 = V2 + a real `invariance.py` + `measure → invariance` at module scope | **10 of 10** |

⭐ Under V1 **`import orpheus` alone still returns rc=0**, so a package-root smoke test
reports GREEN. The existing `test_entry_point_imports_in_a_fresh_interpreter` covers 6
entries and catches only 3 of the 6 deaths; the 4 that must be ADDED
(`numerics.manifold`, `numerics.measure`, `numerics.invariance`,
`numerics.quadrature.registry`) are exactly the uncovered ones.

⟹ **when a carve's goal is to REVERSE an edge, enumerate every name the reversed-from
module imports from the reversed-to one — including CONSTANTS — and inject-and-run before
designing a single gate.** A type moves with the concept; a constant does not, and it is
invisible to a review that reads the concept.

### L74b — Three of the carve's four behaviour changes are INERT on every shipped input, and each needed its own denominator

The plan listed four changes and one exception list. Measured separately:

| change | shipped witnesses | the measurement |
|---|---|---|
| position test at the NODE window (`atol` → `atol·100`) | **0 of 15 rules** | every rule's off-axis barycentre residual is `0.000e+00` (bit-exact — the axial lift puts μ ON the axis) or `≥ 5.774e-01`; a `10^11` gap, nothing in `(1e-13, 1e-11]` |
| `_distinct_azimuths` window from its argument | **0 of 15** | `n_az` identical at merge `1e-9` and `1e-13` on all 15 (8/16/24/40/12/20/36/8/8/2/2/2/4/2/2) |
| `_maximal` on STRICT containment | **0 of 31 members** | 0 distinct pairs among the 31 expressible `SubgroupOfO3` values contain each other (R1's `Cn(1)→Trivial` closed the only known one) |
| deleted kernel step 2 | **32 distinct (rule × group) rows** — a MUST-STAY-GREEN table, not a witness | 10 bare-sphere rules × `{Trivial}`; 3 GL rules × `{Trivial, σ_y, σ_z, SO2_x, O2_x, D_1h}`; 2 folds × `{Trivial, σ_y}` |

⟹ **a plan listing N behaviour changes owes N separate shipped-denominator measurements**,
not one exception list. Three of these four land green and unfalsifiable without a
manufactured fixture (§6c), and the fourth is the opposite shape.

The manufactured witnesses: a BARE-sphere 2-node measure at `(±1, ε, 0)` — bare so the
quotienting group is `{e}`, FINITE, so the position leg genuinely RUNS (on `S²/O(2)_a` the
barycentre is on the axis by construction and the leg is a tautology); four equatorial
nodes at `φ = 0, 5e-10, π/2, π/2+5e-8`; and a six-line duck-typed pre-order stub for
`_maximal`, which reads only `!=` and `.contains`.

### L74c — ⛔ A plan's stated behaviour change was ALREADY TRUE at HEAD

*"`symmetry_groups` on a measure whose only symmetry is `{e}` returns `(Trivial,)` (today
`()`)"*. `[M]` on the elegance review's own fixture (seed 7, 9 random unit nodes) both
`walk` and `bruteforce` read `(Trivial,)` **today** — the `()` was the `Cn(1)`/`Trivial`
alias pair, closed by **R1** a commit earlier. ⟹ the row is a REGRESSION PIN on R1, carries
NO information about R2's `_maximal` change, and its docstring must say so, or the
battery's expected null on it reads as a blind gate. **Measure the "today" side of every
before/after claim, not only the "after".**

### L74d — ⭐ Changing WHICH NODES a derived quantity reads moved 3 of 15 shipped answers the plan never listed

C3's fix is *"`candidate_groups` on the EMBEDDED nodes"*. Simulated over the whole shipped
denominator BEFORE designing gates (`scratch/_r2_opener3.py`): the three `gauss_legendre`
polar marginals move `{O2_x, σ_x}` → **`{O2_x, D_2h}`** (`|C|` 15 → 18), and
`folded_product(4,8)`'s candidate SET shrinks 20 → 18 (the barycentre projection zeroes the
`y` column, so 4 azimuths collapse to 2 and `C_4`/`D_4h` stop being offered) while its
ANSWER stays `{D_2h}`. The widening is correct — `D_2h ⊇ σ_x`, `D_4h`/`D_∞h` are refused
because their `C_4^z`/`C_∞^z` do not normalise `O(2)_x` — and it reds two committed gates,
one of whose NAME (`test_the_candidate_SET_is_untouched_by_the_carve`) becomes false.
⟹ **simulate the changed function over the whole shipped denominator before designing the
gates**; the answer that moves is rarely the one the finding is about.

### L74e — ⭐ A search function's two realizations can return the same SET in a different ORDER

`[M]` on a chart-spelled fold, `maximal_invariance_groups` returns
`['sigma_z','sigma_x','sigma_y']` by `walk` and `['sigma_x','sigma_y','sigma_z']` by
`bruteforce`. The walk pops a stack; the scan iterates the candidate list. ⟹ **every gate
on such a function compares `sorted(...)` or a `set`** — a tuple comparison is a flake
waiting for a candidate-order change.

### L74f — The shim dry-run refuted my OWN predicted fixture value

I predicted the azimuth fixture reads `3` at `atol=1e-13` both before and after. `[M]` the
post-R2 answer is **4**: with the merge window at `1e-13` the `5e-10` pair no longer merges
either, so BOTH the `1e-13` and the `1e-7` rows are witnesses and a third row at `1e-9`
is the control pinning the window the literal used to be. A prediction written into a
parametrize list is a `[M]` claim; the shim is what makes it cheap to refute.

### L74g — ⚠ `ast.col_offset` is UTF-8 BYTES, and my first quantification of the hazard was wrong in the alarming direction

A pre-existing `scratch/_r2_rewrite_is_invariant.py` (not mine; 2026-09-03 01:49) builds
source spans as `cumulative_char_offset[lineno-1] + node.col_offset`. `[M]` `col_offset`
counts **bytes** (`x = 'μμμ'; G.is_invariant(m)` → `col_offset = 14`, char index 11), so
the construction is wrong in principle. I was one command from publishing *"128 of 128
call sites corrupt"*, measured by the PROXY *"a non-ASCII line exists at or before the
call"*. Re-measured with the honest instrument — slice vs `ast.get_source_segment` —
the answer is **0 of 128**: the non-ASCII in these files lives in docstrings and comments,
never on a line before a call, and the cumulative part is char-indexed on both sides so
only the CURRENT line's prefix matters. ⟹ **a latent hazard with 0 witnesses is a rider,
not a defect**, and *"does this construction differ from the correct one on this corpus?"*
is a different question from *"could it?"* — answer the first with the correct
implementation as the reference, never with a proxy for it.

### L74h — the AST rewrite table, for the record

`[M]` **91** `.is_invariant(...)` CALL nodes (AST, `func` an `Attribute`), 5 files: 88 test
calls in 3 files, 2 in `symmetry.py` (the walk's own), **1 production consumer**
(`registry.py:1012`). The plan said *"170 test attribute sites / 3 files"* — 1.9× over; a
LINE census of `\bis_invariant\b` gives 116 lines / 9 files. Shapes (receiver, argument):
`Call/Name` 49, `Attribute/Name` 19, `Name/Name` 18, `Call/Call` 4, `Name/Attribute` 1;
`atol=` on 2 of 91; exactly one positional argument on 91 of 91; **0** sites with a
compound expression on either side, so the evaluation-order swap the rewrite introduces is
immaterial — stated because "immaterial" is a claim. Import sites of the moved names:
**30 in 12 files**, of which `tests/_harness/references.py:31` is on EVERY test module's
import chain (break it → rc=4 / 0 collected / 0 `^FAILED` / 0 `^ERROR`, both scanners
read zero — `L68a`).

### L74i — the R2 mutation battery (`scratch/_r2_{mut.py,battery.sh}`), and the three harness defects its PRE-LANDING run exposed

23 arms (22 + a positive control), modelled on `scratch/_r4_{mut,battery}.sh`. Built
against an API that does not exist yet — `scratch/_r2_edit_production.py` has not run, so
`orpheus/numerics/invariance.py` is not on the tree. `[M]` full run pre-landing:
**23 UNINSTALLABLE, 0 FAILED, 0 ERROR, pristine diff silent, ~14.5 s/arm over 1186 tests
in 9 scope files**; **15 of 23** then BITE-validated against the unlanded API through
`scratch/_r2_shim.py` (`-p scratch._r2_mut -p scratch._r2_shim` — the shim registered
LAST, because pluggy fires `pytest_configure` in LIFO registration order and the first
attempt with the natural order reported *"the module itself is absent"* on every arm).

**⛔ A `_precondition` must be an arm's FIRST statement, and it must raise
`Uninstallable`, not the bite's `RuntimeError`.** `[M]` the first draft's
`h_symmetry_reexports_the_kernel` rebound `symmetry.maximal_invariance_groups =
getattr(INV, "symmetry_groups", None)` BEFORE its precondition ran; pre-landing `INV` is
absent, so it bound a live production symbol to **`None`** and then reported UNINSTALLABLE
— **13 reds attributed to nothing**, on a run whose header says the arm did not install.
A partial install under a failed precondition is worse than a crash: the reds look like a
finding and the header denies the arm ran. ⟹ separate the two verbs by what they mean —
`_precondition` = *the tree is not in the state this arm models* (expected, quiet);
`_bite` = *the mutant is inert on its witness* (a finding) — and let no arm mutate before
its precondition has passed.

**⛔ A bite check that asserts `SUT is mutant` proves the REBIND took, not that the mutant
DIFFERS — `vv` #19 at the harness tier.** `[M]` `i_err045_message_reverted` first asserted
only the rebind and duly installed pre-landing, reddening **0**: the honest message and the
"reverted" message are IDENTICAL there, because the diagnosis R2 corrects has not been
corrected yet. Re-posed to CALL both guards on `folded_product(4,8)` axis `y` and compare
the two message strings, it correctly reports UNINSTALLABLE until the correction lands.
⟹ a bite must compare the mutant's ANSWER against the honest one on a named witness;
"the patch is in place" is the one reading that carries no information about bite.

**⛔ A carve can dissolve a mutation's own distinction — check the arm against the CURRENT
tree, not against the tree the arm was designed for.** `b_single_motion_face_bypasses_the_
closure` was specified as *"match ambiently instead of on the chart"* and spelled
`permutation_preserving(embed(measure), motion.on_points(embed(measure)), …)`. `[M]` the
bite refused it: **after R4** `_embedded_nodes` returns orbit BARYCENTRES, and a σ_y fold's
barycentres carry `y = 0`, so σ_y maps each to itself and the ambient match is
bit-identical to the chart match. The honest pre-tracker-2.2b reading is
`measure.nodes` — the stored REPRESENTATIVES, whose σ_y-mates are absent. One carve
(R4's projector lift) had silently made the intended mutation a no-op.

**Other mechanics worth carrying**: every captured symbol through
`getattr(module, name, None)` so an unspelled name reports UNINSTALLABLE rather than
crashing the battery (the R4 INTERNALERROR lesson); `_rebind` sweeping every `sys.modules`
holder because `from X import name` binds the function OBJECT; the three SOURCE-parsing
gates (the module-scope import census, the deferred-import census, the `_orbit_closure`
call-site count) mutated by a `pathlib.Path.read_text` patch **scoped to one path**, so
nothing is written to disk and the source mutation is crash-safe by construction; the
positive control forced to run FIRST by the driver; and one arm — a module-scope import
cycle — declared **UNINSTALLABLE by construction**, because the gate it must redden spawns
a fresh interpreter that re-imports clean source, with its witness named
(`scratch/_r2_import_probe.py`) rather than left as an unexplained blind row.

## L75 — #434 R3, "the registry names what it spends, what it leaves, and what it owes" (plan + gates + battery DELIVERED 2026-09-03; carve LANDED mid-dispatch and MOVED TWICE MORE)

**Shape.** `AngularSymmetry(continuous_isotropy, discrete_residual)` →
`(spent, unspent, owed)`; stage 0's fold licence moves from the OWED closure to a new
UNSPENT group; `manifold.spent_group` retired for a total predicate
`SubgroupOfO3.is_subset_of_product` (`H ⊆ Γ·K`). Deliverables:
`scratch/_r3_{verification_plan.md,gates_draft.py,test_migration.md,mut.py,battery.sh,
capture.py,physics.py,predicate_laws.py,quadrants.py,census.py}` +
`scratch/_r3_baseline_HEAD.json`.

**Terminal state.** 42 gate rows / 8 classes, 42 passed in 1.84 s, pyright 0.
Battery **16 arms, 16 installed and BIT, 14 redden (32/22/16/13/8/8/7/6/6/4/4/2/1/1),
2 DECLARED NULLS at exactly 0**, baseline 0, SHAs stable, pristine diff clean.
Predicted count delta +42 (`tests/numerics` 3213 → 3255; tree 10965 → 11007).

### The tree moved THREE times during one dispatch

06:04 the production carve; ~06:2x the test migration (15 reds → 0, and the R2
baselines moved `scratch/` → `tests/numerics/data/`); ~07:0x the ELEGANCE pass,
which re-signatured `is_subset_of_product` to KEYWORD-ONLY, delegated its two
conjuncts to a new `Realization.is_subset_of_product`, and replaced
`admits_domain`'s body with `self.domain_refusal(measure) is None` (a new
`-> str | None` verb). Two battery arms went UNINSTALLABLE mid-run and six crashed;
the `shasum` bracketing is what attributed it to the writer instead of alarming.

### Refutations of the brief, measured

1. `measure.quotient_group` returns **`None`**, not `Trivial`, for a bare rule.
2. A D∞h/σ_x fold of the SLAB's domain is **NOT constructible** (two different
   refusals, one layer down) ⟹ the retired `spent_group` coset arm had **no
   constructible witness**, so "total, no `NotImplementedError`" is satisfied
   VACUOUSLY and needs an honest-scope row.
3. The moved cells are **5 of 168**, and **2 are not in the plan's list**:
   `product(4,8)/σ_x` is refused for the **cylinder** too (`σ_x ∉ D_1h`).
4. `TestR2SelectionIsUnchanged` **did not cover R3 — it RED on 4 of 4**:
   `[M]` 96 of 96 domain-mismatch strings moved, 8 of 8 symmetry strings and
   0 of 48 choices did not.
5. The γ⁻¹ arm can NEVER bite (a theorem, §L75a).

### L75a — the inverse direction of a coset search is a THEOREM, not a weak arm

`∃γ∈Γ: γ⁻¹r ∈ K` and `∃γ∈Γ: γr ∈ K` are the SAME claim, because the existential
ranges over a GROUP, closed under inversion (substitute `γ' = γ⁻¹`). `[M]`
**0 disagreements over 891 (H, Γ, K) triples, Γ = C_4 included** — so the nullity
is the group axiom, NOT (as the brief supposed) an artefact of the shipped table's
involution-only Γs. Its sibling: the SIDE of the product (`ΓK` vs `KΓ`) is
un-witnessable in this vocabulary — `[M]` equal as SETS on **all 81** finite
`(Γ, K)` pairs, because every spellable member is built from coordinate-axis
rotations and coordinate-plane mirrors and a 45° mirror is unspellable. Ship both
as DECLARED-NULL arms with their denominators and an INVERTED bite check (the
mutant must AGREE); their green is a licence to read the neighbours, not coverage.

### L75b — a PRODUCT predicate whose shipped rows all have a trivial factor

`[M]` all four geometry rows: slab/sphere `(K=O(2)_x, Γ={e})`, cylinder
`(K={e}, Γ=D_1h)`, cartesian2d `(K={e}, Γ=σ_z)`. So `Γ·K` is always `Γ` or `K` and
**the whole reason the predicate is not `contains` is unexercised by the
registry**. Reachable in the vocabulary — `[M]` **137** triples where Γ is
load-bearing (covered WITH it, refused WITHOUT) — so the §6c witness is
manufactured, four rows, and mandatory. ⟹ census a new relation's SHIPPED
arguments for degenerate factors before crediting its new structure.

### L75c — the physics claim behind a registry field, and its three findings

`unspent` is a claim about the CONTINUUM equation; a gate cannot verify it. What a
gate CAN do is measure the DISCRETE solver's answer, which every downstream fold
rests on: solve a deliberately asymmetric fixed-source problem, compare ψ at
ordinate n with ψ at the ordinate the group element maps it to; a positive leg per
element of the group, a NEGATIVE leg per element outside it. `[M]` 1.0 s for three
fixtures:

| geometry | rule | σ_z | σ_y | σ_x |
|---|---|---|---|---|
| cartesian2d | product(4,8), 5×3, corner source | **0.0** EVEN (32/32 moved) | 6.043e-01 NOT | 8.175e-01 NOT |
| cylinder | folded_product(4,8), refl/vac | **0.0** EVEN (16/16) | **no permutation** | 7.224e-01 NOT |
| slab | gauss_legendre(8) | 0.0 **VACUOUS** (0/8) | 0.0 **VACUOUS** (0/8) | 6.493e-01 NOT (8/8) |

⛔ **(i) The σ_y half of the cylinder's `D_1h` is UN-WITNESSABLE by construction.**
`[M]` a cylindrical `SNMesh` admits only CARRYING quadratures — **15 of 15**
`folded_product` rules pass `assert_carrying_quadrature`, **0 of 20**
`product`/`lebedev`/`level_symmetric` do. Every admissible rule IS the σ_y
quotient, so no cylindrical solve stores both signs of μ_y: Mode 12 at the fixture,
no fixture findable. The honest gate asserts the STRUCTURAL commitment instead.

⭐ **(ii) The vacuity guard caught a claim I would have published.** On the slab,
σ_y and σ_z read `0.0` — and their permutations are the IDENTITY (a 1-D polar rule
carries μ_y = μ_z = 0 on every ordinate, ERR-080's territory). Reported without the
guard, *"the slab is even under σ_y and σ_z"* contradicts `unspent = Trivial` with
an authoritative number. ⟹ every evenness leg asserts `perm moves k/N` with k
written down.

⚠ **(iii) Measure the quadrant census on `Quadrature.mu_x/mu_y`, NOT on
`measure.orbit_barycentres`.** After R4 the barycentres are `P_H p`, so a fold's
mirror column is exactly zero and `np.sign` reports **0 of 4 quadrants for every
fold** — plausible, flattering, wrong. On the ordinate cosines: unfolded 4 of 4;
σ_z fold (LICENSED) **4 of 4**; σ_y fold (REFUSED) **2 of 4**, μ_y > 0 on all 16.

### L75d — three harness defects, each caught by its own guard

* **`textwrap.dedent` strips FOUR spaces from a method's source**, so a
  `_source_mutant` target copied at class indentation never matches. The
  precondition reported UNINSTALLABLE instead of silently no-op'ing.
* **A re-TYPED mutant's re-worded `raise` reds a `match=` gate for a reason the
  mutation is not about.** Three arms reddened
  `test_a3_a_continuous_covering_factor_is_refused` on a message mismatch before
  the arms were rebuilt as `inspect.getsource` transforms (L44i, earning its place
  again).
* **A bite check must call the LIVE class attribute**, not a captured helper that
  routes through the ORIGINAL guard — the finiteness arm's bite read the honest
  predicate and reported the mutant inert (L73j's family, one frame out).
* ⭐ And a *fourth*, in the gates rather than the harness: the non-vacuity guard on
  the independent-reference row (`0 < trues < rows`) fired on `H = Trivial`, where
  the reference answers **99 of 99 True BY THEOREM** (`{e} ⊆ ΓK` always). Branch it
  and name the theorem; do not weaken it.

### L75e — a `-> str | None` refusal verb, and WHICH clause refuses WHICH input

The elegance pass replaced stage 0's boolean with `domain_refusal(measure) -> str |
None`, collapsing TWO guards into one value (L35f: `is None` proves only that SOME
guard fired). It owes an INPUT-side isolation and DISJOINT fragments (L43c). And
⛔ my own first draft asserted the wrong clause: `quotient_onto(S², S²/O(2)_x)`
EXISTS (the entry's own quotient map), so a 1-D rule for a 2-D geometry passes the
ARROW and is refused by COVERAGE. `[M]` the shipped split is **arrow 14 / coverage
3 / both 0** over 4 geometries × 4 registered specs — so the pre-verb disjunctive
message named a SATISFIED fact on all 17 refusals. Which clause refuses which input
is a MEASUREMENT, not an intuition.

### L75f — the red-set composition, read by identity

3 of 14 biting arms redden ONLY the draft (`a_component_leg_reads_dim` 2,
`a_finiteness_guard_dropped` 1, `d_post_init_checks_only_unspent` 1). `vv` #17's
identity clause would read that as "a mirror, not a gate"; L73l's discrimination
says otherwise, and the discriminator is the pre-carve CONSUMER census: the
consumers exist (`admits_domain`, `__post_init__`) and are blind FOR A STATED
REASON — the shipped table's only continuous `spent` is `O(2)_x` and the only
continuous `quotient_group` is the SAME `O(2)_x`, so the distinction cannot reach
the registry; and the two guards have no shipped witness at all (§6c). ⟹ net-new
coverage of public API, not a mirror.

### L75g — the battery's scope, with its exclusion priced

`[M]` by a validated text census the WHOLE R3 API — `select_quadrature` 51,
`GEOMETRY_ANGULAR_SYMMETRY` 45, `AngularSymmetry` 42, `admits_domain` 25,
`admits_symmetry` 16, `domain_refusal` 8, `is_subset_of_product` 16 — has **ZERO**
sites outside `orpheus/numerics/` + `tests/numerics/`. That table is what licenses
scoping to `tests/numerics` (**3255 collected / ~26 s per arm**, 16 arms ≈ 7 min)
and excluding `tests/sn` (~80 min, the pre-merge gate's business). An excluded
directory with no number beside it reads as an oversight.

## L76 — #426, the (n,2n) anisotropy carve (plan + 5 drafts DELIVERED 2026-09-03, PRE-carve; branch `fix/n2n-anisotropy`, HEAD `72aa4a59` → `ef915e2d` mid-dispatch)

Deliverables: `scratch/_426_verification_plan.md` + `scratch/_426_draft_test_{be_reflected,
tape_pin,h5_roundtrip,role_ast,diffusion_n2n}.py`. All five dry-run, `pyright` **0 errors**;
`[M]` 41 rows total — **22 RED today** (the §6c red-before), 19 pass. Predicted delta
+89 → **11096** (54 MEASURED, 34 design, 1 corpus-coupled). The carve is a two-step
data→operator un-truncation: `Isotope.sig2` / `Mixture.Sig2` become Legendre STACKS,
and the (n,2n) operator gains an anisotropic arm on a shared `Transfer*` core.

### L76a — a two-list clamp can delete a THIRD channel, and 2 of 13 isotopes are the witness

`[M]` `load_isotope(n, 294).sig2.nnz` — **H_001 = 0, B_010 = 0** (no MF=6/MT=16
section at all); BE009 8195, U_235 6067, NA023 1451, O_016 372. The solver clamps
`L = min(scattering_order, min(len(m.SigS)−1))` (`sn/solver.py:1359-1363`) and `[M]` it
is SILENT: with one material at `len(SigS)=1`, requests 0/1/2/5 all return **0**.
If a retype makes `Sig2` a list and the clamp joins it into the same `min`, then **any
mixture containing H-1 or B-10 forces the whole solve to P0** — silently deleting the
ELASTIC P1/P2, `[M]` worth **+5787 pcm-relative** on the campaign's own fixture (14× the
effect the campaign exists to add). ⟹ **when a carve turns a scalar datum into a per-ℓ
LIST, census which shipped members have an EMPTY channel before letting the new length
join any `min`** — the empty ones are exactly the members nobody thinks about, and the
regression is 14× the feature. Companion: the same hazard makes the *control arm* of the
flagship gate wrong if it is built by SHORTENING the list (it would then measure the
elastic anisotropy under the (n,2n) name); zero the VALUES at the same length.

### L76b — a "bit-identical by design" step has a DENOMINATOR, and its new capability had 0 witnesses

The plan's step 1 is "bit-identical: no operator reads ℓ≥1 yet, and `scattering_order ≤ 2`
truncates the wider elastic stack to today's". True — and its denominator is the
`scattering_order` the tree actually requests. `[M]` census over `tests/ orpheus/
derivations/ examples/`: **130 × `=0`, 54 × `=1`, 3 × `=3`, 1 × `=2`** — the `=2` is an
`err_msg` STRING, and all three `=3` sites build SYNTHETIC mixtures by hand. ⟹ **no
shipped GENDF-library solve runs above P2**, so the un-clamping the step creates lands
with ZERO witnesses (§6c) — and `[M]` the data it un-clamps is not noise: BE009 elastic
`max|·|` per ℓ = 4.84e+1, 3.61e+0, 1.70e+0, **6.64e-1, 4.54e-1, 3.70e-1, 2.86e-1**, i.e.
ℓ=3 is 18 % of ℓ=1. ⟹ **when a step widens a truncation, the denominator of "bit-identical"
is the set of ORDERS the tree requests, not the set of files it touches** — census the
literal, and if nothing requests the new range, the step owes a witness for the capability
AND an explicit note that its physical consequence is unmeasured.

### L76c — a PHYSICS bound gate is one-sided, and the paired mutation is what proves it

`|Σ_ℓ| ≤ Σ_0` entrywise (`Σ_ℓ/Σ_0 = ⟨P_ℓ(μ)⟩`, `|P_ℓ| ≤ 1`) is a genuine REFERENCE-class
gate needing no solver. `[M]` 0 violations on `{BE009, U_235} × {MT16, MT2}`, ℓ=1…6;
max ratios BE009/MT16 0.9603→0.3190, BE009/**MT2 0.9997**→0.9980. Two consequences:
**(a)** the threshold must be `1 + 1e-9`, never tighter — the elastic ℓ=1 margin is 3e-4;
**(b)** it catches only INFLATION (a stray `(2ℓ+1)` reads ≈2.9, a stray ×2 ≈1.9) and is
blind to DEFLATION. The two-sided catcher is a RATIO-INVARIANCE row: the yield strip is a
row-DIAGONAL, so it cancels in `Σ_ℓ/Σ_0` and the stored ratio must equal the RAW tape
ratio exactly. ⭐ And the way to PROVE the pair rather than assert it is a battery arm that
must be GREEN on one gate and RED on the other: `scale**ℓ` with `scale ≈ 0.5` SHRINKS the
ratio ⟹ the bound stays green ⟹ only the ratio row reds. **A declared one-sidedness needs
its own arm, or it is an unmeasured claim wearing a docstring.**

### L76d — do NOT assert monotone Legendre decay: the thermal channel is not monotone

`[M]` BE009 MT=221 max|·| per ℓ = 1.32e+1, 9.23e-1, 4.78e-1, 3.53e-1, 2.60e-1,
**1.38e-1, 3.10e-1** — ℓ=6 exceeds ℓ=5. MT=16 IS monotone (1.09e-1 → 7.17e-3, 7 of 7) and
MT=2 is. The "smooth forward-peaked kernel" reading is a property of the (n,2n) channel,
not of the ingest; a monotonicity leg written from it would false-red the day the thermal
channel is widened. ⟹ **before promoting an observed regularity to an assertion, run it on
every channel the same code path serves.**

### L76e — the flagship's cost was 10× a naive layout, and the fix REDUCED the tree's cost

First layout of the ingest pin called `convert_gxs("BE009")` inside each parametrized row:
`[M]` **247.57 s** for 19 rows. `[M]` `convert_gxs("BE009")` = **17.80 s** (it builds all
4 temperatures); `convert_gxs("U_235")` = **71.58 s**; the raw parse is 1.96 s / 5.42 s.
Hoisted to one module fixture: **26.98 s**, same verdict. ⭐ And then the sharper move: the
rows belong INSIDE `tests/data/test_n2n_yield_convention.py`, which `[M]` costs **38.43 s
for 10 rows** because it calls `convert_gxs` **twice**. Merged behind one fixture the file
becomes ≈27 s for 29 rows — **the 19 new rows land at NEGATIVE marginal cost**. ⟹ **before
minting a new test file, price the EXISTING file that already pays your fixture's cost**;
the Pattern-2 answer and the cost answer coincide.

### L76f — the moment path crosses a HEAD-RANK split that every candidate fixture is on one side of

`_block_contraction` (`transport/material_field.py:186-199`) dispatches on
`len(head.shape)`: **rank 2** for the real harmonics, **rank 1** for the flat Legendre
basis a 1-D rule binds (#429/ERR-080). `[M]` every fixture in reach is rank-1 or clamped:
the campaign's flagship is a slab under `gauss_legendre(8)`; the existing
`test_n2n_operator.py` fixture is `gauss_legendre(4)`; and the analytic k_inf ladder rows
this plan adds run at `scattering_order = 0` because `[M]` `homo_2eg_n2n` has
`len(SigS) == 1` and the clamp reads it. ⟹ the ℓ≥1 moment path would have landed gated on
ONE of two dispatch arms. **When a carve moves code through a rank/branch dispatch, list
the fixtures per ARM before believing any of them covers it** — and remember a fixture can
be on the right chart and still never reach the arm because a CLAMP took it out.

### L76g — the cheapest end-to-end catcher was 5 ms and the family had ZERO

`[M]` `solve_diffusion_1d({0: get("homo_2eg_n2n").materials[0]}, 4-cell reflective slab,
keff_tol=1e-13)` → `k = 1.6532258064516114`, rel **2.686e-16** vs the analytic
`1.6532258064516119`, in **0.005 s**; under ν₂ₙ 2→1 / 2→0 / 2→3 it reads 20.2 % / 37.2 % /
24.1 % off. The family had `[M]` **0 of 113** catchers because `[M]` 625 mixtures are built
in that tree and **one** has a non-zero `Sig2` — never handed to a solve. ⟹ **when a
census says a whole family is blind, look for an EXISTING registry case before designing
machinery**: the fixture, the analytic reference and the independence (the derivations tree
spells its own `2.0 *` literal) were all already shipped. Same shape closed the SN analytic
tier: adding `2eg_n2n` to the k_inf ladder is 3 lines, `[M]` **+12 rows / +2.4 s**, all
passing today at k rel ≤ 3.2e-14 and spectrum rel ≤ 4.5e-13, and BOTH functionals redden
20.2 % / 10.5 % under the mutation.

### L76h — a `slow`-only catcher is zero coverage, and the not-slow replacement was 0.9 s

`[M]` (inherited from the #428 census, re-priced here) MC's only ERR-023 catcher carries
`@pytest.mark.slow`, so under the canonical `-m "not slow"` the MC tree reads *39 passed /
0 red* with the multiplicity mutated; run alone the same test FAILS in 84 s. The fast
replacement: `solve_monte_carlo(mats, MCParams(n_neutrons=50, n_inactive=10, n_active=30,
seed=42))` = **0.9 s**, `k = 2.04087` vs the analytic `2.045454545454546` (**−0.224 %,
0.47 σ**), and ν₂ₙ 2→1 reads `1.87606` (**−8.28 %, ≈17 σ**). ⟹ **a stochastic method's
"too slow to gate" is usually a statement about the PRECISION target, not about the
catcher** — a 3 % band on a 1 500-history run separates a 20 % defect at 17 σ.
⚠ And the rebinding trap that goes with it: `[M]` `orpheus/mc/solver.py:36` binds
`_N2N_MULTIPLICITY = float(N2NKernel.multiplicity)` at IMPORT, so a battery patching only
the `ClassVar` reports MC inert. Patch every rebinding site (`vv` #17), measured.

### L76i — the §6b census classes a "38 attribute reads" number does not contain

`[M]` my AST pass over `orpheus/ tests/ derivations/ examples/` for `Sig2`/`sig2`:
**39 attribute LOADS**, **18 attribute STORES** (`x.Sig2 = …` — 16 in `tests/`, 1
production, 1 stale diagnostic), **72 kwargs**, **3 subscripts**. A prior census reported
"38 attribute reads"; a STORE is a different member and every one of the 16 test sites
hands a bare `csr_matrix` to a NON-frozen `Mixture`. ⭐ And the subscripts are the sharp
ones: all 3 are `orpheus/moc/core.py`, where `self.sig2[i]` indexes a per-REGION list —
so after the carve `mix.Sig2[0]` and `self.sig2[i]` are two different `[0]`s one line
apart and a mechanical replace conflates them. ⟹ **for a scalar→list retype, split the
census by ast CONTEXT (Load / Store / keyword / Subscript)**; the Store class is invisible
to a "reads" number, and the Subscript class is where the repair goes wrong.
⚠ Sibling: the gate that ENCODES the truncation carries the issue number in its
`pytest.fail` MESSAGE (`test_material_field.py:328-333`) — `grep -rn "#426" tests/` is the
only filter that finds it; a symbol grep does not.

---

## L77 — CS4c step 5, "each binding acts through the body its ends select" (plan DELIVERED 2026-09-04, PRE-carve; branch `refactor/cs4c-step5-construction-selected-bodies`, HEAD `86c17898` = `main f90f7914` + one plan commit)

**Deliverable:** `scratch/cs4c_step5_verification_plan.md` — extends
`scratch/cs4c_verification_plan.md` §11.6's 3-arm B-5 skeleton into 11 gate
classes (G5.1 … G5.9 + G5.2b + G5.4b), an 18-arm battery on three measured
scopes, per-tree deltas, and 6 open rulings. The design under verification is
`.claude/plans/cs4c_binding_design.md` §19 (R-1…R-6 ✅ [R] user 2026-09-04).

### L77a — ⛔⛔ The moment composite's interior is NOT axis-built, so the step's flagship new binding is UNCONSTRUCTIBLE as designed

§19.2's whole mechanism is *"the interior body is fixed at construction by WHICH
END of the retained flux-analysis face the domain's interior is"* — the domain's
interior is either `flux_analysis.domain` (angular) or `flux_analysis.codomain`
(the harmonic-moment space, the 2-D windowed SI iterate).

`[M]` `Mesh2D(8×4)`, `Quadrature.level_symmetric(4)`, `mixture A` 2g, `L = 1`;
`BulkAnalysisOperator(S.flux_analysis, sn.full_field_space).codomain`:

| space | `.axes` | interior shape |
|---|---|---|
| angular composite interior | `[Axis(24,), EnergyAxis(2,), Axis(8,4)]` | `(24, 2, 8, 4)` |
| **moment composite interior** | **`None`** | `(2, 3, 2, 8, 4)` |

`TransferOperator._scalar_interior_space` (`transfer.py:637-647`) is
`FunctionSpace.of_axes(*interior.axes[1:])` read off **the DOMAIN's interior** and
raises its own *"the composite interior must be axis-built to name the scalar
sub-space"* when `axes is None`. `isotropic_energy` is built on it
(`transfer.py:667`) ⟹ a moment-bound sibling raises at FIRST ACCESS.
`FissionOperator.from_solver_data` (`fission.py:309-319`) carries the same
precondition ⟹ a moment-bound F is unconstructible by tier 2.

⭐ **The repair the tree already contains:** the scalar sub-space must come from
the CODOMAIN — which is the ANGULAR composite for BOTH siblings — equivalently
from the retained `source_reconstruction` face. `[M]`
`S.source_reconstruction.codomain == angular_interior` **True**,
`== moment_interior` **False**; `S.flux_analysis.codomain == moment_interior`
**True**. So the widened admission is genuinely TWO-SIDED:
`domain.interior ∈ {flux_analysis.domain, flux_analysis.codomain}` (selects the
body) and `source_reconstruction.codomain == codomain.interior` (re-keyed from
the DOMAIN, which is where it reads today). Today BOTH `__post_init__` guards
compare against `self._interior_space` (`transfer.py:452-464`); the second is
present-tense correct only because every shipped binding is an endomorphism.

⭐ **And the first RED falls out of the same probe, at zero cost:** constructing
the moment-domain sibling today raises
`"ScatteringOperator: the flux-analysis face is bound to a different angular
space than this binding's interior — mint the faces from the SAME posed space
(tier 2 does)."` — i.e. §19.2's widening has its §6c witness in the guard it
widens.

`[M]` `FullFieldSpace.from_blocks(S.flux_analysis.codomain, ffs.trace_space)`
**`==`** `BulkAnalysisOperator(...).codomain` but **`is` False** — the composite
is NOT interned, so an ends gate spelled `is` is a false red.

### L77b — ⛔⛔ A "no carrier dispatch" AST gate is LEXICAL and three carrier parses live one frame out

The plan's done-when (1) reads *"0 `singledispatchmethod` and 0 carrier
`isinstance` inside any `apply`/`apply_transpose`/`solve`"*. `[M]` my AST census
over `orpheus/transport/operators/` (positive control:
`TransferOperator._apply_impl` found) reproduces the design's numbers exactly —
**3 tables / 10 registered arms + 3 base bodies = 13**; **12** carrier
`isinstance` lexically inside those verbs, at the exact lines the plan lists —
AND finds **3 more in HELPERS those verbs call**:
`isotropic_transfer.py:147` (`_scalar_composite_source`, the angular-vs-scalar
family parse, one frame out of `apply`) and `transfer.py:902/:903`
(`add_iso_source`).

A lexical gate reads **0** after the carve while the family parse survives —
§6c's mirror (a gate that goes green while its subject lives). ⚠ The
discriminating filter is the isinstance **TARGET**: `[M]` a further **9** helper
hits are `isinstance(space, FullFieldSpace)` — *space* parses, legitimate,
staying. ⟹ the gate must state its predicate, carry the reachability half, and
NAME its carve-outs (R-5's LMT moment hatch; `add_iso_source`, which §19.7
keeps: `[M 5b]` 16 test nodes + 345 production calls).

### L77c — ⭐ A 200-seed sweep PARTITIONED the gate set exactly along the base/subclass line the design proposes

R-1 proposes a shared base `AngularLift` = the frame's ℓ = 0 conjugation
`R₀ E A₀ / W`, with the ℓ ≥ 1 redistribution on the transfer subclass only.
`[M]` 200 seeds (`default_rng(s)`, `s ∈ 0…199`), GL8 slab, `mixture A` 2g, 20
cells, ψ standard-normal `(8,2,20)` — production fast path vs the frame form:

| binding | `array_equal` | max abs Δ |
|---|---:|---:|
| `F` (`apply` vs `full_fission_kernel/W`) | **200 / 200** | **0.0** |
| `S` at `L = 0` (isotropic) | **200 / 200** | **0.0** |
| `S` at `L = 1` (anisotropic) | **0 / 200** | 2.220446e-16 |

⟹ `array_equal` is legitimate AND seed-STABLE for the BASE's own law (`vv`
anti-#31 satisfied by a sweep, not by one draw), and illegitimate for the
subclass's ℓ ≥ 1 sum, whose draw-stable statistic is the ABSOLUTE
`max |Δ| ≤ 2.3e-16` (a `nulp` band there pins a seed). The measurement is the
strongest available evidence that R-1's split is the right cut — and it came from
running the sweep the tolerance question forces, not from reading the design.
Companion anchor: Euclidean reciprocity `⟨Fψ,χ⟩ = ⟨ψ,Fᵀχ⟩` reads rel
**4.18e-16** (seed 5).

### L77d — ⛔ `-W error::DriftWarning` is NOT a bit-identity wall on this tree; 9 of 19 cases already drift

`lessons L58c` recorded the escalation as a 1-ULP wall (measured then on a
subset). `[M]` today over `tests/sn/regression` (19 collected, 59.9 s): plain run
**19 passed**; escalated **9 failed / 10 passed**, drifts 1–11 ULP
(`slab_2g_homogeneous` 1 · `slab_2g_3reg` 1 · `sphere_2g_homogeneous` 2 ·
`sphere_2g_3reg` 11 · `cyl_1g_folded_4x8` 1 · `cyl_1g_folded_2x4` 4 ·
`cyl_2g_3reg_folded_4x8` 2/4 · `2d_1g_LS4` 3 · `2d_2g_LS4_het_si` 2).

⟹ used as an absolute gate it is 9 false reds; used as a **DELTA** — *the drift
SET is these 9 and each case's ULP count is unchanged* — it is exact and free.
⭐ `2d_2g_p1_aniso_dd_8x4_het_si` is **NOT** in the drift set: it is bit-exact
today AND it is the windowed case (L77e), so it is the step's single best anchor.

### L77e — ⭐⭐ The whole windowed non-endomorphism has a 0.55 s frozen witness, and the first spy that looked for it read ZERO

`[M]` descriptor-protocol spy over
`tests/sn/regression/test_dd_regression.py -k 2d_2g_p1_aniso` (1 passed, **0.55 s**):

```
143  ScatteringOperator.apply  <- HarmonicMomentFlux  @ composite[24x2x8x4]
143  N2NOperator.apply         <- HarmonicMomentFlux  @ composite[24x2x8x4]
143  LegendreMomentTransfer.apply <- HarmonicMomentFlux @ plain[2x3]
143  IsotropicScattering.apply <- ScalarFlux           @ plain[2x8x4]
143  IsotropicN2N.apply        <- ScalarFlux           @ plain[2x8x4]
```

⛔ **My FIRST spy rebound `_apply_impl` and read 0 calls on this same suite.**
`TransferOperator.apply` is an ALIAS (`apply = _apply_impl` at class-body time)
bound to the ORIGINAL dispatcher object, so rebinding `_apply_impl` afterwards
changes nothing the alias sees. The `vv` #29 recipe is the fix and it is
mandatory here: wrap `cls.__dict__[verb]` and call
`descr.__get__(self, cls)(x, …)` so `singledispatchmethod` still dispatches.
A 0 from the wrong handle is indistinguishable from a dead path.

### L77f — ⭐ The plain binding is BIT-FOR-BIT the composite binding today: the ends-select-the-carrier fence's first red is 9 of 18

`[M]` the 18 cells at HEAD, 2g/6-cell diffusion binding (`mixture A`,
`Mesh1D(0…2, 6 cells, vacuum right)`, `rng = default_rng(2026)`):

| operator | binding | ndarray | FullField | ScalarFlux |
|---|---|---|---|---|
| `C` | composite / **plain** | ndarray / ndarray | FullField / **FullField** | raise / raise |
| `IsoTransfer` | composite / **plain** | ndarray / ndarray | FullField / **FullField** | **ndarray / ndarray** |
| `IsoFission` | composite / **plain** | ndarray / ndarray | FullField / **FullField** | **ndarray / ndarray** |

⟹ `binding kind → outcome` is NOT a function today, on any of the three
operators. That is the fence's §6c red-before, and it also means the fence must
NOT ship before R-4's refusals: an expectation table equal to today's behaviour
has no case to catch.

### L77g — ⭐ Two censuses that under-count by exactly one member, and both members are the gate's own positive control

* **The carrier population.** `[M]` a `__subclasses__` walk from
  `numerics.field.Field` after importing only the three package `__init__`s reads
  **19** concrete leaves; after importing **every module** of
  `transport/{fields,source_sinks,residuals}` it reads **20** — the missing one is
  `AngularBoundaryFlux`. Honest population: 7 flux / 7 source-sink / 5 residual /
  1 `CrossSectionField`. §19.5's own enumeration names **5** pairs; the tree has
  **7** (the two `RadialCharacteristic*` pairs are absent from the design text).
  `[M]` `grep "SOURCE_SINK\|role_partner\|as_source"` over both packages = **0**,
  as the design says.
* **The role gate's scope.** `[M]` inserting a base ABOVE `TransferOperator` does
  not change `TransferOperator.__subclasses__()`, so `test_transfer_roles.py` is
  genuinely unchanged — **and therefore nothing gates the new base's own
  population.** That absence is the finding: a one-row addition
  (`_all_subclasses(AngularLift) == {TransferOperator, FissionOperator,
  ScatteringOperator, N2NOperator}`) is owed in the same file.

### L77h — ⭐ The one-level witness is a ROUTE claim with a 0-vs-5309 first red, and its value legs MUST stay green

`[M]` spy over `tests/sn/solve/test_sn_adjoint_certification.py` (15 rows,
56.7 s): `IsotropicScattering.apply_transpose` **5309**,
`IsotropicN2N.apply_transpose` **5309**, `IsotropicFission.apply_transpose`
**0** — D11 reproduced exactly. Both bypasses verified verbatim:
`sn/solver.py:2951` feeds `RadialCharacteristicEmission(F.kernel, …)` and
`fission.py:219` reads `self.energy.kernel.apply_transpose`.

⚠ The two routes are NUMERICALLY IDENTICAL (`F.kernel` **is**
`F.energy.kernel`), so no value gate can ever see the difference — `vv` anti-#26
in its purest form. The instrument is a call counter, installed at
`pytest_configure` (a `cached_property` warmed by an earlier solve masks a later
patch — `L68b`), counting the SPECIFIC verb and asserting the sibling counts are
unchanged so the row is attributable.

### L77i — measured scope costs (the battery's whole budget)

`[M]` serial, canonical flags, this Host `.venv`:

| scope | rows | wall |
|---|---|---|
| **C** = transport + sn/architecture + diffusion + homogeneous + sn/operators | 2332 passed / 1 skip / 1 desel / **16 xf** | **92.75 s** |
| **C+W** = C + sn/regression + 3 windowed files | 2381 / 1 / 1 / 16 xf | **153.77 s** |
| **A** = `test_sn_adjoint_certification.py` (15 rows, 56.66 s) + the ERR-082 record (5 rows, 29.49 s) | 20 passed | **86.2 s** |
| *excluded:* `tests/sn/solve` whole | 208 passed | **258.85 s** (2.8× C+W for 5 reachable rows; the windowed + gate-dispatch files extract at **2.85 s**) |
| *excluded:* `tests/sn/eigenvalue` | 67 passed | **118.74 s** |

Per-tree, `[M]`: transport **738** collected (737 + 1 skip), `sn/operators`
**1283/1284**, `sn/solve` **208/210**, `sn/architecture` **163** (ledger 139 =
129 passed + **10 xfailed**, `--runxfail` = 2 `R1[L,B]` + 8 `R6[B×8]`, every one
red for its documented reason), diffusion **115**, homogeneous **50**.
18 arms ≈ **43 min**.

### L77j — declared non-changes worth WRITING DOWN, or a reviewer reads the carve as a weakening

* `[M]` `FullFieldSpace.axes` is **`None`** for the angular AND the moment
  composite, and `assert_energy_extent_conforms(space, ng=7, …)` **admits** on
  both — so the per-END energy admission is ALREADY inert on every SN composite
  binding, and the moment sibling adds no new inertness (consistent with `L61b`'s
  `skipped_axesless: 578 / raised: 0`).
* `[M]` `test_monomorphic_leaves.py`'s G1.3 asserts `TypeError` + operator name +
  `"FullField"` in the message and its own docstring says *"no assertion here
  reads the mechanism"* ⟹ the one-body admission satisfies all 40 rows unchanged.
  `_leaf_set` builds through `from_solver_data`, so the ctor spellings are
  untouched. **No ledger flip is scheduled for step 5** (§8.1) ⟹ its 10 xf carry.
* `[M]` the homogeneous solver binds `MultiplicationOperator` on `_pose_space(mix)`
  — a PLAIN space — and R-4 KEEPS the ndarray arm on the plain binding. ⟹ the
  older plan's §8.3 step-5 row (*"the ndarray arm's deletion makes the Mode-11
  sentinel vacuous"*) is **partially VOID**; only the surrogate's `# the
  singledispatchmethod` comment re-keys.
* `[M]` `dataclasses.fields`: F = `domain, codomain, energy, frame`;
  T = `domain, codomain, transfer, flux_analysis, source_reconstruction`. F has
  **no faces** and no `isotropic_energy`; `_interior_space` is a **METHOD** on F
  and a **PROPERTY** on T (both spellings are §6b members and a grep for either
  alone returns a confident partial); `total_weight` reads `self.frame.measure` on
  F and `self.flux_analysis.frame.measure` on T, `[M]` both `2.0` on a GL8 slab.
* `[M]` D3 reproduced: on `mixture A` 2g, `N2NOperator.is_isotropic` is **True at
  `scattering_order` 0, 1 AND 2** while `S.is_isotropic` is False at L ≥ 1 ⟹ every
  (n,2n) ℓ ≥ 1 mutation is blind on the legacy fixture, and the battery's positive
  control MUST use the Be-reflected fixture or a synthetic ℓ = 1 `Sig2`.
* `[M]` the diffusion anchor: `FlattenedOperator(loss, template).as_matrix()` is
  `(20,20)`, `M[0,0] = 6.422521008403363`, `‖M‖_F = 38.663330648012135`,
  sha256 `8009de67d27a26cd…`, `k = 0.4036481463911869`;
  `leakage.is_assemblable` / `boundary.is_assemblable` **True**,
  `fission.is_assemblable` **False** ⟹ the lift's `assemble()` is owed for TWO
  leaves (the transfer iso pair), not four.

---

## L78 — #448, "the eigenvalue finalize must return a flux that solves the equation it reports" (gates SHIPPED + battery RUN 2026-09-05, PRE-carve; branch `fix/angular-phantom-support`, HEAD `8cc69e7f`)

**The defect.** `solve_sn`'s finalize (`orpheus/sn/solver.py:2577-2579`) builds
`Solution.angular_flux` from ONE sweep whose rhs is `Fφ/k + Σ_{s,0}ᵀφ + 2Σ_{2n,0}ᵀφ`
— **P0 only**. At every `scattering_order ≥ 1` the ℓ≥1 emission is absent from the
reconstruction while the loss arm the iterate converged against carries it, so the
returned ψ solves a different equation and its own angular moment does not reproduce
the `scalar_flux` shipped beside it. `keff` and `scalar_flux` come from the power
iteration and are unaffected (`[M]` the fixed-source cross-route agrees to 1.67e-11
at BOTH orders) — which is exactly why every eigenvalue-value gate in the tree is
structurally blind.

**Deliverables.** `tests/sn/solve/test_eigenvalue_finalize_reconstruction.py`
(45 rows; `[M]` **10 failed / 35 passed / 44.9 s**, pyright 0), 28 pre-carve `.npy`
anchors under `tests/sn/_data/finalize_reconstruction_448/`, the battery
`scratch/_448_battery.{py,sh}` (11 arms, run), and `scratch/_448_verification_plan.md`.

### L78a — the claim layer that needs no reference: `φ = ∫ψ dΩ` is the DEFINITION

ORPHEUS has no structurally-independent eigenvalue reference for a heterogeneous
reflected 421-group slab (MMS does not prove eigenvalues; there is no closed form).
The way past that is not a weaker gate but a **different claim layer**: `Solution`
ships BOTH members and the scalar one is defined as the angular one's zeroth moment,
so the identity is a real constraint on the finalize with no external truth needed —
and it is a *flux-shape* claim, so the pillar rules are satisfied. `[M]` it separates
by 1.6e6–3.6e6 × its band on 8 arms. ⟹ **when a solver returns two members of one
object, the reduction identity between them is a free L1 gate — and it is the only
gate that can see a defect in the RETURN.**

### L78b — ⛔⛔ two guards whose COMPLEMENTS do not cover the exit

`_exit_balance_defect` opens `if record.fully_converged: return None`; its stated
complement `_certify_within_group_exit` fires on the converged side — **but inside
the INNER solves, on the ITERATE**. Between them nothing ever evaluates the object
the caller receives. `[M]` `balance_defect = None` and `nwarn = 0` on 12/12 converged
rows. ⟹ **when a codebase advertises a "complementary pair" of guards, draw the two
complements and check they COVER — a pair can be complementary in one variable
(converged / not) and both silent in another (iterate / return).** The docstring's own
sentence ("the complement of a guard reaches the states its partner never visits")
was true and did not notice that both partners visit the same *object*.

### L78c — the same diagnostic is LIVE on the truncated path and CORRUPTED there

`[M]` same slab, `keff_tol = flux_tol = 1e-12`, `max_outer` 3 → 6 → 12:
L=0 `balance_defect` 1.2497e-05 → 1.3151e-08 → 8.5933e-12 (falls **1.45e6 ×**);
L=1 7.48654e-02 → 7.48474e-02 → 7.48474e-02 (falls **1.0002 ×**). So every
anisotropic eigenvalue solve that exits on a budget ships a `ConvergenceWarning`
carrying a number that is a #448 floor, not a truncation measure. ⟹ **a diagnostic
whose docstring forbids thresholding can still be gated on whether it RESPONDS to
the knob it claims to measure** — a rate claim, not a magnitude claim. (Ruled OPEN
for the user: `_exit_balance_defect`'s prohibition may be read as absolute.)

### L78d — ⛔⛔ an UNBALANCED manufactured Σ₂ₙ makes φ differ from ∫ψ_conv by an EXACT GLOBAL SCALE

`[M]` `scratch/_448_l0n2n2.py`, same slab, adding `Sig2` to a library mixture:

| mixture | `int(iterate) vs φ` | scale spread | G1(L=0) | G1(L=1) |
|---|---|---|---|---|
| `Sig2 = 0` | 6.492e-11 | [1.000000, 1.000000] | 1.806e-11 ✅ | 2.815e-02 |
| `Sig2 ≠ 0`, `SigT` UNCHANGED | **1.002e-01** | **[1.100212, 1.100212]** | **3.100e-02** ❌ | 5.212e-02 |
| `Sig2 ≠ 0`, `SigT += rowsum(Sig2)` | 1.918e-11 | [1.000000, 1.000000] | 1.238e-11 ✅ | 3.628e-02 |

The spread is constant to 6 d.p. — a PURE global scale, the power iteration's
normalisation reading a medium that emits without removing. The damage is not a
wrong number: the **L = 0 CONTROL reds too**, so the gate attributes nothing and a
reader concludes the defect bites at P0. The balanced spelling is the house one
(`tests/cp/test_verification.py:181`, `tests/mc/test_gaps.py:742`:
`sig_t = sig_c + sig_f + rowsum(sig_s) + rowsum(sig2)`). ⟹ **any manufactured cross
section must be BALANCED into `Σ_t`, and an L = 0 control is the cheapest instrument
that notices when it was not.** ⚠ `[M]` **every** `xs_library` mixture (A/B/C/D ×
2g/4g) ships `Sig2 = 0`, so a fast (n,2n) fixture must be manufactured — there is
no library alternative.

### L78e — ⛔ a SKIPPED step can be inert while a WRONG one is 2.6e7 × the band

Battery M5 (skip `_reflect_outflow_into_inflow` in the finalize) read **0 reds**.
Bite-checked (`scratch/_448_bite.py`): the mutation INSTALLS (1 skip/solve) and moves
the answer by `[M]` **2.03e-13** (slab-reflective) / **2.31e-15** (sphere-reflective)
/ **exactly 0.0** (vacuum), against traces of 4.2e-01 / 2.4e-03 / 1.7e-01. It is
hypothesis (c), the DOF is annihilated: at the fixed point the converged inflow
ALREADY equals `B·ψ_out`, so re-applying `B` is idempotent — the production comment
said so and this measures it. The catcher is **M5b, a WRONG `B`** (double the
reflected trace): `[M]` G1 5.207e-12 → **2.758e-01** and 3.164e-11 → **2.569e-01**,
with the vacuum arm BIT-IDENTICAL. ⟹ **when a "remove the step" mutation is null,
try "corrupt the step" before writing the gate off — idempotence at a fixed point
makes REMOVAL invisible and CORRUPTION loud.** And the consequence for the carve is
a design datum, not a gate: if the fix drops that call, the change is behaviour-
neutral on the converged path and **no value gate anywhere can witness it**, so the
honest artefact is the number, not a test.
⭐ Bonus: M5b also reddened `cart2d` (its y-faces are reflective) — an unpredicted
third witness, found because the arm was RUN.

### L78f — ⭐⭐ the DECLARED PARTIAL NULL that states the gate's own Mode-12 blindness

Battery M9 removes the ℓ≥1 emission EVERYWHERE (`TransferOperator.is_isotropic →
True`, converged solve included). `[M]` **0 G1 reds** — G1[L1] goes fully GREEN,
because a finalize that drops a term which is not there is CONSISTENT — while 7
`G2[L1]` frozen anchors and all 9 activation rows red. ⟹ **G1 measures CONSISTENCY,
never the PRESENCE of the ℓ≥1 term**, and the three test classes partition the claim
(`TestTheLGe1TermIsLive` = presence, `TestKAndPhiAreNotAffected` = value,
`TestTheReturnedFluxIsSelfConsistent` = consistency). An arm designed to go green is
the only instrument that can state that partition.

### L78g — the FINALIZE-PHASE hook: how to mutate one block of a function that reuses shared verbs

`compute_fission_source`, `_cell_average_angular`, `_reflect_outflow_into_inflow` are
all called from the inner solves too; mutating them globally changes the CONVERGED
answer instead of the reconstruction (vv #18's over-powered mutation). The mechanism
is three lines: **wrap the module-level binding of whatever the target block calls
LAST before it** — here `orpheus.sn.solver.power_iteration`, read by the finalize's
own call site at `:2561` — and flip a phase flag on return. Every finalize-only arm
then becomes installable on the PRE-carve tree, which is what let this battery
validate the gate set before the fix existed.

### L78h — the per-arm partition, measured (the anti-#20 ledger)

| finalize-scoped mutation | reds — and ONLY these |
|---|---|
| M2 P0 scatter dropped | ALL 8 arms, BOTH orders (20 of 45) |
| M4 fission rhs ×1.01 | same 20 |
| M6 `_cell_average_angular` ×1.001 | 19 = M2 minus `G3c[L0]` (the defect is measured on `final_psi_a` BEFORE the cell-average runs in `_package_solution` — a correct blindness) |
| M3 (n,2n) P0 dropped | **`slab_vac_n2n` + `be_reflected` ONLY** (12) |
| M5b a WRONG `B` | **the three reflective-face arms ONLY** (13) |
| M5 `B` skipped | **0** (L78e) |
| M9 ℓ≥1 removed everywhere | **0 G1**; 7 G2[L1] + 9 activation (L78f) |
| N1 returned trace ×1.5+0.1 | **0** — bite-checked: `max|Δtrace| = 3.214e-01` on 4.427e-01 (72 %) with the bulk BIT-IDENTICAL, so a real annihilation |
| M0 positive control (wide) | **44** |
| P1…P4 post-carve arms | `Uninstallable`, loudly, naming the symbol the fix owes |

`M2`/`M4` are the arms that prove the **L = 0 CONTROL rows have teeth** — without
them a green L0 is compatible with "the identity is trivially satisfied".

### L78i — the one migration item no `tests/` grep returns

`[M]` `docs/theory/foundations/operator_algebra.rst:4043` carries a **DECLARED**
`.. implements:: scattering-aniso-composite / :by: …TransferOperator.build_aniso_source`
(confirmed declared, not inferred, by `provenance_chain`). Retiring the method deletes
the edge. The equation keeps a second declared implementer (`…kernel`, `:4055`), and
the block's prose describes `_redistribute_ordinates` character for character, so the
`:by:` must be **re-pointed, not deleted**. ⟹ **a retirement audit's three searches
(graph callers, text grep, direct constructors) miss a fourth surface: the corpus's
DECLARED `:by:` provenance edges.** One regex over `docs/theory/**/*.rst` for
`:by:` × the retiring names answers it; here it returns exactly 1 of 26 doc hits.

### L78j — the surviving-vocabulary map, for the migration

| retiring | survivor | note |
|---|---|---|
| `SNSolver._add_scattering_source` | `S.transfer.add_p0_source(Q, φ)` | the field verb IS `add_iso_source`'s whole body |
| `SNSolver._add_n2n_source` | `N2N.isotropic_energy.transfer.add_p0_source` | already the delegator's body |
| `SNSolver._build_aniso_scattering` | `S._redistribute_ordinates(AngularFlux(...))` | guards move to the caller |
| `TransferOperator.add_iso_source` | `self.transfer.add_p0_source` | ″ |
| `TransferOperator.build_aniso_source` | `S._redistribute_ordinates(ψ)` | **the exact body after the two guards** ⟹ the `p1_build_aniso_source` pre-T3 snapshot re-keys with numerics and `nulp` bound UNCHANGED |
| the `None` sentinel contract | `S.is_isotropic` | the predicate the `None` encoded |

⚠ Two `@pytest.mark.sentinel` rows sit on retiring call sites
(`test_scattering_operator.py:239`, `:324`) — **a retired sentinel is a lost
capability-node canary**; the marker migrates with the rewire. Neither the brief nor
a symbol grep flags it.

### L78 §R2 — POST-carve (2026-09-06; the carve landed uncommitted on `f75a9e59` as option B)

Deliverables: the gate module **45 → 86 rows** (+`cart2d_gs` arm, +`TestTheReturnedTrace`,
+`TestTheGaussSeidelArmPosesItsOwnSplitting`), **new** `tests/numerics/test_fixed_point_step.py`
(8 rows, 0.29 s), 4 new post-carve anchors, the re-keyed battery `scratch/_448/battery_r2.{py,sh}`
(12 arms). `[M]` **94 passed / 52.9 s**, pyright 0.

#### L78k — ⛔ a FINALIZE-SCOPED mutation hook needs a CLOSE, not only an OPEN

L78g's hook opens the window at `power_iteration`'s RETURN. It never closed it, so every
production call LATER IN THE SAME TEST was mutated — and the one test that runs a second
production entry after its `solve_sn` is the cross-route oracle
(`solve_sn_fixed_source`). `[M]` arm P1's first run reported *"POSITIVE CONTROL FAILED …
the fixed-source re-solve's scalar flux differs by 1.849e-01"*: the oracle itself was
mutated, so the ψ leg's red attributed nothing. Closing the window at `_package_solution`'s
ENTRY fixes it (`fired` 359 → 14; the row then reds on the ψ leg with the φ control
passing). ⟹ **name the window's two ends, not one**, and note what caught it: the gate's
OWN positive control, which is the whole reason G3a carries one.

#### L78l — ⛔⛔ a declared blindness must name the RIGHT symbol AND stay inside the convergent regime

I declared the trace gate blind to a wrong reflective LAW because "`apply` and
`reflect_inflow_inplace` both route through `_apply_faces`/`_reflect_trace`". Two errors,
each caught by running an arm:

* `[M]` **`_apply_faces` is NOT shared** — it is the gain's outer lift, which merely lifts
  the trace-only `_reflect_trace` onto the full field. Arm P5a was therefore a second
  gain-route mutation wearing a shared-body label (27 reds, including the trace legs).
* `[M]` **`_reflect_trace × 2` is not a perturbation of a reflective eigenvalue problem, it
  is a different and DIVERGENT one** — all 9 of arm P5c's reds were *"did not fully
  converge"*, i.e. the fixture guard, attributing nothing (my own rule: a mutation that reds
  by RAISING has attributed nothing).
* `[M]` **`× 1.001` is the readable one** and CONFIRMS the blindness: `T-law` and `T-conv`
  **green** on all 6 rows while `G1` reds at **6.84e-04** and `G2` reds on all 3 arms.

⟹ the partition, now measured rather than argued: **`G1`/`G2` catch a wrong LAW; the trace
class catches wrong WIRING** (a gain on the wrong operand, `M` and gains disagreeing, the
moment/angular end mismatch, the coupled seed). ⟹ and the transferable rule: **a mutation
of a physical COEFFICIENT must be sized to keep the solve convergent** — otherwise the
fixture's own convergence guard fires first and every red is unattributable.

#### L78m — ⭐ a defect with TWO ENDS needs TWO arms, and the arm you write covers the end you were looking at

`[M]` arm P1 (the ℓ≥1 emission zeroed on `TransferOperator._redistribute_ordinates` — #448
itself, post-carve) reddens 17 rows and leaves **`cart2d` and `cart2d_gs` GREEN**: a
windowed driver's gains are `S.on_moment_domain()`, whose body is `_redistribute_moments`.
Arm P1b patches that end and reddens **exactly those two arms** (6 rows). Without it the
claim *"G1 catches #448 on every arm"* was unverified on 2 of 8 — and nothing in the code,
the docstrings or the red count says so; only running the arm does.

#### L78n — ⛔ `dead_references` on an UNCOMMITTED working tree reports graph staleness, and the discriminator is a validated grep

`[M]` post-carve it read **5 dead targets / 12 doc sites** naming the retired symbols. The
graph is stamped `f75a9e59` = HEAD, with BOTH the carve and an archivist's docs pass
uncommitted on top (`nexus` flags the node `stale` itself). An independent census over
`docs/theory/**/*.rst` for `:meth:|:func:|:class:|:attr:` naming any of the five — **with a
positive control** (8 hits for the surviving `add_p0_source`, 0 for the retired names) —
finds **0**. The archivist had already repaired them. ⟹ on a tree whose docs have moved
since the graph build, `dead_references` describes a state that no longer exists; re-run it
after the next `sphinx-build` and settle the interim with a control-validated grep.

#### L78o — ⛔ a solver entry's STRATEGY DEFAULT decides which production branch a whole module ever poses

`[M]` (qa F-1) `solve_sn(..., inner_schedule="jacobi")` is the default, so **0 of 161**
inner solves in the 45-row module ever built a `ScheduledInvertibleOperator` and the
finalize's scheduled-`M` reconstruction arm had no self-consistency witness anywhere — in a
module whose whole subject is the finalize. ⟹ **before claiming a module covers a
production branch, `inspect.signature` the entry and enumerate its strategy DEFAULTS**; an
arm table built from geometries and solvers alone silently pins one value of every other
knob. The repair is one arm (`cart2d_gs`, +6.0 s/pair) plus its own precondition gate
(`inner.implicit` IS a `ScheduledInvertibleOperator`, and the gains differ) — because an arm
that does not actually pose the branch is a duplicate wearing a new id.

#### L78p — ⭐ a pass-through row asserting EQUALITY is often unreddenable; IDENTITY is what gives it teeth

`test_lagged_source_with_no_gains_returns_the_source` asserted `assert_array_equal(out, q)`.
`[M]` **none** of arms A1–A7 could redden it — equality is satisfied by any body that
allocates a copy. Adding `assert out is q` (the no-gain path is a pure pass-through) made it
reddenable, and arm A8 (`+ 0.0 * p`) reddens it ALONE. ⟹ when a row's subject is "nothing
happened", the assertion has to be about the OBJECT, not its value — and the way you find
out is that no arm in the battery touches it.

#### L78q — the R2 battery's measured partition (the anti-#20 ledger, post-carve)

| arm | reds | ONLY |
|---|---:|---|
| honest | 0 | 94 green |
| P0 positive control (finalize skips `M⁻¹`) | 54 | broad |
| P1 ℓ≥1 zeroed, ANGULAR end | 17 | not the 2 windowed arms |
| P1b ℓ≥1 zeroed, MOMENT end | 6 | **the 2 windowed arms only** |
| P2 one gain (`B`) dropped | 21 | the 3 `B ≠ 0` arms + 3 primitives rows |
| P3 windowed arm gets ANGULAR gains | 22 | the 2 windowed arms — **by TYPED REFUSAL** |
| P4 coupled q½ seed → `None` | 18 | the 2 carrying arms — **by RAISING a production guard** |
| P4b the same seed × 1.5 | 10 | the 2 carrying arms, by VALUE |
| P5b `SNBoundaryOperator.apply` × 2 | 18 | `slab_refl` + `cart2d` (not `cart2d_gs` — masked gain) |
| P5d `_reflect_trace` × 1.001 | 12 | `G1`+`G2` only; trace legs GREEN (L78l) |
| P6 the G-S `implicit` un-split | 6 | **`cart2d_gs` only** |
| A1–A8 (primitives) | 0–3 | every one of the 8 rows has a catcher; A5/A7/A8 redden exactly one each |
