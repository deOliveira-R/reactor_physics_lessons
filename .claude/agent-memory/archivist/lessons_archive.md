# Archivist — Lessons

Read at the START of every invocation. Behavioral corrections only:
"what documentation/Sphinx/knowledge-architecture mistake did I make,
and what discipline did it teach?" The mechanical HOW lives in
`AGENT.md` ("Build-Gating & Cross-Ref Reality", "Close-Out Narrative
Arc") and in the preloaded skills (`vv-principles`, `algebra-of-record`,
`nexus-verification`). This file holds the PROCESS lessons those don't.
Campaign play-by-play (commit hashes, codenames) is retired — a lesson
here stands WITHOUT knowing what the campaign was.

The cross-cutting standard behind most of these: **a doc page is not
done when the prose reads well — it is done when every cross-ref
resolves against the LIVE tree, every claim's V&V level matches the
skill verbatim, every retired symbol leaves no dangling reference, and
the build's WARNING/ERROR/CRITICAL set is unchanged from the pre-edit
`-E` baseline.** Each lesson is one face of that.

---

## L-001 — Verify every claim against the LIVE code, NOT the quoted/docstring prose

→ **The standing directive now lives in AGENT.md Quality Checklist item 6**
(read the live source before citing any convention/shape/decision/result;
brief, docstring, and verdict memo can ALL be stale). The three war-story
faces below are kept for forensic value — they show HOW each surface lies.

The single most recurring trap. A task brief quotes "current stale
text", or a docstring describes a return shape, or a verdict memo
states a recommendation — and ALL THREE can be wrong relative to what
shipped. Three faces:

- **Brief quotes "stale text" that was already fixed.** The user's
  snapshot may pre-date a landed fix. Read the file (or `git show
  <prior-commit>:<path>`) before editing; if the current text already
  matches the desired "after", report "already fixed" rather than
  re-introducing the stale wording.
- **A docstring lies about a convention/shape.** A physics convention
  in a docstring (e.g. an index `r_i` vs `r_j`), or a return layout
  `(N, nx, 1, 1)` claimed while the array is `(N, ng, nx, ny)`, is
  ground-truth ONLY in the live code body. Read the array build / the
  consuming code; never transcribe the docstring's claim into a theory
  page.
- **A verdict memo records the RECOMMENDATION, not the OUTCOME.** An
  elegance/qa verdict's "BLOCKING NIT: drop X" may have been overruled
  — the shipped code took the alternative ("keep + strengthen test").
  Read the CODE to learn which resolution won.
- **A retirement-SHIM docstring freezes a claim that a LATER refinement
  RE-PERMITTED in the canonical forms.** A deprecation shim
  (`sn/sources.py`) carries a docstring stating "the cross-class `iso +
  aniso` dunder is RETIRED, use `from_isotropic` + within-class add" —
  and a brief built on the shim faithfully repeats it. But the shim is
  FROZEN at the commit that minted it; the canonical L2 forms it
  re-exports (`transport/source_sinks/{scalar,angular}_source_sink.py`)
  EVOLVED PAST it: the refined Issue #207 principle (recorded later)
  RE-PERMITTED the cross-class dunder via canonical subspace-containment
  injection (`iso → 1 ⊗ iso`, returning the larger type, commutative).
  Both `__add__` bodies + `__radd__` are live and wired; the module
  docstring (the algebra-of-record) says "PRESERVED". Following the
  brief's "retired" premise would have wrongly PAST-TENSED an accurate
  doc section (the "load-bearing dunder" §) — the live code MATCHED the
  doc, only the names were stale. RULE: when a brief's behavioral
  premise is sourced from a SHIM docstring, verify it against the
  CANONICAL form the shim re-exports (read both the `__add__` body AND
  the canonical module docstring), never against the shim — the shim's
  job is to redirect, and its prose is a snapshot that the canonical
  layer can overrule. (Worked: `sn.sources` retirement, branch
  `refactor/operator-inverse-algebra` — corrected the doc to the live
  subspace-containment narrative, not the shim's "retired" claim.)
- **The brief's OWN discriminator can be over-broad — it names a
  concept, but the LIVE code applies that concept at MULTIPLE layers
  with DIFFERENT types.** A "type-confinement" sweep brief (e.g.
  "confine `TimedFullField` to the driver; the operator apply contract /
  the **solve boundary** / the composite the operators speak → now
  `FullField`") gives a discriminator that is a STARTING HEURISTIC, not
  a per-ref rule. The phrase "solve boundary" named the *within-group
  operator* solve (W-C confined it → `FullField`) — but the SAME page
  also documents the *public* `solve_sn_fixed_source(external_source:
  …)` source argument, whose live signature is STILL
  `np.ndarray | TimedFullField` (the driver-iterate-compatible composite
  source). Applying the discriminator blindly would have wrongly flipped
  an entire accurate section. Resolve EACH ref by reading the LIVE
  signature of the exact symbol it describes: the within-group
  `evaluate_residual` RETURNS `FullField` (UPDATE the doc's stale
  `TimedFullField(bulk=AngularResidual,…)`); the public
  `_build_fixed_source_rhs` RETURNS `TimedFullField` (KEEP). One brief
  phrase, two live boundaries, opposite verdicts. The operator-contract
  refs ("`X.apply` acting on …", the `@overload` surface, `B.apply`
  operates on …) flip to `FullField`; the driver-iterate refs (the
  composite rhs the SI/Krylov inner BUILDS, the `to_flat`/`from_flat`
  GMRES round-trip, `TimedFullField.zeros` initial-guess, the class-gate
  arithmetic, "Layer-4 never sees … carriers") stay `TimedFullField`.
  When a gate EXERCISES an operator on `TimedFullField` via MRO (the
  G-adjoint reciprocity gate), that is the inheritance path, NOT the
  declared contract — the doc states the CONTRACT (`FullField`); add a
  one-clause "the driver's iterate reaches it via MRO" so the KEEP-side
  reader isn't confused.

- **REPRODUCING a cited numerical result during a rewrite routinely
  surfaces a flatly-WRONG pre-existing worked example — fix it (Cardinal
  Rule 1 is supreme), don't transcribe around it.** When the brief gives
  you numbers to verify (here: the n2n double-count moves k by 0.43), run
  the LIVE derivation to confirm them — and while you have the repro
  harness open, sanity-check the page's OTHER worked numbers in the same
  section. A convention-fix brief surfaced a 2-group worked example whose
  `M₂₁=2.0833`/`M₂₂=3.666` gave trace 3.875 while the page then claimed
  k=1.875 three lines later — internally self-contradictory. The live
  `M=A⁻¹F` is rank-1 `[[0.2083,1.6667],[0.2083,1.6667]]` (trace=1.875=k,
  the OTHER eigenvalue is exactly 0 because `F=χ⊗νΣf` is a rank-1 dyad).
  The page's `two-group-M` formula carried a spurious `+ν₂Σf₂/Σr₂` term
  (assumed fission emission in group 2, but `χ₂=0`). This is a teaching
  doc — a self-contradictory worked example is a Cardinal-Rule-1 defect.
  Fix the formula + the worked numbers + tie the result to the trace,
  KEEP the verifies-target labels, and FLAG the scope-expansion in the
  return (the fix went beyond the brief's n2n convention scope). The
  CORRECT final value (k=1.875) was always right — only the intermediates
  were wrong — so this is an arithmetic CORRECTION, not a falsification:
  no tombstone, just a self-consistent rewrite.
- **A deleted class's doc blast radius is the WHOLE `docs/` tree, not the
  brief's named page — grep it and repoint the API pages too.** A brief
  scoped to `docs/theory/X.rst` after a class deletion is the FLOOR; the
  retirement audit's doc search (`grep -rn "<DeletedClass>" docs/`, minus
  `_build`) is the real blast radius. Here #276 deleted `HomogeneousSolver`
  and the brief named only the theory page — but `docs/api/homogeneous.rst`
  AND `docs/api/numerics.rst` BOTH cited the dead class AND carried the
  SAME retired n2n-in-both-matrices convention (the API page is a parallel
  surface that goes stale identically). These render plain-text (no `-W`
  warning, L-002), so the grep gate is the only catch. Repoint the dead
  `:class:` to the live `:func:`, rewrite the parallel convention error,
  and remove the dead class from any "reference implementations of
  protocol P" list (the refounded solver may no longer satisfy P — here
  the direct function is NOT an `EigenvalueSolver`). FLAG (don't silently
  fix) deeper adjacent staleness in those pages outside the deletion's
  radius (here `moc.solver.MOCSolver` is a stale module path — live class
  is `moc.core.MOCSolver` — pre-existing, not #276's; reported, not fixed).
- **The brief's RATIONALE can be subtly wrong, not just its facts — verify the
  ARGUMENT against the math and CORRECT it with a qualifying note, never
  transcribe.** A brief may give a CORRECT conclusion via a FLAWED reason:
  "0-D homogeneous uses the direct engine because an iterative one would be
  dominance-ratio-fragile at the 1e-12 gate" is true IN GENERAL but FALSE for
  THIS problem — the homogeneous `F=χ⊗νΣf` is RANK-1, so `A⁻¹F` has a single
  nonzero eigenvalue (dominance ratio 0) and power iteration would converge in
  ONE step here. The honest doc KEEPS the general argument AND adds a
  `.. note::` recording the rank-1 subtlety (one-step is a fragile consequence
  of F's rank-1 structure, not a guarantee; the direct engine is exact for ANY
  F). Transcribing the reason verbatim would mint a Cardinal-Rule-1-wrong
  teaching claim. The TEST file usually already encodes the subtlety (here
  `TestGeneralizedEigenproblem`'s docstring states F is rank-1 with a single
  nonzero eigenvalue) — read it. Same family: a brief's SIGNATURE can be wrong
  (`power_iteration(A,F)` → really `power_iteration(solver: EigenvalueSolver)`;
  it sees a Protocol boundary, the dense `(A,F)` is never formed — only
  `direct_eigenvalue`/`rqi` take dense `(A,F)`). Read the live `def` first.
- **A MECHANICAL vocabulary-translation can restate a substantively-FALSE
  claim in fresh words — verify the CLAIM'S TRUTH, not just its vocabulary,
  before re-spelling it.** A retirement pass that swaps a retired term for its
  successor (`CAP_APPLY_TRANSPOSE`→`is_adjointable`, `MissingCapability`→
  `MissingAdjoint`) FEELS purely lexical and scoped — but the underlying CLAIM
  may be stale for an ORTHOGONAL reason, and a faithful vocabulary-swap then
  mints a NEW false claim in cleaner language (WORSE than the old dead-ref: it
  reads authoritative). Worked (frozenset-retirement W3): the `operator_algebra`
  G-adjoint section asserted "S/F are not adjointable → the full-loss `.H` is
  unreachable/raises". Translating `CAP_APPLY_TRANSPOSE`→`is_adjointable` I first
  restated it as "S/F are not adjointable" — but LIVE `ScatteringOperator`/
  `FissionOperator` `.is_adjointable` return **True** (they gained `apply_transpose`
  via #112/#118/#276, orthogonal to the frozenset). The whole reachability
  argument was pre-existing-stale. Cardinal Rule 1 forbids shipping (or fresh-
  minting) the false claim, so I corrected it to the verified truth (metric lives
  on the shared `full_field_space`, every leaf carries it → no composite is
  metric-blind; the within-group loss never fuses S/F/B — `_within_group_triple`
  returns `(L+C, S, B)`) + an L-007 supersession `.. note::`, and FLAGGED the
  scope-expansion. RULE: any `:class:`/CAP_* symbol you translate, grep the LIVE
  class for the property it asserts (`grep 'def is_adjointable' … | show return`)
  — the vocabulary swap is safe only after the CLAIM is re-verified. A clean
  `-W` build never catches this (plain-text refs + a false prose claim both
  build green).
- **A carve's OWN sibling changes (adjoint work, pre-inversion redesign) can
  have staled the SAME sections the carve's doc-pass touches — fix the carve's
  dead refs, but FLAG (don't silently deep-rewrite) the sibling-staleness.** The
  `discrete_ordinates` "Capability requirements" posing section carried BOTH
  step-6 dead refs (`CAP_APPLY`/`MissingCapability` — fix required) AND step-3
  pre-inversion staleness (the `inverter`-hook narrative — `SourceIteration` now
  takes a pre-inverted `A_inv`; `KrylovAcceleration.inverter`→`preconditioner`).
  I rewrote the bullets to the verified live contract (SI→`A_inv`+`TypeError`;
  `KEigenvalue`→`A.is_invertible`+`NotInvertible`) AND added a `.. note::`
  flagging the surrounding `inverter` narrative as step-3-stale + deferred. The
  dead-ref fix is in-scope; the deep behavioral rewrite is a separate task.
- **In a CLAIM-CLASSIFICATION correction sweep (classify each repeated site
  A/B/C, correct only the false class), the SAME phrase can be TRUE at one site
  and FALSE at another — the discriminator is a LIVE config detail, not the
  prose.** Worked (#280 Phase 2.5b, `discrete_ordinates.rst` + `tests/sn/`): the
  recurring claim "the cylinder α-dome telescopes the seed away / was already
  exact" is a FALSE mis-attribution for a **product** quadrature (the starting
  direction coincides with the first-swept ordinate #229, so
  `c_in[m0]=(1−τ)/τ ≠ 0` is a LIVE self-coupling on the m0 diagonal; the cold
  `(L+C).solve` was seed-lagged err≈0.57 until the direct-seed fold
  `c_out→c_out−c_in`) yet genuinely TRUE for a **level-symmetric** one
  (`c_in[m0]=0` at raw τ=1 — a DEAD first-ordinate weight annihilating the seed
  at source, NOT telescoping). So a "cylinder was already exact" docstring is
  only classifiable after grepping the site's LIVE quadrature: a
  `Quadrature.level_symmetric(4)` fixture makes it TRUE (LEAVE — e.g.
  `test_282_direct_seed_fixed_point._operator`), a `product` fixture makes it
  FALSE (CORRECT). Second discriminator, SAME word, different class: "α-dome
  telescopes" splits (A) seed-absorption-of-the-SOLVE (false, level-symmetric-
  only) vs (B) weight-summed-scalar-gate-blindness (anti-pattern #8, TRUE, LEAVE
  — the `Σ_n w_n(α_{n+½}−α_{n−½})=0` identity) by WHAT it claims telescopes —
  the seed's effect on the fixed point (A) vs a per-ordinate error's effect on a
  scalar residual (B). Disambiguate by the OBJECT, not the verb. Also: the fix's
  new changelog bullet + the reframed load-bearing note reference #280 Phase
  2.5b; leave the main agent's already-corrected model sites untouched (grep the
  issue tag to find them first).

How to apply: before citing any convention, shape, or design decision,
confirm it against the live source this session. Cross-refs that render
plain-text carry NO warning (see L-002), so a dead/stale `:func:` is a
Cardinal-Rule-1 correctness bug `-W` will never catch — grep the symbol
across the WHOLE `docs/` tree, not just the brief's named page.

---

## L-002 — Unresolvable code-xrefs render as PLAIN TEXT with no warning; this repo is NOT nitpicky

→ **The standing directive now lives in AGENT.md Quality Checklist item 3**
(grep-gate cross-refs; `-W` is blind to a dead code-xref). The detail below —
which ref classes DO warn, and the not-member-`automodule`'d page convention —
is kept as the recall companion for when a cross-ref edit gets subtle.

`-W` does NOT catch a dead `:func:`/`:class:`/`:meth:`/`:attr:` or a
stale alias-xref — they silently render as plain text. The acceptance
gate (count-unchanged from the `-E` baseline) only proves you added no
NEW warning; it is BLIND to staleness.

- After any carve that DELETES or RENAMES a symbol, `grep -rn "<symbol>"
  docs/` and repoint every hit on correctness grounds.
- Distinguish what DOES warn: undefined `[Key]_` citations warn (even
  non-nitpicky); intra-doc dangling `:ref:` warns under `-W`; cross-doc
  dangling `:ref:` renders plain-text. A new `:ref:` to a not-yet-
  existing section MUST create the labelled section in the SAME edit.
- **FORWARD-ref corollary: a NOT-YET-BUILT deliverable's symbol is a
  code LITERAL, never a `:class:`/`:meth:` cross-ref.** When a docs pass
  documents a DEFERRED seam (a changelog row for an open issue, a
  "future hook" bullet — e.g. #280's `SweepOperator.apply_transpose`,
  the `A.H.inverse()` swap law), the planned method/class does NOT exist
  yet. A `:meth:`-ref to it renders plain-text with NO `-W` warning
  (L-002 forward-facing), so the build is BLIND — but it is a
  Cardinal-Rule-1 stale ref regardless (points at a non-existent
  symbol). VERIFY absence before choosing the spelling: `hasattr(Cls,
  "planned_method")` / `.venv/bin/python -c "import ...; hasattr(...)"`
  → if False, write ``planned.method`` as a literal (honest: names the
  deliverable without claiming it links). Same `hasattr` gate distinguishes
  a LANDED seam (flip to a live `:meth:`, per L-007's landed-seam bullet)
  from a still-deferred one (literal). This is the FORWARD twin of the
  RETIRED-symbol rule above (that one repoints dead refs; this one
  refuses to mint premature ones).
- Packages that are not member-`automodule`'d (transport.*, the
  operator/scheme/numerics leaves on several pages) render their
  `:class:`/`:meth:` as plain text BY PAGE CONVENTION. Match the page —
  do NOT half-surface 1–2 leaves by adding an `automodule` while the
  rest of the package stays plain. The `:class:` ref staying plain-text
  is NOT a regression to fix — it is the convention; repoint a dead one
  to the LIVE module path (still plain-text, but now correct) and move on.
- **`:noindex:`-automodule'd is xref-invisible too — a WHOLE package can
  be plain-text page-wide even though it IS `automodule`'d.** An api page
  that `automodule`s every module of a package with `:noindex:`
  (`docs/api/diffusion_1d.rst` does this for all of `orpheus.diffusion.*`)
  registers NO cross-reference targets, so EVERY `:class:`/`:func:` to
  that package renders plain-text everywhere, while sibling packages
  automodule'd WITHOUT `:noindex:` (transport.method, numerics.eigenvalue,
  data.macro_xs.mixture) link normally. Diagnose by HTML link-audit
  (`grep 'href="[^"]*Symbol"' built.html` — empty ⇒ plain-text) + read the
  api page's automodule options; the module appearing in Sphinx's
  "highlighting module code" list means viewcode processed it, NOT that it
  has an xref target. This is NOT a defect to fix by editing the api page
  (often out of scope / forbidden) — keep the semantically-correct
  `:class:` markup (greppable, import-verified, auto-links if `:noindex:`
  is ever lifted), and FLAG the package-wide `:noindex:` as a candidate
  infra fix. The grep/import gate (symbol EXISTS) is the real cross-ref
  check; the link is governed by the untouchable api page. (Worked: #290
  P8 — all diffusion `:class:` refs plain-text via the api page's
  `:noindex:`; all 31 cited symbols import-resolved regardless.)
- **`automodule`-readiness is a MULTI-gate test; the 0-`:label:` check is
  necessary but NOT sufficient.** A leaf with NO `.. math:: :label:` is
  safe from the *duplicate-label* collision — but automodule RENDERS the
  whole docstring under the project's strict config, so it ALSO trips on
  any of: (a) a `:pydata:` (or other non-registered) role → `ERROR:
  Unknown interpreted text role`; (b) a section-underline shorter than its
  title inside the docstring → `WARNING: Title underline too short`; (c) a
  malformed field-list / inline-literal → docutils WARNINGs; (d) **a
  member attribute whose NAME collides globally** (a class `ng`/`n`/`name`
  attr surfaced by automodule makes EVERY pre-existing bare `:attr:\`ng\``
  across OTHER pages ambiguous → `more than one target for cross-reference`
  WARNING ×N, on pages you never touched); (e) **a malformed inline-role
  in ANY ONE rendered docstring** — the classic is a closing role-backtick
  immediately followed by a word char with no whitespace/punctuation
  (``:class:`X`s``) → `WARNING: Inline interpreted text ... start-string
  without end-string`. (e) fires even on a module that passes (a)–(d)
  cleanly (0 `:label:`, 0 `:pydata:`): ONE bad docstring line in ONE method
  blocks the whole `automodule`. The `-E -W` build is the only
  way to see (a)–(e) — a plain build with cached env MASKS them (one
  session: plain build EXIT 0 while `-E` showed 4 ERRORs + a 7-page `ng`
  cascade). A docstring WITH `:label:` must be cross-referenced in prose
  instead (automodule re-registers the label → duplicate-label / "equation
  not found").
- **Signature (e) intersects the "report-don't-edit-.py" constraint — an
  (e)-blocked module is automodule-UNREADY by the SAME rule as (a)–(d):
  cross-reference it in prose (plain-text, consistent with the
  un-automodule'd family) and REPORT the one-line docstring fix that
  unblocks the autodoc.** The plain-text refs are NOT a defect — they MATCH
  the page convention (grep the built HTML: the sibling family on the SAME
  page renders plain-text too). The "surface this type" intent is still met
  for refs pointing at an ALREADY-automodule'd symbol (those link), plus a
  prose bullet-list of the new types with their theory `:ref:` + a
  `.. note::` recording the exact docstring fix and the autodoc block to
  add post-fix. Verify with TWO gates `-W` is blind to: the grep gate
  (every cited symbol exists in live code) AND the rendered-HTML link audit
  (automodule'd→`<a>` link, un-automodule'd→plain `<code>` by convention).
  (Worked: P5 condensation #274 — `energy_grid.py` was (a)–(d)-clean but
  `GroupCondensation.from_grids` had ``:class:`EnergyGrid`s`` → reverted
  the `automodule::` to a prose bullet-list+ref+fix-note, build 3→0
  warnings; `Mixture.condense` refs link (Mixture IS automodule'd),
  `EnergyGrid`/`OverlapBasis` render plain-text like their `numerics.basis`
  siblings.) **SEQUEL — P5.5 reshape (same module family, one reshape
  later):** the (e) blocker `GroupCondensation.from_grids` was DELETED by
  the reshape, so `energy_grid.py` became automodule-ready — I surfaced it
  (anchors for `EnergyGrid.as_measure`/`.as_basis`/`.overlap_to` now LIVE,
  the L-002-deferred fix CLOSED). But a NEW blocker surfaced on the SIBLING
  `wims.py` — signature **(c)**: a module docstring's plain-text
  2-space-indented numbered list ("  1. …  2. …") after an unindented para
  → `ERROR: Unexpected indentation` + `WARNING: Block quote ends without a
  blank line`. Same resolution: automodule the clean pair
  (`energy_grid` + `ornl`), cross-ref `wims` in prose + `.. note::` the
  re-flow fix. LESSON: a retirement that deletes one automodule blocker can
  UNBLOCK a module — re-test automodule-readiness on the reshaped tree
  (the deferred-fix note may now be actionable); but a sibling in the SAME
  package can carry a DIFFERENT blocker class, so `-E -W`-build EACH
  automodule you add, never assume the package is uniformly ready.
- **The scoped resolution when a cohesive cluster is automodule-UNREADY:
  automodule ONLY the clean module(s), cross-reference the rest to the
  theory page** (the `api/numerics.rst` operator-family pattern: it
  automodules `field` but cross-refs `operator` because `operator.py` has
  `:label:` docstrings). Surfacing the whole cluster (fixing `:pydata:`,
  the `ng` collision via `:noindex:`, underlines) is a SEPARATE
  architectural docs task — DEFER it, do not block the carve-doc on it.
  (Worked: the dyad-carve task — `reaction_rate_functional.py` was clean
  (0 `:pydata:`, 0 `:label:`) → automodule'd to render both functional
  classes; `fission`/`scattering`/`multiplication` used `:pydata:` + the
  `ScatteringOperator.ng` global collision → kept as theory cross-refs.)

How to apply: treat the grep gate (symbol exists / page-convention
matched) as the real cross-ref check; the warning count proves only
"added nothing new". Before adding ANY `automodule`, `-E -W` build it in
isolation and watch for the (a)–(d) signatures above — especially the
cross-page `ng`-style cascade, which fires on pages you did not edit.
(The forced-`-E` build, the three-severity grep, and the venv/worktree
facts live in AGENT.md — do not re-derive them.)

---

## L-003 — GREP the V&V matrix for an eq-label BEFORE renaming or removing it

An equation `:label:` may be a `@pytest.mark.verifies(...)` TARGET. If
it is, the test's oracle points at that exact name — renaming or
deleting it breaks the verification edge (and bumps a "no matching
equation node" line). The recurring mistake is rewriting a stale
equation's BODY and changing its LABEL in one motion, orphaning a
verifies-target.

- For a stale equation that IS a verifies-target: KEEP the label name,
  rewrite only the BODY (the claim is unchanged). Split a busy
  derivation into a label-preserving primary + NEW sub-labels for the
  decomposed steps.
- A label-rename ripples: an in-page symbol rename (`s_a → c_a`) flows
  to every `:eq:` and prose site that referenced it — a whole-page
  sweep, not a one-line edit.
- Section-label vs equation-label are different namespaces. `.. math::
  :label: X` → `id="equation-X"`, resolved by `:eq:`. A `.. _X:` anchor
  → `id="X"`, resolved by `:ref:`. When you need a section anchor and
  an equation sharing a name, suffix the section `X-section`. A
  `:label:` inside a CODE docstring is rendered by autodoc but is NOT a
  global `:eq:` target — cite the owning `:class:` and inline the math.
  COROLLARY (caught here): do NOT write `:ref:`X`` where `X` is an
  EQUATION label — it renders plain-text cross-doc / warns intra-doc.
  Point at the equation with `:eq:`X`` and describe the surrounding
  prose ("the note under the production matrix :eq:`fission-matrix`").
- **When an ALGORITHM is replaced (not just an equation rewritten), a
  retired-STEP verifies-target is usually KEPT-AND-REPOINTED to a
  conceptual survivor of the NEW algorithm — not retired.** The
  recurring trap is reflexively retiring a power-iteration step label
  (`fission-source`/`fixed-source-solve`/`keff-update`) because "the
  iteration is gone", which orphans 4–5 test edges and forces test-side
  edits. Instead, ask whether the CONCEPT survives the new method: the
  per-iteration fission source `Q_f=(χ/k)νΣf·φ` → the single dyad
  application `Fφ`; the fixed-source solve `Aφ=Q_f` → the loss-matrix
  solve `M=A⁻¹F`; the production/absorption ratio `k=prod/absn` → the
  eigenvalue extraction `k=λ_max(M)`. Each repointed equation is a REAL
  step of the direct method, the `:label:` name is preserved, and the
  reconciliation table reports "kept-and-repointed → NO test-side edit".
  A `.. note::` under each repointed label states what it "historically
  named" and what direct analogue it now carries (preserves the WHY per
  L-007 without a tombstone — the equation evolved, it wasn't falsified).
  Only RETIRE a step label when the concept genuinely has no survivor
  (here `convergence-rate`: a direct dense eig has no iteration whose
  rate a dominance ratio governs — and it was documented-only, NO test
  edge, so retiring just drops one auto-regen row). (Worked: #276
  homogeneous refounding — power iteration → direct `λ_max(A⁻¹F)`;
  4 step labels kept-and-repointed, `convergence-rate` retired.)
- **PHANTOM verifies-target (the INVERSE of an orphan): a `verifies("X")`
  marker whose `X` has NO `.. math:: :label:` anywhere under `docs/`.**
  The audit flags it (`tests/_harness/audit.py` `_phantom_verifies`) — a
  silently-dropped V&V edge. Fix = EITHER add `:label: X` to the equation
  the test verifies (only if that equation is UNLABELED) OR repoint the
  marker to the real label the equation ALREADY carries. Decide by reading
  what each test verifies + grepping for the equation's existing label:
  when the topic's equations are ALL already labeled (here the LD slab is
  `ld-ubld-d1-reduction` (operator) + `ld-ubld-slope-angular-reduction`
  (the ERR-061 reduction)), the "add a label" branch is impossible (one
  `.. math::` = one label) → REPOINT, per-test, to the accurate label
  (convergence test → the operator; diffusion-limit/frame tests → the
  reduction). BONUS: if the repoint targets were FORMER ORPHANS (labeled,
  0 tests), the repoint kills the phantom AND covers the orphans in one
  move — a net V&V-hygiene win that VALIDATES repoint over add-a-label or
  delete-the-marker. Watch for a SIBLING phantom synonym (here
  `ld-cartesian-1d`, the un-homed 1-D umbrella parallel to the real
  `ld-cartesian-2d`) co-declared on the same tests: if it has no free
  equation home (the natural one is taken by a verifies-target you can't
  rename per this lesson), it is OUT of a single-label task's scope —
  FLAG it, don't half-fix. (Worked: stencil-assembly 2b — `ld-slab`
  4-marker phantom repointed; `ld-cartesian-1d` flagged.) **SEQUEL
  (Task #55): the flagged `ld-cartesian-1d` was RESOLVED by repoint →
  `ld-ubld-d1-reduction`** (the 1-D LD operator equation ALREADY labels
  the natural home; one eq = one label ⇒ mint impossible ⇒ repoint, and
  dedup the mark that already carried it). The SAME task took the OTHER
  branch for a sibling phantom `inverse-as-operator`: **MINT**, because
  its law (`A.inverse().apply ≡ A.solve`, the #226 keystone) was stated
  in PROSE but UNLABELED. Discriminator: repoint when the law's equation
  is ALREADY labeled; mint (with `.. vv-status: <label> documented` for a
  foundation/structural law, L-004, matching a sibling documented label's
  style) when the law is stated but carries no `.. math:: :label:` yet.
  Verify the mint lands in the matrix's "Documented-only" bucket.

How to apply: grep the generated matrix (`docs/.../matrix.rst` or the
audit output) for every label you intend to touch FIRST; preserve
verifies-targets by name (kept-and-repoint when the concept survives;
retire only documented-only labels with no survivor). Also `grep
':label:'` repo-wide before MINTING a new label — duplicate labels
across files collide (a real warning), and a new label can collide with
a same-named partition predicate already living in another page.

---

## L-004 — Representational/structural eq-labels get a `.. vv-status: <label> documented` DIRECTIVE, not prose

A NEW equation that is a field-typing identity, a governing iteration,
a literature-transcribed definition, or a derivation-decomposition step
is NOT a solver claim. It must be tagged so the V&V matrix files it
under "Documented-only", not flagged as an unverified solver claim. The
recurring mistakes: (a) writing the status as prose instead of the
machine-read DIRECTIVE (a `--strict` audit then regresses); (b) leaving
a label that a NEW test's `verifies(...)` points at as an untagged
orphan.

- Structural/representational label → `.. vv-status: <label>
  documented` with a one-line rationale comment naming the bit-identity
  / foundation gate that pins the verifiable content.
- A label a NEW test `@pytest.mark.verifies(...)` points at is
  `implemented` (code + test, no eigenvalue/flux claim). If that test
  verifies a label that does not exist yet, CREATE the label in the new
  section — never leave a verifies-target orphaned.
- Pure derivation-decomposition labels (the affine-cell sub-steps, the
  facewise-separable tensor identity) sit untagged in a verified
  narrative chain by established page convention — match the page's
  siblings rather than inventing a status.

**Orphan-slice adjudication triage (the V7 backfill task, #231 #10).**
When handed a slice of EXISTING orphan labels to sort into SENTINEL /
WIRE / GAP, three discriminators do almost all the work, in order:

1. **SIBLING-CONSISTENCY is the dominant signal.** On the foundation
   operator-algebra pages every label family already has adjudicated
   members — `apply-distributes`/`solve-does-not-distribute` govern the
   whole apply-solve family; `eigen-standard-form` governs the eigen-*
   posing family; `g-adjoint-definition` the g-adjoint family;
   `wdd-forward-recurrence`/`…-three-terms` the WDD/tensor-network family;
   `green-neumann-series`/`matrix-functor-out` the inverse family. An
   orphan that is a mathematical identity / posing statement / structural
   decomposition / literature-transcribed bound in the SAME section as a
   SENTINELED sibling → SENTINEL, with the same rationale shape. This
   made ~30/31 labels SENTINEL in one batch.
2. **A doc that says its pins are `@pytest.mark.foundation` "no
   verifies() by design" GOVERNS its whole label family toward
   SENTINEL, never WIRE.** foundation tests never carry `verifies()`
   (vv taxonomy) — so a defining identity (e.g. g-adjoint reciprocity,
   pinned only by a foundation reciprocity/oracle suite) is a documented
   sentinel even though a test "exercises" it. The page's own explicit
   V&V framing ("the claim is a software/algebra invariant, anchored to
   the structurally-independent oracle") is the ruling, not your instinct
   to wire the named test. Read the test's `pytestmark`/docstring.
   **BUT `@foundation` ALONE does NOT imply SENTINEL** (correction from
   the REFERENCES-part batch, #231 #10 A5): the algebra-of-record
   reference pages (peierls / peierls_nystrom / trajectory_resolvent /
   fn_method / galerkin_spectral / singular_eigenfunction) pin their
   V_αN / V_cg / V_se SymPy-identity labels with tests marked
   `@foundation` AND `@verifies("<label>")` TOGETHER — and a
   verifies-mark on a foundation test DOES close the orphan
   (**audit-proven**: the sphere `peierls-greens-V-alpha-2` identity is
   "covered", NOT an orphan, though its ONLY gates are `@foundation`+
   `@verifies` in `test_peierls_greens_function_symbolic.py`; harness.rst's
   "foundation tests carry no verifies" is a general EXPECTATION the
   reference pages routinely break). So the real discriminator is the
   SIBLING's decorator, not the `@foundation` tag: (a) foundation-NO-verifies
   pins (operator-algebra oracle/reciprocity suites; V_se-cyl.N SymPy
   gates; the variant-α-core "one foundation test per primitive, no
   verifies" file) ⟹ SENTINEL; (b) foundation-WITH-verifies pins (the
   SymPy V_n derivation gates whose file MIXES `@foundation`+`@verifies`
   on the identity-establishing tests) ⟹ WIRE the orphan into the
   PARALLEL foundation+verifies test (e.g. the slab `T_00^slab=2E_3`
   V_α2_slab tests mirror the sphere V_α2 convention → I minted
   `@verifies("peierls-greens-slab-V-alpha-2")` on the closed-form +
   numerical + overall-pass tests, leaving the substitution-algebra tests
   alone since a 2E_3 sign flip doesn't red them).
3. **WIRE (or DEFERRED-WIRE) only when a NON-foundation test tightly
   pins THAT exact equation with sign-flip teeth.** The discriminator:
   `tests/conftest.py` records an untagged test as `level=None` (NOT
   foundation) — so a `np.testing.assert_array_equal` law-test
   (`test_sum_law`/`test_product_law`/`test_scaled_law` pinning
   `[A+B]=[A]+[B]` etc.) is wire-eligible, unlike an `@foundation`
   reciprocity gate. If that catcher lives in a test file OWNED by a
   concurrent agent (do-not-edit list), report it as DEFERRED WIRING
   with exact node ids — do NOT sentinel over it (that papers over a
   genuine gate; harness normative §vv-status forbids it). The task's
   own do-not-edit list ANTICIPATES the wire targets it names.
- **Placement is same-FILE-only (the audit enforces file, not
  position).** BEFORE-the-math `.. (vv-status rationale) …` +
  `.. vv-status: <label> documented` (blank line, then `.. math::`) is
  robust and audit-valid in ANY file: anchor the Edit on the unique
  `.. math::` + `:label: <label>` two-line string and prepend. Match a
  bullet-indented math block's indent (2-space comment, 5-space
  continuation). Self-check = a Python pass asserting every
  `vv-status: X` has a same-file `:label: X` (a typo'd sentinel is a
  hard exit-2 abort); then the permitted single end-of-run audit
  (exit 0 ⇒ no typo/misplacement) — IGNORE foreign orphans/violations
  from concurrent sibling batches.
- **De-freeze a live suite-total** ("**12 tests**:") by dropping ONLY
  the drifting total; KEEP the designed case-list breakdown ("(5) … (5)
  … (2)") — those are structural, not drifting. A foundation suite is
  not an equation-matrix row, so a `:doc:`/…/matrix`` pointer does not
  literally fit; the honest move is drop-total-keep-structure.
- **The "don't bulk-sentinel row-sum / T-matrix / escape-probability"
  caution resolves by the GATE'S MARK CLASS, not the topic.** A brief may
  flag those as WIRE-likely; check each gate individually. On the peierls
  pages the infinite-medium row-sum `Σ K_ij Σt=Σt` and the
  finite-cell-deficit are derivation-context identities whose TESTED
  realisation is a DIFFERENT label (the finite-cell
  `peierls-vacuum-bc-row-sum-gate`, already wired) ⟹ SENTINEL; the
  T-matrix rank-1 gates are `@foundation`-no-verifies ⟹ SENTINEL (matching
  the sphere sibling `peierls-specular-T-matrix`); ONLY escape-probability
  had a real l0-WITH-`verifies` gate (`TestSlabPescClosedForm`,
  "Factor-level verification — slab P_esc") ⟹ WIRE. Corollary: a test
  NAMED for a SIBLING label can pin the ORPHAN's distinguishing content —
  `test_g_prefactor_is_4_over_pi` (verifies `peierls-cyl-3d-mode-formula`,
  the P_esc twin) asserts the 4/π G-prefactor that is the ONLY thing
  distinguishing the orphan `peierls-cyl-3d-gbc-mode-formula` ⟹ add the
  orphan as a SECOND `verifies` target on that same test. Net for a
  continuous-reference-derivation slice: ~94 % SENTINEL is the CORRECT
  ratio (50/53 here), and 0 GAPs is legitimate — every label is either a
  governing/definitional/literature-identity (SENTINEL case a) or has a
  real verifies-eligible gate (WIRE); a derivation page rarely hides a
  load-bearing contract with NO test anywhere.

How to apply: for every eq-label you add, classify it (solver-claim /
representational / verifies-target) and apply the matching status
discipline. This is the V&V-vocabulary-curation duty (Directive 5) at
the label level — the matrix is the audit's source of truth.

---

## L-005 — A stub→rich expansion reads its sources in a fixed priority order; the docstrings are the prose SEED

Expanding a `.. todo:: Archivist` stub (the algebra-of-record handoff)
into rich narrative has a load-bearing source-reading order, and the
recurring mistake is writing from the brief alone:

1. **The close-out / verdict memo** — carries the bug ("confirmed live
   pre-fix"), the architecture-settled framing, the named tests, the
   verification numbers, the honest-interim state.
2. **The production docstrings** (the scheme/operator/class bodies, the
   `supports()` predicate, the SymPy `derive_*` docstrings) — the
   VERBATIM prose seed: the numerical-PDE statement, the contrast notes,
   the per-case table, the lit cites. These are the algebra-of-record.
3. **The test files** — the verification subsection: the named gates,
   what each asserts, the bit-identity-vs-principled-equivalence split.
4. **The SymPy module** (when present) — NEVER expand a stub without
   reading it; the narrative narrates it, does not compete with it. If
   you find an algebra error, return a DISPATCH_REQUEST for the
   method-implementer — never edit the SymPy yourself.

How to apply: read memo → docstrings → tests → SymPy, in that order,
before drafting. The honest scope (what shipped vs what is OWED to a
follow-on slice) comes from the memo's interim-state note — preserve it
verbatim; do not over-claim a wired-but-not-yet-iterating capability.

---

## L-006 — Cross-document duplicate citations are a real warning class; resolve, do not redefine

A `.. [Key]` bib entry duplicated across two standalone theory pages
emits a duplicate-citation warning. The recurring mistake is adding a
fresh `.. [Key]` block on a new page for a reference already defined
elsewhere.

- Before citing, `grep '^\.\. \[Key\]'` — if the entry exists on
  another page, cite it cross-doc (resolve), do NOT redefine.
- Where a page's existing convention already dodges the collision by
  citing a reference as PLAIN TEXT in the Literature list-table
  (because the `.. [Key]_` form would cross-doc-collide), MATCH that
  convention — add new literature as plain text too, no new bib entry.
- Pre-existing duplicate-citation warnings (cross-document cite
  collisions) are a known trade-off for standalone pages and do NOT
  need elimination during a close-out — verify the COUNT is unchanged
  pre/post, not that they are gone.
- FLAG (don't silently use) a conflated bib key when you spot one (e.g.
  a key whose title is one paper but whose cited equation content is
  another) — that is a Cardinal-Rule-1 correctness defect for the
  method-implementer/literature-researcher, surfaced from your
  cross-page vantage.

---

## L-007 — A retirement/relocation doc preserves the WHY and tombstones the wrong claim; it never deletes evidence

When an issue closes by FALSIFICATION (the approach cannot work) or a
type/decomposition is retired but its CONCEPT survives, the close-out's
archival value is HIGHER than a success story — it stops future
sessions re-attempting a dead path. The recurring mistakes are
rewriting history and deleting numerical evidence.

- **Preserve the motivation that LED to the investigation** — flip
  tenses ("is expected to" → "was expected to") but keep the logic. A
  future reader asking "why did anyone try this?" must find the answer.
- **Tombstone, don't delete.** When a new finding invalidates a
  published table/claim on the same page, add a `.. note:: **Retraction
  (date, Issue #N).**` immediately above it: (a) what the claim was, (b)
  why it's wrong (one line), (c) forward-pointer to the new section.
  Numerical VALUES stay; the INTERPRETATION gets the tombstone.
- **Retitle to the concept, KEEP the anchor.** When a type is retired
  but the concept survives in new realizations, retitle the section to
  the concept (not the dead type), KEEP the section anchor (cross-doc
  `:ref:`s auto-pick up the new title), and add a prominent succession
  note naming the new realizations. De-role dead module-path
  `:class:`/`:meth:` to literals (grep gate per L-002, not `-W`),
  past-tense the type, repoint present-tense claims to the realizations.
- The full 9-step CLOSED post-mortem arc + the PARTIAL/OPEN variant +
  the multi-issue audit-table pattern live in AGENT.md "Close-Out
  Narrative Arc" — follow it; don't re-derive it here.
- **A surgical RENAME/re-point brief routinely uncovers adjacent
  SUBSTANTIVE staleness — repoint the dead ref (Cardinal Rule 1), but
  FLAG, don't silently rewrite, the surrounding behavioral-claim
  staleness.** A "re-point `X`→`Y` after `X` was retired" pass crosses
  sections that describe a now-superseded ARCHITECTURE (an intermediate
  collapse state, a since-reversed `domain=None` design, a retired
  factory/test still cited). The dead `:class:`/:meth:` ref to `X` MUST
  be repointed to the live `Y` regardless (it renders plain-text, no
  `-W` warning — L-002 — so the grep gate is the real check). But the
  surrounding PROSE staleness (a "thin subclass" narrative #261 fully
  dissolved; a "stays at `domain=None`" claim W-D reversed; a dead
  `:meth:`SNMesh.zeros_*`` / `:file:` test path) is a SEPARATE task: a
  behavioral-claim rewrite needs its own verify-against-live pass and
  often its own issue. Repoint-in-passing is correct; rewrite-in-passing
  risks minting a NEW false live claim (worse than a dead ref to true
  history). The brief's named file/line list is the scope FLOOR (the
  full grep is the blast radius for the RENAME); substantive-narrative
  fixes beyond the rename are the scope CEILING — report them as flagged
  findings, don't smuggle them in. (Worked: #261 CollisionOperator→
  MultiplicationOperator — the §5.7 "thin subclass" §, the
  "C/S/F stay at domain=None" §, and the whole PR-TYPED-3
  IsotropicSource/PerOrdinateSource § were all substantively stale
  beyond the rename; repointed the dead refs, flagged the three §§.)
- **The dedicated behavioral-rewrite follow-up (the task L-007's prior
  bullet hands off to): when a doc describes a FUTURE/DEFERRED seam and
  a commit since LANDED it, verify the SHAPE that shipped — don't just
  flip "deferred"→"done".** The realized change can close the seam via a
  DIFFERENT mechanism than the doc envisioned, and a naive flip mints a
  new false claim. Read the commit message AND the live code; tell the
  honest "envisioned X, shipped stronger/different Y" story. (Worked,
  #271: the "Deferred typing-completion seam" § envisioned giving `S` a
  BULK `V_bulk` domain so `OperatorSum` would REJECT an `S+B` fold;
  W-D actually gave `S` the COMPOSITE `full_field_space` — same instance
  as L/C/B — so the within-group `(L+C)-S` guard VALIDATES, and the
  once-envisioned "bulk-S ≠ trace-B" rejection NO LONGER applies at all.
  A flip-only rewrite would have claimed the rejection tripwire is now
  live — FALSE. The honest rewrite said the seam closed with a different,
  stronger choice and the fold stays gone for an UNRELATED reason (the
  variadic-driver redesign).) Corollary: when a reversal touches an
  adjoint/metric mechanism, separate the conclusion that SURVIVES (here:
  "the metric applies ONCE at the op level via `_AdjointOperator` reading
  the composite domain") from the stale PREMISE that justified it (here:
  "C carries domain=None so the metric propagates from L by
  first-non-None"). Preserve the conclusion, rewrite the premise — and
  re-derive WHY the conclusion still holds (the `_AdjointOperator` reads
  the SUM's domain, never per-leaf, so a leaf now carrying the composite
  domain is no double-application risk). Read the actual wrapper/composer
  source (`operator.py`) to ground the new "why", never reason from the
  old prose.

---

## L-008 — Generated artifacts are NEVER hand-edited; materialize them, report the REAL number

Several Sphinx inputs are generated on every build by a `builder-inited`
hook (the V&V matrix via `generate_matrix`, the capability matrices via
`generate_capability_matrices`, the `docs/_generated/*.rst` includes via
`generate_rst`). The recurring mistakes are hand-editing the rendered
output and estimating a count instead of running the generator.

- A matrix/capability-table drift clears by running the generator (or
  just building) — never hand-edit the `.inc.rst`. Edit the registry-
  side metadata (`capability_rows()`) or the test that drives it.
- When a refactor changes how many rows the matrix auto-regenerates,
  REPORT the real post-regen number (it can differ from a brief's
  estimate — e.g. an auto-regen dropping 67→54 rows, not the est. −5).
  An auto-regen row delta is NOT a warning.
- In a FRESH worktree, missing generated artifacts (`.. plot::`
  FileNotFoundError on `*.h5`; `.. include::` CRITICAL on
  `_generated/`) are ENV gaps, NOT doc defects — materialize them
  (run the converter / generator, both write only gitignored dirs,
  confirm `git check-ignore` first). Do NOT "fix" the docs to route
  around a missing artifact — that corrupts correct documentation.
- Hard-coded test counts in RST go stale; `pytest --collect-only -q`
  for the current number rather than transcribing one.

---

## L-009 — Section-marker ladder is FILE-LOCAL; underline length is in CODE POINTS; reuse the file's existing same-depth marker

Recurring build-breakers from title machinery:

- The `=`/`-`/`~`/`^`/`'`/`"` underline ladder is per-FILE. Scan the
  file's first-appearance markers (grep the single-char underline rows,
  tally by depth) before picking a level, or you get "Inconsistent
  title style: skip from level N to N+2" — often at sections you did
  NOT touch (introducing a marker at a SHALLOWER depth than the file's
  existing one for that level pushes the old sections down a level).
- Underline length is measured in CODE POINTS, not bytes. An em-dash
  `—` or `÷`/`χ` is 1 code point but multiple bytes — size the
  underline with `len(title)` in Python, never `wc -c`. A global
  normalize-pass that touches PRE-EXISTING tolerated over-runs must
  RESTORE them (scope edits to your own lines).
- When adding a section at an existing depth, REUSE the file's existing
  same-depth marker character (grep for it). Introducing a different
  marker for the same logical depth is the classic "skip level" trigger.

How to apply: map the file's marker ladder first; size underlines with
`len()`; reuse existing depth markers. (The catalog of which file uses
which ladder is in AGENT.md — this lesson is the GENERAL discipline.)

---

## L-010 — V&V vocabulary in prose MUST match the skill verbatim; you are the curator (Directive 5)

You write the prose future readers quote when reasoning about
verification status. The recurring failure is paraphrasing a level
definition or over-claiming what a reference proves. The hard rules
(from `vv-principles`):

- **MMS does NOT verify eigenvalues** — it is source-driven, reaches
  flux-shape / convergence-order only. NEVER write "MMS verifies the
  eigenvalue". Eigenvalue claims need closed-form (`k_∞ = νΣ_f/Σ_a`) or
  semi-analytical references; SI≡Krylov twin agreement is
  necessary-but-NOT-sufficient (needs a structurally-independent leg).
- **L4 (code-to-code) proves zero correctness** — every L4 claim names
  its L0–L2 backing. NEVER "L4 proves correctness".
- **1-group eigenvalue is degenerate** (`k = νΣ_f/Σ_a` flux-shape
  independent) — NEVER "the 1-group test verifies the solver". (1G IS
  fine for a rate/convergence-order claim — declare the claim layer.)
- **Name the pillar** (closed-form / MMS / semi-analytical), not vaguely
  "analytical"; respect each pillar's evidence boundary.
- A **Mode-10 sub-floor term** is closed by STRUCTURAL teeth
  (producer-threads-at-machine-precision + consumed-sign-flip-moves-flux
  + a no-op control leg), NOT a tightened value band — and when no
  isolating regime exists (a boundary-trace slope below the bulk floor
  everywhere), say so: there is NO value-improvement leg to add. A
  prophylactic `.. warning::` ("do NOT write 'S9 recovers 2nd order at
  the boundary'") in the doc itself pre-empts the future over-claim.
- Bug attribution cites the failure mode (1–11) and matches
  `error_catalog.md`; a `catches(ERR-NNN)` claim is a coverage edge,
  not a topic tag.
- **"Adjoint" operator prose: distinguish the EUCLIDEAN transpose (`Aᵀ`, the
  plain group/angle matvec adjoint under the L² inner product) from the metric
  HILBERT adjoint (`A†=G⁻¹AᵀG`, the `.H` wrapper).** A campaign/commit may name
  a new capability "S†" colloquially while the method actually computes the
  Euclidean transpose `Sᵀ` (pinned by reciprocity `⟨Sψ,χ⟩=⟨ψ,Sᵀχ⟩`, NO angular
  Gram). Write the precise object (transpose), note the metric adjoint is the
  separate `.H`, and say the campaign names it "†" colloquially. When the
  FORWARD keeps a fast-path while the ADJOINT rides a different (frame) form,
  that structural asymmetry is what makes the reciprocity gate a GENUINE
  cross-check (two structurally-different representations of one operator), NOT
  a tautology — say so explicitly; it is the structural-independence argument
  at the operator-identity level.
- **The FIRST iterative member of an inverse/solver family that was previously
  all-EXACT has NO bit-id twin to inherit — its claims rest on
  structural-independence anchors ONLY, and it carries NEITHER an eigenvalue
  NOR an MMS claim.** Documenting such a member (e.g. `GreenOperator`, the
  first iterative inverse in the #226 family): state the claim layer as
  foundation / flux-shape against a structurally-independent reference
  (closed-form dense-LU + the multiple-scattering / Neumann expansion), NOT
  inheritance. An iterative *sum inverse* is neither an eigenvalue solver (so
  NO eigenvalue claim) nor source-driven (so NO MMS reference — both pillars
  are INAPPLICABLE, not merely unused). And when the parent's `solve` is
  DEFINED as `self.inverse().apply` (the `OperatorSum` contract), the
  `inverse().apply ≡ solve` equivalence that anchored the family's EXACT
  members is a TAUTOLOGY for this member — exclude it as evidence EXPLICITLY,
  so a later reader doesn't add a green tautology and mistake it for coverage.
  The name-earning distinguishing invariant (here G-Neumann: the splitting a
  generic `A⁻¹` cannot satisfy) is the load-bearing correctness evidence, not
  round-trip.

**Skill-uplift duty:** when you notice a recurring published-prose
anti-pattern, a new failure-mode signature in a close-out, or a new
pillar-evidence-boundary case that `vv-principles`/`error_catalog.md`/
`algebra-of-record` does NOT yet capture, PROPOSE the skill edit in your
return. You read across all close-outs — the skill grows when you feed
it back. (Do not duplicate skill content into this file; point to it.)

---

## L-011 — An overloaded-symbol convention sweep: inventory EVERY meaning, classify each site, replace_all only unambiguous strings

A project-wide symbol-convention change (here: `L` the invertible loss
composite → `A = L + C`, because `L` is reserved for the streaming
leaf) is NOT a blanket rename — the letter is OVERLOADED, and a naive
find-replace corrupts the legitimate uses.

- **Inventory every meaning of the letter FIRST, then classify each
  site mathematically.** `L` meant five different things across two SN
  theory pages: streaming leaf (`\Omega\cdot\nabla`, `L+C`) → KEEP;
  loss composite (`(L,S,F)` triple, `(L-S-F)\psi=q`, `L^{-1}`/`L.solve`
  the sweep, `L.apply`) → FIX to `A`; Legendre order (`L=0` isotropic,
  `L=1`) → KEEP; slab length (`L=5 cm`) → KEEP; generic operator in
  the ABSTRACT-algebra sections (protocol `apply: x\mapsto Lx`,
  `ScaledOperator (\alpha L)`, closure `L=A+0`) → KEEP + FLAG (it does
  NOT denote the composite; swapping it to a neutral symbol is a
  separate style task, out of scope for an L→A fix).
- **VERIFY the target spelling against LIVE code before committing to
  the scope** (L-001 applied to scoping). When the sweep felt alarmingly
  large (~40 `(L,S,F)` sites), reading `iteration.py` signatures
  (`SourceIteration(A_inv)`, `KEigenvalue`'s "`A` the FORWARD invertible
  loss operator", the "(A,S,F) operator triple" comment) proved the
  docs' `(L,S,F)` was STALE vs code — so the comprehensive sweep was a
  Cardinal-Rule-1 doc/code alignment, NOT over-reach. Live signatures
  resolve "is this pervasive spelling stale, or intended?".
- **replace_all ONLY on unambiguous multi-char strings** (`(L, S, F)`,
  `(L - S - F)`, `(L-S)^{-1}`); targeted context-edits for ambiguous
  bare `L`/`L^{-1}`/`L.apply`. Enumerate spacing/punctuation variants:
  `(L - S)` (spaces) MISSES `(L-S)` (no space); `(L, S, F)` MISSES
  `(L, S, F, \psi)` (trailing arg). A FINAL grep of ALL remaining
  `L`-forms — each explicitly re-classified KEEP/FIX — catches the
  variant misses (found 3 stragglers that way).
- **A convention-target symbol can COLLIDE with an existing same-letter
  use in ONE section.** Here `A` (loss operator) collided with `A` (the
  affine SPACE in the #208 torsor section) — but that section already
  used operator-`A` in `r = A\psi - q`, so it was already internally
  inconsistent with its own "connected by `L`" prose. Resolution: use
  the convention's symbol (it was already there), disambiguate AT
  INTRODUCTION by spelling the definition (`the loss operator
  A = L + C`), and FLAG the pre-existing space/operator collision for a
  future disambiguation pass — do NOT rename the other use (out of
  scope), do NOT abandon the convention.
- **Eq-label BODIES change, labels + vv-status stay (L-003).** The
  sweep crossed verifies-target equations (`operator-fixed-source`,
  `sn-streaming-reciprocity`, `streaming-inverse-direct-sum`,
  `octant-direct-sum-tensor-product`): change `L`→`A` in the math,
  keep the label name — even when the label reads "streaming-inverse"
  but the body is now `A^{-1}` (the octant block-structure IS
  streaming-induced; say so in one clause so the label name still
  makes sense). The clean `-W` build confirms every `:eq:` still
  resolves; matrix.rst auto-regens (its only diff was a test-count
  bump from a CONCURRENT agent, not my labels — L-008).

How to apply: for any symbol-convention sweep, map every meaning of the
letter, verify the target against live code, replace_all only the
unambiguous strings, targeted-edit the rest, grep-sweep to re-classify
every survivor, and flag (don't fix) same-letter collisions + the
generic-operator-symbol residue.

---

## L-012 — Merging a RE-STAGED branch's authored docs into a DIVERGED tree: programmatic verbatim splice + path translation + same-merge forward-ref reconciliation

When a long-lived feature branch's DOCS commit can't be raw-applied
(both trees rewrote the pages since the fork), the job is to integrate
the branch's authored CONTENT into today's pages, not cherry-pick the
diff. The disciplines:

- **Extract the branch's `+` block PROGRAMMATICALLY; never hand-retype
  1000+ lines.** For a large authored block carrying verifies-target
  eq-labels + math that MUST be verbatim, hand-transcription is a
  Cardinal-Rule-1 risk. Slice the diff's `+` lines (`diff[start:end]`,
  assert each `startswith('+')`, strip index [1:]), apply path
  translations as string-replaces, write to a temp file, then splice
  via a single `src.count(target)==1`-asserted replace. VERIFY the
  extract before splicing: label counts (each verifies-target ==1),
  anchor counts, first/last lines, residual-old-path==0. A machine
  splice can't introduce a transcription error; an Edit new_string of
  1000 lines can.
- **The branch was authored PRE-reorg — translate EVERY module path to
  the LIVE layout.** Inventory the paths in the diff
  (`grep -oE 'orpheus\.sn\.[A-Za-z._]*'`), map each against the reorg
  (here `orpheus.sn.geometry` → `orpheus.sn.mesh.augmented_mesh`), and
  confirm one prefix-replace covers all occurrences with **zero
  residual**. Verify each translated symbol exists in LIVE code (the
  grep gate — a dead xref renders plain-text, no `-W` warning, L-002).
  Distinguish the moved package (`orpheus.sn.geometry` → `mesh.*`) from
  a same-named-but-UNMOVED one (`orpheus.geometry.reduced_operator`,
  top-level, unchanged) — a blind replace corrupts the latter.
- **Find insertion points by CONTENT/anchor, NEVER the diff's line
  numbers.** The `-` hunk numbers are fork-relative; the tree moved
  (here a section shifted fork-4288 → today-5487). Match the exact
  unchanged target string (assert single match), and MAP the file-local
  marker ladder at the insertion point (the enclosing-section chain,
  L-009) to confirm the block's `~~~~`/`^^^^` levels are valid children
  there — the branch's markers happened to match today's ladder, but
  that is VERIFIED, not assumed.
- **A forward-reference to an issue that LANDED in the SAME merge is now
  stale — flip tense + cross-link (the L-007 landed-seam bullet, sharp
  same-merge instance).** The re-stage bundled Phase-2 content (which
  forward-references #248 as *pending*: "survives Step C … retirement
  tracked under #248") WITH #248 itself (which retired it). Integrating
  both makes the pending-tense a Cardinal-Rule-1 self-contradiction.
  Verify the SHAPE that shipped (`grep 'def __call__'` → fully gone from
  live code), preserve the WHY (why Step C deliberately scoped it out),
  flip "is tracked / survives" → "was subsequently retired under #248",
  and cross-link to the #248 note.
- **Retirement-audit: separate dead CROSS-REFS from stale LITERALS
  before acting.** The requirement's "unresolved-xref hazard" targets
  `:class:`/:meth:`/:attr:` refs (render plain-text, no warning). Grep
  specifically for the XREF FORMS of the retired symbols
  (`:(attr|meth|class):\`[^\`]*\.retired_field\``); if ALL references
  are double-backtick literals (as here for `tau_mm`), there is NO
  build-invisible hazard — the remaining work is pure Cardinal-Rule-1
  literal-claim correction. Fix the literals that DIRECTLY contradict
  the just-integrated content (clean scope: "τ single-sourced in
  spherical_streaming" → in the closure); TOMBSTONE/FLAG the ones
  describing a retired MECHANISM, ESPECIALLY when its root staleness is
  a DIFFERENT issue than the one you're integrating (here the
  `alpha_in is None` slab/curvilinear discrimination was retired by an
  earlier Issue #196 Phase G Step 2.5, surfaced during a #236 audit →
  add a correction `.. note::` grounded verbatim on the live
  `StreamingTerms` docstring, flag for a dedicated rewrite; do NOT
  fabricate the current mechanism, per L-007).

How to apply: read the fork-diff as the CONTENT source, not a
patch; splice the large block programmatically + translate paths +
verify the extract; place by anchor; then run the standing
retirement-audit (repoint xrefs, reconcile same-merge forward-refs,
fix-or-tombstone stale literals) and the `-E -W` build-count gate.

---

## L-013 — A fix that RETIRES a failed-approach family gets a SUCCESS-resolution chapter, not the 9-step CLOSED arc — and proportionate treatment of the big historical narrative it supersedes

When a landed fix works BY retiring a whole failed-strategy family that
a large historical arc built up (here #282 route (a) retired the
`PsiHalfAngleSeed` seed zoo threaded through a 2800-line Phase-D→F→ERR-058
saga), the doc is a **resolution chapter**, not a CLOSED post-mortem. It
still borrows the close-out arc's spine (status banner, what-was-tried-
and-failed table, numerical before/after) but its verdict is "the fix
shipped", not "the path is dead". The disciplines that made this
proportionate and correct:

- **New resolution SECTION as the saga's final chapter** (a `-` h2
  sibling of the close-outs it resolves, placed right before the next
  `=` h1), with a status banner, the structural defect (the back edge),
  the fix, the derivation, the **failed-strategy `.. list-table::`**
  (each retired class as a LITERAL + its failure mode), numerical
  evidence, and the honest-scope caveat. This is the primary deliverable.
- **ONE loud supersession banner at the ARC HEAD** (`.. attention::`
  right under the first superseded section's title) telling the reader
  every "current default / retained strategy" claim in the sections
  below is HISTORICAL, with the forward `:ref:`. This lets the whole
  historical narrative stay as legitimate history WITHOUT rewriting it —
  proportionate. Do NOT tombstone every stale sentence.
- **Targeted retraction tombstones ONLY on the bald factual REVERSALS**
  (an "X is retained (not deleted)" that route (a) deleted; an
  "Infrastructure retained" table listing the retired classes as
  production). `.. note:: **Retraction (date, Issue #N).**` per L-007 —
  the numerical/historical evidence stays; the interpretation gets the
  tombstone.
- **The prior close-out's "Open research paths" is a GOLDMINE — flip the
  one that LANDED.** If the fix implemented a predicted research path
  (here #1 "TRUE-source-driven sweep-side seed" + the full Legendre
  fold, AND its proposed "sweep quadrature order" probe was EXACTLY the
  N-sweep discriminator used), flip it "→ **LANDED as #N**", tell the
  "predicted exactly / one refinement over the prediction" story
  (L-007 landed-seam). This is a powerful principled-resolution
  validation AND closes the research-path loop on the same page.
- **Literalize the retired family's dead xrefs via `replace_all`**
  (L-002/L-011): each `:class:`X``/`:meth:`X.__call__``/`:attr:`X.a``
  is backtick-delimited and unambiguous → `` `X` `` literal naming the
  historical artifact (NOT the surviving successor — a retired STRATEGY
  class and a surviving free FUNCTION of a similar name are different
  concepts; name the class as history, point prose at the function).
  The `-W` build is BLIND to these (plain-text) — grep-gate is the check.
- **Reuse the topic's existing verifies-target labels** (`hebert-3-43x`
  were already the full-fold equations) via `:eq:`; mint new
  `documented`-status labels only for the fix's OWN structural identities
  (augmented-composite, block-triangular, the streaming-manufactures-
  anisotropy identity) — each gets `.. vv-status: <label> documented`
  (L-004) or it lands in the matrix's "Orphan equations" bucket, not
  "Documented-only". Verify the bucket in the regenerated `matrix.rst`.
- The L-012 programmatic-splice + placeholder-underline generator works
  for a large NEW authored section too (not just merging a branch diff)
  — it guarantees code-point-correct unicode underlines (L-009).

How to apply: for a "the fix worked by retiring the failed family"
doc, write the resolution chapter + ONE arc-head supersession banner +
targeted reversal tombstones + the landed-research-path flip; literalize
the dead family xrefs; reuse existing verifies-labels, tag new
structural ones `documented`. Preserve the history, make the
supersession loud, don't rewrite the saga.

---

## L-014 — Deepening a thorough resolution chapter with physics RULINGS + the current-truth vs PLANNED-design-direction split

Two distinct moves recur when a design session digs deeper physics /
architecture for a feature whose *representation* is ALREADY thoroughly
documented (here: the SN curvilinear ψ½ pole seed, whose #282 route-(a)
resolution chapter already covered the defect/fix/representation, and a
`facefield_codim1_design.md` note that dug the deeper physics + a
planned refactor).

- **Augment with the deeper WHY, cross-link the existing WHAT — never
  duplicate.** When the chapter already documents the representation
  (the composite, the block-triangular normal form, the role-quadruple),
  the augmentation's whole value is the PHYSICS beneath it: *why* the
  direct solve exists (the pole is a straight characteristic → pure 1-D
  ODE, `(1−μ²)=0` kills redistribution), *why* the presence-predicate
  looks as it does (topology of the redistribution axis: a periodic CIRCLE
  gives edge-inclusion free + spectral, an INTERVAL's open GL makes you pay
  a seed).  ⚠ **The "zero metric is structural / entirely-grazing angular
  face" ruling this bullet ORIGINALLY carried was REFUTED — see L-015.**
  The ψ½ block's Hilbert STATE metric is the SPD `G_sd = V_cell`, NOT zero;
  `(1−μ²)|_pole = 0` is an OPERATOR coefficient (through-flux M2), not the
  state metric (M3); keep the three measures distinct (M1 moment weight /
  M2 through-flux coefficient / M3 state metric) — only M2 is zero. Give each ruling
  its OWN labelled `~~~~` subsection (`documented` vv-status on any
  literature-transcribed derivation eq per L-004), and cross-link the
  existing representation eqs (`:eq:`…-block-triangular``) rather than
  re-deriving them. The "what was tried and declined" belongs here too —
  a numerically-affordable-but-architecturally-declined alternative (the
  Gauss–Lobatto pole-node study: ~1.2× penalty, but declined to keep the
  cell-centred bulk clean) is exactly the Cardinal-Rule-3 "tried and why
  not adopted" content; cite the scratch-artifact location, mark it
  uncommitted/promote-only-if-adopted.
- **A PLANNED refactor on a current-truth page gets a LOUD
  design-direction admonition + a paired "current state" subsection.**
  When the design note proposes a refactor NOT yet built (a `FaceField`
  codim-1 parent, a `face_streaming_normal` measure, mesh-derived
  presence), the current-truth page must NOT read as if it exists.
  Recipe: (1) a `.. admonition:: Design direction — … (PLANNED, not
  built)` `:class: note` opening with an explicit "not implemented — no
  X/Y/Z type exists yet" + a pointer to the design-note path; (2) unbuilt
  types are LITERALS (``FaceField``), never `:class:` refs (L-002
  forward-ref — a `:class:` to a non-existent symbol renders plain-text,
  no `-W` warning, but is a Cardinal-Rule-1 stale ref); an ASCII
  hierarchy tree in a `::` block sidesteps the issue (all literal); (3)
  a SEPARATE "Current state — the `Optional` block and the N guards"
  subsection stating what IS built, so plan and reality never blur.
  VERIFY the guard/DOF counts against LIVE code, not the design note — a
  scoped grep undercounts (a `_require_*` call site lived in a sibling
  module `operators/streaming.py`, outside the `loss_representation/`
  scope I first grepped; the note's "3+4=7" was right, my scoped grep
  said "2+4"). Verify the enclosing CLASS of a cited call site too (the
  streaming.py site was in `InvertibleOperator`, not `StreamingOperator`).
- **The same-topic forward-ref in an ADJACENT subsystem is usually
  already stale — flag with the shipped evidence, don't rewrite (L-007).**
  A "when #N lands, gate X flips" future-tense claim in a neighbouring
  section (here the assembly-mode "Cartesian-only scope" §) is a
  landed-seam once #N ships; the TEST comment confirms the flip
  (`test_assembly_mode.py`: "route (a) makes the augmented matrix EXACTLY
  block-lower-triangular … replaces, not relaxes, the RED
  characterization"). But if it's a behavioral-claim rewrite in a
  different subsystem (assembly, not the pole-seed physics), FLAG it with
  the verified shipped shape + exact lines for the main agent — a rewrite
  needs its own verify-against-live pass and may interact with a claim
  that DIDN'T change (the bulk assembler staying Cartesian-only).

How to apply: for a "deepen an already-documented feature" task, add the
physics rulings as labelled subsections that cross-link (not duplicate)
the existing representation; give any planned refactor a loud
PLANNED-design-direction admonition (literals for unbuilt types) paired
with a current-state subsection; verify every count/class against live
code; flag adjacent same-topic landed-seam staleness rather than
smuggling a behavioral rewrite into scope.

---

## L-015 — The SUCCESS-CORRECTION doc pass: a doc's own PHYSICS FRAMING was the bug, proven this session; rewrite every site to the corrected story

Distinct from L-013 (retiring a failed APPROACH) and L-014 (deepening a
resolution chapter): here a framing DOCUMENTED AS CORRECT by prior
sessions — even by THIS lessons file (L-014's since-corrected "zero metric
is structural" clause) — was PROVEN WRONG this session and the code fix
landed (SN ψ½ block metric: the retired "ghost metric" `G_sd ≡ 0` → the
SPD state metric `G_sd = V_cell`). The doc job is to make every site tell
the corrected story. The disciplines:

- **Blast radius = the refuted CONCEPT grepped tree-wide, NOT the brief's
  file list.** The brief named 6 sites; `grep "ghost metric\|G_sd = 0\|
  all-zero ghost"` across `orpheus/` + `docs/theory/` found 3 MORE
  (transport-layer field docstrings) carrying the SAME refuted claim. Fix
  them (clean one-line Cardinal-Rule-1 corrections) and FLAG the
  scope-expansion. EXCLUDE frozen archaeology (`.claude/plans/*`, other
  agents' `agent-memory/*`, sibling worktrees) — they keep the old framing
  as history.
- **RENAME (not keep) an anchor whose NAME encodes the REFUTED concept**,
  updating every referencing site — the INVERSE of L-007's
  keep-the-anchor-when-the-concept-SURVIVES. `sn-282-ghost-metric-face` →
  `sn-282-pole-state-metric` because the section now REFUTES the ghost
  framing. Safe only because all inbound refs (here 2, both cross-doc in
  the page I was already rewriting) were updated in the same pass; VERIFY
  in built HTML that the new `id=` exists, the old is gone, and each
  cross-doc `:ref:` resolved to an `<a href>` (section anchors DO resolve
  cross-doc, unlike eq-labels — L-002).
- **Preserve the retired-bug WHY in PAST TENSE (L-007), don't erase.**
  Every `G_sd = 0` / "ghost metric" that SURVIVES the sweep must read as
  history ("the retired ghost bug installed…", "the shipped `G_sd = 0` WAS
  a wrong adjoint", "a reverted ghost would ALSO leave a defect"). A final
  grep re-classifies each survivor KEEP-as-history vs FIX; the ones in the
  fix's OWN validation body (the error message explaining why `0` is
  REJECTED) are correct as-is.
- **The load-bearing content is the CATEGORY-ERROR framing.** When N
  quantities were conflated into one wrong value, the crisp correction is
  a per-quantity `.. list-table::` (M1 moment weight / M2 operator
  coefficient / M3 state metric — each "where it lives / what it governs")
  + the WHY only ONE is zero: an operator coefficient equals the state
  metric ONLY when the face's operator self-block is trivial (spatial
  trace `A_tt` = restriction map → through-flux = state metric; ψ½ `A_ss` =
  banded radial transport op → through-flux 0 ≠ state metric `V_cell`).
  Ground "trivial self-block" in MEASURED norms from the
  derivation-of-record (`A_tt` offdiag ≈2 vs `A_ss` ≈71, ratio ≈35×).
- **A refuted-framing fix that CLOSES a vv failure mode gets a POSITIVE
  reframe naming the mechanism.** The old Mode-12 gotcha ("G-recip is
  IDENTICALLY blind to the seed rows") → "G-recip CATCHES a seed-row error
  (Mode 12 CLOSED, ERR-067)". NEW closure mechanism (Directive-5 skill
  proposal, below): a Mode-12 blindness closes EITHER by gating the OBJECT
  (the skill's canonical remedy) OR by REPAIRING the functional's METRIC so
  the error class LEAVES the invariance group — available exactly when the
  metric WAS the bug (correctness fix ≡ Mode-12 closure). Cite the LIVE
  gate name (grep it — the test was RENAMED `…_is_blind_to_…` →
  `…_catches_…`) and its both-legs subtlety (control leg: unmutated recip
  holds `<1e-12`; mutated leg: flip reds `>1e-6` — without the control a
  broken baseline mimics "caught"). ERR-067 lands in `error_catalog.md`
  from a concurrent QA agent — verify it exists + your wording matches
  before citing (it did).

How to apply: grep the refuted CONCEPT tree-wide (blast radius > brief);
rename refuted-concept anchors + update all refs (verify in HTML);
preserve the bug's WHY in past tense; correct via a per-quantity
category-error table grounded in measured norms; give any closed vv
failure mode a positive reframe naming the closure mechanism; and SHARPEN
any prior lesson (L-014 here) that carried the now-refuted framing.

---

## L-016 — The EVICTION/re-homing doc-pass: a sub-object leaves a block-ON-a-carrier for its OWN coupled composite — the PHYSICS survives, so reframe the CARRIER narrowly

Distinct from L-007 (retirement) / L-013 (retiring a failed approach) /
L-015 (a refuted framing): here a correct sub-object (the ψ½ ray) is
RE-HOMED — it moves from an optional third block ON a carrier
(`FullField`) to its OWN 2-block composite (`RadialCharacteristicComposite`
= System B) coupled to System A via a `CoupledField[ψ_A, ψ_B]` pair. The
governing insight: **an eviction changes the CARRIER, not the PHYSICS**,
so the doc job is a NARROW carrier reframe, not a chapter rewrite.

- **The brief's "~N stale refs" over-counts because most of the
  "3-block" chapter is PHYSICS that survived.** The route-(a) resolution
  chapter (pole straight-characteristic, M1/M2/M3 metric, block-triangular
  walk order, source fold, R12a, circle-vs-interval) describes System B's
  physics and is UNCHANGED by the eviction. Grep for the STALE CARRIER
  FRAMING ("grows a third summand", "augmented (bulk ⊕ trace ⊕ seed)
  composite", "third (seed) block", "optional third block",
  "mixed-presence law", "N guards"), NOT the physics terms. The actual
  stale set was ~4 sites (heading + intro + one gotcha + one gate-norm
  phrase), not the brief's ~18. Reframe those; add ONE focused
  "**Where X lives — System B (the eviction)**" prose paragraph after the
  role-table that states the end-state (2-block carrier + own composite +
  `CoupledField` + retired presence-law + live-illegal-state
  unrepresentable + honest DOF + the `Solution.<member>` biconditional)
  and CROSS-LINKS the coupled 2×2 / M−N algebra to its home page — do not
  re-derive it. KEEP the total-phase-space eq-label (it is still the
  honest DOF sum; L-003) and just reframe the prose around it.
- **A RENAME (helper → new name) is a CLEAN `:func:` repoint, not a
  literalize-as-dead (L-013).** When the retired symbol SURVIVES under a
  new name (`_within_group_triple` → `build_within_group_system`, which
  also grew a record `WithinGroupSystem` carrying the named `A = M − N`
  splitting), repoint every `:func:`/literal to the LIVE name; keep ONE
  historical literal ONLY where the doc explicitly narrates the retirement
  ("the former ``_within_group_triple`` retired into this builder").
  Verify the rename is LANDED in live code first (`grep` the module: the
  old name gone, the new imported) — 13 dead `:func:` refs render
  plain-text with no `-W` warning (L-002), so the grep gate is the only
  catch. On the general/Cartesian sites, the record DEGRADES to the old
  triple (`.implicit_operator=(L+C)`, `.explicit_gains=(S,B_a)`) — say so;
  the coupled M−N grid is the CARRYING-mesh case only. (Those two fields
  were spelled `.resolvent`/`.gains` until 2026-07-28; `resolvent` was a
  misnomer — the field holds the un-inverted forward `M`, not `M⁻¹`.)
- **An IN-FLIGHT concurrent deliverable (the main agent editing prod while
  you write docs) gets post-state prose + a FLAGGED forward-dependency.**
  The LC-triplication collapse was being done in solver.py concurrently
  (`self.L` still live when I read it). The brief directed post-collapse
  prose; I wrote "the within-group L+C is spelled in ONE place — the
  builder" WITHOUT asserting solver internals (line numbers / `self.L`),
  and FLAGGED that the prose asserts the concurrent collapse for the main
  agent to reconcile at commit. Write what the brief directs, avoid the
  internals the brief forbids, flag the dependency.
- **A PLANNED-design admonition whose STRUCTURAL half landed via a
  SEPARATE commit: correct the false claims, keep the still-planned
  vision.** The `FaceField` codim-1 admonition said "no `FaceField` ABC
  exists yet" — but it LANDED (a separate C-series commit,
  `grep 'class FaceField'` + git log confirmed). Cardinal Rule 1: correct
  "PLANNED, not built" → "PARTIALLY built" (structural parent landed +
  ERR-067 metric refutation stands), note the SIBLING mechanism that
  reached the goal DIFFERENTLY (the eviction made presence
  unconstructable-by-design via System B, not via the still-unbuilt
  `PhaseSpaceCarrier`), and preserve the genuinely-unbuilt vision. Verify
  the guard COUNT is not restated (the old "7 guards" was already stale at
  17 live sites) — describe the guard FAMILY (`_require_/_refuse_/_require_leg_pair`)
  and the six-signature leaf-kwarg PROTOCOL (a `.. list-table::` of the
  apply/solve/transpose read-vs-fill contracts) instead of a call-site
  count, which sidesteps the L-014 count-drift trap.

How to apply: for an eviction/re-homing doc-pass, grep the CARRIER
framing not the physics; reframe narrowly + add one end-state paragraph
cross-linking the coupled algebra; clean-repoint the renamed builder
(verify landed, keep one historical literal); write post-state prose for
an in-flight sibling deliverable and flag it; correct a PLANNED
admonition's now-false claims while keeping its unbuilt vision.

---

## L-017 — The freed-name REMINT collision: a retired symbol's name reused for a DIFFERENT live object — disposition every mention by PASSAGE MEANING, not by name; and record a solve-leg un-weave in current-architecture passages only

The sequel to L-016 (the eviction): a refactor RETIRES a symbol AND
remints its freed name onto a **different, still-live** object. Here the
unified ψ½ leaf ``RadialCharacteristicField`` was retired (split into
``RadialCharacteristicInteriorField`` / ``…BoundaryField``) and its freed
name reminted onto System B's **composite** (the ``FullField`` mirror,
``Composite[interior, boundary]``). The name becomes a **homonym across
the remint commit** — the load-bearing discipline is to disposition each
mention by WHAT THE PASSAGE DESCRIBES, never by the name:

- **Current-architecture passage → REPOINT** to the live object (the
  composite ``~orpheus.transport.radial_characteristic_field.RadialCharacteristicField``).
- **Historical record → REWRITE as history** ("the unified single-buffer
  leaf, which then held the ``RadialCharacteristicField`` name; reminted
  onto the composite at 4e-e1b") — a literal, not an xref. A blanket
  find-replace is WRONG; the same name flips meaning at the remint.
- **Grep the FULL MODULE PATH, not the bare name.** A partial mechanical
  rename that already ran (e1b renamed the docs' ``RadialCharacteristicComposite``
  → ``RadialCharacteristicField``) leaves SOME refs correct (the composite
  path ``radial_characteristic_field.RadialCharacteristicField``) and
  OTHERS dead (the retired leaf's ``fields._bases.RadialCharacteristicField``,
  the unified space's ``…space.RadialCharacteristicSpace``, the unified
  ``…radial_characteristic_flux.RadialCharacteristicFlux``). The bare-name
  grep can't tell them apart; the module-path grep + import-gate can.
  Watch for a role FAMILY that split into ``interior ⊕ boundary`` (flux /
  source-sink / displacement / residual → 8 leaves): the unified per-role
  module paths (``…_source_sink.RadialCharacteristicSourceSink``,
  ``…_displacement.RadialCharacteristicDisplacement``) ALL die together —
  audit the whole family, not just the head symbol the brief names.

**Recording a solve-leg UN-WEAVE (inline orchestration → named resolvent).**
When a walk's welded inline orchestration is extracted to a NAMED operator
(``RadialCharacteristicOperator.solve`` / ``.solve_transpose``, the ``A_BB``
resolvent), the ENGINE ref (``carlson_inward_sweep_from_source``) stays
where it names the **march mechanism** (fine — the engine still exists and
is the single source of the march); the **orchestration** ref becomes the
operator's ``solve``. Record the un-weave in the CURRENT-architecture
passages ONLY — the Key-Facts bullet, the walk-triple SOLVE bullet, a
dedicated Cardinal-Rule-2 ``.. note::``, the six-signature protocol note,
and the Development-history changelog. The HISTORICAL saga (the Phase D–F
"one helper, two consumers" sections) that describes the OLD inline
architecture is PRESERVED per L-013 — it already carries the supersession
banner; do NOT rewrite it (proportionate). Two traps:

- **A re-aimed sentinel is a stale V&V claim (Directive 5).** The Mode-11
  wrap-sentinel was re-aimed from the engine onto the operator (class-level
  wrap of ``RadialCharacteristicOperator.solve``) + a NEW S2 "walk source
  has zero ``carlson`` references" tripwire. A doc saying "the sentinel
  confirms the solve executes ``carlson_inward_sweep_from_source``" is
  STALE — VERIFY against the live test (read the test body) and repoint to
  the operator + name the S2 tripwire. "carlson refs went N → 0" is a
  greppable fact — state the measured count.
- **A refuted-framing survivor on an adjacent HISTORICAL changelog line**
  (a "zero-metric" the ERR-067 pass corrected to SPD ``V_cell``, which the
  L-015 tree-wide sweep missed in the Development-history table): fix it
  when you edit the DEAD ref on the SAME line (Cardinal Rule 1 is supreme —
  a changelog is current-truth, not licensed to carry a refuted claim), and
  FLAG the scope-expansion. Don't retroactively inject the LATER
  architecture (System B) into an EARLIER-dated entry — keep the entry's
  date-accurate framing, only correct the dead ref + the refuted claim.

How to apply: for a freed-name remint, grep the full MODULE PATH (+ the
whole split role family), disposition each site by passage meaning
(repoint-live vs rewrite-history), record the un-weave in
current-architecture passages only (preserve the historical saga per
L-013), re-verify any re-aimed sentinel against the live test, and
fix-plus-flag refuted-framing survivors on lines you touch.

---

## L-018 — The CAPSTONE pass: documenting a COMPLETED multi-block coupled-operator architecture as one new taxonomy-culminating section

The completion (step-7) of the eviction/remint arc (L-016/L-017): the ψ½
ray, having moved to its own System B, is now one leaf of a full 2×2
**coupled block operator**, and the campaign LANDED. The doc job is a NEW
capstone `=` section documenting the whole architecture — not an
incremental reframe. Disciplines:

- **Place the capstone as the taxonomy's CULMINATION, cross-linking not
  duplicating.** The coupled block operator is the block-level
  generalization of the page's existing operator-surface taxonomy
  (three-layer surface → materialise → assemble → **N×N block grid**):
  apply→block matvec, assemble→block-offset scatter, solve→block
  substitution. Place it right after the last taxonomy sibling (the
  assembly axis), and OPEN by naming the generalization + `:ref:`-linking
  the sections it generalizes — the section EARNS its narrative place,
  it isn't a bolted-on appendix. Reconcile the pre-existing forward-ref
  ("the record bridges System B into a coupled M−N grid") by repointing
  it to the new same-page `:ref:` and softening stale framing ("bridges"
  evokes the retired bridge object).
- **A naming-dense brief on a landed architecture is an L-001 minefield —
  verify EVERY named object/line-ref/helper against the module-of-record
  before writing.** The brief conflated `A_BA` with
  `RadialCharacteristicReconstruction` (:955) — but LIVE `A_BA =
  RadialCharacteristicEmission` (:1187); Reconstruction is the Fold
  FACTOR *within* A_BA (`A_BA = Fold ∘ K ∘ integrate`). The brief's fold
  helper `fold_moments_to_starting_direction`
  (`starting_direction_space.py`) was renamed to
  `fold_moments_to_radial_characteristic` (`radial_characteristic_space.py`)
  by the campaign's OWN step-1 rename. Read the module docstring + class
  defs + import-verify each symbol before minting a cross-ref; a
  naming-dense brief's line-refs and class-names are the FIRST thing to
  go stale on a fast-moving branch.
- **Document a symbol OVERLOAD as an explicit gotcha, don't paper over
  it.** The class `RadialCharacteristicOperator` calls itself "A_BB" (the
  bare radial march μ∂_r+σ_t) AND the builder's local `A_BB = march −
  B_b` (the loss-grid self-block) — two live meanings. Faithful to the
  code-of-record: define A_BB = the bare march, spell the loss-grid (B,B)
  = A_BB − B_b explicitly ("a naming gotcha to spell out"), tie it to the
  System-A parallel (A_AA = L+C−S−B_a; both self-blocks = transport −
  gains − boundary; both boundary gains lagged in N). Reader inherits the
  overload cleanly instead of tripping on it.
- **The resolvent M and the loss grid A are DIFFERENT grids — state both
  precisely.** The brief's loose "M grid [[LC, Seeding], [None, A_BB]]"
  is the RESOLVENT M (bare-march diagonal, (B,A)=None → upper-triangular
  → direct block substitution). The LOSS grid A is [[L+C−S−B_a,
  +Seeding], [−Emission, A_BB−B_b]] ((B,A)=−Emission present). The
  splitting A = M − N puts B_b + the emission gain in N. Give both grids
  their own `.. math::`; note M(B,B)=bare-march while A(B,B)=march−B_b so
  A=M−N recovers. (B,A)=None-in-M IS the Schur/lag argument: the emission
  is the iterating scattering gain (ρ(M⁻¹N)=0.371), it belongs lagged on
  the rhs, not folded into the one-pass resolvent.
- **D5 for a campaign that RETIRES symbols: it owns its retirement
  doc-debt, but a BROAD PRE-EXISTING stale surface with NO 1:1 successor
  is FLAGGED, not rewritten-in-passing (L-007).** The campaign's OWN
  retired symbols (the ψ½ kwargs, the fused `CoupledInvertibleOperator`
  bridge, the presence guards, the `_within_group_triple`) were ALREADY
  correctly narrated as history by the incremental passes (the step-6
  "retired estate" section + dated changelog entries — literals in
  history blocks, not live xrefs) — verify each is history (grep the
  site's enclosing section title), leave it. But `transport_sweep`
  (retired by THIS campaign's step 6) had 55 refs across ~15 SN-page
  sections from many prior waves, MANY presenting it as current API, with
  NO mechanical successor (the sweep is now the resolvent's `.solve` /
  `sweep_schedule` per context — a behavioral rewrite per site).
  Rewriting ~40 current-API sites in a coupled-block docs pass is the
  exact L-007 anti-pattern. FLAG it (count + no-successor nature) as a
  dedicated sweep-entry-point-retirement pass; fix only campaign-adjacent
  current-API sites (here: zero — no transport_sweep site is in a
  coupled-block section).
- **New structural eq-labels for a landed architecture are all
  `documented` (L-004); one per load-bearing identity, grep-collision-
  check first.** Block matrix, block matvec, the fold, the M−N splitting,
  the loss grid, the free-identity residual, the block substitution —
  each representational/structural (not a solver claim), so `.. vv-status:
  <label> documented`. Grep `:label:` repo-wide before minting (all 9
  collision-free); verify in built HTML each `id="equation-<label>"`
  rendered and each in-prose `:eq:` resolved to `<a>`. Code-xrefs to
  the coupled_system/radial_characteristic classes render PLAIN-TEXT by
  page convention (not member-automodule'd — L-002; the pre-existing
  `build_within_group_system` ref already renders plain-text) — the
  import-gate is the real check, plain-text is NOT a regression.

How to apply: for a completed-campaign capstone, write ONE new
taxonomy-culminating `=` section (cross-link the siblings it generalizes,
don't duplicate); verify every named object/line-ref/helper against the
live module-of-record (naming-dense briefs go stale first); document
symbol overloads + the resolvent-vs-loss-grid distinction as explicit
gotchas; leave the campaign's own already-historical retired symbols,
FLAG the broad pre-existing no-successor surface; tag new structural
labels `documented`.

---

## L-019 — The context-dependent ENTRY-POINT retirement pass: per-site a/b/c disposition grounded in the LIVE successor, never a 1:1 rename

The EXECUTION of the L-018-flagged "dedicated retirement pass": a widely-
referenced entry point (here `transport_sweep`, 56 sites × 5 theory
pages) retires and its successor is **context-dependent** — the same
retired name maps to DIFFERENT live surfaces per site (production
resolvent `.solve` vs the scheduling layer vs a raise-guard vs an
SI-sweep twin method). A mechanical find-replace is FORBIDDEN (per-site
false-claim risk). Disposition EACH site into one of three, grounded in
what the LIVE code does THERE:

- **(a) behavioral rewrite** — the section teaches CURRENT API
  (present-tense claim, OR the whole section framing IS the retired
  entry). Two sub-shapes: an inline symbol-repoint (a present-tense
  claim swaps to the live surface — e.g. "the sweep at `X` consumes both"
  → "the within-group sweep (the resolvent `solve`) consumes both"); and
  a WHOLE-FRAMING rewrite (a section titled/built around the retired
  entry, with a stale code block — e.g. "Quadrature Dispatch" / "Typed
  input") gets its framing rewritten to the current architecture +
  cross-ref, and the stale code block **DELETED**, not symbol-swapped.
- **(b) past-tense history literal** — dev-history / changelog /
  retired-estate / diagnostic narrative. KEEP the name but as a
  double-backtick LITERAL (never a `:func:`), framed past-tense with the
  retirement citation ("the then-production ``X`` entry, since retired at
  step N (R-N.N)"). The campaign's OWN retirement passes usually already
  literalized their sites (grep: they're `` ``X`` `` not `:func:`X``) —
  LEAVE those; only literalize the dead `:func:`/`:meth:` refs OTHER
  waves left behind (they render plain-text, no `-W` warning, L-002 — so
  grep is the only catch).
- **(c) delete** — the clause carries no content once the symbol is gone
  (a "bit-identical to a direct ``X`` call" line; ``X`` as one entry in a
  list whose OTHER members survive → drop it, keep the survivors).

Disciplines that made it correct:
- **Ground EVERY successor in live code THIS session (L-001).** The
  live-grounded successor table is the load-bearing artifact — build it
  FIRST. A brief's "the successor is X" is a STARTING heuristic; the
  per-site live read is the rule (a raise-guard's kwarg was
  `moment_projection` in the doc but `moment_frame` in live code; a
  separate arg `Q_aniso` was GONE entirely — folded into the source; the
  SI-sweep twin pairs with the matvec `_apply_walk` as `…ScanWalk.sweep`,
  found by reading the walk class's methods).
- **For a big superseded-architecture SECTION prefer the L-013 arc-head
  supersession banner + past-tense over a full rewrite** (proportionate).
  A dev-history section framed "Wave-X did Y" that presents the retired
  entry with a code block: add ONE `.. note:: **Superseded (step N)**`
  naming the current architecture + `:doc:` link, past-tense the intro
  verbs, let the historical code block stand under the banner. Reserve
  the full framing-rewrite for CURRENT-architecture reference sections
  (early-doc, NOT "Wave-X"-framed).
- **Retitle a heading that NAMES the retired entry** only after grepping
  for inbound `:ref:` (autosectionlabel); size the new underline in code
  points (L-009).
- **Acceptance = the three-severity `-E -W` count UNCHANGED-from-baseline
  + a `git grep <symbol> -- docs` survivor audit** where every survivor
  is a past-tense literal (no live xref; none presenting the symbol as
  current). The skip-line DIFF (baseline vs post) proves you orphaned no
  OTHER verifies-target while resolving the in-scope ones.

How to apply: build the live-grounded successor table FIRST; disposition
each site a/b/c by what live code does there; banner-not-rewrite the big
history sections; literalize every dead `:func:` (grep-gate — `-W` is
blind); prove the skip-line diff changed only what you intended.

---

## L-020 — The retired symbol whose deletion is a COROLLARY of a design unification: the enclosing section's THESIS is stale, not just the symbol

The sharpening of L-019 for the hardest retirement case. L-019 dispositions
each SITE a/b/c and banners a "Wave-X did Y"-FRAMED history section. L-020
is the case L-019's per-site list UNDERSELLS: the brief hands you a few
"dead-role lines" but those lines are the visible symptoms of an entire
doc-SECTION whose THESIS (its design-rationale premise, presented as
CURRENT) has gone false — because the symbol you were sent to retire is one
FACET of a deeper architectural UNIFICATION that dissolved the very design
the section exists to explain.

Worked (Task #57, `transport_operator_matvec` family + `psi_bc`): the brief
listed 6 sites (discrete_ordinates 13543-13545/13576/13700-13704 +
index_convention 488/1507/1524). Reading the LIVE code (streaming.py apply →
`loss_action` → `_apply_walk`; `loss_representation/__init__.py:1458`
"matvec ≡ sweep, ONE discretization, L21") revealed the matvec deletion was
a corollary of the #206 Phase-C **matvec ≡ sweep = ONE loss-representation
walk** unification — which DISSOLVED the "two distinct discretisations
(FD-operator apply vs WDD sweep) / packed-vector-vs-structured-array layout
difference / deliberately-legacy-pending-PR-INDEX-7" design that FIVE
sections were built to teach as CURRENT. The stale unit was the section
PREMISE, not the line.

The tell (distinct from L-019's "Wave-X-framed" history): a section stated
its stale design in the PRESENT tense as a live rationale — "apply and solve
use different closures **by design**", "What **stayed** deliberately legacy",
"`apply` operates on the **packed 1-D solution vector**". A per-site literalize
leaves the false THESIS standing.

Disposition ladder for a thesis-stale section:
- **THESIS-stale reference/rationale section → SUPERSESSION BANNER at the
  section head** stating the unification + the current one-truth + the campaign
  cite, then past-tense the body verbs + literalize dead roles, PRESERVING the
  historical reasoning under the banner (the Wave-D two-closure narrative +
  ERR-026 stayed — it is still pedagogically load-bearing). Retitle
  "…**(historical)**". This is L-019's banner move promoted from
  history-framed sections to present-tense-rationale sections.
- **The ONE genuinely stale-AS-CURRENT contract → full behavioral section
  REWRITE** to the unified live contract (here the psi_bc/Q_aniso "Vector
  layouts" bullet list → the `FullField` composite: source on
  `rhs.interior.values`, boundary on typed `AngularBoundaryFlux` face views),
  with the retired triple recorded in a trailing `.. note::`. This rewrite is
  what actually KILLS the "Persistent boundary-flux dict psi_bc carrying state"
  bullet (grep it → 0).
- **A moot FUTURE-WORK section (the unification made the planned migration
  unnecessary) → retitle "…(obsoleted)" + `.. note:: Obsoleted by deletion`,
  PRESERVING its `:ref:`-target label** (an `.. _future-work:` anchor is often
  referenced from elsewhere; deleting it dangles those refs — keep the label,
  make the CONTENT truthful). Grep the label before touching it.
- **Co-literalize deleted-SIBLING roles only INSIDE clauses you're already
  rewriting** (a `:func:solution_to_angular_flux` sitting in the same sentence
  as your matvec fix — leaving a live dead-role in a line you're editing is a
  self-inflicted Rule-1 staleness). FLAG the standalone sibling-cluster
  (EquationMap/codec) for its OWN pass — don't chase it tree-wide (L-007).

Disciplines carried from L-019 unchanged: ground every successor in live code
FIRST (L-001); grep-gate every new `:func:`/`:class:`/`:meth:` xref against
live code AND (for Protocol methods) a python `hasattr` probe (L-002 — dead
roles render plain-text, `-W` blind); size retitled underlines in code points
(L-009); acceptance = the three-severity `-E -W` count UNCHANGED-from-baseline
+ a `git grep <symbol> -- docs orpheus` survivor audit where every survivor is
a past-tense literal.

How to apply: after the successor table, read each flagged line's ENCLOSING
SECTION and ask "is the section's PREMISE still true under live architecture?"
If the symbol's deletion is a corollary of a unification (grep the live code
for the "one X" / "matvec ≡ sweep" / "dissolved" fact), the premise is stale
too — banner the rationale section, rewrite the one stale-as-current contract,
obsolete-but-preserve the moot future-work section, and preserve the historical
reasoning under every banner.

---

## L-021 — The bulk-scanner staleness sweep: I am the JUDGMENT layer over a precision-over-recall Haiku pass — re-verify EVERY finding vs LIVE, reject stale-evidence, and a scanner suggestion is a STARTING POINT not the truth

A 200+-finding automated staleness sweep (Haiku scanners, "precision over recall
with command evidence") is compiled by a weaker model; the dispatch brief itself
says "you are the judgment layer". The recurring failures are (a) trusting a
scanner's suggested TARGET, (b) trusting a scanner's stale WARNING-evidence, and
(c) transcribing a fix that mints a NEW false claim. Disciplines that held across
29 files / 120→15 dead-role reduction:

- **A scanner's suggested target can name a symbol that does NOT exist.** The
  scanner conflated a retired Protocol name with the live class:
  `AngularQuadrature.spherical_harmonics` → the scanner said "retarget to
  `orpheus.numerics.quadrature.AngularQuadrature.spherical_harmonics`" but
  `AngularQuadrature` does not exist anywhere (`hasattr` False) — the live class
  is `Quadrature`. ALWAYS `hasattr`/import-verify the SUGGESTED target, not just
  confirm the OLD one is dead. The census resolver (getattr-chain longest
  importable prefix) is the cheap gate: batch-probe every candidate target before
  editing.
- **A scanner's "sphinx-build warns: X not found" evidence can be STALE — check
  the CURRENT clean build.** One finding claimed a `:mod:` bare ref "warns py:mod
  target not found"; the current `-W` build had NO such warning (bare `:mod:legacy_name`
  renders plain-text WITHOUT warning, L-002). REJECT the finding (evidence
  doesn't hold), and if the ref is a page-wide legacy-naming convention used N×,
  leave it (fixing all N is a rename beyond flagged scope) — record the rejection.
- **Reproducing a fix routinely surfaces a NEW false claim the scanner didn't
  flag — fix the CLAIM, not just the role (Cardinal Rule 1 > scope).** Two
  instances THIS pass: (i) a doc said "verified against **SymPy**'s
  `sympy.integrate(...)`" but the live test uses `scipy.integrate.quad` singular
  quadrature — corrected the pillar attribution, not just the test-name role.
  (ii) I nearly wrote "`wigner_seitz_pin_cell` default of 10 fuel + 3 clad + 7
  coolant sub-cells" transcribing a `pwr_pin_equivalent` default — but
  `wigner_seitz_pin_cell` produces region THICKNESSES only; the sub-cell COUNT is
  a `RegionMesh(n_cells=...)` choice at `Mesh1D.from_geometry`. Caught my own
  mid-edit false claim by reading the successor's live body (L-001 applied to my
  OWN draft). RULE: when a retarget crosses a numerical/structural CLAIM, read
  the successor's live def AND re-verify the surrounding claim's truth before
  re-spelling it.
- **THESIS-STALE beats symbol-stale when a whole class/design was deleted.** A
  deleted carrier class (`GeometrySpec`) staled not just its `:class:` refs but
  the SCHEMA table, the migration narrative (prospective "Phase B will split"
  when the split is DONE + the carrier since deleted), the bridge-test bullets,
  AND the rationale section — a full-section rewrite to the current
  `geometry_kind: str` + `to_geometry()` form. Similarly a scattering per-ℓ
  ladder retired for a Funk-Hecke `R∘Λ∘M` kernel staled a `:class:OperatorSum of
  per-ℓ leaves` TABLE ROW (present-tense architecture), not just the Code-Anchors
  entry. Read the ENCLOSING claim's premise against live architecture (L-020),
  not just the flagged line.
- **A `:ref:` anchor placed AFTER a section title does not resolve as a title
  target — put `.. _label:` ABOVE the title.** The only real WARNING I introduced:
  a forward `:ref:` to an anchor I'd defined between a title and its body. Move
  the anchor above the underlined title (blank line between) so `:ref:` picks up
  the section title.
- **Mechanical families are the bulk win — replace_all on unique multi-char
  strings, count-asserted.** CP/MoC test-filename drift (`test_cp_*.py`/`test_moc_*.py`
  → `tests/{cp,moc}/test_*.py`), a module rename (`test_galerkin_spectral_symbolic`
  → `test_carlvik_galerkin_symbolic`, ×8, fn names unchanged), short-path role
  normalization (`~geometry/numerics/data/sn.*`→`~orpheus.*`), numbered-dir relics
  (`07.Thermal.Hydraulics/...`→`orpheus/thermal_hydraulics/solver.py`). Use a
  count-asserting Python script (assert count==1 for uniques, log count>1 for
  known-duplicates) on the LARGE files where reading 20k lines is impractical —
  it's auditable and can't introduce a transcription error. Order matters:
  replace the LONGER stem first (`thermal_hydraulics_dae.py` before
  `thermal_hydraulics.py`) even when disjoint, to be safe.
- **The residual census MUST be fully attributed.** After the sweep, every
  remaining dead-role census hit must map to a KNOWN false-positive class
  (dataclass-field/ctor-param without default → resolver getattr-chain
  limitation; historical framing "no longer exists"; planned/future `(planned)`/
  "scheduled to promote"). Enumerate them; re-read the doc context of any you're
  unsure of; list any that DON'T fit and fix them. 15/15 residuals attributed
  this pass — none required fixing.

How to apply: for any bulk-scanner-compiled fix job, batch-probe every SUGGESTED
target (not just the old one) against live code FIRST; reject findings whose
build-warning evidence the current clean build contradicts; when a retarget
crosses a numerical/structural claim, read the successor's live body and
re-verify the claim; rewrite the enclosing THESIS when a deleted class/design
staled the whole section; and attribute every census residual to a known FP class.

---

## L-022 — The pedagogical RESTRUCTURE + cross-page THEORY-EVICTION pass: keep-anchor makes it ref-safe; the real traps are marker-depth shift, the part-boundary blank line, and the general-vs-consumer split

A page RENAME + section-tree reorg that also EVICTS large theory blocks
from a sibling page into it (here `galerkin_projection.rst` → `frame.rst`,
retitled + reorganized into a Petrov-Galerkin-first tree, absorbing the SN
page's ~2100-line homogenization + condensation GENERAL theory). Distinct
from L-012 (merging a re-staged BRANCH diff into a diverged tree): this is
a clean intra-repo relocation + pedagogical reflow. Disciplines:

- **A cross-page THEORY MOVE is fundamentally ref-safe when you KEEP the
  labels (don't rename them) — Sphinx labels, eq-labels, AND citations
  all resolve GLOBALLY cross-doc.** Moving a `.. _anchor:` + its content
  to another file keeps every `:ref:`/`:eq:` working with ZERO referrer
  edits, as long as each label stays defined exactly ONCE (move, don't
  copy). This is L-007's keep-the-anchor applied to a relocation: the
  brief's "move the label with the content and fix referrers" needs NO
  referrer fixes at all. Proof-of-pattern for citations: `[Hebert2009]_`
  was already defined in `collision_probability.rst` and used in
  `discrete_ordinates.rst` — a live cross-doc citation — so migrating
  `[WIMSD]_`/`[Rahnema2008]_` REFERENCES into a third page resolves to
  their definitions on the SN page identically (verify with the build).
  The ONLY rename that breaks refs is the DOC name (`:doc:X`): fix the
  toctree entry + every `:doc:old` (3 sites here); the page's own
  `:ref:`-label (`galerkin-projection`) is KEPT so its 6 external
  referrers need no touch.
- **The PART-BOUNDARY blank-line trap (the one real build break).** When
  you assemble a page programmatically by concatenating slices, a slice
  that ENDS in content (a migrated block sliced to its last content line,
  no trailing blank) joined directly before the next part's `.. _anchor:`
  GLUES the anchor to the preceding paragraph → the label silently fails
  to register. The symptom is **"undefined label"** (NOT "duplicate") at
  every referrer, even though `grep` shows the anchor present and at
  column 0. Fix: join parts with `\n\n` (guarantee ≥1 blank everywhere)
  or ensure every content-ending slice carries a trailing blank; then a
  triple-blank normalizer caps runs at 2. A col-0 grep for
  "`.. _x:` whose previous line is non-blank" is the pre-build catcher.
- **Marker-depth shift when content re-nests under a DEEPER parent.**
  Evicted `=`-level sections whose subsections were `-`/`~` land under a
  new `-` subsection (`§2c Applied to …`), so every migrated underline
  demotes one level (`-`→`~`, `~`→`^`), LENGTH-PRESERVING (char-for-char
  swap keeps the code-point count matching the unchanged title, L-009).
  Detect a section underline robustly as a **col-0** all-one-marker line
  (len≥4) whose previous **col-0** line is a plain title (non-blank, not
  `..`, not itself all-marker) — col-0 disambiguates it from bullet `-`,
  math, and `* -` list-table rows. Strip the evicted block's top `=`
  title (hand-write the new `-` title + KEEP its anchor); shift only the
  body.
- **The general-theory vs consumer-orchestration split is the judgment
  call — prefer MOVING general theory, keep the consuming stub lean.**
  Homog/cond theory (rate preservation, the PG-frame derivation, the
  metric-fold-vs-bilinear adjoint argument, fractional-overlap, the
  asymmetry law, AND the verification gates — they verify the GENERAL
  property even when the tests live under `tests.sn.*`) → the theory
  page. Only the SN-LAYER orchestration stays in the stub: which driver
  invokes it (`Solution.homogenize`→`MaterialMesh`→re-promote loop;
  `Solution.condense`→per-material representative spectrum→`dict[int,
  Mixture]`), plus the ONE SN-specific equation
  (`energy-condensation-representative-spectrum` — moved to the stub, not
  the theory page) + `:doc:` links to the full treatment. Split the
  evicted section's INTRO too: its general "what X is" framing →
  theory page; its "in ORPHEUS it lives as `Driver.verb` returning T"
  sentences → stub.
- **PROMOTING a buried `.. note::` to a proper subsection (the crux
  content):** preserve the argument VERBATIM (inline math copied
  exactly), convert `.. note::` prose to subsection prose, KEEP its
  `.. _anchor:` (referenced from elsewhere — here from the "unifying
  principle" §), add ONLY the ONE current-architecture design-rationale
  sentence the brief asks for ("the frame was first posed as Galerkin;
  the adjoint requirement forced the re-posing as Petrov-Galerkin" — a
  REASON, not dated process-narrative, L-010), and fix its now-intra-page
  `(:doc:otherpage)` parentheticals.
- **Clean up `(:doc:sibling)` parentheticals that became intra-page.**
  Every `:ref:X (:doc:discrete_ordinates)` where X migrated INTO this page
  is now a wrong forward-pointer (the reader is sent to the wrong page).
  `-W` does NOT catch it (the `:ref:` still resolves globally; only the
  parenthetical lies). Grep `discrete_ordinates` in the destination page
  → 0 after the pass is the gate. Distinguish from a `:ref:` to SN
  content that STAYS (`sn-scattering-adjoint`) — that keeps its cross-doc
  form.
- **Mechanics that held:** programmatic slice-and-reassemble with
  boundary ASSERTS (`F[idx]==expected_title`) fail-loud on line drift;
  compute the new strings and run ALL structural asserts on the in-memory
  result BEFORE writing either file (a failed assert then leaves the tree
  untouched — no `git checkout` recovery needed, process-discipline);
  assert label counts (each verifies-target defined once, in the
  destination; the SN-only eq NOT leaked), the new headers present, the
  retitled originals GONE, `discrete_ordinates`∉destination. The clean
  `-W` build + the regenerated `matrix.rst` mapping every moved
  verifies-target to its tests (17/8/3) is the final proof no edge
  orphaned.

How to apply: for a rename+reorg+eviction, KEEP labels (move don't copy →
ref-safe); fix only `:doc:` (toctree + `:doc:old` sites); `\n\n`-join or
trailing-blank every slice (the glued-anchor→undefined-label trap);
depth-shift migrated underlines col-0-detected + length-preserving;
move general theory + verification, keep a lean SN-orchestration stub with
the one SN-specific eq; promote a buried crux note verbatim + one
rationale sentence; scrub now-intra-page `(:doc:sibling)`; gate on
in-memory asserts before write + the clean `-W` build + the matrix.

---

## L-023 — The template-skeleton ADDITIVE front-matter pass: machine-header dropdown, Key-Facts+Overview→Synopsis fold, gotchas consolidation, essay eviction — under a HARD "don't reorder the middle" non-goal

Distinct from L-022 (a rename+reorg+cross-page eviction): here the SAME
flagship page gets the 9-section template's FRONT MATTER imposed
ADDITIVELY, with an explicit SCOPE non-goal forbidding physical reorder of
the large middle blocks (the aggressive re-level deferred to a later
phase). The job is: add §1 machine header, fold the front matter into §2
Synopsis, consolidate scattered gotchas at the tail (§8), stub §5/§6 with
automation-pending notes, and evict the narrative history essays (§9 keeps
only the changelog). Disciplines:

- **Machine header = a collapsed sphinx-design `.. dropdown::` wrapping a
  `.. code-block:: yaml`, NOT a bespoke directive.** VERIFY the intended
  ingestion directive is unregistered first (`grep add_directive` in the
  INSTALLED pkg — here `sphinxcontrib.nexus` registers `nexus-graph` /
  `verifies` / `implements`, NOT `nexus-meta`); an unknown directive fails
  `-W`. The YAML-in-code-block sidesteps EVERY RST-parse hazard (unicode
  μ/Σ/ᵀ/≡/→/∈, `#` comments, quotes, no roles/labels) because a code-block
  is literal text. Author only what the graph can't cheaply derive
  (conventions, invariants, operator glossary, retrieval aliases); keep
  entry_points/key_types/governing_equation MINIMAL (Nexus derives the
  fuller lists). Populate conventions/invariants FROM the current Key-Facts
  bullets, each fact verified against live code (L-001) — the header is
  CURRENT structured fact, so a stale convention there is a Rule-1 bug.
- **Key Facts + Overview → ONE Synopsis (front-matter FOLD, not two
  sections).** The structured convention/invariant bullets become the
  machine-header YAML (data for the machine); the prose framing becomes a
  dense, NAMED, retrieval-targeted synopsis (the primary embedding target —
  cite methods/operators/decisions BY NAME); the load-bearing clickable
  `:ref:` navigation list stays as a `.. admonition:: Conventions` block
  (the data moved to YAML, but the reader still needs the nav — keeping
  both is NOT a twin-source-of-truth violation: YAML serves the machine,
  the admonition serves the human). Fold Overview IN (don't keep a near-
  duplicate section beside the synopsis — that WOULD be a twin path).
  PRESERVE every citation usage (`[Bailey…]_` etc.) so no definition
  orphans. FLAG the two-sections→one fold for main-agent review (it is the
  one consolidation judgment call in an otherwise-additive pass).
- **`autosectionlabel` OFF ⇒ a section-heading RENAME is fully ref-safe**
  (no implicit anchor exists to break; only explicit `.. _label:` anchors
  are targets). Confirm it's off (`grep autosectionlabel docs/conf.py`) +
  grep the tree for inbound `:ref:` to the old heading's slug (here: none
  to Key-Facts/Overview) before renaming Key Facts → Synopsis.
- **Essay eviction: an intra-page `:ref:` to a deleted essay anchor IS
  `-W`-CAUGHT (unlike a cross-doc `:ref:`, which renders plain-text —
  L-002).** Grep referrers to BOTH essay anchors BEFORE deleting: essay-1's
  anchor had zero referrers (safe outright delete); essay-2's anchor had a
  single INTRA-page referrer, so deleting it would dangle → `-W` fail.
  Repoint the referrer to the DISTILLED-gotcha anchor — the distillation
  (the "why the two errors cancel for homogeneous problems" →
  homogeneous/uniform-rescale gotcha) IS the semantic successor of the
  evicted essay, so the repoint is meaning-preserving, not a band-aid.
  Paste each evicted essay's FULL text into the return for issue
  relocation (git also preserves it) — the outcome already lives in the
  changelog + ERR catalog, but the narrative goes to the originating issue.
- **HARD SCOPE non-goal vs a deliverable's own example → the non-goal
  governs; point, don't move; FLAG.** §D said "move the scattered `~`
  gotcha subsections", naming one at ~L11835 — but that one sits INSIDE the
  SCOPE section's protected range ("do NOT touch the Sweep mega-section
  internals L2534–11895, that is Phase 1d"). When a hard non-goal collides
  with a deliverable's example, the non-goal wins: LEAVE the protected-range
  item, add a `.. seealso::` pointer to its anchor from the consolidated
  section (consolidation-by-discovery), FLAG for the main agent. Extra
  confirmation here: the sweep gotcha's `.. _sn-282-gotchas:` anchor was
  referenced ONLY from within the protected range — moving it would drag
  refs across the Phase-1d boundary. The OTHER named gotcha (~L15444, under
  SNSolver, OUTSIDE the range, no anchor, no referrers) moved cleanly.
- **The three degeneracy gotchas were `**Gotcha**:` BULLETS in Key Facts,
  not a section** — they are load-bearing facts that belong in §8, so lift
  them OUT of the front matter INTO the Gotchas section (as
  consequence→manifests→catcher warning boxes), don't leave them to
  duplicate in the synopsis. Verify the named catcher tests EXIST
  (`grep def test_…` — both `test_dd_per_cell_recurrence_matches_symbolic_derivation`
  and `test_heterogeneous_absolute_keff` did; the L0 one had been RENAMED
  by an earlier fold, so cite the successor the evicted essay already
  named, L-001).
- **§5/§6 stub notes are ADDITIVE `.. note::` blocks at the section HEAD,
  no restructure.** Each names the blocking Nexus issue (flow-graph
  nexus#20 for the Implementation-map/Architecture section; label↔test
  linking for the Verification slice) + "hand-authored until then" + a live
  `:doc:` to the real surface (`../api/numerics`, `../verification/matrix`
  — verify the target exists).
- **Mechanics (L-022 reused):** one programmatic slice-and-reassemble
  script — content-anchored `.index()`/count-asserted `.replace()`,
  code-point underlines via `len(title)`, structural asserts (new anchors
  ==1, old anchors ==0, dangling-ref ==0, machine-header<synopsis<Architecture,
  moved-bullet ==1) that FAIL LOUD before any write (no partial write, no
  git-checkout recovery). Extract the moved gotcha bullets by substring
  (between markers) so no retype. Acceptance = the `-E -W` build EXIT=0 with
  0 WARNING/ERROR/CRITICAL (this branch's Phase-0/1a baseline is a CLEAN
  build, not 1) + HTML render audit (dropdown `sd-*` classes present, new
  `id=` anchors present, repointed `:ref:` renders `<a href="#…gotcha">`).

How to apply: for a template-skeleton additive pass, (1) machine header =
collapsed dropdown + YAML code-block (verify the directive is
unregistered); (2) fold Key-Facts+Overview → one named Synopsis (structured
→ YAML, prose → synopsis, nav-refs → Conventions admonition, flag the
fold); (3) evict essays after grepping BOTH anchors' referrers (intra-page
`:ref:` IS `-W`-caught → repoint to the distilled gotcha), paste full text
into the return; (4) when a hard non-goal collides with a move instruction,
leave-and-point-and-flag; (5) programmatic slice + asserts + clean `-W`
build + HTML audit.

---

## L-024 — In a structural chapter-split, the single-homed check is on the anchor DEFINITION, not raw label mentions: a router page KEEPS its bare `:ref:` back-refs to labels it exported

The `sn_split_catalog.md` STEP-5 wording ("`grep -c '<label>'` in
index.rst MUST be 0") is a *proxy* that only holds when EVERY inbound
ref to the moved label lives in a DIFFERENT file — true for ch3
(`quadrature-types`, sole inbound from MoC), FALSE the moment the
exported label is back-referenced from the router page itself. The
load-bearing check is the anchor **DEFINITION**: `grep -c '^\.\.
_<label>:' index.rst` == 0 (source no longer owns it) + `grep -rn
'^\.\. _<label>:' docs/` == 1 (new chapter owns it). BC was the first
Phase-C cut to hit this: `boundary-conditions` is back-referenced
from index.rst:171 and :16169 (was :16577, shifted −408), so raw
`grep -c` = 2 while anchor-def = 0. Both survivors are bare
`:ref:`boundary-conditions`` — after the cut they become path-immune
CROSS-doc refs (resolve globally, NO `-W` warning; L35 family). Do
NOT "recut" on a nonzero raw count — inspect WHICH lines matched; a
bare `:ref:` with no directional word and no `:doc:` page-qualifier
STAYS. The genuine recut triggers are exactly two: a surviving `..
_<label>:` DEFINITION in source, or a phantom `duplicate label` under
`-E` (L36).

Second BC-specific trap: a bystander page-qualifier on the general
foundations page — `at :ref:`bc-tensor-decompositions` (in
:doc:`/theory/methods/sn/index`)` — is a TRUE falsehood (L35 case c),
even though `-W` stays silent. The `:ref:` is path-immune, but the
adjacent `:doc:` NAMES the old home; `:doc:` targets ARE
path-sensitive, so after the cut the prose sends the reader to the
page the label just LEFT. Repoint the `:doc:` to the new chapter
(`/theory/methods/sn/boundary_conditions`). NB this creates a brief
window where the `:doc:` DANGLES (target file not yet created) — so
order the moves: fix the qualifier (STEP 1) → create the chapter file
(STEP 3) → build (STEP 6); never build in the gap.

Third trap (Verification chapter, 2625-ln / 41-label cut): a name can
be a `.. _X:` **section anchor** (std:label domain, `:ref:`-resolved)
AND a `.. math:: :label: X` **equation label** (math domain,
`:eq:`-resolved) SIMULTANEOUSLY — DIFFERENT Sphinx domains, so they
coexist with NO duplicate-label warning (a clean `-E` baseline proves
it). When the two live in different parts of the page and the split
moves ONLY one, single-homing MUST check the CORRECT namespace: a
MOVING eq-label is verified with `grep -c ':label: <name>'` (== 0 in
source, == 1 tree-wide), NOT `grep -c '^\.\. _<name>:'` — the latter
shows the same-named SECTION anchor that legitimately STAYS and would
falsely read as a recut trigger. Instance:
`sn-mms-{spherical,cylindrical}-aniso-spatial-convergence` each exist
as a section anchor in the sweep area (stays in index.rst) AND an
eq-label in the Verification block (moves to verification.rst); after
the split `:ref:`→section (index) and `:eq:`→equation (verification)
both resolve globally cross-doc. So a name owning BOTH namespaces is
TWO independent single-home checks, not one — and the `-E` build is
the collision oracle (splitting the pair across two files stays clean
BECAUSE the domains differ). Also note the END-boundary rule for a
next-section whose `.. _anchor:` sits ABOVE its header: stop the cut
BEFORE that anchor (it belongs to the STAYING section), not at
header−1 — else the split drags the next section's anchor into the
new file.

How to apply: for any verbatim chapter cut, run the L35 three-way
grep and disambiguate the moving section-anchor label from any
SUPERSTRING label it collides with (`boundary-conditions` vs the
foundations `theory-boundary-conditions`). Verify single-homing with
`grep -c '^\.\. _<label>:'` (definition), not `grep -c '<label>'`
(mentions). Report the raw count too, naming each surviving line as a
legitimate bare back-ref. Fix ONLY (a) intra-source directional prose
whose target left, (b) moved-block prose whose stay-behind target it
now mis-qualifies, (c) bystander `:doc:` page-qualifiers naming the
old home — leaving bare `:ref:`/`:eq:` (path-immune) untouched.

---

## L-025 — AUTHORING a NEW keystone foundational chapter (gather method-specific verified math → GENERALIZE to the abstract object): the within-doc symbol-collision hunt is the sharpest new gate, and the algebra-of-record is the correctness spine

Distinct task-shape from the split/eviction/close-out passes: writing a
NEW shared method-invariant chapter (`foundations/discretization.rst` —
cell balance + Step/DD/LD, derived once so SN/CP/MoC never re-derive).
The move is *gather the already-verified method-specific derivations,
GENERALIZE them to the abstract object, author generically*. Disciplines:

- **The algebra-of-record is the correctness spine — READ it, then RUN
  it, before writing a single equation.** The SN-specific SymPy
  (`orpheus.derivations.discrete.sn.balance`, 7 `derive_*` foundation
  tests) IS the source of truth for the cell balance / DD / WDD /
  curvilinear / flat-flux math (algebra-of-record skill). Generalizing
  = stripping the SN-specific specialization (face areas → 1 for the
  planar cell, keep the generic `|μ|(ψ_out−ψ_in)+Σ_t h ψ̄ = q h`) while
  quoting the SymPy identity verbatim. `pytest`-run the module (7/7
  green) so the presented equations are grounded, not transcribed
  (Cardinal Rule 1). A DIFFERENT concept in the same page can have a
  DIFFERENT algebra-of-record module (LD's 2×2 is `ld_ubld.py` +
  `_ubld.py`, NOT `balance.py`) — name each precisely in the seealso;
  don't let one `:mod:` cite stand for all the math.
- **A NEW page assembled from MULTIPLE overloading sources is the prime
  site for a WITHIN-document symbol collision — hunt it before the
  build (the build is BLIND to it).** L-011 was a cross-PAGE convention
  sweep; this is the NEW-page twin: when you gather from sources that
  each overload a letter, the SAME letter can carry two meanings in ONE
  page. Here `w` = the cell-average BLEND weight (Step w=1, DD w=½) from
  the scheme code AND `w` = the angular QUADRATURE weight in the SymPy
  `ΔA/w` geometry factor — a genuine ambiguity a teaching page must not
  ship (Cardinal Rule 1). `-W` NEVER catches it (both render fine). The
  gate is a manual re-read hunting every reused glyph across the
  gathered sources; resolve by subscripting the lower-frequency meaning
  (`w_m` for the quadrature weight) + a one-line disambiguation note at
  first collision ("`w_m` is the quadrature weight — *not* the blend
  weight `w`; they share a letter in the literature"). Grep the fix is
  complete (`grep -nE '\\Delta A\}\{w\}'` → 0 residual).
- **A designed-but-UNBUILT scheme is a code LITERAL + a traits-anticipated
  note, never a `:class:` (L-002 forward-ref rule, applied to a THIRD
  sibling).** Step is the open #158 arm — `class Step` exists only as a
  docstring EXAMPLE in `scheme.py`, not an importable class. Grep
  confirms (`grep -rn 'class Step' orpheus/` → only the docstring line).
  So write ``Step`` (literal), derive its math on its own merits (the
  w=1 case of the verified blend framework — code-grounded even without
  a Step-specific SymPy), and add a `.. note:: ORPHEUS status` naming
  the anticipated traits on the base Protocol (`is_positivity_preserving
  = True`, the w=1 upwind blend) + the issue. Honest: documents the
  concept fully without claiming a link to a symbol that isn't there.
- **A LOAD-BEARING worked example gets RUN against the live code before
  it's written (Cardinal Rule 1 / Quality item 6).** The thick-cell
  negative-flux contrast (Step +1/5 positive, DD −1/3 strongly negative,
  LD −1/19 mildly negative — "LD better positivity than DD" made
  concrete) was computed by hand THEN reproduced through the real
  `DiamondDifference.update` / `LinearDiscontinuous.update` (build a
  `StreamingTerms` + `CellVisit` + `UpstreamState` directly; `slab_streaming`
  is a mesh-level factory, not a single-cell builder — construct the
  frozen dataclass fields by hand). Exact fractions matched to machine
  precision → the example is verified, not plausible.
- **The two-axis articulation catch: a "spectrum" parameter is often ONE
  of two orthogonal axes — say so, or a fresh reader over-collapses.**
  The blend weight `w` (Step 1 → DD ½ → LD adaptive 1/(1+k)) orders the
  FACE reconstruction; the MOMENT count (Step/DD: 1, LD: 2) is a SECOND,
  orthogonal axis, and the diffusion limit lives in the moment count NOT
  the blend. "LD→Step as w→1 (thick)" is TRUE of the face blend and
  FALSE of the scheme (LD keeps the slope Step lacks). An `.. important::`
  block pinning "the blend is one axis, the moment count another"
  pre-empts the natural mis-read — the doc-side inoculation move
  (vv-principles Mode-10 doc companion generalized to any seductive
  over-collapse).
- **Zero-warning gate on a NEW standalone page ⇒ ALL references PLAIN-TEXT
  (no `.. [Key]` defs, no `[Key]_` cites).** L-006 says standalone pages
  accept duplicate-citation warnings as a trade-off — but a strict
  "0 warnings" brief forbids introducing ANY (a `[Key]_` cite is
  undefined on the new page → warns; a `.. [Key]` def collides with the
  existing def elsewhere → dup-citation warns). Resolve by citing every
  reference as PLAIN TEXT in a Literature `.. list-table::` (author,
  year, title, journal, the load-bearing eq/§ numbers inline) — zero
  citation machinery, zero warnings, AND higher articulation (the
  equation numbers sit in the prose). Mixing plain-text + `[Key]_` is
  inconsistent; go all-plain-text.

How to apply: for a NEW foundational chapter — (1) read+RUN the
algebra-of-record module(s), one per distinct concept; (2) get the
disassembly/outline right before prose (the 7-part skeleton), namespace
every section anchor AND every eq-label to the page (`<page>-` prefix,
grep-confirmed collision-free); (3) hunt within-doc symbol collisions
across the gathered sources (subscript the rarer meaning + a note);
(4) code-literal any unbuilt sibling; (5) RUN every load-bearing worked
number through the live code; (6) all-plain-text references; (7) tag
every eq-label `.. vv-status: <label> documented` with a rationale
comment naming the SymPy/foundation gate (L-004 — they land in
Documented-only, not flagged as unverified solver claims).

---

## L-026 — The split-to-new-page pattern (extract N contiguous H1 sections into a foundations page): identify by STABLE title, slice PROGRAMMATICALLY, carry every label; and the build-INVISIBLE f-string LaTeX-brace trap in the AUTHORED header

The #231 corpus campaign repeatedly splits a monolith page's advanced
deep-dives into their own pages (the operator_algebra reframe alone owes
`operator_inverse_family` / `operator_tensor_network` /
`coupled_block_operator` / `field_algebra` / `wavefront_cochain`). The
mechanical MOVE is a solved recipe; the sharp, build-invisible hazard is
in the AUTHORED header you wrap around it.

- **Locate by STABLE title, never the plan's line numbers** (they drift
  as the monolith is edited). `grep -n "<exact title>"` the four/N
  section titles + the FOLLOWING section's title (the exclusive upper
  bound). Then **prove contiguity**: `awk` the full-width `===` H1
  underline rows in the range and confirm ONLY your N titles appear —
  nothing else lives between the first and last section.
- **Inventory the traveling labels FIRST.** `awk` the range for BOTH
  `^\.\. _<label>:` (section/ref anchors) AND `:label:` (equation
  labels). ALL of them travel verbatim with the content — the plan
  usually names only the headline anchors ("and others"); the grep is
  authoritative. Labels are **path-immune**: inbound `:ref:`/`:eq:` from
  other docs resolve by NAME wherever the label lives, so they survive
  the move with zero edits on the consuming pages.
- **Slice PROGRAMMATICALLY (L-012), never hand-retype.** A Python splice
  reads the source, `block = lines[start-1:end]`, writes `header + intro
  + "".join(block)` to the new page and `prefix + pointer + suffix` back
  to the source. **Guard-assert the boundaries on the LIVE file**: first
  block line == the start `.. _<label>:`, the line after the block ==
  the next section's `.. _<label>:`, and `len(block)`. The verbatim
  block via `"".join` is transcription-safe.
- **⚠ THE TRAP — a Python f-string mangles LaTeX braces in the AUTHORED
  header, and `-W` is BLIND to it.** The moved block is safe (`"".join`,
  no interpolation); the header/intro YOU author is the risk. In an
  `f"""..."""`, ``:math:`A^{-1}` `` becomes ``A^-1`` (the f-string
  evaluates `{-1}` → the string "-1"); `\tfrac{1}{k}`, `{\rm loss}`,
  `\frac{a}{b}` corrupt identically. The mangled ``A^-1`` is **valid
  LaTeX math** — it renders (wrongly, as A⁻1 not A⁻¹), so `-W` never
  warns. This is a Cardinal-Rule-1 teaching defect on a NEW page.
  **Defense:** prefer plain concatenation (not an f-string) for
  math-bearing prose, OR escape every literal brace `{{ }}`; and ALWAYS
  grep the AUTHORED region before building — `grep -nE '\^-1|\{[^}]*\}'`
  over the header/intro lines only (the block's correct `A^{-1}` must
  not be touched) — then eyeball the rendered head. (Worked: split #1
  operator_inverse_family — the intro's two `A^{-1}` mangled to `A^-1`;
  caught by the visual head-read + a `grep 'A\^-1'`, fixed before the
  final build. The 1339-line verbatim block was untouched.) This is
  L-002 (build-blind correctness) ∩ L-012 (programmatic splice).
- **New page shape** (model `discretization.rst`): top `.. _<label>:` →
  over+under `=` title (size the bar with `len(title)` in CODE POINTS,
  L-009) → `.. contents::` `:local:` `:depth: 2` → a PROVISIONAL
  `.. dropdown:: Machine header — \`\`nexus-meta\`\` schema (PROVISIONAL)`
  `:color: muted` with a `code-block:: yaml` → a 1–2¶ intro that links
  UP to the parent (`:doc:`) and gives a semantic-TOC of the N sections
  (same-page `:ref:` to each moved anchor — guaranteed to resolve). The
  source's excised block becomes a **1-paragraph `:doc:` pointer**
  section (do NOT reuse the new page's top label). Wire the new page
  into the `index.rst` toctree (right after its sibling) AND add the
  intro `list-table` row.
- **Orphaned-HTML audit noise.** A `-E` build regenerates every LIVE
  page but does NOT garbage-collect HTML from renamed/deleted sources.
  A safety grep for `oldpage.html#<moved-label>` in the built tree WILL
  hit those orphans and look like a live stale ref. **Discriminate by
  "does the source `.rst` still exist?"** — absent ⇒ stale build
  artifact, out of scope (do not chase it); present ⇒ a real stale ref
  to fix. (Worked: `discrete_ordinates.html`/`loss_representations.html`
  orphans from prior renames carried the pre-move `operator_algebra.html#green-operator`
  href; their sources were long gone — irrelevant to the split.)
- **Split-#2 calibrations (extracting a SINGLE H1 section, vs #1's four).**
  (a) **Single-H1 → near-duplicate page/section title is ACCEPTED.** When the
  extracted block is ONE H1 section, the new page-top title and the block's own
  verbatim H1 are near-identical (page "Tensor-Network Decomposition of S_N
  Operators" over section "Tensor-Network Decomposition of SN Operators (Wave
  T)"). Do NOT "fix" it by rewriting the block's H1 — verbatim is the rule; the
  page-title + block-H1 pair is exactly split-#1's structure (page-title `===`
  followed by moved `===` sections) and builds `-E -W` clean. The two anchors
  differ (`operator-tensor-network` vs `wave-t-tensor-network`) and the titles
  aren't textually identical, so no duplicate-implicit-target warning.
  (b) **Inline `:sub:` in an over+under `=` title is a PROVEN pattern** — the SN
  book's own H1 is `Discrete Ordinates Method (S\ :sub:`N`)`. Mirror it
  (`... of S\ :sub:`N` Operators`), size the bar with `len(raw_title)` in code
  points (L-009; the role markup counts — 53 here), NO redundant `**bold**`
  (titles are already styled; the house convention uses `:sub:`/`:math:` bare).
  (c) **Directional-prose (L35) is mostly a NO-OP — fix ONLY phrases pointing at
  content that STAYS on the source page.** A grep of the block for
  above/below/earlier/later flagged 6 hits; FIVE were intra-block ("MA-Q1
  master condition above", "equations below", "coupling below" — all reference
  sibling subsections/eqs that TRAVEL with the block, so the relative direction
  is preserved on the new page) and one temporal ("later retired at commit").
  Exactly ONE was a genuine cross-page pointer ("operator-algebra types
  documented above" → the tensor-product TYPES that stay on the parent page) →
  rewrote to "documented in :doc:`/theory/foundations/operator_algebra`" (verify
  the target genuinely stays: grep the types on the source page OUTSIDE the
  moved range first). Calibration: in a self-contained deep-dive block, ~1-in-6
  directional phrases need a fix; classify each by whether its referent travels
  or stays, don't blanket-rewrite.
- **Split-#3 calibrations (g-adjoint → `operator_adjoint.rst`; the recipe's THIRD
  single-H1 run held perfectly, ZERO L35 fixes).**
  (a) **STRONGEST f-string-trap defense — author the header/intro AND the pointer
  as pure literals via the Write tool, not through ANY Python string.** The trap is
  a Python string layer corrupting LaTeX; eliminate the layer. `Write` the head to
  `/tmp/head.rst` and the pointer to `/tmp/pointer.rst` — the Write `content` is a
  pure literal (no f-string interpolation AND no backslash-escape processing, so
  `\frac`/`\nabla`/`\dagger`/`{-1}` are ALL safe — even safer than a raw
  `r"""..."""`). The Python assembler then ONLY does read+slice+concat+write; its
  sole string literals are file paths and the guard-assert boundary strings
  (`.. _g-adjoint:` + newline), NEVER math. Still run the mangle-grep (`A\^-1|G\^-1`)
  on the temp files as the gate (confirmed clean). Zero Python string layer over
  authored math = the trap cannot fire.
  (b) **A page's OWN other sections can `:ref:` the to-be-moved label — those become
  source→new-page cross-refs and are path-immune too.** Beyond EXTERNAL inbound refs,
  `grep <label>` the SOURCE page OUTSIDE the moved block: operator_algebra.rst had 5
  in-section `:ref:`g-adjoint`` (Key Facts + eigenvalue-posing region) that, post-move,
  resolve cross-page (HTML-audited: all 5 href `operator_adjoint.html#g-adjoint`, ZERO
  stale to the old location). No edits — same-file path-immunity works exactly like
  cross-file. Orphan gate: `grep -oE 'href="[^"]*oldpage.html#<label>"'` over the WHOLE
  built tree must be 0, then discriminate any hit by source-`.rst` existence.
  (c) **L35 gains two more no-fix categories: the TEMPORAL false-positive and the GONE
  referent.** Split #2 had 1 fix; #3 had ZERO. The 5 grep hits split: 3 intra-block
  ("(above)" / "following argument" / "below (≤3.6e-15)" — travel), 1 **temporal** ("an
  earlier version of this section" — a prior REVISION, not a spatial section; no fix),
  and 1 **gone referent** ("the reachability table below … see its retraction note" —
  the below/retraction-note direction still resolves to the intra-block Supersession
  `.. note::` that travels, but the "reachability table" itself no longer exists
  ANYWHERE on the page, removed in a prior edit). A gone referent is neither travels nor
  stays → OUT of L35 scope (L35 fixes only move-INDUCED breaks) → FLAG as pre-existing
  staleness (L-007), do NOT rewrite. Add temporal + gone to the per-phrase triage so
  the grep's false-positives are dispatched without a fix.
- **Split-#4/#5 calibrations (affine field algebra → `field_algebra.rst`; wavefront cochain →
  `wavefront_cochain.rst`; the brief's line-range OVERSHOT the genuine section BOTH times — the
  contiguity proof EARNED its keep, and this is now a STRUCTURAL, RECURRING pattern, not a
  one-off).**
  (a) **"Prove contiguity" must count ALL H1 `===` underlines, not just anchored ones — an
  ANCHORLESS sibling H1 inside the brief's range is INVISIBLE to the anchor-grep the brief
  author set the boundary with.** Split #4's brief "3844–4383 / up to the next anchor
  `wavefront-flux-cochain`" overshot an anchorless "The composite metric adjoint" H1; split
  #5's brief "3875–4434 / up to `coupled-block-operator`" overshot an anchorless "The inverse
  family" H1 — BOTH `:doc:` pointer stubs a PRIOR split left behind, sitting between the target
  section and the next `.. _label:`. A boundary set by "section anchor → next `.. _label:`"
  jumps clean over a stub (stubs carry no anchor). The `awk`-all-`===`-underlines proof (step
  1) CAUGHT both (2 H1s in the range, not 1). **This ALWAYS happens when the extracted section
  is immediately followed by a prior split's leftover pointer-stub H1 — endemic to a multi-split
  campaign; EXPECT it and run the count-all-H1 proof every time** (in split #5 the coordinator
  even pre-warned "confirm no anchorless sibling H1 inside the range"). The contiguity proof is
  NOT a formality, it is the gate for exactly this. (b) **Narrow the
  extraction to the GENUINE titled section when the trailing sibling's prose ties it to the
  SOURCE page.** The stub said "the adjoint face of the operator algebra **on this page**" —
  a bare directional coda (L35) that BREAKS if moved (the operator algebra stays on source),
  AND it is thematically the operator's adjoint, not the flux field algebra → LEAVE on source,
  move only `affine anchor → composite-adjoint H1`. The stable-title method (which L-026
  PRIORITIZES over line numbers) ends the section at the next H1; the brief's line-range was
  the author's anchor-scan estimate. FLAG the narrowing prominently — the coordinator set the
  wider range and reviews before commit. (c) **A three-way symbol collision folded into the
  split is a mini-L-011 done via the Edit tool BEFORE slicing (brace-safe), NOT in the
  f-string-risky head.** The reframe reserved `A` = full operator `L+C−S−B`; the block misused
  `A` for the sub-composite `L+C` (apply/solve headline + SI-increment machinery `M=A⁻¹(S+B)`,
  `Δψ=A⁻¹r̃`) AND the affine SPACE. Classify each `A`: full-op defect/solve → keep `A` (the
  residual `r=Aψ−q` was already full — reframe the headline TO match it); SI-increment
  machinery → genuinely the SWEEP `(L+C)⁻¹` (the honest spelling once `A` is the full solve —
  else Cardinal-Rule-1-wrong); affine space → `\mathbb{A}`. Apply on the SOURCE via Edit tool
  (exact-literal replacement is brace-safe) BEFORE the programmatic slice; the reconciled block
  then travels verbatim. Escape hatch (does the affine ARGUMENT require `L+C`?) did NOT fire —
  the torsor structure rests on flux-state geometry (no natural zero), independent of which
  operator connects the universes; test the ARGUMENT, not the symbol. (d) **Two guard-gotchas:**
  the mangle-grep FALSE-positives on `\mathbb{A}` + closing backtick (`mathbb\{A\}[^ ]` matches
  the `` ` `` after intact `{A}`) — the REAL gate is "any BARE `A^-1` (no braces)?" = ZERO; and a
  `.count(r"\mathbb{A}")` guard-assert counts OCCURRENCES not grep-LINES (a
  `\mathbb{A}\times V\to \mathbb{A}` line carries 2), so an `==10`-from-a-line-grep assert
  RED-flags its own miscount at 11 — a GOOD failure (asserts run before any write; content was
  right, the assert wrong). Fix the number, re-run; never loosen the assert to make it pass.
- **Split-#3/final calibration (coupled block operator → `coupled_block_operator.rst`; the NEW,
  build-INVISIBLE break class my #1/#2 closeouts MISSED — "named-source-page" consuming prose).**
  The inbound-ref audit is NOT just "does the `:ref:`/`:eq:` resolve?" — it ALWAYS does
  (path-immune, auto-repoints to wherever the label now lives). The break is in CONSUMING-FILE
  PROSE that NAMES the source page: ``see :ref:`X` in :doc:`.../operator_algebra` `` — after the
  move the `:ref:` links to the NEW page while the adjacent "in operator_algebra" sends the
  reader to the OLD page (now just a pointer stub). `-W` is BLIND (the ref resolves; only the
  prose lies) — a Cardinal-Rule-1 staleness the build never catches. **The L35 scan MUST extend
  beyond the moved block + source boundary to a whole-tree sweep of every consuming file** for
  ``:ref:`<moved-label>` … :doc:`<source-page>` `` (whitespace-FLATTENED / multi-line-aware —
  the "in" and the `:doc:` routinely wrap across two RST lines, so a line-grep misses them; use
  a `re.sub(r'\s+',' ',text)` python scan). Repoint the stale `:doc:` page-pointer to the new
  page; leave bare `:ref:`s (no page-pointer prose) and "see X **for the** …" phrasings alone.
  This session the sweep caught 2 in `sn/history.rst` (this split) AND a LEFTOVER from the
  committed split #2 (`sn/index.rst`: ``wavefront-flux-cochain` in :doc:`operator_algebra` ``) —
  fixed both, flagged the #2 leftover. LESSON: run this flattened consuming-prose sweep on EVERY
  split (the "all inbound refs resolve cross-page" HTML audit is necessary but NOT sufficient —
  it proves the LINKS work, not that the PROSE naming the old page is current). When the boundary
  is a real anchor (not an anchorless stub), the contiguity proof is clean — split #3 had exactly
  ONE H1 as the coordinator pre-scanned; still ran the count-all-H1 proof to confirm.
- **Split-#6 calibration (BISECTION: one page → TWO new children, source page DELETED — NOT the
  "extract-N-into-one-page, source-survives-as-a-`:doc:`-stub" model of #1–#5). Two structural
  facts the survivor-stub model never surfaces.** (a) **The source page-LABEL is RE-HOMED to the
  MAJORITY child, not left on a stub.** The source dissolves entirely, so its top
  `.. _<page-label>:` travels (verbatim) above the majority child's NEW title; every external
  inbound `:ref:` (audit the count — 9 sites here) auto-repoints to the child's new FILENAME with
  ZERO edits on the consumers (label path-immunity, exactly like a moved section anchor — HTML-audit
  that they land on `<majority-child>.html#<page-label>`, not the deleted file). The MINORITY child
  gets a BRAND-NEW label (absent pre-split). Single-homing is still the anchor-DEFINITION check
  (L-024): each `^\.\. _<name>:` == 1 tree-wide. (b) **Extracting a MIDDLE H1 while KEEPING its
  trailing H2 subsection → the H2 AUTO-REPARENTS to the PRECEDING H1.** When the brief keeps a
  subsection whose PARENT H1 is the one being extracted (keep "Cell-flattening invariant" H2, extract
  its parent "Cross-section convention" H1), removing the parent header re-attaches the orphaned H2 to
  the nearest preceding higher section (the previous chapter's H1 — "Derivation"). VERIFY TWO things:
  no title-level SKIP results (H1→H2 is legal and builds clean; H1→H3 would warn), AND the reparent is
  SEMANTICALLY intended (the kept subsection must belong under its new parent — the layout round-trip
  genuinely IS derivation content, even though it uses `sig_t` as the example). The brief's
  heading-LEVEL claim can be flatly wrong ("an H3 inside the Derivation chapter" for what the live
  `awk`/grep proves is an H1 SIBLING of Derivation) — L-001: the live underline rows are authoritative,
  and the programmatic guard-asserts on the EXACT header+underline strings catch a drifted boundary
  before any write. (c) **Blank-line glue at each splice follows the file's LOCAL convention** (this
  file: 1 blank between H2 siblings, 2 at H1 transitions) — after removing a middle span, DISCARD the
  bracketing structural blanks so the survivor spacing matches its new neighbours (char-identity applies
  to the moved CONTENT, L-013; the glue matches the destination, L-009). Boundary-crossing `:ref:`
  (a moved span references a label that STAYS, or vice-versa) needs NO syntax change — it silently
  becomes a cross-doc ref and resolves by name; HTML-audit it lands (here the XS page's
  `:ref:`…<sn-cell-flattening-invariant>`` → `indexing_and_layout.html#…`).

How to apply: title-locate → prove contiguity (count ALL `===` H1s) →
grep-inventory every label → programmatic guarded slice → author the
header WITHOUT an f-string over math (or escape braces) and grep it →
build `-E -W` to the unchanged baseline → HTML link-audit the inbound
refs land on the new page → **flattened consuming-prose sweep for
`<moved-label> … in :doc:<source-page>` and repoint the stale
page-pointer** → discriminate orphan artifacts by source existence.

---

## L-027 — The "relocate to page X" brief that the CLOSE READ reveals is ALREADY-fully-on-X → Cardinal-Rule-2 DE-DUPLICATE, not relocate+merge; plus the `ref.ref` caption-gotcha on an alias anchor, and reframe-consistency in a FOLD

A doc-cleanup brief can say "RELOCATE section S from page A to page B,
merging its additive parts in" — and the mandated close read of page B
reveals S is **already fully documented on B**, in equal or greater
detail. The scoping estimate (from a partial read) said "partial
overlap, ~1 additive artifact"; the CLOSE read (of every candidate
landing section) inverts it to "total duplication, ZERO additive."

- **When the content is already canonical on the target, the correct
  action is DE-DUPLICATION, not relocate+merge (Cardinal Rule 2).**
  Replace the source copy with a brief `:doc:` pointer that preserves
  the CONCEPTUAL BRIDGE (why the topic matters to the source page) and
  names the sub-topics, pointing at the canonical target sections; merge
  in NOTHING that already exists. Worked (#231 P4-T3): the "Boundary
  conditions as Wave-0/1 primitives" section on `operator_algebra.rst`
  was ~fully duplicated by `boundary_conditions.rst` — the G_α
  "primitives table" (its supposed one additive artifact) already lived
  there as the richer "SN realization map" list-table (§1794, with α=1
  fast-path columns + a bit-identity note the operator-algebra copy
  lacked); the rank-N eq, the Marshak example, the descriptor-vs-operator
  separation, AND the Wave-11/β1 predecessors were ALL already present.
  So: replace with a `:doc:` pointer, merge nothing.
- **The brief's "additive parts to MERGE IN" list is the SCOPING
  estimate, NOT ground truth — the close read overrides it. FLAG the
  inversion loudly** (the reviewer built the brief on the partial-read
  model). Report each "additive" item as "already at §X of the target,
  deduped" with the section reference, so the reviewer's close review
  can restore any specific piece. This is the T-relocate analogue of the
  split-#4 anchorless-sibling discovery: the contiguity/close-read gate
  is what catches the brief's wrong structural assumption.
- **Carry the moved section's `:ref:`-able labels as ALIAS anchors onto
  the canonical target content** (zero-inbound-ref labels still get
  carried "for outbound-ref integrity" per the brief) — BUT an EQ-label
  (`:label:`, used via `:eq:`) CANNOT be aliased onto a different eq;
  drop it if its eq is duplicated and it has zero `:eq:` refs (flag the
  drop). Std `.. _` anchors alias fine.
- **⚠ NEW `-W`-CAUGHT warning class — `ref.ref` "A title or caption not
  found".** A `.. _label:` placed before a **paragraph** (or any element
  with no title/caption) makes a BARE `:ref:`label`` FAIL under `-W`
  ("Failed to create a cross reference. A title or caption not found").
  Unlike a dead code-xref (plain-text, L-002-silent), THIS one IS gated
  by `-W`. Two fixes: (a) place the alias anchor before a TITLED or
  CAPTIONED element (a section title, or a ``.. list-table:: Caption`` /
  figure — then the caption becomes the link text), OR (b) use
  EXPLICIT-text `:ref:`link text <label>`` (resolves regardless of what
  the label precedes). An anchor before a SECTION TITLE is already safe
  (bare ref gets the title); an anchor before a paragraph/list needs (a)
  or (b). (Worked: `bc-descriptor-tree-vs-operator-tree` before §2419's
  title was fine; `bc-tensor-primitives` before a paragraph warned — I
  moved it before the captioned list-table AND made the pointer
  explicit-text.)
- **Reframe-consistency applies to a FOLD, not only a split/move.** When
  folding a paragraph into a keeper section, the SAME overloaded-symbol
  reconciliation applies: T1's `loss_minus_gains(psi) = A.apply − Σ
  g.apply` used `A` for the sub-composite `(L+C)` (gains = S,B) — the
  pre-reframe collision. Verify against LIVE code (`iteration.py:903` +
  its docstring "the matvec IS the honest `(L+C−S−B)·ψ`"), then fold
  reframe-consistently: spell the sub-composite `(L{+}C)` explicitly and
  identify the result as the full `A = L+C−S−B` applied. A fold is a
  move; a move re-exposes every reconciliation the source had.
- **A stale deferred-follow-up ISSUE-tag is distilled by git, not
  guessed.** T4's "Deferred follow-ups: #260 …, #261 (core relocation…)"
  — `git merge-base --is-ancestor <#261-commit> HEAD` proved #261 landed
  → drop it, keep the still-open #260 (singularize "follow-ups"→
  "follow-up"). Keep the DESIGN RATIONALE ("considered-and-rejected: R/M
  are rank-changing einsums, not valid tensor factors") — only the dated
  tracking tail distills.
- **⚠ Dropping a duplicated EQ-label ALSO drops its `.. vv-status:` — check the
  V&V-matrix consequence and MOVE the status to the survivor.** A de-duplicated
  concept can carry TWO `:label:`s with DIFFERENT V&V status — a `documented`
  twin and an `orphan` twin (same math). Dropping the `documented` one (because
  it is the duplicate) silently DEMOTES the concept to `orphan` (untracked), and
  `-W` is BLIND: the orphan-equation gate is a `docs/verification/matrix.rst`
  REPORT auto-regenerated by `conf.py`'s `generate_matrix` hook, NOT a
  build-breaking check. If the concept is genuinely documented-only (a
  declarative/definitional eq with no numerical result to test — e.g. a
  BC-algebra decomposition), ADD `.. vv-status: <survivor-label> documented` to
  the surviving canonical eq so the accounting is preserved through the de-dup.
  Worked (#231 P4-T3, caught by the main agent in review): dropping
  `bc-rank-n-as-sum-of-products` (documented) left only the orphan
  `bc-rank-n-tensor-decomposition` — fixed by marking the survivor `documented`.
  The retirement-audit's "handle the V&V edges" includes the vv-status
  directive, not just the `:eq:` refs.

How to apply: on a "relocate to X" brief, CLOSE-read every candidate
landing section on X FIRST; if the content is already there, de-dup by
`:doc:` pointer (merge nothing) and FLAG the inversion; carry `.. _`
aliases onto the canonical content (before a titled/captioned element,
or use explicit-text refs to dodge `ref.ref`); a fold is a move — apply
the reframe reconciliation; distill stale issue-tags by `git
merge-base`, keeping the design rationale; when dropping a duplicated
eq-label, MOVE its `.. vv-status:` to the survivor (the `-W`-blind
orphan-demotion). **Mechanical:** a full `-E`
rebuild here EXCEEDS the 120s foreground cap — use `run_in_background`
for the authoritative gate (a foreground poll-loop gets SIGTERM'd at
2 min, killing the build at the final line before "build succeeded"
prints, so the summary never lands even though zero warnings were
raised).

---

## L-028 — The Key-Facts↔changelog metadata-RELOCATION (L10 Sphinx-as-brain ≠ Sphinx-as-history): strip campaign-provenance from the high-traffic invariant section INTO the page-bottom changelog — move it, don't lose it

A theory page's **Key Facts** (highest-traffic section) accretes
campaign-provenance clauses over a long refactor — each invariant bullet
tagged "(Wave O step B.5.2, Issue #208, ``6ef5063``, 2026-06-03)". The
page-bottom **Development-history changelog** is where that provenance
belongs (L10). The task is a RELOCATION, not a rewrite: the invariant
stays in Key Facts, the campaign-metadata moves to the changelog.

- **The strip-list is EXACT and narrow: commit hashes, round/Wave/Phase
  labels ("Wave O step X", "Phase 5a", "carve P4", "S6.4(f)"), branch
  names, landing dates.** KEEP everything else in place — the invariant
  statement, the production-formula **eq-labels + their ``.. vv-status:``
  comments**, every ``:ref:``, every **active gotcha** ("F never enters
  A", "NOT deprecated", the ``(N,ng,nx,ny)`` convention), AND numerical
  evidence (an "18.3× shrink" is a datum, not campaign-metadata). Verify
  the KEEP-set survives with a grep after (eq-label + vv-status count
  unchanged; gotcha phrases present).
- **Issue-# refs (``#208``) are NOT in the strip-list — KEEP them inline
  and FLAG the decision.** They are lightweight GitHub cross-refs a
  reader values, distinct from dated/hash history; the changelog's Issue
  column carrying the same number is acceptable redundancy. (If the
  reviewer wants them stripped too, that's a one-line follow-up.)
- **"Move it, don't lose it" gates what you may strip: NEVER strip a
  DATED milestone whose provenance has no changelog home.** Map each
  bullet's provenance to a destination FIRST: (a) an EXISTING changelog
  row, (b) one of the NEW rows you add, or (c) NONE. For (a)/(b) strip
  freely. For (c) — a dated milestone missing from the changelog and NOT
  in the sanctioned new-row set (worked: the coupled-block 2026-07-12
  bullet) — KEEP its provenance inline and FLAG it (recommend a row);
  stripping it would DELETE the date/branch, violating the relocation
  principle. A round-label whose milestone IS covered elsewhere (the
  "since stencil-assembly 2b" assembly-axis note, already a changelog
  row) strips cleanly — nothing dated is lost.
- **Verify every commit hash is a HEAD ancestor before citing it in a
  new row** (``git merge-base --is-ancestor <hash> HEAD``); pull the
  ``When`` date from ``git show -s --format=%cs <hash>`` (don't trust the
  bullet's inline date — get it from git). New rows go in reverse-chron,
  matching the existing list-table's 4 columns (When / milestone / Issue
  / Where), iteration-rates omitted per the changelog's own preamble; the
  ``Where`` column carries the stripped hashes (``main (Phase 5a,
  ``hash`` / ``hash``)``). The new-row milestone text is the SAME
  invariant you kept in Key Facts, distilled — so Key Facts and the
  changelog agree, neither is lost.
- **A metadata-strip pass routinely surfaces an ORTHOGONAL staleness in
  the same bullets — FLAG, don't fix (out of scope).** Worked: a
  Key-Facts affine bullet still wrote the affine space as bare ``A``
  while the split-out ``field_algebra.rst`` deep-dive had renamed it to
  ``\mathbb{A}`` (the split-#1 reconciliation never propagated to the
  summary bullet). That is a reframe-consistency fix, not metadata —
  report it as a found-in-passing defect, leave it for a scoped pass.

How to apply: map each bullet's provenance to a changelog destination
(existing row / new row / none); strip only hashes/round-Wave-Phase-
labels/branches/dates that HAVE a home; keep invariants + eq-labels +
vv-status + gotchas + issue-refs + numerical data; keep-and-flag any
dated milestone with no changelog home; git-verify every cited hash is a
HEAD ancestor and git-source its date; grep the KEEP-set is intact;
flag orthogonal staleness found in passing.

---

## L-029 — The additive "surface the taxonomy up front" framing pass: verify the gap is REAL (a sibling taxonomy present ≠ THIS one surfaced), then PREVIEW + `:ref:` to the SSOT — never a twin table

When a scoping pass finds a governing taxonomy (a 3-way type partition,
a classification law) stated only MID-STREAM — in the 2nd of the
sections it frames, so a linear reader meets the 1st with no roadmap —
the fix is ONE short additive framing section, not a reorder. (Meta:
the scoping earned its keep by KILLING the big mechanical move — the
flagged apply/solve↔streaming dependency-INVERSION led the coordinator
to KEEP-the-early-section and ABANDON the relocation, leaving this one
high-value additive item. A load-bearing dependency flag can collapse a
multi-move plan to a single preview.)

- **Escape-hatch FIRST — verify the gap is REAL, and MATCH the specific
  taxonomy, not "is there any framing."** The trap: the upstream
  "already there" candidate (Key Facts) stated a DIFFERENT partition —
  the Representation×Role CARRIER taxonomy — which is NOT the
  Operator/Kernel/Functional CODOMAIN partition. The presence of A
  taxonomy is not the presence of THIS one; two coexist in the same
  page. Read/grep for the EXACT partition before concluding it's
  un-surfaced. If it IS already surfaced clearly upstream, FLAG that
  (decline to add ceremony) rather than force the section — the
  coordinator's explicit escape hatch, and Cardinal Rule 2 (no
  ceremony).
- **SSOT-vs-twin — the preview is a prose ROADMAP + `:ref:`, NEVER a
  copied table.** Naming the three arms in prose and `:ref:`-ing the
  canonical codomain-partition table is a pointer that cannot harmfully
  drift; COPYING the table into the preview is the Cardinal-Rule-2 twin.
  When offered "a light RELOCATION of the existing framing" as the
  elegant alternative (SSOT over preview+ref), prefer the additive
  preview when relocation would (a) touch a section you were told NOT to
  modify, or (b) orphan the framing's double-duty — here the partition
  statement is ALSO the Functional section's OWN opening paragraph, so
  relocating it guts that intro. The SSOT stays where it is; the preview
  points at it.
- **`:ref:` the SSOT SECTION anchor, verified live — not a sub-anchor
  that doesn't exist.** The canonical table lived inside the Functional
  section under an UN-anchored subsection; point `:ref:` at the section
  anchor that HOSTS it (``functional-category``), not the subsection.
  Intra-doc `:ref:` warns if dangling (L-002), so the clean `-E -W`
  build confirms resolution, and the HTML link-audit
  (``href="#functional-category"`` with the target title as link text,
  not a plain ``<code>``) confirms it rendered as a hyperlink.
- **Baseline is FRESH, not frozen.** Measure the `-E` baseline THIS
  session before crediting "count-unchanged" — the AGENT.md "1 warning
  (mesh.py ``:paramref:``)" note was STALE (true baseline 0 on
  ``docs/sn-doc-architecture``; baselines drift 9→1→0). An additive
  preview with no new `:label:`/citation and one live `:ref:` is
  provably warning-neutral; the pre/post SET-diff (both 0) proves it,
  and a pure-additive H1 needs the ``len(title)``-sized underline
  (L-009) — 28 code points here.

How to apply: verify the taxonomy is genuinely un-surfaced upstream
(match the SPECIFIC partition, not any framing; flag-and-decline if it's
already there) → author a prose-roadmap preview naming the arms +
`:ref:` to the SSOT section anchor (never copy the table) → prefer the
additive preview over relocation when relocation touches a frozen or
double-duty section → point `:ref:` at a live anchor and HTML-audit the
rendered hyperlink → measure the `-E` baseline fresh and diff the
WARNING/ERROR/CRITICAL set pre/post.

---

## L-030 — Additive `:label:` backfill on a derivation-mirror page: the skeleton is already labeled, so BARE dominates; fill only 5 recognizable gap classes

The #231 G1 batches insert descriptive `:label:` under unlabeled
`.. math::` on the literature-mirror corpus (`docs/theory/references/*`).
PURE ADDITIVE — a `:label:` line only, at option indent, immediately
after the `.. math::`; touch no content/prose/headings. Recount per
file yourself (`grep -cE "^\s*\.\. math::"` vs `grep -cE "^\s+:label:"`),
never trust the brief's count. The adjudication discipline:

- **On a page that mirrors a published derivation CHAIN, the checkpoints
  are usually ALREADY labeled** (governing eq, named kernels/Green's/
  resolvents, final boxed results, paper-numbered key eqs). So the
  unlabeled residue is dominated by TRUE intermediates that correctly
  STAY BARE — "label the skeleton, not every vertebra." The BARE classes
  (each recurred dozens of times): step-to-step algebra between two
  labeled checkpoints; substitution/change-of-variable steps; an
  immediate RESTATEMENT of an already-labeled eq (often cross-ref'd with
  `:eq:` right there — a dead giveaway); a COMPANION definition under a
  labeled eq (the `K_ij = …` under a labeled Nyström eq; the `where
  T_vol is …` operator/kernel defs under a labeled operator form; the
  `where B_LR = …` under a labeled closure); a STANDARD special-function
  integral representation (E₁, Ki₁, Ki₃ — labeled once via its own
  derivation, then restated bare); a test-gate closed-form realization;
  a cost/complexity model; a geometric parametrisation (chord length,
  ρ_max, law of cosines); a SCHEMATIC contrast (`∝`, a template with a
  text "(geometry factor)" placeholder); a numerical RESULT / sanity-
  check evaluation (`k_eff = 1.00421`, a vacuum-limit = 4); and
  deferred-investigation or falsified-heuristic components. Ratios land
  low BY DESIGN — trajectory_resolvent was 2 labeled / 31 bare,
  peierls_nystrom 8 / 88; that is the guidance applied correctly, not
  under-labeling.

- **The GENUINE gaps are a small, recognizable set — fill these:**
  (a) **Governing-eq parallel** — a page's BTE / 3-D transport eq when a
  SIBLING page already labels its own (galerkin_spectral `-bte` →
  mirror on fn_method `fn-method-bte`, peierls_nystrom
  `peierls-transport-equation-3d`); label for cross-page consistency.
  (b) **Unlabeled named-object definition** the page/corpus uses BY NAME
  but never labeled — escape probability `P_esc`, a continuum dispersion
  function `λ(ν)`, a discrete pseudo-eigenfunction, an L² inner product /
  Galerkin-orthogonality principle. (c) **Geometry-parallel gap** —
  cylinder+sphere carry `-nystrom`/`-row-sum-identity` but slab doesn't
  (fill the slab), or vice-versa. (d) **Sibling-parallel result** —
  sphere T-matrix labeled but cylinder/slab aren't; slab labels BOTH P
  and G mode formulas but cylinder only labels P (fill the cylinder G) —
  so grep finds the whole set. (e) **Paper-numbered key eq** in the
  page's ESTABLISHED `<page>-eqNN` family (`galerkin-spectral-eq3`
  joining `-eq4`; `singular-eigenfunction-eq47` joining `-eq46/-eq54`;
  `wm72-eq21d-normalization` joining `wm72-eq30/31/32`). The `-eqNN`
  domain-form-with-number IS the page's precedent, NOT the forbidden
  bare-positional `eq7` — match the family, don't invent a scheme.

- **Mechanics that bit this task (all avoidable):**
  - **zsh does NOT word-split an unquoted `$var`** (bash does). A
    `for n in $names` uniqueness loop silently ran ONCE on the whole
    concatenated string and printed a false "0 collisions". Use an
    explicit literal list (`for n in a b c …`) or `${=names}`. The
    per-name `grep -rn ":label: $n\$" docs/theory` gate (must return
    NOTHING before finalizing) is mandatory — labels are PROJECT-GLOBAL,
    a dup fails `-W`.
  - **Edit `old_string` must match LIVE bytes, not a prior render.** A
    block read earlier as `\mathrm dy` was actually `\mathrm{d}y`
    (braces); the edit missed → re-read the exact lines and retry.
  - **An aligned block (`A &= … \\ B &= …`, NO blank line) is a SINGLE
    align environment** — one `:label:` labels the whole block, SAFE.
    The "don't label multi-equation blocks" rule targets only
    BLANK-line-separated sub-equations (none appeared in this corpus).
  - **A list-nested `.. math::` (2-space indent) takes its `:label:` at
    5 spaces** (directive-indent + 3), not 3.
  - Do NOT run sphinx-build (main agent runs the gate once post-batch).
    Verify instead by recount (math count UNCHANGED ⇒ no block corrupted;
    label count up by exactly N) + the per-name uniqueness grep.

How to apply: recount yourself → learn each page's existing label
FAMILIES first (`grep -nE "^\s+:label:"`) → read each unlabeled block's
surrounding prose and default BARE unless it hits one of the 5 gap
classes → name in the page's family word-order → per-name uniqueness
grep (with a shell that actually word-splits) → apply, then recount to
confirm math unchanged / labels +N.

---

## L-031 — The docutils→bibtex citation migration: whitelist-scope the swap (auto-skips non-keys), indentation-key the block remover (preserves nested notes + footnotes), 3-signal-gate every heading removal

A corpus-wide `[Key]_` → ``:cite:`Key``` migration (sphinxcontrib-bibtex,
#231 Phase G2: 233 swaps + 78 def-blocks across 46 files) is mechanical
but has four traps a blanket regex walks straight into.

- **Scope the swap by a WHITELIST built from `refs.bib` keys (+ ruled
  consolidation aliases), NOT a blanket `\[\w+\]_` regex.** A non-key is
  simply never a replacement target, so the whitelist auto-skips every
  pseudo-site with ZERO line-number logic: `[A]_{:,j}` matrix notation,
  a ``[Foo1234]_`` syntax example, and footnote uses `[#name]_` (the `#`
  is outside `[A-Za-z0-9_]`). Literal `str.replace('[K]_', ':cite:`K`')`
  preserves surrounding punctuation exactly and can NOT touch the def
  line (`.. [K] ` ends in a space, not `]_`), so swap-order is irrelevant.
  Consolidations map alias→canonical (`[PS1982]_`→``:cite:`Pomraning…```);
  the alias key is REMOVED from refs.bib by the bib owner, so emit only
  the canonical and verify zero leaked ``:cite:`AliasKey``` after.
- **Def-block removal MUST be INDENTATION-based, not blank-delimited.** A
  citation body can carry an INTERNAL blank line — a nested `.. note::`
  admonition inside the `.. [Key]` block — so a "consume until blank"
  remover ORPHANS the note (a real dry-run hit). Consume the `.. [Key]`
  line + every following line indented deeper than the marker (internal
  blanks folded in via lookahead: a blank whose next non-blank is
  deeper-indented is internal), stopping at the first dedent-to-base.
  KEY THE REMOVER TO THE WHITELIST so footnotes (`.. [#name]`) and any
  non-citation `.. [x]` survive — a `[^\]]+` remover eats footnotes.
- **Footnotes are a DIFFERENT docutils construct — preserve both halves.**
  Always DRY-RUN-categorize every `[x]_` use AND every `.. [x]` def
  against the whitelist first, printing UNKNOWN keys + STRAY brackets;
  the dry-run is what surfaces the `.. [#name]` footnote family (3 here)
  and any typo'd key before a single byte is written.
- **Emptied-"References"-section removal needs a 3-SIGNAL cleanliness gate
  (grep BEFORE removing any heading):** (1) is `autosectionlabel` enabled
  in conf.py? — if OFF, a bare `References` heading is NOT a cross-ref
  target; (2) any inbound `:ref:` to the section's explicit `.. _anchor:`
  (grep tree-wide — here the sole `.. _bib-*:` citation anchor had zero
  referrers → safe to drop with its citation); (3) directional prose
  ("listed below", "the references section"). All three clean ⟹ REMOVE
  heading+underline+preceding-blanks (asserted script: strip trailing
  blanks, assert underline, assert heading text, pop). Referenced ⟹ keep
  + one pointer line to the bibliography page. MIXED section (docutils
  defs + a plain-text further-reading bullet list / "Internal references"
  subsection) ⟹ KEEP the heading, delete only the def blocks.
- **A NOTE describing the RETIRED docutils cross-doc citation mechanism is
  Cardinal-Rule-1 obsolete post-migration — remove it.** ("Citations
  shared across pages resolve cross-document via Sphinx's docutils
  citation index; only local citations defined here" is now false under a
  central refs.bib.) It housed the ``[Foo1234]_`` skip example, which
  correctly vanishes with it — SKIPPING a pseudo-site (don't swap) is not
  PRESERVING it forever; report the removal as a per-page decision.
- **keylabel style ⟹ the migration is INVISIBLE to readers.**
  `bibtex_default_style = 'keylabel'` renders the label AS the key, so
  ``:cite:`Hebert2009``` displays `[Hebert2009]` — character-identical to
  the retired bracket label. State this in the bibliography page's lead.
- **Python docstring-only constraint is git-diff-verifiable.** A `]_`
  operator can't appear in executable Python, so citation uses/defs live
  ONLY in docstrings — but still GATE it: `git diff <pyfiles> | grep
  '^[-+]' | grep -vE '^[-+]{3}'` and confirm every changed line is
  docstring/reference text, zero `def`/`import`/logic. Confirm the one
  math-notation skip file (`operator.py` `[A]_`) has an EMPTY diff.
- **The new bibliography page is NOT a reference SOLVER.** `.. _anchor:`
  above title, lead para (single citation home; entries in refs.bib /
  Zotero upstream; keylabel; per-page lists retired), then a BARE
  `.. bibliography::` (only cited entries render). Grep for a pre-existing
  `.. bibliography::` (a second one warns) + label collision first. Place
  it in its OWN labelled toctree subsection — dropping it into a
  reference-SOLVER toctree miscategorises it; give it a `-` subsection
  under "Pages" (size the underline in code points, L-009).

How to apply: dry-run-categorize uses+defs against the refs.bib whitelist
and report unknowns; literal-swap only whitelist keys (aliases→canonical);
indentation-key the whitelist-scoped block remover (nested notes +
footnotes survive); 3-signal-gate every heading removal (autosectionlabel
+ inbound `:ref:` + directional prose); remove stale mechanism-notes;
verify `.py` diffs are docstring-only; give the bibliography its own
toctree subsection. Full inline-output identity comes free from keylabel.

---

## L-032 — P10 `:label:` re-namespacing (a label follows its heading's ruling): the SELF-DESCRIPTION oracle, the section-vs-eq-label scope split, and the delimiter-anchored replace that survives a prefix-overlapping sibling

Deferred from a Phase-F HEADING retitle campaign: labels carry tree-wide
`:ref:` blast radius, so the anchor rename is its own pass. The rule: a
label follows its heading's design/record ruling (design-named heading →
RENAME the label to the heading's vocabulary, keep the page prefix;
record/charter heading → KEEP). The disciplines that pass taught:

- **The section's OWN self-description is the strongest oracle** — stronger
  than the label name, and it resolves the brief's genuine hedge. A `sn-282-*`
  family "may be a #282 record" per the label, but the section literally said
  "this section is the **resolution chapter** … those [other] sections are
  preserved as **the record**" — a page drawing the design/record line WITH
  ITSELF ON THE DESIGN SIDE. Combined with (a) the Phase-F map retitling its
  top heading to a design name + "P7's own worked example", (b) all subsection
  headings design-named, (c) it living on the DESIGN page while the record
  lives in the charter chapter — the verdict was RENAME, unambiguously. A
  charter page states its own status too: `curvilinear_numerics.rst` opens
  "This chapter is Part B's **campaign record**" → KEEP-all (incl. its
  issue-styled `sn-issue-196-*` anchors — an issue number inside a charter
  chapter keeps).
- **When the labels form a design FAMILY, P10 is FULL — section anchors AND
  equation labels drop the campaign token together; excluding eq labels is the
  WRONG default.** My pass-1 instinct (rename anchors, EXCLUDE eq labels because
  they carry V&V-matrix + `.. vv-status:` + `:eq:` weight, "flag the mixed
  namespace") was OVERRULED: a `sn-r12a` section anchor beside a
  `sn-282-r12a-predicate` eq label is a *two-spellings* state, and the standing
  naming-consistency rule (a family follows ONE pattern; fix off-pattern members
  in the SAME change) forbids shipping it. The correct pass unifies the whole
  family (`sn-282-*` sections + eq labels → `sn-direct-seed-*`; `wave-t-*` →
  `tensor-network-*`). The eq-label V&V weight is NOT a reason to defer — it is
  a **gate**: a documented-only eq label (silent-class grep of `orpheus/`+`tests/`
  = ZERO, so no `@pytest.mark.verifies("label")`/`catches` edge) renames
  mechanically (the `matrix.rst` auto-regens, L-008; `.. vv-status:` directives
  move with the label). ONLY a verifies-target eq label (a silent-class HIT)
  would orphan a test edge (L-003 phantom-verifies) — THAT is the one to
  flag/defer, and you don't edit `tests/` regardless (report the hit). So: run
  the silent-class grep FIRST; empty ⟹ full unification is safe and MANDATORY;
  hits ⟹ rename docs, report the test edge. (The lone brief-named eq residue
  `region-areas-pincell`→`-pin-cell` was always in scope.)
- **A section anchor can be a PREFIX of a sibling equation label**
  (`sn-282-r12a` ⊂ `sn-282-r12a-predicate`). A bare string replace corrupts
  the sibling. Replace ONLY fully-delimited forms — `.. _X:` · `` `X` `` ·
  `<X>` (the `:ref:`txt <X>`` angle form) · `.. vv-status: X ` (trailing
  space) — where the char after `X` (`:` `` ` `` `>` space) can't match a
  longer label. Do the rename with a script that reports a per-file, per-form
  count and ASSERT it against the pre-computed inbound tally; then a
  corruption grep (the sibling still present, `sn-r12a-predicate` = ZERO)
  proves the delimiters held.
- **Collision + genericness force per-label disambiguation off the clean
  token drop.** Dropping the campaign token gives `sn-282-gotchas`→`sn-gotchas`
  — but `sn-gotchas` already anchored the index page, and bare
  `sn-numerical-evidence` is one of three sibling `*-numerical-evidence`
  families. Disambiguate with a design family prefix tied to CODE vocabulary
  (`sn-direct-seed-*`, from the test file `test_282_direct_seed_*`), grep-check
  every proposed new name for 0 collisions BEFORE the script runs. And once a
  disambiguator prefix exists for TWO members, the naming-consistency rule
  pulls the WHOLE family onto it (a 7-bare/2-prefixed split is off-pattern) —
  the clean endpoint is `sn-direct-seed-*` for all 9 sections + all 5 eq labels,
  not a per-label heading-mirror (see the FULL-P10 bullet above).
- **The deterministic gate is the grep, not the build** (L-002): a cross-doc
  `:ref:` to a renamed-away anchor renders plain-text with NO `-W` warning.
  Proof-by-construction = {no OLD delimited form survives anywhere} ∧ {every
  NEW anchor exists exactly once}; confirm with a rendered-HTML `href=` audit
  on one cross-doc ref (`api/*.html` → `foo.html#new-anchor`). The clean `-W`
  build only catches intra-doc dangling. A code-string ref in `orpheus/`
  (`f"…§peierls-phase5-retreat"`) is silent-class — REPORT file:line + new
  label for the main agent (you don't edit `orpheus/`/`tests/`).
- **Prose two-spelling harmonization rides the same pass** (Surface 3): after
  renaming an anchor, event-name prose ("the Phase 5 retreat" → design name
  "the continuous-µ retreat") harmonizes design-first, KEEPING the historical
  tag where it carries record value ("… (Phase 5's terminal decision)") and
  leaving pure-provenance/file-path mentions ("Phase-5 Round-3 provenance",
  `diag_phase5_*.py`) untouched — fix ONLY where two-naming reads as two
  events. Match the canonical code-point (event-name "continuous-µ" = literal
  µ U+00B5 per the heading; the math object is `:math:`\mu``) — source it from
  the file, don't retype. A consistent issue-TAG beside an established design
  name (loss_representation's "the sweep-inverse-contract discharge (#284)")
  is NOT two-naming → WON'T-FIX with a one-line justification.

How to apply: read each candidate's heading TODAY + the Phase-F batch map +
the section's self-description; charter pages KEEP-all; design headings →
rename anchors (not eq labels — flag the mixed namespace) via a
delimiter-anchored counted script; grep-prove no OLD form survives + new
anchors exist; HTML-audit one cross-doc ref; harmonize event-name prose
design-first; report silent-class `orpheus/`/`tests/` hits.

---

## L-033 — Code-prose rebalance (#231 Phase 2): an operator file's teaching is ALREADY TWIN — expect ZERO MOVED, verify by token-invariance, and keep the CONTRACT tier by the "wrong-simplification guard" test

The pilot (P2-A, `transport/operators/scattering.py`, 73%→63% prose,
docstring 1127→721, comment 196→121) established the calibration for
rebalancing a "documentation-with-code-attached" operator file. The
classify-each-block-into-one-verdict rubric
(CONTRACT/TWIN/MOVED/HISTORY/COMMENT-cut) has a dominant outcome on
operator files that the instinct fights:

- **The operator-algebra book is EXHAUSTIVELY COMPLETE → expect ZERO
  MOVED.** On a heavily-prose operator file the reflex is "some design
  rationale must be book-worthy-but-unwritten". It almost never is. Three
  concepts that FELT unique to the code — the forward-fast-path-vs-adjoint-
  frame asymmetry, N2N-as-a-distinct-moment-operator, the foldable/σ_r
  split + the #215 trap — were EACH fully TWIN after grepping the landing
  chapters (`adjoint.rst` §sn-scattering-adjoint-source; `operator_algebra.rst`
  §integral-kernel-category / §scattering-as-tensor-product-sum;
  `slab_one_group.rst` §si-sigma-r-fold-mismatch + `loss_representation.rst`
  §loss-rep-removal-sigma). **Discriminator: grep the landing chapter for
  the concept BEFORE assuming novelty.** Budget the operator-file batches
  (fission, streaming, boundary, multiplication, isotropic_scattering) as
  TWIN-CUTTING, not MOVED-writing. If you think you found a MOVED, grep
  harder first — Cardinal Rule 3 means the theory shipped with the code.

- **CONTRACT-vs-{HISTORY,TWIN} has three sharp discriminators the pilot
  named** (these are the reusable judgment rules, not the file specifics):
  (a) **A keep-vs-retire decision on a currently-orphan symbol is CONTRACT,
  not HISTORY** — even phrased historically ("Deliberately retained W-F,
  user steered keep"). It is a live constraint: the arm is an *intentional*
  orphan kept for a named future consumer (an OPEN issue), and a naive
  retirement audit would delete it as dead. Keep the keep-decision + the
  open-issue rationale; cut only the date/steering provenance. (b) **A
  `⚠ LATENT … TRAP` / "do NOT" imperative is COMMENT-keep even when its
  EXPLANATION is TWIN** — the derivation goes to a `§`-pointer, but the
  imperative + the falsifying number (46–56 %) + the tracking-issue
  pointers (#2/#215) stay inline, because a future within-group-solver
  editor reads THIS file, not the theory page. (c) **A type-annotation-
  choice rationale that guards a plausible wrong "simplification" is
  CONTRACT** — "returns the concrete `OperatorSum`, not the bare
  `LinearOperator` erasure, so `apply_transpose` stays visible to the
  checker" prevents a modifier from silently breaking a consumer by
  "tidying" the return type. General test for the CONTRACT tier: **"would a
  competent modifier who never leaves the file do the wrong thing without
  this line?"** — if yes, it is CONTRACT regardless of how history-flavored
  the prose is.

- **HISTORY-cut only after confirming the fact is in the record.** The
  module-head Wave-D-extraction narration ("Per Cardinal Rule 2 this
  lifts… bit-identical extraction… SNSolver retains thin delegators") is
  HISTORY *because* slab_multigroup.rst 439–444/578–582 already carries it
  AND the delegators verifiably still live (solver.py:1884/1892/1921).
  Verify BOTH — the record home and the live truth — before cutting; a
  HISTORY claim that is novel-and-recorded-nowhere becomes a
  Development-history dropdown MOVE, not a cut.

- **Verification discipline unique to this task class:** (1) the edits are
  docstring/comment-ONLY, so PROVE zero code change by a **non-string/non-
  comment token comparison vs HEAD** (`tokenize`, drop COMMENT/STRING/
  NL/INDENT…) — 2397==2397 is stronger evidence than `pytest --collect-only`
  (which also passed, 6652 unchanged). The `code lines` count (484→484 via
  `ast`) corroborates. (2) **The sphinx gate is CONDITIONAL on automodule
  status — check it FIRST (`grep -rn "automodule:: <module>" docs/`).** A
  not-`automodule`'d file (P2-A scattering.py) has build-invisible docstrings →
  SKIP the multi-minute build (say why). But an `automodule`'d file (P2-G
  streaming/augmented_mesh/boundary, all in `api/discrete_ordinates.rst`)
  RENDERS its docstrings → the `-E -W` build gate is MANDATORY (capture the
  baseline BEFORE editing; acceptance = W/E/C set unchanged). `:noindex:` does
  NOT exempt it — `:noindex:` only makes the module xref-invisible (L-002); the
  docstrings still render and malformed markup still warns. Two automodule-safe
  moves when trimming: (a) NO `.. math:: :label:` in any of the three docstrings
  → cutting math blocks orphans no `:eq:` (grep-confirm the file's `:label:`
  count is 0 first); (b) KEEP section-title underlines VERBATIM (over-long is
  allowed) — trim only the prose body under a heading, never resize the
  underline, or you risk "Title underline too short". (3)
  **Pointer form = literal greppable `docs/theory/<part>/<file>.rst §<label>`**;
  `§` may point at an EQ-label (`:label:`) when no co-named section anchor
  exists (`mg-inscatter-source`, `sn-scattering-adjoint-source`) — it is a
  human marker, not a rendered role (the file isn't automodule'd), so it
  resolves via grep. Gate every label with `grep -E "^\.\. _X:|:label: X$"`
  and every file with `[ -f ]`; never invent.

- **P2-E CONFIRMATION — ZERO MOVED generalizes PAST operator files to the
  spatial-scheme file class, and the Haiku pre-inventory's MOVED column is
  ~100 % noise.** P2-E was `transport/spatial/{scheme,diamond,linear_discontinuous}.py`
  — NOT operators, different landing chapters (`foundations/discretization.rst`,
  `methods/sn/cartesian_multid.rst`, `methods/sn/loss_representation.rst`). The
  pre-inventory graded **13 MOVED**; re-adjudication overturned ALL 13 to
  TWIN/CONTRACT. Two "needs a new theory page" candidates (an advection–reaction
  interface, a reverse-mode transpose section) each already had a complete home
  — `§discretization-closures` even cross-references the exact code symbols
  (`outgoing_face_from_average`, `reaction_xs`). The ZERO-MOVED result is not
  operator-specific; it is a **Cardinal-Rule-3 consequence** (the theory shipped
  with the code), so budget ANY Phase-2 batch as TWIN-cut + CONTRACT-trim and
  grep the landing chapter before crediting a single MOVED. TWO sharp new
  discriminators the pre-inventory got wrong: (a) **a method that teaches
  d-generic / Kronecker / tensor-product structure is usually BOTH layers** —
  the LAYOUT theory is TWIN (→ `cartesian_multid.rst §spatial-moment-space`),
  but the reconstruction GOTCHAS (a d=1 trailing-axis-append; "keys on
  `face_moment_tail`, NOT a shape probe"; trace-order == inflow-order
  consistency) are CONTRACT. Same object, two layers — KEEP the contract, POINT
  the theory; never let "this is tensor-product teaching" auto-MOVE a method
  whose gotchas fail the wrong-simplification-guard test (overturned 3 LD
  methods: `_ubld_inflow`, `_ubld_outgoing_faces`, `moment_scan_closure`). (b)
  **the bit-identity operation-order discipline is a single-source across
  DOCSTRINGS** (Cardinal Rule 2, not a TWIN-cut): the canonical "explicit left
  fold, do NOT regroup" statement lives at the one helper
  (`_cartesian_streaming_diagonal`); sibling kernels that REPEAT it get trimmed
  to a pointer AT the helper, and the ⚠ do-NOT-regroup imperative stays only at
  the single source. Finally: **a contraction routinely surfaces a
  Cardinal-Rule-1 stale claim** (here a Protocol `is_affine_scannable`
  description said LD "does not qualify", false since #158 Increment B) — L-001
  applies MID-TRIM: verify the claim against LIVE code, FIX + report the
  scope-expansion, never transcribe the stale text into the trimmed form.

How to apply: for a Phase-2 operator-file batch, grep the landing chapters
for every teaching concept FIRST (expect all-TWIN); classify each block by
the "wrong-simplification guard" test for CONTRACT; keep latent-trap
imperatives + open-issue keep-decisions inline; cut TWIN/HISTORY to
greppable literal-path `§`-pointers (eq-label OK); prove zero code change
by token-invariance; skip the sphinx gate iff the file isn't automodule'd.
The lossless map is per-block `file:line | verdict | destination | id`,
written INCREMENTALLY, ending with verdict counts + before/after prose
lines + the 3–5 hardest calls that calibrate the siblings.

---

## L-034 — Code-prose rebalance, CONTRACT-DENSE file classes (#231 Phase 2, batches B + C + D + F + G): a machinery PACKAGE, a DRIVER file, an ABC/algebra-DEFINITION file, a CONTRACT-heavy OPERATOR file, a MESH/phase-space file, and a CURVILINEAR ψ½-operator pair are all contract-dense, so the honest cut is far smaller than the teaching-operator pilot (and that is CORRECT) — but the cut SURFACE differs by class

L-033 calibrated the rebalance on an OPERATOR file (`scattering.py`,
docstring −36 %). Batch B (`sn/loss_representation/{__init__,sweep_graph}.py`,
the SN sweep machinery, docstring −2.6 %) and batch C (`sn/solver.py` +
`numerics/iteration.py`, the SN driver + iteration primitives, docstring
−5 % / comment −16 %) both established that a **contract-dense file is a
DIFFERENT file-class from a teaching file** and the same rubric yields a
much smaller, honest cut. The file-class discriminator is the load-bearing
lesson — and each contract-dense class has its OWN dominant cut-surface:

- **Operator-SURFACE file** (in `transport/operators/`, `sn/operators/`): its
  prose TEACHES the operator algebra, which is 100 % TWIN in the
  operator-algebra book → aggressive TWIN-cutting (−30-40 %). **NUANCE (P2-G,
  batch G): the −30-40 % is for a TEACHING-heavy operator file (`scattering.py`
  was 73 % prose teaching the kernel). A CONTRACT-heavy operator file cuts FAR
  less** — `streaming.py` (docstring −16 %) and `boundary.py` (−4 %) carry the
  apply / solve / adjoint / reflect / split CAPABILITY contracts + the
  `_require_typed_composite` guard + the `_reflect_trace` adjoint-spelling
  ⚠-trap (a het-VACUUM-sphere-only catch) + the `reflect_rows_inplace`
  additive-not-overwrite ⚠-trap. The discriminator WITHIN the operator class is
  teaching-density vs contract-density; on the contract-heavy end the cut is
  campaign-TAGS-on-live-contracts (trim the tag, keep the contract), not
  standalone teaching. Latent-trap imperatives get an explicit `⚠` marker + the
  falsifying detail inline (L-033b), derivation → `§`-pointer. **CURVILINEAR
  sharpening (P2-F, `pole_angular_closure.py` docstring −23 % +
  `radial_characteristic.py` −15 %):** in the ERR-026/ERR-053 subsystem, KEEP
  the MATH FORMULA at point of use even when the teaching AROUND it is TWIN —
  the α-recursion index convention (`c_out=α/τ`, `c_in=(1−τ)/τ·α+α`), the
  `faces[g,m,i]` off-by-one, the τ_raw split formula, the seed-extrapolation
  `t` are each a file-local dependency a modifier depends on (a sign/index slip
  IS the historical hazard); cut the derivation, keep the formula + `§`-pointer.
  Two LATENT-TRAP imperatives were the load-bearing keeps — "do NOT tidy the τ
  arithmetic into a call to `contamination.morel_montry_weights`" (collapses the
  Leg-1 cross-check into a reference-contamination tautology, vv L11) and "do
  NOT re-implement the march at a call site" (single-source orchestration) —
  both fail the "would a file-local modifier do the wrong thing without this
  line?" test, so they stay inline even though their EXPLANATION is TWIN.
- **Sweep-MACHINERY / package file** (the `(L+C)` traversal realization, the
  DAG, the cell kernels): its prose STATES the local contract that *references*
  a book concept — "returns the FULL loss `(L+C)ψ`, Resolution A" is NOT
  teaching Resolution A (that lives at `§loss-rep-resolution-a`), it is THIS
  method's return contract. The "would a modifier who never leaves the file do
  the wrong thing without this line?" test keeps the vast majority. ZERO MOVED
  still holds (grep-confirm the book carries every concept), but the TWIN-cut
  surface is only the **module-head essays + campaign-relocation provenance +
  duplicated measured numbers**, not the method bodies.
- **DRIVER file** (the solver orchestration `sn/solver.py`, the iteration
  primitives `numerics/iteration.py`): its docstrings are the estimator /
  convergence / threat-model CONTRACT a modifier needs (kept near-whole —
  the `compute_keff` R7 role split, #291 leakage, balance identity,
  scale-bridge, the #282 lag-death certificate, the ERR-053 restart trap are
  ALL wrong-simplification guards). The dominant cut surface is therefore
  **COMMENTS, not docstrings**: standalone `#`-comment RETIREMENT TOMBSTONES
  (`_GaussSeidelResolvent`/`_MomentWindowedResolvent`/`_make_sweep_preconditioner`/
  the P1.7 `_build_rhs_*` block — git owns them; the live design is on the
  surviving function) + campaign-STATUS blocks (a 23-line "Issue #168 status"
  Phase-A/B/C/D narration annotating a 1-line default) + the HISTORY TAILS of
  SPLIT method docstrings (a "Scope"/"Verified"/"History:" section narrating a
  landed campaign under a CONTRACT algorithm). Batch C: comment −16 % (−101 ln)
  DWARFED docstring −5 % (−61 ln). Hunt the `#`-comment tombstones FIRST on a
  driver file, not the docstrings.
- **ABC / algebra-DEFINITION file** (the base-class file the book is ABOUT —
  `numerics/operator.py`, the LinearOperator ABC + composers/adjoint/inverse):
  the LEANEST cut of all (P2-D: docstring −2.4 %, comment −2.8 %, −51 ln). Its
  docstrings STATE the binding laws (closure/composition/adjoint-swap/
  homomorphism), the raise-conditions, and the typing-rationale. The trap: a
  Haiku classifier proposes MASS **MOVED** for "closure law"/"composition law"/
  "role classification"/"three-layer surface" (it did — 28). On an ABC file
  those verdicts **INVERT to CONTRACT**: the law IS the in-file contract at
  point-of-use (`OperatorSum.is_invertible`'s "ONLY the LEADING term need be
  invertible" — cut it, the next modifier "fixes" it to require both). The book
  teaches the concept's *derivation* (TWIN); the docstring states the *law the
  class obeys* (CONTRACT — never MOVED, the book already carries the concept →
  ZERO MOVED still holds). Dominant cut surface: inline **campaign-step
  provenance** (citation-vs-narration rule below) + multi-clause HISTORY
  narration stories, NOT the laws. The BATCH-SPECIAL row-8 dual-A bridge was
  ALREADY-SATISFIED here too (verify + report, per L-034's special sub-rule);
  the rebalance-read surfaced + fixed a stale `:ref:`operator-algebra-adjoint``
  → `operator-adjoint` (per the staleness-audit sub-rule).
- **MESH / phase-space file** (P2-G: `sn/mesh/augmented_mesh.py`, the `SNMesh`
  construction + property surface): like the DRIVER class, the dominant cut
  surface is COMMENTS, not docstrings (comment −26 % / −79 ln DWARFED docstring
  −0.6 % / −4 ln). The property + classmethod docstrings ARE the mesh's public
  API contract (the `bc` face-inventory-IS-BC-inventory invariant, `is_1d`'s
  ny==1-phantom gotcha, `full_field_space`'s G-adjoint metric, `dag_walk`'s
  XOR-signature) — kept near-whole. The narration lives in the `_init_core`
  CONSTRUCTION-BODY comment cluster (a 56-line Phase-C/D angular-closure-flip
  story annotating a 2-line default; the Wave-D/E 6-site migration essays; the
  deprecated-accessor tombstones) → hunt those `#`-comment clusters FIRST,
  exactly as on a driver file. The CONSTRAINT still stays even inside a
  construction comment (the CLASS-not-instance closure-bind reason, the "mesh
  provides shape only" B.5.A rule, the how-to-add-a-BC-kind recipe) — cut only
  the wave-flip STORY around it. Also surfaced a CORRECTNESS fix here's sibling:
  `streaming.apply_transpose`'s summary said "Hilbert transpose" while its body
  + the sibling `boundary.py` correctly say **Euclidean** transpose (`.H` is the
  metric Hilbert adjoint) — fixed per L-010 (a rebalance-read surfaces stale
  V&V vocabulary, per the ABC bullet's staleness-audit sub-rule).

Sharp sub-rules (machinery + driver classes):

- **Provenance trimming = citation-vs-narration, applied UNIFORMLY (internal
  consistency).** On a CONTRACT-dense file the inline provenance tags ARE the
  cut. Draw the "constraint stays / narration cuts" line at
  *citation-vs-narration*: TRIM landed campaign-STEP codes (`Wave O`/`carve
  PN`/`taxonomy §NN step N`/`spec §NN`/`Phase 2.5x`/`né _as_dense`/`O.2b`/`W-A
  collapse` — git + the theory page own them) but KEEP bare `#NNN` issue
  anchors (rubric: live-issue one-liners, the more durable traceability) and
  NAMED PATTERNS with theory anchors (`Design C`, `coding-elegance Pattern 2`).
  A bare `#280` is a citation (keep); "carried as documented twins until the
  3rd sibling fired the extraction trigger" is narration (cut). Apply to EVERY
  tag of a class or NONE — a half-stripped file violates internal consistency.
  A retired-SYSTEM lineage note ("the RUNTIME successor to the `CAP_SOLVE` tag")
  is HISTORY even when terse — `CAP_*` is fully retired, no live code references
  it, so it is pure archaeology (aggressive-retirement); the predicate law
  stands alone.

- **A hand-transposed-adjoint / reverse-scan / boundary-block comment body is
  the ALGEBRA-OF-RECORD — KEEP even though it reads like narration.** The
  cotangent routing, the seed-fold transpose, the degenerate-diagonal adjoint,
  the O.4b active-trace block, the moment-frame involution (an ERR-061-class
  diffusion-limit root cause): a modifier editing the adjoint/kernel MUST have
  these. Cutting them is the Cardinal-Rule-1 hazard the brief flags as "3
  constraint-bearing blocks misgraded HISTORY", at scale. A Haiku-style
  pre-classifier marks most of these COMMENT-cut [low-conf]; re-adjudicate EACH
  — nearly all are CONTRACT. Trimming a shape-annotation to save 1-2 lines
  strips a real invariant (the "DD/Step byte-identical" note is a *testable
  negative control*, not chatter).
- **Duplicated measured performance numbers → single-source to the canonical
  theory anchor, but keep ONE inline at the point-of-decision.** A perf basis
  repeated across N class docstrings (e.g. a Fork-B2 0.57-0.84× sweep basis) is
  TWIN with its theory home; point the *descriptive* docstrings there, but LEAVE
  the headline number in the *factory/selector* docstring where the choice is
  made. DRY (Cardinal Rule 2) without stripping numerical evidence from the code.
- **automodule + `:noindex:` makes the Sphinx gate LIVE (divergence from the
  L-033 pilot).** The pilot's `scattering.py` was NOT automodule'd → no build
  gate. A package rendered by `automodule:: … :members: :undoc-members:
  :noindex:` (here `discrete_ordinates.rst`) still RENDERS the docstrings (only
  the xref *targets* are suppressed — L-002), so a malformed docstring breaks
  `-E -W`. RUN the build gate both sides for an automodule'd file; grep-confirm
  0 `.. math:: :label:` / 0 `vv-status` first (cutting then orphans no
  `:eq:`/`verifies` target). Pointer nuance: the ratified literal
  `docs/theory/…rst §<label>` form is brief-correct, but in an automodule'd file
  a `:ref:`<label>`` role would render as a working link — flag it, don't
  unilaterally switch. **P2-F confirmed this on `radial_characteristic.py`
  (automodule'd, 0 warnings both sides) vs `pole_angular_closure.py` (NOT
  automodule'd → invisible, its 3 module-docstring `:label:` blocks cut-safe
  after the grep). TWO positive moves a RENDERED file affords that a
  non-rendered one cannot:** (1) **promote a LATENT-TRAP to a rendered
  `.. warning::` admonition** (the fission-double-apply HAZARD → a visible
  warning box, L-010 prophylactic-warning — better than an inline comment); mind
  the 3-space content indent under the directive. (2) **a section-RENAME during
  the cut stales an in-file back-ref** — renaming module-docstring headings
  ("Scope of this realization"→"Realized surface", "The ONE solve
  orchestration"→"Single source") staled two class-docstring back-references;
  grep the file for the OLD heading text after ANY rename and repoint (the
  rebalance-read staleness sub-rule, applied to in-file section refs).
- **A "fix if MISSING/drifted" BATCH-SPECIAL that turns out ALREADY-SATISFIED
  → verify against the oracle + REPORT satisfied; do NOT touch the correct
  CONTRACT block.** Batch C's SPECIAL 1 (the `notation.rst` row-8 dual-A
  bridge must survive in `iteration.py`'s module head) read as an instruction
  to edit — but the module head ALREADY stated it verbatim (posing +
  A=invertible-resolvent-operand + SN binding `A=L+C` gains `(S,B)`→`L+C−S−B`
  + fission-never-a-gain). The disciplined move: READ the oracle (`notation.rst`
  row 8), READ both ends, confirm the match, REPORT "SATISFIED, no fix" — a
  correct CONTRACT block a special protects is a KEEP, not an edit target.
  Same for a posing-drift special (SPECIAL 2, dated `(A−S−F)` posings): grep
  BOTH files, find zero, report CONFORMANT. A special is a *verification*
  obligation first, an edit obligation only on failure.
- **A rebalance READ surfaces a Cardinal-Rule-1 staleness bug — FIX it in-pass,
  flag it.** Trimming a comment means READING it, which is the only gate that
  catches a stale RAW PATH in prose (`-W` is blind to path strings — L-002; a
  `:class:`/`:func:` renders plain-text, a raw `orpheus/sn/scattering.py`
  string warns nowhere). Batch C: the source-helper comment cited
  `orpheus/sn/scattering.py`; the class lives at
  `transport/operators/scattering.py` (grep the live import). Fixed to the
  class ref (Cardinal Rule 1 supreme), folded into the tag-trim, REPORTED as a
  discrepancy. The rebalance is a free staleness-audit of every comment you open.

How to apply: FIRST classify the file — operator-surface vs machinery/package
vs driver (the folder + "does the prose teach the algebra, state a local
contract that uses it, or orchestrate/interface?"). Contract-dense (machinery
OR driver) ⟹ budget a small, surgical cut and KEEP the method bodies +
constraint comments; the cut surface = machinery's module-head essays +
provenance + duplicated numbers, driver's standalone `#`-comment tombstones +
status blocks + SPLIT-docstring HISTORY tails (hunt comments FIRST on a driver).
Run the Sphinx gate iff automodule'd (a `:noindex:` automodule STILL renders →
gate LIVE). Verify every batch-special as a check first (edit only on failure),
fix any stale raw-path you open, and REPORT the small-cut-is-correct finding
with the file-class rationale so the reviewer doesn't read −2-5 % as timidity.
Cross-links [[lessons-L33]] (the operator-file twin).

---

## L-035 — Orphan-slice adjudication (V7 backfill): the WIRE-vs-SENTINEL discriminator + the conceptual-root / foundation-coexistence corollaries, and the FAST theory-scan self-check

Adjudicating a batch of orphan eq-labels (RST `:label:`s with zero
`verifies` + no sentinel) into WIRE / SENTINEL / GAP has ONE sharp
discriminator, and it is NOT "is it definitional?" (almost everything on
a foundations page is):

- **WIRE** iff an existing test's PRIMARY assertion IS this exact equation
  against a STRUCTURALLY-INDEPENDENT reference — "would a sign/factor flip
  in the equation red this test?" YES. (`inflow_mask == flatnonzero(mu<-eps)`;
  `condensed SigT == fractional flux-weighted hand-sum`; `assert_balanced`
  on the collapsed mixture = the balance-PRESERVATION claim; `[K] ==
  np.linalg.solve(A,F)`.) Spelling MUST copy the `:label:` verbatim
  (a typo'd verifies = a matrix-flagged phantom).
- **SENTINEL** iff one of THREE structural shapes, NOT merely "it reads
  definitional": (a) a GENERAL SCHEMA / CONTINUOUS definition / LITERATURE
  identity whose CONCRETE / DISCRETE / TERMINAL instance is tested under a
  *different* label (general `M R = c_V I` → concrete `pi-r-equals-4pi-i`;
  continuous `Γ_±={Ω·n≷0}` → discrete `inflow-mask-discrete`); (b) a
  NATIVE-vs-LEGACY **bit-identity** regression (`axis_widths == legacy`,
  L-004 representational) — distinct from an independent-reference predicate
  test, which is WIRE; (c) documents code that does not exist yet
  (adjoint-weighted homogenization, blocked on an open issue).
- **GAP** only for a load-bearing COMPUTED contract with NO test anywhere.
  In a mature tree a whole 38-label slice can legitimately be 8 WIRE / 30
  SENTINEL / **0 GAP** — every "gap" turned out either tested (WIRE) or a
  definition/schema/literature identity verified downstream (SENTINEL). Do
  not manufacture a GAP to look thorough.

**Conceptual-root corollary.** A ROOT narrative page (e.g. `path_integral`,
"one object, five methods") states equations that the METHOD pages realize
and verify. ALL its orphans are SENTINEL (harness case-a: "a derivation step
whose terminal result is tested downstream") — EVEN when a formula IS tested,
because it is tested under the METHOD page's OWN label
(`path-integral-transport-correction`'s `D=1/(3Σtr)` is verified via the
diffusion page's `diffusion-coefficient`; `path-integral-generation-series`'s
`k=ρ(A⁻¹F)` via `matrix-eigenvalue`). Wiring the method-page test to the
root-page label is redundant double-labeling — SENTINEL with a rationale that
NAMES the downstream gate so a reader knows it IS tested, just elsewhere.

**Foundation-coexistence corollary.** A `:label:` backfilled in a late
label-pass (#231 Phase G) often has its test in a module-`@pytest.mark.foundation`
file whose docstring still says "software invariant — no theory :label:;
foundation carries NO verifies". That premise is STALE (the label now exists).
Resolve per-test: WIRE the ONE class that pins a COMPUTED physics formula (the
production-weighted `χ_mix` teeth-test `TestChiMixHandReference`), SENTINEL the
pure software-INVARIANT (the `Σχ=1` simplex law — the canonical foundation
case, vv-principles "foundation NEVER carries verifies"). Module-foundation +
class/method-`verifies` COEXIST and produce a real edge — the audit's
`_equation_coverage` reads `m.equations` regardless of the level tag
(`test_mixture_condense.py` is the in-tree precedent: module-foundation, class
`verifies("energy-condensation-rate-preservation")`). Add ONLY the decorator;
don't rewrite the stale docstring (scope-creep).

**FLAGGED-line-range can be stale — the doc's OWN named catcher wins.** A
stage-plan "wire to test at file:266-300" pointer had drifted: 266-300 was a
SPECTRAL cross-engine test that is Mode-12-BLIND to the mutation class (k∞
moves by *exactly* 0 under factor-swap/transpose — similarity + `eig(Mᵀ)=eig(M)`).
The RST prose itself NAMED the correct catcher (`test_K_operator_as_matrix_is_the_resolvent`,
the intrinsic `[K]==solve(A,F)` OBJECT gate). Trust the doc's named gate over
the brief's line number; verify it pins the OBJECT, not the spectrum.

**FAST self-check when N sibling batches edit concurrently.** Do NOT run the
full `python -m tests._harness.audit` (slow pytest collection, and its
tree-wide theory scan trips on sibling batches' in-progress sentinels). Call
`tests._harness.audit._scan_theory_equations(Path('docs/theory'))` DIRECTLY —
it validates every sentinel (same-file rule, spelling, `documented` set) and
resolves wired labels WITHOUT collection, in <1 s. Filter `scan.violations`
to YOUR file set; assert your new sentinels ∈ `scan.documented` and your wired
labels ∈ `scan.all_labels` \ `scan.documented`. Sentinel placement is
indent-agnostic to the parser (`line.strip()` first), but MATCH the enclosing
block's indent for RST rendering (3-space inside a list item / `.. warning::`,
2-space inside a bullet). The `.. (vv-status rationale)` prefix is parser-safe
(the regex needs `vv-status:` immediately after `.. `, and `(vv-status` fails
it). Template-B retitle: COPY the proven underline from the model page
(`collision_probability.rst`'s "Verification — what pins this chapter" =
37 `=`), never re-count by hand.

How to apply: classify each orphan by the 3-shape SENTINEL test / the
independent-reference WIRE test; treat a root narrative page as all-SENTINEL;
resolve foundation-file labels per-test (computed formula → WIRE, invariant →
SENTINEL); trust the doc's named gate over a stale line-range; self-check with
the direct theory-scan, not the full audit. Cross-links [[lessons-L03]]
(phantom/verifies-target hygiene), [[lessons-L04]] (representational →
documented), [[lessons-L10]] (Mode-12 spectral-invisibility vocabulary).

---

## L-036 — GROWING a thin "honest-stub" chapter to full at campaign close: flip the stale status, PRESERVE the landed-earlier section, RECONCILE sibling taxonomies, and deferred-wire the verifies-targets

The A6/ch15 shape (a campaign's closing docs phase): the chapter already
EXISTS as a deliberately-thin honest stub (`methods/sn/adjoint.rst` was
"deliberately thin today ... two layers landed, the third in flight") and
is ALREADY in the part toctree. The task is GROWTH, not authoring-from-
scratch, and it has a fixed anatomy:

- **The primary staleness is the stub's own "in flight / not yet landed"
  status** — flip it (L-007 tense-flip) the moment the campaign's phases
  (here A4/A5) merged. Verify the merge against git/the campaign plan's
  STATUS log, never the stub's frozen prose. The "three layers, two
  landed" framing becomes "three layers, all landed".
- **PRESERVE the already-landed section verbatim.** The thin chapter's one
  substantive section (here the #276 P3 `sn-scattering-adjoint` record)
  stays byte-for-byte — its `:label:`s are live #309 wiring-backlog items
  with exact vv-status rationales; touching them re-opens an adjudicated
  question. GROW AROUND it: new sections slot before (physics + route +
  taxonomy) and after (mechanics + carrier + verification + consumers +
  Development history), the preserved section sitting where it belongs in
  the new flow (S^T is a concrete instance of the "Euclidean transpose"
  category, so it lands right after the three-transposes taxonomy).
- **TAXONOMY RECONCILIATION is the sharpest new move.** When the charter
  asks for a NEW canonical "named landmine" section (the three transposes:
  Euclidean / Hilbert / continuous) that SUBSUMES ≥2 pre-existing sibling
  framings (loss_representation's warning contrasts {walk-Euclidean,
  μ-reversal, continuous}; the thin Key Facts named {Euclidean, Hilbert,
  walk-orientation}), do NOT contradict them — write the chapter as the
  authoritative RECONCILIATION with explicit subset relations:
  walk-orientation ⊂ Euclidean (the streaming realisation of Aᵀ),
  μ-reversal = the continuous adjoint's discrete SIGNATURE, Hilbert rides
  ON TOP of Euclidean via G. A reader who meets any sibling framing
  elsewhere lands on the same taxonomy. State the reconciliation in prose
  ("all three framings are the same taxonomy") so no future reader reads a
  contradiction.
- **Deferred-wire the verifies-target eq-labels you mint** (L-004 #3, the
  concurrent-owner case, made concrete): the certification/entries tests
  carry a comment "verifies un-linked until A6/ch15 mints the
  daggered-eigenproblem label" — they are WAITING for your `:label:`. Mint
  it UN-sentineled (a solver claim with a real L1 gate is a genuine gate —
  NEVER sentinel to paper over the transient orphan), and report DEFERRED
  WIRING with exact `test node → label` node-ids for the main agent (who
  owns the test files this phase). The `-E -W` build passes regardless
  (the audit is a SEPARATE gate the main agent runs AFTER wiring); flag
  the audit dependency loudly ("these N labels are orphans until wired").
  Definitional/literature siblings you mint alongside (the continuous
  adjoint equation) DO get `.. vv-status: <label> documented` — audit-clean
  at your build. Net: for a chapter minting a mix, some labels are
  documented-sentinels (clean now) and 1–2 are un-sentineled deferred-wires
  (clean after the main agent's marker commit).

**Two teaching-doc CORRECTNESS catches this class surfaces** (Cardinal
Rule 1 over faithful transcription):

- **Don't fuse an eigenvalue term and a fixed-source term in one
  equation.** The continuous adjoint written with BOTH a `1/k`-scaled
  fission gain AND an external source `q*` is inconsistent (the `1/k`
  belongs to the eigenvalue problem where `q*=0`). Present the eigenvalue
  form (labeled), then the fixed-source form (`q*=Σ_d`, no fission) in
  prose. A code/brief that hands you "the adjoint equation" generically
  can hide this fusion.
- **A code docstring's operator NAME can be sloppy where the theory must
  be exact.** The `KEigenvalue(A.H, (S+B).H, F.H)` spelling in the
  eigenvalue.py seam + the `solve_sn_adjoint` docstring uses "A" for the
  FIRST arg — which is the RESOLVENT `(L+C)`, NOT the loss
  `A_loss = L+C−S−B` (reading it as A_loss double-subtracts). The code is
  `KEigenvalue(resolvent.H, gain.H, F.H)`; spell `(L+C).H` in the teaching
  doc, unambiguous, and show `A_loss† = (L+C).H − (S+B).H` formed inside.

**Mode-12 live-application EXACTNESS is the vv-curator load** (Directive
5; this campaign twice caught a wrong-WHY here, so the prose is the
corpus's quoted spelling). For the daggered adjoint: `k` is EXACTLY blind
to (i) the factor-ORDER / similarity family (`eig(Mᵀ)=eig(M)` — ALL
factors transposed is a similarity), (ii) ALL vector content, and (iii)
**the G-metric itself** (`G'⁻¹AᵀG'` is metric-similar to `Aᵀ` for ANY
invertible `G'`, so the metric is a free parameter no eigenvalue gate can
EVER see — the sphere vector row is its SOLE catcher, ERR-067 family). But
leaf-transpose **DROPS** (F†→F etc.) are **NOT** k-blind — transposing ONE
factor is not a pencil similarity, k MEASURABLY moves (F†=F: 1.488→0.153
on the 4G ∞ fixture). Get the "blind-to vs not-blind-to" boundary exactly;
the corpus page and the `vv-principles` Mode-12 text must carry the same
measured spelling. Pair every `k`-row prose claim with the vector/pairing
catcher (spectrum, bi-orthogonality, duality, sphere-residual).

**Xref reality for a solver-return-type chapter.** Only the module
`automodule`'d WITHOUT `:noindex:` links (here `numerics.eigenvalue` →
`power_iteration` links); a return type in a NOT-automodule'd module
(`sn.solution` → `AdjointSolution`/`Solution`/`SolutionBase`) and a
`:noindex:`-automodule'd module (`sn.solver` → `solve_sn_adjoint`) BOTH
render plain-text BY CONVENTION (L-002) — consistent with the pre-A6
chapter's own `ScatteringOperator` refs. NOT a defect and NOT to be
"fixed" by adding an automodule: `sn.solution` carries `.. math:: :label:`
docstrings (homogenize/condense derivations), so automodule'ing it trips
duplicate-label collisions. Spot-check by (a) `-E -W` build EXIT 0
(catches intra-doc `:ref:`, all `:eq:`, `:cite:`), (b) grep the built HTML
for RAW `:class:`...`` role markup leaking (means a broken role) — none
should leak; every role renders as `<a>` link OR plain `<code>`, and (c)
confirm the ONE indexable module's refs actually link (typo-catch).

---

## L-037 — FLIPPING a "documented-future seam" to LANDED across an existing rich page: the stale-status blast radius is the WHOLE page (grep it), the wrong-rule can hide in a literature-table cell, and the wired⟹no-sentinel flip is verified against the LIVE test

The sibling of L-036 (which GROWS a thin stub chapter). Here an
EXISTING, rich foundations page carries a `documented-future seam` for
campaign X that just landed; the task is (a) flip every stale-status
claim to landed, (b) grow the seam section into the full landed
taxonomy, (c) un-sentinel the now-wired labels. Three distinctive
disciplines, none of which the brief's named file/line list fully
scoped:

- **The stale-status blast radius is the WHOLE page, not the brief's
  named 2–3 sites.** A "campaign X landed" brief names the capstone
  status block + one table row + one passage — but the SAME
  "blocked / not built / pending / documented-future seam / lands with
  P6 / in flight" claim is SCATTERED across Key Facts, the chapter
  overview bullet, section-body prose, a second table, AND a
  "V&V evidence lands with P6" closing line. Grep the page for EVERY
  future-tense/blocked token about X
  (`blocked|not built|not yet|pending|in flight|future seam|documented
  theory only|lands with P6`) and flip each on correctness grounds
  (Cardinal Rule 1) — the brief's list is the FLOOR. Worked (#281 P6):
  the brief named 3 sites; the grep surfaced SEVEN. Watch the
  "one remaining not-built discipline is X" overview bullet especially
  — when X lands, that sentence must RE-POINT to the *actually*-still-
  unbuilt sibling (here the least-squares dense-cross-Gram frame), not
  just drop X; a naive delete leaves the "one remaining" count wrong.
- **A "fix the loose (φ→φ*) phrasing" sub-task has a blast radius too —
  and the wrong rule hides in a LITERATURE-TAXONOMY TABLE CELL.** The
  brief named 2 prose sites; grep for the concept surfaced a THIRD in a
  "canonical pairs" table row. The tell for the bare-φ* trap is not the
  test-basis cell alone — it is the test/trial PAIRING: a row that lists
  `test = φ*·1_R` against an INDICATOR `trial = 1_R` silently encodes
  the bare-adjoint rule `∫φ*Σ/∫φ*` (worth-nonzeroing), NOT the bilinear
  `∫φ*Σφ/∫φ*φ`. The correct cell, matching the landed code (a capture
  gate asserts the frame weight IS the pair), is the PRODUCT weight
  `(φ*⊙φ)·1_R` against the indicator trial. Discriminator: with an
  indicator trial, the weight must be the PRODUCT; only a φ-weighted
  trial makes a bare-φ* test correct. Fix it, flag the scope-expansion.
- **The wired⟹no-sentinel flip is verified against the LIVE test, not
  the brief's assertion.** The brief said "both labels NOW carry
  verifies() (C1 stacks both; C4 stacks both) — REMOVE the sentinels."
  Per L-036's deferred-wire case a brief can say "wired" when the test
  is actually a WAITING verifies-target, so READ the live test files
  FIRST (`grep -n 'verifies\|class Test' <file>`) to confirm the
  `@pytest.mark.verifies("<label>")` decorators are really present and
  stacked — here they were (C1 stacks both labels, C2/C4 stack them).
  Then remove BOTH the `.. vv-status: <label> documented` directive AND
  rewrite (do not delete) its `.. (vv-status rationale)` comment to a
  plain `.. (Wired P6, #281 — no vv-status sentinel.) …` note naming the
  gates — the note prevents a future auditor from "helpfully" re-adding
  a sentinel to a long-documented-only label whose neighbours still
  carry them. Self-check with the FAST theory-scan (L-035): the flipped
  labels must show `label_exists=True, documented=False` with 0
  file-local violations (the label left the documented set cleanly and
  is now a covered verifies-target).
- **Grow the taxonomy by INCLUDING the generated fragment, adding
  UNLABELED supporting math, and keeping the ONE preserved verifies-
  target label byte-identical.** The five per-channel collapse rules
  come from `.. include:: ../../_generated/<name>.inc.rst` (same
  `../../_generated/` depth from `docs/theory/foundations/` as from
  `docs/theory/verification/`), NEVER hand-transcribed (L-008). The T0
  keystone / T1b angular / T4 balance / T6 carrier equations I add as
  the narrative are SUPPORTING identities — leave them UNLABELED (no new
  orphan/sentinel obligation; `git diff | grep ':label:'` must show ZERO
  net label change). The single labelled equation the section owns
  (`sn-homogenization-adjoint-weighted`, a verifies-target) is re-emitted
  BYTE-IDENTICAL inside the rewrite so git matches it as unchanged
  context — never rename a verifies-target while rewriting its prose
  (L-003).

How to apply: for a "campaign landed, modernize the page" task, (1)
grep the WHOLE page for stale-status tokens and flip each; (2) grep the
wrong-rule concept (not just the brief's 2 sites) — it hides in tables;
(3) verify verifies() markers in the LIVE test before un-sentineling,
then fast-theory-scan; (4) include the generated fragment, add
supporting math UNLABELED, keep the verifies-target label byte-identical.
Cross-links [[lessons-L36]] (the stub-growth sibling), [[lessons-L35]]
(the fast theory-scan self-check + WIRE/SENTINEL discriminator),
[[lessons-L03]] (never rename a verifies-target), [[lessons-L08]]
(generated artefacts are included, never hand-edited).

---

## L-038 — Auditing a "is the terminal docs phase done?" charter: a multi-phase campaign's LAST docs phase is often already-executed incrementally by the earlier phases' doc passes — verify by the page's OWN self-identification + build + cross-ref gate, don't infer a gap from an open plan line

A read-only "how much of Phase-N docs is already satisfied?" audit (here
the frame-projection campaign's P7 charter) has a recurring answer:
**effectively-done**, because each earlier phase (P3/P5/P6) ran an
archivist doc-pass that landed its own slice INTO the eventual capstone
page, so the "final docs phase" was executed piecewise before it was
formally reached. The audit discipline:

- **The plan's phase-line is a STALE tracking artifact, not the ground
  truth** (process-discipline: trust git/the shipped page, not a frozen
  plan claim). The driver plan still read "NEXT = P4.5 … P7 pending" while
  P4.5–P6 had all landed and the P7 page was written. NEVER infer "P7 is
  a gap" from an open plan bullet — read the shipped page.
- **A campaign's capstone page usually SELF-IDENTIFIES.** The decisive
  evidence was the page's own front-matter note titled "What shipped since
  (P3 / P5 / P7)" stating *"This page (P7) is the capstone…"*. Grep the
  candidate page's intro/Key-Facts/notes for the phase tag (`P7`, the
  issue #) — the campaign often already declared the page done in prose.
- **The plan's task-number ≠ the GitHub issue number.** The plan said
  "Tasks #46–#52 (P1–P7)"; #46–#52 are actually unrelated
  Thermal-Hydraulics/Kinetics issues — the plan used INTERNAL task
  numbering that COLLIDES with real issue numbers. The real trackers were
  the phase issues (#268/#226/#281/#275). Resolve "which issue tracks
  this?" by reading each candidate's title, never by trusting a plan's
  bare `#N`. A terminal-docs phase frequently has NO dedicated issue —
  its deliverables ride the phase issues' doc-passes.
- **Per-item verdict method for a content charter:** for each named
  deliverable, (1) locate the anchor/label (`grep -rn "<label>"
  docs/`), (2) READ the section (not the heading) and judge it against
  the articulation standard (does it carry the rejected alternatives, the
  structural WHY, the honest-scope seam?), (3) confirm the `-E -W` build
  is clean (charter's "-W clean" clause), (4) grep-gate the cross-doc
  `:ref:`/`:eq:` targets the section uses (the -W-BLIND plain-text class,
  L-002) — a Feynman-grade section with a dangling cross-doc ref is not
  actually done.
- **Distinguish a documented SEAM from a GAP.** A charter is DONE even
  when the page carries "stays a documented seam until consumer X exists"
  notes (here: the anisotropic-order Σ_{s,ℓ} moment-resolved pairing; the
  LeastSquaresFrame/GEC-rank>0 #275). A correctly-declared future-consumer
  seam (L-002 forward-ref discipline: literal not premature `:class:`) is
  the OPPOSITE of a gap — it is the honest-scope boundary the charter
  never asked to cross. Do NOT list a documented seam as owed work.
- **A charter's literal "the condensation PAGE" can be correctly
  delivered as a SECTION of a shared page** (DRY, Cardinal Rule 2): one
  frame page with space + energy as sibling PG sections beats two pages
  that duplicate the PG machinery. Flag the wording-vs-form deviation as
  INTENTIONAL, not a missing page — recommending a standalone page would
  MINT a twin-path violation.

Net verdict shape for such an audit: "effectively-done; residuals are
bookkeeping (mark the plan line ✅, no dedicated issue needed for
already-shipped work) — NOT a gap-fill pass." Cross-links [[lessons-L37]]
(the flip-a-seam-to-landed sibling), [[lessons-L02]] (the -W-blind
cross-ref grep-gate), and AGENT.md process-discipline (trust git, not the
frozen plan claim).

---

## L-039 — AUTHORING a campaign-CAPSTONE theory page (a completed feature's WHOLE story) from an algebra-of-record + plan memos + the error catalog: the narrative arc is motivation→derivation-of-record→design→discoveries→evidence→scope, and the vv-status decision for algebra-of-record SymPy-identity labels is verifies-COVERED (peierls foundation+verifies), NOT documented

Distinct from L-025 (a NEW shared-INVARIANT foundations chapter: gather
method-specific → generalize) and L-013/L-018 (a resolution/capstone
chapter of an ARC on an EXISTING page): here the whole task is a NEW
standalone page telling a COMPLETED campaign's whole story (consistent
DSA #2 — Fourier motivation, the four-step derivation, the design
decisions, the discoveries, the measured evidence, the honest scope).
The source-reading order and the label-status decision are the
load-bearing lessons.

- **Source-reading order for a capstone (extends L-005):** (1) the
  ROADMAP/plan-of-record (the phase structure, the RULINGS with dates —
  R4/R5/R6, the deviations); (2) the LITERATURE MEMO (the equations with
  paper-numbers + the errata/normalization watch-items — here Alcouffe
  (17)/(23) sign errata, the Σw=1-vs-2 map); (3) the ALGEBRA OF RECORD
  (`derivations/discrete/sn/dsa.py` — the SymPy `derive_*` functions ARE
  the equations; READ it, don't transcribe the memo's paraphrase); (4)
  the PRODUCTION code (the shipped shape — admission guards, the trace
  arm, the foldable accessors); (5) the ERROR CATALOG (the discoveries'
  full stories — ERR-070/071 parts); (6) the EVIDENCE PACK (the
  authoritative measured tables — SELECT the load-bearing ones). The
  memo is the NAVIGATION layer; the SymPy module + production code are
  the CORRECTNESS spine (algebra-of-record skill). VERIFY every DD-member
  collapse against the LIVE production body (`cell_update` returns the
  edge average; `moment1_update` the (28b) form) — the page's worked
  forms are code-grounded, not memo-transcribed.
- **The vv-status decision for an algebra-of-record page is the sharp
  new call (extends L-004 to the AUTHORING case).** When the brief says
  "mint `:label:`s on the key equations, then add `verifies()` markers",
  the derivation-identity labels (Larsen (27), (23a–f), (28), Marshak,
  the (33) synthesis) are algebra-of-record SymPy-identity gates — WIRE
  them foundation+verifies → **covered**, the peierls precedent (L-004
  case b: `test_case_method_*` carry BOTH `@foundation` AND
  `@verifies`), NOT `.. vv-status: documented`. The audit ACCEPTS
  foundation+verifies (confirmed: `test_dsa_rules` foundation gates
  verifies-cover their labels, 0 orphans). Reserve `documented` for the
  PURE-LITERATURE / STRUCTURAL labels with NO tight test (the ρ_SI=c
  motivating collapse, the 0.2247c continuum bound, the M=(I+𝒞)∘(L+C)⁻¹
  composition identity). Discriminator: does a test genuinely PIN this
  exact equation? derivation identity / object law / rate bound with a
  gate → verifies-covered; motivating/definitional literature with no
  gate → documented. **Mechanics:** the audit's `testable_labels =
  theory_labels − documented_labels`, so a `documented` label is
  EXCLUDED from the orphan gate and a `verifies`-covered label is NOT an
  orphan — either avoids orphan regression, but documented+verifies
  TOGETHER is muddy (a documented label with a test edge); prefer the
  clean split. If you wrote `documented` on a label you then decide to
  wire, REMOVE the directive (keep the rationale as a plain `..` comment
  naming the catcher) before adding `verifies()`, and re-run the audit
  to confirm 0-orphan.
- **The capstone's Key-Facts + narrative sections earn 13 labels; the
  page-prefix (`sn-dsa-*`) + a grep-collision check before writing is
  mandatory (L-003).** Author the head/intro as pure Write/Edit literals
  (no Python f-string over math — the L-026 brace trap); the f-string
  mangle grep (`A\^-1|G\^-1`) confirmed clean because I never routed math
  through a Python string.
- **Migrate a gate-file docstring's MEASURED-FACTS record to the page,
  keep its CONTRACT (enforcer NOTE f1; extends L-028/L-033).** A rate-tier
  test module's docstring carrying a "Measured design facts" section
  (D11/S2/ladder numbers) MOVES to the theory page's evidence tier; shrink
  the docstring to a greppable pointer (`docs/.../acceleration.rst`
  §`sn-dsa-rate-and-stability`) that keeps only the test CONTRACT (the
  #215-catcher mutation-matrix statement). A SIBLING test docstring that
  is already a pure contract statement + ERR narrative (the sweep-inverse
  gate) STAYS — the enforcer note says "keep its contract statement", and
  the ERR-071 story's canonical homes are the page + `error_catalog.md`.
- **The brief's paraphrased numbers/paths/targets are STARTING HEURISTICS
  — verify against the evidence + the tree (L-001 across the board):**
  (a) the brief's "Krylov 2554→21" was SI+DSA reflective in the evidence
  pack (Part A) — used the authoritative table, flagged the paraphrase;
  (b) the brief's `docs/theory/methods/diffusion/diffusion_1d.rst` path
  had no `diffusion/` subdir (real: `methods/diffusion_1d.rst`); (c) the
  brief's "point the drifted xref at field_algebra" — the drifted target
  was `operator_algebra` (only 1 passing DSA mention), and the NEW
  capstone page is the authoritative DSA home, so I pointed it THERE
  (better target, flagged the deviation); (d) the brief's "flip the
  field_algebra as_dsa_source promise" was ALREADY the landed-truth
  (L-001 already-fixed — reported, no action). FLAG each deviation in the
  return; the brief is the floor, the live tree is the rule.

---

## L-040 — RETIRING a per-X flag from the docs: the blast radius includes the TABLE COLUMNS that paraphrase it without naming it, and the flag's own JUSTIFICATION prose is usually independently FALSE

The symbol-grep (L-002) is the FLOOR of a retirement's doc radius, not the
ceiling. A `ClassVar`/field is documented in two registers: by NAME (which
greps) and by CONCEPT (which does not). Both must die.

- **Grep the symbol AND its human-readable paraphrase.** A brief listing
  N literal hits is complete only for the NAME register. Worked
  (`BoundaryTraceLaw.creates_sweep_cycle`, 2026-07-30): the brief's 7
  literal hits were exact — and MISSED 17 more cells, because the
  foundations page tabulated the flag under the header **"Sweep-cycle
  flag"** (6 per-law `True`/`False` values) and the SN page's resolution
  table carried 10 value cells under a header that DID grep. The
  paraphrase grep (`grep -rni "sweep.cycle"`) is what found them. RULE:
  after the symbol grep, grep the CONCEPT the symbol names (hyphen/space
  variants, the column header you'd write for it) — a `list-table`
  column is a documentation surface with no symbol in it.
- **Dropping a table column is a 3-part edit**: header cell, every row's
  value cell, AND the `:widths:` list (which must still match the new
  column count). Verify in the RENDERED HTML, not the source: parse the
  built page for `<col>` count + `<th class="head">` list + per-`<tr>`
  `<td>` counts. A widths/column mismatch is a real `-W` warning, but a
  silently-wrong-but-consistent table is not.
- **A column can often be REPLACED rather than deleted** — with the true,
  intrinsic property the false one was gesturing at. The `Sweep-cycle
  flag` column became `Trace-edge family` (none-the-inflow-is-data /
  same-face back-edge / opposite-face pair), read off the replacement's
  algebra of record. This keeps the pedagogical slot, makes the claim
  true, and teaches the exact distinction that killed the flag (the law
  owns its EDGE STRUCTURE; the configuration owns CYCLE-NESS). Prefer
  replace-with-the-true-invariant over delete-and-leave-a-hole.
- **The paragraph that JUSTIFIED the flag is the highest-risk prose on
  the page — re-verify it against the replacement, don't just delete the
  flag's name from it.** A per-X boolean's doc always ships a "and here
  is why X is False for these kinds" closing paragraph, and that
  paragraph inherits the flag's wrongness. Here: "Vacuum, white, albedo,
  and prescribed-inflow are all cycle-free" — FALSE for white, since
  `white|white` is cyclic for the same reason `reflective|reflective` is
  (the replacement gate's OWN docstring says so: "white on BOTH faces is
  not [acyclic]"). Read the replacement module + its test docstrings and
  rewrite the justification to the structural truth (only laws that
  supply the inflow as DATA contribute no edge at all, hence are
  unconditionally cycle-free). Cardinal Rule 1 outranks minimal-diff.
- **Option (b) — keep a retired-note section — over option (a) delete —
  whenever the retirement carries a DESIGN LESSON, and the L-007
  retitle-to-the-concept/KEEP-the-anchor move is what makes it free.**
  Retitling `The ``creates_sweep_cycle`` signal` → `Sweep cycles: a
  configuration property, not a per-law flag` kept the `bc-sweep-cycle`
  anchor, so both live `:ref:`s (one CROSS-DOC, which would have
  silently rendered plain-text if the anchor died — L-002) kept
  resolving AND auto-picked up the new title as their link text. Verify
  that payoff in the HTML (`grep 'href=".*#anchor"'` → confirm it is an
  `<a>` carrying the NEW title).
- **Structure the retired-note as an increasing-importance ladder, and
  put the un-fixable reason LAST.** Three findings: (1) zero production
  readers, (2) the attached claim was false, (3) it could not have
  worked in principle. Only (3) generalizes — so it gets the space, the
  measured truth table, and a named design rule in an `.. admonition::`
  ("a law may carry only what is intrinsic to it"). (1) and (2) are
  facts; (3) is the archaeology that stops re-invention. Sharpen (3)
  with the "one value, two different facts" tell: the flag read `True`
  for reflective meaning "can take part in a loop others close" and
  `True` for periodic meaning "closes a loop alone" — one boolean
  carrying two structurally different claims IS the proof the property
  does not live on the type.
- **Name the replacement's V&V level from the gate's marks, not from
  instinct.** The SCC gate is `@pytest.mark.foundation` with NO
  `verifies(...)` (verifies ⊥ level) — so the prose says "software/
  structural invariant of a discrete construction, not an equation
  claim", and cites the mutation-teeth test by what it proves (dropping
  the boundary edge FALSELY certifies acyclic). Never upgrade a
  foundation gate to an L-level in prose to make the section sound
  better-verified.
- **Running the mandated `-E` build REGENERATES `docs/theory/verification/
  matrix.rst`** (L-008 `builder-inited` hook) — and on a dirty branch it
  will absorb rows from OTHER uncommitted work (here +126 foundation
  tests from a sibling campaign, plus the ERR count). That is a
  legitimate by-product, NOT your edit; never revert it (it is
  generated), and REPORT it explicitly so the committer knows what the
  extra modified file is and can choose to stage it.

---

## L-041 — The DOC-ONLY "retire the false promises" pass on a subsystem under structural restoration: keep the declaration, make the CLAIM true; and prove doc-only by AST, not by eyeball

A B0 "clean before extending" phase hands you N measured docstring/prose
claims naming a consumer, capability or behaviour that does not exist. The
job is NOT deletion — it is making each claim TRUE while preserving what a
later phase needs. Disciplines that held across 18 items in 8 files
(boundary machinery, 2026-07-30):

- **Retire the CLAIM, keep the DECLARATION, and name the phase that will
  fill it.** When a brief says "do NOT delete these properties — phase B1
  populates them", the honest rewrite is three-part: state the measured
  present ("**currently unpopulated** — returns ``None`` on every law, read
  by nothing"), state HOW production reaches the same information today
  (the realizers recover `G` from the law's CLASS and `R` from
  ``law.albedo``), and name the landing ("**B1** mints the typed spec and
  populates this; the declaration is kept for that landing"). A reader then
  cannot mistake the property for live machinery *or* delete it as dead.
- **MEASURE the override lattice; never trust a "where applicable" hedge.**
  A doc saying "each subclass overrides the five universal invariants where
  applicable" is unfalsifiable prose over a lattice you can compute in ten
  lines: `ast.parse` each base body and count non-docstring statements;
  compare `getattr(Law, m).__func__ is not getattr(ABC, m).__func__` per
  (law × method). Here it produced the load-bearing numbers — 4 of 5 bases
  EMPTY, 2 of 5 overridden by NOBODY, 4 of 7 laws overriding nothing — and
  turned one hedged sentence into a per-row "*Intended* … / *Implemented*
  …" table. Also grep `raise <TypedError>` per row: a "Pinned error" column
  can name an exception **never raised anywhere in production** (ERR-040
  here), which is a second, independent hollowness.
- **A false-promise item's real scope is the CLAIM'S grep, not the brief's
  file:line.** Two of the strongest finds came from the closing gate, not
  the list: the same "downstream consumers (sensitivity adjoints) require
  the outflow trace" justification lived in a SECOND file
  (`sn/boundary/realizer.py`), and a self-contradiction the brief located
  in a docstring had a TWIN in a `#` comment 400 lines away (`B_b` …
  "present-zero bulk and trace" vs the class docstring's "the composite has
  no such slots"). Build the gate as one grep per item, run it AFTER the
  edits, and treat every extra hit as in-scope.
- **When two symbols name the SAME concept, the fix is a note that TYPES
  them, plus the collision's own tell.** A package writing `R = G_refl · α`
  (R = composite) beside an ABC writing `γ₋ψ = R G γ₊ψ + q` (R = response
  factor) will corrupt the phase that mints `R` as a type. Resolve by
  splitting every bullet into `G = …` / `R = …` and adding ONE `.. note::`
  stating the convention and naming what the composite is called instead
  (`R ∘ G`, never `R`). Then grep the concept: sibling modules using a
  DIFFERENT decomposition (`R_white = G_diff ⊗ α`, `R = Σ_α c_α G_α`) are a
  separate framing — FLAG them, don't sweep them into a scoped fix.
- **An error-message string inside `raise` is an EXECUTABLE statement —
  report it, don't edit it, under a doc-only constraint.** `apply_transpose`
  routing through a shared `_apply_faces` whose message says "``.apply``" is
  a real defect, but tests `pytest.raises(match=...)` on those strings.
  Report with file:line + the exact string + the reason it was withheld.
- **Prove doc-only with an AST gate.** Strip docstrings from every
  `Module`/`ClassDef`/`FunctionDef` body and compare `ast.dump(HEAD)` vs
  `ast.dump(now)` per file — "AST IDENTICAL" is the claim, a diff hunk is
  the exception. On a file the USER also edited this session, `ast.unparse`
  the stripped trees and `difflib` them: the printed hunk should be exactly
  their change (here one `kind` property) and nothing of yours. This is
  stronger and faster than reading a 450-line diff.
- **A brief's enumerated item can be a LATER phase's acceptance-gate text —
  leave it and say so.** The `SNBoundaryOperator` docstring's "``.H`` — the
  one channel by which the white-BC adjoint becomes available" is FALSE
  today (measured: `B.H` raises with a white face) but is quoted verbatim
  as phase B5's gate. Editing it would destroy the gate. Not on the item
  list ⟹ report as a deliberate non-edit with the owning phase.
- **Verify every issue number you cite** (`gh issue view N --json state`) —
  a docs pass that redirects a retired claim to "#183 tracks this" is
  minting a new claim, and a closed/mis-numbered issue is the same class of
  defect you are removing.
- **Baseline drift is real: this repo's `-E -W` baseline is now ZERO**
  WARNING/ERROR/CRITICAL (AGENT.md still says 1 — the `Mesh1D.from_geometry`
  `:paramref:` ERROR is gone). Re-measure the baseline every session; never
  assume the recorded number.

---

## L-042 — Auditing a corpus against a just-landed multi-commit REFACTOR: the phase-lag, the letter collision, and the retraction that INVERTS

The successor task to L-041's doc-only B0 pass: three commits land
(factor re-assignment · a new primitive · a domain narrowing) and you
must find and fix every `docs/theory/**` claim they falsified. 46 claims
adjudicated across 6 files; disciplines that were NOT already in L-041:

- **A brief's lead can be directionally inverted — settle it with
  `git show <fix>^:<path>`, not the ⚠ alone.** The lead said "the page
  says `apply_transpose` writes the `outflow_indices_for_face` slots;
  the code's ⚠ names that the WRONG spelling." Reading the ⚠ *precisely*:
  the wrong spelling is scattering over the law's own **codomain**
  (Γ₋); post-change the transpose genuinely DOES land on the Γ₊ rows,
  so the sentence is incidentally true NOW. The doc was still wrong —
  because it described an output **projection** that the PRE-change code
  never performed either (it wrote the whole face). Only the pre-commit
  body settled which of three readings was right. A ⚠ names a HAZARD,
  not what the code did; when a lead claims "the doc documents a known
  bug as the contract", read BOTH bodies.
- **A phase-N doc pass leaves phase-(N−1)'s falsifications behind — audit
  the PARAGRAPH FAMILY, not the commit's diff.** Phase B1 populated the
  two factor properties on all seven laws and never touched the theory
  page; B3.0's doc pass fixed only what B3.0 moved. Result: a correctly
  re-typed G/R section sitting three screens from "`geometry_map` and
  `response_kernel` return the ABC's `None` on **every** law and are read
  by nothing". The reader cannot tell which phase staled what, so a
  scoped-to-phase-N audit ships a self-contradicting page. Fix it and
  FLAG the scope expansion. (The `-E -W` build is blind to all of it.)
- **Replace an unfalsifiable inventory sentence with a MEASURED table**
  (the L-041 override-lattice move, now for property VALUES): one
  `python -c` over the seven laws printing `law.geometry_map` /
  `law.response_kernel` turned the false sentence into a 7-row
  ground-truth table the next reader can re-run.
- **One letter, two decompositions, two pages — type them both in ONE
  `.. warning::`.** The rank-N expansion `B = Σ G_α ⊗ A_α` (a sum over
  TERMS) collides with the affine factorisation `R G` (a factorisation of
  ONE term). Tell: a Marshak formula `R = c₁G_refl + c₂G_diff` that uses
  `R` for the whole composite AND files a Lambertian average under `G`.
  Fix = rename the colliding prose symbol (`R`→`B`), add ONE warning
  stating both decompositions and naming the composite **`R ∘ G`, never
  `R`**, then correct the mis-tiered rows. Extends L-041's same-module
  collision to the cross-page case.
- **A retraction can INVERT a claim, not just kill it — give each item its
  own `**Disposition:**`.** Three published "why this matters"
  consequences: #1 *measured not to exist* (the phantom future consumer —
  the declared-capability-no-consumer pattern), #2 **inverted** (the
  argument was "the realization is a self-adjoint idempotent projector";
  once the domain narrows, the operator is not an endomorphism, so
  idempotence is not even a well-typed question and the type tag the
  argument rejected is now the right one), #3 *right observation, wrong
  layer* (the uniformity was real; the mechanism was one layer too
  shallow). Per-item dispositions preserve the intellectual content that
  a blanket tombstone destroys — and #2's inversion is the single most
  instructive line on the page.
- **A "the gate still does X" claim is verified against the TEST BODY,
  and you must COUNT the rows.** A snapshot suite's vacuum case had been
  re-posed in the body while its class docstring still described the old
  semantics (so quoting the docstring would have re-minted the falsehood
  — L-001 in a test file). And only **3 of 7** cases were re-posed: the
  un-narrowed four still feed the full face. My first draft said "every
  case was re-posed" — a fresh falsehood, caught by reading all seven
  bodies. Also: the mixed-BC row is an `xfail(strict=True)`; document it
  as an **honest red that flips on the next phase**, never a suppression.
- **Put the Mode-12 blindness IN the prose, beside every table of
  realized operators.** Measured `|Γ₊| = |Γ₋|` on every quadrature × face
  ⇒ a shape assertion cannot distinguish `Γ₊→Γ₊` from `Γ₊→Γ₋`. The next
  reader's instinct is to "check" a typing claim by output shape; one
  sentence ("read the *declared spaces*, never the output shape") is
  worth more than the table it annotates. Same family: a three-way
  partition (inflow ⊔ outflow ⊔ **tangential**) needs "**not inflow** is
  NOT **outflow**" restated wherever a complement could be spelled.
- **`python -c` every numeric constant a doc asserts.** A page said the
  tangential band is "default `ε = 1e-12`"; measured
  `TANGENTIAL_EPS = 4·np.finfo(float64).eps ≈ 8.9e-16` — four orders out,
  never warned, pre-existing.
- **A "mitigation" sentence can be a NON-SEQUITUR wearing a hedge.**
  "The adapters carry every tangential ordinate at μ = 0 **so** ψ = 0
  there for a properly-initialised flux": μ≈0 is the *definition* of
  tangential, and it does not make ψ vanish. The honest weaker statement
  (no operator writes tangential slots, so a zero-initialised carrier
  keeps them zero) is an INITIALISATION property; the genuinely
  structural neighbour (tangential rows carry zero *metric weight*) is a
  different fact. Split them.
- **The taxonomy the brief asks you to ADD may already exist in a
  docstring the concurrent agent just wrote.** Grep the label before
  authoring. Here `.. _bc-method-realizability:` was already DEFINED in a
  package `__init__` docstring — inert only because that package is not
  `automodule`'d. The project pattern is **page owns the label, docstring
  `:ref:`s it**; use the same label name on the page so the two agree,
  and report the latent duplicate rather than inventing a second name.

---

## L-043 — The brief's "MEASURED evidence, do not re-derive" block is a CLAIM, not data: reproduce it, because a wrong attribution there propagates into the corpus as fact

L-001 says verify the brief's *facts*; L-043 is its numeric face, and it
is sharper because the brief actively tells you **not** to check. A
doc-repair brief that hands you a measured-evidence block ("bit-identical
on A and B, 1 ULP on C and D — do not re-derive") is handing you the
single most quotable content in the whole task: numbers go into a table,
and a table is what the next session cites. If the attribution is wrong,
you have laundered an error into the algebra of record with your own
credibility on it.

- **"Do not re-derive" means "don't burn a session on it", not "don't
  check".** Cost-scope it: reconstructing a retired operator body from
  the `git diff` you already read is ~10 lines. Do that. Reserve the
  deference for things that genuinely need a solver run.
- **Worked (B3.4a):** brief said white was *bit-identical* on
  `gauss_legendre(8)` AND `product(2,4)`, 1 ULP on `lebedev(17)` /
  `level_symmetric(6)`. Reproduction: `product(2,4)` is NOT bit-identical
  — and it is precisely the quadrature where the retired `> 0.0`
  classifier disagrees with `TANGENTIAL_EPS`, so it CANNOT be
  (the two operators are not computing the same functional there).
  Publishing the brief's line would have told a future reader "the
  change was FP-neutral everywhere", **exactly wrong on the one
  quadrature that motivated the phase**.
- **The deeper find, and the reason to reproduce at all: two mechanically
  DIFFERENT effects can measure the same.** Reduction-order drift
  (padding removed → numpy's pairwise tree reassociates) and a genuine
  mis-classified-row VALUE bug both read ≤ 1 ULP on an `O(1)` probe —
  because the offending weight is itself `O(ε)` (measured `cos_w` =
  7.85e-17 against a norm of 2.5651, so Δnorm = **0.0 exactly** and the
  entire discrepancy is ψ-weighted in the NUMERATOR). So the error scales
  with the **flux ratio**, is unbounded by floating point, and reaches
  6.1e-05 at a 1e12 ratio. A ULP table therefore CANNOT justify such a
  change; the justification is structural ("one classifier, not two").
  Say that in the doc — and add a `.. warning::` that the 1-ULP row is
  not evidence of equivalence, or the next reader re-derives the wrong
  conclusion from your own table.
- **While the probe harness is open, AUDIT THE WHOLE INVENTORY — the
  brief's sample is never the population.** Sweeping all production
  quadratures × all six faces (~15 lines) turned a two-row anecdote into
  a scoping law: the disagreement occurs ONLY for the `product` family
  and ONLY on `xmax`/`xmin`/`ymax` (`ymin` has the same tangential count
  and zero mis-admissions — the sign flip moves the round-off across
  zero). Two publishable consequences fall out that no one had stated: a
  **tangential-count audit is not a sufficient screen** (`lebedev` has 12
  per face and mis-admits none; `level_symmetric` has none at all), and
  the exposure is **face-asymmetric within one quadrature**, so a fixture
  exercising one face can be green while its opposite is wrong.
- **The audit routinely falsifies a claim in a file you may not edit.**
  Here it killed "that is every production quadrature but
  `gauss_legendre`" in a `.py` docstring outside the permitted set. FLAG
  it with the measured replacement sentence verbatim, so the fix is a
  paste for whoever owns the file — a flag without the corrected text is
  a ticket, not a hand-off.

How to apply: treat every number the brief marks MEASURED as
unverified-until-cheap-repro; reproduce, widen to the full inventory,
and publish only what you measured — attributing anything you could not
reproduce to its source rather than adopting it.

---

## L-044 — A retirement's doc blast radius in UN-autodoc'd modules is invisible to `-W` AND to `-n`; and the same retirement silently DEMOTES the rewired tests' claim class

Two findings from one sweep (retirement of `orpheus/sn/quadrature.py`:
4 per-family adapter classes + an `AngularQuadrature` Protocol → named
classmethod factories on the single `Quadrature`).

**(a) Nitpicky mode is NOT the missing gate.** L-002 says a dead
Python-domain role renders plain-text with no `-W` warning and suggests
`-n` as the stronger probe. MEASURED here: `-n` saw **zero** of the 22
dead `orpheus.sn.quadrature` refs — because Sphinx only nitpicks what it
RENDERS, and none of the carrying modules (`numerics/measure.py`,
`numerics/operator.py`, `numerics/quadrature/*`, `geometry/reduced_operator.py`)
is `automodule`'d, nor is any `tests/**` file ever read. So for a
docstring in an un-autodoc'd module the ONLY gate is `grep`. Before
concluding "`-n` would have caught this", check `grep -c "docstring of
<module>" <nitpick.log>` — 0 means the module is invisible at every
severity. Corollary for the fix: edits to such docstrings CANNOT move the
warning count, so "count unchanged" proves nothing about them — the
acceptance evidence is the grep inventory plus a per-hit
KEEP/FIX adjudication, and the build is only a no-regression control.
(The `-n` counts were 6493 py-domain lines / 6964 total, byte-identical
pre and post by set-diff; the normal `-E -W` build is at **0** warnings
now — the `mesh.py` `:paramref:` baseline in AGENT.md is STALE.)

**(b) The sharper find — a retirement can DEMOTE a gate's claim class
without touching one line of the test body.** Four tests named
`test_*_bit_identical_to_legacy_adapter` had their comparison target
`sed`-migrated by the retirement commit (`LebedevSphere(order)` →
`Quadrature.lebedev(order)`), keeping the local name `legacy`. But the
factory *calls* the very rule function under test, so what was a
two-implementation bit-identity gate became a value compared with itself
routed through a wrapper — it can NEVER detect the node drift its
docstring still advertised. The test stayed green, the name stayed
authoritative, and the docstring kept the stronger claim. RULE: when a
retirement rewires a comparison target, re-ask **"are the two sides still
independently produced?"** — if the survivor is the caller of the other,
the gate has silently dropped to a pass-through check and every doc/
docstring crediting it must be re-scoped (name the real pin — here the
cylindrical regression snapshots — rather than deleting the gate's
description). The tell in a diff is a variable still called `legacy`
beside a brand-new API. The tree had already self-corrected exactly ONE
of the four (`test_rules_1d.py`, with the excellent phrasing "compares a
value with itself routed through a wrapper … the drift gate is the
separate test below") — reuse an in-tree honest framing when one exists
rather than inventing vocabulary.

**Adjudication shape that worked.** Every surviving mention became a
past-tense double-backtick LITERAL ("the four SN-side wrappers this
docstring used to point at (``…LebedevSphere``) were retired into
classmethod factories on the one ``Quadrature`` type"); every live claim
got a `:meth:` at the successor. Two mechanical gates make the sweep
auditable: `grep -rn "<retired path>" … | grep -v '``'` must be EMPTY (no
surviving hit outside a literal) and a role-regex over the retired names
must be EMPTY. Both beat reading the diff. Widen the grep from the
module PATH to the bare CLASS NAMES — that surfaced 3 more dead roles
the path-grep missed, and separates the genuinely-dead from the
live-homonym (here `name="LebedevSphere"` is a **live registry
`QuadratureSpec` identifier** asserted by ~30 tests — never "fix" it).
The residue — a retired class name used as an informal FAMILY LABEL in
prose ("ProductQuadrature(2x4)", "LS-family (``LevelSymmetricSN``…)"),
~25 sites — is a separate sweep: FLAG it, don't half-do it.

How to apply: for any retirement, run the grep inventory yourself and
adjudicate hit-by-hit (history KEEPS, present-tense-false FIXES); do not
let a clean `-n` build imply the docs are clean, and re-derive what each
rewired test can still SEE before repeating its old claim.

---

## L-045 — `tools/check_docstring_xrefs.py` is the gate L-044 said did not exist; and in `tests/` a dead xref is a TRIPWIRE for a false claim about what the gate proves

L-044 concluded that for an un-`automodule`'d docstring "the ONLY gate is
grep". That is now superseded by a real gate, committed this session:
**`tools/check_docstring_xrefs.py`** resolves every *fully-qualified*
Python-domain role by **importing** it, so render coverage is irrelevant.
`.venv/bin/python tools/check_docstring_xrefs.py tests --quiet` →
`DEAD TARGETS : 0` is a hard, cheap acceptance criterion; exit 1 gates CI.
It deliberately ships an EMPTY `ALLOWLIST` — never add to it. Two design
facts to respect: (i) it skips UNQUALIFIED refs (`:meth:`Quadrature.product``)
because Sphinx resolves those against module context and flagging them
manufactures false positives — so an unqualified dead ref still needs
grep; (ii) it separates "module exists but raises on import" (a TOOLING
problem) from "genuinely absent" (a dead ref) — opposite fixes.

**Why `tests/` is the sharpest surface.** It carried 495 fully-qualified
xrefs and **nothing had ever checked one of them**: Sphinx never reads
`tests/` at any severity, so `-W` and `-n` are both structurally blind
(L-044(a)). First run: **41 dead targets across 62 sites**.

**The load-bearing finding — a dead ref in a TEST docstring is a
tripwire, not a typo.** A test docstring states what the test PINS, so
the retirement that killed the ref usually also invalidated the
surrounding CLAIM. Rate measured here: 3 of 41 dead targets sat inside a
present-tense-FALSE claim, and one was a whole-file misdescription
(`test_unified_matvec_sphere.py` still advertised a six-step
unified-vs-legacy bit-identity chain; BOTH implementations had been
deleted and the surviving file holds two σ_t/zero sanity gates — the
accurate story was already written in a CLASS docstring three screens
down). Another (`test_native_matvec.py`) had a 7-item pin list whose
item 5 asserted the **inverse** of the live gate (docstring: "face
residual zero at non-outflow ordinates"; test name:
`test_outer_face_inflow_slots_carry_the_identity`), plus one retired and
one inverted item. RULE: on every dead ref in `tests/`, read the test
BODY, not just the sentence — then REPORT the false claim explicitly
rather than quietly repointing the link, because a wrong claim about
what a gate proves is worse than a dead link (Mode-11/Mode-12 adjacent:
the docs are the only place the claim is written down).

**The four-way adjudication that worked** (per site, never blanket):
REPOINT (symbol moved — the majority, 46/62 here) · PAST-TENSE LITERAL
(the sentence is history; flip the tense and demote the role to
double-backticks, since a role PROMISES a live link) · REWRITE (the
claim is present-tense-false) · DELETE (rare; here **zero** — every
sentence carried content). A useful sub-case: a not-yet-built module
(`orpheus.derivations.registry`, "gets promoted into … per the wave3
plan") is a LITERAL, and while there also verify the cited PLAN FILE
still exists — `.claude/plans/wave3/` did not.

**The brief's own successor map is a hypothesis.** It said
`orpheus.sn.angular_flux.AngularFlux` "now lives at
`orpheus/transport/fields/angular_flux.py`". `git log --diff-filter=D`
on the old path showed the opposite: the legacy class was a *different*
object (bulk + conflated boundary buffer + history) DELETED at
`d8843ba9`, replaced by the composite `TimedFullField` whose bulk is the
same-named L2 class. Same name, different object ⇒ the 5 sites split
into past-tense literals (history) and one REWRITE (a module docstring
naming the wrong return carrier while the gate asserts
`isinstance(state, TimedFullField)`). ALWAYS run the deletion-commit
read before trusting a "just moved" claim.

**Mechanics.** A module-path rename is a mechanical `str.replace` over
`tests/**` — but ORDER the mapping longest-first (`_quadrature_recipes`
before `_quadrature`) and first prove every occurrence is inside a
docstring/comment (a live import cannot contain a dead path, so
`grep -v` the role spellings and eyeball the remainder). 43 replacements
landed that way; the ~19 judgment sites were hand-edited.

**Proving doc-only.** The reviewer's check IS the author's check: parse
HEAD and worktree, strip docstrings, compare `ast.dump`. 42 files,
0 diffs. This also proves no `@pytest.mark.verifies/catches` moved —
those are V&V registry edges. Note what it does NOT cover: comments
(absent from the AST, and legitimately editable), and **f-string
assertion messages, which ARE code** — two stale `cells_view` mentions
had to be left in `_require(...)` messages and REPORTED instead (L-041).

**Free catches worth taking.** The tool's `DECIDABLE_ROOTS` excludes
`tests`, so intra-tree `:mod:`tests.…`` refs are ungated — a 20-line
ad-hoc resolver found 5 more dead ones (a partly-executed
`test_trajectory_resolvent_*` rename family). Raw path strings with LINE
NUMBERS (`orpheus/derivations/peierls_geometry.py:2906`, ~10 sites) are
the opposite call: repointing the directory without re-verifying the
line implies a verification you did not do — FLAG, don't half-fix.

How to apply: run the tool as the acceptance gate on any tests/ or
docstring sweep; treat each dead ref as a claim to re-verify against the
test body; adjudicate four-way; prove doc-only by AST; report every false
claim you find and never fix the underlying gate yourself.

---

## L-046 — An import-resolving xref gate OVER-reports: an annotation-only class attribute is live to Sphinx and dead to `getattr` — prove the anchor before degrading a ref to make a gate green

The `orpheus/` half of L-045's sweep: **30 dead targets / 37 sites**, all
residue of module MOVES the corpus-side edit had already fixed while the
docstrings were left behind (`8cda6b73` rewrote four theory pages for the
`TransportSolver` retirement; the four *modules* naming it kept their
refs for months).

**The finding that matters: 5 of the 30 were NOT dead.**
`resolve()` imports the longest importable prefix and `getattr`s the
remainder — so a class attribute that exists only as a **class-level
annotation** reports `missing` while being a perfectly live symbol:

| target | declaration | verdict |
|---|---|---|
| `Field.UNITS` | `UNITS: ClassVar[Unit]` + `#:` comment | **renders as a LIVE `<a href="#…Field.UNITS">`** |
| `DiscreteMeasure.nodes` | `nodes: np.ndarray` | dataclass field |
| `AngularSymmetry.continuous_isotropy` | `continuous_isotropy: SubgroupOfO3` | dataclass field |
| `AngularTraceSpace.omega_dot_n` | `omega_dot_n: NDArray = field(...)` | dataclass field |
| `WithinGroupSystem.loss` | `loss: "CoupledOperator"` | dataclass field |

The decisive evidence is a **rendered-anchor grep in a FRESH build**:
`grep -o 'id="orpheus.numerics.field.Field.UNITS"' <build>/api/numerics.html`
→ present, with three inbound `href="#…"`. So autodoc DOES emit
`py:attribute` for annotation-only members; the gate's probe cannot see
them. Turning that ref into a literal to make the gate green would
DELETE a working link — the exact inversion of the gate's purpose. Leave
it, report it with the anchor as proof, and hand over the resolver patch
(after the `getattr` `AttributeError`, accept
`attribute in getattr(obj, "__annotations__", {})`). **Never edit a gate
you were not asked to edit, and never mutilate a true ref to satisfy
one.**

**The mirror-image class — genuinely unresolvable, and worth fixing.**
`napoleon_use_ivar = True` (docs/conf.py) makes numpydoc
`Attributes`/`Parameters` entries render as `:ivar:`/`:param:` FIELDS,
which mint **no** `py:attribute` target. So an instance attribute
assigned in `__init__` (`self.bc = …`, `self.pole_angular_closure = …`,
`self.eigenvalue_method = …`) is unresolvable in EVERY build no matter
how well documented. Honest spelling: a live `:class:` role on the owner
+ the attribute as a literal — "the realized ``bc`` dict on
:class:`~…DiffusionMesh`". Three such sites here. Check
`napoleon_use_ivar` before assuming an `Attributes` section creates
targets.

**Adjudication mix (25 genuine).** 12 REPOINT · 4 past-tense literal ·
9 REWRITE. The REWRITE rate is much higher than a rename sweep would
suggest because a *move* leaves a true-but-relocated symbol, whereas a
**deletion** leaves a sentence whose PREMISE died: the paragraph that
justified the deleted thing inherits its wrongness. Three shapes recurred:
(i) a **self-contradicting file** — `geometry/boundary/__init__.py`
past-tensed the dissolved realizer registry at line 173 and still
present-tensed "three stub realizers self-register at import time" at
line 466; (ii) a **completed migration still written as future work** —
`reduced_operator.py` said consumers "will migrate in Wave G" while
`SNMesh.__init__` already calls its factories; (iii) a **docstring
contradicting its own body** — `BasisSpace.solve_critical` documented a
`d=None` fallback the code deletes with a `ValueError`.

**A retirement DEMOTED a gate, again (L-044's rule, live).** The
`reduced_operator` docstring credited
`tests/geometry/test_reduced_operator.py` with hash-equality "vs the
legacy SNMesh setup methods". Those methods are gone; the test now
compares `spherical_streaming(mesh, quad)` against `sn_mesh.reduced.*`
— the value that same factory produced, through the mesh constructor —
and the two remaining `SNMesh.face_areas`/`delta_A` legs are DEPRECATED
read-throughs to the same object. It pins the WIRING, not the math. The
fix is a `.. warning::` naming what survives (the SN curvilinear
snapshots + the τ producer-equivalence floor), not a deleted claim. Same
stale claim sits in `docs/theory/foundations/structured_geometry.rst`
§"Bit-identical contract" — whose very next section says the methods
"no longer exist".

**Also worth a look every time:** a `:mod:` naming an UNBUILT plan target
(`orpheus.transport.problems`) must be a literal (L-002/L-014); a type
name invented in prose and never minted (`FlatVectorLike` — grep says it
never existed in `orpheus/`) is a dead ref with no successor, so name the
actual construct (here the duck-typed *ravellable* `to_flat` /
`from_flat` pair, deliberately unnamed so `numerics` need not import
`transport`); and a **code block whose first import raises `ImportError`**
(`docs/theory/references/trajectory_resolvent.rst:3816` still imports
`GeometrySpec`) is the hardest MUST-FIX of the tense discriminator.

How to apply: run the tool, then adjudicate EVERY hit against the live
tree before touching it — a `getattr`-based gate over-reports on
annotation-only members and under-reports on unqualified refs (L-045).
Prove a contested hit with a rendered-anchor grep in a FRESH build.
Fix the mirror class (instance attributes) properly. Expect the deletion
residue to be a stale PARAGRAPH, not a stale token.

---

## L-047 — The `docs/` half of the xref sweep: a dead ref in an `api/` page is a retired API SURFACE (rewrite the section), and `:noindex:` makes a whole page's roles plain text so "would `-n` catch it?" is moot

Closing the trilogy after L-045 (`tests/`, 41 dead / 62 sites) and
L-046 (`orpheus/`, 30 / 37). `docs/`: **20 dead targets across 24
sites in 15 pages → 0**, `-E -W` EXIT 0 with the diagnostic set
unchanged from a freshly-measured **0**-warning baseline.

**The tree-specific shape: `api/` prose dies in BLOCKS, not tokens.**
7 of the 24 sites were one page's one section — `docs/api/geometry.rst`
"Factories", listing `Zone` / `mesh1d_from_zones` / `pwr_pin_equivalent`
/ `pwr_slab_half_cell` / `homogeneous_1d` / `slab_fuel_moderator`, all
retired in one Phase-F commit (`81b083be`). Six repoints would have been
six lies: the successors are not renames but a **re-layering** (geometry
`StructuredGeometry` + `Region`, then mesh `Mesh1D.from_geometry` +
`RegionMesh`), so a section whose thesis is "the factory layer is the
recommended construction path" is stale as a THESIS. The tell that
distinguishes this from a rename sweep: the retired names' own module
docstring already carried the successor map (`factories.py` opens with
"Phase F retired … The 1-D path is now …"). **Read the surviving
module's docstring before planning repoints** — a well-retired module
tells you whether you owe N edits or one rewrite. Ratio here: `docs/`
came out **7 REPOINT · 5 mirror-class · 4 past-tense literal · 8
REWRITE**, i.e. a third rewrites, matching L-046's finding that
deletions leave stale PARAGRAPHS.

**`:noindex:` suppresses the WHOLE page's anchors — measured.**
`docs/_build/html/api/method_of_characteristics.html` contains **zero**
`id="orpheus.*"` anchors; so does `api/discrete_ordinates.html`. Every
`automodule` on those pages carries `:noindex:`, which renders docstrings
but mints no `py:` targets — so *every* py-domain role there is plain
text whether or not the symbol exists, and a live `href` to
`#orpheus.sn.solver.SNSolver` sits in the tree pointing at an anchor that
was never created. Consequence for L-002/L-044's "would `-n` have caught
it?": on this corpus the question is doubly moot — nitpicky can only
nitpick what it RENDERS, and here rendering doesn't mint targets either.
The import-checker is the only gate. (Corollary when rewriting such a
page: adding an `automodule` is still worth it for the *docstrings*, but
do not expect it to make roles link.)

**The `napoleon_use_ivar` mirror class is not a curiosity — it was 5 of
24 sites** (`SNMesh.scheme`, `SNMesh.pole_angular_closure`,
`SNSolver.inner_solver`, `KEigenvalue.eigenvalue_method`, plus
`MOCQuadrature.n_azi_2` which never existed at all). An
`__init__`-assigned attribute is unresolvable in EVERY build; adding
autodoc coverage will NOT revive it. Standard rewrite, applied five
times: a live `:class:` on the owner + the attribute as a literal, phrased
so the sentence says where the value comes from — "the ``scheme``
attribute that :class:`SNMesh` realizes in its constructor", "the
``eigenvalue_method`` constructor selector on :class:`KEigenvalue`".
Contrast the OVER-report class (L-046): the checker now falls back to
`__annotations__` across the MRO, so dataclass fields and `ClassVar`s
resolve correctly and need no defensive verification.

**Three dead refs sat on a claim that a rename had INVERTED, not just
moved.** (a) `operator_inverse_family.rst` published a 3-line code block
for `_seeded_inverse` reading
`return cast(SupportsSeededApply, cast(SupportsInverse, A).inverse())`
with prose crediting "the CALLER's `A.is_invertible` guard". The live
`seeded_inverse` (public since #276 A4, `cd000c2e`) has **no cast at
all** — two `TypeGuard` bridges — and runs its **own** `invertible()`
guard raising `NotInvertible`. The published sentence named the wrong
mechanism AND the wrong guard owner; a repoint alone would have left both
falsehoods with a working link. (b) `api/method_of_characteristics.rst`
claimed "azimuthal angles are adjusted slightly from an even distribution
to satisfy the cyclic condition" — `MOCQuadrature.create` is a plain
midpoint-even `linspace`; cyclicity is reached from the **ray-spacing**
side (`effective_ts = (t_max−t_min)/n_rays` per angle). (c) A
`.. code-block:: python` introduced by "A user constructs one with:"
opened on `from orpheus.derivations.common.geometry_spec import
GeometrySpec` (module deleted at `81b083be`) *and* used `np` with no
`import numpy`. **RUN every doc code block that a present-tense sentence
promises works** — this one now does, verified end-to-end
(`k_eff = 1.0000000000000002`), as does the new geometry construction
block. A dead import is the loudest possible dead ref and no build sees it.

**Scope discipline against an owner issue.** Two sites sat on a page an
OPEN issue (#286) already owns. Correct move was neither "defer, it's
theirs" nor "fix everything": fix the dead refs, fix the **measured**
adjacent falsehoods (`ReducedStreamingOperator` has no `tau_mm` field;
2 of 8 claimed deprecated properties survive on `SNMesh`), leave the
issue's genuine mechanism-rewrite item alone, then comment with a
measurement table and the residue's **corrected path** (the issue named
`docs/theory/discrete_ordinates.rst`, which no longer exists — the
section is now `docs/theory/methods/sn/index.rst:472`). An owner issue
whose cited paths have rotted is worth less than the five minutes it
costs to re-point them.

**A retirement DEMOTED a gate, again — and a sibling agent proved it
independently.** `structured_geometry.rst` credited
`tests/geometry/test_reduced_operator.py` with bit-identity "vs the
legacy SNMesh setup methods"; those methods are gone and `SNMesh.__init__`
now calls the factories itself, so the surviving legs compare a fresh
factory call against the value that same factory produced. Fix: past-tense
the history, `.. warning::` the demotion, name the gates that DO carry the
math (`sphere_*`/`cyl_*` regression snapshots,
`test_tau_producer_equivalence.py`, `test_alpha_closed_form.py`). A
concurrent `tests/geometry/` pass was simultaneously renaming those legs
to `test_*_is_the_factory_value` with a docstring reading "`array_equal(x,
x)` for any face-area math whatsoever" — same verdict, reached
separately. **When you suspect a gate was demoted, say so in the doc; the
convergence is evidence, and the doc is the only place the claim is
written down.**

**THE SESSION'S SHARPEST CORRECTION — I named a gate that was BLIND to
the claim I credited it with.** Having correctly demoted the
bit-identity gate, I wrote "the mathematical content is pinned
elsewhere: … `test_tau_producer_equivalence.py` + `test_alpha_closed_form.py`
for the closure coefficients themselves." The τ half was FALSE, and it
was falsified by measurement, not argument: under fully-garbaged
`spherical_streaming`/`cylindrical_streaming` factories that file passes
**5 tests in 0.03 s**. Cause — the same #236 Step C move I had just
documented two screens earlier in my OWN τ-ownership note: τ left the
reduced operator for the angular closure, so the gate compares
`morel_montry_tau_per_level` against `morel_montry_weights`, both
derived from `(μ, w)` alone. It pins a quantity `reduced_operator.py`
no longer produces. **I had written the premise and still drew the
wrong conclusion from it**, because "τ is a closure coefficient" and
"the reduced operator carries the closure coefficients" were both true
sentences on the page and I composed them without checking the
referent.

The rule this earns, and it is the doc-prose analogue of
`vv-principles`' "a `catches` marker is a COVERAGE CLAIM, not a topic
tag": **a doc sentence of the form "gates X, Y pin claim C" IS a
coverage claim, and it must be justified the same way — by a mutation
that reddens X and Y, never by topical adjacency.** Corollaries:

* **Replacing a demoted gate is the moment of maximum risk.** Having
  just proven gate A blind, the reflex is to reach for the
  nearest-sounding sibling. That reflex is exactly what produced the
  error; the sibling inherits neither A's scope nor the claim's.
* **The correct citation is PER FIELD, not per topic.** The measured
  replacement is five different files, one per array — `delta_A` →
  the closed-form `test_delta_A_magnitude` (its **sole** catcher; the
  snapshots are legitimately blind because `delta_A` has no production
  consumer); `alpha_half` → `test_per_ordinate_flat_flux_consistency[SPHERICAL]`
  (`catches` ERR-006/007); `alpha_per_level` → `test_alpha_closed_form.py`,
  **cylindrical-α only** (every fixture is `CoordSystem.CYLINDRICAL`);
  `redist_dAw` → `test_streaming_equilibrium_curvilinear.py`'s L0
  `φ = Q/(Σ_t(1−c))` identity — and NOT the flat-flux identity, which
  recomputes `ΔA/w` instead of reading the production array;
  `face_areas` → `tests/geometry/test_geometry.py` on the producer
  `compute_areas_1d`. A single "these gates cover it" sentence cannot
  be true at that granularity.
* **The SAME gate cited for TWO different claims can be right once and
  wrong once — narrow the correction, do not sweep it.** The τ gate
  appears twice on this page: at the `morel-montry-clamp` vv-status
  rationale (**correct** — it does pin τ) and in the demotion warning
  (**false** — it does not pin the reduced-operator arrays). The
  coordinator had to send a second message narrowing the first,
  because a blanket "that gate is wrong" would have destroyed a true
  citation. When told "citation of X is false", ask *false for WHICH
  claim* and grep every occurrence before editing any.
* **When a sibling agent corrects a shared claim in code you may not
  edit, MIRROR its wording rather than re-deriving prose.** The
  corrected `reduced_operator.py` docstring already carried the
  measurement and the per-field table; re-verifying each catcher
  against the live tree (5 for 5) and mirroring cost minutes and
  guarantees code and corpus say the same thing.

**Free catches en route.** A `scipy` role can die by UPSTREAM removal —
`scipy.special.sph_harm` was deprecated in 1.15 and removed in 1.17 (the
tree runs 1.17.1); successor `sph_harm_y` has a **swapped `(n, m)`**
argument order, which belongs in the fixed sentence. A `:mod:` naming a
planned promotion (`orpheus.derivations.common.chord_oracle`) needs the
literal AND a check of the plan file it cites
(`.claude/plans/trajectory_resolvent_hindsight_refactor.md` — gone; the
"scheduled" framing had to become "open opportunity, trigger = a second
consumer"). And a rename ripples past the roles: `_select_si_resolvent →
_select_si_splitting` had 1 dead role plus **3 present-tense literal
mentions** on two other pages, invisible to the checker (literals aren't
roles) — after fixing the flagged role, grep the OLD NAME tree-wide and
adjudicate every literal by tense.

**Sanity note on the by-product.** The `-E` build regenerated
`docs/theory/verification/matrix.rst` (+2 foundation tests,
`test_docstring_xrefs`), because the committed matrix predated that test
file landing in HEAD. Legitimate; report it, never revert it (L-008).

How to apply: run the checker as the gate on `docs/` too; before planning
repoints read the surviving module's docstring to learn whether you owe N
edits or one section rewrite; expect ~⅓ REWRITE and ~⅕ mirror-class; RUN
every promised code block; grep the old name for literals after fixing the
roles; and when a page has an owner issue, fix + measure + comment with
corrected paths rather than defer or annex.


## L-048 — An equation in a doc has TYPES and a SCOPE, and NO gate checks either: a type-incoherent identity and a one-instance proof stated for a whole class both build clean forever

**The task.** Correct two defects in the boundary-condition *algebra of
record* (`docs/theory/foundations/boundary_conditions.rst`, the
`bc-factor-roles` section) as step 0 of campaign step G6.3. Both defects
were in *published mathematics* — not in a cross-reference, a symbol name,
or a status claim — so every gate this agent owns was green on them:
`-E -W` at 0 warnings, `check_docstring_xrefs` at DEAD TARGETS 0, and the
V&V matrix regenerating without complaint, for as long as the claims had
shipped.

### Defect class A — the identity does not TYPE-CHECK

The rank-one theorem concluded, verbatim, **`R ∘ G = R`**. With
`G : Γ₊ → Γ₋` and the classifying `R : Γ₋ → Γ₋`, the left side is
`Γ₊ → Γ₋` and the right side is `Γ₋ → Γ₋`: they cannot be equal as
operators. The step had silently identified `Gᵀv` with `v` by treating
`v = |Ω·n|` as a *function* without tracking which half-trace it is
restricted to — **harmless until campaign phase B3.2 narrowed the SN law
onto `γ±` and made the two halves genuinely distinct spaces**, a type
abuse from that commit onward.

Note the shape: nobody introduced an error. A *code* carve (B3.2)
retroactively falsified a *math* sentence three chapters away, and the
falsification is a **type mismatch inside an equation** — a defect class
with no gate at all. The correct statement (`R ∘ G = u ⊗ (Gᵀv)`, and
`Gᵀv = v` *as a function*, so the composite is `G`-INDEPENDENT rather than
equal to `R`) preserves the theorem's whole content and every downstream
conclusion, including "the B3 correction leaves the composite unchanged".

⇒ **Read an equation's DOMAINS AND CODOMAINS, not just its symbols.** For
every displayed identity on a page you touch, name the space each side
lives in and check they agree. When a page's spaces have recently been
NARROWED (a half-trace split, a domain restriction, a role-keyed retype),
that narrowing is a licence to re-type-check every identity mentioning the
affected spaces — the old ones were written when the spaces coincided.

Bonus catch from doing this properly: the slogan's real hypothesis is that
`v` is `G`-INVARIANT, not merely that `R` is rank-one (a rank-one `R` with
`v = δ_{Ω₀}` makes `G` fully observable). Keep the memorable slogan — it is
cited by `:ref:` from 5 code/test sites — and add ONE sentence naming the
hypothesis that actually does the work.

### Defect class B — proven for ONE instance, stated for the CLASS

A paragraph headed "**The crossing is geometric**" argued — correctly —
that the specular mirror is the unique ambient isometry fixing the face,
exchanges the hemispheres, and preserves `|Ω·n|`. It then closed *"which
is why `G` and not `R` carries it"* — for **every** law. False for a law
with no isometry: a wall is not a quotient (the page's own sufficient test
says so), so nothing provides the crossing geometrically and the
**physics** does it, by integrating the outflow and re-emitting an inflow.

⇒ **On any "which is why X" generalisation, ask: for WHICH instance is the
argument given, and is the conclusion stated for the CLASS?** The tell is a
paragraph that opens on one concrete object (a mirror, a slab, a 1-group
case) and closes on a universal. Scope the existing argument to the case it
proves — keep it verbatim in substance — and *add* the missing case. Do not
rewrite the proof; it was never wrong, only over-quantified.

Corollary that made the fix stronger: the honest law ("whichever factor is
non-trivial carries the crossing") has **boundary cases worth writing out**
rather than hiding. Rank-0 laws cross vacuously; a bare scalar response on
an ANGULAR trace is non-trivial in *magnitude* and trivial in *angular
structure*, so NEITHER factor crosses — which is exactly the
already-documented realizer REFUSAL. Boundary cases that turn out to be
shipped refusals are the best possible evidence that the reformulated law
is the right one.

### The framing that DISSOLVED both — taxonomy ≠ computational factorization

Both defects had one root: `R ∘ G` is a **TAXONOMY** (does this law's
content come from geometry or from physics? — decided by multiplicativity +
the quotient test) and the page presented it as also the **computational
factorization**. As a classification `G : Γ₊→Γ₋`, `R : Γ₋→Γ₋` is coherent;
as a realized typing it is false, and `[M]` **no realized response in the
tree is an endomorphism of `Γ₋`** — the Lambertian's realization types
itself `Γ₊ → Γ₋` in its own first line.

⇒ **When a doc carries a factored / classified form of an object, state
explicitly whether it is a classification or a recipe** — and check the
declaration tier against the REALIZATION before writing either. Two
successive campaign designs were built on this conflation, each refuted by
one read of the realizing class. A dedicated short section ("the taxonomy
and the factorization are different questions") is the right shape: it
gives the distinction a citable anchor so the next design cannot re-make
the assumption silently.

### What I got RIGHT and should repeat

- **Reproduced every number before publishing it.** Re-ran the design
  probe (`R* = G₊⁻¹RᵀG₋`, `G_S` over 11 orders): `0.0` / `1.110e-16` /
  `1.110e-16` / `7.628e-01` degenerate, weighted adjoint law at exactly
  `0.0`. Verified the claimed one-line transpose
  `Rᵀ(φ) = (cos_w/norm)·Σφ` against the DENSE transpose of the operator's
  own `apply` — `max err 0.0`. Verified `is_adjointable is False`.
  Re-measured the plan's `2/2, 4/4, 49/49` `|Γ₊| = |Γ₋|` claim and
  WIDENED it (6 quadratures × every face, incl. `level_symmetric(6)`
  24/24) instead of transcribing the sample.
- **Refused to cite an ephemeral path.** The probe lived under
  `$CLAUDE_JOB_DIR/tmp/`, which no future reader can open, and
  `scratch/g6_design_measurements.md` did NOT record it. Instead I
  DESCRIBED the construction (shapes, metrics, comparison) so the table is
  reproducible from the page alone. **A path that will not exist is a
  stale raw path the moment it is written** — describe, don't cite.
- **Reported rather than edited a concurrently-edited code file.**
  `orpheus/geometry/boundary/_factors.py` was being rewritten by the main
  agent DURING my session (its mtime moved mid-task; it had already fixed
  defect B and the taxonomy framing there, but NOT the `R ∘ G = R`
  identity). Two residual sites in it were REPORTED, not touched.
- **Kept the heavily-referenced anchor.** `bc-factor-roles` has 8 inbound
  `:ref:`s across docs + code + tests; the fix went INSIDE it and the two
  new ideas got their own new anchors, so zero referrers moved.

### Two mechanical facts worth keeping

- **`⭐` and `⛔` have ZERO occurrences anywhere in `docs/`.** They are the
  agent-memory / plan / code-docstring vocabulary. `⚠` (9 sites) and `✓`
  (43) ARE corpus vocabulary. I drafted with `⭐`/`⛔` carried over from
  the plan and had to strip them — **grep a glyph in `docs/` before
  importing a marker from a plan or a docstring.**
- **A Python `SyntaxWarning` appears in the Sphinx log and is MISSED by a
  case-sensitive `grep 'WARNING:|ERROR:|CRITICAL:'`.** A non-raw docstring
  containing `\Gamma` produced `SyntaxWarning: "\G" is an invalid escape
  sequence` mid-build; it does not bump the exit code either. Add
  `SyntaxWarning` to the build-log grep — and before reporting one in a
  file the main agent is editing, re-`py_compile` the LIVE file (mine was
  already fixed a minute later).

### vv-status for an equation whose CODE does not exist yet

The new adjoint identity got `.. vv-status: bc-response-factored-adjoint
documented` — correct, because the FACTORED spelling is not built (the
Lambertian ships as ONE operator with `is_adjointable=False`; the
factorization is a later campaign step), so no `verifies` marker has a
function to point at. The rationale comment names the **precondition**
gates that DO exist (`test_the_half_trace_metric_is_strictly_positive`
pins the non-degeneracy the identity requires;
`test_the_metric_is_not_euclidean` pins that the metric is load-bearing at
all) and ends with *"when the factorization lands, WIRE a test to this
label and REMOVE this sentinel"* — so the sentinel carries its own exit
condition. Documented-only count moved 524 → 525 on regen, as expected.

⚠ **UPDATE 2026-08-04 — that exit condition FIRED six days later** (G6.3
step 3b) and the sentinel's rationale became present-tense-false in the
exact words quoted above. See **L-049** for what to do when you are not
the owner of the generated artefact the un-sentineling would move: keep
the DIRECTIVE, rewrite the RATIONALE to state that the precondition
expired + why the directive is still there + the gate that now exists,
and flag it. A sentinel that carries its own exit condition still needs
somebody to notice the condition fired — nothing in the build does.

⚠ **The `-E` regen also absorbed a row from the main agent's uncommitted
work** (`numerics/test_angular_face_trace_space`, +56 foundation tests,
8478 → 8534 total, shifting every share percentage). Legitimate
by-product; report it, never revert it (L-008).

How to apply: type-check every displayed identity against the spaces its
sides live in, especially after a narrowing carve; scope every "which is
why" to the instance actually proven and add the missing case, writing out
the boundary cases; say whether a factored form is a classification or a
recipe, and check the declaration against the realization FIRST; reproduce
and WIDEN every measured number; describe a probe instead of citing an
ephemeral path; grep a glyph in `docs/` before importing it; add
`SyntaxWarning` to the build-log grep.

---

## L-049 — A class retirement's docs blast radius is THREE tense classes, not two; and a composite `[M]` cannot certify its factors

**Task (2026-08-04):** repoint the dead xrefs left by deleting the welded
`AngularAverageOperator` (`orpheus/sn/boundary/angular.py`), replaced by
`IsotropicEmissionOperator @ PartialCurrentOperator`. 19 mentions in
`docs/` source: 12 role-bearing (the checker's list) + 7 literals the
checker cannot see. Bounded, "do not restructure".

### The THIRD tense class — a falsified PREDICTION

The brief's discriminator was the standard two: present-tense claim ⟹
REPOINT; past-tense history ⟹ DE-ROLE to a literal. Both applied and both
were right. But 4 of the 12 role sites, and the richest prose on the page,
were neither — they were **future-tense predictions written while the
replacement was still a plan**, and the landing falsified them:

* *"The type that will host it exists —* `ScalarTraceSpace` *… the
  per-face accessor and the factored `B ∘ C` spelling are* **not built
  yet**." Both halves shipped — but the host is NOT `ScalarTraceSpace`. It
  is a new `Γ`-ladder tier (`AngularTraceSpace.current_space`, a unit
  metric, per-face) and the shipped `current_space` docstring goes out of
  its way to say `ScalarTraceSpace` is a *different object* (the `(J⁺,J⁻)`
  pair for the whole boundary under the face-AREA metric — hosting one in
  the other double-counts the area weight). "Not built yet" → "built"
  would have shipped a page asserting the wrong type.
* *"that is phase **B5**, which is what makes its adjoint structurally
  available"* — landed at **G6.3 step 3b**, and by *factoring* rather than
  by the predicted `u ⊗ v` typing, which **dissolved** the transpose
  ambiguity instead of resolving it.

⭐ **A deferral contract names THREE things — the MECHANISM, the
HOST/TYPE, and the PHASE — and a landing can falsify any subset
independently.** Check each separately against the shipped code; do not
let "the seam closed" license a blanket tense-flip. (Sharpens lessons §4's
"verify the SHAPE that shipped": the shape is three fields, not one.) The
honest repair preserves the prediction and tombstones it — I kept the
sentence and added a `.. note::` naming what was predicted, that it did
not hold, and *why the shipped host is a different object* — which is
strictly more informative than the correct sentence alone would have been.

### A composite `[M]` measurement cannot certify its own FACTORIZATION

The page carried, correctly, `Rᵀ(φ) = (cos w / norm)·Σφ` verified `[M]`
bit-exactly. It built that from `Cᵀ(s) = cos w · s / norm` and
`Bᵀ(φ) = Σφ` — and the SHIPPED split is the other way round:
`Cᵀ(s) = cos_w ⊗ s` (no `/norm`) and `Bᵀ(φ) = Σφ / norm`. Composite
identical, per-factor formulas both wrong. Re-measured live: every one of
the four identities reads `0.0` on `product(2,4)`/`xmax`, so **no
measurement on the page could ever have caught it.**

⭐ The structural reason matters and belongs in the prose: the
normalisation lives in `B` because `C` produces a **current** and `B` must
produce an **intensity**, so the division is the unit-changing step —
which is what leaves `S(f)` carrying an honest `J⁺` and lets an albedo
enter as the pure scalar law `J⁻ = α J⁺`. A doc that gets the split
backwards silently deletes that argument. **When a doc factors an operator,
verify EACH factor's formula against its own live `apply_transpose`, not
the composite** — and re-verify the design-probe description too (this
page's probe also built `C` with the `/norm`; one clause fixed it).

### The vv-status sentinel whose exit condition fired, when you don't own the artefact

`.. vv-status: bc-response-factored-adjoint documented` (minted at L-048)
was correct while no production function realized the identity. The
factorization landed and the label now has a `verifies` marker on the
SHIPPED chain — so the directive is removable. But removing it
re-categorises a **generated** artefact (`theory/verification/matrix.rst`)
in a session whose remit is dead-xref repair, with the code owner mid-carve.

⭐ Resolution: **keep the DIRECTIVE, rewrite the RATIONALE.** The comment
now opens `⚠ THE SENTINEL'S PRECONDITION EXPIRED AT G6.3 step 3b AND THE
DIRECTIVE BELOW IS NOW REMOVABLE — left in place only because
un-sentineling re-categorises a GENERATED artefact and is owed the
regeneration`, quotes the superseded rationale verbatim as history, and
names the exact gate (`tests/sn/operators/test_lambertian_chain.py::
TestReciprocityAgainstTheMirrorFace::test_H_is_pointwise_the_mirror_face_kernel`).
Zero false text, zero silent V&V-category change, next session has the
whole decision. Contrast the two failure modes it avoids: flipping the
category unasked (invisible to `-W`, changes a generated table) vs.
leaving the quoted-false rationale (a future reader re-derives a dead
precondition).

### Mechanics worth keeping

* **`git status` at task start named the branch, not the one in the
  session snapshot** (`refactor/operator-strategy-layers`, not `main`),
  and by task end the main agent had COMMITTED the code side underneath
  me (`b4f0f5c9`). Re-read git before every claim about tree state.
* **The baseline `-E -W` warning was 1 and it was TRANSIENT** —
  `verification matrix regeneration failed: pytest collection failed
  (exit 2)`, caused by the main agent's half-saved test edits. `pytest
  --collect-only -q` succeeded moments later. Before attributing a
  baseline warning to the corpus, re-run the underlying tool directly.
  Final build: 0 warnings, EXIT 0.
* **`orpheus.sn.boundary.angular` is not `automodule`'d anywhere**, so
  every role on the BC page — the dead ones I removed AND the live ones I
  added — renders as plain text with no `href`. Verified in built HTML
  (`xref py py-class` spans, zero `<a>`). Matching the page convention is
  correct; adding an `automodule` for the two leaves I touched would be
  half-surfacing. The refs are still right and become links the day the
  package is surfaced.
* **A verbatim historical ERROR MESSAGE quoted in past tense
  (`` ``AngularAverageOperator.apply: psi.shape[0] = |Γ₊|, expected N`` ``)
  is correct history and already a literal** — leave it entirely. That was
  1 of the 19 sites and the only one needing no action.
* **Inline literals wrapping across two source lines render as ONE
  `<code>`** — the pre-existing table cells already did this, so
  ``` ``(IsotropicEmissionOperator(...) @ PartialCurrentOperator(...))\n
  & IdentityOperator()`` ``` is safe; `-W` catches the `:widths:`/column
  mismatch if you get a row wrong.
* **Measure the operator tree, don't infer it.** The two "MEASURED" code
  comments claim a realized `repr` shape; I walked the live tree by
  `__slots__`/`__dict__` (`realize_recursively` → `OperatorSum` →
  `ScaledOperator` → `TensorProductOperator` → `OperatorProduct(B, C)` →
  `IdentityOperator`) before writing it, and it matched
  `orpheus/geometry/boundary/white.py`'s own spelling exactly. Use the
  production module's spelling when it has one — that is the SSOT.

**Residue flagged, not fixed (owner = main agent):** 7 present-tense
`orpheus/` mentions the import-based checker cannot see (unqualified role
at `sn/boundary/angular.py:271`; "the angular primitives the realizer
**consumes**: `AngularAverageOperator`" at `sn/boundary/__init__.py:10`;
the realization-map line at `sn/boundary/realizer.py:50`; the live
`WhiteBoundary`-arm comment at `realizer.py:774`; two at
`geometry/boundary/__init__.py:241,349`) plus a claim INVERSION at
`geometry/boundary/_factors.py:1050` — `SpecularReemission.is_adjointable`
still reads *"TRUE, **unlike the Lambertian's**"* three lines after the
Lambertian's flipped to `True`.

How to apply: split a retirement's doc sites into THREE tense classes
(present-false / past-history / falsified-prediction) and check a
prediction's mechanism, host and phase separately; verify each FACTOR of a
factored operator against its own live method, never via the composite's
measurement; when un-sentineling would move a generated artefact you don't
own, keep the directive and rewrite the rationale; re-run the underlying
tool before crediting a baseline build warning.

---

## L-050 — A brief's "measured, post-carve" number can be a PRE-carve number in disguise; re-measure BOTH sides of a carve, and never let a 12-dp printout stand in for bit-exactness

The brief told me the post-carve two-channel equivalence was "to solver
tolerance, NOT bit-identically: `|φ_D0 − φ_C|_inf = 1.998e-13`,
`array_equal = False`", and asked me to document that as the current
state with a `⚠`. It was a real measurement — of the **pre**-carve tree,
between a delivery through the operator block and a delivery through the
source channel (two structurally different computations reaching one
fixed point). Post-carve the two channels are the SAME float program, so
the difference collapses to **exactly `0.0`** with `array_equal = True`
on every fixture and both inner solvers. Documenting the brief's framing
would have published the inverse of the carve's most important
consequence — and would have justified an `rtol` gate that is
structurally BLIND to the defect the carve removed (`2.9e-14` relative
sails through `rtol = 10 × inner_tol`).

Second, worse trap in the same brief: it asserted the converged inflow
trace is exact — "measured `5.000000000000` / `2.500000000000` /
`0.000000000000` at 12 dp on both fixtures" — and specified a keystone
gate as one `assert_array_equal` parameterized over BOTH inner solvers.
Measured: exact on source iteration (the sweep *writes* the seed, so `q`
arrives as a copy — `0.0` deviation, every fixture, every tolerance) and
**NOT** exact on Krylov (GMRES *solves* for the trace rows, so the
reading carries the iteration residual: 1–23 ULP at `inner_tol = 1e-13`,
and **27 580 ULP** at `1e-10`). Twelve decimal places of `2.500000000000`
cannot resolve `8e-15` at 2.5 — the printout that "proved" exactness was
blind to the effect by three orders of magnitude, and the plan never
caught it because the pre-carve Krylov leg *raised* instead of producing
a reading. The specified gate would be red on Krylov for a reason with
nothing to do with the claim.

Third: my OWN verification probe hid a failure the same way. A widened
bit-identity check used a bare `assert` inside a script I ran under
`python -O` — vv Mode 8, in my own instrument. It reported clean; the
assertion never executed.

How to apply: (1) when a brief hands you a "measured" number about a
change, ask WHICH SIDE of the change it was measured on — a pinned
worktree at the pre-carve commit (`git worktree add <dir> <hash>`) makes
both sides cheap, and re-measuring both turns one asserted number into a
before/after table that is the strongest content on the page. (2) A
venv's setuptools *editable* install hooks `sys.meta_path`, which
OUTRANKS `sys.path` — `PYTHONPATH=<worktree>` silently loads the MAIN
tree instead; strip the editable finder before importing, and PRINT
`orpheus.__file__` as the proof (my first pre-carve run measured the live
tree and looked plausible). (3) Never accept a fixed-decimal printout as
evidence of bit-exactness — assert `x == v` or print `float(x).hex()`.
(4) An exactness claim that holds on one inner solver and not another is
a per-leg gate, not a weakened gate: say `assert_array_equal` on the
exact leg and `rtol = SAFETY × inner_tol` on the iterative leg, and say
explicitly "do not relax the exact leg to match". (5) Run your own
probes WITHOUT `-O`, or raise instead of asserting.

---

## L-051 — a "measured, already-done" brief: reproduce it, and expect the
## reproduction to move the SCOPE (#341 "regular splitting" corpus sweep)

**Context.** Brief: strike the false term *regular splitting* at "seven
live doc sites"; the code sites were "already done — do NOT redo"; the
gate was `-W` EXIT 0. Delivered: 9 doc sites + 2 missed code sites + a
blocking build ERROR + a second, independent false claim.

**(a) A multi-word term takes an INFIX — grep each token, not the pair.**
The brief's grep was `"regular splitting|regular-splitting"`. The corpus
also spelled it **`regular matrix splitting`**, so two live present-tense
sites were invisible: `cartesian_multid.rst:3840` ("exactly a **regular
matrix splitting**") and `history.rst:836`. One `grep -rn "regular"` over
`docs/theory` (37 hits, ~30 seconds to triage) found both. This is the
cheap mechanic under theme 4's "grep the CONCEPT": for an
`<adjective> <noun>` term, grep the ADJECTIVE alone and triage, because
the noun phrase routinely grows a word in the middle.

**(b) "Already done in code" ≠ "gating green" — measure the `-E` baseline
before believing the brief's premise.** The `-E` baseline was **EXIT 0,
1 diagnostic: `ERROR: Malformed table`** in
`orpheus/sn/solver.py:docstring of solve_sn_fixed_source:86` — the ⚠ rate
table the SAME upstream code pass had just added, with every data row's
numerals straddling the `===`-defined column gaps (offsets 44 and 57).
So the brief's own acceptance gate (`-W` EXIT 0) could not have passed on
the tree it handed me. Fixing it was blocking AND in scope (same issue,
doc-only). Diagnose a simple table by reconstructing the column spans
from the separator with `re.finditer(r'=+', sep)` and flagging non-space
characters inside the gaps — instant, and it names the offending offsets.

**(c) ⭐⭐ A ratio is a ratio OF AN OBSERVABLE — ask which one before
citing it.** The brief handed me a spectral result and an investigation
memo whose §6 argued a conclusion from `n_GS/n_J` values. Those were
**ρ-DERIVED** (`ln ρ_J / ln ρ_GS` from an ARPACK eigen-solve of `M⁻¹N`);
every table already published in the corpus reports **SWEEP COUNTS** from
a real solve. `[M]` I re-measured five memo rows as sweep counts with a
control first: the control reproduced the published `1631 / 838 = 1.946`
**exactly**, and then **4 of 5 rows disagreed in SIGN** (memo `0.576`
"G-S wins" vs measured `1499/599 = 2.503` "G-S loses"). Both instruments
are individually sound — the memo validates ρ against a fitted residual
decay to 4 decimals. They simply measure different things: when
`ρ_GS ≈ ρ_J` to a fraction of a percent, the asymptotic ratio is wildly
sensitive while the sweep count is dominated by the transient, the
residual constant, and (per the memo's own §1.3) a frozen null-space
component the stopping test cannot see. **Rule: publish only the
observable you measured, name it in the caption, and never let a
rate-ratio and a count-ratio share a column heading.** The memo's
CONCLUSION survived (I established it independently on the other side);
its specific rows did not.

**(d) A technical term with a theorem attached is the ONLY carrier —
absence of a paraphrase is the finding, not reassurance.** I grepped
`Varga`, `comparison theorem`, `Stein-Rosenberg`, `monotone`, `M-matrix`,
`no slower than`, `never slower`, `at least as fast`, `ρ_GS`, `ρ_J` and
every `splitting` in `docs/theory`. The corpus **never once** wrote the
guarantee in prose. That is exactly why the word survived nine sites: a
reader who knows Varga supplies `ρ_GS ≤ ρ_J` silently, and a reader who
does not sees a decoration. So for a named theorem-bearing term, every
occurrence is load-bearing and none can be dismissed as decoration —
and the correction owes the corpus ONE place where the theorem is stated
and its failing hypothesis named, or the next author re-adds the word.

**(e) One home for the reason, `:ref:` from the rest.** The brief said
"point at the canonical code warning rather than restate the derivation".
Resolved by Cardinal Rule 3: a module docstring is a construction-site
note, a theory page is the brain. New H3
`sn-boundary-gs-not-regular` in `cartesian_multid.rst` (the page that
owns the boundary-G-S schedule) carries the whole derivation; the other
8 doc sites and 6 code sites `:ref:` it. Verified in built HTML: 5 of 5
cross-doc referrers resolve to real `href`s (a cross-doc `:ref:` renders
plain text with NO warning — the build cannot tell you this).

**(f) Importing algebra from a memo/code into a theory page imports its
SYMBOL COLLISIONS.** `A_a` (face area) collided with `A = L+C−S−B` (the
loss operator the whole section is about); `Σ` (transmission matrix)
collided with `Σ_t`. Resolution that satisfies the ratified
internal-consistency directive: **keep the code's spelling** and pay for
it with an explicit `.. note::` naming both overloads and their
disambiguators ("`A_a` always carries its axis subscript"; "`Σ` never
carries a `t`/`s` subscript"), stating that consistency with the
construction site outranks the local awkwardness. Do not silently rename
into the docs — that creates a code↔corpus spelling twin.

**(g) `.. (vv-status rationale)` is NOT machine-read — verified, not
assumed.** The scanner regex is `^\.\.\s+vv-status:(.*)$`
(`tests/_harness/audit.py:405`), which the `.. (vv-status rationale)`
comment does not match. So the rationale is free prose and the directive
line is the contract. Self-check without pytest:
`A._scan_theory_equations(Path("docs/theory"))` → fields
`all_labels / documented / skipped / violations`; I read **0 violations,
860 labels, 532 documented**, matching the auto-regenerated matrix
(531 → 532 from my one new label).

**(h) Scope ruling for an OPEN issue's investigation memo.** Published
(i) what the brief handed me and I re-derived (the `Σ = (2/D)1wᵀ − I`
spectrum; checked numerically at `d ∈ {2,3,4}`, plus the step-differencing
contrast `{0}^{d−1} ∪ {(D'−Σ_tV)/D'}` which makes the undamped subspace a
property of the DIAMOND closure, not of transport), and (ii) the two
counterexamples I measured myself. **Withheld** the memo's octant-order
law, its 25-pattern enumeration and its `max_a L_a > Σ_b L_b` predicate:
live findings with an unacted recommendation on an OPEN issue are the
main agent's to publish. Tombstoned, did not delete, the refuted
`ndim` reading — in all three places it appeared (theory page, a sibling
page's Key Facts, and the production docstring's user-facing
recommendation), because a half-done correction leaves a page
contradicting itself (vv anti-pattern #21's aggravator).

---

## L-052 — two dead-ref instruments DISAGREE BY DESIGN; the disagreement IS the triage

Task: the `nexus dead-references` SessionStart hook reported **21 targets /
30 sites** while `tools/check_docstring_xrefs.py` reported **0 dead across
14 914 roles**. The instinct is "one of them is broken". Neither was.

**The three-part scope story (measured 2026-08-09, `refactor/operator-strategy-layers`):**

1. **Trees.** The gate's default `roots = ["orpheus", "tests", "docs"]`
   (`tools/check_docstring_xrefs.py:199`). `examples/`, top-level
   `derivations/`, `scratch/` and `tools/` are never walked. Nexus walks
   the whole project. ⇒ 7 of 30 sites invisible to the gate for this
   reason alone. NOTE the gate's NAME understates it: it DOES read `.rst`
   whole-file (`iter_text_blocks`), so `doc:` sites in theory pages ARE
   in scope — "docstring" in the filename is wrong by omission.
2. **Roots.** `DECIDABLE_ROOTS = ("orpheus","numpy","scipy","sympy","pytest","matplotlib")`
   excludes `tests`, `tools`, `derivations` — all three ARE importable, so
   this is a coverage gap, not an impossibility. And UNQUALIFIED refs
   (`:func:`compute_G_bc``) are skipped BY DESIGN (the tool refuses to
   emulate Sphinx's module-context resolution rather than manufacture
   false positives). ⇒ 6 more targets.
3. **⭐ The semantic split, and the one that matters.** The gate resolves
   by **IMPORT**; nexus resolves by **RENDERED TARGET**. A live module
   that no `api/` page `automodule`s is *resolved* to the gate and *dead*
   to the hook. That is the ENTIRE bucket-C class (12 of 21 targets here),
   and both tools are right: the symbol exists (gate) and the role has no
   link target (hook). Neither number is a bug; **the set difference is
   the triage** — hook-minus-gate ≈ "un-surfaced but live" (issue #302),
   gate-and-hook-agree ≈ "genuinely retired or moved".

⇒ never write "all trees at 0" in a memory index from a gate run. Write
the trees, the roots, and the resolution SEMANTICS, or the claim is
present-tense-false the day someone points a second instrument at it.

**Two false-negative classes found in the gate while establishing this:**

- **PEP-420 namespace packages resolve.** `orpheus/derivations/continuous/{pn_method,
  spn_method,spectral_collocation,spectral_resolvent,escape_probability}/`
  each contain ONLY a `README.md` — no `__init__.py`, no Python.
  `importlib.import_module` succeeds anyway (`__file__ is None`, 0 members),
  so `resolve()` returns True. A `:mod:` role at such a target can NEVER
  resolve in Sphinx. Discriminator to add if the gate is ever hardened:
  `mod.__file__ is None`.
- **⭐ A role wrapped INSIDE its dotted path is invisible to BOTH tools.**
  `:func:`~orpheus.numerics.eigenvalue.\n        power_iteration`` — docutils
  collapses the newline+indent to a space, so the target becomes
  `...eigenvalue. power_iteration` and never resolves; the gate's
  `extract_target` returns `None` on any target containing whitespace, so
  it SKIPS it. Measured tree-wide: **15 such roles** (orpheus ×13,
  tests ×1, examples ×1). The discriminator is NOT "role spans lines" —
  ~180 multi-line roles are FINE because they break at the
  `display <target>` boundary. The regex that finds only the broken class:
  `\.\s*\n\s*\w` (or `\w\s*\n\s*\.\w`) inside the pre-`<` head.

**Bucket discipline that made the triage cheap.** (A) MOVED / (B) RETIRED
/ (C) NEVER-AUTODOC'D is decided by TWO probes, not one: does the symbol
import (A/B vs C), and is its module in the `automodule` set
(`grep -o "automodule:: (\S+)"` over `docs/**/*.rst`, 49 here). Anything
that imports AND is un-automodule'd is C — hands off, it is #302's.

**⚠ The reported TARGET NAME can be an artifact of a THIRD tree.** Six
"dead targets" were named `orpheus.derivations.peierls_geometry.*` — a
module deleted at `bda76faf`. No doc page contains that string. The node
exists ONLY because three `scratch/derivations/diagnostics/*.py` still
`import` from the dead path, minting an `unresolved` ast_only node; nexus
then attached the theory pages' UNQUALIFIED `:func:`compute_G_bc`` roles
to it by suffix match. The functions are alive at
`...peierls_nystrom.geometry`. ⇒ before believing a dead target's NAME,
`SELECT source,type FROM edges WHERE target=?` on `graph.db` and read the
edge TYPES: `documents` = a doc page, `references` = a docstring,
`type_uses` = a **type annotation** (i.e. real code, not prose), `calls` =
an import/call that MINTED the name.

**A `type_uses` site is a CODE bug wearing a doc-ref costume.** Three of
the 30 "sites" were `TYPE_CHECKING` imports in `examples/` and
`derivations/`, i.e. `ModuleNotFoundError` at runtime (or a pyright red)
— not prose at all. Fix the import, then ask whether the annotated body
still matches the successor type: `examples/discrete_ordinates/plotting.py`
annotated `result: SNResult`, and the live `Solution` has no `.geometry`,
no `.eg`, and a `ScalarFlux` with **no `__getitem__`** — so a bare repoint
would have replaced a dead name with a live LIE. Published the repoint
PLUS a measured `.. warning::` naming the three surviving stale accesses.

**The unit of repair is the TARGET, not the reported site.** Nexus counted
3 sites for `orpheus.sn.geometry.SNMesh`; the live tree had **13**
(12 × `derivations/diagnostics/*.py` + 1 × `examples/`), because
dead-references only reports `documents`/`references`/`type_uses` edges
and most were plain `import` statements. Fixing 3 and leaving 10 would
have been the half-correction anti-pattern. Also: nexus counts doc sites
**per PAGE**, so "compute_P_esc ×2 sites" was 9 role occurrences.

**Adjacent classes found, REPORTED not fixed** (each is its own pass):
(a) `derivations/diagnostics/` is broadly bit-rotted — after my repoint,
**15 of 39** files still carry an unresolvable `orpheus` import
(`orpheus.sn.quadrature`, `orpheus.sn.operator`, `orpheus.sn.boundary_realizer`
are all gone); say so, or the repoint reads as "these scripts run now".
(b) `docs/theory/references/trajectory_resolvent.rst` carries **31 stale
`:file:` paths** — a page rename (`peierls_greens.rst` →
`trajectory_resolvent.rst`) was applied to the TEST FILENAMES in prose,
which were never renamed. Raw paths warn NOWHERE. Fixing the 1 in my
brief and leaving 30 would have made the page contradict itself; report
the class.

**Concurrency hazard on a busy branch.** The after-measurement grew ONE
new dead target that was not mine: another agent's #340 N2b docstring
landed in `orpheus/cp/solver.py` mid-session with a wrapped role. `git
status` at session start is the only way to attribute; state the
before/after BOTH ways (raw, and net-of-not-mine). Same for
`docs/theory/verification/matrix.rst`, which my `-E` rebuild regenerated
to absorb their +1 foundation test — legitimate by-product, report it,
never revert it.

**Result.** 21 → 14 targets / 30 → 17 sites (13 + 1-not-mine); the 8
cleared are exactly buckets A and B. Gate stayed at 0 dead. `-E -W`-clean
baseline and after builds both EXIT 0 with a byte-identical (empty)
WARNING/ERROR/CRITICAL set.

**Quality self-assessment.** Derivation depth n/a · Cross-references 5 ·
Numerical evidence 4 (before/after counts, per-tree measurements, the
15-role scan — no convergence table is possible on a reference-integrity
pass) · Failed approaches 4 (recorded WHY the peierls_geometry name is an
artifact, and why 3 adjacent classes were left) · Code traceability 5 ·
Derivation source n/a.

---

## Quality self-assessment rubric (Directive 3)

Rate each output 1–5 and log the weakest dimension in the return:
Derivation depth · Cross-references · Numerical evidence · Failed
approaches · Code traceability · Derivation source (from `derivations/`,
never hand-written). The recurring weak dimension on TERMINOLOGY/ROUTING
fixes is "numerical evidence" — structurally absent (no flux moves → no
convergence table), not a deficit; say so rather than manufacturing one.

---

---

## L-053 — a per-site adjudication TABLE is an instrument too: audit its SKIP clause, its "retired" verdicts, and its `hasattr` evidence

**Context.** #346 W1: a 91-site / 64-distinct ruling table (`scratch/issue_346_w1_adjudication.md`),
built by `w1_list.py` over `graph.db` + the `check_docstring_xrefs` resolver, sorted into
Groups 0–5 with a per-site fix already decided. The brief said "do not re-derive the rulings;
apply them", and warned the instrument had mislabelled things three times. Applying it
faithfully still produced **five ruling corrections**, each from a different structural cause.

### (a) The SKIP clause is a false-NEGATIVE source, symmetric to the false positives the brief warned about

`w1_list.py` keeps a site only if the target is absent from the graph by **tail match**
(`s.endswith("."+t)`). That clause is what suppressed the brief's *known* false positives — and
it equally suppressed **alive-but-unqualified** roles whose tail happens to exist:
`:class:`Field`` on the very line I had to edit, `:class:`AngularFlux``,
`:class:`~geometry.mesh.BC`` (8 lines above three BC-member sites I *was* given), and the whole
`~geometry.*` / `~transport.*` prefix-omission family across ≥6 pages. `7d7661b2`'s own commit
message names it: **1431 relative roles across 49 `.rst` pages**, deliberately deferred.

⟹ When a work-list is generated by *"report if NOT in X"*, the deliverable owes a sentence on
what the NOT-clause hides. And the boundary decision is **file-convention**, not tidiness:
qualifying `~geometry.mesh.BC` on one line while lines 15/16/59/103 of the same file keep the
project-wide short form makes the file *less* internally consistent. I fixed `Field` (whose page
convention IS fully-qualified) and left the `~geometry.*` family, reporting the count.

### (b) "RETIRED → literal" must first ask *retired, or MOVED?*

Seven sites (`peierls_slab` ×6, `peierls_cylinder`) were listed under "Group 4 — RETIRED …
past-tense history → literals". `git log --oneline --diff-filter=D --all -- '*peierls_slab*'`
→ `bda76faf refactor(derivations): reorganize into common/discrete/continuous architecture` —
a pure `git mv`. The module is **alive** at
`orpheus.derivations.continuous.peierls_nystrom.slab`, and the page's own next line says it is
*"retained indefinitely, not retired, as an independent cross-check implementation"*. Six of the
seven sentences are PRESENT tense or IMPERATIVE ("modifications … should preserve"). A literal
would have thrown away a live link and past-tensed a live module.

⟹ Run `--diff-filter=D` on the old path **before** accepting any "retired" verdict. One command.

### (c) A dead `:attr:` can be a TRUE claim — `hasattr(Cls, x)` is the wrong probe

The table's `[M]` for Group 3 item 2 was `hasattr(SNMesh, "mesh") is False`, ruling
*"After C5.1, `SNMesh.mesh` is inbound provenance only … `None` when built from axes"* as
present-tense-FALSE and demanding a rewrite. Constructing the object says otherwise:
`SNMesh(Mesh1D(...), Quadrature.gauss_legendre(4), {0: mix}).mesh` → a `Mesh1D`. The attribute is
set on the **instance** by the base `MaterialMesh._init_data`, with no class-level annotation —
so `getattr(cls, …)` fails, autodoc mints no target, and the role renders plain text while the
*sentence* is exactly right. `augmented_mesh.py` even spells the doc's other half verbatim:
`mesh = legacy_mesh_from_axes(...) if len(axes) <= 2 else None`.

⟹ Group-3 ("prose is false") and Group-0 ("alive, unlinkable") are separated by **constructing
the object**, never by a class-level `hasattr`. The fix is a literal + a live `:class:` ref +
one clause saying *why* it cannot be a role — not a rewrite of a true paragraph.

### (d) A "role misuse → literal" ruling loses to the page's own convention six lines up

`:meth:`B.apply`` was ruled Group 5 ("names an instance, no qualification can ever make it
resolve"). Six lines above, the same page already writes
`:meth:`B.apply <orpheus.sn.operators.boundary.SNBoundaryOperator.apply>`` — `B` *is* the
`SNBoundaryOperator` and `.apply` *is* a real method. Literalising it would have been a
regression against a live, working, explicit-title link in the same paragraph block.

Conversely `:class:`ReflectiveBoundary.apply`` / `:class:`WhiteBoundary.apply``, ruled Group 2
(*repoint to `:meth:`~orpheus.geometry.boundary.reflective.ReflectiveBoundary.apply``*), resolve
**DEAD** — those classes carry `realize` / `geometry_map` / `response_kernel` / `source`, no
`apply` — and sit in a subsection headed *"What was tried and rejected"*, twenty lines under the
page's own sentence *"Calling `law.apply(psi)` raises `AttributeError` at runtime"*. Repointing
would have minted a live role at a method the page explicitly says the type system removed.

⟹ Two adjacent `X.apply` sites, opposite correct answers. **Resolve `Instance.method` by asking
what the instance's TYPE is and whether that type has the method** — never by the shape of the
target string.

### (e) Fixing one role can drop you into a page-wide SYMBOL-INDEX collision — measure, mark, refuse

The Group-4 row `_ki4_lookup` (one table cell) opened this:

| claim | site | measured |
|---|---|---|
| cylinder kernel is `Ki_4` | `collision_probability.rst` ×8 + 5 sibling pages | code ships `_ki3_mp`; `[M]` `_ki3_mp(0) = 0.7853961`, `_ki3_mp(1) = 0.2378450` = **standard** `Ki_3` (`π/4`, `0.2378450`) |
| `:eq:`ki3-def`` defines `Ki_3` | `= ∫₀^{π/2} e^{−τ/sinθ} sinθ dθ` | that is the **standard `Ki_2`** (one power of `sinθ` short) |
| `F(0) ≈ 0.4244` | geometry-comparison table | `= 4/(3π)`; matches **neither** convention |
| `self._kernel = Ki_4` | prose | live line is `self._kernel = _ki3_kernel` (`= _ki3_mp`) |
| "20 000-point lookup table" | Key Facts line 24 | retired at Phase B.4 / #94 → degree-63 Chebyshev interpolant of `e^τ Ki_3(τ)` |

So the page runs a **consistent internal index one below the standard**, under which its `Ki_4`
IS the shipped function — the *structure* of `:eq:`second-diff-cyl`` / `:eq:`self-cyl`` is right
and only the *symbol* is wrong; but `ki3-def` is separately off by a power, and `F(0)` is simply
wrong. Those three labels carry **64 / 24 / 54** `verifies()` tests
(`docs/theory/verification/matrix.rst`).

⟹ **The tests cannot see this.** They pin the code's numbers; the equation's *symbol* is outside
everything they measure — a documentation-side Mode-12: the measured functional (the test) is
invariant to the error class (the name). No gate exists, and none can be added without changing
what the equations say.
⟹ Correct move: repoint the role, **measure** the discrepancy into a `.. warning::` with an
anchor, fix only the unambiguously-wrong number (`F(0)`), and explicitly REFUSE the 142-test
re-indexing as a numerics adjudication. Renaming across ~30 corpus sites inside an xref pass
would be an unreviewable physics edit riding in a hygiene commit.

### (f) "Qualify so it resolves" is TWO claims, and they come apart — measure which one you bought

`[M]` post-build, live `href` counts for repointed targets:
`EigenvalueSolver` **43**, `Field` **30**, `numpy.array_equal` **6** — real links.
`KEigenvalue` **0**, `AngularFlux.zeros_on` **0**, `BC.vacuum` **0**, `SNMesh.axes` **0**,
`peierls_nystrom.slab` **0** — still plain text, because their modules are `:noindex:`-`autoclass`'d
(`docs/api/geometry.rst`) or not `automodule`'d at all. The gate and the graph are satisfied; the
reader is not.

⟹ Never write "these now resolve" unqualified. Write "**import-** and **graph-**resolvable;
N of M also render as links, the rest await #302 surfacing" — and check with
`grep -o 'href="[^"]*#<target>"'` in the built HTML, which is one command.

### What worked (repeat)
* **Every structural assert on the in-memory result BEFORE any write.** Four scripts aborted on a
  bad anchor/count and left the tree untouched — no `git checkout` recovery, which this tree forbids.
* **Anchor on CONTENT, never the table's line numbers.** They shifted by tens of lines after the
  first insert into `boundary_conditions.rst`; a line-keyed second pass would have corrupted it.
* **`resolve()` from the gate itself as the pre-flight probe.** Every target import-verified
  *before* it was written, so the final gate run was a confirmation, not a discovery.
* Assert failures on my OWN new text (a literal `` ``fn_method.core.x_function`` `` inside the note
  that says the module does not exist) are a good sign the assert is tight — narrow it to the ROLE
  form, don't weaken it to a substring.

---

## L-054 — a landed CARVE's docs pass: grep the CONCEPT in every SPELLING, and the brief's page count is an over-count AND an under-count at once

**Instance.** Q5.6.4 (branch `refactor/operator-strategy-layers`, `3dda18ca` + `d5067c4d`):
the cylinder's angular cell partition moved from the η-midpoint (chord) to the midpoint in ω,
and the `[½,1]` τ absorber retired. Brief asked for a label rename
(`morel-montry-clamp` → `morel-montry-closure`), a `cases` split into closure + partition,
a two-page claim repair, and a concept sweep it enumerated as **17 pages**.

### The page count was wrong in BOTH directions, from ONE cause: I grepped SPELLINGS, not the concept

The brief's list came from a grep of `clamp|absorber|[½,1]|τ_raw`.

* **Over-count — 11 of 17 pages were FALSE POSITIVES**, all from one word: `absorber` is
  ALSO a *material* ("pure absorber", "thick absorber", "cavity-absorber"), and `clamp`
  is also a GMRES `restart` clamp, an interpolant clamping to zero, and the LD weight's
  own legitimate `[½,1]`. Real M-M-clamp content lived on **6** pages
  (`foundations/structured_geometry`, `verification/sn`, `methods/sn/{curvilinear_one_group,
  curvilinear_numerics,history,curvilinear_multigroup}`). Cost of not checking: a sweep that
  "fixes" a physics term.
* **Under-count — one page the brief did NOT list carried a present-tense-false BOUND.**
  `methods/sn/angular_quadrature.rst:369` read *"the raw march-start weights satisfy
  `τ_raw ∈ [1/5, 4/5]` with the **bit-exact** reversal identity"* — both halves stale
  (now `[1/4,3/4]`, 0.5–12 ULP). It was invisible to the brief's grep because the page
  spells it **`\tau_{\rm raw}`** — a LaTeX spelling that matches neither `tau_raw` nor
  `τ_raw`. ⟹ **for any math symbol, grep at least three spellings: the ASCII identifier
  (`tau_raw`), the Unicode prose form (`τ_raw`), and the LaTeX role body
  (`tau_{\rm raw}` / `tau_{{\rm raw}` / `tau^{\rm raw}`).** I found it only because I ran a
  *residual* sweep on the NUMBER (`tfrac15|tfrac45`) after the build was green.

### A retirement propagates to a page's BOUNDS and its DIAGNOSTIC-CLASS claims, not only its symbols

Three shapes I had to fix that no symbol grep reaches:

1. **A numeric BOUND is a claim about the retired object.** `[1/5,4/5]` and "bit-exact
   reversal" were properties of the CHORD partition. Present in a theory eq
   (`morel-montry-folded-arc`, a live `verifies()` target — keep the label, rewrite the
   BODY), in a production docstring (`directional.py:717`), and on an unlisted page.
   `[M]` I re-measured all five rows from the live producer before publishing:
   `[0.292893,0.707107] / [0.259892,0.740108] / [0.252425,0.747575] / [0.250603,0.749397]
   / [0.250151,0.749849]` with 0.5/0.5/2/7/12 ULP — matched the live test docstring exactly.
2. **A section's THESIS can rest on a requirement that no longer exists.** `verification/sn.rst`
   argued the cylinder floor "structurally blocked" partly via *"No partition
   (midpoint/cumulative-weight/ordinate-interior) gives `τ_raw ∈ [½,1]` with bounded edges."*
   `[½,1]` was never a requirement — it was the ABSORBER's box. So the paragraph searched for
   a partition satisfying a condition no reference imposes, and a partition satisfying the
   REAL predicate (P3, `[0,1]`) had just shipped. Fix = `.. note:: Retraction` quoting the
   old text, keeping the surviving half (the azimuthal-duplication argument, which is about
   the MARCH not the partition) and re-deriving the cumulative-weight observation as a
   **P3** failure with the measured ladder.
3. **A "the floor is independent of X" claim can be REFUTED by the very carve that retires X.**
   `curvilinear_one_group.rst` said the residual floor "is independent of the spatial closure,
   the default, and the τ-clamp". `[M]` removing the clamp moves the floor 1.8–3.4×. The
   quadrature-scaling attribution survives; the independence clause does not. ⟹ **after a
   retirement, grep the retired thing's name inside "independent of" / "unaffected by" /
   "does not depend on" sentences** — a negative claim about it is exactly the claim the
   retirement can falsify.

### Publishing a `[M]` number the commit hands you: check its CONFIGURATION, because agreement can DEGRADE with refinement

The commit stated the closed form `τ_m = ½ + ½·cot(ω_m)·tan(Δω/4)` "verified to `1.67e-16`".
`[M]` I re-measured on `folded_product(n_mu=4, n_phi)`, max over all four levels:
`1.1e-16 / 2.2e-16 / 7.8e-16 / 7.4e-15 / 2.3e-14` at `n_φ = 4/8/16/32/64`. The single figure
is a small-`n_φ` reading; publishing it bare would have implied a machine-epsilon identity
that **degrades two orders** by `n_φ=64` — and the shipped gate knows this (`atol=1e-13`).
⟹ **an agreement number is a LADDER unless proven flat. Measure the ladder, publish the
ladder, name the gate's tolerance, and say a finer row must widen it (`vv-principles` #16).**
Same trap on a per-level ratio: the code's "(spread 0.30→1.53)" looks like a convergence
sequence and is actually the **`n_φ=16` row across one level** (`[M]` 0.59–1.41 at 8,
0.30–1.53 at 16, 0.08–1.57 at 64) — read a two-number "spread" as *one* measurement until
you have re-run it.

### A retired MODULE whose FUNCTION survives by name is a semantic trap, not a repoint

`derivations/discrete/sn/contamination.py` retired into `angular_differencing.py`, and
`morel_montry_weights` **survives by name** — but it now DELEGATES to production, so it is
**no longer an independent reference**. Four `:func:`…contamination.morel_montry_weights``
refs on `curvilinear_one_group.rst` sat inside sentences crediting it as
*"the structurally-independent reference"*. A mechanical repoint to the live path would have
produced four working links to four false COVERAGE claims (lessons §7). Correct treatment:
**de-role to a past-tense literal in the historical narrative + ONE new anchored
`.. note::` (`sn-tau-reference-migration`) that states the delegation, why it was chosen
(a reference must not be free to drift into a second definition — which is exactly how the
old module's cylinder arm went wrong, `[M]` τ off by 6.8e-2), and what each arm compares
against NOW** (hand-authored: an inline cumulative-weight expression on the sphere, the
analytic closed form on the cylinder, with the retired chord convention as its
`vv-principles` #19 negative control). One anchor, four referrers — not four rewrites.

### A BLIND diagnostic earns a `.. warning::` on the page that recommends it

`contamination_beta` is *identically zero on a σ_y-folded arc for ANY antisymmetric edge set*
— `[M]` production `+6.94e-18`, edges×0.5 `+3.47e-18`, edges **CUBED** `+1.73e-18`, random
antisymmetrised `−3.47e-18`; only breaking antisymmetry moves it (`−3.53e-03`). It certified
a convention that **diverges the solve**. The theory page said "the module computes β for any
quadrature and geometry" with no caveat. ⟹ when a carve discovers a published diagnostic is
Mode-12 blind on the shipped rule, the doc owes a `.. warning::` with the *garbage-passes*
table, and must name the instrument that DOES discriminate (here ν-closure, solve-free —
`[M]` reproduced the whole table exactly: arc/chord `1.000000`, clamped `1.016389→1.000030`,
`τ≡½` `1.164784→1.002412`).

### Two build defects my own new text introduced — both from writing prose, both -W-caught

* `*"… (*:math:`X`*), …"*: an italic run interrupted by a `:math:` role gives
  **"Inline interpreted text or phrase reference start-string without end-string"**.
  Fix: escape the seam — `(*\ :math:`X`\ *)`.
* A hand-aligned simple table whose header cell (`:math:`n_\varphi` 8→16`, 22 chars) overflows
  its `===` column (18) → **`ERROR: Malformed table`**. Fix: use `list-table` for anything with
  a role in a header cell. **Never hand-align a simple table containing a `:math:` role** —
  the role's source length is not its rendered length and the column arithmetic is invisible.

⚠ Both were caught only because I re-ran `-E -W` after writing. Two of my four builds were
**wasted** by launching before the last edit landed: ⟹ **finish EVERY edit and EVERY residual
grep, then build once.** Sequence the session as: baseline `-E -W` → all edits → all greps →
xref gate → AST doc-only proof → ONE verification build.

### What went right, keep doing

* **Baseline re-measured this session: EXIT=0, ZERO W/E/C** — so the gate was count-**equal**,
  not count-unchanged-from-a-quoted-number. Final: EXIT=0, zero W/E/C.
* **`tools/check_docstring_xrefs.py orpheus tests docs --quiet` → `DEAD TARGETS: 0`** before
  and after (11 272 → 11 274 decidable). This is what proved the retired-symbol sweep, since
  `pole_angular_closure` has no `automodule` and no build severity can see its refs.
* **AST-with-docstrings-stripped vs `HEAD`** proved both production edits doc-only
  (`reduced_operator.py`, `directional.py` → `identical = True`).
* **Rendered-href check in the built HTML** for every new `:ref:`/`:eq:` I minted
  (`sn-tau-absorber-retirement` 10 hrefs, `angular-cell-partition` 11, etc.) — cross-doc
  `:ref:` renders plain text with no warning, so this is the only proof.
* **Three orphaned July HTML files** (`theory/structured_geometry.html`,
  `theory/discrete_ordinates.html`, `theory/methods/sn/verification.html`) still carry
  `equation-morel-montry-clamp`. Discriminated by `test -f <source>.rst` → no source ⟹
  orphaned build output from a page split, NOT a live stale ref. Do not `rm -rf docs/_build`.
* **Programmatic splice with all structural asserts run BEFORE the write** caught two of my
  own mistakes (a surviving `morel-montry-clamp` on a line ABOVE my slice; a wrong expected
  ref count) with the tree left untouched.

### Scope discipline that held

`sn-tau-mm-raw` on `verification/sn.rst` is the SAME naming-honesty disease the brief flagged
(a label spelling "raw" on an equation with no raw/clamped distinction), 60 lines from the
rename. I did **not** rename it — a label rename has a V&V-matrix footprint and the brief
named exactly ONE. I fixed the equation BODY's notation (`\tau_n^{\rm raw}` → `\tau_n`),
left the label, and put the reasoning + the exact 3-site rename cost in a `.. NOTE` comment
beside its `vv-status` directive so the next session can execute it in one pass. ⟹ **when a
carve reveals a second instance of the disease it fixes, fix the CLAIM and document the
rename cost in place; do not annex the rename.**

---

## L-055 — the brief's measured number can be TRUE and its FRAMING already refuted, by a test module committed the SAME DAY

**Instance.** Q5.6.4 follow-on, branch `refactor/operator-strategy-layers`, 2026-08-11: a
citation-defect repair (Hébert-vs-BMC as the source of the weighted M-M `τ`) carried one
side-task — *"`docs/theory/verification/sn.rst` ~1186 asserts 'Positivity is never needed'.
We have just measured the HALF-ANGLE flux reaching `min ψ̂ ≈ −77` on a normalised cylinder
problem under the shipped convention. Determine whether the claim is scoped to the converged
SCALAR flux (fine) or is general (falsified)."*

### What I nearly published

I reproduced `−77.1643` exactly (`folded_product(4,32)` level 0, `exp(−6cos ω)` profile,
positive constant seed, through the production `compute_psi_half_per_level`), extended it to
all four τ conventions (`chord −229.7 / chord+absorber −23.3 / arc −77.2 / τ≡½ −24.2`), and
drafted a `.. warning::` ending in *"⚠ **Coverage gap**: `[M]` there is **no** ψ̂-positivity
gate on either arm."* Evidence: an untracked `scratch/` QA memo saying exactly that, plus my
own grep of the 15 `tests/` files mentioning `psi_half` for an assert-on-the-same-line.

### What the tree actually said

`tests/sn/sweep/curvilinear/test_psi_half_positivity.py` — **19 `foundation` rows, committed
the same day**, and it is a CHARACTERISATION module whose docstring *pre-emptively refutes the
`−77` framing*: `[M]` on a heterogeneous 2G cylinder with the **marched ψ½ seed** (the
production value path) ψ̂ is strictly **POSITIVE** — `+0.1337/+0.1286/+0.1287` at
`n_φ = 6/8/16`, i.e. 0.88/0.93/0.98 × `min ψ`. Only an **inconsistent** (zero) seed goes
negative — `−12.09/−16.35/−25.89` on the *same* converged flux — bounded by the recurrence's
worst partial amplification `A(M) = max_m Π(1−τ_k)/τ_k = 2.73/3.36/4.73`. ⟹ *the sign is a
property of the SEED's consistency, not of the scheme*, and my `−77` is an inconsistent-seed
statement. Reproduced all of it; 19 passed in 3.7 s.

⭐ **Rule: when a brief hands you a measured number to PUBLISH, grep `tests/` for a module
NAMED after the phenomenon before you write its interpretation.** A `scratch/` memo is by
construction OLDER than the tests it motivated, and a same-day characterisation module is the
likeliest home of the corrective framing. My line-based `psi_half` + assert grep missed it
(`vv-principles` #21: the subject and its assertion sit on different lines, and the module's
real evidence is in `pytest.fail` messages and the docstring, not in `assert` lines) — the
grep that found it was the read-only `tests/` sweep the brief asked for *for reporting*.

⭐ **Corollary — the correct verdict on the claim was a THIRD option the brief did not
offer.** Not "scoped to the scalar flux (fine)" and not "general (falsified)": the claim is
*general in wording, sphere-in-evidence, and substantively TRUE on the cylinder's production
path as a characterisation*. What is false is the word **"never"**. So the edit scopes the
heading (`The clamp buys no positivity on the SPHERE'S converged solve`), states the seed
taxonomy with both measured tables, keeps W1's conclusion standing (the clamp reduces the
excursion ~10× but does not remove it, and neither does `τ≡½` — the exposure is
`(1−τ)/τ`, a property of the *angular diamond family*), and points at the owning module. When
a brief offers a binary and the tree supports neither pole, say so and publish the third.

### `ref.ref` also fires for an anchor before a **BOLD RUN-IN HEADING**

`.. _label:` immediately above `**(2) Some Title.**  Prose…` gives
*"Failed to create a cross reference. A title or caption not found"* at every bare `:ref:` to
it — **5 hits across 4 files, all from one anchor**, and `-W` turns them into EXIT=1. A
run-in bold heading *looks* like a heading and is not one. Two fixes, both used here:
promote it to a real titled subsection (what I did for `sn-tau-source-of-record`), or use
explicit link text `` :ref:`the β-blindness warning <sn-tau-beta-diagnostic-blind>` `` (what
I did for an anchor that legitimately sits above a `.. warning::`). ⚠ **Do not open the new
title with `(1)` / `(2)`** — that is an enumerated-list marker in RST; use
`Correction 1 — …`.

### Build sequencing, again

Four `-E -W` builds where two would have done: baseline (0 W/E/C), a verify launched *before*
the last edit landed (wasted), a third that caught the 5 `ref.ref` warnings (earned), a
fourth that closed at EXIT=0 / 0 W/E/C. The wasted one is exactly L-054 §9's warning. The
earned one is the argument for never skipping the post-edit build even when the xref gate is
green — `tools/check_docstring_xrefs.py` reported `DEAD TARGETS: 0` while all 5 `ref.ref`
warnings were live, because it gates **Python-domain roles**, not `:ref:`.

### Doc-only proof and the generated-artefact by-product

9 production `.py` files touched, all proved DOC-ONLY by AST comparison against `HEAD` with
docstrings stripped. The `-E` build regenerated `docs/theory/verification/matrix.rst`
(9544 → 9628 tests) absorbing rows from **another agent's uncommitted `tests/sn/sweep/` work**
(`test_psi_half_positivity` +19, `test_angular_cell_partition` +56,
`test_tau_producer_equivalence` 5→14) — the L-008 by-product: never revert it, report it.

---

## L-056 — A labelled equation drifted from its own prose; a scope revocation landed mid-edit

**Task.** Repair `docs/theory/foundations/discrete_measures.rst` §"Quadrature selection
algorithm", which described the pre-2026-08-02 quadrature selector (4 stages, declared-tag
symmetry gate) while the code had shipped 5 stages with a computed `Sym(Q)`. Baseline and
final `-E -W` builds both EXIT=0 / **0** W-E-C. One tracked file edited: +415/−74.

### 1. ⭐⭐ The DEFINITION LIST is the tell that a labelled equation drifted

The commit that fixed the design (`e7d44f3c`, 2026-08-02) rewrote the geometry table, the
worked examples, the rejection messages **and the predicate quoted inside the equation's own
`.. (vv-status rationale)` comment** — and left the `.. math::` body alone. So the page
carried the corrected claim in a comment eight lines below the false claim it annotated.

The mechanical tell needs no code and no build: **the "where" list under the equation defined
`𝒟_Q` and `Sym(Q)` — two symbols absent from the equation — and omitted `G_Q`, which was in
it.** A definition list that does not match its own equation is a correction that stopped one
line short. Add this to the reading pass on ANY page with `.. math:: :label:` + a where-list.

Why nothing catches it: the label EXISTS, so every `:eq:` resolves, `-W` is silent at every
severity including `-n`, and the V&V matrix lists the label as covered — coverage is recorded
against the *label*, not against what the label says. This is `coding-standards`' "a labelled
equation is an API" clause, met in the wild.

⟹ I published the tell IN the page as a `.. admonition::`, not just in the fix. A reading
skill archived where the next reader of that equation will hit it.

### 2. ⛔ A mid-task scope REVOCATION on a file already edited: revert by re-editing, and
**publish the patch you backed out**

The brief said "correct BOTH sites", naming `registry.py:106-107`. I fixed it. A mid-task
addendum then said *"do not edit `registry.py` … confine yourself to the .rst"* — the
coordinator was editing that file concurrently.

- Reverted by **re-editing** (never `git checkout` — the tree carries uncommitted-by-policy
  state), then **proved it** with `git diff --quiet -- <path>`. Do not report "reverted"
  without that proof; an `Edit` round-trip can leave whitespace.
- The backed-out content is not lost: it goes in the RETURN, verbatim, as an apply-ready
  patch. Four `registry.py` sites were owed and the addendum named only two.
- ⚠ **`git status` then showed the file Modified again** — the coordinator's concurrent pass.
  Discriminate by grepping YOUR OWN distinctive strings (`grep -c "teaching artifact"` → 0),
  not by the porcelain flag.

### 3. ⭐ A tombstone that asserts the state of ANOTHER file is false the moment either file moves

My §"Why a registry" tombstone read *"…and the module docstring said the same thing … until
2026-08-14"*. After the revert that was FALSE (the module still said it); after the
coordinator's pass it would be false the other way. Both directions wrong from one sentence.

⟹ **Write a twin's history in the PAST tense of the CLAIM, never of the file's state.** Landed
form: *"The promise was minted twice: the module's own docstring stated it more concretely
still, as '…'. That duplication is itself the tell — a claim about a *rendering* mechanism had
never been checked against the rendering, and the module asserting it is the one that is not
rendered."* True whatever the other file does next. Same rule as §2's patch-in-the-return:
**your page may only assert what your page controls.**

### 4. ⭐ Fixing half a claim in one file is WORSE than fixing none (vv #21, met head-on)

The brief scoped me to the §starting at line 682. Three screens ABOVE it, the same page's
"Symmetry groups for quadrature invariance" section still opened *"Quadrature selection in
ORPHEUS therefore reduces to a containment check"* with the retired whole-group mapping
(slab → `SO(2)×σ_x`, sphere → `O(3)`) and closed by calling containment *"sufficient to
preserve every symmetry the geometry exhibits"*. Had I stopped at the brief's boundary the
page would contradict itself and be citable for EITHER sentence.

Repair shape, with the equation and its `vv-status: documented` label untouched (it is
`implements`-cited from `orpheus/numerics/symmetry.py:371,481`): **re-scope the equation to
what it actually is** — the order relation on the `O(3)` lattice, `H ⊆ K ⟺ ∀g∈H, g∈K`,
decided by `SubgroupOfO3.contains` — then a `.. warning::` saying *this relation is not the
selection gate*, with both reasons (a rule's symmetry is a question about NODES; the geometry
side is not one group), then the ⛔ preserving the retired sentences.

⟹ **After repairing a section, grep the WHOLE FILE for the retired predicate's spellings and
adjudicate every hit by tense.** Here: `G_Q`, `G_{\text{geom}}`, `G_{\text{quad}}`,
`four-stage`, `docstring narrates`. Final state: 6 survivors, every one inside a ⛔/history
admonition.

### 5. ⭐ A stale FORMULA can be right on a biased subset of the grid — spot-checking confirms it

Stage 2 gave the level-symmetric degree as `max(3, N−1)` and the positivity frontier as
`S_12` with `[M] −0.027 @ S_14`. Both were honest measurements **of the pre-#337 seed**
`μ₁² = 4/(N(N+2))`; #337 (`59bb38a0`, 2026-08-08) adopted the moment-matched root and moved
both. `[M]` at HEAD over even N ∈ [2,24]: degrees `3,5,7,9,11,11,15,15,17` at S2…S18 (**no
clean formula in N**), min weight at S14 `+0.01299` (positive), first refusal at **S20**.

The sharp part: `max(3, N−1)` is **right at S2, S12, S16, S18** and wrong at S4/S6/S8/S10/S14
— 4 of 9 buildable orders confirm it, **including S12, the order the retired frontier itself
made salient**. A spot-check drawn from the stale claim's own neighbourhood is biased toward
confirming it. (vv #13's congruence-class disguise, one level up: not a sampled group but a
sampled *parameter grid*.) I published this as its own ⚠ and pointed at the sweeping gate
`tests/numerics/test_advertised_degree_is_measured.py` (verified: it sweeps S2…S18).

⟹ And the fix for the drift is **not a better number** — it is a POINTER. The SSOT
(`docs/theory/methods/sn/angular_quadrature.rst` `quadrature-ls-positivity` +
`rules_sphere.py`) was already correct; `discrete_measures.rst` was the only page carrying the
stale copy as current. I replaced the numbers with a `:ref:` and said *why*: the frontier is
discovered by attempting the construction, so a second copy is exactly the thing that drifts.

### 6. Measured, not asserted — the evidence I generated before writing

Ran each before it appeared on the page: the 5 worked examples (all reproduce, incl.
`cylinder d=4 → LevelSymmetricSN(6)` fallback, 48 nodes); the stage-0/1 independence table
(4 rows, `Lebedev(5)` slab `✗/✓`, `GL(8)` cylinder `✗/✓`, `product(3,5)` cylinder `✓/✗`,
`product(3,6)` `✓/✓`); `spec.__doc__ is type(spec).__doc__` **True for all four** with one
shared `id()` and no instance `__doc__`; declared tags now honest (`σ_x`, `O_h`, `O_h`,
`D_{n_φ h}`, and `D_5h`/`D_6h` are **computed** by `_derived_product_group`, so citing them as
evidence about NODES is sound); the lattice-route failure `understated GL → lattice False /
nodes True`; `select_quadrature` has **no production consumer** (grep: def + export + 1
docstring + the test module).

### 7. Mechanics that worked

- Splice by line index with **guard asserts on live boundary strings** + structural asserts on
  the in-memory result **before** the write. First run FAILED on a bad assert (`\;` in a
  non-raw string) and the tree was untouched — the whole point.
- New content authored as a **pure literal** via `Write` to `scratch/`, so no Python string
  layer touched the LaTeX; removed after splicing (it would be a twin).
- An **enumerated list starting at `0.`** is legal: docutils sets `start="0"` and emits only
  an INFO (`report_level=1`), which Sphinx suppresses at default verbosity. Verified with a
  standalone `publish_doctree` probe before committing to the numbering that matches the code.
- ⚠ **Do NOT wrap quoted stale text in `*…*` when it contains `:math:`/`:eq:` roles** —
  docutils does not nest inline markup. Use the page's own idiom instead:
  ``⛔ X read :math:`...` until <date>``.
- Glyph check before use: `⛔` 12 files, `⚠` 17, `⭐` 10, `✓` 10, `✗` 2 in `docs/`. **The
  digest's "`⭐`/`⛔` have ZERO occurrences in docs/" is STALE** — they are corpus vocabulary
  now. Grep, don't recall.
- Two builds, not four (L-054 §9 held): baseline → all edits → all greps → xref gate → verify.

---

## L-057 — a new theorem lands ONE HOP from a page that already owns half of it; and its universal falsifies five sibling claims

**Task (2026-08-15, #344, branch `refactor/track-b-remainder`).** Document `LossKernelGauge`:
`A = L+C−S−B` is exactly singular on a `d ≥ 2` Cartesian diamond box with ≥2 reflective axis
pairs; a converged solve returns an arbitrary member of a solution manifold; a `G`-orthogonal
projector returns the canonical one. Brief named three candidate homes.

### 1. ⭐⭐ The home is decided by WHERE THE MECHANISM'S LOCAL HALF ALREADY LIVES, not by topic

The obvious homes were "boundary_conditions" (the BC is half the precondition) and "solver"
(the gauge fires at the exit). Both wrong. `cartesian_multid.rst` already carried
`.. math:: :label: dd-face-transmission-spectrum` — `Σ = (2/D)·1wᵀ − I`, its `−1` eigenvalue of
multiplicity `d−1`, the "undamped sawtooth", and the #340 measurement of that sawtooth's
signature at convergence — used there **only negatively**, as the obstruction to a Varga
comparison theorem. The kernel IS that local mode closed around a reflective loop. Filing it
anywhere else would have re-derived the transmission spectrum (a Cardinal-Rule-2 twin) and left
the existing section's reader with an incomplete story.

⟹ **before choosing a home, grep the corpus for the new result's LOCAL half.** If a page
already derives and labels the local fact, the global result is a downstream H1 on that page,
and its opening sentence should say *what the existing section stopped short of*. The module
docstring's own `:doc:` pointer named the wrong page; repointing it to the new `:ref:` was
part of the deliverable (a doc-only edit, AST-proved).

### 2. ⭐⭐ A new theorem is also a QUANTIFIER AUDIT — the doctrine it amends is asserted in N places

The result's headline ("a splitting shares a solution SET, not a POINT, when A is singular")
falsifies an unqualified universal that the corpus states **9 times** across 7 files. A
windowed regex (`fixed point` within ±2 lines of `invarian|same|shares`) found them; the
per-site adjudication was NOT uniform:

| site | verdict |
|---|---|
| `cartesian_multid` Key Facts "Only the fixed point is schedule-invariant" | present-tense FALSE → scope to **bulk** |
| `cartesian_multid` FP-invariance ¶ + *What survives* ¶ | FALSE as a universal → keep the sentence, add a `.. note::` tombstone (§3 discipline) |
| `solver.rst` Key Facts + "Two Inner Solvers" ¶ | FALSE → scope + tombstone |
| `foundations/cross_section_data` "identical to the plain (Jacobi) sweep" | FALSE → scope to bulk |
| `slab_one_group` Key Facts "(any consistent splitting … shares ψ*)" | true IN CHAPTER (d=1 kernel-free) but an explicit universal → one scope clause |
| `slab_one_group:852` (slab Krylov ≡ SI) | chapter-scoped, kernel-free → LEAVE, report |
| `foundations/boundary_conditions` C5.5 gate prose | ⭐ the gate's own FIXTURE is singular — see §3 |
| `loss_representation` scan-march gate | measurand is `scalar_flux` → name the measurand |
| `verification/sn.rst` T4 three-splittings | both configs kernel-free → say WHY it is legitimate |
| `foundations/boundary_conditions:2368/2433` (two source-DELIVERY channels) | ⛔ NOT a splitting claim — leave |

⟹ the tell that separates "leave" from "fix" is **which object the sentence quantifies over**:
a claim about *two delivery routes of one iteration* is untouched; a claim about *two splittings*
is not. And a chapter-scoped truth still needs a clause when the sentence contains an explicit
`any …` — a reader arrives at Key Facts by search, not by reading the chapter title.

### 3. ⭐⭐ Auditing a gate's PROSE, I found the gate's own fixture is in the new pathological class

`foundations/boundary_conditions.rst` credits a C5.5 "Mode-9 G-S ≡ Jacobi FP-invariance" gate on
a box described as *breaking every degenerate coincidence*: **x-reflective / y-vacuum /
z-reflective**, cells `(5,3,4)`. That is **two** reflective axis pairs. `[M]` I built it:
`dim ker A = 36` (`= n_g·(N/4)·n_y`). So G-S and Jacobi do **not** return the same trace there —
the gate survives only because what it asserts (`keff` + normalized flux *shape*) is mirror-even
and therefore blind by the new theorem. Publishing "the gate is sound BECAUSE its measurands are
mirror-even, and a future strengthening must not reach for the raw trace" is worth more than
either deleting the claim or leaving it.

⟹ **when a new result defines a pathological configuration class, MEASURE every gate fixture the
corpus names against the class predicate.** It is one `_as_sn_mesh(...)` + one attribute read,
and it converts a prose repair into a durable design constraint.

### 4. ⭐ The brief's headline rule was TRUE-on-its-fixtures and FALSE as stated

Brief: *"Excited iff the FIRST axis has an ODD cell count."* Reproduced exactly (5 excited rows
to 7 s.f., 6 inert). But two further measurements I ran unprompted:

- `dim ker A = 12` at `(2,2) (3,4) (4,4) (5,6) (6,8)` — **parity does not touch the kernel**;
- at even `n_x`, an **anisotropic** source `(1+μ_x)/W` excites it anyway: `1.756363e-02`
  (vs `6.7e-14` for the uniform isotropic source on the same mesh).

So the rule is about **excitation by a symmetric source**, never about the operator, and
"even `n_x` is safe" would have shipped as a mesh property. Published as a `.. warning::` with
both tables and the imperative *assert `dim ker == 0`, never infer kernel-freedom from a mesh
property*. This is `vv-principles` #13's congruence-class trap seen from the doc side.

### 5. ⭐ The "refuted witness gate" is the highest-value paragraph in the page

The campaign's FIRST acceptance-gate design (the `|Ω·n|⁰` full-face moment) **could not have
failed** — a face-summed moment is mirror-even and every kernel mode is mirror-odd, so the gate
was unfalsifiable by the campaign's own theorem, three sections above its own design. I
reproduced the null (`0.0` on 7 of 8 face rows, `~1e-16` on the rest, while the trace moves
6.08 %) and published the refutation beside the theorem that predicts it. A close-out's
*falsified-design* paragraph is what stops the next session re-deriving a green-forever gate.

### 6. ⛔ The changelog entry was BLOCKED by the page's own contract — report, do not fake

`history.rst`'s header states: *"Every entry below is merged to main … a new entry lands with
its merge hash or not at all."* `[M]` `git merge-base --is-ancestor f934ff57 main` → **NO**
(branch is 15 ahead). Every one of the 21 existing entries carries a merge hash. ⟹ writing the
entry now would mint exactly the class of falsehood the page's own 2026-07-24 repair retired.
Delivered the ready-to-paste row in the RETURN instead, with the reason.

⟹ **a deliverable can be blocked by a page's own stated contract; that is a finding, not a
failure to deliver.** Check a changelog's header rules before writing to it.

### 7. Measured, not asserted — everything on the page came from my own probes

Parity table (11 meshes) · Jacobi + Krylov controls · tangential vs normal currents on 2
quadratures · 8 mirror-even functionals before/after · `dim ker A` on 12 configurations
(incl. graded, mixed-BC at two different vacuum axes, LD, d=1) · closure registry
damping × ndim · T-component `G ≡ 0` on 3 rules · projector idempotence + uniform-trace
annihilation (`5.0e-18`) · build/apply cost · **the strongest single row**: gauged trace
recovers the analytic flat answer `6.09e-02 → 1.04e-13` with `‖Π(t−t_exact)‖/‖·‖ = 1.00000000`.

⚠ Two brief numbers I could NOT reproduce as stated and therefore did not publish: the plan's
`‖t_SI − t_Krylov‖ = 1.320828` (a *heterogeneous* fixture I did not have; mine reads `0.124184`
on the homogeneous one — published with MY configuration), and the memo's `41.1 ms` build
(mine: `22.0 ms` at d=3, the fused-SVD implementation). **Publish your own number with your own
configuration; never relay one whose fixture you cannot state.**

### 8. Mechanics

- Fragment authored as a **pure literal** via `Write` to `/tmp`, then spliced by a script with
  (a) underline-length + title-level-skip checks, (b) a `list-table` `:widths:`-vs-cell-count
  checker, (c) an odd-backtick scan — **all run on the fragment before any write**. The
  `:widths:` checker caught a real 3-cell row in a 2-column table.
- ⚠ `Edit` failed on a paragraph whose text I had just read: the source carried a **typographic
  apostrophe** `’` (U+2019) in *"splitting's"*. Shorten the anchor or `repr()` the live line.
- Two builds only (baseline `-E -W` → all edits → all greps → xref gate → AST doc-only proof →
  one verify build). Both EXIT=0 with **0** WARNING/ERROR/CRITICAL — the set unchanged.
- Post-build link audit is what proved the refs: **9 of 9** cross-doc `:ref:`s render as real
  `href`s, the cross-doc `:eq:` resolves, and 6 of 9 code-xrefs link. The 3 that render plain
  text (`solve_sn`, `SNMesh.loss_kernel_gauge`, `IterationHistory.*`) are the page's
  `:noindex:`-automodule convention — `[M]` **0** hrefs and **0** anchors tree-wide for those
  targets *before* my edit, so it is not a regression and half-surfacing one module is forbidden.
- V&V matrix auto-regenerated `534 → 539` documented sentinels (+5, exactly my labels); orphan
  count unchanged at 2. Never hand-edited.

---

## L-058 — a path census keyed to ONE artefact's FILENAME misses its SIBLINGS in the same directory

**Task** (2026-08-15, branch `chore/nexus-project-config`): repoint four instruction surfaces after
the Nexus graph's location moved from a hardcoded `docs/_build/html/_nexus/graph.db` to a single
declaration in `.nexus/config.toml` (`[graph]`), resolving in ORPHEUS to
`docs/_build/html/graph/graph.db`. Brief was explicit and correct about its own limit: *"I censused
with `grep -rln "graph\.db"` which only catches that exact spelling — if the concept is stated some
other way, I want to know."*

### The finding the brief's census could not reach

A build directory holds an artefact **family**, not one file. Measured after a fresh `-E` build:
`docs/_build/html/graph/` contains `graph.db`, `graph.json`, `graph.html`, `traces/` — and
`docs/_build/html/_nexus/` **does not exist at all**.

`docs/index.rst` carried, under a `Knowledge Graph` heading:

```rst
`Open interactive graph explorer <_nexus/graph.html>`_
```

`[M]` the shipped homepage rendered `href="_nexus/graph.html"` → **404**, while
`docs/_build/html/graph/graph.html` (627 053 bytes) sat un-linked. A dead link on the docs
homepage, invisible to every census keyed to `graph.db`, and invisible to every build:
**a raw relative hyperlink is checked by Sphinx at NO severity** (unlike `:doc:`/`:ref:`, which
warn). So `-W`, `-n` and `check_docstring_xrefs.py` are all silent — the last one by design, it
gates Python-domain roles.

⟹ **When a DIRECTORY moves, census the directory, not the file.** Grep the parent segment
(`_nexus/`) and each sibling extension (`graph\.(db|json|html)`), not just the one filename the
brief names. One extra alternation caught a user-facing 404 that four rounds of `graph.db` grepping
could not.

⚠ **Grep hygiene that cost two wasted rounds:** `_nexus` as a bare pattern matches `mcp__nexus__*`
— every MCP tool name in every agent/rule/skill file, 559 KB of output. Anchor it as `_nexus/`
(with the slash) and the census collapses to 9 real lines.

### The residual, stated rather than hidden

A static RST hyperlink has no mechanism to ASK where the graph lives — it cannot run
`nexus config db`. So the repair necessarily mirrors `[graph].output`, i.e. it IS a second
declaration. Rather than pretend otherwise, the fix ships an RST comment above the link naming the
coupling and the reason nothing verifies it. A second declaration that **announces itself as a
mirror** is the honest floor when the single-source mechanism is unavailable; a silent one is the
defect.

### `--db` optionality — the fix is DELETION, not substitution

`[M]` `--db` is optional on all 10 subcommands probed; `resolve_db(explicit, start)`
(`sphinxcontrib/nexus/project.py:88`) encodes **flag > `[graph].db` in the nearest
`.nexus/config.toml` > `LEGACY_DB = _nexus/graph.db`**. So in documentation the correct edit for
most examples is to **delete the flag**, not to update its value — updating it is what mints the
next stale literal. `nexus-exploring/reference.md` taught `--db <path>` on **16** command lines;
all 16 came out, replaced by one header sentence stating the precedence.

⭐ **But keep the flag where naming a file is the POINT of the example.** Per-example judgement,
not find-and-replace: `nexus analyze . --db /tmp/scratch-graph.db` is now *better* documentation
than before, because with a default in place the flag finally has a teachable meaning — a
deliberate override (a scratch graph, a second checkout, a graph you are diffing against).

### The asymmetry a config-driven CLI introduces, which no doc stated

`[M]` `--project-root` exists on `analyze`/`serve`/`config`/`file-brief`/`staleness`/`retest`/
`changes`/`rename`/`briefing`/`audit`, and **`status` REJECTS it**
(`nexus: error: unrecognized arguments: --project-root`). Since `resolve_db` walks up from
`--project-root` *or cwd*, the read-only query subcommands are **cwd-anchored**. Run `nexus status`
from a scratch directory and you get `Error: _nexus/graph.db does not exist / Run 'nexus analyze'
or 'sphinx-build' first` — a message that reads as *"the graph was never built"* when the truth is
*"you are standing in the wrong directory"*. Both skills now state it.

### My own guard asserted the wrong thing — and that is the system working

The splice guard `assert len(out) < len(src)` fired red. The content was fine; the *guard* was
wrong (a 5-line header replacing a 1-line one outweighs 16 × 12 removed characters). The file was
**untouched** — which is exactly the §5 discipline's payoff: every structural assert runs on the
in-memory result BEFORE any write, so a false red costs one re-run and never a `git checkout`.
⭐ The transferable half is `vv-principles` #4's VERIFY sharpening turned on my own instrument:
**a failed check is not a finding until you have diagnosed WHOSE failure it is.** An earlier assert
in the same script had already caught a real miscount (16 flags, not the 15 I eyeballed) — so the
instrument had a positive control before it produced its false red.

### Verified-not-assumed, and left alone

The brief predicted `AGENT.md:104` (`_build/html/ ← Build output (includes Nexus graph.db)`) was
still true and said to verify rather than assume. `[M]` `find docs/_build/html -name graph.db` →
`docs/_build/html/graph/graph.db`, i.e. still *under* `_build/html/`, and the line names no
subpath. TRUE → left. `AGENT.md:508` names `graph.db` bare with no path, and its claim (no graph in
a fresh worktree until the first `-E` build) still holds — `.nexus/config.toml` is tracked, so the
path resolves fine while the file does not exist. TRUE → left. **Two lines correctly NOT edited is
a deliverable**, and saying so is what stops the next session re-opening them.

### Gate

`-E -W --keep-going` baseline **re-measured this session**: `0` WARNING/ERROR/CRITICAL, EXIT=0.
Post-edit identical: `0`, EXIT=0. Built-HTML proof rather than source proof:
`href="graph/graph.html"` with the target present, and `grep -c '_nexus'` = **0** on both
`index.html` and `development.html`.

---

## L-059 — a machine-readable DECLARATION is a doc surface, and its blast radius is the CLAIM it displaces

**Task (2026-08-17, nexus #82).** Author `.. implements::` declarations for one theory page
(`docs/theory/methods/sn/loss_representation.rst`), because a nexus change made *declaring any
implementer of an equation stand the guessing down for that whole equation*. Plus a fix-on-sight
for a falsified docstring. Landed 28 directives over 14 equations, a new H2 recording the three
equations that deliberately declare nothing, and 19 code-docstring corrections.

### 1. ⭐⭐ Measure the instrument you are replacing — the number is the section's whole argument

The brief supplied the mechanism ("all 14004 `implements` edges are guesses"). It did not supply
what the guesses *were*. Four SQL queries against `.nexus/graph.db` turned a policy statement into
the page's load-bearing content:

| measured, pre-declaration, over the page's 14 declared equations | value |
|---|---|
| inferred `implements` edges the heuristic wrote | **397** |
| true implementers (what I then declared) | **28** |
| of those, found by the heuristic | **6** (21 % recall) |
| precision | **1.5 %** |
| guess sets for `loss-rep-LpC` vs `loss-rep-facewise-separable` | **identical, 23 for 23** |
| of `loss-rep-LpC`'s 23 guesses, how many are its 2 real implementers | **0** |

The last row is the one that explains the mechanism rather than scoring it. Both real implementers
live in `orpheus.sn.operators.streaming`; the shared token is the *package name*
`loss`/`representation`, so the guess set is exactly "the membership list of
`orpheus.sn.loss_representation`" — a set that **cannot contain them by construction**. So the
failure is not "imprecise"; it is *not a claim about the equation at all*.
⟹ **When you replace an instrument, publish its measured error, not its described one.** One query
per equation; the table writes the section.

### 2. ⭐⭐ Writing the explanation MINTED new guesses — the honest edit made the metric worse

Post-build the three UNDECLARED equations' guess counts had *risen* (23→24, 23→25, 23→24).
Cause, diffed: my own prose. Explaining why `MaterialXSField.foldable_sigma` is **not** the
implementer of `loss-rep-removal-sigma` added a `:meth:` xref to it — and `foldable_sigma` shares
the token `sigma`, so it was promptly inferred as a new implementer *of that equation*. Same for
`LossRepresentation.streaming_action` on all three.
⟹ **Citing a symbol in order to say it is NOT the implementer is enough to make the heuristic name
it as one.** Two consequences: (a) NEVER publish a live guess count — quote the frozen
pre-declaration measurement or tell the reader to re-run; (b) this is a real finding for the tool,
not a curiosity: an equation with no declaration gets *worse* every time its page is improved.

### 3. ⭐ The authoring contract inverts the usual risk: INCOMPLETE is worse than ABSENT

Because declaring stands the guessing down *per equation, not per pair*, an equation declared with
one of its two implementers shows **only the one** — the guess that used to cover the second is
switched off. So the failure mode of this doc surface is silent under-coverage produced by an act
that looks like an improvement. Discipline: for every equation, ask *what else computes this?*
before writing the first directive. Seven of the fourteen needed 2–4 directives (DD arm + LD arm;
forward + transpose; scheme door + the schedule that folds its transverse term).

### 4. ⭐⭐ The brief's site census was 6 of 18 — a windowed CONCEPT grep found the rest

Deliverable 2 named `streaming.py:546-548` plus six "may be inherited" sites. A windowed regex
(`subtract` within ±4 lines of `Resolution A|StreamingOperator.apply|operator subtract`) found
**18 sites in one file**, all asserting the same present-tense falsehood — that
`StreamingOperator.apply` *subtracts* σ. It does not: since #257 S8b its whole body is
`streaming_action(psi)` = `loss_action(0, psi)`. Leaving 12 copies of the sentence I was fixing
would have been the exact half-fix vv #21 warns about. Fixed all 18 + the brief's one, and
REPORTED the expansion.
⚠ The tell that the file already knew: at `__init__.py:371-375` the *corrected* framing
("`StreamingOperator.apply` calls this directly (#257 S8b) so L reads no σ") sits **one line above**
the stale sibling docstring at :376. A correction pass that stops at the method it was looking at
leaves the file citable for either sentence.

### 5. ⭐ The same falsehood was in the RST — including the page's own Key Facts card

The brief scoped Deliverable 2 to code. The theory page carried the identical claim in **four**
places, one of them the **Key Facts** admonition ("*Gotcha — the operator subtracts C once*"), and
another the prose wrapping `loss-rep-resolution-a` — the very equation I was adding four
`.. implements::` blocks to. Not fixing it would have put my new prose three screens from its own
contradiction, with me as the second voice.
Fix shape, all four: keep the **equation** (the identity `Lψ = (L+C)ψ − σ_t⊙ψ` is TRUE), keep the
**label** (two live `verifies()` markers point at it), correct only the *attribution*, and tombstone
the history in the past tense with a `.. note::` naming the two carves (#240 Step B removed the leaf
sum; #257 S8b removed the subtraction). Retitled the section from "the operator's only glue" to
"one action, two readings of σ" — safe, because the section carries no `.. _anchor:` and the only
tree hits were orphaned `_build/` HTML.

### 6. ⭐ A quantifier in a page table needs its own census, not the brief's

The brief's ⛔ ruling said "the only sites forming `σ_t − σ_s0` are `derivations/.../dsa.py:632`/`:1023`".
`[M]` there are **four**, and the two the brief missed are the more interesting ones:
`orpheus/sn/acceleration/dsa.py:328` (`quarter = 0.25 * (sigma_t - sigma_s0) * h  # ¼ σ̂_R h`) is
**production**, not a derivations mirror; and `orpheus/derivations/continuous/mms/sn.py:1892`
(`SigC=np.array([sigma_t - sigma_s0])`) is a **capture cross-section** — identical arithmetic, a
*material datum* rather than an operator diagonal, numerically coincident with σ_r only because the
fixture has one group. The brief's grep keyed on `sigma_r =`/`sig_r =`, i.e. on the NAME; two of
four sites never bind the name. The ruling survived and got stronger — but the count did not.
⚠ Beware the mirror trap: `sig_r` in `thermal_hydraulics/` and `kinetics/` is a **radial stress**.
A short suffix collides as badly as a one-letter symbol.

### 7. ⭐ "Implemented by nothing" is a CLASSIFICATION, and it is the durable half

Three labels declare nothing, for three *different* reasons, and naming the kinds is what makes the
section reusable: **superseded path** (`loss-rep-leaf-sum` — two independent retirements, the #240
override and the #257 σ-removal, each alone sufficient to make the route unreachable; the identity
stays true, the code that walked it is gone) · **notation** (`loss-rep-removal-sigma` — a definition
with no production caller; every site that computes the arithmetic computes a *different operator*)
· **declared tag** (`loss-rep-facewise-separable` — a `ClassVar[bool]` a human set after doing the
tensor-product argument by hand; the implementer of the *criterion* is the page).
⟹ *A statement can be true, labelled, and implemented by nothing* — and the cases where that is
correct are enumerable, which is precisely what an inference cannot know.

### 8. Gates (all re-measured this session)

* `-E -W --keep-going` baseline **0** W/E/C, EXIT=0 → post-edit **0**, EXIT=0.
* `tools/check_docstring_xrefs.py orpheus tests docs`: HEAD baseline (via `git archive` into a temp
  tree) **81 dead / 124 sites**; post-edit **81 / 124** — identical, while adding **80** xref roles.
  Measuring the baseline from `git archive HEAD` is the cheap way to get a true before/after on a
  dirty tree.
* AST doc-only proof vs `HEAD` for both edited `.py` files (docstrings stripped, `ast.dump` compared)
  — re-run after the *last* edit, not the first.
* Graph confirmation is the real acceptance test for this deliverable: query
  `edges WHERE type='implements'` per equation after the rebuild. 397 → **28**, each equation's count
  equal to the multiplicity declared. A directive that fails to resolve DOES warn
  (`target %r not found in graph — skipping`), so `-W` gates the `:by:` paths — but only the graph
  query proves the *count*.
* Pre-flighted every `:by:` path and every equation label against `graph.db` **before** writing a
  line (23 targets, 17 labels, all resolved). Cheaper than a 6-minute build per typo.

### 9. Reported, not fixed — `tests/` is not mine

Three test-module docstrings assert the retired mechanism. All three were left alone and reported:

* `tests/sn/operators/test_loss_action_convention.py:3-9,20-22` — "*the operator's `apply` applies
  the ONLY algebra glue, the Resolution-A collision subtraction*" and "*`apply` is DEFINED as
  `loss_action − σ_t·ψ`*". ⭐ Its own **function**-level docstring (`:133,:141`) is already correct
  ("the **+C glue**", "the affine relation") — the module header lags the body it introduces.
* `tests/sn/operators/test_streaming_operator.py:8-19` — a `:=` **definition** section titled
  "Resolution A — subtractive definition", plus "*L carries σ_t at constructor time*".
  `[M]` `StreamingOperator` is a dataclass with **one** field (`sn_mesh`) and no `sigma_t` attribute.
  A docstring asserting a constructor signature that does not exist is the loudest class of stale.
* `tests/sn/sweep/core/test_phase_c_gates.py:22,25,371` — names `:class:`CollisionOperator``
  (`[M]` retired at #261, importable from nowhere) and attributes the composite matvec to the leaf
  sum. Its *conclusion* `(L+C).apply(ψ) = M(ψ;σ_t)` is TRUE; only the mechanism is stale.

---

## L-060 — a SPEC has a headline and a table, and they keep different clocks; plus: a node that EXISTS can still not RESOLVE

**Task (2026-08-17, nexus #82, sibling of [[L-059]]).** Author `.. implements::` declarations for
`docs/theory/foundations/operator_algebra.rst` from an explorer-written spec, record the
no-implementer taxonomy, and repair four drift findings the exercise exposed. Landed **57**
directives over **32** equations, a new H1 recording the contract + the 8 un-declarable equations,
and repairs A–E. `-E -W` EXIT 0 / 0 W-E-C, unchanged from a freshly measured baseline of 0.

### 1. ⭐⭐ Count the spec's TABLE; its headline is a summary and summaries rot

The spec file's own headline read **"21 of 40 declarable, 19 NONE"**, and the dispatching brief
inherited it verbatim ("The 19 NONE equations…", with a kind breakdown of 4+1+1+2 = **8**, which
does not sum to 19). One `re.findall` over the table rows: **40 rows, 32 declarable, 8 NONE, 57
implementer slots over 55 distinct symbols.** The table was right and internally consistent — its
`§3` kind taxonomy names exactly the 8 — and only the headline was wrong.

⟹ **A spec is its table.** Before designing to a brief's counts, re-derive them from the artefact
in one command; a wrong headline propagates into the brief, into the section you write, and into
the return. Had I written "19 NONE" into the page it would have been a published falsehood with an
authoritative-looking provenance chain (explorer → main agent → me).

### 2. ⭐⭐ "All N node IDs resolve" ≠ "all N `:by:` targets bind" — the DIRECTIVE's resolver is narrower than the graph

The spec certified *"every dotted path existence-checked against `.nexus/graph.db` — 55 of 55 node
IDs resolve"*. True. Two of them still would not have bound.

`_node_id_for_target` (`sphinxcontrib/nexus/directives.py`) tries the literal string, then
`py:function:`, `py:method:`, `py:class:` — **and nothing else**. `Domain` / `Codomain` are
`TypeVar`s, so their nodes are `py:data:orpheus.numerics.operator.Domain`. A bare
`:by: orpheus.numerics.operator.Domain` logs *"target not found in graph — skipping"* and lands
nothing — which under `-W` is a red build, and without `-W` is a silent no-op.

Fix: the directive **accepts an already-prefixed node id** (`if target in graph: return target`),
so `:by: py:data:orpheus.numerics.operator.Domain` binds. Measured post-build: 57 directive edges,
0 skipped.

⟹ Pre-flight `:by:` paths **through the resolver's own prefix list**, not by asking "does a node
with this name exist". [[L-059]]'s graph pre-flight would have passed both. And write the reason
into the page — a future author copying the two `py:data:` lines needs to know why they differ.

### 3. ⭐⭐ A brief's "sharpest observation" is a HYPOTHESIS with a computable confusion matrix

Both the spec and the brief pressed one finding: *the page already labels its own two classes* —
every NONE row's `.. (vv-status rationale)` says "Mathematical identity" / "Definitional", while
every declarable row's names a verb, a value, or a file. The brief even pre-corrected one half of
it. It is still not a classifier, and the check is four greps:

| over the audited 40 | NONE (8) | declarable (32) |
|---|---|---|
| carries a rationale block at all | **6** | **22** |
| rationale contains *identity* | 5 of 6 | **11 of 22** |
| contains *"not a solver claim"* | 1 of 6 | **5 of 22** ← points the wrong way |
| cites a `tests/` file (the "declarable" signal) | **2 of 6**, incl. `operator-solve` | 13 of 22 |

A third of the page carries no rationale at all, and the word *identity* appears in half the
**declarable** rows. The cause is a real ambiguity, and naming it is the finding that survives:
an **identity between quantities** (`apply-solve-parallel-identity`) can have no carrier, while an
**identity between types** (`carrier-grid-operator-typing`, `harmonic-frame-is-galerkin`,
`product-solve-reroute`) is *exactly* a claim about a class declaration. Both are honestly called
identities.

⟹ Publish the measured split (`{identity, law, canonical-form} → NONE`;
`{typing-rule, definition} → look for a declaration site`) and publish the **refutation of the
keyword tell** beside it, because the next reader will otherwise re-derive the keyword heuristic
and ship it.

### 4. ⭐⭐ Before repairing a stale equation, ask whether the corpus already states it CORRECTLY

Finding A (`keff-as-integrated-rates` present-tense-false: `(n,2n)` in the numerator, no leakage)
was real and independently confirmed against `SNSolver.compute_keff`. But a 3-command corpus
census after the repair found `docs/theory/methods/sn/solver.rst` already carrying the shipped form
as `:eq:`sn-keff-update`` under `:ref:`sn-keff-estimator`` — with the divergence-telescoping
derivation, the leakage functional, and the wiring to the cross-engine gate. My repair had just
minted a **twin**.

Correct shape, applied: keep the equation (a labelled equation is an API and must not state a
falsehood) + an `.. important::` naming the SSOT, saying what *this* page's claim actually is (the
**typing** claim: both ends of the ratio are the same typed functional), and instructing that
future changes are edited there and mirrored here. Same census on finding C was *evidence the
repair was right*: `frame.rst`, `sn/slab_multigroup.rst`, `sn/cartesian_multid.rst` and the class
docstring **all already spelled it `Λ`** — `operator_algebra.rst` was the sole holdout writing `S`,
contradicting its own rejection note 900 lines up.

⟹ The census is cheap and it does two jobs: it stops the repair becoming a twin, and it tells you
whether you are fixing an outlier or inventing a convention. Run it **before** drafting, not after.
⚠ It also surfaced a symbol collision: the SSOT writes leakage `L`, this page writes
`L = Ω·∇` everywhere. Resolved by `L_{\rm leak}` **plus a note naming both spellings** — never
silently.

### 5. ⭐ My own uniqueness guards were substring bugs — twice, and labels are PREFIXES of each other

Two splice guards fired before any write (so cost nothing, per the assert-before-write discipline):

* `result.count(":label: operator-apply") == 1` — **fails**, because `:label: operator-apply` is a
  substring of `:label: operator-apply-transpose`.
* `result.count("Development history") == 1` — **fails**, because an `.. admonition:: Development
  history — G6.3 step 8.0` sits 1000 lines above the section.

⟹ A uniqueness guard over labels/titles compares **exact lines**
(`sum(1 for l in lines if l.strip() == …)`), never substrings. Eq-label families are built by
suffixing (`X`, `X-transpose`, `X-section`), so the prefix collision is the *normal* case here, not
an edge case. And diagnose a red guard as possibly the GUARD's error first — both of these were.

### 6. ⭐⭐ The plan-authoring quantifier clause applies to the prose YOU publish — I broke it twice in one section

Both caught by re-measuring my own sentences before the final build:

* *"…:eq:`scattering-carrier-grid`, :eq:`scattering-aniso-composite` and
  :eq:`reaction-rate-kinf-oracle` with three each"* reads as an enumeration of the
  multi-implementer equations. There are **15**, and **five** have three. Replaced with the count.
* *"60 on :eq:`operator-solve` alone, where every ``solve`` in the tree matches the label by name"*.
  Measured the 60 sources: only **5** are named `solve`; the rest are five `apply` methods, three
  `solve_fixed_source`, `is_invertible`, `is_adjointable`, `outer`, and whole classes. The match is
  on the **other** token. The corrected sentence is strictly more damning *and* true.

⟹ Every "each / every / all" in a doc paragraph is a universal over a set you can count in one
command. Count it. The measured version is almost always the better sentence.

### 7. The mechanism verified itself mid-session, by accident

A concurrent archivist's Sphinx build rebuilt `.nexus/graph.db` **between** my repair pass (which
carried 6 declarations) and my declaration pass. That snapshot therefore held both sides of the
comparison in one graph: the **3 declared** equations carried **0** inferred edges; the **37
undeclared** ones carried **771** (median 11, max 58). Post-pass: **57 directive, 0 inferred** on
the 32 declared; **166 inferred** remaining, every one on the 8 that cannot be declared.

⟹ When an accident hands you a clean before/after in one artefact, **say so in the page** — a
measurement that carries its own control is worth more than two dated ones. And note the corollary
the page now records: an equation that legitimately has NO implementer keeps its guesses forever,
because the stand-down is triggered by a declaration and **there is no way to declare an absence**.

### 8. Placement rule for a directive whose body RENDERS

`.. implements::` with a body emits a `<div class="docutils container">` — visible prose, no visual
marker. 57 of them dropped "immediately after the `.. math::`" would land mid-sentence wherever the
equation is followed by its own where-list. Rule adopted and stated in the splice script: **after
the math block, unless the next paragraph is a grammatical continuation of the equation's sentence
(`where …`, `so …`, `with …`, `and identically for …`) — then after that paragraph.** Encoded as a
per-label `skip` flag, previewed before writing. Every body opens `**Implemented by** …` so it
reads as an annotation rather than as body text.

### 9. Reported, not fixed

* The spec's headline count (§1) — flagged in the return, spec file not edited (not mine).
* `nexus` `_node_id_for_target` should try `py:data:` (§2) — a real tool gap; a bare TypeVar name
  is the natural thing an author writes.
* Eight further labelled equations on the page were outside the audit's scope
  (`carrier-grid-interchange-witness`, `tensor-product-axis-wise-composition`,
  `sum-of-tensor-products`, `octant-direct-sum-tensor-product`, and the four `eigen-*`).
  `[M]` all eight attract **zero** edges of either kind — unfinished, not wrong. Recorded in the
  page's own coverage subsection so the next pass finds them.

---

## L-061 — a mechanical port's WARNING COUNT is a non-representative sample of its DEFECT COUNT

**Task.** Clear 20 Sphinx warnings in `docs/theory/verification/error_catalog.rst` — a 5790-line
RST port of the 79-entry L0 error catalogue from `.claude/skills/vv-principles/error_catalog.md`,
done by a throwaway script. Brief: "It handled the bulk correctly… 20 warnings remain, and they
are genuine per-entry judgement calls." Result: 20 → 0, `EXIT=0`, 79 entries / 258 catchers intact,
and the xref gate's `error_catalog.rst` rows gone (75 → 71 dead sites tree-wide).

### The premise the measurement refuted

"The bulk is correct" was false, and one command showed it. **In RST there is no legal run of 3+
backticks outside a literal block**, so a run-length histogram is a total census:

```
RST: {1: 186, 2: 4010, 3: 678, 4: 152}      MD: {1: 4332, 2: 846, 3: 46}
```

**830 mangled delimiters on 339 lines, zero inside code blocks.** The 20 warnings were the ~2 %
of that class where the imbalance failed to cancel *within a paragraph*. Rendered HTML proved
it visible: `<code>`psi_right = fi[:, n, i, 0]``</code>`, `<code>`de8822d`</code>`.

⟹ **Before fixing warning #1 of a port, census the delimiter alphabet of the target language.**
The warning count measures where a *parser* choked; it does not measure where the *render* is wrong.

### One root cause, three surface families

The script's `` `x` `` → ``` ``x`` ``` regex was **LINE-LOCAL**, and a code span that WRAPS a line
defeats it three different ways:

| MD form | what the script did | symptom | count |
|---|---|---|---|
| `` ``x`` `` on one line | added a pair → 3–4 backtick run | mostly silent; stray backticks | 830 runs / 339 lines |
| `` `x` `` wrapping a line | converted ONE side → 1-vs-2 | **warns**, or cancels silently | 14 spans |
| `` `x` `` wrapping a line | converted NEITHER side | silent `<cite>` (italic, not code) | 16 spans |

`grep -c '<cite>' built.html` is the census for the third — **`default_role` is unset in this
project**, so every surviving single-backtick span renders *italic* instead of monospace. It is
the smoking gun for any Markdown→RST port, and it is invisible at every build severity.

### The port's own SOURCE is the oracle — it turns a 415-site blanket edit into a proof

Normalise, then check every restored literal's CONTENT against the Markdown:

```
inline literals: 2443   content not found verbatim in the .md: 5
   → 2 authored by the port itself (its new header note), 3 the `\|` class below
```

Same for prose, filtering the expected transformations (MD `#` headers → `:title:`, MD `|` tables
→ `list-table`, the replaced preamble): **3648 of 3653 MD prose lines ≥45 chars survive verbatim**;
the 5 exceptions were 1 intentional repoint, 3 artefacts of my own per-line de-markup on wrapped
`:math:` roles, and 1 correct `[x](url)` → `` `x <url>`_ `` conversion.

⟹ **A bulk delimiter edit is guarded by `src.replace('`','') == new.replace('`','')`** — proves
only backticks moved — plus an exact character-count delta and an unchanged line count. Cheap,
total, and it converts "risky mass edit" into "verified transformation". Do the write only after
the guards pass, so a failed assert leaves the tree untouched (no `git checkout` recovery needed).

### ⭐⭐ PROBE docutils, do not reason about it

I predicted the ERR-079 warning came from *emphasis containing an inline literal*. **Wrong.** One
`publish_doctree` probe with 6 one-line cases settled three entries at once:

| construct | docutils |
|---|---|
| `*"… ``lit`` …"*` (emph ⊃ literal) | **0 warnings** — and renders RAW backticks |
| `*"… **strong** …"*` (emph ⊃ strong) | **WARNS** ← the actual culprit |
| `key=``"zero"``` | **WARNS** (`=` forbidden before inline markup) |
| `key=\ ``"zero"``` | clean, identical render |
| `γ_-` in prose | `ERROR: Unknown target name: "γ"` |
| `γ\_-` | clean, identical render |
| `1. text:` then `   - a` (no blank line) | `Unexpected indentation` + `Block quote ends…` |
| `(Wave 2)\n+ the typed error` (no blank line above) | clean — `+` mid-paragraph is not a bullet |

A stub-directive/stub-role `publish_doctree` harness (register `error-entry` etc. as pass-throughs)
re-checks a 5790-line file in **under a second** vs a ~4-minute `-E` build. Build twice, iterate
in docutils.

### The Markdown discriminator for an indented block

CommonMark: **an indented code block cannot interrupt a paragraph.** So the fix differs by whether
a blank line precedes:

- **blank line before** ⟹ genuine code block ⟹ `.. code-block:: text` (mandatory when the body
  contains `*` — ERR-023's `w *= 2.0` was read as emphasis).
- **no blank line** ⟹ lazy paragraph continuation ⟹ blank lines around it → **block quote**, which
  is also what the port's other 14 indented blocks became, so it is the consistent choice.

### RST forbids inline markup after most characters — a port hits this constantly

Openers must follow whitespace or one of ``- : / ' " < ( [ {``. Markdown has no such rule, so the
port left 9 literals and **2 `:math:` roles** opening after `=`, `.`, `~`, `§`, `↔`, `*`. The
roles **were not rendering at all** — `~:math:`\mathcal{O}(h^{1.3})`` produced
`<cite>mathcal{O}(h^{1.3})</cite>` (role dead, LaTeX backslash silently eaten) and **no build at
any severity said so**. Fix is one character: `~\ :math:`…``. Tell = `<cite>` in built HTML.

### `\|` is right in prose and WRONG inside a literal

The script escaped 37 pipes; the MD had 0. In prose `\|` renders `|` (fine). Inside `` `` `` RST
does not process escapes, so it renders a **visible backslash** — measured in the HTML. Exactly 3
sites; the other 34 are harmless. Discriminate by context, don't blanket-revert.

### ⭐ Adjudicating dead `:mod:` in a HISTORICAL narrative — tense AND object survival

Four dead `orpheus.sn.spatial.*` targets, all inside ERR-026's "What Wave H Phase A/B added"
narrative. `git log --diff-filter=D` split them, and the answer is **not** uniform even though
three of four modules survived a pure `git mv`:

| site | sentence | module fate | named object fate | verdict |
|---|---|---|---|---|
| `boundary_face_flux` | "What Phase A added … Protocol (X)" | **DELETED** `3fd1302f` | Protocol retired | ``literal`` |
| `pole_angular_closure` | "What Phase B added … Protocol (X) with three strategies" | renamed `588f2429` | Protocol + 2 of 3 strategies retired (#248) | ``literal`` |
| `pole_angular_closure` | "**Documented in** A, X, and Y" | renamed | claim still true there | **repoint** |
| `diamond` | "**Citations updated in** A, B, X" | moved `5b6598f0` → `transport/spatial` | corrected BMC 2010 citation IS at `:51` | **repoint** |

⟹ **A surviving module does not license a repoint; a surviving CLAIM does.** Row 2 is the trap —
same file, same rename, opposite verdict from row 3, because the *sentence* names objects that no
longer exist there. Three corroborations, all free: the live tree's own prose spells the retired
names as ``literals`` (`sn/sweep/__init__.py:37`, `pole_angular_closure.py:93-95`); the SAME
catalogue entry already spelled the deleted path as a literal **130 lines below** the site; and a
list of three `:mod:` roles where two are live argues against making the third a literal.

### ⭐ A dead `:doc:` from a Markdown port is a PATH-FORM error, not a missing page

`:doc:`docs/theory/methods/sn/index`` — the page **exists**. MD authors write repo-root paths;
Sphinx wants a docname. Fix `/theory/methods/sn/index`, don't rewrite the prose. **Check the page
exists before concluding the reference is dead** — the brief and the warning both read as
"pointing at nothing".

### What I deliberately did NOT fix, and why

`[M]` **32 rendered `<strong>`/`<em>` elements contain raw `` `` ``** (86 raw pairs in page text
outside `<code>`) — Markdown bold/italic *containing* a code span, which RST cannot nest. Zero
warnings. Unlike the delimiter class (pure arithmetic, provably content-identical), each repair
must **choose where to break the emphasis run**, i.e. exactly the per-site judgement the brief
scoped to 20. Reported with the line list instead. Post-fix the class is *clean* `` `` `` pairs,
so a follow-up pass is mechanical.

⟹ **The scope line that held: fix what is provably content-identical, report what needs a
choice.** State the expansion loudly (830 + 30 + 3 + 2 sites vs a 20-warning brief) and give the
measurement that forced it.

---

## L-062 — a cross-reference inside HISTORY is a category error, and BOTH gates are blind to it

**Task (2026-08-18, branch `docs/err026-history-is-not-a-crossref`).** `docs/theory/verification/
error_catalog.rst` ERR-026 carried **29 python-domain roles, 20 unique, 15 of them dead**, all in
one 154-line block of Wave-E/Wave-H project archaeology. Ruling applied: **an ERR entry's body is
past-tense archaeology; a role is a present-tense claim that the symbol exists NOW at THAT path.
The two cannot be combined** — the catalogue exists *because* the code moved on, so a role inside a
historical narrative is guaranteed to go false. 29 → 13 roles, 0 unresolved; `-E -W` EXIT 0, W/E/C
count 0 = 0 baseline.

### 1. Why nothing caught it — and the ONE-LINE fix, measured

Two instruments, both silent, for two *different* reasons:

* **nexus dead-references** judged only 3 of 15 — the 10 bare roles (``:class:`SNStreamingOperator```)
  are "undecidable" and filtered out.
* **`tools/check_docstring_xrefs.py`** — my digest calls this "THE gate". `[M]` it reported the same
  **81 dead / 124 sites** before AND after. It is not that it judged them alive: **it DECLINED all
  15**, and 3 of those it could have decided.

`[M]` the mechanism, by direct call with `namespaces=()` (what an `.rst` page has — the project
carries zero `currentmodule`):

| target | role | `resolve()` | `judge()` |
|---|---|---|---|
| `orpheus.geometry.boundary.BoundaryOperator` | `class` | `(False,'missing')` | **DECLINED** |
| `orpheus.sn.geometry.SNMesh` | `class` | `(False,'missing')` | **DECLINED** |
| `tests.sn.test_snstreamingoperator.test_apply_…` | `func` | `(False,'missing')` | **DECLINED** |
| `orpheus.sn.spatial.pole_angular_closure` | **mod** | `(False,'missing')` | **DEAD** ✅ |

`judge()`'s last clause re-checks the target's HEAD *carrying the original role*:
`candidate_paths(head, namespaces, role)`. For a single-segment head like `orpheus` under a
non-`mod` role, `bare_module_guess` fires (`"." not in target and role != "mod" and not
hasattr(builtins, root)`), so the head is treated as relative → with no namespaces the candidate
tuple is `()` → `any(())` is False → DECLINED. `:mod:` is exempt from that guard, which is exactly
why only `:mod:` dead targets are ever reported on a page.

⟹ **on an `.rst` page the gate reports `:mod:` and NOTHING else.** One line fixes it — the head of a
dotted path *is* a module reference:

```python
head_role = "mod" if "." in target else role
if not any(lookup(c)[0] for c in candidate_paths(head, namespaces, head_role)):
```

`[M]` blast radius, patched COPY vs shipped, both run on a pristine `git archive HEAD` tree so
REPO_ROOT stays self-consistent: `docs/` goes **49 dead / 71 sites → 207 dead / 255 sites**. The
gate is blind to **158 dead targets across 184 sites in `docs/` alone**, every one a fully-qualified
`:class:`/`:func:`/`:meth:`/`:attr:` on a page. The patched copy flags exactly the 3 ERR-026 roles
pre-edit and zero post-edit — a positive control on the instrument (vv #17) that the count-diff
alone could never give.

⚠ **My first attempt to measure this in-process was itself broken** and read "0 dead" for BOTH arms
while a subprocess on the same tree read 49 — monkeypatching `g.judge` and calling `g.main()` twice
does not work (module-level memo/lru_cache state). Caught only because 0 contradicted a 14 I had
already measured on a subdirectory. Patch a COPY and run it as a subprocess.

### 2. The corpus's own prose is the corroborating oracle — count both spellings

Before de-roling, count how the SAME name is already spelled elsewhere. `[M]` inside the ERR-026
entry: `MorelMontryAngularSweep` **5 literals / 3 roles**, `SNMesh` **4 / 2**, `BoundaryFaceFlux`
**2 / 2**, `transport_operator_matvec` **2 / 1**, `LegacyTauSymmetricInterpolation` **2 / 3**. The
later phases (D, E, …) had already settled on literals; the roles were confined to the earlier
sections. So the entry was *already internally inconsistent* and de-roling made it consistent — that
census IS the evidence the ruling is right, and it is one command. Same trick found a sibling page
(`docs/theory/methods/sn/curvilinear_one_group.rst:2525`) already spelling the deleted test as
``a literal`` and calling it "Phase B's empirical test" — the exact phrasing to copy.

### 3. ⭐⭐ A SURVIVING CLASS does not license keeping the role — the surviving CLAIM does

The brief's LIVE table said keep `MorelMontryAngularSweep` as a role (it exists, at
`orpheus/sn/sweep/pole_angular_closure.py:1308`). I literalised all 3 anyway, because the criterion
is not *does the symbol resolve* but *does a working link mislead about what THIS sentence says* —
the same reasoning the brief itself used to literalise the live `SNMesh`. `[M]` the Phase-B site
describes the class "with starting condition ``ψ_{1/2}=0``", and the SAME entry's Phase-D section
records that as `ZeroSeed`, "Phase B's hardcoded `psi_half_left = 0`", replaced by
`psi_half_seed: … = field(default_factory=CarlsonInwardSweep)`. A link from that sentence lands on a
class whose default contradicts it.

⟹ the rule, stated in the page so it is checkable: **a name is a ``literal`` whenever the sentence
around it describes the code as it then was; a role is used only where the sentence is a
present-tense claim about something that exists now.** That single sentence adjudicates all 29 —
including why the five `:mod:` roles STAY (their sentences are "Documented in X" / "Citations
updated in X", present-tense claims I verified: `[M]` Bailey-Morel-Chang 2010 present 9× in
`reduced_operator.py`, 2× in `transport/spatial/diamond.py`, 11× in `sn/sweep/pole_angular_closure.py`).

**Nothing is lost by literalising, because the live pointers move to ONE place** where their tense
is present — a head-of-block `.. note::` that declares the convention AND says where the objects
went. That is the brief's own "live pointers belong in the status/catcher fields", realised.

### 4. Two brief classifications refuted, both by the same probe error class

* **`orpheus.derivations.continuous.sood_registry` is LIVE**, not "file missing" — it is a
  **package** (`sood_registry/__init__.py` + `la13511.py` + …), imports clean. A `.py`-only
  existence check misses a package. → kept as a role (6 live targets, not 5).
* **`SNMesh.pole_angular_closure` is LIVE**, not a dead attr — set on the INSTANCE at
  `orpheus/sn/mesh/augmented_mesh.py:399` (`self.pole_angular_closure: PoleAngularClosureBase =
  closure_cls(self)`). My own AST index missed it for the same reason a `hasattr(Cls, …)` probe does
  (L-053c). It still became a literal, but for the *sentence-tense* reason, not a dead-target reason —
  and getting the reason right is what stops the next reader "repairing" it back.
* Third, smaller: `orpheus.geometry.boundary` is a live **package**; it is the CLASS `BoundaryOperator`
  that is gone — and a live homonym exists at `orpheus/numerics/operator.py:437`
  (a `_BlockRoleMeta` marker, unrelated). Repointing would have been a false attribution (L-017).

⟹ **before calling a dotted target dead, decide WHICH segment died** — package, module, class, or
attribute. The four have different repairs and only one of them is "de-role".

### 5. The same category error one register DOWN: raw file paths

The ruling fixes roles. It does not touch the *other* present-tense claim a history block makes:
a ``tests/…/foo.py`` **path**. `[M]` in the ERR-026 entry, **14 of 14** distinct `tests/*.py` paths
no longer exist (`tests/sn/spatial/` → `tests/sn/sweep/`, `tests/sn/l1_analytical/` →
`tests/sn/verification/…`). Catalogue-wide: **40 of 100** distinct raw file paths written as
literals are gone — 31 of 72 `tests/`, 9 of 24 `orpheus/`. A raw path warns at no severity, is
invisible to the xref gate (which judges roles), and to nexus (which judges targets).

⟹ the note's third sentence is the prophylactic that matters most: ***which* tests catch ERR-NNN is
never prose — it is the `@pytest.mark.catches("ERR-NNN")` marker set**, `nexus errors` /
`context('vv:error:ERR-NNN')`. Write that once at the head of a history block and the whole class
stops being minted.

### 6. Mechanics

* **Guarded splice, all asserts before the write**: per-replacement counts; an exact
  `len(out) == len(src) + Σ n·(len(new)−len(old))` arithmetic delta; unchanged line count for the
  swap step; the final role list compared to an explicit expected list; `not re.search(r"`{3,}")`;
  `.. error-entry::` count unchanged at 79; and the decisive one — **`src[:i] == out[:k]` and
  `src[j:] == out[m:]` around the entry's own boundaries**, which proves byte-identity of the other
  78 entries in one line.
* **Roles resolve ≠ roles link.** `[M]` none of the 5 classes in my note has an `id=` anchor
  ANYWHERE in the fresh build, so all 5 render plain text — as do their 16–29 sibling sites each
  elsewhere in the corpus. Keep the role anyway: it is the corpus convention, it becomes a link the
  moment the module is surfaced, and — the real argument — **a role is machine-checked by the xref
  gate and a literal is unchecked forever.**
* **A role→literal sweep leaves ONE ragged paragraph per shrunken run.** `[M]` 25 sub-55-char lines
  in the edited region, 24 pre-existing; exactly one paragraph got 3 short lines in a row. Re-wrap
  that one (guard: `new.replace(" ","").replace("\n","") == s.replace(...)`), leave the rest — a
  line-local diff where every changed line shows exactly one swap is what makes the review cheap.
* ⚠ **I broke my own build-sequencing rule twice** (L-054): launched the verification build, then
  found a re-wrap, then found an over-claim, then found a self-inconsistency — four builds. Each
  find was correct and cheap in isolation; the fix is to run the *self-consistency* pass on new
  prose (does my own declared rule hold for every name I wrote?) BEFORE the first build, not after.
* ⚠ **Verify a successor claim against the retiring COMMIT BODY, not the successor's existence.**
  I first wrote that `solution_to_angular_flux*` / `transport_operator_matvec*` "were absorbed into
  the SN operator algebra". `[M]` `4a53737e` says the codec family "became orphan in production"
  after the bare-ndarray contract collapsed at every leaf, and `975edc51` deleted the matvec helpers
  as "without a remaining call site" — they were **retired outright, with no successor**.
  `SNStreamingOperator` really was re-layered (`400ca33d`: `SNSolver.L` → `StreamingOperator` +
  collision multiplier). Same paragraph, two different fates; "absorbed" was true of one and false
  of two.

---

## L-063 — An ONTOLOGY OVERTURN: rewriting the page whose thesis was refuted

**Task (2026-08-19, CS3 step 5, branch `refactor/cone-field-algebra`).** The code carve
had landed (4 commits): flux moved from an *affine space* `𝔸` over a difference space `V`
to the *positive cone* `K ⊂ V` of an ordered vector space. `FluxRole` and the whole
`transport/displacements/` package (8 modules, 7 leaves) were deleted. My job: make the
corpus teach the cone, with the affine era kept as dated history.

### (a) ⭐⭐ A DOC-SIDE OVERTURN IS NOT A RETIREMENT SWEEP — the unit is the ARGUMENT, and
### the load-bearing edit is re-deriving arguments whose CONCLUSION survives.

The retirement grep (32 dead Python-domain refs) was the *easy* half and it finished in one
pass. The hard half had no dead symbol in it at all: `operator_algebra.rst` carried a
**five-obstruction proof** that `Carrier[Representation, Role]` is structurally impossible,
and its obstruction **(a)** read *"the Flux role must make `flux + flux` **raise** while the
Source role must make `source + source` **succeed**"*. That premise is now false — and the
**conclusion is still true**. A retirement sweep either deletes the obstruction (destroying
a correct proof) or leaves it (a false premise under a true theorem). Neither is right.
⟹ **re-derive the argument from what survives, keep the conclusion, tombstone the example.**
`[M]` the live tree hands you the replacement: `AngularFlux.__dict__` has **no `__add__`/
`__sub__`** (MRO `AngularField → BulkField → Field → ABC`) while `AngularSourceSink` **does**
(the iso→per-ordinate containment injection), so the axis that "changes the arithmetic
interface" **inverted** — Source, not Flux. Obstruction (a) survives verbatim in force with
a different worked example plus a second leg the old text never needed (class identity *is*
units identity, and erasure would collapse `type(self) is type(other)` across every role).

⚠ The same shape recurred five times on one page: the role-axis asymmetry section, the
`(Moment, Displacement)` contrast, the fibration note, the conclusion sentence, the
vv-kind table. **Grep the retired symbol to FIND the sites; then read the enclosing ARGUMENT
to decide the edit.** A per-site symbol swap would have shipped five false premises.

### (b) ⭐⭐ RETIRE the eq-LABEL when its NAME encodes the refuted concept; KEEP it when only
### its ADJECTIVE is stale — and the discriminator is the label's BODY, not its name.

Four `:eq:`-cited labels. `[M]` 0 `@pytest.mark.verifies` markers on any (grep
`orpheus/ tests/`), 3 external `:eq:` citers, all in prose I was rewriting anyway.

| label | body states | fate |
|---|---|---|
| `affine-torsor-algebra` | the RETIRED claim (4 torsor axioms) | **RETIRED** → new `flux-vector-algebra`; 2 citers repointed |
| `affine-contraction-ratio` | ρ = ‖Δψⁱ‖/‖Δψⁱ⁻¹‖ — still TRUE, still shipped | **RENAMED** `iterate-contraction-ratio` |
| `affine-true-error` | ‖Δψ‖/(1−ρ) — still TRUE, still shipped | **RENAMED** `iterate-true-error` |
| `affine-typed-residual-eq` | r = (L+C−S−B)ψ − q — untouched by the overturn | **KEPT + annotated** |

⟹ The residual one is the interesting call. Its `affine-` prefix is a **historical artefact
of the page's former title**, not a claim: the residual role was never affine. Its *section*
anchor `affine-typed-residual` has **8 cross-doc `:ref:` citers** (boundary_conditions ×6,
coupled_block_operator ×1, self), and a cross-doc dangling `:ref:` renders plain text with
**no warning at any severity**. Renaming buys cosmetics and risks a silent break.
⟹ **KEEP, and put a `.. note::` at the anchor saying the prefix is stale and why** — so the
next reader greps `affine`, lands there, and is told in one paragraph rather than
re-litigating it. A stale NAME is not a false CLAIM; only a body can be false.

⭐ And the retired equation still had to be *shown* (the history section needs it). Solution:
display it as an **UNLABELLED `.. math::`** with one parenthetical saying why —
*"a labelled equation is an API; these lines state a retired claim, so they must not be
citable."* An unlabelled block cannot become an `:eq:` API by accident.

### (c) ⭐⭐ The sentinel bookkeeping is a SAME-FILE constraint you can check in 1 s — and
### the matrix is GENERATED, so never hand-edit it.

`tests/_harness/audit.py:405` — `.. vv-status: <label> documented` must name a `:label:`
in **the same file**, and `documented` is the only legal status. Renaming a label without
its sentinel is a hard audit violation. Run the scanner directly (sub-second, no pytest):
`from tests._harness.audit import _scan_theory_equations as scan; scan(Path("docs/theory"))`
→ `.violations` / `.documented`. `[M]` 0 violations; population **539 → 540** (4 old → 5 new;
I added `positive-cone-definition` because the cone is the page's new subject).
`docs/theory/verification/matrix.rst` regenerates at `builder-inited` from `conf.py`'s
`_GENERATORS`, so the sentinel list fixes itself — report the post-regen number, never edit.

### (d) ⭐⭐ REPRODUCE the witness, and the reproduction may REFUTE the gate's own prose.

The ruling's decisive measurement is *"DD does not preserve K, so a ψ≥0 type would refuse
production output"*. The gate `tests/sn/solve/test_cone_membership_witness.py` freezes
`min ψ = −6.399383e-01`. I reproduced it through the public entry — exact to the digit —
**and the gate's docstring is wrong about its own fixture**: it says the pair differs in
*"ONE parameter (`nx`) … half the optical cell size"*, but `_solve(nx=2, width=20)` and
`_solve(nx=4, width=40)` both have `Δx = 10`, i.e. `Δx·Σ_t = 100` **identical in both legs**.
⟹ **The argument is STRONGER than its prose** (holding the cell size fixed kills the
"different discretization scale" explanation outright), so the fix is to publish the correct
framing, not to weaken the claim. I also ran two scans nobody asked for and they turned one
frozen number into the mechanism: the **cell-SIZE** scan reproduces the textbook DD
positivity limit exactly (`Δx·Σ_t = 1` in K at `+5.8e-2`; `= 2` already out at `−8.7e-1`),
and the **cell-COUNT** scan at fixed `Δx·Σ_t = 100` shows `nx=2` is the *only* in-cone row
(`nx=3,4,5,6` → `−6.42e-1, −6.40e-1, −6.38e-1, −6.36e-1`). Two tables, ~90 s, and the page
teaches a mechanism instead of quoting a constant.
⚠ I may not edit `tests/` — so the docstring correction is REPORTED, not applied.

### (e) ⭐ Two SKILL files the brief did not scope carried the retired ontology as a POSITIVE
### precedent — and my own repair would have imported the falsehood via its cross-reference.

Brief scope was "the one skill file" (`coding-elegance` #18). `[M]` grep `.claude/skills/`:
`cross-domain-frames/reference.md` (the A.1 frame row + §192/§201 fix-suggestions) and
`numerical-bug-signatures/SKILL.md` §479/§488 also cite `FluxDisplacement` / "flux states
are an affine space" as live. And #18's *corrected* text points readers at A.1 for the frame
— i.e. **the repair cites a stale page** (`coding-standards`' "a cross-reference is a
load-bearing dependency"). ⟹ I flagged the staleness **inline at the pointer** (*"A.1's
frame is sound; its ORPHEUS worked example is NOT"*) so the repair cannot import the
falsehood, and reported both files as owed follow-ups rather than editing out of scope.

⭐ The **reversal** of an anti-pattern is more valuable than its statement, so #18 was
rewritten to lead with what survives (*"NEVER STRAND the convergence data — give it a home
on the object that knows 'previous', which is the ITERATION"*) and to carry the falsified
version verbatim beneath it, plus the checkable test the reversal yields: **(a) is there a
canonical zero?** (distinguished by the domain, not chosen) **(b) is superposition
physical?** Two yeses ⟹ vector space, one type, diagnostics on the record.

### (f) ⭐⭐ The two-sided rule the whole overturn distils to — worth quoting into any page
### that invokes "make illegal states unrepresentable".

Mint the invariant **iff** (1) every value the type admits is legal **AND** (2) every legal
value is admitted. **Half 2 is the one that gets skipped, because it is a claim about the
PRODUCERS, not about the concept.** When it fails the invariant does not prevent a bug — it
**refuses correct output**, and the pressure is then to weaken it, silence it, or route
around the type. Here half 2 failed twice independently: algebraically (K is not closed
under difference or negative scaling, and increments/errors/Krylov directions all live
outside it) and numerically (DD ships negative flux).

### (g) The mechanical residue, all measured

- **Dead Python-domain refs: 32, not the briefed 23.** The brief (and the step-3 commit
  body) counted `orpheus.transport.displacements.*` only; `orpheus.transport.fields._flux_role.*`
  is another **9**. Grep BOTH retired module paths, not the one the commit message names.
- ⚠ **`orpheus/transport/displacements/` still IMPORTS** — an untracked `__pycache__` leaves
  a PEP-420 namespace package (`__file__ is None`, 0 members), so a naive
  `importlib.import_module` probe reports it LIVE (L-052's known false negative). Probe a
  SUBMODULE, or check `__file__ is None`.
- **`tools/check_docstring_xrefs.py`: HEAD 1 dead → working tree 0.** The one it saw was the
  `:mod:` I fixed. It is BLIND to the other 31 (L-062's unlanded `head_role` bug) — my own
  import probe over **727** orpheus-rooted roles across the 8 edited pages is the real gate.
- **`fuel_behaviour.rst:303` "Displacement-Based Constraint" is a MECHANICAL displacement**
  (fuel pellet) — the overloaded-word false positive the brief's grep list contained. Triage
  by MEANING before touching.
- **Build: `-E -W --keep-going` EXIT=0, WARNING/ERROR/CRITICAL set byte-identical to the
  pre-edit `-E` baseline (both empty), 0 `SyntaxWarning`.** 11 anchors + 5 equations + 26
  live code links rendered on the new page.

### (h) ⭐ The changelog contract BLOCKED the obvious home — and the page-local one was right

`docs/theory/methods/sn/history.rst` states *"a new entry lands with its merge hash or not at
all"*, and CS3 is unmerged. `operator_algebra.rst`'s history has the escape hatch
(*"entries marked (in development) live on an unmerged feature branch"*). ⟹ I gave
`field_algebra.rst` its **own** Development history following that convention verbatim, put
a short row on `operator_algebra.rst`'s (its Role axis genuinely moved), and left
`history.rst` alone except to tombstone its 2026-06 row's *affine half* while explicitly
preserving its *typed-residual half* — one row, two halves, opposite fates.

---

## L-064 — Seeding a NEW page from a design-dialogue record: the dialectic is the deliverable, and a retrodiction table is a plan-§2 trap in the corpus

**Task.** CS1 step 5, campaign 1 ("operators born bound"), branch
`feature/cs1-energy-space`, 2026-08-20. Write `docs/theory/foundations/spaces.rst`
from scratch as `field_algebra.rst`'s sibling (that page owns the ELEMENT algebra;
this one owns the SPACES), register it, add one `automodule`, and micro-edit three
`cone_violations` sites. Source of record: `.claude/plans/cs1_energy_space_design.md`
§A/§B/§F + Appendix A (a preserved user⇄agent design dialogue, rounds 2–6).
Result: 1158-line new page, `-E -W` EXIT=0, warning set unchanged (0 both sides),
141 role targets 0 dead, `DEAD TARGETS: 0`, vv violations 0, sentinels 540 → 541.

### 1. ⭐⭐ A DIALECTICAL SEED PAGE is its own doc shape — and it is NOT the 9-step close-out arc

The close-out arc is for a CLOSED investigation whose answer is *"this cannot
work"*. A **seed page** is the opposite event: a design dialogue CONVERGED, the
first slice shipped, and the page exists so the next phase builds on the reasoning
rather than re-deriving it. The user's own instruction was the giveaway — *"make
sure we don't lose it so the archivist can write it later — it was hard to steer
until we got it out."*

The shape that worked, in order:

1. **Key Facts** — including the doctrine's *one-line discriminator tests* verbatim.
   A doctrine that cannot be applied in one sentence has not been articulated.
2. **The taxonomy** (what the new type IS — four slots, a `list-table`).
3. **The theorem** (why one instance's measure is FORCED, not chosen).
4. **The doctrine, DIALECTICALLY** — the question, then version 1 REFUTED with its
   refuting question, then version 2 REFUTED with its refuting question, then the
   standing doctrine, then the retrodictions.
5. **Fences** (what is NOT built, per phase).
6. **Development history**.

⭐ **The refutations must be typographically first-class.** I used
`.. admonition:: ⛔ The refuting question — …` / `:class: error`, one per refuted
version, titled with the QUESTION rather than with the verdict. Rationale: a
reader skimming for the answer stops at the standing doctrine; a reader who meets
only the final statement re-derives version 1 within a week, because **both refuted
versions are almost right**. Version 1 (compactness) reproduced both prior rules
and failed only on energy; version 2 was a one-word patch. The refuting question is
the transferable content — *"where does energy sit?"* and *"what is the measure of
(0,∞)?"* — not the verdict.

⭐ **Name what the doctrine does to the tension it settled.** The two prior rules
(the report's §I.9 retract vs §I.11 quotient) were BOTH right, about different
clauses; what was missing was a second FORK nobody had stated. Writing *"it does
not pick a winner"* explicitly is what stops the next reader hunting for the loser.

### 2. ⭐⭐ A RETRODICTION TABLE is `plan-authoring` §2's aspirational-row trap, moved into the corpus

I wrote *"Every entry below is a layout the tree already ships"* over six rows —
and row 6 was the **buckling member**, which is a prediction of campaign 2. Caught
by my own "count every universal you publish" habit, before the build.

The defect is exactly the plan rule: a table headed by a property of the tree reads
ENTIRELY as a survey of what IS, so one aspirational row is indistinguishable from
the observations. In a PLAN that costs a session; in the CORPUS it is worse — the
table is the doctrine's evidence, and a reader who later discovers one row is
unbuilt has grounds to discount all six.

⟹ **A published confirmation table gets a STATUS column, in the row.** Not prose
above or below it. Mine became `[M] **ships**` × 5 and
`⛔ **NOT built** — a prediction (campaign 2)`, plus a `⚠` lead-in naming the row.
And the honest re-heading is *"rows the doctrine was NOT built from"* — which is
the actual epistemic claim (a retrodiction is a prediction of something not used to
build the theory), not *"layouts the tree ships"*.

### 3. ⭐ Cross-referencing an SSOT: name the REGISTER your page owns, not just the fact

The brief flagged that `energy-condensation-counting-measure` already existed
(`frame.rst`, inside `sn-energy-condensation`) and forbade a twin. L-060's rule is
"cite the SSOT + say which claim THIS page owns". The sharpening this task added:
**the two treatments are the same fact in two REGISTERS, and naming the register is
what makes the second treatment not a twin.**

- `frame.rst` owns it in the **measure register**: `w_g = 1`, not `w_g = Δu_g`,
  derived from Hébert Eq. 3.96/3.97 (distribution vs function averaging), gated by
  rate preservation.
- `spaces.rst` owns it in the **metric register**: `G_E = I`, hence `V ≅ V*`
  isometrically along energy, hence the adjoint there is the plain transpose —
  and the fact that `EnergyAxis` now REFUSES weights at construction.

I derived it a third way (covariant group-INTEGRALS × contravariant group-AVERAGES)
as an *unlabelled* `.. math::` chain, cited the SSOT's label for the claim, and
opened with `.. important:: Single source of truth … Edited there, consumed here.`
**Net new labels on a 1158-line page: ONE** (`spaces-axis-product`, the space =
⊗ axes / shape = concatenation / metric = ⊗ of factor measures identity), sentinelled
`documented` with a rationale naming the foundation battery.

### 4. ⚠ Two agreeing sources can both be wrong about a DATE — git is the arbiter

The brief said the byte gate held *"dated 2026-08-21"*, and
`tests/sn/architecture/test_monomorphic_leaves.py:668` independently says
*"CS1 step 3b (2026-08-21)"*. `[M]` `git log --date=short` puts every CS1 commit
(`1afff47b` … `6da1b23c`) on **2026-08-20**, and the session date was 2026-08-20 —
i.e. both surfaces carry a FUTURE date. Two independent agreeing sources felt like
corroboration; they were one mis-dating copied forward. Publish the git date.

### 5. ⭐ The "only producer" claim, and the collision the grep that proves it exposes

To publish *"every harmonic-moment space is still legacy"* I needed the closure
argument, not the universal: `of_axes` is the only ROOT producer of an `axes`
record (`*` and `dual` merely THREAD one, so both need an axis-built ancestor).
Publishing the derivation instead of the universal is strictly better — it stays
true as the tree grows.

⭐ The grep that established it (`grep -rn "axes=" orpheus/`) surfaced a live
gotcha worth its own note: **`mm.axes` is the GEOMETRIC tuple (`(AxisMesh,)`) and
`mm.bulk_space.axes` is the SPACE-FACTOR tuple (`(EnergyAxis, Axis)`) — the same
attribute NAME on one object, neither derived from the other.** Production already
imports around it (`from orpheus.numerics.axis import Axis as SpaceFactorAxis`).
Three senses of "axis" live in this corpus (space-factor / geometric / symmetry);
the page opens with a note enumerating them and pointing at the rename issue (#393).

### 6. The `V` collision, and the sibling page that had already solved it

A NEW page assembled from multiple sources is the prime site for a within-document
symbol collision (L-011/L-034) — and here it was `V` (the function space, the
page's subject) vs `V` (cell volume, the weight that makes clause 1 work). They
meet in EVERY clause-1 sentence. `field_algebra.rst` had already ruled on it
(`V_cell` written out in full), so the fix was to **adopt the sibling's ruling and
say so**, not to invent one — 5 sites swept programmatically with a `count == 1`
assert per substitution, plus a note citing the sibling for the same reason.

### 7. Numerical evidence a doc page can carry when nothing converges

A terminology/architecture page has no convergence table (the routine "weakest
dimension"). Three measurements carried real weight here and cost ~90 s:

- **The identity bridge, demonstrated**: quotient point (`volumes=[1.0]`) and a
  one-cell slab (`volumes=[2.0]`) mint `energy(2,)*spatial(1,)#<digest A|B>` — same
  shape `(2,1)`, different digests, `==` is `False`. ⭐ Published the name FORM, not
  the hex: a digest in prose is a stale-number risk with no SSOT but the code.
- **The quotient weight is CONSUMED, not decorative**: same flux `φ=(1,1)`, same
  mixture, production rate `0.225` (V=1) vs `0.450` (V=2) through
  `IntegratedReactionRate.evaluate → mesh.volume_measure`. That turns a doctrinal
  claim ("the pairing consumes it") into an arithmetic one.
- **The axis laws**, each run: ones→None canonicalization (`==` and `hash` agree),
  `-0.0`≡`+0.0` bytes, non-finite refused, signed weights legal, `synthetic !=
  from_grid`, `EnergyAxis != Axis`, weighted `EnergyAxis` refused.

### 8. Findings reported, not fixed (scope discipline held)

- `docs/theory/foundations/infinite_medium.rst:1153` and its `:1243-1244` code
  block still teach `basis_shape=(ng, 1)` and a bare `from_solver_data(mat_xs=…)`
  as the current homogeneous spelling. `[M]` I RAN both the doc's four lines and
  production's: **both execute and give `k_inf = 1.8750000000000009`** — the
  keyword survives as an optional override, so this is *stale description*, not a
  broken block. That measurement is what turned a "fix on sight" impulse into a
  correct FLAG: a behavioral rewrite of another chapter's worked example is exactly
  the in-passing rewrite my own rule forbids.
- `orpheus.numerics.space` and `orpheus.transport.mesh.material_mesh` are
  `automodule`'d NOWHERE, and `docs/api/homogeneous.rst` uses `:noindex:` — so
  `FunctionSpace.of_axes`, `has_coordinate_cone`, `MaterialMesh.bulk_space` and
  `solve_homogeneous_infinite` render PLAIN TEXT (measured: 0 `href`s). Page
  convention, not a defect; surfacing them is its own task.

### 9. Gate sequencing (what it cost)

I built **five** times where two would do — every extra build bought by an edit
made after launching a build. The self-consistency pass (universals, symbol
collisions, aspirational rows) must be run to EXHAUSTION before the first
verification build, not interleaved with it. What DID work: a single python
self-check script asserting short-underlines / ladder order / per-table column
consistency / widths-sum / label+anchor uniqueness / role import-resolution /
`:eq:`+`:ref:`+`:doc:` corpus resolution — re-runnable in ~2 s, and it caught every
structural defect before any build.

`docs/theory/verification/matrix.rst` regenerated on the `-E` build and shows the
CS1 campaign's uncommitted TEST work (9868 → 9920 collected; new rows for
`numerics/test_axis`, `numerics/test_space_of_axes`,
`homogeneous/test_operator_spaces`, `homogeneous/test_byte_stability`;
`architecture/test_monomorphic_leaves` 102 → 98, corroborating the four deleted
strict-xfails) alongside my own `540 → 541` sentinel row. Legitimate by-product —
report it, never revert it.

---

## L-065 — Resolving an N-WAY CONTRADICTION: when the corpus states one object three
## incompatible ways, the disagreement IS the diagnosis (a hidden parameter is unnamed)

**Task, 2026-08-23.** Record F-0 of `.claude/plans/frame_square_recarve.md` — a landed
metric repair — across `foundations/spherical_harmonics.rst`, `foundations/frame.rst`,
`conventions/normalization.rst`, `foundations/operator_algebra.rst`,
`verification/error_catalog.rst`. Branch `feature/cs1-energy-space`. Code/tests already
landed and OFF LIMITS (read-only). 6 files, +1246/−81. `-E -W` EXIT=0, warning set
unchanged (0 ↔ 0); vv violations 0, sentinels 541 → 545; `DEAD TARGETS 0`; my own
import probe over the 16 roles on added lines: 0 dead.

### 1. ⭐⭐ The brief handed me THREE published statements of "the adjoint of M". All
### three were internally consistent. That is not three bugs — it is ONE missing parameter.

The corpus said, in three places:

| site | claim |
|---|---|
| `frame.rst` eq `galerkin-strict-adjoint-vs-reconstruction` | "the strict adjoint is the NAKED synthesis (no factor)" |
| `spherical_harmonics.rst` eq `hilbert-adjoint-equals-metric-times-S0` | `Π* = g_C·S₀`, `g_C = 4π/(2ℓ+1)` |
| `normalization.rst` prefactor-ledger row | "The adjoint Π*: carries **Nothing** — the naked reconstruction" |

Worse, `frame.rst`'s **equation** and the **prose four lines below it** disagreed with
each other *inside one admonition* (naked vs `g_C·S₀`) and had shipped that way for
months, warning-free.

⟹ **The reflex "find which one is right and fix the other two" is WRONG here.** An
adjoint is defined by a PAIR of inner products; every one of the three was the correct
adjoint under a *different* coefficient metric (Euclidean / continuum Gram / Parseval
inverse), and none of them named its metric. The repair is therefore **not** N local
corrections — it is *naming the parameter*, once, in a table at the point of definition:

    | Coefficient metric | Where it lives | The Π* it induces |
    | Euclidean          | the bare-transpose reading | S₀ |
    | continuum g_C      | `SphericalHarmonicSpace.from_L` | g_C·S₀   ⛔ pre-F-0 |
    | Parseval G⁻¹       | `FrameBase.basis_space`         | S₀∘G⁻¹ = R/W  shipped |

Each of the three sites then becomes a POINTER into one row, and none can rot
independently again.

⭐ **The generalisable tell, and it costs nothing to look for: two published statements
of the same object that disagree, where each is defended by a correct-looking argument,
means a parameter both arguments quietly fixed differently.** Do not adjudicate; find the
parameter. (Same shape as `vv-principles` #24(b): when a ranking is explained by a
mechanism nobody was debating, the debate was mis-framed.)

### 2. ⭐⭐ A DESIGN PROBE goes stale against the repair it motivated — and it does so
### SILENTLY, because it still runs and still prints plausible numbers.

The plan cited `scratch/probe_f1_parseval.py` for its headline `Parseval ratio 118.7`.
The probe reads `G_stored = frame.test_space.inner_product_weights` — which pre-repair
was the continuum metric and **post-repair IS the Parseval metric**. Run today it prints
`ratio = 1.000` in the row labelled *stored*. It has not broken; it has silently changed
what it measures.

⟹ Three consequences, all mandatory:
- **Never cite a pre-repair probe path as the reproducer of a post-repair page.** Publish
  the **CONSTRUCTION** instead — I wrote the exact 6-line recipe (build the frame, draw
  `default_rng(1234)` unmasked, synthesize, analyse, read five residuals off five named
  attributes) so the table regenerates from the page with no file dependency. (L-048's
  "describe a probe, never cite an ephemeral path", sharpened: the reason is not only that
  `scratch/` is untracked — it is that the probe's SEMANTICS moved.)
- **Re-measure every number against the LIVE tree with your OWN probe.** Mine
  (`probe_f0_doc.py`, ~60 lines) reproduced the theorem/Parseval/closure table for all
  6 sphere families and refuted two inherited figures (below).
- **Report the probe's staleness upward** — the main agent owns `scratch/`.

### 3. ⭐⭐ A SEED-DEPENDENT number must be published as its BOUND, never its value —
### and then find the exact quantity hiding behind it.

Plan: `Parseval ratio 118.7`. Mine, same rule (LS4, L=1): **81.4**. Both correct; the
ratio is a *moment-energy-weighted average* of the per-ℓ factors `(4π/(2ℓ+1))²`, so it
moves with the coefficient draw. Publishing either bare number invites a future session
to "fail to reproduce" a true result.

⟹ Publish the **draw-independent** statement: *it lies between the extreme factors
PRESENT AT THAT L* — `[17.5, 157.9]` at L=1, `[6.3, 157.9]` at L=2 — *and can therefore
never be 1*. ⚠ note the quantifier: I first wrote `[6.3, 157.9]` for a sentence covering
both L values, which is false at L=1 (the ℓ=2 factor does not exist there). A bound is a
universal; `plan-authoring` §2 applies to it.

⭐ **Then look one level down for the number that IS exact.** The *ratio of the two
adjoints* on a single-ℓ unit input is `(4π/(2ℓ+1))²` — measured to `≤2.8e-16` relative at
every ℓ, seed-free, and strictly more useful than the average it produces. A
draw-dependent aggregate almost always has an exact per-mode parent; find it and publish
that.

### 4. ⭐⭐ `.. no-implementation::` has a class the taxonomy was missing:
### AN IDENTITY BETWEEN TWO QUANTITIES THAT ARE EACH COMPUTED.

L-059/L-060 give the classes `{identity, law, canonical-form} → NONE` and
`{typing-rule, definition} → look for a carrier`. This task produced a sharper case, and
it is the one most likely to be mis-declared because it *looks* declarable:

`φ = Mψ = Gc`. Both sides ship — `Mψ` is `_FrameAnalysis.apply`, `Gc` is
`FrameBase.discrete_gram` — and the **identity itself is evaluated nowhere**. Same for
`d_ℓ·G_ℓ = W`: both factors ship, their product is never formed (that is the POINT — the
identity is what lets the kernel carry one `1/W` scalar instead of a per-ℓ table).

⟹ **Declaring either side asserts that one of them IS the identity.** Use
`no-implementation :kind: identity` and say in the block *which* symbol computes *which
side*, plus what the suite measures instead (here: the identity's CONSEQUENCE, the
isometry). `[M]` this stood 17 + 16 name-token guesses down to 0 on two labels; a third
(`galerkin-strict-adjoint-vs-reconstruction`, a *contrast* between two separately-declared
faces) went 2 → 0 and one of its two guesses was `solve_sn_adjoint`, an SN solver entry
point that never touches a spherical-harmonic face.

⭐ The mirror, same session: `sh-space-metric` had NO declaration and 3 genuine
implementers (`metric_per_ell` → `_padded_metric_tensor` → `from_L`). The re-derivation
of a sibling equation's `implements::` set is where you find them: `metric_per_ell` LEFT
`hilbert-adjoint-…` (it is the continuum Gram, no longer that equation's factor) and had
to LAND somewhere — an equation losing an implementer is a prompt to ask which equation
gains it.

### 5. The `documented` sentinel marks the KIND, not the coverage — sibling consistency
### decides, and the matrix legitimately lists a label in TWO places.

`hilbert-adjoint-equals-metric-times-S0` is verifies-covered (`[M]` 2 → **9** tests after
F-0) *and* carries `.. vv-status: … documented`, so the generated matrix lists it under
both "verified by N tests" and "Documented-only equations". That reads like a bug and is
not: on this corpus `documented` marks *representational / face-distinction / literature*
KIND, and its siblings on the same page (`scattering-spectral-theorem`,
`galerkin-strict-adjoint-vs-reconstruction`, `moment-projection-transpose-T`) all do the
same. **Do not "clean it up"** — you would be re-categorising a whole convention from one
label, and it moves a generated artefact. Keep the directive; write the RATIONALE comment
if it is missing (this one had none — I added one naming all four gates).

### 6. ⭐ A three-way SYMBOL COLLISION, and the resolution that keeps every established
### spelling: rename only the one with no constituency.

The frame page already used `W` for **the coefficient space** (`R : W → V`) and
`⟨·,·⟩_W` for **the quadrature-weighted nodal metric**; the code and
`normalization.rst`'s ledger use `W = Σ w_n` for **the scalar total weight**. My
derivation needed a fourth: the weight **matrix**.

⟹ L-051's rule (keep the code's spelling, pay with a `.. note::`) resolves this cleanly
once you notice the four are not symmetric: three have constituencies (a page convention,
a page convention, the code + the ledger), the matrix has none. So write the matrix as
`\mathrm{diag}(w)` — never `W` — and open the section with a `.. warning::` naming all
three survivors and stating the rule. Cost: one admonition; benefit: every equation in a
650-line section is unambiguous.

### 7. ⭐⭐ The reusable close-out shape for a LATENT defect: "why nothing caught it" is
### THREE independent shields, and the third one is the dangerous sentence.

I wrote it three times (frame.rst, spherical_harmonics.rst, error_catalog.rst) and it is
the load-bearing pedagogy of the whole record:

1. **Consistency is not correctness.** The defining adjoint identity
   `⟨Mψ,c⟩_g = ⟨ψ,M*c⟩_W` held at the round-off floor (`[M]` `9.5e-16` at L=1, **exactly
   `0.0`** at L=2) — because `.H` is *built from* the stored metric. It is true for
   **every** SPD metric and therefore carries **zero information about which one is
   installed**. The instrument that CAN fail compares the metric to something defined
   without it (here Parseval: the field's own norm).
2. **Composed chains are immune** — interior metrics cancel, so the production kernel
   never reads a face's `.H`, and the 0-ULP canary is green by construction.
3. **Only end-of-chain adjoints are exposed, and there were none.** `[M]`
   `grep -rn "analysis\.H\|reconstruction\.H" orpheus/` → exactly one hit, a docstring.

⛔ **Shield 3 is where a close-out goes wrong.** "No consumer exists" is not safety, it is
**latency** — write it that way, with the clock: *the defect becomes live with the first
adjoint consumer, which is why the metric had to be right before those land*. A page that
reports shield 3 as reassurance teaches the next session to defer.

### 8. Extending an ERR entry vs minting a new one — the decision, and the one thing to
### check first.

F-0 is the THIRD chapter of ERR-039 (Wave 0 → Phase 1 → F-0), all "metric / transpose /
adjoint conflation on the same operator pair", each one level deeper: *wrong operator* →
*right operator, unasked metric* → *right Gram, WRONG SIDE*. Extending was right, and the
decisive check is not narrative tidiness — it is that **the landed gates already carry
`catches("ERR-039")`**, so a new number would silently orphan them and I cannot edit
`tests/`. ⟹ **Read the catching tests' markers BEFORE choosing the ERR number**; the
marker set is the constraint, the narrative is not.

Also: mark the superseded chapter IN PLACE (`⛔ superseded 2026-08-23; see the F-0
chapter below`) on the *bullet* that states the retired formula, not only in the new
chapter — a reader who lands on the Phase-1 list must not read it as current.

### 9. Corrections that the WIDENED sweep produced (the brief named 4 of 6 sites)

The brief's grep list found `spherical_harmonics.rst`, `frame.rst` ~2700, and
`normalization.rst:161`. Running the sweep myself added:
- **`frame.rst` "Numerical evidence"** — item 2 still read `M* = g_C S_0`, ~1100 lines
  from the note the brief pointed at (L-044's "audit the PARAGRAPH FAMILY, not the diff").
- **`frame.rst` Schur bullet** — `g_C` as "the SO(3) Plancherel weight": TRUE (it is the
  continuum Gram) but now one reciprocal away from the frame's metric; qualified rather
  than changed.
- **`operator_algebra.rst:3295`** — "the addition-theorem `R`, not the W-weighted adjoint
  of `M`": the *negation* survived the repair while its vocabulary died. Post-F-0 `R = W·M*`
  exactly, so the sentence is improved by stating the relation instead of denying one.
- **`normalization.rst`'s ledger row was the THIRD contradiction** and the brief did not
  know it existed — it was found by grepping `W-weighted`, not `g_C`.
⭐ And the payoff nobody asked for: that page's own "unification the canon misses"
(`(2ℓ+1)/W`, W = 4π sphere / 2 slab) **IS** the Parseval metric `G⁻¹ = (2ℓ+1)/W`. A sweep
for staleness turned into the strongest single piece of corroboration on the page —
*always read what the stale site was TRYING to say.*

### 10. Findings reported, NOT fixed (code/tests are off limits)

- **`orpheus/numerics/frame.py:116-119`** (the `_DISCRETE_GRAM_DIAGONALITY_RTOL` docstring):
  says the slab live off-diagonals "sit at ~0.5 of the Cauchy–Schwarz scale
  `√(G_jj G_kk)`". `[M]` relative to the C–S scale they are **0.9347**; **0.5774** is
  relative to the largest DIAGONAL. Verdict unaffected (threshold `1e-10`), but the
  *stated normalisation* is wrong and the same wording is copied into two
  `tests/numerics/test_frame.py` docstrings. ⟹ when a docstring quotes a ratio, check
  WHICH denominator — two plausible ones differ by 1.6× here.
- **`spherical_harmonic_space.py`** class docstring's `inner_product_weights` parameter
  still says "row ℓ holds 4π/(2ℓ+1)". `from_L`'s docstring WAS updated by the F-0 commit;
  the class-level parameter description was not — and the frame-dressed object IS a
  `SphericalHarmonicSpace` (built by `dataclasses.replace`), so it is present-tense-false
  for the majority instance. A half-done docstring sweep, exactly `vv-principles` #21.
- `scratch/probe_f1_parseval*.py` no longer reproduce their own headline (see §2).
- The `check_docstring_xrefs.py` `.rst` blind spot (L-062's one-line `head_role` fix) is
  **still unlanded** — re-confirmed: it gates only `:mod:` on `.rst` pages, so my own
  import probe was the acceptance evidence for the 16 roles I added.

### Quality self-assessment

| dimension | score | note |
|---|---|---|
| Derivation depth | 5 | φ = Gc from three re-associated products; the metric SOLVED for, not asserted; both general adjoint sandwiches collapsed step by step; the SH-specific collapse isolated as its own labelled identity |
| Cross-references | 5 | 16 roles added, 0 dead by import probe; 5 new anchors, every cross-doc `:ref:` verified as a rendered `href` |
| Numerical evidence | 5 | 7-frame × 5-residual table + the slab refusal table + the indicator instance + the pre/post ratio table — every figure re-measured this session, with the construction published |
| Failed approaches | 5 | the three-metric table IS the failed-approach record; ⛔ pre-F-0 equation preserved unlabelled; the 3-shield "why nothing caught it"; the refuted LS₈ 24 % claim |
| Code traceability | 5 | 17 declared `implements::` across 4 labels + 3 `no-implementation` blocks, all pre-flighted against `graph.db` and verified in the rebuilt graph |
| Derivation source | 4 | derived from the LIVE code + my own probe (no `derivations/` script exists for frame algebra; the plan's probes are pre-repair scratch — flagged) |

## L-066 — A FACTORY-TIER retirement is a THREE-TENSE sweep, and the disposition is decided by the SENTENCE, not by the symbol

**Task (2026-08-24, branch `feature/cs1-energy-space`, campaign 1 CS4b S5).** CS4b S5
retired the mesh-keyed sugar tier on every transport field leaf — `from_mesh(values, mesh)`,
`zeros_on(mesh)`, `from_ndarray(arr, mesh)` deleted from `AngularField` / `ScalarField` /
`FaceField` and every concrete leaf, `MomentField.from_ndarray` too, plus the
`spatial_moments=` int on those factories. Replacement is SPACE-primary: `Leaf(values=…,
space=…)` and `Leaf.zeros(space)` on the carrier's cached mints (`angular_bulk_space`,
`bulk_space`, `angular_trace`, `scalar_trace`, the two ψ½ spaces), with a NEW mint
`SNMesh.angular_trial_space` replacing the retired int. Composites went space-keyed
(`FullField.zeros(..., space=)`, `RadialCharacteristicField.flux_zeros/source_zeros`,
presence-gated). Brief supplied a ~30-hit grep list + three disposition rules.
Gates: `-E -W` EXIT=0, W/E/C/SyntaxWarning set unchanged (**0 ↔ 0**); `DEAD TARGETS 0`;
my own import probe over the 37 qualified roles on added lines = **0 dead**; vv violations
0, sentinels 545 unchanged; 6 doc files, +360/−106.

### The load-bearing finding: three tenses, one symbol, three different repairs

A factory retirement scatters the SAME token across three grammatical registers, and the
symbol grep cannot tell them apart. Sorting the 30 hits by TENSE gave a clean 3-way split
that turned out to be the whole job:

| register | example (verbatim) | repair |
|---|---|---|
| **live guidance** — prose/tables/code telling the reader how to build a field TODAY | *"``from_mesh_laws`` returns exactly ``zeros_on``"*; the ladder table's bottom rung; a runnable `code-block` | re-word to the space-primary spelling **with the right carrier mint**; a wrong mint is a fresh falsehood, so measure it |
| **history** — a landed change's own narrative, an ERR-NNN post-mortem, a "before X, callers hand-rolled…" | *"a hand-rolled ``from_mesh(trace.values.copy())`` beside a Pattern-4 factory"* | **prose STAYS** (past tense is correct history); a `:meth:` role pointing at the deleted target is DOWNGRADED to a ``literal`` keeping the exact old name |
| **landed-but-written-as-future** — a step record whose "NEXT sub-step" already shipped | *"When they land, the only change at the factory call sites is passing ``spatial_moments=…``"* | re-tense to past + a dated `.. note::` saying where the capability lives now |

⭐ **The third register is the one a symbol grep is worst at and the one that costs the
most**, because it reads as a *plan*, not as a claim, so nobody re-checks it. `[M]` in
`cartesian_multid.rst` an entire "Construct-general, select-narrow — what this step does and
does NOT do" subsection was present tense about a state two campaigns old: *"No production
field selects it. The ``spatial_moments`` factory parameter defaults to ``1`` at EVERY call
site and is NOT auto-read from ``mesh.scheme.spatial_basis_per_axis``"* — while
`solver.py:920` and `streaming.py:999` both pass exactly that expression, and the leaf route
had since moved to `angular_trial_space` entirely. Repair shape that worked: **re-tense the
bullets in place** (adding *(as of S3)* to the one that is a dated observation), flip the
section TITLE's verb too ("does and does NOT do" → "did and did NOT do" — free, since the
section carried no `.. _anchor:`), and append ONE dated `.. note::` carrying the live
mechanism + the measured shapes + the named survivors. Do NOT delete the bullets: they are
the reason the capability was built default-OFF, which is the durable content.

### Two-step ownership history is worth its own paragraph (the brief's disposition (c))

The `indexing_and_layout.rst` allocator section had ALREADY been corrected once — a 2026-08-10
`.. note:: **Correction (Issue #346)**` recording that the allocator moved OFF `SNMesh` and
ONTO the leaf. S5 moved it again (off the mesh KEY onto the space KEY). ⟹ the honest shape is
a **second `.. note::` beside the first**, not a rewrite of the first: *"#346 moved the owner;
S5 moved the key; the leaf is still the owner"*. Both notes carry a one-command `[M]`
(`[n for n in dir(SNMesh) if "zero" in n.lower()] == []` / `hasattr(AngularField,"zeros_on")
is False`). A reader landing on either correction now sees the whole arc.

⭐ **And the trap inside that section: it QUOTED a production docstring that no longer
exists.** The old text read *"The production docstrings say so in as many words — «the uniform
leaf-side allocator … replaces the retired `SNMesh.zeros_*` mesh-side factories»"*. `[M]`
`grep -rn "uniform leaf-side" orpheus/` → **0 hits**; `Field.zeros`'s docstring now says the
opposite-keyed thing. A quotation is a claim about a FILE, invisible to every gate (no role,
no label, no path). ⟹ **grep the quoted STRING, not just the symbol, whenever a doc quotes
code.** Same class as the raw-file-path defect of L-062, one register up.

### Measure the replacement before you write it (the mint is a choice, and one is wrong)

The re-word target is not mechanical: `zeros_on(mesh)` maps to **five** different mints
depending on the leaf family, and `angular_bulk_space` vs `angular_trial_space` is a real
choice at every angular site. I built one probe and published its table:

`[M]` vacuum slab, `N = 4` (GL), `ng = 2`, `nx = 4`:
`angular_bulk_space (4,2,4)` · `bulk_space (2,4)` · `angular_trace (16,)` ·
`full_field_space` blocks `(4,2,4)+(16,)`; **DD: `angular_trial_space is angular_bulk_space`
→ True** (same cached instance). Same slab under `LinearDiscontinuous`: trial reads
`(4,2,4,2)`, axes `angular(4)·energy(2)·spatial(4)·spatial_moment(2)`, the moment factor
MODAL with measure `(1, 1/3)`. Sphere (4 cells, GL(4)): ψ½ blocks `(16,)` cells + `(4,)`
corner; the same `flux_zeros` call on the slab RAISES the R12a diagnosis.

⭐ The `is`-identity at DD is the single most useful sentence in the rewrite — it is what
makes "read the trial mint" safe advice at *every* scheme width, so the doc's worked example
can carry one line instead of a branch.

### The doc code block was ALREADY broken, in a way unrelated to the retirement

`verification/sn.rst`'s composite-source example built `TimedFullField(bulk=…, boundary=…)`.
`[M]` the dataclass fields are `interior` / `boundary` — `bulk=` has not existed for
campaigns. Found only by RUNNING the block (lesson-2 rule), which I had to do anyway to pick
the mint. Fixed in the same edit and pinned the result in prose (`[M] max φ = 1.8265` on the
stated fixture) so the next reader can falsify it. ⟹ **a retirement sweep that touches a code
block owes that block a RUN**, and the run finds defects the sweep was not looking for.

### Housekeeping observed, not caused

* `docs/theory/verification/matrix.rst` is regenerated by the `builder-inited` hook, so the
  FIRST `-E` build of a session materialises whatever drift the landed code/test commits left
  (here `10063 → 10067` collected, `test_meshless_construction 5 → 8`). It shows up in
  `git diff` and is **not** a hand edit — report it, never revert it, never edit it.
* My added-role probe's one "DEAD" hit was an **unqualified** role
  (`:meth:`BulkField._compose_spatial_moments``) on a line I only re-tensed. Unqualified roles
  resolve by Sphinx module context and the gate skips them by design — a false positive of the
  probe, not a defect. Check the diff (`git diff | grep '^[+-].*symbol'`) before believing the
  probe on an unqualified target.

---

## L-067 — Documenting a MEASURED machinery: the brief's numbers were a sample, the record's `[M]` carried a confound, and the gate that certifies me is blind to its own class

**Task.** CS4b S7 docs half (branch `feature/cs1-energy-space`, 2026-08-24): write the
flagship "axis collapse pair" section on `spaces.rst`, repair `infinite_medium.rst` for the
EE-1 typed-rate landing + the K2 pose split, fix the `field_algebra` mesh-identity rows, the
frame-square `Π* = R` contradiction, and the changelogs. Six pages, +1407/−104. Final
`-E -W` EXIT=0 with the WARNING/ERROR/CRITICAL/SyntaxWarning **set unchanged (0 ↔ 0)** from a
freshly measured baseline; vv violations 0, sentinels 545 → 549; `dead_references` 1 (a
pre-existing PRODUCTION docstring, not mine).

### 1. ⭐⭐ "Bit-exact" is a property of the DRAW, and a gate's seed is a sample of size ONE

The brief and the gate docstrings both said `R ∘ E = id` is `[M]` **BIT-EXACT**. It is not a
property of the operators. `R∘E` computes `Σ_n w_n·(φ/Σw)`; whether the round-off floor is
*zero* depends on how those products re-associate for the particular numbers involved.

`[M]` on the gate's own synthetic fixture (`w = [0.3,0.7,0.5,0.5]`, `Σw = 2.0`):

| row | `np.array_equal` holds on |
|---|---|
| `R∘E = id` (G6.1) | **1156 of 2000** seeds — fails on 844, worst rel `1.480e-16` (~1 ULP) |
| `P = E∘R` idempotent (G6.2) | **143 of 200** seeds — fails on 57 |
| both, on the shipped SN carrier (GL4 slab) | **200 of 200** |

Both gate rows are green because the seeds they hard-code (`1` and `2`) land in the exact
set. **Change the seed and they red.** The SN carrier is robustly exact because `Σw = 2`
exactly *and* the symmetric GL weights re-associate cleanly — which is what licenses
`np.array_equal` on the production-facing rows and does NOT license it on the synthetic one.
⟹ Publish a **bound over ≥200 draws** with the norm and the seed family written out
(`max‖a−b‖_∞/‖b‖_∞` over `default_rng(1000+k)`), never a single reading; and say WHICH
fixture is robustly exact and why. Reported the seed-fragility upward — I do not edit `tests/`.

⚠ The dual, same session: the **tightness** rows (minted kernels vs the literal frame's face
contents) ARE robustly bit-exact — `[M]` 200/200 on all three correspondences — because both
sides evaluate the same reduction in the same order. So "bit-exact" is sometimes a property
of the construction and sometimes of the draw, and only the measurement separates them.
Saying which is the whole content of the row.

### 2. ⭐⭐ A COINCIDENCE claim needs its family, and this one is false exactly where it matters

Brief + production docstring + gate docstring all carry: *"the gram einsum is bit-identical to
`weights.sum()` on 8 of 8 probed fixtures (n ∈ {2,4,5,6,16,64} incl. GL64's inexact Σw)"*.
`[M]` mine, two weight families:

| n | `leggauss` weights | `linspace(0.1,1.3,n)` |
|---|---|---|
| 2, 4, 5, 6 | identical | identical |
| **16, 64** | **NOT identical** (`2.0000000000000004` vs `2.0`) | **NOT identical** |

The Gram is `einsum("n,nj,nk->jk", w, T, T)`; `ndarray.sum` is a pairwise reduction. And on
the **shipped SN quadratures** the split lands at **GL8**: divisor `1.9999999999999998` vs
`quad.weights.sum() = 2.0`, so `AngularSourceSink.from_isotropic` differs from a hand-written
`Q/Σw` by `2.0e-16` relative in production. ⟹ The structural claim (*the divisor IS the
frame's `discrete_gram[0,0]`*) is exact by construction and is what a gate must pin; the
coincidence with `weights.sum()` is a fixture accident and must never be relied on. Published
the ladder as a table with its two columns side by side.

### 3. ⭐⭐ A design record's `[M]` can carry a CONFOUND — two settings moved together, one got the credit

The record read: *"sphere GL L=1 (DIAGONAL Gram): `face.H(e₀φ) == E(φ)` to 2.2e-16; Slab L=2
(DENSE Gram): un-physical"* — attributing the split to **geometry**. It cannot be geometry:
the angular frame is built from `sn.quad`, which knows nothing about the spatial coordinate
system. Running the crossed cell:

| fixture | Gram max off-diag | `face.H(e₀φ)` vs `E φ` | `reconstruct(e₀φ)/W` vs `E φ` |
|---|---|---|---|
| slab L=1 | 5.6e-17 | 5.6e-17 | **0.0** `array_equal` |
| sphere L=1 | 5.6e-17 | 1.1e-16 | **0.0** |
| slab L=2 | 1.155 | 16.17 | **0.0** |
| sphere L=2 | 1.155 | **16.17** | **0.0** |

⟹ the discriminator is the **Gram's diagonality**, i.e. **L**, and geometry is inert. A 1-D
polar rule has no azimuthal nodes, so the m≠0 modes are not orthogonal under it — `[M]`
`gauss_legendre(8)` at L=2 reads the SAME `1.155` / `16.17`, so refining the order does not
fix it. The clean, **metric-free** statement (`E = reconstruct(e₀·)/W`) is bit-exact in all
four. ⟹ **When a record's two arms differ in more than one setting, run the crossed cell
before publishing either as the cause.** The correction made the section stronger: it is the
reason the collapse pair is minted from an *indicator* frame instead of lifted out of the
harmonic one — it must keep working where the harmonic metric does not exist.

⚠ And: the committed probe `scratch/probe_s6_q5_dissolution.py` is the SLAB L=2 arm — the arm
the record itself calls un-physical — so run as committed it prints `1.617e+01` while a
**production docstring** cites that path for `2.2e-16`. A scratch probe cited from `orpheus/`
is a raw-path claim about a file that no instrument checks (L-062, one register up).

### 4. ⭐⭐ The xref gate's `head_role` bug is ROLE-scoped, not `.rst`-scoped — my own memory was too narrow

L-053/L-062 recorded this as *"on an `.rst` PAGE that gate reports `:mod:` and nothing else"*.
`[M]` it is worse and simpler: `judge(target, ns, role)` re-checks the target's HEAD **carrying
the original role**, and `candidate_paths("orpheus", ns, "meth")` returns
`('<namespace>.orpheus',)` — which never resolves — so every **dead** fully-qualified
`orpheus.*` target under a non-`mod` role is DECLINED, in `.py` docstrings exactly as in
`.rst`. Live ones return ALIVE earlier and are unaffected, which is why the gate looks healthy.

- `judge("orpheus.numerics.space.FunctionSpace.definitely_not_here", role="meth")` → **DECLINED**
- `judge("orpheus.numerics.does_not_exist", role="mod")` → **DEAD**

⟹ **`DEAD TARGETS: 0` certifies `:mod:` targets and nothing else.** The one-line fix
(`head_role = "mod" if "." in target else role`) applied to a COPY, run as a subprocess from
inside the repo (it resolves paths against `REPO_ROOT`, so a `/tmp` copy scans 0 files), read
**1 dead target / 2 sites** where the stock gate read 0 — one of them my own new xref, one a
pre-existing production docstring. ⟹ the acceptance evidence for a page is still YOUR OWN
import probe; the gate is a `:mod:` check.

### 5. ⭐ Two independently-vocabularied instruments agreeing IS the acceptance evidence

nexus `dead_references` (resolves by RENDERED target) and the patched gate (resolves by
IMPORT) returned **exactly the same single finding** — `FaceField.from_face_arrays` at
`face_layout.py:355`. Neither alone would have been persuasive: the stock gate said 0, and
nexus's set-difference with the gate is normally noisy (L-052). Convergence from two different
resolution mechanisms is what makes a one-line report actionable.

### 6. ⭐ A brief can name the wrong CLASS for a method — and the same error is already in production

The brief said *"`FaceField.from_face_arrays` is the typed entry"*. `[M]` `hasattr(FaceField,
"from_face_arrays")` is **False**; it lives on `BoundaryField`. I wrote the brief's spelling
into a changelog row and my own selfcheck caught it — and the SAME wrong class is in
`orpheus/numerics/face_layout.py:363`, which is where the brief's author almost certainly read
it. ⟹ a brief's symbol claim inherits the tree's own errors; `hasattr` every method-on-class
before minting a role.

### 7. Repair shapes worth reusing

- **A self-contradicting Key Facts block, 12 lines apart.** `frame.rst` promised
  `GalerkinFrame ⟹ Π* = R` and, twelve lines below in the same admonition, `M* = R/W`. The
  post-F-0 truth is neither: Galerkin fixes *which basis* the adjoint re-synthesises on (the
  trial one, `M* = S₀∘G⁻¹` — a **canonical** dual), and the metric stays. Fix at BOTH poles
  (`Π* ∝ R` in the diagram + a ⚠ clause naming ERR-039/051 and the indicator counter-example),
  never at one.
- **A "single-sourced through X" claim is two claims.** `operator_algebra.rst` said the
  `iso + aniso` dunder is single-sourced through `from_isotropic`. `[M]` the dunder's body is
  `self.values[None] + other.values` — the **plain** broadcast — while `from_isotropic` applies
  `1/Σw`. They differ by exactly the axis's total weight, i.e. they are the two arrows the whole
  section I was writing exists to keep apart. The repair writes the ⚠ *and* points at the new
  section, so the falsehood becomes the worked example.
- **A retired guard tier leaves a stale REASON attached to a surviving FACT.** `verification/sn.rst`
  said the composite is re-homed "because `TimedFullField` algebra enforces mesh identity". The
  re-home still happens; the reason is now space CONTENT. Keep the instruction, replace the
  reason, and say what changed — a reader who trusts the old reason will "optimise away" the
  re-home for a twin carrier.
- **The production helper's own docstring lied the same way.** `_require_typed_composite`'s
  docstring says *"(2) `field.interior.mesh` is the operator's SAME `sn_mesh` instance"* over a
  body that compares `field.interior.space != field.interior.space_on(sn_mesh)`. Reported.
- **A colliding bare step number.** `spaces.rst` said the V/V* condensation morphisms are
  "scheduled for S7" — a plan-internal number that collides with CS4b's own step S7, which
  landed that day and built none of it. Disambiguated at BOTH sites (`plan-authoring` §9b in the
  corpus).
- **A fence row that fell.** "Only the scalar bulk is axis-built; every other space is legacy"
  → `[M]` the angular bulk and the scheme-widened trial space are axis-built too and report
  `has_coordinate_cone is True`; what is still legacy is the composite and the flat traces.
  Re-title the fence to what is actually still fenced.

### 8. The changelog routing, again

`methods/sn/history.rst` contracts *"a new entry lands with its merge hash or not at all"*, so
an unmerged branch is BLOCKED there — while `spaces.rst`, `field_algebra.rst` and
`operator_algebra.rst` each carry the *(in development)* escape hatch. ⟹ route the entry to
the page whose SUBJECT moved and which permits the hatch; report the SN row ready-to-paste.

---

## L-068 — Discharging a merge-hash contract: the blast radius is the BRANCH NAME, not the blocked page

**Task (2026-08-24).** `feature/cs1-energy-space` ff-merged to `main` at `55bb47b9`
(90 commits, 264 files). My held contract — `methods/sn/history.rst` contracts
*"a new entry lands with its merge hash or not at all"* — was finally
dischargeable, and the dispatch was scoped to "write the row(s)".
Landed `68d265ef` on `docs/sn-history-campaign1-landing`.

### 1. ⭐⭐ The merge event falsifies the SIBLING pages the same instant

L-067 taught the *routing* rule: when `history.rst` blocks you, route the entry to
the page that carries the `*(in development)*` hatch (`spaces.rst`,
`field_algebra.rst`, `operator_algebra.rst`). What it did not say is that the
hatch is a **debt**, and the merge is what calls it in. The moment the branch
merges, every `*(in development)* branch ``<name>``` cell is present-tense-FALSE —
and nothing points at them, because the dispatch names only the blocked page.

⟹ **On discharging the contract, grep the BRANCH NAME across `docs/` first.**
`[M]` `grep -rn "cs1-energy-space" docs --include="*.rst"` → exactly **3 cells**
(2 on `spaces.rst`, 1 on `field_algebra.rst`), all replaced with
`merged @ ``55bb47b9`` —`. One command, one minute, and it is the difference
between a corpus that agrees with git and one that says three campaigns are still
in flight. ⚠ Also grep `"in development)\*"` corpus-wide — that catches a hatch
whose branch was named differently, and it found the standing *explanatory
sentence* on `operator_algebra.rst` (a convention note with no instances, which
correctly STAYS).

### 2. ⭐ A DATE in a prose history block is a git question, and it drifts by one

`frame.rst:4744` read `**2026-08-24 — step F-1, the mint**`. `[M]`
`git log --format="%h %ad" --date=iso -1 3dfea889` → **2026-08-23 16:19:54**.
Its F-0 sibling four paragraphs up was right; the S6.0b block below was right.
One block, one day off — written from "which session was I in", not from git.
This is lessons §1's *"Git is the arbiter for dates"* (L-064) applied to a prose
changelog rather than a claim, and the tell is free: **when you cite a commit's
date into a NEW row, you have already looked it up — diff it against every
existing prose block naming the same commit.**

### 3. Row granularity: group by THESIS, and let the page's own precedent settle it

The merge carried five architecturally distinct milestones. One consolidated row
or five? The page answers itself: `[M]` the #280 coupled-block campaign holds
**six** rows across 2026-07-05…07-12, every one stamped `(merged @ ``3f0b8c74``)`.
So per-milestone rows sharing one merge hash IS the convention, and the `Where`
format is `` `<step hash>` (merged @ ``<merge hash>``) ``.

⟹ **Group by the THESIS that moved, never by the plan's phase labels.** The
campaign's own step boundaries (S4 / S4-amendment / F-0 / F-1) cut across
subjects: S4 is field-layer, the S4-amendment is operator-layer, and they landed
in one session. The rows I wrote are *"a field is an element of a space"*,
*"the frame owns its metric and mints bound faces"*, *"an operator is not an
operator without its two spaces"*, *"S/F/C are kernels"*, *"the space layer gains
axes"* — five theses, not five phases.

⚠ And **strip the plan-internal tokens on the way in.** My first draft carried
*"the standing R2 hazard"* and *"a ``§6b`` call-site set"* — a plan's risk label
and a rules-file section number, both meaningless in the corpus and both colliding
with live campaigns' own numbering (L-067's bare-step-number rule). Rewrote to
*"the hazard the monomorphic-leaves suite had catalogued"* and *"a call-site set
that is complete by symbol grep"*. The corpus says what the thing IS.

### 4. ⭐⭐ A row whose MECHANISM a later row in the same merge overturned gets an in-place ⛔, not a rewrite

The S4-amendment's step A1 (`6e04a749`, 2026-08-22) bound `HarmonicFrame` itself
to its two field spaces at construction. F-1 (`3dfea889`, 2026-08-23) **reverted
it** — a frame is a shared FACTORY, so the binding belongs on the faces it mints
(`[M]` live: `HarmonicFrame.__init__(self, basis, measure)`). Both landed in the
same merge, one day apart.

Writing A1 in the present tense would have shipped a falsehood; deleting it would
have destroyed the reason the correction happened. So the 2026-08-22 row carries
its (d) clause plus, in place:

> ⛔ **Superseded the next day by F-1** (the row above): the binding belongs on the
> *faces* a frame mints, not on the shared factory. The amendment's *demand* stands
> unchanged — it is what made the misplacement visible in the first place.

That last sentence is the load-bearing one: it says which HALF survived. This is
`plan-authoring` §3 (edit the refuted premise in place) landing in the corpus, and
in a reverse-chronological table *"the row above"* is a correct pointer.

### 5. The gate stack for a changelog-only edit — and why the standard gate is not enough

Baseline and post, both forced `-E`: EXIT=0, W/E/C/SyntaxWarning **0 ↔ 0**;
`check_docstring_xrefs.py` `DEAD TARGETS 0`; nexus `dead_references`
`total_dead 0`; vv-status `violations 0 / sentinels 549` (unchanged — the rows add
no `:label:`); `verification/matrix.rst` regenerated byte-identical.

But per L-062/L-067 that gate is ROLE-scoped-blind, so the acceptance evidence was
**my own import probe over the added lines**: parse every
`:role:`target`` out of `git diff --unified=0 -- docs/` (flatten whitespace FIRST —
roles wrap), strip the `<display <target>>` form and the `~`, then walk
`importlib` + `hasattr`. `[M]` 36 distinct qualified roles, **0 dead** — and it
caught two before the build: `orpheus.data.mixture.Mixture` (lives at
`orpheus.data.macro_xs.mixture`) and `orpheus.numerics.operator.AdjointOperator`
(private `_AdjointOperator`; rewrote the prose to "the generic adjoint wrapper"
rather than xref a private class).

⭐ **And the render check the build cannot do**: slice the built HTML between the
new rows' first and last distinctive phrases, strip tags, unescape, and count
**visible backticks** and **surviving `:role:` spellings**. `[M]` 0 and 0 in my
rows. This is the only instrument that proves the markup MEANT what it said —
see §6 for what it found in the rest of the page.

### 6. ⭐⭐ RST cannot nest inline markup — and the census is exhaustive, so measure it with the ISSUE'S OWN instrument

The same render check reported **84 visible backticks on the page** — none mine,
all pre-existing, all one mechanism: `**bold naming ``a symbol``**`. RST forbids
nested inline markup, so the inner delimiters render literally. Silent at every
severity.

`#379` already owned this, scoped to the error catalogue at `[M]` 32 runs. Rather
than file a sibling, I re-ran **#379's own grep** corpus-wide:
`<(strong|em)>[^<]*``[^<]*</(strong|em)>` → **125 runs across 25 live pages**,
with the catalogue's 32 reproducing exactly (a control that the instruments agree)
at 26 % of the total. Posted as a widening comment with a suggested retitle.

Three things worth carrying:

- ⚠ **Exclude `_build` pages whose `.rst` no longer exists.** `[M]` 12 orphaned
  pages carry a further **76 runs** no source edit can reach — counting them
  inflates the figure by 60 %. (Also exclude `_modules/`: a literal backtick in a
  viewcode source listing is correct output.) Same trap my memory already warns
  about for stale-ref greps, here in a *measurement*.
- ⭐ **The strictly worse sibling ranks above it**: the same RST rule (inline
  markup may not open after `. * ~ § ↔ =`) kills **104 interpreted-text ROLES
  across 28 live pages** — they survive into rendered prose as their own source
  spelling, with the LaTeX backslash eaten. `[M]` the commonest survivor is
  `` :math:`mu` `` (10 sites), then `` :math:`tau` `` (6). A visible backtick is
  ugly; a dead `:math:` is a **missing equation**.
- ⭐ **Both greps are a CENSUS, not a sample**, and say so when publishing: RST
  admits no role spelling in rendered prose and no stray delimiter outside a
  literal block, so a hit is a defect by construction. That is the L-061 argument
  (a warning count is a non-representative sample of a fidelity-loss class) reused
  as the *justification* for the number rather than as a caution about it.

### 7. Numbers re-derived rather than relayed

- The byte gate: the plan says "D5 8/8". Ran it — `tests/homogeneous/test_byte_stability.py`,
  **8 passed**, and its own fixture docstring says *"exhaustive over what the tree
  ships"*, which is the phrase the row now uses.
- The GL8 correction: the plan's banner said the probe's ladder "skipped GL8". My
  draft embellished it to *"a ladder of eight fixtures that broke every arithmetic
  pattern"* — a property of the ORIGINAL probe I could not verify (only the gate's
  CURRENT list is knowable). Cut back to exactly what the gate docstring asserts.
  Same reflex as lessons §1's *"count your own universals"*, applied to an
  adjective.
- CS1's slot claim: my draft read *"the slot the kernel-binding phase then tightens
  to MANDATORY"*. `[M]` CS4a K2b made **F**'s space mandatory, not S's
  (`ScatteringOperator.__init__(..., space: FunctionSpace | None = None)` still
  ships). Scoped to `MANDATORY on :math:`F``.

---

## L-069 — The RENDER is the only gate for inline markup, and a LITERAL is not a role

**Task (2026-08-26).** Archive a three-probe investigation (literature sweep +
SymPy derivation + an original asymptotic derivation) into
`docs/theory/methods/sn/curvilinear_one_group.rst` and
`docs/theory/foundations/discretization.rst`: the tensor-product factorization
of the curvilinear angular-redistribution operator, the τ-arity theorem, the
Padé positivity ladder, the seed cone risk, and a refutation of Morel–Montry's
own 1984 summary rule. Mid-task the code carve I had been told not to name
LANDED, adding three stale-reference repairs.

### 1. ⭐⭐ A double-backtick LITERAL renders a backslash VERBATIM

In a `list-table` of measured values I wrote ``` ``1.4\times10^{-6}`` ``` for two
cells. A literal does exactly what a literal promises: the built page carried
the characters `1.4\times10^{-6}` in prose. `-W` EXIT=0, `-n` blind,
`check_docstring_xrefs.py` blind (it gates TARGETS, not whether a span parsed),
nexus `dead_references` blind. **The only instrument that saw it was slicing
the built HTML and counting raw TeX outside the MathJax spans.**

⟹ **a number in scientific notation is `:math:`1.4\times10^{-6}``, never a
literal.** The discriminator: does the cell contain a backslash? If yes it is
math, not code.

### 2. ⭐⭐ `**``value``**` is the commonest way to mint the nested-markup defect

RST cannot nest inline markup (L-068), and the shape I reached for *fourteen
times in one session* is a bold-wrapped literal in a numeric table cell —
`- **``-0.200000``**` — because I wanted the negative rows to stand out. Every
one rendered as ``` ``-0.200000`` ``` with four visible backticks. Plus one
`**`[M]` this is what ships**` (bold wrapping a `<cite>` span), two more.

⟹ **in a table cell a literal already carries its own visual weight; NEVER wrap
it in `**`.** Emphasis goes in the surrounding prose, which is where the reader
needs the interpretation anyway. Guard: `assert "**``" not in text` before the
write — one line, catches the whole family.

### 3. ⭐ A bare `:ref:` to a section whose TITLE contains `:math:` leaks raw TeX

`The dome closes — :math:`\alpha_{M+1/2} = 0` as an admission contract` is a
section title. A bare `:ref:` to it pulls the title as link text and the math
arrives as the literal characters `\alpha_{M+1/2} = 0`. Pre-existing on **4**
sites of that page — so it is page behaviour, not something I introduced — but
my two new sites made it five, and the fix is free: explicit link text
`` :ref:`the dome-closure contract <sn-alpha-dome-closes>` ``.

⟹ before adding a bare `:ref:`, look at the TARGET'S TITLE. Math in a title ⟹
explicit link text. (Same reflex as the admonition-anchor rule, different
cause: that one WARNS, this one is silent.)

### 4. ⭐⭐ Build the render checker carefully — its own regex is a false-negative source

Two instrument bugs, both of which made the checker useless in opposite
directions:

- **Sphinx emits display math as `<div class="math notranslate nohighlight"
  id="equation-X">`**, so a `<div class="math[^"]*">` strip misses EVERY
  numbered equation and the checker reports ~1000 false raw-TeX hits (it is
  reading correct MathJax source). Fix: `<div class="math[^"]*"[^>]*>`. Same
  for the inline `<span>`.
- **The `<head>` MathJax macro configuration is raw TeX in the page**
  (`"Sigt": ["\\Sigma_{\\mathrm{t},#1}", ...]`), so a whole-page scan always
  reports hits. Slice by `<section id="...">` to the id of the NEXT section.

⟹ **and the source-side regex alternative does not work at all.** A
`\*\*[^*]*``[^*]*\*\*` scan over my new blocks returned **26 suspects, 0 real**:
`**A** … **B**` matches as one run whenever no `*` sits between them, so every
adjacent pair of bold runs is a false positive. The render check found 3 real
classes with 0 false positives. **The rendered page is the instrument; the
source is not.**

### 5. ⭐⭐ Expand the series yourself — "monotone and positive" can still be INCONSISTENT

The source memo offered a positivity/accuracy trade inside the lumped-LD family
and named `(λ,ν) = (0,½)` as *"genuinely monotone at the cost of dropping to
**first** order"*, with transmission `2/((1+τ)(2+τ))`. I re-derived the family
from scratch (nodal DG cell, one free parameter per row) and the transmission
reproduces exactly — and the order label is wrong. `a'(0) = −3/2`, not `−1`:

| cells over a fixed `Σ_t X/|μ| = 1` | 10 | 100 | 1000 | 10000 |
|---|---|---|---|---|
| `(0,½)` | 0.2367 | 0.2245 | 0.2233 | **0.2231** |

It converges cleanly — to `e^{−3/2} = 0.223130`, not `e^{−1} = 0.367879`. It is
`vv-principles` #5 in its purest form: **a correct rate to the wrong limit**,
and both of the properties the memo checked (sign-preservation, `A⁻¹ ≥ 0`) are
perfectly true of it. Consistency is a THIRD property neither test sees.

⭐ The correction was cheap and produced a better object: solving
`a'(0) = −1` symbolically gives `ν = 1 − λ` (a ONE-parameter family, not two),
monotonicity gives `λ ≤ 0`, and the nearest monotone consistent member is
`(0,1)` with `a = 1/(1+τ_opt/2)²` — strictly positive, `A⁻¹ ≥ 0`, genuinely
first order. **A refuted memo claim replaced by a derived one is the best
possible outcome of "verify every number you cite".**

⟹ when a memo states an ORDER, expand the series. One `sp.series(a - exp(-t))`.

### 6. ⭐ Read the class docstring of the object you are theorising about

The carve landed mid-task and `AngularRedistribution`'s own docstring **already
states the tensor-product factorization** and cites the same memo. Two
consequences: (a) my chapter is the theory home for a structure the code
asserts, not a twin — say so; (b) **align to the code's exact spelling**
(`R_spatial ⊗ A_angular(τ, α, w)`), because internal consistency between code
and corpus outranks brevity (L-051). I had drafted `R ⊗ A_ang`.

### 7. ⭐ A `.. vv-status:` sentinel works INDENTED — check before relocating one

`tests/_harness/audit.py`'s `sentinel_re.match(stripped)` matches the STRIPPED
line, and the only rule is same-FILE. I nearly moved one out of a `.. warning::`
block on the belief it needed column 0. Read the scanner (30 s) instead of
reasoning about it. `[M]` 15/15 new labels registered, 0 violations,
documented 549 → 564.

### 8. ⭐ A retirement's stale REASON outlives its stale NAME

Site 1 of the carve's blast radius read *"``alpha_half`` … stay on the geometry
side — they are genuinely geometric"*. The NAMES were the greppable half; the
load-bearing half was the **reason**, which the factorization refutes (the dome
is the ANGULAR factor, a function of `(quadrature, coord)` alone). The same
false reason appeared a second time 1200 lines away in a development-history
item, where the names were correctly past-tensed and the reason was not.
⟹ after fixing a retired name, read the sentence that JUSTIFIES it.

### 9. ⭐ Reproduce a claim from the SHIPPED function, with the shapes it wants

`affine_scan_coefficients` takes `V` at `(N, nx)`, not `(nx,)` — my first two
attempts died on `V[:, None, :]`. Fed correctly, DD and LD reproduce the Padé
ladder to `1.1e-16` / `1.2e-16` over six optical depths, which converts "the
closed forms are the shipped scheme's" from an assertion into a measured bound.
Same for the seed: `carlson_inward_sweep_from_source` on 8 cells at
`Σ_t Δr = 3` returns `+0.4, −0.08, +0.016, …` — ratio `−0.2 = (2−3)/(2+3)`
exactly, i.e. the shipped seed march sign-alternates, measured on production.

### 10. Verified first-hand against the rendered scans (not the memos)

- **Adams–Martin 1992 App. A, printed p. 160** — read the page: (A.1a)/(A.1b)
  carry `+r_kΔr_k`, `−Δr_k²/6` / `+Δr_k²/6`, `−r_kΔr_k/3`. Two minus signs on
  the `ψ^x`-coupled entries; magnitudes match the Gram exactly. The sibling
  removal block `σ_tk[V_kψ + W_kψ^x]` / `σ_tk[W_kψ + X_kψ^x]` is symmetric on
  the same page — the typo argument is visible without leaving the page.
- **Hill 1975 ONETRAN, printed pp. 9–11** — Eq. (30) plain angular diamond
  *pointwise in r*; Eq. (32) applies it to the two-point spatial AVERAGE; (35a)
  shows the redistribution as `(α/w)[ΔA_i; z_5]⊗[1,1]`, manifestly rank-1;
  (36)–(38) the starting direction. The rank contradiction is real.
- **MWS 1996, printed p. 452** — Eqs. (74)/(75)/(76) and the verdict quote,
  verbatim; and **they name the Padé degrees themselves**, so the whole ladder
  framing is literature-backed rather than ours.
- **Palmer–Adams 1993 = UCRL-JC-111847** (the code docstring says
  UCRL-ID-114256, which is Palmer's *thesis* — reported, not edited).
  Their LD verdict is quoted as PREVIOUS work (their ref [5], Palmer–Adams
  1991), a nuance worth preserving.

### 11. My own derivations that replaced relayed numbers

- flat-flux row-1 identity: sphere `A_+ + A_- − 2V/h = 4πh²/3 = R_10` **exactly**
  (symbolic); cylinder both `= 0`, so the gate reads `0 = 0` there — the
  "run it on the SPHERE" rule is a theorem, not a measurement.
- `R_01/R_00 = h/(3(r_-+r_+)) ≤ 1/3` with equality iff `r_- = 0`, so `R` is SPD
  on every admissible cell and `det R ≠ 0` — which is what makes "β = 0 is
  necessary as well as sufficient" true.
- the 2×1 rectangular column `[ΔA ; ΔA·h/(6r_c)] = [ΔA ; 4πh²/3]`, matching
  ONETRAN's own `[ΔA_i ; z_5]`.
- `β⁻/β⁺` half-range split at the M-M τ: `+0.101808 / −0.101808` (N=4) …
  `+0.124610 / −0.124610` (N=32), sum ~1e-17 — reproduces the memo to every
  digit and is the evidence for "β = 0 is a GLOBAL identity across μ = 0".
- `β_e` sign flip `+9.107e-01 (N=2) → −1.111e-01 (N=4)`, and the `|μ_s+1|`
  equivalents `0.1132 / 0.0161 / 0.0053 / 0.0015 / 0.0004`; and
  `morel_montry_beta = 1.5 × β` bit-for-bit, so the shipped instrument IS the
  object the seed analysis needs.

## L-070 — the α-dome citation retraction + the MoC/CP sharing claim (2026-08-27, branch `fix/alpha-doc-claims-that-are-false`)

**Task.** Two present-tense-false corpus claims: (1) the α-dome recursion attributed to
"Bailey 2009" in prose AND in a live `:label:`; (2) `reduced_operator` "shared by SN, MoC,
and CP curvilinear sweeps". Fix both with meaning- and tense-triage. Doc-only mandate; main
agent commits; `orpheus/derivations/discrete/sn/angular_differencing.py` off-limits
(concurrent edit).

**Gates.** `-E -W` baseline EXIT=0, W/E/C = 0/0/0, re-measured this session. Verification
build identical (0/0/0, EXIT=0). `pytest -O -q -p no:randomly -k "structured_geometry or
reduced_operator or alpha"` → **342 passed, 9792 deselected, 5 warnings in 339.52s**.
`check_docstring_xrefs.py` DEAD TARGETS 0. vv scanner 0 violations, `documented` 564.
Render check: dead roles on the edited page **2 → 0**.

### 1. ⭐⭐ A brief's "these citations are CORRECT, do not touch" list is a CLASSIFICATION and it can be WRONG — resolve each site by its BIBLIOGRAPHIC ENTRY, never by the author-year string

`[M]` "Bailey … 2009" in this tree is **two different papers by the same four authors**:

| entry | spelling in tree | cited for | verdict |
|---|---|---|---|
| **(B)** retracted | *"A piecewise linear finite element discretization of the **diffusion** equation for arbitrary polyhedral grids", **JCP 227**, 3738-3757* | "Eq. 50 (dome recursion), Eq. 74 (Morel–Montry)" | the wrong-paper citation retracted at Issue #168 Phase B |
| **(A)** different | *"A piecewise linear **discontinuous** finite element spatial discretization of the **transport** equation", **Ann. Nucl. Energy 35**, 1929-1936* | the η-ascending level-ordering / `level_structured` convention | a *different* entry; unverified, LEFT + reported |

The brief's protect-list named `orpheus/transport/spatial/scheme.py:42` as "the quadrature
paper". `[M]` it is entry **(B)** verbatim, cited for *"Eq. 50 (dome recursion) and Eq. 74
(Morel–Montry) feed the curvilinear cell update"* — i.e. the exact retracted claim. Fixed,
and the disagreement reported with the two entries side by side. What licensed the override
was the brief's OWN governing rule: *"Only the DOME-RECURSION attributions are wrong. Read
each hit and decide by what it asserts."* An enumerated list is a provisional triage; the
rule outranks it.

⟹ **an author-year collision across two of one author's papers is invisible to every grep.**
Build the triage on the JOURNAL + TITLE, and expect the brief's site census to be a sample
(mine found 26 candidate lines vs the brief's ~12; 9 were a whole class the brief had not
separated).

### 2. ⭐⭐ MEASURE whether a dangling `:eq:` warns before deciding rename-vs-keep — it DOES, which flips L-063's caution

Throwaway 2-file Sphinx project, positive + negative control, ~10 s:

```
plain : build succeeded, 1 warning.  WARNING: equation not found: no-such-label-anywhere [ref.eq]
-W    : EXIT=1, build finished with problems
render: the LIVE label emits href="#equation-live-label"; the dead one appears as raw text
```

So `:eq:` is in the **gated** class with `:doc:`/intra-doc `:ref:`, NOT in the silent class
with `:func:`/`:class:`/cross-doc `:ref:`. L-063's third fate ("KEEP + note, because renaming
risks a silent break") was argued from a label with **8 cross-doc `:ref:` citers**. With
`:eq:` citers only, the build catches every miss ⟹ **RENAME is safe.** Never carry that
caution across ref-role classes without re-measuring.

### 3. ⭐ Name the OBJECT, not the paper, in an eq-label

`bailey-dome-recursion` → `alpha-dome-recursion` (8 sites, 4 files, guarded by exact length
arithmetic + `out.replace(NEW,OLD) == src`). **A label naming a citation is a latent
staleness bug by construction**: attributions get retracted, equations do not. Checks run
before adopting the name, in this order:
1. `grep tests/` for the OLD name → **0** ⟹ no `verifies()` edge to orphan.
2. `grep` the NEW name across `docs orpheus tests tools` AND the prose corpus
   (`.claude/lessons.md`, `plans/`, `agent-memory/`) → **0** (plan-authoring §1: a free name
   can be free *because it was rejected*).
3. Family fit: siblings are `alpha-recursion`, `alpha-cylindrical`, `alpha-dome-closure`,
   `sn-alpha-dome-closes`.
4. Move the `.. vv-status:` sentinel **in the same edit** (L-027) and let `matrix.rst`
   regenerate — `[M]` the diff was exactly one row moving alphabetically, no hand-edit.
   Verified with `_scan_theory_equations(Path('docs/theory'))`: old label absent from
   `all_labels` AND `documented`, new label present, 0 violations.

⚠ **And I found a genuine two-labels-one-equation duplicate**: `alpha-recursion`
(`curvilinear_one_group.rst`, **the** `verifies` target, 115 tests) states the same recurrence
as `alpha-dome-recursion` (`structured_geometry.rst`, `documented` sentinel, 0 tests). Brief
said *say so and recommend, do not collapse* — right call: collapsing moves a generated
V&V-matrix row and re-points markers a docs pass may not touch. Published as a `.. note::
**Two labels, one recursion.**` naming the **register** each page owns (geometry-primitive vs
discretisation) — the L-064 "name the register, not just the fact" move applied to a twin.

### 4. ⭐⭐ For a "family X does not use Y" claim, the load-bearing evidence is a CAPABILITY THAT EXISTS AND DECLINES Y — never an absence

*"MoC and CP have not migrated yet"* and *"MoC and CP never form this term"* both predict
zero hits, so a census alone cannot separate them, and the first reading **licenses work**
(go wire them up). What separates them is a positive fact: `[M]`
`orpheus.moc.geometry.MOCMesh` ray-traces **concentric annuli** on a **cylindrical**
`Mesh1D` (`_ray_circle_intersections`); `CPSolver._setup_spherical` is a real **sphere**;
`MCMesh` admits a real **cylinder**. Three shipped curvilinear capabilities, zero α. An
*absent* capability could never have refuted the migration reading.

⚠ My first draft used "CP ships a sphere and MC ships a cylinder" as "the two curvilinear
counter-examples" for a claim about **MoC and CP** — MC is not in the claim, and MoC's own
counter-example (the annuli) was the strongest one and I had not looked for it. Caught in the
self-consistency pass. ⟹ when refuting a claim about {A, B}, the counter-examples must come
from {A, B}.

### 5. ⭐ A structural claim publishes as an IFF with numbered conditions + a per-family adjudication table + a "what WOULD change this answer" note

The α dome is needed **iff** (1) an angular unknown survives discretisation with a direction
index, (2) the index is read in a **local rotating frame**, (3) its derivative is
**collocated**. MoC fails (2) — Ω is fixed in the global frame, `Ω·∇ = d/ds` is chart-free,
curvature relocates into segmentation (`[M]` `moc/core.py` forms
`τ = Σ_t·ℓ_seg/sinθ_p` and attenuates; no ordinate touches its neighbour). CP fails (1) —
angle is integrated into the kernel first (`[M]` `F(τ)=e^{-τ}` sphere, `Ki_3` cylinder, `E_3`
slab). MC fails (1) — directions are sampled, not indexed.

The **"what WOULD change this answer"** note is what makes it falsifiable rather than an
assertion, and it pre-empts the "so it's just not built yet" re-reading: a DG/FE-**in-angle**
or spherical-harmonics scheme satisfies 1 and 2 and fails 3, and would need a mass/stiffness
pair in μ, not this recursion. None exists here.

### 6. ⭐⭐ The CONTROL column of a census table you PUBLISH moves under your own edits

I drafted the control row from a **pre-edit** census (`sn 36/66/16/66/44/2`,
`geometry 33/122/62/46/3/13`). My own ⛔ tombstones name the module, so post-edit it is
`sn 36/66/16/67/44/3`, `geometry 36/124/56/50/3/14` — the table would have shipped
unreproducible against its own tree. ⟹ **re-measure a published census AFTER the last edit**,
and prefer the **file list** (stable) over the raw count for the load-bearing universal:
*"only twelve files in `orpheus/` name the module — the module itself + `geometry/__init__.py`;
six under `sn/`; four under `transport/spatial/`; and one derivations file"*.
⚠ Same pass, same defect class: my first universal read *"every consumer lives in
`orpheus/sn/`, `orpheus/transport/` or `orpheus/derivations/`"* — it omitted
`orpheus/geometry/` (the module's own package, 3 hits in `__init__.py`). A universal written
about "consumers" silently excluded the home package.

### 7. ⭐ The bold-swallowed role — probe the three repair idioms, don't reason

`**per-:math:`\mu`-level**` (2 pre-existing sites on the page I was editing) ships the LITERAL
characters `:math:`mu``: RST does not nest inline markup, so the role dies inside `**…**` and
the LaTeX backslash is eaten. Silent at every severity; the rendered HTML is the only
instrument. `publish_doctree` on four one-liners settled it in one call:

| form | `astext()` |
|---|---|
| `**per-:math:`\mu`-level**` | `per-:math:`mu`-level` ⛔ dead |
| `**per-**\ :math:`\mu`\ **-level**` | `per-\mu-level` ✅ bold AND role |
| `per-:math:`\mu`-level` | `per-\mu-level` ✅ |
| `per-:math:`\mu`-**level**` | `per-\mu-level` ✅ but odd emphasis |

Repaired with the escaped-seam form (preserves the author's bold). ⚠ The other class on the
same page — `**``1.016389``**` numeric cells, and `` ``source_iteration``** `` on
`curvilinear_numerics.rst:1966` — is cosmetic (a stray backtick) and belongs to **#379**;
LEFT and reported. **Rank a dead role above a stray backtick: a dead `:math:` is a missing
equation.**

### 8. Tense triage — the three registers, worked

| register | example | repair |
|---|---|---|
| present-tense-FALSE | "shared by SN, MoC, and CP curvilinear sweeps" | rewrite + ⛔ tombstone quoting the old text with its date |
| aspirational/forward, now refuted ON THE MERITS | "MoC and CP campaigns (post-Wave-1) reuse this primitive" | ⛔ **retracted … closed as NOT APPLICABLE, not as pending** — the distinction is the whole point |
| stale REASON on a surviving FACT | "lives in geometry **so MoC and CP can consume it**" | keep the instruction, **replace the reason** ("because it is CHART data") + ⛔ note |
| correct HISTORY | `reduced_operator.py:12` "— **not** 'Bailey 2009', the wrong-paper citation…" | LEAVE |

⚠ The aspirational row is the one that needs judgement. "Leave it, it's only a plan" ships a
plan that a future session will execute. Closing it **NOT APPLICABLE with the structural
reason** is what stops that.

### 9. What I found and did NOT fix (each with its reason)

- **9 entry-(A) quadrature sites** (`registry.py:280,470`, `rules_product.py:85,578`,
  `rules_sphere.py:164,175,271`, `directional.py:292`, and the doc mirror
  `discrete_measures.rst:955`). A *different* bibliographic entry from the retracted one,
  cited for an ordering convention. Whether ANE 35:1929-1936 exists and has an Eq. 50 needs
  the paper → `literature-researcher`. **Directive 4: demand, don't guess.**
- **3 `tests/` sites** — I do not edit `tests/`.
  `test_sph_sweep_regression.py:60` is the clearest (`α_{n+1/2} = α_{n-1/2} − w_n μ_n
  (Bailey et al. 2009 Eq. 50)` = entry (B) verbatim).
- **`**``literal``**` nested-markup cells** — #379's class, corpus-wide, cosmetic.


### L-070 addendum — the census reconciliation (same day, coordinator-raised)

The coordinator could not reproduce the CONTROL counts I published in the
`reduced_operator.py` docstring (`36/66/16/66/44/2`, `33/122/62/46/3/13`) and offered three
candidate causes. **All three were live at once**, which is why the numbers looked plausible:

1. **PRE-EDIT.** Measured before my own ⛔ tombstones landed — and those blocks *name the
   module*, so the correction raised several of the very counts it published. Current values
   of that same partition: `36/66/16/67/44/3`, `36/124/56/50/3/14`.
2. **PARTITION MISMATCH — the primary defect.** The prose listed six spellings
   (`reduced_operator`, `ReducedStreamingOperator`, `AngularRedistribution`, bare `alpha`,
   `delta_A`/`face_areas`, `redistribut*`) while the numbers came from a **different** six:
   the first three collapsed into ONE family regex, plus two columns — `.reduced` and
   "connection coefficient" — that the prose never names. So the 5th and 6th numbers belonged
   to spellings no reader could see, and the first three spellings had no number at all.
3. **CONFIGURATION.** Mine was `re.I` and unanchored, so `redistribut` absorbed every
   `AngularRedistribution`: **67** where the coordinator's `\bredistribut\w*` reads **56**.

⭐⭐ **The rule: a POSITIVE CONTROL must be NON-ZERO — its particular value carries no part of
the argument — so freezing it is pure liability.** The zeros are the finding (falsifiable: the
day one stops being zero the claim is refuted); the controls are an instrument check. Publish
the **predicate**, not the table (`plan-authoring` §9). Applied here: the census `list-table`
became a `.. code-block:: python` carrying the exact 8 patterns, the root, the
occurrence semantics, and its own `assert`s — controls non-zero, subjects zero — under a new
`~`-level section `Reproducing the census — the predicate, not a table of counts`.

⭐⭐ **And I had minted a TWIN SOURCE inside one pass**: the page's `.. important::` block named
one "six independent spellings" set and the census table's column headers named a *different*
one. Two definitions of the same instrument, 900 lines apart, both mine, same afternoon. ⟹
when an evidence set is cited more than once, define it ONCE in a labelled block and make
every other site a `:ref:` pointer.

⭐ **Adopting the coordinator's exact patterns was the right move, not a concession**: two
independently-vocabularied instruments that agree is the acceptance evidence (L-067/L-052).
I kept their six verbatim and *added* the two paraphrase spellings theirs lacked (`.reduced`,
`connection[ -]coefficient`) — grepping the concept's paraphrase is the point (L-054).

⭐⭐ **RUN A PUBLISHED RECIPE AS PUBLISHED.** Extract the code block back out of the `.rst`,
`compile()` it, and `exec` it — its own asserts are the verdict. A recipe that does not run is
the same defect class as a number that does not reproduce, and nothing in the build checks it.
`[M]` 25 lines extracted, compiled, executed, asserts passed.

⚠ **Two more frozen counts fell out of the same sweep, both mine, both wrong:**
- *"referenced by 12 files under `orpheus/sn/` and 8 inside `orpheus/transport/`"* → replaced
  by its predicate (verified as published: controls 141/135, four subjects 0).
- *"Only twelve files in `orpheus/` name the module"* → `[M]` **thirteen**, and my own prose
  enumeration in the same sentence summed to 13. A frozen count contradicting its own adjacent
  enumeration. ⟹ **prefer an ENUMERATION to a COUNT** — a list can be checked by reading it.

⛔ **And the self-inflicted build failure, which my own digest already warned about:** the new
`.. _connection-coefficient-census:` anchor sat above a **paragraph**, so a bare `:ref:` has no
title to derive → 3 × `WARNING: … A title or caption not found … [ref.ref]`, EXIT=1. Fixed by
promoting it to a real titled `~`-level section (ladder verified: `=` 5, `-` 339, `~` 535, all
before the new section). ⭐ The build DOES catch this class — it is the intra-doc `:ref:` case,
not the silent cross-doc one — so the cost was one build, not a shipped dead link.

⚠ **Scope discrimination when the coordinator edits concurrently:** the porcelain flag is not
authorship. `git status` showed `angular_differencing.py` + 4 `tests/` files modified; a
signature grep matched 3 of them, but the only shared token was the **date** `2026-08-27`.
Reading the matched lines settled it (their prose: *"Lathrop's Eq. 25"*, which I never wrote).
⟹ discriminate by CONTENT, and pick a signature that is not a date.


### L-070 addendum 2 — the chimera citation re-point (2026-08-27)

A `literature-researcher` settled the entry-(A) question I had reported as "found but NOT
fixed": the record *"Bailey, Adams, Yang, Zika (2009), Ann. Nucl. Energy 35, 1929-1936"*
**does not exist**. 9 sites re-pointed / retracted. Gates: `-E -W` EXIT=0, 0/0/0.

### 1. ⭐⭐ Before minting a citation to EQUATION N, grep the corpus for what it already says about EQUATION N

The brief characterised BMC 2010 **Eq. (52)** as *"the η-ascending level ordering /
per-ξ-level edge-cosine recursion"*. True — and Eq. (52) states **two** things, and the
corpus already carries a **measured refutation of the other half**:

> `structured_geometry.rst` §`sn-tau-absorber-provenance` + `sn/angular/closure.py:1033-1070`
> (Q5.6.4, 2026-08-11): imposing Eq. (52)'s *partition* (a cell's η-measure equals the
> ordinate's weight) violates P3 on the shipped rule — ordinates outside their own cell go
> **0/4 → 4/8 → 12/16 → 28/32** at `n_φ = 8/16/32/64`, the solve **diverges (NaN)** from
> `n_φ ≥ 16`, and the mismatch ratio WIDENS with refinement (`[0.5858, 1.4142]` →
> `[0.0770, 1.5683]`). *"BMC Eq. (52) is not a law; it is the statement that in THEIR
> quadrature the weight equals the cell's η-measure."*

Citing Eq. (52) bare at the three ordering sites would have imported, at
`confidence = 1.0`, a claim the corpus explicitly refutes — and it would have contradicted
`structured_geometry.rst:634`, which already cites `:cite:`BaileyMorelChang2010`` Eq. 52
for exactly that refutation. ⟹ every Eq.-(52) citation I wrote names the **ORDER half
only** and points at the refutation. This is L-060's census-before-repair rule applied to an
**equation NUMBER** rather than to a formula: the corpus's prior reading is the constraint.
⭐ The two readings turned out to be the *same equation* (accumulating a level's weights
from `−sinθ` to `+sinθ` gives `Σw̄ = 2 sinθ`; BMC's `√(1−ξ²)` is ORPHEUS's `sinθ` because
their axis letters are the mirror of ours) — so the brief was right and incomplete, which is
the harder case to catch than a brief that is wrong.

### 2. ⭐ A fictitious citation is a RE-POINT when its equation numbers are right — and the tombstone is worth more than the fix

Every field traced to a *different real* publication (title → LLNL-CONF-407632 (2008);
authors → the already-retracted JCP 227; `ANE 35, 1929-1936` → Zio & Zoia 2008;
year 2009 → nothing). That is precisely why it survived two prior citation audits, so the
field-by-field origin table is the durable artefact. ⭐ And the cheap self-refutation:
**(author, year, volume) is over-determined — a journal volume pins its year**, so
"vol. 35" (= 2008) refuted "2009" before any lookup. Worth running on any citation
carrying all three.

### 3. ⭐ ONE canonical record, N pointers — applied the round after learning it

The BMC equation-number map (Eq. 11 sphere-α / Eq. 50 R-Z-α / Eq. 52 edge-cosine
accumulation / Eq. 74 M-M τ), the ⚠ **published-typo** warning (Eq. (50)'s printed RHS is
self-referential — corrected against the correctly-printed spherical twin Eq. (11)), and
the Eq.-(52) scoping note live **once**, at a new `:ref:`bmc-equation-map`` beside the two
existing Corrections. Nine sites point at it. Without this the typo warning alone would have
been copied 5×, and the next correction would have had 5 places to miss.

### 4. ⭐ "Use the bib key" is a RECORD instruction, not a RENDERING one — match the page's convention

`discrete_measures.rst` carries a plain-text `References` section and, pre-edit, **zero**
`:cite:`. Minting the page's only `:cite:` would have put two citation systems on one page.
Resolution that honours both: plain-text inline + a full entry in the page's OWN References
block. Report the deviation and its reason; the coordinator wanted the right *record*, which
plain text names just as well.

### 5. ⚠ When the cited claim is a CONVENTION, the fix is a RETRACTION — there is no equation to re-point to

`directional.py:292` credited *"Bailey 2009 / Hébert convention"* for the axis assignment
`(η, ξ, μ) = radial, azimuthal, axial`. `[VERIFIED ON SCAN]` **both** sources use the
opposite (Hébert (3.152)/(3.157) p. 91; BMC Eq. (48) p. 156: μ = radial, η = azimuthal,
ξ = axial). ORPHEUS may name its own axes; it may not credit the naming to sources using the
other one. So the honest edit deletes the attribution, says why, and **leaves the convention
prose untouched** — whether any arithmetic depends on the assignment is a separate,
unperformed audit, and saying so in the tombstone is what stops the next reader "fixing" the
axes.

### 6. Mechanics that paid

- **Per-edit `count(old) == 1` + assert-before-write over a 10-edit batch**: one guard fired
  (title 67 code points, underline 66) with the tree **untouched** — no `git checkout`
  recovery needed.
- **AST doc-only proof per production file** answered the coordinator's explicit question
  before it could become a dispute: all four `orpheus/numerics/quadrature/*.py` are
  DOC/COMMENT-ONLY vs `HEAD`. Pair it with `python -W error::SyntaxWarning` + `py_compile`
  + `import` — a docstring edit that adds `\b`/`\e` to a NON-raw docstring is a real
  `SyntaxWarning` the `-W` sphinx gate never sees.
- **Residual grep read by TENSE**: every surviving "2009" in the quadrature package is inside
  a ⛔ tombstone. That is the acceptance criterion, not zero hits.


---

## L-071 — P4.9a docs sweep: a RED baseline is a gift, and a declaration is a measurable act (2026-08-28, branch `refactor/unweld-p49a-closure-owns-march`, commit `ca852c44`)

**Task.** Teach five theory pages a landed carve: `DiamondDifference.update` stopped
applying the Morel–Montry angular march; `cell_balance_terms`/`CellBalanceTerms` retired;
the L2 visit family (`CellVisit.tau/c_in/c_out`, `UpstreamState.angular_upstream`,
`CellResult.outgoing_angular_state`) went purely spatial; the closure began minting its
own scan constants. Brief named 4 files / ~41 sites and 3 `.. implements::` blocks.

### 1. ⭐⭐ The `-E` BASELINE WAS RED, AND THE FOUR WARNINGS *WERE* MY WORKLIST

`[M]` pre-edit `-E -W` = **4 warnings, EXIT≠0**, all `nexus.directive`
"*the ontology refuses this edge — source is 'unresolved'*", one per dead `:by:` target.
A retired `:by:` DOES warn — so the `.. implements::` half of a retirement sweep is in the
**gated** class, unlike every prose role.

Two consequences worth carrying:
- **The acceptance gate becomes `4 → 0`, not "count unchanged".** Far stronger evidence,
  and free. A brief saying "verify `-W` clean" was describing an end state, not a baseline;
  measuring the baseline is what turned a vague instruction into a checkable predicate.
- **The build's own warning set is a better census than the brief's grep.** The brief named
  three blocks in `index.rst`; warning #1 was in `operator_algebra.rst`, on the label
  `streaming-action-cell-balance`, which no grep for the three named labels would find.
  ⟹ **on any retirement touching declarations, run the `-E` baseline FIRST and read its
  warnings as the site list.**

`[M]` the edge count is a second, independent check: `directives: wrote N edges` went
**400 → 412**, and +12 is exactly what I added (+1 slab migration, +2 degenerate, +5
`pole-mm-recurrence`, +2 `dd-mm-angular-recurrence`, +2 `dd-mm-scan-split`). Arithmetic
that reconciles is the cheapest proof a declaration pass did what it claims.

### 2. ⭐⭐ THREE FATES FOR A DEAD `:by:`, AND THE DISCRIMINATOR IS THE *SURVIVOR'S* STATE

Not one repair — three, and picking wrong is silent under-coverage (L-059: declaring
1-of-N switches inference off for the whole equation).

| survivor already declared on this label? | fate | count |
|---|---|---|
| **no** | **MIGRATE** the edge to it | `dd-slab-scalar`, 9 → 9 |
| **yes** | **REMOVE** — the retirement collapsed two implementers into one | `dd-curvilinear-scalar`, 6 → 5 |
| yes, **but the equation names arithmetic the retirement RELOCATED** | remove + **ADD the new home(s)** | `dd-cylindrical-degenerate`, 3 → 4 |

⟹ **check whether the survivor is already on the label before writing either edit.** One
grep. The migrate case is the dangerous one — it *looks* like a removal.

### 3. ⭐⭐ ADJUDICATING "does `X` still implement this?" — READ WHAT THE EQUATION STATES, NOT WHAT `X` LOST

The brief asked me to rule on `DiamondDifference.update` for `dd-cylindrical-degenerate`.
The tempting answer is "the march left, so drop it". Wrong, and the discriminator is
mechanical: **the equation's body**. `dd-cylindrical-degenerate` states
`denom = (ΔA/w)c_out + Σ_t V` and `numer = QV/W + (ΔA/w)c_in ψ_{n−1/2}` — a **balance**.
`update` still forms that quotient. What it stopped evaluating is `ψ_{n+1/2}`, which
**this equation does not write**; that is `dd-mm-angular-recurrence`, a different label.

⭐ And the mirror move, which is what makes the ruling complete rather than merely
defensible: **a relocation that moves a PRODUCT out of a callee and into its caller owes
the caller a declaration.** The equation writes `(ΔA/w)·c_out` and `(ΔA/w)·c_in·ψ` as
products; post-carve those are formed *only* in the walk (`_OneDimScanWalk._run`), which
passes them down assembled. Declaring only the balance sites would leave the equation's two
most geometry-specific factors implemented by nothing.

### 4. ⭐⭐ A "SINGLE PRODUCTION SPELLING" RULING IS A DECLARATION OPPORTUNITY WITH A MEASURABLE PAYOFF

`[M]` before: `dd-mm-angular-recurrence` carried **32 inferred implementers, every one
matched on the token `angular`** — a membership list of `orpheus/sn/angular/`, containing
`AngularBoundaryFlux`, `AngularTraceSpace`, `alpha_dome`… none of which march anything.
`pole-mm-recurrence` carried **1**, `_OneDimScanWalk._ensure_pole_mirror`, via `pole` — a
method that mirrors pole *faces*.

After declaring: **32 → 0** and **1 → 0**, with 5 `verifies` edges on `pole-mm-recurrence`
untouched. ⟹ when a carve rules "X has ONE production spelling", the label naming X is
almost always sitting on a pile of token guesses; the ruling is the moment to retire them.

⚠ And get the SET right (L-059): `pole-mm-recurrence` has **two lines** (seed + step), so
its five implementers are the step function, the batch kernel that writes the seed AND
loops the step, the public exposure the 5 `verifies` tests actually call, the mesh-bound
wrapper, and the per-cell entry. Declaring only `march_psi_half_step` would have refuted
the tests, which run `compute_psi_half_per_level`.

### 5. ⭐⭐ THE DIRECTIVE RESOLVER TAKES `py:data:` TOO — L-060's PREFIX LIST WAS INCOMPLETE

L-060 recorded the `:by:` resolver as trying *"the literal string, then `py:function:` /
`py:method:` / `py:class:` and **nothing else**"*. `[M]` REFUTED: a pre-existing
`:by: …diamond._DD_W` (a module-level `float`, node id `py:data:…`) binds fine and drew
**no** warning in either build. My pre-flight checker flagged it as unbindable — a false
positive from the frozen list. ⟹ pre-flight `:by:` against the **build**, or accept that a
4-prefix check over-reports; the authoritative answer is the warning.

### 6. ⭐⭐ A SINGLE-DRAW BIT-EQUALITY STATISTIC, RELAYED FROM A MEMO INTO **FIVE CODE SITES**

The brief instructed me to quote *"59 % bit-equal / max 204 ULP, `scratch/…§F2`"*.
`[M]` mine, same fixture (`folded_product(4,6)` cylindrical τ), **200 seeds × 2400 evals**:

| quantity | measured | stable? |
|---|---|---|
| bit-equal fraction | **46.21 – 51.42 %** (mean 48.66) | draw-dependent — publish the BAND |
| `max |A−B|`, 4.8e5 evals | **1.776e-15** | ✅ reproduces the memo exactly |
| `max` ULP, same 200 seeds | **113 – 91 839** | ⛔ 3 orders of magnitude — NOT a statistic |
| τ bitwise ½ ⟹ bit-equal | **100 %**, 0 ULP; `2 of 12` ordinates | ✅ structural, draw-free |

So **59 % lies outside the band and 204 sits at the bottom of the ULP range** — and the
figure had been copied into `closure.py:520`, `closure.py:1387-1388`, and **two test
docstrings** (`test_pole_angular_closure.py:513,565`). Five sites, one un-reproducible draw.

⭐ The transferable half is the *decomposition*, not the refutation: **an absolute
difference is a property of the fixture; a bit-equal FRACTION and a ULP gap are properties
of the draw.** The ULP metric explodes wherever the two terms nearly cancel while `|Δ|`
stays at the round-off floor — so it is the *worst* of the three to freeze in a docstring,
and it is the one people reach for because it sounds precise.

⟹ publish `max |Δ|` as the number, the fraction as a band, the structural cause (τ = ½
exactly ⟹ `1/τ = 2.0` and `(1−τ)/τ = 1.0` are exact) as the explanation — and say so in the
page when the tree's own docstrings carry the unstable form (I wrote a `.. warning::`
naming the docstring, since I cannot edit `orpheus/`).

⭐ Also: `[M]` `fp(4,6)` carries **six distinct float64 τ**, not three — three nominal
values each as a 1-ULP-apart pair, of which only one member of one pair is exactly ½. A
bit-identity claim validated on "the τ = ½ ordinate" is reading a coin (`vv` #31/#13).

### 7. ⭐ THE HONEST-SCOPE NOTE THE BRIEF'S HEADLINE WOULD HAVE LOST

Brief item 1: *"The march has ONE production spelling."* True **and scoped**: `[M]` the
scan-normal form (`τ⁻¹ψ̄ − ((1−τ)/τ)ψᵃ`) survives as the closure's minted constants and is
consumed at 3 sites (forward + two transpose arms). What P4.9a achieves is
`grep -c "1.0 - tau" orpheus/transport/ == 0` and **one owner**, not one spelling.

The forms **partition the ordinate set** (A on degenerate + batch, B on non-degenerate), so
no input is ever evaluated both ways — which is what makes "welded by gate, not unified by
spelling" a design rather than an excuse. `[M]` `march_psi_half_step` has exactly **2
callers**; that closes the population.

⟹ the production docstring already carried the honest scope. **Read the owner's docstring
before writing the headline** — the code was more careful than the brief's summary.

### 8. ⭐⭐ NAMING THE *FORCING* IS THE LOAD-BEARING CONTENT OF AN UN-WELD DOC

A reader who takes a Pattern-2 twin for carelessness will re-introduce it. `[M]` by AST over
`tests/test_layer_imports.py`: `transport` ∈ L2, `sn` ∈ L3,
`FORBIDDEN_EDGES["transport"] = L3_PACKAGES`, enforced per module by a
`@pytest.mark.foundation` parametrized gate. **The scheme could not call the closure — it
could only re-spell the relation.** So the repair is not "delete the copy", it is "move the
responsibility to the layer that sees both". That sentence is the whole doc.

⟹ for any un-weld: **grep the layer contract and quote the forbidden edge.** It converts
"someone duplicated this" into "the architecture manufactured this", which is the only
framing that survives.

### 9. ⭐ A GUARD RE-KEYED FROM A PRESENCE-TEST ONTO A VALUE SIGNAL IS A DOC-WORTHY UPGRADE

LD refused curvilinear visits by `upstream_state.angular_upstream is not None`. Retiring
that field would have left the guard **silently unreachable** (`vv` #28's temporal twin —
a defaulted presence-test with nothing to detect). Re-keyed onto `face_area_inner !=
face_area_outer` **or** a non-neutral assembled contribution. Say *why* that is stronger,
not just that it changed: a value-keyed guard is reachable by calling the scheme directly,
so its witness needs no mesh and no earlier guard can preempt it.

### 10. ⭐⭐ THE `.. implements::` BODY PLACEMENT RULE BIT TWICE, IN THE SAME SESSION

L-060's rule (place after the `.. math::` unless the next paragraph is a grammatical
continuation) is easy to *know* and easy to *violate*, because you write the directive
while thinking about the equation, not about the sentence. `[M]` I placed both new blocks
mid-sentence — *"…does need the raw τ: [eq] [DIRECTIVE BODY] and read it from…"* and
*"…precomputed the split [eq] [DIRECTIVE BODY] consumed at…"* — and had to splice both
back out. ⟹ **after writing any directive with a body, read the sentence that spans it
out loud.** A `where …` / `and …` / `consumed at …` / `with …` opener is the tell.

### 11. ⭐⭐ THE PATCHED XREF GATE, WITH AN END-TO-END POSITIVE CONTROL — AND A MEMORY CORRECTION

L-062/L-067's `head_role` bug is **still unlanded**. Working recipe, now with the control:
copy the gate to `scratch/<name>.py` — **depth 1 from the repo root**, since
`REPO_ROOT = __file__.parent.parent` (a `/tmp` copy AND a `scratch/_dir/` copy both scan
**0 files**, silently) — patch
`head_role = "mod" if "." in target else role`, run as a subprocess.

⭐ **The control that makes the negative believable:** write a throwaway `docs/_ctl.rst`
with two deliberately dead roles and one live one, run both gates, delete it. `[M]` stock
**0 dead**, patched **2 dead / 2 sites**, `decidable` 5310 → 5312 (exactly +2). Without
that, `DEAD TARGETS: 0` is indistinguishable from a broken scan.

⛔ **Memory correction:** L-062 recorded the patch taking `docs/` from 49 dead → 207 on a
pristine tree (2026-08-18). `[M]` today, corpus-wide over `docs orpheus tests`, 984 files
/ 16 068 roles: **patched = 0 dead**, same as stock. The corpus was cleaned in the interim.
Do not quote 207 as a live expectation.

⭐ And the two-instrument agreement (L-067): nexus `dead_references` (by RENDERED target)
**0 dead / 52 checked** and the patched gate (by IMPORT) **0 dead**. Independently
vocabularied, same answer — that is the acceptance evidence.

### 12. ⭐⭐ THE RENDER CHECK NEEDS A **PROVENANCE** STEP, OR IT INDICTS THE WHOLE PAGE

My render checker fired on `curvilinear_one_group.html` (32 visible backticks, 8 unrendered
`:math:` roles, 29 `<cite>`) and `operator_algebra.html` (40 / 2 / 5). **Every one was
pre-existing.** Proving that is the step L-069 does not name:

⟹ **extract each offender's source pattern and test it against `git show HEAD:<file>`.**
`[M]` 15 of 16 matched immediately; the 16th (`independent of :math:`\sigma_t``) was a
false negative from a **line wrap** — the source splits `**independent of\n:math:`…`**`,
so a single-line pattern misses it. Confirmed by line number instead (HEAD:811 →
current:815, absent from `git diff`). ⟹ when a source-pattern check says "mine", re-check
by LINE, not by string.

⭐ The cheaper primary evidence: **slice the rendered HTML to your own new section's id**
and count there. `[M]` mine: **0 visible backticks, 0 unrendered roles**, 2 tables, 2 code
blocks, 4 admonitions, and 4 live `href`s for the cross-doc `:ref:` I minted.

### 13. ⭐⭐ `<cite>` IS *NOT* ALWAYS A DEFECT — CHECK THE CORPUS CONVENTION BEFORE "FIXING" IT

L-061 calls `<cite>` the smoking gun of a Markdown port. My section had **6**, all
`` `[M]` ``. `[M]` corpus-wide the measured-marker is spelled single-backtick **184** times
vs double-backtick **110** — and `curvilinear_one_group.rst` itself carries **29** single
(21 pre-existing, 8 mine). So the italic rendering is the page's own convention, and
"fixing" my 8 would make my text inconsistent with the 21 above it.

⟹ L-061's rule is about *code spans that should be monospace*. **Before treating a
`<cite>` as a port artifact, count both spellings of that token corpus-wide.** (The 184/110
split is itself a real corpus inconsistency — report it, don't resolve it inside an
unrelated sweep.)

### 14. Sites the brief did not name (its census is always a sample — L-059)

- `operator_algebra.rst:760` — a 4th dead `:by:`, found by the **build**, not by grep.
- `history.rst:967` — a 5th file, live `:meth:` role on the retired `_make_cell_visit`.
- The B2/B3 "three live consumers" narrative — two of three consumers re-homed; the
  `.. code-block::` spelling `terms = cell_balance_terms(…)` is now un-runnable.
- The **L16/Pattern-5 rationale paragraph** (*"each consumer derives the trivial 1/τ…"*) —
  present-tense-false, and `cache.py`'s own comment records the revision. Classic L-069:
  the stale REASON outlives the stale NAME, and only the name is greppable.
- `index.rst`'s "Slab vs curvilinear discrimination" note, which pointed at
  `upstream_state.angular_upstream is None` as the *current* mechanism.

### 15. CODE-side, reported not fixed (docs-only brief)

1. `march_psi_half_step` / `advance_psi_half` docstrings + 2 test docstrings carry the
   un-reproducible `59 % / 204 ULP`, and cite the **untracked** `scratch/p4_9a_verification_plan.md`
   (L-048: describe the probe, never cite an ephemeral path).
2. `cell_balance.py` module docstring still describes slab as
   `alpha_in = alpha_out = 0.0, tau_mm = 1.0` — `[M]` those three `StreamingTerms` fields
   were deleted at #236 Step C; live fields are
   `{chord_length, mu, face_area_inner, face_area_outer, delta_A_over_w, volume, abs_mu}`.
3. `index.rst` documents `CellVisit.face_area_downstream: float | None`; live it is
   `float = 0.0`. Pre-existing, flagged in place.

### Gates

`-E -W` **0/0/0, EXIT=0** (baseline **4**) · xref gate 0 dead, patched gate 0 dead with a
2-of-2 positive control · nexus `dead_references` 0 dead / 52 checked · vv-status scan **0
violations**, new label `sn-p49a-march-forms` registered `documented` · declared edges 400
→ 412 (reconciled) · guesses 32 → 0 and 1 → 0 · render check on the new section clean ·
`matrix.rst` regenerated (10143 → 10155 tests; picks up the renamed
`test_cell_visit_c_stamp` → `test_closure_constant_map` and two new modules).

### Quality self-assessment (Directive 3)

Derivation depth **4** · Cross-references **5** · Numerical evidence **5** (the 200-seed
band + the τ=½ mechanism is strictly better than the single draw I was handed) · Failed
approaches **5** (the forced twin, with its layer contract) · Code traceability **5** (+12
declared edges, 33 guesses retired) · Derivation source **3** — the one weak dimension:
the Form-A/Form-B comparison is a **published recipe**, not a `derivations/` script. It is
a floating-point property of two spellings, which is arguably the recipe's natural home,
but a `derivations/` module would let a gate consume the band.

---

## L-072 — P4.9b: the operator poses with its two closures (2026-08-28/29)

**Task.** Teach the Sphinx corpus what landed in `b253732f..d14dd545`: a new
architecture section, a changelog row, and a rename sweep (`pole_angular_closure` →
`angular_closure`). Docs-only; `orpheus/` and `tests/` off-limits. Commit `9c3eb60a`,
branch `refactor/p4-9b-streaming-operator-poses`, 10 files, +823/−52.

### 1. ⭐⭐ A rename sweep's population has FOUR classes, and TENSE only separates two

The received rule ("past-tense history stays; present-tense-false is a MUST-FIX") sorts a
symbol retirement. It does **not** sort a *vocabulary* retirement, because two further
classes have nothing to do with tense:

| class | what it is | verdict | worked example |
|---|---|---|---|
| **vocabulary** | prose naming the live family / attr / kwarg / module / test file | UPDATE | *"produced solely by the pole-angular closure"* → *"angular closure"* |
| **period history** | a past-tense narrative naming the thing as it was named then | KEEP verbatim | the whole `curvilinear_numerics.rst` Phase-B/D/F chapter |
| **ADDRESS** | a section anchor / eq-label carrying the retired word | **KEEP, and say why** | `sn-pole-angular-closure-protocol` |
| **genuine referent** | the word still denotes the thing | KEEP | Hébert's *Carlson coupled-pole* seed; the sphere's polar cap; μ = −1 |

`[M]` on this corpus the split was **14 updated · 32 period-history · 9 address · 3 genuine**
(plus 7 lines I ADDED that deliberately name the old spelling, to record the rename).

⭐ **The ADDRESS class is the one a sweep gets wrong, in the flattering direction**: renaming
an anchor *feels* like completing the job, and the break is silent — a cross-document
`:ref:` that misses renders as plain text at every build severity (L-070's measurement:
only `:eq:` warns). `[M]` this anchor had **3 cross-document + 3 intra-document** citers.

⭐⭐ **And the page had already made this exact decision once, for a different word.** The
section's own "Contract evolution" note said: *"The section anchor … is retained (it is
cross-referenced from … and elsewhere); only the human label 'protocol' is now loose."* So
the right move was **extend the existing note**, not mint a new one — the second correction
reads as the same discipline applied twice rather than as two unrelated caveats. ⟹ before
ruling on an anchor, grep the anchor's own section for a note explaining why it is spelled
that way; a mature page has usually already been here.

### 2. ⭐⭐ A brief's population claim can be ZERO — and the surplus is where the work is

The brief asked me to sweep *"`StreamingOperator(sn_mesh)` ctor spellings in prose (~a
dozen in docs)"*. `[M]` a whitespace-flattened scan over every `.rst` (so a wrapped call
cannot hide) returns **0** — the corpus never carried the constructor spelling at all; the
only hits after my pass are my own new prose. Nothing was wrong with the brief's *concern*;
its *census* was a guess.

⭐ The same greps that refuted it found a defect the brief never named: the
`curvilinear_one_group.rst` **Key Facts** bullet still said τ *"[is] delivered to the
stateless spatial scheme as `CellVisit` **data** (c_in, c_out, τ), stamped at one
production site"* — `[M]` `dataclasses.fields(CellVisit)` is
`('cell_idx', 'streaming_terms', 'face_area_downstream')`; all three fields **and** the
stamping method were retired at **P4.9a**, the phase whose docs I had written the day
before. A Key Facts card is the highest-leverage stale surface on a page, and it went stale
against *my own* previous pass.

⟹ **run the brief's own census before writing to it**, and treat the delta both ways: a
zero means don't write that section, a surplus means the brief's author could not see it.

### 3. ⭐⭐ Reproducing a relayed mutation count caught MY harness first (vv #17 → #18)

The design record's headline for the no-guard ruling is *"a `pose` that MINTS fresh objects
reddens 5 rows, every one structural; no value assertion moves."* Reproduced in-process (a
pytest plugin monkeypatching `StreamingOperator.pose`; no file edited on disk):

* **First run: 9 reds.** Four of them were `AttributeError:
  'IdentityAngularClosure' object has no attribute 'redistribution_pairing'` — my mutant's
  Cartesian arm was simply broken, and its extra reds were the harness's, not the
  invariant's. That is vv #17 (a broken instrument) producing vv #18's symptom (a false
  *rich* verdict) in one probe.
* **In-class repair** — mirror `SNMesh._init_core`'s Cartesian arm exactly
  (`angular_redistribution(quad, coord)` + a zero pairing) so **only identity** changes.
  Re-run: **5 reds of 65**, exactly the relayed set — the pose-identity gate, the
  one-instance gate, and all three **closure** rows of the route gate on their activation
  legs. **60 pass**, including every `array_equal` pin. The slab/scheme row survives.

⟹ a mutation count is cheap to reproduce and you should, because the first run is usually
yours. And note which half reproduced: the *identity* of the red set, not just its size —
per `coding-standards`, an equal-sized but disjoint set is the failure mode.

### 4. ⭐⭐ Two relayed `[M]` numbers, one stable and one fixture-bound — publish both halves

The phase's perf argument: *"the operator is built 6–10× per solve, so a per-operator memo
costs up to 24.65 % of a slab solve (8.78 ms build, GL16/nx=200)."* Re-measured on a 2-group
fissile fixture I can name:

| quantity | phase's `[M]` | mine | verdict |
|---|---|---|---|
| one Stratum-1 build, GL16/nx=200 | 8.78 ms | **8.84 ms** (min of 5) | reproduces |
| operators per k-eigen solve | 6 (slab) / 10 (sphere) | **42 / 38 / 40 / 43** | fixture-bound — it scales with the OUTER count |
| consequence | +24.65 % | **+68 %** (42 × 8.84 ms on a 546.6 ms solve) | mine is 2.8× stronger |
| Stratum-1 builds per solve | 1 | **1** on all four | reproduces |

⟹ the *stable* halves are the per-build cost and the count's **scaling law**; the
percentage is a fixture reading. Publishing mine with its configuration made the ruling's
own case better than the number I was handed — which is the usual outcome of §L-057's rule,
and worth remembering as an incentive rather than a chore.

⭐ A second independent corroboration worth more than the relay: the phase's F7 reported the
scheme/closure activation asymmetry as counts on its fixture. Counting the two per-cell
entries myself over full solves gives **656 / 0**, **0 / 5 552**, **0 / 24 928** on
slab / sphere / cylinder — the **zeros are exact**, which turns "carry both geometries" from
advice into a theorem about the gate.

### 5. ⭐ The page I was told to append to carried a UNIVERSAL my row would falsify

`history.rst`'s header read *"Every entry below is **merged to main** — … a new entry lands
with its merge hash or not at all."* The task was to add a row for work on an unmerged
branch. Three options, and only one is honest: don't add the row (disobeys the task), add it
silently (falsifies the page's own header — the defect this whole discipline exists to
prevent), or **repair the universal to the convention the corpus actually runs**. The
sibling table in `operator_algebra.rst` already spells it: *"Entries marked (in development)
live on an unmerged feature branch … trust `git` over this table."* Adopting that wording is
internal consistency, not a weakening — the strong half ("trust git, never a frozen note")
survives verbatim.

⟹ **before appending to any list, read its own header for a universal your row would
break** — an index can contradict itself, and the contradiction lands in the reader's lap.

### 6. ⭐⭐ The BASELINE-DIFF render check needs no provenance reasoning

L-069 says the rendered page is the instrument for nested-markup defects, and L-068 says a
page-wide count indicts pre-existing prose unless you slice to your own section. There is a
cheaper general form: **keep the pre-edit `-E` build, and diff per-page (visible backticks,
unparsed `:role:` spellings) before vs after.** A delta of zero is the proof, with no
provenance argument at all.

`[M]` it earned its keep immediately: I wrote ``**``assert`` became a ``raise``**`` — the
exact `**``literal``**` nesting L-069 already records — and the diff caught **8 visible
backticks** on a page whose absolute count is otherwise irreducible. Fixed by taking the
bold off the literals (`**The Stratum-1 admission contract now raises.**`). Final state:
**0 regressions on all 10 pages**, and the new section renders 0 backticks / 0 dead roles /
5 tables / 3 code blocks / 6 subsections with all 4 internal links resolving.

⚠ The `<cite>` column moves and that is **correct**: `` `[M]` `` is this corpus's marker and
renders `<cite>[M]</cite>` (`[M]` P4.9a's section carries 6, mine 10). Do not "fix" it.

### 7. Verifying the section's own claims — three that changed what I wrote

1. **`SNMesh.__eq__`.** The design memo said the scheme type comparison is in `__eq__`
   (`augmented_mesh.py:587`). `[M]` `SNMesh.__eq__ is object.__eq__` → **True**, and two
   identically-built meshes compare **unequal**; the comparison lives in
   `is_same_phase_space`, whose docstring *also* says the angular closure is deliberately
   EXCLUDED. My draft note was flatly false and became a much better one.
2. **The published recipe.** The code block counting operators per solve was extracted back
   out of the `.rst`, `compile()`d and `exec`d — its own asserts pass. A recipe that does
   not run is the same defect as a number that does not reproduce (L-070).
3. **Field roles.** `hasattr(StreamingOperator, 'spatial_closure')` is **False** (dataclass
   annotation only), and `api/discrete_ordinates.html` carries **zero** `id="orpheus.sn.
   operators.streaming.*"` anchors — so fields are literals and the class role is the page
   convention (82 pre-existing `StreamingOperator` mentions agree).

### 8. CODE-side, reported not fixed (docs-only brief)

`[M]` **22** residual pole-vocabulary sites survive the mechanical rename — 16 in
`orpheus/`, 6 in `tests/`. Several are legitimate history (the retired `PoleAngularClosure`
**Protocol** as a proper noun); ~11 are **present-tense** descriptions of the LIVE family
and are the fix list:

* `sn/angular/closure.py:241` (family ABC docstring), `:2138` (`IdentityAngularClosure`),
  `:2261` **and `:2276` — the latter is a `raise` MESSAGE**, i.e. an API the moment a test
  pins it (grep the shortest distinctive fragment first).
* `sn/mesh/augmented_mesh.py:396` (section banner), **`:562`** (the `is_same_phase_space`
  docstring, quoted in my new section), `:734` / `:797` (both ctor docstrings).
* `transport/mesh/axis.py:470`, `transport/spatial/diamond.py:222`,
  `transport/spatial/linear_discontinuous.py:293`.
* `transport/spatial/scheme.py:1349` names `PoleAngularClosure.angular_adjoint` — a **dead
  class reference**, not just stale vocabulary.

### Gates

`sphinx -E -W` **EXIT=0, 0 WARNING / 0 ERROR / 0 CRITICAL / 0 SyntaxWarning**, identical to a
freshly-measured `-E` baseline (also **0** — the old "baseline 4" reading is void, re-measure
every session) · rendered baseline-diff **0 regressions / 10 pages** · nexus
`dead_references` **0 dead / 52 checked** · `check_docstring_xrefs` **0 dead / 985 files /
16 100 roles**, and a `head_role`-patched copy **0** too, proven live by a planted
2-dead-1-live control page (stock read 0, patched read 2).

### Quality self-assessment (Directive 3)

Derivation depth **4** (an architecture section, not a derivation — the closest thing is the
four-attack table and the lifetime argument) · Cross-references **5** · Numerical evidence
**5** (four independent re-measurements, each with its configuration; two of them
strengthened the phase's own case) · Failed approaches **5** (the four attacks, the refuted
"silent wrong k", the three silently-green keystone traps, and my own harness bug published
as a caution) · Code traceability **5** · Derivation source **3** — again the weak
dimension: the perf/count evidence is a **published recipe** in the page rather than a
`derivations/` module a gate could consume. Same finding as L-071; two sessions running,
which makes it a pattern rather than an instance.

---

## L-073 — a capability flip's staled DEFERRAL CONTRACTS: the census must be run at the CLASS, and the tree's own corrections are the model text

**Task** (2026-08-29, branch `docs/p0-record-and-carrying-prose`): a meaning-triaged prose sweep
after the Q5.6.3 cylindrical-admission flip (`1689faf4`, 2026-08-08) turned every ADMITTED cylinder
CARRYING. Prose only, zero behavior change, no commits.

### The measurements that mattered

- **I reproduced the brief's ground truth rather than adopting it.** Built one `SNMesh` per
  (chart × rule) and read the carrying state directly:
  `SLAB/GL4 → levels=(), space=None`; `SPHERE/GL4 → levels=(0,), _carrying=[0]`;
  `CYL/folded_product(2,4) → levels=(0,1), _carrying=[0,1]`; `CYL/product` and `CYL/level_symmetric`
  **REFUSED** by `assert_carrying_quadrature`; `CYL/lebedev`, `CYL/gauss_legendre` refused one guard
  earlier by `cylindrical_streaming`'s structure-less check. 8 rows, ~20 s.
- **⭐⭐ The brief's ground truth was RIGHT and INCOMPLETE, and the tree already said so.**
  `closure.py:1805-1829` carries a `[M]` 2026-08-26 census: `assert_carrying_quadrature` has
  **ONE call site**, inside `case CoordSystem.CYLINDRICAL`; **the SPHERICAL arm calls no admission
  gate**, so a μ = −1-noded (Gauss–Lobatto) sphere rule builds a production `SNMesh` and reaches the
  non-carrying branch **at 6 of 11 orders, over 75 reachable non-carrying levels**. So
  *"the slab is the only admitted non-carrying 1-D geometry"* is true **of the shipped `Quadrature`
  constructors** (there is no `Quadrature.gauss_lobatto` — `[M]` `dir(Quadrature)`) and false as a
  structural universal. Writing the unqualified universal would have licensed retiring a live branch.
- **Cheap confirmation the code-side fix is safe:** the three stale `raise`-message parentheticals
  (`"a seedless mesh (Cartesian, or a non-carrying cylinder, R12a)"`) are **NOT pinned** — `[M]`
  tests match only the OPENING clause (`"carries no starting-direction ray"` ×2,
  `"carries no radial-characteristic ray"` ×1); `grep "Cartesian, or a non-carrying" tests/` = **0**.

### The rules this earned

1. **⭐⭐ For a capability FLIP, the population is the CLASS the flip moved, and a co-occurrence
   window is the only filter that finds it.** A line-based grep for the retired pairing misses every
   instance the formatter wrapped (vv #21). I censused `non[-_ ]?carrying` (validated against
   `non-carrying` / `non_carrying` / `noncarrying` / `NON-carrying`) over `tests orpheus docs`
   minus `_build`: **151 hits / 42 files**; then a **±3-line window** co-occurrence with
   `cylind|\bcyl\b|_cyl|cyl_|CYL` split it **79 paired / 72 unpaired**. The paired half is the
   candidate set; the unpaired half is almost all correct general contract (`"``None`` on
   non-carrying meshes"`) and must NOT be touched.
2. **⭐⭐ The acceptance predicate is a QUALIFIER window, not a co-occurrence count — and the count
   goes UP when you succeed.** Post-edit the paired count rose **79 → 80**, because a correction
   names what it corrects. The gate that actually decides is *paired AND lacking a
   Q5.6.3/admission/refus/Until/unconstructible/⛔/HISTORY token within ±5 lines*: pre-edit that set
   was large, post-edit it is **3**, and all three are CODE (the `non_carrying_levels` def, its call
   site, and the `raise` f-string) — i.e. exactly the out-of-scope class. Publish that predicate,
   never the raw count.
3. **⭐⭐ In a flip sweep the tree is FULL of already-correct model text — adopt its spelling
   verbatim instead of inventing one.** Six sites already carried the right sentence
   (`augmented_mesh.py:874-880`, `loss_representation:4289`, `loss_representation:4740-4747`,
   `test_radial_characteristic_carrier.py:180`, `curvilinear_numerics.rst` ×4,
   `loss_representation.rst:2872`), and the main agent's already-fixed `test_assembly_mode.py`
   supplied the house phrases (*"since Q5.6.3 the slab is the only admitted non-carrying 1-D
   geometry"*, *"the constructible witness for this arm is a Gauss-Lobatto SPHERE rule — #415"*).
   Reusing them makes the sweep internally consistent for free and stops me minting a competing
   vocabulary.
4. **⭐⭐ A STALE HEADER over a CORRECTED BODY is the flip's signature defect — and the body is the
   evidence, so cite it rather than re-deriving.** Repeatedly a docstring headline asserted the dead
   claim while its own body, 20-30 lines down, already stated the flip: `test_psi_half_coupling.py`
   `:2928`/`:2983` said *"slab AND cylinder → 1×1"* over a body at `:3000-3012` that **builds a
   folded cylinder and asserts 2×2**; `loss_representation:4717/4721` said *"cylinder
   non-carrying"* 20 lines above `:4740` *"every level carrying (Q5.6.3 admission)"*; `:4331`
   described the #280 2.5b fold in the present tense 29 lines below its own `HISTORY` note saying the
   fold was retired. ⟹ **read ±30 lines around every candidate before drafting**: half the fixes
   write themselves from the neighbouring truth.
5. **⭐ A flip does not only stale "X is live" — it stales "X is UNTESTABLE" and "X is
   unreachable", and those read as settled facts nobody re-checks.** Two of the sharpest fixes were
   mirrors, not instances: `test_psi_half_coupling.py:2500` recorded *"a multi-carrying-level
   indexing bug is UNTESTABLE with current geometry (cylinder is non-carrying) — an inherited blind
   spot, noted not faked"* — `[M]` the admitted folded cylinder carries on EVERY level, so the
   fixture is now constructible and the blind spot is a **fixture gap, not a geometry limit**; and
   `curvilinear_one_group.rst:6723` said the edge-extrapolation inline *"is unreachable through the
   mesh"*, refuted by the sphere-Lobatto census. A sweep scoped only to "live X" claims misses both.
6. **⭐⭐ Prove "prose only" with an AST DIFF, not with a reading.** Two checks, ~10 lines each,
   both run before the build: (a) tokenize both revisions and compare the token stream with STRING
   *values* dropped — catches any code edit; (b) `ast.dump` after replacing every module/class/function
   docstring with `"<DOCSTRING>"` — proves **no `raise` message, no `match=`, no other literal**
   moved. Both returned identical for all 13 `.py` files. That is the only evidence that separates
   "I meant to touch only prose" from "I touched only prose", and it makes the `raise`-string
   exclusion auditable instead of promised.
7. **A pristine `-E` baseline built from `git archive HEAD` carries UNTRACKED-DATA artifacts —
   read the traceback before counting them.** The baseline read `2 warnings`; both were
   `Exception occurred in plotting infinite_medium-{1,2}`, `[M]` from `load_isotope("H_001")`
   failing because the nuclear-data files are untracked and so absent from the archive. The live
   tree builds **EXIT=0, 0 warnings**. Quoting `2` as the baseline would have made a clean build look
   like an improvement I did not make. (And `rm -rf` inside a compound Bash command is refused here —
   use a fresh `mkdir -p <newdir>` instead of clearing an old one.)

**Gates run:** `-E` baseline (HEAD archive) vs `-E` post-edit, both `EXIT=0`, WARNING/ERROR/CRITICAL/
SyntaxWarning set **2 (artifact) → 0** · `tools/check_docstring_xrefs.py orpheus tests docs` →
**DEAD TARGETS 0** across 16 101 roles · nexus `dead_references` → **0 dead / 52 checked** (the
two-instrument agreement, L-067) · `py_compile -W error::SyntaxWarning` on 13 files · production
imports · 230 tests collected · 35 passed / 4 xfailed on two touched modules.

**Scores:** derivation depth n/a · cross-refs 5 (5 roles added, every one owner-verified by AST) ·
numerical evidence 5 (8-row carrying matrix measured myself; 151/79/3 census numbers; commit hashes
verified as ancestors) · failed approaches 5 (three ⛔ tombstones written in place, never deletions) ·
code traceability 5 · derivation source n/a.

---

## L-074 — CS5 axis-generator doctrine: a page's own ASPIRATIONAL phrase becomes a LIE when the code ships the thing (2026-08-29, branch `feature/cs5-axis-generator`, commits `4e7b8977` + `b0bfc06c`)

**Task.** Archive campaign-1 phase CS5 — `Axis.generator` (provenance, never
identity), the `measure.axis(label)` / `quad.axis(label)` mints, the identity
exclusion, the rank-d seam, the `AngularMeasure` Protocol widening.

### 1. ⭐⭐ The page had ALREADY promised the feature — under a phrase that now means the OPPOSITE

`spaces.rst` opened with *"an axis carries exactly four things — an index
shape, a factor measure, a basis kind, **and the identity of the generator that
produced it**"*, and its four-slot table's fourth row was `identity —
structural, per subclass`. Pre-CS5 the phrase meant *"identity records what
KIND of generator produced this factor"* (an `EnergyAxis` is not an `Axis`).
Post-CS5 there is a real `generator` FIELD whose governing ruling is
**provenance is never identity** — the exact inverse of what the phrase now
reads as. The machine header's `role:` string and `foundations/index.rst`'s
summary carried the same words.

⟹ **When a landing change gives a real name to a phrase the page was already
using loosely, the phrase is now a MIS-STATEMENT, not a head start.** The fix is
not to delete it — it is to write the disambiguation as a `.. warning::` naming
both readings and saying which one CS5 installed. `-W` is silent; only reading
the intro against the new field finds it.

⚠ The tell is grep-able and cheap: `grep -rn "<the new field name>" docs/` BEFORE
writing, and read every hit as a claim about the NEW thing even when it predates it.

### 2. ⭐⭐ A REFUSAL in the page's "what was tried" section was half-falsified — and the reconciliation is the best content on the page

`spaces.rst` §"What was tried, and what refuted it" carried *"An `Axis` →
measure accessor — refused … the axis stays four slots and nothing more."*
CS5 adds a fifth slot AND makes the generator reachable from the axis — which
is literally what the refusal was avoiding.

`[M]` from the live tree, the two are compatible and the ARROW is why:
- the refused thing points **axis → measure** and would have had to
  **manufacture** its output (a pre-CS5 axis dropped the nodes, so the only node
  set it could synthesise is the index set — `[M]` `frame.py:~712`
  `nodes=np.arange(n)`, `support=f"index({label})"`);
- CS5 points **generator → axis** and manufactures nothing.
- `[M]` the collapse pair STILL builds its own index-space measure and never
  reads `axis.generator` — so nothing changed there, deliberately.

⟹ **preserve the refusal verbatim, move only its TENSE, and add a dated
`.. note::` whose content is the arrow-direction argument.** The refutation +
its reconciliation is worth more than either alone; and "the axis can now reach
a measure" invites exactly the wrong inference at the collapse-pair call site,
so BOTH halves must be stated together.

### 3. ⭐⭐ The gate's OWN roster said "EXHAUSTIVE … these are the four `Quadrature` classmethod factories" — `[M]` there are FIVE

`_RULES` in `tests/numerics/test_axis_generator.py` invokes vv-principles #31's
finite-roster corollary *by name* and lists `gauss_legendre / level_symmetric /
product / lebedev`. `[M]`
`[n for n,v in vars(Quadrature).items() if isinstance(v, classmethod)]` = **5** —
**`folded_product` is missing**, and it is the σ_y-folded cylindrical CARRYING
rule the curvilinear MMS case builders default to
(`derivations/continuous/mms/sn.py:2022`), i.e. the member with the richest
`level_indices` — the axis the roster exists to gate. `[M]` it works through the
mint (N=16, 4 levels, section law `True`).

⟹ **an "exhaustive over the shipped family" claim is a universal owing its
denominator — enumerate the family with `vars(cls)` / `isinstance(v,
classmethod)`, never from the roster's own list.** A roster that CITES the
exhaustiveness rule reads as having applied it (the plan-authoring
rule-vocabulary echo, in a test file). Reported, not repaired — I do not edit
`tests/`; and my own prose was corrected from "four shipped factories" to
"four of the five", with the gap named in the gate table.

### 4. ⭐ The simulated-inclusion probe: prove a design ruling WITHOUT mutating production

The commit claims the identity exclusion is *"structurally mandatory, not
taste"*. Reproducing it needs no edit to `axis.py` — a four-line **subclass**
appending `self.generator` to `_identity_key` shows it end-to-end:

```python
class _WithGeneratorInKey(Axis):
    def _identity_key(self): return (*super()._identity_key(), self.generator)
```
`[M]` `a1 == a2` → `ValueError: truth value of an array … ambiguous`;
`hash(a1)` → `TypeError: unhashable type: 'Quadrature'`. Root causes measured
separately: `Quadrature.__dataclass_params__.frozen is False` + `eq=True` ⟹
`__hash__ = None`; `DiscreteMeasure` is `frozen=True, eq=True` over ndarrays.

⟹ **when a doc must justify a NEGATIVE design ruling ("this field may never
enter the key"), simulate the rejected design in a subclass and publish the
traceback.** Safer than mutation (no production file is touched, so
`process-discipline`'s crash-unsafe-revert hazard cannot bite), and strictly
better evidence than restating the docstring.

### 5. ⭐⭐ A landed change can SILENTLY PRESERVE a published measured table — and that survival is publishable

`field_algebra.rst` carries a `[M]` 2026-08-24 fiber table whose top row reads
*"twin carrier → `angular_bulk_space ==` **True**"*. Post-CS5 each twin holds a
DISTINCT `Quadrature` instance, now recorded in the axis. `[M]` re-measured on
the same fixture: `a.quad is not b.quad` **True**, `a.angular_bulk_space ==
b.angular_bulk_space` **True**, hashes equal, moved-edge row still **False**.
Had provenance entered the key the row would not have flipped to `False` — it
would have RAISED.

⟹ **after any change that puts a new object inside a published table's subject,
re-measure the table and, if it survives BY DESIGN, say so with a dated note.**
A future reader who notices the new instance-carrying field will otherwise
assume the table went stale. The note also lands the page's own F2 doctrine
("compare space CONTENT, never provenance") one layer down, which is why it
belongs there rather than only on the spaces page.

### 6. ⭐ Two universals I published and had to correct, both by counting

(a) *"CS5 retired the literal path at the three shipped nodal mint sites"* —
`[M]` from the diff, **two** sites retired the literal
(`SNMesh.angular_bulk_space`, `MaterialMesh.bulk_space` rank-1 arm); the third
(homogeneous `_pose_space`) KEPT its literal and gained an honest-`None`
comment, and the rank-d arm keeps its literal by contract.
(b) *"every generator-ful axis in the tree today is nodal"* — a census claim I
could not run cheaply over every scheme; replaced with the **closure argument**
(`[M]` `hasattr(Basis, "axis")` is `False` and no subclass defines it; both
shipped mints are NODAL by construction ⟹ no MINT can produce a modal
generator-ful axis). Closure argument stays true as the tree grows; the census
would not (L-064).

### 7. The render check caught exactly one nested-markup failure, and my regex guard did NOT

`**The law's domain is MINTED axes, and a hand-passed ``generator=`` can lie.**`
— a literal in the MIDDLE of a bold run. My pre-write guard was
`assert "**``" not in text` (L-069's shape: literal at the START of the bold
run) and it passed. Only the built-HTML slice found it (4 visible backticks).
A regex for "literal inside a bold run" is **unusable** on this corpus — it
matches the closing `**` of one run to the opening `**` of the next and reported
119 false positives on one page.

⟹ **the HTML slice IS the gate; do not try to replace it with a source regex.**
Slice each new region between two distinctive phrases, strip tags, unescape,
and require `visible backticks == 0` and `surviving :role:` spellings `== 0`.
⚠ Anchor the slice with `rfind` for the START (the TOC repeats section titles —
a `find` gave a 204-char fragment that trivially "passed").

### 8. Gate inventory that worked, in order

1. `-E` baseline BEFORE any edit: `EXIT=0`, **0** WARNING/ERROR/CRITICAL/SyntaxWarning.
2. Own probes for every number published (5 probe scripts, deleted after).
3. `-E` verification build after each edit batch; acceptance = the severity SET
   unchanged (0 → 0).
4. HTML slice render check over **all 10** edited regions.
5. Import probe over **19** new fully-qualified roles → 0 dead.
6. `tools/check_docstring_xrefs.py docs orpheus tests --quiet` → `DEAD TARGETS: 0`
   (986 files / 16 180 roles / 13 485 decidable).
7. nexus `dead_references` → `0 dead / 52 checked`.
8. Anchor/link census: every new `.. _label:` has **exactly 1** anchor and **≥1**
   inbound link (three started at 0 links — decorative anchors — and were given
   real citers rather than left as leads).
9. V&V matrix: sentinel count **565 → 567** (exactly my two labels), orphan count
   unchanged at **2**, both new labels in *Documented-only*.

### 9. Reported upward, code-only (I do not edit `tests/` or `orpheus/`)

- **The `_RULES` roster gap** (§3 above) — `folded_product` missing from a
  self-declared-exhaustive roster.
- **Two new gates land UNMARKED.** `[M]` the matrix's `unmarked` count went
  **8 → 10**; `tests/sn/angular/test_redistribution.py` tags per-test with
  `@pytest.mark.foundation` and its module docstring says so, but
  `TestG9TheProtocolDeclaresWhatItsConsumersRead`'s two methods carry no marker.
  The module docstring is therefore present-tense-false about its own contents.
- **`orpheus/numerics/axis.py`'s module docstring heading reads "The four
  slots, precisely" above FIVE bullets**, and its opening line still says
  *"(index shape, factor measure, basis kind, generator identity)"* — the same
  ambiguity §1 repaired in the corpus.
- Not a defect: `discrete_measures.rst:646`'s pre-existing *"the four shipped
  quadrature families … span the seven named entries"* SURVIVES `folded_product`
  — `[M]` its `invariance_group` is `None`, so it adds no lattice entry.

---

## L-075 — P4-remainder: DISCHARGING a seam you wrote yourself, and the day-old prose that was already false (2026-08-29, branch `feature/p4rem-producer-binds-axis`, commits `ac485104` + `ad04e236` + `1fb70c15`)

**Task.** Land the P4-remainder in the corpus: the producer binds `angular_axis`,
`AngularRedistribution.quadrature` (the courier) dies, `_weight_of` retires, the
cylinder admission probe reads the declared contract. Sweep
`curvilinear_one_group.rst`, discharge CS5's seam/fence rows in `spaces.rst`, add
the K1/K2/K3 + G5 gate rows, record the decoy catalogue.

### 1. ⭐⭐ MY OWN CS5 PAGE SHIPPED A CLAIM THAT WAS ALREADY FALSE WHEN IT LANDED

The CS5 docs pass (L-074) REPORTED three code-side gaps upward. The coordinator
FIXED all three in `cb3cd15b` — and `[M]` `git log -1 --format=%ci` puts
`cb3cd15b` and my docs commit `f8c69117` at the **same second** (19:41:19). So the
page shipped saying *"the gap is reported, not repaired here"* about a gap that
was repaired in the same batch.

`[M]` the live roster: `vars(Quadrature)` + `isinstance(v, classmethod)` = 5, and
`tests/numerics/test_axis_generator.py::_RULES` now carries all five including
`folded_product(4,8)`. TWO sites on `spaces.rst` were present-tense-false (the G4
gate row's ⚠ block, and a `(vv-status rationale)` comment saying "four of the
five").

⟹ **A REPORTED gap is a claim with the shortest shelf life in the corpus, because
the report is what triggers its repair.** When a docs pass reports N code-side
gaps upward, the NEXT pass must re-measure all N before quoting any of them —
and the tell is free: a reported gap and its repair commit will share a
batch/timestamp. Prefer publishing the gap **as history with its repair**
("shipped with four; the fifth landed the same day at `<hash>`, which is the
finite-roster corollary demonstrating itself") over publishing it as an open gap,
because the history version cannot rot.

### 2. ⭐⭐ THE DISCHARGE OF A SEAM IS AN EDIT TO **FOUR** SURFACES, AND THE SECTION'S OWN COUNTING SENTENCE IS THE ONE NOBODY EDITS

`spaces.rst` stated the seam in four places, each needing a different edit:
1. the `spaces-generator-seams` bullet (past-tense the withholding, keep the WHY);
2. the section's opening sentence *"**Three** arms of the design are deliberately
   not built"* — a universal that silently became **two**;
3. the `spaces-fences` row (its own KEEP/past-tense treatment);
4. the `spaces-generator-protocol` closing paragraph (a *different* promise —
   "hardens to a direct read when the courier dissolves").

⟹ after a seam discharge, grep the section for its own **cardinal number** and for
every forward-looking verb (*"lands with"*, *"hardens when"*, *"becomes real
when"*), not only for the seam's noun. #2 is the one a symbol grep cannot find.

### 3. ⭐⭐ A ROUTE RE-POINT NEEDS ITS OWN DOCTRINE SECTION — a value gate here is `X == X`

The binding's evidence class is unlike anything else on the page and deserved a
labelled anchor (`spaces-generator-route-gate`) rather than a paragraph:
`op.angular_axis.generator is quad` for the very `quad` the factory was handed —
i.e. the *same object* the retired courier held — so every before/after value
comparison is literally `X == X`, green under a correct re-point AND under one
that silently kept the old path. That is `vv-principles` #19 relocated from a
*metric* to a *data route*, and it is worth naming because the next re-point of
this shape (Campaign 2's strategy layer) will need the same instrument.

### 4. ⭐⭐ THE DECOY CATALOGUE: A TEST DOCSTRING'S ONE-LINE ATTRIBUTION WAS WRONG, AND THE REFUTATION IS THE PUBLISHABLE CONTENT

The helper's docstring says *"the α-dome guard REFUSES rolled/negated/reversed
nodes on every curvilinear chart"*. `[M]` mine, both tiers, sphere `gauss_legendre(4)`
+ cylinder `folded_product(4,6)`:

| decoy | axis-blind | α-dome tier | closure-mint tier |
|---|---|---|---|
| nodes ×0.9 | yes | admitted | **admitted** |
| nodes rolled 1 | yes | **REFUSED** (`Σ w·µ` → −0.366 / +0.239) | n/a |
| nodes negated | yes | admitted | **REFUSED** (P3, τ ∉ [0,1]) |
| nodes reversed | yes | admitted | **REFUSED** (P3) |
| weights ×0.9 | **no** | admitted | order-dependent: admitted N=2, refused N=4/6/8 (τ=1.195/1.047/1.059) |

The dome refuses **only the roll** — its contract is the antisymmetry
`Σ w_n µ_n = 0`, which a scale, a sign flip and a reversal all preserve (`[M]`
±5.6e-17 in every case). Negation and reversal die one tier later, at the
Morel–Montry **P3** membership guard. ⟹ **a decoy catalogue is a statement about
TWO contracts**; citing only the first sends the next session to the wrong file.
And the Cartesian chart admits everything (its dome is the neutral zero), so
"on every curvilinear chart" was doing real work in the original sentence and
still could not save it.

Also worth publishing rather than the bare number: the cylinder's **8 of 12**
keystone floor is an identity of the rule — `[M]` 4 of 12 ordinates have
`mu_x == 0.0` exactly (the ω = π/2 member of each of 4 levels) and `0.9 × 0 = 0`;
and K2's **4 of 12** is the *palindrome* `[0.440, 0.814, 0.814, 0.440]` fixing half
the levels under a roll of one. A floor with a mechanism cannot drift silently.

### 5. ⭐⭐ THE BRIEFED PAGE SWEEP FOUND ITS REAL DEFECT ONE PHASE UPSTREAM — P4.5–P4b HAD NO DOCS PASS

The brief said "sweep for courier prose". `[M]` the courier appears in `docs/` at
**exactly one site** (the one the coordinator had already re-pointed), the two-arg
closure ctor at **zero**, and `angular.quadrature` at **zero**. The surplus is
where the work was, and it was P4.7-era:

- `curvilinear_one_group.rst:485` *"**Nothing precomputes this factor.** Each
  consumer forms it where it is used"* — `[M]` **two of the three** formers
  precompute (the closure's `_dAw_per_level` at construction, P4.9a; the scan
  cache's chain-ordered row at build, P4.7); only the degenerate cylinder arm of
  the walk still forms it at use. The surviving claim is *"no store that owns
  NEITHER factor holds the product"* — which is the factorization argument the
  section is actually making. Sibling spelling on `curvilinear_multigroup.rst:179`.
- `index.rst` *"it keeps ``mu``/``abs_mu`` and ΔA/w"* — `[M]`
  `dataclasses.fields(StreamingTerms)` = `('face_area_inner','face_area_outer',
  'volume','abs_mu')`; P4.7 shed `mu`, `chord_length`, `delta_A_over_w`. The
  CONCLUSION (the packet is not geometry-only) survives on ONE field, which is its
  strongest form.
- `index.rst:1019` still called the packet *"the **purely geometric** primitive"*
  while a note 75 lines below refuted exactly that — the vv #21 self-contradicting
  file, created by a correction pass that fixed the note and not the bullet.
- `structured_geometry.rst` — *"populated fields are geometry-dependent (slab is
  minimal)"* and *"the ``alpha_in is None`` test discriminates slab from
  curvilinear"*, both about fields that no longer exist; plus the
  `tau-ownership-note` saying `morel_montry_tau_per_level` is called *"by
  `SNMesh` against the quadrature and its own `self.coord`"* — `[M]` its ONE
  production caller is `MorelMontryAngularSweep.__init__`.

⟹ **a phase that lands with no docs pass leaves its staleness for the NEXT
phase's sweep to find, and the next sweep's brief will not name it.** Budget for
it: my briefed scope was 3 pages, the honest scope was 5.

### 6. ⭐ A DISCHARGED `*(in development)*` HATCH IS A SECOND, INDEPENDENT DISCHARGE

`[M]` CS5's `4e7b8977` + `b0bfc06c` are ancestors of **main**, so the dev-history
row's *(in development)* cell was false — and a LINE-based grep for
`"in development"` finds **nothing**, because the phrase wraps (`*(in\ndevelopment)*`).
Only a multi-line regex over the corpus finds it (vv #21's windowed search, at the
page-preamble scale). ⟹ on every dispatch that adds a dev-history row, run the
multi-line hatch census FIRST and reconcile every hit against
`git merge-base --is-ancestor <hash> main`.

### 7. ⚠ SELF-CHECK THAT SAVED A FALSE FINDING

I probed `op.angular_axis.weights.flags.writeable` and printed it under the label
`"read-only:"` — the value `False` read as *"not read-only"* and I nearly filed
`spaces.rst`'s LIVE-REFERENCE warning as stale. The label was inverted, not the
value (`[M]` re-probed: `writeable=False` on all three mint paths, so the warning
is correct). `vv` #4's VERIFY sharpening, on my own instrument: **diagnose whose
failure it is before publishing a refutation** — and do not print a boolean under
a label that negates it.

### 8. Verification recipe used (reusable)

- `-E` baseline **and** every verification build; grep `WARNING:|ERROR:|CRITICAL:|SyntaxWarning`.
  `[M]` 0 → 0, EXIT=0, three builds.
- `tools/check_docstring_xrefs.py` is role-blind (L-067); ran a **patched copy at
  `scratch/_p4rem_xref_gate.py`** (depth 1, `head_role = "mod" if "." in target else role`)
  with an end-to-end positive control — `[M]` stock **0** vs patched **2** on a
  throwaway `docs/_ctl.rst`, and **0 dead / 13512 decidable / 986 files** tree-wide
  with the control removed. The control is what makes the zero mean anything.
- `mcp__nexus__dead_references` → **0 dead / 52 checked / 52 rescued**.
- `tests._harness.audit._scan_theory_equations(Path("docs/theory"))` → **0
  violations / 567 documented** (signature takes `theory_dir`; the no-arg call
  `TypeError`s).
- Programmatic `list-table` column check (widths vs per-row item count) over every
  edited file — 56 tables, all OK, BEFORE the first build.
- Underline-length check in CODE POINTS + marker-ladder first-appearance scan
  (`spaces.rst` is `=`/`-`/`~`, **no** `^` — do not introduce one).
- HTML slice for nested markup: `[M]` my edits added **0** visible backticks
  (8 introduced in a `::` code block were removed for style); the residual
  **14** bold-runs-with-nested-markup on the swept pages (spaces 2,
  structured_geometry 3, curvilinear_one_group 9) are pre-existing **#379**.
- Auto-matrix: `[M]` 10215 → **10236 = +21**, exactly the predicted delta;
  `unmarked` unchanged at 8.

---

## L-076 — P7 non-diagonal metric: a gate's NAME is a universal, and the source regex cannot see a bold run that WRAPS (2026-08-30, branch `feature/p7-nondiagonal-metric`, commits `6a0e0473` + `bae73fa7` + `f1f30cea` + `af9f95f1`)

**Task.** Document P7 of the streaming campaign — the `HilbertMetric` family
(`DiagonalMetric` / `DenseMetric` / `FactoredMetric`), the space's third metric
source and three-arm exclusivity guard, the frame's DENSE dressing, and the
re-posed curvilinear refusal. Brief named 4 work items over 3 files; honest
scope came out **8 files** (the two extras found by my own census: a
`normalization.rst` warning, a `spherical_harmonics.rst` warning + table row, an
`sn/history.rst` past-tense slip, and two forward pointers in
`foundations/index.rst` + `api/numerics.rst`).

Baseline and result: `-E` **0 W / 0 E / 0 C**, EXIT=0, both sides. Generated
artefacts moved exactly as predicted (`matrix.rst` documented labels 567 → 568;
`no-implementation` declarations 17 → 18; total tests 10236 → 10266 from the
code side). `dead_references` 0/52. `check_docstring_xrefs` 0 dead / 988 files.
Corpus `:ref:`/`:eq:`/`:doc:` resolution 0 dangling.

### 1. ⭐⭐ A GATE'S NAME IS A UNIVERSAL — measure it against the whole shipped family, and the measurement can produce a THEOREM

The tree's D3 gate is named
`test_the_scalar_frame_square_collapse_is_a_sphere_family_property`, and its
docstring says the slab's failure is because *"the slab's live ℓ=2 Gram diagonal
`[0.4, 0.8, 0.8]` is not a per-ℓ scalar"*. Both halves are true of the slab. The
NAME is false as a universal, and the *mechanism* is not the per-ℓ-scalar
reading either.

`[M]` closure residual `rel ‖M*y − Ry/W‖`, 200 seeds each, at HEAD:

| frame | verdict | per-ℓ live-diag spread | rel band |
|---|---|---|---|
| the six DIAGONAL sphere frames | `DIAGONAL` | ≤ 3.4e-15 | ≤ 9.5e-16 |
| `gauss_legendre(8)` L=2 | `DENSE` | 6.0e-1 | 0.300 – 10.18 |
| `product(4,4)` L=2 | `DENSE` | 1.75 | 3.1e-3 – 0.333 |
| `level_symmetric(4)` L=3 | `DENSE` | 3.3e-1 | 3.4e-2 – 0.155 |
| `folded_product(4,6)` L=3 | `DENSE` | 8.3e-1 | **3.2e-16 – 2.8e-15** |

- `product(4,4)` is a **sphere** rule that BREAKS the collapse ⟹ the name's
  "sphere family" is not the discriminator.
- `folded_product(4,6)` L=3 is `DENSE` with a NON-constant per-ℓ live diagonal
  and **closes anyway** ⟹ the per-ℓ-scalar reading is not the mechanism.

⭐ Chasing that one row produced the decidable form. `M* = R/W` ⟺
`Y(G⁺ − diag(d)/W) = 0`: the metric and the reconstruction weights need agree
only **modulo `ker Y`**. `[M]` on `folded_product(4,6)` L=3 the only live
off-diagonal couples two ℓ=3 slots whose 2×2 block `[[0.6732, 0.8691],
[0.8691, 1.1220]]` has **det −8.7e-17, rank 1** — the two harmonics are
linearly *dependent* on that folded node set — so `‖Y·D‖_∞ = 4.4e-16` with
`‖D‖_∞ = 0.557`; on the slab `‖Y·D‖_∞ = 6.30`. Published as a theorem plus the
five-row table, replacing a correlation.

⟹ **when a doc must state why a gate's population is what it is, run the gate's
predicate over the whole shipped family before repeating the gate's own name.**
The honest published statement became *"`DIAGONAL` is SUFFICIENT; `DENSE` does
not decide it"*, which is strictly stronger than either the gate name or the
plan's version, and it is what stops a future session "fixing" the gate by
adding the DENSE params.

### 2. ⭐⭐ A SOURCE regex cannot see a nested-markup defect in a bold run that WRAPS — but a MULTI-LINE regex DIFFED AGAINST `HEAD` can, and it needs no build

L-074 said *"the HTML slice IS the nested-markup gate; a source regex CANNOT
replace it"*. Half right, and the half that is wrong cost me a build.

`[M]` my pre-splice check ran `re.finditer(r"\*\*(.+?)\*\*", line)` **per line**
and reported **0**. The rendered HTML slice then showed **4 visible backticks**
— from `**``DIAGONAL`` is sufficient for the scalar closure; ``DENSE`` does\nnot
decide it.**`, a bold run spanning two source lines. (Probed, not reasoned:
`publish_doctree("A **bold with ``lit`` inside** end.")` → `A bold with ``lit``
inside end.` — RST does not nest, and it is silent at every severity.)

⭐ The instrument that works **before** a build, and isolates MY hits from the
corpus's pre-existing ones:

```python
rx = re.compile(r"\*\*(?!\s)((?:[^*]|\*(?!\*))+?)(?<!\s)\*\*", re.S)   # re.S is the point
hits = lambda txt: {" ".join(m.group(1).split())[:90]
                    for m in rx.finditer(txt) if "``" in m.group(1)}
new = hits(Path(f).read_text()) - hits(git_show(f"HEAD:{f}"))
```

`[M]` over the 8 edited files: **1 new** (mine, fixed) against **28 + 18
pre-existing** in `error_catalog.rst` and `sn/history.rst`. Without the
set-difference the signal is buried; with it, the answer is one line. Keep the
HTML slice as the confirming gate (`rfind`-anchored, per L-074) — but run the
source diff first, because it costs no build.

⚠ Corpus finding to report: this defect class is endemic —
**46 pre-existing instances** across two pages. It renders wrong, silently, at
every severity. Out of scope for a phase docs pass; worth an issue.

### 3. ⭐⭐ An OPERATOR-movement claim has a DRAW-FREE form — build the matrix column by column, don't probe it with a vector

The phase's D5 gate docstring publishes `max|Δ M.H| = 8.246 — rel 0.8995` for
the production `product(4,4)` L=2 frame. `[M]` reproduced: that is **one draw's
reading** — over 200 seeds the same relative movement bands **0.879–0.986**, and
on the slab it bands **0.53–4.55**.

⭐ The draw-free instrument costs `K` applies: feed the unit coefficient vectors
`e_k` through both adjoints and assemble the two matrices. `[M]`

| frame | operator `max\|Δ\|` | rel (max-norm) | rel (Frobenius) |
|---|---|---|---|
| `product(4,4)` L=2 | 12.49 | 0.994 | **0.985** |
| `gauss_legendre(8)` L=2 | 12.39 | 0.986 | **0.980** |
| `level_symmetric(4)` L=3 | 12.49 | 0.994 | **0.980** |

Published `98 %` in Frobenius relative, with the reading that makes the number
mean something: *the two operators are not a small correction apart, they are
essentially unrelated* — which is the correct framing for a repair whose "before"
state the tree's own docstring calls *"the stored-metric sandwich, NOT the
physical Hilbert adjoint"*. This is L-071's three-flavours lesson moved from a
FLOAT-agreement claim to an OPERATOR-movement claim: the flavour to publish is
the one computed on the operator, not on a probe.

### 4. ⭐ Publish the ANALYTIC threshold, not a scan point — a coarse scan mints a wrong constant

`_DENSE_METRIC_RCOND`'s docstring says the Parseval ratio *"breaks only at
`>= 5e-2`"*. `[M]` it is already broken at `3e-2` (`0.991414787`). The cliff is
not a scan result at all: `np.linalg.pinv`'s `rcond` is relative to `σ_max`, so
truncation begins at `σ_min^live/σ_max = 4.745e-2 / 2.708 = **1.7524e-2**`.
Published the analytic threshold and the "ten orders below the cliff, five above
the noise floor" placement, and softened my own scan claim from *"for every
`rcond` in [1e-15, 1e-2]"* to *"at every scanned `rcond` … as it must, since no
truncation can occur below the cliff"*.

Same class, same file: the module docstring's *"`G G⁺ G = G` to 9.99e-16"* — `[M]`
three reasonable norms of that residual are `1.554e-15` (max-abs), `7.77e-16`
(rel to `max|G| = 2`) and `7.75e-16` (Frobenius ratio). **Write the norm.**
And *"a live-block eigenvalue at 6.82e-17"* is a noise-floor digit that does not
reproduce (`eigvalsh` gives `8.21e-17`, SVD `6.02e-17`); published the structural
form instead — *5 live slots, rank 4, smallest live eigenvalue at the round-off
floor, fifteen orders below the smallest genuine mode `4.745e-2`*.

### 5. ⭐ `hasattr(Cls, field)` is FALSE for a dataclass field with no class default — `dataclasses.fields` is the cheap oracle

My role-import probe over the 5 edited pages reported **3 dead `:attr:`
targets** — all three `ReducedStreamingOperator.angular_axis`. `[M]`
`hasattr(R, 'angular_axis')` is `False` while
`[f.name for f in dataclasses.fields(R)]` lists it: it is an instance attribute
with no class-level default. All three roles are LIVE.

This is L-053(c) (*construct the object, never probe the class*) with a cheaper
form for the dataclass case. ⟹ any import probe that walks `getattr` chains over
project symbols must fall back to `dataclasses.fields` before reporting a dead
`:attr:`, or it manufactures false positives on exactly the newest code.

### 6. ⭐ A section RENAME is cheap when you count citers FIRST — and the section's own note is what makes a stale pointer diagnosable

`frame-parseval-dense-refusal` encoded the REFUTED mechanism ("refusal") in its
anchor name. `[M]` citers: **1** cross-doc (`normalization.rst`, which I was
editing anyway) and **0** in `.claude/` / `scratch/`. So the L-063 caution
(*renaming risks a silent cross-doc break*) does not bind, and the rename to
`frame-parseval-dense-arm` shipped with its one citer in the same edit.

⭐ The move that costs nothing and pays later: a `.. note::` at the renamed
anchor recording the old name, why it moved, and *"a stale pointer to the old
name renders as plain text at every build severity; if you meet one, it predates
P7."* That converts an undiagnosable dead link into a self-explaining one.

### 7. ⭐ A changelog's chronological ORDER is per-page — read the dates before placing

`spaces.rst`'s Development history is a `list-table`, **reverse**-chronological
(latest first). `frame.rst`'s is prose blocks, **forward** chronological (oldest
first). I placed the P7 block in `frame.rst` immediately after the F-0 entry it
tombstones — natural, and wrong. Caught by a two-line check
(`re.finditer(r'^\*\*(\d{4}-\d{2}-\d{2})', t, re.M)` then `== sorted(...)`) and
moved to the end before the References section. ⟹ run that check as part of any
changelog insertion; it is free and it is the only thing that sees the mistake.

### 8. The event-class shape that worked: a REFUSAL BECOMES A CAPABILITY

Not a close-out, not a retirement — a *refusal repaired*. The shape:

1. **Keep the diagnosis, past-tense the verdict.** The slab Gram table, the
   impossibility argument and the "no diagonal candidate" claim are UNCHANGED
   and are the reason the repair was possible. Only *what the frame did with
   them* moved.
2. **Publish the refusal era verbatim under a ⛔**, with the sentence that says
   why it was correct at the time (*"a diagonal metric is not merely unavailable,
   it is provably insufficient, and nothing in the space layer could express the
   alternative"*).
3. **Split the debt.** "Recorded debt (CS4c)" had two halves; P7 discharged one.
   The note now says which half landed, which two remain (the legs, and
   retiring `_AdjointOperator` into them), and — the load-bearing bit — *what P7
   changed for them*: they now have exactly one metric arithmetic to wrap.
4. **Say what did NOT ride along**, with its own measured section (§1 above).
   That is the sentence that stops the next reader over-reading the repair.
5. **New ERR chapter, not a new number** — the landed gates carry
   `catches("ERR-039")` (`test_no_diagonal_metric_can_satisfy_parseval_on_a_dense_frame`
   and `test_the_dressing_lands_parseval_on_the_production_anisotropic_frame`),
   so a new id would orphan them (L-065's rule, applied again).

### 9. ⚠ Reported upward (code-side, not editable here)

- D5's docstring `max|Δ| = 8.246 / rel 0.8995` is one draw (band 0.879–0.986;
  draw-free `rel_F = 0.985`).
- `test_the_scalar_frame_square_collapse_is_a_sphere_family_property` — the NAME
  over-generalises (§1); the assertion (`rel > 0.5`) is fine.
- `_DENSE_METRIC_RCOND` docstring's `>= 5e-2` cliff (true value `1.75e-2`);
  the module docstring's `9.99e-16` (norm unstated) and `6.82e-17` (noise digit);
  the matmul-ULP figures (`1360 of 2000`, `1792 ULP`) are one draw.
- "11 of 41 shipped frames" (plan/commit message) — my enumeration gives
  **10 of 30** angular constructions + the non-angular overlap frame.
- 46 pre-existing nested-markup defects in `error_catalog.rst` (28) and
  `sn/history.rst` (18).
- `orpheus/numerics/metric.py` is NOT `automodule`'d (nor is `space`/`frame`/
  `operator`), so its `:class:` refs render plain text by page convention —
  DEFERRED, not a defect. `[M]` it carries **0** `.. math:: :label:` blocks, so
  it is a SAFE automodule candidate whenever `numerics` is surfaced as a package.

### Quality self-assessment (Directive 3)

| dimension | score | note |
|---|---:|---|
| Derivation depth | 5 | the `Y(G⁺ − diag(d)/W) = 0` decidable form + the Penrose-identity Parseval chain, both derived rather than asserted |
| Cross-references | 5 | 0 dangling corpus-wide; the renamed anchor's one citer moved with it; 4 new forward pointers |
| Numerical evidence | 5 | every published number re-measured this session; three point-values replaced with measured BANDS; one table (the frame-square split) is new measurement |
| Failed approaches | 5 | the refusal era preserved verbatim with the reason it was correct; the pre-P7 propagation values published as counterfactuals |
| Code traceability | 4 | roles resolve but `numerics.metric` is not automodule'd, so they render plain text (page convention, deferred) |
| Derivation source | 3 | no `derivations/` script exists for the metric family; the source of record is the module + the gates. Not a gap I can close, and the L0 pins are hand-derived binary-fraction literals, which is the right instrument here |

Weakest: **derivation source** — structurally, not by neglect: a metric
realization is numerics machinery, not a physics derivation, and its correctness
evidence is exact-arithmetic literals plus the wrong-metric discriminator, both
of which the tree already ships.

---

## L-077 — CS4c step 4 (fission rebind): a class SPLIT rots the "same class everywhere" thesis, and two sibling harmonizations can differ in KIND (2026-08-31, branch `refactor/cs4c-step4-fission-binding`, HEAD `fadad026`)

**Task.** Corpus pass for the CS4c step-4 fission rebind: one datum
(`FissionKernel`) became TWO bindings — `IsotropicFission` (energy, the scalar
dyad) and `FissionOperator` (angular, the frame's ℓ=0 conjugation) — plus an
N2N harmonization onto the same shape.

### (a) ⭐⭐ A class SPLIT is the staleness class a symbol grep cannot rank

`grep FissionOperator docs/` returned **50 hits / 19 files** — and the symbol
still exists, so every hit "resolves". The defect is that ~15 of them meant
*the scalar dyad*, which moved to a NEW class. No xref gate, no `-W`, no
`dead_references` can see this: the target is alive and the sentence is wrong.

⟹ **After a split, the instrument is an AST census of PRODUCTION CONSTRUCTION
SITES, per package** — not a doc grep. `[M]` mine, `ast.Call` over `orpheus/`:

| class | sites | packages |
|---|---:|---|
| `FissionOperator` | **1** | `sn` only |
| `IsotropicFission` | **4** | diffusion, homogeneous, sn, transport |
| `MultiplicationOperator` | 3 | diffusion, homogeneous, sn |

That table decided every one of the ~15 adjudications in one command, and it
is the evidence the corrected prose now carries.

### (b) ⭐⭐ The split refuted a THESIS, in three places, and the fix STRENGTHENS it

`path_integral.rst` ×2 + `foundations/index.rst` ×1 asserted *"`MultiplicationOperator`
and `FissionOperator` are the **same Python classes** instantiated by SN,
diffusion and the infinite-medium solver"* — the corpus's load-bearing
shared-code claim, in the ROOT page and the PART index. Post-split: false for F,
true for `MultiplicationOperator`. ⭐ The repair is not a downgrade — fission had
been the one channel with a single class serving a scalar AND an angular
consumer, so it read as the *cleanest* example of sharing while hiding the
*shape* of the sharing; after the split all three reaction channels share the
same two-binding shape. **Write the correction as the thesis getting sharper,
with the census as its evidence**, and add the machine-header key the census
implies (`angular_bindings:` beside `all_three_consumers:`).

### (c) ⭐⭐ A published CODE BLOCK is the highest-severity staleness in the corpus

`infinite_medium.rst` showed a 4-line "constructed in four lines" example whose
first constructor call was `FissionOperator.from_solver_data(...)`; live code
is `IsotropicFission.from_material_xs(...)`. A code block promises
reproducibility, so it fails Cardinal Rule 1 harder than prose does — and
nothing gates it. ⟹ **after any constructor/signature change, grep
`.. code-block:: python` bodies for the changed symbol specifically**, ahead of
the prose sweep.

### (d) ⭐⭐ A `.. implements::` whose transcription QUOTES the body rots when the body MOVES — and its three fates are visible in one read

`fission-as-dyad` carried two declarations transcribing
`outer(self.chi, self.production_rate)` and
`ReactionRateFunctional(self.mat_xs.fission_production_field)`. `[M]`
`hasattr` sweep: **`FissionOperator.chi`, `.sig_p`, `.mat_xs` are all GONE**;
the live body is `self.fission.gather_chi(...)` on the NEW class. The right
answer was neither pure MIGRATE nor pure REMOVE but **BOTH-with-roles**: declare
the new arithmetic home AND keep the old names declared *as delegations*, saying
so — because the Protocol gate and a production consumer still reach the dyad
through the old names, and dropping them would under-declare the equation.
⭐ Net `directives: wrote N edges` 412 → **415**, predicted exactly (+2 on
`fission-as-dyad`, +1 on `multigroup`); a mismatch would have meant a `:by:`
silently failed to bind.

### (e) ⭐⭐ TWO sibling harmonizations described alike can differ in KIND — measure both, and the difference is a THEOREM

Both N2N and F replaced hand arithmetic with the *same* product reversal, and
both production docstrings say *"a pure IEEE-754 order change,
principled-equivalent, gated at tolerance."* I drafted one shared note saying
that. `[M]` on real fixtures it is **false for N2N and true for F**:

| channel | draws | `array_equal` | `max\|Δ\|` | max ULP |
|---|---|---|---|---|
| N2N | 200 seeds × GL n = 2/4/6/8/16 | **1000 / 1000** | **0** | 0 |
| F | 200 seeds × lebedev 17/11 + LS4 | **0 / 200** each | 8.33e-17 | 4–5 |

And the CAUSE is structural, not fixture luck: at ℓ = 0 the outer factors
degenerate (`R₀ᵀ` = plain ordinate sum, `M₀ᵀ` = per-ordinate ×wₙ), so the chain
does the same ops in the same ORDER as the retired broadcast — N2N's retired
spelling divided by W at the end, F's divided *before* `Kᵀ`. ⟹ **when two
changes are described by one sentence, run both**; and pair a bit-exact reading
with its structural reason (vv #31: one draw is a property of the draw, a sweep
+ a mechanism is a property of the binding). The measured pair went into the
page as a `.. list-table::` with a ⚠ *do not pin F at `array_equal` on the
strength of N2N's result*.

### (f) ⭐ A page can contradict ITSELF 80 lines apart, and the older half is the one a reader trusts

`slab_multigroup.rst` states `A = L + C − S − N₂ₙ − B` at §480 (the step-3
extraction narrative) and `A = L + C − S − B` **twice** at §558 (the operator
section, two display equations). Both authored, one stale. ⟹ after any algebra
change, grep the changed algebra's OLD spelling **within each page that carries
the new one** — a page that learned the correction in one section is the most
likely place for the uncorrected twin.

### (g) ⭐⭐ When a residue census is 37 sites and a SIMPLIFICATION, DECLARE it — do not sweep it, and do not stay silent

`[M]` **37** SN-chapter sites still spell `A = L+C−S−B` (step-3 residue). Three
options, and the middle one is right: (1) sweep all 37 — a numerics
adjudication riding inside a fission docs pass, and it costs pedagogy where
Σ₂ₙ ≡ 0 by fixture; (2) silence — leaves 37 present-tense-false sites; (3)
**declare the simplification at the chapter root** (machine header
`composites.A` + a `.. note::` naming it *"a deliberate simplification, not the
shipped member list"*, pointing at the canonical eq-label), fix only the sites
genuinely describing the SHIPPED composite, and report the census with its
denominator. (3) makes every remaining site honest by construction and leaves
the sweep as a scoped follow-up.

### (h) ⭐ A gate's DESIGN properties are publishable theory, not test trivia

G-F1 (the χ↔νΣf-coupled condensation) had two properties worth lifting into
`frame.rst`: its morphisms are **hand-built in the test body** (structurally
independent, vv L11) and it **asserts its own activation precondition** (a
1-fine-per-coarse target makes `average ≡ marginalize` and every control go
silent — Mode 12 at the fixture — so an identity condensation is REFUSED with
its own red row). Publishing those two paragraphs is what stops a future
fixture edit from silently de-fanging the law, and no test docstring is read by
the person editing the fixture.

### (i) Mechanics confirmed / re-measured

- **Baseline re-measured, not quoted:** `-E -W` EXIT=0, **0** W/E/C/SyntaxWarning
  at HEAD `fadad026`. Post-pass identical.
- **The patched xref gate needs its positive control every time.** A throwaway
  `docs/_ctl_*.rst` with 2 dead roles + 1 live: stock **0**, patched **2 dead**.
  Control removed → **0/0**. Without it, `DEAD TARGETS: 0` is indistinguishable
  from a broken scan. (L-071 mechanics; the copy must sit at `scratch/<name>.py`,
  depth 1.)
- **A documented-sentinel label adds NO test.** `tests/_harness/audit.py` computes
  `testable_labels = theory_labels − documented_labels`, so my +3 labels moved
  the matrix's sentinel count 571 → **574** and the collected total not at all
  (its +27 was entirely code-side, in 4 test modules). Predict the sentinel
  count, not the test count, for a documented-only labels pass.
- **A `.. note::` needs a blank line before the paragraph that follows it** —
  the one warning I introduced (`Explicit markup ends without a blank line`),
  caught by the `-W` build in one cycle.

---

## L-078 — a PHYSICS claim vs a MODEL claim: the two-kind sort, and the sites the
##          brief could not name

**Task** (2026-08-31, branch `docs/n2n-isotropy-claim`, commit `6906f2a2`).
Eight corpus/docstring/test sites asserted *"(n,2n) emission **is**
isotropic"* as a fact about the reaction. The evaluated GENDF data ORPHEUS
itself ships refutes it. Correct everywhere without weakening any test.

### The governing move: sort each site into (a) PHYSICS or (b) MODEL

The brief supplied the rule and it is the transferable content. A correction
pass over "X is isotropic" is NOT a search-and-replace, because the same
sentence fragment is FALSE about the reaction and TRUE about the operator:

- **(a) a claim about the REACTION** — *"(n,2n) emission is isotropic"*,
  *"the (n,2n) reaction is a DISTINCT isotropic group transfer"* ⟹ FALSE,
  correct it.
- **(b) a claim about the MODEL/CODE** — *"only the `[0,0]` block is read and
  written"*, *"the composite action IS the isotropic lift"*, *"N2N wrote a
  non-zero ℓ≥1 block"* ⟹ TRUE of what ships. **Keep the assertion**; make
  explicit that it describes a TRUNCATION.

⭐ The failure mode is symmetric and both halves are easy: weakening a true
code claim because its neighbouring physics claim was false, and leaving a
false physics claim standing because its neighbouring code claim is right.
`material_field.py:345` had **both in one sentence** — *"every ℓ≥1 block
stays zero — (n,2n) emission is isotropic"*: keep the clause before the dash,
replace the clause after it.

### ⭐⭐ TWO FILES CONTRADICTED THEMSELVES, and the hedge was the true half

`n2n.py:4` and `adjoint.rst:691` both quote the CS4c ruling — *"(n,2n) is
scattering-like — a group transfer **which in principle carries its own
anisotropy**"* — and then assert flatly, twenty lines later, that the
emission IS isotropic. The **hedge was right and the flat assertion was
wrong**, i.e. the page's own more-cautious sentence was its correct one.

⟹ The repair shape: **promote the hedge to the measurement, demote the flat
assertion to a stated modelling choice** — and where the hedge sits inside a
QUOTED RULING, keep the quote **verbatim** (it is the record of what was
argued on a date) and put the promotion in a dated paragraph beside it,
saying the ruling is *strengthened*: the anisotropy axis it declined to
foreclose is real, not hypothetical. (`plan-authoring` §3 in the corpus.)

### The sweep: three independently-vocabularied filters, all with controls

A line-based `grep "n,2n.*isotropic"` misses every wrapped instance. What
worked, in order:

1. **Windowed** (subject within ±2 lines of the predicate) with an
   **identifier-stripping** pass — `IsotropicN2N`, `isotropic_scattering`,
   `assemble_per_ordinate_isotropic`, `K_iso` are NAMES, not claims. Without
   the strip: 388 hits. With it: **98**, all readable.
2. **Independent vocabulary** that never spells the subject — `doubling`,
   `two neutrons`, `multiplication channel`, `multiplicity`,
   `\Sigma_2\b`. Found `infinite_medium.rst` and the
   `isotropic_scattering.py` module docstring the first filter reached
   differently. Both filters converged.
3. **A third, isotropy-word-FREE predicate** — `no angular dependence`,
   `angularly flat`, `single Legendre`, `P0-only`, `no moment tensor`.
   15 hits, **all** fixture names (`solver_2g_p0_n2n`) or my own new text.

⭐ **The residual filter's POSITIVE CONTROL caught a real hole in itself.**
I fed it the four verbatim pre-edit strings; three matched and the fourth
(*"it must be isotropic (ℓ=0 only)"*) did **not** — my copula alternation was
`(is|are|being|it's)` and the word before `isotropic` was `be`. Widened to
`(is|are|be|being|been|remains?|stays?|it's)`. Without the control the sweep
would have reported that site clean.

### ⭐⭐ The brief's starting set was 7 sites; the honest set was 8 — and the
### miss was in the SAME section as a named site

`slab_multigroup.rst:504` read *"Note also what the extraction did **not**
touch: **the emission is isotropic**, so the operator keeps the reaction-rate
fast path"* — 70 lines below `n2n-reactions`, which the brief did not name at
all. It is invisible to a subject-first grep because the subject
(*"the emission"*) is a pronoun-like back-reference to the section's topic.
⟹ **a windowed sweep must window on the PREDICATE too**, not only the
subject, when the subject can be carried by section context.

### ⭐⭐ Reproduce the numbers, and one CONTRAST did not survive

Every headline reproduced exactly with my own probe (NL table over all 13
files; `‖P1‖∞/‖P0‖∞ = 0.6897`; `μ̄ = +0.2783`; shares `61.747 % / 44.870 %`;
rank 50, rank-1 error `0.5818`; control `2/(3A) = 0.074615`, 0.00 %) —
**except one**. The source memo and the issue both wrote *"μ̄ = +0.278
against **+0.094** for elastic on the same nuclide"*, glossed as *"~3× that
of elastic"*. `[M]` those two numbers are summed over **different energy
windows**: `+0.094` is elastic over all 421 live rows (dominated by the
low-energy s-wave region, per-group `0.0746`), while `+0.278` is (n,2n) over
its **50** open rows. Over the SAME 50 groups elastic is **`+0.4264`** — i.e.
elastic is MORE forward-peaked there, and the "3×" claim inverts.
⟹ **I published the (n,2n) figure alone with its window stated and made the
elastic contrast STRUCTURAL instead** (*MT=16 stores NL = 7, the same order
as elastic, which stores 7 in 13 of 13 files*) — a denominator-clean
comparison. `plan-authoring` §2's quantifier clause, met in a relayed
contrast rather than in a count.

### ⭐ The un-`automodule`'d majority: the build cannot see 5 of the 11 files

`[M]` `orpheus.transport.operators.*` and `orpheus.transport.material_field`
have **automodule = 0, html_pages = 0** — the docstrings I edited are never
rendered, so `-W` is silent about them at every severity, and `-n` would be
too. Per AGENT.md I did NOT add an `automodule` for the leaves I touched.
The substitute gates that DO see them:

- the **patched** `check_docstring_xrefs.py` (head-role fix), which resolves
  by IMPORT — with an end-to-end positive control on a throwaway
  `docs/_ctl_n2n.rst`: **stock 0, patched 2**, then 0 across 999 files with
  the control removed;
- nexus `dead_references` (by rendered target): 0 dead / 52 checked;
- a **differential docutils parse, HEAD vs working tree**, counting roles
  that survive as literal text (the silent non-rendering class):
  `HEAD = 1, now = 1` — no regression, the 1 pre-existing.

⚠ **That docutils harness needed two goes.** v1 walked `dir(module)` and
reported **89 "problems"** that were all `dict.__doc__` on `__annotations__` /
`__dataclass_fields__`. v2, scoped to the five edited docstrings, reported
**0** — and its positive control ALSO reported 0, i.e. it was blind. Rendering
the control directly showed why: `**bold**:math:\`x\`` **parses fine** in
docutils (my "fix" was unnecessary, though harmless and more consistent with
the file's own `ℓ` spelling), while the genuinely-bad `text~:math:\`x\``
produces **no message at all** — it silently degrades to literal text plus a
`<title_reference>`. ⟹ for the silent class the instrument is *"count role
spellings surviving in the rendered output"*, never *"count system
messages"*; and only `:math:` is testable that way, because bare docutils
does not know the Sphinx `:class:`/`:attr:`/`:meth:`/`:func:` domain roles.

### The doc SHAPE for this event class

- ONE home for the whole measurement set — a `.. warning::` with its own
  anchor, in the section where the over-claim would be re-minted. Everything
  else **points** (the brief's own constraint 5, and it keeps the numbers
  single-sourced).
- The anchor sits above an admonition, so **every reference uses explicit
  text** (`` :ref:`the truncation warning <label>` ``) — a bare `:ref:` there
  is the `ref.ref` "*title or caption not found*" class. Verified all five
  resolved as real `href`s in the built HTML, not by the warning count.
- The **physics home** of the reaction (`n2n-reactions` in
  `slab_multigroup.rst`) gets a short `.. important::` before any algebra;
  the **data-layer home** (`cross_section_data.rst`, at the MF=6 record
  structure where the Legendre loop is displayed) gets the truncation
  recorded where the drop happens; **Key Facts** and the **machine header**
  each get one clause, because those are what a reader quotes.
- A class whose NAME encodes the truncation (`IsotropicN2N`) gets an
  **"On the name"** paragraph — and the file already had the precedent, on
  `IsotropicFission`, where the same prefix carries family signal instead of
  a contrast. Follow the sibling's shape rather than inventing one.

### The eq-label was CORRECT — only its premise was wrong

`sn-n2n-isotropic-lift` states what the code computes, so its **body was not
touched**. I read all four citers (`:768`, `:782`, `:880`, and the generated
`matrix.rst:1388`): the transpose derivation differentiates the lift and is
unaffected; the vv-status rationale describes the model. No companion note
was needed on any of them, and `matrix.rst` is generated — predicted and
confirmed **unchanged**, because the pass adds no eq-label.

### ⛔ REPORTED, not fixed — a DIFFERENT false claim about the same channel

`docs/theory/foundations/cross_section_data.rst:582-700`, "Reactions Not
Included: (n,2n), (n,3n), (n,4n)", is present-tense-false **for (n,2n)**:
`[M]` `gendf.py` calls `_extract_mf6(16, …)` and populates `sig2`;
`solver.py:1713` puts `emission_n2n` in the k denominator; the shipped
algebra is `A = L + C − S − N₂ₙ − B`. Its "ORPHEUS's current balance equation
assumes a 1-in-1-out scattering model" and its deferred implementation sketch
were true of MT=17/37 only. ⟹ **out of brief scope (a different claim class,
spanning CP/MoC/MC), so REPORTED rather than swept.** The rule that decided
it: *fix the claim you were sent for; report the neighbouring one with its
`file:line` proofs, do not let a correctness sweep acquire a second subject.*

---

## L-079 — a NEW foundations page for a LEVEL nobody owned (`manifolds.rst`, #429 tracker 2.0a)

**Task.** Document the `Manifold` mint (`b8c05d16`, branch `fix/angular-phantom-support`):
the point set a measure lives on, level 1 of a three-level stack whose levels 2 and 3
`spaces.rst` owns. Placement was mine to decide.

**Shipped.** New `docs/theory/foundations/manifolds.rst` (≈1600 lines, 2 new eq-labels both
`documented`-sentineled, 29 anchors) + 5 edited pages (`index`, `spaces`, `discrete_measures`,
`spherical_harmonics`, `error_catalog`) + the regenerated `matrix.rst`. `-E -W` **0/0/0 both
sides**, EXIT=0, warning-SET diff `{}` both ways; `dead_references` 0/52; patched xref gate
**0 dead / 1004 files** with a live positive control; theory scan 905→907 labels /
574→**576** documented / 0 violations (predicted exactly); 40 foundation gates pass under `-O`.

### The placement ruling, and its test

Three candidates: a section in `spaces.rst`, a section in `discrete_measures.rst`, a new page.
**Ruled: a new page**, on three grounds, of which only the first is decisive:

1. ⭐⭐ **The level-1 doctrine cannot be homed inside a level-2 page without re-committing the
   exact conflation it exists to end.** A section titled *"a function space is not a domain"*,
   sitting inside the function-space page, is structurally self-undermining.
2. **Three consumers, no owner.** `discrete_measures` owns `support`-as-a-tag, `spaces` owns
   the `L2[...]` name, `spherical_harmonics`/`error_catalog` own the forgery. None is
   subordinate to another; a shared page is the SSOT, not a twin.
3. Size: `spaces.rst` is 3871 lines already.

⚠ **The twin risk is real and had to be actively managed.** `discrete_measures.rst` already
owns the *support-propagation table*; `spherical_harmonics.rst` + `frame.rst` already own
*Funk–Hecke + Schur* and the `RΛM` factorization. My first draft of "Consequence 4" restated
Funk–Hecke and was **rewritten** to own only the register nobody had — the **Gelfand-pair /
double-coset** framing (`[M]` `grep "Gelfand|double coset|zonal spherical"` over `docs/theory`
= **0** pre-edit) — opening with *"Edited there, consumed here"* and pointing at both homes.
That is L-064's "name the REGISTER your page owns" applied at page-mint time.

### ⭐⭐ Five inherited numbers; ONE was false, and it was the one in Key Facts

Every `[M]` in the brief/plan was re-run. Results:

| inherited | mine | verdict |
|---|---|---|
| forgery norms `[0.1834, 0.9603]`, 0 of 8 on `S^2` | identical | ✅ |
| `18 .support` string-manipulating reads, all in `orpheus/` | identical (and I added the denominator: **62** reads, 31/31) | ✅ strengthened |
| `87 support=` sites | identical (29 orpheus / 58 tests) | ✅ + predicate |
| `S²/SO(2)`: `P = diag(1,4p₂)`, `det P = 4p₂`, stratum `±1` | `sp.simplify(mine − shipped) == 0` both | ✅ |
| **"the frame's level-2 arrow type-checks — shapes `(8,3) → (8,)`"** | ⛔ **FALSE** — that is `measure.nodes.shape`. The ARROW is `L2[S^2] (8,) → spherical_harmonic_space (3,5)` | ⛔ **refuted** |

⭐ The refuted one had been copied into **Key Facts**, i.e. the sentence a reader quotes. And
the correction produced something better than the fix: `measure.space.name` is `[M]` **`L2[S^2]`**
— the forged level-1 tag propagating *upward verbatim* into the derived level-2 name. *"A derived
name is only as true as what it derives from"* is now a published paragraph that the wrong number
would have hidden. ⟹ **when a relayed shape/number is wrong, ask what the RIGHT reading shows;
it is often the better exhibit.**

⚠ Two more inherited claims did not survive as stated and were replaced by my own census:
`"IndicatorBasis is minted against three manifolds over five sites"` → `[M]` **18 ctor sites, 4
in `orpheus/`**, over three families (index set / `ℝᵈ` at two ranks / energy) — *three* was right,
*five* was not; and `"measure.quotient performs no lookup and no check"` → **half false**: it
*does* gate the MEASURE via `orbit_certificate`; what is ungated is the **TAG**. (Lessons §1's
physics-vs-model sort, on a code claim: two different objects, one gated, one asserted.)

### ⭐⭐ My own reproduction FAILED first, and the failure was mine

Re-deriving step 4, `sp.Matrix(...).subs(x**2 + y**2, p2)` **silently fails** on `4x² + 4y²`
(no literal `x²+y²` node), so I got `det P = 4x²+4y²`, an empty stratum, and an apparent
disagreement with the shipped entry. `factor` before substituting fixes it. Cost ~2 minutes,
because I diagnosed whose failure it was before reporting. ⭐ **I published the trap** as a `⚠`
note in the derivation — a reproduction hazard the next re-deriver will hit, and its presence
is what makes the surrounding table's agreement claim credible.

### ⭐⭐ The finding NOBODY briefed: the tree already performs this lookup

`AngularSymmetry.support` (predating the mint) already computes `S²/G⁰` from the spent group by
catalogue lookup **in the string vocabulary**, raising `NotImplementedError` with the same shape
of message. Measured, three rows: `SO2` → both answer and **agree** (`'[-1,1]'`); `Trivial` →
registry `'S^2'`, catalogue **raises** (a real gap — `S²/{e}` is legal and trivially derivable);
`Oh` → both raise. That one table is (i) the cheapest evidence the mint is a *re-typing* not a
rival, (ii) a measured seam, (iii) a Pattern-2 twin the migration must collapse. Found by
reading `discrete_measures.rst`'s own `orbit_certificate` section, not by any brief.

### The engine-seed ruling needed a COUNT, and the count is not 8

D0.1's falsifiable form is *"could an engine populate these fields without a new type?"*.
`[M]` `dataclasses.fields(Quotient)` — the procedure emits 8 outputs and **6 are slots**; the
**chart** (only its codomain ships, as `realization`) and the **pushforward measure** are not.
Published as `6 of 8` with a per-output table. ⟹ **a ruling whose compliance is claimed but not
counted is not checkable** — and stating the fraction cost one `dataclasses.fields` call.

### ⛔ The nested-markup class only the HTML slice sees — and a source scan that DOES work

`-W` was clean while **two** literals sat inside bold runs (`**Why … ``domain`` … .**`). My
first source regex (L-074's known false-positive generator) reported **88** candidates, 86 of
them junk; the HTML slice (`rfind`-anchored, tags stripped, unescaped) found the true **2**.
⭐ **Then a source scan that agrees exactly**: strip `code-block` bodies first, match
`\*\*(non-greedy, no blank line)\*\*`, and bound the run at ≤200 chars — **2 hits, 0 false
positives**, same set as the HTML. Folded into the re-runnable self-check, so the expensive
build is no longer the only instrument for this class. ⚠ The `**` and backtick runs surviving
in RENDERED text from `x**2` inside a literal and from code-blocks are the false positives the
strip/bound kills.

### ⚠ My own role-resolution check was a FILTER defect, caught by counting

First version matched `:role:`~a.b.C`` only and reported **34** roles clean — the page has
**56**, because 22 use the `` :role:`display <a.b.C>` `` form. Caught by counting all
`:(class|func|meth|attr|mod|exc|data):`` openers and comparing. Fixed, plus a **positive
control** (two synthetic dead roles must both be reported) so a clean run cannot be confused
with a broken scan. ⟹ *count the population your filter is supposed to cover, then compare* —
a validated pattern over the wrong predicate is still the wrong answer.

### Placement/wiring mechanics that worked

- Toctree: `manifolds` inserted **before** `discrete_measures` (a measure needs a manifold to
  live on) + a `what it settles` row. ⚠ My first row said *"the point-set layer underneath both
  of those"* — a dangling reference, since the row above it is `cross_section_data`. Name the
  pages.
- `spaces.rst`: one Key Facts bullet (*"…and a space is NOT a domain"*), one seam-table row
  (`FunctionSpace.manifold`, the level-2 register), `related: [manifolds, …]` in the machine
  header. No Development-history row — the mint is not a space-layer milestone.
- `discrete_measures.rst`: the `support`-is-a-`str` Key Fact is **STILL TRUE** (`Space = str`
  lives at `measure.py:111`, zero `Manifold` consumers) ⟹ **not past-tensed**; given a
  `⚠ Still true of what ships, and no longer the only option` forward pointer instead.
- `error_catalog.rst` ERR-080: a pure ADDITION to the Fix bullet naming the type that supplies
  the first two of its three structural repairs, with `⛔ no production consumer yet`. Nothing
  weakened; the entry's id, title and catchers untouched.

### ⛔ Deliberately NOT done, and why

**No `automodule` for `orpheus.numerics.manifold`.** `[M]` 6 of 48 `automodule` directives are
`orpheus.numerics.*`, and `measure`/`space` — the module's two siblings in the same three-level
stack — are not among them. Surfacing level 1 alone makes `Manifold` a live link while
`FunctionSpace` beside it in the same sentence stays plain text: inconsistent half-surfacing.
Recorded as a seam row with the consequence spelled out (this page's Python-domain roles render
as plain text, so a stale one is invisible at every severity and needs the import-grep gate).

### ⚠ REPORTED, code-only (not fixed — docs pass)

`orpheus/numerics/manifold.py:19` carries `:ref:`ERR-080 <vv-error-ERR-080>``. `[M]`
`ErrorEntryDirective.run()` emits a `container` + `rubric` and **no `nodes.target`**, so no such
label exists; the string appears nowhere else in `docs/` or `orpheus/`. Harmless today (the
module is unrendered) and a guaranteed `undefined label` the day it is `automodule`'d.

---

## L-080 — the two-slot ruling: a gap I published was repaired by its own commit, and a "discriminator" that a construction law forbids from discriminating

**Task (2026-08-31/09-01).** Update `docs/theory/foundations/manifolds.rst` for a data-model
change landing in `orpheus/numerics/manifold.py`: `Quotient` gains a second coordinate slot
(`fundamental_domain`), `singular_stratum` is retyped `tuple[float,...] → Any | None`, and
`Ball` / `FundamentalDomain` are minted. Branch `fix/angular-phantom-support`; the code side
committed **mid-session** as `b55bba56` (I had been reading it uncommitted — check `git log`
before writing a history row).

**Delivered.** +1152 / −85 on the page; 8 new section labels + 1 new eq-label
(`manifold-s2-mod-mirror`, `documented`). `-E -W --keep-going` EXIT=0, **0/0/0 both sides**,
warning-SET diff `{}`. Theory scan 907 → **908** labels / 576 → **577** documented / 0
violations — exactly predicted. `matrix.rst` sentinel 576 → **577**; directive edges **415 →
415** (predicted: no `.. implements::` added). Stock + head-role-patched xref gates **0 dead**
with a live positive control (stock 0 / patched 2 on an injected control page); nexus
`dead_references` **0 / 52**.

### ⭐⭐ A gap I REPORTED had a shelf life of ZERO — its own commit closed it

The page's twin-lookup table shipped a row reading *"``Trivial`` → `NotImplementedError` — ⛔
the catalogue lacks the identity quotient"*. `[M]` `git show fba4205a -- orpheus/...` : the
**same commit that published that table added `_mod_trivial`**, and its own message says so
("Comparing the two showed my catalogue RAISED on S^2/{e} … Fixed by DERIVING it"). The
mechanism is not carelessness: *comparing two implementations is simultaneously what exposes a
gap and what motivates fixing it*, so within one session the table and the tree diverge and the
table is written first.

⟹ **Re-run every gap-claim's own check against the FINAL working tree, after the session's last
code edit — not when the table was drafted.** And publish the outcome as history with its repair
hash, which cannot rot: *"the row read X; the same commit closed it by deriving …; the row is
corrected here as history rather than deleted, because a gap reported into the corpus has the
shortest shelf life of anything on a page."* (Strictly stronger than L-075, which said the shelf
life is *short*.)

### ⭐⭐ The briefed discriminator was un-reproducible — and the real finding is that a CONSTRUCTION LAW forbids it from discriminating

Brief: *"the first catalogued entry could not expose the fork, because for `SO(2)` the chart and
the section coincide in dimension."* `[M]` they coincide in **both** entries — `Interval(-1,1)`
1 vs an `SO(2)` half-meridian `FundamentalDomain(SPHERE,(e_y,−e_y,e_x),…)` 1; `Ball(2)` 2 vs the
σ_y hemisphere 2 — and the new `Quotient.__post_init__` **gates that agreement**. A quantity a
construction law forces to agree cannot tell two cases apart (vv #19's shape at the design tier).

The two reproducible reasons, published instead: (1) **no section of `S² → S²/SO(2)` is
canonical** (every half-meridian is one), so there was nothing to put in a second slot; (2) `[M]`
the tree's SO(2) data is **already chart coordinates** — `gauss_legendre(8).measure.nodes` is
`(8,)` — while `folded_product(4,8).measure.nodes` is `(16,3)`, the base's ambient columns.

⟹ when a brief offers *"they coincide, therefore no fork"*, ask **what would have to be true for
them NOT to coincide** — if the answer is "a gate would raise", the coincidence is a law and the
sentence is inverted.

### ⭐⭐ Two lookups of "the same" fact were NOT a Pattern-2 twin — they take different ARGUMENTS

The page framed `AngularSymmetry.support` and `Manifold.quotient` as a twin to collapse, and I
was about to publish "the catalogue now answers a row the registry has not been extended to".
`[M]` reading the registry: `support` is defined as `S²/G⁰`, the **continuous isotropy** a
dimensional reduction spends, and `GEOMETRY_ANGULAR_SYMMETRY["cylinder"]` is
`continuous_isotropy=Trivial, discrete_residual=Dnh(2)` — a mirror lives in the **discrete
residual Γ**, so `support` *structurally cannot* answer `S²/σ_y`. The honest statement is that
the registry's lookup is the **special case H = G⁰**, so the collapse is
`support = base.quotient(G⁰).realization.name`, not a merge of two tables.

⟹ **before calling two implementations of "the same" lookup a twin, check they take the same
argument.** Same return type + same shape of refusal is not sameness.

⭐ Chasing that distinction found a **latent break nobody had measured**: stage 0 is a string
comparison (`admits_domain` is `measure.support == self.support`), and `[M]`
`GEOMETRY_ANGULAR_SYMMETRY["cylinder"].admits_domain(folded_product(4,8).measure)` is **False**
(`'S^2'` vs `'S^2/sigma_y'`). Latent only because `[M]` `folded_product` is not in
`quadrature_registry` (4 specs ship). Published as a seam with its trigger condition.

### ⭐ The HTML slice caught FOUR nested-markup defects — and its `rfind` anchor silently died

`-W` clean while the page carried 3 × ``**``literal``**`` and 1 × a `:math:` role inside
`*emphasis*` — the last one **leaked the role name as literal text** (`:math:`M/H`` rendered
verbatim). L-074's anchor trap bit again and worse: `rfind('<section id="manifolds">')` matched
**nothing** in this theme (it emits `id="manifolds"` on another element), so the slice was
**length 1** and reported "0 backticks — clean". ⟹ **assert the slice contains known page prose
before believing its verdict** (`assert "Procesi" in text and "<a new heading>" in text`), and
anchor on `role="main"` … `<footer|class="related"|sphinxsidebar`. Fixes: split the bold around
the literal (``**Why …** ``x`` **…**``), and move a role out of emphasis.

### ⭐ Two counts I had published myself were wrong when written, and no gate can see either

*"Nine variants"* — `[M]` `git show b8c05d16:… | grep -c '^class .*(Manifold)'` = **8** at the
mint (and the page's own table listed 8 rows, so the prose disagreed with its own table); ten
now. *"30 test functions, 40 collected rows"* — `[M]` **32 / 44** at that commit, **42 / 56**
now. Both are universals about a roster. ⟹ **publish a roster count with the command that
produces it**, and prefer a second instrument: the generated `matrix.rst`'s
`numerics/test_manifold` row independently confirmed 56.

⚠ And a section title is a count too: mine read *"The four realizations that were tried"* over a
**five**-candidate table, in the same edit.

### The measured refusal matrix, as published (the page's load-bearing evidence)

`[M]` 6 candidates × 5 inputs, cell by cell; "REFUSE (shape)" is a raised `ValueError`, not
`False`. `SPHERE` ADMITS the orbit twins (so `Quotient.contains` becomes bit-for-bit
`SPHERE.contains` — no input separates `M/H` from `M`); `RealSpace(2)` / the square / `Ball(2)`
all ADMIT the **charted** forgery (`[M]` `max|(μ,0)|² = 0.9221566084920586 < 1` — Mode 12, the
chart drops exactly the corrupted coordinate); the hemisphere alone admits the nodes and refuses
both. ⚠ **The shipped two-slot row does not dominate every cell** — it still admits the charted
forgery, correctly, and saying so is what stops the next reader reading the design as a fix for
the chart's blindness.

### Numbers reproduced independently before publishing (all agreed)

Molien `M(t) = 1/((1−t)²(1−t²)) = 1+2t+4t²+6t³+9t⁴+12t⁵`, difference to the free algebra `0`;
minimality `dim(𝔪/𝔪²)` = 2/1/0/0/0 (the `k ≥ 2` predicate — `k ≥ 1` reports "0 new generators"
in every degree, a self-consistent wrong answer); syzygy `I = (0)` by lex elimination, Jacobian
det `−2y`, rank 3; `c∘c⁻¹ = id`, `∂y/∂p_i = −p_i/√(1−p₁²−p₂²)`, `∫_{D²} dp₁dp₂/|y| = 2π`, `dc`
annihilates `e_y` on the stratum; the march seeds `(−√(1−μ_p²),0,μ_p)` on `S²` to **0.0** and on
the stratum (`1−η²−μ² = 0.0`) exactly, while the 16 nodes are strictly interior
(`1−η²−μ² ∈ [0.0378, 0.7549]`); α closes at both level ends; node azimuths `ω/π ∈ {0.125, 0.375,
0.625, 0.875}` ⟹ edges `{0, ¼, ½, ¾, 1}`.

⚠ One sign I had to widen: the Jacobian determinant is `±2x_a` (ordering-dependent); `−2y` only
for the shipped `a = y` ordering. **The rank is what carries the argument** — say so.

### ⚠ REPORTED, code-only (not fixed — docs pass)

1. `orpheus/numerics/manifold.py` `__all__` omits **`Ball`** and **`FundamentalDomain`** — `[M]`
   `[c for c in Manifold.__subclasses__() if c.__name__ not in __all__]` = both, while
   `tests/numerics/test_manifold.py:40` imports them by name. Two public variants outside the
   declared public surface.
2. `AngularRedistribution.mu_start_per_level` holds a **radial** cosine `η = −sinθ_p`, not a
   polar `μ`; and its docstring spells the level's polar cosine `ξ_p`, while `ξ` elsewhere is
   `μ_y`, the azimuthal cosine the fold quotients. Values unambiguous, symbols not.
3. `test_the_half_space_is_CLOSED_because_production_marches_from_it` hard-codes the seeds to 12
   dp; `[M]` those literals sit **4.79e-13** off `S²` against `_MEMBERSHIP_ATOL = 1e-12`, while
   the production values are exact (`0.0`). Green, and on half the tolerance budget for no
   reason.

---

## L-081 — a derived property's doc pass: the brief's target section did not exist, and four present-tense-false claims sat in the pages I was sent to edit

**Task.** #429 tracker **2.1b** (2026-09-01, branch `fix/angular-phantom-support`): document a
concrete `@final` DERIVED property `Basis.invariance_group -> SubgroupOfO3 | None` on the `Basis`
ABC — a `match` on the TYPE of `self.domain` (a `Quotient` of the sphere → its `by`; the bare
`Sphere` → `Trivial`; anything else → `None`). Docs-only; the main agent owned `orpheus/` and
`tests/` (a gate run was in flight).

### (a) The brief's item-2 target DID NOT EXIST — run the brief's own census first

The brief said: *"`docs/theory/foundations/discrete_measures.rst` — wherever `quotient_group` /
the HAS-versus-SPENT distinction is explained for the measure … add the basis-side sentence"*.
`[M]` `grep quotient_group docs/theory/foundations/discrete_measures.rst` → **0 hits**, and the
page's heavy use of *"spent"* is a **different object**: the registry's `G^0`, the continuous
symmetry a *geometry* spends by dimensional reduction (`AngularSymmetry`, §"Spent and owed"), not
the group a *measure* was folded by. Two vocabularies, one word, no overlap in the sentences.

⟹ the honest move was to WRITE the missing home: a new `-` subsection
`measure-has-versus-spent` under "Composition algebra — metadata propagation" that names both
group slots, then the basis-side consequence with a cross-ref. The surplus is where the work is
(L-072/L-075's rule, again).

⚠ And a claim I nearly shipped in that new section: *"Two of the fields in the table above name a
subgroup of O(3)"* — `[M]` **false**: the metadata-propagation table carries `invariance_group`
and NOT `quotient_group`, precisely because the latter is derived and there is nothing to
propagate. Caught by re-reading the table I was pointing at.

### (b) FOUR present-tense-false claims, all repealed by the campaign's OWN sibling steps hours earlier

None was in the brief. All four were true when written and repealed by 2.0c / 2.1 landing the
same day — `plan-authoring` §3's *"a fact can die by being FIXED"*, at corpus scale:

| site | the false claim | repealed by |
|---|---|---|
| `manifolds.rst` §(b) | *"`support` is still a `str`, so `measure.py:331` derives a correct name from an untyped tag"* | 2.0c (`support: Manifold`; the derivation moved to `measure.py:371`, `f"L2[{self.support.name}]"`) |
| `manifolds.rst` three-levels note | *"The slot is therefore `domain`, and it is :ref:`not yet built`"* | 2.1 (abstract on the ABC, `[M]` 6 of 6 subclasses answer) |
| `manifolds.rst` seams table | *"one of them (`basis/indicator_basis.py:284`) **hard-codes** it and `[M]` is already **false** for the energy-grid basis"* | 2.1 (`:355` now `f"L2[coarse_cells({self.domain.name})]"`) |
| `spaces.rst` seams table | *"`[M]` `measure.py:331` still derives `f"L2[{self.support}]"` from a `str`"* | 2.0c |
| `error_catalog.rst` ERR-080 Fix bullet | *"⛔ That type has **no production consumer yet** — it is a capability"* | 2.0c + 2.1 + 2.4 (measure `support`, basis `domain`, the slab's declared orbit space) |

⭐ The ERR-080 one is the sharpest: the SAME entry carries a `✅ Progress 2026-09-01 (tracker
2.4)` block **170 lines above** announcing the first production consumer, and then the Fix bullet
denies one exists. A page contradicting itself, and the stale half is the quotable one (L-077).
⟹ **after a multi-step campaign day, grep the pages for the PREMISE each landed step repealed**,
not for the step's own name. The repair shape that works: keep the numbers, keep the verdict,
tombstone the premise in place (`⛔ This clause read "…" until 2026-09-01: true when written, and
repealed hours later by 2.0c, which is the campaign's own step`).

### (c) The universal I had to fix TWICE, and the exhaustive table that replaced it

Draft 1 said *"the **four** shipped angular rules realise three of the four combinations"*. `[M]`
`vars(Quadrature)` + `isinstance(v, classmethod)` says **FIVE** factories — `gauss_legendre`,
`lebedev`, `level_symmetric`, `product`, `folded_product` — the same finite-roster miss L-074
recorded on a *gate's* roster, now in my own prose. Re-measured, all five:

| rule | `support.name` | HAS (`invariance_group`) | SPENT (`quotient_group`) |
|---|---|---|---|
| `lebedev(17)` | `S^2` | `OctahedralOh` | `None` |
| `level_symmetric(8)` | `S^2` | `OctahedralOh` | `None` |
| `product(4,8)` | `S^2` | `Dnh(8)` | `None` |
| `gauss_legendre(8)` | `S^2/SO2_x` | `Mirror('x')` | `SO2('x')` |
| `folded_product(4,8)` | `S^2/sigma_y` | **`None`** | `Mirror('y')` |

⭐⭐ The two bottom rows are the whole HAS/SPENT argument, and I would not have had them from a
four-row table: the slab carries **two different groups in two slots on one measure** (so no
single field could hold both), and the fold HAS *nothing* **because** it spent σ_y — a fold keeps
one representative per orbit, so the survivors are no longer closed under the mirror. *Spending a
symmetry destroys having it.* The exhaustive census turned a design note into a theorem with a
witness.

⭐ Same fix on the PAIRING table: draft 1 captioned it *"the four pairings the tree can form"*.
Re-run on the pairing the tree **actually forms** — each rule against the basis its own
`angular_frame(2)` binds — gives **5 of 5**, and the finding gets strictly stronger: `[M]`
**exactly 1 of 5 fails** (`Trivial ⊇ SO2('x')` → False), and it is the 1-D one. ⚠ I then wrote
*"which is the same denominator ERR-080's own scope census reports"* — **false**, that census
counts `(constructor, order)` rows (7 of 15). Struck and replaced with an explicit ⚠ saying the
two denominators are not comparable.

### (d) A quoted tracker row is a quotation — check it against the plan

I published *"the tracker read `Basis.invariance_group`, absent, to be answered by the six
subclasses"* in italic quotation marks. `[M]` the plan's verbatim text is *"`Basis.invariance_group`
— absent; derivable for every shipped basis"*, with the *"0 of 6 subclasses answer it"* coming
from a **separate** census clause (§V.5h(b)). My version fused two sources and invented the
"six overrides" framing as if the tracker had asked for it. Corrected to the verbatim pair, with
"which invited six overrides" clearly marked as the inference. ⟹ **anything inside quotation
marks gets grepped out of its source before it ships**; a paraphrase in italics reads as a quote.

### (e) The two claims I had to soften, both over-reach in the same direction

1. *"`invariance_group` — the **largest** group known to map the node set to itself"*. The field
   is a **stored declaration**, not a computed stabiliser, and its `None` means *unspecified*.
   Writing "largest" would have licensed reading it as a maximal-stabiliser guarantee.
2. The `Sphere()` arm's *why*: *"the full degree-L real harmonics share no symmetry"* — `[M]`
   FALSE at `L = 0` (`space.shape == (1,1)`, a single constant, O(3)-invariant), which is the
   very lower-bound caveat the next subsection makes. Rewritten to *"a domain of S² promises no
   invariance, whatever the individual functions happen to have"* — the property is a
   **declaration read off a type**, never a stabiliser.

### (f) Findings reported upward, not fixed (code-side)

- ⛔ **`orpheus/numerics/measure.py:417` cites a method that does not exist.**
  `quotient_group`'s docstring reads *"(:meth:`restrict`, :meth:`consolidate`, :meth:`reorder`,
  …)"*; `[M]` `hasattr(DiscreteMeasure, "reorder")` is **False** and it is the only occurrence
  tree-wide. Found by COPYING the sentence into the corpus — my copy tripped my own role-import
  probe, and nothing else could see it: the role is **unqualified**, so
  `tools/check_docstring_xrefs.py` skips it by design (`DEAD TARGETS: 0`) and nexus
  `dead_references` returns `0 dead / 52 checked`, because `numerics/measure.py` carries no
  `automodule`. ⟹ **the act of quoting a docstring into the corpus is itself an instrument** —
  a fully-qualified copy is decidable where the unqualified original is not.
- Live methods for the sentence: `restrict`, `consolidate`, `partition_by`, `pushforward`,
  `quotient`, `on_orbit_space`, `with_metadata`.

### (g) What worked, and the numbers

- **Self-check-before-build, run to exhaustion.** One ~3 s python script (underline lengths in
  CODE POINTS · file-local marker ladder · per-table column consistency incl. **empty cells**
  (`^     -$`) · `:widths:` sums · corpus-wide anchor/eq-label uniqueness · `:ref:`/`:eq:`/`:doc:`
  resolution · **role import-resolution**, 419 qualified `orpheus.*` roles, 0 dead). ⚠ My first
  version's table parser required a trailing space after the cell dash and reported 2 false
  RAGGED tables — **validate the parser against a known-good member** before believing its
  negatives (the `nexus-tools` positive-control rule, applied to my own instrument).
- **The unqualified-role decision is a CONVENTION question, not a correctness one.** My 6
  unqualified `:class:` roles (`Manifold`, `Quotient`, `EnergyGroups`, `Interval`, `RealSpace`,
  `IndexSet`) match the page's existing 4/3/4/3/1/1 and `[M]` **0** of them resolve to an href
  — the module has no `automodule`, and the page's own seams table records that as a known gap.
  Kept plain, per the no-half-surfacing rule.
- **Build discipline: I broke my own two-build rule and paid FOUR builds**, every extra one
  bought by an edit made after launching. Baseline `-E -W` = **0** W/E/C, EXIT=0; final = **0**,
  EXIT=0; sets identical. ⟹ the self-consistency pass (universals, quotations, denominators) must
  finish *before* the first verification build, not beside it.
- **Matrix prediction held and the sentinel count was the check.** `[M]` 10584 → **10595**
  (+11, all `foundation`, all `numerics/test_basis_domain` 13 → **24**); documented-sentinel count
  **576 → 576**, exactly as predicted since 2.1b adds no `.. math:: :label:`. Predicting BOTH
  registries is what makes the +11 an explanation rather than a coincidence (L-077's rule).

### Quality self-assessment

Derivation depth **5** (the `F(M/H) ≅ F(M)^H` isomorphism written out, with the descent argument,
and the HAS/SPENT asymmetry derived from it) · Cross-references **5** (419 qualified roles
verified by import; 6 new anchors, 12 inbound hrefs verified in the built HTML) · Numerical
evidence **5** (three exhaustive measured tables, every denominator enumerated) · Failed
approaches **4** (the dissolved six-override design is documented as the tracker's own refuted
ask; the `support`-instead-of-`domain` refutation was already on the page) · Code traceability
**5** · Derivation source **3** — the isomorphism is textbook and hand-written; there is no
`derivations/` script for it and one would be ceremony, but the *rating* is honestly low.
**Weakest dimension: derivation source**, structurally so for a type-law page.

---

## L-082

**2026-09-02 — #429 tracker 2.3: the point-set category gets its ARROWS.** Docs-only pass over
`manifolds.rst` (+883/−42, a new `=`-level section on the arrows), `discrete_measures.rst`
(+46/−4), `spherical_harmonics.rst` (+32/−4), `error_catalog.rst` ERR-080 (+68/−3), plus the
regenerated `matrix.rst`. `-E -W` baseline **0** W/E/C → final **0**, EXIT=0.

### (a) A brief's "zero production consumers" about a TYPE is a claim about one CONSTRUCTION

The brief said *"`Ball` (minted 2026-08-31) had 0 production consumers before this step;
`barycentre` is its first"*. `[M]` `git grep "Ball(" HEAD -- orpheus tests` returns **six** lines:
the class definition, one `match` pattern (`case RealSpace(d=d) | Ball(d=d)` in `ambient_dim`),
and **four** constructions — every one of them `Ball(2)`, three in `tests/` and one in production
(`manifold.py:922`, the `S^2/sigma_y` entry's `realization`). So `Ball` *was* consumed. What had
never existed is **`Ball(3)`**, and what is new *in kind* is that a `Ball` is now the **codomain of
an arrow** rather than a field of a catalogue entry.

⟹ census a type's constructions **with their arguments**, and say which *kind* of use is new. The
production docstring carries the same overstatement (`ManifoldMap`'s *"the barycentre map's honest
codomain (`Ball`) had 0 production consumers until it did"*) — reported, not edited.

### (b) ⭐⭐ A retirement can delete the corpus's own WORKED EXAMPLE of a rule that survives

ERR-080's **Lesson** bullet taught the greppable tell — *a constructor writing a membership claim
as a **literal** while its neighbours derive theirs* — and its exhibit was `spherical_product`,
where `support=SPHERE` *"sits between `invariance_group` (COMPUTED …) and `exactness`
(DERIVED …)"*. Tracker 2.3 removed **exactly that literal**. The rule is untouched; its evidence
went present-tense-false, and nothing greps it (the *neighbours* are still there verbatim — `[M]`
both comments survive and now annotate **consecutive** kwargs).

⟹ after any retirement, grep the corpus for the retired construct **as an EXHIBIT**, not only as a
symbol: a sentence of the form *"here X sits between Y and Z"* dies when X moves even though X's
name never appears in a role. Repair = past-tense the exhibit, keep the rule, **re-census the
tell**: `[M]` `grep -rn "support=SPHERE" orpheus/` = **4** live constructions, of which 3 are
honest tabulations (`UNIFORM_ON_SPHERE`, Lebedev, level-symmetric) and 1 is the forgery. A literal
is a *tell*, not a verdict — what makes the fourth a forgery is that `Sphere().contains` refuses it.

### (c) A "phase N mints the typed X" prediction: the PHASE lands, a type lands, X does not

`manifolds.rst` predicted *"the choice belongs to the step that mints the typed `Chart` — not to
the orbit-space derivation"* and named tracker 2.3. 2.3 landed **on the day**, minted
`ManifoldMap`, and **made no section**. Three of `coding-standards`' falsified-prediction
components split apart: PHASE right, MECHANISM half-right (a typed map, differently named),
DELIVERABLE absent. ⭐ The *reason* is the transferable half and it came from the naming ruling: a
chart is `M ⊃ U → ℝⁿ`, and **only the inverse of the Archimedes map is one** — the retraction lands
on a `Quotient`, the barycentre on a `Ball` — so a type called `Chart` would have mis-described two
of its own three instances. `[M]` `Quotient.fundamental_domain` still has **zero** readers outside
`manifold.py` (the field, its `__post_init__` gate, `contains`'s dispatch, three builders).

### (d) ⛔ The xref gate's `head_role` blindness is at the HEAD-CHECK line — and an inert patch reads exactly like a clean tree

L-062/L-067 record the one-line fix as `head_role = "mod" if "." in target else role`. I applied it
to the **first** `candidate_paths(target, namespaces, role)` call. `[M]` patched == stock ==
`DEAD TARGETS: 0` — which reads as *"the corpus is clean"*. It is not where the decline happens:
`judge()` returns `Judgement(Outcome.ALIVE|...)` from that call and only later runs the head-check

```python
head = target.split(".")[0]
if not any(lookup(c) [0] for c in candidate_paths(head, namespaces, role)):
    return Judgement(Outcome.DECLINED)
```

— and `candidate_paths("orpheus", (), "class")` is `()` on an `.rst` page's empty namespace, so
`any(())` is False ⟹ DECLINED. Patch **that** line. `[M]` with a throwaway `docs/_ctl.rst`
carrying two dead + one live role: **stock 0 dead**, **patched 2 dead / 2 sites**, `decidable`
5797 → 5799. Corpus reading, control removed: patched over `docs orpheus tests` = **0 dead**,
1006 files / 16 886 roles / 14 184 decidable.

⟹ **the control must SPLIT the two gates.** L-071 says "run an end-to-end positive control"; the
sharpening is that *stock == patched* is itself the tell that the patch is inert — an equal reading
carries no information about where the blindness lives. (The `head_role` fix is still **unlanded**
in `tools/check_docstring_xrefs.py`; the ERR-026 branch owes it.)

### (e) ⛔⛔ A trailing SPACE before a closing role backtick swallows the sentence — `-W` is silent

`` :math:`\max\lvert … \rvert = ` `` (note the space) does not close the role. The rendered HTML
carried a literal `` ` **0.0** for :math:`a \in \{x,y,z\}\) `` — raw markup, on a build that
reported **0 warnings**. Caught only by the tag-stripped HTML scan for `**` / ` `` `.

⟹ two gates, both cheap: the **render** scan (authoritative), and a corpus-wide source regex that
localises it in one line — ``:(?:math|ref|eq|doc|class|func|meth|attr|mod|data|exc|cite|term):`[^`]*\s` ``.
`[M]` after the fix: **0** corpus-wide, so mine was the only one.

### (f) Smartquotes mis-directs a closing `"` that follows an inline literal — extend the quote

`` **… the typed** ``Chart``\ **" until 2026-09-02.** `` renders `Chart“ until` — a LEFT quote where
a right one belongs, because the `"` has no preceding word character. Detector:
`re.finditer(r"“(?=\s*(until|and|,|\.))", stripped_html)` (⚠ it also flags a quotation that *opens*
with one of those words — read the hit, don't count it). ⭐ The fix that costs nothing: **end the
quoted fragment on a WORD by extending it**, since the extension is usually verbatim anyway
(`… the typed `Chart` — not to the orbit-space derivation"`). Shortening would have been the
natural move and is the wrong one.

### (g) ⭐⭐ A composition/functoriality law is measurable ON THE SHIPPED OBJECT — find the chain

`(ψ∘φ)_*μ = ψ_*(φ_*μ)` reads as an abstract type law until you look for a shipped chain. `[M]`
`Quadrature.folded_product` **is** one: `[-1,1]×S¹ --archimedes--> S² --retraction--> S²/σ_y`, and
`retraction @ archimedes` type-checks. One-shot vs two-step vs the shipped rule: `array_equal` on
nodes and weights, support by **identity** with the catalogue entry, on **5 of 5** configurations.
⚠ And the fixture must be stated: the shipped fold uses the **staggered** circle rule
(`Σ = ∅`); `[M]` node-aligned puts **4** nodes on `Σ = {ξ=0}` and folds 16 atoms into **10** orbits
(sizes `[1,1,1,1,2,2,2,2,2,2]`, the singletons being the fixed points) against staggered's **8**,
all of size 2. The functoriality half is fixture-independent; the agreement-with-the-shipped-rule
half is a statement about which circle rule ships.

### (h) An HTML slice anchored on the NEXT SECTION'S TITLE can land inside your own section

L-074 says anchor with `rfind` because the TOC repeats titles. Insufficient: a `:ref:` renders as
the **target's title**, so if your new section cross-references the next one, `rfind(next_title)`
lands *inside* you. `[M]` my first slice read **1 659** chars of a **21 909**-char section and
reported "0 leaks" — a designed-green reading. ⟹ anchor both ends on a **distinctive sentence**,
and sanity-check the slice LENGTH against the source's line count before believing its verdict.

### (i) Reproduced, and worth carrying: the ERR-080 restatement

`[M]` on `gauss_legendre(8)`: `_harmonic_frame_measure()`'s nodes are `np.array_equal` to
`barycentre(measure.support)(measure.nodes)`; `Ball(3).contains` → **True**, `Sphere().contains` →
**False**, norms `0.183435 … 0.960290`. So the defect is *the barycentre map with a forged
codomain* — the arithmetic was never wrong, a **type** is. That single sentence replaced three
paragraphs of mechanism in the catalogue entry and in two theory pages.

Other reproductions, all independent of the brief: 60-of-60 product bit-identity (nodes + weights);
`π ∘ archimedes_a = pr₁` **bit-exactly** over 500 random `(μ,φ)` per axis, `‖image‖−1 ≤ 2.22e-16`;
`1−‖μê_a‖² = 1−μ²` to **0.0**; `_embedded_nodes == barycentre` on 12 rows; the `manifold ⇄
exactness` two-cycle killing **5 of 5** fresh import orders on a throwaway package with the measured
topology (positive control: 5 of 5 clean with the `TYPE_CHECKING` guard restored); `exactness.py`
imports `manifold` at module scope **twice** (`:115`, `:116`).

### Quality self-assessment (Directive 3)

Derivation depth **5** (the barycentre geometry, the hat-box, `π∘φ = pr₁`, the fold as a two-arrow
chain, all derived and measured) · Cross-references **5** (348 fully-qualified roles checked by
import, 0 dead; every `:ref:`/`:eq:`/`:doc:` resolved) · Numerical evidence **5** (60/60, 12/12,
5/5, 500×3, two positive controls) · Failed approaches **4** (the falsified `Chart` prediction, the
refuted `Ball` claim, the inert xref patch — all published) · Code traceability **5** ·
Derivation source **3** — no `derivations/` script; the algebra (hat-box, orbit barycentre,
Procesi–Schwarz) is textbook and already lives in the catalogue's SymPy regression tests.
**Weakest dimension: derivation source**, structurally so for a type-law page.

## L-083 — the catalogue entry's own arrow (#429 tracker 3.1): a "picklable" claim that splits in two, a stabiliser bigger than its group, and an import cycle whose verdict is placement-independent

**Task.** Docs-only pass for #429 tracker 3.1 on branch `fix/angular-phantom-support`
(HEAD `3623adc2` + uncommitted code). Two new `Quotient` fields — `orbit_coordinates`
(+ the derived `quotient_map` property) and `reference` — plus the registry twin
`AngularSymmetry.reference` collapsing onto the entry. Edited
`docs/theory/foundations/manifolds.rst` (+1226/−…), `discrete_measures.rst` (+58),
`error_catalog.rst` (+55); `matrix.rst` regenerated. Baseline and final `-E -W`
both **0** W/E/C, EXIT=0.

### (a) ⛔ REFUTED: "picklable, `[M]` pickle round-trip equal" is TWO claims and only one holds

The brief said the `functools.partial(_ambient_columns, …)` spelling is
*"picklable, `[M]` pickle round-trip equal"*. `[M]` over all **7** shipped quotients
of `S^2` (six catalogue keys + the derived identity): the callable round-trips with
no `PicklingError` and gives bit-identical output **7 of 7** — that is the property
the spelling was chosen for, and it holds, and a `lambda` would fail it. But
`pickle.loads(pickle.dumps(f)) == f` is **1 of 7**: `functools.partial` inherits
`object.__eq__`, so only the trivial entry's plain module-level `_all_coordinates`
compares equal, and for a reason that does not generalise (a function pickles *by
reference*, so unpickling returns the same object).

⭐ **The refutation produced a better argument than the claim.** The load-bearing
consequence is what `field(compare=False)` *buys*: `[M]` `pickle.loads(pickle.dumps(q))
== q` is **True 7 of 7** precisely BECAUSE the callable is excluded from `__eq__` —
and the entry is memoised into a cache and used as a dict key. So the exclusion is
load-bearing for **serialisation**, not merely "a function has no value equality"
(which is what the production comment says, and is true but weaker). ⟹ when a brief
conflates *X works* with *X compares equal*, measure both; the gap is usually where
the design's real justification lives.

### (b) ⭐⭐ A quotient map's own stabiliser can be BIGGER than the group the entry names — an EXACT predicate that still under-determines

Writing the "H-invariance, with a negative leg" row I chose the natural negative
element (a mirror about a different axis) and it **failed on 3 of 7** — the three
`SO2(a)` entries. Not a defect: `[M]` `π_a` is bit-exactly unchanged under `σ_b`
for `b ≠ a`, and `σ_b ∉ SO(2)_a`. The reason is structural — a reflection in a plane
*containing* the axis maps each constant-`μ` circle to itself, so `O(2)_a` and
`SO(2)_a` induce the **same orbit partition** of `S²`, hence the same orbit space,
invariants and map. The mirror family is the contrast that makes it checkable: there
the stabiliser is exactly `⟨σ_a⟩` and `σ_x` genuinely moves `π_y`'s image.

⟹ **a quotient map determines the PARTITION, and the partition does not determine
the group.** `Quotient.by` is a *declaration*; nothing derives it, and
`Basis.invariance_group` reads it. ⭐ And the epistemic shape is one register sharper
than ERR-072: that one is a group predicate under-determining its group because it was
**sampled**; this one under-determines while being **exact**, so no refinement can
fix it. Re-run with a rotation about a different axis the negative leg is **7 of 7**.
⟹ a "negative leg" must be chosen outside the measured functional's TRUE stabiliser,
not outside the group you happen to be documenting (vv #17's control clause, at the
design tier).

### (c) ⭐⭐ A §6d import-cycle verdict needs BOTH module-scope placements, and the package `__init__` is what makes it placement-independent

The brief measured *"function-scope alive 7/7, module-scope dead 7/7 —
`ImportError: cannot import name 'DiscreteMeasure'`"*. Reproduced on my own **renamed
shadow copy** (`shadowpkg`, so the editable install's `sys.meta_path` finder cannot
serve the real tree — every subprocess prints the `__file__` it loaded; L-050's trap,
defeated by renaming rather than by stripping the finder). Verdict reproduced; the
**message did not**, and chasing that produced the better section:

| placement | alive | first failure |
|---|---|---|
| function scope (shipped) | **7 of 7** | — (the positive control) |
| module scope, **top** of file | **0 of 7** | `cannot import name 'Manifold'` — `exactness` is reached first |
| module scope, **bottom**, every name bound | **0 of 7** | `cannot import name 'DiscreteMeasure'` — one hop on, in `generating_measure`'s own `measure` import |

⭐ Testing the *most favourable* module-scope position is what turns "module scope is
dead" from a sample into a claim: bottom-placement is the one a reader would propose
as the fix, and it is dead too. ⭐⭐ And the reason all seven entry points agree is
`orpheus/numerics/__init__.py` eagerly importing `.measure` at module scope — the
package body runs first, so the entry point has no say and the effective order is
fixed. That is publishable: it says the cycle is **not** order-dependent, unlike the
one the module's guard was written for (which is exactly what let a smoke test pass
on a broken façade).

⭐ **The transferable rule the section is built on:** a `TYPE_CHECKING` guard defers a
**name** and can never carry a **value**. 3.1 needs both — the *type*
`ReferenceMeasure` (annotation only, erased under `from __future__ import
annotations`) and the *value* `LEGENDRE` — so it needs two different mechanisms, and
the corpus had been treating "the cycle blocks the import" as "the cycle blocks the
slot". `[M]` the type's import is free because `ReferenceMeasure` is a
`@runtime_checkable` `Protocol` of three members that `LEGENDRE` (a
`GeneratingMeasure`) satisfies **structurally**.

### (d) ⭐ Its safety condition is a property of the CALL SITES, so it can be broken from outside the module

The function-scope import is safe iff the function never runs during module
initialisation. `[M]` by AST over `orpheus/` **with call depth tracked**: **8** calls
that can mint a quotient (7 `.quotient(…)` + 1 `.on_orbit_space(…)`), **0 of 8** at
module scope. ⭐ Publish the *total* as the positive control — a filter finding zero
CALLS prints the same safe-looking zero as one finding zero MODULE-SCOPE calls, and
only the first number separates them. And say the hazard is external: a future
module-scope `SPHERE.quotient(...)` anywhere re-opens the cycle from the other end,
gated by nothing; the check's predicate is *call depth zero*, not the spelling.

### (e) The brief's other numbers, re-derived

| brief | mine | verdict |
|---|---|---|
| `dataclasses.fields(Quotient)` = 12 | 12 | ✅ |
| `manifold.py` `:612 / :667 / :704 / :1084 / :1099 / :1194 / :1238 / :1371 / :1421` | all exact | ✅ |
| registry `:910` `.reference`, `:979` `UNIFORM_ON_SPHERE` | exact; the `support.reference` read is `:965`, brief said `:964` | ✅ (±1) |
| four geometries answer as before, slab/sphere by identity | reproduced, `is`-identity confirmed | ✅ |
| π ∘ φ = pr₁ bit-exact 12 of 12 | 12 of 12 | ✅ |
| β ∘ π = axial projection 3 of 3 | 3 of 3 | ✅ |
| ∫μ² d(π\_\*μ) ≈ 4π/3 | `4.18879020478639`, **1 ULP** from 4π/3 | ✅, sharpened |
| symbolic-invariants tie 5 of 5 | **7 of 7** (brief's 5 was a sample of the 7-entry roster) | ✅, widened |
| `.reference` reads in `orpheus/` = 9 | **10** by AST (the new `support.reference` is the 10th) | ⚠ off by one |
| 3 `match Quotient(...)` patterns incl. one "inside `barycentre`" | 3 patterns, but `barycentre` uses `isinstance`; the third is `ambient_dim` `:1484` | ⚠ site mis-attributed, conclusion right |
| `class Descent` / `class IsometryGroup` = 0 | 0 / 0 | ✅ |

### (f) ⭐ The honest-scope finding the brief did not state: 3.1's two halves have OPPOSITE consumption

`[M]` over `orpheus/`: `Quotient.reference` has **one** production reader
(`registry.py:965`) while `quotient_map` and `orbit_coordinates` have **zero** outside
their own module — ten and three occurrences respectively, all in
`tests/numerics/test_manifold.py`. So half of 3.1 is CONSUMED and half is a
capability. Stated in Key Facts, the new section's `.. warning::`, the ERR-080 block
and the changelog row, per L-079's three-places rule for a zero-consumer mint.

### (g) ⭐ A gotcha found by writing, not by grepping: the reference lives in the CHART's coordinates

`[M]` `GEOMETRY_ANGULAR_SYMMETRY['slab'].support.name` is `'S^2/SO2_x'` while its
`reference.support.name` is `'[-1,1]'`, and `grep` finds **no** read of
`reference.support` in `orpheus/`. Not a defect — `2π dμ` is naturally written in the
invariant, which is where `quotient_map`'s image lands — but a future gate must assert
`entry.reference.support == entry.realization`, **never** `== entry`: the second would
demand a measure carry an axis the measure genuinely does not know. This is the
two-coordinate-systems asymmetry one register down, and it needed a `.. warning::`
because nothing in the tree states it.

### (h) Instrument notes

- **My HTML render slice was designed-green on the first attempt**: `body.find("The
  second operand…")` matched the `.. contents::` TOC entry, giving a **749**-char
  slice that reported "0 backticks". L-082's trap, hit again. Fixed by anchoring on
  the section `id=`s with `rfind` for the end **and asserting the slice length**
  (54 315 chars) plus seven known-content probes. Result: 0 stray backticks (the 2
  found are inside a `.. code-block:: text` quoting the production `NotImplementedError`
  verbatim, which contains a literal backtick), 0 leaked roles, 0 smartquote
  mis-directions, 21 `<cite>` all `[M]`, 28 internal links rendering.
- **The xref gate needed its split control again.** With a throwaway `docs/_arch31_ctl.rst`
  carrying 2 dead + 1 live role: **stock 0 dead**, **patched 2 dead / 2 sites**,
  `decidable` +2 — the split is what proves the patch is live. Control removed:
  patched **0 dead** over 1006 files / 16 967 roles / 14 263 decidable. L-082's
  `head_role` fix is still **UNLANDED** in `tools/check_docstring_xrefs.py`.
- **My own structural self-check's one "failure" is the known false positive**: the
  duplicate `boltzmann` eq-label is a `.. code-block:: rst` EXAMPLE in
  `verification/harness.rst`. A label scanner that cannot see literal blocks reports
  the corpus documenting itself.
- **`rm -rf` inside a compound Bash command is refused here** (as `process-discipline`
  records) — the shadow copy at `scratch/_arch31_shadow/` survives; its rebuild recipe
  is in `scratch/_arch31_cycle.py`.
- **Predicted and confirmed:** +1 `documented` label ⟹ `matrix.rst` sentinels
  **577 → 578**, `manifold-quotient-pushforward` in the documented list. The test-count
  moves in the same regenerated file (10616 → **10638**, `test_manifold` 70 → **91**,
  `test_registry` 74 → **75**) are the CODE side's uncommitted tests absorbed by the
  `-E` rebuild — a legitimate by-product, reported not reverted.

### (i) Quality self-assessment

Derivation depth **5** (the hat-box and the weighted-disk Jacobian both derived from
scratch and mass-checked to 4π) · Cross-references **5** (135 project roles, 0 dead;
28 rendering links in the new sections) · Numerical evidence **5** (every table row
re-measured; three brief numbers corrected) · Failed approaches **5** (three
tombstoned rows, one discharged deferral, one prediction verified verbatim) · Code
traceability **5** · Derivation source **3** — the two pushforwards are derived in the
PAGE with a SymPy cross-check I ran in `scratch/`, not in a tracked
`derivations/` module. ⟹ **weakest dimension: derivation source.** The honest ask is
below.

---

## L-084 — the moment space's single home (#429 tracker 2.5): a page that already ⛔-stamps the metric the step binds, and a table that was 3× more measurable at the OPERATOR level

**Task.** Document #429 tracker 2.5 — *the angular moment space is READ off the
frame, never minted from `L`*. Doc-only (a 13-tree pytest gate + a mutation
battery were running; `orpheus/` and `tests/` were READ-ONLY). Baseline `-E -W`
**0** W/E/C EXIT=0; final **0**, EXIT=0; xref gate `DEAD TARGETS: 0`; nexus
`dead_references` 0 dead / 52 checked both sides.

### (a) ⭐⭐ The brief's target pages were the wrong two, and the RIGHT page already
carries a ⛔ verdict AGAINST the thing the step does

The brief said the new section belongs on `frame.rst` **or** `spaces.rst`,
"whichever owns the moment-space derivation". `[M]` `grep -rn moment_space_on
docs/` = **0** — neither owns it, and the phrase is not in the corpus at all. The
decisive test was `AGENT.md`'s *self-undermining-if-homed-elsewhere*: the step's
load-bearing choice is `basis.space` (continuum Gram) over `basis_space`
(Parseval `G⁻¹`), and **only `frame.rst` owns that distinction** (F-0's
§*"What was wrong before"*). Homing it anywhere else would have restated F-0's
metric material as a twin.

⚠ **And the hazard nobody flagged:** `spherical_harmonics.rst`'s three-metric
table stamps the continuum Gram with *"⛔ **What the frame exposed before F-0** —
the wrong side for covariant moments"*, and the step binds Λ's ends to **exactly
that space**. A reader lands on that ⛔ and concludes the step re-opened F-0. The
paragraph that resolves it is now the section's centre: F-0's verdict is about
the **analysis face's CODOMAIN** (where the value IS a covariant moment); Λ's ends
are an **endomorphism's** domain/codomain, and an ℓ-diagonal metric commutes with
a per-ℓ scalar. ⟹ **when a step binds an object the corpus has already condemned
in a different role, the doc's first job is to name the ROLE the condemnation was
scoped to** — not to repeat the step's own "it is bit-identical" argument.

### (b) ⭐⭐ An OPERATOR-movement table measured with a probe vector is a one-draw
reading — build the matrix column by column and the numbers get BETTER, not just safer

L-076 says this for a gate docstring; here it changed the published table. Probe
vector (`default_rng(3)`, relative Frobenius) vs column-by-column (`e_k` through
both arms):

| quantity | one draw | draw-free (operator) |
|---|---|---|
| inert rows | `≤1e-12` (23/33) | `≤1.045e-16` (23/33) |
| DIAGONAL movers | `4.26e-2 … 8.68e-2` | `9.70e-2 … 1.372e-1` |
| DENSE movers | `9.07e-2 … 9.881e-1` | `1.082e-1 … 1.5839` |
| `Λ* = Λᵀ` under the continuum end | `≤1.82e-16` | **exactly `0.0`, 33/33** |

The last row is the one worth having: at operator level the `g_C Λᵀ g_C⁻¹`
sandwich is a per-mode scalar times its own reciprocal, so the identity is
**bit-exact**, and the `1.8e-16` is the reduction order of a random application.
Publishing the draw-free form also moved the headline from "up to 99 %" to
"up to 158 %".

⭐ **And read the movers by IDENTITY, not by size.** All 10 observable rows are
`gauss_legendre` or `folded_product` at `L ≥ 1` — the two families whose discrete
Gram is `m`-dependent, i.e. *exactly* ERR-080's forged-azimuth rules and the σ_y
fold. On the six full-sphere degree-exact rules the fork is inert on all 18 rows.
So the wrong choice would have been invisible to every full-sphere regression
fixture and wrong on precisely the rules the campaign is repairing.

### (c) ⭐⭐ A brief's `[M]` census needs its PREDICATE — "eight homes" was right and
its command was not

The brief said *"eight homes … seven production sites"*. Re-run, `git grep -n
"SphericalHarmonicSpace.from_L" HEAD -- orpheus/` returns **13 lines**: **8
executable calls** (7 re-mints + the basis's own `space`) and **5 docstring
mentions**. The counts are right; the command as cited returns a number that
matches neither. Published form: *"13 lines, of which 8 are executable calls"*.
Post-step the same command returns **6** lines, **1** executable — and the
honest rider is that `SphericalHarmonicSpace.truncated` calls `type(self).from_L`
**inside the space's own module**, which the grep cannot see and which is where a
family is entitled to name itself.

Likewise the brief's *"metric-DIFFERENT on 12 of 12 shipped (rule, L) rows"*: my
census over the gate's own roster is **33 of 33** (11 rule constructions drawn
from all **five** `Quadrature` classmethod factories × `L ∈ {0,1,2}`). The 12
is not reproducible from anything the brief names. And the brief's
*"`apply_metric` moves 96–161 %"* did not reproduce under any norm I tried —
replaced by the exact per-ℓ ratio `[(2ℓ+1)/4π]²` (`6.33e-3 / 5.70e-2 / 1.58e-1`),
which is draw-free and reproduces the F-0 page's own `157.9 / 17.5 / 6.3`.

### (d) ⭐⭐ The step's own acceptance measurement was WEAKER than the one available —
and the stronger one is a statement about a *strict-xfail* gate

The brief cited slab fluxes `array_equal` at `L ∈ {0,1,2}` from a `scratch/`
`.npz` I must not cite (ephemeral path). Rebuilt from the ERR-080 gate's own
fixture (1-group infinite medium, `gauss_legendre(8)`, 4 cells, reflective ×2,
uniform per-ordinate source, Krylov `inner_tol=1e-13`, `max_inner=5000`) and run
against a `git archive HEAD` tree in a subprocess with the editable finder
stripped (`sys.meta_path` filtered, `orpheus.__file__` asserted): `array_equal`
at **L = 0, 1, 2 AND 3**, `max|Δ| = 0.0` on all four.

⭐ The extra order is the point, and it is a V&V argument rather than a bigger
number: **a pre-step must be bit-identical even where the answer is WRONG**,
because `L = 2, 3` are `xfail(strict=True)` rows and any movement there could
flip an XPASS without repairing anything. Framed that way, "nothing moved at the
defective orders" stops being a disappointment and becomes the acceptance
criterion.

### (e) ⭐ Reading the tests to write the doc surfaced an UNDOCUMENTED asymmetry
older than the step

`tests/transport/frames/test_harmonic_frame.py` asserts `face.codomain ==
HarmonicMomentFlux.zeros_for_mesh_and_L(m, L).space`. `[M]` on a 2-group slab at
`L = 0,1,2` that is `True` at every order **and the two heads' metrics differ at
every order** — the face's head is the frame's *dressed* `basis_space` (matrix
metric at `L = 2`, `inner_product_weights is None`), the field's head is the
*continuum* one. Identity is `(name, shape)`, so nothing in the tree can tell
them apart. Unchanged by 2.5 (pre-step the field head was `from_L(L)`, same
continuum Gram) — so it is a **gotcha to record**, not a defect to report, and it
belongs in the doc precisely because a reader just told *"the space is read off
the frame"* will assume the field inherits the face's metric.

### (f) Predicted-then-measured generated artefact

One `documented` eq-label ⟹ `matrix.rst` sentinel **578 → 579**, predicted from
`tests/_harness/audit._scan_theory_equations` before the build and confirmed
after. The build also absorbed the CODE side's +37 collected tests
(`test_moment_space_is_read_off_the_frame` 36 rows + `test_harmonic_frame` +1) —
a legitimate dirty-tree by-product, reported not reverted.

### (g) Instruments, and one of mine that lied

- The added-text nested-markup scan found exactly one real defect — a `:math:`
  role inside `**bold**` in my own ERR-080 block — among 9 candidates; the other
  8 were `* -` list-table bullets that my `*…*` emphasis regex reads as emphasis.
  ⟹ a list-table body makes an emphasis scan ~90 % false positive; adjudicate,
  never count.
- ⛔ **My HTML slice was 302 177 chars — the whole page — because I anchored on
  the bare label text, which also appears in the TOC.** It "passed" while
  measuring the wrong region, and only the length looked wrong. Anchoring on
  `id="<label>"` with `rfind` gave 38 183 chars and the sanity phrases. (L-080's
  rule, re-broken by using the label rather than the `id=` attribute.)
- Single-backtick scan over the diff caught `` `.H` `` (a title-reference, not a
  literal) in my prose. Its 40 sibling hits were all inside the manifolds
  `code-block:: yaml` status string, where single backticks are the existing
  convention — the scan is only usable per-file-context.

### Quality self-assessment (Directive 3)

| dimension | score | note |
|---|---|---|
| Derivation depth | 5 | the fork derived from commutation of an ℓ-diagonal metric with a per-ℓ scalar, with the exact reciprocal ratio |
| Cross-references | 5 | 17 qualified roles, all import-resolved; 4 `:ref:` hrefs verified in the built HTML |
| Numerical evidence | 5 | 33-row census, operator-level draw-free table, end-to-end pre/post `array_equal` on 4 orders |
| Failed approaches | 4 | the F-0 hazard and the demoted canary are recorded; no *falsified* alternative existed to narrate |
| Code traceability | 5 | every claim carries its `file:line` or its command |
| **Derivation source** | **2** | ⛔ again: all six measurements live in `scratch/` probes, not a tracked `derivations/` module. Same weakest dimension as L-083. |

**Standing ask (third session running):** a tracked home for frame/metric
probes — `derivations/numerics/frame_metric_forks.py` exposing
`derive_parseval_vs_continuum_ratio()` and `derive_lambda_adjoint_metric_inertia()`
returning the two tables above — so the page can cite a module instead of
describing a construction.

## L-085 — the FIX lands: a repair changes the BASIS, and the corpus's shape contract turns out to be one family's

**#429's fused commit (0.1b + 0.6 + 2.2 + 3.4 + 3.4b), ERR-080 CLOSED, 2026-09-02.**
13 pages, +1756/−240, `-E -W` 0→0 EXIT=0, patched-xref `DEAD TARGETS: 0` (with an
end-to-end positive control reading 2), nexus `dead_references` 0/52, sentinels
579 → **582** (exactly my three new `documented` eq-labels).

**(a) ⭐⭐ A CORPUS-WIDE SHAPE CONTRACT can be one family's layout wearing a
universal.** The brief named `operator_algebra`/`adjoint` for "Λ's contraction by
head rank". `[M]` the claim `(L+1, 2L+1, ng, *spatial)` is asserted as THE moment
layout at **9 sites over 7 pages** — including `conventions/indexing_and_layout.rst`
(the convention page, ×3) and `methods/sn/slab_multigroup.rst`, whose subject is
the one chart where it is now FALSE (`[M]` a `gauss_legendre(8)` phase space gives
`(3, 1, 4)`, i.e. `(L+1, ng, nx)`; `level_symmetric(8)` gives `(3, 5, 1, 4)`).
⟹ **when a repair adds a SECOND member to a family the corpus only ever had one of,
grep the SHAPE, not the symbol** — the symbol (`HarmonicMomentFlux`) is unchanged
and every hit resolves.

**(b) ⭐⭐ The flagship WITNESS of a neighbouring page can be the defect, tabulated.**
`frame.rst`'s dense-arm section had the slab GL(8) L=2 Gram as its witness, with rows
reading *"live slots per degree [1,1,3]"* and *"diagonal 0.8 on the two surviving
ℓ=2, m≠0 slots"*. Those two `0.8`s ARE ERR-080's fabricated columns. `[M]` post-fix
the same frame is **DIAGONAL**, off-diagonal `8.8e-17`, diagonal `2/(2ℓ+1)` exactly,
and `spaces.rst`'s three-way frame-square table moves it from *"no G_ℓ exists at any
metric, residual 0.30–10.2"* into the row where the closure **HOLDS**
(`[M]` ≤ 5.1e-16 over 200 seeds). ⟹ preserve the numbers, tombstone the
interpretation, and RE-MEASURE the replacement — mine found a new witness of the
*good* kind (`gauss_legendre(2)` L=2: DENSE, rank-deficient, closure holds at
≤1.6e-15) whose rank deficiency has a **closed form** (P_n vanishes at GL_n's own
roots), which is strictly stronger than the row it replaces.

**(c) ⚠ Two brief numbers refuted by re-measurement, both about the SAME sentence's
two observables.** The production docstring says *"with pure `lpmv` [the slab flux]
moves by 4e-16"*. `[M]` on ERR-080's own fixture, against a pinned `git archive HEAD`
tree: `array_equal` at L=0 either way, and at **L=1 the flux moves by 2.753e-14** —
~60× the quoted figure, because `4.44e-16/8.88e-16` is the memo's **TABLE**-level
number and the docstring's sentence is about the **FLUX**. A `1e-16` table
perturbation is amplified by the Krylov solve. Same shape one line over: *"`eval_legendre`
differs at ℓ≥2 by up to 8e-16"* — `[M]` my max over GL(2,4,8,16) at L≤4 is **4.777e-16**;
the 8.0e-16 is over a wider (unstated) rule set. ⟹ a float-agreement figure needs its
OBSERVABLE named, not just its fixture (L-051's ratio rule, moved from a ratio to a
perturbation).

**(d) ⭐ A "still OPEN" clause can be repealed while its NEIGHBOUR stays exactly
true, in one sentence.** ERR-080's Key Facts bullet read *"the membership PREDICATE
is still not enforced at construction … so the forged measure is still constructible.
ERR-080 is open"*. `[M]` half 1 is STILL TRUE — a forged `DiscreteMeasure` on `SPHERE`
constructs today — and half 2 is false: what closed the defect is the refusal at the
BASIS (0.6) and at the FRAME (G0), not at the measure. Splitting the sentence in place
is what keeps the seams table honest; past-tensing the whole bullet would have deleted
a live seam (2.0b).

**(e) ⛔ The pages I EDITED carry 152 pre-existing nested-markup leaks and I added 8
of my own before catching them.** The authoritative gate is the RENDERED HTML with
`<pre>`/`<code>` stripped, sliced BETWEEN consecutive section `id=`s (an unbounded
60 000-char slice reported leaks from four sections away). A source-side regex over
the diff was ~90 % false-positive. My eight were all one shape: **bold or a literal
nested inside an italic quotation** (`*"… **X** …"*`, `*"… ``Y`` …"*`) — the natural
spelling for a §3 tombstone. ⟹ tombstone a quoted claim with plain quotes plus
`, verbatim,`, never with an outer `*…*`. Attribution after the fix: `[M]` **0 of 176**
leaks trace to a line I added (checked by matching each leak's stripped context
against my diff's `+` lines).

**(f) ⭐ The test tree MOVED under me mid-session, and the moving part was the
evidence I was about to cite.** At my first census `tests/numerics/test_legendre_basis.py`
and `test_descent.py` **did not exist** (the ERR-080 gate's docstring names them in the
present tense); by the final build both ship (15/32 and 9/20 rows) and
`numerics/test_manifold` had gone 70 → **108** rows. ⟹ re-run every count against the
FINAL tree after the last build, and prefer the generated `matrix.rst` row as the
second instrument (L-080, one level up: here the shelf life was ~90 minutes).

**(g) The G0 predicate is ONE arrow, and that is the publishable insight.** The corpus
had stated the pairing as the lattice containment `G_spent ⊆ G_have`. `[M]` all seven
shipped pairings: the containment is only the `K ⊆ H` arm, and the other two admits are
pairings it **cannot express** — in particular Legendre-on-a-full-sphere-rule
(`[M]` `lebedev(11)` → a `(50,3)` table). ⟹ when a shipped predicate is *wider* than the
one the corpus derived, say so at the derivation's table too, or the page reads as if the
narrower one shipped.


---

## L-086 — the naming law (#432): a page-wide RENAME whose brief describes a design the review then replaced, and a caveat that became the theorem

**Task.** #429 tracker 1.9 / GitHub #432: `SubgroupOfO3.O2(axis)` — the pointwise
stabiliser of a coordinate axis — joins the lattice, and the axial orbit space is
RE-KEYED from the rotation half onto it (`S^2/SO2_x` → `S^2/O2_x`). Nine `docs/` pages,
`+1144/−355`. Baseline and final `-E -W` both **0** W/E/C, EXIT=0.

**(a) ⭐⭐ THE HEADLINE: a mid-task design delta made a whole subsection of my new prose
describe a REJECTED design — and the rejected design was the one I had MEASURED.**
I had written, with `[M]` markers, *"nine catalogue keys, six entries; the three `SO2_a`
keys are kept deliberately so the refusal carries the diagnosis"* — reproducible, exact,
and the elegance review had already replaced it: the refusal moves to
`Quotient.__post_init__` (a construction invariant on `by == by.orbit_stabiliser`, so
`dataclasses.replace` is refused too) plus the catalogue DOOR, and the three decoy keys
go away. ⟹ **when a brief says the change is "already in the working tree", ask whether
a REVIEW is still running on it** — a measured `[M]` is worthless if it measures a
transient. The repair that worked: keep the rejected design as a `.. note::` titled
*"A rejected first design, kept because it is the tempting one"*, with the four concrete
costs (validation inside a derivation · a second function-scope import that falsifies the
module's own "one runtime edge" paragraph · three keys deriving nothing · `replace()`
still accepted). ⭐ That note is now the strongest paragraph in the section, because it
is the only place the *reason* for the construction invariant is falsifiable.

**(b) ⭐⭐ A page's own ⚠ CAVEAT can be the THEOREM the next step is built on — read
every caveat as a candidate ruling before writing a new section.** `manifolds.rst`
already carried a measured note: *"the map's own stabiliser is BIGGER than H for the
axial family, so H-invariance cannot recover `by` — it is a declaration, not a computed
stabiliser"*, with `[M]` π_a invariant under σ_b for b≠a. Every measurement in it is
unchanged by #432; what moved is the CONCLUSION. Written as a caveat it says *`by` is
free*; read once more it is the argument for a rule — *if the map cannot tell two groups
apart, do not let the catalogue offer two names for one point set*. ⟹ the edit is a
`✅ And that is exactly why …` paragraph appended IN PLACE, plus a `⛔ this note ended
at "…genuinely indistinguishable" until <date>, and a reader who stopped there would
conclude `by` is free — it is not; it is DETERMINED, by the orbit partition plus
maximality.* Deleting the caveat would have destroyed the derivation of the rule.

**(c) ⭐⭐ Two brief numbers refuted, and the refutations were the better content.**
(i) *"+3 candidates cost ≤ 5 % per walk"* — `[M]` min over 15 interleaved repeats,
**11.3 – 26.2 %** (slab 5.0→5.6 ms, `product(4,8)` 107.5→135.6, `level_symmetric(4)`
320.8→367.1, `lebedev(11)` 283.8→329.9, `folded_product(4,8)` 103.5→115.2). (ii) The
brief's *"`O2(x) ⊇ D_1h = {e, σ_z}`"* — `[M]` `_group_elements(Dnh(1))` is **order 4**,
the Klein group `{e, σ_y, σ_z, C_2^x}`, which is *why* it sits in `O(2)_x` and in no
other axial stabiliser; the two-element reading would not have explained the axis
dependence. ⭐ And the number the brief did NOT have: the walk's answer **shrinks**,
`{SO2_x, σ_x, σ_y, σ_z}` → `{O2_x, σ_x}`, because σ_y, σ_z are ABSORBED by O(2)_x while
σ_x flips the axis and is absorbed by nothing. A simplification is a better headline
than a cost.

**(d) ⭐ Reproduce a "nothing moved" claim against a PINNED tree, not by reasoning.**
`AngularSymmetry.support` now calls `SPHERE.quotient(O2(a))`, so the pre-change arm
cannot be simulated in the live tree — `dataclasses.replace(sym, continuous_isotropy=
SO2(a))` RAISES. `git archive HEAD orpheus | tar -x` into a temp dir, run the same
24-row stage-0 probe in a subprocess with the editable finder stripped and
`orpheus.__file__` asserted: **24 of 24 identical**, and the pre-tree independently
confirms the old names (`S^2/SO2_x`, `L2[S^2/SO2_x]`, `quotient_group == SO2_x`) and the
old refusal (all three `S^2/sigma_y → S^2/SO2_a` arrows `None`). Same trick gives the
before/after column of every table for free.

**(e) ⭐⭐ A compatibility-law re-run needs its CONTROL in the same script.** The page
recorded *"15 groups × 6 fixtures, 0 violations over 342 (edge × fixture) pairs"* with
no definition of "edge". Reconstructing it (ordered pairs `B ≠ A` with `B.contains(A)`)
reproduced **57 edges / 342 pairs / 0** exactly — and only THAT made the widened reading
(18 groups → **75 edges / 450 pairs / 0**) a widening rather than a different instrument.
⟹ re-derive a recorded aggregate's DEFINITION by matching its old number before quoting
the new one.

**(f) ⭐ A "documents the DESTINATION" instruction leaves one unavoidable dead xref, and
that is the report's job.** `SubgroupOfO3.orbit_stabiliser` does not ship yet, so the
patched xref gate reads **1 dead / 2 sites** and nexus `dead_references` reads **1 of 53
checked** — the SAME single finding from two independently-vocabularied instruments,
which is the acceptance evidence that it is the only one. Stock gate: **0** (its
`head_role` blindness, L-062/L-082 — still unlanded). ⟹ when a pass is told to describe
an unlanded symbol, say in the report exactly which role goes dead and when it resolves.

**(g) Findings NOT in the brief, all present-tense-false on pages I was already in.**
FOUR `ERR-080 remains open` / `ERR-080 is OPEN` clauses in body prose (the machine
header's blanket *"every 'still OPEN' clause … was repealed"* does not reach a reader who
lands on the paragraph); `frame.rst`'s *"the third member … does not ship: ERR-080 stays
open"* with a `[M]` **five subclasses / two satisfy the surface** census that is now
`[M]` **6 / 3**; `cartesian_multid.rst`'s *"once #429 tracker 3.4 lands"* and
`operator_algebra.rst`'s *"tracker 3.4 will bind"*, both landed. ⟹ a rename sweep is a
free staleness audit of every page it opens, and the FUTURE-TENSE promise is the class
the symbol grep cannot find (L-066's third register, again).

**(h) ⚠ My own new tombstones re-introduced the L-085 nesting leak, five times.**
`*"… **bold** …"*` and `*"… ``literal`` …"*` are the natural §3 spelling and both leak.
⭐ My first source-side scanner MISSED three of them: `RX_I = \*((?:[^*])+?)\*` cannot
match an italic run that CONTAINS `**`. Widened to `\*(.{1,600}?)\*` with a
`(?<![*\w])` / `(?![*\w])` guard and set-differenced against `git show HEAD:<file>`, it
found all five. ⟹ the italic-run scanner must allow `*` inside the body, or it is blind
to exactly the case the rule exists for.

---

## L-087 — the Γ-slot (#429 tracker 2.2b): a mid-task DESIGN DELTA that landed WHILE I measured, and a "half closed" verdict the delta made whole

**Task.** Docs pass for #429 tracker 2.2b — the invariance question moves onto the
ORBIT SPACE, and the quadrature registry's stage 0 becomes a lattice relation.
Branch `fix/angular-phantom-support`, HEAD `4b7d24c3`, production carve UNCOMMITTED.
Pages: `manifolds.rst` (+1232), `discrete_measures.rst` (+112), `error_catalog.rst`
(+42), `frame.rst` (+18), + regenerated `matrix.rst`.

### 1. ⛔⛔ The tree was BEING EDITED while I probed, and a half-written module gives
### plausible-looking wrong answers

The coordinator's mid-task message said an elegance-review delta was "accepted;
landing in the next ~hour" and that measurements after ~19:45 would read the new
behaviour. `[M]` at 18:26 `manifold.py` already defined `ambient_representatives`
and `spent_group` while `symmetry.py` still CALLED `section_coordinates` — so
`ordinate_permutation` raised `AttributeError: 'Quotient' object has no attribute
'section_coordinates'` on every rule. I had ALREADY run a probe in that window and
recorded `TypeError`/`AttributeError` rows as if they were behaviour.

⟹ **When told a change is landing, do not poll on ONE symbol — poll on the
INVARIANT that the rename is complete tree-wide** (`git grep -q old_name --
orpheus/` returning empty), and take every "before" reading from the pinned
`git archive HEAD` copy, which cannot move. My first poller fired on
`def spent_group && def ambient_representatives` and was **premature**; the second,
on `! git grep section_coordinates orpheus/`, fired at 18:31:18 and was right.

⚠ And the aggravator: a half-landed tree does not raise ImportError. It raises
`AttributeError` deep inside a call, which reads as *"this rule does not support
that question"* — a plausible domain answer.

### 2. ⭐⭐ TWO of the seven delta items shipped DIFFERENTLY from the message, and
### the shipped forms are better — document the CODE, report the delta

The coordinator's list is a brief and obeys L-001: verify, then write.

| the message said | `[M]` what shipped |
|---|---|
| `is_normalised_by` **REFUSES** a translated motion | it takes `motion.linear_part` — a point group acts on DIRECTIONS and a translation does not move one, matching `ordinate_permutation`'s wrap convention. `[M]` a pure translation answers `True` for every family |
| `_ambient_orbit_space` becomes **`Ball(3).quotient(Trivial)`** | `[M]` `RealSpace(3).quotient(Trivial)`, name `'spatial_R3/Trivial'`; the docstring's reason is better than the brief's (*"a zero-padded interval or planar rule lands OFF the sphere, and the container must honestly contain what is put in it"*) |

Both were caught by running the probe and then READING the shipped body. Neither is
a defect; publishing the brief's version would have been.

### 3. ⭐⭐ A verdict I had already WRITTEN AND MEASURED — "II.11 is HALF closed" —
### became fully closed by the delta, and only a re-run found it

Pre-delta I measured `orbit_certificate(gauss_legendre_on_mu(8), σ_x) → None` and
published a table row reading *"⛔ II.11 SURVIVES here — a BARE support keeps the
1-D shape refusal"*, with a §-heading to match (*"the II.11 lead is HALF
closed"*). The delta routed a bare support through `RealSpace(3)/{e}` and DELETED
the shape test, so `[M]` post-delta that same call returns **2 permutations**. A
section title, a table row, a closing paragraph and the machine header all had to
flip from HALF to CLOSED.

⟹ **A "what survives" verdict is the most delta-fragile sentence a close-out can
carry**, because it is a claim about the ABSENCE of a repair, and a concurrent
review's whole job is to add repairs. Re-run every *"X still …"* clause after the
last code edit, not just the numbers. (L-080's zero-shelf-life rule, at section
scope rather than at gap scope.)

### 4. ⭐ What SURVIVED the delta unchanged — and re-measuring it is what proved the delta safe

`[M]` all against the pinned `4b7d24c3` copy, re-run after the landing:
`is_invariant` over `candidate_groups` — fold **4 of 21** flip (σ_y, C_2, D_1h,
D_2h, all False→True), `gauss_legendre(8)` **0 of 15**, `product(4,8)` **0 of 23**;
`walk(fold)` `{σ_x, σ_z}` → `{D_2h}` with slab/product unchanged and brute-force
agreement **6 of 6** both sides; the compatibility law **0 violations at 342 AND
450** (edge × fixture) pairs, both sides; `_embedded_nodes` ≡ `barycentre`
**12 of 12** both sides; stage-0 refusals **12 → 10 of 20** with **no** pair going
True→False; the `z`-marginal's and the chart rule's answers group-for-group
identical.

### 5. ⭐⭐ The brief's own design memo carried a claim that was false on BOTH trees

*"a fold by σ_x of the σ_y-fold works today and stays"* — `[M]`
`folded_product(4,8).measure.quotient(Mirror("x"))` raises `NotImplementedError`
(no catalogue entry for `S^2/sigma_y/sigma_x`) at HEAD **and** after. What 2.2b
changed on that verb is only the σ_y row, and it changed the REASON (from
`orbit_certificate`'s misleading *"this measure is not sigma_y-invariant"* — the
ambient reading — to the door's theorem), not the refusal.

### 6. ⭐ A design's ONE-EXPRESSION form is a publishable argument, and the naive
### two-conjunct form is the exhibit

My pre-delta stage 0 was *"the arrow exists AND (`X == D` OR `Γ ⊇ X.by`)"* — the
equality special case looked like a convenience. `[M]` it is load-bearing: without
it the predicate refuses the geometry's OWN domain, because a slab's rule lives on
`S^2/O(2)_x` and `σ_x ⊉ O(2)_x` (an infinite group cannot sit in a finite one).
`spent_group(D, X)` — `{e}` for the identity, `target.by` for a fold of the base,
`NotImplementedError` naming the missing work for the induced map — makes it ONE
expression with no special case. Publish the failed spelling as the reason the
shipped one is one expression.

### 7. ⭐ Two ERR-072 recurrences, both worth a catalogue note

(a) The right-angle sample OVER-CERTIFIES the new NORMALISER predicate too:
`[M]` over 8 (G, H) pairs with G continuous, `{0, π/2, π, 3π/2}` answers `True`
on **2** where the exact criterion says `False` — `(SO(2)_x, D_2h)` and
`(SO(2)_z, D_2h)`, because the quarter turns permute `{σ_y, σ_z}` back into
`D_2h`. The ten-angle incommensurate sample agrees **8 of 8** (the positive
control). Added to ERR-072 as a dated progress note.
(b) The kernel's position test (node on the axis / at the origin) would be a
**tautology** on an axial entry — the barycentre lift is on the axis by
construction — so the shipped kernel runs it ONLY for a finite quotienting group
and answers the continuous case by `G⁰ ⊆ H`. `[M]` step 2 does not subsume it:
on `S^2/O(2)_z`, `H ⊇ SO(2)_z = G⁰` while `H ⊉ D_∞h`, so `D_∞h` reaches step 3
and is admitted there. vv #19 at the kernel rather than at a gate.

### 8. ⭐ A RENAME's rationale can be this page's own subject

`section_coordinates` → `ambient_representatives`: a *section* is a choice of
representative, a point OF the base, and the axial arm returns the orbit's
BARYCENTRE, which is inside the ball. The old name promised a codomain it does not
land in — literally ERR-080's defect one level up, caught on the NAME alone before
the step landed. That paragraph is the best thing in the chapter.

### 9. ⚠ The test tree moved TWICE under me (L-085 again)

`matrix.rst` went 10831 → **10979** collected rows between my first and last build
(`numerics/test_manifold` 119→143, `test_symmetry` 133→230, `test_registry`
80→107) — the concurrent test-architect's gates. My own contribution is exactly
**+3 sentinels** (584 → 587), as predicted. The page's Verification section cited
*"70 test functions, 108 collected rows"*; `[M]` at my final build it is **91 /
143**, so the ladder gained a rung. ⟹ re-read every generated-artefact citation
after the LAST build, and treat the matrix as the second instrument for both.

### 10. Gates

`-E -W` baseline **0** WARNING/ERROR/CRITICAL/SyntaxWarning, EXIT=0 → final
**0**, EXIT=0 (set unchanged, measured as a set). Stock xref gate **0 dead /
14618 decidable**; the L-062-patched copy at `scratch/` depth 1, run as a
subprocess with an end-to-end positive control (a throwaway `docs/_22b_ctl.rst`
with 2 dead + 1 live role: stock **0**, patched **2 dead / 2 sites**), also
**0 dead**. nexus `dead_references` **0 dead / 52 checked**. My own import probe
over the four edited pages: **850** fully-qualified project roles, **0** dead,
positive control 2/2. Rendered slice of the new chapter: **0** visible backticks,
**0** leaked role openers. Source-side nested-literal-in-bold, set-differenced
against HEAD: **0 NEW** on all four pages. Audit: 917→**920** labels,
584→**587** documented, **0** violations.

## L-088 — R1 of #434: a carve whose whole claim is "no answer changed", and the FULL GRID as the only honest instrument

**The task.** `#434 R1` re-poses every question about a subgroup of `O(3)` as a
computation on its *realization* (identity component + coset representatives),
retiring two hand-written relation surfaces, thirteen per-family functions, a
group cache and a type alias. I owned the docs half: 3 hand-edited pages
(`manifolds.rst` +722/−65, `discrete_measures.rst` +75/−16,
`error_catalog.rst` +35/−4) plus the regenerated `matrix.rst`. `-E -W` 0 → 0,
EXIT=0; patched xref gate `DEAD TARGETS: 0` (stock agrees; positive control
reads 2); nexus `dead_references` 0 dead / 52 checked; rendered-leak
set-difference vs a pinned HEAD build: **0 added** on the two pages I wrote
prose into.

### (a) ⭐⭐ When a carve's ENTIRE claim is "no answer changed", the instrument is the FULL GRID — and the two zeros are the finding

A retirement of a hand table is exactly the kind that can silently move an
answer nobody looks at, because the answers anyone *thinks* to name are the
ones the table already got right. So the honest denominator is not the edges
the plan enumerates; it is every ordered pair. `[M]` against a pinned
pre-carve tree (`git archive HEAD`, editable finder stripped,
`orpheus.__file__` asserted per subprocess): `contains` **0 of 729**,
`normalises` **0 of 729** (27 × 27 spellings), `is_invariant` **0 of 270**
(10 shipped rules × 27 groups), the walk **0 of 10 rules**. Three readings
move and each was named in advance. ⟹ **publish the grid, and say in the page
why a grid rather than a list** — that sentence is what stops the next reader
re-deriving the coverage argument.

⭐ The companion move that made the widening credible: **reproduce the
recorded aggregate's OLD numbers before quoting a new one** (L-086's rule).
The page carried `57 edges / 342 pairs / 0` and `75 / 450 / 0` for the vv-#15
compatibility law; both reproduce EXACTLY on the pre-carve *and* the carved
tree, so my widened `175 edges / 1750 pairs / 0` reads as the same instrument
at a bigger denominator rather than a different one. Four readings, two trees,
one claim.

### (b) ⭐⭐ A concurrent carve RENAMED a method I had already published — the xref gate caught it, the build could not

Mid-task the coordinator landed further R1 work: `Realization.images` →
`Realization.generic_images`, `IdentityComponent.conjugated_by` retired, a
`__post_init__` added, `_tags_contain` / `_orbit_space_of` / `_IDENTITY_3` /
`_NODE_WINDOW_FACTOR` minted, and the kernel's `Trivial` fast path removed.
`sphinx -E -W` stayed at **0** through all of it — a `:meth:` at a renamed
member renders as plain text. The patched xref gate found it in one run.
⟹ **on a live branch, re-run the xref gate after EVERY build, not once at the
end**, and re-read the public surface (`dir(cls)`) rather than trusting the
module you read an hour ago. `[M]` every numeric claim I had published
re-measured identically on the new tree — it was only the NAME that moved,
which is precisely the class `-W` cannot see.

### (c) ⭐⭐ The test tree moved THREE times under me: 230 → 212 → 215 in one afternoon, and the fix is to publish a DIRECTION

A Verification section that quotes a per-module row count is a hostage on a
live branch. I wrote `212 + 2`, rebuilt, and the matrix read `215 + 2`. ⟹ when
a count is *being changed by the very carve you are documenting*, publish the
**direction and its mechanism** and point at the generated matrix for the
value: *"the row is FALLING, because the carve retires the gates that pinned
the per-family arms it dissolved"* is durable; `212` is stale on the next
build. ⭐ And a falling test count is the EXPECTED shape when a carve removes
spellings — say so, or a later reader reads it as lost coverage.

### (d) ⛔ My own tombstone leaked, and only a rendered SET-DIFFERENCE could tell it from 132 pre-existing ones

Following §3 I quoted the retired paragraph verbatim — and spelled it
`*"… (``symmetry._axial_contains``)"*`, an outer italic run around a literal.
That is L-085's exact rule, broken by its own author. The raw leak inventory
on these pages is **34 / 8 / 90**, all tombstones from earlier campaigns, so a
COUNT is useless; the acceptance criterion is the multiset difference against a
build of the pinned HEAD tree. `[M]` before the fix: manifolds ADDED 2; after:
**0 added, 0 removed**. ⚠ And read `added == removed` as a CONTEXT SHIFT, not
two events: `error_catalog` showed `+2 / −2`, the same pre-existing
``vv-principles``-inside-bold leak whose ±90-char window my inserted paragraph
had moved. ⟹ **build HEAD's docs once into a scratch tree** (`git archive HEAD
docs tools tests`, `PYTHONPATH=<tree>`); it is a few minutes and it converts an
unusable count into a decidable answer. ⚠ Do NOT substitute a source-side
italic regex: over a page of list-tables it reported 10 hits and **all 10 were
`* -` bullets**.

### (e) ⭐ Inherited counts: two of three did NOT reproduce, and both were in text I was about to publish

The plan's ledger gave `_contains` "109 lines, 31 dispatch sites" and "eleven
per-family helpers". `[M]` by AST: **109 lines** (reproduces) but **28**
dispatch sites inside `_contains` (24 `isinstance` + 4 `is _NamedSubgroup.X`),
and **thirteen** functions retired, not eleven. The ledger's 31 came from a
different predicate over a different scope. ⟹ a relayed count needs its
PREDICATE re-derived, not just its number re-run — and the module-wide pair I
measured instead (`86 → 31` sites under one stated predicate) is the more
useful number anyway.

### (f) ⭐ Two docstring claims to REPORT rather than repair, both quantifier defects

`orbit_stabiliser`'s docstring says *"Exactly two MEMBERS are not their own
stabiliser"*; `[M]` **4 of 26** distinct members move (`SO2_x/y/z`, `SO3`) —
it is two FAMILIES, and the axial family has three axes. The page carried the
same word and I corrected it there. And `_tags_contain`'s `[M] 419 per walk on
a slab rule` measures **420** on my instrument (217 repeats + 203 distinct,
stable cold and warm). Everything else verified: all 11 containment relations
the `contains` docstring lists "for the record", the `1.3e-15` orthogonality
defect (**1.332e-15**), the ERR-072 2-of-8 over-certification, the 8-of-8
incommensurate positive control.

### (g) ⭐ A memo/perf claim is publishable EXPERT context when you measure it yourself

The obvious objection to "compute, don't tabulate" is cost. `functools.cache`'s
own `cache_info` answers it in three lines: `[M]` one walk on `lebedev(9)`
asks **1152** containment questions of which 629 are literal repeats, and
builds **24** groups from 1193 reads. *"A hand table is not buying speed here;
it is buying a second, unverifiable copy of the answer"* is the paragraph that
pre-empts the objection — and it only exists because I ran the memo counters
instead of quoting the docstring's `41 times / 9.3 s`.

---

## L-089 — R4 of #434: the code moved UNDER me twice, and the second time it changed a claim I had already published

**Task.** The docs half of #434 R4 ("the lift is a derivation output, and an orbit
space's dimension is a theorem") — `orpheus/numerics/manifold.py` mostly, plus
`symmetry.py`, `basis/descent.py`, `quadrature/directional.py`, all UNCOMMITTED on
`fix/angular-phantom-support`. Deliverables: a census, a re-derivation on
`manifolds.rst`, ERR-080's wording, the build + `dead_references`, and a report.

**Output.** 4 pages, +1127 / −196 (1 of them the regenerated matrix). `sphinx -E -W -q`
**EXIT=0, log 0 bytes** (a completely clean build, so the count-unchanged gate is met
absolutely). `check_docstring_xrefs` DEAD 0; my own import probe with controls 429/0;
nexus `dead_references` 0 dead / 52 checked; V&V sentinels **591 → 593** (my two new
`documented` labels), `numerics/test_manifold` **143 → 240** (code-side), collected
**10964 → 11061**.

### The finding that dominates the session: a LIVE branch under two other agents

`git status` at dispatch showed 4 modified `orpheus/` files. By the time I probed the
code for the second time it showed **20**, including `tests/` (the R4 gates had landed)
and a second pass of `orpheus/numerics/manifold.py` from the elegance review. Three
concrete consequences, in the order I hit them:

1. **The brief's premise "tests are not yet written" was already false.** The R4 gate
   set — 9 `TestR4*` classes, 27 functions, `[M]` 102 collected rows — was in the tree.
   I only noticed because a census I ran to CHECK a plan claim (*"no test pins
   `sigma_y/Trivial`"*, `[M]` **2 hits**) returned hits in a test file that should not
   have had them. ⟹ *a census that disagrees with a brief is evidence about the TREE,
   not about the census.* The result: I could name the gates instead of writing
   "the gates #434 R4 lands", which is the difference between a coverage claim and a
   citation.
2. **The elegance review re-shaped `__post_init__` from THREE clauses to FOUR** (adding
   a lift-codomain **ambient-width** gate), moved `_generic_orbit_dimension(group, base)`
   onto `SubgroupOfO3.generic_orbit_dimension(points)`, and turned the single generic
   POINT into a probe **SET** with a MAXIMUM. I had already written and BUILT three
   clauses and a single point. Caught by re-reading the live `__post_init__` while
   hunting a *different* claim (the `repr` one below), not by any check I had planned.
3. **`lift_codomain` flipped from `field(compare=False)` to compared**, with a measured
   justification I could not have invented: `[M]` with it excluded,
   `replace(entry, lift_codomain=SPHERE)` compared **EQUAL** to the catalogue entry and
   `barycentre`'s `functools.cache` then answered for both — ERR-080's own shape
   re-minted by the field built to refuse it. My published sentence *"both are
   `field(compare=False)`"* was false in three places.

⟹ **the rule.** On a live branch, the re-read is not a pre-flight, it is a LOOP: re-read
the module's public surface (`dataclasses.fields`, `dir(cls)`, the `__post_init__` body)
**after every build**, and treat any sentence naming a field's `compare`/`repr`/default,
a guard's clause COUNT, or a helper's SIGNATURE as the highest-decay class. `-W` is
silent on all three — the build was EXIT=0 with the false compare claim in it.

### The instrument that found the one defect I introduced

The **rendered-HTML slice** (L-069/L-080), anchored on `role="main"`, with display math,
inline math, `<pre>` and `<code>` stripped, counting visible backticks and leaked
`:role:` spellings. It caught `*"a map :math:`M/H \to M`, for any entry"*` — a role
opening inside an emphasis run — rendering as **`M/H to M`, the LaTeX backslash eaten**,
on a **0-warning** build. My source-side `re.S` differential (L-076) said `new=0` and was
right about *nesting* and blind to this: the defect is not `**…``…``…**`, it is a role
that never parsed. ⟹ keep BOTH: the source diff is free and runs every edit; the render
slice is the only thing that sees a role that did not parse.

⭐ The per-SECTION slice is what made the page-wide counts usable: manifolds.html carries
74 visible backticks and 1 leaked `:mod:` role, and slicing to my eight `id=`s gave
**0 / 0 on all eight**, with both survivors proven pre-existing by `git show HEAD:`.
A page-wide count would have indicted the whole page for someone else's prose.

⚠ And a harness trap of my own: `nohup … &` inside a `run_in_background` Bash call
reports "completed, exit 0" for the *shell*, not for sphinx. I read a STALE `manifolds.html`
and concluded my fix had not taken. Run the verification build in the FOREGROUND with a
long timeout, or assert a distinctive new phrase is in the built HTML before believing it.

### Four claims in the module's own docstrings that are FALSE (reported, not edited)

1. `_coordinate_chart`: *"the columns are visible in the ``repr``"* — `[M]` both
   `orbit_coordinates` and `lift_coordinates` are `field(repr=False)`, and
   `repr(SPHERE.quotient(O2("x")))` contains neither `_ambient_columns` nor
   `_embed_columns` nor `functools.partial`. The picklability half is true.
2. `__post_init__` clause 4: *"both forgeries of clause 2 ship `fundamental_domain=None`
   and this clause returns early on them"* — `[M]` HALF false: the `S²/σ_x`-on-`[-1,1]`
   forgery carries `FundamentalDomain(SPHERE, ((1,0,0),), 'x>=0')`, `dim` 2, which
   against a 1-D realization violates clause 4 **as well**. The clause still needs its
   own input; the reason is ORDERING (clause 2 runs first), not absence.
3. The module docstring's read-set list omits `group.name`, which `[M]` is read **13**
   times — more than the eight members it does name put together. The previous version
   named it; the R4 rewrite dropped it while adding the four new ones.
4. `manifolds.rst`'s import table cited `manifold.py:92/:93/:1194` and
   `symmetry.py:102/:103`; `[M]` live they are `:96/:97/:1679` and `:105/:106`, and
   92/93/1194 were **already stale at HEAD**. The table's own preamble says the numbers
   are re-derived rather than carried — so this is the preamble's contract going unpaid.
   Its caption *"Every edge among …"* also over-promises by three (`manifold.py:1432`
   function-scope `→ symmetry`; `measure.py:116/:120` TYPE_CHECKING).

### Numbers I re-measured rather than relayed, and what changed

| the plan / brief said | `[M]` mine | verdict |
|---|---|---|
| `barycentre` 41 lines in manifolds.rst | **47 hits / 42 lines** | brief counted LINES; off by one |
| *"no test pins `sigma_y/Trivial`" (0 hits)* | **2 hits**, both docstrings of the new R4 gate | refuted as stated; the substance ("no ASSERTION pins it") survives |
| `max\|section − P_H\| = 9.943e-01 / 9.735e-01 / 9.778e-01` | `9.748e-01 / 9.932e-01 / 9.953e-01` | a DRAW; the two maps differ in column `a` alone, where the gap IS \|x_a\|, **supremum exactly 1** — published the bound |
| *"both folded call sites pass `axis="x"`"* | a runtime spy over the harness's 6 consumer modules: **62** calls, 4 `(support, axis)` cells, fold × **x** only, 327 passed | corroborated, and strictly stronger than a source read |
| harness residuals `1.15e+00 / 1.19e+00`, `31 of 33` | `1.155e+00 / 1.189e+00`, **31 of 33** by re-installing the pre-R4 pass-through in the same interpreter | reproduced exactly |
| min chart separation `1.155/4.403e-01/2.751e-01/1.510e-01` | identical | reproduced |
| `_embedded_nodes == barycentre` 12 of 12 | **12 of 12** | reproduced |
| the trapezoid ladder | `2.220e-16 / 3.331e-16 / 6.661e-16 / 1.554e-15 / 2.831e-14` at n = 8/16/32/64/1024 | reproduced; **more points is worse** |

⭐ **The independent reference I built and would build again**: `P_H` from the group's
**realized matrices** — an orthonormal basis of `⋂ ker X ∩ ⋂ ker(r−I)` by SVD, then
`B Bᵀ`. It reads no column index, so a swapped pair or an off-by-one scatter moves the
SUT and leaves it fixed; `array_equal` on 8 of 8 entries × 41 vectors, `max|Δ| = 0.0`.
Two more, one per family: the finite group's own MEAN (`array_equal`, exact — `(x+(−x))/2`
is `0.0` in IEEE-754) and the orbit-circle trapezoid. Three references, three structural
angles, no shared code above numpy.

### The doc SHAPE this event class wanted

A **branch-becomes-one-formula** carve (N per-family arms collapse into a single
derivation output). What worked:

1. **The general statement gets its own labelled equation and ONE home** — here
   `manifold-reynolds-projector` in the ARROWS chapter, where the map's codomain
   argument already lived — and every other site POINTS at it. The lift section then
   narrates the FIELD and cites the label; nothing is stated twice.
2. **Retitle the section that said "one per catalogued family"**, keep the anchor.
   The old title is the claim the carve refutes.
3. **The retired arm gets a `.. note::` with WHAT WAS LOST, stated precisely** — here:
   the section lands ON `S²` and the barycentre does not, so a consumer needing a
   *direction* no longer gets one from the lift; `[M]` none does; and the section's
   IMAGE survives in `fundamental_domain`, so what retired is the *map into it*.
4. **A rename with three generations gets ONE bullet list, not three paragraphs**
   (`section_coordinates` → `ambient_representatives` → `orbit_barycentres`), closing on
   the transferable rule: *a name that must be qualified per argument is a disjunction
   wearing a noun.*
5. **The Mode-12 blindness gets a LABELLED subsection**, because it is a design
   constraint on the gates rather than a caveat — with the three consequences numbered
   (no end-to-end catcher; assert at the ambient tier; the round trip is a declared-blind
   leg) and the discriminator's magnitude on both sides (`O(1)` ambient vs **exactly
   zero** through the chart).

### The self-check that paid, and its three standing false positives

One ~120-line Python pass, ~2 s, run to EXHAUSTION before the first build: label/anchor
uniqueness (exact-line), underline code-points + marker ladder, `list-table` column
consistency + `:widths:`, `:ref:`/`:eq:`/`:doc:` resolution corpus-wide, role import
resolution, source-side nested markup vs `HEAD`, trailing-space-before-closing-backtick,
and `_scan_theory_equations`. It has three KNOWN false positives on this corpus, and
recording them is what keeps the next run cheap: the `boltzmann` "duplicate" is a
`.. code-block:: rst` EXAMPLE in `harness.rst`; a "ragged" `list-table` row is a legal
EMPTY cell (`^     -$`, no trailing space); and a "dangling" `:doc:` is a RELATIVE
docname. Two builds total (plus one wasted on the `nohup` illusion).

---

## L-090 — R2 of #434: the kernel changes HOUSE, and the hardest half of the sweep is a `[M]`'s DENOMINATOR

**Task.** Docs pass for carve R2 of #434 — the invariance kernel leaves
`numerics/symmetry.py` for a new `numerics/invariance.py`, the five verbs move onto
`DiscreteMeasure`, `SubgroupOfO3.is_invariant` is DELETED with no façade, and the import
direction reverses (`manifold → symmetry` at module scope). 4 pages, **+839/−368**
(`manifolds` +481/−343, `discrete_measures` +285/−15, `error_catalog` +66/−4,
`structured_geometry` +1/−1); baseline and final `-E -W` both EXIT=0 with a **0-byte**
log; sentinels **593 → 593** (no eq-label added, as the verification plan predicted);
`dead_references` **2 dead / 3 sites, all in `tests/`**, 0 in `docs/`.

### 1. ⭐⭐ A `[M]` whose DENOMINATOR is a COMPUTED SET has a shelf life the FINDING does not

Two sites carried *"the reduction agrees on **150 of 150** (sphere rule × candidate
group) rows"*, measured 2026-09-02. `[M]` re-measured it is **144 of 144** — and the
FINDING (every row identical) is unchanged. The denominator is
`sum(len(candidate_groups(rule)))`, i.e. an output of the very machinery the campaign
keeps re-deriving; it moved twice in two days (R1's one-spelling merge, R2 reading the
azimuth count off the orbit barycentres). ⟹ when a `[M]`'s denominator is a *computed*
population rather than a fixed roster, **write what computes it**, and expect the number
to rot while the row does not. The repair that reads best is not a tombstone but a
sentence: *"the finding is unchanged and only the DENOMINATOR moved, because it is the
size of a candidate set and the candidate set has been re-derived twice since."*

### 2. ⭐⭐ The plan named TWO intended behaviour moves; the WHOLE roster had THREE

`§II.R2` listed *"gauss_legendre walks → {O2_x, D_2h}, the folded candidate set 20 → 18"*.
Running all 11 shipped rules against a pinned `git archive HEAD` tree: `[M]`
**`folded_product(4,6)`'s walk also moves**, `{D_1h, σ_x} → {D_2h}`, by the same
mechanism (its stored representatives have 3 azimuths, its barycentres 2, so `C_3`/`D_3h`
leave the candidate set and `C_2`/`D_2h` enter). It is a *strengthening* — `D_2h` contains
both answers it replaces — but it is a third row nobody had written down, and it was free
to find. (L-074's finite-roster rule, third instance.) ⭐ And the sharpest framing came
from separating the two questions: the invariance PREDICATE moved **0 of 330** (11 rules ×
a FIXED 30-spelling group list), while the WALK moved on **4 of 11** — because what
changed is *which questions the walk thinks to ask*, not what any answer is.

### 3. ⭐⭐ "ONE closure" is a CALL-SITE COUNT, and that is the only form of it worth publishing

Three docstrings had claimed one closure while two functions carried an identical inlined
lambda. R2's repair is checkable: `[M]` by AST, `_orbit_closure` has **one** call site
tree-wide (inside `_orbit_space_closure`), `_orbit_space_closure` has **three** (one per
verb that needs it), and `images_of` is now REQUIRED rather than defaulting to the ambient
action — *"a default nobody uses is a second code path that only a future caller can
discover"*. Publish the call graph; "these three cannot disagree" is then a statement
about the tree rather than about anyone's intention.

### 4. ⭐⭐ An architectural step can be RIGHT and have NO shipped discriminator — say so instead of crediting it

The kernel matches nodes in the orbit space's CHART coordinates through each motion's
`induced_action`, and it is tempting to read that as the step that fixed the fold. `[M]`
over **1027** (rule × group × element) rows where the element normalises the quotienting
group, an ambient nearest-neighbour match performed *on the barycentres* returns the
identical permutation on **1027 of 1027**. What moves an answer is reading the
**barycentres** instead of the stored representatives (`[M]` σ_y on `folded_product(4,8)`:
`None` against the stored representatives, the **identity permutation** against the
barycentres, chart or ambient alike). ⟹ a gate on the chart step has no shipped witness
and cannot have one until an entry ships whose Reynolds projector is not injective on
orbits. That is `vv-principles` #19 at the DESIGN tier: name the inert arm in the page,
with its denominator, so the next reader does not credit it.

### 5. ⛔ My AST import census had TWO silent filter defects, and the second hid the load-bearing edge

`plan-authoring` 2026-08-31 records that a filter on `node.module.startswith("orpheus")`
drops every RELATIVE import. I knew that and wrote it correctly. `[M]` the census still
reported **`measure → invariance` as ABSENT** — the single most load-bearing runtime edge
in the carve — because `from orpheus.numerics import invariance as _invariance` has
`node.module == "orpheus.numerics"`: it is an edge to the **SUBMODULE**, and a filter
comparing `node.module` against a module set cannot see it. ⟹ an import census needs
BOTH resolutions and a positive control per shape. Caught only because the answer
contradicted the carve's own comment.

### 6. ⭐ A rejected-design note can have ONE of its N costs EXPIRE — date the clause, keep the note

`manifolds.rst` carried *"a rejected first design, kept because it is the tempting one"*
listing FOUR costs, one of which was *"needs a second function-scope import
(`SubgroupOfO3`) so the module's own 'one runtime edge' paragraph stops being true"*. R2
reversed the import direction, so that cost is now zero. Deleting the clause destroys the
record; leaving it ships a false cost. The repair is a dated `⛔` naming the expired
clause AND saying the ruling does not depend on it — *"the three surviving costs are each
sufficient on their own"*. (L-074's half-falsified-refusal shape, applied to a cost list.)

### 7. ⭐ Reproducing the REFUTED variant on a renamed shadow package costs ~1 minute and IS the ruling's evidence

The plan's opener refuted R2-as-written (`symmetry` still reading the axis table from
`manifold`) with *"3 of 9 entry points clean"*. Reproduced independently — full copy of
`orpheus/` into `shadowpkg`, every `\borpheus\b` rewritten, one fresh interpreter per
(variant, entry point) — `[M]` **V0 10 of 10, V1 3 of 10**, and the seven that die name
`ImportError: cannot import name 'Quotient' from partially initialized module`. ⭐ The
publishable half is the three SURVIVORS: `import orpheus` alone is one of them, so a smoke
test on the package root reports GREEN on a façade that cannot serve one numerics entry
point. No production file touched, so the crash-unsafe-revert hazard cannot bite.

### 8. The section that was NAMED after the thing that no longer exists

`.. _manifold-import-cycle:` had six in-page citers and a title reading *"The module
imports nothing from ``numerics`` at MODULE scope — on purpose"* — present-tense-false the
moment R2 landed. Ruling: **keep the label** (a stale NAME is not a false CLAIM; all six
citers are intra-doc, so a rename would be caught by `-W`, but the section is still where
the cycle is documented), **retitle** the section (a bare `:ref:` renders the target's
TITLE, so every citer improves), and give the history its own `~` subsection so the
falsified claims stay verbatim under a `⛔`. Three of the six citers were themselves
present-tense-false and needed their own edits — a label's citers are a blast radius.

### 9. Where the module's own documentation went, and why not a new page

`invariance.py` is *a measure × a group*, and the corpus already had both homes: the
orbit-space kernel's MATHEMATICS on `manifolds.rst` (`manifold-one-invariance-kernel`,
inside the chapter that derives the normaliser criterion) and the MEASURE on
`discrete_measures.rst`. A new page would have split an argument that flows
normaliser → lift → induced action → kernel. Ruling: the kernel's three conjuncts stay on
`manifolds.rst` (re-derived); the module's own section — the BOUNDARY register: why the
verbs are the measure's, why no façade, the call-site proof of one closure, the
Reynolds-projector argument for the barycentres, the two gotchas — is a new `-` section on
`discrete_measures.rst`, five `~` subsections, plus one Key Facts bullet. `[M]` no
`automodule` exists for any of `numerics.{symmetry,manifold,measure}`, so `invariance`
gets none either and every `:mod:` renders plain text by page convention.

---

## L-091 — R3 of #434: a SLOT SPLITS IN TWO, and the retired name is the least of it

**Task.** Docs pass for the last carve of #434. `AngularSymmetry(continuous_isotropy,
discrete_residual)` becomes `AngularSymmetry(spent, unspent, owed)`; stage 0 becomes
"the descent arrow exists ∧ `H ⊆ Γ·K`"; `manifold.spent_group` retires. Deliverables
V3 + N6 of `scratch/_r3_elegance_findings.md`. Pages: `manifolds.rst`,
`discrete_measures.rst`, `error_catalog.rst`, `frame.rst`. `[M]` +1115/−325 over 4
pages; `-E -W` EXIT=0 with a **0-byte** log both sides; sentinels **593 → 593**;
`dead_references` 0 dead / 52; xref gate `DEAD TARGETS: 0`; my own import probe over
**923** fully-qualified `orpheus.*` roles on the four pages, 0 dead.

### (a) ⭐⭐ A FIELD SPLIT is not a rename — one of the two survivors keeps the retired
### name's LETTER, and the corpus's Γ now means the opposite half

The brief called this out (V3 item 4) and it is worth generalising. A rename gives one
old name and one new name; a SPLIT gives one old name and two new ones, and the corpus
symbol usually follows the *wrong* survivor. Here `discrete_residual` (Γ, the owed
closure) split into `unspent` (the NEW fold licence) and `owed` — and R3 gave Γ to
`unspent`. So every pre-R3 sentence saying "Γ" is not merely stale, it is **inverted**.

⟹ The instrument that worked: a **SYMBOLS block** as a labelled `.. admonition::`
(`quadrature-symbols`) with a `Symbol | Field | Meaning | Was` table — the fourth
column is what makes it a tombstone rather than a glossary — plus an explicit
grep-able discriminator sentence: *"a page that pairs Γ with G⁰, or calls Γ a
residual, predates 2026-09-03"*. The elegance review had asked for the symbols block
as an architectural opportunity; it turned out to be the load-bearing artefact of the
whole pass, because it is the only place the two bindings sit side by side.

⭐ And the ANCHOR: `manifold-gamma-slot` (7 citers on 3 pages) now names a letter it no
longer holds. Retitled to the concept (`The registry's ledger — …`), anchor KEPT, with
a head `.. note::` saying the name is a fossil and why (a cross-doc `:ref:` that dangles
is silent at every severity). `[M]` all 7 citers render the new title automatically.

### (b) ⭐⭐ A `[M]` COUNT whose member set is not stated is NOT reproducible — and the
### spread across plausible member sets is the finding

The production docstring carried *"197 such triples over the expressible members"*
(triples where `H ⊆ ΓK` while neither factor contains `H`). I could not reproduce it.
`[M]` on a natural 21-member set (`Trivial, SO(3), O(3), O_h, I_h, D_∞h, σ_{x,y,z},
SO(2)_{x,y,z}, O(2)_{x,y,z}, C_{2,3,4}, D_{1h,2h,3h}`; 12 finite ⟹ 21·12·21 = **5292**
triples, the review's own denominator) the answer is **217**; swapping `D_3h → C_6`
gives **181** and `C_3 → D_4h` gives **255**. The denominator matched and the numerator
did not, which is exactly the signature of an unstated member set.
⟹ published MY count WITH the member list enumerated in the prose, plus a ⚠ saying the
count is a property of the member set and **the witness is what to quote**
(`O(2)_x ⊆ O_h·SO(2)_x` while neither factor contains it — that one is a theorem-shaped
fact, reproducible forever). The sibling figure in the same docstring, *"441 of 441
ordered pairs over 21 members"*, reproduced EXACTLY on the same set — so one number in
one sentence was reproducible and its neighbour was not.

### (c) ⭐⭐ A "shipped" denominator can contain a CONSTRUCTED member, and only
### enumerating the set finds it

`domain_refusal`'s docstring says *"over 4 geometries × 7 shipped rules the 17 stage-0
refusals split 14 arrow / 3 coverage / 0 both"*. The split reproduces to the row — but
only for ONE 7-rule set, and its seventh member is `product(4,8).quotient(Mirror("z"))`,
which no factory ships. `[M]` the five `Quadrature` factories alone give 9 admitted / 8
arrow / 3 coverage of 20; adding `gauss_legendre_on_mu` gives 12 arrow; only the
constructed σ_z fold reaches 14/3/0. It has to be there — it is the only input that
separates the cylinder's Γ from the plane's — so the set is RIGHT and the word
"shipped" is wrong. ⟹ when a docstring hands you an `n`-row denominator, **enumerate
candidate sets until one reproduces**, then publish the enumeration; the search itself
is the evidence.

### (d) ⭐⭐ A test docstring's "this leg is INERT at the selector tier" is a
### NEGATIVE claim about production, and the in-process neutering measures it in 20 s

The R3 gate class carried an honest-scope note: *"at `select_quadrature` the coverage
leg is INERT: nothing registered is a fold, and the 1-D rule is refused for the cylinder
by stage 2's V conjunct first"*. `[M]` both clauses are false. `GaussLegendre1D` IS
registered and its support IS a fold (`S^2/O(2)_x`, `H = O(2)_x`), and the shipped log
for `select_quadrature("cylinder", 5)` reads `domain mismatch: … a fold by O2_x …
(unspent D_1h, spent Trivial)` — stage 0's coverage clause, not stage 2. Measured by
monkeypatching `SubgroupOfO3.is_subset_of_product` to return `True` in-process
(restored in a `finally`, verified by identity): the rejection MOVES to stage 2's V
conjunct and the CHOSEN rule is unchanged (`LebedevSphere(order=5)` either way).
⟹ the honest statement is *"the leg changes the REASON at the selector tier and no
selection"*, which is what shipped in the page — and it is a stronger scope note than
the one it replaces, because it names what a mutation would and would not move.
⚠ Report the test docstring; do not edit `tests/`.

### (e) ⭐ Re-measure a SIGN-COUNT before publishing it — `sign(0)` is a third class

I inherited *"2 of 4 (sign μ_x, sign μ_y) sweep quadrants empty"* for the σ_y fold and
wrote a comparison sentence for the unfolded rule from the same probe's summary line:
*"all 8 octants populated, 4 nodes each"*. `[M]` false. `product(4,8)` has 32 nodes of
which **16 lie ON a coordinate plane** (`μ_x` or `μ_y` exactly 0); the four
strictly-signed quadrants carry 4 each and the eight strictly-signed octants carry **2**
each. The fold's own figure is right (its 16 nodes are all strictly signed,
`μ_y ∈ [+0.1945, +0.8688]`). ⟹ when a claim counts SIGN CLASSES, `sign(x) == 0` is a
class of its own and a symmetric product rule puts half its nodes there; say
"strictly-signed" and give the on-plane count.

### (f) ⭐ The equality SHORT CIRCUIT is the page's own argument, and the carve turns it
### into history WITHOUT weakening it

`manifolds.rst` carried a ⭐ paragraph arguing that stage 0 must be ONE expression
because the naive two-conjunct spelling needs a special case for equality — with the
measured reason (`σ_x ⊉ O(2)_x`, an infinite group cannot sit in a finite one). R3's
coverage test needs no such case: `[M]` the slab's own rule reads
`O(2)_x ⊆ {e}·O(2)_x`, TRUE. ⟹ do not delete the argument — it is the *derivation of
the requirement the new predicate satisfies*. Re-home it into a dated ⛔ admonition
("The second conjunct read something else until …"), numbered 1. and 2., where 2. IS
the short-circuit argument now reading as the reason the coverage form is better. The
falsified design and its replacement's justification are the same sentence.

### (g) ⭐ Deliverable shape for this class: manifolds owns the LATTICE, the selection
### page owns the LEDGER

Two pages both wanted the material. Split by register: `manifolds.rst` (the point-set /
group layer) got the coverage THEOREM (two-step derivation off
`:eq:`manifold-group-as-component-and-cosets``), a new row in its own *One body per
question* table, and the measured 28-row grid; `discrete_measures.rst` (the selection
algorithm) got the three-fact ledger, the SYMBOLS block, the per-geometry derivation of
Γ with the cosine conventions, and the worked examples. Each cites the other once. `[M]`
zero new eq-labels on either page — both existing labels re-worded and kept (they are
`:eq:` APIs with 4 citers between them), so sentinels moved 593 → 593.

### (h) ⛔⛔ An ERR entry with no marker REDDENS A GATE — a docs-only pass can break the
### suite, and only running the gate finds it

ERR-081 shipped with its catching class named in prose. Two consequences, and I found
the second only by running the gate: (1) the build regenerated
`.claude/skills/vv-principles/error_index.md` from **80 entries · 0 uncaught** to
**81 entries · 1 uncaught** with a new "⛔ Uncaught" heading (correct, generated, never
hand-edit); and (2) `[M]`
`tests/test_error_catalogue_reconciles.py::test_every_declared_entry_has_a_catching_test`
goes **RED** — *"1 catalogued defect(s) have no `@pytest.mark.catches`: ['ERR-081']"*.
Its docstring offers *"or say in the entry why no test can exist"*, but the assertion
is `_catalogue_ids() - _marker_ids()` and parses no exemption, so the marker is
MANDATORY. ⟹ after minting any `.. error-entry::`, run that module (0.2 s, no venv
beyond pytest) and report the `@pytest.mark.catches("ERR-NNN")` as a BLOCKING companion
edit. A docs-only pass with an "I do not edit `tests/`" constraint can still put the
tree red, and the `-W` build is silent about it.

---

## L-092 — #428: splitting a "Reactions Not Included" section when one of the three named channels IS handled

**Task.** `docs/theory/foundations/cross_section_data.rst` §"Reactions Not Included:
(n,2n), (n,3n), (n,4n)" (:693-806 at HEAD `8707c53a`) was present-tense-FALSE for
MT=16: the channel is extracted, carried by six solver families, and in the k balance
of all of them. A `qa` census (`scratch/_428_four_solver_check.md`) supplied a
per-solver fact table + a 16-row cross-family k table. Deliverable: split the section,
fix four stale `file:line` refs, correct the `solve_sn_adjoint` docstring's
`A_loss = L+C-S-B`. Result: 3 files, +478/−94, `-E -W` EXIT=0 with a **0-byte log both
sides**, `dead_references` 0 dead / 52 checked, my own xref gate 159 roles / 0 dead.

### (a) ⭐⭐ A SECTION HEADER can be a class-level falsehood, and the page usually already
### carries the true account — the repair is a SPLIT BY FACT, never a re-word

The header named three MTs and asserted one predicate ("not extracted") over all
three. `[M]` it is true of MT=17/37 and false of MT=16 — and the SAME page carried the
correct MT=16 account **~200 lines above** (the extracted-MT list-table, the P0
truncation warning, the `mf6-yield-convention` section, the pre-#427 defect record).
So a reader could cite one file for either version (vv #21's aggravator at page scale).

⟹ the shape that worked: **two H2 sections, `n2n-handled` and `n2n-excluded-channels`,
each opening by naming what the OTHER one covers.** The handled one carries the datum
facts + the ruling + the per-solver table + the evidence; the excluded one keeps the
threshold rationale, the regimes and the deferred sketch, re-scoped. Every sentence the
census marked MT=17/37-only survived verbatim or with its quantifier narrowed; nothing
was deleted.

⚠ And the two steps that made the split *usable* rather than merely correct:
- **the sketch's steps 4–5 had to be RE-AIMED, not dropped.** Step 4 read *"every
  transport solver must account for the multiplicity"* — a to-do that is DONE. Rewritten
  as *"this step is not the open item it used to be … MT=17/37 would reuse that
  machinery with ν = 3 and ν = 4 rather than introduce it"*, it becomes the strongest
  argument in the section: the precedent exists and is measured.
- **the `#63` line said "tracked in", and #63 is CLOSED** — titled *"Data: Document
  (n,3n) and (n,4n) exclusion rationale"*, i.e. the issue's own title already scoped
  itself to the split being proposed. Re-framed as *the record of the decision, not an
  open work item* (plan-authoring §9(b)).

### (b) ⭐⭐ REPRODUCE the census's REFERENCE, and expect its per-row RESIDUALS not to
### reproduce — publish the reference + your own rows + the relayed figure as a BOUND

The census's 16-row table gave `k` to 16 digits per family. `[M]` mine:
- the **closed-form reference reproduces BIT-IDENTICALLY** — `1.6532258064516119`
  (Σ₂ on) and `1.2896126760563373` (off, +28.20 %) — as do all three datum-layer
  identities (`balance_residual == [0. 0.]`, `emission_matrix() == 2Σ₂ᵀ`,
  `absorption_xs = [0.072 0.165]`);
- the **homogeneous solver** is bit-identical to the reference;
- **diffusion / SN-fwd-SI / SN-adjoint did NOT reproduce the memo's residuals**
  (mine `2.7e-16` / `3.0e-10` / `1.6e-11` vs the memo's `8.1e-16` / `5.2e-14` /
  `1.0e-13`) — because a residual is a property of the MESH, QUADRATURE and
  TOLERANCES, and the memo stated none of them per row.

⟹ **a per-solver agreement residual is a run property, not a channel property.** The
publishable shape is: the reference (bit-identical, with its full input matrices so the
page regenerates it), YOUR rows with their configuration written into the table's own
first column (*"diffusion, 10-cell reflective slab, width 10"*), and the relayed sweep
as a BOUND with a `.. note::` saying the digits move. Never a 16-digit table whose
fixture you cannot state (L-057, L-050).

⭐ The one relayed number worth publishing verbatim is the **stochastic** one, because
its σ makes it self-describing: MC `1.655710 ± 0.001525` is **1.63 σ** from the closed
form, control `0.31 σ` — i.e. UNBIASED, which is the claim that had to be made because
ERR-023's *title* reads present-tense.

### (c) ⭐ A catalogue TITLE is a defect name, not a state — say so where a reader will
### quote it

ERR-023 is titled *"MC solver silently ignores Sig2 (n,2n) reactions"*. #428's own body
quoted that title as evidence MC might still be broken. `[M]` the defect was fixed at
#23, the catcher (`tests/mc/test_gaps.py::test_mc_n2n_keff_matches_analytical`) still
has teeth — and it is `@pytest.mark.slow`, so the canonical `-m "not slow"` gate never
runs it (#405). Shipped as a `.. warning::` naming all three facts.

### (d) ⭐⭐ A "Limitations and Future Work" table is a PRESENT-TENSE claim surface, and
### the fixing page is the likeliest place to find the stale row

`docs/theory/methods/monte_carlo.rst:1407` listed *"Solver ignores Sig2 (n,2n)
reactions"* under Limitations — while the SAME page documents the fix at `:755-764`
(the weight-doubling convention, ERR-023, the catcher by name). 650 lines apart, one
page, two tenses. Not in my brief; found by grepping the corpus for the CLAIM
(`does not extract|1-in-1-out|ignores.*sig2|silently ignor`) rather than for the
section I was editing. ⟹ **repair by retiring the ROW, not deleting it** — keep the
tracking ID so it still resolves, mark it `⛔ RESOLVED (#23)`, and point at the record
and the new section.

### (e) ⭐⭐ A pre-existing `**``literal``**` NESTING BUG travels forward when you rewrite
### a section, and only the RENDERED HTML sees it

The old sketch spelled its file names `**``orpheus/data/micro_xs/gendf.py``**` — an
inline literal inside strong emphasis, which RST cannot nest: the backticks render
LITERALLY. I carried all three forward verbatim. `-W` is silent; the source grep
`grep -n '\*\*``'` finds them, and so does L-074's HTML gate (`re.finditer(r"\`{2,}")`
over tag-stripped, unescaped HTML → **6 visible runs**, 3 sites × 2). ⟹ run the HTML
gate on every page you touch, and grep `\*\*``` before shipping a rewritten section.

### (f) ⭐ A `:ref:` to a label sitting on a `.. warning::` (not a section title) MUST
### carry explicit text — bare `:ref:` is a `-W` FAILURE, twice in one session

`WARNING: Failed to create a cross reference. A title or caption not found: 'X'
[ref.ref]`. A directive has no title to borrow. Spell it
`:ref:`the truncation warning <sn-n2n-p0-truncation>``. Both of my new anchors on
warnings (`n2n-p0-truncation-at-ingest`, and the existing `sn-n2n-p0-truncation`) hit
this. Note this IS caught by `-W` — unlike a dead code-xref — so it is cheap.

### (g) ⭐⭐ CITING a nuclide in a new section obliges you to check the page's own ROSTER

My split names Be-9 as a shipped MT=16 carrier. `[M]` the page's own nuclide table
listed **12** and Be-9 was absent — while `convert_gxs_to_hdf5.py` globs `*.GXS` (13
files) and the page's OWN P0-truncation warning already says *"the 13 shipped GENDF
files"* and cites Be-9 heavily. The tape landed `ea06fbbd` (2026-08-31) with no roster
update. Fixed on internal-consistency grounds (12 → 13 + the Be-9 row, temps/σ₀
`[M]` from `BE009.GXS`'s own MF=1/MT=451 header: 4 temps, 6 σ₀).
⚠ The near-miss: `infinite_medium.rst:1863` also says *"12 isotopes"* — and it is
CORRECT, because it counts a PWR-cell mixture, not the library. **Read what a number
counts before "fixing" it.**

### (h) ⭐ A MEASURED tape census beats a relayed one, and the MEMBER SET is the payload

`[M]` mine, by ENDF-6 column layout (cols 71-72 = MF, 73-75 = MT) over the 13 `.GXS`:
MT=16 **11 of 13** (BE009, B_011, NA023, O_016, U_235, U_238, ZR090…ZR096; absent
B_010, H_001), MT=17 **6 of 13** (U_235, U_238, ZR091, ZR092, ZR094, ZR096), MT=37
**2 of 13** (U_235, U_238), MT=2 13 of 13. Reproduces the census member-for-member.
The member set is what refutes the old prose's *"heavy isotopes (U-235, U-238,
Pu-239, …)"*: there is **no Pu on the tapes**, and MT=17 is carried by four
**zirconiums** (L-091's enumerate-the-members rule, paying again).

### (i) ⭐ The DOCSTRING fix that motivated the task had a SECOND false clause one
### paragraph down

Briefed: `solve_sn_adjoint`'s `(A_loss = L+C-S-B)` at `:2983`. `[M]` the same docstring
ALSO spells the daggered triple `((L+C).H, (S+B).H, F.H)` and the loss dagger
`(L{+}C).H - (S{+}B).H` at `:2987`/`:2989` — the same falsehood, twice more, in the
sentence the equation is explaining. Both fixed to the canonical
`A = L + C − S − N₂ₙ − B` that `n2n.py:17` and `coupled_system.py:104` carry, plus one
new clause naming the daggered emission `(ν₂ₙΣ₂ₙᵀ)ᵀ = ν₂ₙΣ₂ₙ` (verified against
`n2n.py:336-356`'s own transpose docstring). ⚠ `#425` tracks **37** `L+C-S-B` sites in
the SN chapter and `sn/index.rst:34` already carries a machine-header key explaining
the pedagogical spelling — so the two other solver.py hits (`:267`
`_evaluate_arm_residual`, `:657` `_exit_gauge_trace`) are #425's, NOT adjoint
docstrings, and were correctly left.

### (j) ⭐ A verbatim-quoted RULING elsewhere in the corpus constrains your PARAPHRASE

`adjoint.rst:706` quotes the CS4c §14.1 ruling verbatim, including its *"in principle
carries its own anisotropy"* hedge AND a dated note saying the hedge is now a
MEASUREMENT. My paraphrase had to (1) say it is a paraphrase and point at the verbatim
copy, and (2) carry the strengthening — otherwise the data page would re-import a hedge
the corpus has already retired. ⟹ **before paraphrasing a ruling, find where it is
quoted verbatim and read what the corpus has since said ABOUT it** (L-081's
quotation rule, in the other direction).

## L-093 — #426 step 1: a truncation that MOVES tiers, and the artefact-shaped staleness no symbol grep sees

Branch `fix/n2n-anisotropy`, 2026-09-03. Step 1 of #426 made the data layer lossless in ℓ
(`Isotope.sig2` / `Mixture.Sig2` → `list[csr_matrix]`; every scattering channel keeps the
tape's 7 orders where a hard-coded `range(3)` had cut P3..P6; HDF5 store format 2). My pass:
the sentences step 1 makes present-tense-false, plus the measurement into the corpus.
4 `.rst` + 4 `orpheus/` docstrings; `-E -W` EXIT=0 with a **0-byte log both sides**;
`dead_references` 0/52; stock xref gate 0/14821; my own import probe 258 roles / 0 real dead;
sentinels **593 → 594** (exactly my one new label).

### (a) The load-bearing shape: a truncation that MOVES needs a TIER TABLE, not a tense flip

The natural reading of "step 1 fixes #426" is *the P0 claim is now false*. It is not. Two
sentences that had always travelled together came apart:

- *"ORPHEUS models (n,2n) emission as isotropic"* — **still TRUE**;
- *"…because the data layer truncates at P0, unrecoverably"* — **now FALSE**.

So every site needed sorting by TIER, not by tense, and the residue is only adjudicable once
you know where the model now lives. `[M]` by AST over `orpheus/` (subscript-0 reads of a
`Sig2`/`sig2` attribute or name, docstrings excluded): **9** sites — **2** are the model
(`N2NKernel.from_mixture`, `MaterialXSField._build_dense_caches`) and **7** are ℓ=0 *by
physics* (a reaction rate IS the P0 row sum; CP/MoC/MC sources are isotropic by construction,
each already carrying an inline comment saying so). ⭐ And the census's own blind spot is the
finding's third site: `N2NOperator`'s `HarmonicFrame.for_space(interior, 0)` is **not a
`Sig2[0]` read at all**, so a predicate over the subscript structurally cannot return it. Say
that in the page — a reader who greps `Sig2[0]` to find "the truncation" will find 9 sites,
2 of them right, and miss the frame.

⟹ the publishable output is a two-column **tier table** (data ✅ lossless / operator ⚠ still
P0) plus the explicit warning that the 7 by-physics reads must **not** be "fixed" with it.

### (b) A regenerated LOCAL CACHE is a documentation surface, and nothing greps it

The page's **File Sizes** table (H-1 12.3 MB, U-235 50.0, U-238 37.8, O-16 10.8, Zr ~11) and
its *"processes all **12** `.GXS` files"* were both falsified by step 1, and **neither
contains a symbol**. `[M]` on the regenerated store: **13** tapes, U-235 **99.0**, U-238
**80.3**, H-1 **29.3**, O-16 **25.1**, Zr 20.0–25.8, **total 438.5 MB**, growth ×1.98–2.38,
7–8 min to rebuild. The `.h5` store is untracked and gitignored — so it is invisible to
`git status`, to `-W`, to the xref gate and to `dead_references`, and the only instrument is
`ls -l` after the regeneration finishes.
⟹ **after any change to a serialization format, re-measure the artefact table by listing the
artefacts**, and treat a file-count in prose as a claim about a glob (`ls *.GXS | wc -l`).

### (c) A format VERSION catches a LAYOUT change and is structurally blind to a VALUES change

Step 1 added `H5_FORMAT = 2` and a loud loader refusal, which invites *"the stale-store
warning can go now."* It cannot. `H5_FORMAT` is a hand-set constant: a change that moves only
the **values** (which is exactly what #427's yield fix was) leaves the layout identical, so
nothing can distinguish the old numbers from the new. ⟹ the two kinds of stale store need
**opposite prose** in the same paragraph — refused loudly / still silent — and the second is
the reason the old warning survives.

### (d) A relayed physical EXPLANATION can fail while its measurement stands

The brief's *"99.9 % is the reflector's — U-235's MT=16 is 13× weaker"*: the effect
reproduced exactly, the gloss did not. `[M]` U-235's peak (n,2n) reaction XS is **larger**
than Be-9's (0.813 b vs 0.559 b, ratio **0.69**) — the 13× must be a density- or
spectrum-weighted quantity nobody stated. Replaced with what I could measure: the
reflector-only control reproduces the full arm to **0.20·10⁻⁵** (`−412.05` vs `−412.25`,
same ℓ≤2 arm), and Be-9's MT=16 is open over **50** incident groups against U-235's **22**.
⚠ I also caught myself comparing the control against the **wrong arm** (`−413.55`, the ℓ=1
row) — a 1.50 discrepancy that would have read as agreement inside a "within 2·10⁻⁵" claim.
⟹ **a control and its subject must be the same arm**; state the arm in the sentence.

### (e) "Same nnz across ℓ" is an ISOTOPE property, and one isotope taught me the wrong universal

I wrote *"every higher order has the same 6067 non-zeros, since the yield strip is a row
diagonal and cannot change a sparsity pattern"*. The mechanism clause is right and the
conclusion is **false**: `[M]` U-235 294 K reads **6067, 6067, 5834, 5334, 3165, 2773, 1887**
for ℓ=0..6, while Be-9's (n,2n) stack is **8195 at all seven**. Sparsity is a property of the
**tape** (the evaluation stores genuine exact zeros in the higher moments), not of the ingest.
⟹ a gate assuming a shared sparsity pattern across ℓ is right on Be-9 and wrong on U-235;
publish the ladder, not the universal. (vv #13's finite-roster corollary, on the isotope axis.)

### (f) Two roundings of ONE source number read as a discrepancy between two surfaces

The plan said the thermal fixture's Δρ is `−51.1`; the mid-task brief said `−51.2`. Neither
is wrong — the source table says **−51.15**. ⟹ when two trusted surfaces disagree in the last
digit, go to the artefact and publish **its** precision; do not adjudicate between roundings.
⭐ And the free control while you are there: I re-derived all **24** derived columns
(Δk / Δk/k₀ / Δρ) from the recorded k values and they reproduce the source table exactly —
which is what licenses publishing the table as the corpus SSOT.

### (g) ⛔ The tree moved under me INSIDE a paragraph I had already built clean

An elegance pass landed on `hdf5_io.py` mid-task: `_REGENERATE` → `_REGENERATE_HINT`, and the
loader's two inline order-count expressions were single-sourced into `_n_orders` / `_order_key`.
My HDF5 section had already published the retired spelling verbatim —
`` `max(int(k[1:]) for k in sig2_grp) + 1` `` — on a **0-warning EXIT=0** build. L-089 again,
with a sharpening worth carrying: **a paragraph that QUOTES a code EXPRESSION is a higher-decay
class than one that names a HELPER**, because a refactor that preserves behaviour re-spells the
expression and leaves the name. ⟹ prefer *"one helper, `_n_orders`, serves both stacks"* over
transcribing its body; the sentence then survives the elegance pass that is coming for it.

### (h) A doc `.. code-block::` is the one staleness you can PROVE, so prove it

`docs/theory/methods/sn/index.rst` built a `Mixture(..., Sig2=csr_matrix(np.zeros((2,2))), ...)`
and ran `solve_sn` on it. Post-retype that is not merely stale — executed, it raises
`ValueError: the (n,2n) matrix is a square (ng, ng) group-transfer matrix; got shape (1, 2)`.
Running the old spelling and the new one costs four lines and converts "this looks stale" into
a demonstrated Cardinal-Rule-1 bug. (L-077's highest-severity-staleness rule, with the receipt.)

### (i) My own role probe's single DEAD was L-053(c), not a defect

258 `orpheus.*` roles on the touched files, 1 flagged: `SNMesh.axes`. `[M]` `hasattr(SNMesh,
'axes')` is **False** and `SNMesh` is **not a dataclass**, so both my class-level probe and its
`dataclasses.fields` fallback (L-076) miss it — `axes` is assigned in `__init__`. Constructing
an `SNMesh` gives `hasattr(sn, 'axes') → True, tuple, len 1`. Pre-existing, not mine.
⟹ a role probe needs a **third** fallback after `hasattr` and `fields`: construct the object.

### (j) Residual-sweep hygiene that worked

Every remaining hit of the retired spellings (`sig2_data[(0, 0)]`, "ONE matrix", "keeps only
ℓ=0", "unrecoverable downstream") is now inside a **dated ⛔ tombstone quoting it as history**
— 0 present-tense survivors, and the sweep's own patterns are the positive controls (each was
written from the pre-edit strings, so a pattern that finds nothing would be indicting itself).
⚠ Two of my tombstones were first written as `*"…"*` wrapping a `**bold**` / ``` ``literal`` ```
— L-085's leak pattern, authored by the agent who records it. Rewritten to plain quotes +
`, verbatim,`; the rendered-HTML slice then read **0 backticks / 0 leaked roles** on all six
new anchors (slices 752–12542 chars, each asserted to contain a known phrase).

---

## L-094 — #426 step 2: a truncation DIES, and the corpus pass is a BUILD repair first

**Task.** #426 step 3, the corpus pass for the landed carve `1a3b78ec`
("one transfer family — the (n,2n) gain is anisotropic"). Brief: 4 named page groups, a
retired-spelling sweep, a new ERR entry, and the SN changelog. Scope: `docs/**/*.rst` only
(a mutation battery was running in-process against `orpheus/`/`tests/`).

### (a) ⛔ The `-E -W` BASELINE WAS ALREADY RED, and every warning was the carve's own

`[M]` EXIT=1, **13 WARNINGs**, all `[nexus.directive]`: `.. implements::` blocks whose
`:by:` the carve had retired or moved (`ScatteringOperator._assemble_per_ordinate_source`,
`ScatteringMaterialField.add_p0_source`, `IsotropicScattering.apply`,
`LegendreMomentScattering`, `N2NMomentOperator`, `ScatteringOperator.{kernel,_apply_impl,
build_aniso_source}`). AGENT.md says the acceptance gate is *count-unchanged from a freshly
measured baseline* — and here that rule would have licensed shipping 13 errors.

⟹ **When a carve lands WITHOUT its docs pass, the baseline IS the deliverable's first
item.** Measure it; if it is red and the red is the carve's, the gate becomes EXIT=0, not
count-unchanged. State which it is in the report's §0, before any prose.

⭐ And the shape of what breaks: a `.. implements::` `:by:` is the ONE doc surface that
resolves against the GRAPH rather than by import, so it warns where a `:class:` role sits
silent. A carve that renames a method breaks the declarations loudly and the cross-refs
quietly — expect both, and do not read the loud ones as the whole radius.

### (b) ⭐⭐ AN INHERITED MEMBER REF IS NOT DEAD — `dead_references` rescues it, and that
### decides how wide the mechanical sweep should be

When a body moves onto a new shared CORE and the old class becomes a thin subclass, every
`:attr:`OldClass.member`` still resolves — by inheritance, in Python AND in nexus (`[M]`
`rescued: 66` of 75 checked). So the sweep is NOT "re-point every member ref"; it is:

| the ref | fate |
|---|---|
| a member that was RENAMED or RETIRED (`full_scatter_kernel`, `.energy`, `scattering_order`) | **DEAD — must re-point** |
| a member that MOVED to the core and is inherited (`.kernel`, `.apply`, `.frame`) | resolves; re-point only where the SENTENCE claims it is defined there |
| a `:by:` target | **must name the DEFINING node** — inheritance does not rescue a graph edge |

`[M]` 33 `ScatteringOperator.<member>` / `N2NOperator.<member>` refs corpus-wide; 20
"MOVED", and only the 5 genuinely renamed ones were dead. Re-pointing all 33 would have
been churn AND would have lost information (a reader of S's chapter wants S's name).

### (c) ⛔⛔ ON A LIVE BRANCH THE ELEGANCE PASS LANDS INSIDE YOUR TASK — re-read the surface
### after EVERY build, and the thing it moves is the thing you just wrote about

L-089's loop, at full force. `[M]` between my first `vars(TransferOperator)` dump and my
last, `orpheus/` gained 25 modified files and then a commit (`f52877db`, *"a transfer role
is two class constants and no code"*):

* `TransferOperator.scattering_order` → **`legendre_order`** — this was the ONE real dead
  role my probe found, and it did not exist when I started;
* `LegendreMomentTransfer.from_field` → **`on_basis`**;
* **`from_solver_data` MOVED from the role subclasses to the CORE**, the roles keeping
  `channel`; then `channel` became a **`ClassVar` holding a bound constructor**, not a
  classmethod — so *"each carries only its extraction classmethod"* went from true to
  false to false-in-a-second-way, and I wrote it TWICE before re-reading.

⭐ **The correction was strictly better prose**: "a role is two class constants and no
code" is sharper than "only its extraction classmethod", and the AST gate's own tightening
("refuse ANY method on a role") is the sentence's evidence. `sphinx -E -W` was **EXIT=0 with
the false text in it, in every build**.

⟹ Read `git log --oneline -3` and `git status --porcelain -- orpheus/` **at the end**, not
only at the start; a clean `orpheus/` late in a task means the pass COMMITTED, and the
commit message is the diff of your own prose's premises.

### (d) ⭐⭐ RE-RUN A PUBLISHED CENSUS; ITS COUNT CAN BE RIGHT AND ITS MEMBERS INVENTED

`adjoint.rst` carried `[M] 9 Sig2[0] sites = 2 model + 7 physics`, with the seven
enumerated. Re-run by AST on the post-carve tree: **7 sites, all 7 correct**. Two findings,
and the second is the transferable one:

1. the two model sites are gone, and `material_xs_field.py:682` **changed COLUMN without
   changing line** — it was a model site while the gain read it, and is now a reaction-rate
   site (removal + the σ_r fold predicate). The same expression, right for a different
   reason ⟹ a `fate at step 2` COLUMN on the old table, not a deletion.
2. ⛔ one of the seven enumerated members — *"``gendf.py``'s `if sig2[0].nnz > 0` guard,
   ×1"* — **does not exist and never did**: `grep -rn nnz orpheus/` at HEAD and at both
   parents returns a docstring, a `repr` and the HDF5 schema. The COUNT was right; a
   MEMBER was invented. A re-read cannot find that; only a re-run can.

### (e) ⭐⭐ A `verifies()` MARKER DECIDES WHICH BODY A LABEL KEEPS — read the TEST BODY,
### not the label's name

Generalising an equation, the natural move is to broaden the existing label and mint a new
one for the special case. That is **backwards** when the existing label is a `verifies()`
target: `n2n-source` is targeted by `test_solver_components.py:175` on
`SNSolver._add_n2n_source`, whose live body is
`self.n2n_op.isotropic_energy.transfer.add_p0_source(Q, phi)` — a **P0** claim. I had
already broadened it before reading the body.

⟹ **Order of operations for generalising a labelled equation:** grep `tests/` for the
label → read the claiming test's BODY (not its name, not its docstring) → the existing
label KEEPS the body its marker asserts → mint the NEW label for the generalisation. Same
for `sn-n2n-isotropic-lift` / `sn-n2n-adjoint-source`: both stayed, re-scoped to the ℓ = 0
block, with the RANKING against the new per-ℓ labels stated in prose so a future citer
picks the right one.

### (f) ⚠ An `.. error-entry::` HAS NO ANCHOR — `:ref:`ERR-NNN <err-nnn>`` is silent death

`[M]` the directive (`sphinxcontrib/nexus/directives.py:ErrorEntryDirective`) emits a
`container` + `rubric` with **no `id`**. A cross-doc dangling `:ref:` renders as plain text
with no warning at any severity, so the citation would have looked fine forever. The corpus
convention is plain-text `ERR-NNN` + `:ref:`the L0 error catalogue
<theory-verification-error-catalog>``. Check the DIRECTIVE's `run()` before inventing an
anchor scheme.

⭐ Sibling, caught by `-W` this time: a `:ref:` to a label sitting on an `.. important::`
(not a section title) MUST carry explicit text — bare `:ref:`x`` is `ref.ref` *"A title or
caption not found"*, and it fires **cross-doc** as well as intra-doc.

### (g) ⭐⭐ A LADDER MEASURED BEFORE THE CARVE IS A DIFFERENT MEASUREMENT OF A DIFFERENT
### TREE — both legs of the ratio moved

The flagship gate's docstring relays step 1's elastic ladder (`−229/−163/−175/−173 Δk·1e5`
at `L = 3…6`). Post-carve **both legs moved**: the `L = 2` baseline went
`1.0953221881419453` → `1.0911996566537725`, and the (n,2n) moments now enter at every
order. I re-measured (7 arms, ~45 s, `scratch/_426_step3_order_ladder.py`) and published
**mine**: `−235.06/−167.34/−179.56/−177.48`, with a ⚠ naming why step 1's is not a
re-rounding of the same thing.

⭐ **And the row that is NOT evidence:** the shipped ladder's `ℓ ≤ 6` arm equals `ℓ ≤ 2`
**to the bit** — because every arm runs at `scattering_order = 2`, so `Λ` has three blocks
and `ℓ ≥ 3` is never read. Quoted as *"higher orders add nothing"* that row quotes the
truncation, not the physics. Say so at the table, or the next reader banks it.

### (h) ⭐ `**:math:` WITH NO SEPARATOR IS THE ONE NESTING THAT BREAKS

`[M]` the corpus's normal `**word** :math:`x`` idiom is fine at ~60 sites; only
`**:math:`\mu`-reversal**` — a role opening *immediately* after a `**` start-string —
fails, rendering `` :math:`mu`-reversal `` as literal text with the backslash eaten, on a
**0-warning** build. Two such sites were pre-existing in `adjoint.rst` and two more
(`**Scattering (:math:`S`)**`) in `slab_multigroup.rst`; both pages went to **0 visible
backticks**. Fix by `:math:`\mu`\ **-reversal**` or by moving the bold onto a word.

⟹ The MINE-vs-pre discriminator that made this cheap: scan for `(\*\*|\*)"?``` and
`\*\*[^*]*:math:`` and test each hit line against `git show <carve-hash>:<file>` — one
pass separates *your* nesting bug from the page's.

### (i) The gate battery that ended this task, and why four instruments not one

| instrument | reads | verdict |
|---|---|---|
| `sphinx -E -W -q` from REPO ROOT | directives + labels + intra/cross-doc `:ref:` | 13 → **0**, 0-byte log |
| `mcp__nexus__dead_references` | RENDERED targets, with inheritance/re-export rescue | 9/16 sites → **0/65** |
| `tools/check_docstring_xrefs.py` | fully-qualified roles, by IMPORT | 0 both sides (role-blind, L-062/L-067 — proves nothing alone) |
| own import probe over `docs/` roles | every `orpheus.*` role, 3 fallbacks | 1 → **0 of 1355** |

⚠ **The own-probe needs THREE fallbacks or it cries wolf 26 times**: `hasattr` →
`dataclasses.fields` → `typing.get_type_hints`/`__annotations__` → a regex for
`self.<name>` in the class source. Without them, every dataclass field (`Mixture.Sig2`) and
every `__init__`-assigned attribute reads dead. `[M]` 26 → 3 → and the last two
(`SNMesh.axes`, `SNMesh.axis_widths`) needed L-093's CONSTRUCT-the-object step to clear.

### (j) The retired-spelling census: the COUNT is not the gate, the ADJUDICATION is

`[M]` before/after over `docs/` source with `_build` excluded (⚠ include it and
`ScatteringKernel` reads **1704** instead of 4 — the built HTML is in the tree). Three
counts ROSE (`N2NMomentOperator` 8 → 11, `add_emission` 3 → 4, `moment_emission` 0 → 3),
which is **correct**: a retirement that is narrated costs more words than one deleted.
⟹ the acceptance evidence is a regex classifying every survivor **role-vs-literal**:
`[M]` 45 survivors, **0 roles**, all ``literal`` in dated past-tense prose.

---

## L-095 — #448: the finalize's twin path, and the pre-carve worktree that was FREE

**Task.** Docs pass for the SN eigenvalue-finalize carve (#448): mint ERR-083, re-point 30
retired-symbol sites, write the finalize's own theory section, add the SN changelog row.
Branch `fix/sn-eigenvalue-finalize-448`, production carve **UNCOMMITTED**, a
`test-architect` editing `tests/` concurrently.

**Exit.** `sphinx-build -E -W -q` **EXIT=0, 102-byte log (0 diagnostics)** from a baseline of
**EXIT=1, 1 WARNING + 4 nexus `catches` messages**. `dead_references` **0/67**. My own
dotted-role probe **0 dead / 7056**; `:ref:`/`:eq:`/`:doc:` **0 dangling**; `:by:` **0 dead /
416**; stock gate **0/14 968**; HTML backtick gate **0 runs** in all 8 new slices. 16 docs
files, +956/−128.

### (a) ⭐⭐ THE PRE-CARVE WORKTREE IS FREE WHEN THE CARVE IS UNCOMMITTED — **HEAD IS the
before-tree**

L-050 says a pinned pre-carve worktree turns one number into a before/after table, and reads
as expensive. It is not, in the normal case: while a carve is uncommitted,
`git worktree add /tmp/x HEAD --detach` **is** the pre-carve tree, in ~20 s. I ran the SAME
probe on both and got a genuine before/after for ERR-083's two tables — the half nobody could
have relayed me honestly.

⚠ L-050's venv trap is real and I hit the guard: the editable install hooks `sys.meta_path`
and OUTRANKS `sys.path`, so strip the finder and **print `orpheus.__file__` as proof**:

```python
sys.meta_path = [f for f in sys.meta_path if "editable" not in type(f).__module__.lower()]
sys.path.insert(0, "/tmp/orph448_pre")
import orpheus; assert orpheus.__file__.startswith("/tmp/orph448_pre")
```

⟹ **when the brief hands you a pre-carve `[M]` table, ask whether HEAD is that tree.** If it
is, measuring it yourself costs less than deciding whether to trust the relay.

### (b) ⭐⭐ A BEFORE/AFTER TABLE NEEDS ITS **COMPARABLE STATISTIC** NAMED — the absolute row
can move the "wrong" way while the claim holds

`[M]` the post-fix `balance_defect` at `L = 0, max_outer = 3` is **13.7× LARGER** than the
pre-fix one (1.2497e-05 → 1.7108e-04), and `|∫ψdΩ−φ|/|φ|` is 3.5× larger — on the arm the
fix was supposed to leave alone. Both readings are correct: the two finalizes CONSTRUCT the
returned flux differently, so at a truncated exit their defects are not comparable at all.
What is comparable is the **ratio down the budget** — before 1.45e6 (L=0) / **1.0002×** (L=1),
after 1.43e7 / 3.46e7.

⟹ publishing the table without that sentence would have shipped a regression narrative inside
a repair entry. **Any cross-carve table owes a line saying WHICH statistic is comparable and
why the others are not.** The tell that you need one: the control column moved.

### (c) ⭐⭐ A CARVE'S OWN DOCSTRING EDITS ARE UNVERIFIED CLAIMS — census the verb it says
SURVIVES

The carve rewrote `SNBoundaryOperator.reflect_inflow_inplace`'s docstring to *"Its production
consumer is the octant-group Gauss-Seidel resolvent's face-restricted inter-group reflect"*.
`[M]` by AST over `orpheus/`: **0 Call sites, 0 attribute references** — for it AND for its
ψ½ sibling. The scheduled sweep binds `SNMaskedBoundaryOperator.reflect_rows_inplace`
(`scheduled_invertible.py:274`), which the SAME docstring's ⚠ paragraph says correctly four
lines below, and which also records that `_GaussSeidelResolvent` was **dissolved**.

⭐ The shape: a retirement asks *"who called the thing I am removing?"* and answers it well;
nobody asks *"and does the thing I say still has a caller actually have one?"* ⟹ **when a
carve retires a caller, AST-census the CALLEE it claims survives** — and read the rest of
that callee's own docstring, because the contradiction is usually already in it.

### (d) ⭐⭐ THE HTML GATE CAUGHT WHAT `-W` **AND** MY SOURCE REGEX BOTH MISSED — a role
several lines INSIDE an open `**bold**` run

I ran L-074's source scan for `**``` and `**:role:` **adjacency**: 19 hits, all pre-existing
(verified per-line against `git show HEAD:<file>`). The build read **EXIT=0**. The HTML slice
then found a real one:

```rst
⛔ **The complement of a guard … until 2026-09-06 (#448 /
:doc:`ERR-083 </theory/verification/error_catalog>`).**  The two verbs
```

— the role is three lines inside a bold run that OPENED earlier, so no adjacency pattern sees
it, and it renders as literal `:doc:\`…\`` text on a zero-warning build.

⟹ the source-side check must **pair `**…**` spans across newlines and look for a role or
literal INSIDE** (`re.compile(r"\*\*(.+?)\*\*", re.S)` over the added lines, then
`:[a-z:]+:\`` on the inner). And the HTML slice stays the authority. Cheap and decisive:
strip tags, unescape, slice by section title, `re.findall(r"`+")` — **0 is the gate**.

### (e) ⭐⭐ A MODULE UNDER CONCURRENT EDIT: publish NOTHING you count in it

The gate module grew **45 → 86 rows** and its arm registry **7 → 8** (`cart2d_gs`, added at
R2) while I wrote. I had already drafted "45 rows over seven arms" from the brief. Caught by
re-censusing `_ARMS` **by AST at write time** (`ast.AnnAssign` → dict keys) rather than
trusting the module's own §6c table, which still listed 7.

⟹ two moves, both cheap: (1) point at the artefact that RE-MEASURES it (`matrix.rst`
regenerates the row count) instead of quoting a number — `plan-authoring` §9 applied to a
sibling agent's file; (2) for a count you must state, **date it and name the census**
(*"`[M]` 8 arms on 2026-09-06"*), and check whether the pre-carve claim used a different
denominator (the "ten rows RED before" is right *for the 7-arm registry it then had*, and
saying so is what keeps it true).

### (f) ⭐ `catches("ERR-NNN")` markers make `-W` FAIL, and `grep WARNING` UNDER-counts them

Baseline was EXIT=1. `grep -cE "WARNING:|ERROR:|CRITICAL:"` read **1**; the log had **5**
lines. The four nexus `catches` messages carry **no `WARNING:` prefix** — they are the merge's
own per-marker diagnostics. ⟹ on an error-catalogue task, read the whole log, never the grep
count; and the gate is **EXIT=0**, because both classes are the carve's own red (L-094).

### (g) ⭐ My dotted-role probe needed a THIRD fallback: the **MRO**

L-093 added "construct the object" for `SNMesh.axes`. Cheaper and sufficient: search
`self.<attr>\s*[:=]` over **every class in `cur.__mro__`**, not just the named class —
`SNMesh.axes` / `.axis_widths` are set by `MaterialMesh._init_data` on the BASE. That took my
probe from 8 false dead to **0**. The four fallbacks in order: `hasattr` → `dataclasses.fields`
→ `self.x=` **across the MRO** → construct.

⚠ And two probe-hygiene notes that cost me a cycle each: strip the `py:method:` / `py:func:`
/ `py:data:` **prefix form** of `:by:` (8 false dead), and treat a leading `!` as a
deliberately-SUPPRESSED xref (6 false dead — `docs/theory/references/peierls_nystrom.rst`
uses `:py:func:`!old.name`` for a "Retired symbols" table, which is the right idiom and
worth copying).

### (h) ⭐⭐ A DELEGATOR LIST'S STATED **REASON** can be false while its membership is right

`slab_multigroup.rst` said four delegators *"remain as thin wrappers … for the EigenvalueSolver
Protocol surface"*. `[M]` the Protocol declares **five** members
(`initial_flux_distribution`, `compute_fission_source`, `solve_fixed_source`, `compute_keff`,
`measure_stopping_criteria`) and **only one** of the four is among them. The real reason three
of them survived is that the finalize called them — i.e. **the sentence's own false
justification is where the defect lived**. ⟹ when a doc explains why dead-looking code is
kept, check the explanation against the named surface; a wrong reason is a stronger finding
than a stale name, and it belongs in the ERR entry.

⚠ The brief said the Protocol declares **three**. A relayed member list is a SAMPLE
(`plan-authoring` §2) — `sorted(Proto.__protocol_attrs__)` settles it in one line.

### (i) ⭐ The ERR entry's independence claim: `0.000e+00` is the WEAKEST number on the page

Post-carve the returned flux is bit-identical to my independently-assembled one-step
reconstruction. That is L-050 exactly: **the two spellings are one float program now**, so
`0.0` measures single-sourcing, not agreement. The entry says so and carries the *cross-route*
row (the fixed-source entry's own reconstruction) as the independence claim instead — with
its own control (the two entries' **φ** agreed at 1.67e-11 while their **ψ** differed by
1.4728e-01, which is what proves the disagreement was ψ-only).

### (j) The site inventory — 30, not the briefed 26

`_add_scattering_source` **8** (briefed 7) · `_build_aniso_scattering` 4 · `_add_n2n_source`
**5** (4) · `add_iso_source` 3 · `build_aniso_source` **10** (8). Plus `_psi_typed` 4 (3 live)
and `_reflect_outflow_into_inflow` 5. `SNSolver._boundary_flux`: **0** doc sites — a narrowed
grep, because the bare token matches 32 lines of `ScalarBoundaryFlux` / `angular_boundary_flux`.
Post-edit: **36 surviving mentions, 0 xref roles**, every one a dated literal.

---

## L-096 — CS4c step 6.1: the identity flip, and the CLASS LIST that three surfaces got wrong together

**Task.** The `FunctionSpace.__eq__` identity flip (structural equality for axis-built spaces)
landed while I wrote; my job was to make 6 `docs/**/*.rst` stop promising it. Files:
`spaces.rst` (Key-Facts bullet, the `spaces-identity-bridge` section, the CS5 generator
argument, the roadmap row, the CS1 quadrant table, a dated changelog row), `history.rst` (a
dated row + a NEW changelog entry), `api/numerics.rst`, `operator_adjoint.rst` ×2,
`manifolds.rst`, `boundary_conditions.rst`. No build (one at a time is a hard rule here); no
commits.

### (a) ⭐⭐ The carve was IN THE WORKING TREE — so the code's own docstrings were the spec

`git status --porcelain -- orpheus/` at pick-up read `M orpheus/numerics/{axis,space}.py`
(+99 −49) even though `git status` in my dispatch snapshot showed nothing: **the main agent
started writing between the brief and my first command.** Reading `git diff -- orpheus/` in
full beat paraphrasing the brief — I adopted the code's own spelling (*"an axis-built space
compares and hashes by its `axes` tuple directly"*, *"the derived name survives as the readable
label"*), which is internal consistency for free. ⟹ **on a live branch, `git diff -- orpheus/`
is the FIRST command of a docs pass about an in-flight carve, not `git show HEAD:`.**

### (b) ⭐⭐ THE FINDING: a class LIST shared by the brief, the landed code docstring and my draft was wrong for 3 of 7 — and the error made a ⛔-reserved seam look closed

All three surfaces said *"the digest-named leaf classes — `FullFieldSpace`, the trace spaces,
`SphericalHarmonicSpace`, `LegendreSpace`, `SpatialMomentSpace`, `RadialCharacteristicSpace`"*.
`[M]` by CONSTRUCTING each one:

| class | `.name` on a real mint | digest? |
|---|---|---|
| `FullFieldSpace.from_blocks` | `full_field#4bbf8616c6f95085` | ✅ |
| `AngularTraceSpace` / `ScalarTraceSpace` | `angular_trace#…` / `scalar_trace#…` | ✅ |
| `RadialCharacteristic{Interior,Boundary}Space` | `{_SPACE_NAME}#…` | ✅ |
| `SphericalHarmonicSpace.from_L(2)` | **`'spherical_harmonic_space'`** | ⛔ |
| `LegendreSpace.from_L(2)` | **`'legendre_space(S^2/O2_x)'`** | ⛔ |
| `SpatialMomentSpace` | **`'spatial_moment_space'`** | ⛔ |

⭐ **Why it is load-bearing rather than pedantic:** the brief ALSO said (correctly) *"leave the
metric-blind seam statements untouched — the SH head is still metric-blind after this commit"*.
Those two claims are **inconsistent**: if the SH head were digest-named, its `(name, shape)`
WOULD be content identity and the seam would look closed. Publishing the list as given would
have made the corpus contradict its own ⛔. The corrected law — *"axes-less spaces keep
`(name, shape)`; for the five content-digest-named classes (four digest-folding factories) that
IS content identity; for the moment heads the name is a FAMILY tag whose only identifying
content is the order carried in `shape`"* — is what makes both true at once, and it EXPLAINS
the seam instead of merely reserving it. ⟹ **when a brief hands you a class list AND a
carve-out, check the list against the carve-out: a list that contradicts its own exception is
the finding.** Reported the same wrong list in the landing code docstring upward (docs-only
scope; I do not edit `orpheus/`).

The cheap instrument: don't grep for `#` in the factories — **construct each class and print
`.name`.** A grep for `hashlib` found 4 of the 4 factories but says nothing about the 3 that
have none, which is the half that mattered.

### (c) ⭐ A pre-existing false sentence surfaced by the triage, invisible to every gate

`operator_adjoint.rst:360` claimed *"``name = "full_field"`` … so two composites over meshes of
the same total dimension compare equal"* — false since the **CS4b S3/S4** digest re-key (that
bare name IS the R2 block-blindness the re-key retired). `-W` cannot see it: it is prose, and
every xref in it resolves. Found only because item 6 said *"triage each"* and I built the
object. ⟹ a triage item whose expected answer is *"still true, leave it"* is where the
pre-existing falsehoods live — build the object anyway.

### (d) ⭐ The two rows that ARE a flip, and the one that is DERIVABLE

For an `__eq__` change, the publishable evidence is not "the doctrine still holds" but the
rows that MOVED. `[M]` on the carve: axis-built vs a hand-named twin carrying its exact
`(name, shape)` went `True` → **`False`** (a label stopped being a *credential*), and
`A * B` vs `of_axes(*A.axes, *B.axes)` went `False` → **`True`** even though the two derive
DIFFERENT names (a name stopped being the *identity* — Q-T4 realized). The two UNCHANGED rows
were published beside them so the table cannot read as "everything moved".
⭐ **The "before" column cost nothing and needed no worktree**: the pre-flip body is exactly
`self.name == other.name and self.shape == other.shape` (`git show <HEAD>:…`), so it is
DECIDED by the names the probe printed — publish it as `[R]`, with the derivation stated. When
the old law is a pure function of quantities you already measured, a pinned worktree (L-095) is
over-engineering.

### (e) ⚠ MY OWN quantifier defect, caught on the final read

I published *"the ``[M]`` **four** content-digest-named classes — A, B, C **and the two**
D"* — a list of FIVE under the word *four*. The number counted digest *factories*; the list
counts *classes*, and two share a factory. ⟹ **when a count and a list sit in one sentence,
check they count the same objects** — the slip survived three re-reads because both halves were
individually right. Corrected to *"(five classes, four digest-minting factories)"*.

### (f) ⛔ Ambiguous-name discipline, applied AT THE BANNER

"S3" is a 3-way homonym here (CS4b S3 re-key LANDED · #240 D5b-S3 LANDED · this flip). Every
new sentence says *"the identity flip (structural ``__eq__``), CS4c step 6, 2026-09-07"*. The
retired roadmap row is preserved VERBATIM inside a ⛔ tombstone, and the tombstone itself carries
the disambiguation (*"the 'S3' of that retired sentence is this plan-internal step, NOT the
landed CS4b S3 re-key"*) — `plan-authoring` §3's ambiguous-name clause says disambiguate at the
BANNER, because the banner is what gets read. Post-pass census: **0** promissory `S3` left in
`docs/`; the 6 survivors are all landed homonyms.

### (g) ⭐ Gates without a build (Sphinx unavailable — one build at a time)

The instrument set that worked, and its readings: a **pre-edit vs post-edit docutils error-SET
diff** (`git show HEAD:<file>` vs the working tree, Sphinx-only roles/directives filtered as
noise) — **0 before / 0 after on all 6, 0 NEW, 0 GONE**; a nested-markup gate over my 246 ADDED
lines (`**``literal``**`, `**:role:`, 3+ backtick runs) which caught **2 real hazards**
(`**``False``**`, `**``True``**`) nothing else would have; `:ref:` targets against the corpus
label set (1, live); python xrefs **import-resolved** (7 of 7 live, plus the bare
`:meth:`__eq__`` corpus idiom with 38 precedents); `:doc:` targets against the filesystem; and a
79-column check on added lines. ⟹ this set is now my default when a build is unavailable — the
error-SET diff is the honest substitute for a warning count, and the added-lines gates find what
a build never would.

---

## L-097 — CS4c step 6 item 6.2a: the carve landed MID-TASK, and the surplus was a claim CS4b falsified two weeks earlier

**Task.** 2026-09-07, dispatched at `main` @ `77a12286`. Brief: 8 sites, `docs/**/*.rst`
only, no build, no commit. `*` (`TensorProductSpace.from_factors`) stops densifying; the
dense outer-product weights builder, the mixed-product bridge (`_dense_axes_weights`) and
the `FunctionSpace._broadcast_metric` shim retire. Delivered: 3 files
(`foundations/spaces.rst`, `foundations/operator_adjoint.rst`, `methods/sn/history.rst`),
+282/−51, 0 new eq-labels.

### ⭐⭐ (1) `git status` was EMPTY at dispatch and the code landed while I read — the §0 measurement was stale within one tool call

`[M]` at dispatch `git status --porcelain -- docs/ orpheus/ tests/` was empty and
`grep` over `orpheus/numerics/space.py` still found all four retiring symbols. I wrote
that into the report as §0. Two reads later, `space.py`'s `_tensor_product_factored_metric`
docstring began *"**History.** Until 6.2a ``*`` had a DENSE arm…"* — the carve had landed
in the working tree (6 `orpheus/` files, +141/−157, plus 3 `tests/` files).

⟹ **the L-089 loop is not "re-read after every build" — on a same-commit docs+code task
it is "re-read after every FILE".** The tell that caught it was a docstring that already
narrated the change I was about to document; had I not re-run `git status` I would have
published a §0 that reads as a measurement and is a snapshot of a tree that no longer
existed.

⭐ **And the landing was a GIFT, not a hazard: it turned every design claim from relayed
to verifiable.** The brief's entry rule (*"one positioned entry per AXIS of an axis-built
factor, one per dense-slot leaf, a metric object verbatim"*) stopped being something I had
to take on faith. Three lines:

```
h = SphericalHarmonicSpace.from_L(2)                  # the dense-slot leaf head
b = FunctionSpace.of_axes(energy(2,), spatial(4,) weighted)
(h * b).metric.entries
→ [((3,5), DiagonalMetric), ((2,), None), ((4,), DiagonalMetric)]
```

— the head as ONE entry, then one entry PER AXIS of the axis-built factor, `None` for the
counting-measure energy axis. `axes` is `None`, `inner_product_weights` is `None`. The
published rule is now witnessed by construction rather than relayed.

### ⭐⭐ (2) TWO of the brief's own premises were superseded by the landing — and one of them was the CENSUS's stated reason, not its conclusion

(a) The brief and the code census (`explorer_boundary_recensus.md:159`) both said the P7
factored arm *"itself densifies — it calls `f._dense_axes_weights()` at `:939` and wraps
the DENSIFIED array in a `DiagonalMetric`"*. `[M]` on the landed tree the builder positions
per axis (`for ax in f.axes: entries.append((ax.shape, … DiagonalMetric(ax.weights)))`,
with the comment *"Per AXIS, never per factor"*). The census's CONCLUSION (re-point per
axis) was right; its stated MECHANISM was already history. I published the conclusion and
not the reason.

(b) The brief named the surviving `*` occupants as *"the head ⊗ bulk moment product until
6.2c; the trace/spatial-moment products"*. `[M]` mine, AST `BinOp(Mult)` over
`orpheus/**/*.py` whose unparsed source names a space/head/basis: **7 hits, 4 space
products, ALL harmonic/moment** — `harmonic_frame.py:484` and `:491`, `_bases.py:436` and
`:913`; the other three are `derivations/` arithmetic. **There is no trace-space `*`
product.** I published the measured set and dropped the brief's phrase.

⟹ same shape as L-096: a brief's list is a classification and it can be wrong. The cheap
instrument here was an AST census, not a grep — `*` is unspellable as a symbol.

### ⭐⭐ (3) THE FINDING: the surplus was a paragraph CS4b falsified, sitting in the section I had to edit

`spaces.rst` carried, on the nodal/modal `has_coordinate_cone` table:

> `[M]` **the refusal has no production witness yet, deliberately.** The only axis mint in
> `orpheus/` today is `MaterialMesh.bulk_space`, whose factors are both `NODAL`. … The arm
> becomes production-reachable when CS2 mints the harmonic axis.

**Both halves false**, and *neither by 6.2a*:

* `[M]` AST CALL sites (not textual mentions — `material_mesh.py:412` is a docstring) of
  `of_axes` in `orpheus/`: **7**, not 1.
* `[M]` the `False` row HAS a production occupant: `SNMesh.angular_trial_space`
  (`augmented_mesh.py:1247`) appends `scheme.moment_axis(...)` — `kind=BasisKind.MODAL` at
  `transport/spatial/scheme.py:1726` — to an axis-built base when the scheme is
  multi-moment, and (walking `DiscretizationSchemeBase.__subclasses__()` recursively)
  `LinearDiscontinuous.is_multi_moment` is `True`, `DiamondDifference`'s `False`.

⟹ **a docs pass for item N is where item N−k's un-swept rot surfaces, and the trigger is
proximity, not predicate** — I found it only because my `of_axes`-closure sentence was a
near-twin of that paragraph's and I went to align vocabulary. (L-072/L-075's "a phase that
lands with no docs pass leaves its rot for the NEXT phase's sweep", now with the mechanism
named: the *next* sweep discovers it when it edits an ADJACENT sentence.)

⚠ **And the discipline that kept the repair honest: I measured the OCCUPANT, not the
FIRING.** An axis-built space with a MODAL factor exists in production; whether any
consumer asks `has_coordinate_cone` of it is a *different* census. The page now says that
in one clause instead of upgrading the claim — vv #29's not-run/no-consumer distinction,
applied to prose.

### ⭐ (4) A `[M]` exposure COUNT the carve moved, and the predicate that makes it reproducible

`spaces.rst`'s *"an empty weights slot no longer means Euclidean"* block carried
*"198 lines across 52 files … 20 are `is None`/`is not None` branches and only **three**
are production — all three inside `space.py`"*. `[M]` re-measured: **221 / 57 / 28 /
four**. I published mine WITH the filter (the 28 *includes prose lines*; the four are
*branches*), because my own first grep counted a docstring line as a branch.

⭐ The honest gloss took one extra check: the fourth branch is a change of **SPELLING**,
not of exposure — `[M]` at `77a12286` the builder funnelled a dense-slot leaf and an
axis-built factor through ONE local `w is not None` test (the latter via the densifier),
where the arms are now separate. I had first written "the fourth branch is genuinely new"
and retracted it against `git show HEAD:` before publishing.

### ⭐ (5) A roadmap row's PREDICTED MECHANISM was refuted by its own landing — and that is the row's best content

The row read *"CS2. The legacy `*` path … its own gates live in a separate test module so
the retirement is a **file-level move**."* `[M]` from `git diff -- tests/`: nothing moved.
`test_space_algebra.py`'s two dense-slot rows and `test_space_of_axes.py`'s mixed-product
third leg were **re-keyed in place** — each still asserts the same outer product, now as
`tp.apply_metric(...)` instead of as a stored tensor — and the new band lives in a third
module. ⟹ **a ✅ LANDED row is not a tense flip when the row also predicted a MECHANISM:
say which half landed and which was refuted.** *"A gate that pins behaviour migrates with
the behaviour; it does not travel with a file"* is the sentence a future reader needs, and
it only exists because the landing contradicted the plan.

### ⭐ (6) The xref bug only the IMPORT gate could see, and the `hasattr` trap under it

I wrote `:attr:`Solution.scalar_flux <orpheus.numerics.solution.Solution.scalar_flux>``.
`[M]` there is no `orpheus.numerics.solution` — it is `orpheus.sn.solution`. `-W` renders a
dead Python-domain role as plain text, silently. ⚠ And the corrected target ALSO fails a
naive check: `hasattr(Solution, "scalar_flux")` is **False**, because it is a dataclass
FIELD with no default — the L-076 ladder (`hasattr` → `dataclasses.fields` → `self.x=`
across `__mro__` → construct) is what resolves it, and the corpus's own
`solver.rst:649` confirms the spelling.

### ⭐ (7) `**` PARITY is a usable no-build gate — once you strip literals

Raw `**` count on `spaces.rst` went 960 (even) → 999 (**odd**), which reads as an
unbalanced bold run. `[M]` the culprit is `` ``orpheus/**/*.py`` `` — a `**` inside an
inline literal, not a markup token. Stripping ``…`` and `:math:`…`` first gives EVEN → EVEN
on all three files. ⟹ **strip literals and math before any markup-parity check**, or the
gate cries wolf on the one construction a docs pass about globs is guaranteed to add.

⚠ Sibling, re-confirmed: the naive `\*\*(.+?)\*\*` + `re.S` "role inside bold" probe is
**useless** — it pairs one run's CLOSING `**` with the next run's OPENING one and reports
**132** hits on a clean file. The gate that works is L-094's precise one: `\*\*:[a-z]+:`` (a
role opening with NO separator after `**`), plus `` \*\*`` `` / `` ``\*\* `` and `` `{3,} ``.
All three read **0** here.

### Gates (no Sphinx build — the main agent runs the single pre-commit build)

| gate | result |
|---|---|
| docutils permissive parse, error-SET diff pre-vs-post, 3 files | **0 / 0** each |
| role-after-`**` · bold-abuts-literal · 3+ backticks (ADDED lines) | **0** |
| `**` parity outside literals/math, pre-vs-post | EVEN → EVEN ×3 |
| `:ref:` / `:eq:` vs corpus label set; `:doc:` vs filesystem (ADDED) | **0** dead |
| python xrefs IMPORT-resolved (ADDED) | **23 live / 0 dead** |

### Reported, not edited

`tests/numerics/test_space_of_axes.py:244` still names `FunctionSpace._broadcast_metric` in
a docstring; the carve updated three other docstrings in that same file and missed it. The
independence ARGUMENT it makes is intact (and stronger post-6.2a) — only the symbol died.

### Score

| dimension | score | why |
|---|---|---|
| Derivation depth | 4 | the entry rule is stated AND witnessed by construction; no new math |
| Cross-references | 5 | 23 xrefs, all import-resolved; one dead one caught pre-publication |
| Numerical evidence | 5 | every number carries its configuration; the ladder is CITED, not copied |
| Failed approaches | 5 | 3 tombstones + a refuted roadmap MECHANISM + 2 surplus retractions |
| Code traceability | 5 | every claim measured against the live working tree, incl. `git show HEAD:` |
| Derivation source | n/a | no `derivations/` content in scope |

---

## L-098 — CS4c step 6 item 6.2b: a TABLE'S CAPTION owns its columns, and a brief's site-ordinal can point at a non-sequitur

**Task.** `docs/**/*.rst` only, for the item that gives the harmonic-moment space to
the SN carrier (`SNMesh.moment_space(L, *, spatial_moments=1)`, a keyed cache) while
the moment field family becomes readers. Four briefed sites, three files touched
(`frame.rst`, `spaces.rst`, `history.rst`; +179/−14). Every gate clean; two
deliberate deviations from the brief, both on correctness/exposition grounds.

### (a) ⭐⭐ A `.. list-table::` CAPTION fixes what each column MEANS — so "update column X" can be an instruction to falsify the caption

The brief said: *"the SITE column now reads `SNMesh.moment_space(L)` (the hub)"*.
The table is captioned **"`[M]` The seven re-mint sites, at the pre-2.5 tree"**, and
its columns are `Site | What it was minting | Now reads`. Putting the hub in **Site**
would have made the caption false — the hub did not exist at the pre-2.5 tree.

⟹ **In a historical table, exactly ONE column is present-tense, and it is the only
one that can rot.** Find it by reading the caption, not the brief. Here the honest
edit was the third column: `` `SNMesh.moment_space(L)` — the hub, which reads
`mesh.quad.angular_frame(L).basis.space` `` — same information, caption still true,
and the historical row still answers *"which site used to re-mint?"*.

⚠ The generalisation, and it is cheap: **before editing a table cell, read the
caption and the column header as a CONJUNCTION.** A cell's meaning is
`caption × header`, and a brief written from the cell's *content* cannot see either.

### (b) ⭐ A brief's SITE-ORDINAL ("the first site") is a position; place by CONTENT and report the gap

The brief listed a three-site "metric-blind cluster" (`:4142-4153` / `:4326-4332` /
`:4380-4386`) and said *"add ONE dated sentence at the first site"*. The sentence it
asked for is about the **field-vs-face** metric fork; the first site is the
frame-internal `basis_space`-vs-`basis.space` measurement table, ~180 lines earlier,
where the reader has not yet met a field space at all. Written there it is a
non-sequitur that also front-runs its own exposition.

⟹ Place by CONTENT (here: the ⚠ **Gotcha** paragraph, the corpus's only statement of
the field-vs-face asymmetry), leave all three sites' claims untouched as instructed,
and **write the deviation into the report with the reason**. A silent relocation
reads as sloppiness; a stated one is a finding about the brief.

### (c) ⭐⭐ "Protocol X retires" can mean RENAMED ONTO A DIFFERENT SURFACE — and the new surface's predicate is the honest one, which is a publishable sentence

Brief: *"`_angular_head_space` and the `_CarriesQuadrature` protocol retire"*. `[M]`
at the landing: `_angular_head_space` genuinely retires (`hasattr` **False**);
`_CarriesQuadrature` is **replaced** by `_CarriesMomentSpace`, demanding
`moment_space` instead of `quad`. Not a rename — a different *question*.

The prose gain is real and would have been lost by a mechanical past-tensing: the
refusal used to be keyed on *carrying a quadrature* and is now keyed on *owning the
space*, **which is the honest predicate** — a carrier could in principle carry a
quadrature and still own no moment space. That sentence exists only because I
compared the two Protocols' surfaces instead of accepting "retires".

⟹ For every briefed retirement, ask **"retired, or re-surfaced?"** and diff the OLD
surface against the NEW one. `hasattr` on the module answers the first half in one
line; the second half is a read.

⚠ And the residual-census reading that goes with it: after the pass,
`_CarriesQuadrature` still has **1** hit in `docs/` — my own past-tense quotation of
the superseded sentence. That is the retirement rule working, not a miss. A census
that reports 0 for a retired name whose history you were supposed to preserve means
you deleted the history.

### (d) ⭐ A brief's "now spelled ONCE as `<new_name>`" is a PREDICTION about a carve still being written — and it did not ship

Brief: the append-iff->1 rule *"now spelled ONCE as `compose_spatial_moments` in
`_bases.py`"*. `[M]` at the end of the pass: `BulkField._compose_spatial_moments` is
**unchanged and still private**; `compose_spatial_moments` module-level is **False**;
the hub inlines its own `spatial_moment_tail` check, so the shared single source is
`spatial_moment_tail`, not a shared composer.

⟹ Consequence for the docs: the **three** `:meth:`BulkField._compose_spatial_moments``
refs (`operator_algebra.rst`, `cartesian_multid.rst` ×2) are LIVE and must NOT be
re-pointed on a brief's say-so. But a hoist later WOULD kill them silently
(Python-domain xref, no warning at any severity), so the report hands the renaming
commit an explicit *"grep `docs/`"*. **A brief describing code that is still being
typed is a forecast; verify the symbol exists before re-pointing anything at it, and
say in the report what breaks if the forecast comes true later.**

### (e) ⭐ The `*`-count census is an AST question, and a name-filtered `BinOp(Mult)` walk answers it exactly

`spaces.rst` carried *"`[M]` … four production sites"* for the `*` tensor product.
Re-run mine (all `ast.BinOp` with `Mult` whose `ast.unparse` mentions `space` /
`of_axes` / `from_per_axis`, positive control = the frame's two known sites): **4**,
and the SAME four post-carve — `augmented_mesh.py:1306, :1309`,
`harmonic_frame.py:490, :497`. So the census *count did not move* and its
**field-side member did**. That is the publishable sentence, and it needed the census
to be re-run rather than reasoned: the natural guess ("the hub adds a site") is wrong,
because the field-side site MOVED rather than being added.

⚠ The corroborating structural find: `_compose_spatial_moments`'s axes-less arm is
now a `raise TypeError`, so a `*` can no longer be reached through the fields'
composer at all — which is *why* the count stayed at 4.

### (f) ⚠ Two code-side findings, reported upward (I do not edit `orpheus/`)

1. `SNMesh.moment_space`'s own new docstring cites
   ``:func:`~orpheus.numerics.moment_layout.spatial_moment_tail``` — `[M]` **False**
   on `hasattr`; the function is in `numerics.spaces.spatial_moment_space`
   (`moment_layout` owns `face_moment_tail`). A **plausible sibling module** is the
   dangerous shape: it reads correct, resolves to nothing, warns at no severity.
   (L-095's *"a carve's own docstring edits are unverified claims"*, with a `:func:`
   target instead of a consumer claim.)
2. `BulkField._compose_spatial_moments`'s docstring bullet 2 still describes the
   axes-less arm as *composing a `SpatialMomentSpace` via `*`* — the body now
   **refuses** it. A docstring advertising a capability the body raises on.

### Gates, for the record

docutils error-SET diff pre-vs-post: `frame.rst` 515→515, `spaces.rst` 137→137,
`history.rst` 0→0, NEW set empty. Added-lines markup gates (3+ backticks /
`**``lit``**` / role-abutting-`**` / role inside an open `**…**` run with `re.S` /
`**` parity on literal-and-math-stripped text): **0**. `:ref:` against 881 harvested
labels: 2/2 resolve. Python xrefs import-resolved: 12 roles, 9 distinct targets,
**0 dead** — including the not-yet-committed `SNMesh.moment_space`, which resolves
against the working tree. `nexus dead_references` 0/68 — **corroborating only**, the
graph is a pre-carve snapshot.

⭐ One extra gate worth keeping: when quoting a superseded sentence that mixes
emphasis and inline literals, **render the note through `publish_string(...,
writer_name="pseudoxml")` and read the `<emphasis>`/`<literal>` alternation**. The
`\ ` null-whitespace escape between a literal and a following `*` is correct RST and
docutils is silent either way, so the render is the only proof the quote is not
leaking markers.
