# Archivist — Lessons (hot digest)

Read FIRST, every dispatch. One rule per lesson, imperative, with the failure→correction core that
earned it. War stories, evidence and `file:line` detail live in **`lessons_archive.md`** — open a
`→ L-0NN` section on demand. Mechanical HOW is NOT repeated here: build-gating, venv/worktree facts
and the 9-step close-out arc in `AGENT.md`; V&V vocabulary in `vv-principles`; Branch-1/Branch-2 in
`algebra-of-record`.

**THE SPINE — a page is DONE when** every cross-ref resolves against the LIVE tree · every claim
was verified against live code THIS session · every claim's V&V level matches the skill verbatim ·
every retired symbol leaves no present-tense-false mention · the build's WARNING/ERROR/CRITICAL
**set** is unchanged from a freshly-measured `-E` baseline. Every rule below is one face of that.

---

## 1. Ground truth is the LIVE tree — every other surface lies eventually

- **⭐⭐ WHEN A CARVE AXIS-IFIES A FACTOR, RE-DERIVE WHICH *ARM* OF THE PRODUCT RULE EACH
  SURVIVING SITE TAKES — re-counting the census is the easy half and misses the
  inversion.** `[M]` a rule read *"`*` is for a product whose factors are NOT all
  axis-built — four sites"*; the item killed the two TAIL products, so I re-counted to
  **two** and then asserted from reasoning that `*` still yields an axes-less product.
  FALSE: `from_factors` concatenates when EVERY factor is axis-built, and a prior item had
  axis-ified the head — so neither survivor is the mixed case, on 4 probed rows (width 1/2
  × diagonal/DENSE Gram). The stronger sentence the refutation bought: the two sites are a
  **retirement candidate**, not a counter-example. → L-101
- **⭐⭐ THE PRE-CARVE STATE IS MEASURABLE WITHOUT A PINNED WORKTREE — build a STAND-IN for
  the retired object, and the EQUAL-SHAPE control is the finding.** `FunctionSpace(name=…,
  shape=(4,))` in place of the retired axes-less class: the product's `axes` is `None`
  where the scheme's axis gives four labelled axes — **`shape` equal both ways**, which is
  exactly why `(name, shape)` identity could not see the two spellings. A worktree diff
  yields the two values and NOT that control, because the control is a construction.
  Prefer the stand-in when the retired object is shape-characterisable. → L-101, L-095, L-050
- **⭐ FOR ANY "ALL N OF FAMILY F DO X", RE-DERIVE F's MEMBERS, NOT |F| — a family can turn
  over with its count fixed.** `[M]` *"all three subclasses defining `__post_init__`"* is
  still three and the third member CHANGED (one retired, one arrived). A count-only check
  reports "unchanged"; a tense flip loses that the rule held across the substitution. Write
  both: past-tense the finding, `[M]` the roster today. Sibling, same page: *"the
  metric-blind family-tagged class describes only X"* → **EMPTY**. → L-101
- **⭐ RETITLING A SECTION REWRITES EVERY BARE `:ref:` TO IT — grep the label and
  explicit-text them in the SAME edit.** A bare `:ref:` renders the TITLE, so retitling
  from a retired class name to the invariant made four sentences read *"the same convention
  the *<the whole new title>* factor surfaces"* — and no build at any severity complains,
  because the ref RESOLVES. Keep the LABEL (docstrings key on it), fix the call sites.
  → L-101, L-099
- **⛔ A BLOCK REPLACEMENT THAT ENDS MID-PARAGRAPH IS SWALLOWED BY THE DIRECTIVE YOU
  INSERT.** My `.. note::` absorbed the sentence's tail into its body — the pass's ONLY new
  docutils diagnostic (`Explicit markup ends without a blank line`). Extend `old` through
  the sentence, or re-emit the tail at body level. This is the case the standalone
  docutils error-SET diff exists for. → L-101
- **⚠ DO NOT REBUILD THE KNOWN-USELESS BOLD-NESTING REGEX — the working pair is ADJACENCY +
  PARITY.** I rewrote `\*\*(.+?)\*\*` + `re.S` and got **7** false positives on my own
  lines. Working: the four adjacency patterns (0 hits) plus `**`/`*` PARITY per NEW
  paragraph after stripping roles and literals — and the parity gate must SKIP `.. math::`
  blocks (its one hit was `*\text{spatial}`). Parity is the source-side substitute for the
  HTML slice when Sphinx is unavailable. → L-101, L-097, L-095
- **⭐ A RETIRED GATE MODULE'S CLAIM LIST IS A MIGRATION CHECKLIST — walk it and ask, per
  claim, WHICH LIVE MODULE ASSERTS THIS NOW.** `[M]` the retired module was the only
  witness for `find_factor`'s `KeyError`-when-absent, and the sibling's copy was replaced
  by a label census in the same carve — a documented structural assertion now unpinned. A
  docs pass is where that list is in front of you. Report; you do not edit `tests/`.
  → L-101
- **⭐⭐ WHEN A RULING OVERTURNS A LANDED ONE, BANNER THE SECTION *TITLE* — §3 guarantees the
  refuted answer is what a top-down reader meets FIRST.** A page whose sub-section is titled
  with the QUESTION (*"Which space, though — X or Y?"*) or with a VERDICT (*"Why X is the
  right end"*, *"unchanged by item N"*) carries the old answer in its highest-decay sentence.
  Shape that worked: `.. important::` **under the title** (answered twice, both dates, the
  one-sentence ruling) → the original argument preserved verbatim, prefixed *(Written
  <date>, and preserved:)* → `⛔ REFUTED` with per-finding denominators → a NEW labelled
  section for the ANSWER so siblings `:ref:` the resolution, not the debate. Banner every
  sibling heading too. → L-100
- **⭐⭐ TWO `[M]` NUMBERS CAN BOTH BE HONEST AND DISAGREE — the difference is the STATISTIC;
  never relay a brief's "REFUTED" without asking what each side MEASURED.** `[M]` briefed
  *"10 of 33 was refuted → 5 of 33"*; both are right (whole-MATRIX Frobenius vs applied to
  ONE draw). What refutes the *conclusion* is a third statistic nobody had taken — **0 of 33**
  on a physical `φ = Mψ`, the differing columns being off the range of `M`. ⟹ retract the
  READING, not the number, and name the two defects separately (the statistic ranges over
  inputs production cannot produce; the comparison has one leg — the other end's cost was
  never priced). → L-100
- **⭐⭐ A `==`-GATE'S BLINDNESS IS A PROPERTY OF THE IDENTITY RELATION, NOT OF THE GATE — a
  re-typing silently upgrades or downgrades every equality assertion in the corpus, with no
  file touched.** `[M]` *"identity stays metric-blind … the invariant survives the dressing"*
  + a `vv #19` note saying a `==` gate "cannot adjudicate" both INVERTED when the measure
  entered the identity (0 of 33 where it was 33 of 33). ⟹ after such a carve grep
  `metric-blind` / `cannot tell them apart` / `(name, shape)`-equal and re-derive each: some
  go false, some become STRONGER claims that now under-sell a real gate. → L-100
- **⭐⭐ A "declared and unbuilt" SEAM is discharged by a different phase than it predicts —
  audit FOUR clauses, don't flip a tense.** `[M]` *"no MODAL axis has a generator; the arm
  becomes real when CS2 mints the harmonic axis"*: wrong PHASE (the metric's), wrong MECHANISM
  (`hasattr(Basis,"axis")` is **still** False — built inside `for_basis`, so the section law
  still does not range over MODAL axes), wrong OBJECT (the generator is the FRAME on a dressed
  head, not the basis) — and right once. Ask: predicted phase? mechanism? object? does the law
  that rode on it now hold? → L-100
- **⭐ A RETIRED ACCESSOR THAT RETURNED A METRIC-TWIN SPACE needs a per-site choice, and its
  dead `:attr:` refs are invisible at EVERY Sphinx severity.** `[M]` 11 dead refs in
  `docs/theory` + 4 in `orpheus/` docstrings only `dead_references` sees. **Live** description
  → re-point to the successor AND re-word if the sentence says a SPACE carries the probe (an
  arrow does not); **historical** entry → old name as a plain ``literal`` + name the successor,
  never re-point history's subject. → L-100
- **⭐ READ THE CHANGELOG'S OWN EXCEPTION CLAUSE, AND VERIFY THE ISSUE COLUMN.** The "lands with
  its hash" preamble excepts only a Where naming an *unmerged branch*; a row saying
  *"uncommitted … on `main`"* is not covered — `[M]` its commit subject matched an ancestor of
  HEAD, so I stamped it. Separately: `gh issue view` every number before writing it — my drafted
  `#448` was a different piece of work, and `—` is the convention when nothing tracks the row.
  → L-100, L-099

**Meta-rule: the brief is the FLOOR; live code is the rule.** Brief, docstring, verdict memo,
retirement shim, scanner finding, plan line and "MEASURED" block are point-in-time snapshots.
Verify, then write, then FLAG every scope-expansion the verification forced.

- **⭐⭐ ON A SAME-COMMIT DOCS+CODE TASK, RE-RUN `git status` AFTER EVERY FILE, NOT EVERY
  BUILD — the carve can land BETWEEN two reads.** `[M]` dispatched at an empty
  `git status` with all four retiring symbols present; two reads later 6 `orpheus/` +
  3 `tests/` files were modified and my §0 "the code has not landed" was a snapshot of a
  tree that no longer existed. The tell: a docstring that already NARRATES the change you
  were sent to document. ⭐ And the landing is a GIFT — it turns the brief's design rule
  from relayed to witnessed: construct the product and print `metric.entries`
  (`[((3,5), DiagonalMetric), ((2,), None), ((4,), DiagonalMetric)]` = one entry for the
  dense-slot leaf, one PER AXIS of the axis-built factor, `None` for a counting measure).
  → L-097, L-089
- **⭐⭐ A TABLE'S CAPTION OWNS ITS COLUMNS — "update column X" can be an instruction to
  FALSIFY the caption.** A cell's meaning is `caption × header`; a brief written from the
  cell's CONTENT sees neither. `[M]` briefed *"the SITE column now reads
  `SNMesh.moment_space(L)`"* on a table captioned *"the seven re-mint sites, **at the
  pre-2.5 tree**"* — the hub did not exist then. In a historical table exactly ONE column
  is present-tense (here *"Now reads"*), and it is the only one that can rot: move that
  one. ⭐ Sibling, same brief: a **site-ORDINAL** ("add the sentence at the first site") is
  a position, and the content can be a non-sequitur there (a field-vs-face sentence at a
  frame-internal table, 180 lines before the reader meets a field space). Place by
  CONTENT, leave the named sites' claims untouched, and write the deviation + its reason
  into the report — a silent relocation reads as sloppiness, a stated one is a finding
  about the brief. → L-098
- **⭐⭐ "Protocol X RETIRES" often means RE-SURFACED — diff the OLD surface against the
  NEW, because the new predicate is the publishable sentence.** `[M]` `_angular_head_space`
  really retired (`hasattr` False) while `_CarriesQuadrature` became `_CarriesMomentSpace`,
  demanding `moment_space` instead of `quad` — a different QUESTION, and the honest one
  (*a carrier could carry a quadrature and still own no moment space*). One `hasattr`
  answers "gone?"; only a read answers "replaced by what?". ⚠ And a residual census that
  reports **0** for a retired name you were meant to keep as history means you deleted the
  history — mine correctly reads 1, my own past-tense quotation. → L-098
- **⭐ A brief describing code STILL BEING TYPED is a FORECAST — verify the symbol before
  re-pointing anything at it, and report what breaks if the forecast lands later.** `[M]`
  briefed *"now spelled ONCE as `compose_spatial_moments`"*; at the end of the pass the
  module-level name is **False** and the private classmethod is unchanged, so 3 live
  `:meth:` refs must NOT move — but a later hoist kills them in silence, so the report
  hands the renaming commit an explicit *"grep `docs/`"*. → L-098
- **⭐ A `*`/operator-count census is an AST question and the natural GUESS is wrong.**
  `[M]` a name-filtered `BinOp(Mult)` walk (positive control = the two known sites) read
  **4** before AND after the carve — the count did not move, the field-side MEMBER did.
  "The hub adds a site" is the reasoned answer and it is false, because the composer's
  axes-less arm became a `raise`. Re-run the census; the *member moved, count didn't* is
  the sentence. → L-098
- **⭐⭐ A CARVE THAT *PROMOTES* A HELPER LEAVES ITS OLD NAME IN A DOC
  `code-block::` — the one prose surface that must COMPILE, and the one a
  symbol-keyed audit misses because the helper MOVED rather than retired.**
  `[M]` 6.3 promoted `_require_typed_composite` onto `FullField.require_member`;
  the brief listed the two retired VERBS and never the helper, while
  `loss_representation.rst` taught the dead name **3×, twice inside a
  `.. code-block:: python`** (`grep orpheus/` = **0**; the only tree hits assert
  its ABSENCE). ⟹ census the OLD name in `docs/` beside the briefed symbols, and
  treat a code-block as the page's highest-decay line. → L-099, L-089
- **⭐⭐ A BIT-IDENTITY AND ITS POSITIVE CONTROL ARE DIFFERENT EPISTEMIC OBJECTS
  — publish the identity, FLOOR the control.** `[M]` a commit draft and a gate
  docstring independently said the control moves the answer by *1.1–2.6*; both
  were **one draw** (`seed=101`). Over 40 seeds × 4 geometries `array_equal` is
  **40/40** everywhere (structural — once the rows are zeroed the two bodies are
  the same arithmetic) while the control spans **0.515–5.198**, so the band was
  never a band. ⭐ The GATE was fine (it asserts `delta >= 1e-3`, a floor); only
  the docstring's "MEASURED spread" published a draw. Two surfaces agreeing on a
  number is ONE surface until you check they do not share a seed. → L-099, L-071, L-064
- **⭐ ASK WHETHER WHAT RETIRED IS THE ROUTE OR ITS SPELLING — a two-route table
  whose route-2 VERB dies usually keeps its row.** `[M]` the retired
  whole-trace assignment's route survives as the masked ADDITIVE verb the G-S
  resolvent binds; past-tensing the row would have left a one-route table titled
  *two*. Re-pointing bought a strengthening: under G-S the routes are not
  alternatives but the two halves of ONE splitting. Sibling: a SECTION titled by
  a retired symbol keeps its LABEL and is re-titled onto the **invariant** (the
  core), never onto the successor — successors retire too. → L-099
- **⭐⭐ BEFORE DEFERRING TO A CONVENTION N NEIGHBOURS FOLLOW, READ THE PAGE'S OWN
  STATEMENT OF IT — the exception clause decides your case.** `[M]` 3 changelog
  entries said *"uncommitted at the time of writing"* with ~5 neighbours doing the
  same; the preamble says an entry lands with its hash and *"the only exception is
  an entry whose Where names an unmerged BRANCH"* — these named `main`. Stamped.
  "Everyone else does it" is not evidence about which side of the clause you are
  on. → L-099
- **⭐⭐ A DOCS PASS FOR ITEM N IS WHERE ITEM N−k's UN-SWEPT ROT SURFACES, and the trigger
  is PROXIMITY, not predicate.** `[M]` *"the refusal has no production witness yet … the
  only axis mint in `orpheus/` is `MaterialMesh.bulk_space`"* — both halves false, and
  neither by my item: `of_axes` has **7** AST CALL sites (a textual grep over-counts — one
  hit is a docstring), and `SNMesh.angular_trial_space` appends a `BasisKind.MODAL`
  `moment_axis` to an axis-built base whenever `LinearDiscontinuous.is_multi_moment`
  (`True`; `DiamondDifference` `False`). Found only because I went to align vocabulary with
  an adjacent sentence. ⚠ Repair the OCCUPANT, not the FIRING: "such a space exists in
  production" is measured, "the refusal fires" is a different census — say so in a clause
  rather than upgrading the claim. → L-097, L-075, L-072
- **⭐ A ✅ LANDED ROW IS NOT A TENSE FLIP WHEN THE ROW ALSO PREDICTED A MECHANISM.** A
  roadmap row promised the retirement would be *"a file-level move — its gates live in a
  separate test module"*; `[M]` `git diff -- tests/` shows nothing moved, the behavioural
  rows were **re-keyed in place** (same outer product, now via `apply_metric`). Publish the
  refutation: *"a gate that pins behaviour migrates with the behaviour, not with a file"* is
  the sentence a reader needs, and it exists only because the landing contradicted the plan.
  → L-097
- **⭐ STRIP LITERALS AND MATH BEFORE ANY MARKUP-PARITY CHECK.** Raw `**` count went even →
  **odd** and the culprit was `` ``orpheus/**/*.py`` `` — a `**` inside an inline literal.
  Sibling, re-confirmed: naive `\*\*(.+?)\*\*` + `re.S` "role inside bold" is USELESS
  (pairs one run's closing with the next's opening — **132** hits on a clean file); the
  gate that works is `\*\*:[a-z]+:``, `` \*\*`` ``/`` ``\*\* ``, and `` `{3,} ``. → L-097, L-094
- **⭐⭐ WHEN THE CARVE IS UNCOMMITTED, HEAD *IS* THE PRE-CARVE TREE — a before/after
  table costs 20 s.** `git worktree add /tmp/x HEAD --detach`, run the SAME probe on both.
  ⚠ strip the venv's editable `sys.meta_path` finder and PRINT `orpheus.__file__` as proof.
  ⟹ before trusting any relayed pre-carve `[M]`, ask whether HEAD is that tree. **And any
  cross-carve table owes a line naming WHICH statistic is comparable**: `[M]` my post-fix
  L=0 balance defect is **13.7× LARGER** (the two finalizes construct ψ differently), so
  only the RATIO down the budget compares — publishing the raw rows would have shipped a
  regression narrative inside a repair entry. The tell that you need that line: the CONTROL
  column moved. → L-095, L-050
- **⭐⭐ A CLASS LIST HANDED BY A BRIEF *AND* BY THE LANDING CODE'S OWN DOCSTRING CAN BE
  WRONG TOGETHER — CONSTRUCT EACH MEMBER AND PRINT THE PROPERTY.** `[M]` *"the digest-named
  leaf classes"* was right for 5 and wrong for 3 (`SphericalHarmonicSpace.from_L(2).name` is
  `'spherical_harmonic_space'`, a FAMILY tag, not a digest). ⭐ The tell was already in the
  brief: it ALSO said *"leave the metric-blind seam — that head is still metric-blind"*, which
  the list CONTRADICTS. ⟹ **check a briefed list against the brief's own carve-out; a list
  that contradicts its own exception IS the finding.** A `grep hashlib` finds the members that
  have it and is silent on the ones that do not — that half is the one that matters. → L-096
- **⭐ AN `__eq__`/IDENTITY CHANGE IS PUBLISHED AS THE ROWS THAT MOVED, and the "before"
  column is usually DERIVABLE — no pinned worktree.** `[M]` axis-built vs a hand-named twin
  `True`→**`False`** (a label stopped being a credential); `A*B` vs `of_axes(*A.axes,*B.axes)`
  `False`→**`True`** though the two derive DIFFERENT names (a name stopped being the identity).
  Publish the UNCHANGED rows beside them. The old law being a pure function of quantities you
  already measured (`name`, `shape`) makes the before-column `[R]`, stated. → L-096, L-095
- **⭐ WHEN SPHINX IS UNAVAILABLE, the substitute is a pre-edit-vs-post-edit docutils
  error-SET diff plus three gates over your ADDED LINES ONLY** (bold-abuts-literal /
  role-after-`**` / 3+ backticks · `:ref:` against the corpus label set · python xrefs
  IMPORT-resolved · `:doc:` against the filesystem · 79 columns). `[M]` 0/0 on 6 files, and the
  added-lines gate caught **2** real `**``literal``**` nestings no build would have flagged.
  ⚠ And check a count against the LIST beside it: I wrote *"**four** classes — A, B, C and the
  two D"* (four = FACTORIES, five = classes) and caught it only on the final read. → L-096
- **⭐⭐ A CARVE'S OWN DOCSTRING EDITS ARE UNVERIFIED CLAIMS — AST-census the callee it says
  still has a consumer.** `[M]` the carve wrote *"its production consumer is the G-S
  resolvent"* onto a verb with **0 Call sites and 0 attribute refs** tree-wide, four lines
  above its own ⚠ paragraph naming the DIFFERENT verb that really is bound. A retirement
  answers *"who called what I removed?"* and nobody asks the dual. → L-095
- **⭐⭐ THE HTML SLICE CATCHES A ROLE SEVERAL LINES INSIDE AN OPEN `**bold**` RUN; an
  ADJACENCY regex does not, and `-W` is EXIT=0 either way.** `[M]` mine rendered
  `:doc:`…`` as literal text on a clean build. ⟹ source-side, pair `\*\*(.+?)\*\*` with
  `re.S` over your ADDED lines and look for `:[a-z:]+:\`` INSIDE; then run the HTML gate
  (strip tags → unescape → slice by section title → `re.findall(r"\`+")`, **0** is the
  gate). Discriminate mine-vs-pre by testing each hit line against `git show HEAD:<file>`.
  → L-095, L-074
- **⭐⭐ A MODULE UNDER CONCURRENT EDIT: publish NOTHING you count in it.** `[M]` the gate
  module went **45 → 86 rows** and **7 → 8 arms** while I wrote. Point at the artefact that
  RE-MEASURES it (`matrix.rst` regenerates row counts); for a count you must state, date it
  and name the census (`ast.AnnAssign` over the registry, not the module's own stale table),
  and check whether a pre-carve claim used a different denominator. → L-095, L-074
- **⭐⭐ A DOC'S STATED **REASON** for keeping dead-looking code can be false while its
  membership is right — and that is the stronger finding.** `[M]` *"kept for the
  EigenvalueSolver Protocol surface"*: the Protocol declares **5** members
  (`sorted(Proto.__protocol_attrs__)`, briefed as 3) and only ONE of the four listed
  delegators is among them; the real reason three survived is the defect itself. → L-095
- **⭐ `catches("ERR-NNN")` makes `-W` FAIL, and `grep WARNING` UNDER-counts it** — the
  nexus per-marker lines carry no `WARNING:` prefix (`[M]` grep read 1, the log had 5).
  Read the whole log. ⭐ And my role probe's fallback ladder is now FOUR: `hasattr` →
  `dataclasses.fields` → `self.x=` **across `cls.__mro__`** → construct; plus strip the
  `py:method:` prefix form and treat a leading `!` as a deliberately-suppressed xref
  (8 + 6 false dead). → L-095, L-093, L-076

- **⛔⛔ A CORPUS PASS FOR A CARVE THAT LANDED WITHOUT ONE IS A BUILD REPAIR FIRST —
  measure the baseline before believing the brief's framing.** `[M]` #426 step 3 opened at
  **EXIT=1, 13 WARNINGs**, all `[nexus.directive]`: a `.. implements::` `:by:` resolves
  against the GRAPH, so a carve's renames break DECLARATIONS loudly and `:class:` refs
  silently. "Count unchanged" would have licensed shipping 13 errors ⟹ when the red is the
  carve's own, the gate becomes **EXIT=0**, stated in §0 of the report. → L-094
- **⭐⭐ AN INHERITED MEMBER REF IS NOT DEAD — `dead_references` RESCUES it (`[M]` 66 of
  75), and that decides the sweep's width.** After a body moves onto a shared CORE: a
  RENAMED/RETIRED member is dead and must re-point; a member that merely MOVED resolves, so
  re-point only where the SENTENCE claims it is defined there; a `:by:` target must name
  the DEFINING node (inheritance never rescues a graph edge). `[M]` 33 refs, 20 "moved", 5
  actually dead — re-pointing all 33 is churn AND loses the chapter's own vocabulary. → L-094
- **⛔⛔ THE ELEGANCE PASS LANDS INSIDE YOUR TASK AND MOVES THE THING YOU JUST WROTE
  ABOUT.** `[M]` 25 `orpheus/` files + a commit mid-task: `scattering_order` →
  `legendre_order` (the ONE real dead role, which did not exist when I started),
  `from_field` → `on_basis`, and `from_solver_data` moving to the core while `channel`
  became a **ClassVar, not a classmethod** — so one sentence of mine went true → false →
  false-again, at EXIT=0 every build. Run `git log --oneline -3` and
  `git status --porcelain -- orpheus/` **at the END**: a clean `orpheus/` late means the
  pass COMMITTED, and its message is the diff of your premises. ⭐ The correction was
  sharper prose ("a role is two class constants and no code"). → L-094, L-089
- **⭐⭐ RE-RUN a published census: its COUNT can be right and a MEMBER invented.** `[M]` a
  `9 Sig2[0] sites` census re-ran to **7, all correct** — and one of its seven enumerated
  members, a `if sig2[0].nnz > 0` guard, **does not exist at HEAD or either parent**. A
  re-read cannot find that. ⭐ Sibling: a site can change COLUMN without changing LINE (the
  dense cache went model → reaction-rate), so the repair is a `fate` COLUMN on the old
  table, never a deletion. → L-094
- **⭐⭐ A `verifies()` MARKER DECIDES WHICH BODY A LABEL KEEPS — read the TEST BODY.** When
  generalising a labelled equation the natural move (broaden the old label, mint a new one
  for the special case) is BACKWARDS if the old label is a marker target: `[M]`
  `n2n-source`'s claiming test asserts `_add_n2n_source`, a **P0** body. Order: grep
  `tests/` for the label → read the body → the existing label keeps what its marker
  asserts → mint the NEW label for the generalisation → state the RANKING in prose. → L-094
- **⚠ An `.. error-entry::` has NO `id` — `:ref:`ERR-NNN <err-nnn>`` is silent death.**
  `[M]` the directive emits `container` + `rubric` only; cross-doc dangling `:ref:` renders
  plain-text unwarned. Cite plain `ERR-NNN` + `:ref:`the L0 error catalogue
  <theory-verification-error-catalog>``. ⭐ Sibling `-W` DOES catch: a `:ref:` to a label on
  an `.. important::`/`.. warning::` needs explicit text, cross-doc as well as intra-doc.
  → L-094, L-092
- **⭐⭐ A LADDER MEASURED PRE-CARVE IS A DIFFERENT TREE — check BOTH legs of the ratio.**
  `[M]` step 1's elastic ladder could not be quoted post-step-2: the baseline moved AND the
  other channel's moments started entering. Re-measured mine (7 arms, 45 s). ⭐ And the row
  that is NOT evidence: an `ℓ ≤ 6` arm equal to `ℓ ≤ 2` **to the bit** is the SOLVE's order
  (`Λ` has 3 blocks), not convergence — say so at the table. → L-094
- **⭐ `**:math:`` with NO separator is the one nesting that breaks** — `[M]` the corpus's
  `**word** :math:`x`` idiom is fine at ~60 sites; a role opening *immediately* after `**`
  renders the backslash-eaten role as text on a 0-warning build (4 pre-existing sites, 2
  pages → **0** backticks). Discriminate MINE-vs-pre by testing each hit line against
  `git show <carve-hash>:<file>`. → L-094, L-074
- **⭐⭐ A TRUNCATION THAT MOVES TIERS needs a TIER TABLE, not a tense flip — sort every
  site by TIER, never by tense.** *"ORPHEUS models (n,2n) emission as isotropic"* stayed
  TRUE while *"…because the data layer truncates"* went FALSE, in the same sentence. `[M]`
  by AST, 9 `Sig2[0]` reads = **2 model + 7 ℓ=0-BY-PHYSICS** (a reaction rate IS the P0 row
  sum; CP/MoC/MC are isotropic by construction) ⟹ publish a two-column data/operator table
  AND warn that the 7 must not be "fixed" with it. ⭐ The census's blind spot IS the third
  model site: the frame's `for_space(interior, 0)` is not a `Sig2[0]` read, so the predicate
  structurally cannot return it — say so, or a reader greps and misses it. → L-093
- **⭐⭐ A regenerated LOCAL CACHE is a doc surface with NO SYMBOL in it — `ls -l` is the only
  instrument.** A store-size table and *"processes all **12** `.GXS` files"* were both
  falsified by a serialization change; the `.h5` store is untracked, so `git status`, `-W`,
  the xref gate and `dead_references` are ALL blind. `[M]` 13 tapes, 438.5 MB, ×1.98–2.38,
  7–8 min. ⭐ Sibling: a format-VERSION bump catches a LAYOUT change and is **structurally
  blind to a VALUES change** (a hand-set constant) — the two stale-store kinds need OPPOSITE
  prose in one paragraph, and the second is why the old warning survives. → L-093
- **⭐⭐ A relayed physical EXPLANATION can fail while its measurement stands.** *"99.9 % is
  the reflector's — U-235's MT=16 is 13× weaker"*: effect exact, gloss refuted (`[M]`
  U-235's peak (n,2n) XS is **larger**, ratio 0.69). Replace with a measured control + a
  checkable structural fact (50 vs 22 live incident groups). ⚠ And a control must be the
  **same ARM** as its subject — I compared an ℓ≤2 control to an ℓ=1 row (1.50 apart) inside
  a "within 2·10⁻⁵" claim. ⭐ Free control: re-derive every derived column from the recorded
  raw values (24 of 24 reproduced) — that is what licenses publishing a table as SSOT. And
  when two surfaces disagree in the last digit (−51.1 vs −51.2), the artefact says −51.15:
  publish ITS precision, never adjudicate between roundings. → L-093
- **⭐ "Same nnz across ℓ" is an ISOTOPE property — one isotope taught me the wrong
  universal.** `[M]` U-235 6067/6067/5834/5334/3165/2773/1887 vs Be-9 **8195 at all 7**;
  sparsity is the TAPE's (genuine exact zeros), not the ingest's, so a right mechanism clause
  ("a row diagonal cannot change sparsity") carried a false conclusion. → L-093
- **⛔ A paragraph QUOTING a code EXPRESSION is higher-decay than one naming a HELPER** — an
  elegance pass single-sourced two inline expressions into `_n_orders`/`_order_key` mid-task
  and falsified a paragraph I had already built clean at EXIT=0. Name the helper. (L-089's
  loop, sharpened.) ⭐ And a doc `.. code-block::` is the one staleness you can PROVE: run the
  old spelling (`ValueError … got shape (1, 2)`) and publish the receipt. → L-093, L-089
- **⭐ A role probe needs a THIRD fallback after `hasattr` and `dataclasses.fields`:
  CONSTRUCT the object.** My single DEAD of 258 was `SNMesh.axes` — not a dataclass, assigned
  in `__init__`; live on the instance. (L-053(c), which my L-076 fallback does not cover.)
  → L-093
- **⭐⭐ A SECTION HEADER can be a class-level falsehood, and the page usually already
  carries the true account — split by FACT, never re-word.** A §*"Reactions Not Included:
  (n,2n), (n,3n), (n,4n)"* asserted one predicate over three MTs; `[M]` true of 17/37,
  false of 16, and the correct MT=16 account sat **200 lines above** on the same page (vv
  #21's aggravator at page scale). Repair: two H2 sections each opening by naming what the
  OTHER covers; every 17/37-only sentence survives with its quantifier narrowed; the
  deferred sketch's DONE steps get RE-AIMED (*"not the open item it used to be — 17/37
  would reuse that machinery at ν = 3/4"*), which turns a stale to-do into the section's
  strongest argument. → L-092
- **⭐⭐ Reproduce a census's REFERENCE; expect its per-row RESIDUALS not to reproduce.**
  `[M]` the closed-form `k_inf` came back BIT-IDENTICAL (and all three datum identities),
  while diffusion/SN-fwd/SN-adjoint residuals differed by orders (`2.7e-16 / 3.0e-10 /
  1.6e-11` vs a memo's `8.1e-16 / 5.2e-14 / 1.0e-13`) — a residual is a property of MESH ×
  QUADRATURE × TOLERANCE, and the memo stated none per row. ⟹ publish the reference with
  its full inputs (so the page regenerates it), YOUR rows with the configuration IN the
  table's first column, and the relayed sweep as a BOUND under a `.. note::` saying the
  digits move. ⭐ The one relayed number safe verbatim is the STOCHASTIC one — a σ makes it
  self-describing (`1.655710 ± 0.001525` = **1.63 σ** ⟹ unbiased). → L-092, L-057, L-050
- **⭐⭐ A "Limitations and Future Work" table is a PRESENT-TENSE claim surface — and the
  page that documents the FIX is the likeliest home of the stale row.** `[M]`
  `monte_carlo.rst` listed *"Solver ignores Sig2 (n,2n)"* as a limitation 650 lines below
  its own account of the fix. Found by grepping the CLAIM (`does not extract|1-in-1-out|
  ignores.*sig2`), never the section under edit. Repair: retire the ROW (keep the tracking
  ID so it resolves, mark `⛔ RESOLVED (#NN)`, point at the record) — do not delete it.
  ⭐ Sibling: a catalogue TITLE is a DEFECT NAME, not a state — ERR-023's *"MC solver
  silently ignores Sig2"* reads present-tense to every quoter; ship a `.. warning::`
  saying fixed / by which issue / which catcher (and that it is `slow`-marked, so the
  canonical gate never runs it). → L-092
- **⭐⭐ A pre-existing `**``literal``**` NESTING bug travels forward when you rewrite a
  section verbatim, and only the RENDERED HTML sees it.** I carried three forward from the
  section I was replacing; `-W` silent, L-074's HTML gate read **6 visible backtick runs**.
  Grep `\*\*``` on the source AND run the HTML gate on every page you touch. ⭐ And its
  cheap sibling: a `:ref:` to a label on a `.. warning::` (not a section title) MUST carry
  explicit text — bare `:ref:` is `ref.ref` *"A title or caption not found"*, a real `-W`
  failure (twice in one session). → L-092, L-074
- **⭐⭐ Naming a nuclide / fixture / member in a NEW section obliges you to check that
  page's own ROSTER.** `[M]` my split cited Be-9; the page's nuclide table listed **12**
  and omitted it, while the converter globs `*.GXS` (**13**) and the page's own truncation
  warning already says "the 13 shipped files". ⚠ The near-miss: a sibling page's *"12
  isotopes"* is CORRECT — it counts a PWR-cell mixture, not the library. **Read what a
  number COUNTS before fixing it.** → L-092, L-091
- **⭐ The docstring fix a brief names usually has a SECOND false clause one paragraph
  down.** Briefed one `A_loss = L+C-S-B`; `[M]` the same docstring spelled the daggered
  triple and the loss dagger the same wrong way twice more, in the sentence explaining the
  equation. ⚠ And the other hits in the file were a DIFFERENT issue's (a tracked
  pedagogical spelling with its own machine-header key) — resolve each hit by whether it
  is the adjoint's claim or the chapter's convention. → L-092
- **⭐ Before paraphrasing a RULING, find where the corpus quotes it VERBATIM and read what
  has since been said ABOUT it.** `adjoint.rst` carries the ruling word-for-word plus a
  dated note retiring its *"in principle"* hedge; a bare paraphrase would have re-imported
  the hedge onto the data page. Say it is a paraphrase, point at the verbatim copy, carry
  the strengthening. → L-092, L-081
- **⭐⭐ A `[M]` COUNT with no MEMBER SET stated is not reproducible — and the spread
  over plausible sets is the finding.** A production docstring's *"197 such triples
  over the expressible members"* would not reproduce: `[M]` on a natural 21-member
  group set (12 finite ⟹ 21·12·21 = **5292**, the review's OWN denominator) it is
  **217**; swapping one member reads 181 or 255. The denominator matched and the
  numerator did not — the signature of an unstated set. Its neighbour in the same
  docstring (*"441 of 441 ordered pairs"*) reproduced EXACTLY on the same set, so one
  sentence held one reproducible number and one not. ⟹ publish YOUR count WITH the
  members enumerated, and quote the WITNESS (`O(2)_x ⊆ O_h·SO(2)_x` while neither
  factor contains it) — a witness is a theorem, a count is a fixture. → L-091
- **⭐⭐ A "shipped" denominator can contain a CONSTRUCTED member — enumerate candidate
  sets until one reproduces.** *"4 geometries × 7 shipped rules ⟹ 17 refusals, 14
  arrow / 3 coverage / 0 both"* reproduces to the row for exactly ONE 7-rule set whose
  seventh member no factory ships (a σ_z fold built by `.quotient(Mirror("z"))`). `[M]`
  the five shipped factories alone give 8 arrow / 3 coverage of 20. The set is RIGHT —
  it is the only input separating two geometries' Γ — and the WORD is wrong. The search
  is the evidence; publish the enumeration. → L-091
- **⭐⭐ A test docstring's "this leg is INERT in production" is a NEGATIVE claim, and an
  in-process neutering measures it in 20 s.** `[M]` a gate's honest-scope note said
  *"nothing registered is a fold, and the 1-D rule is refused by stage 2 first"* — both
  false: the registered `GaussLegendre1D` IS a fold (`S^2/O(2)_x`) and the shipped log
  shows stage 0's coverage clause refusing it. Monkeypatching the predicate to `True`
  (restored in `finally`, identity-verified) moved the rejection to stage 2 and left the
  CHOSEN rule unchanged ⟹ the honest scope is *"changes the REASON, not the
  selection"*, which is strictly stronger. Report the docstring; you do not edit
  `tests/`. → L-091
- **⭐ Re-measure a SIGN-COUNT: `sign(0)` is a third class.** I relayed *"all 8 octants
  populated, 4 nodes each"* for a product rule; `[M]` **16 of its 32 nodes lie ON a
  coordinate plane**, the four strictly-signed quadrants carry 4 and the eight
  strictly-signed octants carry **2**. Say "strictly-signed" and give the on-plane
  count. → L-091
- **⭐⭐ A `[M]` whose DENOMINATOR is a COMPUTED SET has a shelf life the FINDING does
  not — write what computes it.** `[M]` two sites read *"the reduction agrees on **150 of
  150** (sphere rule × candidate group) rows"*; re-measured it is **144 of 144**, every
  row still identical. The denominator is `sum(len(candidate_groups(r)))`, an output of
  the machinery the campaign keeps re-deriving (it moved twice in two days). Repair: not
  a tombstone but *"the finding is unchanged and only the DENOMINATOR moved, because it
  is the size of a candidate set"*. → L-090
- **⭐⭐ A plan's list of INTENDED behaviour moves is a sample — run the whole shipped
  roster.** The plan named two; `[M]` against a pinned `git archive HEAD` tree there are
  **three** (`folded_product(4,6)`'s walk `{D_1h,σ_x}` → `{D_2h}`, same mechanism, a
  strengthening). ⭐ And separate the two questions: the invariance PREDICATE moved **0 of
  330** (11 rules × a FIXED 30-spelling list) while the WALK moved on **4 of 11** — what
  changed is which questions the walk thinks to ask, not what any answer is. → L-090,
  L-074
- **⭐ A rejected-design / refusal note can have ONE of its N costs EXPIRE.** One of four
  costs listed against a rejected design (*"needs a second function-scope import"*) went
  to zero when the carve reversed the import direction. Date that clause with a `⛔`, keep
  the note, and say the ruling does not depend on it (*"the three surviving costs are each
  sufficient on their own"*). → L-090, L-074
- **⛔⛔ ON A LIVE BRANCH THE RE-READ IS A LOOP, NOT A PRE-FLIGHT — re-read the module's
  public surface AFTER EVERY BUILD.** Dispatched at 4 modified `orpheus/` files, I finished
  at 20: the R4 gates landed mid-task (so a briefed "no test pins X" was `[M]` **2 hits**,
  and I could NAME the gates instead of promising them), and an elegance pass re-shaped
  `__post_init__` from THREE clauses to **FOUR**, turned a single generic POINT into a
  probe SET with a MAXIMUM, and flipped `lift_codomain` from `compare=False` to COMPARED —
  falsifying a sentence I had already written and BUILT clean. ⟹ the highest-decay class is
  a sentence naming a field's `compare`/`repr`/default, a guard's clause COUNT, or a
  helper's SIGNATURE: `-W` is silent on all three (EXIT=0 with the false compare claim in
  it). Read `dataclasses.fields`, `dir(cls)` and the `__post_init__` BODY, every time.
  ⭐ And the tell that caught it: a census that DISAGREES with the brief is evidence about
  the TREE, not about the census. → L-089
- **⭐⭐ Before minting a citation to EQUATION N, grep the corpus for what it already says
  about EQUATION N.** A brief's characterisation can be RIGHT and INCOMPLETE — the harder
  case. `[M]` BMC Eq. (52) states TWO things; the corpus already carries a MEASURED refutation
  of the second (imposing its partition violates P3, `0/4→4/8→12/16→28/32`, NaN at `n_φ ≥ 16`,
  mismatch widening `[0.586,1.414]→[0.077,1.568]`) and a page already cites Eq. 52 FOR that
  refutation. A bare citation would have imported the refuted half at 3 sites at
  `confidence = 1.0`. Cite the half your sentence claims, and point at the refutation.
  (L-060's census-before-repair, applied to an equation NUMBER.) → L-070
- **⭐ A fictitious citation is a RE-POINT when its equation numbers are right — and the
  field-by-field origin table is worth more than the fix**, because a record whose every
  field traces to a *different real* publication is what survives audits. ⭐ Cheap
  self-refutation: **(author, year, volume) is over-determined — a journal volume pins its
  year**, so "vol. 35" (=2008) refutes "2009" before any lookup. → L-070
- **⭐ "Use the bib key" is a RECORD instruction, not a RENDERING one.** `[M]` the target page
  had a plain-text `References` block and ZERO `:cite:`; minting its only one puts two
  citation systems on one page. Plain-text inline + an entry in the page's OWN References
  block names the same record and keeps the convention. → L-070, L-006
- **⚠ When the cited claim is a CONVENTION, the fix is a RETRACTION — there is no equation to
  re-point to.** A comment credited two sources for an axis assignment BOTH use the opposite
  of. Delete the attribution, say why, and leave the convention prose untouched — naming the
  un-performed audit in the tombstone is what stops the next reader "fixing" the axes. → L-070
- **⭐⭐ A brief's "these are CORRECT — do not touch" LIST is a classification, and it can be
  wrong; the brief's governing RULE outranks its list.** `[M]` "Bailey … 2009" is TWO papers by
  the same four authors — the retracted *JCP 227* **diffusion**/polyhedral entry and an
  *Ann. Nucl. Energy 35* **PWLD-transport** entry — so an author-year string is invisible to
  every grep. A protect-listed site (`transport/spatial/scheme.py:42`) was the RETRACTED entry
  verbatim, cited for "Eq. 50 (dome recursion) … feed the curvilinear cell update". Resolve
  every citation site by its **journal + title**, and expect the brief's census to be a sample
  (26 candidate lines vs ~12 briefed). → L-070
- **⭐⭐ For a "family X does not use Y" claim, the evidence is a CAPABILITY THAT EXISTS AND
  DECLINES Y — an absence cannot separate "never" from "not yet", and "not yet" LICENSES WORK.**
  `[M]` MoC ray-traces concentric annuli on a cylindrical mesh, CP ships a real sphere, MC a
  real cylinder — three curvilinear capabilities, zero α. ⚠ When refuting a claim about {A,B},
  the counter-examples must come from {A,B}: my draft used CP+MC for a claim about MoC+CP, and
  MoC's own annuli were the strongest one I had not looked for. → L-070
- **⭐ TWO AGREEING SOURCES CAN BOTH BE WRONG — corroboration is not independence.** A brief said
  a gate held *"dated 2026-08-21"* and an in-tree comment independently agreed; `[M]`
  `git log --date=short` puts every commit of that step on **2026-08-20** (a FUTURE date on both
  surfaces — one mis-dating copied forward). Git is the arbiter for dates and merge status; two
  prose surfaces agreeing is one surface. → L-064
- **⭐ Publish the CLOSURE ARGUMENT, not the universal it implies.** Instead of *"every harmonic
  space is legacy"*, write *"`of_axes` is the only ROOT producer of an `axes` record (`*` and
  `dual` merely THREAD one), therefore …"* — the derivation stays true as the tree grows, and the
  grep that establishes it is free reconnaissance (mine surfaced `mm.axes` = the GEOMETRIC tuple
  vs `mm.bulk_space.axes` = the SPACE-FACTOR tuple: one attribute NAME, one object, neither
  derived from the other — a publishable gotcha). → L-064
- **Read the live `def`/body before citing any convention, shape, signature or design decision.**
  Seen: a docstring lying about an index convention and a return layout; a verdict memo recording
  the RECOMMENDATION while the code shipped the alternative; a brief naming args the live Protocol
  never takes. → L-001
- **⭐⭐ A SPEC is its TABLE — re-derive the brief's counts from the artefact in one command.** A
  spec file's headline read "21 of 40 declarable, 19 NONE" over a table that is `[M]` **32/8**;
  the brief inherited "19" verbatim (its own kind breakdown summed to 8). A headline is a summary
  and summaries rot while the table stays right. → L-060
- **⭐⭐ A brief's "sharpest observation" is a HYPOTHESIS with a computable confusion matrix.** A
  briefed *"the page already labels its own two classes"* keyword-tell died on four greps: the
  word *identity* sits in 5 of 6 un-implementable rationales AND **11 of 22** implementable ones;
  *"not a solver claim"* points the WRONG way (1 of 6 vs 5 of 22); a third of the page carries no
  rationale at all. Publish the measured split AND the refutation, or the next reader re-derives
  the heuristic and ships it. The surviving distinction was real: an identity between
  **quantities** has no carrier, an identity between **types** is a claim about a class
  declaration. → L-060
- **⭐⭐ Every "each / every / all" in prose YOU publish is a universal you can count in one
  command — count it.** I broke this twice in one new section and caught both by re-measuring:
  "…and X, Y, Z with three each" enumerated 3 of 5 (and read as all 15 multi-implementer rows);
  "where every ``solve`` in the tree matches the label" was `[M]` **5 of 60**. The measured
  sentence was strictly better both times. (plan-authoring §2, applied to the corpus.) → L-060
- **⭐ Verify a SUCCESSOR claim against the RETIRING COMMIT'S BODY, not against the successor's
  existence.** A live class with the right shape is not evidence the dead one became it. `[M]` I
  wrote that three symbols "were absorbed into the operator algebra"; the commit bodies say ONE was
  re-layered (`SNSolver.L` → two leaves) and the other two were **retired outright** — one "became
  orphan in production", the other "without a remaining call site". One paragraph, two fates, and
  "absorbed" was false for two of three. `git log --diff-filter=D` / `-S <symbol>` then read `%b`.
  → L-062
- **A retirement SHIM's docstring is frozen at the commit that minted it — verify against the
  CANONICAL form it re-exports.** A shim called a cross-class dunder "retired"; the canonical
  modules had since RE-PERMITTED it, so the brief would have past-tensed an accurate section.
  → L-001
- **A brief's discriminator is a heuristic, not a per-ref rule — one phrase can be TRUE at one site
  and FALSE at another.** Resolve EACH ref by the exact symbol's live signature / the site's live
  fixture, never by the phrase. → L-001
- **A brief's RATIONALE can be wrong while its conclusion is right, and a MECHANICAL vocabulary
  swap can restate a FALSE claim in fresh, authoritative words.** Re-verify the CLAIM before
  re-spelling it — grep the live class for the property the sentence asserts; `-W` catches neither.
  → L-001
- **Reproduce every number you cite, and sanity-check its neighbours while the harness is open** —
  one worked example's intermediates contradicted its own result three lines later. → L-001
- **A COMPOSITE's measured identity cannot certify its FACTORS — verify each factor against its own
  live method.** A page's `Rᵀ = (cos w/norm)·Σφ` was `[M]` bit-exact while BOTH per-factor formulas
  it was built from were wrong (the shipped split puts `1/norm` in `B`, not `C`, because `C` makes a
  *current* and `B` an *intensity*); every measurement on the page reads `0.0` either way, so no
  gate could see it. Re-verify the design-probe's description too. → L-049
- **Ask WHICH SIDE of a carve a brief's "measured" number came from — and re-measure BOTH.** A
  briefed "post-carve: agrees to solver tolerance, `1.998e-13`, `array_equal=False`" was a PRE-carve
  number (two structurally different deliveries reaching one fixed point); post-carve the two
  channels are the same float program and collapse to **`0.0`**, `array_equal=True`. Publishing the
  brief would have inverted the carve's headline AND justified an `rtol` gate blind to the very
  defect (`2.9e-14` sails through `10×inner_tol`). A pinned pre-carve worktree makes both sides
  cheap and turns one number into a before/after table. ⚠ the venv's **editable install hooks
  `sys.meta_path`, which OUTRANKS `sys.path`** — `PYTHONPATH=<worktree>` silently loads the MAIN
  tree; strip the editable finder and PRINT `orpheus.__file__` as proof. → L-050
- **Never accept a fixed-decimal printout as evidence of bit-exactness.** "`2.500000000000` at 12 dp"
  cannot resolve `8e-15` at 2.5. Measured: the converged inflow trace is exact on SI (the sweep
  *writes* the seed) and NOT on Krylov (GMRES *solves* the trace rows — 1–23 ULP at `tol=1e-13`,
  27 580 ULP at `1e-10`, i.e. the iteration residual, not FP noise). An exactness claim true on one
  inner solver is a **per-leg** gate (`array_equal` on the exact leg, `rtol=SAFETY×inner_tol` on the
  iterative one) — say "do not relax the exact leg to match". Assert `x == v` or print
  `float(x).hex()`. And run your OWN probes without `-O`: a bare `assert` in my widened check was
  stripped (vv Mode 8, in my own instrument). → L-050
- **⭐⭐ Before you PUBLISH a number a brief hands you, grep `tests/` for a module NAMED after
  the phenomenon — and read its docstring, not its `assert` lines.** A `scratch/` memo is by
  construction OLDER than the tests it motivated. A briefed "we just measured `min ψ̂ ≈ −77`
  under the shipped convention" was reproducible to the digit AND its framing was already
  refuted by a 19-row `foundation` module committed the SAME DAY: on the production
  (marched-seed) path ψ̂ is strictly POSITIVE (`+0.134/+0.129/+0.129`, 0.88–0.98 × `min ψ`);
  `−77` is an INCONSISTENT-seed statement. I had drafted "⚠ coverage gap: no ψ̂ gate on either
  arm" — false. My line-based grep missed the module because its evidence lives in the
  docstring and in `pytest.fail` messages, not in `assert` lines (vv #21). → L-055
- **⭐⭐ Reproduce the witness — the reproduction can REFUTE the GATE'S OWN PROSE while
  STRENGTHENING its claim.** A cone-violation gate's docstring said its two legs differ in
  *"ONE parameter … half the optical cell size"*; `[M]` both legs have `Δx·Σ_t = 100`
  **identically** (`nx=2,width=20` vs `nx=4,width=40`). The argument is stronger than its
  prose — holding cell size fixed kills the rival explanation outright — so publish the
  CORRECT framing, never weaken the claim, and REPORT the docstring (you don't edit `tests/`).
  ⭐ Then run the two scans nobody asked for: a cell-SIZE scan reproduced the textbook DD
  positivity limit exactly (`Δx·Σ_t = 1` in K, `= 2` already out) and a cell-COUNT scan showed
  the benign row is the only one — ~90 s turned one frozen constant into a mechanism. → L-063
- **⭐⭐ "Make illegal states unrepresentable" is TWO-sided, and half 2 is the one skipped.**
  Mint the invariant iff (1) every admitted value is legal AND (2) every legal value is
  admitted. Half 2 is a claim about the PRODUCERS, not the concept; when it fails the
  invariant does not prevent a bug, it **refuses correct output**. Quote this into any page
  invoking the pattern. → L-063
- **⭐ When a brief offers a BINARY verdict and the tree supports neither pole, publish the
  third.** "Scoped to the scalar flux (fine)" vs "general (falsified)" both missed it: the
  claim was general-in-wording, sphere-in-evidence, and substantively true on the cylinder's
  production path *as a characterisation*. Only the word "never" was false. Scope the heading,
  publish the seed taxonomy with both measured tables, keep the original conclusion standing
  where it survives, and point at the owning gate. → L-055
- **⭐⭐ A DESIGN PROBE goes stale against the repair it motivated, SILENTLY — it still
  runs and still prints plausible numbers.** `[M]` the plan's probe read
  `frame.test_space.inner_product_weights` as "the stored metric"; post-repair that IS
  the repaired metric, so the row labelled *stored* now prints `1.000` and the headline
  `118.7` is unreproducible from the file the plan cites. ⟹ never cite a pre-repair probe
  path as a post-repair page's reproducer — publish the **CONSTRUCTION** (build the object,
  name the seed, name the five attributes the residuals come off) so the table regenerates
  from the page; re-measure with your OWN probe; report the staleness upward. → L-065
- **⭐⭐ A SEED-DEPENDENT number is published as its BOUND, never its value — then find the
  exact per-mode parent behind it.** `[M]` plan `118.7` vs mine `81.4`, both correct: the
  Parseval ratio is a moment-energy-weighted average of the per-ℓ factors. Publish the
  draw-independent claim (*lies between the extreme factors PRESENT AT THAT L —
  `[17.5,157.9]` at L=1, `[6.3,157.9]` at L=2 — so never 1*), and note a bound is a
  universal (plan-§2: I first wrote the L=2 range over a sentence covering both). ⭐ Then
  look one level down: the RATIO OF THE TWO ADJOINTS on a single-ℓ unit input is exactly
  `(4π/(2ℓ+1))²`, `[M]` to `≤2.8e-16` — seed-free and strictly more useful. → L-065
- **⭐⭐ Decompose a float-agreement claim into its THREE quantities — only one is a property
  of the FIXTURE.** `|Δ|` is stable and is the number to publish; a **bit-equal FRACTION** is a
  property of the draw (publish the band); a **ULP gap** is worse than draw-dependent and is the
  one people freeze because it sounds precise. `[M]` same fixture, 200 seeds: `max|Δ|` =
  `1.776e-15` (reproduces a memo exactly), fraction `46.21–51.42 %` (the memo's `59 %` is
  OUTSIDE it), max ULP `113 – 91 839` (the memo's `204` is the floor) — because ULP explodes
  where the two terms nearly cancel while `|Δ|` stays at round-off. Always seek the
  **draw-free structural cause** instead: `τ` bitwise `½` ⟹ `1/τ=2.0` and `(1−τ)/τ=1.0` are
  exact ⟹ 100 % agreement, and `[M]` only `2 of 12` ordinates qualify (the rule carries SIX
  distinct float64 τ, three nominal values each as a 1-ULP pair). ⚠ Such a figure gets COPIED:
  mine sat in 2 production docstrings + 2 test docstrings. → L-071
- **⭐⭐ "BIT-EXACT" IS USUALLY A PROPERTY OF THE DRAW — publish a BOUND over ≥200 seeds, with
  the norm written out.** A brief and two gate docstrings said `R∘E = id` is `[M]` bit-exact;
  `[M]` on the gate's OWN fixture `np.array_equal` fails on **844 of 2000** seeds (~1 ULP), and
  the idempotence row on 57 of 200 — both gates are green only because their hard-coded seeds
  land in the exact set. On the SHIPPED SN carrier it is 200/200 (Σw = 2 exactly AND symmetric
  GL weights re-associate). ⭐ The DUAL, same page: a *tightness* row (two spellings of the
  same reduction, same order) IS robustly bit-exact, 200/200 — so say which kind you have,
  because only the measurement separates construction-exact from draw-exact. Report a
  seed-fragile gate upward; you do not edit `tests/`. → L-067
- **⭐⭐ When a memo states an ORDER of accuracy, EXPAND THE SERIES — "monotone and positive"
  can still be INCONSISTENT.** A memo's lumped-LD member was published to me as *"genuinely
  monotone at the cost of dropping to **first** order"*, transmission `2/((1+τ)(2+τ))`. The
  transmission reproduces exactly and the order label is false: `a'(0) = −3/2`, so over a fixed
  thickness it converges cleanly (10→10⁴ cells: `0.2367 → 0.2231`) **to `e^{−3/2}`, not
  `e^{−1}`** — vv #5's correct-rate-to-the-wrong-limit, and both properties the memo DID check
  (sign-preservation, `A⁻¹ ≥ 0`) are true of it. Consistency is a THIRD property neither test
  sees. ⭐ The correction pays: solving `a'(0) = −1` gave `ν = 1−λ` (a ONE-parameter family, not
  two) and a genuinely monotone consistent member `(0,1)`, `a = 1/(1+τ_opt/2)²`. One
  `sp.series(a - exp(-t))`. → L-069
- **⭐ Read the CLASS DOCSTRING of the object your new chapter theorises about — the code may
  already state your theorem.** A carve landing mid-task made my tensor-product factorization
  the code's own words. Two moves: say the chapter is the theory HOME for a structure the code
  asserts (not a twin), and adopt the code's exact spelling (`R_spatial ⊗ A_angular`) — internal
  consistency outranks brevity. → L-069, L-051
- **⭐⭐ A COINCIDENCE claim ("X is bit-identical to Y on 8 of 8 fixtures") needs its FAMILY,
  and it is usually false where it matters.** `[M]` `frame.discrete_gram[0,0] == weights.sum()`
  holds at n ∈ {2,4,5,6} and FAILS at 16 and 64 on BOTH `leggauss` and `linspace` weights (an
  einsum reduction vs a pairwise one) — and on the shipped SN quadratures it fails at **GL8**,
  where `from_isotropic` differs from `Q/Σw` by 2.0e-16 in production. Publish the ladder, and
  say which side is exact BY CONSTRUCTION (the Gram) so a future gate pins that one. → L-067
- **⭐⭐ A design record's `[M]` can carry a CONFOUND — run the CROSSED CELL before repeating
  its cause.** A record blamed GEOMETRY for a split ("sphere works, slab doesn't"); `[M]` the
  crossed cell shows geometry is inert and **L** decides (slab L=1 works, sphere L=2 fails
  identically to slab L=2), because the object under test was built from a component that
  knows no geometry. The correction STRENGTHENED the section. ⚠ And the committed probe was
  the record's own un-physical arm, cited by a PRODUCTION docstring for the other arm's
  number. → L-067
- **⭐ A brief's "Class.method" inherits the TREE's own errors — `hasattr` before minting the
  role.** `[M]` `FaceField.from_face_arrays` does not exist (it is `BoundaryField`'s), and the
  same wrong class sits in the production docstring the brief was read from. → L-067
- **A "MEASURED, do not re-derive" block is a CLAIM** — that means "don't burn a session", not
  "don't check". A bit-identity attribution was wrong on exactly the configuration that motivated
  the change; widen the repro to the WHOLE inventory, since the brief's sample is never the
  population. Two mechanically different effects can also measure the same (reduction-order drift
  vs a real value bug both read ≤1 ULP when the offending weight is `O(ε)`), so a ULP table cannot
  justify such a change — give the structural reason and `.. warning::` that the ULP row is NOT
  evidence of equivalence. → L-043
- **You are the judgment layer over any bulk scanner.** Import-verify the SUGGESTED target (one
  named a class existing nowhere); reject findings whose evidence the current clean build
  contradicts; when a retarget crosses a numerical/structural claim read the successor's live body;
  attribute every residual to a known false-positive class. → L-021
- **A naming-dense brief on a fast-moving branch goes stale FIRST** — import-verify every class,
  helper and line-ref before minting a cross-ref. → L-018, L-039
- **A plan's phase-line and internal task numbers are stale tracking artifacts.** Never infer
  "phase N is a gap" from an open plan bullet — read the shipped page; never trust a plan's bare
  `#N` (internal numbering COLLIDES with real issues); verify every issue you cite. → L-038, L-041
- **Verify a "the gate still does X" claim against the TEST BODY, and COUNT the rows.** A class
  docstring described semantics its body had abandoned, and only 3 of 7 cases were re-posed — my
  draft's "every case" was a fresh falsehood. Likewise `python -c` every numeric constant a doc
  asserts (one was four orders off, unwarned, for months). → L-042
- **An EQUATION has TYPES and a SCOPE, and NO gate checks either — read the domains/codomains, and
  ask which instance the proof covers.** A published `R∘G = R` could not type-check
  (`Γ₊→Γ₋` vs `Γ₋→Γ₋`); nobody introduced it — a CODE carve narrowing the spaces retroactively
  falsified a math sentence three chapters away. Separately, "the crossing is geometric … which is
  why G carries it" was proven for the mirror and stated for EVERY law. A narrowing carve is a
  licence to re-type-check every identity naming the affected spaces; a "which is why X" closing a
  one-instance argument is a licence to re-quantify it. Fix by SCOPING the proof and ADDING the
  missing case — never rewrite a proof that was only over-quantified — and write out the boundary
  cases (they turned out to be a shipped realizer REFUSAL, the best evidence the new law is right).
  Root cause of both: a factored form presented as BOTH a classification and a computational
  recipe; say which it is, and check the declaration tier against the REALIZATION first. → L-048
- **Describe a probe, never cite an ephemeral path.** A `$CLAUDE_JOB_DIR/tmp/` script no reader can
  open is a stale raw path the moment it is written (and `scratch/` is untracked). Publish the
  construction — shapes, metrics, comparison — so the table regenerates from the page. Reproduce AND
  WIDEN every measured number a plan hands you (a 3-sample `|Γ₊|=|Γ₋|` claim became 6 quadratures ×
  every face). → L-048
- **⭐⭐ Publish YOUR number with YOUR configuration; never relay one whose fixture you cannot
  state.** Two brief/plan numbers did not reproduce because they were measured on fixtures I did
  not have (a heterogeneous SI-vs-Krylov gap; a pre-fusion build time). Re-measuring gave
  different, correct values on a fixture I could name — and the re-measurement also refuted the
  brief's headline *as stated*: "excited iff `n_x` is ODD" is true of the symmetric fixtures and
  false as a mesh property (`[M]` `dim ker A = 12` at even AND odd `n_x`; an anisotropic source
  excites the even one at `1.76e-2`). Publish the scoped rule + the counter-row. → L-057
- **⭐⭐ A ratio is a ratio OF AN OBSERVABLE — name it before citing it.** A memo's `n_GS/n_J` was
  ρ-DERIVED (`ln ρ_J/ln ρ_GS` from an eigen-solve) while every published table reported SWEEP
  COUNTS. My control reproduced the published `1631/838` **exactly**, then **4 of 5** memo rows
  disagreed in **SIGN** (`0.576` "wins" vs measured `2.503` "loses") — two individually sound
  instruments, different observables, near-degenerate rates. Publish only the observable you
  measured; never let a rate-ratio and a count-ratio share a column heading. → L-051
- **"Already done in code" ≠ "gating green".** A brief's "don't redo the code" premise said nothing
  about the build: the `-E` baseline carried an `ERROR: Malformed table` in the very docstring that
  pass had just edited, so the brief's own `-W` gate could not have passed. Fixing it was blocking
  AND in scope. Diagnose a simple table by rebuilding column spans from the `===` separator
  (`re.finditer(r'=+', sep)`) and flagging non-space chars in the gaps. → L-051
- **Importing algebra from code/memo imports its SYMBOL COLLISIONS** (`A_a` face-area vs the loss
  operator `A`; `Σ` transmission vs `Σ_t`). KEEP the code's spelling — internal consistency outranks
  local awkwardness — and pay with an explicit `.. note::` naming each overload AND its
  disambiguator. Never silently rename into the docs: that mints a code↔corpus twin. → L-051

---

- **⭐⭐ A gap YOU REPORTED upward has the shortest shelf life in the corpus — the
  report is what triggers its repair, so re-measure it before quoting it.** `[M]` my
  own CS5 page shipped *"the gap is reported, not repaired here"* about a roster the
  coordinator had fixed in the SAME batch (`cb3cd15b` and my `f8c69117` share a commit
  timestamp to the second) — two present-tense-false sites, one of them a
  `(vv-status rationale)` comment. ⟹ publish a repaired gap **as history with its
  repair hash** ("shipped with four; the fifth landed the same day — the finite-roster
  corollary demonstrating itself"), which cannot rot, never as an open gap. → L-075
- **⭐⭐ Run the BRIEF'S OWN CENSUS before writing to it — the count can be ZERO, and the
  surplus is where the work is.** A briefed *"~a dozen `StreamingOperator(sn_mesh)` ctor
  spellings in docs"*: `[M]` whitespace-flattened over every `.rst`, **0** — the corpus never
  carried the spelling. The same greps found what the brief never named: a **Key Facts**
  bullet still saying τ is *"delivered … as `CellVisit` data (c_in, c_out, τ), stamped at one
  production site"*, `[M]` `dataclasses.fields(CellVisit)` = 3 other names — all three fields
  and the stamping method retired one phase earlier, by **my own previous pass**. ⭐ Same
  shape at the P4-remainder: courier prose `[M]` **1 site** (already fixed), two-arg ctor
  **0**, `angular.quadrature` **0** — while every real find was P4.5–P4b staleness the brief
  never named, because **a phase that lands with no docs pass leaves its rot for the NEXT
  phase's sweep to discover**. Budget for it: briefed 3 pages, honest scope 5.
  → L-072, L-075
- **⭐⭐ A relayed MUTATION COUNT is cheap to reproduce, and the first run's extra reds are
  usually YOURS.** A relayed *"the minting-`pose` mutation reddens 5 rows, all structural"*
  came back **9** — four of them my mutant crashing on the Cartesian arm (vv #17 producing
  #18's flattering symptom in one probe). Repaired **in-class** (mirror the hub's own
  construction arm so only IDENTITY changes): exactly **5 of 65**, the same set by name,
  60 value assertions green. Reproduce the red set's IDENTITY, not just its size. → L-072
- **⭐⭐ A phase hands you two numbers; one is stable and one is fixture-bound — publish both
  halves and yours wins.** `[M]` the per-build cost reproduced to the digit (8.84 vs 8.78 ms)
  while the operator COUNT did not (mine **38–43**/solve vs a briefed 6–10 — it scales with
  the OUTER count), so the consequence is **+68 %**, not 24.65 %. The stable halves are the
  unit cost and the *scaling law*; the percentage is a fixture reading. Re-measuring made the
  ruling's own case 2.8× stronger. → L-072, L-057
- **⭐⭐ When a landing change gives a REAL NAME to a phrase the page already used
  loosely, the phrase becomes a MIS-STATEMENT — not a head start.** `spaces.rst`
  had promised *"an axis carries … the identity of the generator that produced
  it"*, meaning *identity is typed per subclass*; CS5 shipped a `generator` FIELD
  whose ruling is **provenance is NEVER identity** — the exact inverse of how the
  sentence now reads, and it was echoed in the machine header and the foundations
  index. `-W` is silent. Grep the NEW field's name across `docs/` BEFORE writing
  and read every hit as a claim about the new thing even when it predates it; fix
  by a `.. warning::` naming BOTH readings, never by deletion. → L-074
- **⭐⭐ A REFUSAL in "what was tried" can be half-falsified, and the ARROW is the
  reconciliation.** *"An `Axis` → measure accessor — refused … the axis stays four
  slots"* vs CS5's fifth slot: `[M]` the refused arrow (axis → measure) would have
  had to MANUFACTURE its output (the axis had dropped the nodes; the only synthesis
  is the index set, which is what `frame.py`'s collapse mint builds locally); CS5's
  arrow (generator → axis) manufactures nothing. `[M]` the collapse pair still
  builds its own index-space measure and never reads `axis.generator`. Preserve the
  refusal VERBATIM, move only its tense, add the dated arrow argument — and state
  the unchanged call site too, because "the axis can now reach a measure" invites
  the wrong inference there. → L-074
- **⭐⭐ An "EXHAUSTIVE over the shipped family" roster is a universal owing its
  denominator — enumerate the family, never trust the roster's own list.** `[M]` a
  gate citing vv #31's finite-roster corollary BY NAME listed "the four `Quadrature`
  classmethod factories"; `vars(cls)` + `isinstance(v, classmethod)` says **five** —
  `folded_product`, the σ_y-folded cylindrical CARRYING rule the curvilinear MMS
  builders default to, i.e. the member richest in the very datum the roster gates.
  Citing the rule reads as having applied it. → L-074
- **⭐⭐ To justify a NEGATIVE design ruling, SIMULATE the rejected design in a
  SUBCLASS and publish the traceback — never mutate production.** Four lines
  appending the excluded field to `_identity_key` proved "the exclusion is
  structurally mandatory": `[M]` `==` → `ValueError (ambiguous truth value)`,
  `hash` → `TypeError (unhashable)`. No production file touched ⟹ the crash-unsafe
  revert hazard cannot bite, and it beats restating the docstring. → L-074
- **⭐⭐ A landed change that SILENTLY PRESERVES a published measured table is
  itself publishable.** CS5 put a distinct `Quadrature` instance inside the subject
  of `field_algebra`'s twin-carrier fiber row; `[M]` re-measured, the row still
  reads `True` (and would have RAISED, not flipped, under an inclusion). Re-measure
  such a table and add a dated note saying it survived BY DESIGN — otherwise the
  next reader who spots the new field assumes staleness. → L-074
- **⚠ The HTML slice IS the nested-markup gate; a source regex CANNOT replace it,
  and anchor the slice with `rfind`.** L-069's guard (`"**``" not in text`) sees a
  literal at the START of a bold run and MISSES one in the middle
  (``**… ``generator=`` …**`` shipped 4 visible backticks). A "literal inside bold"
  regex is unusable here — it joins one run's closing `**` to the next run's
  opening and reported **119** false positives on one page. And the TOC repeats
  section titles, so a `find`-anchored slice silently checked a 204-char fragment
  and "passed". → L-074

- **⭐ Verify the memo's `file:line` ATTRIBUTION, not just its fact.** A design memo put a
  scheme-type comparison in `SNMesh.__eq__`; `[M]` `SNMesh.__eq__ is object.__eq__` is **True**
  and two identically-built meshes compare **unequal** — the comparison lives in
  `is_same_phase_space`, whose docstring also records that the angular closure is deliberately
  EXCLUDED. My draft note was false and became a much better one. → L-072

- **⭐⭐ A GATE'S NAME IS A UNIVERSAL — run its predicate over the whole shipped family
  before repeating it, because the measurement can hand you a THEOREM.** A gate named
  `..._is_a_sphere_family_property` was `[M]` false twice over: `product(4,4)` IS a sphere
  rule and BREAKS the property, and `folded_product(4,6)` L=3 is DENSE with a non-constant
  per-ℓ diagonal and SATISFIES it (`rel ≤ 2.8e-15` over 200 seeds). Chasing the outlier gave
  the decidable form — `M* = R/W ⟺ Y(G⁺ − diag(d)/W) = 0`, agreement only *modulo* `ker Y`;
  its coupled 2×2 block is `det −8.7e-17`, **rank 1** (linearly dependent harmonics), so
  `‖Y·D‖ = 4.4e-16` at `‖D‖ = 0.557` vs `6.30` on the slab. Publish *"DIAGONAL is SUFFICIENT;
  DENSE does not decide it"* — strictly stronger than the gate's own name, and it stops the
  next reader "fixing" the gate by adding the DENSE params. → L-076
- **⭐⭐ An OPERATOR-movement claim has a DRAW-FREE form: build the matrix column by column
  (`e_k` through both arms), never probe it with one random vector.** `[M]` a gate docstring's
  `max|Δ M.H| = 8.246, rel 0.8995` is ONE DRAW — the same relative movement bands
  **0.879–0.986** over 200 seeds (0.53–4.55 on another frame), while the operator-level
  Frobenius relative is a stable **0.980–0.985** on all three frames. L-071's three-flavours
  rule, moved from a float-agreement claim to an operator-movement claim. → L-076
- **⭐ Publish the ANALYTIC threshold, not a scan point — and WRITE THE NORM.** A pinned-`rcond`
  docstring said the cliff is *"at ≥ 5e-2"*; `[M]` it breaks at `3e-2` and the true edge is
  `σ_min^live/σ_max = 1.75e-2` (pinv's rcond is relative to `σ_max` — no scan needed). Same
  file: *"`G G⁺ G = G` to 9.99e-16"* is three different numbers under three reasonable norms
  (`1.554e-15` max-abs / `7.77e-16` rel-to-`max|G|` / `7.75e-16` Frobenius), and a quoted
  noise-floor eigenvalue (`6.82e-17`) does not reproduce at all — publish the STRUCTURE
  (*5 live slots, rank 4, smallest live mode `4.745e-2`*). → L-076
- **⭐⭐ A CLASS SPLIT is the staleness class no gate can rank — the symbol still exists, so
  every hit resolves and every stale sentence reads fine.** `[M]` `FissionOperator` = 50 doc
  hits / 19 files after the fission channel split into an ENERGY and an ANGULAR binding; ~15
  of them meant the scalar dyad, which moved to a NEW class. ⟹ the instrument is an **AST
  census of production CONSTRUCTION SITES, per package** (`ast.Call`, func = Name or
  `Cls.classmethod`), not a doc grep: `[M]` `FissionOperator` **1 site, `sn` only** vs
  `IsotropicFission` **4 sites across diffusion/homogeneous/sn/transport`. That one table
  decided all ~15 adjudications and became the corrected prose's evidence. → L-077
- **⭐⭐ A split can refute the corpus's own THESIS — and the repair makes it SHARPER, so
  write it that way.** Three sites (a root page ×2 + a PART index) asserted *"X and
  `FissionOperator` are the **same Python classes** in SN, diffusion and homogeneous"*.
  Post-split: false for F, true for the sibling. ⭐ Fission had been the ONE channel with a
  single class serving a scalar AND an angular consumer — which is why it read as the
  cleanest example of sharing while hiding the *shape* of it; after the split all three
  channels share one two-binding shape. Publish the census as the sentence's evidence, and
  add the machine-header key it implies. → L-077
- **⭐⭐ A published `.. code-block:: python` is the highest-severity staleness there is** —
  it promises reproducibility, nothing gates it, and `[M]` mine had a constructor call
  (`FissionOperator.from_solver_data`) the live tree replaced. ⟹ after any
  constructor/signature change, grep the CODE-BLOCK bodies for the changed symbol BEFORE
  the prose sweep. → L-077
- **⭐⭐ TWO sibling changes described by ONE sentence can differ in KIND — run both.** Both
  N2N and F swapped hand arithmetic for the same product reversal, and both production
  docstrings say *"a pure IEEE-754 order change… gated at tolerance"*. `[M]` N2N is
  **bit-identical 1000/1000** (200 seeds × 5 GL orders, `max|Δ| = 0`) and F is **0/200
  bit-equal, ≤5 ULP** on three angular rules — because at ℓ=0 the outer factors degenerate
  (`R₀ᵀ` = ordinate sum, `M₀ᵀ` = per-ordinate ×wₙ) so N2N's chain does the same ops in the
  same ORDER, while F's retired spelling divided by W on the other side of `Kᵀ`. Publish the
  pair as a table with a ⚠ *do not pin the other channel at `array_equal` on this one's
  strength* (vv #31). And REPORT the docstring that understates itself. → L-077
- **⭐ A page can contradict ITSELF 80 lines apart, and the stale half is the one a reader
  quotes.** One page stated the new algebra in its extraction narrative and the OLD algebra
  twice in its operator section. ⟹ after an algebra change, grep the OLD spelling **within
  each page that already carries the new one** — a page that learned the correction is the
  likeliest home of the uncorrected twin. → L-077
- **⭐⭐ Sort every site into (a) a claim about the PHYSICS vs (b) a claim about the
  MODEL/CODE — the SAME fragment is false about the reaction and true about the
  operator.** *"(n,2n) emission is isotropic"* ⟹ FALSE, correct it; *"only the `[0,0]`
  block is written"* ⟹ TRUE of what ships, KEEP the assertion and say it is a
  TRUNCATION. Both errors are easy: weakening a true code claim because its neighbour
  was false, and leaving a false physics claim because its neighbour is right. `[M]` one
  docstring carried both **in one sentence** across a dash. → L-078
- **⭐⭐ When a page CONTRADICTS ITSELF, the HEDGE is usually the true half — promote it
  to the measurement, demote the flat assertion.** `[M]` two files said *"in principle
  carries its own anisotropy"* and then, 20 lines later, *"emission IS isotropic"*. Where
  the hedge sits in a QUOTED RULING, keep the quote verbatim and add a dated paragraph
  saying the ruling is **strengthened** — the axis it declined to foreclose is real.
  → L-078
- **⭐⭐ A relayed CONTRAST needs both sides' denominators — one can be a different
  population.** `[M]` a memo's *"μ̄ = +0.278 vs +0.094 elastic, ~3×"* summed the two over
  DIFFERENT energy windows; over the same 50 groups elastic is **+0.4264** and the "3×"
  inverts. Every other number reproduced exactly. ⟹ publish your figure with its window,
  and replace the contrast with a STRUCTURAL one (*"MT=16 stores NL=7, the same order as
  elastic, which stores 7 in 13 of 13 files"*). → L-078
- **⭐⭐ Re-run EVERY inherited number; the one that is false is the one already copied into
  KEY FACTS.** 4 of 5 briefed `[M]`s reproduced exactly (forgery norms; `18` string-manipulating
  `.support` reads; `87` `support=` kwargs; the whole `S²/SO(2)` derivation, `simplify(mine −
  shipped) == 0`). The fifth — *"the frame's level-2 arrow type-checks, shapes `(8,3) → (8,)`"* —
  was `measure.nodes.shape`, not the arrow (`[M]` `L2[S^2] (8,) → spherical_harmonic_space
  (3,5)`). ⭐ **The right reading was the better exhibit**: `measure.space.name` is `[M]`
  `L2[S^2]`, the forged level-1 tag propagating verbatim into the DERIVED level-2 name — a
  publishable paragraph the wrong number hid. ⟹ when a relayed shape is wrong, ask what the
  correct one SHOWS. → L-079
- **⭐ A brief's code claim splits into a GATED half and an ASSERTED half — publish both.**
  *"`measure.quotient` performs no lookup and no check"* is half false: it DOES gate the measure
  (`orbit_certificate` refuses a non-invariant one); what is ungated is the **tag**
  (`f"{support}/{group.name}"` mints `'not_a_manifold_at_all/sigma_y'` without complaint). Two
  objects, one gated, one asserted. Same session: *"minted against three manifolds over five
  sites"* → `[M]` **18** ctor sites, **4** in `orpheus/`, three families — *three* right, *five*
  not. → L-079
- **⭐⭐ Your own reproduction can fail for YOUR reason — diagnose before reporting, then PUBLISH
  the trap.** `sp.Matrix(...).subs(x**2+y**2, p2)` silently fails on `4x²+4y²` (no literal node),
  giving a bogus `det P` and an apparent disagreement with the shipped entry; `factor` first.
  Publishing it as a `⚠` reproduction note is what makes the neighbouring agreement table
  credible. → L-079
- **⭐ `hasattr(Cls, field)` is FALSE for a dataclass field with no class default — fall back to
  `dataclasses.fields` before reporting a dead `:attr:`.** `[M]` my own role-import probe
  minted 3 false dead targets this way, all on the newest code. L-053(c) (*construct the
  object*) with a cheaper oracle. → L-076
- **⭐⭐ A gap YOU report can have a shelf life of ZERO — its OWN commit can close it.** `[M]`
  my twin-lookup table shipped *"`Trivial` → `NotImplementedError` — ⛔ the catalogue lacks the
  identity quotient"*, and the same commit that published the table added the fix (its own
  message says so). The mechanism is structural, not sloppy: **comparing two implementations is
  simultaneously what exposes a gap and what motivates repairing it**, so within one session the
  table is written before the tree it describes. ⟹ re-run every gap-claim's check against the
  FINAL tree, after the session's last code edit, and publish the outcome as history-with-its-
  repair-hash. (Strictly stronger than L-075's *"shortest shelf life"*.) → L-080
- **⭐⭐ When a brief offers *"they coincide, therefore no fork"*, ask what would have to be true
  for them NOT to coincide — if the answer is "a gate would raise", the coincidence is a LAW and
  the sentence is inverted.** `[M]` a briefed *"for `SO(2)` the chart and the section coincide in
  dimension"* holds in BOTH entries (1/1 and 2/2) **because the new `__post_init__` gates it** —
  a quantity a construction law forces to agree cannot discriminate anything (vv #19 at the
  design tier). The two reproducible reasons were different in kind: no canonical section exists
  for a positive-dimensional group, and the tree's `SO(2)` data is already chart coordinates
  (`(8,)` vs the fold's `(16,3)`). → L-080
- **⭐⭐ Before calling two implementations of "the same" lookup a Pattern-2 twin, check they take
  the same ARGUMENT.** `[M]` `AngularSymmetry.support` computes `S²/G⁰` — the *continuous*
  isotropy — so a mirror (a member of the *discrete* residual Γ) is a row it **structurally
  cannot** answer, not one it has not been extended to. Publishing "the catalogue overtook its
  twin" would have aimed the repair at the wrong half; the true statement is that the registry is
  the special case `H = G⁰`, so the collapse is `support = base.quotient(G⁰).realization.name`.
  ⭐ Chasing the distinction found a latent break: stage 0 is a STRING compare, and `[M]`
  `cylinder.admits_domain(folded_product(4,8).measure)` is **False**. → L-080
- **⭐⭐ A campaign day lands SIBLING steps, and each one repeals a PREMISE somewhere else in the
  corpus — grep for the premise, not for the step's name.** `[M]` FOUR present-tense-false claims,
  none in my brief, all true when written and repealed hours earlier by 2.0c/2.1: *"`support` is
  still a `str`"* · *"the slot is `domain`, and it is **not yet built**"* · *"`indicator_basis.py`
  **hard-codes** it"* (×2 pages). ⭐ The sharpest is ERR-080's entry **contradicting itself 170
  lines apart** — a `✅ Progress (2.4)` block announcing the first production consumer, then a Fix
  bullet reading *"⛔ That type has no production consumer yet"*. Repair shape: keep the numbers
  and the verdict, tombstone only the PREMISE, in place (`⛔ This clause read "…" until <date>:
  true when written, and repealed hours later by <step>, which is the campaign's own step`).
  → L-081
- **⭐⭐ The brief's target SECTION can be absent, and a page's shared WORD can be a different
  object.** Briefed to extend *"wherever `quotient_group` / HAS-vs-SPENT is explained"*: `[M]`
  **0** hits on that page, and its heavy *"spent"* is the registry's continuous `G^0` (a geometry
  spends it), not a measure's fold group. Write the missing home rather than bolting onto the
  homonym. ⚠ And the claim I nearly shipped in it — *"two of the fields in the table above name a
  subgroup"* — was false: the propagation table carries only the STORED one, because a derived
  field has nothing to propagate. → L-081
- **⭐⭐ Widen a table to the WHOLE shipped roster before writing prose over it — the added row
  is usually the one that turns a design note into a theorem.** I drafted *"the **four** shipped
  angular rules"*; `[M]` `vars(Quadrature)` + `isinstance(v, classmethod)` says **FIVE**
  (L-074's finite-roster miss, now in my own prose). Re-run at 5, the bottom two rows ARE the
  argument: the slab carries **two different groups in two slots on one measure**
  (`HAS Mirror('x')` / `SPENT SO2('x')`), and the σ_y fold `HAS None` **because** it spent σ_y —
  *spending a symmetry destroys having it*. ⭐ Same fix on the pairing table: re-posed as the
  pairing the tree ACTUALLY forms (each rule vs the basis its own `angular_frame(2)` binds) it
  reads **1 of 5 fails**, strictly stronger than four hand-picked rows. ⚠ Then I wrote *"the same
  denominator ERR-080's scope census reports"* — **false**, that one counts `(constructor, order)`
  rows (7 of 15); struck, and the incomparability published as a ⚠. → L-081
- **⭐⭐ A quoted tracker/plan row is a QUOTATION — grep it out of its source before it ships.**
  I published an italic *"the tracker read …"* that fused two separate plan clauses and invented
  the framing they "asked for". A paraphrase in quotation marks reads as verbatim. → L-081
- **⭐ Soften every superlative and every "share no X" — check the DEGENERATE member.**
  *"the **largest** group known to map the node set to itself"* (it is a stored DECLARATION, not
  a computed stabiliser) and *"the full degree-L harmonics **share no symmetry**"* (`[M]` false at
  `L=0`: `space.shape == (1,1)`, one constant, O(3)-invariant — the page's own lower-bound
  caveat). → L-081
- **⭐⭐ Quoting a docstring INTO the corpus is itself an instrument.** Copying
  `quotient_group`'s *"(:meth:`restrict`, :meth:`consolidate`, :meth:`reorder`, …)"* tripped my
  role-import probe: `[M]` `hasattr(DiscreteMeasure, "reorder")` is **False**, only occurrence
  tree-wide. Nothing else sees it — the role is UNQUALIFIED (the xref gate skips those by design,
  `DEAD TARGETS: 0`) and the module has no `automodule` (nexus `dead_references` = 0 dead / 52
  checked). A fully-qualified copy is decidable where the unqualified original is not. → L-081
- **⭐⭐ A brief's "zero production consumers" about a TYPE is a claim about ONE CONSTRUCTION —
  census with ARGUMENTS.** `[M]` a briefed *"`Ball` had 0 production consumers"* was false:
  `git grep "Ball(" HEAD` = 6 lines, of which **four** are constructions, all `Ball(2)`, one in
  production (the sigma_y entry's `realization`). What had never existed is **`Ball(3)`**, and what
  is new *in kind* is a `Ball` as an arrow's **codomain** rather than an entry's field. The
  production docstring carries the same overstatement — report it, don't edit it. -> L-082
- **⭐⭐ A retirement can delete the corpus's own WORKED EXAMPLE of a rule that survives, and no
  symbol grep sees it.** ERR-080's *"a constructor writing a membership claim as a **literal** while
  its neighbours derive theirs"* had `spherical_product`'s `support=SPHERE` *"sitting between
  `invariance_group` and `exactness`"* as its exhibit; 2.3 removed exactly that literal. Rule
  intact, evidence false, and the sentence never names the retired construct in a role. ⟹ grep the
  retired construct **as an exhibit** (*"here X sits between Y and Z"*), past-tense the exhibit,
  keep the rule, and **re-census the tell** (`[M]` 5 -> 4 live: 3 honest tabulations + 1 forgery —
  a literal is a tell, not a verdict). -> L-082
- **⭐ A "phase N mints the typed X" prediction can land on the DAY, mint a type, and still not
  deliver X.** 2.3 minted `ManifoldMap` and made no section; PHASE right, MECHANISM half-right,
  DELIVERABLE absent. ⭐ The transferable half is the naming ruling that caused it — *a chart is
  `M ⊃ U → R^n`, and only the INVERSE of the shipped map is one* — so `Chart` would have
  mis-described 2 of its own 3 instances. Publish the reason, not just the tense flip. -> L-082
- **⭐⭐ A composition law is measurable ON THE SHIPPED OBJECT — look for the chain before calling
  it abstract.** `folded_product` **is** `retraction o archimedes`, so functoriality is `[M]` 5 of 5
  (one-shot == two-step == the shipped rule, support by identity). ⚠ State the fixture: the fold
  ships the STAGGERED circle rule (`Sigma = empty`); node-aligned puts 4 nodes on `Sigma` and folds
  16 -> **10** orbits with four singletons, so a wrong shift agrees with itself and not with
  production. -> L-082
- **⭐⭐ "Picklable" and "round-trips EQUAL" are TWO claims, and a `functools.partial`
  splits them.** `[M]` over 7 shipped entries the callable round-trips with identical
  output **7 of 7** (a `lambda` would raise) and compares equal **1 of 7** — only the
  plain module-level function, which pickles *by reference* and so returns the same
  object. ⭐ The refutation gave the better argument: `field(compare=False)` is what
  makes the OWNING dataclass round-trip equal 7 of 7, so the exclusion is load-bearing
  for serialisation (the object is a cache key), not merely "a function has no value
  equality" — which is what the production comment says. → L-083
- **⭐⭐ A negative leg must sit outside the measured functional's TRUE stabiliser, not
  outside the group you are documenting.** My natural choice failed on 3 of 7 and the
  failure was the finding: `[M]` the quotient map `π_a` is bit-exactly invariant under
  `σ_b` (b≠a), because a reflection in a plane CONTAINING the axis fixes every
  constant-μ circle — so `O(2)_a` and `SO(2)_a` induce the same orbit partition. ⟹ a
  quotient map determines the PARTITION and the partition does not determine the group;
  `Quotient.by` is a DECLARATION. Sharper than ERR-072: that predicate under-determines
  because it is SAMPLED, this one while being EXACT. Re-chosen, the leg is 7 of 7.
  → L-083
- **⭐⭐ A §6d import verdict needs BOTH module-scope placements — and the package
  `__init__` is usually why the answer is placement-independent.** Top-of-file and
  bottom-of-file die with DIFFERENT `ImportError`s (`Manifold` vs `DiscreteMeasure`,
  one hop apart); testing the *most favourable* position is what turns a sample into a
  claim. `[M]` 7/7 alive at function scope, 0/7 at either module placement — identical
  across all seven entry points because `numerics/__init__.py` eagerly imports
  `.measure`, so the entry point has no say. Run it on a **RENAMED** shadow package
  (`shadowpkg`), which defeats the editable finder more simply than stripping it, and
  print `__file__` per subprocess. ⭐ The rule the section rests on: **a
  `TYPE_CHECKING` guard defers a NAME and can never carry a VALUE** — so a type and a
  value need two mechanisms, and "the cycle blocks the import" is not "the cycle blocks
  the slot". ⭐ Its safety condition is a property of the CALL SITES (`[M]` 0 of 8 at
  depth 0, by AST) and can be broken from outside the module; publish the TOTAL as the
  positive control, since zero-calls and zero-module-scope-calls print the same zero.
  → L-083
- **⭐ Two new fields of one step can have OPPOSITE consumption — census each, don't
  average.** `[M]` `Quotient.reference` has ONE production reader; `quotient_map` has
  ZERO (ten occurrences, all in one test module). Half the step is consumed, half is a
  capability; L-079's three-places rule applies to the second half only. → L-083
- **⭐⭐ When a step binds an object the corpus has already ⛔-CONDEMNED, the doc's first
  job is to name the ROLE that condemnation was scoped to.** #429 2.5 binds Λ's ends to the
  CONTINUUM Gram, which `spherical_harmonics.rst`'s three-metric table stamps *"the wrong
  side for covariant moments"*. Not a regression: F-0's verdict is about the analysis face's
  CODOMAIN; Λ's ends are an ENDOMORPHISM's, and an ℓ-diagonal metric commutes with a per-ℓ
  scalar. ⭐ The same test picked the host page: the brief offered two, `[M]` `moment_space_on`
  = **0** doc hits (neither owns it), and only `frame.rst` owns `basis.space`-vs-`basis_space`
  — homing it elsewhere would twin F-0's metric narrative. → L-084
- **⭐⭐ An OPERATOR-movement table built column-by-column beats a probe vector, and the
  numbers get BETTER.** `[M]` one draw vs `e_k`-columns: inert band `1e-12` → **1.045e-16**,
  DENSE movers `≤0.988` → **1.5839**, and `Λ* = Λᵀ` under the continuum end goes from
  `≤1.82e-16` to **exactly 0.0 on 33/33** (the `g_C Λᵀ g_C⁻¹` sandwich is a scalar times its
  reciprocal). ⭐ Then read the movers by IDENTITY: all 10 are `gauss_legendre`/`folded_product`
  at `L ≥ 1` — the `m`-dependent-Gram families — so the wrong binding would have been
  invisible to every full-sphere fixture and wrong exactly where ERR-080 lives. → L-084, L-076
- **⭐⭐ A pre-step's acceptance is bit-identity WHERE THE ANSWER IS WRONG.** The brief cited
  `L ∈ {0,1,2}` from a `scratch/.npz`; rebuilt on the ERR-080 gate's own fixture against a
  `git archive HEAD` subprocess (editable finder stripped, `orpheus.__file__` asserted) it is
  `array_equal` at **L = 0,1,2 AND 3**, `max|Δ| = 0.0`. The extra order is the argument: `L = 2,3`
  are `xfail(strict=True)` rows, so movement there could flip an XPASS without repairing
  anything. → L-084
- **⭐ A brief's `[M]` census owes its PREDICATE, and the cited command usually returns a third
  number.** *"eight homes"* was right; `git grep "…from_L" HEAD -- orpheus/` returns **13 lines
  = 8 executable + 5 docstrings**. Publish *"13 lines, of which 8 are calls"*. Same pass: a
  briefed *"12 of 12 (rule, L) rows"* is **33 of 33** on the gate's own roster (11 constructions
  from all **five** factories × L∈{0,1,2}), and a briefed *"`apply_metric` moves 96–161 %"* did
  not reproduce under any norm — replaced by the exact per-ℓ ratio `[(2ℓ+1)/4π]²`. → L-084
- **⛔ Anchor an HTML slice on `id="<label>"`, never on the bare label text — the TOC carries it
  too.** `[M]` my slice read **302 177** chars (the whole page) and "passed"; `rfind('id="…"')`
  gave **38 183** and the sanity phrases. (L-080's rule, re-broken one level down.) ⚠ And an
  emphasis regex over a page with list-tables is ~90 % false positive (`* -` bullets) —
  adjudicate every hit, never count them. → L-084, L-080
- **⭐ A section TITLE is a count too, and a roster count needs the command that produces it.**
  One edit shipped *"The four realizations…"* over a five-row table; the same page carried two of
  my own earlier universals that were wrong when written (*"Nine variants"* — `[M]` 8 at the
  mint, and the page's own table listed 8; *"30 test functions, 40 collected rows"* — `[M]` 32 /
  44). Prefer a second instrument: the generated `matrix.rst` row confirmed the test count
  independently. → L-080
- **⭐⭐ A corpus-wide SHAPE CONTRACT can be one family's layout wearing a universal — grep
  the SHAPE, not the symbol.** `[M]` `(L+1, 2L+1, ng, *spatial)` is asserted as THE moment
  layout at 9 sites / 7 pages, including the CONVENTION page and the SLAB page, where a 1-D
  rule now gives `(L+1, ng, nx)`. The carrier class never changed, so every xref resolves.
  Same pass: the neighbouring page's **flagship dense WITNESS was the defect, tabulated**
  (the two `0.8`s in `frame.rst`'s slab Gram ARE the fabricated columns) — preserve the
  numbers, tombstone the interpretation, re-measure the replacement (mine found a
  strictly-stronger one with a closed-form cause). → L-085
- **⭐⭐ A float-agreement figure needs its OBSERVABLE, not just its fixture.** A docstring's
  *"with pure `lpmv` the flux moves by 4e-16"* is the memo's **table**-level number; `[M]` the
  FLUX moves **2.753e-14** (Krylov amplifies a 1e-16 table perturbation). ⟹ two brief numbers
  refuted, both by naming which quantity was measured. → L-085
- **⛔ Tombstone a quoted claim with plain quotes + `, verbatim,` — NEVER an outer `*…*`.**
  Bold/literal nested inside an italic quotation is the natural §3 spelling and it LEAKS
  (`-W` silent). The gate is the rendered HTML with `<pre>`/`<code>` stripped, sliced BETWEEN
  consecutive section `id=`s; a source regex over the diff is ~90 % false-positive. `[M]` 8
  self-inflicted, all fixed; 0 of 176 remaining leaks trace to a line I added. → L-085
- **⭐ A "still OPEN" clause can be repealed while its NEIGHBOUR stays exactly true.** `[M]`
  ERR-080's bullet: the membership predicate is STILL unenforced (a forged measure still
  constructs) and the defect is STILL closed — the refusal moved to the basis and the frame.
  Split in place; past-tensing the bullet would have deleted a live seam. → L-085
- **⭐⭐ A brief saying "the change is ALREADY in the working tree" does not mean the
  DESIGN is settled — ask whether a review is still running, because a measured `[M]`
  of a transient is worthless.** A mid-task delta replaced the design a whole subsection
  of my new prose described *and had measured* (nine catalogue keys with three decoy
  refusal routes → six keys, the refusal re-homed onto a construction invariant
  `by == by.orbit_stabiliser` plus the catalogue door). ⟹ keep the rejected design as a
  `.. note:: A rejected first design, kept because it is the tempting one` listing its
  concrete costs — it becomes the only falsifiable statement of WHY the invariant exists.
  → L-086
- **⭐⭐ A page's own ⚠ CAVEAT can be the THEOREM the next step is built on — read every
  caveat as a candidate ruling before writing a new section.** `[M]` *"the map's own
  stabiliser is BIGGER than H … `by` is a declaration, not a computed stabiliser"* was
  measured, unchanged, and its CONCLUSION inverted: if the map cannot tell two groups
  apart, do not let the catalogue offer two names for one point set. Append
  `✅ And that is exactly why …` IN PLACE plus a `⛔ this note ended at "…" until <date>`,
  never delete — the caveat is the rule's derivation. → L-086
- **⭐⭐ Re-derive a recorded aggregate's DEFINITION by reproducing its OLD number before
  quoting a new one.** A page's *"0 violations over 342 (edge × fixture) pairs"* never
  defined "edge"; reconstructing it and hitting **57 edges / 342 / 0** exactly is what
  made the widened **75 / 450 / 0** a widening rather than a different instrument. Same
  pass: two brief numbers refuted (a *"≤ 5 %"* walk cost is `[M]` **11.3–26.2 %**; a
  *"D_1h = {e, σ_z}"* is `[M]` **order 4**, the Klein group, which is *why* it sits in
  one axial stabiliser only), and the number the brief lacked was the better headline —
  the walk's answer SHRINKS, `{SO2_x,σ_x,σ_y,σ_z}` → `{O2_x,σ_x}`. → L-086
- **⛔ My italic-run leak scanner was blind to the case it exists for: `\*((?:[^*])+?)\*`
  cannot match an italic run CONTAINING `**`.** Three of five self-inflicted L-085
  tombstone leaks were invisible until the body class was widened to `.{1,600}?` with
  `(?<![*\w])`/`(?![*\w])` guards, set-differenced against `git show HEAD:<file>`. → L-086
- **⭐ When a pass is told to document an UNLANDED symbol, name the dead role in the
  report.** `[M]` the patched xref gate and nexus `dead_references` both returned the
  SAME single finding (the not-yet-shipped accessor) — two independently-vocabularied
  instruments agreeing is what proves it is the only one. → L-086
- **⛔⛔ When a coordinator says a change is LANDING, poll on the INVARIANT that the rename
  is complete tree-wide, never on one new symbol — a half-written tree answers plausibly.**
  `[M]` `manifold.py` had both new symbols while `symmetry.py` still CALLED the old one, so
  `ordinate_permutation` raised `AttributeError: 'Quotient' has no attribute
  'section_coordinates'` on every rule — which reads as *"this rule does not support that
  question"*, not as a broken tree. The right poll is `! git grep -q <old> -- orpheus/`; the
  right "before" is the pinned `git archive HEAD` copy, which cannot move. → L-087
- **⭐⭐ A mid-task delta's message is a BRIEF: `[M]` 2 of its 7 items shipped DIFFERENTLY,
  and both shipped forms were better.** *"REFUSES a translated motion"* → it takes
  `motion.linear_part` (a point group acts on directions); *"`Ball(3).quotient(Trivial)`"* →
  `RealSpace(3).quotient(Trivial)`, with a better stated reason. Run the probe, then READ the
  shipped body; publish the code, report the delta. → L-087
- **⭐⭐ A "what SURVIVES / still open" verdict is the most delta-fragile sentence a close-out
  carries** — it is a claim about the ABSENCE of a repair, and a concurrent review's whole job
  is to add repairs. `[M]` my measured *"II.11 is HALF closed — a BARE support keeps the 1-D
  shape refusal"* became fully closed by the delta (`orbit_certificate` on the bare chart rule:
  `None` → 2 permutations), flipping a section title, a table row, a paragraph and the machine
  header. Re-run every *"X still …"* clause after the LAST code edit, not just the numbers.
  (L-080 at section scope.) → L-087
- **⭐ A design's ONE-EXPRESSION form is a publishable argument, and the naive two-conjunct
  form is its exhibit.** My stage 0 was *"the arrow exists AND (`X == D` OR `Γ ⊇ X.by`)"*;
  `[M]` the equality case is load-bearing, not a convenience — without it the predicate refuses
  the geometry's OWN domain (`σ_x ⊉ O(2)_x`). Asking what the ARROW SPENDS (`{e}` for the
  identity) removes the special case. Publish the failed spelling as the reason. → L-087
- **⭐ A RENAME's rationale can be the page's own subject — say so.** `section_coordinates` →
  `ambient_representatives`: a *section* is a point OF the base and the axial arm returns the
  BARYCENTRE, inside the ball — the old name promised ERR-080's own forged codomain, caught on
  the NAME alone. → L-087
- **⭐ The test tree can move UNDER you mid-session, and the mover is the evidence you were
  about to cite.** `[M]` two test modules the gate's docstring named in the present tense did
  not exist at my first census and shipped by my final build; `test_manifold` went 70 → 108
  rows. Re-run every count after the last build, with `matrix.rst` as the second instrument.
  → L-085
- **⭐⭐ When a count is being moved BY THE CARVE YOU ARE DOCUMENTING, publish the DIRECTION
  and its mechanism, not the value.** `[M]` `test_symmetry` read 230 → 212 → **215** across
  three of my own builds in one afternoon. *"The row is FALLING, because the carve retires the
  gates that pinned the per-family arms it dissolved"* is durable and explains why a falling
  count is not lost coverage; `212` is stale on the next build. (L-085's rule, sharpened for a
  count with a live author.) → L-088
- **⭐⭐ When a carve's ENTIRE claim is "no answer changed", the instrument is the FULL GRID —
  and the ZEROS are the finding.** The edges anyone thinks to name are the ones the retired
  hand table already got right. `[M]` `contains` **0 of 729**, `normalises` **0 of 729**,
  `is_invariant` **0 of 270**, walk **0 of 10**, against a pinned `git archive HEAD` tree. Say
  IN THE PAGE why a grid and not a list. ⭐ Reproduce the recorded aggregate's OLD numbers on
  BOTH trees first (57/342/0 and 75/450/0 four ways) — that is what makes a widened 175/1750
  read as the same instrument. → L-088, L-086
- **⛔ A rendered-leak COUNT is useless on a tombstone-heavy page — build HEAD's docs and take
  the multiset DIFFERENCE.** `[M]` raw inventory 34/8/90, essentially all pre-existing; my own
  `*"… (``literal``)"*` tombstone was findable only as `ADDED 2 → 0`. ⚠ Read `added == removed`
  as a CONTEXT SHIFT (my inserted paragraph moved a pre-existing leak's ±90-char window), not
  as two events. ⚠ A source-side italic regex is unusable: 10 hits, **all 10** `* -` list-table
  bullets. `git archive HEAD docs tools tests` + `PYTHONPATH=<tree>` costs minutes and converts
  an unusable count into a decidable answer. → L-088, L-085
- **⭐ A relayed COUNT needs its PREDICATE re-derived, not just its number re-run.** `[M]` a
  ledger's *"109 lines, 31 dispatch sites"*: the 109 reproduces, the 31 is **28** under a
  stated AST predicate; its *"eleven per-family helpers"* is **thirteen**. The module-wide pair
  I measured instead (86 → 31 sites, one predicate) was the better number anyway. → L-088
- **⭐ `functools.cache`'s own `cache_info` turns "but isn't computing slower than a table?"
  into publishable expert context.** `[M]` one walk asks 1152 containment questions of which
  629 are literal repeats, and builds **24** groups from 1193 reads ⟹ *"a hand table is not
  buying speed here; it is buying a second, unverifiable copy of the answer."* Measure it
  rather than quoting the docstring's own figure. → L-088

## 2. The build is BLIND to most doc-correctness defects — grep is the gate

**Meta-rule: `-W` proves only "I added no NEW warning". The acceptance evidence for a correctness
sweep is a grep inventory with a per-hit KEEP/FIX adjudication.**

- **⛔ An AST IMPORT CENSUS needs TWO resolutions, not one — and the second hid the
  load-bearing edge.** Relative imports (`level > 0`, unqualified `.module`) is the known
  trap. The new one: `from orpheus.numerics import invariance as _invariance` has
  `node.module == "orpheus.numerics"`, so it is an edge to the **SUBMODULE** and a filter
  comparing `node.module` against a module set reports it ABSENT — `[M]` mine hid
  `measure → invariance`, the single most load-bearing runtime edge in the carve. One
  positive control per import SHAPE. → L-090
- **⭐⭐ The source diff and the RENDER slice see DIFFERENT defect classes — keep both.**
  L-076's `re.S` source differential is free and catches NESTING (`**``x``**`); it is blind
  to a role that never PARSED. `[M]` mine said `new=0` while the rendered page carried
  `*"a map :math:`M/H \to M`, …"*` as **`M/H to M`** — a role opened inside an emphasis
  run, the LaTeX backslash eaten, on a **0-warning EXIT=0** build. ⭐ Slice the built HTML
  to YOUR OWN section `id=`s (mine: **0 backticks / 0 leaked roles on 8 of 8**, against 74
  and 1 page-wide, both survivors proven pre-existing by `git show HEAD:`) — a page-wide
  count indicts someone else's prose. ⚠ `nohup … &` inside a background Bash call reports
  the SHELL's exit, not sphinx's: I read a stale HTML and thought my fix had not taken.
  Build in the FOREGROUND, or assert a distinctive new phrase is in the page. → L-089
- **⛔ On a LIVE branch, re-run the xref gate after EVERY build, not once at the end.** A
  concurrent carve renamed a method I had already published (`Realization.images` →
  `generic_images`) plus retired a sibling; `sphinx -E -W` stayed at **0** throughout, because
  a `:meth:` at a renamed member is plain text. The patched gate found it in one run. `[M]`
  every NUMBER I had published re-measured identically on the new tree — only the NAME moved,
  which is exactly the class the build cannot see. Re-read the public surface (`dir(cls)`),
  never the module you read an hour ago. → L-088
- **Unresolvable `:func:`/`:class:`/`:meth:`/`:attr:` render as PLAIN TEXT with no warning.** After
  any carve that deletes or renames a symbol, `grep -rn "<symbol>" docs/` and repoint every hit.
  → L-002
- **`-n` (nitpicky) is NOT the missing gate.** MEASURED: `-n` saw ZERO of 22 dead refs, because
  Sphinx only nitpicks what it RENDERS and the carrying modules were not `automodule`'d
  (`tests/**` is never read). Edits to such docstrings cannot move the warning count, so "count
  unchanged" proves nothing. → L-044
- **`tools/check_docstring_xrefs.py` IS the gate — run it, don't grep blind.** It resolves every
  FULLY-QUALIFIED role by IMPORTING it, so render coverage is irrelevant;
  `… <tree> --quiet` → `DEAD TARGETS : 0` is the acceptance criterion. Never touch its empty
  ALLOWLIST. It skips UNQUALIFIED refs by design (Sphinx resolves those by module context), and it
  is blind to LITERALS — so after fixing a renamed symbol's roles, grep the OLD NAME tree-wide and
  adjudicate every ``literal`` by tense (`_select_si_resolvent`: 1 dead role + 3 live-prose
  literals on two other pages). → L-045, L-046, L-047
- **⭐⭐ …but ON AN `.rst` PAGE that gate reports `:mod:` and NOTHING ELSE — never read an unmoved
  `DEAD TARGETS` as "my page is clean".** `[M]` I fixed 15 dead roles and the count sat at
  **81/124 both sides**: `judge()`'s head-check re-checks the target's head *carrying the original
  role*, so a single-segment head (`orpheus`) trips `bare_module_guess` under any non-`mod` role,
  and with a page's empty namespace the candidate tuple is `()` → DECLINED. One line fixes it
  (`head_role = "mod" if "." in target else role`); `[M]` on a pristine `git archive HEAD` tree that
  takes `docs/` from **49 dead/71 sites → 207/255** — the gate is blind to 158 dead
  `:class:`/`:func:`/`:meth:`/`:attr:` targets in `docs/` alone. Until it lands, the acceptance
  evidence for a page is YOUR OWN import probe over its roles. ⚠ Measure such a patch as a COPY run
  as a SUBPROCESS — monkeypatching `judge` and calling `main()` twice in-process read `0` for both
  arms while a subprocess read 49. → L-062
- **⭐⭐ …and the blindness is ROLE-scoped, NOT `.rst`-scoped — `DEAD TARGETS: 0` certifies
  `:mod:` targets and NOTHING else, in `.py` docstrings too.** (The entry above read "on an
  `.rst` page" until 2026-08-24; that was too narrow.) `judge()` re-checks the target's HEAD
  carrying the ORIGINAL role, and `candidate_paths("orpheus", ns, "meth")` →
  `('<ns>.orpheus',)`, which never resolves ⟹ every DEAD fully-qualified `orpheus.*` target
  under a non-`mod` role is DECLINED; live ones return ALIVE earlier, which is why the gate
  looks healthy. `[M]` `judge("…FunctionSpace.definitely_not_here", role="meth")` = DECLINED vs
  `judge("orpheus.numerics.does_not_exist", role="mod")` = DEAD. The one-line fix
  (`head_role = "mod" if "." in target else role`) on a COPY, run as a SUBPROCESS **from inside
  the repo** (it resolves against `REPO_ROOT`, so a `/tmp` copy scans 0 files), read 1 dead / 2
  sites where the stock gate read 0. → L-067, L-062
- **⛔⛔ A TRAILING SPACE before a closing role backtick swallows the sentence, and `-W` is
  SILENT.** `` :math:`… \rvert = ` `` does not close; the HTML carried raw `` ` **0.0** for
  :math:`a … `` on a **0-warning** build. Two gates: the tag-stripped **render** scan for `**` /
  double-backtick (authoritative), and a one-line corpus regex that localises it —
  ``:(?:math|ref|eq|doc|class|func|meth|attr|mod|data|exc|cite|term):`[^`]*\s` `` (`[M]` 0
  corpus-wide after my fix, so mine was the only one). -> L-082
- **⭐ Smartquotes mis-directs a closing `"` that follows an inline literal** (`` typed** ``X``\ **" ``
  renders `X“`). Detect with `re.finditer(r'“(?=\s*(until|and|,|\.))', stripped_html)` — ⚠ it also
  flags a quote that OPENS with one of those words, so read the hit. Fix by **extending** the quoted
  fragment to end on a WORD; the extension is usually verbatim anyway, and shortening is the natural
  wrong move. -> L-082
- **⛔ An HTML slice anchored on the NEXT SECTION'S TITLE can land INSIDE your own section** — a
  `:ref:` renders as the target's TITLE, so `rfind(next_title)` is not a boundary when you cite the
  next section. `[M]` my slice read **1 659** of **21 909** chars and reported "0 leaks", a
  designed-green reading. Anchor BOTH ends on a distinctive SENTENCE and sanity-check the slice
  LENGTH. (Sharpens L-074, which only warned about the TOC.) -> L-082
- **⭐ Two independently-VOCABULARIED instruments agreeing IS the acceptance evidence.** nexus
  `dead_references` (by RENDERED target) and the patched gate (by IMPORT) returned the SAME
  single finding; neither alone was persuasive (the stock gate said 0; nexus's set-difference
  with the gate is normally noisy). → L-067, L-052
- **⭐⭐ The patched gate needs a copy at DEPTH 1 and an END-TO-END positive control — and its
  old number is stale.** `REPO_ROOT = __file__.parent.parent`, so a `/tmp` copy AND a
  `scratch/_dir/` copy both scan **0 files, silently**; put it at `scratch/<name>.py`. Then write
  a throwaway `docs/_ctl.rst` with two dead roles + one live, run both gates, delete it: `[M]`
  stock **0**, patched **2 dead / 2 sites**, `decidable` +2 — without that, `DEAD TARGETS: 0` is
  indistinguishable from a broken scan. ⛔ **L-062's "49 → 207 dead in `docs/`" is void**: `[M]`
  2026-08-28, corpus-wide over `docs orpheus tests` (984 files / 16 068 roles), patched = **0**,
  same as stock. The corpus was cleaned; do not quote 207 as a live expectation. → L-071, L-062
- **⭐⭐ A per-site adjudication TABLE is an instrument — audit its SKIP clause, its "retired"
  verdicts, and its `hasattr` evidence.** Applying a 91-site ruling table faithfully still yielded
  FIVE corrections: (a) its *keep-if-absent-from-graph* filter hid ~1400 alive-but-unqualified
  roles, incl. one on a line I had to edit — a NOT-clause is a false-NEGATIVE source; (b) 7
  "RETIRED" sites were a pure `git mv` (`--diff-filter=D` on the old path answers this in one
  command) and 6 of them read PRESENT/IMPERATIVE, so a literal would have killed a live link;
  (c) `hasattr(Cls,"mesh") is False` ruled a TRUE paragraph false — the base class sets it on the
  INSTANCE, so **construct the object**, never probe the class; (d) two adjacent `X.apply` sites
  had OPPOSITE right answers (one qualifies — the page already links it 6 lines up; one is dead —
  the type has no `apply`), so resolve `Instance.method` by the instance's TYPE, never by the
  target string's shape; (e) one table cell opened a page-wide SYMBOL-INDEX collision. → L-053
- **⭐ "Qualify so it resolves" is TWO claims that come apart — say which one you bought.** `[M]`
  post-build hrefs: `EigenvalueSolver` 43, `Field` 30, `numpy.array_equal` 6 — real links;
  `KEigenvalue`/`SNMesh.axes`/`BC.vacuum`/`zeros_on`/`peierls_nystrom.slab` **0**, still plain
  text (`:noindex:` autoclass, or no automodule). Import-/graph-resolvable ≠ rendered link; check
  with `grep -o 'href="[^"]*#<target>"'` in the built HTML. → L-053
- **⭐⭐ A page can run a symbol convention ONE INDEX BELOW the code's, and no test can see it.**
  CP's `Ki_4` IS the shipped standard-`Ki_3` (`[M]` `_ki3_mp(0)=0.7853961=π/4`), its `ki3-def` is
  the standard `Ki_2`, its `F(0)=0.4244=4/(3π)` matches neither — across 3 labels carrying
  64/24/54 `verifies()` tests that pin the code's NUMBERS and are structurally blind to the
  equation's NAME (a doc-side Mode 12). Repoint the role, MEASURE the collision into an anchored
  `.. warning::`, fix only the unambiguously-wrong number, and REFUSE the ~30-site re-indexing as
  a numerics adjudication — a physics rename must not ride inside an xref pass. → L-053
- **⭐ NEVER say "all trees at 0" — say the TREES, the ROOTS and the SEMANTICS.** The gate walks
  only `orpheus tests docs` (NOT `examples/`, top-level `derivations/`, `scratch/`, `tools/`) and
  judges only `DECIDABLE_ROOTS` (which omits importable `tests`/`tools`/`derivations`); its NAME
  also understates it — it DOES read whole `.rst`, so `doc:` sites ARE covered. Decisive: the gate
  resolves by **IMPORT**, the `nexus dead-references` hook by **RENDERED TARGET**, so a live
  un-`automodule`'d module is *resolved* to one and *dead* to the other. Both right — the SET
  DIFFERENCE is the triage (hook-only ⇒ un-surfaced-but-live = #302; both ⇒ really moved/retired).
  Measured 21/30 vs 0/14 914 on the same tree. → L-052
- **Two gate false-negatives to know:** a **PEP-420 namespace package** (a dir with only a
  `README.md`) IMPORTS fine (`__file__ is None`, 0 members) so `:mod:` refs at it read "resolved"
  though Sphinx can never link them; and a role **wrapped INSIDE its dotted path**
  (`:func:`~a.b.\n  c``) is skipped by the gate (whitespace ⇒ `extract_target` → `None`) AND
  renders plain text. 15 such roles tree-wide; the discriminator is `\.\s*\n\s*\w` in the pre-`<`
  head — ~180 multi-line roles that break at the `display <target>` seam are FINE. → L-052
- **Before believing a dead target's NAME, read its graph EDGES**
  (`SELECT source,type FROM edges WHERE target=?` on `graph.db`). A name can be an artifact minted
  by a THIRD tree: six `orpheus.derivations.peierls_geometry.*` targets existed only because
  `scratch/` scripts still import the deleted path, and nexus suffix-matched the theory pages'
  unqualified roles onto it. Edge type decodes the site class: `documents`=page ·
  `references`=docstring · `type_uses`=**a type annotation, i.e. a CODE bug** · `calls`=the import
  that minted it. And nexus counts doc sites **per PAGE** (2 "sites" was 9 roles), while the unit
  of repair is the TARGET tree-wide (3 reported sites, 13 real). → L-052
- **⭐⭐ A dangling `:eq:` DOES warn — MEASURED, so it is in the GATED class, not the silent
  one, and that FLIPS L-063's rename caution.** `[M]` throwaway 2-file Sphinx project with
  positive+negative controls, ~10 s: `WARNING: equation not found: <label> [ref.eq]`, EXIT=1
  under `-W`; the live label emits an `href`. L-063's "KEEP + note, renaming risks a silent
  break" was argued from **8 cross-doc `:ref:` citers**; with `:eq:` citers only the build
  catches every miss ⟹ RENAME is safe. Never carry a ref-role caution across role CLASSES
  without re-measuring. → L-070
- **⭐⭐ A POSITIVE CONTROL must be NON-ZERO — its VALUE carries no part of the argument — so
  publish the PREDICATE, never the table of counts.** A frozen control count in a docstring
  rots on the next edit of the files it counts, which is the defect class most correction
  passes exist to repair. Shape that works: a `.. code-block:: python` carrying the exact
  patterns + root + occurrence semantics + its own `assert`s (controls non-zero, subjects
  zero), under a LABELLED section every other site `:ref:`s. Keep the ZEROS — they are the
  falsifiable finding. ⭐ And **RUN THE RECIPE AS PUBLISHED**: extract the block back out of
  the `.rst`, `compile()`, `exec` — its own asserts are the verdict; a recipe that does not
  run is the same defect as a number that does not reproduce, and no build checks it.
  ⭐ Prefer an ENUMERATION to a COUNT (a list can be checked by reading it): my "twelve files"
  was **thirteen**, and my own prose enumeration in the same sentence summed to 13. → L-070
- **⭐⭐ When one pass cites an evidence set TWICE, it mints a TWIN SOURCE — define it once,
  `:ref:` it everywhere.** `[M]` mine: an `.. important::` block naming one "six independent
  spellings" set and a census table's column headers naming a DIFFERENT six, 900 lines apart,
  same afternoon. The published numbers then belonged to the table's partition while the prose
  beside them named the other — so two of six numbers described spellings no reader could see.
  ⚠ Three independent causes were live at once (pre-edit values · partition mismatch ·
  `re.I`+unanchored `redistribut` absorbing every `AngularRedistribution`, 67 vs 56), which is
  exactly why the numbers looked plausible. → L-070
- **⭐ Adopting a reviewer's EXACT patterns is evidence, not concession** — two
  independently-vocabularied instruments agreeing is the acceptance evidence. Keep theirs
  verbatim and ADD what yours had that theirs lacked (here the paraphrase spellings `.reduced`
  and `connection[ -]coefficient`). → L-070, L-067
- **⚠ When a coordinator edits concurrently, the porcelain flag is NOT authorship — and a DATE
  is not a signature.** `git status` showed 5 files I never opened; my signature grep matched 3
  of them on the shared `2026-08-27`. Reading the matched lines settled it. Discriminate by
  CONTENT, with a token only you would write. → L-070, L-056
- **⭐⭐ The CONTROL column of a census table you PUBLISH moves under your own edits.** I drafted
  `sn 36/66/16/66/44/2` pre-edit; my own ⛔ tombstones name the module, so post-edit it is
  `.../67/44/3` — the table would have shipped unreproducible against its own tree. Re-measure
  AFTER the last edit, and prefer the **file list** (stable) to the count for the load-bearing
  universal. ⚠ Same pass: "every consumer lives in `sn/`, `transport/`, `derivations/`" silently
  omitted the module's OWN package. → L-070
- **⭐⭐ For an un-`automodule`'d module the build sees NOTHING — substitute a
  DIFFERENTIAL docutils parse, HEAD vs working tree, counting roles that SURVIVE AS
  TEXT.** `[M]` all 5 code files I edited: `automodule = 0`, `html_pages = 0`. ⚠ Two
  harness traps: walking `dir(module)` reports 89 "problems" that are `dict.__doc__` on
  `__annotations__`/`__dataclass_fields__`; and a *system-message* count is BLIND to the
  silent class — `text~:math:` emits **no message** and degrades to literal text, while
  `**bold**:math:` (which looks illegal) parses FINE. Only `:math:` is testable this way
  (bare docutils does not know the Sphinx domain roles). Verdict: `HEAD = 1, now = 1`.
  → L-078
- **⭐⭐ A residual filter's POSITIVE CONTROL is the pre-edit STRINGS, verbatim — and the
  one that fails is the point.** `[M]` 3 of my 4 controls matched; the 4th
  (*"it must be isotropic"*) did not, because my copula class was `(is|are|being|it's)`
  and the word was **`be`**. Widened to `(is|are|be|being|been|remains?|stays?|it's)`.
  Without the control that site would have read clean. ⭐ And window on the **PREDICATE**
  as well as the subject: the one site the brief missed said *"**the emission** is
  isotropic"* — a section-context back-reference no subject grep can see. → L-078
- **⭐ Strip the IDENTIFIERS before judging a prose claim.** `IsotropicN2N`,
  `isotropic_scattering`, `K_iso`, `assemble_per_ordinate_isotropic` are NAMES, not
  claims; masking them took a windowed sweep from **388** unreadable hits to **98**
  adjudicable ones. Then run a second filter in vocabulary that never spells the subject
  (`doubling`, `two neutrons`, `multiplicity`) and a third that never spells the
  predicate (`no angular dependence`, `single Legendre`, `P0-only`). → L-078
- **⭐⭐ The literal-inside-BOLD class is invisible to `-W`, and there IS a source scan that
  matches the HTML slice exactly.** L-074 said a source regex is unusable — true of the naive
  one (`[M]` **88** candidates, 86 junk). Two constraints fix it: **strip `code-block` bodies
  first** (kills `x**2` / `mu**2`), and **bound the run at ≤200 chars** with a non-greedy
  `\*\*(no blank line)\*\*`. `[M]` **2 hits, 0 false positives, same set the HTML slice
  found**. Both real: `**Why … ``domain`` … .**`. Keep the HTML slice as the oracle; use the
  source scan every edit. → L-079, L-074
- **⛔ …and ASSERT THE SLICE CONTAINS KNOWN PAGE PROSE, or a dead anchor reports "clean".** `[M]`
  `rfind('<section id="manifolds">')` matched **nothing** in the current theme (it emits
  `id="manifolds"` on a different element), the slice came out **length 1**, and the gate
  printed *0 backticks*. Anchor on `role="main"` … `<footer|class="related"|sphinxsidebar`, then
  `assert "<known phrase>" in text and "<a heading you just added>" in text`. With that fixed it
  found **four** defects `-W` could not: 3 × ``**``literal``**`` and one `:math:` inside
  `*emphasis*`, the last **leaking the role name as literal text**. Fixes: split the bold around
  the literal (``**Why …** ``x`` **…**``), and move the role out of the emphasis. Also scan the
  rendered text for leaked `:(math|ref|eq|class|meth|…):`` openers — a second, cheaper tell for
  the same class. → L-080, L-074
- **⭐⭐ COUNT the population your role-filter must cover before trusting its zero.** My check
  matched `:role:`~a.b.C`` only and reported **34** roles clean; the page has **56** — 22 use
  `` :role:`display <a.b.C>` ``. Compare against a count of all `:(class|func|meth|attr|mod|
  exc|data):`` openers, and give the checker a **positive control** (two synthetic dead roles
  must both be reported) so a clean run cannot be a broken scan. → L-079
- **⭐ Two defects YOUR OWN new prose introduces, both `-W`-caught, both mechanical:** (a) an
  italic run interrupted by a role — `*"… (*:math:`X`*) …"*` — gives *"Inline interpreted text …
  start-string without end-string"*; escape the seam: `(*\ :math:`X`\ *)`. (b) **NEVER hand-align a
  simple `===` table containing a `:math:` role** — the role's SOURCE length is not its rendered
  length, so the column arithmetic is invisible and you get `ERROR: Malformed table`; use
  `list-table`. → L-054
- **Beyond AGENT.md's warn-list, two more DO warn:** a `:widths:`/column mismatch, and `ref.ref`
  "*A title or caption not found*" — a bare `:ref:` to an anchor sitting before a PARAGRAPH,
  **including a BOLD RUN-IN HEADING** (`**(2) Title.**  Prose…` looks like a heading and is
  not one; ONE such anchor cost 5 warnings across 4 files and EXIT=1). Fix: promote to a real
  titled subsection — but never open the title with `(1)`/`(2)`, an RST enumerated-list
  marker; use `Correction 1 — …` — or use explicit text `` :ref:`text <label>` `` when the
  anchor legitimately sits above an admonition. ⚠ **`check_docstring_xrefs.py` is BLIND to
  this class** — it gates Python-domain roles, and reported `DEAD TARGETS: 0` while all 5
  were live; only the `-E -W` build sees them. Raw path strings in prose warn NOWHERE.
  → L-002, L-027, L-040, L-055
- **Grep `SyntaxWarning` in the build log too — a case-sensitive `WARNING:|ERROR:|CRITICAL:` MISSES
  it** and it does not bump the exit code. A non-raw docstring with `\Gamma` emits
  `SyntaxWarning: "\G" is an invalid escape sequence` mid-build. Before reporting one in a file
  another agent is editing, `py_compile` the LIVE file — mine was fixed a minute later. → L-048
- **Grep a glyph in `docs/` before importing a marker from a plan — and re-grep, because this
  answer MOVED.** `[M]` 2026-08-14: `⛔` 12 files, `⚠` 17, `⭐` 10, `✓` 10, `✗` 2. All are corpus
  vocabulary now; the old "⭐/⛔ are zero in docs/" reading is retired. → L-048, L-056
- **⭐⭐ A mechanical PORT's warning count is a non-representative sample of its defect count —
  census the target language's delimiter alphabet before fixing warning #1.** RST has no legal
  run of 3+ backticks outside a literal block, so a run-length histogram is a TOTAL census: a
  briefed "handled the bulk correctly, 20 warnings left" was `[M]` **830 mangled delimiters on
  339 lines** + 30 broken spans; the warnings were the ~2 % where the imbalance failed to cancel
  inside a paragraph. One root cause — a LINE-LOCAL `` `x` ``→``` ``x`` ``` regex meeting a code
  span that WRAPS a line — with three surfaces: over-added pair (silent stray backticks),
  one-side-only (warns), neither-side (silent `<cite>`). **The port's SOURCE is the oracle and
  makes the mass edit a PROOF**: post-fix 2438/2443 literal contents and 3648/3653 prose lines
  appear VERBATIM in the `.md`, every exception explained. → L-061
- **⭐⭐ `<cite>` in the built HTML is the smoking gun of a Markdown port** (`default_role` unset
  ⟹ a surviving single-backtick span renders ITALIC, not monospace) — and its sibling: **RST
  forbids inline markup after most chars** (`= . ~ § ↔ *`), which Markdown does not, so a port
  leaves roles that DO NOT RENDER AT ALL. `[M]` two `:math:` roles opening after `~` produced
  `<cite>mathcal{O}(h^{1.3})</cite>` — role dead, LaTeX backslash eaten, **silent at every build
  severity**. Fix is one char (`~\ :math:`…``). Census with `grep -c '<cite>'`. And `\|` added by
  a port is RIGHT in prose, WRONG inside a literal (renders a visible backslash). → L-061
- **⭐ A dead `:doc:` from a Markdown port is usually a PATH-FORM error, not a missing page** —
  MD authors write repo-root paths (`docs/theory/…/index`), Sphinx wants a docname
  (`/theory/…/index`). Check the page EXISTS before rewriting the prose; the warning and the
  brief both read as "points at nothing". → L-061
- **⭐⭐ The WHERE-LIST is the tell that a labelled equation drifted from its own prose.** A
  definition list defining symbols ABSENT from the equation it annotates, and omitting one that is
  IN it, is a correction that stopped one line short. Needs no code and no build — and no build can
  help: the label EXISTS, so every `:eq:` resolves, `-W`/`-n` are silent, and the V&V matrix reports
  the label covered because coverage is keyed to the LABEL, not to what it says. Seen: a 2026-08-02
  fix rewrote the geometry table, the worked examples, the rejection messages AND the predicate
  quoted in the equation's own vv-status comment — leaving the `.. math::` body stating the retired
  claim 8 lines above its own correction. Publish the tell IN the page. → L-056
- **⭐⭐ A dead `:by:` has THREE fates, and the discriminator is the SURVIVOR's state, not the
  retirement's.** Survivor NOT yet declared on this label ⟹ **MIGRATE** (removing orphans the
  equation); already declared ⟹ **REMOVE** (the retirement collapsed two implementers into one);
  already declared **but the equation names arithmetic the retirement RELOCATED** ⟹ remove **and
  ADD the new home**. `[M]` one sweep hit all three (9→9, 6→5, 3→4). ⭐ And the adjudication rule
  when a symbol *lost a capability*: **read what the EQUATION states, not what the symbol lost** —
  a label stating `denom`/`numer` is a BALANCE, so a method that still forms that quotient stays
  declared; the relation it stopped evaluating lives on a *different* label. ⭐ Mirror: a
  relocation that moves a PRODUCT from callee to caller owes the CALLER a declaration, or the
  equation's most specific factors are implemented by nothing. → L-071, L-059
- **⭐⭐ A "single production spelling" ruling is a DECLARATION opportunity with a measurable
  payoff — the label naming the thing is sitting on token guesses.** `[M]` `dd-mm-angular-recurrence`
  carried **32** inferred implementers, every one matched on the token `angular` (a membership
  list of `sn/angular/`); `pole-mm-recurrence` carried 1, via `pole`, a method that mirrors pole
  FACES. Declaring: **32 → 0** and **1 → 0**, `verifies` edges untouched. ⚠ Get the SET right —
  a two-line equation (seed + step) needs the step fn, the batch kernel that writes the seed AND
  loops, **the public exposure its `verifies` tests actually call**, the mesh-bound wrapper and
  the per-cell entry. → L-071
- **⭐⭐ A `.. implements::`/`.. verifies::` DECLARATION is a doc surface whose failure mode is
  INCOMPLETENESS, and declaring is the only thing that stands the guessing down.** Nexus infers
  code↔equation links from shared name TOKENS; declaring ONE implementer switches inference off
  for the WHOLE equation, so an equation declared with 1 of its 2 implementers shows only 1 —
  under-coverage produced by an act that looks like an improvement. Ask *what else computes this?*
  before the first directive (7 of 14 needed 2–4: DD arm + LD arm; forward + transpose; the scheme
  door + the schedule that folds its term). Pre-flight every `:by:` path and label against
  `graph.db` before writing (a bad `:by:` DOES warn, so `-W` gates paths — but only a
  post-rebuild `edges WHERE type='implements'` query proves the COUNT). MEASURE the instrument you
  displace: `[M]` 397 guesses vs 28 true implementers on one page, **21 % recall, 1.5 % precision**,
  and the guess sets for two unrelated equations **identical 23-for-23** — because the matched token
  was the PACKAGE name, so the set is a module membership list that *cannot contain* implementers
  living elsewhere (`loss-rep-LpC`: **0 of 23**). → L-059
- **⭐⭐ Writing the explanation MINTS new guesses — citing a symbol to say it is NOT the
  implementer makes the heuristic name it as one.** `[M]` adding two `:meth:` xrefs while
  explaining three undeclared equations raised their guess counts 23→24/25/24. ⟹ NEVER publish a
  live guess count (quote the frozen pre-declaration measurement or say "re-run"), and know that an
  undeclared equation gets WORSE every time its page is improved. → L-059
- **⭐⭐ "All N node IDs resolve" ≠ "all N `:by:` targets bind" — but a FROZEN prefix list is
  itself a stale claim.** L-060 recorded the resolver as trying the literal string, then
  `py:function:`/`py:method:`/`py:class:` and *"nothing else"*; `[M]` 2026-08-28 REFUTED — a
  `:by:` at a module-level `float` (`py:data:` node) binds and draws no warning. A bare dotted
  name at a `TypeVar` still failed once, so the honest rule is: **a bad `:by:` DOES warn, so the
  `-W` BUILD is the pre-flight** — a hand-written prefix check over-reports. ⭐⭐ And the dual,
  which is the bigger prize: **a RED `-E` baseline's `nexus.directive` warnings ARE the retirement
  site list**, one per dead `:by:`, and they find blocks no label-grep reaches (mine surfaced a
  4th file the brief never named). Read the baseline before the brief. Acceptance becomes `4 → 0`,
  and `directives: wrote N edges` reconciles the count (`[M]` 400 → 412 = exactly my +12).
  → L-071, L-060
- **⭐⭐ An equation with NO implementer keeps its guesses forever — you cannot declare an
  absence.** `[M]` post-pass on one page: 57 directive / **0** inferred on the 32 declared; **166**
  inferred remaining, every one on the 8 that cannot be declared (60 on `operator-solve` alone).
  That residue is the ceiling on what authoring can retire, and it is the argument for a
  machine-readable KIND. → L-060
- **⭐⭐ `no-implementation` has a class that LOOKS declarable: an IDENTITY BETWEEN TWO
  QUANTITIES THAT ARE EACH COMPUTED.** `φ = Mψ = Gc` — `Mψ` is the analysis face, `Gc` is
  `discrete_gram`, and the identity is evaluated nowhere; same for `d_ℓ·G_ℓ = W`, whose
  two factors ship and whose product is never formed (that IS the point — it lets the
  kernel carry one `1/W` scalar). Declaring either side asserts that one of them IS the
  identity. Use `:kind: identity`, say which symbol computes which SIDE, and name what the
  suite measures instead (the CONSEQUENCE). `[M]` 17+16 guesses → 0 on two labels, and a
  *contrast* label went 2 → 0, one of its guesses being an SN solver entry point that never
  touches the faces it names. ⭐ Mirror: re-deriving one equation's `implements::` set is
  where you find another's — an implementer that LEAVES one equation has to LAND somewhere
  (`metric_per_ell` left the adjoint, gained `sh-space-metric`, 3 implementers, 0 declared
  before). → L-065
- **⭐ The `documented` sentinel marks the KIND, not the coverage — a label can honestly sit
  in BOTH matrix lists.** `[M]` `hilbert-adjoint-…` is verified by 9 tests AND sentineled;
  so are its page siblings. On this corpus `documented` = representational /
  face-distinction / literature KIND. Do NOT "clean up" the redundancy from one label — that
  re-categorises a convention and moves a generated artefact. Keep the directive; ADD the
  rationale comment if missing. → L-065
- **⭐ "Implemented by nothing" is a CLASSIFICATION worth a section — and the classes are
  enumerable, which is what an inference cannot know.** `{identity, law, canonical-form}` → NONE;
  `{typing-rule, definition}` → look for a declaration site (a typing rule CAN have a materialized
  carrier — a class, a Protocol parameter list, typed methods; an identity cannot). Three further
  classes seen: **superseded path** (the
  identity stays TRUE, the code that walked it is gone — check for INDEPENDENT retirements, there
  were two) · **notation** (a definition whose arithmetic IS computed elsewhere, for a *different
  operator* — declaring it is a false attribution at `confidence=1.0`) · **declared tag** (a
  `ClassVar[bool]` a human set after doing the math by hand; the implementer of the *criterion* is
  the page). → L-059
- **A not-yet-built symbol is a code LITERAL, never a `:class:`/`:meth:`.** Gate with `hasattr`;
  the same probe flips a LANDED seam to a live role. → L-002, L-014, L-025
- **Plain-text refs are often the page CONVENTION, not a defect** — un-`automodule`'d packages, and
  `:noindex:`-automodule'd ones, are plain-text page-wide. MEASURED: `api/method_of_characteristics.html`
  and `api/discrete_ordinates.html` carry **zero** `id="orpheus.*"` anchors, so `:noindex:` renders
  docstrings but mints NO targets and leaves live `href`s pointing at anchors that never existed.
  Adding an `automodule` there is still worth it for the DOCSTRINGS — just don't expect roles to
  link. Match the page, repoint dead refs to the LIVE path, never half-surface 1–2 leaves.
  → L-002, L-034, L-047
- **`automodule`-readiness is MULTI-gate; "0 `:label:`" is necessary, not sufficient** — it also
  trips on an unregistered role, a short docstring underline, a malformed field-list, a member-name
  collision cascading onto pages you never touched, and a closing role-backtick followed by a word
  char. `-E -W`-build EACH in isolation; if a cluster is unready, automodule only the clean module,
  prose-ref the rest, REPORT the unblocking fix. → L-002
- **Labels are PATH-IMMUNE; `:doc:` is PATH-SENSITIVE.** A moved label needs zero referrer edits —
  the break is CONSUMING PROSE naming the old page (`` see :ref:`X` in :doc:`.../oldpage` ``): the
  link goes to the new page while the prose sends the reader to the old one. Sweep the tree with a
  whitespace-FLATTENED scan (the `:doc:` routinely wraps). → L-024, L-026
- **Any doc-cleanup pass is a free staleness audit** — reading a line to trim it is the only gate
  catching a stale RAW PATH or a stale V&V word. → L-034
- **That gate also OVER-reports — its `getattr` probe is blind to an annotation-only class
  attribute (`ClassVar`, dataclass field), which autodoc DOES publish.** 5 of 30 `orpheus/` hits
  were live; `Field.UNITS` renders a real `href` in a FRESH build. Prove a contested hit with a
  rendered-anchor grep, then LEAVE it and report — never mutilate a true ref to green a gate,
  never edit a gate you weren't asked to edit. Mirror class, genuinely unresolvable and worth
  fixing: with `napoleon_use_ivar = True` an `Attributes`/`Parameters` entry mints NO target, so
  an `__init__`-assigned attribute needs a live `:class:` + a literal — **5 of 24 `docs/` sites**,
  and autodoc coverage will NEVER revive them. Phrase the replacement so the sentence says where
  the value comes from ("the ``scheme`` attribute that :class:`SNMesh` realizes in its
  constructor"). → L-046, L-047
- **In `docs/api/`, dead refs cluster by SECTION: the unit of repair is the retired API SURFACE.**
  7 of 24 sites were ONE section listing 6 factories retired in one commit; the successors were a
  re-LAYERING, not renames, so 6 repoints would have been 6 lies. Read the surviving module's own
  docstring FIRST — a well-retired module states its successor map and tells you whether you owe N
  edits or one rewrite. Expect ~⅓ REWRITE on a deletion-driven sweep. → L-047
- **RUN every doc code block a present-tense sentence promises works.** One opened on an import of
  a module deleted months earlier AND used `np` with no `import numpy`; no build sees either. A
  dead import is the loudest possible dead ref. → L-047
- **A `scipy`/3rd-party role can die by UPSTREAM removal** — `scipy.special.sph_harm` was removed
  in 1.17; the successor `sph_harm_y` has a SWAPPED `(n, m)` order, which belongs in the fixed
  sentence, not just the target. → L-047
- **⭐⭐ RST CANNOT NEST INLINE MARKUP, and the render check is the ONLY instrument that sees
  it.** `**bold naming ``a symbol``**` renders the inner delimiters LITERALLY; the same rule
  (markup may not open after `. * ~ § ↔ =`) kills a role OUTRIGHT, eating the LaTeX backslash —
  `:math:`\mu`` ships as the word `mu`. Silent at every severity, invisible to
  `check_docstring_xrefs.py` (it gates TARGETS, not whether the role parsed). The check: slice
  the built HTML between your new section's first/last distinctive phrases, strip tags,
  unescape, count **visible backticks** and **surviving `:role:` spellings** — both must be 0.
  ⭐ **The shape YOU will write is `**``value``**` in a numeric `list-table` cell** — I did it
  **14 times in one session**, every time to make a negative row stand out. A literal already
  carries visual weight in a table; NEVER bold it. One-line guard before the write:
  `assert "**``" not in text`. → L-069, L-068
- **⭐⭐ A LITERAL renders a backslash VERBATIM — a number in scientific notation is a `:math:`
  role, never a `` ``literal`` ``.** ``` ``1.4\times10^{-6}`` ``` shipped as those characters in
  prose; `-W`, `-n`, the xref gate and nexus `dead_references` are ALL blind. Discriminator: a
  backslash in the cell ⟹ math, not code. And a bare `:ref:` to a section whose TITLE holds
  `:math:` leaks the raw TeX into the link text — check the target's title, use explicit link
  text (silent; the admonition-anchor sibling WARNS). → L-069
- **⭐⭐ Build the render checker with care — both failure directions are its own regex.** Sphinx
  emits display math as `<div class="math …" id="equation-X">`, so `<div class="math[^"]*">`
  misses EVERY numbered equation and reports ~1000 false TeX hits; use `[^>]*>`. The `<head>`
  MathJax macro config is raw TeX too — slice by `<section id=…>` to the NEXT section's id, not
  by prose phrases. ⛔ And do NOT substitute a SOURCE-side regex: `\*\*[^*]*``…` matched
  **26 suspects, 0 real** on my blocks, because `**A** … **B**` is one match whenever no `*`
  sits between them. The rendered page is the instrument. → L-069
  ⭐ Corpus-wide it is a CENSUS not a sample (RST admits neither in rendered prose), `[M]`
  **125** nested runs / 25 pages and **104** dead roles / 28 pages; rank the dead roles ABOVE
  the backticks — a stray backtick is ugly, a dead `:math:` is a MISSING EQUATION. ⚠ Exclude
  `_modules/` (viewcode listings) AND every `_build` page whose `.rst` is gone (`[M]` 12
  orphans carried 76 more runs — a 60 % inflation). → L-068
- **⭐⭐ The render check owes a PROVENANCE step, or it indicts the whole page instead of your
  edit.** A page-wide count fires on pre-existing prose (`[M]` mine: 32 backticks + 8 dead roles
  + 29 `<cite>`, **all** pre-existing). Cheapest primary evidence: **slice the HTML to your own
  section's `id=` and count there** (mine: 0 / 0, plus the tables, code-blocks and cross-doc
  `href`s rendering). To adjudicate the rest, test each offender's source pattern against
  `git show HEAD:<file>` — and when that says "mine", **re-check by LINE, not by string**: a
  source line-wrap (`**independent of\n:math:`…`**`) makes a single-line pattern miss.
- **⭐⭐ `<cite>` is NOT always a port artifact — count both spellings before "fixing" it.**
  L-061's rule is about code spans that should be MONOSPACE. `[M]` the measured-marker is
  spelled `` `[M]` `` **184** times corpus-wide vs ``` ``[M]`` ``` **110** — so the italic
  `<cite>` is the convention, and on the page I edited 21 of the 29 were pre-existing.
  Normalising my 8 would have made my text the inconsistent one. → L-071, L-061
- **⭐ Widening someone else's issue: re-run THEIR instrument, not yours.** #379 owned this class
  at `[M]` 32 runs "in the error catalogue"; running its own grep corpus-wide reproduced the 32
  exactly (the control that the instruments agree) and showed it is 26 % of the total. A
  comment carrying their number + the wider denominator + the exclusions retitles the issue;
  a fresh issue with a different regex would just have forked the count. → L-068

---

- **⭐⭐ KEEP THE PRE-EDIT `-E` BUILD AND DIFF IT — a per-page rendered delta needs no
  provenance argument.** L-068's slice-to-your-own-section and L-069's rendered-page rule both
  cost reasoning; a **baseline diff** (visible backticks + unparsed `:role:` spellings, per
  page, before vs after) is proof by zero. `[M]` it caught ``**``assert``**`` — the exact
  `**``literal``**` nesting L-069 already records, written again — as **8 backticks** on a page
  whose absolute count is otherwise irreducible. ⚠ The `<cite>` column MOVES and that is
  correct: `` `[M]` `` is this corpus's marker and renders `<cite>[M]</cite>`. → L-072, L-069
- **⭐⭐ Before appending to a list, read its own header for a UNIVERSAL your row would
  falsify.** `history.rst` opened *"Every entry below is **merged to main** … a new entry lands
  with its merge hash or not at all"*, and the task was a row for unmerged branch work. Adding
  it silently is the exact defect the discipline exists to prevent; the fix is to repair the
  universal to the convention the corpus already runs (the sibling `operator_algebra.rst`
  table spells it), keeping the strong half — *trust git, never a frozen note* — verbatim.
  An index can contradict itself. → L-072

- **⭐⭐ A SOURCE regex CAN gate nested markup — if it is `re.S` AND set-differenced against
  `git show HEAD:<file>`.** L-074 said only the HTML slice works; the half that is wrong is
  fixable. `[M]` my per-LINE `\*\*(.+?)\*\*` check read **0** while the rendered slice showed
  **4** visible backticks, from a bold run spanning two source lines. The instrument:
  `rx = re.compile(r"\*\*(?!\s)((?:[^*]|\*(?!\*))+?)(?<!\s)\*\*", re.S)`, keep hits containing
  ` `` `, and subtract the same set computed on `HEAD` — `[M]` **1 new** (mine) against
  **46 pre-existing** across two pages, in one command, with no build. Keep the `rfind`-anchored
  HTML slice as the confirming gate; run the source diff FIRST because it is free. → L-076, L-074

## 3. A `:label:` is a V&V edge — grep the matrix before touching it

- **⭐⭐ A DOCUMENTED-sentinel label adds NO test — predict the SENTINEL count, not the test
  count.** `tests/_harness/audit.py` computes `testable_labels = theory_labels −
  documented_labels`, so `[M]` my +3 labels moved `matrix.rst`'s *"**N** labels carry the
  sentinel"* 571 → **574** and the collected total not at all (its +27 was entirely
  code-side, 4 test modules). The dual of L-076's registry-parametrized +1: know which
  registry your label joins. → L-077
- **⭐⭐ A `.. implements::` that TRANSCRIBES the body rots when the body MOVES, and the fate
  is often BOTH-with-roles, not migrate-or-remove.** `[M]` two declarations quoted
  `outer(self.chi, …)` / `ReactionRateFunctional(self.mat_xs…)` after `.chi`, `.sig_p` and
  `.mat_xs` were all retired. Declare the NEW arithmetic home **and keep the old names
  declared as DELEGATIONS, saying so** — a Protocol gate and a production consumer still
  reach the object through them, so dropping them under-declares the equation. ⭐ Predict
  the `directives: wrote N edges` delta and check it: 412 → **415**, exactly my +3; a
  mismatch means a `:by:` silently failed to bind. (Extends L-071's three fates with a
  fourth.) → L-077
- **⭐ A GATE'S DESIGN properties are publishable theory.** A condensation gate's morphisms
  being **hand-built in the test body** (structurally independent, vv L11) and its
  **asserted activation precondition** (a degenerate fixture makes every control go silent,
  so it is REFUSED with its own red row) belong in the theory page — the person editing the
  fixture never reads the test docstring. → L-077

- **NEVER rename or delete a label a `@pytest.mark.verifies(...)` targets.** For a stale equation
  that IS a verifies-target, keep the label and rewrite only the BODY. Run the silent-class grep of
  `orpheus/`+`tests/` FIRST: empty ⟹ safe to rename; a hit ⟹ report the test edge (you don't edit
  `tests/`). → L-003, L-032
- **When an ALGORITHM is replaced, a retired-STEP label is usually KEPT-AND-REPOINTED to a
  conceptual survivor** (reflexively retiring iteration-step labels orphans test edges). Ask whether
  the CONCEPT survives, `.. note::` what it historically named, and retire only a documented-only
  label with no survivor. → L-003
- **PHANTOM verifies (marker whose label exists nowhere): repoint if the equation already carries a
  label, MINT only if the law is prose-stated but unlabeled** (one `.. math::` = one label). → L-003
- **Classify every label you add.** Structural / representational / literature-transcribed ⟹ the
  machine-read DIRECTIVE `.. vv-status: <label> documented` + a rationale comment naming the gate
  (prose status does NOT count — a `--strict` audit regresses). A NEW test's verifies-target ⟹
  leave it un-sentineled; never sentinel to paper over a transient orphan. → L-004, L-036
- **Algebra-of-record SymPy-identity labels are verifies-COVERED, not documented** — foundation +
  verifies COEXIST and produce a real edge. Reserve `documented` for motivating/definitional
  literature with no tight gate; the two together is muddy. → L-039, L-035
- **Orphan adjudication.** WIRE iff an existing test's PRIMARY assertion IS this equation against a
  structurally-independent reference ("would a sign flip red it?"). SENTINEL for exactly three
  shapes: a general/continuous identity whose concrete instance is tested under a DIFFERENT label;
  a native-vs-legacy bit-identity regression; code that does not exist yet. GAP only for a
  load-bearing computed contract with no test anywhere — never manufacture one to look thorough (a
  38-label slice legitimately came out 0 GAP). Sibling-consistency dominates, and a ROOT narrative
  page's orphans are ALL sentinel (its formulas are tested under the METHOD pages' own labels —
  name that downstream gate in the rationale). → L-035, L-004
- **Un-sentineling is verified against the LIVE test, not the brief** (a brief says "wired" when the
  marker is still WAITING). After removing the directive, rewrite — don't delete — its rationale
  comment to a plain note naming the gates, so a future auditor doesn't re-add a sentinel. → L-037
- **When the exit condition FIRES but you don't own the generated artefact un-sentineling would move:
  keep the DIRECTIVE, rewrite the RATIONALE.** Open it `⚠ PRECONDITION EXPIRED … REMOVABLE`, quote
  the superseded text verbatim as history, name the exact gate that now exists. Avoids both silently
  re-categorising a generated table and leaving quoted-false text. A sentinel carrying its own exit
  condition still needs somebody to NOTICE it fired — no build does. → L-049, L-048
- **Dropping a duplicated eq-label ALSO drops its `.. vv-status:`, silently DEMOTING the concept to
  orphan** — and `-W` is blind (the orphan gate is a generated REPORT, not a build check). Move the
  status to the survivor. → L-027
- **Backfilling labels on a derivation-mirror page: BARE dominates** (the checkpoints are already
  labeled; the residue is true intermediates). Fill only the recognizable gap classes: a governing
  eq parallel to a sibling page · an unlabeled object the corpus uses BY NAME · a geometry/sibling
  parallel gap · a paper-numbered eq in the page's established family. 2-of-31 is correct. → L-030
- **⭐⭐ Name the OBJECT, not the paper — an eq-label naming a citation is a latent staleness
  bug by construction** (attributions get retracted, equations do not).
  `bailey-dome-recursion` → `alpha-dome-recursion`. Order of checks: `grep tests/` for the OLD
  name (0 ⟹ no `verifies` edge) · grep the NEW name across code AND the PROSE corpus
  (plan-authoring §1: free can mean *rejected*) · family fit · move the `.. vv-status:` in the
  SAME edit and let `matrix.rst` regenerate (`[M]` exactly one row moved alphabetically; verify
  with `_scan_theory_equations(Path('docs/theory'))` — old label gone from `all_labels` AND
  `documented`, 0 violations). → L-070
- **⭐ Two labels for ONE equation: publish the REGISTER each page owns, do not collapse.**
  `alpha-recursion` (methods page, the `verifies` target, 115 tests) and
  `alpha-dome-recursion` (foundations page, `documented`, 0 tests) state the same recurrence.
  A `.. note:: **Two labels, one recursion.**` naming geometry-primitive vs discretisation is
  the right output; collapsing moves a generated matrix row and re-points markers a docs pass
  may not touch. → L-070, L-064
- **Section-label and equation-label are DIFFERENT namespaces** that coexist under one name with no
  warning — a name owning both is TWO independent single-home checks. Verify with
  `grep -c '^\.\. _X:'` or `grep -c ':label: X'`, never a raw mention count. → L-024, L-003

---

## 4. Retirement & staleness: three greps, and the unit is the THESIS

- **⭐⭐ When a residue census is LARGE and the residue is a SIMPLIFICATION, DECLARE it — do
  not sweep it, do not stay silent.** `[M]` **37** SN-chapter sites still spelled the
  pre-extraction algebra `A = L+C−S−B`. Sweeping all 37 is a numerics adjudication riding
  inside an unrelated docs pass, and it costs pedagogy where the extra term is ≡ 0 by
  fixture; silence leaves 37 false sites. The third option: **declare the simplification at
  the chapter root** (machine header + a `.. note::` naming it *"a deliberate
  simplification, not the shipped member list"*, pointing at the canonical eq-label), fix
  only the sites genuinely describing the SHIPPED object, and report the census with its
  denominator as a scoped follow-up. → L-077

- **⭐⭐ DISCHARGING A SEAM is an edit to FOUR surfaces, and the one nobody edits is the
  section's own CARDINAL NUMBER.** `spaces.rst` stated CS5's third seam in the seams
  bullet, the fences row, a *different* forward promise in the sibling section
  (*"hardens to a direct read when the courier dissolves"*) — and in the opening
  sentence *"**Three** arms … are deliberately not built"*, a universal that silently
  became two. ⟹ after any discharge grep the section for its cardinal number and for
  every forward-looking verb (*lands with*, *hardens when*, *becomes real when*), not
  only for the seam's noun; and past-tense the WHY in place (the reason a gate was
  withheld is the transferable content). → L-075
- **⭐ A `*(in development)*` hatch is a SECOND, independent discharge — and a
  LINE-based grep cannot find it**, because the phrase wraps (`*(in\ndevelopment)*`);
  `[M]` `grep "in development" spaces.rst` = **0** while the page carried two. Run a
  multi-line regex over `docs/`, then reconcile each hit against
  `git merge-base --is-ancestor <hash> main`. → L-075, L-068
- **⭐⭐ DISCHARGING A MERGE-HASH CONTRACT: the blast radius is the BRANCH NAME, not the blocked
  page.** L-067 gave the routing rule (blocked on `history.rst` ⟹ route to a page carrying the
  `*(in development)*` hatch). The hatch is a DEBT and the merge calls it in: every
  `*(in development)* branch ``<name>``` cell goes present-tense-false the same instant, and
  nothing points at them because the dispatch names only the blocked page. ⟹ on discharge,
  `grep -rn "<branch>" docs --include="*.rst"` FIRST (`[M]` 3 cells on 2 sibling pages → `merged
  @ ``<hash>``), then `grep "in development)\*"` corpus-wide to catch a differently-named hatch —
  that second grep also finds the standing EXPLANATORY sentence, which correctly STAYS. → L-068
- **⭐ A DATE in a prose history block is a git question and drifts by one day.** `frame.rst`
  said `2026-08-24 — step F-1` where `git log --date=iso` puts the commit at **2026-08-23**;
  its F-0 sibling four paragraphs up and its S6.0b block below were both right. Written from
  "which session was I in", not from git. The check is free: **when you look a commit's date up
  for a NEW row, diff it against every existing prose block naming that commit.** → L-068, L-064

**Meta-rule: grep the SYMBOL, the full MODULE PATH, and the CONCEPT'S human paraphrase. Then ask of
each hit's ENCLOSING SECTION: "is the PREMISE still true?"**

- **⭐⭐ A FIELD SPLIT is not a rename: one survivor inherits the retired name's LETTER,
  and the corpus symbol usually follows the WRONG one.** `discrete_residual` (Γ, the
  OWED closure) split into `unspent` + `owed`, and the carve gave Γ to `unspent` — so
  every pre-split "Γ" sentence is INVERTED, not merely stale. ⟹ ship a **SYMBOLS
  block** as a labelled admonition with a `Symbol | Field | Meaning | Was` table (the
  fourth column is what makes it a tombstone) plus a grep-able discriminator
  (*"a page that pairs Γ with G⁰, or calls Γ a residual, predates <date>"*). And an
  ANCHOR naming the retired letter is RETITLED to the concept, anchor KEPT, with a head
  `.. note::` saying the name is a fossil — `[M]` all 7 citers on 3 pages then render
  the new title for free. → L-091
- **⭐ A page's own ⭐ argument FOR a special case becomes the derivation of the
  requirement its replacement satisfies — re-home it, never delete it.** The
  equality-short-circuit argument (`σ_x ⊉ O(2)_x`, so reading the fold group against
  the geometry's own domain refuses that domain) is exactly why the successor predicate
  needing no such case is better (`[M]` `O(2)_x ⊆ {e}·O(2)_x`). Put it inside the dated
  ⛔ as numbered item 2, where the falsified design and its replacement's justification
  are one sentence. → L-091

- **⭐⭐ An ONTOLOGY OVERTURN is not a retirement sweep — grep the retired symbol to FIND the
  sites, then read the enclosing ARGUMENT to decide the edit.** The dead-ref half finishes in
  one pass; the load-bearing half has no dead symbol in it. A five-obstruction proof that
  `Carrier[Rep, Role]` is impossible rested on *"the Flux role must make `flux + flux`
  raise"* — premise now false, **conclusion still true**. Deleting it destroys a correct
  proof; leaving it ships a false premise. ⟹ re-derive from what survives, keep the
  conclusion, tombstone the example — and the live tree hands you the replacement: `[M]`
  `AngularFlux` now defines NO `__add__`/`__sub__` while `AngularSourceSink` does, so the
  "changes the arithmetic interface" axis **inverted**. Same shape hit 5 sites on one page.
  → L-063
- **⭐⭐ An eq-LABEL is RETIRED when its NAME encodes the refuted concept and RENAMED when
  only its ADJECTIVE is stale — the discriminator is the label's BODY.** 4 labels, 4 fates:
  body states the retired claim ⟹ retire + repoint every `:eq:` citer; body still true ⟹
  rename to a live name; body untouched by the overturn ⟹ **KEEP + `.. note::` at the anchor
  saying the prefix is a historical artefact** (that one had **8 cross-doc `:ref:` citers**,
  and a cross-doc dangling `:ref:` renders plain text at every severity — renaming buys
  cosmetics and risks a silent break). A stale NAME is not a false CLAIM. ⭐ The retired
  equation still gets SHOWN in the history section — as an **UNLABELLED `.. math::`** with
  one line saying why, so it cannot become an `:eq:` API by accident. → L-063
- **⭐ Check the vv-status bookkeeping in 1 s, and never hand-edit the matrix.** A sentinel
  must name a `:label:` in the SAME file (`tests/_harness/audit.py`), so a rename without its
  sentinel is a hard violation: `from tests._harness.audit import _scan_theory_equations`
  → `.violations` / `.documented`, sub-second, no pytest. `matrix.rst` regenerates at
  `builder-inited`; report the post-regen count (`[M]` 539 → 540). → L-063
- **⭐⭐ Two SKILL files outside a brief's scope can carry the retired ontology — and your own
  repair can IMPORT the falsehood through its cross-reference.** #18's corrected text points
  at `cross-domain-frames` A.1, whose worked example is the retired type. ⟹ flag the
  staleness **inline at the pointer** ("A.1's frame is sound; its example is NOT"), report the
  out-of-scope files, never silently edit them. ⭐ And a REVERSED anti-pattern leads with what
  survives, carries the falsified version verbatim beneath, and ships the checkable test the
  reversal yields. → L-063

- **⭐⭐ A retirement's stale REASON outlives its stale NAME, and only the name is greppable.**
  A carve's blast-radius list gives you the symbols; the load-bearing half is the sentence that
  JUSTIFIES each one. `[M]` two sites 1200 lines apart both read *"…stay on the geometry side —
  they are genuinely geometric"*: one had stale names (found by grep), the other had correctly
  past-tensed names and the SAME false reason (found only by reading). ⟹ after fixing a retired
  name, read the sentence that explains it. → L-069
- **⭐ A retired guard TIER leaves a stale REASON attached to a surviving FACT — replace the
  reason, keep the instruction, and say what changed.** "The composite is re-homed *because*
  the algebra enforces mesh identity" — the re-home still happens; the tier is now space
  CONTENT, so a reader trusting the old reason will optimise the re-home away for a twin
  carrier. Same shape three sites over: a "single-sourced through X" claim where `[M]` the two
  spellings differ by exactly Σw (the plain broadcast vs the normalized section) — the repair
  turns the falsehood into the worked example and points at the section that keeps them apart.
  ⚠ The production helper's own docstring carried the same dead tier. → L-067
- **⭐ A bare plan-internal STEP NUMBER in the corpus collides with a live campaign's.**
  `spaces.rst` said a deferred item was "scheduled for S7" while CS4b's own step S7 landed that
  day and built none of it. Disambiguate at EVERY site, and re-title a FENCE row that has
  fallen ("only the scalar bulk is axis-built" → `[M]` the angular bulk and the trial space are
  too; what is still fenced is the composite and the flat traces). → L-067
- **⭐⭐ When the corpus states ONE object N incompatible ways and each is internally
  consistent, that is not N bugs — a hidden PARAMETER is unnamed.** `[M]` three published
  `Π*` (naked `S₀` / `g_C·S₀` / `S₀∘G⁻¹`), plus one admonition whose EQUATION and PROSE
  disagreed with each other, warning-free for months. All three are the correct adjoint
  under a different coefficient metric; none named its metric. ⟹ do NOT adjudicate — name
  the parameter ONCE in a `list-table` at the point of definition (metric | where it lives
  | the adjoint it induces), then make every site a POINTER into one row, so none can rot
  independently again. The tell is free: two defended statements of the same object that
  disagree. → L-065
- **⭐⭐ The reusable close-out shape for a LATENT defect is THREE shields, and shield 3 is
  the dangerous sentence.** (1) *Consistency is not correctness* — the defining identity
  held at the round-off floor because `.H` is BUILT FROM the stored metric, so it is true
  for every SPD metric and carries ZERO information about which is installed; the
  instrument that can fail compares the metric to something defined without it. (2)
  *Composed chains are immune* — interior metrics cancel. (3) *No end-of-chain consumer
  existed* (`[M]` one grep hit, a docstring). ⛔ Write (3) as **latency, with the clock**
  ("live with the first adjoint consumer, which is why the metric had to be right before
  those land") — reported as reassurance it teaches the next session to defer. → L-065
- **⭐ Extending an ERR entry vs minting a new number: the LANDED MARKERS decide, not the
  narrative.** F-0 became ERR-039's third chapter because the shipped gates already carry
  `catches("ERR-039")` and I cannot edit `tests/` — a new number would silently orphan
  them. Read the catching tests' markers BEFORE choosing. And mark the superseded chapter
  IN PLACE, on the bullet stating the retired formula, not only in the new chapter. → L-065
- **⭐ A three-way SYMBOL COLLISION: rename only the one with NO constituency.** `W` was the
  coefficient space (page convention), the quadrature metric subscript (page convention),
  and the scalar total weight (code + ledger); my derivation needed a fourth, the weight
  MATRIX — the only one with no constituency. Write it `\mathrm{diag}(w)`, keep the other
  three, and open the section with a `.. warning::` naming all three survivors. → L-065, L-051
- **⭐⭐ Before repairing a stale equation, census the CORPUS for a page that already states it
  right — the census does two jobs and both are load-bearing.** (a) It stops the repair minting a
  TWIN: my `keff-as-integrated-rates` fix restated a formula whose SSOT already ships as
  `:eq:`sn-keff-update`` under `:ref:`sn-keff-estimator`` *with* the derivation and the gate — the
  correct shape is keep the equation (a label is an API and must not be false) **+** an
  `.. important::` naming the SSOT and saying which claim THIS page owns, plus "edited there,
  mirrored here". (b) It tells you whether you are fixing an outlier or inventing a convention: on
  the `S → Λ` repair, `[M]` **3 sibling pages + the class docstring already wrote `Λ`** and the
  edited page was the sole holdout — that census IS the evidence the repair is right. Run it
  BEFORE drafting. ⚠ It also surfaces symbol collisions across pages (SSOT writes leakage `L`,
  my page writes `L = Ω·∇`): resolve with a local subscript **plus a note naming both
  spellings** — never silently. → L-060

- **⭐⭐ A brief's SITE CENSUS is a sample; run the windowed CONCEPT grep yourself before fixing
  one.** `[M]` a brief naming 1 + 6 "may be inherited" sites: a regex for the predicate within ±4
  lines of the subject found **18 in one file**, all present-tense-false the same way. Leaving 12 is
  the exact half-fix vv #21 warns about. ⚠ The tell that the file already knew better: the
  CORRECTED framing sat **one line above** a stale sibling docstring. And the same falsehood was in
  the RST too — four places, one of them the page's own **Key Facts** card and one the prose
  wrapping the very equation I was declaring against. Fix shape: keep the equation (the identity is
  TRUE), keep the label (live `verifies()`), correct only the ATTRIBUTION, tombstone the mechanism
  in past tense naming the carve. Retitling is safe iff the section carries no `.. _anchor:` — and
  tree hits in `_build/` are orphaned HTML, not references. → L-059
- **⭐ A ⛔ ruling's quantifier needs YOUR census — the brief greps the NAME, and the sharpest sites
  never bind it.** `[M]` "the only two sites forming `σ_t − σ_s0`" was **four**: one in PRODUCTION
  (inline, `# ¼ σ̂_R h`) and one an MMS **capture** cross-section — identical arithmetic, a material
  datum, coincident only at 1 group. The ruling survived and got stronger; the count did not.
  ⚠ Mirror trap: `sig_r` in `thermal_hydraulics/`+`kinetics/` is a RADIAL STRESS — a short suffix
  collides as badly as a one-letter symbol. → L-059
- **⭐⭐ A math symbol has THREE spellings — ASCII id (`tau_raw`), Unicode prose (`τ_raw`), LaTeX
  role body (`tau_{\rm raw}` / `tau^{\rm raw}`) — and a brief's page list is built from ONE.** So
  it is over- AND under-counted at once: 11 of a briefed 17 pages were false positives from one
  overloaded word (`absorber` is also a MATERIAL; `clamp` is also a GMRES restart), while an
  UNLISTED page carried a present-tense-false bound spelled only in LaTeX. Also grep the NUMBER
  (a stale `[1/5,4/5]` found the page no symbol grep reached). → L-054
- **⭐⭐ When a DIRECTORY moves, census the DIRECTORY and every artefact SIBLING — a grep keyed to
  one filename is blind to the rest of the family.** A `graph.db` census (4 rounds, brief-supplied)
  missed `docs/index.rst`'s `<_nexus/graph.html>` — a **404 on the docs homepage**, since a fresh
  `-E` build leaves no `_nexus/` at all while `graph/graph.html` (627 KB) sits un-linked. Grep the
  parent segment plus each extension (`_nexus/`, `graph\.(db|json|html)`); anchor the SLASH or
  `_nexus` matches every `mcp__nexus__*` tool name (559 KB of noise). ⚠ **A raw relative hyperlink
  is checked at NO severity** — unlike `:doc:`/`:ref:`. And when a static link genuinely cannot ASK
  (no CLI from RST), ship the mirror WITH an RST comment naming the coupling: a second declaration
  that announces itself beats a silent one. → L-058
- **⭐ A now-optional flag is fixed by DELETION, not by updating its value** — updating mints the
  next stale literal. `--db` resolves via `.nexus/config.toml`, so 16 `--db <path>` lines came out
  of one reference for one precedence sentence. KEEP it where naming a file is the example's POINT
  (a scratch/override graph), which makes the flag *better* documented than when it was mandatory.
  Report the resolution ASYMMETRY too: `status` REJECTS `--project-root`, so read-only subcommands
  are cwd-anchored and their "does not exist" error reads as *"never built"*. → L-058
- **⭐⭐ A surviving MODULE does not license a repoint; a surviving CLAIM does — adjudicate a dead
  `:mod:` by SENTENCE TENSE *and* by whether the NAMED OBJECT survived.** `[M]` 4 dead
  `orpheus.sn.spatial.*` in one historical entry: 3 of 4 modules survived a pure `git mv`, yet the
  verdicts split 2 literal / 2 repoint. The trap is two sites in the SAME file with the SAME rename
  and opposite answers — "**Documented in** X" (claim still true there ⟹ repoint) vs "what Phase B
  added … Protocol (X) with three strategies" (Protocol + 2 of 3 strategies since retired ⟹
  literal). Three free corroborations: `git log --diff-filter=D` on the old path; the LIVE tree's
  own prose (it spells the retired names as ``literals``); and the same page already spelling the
  deleted path as a literal 130 lines below. A list where 2 of 3 `:mod:` are live argues against
  literalising the third. → L-061
- **⭐⭐ A CROSS-REFERENCE INSIDE HISTORY IS A CATEGORY ERROR — a body that records the code *as it
  then was* cannot carry a role, which claims the symbol exists NOW at THAT path.** The rule that
  adjudicates every site, and belongs IN the page as a head-of-block `.. note::`: *a name is a
  ``literal`` whenever the sentence around it describes the code as it then was; a role is used only
  where the sentence is a present-tense claim about something that exists now.* So a SURVIVING CLASS
  does not license keeping the role — the surviving CLAIM does: I literalised a live class because
  its sentence stated a `ψ_{1/2}=0` default the SAME entry's later section records as replaced.
  Nothing is lost — the live pointers move to ONE forward-orientation paragraph where their tense is
  present. Corroborate before editing by counting BOTH spellings of each name in the same file
  (`[M]` 5 literals/3 roles, 4/2, 2/2 — the page had already settled on literals and was
  self-inconsistent). → L-062
- **⭐ Before calling a dotted target dead, decide WHICH SEGMENT died** — package / module / class /
  attribute have different repairs and only one is "de-role". `[M]` two brief classifications
  refuted by the same probe error: `…continuous.sood_registry` is a live **package** (a `.py`-only
  check misses it) and `SNMesh.pole_angular_closure` is a live **instance** attribute
  (`self.x = …` in `__init__`; L-053c again). And a dead CLASS under a live PACKAGE may have a live
  HOMONYM elsewhere — repointing is then a false attribution. → L-062
- **⭐⭐ A raw FILE PATH in a literal is the same category error one register down, and no
  instrument sees it** (roles are gated, paths are not). `[M]` in one ERR entry **14 of 14**
  `tests/*.py` paths no longer existed; catalogue-wide **40 of 100**. Fix the class, not the
  paths: state once, at the head, that *which* tests catch ERR-NNN is never prose — it is the
  `@pytest.mark.catches("ERR-NNN")` marker set, read with `nexus errors` /
  `context('vv:error:ERR-NNN')`. → L-062
- **⭐⭐ A FACTORY-TIER retirement sorts by TENSE into three registers, one repair each.**
  **Live guidance** (how to build a field today) ⟹ re-word to the successor spelling, and
  the successor is a CHOICE you must measure — `zeros_on(mesh)` maps to five different
  carrier mints by family. **History** (a landed change's narrative, an ERR post-mortem) ⟹
  prose STAYS in past tense; only a `:meth:` role at the deleted target is downgraded to a
  ``literal`` keeping the exact old name. **Landed-but-written-as-future** (*"when they land,
  the only change is passing X"* — where they landed two campaigns ago) ⟹ re-tense in place,
  flip the section TITLE's verb too if it carries no `.. _anchor:`, and append ONE dated
  `.. note::` with the live mechanism; never delete the bullets, they carry WHY. The third
  register is the costliest and the least greppable — it reads as a plan, not a claim.
  → L-066
- **⭐⭐ A doc that QUOTES a production docstring is making a claim about a FILE, and no
  instrument checks it.** `[M]` a section quoted *"the uniform leaf-side allocator … replaces
  the retired `SNMesh.zeros_*` factories"* — 0 hits in `orpheus/`, and the live docstring says
  the opposite-keyed thing. Grep the quoted STRING, not just the symbol. Same class as L-062's
  raw file paths, one register up. → L-066
- **⭐ When a section has ALREADY been corrected once, add a SECOND dated `.. note::` beside
  the first — do not rewrite it.** Ownership moved twice (mesh → leaf at #346, mesh-key →
  space-key at CS4b S5); two notes give the reader the whole arc, each with its own
  one-command `[M]`. → L-066
- **⭐⭐ An aspirational item refuted ON THE MERITS closes as NOT APPLICABLE, never as
  pending — and a stale REASON on a surviving FACT keeps the instruction and swaps the
  reason.** "Leave it, it's only a plan" ships a plan a future session will execute. Four
  registers, one repair each: present-tense-FALSE ⟹ rewrite + dated ⛔ tombstone quoting the old
  text · aspirational-but-refuted ⟹ ⛔ *closed as NOT APPLICABLE* + the structural reason ·
  stale-reason ⟹ keep the imperative, replace the *because* · correct history ⟹ LEAVE. → L-070
- **A retirement propagates to BOUNDS and to NEGATIVE claims, which no symbol grep sees.** A
  numeric bound is a claim about the retired object (re-measure it from the live producer). And
  grep the retired name inside `independent of|unaffected by|does not depend on` — a negative
  claim about X is exactly what retiring X can falsify (`[M]` "the floor is independent of the
  τ-clamp" → removing it moved the floor 1.8–3.4×). → L-054
- **A retired MODULE whose FUNCTION survives by NAME is a semantic trap, not a repoint** — if the
  survivor now DELEGATES to production it is no longer an *independent reference*, so a mechanical
  repoint yields working links to false COVERAGE claims. Fix: past-tense literals in the history +
  ONE anchored `.. note::` stating the delegation, WHY, and what each arm compares against now.
  → L-054
- **When a carve proves a published diagnostic is Mode-12 BLIND on the shipped rule, the page owes
  a `.. warning::` with the garbage-passes table AND the instrument that DOES discriminate.**
  → L-054

- **A DELETION (unlike a MOVE) leaves a stale PARAGRAPH, not a stale token** — a move leaves a
  true-but-relocated symbol, a deletion leaves a sentence whose premise died. Three shapes recur:
  a file that past-tenses a retirement in one section and PRESENT-tenses it in another; a LANDED
  migration still written as future work ("consumers will migrate in Wave G"); a docstring
  contradicting its own body (a documented `None` fallback the code `raise`s on). → L-046
- **Preserve the WHY; tombstone, don't delete.** Flip tenses, keep the logic. When a finding
  invalidates a published table, add `.. note:: **Retraction (date, Issue #N).**` above it — values
  stay, the INTERPRETATION gets the tombstone. → L-007
- **Retitle to the CONCEPT and KEEP the anchor when the concept survives; RENAME the anchor when its
  name encodes a REFUTED concept** (updating every inbound ref in the same pass, verified in built
  HTML). Keeping the anchor is what makes a retired-note section free — cross-doc `:ref:`s keep
  resolving and auto-pick up the new title. → L-007, L-015, L-040
- **A retirement that REMINTS the freed name onto a different live object makes the name a homonym
  across one commit — disposition each mention by what the PASSAGE DESCRIBES, never by the name.**
  Blanket find-replace is wrong; grep the full module path (the bare name cannot tell live from
  dead) and audit the whole split role FAMILY, not the head symbol. → L-017
- **Per-site ladder for a widely-referenced retired entry point:** (a) behavioral rewrite where the
  section teaches CURRENT API — including DELETING a stale code block rather than symbol-swapping
  it; (b) past-tense double-backtick LITERAL in history/changelog narrative; (c) delete where the
  clause carries no content. Build the LIVE-grounded successor table FIRST — the successor is
  context-dependent and a 1:1 rename is forbidden. → L-019
- **When the deletion is a COROLLARY of a design unification, the SECTION'S THESIS is stale, not
  just the line.** The tell is a stale design stated in the PRESENT tense as live rationale ("by
  design", "what stayed deliberately legacy"). Banner the rationale section preserving its
  reasoning, fully rewrite only the one genuinely-stale-as-current contract, and retitle a moot
  future-work section "(obsoleted)" while KEEPING its `:ref:` anchor. → L-020, L-013
- **Grep the CONCEPT, not only the symbol — a `list-table` COLUMN is a documentation surface with no
  symbol in it** (a brief's 7 literal hits missed 17 cells under a paraphrased header). Dropping a
  column is a 3-part edit (header · every value cell · `:widths:`) verified in RENDERED HTML; prefer
  REPLACING it with the true intrinsic property. And the paragraph that JUSTIFIED the retired flag
  inherits the flag's wrongness — re-verify it. → L-040
- **In `tests/`, a dead xref is a TRIPWIRE for a false CLAIM, not a typo** — a test docstring says
  what the test PINS, so the retirement that killed the ref usually invalidated the sentence too.
  Read the test BODY, then REPORT the false claim (never quietly repoint, never fix the gate). Seen:
  a module docstring advertising a unified-vs-legacy bit-identity chain when BOTH implementations
  were deleted; a pin list whose item asserted the INVERSE of the live gate. Adjudicate four ways —
  REPOINT (majority) · PAST-TENSE LITERAL (a role PROMISES a live link) · REWRITE · DELETE (rare;
  0 of 62 here). A not-yet-built module is a LITERAL — and check its cited PLAN FILE exists too.
  → L-045
- **The brief's successor map is a HYPOTHESIS — run `git log --diff-filter=D` on the old path.**
  "X now lives at Y" hid a DELETED legacy class whose replacement merely reuses the name; same name,
  different object splits the sites into history-literals and a rewrite. → L-045
- **A dead ref can sit on a claim a rename INVERTED, not merely moved** — a published code block
  showed a `cast`-based helper whose live body has no cast and owns the guard the prose credited to
  the CALLER; a repoint alone leaves two falsehoods with a working link. Read the live body, then
  re-state the mechanism. → L-047
- **When a page has an OPEN owner issue: fix the dead refs + the MEASURED adjacent falsehoods,
  leave the issue's genuine rewrite item, and comment with a measurement table AND the residue's
  CORRECTED path** (#286 named a page that no longer exists). Neither "defer, it's theirs" nor
  "annex it". → L-047
- **A retirement can DEMOTE a gate's claim class without touching the test body.** When a rewire
  points a comparison at the successor, re-ask "are the two sides still INDEPENDENTLY produced?" —
  if the survivor CALLS the other, the gate became a pass-through check and every doc crediting it
  must be re-scoped (name the real pin). The tell in a diff is a variable still called `legacy`
  beside a brand-new API. → L-044
- **Doc-only "retire the false promise": keep the DECLARATION, make the CLAIM true** — state the
  measured present, how production reaches that information today, and the phase that fills it.
  → L-041
- **Replace an unfalsifiable inventory sentence with a MEASURED table** — "each subclass overrides
  these where applicable" is prose over a lattice computable in ten lines. → L-041, L-042
- **Give each retracted consequence its own `**Disposition:**` — a retraction can INVERT a claim,
  not just kill it** (one conclusion flipped to the opposite type-tag once the domain narrowed).
  → L-042
- **A phase-N doc pass leaves phase-(N−1)'s falsifications behind — audit the PARAGRAPH FAMILY, not
  the commit's diff.** A correctly re-typed section three screens from a sentence the PREVIOUS phase
  falsified ships a self-contradicting page. → L-042
- **⭐⭐ Fixing HALF a claim in one file is worse than fixing none — after repairing a section,
  grep the WHOLE FILE for the retired predicate's spellings and adjudicate every hit by tense.**
  A brief scopes you to a section; the same falsehood routinely survives three screens above it,
  and a self-contradicting page is citable for EITHER sentence (vv #21's aggravator). Seen: the
  repaired selection §, with the upstream § still opening "quadrature selection therefore reduces
  to a containment check". Re-scope the survivor to what the equation ACTUALLY is (an order
  relation), add a `.. warning::` saying it is NOT the gate, keep the label + `vv-status`
  untouched (it was `implements`-cited from production). Grep list = every spelling of the retired
  symbols plus the stage COUNT. → L-056
- **⭐ A tombstone may only assert what YOUR page controls.** "…and the module docstring said the
  same until <date>" is false the moment the other file is fixed — or reverted. Write a twin's
  history in the past tense of the CLAIM ("the promise was minted twice"), never of the file's
  state. → L-056
- **⭐ A stale FORMULA can be right on a biased subset of the grid, so a spot-check CONFIRMS it.**
  `max(3, N−1)` matched the measured level-symmetric degree at S2/S12/S16/S18 and missed at
  S4/S6/S8/S10/S14 — and S12 is the order the stale frontier itself made salient. Cite the
  SWEEPING gate, and fix a drifted number with a `:ref:` to the SSOT, not a fresher copy: the
  producer discovers it by construction, so any second copy is the thing that drifts. → L-056
- **FLAG, don't silently rewrite, adjacent SUBSTANTIVE staleness.** Repoint-in-passing is correct;
  behavioral-rewrite-in-passing risks minting a NEW false live claim (worse than a dead ref to true
  history). Exception: fix a refuted-claim survivor on a line you are ALREADY editing. → L-007,
  L-014, L-018
- **When a doc describes a DEFERRED seam that since LANDED, verify the SHAPE that shipped — don't
  just flip "deferred"→"done".** The change routinely closes the seam by a DIFFERENT mechanism;
  separate the surviving conclusion from the stale PREMISE, and re-derive why it still holds.
  → L-007
- **A retirement leaves THREE tense classes, not two — the third is a falsified PREDICTION.** Beyond
  present-false (repoint) and past-history (de-role to a literal), future-tense prose written while
  the replacement was a plan names **a MECHANISM, a HOST/TYPE and a PHASE**, and a landing can
  falsify any subset independently — check each against shipped code. Seen: "the type that WILL host
  it exists — `ScalarTraceSpace`" (shipped host was a NEW ladder tier the live docstring explicitly
  distinguishes from it) and "that is phase B5" (landed at G6.3, by factoring not by the predicted
  `u⊗v` typing). Preserve the prediction and `.. note::` WHY it didn't hold — more informative than
  the corrected sentence alone. → L-049
- **A stale-status blast radius is the WHOLE page.** Grep every future-tense/blocked token
  (`blocked|not built|not yet|pending|in flight|future seam|lands with`) — a brief naming 3 sites
  had 7. A "the one remaining unbuilt X" sentence must RE-POINT to the still-unbuilt sibling when X
  lands, not just drop X. → L-037

---

- **⭐⭐ A VOCABULARY retirement has FOUR classes, and tense separates only two.** Beyond
  *update the live vocabulary* / *keep period history* sit: an **ADDRESS** (a section anchor or
  eq-label carrying the retired word — KEEP, and say why: a cross-doc `:ref:` miss is silent at
  every severity) and a **genuine referent** (Hébert's *Carlson coupled-pole* seed, a sphere's
  polar cap, μ = −1 — the word still denotes the thing). `[M]` on one sweep: **14 updated · 32
  period-history · 9 address · 3 genuine**, plus 7 lines deliberately ADDED naming the old
  spelling to record the rename. ⭐ And check the anchor's own section first — a mature page has
  usually already ruled on it once, for a different word, and **extending that note beats
  minting a second caveat**. → L-072

- **⭐ A section RENAME is cheap when you count citers FIRST — and the renamed section's own
  `.. note::` is what makes a stale pointer diagnosable.** An anchor whose NAME encoded the
  refuted mechanism (`...-dense-refusal`) had `[M]` **1** cross-doc citer and **0** in
  `.claude/`/`scratch/`, so L-063's silent-cross-doc-break caution did not bind; renamed with
  its citer in the same edit, and the note records the old name + *"a stale pointer renders as
  plain text at every severity; if you meet one, it predates P7."* → L-076

## 5. Page surgery: slice programmatically, assert before writing

- **⛔ A mid-task scope REVOCATION on a file you already edited: revert by RE-EDITING, prove it
  with `git diff --quiet -- <path>`, and publish the backed-out patch in your RETURN.** Never
  `git checkout`. Afterwards the file may show Modified again — that is the concurrent editor, so
  discriminate by grepping YOUR OWN distinctive strings, not by the porcelain flag. The addendum
  named 2 of the 4 sites I had found, so the return is the only place the other 2 survive.
  → L-056
- **An enumerated list starting at `0.` is legal** (docutils sets `start="0"`, INFO-level only,
  suppressed at Sphinx's default verbosity) — use it when the code numbers its stages 0..N, so a
  runtime message names the paragraph that explains it. Probe with `publish_doctree` first.
  ⚠ And never wrap quoted stale text in `*…*` when it contains `:math:`/`:eq:` roles — docutils
  does not nest inline markup; use the page's own ``⛔ X read Y until <date>`` idiom. → L-056
- **Never hand-retype a large block.** Read → slice → `"".join` → write, with guard-asserts on the
  LIVE file's boundary strings and lengths, and ALL structural asserts run on the in-memory result
  BEFORE any write (a failed assert then leaves the tree untouched — no `git checkout` recovery
  needed). A machine splice cannot mis-transcribe. ⚠ **A red guard may be the GUARD's error —
  diagnose WHOSE failure it is before touching content** (`vv` #4's VERIFY sharpening, turned on
  your own instrument): a `len(out) < len(src)` guard fired on a splice that legitimately GREW,
  while an earlier assert in the same script had already caught a real miscount — that positive
  control is what made the false red cheap. → L-012, L-022, L-023, L-026, L-058
- **⭐⭐ PROBE docutils, never reason about it — and stand up a stub-directive harness so you
  iterate in 1 s, not 4 min.** I predicted a warning came from *emphasis ⊃ inline literal*: WRONG
  (that is silent and renders raw backticks); *emphasis ⊃ **strong*** is what warns. One
  `publish_doctree` call with 6 one-liners settled three entries at once (`key=``x``` warns,
  `key=\ ``x``` is clean; `γ_-` errors, `γ\_-` is clean; a nested list needs a blank line, a `+`
  mid-paragraph does not). Register `error-entry`/roles as pass-throughs and re-check a 5790-line
  file in under a second. **Markdown discriminator for an indented block:** blank line before ⟹
  a real code block (⟹ `.. code-block:: text`, mandatory if the body holds a `*`); no blank line
  ⟹ a lazy paragraph continuation (⟹ blank lines around it → block quote). → L-061
- **⭐⭐ When the edit is confined to ONE section of a big file, the strongest guard is BOUNDARY
  BYTE-IDENTITY:** `src[:i] == out[:k]` and `src[j:] == out[m:]`, with `i/j` and `k/m` the section's
  own delimiters. Two lines prove the other 78 entries of a 5800-line catalogue are untouched — far
  stronger than any per-edit count, and it makes a 22-site sweep defensible. Pair it with an exact
  `len(out) == len(src) + Σ n·(len(new)−len(old))` arithmetic delta per replacement table. → L-062
- **⭐ Run a SELF-CONSISTENCY pass on prose YOU authored before the first build, not after.** New
  prose that DECLARES a rule must obey it: I built four times because I kept finding my own note
  violating its own convention (one live class left a literal), over-claiming a successor, and
  leaving one paragraph ragged. Ask of every name in your new text: *which branch of my stated rule
  does this take?* — then build once. → L-054, L-062
- **⭐ Guard a bulk DELIMITER edit with `src.replace('`','') == new.replace('`','')`** — proves
  only backticks moved — plus an exact char-count delta and unchanged line count, all asserted
  BEFORE the write. That is what makes a 415-site blanket edit defensible rather than reckless.
  → L-061
- **⭐ A uniqueness guard over LABELS or TITLES compares EXACT LINES, never substrings.** Two of
  mine fired (before any write, so free): `count(":label: operator-apply")==1` fails because it is
  a substring of `:label: operator-apply-transpose`, and `count("Development history")==1` fails
  because an `.. admonition:: Development history — …` sits 1000 lines above the section.
  Eq-label families are BUILT by suffixing (`X`, `X-transpose`, `X-section`), so the prefix
  collision is the normal case here. Use `sum(1 for l in lines if l.strip() == …)`. → L-060
- **⛔ The directive-body placement rule RE-BIT, twice in one session** — it is easy to know and
  easy to violate, because you write the directive thinking about the EQUATION, not the sentence.
  ⟹ **after writing any directive with a body, read the sentence that spans it out loud**; an
  opener of `where …` / `and …` / `with …` / `consumed at …` is the tell. → L-071
- **⭐ A directive whose BODY renders needs a placement rule, or 50 of them land mid-sentence.**
  `.. implements::`/`.. verifies::` with a body emit a plain `<div class="docutils container">` —
  visible prose, no marker. Rule: **after the `.. math::` block, unless the next paragraph is a
  grammatical continuation of the equation's sentence** (`where …`, `so …`, `with …`, `and
  identically for …`) — then after that paragraph. Encode it as a per-label `skip` flag and
  PREVIEW before writing; open every body `**Implemented by** …` so it reads as an annotation.
  → L-060
- **⚠ An f-string mangles LaTeX braces in the header YOU author, into valid-but-wrong LaTeX `-W`
  never sees** (`A^{-1}` → `A^-1`). Strongest defense: author heads/intros/pointers as pure literals
  via the Write tool so no Python string layer touches math; then grep for bare `^-1`. → L-026
- **Locate by STABLE TITLE, never by the brief's line numbers — and prove contiguity by counting ALL
  H1 underlines in the range, not just anchored ones.** An ANCHORLESS sibling H1 (typically a prior
  split's leftover `:doc:` pointer stub) is invisible to the anchor-grep the brief's author used, so
  the range overshoots. Endemic to multi-split campaigns. → L-026
- **A cross-page theory MOVE is ref-safe if you KEEP the labels** (move, don't copy — defined exactly
  once). Only `:doc:` needs fixing (toctree + every `:doc:old`), plus now-intra-page
  `(:doc:sibling)` parentheticals, which lie without warning. → L-022, L-026
- **Splice mechanics that broke builds:** a slice ending in content, joined directly before the next
  `.. _anchor:`, GLUES the anchor to the preceding paragraph — the label silently fails to register
  and referrers report "undefined label" (NOT "duplicate") though grep shows it at column 0; join
  parts with `\n\n`. Re-nesting under a deeper parent demotes every migrated underline one level,
  LENGTH-PRESERVING (detect an underline as a col-0 all-one-marker line whose previous col-0 line is
  a plain title). Removing a middle H1 while keeping its trailing H2 AUTO-REPARENTS the H2 — verify
  no title-level SKIP and that the new parent is intended. → L-022, L-026
- **When the brief says "relocate to page X" and the CLOSE READ shows the content is already
  canonical on X, the action is DE-DUPLICATION (Cardinal Rule 2), not relocate+merge.** Replace with
  a `:doc:` pointer preserving the conceptual bridge, merge nothing, FLAG the inversion (the brief
  was built on a partial read). A FOLD is a MOVE — it re-exposes every symbol reconciliation the
  source had. → L-027
- **Prefer an additive prose ROADMAP + `:ref:` to the SSOT over copying a table (the twin) or
  relocating a double-duty section** — and first verify the gap is REAL (the presence of *a*
  taxonomy is not the presence of *this* one). → L-029
- **Metadata relocation, not deletion:** strip campaign provenance (hashes, phase labels, dates)
  from a high-traffic invariant section INTO the changelog, KEEPING invariants, eq-labels +
  vv-status, active gotchas, issue-`#N` refs and numerical data. Map each item to a destination
  FIRST — a dated milestone with NO changelog home keeps its provenance inline, flagged. → L-028
- **An overloaded-symbol sweep: inventory every MEANING of the letter first, classify each site
  mathematically, `replace_all` only unambiguous multi-char strings (enumerate spacing variants),
  targeted-edit the rest, then re-classify EVERY survivor in a final grep.** Flag same-letter
  collisions rather than renaming out of scope. A NEW page assembled from multiple sources is the
  prime site for a WITHIN-document collision the build cannot see — hunt every reused glyph and
  subscript the rarer meaning. A section RENAME can also stale an in-file back-reference: grep the
  file for the OLD heading text after a retitle. → L-011, L-025, L-034

---

- **⭐⭐ A REFUSAL BECOMING A CAPABILITY is its own arc — five moves, and move 4 is the one
  that stops the over-read.** (1) KEEP the diagnosis, past-tense only the verdict — the
  impossibility table is *why the repair was possible*, and it is unchanged. (2) Publish the
  refusal era verbatim under a ⛔ **with the sentence saying why it was correct at the time**.
  (3) SPLIT the recorded debt: which half landed, which remain, and *what the landing changed
  for them* (here: the legs now have exactly one metric arithmetic to wrap). (4) Say what did
  NOT ride along, with its own measured section — a correct metric does not buy every identity
  that mentions a metric. (5) NEW ERR **chapter**, not a new number: the landed gates already
  carry `catches("ERR-039")`, so a new id would orphan them (L-065). → L-076
- **⭐ A changelog's chronological ORDER is per-page — check the dates before placing, in two
  lines.** `spaces.rst`'s history `list-table` is REVERSE-chronological; `frame.rst`'s prose
  blocks are FORWARD. I placed a new block right after the entry it tombstones (natural, wrong)
  and caught it with `re.finditer(r'^\*\*(\d{4}-\d{2}-\d{2})', t, re.M)` then
  `== sorted(...)`. Free, and the only thing that sees the mistake. → L-076

## 6. Match the doc SHAPE to the event class

- **⭐⭐ A LEDGER GAINING A FIELD splits across two pages by REGISTER, not by size.**
  The point-set/group page takes the THEOREM (the new predicate derived off the page's
  own decomposition equation), a row in its existing *one body per question* table, and
  the measured admission grid; the ALGORITHM page takes the ledger, the SYMBOLS block,
  the per-geometry derivation with its coordinate conventions, and the worked examples.
  Each cites the other once. `[M]` zero new eq-labels — both existing `:eq:` APIs
  re-worded with their labels kept, so sentinels did not move. → L-091

- **⭐⭐ A KERNEL CHANGING HOUSE (a module split + a receiver↔argument rename) splits into
  TWO registers, and putting both in one place is the twin.** The kernel's MATHEMATICS
  stays where its argument already flows (three conjuncts, in the chapter that derives the
  normaliser criterion); the module's own section owns the BOUNDARY — why the verbs are the
  measure's, why no façade, the call-site proof of one closure, the numerical evidence — and
  goes on the page that owns the OTHER operand. `[M]` no `automodule` exists for the sibling
  modules, so the new one gets none and `:mod:` renders plain text by page convention.
  ⭐ Four moves that carry the weight: **(a)** "ONE closure" publishes as a CALL-SITE COUNT
  (`[M]` 1 caller / 3 callers, and the ambient default REMOVED — *a default nobody uses is a
  second code path*), never as prose; **(b)** an architectural step that is right and has NO
  shipped discriminator gets named as inert WITH its denominator (`[M]` the chart match ==
  the ambient-on-barycentres match on **1027 of 1027**; what moves an answer is reading the
  barycentres, not the chart) — `vv` #19 at the design tier; **(c)** the REFUTED variant is
  reproduced on a renamed shadow package (`[M]` 10/10 vs **3/10**, ~1 min, no production file
  touched), and the publishable half is the three SURVIVORS, one of which is `import orpheus`
  itself; **(d)** a section NAMED after the thing that no longer exists keeps its label and
  is RETITLED (a bare `:ref:` renders the target's title, so every citer improves) with the
  falsified claims verbatim under a `⛔` in their own subsection — and its citers are a blast
  radius: 3 of 6 were themselves present-tense-false. → L-090
- **⭐⭐ A BRANCH-BECOMES-ONE-FORMULA carve (N per-family arms collapse into one
  derivation output) has a five-move shape.** (1) The general statement gets ONE labelled
  equation in ONE home — the chapter where the object's own argument already lives — and
  every other site POINTS at it. (2) Retitle the section whose title states the claim the
  carve refutes ("one per catalogued family"), keeping the anchor. (3) The retired arm
  gets a `.. note::` saying WHAT WAS LOST, precisely: here the section lands ON the base
  and the projector does not, `[M]` no shipped consumer needs that, and the section's
  IMAGE survives in another field — so what retired is the *map into it*. (4) A name with
  three generations is ONE bullet list closing on the transferable rule (*a name that must
  be qualified per argument is a disjunction wearing a noun*). (5) The Mode-12 blindness
  gets a LABELLED subsection with its three consequences NUMBERED, because it constrains
  the gates rather than caveating them — and with the discriminator's size on BOTH sides
  (`O(1)` ambient vs **exactly zero** through the chart). → L-089
- **⭐⭐ A NEW LAYER nobody owned gets its OWN PAGE, and the decisive argument is
  self-undermining-if-homed-elsewhere.** Documenting level 1 of a three-level stack: a section
  titled *"a function space is not a domain"* **inside the function-space page** re-commits the
  conflation it exists to end. Two supporting grounds: three consumer pages, none subordinate
  (SSOT, not twin); and the host page was already 3871 lines. ⭐ **Manage the twin risk
  actively** — my first draft restated `Funk–Hecke`, which two pages own; rewritten to own only
  the register that was `[M]` **0 hits** corpus-wide (Gelfand pair / double coset), opening
  *"Edited there, consumed here"*. Wiring: toctree slot by DEPENDENCE (a measure needs a
  manifold ⟹ before `discrete_measures`), one Key Facts bullet + one seam row + a `related:`
  machine-header key on the sibling, a forward pointer (never a past-tensing) on the page whose
  claim is **still true of what ships**, and a pure ADDITION to the ERR entry. ⚠ Name the pages
  in an index row — my *"underneath both of those"* dangled onto the wrong neighbour. → L-079
- **⭐⭐ For a MINT WITH ZERO CONSUMERS, the ⛔ *this is a capability, not a fix* clause belongs
  in Key Facts, the seam table's FIRST row, AND every page you touch.** `Manifold` ships and
  ERR-080 stays open; a reader who meets the refusal predicate without that clause concludes
  the defect is repaired. State it three times, each with `[M]` (zero importers; `Space = str`
  still at `measure.py:111`; the `xfail(strict=True)` gate still red). → L-079
- **⭐⭐ Publish an engine/ruling's COMPLIANCE as a FRACTION, and audit the seed against the
  spec.** D0.1's falsifiable form is *"could an engine populate these fields without a new
  type?"*; `[M]` `dataclasses.fields` says the procedure emits 8 outputs and **6 are slots**
  (the chart ships only as its codomain; the pushforward measure not at all). *A ruling whose
  compliance is claimed but not counted is not checkable* — one call. → L-079
- **⭐⭐ Look for the lookup the tree ALREADY performs — it is the cheapest evidence a mint is a
  RE-TYPING, and it hands you a seam.** `AngularSymmetry.support` predates the `Manifold` mint
  and already catalogues `S²/G⁰` in the string vocabulary with the same refusal shape. `[M]`
  three rows: `SO2` both answer and **agree**; `Trivial` registry answers / catalogue **raises**
  (a real gap); `Oh` both raise. One table = re-typing evidence + a measured gap + the Pattern-2
  twin the migration must collapse. Found by reading the sibling page, not the brief. → L-079
- **⭐⭐ A ROUTE RE-POINT (the datum is unchanged, only WHERE it is fetched) earns its own
  LABELLED doctrine section — because every value gate over it is `X == X`.** `[M]`
  `op.angular_axis.generator is quad` for the very quad the factory was handed, i.e. the
  same object the retired courier held, so a before/after value comparison is green under
  a correct re-point AND under one that silently kept the old path (`vv` #19, moved from a
  *metric* to a *data route*). Publish: the `X == X` observation, the **decoy** instrument
  (a generator carrying different data behind an identity-EQUAL axis), and the per-decoy
  admissibility table. → L-075
- **⭐⭐ A DECOY CATALOGUE is a statement about TWO contracts, and the one-line attribution
  in the test helper was wrong.** `[M]` the α-dome guard refuses only the **roll** (its
  contract is `Σ w·µ = 0`, which scale/negate/reverse all preserve to ±5.6e-17); negation
  and reversal die one tier later at the closure's **P3** τ∈[0,1] guard; a weight decoy is
  order-dependent (admitted N=2, refused N=4/6/8). Publish the MECHANISM behind each
  measured floor, never the number: the cylinder's `8 of 12` is *4 ordinates have
  `mu_x == 0.0` and `0.9 × 0 = 0`*; the roll's `4 of 12` is *the level |µ| sequence is a
  palindrome*. A floor with a mechanism cannot drift silently. → L-075
- **⭐⭐ For a MODELLING-TRUNCATION correction the shape is ONE anchored
  `.. warning::` carrying the whole measurement set, and everything else POINTS.**
  Homes, one clause each: the **physics** home (the reaction's own section) gets a short
  `.. important::` before any algebra; the **data-layer** home gets the truncation
  recorded where the drop happens (at the displayed record structure, not in a "future
  work" list); **Key Facts** and the **machine header** get one clause each, because
  those are what a reader quotes. ⚠ The anchor sits above an admonition ⟹ every citer
  needs EXPLICIT text (`` :ref:`the truncation warning <label>` ``), and you verify the
  `href`s in the built HTML, not the warning count. A class whose NAME encodes the
  truncation gets an **"On the name"** paragraph — check the sibling class first, the
  precedent usually already exists. → L-078
- **⭐ A correction sweep must not acquire a SECOND SUBJECT.** Chasing one false claim I
  found a different one on the same channel (a section saying (n,2n) is not extracted and
  the balance is 1-in-1-out; `[M]` false at 3 `file:line`s). Fix the claim you were sent
  for; REPORT the neighbour with its proofs. → L-078
- **⭐⭐ A CHANGELOG ROW for a big merge groups by THESIS, never by the plan's phase labels —
  and the page's own precedent settles one-row-vs-many in one grep.** `[M]` `history.rst`'s #280
  campaign holds SIX rows sharing one merge hash, so per-milestone rows are the convention and
  the `Where` format is `` `<step>` (merged @ ``<merge>``) ``. The campaign's step boundaries cut
  ACROSS subjects (one session landed a field-layer step and an operator-layer step), so five
  phases became five theses instead. ⚠ Strip plan-internal tokens on the way in — my draft
  carried *"the standing R2 hazard"* and *"a `§6b` call-site set"* (a plan risk label and a
  rules-file section, both colliding with live campaigns' numbering); the corpus says what the
  thing IS. → L-068
- **⭐⭐ When a LATER row in the SAME merge overturned an earlier row's MECHANISM, tombstone in
  place and name which HALF survived.** Present-tense would ship a falsehood; deleting destroys
  the reason the correction happened. Shape: the row states what it did, then
  `⛔ **Superseded the next day by <step>** (the row above): … The amendment's *demand* stands
  unchanged — it is what made the misplacement visible.` In a reverse-chronological table
  *"the row above"* is a correct pointer. (`plan-authoring` §3, landing in the corpus.) → L-068
- **⭐⭐ A DIALECTICAL SEED PAGE (a design dialogue CONVERGED, first slice shipped) is NOT the
  9-step close-out arc** — that arc is for a CLOSED "cannot work". Order: Key Facts *carrying
  the doctrine's one-line discriminator tests verbatim* → the taxonomy → the theorem → **the
  doctrine dialectically** (question → v1 REFUTED → v2 REFUTED → standing → retrodictions) →
  fences per phase → dev history. ⭐ Give each refutation its own `.. admonition:: ⛔` **titled
  with the REFUTING QUESTION, not the verdict** — both refuted versions are *almost* right, so
  the question is the transferable content and a reader who meets only the final statement
  re-derives v1 within a week. ⭐ And say explicitly what the doctrine does to the tension it
  settled ("it does not pick a winner — both prior rules are right about different clauses");
  otherwise the next reader hunts for the loser. → L-064
- **⭐ A SELF-CONTRADICTING Key Facts block, twelve lines apart, and the truth is NEITHER
  pole.** `frame.rst` promised `Galerkin ⟹ Π* = R` and, 12 lines below in the same admonition,
  `M* = R/W`. Post-F-0 the promise is that the adjoint re-synthesises on the TRIAL basis
  (`M* = S₀∘G⁻¹`, a *canonical* dual) — the metric stays. Fix at BOTH poles, name the ERR
  numbers the bare form is, and cite the standing counter-example. → L-067
- **⭐ Changelog ROUTING for an unmerged carve: `methods/sn/history.rst` contracts "merge hash
  or not at all" and BLOCKS you; `spaces.rst` / `field_algebra.rst` / `operator_algebra.rst`
  each carry the *(in development)* hatch.** Route the entry to the page whose SUBJECT moved
  AND which permits the hatch; report the blocked row ready-to-paste. → L-067, L-063
- **⭐⭐ A RETRODICTION / confirmation table is `plan-authoring` §2's aspirational-row trap moved
  into the CORPUS — and it costs more here.** A table headed by a property of the tree reads
  ENTIRELY as a survey of what IS, so one unbuilt row is indistinguishable from the observations
  and, once found, discredits every row. ⟹ **STATUS column, in the row** (`[M] ships` vs
  `⛔ NOT built — a prediction`), never prose above or below; and head it *"rows the doctrine was
  NOT built from"* (the real epistemic claim) rather than *"layouts the tree ships"*. Caught in
  my own draft by counting my own universal. → L-064
- **⭐ Citing an SSOT: name the REGISTER your page owns, not just the fact** — the same fact in a
  different register is not a twin. `frame.rst` owns the counting measure in the MEASURE register
  (`w_g = 1` vs `Δu_g`, Hébert-derived, rate-preservation-gated); my page owned it in the METRIC
  register (`G_E = I` ⟹ `V ≅ V*` ⟹ adjoint = plain transpose ⟹ construction refuses weights).
  Derive it a third way as UNLABELLED math, cite the SSOT's label, open with
  `.. important:: … Edited there, consumed here.` **Net new labels on a 1158-line page: ONE.**
  → L-064
- **⭐⭐ A STRUCTURAL claim ("X is not merely unmigrated, it can never apply") publishes as an
  IFF with NUMBERED conditions + a per-family adjudication table + a `.. note::` "what WOULD
  change this answer".** The last is what makes it falsifiable rather than an assertion, and it
  is the only thing that pre-empts the "so it's just not built yet" re-reading. Instance: the α
  dome is needed iff (1) an angular unknown survives discretisation indexed, (2) in a LOCAL
  ROTATING frame, (3) with its derivative COLLOCATED — MoC fails 2, CP and MC fail 1, and a
  DG/FE-in-angle scheme would fail 3 and need a different object. → L-070
- **⭐⭐ An UN-WELD doc's load-bearing content is the FORCING, not the twin.** A reader who
  takes a Pattern-2 twin for carelessness re-introduces it. Grep the layer contract and quote the
  forbidden edge: `[M]` by AST, `FORBIDDEN_EDGES["transport"] = L3_PACKAGES`
  (`tests/test_layer_imports.py`, `foundation`-gated per module) means an L2 scheme **could not
  call** the L3 closure that owns the relation — it could only re-spell it. That converts
  *"someone duplicated this"* into *"the architecture manufactured this"*, and it is the only
  framing under which the repair reads as moving a RESPONSIBILITY rather than deleting a copy.
  ⭐ Pair it with the honest scope (the headline is nearly always over-broad: `[M]` "ONE
  production spelling" really meant *one OWNER*; the scan-normal form survives at 3 sites, and
  the two forms **partition** the input set, which is what makes "welded by gate" a design).
  ⭐ And a guard re-keyed from a retired field's PRESENCE onto a VALUE signal is doc-worthy —
  say WHY it is stronger (reachable by calling the SUT directly; no earlier guard can preempt
  it), not merely that it changed. → L-071
- **Stub → rich narrative: read memo → production docstrings → tests → SymPy, in that order.** The
  docstrings are the VERBATIM prose seed; the memo carries the honest interim scope — preserve it,
  don't over-claim. Never expand a stub without reading the SymPy; on an algebra error
  DISPATCH_REQUEST the method-implementer, never edit it yourself. → L-005
- **Campaign capstone (a completed feature's whole story): roadmap → literature memo → algebra of
  record → production code → error catalog → evidence pack.** The memo NAVIGATES; the SymPy module
  and production code are the CORRECTNESS spine. Arc: motivation → derivation-of-record → design →
  discoveries → evidence → honest scope. → L-039
- **A fix that works BY RETIRING a failed-approach family gets a SUCCESS-RESOLUTION chapter, not the
  9-step CLOSED arc**, and treats the superseded saga PROPORTIONATELY: ONE loud `.. attention::`
  supersession banner at the arc head + targeted tombstones on the bald factual REVERSALS only.
  Don't tombstone every stale sentence; don't rewrite the history. Flip any prior close-out's "open
  research path" that LANDED. → L-013
- **Deepening an already-documented feature: add the WHY, cross-link the WHAT, never duplicate.** A
  PLANNED refactor on a current-truth page gets a loud `PLANNED, not built` admonition (literals for
  unbuilt types) PAIRED with a "current state" subsection so plan and reality never blur. Verify
  every count against live code — a scoped grep undercounts; prefer describing the guard FAMILY over
  a call-site count. → L-014
- **An EVICTION changes the CARRIER, not the PHYSICS** — grep the stale carrier framing, not the
  physics terms (the brief over-counts; most of the chapter survived). Reframe narrowly plus ONE
  end-state paragraph cross-linking the new algebra. → L-016
- **A completed architecture earns ONE new taxonomy-CULMINATING section, opening by naming the
  generalization and `:ref:`-linking what it generalizes** — never a bolted-on appendix. Document
  symbol OVERLOADS as explicit gotchas. → L-018
- **⭐⭐ A NEW THEOREM: home it where its LOCAL half is already derived, then audit the
  UNIVERSAL it amends tree-wide.** The obvious homes (the BC page, the solver page) were both
  wrong — the page that already labelled the local `−1` face mode owned half the mechanism, so
  the global result is a downstream H1 there and anywhere else mints a twin. Then: the headline
  ("a splitting shares a solution SET, not a POINT, when `A` is singular") falsified an
  unqualified claim asserted **9× across 7 files**; a windowed regex finds them and the
  adjudication is NOT uniform — scope-to-**bulk** where the measurand is bulk, `.. note::`
  tombstone where the sentence stays, one clause where a chapter-scoped truth still says "any",
  and LEAVE where the sentence quantifies over something else (two source-DELIVERY routes of one
  iteration is not a splitting claim). ⭐ Auditing the prose, MEASURE every gate fixture the
  corpus names against the new pathological predicate — one of them WAS singular
  (`[M]` `dim ker A = 36`), and saying *why the gate survives anyway* (its measurands are
  mirror-even) beats deleting or keeping the claim. ⛔ And the changelog entry can be BLOCKED
  by the page's own contract ("merged to main, with its hash, or not at all") — report the
  ready-to-paste row, never fake the hash. → L-057
- **A NEW foundational chapter: READ then RUN the algebra-of-record module(s) — one per distinct
  concept — before writing one equation.** Generalize by stripping the method-specific
  specialization while quoting the identity verbatim; RUN every load-bearing worked number through
  live code so the example is verified, not plausible. → L-025
- **Growing a thin honest-stub chapter at campaign close:** flip its own stale "in flight" status
  (verified against git, not its frozen prose), PRESERVE the already-landed section byte-for-byte
  and grow AROUND it, and RECONCILE sibling taxonomies explicitly with subset relations rather than
  contradicting them. Report DEFERRED WIRING with exact `test node → label` ids when tests you may
  not edit await labels you mint. → L-036
- **"Is the terminal docs phase done?" almost always answers effectively-DONE** — each earlier
  phase's doc pass landed its slice into the eventual capstone. Verify by the page's own
  SELF-IDENTIFICATION plus the build plus a cross-ref grep-gate. A documented SEAM is the OPPOSITE
  of a gap; and a charter's literal "the X page" can be correctly delivered as a SECTION of a shared
  page (a standalone page would MINT a twin path). → L-038
- **⭐ An ONTOLOGY-OVERTURN changelog goes on the page whose THESIS moved.** `history.rst`
  contracts "a new entry lands with its merge hash or not at all", so an unmerged carve is
  BLOCKED there; `operator_algebra.rst`'s history has the `*(in development)* <branch>`
  escape hatch. ⟹ give the rewritten page its OWN Development history following that
  convention verbatim, a short row on the sibling whose axis genuinely moved, and on the
  blocked page tombstone only the falsified HALF of its row (one row, two halves, opposite
  fates). → L-063
- **Merging a re-staged branch's docs into a diverged tree:** read the fork-diff as a CONTENT
  source, not a patch; splice programmatically; translate EVERY module path to the live layout (a
  moved package vs a same-named unmoved one) with zero residual; place by anchor, never by the
  diff's line numbers; reconcile forward-refs landing in the SAME merge. → L-012

---

- **⭐ The POSING-CONTRACT section (an operator gains its explicit arguments) has six parts,
  in this order:** (1) the fields + why there is **no default** (an active choice), with the
  half of *illegal-states-unrepresentable* you are NOT claiming stated out loud; (2) why the
  substrate KEEPS what it kept — a partition, not a move, with the read-set allowlist as a
  table; (3) the guard ruling as an **attack table** (one row per attack, each with its
  measured outcome, and a `.. warning::` retracting any attack whose stated reason was
  reasoned rather than run); (4) the **route** gate — a route claim needs a route instrument
  (vv #26), so publish the swap, its pre-carve deviations, and every way it goes silently
  green; (5) the performance ruling quoted **verbatim** with a lifetime argument and a
  reproducible cost table; (6) *What moved, concretely*, mirroring the predecessor phase's
  closing subsection so the two read as one arc. → L-072

## 7. V&V vocabulary — you are the curator (Directive 5)

You write the prose future readers QUOTE about verification status. Match `vv-principles` verbatim;
never paraphrase a level definition. → L-010

- **Never** "MMS verifies the eigenvalue" (source-driven: flux-shape / convergence-order only) ·
  never "L4 proves correctness" (name its L0–L2 backing) · never "the 1-group test verifies the
  solver". NAME the pillar (closed-form / MMS / semi-analytical), not vaguely "analytical". → L-010
- **Never upgrade a `@pytest.mark.foundation` gate to an L-level in prose to make a section sound
  better-verified** — read the marks and say "software/structural invariant of a discrete
  construction, not an equation claim". → L-040
- **A doc sentence "gates X, Y pin claim C" IS a coverage claim** — the prose analogue of
  `vv-principles`' "a `catches` marker is a COVERAGE CLAIM, not a topic tag". Justify it by a
  MUTATION that reddens X and Y, never by topical adjacency. Cite **per field**, not per topic
  (5 arrays needed 5 different files; one had a SOLE catcher, another was cylindrical-only).
  Highest-risk moment is REPLACING a gate you just demoted — the nearest-sounding sibling
  inherits neither scope. I credited a τ gate for reduced-operator arrays it passes in 0.03 s
  under fully-garbaged factories, two screens after writing the note explaining τ had LEFT that
  operator. → L-047
- **The SAME gate cited for TWO claims can be right once and wrong once — narrow, never sweep.**
  On "citation of X is false", ask *false for WHICH claim* and grep every occurrence before
  editing any; a blanket fix destroys the true citation. → L-047
- **Distinguish the EUCLIDEAN transpose `Aᵀ` from the metric HILBERT adjoint `A† = G⁻¹AᵀG`.** A
  campaign may colloquially call the former "†"; write the precise object. A docstring summary
  saying "Hilbert transpose" over a body computing the Euclidean one is a real defect. → L-010,
  L-034
- **A Mode-10 sub-floor term is closed by STRUCTURAL teeth, not a tightened value band** — and when
  no isolating regime exists, SAY there is no value-improvement leg to add. Pair the honest-scope
  note with a prophylactic `.. warning::`: the test pins the math, the warning pins the LANGUAGE.
  → L-010
- **Get a Mode-12 blindness boundary EXACTLY right** — a `k`-row is blind to the
  factor-order/similarity family, to all vector content, and to the metric itself, but NOT to a
  single-leaf transpose drop. Pair every `k`-claim with its vector/pairing catcher; a METRIC-REPAIR
  closure needs its CONTROL leg described, or a still-broken baseline mimics "caught". → L-036, L-015
- **The FIRST iterative member of a previously all-exact family has no bit-id twin to inherit** —
  claim foundation / flux-shape against a structurally-independent reference, and EXCLUDE the
  round-trip tautology explicitly so a reader doesn't mistake it for coverage. Related: never fuse
  an eigenvalue and a fixed-source term in one equation. → L-010, L-036
- **Skill-uplift duty:** propose the `vv-principles` / `error_catalog.md` / `algebra-of-record` edit
  in your return whenever you meet a published-prose anti-pattern or evidence-boundary case the
  skill doesn't capture. The skill grows when you feed it back. → L-010

---

## 8. Code-prose rebalance (docstring/comment trimming)

- **Expect ZERO MOVED.** Cardinal Rule 3 means the theory shipped WITH the code, so a concept that
  FEELS unique to the file is almost always already TWIN in the landing chapter. Grep the landing
  chapter before crediting one MOVED; a pre-classifier's MOVED column is ~100 % noise. → L-033
- **The CONTRACT test: "would a competent modifier who never leaves this file do the wrong thing
  without this line?"** If yes it is CONTRACT however history-flavored — including a keep-vs-retire
  decision on an intentional orphan, a ⚠ latent-trap imperative (keep the imperative + the
  falsifying number inline, derivation to a `§`-pointer), and a type-annotation rationale guarding a
  plausible wrong "simplification". → L-033
- **FILE-CLASS sets the size and SURFACE of the honest cut.** Teaching-heavy operator ⟹ aggressive
  TWIN-cut. Contract-heavy operator / machinery / ABC ⟹ small cut; the surface is module-head
  essays, campaign provenance and duplicated numbers, NOT method bodies. Driver / mesh ⟹ hunt
  standalone `#`-COMMENT tombstones and campaign-status blocks FIRST (comments dwarf docstrings). A
  −2 to −5 % cut is CORRECT — report the file-class rationale so it isn't read as timidity. → L-034
- **Provenance trimming = citation-vs-narration, applied uniformly** (trim landed campaign-STEP
  codes; KEEP bare `#NNN` anchors and named patterns; half-stripping violates internal
  consistency). But a hand-transposed-adjoint / reverse-scan comment body IS the algebra of record
  — KEEP it though it reads like narration. → L-034
- **A batch "special" is a VERIFICATION obligation first, an edit obligation only on failure** —
  read the oracle, read both ends, report SATISFIED, never touch a correct CONTRACT block. → L-034
- **Prove the edit is doc-only by AST/token comparison vs HEAD** (`tokenize` dropping
  COMMENT/STRING, or `ast.dump` with docstrings stripped), not by reading the diff. The AST check
  also proves no `verifies`/`catches` marker moved — but it is BLIND to comments (fine, they are
  editable) and an **f-string assertion message is CODE**: leave it and REPORT. → L-041, L-045 Run the Sphinx
  gate **iff** the file is `automodule`'d — `:noindex:` does NOT exempt it. A RENDERED file affords
  two extra moves: promote a latent trap to a real `.. warning::`, and repoint in-file back-refs
  after a heading rename. → L-033, L-034, L-041

---

## 9. Gates, generated artefacts, tooling

- **Generated artefacts are NEVER hand-edited** (V&V matrix, capability tables,
  `_generated/*.inc.rst`) — fix the registry-side metadata and report the REAL post-regen number. A
  `-E` build on a dirty branch absorbs rows from OTHER uncommitted work: a legitimate by-product —
  never revert it, REPORT it. In a fresh worktree a missing generated artefact is an ENV gap, not a
  doc defect: materialize it, never route the docs around it. Never transcribe a hard-coded test
  count. And orphaned built HTML from a renamed source looks like a live stale ref — discriminate by
  "does the source `.rst` still exist?". → L-008, L-040, L-026
- **⭐ A retirement's dead-ref count is under-reported by the commit that made it** — a brief
  and a commit body both said "23 dead refs" (one retired package); the second retired module
  path added **9** more. Grep every retired path, not the one the message names. ⚠ And a
  deleted package can still IMPORT: an untracked `__pycache__` leaves a PEP-420 namespace
  package (`__file__ is None`, 0 members) that a naive `import_module` probe calls LIVE —
  probe a SUBMODULE. `[M]` the xref gate saw 1 of 32 (L-062's unlanded `head_role` bug); my
  own import probe over 727 roles across 8 edited pages is the real gate. → L-063
- **⭐ Measure the xref-gate baseline from `git archive HEAD` into a temp tree** — the cheap way to
  get a TRUE before/after on a dirty working tree (`git archive HEAD orpheus tests docs tools | tar
  -x -C <tmp>`, then run that tree's own copy of the gate on it). `[M]` 81 dead / 124 sites both
  sides while adding 80 xref roles. Its file-count will differ (untracked files); the DEAD number is
  the gate. → L-059
- **⛔⛔ The xref gate's `head_role` blindness is at the HEAD-CHECK line, not the first
  `candidate_paths` call — and an INERT patch reads exactly like a clean tree.** `judge()` returns
  ALIVE from the first call and only later runs `if not any(lookup(c)[0] for c in
  candidate_paths(head, namespaces, role))`, where `candidate_paths("orpheus", (), "class")` is
  `()` on an `.rst` page ⟹ DECLINED. Patch THAT line. `[M]` with a throwaway `docs/_ctl.rst` (2
  dead + 1 live role): stock **0 dead**, patched **2 dead / 2 sites**, `decidable` 5797 -> 5799.
  ⟹ the control must SPLIT the two gates: **stock == patched is itself the tell that the patch is
  inert**. Corpus reading (control removed) over `docs orpheus tests`: **0 dead**, 1006 files /
  16 886 roles / 14 184 decidable. The fix is still UNLANDED in
  `tools/check_docstring_xrefs.py`. -> L-082
- **⛔⛔ An ERR entry WITHOUT its `catches` marker REDDENS A GATE — a docs-only pass can
  break the suite, and only running the gate finds it.** `[M]` adding ERR-081 moved the
  generated `.claude/skills/vv-principles/error_index.md` 80/0-uncaught → 81/1 (never
  hand-edit it) AND turned
  `tests/test_error_catalogue_reconciles.py::test_every_declared_entry_has_a_catching_test`
  RED; its docstring offers *"or say in the entry why no test can exist"* but the
  assertion parses nothing, so there is **no machine-readable exemption**. ⟹ after
  minting any `.. error-entry::`, RUN
  `pytest tests/test_error_catalogue_reconciles.py`, and report the required
  `@pytest.mark.catches("ERR-NNN")` as BLOCKING, not as a nicety. → L-091
- **⭐ This corpus's structural self-check has THREE standing false positives — record
  them or every run re-litigates.** `[M]` 2026-09-03: the `boltzmann` "duplicate eq-label"
  is a `.. code-block:: rst` EXAMPLE in `harness.rst`; a "ragged `list-table`" row is a
  legal EMPTY cell (`^     -$`, no trailing space — match `^     -(\s|$)`); a "dangling
  `:doc:`" is a RELATIVE docname. Everything else the pass reports is real. → L-089
- **⭐⭐ Validate your OWN self-check parser against a known-good member before believing its
  negatives.** My list-table column checker required a trailing space after a cell's `-`, so a
  legal EMPTY cell (`^     -$`) read as a ragged table — two false positives on PRE-EXISTING
  tables, which is exactly the direction that wastes a cycle chasing a non-defect. Fix:
  `^     -(\s|$)`. Same for a corpus label-uniqueness scan: my "duplicate `boltzmann` eq-label"
  was a `.. code-block:: rst` EXAMPLE, not a label — a scanner that cannot see literal blocks
  reports the corpus's own documentation of itself. (`nexus-tools`' positive-control rule, turned
  on the instrument I wrote.) → L-081
- **⭐ The two-build rule is broken by EDITING AFTER LAUNCH, every time — this pass cost FOUR.**
  Each extra build was bought by one late correction (a quotation, a role qualification, a
  denominator). ⟹ the self-consistency pass — universals, quotations, denominators, superlatives
  — runs to EXHAUSTION before the first verification build. Baseline and final were both **0**
  W/E/C, EXIT=0, so the acceptance evidence held; the cost was pure wall-clock. → L-081
- **RE-MEASURE the `-E` baseline every session; never assume a recorded number** (it has drifted
  9 → 1 → 0; measured 0 again 2026-08-11). Diff the WARNING/ERROR/CRITICAL SET pre/post, not just
  the count. A full `-E` rebuild can exceed the 120 s foreground cap — background it, or the
  poll-loop is SIGTERM'd at the final line. → L-029, L-041, L-027
- **⭐ SEQUENCE the session so you build TWICE, not four times:** baseline `-E -W` → *all* edits →
  *all* residual greps → xref gate → AST doc-only proof → ONE verification build. Two of four
  builds were wasted by launching before the last edit landed (a residual grep always finds one
  more site). ⚠ **Re-broken on a NEW page: FIVE builds**, every extra one bought by an edit made
  after launching — the self-consistency pass (universals, symbol collisions, aspirational rows)
  must run to EXHAUSTION *before* the first verification build, never interleaved with it.
  → L-054, L-064
- **⭐ ONE re-runnable python self-check beats the build for structure, at ~2 s:** short-underline
  detection + ladder-order (first-appearance) + per-table column consistency across EVERY row +
  `:widths:` sum + label/anchor uniqueness (EXACT-line compare) + role import-resolution +
  `:eq:`/`:ref:`/`:doc:` resolution against the whole `docs/` corpus. It caught every structural
  defect on a 1158-line new page before any build ran. → L-064
- **An agreement `[M]` number handed to you is a LADDER unless proven flat** — a briefed "verified
  to 1.67e-16" was a small-fixture reading that degrades to `2.3e-14` two refinements later, and
  the shipped gate already knew (`atol=1e-13`). Measure the ladder, publish the ladder, name the
  gate's tolerance, say a finer row must widen it (`vv-principles` #16). Likewise read a
  two-number "spread 0.30→1.53" as ONE row, not a sequence, until re-run. → L-054
- **Title markers (AGENT.md owns the rule): prefer COPYING a proven underline from a model page
  over re-counting code points.** → L-009, L-035
- **Citations: `grep '^\.\. \[Key\]'` before citing or defining** — resolve cross-doc, never
  redefine; match a page's plain-text convention; on a strict-zero-warning NEW page go ALL
  plain-text (a Literature `list-table` with equation numbers inline — higher articulation, zero
  machinery). Pre-existing duplicate-citation warnings are a known trade-off: verify the count is
  unchanged. FLAG a conflated bib key. → L-006, L-025
- **A corpus-wide mechanical migration is dry-run-first and WHITELIST-scoped** (which auto-skips
  every pseudo-site); key any block remover to INDENTATION too, or it eats footnotes. → L-031
- **Self-check the V&V scan directly, not via the full audit** — the theory-equation scanner runs in
  <1 s, avoids pytest collection, and doesn't trip on sibling batches' in-progress sentinels.
  ⭐ **A `.. vv-status:` sentinel WORKS INDENTED** — `sentinel_re.match(stripped)`, same-FILE rule
  only, so one inside a `.. warning::` is found. Read the 30-line scanner instead of reasoning
  about it (I nearly relocated one for nothing). → L-035, L-069
- **⭐ To prove a published closed form IS the shipped scheme's, CALL the shipped function — and
  budget one round for its shapes.** `affine_scan_coefficients` wants `V` at `(N, nx)`, not
  `(nx,)`. Fed correctly, DD/LD reproduced my Padé ladder to `1.1e-16` / `1.2e-16` over six
  optical depths, and `carlson_inward_sweep_from_source` showed the shipped seed march
  sign-alternating at ratio `−0.2 = (2−3)/(2+3)` exactly. That turns "these are the shipped
  forms" from an assertion into a measured bound, in ~3 minutes. → L-069
- **zsh does NOT word-split an unquoted `$var`** — a uniqueness loop ran once on the concatenated
  string and printed a false "0 collisions". An `Edit` `old_string` must match LIVE bytes. → L-030
- **An error-message string inside `raise` is EXECUTABLE — report it, don't edit it, under a doc-only
  constraint** (tests `pytest.raises` match on those strings). Same for a brief item that is a LATER
  phase's acceptance-gate text: leave it, name the owning phase. → L-041
  ⭐ **…and CHECK WHICH SUBSTRING is pinned before reporting it as immovable.** `[M]` three stale
  `raise` parentheticals (*"a seedless mesh (Cartesian, or a non-carrying cylinder, R12a)"*) —
  tests match only the OPENING clause (`"carries no starting-direction ray"`), and
  `grep "Cartesian, or a non-carrying" tests/` = **0**. The report then says *"unpinned, safe to
  correct"* instead of *"pinned, leave it"*, which is a different instruction to the code owner.
  → L-073

- **⭐⭐ A CAPABILITY FLIP stales DEFERRAL CONTRACTS, and the population is the CLASS the flip moved
  — census with a ±3-line CO-OCCURRENCE window, never a line grep.** `[M]` `non[-_ ]?carrying` over
  `tests orpheus docs` minus `_build` = **151 hits / 42 files**; windowed against
  `cylind|\bcyl\b|_cyl|cyl_|CYL` → **79 paired / 72 unpaired**. The unpaired half is general
  contract (*"``None`` on non-carrying meshes"*) and must NOT be touched. → L-073
- **⭐⭐ The acceptance predicate is a QUALIFIER window — the co-occurrence count RISES when you
  succeed** (79 → 80: a correction names what it corrects). Gate on *paired AND lacking a
  Q5.6.3/admission/refus/Until/unconstructible/⛔/HISTORY token within ±5 lines*: mine ended at
  **3**, all three CODE (a def, a call site, a `raise` f-string). Publish the predicate, not the
  count. → L-073, L-070
- **⭐⭐ The flip's signature defect is a STALE HEADER over an ALREADY-CORRECTED BODY — read ±30
  lines and cite the body instead of re-deriving.** A docstring said *"slab AND cylinder → 1×1"*
  over a body that BUILDS a folded cylinder and asserts 2×2; a comment described a retired fold 29
  lines below its own `HISTORY` note retiring it. Half the fixes write themselves from the
  neighbouring truth, and the tree's already-correct sites are the model text — adopt their
  spelling verbatim rather than minting a rival vocabulary. → L-073
- **⭐ A flip also stales "X is UNTESTABLE" and "X is unreachable" — the MIRRORS read as settled
  facts nobody re-checks.** `[M]` *"a multi-carrying-level indexing bug is UNTESTABLE with current
  geometry"* became a fixture gap (the admitted cylinder carries on every level); *"this inline is
  unreachable through the mesh"* was refuted by the tree's own sphere census. Scope a sweep to
  "live X" claims only and you miss both. → L-073
- **⭐⭐ A GATE is not a denominator — count its CALL SITES, and expect the brief's universal to be
  scoped.** `assert_carrying_quadrature` has ONE call site, inside `case CYLINDRICAL`; the SPHERICAL
  arm calls no admission gate, so `[M]` (in-tree census, 2026-08-26) a μ = −1-noded Gauss-Lobatto
  sphere rule reaches the non-carrying branch at 6 of 11 orders / 75 levels. *"The slab is the only
  admitted non-carrying 1-D geometry"* is true of the SHIPPED `Quadrature` constructors (no
  `gauss_lobatto` exists) and false as a structural universal — publishing the unqualified version
  would have licensed retiring a live branch. → L-073
- **⭐⭐ Prove "prose only" with an AST DIFF, not a reading** — (a) token stream with STRING values
  dropped, (b) `ast.dump` after stubbing every docstring to `"<DOCSTRING>"`. Both identical ⟹ no
  `raise` message, no `match=`, no code moved. ~10 lines each; it is what makes the
  message-literal exclusion auditable instead of promised. → L-073
- **A pristine `-E` baseline from `git archive HEAD` carries UNTRACKED-DATA artifacts — read the
  traceback before counting them.** `[M]` its 2 warnings were `plot_directive` exceptions from
  `load_isotope` (data files untracked ⟹ absent from the archive); the live tree builds 0. Quoting
  2 would have credited me with a fix I did not make. ⚠ `rm -rf` inside a compound Bash command is
  refused here — `mkdir -p <fresh dir>` instead. → L-073, L-051

---

## Quality self-assessment (Directive 3)

Rate 1–5 and log the weakest dimension: Derivation depth · Cross-references · Numerical evidence ·
Failed approaches · Code traceability · Derivation source. On TERMINOLOGY / ROUTING / retirement
passes the weak dimension is routinely "numerical evidence" — structurally ABSENT (no flux moves ⟹
no convergence table), not a deficit. Say so, don't manufacture one.
