# Coding standards — the minimum-quality floor

These are **preemptive minimum standards** every contributor (main agent and every
sub-agent) follows by default, regardless of whether they are chasing elegance. They are
the *floor*. For the *ceiling* — recognizing how bad code manifests and refactoring toward
excellence — code-producing agents load the **`coding-elegance` skill**. (See also Cardinal
Rules 1 "Correctness" and 2 "Architecture" in CLAUDE.md.)

## Clean before extending

Before adding a capability to a class/module — especially on a new design — first run a
cleanup pass on that layer: collapse double paths, move concepts to their native place,
delete dead shims, fix twin sources of truth. The new capability should then land as a
**no-op extension through the single generic body**, not as a new arm grafted onto
structural debt.

- When a plan proposes a capability extension, insert a dedicated **cleanup phase before
  the capability phase**. Order findings into must-precede vs independent-polish vs
  explicit-WAIT; gate each cleanup substep as bit-identical where possible.
- Rationale: extending a layer that carries debt grows a third arm on every seam. (C5,
  2026-06-11: `from_axes` round-tripped axes→legacy-mesh→axes, so 3-D admission would have
  needed a new arm in the converter, constructor, AND trace gate — until the cleanup
  inversion let 3-D flow through the one generic body.)

## Type vs property — before minting a type

A representation earns its own **type** only when it is genuinely a different object, not the
same object wearing a label. The decidable (grep-checkable) criterion: mint a type **iff**
(a) there are **≥2 non-isomorphic bases/realizations** of the concept, AND (b) a
**non-identity morphism** is actually applied to it. Otherwise the concept is a **property**
— a field or flag on an existing type.

- If the only "change of basis" is the identity (one realization, no transform), a separate
  "type" is theatrics: it adds ceremony and a conversion seam without making any illegal
  state unrepresentable. Make it a property.
- This is the *type-minting* corollary of the `coding-elegance` "defer abstraction until ≥2
  instances" rule. Worked: an expanded-order spatial moment with a single basis and an
  identity change-of-basis is a `property`, not a `SpatialOrder` type.
- **Corollary — an axis that changes the ARITHMETIC INTERFACE cannot be a phantom type
  parameter.** The criterion above decides *whether* to mint; this decides *how to encode
  it*. `Generic[Tag]` is erased at runtime and does not specialize dunders, so every
  instantiation shares ONE `__add__` body. If two values of the axis need different
  arithmetic signatures — a torsor `A×V→A` that must FORBID `A×A`, versus a vector
  `V×V→V` — no shared body satisfies both, and the encoding must be a distinct class.
  Decision lattice: axis changes the **arithmetic** ⇒ class; axis changes the **shape** ⇒
  class; only an axis that changes NEITHER may be a phantom parameter. Negative
  discriminator: an implementation that "passes" only by branching on a stored tag field
  at runtime is stringly-typed dispatch — `replace(obj, tag=Other)` type-checks and walks
  straight through the gate the type was minted to be.

## A bare `assert` in `orpheus/` is not a contract — the canonical runner strips it

`.claude/rules/vv-testing.md` makes **`python -O -m pytest`** canonical, and `-O` sets
`__debug__ = False` and removes every `assert` statement at compile time. So a numerical or
domain contract expressed as a bare `assert` in production **does not run in the suite that
matters**, and the code ships accepting exactly the input the assert was written to refuse.

- **The discriminator is what the assert is FOR**, and it is grep-checkable — run
  `grep -n "^\s*assert " orpheus/` and sort the hits:
  - **type-narrowing** (`assert x is not None` for pyright) — fine to strip; the failure
    downstream is an immediate `AttributeError`, and the assert was never the guard.
  - **a numerical / domain / admission contract** (a tolerance, an invariant, an
    antisymmetry, a shape law) — **MUST be a real `raise`.** Model it on the nearest
    existing admission guard so the vocabulary stays greppable
    (`_assert_tau_within_unit_interval`, `_assert_alpha_dome_closes`).
- **Prove it, don't argue it.** The demonstration is four lines: run the guard's own
  arithmetic on a deliberately-bad input under plain `python` and under `python -O`. If the
  second one returns instead of raising, the contract is inert.
- ⚠ Corollary for the retirement audit: converting one costs a 3-search pass like any other
  retirement, **and the shortest distinctive fragment of the old assert's message is what
  tests pin** — grep that, not your new wording.

> `[M]` 2026-08-12. `α_{M+1/2} = 0` is a genuine admission contract on every curvilinear
> quadrature (it is a *consequence* of the measure's antisymmetry, not an axiom of the
> one-sided Lathrop–Carlson recursion). It was enforced on the **sphere** by
> `assert abs(alpha[N]) < 1e-12` and on the **cylinder** by nothing at all. Demonstrated on
> the verbatim recursion: a measure closing at `alpha[N] = +0.2000` is REFUSED under plain
> `python` and **ACCEPTED** under `python -O`. Fixed at `bea6a367` — but note the fix was
> *not* "add a check to the cylinder arm": the recursion had **three** copies, which is
> precisely why the contract could live on one arm only. Cardinal Rule 2 first, then the
> guard.

## A guard is elegance debt — tag it, and name what retires it

A runtime guard (`require_member`, `admit_composite`, a typed refusal on an alien
carrier) is a **signal that the architecture failed the elegance standard** of making
the mistake unspellable (`coding-elegance` Pattern 4). It is a legitimate protection
*today*; it is not the target state. **The ultimate state is to not need the guard.**
(`[R]` user, 2026-09-07, ruling the R6 carrier guard: *"we're creating a protection, but
the ultimate state is to 'not need a guard'"*.)

- **Every guard that lands carries a greppable marker in its docstring:** the token
  **`ELEGANCE-DEBT[guard]`**, the issue number, and ONE sentence naming the structural
  change that makes the guarded mistake unspellable (e.g. *"retires when B is bound on
  its own trace end — R18"*). `grep -rn "ELEGANCE-DEBT" orpheus/` is then the debt ledger.
- The issue is filed **with the carve that lands the guard**, never before (a guard
  without its retirement plan is an unpriced debt; a plan without its guard is a
  promise). A step that lands the structural change deletes the guard AND the tag in
  the same commit (the retirement rule below), and the mutation battery must show the
  mistake is now unspellable — not merely refused.
- ⚠ The tell that a guard is being mistaken for architecture: its docstring justifies
  the *check* rather than naming the *shape that would make the check unnecessary*.

## Retire as you go (aggressive retirement)

Superseded code is **noise that obscures signal** — it makes the codebase harder to read,
breeds "which path is canonical?" confusion, and invites accidental extension of the wrong
path. Retirement is a first-class deliverable, not optional cleanup.

- Every refactor that introduces a more elegant pattern **MUST retire its predecessor**.
- Deprecation aliases / compatibility shims live for **one merge cycle only** — remove them
  on the next pass. Never preserve backward-compat code unless the user explicitly
  authorizes it (e.g. a public API on a versioned release).
- Treat the **retirement audit as its own numbered substep**; a plan should carry a
  "retirement list" enumerating what gets deleted (with `file:line`).
- **A retirement's own past-tense NOTE is a confidence trap — discriminate the surviving
  references BY TENSE.** A batch that deletes a symbol and adds an honest "`X` existed
  until 2026-07 and was retired with zero consumers" note *reads* like a completed audit;
  it is not. Grep the deleted name across the whole tree and sort the hits: past-tense
  ("existed", "was retired") is correct history and STAYS; a **present-tense claim**
  ("provides an ergonomic shortcut") or an **imperative instruction** ("Apply the marker —
  `@verify.lN(...)`") is a MUST-FIX — a contributor following it hits `ImportError`, and a
  maintainer re-adds the symbol "to match the doc", reopening the retired twin. Expect the
  offenders to be PRE-EXISTING lines outside the diff, sometimes ~50 lines from the
  batch's own note.
- **Retirement means test migration:** retiring a symbol includes **rewiring its tests to
  the successor**, not deleting them with the symbol. Behavioral test (correctness contract)
  → rewire to the new API; API-smoke test (symbol exists) → delete; characterization test
  (e.g. FD-vs-WDD delta) → keep under `tests/<module>/characterization/`. Pure delete-only
  retirement is incomplete — it loses coverage. Inventory with `grep -rn "<symbol>" tests/`.
- **Retirement means MARKER migration too.** A retired test takes its
  `@pytest.mark.catches(...)` / `verifies(...)` with it: the successor asserting the same
  invariant must be re-tagged, or the coverage edge silently disappears while the
  error-catalog entry keeps naming the dead test class — an audit "MISSING" whose stated
  L0 test still *reads* plausible. Grep the catalog and the `tests/_harness` registry for
  the retired symbol alongside the code grep.
- **A rewire can silently DEMOTE a gate's claim class without touching one line of the
  test body.** When a retirement re-points a comparison target at the successor, re-ask
  **"are the two sides still INDEPENDENTLY produced?"** If the survivor is the *caller* of
  the other, a two-implementation bit-identity gate has become a value compared with
  itself through a wrapper: green forever, keeping its authoritative name, unable to
  detect the drift its docstring advertises — and invisible in review because BOTH sides
  are genuine production calls. The tell in a diff is a local variable still called
  `legacy` beside a brand-new API. The mutation test is one line: replace the SUT's body
  with garbage; if the pin stays green it was never a pin. Do not delete the gate —
  re-scope every doc and docstring crediting it, and name the pin that actually survives.
  **Then check that the named replacement is real:** a redirect to "the regression
  snapshots pin this" is worthless if those snapshots were re-baselined BY the same carve
  (measured 2026-08-03: three `cyl_*` snapshots cited as the pre-carve anchor had all been
  re-captured by the consolidation commit itself).
- **⭐ And the MIRROR, which nothing prompts you to look for: a retirement can silently
  PROMOTE a gate's claim class, and the stale docstring then talks a load-bearing gate
  DOWN.** The demotion above is hunted because a weakened gate is a risk. A *strengthened*
  one raises no alarm — the suite is greener, nothing fails — so its description is never
  re-read, and it keeps advertising the weaker claim it had before. That is how a real
  gate gets deleted as redundant: an audit reads "these checks are tautological", believes
  the docstring over the code, and removes the only witness to a property.
  The check is the same question asked in the other direction, and it costs one reading:
  after a retirement, for every gate that *survived* untouched, re-derive what its
  assertion now compares. If the docstring says "tautological", "by construction", or
  "restates the definition", that is a claim about an implementation which may no longer
  exist.
  > `[M]` 2026-08-17. `tests/sn/operators/test_loss_action_convention.py` asserted
  > `apply(ψ) == loss_action(σ_t, ψ) − C.apply(ψ)` and its own header called that check
  > *"tautological (`apply` is DEFINED as `loss_action − σ_t·ψ`)"* — true when written.
  > #257 S8b made `apply` σ-free (`loss_action(0, ψ)`), so the same line now reads
  > `loss_action(0,ψ) == loss_action(σ_t,ψ) − σ_t⊙ψ`: the **affinity of the walk in σ**,
  > a falsifiable property of two independently-evaluated walks. The gate went from
  > restating a definition to being the only check of an algebraic property, with **no
  > line of the test body changing**, and its docstring went on disclaiming it for months.
- **⭐ SINGLE-SOURCING A DUPLICATE DEMOTES EVERY GATE THAT COMPARED ITS COPIES — and this
  is the case where the demotion is CORRECT, so it is the one you must not resolve by
  backing out.** The two clauses above cover a survivor that *calls* the other side; the
  `vv-principles` #22 sibling covers a test that *hands one object to both* sides. This
  third case has neither tell: nothing calls anything, no object is shared, and the gate's
  body is untouched. What changed is that the two things it compares are now **derived from
  one constant or one rule**, so no input exists that could make them disagree.
  Prevention beats detection (Patterns 2 and 4), so the fix stays — **the gate's
  DESCRIPTION is what must move.** The decision procedure, in order:
  1. **Ask what input could still make the two sides differ.** "None" ⟹ tautological. It
     is a *design-time* question, not a mutation question — an in-class mutation reddens it
     for the wrong reason, since mutating the single source moves both sides together.
  2. **Hunt for an EXTERNAL hand-written pin before concluding you lost coverage.** If some
     test already asserts the set/values against a literal authored independently of the
     new single source, the carve cost nothing. If none exists, you traded a real gate for
     none and owe a replacement — that is the whole risk of this move.
     ⭐ **And the pin you reach for FIRST is the one most likely to be blind — check that
     it MOVES under the old value before citing it.** This applies to every re-baseline,
     not only to single-sourcing: the candidate reference that comes to mind is
     "green, authoritative, and nearby", and *nearby* is the problem — a gate in the same
     module as the moved values is usually built on the same simplifying fixture that
     makes the change invisible. A green reading is compatible with *loaded* and with
     *blind* (`vv-principles` #19), so it is the OLD-value reading that licenses a
     re-baseline. One in-process mutation answers it; skip it and you record a false
     justification, which is worse than recording none because an audit will trust it.
     > `[M]` 2026-08-12, task #51 — **twice in one session, two unrelated blindness
     > mechanisms, both in the failing rows' own file.** (a) Cartesian octant: the
     > obvious licence was `test_2d_octant_sweep_closed_form_anchor` (`φ = Q/Σ_t`), green
     > at HEAD — but it is an all-reflective FLAT infinite medium, so it reads the
     > quadrature *only* through the total weight. `sum(w) = 4π` to **0.000e+00**
     > (bit-exact) at LS4/LS6/LS8 before AND after, while `μ₁` moved
     > `0.408248290463863 → 0.350021174581541` (14 %). It is a Σw-normalisation gate
     > (ERR-004/025), structurally blind to node PLACEMENT. The real licence was the LS
     > moment-exactness / advertised-degree suite. (b) Cylinder τ: every flat-flux L0
     > anchor is blind because the M-M recurrence gives `(ψ−(1−τ)ψ)/τ = ψ` for **every**
     > τ — including the `@verifies("streaming-equilibrium")` gate sitting in the same
     > file as three of the failing rows. The real licence was
     > `test_cyl_tau_equals_the_ANALYTIC_closed_form_not_the_chord_convention`, 1 of the
     > 32 gates a whole-suite old-τ mutation actually reddens.
  3. **Keep the gate only for what it still tests, and say so in its own docstring** — not
     in a plan, not in the commit. A gate wearing an authoritative name for a comparison
     that cannot fail is worse than no gate: it is a coverage claim an audit will trust.
  > `[M]` 2026-08-09, #345. `capability_rows()` and the reference registry were two
  > hand-written enumerations, and their `r_0` name tags had *already* diverged
  > (`round(r0*100)` vs `round(r0/R_out*100)`, agreeing only because every shipped
  > `R_out` is `1.0`). Writing the promised row→registry join would have detected that;
  > hoisting the grid to one constant + one `reference_name()` made it **unspellable** —
  > and made the join tautological in the same commit. Step 2 saved it:
  > `test_builder_keyset_is_the_shipped_class_a_inventory` already pinned all 13 names
  > against a literal written independently of both, so the name set stayed anchored. The
  > join was kept, re-described in its docstring as testing the *discovery-and-registration
  > path* (which has no other catcher), and explicitly disclaimed as no longer able to
  > catch a spelling divergence.
- **⭐ And the ENFORCEMENT side of the same move, which the three clauses above do not
  reach: retiring a duplicate promotes whatever KEPT THE COPIES EQUAL from redundant to
  load-bearing — and that thing is usually a production guard with no test.** The clauses
  above all ask what happens to the *gates that compared the copies*. This asks what
  happens to the *mechanism that made comparing them pointless*. While the duplicate
  lives, that mechanism is over-determined: if it broke, the two copies would disagree and
  something might notice. Delete one copy and it becomes the **sole** guarantor of the
  survivor's correctness — with no change to its own code, so nothing prompts a re-look,
  and the suite only gets greener. This is the MIRROR clause's silent-promotion shape
  applied to a `raise` instead of a gate, and it is the more dangerous half, because a
  guard is not something an audit thinks to ask for coverage of.

  ⟹ **Before retiring a duplicated field, answer two questions in order:** *what makes
  the copy provably redundant?* — the answer is the mechanism — **and** *does that
  mechanism have a witness?* Grep the shortest distinctive fragment of its message (the
  message clause below). If the answer is none, write one **in the same commit**: the
  retirement created the exposure, so a follow-up issue is the wrong home.

  ⭐ The migration is usually free, because the tests that asserted the retired copy were
  *tautologies* — a stored literal read straight back — and they are exactly the right
  place to put the guard's witness. Same test names, same concept, real teeth.

  > `[M]` 2026-08-27, un-weld P4.1a. Retiring `ReducedStreamingOperator.coord` (a copy of
  > `mesh.coord`): what made it redundant was that each of the three factories *validates*
  > `mesh.coord` against the literal it then stored, so the identity held **by
  > construction**, not merely on 3/3 shipped fixtures. `grep "requires .* mesh"` returned
  > **3 hits, all three the production `raise` lines — zero witnesses tree-wide.** After
  > the retirement those guards are the only reason `op.mesh.coord` is the operator's
  > chart. The three `TestProperties` chart tests had been asserting the stored literal;
  > rewritten as the guards' witnesses (`vv-principles` #11 — one positive leg, two
  > negative legs each, matching the production message) they cost one edit and closed the
  > exposure in the commit that opened it.
- **The retirement audit's blast radius is THREE searches, not one** (4–5 agents converged on
  this independently): (1) **graph callers** (`nexus impact`/`callers`) — necessary but NOT
  sufficient; the call graph misses property-reached leaves (`callers()==0` but live via a
  `cached_property`), class-name *bypass* consumers, and direct constructors of a guarded
  type; (2) **text-grep the symbol across code, tests, AND `docs/`** — an unresolved
  **Python-domain** cross-reference (`:func:`/`:class:`/`:meth:`/`:mod:`) renders as plain
  text with **no `-W` warning**, so the Sphinx gate does NOT catch a code retirement's doc
  blast radius — **and `-n` (nitpicky) does NOT save you either.** (This clause read
  "unless the build runs `-n`" until 2026-08-03; that was false, and it told every
  retirement audit it was covered when it was not.) Sphinx can only nitpick what it
  RENDERS, so a docstring in an un-`automodule`'d module is invisible at EVERY severity,
  as is every file under `tests/`. That is the majority case here, not an edge case:
  the doc source carries only ~45 live `automodule` directives, and the whole of
  `numerics/measure.py`, `numerics/operator.py` and `numerics/quadrature/` is among the
  many with zero — which is exactly why a module retirement left 22 dead `:class:`/`:mod:`
  refs that no build of any severity could see. Before concluding "`-n` would have caught
  this", check whether the module is rendered at all; if it is not, **grep is the only
  gate**, and an unchanged warning count proves nothing about it. (Measured 2026-07-15, Sphinx 9.1.0:
  `:doc:` and `:ref:` **do** warn — `ref.doc` / `ref.ref` — so *page* moves and *label*
  retirements ARE gated by `-W`; the silent class is the Python-domain roles, plus **raw path
  strings** in prose/docstrings, which no build ever checks. A path assembled from segments —
  `REPO_ROOT / "docs" / "theory"` — is invisible to a path-grep too; grep the **last
  segment**.)
  ⭐ **And the measurement showing a COMPLETE import audit is still a PARTIAL
  audit: a symbol has more than one spelling SURFACE.** The clause above is
  argued from `docs/`; this is the same defect one surface over, in `.py`
  files, and it survived every check that was run.
  > `[M]` 2026-08-28, un-weld P4.4 (4 symbols, `geometry/` → `sn/mesh/`). The
  > import audit was done by **AST** (not grep), the residual check ran in
  > Python **with a positive control**, and returned **0**. The affected suite
  > was green, `pyright` 0, `sphinx -W` **clean**. `[M]`
  > `mcp__nexus__dead_references` then found **5 dead targets / 9 sites** —
  > `:class:`/`:func:`/`:attr:` cross-references to the old path sitting in
  > **docstrings** in `orpheus/sn/angular/closure.py`,
  > `transport/spatial/{scheme,cell_balance}.py` and two test modules. Nothing
  > else could see them: the module is not `automodule`'d, so `-W` is silent at
  > every severity. ⭐ The transferable half: the residual filter was
  > **validated and correct** — it was run over the wrong *surface* (import
  > statements), not with the wrong *pattern*. A positive control proves your
  > regex finds what you point it at; it says nothing about whether you pointed
  > it at the whole corpus. ⟹ **`dead_references` is the only instrument that
  > reads the docstring surface** — run it before calling any retirement or
  > re-home done, and again after the fix (this one went 9 → **0 dead / 52
  > checked**).
  ⭐ **A MATH symbol has THREE spellings, and a number is a fourth.** The
  concept-grep rule below tells you to widen the vocabulary; this says the
  *same symbol* is already spelled three incompatible ways in one repo, so a
  grep for any one of them returns a confident partial answer: the **ASCII
  identifier** (`tau_raw`), the **Unicode prose** form (`τ_raw`), and the
  **LaTeX role body** (`\tau_{\rm raw}`) — which matches neither of the others.
  Grep all three, and then grep the **NUMBER** the claim carries, because a
  stale figure often outlives every spelling of its symbol.
  (2026-08-11, Q5.6.4: a retirement sweep briefed with `tau_raw` + `τ_raw`
  reported clean. `docs/theory/methods/sn/angular_quadrature.rst:369` still
  asserted *"`\tau_{\rm raw} \in [1/5, 4/5]` with the **bit-exact** reversal
  identity"* — both halves present-tense-false after the ω-partition carve.
  It was found only by grepping `tfrac15|tfrac45`, i.e. the NUMBER in its
  LaTeX spelling. The page was also absent from the audit's own file list,
  because that list was built by the same two-spelling grep.)
  ⚠ Dual hazard, same measurement: the audit's file list was simultaneously
  **over**-counted ~3× because `absorber` is also a *material* (`pure
  absorber`, `cavity-absorber`) and `clamp` is also a GMRES `restart` clamp —
  11 of 17 flagged pages were false positives. A concept grep needs its hits
  triaged by MEANING before any of them is called a site.

- ⭐ **A symbol grep cannot see a name that lives inside a STRING — and the
  `getattr(obj, "name", default)` form is the one that bites, because it fails
  in the DEFAULT's direction rather than raising.** `\.name\b` and
  `name\s*[:=]` are the natural residual patterns and neither matches
  `getattr(x, "name", None)`, `hasattr(x, "name")`, `setattr`, or a
  `__getattr__` table key. After the retirement, the call does not raise: it
  silently returns the default, and every branch keyed on that default flips.
  ⟹ **run `grep -rnE "['\"]<symbol>['\"]"` as part of the audit, on every
  retired name**, and read what each hit's default *means* — a `None` default
  on a field that was `None` for exactly one case turns "is it that case?"
  into "always yes".
  > `[M]` 2026-08-26, P1 item 8. Retiring `SNMesh.curvature` (whose `None`
  > **was** the Cartesian case), my residual grep returned only prose and I
  > called the set closed. `tests/sn/operators/test_native_matvec.py:392` read
  > it as `curv = getattr(sn_mesh, "curvature", None)` and branched on
  > `curv is None`, so after the retirement **every curvilinear mesh took the
  > slab branch** — 2 reds, sphere and cylinder. ⚠ The aggravator: I had run
  > exactly this string-form check for `mu_start` **one item earlier** and
  > confirmed it clean. The habit did not transfer across two commits by the
  > same author on the same afternoon, which is why this is a rule and not a
  > reminder. It failed loudly only because the assertion on the wrong branch
  > happened to be falsifiable; a `getattr` default that matches the common
  > case fails silently and green.

**Grep the CONCEPT, not only the symbol:** a field/flag is documented in two
  registers — by NAME, which greps, and by PARAPHRASE, which does not. A `list-table`
  column headed "Sweep-cycle flag" carries per-law values with no symbol in any cell; one
  audit's 7 exact hits missed 17 further cells. After the symbol grep, grep the
  hyphen/space variants of the concept the symbol names. And the paragraph that JUSTIFIED
  the retired thing inherits its wrongness — re-verify that prose against the replacement
  rather than only deleting the dead name from it. (3) **direct constructors** of any guarded type (a guard-at-source change
  reaches every `T(...)` caller, not just the factory path). Run all three, then retire.
- ⭐ **A LABELLED EQUATION is an API, and correcting the prose around it does not correct
  it.** The message-string clause below says a string becomes an API the moment a test pins
  it; a `.. math:: :label:` is an API the moment anything writes `:eq:`. The failure mode is
  specific and it is invisible to every gate: a page learns something, records the
  correction **in prose**, and leaves the labelled equation stating the original claim — so
  the page now carries both, and every `:eq:` citer inherits the false one. Sphinx cannot
  help: the reference RESOLVES (the label exists), so `-W` is silent at any severity, and
  the V&V matrix will happily report the label as covered.
  ⟹ When a correction touches a claim that any equation states, grep `:eq:`<label>`` and
  read **every** citer, then fix the equation itself — not only the paragraph you were
  looking at. And check the label's `verifies()` marker still means what it says: a test can
  legitimately verify the *kernel's API default* while the equation now states the
  *scheme's* value, and those are different claims wearing one label.
  > `[M]` 2026-08-13, task #67. `pole-mm-recurrence`'s first line read
  > `\phi_{1/2,i,g} = 0` while production marches the seed as an ODE. The page had ALREADY
  > condemned it twice — "replacing the hardcoded zero that Phase B had baked in" and a
  > `ZeroSeed` row reading "the pre-ERR-026 term-initialisation bug … wrong off flat flux" —
  > and a sibling page carried a subsection titled "The bug Phase B baked in". All ~2500
  > lines from the equation, none of it reaching it. Four sites inherited the zero,
  > including the page's own **Key Facts** card. Sphinx built clean throughout, and the
  > sole `verifies()` marker on the label was a suite that passes NO seed, so the matrix
  > reported the equation covered while the covering rows asserted the kernel default.
- ⭐ **A CROSS-REFERENCE is a load-bearing dependency: repairing a claim obliges you to
  check the section you point readers AT.** The clause above is about a correction that
  stops short of an equation in the same page; this is about one that stops at the page
  boundary. When you strengthen an argument and cite another section for the details, that
  section is now carrying part of your claim — and if it still states the version you just
  retired, your repair has *imported* the falsehood rather than fixing it, while reading as
  more rigorous for having a citation. Neither Sphinx nor grep helps: the reference
  RESOLVES, and the cited text contains none of your new vocabulary.
  ⟹ the check is mechanical and cheap: **for every `:ref:`/`:doc:`/`:eq:` you ADD or LEAN
  ON while making a correction, read the target and ask whether it still says the old
  thing.** Bounded — you are checking the handful you cite, not the page's whole reference
  graph.
  > `[M]` 2026-08-14, quadrature Q6-C. The theory page's stage-2 argument was rewritten to
  > say a degree means nothing without its reference measure, and cross-referenced the
  > *1-D primitive constructors* section for the Gauss rules. That section still stated
  > **both** rules as "`degree_of_exactness = 2n - 1`", with the Chebyshev one qualified
  > only by the prose "in the weighted sense" — i.e. exactly the bare-integer half-claim
  > the new argument exists to refute, sitting at the end of its own citation. Found by
  > the agent doing the repair, not by any build: `-W` was clean throughout, and the
  > section contains neither "reference" nor "claim" for a grep to catch.
- **Retiring a MESSAGE STRING: grep the SHORTEST distinctive fragment, never the full
  sentence.** An exception/log message is an API the moment a test pins it, and tests pin
  **substrings**. A grep for your own longer wording is strictly LESS sensitive than the
  consumer's pattern, so it returns a confident, empty, wrong answer. (2026-08-06, G6.3 step
  8.0: retiring `OperatorSum`'s inline check onto a shared helper reworded
  `"OperatorSum requires equal domains"`. The audit grepped `requires equal domains` — which
  matches the production line — and reported only the definition site. Two gates matched on
  `"equal domains"` alone and went red in the wide run; a third reference was prose. The
  correct pattern was the two-word fragment.) Corollary: when the audit does find pins,
  prefer **keeping the established vocabulary** over renaming it — the phrase is load-bearing
  provenance ("this guard fired, not some incidental raise"), and it is greppable precisely
  because it has not drifted.
- **Routing a call site through the ALGEBRA silently raises its operand requirement from
  "has the verb" to "IS the type" — and only duck-typed test doubles notice.** Re-spelling
  `f.apply(g.apply(x))` as `(f @ g).apply(x)` is arithmetic-neutral by construction, so it
  reads as a pure re-spelling; it is not. `@` needs `__matmul__`, i.e. a real operator,
  where `.apply` needed only an attribute. Production is usually unaffected (its objects
  come from a factory that already returns the type), which is exactly why the breakage
  surfaces in a *test* and looks like a broken test rather than a contract change. Fix it
  by making the surrogate honour the contract it stands in for; do NOT add a runtime guard
  for a case the type system now covers (that is the harmful-stub anti-pattern). (2026-08-06,
  G6.3 step 8: a `_NoTransposeLaw` stub with only `apply` hit
  `TypeError: unsupported operand type(s) for @`.)
- **And a retirement onto a SHARED helper moves the raise's provenance one frame out.** Any
  gate asserting the innermost traceback frame is now asserting the helper, which is
  reachable on behalf of *every* caller — so the gate silently widens from "this composite
  refused" to "something refused". Re-point it at the helper AND assert the CALLER frame (or
  an owner tag in the message), or the retirement demoted a provenance pin while leaving its
  name intact — the same defect class as the fuller-view/bit-identity demotion above.
- **Mass-deletes are retirements too — and untracked shadow-copies mask the breakage.** A
  "chore: mass-delete old diagnostics" sweep owes each file the same 3-search audit, with two
  sharpenings: (a) grep the **module/script NAME**, not only its symbols — a subprocess-worker
  import (`from diag_x import f` inside a `textwrap.dedent` worker string) is text the call
  graph never sees; (b) a diagnostic consumed by a tracked test is **production
  infrastructure** (the instrument behind a pinned baseline), never "shipped/falsified"
  debris. If a consumer stays green after the delete, check WHERE its import resolves: an
  untracked shadow (a scratch/ working copy, a stale `__pycache__`) can serve the import and
  keep the breakage silent until the shadow evaporates. (2026-07-13: `15486f66` mass-deleted
  `diag_cin_aware_split_basis_keff` while the CP rank-n protocol test's worker consumed it;
  an untracked scratch copy masked the loss for ten weeks, then vanished — recovery had to
  route through a surviving `.pyc`'s `co_filename` back into git history.)

### The mirror — landing a deferred capability stales its DEFERRAL CONTRACT

Retirement's mirror image, and just as blocking. When a change flips a case from *deferred*
to *implemented*, every docstring naming that case "raises / deferred / not yet supported"
becomes present-tense-FALSE, and the blast radius is the same three searches.

- The recurring half-cleanup signature: **the human-facing prose ledger gets rewritten and
  the machine-facing contracts do not** — the `@runtime_checkable` Protocol stub, the BASE
  class's default docstring that the next implementer inherits, the sibling class's own
  docstring, and public operators in files the diff never touched.
- Grep the deferred case's NAME and its prose forms across the package, then discriminate
  **by arm**: landing a matvec-transpose does not un-defer the transpose-SOLVE. Only the
  one flipped row changes; the still-genuine future seams stay.
- In a campaign-closing commit this is blocking — a `Closes #NN` trailer is internally
  inconsistent with a status the tree still tags deferred.

### Exception — keep a relinquished *fuller view* as a verification oracle

Retirement targets a *superseded* predecessor (same job, done worse). It does NOT target a
**fuller view of a concept that an optimization relinquished** (full field → rolling window;
full angular flux → moments; dense operator → factored form). The fuller view is a
**verification pathway** that pins the optimized path's reference — keep it.

- Make the keep-vs-retire decision EXPLICIT; never let a fuller view fall out silently or
  sit half-alive (orphaned-but-undeleted). It is either a wired, exercised oracle or it is
  retired.
- The oracle is NOT production-reachable and MUST share the optimized path's kernel (only
  the representation/storage differs), pinned by a permanent **end-to-end** equivalence test
  (`optimized ≡ fuller-view-oracle`), bit-identical or principled-equivalent per
  `vv-principles`.
- **Discriminator:** keep it only if it is a genuine structural reference. A "fuller view"
  that is the SAME math merely *procedurally rearranged*, AND already verified by a
  structurally-independent oracle elsewhere (MMS, closed form), is genuine redundancy —
  retire it. (`vv-principles` L11: procedural ≠ structural independence.) Worked: the 2-D
  rolling-window sweep kept `_sweep_2d_full_field` as a typed-`WavefrontFlux` oracle; the
  per-ℓ scattering kernel was retired as mere procedural rearrangement.
