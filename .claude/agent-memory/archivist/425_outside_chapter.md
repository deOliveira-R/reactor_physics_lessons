# #425 — the within-group algebra outside the SN chapter (2026-09-07)

Branch `docs/425-within-group-algebra`, 17 pages of `docs/theory/**` outside
`methods/sn/`. Concurrent second archivist on the SN chapter. Census instrument
`scratch/_425/census.py`; residual 81 → **2** (both HISTORY), 5-term 9 → **80**,
0 new eq-labels. What transfers:

---

## 1. ⭐⭐ I CANNOT run `sphinx-build`, and a DOCUTILS parse is a real substitute — it caught 2 `-W` breakages of mine that NOTHING else did

Register a permissive directive/role shim (`Directive` with `option_spec=None`
returning `[]`; a role returning `nodes.literal`) for the ~40 Sphinx
directives/roles this corpus uses, then `publish_doctree(text,
settings_overrides={"report_level":2,"halt_level":6,"warning_stream":io.StringIO(),
"file_insertion_enabled":False,"strip_comments":True})` per file, and filter out the
`Unknown directive type|Unknown interpreted text role|Duplicate .* target|Citation|
Footnote|Hyperlink target|Undefined substitution` classes. What SURVIVES the filter is
structural and real.

`[M]` it found exactly two, both mine, both invisible to every other gate I ran
(backtick parity, ref/eq resolution, table cell counts, nested-markup scan):
- **`ERROR/3 Unexpected indentation`** — an edit-then-revert cycle had eaten the
  blank line between a paragraph and the following `.. (vv-status rationale)`
  comment. A comment directly abutting prose is an ERROR, and the file *looks*
  fine.
- **`WARNING/2 Bullet list ends without a blank line; unexpected unindent`** — my
  replacement text put a continuation word at column 0 inside an indented bullet.

⟹ **run it on every file you touch, before reporting.** It is ~3 s for 17 files and
it is the only instrument I have that reads RST *as a parser*. Validated by the fact
that it reddened on my own two defects and went to 0 after the fix (its own positive
control).

⚠ Its noise floor is per-corpus: whitelist by MESSAGE CLASS, never by count, and
**diff against `git show HEAD:<file>`** for anything you did not obviously cause —
6 list-table cell-count anomalies I flagged were all present at HEAD (my crude
table parser's artifacts), and only the HEAD comparison separated them.

## 2. ⛔ A CONTINUATION LINE STARTING WITH `-`, `+` OR `*` INSIDE A TABLE CELL IS A BULLET, and it silently eats the role

Breaking a long `:math:` across lines is standard here (`[M]` 33 line-broken
`:ref:` roles ship in the corpus). But inside a `list-table` cell the content
column is fixed, so:

```
        - :math:`(L+C).\text{apply} - S.\text{apply}
          - N_{2n}.\text{apply} - B.\text{apply}`      ⛔ docutils sees a BULLET
```

⟹ **break AFTER the binary operator, never before it** — `... - S.\text{apply} -` /
`N_{2n}.\text{apply} ...`. LaTeX does not care; docutils does. `+` is a bullet
character too, so the same trap fires on an additive expression.
Cheap scan: over your ADDED lines, flag every `^\s*[-+*] ` and ask whether it is a
real bullet.

## 3. ⭐⭐ A TERM-COUNT IN PROSE IS THE MEMBER-LIST CLAIM THE CENSUS STRUCTURALLY CANNOT SEE — and its referent is often AMBIGUOUS

A regex census over `L + C - S - B` finds equations. It cannot find
*"the **five**-operator algebra"*, *"the **five** leaves"*, *"implemented by
**five** leaf operators"*, *"all **five** operators — L, C, S, F, B"*, *"the α-row
needs a **sixth** leaf joining L, C, S, F, B"* — every one of which is the same
claim in words. `[M]` 9 such sites outside the chapter, on 4 pages, none a census hit.

⚠ And the triage is NOT mechanical: of 29 `(four|five|six)[- ](term|operator)` hits,
**most were a different four** — ERR-039's four conflated operators, the CP/Peierls
four-term second-difference, a carrier grid's four leaf TYPES, "four operator-algebra
questions", "seven operator-equivalence tests". Grep the count-word, then triage
every hit **by referent**.

⭐ The nastiest member: *"the **four operator families** occupy disjoint blocks"*
over a block matrix whose underbraces name **five** operators in **three**
`BlockRole` groups. The number matched nothing on the page. Do not bump such a
number — **re-derive what it counts** (here: the enum has 3 members, so I named the
roles and dropped the count).

## 4. ⭐ A DOC THAT QUOTES CODE IS THE HIGHEST-DECAY SENTENCE ON THE PAGE — grep the quoted expression against the live source

Three stale code quotes, all in `.. implements::`-adjacent or "what the builder
returns" prose, all `-W`-silent, all found only by reading the live function:
`A_AA = LC - S - B_a`, ``explicit_gains`` ``(S, B_a)``, and
`KEigenvalue((L+C).H, (S+B).H, F.H)` — live: `... - S - N2N - B_a`,
`(S, N2N, B_a)`, `((L+C).H, (S+N2N+B).H, F.H)`.

⭐ And the sharpest instance was in `orpheus/` itself: `sn/coupled_system.py`'s
docstring at `:502` spells the loss `A_AA = L+C−S−B_a` and, **in the same
sentence**, the gain grid as `[[S+N2N+B_a, …]]` — four members on one side, five on
the other, 49 lines above the code that composes five. A doc can contradict itself
inside one sentence.

## 5. ⛔ A BRIEF'S "ZERO mentions of X" IS A CENSUS, AND THE PAGE CONTRADICTING ITSELF IS THE STRONGER FINDING

Briefed: *"`operator_algebra.rst` … ZERO mentions of N_{2n}"*. `[M]` **5**
(`:788, :807, :3457, :3570, :3844`) — including a correct, rich derivation of
`N_{2n} = R Λ_{2n} M / W` with the yield `y_S = 1, y_{2n} = 2` matching the live
`ClassVar` roles exactly. So the page was not ignorant of the operator; its
**definitional header, Key Facts and vv-status rationale comments** were stale while
its **body** was right — vv #21's self-contradicting-file aggravator at page scale.

⟹ that changed the repair: not "teach the page a new concept" but "make the
definition agree with the body, and CITE the body" (`:ref:`scattering-binding-cs4c``).
Always run the brief's own negative census first; a refuted zero re-shapes the work.

## 6. ⭐⭐ A CROSS-SOLVER "the other method does X instead" CLAIM IS THE ONE NOBODY RE-READS

`diffusion_1d.rst` asserted *"S_N poses (n,2n) production-side instead — both are
consistent posings of the same balance"*. True when written; **retired by ERR-065
(R7, 2026-07-03)**, two months before this pass, and reinforced by CS4c step 3.
`[M]` `SNSolver.compute_keff` returns `production / (absorption + leakage -
emission_n2n.sum())` — νΣf-only numerator, emission subtracted: **loss-side, the same
side as diffusion's.**

Nothing points at such a sentence: it lives on page A and its truth lives in solver
B, so neither method's own maintainer re-reads it. ⟹ when a page says *"the other
solver does the opposite"*, open the other solver's function **and** grep the error
catalogue for the term — the catalogue entry is where the reversal is recorded.
Repair shape: a `.. note::` preserving the retired sentence verbatim in quotes,
naming what retired it, and stating the surviving difference (here: same SIDE,
different GROUPING).

## 7. ⭐ A `:ref:` TO A PARAGRAPH ANCHOR NEEDS EXPLICIT TEXT — and a paragraph anchor is what you want for a "one place says it" note

Single-sourcing the composition note as `.. _operator-algebra-two-gains:` above a
**paragraph** (not a section title) is the right Pattern-2 move — 11 pages then point
at one paragraph instead of restating it. But a bare `:ref:`label`` on a non-title
target is `ref.ref` *"A title or caption not found"*, a real `-W` failure. Write
`:ref:`the two collision gains <operator-algebra-two-gains>`` everywhere, from the
first use.

⭐ And it costs **0 equation labels**: `tests/_harness/audit.py` builds `all_labels`
from `.. math:: :label:` ONLY, so a `.. _x:` anchor cannot move the documented-label
gate. Verify that by reading the harness, not by assuming the namespaces are separate.

## 8. The annotation-window discipline a mechanical done-when forces

The done-when was *"unannotated residual == your listed HISTORY sites"*, annotation
matched within **±3 lines**. Twice the page already carried the correct scope note —
`coupled_block_operator.rst` explained the diffusion fusion perfectly, 2–6 lines
below the site — and it did not count. ⟹ a scope note is only load-bearing **beside**
the claim it scopes; a correct explanation one paragraph away is read by nobody who
lands on the equation. Moving it up was a genuine improvement, not gate-gaming.

## 9. Verdict discipline that held up

Four verdicts, and the only hard calls were (ii)-vs-(iii):
- **(ii) fixture/method special case** — the honest test is *does the method's
  composition site fuse, or is the fixture Σ₂ₙ-free?* Both are checkable: read the
  composition site (`diffusion/solver.py:248-254`), or read the fixture's
  `make_mixture(...)` call for a `sig_2=` kwarg (`sig_2` defaults to `None` ⟹ Σ₂ₙ ≡ 0,
  so ABSENCE of the kwarg is the proof).
- **(iii) history** — a dated changelog row, an ERR narrative, "an earlier version of
  this section", "that bypass is retired", "#331 recorded that". 7 sites, listed.
- ⚠ The Wave-O boundary-extraction record was the genuine grey zone: present-tense
  grammar, historical subject (its cited FP captures predate the operator). Resolved
  by **dating the spelling in place** — *"the two-gain spelling here is Wave O's,
  which is what the cited captures were taken against; N_{2n} joined at CS4c step 3
  and rides this argument unchanged"* — which keeps the evidence honest instead of
  retro-fitting a member list onto a measurement that never saw it.
