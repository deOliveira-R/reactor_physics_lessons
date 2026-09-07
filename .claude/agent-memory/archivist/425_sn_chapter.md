---
name: 425-sn-chapter-within-group-algebra
description: "#425 SN-chapter pass (2026-09-07): making 15 theory pages state the shipped five-member within-group algebra. Lessons on adjudicating a spelling sweep, the gates that work when Sphinx cannot be run, and the universals I published and had to retract."
metadata:
  type: project
---

# #425 — the SN chapter states the algebra the tree composes

**What.** `docs/theory/methods/sn/*.rst` (15 pages) had to spell
`A = L + C − S − N_{2n} − B` wherever they state the GENERAL within-group
algebra, and declare the scope wherever they keep the four-term form.
Concurrent with a second archivist instance on every other `docs/theory/`
page. Branch `docs/425-within-group-algebra`, HEAD `9ac5269e`.

**Why it matters going forward:** the lessons below are about *spelling
sweeps* — a class of pass where the census is the population FLOOR and the
adjudication is the work.

---

## 1. ⭐⭐ A spelling census is a FLOOR — the sites it cannot see are the ones that decide the pass

The instrument counted `L+C−S−B` patterns. Three whole classes sat outside it
and each was load-bearing:

- **The SPLITTING of the algebra.** `ψ_{n+1} = (L+C)^{-1}(Sψ + Bψ + q)` and
  `M = (L+C)^{-1}(S+B)` carry no `L+C−S−B` substring, so the census is blind —
  and leaving them makes the page state a five-member operator whose own
  splitting drops a term. `[M]` the shipped splitting is
  `explicit_gains=(S, N2N, B_a)` (`orpheus/sn/coupled_system.py:579`;
  `solver.py:1311` unpacks the triple), so the correction is verified, not
  inferred. ⟹ **when a pass changes an OPERATOR, grep its SPLITTING, its
  ITERATION MATRIX and its PRECONDITIONER spelling too.**
- **A SECTION HEADING.** Two headings named the count/spelling
  (`The four-operator within-group equation`,
  `The (L + C − S − B)·ψ = (1/k)·F·ψ framing at the solver level`). Neither is
  a `:ref:` target (no `autosectionlabel` in `docs/conf.py` — check that
  before renaming), so both were safe to rename; both would have contradicted
  the equation three lines below.
- **A `.. (vv-status rationale)` COMMENT.** Not rendered, machine-facing, and
  it restated the retired member list verbatim under the equation it
  annotates.

## 2. ⭐⭐ The word "four-operator" collides — count the SET, not the number

`index.rst` calls the algebra "five operators" meaning `{L,C,S,B,F}`;
`slab_one_group` called it "four operators" meaning `{L,C,S,B}`. Adding
`N_{2n}` makes those six and five. Two different sets, adjacent pages, same
numeral. ⟹ **when a count changes, write the MEMBERS beside it** or rename to
a countless form (`The within-group operator equation`).

## 3. ⛔ I published three chapter-wide universals and had to retract all three

I wrote *"every fixture in this chapter is Σ₂ₙ ≡ 0, so no number moved"* into
`index.rst`, `slab_one_group.rst` (twice) and `slab_multigroup.rst`. **False**
— `adjoint.rst` carries the Be-reflected fast-slab (n,2n) anisotropy ladder,
and `tests/sn/eigenvalue/test_keff_estimator_gate.py` + the finalize
reconstruction gate INJECT a nonzero `Sig2`.

The replacement is a real census I can hand anyone:

- `[M]` 2026-09-07: **12 of 12** `xs_library` mixtures (regions A–D × 1/2/4
  groups, `orpheus.derivations.common.xs_library.get_mixture`) carry a `Sig2`
  with **zero** non-zeros. The denominator is complete — `get_xs`'s docstring
  enumerates exactly `"A","B","C","D"` × `"1g","2g","4g"`.
- `[M]` every MMS mixture mints `Sig2 = csr_matrix(np.zeros((ng, ng)))` —
  5 sites in `orpheus/derivations/continuous/mms/sn.py` (`:198, :460, :1030,
  :1889, :3455`), one of them commented `# no (n,2n)`.
- The nonzero-`Sig2` fixtures are all **injected on purpose**, and they live
  in exactly the gates whose pages already spell `N_{2n}` out.

⟹ **the honest claim is about the PASS, not the chapter**: *"this pass changed
no measured value — every edit is an algebra spelling"*, which is checkable
from the diff (I verified: no numeral changed except `y = 1/2`, `1/k`, dates
and issue numbers).

## 4. ⭐ Sphinx cannot be run in a two-instance pass — three gates replace it

One build at a time, so `-W` was unavailable. What worked:

- **A standalone `docutils` parse with permissive stubs.** Register every
  Sphinx directive as a nested-parsing `Directive` (and the literal ones as
  no-ops) and every role as `GenericRole`, then filter the warning stream to
  STRUCTURAL classes only (`Title underline too short`, `Malformed table`,
  `Explicit markup ends without a blank line`, `… start-string without
  end-string`, `Unexpected indentation`, …). **It caught a real `-W` failure**:
  my `.. note::` insertion had swallowed the tail of the paragraph after it,
  leaving an unindented continuation. Nothing else would have found it.
- **The cross-ref gate is a SET MEMBERSHIP test**, not a build: harvest every
  `^\.\. _label:` and every `:label:` across `docs/`, then check every
  `:eq:`/`:ref:` on my ADDED lines against them. **It caught a dangling
  `:ref:`sn-n2n-first-class`` I had invented** (the real anchor is
  `n2n-reactions`).
- **The nested-markup gate needs a working mine-vs-pre discriminator.** My
  first attempt compared the LINES OF THE BOLD RUN against `git show HEAD:`,
  and the run's first line is a FRAGMENT (it starts at the `**`), so
  everything read as new — 47 false positives. Compare the **full source
  lines the run spans** (`txt[:m.start()].count("\n")` → slice `lines`).
  Corrected: 0 new bold runs containing a role or literal.

## 5. ⭐ Two instances on one corpus converged — and the check is worth running

`slab_one_group`'s `si-within-group-operator-eq` is cited by
`docs/theory/verification/sn.rst` (the other instance's page), which repeats
the display. I made the label five-term; **independently, they made their
display five-term too.** Reading the other instance's file before finishing
turned an unverifiable coordination worry into corroboration for my hardest
verdict. Same for `docs/theory/methods/diffusion_1d.rst`, which now carries
exactly the split I used (four-term = the diffusion solver's own composition,
five-term = the general algebra).

## 6. The declared-scope idiom that satisfied a mechanical done-when

The done-when was *"an unannotated residual equal to exactly the listed HISTORY
sites"*, with the annotation regex matching `N_{2n}` / `\Sigma_{2n} \equiv 0`
/ `(n,2n)-free` within ±3 lines. Two traps:

- **`IsotropicN2N` does not match `N_{2n}`.** Two sites sat inside prose that
  said "sums IsotropicScattering with IsotropicN2N into the one S" and still
  read as unannotated. The fix was also the better prose: *"that `S` **is**
  `S + N_{2n}`, so the four-term spelling is a statement about the
  COMPOSITION, not about the member list."*
- **A YAML machine-header line** carried `Sigma_2n = 0`, which matches no
  accepted spelling. Rewriting it as `(n,2n)-free by construction` both
  annotated it and read better.

## 7. The four-verdict rubric, and the one fork it did not decide

(i) general → five-term · (ii) declared Σ₂ₙ-free special case → keep + scope
· (iii) history → untouched · (iv) labelled equation → fix the equation, then
read every `:eq:` citer.

**The fork:** `slab_one_group.rst` is the one-group slab chapter — its
fixtures ARE Σ₂ₙ-free, so (ii) was available for all seven sites. I chose (i)
because three independent things said the sites are GENERAL: the page calls it
*"the honest operator algebra of `operator_algebra.rst`"*; the labelled
equation's own vv-status rationale says *"it names the operator algebra"*; and
the label is cited from a page outside the chapter in a general fixed-source
context. Recorded as a fork in the report, and later corroborated by lesson 5.
