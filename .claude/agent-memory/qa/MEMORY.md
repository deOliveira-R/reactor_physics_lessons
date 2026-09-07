# QA Agent Memory — index

## 1. Lessons — a HOT digest over a COLD archive

Two files, read at different times. Do **not** summarize lesson content here.

- **[lessons.md](lessons.md)** — the HOT digest (~760 lines). Behavioral rules
  only: one imperative rule + its failure→correction core + a
  `→ lessons_archive.md L-0NN` pointer. Nine sections: **A** mutation mechanics,
  **B** structural blindness, **C** structural independence, **D**
  re-baseline/bit-identity, **E** markers & audit surface, **F** claim-scope,
  **G** doc-correctness, **H** mechanics/environment, **I** the map of what is
  already in `vv-principles`/`numerical-bug-signatures` (point, don't restate).
  **Read this before every review.**
- **[lessons_archive.md](lessons_archive.md)** — the COLD archive (~4800 lines,
  L-001..L-076, append-ordered). War stories, evidence, `file:line`, measured
  tables, verdicts. **Open only the specific `L-0NN` a digest rule points at** —
  never read it whole (it is ~60K tokens).

Maintenance: a new lesson appends `L-0NN` to the archive AND lands a 2–5 line
rule in the digest. Sharpen the digest in place; **it is now past the ~400-line
distillation trigger — distill before the next append** (encode a shared
meta-lesson once, list its instances); never truncate.

## 2. Active / in-flight state

**#428 four-solver (n,2n) census — 2026-09-03.** ⏹ **DELIVERED** (HEAD
`8707c53a`, branch `fix/n2n-anisotropy`; READ-ONLY — `git status --porcelain
-- orpheus/ tests/ docs/` empty throughout, revert proven by gate-green-again
163/163). Memo `scratch/_428_four_solver_check.md`.
**Verdict: all SIX families HANDLE MT=16** (removal ×1 in `Σt`/`Σa`, emission
`2Σ₂ᵀ` from the one home `N2NKernel.multiplicity`, channel in the k balance);
**ERR-023 is FIXED** and MC is unbiased (`1.63 σ`). The doc's *"every transport
solver assumes 1-in-1-out"* is present-tense-false for all six. Instrument = a
ONE-ATTRIBUTE mutation (`multiplicity` 2→1 and 2→0) as an in-process plugin,
reds read per tree. Five findings: **F-2 diffusion HANDLES it with 0 of 113
witnesses** (625 mixtures in that tree, 1 with Σ₂≠0 and it never reaches a
solve); **F-4 ERR-023's only catcher is `@slow`**, so `-m "not slow"` never
adjudicates it (39 passed/0 red at the gate; FAILS in 84 s run alone);
**F-3** SN's ν₂ₙ pinned at the operator tier only (8 reds) — `homo_2eg_n2n`
absent from the SN k_inf ladder, and the ERR-065 gate is correctly
self-consistent (2 claims, 1 marker); **F-5** the multiplicity census's
name-net misses `sig_2`; **F-1** `solve_sn_adjoint`'s docstring omits N₂ₙ
(the code is right — both arms measured). Lesson **L-079**, digest **E7** /
**A17**.
⚠ **TWO skill items OWED and NOT landed** (the brief forbade tracked-file
edits): **E7** (a `catches` marker on a `slow` test is coverage the canonical
gate never RUNS — a new class beside Mode 8's nine, all of which are about a
gate that cannot FAIL) and **A17** (a two-stage census filter needs a positive
control per STAGE). Drop-in text is in the digest rules and L-079 §findings 2/4.

**#448 SN eigenvalue finalize — UNCOMMITTED-DIFF REVIEW, 2026-09-06.** ⏹
**DELIVERED** (branch `fix/sn-eigenvalue-finalize-448`; READ-ONLY — `git
status --porcelain -- orpheus/ tests/` md5 identical mid- and end-session).
Probes `scratch/_448_qa/probe{1..4}*.py`, mutation plugin
`scratch/_448_qa/_mut448.py`. **Verdict: the algebra is COMPLETE and correct
on all five arms** (1-D SI, 2-D windowed SI, coupled sph/cyl, Krylov, and the
opt-in boundary-G-S); gate module 45/45 green; every arm's returned bulk is an
`AngularFlux`; the coupled ψ½ and the returned trace (both DECLARED blind)
measure right at ~1e-11. **Six findings, none a numerical defect**: the G-S arm
has no self-consistency witness (`[M]` 161 solves / 0 G-S splittings in the
gate module; the only catcher is the GAUGE pair, wrong subject per vv#18); the
trace-provenance gate the module itself declares owed was not written; ERR-083
is unminted while 4 gates claim it (`sphinx -W` will fail — verified against
`merge.py:641-660`, `adopted=True`); 34 doc sites + one DECLARED `:by:` edge
are dead; 13 duplicate/mangled imports; the two new `__all__` functions have no
direct test. ⭐ The BAND's stated mechanism was wrong (a tolerance sweep whose
`n_outer` never moved) — landed in `vv-principles` #13 as the fourth disguise.
Lesson **L-080**, digest **A18** (phase-scoped mutation by RECORD rewrite) /
**A19**.

**#429 symmetry/quotient carve — TERM-LEVEL REVIEW, 2026-09-02.** ⏹
**DELIVERED** (HEAD `c1fca8bd`, branch `fix/angular-phantom-support`;
READ-ONLY — no tracked file edited, revert proven by `diff -q` vs pristine
copies AND gate-green-again 670/670). 9 probes `scratch/_rev_qa_*.py`,
13-arm mutation battery `scratch/_rev_qa_arm_*.log` (positive control 111
reds). CONFIRMED by brute force built in plain numpy: `is_normalised_by`
5103/5103, `normalises` 729/729, `contains` 575/576 (the 1 was MY improper
"SO(3)" sample), `orbit_stabiliser` 24/24 against a genuine maximum search,
vv#15 law 1260/1260, barycentre equivariance 183/183. EIGHT findings, all
`[M]`: (1) `identity_component` false on 12 of 22 members, 0 consumers;
(2) `is_invariant`'s O_h/I_h docstring bullets describe RETIRED code (the
I_h one advertises an ERR-072-shaped 12-element sample); (3) the
"brute-conjugation CONTROL" is production α-renamed (AST-proven); (4)
registry stage 0 re-uses Γ (a rule-CLOSURE requirement) as a licence to
FOLD — the σ_y fold is admitted for `cartesian2d`, emptying 2 of 4 sweep
quadrants; (5) both kernel short-circuits 0-red; (6) three arms of
`_identity_component_normalises` have ONE catcher each; (7)
`candidate_groups` branches on node STORAGE WIDTH — one fold reports
`{D_2h}` or `{σ_x,σ_y,σ_z}` by spelling; (8) `Cn(1)` vs `Trivial`, one
group two doors. Lessons **L-077**, digest rules **A15**/**A16**.
⚠ **TWO skill items OWED and NOT landed** (the brief forbade tracked-file
edits): the Γ-reuse anti-pattern and the α-normalised-AST control check —
drop-in text is in the review's final message and in L-077 §findings 3-4.

### ⚠ The ONE still-open debt

**Nexus V&V demand memo (L-070, 2026-08-15) — two novel rationales still OWED to
`vv-principles`, `[M]` re-verified 2026-08-30 (0 hits in the skill):** N1
*inferred-relation-under-a-DECLARED-name*, N3 *a recall counter placed DOWNSTREAM
of a filter cannot count what the filter dropped*. Drop-in text: L-070 §12 (they
live in digest **A11** only). N2 (the Mode-8 DECODER dual) ✅ landed.
**Land N1/N3 before the next review closes** — and with them the TWO new
items from L-077 (Γ-reuse; the α-AST control check).

### ⏹ Complete — one line each; the evidence is the lesson, the tense is git

- **#426 (n,2n) anisotropy — INDEPENDENT REPRODUCTION** (2026-09-03, branch
  `fix/n2n-anisotropy`, READ-ONLY) → **REPRODUCED**: all three k to 9 dp
  (`1.095322188 / 1.091186690 / 1.091199657`), C0 exactly 0.0. The claim's
  "−413.55 pcm" vs my "−377.56" was a **UNIT**, and the convention **inverts**
  the study's thin-vs-thick conclusion (D1). Both probes' shared conventions
  closed against PHYSICS (upper-triangularity 8195/8195; entrywise
  `|Σ_ℓ|/Σ_0 ≤ 0.96`, 0 > 1 ⟹ no stray `(2ℓ+1)`). 6 findings D1–D6, 3 attacks
  withdrawn. L-078 / **F21–F23**. Memo `scratch/_426_repro.md`.
  ⚠ **ONE skill item OWED** (brief forbade tracked-file edits): the
  overloaded-unit-name rationale for `vv-principles` §Anti-patterns —
  drop-in text in the review's final message and L-078.

- **CS4c step 0 feeding census** (2026-08-30) → above. L-076 / **A14**.
- **SN specialization audit** (2026-08-26) → `scratch/specialization_audit.md`;
  findings (1)+(2) REMEDIED by P1 (`ebe5d22f`, `37d6d1af`); the class is
  `StreamingCoefficientCache` now. Still-live residue: the DD-hardcoded `2.0`s in
  `psi_half_angle_seed.py:180-185`; `cylindrical_streaming`'s docstring
  recommending two quadratures its own guard refuses; the `Bailey 2009` citation
  its module header condemns. L-075 / **A13**.
- **CS4a-R Phase-1 gate review** (2026-08-21) → 12 findings; the `#17` sharpening
  ✅ landed. L-074 / **A12**.
- **CS4a round-1 design-assembly reviews** (2026-08-20) →
  `scratch/cs4a_attack_{algebra,physics,parsimony}.md`; all four OWED skill items
  ✅ landed (`vv#29`, `#28`'s 7-of-13 correction, `#30`, `plan-authoring` §10
  third shape). Still-live: all three assemblies miss `La13511Case`
  (`sood_registry/la13511.py:171`). L-072 / L-073.
- **Task 51 — the 7 CYL snapshot reds** (L-069) →
  `scratch/task51_cyl_snapshot_audit.md`; verdict RE-BASELINE all 7. Two doc
  repairs still stand: the FALSE `[M]` *"at M=2 τ did not change at all"*
  (`tests/sn/regression/_generate_snapshots.py`; `Δτ = 2.071e-01`) and
  `test_streaming_operator.py:869`'s CYL bit-identity claim.
- **Q5.6.4 SN cylindrical τ** (L-068) → `scratch/q64_attempt2_qa_review.md`;
  refutes link C6. Tree-carried findings: the false Hébert 3.437/3.439 citation;
  no value gate on the partition producer; **one** catcher on the cylinder τ;
  `alpha_defect_beta`/`nu_closure_residual` have no test consumer; no
  `ψ̂`-positivity gate.
- L-001..L-062 SN campaigns: all merged to `main`.

⛔ **Every branch/hash/line above is a SNAPSHOT — reconcile with git and a grep
before acting on any of it** (`process-discipline`; measured to have lied twice).

## 3. Durable reference (topic files)

- [field_role_typing_apply_sourcesink_contract.md](field_role_typing_apply_sourcesink_contract.md)
  — re-checkable SN role contract (`.apply`=AngularSourceSink,
  `.solve`=AngularFlux) + the A2D-1 source-hash-pin update procedure +
  affine-gate test-migration playbook. Cited by `qa/AGENT.md` enforcement
  rule #10 — **durable**.
- [phase1_moment_space_review.md](phase1_moment_space_review.md) — the
  ERR-039 moment-space P1.1–P1.7 verification-of-record; cited as the
  verification artifact by 3 plan/agent files — **durable**.
- [issue_247_legA_review.md](issue_247_legA_review.md) — full #247 Leg A
  (slope-source) review; distilled into L-037. **Retire candidate** (merged
  campaign; the reusable behavior is in the lesson).
- [issue_251_legB_review.md](issue_251_legB_review.md) — full #251 Leg B
  (boundary face-slope) review; distilled into L-038. **Retire candidate**
  (merged campaign; the reusable behavior is in the lesson).
