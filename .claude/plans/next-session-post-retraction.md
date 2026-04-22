# Next-session plan — rank-N BC closure research, post-retraction

**Branch entering this work**: `feature/rank-n-cin-aware-basis` (at commit
`7d02434` — the retraction commit).
**Author**: Claude Opus 4.7, 2026-04-22.
**Audience**: fresh Claude Code session with zero context. Read this
document in full before running any code.

---

## 0. Two-paragraph summary (read first)

After ~350k tokens of investigation (Issue #119 close-out + Issues
#120/#121 opened as follow-ups + Experiments E1–E6), the
**c_in-aware split-basis adaptive-scale rank-(1,1,1) closure** we
pursued in Issue #120 has been **empirically falsified at matched
RICH quadrature**. A claimed "breakthrough" of 100–1000× improvement
over F.4 at σ_t·R ≥ 5 turned out to be a cancellation of F.4's
BASE-quadrature noise (~0.1–0.37%) against split-basis's own
quadrature residual. When both closures are run at RICH quadrature
(4, 8, 64), F.4's structural floor (~0.003%) emerges cleanly below
split's structural floor (~0.07%), and F.4 wins 6/6 points tested at
σ_t·R ≥ 5.

The production closure **remains F.4 scalar** (from the Issue #119
close-out). Two research angles remain open: **Issue #121 (PCA
sectors, Sanchez-Santandrea 2002)** — not yet tested at matched
quadrature — and a **newly-proposed Direction Q** — the principled
derivation of F.4's Lambert/Marshak primitive mismatch, which is the
actual load-bearing "trick" that gives F.4 its accuracy advantage
over all formally-consistent rank-N closures we tried.

---

## 1. Where things actually stand

### 1.1 Production closure (unchanged)

- **F.4 scalar rank-2 per-face closure** (Hébert 2009 Eq. 3.323 =
  Stamm'ler 1983 Ch. IV Eq. 34).
- Accuracy: ~0.003% at σ_t·R=10, ρ=0.3 at RICH quadrature; ~0.06% at
  σ_t·R=5, ρ=0.3 at RICH.
- `NotImplementedError` guard in `peierls_geometry.py`
  `build_closure_operator` for `n_bc_modes > 1, reflection="white"`
  **STAYS IN PLACE**.

### 1.2 Legitimate structural findings that survive the retraction

These are independent of the RICH-vs-BASE quadrature issue. Treat as
solid ground for future work:

- **L1** — the isotropic white BC at outer is rank-1 in the constant-µ
  direction. No outer-basis decoration alone helps; improvements must
  come from inner modes, basis-convention asymmetry, richer volume
  representation, or non-white BCs.
- **L2** — F.4's accuracy advantage over formally-consistent rank-N
  closures comes from a **basis mismatch**: Lambert-basis P_esc/G_bc
  (no µ weight on outgoing side) paired with Marshak-basis W
  (µ-weighted). Formally inconsistent; empirically load-bearing.
- **L8** — the rank-(1,1,1) closure has a hidden scale gauge DOF in
  the inner basis normalization. At BASE quadrature this gives a
  factor-18 residual swing (Legendre scale √2 vs Jacobi-c² scale √3),
  but L16 shows this "swing" is into the quadrature-noise floor, not
  into F.4's structural floor.
- **L10** — split basis is catastrophic at thin τ (σ_t·R ≤ 2.5), with
  86% err at σ_t·R=1, ρ=0.3 for any inner basis. The split is
  physically meaningless when attenuation is weak.
- **L11** — the formula scale²_opt = (1+6ρ)/(3ρ) IS empirically the
  best constant scale for the split basis — it captures the
  Eddington-like (1/3) + cavity-like (1/ρ) structure — but this
  "optimum" is relative to the quadrature-noise-dominated error
  surface, not to the true k_eff.
- **L12** — the rank-N plateau at ~1% (without scale calibration) is
  universal across basis types (Legendre, Jacobi c^α, exp(−β·s)
  asymptote, PCA sectors). It's a basis-METRIC barrier, not a
  basis-TYPE barrier.

### 1.3 Definitively falsified hypotheses

- **RH3** — c_in-aware split basis breaks below F.4. Falsified E1 at
  BASE (basis rotation of Marshak rank-N=2). Re-confirmed E4.2 at
  RICH: structural floor 21× above F.4.
- **RH4** — Lambert P/G generalizes to split basis. Falsified E2.4:
  catastrophic (33–737% err).
- **RH5** — Adding inner modes breaks the plateau. Falsified E2.2/E2.3:
  inner mode ≥2 carries <0.1% of the flux energy.
- **RH6** — Physics-asymptote basis β=τ is universal. Falsified E3.2:
  only works at σ_t·R ≥ 20.
- **RH7** — Galerkin fixed-point adaptation breaks the plateau.
  Falsified E3.4 v1.
- **RH8** — rank-(1,1,2) with 2D adaptive scales breaks below
  rank-(1,1,1) floor. Falsified E5: α_1 ≈ 1.0 uniformly (2nd mode's
  scale DOF is empty).
- **RH10** — rank-(1,1,1) split with scale-optimum beats F.4 at
  matched quadrature. Falsified 2026-04-22 (this is the retraction).

### 1.4 Files in the branch

All diagnostics live in `derivations/diagnostics/`:

- `diag_cin_aware_basis_derivation.py` — symbolic foundation (E1).
- `diag_cin_aware_finite_sigma_t.py` — finite-σ_t W matrix (E1).
- `diag_cin_aware_split_basis_keff.py` — full closure implementation
  (E1 + E2). The BIG SCRIPT — 600+ lines with P/G/W primitives.
- `diag_cin_split_inner_enrichment.py` — rank-(1,1,N) scan (E2.2/E2.3).
- `diag_cin_split_lambert_pg.py` — Lambert convention test (E2.4).
- `diag_cin_split_jacobi_inner.py` — Jacobi-c² inner test (E2.6).
- `diag_cin_split_alpha_scan.py` — Jacobi α-scan (E3.1).
- `diag_cin_split_asymptote_basis.py` — exp(−β·s) inner (E3.2).
- `diag_cin_split_galerkin_adaptive.py` + `_v2.py` — Galerkin
  adaptive basis (E3.4).
- `diag_cin_split_source_decomposition.py` — scale-DOF discovery
  (E3.5) + helper functions used everywhere.
- `diag_cin_split_scale_scan.py` + `_optimum_fit.py` + `_formula_test.py`
  — scale formula testing (E3.6/E3.7).
- `diag_cin_split_scale_derivation_eddington.py` — L11 analytical check.
- `diag_cin_split_scale_symbolic_derivation.py` — L11 symbolic.
- `diag_pca_sectors_hollow_sph.py` — PCA sectors quick probe (E6).
- `diag_cin_split_regime_switched.py` — E4 regime-switched scan.
- `diag_cin_split_rank112_adaptive.py` — E5 rank-(1,1,2) scan.
- `diag_cin_split_scale_precision_check.py` — precision probe
  (incomplete — timed out; can re-run at RICH if needed).

These artifacts are TESTED at BASE quadrature. For any future
comparison with F.4, **re-run at RICH quadrature** (n_panels=4,
p_order=8, n_ang=64) to avoid the BASE-noise trap.

### 1.5 Documentation artifacts in the branch

- `.claude/plans/rank-n-closure-research-log.md` — MASTER research log
  (~1400 lines, includes retraction). Fresh sessions should read the
  top-of-file "CURRENT STATE" blockquote + scan for "RETRACTION" and
  "E4 REVISED VERDICT".
- `.claude/plans/verification-spec-split-adaptive.md` — **MARKED
  OBSOLETE** at top. Do not implement those tests; the closure they
  verify is the falsified one.
- `.claude/agent-memory/numerics-investigator/peierls_cin_aware_split_basis.md`
  — updated with FINAL STATUS (F.4 wins at RICH).
- `.claude/agent-memory/numerics-investigator/peierls_inner_metric_frontier.md`
  — legacy (written during the false-breakthrough period); still
  contains correct structural observations but the "frontier" framing
  is outdated.
- `.claude/agent-memory/literature-researcher/hebert_2009_ch3_interface_currents.md`
  — full Hébert 2009 Ch. 3 extraction (including V-S renormalization
  recipe from Eqs. 3.347–3.352). Still valid.

---

## 2. The next two research directions — ranked

### Direction Q (NEW; proposed 2026-04-22): principled derivation of F.4's Lambert/Marshak mismatch

**Why this is highest priority**: L2 says F.4's win over all our
formally-consistent closures comes from a specific basis asymmetry
(Lambert P/G + Marshak W). This is the actual knob that moves the
structural floor. Understanding WHY this mismatch works, and whether
it has a principled explanation in Sanchez-McCormick reciprocity or
adjoint weighting, is the load-bearing open physics question.

**What this would look like** (~2-3 sessions of careful work):

1. **Literature pass**: the literature-researcher agent flagged four
   candidate references during the previous session:
   - **Bogado Leite (1998) ANE 25:347–356, DOI 10.1016/S0306-4549(97)00026-1**
     — "Revised interface-current relations for the unit-cell transport
     problem in cylindrical and spherical geometries." Exact-domain
     match, 1-citation orphan, potential forgotten prior art. Get the
     PDF via interlibrary loan.
   - **Sanchez (2014) NSE 177(1), DOI 10.13182/NSE12-95** — "On P_N
     Interface and Boundary Conditions." Rigorous IC-BC degeneracy
     theory via solid harmonics. Closest theoretical framework for a
     gauge-DOF argument.
   - **Corngold (2002+2004) ANE 29/30** — Peierls/Bickley-Naylor
     algebra in cylinder. If 1/ρ has a clean geometric meaning, one
     of these derivations might show it.
   - **Wio (1984) ANE 11 / Krishnani (1985) ANE 12** — CP kernel
     transformation laws under geometric scaling.

2. **Symbolic derivation**: set up the Peierls integral equation with
   BOTH Lambert-basis P/G and Marshak-basis P/G side by side. Derive
   k_eff at rank-1 in each convention. Compute the leading-order
   difference. What term does Lambert pick up that Marshak misses (or
   gets wrong)?

3. **If derivable**: is there a rank-N generalization that preserves
   the Lambert "trick"? E2.4 showed a NAIVE Lambert-split gives 737%
   err — the trick must be more subtle than "use Lambert everywhere."
   Maybe Lambert at outer + Marshak at inner, with a specific
   normalization relation.

4. **If not derivable**: document why and close Issue #120 with a
   "proven dead end" verdict.

Budget: 2 sessions of 2–4 hours each, first for literature + symbolic
setup, second for derivation or verdict.

### Direction C (Issue #121): Sanchez-Santandrea 2002 PCA sectors

**Why this is second**: E6 tested PCA sectors at BASE quadrature and
found the same ~1% plateau as Legendre WITHOUT scale calibration.
BUT: the scale-gauge argument (L8) applies to ANY basis — so PCA
sectors with **per-sector scale calibration** have been untested.
Whether they break below F.4's RICH-quad structural floor is an
empirical question analogous to E4.2 but for PCA.

**What this would look like** (~1 full session of implementation +
scan):

1. Extend `diag_pca_sectors_hollow_sph.py` to optimize per-sector
   scales (M scales for M sectors).
2. Run at RICH quadrature on the same 6-point reference grid
   (σ_t·R ∈ {5, 10, 20}, ρ ∈ {0.3, 0.5}).
3. Compare against F.4 at matched RICH. Does PCA sectors + scale
   calibration beat F.4's structural floor? Or does it also plateau
   at ~0.07% like rank-(1,1,1)?

If PCA also plateaus above F.4, the "rank-N white-BC on hollow
curvilinear cells has a structural barrier at F.4's accuracy" becomes
a much stronger conclusion. If PCA wins, Direction Q becomes even
more interesting (why do sectors beat continuous bases?).

Budget: 1 session of 3–5 hours (implementation + scan + analysis).

### Not-next-priority but don't forget

- **Cylindrical extension**: untouched. The scale gauge DOF and 1/(3ρ)
  rule MIGHT transfer to hollow cylinder (different Eddington factor:
  ⟨µ²⟩ = 1/2 in 2D?). Would be a quick port once sphere closes.
- **Verification spec**: the existing spec in
  `verification-spec-split-adaptive.md` is obsolete. A new spec will
  be needed IF Direction Q or Direction C yields a shippable closure.
- **Issue #119**: remains CLOSED. F.4 is production.
- **Issue #120**: should be updated with the falsification verdict
  and closed (or left open if Direction Q keeps it alive — see §3).

---

## 3. Session-end housekeeping

At the end of this session, before the next one:

1. **Update Issue #120** with the empirical falsification. Decide
   whether to close it (verdict is "falsified") or leave it open as
   an umbrella for Direction Q. Recommend: close with a link to
   Direction Q if filed as a new issue.
2. **File a new Issue for Direction Q** if the user approves it:
   "Principled derivation of F.4's Lambert/Marshak primitive
   asymmetry." Label `module:cp,module:derivations,type:research`.
3. **Sphinx docs**: no change needed. F.4 is still documented as the
   production closure in the Phase F.5 section. The research log
   lives in `.claude/plans/` and is intentionally separate from
   Sphinx.

---

## 4. Critical gotchas for the next session

1. **Always use RICH quadrature** for any comparison with F.4.
   BASE (n_panels=2, p_order=4, n_ang=32) hides F.4's structural
   floor under ~0.1% quadrature noise and INVENTS cancellation wins.
   Use RICH = (4, 8, 64) or MED = (3, 6, 48) for serious comparisons.

2. **The scale gauge DOF (L8) is real but doesn't break structural
   floors**. It only redistributes error within the
   quadrature-noise-limited error surface. Any new "gauge
   calibration" idea must be tested at RICH quadrature from the start.

3. **The NotImplementedError guard STAYS** until a future closure
   beats F.4 at RICH. Direction Q or Direction C might produce one;
   lift the guard only when there's a shippable closure with a
   verification spec.

4. **The `.claude/agent-memory/numerics-investigator/` memos** contain
   contradictions across sessions because the retraction happened
   after some memos were written. Trust the research log's "CURRENT
   STATE" blockquote and the E4 REVISED VERDICT section. When in
   doubt, re-run the critical measurement at RICH.

5. **Don't dispatch the numerics-investigator on "verify the
   breakthrough"** — there is no breakthrough to verify. Dispatch on
   Direction Q (literature search + symbolic derivation) or
   Direction C (PCA + scale calibration at RICH).

---

## 5. Reading order for the next session

1. This document (top to bottom, ~5 min).
2. Research log `.claude/plans/rank-n-closure-research-log.md`:
   - The top-of-file "CURRENT STATE" blockquote.
   - The "RETRACTION" section (line ~874).
   - "E4 REVISED VERDICT" section (line ~1129).
   - L1–L12, L13–L16 lessons (scan for structural insights).
   - RH1–RH10 rejected hypotheses (don't redo them).
3. The chosen direction (Q or C) dossier:
   - For Q: `hebert_2009_ch3_interface_currents.md` (Sanchez-McCormick
     reciprocity machinery), then the literature candidates above.
   - For C: `diag_pca_sectors_hollow_sph.py` (existing probe), then
     the Sanchez-Santandrea 2002 NSE paper if not already extracted.

---

## 6. Recommended first action for the next session

**Dispatch the literature-researcher agent** to retrieve Bogado Leite
(1998) via interlibrary loan or a DOI resolver, AND to extract any
anisotropic-IC content from Krishnani (1982) and Mohanakrishnan (1982).
Those three references are the most likely to contain either the
Lambert-Marshak mismatch principled derivation or a precedent for
adaptive scale calibration.

While the literature agent works, **start a symbolic SymPy derivation**
of the F.4 closure equation with Lambert vs Marshak P/G side by side.
Goal: derive the k_eff difference at leading order and identify the
specific term(s) that give F.4 its accuracy advantage.

If neither literature nor symbolic derivation yields insight in ~2
hours of work, pivot to Direction C (PCA sectors at RICH).

---

## End of plan
