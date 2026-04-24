# Plan: rank-N closure on Class B (solid cyl/sph), MR×MG test posture

**Author**: Plan agent (Claude Opus 4.7), 2026-04-24.
**Issues**: #100 (sphere rank-1 insufficient), #103 (rank-N closure for cyl+sphere), #112 (rank-N 3D normalization), #114 (ρ-subdivision — prerequisite check).

## 1. Executive summary

- **What we are building**: a multi-region × multi-group test posture for `build_white_bc_correction_rank_n` on **Class B** (solid cylinder, solid sphere) cells. Target: a small set of pytest classes that compare rank-1 Mark vs rank-N (`n_bc_modes` ∈ {1, 2, 3, 5, 8}) k_eff on the same cell, with the analytical CP-matrix `k_inf` from `cp_cylinder._build_case` / `cp_sphere._build_case` as the absolute reference.
- **Why now**: Class B currently ships zero continuous Peierls references (the rank-1 Mark 21 % floor is too loose; `peierls_cases.build_one_surface_compact_case` raises `NotImplementedError`). The 2026-04-22 falsification (research-log L21) closed the rank-N path **only for Class A two-surface hollow geometry** — it never tested rank-N on Class B at all, and it never tested **any** rank-N configuration in MR×MG.
- **Test posture from the start**: 1G/1-region (sanity / reproduce 2026-04-18 single-region table), 2G/1-region (sanity), and the new regime — **2G/2-region rank-N on solid cyl/sph** with non-trivial Σ_t,A vs Σ_t,B difference. Decision is binary: either rank-N still doesn't beat rank-1 Mark (falsification cleanly extends, close Class B with rank-1 Mark + loose tolerance or pursue Issue #101) or a hidden bug surfaces (Issue #131 precedent — apply probe-cascade).
- **Expected duration**: 3–5 sessions. The infrastructure is largely already there (see §4); cost is dominated by quadrature wall time and the probe-cascade investigation if Outcome B fires.
- **Auxiliary deliverable**: regardless of outcome, a regression test suite that pins Class B's rank-N behaviour in MR×MG so the question "does rank-N help in MR×MG on Class B?" never has to be re-asked.

## 2. Background and prior art

### 2.1 What 2026-04-22 proved

The hollow-sphere research log (`.claude/plans/rank-n-closure-research-log.md`) ran 13 experiments + 6 cross-domain frame attacks across 2026-04-21..22. **L21** is the close-out lesson:

> Three cross-domain frames tested this session (Frames 1, 2 step 1, and 6), all falsified cleanly. Combined with the earlier falsifications in this research program — Issue #120 split-basis (L16 retraction), Issue #121 PCA sectors at RICH, Issue #122 Lambert/Marshak rank-N generalization — every structural approach to beat F.4 via angular basis refinement has failed. The rank-N plateau at ~0.1 % (vs F.4's 0.003 % floor) is a hard barrier from the Schur-reduction nature of F.4, **not a basis-choice problem**.

### 2.2 What 2026-04-22 did NOT prove

- L21 closes **rank-N per-face on hollow (Class A) cells against the F.4 reference**. The reference there was the F.4 closure (rank-1 per face × 2-face transmission matrix), not rank-1 Mark.
- L21 was tested on hollow sphere only; cylinder was deferred ("with L19 in hand, any cylinder scan must start at RICH+panels").
- L21 was tested at 1G × 1-region only across the 6-point (σ_t·R, r_0/R) grid. **Multi-region with non-trivial inter-region Σ_t was never exercised.**
- Class B (solid cyl/sph, 1 surface) was never touched. F.4 collapses to rank-1 Mark on Class B (no second face), so the L21 reference (F.4) does not exist as a benchmark. The relevant rank-N comparison on Class B is rank-N vs rank-1 Mark vs analytical CP-matrix k_inf.

### 2.3 The Issue #131 precedent

`docs/theory/peierls_unified.rst §theory-peierls-slab-polar-g5-diagnosis`: the 2-region 2-group slab parity gap (1.5 %) was invisible in the 1G/1-region and 2G/1-region tests (which passed at 1e-8). It only surfaced once 2G × 2-region was exercised. Root cause: `compute_P_esc_outer/inner` + `compute_G_bc_outer/inner` had separate branches for `len(radii) == 1` (closed-form ½·E₂) vs `len(radii) > 1` (finite-N GL) — and the multi-region branch was a wasteful underconvergent quadrature where a closed-form integral existed.

The same anti-pattern could exist on the curvilinear side: the rank-N primitives `compute_P_esc_mode` / `compute_G_bc_mode` may have closed-form structure for special cases that a finite-N GL is silently approximating. Single-region testing would not discriminate between "GL converged" and "closed-form silently used"; multi-region with non-trivial Σ_t breakpoints forces the integrand to span multiple piecewise pieces, exposing GL underconvergence.

## 3. The hypothesis under test

**Multi-region × multi-group rank-N on Class B may surface bugs the 2026-04-22 single-region single-group falsification missed, OR may extend the falsification cleanly. Either outcome is informative.**

Specifically:

- **H_A (clean extension)**: rank-N on Class B in MR×MG plateaus at the same 1G×1-region floor (Issue #112's table: rank-2 sphere R=1 MFP ~1.22 %, rank-2 cyl ~8.3 %, etc.). Conclusion: the falsification truly extends to Class B MR×MG. Class B references close as "rank-1 Mark, accept the 21 % floor" or pursue Issue #101 (analytical chord-based Ki₁).
- **H_B (hidden bug — Issue #131 template)**: rank-N exhibits behaviour in MR×MG (e.g., gap that grows with Σ_t,A − Σ_t,B, or a sign flip between adjacent ranks) that 1G×1-region missed. Probe-cascade to localise.
- **H_C (rank-N actually beats rank-1 Mark in MR×MG)**: would invalidate the 1G×1-region falsification claim (which was, recall, recorded against `k_inf = 1.5` not against an actual MR×MG analytical reference). Open new investigation.

H_A is the strong prior. H_B is the user's worry — explicit echo of the Issue #131 lesson. H_C is unlikely but cannot be ruled out without doing the work.

## 4. Architecture — what exists, what needs to change

### 4.1 What is already wired (no new code needed)

The infrastructure is mostly in place — this is a key finding.

- **Solid Class B with `n_bc_modes ≥ 2` is already callable** through `peierls_geometry.solve_peierls_mg(..., boundary="white_rank1_mark", n_bc_modes=N)`. The chain is:
  - `solve_peierls_mg → _build_full_K_per_group → build_white_bc_correction_rank_n` (`peierls_geometry.py:4119-4125`) — `closure == "white_rank1_mark"` branch; `n_bc_modes` is already plumbed through.
  - `build_white_bc_correction_rank_n → build_closure_operator(reflection="marshak", n_bc_modes=N)` for n_bc_modes ≥ 1 (`peierls_geometry.py:3890-4009`).
  - `build_closure_operator` falls through the `use_rank2 = (… and geometry.n_surfaces == 2)` branch when `n_surfaces == 1`, going to the legacy single-surface assembly using `compute_P_esc + compute_G_bc` (mode 0) and `compute_P_esc_mode + compute_G_bc_mode` (modes ≥ 1). Both mode-≥1 primitives are implemented for both `cylinder-1d` and `sphere-1d` and dispatch correctly on solid cells.
- **Multi-region is supported**: `solve_peierls_mg` walks per-region `sig_t[r, g]` correctly, the K-matrix builder calls `geometry.optical_depth_along_ray(r_i, cos_om, rho_max_val, radii, sig_t)` which traverses the piecewise-constant Σ_t along each ray.
- **Multi-group is supported**: Issue #104 (commits 1+2, 2026-04-24) shipped `solve_peierls_mg` with full ng support; the rank-N (`n_bc_modes`) parameter is already plumbed through (`peierls_geometry.py:4180,4305`).
- **Analytical reference**: `cp_cylinder._build_case(ng_key, n_regions)` and `cp_sphere._build_case(ng_key, n_regions)` already produce `k_inf` for {1G, 2G, 4G} × {1, 2, 4} regions via the analytical CP-matrix path (`kinf_from_cp` in `_eigenvalue.py`). Reusing the same `_RADII` and `_MAT_IDS` keeps the test cells aligned with shipped CP solver verification cases.

### 4.2 What needs lifting

- **Shape wrappers `solve_peierls_cylinder_1g/_mg` and `solve_peierls_sphere_1g/_mg` do NOT expose `n_bc_modes`** (`peierls_cylinder.py:263-332`, `peierls_cylinder.py:365-421`, sphere mirrors). The test driver should either (a) call `peierls_geometry.solve_peierls_mg` directly for the rank-N sweep, or (b) thread a single `n_bc_modes` kwarg through the four shape wrappers. Recommendation: (a) for the test code (avoids API drift if the experiment falsifies), (b) only if the experiment confirms a rank-N win and the result needs to ship as a new reference.
- **`peierls_cases.build_one_surface_compact_case` raises `NotImplementedError`** (`peierls_cases.py:303-332`). This stays raised for the test phase — the test driver bypasses the registry and calls `solve_peierls_mg` directly. Only at acceptance (Outcome B/C) would this builder lift.
- **Class B currently has no `_build_peierls_cylinder_solid_case` registered**. The unused `_build_peierls_cylinder_case` (`peierls_cylinder.py:431`) and `_build_peierls_sphere_case` (`peierls_sphere.py:436`) exist but are not called from `peierls_cases.cases()`. They are 1G-only and would need lifting to MR×MG if Outcome C ever fires.
- **`compute_P_esc_mode` / `compute_G_bc_mode` audit for closed-form anti-pattern (Issue #131 echo)**: the inner-loop branch on `len(radii) == 1` for the optical-depth walker (`peierls_geometry.py:2817-2822` and `2846-2853`) is the same shape as the slab bug. For curvilinear the walker is non-trivial (chord crossings of annular boundaries) so the underlying integral is unlikely to have a closed form, but the audit must be performed in Probe D regardless. This is **not** a code change up front — it's a probe target if H_B fires.

### 4.3 Existing non-Class-B regression tests to leverage

`tests/derivations/test_peierls_rank_n_bc.py` already exercises rank-N on solid sphere and cylinder, but only at 1G × 1-region with vacuum and white BC at fixed (R, ν Σ_f). Eleven xfails are pinned there for the unresolved Issue #112 (sphere thin-cell plateau, cylinder N≥3 divergence). The new MR×MG tests live in a sibling file (`tests/derivations/test_peierls_rank_n_class_b_mr_mg.py`) so they don't entangle with the 1G×1G xfail bookkeeping.

## 5. Test plan

### 5.1 Test fixtures

Two fixtures, each parametrising over `n_bc_modes`:

- **CylSolidRankNFixture(ng_key, n_regions, n_bc_modes)** — parametrised over (`ng_key`, `n_regions`, `n_bc_modes`). Uses `cp_cylinder._RADII` and `cp_cylinder._MAT_IDS` so the geometry / XS / radii align bit-exactly with `cp_cylinder._build_case(ng_key, n_regions).k_inf`. Calls `peierls_geometry.solve_peierls_mg(CYLINDER_1D, …, boundary="white_rank1_mark", n_bc_modes=N)`.
- **SphSolidRankNFixture(ng_key, n_regions, n_bc_modes)** — sphere analogue, uses `cp_sphere._RADII` / `cp_sphere._MAT_IDS`, calls `solve_peierls_mg(SPHERE_1D, …)`.

### 5.2 Test classes

**A. Sanity baselines (must reproduce 2026-04-18 falsification)**

- `TestCylSolid1G1RegSanityBaseline` and `TestSphSolid1G1RegSanityBaseline` — `ng_key="1g"`, `n_regions=1`. Parametrised over `n_bc_modes ∈ {1, 2, 3, 5, 8}`. Pin |k_eff − k_inf| against Issue #112 published table per-rank. Sanity gate: rank-1 reproduces the 21 % cyl / 27 % sph at R=1 MFP; rank-2 reproduces the ~1.22 % sphere / 8.3 % cyl improvement; rank-3 cylinder reproduces the 27 % divergence (xfail-pinned to Issue #112 Phase C).

**B. 2G 1-region sanity**

- `TestCylSolid2G1RegSanity` / `TestSphSolid2G1RegSanity` — `ng_key="2g"`, `n_regions=1`. Parametrised over `n_bc_modes ∈ {1, 2, 3}`. Reference: `cp_cylinder._build_case("2g", 1).k_inf`. Asserts the 2G/1-region behaviour mirrors the 1G/1-region pattern (|k_eff − k_inf| within structural floor). If this disagrees with the 1G pattern → suspect 2G ng-coupling pre-bug.

**C. 2G 2-region — the new regime (the load-bearing test)**

- `TestCylSolid2G2RegRankN` / `TestSphSolid2G2RegRankN` — `ng_key="2g"`, `n_regions=2`. Parametrised over `n_bc_modes ∈ {1, 2, 3, 5, 8}`. Reference: `cp_cylinder._build_case("2g", 2).k_inf` (a fuel + moderator config; `LAYOUTS[2]` gives mat_ids `[2, 0]` from `_xs_library`).
  - **Test C.1**: |k_eff(rank-N) − k_inf| sweep — pin the curve. Outcome A: matches 1G structural pattern. Outcome B: gap that doesn't reproduce 1G pattern.
  - **Test C.2**: signed-error monotonicity — does rank-N improve or worsen vs rank-1 (per L19)?
  - **Test C.3**: row-sum identity — does (K_vol + K_bc) · 1 ≈ Σ_t (per region)? This is conservation; the rank-N primitives should improve this with N (cf. `tests/derivations/test_peierls_rank_n_conservation.py`).

**D. 2G 4-region rank-N (parametrised, slow)**

- `TestCylSolid2G4RegRankN` / `TestSphSolid2G4RegRankN` — `ng_key="2g"`, `n_regions=4`. Reference: `cp_cylinder._build_case("2g", 4).k_inf`. `@pytest.mark.slow`. Parametrised over `n_bc_modes ∈ {1, 2, 3}` only (cost). Establishes the MR×MG behaviour with more region heterogeneity.

### 5.3 Quadrature posture

Per L19 (Issue #123), any rank-N closure comparison must run at ≥ 2 quadratures. The new tests run at:

- **BASE**: `n_panels_per_region=2, p_order=3, n_angular=24, n_rho=24, n_surf_quad=24, dps=15`. Cost: ~5–10 s per cyl/sph k_eff at 1G/1R; 2× at 2G; 4× at 2-region (block matrix doubles). Total fixture-build time at BASE for the full 2G/2-region rank-N sweep: ~5 min.
- **RICH**: `n_panels_per_region=4, p_order=5, n_angular=64, n_rho=48, n_surf_quad=64, dps=20`. Cost: ~30–60 s per evaluation; per-test time ~5 min. `@pytest.mark.slow`.

Signed-err stability under BASE→RICH refinement is the L19 primary acceptance gate.

### 5.4 Diagnostic probes (only invoked if H_B fires)

If the 2G/2-region rank-N gap exceeds the 1G/1-region structural floor by > 5×, run probe-cascade per `.claude/skills/probe-cascade/SKILL.md`:

- **Probe A** (1G 2-region rank-N): drop multi-group. If gap persists → bug is in MR coupling, not MG.
- **Probe B** (2G 2-region rank-1, vacuum BC): drop closure. If gap persists → bug is in volume kernel for MR×MG, not closure. Cross-check: the existing 2G/2-region vacuum cyl/sph Peierls solves are gated against analytical references? They are not — Class B has no Peierls references at all today. So this probe is partially exposed only by Issue #131-style closed-form audit.
- **Probe C** (2G 2-region rank-N, homogeneous Σ_t,A = Σ_t,B): isolate inter-region Σ_t difference from MR mesh. If gap vanishes when Σ_t is homogeneous → bug is in the optical-depth walker or rank-N primitive's response to piecewise Σ_t.
- **Probe D** (rank-N primitive convergence under quadrature refinement at the failing config): does `compute_P_esc_mode` / `compute_G_bc_mode` plateau under `n_angular` refinement? If yes → look for closed-form structure (the Issue #131 anti-pattern).
- **Probe E** (candidate fix as pure function): if Probe D pins down a closed form, write a fix as a probe and validate against the failing config. Promotion-ready.

Each probe lives at `derivations/diagnostics/diag_class_b_rank_n_probe_{a,b,c,d,e}_*.py`.

## 6. Decision tree

| Outcome | Trigger | Action |
|---------|---------|--------|
| **A — clean extension** | Test C.1 shows rank-N does NOT improve significantly over rank-1 in 2G/2-region; 1G/1-region pattern preserved | Document falsification extension in Sphinx (new subsection in `peierls-rank-n-per-face-closeout` or sibling). **Issue #100, #103 close** with verdict "rank-N falsified across single-region single-group AND multi-region multi-group on Class B". `peierls_cases.build_one_surface_compact_case` stays raising `NotImplementedError` until Issue #101 (chord-based Ki₁ analytical) lands. Optional: add Class B references at very thick R (R=10+ MFP) with rank-1 Mark and a loose tolerance (5 % at 5 MFP; 1 % at 10 MFP). |
| **B — hidden bug** | Probe-cascade finds a localised bug (e.g., closed-form-avoidance in rank-N primitive's MR branch) | Fix landed as new commit; rerun rank-N sweep. The 2026-04-22 falsification's claim "rank-N can't beat rank-1 Mark on Class B" has to be RE-tested post-fix. Open new sessions: re-run the full rank-N exploration on the corrected code. **Issues #100, #103 stay open**, retitled to reflect the fix scope. |
| **C — rank-N actually beats rank-1 Mark in MR×MG** | C.1 / C.2 show monotone rank-N improvement on Class B that 1G/1-region single-region missed | Open follow-up investigation. The 2026-04-18 1G/1-region single-region table (`build_white_bc_correction_rank_n` docstring lines 3934–3961) becomes anomalous and would need explanation. **Issues #100, #103 stay open**, with new comment recording the MR×MG-specific evidence. Likely intermediate to filing a new Class B continuous reference. |

Outcome A is the strong prior. Outcome B is the user's stated concern. Outcome C is unlikely but cannot be ruled out a priori.

## 7. Acceptance criteria

The session is complete (regardless of outcome) when:

- (a) **Regression test suite landed**: at minimum `tests/derivations/test_peierls_rank_n_class_b_mr_mg.py` with the 2G/2-region cyl + sph rank-N parametrisation. Tests pass at the BASE quadrature; RICH-quadrature variants `@pytest.mark.slow`.
- (b) **L19 signed-error stability table** for rank-N on Class B at the 2G/2-region anchor (R=1 MFP, σ_t·R typical, fuel/moderator XS), at BASE × RICH quadratures.
- (c) **Sphinx documentation update** in `docs/theory/peierls_unified.rst`. Outcome A: a new subsection `:ref:peierls-rank-n-class-b-mr-mg-closeout` extending the existing F.5 Phase close-out to Class B + MR×MG. Outcome B: a new subsection `:ref:peierls-class-b-rank-n-bug-fix-N` documenting the bug + fix per Issue #131 template (which is the Sphinx `theory-peierls-slab-polar-g5-diagnosis` precedent).
- (d) **Issues updated**:
  - Outcome A: #100, #103 closed with the "extends to MR×MG" rationale. #112 stays open or closes per the Issue #112 Phase A/B/C status separately (Stepanek slab calibration is independent).
  - Outcome B: comment on #100/#103 with the bug reproducer, fix link, post-fix sweep table; possibly retitle.
  - Outcome C: comment with the new evidence, new follow-up issue filed for the structural investigation.
- (e) **Research log update**: `.claude/plans/rank-n-closure-research-log.md` gets a new "Class B MR×MG closeout" section recording outcome.

## 8. Risks and prerequisites

### 8.1 Prerequisites

- **Issue #114 (ρ-subdivision)**: known issue for the curvilinear K-matrix when the radial quadrature does not subdivide at panel-boundary crossings — sub-polynomial convergence in `n_rho`, ~1–5 % error. **Currently OPEN.** Verify status before starting the C-class tests; if not resolved, the BASE-quadrature signed errors will inherit the #114 floor and L19 stability assessments will be polluted. Mitigation: use RICH quadrature (`n_rho=48+`) where #114's floor is sub-1 % and use signed-err deltas not absolute errors.
- **Issue #131 fix is in production**: confirmed shipped (closed 2026-04-24; `_SLAB_VIA_UNIFIED` defaults `True`). Slab is not a Class B target so #131 does not directly apply, but the anti-pattern audit informs probe D.
- **Issue #104 multi-group machinery**: shipped (closed 2026-04-24); `solve_peierls_mg` with `n_bc_modes` plumbing through is verified.
- **Probe-cascade skill**: must be loaded explicitly into the implementation session (`.claude/skills/probe-cascade/SKILL.md`).

### 8.2 Risks

- **R1 — Quadrature-noise contamination of L19 signed errors at BASE on cyl thin cells (cf. L20)**: ULTRA quadrature is unreachable at 120 s/point on devcontainer for σ_t·R = 5 cyl reference points. Mitigation: only L19-gate at σ_t·R ≥ 5; document σ_t·R ≤ 2.5 results as "BASE-only, do not trust signed sign". Class B 2G/2-region default cell is R=1 MFP per `cp_cylinder._RADII[2] = [0.5, 1.0]`, so this risk applies acutely to the headline test. **Mitigation**: also test 2G/2-region at thicker R via a synthetic XS scale-up (e.g., 5× Σ_t,A, 5× Σ_t,B) to push σ_t·R into the L19-resolvable regime.
- **R2 — Cylinder rank-N divergence at N ≥ 3 (Issue #112 known)**: the 1G/1-region cylinder rank-N table in `build_white_bc_correction_rank_n` docstring lines 3953–3961 shows N=3 cyl R=1 MFP at 26.7 % and N=8 at 107 %. This is structural per Issue #112 Phase C (Knyazev Ki_{k+2}). The MR×MG tests must avoid N ≥ 3 on cylinder thin cells unless Phase C lands first. Mitigation: parametrise rank-N to {1, 2, 3, 5} for cyl, {1, 2, 3, 5, 8} for sphere; mark N ≥ 3 cyl thin-cell points as expected-divergence.
- **R3 — Test wall time**: 5 ranks × 2 geometries × 2 ng × 2 region-counts × 2 quadratures = 80 evaluations at 5–60 s each = 5–60 min total. This is a slow test suite. Mitigation: BASE-only by default, RICH `@pytest.mark.slow`, ULTRA off-suite manual.
- **R4 — Probe-cascade cost if H_B fires**: probe sequences A–F at modest quadrature can take an additional 2–4 hours of investigation per the Issue #131 precedent. Budget for this in the session count.

### 8.3 Non-goals

- **Not** lifting `peierls_cases.build_one_surface_compact_case` to register references unless Outcome B/C fires AND the resulting rank-N converges to a publishable tolerance. If Outcome A fires, the registration helper stays unimplemented; Class B remains "no continuous Peierls references" until Issue #101 lands.
- **Not** revisiting the Class A hollow rank-N falsification. L21 stands; this plan is orthogonal.
- **Not** implementing Issue #112 Phase A/B/C (Stepanek slab calibration → canonical sphere → Knyazev cylinder Ki_{k+2}). Those phases are independent and can run in parallel; the test posture here is agnostic to which phase is shipped.

## 9. Estimated budget

- **Best case (Outcome A)**: 2 sessions. S1 = test driver + BASE sweep + sanity-baseline reproduction + initial 2G/2-region table. S2 = RICH sweep + L19 stability + Sphinx documentation + issue closure.
- **Mid case (Outcome A with Issue #114 quirks)**: 3 sessions (add a third for documenting the σ_t·R-dependent mitigation).
- **Worst case (Outcome B)**: 4–5 sessions. S1 same; S2 detect MR×MG anomaly; S3 probe-cascade A→D; S4 candidate-fix Probe E + integrated rerun; S5 Sphinx + tests promotion + research-log + issue updates.
- **Outcome C is open-ended** but should produce a "evidence + new issue filed" deliverable in 2 sessions even if no fix lands.

LoC delta:

- New test file: ~400–600 LoC. (matches `test_peierls_rank_n_bc.py`'s scope.)
- Probe diagnostics if needed: ~200 LoC each, 4–5 probes.
- Sphinx: ~50–150 lines depending on outcome.
- Production code: 0 LoC if Outcome A; ~50–200 LoC if Outcome B (the targeted fix).

Commit count: 3–6, scaled by outcome.

---

## Critical Files for Implementation

- `orpheus/derivations/peierls_geometry.py` (the rank-N closure + `solve_peierls_mg` + `build_white_bc_correction_rank_n` + `compute_P_esc_mode` / `compute_G_bc_mode` — all the surface area for the test driver and any probe-cascade fix)
- `orpheus/derivations/peierls_cases.py` (only modified at acceptance — touched only if Outcome B/C fires)
- `orpheus/derivations/cp_cylinder.py` and `orpheus/derivations/cp_sphere.py` (analytical k_inf reference via `_build_case`; reused as test gold)
- `tests/derivations/test_peierls_rank_n_class_b_mr_mg.py` (NEW; the test deliverable)
- `docs/theory/peierls_unified.rst` (Sphinx update — new subsection at acceptance)
