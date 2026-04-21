# Issue #119 — Rank-N per-face white BC: commit to Marshak basis and close the WS identity

**Branch entering this work:** `investigate/peierls-solver-bugs` (all Phase F commits landed, plus F.5 infra + diagnostics).
**Author of this plan:** Claude Opus 4.7, 2026-04-21.
**Audience:** fresh Claude Code session with zero context. Read §§0-4 in full before touching any code.

---

## 0. One-paragraph summary (read first)

Phase F landed rank-2 per-face white BC for slab + hollow cyl + hollow sphere; the slab case closes the Wigner-Seitz identity `k_eff = k_inf` bit-exactly to the legacy E_2/E_3 formula (dual-route verified), but curved hollow cells leave a 1-13 % residual that the plan §3.4 said rank-N per-face would close. The subsequent F.5 infrastructure commit (`d890a1e`) landed a (2N×2N) transmission matrix and per-face mode primitives for hollow sphere that match the scalar limit bit-exactly at N=1 — BUT the N≥2 closure does not reduce the residual. The numerics-investigator dispatch (commit `b9bc3df`) diagnosed the root cause with HIGH confidence: **measure mismatch — P/G primitives live in the Lambert angular-flux basis while the W transmission matrix lives in the Marshak partial-current basis; at N=1 the mismatch is invisible because P̃_0 ≡ 1.** Seven normalisation recipes were tested via `derivations/diagnostics/diag_rank_n_sph_keff_probe.py`; none closes below 5 % at N=2. The single-fix recipe we haven't tried is reformulating the P/G mode primitives entirely in the canonical Sanchez-McCormick 1982 §III.F partial-current moment basis (μ weight in the integrand + (2n+1) expansion factor, `reflection_marshak` convention). Marshak is the strictly cleaner choice because (a) W is already Marshak, (b) the existing single-surface rank-N path is Marshak, (c) the physical white BC `J⁻ = J⁺` is naturally mode-wise in Marshak, and (d) the `reflection_marshak(N) = diag(1, 3, …, 2N-1)` builder was designed for it. Commit this session to Marshak and rewrite only the per-face mode primitives — the transmission matrix and the tensor-assembly scaffolding stay put.

---

## 1. Context state — what's landed

### 1.1 Commits on `investigate/peierls-solver-bugs` (most recent first)

- `b9bc3df` — **diagnostics** (Issue #119): three scripts + memo pinpointing measure mismatch.
- `d890a1e` — **F.5 infrastructure** (Issue #119): rank-N transmission matrix + per-face mode primitives. Verified at N=1.
- `7eabaec` — Phase F.4 sphere: hollow sphere rank-2 white with `exp(-τ)` chord decomposition.
- `fcc001e` — Phase F.4 cylinder: hollow cyl rank-2 white with Ki_3 chord decomposition.
- `f304f6d` — Phase F.3.5: `solve_peierls_1g(boundary="white_rank2")` API wiring.
- `33e6374` — Phase F.3: slab rank-2 white with T-feedback (bit-exact to legacy `peierls_slab` E_2/E_3 at rtol 1e-13).
- `4217a85` — Phase F.1 + F.2: `inner_radius` + per-surface primitives.

### 1.2 What's verified at N=1

- **Transmission matrix** `compute_hollow_sph_transmission_rank_n(r_0, R, radii, sig_t, n_bc_modes=1, dps)` returns the same 2×2 matrix as scalar `compute_hollow_sph_transmission` — BIT-EXACT (`tests/derivations/test_peierls_rank2_bc.py::TestRank2SlabKEffKInfConvergence::test_rank_n_sph_transmission_N1_matches_scalar_bit_exact`).
- **Per-face mode primitives** `compute_P_esc_{outer,inner}_mode(n=0)` and `compute_G_bc_{outer,inner}_mode(n=0)` match their scalar counterparts BIT-EXACTLY (`test_sphere_mode_primitives_match_scalar_at_mode_0`).
- **Sanchez-McCormick reciprocity** `A_k · W_jk^{mn} = A_j · W_kj^{nm}` (transposed mode indices) holds to 1e-14 (`test_rank_n_sph_transmission_reciprocity_transposed_modes`).
- **Phase F.4 scalar rank-1-per-face**: homogeneous hollow sphere, Σ_t=1, Σ_s=0.5, νΣ_f=0.75, k_inf=1.5, at r_0/R=0.3 gives 3.3 % residual (moderate quadrature n_p=2, p=4, dps=15); at r_0/R=0.1 gives 0.4 %.

### 1.3 What's known broken at N≥2

- Seven normalisation conventions attempted — table in §3 below. None close to ≤ 0.1 %.
- `build_closure_operator(reflection="white", n_bc_modes > 1)` on hollow sphere currently raises `NotImplementedError` with an explicit pointer to Issue #119 (`test_rank_n_white_closure_raises_pending_normalisation` pins this).

### 1.4 Literature confirmation

- Sanchez & McCormick 1982 *NSE* 80 481-535 §III.F (canonical rank-N per-face math).
- Hébert 2020 *Applied Reactor Physics* 3rd ed. Ch. 3.
- **No published rank-N k_inf benchmark for homogeneous hollow sphere exists** — ORPHEUS will be the first once this closes (per literature-researcher's confirmation memo at `.claude/agent-memory/literature-researcher/rank_n_interface_current_canonical.md`).

---

## 2. The bug — measure mismatch

### 2.1 The two moment bases

Every angular distribution ψ(μ) on the hemisphere [0,1] can be expanded in shifted Legendre polynomials P̃_n(μ). Two canonical inner products live on this space:

- **Lambert / angular-flux moment** : `a_n = ∫_0^1 ψ(μ)·P̃_n(μ) dμ` — simply projects ψ onto P̃_n.
- **Marshak / partial-current moment** : `J_n = ∫_0^1 ψ(μ)·μ·P̃_n(μ) dμ` — projects the μ-weighted distribution onto P̃_n.

Gram matrices of the two inner products differ:

```
B^L_{mn} = ∫ P̃_m·P̃_n dμ    = diag(1, 1/3, 1/5, …)      — DIAGONAL
B^μ_{mn} = ∫ μ·P̃_m·P̃_n dμ  = [[0.5, 0.167, 0],
                                 [0.167, 0.167, 0.067],
                                 [0, 0.067, 0.1]]        — NOT DIAGONAL
```

### 2.2 Which component lives in which basis (at commit `d890a1e`)

| Component | File:line | Integrand | Basis |
|---|---|---|---|
| `compute_P_esc_outer_mode(n)` | `peierls_geometry.py:1692-1715` | `sin θ · P̃_n(μ_exit) · K_esc` | **Lambert** |
| `compute_P_esc_inner_mode(n)` | `peierls_geometry.py:1768-1790` | `sin θ · P̃_n(μ_exit) · K_esc` | **Lambert** |
| `compute_G_bc_outer_mode(n)` | `peierls_geometry.py:2062-2093` | `sin θ · P̃_n(μ_s) · e^{-τ}` | **Lambert** |
| `compute_G_bc_inner_mode(n)` | `peierls_geometry.py:2140-2165` | `sin θ · P̃_n(μ_s) · e^{-τ}` | **Lambert** |
| `compute_hollow_sph_transmission_rank_n` | `peierls_geometry.py:2921-3050` | `cos θ · sin θ · P̃_n(cos θ)·P̃_m(·)·e^{-τ}` | **Marshak** (μ = cos θ baked in) |

Proven numerically via σ_t → 0 limit (see `derivations/diagnostics/diag_rank_n_sph_normalisation_measure_mismatch.py`): at the outer-surface observer node, the Lambert-basis P_1 limits to ~0.09 while the Marshak-basis integrand limits to a different value depending on the cos θ factor.

### 2.3 Why N=1 works but N≥2 fails

At N=1 every tensor collapses to scalars (P̃_0 ≡ 1 makes both bases identical at rank 1); the assembled `G·(I−W)⁻¹·P` looks correct even though G,P and W live in different inner products. At N≥2 the off-diagonal couplings of B^μ (0.167, 0.067, …) have no counterpart in the diagonal B^L — so the matrix product `G_lambert · (I−W_marshak)⁻¹ · P_lambert` evaluates to a nonsensical closure.

### 2.4 The seven conventions already tested (all fail to close ≤ 0.1 %)

Probe: R=5, r_0/R=0.3, Σ_t=1, Σ_s=0.5, νΣ_f=0.75 (`derivations/diagnostics/diag_rank_n_sph_keff_probe.py`):

| Convention | N=1 err | N=2 err |
|---|---|---|
| Shipped `(I − W)⁻¹` | 3.031 % | 5.590 % |
| `(I − W)⁻¹ · diag(1, 3, 1, 3)` | 3.031 % | 14.816 % |
| `(I − B^μ⁻¹ W)⁻¹` | 3.031 % | 6.358 % |
| μ-weighted P,G + `(I − W)⁻¹` | 3.031 % | 5.284 % |
| μ-weighted P only | 3.031 % | 5.429 % |
| μ-weighted G only | 3.031 % | 5.419 % |
| Converter C = B^μ(B^L)⁻¹ on P | 3.031 % | similar |

**The fact that NO simple rearrangement closes the identity strongly suggests a SECOND bug stacked on the measure mismatch** — most likely in the per-face divisor normalisation (`R²`/`r_0²` may need mode-dependence via B^μ_{nn}) or in the `(2n+1)` Gelbard factor placement. Diagnostic harness is ready to probe this once the primary fix is in.

---

## 3. Strategic choice: commit to Marshak basis

### 3.1 Why Marshak

Six reasons, priority-ordered:

1. **W is already Marshak** — switching W to Lambert would require rewriting `compute_hollow_sph_transmission_rank_n` and breaking the Sanchez-McCormick 1982 formulas it implements.
2. **Existing single-surface rank-N is Marshak** — `compute_P_esc_mode`, `compute_G_bc_mode` (lines 1862-2081) use `(ρ/R)²` Jacobian + `reflection_marshak(N) = diag(1, 3, …, 2N-1)`. Per-face Marshak keeps the stack consistent.
3. **Physical white BC is mode-wise in Marshak**: `J⁻_m = J⁺_m` is the partial-current equality — no basis change required.
4. **Canonical literature**: Sanchez-McCormick 1982 §III.F, Hébert 2020 Ch. 3, all use partial-current moments.
5. **`reflection_marshak(N) = diag(1, 3, …, 2N-1)`** is the Gelbard `(2n+1)` expansion factor of `ψ(μ) = Σ (2n+1)·a_n·P̃_n(μ)` — designed for Marshak.
6. **Per-face divisor `R²`/`r_0²`** is the sphere-area surface normalisation, which is natural in the partial-current formulation where J^+ = ∫ μ · ψ^+ dμ carries the cosine weight.

### 3.2 One subtle cost — and how to handle it

The scalar rank-1 primitives (`compute_P_esc_outer`, `compute_G_bc_outer`) use the **Mark** convention: `(1/2) · ∫ sin θ · K_esc dθ` (no μ weight; the `1/2` encodes `∫μ dμ`). Mark scalar and Marshak n=0 moment differ by a factor related to `∫μ dμ = 1/2`.

If you naively replace mode-0 with `Marshak n=0 = ∫ sin θ · μ · K_esc dθ`, the **Phase F.4 rank-2 white BC on hollow sphere (3.3 % at r_0/R=0.3) stops being bit-exact** — the `test_hollow_sph_rank2_beats_rank1_mark` pin in `test_peierls_rank2_bc.py` catches this shift.

**Two clean ways to handle it:**

- **(A) Preserve the Mark scalar form and bridge to Marshak for n ≥ 1 only.** Route mode-0 through the existing scalar primitives; for n ≥ 1 use new `_marshak`-suffixed primitives that emit Marshak partial-current moments. In the assembly, apply a (0, 0)-only bridge factor to make mode-0 match Marshak n=0 when it couples into the W·(I-W)⁻¹ inversion.

- **(B) Fully migrate to Marshak and regenerate the Phase F.4 pin values.** The pin is `e_white < e_mark/5 and e_white < ceiling` — the absolute value changes slightly under Marshak but the 5× improvement over rank-1 Mark and the ceiling constraints (1 %, 2 %, 5 % per r_0) should still hold. Update ceilings if needed.

**Recommendation: route (A)** for the first implementation pass — preserves strict bit-exact backwards compatibility with Phase F.4. If the bridge is cleanly expressible, the diff is surgical; if not, pivot to (B) and regenerate pin values.

---

## 4. Detailed implementation recipe

### 4.1 Commit sequence

Three commits, each self-contained:

- **F.5.1** `feat(derivations): Marshak partial-current basis for rank-N per-face P/G (Issue #119)`
- **F.5.2** `feat(derivations): Marshak rank-N closure assembly for hollow sphere (Issue #119)`
- **F.5.3** `test(derivations): rank-N k_eff convergence suite for hollow sphere (Issue #119)`

### 4.2 F.5.1 — Redefine per-face mode primitives in Marshak basis

**Files modified:** `orpheus/derivations/peierls_geometry.py`.

Define new functions (DO NOT drop the current Lambert-basis mode primitives — leave them in place for diagnostic parity; phase them out after closure is verified):

```python
def compute_P_esc_outer_mode_marshak(geometry, r_nodes, radii, sig_t, n_mode, *, n_angular=32, dps=25):
    """Marshak partial-current moment n on the outer surface.

    P_esc_out_marshak^(n)(r_i)
      = 2 · ∫_rays_reaching_outer sin θ · μ_exit · P̃_n(μ_exit) · K_esc(τ) dθ

    At n=0: reduces to 2·∫ sin θ · μ_exit · K_esc dθ (scaled by the
    sphere prefactor; NOT equal to the Mark scalar — see below).
    """
    # Implementation: copy compute_P_esc_outer_mode, insert `mu_exit *`
    # factor into the accumulator. Integrated over full angular range
    # but with Model A guard to skip rays hitting inner first.
```

Similarly for `compute_P_esc_inner_mode_marshak`, `compute_G_bc_outer_mode_marshak`, `compute_G_bc_inner_mode_marshak`.

**Key rules (Marshak convention, per Sanchez-McCormick 1982):**

- Integrand carries `μ · P̃_n(μ)` explicitly.
- **NO** `(ρ/R)²` surface-to-observer Jacobian — that's a different convention (Gelbard DP_{N-1} single-surface; we're doing per-face with μ weight instead).
- **NO** `(2n+1)` Gelbard factor in the integrand — that belongs to the `reflection_marshak` construction.
- Model A cavity handling: rays hitting inner first are excluded from `_outer_marshak`, only rays hitting inner are counted in `_inner_marshak`.

**Verification script (write before the commit):** `derivations/diagnostics/diag_rank_n_sph_marshak_primitives_sigt_zero.py`.

Purpose: at σ_t → 0, verify each primitive reduces to the closed-form Marshak moment:

```
compute_P_esc_outer_mode_marshak(n=0) → 2·sphere_pref · (1 − sin²θ_c)/2
                                       = (1/2)·(1 − (r_0/R)²)
compute_P_esc_outer_mode_marshak(n=1) → 2·sphere_pref · (specific closed form)
```

Expected: all four primitives converge to the hand-computed Marshak moment limits to 1e-10.

### 4.3 F.5.2 — Assembly in `_build_closure_operator_rank_n_white`

**File:** `orpheus/derivations/peierls_geometry.py` (helper currently at line 3356).

Current state: raises `NotImplementedError` via the guard in `build_closure_operator` (line 3170).

Changes:

1. Remove the `NotImplementedError` guard in `build_closure_operator`.
2. Route `_build_closure_operator_rank_n_white` to use the new `_marshak` primitives for n ≥ 1 and keep scalar `compute_P_esc_{outer,inner}` + `compute_G_bc_{outer,inner}` at n = 0 (bridge route A).
3. Drop the `diag(2n+1)` post-multiplication in R. Final form: `R = (I − W)⁻¹`.
4. Per-face divisor stays: `R²` for outer, `r_0²` for inner.
5. The mode-0 bridge factor: if route (A) is chosen, insert a scalar renormaliser that makes the mode-0 rows/columns of G and P match what Marshak n=0 would have produced. Numerical-investigator memo suggests this is `B^μ_{00} / B^L_{00} = 0.5 / 1 = 0.5`. Verify empirically at N=1 that the closure reduces bit-exactly to the Phase F.4 scalar rank-2 white.

**Verification at N=1:** `bc = build_closure_operator(geom, ..., reflection="white", n_bc_modes=1)` MUST produce the same `.as_matrix()` as `reflection="white", n_bc_modes=1` did at commit `7eabaec` (Phase F.4 sphere). The Phase F.4 tests are the regression pins.

### 4.4 F.5.3 — Acceptance tests

**File:** `tests/derivations/test_peierls_rank2_bc.py` (add test class `TestRankNPerFaceHollowSphConvergence`).

Target tests (all at R=1.0, r_0/R ∈ {0.1, 0.2, 0.3}, moderate quadrature n_p=2, p=4, dps=15):

1. `test_rank_n_N1_matches_phase_f4_scalar_bit_exact` — the scalar rank-1-per-face K_bc at `n_bc_modes=1` reproduces the Phase F.4 pin bit-exactly. Regression pin for the Marshak-bridge normalisation.
2. `test_rank_n_monotone_convergence` — k_eff residual decreases monotonically with N ∈ {1, 2, 3, 4}.
3. `test_rank_n_N2_at_r0_0p3_below_0p1_percent` — the acceptance gate: at r_0/R=0.3, N=2 residual ≤ 1e-3 (1000 ppm = 0.1 %), a 33× improvement over N=1. If the measure-unification fix alone doesn't reach this, the "second bug" is still present and needs the follow-on §6 work.
4. `test_rank_n_sanchez_reciprocity` — re-check transposed-index reciprocity holds after any normalisation change.
5. `test_rank_n_solid_limit` — at `inner_radius = 1e-8 · R` (regime B), rank-N closure converges to the solid-cyl/sph rank-N single-surface result (which is the existing `build_white_bc_correction_rank_n`).

**Monte-Carlo cross-check (optional, recommended):** compute the 2×2 scalar W values for hollow sphere r_0/R=0.3, Σ_t=1 via a simple Monte-Carlo ray-tracer (1M samples). Should match the analytical W at 1e-3. If yes, W is unambiguously correct and any remaining residual lives in P/G or the closure assembly — not in transmission.

### 4.5 Sphinx documentation

**File:** `docs/theory/peierls_unified.rst`.

Add a new `:ref:` subsection under the Phase F discussion (near the Key Facts list around line 115):

```
.. _peierls-rank-n-per-face-marshak:

Rank-N per-face Marshak closure (Phase F.5, Issue #119)
=========================================================

[~150 words on the Marshak partial-current moment basis, the (I − W)⁻¹
closure, why Lambert doesn't work, and the bit-exact N=1 regression.]
```

Include the 2×2 vs 4×4 vs 2N×2N W matrix structure, the reciprocity formula with transposed indices, and the derivation of the σ_t → 0 closed-form moments.

---

## 5. Validation plan — convergence target

### 5.1 Acceptance targets

| r_0/R | Rank-1 Mark | Scalar rank-1-per-face (F.4) | **Rank-N target (N ≥ 2)** |
|---|---|---|---|
| 0.1 | 27 % | 0.4 % | ≤ 0.01 % |
| 0.2 | 29 % | 1.2 % | ≤ 0.05 % |
| 0.3 | 31 % | 3.3 % | ≤ 0.1 % |

At N=4, residual should drop to 1e-5 or better at all three r_0 values — the rank-N convergence "saturates" near quadrature accuracy (n_p=2, p=4, dps=15 gives ~1e-4 floor; refined quadrature n_p=8, p=6, dps=25 would reach ~1e-8).

### 5.2 Regression pins (must stay green after changes)

- All 93 Phase F tests.
- All 4 F.5 infrastructure tests from commit `d890a1e`.
- Legacy single-surface rank-N single-surface tests (`tests/derivations/test_peierls_rank_n_*.py`) — unchanged by this work.

### 5.3 If convergence does NOT reach ≤ 0.1 % at N=2

That means the "second bug" identified in §2.4 is real. Candidates:

- **Per-face divisor** `R²`/`r_0²` may need a mode-dependent factor from `B^μ`. Probe: multiply G by `diag(B^μ_{nn})` per surface.
- **(2n+1) placement** in the reflection: post-multiply R by `diag(1, 3, …, 1, 3, …)` after the `(I − W)⁻¹` inversion, not before. Probe: augment reflection builder.
- **G's `exp(-τ)` kernel** may need a `1/μ_s` factor to convert observer-centred angular measure `sin θ dθ` to surface-area measure. Probe: multiply G by `diag(1/μ_s)` integrand.

All three are cheap to test via the existing `diag_rank_n_sph_keff_probe.py` harness (add a new convention row).

### 5.4 Cross-geometry extensibility check (optional, at end of F.5)

Hollow cyl rank-N and slab rank-N use the SAME structural recipe (Marshak per-face primitives + W + (I−W)⁻¹) with geometry-specific P/G/W integrals. Once sphere closes, extending to cyl + slab is mechanical — file follow-up issues #120 (hollow cyl rank-N) and #121 (slab rank-N, subsumes existing #118 if still open).

---

## 6. Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Measure unification alone leaves ~1-5 % residual (second bug) | **High** | Medium | §5.3 three candidate probes ready via `diag_rank_n_sph_keff_probe.py` |
| N=1 bit-exact regression breaks under bridge (A) | Medium | Medium | Route (B) fallback; regenerate Phase F.4 pin values with conservative ceilings |
| Sanchez-McCormick reciprocity breaks after primitive change | Low | High | `test_rank_n_sanchez_reciprocity` runs each commit |
| Single-surface rank-N tests regress | Low | High | Per-face path only touches `_build_closure_operator_rank_n_white` which is hollow-only; single-surface path unchanged |
| mpmath.quad tolerance issues at N=3,4 | Low | Low | Increase `dps` from 20 to 30 for higher modes if GL inner integration flakes |

---

## 7. Out of scope

- **Hollow cylinder rank-N** — same recipe; follow-up issue after sphere closes.
- **Slab rank-N per-face** — subsumes `#118`; rank-1 per-face already closes WS at legacy precision, so F.5 for slab is cosmetic extensibility. Follow-up issue.
- **Multigroup rank-N** — blocked on `#104`.
- **Specular / albedo at rank-N** — trivial once W is the right matrix; 1-line adds.

---

## 8. Reading order for the next session

1. **`.claude/lessons.md`** (unconditional per CLAUDE.md cardinal rule).
2. **`mcp__nexus__session_briefing()`** (unconditional).
3. **This plan** in full (§§0-6).
4. **`.claude/agent-memory/numerics-investigator/peierls_rank_n_measure_bug.md`** — diagnostic session notes.
5. **`.claude/agent-memory/literature-researcher/rank_n_interface_current_canonical.md`** — Sanchez 1982 canonical formulas + reciprocity with transposed modes.
6. **GitHub Issue #119 comments** — full progress trace, three comments so far.
7. **`derivations/diagnostics/diag_rank_n_sph_normalisation_measure_mismatch.py`** — Gram matrix proof, σ_t → 0 tests.
8. **`derivations/diagnostics/diag_rank_n_sph_keff_probe.py`** — seven tested conventions with k_eff residuals. Contains `test_N1_baseline_residual` (regression pin) and `test_N2_improves_over_N1` (acceptance target).
9. **`orpheus/derivations/peierls_geometry.py`**:
   - Lines 1610-1790: per-face mode primitives (`compute_P_esc_{outer,inner}_mode`).
   - Lines 2040-2175: per-face mode response (`compute_G_bc_{outer,inner}_mode`).
   - Lines 2921-3050: rank-N transmission `compute_hollow_sph_transmission_rank_n`.
   - Lines 3160-3185: `build_closure_operator` guard.
   - Lines 3356-3450: `_build_closure_operator_rank_n_white` assembly helper.
10. **`tests/derivations/test_peierls_rank2_bc.py`** — all F.3-F.5 tests, particularly `TestRank2SlabKEffKInfConvergence` class at the end which has F.5 infra tests and the slow convergence test.

Then start at §4.2 of this plan (F.5.1 — redefine per-face mode primitives in Marshak basis) and execute.

---

## 9. Acceptance criteria — when is Issue #119 done?

### 9.1 F.5.1 + F.5.2 + F.5.3 (single session, ~3-5 hours)

- [ ] Four new `_marshak`-suffixed per-face mode primitives for hollow sphere.
- [ ] `diag_rank_n_sph_marshak_primitives_sigt_zero.py` verification script, σ_t → 0 closed forms match at 1e-10.
- [ ] `_build_closure_operator_rank_n_white` assembles `G · (I − W)⁻¹ · P` with Marshak P/G for n ≥ 1 + mode-0 bridge.
- [ ] `NotImplementedError` guard in `build_closure_operator` lifted.
- [ ] All 5 acceptance tests from §4.4 pass.
- [ ] All 93 Phase F tests + 4 F.5 infra tests green.
- [ ] Monte-Carlo W cross-check (optional but strongly recommended) within 1e-3.
- [ ] Sphinx `:ref:peierls-rank-n-per-face-marshak` subsection lands.
- [ ] Commit: single commit or three per §4.1 depending on how cleanly it can be decomposed.

### 9.2 Close-out

- [ ] Issue #119 closed with final-numbers summary comment.
- [ ] Follow-up issues filed: #120 hollow cyl rank-N, #121 slab rank-N (if needed).
- [ ] Phase F Sphinx Key Facts updated with the final N-convergence table.
- [ ] Consider filing a short note for publication (no published rank-N k_inf benchmark for homogeneous hollow sphere exists).

---

## 10. File inventory (what to touch)

### Modified

- `orpheus/derivations/peierls_geometry.py` — **major edits**:
  - Add 4 new `_marshak`-suffixed primitives (P_esc outer/inner, G_bc outer/inner).
  - Rewrite `_build_closure_operator_rank_n_white` assembly.
  - Lift `NotImplementedError` guard in `build_closure_operator`.
- `tests/derivations/test_peierls_rank2_bc.py` — add `TestRankNPerFaceHollowSphConvergence` class with 5 acceptance tests.
- `docs/theory/peierls_unified.rst` — new `:ref:peierls-rank-n-per-face-marshak` subsection.

### Added

- `derivations/diagnostics/diag_rank_n_sph_marshak_primitives_sigt_zero.py` — σ_t → 0 Marshak moment verification (promote to a test if it proves useful).

### Not touched (by design)

- `compute_hollow_sph_transmission_rank_n` (already correct Marshak).
- `compute_hollow_cyl_transmission` (hollow cyl rank-2 — separate follow-up for rank-N).
- `compute_slab_transmission` (slab — separate follow-up).
- All single-surface rank-N code (`compute_P_esc_mode`, `compute_G_bc_mode`, `build_white_bc_correction_rank_n`).

---

## 11. Estimated effort

- **Session 1 (F.5.1 + F.5.2)**: ~200 lines of primitives + ~150 lines of assembly + ~80 lines of σ_t → 0 verification. Estimated 3-4 hours. Risk: medium (measure unification is the proven first hit, but second-bug probes may be needed).
- **Session 2 (F.5.3 if second-bug probes needed)**: ~100 lines of fix + ~50 lines of additional test + Sphinx update. Estimated 2 hours. Risk: low once root cause is re-identified.

Total: 3-6 hours across 1-2 focused sessions.

---

## End of plan

Issue #119 is the final correctness piece for curved hollow geometries. The diagnostic phase is done; the fix recipe is known; the regression harness is ready. Committing to Marshak basis aligns the stack with literature, preserves the existing infrastructure, and makes the implementation mechanical. Only the P/G primitives need rewriting.

After F.5 lands:
- Hollow sphere white BC closes WS identity to machine precision at N ≥ 2.
- All three Class-A geometries (slab + hollow cyl + hollow sphere) achieve quadrature-limited k_eff = k_inf.
- Hollow cyl rank-N becomes a mechanical follow-up using the same recipe.
- Slab rank-N subsumes #118 (currently a `NotImplementedError` in `compute_P_esc_mode` for slab).
- Path clears for multigroup rank-N BC (needs #104 first).

The hard part — identifying the root cause — is already done. Next session is implementation + validation.
