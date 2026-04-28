# Plan: Phase 4 — Multi-bounce specular rollout (cyl + slab) and continuous-µ reformulation

**Author**: Claude Opus 4.7, 2026-04-28.
**Branch (proposed)**: `feature/peierls-specular-mb-phase4` (off `main` after `feature/peierls-specular-bc` merges).
**Predecessor**: branch `feature/peierls-specular-bc` (commit `9178cc6`) — sphere multi-bounce shipped.
**Inputs**: two numerics-investigator reports just landed:

- `specular_mb_overshoot_root_cause.md` — sphere multi-bounce overshoot at N≥4 is a **fundamental matrix-Galerkin divergence** (resolvent norm grows unboundedly because `1/(1-e^(-σ·2Rµ))` has a singularity at grazing µ→0 that the matrix inverse cannot resolve).
- `specular_mb_phase4_cyl_slab.md` — cyl/slab T derivations + verdicts:
  - **Slab MB**: SAFE TO SHIP at any N (geometric immunity — slab chord = L/µ → ∞ at grazing, exponential transmission decay; off-diagonal block-T structure caps `ρ(T·R) ≤ 0.08` across all N at thin τ_L=2.5).
  - **Cyl MB**: same overshoot as sphere (different mechanism — R-conditioning + multi-bounce amplification rather than continuous-µ divergence). Same N∈{1,2,3} envelope + UserWarning at N≥4.

## 1. Executive summary

- **What we are shipping in Phase 4**: cyl + slab `boundary="specular_multibounce"` closures, plus a tightened sphere warning, plus regression-test promotion of the diagnostic ladders.
- **What we are NOT shipping in Phase 4**: the continuous-µ reformulation (Phase 5 — research-grade rework requiring a literature pull and substantial re-architecture; out of scope for one session).
- **Expected duration**: 2-3 sessions for Phase 4. Phase 5 is open-ended (literature first).
- **Why this split**: slab MB is a clean shippable win (no user-facing pathology). Cyl MB matches sphere behavior — same envelope, same warning, same N=1=Hébert algebraic identity. These two fill out the multi-bounce family across all three geometries. The continuous-µ reformulation is the "real fix" but requires significantly more design work and is the right next research direction.

## 2. Background — what's known after the two investigations

### 2.1 Sphere multi-bounce overshoot (investigation #1)

**Verdict: FUNDAMENTAL, not fixable in matrix form.**

Empirical observation:
```
thin sphere τ_R=2.5: N=1 -0.27%; N=4 +0.43%; N=8 +5.62%
very thin τ_R=1:    N=1 -0.13%; N=4 +0.44%; N=8 +9.48%
```

Root cause (operator-norm proof + 12 diagnostics):
- The continuous limit of `(I - T·R)⁻¹` in basis-coef space is the multiplication operator `f(µ) → f(µ)/(1 − e^(−σ·2Rµ))`.
- This operator **diverges pointwise at µ = 0** (grazing rays — chord 2Rµ → 0 → no attenuation → infinite-bounce summation).
- In the true continuous-µ integral the µ-weight cancels this singularity (`µ/(1−e^(−2σRµ)) → 1/(2σR)` finite).
- **But the matrix-Galerkin projection** distributes the µ-weight across separate factors (P, T, G) and the matrix inverse `(I − T·R)⁻¹` does not preserve the cancellation.
- As N grows, the basis resolves grazing modes more sharply, exposing the divergence. `‖(I−TR)⁻¹‖₂` at thin τ_R=2.5: 1.08 (N=1) → 53.9 (N=25). Unbounded.

MC ground truth confirms physical specular = k_inf at homogeneous (so the user's intuition is correct — the problem is the rank-N construction, not the physics).

### 2.2 Cyl multi-bounce (investigation #2)

**T_mn^cyl Knyazev expansion:**
```
T_mn^cyl = (4/π) ∫_0^(π/2) cos α · Σ_{k_m,k_n} c_m^{k_m} c_n^{k_n}
                 · (cos α)^(k_m+k_n) · Ki_(k_m+k_n+3)(τ_2D(α)) dα
```
The Ki order is **3+k_m+k_n** (one higher than `compute_P_esc_cylinder_3d_mode`'s Ki_(2+k)) because T carries an additional µ_3D = sin θ_p factor for partial-current weight.

Rank-1 identity (verified to 1e-14): `T_00^cyl ≡ P_ss^cyl` (same as `compute_P_ss_cylinder`).

Convergence (thin τ_R=2.5 cyl, fuel-A-like):
```
N=1: bare -2.95% / MB -0.34% ✓
N=2: bare -2.63% / MB -0.32% ✓
N=3: bare -2.33% / MB -0.23% ✓
N=4: bare -2.01% / MB +0.03% ⚠ overshoot
N=8: bare -1.05% / MB +1.27% ⚠
```

Same overshoot pattern as sphere. Different mechanism: cyl resolvent norm IS bounded (1.07 continuous), but **R = (1/2) M⁻¹ is poorly conditioned at high N** and the geometric series amplifies this to user-visible drift. UserWarning at N≥4 mirrors sphere.

### 2.3 Slab multi-bounce (investigation #2)

**Slab T is per-face block off-diagonal:**
```
T_slab = [[0,    T_oi],
          [T_io, 0   ]],   T_io = T_oi by face symmetry

T_oi^(mn) = 2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^{-τ_total/µ} dµ
          = 2 Σ_{k_m,k_n} c_m^{k_m} c_n^{k_n} · E_{k_m+k_n+3}(τ_total)
```
Self-blocks `T_oo = T_ii = 0` exactly: a single transit at constant µ direction cannot leave and return to the same face without a reflection at the other face.

Rank-1 identity (verified to 1e-14): `T_oi^(0,0) ≡ 2 E_3(σL)` (the slab P_ss analog).

**Geometric immunity to the sphere pathology**:
- Slab chord = L/µ → ∞ at grazing (NOT zero like sphere).
- Transmission e^(-τ/µ) → 0 EXPONENTIALLY at grazing.
- T_op^slab(µ) = e^(-σL/µ) → 0 at µ=0 (not 1, as in sphere).
- Off-diagonal block structure: ρ(T·R) ≤ 0.08 across all N at thin τ_L=2.5.

Convergence (thin τ_L=2.5 slab):
```
N=1: bare -2.83% / MB -0.30% ✓
N=4: bare -2.75% / MB -0.24% ✓
N=8: bare -2.67% / MB -0.16% ✓
N=16: bare -2.67% / MB -0.16% ✓ (plateau — quadrature noise floor)
```
**Monotone all the way to N=16+. NO warning needed.**

Robustness sweep across τ_L ∈ [0.5, 10]: all PASS without warning.

## 3. Phase 4 deliverables

### Phase 4a — Slab multi-bounce ship (Session 1, ~3 hours)

1. **`compute_T_specular_slab(radii, sig_t, n_modes)`** in `peierls_geometry.py`:
   - Returns the (2N × 2N) block off-diagonal T matrix.
   - Closed-form via E_n: `T_oi^(mn) = 2 Σ_k c_m^k_m · c_n^k_n · E_(k_m+k_n+3)(τ_total)` (homogeneous; multi-region τ along the chord).
   - Self-blocks T_oo = T_ii = 0 exactly.
   - Per the `_shifted_legendre_monomial_coefs(n)` helper.
2. **Wire `closure="specular_multibounce"`** for slab in `_build_full_K_per_group`:
   - Build P_slab (2N × N_x), G_slab (N_x × 2N) from per-face primitives (same as `closure="specular"` slab branch).
   - R_slab = block_diag(R_face, R_face) with R_face = (1/2) M⁻¹.
   - K_bc = G_slab @ R_slab @ np.linalg.solve(I − T_slab @ R_slab, P_slab).
   - **NO UserWarning at any N** (per the geometric immunity finding).
   - Update the docstring + dispatch help message.
3. **Tests** (`test_peierls_specular_bc.py`):
   - `test_specular_multibounce_slab_rank1_equals_hebert_slab_analog`: at N=1, scalar `T_oi[0,0] = 2 E_3(σL)`. Verify the resulting `K_bc^slab_mb|_{N=1}` matches the rank-1 Hébert-equivalent for slab (whatever that turns out to be — derive the closed form).
   - `test_specular_multibounce_slab_lifts_plateau_thin`: thin slab τ_L=2.5, rank-1 within 0.5 % of k_inf, rank-3 within 0.3 %.
   - `test_specular_multibounce_slab_robustness_high_N`: N=4, 8, 16 all stay within 0.5 % (no overshoot — pins the geometric-immunity claim as a regression).
   - Promote `diag_specular_mb_phase4_07_synthesis.py` (slab portion) to a regression test.

### Phase 4b — Cyl multi-bounce ship + sphere warning tighten (Session 1-2, ~4 hours)

1. **`compute_T_specular_cylinder_3d(radii, sig_t, n_modes, n_quad=64)`** in `peierls_geometry.py`:
   - Returns the (N × N) Knyazev T matrix.
   - Closed form: `T_mn = (4/π) ∫_0^(π/2) cos α · Σ_{k_m,k_n} c_m^{k_m} c_n^{k_n} · (cos α)^(k_m+k_n) · Ki_(k_m+k_n+3)(τ_2D(α)) dα`.
   - Multi-region τ_2D via standard cylinder-shell intersection (same pattern as `compute_P_ss_cylinder`).
2. **Wire `closure="specular_multibounce"`** for cyl in `_build_full_K_per_group`:
   - Build P, G from `compute_P_esc_cylinder_3d_mode` and `compute_G_bc_cylinder_3d_mode` (same as cyl branch of `closure="specular"`).
   - R = (1/2) M⁻¹ via `reflection_specular(N)`.
   - K_bc = G @ R @ np.linalg.solve(I − T @ R, P).
   - **UserWarning at N≥4** mirroring sphere.
3. **Sphere warning tighten**: change `if n_bc_modes >= 5:` to `if n_bc_modes >= 4:` in sphere `closure="specular_multibounce"` branch (per investigator #1 recommendation: overshoot already starts at N=4).
4. **Sphere docstring update**: replace "high-N pathology" framing with "fundamental matrix-Galerkin divergence at grazing-µ" — point at the operator-norm growth and the continuous-limit singularity. Reference `specular_mb_overshoot_root_cause.md`.
5. **Tests**:
   - `test_specular_multibounce_cyl_rank1_equals_pss`: at N=1, `T_00^cyl = P_ss^cyl` (1e-14 algebraic identity).
   - `test_specular_multibounce_cyl_lifts_plateau_thin`: thin cyl τ_R=2.5 rank-1 within 0.5 %, rank-3 within 0.3 %.
   - `test_specular_multibounce_cyl_warns_at_high_N`: assert UserWarning emitted at N=4.
   - Update existing sphere test `test_specular_multibounce_warns_at_high_N` to N=4 (was N=5).

### Phase 4c — Documentation + agent memory (Session 2, ~2 hours)

1. **Sphinx update** in `docs/theory/peierls_unified.rst`:
   - Extend the multi-bounce subsection with the cyl Knyazev T derivation and the slab per-face block T derivation.
   - Per-geometry table: which geometries have geometric immunity vs which inherit the matrix divergence.
   - Updated convergence ladders: cyl thin (overshoot at N≥4) and slab thin (monotone to N=16).
   - Update the closure compatibility table (`Class B — closures shipped per shape`) with `specular_multibounce` entries for cyl and slab.
2. **Agent memory updates**:
   - New `specular_bc_multibounce_cyl_slab_shipped.md`.
   - Update `MEMORY.md` index.
   - Mark `specular_mb_phase4_cyl_slab.md` as "shipped" in its frontmatter.

### Phase 4d — Issue updates (Session 2, ~30 min)

1. Update GitHub Issue #100, #103, #132 with the multi-bounce status:
   - #132 partial resolution: specular_multibounce (sphere/slab) gives Hébert-quality at thin cells. Cyl matches sphere envelope.
   - Reference the operator-norm divergence as a structural finding worth a follow-up.

## 4. Phase 5 (next-next-session) — continuous-µ reformulation

This is the proper fix for the sphere/cyl matrix-Galerkin divergence. **Out of scope for Phase 4**.

### 4.1 Conceptual sketch

Replace the matrix inverse `(I − T·R)⁻¹` with a **direct multiplication** by the continuous-µ multi-bounce factor:

```
For sphere: f_mb(µ) = 1/(1 − e^(−σ·2Rµ))     (multiplied by µ-weight)
For cyl:    f_mb depends on (α, θ_p) — needs the Knyazev unification
For slab:   f_mb(µ) = 1/(1 − e^(−σ·L/µ))     (regular at both µ=0 and µ=1)
```

The K_bc is then a single µ-integral that combines G's surface response, R's specular reflection physics, and the multi-bounce factor — all together — without ever building a separate T matrix and inverting a Galerkin projection.

### 4.2 Why this is hard

- **Sphere/cyl µ-resolved kernel is 3-D**: need to handle (µ_2D, θ_p) jointly without losing the multi-bounce summation.
- **Multi-region τ(µ)** along the chord: the chord's path through annular regions changes with µ; the `e^(-τ(µ))` is piecewise-smooth in µ.
- **Maintaining the rank-N angular structure at the surface**: still want shifted-Legendre P̃_n basis for the SURFACE flux, but the µ-integration becomes adaptive (bounded but possibly singular integrand near µ=0 for sphere, regular for slab).
- **No closed-form for general τ(µ)**: have to use adaptive quadrature (Gauss-Kronrod or scipy `quad`) with explicit subdivision at the impact-parameter discontinuities.

### 4.3 Phase 5 starting tasks

1. **Literature pull** (literature-researcher agent): find canonical references on continuous-µ multi-bounce kernels for sphere/cyl/slab transport. Stamm'ler, Hébert, Sanchez chapter on integral transport. Likely there's a textbook treatment of this.
2. **SymPy derivation script** at `derivations/peierls_specular_continuous_mu.py`: lay out the µ-resolved kernel for sphere (the simplest test case).
3. **Prototype implementation** as `compute_K_bc_specular_continuous_mu_sphere` — a SLOW reference solver (adaptive quadrature, no matrix Galerkin). Test at thin sphere; verify it converges to k_inf as adaptive tolerance tightens.
4. **Decide ship vs research**: if the reference solver is too slow for production, keep the matrix form (with cap N≤3) as the production closure and ship the continuous-µ form as an "exact reference" for verification only.

## 5. Test plan (Phase 4)

### 5.1 New tests

- 6 cyl/slab MB tests (rank-1 identity, thin-cell lift, warning emission, robustness)
- Update 1 sphere MB test (warning threshold change)

### 5.2 Regression coverage

After Phase 4:
- sphere MB: 3 existing tests + 1 updated (warning threshold N=4)
- cyl MB: 3 new tests
- slab MB: 3 new tests
- Total specular MB tests: ~10

Plus the existing 18 specular tests (Phase 1 + 2 + 3 sphere MB).

### 5.3 Quadrature posture

Same as Phase 1-3:
- BASE for fast iteration: p_order=4, n_panels=2, n_quad=24, dps=20.
- RICH for stability check: p_order=6, n_panels=4, n_quad=48, dps=30.
- ULTRA off-suite (only for spot-checks).

## 6. Decision tree

| Outcome | Trigger | Action |
|---------|---------|--------|
| **A — full Phase 4 success** | Slab MB ships warning-free; cyl MB ships with N≥4 warning; sphere warning tightens; tests all green | Ship; close out as Phase 4 complete; open Phase 5 (continuous-µ research issue). |
| **B — slab MB fails geometric-immunity claim at higher N** | Investigator #2's claim of "monotone to N=16" doesn't survive a wider regime (e.g., very-thin τ_L=0.1) | Investigate — slab might still need a warning at extreme thin. Adjust gates and ship. |
| **C — cyl MB rank-1 identity fails** | `T_00^cyl ≠ P_ss^cyl` empirically | Re-derive T_00^cyl from first principles. May indicate the Knyazev expansion has a normalization bug we missed. Block ship until resolved. |
| **D — Both cyl and slab need rework** | Both fail their respective ship gates | Drop Phase 4a/4b; do only Phase 4c (sphere warning + docs). Punt cyl/slab MB to Phase 5 (continuous-µ-only). |

## 7. Acceptance criteria

Phase 4 is complete when:

- (a) `compute_T_specular_slab` shipped with closed-form per-face block T.
- (b) `compute_T_specular_cylinder_3d` shipped with Knyazev Ki_(3+k_m+k_n) expansion.
- (c) `closure="specular_multibounce"` accepted by `solve_peierls_*g` for SLAB and CYLINDER (sphere already shipped Phase 3).
- (d) Slab MB has NO warning at any N; cyl MB has UserWarning at N≥4 mirroring sphere.
- (e) Sphere MB warning tightened from N≥5 to N≥4 + docstring updated.
- (f) Per-geometry rank-1 identities verified to 1e-14:
  - Sphere: `T_00 = P_ss`
  - Cyl: `T_00^cyl = P_ss^cyl`
  - Slab: `T_oi[0,0] = 2 E_3(σL)`
- (g) Per-geometry thin-cell convergence ladders pinned in regression tests.
- (h) Sphinx section extended; closure tables updated; agent memos shipped.
- (i) GitHub Issues #100, #103, #132 updated with multi-bounce status.

## 8. Risks and prerequisites

### 8.1 Prerequisites

- **`feature/peierls-specular-bc` merged to main** (or rebased onto). Phase 4 builds on the Phase 1-3 specular infrastructure.
- **Investigator #2's diagnostic scripts are committed** (they're untracked on the current branch). Promote useful ones to tests; archive the rest in `derivations/diagnostics/`.

### 8.2 Risks

- **R1 — Slab MB breaks at very-thin (τ_L < 0.5)**: the geometric immunity argument relies on the chord blowing up at grazing. For extremely thin cells the chord may not be enough to suppress. **Mitigation**: include a τ_L=0.1 case in the robustness test.
- **R2 — Cyl MB warning trips on legitimate use**: if users routinely run cyl at N=4 because they don't know about the warning, they'll hit confusion. **Mitigation**: make the warning message extremely clear ("for cylinder, use N∈{1,2,3} or `boundary='specular'`"); link to docs.
- **R3 — Continuous-µ reformulation (Phase 5) doesn't yield a clean form for cyl**: the 3-D Knyazev structure may not admit a clean µ-resolved direct integration. **Mitigation**: literature pull first; if no clean form exists, the matrix form with N≤3 cap is the production answer indefinitely.
- **R4 — Sphere docstring framing change confuses existing users**: changing from "high-N pathology" to "fundamental matrix-Galerkin divergence" might alarm users who are happy with N≤3 production use. **Mitigation**: keep the practical guidance front-and-center ("use N∈{1,2,3} for production; N≥4 is research/verification only").

### 8.3 Non-goals (Phase 4)

- **Not** implementing the continuous-µ reformulation — that's Phase 5.
- **Not** addressing the underlying R = (1/2) M⁻¹ conditioning at high N for cyl — same root cause as sphere; same mitigation (low N).
- **Not** changing the bare `boundary="specular"` API — only the multi-bounce variant.
- **Not** revising the Phase 1-3 work — it ships as is.

## 9. Estimated budget

- **Best case**: 2 sessions. S1 = slab MB ship + tests + cyl T derivation. S2 = cyl MB ship + warning tighten + docs + memos.
- **Mid case**: 3 sessions if R1 (slab very-thin) or R3 (cyl Knyazev T has subtle bug) fires.
- **Worst case**: 4 sessions if both cyl and slab need re-derivation (Decision D).

LoC delta:
- `compute_T_specular_slab` + wiring: ~80 LoC.
- `compute_T_specular_cylinder_3d` + wiring: ~120 LoC.
- Sphere warning + docstring update: ~10 LoC.
- Tests: ~250 LoC.
- Sphinx updates: ~150 lines.
- Agent memos: ~250 lines.

Total: ~860 LoC + ~400 lines of docs/memos.

Commit count: 3-4, scaled by outcome.

## 10. Phase 5 (multi-session research project)

After Phase 4 lands, the next big direction is the **continuous-µ multi-bounce reformulation** that bypasses the matrix-Galerkin divergence entirely. Tasks:

1. **Literature pull**: dispatch `literature-researcher` for canonical multi-bounce kernel formulations (Stamm'ler, Hébert, Sanchez).
2. **SymPy derivation**: µ-resolved kernel for sphere with multi-region τ(µ).
3. **Reference implementation**: SLOW adaptive quadrature solver as the verification reference.
4. **Production decision**: is the continuous-µ form fast enough for production, or does it stay as a verification-only reference?
5. **Cross-verify** the continuous-µ reference against the rank-N production code at N∈{1,2,3} (where rank-N is reliable).

Phase 5 is open-ended; expect 4-6 sessions if pursued.

---

## Critical files for Phase 4

- `orpheus/derivations/peierls_geometry.py` — `compute_T_specular_slab`, `compute_T_specular_cylinder_3d` to add; sphere warning to update; cyl/slab `closure="specular_multibounce"` dispatch branches to wire
- `tests/derivations/test_peierls_specular_bc.py` — new tests
- `derivations/peierls_specular_multibounce.py` (NEW) — SymPy derivation of cyl + slab T matrices
- `derivations/diagnostics/diag_specular_mb_phase4_*.py` — investigator #2's 9 diagnostics, to archive + promote useful ones
- `docs/theory/peierls_unified.rst` — multi-bounce subsection extension
- `.claude/agent-memory/numerics-investigator/specular_mb_phase4_cyl_slab.md` — investigator #2's report (already exists)
- `.claude/agent-memory/numerics-investigator/specular_mb_overshoot_root_cause.md` — investigator #1's diagnosis (already exists)
