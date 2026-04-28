---
name: Specular multi-bounce cyl + slab shipped (Phase 4)
description: closure="specular_multibounce" rolled out from sphere-only to all three geometries. Slab MB ships with NO warning (geometric immunity); cyl MB ships with N>=4 UserWarning mirroring sphere; sphere warning tightened from N>=5 to N>=4.
type: project
---

# Specular MB Phase 4 — cyl + slab shipped (2026-04-28)

Branch `feature/peierls-specular-bc`, follow-up commit on top of
`9178cc6` (Phase 3 sphere MB). Per the
`.claude/plans/specular-bc-phase4-multibounce-rollout.md` plan.

## Why: investigator's two reports

Two memos drove the rollout:

- `specular_mb_overshoot_root_cause.md` — sphere MB at N≥4 is a
  fundamental matrix-Galerkin divergence (continuous-µ resolvent
  `1/(1-e^(-σ·2Rµ))` singular at grazing µ→0; matrix form has
  unbounded operator norm). Recommendation: tighten warning from
  N≥5 to N≥4.
- `specular_mb_phase4_cyl_slab.md` — derived T^cyl (Knyazev
  Ki_(3+k_m+k_n)) and T^slab (per-face block off-diagonal). Slab
  MB CONVERGES MONOTONICALLY (geometric immunity: chord = L/µ → ∞
  at grazing → transmission e^(-σL/µ) → 0 exponentially). Cyl MB
  has same N≥4 overshoot envelope as sphere but for a different
  reason (R = (1/2) M^{-1} ill-conditioning amplified by geometric
  series, not resolvent-norm divergence — continuous-limit cyl
  resolvent stays bounded at sup ≈ 1.07).

## What shipped

1. **`compute_T_specular_slab(radii, sig_t, n_modes, n_quad=64)`**
   in `orpheus/derivations/peierls_geometry.py`:
   - (2N × 2N) block off-diagonal: `T_slab = [[0, T_oi], [T_oi, 0]]`
     with self-blocks exactly zero.
   - `T_oi^(mn) = 2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^(-τ_total/µ) dµ`
     (homogeneous; multi-region τ_total = Σ σ_t,k · L_k).
   - Rank-1 identity: `T_oi^(0,0) = 2 E_3(τ_total)` exact (verified
     to 1e-14 in `test_specular_multibounce_slab_rank1_equals_2E3_identity`).
2. **`compute_T_specular_cylinder_3d(radii, sig_t, n_modes, n_quad=64)`**
   in `orpheus/derivations/peierls_geometry.py`:
   - Knyazev expansion: `T_mn = (4/π) ∫_0^(π/2) cos α · Σ_{k_m,k_n}
     c_m^k_m · c_n^k_n · (cos α)^(k_m+k_n) · Ki_(3+k_m+k_n)(τ_2D(α)) dα`.
   - Multi-region τ_2D via standard cylinder-shell intersection.
   - Pre-evaluates Ki using `ki_n_float` (not `ki_n_mp`) for speed —
     1000× faster, accuracy 1e-12+.
   - Rank-1 identity: `T_00^cyl = P_ss^cyl` exact (verified to 1e-14).
3. **`_build_full_K_per_group` dispatch** — single
   `closure="specular_multibounce"` branch dispatches by
   `geometry.kind`:
   - **slab-polar**: per-face P/G via `_slab_E_n` closed form,
     2N × 2N block-diagonal R, 2N × 2N T_slab; NO UserWarning at
     any N.
   - **sphere-1d**: same no-Jacobian P/G + Knyazev sphere T;
     UserWarning at N ≥ 4 (was N ≥ 5).
   - **cylinder-1d**: 3-D-corrected P/G via
     `compute_P_esc_cylinder_3d_mode` / `compute_G_bc_cylinder_3d_mode`
     (matching the bare specular cyl branch); N × N cyl T;
     UserWarning at N ≥ 4 mirroring sphere.
4. **Sphere warning tightened** from N ≥ 5 to N ≥ 4 per
   investigator #1 recommendation. Docstring rewritten to point at
   the operator-norm divergence + Phase 5 sketch for the
   continuous-µ reformulation.

## End-to-end k_eff smoke results (matches investigator's table)

Fuel-A-like 1G (σ_t=0.5, σ_s=0.38, νσ_f=0.025, k_inf=0.20833),
characteristic length 5 cm, τ = 2.5, BASE quadrature:

| N | sphere bare/MB | cyl bare/MB | slab bare/MB |
|---|----------------|-------------|--------------|
| 1 | -8.31 / **-0.27** ✓ | -2.95 / **-0.17** ✓ | -2.83 / **-0.30** ✓ |
| 3 | -6.24 / **-0.12** ✓ | -2.33 / **-0.14** ✓ | -2.77 / **-0.27** ✓ |
| 4 | -6.05 / +0.43 ⚠   | -2.01 / -0.04 ⚠ (warn) | -2.75 / **-0.24** ✓ |
| 8 | -3.22 / +5.62 ⚠⚠  | -1.05 / +1.27 ⚠ | -2.67 / **-0.16** ✓ |

Slab MB monotonically improves all the way to N=16+; sphere/cyl MB
overshoot at N≥4. UserWarning correctly emitted for sphere/cyl at
N≥4, suppressed for slab.

## Tests added (`tests/derivations/test_peierls_specular_bc.py`)

- `test_specular_multibounce_warns_at_high_N` — sphere warns at N=4,
  not at N=3 (replaces the prior N=5 boundary test, which was
  removed since the threshold tightened).
- `test_specular_multibounce_cyl_rank1_equals_hebert` — cyl rank-1
  bit-equals `boundary="white_hebert"` (Knyazev T_00^cyl = P_ss^cyl).
- `test_specular_multibounce_cyl_lifts_thin_plateau` — thin cyl
  τ_R=2.5 rank-1 within 0.5%, rank-3 within 0.3%, > 1% lift over
  bare specular.
- `test_specular_multibounce_cyl_warns_at_high_N` — cyl warns at N=4,
  not at N=3.
- `test_specular_multibounce_slab_rank1_equals_2E3_identity` —
  algebraic identity at 1e-14 across thin/thick/very-thin/MR cases;
  self-blocks exactly zero.
- `test_specular_multibounce_slab_rank1_lifts_plateau` — thin slab
  τ_L=2.5 rank-1 within 0.5%, > 1% lift over bare specular.
- `test_specular_multibounce_slab_monotonic_high_N` — slab MB
  monotonic at N ∈ {1, 4, 8}, no overshoot, NO warning at any N
  (geometric-immunity regression).

## Sphinx update (`docs/theory/peierls_unified.rst`)

- Class B closure table widened to 4 columns (cyl / sphere / slab)
  with `specular_multibounce` row populated for all three
  geometries. The table summarises the per-geometry T form,
  rank-1 identity, and warning policy.
- Multi-bounce subsection rewritten: per-geometry T derivation
  (sphere / cyl / slab), per-geometry pathology analysis (sphere
  fundamental divergence; cyl R-conditioning; slab geometric
  immunity), per-rank convergence ladder for all three geometries,
  best-use envelope, and a Phase 5 sketch for the continuous-µ
  reformulation.
- V&V test reference list extended with the 7 new MB tests.

## What did NOT ship (Phase 5 territory)

- **Continuous-µ reformulation**: bypasses the matrix-Galerkin
  pathology by integrating `µ / (1 - e^(-σ·2Rµ))` directly with
  adaptive quadrature. Out-of-scope for Phase 4; see investigator
  memo `specular_mb_overshoot_root_cause.md` for the conceptual
  sketch.
- **Underlying R = (1/2) M^{-1} conditioning fix for cyl at high
  N**: same root cause as sphere (basis conditioning); same
  mitigation (low-N envelope).

## Files touched

- `orpheus/derivations/peierls_geometry.py` (+~360 LoC: two new T
  funcs + dispatch refactor + sphere docstring rewrite)
- `tests/derivations/test_peierls_specular_bc.py` (+~270 LoC)
- `docs/theory/peierls_unified.rst` (+~210 lines)
- `.claude/agent-memory/numerics-investigator/MEMORY.md` index entry
