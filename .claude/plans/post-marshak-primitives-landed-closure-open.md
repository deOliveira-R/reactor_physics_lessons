# Post-session notes — Phase F.5 Marshak primitives landed, closure still open

**Session:** 2026-04-21 (continuation of Issue #119 Marshak-closure plan)
**Author:** Claude Opus 4.7 (1M context)
**Target for next session:** Next Claude Code session picking up Issue #119.

---

## Context

Followed `post-issue-119-rank-n-marshak-closure.md` plan's recommendation to commit to the Marshak partial-current-moment basis. The plan's core hypothesis was that measure mismatch (Lambert P/G vs Marshak W) was the sole cause of the rank-N hollow-sphere closure failure at N ≥ 2. **The hypothesis is incomplete** — a 16-recipe scan confirms no simple basis-choice fix closes the gap.

## What landed this session

### Code changes in `orpheus/derivations/peierls_geometry.py`

- Four new Marshak primitives (named with `_mode_marshak` suffix):
  - `compute_P_esc_outer_mode_marshak`
  - `compute_P_esc_inner_mode_marshak`
  - `compute_G_bc_outer_mode_marshak`
  - `compute_G_bc_inner_mode_marshak`

  All use `sin θ · µ · P̃_n(µ) · K_esc(τ)` integrand — partial-current measure consistent with the existing `compute_hollow_sph_transmission_rank_n` W matrix.

- `NotImplementedError` guard in `build_closure_operator` for `n_bc_modes > 1` + `reflection="white"` on a 2-surface cell: **kept**, with updated message referring to the recipe scan.

- `NotImplementedError` guard in `solve_peierls_1g(boundary="white_rank2")` for `n_bc_modes > 1`: **kept**.

### Tests added to `tests/derivations/test_peierls_rank2_bc.py`

- `test_sphere_marshak_G_equals_4x_P_sigt_zero` — Sanchez-McCormick sphere identity holds in Marshak basis.
- `test_sphere_marshak_mode0_smaller_than_lambert_mode0` — µ weight verified in integrand.

### Diagnostic script added

- `derivations/diagnostics/diag_rank_n_sph_marshak_primitives_sigt_zero.py` — σ_t → 0 verification against mpmath reference + two pytest checks (`test_marshak_P_outer_matches_reference_sigt_zero`, `test_marshak_G_is_4x_P_at_sphere_sigt_zero`).

## What did NOT work

Recipe scan at R=5, r_0/R=0.3, homogeneous (Σ_t=1, Σ_s=0.5, νΣ_f=0.75), using direct K_bc assembly with `composite_gl_r(inner_radius=r_0)`:

| Recipe | N=1 err | N=2 err | Comment |
|---|---|---|---|
| Scalar Phase F.4 (Lambert mode-0 + scalar W) | — | **0.077 %** | Current working path |
| L / M\*D + (I-W)⁻¹ | — | 1.374 % | Best mode-1 recipe |
| L / L + (I-W)⁻¹ (shipped pre-F.5) | — | 3.928 % | Mode-1 adds noise |
| M / M + (I-W)⁻¹ (all Marshak) | — | 10.863 % | Plan's recommended fix — NOT fix |
| L\*D / L\*D + (I-W)⁻¹ | — | 46.333 % | Gelbard both sides — very wrong |

Legend: L = Lambert primitives (no µ), M = Marshak primitives (with µ), \*D = post-multiply column/row by diag(1, 3) Gelbard factor per face; R = (I-W)⁻¹ unless noted. 16 combinations tested. All degrade accuracy vs the N=1 baseline.

**The mode-1 contribution is DEGRADING accuracy in every recipe tested.** This is the smoking gun for a deeper bug.

## Orthogonal bug discovered

`solve_peierls_1g` at `orpheus/derivations/peierls_geometry.py:3978` calls `composite_gl_r(radii, n_panels, p_order, dps=dps)` **without** `inner_radius`. For hollow-cell geometries this gives r_nodes spanning `[0, R]` instead of `[r_0, R]`, so the radial quadrature erroneously includes points inside the cavity.

This inflates the Phase F.4 N=1 residual from the quadrature-limited 0.077 % to 1.5 % for the R=5, r_0/R=0.3 case. It is likely also responsible for the "3.031 %" baseline cited throughout the Issue #119 / measure-mismatch investigation (via `boundary="white"`, which routes through the same broken `composite_gl_r` call plus the single-surface `build_white_bc_correction_rank_n`).

**The "3.031 % N=1 baseline" in `diag_rank_n_sph_keff_probe.py::test_N1_baseline_residual` is therefore spurious — it's a quadrature bug, not a rank-1 closure characteristic.** Once fixed, the rank-N per-face gate tightens from "beat 3 %" to "beat 0.08 %" — much harder.

**Fix recipe**: add `inner_radius=geometry.inner_radius` to the `composite_gl_r` call in `solve_peierls_1g`. One-line fix. Recommend committing before resuming rank-N investigation.

## Where the bug likely lives (next-session candidates)

After eliminating basis-choice fixes, the remaining candidates:

### 1. Mode-n primitive definition needs structural change beyond µ-weight

The current `compute_P_esc_outer_mode(n)` at n ≥ 1 uses `sin θ · P̃_n(µ_exit) · K_esc(τ)` (Lambert). The Marshak variant adds µ. But both may be WRONG in structure at n ≥ 1:

- Gelbard DP_{N-1} Jacobian `(ρ_max/R)²` baked into the *single-surface* `compute_P_esc_mode` works. Why doesn't a per-face version work?
- The per-face formulation has TWO surfaces contributing, each carrying its own Jacobian convention. Possibly the outer-face and inner-face primitives need DIFFERENT Jacobians (not the uniform "no Jacobian" convention).

### 2. Transmission matrix W_oi^{mn} at n ≥ 1

`compute_hollow_sph_transmission_rank_n` passes:
- Scalar N=1 bit-exact regression (`test_rank_n_sph_transmission_N1_matches_scalar_bit_exact`).
- Sanchez-McCormick reciprocity `A_k · W_{jk}^{mn} = A_j · W_{kj}^{nm}` (`test_rank_n_sph_transmission_reciprocity_transposed_modes`).

But the cross-face blocks (outer-inner, inner-outer) at modes n ≥ 1 have not been independently verified against a Monte-Carlo ray-tracer or an analytical σ_t → 0 form. A weekend-effort MC cross-check with 1M samples would isolate whether W is correct.

### 3. Start from scratch — canonical Sanchez-McCormick 1982 §III.F derivation

Drop the bottom-up tinkering and re-derive the full closure equations from the Sanchez & McCormick 1982 NSE 80:481-535 §III.F paper:

- Verify the definition of "moment" used there (Lambert vs Marshak, with or without Gelbard (2n+1)).
- Derive P, G, W, R formulas matching that convention.
- Implement fresh, compare primitives against my current code term-by-term.

This is ~2 hours of literature reading + ~2 hours of re-implementation. Likely the right path if candidate 1/2 don't yield in a few iterations.

## Recommended next-session sequence

1. **Fix inner_radius bug in `solve_peierls_1g`** (1-line fix + regression test). Commit separately — unblocks all hollow-cell solver work.
2. **Monte-Carlo cross-check on W_oi^{01}, W_io^{10}** at a specific (r_0/R, σ_t, m, n). If W is wrong, rebuild from Sanchez-McCormick directly.
3. **Derive per-face mode-n primitive from scratch** using Sanchez-McCormick §III.F. Compare term-by-term with current code. Identify where they diverge.
4. If steps 2-3 identify the bug, implement + land the rank-N per-face closure + update tests.

## Files changed this session

```
orpheus/derivations/peierls_geometry.py       (+ 4 Marshak primitives, ~270 lines)
tests/derivations/test_peierls_rank2_bc.py    (+ 2 foundation tests)
derivations/diagnostics/diag_rank_n_sph_marshak_primitives_sigt_zero.py  (new)
```

## Status at commit

- All Phase F tests green (re-confirmed).
- `n_bc_modes > 1` still raises `NotImplementedError` with updated message.
- Marshak primitives + σ_t → 0 verification form usable infrastructure for the next attempt.
- Issue #119 remains OPEN.

## Comments for Issue #119

A concise progress-comment summary (for user to post to the issue) is written at `/tmp/issue_119_comment.md` — lists the 16-recipe scan results, the orthogonal inner_radius bug, and the candidate next steps.
