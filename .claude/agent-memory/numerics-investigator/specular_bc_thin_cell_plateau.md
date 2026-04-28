---
name: Specular BC thin-cell structural plateau (rank-N misses multi-bounce)
description: Rank-N specular K_bc = G·R·P is single-bounce — needs Hébert-style (1-P_ss)⁻¹ generalisation. Naive matrix multi-bounce (I-T·R)⁻¹ works at N≤3 but fails at N≥4 due to grazing-mode spectral pathology of R_spec.
type: project
---

# Specular BC thin-cell plateau (Issue tracked, 2026-04-27)

Branch: `feature/peierls-specular-bc`. Status: ROOT CAUSE IDENTIFIED, FIX
NOT SHIPPED. Plateau is structural — the implementation is mathematically
self-consistent for the single-bounce closure but a **multi-bounce
correction is missing**.

## TL;DR

For homogeneous thin sphere (σ_t·R = 2.5, σ_s/σ_t=0.76, k_inf=0.208):

- `boundary="specular"` plateaus at **-6.25 % at RICH quadrature**
  for N=4,6,8,10 (NOT a quadrature artifact).
- `boundary="white_hebert"` recovers k_inf to **-0.27 %** at rank-1.
- `boundary="white_rank1_mark"` gives -8.42 % (specular rank-1 ≡ Mark
  rank-1 by construction).

The 30× improvement Hébert→Mark comes from the `(1 - P_ss)⁻¹` factor
on K_bc^Mark, which has NO clean rank-N generalisation in the
G·R_spec·P formulation.

## What's broken structurally

`K_bc = G·R·P` is a SINGLE-BOUNCE operator (source → boundary →
flux). Hébert wraps the rank-1 K_bc with a scalar multi-bounce factor
1/(1-P_ss). For the thin sphere this factor is only 1.083 (P_ss=0.077)
yet that 8 % boost in K_bc moves k_eff from -8.4 % to -0.27 %.

The matrix generalisation `K_bc^corr = G · R · (I - T·R)⁻¹ · P`
where `T_mn = 2 ∫_0^1 µ P̃_m P̃_n e^{-σ_t · 2R · µ} dµ` (sphere
chord transit) was implemented and tested:

- At rank-1: T[0,0] = P_ss to 1e-16, recovers Hébert exactly. ✓
- At low N: corrected -0.27 % (N=1) → -0.12 % (N=3). Excellent.
- At high N: OVERSHOOTS — corrected N=6 = +2.31 %, N=8 = +5.62 %.

High-N failure is spectral: ρ(T·R) approaches 1 as N grows because
R_spec entries grow O(N²) (largest at N=10 = 81) and T captures
grazing modes (chord 2Rµ → 0 as µ → 0) that survive reflection
forever.

At thin τ_R=2.5 N=10: ρ(T·R) = 0.87. At very-thin τ_R=1.0 N=10:
ρ(T·R) = 0.94 — geometric series numerically unstable.

## What was REFUTED

| Hypothesis                                       | Diag    | Verdict                                |
| ------------------------------------------------ | ------- | -------------------------------------- |
| K·1 row-sum violation                            | 01,02   | Hebert ALSO has σt·K·1≈0.5; not the contract |
| Restoring (ρ_max/R)² Jacobian fixes it           | 14      | Makes -8.4 % → -48 %. Wrong.           |
| µ-weighted (Marshak) P primitive                 | 14      | -8.4 % → -26 %. Wrong.                 |
| Mode truncation of ψ⁺(µ) is bottleneck           | 05      | L2 trunc err 1.6 % at N=4 < 5.6 % k_eff err |
| Per-row K_bc renormalisation forces conservation | 13      | Makes -8.4 % → -75 %. Wrong direction. |
| Scalar (1-P_ss)⁻¹ on rank-N K_bc                 | 04      | N=1 perfect (-0.27%), N≥2 overshoot     |

## Correct contract (NOT "K·1 = 1")

For homogeneous infinite-medium-equivalent cell, the eigenvector
under specular BC is NOT uniform when the cell is thin (diag 10:
specular eigvec ranges [0.91, 1.00] at N=6 thin sphere; Hebert eigvec
ranges [0.998, 1.00]). σt·K·1 ≈ 0.5 for BOTH closures because the
flux is non-uniform — the row sum is a misleading metric.

The right contract is `k_eff → k_inf as N → ∞` AT SUFFICIENT
QUADRATURE. The plateau at -6.25 % from N=4 onward at RICH is the
true structural error, not numerical noise.

## Recommended path forward

1. **Document the plateau** as a known limitation. Update
   `test_specular_2G_homogeneous_converges_to_kinf_2G` to either:
   (a) add an `xfail("thin-cell plateau, see GH#XXX")` or
   (b) set `err_gate = 0.07` (loose) and document why.

2. **Open a research GitHub issue** for "rank-N specular multi-bounce
   correction" with the matrix-correction draft (diag 06) and the
   spectral-pathology analysis as starting evidence.

3. **Use `boundary="white_hebert"`** for the 2G/1R thin-cell test
   instead of `boundary="specular"`. white_hebert is the production-
   quality closure for thin homogeneous Class B cells; specular is
   for HETEROGENEOUS verification at thick cells.

4. **The Phase 2 method-of-images cross-verification** (per
   `specular_bc_phase1_implementation.md`) would settle the matter.
   Image-series specular k_eff should give the EXACT specular result
   independent of any rank-N truncation. If it ALSO gives -6.25 %
   on thin cell, the user's expectation is wrong (specular ≠ k_inf
   for thin cell). If it gives k_inf, our rank-N truncation is the
   bug.

## Diagnostic files (derivations/diagnostics/)

- `diag_specular_thin_01_row_sum.py` — K·1 profile thin vs thick.
- `diag_specular_thin_02_compare_closures.py` — Mark/Hébert/specular
  K·1 comparison; debunks K·1=1 contract.
- `diag_specular_thin_03_keff_compare.py` — k_eff per closure
  (Hébert -0.27 %, specular -8.4 %).
- `diag_specular_thin_04_apply_hebert_to_spec.py` — scalar Hébert ×
  specular: N=1 perfect, N≥2 overshoot.
- `diag_specular_thin_05_psi_modes.py` — ψ⁺(µ) projection: thin
  decays FASTER than thick. Truncation is not bottleneck.
- `diag_specular_thin_06_multi_bounce.py` — **CORE DIAG**. Build
  matrix T, apply `(I - TR)⁻¹` correction. Works at low N, fails
  at high N due to grazing modes.
- `diag_specular_thin_07_implied_factor.py` — brentq for α: fails
  because k_eff(α) is non-monotonic.
- `diag_specular_thin_08_alpha_scan.py` — α scan: shows sharp peak
  of k_eff(α) near α=2.
- `diag_specular_thin_09_rich_keff.py` — confirms structural plateau
  at RICH precision.
- `diag_specular_thin_10_eigvec.py` — eigvec comparison: Hébert
  uniform [0.998, 1.00], specular non-uniform [0.91, 1.00].
- `diag_specular_thin_11_synthesis.py` — **PROMOTE-CANDIDATE**:
  pins -6.25 % plateau at RICH for regression detection.
- `diag_specular_thin_12_balance.py` — implied "leakage" calc
  (misleading: 26-44 % implied — but really just non-uniform eigvec).
- `diag_specular_thin_13_conservation_fix.py` — per-row K_bc
  renorm: fails (k_eff -75 %).
- `diag_specular_thin_14_jacobian_restored.py` — Jacobian /
  µ-weight variants: ALL worse than current "no_jacobian".

## Promotion recommendation

- `diag_specular_thin_11_synthesis.py` → `tests/derivations/test_peierls_specular_bc.py`
  as `test_specular_thin_sphere_plateau_pinned_for_regression` once
  the user confirms the documentation strategy (mark as known
  limitation vs xfail vs research issue).
- `diag_specular_thin_06_multi_bounce.py` is the **starting point
  for the eventual fix**: `(I - TR)⁻¹` works at low N. Find a way
  to suppress grazing modes (e.g., resolve T in a different basis,
  or cap N in the multi-bounce closure separately from the K_bc
  rank).
- All others are exploratory; keep until a fix lands, then prune.
