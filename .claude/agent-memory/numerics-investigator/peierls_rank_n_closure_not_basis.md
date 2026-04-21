---
name: Rank-N closure error is NOT a missing constant factor
description: Empirical scaling scan proves the rank-N hollow-sphere per-face closure bug depends on optical thickness, ruling out simple factor-correction fixes.
type: project
---

# Rank-N hollow sphere: closure error is geometry-dependent (2026-04-21)

## Key negative result

The plan's "Jacobian + B^µ" closure recipe does NOT close to 0.1 %.
Exhaustive N=2 scan at R=5, r_0/R=0.3, sig_t=1 (k_inf=1.5):

- Best recipe: **jac=0, mu=0, Model-P=A, Model-G=B, (I-W)^-1** → **3.87 %** err.
- All Jacobian-on variants: 18-48 % err (much worse).
- All B^µ conversion variants: 4.7-5.8 % err (slightly worse).
- All multi-bounce Marshak formulations (G_L Bmu^-1 W (I - Bmu^-1 W)^-1 P_L etc.): 4.8-21 % err.

## Critical new findings

### N=1 bit-exact reduction requires Model-P=A + Model-G=B

- `compute_P_esc_outer` (F.4 scalar) IS Model A — excludes rays blocked by inner.
- `compute_G_bc_outer` (F.4 scalar) IS Model B — includes cavity-crossing rays.
- Current per-face mode primitives (`compute_P_esc_outer_mode`,
  `compute_G_bc_outer_mode`, `compute_G_bc_outer_mode_marshak`) use Model A
  uniformly. At N=1 this gives 0.135 % residual (worse than F.4's 0.077 %).
- With the correct P=A, G=B split at N=1, residual drops to **bit-exact
  match** with F.4 scalar (0.077 %).

This is necessary for N ≥ 2 closure but NOT sufficient.

### Mode-1 primitives contribute spurious amplitude

Zero-out isolation: setting P's mode-1 rows (or G's mode-1 cols) to zero
drops N=2 residual from 3.87 % to 0.17 %. Zeroing W's mode-1 blocks only
reduces residual to 3.9 %. **The spurious contribution is in the mode-1
P and G primitives, not in W.**

### Scaling scan reveals geometry-dependence

Optimal uniform scale `c` applied to mode-1 P and G (c_P=c_G=c) that
minimises N=2 residual:

| Geometry | sig_t | c_opt | err_opt |
|---|---|---|---|
| R=5, r_0=1.5 | 0.5 | 0.05 | 0.59 % |
| R=5, r_0=1.5 | 1.0 | 0.16 | 0.001 % |
| R=5, r_0=1.5 | 2.0 | 0.36 | 0.003 % |
| R=5, r_0=1.5 | 4.0 | 0.60 | 0.086 % |
| R=10, r_0=3.0 | 1.0 | 0.36 | 0.003 % |

**c_opt scales with optical thickness** (sig_t × R). This RULES OUT any
fixed-constant fix (1/3, 1/(2n+1), sqrt(2n+1), µ-weight factor).

The correct fix must be a basis/normalisation restructuring that
naturally produces the right amplitude across problem parameters. Best
remaining candidates:

1. **Restructure Mode-n primitive integrand** — the correct form may
   involve an r-dependent Jacobian that's NOT `(ρ_max/R)²` (which made
   things much worse in this session's scan).
2. **Re-derive from Sanchez-McCormick §III.F directly** — the existing
   primitives are improvised from the single-surface `compute_P_esc_mode`
   form, which has its own (ρ_max/R)² Jacobian. For the per-face
   decomposition the correct normalisation may require splitting the
   surface-Jacobian contribution between the two faces in a problem-
   dependent way.
3. **Monte-Carlo cross-check on P_1, G_1** independently, separate from W.

## Orthogonal bug fixed

`solve_peierls_1g` did NOT pass `inner_radius` to `composite_gl_r`,
leaving quadrature nodes inside the cavity for hollow cells. Fixed at
`peierls_geometry.py:3996`. Regression test landed at
`test_peierls_rank2_bc.py::test_solve_peierls_1g_hollow_sph_white_rank2_inner_radius_plumbing`.
Without the fix, hollow sphere rank-2 residual was 1.5 % (quadrature
error). With the fix it's back to 0.08 %.

## Diagnostics committed

- `derivations/diagnostics/diag_rank_n_15_N1_reduction_model_split.py`
  — Three tests: F.4 regression gate (<0.1 %), Model-A-both documented
  (~0.13 %), Model-split P=A/G=B bit-exact match to F.4.
- `derivations/diagnostics/diag_rank_n_16_mode1_scale_sensitivity.py`
  — Three tests: current N=2 residual 3.87 %, zeroing P mode-1 restores
  ~0.17 %, optimal c is geometry-dependent.

## Prior memory updates

- ``peierls_rank_n_W_mixed_basis.md`` claim "W is mixed-basis" is
  **partially superseded**. MC does verify W as an integral, but the
  closure failure is NOT fixed by any reflection reformulation
  `(I - B^µ^-1 W)^-1` or multi-bounce variant. The mode-1 **primitives**
  themselves need structural rework.
- ``peierls_rank_n_measure_bug.md`` "measure mismatch is the sole cause"
  hypothesis is **refuted** by the full 16+ recipe scan + scaling scan.
