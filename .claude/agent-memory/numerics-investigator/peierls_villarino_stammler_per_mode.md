---
name: Per-mode Villarino-Stamm'ler does NOT close the rank-N hollow-sphere plateau
description: Hypothesis C test at sigma_t*R=5, r_0/R=0.3 — enforcing F.4 row-sum conservation per diagonal mode block preserves reciprocity at 1e-16 but MAKES k_eff WORSE (1.42% -> 1.87% on Sanchez µ-ortho, essentially no change on shipped Marshak 10.86% -> 10.83%). Verdict: Branch C (close Issue #119 with F.4 as production).
type: project
---

# Novel per-mode Villarino-Stamm'ler normalisation on rank-N W: VERDICT NEGATIVE

## Investigation date
2026-04-21, on `investigate/peierls-solver-bugs` at commit `ed69a09`.

## Setup
Hollow sphere, `R=5, r_0=1.5`, homogeneous `Σ_t=1, Σ_s=0.5, νΣ_f=0.75`,
`k_inf=1.5`. Radial mesh: 8 composite Gauss-Legendre nodes (4 panels × 2 pts).

## Recipe (Hypothesis C, simplest-first)

Extract each diagonal mode-n surface 2×2 sub-block of the rank-N W:

```
W_sub[n] = [[W[n, n],     W[n, N+n]],
            [W[N+n, n],   W[N+n, N+n]]]
```

Build Hébert Eq. 3.347 symmetric t-matrix:

```
t = [[(S_out/4) W_sub[0,0], (S_out/4) W_sub[0,1]],
     [(S_in /4) W_sub[1,0], (S_in /4) W_sub[1,1]]]
```

Solve the 2-unknown linear system for `z_outer^n, z_inner^n` from
Hébert Eq. 3.351 with `g_target = (S/4) · (W_oo_F4 + W_oi_F4)` on
outer and `(S_in/4) · (W_io_F4 + W_ii_F4)` on inner (F.4 scalar row
sums as the per-mode target). Apply symmetric correction
`ŵ_ij = (z_i + z_j) w_ij`.

Off-diagonal mode couplings (`n != m`) are left untouched — the
correction is strictly per diagonal mode sub-block.

## Residual table (sigma_t·R = 5, r_0/R = 0.3)

**Pipeline A — Sanchez µ-weighted-orthonormal primitives + µ-ortho W**
(matches the 1.42% plateau of `diag_sanchez_N_convergence.py`):

| N | err_raw | err_vs | recip_raw | recip_vs |
|---|---------|--------|-----------|----------|
| 1 | 2.55 %  | 2.17 % | 1e-16     | 1e-16    |
| 2 | **1.42 %** | **1.87 %** WORSE | 2e-16 | 2e-16 |
| 3 | 1.42 %  | 1.87 % WORSE | 2e-16 | 2e-16 |
| 4 | 1.43 %  | 1.88 % WORSE | 2e-16 | 2e-16 |

**Pipeline B — shipped Marshak primitives + shipped MC-verified W**
(this is the path `_build_closure_operator_rank_n_white` takes behind
the `NotImplementedError` guard):

| N | err_raw | err_vs | recip_raw | recip_vs |
|---|---------|--------|-----------|----------|
| 1 | 13.53 % | 13.53 % | 0 | 0 |
| 2 | 10.86 % | 10.83 % | 1e-16 | 1e-16 |
| 3 | 10.70 % | 10.66 % | 1e-16 | 1e-16 |
| 4 | 10.70 % | 10.66 % | 1e-16 | 1e-16 |

**N=1 is untouched in Pipeline B** because at rank-0 the shipped W
*already* has `W_oo + W_oi = 1` at σ_t=0 (F.4 identity baked in), so
the V-S z's come out to exactly 1/2 and the correction is identity.

## Conservation actually enforced

At σ_t=0 the scheme hits its design target for every mode:

```
F.4 scalar row sums: outer = 1.910, inner = 0.090

Raw W row sums (N=3):
 n=0: outer 2.137 -> 1.910 ✓, inner 0.118 -> 0.090 ✓
 n=1: outer 1.191 -> 1.910 ✓, inner 0.062 -> 0.090 ✓
 n=2: outer 0.547 -> 1.910 ✓, inner 0.012 -> 0.090 ✓
```

The additive symmetric V-S correction DOES force per-mode row
conservation. The scheme does what it claims. It just doesn't help.

## Reciprocity

Preserved at machine precision (1e-16) by construction: the additive
`(z_l + z_m)` factor is symmetric in l ↔ m, so `ŵ_ij = (z_i + z_j)
W_ij` inherits the symmetry of `A_i W_ij`. Verified empirically for
all N, both pipelines, pre- and post-V-S.

## Why it fails (interpretation)

The 1.42 % plateau is a **mode-coupling** failure, not a conservation
failure. The 2×2-per-mode V-S fix is diagonal in mode index — it
forces `W_oo[n,n] + W_oi[n,n] = target[n]` but leaves all
`n≠m` cross-mode entries untouched. The structural obstruction
(P̃_n(c_in) ≠ P̃_n(µ_emit)) bleeds across modes. Correcting the
diagonal blocks **distorts** the balance between diagonal and
off-diagonal coupling — so instead of improving k_eff, it shifts the
closure further off.

This is what Hébert warns about implicitly: V-S is defined on the
rank-0 scalar CP primitives (Eqs. 3.347-3.352); extending it to
rank-N is unvalidated. The per-mode extension is not in any
reference (Ligou, Sanchez 2002, Stamm'ler Ch. IV, Stacey Ch. 9,
Hébert 2009 Ch. 3 — all five scalar/DP-0 in curvilinear).

## Recommendation

**Branch C — close Issue #119 with F.4 scalar as production**.

The five-reference synthesis already recommended this. Hypothesis C
falsifies the "V-S rescue" speculation explicitly, at zero code cost
(30 lines of V-S iteration + patch).

Hypotheses A (weakly conservative `g^ρ = δ_{ρ,0} g^(0)`) and B
(vacuum-source-derived `g^ρ`) are unlikely to do better: B is still
diagonal-in-mode like C, and A forces n≥1 to zero which is
incompatible with F.4's mode-0 correctness. The structural
obstruction is cross-mode, not per-mode.

**Do NOT lift the NotImplementedError guard.** Keep the Marshak
primitive + rank-N W infrastructure for future research (e.g.,
geometry-adapted `{P̃_n(c_in(µ))}` inner basis, the user's
"Direction C" in the next-session plan).

## Code artefacts

- `derivations/diagnostics/diag_rank_n_villarino_stammler_per_mode.py`
  (~500 lines, 2 pytest tests, both pass, ~3s runtime).
  Documents:
  - `apply_vs_per_mode(W, N, S_out, S_in, g_out, g_in)` — reusable
    primitive for future per-mode V-S experiments.
  - `build_K_bc_vs` / `build_K_bc_shipped_marshak_vs` — both
    pipelines patched with V-S.
  - Residual table generator + per-mode conservation verifier.
  - Reciprocity check `S_out · W_io[m,n] = S_in · W_oi[n,m]`.

Do NOT promote to `tests/cp/` as a success gate; promote as a
**plateau-persistence regression gate** if the eventual close-out
plan wants an explicit "Hypothesis C falsified" record.

## Cross-references

- `.claude/agent-memory/literature-researcher/hebert_2009_ch3_interface_currents.md`
  §12 (V-S equations 3.347-3.352 verbatim + 8-question analysis
  including "probably won't help because V-S is rank-0 only").
- `.claude/agent-memory/numerics-investigator/peierls_rank_n_sanchez_closure_failed.md`
  (60-recipe scan establishing the 1.42 % plateau).
- `derivations/diagnostics/diag_rank_n_sanchez_conservation_probe.py`
  (structural diagnosis: `W_oo[n,n] + W_io[n,n] = 0.28, 0.13, 0.09`
  for n=1,2,3 at σ_t=0 — what we targeted with V-S).
- `derivations/diagnostics/diag_sanchez_N_convergence.py`
  (pipeline A baseline 1.42 % plateau at N=1..4).
