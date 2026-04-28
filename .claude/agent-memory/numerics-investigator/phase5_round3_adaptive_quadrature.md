---
name: Phase 5+ Round 3 — Adaptive quadrature for r-dependent diagonal singularity FAILED
description: All four brief approaches (per-pair half-M1, chord substitution, Galerkin diagonal cell-average, full Galerkin double-integration) FAIL on end-to-end k_eff. Per-pair K_bc[i,j] off-diagonals do NOT match Phase 4 specular_multibounce or Sanchez Eq. (A6) — different objects in different normalisations. Chord substitution makes off-diagonals MACHINE-PRECISE (rel 1e-9) but diagonal still log-divergent. The continuous-µ form does NOT Nyström-discretise — recommend R3-C ABANDON Phase 5 production wiring.
type: project
---

# Phase 5+ Round 3 — Adaptive quadrature (2026-04-28)

## TL;DR

**Verdict: SHIP STATUS NOT SHIPPED. All four brief approaches fail.**

The hypothesis "with the corrected `(1/2)·µ·T(µ)` factor and adaptive
quadrature, the continuous-µ form converges to the Phase 4 reference"
is FALSIFIED on every approach. The per-pair µ-resolved K_bc[i,j] is a
DIFFERENT OBJECT than both Phase 4 specular_multibounce K_bc and
Sanchez Eq. (A6) K_bc, even on off-diagonals.

The diagonal log-divergence is unchanged by the multi-bounce factor
choice (M2 weight=1 ↔ BACKUP weight=(1/2)·µ both fail). Chord
substitution `s = √(µ²-µ_min²)` makes off-diagonals MACHINE-PRECISE
to 1e-9 (large win for off-diagonals, but) the diagonal singularity
moves but doesn't disappear. End-to-end k_eff fails on every approach
by 35-46%.

**Recommendation: R3-C — ABANDON Phase 5 production wiring.**
``closure="specular_multibounce"`` at N ≤ 3 is the production form
forever. Phase 5 demoted to research-grade reference (Sphinx-only).

## Empirical evidence

### Approach 1 — Per-pair half-M1 quadrature

Same per-pair structure as M2 PRIMARY but with the BACKUP-corrected
`(1/2)·µ/(1-e^{-σ·2Rµ})` multi-bounce factor. Q-divergence on
diagonal entries IDENTICAL to M2 weight=1:

| (i,j) | type | Q=16 | Q=64 | Q=128 | Q=512 | rel(128 vs 512) |
|-------|------|------|------|-------|-------|------------------|
| (0,0) | DIAG | 0.064 | 0.090 | 0.103 | 0.130 | **0.20** |
| (1,1) | DIAG | 0.120 | 0.165 | 0.188 | 0.234 | **0.20** |
| (2,2) | DIAG | 0.060 | 0.079 | 0.089 | 0.108 | **0.18** |
| (0,1) | OFF  | 0.0232 | 0.0237 | 0.0238 | 0.0239 | 0.0025 |
| (others off-diag) | similar to off-diagonal in M2 |

Confirms: diagonal Q-divergence is independent of the multi-bounce
factor weight choice (M2 finding holds with the BACKUP correction).

### Approach 2 — Chord substitution `s² = µ² - µ_min²`

**Massive win on off-diagonals**: `dµ = (s/µ) ds` jacobian absorbs
ONE of the `1/cos(ω)` factors at the visibility cone endpoint.

| (i,j) | type | rel(128 vs 512) |
|-------|------|------------------|
| OFF entries | **rel ≈ 1e-9** (MACHINE PRECISION) |
| DIAG entries | rel ≈ 0.20 (unchanged log divergence) |

The chord substitution is the **right substitution for the
off-diagonals** but fails on the diagonal because BOTH cos(ω) factors
vanish at the same µ_min(r) — only one is absorbed by the substitution.

### Approach 4 — Galerkin diagonal cell-average regularisation

End-to-end smoke test on R=5, σ_t=0.5 (τ_R=2.5) with `r_wts`-scaled
δ-cell averaging on the diagonal:

| Q | k_eff | rel_inf | rel_heb |
|---|-------|---------|---------|
| 16 | 0.119 | -42.7% | -42.7% |
| 256 | 0.120 | -42.3% | -42.2% |

Catastrophic failure: regularisation produces a kernel with
fundamentally wrong magnitude. δ-sweep (0.1 to 2.0): all give
rel_heb in [-44%, -42%] — INSENSITIVE to δ. The structural problem
isn't δ, it's the formula itself.

### Approach 5 — Zero diagonal sanity check

Set K_bc[i,i] = 0:

| Q | k_eff | rel_heb |
|---|-------|---------|
| 32 | 0.113 | -45.9% |
| 128 | 0.113 | -45.8% |

Worse than Approach 4 by ~3% — confirms diagonal IS load-bearing,
not negligible. The Hadamard-finite-part interpretation cannot rescue
this.

### Approach 6 — Nearest-neighbour diagonal interpolation

Replace K_bc[i,i] with mean of K_bc[i, i±1]:

| Q | k_eff | rel_heb |
|---|-------|---------|
| 16 | 0.116 | -44.4% |
| 256 | 0.116 | -44.0% |

Same regime as Approach 4 — ad-hoc interpolation gives the same
WRONG answer, regardless of how the diagonal is regularised. The
issue is structural, not regularisation-choice.

### Smoking gun: per-pair K_bc[i,j] ≠ Phase 4 K_bc[i,j]

Comparing Phase 4 `closure="specular_multibounce"` rank-3 vs A2
chord-substitution at N=3 nodes:

```
K_bc Phase 4 (rank-3 specular_mb):
[[1.5e-3   4.6e-2   1.6e-2]
 [1.5e-3   5.0e-2   7.0e-2]
 [2.6e-4   3.6e-2   3.3e-1]]   ← row dominated by surface node

K_bc A2 (chord subst, Q=128):
[[2.06e-1  2.39e-2  4.91e-3]
 [1.81e-3  3.68e-1  8.92e-3]
 [1.41e-3  3.84e-2  1.61e-1]]  ← diagonal dominated, different shape
```

Off-diagonal ratio Phase4/A2 ranges 0.007 to 7.88 — NOT a constant or
separable Jacobian. The two K_bc forms are **different objects**.

Hybrid test: A2 off-diagonals + Phase 4 diagonal substituted in →
k_eff = 0.135 (-35% rel_heb). Confirms: even Phase 4's diagonal can't
rescue A2's off-diagonals.

### Smoking gun: A2 vs Sanchez Eq. (A6)

Comparing A2 (per-pair half-M1) vs `compute_K_bc_specular_continuous_mu_sphere`
(Sanchez 1986 Eq. (A6)) on same r_nodes:

Ratio K_a2/K_sanchez varies from **0.04 to 1.34** — column-dependent
non-uniform pattern. Confirms Front C's earlier finding that
Sanchez↔ORPHEUS conversion is non-scalar. A2 and Sanchez are
DIFFERENT INTEGRAL EQUATIONS in different normalisations.

### Approach R3-B — Full Galerkin double-integration

Last-ditch attempt: `K^{Gal}_{ij} = ∫∫ L_i(r)L_j(r') K_bc(r,r') dr dr'`
with constant trial functions on cells defined by GL panels. End-to-end
k_eff:

| n_inner | Q | k_eff | rel_heb |
|---------|---|-------|---------|
| 2 | 32 | 0.135 | -32.8% |
| 4 | 64 | 0.130 | -35.3% |
| 8 | 128 | 0.128 | -36.6% |

Convergence MONOTONIC but to the WRONG ANSWER. As `n_inner` increases,
k_eff DECREASES — adding more cells averages over more of the (now
smooth on each cell) singularity, but the underlying integral is
still divergent on the diagonal cells.

## Why every approach fails — root cause

The continuous-µ K_bc kernel is HYPERSINGULAR:

```
K_bc(r, r') = ∫ G_in(r,µ) F_out(r',µ) f_∞(µ) dµ
```

where `f_∞` is bounded but `G_in · F_out` carries
`1/(cos(ω(r,µ)) cos(ω(r',µ)))`. At `r = r' = r_*`, both factors
vanish at SAME `µ_min(r_*)`, producing a log-non-integrable diagonal.

This is a **Cauchy principal-value / Hadamard finite-part kernel**.
The "right" continuous-µ K_bc is NOT the pointwise integral but its
DISTRIBUTIONAL value, which requires regularisation.

The Phase 4 matrix-Galerkin form `(I-T·R)^{-1}` finesses this by
projecting onto a finite-rank modal basis BEFORE integrating over µ.
The basis truncation IS the regularisation — modal coefficients see
only the "averaged" diagonal, not the pointwise hypersingular value.

## Why no Nyström discretisation exists

A Nyström quadrature for `K_ij = w_j · K(r_i, r_j)` requires
pointwise K(r,r') to be a finite continuous function. Our kernel is
NOT — it's hypersingular on `r = r'`. **No Nyström discretisation of
the continuous-µ form can converge.**

The four brief approaches all attempt to "fix Nyström" by various
regularisation schemes:
- A1: ignore the singularity (Q-diverges, MOOT)
- A2: chord substitution (off-diag fix; diag still divergent)
- A3 (not run): explicit Gauss-Kronrod with subdivision (would also
  fail — kernel is non-integrable, not just badly-conditioned)
- A4: cell average (gives wrong magnitude)

None can succeed in principle.

## Production wiring status

**NOT SHIPPED.** Phase 5 production wiring is BLOCKED by:

1. The continuous-µ K_bc is hypersingular on the diagonal —
   non-Nyström-discretisable.
2. Galerkin double-integration (the natural cure) does not match
   Phase 4 reference numerically — different formula despite same
   physical interpretation.
3. The Sanchez 1986 Eq. (A6) reference and the M1/M2 ORPHEUS-native
   forms are different objects in different normalisations
   (Sanchez↔ORPHEUS Jacobian is non-scalar — confirmed across Front A,
   Front B, M2 PRIMARY, Round 3 A2-vs-Sanchez).

## Recommendation: R3-C — Abandon Phase 5 production wiring

`closure="specular_multibounce"` at N ≤ 3 is the production form
**forever** for sphere/cylinder specular BC. Slab specular_multibounce
ships at any N (geometric immunity per phase4 cyl/slab memo).

Phase 5 reduces to:
- Sphinx documentation of Sanchez 1986 Eq. (A6) reference kernel
  (already in `compute_K_bc_specular_continuous_mu_sphere`)
- Round 1 + 2 + 3 memos as permanent record of what was tried and why
  each approach fails
- The `closure="specular_continuous_mu"` `NotImplementedError` stays;
  message updated to point to the failure analysis

## What WAS accomplished in Round 3

1. **Confirmed (3rd time)** the diagonal log-divergence persists
   independent of multi-bounce factor choice (M2 weight=1, BACKUP
   weight=(1/2)·µ both fail equally on diagonal Q-convergence at
   `~0.20 per Q-doubling`).

2. **Discovered** chord substitution `s² = µ² - µ_min²` makes
   off-diagonal Q-convergence MACHINE-PRECISE (rel 1e-9 at Q=128).
   This is a useful technique for any future µ-resolved per-pair
   integral that doesn't have diagonal singularity (e.g., emission
   from one cell to another in a different geometry).

3. **Proved** Phase 4 K_bc[i,j] structure is qualitatively different
   from per-pair µ-resolved K_bc[i,j] — not a normalisation
   difference. The matrix-Galerkin and continuous-µ forms compute
   genuinely different integral operators.

4. **Confirmed** the M1 sketch from cross-domain memo and the BACKUP
   `(1/2)·µ` factor are derived for SCALAR (separable test kernel,
   F = G = e^{-aµ}), not the general `K_bc(r, r')` operator. Round 2
   BACKUP's caveat ("separable kernel does not exercise r-dependent
   diagonal") is empirically confirmed: the BACKUP closed form has
   no production utility.

## Possible Round 4 directions (not pursued)

If the user insists on Phase 5 production wiring:

- **R4-A**: Symbolic derivation of Hadamard finite-part for the
  diagonal value. Likely produces a closed-form involving log(µ_min)
  + smooth term. Risk: gauge ambiguity in the regularisation.
- **R4-B**: Sanchez-style integral-equation discretisation (treat
  `J^+(µ)` as the unknown, NOT `φ(r)`). Different integral equation
  with different (potentially non-singular) kernel. Effectively the
  augmented Nyström direction (Issue #132 augmented_nystrom memo,
  WONTFIX'd).
- **R4-C**: Operator-norm regularisation: pre-multiply K_bc by a
  smoothing operator that absorbs the singularity. Lukas-Marshak
  style.

None of these are likely to succeed without significant additional
investigation, all carry the risk of producing artefacts at the
regularisation gauge level.

## Files shipped

All in `derivations/diagnostics/`:

- `diag_phase5_round3_adaptive_quadrature.py` — main test file with
  approaches 1, 2, 4, 5, 6
- `diag_phase5_round3_phase4_compare.py` — proves per-pair K_bc ≠
  Phase 4 K_bc (off-diagonals differ by 0.007–7.88×)
- `diag_phase5_round3_check_against_sanchez.py` — proves per-pair
  K_bc ≠ Sanchez Eq. (A6) (column-dependent ratio)
- `diag_phase5_round3_phase4_structure.py` — Phase 4 structure
  diagnostics across rank N (shows surface-row dominance pattern)
- `diag_phase5_round3_galerkin_full.py` — full Galerkin
  double-integration smoke test (-32% to -38% rel_heb)

## Lessons learned

- **Round 2 BACKUP's separable test kernel proves the multi-bounce
  factor structure is right WHEN R-COUPLING IS REMOVED**, but does
  NOT exercise the operator-level singularity that's the actual
  blocker. Future BACKUP-style validations need r-coupled probe.
- **Chord substitution is excellent for visibility-cone integrals**
  with single endpoint singularity (off-diagonals). Off-diagonal
  Q-precision goes from 1e-3 (plain GL) to 1e-9 (chord subst) at
  Q=128. This is a portable technique for Sanchez-style integrals.
- **Per-pair Nyström and matrix-Galerkin compute DIFFERENT things**.
  The matrix-Galerkin form's `(I-T·R)^{-1}` mixes different rank
  modes; the per-pair Nyström samples K(r_i, r_j) at single points.
  These are not numerically equivalent without an explicit
  modal-projection step that we never had.
- **All four R3 brief approaches were predictable failures from theory.**
  In hindsight, the smoking gun was already in the M2 K_max=0
  diagnostic: K^(0) (no multi-bounce) Q-diverges on the diagonal —
  that's the kernel itself being singular, not just T(µ). Should
  have stopped after A1 confirmed this.

## LoC delta

- Diagnostics: +650 LoC across 5 new files
- Production code: 0 LoC (NotImplementedError unchanged)
- Sphinx docs: 0 LoC (separate task — recommend updating
  `closure="specular_continuous_mu"` `NotImplementedError` message
  to reference this memo and Round 2 PRIMARY)
- Tests: 0 LoC promoted to permanent suite (diagnostics are
  research-grade only)
