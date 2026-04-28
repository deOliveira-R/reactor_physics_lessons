---
name: Phase 5+ Round 2 — M2 bounce-resolved expansion FAILED at K_max=0
description: M2 geometric-series expansion of T(µ) does NOT bypass the diagonal singularity. The 1/cos²(ω) factor at µ → µ_min(r) is PRESENT IN K_bc^(0) (bare specular, no multi-bounce). At fixed K_max, diagonal entries Q-DIVERGE linearly with log(Q) — same +20%/Q-doubling rate Front A measured for Sanchez. Off-diagonal entries Q-converge spectrally. The M2 form is K_max-converged at K_max=5, weight=1 (Sanchez form, NOT µ in numerator), so the bounce truncation works but the per-bounce integral itself is divergent on the diagonal. The continuous-µ form is structurally singular regardless of M1/M2/Sanchez/native — singularity is in F·G·1/cos²(ω) at the visibility cutoff, NOT in T(µ).
type: project
---

# Phase 5+ Round 2 — M2 bounce-resolved expansion (2026-04-28)

## TL;DR

**Verdict: FAIL at smoke test, but NEW structural insight.**

The M2 geometric-series expansion of T(µ) is well-posed
mathematically: K_max-truncation converges within 0.1 % of the
infinite series at K_max=5 (geometric ratio ~0.25 per bounce). But the
PER-BOUNCE integrand `K^(k) = 2 ∫ G_in(r_i,µ)·F_out(r_j,µ)·e^{−k·τ_ch(µ)} dµ`
has the SAME 1/µ²-type diagonal singularity that killed Sanchez
(Phase 5a) and the M1 ORPHEUS-native form (Front C). The singularity
is in `F_out · G_in / cos²(ω)` at `µ → µ_min(r)`, NOT in T(µ).

Singularity LOCATION moved (Front C: at µ=0; M2: at µ=µ_min(r)) but
NOT eliminated. The substitution `u² = (µ²−µ_min²)/(1−µ_min²)` removes
ONE factor of cos(ω) but not the second when r_i = r_j (diagonal).

## Empirical evidence

### 1. Weight pinning ✓

The CORRECT M1 weight is `w(µ) = 1` (Sanchez form), NOT `w(µ) = µ`
(parent agent's M1 sketch). Confirmed by rank-1 cross-check:

| K^(0) form (k=0 only) | k_eff | |k_M2 - k_bare_Phase4| |
|------------------------|-------|------------------------|
| weight=µ              | 0.170340 (-18.24%) | 2.05e-2 |
| weight=1              | 0.191015 (-8.31%)  | 2.21e-4 |

Phase 4 closure='specular' rank-1 bare gives k_eff = 0.190795 (-8.42%).
weight=1 matches it to 4 figures; weight=µ is 100× worse. SymPy V2's
verdict (no µ in numerator of T) is empirically correct.

### 2. K_max truncation: spectral convergence ✓

| K_max | k_eff (weight=1) | rel_kinf  |
|-------|------------------|-----------|
| 0     | 0.264239 (Q=512) | +26.83 %  |
| 1     | 0.276127         | +32.54 %  |
| 5     | 0.276738         | +32.83 %  |
| 50    | 0.276740         | +32.84 %  |

Geometric series converges to within 1e-5 by K_max=5. Per-bounce decay
ratio = 0.25 (consistent with `e^{-σ·2R·µ_eff}` for an effective
µ ≈ 0.27).

### 3. Q-convergence at fixed K_max: FAILS ✗

`compute_K_bc_M2_per_pair` (per-pair µ-quadrature on joint visibility
cone `[max(µ_min_i, µ_min_j), 1]`) at K_max=10:

| Q   | k_eff   | rel_inf  | ‖ΔK‖_F/‖K‖ |
|-----|---------|----------|-------------|
| 16  | 0.184913 | -11.24 % | nan        |
| 32  | 0.197666 | -5.12 %  | 0.138      |
| 64  | 0.211286 | +1.42 %  | 0.124      |
| 128 | 0.226148 | +8.55 %  | 0.112      |
| 256 | 0.242609 | +16.45 % | 0.102      |

‖ΔK‖_F decreasing but `k_eff` MONOTONIC INCREASING with Q —
NOT converging. Hits k_inf at Q ≈ 64 by accident.

cos(ω)-substitution `u² = (µ² − µ_min²)/(1 − µ_min²)` made it WORSE
(k_eff → +163% at Q=256) because the substitution kills ONE cos(ω)
factor but doubles the OTHER's contribution near the diagonal.

### 4. Diagonal vs off-diagonal: surgical proof ✗ (THE smoking gun)

3-node grid at R=5, σ_t=0.5. K_max=0 (NO multi-bounce — bare
specular only):

| (i,j)  | type | Q=16 | Q=64 | Q=128 | Q=512 | rel(128 vs 512) |
|--------|------|------|------|-------|-------|------------------|
| (0,0)  | DIAG | 0.129 | 0.180 | 0.206 | 0.259 | **0.20** |
| (1,1)  | DIAG | 0.266 | 0.370 | 0.423 | 0.528 | **0.20** |
| (2,2)  | DIAG | 0.203 | 0.277 | 0.315 | 0.390 | **0.19** |
| (0,1)  | OFF  | 0.0464 | 0.0473 | 0.0475 | 0.0476 | 0.0026 |
| (0,2)  | OFF  | 0.00953 | 0.00973 | 0.00976 | 0.00978 | 0.0025 |
| (1,2)  | OFF  | 0.01888 | 0.01926 | 0.01932 | 0.01937 | 0.0025 |
| (2,0)  | OFF  | 0.00273 | 0.00278 | 0.00279 | 0.00280 | 0.0025 |
| (2,1)  | OFF  | 0.0814 | 0.0831 | 0.0834 | 0.0837 | 0.0027 |

OFF-DIAGONAL entries Q-converge at algebraic rate ~Q^{-1} (rel = 0.0025
at Q=128). DIAGONAL entries grow LINEARLY with `log(Q)` — same +20%
per Q-doubling that Front A measured for Sanchez K[surf,surf]. Same
behavior at K_max=1, 5, 50 (changing K_max only adds a converged
bounce-series amplitude; the singularity is INDEPENDENT of K_max).

This is the **Phase 5 1/µ² diagonal singularity** in M2 form:

- F_out · G_in carries `1/cos(ω_i) · 1/cos(ω_j)` Jacobian
- At `µ = µ_min(r) = √(1 − (r/R)²)`, `cos(ω(r,µ)) → 0`
- For r_i = r_j (diagonal), BOTH factors vanish at the SAME µ_min;
  product behaves as `1/cos²(ω) ∝ 1/(µ² − µ_min²)` — NON-INTEGRABLE.

For OFF-diagonal r_i ≠ r_j, `µ_min_i ≠ µ_min_j`, so on
`[max(µ_min_i, µ_min_j), 1]` only ONE cos(ω) factor vanishes
(integrable √-singularity), the other is bounded.

## Why M2 was supposed to bypass this and didn't

The cross-domain memo's M2 frame proposed:

```
T(µ) = Σ_k e^{-k·τ_chord(µ)}     (geometric series, k=0..∞)
```

Each `e^{-k·τ}` IS bounded at µ=0. Front C and prior fronts assumed
the singularity was IN T(µ)/at µ=0. M2 was designed to remove
SINGULARITY IN T(µ) AT µ→0.

But the singularity Sanchez and Front A actually identified
(*1/µ² at the SURFACE diagonal* per SymPy V4) is DIFFERENT — it is in
`F_out · G_in` (specifically, in the chord-projection Jacobian
`µ_*^{-1}` of Sanchez Eq. (A6), or equivalently in the `1/cos²(ω)` of
the µ-resolved native primitives). This singularity is INDEPENDENT
of T(µ), so the M2 reformulation does NOT bypass it.

The cross-domain memo's `µ → 0 cancellation via `µ·T(µ) → 1/(2σR)``
was a misdiagnosis: T(µ) IS bounded if multiplied by µ, but the
problematic factor isn't T(µ) — it's the Jacobian in F·G itself.

## Production wiring status

**NOT SHIPPED.**

The M2 form is K_max-converged but not Q-converged on the diagonal.
Routing through the Phase 5 wiring path (replacing `NotImplementedError`)
would expose the same `K[surf, surf] → 2.33` divergence Front A
measured for Sanchez. Same blocker, different rephrasing.

## What the diagnostic actually accomplished

1. **Weight pinning**: The Sanchez M1 form (no µ in T's numerator) is
   the ONLY weight choice consistent with the Phase 4 bare-specular
   closure at rank-1. SymPy V2's algebraic prediction is confirmed
   numerically. Future Phase 5+ implementations should use weight=1.
2. **K_max truncation bound**: The geometric series converges at
   ratio ≈ 0.25 for σ_t·R = 2.5 (5 % off Hebert at K_max=10). For
   thicker τ_R the ratio is smaller (faster convergence). K_max=10
   is the recommended truncation.
3. **Diagonal singularity LOCATED**: at `µ → µ_min(r)`,
   `cos(ω(r,µ)) → 0`, yielding `1/cos²(ω) ∝ 1/(µ²−µ_min²)`
   non-integrable on the visibility cone for `r_i = r_j`. The
   substitution `u²=(µ²-µ_min²)/(1-µ_min²)` removes ONE cos factor
   only.

## Possible Round 3 directions

The continuous-µ form is now empirically pinned as **structurally
singular at every (r_i, r_i) diagonal**, regardless of T(µ)
reformulation. Three remaining mathematical approaches exist:

### Direction R3-A: Singularity subtraction with TWO closed-form addbacks

Front B subtracted the M1 leading-order `c/µ` and got the smooth
remainder integrable, BUT the discarded singular piece had
divergent integral. M2 has the SAME issue at the diagonal.

The fix would require finding the closed-form value of
`∫ 1/cos²(ω(µ)) · (smooth) dµ` over the visibility cone. cos(ω) =
`√((µ²−µ_min²)/(1−µ_min²))`, so `1/cos²(ω) = (1−µ_min²)/(µ²−µ_min²)
= (1−µ_min²)/((µ−µ_min)(µ+µ_min))`. Partial fraction → `1/(µ−µ_min)`
log-divergent, but a regularised value can be computed via
Hadamard finite-part if one accepts the regularisation as the
"physical" diagonal value.

### Direction R3-B: Galerkin (cell-averaged) diagonal

Replace pointwise `K[i,i]` with the cell-integrated form:

```
K^{Gal}_{i,i} = ∫∫ L_i(r) · L_i(r') · K(r, r') dr dr'
```

where `L_i(r)` is the trial-function for cell i. The double integral
smooths the diagonal singularity (the off-diagonal samples r ≠ r'
exclude µ_min coincidence). This is the standard cure for
hypersingular boundary integrals (Burton-Miller etc.). Cost: every
diagonal entry becomes a 2-D integral.

### Direction R3-C: ABANDON Phase 5 production wiring

Already-shipped `closure="specular_multibounce"` at N ≤ 3 covers the
production case. Phase 5 reduces to research-grade verification
artifact only — Sphinx documents Round 1 + Round 2 as a permanent
record of why the continuous-µ form does not Nyström-discretise.

## Decision recommendation

**R3-C** for the next round, with R3-B reserved for a future deep
dive if shipping Phase 5 as a pure Galerkin form ever becomes
needed. R3-A is risky (regularisation gauge ambiguity) and gains
little over the existing N=1,2,3 envelope.

## Files shipped

All in `derivations/diagnostics/`:

- `diag_phase5_round2_m2_bounce_resolved.py` — main M2 form,
  weight pinning, K_max truncation, Q sweep at fixed K_max
- `diag_phase5_round2_q_at_fixed_K.py` — Q convergence per-bounce
  proves K^(k) NOT Q-converging
- `diag_phase5_round2_integrand_audit.py` — visibility cone reveals
  step discontinuity at µ=µ_min
- `diag_phase5_round2_per_pair_quadrature.py` — per-pair quadrature
  on joint visibility cone; off-diagonal converges, diagonal does not
- `diag_phase5_round2_cosomega_subst.py` — `u²=(µ²-µ_min²)/(1-µ_min²)`
  substitution; makes Q-divergence WORSE (kills wrong cos(ω))
- `diag_phase5_round2_K0_alone.py` — proves K^(0) (single-bounce, no
  T(µ)) ALSO Q-diverges on the diagonal; smoking gun

## Lessons learned

- **The cross-domain "M2 bypasses µ→0 singularity" insight was
  half-right.** T(µ) has a removable simple pole at µ=0 (cancelable by
  µ in numerator); but that's NOT the dominant singularity. The
  dominant one is the F·G·1/cos²(ω) diagonal, which is at
  `µ=µ_min(r)`, not at µ=0.
- **Bare specular K^(0) Q-diverges on the diagonal** in the
  µ-resolved native form. This says the Phase 4 ω-quadrature
  (which is convergent) and the M2 µ-quadrature ARE NOT
  EQUIVALENT — the change of variables ω → µ INTRODUCES the
  diagonal singularity. Phase 4 doesn't see it because of the
  `sin(ω)` measure factor on the ω basis.
- **Weight pinning by rank-1 cross-check is a fast, definitive
  test.** Whenever there's an algebraic ambiguity about Jacobians
  (µ in numerator vs not), comparing the bare K^(0) to a known-good
  reference (closure='specular' rank-1) settles it in one Q-sweep.
- **The Phase 4 ω-quadrature should be considered the production
  form for sphere specular.** Its convergence is rank-N basis
  rate-limited (overshoots at N≥4 per Trefethen-Embree multiplication-
  operator theorem) but bounded at low N. Continuous-µ doesn't have
  this advantage.
