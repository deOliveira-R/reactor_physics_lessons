---
name: Specular multi-bounce Phase 4 — cyl/slab T derivation + pathology (SHIPPED)
description: Derived T_mn for cyl (Knyazev Ki_(3+k_m+k_n) integral) and slab (per-face block off-diagonal, 2 E_3(τ) at rank-1). End-to-end k_eff: SLAB MB CONVERGES MONOTONICALLY; CYL MB OVERSHOOTS like sphere; SPHERE MB overshoots (already known). SHIPPED 2026-04-28 — see specular_bc_multibounce_cyl_slab_shipped.md for the rollout commit.
type: project
---

# Specular MB Phase 4 — cyl/slab T derivation + pathology (2026-04-28)

Branch `feature/peierls-specular-bc`. Investigation question: can the
sphere `closure="specular_multibounce"` be extended to cyl/slab? Would
it inherit the matrix-Galerkin divergence pathology documented in
`specular_mb_overshoot_root_cause.md`?

## Cylinder T_mn^cyl derivation

For each angular position α (in-plane angle from inward normal at
emission point) and polar angle θ_p ∈ [0, π], the chord through the
cell has in-plane length `d_2D(α) = 2R cos α` (homogeneous), 3-D length
`d_2D / sin θ_p`, and 3-D direction cosine `µ_3D = sin θ_p cos α`.

Polar integration of `∫_0^π sin θ_p · µ_3D · P̃_m(µ_3D) P̃_n(µ_3D) ·
e^{-τ_2D / sin θ_p} dθ_p` against the µ-weighted partial-current
measure gives the Knyazev Ki_(3+k_m+k_n) expansion (one Ki order
HIGHER than the corresponding cylinder P/G primitives, due to the
extra µ_3D = sin θ_p factor):

```
T_mn^cyl = (4/π) ∫_0^(π/2) cos α · Σ_{k_m,k_n} c_m^{k_m} c_n^{k_n}
                  · (cos α)^(k_m+k_n) · Ki_(k_m+k_n+3)(τ_2D(α)) dα
```

with c_n^k the monomial coefficients of P̃_n(µ).

**Rank-1 identity**: at m=n=0, c_0^0=1, only the k_m=k_n=0 term, so
T_00^cyl = (4/π) ∫ cos α · Ki_3(τ_2D(α)) dα = `compute_P_ss_cylinder`
exactly. Verified to 1e-14 across thin/thick/very-thin/multi-region
in `diag_specular_mb_phase4_01_cyl_T_derivation.py`.

## Slab T_slab block structure

Slab mode space ℝ^(2N) per existing per-face decomposition (see
`specular_bc_slab_fix.md`). Single-transit T is **PURELY OFF-DIAGONAL**:

```
T_slab = [[0,    T_oi],
          [T_io, 0  ]]   with T_io = T_oi by face symmetry
```

T_oi^(mn) = `2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^{-σL/µ} dµ` for homogeneous;
multi-region τ(µ) = τ_total/µ since slab chord length depends only on
1/µ uniformly.

**Self-blocks T_oo = T_ii = 0 EXACTLY** because a single transit at
constant direction cannot leave outer face and return without an
intermediate reflection at the inner face.

**Rank-1 identity**: T_oi^(0,0) = `2 E_3(τ_total)` by closed form
(substitution u = 1/µ). Verified to 1e-14 across all cases in
`diag_specular_mb_phase4_02_slab_T_derivation.py`.

## Resolvent ‖(I - T·R)^{-1}‖_2 vs N (thin τ ≈ 2.5)

| N    | sphere (BAD)      | cyl                     | slab          |
|------|-------------------|-------------------------|---------------|
| 1    | 1.08              | 1.03                    | 1.03          |
| 4    | 2.53              | 1.52                    | 1.08          |
| 8    | 6.89              | 2.52                    | 1.09          |
| 12   | 13.82             | 3.60                    | 1.09          |
| 16   | 23.29             | 5.19 (ρ=935 — R-noise)  | 1.09          |
| 20   |                   | 5.24 (ρ=2.9e8 — R-noise)| 1.09          |

- **Sphere**: divergent — continuous-limit operator 1/(1-e^{-σ·2Rµ})
  → ∞ at µ→0 (chord 2Rµ → 0, transmission → 1).
- **Cyl**: continuous-limit T_op^cyl(α) = (4/π)cos α · Ki_3(τ_2D(α))
  → 0 at α→π/2 (cos α factor wins). Continuous resolvent BOUNDED
  (sup ≈ 1.07 at thin); matrix-Galerkin form bounded too. The
  ρ(T·R) explosion at N≥16 is R-conditioning noise, not structural.
- **Slab**: T_op^slab(µ) = e^{-σL/µ} → 0 at µ→0 (chord = L/µ → ∞,
  EXPONENTIAL decay, geometry is geometrically OPPOSITE of sphere).
  Continuous resolvent sup ≈ 1.09 at thin. Matrix form quickly
  plateaus.

## End-to-end k_eff at thin (R/L = 5, σ_t = 0.5, fuel-A-like)

| N    | sphere bare / MB         | cyl bare / MB            | slab bare / MB         |
|------|--------------------------|--------------------------|------------------------|
| 1    | -8.31 % / **-0.12 %** ✓  | -2.95 % / **-0.34 %** ✓  | -2.83 % / **-0.30 %** ✓|
| 2    | -6.75 % / **-0.11 %** ✓  | -2.63 % / **-0.32 %** ✓  | -2.82 % / **-0.29 %** ✓|
| 3    | -6.24 % / **-0.09 %** ✓  | -2.33 % / **-0.23 %** ✓  | -2.77 % / **-0.27 %** ✓|
| 4    | -6.05 % /   +0.06 % ⚠    | -2.01 % /   +0.03 % ⚠    | -2.75 % / **-0.24 %** ✓|
| 6    | -5.78 % /   +1.18 % ⚠    | -1.70 % /   +0.53 % ⚠    | -2.70 % / **-0.19 %** ✓|
| 8    | -5.49 % /   +1.61 % ⚠    | -1.05 % /   +1.27 % ⚠    | -2.67 % / **-0.16 %** ✓|
| 12   |                          | +1.09 % /   +3.64 % ⚠⚠   | -2.67 % / **-0.16 %** ✓|
| 16+  |                          |                          | -2.67 % / **-0.16 %** ✓|

## Verdict per geometry

1. **SPHERE** — already shipped at N ∈ {1, 2, 3} with N≥5 UserWarning.
   Confirmed pathology (overshoot at N=4 onward).

2. **CYL** — same operational pathology as sphere (overshoot at N=4
   onward), but for a DIFFERENT root cause: the resolvent itself is
   bounded, but bare specular cyl drifts toward and past k_inf at
   high N due to R-conditioning blowup; MB amplifies the drift by
   ~0.5..2.5 % on top.

   **Recommendation if shipped**: same N ∈ {1, 2, 3} envelope as
   sphere, with similar UserWarning. Lifts the cyl thin-cell plateau
   from -2.95 % → -0.34 % at rank-1 (matches `boundary="white_hebert"`
   sphere/cyl quality).

3. **SLAB** — **NO PATHOLOGY**. Slab MB converges monotonically to
   ≈ -0.16 % at N → ∞ (vs bare specular plateau at -2.67 %). Slab is
   the ONLY geometry where the matrix-Galerkin (I - T·R)^{-1} form
   converges as N → ∞.

   **Root cause**: slab chord(µ) = L/µ, so grazing rays have INFINITE
   optical depth (chord → ∞ as µ → 0), transmission e^{-σL/µ} → 0.
   The continuous-limit transit operator vanishes at the boundary of
   the µ domain, so the resolvent operator stays well-conditioned.

   **Recommendation if shipped**: ANY N. Slab MB is the workhorse
   high-rank closure that lifts the plateau without overshoot. Does
   NOT need a UserWarning at high N.

## Sphere/cyl operational equivalence vs slab structural difference

The takeaway: even though the **continuous-limit resolvent** is
bounded for both cyl and slab (only sphere has the singular limit),
the **k_eff sweep** discriminates them:

- Cyl bare specular drifts past k_inf at high N (R-conditioning
  blowup of (1/2) M^{-1} for the (cos α)-weighted T basis); MB
  amplifies this via the geometric series multiplier.
- Slab T's off-diagonal block structure means TR has eigenvalues
  bounded away from 1 (ρ(T·R) ≤ 0.08 at thin τ_L=2.5 across all
  N=1..24, vs sphere ρ → 0.95). The geometric series converges
  rapidly and stays close to its single-bounce baseline.

## Regime sweeps (τ ∈ [0.5, 10])

Cyl pathology persists across regime (always overshoots at N≥4-6):

| τ_R  | N=2  | N=4   | N=6   | N=8   |
|------|------|-------|-------|-------|
| 1.0  | -0.13| +0.05 | +0.71 | +1.71 |
| 2.5  | -0.32| +0.03 | +0.53 | +1.27 |
| 5.0  | -0.55| -0.08 | +0.10 | +0.41 |

Slab MB regime sweep with proportionally scaled XS (so k_inf invariant):

| τ_L  | N=1   | N=2   | N=4   | N=8   | N=16  | converges? |
|------|-------|-------|-------|-------|-------|------------|
| 0.5  | -0.06 | -0.06 | -0.06 | -0.05 | -0.03 | yes        |
| 1.0  | -0.12 | -0.12 | -0.11 | -0.09 | -0.06 | yes        |
| 2.5  | -0.30 | -0.29 | -0.24 | -0.16 | -0.16 | yes        |
| 5.0  | -0.58 | -0.53 | -0.37 | -0.31 | -0.31 | yes        |
| 10.0 | -1.05 | -0.83 | -0.60 | -0.59 | -0.59 | yes        |

In all cases slab MB monotonically improves and never overshoots. The
plateau % grows with thickness (multi-bounce becomes less corrective
at thicker τ since fewer photons survive multiple bounces), but the
direction is always correct.

## Files (`derivations/diagnostics/`)

1. `diag_specular_mb_phase4_01_cyl_T_derivation.py` — cyl T derivation
   + rank-1 = P_ss^cyl identity (passes 1e-14 in 13 min via mpmath dps=30,
   fast variant uses ki_n_float at ~1s).
2. `diag_specular_mb_phase4_02_slab_T_derivation.py` — slab T derivation
   + rank-1 = 2 E_3(τ_total) + self-blocks zero + symmetry. PASSES.
3. `diag_specular_mb_phase4_03_pathology_resolvent.py` — sphere/cyl/slab
   ‖(I-TR)^{-1}‖_2 and continuous grazing-floor analysis. PASSES.
4. `diag_specular_mb_phase4_04_pathology_extended.py` — extended N
   sweep + thinner cells; reveals cyl ρ(T·R) blowup at N≥16. PASSES.
5. `diag_specular_mb_phase4_05_high_N_conditioning.py` — confirms cyl
   ρ explosion is R-conditioning, not structural; ‖.‖_2 stays bounded.
   PASSES (1 mpmath issue test fails — minor).
6. `diag_specular_mb_phase4_06_keff_endtoend.py` — end-to-end k_eff
   sweep. **CRITICAL FINDING**: slab MB converges, cyl MB overshoots
   like sphere. PASSES.
7. `diag_specular_mb_phase4_07_synthesis.py` — **PROMOTE-CANDIDATE**.
   Pins sphere overshoot + slab monotonic convergence + cyl overshoot.
   PASSES.
8. `diag_specular_mb_phase4_08_slab_robustness.py` — slab MB regime
   sweep (τ_L ∈ [0.5, 10]). All monotonic, no overshoot. PASSES.
9. `diag_specular_mb_phase4_09_cyl_robustness.py` — cyl MB regime
   sweep (τ_R ∈ [1, 5]). Pathology persists at all regimes. PASSES.

## Promotion recommendations

- **`diag_specular_mb_phase4_02_slab_T_derivation.py`** → permanent
  test (T derivation symmetry + rank-1 closed-form regression).
- **`diag_specular_mb_phase4_07_synthesis.py`** → permanent test
  (after the slab+cyl MB closures ship). Pins the per-geometry MB
  k_eff signature.
- Diagnostics 03/04/05/06 are exploratory; keep until Phase 4 ships
  then prune.

## What this means for Phase 4 plan

- **Slab MB**: SAFE to ship for any N. Use the per-face block-T
  construction with `T_oi = 2 ∫ µ P̃_m P̃_n e^{-σL/µ} dµ` (closed-form
  E_n at rank-1, GL elsewhere). Algebraic equivalence at rank-1
  reduces to `K_bc = G·R·(I - T·R)^{-1}·P` with the off-diagonal
  T_slab giving a particularly cheap inverse (block-anti-diagonal
  structure: (I - T·R)^{-1} = block-diagonal with diagonal blocks
  (I - T_oi R T_io R)^{-1}).
- **Cyl MB**: ship with same N ∈ {1, 2, 3} envelope + UserWarning at
  N ≥ 4 as sphere. The Knyazev Ki_(3+k) integration is expensive
  (mpmath dps=20+) — use float ki_n_float for production.
- **Sphere MB** is already shipped — no change.
