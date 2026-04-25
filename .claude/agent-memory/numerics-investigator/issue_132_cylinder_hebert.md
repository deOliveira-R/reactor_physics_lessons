---
name: Issue #132 cylinder Hébert (1-P_ss)⁻¹ × rank-1 Mark
description: Cylinder analog of sphere Hébert closure — P_ss^cyl derived (Bickley Ki₃) and verified, but the geometric-series factor alone leaves ~10 % residual k_eff error on Class B 1G/1R. Root cause structural: cylinder compute_G_bc 2-D-projected-cosine kernel is biased by ~8 % (row-sum K·1 = 0.89·σ_t under Hébert closure). Knyazev 3-D Ki_{2+k} correction (Issue #112 Phase C) is REQUIRED before the Hébert path becomes useful.
type: project
---

# Issue #132 — Cylinder Hébert closure: viability investigation

## Bottom line

**Hypothesis 1 (P_ss derivation): CONFIRMED.** Closed-form 1-D
quadrature `P_ss^cyl(τ_R) = (4/π) ∫₀^{π/2} cos α · Ki_3(2 τ_R cos α) dα`
verified vs Monte Carlo to ~1e-4 (MC stderr-limited) over τ_R ∈ [0.1,
5]. Multi-region extension uses the SAME chord-segment-in-annulus
formula as sphere; sanity gates (Ki_3(0) = π/4, τ_R → 0 ⇒ P_ss → 1,
τ_R → ∞ ⇒ P_ss → 0) all pass bit-exact.

**Hypothesis 2 (Hébert series alone fixes 22 %): FALSIFIED.** Applying
1/(1-P_ss^cyl) to the cylinder rank-1 Mark closure on Class B:
- 1G/1R: −21.85 % → **−10.78 %** (improvement 11 pp, NOT <1 %)
- 1G/2R: −22.54 % → **−17.48 %**
- 2G/1R: −82.06 % → **−50.65 %**
- 2G/2R: −76.67 % → **−56.55 %**
The residual ~10 % gap on 1G/1R is the smoking gun for kernel bias.

**Hypothesis 3 (kernel needs Knyazev 3-D correction): CONFIRMED via
row-sum partition probe.** With Hébert closure applied:
- Sphere: K·1/σ_t = 0.999 ± 0.0005 (essentially exact)
- Cylinder: K·1/σ_t = **0.89 ± 0.07** (mean −7.6 %, range −11 % to +7 %)
Quadrature scan (n_angular ∈ {16, 24, 48} × n_surf ∈ {16, 32, 64, 128})
shows the 7.6 % bias is **structural**, not under-resolved (every
combination gives the same dev to 5 decimals).

## Why: missing 3-D polar-angle integration

The cylinder `compute_G_bc` (peierls_geometry.py:1493-1518) uses the
surface-centred Ki_1/d kernel with **2-D projected cosine**:
`G_bc(r_i) ∝ ∫_φ Ki_1(τ_surf(r_i,φ)) / d(r_i,R,φ) dφ`. For the
infinite cylinder, the inward angular flux carries a separate polar
angle θ_p (out of the (r,φ) plane) that the 2-D projection collapses.
The correct 3-D treatment integrates θ_p analytically and produces
**higher-order Bickley functions Ki_{2+k}**, mirroring how the cylinder
volume kernel uses Ki_1 (vs the slab's E_1) (Knyazev 1993, Atomic
Energy 74, DOI 10.1007/BF00844623; Sanchez 1982 NSE 80 §IV.A
Eqs. 47–52 give the canonical Ki_1 + Ki_3 structure).

The Hébert (1-P_ss)⁻¹ factor is GEOMETRICALLY IDENTICAL to the sphere
case (same Hébert §3.8.5 Eq. (3.323)) — but the underlying rank-1 Mark
K_bc that it multiplies has the wrong angular integration scheme. The
geometric series amplifies whatever Mark gives — so a 7.6 % kernel
under-prediction becomes ~10-20 % k_eff error after the eigenvalue
power iteration.

## Recommendation: PARTIAL — derivation ships, closure path BLOCKED on #112

1. **P_ss^cyl derivation (this work) is ready to ship** as a
   `compute_P_ss_cylinder` primitive in peierls_geometry.py paralleling
   `compute_P_ss_sphere`. Test gates already in
   `derivations/diagnostics/diag_cylinder_hebert_pss.py` (6 tests, all
   pass) — promote to `tests/cp/test_cylinder_pss.py` when shipping.

2. **Do NOT extend `boundary="white_hebert"` to cylinder** in its
   current `compute_G_bc` form. The 10–17 % residual on 1G is worse
   than several existing baselines (the bare rank-1 Mark gives ~22 %;
   F.4 rank-1 gives ~3 % for sphere). Shipping a partial fix would
   create a false sense of progress.

3. **Block on Issue #112 Phase C (Knyazev Ki_{2+k}).** The cylinder
   compute_G_bc must be re-derived as a 3-D integral; once the row-sum
   partition probe gives K·1 ≈ σ_t to <1 %, then ALSO apply (1-P_ss)⁻¹
   and re-run the Class B suite. Expected outcome (by analogy with
   sphere): <1.5 % k_eff error on 1G/1R and 2G/1R after both fixes.

## Files / data

- `derivations/diagnostics/diag_cylinder_hebert_pss.py` — 6 tests,
  derivation + MC verification + multi-region. PASS at 33 s.
- `derivations/diagnostics/diag_cylinder_hebert_keff.py` — Class B
  k_eff scan (1G/1R, 1G/2R, 2G/1R, 2G/2R). One pytest test, parametrised.
- `derivations/diagnostics/diag_cylinder_hebert_diagnose_residual.py`
  — row-sum K·1 partition probe + fixed-source pure-scatter probe.
  Localises the bias to `compute_G_bc` cylinder branch.

Reference k_inf (cp_cylinder._build_case): 1G/1R = 1.5, 1G/2R = 0.99,
2G/1R = 1.875, 2G/2R = 0.74 — these are the targets for the post-#112
re-test.

## Why (project memory):

The Hébert geometric series captures multi-bounce reflections off
the white BC and is geometry-agnostic (depends only on the cell-level
P_ss). The kernel-side correction (Knyazev for cylinder, Lambertian
observer-form for sphere) is what makes the surface-to-volume Green's
function physically correct in 3-D. The two corrections are
INDEPENDENT and BOTH required; previous Issue #132 framing implied
only the Hébert factor was missing.

## How to apply:

If asked to ship `boundary="white_hebert"` for cylinder: refuse and
point to this memory + Issue #112 Phase C dependency. Do ship the
P_ss^cyl primitive itself (it's a useful diagnostic tool independent
of the closure pathway). When Issue #112 lands the 3-D `compute_G_bc`,
re-run `diag_cylinder_hebert_diagnose_residual.py` first — if row-sum
K·1 / σ_t is within 1 % then re-enable the cylinder branch in
`_build_full_K_per_group` (peierls_geometry.py:4252-4262).
