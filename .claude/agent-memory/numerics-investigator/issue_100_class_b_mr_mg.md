---
name: Class B rank-N MR catastrophe = mode-0 normalization hack
description: Sphere 1G/2R rank-2 = +57% k_eff bug diagnosed as mode-0/mode-n≥1 routing mismatch in build_closure_operator
type: project
---

**Date**: 2026-04-25. **Issue**: #100, #103. **Branch**: feature/rank-n-cin-aware-basis.

**Failing case**: sphere 1G/2R, radii=[0.5, 1.0], σ_t=[1, 2] with cp_sphere LAYOUTS=["A","B"]
(A inner = fissile, B outer = strong scatterer σ_s=1.9). Analytical k_inf = 0.6479728. Production
solver gives:
- rank-1 Mark: 0.551 (-15.0 %)  ← acceptable
- rank-2 Marshak: 1.015 (+56.7 %) ← catastrophic
- rank-3..8: 1.075..1.081 (+66 %) ← plateau, NOT converging to k_inf

**1R control** (σ_t=1, radii=[1.0], A only, k_inf=1.5):
- rank-1: 1.096 (-26.95 %)
- rank-2: 1.483 (-1.10 %) ← essentially correct
- rank-3..8: 1.533..1.538 (+2.4 %)

**Diagnosis** (Probe-cascade B, C, D, E, F, G, H — `derivations/diagnostics/diag_class_b_rank_n_probe_*.py`):

1. Probe B (vacuum BC, K_vol alone): 2R routing-invariance gap is ~2e-4 (Issue #114-style ρ-quad noise). K_vol is fine.
2. Probe C (homogeneous σ_t=1 with radii=[0.5,1] vs radii=[1]): rank-2 differs by ~1e-3. Pure routing OK.
3. Probe D (n_angular sweep on compute_P_esc_mode/G_bc_mode for 2R-Z): convergence is algebraic ~1/N (not bit-exact, but ~1e-5 at n_ang=192). Primitives essentially correct.
4. Probe E (per-node K·σ_t = σ_t identity): no localized large defect. Defects 5-7 % rms — but 1R control has 9 % rms with k_eff still right. Conservation defect is NOT a strong predictor.
5. Probe F (per-mode K_bc isolation): adding mode-1 jumps k_eff +84 % on 2R-Z (vs +35 % on 1R control). mode-N converges to +66 % above k_inf on 2R-Z, to -1.1 % on 1R. **mode-1 contribution scales differently in MR.**
6. Probe G (canonical mode-0 vs legacy mode-0): replacing legacy `compute_P_esc/compute_G_bc` for mode-0 with `compute_P_esc_mode(n=0)/compute_G_bc_mode(n=0)` (same Jacobian-weighted form as n≥1) gives:
   - 1R rank-2: -29.3 % (vs LEGACY -1.10 %)
   - 2R-Z rank-2: -28.0 % (vs LEGACY +56.7 %)
   - **Both cases plateau at ~-25 % under canonical** — CONSISTENT but worse-in-1R.
7. Probe H (thickness scan on 2R-Z): catastrophe is thin-cell only. At m=5 (σ_t,B = 10, R_outer/MFP=0.05), all ranks agree to <0.1 %. The hack works when leakage is small.

**Root cause** (likely): `build_closure_operator` (peierls_geometry.py:3618-3622) routes mode 0 through legacy `compute_P_esc` (no surface Jacobian, "isotropic escape probability") while modes n ≥ 1 use `compute_P_esc_mode` (with `(ρ_max/R)²` surface-to-observer Jacobian). This is documented in the function docstring as a "for bit-exact rank-1 regression" compromise. The two forms live in *different normalisation spaces*. In 1R the mismatch is partially absorbed by the rank-1 Mark closure being already ~correct (the legacy mode-0 is calibrated to make rank-1 nearly exact); when you add mode-1 the residual error happens to be small. In 2R with strong-scatterer outer, the calibration breaks and the mismatch is amplified.

**The fix is NOT a one-line edit.** Probe G shows that just switching mode-0 to canonical makes everything WORSE in 1R. The rank-N closure needs a re-derivation with *consistent* mode-0 and mode-n≥1 partial-current normalisation (probably the F.4 Schur reduction with proper Marshak DP_N basis on Class B), or accept the closure as a tuned heuristic only valid for 1R / thick-cell.

**Falsification of the original H_A/H_B/H_C tree (plan §3)**: outcome is **H_B (hidden bug)** — the bug is the mode-0 routing hack masquerading as a closure when it's really a calibration. Issues #100 and #103 should NOT close until the canonical normalisation question is resolved.

**Promotion suggestion**: probe G + H consolidated into `tests/derivations/test_peierls_rank_n_normalization_consistency.py` as an XFAIL with detailed reproducer. Probe C is a clean MR-routing-invariance regression test (passes today within 1e-3 — pin it).

**Known unknowns**: this analysis is for sphere only. Cylinder 2R-Z rank-2 also degrades (8 % → 18 %) but cylinder N≥3 is already known-divergent (Issue #112 Phase C) so the cylinder signal is partially confounded.
