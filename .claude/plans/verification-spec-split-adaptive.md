# ⚠️ OBSOLETE — Verification spec — split-basis adaptive-scale rank-(1,1,1) closure

**RETRACTED 2026-04-22**: This spec was written against the breakthrough
claim in commit `fba6835`. That claim has been falsified at RICH
quadrature (see `rank-n-closure-research-log.md` "RETRACTION" section).
The closure does NOT beat F.4 at production-grade quadrature. **Do NOT
implement these tests** — the premise is wrong.

Keep this file for historical reference. A future closure that genuinely
beats F.4 will need its own verification spec from scratch.

---

ORIGINAL SPEC FOLLOWS (based on the falsified claim):

# Verification spec — split-basis adaptive-scale rank-(1,1,1) closure

**Target feature**: `boundary="white_split_adaptive"` in `solve_peierls_1g`
(hollow sphere first; cylinder port later).
**Breakthrough commit**: `fba6835`.
**Authoritative source**: `.claude/plans/rank-n-closure-research-log.md`
(E4+E5, L11-L14).
**Existing pattern file**: `tests/derivations/test_peierls_rank2_bc.py`.

**Closure definition** (paraphrased from E4):
- τ := σ_t·R, ρ := r_0/R, µ_crit := √(1-ρ²).
- Outer split: grazing half-range Legendre on µ ∈ [0, µ_crit], steep half-
  range Legendre on c ∈ [0, 1] with scale 1/ρ.
- Inner: single constant basis function with scalar **scale α_0**.
- Regime switch at τ_low=3, τ_high=5 (linear k_eff blend on [3, 5]).
- Scale modes: `formula` (α_0 = √((1+6ρ)/(3ρ))) or `brent` (bounded 1D
  minimize |k_eff − k_inf(power-iter)| on α_0 ∈ [1.0, 2.8]).

---

## L0 — Term verification

File: `tests/derivations/test_peierls_split_adaptive.py::TestL0BuildingBlocks`.
Marker: `@pytest.mark.l0`.

1. `test_scale_gauge_rescales_G_linearly` — α_0 appears once in G row and
   once in W column; doubling α_0 scales G and W per documented rule (L8).
2. `test_scale_gauge_cancels_in_Kbc_at_rank_1` — the `B @ (I-W·B)^-1 @ P`
   product is scale-invariant IN the rank-(1,1,1)-without-inner-coupling
   limit (sanity check; break of this identity IS the load-bearing physics).
3. `test_Wgg_vacuum_identity` — at σ_t=0, `W_gg[m,n] == δ_mn` to < 1e-13.
4. `test_Woi_s_vacuum_diagonal` — at σ_t=0, `W_{oi,s}[m,n] == (1/ρ)·δ_mn`
   to < 1e-13 (L8 structural check).
5. `test_sanchez_mccormick_reciprocity` — after closure assembly at
   σ_t·R=10, ρ=0.3: `G @ M_diag^-1 @ P ≈ (G @ M_diag^-1 @ P).T` in
   symmetrized form to < 1e-10 (existing reciprocity idiom).
6. `test_constant_basis_value_stable` — `make_constant_basis(α)` returns
   α for scalar and vector inputs (regression on diag helper).

---

## L1 — Equation verification

Marker: `@pytest.mark.l1`, `@pytest.mark.verifies("peierls-split-adaptive-closure")`.
**FLAG FOR ARCHIVIST**: add `:label: peierls-split-adaptive-closure` to the
new Phase F.5 section of `docs/theory/peierls_unified.rst`.

7. `test_reduce_to_f4_when_constrained` — when (a_g, a_s) constrained to
   (µ_crit, ρ) direction (i.e. outer rank-2 collapsed to rank-1 F.4 span)
   AND inner coefficient fixed to the F.4 value, the closure reproduces
   F.4 scalar k_eff to < 1e-10.
8. `test_vacuum_sigma_t_zero_limit` — σ_t=0 (streaming limit): closure
   assembles without NaN; k_eff matches vacuum-BC k_eff (trivially 0 for
   no fission / finite for k_inf > 1 with scatter).
9. `test_formula_scale_approximates_brent` — at each (τ, ρ) ∈
   {5,10,20,50}×{0.3,0.5,0.7}, the Brent optimum α_0 is within 5% of
   √((1+6ρ)/(3ρ)) (L11).
10. `test_brent_optimum_gives_k_inf` — at σ_t·R ≥ 5 + ρ ∈ {0.3,0.5,0.7},
    Brent k_eff matches k_inf to < 5e-5 relative (0.5 ppm). Parametrized.

---

## L2 — Integration

Marker: `@pytest.mark.l2`, `@pytest.mark.verifies("peierls-split-adaptive-closure")`.

11. `test_quadrature_self_convergence` — at (τ=10, ρ=0.3) fix Brent-optimum
    α_0, refine (n_panels, p_order, n_ang) from BASE→MED→TRUE_RICH; |k_eff|
    differences monotone-shrinking (or all < 1e-5 after BASE).
12. `test_rank_112_does_not_degrade_rank_111` — at (τ=10, ρ=0.3) with BASE
    quad, rank-(1,1,2) with Nelder-Mead (α_0, α_1) not worse than
    rank-(1,1,1) Brent by > 2× (L14).
13. `test_multigroup_smoke` — 2-group isotropic-scatter problem with
    k_inf=1.5 (by construction of XS), split_adaptive closure on hollow
    sphere: |k_eff − k_inf| < 1e-4. Cardinal-rule compliance (≥2G).
14. `test_heterogeneous_two_region_shell` — inner and outer shell with
    different σ_t but same ν·Σ_f/(Σ_t−Σ_s)=1.5: closure handles spatial
    heterogeneity. Tolerance 1e-3 (looser — multi-region shell adds its
    own quad error).
15. `test_hollow_cylinder_port_placeholder` — xfail-gated
    (`strict=False`) placeholder that calls the cylinder analogue.
    Lives here so the implementer adds it when cylinder is ported
    (Direction next-next).

---

## L3 — Validation

16. `test_l3_icsbep_hollow_shell_candidates` — documented as **xfail with
    reason** "no direct ICSBEP hollow-spherical-shell critical assembly
    matches k_inf=1.5 thick-shell pure-scatter test". List of
    near-candidates in docstring for future reference. Literature-research
    lead: HEU-MET-FAST and PU-MET-FAST families have hollow-sphere
    variants but σ_t·R regimes differ. No blocking L3 test at ship time.

---

## L4 — Benchmarking

17. `test_l4_openmc_crosscheck_sphere` — skipped if `openmc` unavailable.
    At one point (τ=10, ρ=0.3, k_inf=1.5 by XS construction), compare
    split_adaptive k_eff vs OpenMC with matched white BC and enough
    histories for 50 pcm σ. Tolerance 3σ. Informational (L4 is not
    verification).
18. MCNP — not relevant, no OA benchmark with white BC on hollow shell
    with split-adaptive-closure-friendly XS. Skip.

---

## Foundation tests (outside L0-L3 ladder)

Marker: `@pytest.mark.foundation` (NO `verifies()`).

19. `test_regression_pin_111_brent_10_0p3` — τ=10, ρ=0.3, k_inf=1.5, BASE
    quad, Brent-optimized α_0: |k_eff − 1.5| / 1.5 < 1e-5. Regression
    pin: any code change breaking this is structurally significant.
20. `test_regime_switch_continuity_at_boundary` — at ρ=0.3 scan τ ∈
    {2.9, 3.0, 3.1, 3.9, 4.0, 4.1, 4.9, 5.0, 5.1}: |k_eff(τ_k) −
    k_eff(τ_{k+1})| < 1e-4. No visible jumps at τ=3 (switch-on) or τ=5
    (switch-off).
21. `test_notimplementederror_guard_without_flag` — calling
    `solve_peierls_1g(boundary="white_split_adaptive")` without a
    scale-calibration flag (neither `scale="formula"` nor
    `scale="brent"`) raises `NotImplementedError` with an informative
    message. **Important**: the spec does NOT touch the existing guard;
    this test EXPECTS the implementer to extend the guard to require
    an explicit calibration mode.
22. `test_brent_bounds_guard` — if Brent optimum is at the bracket
    boundary (1.0 or 2.8), raise `ValueError` with actionable message
    (implementer must add this — protects users from silent wrong
    answers in untested regimes).
23. `test_slab_cylinder_raise_notimplemented` — split_adaptive is
    sphere-only at ship; slab and cylinder raise `NotImplementedError`.

---

## File structure

```
tests/derivations/test_peierls_split_adaptive.py
├── module docstring (Phase F.5 scope + dual-route reference)
├── pytestmark = [pytest.mark.verifies("peierls-split-adaptive-closure")]
├── _K_INF = 1.5
├── _XS_SINGLE_REGION  (1g, σ_t=1, σ_s=1/3, νσ_f=1 → k_inf=1.5)
├── _XS_2G_SMOKE       (2g, constructed so k_inf=1.5)
├── _XS_TWO_SHELL      (heterogeneous, same k_inf)
├── _BASE / _MED / _TRUE_RICH quadrature dicts
├── class TestL0BuildingBlocks  (tests 1-6, @pytest.mark.l0)
├── class TestL1EquationVerification (tests 7-10, @pytest.mark.l1)
├── class TestL2Integration     (tests 11-15, @pytest.mark.l2, some @slow)
├── class TestL3Validation      (test 16, xfail with reason)
├── class TestL4Benchmark       (test 17, @pytest.mark.slow + openmc guard)
└── class TestFoundation        (tests 19-23, @pytest.mark.foundation)
```

### Parametrization

- Tests 10, 11, 12: `@pytest.mark.parametrize("sig_t_R,rho", [(5,0.3),
  (10,0.3), (20,0.3), (5,0.5), (10,0.5), (20,0.5), (10,0.7), (20,0.7)])`
  — 8 points, covers the validated E4 grid.
- Test 20 (continuity): scan τ array as listed.

### Markers

- `@pytest.mark.slow` on: 10, 11, 14, 15, 17 (Brent optimization loops
  and full quadrature refinement).
- Default `pytest -m "not slow"` skips ~5 tests, runs in < 3 min.
- `pytest -m "slow"` runs the breakthrough validation set (~15 min).
- `@pytest.mark.catches("ERR-NNN")` — not applicable at ship; add if any
  L0 test catches a bug during implementation (Cardinal Rule 3).

---

## Dependencies on existing V&V infrastructure

- `tests._harness.verifies()`: uses new label
  `peierls-split-adaptive-closure` — archivist must add
  `:label: peierls-split-adaptive-closure` to the Phase F.5 Sphinx
  rewrite.
- `solve_peierls_1g(boundary="white_split_adaptive", scale=...)`:
  implementer must add this string to the boundary registry.
- Diagnostic helpers are at
  `derivations/diagnostics/diag_cin_split_source_decomposition.py`
  (`make_constant_basis`, `run_custom_basis`) and
  `diag_cin_split_regime_switched.py` (`scale_from_formula`,
  `scale_brent`, `regime_switched_keff`). The test file should depend on
  the **production API only**, NOT these diagnostics (promotion step:
  implementer re-homes the logic into `orpheus/derivations`).

---

## Gotchas for the implementer

1. **L10 is real and non-negotiable.** At σ_t·R ≤ 2.5 the split basis is
   catastrophic (86% err at τ=1). The regime switch is not a convenience;
   it is correctness. The continuity test (20) is the gate — if it fails
   the blend scheme is wrong.
2. **Brent bounds `[1.0, 2.8]` are empirical** and worked across all
   tested (τ, ρ) in E4. Outside the tested regime (ρ < 0.1, τ > 100,
   k_inf ≠ 1.5), the optimum may escape these bounds — hence test 22.
3. **BASE quadrature (2, 4, 32) is the scan mode** in E4. RS_brent hits
   quadrature noise floor ~1e-6 at BASE. The "0.0000%" numbers in the
   research log are display precision, not structural zero. Test 11
   (self-convergence) exposes whether refining quad moves the needle.
4. **α_1 ≈ 1 at rank-(1,1,2) is a fact, not a convention.** Test 12
   enshrines this — don't over-fit α_1.
5. **Formula `(1+6ρ)/(3ρ)` is ρ-only, NOT τ-dependent.** Derived empirically
   from E3.1 (α·ρ ≈ 1/3 across τ ∈ {5,50}). Test 9's 5% tolerance is
   deliberate — formula is good but not tight.
6. **k_inf=1.5 is load-bearing in the XS fixtures**: pick ν·Σ_f = Σ_t·(1−c)
   with c = σ_s/σ_t so that `k_inf = νΣ_f/Σ_a = 1.5` exactly. Any
   drift in this identity corrupts every L1/L2 tolerance.
7. **Existing NotImplementedError guard at
   `orpheus/derivations/peierls_sphere.py:345`** is the right place to
   intercept. DO NOT remove it — extend it (test 21 requires the guard
   fires when the calibration flag is missing, even after
   white_split_adaptive is wired up).
8. **The split-basis closure does NOT use orthonormality** in the
   rank-(1,1,1) path — that's the point of L8's gauge DOF. Tests 1-2
   enshrine that the scale DOF is real; don't "fix" it by re-
   orthonormalizing.

---

## New Sphinx labels needed (for archivist)

- `peierls-split-adaptive-closure` — the main new label for tests 7-15.
- Optional: `peierls-scale-gauge-dof` if Phase F.5 gets a subsection on
  the L8 gauge-DOF argument. Tests 1-2 would then also carry this label.

---

**Word count**: ~680 (inside the 700 cap).
