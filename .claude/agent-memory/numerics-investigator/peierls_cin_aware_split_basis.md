---
name: c_in-aware split basis for hollow-sphere rank-N closure (structural verified, empirical fail)
description: Novel split-basis (grazing/steep/inner) closure with W_{oi,s} diagonal at σ_t=0 and rigorous Sanchez-McCormick time-reversal reciprocity; k_eff residual is STRICTLY WORSE than F.4 scalar at all tested (σ_t·R, r_0/R) because the Marshak-basis outer BC convention inherited by split basis mismatches F.4's effective Lambert-basis P/G primitives.
type: project
---

# c_in-aware split basis for hollow-sphere rank-N closure

## STATUS (2026-04-21, numerical verdict)

**STRUCTURAL: PASS.** Orthonormality, W block structure, F.4 decomposition
(µ_crit, ρ), Sanchez-McCormick time-reversal reciprocity all bit-exact.

**EMPIRICAL: FAIL.** Rank-(1,1,1) is ~10× WORSE than F.4 at the standard
test point (σ_t·R=5, r_0/R=0.3: 1.29% vs 0.12%). Adding more modes in
outer grazing/steep sub-bases gives ZERO improvement. Adding inner modes
saturates at ~0.99% residual. Gate (≤0.01% at rank-(1,1,1)) MISSED by
two orders of magnitude. DO NOT ship.

Full scan at r_0/R=0.3:

| σ_t·R | F.4 scalar | split-(1,1,1) | Ratio |
|-------|------------|----------------|-------|
| 1.0   | 2.36%      | 85.6%          | 36×   |
| 2.5   | 0.347%     | 3.86%          | 11×   |
| 5.0   | 0.122%     | 1.29%          | 10×   |
| 10.0  | 0.246%     | 0.934%         | 3.8×  |
| 20.0  | 0.365%     | 0.427%         | 1.2×  |

Split basis degrades accuracy uniformly, worst in thin-optical-depth regime.

## Diagnostic root cause

The existing Marshak rank-N closure (`build_white_bc_correction_rank_n`,
Phase F.5) gives:

- N=1 Marshak: err = 2.09%
- N=2 Marshak: err = 1.36%  ← **same plateau as my split-(1,1,1) = 1.29%**
- N=3 Marshak: err = 1.36%
- F.4 scalar (Lambert P/G, Marshak W): err = **0.12%**

My rank-(1,1,1) ≈ Marshak-N=2 because they carry the same BC information
(µ_crit · P̃_0^g + ρ · P̃_0^s = constant-in-µ = Marshak P̃_0). The
split basis just reshuffles the same 2-D outer mode-0 subspace into
grazing/steep coordinates, producing the same closure algebra at the
effective rank.

**F.4's 0.12% win is NOT from the outer-domain split — it's from using
Lambert-basis P_esc/G_bc** (integrand `sin θ exp(-τ)`, no µ_s weight on
the outgoing side). My split basis inherits the Marshak convention
(matches W basis), which is the formally-consistent choice but has
strictly worse k_eff than F.4's basis-mismatched recipe.

## Why the hypothesis failed

The memo hypothesis was: split basis structurally diagonalizes W at σ_t=0,
so rank-(1,1,1) should unlock new DOFs and break F.4's 0.077% ceiling.
Two observations invalidate it:

1. **The "extra DOF" (µ_crit, ρ)⊥ direction at mode-0 (the vector
   (-ρ, µ_crit)) gets ZERO drive from the white BC.** White re-emission
   is strictly in the (µ_crit, ρ) direction by construction. So the new
   DOF cannot be excited by the boundary.

2. **Higher-N outer sub-basis modes (N_g≥2 or N_s≥2) also get zero drive
   from white BC** (white re-emission is rank-1 at outer mode-0). These
   modes couple to the volume source only via P_esc, but volume
   emission on outer surface at mode n≥1 is tiny (few percent of total
   flux escape), and the corresponding transmission back into the volume
   through G_bc at mode n≥1 is the Marshak convention — already
   captured by `build_white_bc_correction_rank_n` for equal-DOF N.

Conclusion: the split basis is a **basis rotation** of the outer
partial-current space that does not introduce new information. It is
mathematically equivalent to the existing Marshak rank-2 (2 outer
modes + 1 inner mode) and numerically gives the same residual.

## What WOULD beat F.4

F.4's Lambert-basis trick is unprincipled (basis mismatch between P/G
and W) but empirically effective. To break the 0.12% ceiling while
staying basis-consistent would require:

1. **Include angular flux moments the white BC CANNOT re-emit.**
   The cavity inner surface might be the right place — higher-N inner
   modes see the full structured angular dependence of the re-entering
   steep rays. My rank-(1,1,2) → 0.99% vs rank-(1,1,1) → 1.29% confirms
   this is the dominant direction. But the improvement plateaus at 2
   inner modes: rank-(1,1,3) ≈ rank-(1,1,2).

2. **Revisit the volume Nyström basis.** With the σ_t R=5, r_0/R=0.3
   settings even F.4 gives 0.12% — mostly quadrature limited. The
   0.077% memo number was likely at higher quadrature. Refining
   n_panels or p_order gets F.4 itself below 0.1%.

3. **Accept the Lambert-basis mismatch as a known "working recipe"** —
   it's what F.4 does and it works. The formal inconsistency is a
   documentation issue, not a bug.

## Recommendation

**Abandon the c_in-aware split basis as a production closure.** Keep
the structural derivations as a reference result:

- `diag_cin_aware_basis_derivation.py`: orthonormality + symbolic W
  diagonalization proof — KEEP (educational value).
- `diag_cin_aware_finite_sigma_t.py`: W at finite τ, block structure
  diagnostics — KEEP (textbook-grade result for Sphinx).
- `diag_cin_aware_split_basis_keff.py`: this investigation's empirical
  verdict — KEEP (documents why split basis doesn't help).

Do NOT lift the `NotImplementedError` guards on `boundary="white_rank2"`
with n_bc_modes > 1. The existing Marshak rank-N path is already wired
(via `boundary="white"`) and has the same residual as split basis; users
can already reach rank-N Marshak if they want it.

The investigation has converged: 1D curvilinear rank-N white BC is
FORMALLY at the 1.36% plateau (Marshak) or EMPIRICALLY at 0.12%
(F.4 scalar + Lambert-basis mismatch). These two numbers are the
textbook state of the art per the four-reference synthesis (Issue #119
close-out).

## Files produced this session

- `/workspaces/ORPHEUS/derivations/diagnostics/diag_cin_aware_split_basis_keff.py`
  — full split-basis implementation + k_eff scan. Run: 5-8 min.
- Structural tests (included in diag_cin_aware_split_basis_keff.py):
  all pass to machine precision.

## References

- `.claude/plans/next-session-rank-n-hebert-and-beyond.md` — prior plan.
- `peierls_rank_n_sanchez_closure_failed.md` — Sanchez-McCormick §III.F.1 memo.
- `docs/theory/peierls_unified.rst` §F.4/F.5 — shipped closures.
- Sanchez-McCormick 1982 NSE 80 — original §III.F.1 derivation.

---

## EXPERIMENT 2 (2026-04-21): Enrichment + basis-variant probe

Attacking the 0.99% rank-(1,1,N) plateau along four axes.

### E2.1 — F.4 quadrature floor

F.4's 0.122% IS QUADRATURE ERROR. Refining `n_ang` from 32→64 at
`(n_panels=2, p_order=4)` drops err to 0.003%. At `(4,8,96)` err = 0.0025%.
Non-monotone in refinement — radial/angular errors partially cancel at
baseline. **The "0.077% reported by Issue #119 close-out" is a specific
quadrature coincidence, not a structural residual.** The true F.4
structural floor is < 0.01%. Any future closure claim must match
quadrature.

### E2.2 — Plateau is structural at 2×quadrature

rank-(1,1,N) at (n_panels, p_order, n_ang) = (4, 8, 64):
- rank-(1,1,1) = 1.372%
- rank-(1,1,N≥2) = 1.081% (plateau, up to N_i=8)

At baseline (2,4,32):
- rank-(1,1,1) = 1.290%
- rank-(1,1,N≥2) = 0.99%

Refining quadrature makes the plateau WORSE (1.08% > 0.99%). Confirms
the plateau is structural, not quadrature-limited. Direction is
orthogonal to F.4's (quadrature-limited).

### E2.3 — Residual lives in inner mode-1, outer mode-0

Self-consistent inner-surface mode energies in rank-(1,1,8):
- mode 0: 89.76%
- mode 1: 10.15%
- modes 2-7: < 0.04% each

Residual (rank-(1,1,1) − rank-(1,1,8)) on inner, projected onto half-range
Legendre: dominated by mode-1 (coeff −0.206). Once rank-(1,1,2) captures
mode-1, nothing further to gain. BUT k_eff still at 0.99% err. So the
rank-(1,1,2) closure **has both significant inner modes resolved** yet
still fails at 1%. The 1% must come from the METRIC, not the basis
dimension.

### E2.4 — Lambert P/G breaks the split basis (RH4 refuted)

Replace P_esc/G_bc with Lambert-convention (no µ_exit weight on outgoing)
while keeping Marshak W:

rank-(1,1,1) = **32.99% err** (catastrophic vs F.4's 0.12%).
rank-(3,3,3) = 33.04% (no recovery).
Cross σ_t·R (rank-(1,1,2)): 737%, 73%, 33%, 21%, 13% (σ_t·R=1,2.5,5,10,20).

F.4's Lambert trick is N=1-specific. In the split basis, the Marshak W
with Lambert P/G introduces uncompensated scale factors on the outer-to-
inner coupling that the closure cannot self-correct.

### E2.6 — Jacobi-c² inner basis: POINT win, NOT universal

Swap inner from Legendre (c-weighted) to Jacobi (c²-weighted). All else
identical:

**σ_t·R=5, r_0/R=0.3 (the standard test point)**:
| quadrature | F.4     | Legendre-inner | Jacobi-c² |
|------------|---------|----------------|-----------|
| (2,4,32)   | 0.122%  | 1.29%          | 0.072%    |
| (4,8,64)   | 0.058%  | 1.37%          | 0.004%    |
| (4,8,96)   | 0.025%  | —              | —         |

**Cross σ_t·R (r_0/R=0.3, base quadrature, rank-(1,1,1)):**
| σ_t·R | F.4     | Legendre | Jacobi-c² | verdict  |
|-------|---------|----------|-----------|----------|
| 1.0   | 2.36%   | 85.6%    | **247%**  | CATASTRO |
| 2.5   | 0.35%   | 3.86%    | **9.4%**  | WORSE    |
| 5.0   | 0.12%   | 1.29%    | **0.07%** | IMPROVED |
| 10.0  | 0.25%   | 0.93%    | 0.27%     | similar  |
| 20.0  | 0.36%   | 0.43%    | **0.08%** | IMPROVED |

Conclusion: Jacobi-c² is NOT a universal replacement. It's tuned to the
high-σ_t regime where grazing-rays are heavily attenuated. In thin
regimes it diverges violently. Cannot ship.

BUT: the 0.004% result at σ_t·R=5 with rich quadrature PROVES the 0.99%
Legendre plateau is NOT a fundamental information-content barrier. A
better basis exists. The direction: find basis that adapts to σ_t·R
AND preserves reciprocity, orthonormality.

### W structure with Jacobi-c²

At σ_t=0, W_si[0,0] = √6 / (2ρ) ≈ 4.082 — NOT diagonal (mixes units).
Reciprocity W_si = W_is preserved (4.082 each). The off-diagonal structure
reflects the inner-product mismatch (Legendre outer uses c-weight; Jacobi
inner uses c²-weight). This is a stylistic inconsistency that
nevertheless gives better k_eff at σ_t·R=5 — another unprincipled trick
analogous to F.4's Lambert/Marshak mismatch.

### Lessons for future sessions

L5 (from main log): F.4's 0.122% is quadrature, not structural. Baseline
is <0.01%.
L6 (from main log): Lambert P/G is N=1-specific. Don't try in higher-N.
L7 (from main log): Inner-surface METRIC (weight in inner product) is the
knob. Jacobi-c² works in one regime, Legendre-c works in another.
Adaptive basis is the frontier.

### Next-session recommendations

1. (HIGH PAYOFF) Derive the correct inner-surface metric from an
   asymptotic analysis of the self-consistent ψ^+_inner at large σ_t·R.
   If ψ^+_inner(c) → P_2(c) in some normalization, we need a basis that's
   orthonormal against that kernel. See Direction I in main log.

2. (MEDIUM) Investigate WHY Jacobi-c² works at σ_t·R=5 specifically.
   Possibly the c² weighting compensates for the (1/ρ) factor in the
   steep-to-inner transmission. A careful dimensional analysis might
   reveal a principled weight.

3. (MEDIUM) Try other `α` values in Jacobi (α=0.5, 1.5, 2) at σ_t·R=5 and
   σ_t·R=10 to find the optimal weight and see if it shifts monotonically
   with optical thickness.

4. (LOW) Direction G — non-white BC (anisotropic albedo). Break the
   rank-1 BC bottleneck L1 by using a non-physical BC that populates more
   outer modes. Information-theoretic exercise only; not production-relevant.

---

## EXPERIMENT 4 + 5 (2026-04-22): Regime-switched + rank-(1,1,2) 2D

### E4 — Regime-switched closure (Direction M ACTIVE-SHIP)

Design: σ_t·R ≤ 3 → F.4, σ_t·R ≥ 5 → split-basis rank-(1,1,1)
with SCALE calibrated either by formula `(1+6ρ)/(3ρ)` or by bounded
1D Brent on [1.0, 2.8]; linear k_eff blend in 3 < σ_t·R < 5.

**E4.1 main scan (BASE quadrature)**:
- RS_brent: **12/12 wins vs F.4** at σ_t·R ≥ 5, ratios 87k-2.6M×
- RS_form: 8/12 wins; LOSES at σ_t·R=5 (2-5× worse than F.4)
- Thin regime (σ_t·R ≤ 2.5): safely reduces to F.4 by design

**CAVEAT**: at BASE quad, both rank-(1,1,1)+Brent and F.4 sit on their
quadrature floors. "0.0000%" values are noise. RS_brent's structural
advantage over F.4-at-RICH pending E4.2 validation.

### E5 — rank-(1,1,2) 2D adaptive scales

Nelder-Mead 2D optim of (α_0, α_1) where basis = [α_0, α_1·(2c-1)].
13/14 points before timeout. α_1 UNIVERSALLY ≈ 1.01-1.08 (the natural
Legendre normalization). 2D optim gives SAME err as 1D Brent on α_0
at all tested points.

**Verdict**: rank-(1,1,2) 2D adaptive does NOT beat rank-(1,1,1) 1D
Brent. The scale-gauge DOF lives in mode-0 ONLY — higher modes don't
have independent tuning freedom that matters. This is consistent with
E2.3 (mode-1 carries only 10% of inner-flux energy) and L4 (rank-(1,1,N)
plateau at 0.99% for N ≥ 2).

### New lessons

**L11 (ship-track)**: Regime-switched(brent) is a universal closure
at BASE quad. Ready for production-grade pytests after RICH-quad validation.

**L12**: Scale-gauge DOF (L8) is rank-(1,1,1)-specific. α_1 ≈ 1
optimal uniformly → no 2D gauge freedom to exploit.

**L13**: RS_brent at BASE quad is quadrature-limited (≤ 1e-6 err),
NOT structural. Must validate at RICH quad before shipping claim.

### Files produced in E4+E5

- `/workspaces/ORPHEUS/derivations/diagnostics/diag_cin_split_regime_switched.py`
  — Regime-switched scan; both formula and brent variants; 32-point scan.
  Pytest tests included (`@pytest.mark.slow`).
- `/workspaces/ORPHEUS/derivations/diagnostics/diag_cin_split_rank112_adaptive.py`
  — rank-(1,1,2) 2D Nelder-Mead optim; 14-point scan.

### E4.2 RICH validation — FALSIFIED

Transferred BASE-optimized scales to RICH quadrature (4, 8, 64) and
compared F.4 vs split-rank-(1,1,1) at those same scales:

| σ_t·R | ρ   | scale   | F.4 RICH | split RICH | F.4 wins by |
|-------|-----|---------|----------|------------|-------------|
| 5.0   | 0.3 | 1.7184  | 0.058%   | 0.076%     | 1.3×        |
| 10.0  | 0.3 | 1.8066  | 0.003%   | 0.041%     | 13×         |
| 20.0  | 0.3 | 1.7087  | 0.006%   | 0.016%     | 2.7×        |
| 50.0  | 0.3 | 1.7783  | 0.017%   | 1.507%     | 88×         |
| 20.0  | 0.5 | 1.6165  | 0.005%   | 0.060%     | 11×         |
| 10.0  | 0.5 | 1.6622  | 0.017%   | 0.039%     | 2.3×        |

**6/6 F.4 wins at RICH quad.** The BASE-quad "wins" (87k-2.6M× ratios)
were quadrature-error cancellation artifacts, not structural.

Companion RICH scale scan at σ_t·R=10, ρ=0.3 (full bracket [1.2, 2.3]):

| scale | err @ RICH |
|-------|------------|
| 1.2   | 1.259%     |
| 1.4   | 1.042%     |
| 1.6   | 0.690%     |
| **1.8** | **0.069%** (MINIMUM) |
| 2.0   | 1.127%     |
| 2.2   | 3.484%     |

**RICH-optimum scale 1.8 matches BASE-optimum 1.8066 to 3 digits.**
So scale IS physics-defined (L8 gauge DOF is real). But at the
RICH-optimum, err = 0.069% — **21× WORSE than F.4 (0.003%)**.

**L14 (FINAL)**: rank-(1,1,1) split has a STRUCTURAL floor ~0.07%
at σ_t·R=10, ρ=0.3 RICH. F.4 at RICH gives 0.003% — structurally
better. The gauge DOF from L8 is real and quadrature-invariant at
its optimum, BUT the truncation at N_i=1 inner mode + Marshak-
consistent basis is a stricter structural bound than F.4's Lambert/
Marshak mismatch. E4's BASE-quad "wins" (87k× ratios) were
quadrature noise at BOTH closures' floors.

### Why F.4 is structurally better than rank-(1,1,1) split

F.4 uses Lambert P/G + Marshak W (basis mismatch). This implicitly
accesses a 1-dimensional angular subspace that formally-consistent
rank-N Marshak/split bases cannot reproduce (per L6, Lambert-ification
of the split basis is catastrophic: 737% err at thin τ).

F.4 is on a structurally special 1-parameter family that rank-N bases
cannot access within a single basis convention. Breaking below F.4
would require a fundamentally different approach — NOT more modes,
NOT scale calibration, but a principled reproduction of F.4's
asymmetric basis weighting in a rank-N setting.

### FINAL STATUS (2026-04-22)

**DIRECTION M REJECTED.** Split-basis rank-(1,1,1)+Brent-calibrated-scale
is NOT a shippable universal closure. At matched quadrature, F.4 remains
strictly better.

### Recommendation

1. **Close Issue #120** with verdict "empirical falsification at matched
   quadrature". The c_in-aware split basis is structurally sound but
   empirically dominated by F.4 at every thick-τ point tested at RICH.
2. **Do NOT lift NotImplementedError guards** on
   `boundary="white_rank2"` or related. F.4 is the production path.
3. **If pursuing further**: rank-(1,1,1)+RICH-quad-Brent has NOT been
   tested. Very expensive (~15-30 min per point). Unlikely to yield
   a universal formula per L14. The scale scan shows monotonic decrease
   up to scale=1.5 with no minimum — suggests RICH-optimum scale is
   either very large or the regime is entirely dominated by F.4.

### Residual question for future session

Does rank-(1,1,1) at RICH-quadrature-optimized scale beat F.4 at RICH?
Scale scan at σ_t·R=10, ρ=0.3, RICH shows monotonically decreasing
err from scale=1.2 (1.26% err) to 1.5 (0.89% err). Full scan to
scale=3.0 needed to confirm no minimum exists in the accessible range.
If no minimum, L14 proven conclusively: scale is quadrature-defined,
not geometry-defined.
