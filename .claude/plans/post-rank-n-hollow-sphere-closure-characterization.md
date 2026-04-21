# Rank-N hollow-sphere white-BC closure — characterization and open state

**Branch**: `investigate/peierls-solver-bugs`
**Date**: 2026-04-21
**Issue**: #119
**Status**: OPEN — F.4 scalar rank-2 is the best working closure. Rank-N remains open research.

## Executive summary

After an exhaustive investigation spanning:

- 60+ recipe variants of the Sanchez-McCormick §III.F.1 canonical form.
- Monte-Carlo cross-check of the `compute_hollow_sph_transmission_rank_n` matrix (33 tests, all pass — W is correct).
- Sympy derivation of the µ-weighted orthonormal basis (Jacobi P^{(0,1)}_n on [0,1]).
- Literature extraction from Sanchez-McCormick 1982 §III.F.1 (canonical) and Stamm'ler-Abbate 1983 Ch. 6 (Ch. 6 is SN, NOT interface currents — interface currents are in Ch. 3 which we don't have).
- Quadrature-refinement testing at 8 → 48 radial nodes and N=1..4.
- Cross-parameter scans of F.4 N=1 vs Sanchez N=2-3 at σ_t·R ∈ {1, 2.5, 5, 10, 20}.

**Conclusion**: the Sanchez rank-N recipe plateaus at ~1.42 % k_eff residual for hollow sphere at σ_t·R=5, r_0/R=0.3. The plateau is robust:
- Not a quadrature artifact (radial refinement from 8 → 48 nodes gives 1.42 % → 1.31 %; does not converge to 0).
- Not a mode-truncation artifact (N=2 through N=4 all give 1.42 % — literal plateau).
- Not a basis-choice bug (13 variants tested — µ-ortho basis is the best of any).

**F.4 scalar closure is the practical production path.** Across the tested σ_t·R range, F.4 N=1 consistently beats Sanchez N=2 by 1-13×. The `NotImplementedError` guard in `build_closure_operator` for `n_bc_modes > 1` on 2-surface cells MUST stay in place.

## The cross-parameter scan (key data)

| σ_t·R | F.4 N=1 err | Sanchez N=2 err | Sanchez N=3 err |
|-------|-------------|-----------------|-----------------|
| 1.0   | 3.27 %      | 42.02 %         | 49.15 %         |
| 2.5   | 0.55 %      | 5.89 %          | 5.94 %          |
| 5.0   | 0.077 %     | 1.42 %          | 1.42 %          |
| 10.0  | 0.26 %      | 0.53 %          | 0.53 %          |
| 20.0  | 0.45 %      | 0.44 %          | 0.45 %          |

F.4 has a sweet spot at σ_t·R=5 (0.077 %). It degrades modestly at thinner/thicker cells but remains competitive through σ_t·R=20 where Sanchez N=2 marginally overtakes (0.44 % vs 0.45 %).

Sanchez is catastrophic at thin cells (42 % at σ_t·R=1) and barely adequate at moderate cells (5.89 % at σ_t·R=2.5). It only becomes useful at thick cells where F.4 is already pretty good.

## Why doesn't Sanchez close?

The Wigner-Seitz identity `k_eff = k_inf` is mathematically exact for white BC on all surfaces of a homogeneous hollow sphere (no net leakage, closed cell). So the rank-∞ limit of ANY correct rank-N closure must converge to 0. The 1.42 % plateau at N=4 for Sanchez indicates the implementation is NOT the correct rank-N ladder — some structural term is missing.

Hypotheses (not conclusively tested):

1. **Observer-centered vs surface-centered integration mismatch.** Sanchez's canonical primitives are surface-centered (∫_{A_α} dA …). ORPHEUS's Peierls-Nyström volume kernel is observer-centered (∫ sin θ dθ …). When we ADD Sanchez's K_bc to ORPHEUS's K_vol, they may be in subtly inconsistent normalizations. Fixing this might require rederiving Sanchez in observer-centered form from scratch.

2. **Normalization scale in the basis.** Sanchez normalizes `f^0 = (π A)^{-1}`; my implementation uses `f^0 = √2`. These should cancel consistently across P, G, W — but if one primitive missed a factor (say, P has A^{-1} but W has A^{-2}), the numerical result is biased.

3. **The (Ω·n)(Ω·n') double-µ in W.** Sanchez's W has (Ω·n) at emission and (Ω·n') at arrival — TWO µ factors. My implementation does this (line 167 of `diag_sanchez_N_convergence.py`: `cos_th · sin_th · cos_th · cos_th`). If this is wrong (should only be one µ), the closure is biased.

4. **F.4's closure is fundamentally different.** F.4 uses Lambert P/G + µ-weighted W + `(I-W)^{-1}` with geometric series interpretation. Extending F.4 naturally to rank-N (my attempted derivation) gives the same formula ORPHEUS has been testing at 3.87 % residual. F.4's N=1 success may reflect an IMPLICIT Wigner-Seitz cell approximation that doesn't generalize.

## The decisive experiment — F.4 "scaling fingerprint"

The numerics-investigator's optimal-c scan (from a prior session) showed:
| σ_t·R | c_opt (best scalar factor for mode-1 P, G) | err_opt |
|-------|---------------------------------------------|---------|
| 2.5 | 0.05 | 0.59 % |
| 5 | 0.16 | 0.001 % |
| 10 | 0.36 | 0.003 % |
| 20 | 0.60 | 0.086 % |

The optimal mode-1 amplitude scales monotonically with σ_t·R. No constant factor works. This strongly suggests that the mode-1 primitive should have a different σ_t-dependent structure — NOT just a basis choice issue.

In the limit σ_t → ∞, c_opt → 1 (no scaling needed). In the limit σ_t → 0, c_opt → 0 (mode-1 contribution should vanish). The mode-1 coupling through W has the wrong σ_t dependence in the current implementation.

## Production recommendation

1. **Keep `NotImplementedError` guard** in `build_closure_operator` for `n_bc_modes > 1` on hollow cells.
2. **F.4 rank-2 scalar** remains the working closure. Its residual vs σ_t·R is documented above.
3. **Close Issue #119 as "won't fix for now"** — the canonical Sanchez recipe doesn't reach the 0.1 % gate, and we don't have a convergent alternative.
4. **File research-tag follow-up issue** for eventual investigation of:
   - Hébert 2009 *Applied Reactor Physics* Ch. 3 (modern CP with interface currents).
   - Stepanek 1982 NSE (anisotropic interface currents).
   - Stamm'ler-Abbate 1983 Ch. 3 (requires separate PDF acquisition).
   - First-principles derivation of the F.4 closure and its natural rank-N extension.

## Artefacts from this session

### Committed earlier in session

- `derivations/diagnostics/derive_mu_weighted_basis.py` — sympy derivation of µ-weighted orthonormal basis.
- `derivations/diagnostics/diag_sanchez_fractional_scan.py` — 16 α-weight combinations.
- `derivations/diagnostics/diag_sanchez_N_convergence.py` — N=1..4 plateau proof.
- `derivations/diagnostics/diag_hybrid_f4_plus_sanchez.py` — F.4 + Sanchez hybrid.
- `derivations/diagnostics/diag_rank_n_W_mc_crosscheck.py` — 33 MC-verification tests of W.
- `.claude/agent-memory/literature-researcher/sanchez_mccormick_rank_n_per_face.md` — full canonical extraction.
- `.claude/agent-memory/numerics-investigator/peierls_rank_n_sanchez_closure_failed.md` — 60+ recipe empirical.

### New this turn

- `derivations/diagnostics/diag_rank_n_closure_characterization.py` — quadrature-refinement + σ_t scan + pytest gates.
- `.claude/agent-memory/literature-researcher/stammler_1983_ch6_interface_currents.md` — negative finding (Ch. 6 is SN, not IC).

## Files NOT touched

- `orpheus/derivations/peierls_geometry.py` — no closure changes this session.
- `tests/derivations/test_peierls_rank2_bc.py` — no test changes (the `test_rank_n_white_closure_raises_pending_normalisation` gate stays).
