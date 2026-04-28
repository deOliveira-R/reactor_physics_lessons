---
name: Phase 5 retreat — continuous-µ K_bc is hypersingular, not Nyström-discretisable
description: After 3 rounds of investigation, Phase 5 production wiring is ABANDONED. Continuous-µ K_bc kernel is HYPERSINGULAR (Hadamard finite-part); matrix-Galerkin Phase 4 form is the permanent production path. Independently confirmed by 3 numerics-investigator agents.
type: project
---

# Phase 5 retreat — continuous-µ K_bc is hypersingular (2026-04-28)

After Phase 5a + 3 rounds of numerics-investigator dispatches, Phase 5 production wiring is **ABANDONED** with a structural conclusion: the continuous-µ K_bc kernel is hypersingular (Hadamard finite-part / Cauchy principal-value type) and **not Nyström-discretisable in principle**. The matrix-Galerkin Phase 4 form `closure="specular_multibounce"` is the **permanent production path**.

## Round-by-round trail

| Round | Approach | Result | Key finding |
|---|---|---|---|
| 5a | Sanchez 1986 Eq. (A6) reference impl | Failed smoke test | Kernel magnitudes 4 orders off Hebert; two blockers identified (Jacobian + diagonal singularity) |
| R1 Front A | Empirical Jacobian via rank-1 cross-check | FALSIFIED | No separable conversion exists; SVD shows rank-mismatch |
| R1 Front B | Singularity subtraction | PARTIAL | Closed-form leading orders identified; analytical add-back divergent |
| R1 Front C | ORPHEUS-native bypass of Sanchez cosh | FAILED | Same singularity in T(µ); k_eff oscillates with Q |
| R2 PRIMARY | M2 bounce-resolved expansion | FAILED | **Singularity is in F_out·G_in, NOT T(µ)** — at K_max=0 (no MB), diagonal still diverges |
| R2 BACKUP | Symbolic N→∞ matrix-Galerkin limit | **CLOSED FORM** | `f_∞(µ) = (1/2) µ / (1 − e^{−σ·2Rµ})` bounded at µ=0; via Bose-Einstein polylog |
| R3 PRIMARY | Adaptive quadrature near µ_min(r) | FAILED | 6 approaches tried (subtraction, u²-CoV, GK, Galerkin); k_eff -34% to -50% |
| R3 SECONDARY | Galerkin double-integration | FAILED | 2-D `∫∫ L_i L_j K dr dr'` gives same log(Q_µ) divergence node-by-node |

## Structural finding (independently confirmed by 3 agents)

The µ-resolved primitives `F_out(r, µ) · G_in(r, µ)` carry a `1/(cos(ω_i) cos(ω_j))` Jacobian. At the discrete Nyström diagonal `r_i = r_j = r`, both cosines vanish at the SAME `µ_min(r) = √(1 − (r/R)²)`, yielding non-integrable `1/(µ² − µ_min²)` on the visibility cone `µ ∈ [µ_min, 1]`.

**Critical observation** (Round 2 M2 PRIMARY, smoking gun): this singularity persists at K_max=0 (NO multi-bounce — bare specular alone). The Phase 4 matrix-Galerkin form `(I − T·R)^{-1}` ABSORBS this singularity via basis projection (the rank-N shifted-Legendre projection acts as smoothing). Removing the basis exposes the bare singularity.

**Mathematical type**: Hadamard finite-part / Cauchy principal-value. Standard Nyström sampling diverges; standard Galerkin double-integration diverges (R3 SECONDARY confirmed); singularity subtraction's analytical add-back is divergent (R1 Front B). The kernel is genuinely outside the Fredholm-second-kind class that ORPHEUS's existing `build_volume_kernel_adaptive` can handle.

**Phase 4 vs Phase 5 are different operators**: not different discretisations of the same operator. R3 PRIMARY's smoking-gun side-by-side at N=3:
- Phase 4 K_bc (rank-3): surface row dominates, structured
- Per-pair K_bc (Round 3): diagonal dominates, ratio-spread 0.007 to 7.88 vs Phase 4
- Phase 4's `(I − T·R)^{-1}` mode-mixing cannot be replicated by simple integration

## What's KEEPABLE from Phase 5

1. **Closed-form multi-bounce factor** (Round 2 BACKUP):
   ```
   f_∞(µ) = (1/2) · µ / (1 − e^{−σ·2Rµ})
   K_∞^half = (1/(8a²)) [π²/6 − Li_2(e^{-2a}) + 2a ln(1 − e^{-2a})]
   ```
   Bounded at µ=0 (limit = 1/(4a)). The `(1/2)` from `R = (1/2) M^{-1}`; `µ` from µ-weighted basis Gram. Useful for theoretical reference and possible future asymptotic analysis. SymPy bit-exact via Bose-Einstein polylog.

2. **Visibility-cone substitution** (R3 PRIMARY + SECONDARY): `u² = (µ² − µ_min²)/(1 − µ_min²)` gives MACHINE PRECISION (1e-9 at Q=128) off-diagonal Q-convergence for any µ-resolved per-pair integral with a single-endpoint visibility-cone singularity. **Portable technique** — could be promoted into `orpheus.derivations._kernels` as a utility for future visibility-cone work.

3. **Reference implementation** `compute_K_bc_specular_continuous_mu_sphere` (Phase 5a) shipped and SymPy-verified. Useful as a research-grade reference for theoretical comparisons (e.g., cross-checking Pomraning-Siewert 1982 if pulled in the future).

4. **Sanchez 1986 literature memo** (`phase5_sanchez_1986_sphere_specular.md`) — full extraction of Eqs. (A1)–(A7) with notation map.

5. **Latent stability bug fix** at `peierls_geometry.py:2592-2598` (Front B incidental find): the chord-projection `µ_*² = ρ'² − ρ²(1−µ²)` form had catastrophic cancellation; replaced with numerically-stable `(ρ'² − ρ²) + ρ²·µ²` rewrite.

## What's PERMANENTLY DEAD

- Phase 5 production wiring (`closure="specular_continuous_mu"` raises NotImplementedError forever)
- All four singularity-handling approaches (subtraction, u²-CoV, GK adaptive, Gauss-Jacobi)
- Galerkin double-integration over Lagrange basis (R3 SECONDARY)
- Bounce-resolved M2 expansion as a Nyström target (R2 PRIMARY)
- Hadamard finite-part regularisation (gauge-ambiguous; R3 PRIMARY flagged as "rounds 4-5 risk")

## Production verdict

`closure="specular_multibounce"` (Phase 4 matrix-Galerkin form) is the **permanent production path** for multi-bounce specular at all three geometries:
- Sphere/cyl thin cells: N ∈ {1, 2, 3} (UserWarning at N≥4)
- Slab: any N (geometric immunity)

The Phase 4 docstring reference to "Phase 5 — proper fix" is **withdrawn**. There is no proper fix in the continuous-µ discretisation framework — the matrix-Galerkin form is structurally the correct discretisation, and its rank-N gating reflects the kernel's intrinsic difficulty rather than a basis-truncation artefact.

## Files

### Closed out / shipped (Phase 5 retreat)

- This memo: `.claude/agent-memory/numerics-investigator/specular_continuous_mu_phase5_retreat.md`
- Phase 5a baseline: `specular_continuous_mu_phase5a_closeout.md` (the starting point)
- Round 1: `phase5_front_a_jacobian_conversion.md`, `phase5_front_b_singularity_subtraction.md`, `phase5_front_c_orpheus_native.md`
- Round 2: `phase5_round2_m2_bounce_resolved.md`, `phase5_round2_backup_symbolic_limit.md`
- Round 3: `phase5_round3_adaptive_quadrature.md`, `phase5_round3_galerkin_double_integration.md`
- Sphinx: `docs/theory/peierls_unified.rst` §peierls-phase5-retreat (this same content in user-facing form)
- Code: `orpheus/derivations/peierls_geometry.py` — dispatch raises NotImplementedError with retreat message; reference impl preserved
- SymPy: `derivations/peierls_specular_continuous_mu.py` (4/4 PASS — math is right, discretisation is fundamentally not Nyström-compatible)
- Diagnostic record: 14 diagnostics in `derivations/diagnostics/diag_phase5_*.py`

### GitHub

- Issue #133 (Phase 5+ tracking) — closed wontfix with reference to this retreat memo

## Lessons for future investigations

1. **Hypersingular kernels are not failures of method, they're intrinsic to the kernel**. After 3 rounds of trying every standard quadrature trick, the conclusion was structurally clean — this is a Hadamard finite-part kernel. Future investigators should test for the singularity TYPE early via leading-order asymptotic analysis at the diagonal BEFORE attempting production wiring.

2. **The matrix-Galerkin form's mode-mixing is load-bearing**. The intuition that "the matrix form is a basis-truncation artefact and the continuous form is the correct limit" was WRONG. The matrix form is the correct discretisation; the basis projection is performing essential smoothing. Future research-grade reformulations should be cross-checked at low N against the matrix form to detect operator mismatch.

3. **Three independent agents converging on the same diagnosis is a strong signal**. M2 PRIMARY isolated the diagonal singularity; R3 PRIMARY confirmed it via 6 different approaches; R3 SECONDARY confirmed it via Galerkin double-integration. Three orthogonal mathematical attacks producing the same finding is high-confidence structural evidence — time to retreat rather than chase rounds 4-5 into gauge-ambiguous territory.

4. **5-round budget was the right framing**. The user's instruction "after 5 rounds retreat" anchored the decision to stop at the right moment. We retreated at round 3 with strong structural evidence rather than burning rounds 4-5 on Hadamard finite-part / augmented Nyström / operator-norm pre-smoothing approaches that R3 PRIMARY explicitly flagged as gauge-ambiguous and unbounded budget.

5. **Useful side-findings should still be promoted**. The visibility-cone substitution `u² = (µ² − µ_min²)/(1 − µ_min²)` is a clean technique that came out of the failed Phase 5 work. Future µ-resolved per-pair integrals (e.g., for hollow-cell Peierls extensions) can use it directly. Don't let a failed primary mission discard genuinely useful subsidiary findings.
