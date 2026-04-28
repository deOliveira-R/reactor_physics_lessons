---
name: Phase 5+ Front A — Sanchez↔ORPHEUS K_bc Jacobian conversion FALSIFIED
description: Empirical rank-1 cross-check at three τ_R values shows K_bc^Hebert / K_bc^Phase5 is NOT separable (σ_2/σ_1 ≈ 0.4 universally). Phase 5 kernel additionally fails to converge with quadrature (1/µ² non-integrable surface-diagonal singularity diverges as n_quad grows: K[surf,surf] = 1.24 → 2.33 over n_quad ∈ {16, 512}). Front A is dead — no scalar Jacobian conversion exists.
type: project
---

# Phase 5+ Front A — Jacobian conversion FALSIFIED

Diagnostic shipped at `derivations/diagnostics/diag_phase5_jacobian_a01_rank1_crosscheck.py`.

## Verdict

**FAIL**. The hypothesised Jacobian conversion `K_bc^Hebert[i,j] = α(r_i, r_j) · K_bc^Phase5[i,j]` is NOT separable in `(i, j)`, and the Phase 5 kernel additionally fails to converge in `n_quad`. Front A is dead. The parent must dispatch Fronts B (different operator-level interpretation, possibly involving a singularity-subtraction reformulation) or C (abandon Phase 5 production wiring).

## Empirical evidence

### 1. Ratio matrix is NOT rank-1 separable

At three (R, σ_t) configurations on a 4-panel composite GL mesh (8 r_nodes), the SVD of the ratio `R[i,j] = K_bc^Hebert / K_bc^Phase5` gives:

| Config        | τ_R  | σ_2/σ_1 (ratio matrix) | ratio max/min |
|---------------|------|------------------------|---------------|
| R=5, σ=0.5    | 2.5  | 0.447                  | 1.6 × 10⁴     |
| R=5, σ=1.0    | 5.0  | 0.417                  | 1.3 × 10⁴     |
| R=5, σ=2.0    | 10.0 | 0.377                  | 9.3 × 10³     |
| R=2, σ=2.0    | 4.0  | 0.450                  | similar       |

For a separable conversion, σ_2/σ_1 should be ≤ 1e-3. Empirical 0.4 is FOUR orders of magnitude too large. The "best rank-1 approximant" α_recv(r_i)·α_src(r_j) captures only ~80% of the variance; the residual matrix has structure that cannot be absorbed by per-row / per-column scalings.

### 2. Phase 5 kernel is NOT rank-1 by itself

K_phase5 SVD at R=5, σ=1.0:
- σ_1 = 1.90, σ_2 = 0.13, σ_3 = 0.018 (rank-1 captures 99.5% energy, rank-2 = 99.99%, etc.)

By contrast K_bc^Hebert is rank-1 BY CONSTRUCTION (residual σ_2/σ_1 = 4 × 10⁻¹⁷). The two operators have fundamentally different rank structure. **No multiplicative diagonal-similarity transform `D_recv · K_p5 · D_src` produces a rank-1 matrix from a rank-(>1) matrix.** This is the structural blocker.

### 3. Phase 5 fails uniform-flux conservation

For a homogeneous c=1 sphere, `K_full @ 1 = Σ_t = 1` (uniform flux is a fixed point of the integral operator, up to leakage). At R=5, σ_t=1:

```
K_full_hebert @ 1 = [0.99994, 0.99993, 0.99989, 0.99983, 0.99977, 0.99956, 0.99884, 0.99691]
K_phase5    @ 1 = [0.0125,  0.0144,  0.0224,  0.0367,  0.0500,  0.0964,  0.305,   2.033 ]
```

K_phase5 row sums vary by 2 orders of magnitude — and in particular blow up at the surface row. K_full_hebert is uniform to <0.4%. No constant Jacobian fixes this.

### 4. Phase 5 violates spherical reciprocity

For a true sphere transport Green's function: `K(r_i, r_j) · r_j² = K(r_j, r_i) · r_i²`. K_phase5 fails by factors of 5–20:

```
i=7, j=3: K[i,j]·r_j² = 2.06e-2,  K[j,i]·r_i² = 4.13e-1   (ratio 0.05)
i=6, j=3: K[i,j]·r_j² = 1.36e-2,  K[j,i]·r_i² = 1.56e-1   (ratio 0.087)
```

This is a structural violation, not a normalisation issue. A simple Jacobian rescaling cannot make a non-reciprocal operator reciprocal; it would require an operator transformation (e.g., adjoint averaging).

### 5. Phase 5 fails quadrature convergence (1/µ² singularity divergence)

Diagonal entry `K[7,7]` at the surface (i=j=7, r_node ≈ 4.83 ≈ R=5) as n_quad varies:

| n_quad |  K[7,7]   |  max_K    |
|--------|-----------|-----------|
| 16     | 1.24      | 1.24      |
| 32     | 1.46      | 1.46      |
| 64     | 1.67      | 1.67      |
| 128    | 1.89      | 1.89      |
| 256    | 2.11      | 2.11      |
| 512    | 2.33      | 2.33      |

K[7,7] grows linearly with n_quad — confirming the **`1/µ²` non-integrable surface-diagonal singularity** documented by SymPy V4 in `derivations/peierls_specular_continuous_mu.py`. The integral does NOT exist as a Lebesgue integral at the surface diagonal; finite-N GL is sampling a divergent integrand. ‖K_256 − K_128‖_rel ≈ 10%, ‖K_128 − K_64‖_rel ≈ 12%. **No Cauchy convergence.**

This blocks production wiring even before the Jacobian-conversion question can be settled: the kernel itself is ill-defined as currently implemented.

### 6. Smoke test on solve_peierls_1g

Sphere R=5, σ_t=1, c=1 (Σ_s=0.7, νΣ_f=0.3) → expected k_eff = 1.0 exactly.

| K choice                              | k_eff   |
|---------------------------------------|---------|
| `closure="white_hebert"` (reference)  | 0.9958  |
| K_vol + K_phase5 (raw, no Jacobian)   | 0.7995  |
| K_vol + K_phase5 · (4π r² w)          | 0.7510  |
| K_phase5 · (4π r² w) ALONE            | 0.0796  |
| K_vol + K_phase5 · σ_t                | 0.7995  |
| K_vol + K_phase5 · r_wts              | 8.4884  |

None of the simple Jacobian candidates produces k_eff in the right ballpark. There is no scalar multiplicative correction; the answer is OPERATOR-level different.

## Why Front A failed (operator-level interpretation)

1. **Sanchez `g_h(ρ' → ρ)` is a Green's function for `J_R(ρ) = ρ · I(ρ)`**, not for the 3-D scalar flux φ (Sanchez Eq. 5 reduces the 3-D sphere problem to a 1-D slab via this transformation). The integrand cosh(ρµ)·cosh(ρ'µ_*) is the slab-like even-extension of `J_R`. A direct Nyström discretisation of `g_h` does NOT discretise the same integral operator that ORPHEUS's `K_full` discretises. The two live in different function spaces (J_R vs φ).

2. **The `1/µ²` non-integrable surface-diagonal singularity** means the kernel `g_h(ρ' → ρ)` is NOT a kernel-class function. It exists only as a distributional kernel; integrals against bounded test functions converge but pointwise sampling at quadrature nodes does not. ORPHEUS's Nyström discretisation requires kernel evaluations at nodes — a hard incompatibility.

3. **Hebert's `(1-P_ss)⁻¹` rank-1 outer product is a model reduction**, not the same operator as Sanchez's full Green's function. They differ in physical content: the Hebert form sums Mark-closure single-bounce contributions geometrically; Sanchez closes the BC at the kernel level. They CAN agree on `k_eff` (Hebert hits 0.05% on 1G/1R sphere) without their kernels being similar.

## Recommendations

### What NOT to do
- **Do not** ship `closure="specular_continuous_mu"` as a wrapper around the current `compute_K_bc_specular_continuous_mu_sphere`. The kernel is ill-defined (point 5 above) and structurally incompatible with the Hebert reference (points 1–4).
- **Do not** search for a more complex "non-separable" Jacobian. The reciprocity violation (point 4) and rank mismatch (point 2) cannot be repaired by similarity transforms.

### Front B candidates (operator-level reformulation)

If Phase 5 is still desired, ORPHEUS would need:

- **B1 — Singularity subtraction**. Decompose `g_h(ρ', ρ) = g_h^smooth(ρ', ρ) + (1/µ²)·δ_surface · g_h^analytical`. Treat the singular part analytically (closed-form) and Nyström-discretise only the smooth remainder. This is the analog of how `build_volume_kernel` handles the `E_1(τ)` log singularity in the vacuum kernel.
- **B2 — Reformulate the integral equation in J_R variable**. Solve `J_R(ρ) = ∫ g_h(ρ', ρ) · F(ρ') dρ' + G(ρ)` directly in Sanchez's J_R framework (no φ ↔ J_R conversion at the operator level). Build a separate Phase 5 solver pipeline that discretises Sanchez's reduced equation, not the 3-D Peierls equation. Convert J_R → φ as a post-processing step. Higher engineering cost; cleaner mathematical correspondence.
- **B3 — Wiener-Hopf factorisation**. Sanchez Eq. (A6) has the structure of a Volterra-Hammerstein integral equation in ρ; the spectral multi-bounce factor T(µ) is the Mellin-transform of the half-line kernel. A Wiener-Hopf or Carleman factorisation could absorb the singularity into a closed-form residue.

### Front C — Abandon Phase 5 production wiring

If users are satisfied with `closure="specular_multibounce"` at N ∈ {1,2,3} (sphere/cyl thin cells) plus N=any for slab, Phase 5 is research-grade only. The shipped `compute_K_bc_specular_continuous_mu_sphere` remains useful as a literature cross-check for ω₁=0 cases against Pomraning-Siewert 1982. Document in Sphinx that Front A was tested and falsified; close out Phase 5+ as research curiosity.

## Files

- Diagnostic: `/workspaces/ORPHEUS/derivations/diagnostics/diag_phase5_jacobian_a01_rank1_crosscheck.py` (3 parametrised tests, all PASS as informational — no assertions because the conclusion was empirical exploration)
- Reference impl: `/workspaces/ORPHEUS/orpheus/derivations/peierls_geometry.py:2440` (`compute_K_bc_specular_continuous_mu_sphere`)
- SymPy verification: `/workspaces/ORPHEUS/derivations/peierls_specular_continuous_mu.py` V4 (1/µ² singularity)
- Phase 5a closeout: `/workspaces/ORPHEUS/.claude/agent-memory/numerics-investigator/specular_continuous_mu_phase5a_closeout.md`
- Sanchez 1986 lit memo: `/workspaces/ORPHEUS/.claude/agent-memory/literature-researcher/phase5_sanchez_1986_sphere_specular.md`

## Lessons learned

- **A scalar-Jacobian hypothesis can be FALSIFIED in a single SVD**. The σ_2/σ_1 ratio of `K_orpheus / K_reference` IS the test for "is there a multiplicative correction?". 0.4 means no.
- **Reciprocity is a robust diagnostic for transport Green's functions**. Sphere reciprocity `K(i,j)·r_j² = K(j,i)·r_i²` is satisfied by K_vol but violated by K_phase5 by 5–20×; this confirms the kernel is in a different function space.
- **Quadrature divergence != quadrature underconvergence**. K[surf,surf] growing linearly with n_quad is the signature of a NON-INTEGRABLE singularity, not a slowly-convergent integral. SymPy V4 had already flagged this; the empirical scan confirms it. Phase 5 needs singularity subtraction before any production use.
- **Operator-level questions ("are these the same operator?") cannot be settled by `K @ 1` alone**. Need rank structure + reciprocity + uniform-flux response + asymptotic behaviour. All four agree here.

## Promotion recommendation

The diagnostic `diag_phase5_jacobian_a01_rank1_crosscheck.py` can stay as an informational test — if a future Phase 5+ effort proposes a different Jacobian and claims to fix this, the same SVD should immediately fail (or pass) the new approach. **Do NOT promote** the surface-diagonal divergence sub-script to permanent tests because there's no production target for it; document it in Sphinx (`§peierls-phase5-front-a-falsified`) as a permanent record of why Phase 5+ was bypassed.
