---
name: Specular multi-bounce overshoot root cause
description: K_bc^mb = G·R·(I-TR)^-1·P at N≥4 overshoots k_inf. Root cause = continuous limit (I-TR)^-1 is multiplication by 1/(1-e^(-σ·2Rµ)) which DIVERGES at µ=0; matrix-Galerkin form has unbounded operator norm as N→∞. Both bare AND multi-bounce specular diverge at high N.
type: project
---

# Specular multi-bounce overshoot root cause (2026-04-27)

Branch `feature/peierls-specular-bc`, commit `9178cc6`. User flagged
the overshoot as suspicious. 13 diagnostics confirm:

## TL;DR

The user's intuition ("multi-bounce should approach k_inf, not overshoot")
is RIGHT. The overshoot is a **fundamental divergence** of the rank-N
matrix construction, not a basis-condition artifact.

Two compounding failures:

1. **(I-T·R)⁻¹ is the Galerkin projection of an unbounded operator**.
   In continuous-µ space, T·R = mult by `e^(-σ·2Rµ)` (spectrum (e^(-2σR), 1])
   and `(I-T·R)⁻¹` = mult by `1/(1-e^(-σ·2Rµ))` which **DIVERGES at µ=0**
   (grazing modes). Matrix-Galerkin projection of this divergent operator
   has operator norm growing without bound as N → ∞:

   At thin τ_R=2.5: ‖(I-TR)⁻¹‖₂ = 1.08 (N=1) → 53.9 (N=25).
   At very-thin τ_R=1.0: 1.42 (N=1) → 86.4 (N=20).

2. **Bare K_bc = G·R·P also diverges at high N**. R = (1/2)M⁻¹ is
   ill-conditioned (cond(M) grows polynomially); R amplifies noise as
   ~60x at N=8, ~365x at N=12. Bare specular k_eff at thin τ_R=2.5:

   | N | bare specular |
   | --- | --- |
   | 1 | -8.42 % |
   | 4 | -5.61 % |
   | 8 | -3.22 % |
   | 10 | -0.79 % (LOOKS converging) |
   | 12 | +1.90 % (overshoot starts) |
   | 20 | +9.68 % |
   | 25 | +145.80 % (catastrophic) |

   So bare specular itself is **NOT a converging method** as N → ∞. The
   prior "thick sweet spot at N=4" is a coincidence in the convergence
   window.

## Diagnostic chain (numbered by file)

`derivations/diagnostics/diag_specular_overshoot_*.py`:

| # | Diagnostic | Conclusion |
|---|-----------|-----------|
| 01 | Term-by-term geometric series | Series converges per N to a fixed limit, but limit overshoots k_inf |
| 02 | T·R spectrum vs continuous T_op | Eigenvalues approach 1 from below as N→∞; matrix-form ρ(T·R) is correct in basis-coef sense |
| 03 | M⁻¹·T·R·M = K_op (basis-coef form) | T·R in J⁺ space = M·K_op·M⁻¹; operator interpretation correct |
| 04 | T quadrature convergence | T fully converged at n_quad=16 (not a quadrature artifact) |
| 05 | MC homogeneous specular sphere | k_eff = k_inf exactly; <path> matches inf medium to 0.01 % at thin and very-thin |
| 07 | Per-bounce flux contribution magnitudes | <φ_bounce_k> decays cleanly per N; eigenvalue distortion is from SHAPE not magnitude |
| 08 | Eigenvector shape vs N | Eigvec progressively peaks at boundary cell as N grows; max/min ratio 1.002 (N=1) → 1.180 (N=8) |
| 09 | Bare vs MB element-wise | MB amplifies K_bc[r=R-ε,:] by 6.8 % vs interior ~0 % — non-uniform amplification |
| 10 | Scalar Hebert factor on bare | Scalar `1/(1-P_ss)` × bare K_bc ALSO diverges at high N (and worse than matrix MB) |
| 11 | Bare specular convergence as N→∞ | Bare diverges: -8.4 % (N=1) → -0.79 % (N=10) → +145.8 % (N=25) |
| 12 | R_spec conditioning | cond(M) grows polynomially; R_spec amplifies noise 365× at N=12 |
| 13 | ‖(I-TR)⁻¹‖₂ vs N | Confirms unbounded growth — fundamental divergence |

## Why this is fundamental (not fixable in matrix form)

The continuous functional form of multi-bounce specular sphere is:

```
K_bc^mb(r_i, r_j) = ∫_0^1 G_kernel(r_i, µ) · F_kernel(r_j, µ) ·
                          [µ / (1 - e^(-σ·2Rµ))] · 2 dµ
```

The integrand has the µ weight which CANCELS the singularity at µ=0:
`µ / (1 - e^(-2σRµ)) → 1/(2σR)` as µ→0, finite. So the **integral** is
well-defined and bounded. K_bc^mb as a continuous-µ integral converges
to the right answer (= k_inf for homogeneous sphere by MC verification).

But the **matrix-Galerkin projection** doesn't see this cancellation —
it splits the µ-weighted measure across (a) the partial-current operator
P (which contains one µ from the outgoing partial-current measure) and
(b) the response operator G (which contains another µ from the inward
partial-current measure). The **per-bounce operator** T·R inherits ONE
µ weight from T's `2 ∫ µ P̃_m P̃_n e^(-τ) dµ` — which is OK at the FIRST
bounce. But the geometric series `(I - TR)⁻¹` factor is the µ-INDEPENDENT
inverse of (I-mult by e^(-τ)) which is the divergent multiplication
operator.

In short: **the µ weight needed to tame the grazing singularity is
distributed throughout the construction (P, T, G all carry µ-factors)
but the matrix inverse `(I-TR)⁻¹` doesn't preserve the cancellation**.
As N → ∞ the basis resolves grazing modes more finely, exposing the
divergence.

## What WOULD fix it (Phase 4 work)

Re-derive K_bc^mb in **continuous-µ form** with explicit µ-resolved
kernels:

- `F_out(r_j, µ)` — source-to-boundary partial-current density per
  exit-angle µ (replaces P)
- `G_in(r_i, µ)` — boundary-to-interior flux contribution per inward
  angle µ (replaces G)
- Multi-bounce factor `f_mb(µ) = µ / (1 - e^(-σ·2Rµ))` — DIRECT
  multiplication, not matrix inverse

Then K_bc^mb(r_i, r_j) = `2 ∫_0^1 G_in(r_i, µ) · F_out(r_j, µ) · f_mb(µ) dµ`,
with quadrature directly in µ.

This bypasses the rank-N basis entirely and avoids the divergence.
But it requires re-architecting the Peierls geometry primitives (P, G)
to expose their µ-resolved form. The current rank-N machinery integrates
these implicitly; extracting the µ-resolved form is a substantial
refactor.

## Practical recommendation

The shipped `closure="specular_multibounce"` is mathematically OK at
**N ∈ {1, 2, 3}** for ANY thin sphere. The N≥5 UserWarning is
**too lenient** — overshoot already starts at N=4 (+0.43 % at thin
τ_R=2.5, +0.45 % at very-thin τ_R=1.0).

Recommend:

1. **Tighten the UserWarning to N≥4** (was N≥5).
2. **Document the structural divergence** in the docstring of
   `compute_T_specular_sphere` — the ‖(I-TR)⁻¹‖ growth IS the
   inevitable consequence of the basis projection, not just numerical
   pathology.
3. **For thin cells at N≥4**, document that the only converging
   specular closure requires a continuous-µ K_bc (Phase 4 follow-up).
4. **Open a research GitHub issue** for "continuous-µ multi-bounce
   specular K_bc" with this analysis as starting evidence.

## What NOT to try (refuted hypotheses)

- **Scalar `1/(1-P_ss)` on bare K_bc at high N**: works at N=1
  algebraically, but diverges WORSE than matrix MB at N≥3 (29.9 % at
  very-thin N=10 vs 17.9 % for matrix MB at same).
- **Higher T quadrature**: T already converged at n_quad=16; bumping to
  2048 doesn't change anything (BASE diag 04).
- **Different basis (Jacobi, Chandrasekhar H-functions, etc.)**: any
  basis that preserves the partial-current physics will inherit the
  same `1/(1-e^(-τ(µ)))` divergence at µ=0. The basis is not the
  obstruction; the matrix-inverse formulation is.

## Files (all in derivations/diagnostics/)

- `diag_specular_overshoot_01_geometric_series.py` — term-by-term partial sums
- `diag_specular_overshoot_02_TR_spectrum.py` — TR eigvals vs N
- `diag_specular_overshoot_03_basis_coef_op.py` — M⁻¹TM = K_op identity
- `diag_specular_overshoot_04_n_quad_T.py` — T quadrature convergence
- `diag_specular_overshoot_05_mc_multibounce.py` — MC ground truth (k_eff = k_inf)
- `diag_specular_overshoot_07_term_magnitudes.py` — per-bounce flux magnitudes
- `diag_specular_overshoot_08_eigvec.py` — eigvec shape vs N
- `diag_specular_overshoot_09_compare_bare_vs_mb.py` — element-wise K_bc comparison
- `diag_specular_overshoot_10_scalar_hebert_factor.py` — scalar Hebert × bare comparison
- `diag_specular_overshoot_11_continuous_kernel.py` — bare specular divergence at N≥12
- `diag_specular_overshoot_12_R_conditioning.py` — R_spec conditioning blow-up
- `diag_specular_overshoot_13_resolvent_norm.py` — ‖(I-TR)⁻¹‖₂ growth proof

## Promotion candidates

- **diag_05 (MC ground truth)** → `tests/derivations/test_peierls_specular_bc.py`
  as `test_mc_specular_sphere_recovers_kinf` (slow regression, gates the
  fundamental "specular sphere = k_inf" physics).
- **diag_11 (bare specular convergence)** → `tests/derivations/test_peierls_specular_bc.py`
  as `test_bare_specular_known_divergence_window` (pins the convergence
  window N ∈ [4, 10] for thin sphere; alerts if R_spec conditioning
  changes).
- **diag_13 (resolvent norm)** → `tests/derivations/test_peierls_specular_bc.py`
  as `test_specular_multibounce_resolvent_norm_growth` (regression on
  the documented divergence; if a future fix bounds the norm, this
  test should be updated).
