---
name: Slab specular BC — per-face divisor fix
description: The slab specular plateau (-5%/+7% at rank≥2) was a single-line bug — combined-face divisor (=2) inherited from legacy Mark instead of single-face divisor (=1) for per-face decomposition. Fix shipped 2026-04-27.
type: project
---

# Slab specular BC — root cause was the surface area divisor

Investigation date: 2026-04-27, branch `feature/peierls-specular-bc`.

## TL;DR

The user's reported plateau ("Marshak +6.84% at N=2; (1/2)M^{-1} -5.16%
at N=2") was caused by passing the **combined-face** surface area divisor
(=2 for slab, from `geometry.rank1_surface_divisor(R)`) into the
per-face mode-N specular K_bc construction. The combined-face divisor is
correct for legacy Mark (which uses `compute_P_esc` = ½ E_2_outer + ½
E_2_inner — both faces in one primitive). The per-face decomposition
treats each face as area=1 individually, so divisor=1.

With divisor=1 the per-face block-diagonal R = (1/2) M^{-1} converges to
k_inf monotonically with both N AND mesh refinement.

## Verification results (BASE quad, fuel A 1G, L=5cm, k_inf=1.5)

| N   | k_eff (specular)   | rel %      | Mark legacy at same quad |
| --- | ------------------ | ---------- | ------------------------ |
| 1   | 1.4970816989       | -0.19455 % | 1.4970816989 (bit-exact) |
| 2   | 1.4971975266       | -0.18683 % |                          |
| 3   | 1.4974612035       | -0.16925 % |                          |
| 4   | 1.4977406800       | -0.15062 % |                          |

Monotonically improving in N. Rank-1 bit-exactly matches Mark legacy
(eigenvalue equivalence — see below).

## The "rank-1 bit-exact" subtlety

At rank-1, the K_bc matrices are NOT element-wise equal between Mark
legacy and per-face block-diag specular:

- `K_bc_mark_legacy = K_oo + K_oi + K_io + K_ii` (4 face-pair products)
- `K_bc_specular_blockdiag = K_oo + K_ii` (only same-face products)

But the difference `K_bc_mark - K_bc_spec = (1/2)(G_o-G_i)(P_i-P_o)` is
**anti-symmetric around L/2** for homogeneous slab. The dominant
eigenvector of K is symmetric (uniform-ish for homogeneous slab), so
the anti-symmetric perturbation does not affect the dominant eigenvalue.
Verified numerically to 1e-15 in `diag_slab_specular_08_*.py`.

## Why the divisor matters

Look at `geometry.rank1_surface_divisor(R)` in `peierls_geometry.py:860`:

```python
if self.kind == "slab-polar":
    return 2.0   # "two unit-area faces at x=0 and x=L"
```

The docstring says: "ratio A_j/A_d = w_j/2, divisor 2". A_d = 2 is the
TOTAL surface area (2 faces × 1 unit area each), used by Mark to
normalize the COMBINED-face primitive. For per-face decomposition,
each face has area A_d_face = 1.

The current `_build_full_K_per_group` passes this divisor into both the
sphere/cylinder and slab specular branches uniformly. For sphere/cylinder
this is correct (one surface). For slab, the per-face split must override
to 1.

## Diagnostic cascade (1.5 hours)

1. `diag_slab_specular_01_row_sum.py` — K·1 row sum profile shows
   per-face block-diag candidates undershooting, but reveals that
   ALL my candidates (with divisor=2 inherited bug) deviate the same way.
2. `diag_slab_specular_02_face_coupling.py` — proves
   `K_bc_legacy = K_oo + K_oi + K_io + K_ii` (combined-face decomposes
   into 4 components in per-face basis).
3. `diag_slab_specular_03_rank1_check.py` — Mark legacy doesn't actually
   give k_eff = k_inf for slab; it converges to ~-0.19% at moderate
   quad. So Σ_t·K·1 = 1 was the wrong contract anyway.
4. `diag_slab_specular_04_eigvec_compare.py` — sweep over R candidates
   {block-diag Marshak, block-diag (1/2)M^{-1}, ones(N,N) per-face,
   full ones(2N,2N), Mark-like cross-coupling}. Different R choices
   give wildly different k_eff but ALL block-diag variants plateau at
   ~-8%. The `Mark-like (I + I cross)` variant gives +1% (close to but
   not equal to Mark).
5. `diag_slab_specular_05_convergence.py` — confirms block-diag
   plateaus as N → ∞. Doesn't converge.
6. `diag_slab_specular_06_divisor_check.py` — **THE FIX**. Sweep over
   divisor ∈ {2.0, 1.0, 0.5}. Divisor=1 makes everything work:
   rank-1 (1/2)M^{-1} matches Mark legacy bit-exactly; rank-2..8
   monotonically converges toward k_inf.
7. `diag_slab_specular_07_final_convergence.py` — mesh × N sweep
   confirms convergence.
8. `diag_slab_specular_08_rank1_bitexact.py` — element-wise math:
   K_bc_legacy = K_bc_per_face_div2 + (1/2)(G_o ⊗ P_i + G_i ⊗ P_o)
   relating the two formulations exactly. The per-face_div1 differs
   from Mark by an anti-symmetric matrix that doesn't affect the
   dominant eigenvalue.
9. `diag_slab_specular_09_promote.py` — final validation gate
   (test 1: rank-1 bit-exact; test 2: rank-N convergence). PROMOTED
   to `tests/derivations/test_peierls_specular_bc.py` as
   `test_specular_slab_rank1_equals_mark_kinf` and
   `test_specular_slab_homogeneous_converges_to_kinf`.

## Files modified

- `orpheus/derivations/peierls_geometry.py` — added slab-polar branch
  in `_build_full_K_per_group`'s `closure="specular"` block (~75 lines).
  Uses inline `_slab_E_n` for closed-form per-face primitives. Removed
  the NotImplementedError.
- `tests/derivations/test_peierls_specular_bc.py` — replaced
  `test_specular_slab_raises_not_implemented` with two new tests for
  rank-1 equivalence and rank-N convergence. Added SLAB_POLAR_1D
  import.

## Suggested cleanup

Diagnostics 01-08 are exploratory and can be DELETED (the fix is
documented in agent memory and the promoted tests). Diagnostic 09 is
the validation gate and is fully captured by the new pytest tests
in `test_peierls_specular_bc.py`.

## Lessons

- **Always verify the divisor and prefactor when inheriting per-face
  primitives from a combined-face implementation.** This is the third
  similar bug in the slab path (Issues #131, #132, now this).
- "K·1 = 1 for k=k_inf with uniform φ" is NOT a contract — it requires
  the eigenvector to be EXACTLY uniform, which doesn't hold for any
  finite-mesh slab even with infinite-medium-equivalent BC. The
  correct contract is "k_eff → k_inf as mesh refines AND N → ∞".
- The Mark-legacy / per-face-block-diag dominant-eigenvalue
  equivalence at rank-1 is a happy coincidence of homogeneous-slab
  symmetry; the K_bc matrices ARE different element-wise. Future
  derivations should not rely on element-wise equality.
