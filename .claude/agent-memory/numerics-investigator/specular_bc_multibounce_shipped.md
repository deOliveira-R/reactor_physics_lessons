---
name: Specular BC multi-bounce correction shipped
description: closure="specular_multibounce" wraps the bare specular K_bc=G·R·P with the matrix-form (I-T·R)^{-1} multi-bounce correction. Lifts thin-cell plateau to Hébert quality at N∈{1,2,3}. Sphere-only; cyl/slab raise NotImplementedError.
type: project
---

# Specular BC multi-bounce correction shipped (2026-04-27)

User ask: "Let's try the multi-bounce correction." Followed up the
documented thin-cell plateau (`specular_bc_thin_cell_plateau.md`) with
a production implementation.

## What shipped

**`closure="specular_multibounce"`** in `_build_full_K_per_group`
(branch `feature/peierls-specular-bc`):

```python
K_bc^spec_mb = G · R · (I - T·R)^(-1) · P
```

where T = `compute_T_specular_sphere(radii, sig_t, N)` is the
multi-region surface-to-surface partial-current transfer matrix:

```
T_mn = 2 ∫_0^1 µ P̃_m(µ) P̃_n(µ) e^(-τ(µ)) dµ
```

Multi-region τ(µ) accumulates Σ_t,k · ℓ_k(µ) per annulus along the
chord at angle µ from the inward surface normal (chord length 2Rµ;
impact parameter h = R√(1-µ²)).

## Verified behavior (BASE quad)

**Thin sphere fuel-A-like (σ_t=0.5, R=5, τ_R=2.5, k_inf=0.20833)**:

| N | bare specular | specular_multibounce |
|---|---------------|----------------------|
| 1 | -8.42 % | **-0.27 %** ✓ |
| 2 | -6.73 % | **-0.25 %** ✓ |
| 3 | -6.04 % | **-0.12 %** ✓ |
| 4 | -5.61 % | +0.43 % ⚠ overshoot |
| 6 | -5.21 % | +2.31 % ⚠ |
| 8 | -3.22 % | +5.62 % ⚠ |

**Thick sphere fuel A (σ_t=1.0, R=5, τ_R=5, k_inf=1.5)**:

| N | bare specular | specular_multibounce |
|---|---------------|----------------------|
| 1 | -0.82 % | -0.25 % |
| 2 | -0.66 % | -0.22 % |
| 3 | -0.40 % | -0.10 % |
| 4 | **-0.07 %** ✓ | +0.19 % (worse) |

## Best-use envelope

- `boundary="specular_multibounce"`, sphere, **N ∈ {1, 2, 3}**, any τ_R.
  Best for thin (lifts plateau); comparable to bare specular for thick at low N.
- `boundary="specular"`, any geometry, N as needed. Thick-cell sweet spot
  at N=4 for sphere.
- `boundary="white_hebert"`, sphere/cyl, rank-1 only. Algebraically
  identical to `specular_multibounce` at rank-1 sphere.

## High-N pathology (documented in code + warning)

At N≥4 the spectral radius ρ(T·R) approaches 1 (grazing modes µ→0
have chord 2Rµ→0 so transmission e^(-τ)→1, surviving infinite
reflections without attenuation). The geometric series (I - T·R)^(-1)
becomes nearly singular and the closure overshoots k_inf.
Implementation emits `UserWarning` for N≥5 to flag the pathology.

## Algebraic identity (rank-1)

At N=1, R = [[1]] and T_00 = P_ss exactly (verified to 1e-16 in the
test). Then (I - T·R)^(-1) = 1/(1 - P_ss) and the construction becomes:

```
K_bc^spec_mb|_{N=1} = G · 1 · (1/(1-P_ss)) · P = G · P / (1 - P_ss)
                    = K_bc^Hebert|_{rank-1}
```

So `boundary="specular_multibounce"` at N=1 is bit-equal to
`boundary="white_hebert"` for sphere. Test
`test_specular_multibounce_rank1_equals_hebert` pins this.

## Sphere-only — cyl/slab Phase 4

The cylinder analog requires a 3-D Knyazev surface-to-surface T
matrix (the in-plane chord has a polar-angle integration via Ki_(2+k)).
The slab analog needs a per-face block-T construction (each face has
its own self-bounce kernel; coupling to the antipodal face is
periodic-like). Both are tracked as Phase 4 follow-up.

For non-sphere geometries the dispatch raises NotImplementedError
with guidance: use `boundary="specular"` (single-bounce, any geom),
or `boundary="white_hebert"` (cyl rank-1, scalar Hébert).

## Files touched

- `orpheus/derivations/peierls_geometry.py`:
  - `compute_T_specular_sphere(radii, sig_t, N, n_quad=64)` (~95 lines)
  - `closure="specular_multibounce"` dispatch (~95 lines, sphere-only)
  - Updated error message at end of `_build_full_K_per_group`
- `tests/derivations/test_peierls_specular_bc.py`:
  - `thin_sphere_fuelA_like_1G` fixture
  - `test_specular_multibounce_rank1_equals_hebert` (PASS)
  - `test_specular_multibounce_thin_sphere_lifts_plateau` (PASS)
  - `test_specular_multibounce_rejects_non_sphere` (PASS)
- `docs/theory/peierls_unified.rst`:
  - New "Multi-bounce-corrected specular for thin sphere" subsection
  - Class B closure table updated with `specular_multibounce` entry
  - Test list extended

## Test runtime

3 new tests pass in 4.71s total (no MG, just fast 1G end-to-end at
BASE quad). Combined with existing specular suite: 18 tests, all
pass.
