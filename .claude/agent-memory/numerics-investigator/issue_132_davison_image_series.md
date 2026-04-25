---
name: Issue 132 — Davison method-of-images for sphere white BC
description: Image-series viability probe FALSIFIED. Image series converges (n_max=5 saturates) but for homogeneous 1G/1R gives k_eff=0.704 vs cp_sphere k_inf=1.5 (-53% err). Specular ≠ white BC; method-of-images is structurally inapplicable to Mark closure.
type: project
---

# Issue #132 viability probe — Davison method-of-images

**Verdict**: ABANDON. The image series is well-defined for a homogeneous σ_t sphere, converges fast (n_max=5 already saturates within 1e-6), but converges to the WRONG eigenvalue — solving the SPECULAR-reflection problem, NOT the white-BC problem.

## Why: structural reason (Step 1)

Method of images requires the BC to act POINTWISE on the angular flux:
- vacuum: `ψ⁻(r_b, Ω·n<0) = 0`           — image at sign-flipped position
- specular: `ψ⁻(r_b, Ω) = ψ⁺(r_b, Ω - 2(Ω·n)n)` — image with angular flip

White (Mark) BC re-emits with the AVERAGE angular distribution:
- `ψ⁻(r_b, Ω) = J⁺(r_b)/π = (1/π) ∫_{Ω·n>0} (Ω'·n) ψ(r_b, Ω') dΩ'`

The re-emission is independent of the per-ray Ω, which CANNOT be reproduced by mirror images of the source. **No method-of-images formulation exists for white BC**, even on a homogeneous sphere.

## Empirical confirmation (Step 3, with proper quadrature)

After fixing Step 2's trapezoidal-quadrature artifact (log singularity at r'=r needed adaptive quad with explicit singular-point splitting):

- Sanity gate PASSES: vacuum-BC λ_max(M) → 1/σ_t = 1.0 as R→∞ ✓
- Image series CONVERGES: n_max=5 saturates within 1e-6 ✓
- **BUT** for sphere 1G/1R (σ_t=1, σ_s=0.5, νσ_f=0.75, R=1):
  - cp_sphere white-BC reference: k_inf = 1.500000
  - Davison image series k_eff = 0.703993 (n_max=20+)
  - Error: -53% — PERMANENT, NOT quadrature-related

The image series gives the result of a **2-mfp slab vacuum-BC problem**, not the spherical white-BC problem. The Davison u=rφ substitution makes the spherical Peierls equation LOOK like a 1-D slab, but the slab problem with specular images is a fundamentally different physics.

## Why MR breakdown is even worse (Step 4)

For multi-region σ_t, image points lie OUTSIDE the physical sphere where σ_t is undefined. Any choice of "material extension" (e.g., extend outer region to infinity) is ad-hoc and does NOT recover the cp_sphere reference.

## Diagnostic scripts (committed)

- `derivations/diagnostics/diag_sphere_davison_image_01_derivation.py` — SymPy image positions/signs, structural breakdown explanation
- `derivations/diagnostics/diag_sphere_davison_image_02_numerical_truncation.py` — naive trapezoidal (quadrature artifact, not promotable)
- `derivations/diagnostics/diag_sphere_davison_image_03_proper_quadrature.py` — adaptive quad, sanity gate PASSES, eigenvalue test FALSIFIED
- `derivations/diagnostics/diag_sphere_davison_image_04_multiregion.py` — MR structural barrier explanation

## Promotion recommendation

NONE of these should be promoted to `tests/`. They are negative results — image method does not apply. The infrastructure (image-series kernel + adaptive quadrature) might be reused IF a future investigation tests something else (e.g., specular-BC verification), but Issue #132 itself is closed.

## What this leaves for the chi=[0,1] limitation

The Hébert (1-P_ss)⁻¹ closure remains the shipped path for white-BC sphere. The chi-monotone Mark uniformity overshoot (commit 76b11e8 catalog: -1.5% at chi=[1,0], +6.6% at chi=[0,1], +10.3% at 1G) is NOT addressable by method-of-images. Alternative paths to consider:
- Higher-order Marshak (rank-N closure) — already falsified for both Class A and Class B (memory: peierls_rank_n_*.md)
- Direct SN reference for verification (different solver class)
- The pointwise Hébert-correction term in §3.8.5 that goes BEYOND rank-1 Mark (not yet implemented)
- An explicit per-mode angular-flux moment expansion at the surface (research direction)

## Reference

Hébert (2009/2020) §3.8.4-3.8.5 is the canonical home for sphere CP white-BC closure. The textbook does NOT discuss method-of-images for sphere — confirming that this path was never the canonical approach.
