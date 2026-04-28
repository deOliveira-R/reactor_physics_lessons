---
name: Specular BC Phase 1 implementation findings
description: How `boundary="specular"` was wired, why we use no-Jacobian P primitives, and where Phase 2-3 work goes next
type: project
---

# Specular BC Phase 1 — implementation findings (2026-04-27)

Plan: `.claude/plans/specular-bc-method-of-images.md`. Phase 1 (mode-space
implementation + foundation tests + Sphinx) shipped on branch
`feature/peierls-specular-bc`.

## Key derivation result

R_specular in the rank-N Marshak shifted-Legendre basis is:

    R_spec = (1/2) M^{-1}

where `M_nm = ∫₀¹ µ P̃_n(µ) P̃_m(µ) dµ` is the partial-current overlap
matrix on the half-range basis. **M is symmetric tridiagonal** with
closed form

    M_nn       = 1/(2(2n+1))
    M_{n,n+1}  = M_{n+1,n} = (n+1)/(2(2n+1)(2n+3))

The contract `2 M R_spec = I` (i.e., J⁻_m = J⁺_m for all m=0..N-1)
holds at every rank by construction. SymPy verification at N=1..5 lives
in `derivations/peierls_specular_bc.py`.

**Importantly: R_spec is NOT parity-diagonal** (the plan's hand-wave
of `diag(+1, -1, +1, -1, ...)` was wrong — that came from a µ → 1-µ
half-range mirror argument, but specular at a curvilinear surface
preserves the absolute cosine, so in the |µ| ∈ [0,1] convention
specular is the **identity-like** condition with the (2n+1) Gelbard
factor folded in via the µ-weighted overlap M).

R_spec is **dense from rank 2 onward**: at N=2, R_spec = (1/2)·[[3,-3],
[-3,9]]. Off-diagonal entry R[0,1] = -3/2 ≠ 0 distinguishes specular
from `reflection_marshak` (strictly diagonal).

## Critical implementation gotcha — the (ρ_max/R)² Jacobian

`compute_P_esc_mode` (line 2935 of peierls_geometry.py) carries an
extra `(ρ_max/R)² · (1/R²)` Jacobian that does NOT appear in the
canonical observer-centred derivation for sphere (where ρ_max = s
along the chord, so ρ²/s² = 1). The factor was empirically introduced
during the Issue #132 rank-N closure calibration for diagonal Marshak
(see agent memory `direction_q_lambert_marshak_derivation.md`) and
contributes a partial calibration coincidence with the (2n+1) Gelbard
factor.

For diagonal R (Mark, Marshak), the spurious factor is harmless —
modes don't cross-couple, so the per-mode normalization stays
self-consistent.

For **dense R_spec**, the factor is **destructive**: the off-diagonal
R[0, n>0] couplings cross-couple mode 0 (which uses LEGACY
compute_P_esc with no Jacobian — what `build_closure_operator` uses
for n=0) with modes n≥1 (with the spurious Jacobian). The basis
mismatch makes specular DIVERGE from k_inf for homogeneous sphere
(observed: k → ~1.27 instead of k_inf = 1.5).

**Fix shipped in `_build_full_K_per_group`'s "specular" branch**:
custom inline assembly using a **no-Jacobian P primitive** for ALL
modes (n=0 included, where it reduces exactly to compute_P_esc since
P̃_0 = 1). With the uniform basis, sphere specular converges
monotonically:

    N=1: -0.82 % (rank-1 Mark calibration limit)
    N=2: -0.66 %
    N=3: -0.40 %
    N=4: -0.07 %  ← sweet spot
    N=5: +0.11 % (slight overshoot, R_spec conditioning)

## Cylinder rank-N — FIXED via Knyazev Ki_(2+k) primitives (Phase 1.5)

The original cylinder `compute_G_bc_mode` used the surface-centred
`Ki_1/d` form which evaluated `P̃_n` at the in-plane cosine `µ_2D` and
absorbed polar integration via `Ki_2(τ)` — only consistent at n=0
(where P̃_0 = 1 makes the polar integral trivial). For n ≥ 1 the
3-D direction cosine is `µ_3D = sin(θ_p) · µ_2D` and the polar
integration with P̃_n weight expands into the Knyazev `Ki_{2+k}`
series.

**Fix shipped**: new `compute_P_esc_cylinder_3d_mode` and
`compute_G_bc_cylinder_3d_mode` primitives in peierls_geometry.py.
For each mode n, expand `P̃_n(µ_3D) = sum_k c_n^k µ_2D^k sin^k(θ_p)`
and integrate analytically over θ_p using the identity
`∫_0^(π/2) sin^(k+1)(θ_p) exp(-x/sin θ_p) dθ_p = Ki_(k+2)(x)`. The
shifted-Legendre monomial coefficients `c_n^k` are computed via a
new helper `_shifted_legendre_monomial_coefs(n)` (numpy-based, with
LRU cache).

Cylinder convergence after fix:
    N=1: -0.31 % (Knyazev rank-1 ≡ Hébert rank-1, NOT Mark anymore)
    N=2: -0.28 %
    N=3: -0.21 %
    N=4: -0.11 %  ← within 0.5 % gate
    N=5: -0.04 %
    N=6: -0.02 %  ← within 0.05 %

See `derivations/peierls_cylinder_3d_mode_n.py` for the SymPy
derivation.

## Slab — FIXED (2026-04-27)

Slab specular SHIPS. Same per-face mode decomposition (mode space
A = R^(2N), one set per face, block-diagonal R = block_diag(R_face,
R_face) with R_face = (1/2) M^{-1}). The per-face P/G primitives are:

    P_face^(n)(x_i) = (1/2) sum_k c_n^k · E_(k+2)(τ_perp)
    G_face^(n)(x_i) = 2     sum_k c_n^k · E_(k+2)(τ_perp)

(no-µ-weight basis, mode 0 reproduces legacy `compute_P_esc_outer`
exactly).

The original "Marshak +7%, (1/2)M^{-1} -5%" plateau reported in this
memo was a bug, NOT a structural limitation. Root cause: the per-face
K_bc construction inherited the combined-face surface area divisor
(=2 for slab from `geometry.rank1_surface_divisor`) from the legacy
Mark code path. The combined-face divisor is correct for legacy Mark
(which sums BOTH faces into a single primitive). The per-face split
needs the **single-face divisor (=1)**.

Fix: in `_build_full_K_per_group`'s slab specular branch, set
`DIVISOR_PER_FACE = 1.0` overriding the geometry-level
`rank1_surface_divisor` lookup.

Convergence ladder at BASE quad (p=4, pn=2, q=24, dps=20):
- N=1: -0.195% (bit-equal to Mark legacy)
- N=2: -0.187%
- N=3: -0.169%
- N=4: -0.151%

See agent memory `specular_bc_slab_fix.md` for the full diagnostic
cascade (1.5 hours, 9 diagnostics, divisor finding at step 6).

## Phase 2 — Davison method-of-images cross-verification (next session)

The plan's load-bearing deliverable is the **cross-verification gate**:
mode-space specular k_eff vs image-series specular k_eff agreement to
machine precision. The image-series infrastructure was prototyped in
the Davison investigation (commit 30335f2 / agent memory
`issue_132_davison_image_series.md`) but used the wrong sign convention
for white BC. For SPECULAR, the same image-series construction with
corrected sign IS the right kernel.

Files to resurrect:
- `derivations/diagnostics/diag_sphere_davison_image_{01..04}_*.py`
- agent memory `issue_132_davison_image_series.md` for image positions
  and convergence behaviour

## Phase 4 — Heterogeneous Class B verification

Once cylinder is fixed (Phase 3) and image-series cross-verifies
(Phase 2), run the canonical test cases:

    sphere {1G/1R, 1G/2R, 2G/1R, 2G/2R}
    cylinder {1G/1R, 1G/2R, 2G/1R, 2G/2R}

Compare specular vs cp_sphere/cp_cylinder k_inf vs white_hebert k_eff.
Heterogeneous specular gives the angularly-exact pointwise eigenvalue;
divergence from cp k_inf and white_hebert MEASURES the Mark closure
error directly. This is the verification payoff of the whole plan.

## Files touched in Phase 1

- `orpheus/derivations/peierls_geometry.py`:
  - `reflection_specular(n_modes)` (~70 lines)
  - `closure="specular"` branch in `_build_full_K_per_group` (~80 lines)
  - Updated error message
- `derivations/peierls_specular_bc.py` (new, ~180 lines)
- `tests/derivations/test_peierls_specular_bc.py` (new, ~190 lines)
- `tests/derivations/test_peierls_closure_operator.py`:
  - 3 new foundation tests for reflection_specular (rank-1 = Mark,
    contract, dense off-diag)
- `docs/theory/peierls_unified.rst`:
  - New section `:ref:peierls-specular-bc` (~250 lines)
  - Class B closure table updated
