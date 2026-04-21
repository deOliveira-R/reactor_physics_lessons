---
name: Rank-N hollow-sphere W is MIXED basis (Lambert-in, Marshak-out)
description: Monte-Carlo + σ_t=0 closed-form evidence proving the transmission matrix W integrand is correct as an integral but is a mixed-basis operator. The closure assembly must account for this.
type: project
---

# Rank-N hollow-sphere W matrix: mixed-basis diagnosis (2026-04-21)

## The W matrix is correct as an integral

Monte-Carlo cross-check at R=5, r_0=1.5, Σ_t=1, 4M samples, 14
(m,n) pairs in the W_io and W_oo blocks — all agree with
`compute_hollow_sph_transmission_rank_n` to within statistical
precision (< 5σ). See `derivations/diagnostics/diag_rank_n_W_mc_crosscheck.py`.
σ_t=0 closed forms for W_io^{0,0}, W_io^{0,1}, W_io^{1,0} all match
to 1e-10. Reciprocity `A_k·W_{jk}^{mn} = A_j·W_{kj}^{nm}` holds to 1e-13.
**W is NOT the bug.**

## W IS mixed-basis

σ_t=0 values:

- W_io^{0,0} = s² = 0.09 (for s = r_0/R = 0.3)
- W_io^{0,1} = 4/3 - (4/3)(1-s²)^{3/2} - s² ≈ 0.0859
- W_io^{1,0} = s²/3 = 0.03

The integrand `cos θ · sin θ · P̃_n(cos θ) · P̃_m(c_in) · e^{-τ} dθ`
uses:

- **Emission side** (index n, cos θ): Ptilde_n(μ_emit) as a raw
  Lambert angular-flux mode function. The `cos θ · sin θ` factor is
  purely the dA_s measure on the emission surface (cos θ = μ_e
  converting observer-dΩ to surface-dA, and sin θ is polar Jacobian).
  It is NOT a μ-weight on the emission test function Ptilde_n.
- **Arrival side** (index m, c_in): Ptilde_m(μ_arrive), and the
  overall construction automatically produces the partial-current
  Marshak moment ∫ μ_arrive · Ptilde_m · ψ_arrive dμ_arrive · (2π ...)

Proven with `test_W_emission_basis_is_lambert_not_marshak`: the σ_t=0
value of W_io^{0,1} = 0.0859 matches the Lambert-emission-basis
closed form EXACTLY (1e-16), and differs from the Marshak-emission
closed form (0.0840) by 2e-3 — unambiguous.

## Why the current closure fails at N ≥ 2

The Phase F.5 per-face assembly in
`_build_closure_operator_rank_n_white` applies `(I - W)^{-1}` directly
on a W that takes **Lambert** emission modes in and outputs
**Marshak** arrival modes. Semantically this is nonsensical — the
reflection cannot iterate W · W because the output basis differs from
the input basis.

The correct reflection operators are (equivalently):

(A) `(I - (B^μ)^{-1} W)^{-1}` — converts Marshak output back to
    Lambert for the next emission.
(B) `(I - W (B^μ)^{-1})^{-1}` — applied on the other side.

Both give the same closure on the full product G · R_eff · P but
require P/G primitives to be in compatible bases with W.

## Required P/G basis

For the closure `K_bc = G · R_eff · P` to be basis-consistent with
W taking Lambert-in Marshak-out:

- P[kN+n, j] must output the **Lambert angular-flux coefficient c_n**
  of the emission ψ^+_k from a unit volumetric source at j. That is:
  P(q) should give the Lambert coeffs of the uncollided emission.
- G[i, kN+m] must process a **Marshak partial-current moment J^-_m**
  and return its volume response at i.

Currently:

- `compute_P_esc_outer_mode` (Lambert): integrand `sin θ · Ptilde_n(μ_e)
  · K_esc` — NO (ρ/R)² Jacobian. Likely NOT the right Lambert
  angular-flux coefficient. Missing Jacobian to convert from
  source-weighted observer integration to surface-ψ-coefficient.
- `compute_P_esc_outer_mode_marshak` (Phase F.5 Marshak): integrand
  `sin θ · μ_e · Ptilde_n(μ_e) · K_esc` — has μ but NO (ρ/R)².
  ALSO missing the surface Jacobian.
- `compute_P_esc_mode` (single-surface rank-N, KNOWN CORRECT for
  solid geometry N≥2): integrand `sin θ · (ρ/R)² · Ptilde_n(μ_e) ·
  K_esc` — HAS (ρ/R)² Jacobian, NO μ. The `(ρ/R)²` is `d²/R²` which
  equals `dA_s/dΩ_obs · μ_s / R²`. Combined with dA_s, this gives the
  surface-angular-flux-coefficient normalization.

**The per-face primitives should likely carry a similar `(ρ/R)²`-like
 Jacobian factor (or equivalent) to produce correctly-normalized
 Lambert or Marshak surface-moment coefficients.** Adding only μ
 (the Marshak variant) is NOT sufficient because it doesn't convert
 the observer-angular integration measure to the surface-angular-flux-
 coefficient measure.

## Confirmed: W is NOT the fix target

Prior sessions' 16-28 recipe scans combining Lambert/Marshak P and G
choices with various R = (I-W)^{-1} modifications did not succeed
because the P/G primitives themselves are neither correct Lambert
nor correct Marshak surface coefficients — both lack the surface
Jacobian.

## Next-step hypotheses (for Phase F.5 closure)

1. Copy the `(ρ_max/R)²` Jacobian structure from the single-surface
   `compute_P_esc_mode` into the per-face primitives, choose a clean
   basis (Marshak appears canonical per Sanchez-McCormick), and use
   `(I - (B^μ)^{-1} W)^{-1}` for the reflection.
2. Alternatively derive W's integrand with μ_emit weight (extra cos θ
   on emission side) making W pure Marshak-to-Marshak, then use
   `(I - W)^{-1}` directly. This changes the W formula and must be
   re-validated against Sanchez-McCormick Eq. III.F.xx.

## Artifacts

- `derivations/diagnostics/diag_rank_n_W_mc_crosscheck.py` — 33 pytest
  checks: MC cross-check, σ_t=0 closed forms for W_io^{0,0}, W_io^{0,1}
  (Lambert basis probe), W_io^{1,0}, reciprocity, mode asymmetry,
  matches-analytical-high-res. Promote to `tests/derivations/` once
  Phase F.5 closure lands.
