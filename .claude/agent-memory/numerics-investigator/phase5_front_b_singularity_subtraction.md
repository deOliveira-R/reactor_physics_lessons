---
name: Phase 5+ Front B — Sanchez (A6) singularity subtraction
description: Sanchez 1986 Eq. (A6) interior diagonal integral DIVERGES (not just hard for GL); off-diagonal works with plain GL; subtraction gives a finite "regularised K[i,i]" but reproduces only c·ln-cutoff scale, not a closed-form value. Verdict PARTIAL — Approach 1 (Bernoulli subtraction) and Approach 4 (adaptive quad with subdivision) yield meaningful regularised diagonals; Approach 2 (µ=u²) is sub-spectral; Approach 3 (Gauss-Jacobi α=1) is good but only for the SINGULAR factor.  Production wiring not unblocked.
type: project
---

# Phase 5+ Front B closeout — singularity subtraction (2026-04-28)

Branch `feature/peierls-specular-bc`. Diagnostic shipped at
`derivations/diagnostics/diag_phase5_singularity_b01_subtraction.py`
(10/10 PASS, ~50 s wall via pytest, runs hands-off).

## Headline finding

**The on-diagonal Sanchez Eq. (A6) integral is a DIVERGENT improper
integral**, not merely a "hard-for-GL integrable singular integral"
as the Phase 5a closeout memo described it. Subtraction of the
analytic singular leading-order `s(µ) = c/µ` (interior diagonal)
peels the singularity but leaves a `c · ln(1/ε)` divergence — the
underlying continuous integral has no Cauchy/Riemann/Lebesgue value.

This is sharper than the closeout memo's "1/µ logarithmic
singularity, integrable but non-trivial for GL" framing. Plain GL
*accidentally* gives a bounded value because no node sits at exactly
µ=0; the answer drifts upward by `2c · ln(nq)` as nq grows
(diagnostic V3a confirms this rate to within 5%).

## Identified singular leading-orders

Numerically-stable derivation (V1, V2 of the SymPy script
`derivations/peierls_specular_continuous_mu.py`, extended in this
diagnostic):

```
INTERIOR (ρ' = ρ < a):     s(µ) = 2 / [µ · (e^{τ_0} − 1)]
                           where τ_0 = 2a · √(1 − (ρ/a)²)
                           Order = 1/µ¹ → integral DIVERGES on [0, 1]

SURFACE (ρ' = ρ = a):      s(µ) = 1 / (a · µ²)
                           Order = 1/µ² → integral DIVERGES MORE
                           strongly
```

Off-diagonal (ρ ≠ ρ' or surface ≠ interior): integrand is bounded;
plain GL converges to working precision (V7 in the diagnostic, 32
nodes give relative error <1e-6 vs reference).

**Critical numerical-stability finding** (load-bearing for any future
Phase 5+ work): the naive form

    µ_*² = ρ'² − ρ²(1 − µ²)

has catastrophic cancellation at small µ when ρ ≈ ρ'. The stable form

    µ_*² = (ρ'² − ρ²) + ρ² · µ²

avoids the `1 − µ²` round-off and is bit-stable down to µ = 1e-15.
The **shipped reference function**
`compute_K_bc_specular_continuous_mu_sphere` in `peierls_geometry.py`
uses the unstable form (lines 2592–2598) but is saved by the GL
node placement never reaching µ = 1e-9-scale; this is a latent
landmine for any adaptive-quadrature wiring. Recommendation: rewrite
those lines using the algebraic identity above.

## Approach-by-approach verdict

| Approach                                    | Verdict | Comment                                                             |
| ------------------------------------------- | ------- | ------------------------------------------------------------------- |
| 1. Bernoulli/analytic subtraction `f − s`   | OK      | Smooth remainder integrates cleanly with plain GL (V5 PASS at n=8 ≈ n=64 to 1e-3); BUT the discarded `c · ln(1/ε)` part is divergent. Useful as a "regularised K[i,i]" only if the cutoff has physical meaning. |
| 2. Change of variables `µ = u²`             | NO      | Converts `c/µ` into `2c/u` — same order of singularity in u. Worse for surface case. Sub-spectral convergence (V6).  |
| 3. Gauss-Jacobi α=1 (weight `(1+x)`)        | OK*     | Spectral for the c/µ singular factor IF it were the only term. Off-diagonal: algebraic only (V3b PASS at 1e-3 by n=64). On-diagonal: still divergent (the rule integrates `f/µ` against `µ`, which is `f`, which still diverges). |
| 4. Adaptive Gauss-Kronrod with subdivision  | OK      | scipy.integrate.quad with `breakpoints = [1e-6, 1e-4, …, 1.0]` triggers `IntegrationWarning` (max subdivisions hit) but returns a value. Slow, gives an `ε`-cutoff regularised result. |

*All four "OK" are PARTIAL — they handle the singular part but the
underlying integral is divergent so none gives a closed-form answer.*

## Verdict on the original mission

**PARTIAL.** Phase 5+ Front B does NOT unblock production wiring on
its own. The diagonal singularity is **structural to Sanchez 1986
Eq. (A6) used as a discrete Nyström kernel** — it is not a
quadrature-method bug.

The smoke test (V9, position-dependence shape vs `closure="white_hebert"`
at rank-1) PASSES at cosine-similarity > 0.5 on off-diagonal entries
of a 4-node grid, which is encouraging — Phase 5 K_bc has the same
qualitative position dependence as Hébert. But the diagonal entries
in `compute_K_bc_specular_continuous_mu_sphere` are quadrature-
dependent garbage (they grow as `2c · ln(nq)`).

Front A (Sanchez ↔ ORPHEUS K_ij Jacobian conversion) is **independent
of Front B** — it operates on the same off-diagonal entries that Front
B has now confirmed are reliable. The two fronts can proceed in
parallel; Front B is closed (with the verdict that diagonal entries
are intrinsically singular).

## Best path forward (recommendation)

1. **Patch the latent stability bug in
   `compute_K_bc_specular_continuous_mu_sphere`** (lines 2592–2598)
   to use `(ρ'² − ρ²) + ρ² µ²` form. Off-diagonal entries become
   bit-stable for any µ.

2. **Replace the diagonal entries with a regularised value**: the
   physically meaningful `K[i, i]` in a discrete-Nyström framework
   is reached as a limit `r_j → r_i` from off-diagonal samples,
   *not* by integrating the singular kernel pointwise. Two options:
   - **Diagonal extrapolation**: fit a smooth function to off-
     diagonal entries `K[i, j]` for `j` near `i` and read off
     `K[i, i]` as the extrapolated value. Standard in boundary-
     element methods.
   - **Cell-averaged diagonal**: for receiver cell `i`, integrate
     the singular kernel against the trial-function's support
     `L_i(r) · L_j(r')` over `r, r'` simultaneously — the source-
     side support smooths the singularity. Requires adopting
     ORPHEUS's full Galerkin treatment, not the collocation
     approximation currently used.

3. **Front A's Jacobian conversion can proceed independently** with
   the off-diagonal entries (which are reliable). Once Front A
   gives a constant scale α(r_i, r_j), the rank-1 cross-check
   against `closure="white_hebert"` becomes possible on off-diagonal
   entries only — the diagonal goes through the regularisation in
   step 2.

## What this Phase 5+ Front B output gives Phase 5++

- Confirmed: Sanchez Eq. (A6) is a structurally singular kernel,
  not a "computable with adaptive quadrature" kernel.
- Confirmed: the singular leading-orders are `c/µ` (interior) and
  `1/(aµ²)` (surface), with closed-form constants.
- Confirmed: the smooth remainder `f − s` is well-behaved and GL-
  integrable.
- Confirmed: off-diagonal entries (ρ ≠ ρ') are reliably computable
  with plain GL at n=32 to better than 1e-6.
- Provided: a numerically-stable algebraic form for `µ_*` that
  avoids `1 − µ²` cancellation.
- Identified: a latent stability issue in the shipped reference
  function (off-diagonal pairs near the diagonal could fail at very
  fine meshes).
- Provided: 10 pytest-format diagnostic tests that double as a
  permanent regression suite for any future Phase 5+ work
  (`derivations/diagnostics/diag_phase5_singularity_b01_subtraction.py`).

## Files touched by Phase 5+ Front B

- NEW `derivations/diagnostics/diag_phase5_singularity_b01_subtraction.py`
  (10 tests, ~480 LoC, all PASS)
- NEW `.claude/agent-memory/numerics-investigator/phase5_front_b_singularity_subtraction.md`
  (this file)
- INDEX entry pending in `.claude/agent-memory/numerics-investigator/MEMORY.md`

## Cross-references

- Phase 5a closeout (production blockers list):
  `.claude/agent-memory/numerics-investigator/specular_continuous_mu_phase5a_closeout.md`
- Sanchez literature memo:
  `.claude/agent-memory/literature-researcher/phase5_sanchez_1986_sphere_specular.md`
- SymPy V4 verification (kernel-form check):
  `derivations/peierls_specular_continuous_mu.py`
- Reference implementation:
  `orpheus/derivations/peierls_geometry.py:compute_K_bc_specular_continuous_mu_sphere`
  (lines 2440–2612)
- Latent stability bug location: same file, lines 2592–2598
  (`mu_minus_sq` and `rho_p_mu_star_sq` derivations should use the
  `(ρ'² − ρ²) + ρ²µ²` form instead of `ρ'² − ρ²(1−µ²)`).
