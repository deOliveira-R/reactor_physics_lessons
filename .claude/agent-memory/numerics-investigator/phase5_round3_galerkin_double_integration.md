---
name: Phase 5+ Round 3 SECONDARY — Galerkin double-integration FAILED
description: Galerkin double-integration over panel cells with Lagrange test functions does NOT smooth the diagonal singularity. The 2-D integral ∫∫ K_continuous(r,r') dr dr' diverges with Q_µ at the same log rate as the per-pair K(r,r) — Nyström sampling at fixed Q_µ inherits the µ-divergence node-by-node, and the panel-pair sum captures the divergence. SHIPPED 8 diagnostics; production wiring NOT recommended.
type: project
---

# Phase 5+ Round 3 SECONDARY — Galerkin double-integration (2026-04-28)

## TL;DR

**Verdict: FAIL at smoke test, with NEW structural insight.**

The Galerkin double-integration approach
```
K_bc[i,j] = (σ_t,i/d) ∫∫ L_i(r) L_j(r') · rv(r') · K_continuous(r, r') dr dr'
```
does NOT smooth the diagonal singularity in `K_continuous(r, r)`. While
the diagonal singularity is an integrable `log|r-r'|` in 2-D (theoretically
finite integral), the **Nyström sampling at fixed Q_µ inherits a Q_µ-divergence
at every grid node** (each K_pair[r_qP, r_qP] grows as `log Q_µ`). The
panel-pair sum then captures this divergence: `∫∫ K dr dr' grows linearly
with log Q_µ`.

Test at panel `[1, 4]`, Q_r=16:
| Q_µ  | ∫∫ K_pair  | Δ vs prev |
|------|------------|-----------|
| 32   | 2.183      |           |
| 64   | 2.332      | +0.149    |
| 128  | 2.482      | +0.150    |
| 256  | 2.631      | +0.150    |
| 512  | 2.782      | +0.151    |
| 1024 | 2.932      | +0.150    |

**Each Q_µ-doubling adds ~0.150** — perfect log-linear growth.

Smoke-test k_eff at thin sphere (R=5, σ_t=0.5, k_inf=0.208, Hébert reference
-0.12 %): Galerkin v2 with vis-cone+u² subst per-pair quadrature gives k_eff
= 0.103 to 0.135 across (n_quad_r, n_quad_µ) = {4..16, 64..128} — **best
case is -34 % off, worst is -50 %**. INCREASING n_quad_r WORSENS the result
(monotone divergence), confirming the Galerkin smoothing fails.

## Approach taken

Implemented the cross-domain memo's Galerkin recipe step-by-step:

1. **Panel structure reuse**: ORPHEUS's `composite_gl_r` provides panel
   boundaries `[pa, pb]` and the local-panel Lagrange basis is built
   in-place via `_lagrange_basis_local`.

2. **K_continuous(r, r')** built using Front C's `_F_out_at` and `_G_in_at`
   primitives evaluated at arbitrary (r, r'), with the multi-bounce factor
   `f(µ) = µ/(1-e^{-2aµ})` (FULL M1 form) integrated over µ ∈ [µ_visible, 1]
   via vis-cone restriction + u² substitution (CORRECTED form — see below).

3. **Triple integral**: outer (r, r') panel-pair GL quadrature, inner µ
   integration per-pair. Total cost O(N_panels² · n_quad_r² · n_quad_µ ·
   N_total^2_evals) per K_bc assembly.

## Key debugging finding

**Front C / M2 / Galerkin per-pair K all have a SECOND bug** beyond the
M2 diagonal singularity: plain GL on `µ ∈ [0, 1]` misses the visibility
step at `µ_visible = max(µ_min(r), µ_min(r'))`. K_continuous(r, r', µ) =
0 for µ < µ_visible (chord doesn't cross both r and r'). At µ = µ_visible,
the integrand has a √-singularity from the vanishing cos(ω) factor.

Fix (`diag_phase5_round3_visibility_cone_quad.py`):
- Restrict µ to `[µ_visible, 1]`
- Substitute `u² = (µ - µ_lo)/(1 - µ_lo)`, `dµ = 2(1-µ_lo)·u du` —
  the Jacobian factor `u du` cancels the `1/√(µ-µ_lo)` endpoint singularity

With this fix, **off-diagonal** K(r, r') for r ≠ r' converges to MACHINE
PRECISION at Q=16 (verified for 5 pairs with |r-r'| ∈ {0.02, ..., 4.0}).

This is a real improvement that should be promoted as a fix for ANY future
continuous-µ K_bc Nyström sampling. **HOWEVER**, the diagonal r=r' STILL
log-diverges with Q_µ (M2 finding intact), and the 2-D Galerkin integral
inherits that.

## Why Galerkin fails — the structural argument

For 2-D integrability of `f(r, r') ~ -log|r-r'|`, we have `∫∫ f dr dr' ~ W²·log W`
which IS finite. But this is the **TRUE** 2-D integral.

The **Nyström-sampled** 2-D integral is:
```
I_Nyström(Q_r, Q_µ) = Σ_a Σ_b w_a w_b K_pair(r_a, r_b; Q_µ)
```
where `K_pair(r_a, r_b; Q_µ)` is the µ-integral computed via Q_µ GL nodes.

At fixed Q_µ, K_pair grows as `log Q_µ` at every diagonal point. Off-
diagonal entries are bounded uniformly, but **GL sub-points r_a are
distinct**, so K_pair[a, a] is bounded — but **at finite Q_µ, the
boundedness is `log Q_µ`-dependent**.

When we average over the panel via `Σ w_a w_b K_pair[a, b]`, the
diagonal contributions Σ w_a² · K_pair[a,a] sum a `log(Q_µ)` term per node,
giving total panel-integrated divergence proportional to `log Q_µ · Σ w_a²
≈ log Q_µ · W/Q_r` per panel.

So `I_Nyström ~ I_true + (W·log Q_µ)/Q_r + ...`. As both Q_r and Q_µ →
∞ in some scaling, I_Nyström → I_true. But the empirical Q_µ-divergence
at fixed Q_r dominates the algebraic Q_r-convergence to I_true.

**The fundamental problem**: the µ-integral itself is divergent at the
DIAGONAL. The 2-D integral over (r, r') is finite ONLY in the principal-
value / Hadamard-finite-part sense — i.e., `∫∫ K(r, r') dr dr' :=
lim_{ε→0} ∫∫_{|r-r'|>ε} K(r, r') dr dr'`. But when we evaluate
K_pair(r_qP, r_qQ) at a specific (r_qP, r_qQ) with r_qP very close to
r_qQ, the µ-integral DIVERGES — it's NOT a regular value of the 2-D
integrand.

To make Galerkin work, we'd need:
- Either: a closed-form **regularized** value of K(r, r) — Hadamard
  finite part — which in turn requires solving a nontrivial regularization
  gauge problem (memory's `phase5_round2_m2_bounce_resolved.md` Direction
  R3-A).
- Or: replace the singular 2-D integrand by a piecewise-defined kernel
  that's finite everywhere; only the off-diagonal Galerkin sub-integrals
  contribute, and the diagonal contribution is computed by a separate
  closed-form at exactly r=r'.
- Or: use **log-singular Gauss quadrature** in the (r, r') sub-integration
  to handle the log-singular kernel at spectral rate.

None of these is a quick fix.

## Files shipped (all in `derivations/diagnostics/`)

1. **`diag_phase5_round3_galerkin_double_integration.py`** — initial
   Galerkin v1 attempt with plain GL on µ ∈ [0, 1] (BROKEN — missed
   visibility step). 3 probes; smoke test fails wildly with Q-oscillation.
2. **`diag_phase5_round3_galerkin_diag_audit.py`** — 4 audits revealing:
   - Audit A: plain-GL µ-integration Q-oscillates even off-diagonal
     (visibility step is the culprit).
   - Audit B: diagonal r=r' Q-diverges (~log Q_µ).
   - Audit C: 2-D panel integral grows with Q_µ, decreases with Q_r.
   - Audit D: K(r, r+ε) ~ -A·log(ε) — confirmed log-singular diagonal.
3. **`diag_phase5_round3_visibility_cone_quad.py`** — vis-cone + u²
   substitution per-pair quadrature. **3 tests, all PASS**:
   - Vis-A: vis-cone alone converges off-diagonal at algebraic rate.
   - Vis-B: with u² subst, off-diagonal Q-converges to MACHINE PRECISION
     at Q=16 (5 pairs).
   - Vis-C: diagonal still log-divergent with substitution (M2 confirmed).
4. **`diag_phase5_round3_convention_check.py`** — scans HALF/FULL/DOUBLE
   M1 + Front C conventions in Nyström assembly. **None matches** Hébert
   reference (best is -15 % off; worst is +112 %). Convention is NOT the
   fix — the per-pair K is structurally wrong on the diagonal.
5. **`diag_phase5_round3_galerkin_v2.py`** — corrected Galerkin v2 with
   vis-cone+subst per-pair K and panel-pair Lagrange weights.
   - v2-A: smoke test fails -34% to -50% across (n_quad_r, n_quad_µ).
   - v2-B: 2-D panel integral grows linearly with log Q_µ — direct
     proof of structural divergence.

## Promotion candidates

NONE for production.

The vis-cone+u² substitution machinery (#3 above) MAY be useful as a
component of any future continuous-µ work, but the per-pair K it produces
is only meaningful OFF-DIAGONAL. As a permanent test it would be:
`tests/derivations/test_peierls_specular_continuous_mu.py::test_visibility_cone_offdiag_qconvergence`
gated at machine precision for the listed off-diagonal pairs. **Defer
promotion** until a working closure consumes this primitive.

## Recommendation for round 4

**R4-C (memory's R3-C): ABANDON Phase 5 production wiring.**

Both PRIMARY (adaptive µ-quadrature in µ-space) and SECONDARY (Galerkin
in r-space) failed for the SAME root cause — the µ-integrand at r=r' is
non-integrable. Front C / M1 / M2 / native / Sanchez (A6) / Galerkin
all give different rephrasings of the same divergent integral.

Production options remaining:
1. **Hadamard finite-part regularization** (R3-A from M2 memo) — picks
   a regularization gauge that may or may not match Phase 4's high-N
   limit. Gauge dependence is the risk.
2. **Closed-form diagonal**: compute K(r, r) by a separate analytic
   expression (Volterra-type integral identity) — requires deriving
   Sanchez 1986 §A or equivalent at the diagonal. Speculative.
3. **Ship `closure="specular_multibounce"` at N ≤ 3 only** (already
   shipped per `specular_bc_multibounce_shipped.md`). At N=1, this
   IS algebraically equal to Hébert white BC. At N=2, 3, it gives
   slight rank-N improvements (~0.01-0.1 % at thin sphere). This is
   the production-ready path; Phase 5 is research artifact only.

## Reconciliation with cross-domain memo's expectation

The memo predicted: "the L_i · L_j weighting smooths the diagonal singularity."
This is TRUE for an integrable singularity in (r, r'), but **the singularity
isn't intrinsically in (r, r') — it's in the µ-integrand at r = r'**. The
2-D Galerkin sums over (r, r') sample-points where r is close to r', and
the µ-integral at those points blows up at the same log(Q_µ) rate as at
the exact diagonal.

In BEM/IGA, the equivalent fix is to use **Duffy transformation** + **log-
singular Gauss quadrature** in the (r, r') panel sub-integration, but only
when the kernel `K(r, r')` is log-singular as a 2-D function. Here it's
log-singular as a 1-D function (along the diagonal), which is a DIFFERENT
issue and requires the µ-integrand itself to be regularized first.

## Lesson for future Phase 5 attempts

Whichever Phase 5 reformulation is tried, the FIRST diagnostic should be:
**compute K_continuous(r, r) via vis-cone+u² subst at Q_µ = {64, 256,
1024, 4096} and check for log(Q_µ) growth**. If the diagonal log-diverges,
the formulation has the M2 structural pathology and the same fate as
Front C / M1 / M2 / Galerkin. No further investigation needed unless a
NEW regularization scheme is brought in.

## Coordination with PRIMARY

PRIMARY is working on adaptive µ-quadrature (the same µ-integrand, with
denser nodes near `µ = µ_visible(r,r)` to chase log convergence). My
analysis predicts: PRIMARY will see the same log(Q_µ) divergence at the
diagonal, just shifted to (Q_µ_grid_density) divergence. Adaptive
quadrature CAN converge spectrally for an INTEGRABLE singularity at a
known location (Gauss-Jacobi etc.), but cannot rescue a NON-INTEGRABLE
singularity. PRIMARY's success would prove me wrong; consider their
results as the definitive disambiguation.
