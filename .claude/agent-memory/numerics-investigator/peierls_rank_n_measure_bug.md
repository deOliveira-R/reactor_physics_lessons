---
name: Rank-N hollow-sphere per-face measure-mismatch bug
description: The P/G primitives and W transmission matrix for rank-N hollow-sphere per-face white BC live in different angular-moment bases.
type: project
---

# Rank-N hollow sphere: P/G and W live in different measures

## The bug

In `orpheus/derivations/peierls_geometry.py` at commit `d890a1e`
(Phase F.5 infra), the rank-N per-face white-BC closure has a
fundamental inner-product mismatch:

- `compute_P_esc_{outer,inner}_mode(n)` and `compute_G_bc_{outer,inner}_mode(n)`
  (lines 1631-2165) compute **angular-flux (Lambert) moments**:
  integrand `sin(theta) * P_tilde_n(mu_exit) * f(theta)` — NO mu weight.
  Verified at sig_t -> 0: P_0 -> 1, P_1 -> 0 at the outer-surface node.

- `compute_hollow_sph_transmission_rank_n` (lines 2921-3050) computes
  **partial-current (mu-weighted / Marshak) moments**: integrand
  `cos(theta)*sin(theta) * P_tilde_n * P_tilde_m * exp(-tau)` — mu IS
  included. Verified at sig_t -> 0: W_oo[0,0] = 0.910, W_oo[0,1] = 0.247
  (matches mu-weighted half-range integral to 1e-10).

At N=1 the difference is invisible because P_tilde_0 = 1 collapses both
into scalar coefficients. At N>=2 the matrix product
`G @ (I-W)^-1 @ P` couples different bases — that's why adding mode 1
worsens the closure in every convention the user tested (C: 3.3% -> 5.6%,
A/B: 3.3% -> 14.8%).

## Gram matrices (numerical evidence)

At shifted-Legendre N=3:
- Lambert Gram `B^L = diag(1, 1/3, 1/5)` — diagonal, confirms P/G primitives orthogonal in flux measure.
- mu-weighted Gram `B^mu = [[0.5, 0.167, 0], [0.167, 0.167, 0.067], [0, 0.067, 0.1]]` — NOT diagonal. Contains the off-diagonal mode-coupling that W inherits.
- Converter `C = B^mu (B^L)^-1 = [[0.5, 0.5, 0], [0.167, 0.5, 0.333], [0, 0.2, 0.5]]` — its (0,0) entry is 0.5, which IS the hemispheric factor absorbed in the scalar rank-1 prefactor 1/2 in `compute_P_esc` (line 1493).

## The correct closure (hypothesis, not yet proven to close identity)

The right closure at N>=2 must put all three factored tensors in the SAME inner-product space. Two clean routes:

**Route A — move W to Lambert basis (angular-flux moments)**:
`W_lambert = (B^mu)^-1 @ W_current`  (per-face block-diagonal converter).
Reflection: `R_eff = (I - W_lambert)^-1`.
Tested: reduced outer-node K_bc*1 delta from +5% to +3%, but k_eff
residual still 6.4% — partial improvement, not a full fix.

**Route B — move P/G to partial-current basis**:
Add mu_exit weight to the P/G mode-n>=1 integrands; keep `(I-W)^-1`.
Tested: similarly partial, k_eff residual 5.3%.

**Route C — pre-multiply P by (2n+1) per face**:
Not fully tested — needed for the Gelbard expansion consistency.

NONE of the 7 conventions tested (see `diag_rank_n_sph_keff_probe.py`) produces <=0.1% residual at N=2.

## Key numerical evidence (R=5, r_0/R=0.3, sig_t=1, sig_s=0.5, nuSigf=0.75)

| Convention | N=1 residual | N=2 residual |
|---|---|---|
| Shipped (I-W)^-1 | 3.031% | **5.590%** (got worse) |
| (I-W)^-1 * diag(1,3,1,3) | 3.031% | **14.816%** (much worse) |
| (I - Bmu_inv W)^-1 | 3.031% | 6.358% |
| mu-weighted P,G + (I-W)^-1 | 3.031% | 5.284% |

The fact that NO simple recipe works cleanly at N=2 suggests there may be MULTIPLE bugs stacked:

1. Measure mismatch (confirmed via sig_t->0 limit).
2. Surface normalization — possibly the per-face divisor `R^2` / `r_0^2` in G needs to be the MARSHAK-MODE-SPECIFIC divisor, i.e. `R^2 * (2n+1)` or `R^2 * B^L_{nn}`.
3. The reflection_marshak `(2n+1)` factor at N>=1 may need to be applied to EACH mode individually per face, not as a global per-face diag.

## Scripts

All in `derivations/diagnostics/`:
- `diag_rank_n_sph_normalisation_measure_mismatch.py` — proves the Gram mismatch + shows row-sum shifts.
- `diag_rank_n_sph_normalisation_probe.py` — scans mu-weighted P/G primitive variants.
- `diag_rank_n_sph_keff_probe.py` — actual k_eff residual scan via monkey-patched closure.

## Recipe for next session

1. Reformulate P/G primitives in the canonical partial-current moment basis (Sanchez-McCormick 1982 §III.F), which has μ-weight AND (2n+1) expansion normalization.
2. Verify at sig_t -> 0 that the primitive reduces to 2*(2n+1)*int mu*P_tilde_n*psi^+ dmu (the well-defined Marshak partial-current moment).
3. Reformulate G as response per unit partial-current mode input, via psi^-(mu) = (2n+1)*a_n*P_tilde_n(mu) with delta on one mode.
4. Restate white closure as Marshak: J^-_m = J^+_m (partial current equality mode-by-mode). Coupling surface-to-surface carries via W directly.
5. Verify the closure at N=1 reduces bit-exactly to the proven Phase F.4 scalar rank-2 result.

## Related context

- Phase F.4 N=1 proven at sphere: k_eff = k_inf to 3% at r_0/R=0.3 (`tests/derivations/test_peierls_rank2_bc.py`).
- Sanchez-McCormick 1982 NSE 80, 481-535 §III.F — canonical rank-N per-face math.
- Mode primitives live in `orpheus/derivations/peierls_geometry.py` lines 1631-2165.
- W rank-N at lines 2921-3050.
- Assembly at line 3356 `_build_closure_operator_rank_n_white` (currently guarded NotImplementedError).
