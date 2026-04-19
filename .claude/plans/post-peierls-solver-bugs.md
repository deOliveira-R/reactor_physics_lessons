# Peierls solver bug investigation — fresh-session handoff

**Branch:** `investigate/peierls-solver-bugs` (off `feature/mms-broad`)
**Investigator:** numerics-investigator, 2026-04-18
**Scope:** Slab + sphere Peierls K-matrix assembly bugs surfaced during
step-wise verification against the mpmath-adaptive reference in
`orpheus/derivations/peierls_reference.py`.

## Executive summary

Two independent but mathematically-related bugs in the Peierls Nyström
K-matrix assembly. Both manifest as **O(h) convergence where O(h^p) is
expected**, and both boil down to the same underlying issue: fixed-order
Gauss-Legendre cannot integrate `kernel × Lagrange_basis_j` when the
kernel has a near-singular or rapidly-varying structure within the
integration domain and/or the Lagrange basis has a derivative
discontinuity (kink) inside that domain.

- **Slab bug** (confirmed, reproduced, isolated):
  `peierls_slab._build_kernel_matrix` applies singularity subtraction
  only to **same-panel** (i, j) entries. Cross-panel entries where the
  observer is within a small optical distance of the source panel
  boundary suffer ~1% relative error; refining panels reduces this only
  at O(h) rate. Diagnostic:
  `derivations/diagnostics/diag_slab_kvol_panel_boundary_bug.py`.

- **Sphere bug** (confirmed with three-way arbitration, isolated):
  `peierls_geometry.build_volume_kernel` uses fixed-order GL over
  (omega, rho). Along a ray, the source position
  r'(rho) = sqrt(r_i^2 + 2 r_i rho cos(omega) + rho^2) crosses panel
  boundaries of the spatial Lagrange basis, where L_j(r') has
  derivative discontinuities. Fixed GL cannot handle the kinks,
  producing 0.5-5% relative errors on K entries (15-30% for
  origin-crossing rays). Diagnostic:
  `derivations/diagnostics/diag_sphere_kvol_ray_crossing.py`.

## (a) Slab bug — one-sentence statement

`peierls_slab._build_kernel_matrix` uses naive fixed-order GL for
cross-panel K[i, j] entries, producing ~1% relative error when the
observer x_i is within a small optical distance of the source panel
boundary (log singularity just outside the source panel); correct
Nyström requires product-integration (as already implemented for
same-panel entries via `_product_log_weights`) extended to all panels.

## (b) Sphere bug — one-sentence statement

`peierls_geometry.build_volume_kernel` uses fixed (omega, rho) GL
without subdividing the rho integration at values where
r'(rho) = sqrt(r_i^2 + 2 r_i rho cos(omega) + rho^2) crosses spatial
panel boundaries, so the Lagrange basis kinks along each ray are not
resolved — producing 0.5-5% K errors that only decay as O(1/n_rho)
under quadrature refinement.

## (c) Three most important code regions to change

1. `orpheus/derivations/peierls_slab.py:155-164` (off-diagonal-panel
   code path). Currently:
   ```python
   for i in range(N):
       for j in range(N):
           if node_panel[i] == node_panel[j]:
               continue
           tau = optical_path(x_nodes[i], x_nodes[j], g)
           if tau > 0:
               K[i, j] = e_n_mp(1, tau, dps) * w_nodes[j] / 2
   ```
   **Fix**: replace the naive collocation with per-panel Lagrange-basis
   integration that handles the near-log behavior. Either (a) extend
   `_product_log_weights` to integrate the full Lagrange basis L_j
   against the true E_1 kernel panel-by-panel (no singularity subtraction
   needed since x_i is outside the source panel — just adaptive
   quadrature or a higher-order tensor-product rule), or (b) keep the
   collocation but perform it per-panel with a sub-panel adaptive rule
   for adjacent panels.

2. `orpheus/derivations/peierls_geometry.py:491-516` (rho loop inside
   `build_volume_kernel`). Currently:
   ```python
   h = 0.5 * rho_max_val
   rho_pts = h * ref_rho_nodes + h
   rho_wts = h * ref_rho_wts
   for m in range(n_rho):
       rho = rho_pts[m]
       ...
       K[i, :] += weight * L_vals
   ```
   **Fix**: compute `rho_crossings_for_ray(cos_om, rho_max_val)`
   (already implemented in `peierls_reference.py:238-255` — copy it in)
   and run a separate GL rule on each subinterval between crossings.
   This restores spectral convergence in n_rho per sub-interval.

3. `orpheus/derivations/peierls_slab.py:73-106` (_product_log_weights).
   Currently takes one panel and one observer. Refactor to accept a
   cross-panel pair and return `int_{panel_B} kernel(x_i, x') L_j(x')
   dx'` for j in panel_B using adaptive mpmath.quad (no singularity
   subtraction needed since the singularity isn't in panel_B, but the
   near-log behavior as x' approaches the boundary closest to x_i
   requires adaptive/sub-divided quadrature).

## (d) Does this fully explain the V-canonical thin-R sphere plateau?

**Partially — likely fully for the thin-R plateau, partially for the
Sanchez 0.42% tie-point offset.**

- V-canonical R=1 sphere: plateau at ~22% (with mode-0) and V1 plateaus
  at 2.5% (with rank-N fix-up). The sphere K-matrix has **2-5% entry
  errors** at R=1 (e.g. K[3,3] at 1.1%), directly visible in
  `diag_sphere_kvol_ray_crossing.py`. Under quadrature refinement the
  error decays sub-exponentially (4.3% → 1.1% → 0.4% → 0.24% at
  n=8,32,64,128). **Any downstream k_eff computation inherits this
  ~1% floor.** This matches the V1 plateau magnitude well and suggests
  V1 looked like a "fix" only because its rank-N coefficients happened
  to cancel some of the K bug.

- Sanchez cylinder R=1.9798 tie-point at 0.42%: cylinder shares
  `build_volume_kernel`, so same ray-crossing bug applies. The 0.42%
  is consistent with the ~1% K-matrix error smoothed into the spectral
  radius of the full integral operator. Sanchez's scatter/fission-split
  ambiguity could still contribute, but fixing the sphere/cylinder
  K-matrix bug is expected to bring this well inside 0.1%.

- The rank-N investigation memory note about **"conservation test
  K·[1] = 1 error 3%→7% for sphere R=10 at N=2"**: this is NOT the
  same bug — the volume-kernel row-sum K_vol @ 1 is clean to 1e-15.
  The rank-N conservation failure must come from the K_bc (boundary
  closure) addition. That is a separate investigation; current bugs
  concern K_vol only.

## (e) Surprises encountered

1. **The slab bug is NOT only at panel-boundary neighbors.** All
   cross-panel K entries (K[3,4..7]) show ~1% error in the 2-panel p=4
   case, even when x_i is far from the source panel. This is because
   the integrand `E_1(sigma_t |x_i - x'|) * L_j(x')` has non-polynomial
   structure everywhere (E_1 is not polynomial); fixed GL at p=4 is
   simply insufficient for the full Lagrange-basis product integration
   across any panel. The neighbor case is merely WORST. **So "extend
   `_product_log_weights` to adjacent panels only" would not fix it —
   the fix needs to treat ALL cross-panel entries with basis-aware
   integration, not naive collocation.**

2. **The sphere polar-adaptive reference itself is imperfect at
   origin-crossing rays.** K[3,0] shows polar-vs-shell disagreement of
   1.35e-2 and K[5,0] 2.55e-2. The mpmath.quad subdivision in
   `rho_crossings_for_ray` does handle the r_b crossings, but the
   near-origin geometry (where tiny angular cone covers the origin
   basis function L_0) makes the adaptive integrator struggle. **The
   shell-average form is the more robust reference for deep-interior
   source, shallow-interior observer cases.** Consider promoting it
   to `peierls_reference.py`.

3. **Row-sum convergence in n_rho is NON-MONOTONIC** for the sphere:
   for `K[3,3]` the rel error at n_rho=8 is 4.3%, at n_rho=16 worsens
   to 8.0%, then n_rho=32 drops to 1.1%. This is the hallmark of
   integrating through a derivative discontinuity — GL nodes happen
   to straddle a kink at some orders and not others. **Tests that
   only check "error decreases under refinement" would miss this;
   needs a ratio-based or absolute-tolerance gate.**

4. **The full row-sum K @ 1 is CORRECT to 1e-15 despite K[i,j] being
   wrong by 0.5-5%!** This is the biggest surprise of the
   investigation. Explanation: `sum_j L_j(r') = 1` identically
   (partition-of-unity of the Lagrange basis), so the summed integrand
   of the row-sum has NO kinks at panel crossings even though each
   L_j individually has them. The ray-crossing bug CANCELS in the
   row-sum by exact partition-of-unity cancellation. **Consequence:
   every existing test based on "K @ 1 conservation" (e.g.
   `test_peierls_sphere_white_bc`, row-sum conservation at R=10 in
   the rank-N investigation) is BLIND to this bug.** Only element-wise
   K[i, j] tests can catch it. The rank-N memory note about "K·[1]=1
   passes at mode 0, fails at mode n>0" is consistent with this
   — at rank-N the test effectively probes a non-uniform L_j
   combination, breaking partition-of-unity cancellation.

4. **The slab eigenvalue test gates at 2% (not 1% as suggested in the
   task description)**: `test_k_eff_at_R_equals_1_dot_9798` asserts
   `|k - 1| < 0.02`. The 0.42% observed is consistent with the K-matrix
   bug. Ratcheting the test to `< 5e-3` AFTER the fix would catch any
   future regression of either bug via its downstream eigenvalue
   footprint.

## Reproduction commands

```bash
# Slab bug (fast)
python derivations/diagnostics/diag_slab_kvol_panel_boundary_bug.py
# Expected: panels=2 worst_cross_panel_rel_err ~ 1.5e-2, row_sum 4.1e-3,
#           ratios 2-3x (O(h), not spectral)

# Sphere bug (slower, ~30s for n=128 scan)
python derivations/diagnostics/diag_sphere_kvol_ray_crossing.py
# Expected: K[0,0] prod_vs_shell ~ 1.6e-3,  K[3,3] 1.1e-2,
#           K[5,0] 3.2e-1 (near-origin failure).
#           Quadrature scan plateaus at ~2e-3 even at n=128.

# Failing test (existing)
pytest tests/derivations/test_peierls_reference.py -v
# TestSlabKMatrixElementwiseVsReference fails with ~1e-3 relative
# error at 4 test entries.
```

## Proposed fix plan

### Slab (peierls_slab.py)

1. Refactor `_product_log_weights` to a generic
   `_product_kernel_weights(source_panel, observer_x, nodes, ..., kernel=E_1)`
   that integrates `L_j(x') * kernel(sigma_t |observer - x'|)` against
   each basis function using mpmath.quad (same-panel: subdivide at
   observer; cross-panel: adaptive over the source panel).
2. In `_build_kernel_matrix`, iterate over (observer_panel,
   source_panel) pairs and compute weights with the new function,
   using full singularity subtraction when same-panel and full
   non-singular adaptive integration when different-panel.
3. Alternatively, keep singularity subtraction only for same-panel and
   use high-order GL per sub-interval (split at the `rho_crossings`
   near-singularity line) for cross-panel. Simpler, but less elegant.

### Sphere / cylinder (peierls_geometry.py)

1. Compute `rho_crossings_for_ray(cos_om, rho_max_val)` once per
   (i, omega) pair — the same function already exists in
   `peierls_reference.py:238-255`.
2. Split the rho integration into sub-intervals at each crossing and
   apply the existing GL rule on each sub-interval. Keep n_rho per
   sub-interval as a parameter (default 8 or so; total rho nodes
   scale as n_rho * n_crossings).
3. Regression test: `K @ [1]` row-sum identity (both slab + curvilinear)
   must converge to < 1e-8 at moderate quadrature orders.

## Minimal regression tests (gate AT 1e-10)

To be added to `tests/derivations/` after the fix:

```python
def test_slab_K_cross_panel_elementwise():
    """2 panels, p=4, L=1, Sigma_t=1. K[i,j] for all cross-panel (i,j)
    must agree with mpmath-adaptive reference to 1e-10."""
    K, x_nodes, pbs = _build_slab_K(L=1.0, sig_t=1.0, n_panels=2, p_order=4, dps=40)
    N = len(x_nodes)
    for i in range(N):
        for j in range(N):
            if node_panel[i] == node_panel[j]:
                continue
            ref = slab_K_vol_element(i, j, x_nodes, pbs, L=1.0, sig_t=1.0, dps=40)
            assert abs(K[i, j] - ref) / abs(ref) < 1e-10

def test_sphere_K_elementwise_vs_shell_average():
    """R=1, Sigma_t=1, 2 panels, p=3. K[i,j] production must agree with
    shell-average reference to 1e-8 on all "easy" entries (observer
    not near-origin seeing source basis near-origin)."""
    # Iterate entries, assert abs diff vs shell_avg_K < 1e-8
    ...
```

## Files touched this session (all new, no production edits)

- `derivations/diagnostics/diag_slab_kvol_panel_boundary_bug.py`
- `derivations/diagnostics/diag_sphere_kvol_ray_crossing.py`
- `.claude/plans/post-peierls-solver-bugs.md` (this file)

## GitHub issues to file (by fresh session)

1. `module:derivations module:cp level:L1 type:bug` — "Peierls slab
   K-matrix: cross-panel integration not basis-aware (~1% error,
   O(h) convergence)". Reference `peierls_slab.py:155-164`,
   `diag_slab_kvol_panel_boundary_bug.py`.
2. `module:derivations module:cp level:L1 type:bug` — "Peierls
   curvilinear K-matrix: rho integration does not subdivide at ray
   crossings of spatial panel boundaries (~1-5% error, sub-polynomial
   in n_rho)". Reference `peierls_geometry.py:491-516`,
   `diag_sphere_kvol_ray_crossing.py`.
