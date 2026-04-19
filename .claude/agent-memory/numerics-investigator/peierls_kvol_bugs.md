---
name: Peierls K_vol integration bugs
description: Cross-panel near-singular and ray-crossing-panel-boundary bugs in Peierls Nystrom K-matrix assembly
type: project
---

# Peierls K_vol slab and sphere/cylinder bugs — 2026-04-18

## Rule

`peierls_slab._build_kernel_matrix` and
`peierls_geometry.build_volume_kernel` both use fixed-order GL without
basis-aware adaptive subdivision where the integrand is non-smooth,
producing ~1% K[i,j] errors that decay only O(h).

- **Slab**: cross-panel K entries use collocation `E_1(tau)*w_j/2`
  instead of `int (E_1*L_j) dx'`; error peaks (~1.5%) when observer
  is within small optical distance of source panel boundary.
- **Sphere/cylinder**: rho integration uses fixed GL across the whole
  [0, rho_max] interval, but L_j(r'(rho)) has kinks wherever
  r'(rho) = sqrt(r_i^2 + 2 r_i rho cos(om) + rho^2) crosses a spatial
  panel boundary. Fix exists in reference at
  `peierls_reference.py:238-255` (`rho_crossings_for_ray`).

**Why**: fixed GL of degree p-1 cannot resolve (a) near-log kernel near
adjacent panel boundaries, (b) derivative discontinuities of Lagrange
basis along non-straight arcs through panel boundaries.

**How to apply**: before doing ANY Peierls verification, include
element-wise K[i,j] checks — NOT just row-sum K·1. Partition-of-unity
masks this: `sum_j L_j = 1` cancels the kinks in row-sum for sphere
(row-sum 1e-15 clean despite K[3,3] wrong by 1.1%). Slab row-sum IS
affected (~1e-3 at panels=2) but less than individual entries
(1.5e-2).

## Decisive evidence

Three-way arbitration for sphere R=1 2-panel p=3:
- polar-adaptive ref (rho_crossings subdivided) agrees with
  shell-average ref (1-D partition-of-unity-cancelled form) to
  2e-11 to 1e-16 on well-conditioned entries.
- Production disagrees with both by 5e-4 to 5e-3.
- This rules out reference error, confirms production bug.

Slab quadrature scan at K[3,7] (cross-panel, far):
- p=4 GL integration: err 1.1e-2
- p=8 GL: 6.1e-5
- p=16 GL: 6.4e-9 (spectral, once p is high enough)
  → the bug is that fixed p=4 per panel is too low for cross-panel
  Lagrange-basis integration, not that the formula is wrong.

## Explains

- **V-canonical thin-R sphere 22% plateau** — likely.
- **V1 sphere R=1 2.5% plateau** — likely.
- **Sanchez 1982 cylinder tie-point 0.42% offset** — likely.
- **Row-sum K·1 at rank N=0 is exact** (partition of unity cancels).
- **Row-sum breaks for N>0** where `sum_j alpha_j L_j != 1` — this IS
  the K_vol bug appearing, not (or in addition to) a K_bc bug.

## Scripts

- `derivations/diagnostics/diag_slab_kvol_panel_boundary_bug.py`
- `derivations/diagnostics/diag_sphere_kvol_ray_crossing.py`
- Handoff plan: `.claude/plans/post-peierls-solver-bugs.md`

## Fix locations

- `peierls_slab.py:155-164` (cross-panel GL) → refactor to integrate
  Lagrange basis adaptively per panel pair.
- `peierls_geometry.py:491-516` (rho loop) → subdivide rho at
  `rho_crossings_for_ray(cos_om, rho_max)` and apply GL per sub-interval.

## Gotchas

1. Slab row-sum is only ~10x less sensitive than element-wise (drops
   from 1.5e-2 per-element to 1.75e-3 at near-boundary observer).
2. Sphere row-sum is 12+ orders of magnitude LESS sensitive (1e-15
   vs per-element 1e-2). Any reliance on K·1=1 for sphere
   verification is blind to this bug.
3. Existing eigenvalue tests gate at 1-2%, so the ~1% K bug does
   not fail them — they are masking.
