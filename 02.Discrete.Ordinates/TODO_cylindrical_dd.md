# Cylindrical 1D SN Azimuthal Diamond-Difference

## Status: RESOLVED

Fixed in commits `fb3e976` through `5e00333` on branch `feature/geometry-module`.
25 cylindrical tests pass, 0 xfail.

## Root Cause (discovered through investigation)

The original diagnosis (sign convention / αψ tracking) was **wrong**.
The actual root cause was two bugs in the balance equation:

### Bug 1: Wrong α recursion

The code used `α = cumsum(+w·ξ)` with the azimuthal cosine ξ (`mu_y`).
The correct recursion (Bailey et al. 2009, Eq. 50) is
`α = cumsum(−w·η)` with the radial cosine η (`mu_x`), and ordinates
must be sorted by increasing η within each level.

### Bug 2: Missing ΔA/w geometry factor

The redistribution term in the balance equation must include
`ΔA_i / w_m` where `ΔA_i = A_{i+1/2} − A_{i-1/2}`.  Without this
factor, the streaming and redistribution do NOT cancel per-ordinate
for a spatially flat flux, creating artificial angular anisotropy
that worsens with mesh refinement near r = 0.

### Why the original hypothesis was wrong

The sign convention (`−` vs `+` before the redistribution) is absorbed
into the α definition: `cumsum(−η·w)` with `+` sign gives the same
physics as `cumsum(+ξ·w)` with `−` sign for symmetric quadratures.
The real issue was the missing geometry factor `ΔA/w` that ensures
per-ordinate flat-flux consistency.

Six approaches were tested before the correct fix was found:
reverse sweep, step closure, L&M starting direction, bidirectional
sweep, scaled α, and zero redistribution.  All failed because they
addressed symptoms, not the root cause.

## The Fix (Bailey et al. 2009 formulation)

The correct 1D cylindrical balance equation:

```
η_m [A_{i+1/2} ψ_{i+1/2} − A_{i-1/2} ψ_{i-1/2}]
  + (ΔA_i / w_m) [α_{m+1/2} ψ_{m+1/2} − α_{m-1/2} ψ_{m-1/2}]
  + Σ_t V_i ψ_{i,m}
  = S_i V_i
```

with `α_{m+1/2} = α_{m-1/2} − w_m · η_m` and ordinates η-sorted.

### Additional improvements applied

1. **Morel–Montry angular closure weights** (Bailey Eq. 74):
   τ_m replaces the standard DD weight (0.5) with position-dependent
   values, forcing the contamination factor β to zero.  Eliminates
   the Morel–Montry flux dip at r = 0.

2. **Consolidated ΔA/w into SNMesh**: `redist_dAw` (spherical) and
   `redist_dAw_per_level` (cylindrical) precomputed once, used by
   both the DD sweep and the BiCGSTAB operator.

3. **Same fix applied to spherical**: the spherical sweep had the same
   missing ΔA/w factor.  Fixed-source flux spike at r = 0 reduced
   from 5.1× to 1.1×.

4. **BiCGSTAB operators**: both spherical and cylindrical explicit
   transport operators updated with ΔA/w and M-M weights.
   Multi-group spherical BiCGSTAB (previously unstable) now converges.

## Results

| Test | Before | After |
|------|--------|-------|
| Homogeneous 1G/2G/4G (Product + LS) | Exact | Exact |
| Heterogeneous 1G, 5→10→20 cells | 1.15→0.90→0.52 (diverges) | 0.977→0.984→0.987 (converges) |
| Heterogeneous 2G, 4×8 vs 8×8 | 0.54 vs 0.91 (67% gap) | 0.723 vs 0.723 (<0.01%) |
| Fixed-source flux range (40 cells) | [0.59, 5.09] (spike) | [0.51, 1.12] (bounded) |
| Contamination β (cylindrical) | ~2.0 | ~1e-16 (machine zero) |

## Remaining TODOs

- **Gauss-type azimuthal quadrature**: the equally-spaced Product
  quadrature gives duplicate η values (paired ±ξ), producing
  alternating M-M weights τ = [0.5, 1.0, ...].  A non-uniform φ
  quadrature with distinct η would give smoothly varying τ and
  potentially better angular accuracy.

- **φ-based cell-edge computation**: for non-product quadratures,
  transforming actual φ cell boundaries to η-space (instead of
  midpoint interpolation) could give more accurate M-M weights.

## References

- Bailey, Morel & Chang (2009) NSE — Eq. 50 (α), Eq. 53–54 (WDD), Eq. 74 (M-M weights)
- Morel & Montry (1984) Transport Theory & Stat. Physics, 13:5
- Lewis & Miller (1984) §4.5 — starting direction treatment
- Carlson & Lathrop (1965) — α coefficient conservation property
