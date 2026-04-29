# Plan: Surface-centred angular quadrature recipe (L3)

**Author**: Claude Opus 4.7, 2026-04-29
**Predecessor branch**: `feature/quadrature-architecture` (commits `b281a97` Q1 → `6c67466` Q6).
**Scope**: Build the `surface_centred_angular_quadrature` recipe, the second geometry-aware recipe in `_quadrature_recipes.py`, and migrate the four legacy cylinder G_bc surface-centred branches to use it. This completes the contract-routing of every production quadrature call in `peierls_geometry.py`.

## 1. Why this is its own plan

Q1–Q6 routed every observer-centred ω-sweep through `observer_angular_quadrature`, the unified chord-impact-parameter through `chord_quadrature`, every static plain GL through `gauss_legendre`, every composite GL through `composite_gauss_legendre`, and every adaptive `mpmath.quad` through `adaptive_mpmath`. **One pattern remained**: the cylinder G_bc legacy form integrates over a *surface-centred* angle φ where the chord goes from observer at radius `r_i` to a surface point at `(R, φ)` in 2-D polar from origin. The kink structure differs from the observer-centred case in a non-trivial way (per-shell quadratic formula instead of `arcsin(r_k/r_obs)`), and `observer_angular_quadrature` does not apply.

The four sites are tagged in their docstrings as preserved-for-backward-compatibility with rank-1 Mark closure tests; the modern paths (`compute_G_bc_cylinder_3d` and `compute_G_bc_cylinder_3d_mode`) are observer-centred and already migrated. The legacy paths are still on `gl_float`. Migrating them is genuine architectural work because:
- the kink math has to be derived (not just implemented),
- the recipe needs L0 tests pinning the kink positions,
- the existing rank-1 Mark closure tests are tight (`rtol=1e-10`) so any quadrature shift surfaces as a test regression.

This document lays out everything a future session needs to do the work cold.

## 2. Math — the surface-centred kink structure

### 2.1 Geometry setup

The cylinder G_bc surface-centred form integrates over the angle φ of a surface point (in 2-D polar from origin) seen from an interior observer at radius `r_i`. With both points in the (x, y) plane:

- Observer at P = (r_i, 0).
- Surface point at Q(φ) = (R cos φ, R sin φ), φ ∈ [0, π] (the other half-plane is folded by symmetry).
- Chord length: `d(r_i, R, φ) = sqrt(r_i² + R² − 2 r_i R cos φ)`.

The legacy kernel is `Ki_1(τ_surf) / d` weighted by `dφ`, where `τ_surf` is the chord optical depth.

### 2.2 Impact parameter and shell crossings

The chord from P to Q has minimum distance to origin (impact parameter)

```
  b(r_i, R, φ) = r_i R |sin φ| / d(r_i, R, φ)
```

Derivation: the line through P with direction `(Q − P)/d` has perpendicular distance to origin `|P × (Q − P)| / d`, and the 2-D cross product `P × (Q − P) = r_i · R sin φ`.

The chord crosses the cylindrical shell of radius `r_k` iff `b(r_i, R, φ) < r_k`, i.e.,

```
  r_i² R² sin² φ < r_k² · d²(r_i, R, φ)
                = r_k² (r_i² + R² − 2 r_i R cos φ)
```

Substituting `c = cos φ` (so `sin² φ = 1 − c²`) and rearranging into a quadratic in `c`:

```
  r_i² R² c² − 2 r_k² r_i R c + (r_k² (r_i² + R²) − r_i² R²) > 0
```

The discriminant simplifies (this is the key non-trivial step):

```
  Δ = 4 r_i² R² · (r_k² − r_i²)(r_k² − R²)
```

For interior shells with `r_i ≤ r_k ≤ R`: Δ ≤ 0 (one factor non-positive, the other non-negative), the quadratic has no real roots and is positive everywhere, so the shell is **always penetrated** (consistent intuition: a chord from an observer inside or on shell `r_k` to the surface always crosses `r_k`).

For shells with `r_k < r_i ≤ R` (shells **interior** to the observer): Δ > 0, two real roots

```
  c± = (r_k² ± sqrt((r_i² − r_k²)(R² − r_k²))) / (r_i R)
```

with `c− ≤ c+` and both in [−1, 1]. The shell is **not** penetrated for `c ∈ (c−, c+)` (i.e., for `φ ∈ (arccos(c+), arccos(c−))`); penetrated otherwise. The chord becomes tangent to shell `r_k` at exactly two φ values:

```
  φ_tangent_high(r_k) = arccos(c−)   ← upper-φ tangent
  φ_tangent_low(r_k)  = arccos(c+)   ← lower-φ tangent
```

(Note: `arccos` is monotone-decreasing, so `c− < c+` ⇒ `arccos(c−) > arccos(c+)`.)

### 2.3 Number of kinks per observer

For an observer at `r_i ∈ (r_{j−1}, r_j]` (in shell j), interior shells are those with `r_k < r_i`, i.e., `k = 1, …, j−1`. Each interior shell contributes **two** tangent angles inside φ ∈ (0, π). The outer surface r_N = R is the upper-φ endpoint; shells r_k ≥ r_i contribute zero tangents.

So for an observer in shell j of an N-shell cell, the number of interior φ-kinks is `2 · (j − 1)`.

For an observer in the **innermost** shell (j = 1), there are zero interior tangents and the rule degenerates to plain GL on (0, π) — bit-equivalent to the legacy `gl_float(n, 0, π)` call.

### 2.4 Compare with observer-centred

For comparison, `observer_angular_quadrature` (already shipped) uses the simpler kink locations

```
  ω_tangent(r_k) = arcsin(r_k / r_i)   (and π − arcsin)
```

This is the closed-form simplification of the surface-centred quadratic when one of the two endpoints (the observer) is at the origin of the angular coordinate. The surface-centred case has the observer at finite radius, so the tangent angles depend on **both** r_i and R via the chord-quadratic, not just on r_k / r_i.

## 3. The recipe — `surface_centred_angular_quadrature`

### 3.1 Proposed API

```python
# orpheus/derivations/_quadrature_recipes.py

def surface_centred_angular_quadrature(
    r_obs: float,
    R: float,
    radii: np.ndarray,
    n_per_panel: int,
    *,
    phi_low: float = 0.0,
    phi_high: float = np.pi,
    dps: int = 53,
) -> Quadrature1D:
    r"""Surface-centred φ-quadrature on [phi_low, phi_high] for the
    legacy cylinder G_bc form.

    For an interior observer at radial position `r_obs` integrating
    over surface points at angle φ on a cylinder of outer radius `R`
    (chord d² = r_obs² + R² − 2 r_obs R cos φ), the integrand has
    derivative singularities at the tangent angles where the chord
    becomes tangent to each interior shell radius r_k < r_obs.
    Tangent angles are roots of a quadratic in cos φ — see
    `.claude/plans/surface-centred-quadrature.md` §2 for the
    derivation.

    Parameters
    ----------
    r_obs, R : float
        Observer radial position and outer radius. Must satisfy
        0 < r_obs ≤ R.
    radii : np.ndarray
        Outer shell radii. Tangent angles are computed only for those
        r_k < r_obs.
    n_per_panel : int
        Plain GL nodes per sub-panel.
    phi_low, phi_high : float, keyword-only
        Integration interval. Defaults to [0, π].
    dps : int, keyword-only
        Decimal precision for the underlying GL nodes. Default 53.

    Returns
    -------
    Quadrature1D
        Composite GL with subdivision at the tangent angles ∈
        (phi_low, phi_high). For an observer in the innermost shell
        (no interior tangents) returns plain GL on the full interval
        — bit-equivalent to ``gl_float(n_per_panel, phi_low, phi_high)``.
    """
```

### 3.2 Implementation sketch

```python
def surface_centred_angular_quadrature(r_obs, R, radii, n_per_panel,
                                       *, phi_low=0.0, phi_high=np.pi,
                                       dps=53):
    if not 0.0 < r_obs <= R:
        raise ValueError(...)
    if not phi_low < phi_high:
        raise ValueError(...)

    radii = np.asarray(radii, dtype=float)
    interior = radii[radii < r_obs]  # shells strictly interior

    # For each interior shell r_k, two tangent angles from the chord
    # quadratic: c± = (r_k² ± sqrt((r_i² − r_k²)(R² − r_k²))) / (r_i R).
    if len(interior) == 0:
        candidates = np.array([])
    else:
        rk = interior
        disc = (r_obs ** 2 - rk ** 2) * (R ** 2 - rk ** 2)
        # rk < r_obs ≤ R ⇒ disc ≥ 0
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        c_plus = (rk ** 2 + sqrt_disc) / (r_obs * R)
        c_minus = (rk ** 2 - sqrt_disc) / (r_obs * R)
        # Clip to [-1, 1] to guard against rounding past the closed
        # interval where arccos is defined.
        c_plus = np.clip(c_plus, -1.0, 1.0)
        c_minus = np.clip(c_minus, -1.0, 1.0)
        candidates = np.concatenate([np.arccos(c_plus), np.arccos(c_minus)])

    inside = (candidates > phi_low) & (candidates < phi_high)
    tangents = np.sort(candidates[inside])

    breakpoints = np.concatenate([[phi_low], tangents, [phi_high]])
    return composite_gauss_legendre(breakpoints.tolist(), n_per_panel, dps=dps)
```

### 3.3 Edge cases

- **Observer in innermost shell** (`r_obs ∈ (0, r_1]`): no interior shells, no tangents, plain GL. Degenerate case — should be bit-equivalent to current `gl_float(n_per_panel, phi_low, phi_high)`.
- **Observer on outer surface** (`r_obs = R`): allowed (the chord-quadratic still gives valid tangent angles). Useful for surface-source integrals.
- **`r_obs > R`**: rejected (observer outside the cell — no physical chord).
- **Degenerate tangent** at `r_k = r_obs`: discriminant is zero, both tangent angles collapse to a single value `c± = r_obs / R`. Single tangent point, still well-defined.
- **Numerical clipping at ±1**: the discriminant is non-negative for `r_k ≤ r_obs ≤ R` but rounding can push `c±` slightly outside [−1, 1] — clip before `arccos`.

## 4. Migration targets

Four legacy cylinder G_bc surface-centred branches, all in `orpheus/derivations/peierls_geometry.py`. Each currently uses `phi_pts, phi_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)` and a doubly-indexed loop over (observer, angular_node). After Q3 migration they're the only remaining `gl_float` consumers in this module besides the K_vol non-adaptive path (which is its own thing).

| File:line | Function | Branch | Pattern | Test exposure |
|-----------|----------|--------|---------|---------------|
| `peierls_geometry.py:1569` | `compute_G_bc` | cylinder | `Ki_1/d` over surface point | rank-1 Mark closure for cylinder (sample: `test_specular_cylinder_homogeneous_converges_to_kinf`) |
| `peierls_geometry.py:3112` | `compute_G_bc_outer` | cylinder | per-face Ki_1/d | per-face rank-N Mark closure |
| `peierls_geometry.py:3217` | `compute_G_bc_inner` | cylinder | per-face Ki_1/d at inner surface | hollow-cylinder F.5 closure |
| `peierls_geometry.py:3966` | `compute_G_bc_mode` | cylinder | mode-n Ki_1/d with `P̃_n(|µ_s|)` factor | rank-N Mark closure |

The pattern is uniform — each site has the structure

```python
phi_pts, phi_wts = gl_float(n_surf_quad, 0.0, np.pi, dps)
cos_phis = np.cos(phi_pts)
sin_phis = np.sin(phi_pts)
inv_pi = 1.0 / np.pi
for i in range(N):
    r_i = r_nodes[i]
    total = 0.0
    for k in range(n_surf_quad):
        cf = cos_phis[k]
        d_sq = r_i * r_i + R * R - 2.0 * r_i * R * cf
        d = np.sqrt(max(d_sq, 0.0))
        if d <= 0.0:
            continue
        # ... compute τ along the chord, kernel = Ki_1(τ) / d ...
        total += phi_wts[k] * kernel
    G[i] = (some prefactor) * total
```

Migration template (analogous to Q3's `observer_angular_quadrature` migration):

```python
def _per_obs(r_i: float) -> float:
    q = surface_centred_angular_quadrature(
        r_obs=r_i, R=R, radii=radii, n_per_panel=n_surf_quad, dps=dps,
    )
    cos_phi = np.cos(q.pts)
    integrand = np.fromiter(
        (per_node_kernel(r_i, float(c)) for c in cos_phi),
        dtype=float, count=len(q),
    )
    return prefactor * q.integrate_array(integrand)

return np.array([_per_obs(float(r_i)) for r_i in r_nodes])
```

For `compute_G_bc_mode` (cylinder branch), the kernel additionally evaluates `P̃_n(|µ_s|)` where `µ_s = (R − r_i cos φ) / d` — preserved as-is in the per-node kernel.

## 5. L0 test plan (`tests/derivations/test_quadrature.py`)

Add five tests for the new recipe, mirroring the structure of `observer_angular_quadrature`'s tests:

1. **No interior shells → plain GL.** Observer in innermost shell (`r_obs ∈ (0, r_1]`) → recipe returns one panel = `(phi_low, phi_high)`, weights sum to `phi_high − phi_low`, bit-equivalent to plain GL.
2. **Tangent angles inserted correctly.** Multi-shell case where the analytical closed-form `c± = (r_k² ± sqrt((r_i²−r_k²)(R²−r_k²)))/(r_i R)` can be hand-computed for one or two shells; verify panel boundaries match `arccos(c±)` to 1e-14.
3. **Window filtering.** Tangents outside `(phi_low, phi_high)` are dropped (e.g., partial-φ-range integrals).
4. **Constant integrand recovers length.** `∫_{phi_low}^{phi_high} 1 dφ = phi_high − phi_low` regardless of subdivision (sanity).
5. **Input validation.** `r_obs ≤ 0`, `r_obs > R`, non-monotone radii, bad `phi_low/phi_high` order.

Optional sixth test: cross-check against `observer_angular_quadrature` in the special case where the surface-centred and observer-centred forms agree (which they do **not** in general, but at the observer boundary `r_obs = R` with a 1-region cell, both reduce to plain GL).

## 6. Regression tests for the migrated branches

The four sites are exercised transitively by the cylinder rank-1 Mark closure tests. Per the Q3 audit, the relevant tests in `test_peierls_specular_bc.py`:

- `test_specular_cylinder_homogeneous_converges_to_kinf` (cylinder, `boundary="white_hebert"` and `"specular"`)
- `test_specular_2G_homogeneous_converges_to_kinf_2G[cylinder]`
- `test_specular_heterogeneous_1G2R_converges[cylinder]` (multi-region — the test that genuinely exercises the kink-resolution)
- `test_specular_heterogeneous_2G2R_converges[cylinder]`
- `test_specular_multibounce_cyl_rank1_equals_hebert` (cross-check between specular_multibounce-rank-1 and white_hebert-rank-1 — both go through the legacy cylinder G_bc)
- `test_specular_multibounce_cyl_lifts_thin_plateau`

For homogeneous (1 region) the recipe degenerates to plain GL and these tests should be bit-equivalent. For multi-region (1G2R, 2G2R) the recipe inserts kinks — accuracy improves, k_eff shifts within the test's `rtol=1e-5` (multi-region) or `rtol=1e-8` (rank-1 equality) tolerances.

**Failure mode to watch**: if `rank1_equals_hebert` cyl test diverges, the white_hebert and specular_multibounce paths are using *different* quadratures (one migrated, one not). The cyl rank-1 Mark closure has TWO sites that need migration in lockstep:
- `compute_G_bc` cylinder branch (line 1569) — used by both `white_hebert` and bare-`specular` rank-1.
- The surface-centred `Ki_1/d` is the legacy, but the `compute_G_bc_cylinder_3d` (Q3-migrated, observer-centred) is what the *corrected* form uses.

Audit before committing: ensure all four legacy sites + their callers are migrated together so the algebraic identity at rank-1 holds on the post-migration path.

## 7. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `arccos` input slightly outside [−1, 1] from rounding | Medium | `np.clip(c_pm, -1.0, 1.0)` before arccos; also guards the degenerate `r_k = r_obs` case |
| Shell ordering — `radii` may not be strictly increasing in some test fixtures | Low | Defensive `np.sort(...)` on the candidate tangents (already done in `observer_angular_quadrature`); same here |
| Bit-equivalence loss on rank-1 Mark closure cylinder homogeneous tests | Low | Innermost-shell observers degenerate to plain GL; for observer in shell j > 1 the kink subdivision improves accuracy. Tests at `rtol≥1e-8` should accommodate |
| `compute_G_bc_mode` mode-n cylinder branch has a `P̃_n(\|µ_s\|)` factor; verify it stays correct after migration | Medium | Test 6.h (a per-mode rank-N test on cylinder) should catch any sign / index errors |

## 8. Estimated scope

| Phase | What | LoC |
|-------|------|-----|
| L3.1 | Recipe + 5 L0 tests + Sphinx note | +120 / 0 |
| L3.2 | Migrate 4 cylinder G_bc legacy branches | +80 / −150 |
| L3.3 | Regression sweep + commit | 0 |
| **Total** | | **net −50 LoC, 1 session** |

## 9. Hand-off checklist

When picking this up:

1. Read this plan + `.claude/plans/quadrature-architecture.md` (the parent plan that ended at Q5+Q6).
2. Read commits `b281a97` (Q1, the contract) and `50f05ae` (Q3, the observer-angular recipe migration) to ground in the existing architecture.
3. Implement the recipe in `orpheus/derivations/_quadrature_recipes.py` next to `observer_angular_quadrature`. Keep the implementation close to the sketch in §3.2 — the math is in §2.
4. Write the 5 L0 tests in `tests/derivations/test_quadrature.py` per §5.
5. Verify L0 + 91 existing kernel/contract tests pass; verify Sphinx -W clean.
6. Migrate the 4 cylinder G_bc branches per §4. Match the `_per_obs` template; the per-node kernel is unchanged from the legacy code.
7. Run the cylinder rank-1 Mark closure regression tests per §6. Watch for the `rank1_equals_hebert` test as the canary (it's the algebraic identity between two paths that must stay consistent).
8. Commit as L3.

## 10. Non-goals (explicit)

- This plan does **not** propose a new mathematical formulation. The surface-centred form is preserved bit-equivalently — what changes is the *quadrature*, not the kernel.
- This plan does **not** unify the surface-centred and observer-centred recipes. They genuinely have different kink math (closed-form `arcsin(r_k/r_obs)` vs quadratic-formula `arccos(c±)`); forcing them into one recipe would obscure that.
- This plan does **not** retire the legacy cylinder G_bc forms. They are documented as backward-compat with rank-1 Mark closure tests; the corrected forms (`compute_G_bc_cylinder_3d`) live separately.

## 11. Reference reading

- `.claude/plans/quadrature-architecture.md` — parent plan, Q1–Q5 design.
- `.claude/plans/visibility-cone-substitution-rollout.md` — predecessor plan, set the context for vis-cone substitution that became Q1's primitive.
- `orpheus/derivations/_quadrature_recipes.py` — `observer_angular_quadrature` is the structural sibling; this recipe should look very similar in shape.
- `orpheus/derivations/peierls_geometry.py:1569, 3112, 3217, 3966` — the four legacy migration sites.
- `docs/theory/peierls_unified.rst` §22 — the coordinate-transform catalogue. After L3 lands, add a short §22.8 paragraph describing the surface-centred recipe (mirror what §22 intro already says about `observer_angular_quadrature`).
