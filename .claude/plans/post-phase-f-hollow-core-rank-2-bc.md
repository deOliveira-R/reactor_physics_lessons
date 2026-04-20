# Phase F — Hollow-core + rank-2 BC (per-face mode space)

**Branch entering this work:** `investigate/peierls-solver-bugs` (ahead of `main`; uncommitted items per session-end state).
**Author of this plan:** Claude Opus 4.7, 2026-04-21.
**Audience:** fresh Claude Code session with zero context. Read §§0-3 in full before touching any code.

---

## 0. One-paragraph summary (read first)

The BC tensor-network machinery in `peierls_geometry.py` currently supports rank-1 Mark white BC only, with a 16-40% k_eff error at ≤1 MFP cells across slab, cylinder, and sphere. Issue #110 (Phase F) is the architectural fix: extend `CurvilinearGeometry` to carry `inner_radius`, add inner-surface BC primitives, and generalize `BoundaryClosureOperator` to a per-face mode space `A = ℝ^(N_modes × N_surfaces)` with block-diagonal reflection `R = diag(R_outer, R_inner)`. This rank-2 structure is what the slab's legacy `peierls_slab._build_system_matrices` already implements (lines 277-330) via the `E_2`/`E_3` transmission relations — Phase F ports that pattern into the unified framework, at the same time enabling hollow cylindrical and spherical cells (BWR plenum, TRISO, pebble-bed). The Class-A topological unification from `.claude/plans/post-cp-topology-and-coordinate-transforms.md` §1 says all three 2-boundary geometries (slab + hollow cyl + hollow sph) are diffeomorphic to `[a, b]` and share rank-2 BC structure — once Phase F lands, `k_eff(white) = k_inf` holds at machine precision for all three (currently only asymptotic). Solid cylinder/sphere stay at rank-1 (Class B: single boundary + r=0 singularity); improving them requires rank-N angular expansion, which is a separate issue (#103). Phase F is scoped as 5 commits (F.1-F.5); it's realistically a 2-3 session effort.

---

## 1. Strategic goal (the "next level")

> Lift white-BC eigenvalue and uniform-source flux to **machine-precision** exactness for all 2-boundary geometries (slab + hollow cyl + hollow sph) via per-face rank-2 closure, and simultaneously add the capability to model hollow-core cells.

Today rank-1 gives:

| Cell (homogeneous, Σ_t = 1, Σ_s = 0.5, νΣ_f = 0.75; k_inf = 1.5) | k_eff(white) | rel err |
|---|---|---|
| Slab, L = 0.5 MFP | 0.9054 | **39.6%** |
| Slab, L = 1 MFP | 1.2526 | **16.5%** |
| Slab, L = 3 MFP | 1.4390 | 4.1% |
| Slab, L = 10 MFP | 1.4675 | 2.2% |

After Phase F (for homogeneous Class-A cells): all rows should be ≤ 1e-10 rel err — the exact Wigner-Seitz identity that should hold analytically for any finite L.

---

## 2. Current state entering this plan

### 2.1 What just landed (2026-04-21 sessions)

Commits on `investigate/peierls-solver-bugs` from the BC session (most recent first):
- `05d55a8` — slab-polar BC tensor-network support + Wigner-Seitz analytical (closed #118)
- `2538cfe` — slab white-BC analytical reference + BC-machinery slab-polar gap (historical; algebra bug corrected in 05d55a8)
- `1c0c48d` — retire slab-local duplicate helpers
- `25473e3` — retire composite_gl_y alias + optical_depths_pm chord walker
- `a4489d6` — vacuum-BC analytical references for cylinder + sphere (milestone)
- `e0fb8a2` — unified verification primitive + archive of moment form

### 2.2 BC machinery state (after 05d55a8)

- **`CurvilinearGeometry`** (`peierls_geometry.py` L202-692): kinds `{slab-polar, cylinder-1d, sphere-1d}`. All methods polymorphic. **No `inner_radius` support** — assumes observer inside solid cell.
- **`compute_P_esc`** (L1064-1112): polymorphic. For slab, uses closed-form `(1/2)·(E_2 + E_2)` sum over both faces. For cyl/sph, uses `gl_float` over the polar angle.
- **`compute_G_bc`** (L1115-1272): explicit branches for slab (closed-form `2·(E_2 + E_2)`), sphere (observer-centred θ), cylinder (surface-centred φ with `Ki_1/d`).
- **`compute_P_esc_mode`** / **`compute_G_bc_mode`** (L1272-1467): raise `NotImplementedError` for slab at `n_mode > 0` (pointing to this plan).
- **`BoundaryClosureOperator`** (L1496-1602): `K_bc = G · R · P` with mode space `A = ℝ^N` (SINGLE surface only).
- **`reflection_{vacuum,mark,marshak}`** (L1605-1654): scalar-mode reflection operators on A.
- **`build_closure_operator`** (L1657-1769): dispatches to the primitives; routes mode 0 through the legacy (Mark) P_esc/G_bc, modes n ≥ 1 through the `_mode` variants with `(ρ_max/R)²` Jacobian.

Factor-level gates: slab `P_esc` and `G_bc` match closed-form `E_2` at 1e-14 (machine precision). Structural symmetry and first-order row-sum checks pass.

### 2.3 What Phase F extends

Two independent axes:

1. **Geometry**: `inner_radius > 0` — hollow cylinder and hollow sphere become supported.
2. **BC structure**: `A = ℝ^(N_modes × N_surfaces)` — per-face reflection, which is what the slab's legacy code (`peierls_slab._build_system_matrices` L277-330) already implements for flat faces.

These two axes interact: a hollow cyl/sph has `N_surfaces = 2` (outer + inner); a slab has `N_surfaces = 2` (face_0 + face_L); a solid cyl/sph has `N_surfaces = 1` (outer). Phase F uniformly handles all three.

---

## 3. Mathematical core — rank-2 white-BC derivation

### 3.1 The partial-current balance on 2 surfaces

For a Class-A cell (slab [0, L], hollow cyl [r_0, R], hollow sph [r_0, R]) under white BC (albedo = 1, isotropic re-emission) on both surfaces, the Peierls integral equation with uniform source S = 1 on a pure absorber is:

```
φ(r) = (vacuum term) + 2·J⁻_{out}·G_{bc,out}(r) + 2·J⁻_{in}·G_{bc,in}(r)
```

where:
- `G_{bc,out}(r)` = scalar flux at r from unit uniform isotropic inward partial current on the OUTER surface only
- `G_{bc,in}(r)` = scalar flux at r from unit uniform isotropic inward partial current on the INNER surface only
- `J⁻_{out}`, `J⁻_{in}` = re-entering partial currents at each surface (two independent scalars)

The partial-current balance at each surface:
```
J⁺_{out} = P_{esc,out} · q + T_{in→out}·J⁻_{in} + 0 · J⁻_{out}
J⁺_{in}  = P_{esc,in}  · q + 0 · J⁻_{in} + T_{out→in}·J⁻_{out}
```

where `T_{out→in}` is the transmission coefficient from outer surface to inner (incoming isotropic at outer, outgoing at inner). For slab this is `2·E_3(τ_L)`; for hollow cyl/sph analogous chord-integration forms.

White BC closes each:
```
J⁻_{out} = J⁺_{out}
J⁻_{in}  = J⁺_{in}
```

Substituting and solving the 2×2 linear system for (J⁻_{out}, J⁻_{in}):
```
[1 - 0        -T_{in→out}] [J⁻_{out}]   [P_{esc,out}·q]
[-T_{out→in}  1 - 0       ] [J⁻_{in} ] = [P_{esc,in}·q ]
```

For symmetric cells the 2×2 collapses via J⁻_{out} = J⁻_{in}, recovering rank-1. For asymmetric (hollow with r_0 ≠ 0 or slab with different BCs per face), the full rank-2 is needed.

### 3.2 Per-face mode space

Generalize `A` from `ℝ^N` to `ℝ^(N × 2)` where the 2 = {outer, inner} surfaces:
- `A_{out} = ℝ^N` = scalar + Legendre moments on outer surface
- `A_{in}  = ℝ^N` = scalar + Legendre moments on inner surface
- `A = A_{out} ⊕ A_{in}` (direct sum)

The tensors become:
- `P : V → A` — shape `(2·N, N_r)`, with rows 0..N-1 carrying outer-surface moments, rows N..2N-1 carrying inner-surface moments
- `R : A → A` — shape `(2·N, 2·N)`, block-diagonal: `R = diag(R_{out}, R_{in})`
- `G : A → V` — shape `(N_r, 2·N)`, analogous to P

`K_bc = G · R · P` is unchanged as a composition; only the internal structure of A changes.

### 3.3 Rank-1 recovery when inner surface is absent

For solid cyl/sph (`inner_radius = 0`), the inner surface degenerates to a point (cyl) or doesn't exist (sphere at r = 0). In these cases, `A_{in}` contributes nothing:
- `P_{esc,in} = 0` (no volume source escapes to a non-existent surface)
- `G_{bc,in} = 0` (no re-entry contribution from a non-existent surface)
- `R_{in}` can be anything (zeroed out by P and G being zero)

The effective mode space reduces to `A_{out}` and rank-2 = rank-1. **Bit-exact recovery of rank-1 for solid cyl/sph is the regression gate** — existing tests must pass unchanged.

### 3.4 Verification: k_eff = k_inf for homogeneous Class-A cell

The acid test for rank-2 correctness: for any homogeneous cell with `Σ_s ≠ 0` and/or `νΣ_f ≠ 0`, white BC on all surfaces, `k_eff` must equal `k_inf = νΣ_f/Σ_a` to machine precision (1e-10), independent of L.

This is because Class-A cells with white BC on both boundaries conserve neutrons — they are indistinguishable from infinite medium at equilibrium. Rank-1 Mark misses the `T·J⁻` self-feedback transmission; rank-2 captures it explicitly via the per-surface coupled balance.

**Three test cases constitute the acceptance criterion for Phase F:**
- Homogeneous slab `[0, L]` with L ∈ {0.5, 1, 3, 10} MFP → k_eff = k_inf at 1e-10 for all L
- Homogeneous hollow cylinder `[r_0, R]` with `r_0 = 0.3·R`, various R → k_eff = k_inf at 1e-10
- Homogeneous hollow sphere `[r_0, R]` with `r_0 = 0.3·R`, various R → k_eff = k_inf at 1e-10

### 3.5 Hollow-core chord integration

For a hollow cell, a ray from observer r_i may traverse the **cavity** (interior to r_0) before exiting. The cavity has `Σ_t = 0` (void) — the ray continues without attenuation but the path length contributes to ρ_max. Walker specifics in `optical_depth_along_ray`:

- Intersection roots: rho_inner_− < rho_inner_+ < rho_outer if ray crosses the inner sphere
- The cavity segment `[rho_inner_−, rho_inner_+]` contributes 0 to τ but advances ρ
- After the cavity, the ray re-enters the annular material and attenuates until exit

For P_esc_outer(r_i): probability of escape TO THE OUTER SURFACE. Rays can reach outer directly (without crossing inner) OR through cavity traversal (rho_max = rho_outer, path = outer-ray with cavity segment).
For P_esc_inner(r_i): probability of escape TO THE INNER SURFACE. Ray must traverse from r_i to rho_inner_− (the first inner intersection) without exiting through outer first.

---

## 4. F.1 — `inner_radius` in `CurvilinearGeometry`

### 4.1 Scope

Pure extension. No behavioural changes to existing solid-geometry code paths.

### 4.2 Dataclass changes

```python
@dataclass(frozen=True)
class CurvilinearGeometry:
    kind: str  # "slab-polar", "cylinder-1d", "sphere-1d"
    inner_radius: float = 0.0  # r_0. Defaults to 0 (solid case; hollow if > 0).

    def __post_init__(self):
        # Existing kind validation
        ...
        # New:
        if self.kind == "slab-polar" and self.inner_radius != 0.0:
            raise ValueError(
                "slab-polar does not carry inner_radius (use face_0 / face_L "
                "per-surface BC directly)"
            )
        if self.inner_radius < 0.0:
            raise ValueError(f"inner_radius must be >= 0, got {self.inner_radius}")
```

Slab's "two faces" are positional (x=0 and x=L) rather than parametric via `inner_radius`. The per-face split is implicit in the slab geometry.

### 4.3 `rho_max` extension (hollow case)

For hollow cyl/sph with `inner_radius > 0`, `rho_max(r_obs, µ, R)` returns the distance to the **outer** boundary by default. A new method exposes the inner intersections:

```python
def rho_inner_intersections(self, r_obs: float, cos_omega: float) -> tuple[float | None, float | None]:
    """Signed intersections of the ray with the inner spherical/cylindrical
    shell at r = inner_radius. Returns (rho_in_minus, rho_in_plus) — the
    first and second intersections along the ray, or (None, None) if the
    ray does not reach the inner sphere.

    For slab, returns (None, None) — slab uses face_0 / face_L directly.
    """
    if self.inner_radius == 0.0 or self.kind == "slab-polar":
        return (None, None)
    # Quadratic in ρ: (r_obs + ρ·cos_ω)² + (ρ·sin_ω)² = r_0²
    # Same as `rho_max` but solving for inner radius instead of R.
    disc = r_obs**2 * cos_omega**2 - (r_obs**2 - self.inner_radius**2)
    if disc < 0:
        return (None, None)
    sqrt_disc = np.sqrt(disc)
    rho_minus = -r_obs * cos_omega - sqrt_disc
    rho_plus = -r_obs * cos_omega + sqrt_disc
    # Only positive ρ counts (ray goes forward)
    rho_minus = rho_minus if rho_minus > 0 else None
    rho_plus = rho_plus if rho_plus > 0 else None
    return (rho_minus, rho_plus)
```

### 4.4 `optical_depth_along_ray` extension

When the ray crosses the cavity `[rho_inner_−, rho_inner_+]`, the walker needs to treat that segment as zero-Σ_t. Implementation: add inner intersections to the `crossings` list and zero-out the `Σ_t` for the cavity segment:

```python
def optical_depth_along_ray(self, r_obs, cos_omega, rho, radii, sig_t):
    ...  # existing solid-geometry code walks annular crossings at radii[:-1]
    
    if self.inner_radius > 0:
        rho_in_minus, rho_in_plus = self.rho_inner_intersections(r_obs, cos_omega)
        # Insert cavity boundaries into crossings list
        # For segments entirely inside the cavity, use sig_t_cavity = 0
        ...
```

### 4.5 Tests (F.1)

- `TestCurvilinearGeometryHollow` — existence of `inner_radius` field, post-init validation
- `TestRhoInnerIntersections` — closed-form checks for rays from observer:
  - Tangent ray to inner shell: single intersection at `r_0 = r_obs · sin(tangent angle)`
  - Radial inward ray (`cos_omega = -1`): `rho_inner = r_obs - r_0`
  - Ray that misses inner shell: returns (None, None)
- `TestOpticalDepthAlongRayHollow` — multi-region with cavity:
  - Observer in annulus, ray straight through cavity: `τ = Σ_t·(path in annulus only)`
  - Ray that exits through outer without reaching inner: same as solid case
- Regression: all existing solid-geometry tests pass unchanged (default `inner_radius = 0.0`)

---

## 5. F.2 — Per-surface P_esc and G_bc primitives

### 5.1 API design

**Decision**: introduce per-surface functions, leave existing summed functions as rank-1 convenience.

New functions in `peierls_geometry.py`:

```python
def compute_P_esc_outer(geometry, r_nodes, radii, sig_t, ...) -> np.ndarray:
    """Probability of uncollided escape to the outer boundary only."""

def compute_P_esc_inner(geometry, r_nodes, radii, sig_t, ...) -> np.ndarray:
    """Probability of uncollided escape to the inner boundary only.
    Returns zeros for solid geometries (inner_radius = 0).
    For slab, returns escape to face x=0."""

def compute_G_bc_outer(geometry, r_nodes, radii, sig_t, ...) -> np.ndarray:
    """Scalar flux at observer from unit uniform isotropic inward J⁻
    on the outer boundary only."""

def compute_G_bc_inner(geometry, r_nodes, radii, sig_t, ...) -> np.ndarray:
    """Analog for inner boundary. Zero for solid geometries.
    For slab, this is the face x=0 contribution."""
```

Existing `compute_P_esc` and `compute_G_bc` stay as summed-over-surfaces convenience:
```python
def compute_P_esc(geometry, ...) -> np.ndarray:
    return compute_P_esc_outer(geometry, ...) + compute_P_esc_inner(geometry, ...)
```

This keeps backwards compat and makes the per-face split opt-in.

### 5.2 Slab per-face implementations

For slab homogeneous single-region:

```python
# compute_P_esc_outer(SLAB_POLAR_1D, ...) — escape through face x = L
# P_esc_L(x_i) = (1/2)·E_2(Σ_t·(L - x_i))
# 
# compute_P_esc_inner(SLAB_POLAR_1D, ...) — escape through face x = 0  
# P_esc_0(x_i) = (1/2)·E_2(Σ_t·x_i)

# compute_G_bc_outer(SLAB_POLAR_1D, ...) — unit J⁻(L) source at face L
# G_bc_L(x_i) = 2·E_2(Σ_t·(L - x_i))
#
# compute_G_bc_inner(SLAB_POLAR_1D, ...) — unit J⁻(0) source at face 0
# G_bc_0(x_i) = 2·E_2(Σ_t·x_i)
```

Sum matches the existing `compute_G_bc` / `compute_P_esc` SLAB_POLAR_1D branches, giving rank-1 recovery by construction.

### 5.3 Hollow cylinder per-surface

```python
# compute_P_esc_outer(cyl_hollow, r_i, ...) — probability the isotropic
# emission from r_i reaches r = R (not r = r_0 first).
#
# Integral: for each cos_omega, determine if ray reaches outer before inner;
# if yes, add exp(-τ(r_i, ρ_outer)) · angular_weight.
#
# compute_P_esc_inner(cyl_hollow, r_i, ...) — probability of reaching r_0
# (not R first). Nonzero only for rays that cross the inner shell.
```

### 5.4 Hollow sphere per-surface

Analogous, with sphere angular integration.

### 5.5 Tests (F.2)

- `TestSlabPescPerFace` — `P_esc_outer` matches `(1/2)·E_2(Σ_t·(L-x))` at 1e-14; `P_esc_inner` matches `(1/2)·E_2(Σ_t·x)` at 1e-14; sum matches existing `P_esc` at machine precision
- `TestSlabGbcPerFace` — analog for G_bc
- `TestHollowCylPesc` — basic sanity: `P_esc_inner + P_esc_outer = P_esc_total` (rank-1 sum); values positive, bounded by [0, 1]
- `TestHollowCylGbcInner` — unit inward J⁻ at r=r_0, flux at r_i > r_0 decays monotonically as r_i → R
- `TestSolidCylPescInnerZero` — for solid cyl (`inner_radius = 0`), `P_esc_inner` returns all zeros (regression safeguard)

---

## 6. F.3 — Per-face `BoundaryClosureOperator`

### 6.1 Scope

This is the **architectural heart** of Phase F. Changes `BoundaryClosureOperator` to support mode space `A = ℝ^(N_modes × N_surfaces)`.

### 6.2 Extended dataclass

```python
@dataclass(frozen=True)
class BoundaryClosureOperator:
    P: np.ndarray  # (N_modes * N_surfaces, N_r)
    R: np.ndarray  # (N_modes * N_surfaces, N_modes * N_surfaces) — block-diag
    G: np.ndarray  # (N_r, N_modes * N_surfaces)
    
    n_bc_modes: int = 1       # N_modes (scalar + Legendre moments)
    n_surfaces: int = 1        # 1 (solid) or 2 (Class A: slab + hollow cyl/sph)
    
    @property
    def closure_rank(self) -> int:
        return np.linalg.matrix_rank(self.R)
    
    def apply(self, q: np.ndarray) -> np.ndarray:
        return self.G @ (self.R @ (self.P @ q))
    
    def as_matrix(self) -> np.ndarray:
        return self.G @ self.R @ self.P
```

Mode layout convention (for rank-2):
- Rows/cols 0..N_modes-1: outer-surface modes (matches existing rank-1 layout → backwards compat)
- Rows/cols N_modes..2·N_modes-1: inner-surface modes

### 6.3 New reflection builders

```python
def reflection_mark_rank2(n_modes_per_surface: int, n_surfaces: int) -> np.ndarray:
    """Block-diagonal Mark reflection: R = diag(R_outer_mark, R_inner_mark)
    where each block is the rank-1 Mark projector (scalar mode only)."""
    
def reflection_marshak_rank2(n_modes_per_surface: int, n_surfaces: int) -> np.ndarray:
    """Block-diagonal Marshak/Gelbard reflection."""

def reflection_vacuum_rank2(n_modes_per_surface: int, n_surfaces: int) -> np.ndarray:
    """All zeros (R = 0) — same as rank-1 vacuum but sized correctly for
    per-face mode space."""
```

The existing rank-1 `reflection_mark`, `reflection_marshak`, `reflection_vacuum` remain (for backwards compat with solid geometries).

### 6.4 `build_closure_operator` extension

```python
def build_closure_operator(
    geometry, r_nodes, r_wts, radii, sig_t,
    *,
    n_angular=32, n_surf_quad=32, dps=25,
    n_bc_modes=1,
    reflection: str | np.ndarray = "marshak",
    n_surfaces: int | None = None,  # NEW: None → auto-detect from geometry
) -> BoundaryClosureOperator:
    """... (updated docstring)
    
    n_surfaces inference:
      - slab-polar: 2 (face_0 + face_L)
      - cylinder/sphere with inner_radius > 0: 2 (outer + inner)  
      - cylinder/sphere with inner_radius == 0: 1 (outer only)
    """
    if n_surfaces is None:
        n_surfaces = geometry.n_surfaces  # new property
    
    N = n_bc_modes
    N_r = len(r_nodes)
    
    # Build per-surface P and G, stack into block structure
    P = np.zeros((N * n_surfaces, N_r))
    G = np.zeros((N_r, N * n_surfaces))
    
    # Outer surface — modes 0..N-1
    P_outer = compute_P_esc_outer(...)  # + modes ≥ 1
    G_outer = compute_G_bc_outer(...)
    P[0, :] = rv_outer * r_wts * P_outer  # Mode-0 layout matches rank-1 legacy
    G[:, 0] = sig_t_n * G_outer / divisor_outer
    # ... higher modes
    
    if n_surfaces == 2:
        # Inner surface — modes N..2N-1
        P_inner = compute_P_esc_inner(...)
        G_inner = compute_G_bc_inner(...)
        P[N, :] = rv_inner * r_wts * P_inner
        G[:, N] = sig_t_n * G_inner / divisor_inner
        # ... higher modes
    
    # Build block-diagonal R
    if isinstance(reflection, str):
        R = reflection_map_rank2(reflection, n_bc_modes, n_surfaces)
    else:
        R = np.asarray(reflection, float)
    
    return BoundaryClosureOperator(
        P=P, R=R, G=G,
        n_bc_modes=n_bc_modes, n_surfaces=n_surfaces,
    )
```

### 6.5 Rank-1 bit-exact recovery

For solid cyl/sph (n_surfaces = 1), the new code path must produce **byte-exact** output as the existing rank-1 `build_closure_operator`. This is the regression gate: `test_peierls_closure_operator.py` continues passing unchanged.

### 6.6 Tests (F.3)

- `TestBoundaryClosureOperatorRank2Structure` — per-face stacking layout, R block-diag
- `TestRank2ReductionToRank1` — for solid cyl/sph, rank-2 build produces same K_bc as rank-1 (bit-exact)
- `TestRank2SlabHomogeneousKEffKInf` — **THE KEY GATE**: homogeneous slab `[0, L]` with fission + scatter, white BC via rank-2 Mark, gives `k_eff = k_inf` to 1e-10 for L ∈ {0.5, 1, 3, 10} MFP
- `TestRank2HollowCylKEffKInf` — same for hollow cyl with `r_0 = 0.3·R`
- `TestRank2HollowSphKEffKInf` — same for hollow sphere
- `TestRank2SymmetryUnderFaceSwap` — slab with mirror swap face_0 ↔ face_L gives identical K_bc (for symmetric problem)

---

## 7. F.4 — Hollow-sphere analogs + Sphinx docs

### 7.1 Implementation

Mirrors F.1-F.3 for sphere geometry. Most changes are already unified; only sphere-specific chord integration for `compute_G_bc_inner` needs derivation (the inner sphere's observer-centred θ integration with the cavity-traversal ray walker).

### 7.2 Sphinx docs

Extend `docs/theory/peierls_unified.rst`:

- New subsection `.. _peierls-rank-2-bc:` — the per-face closure derivation from §3 of this plan
- Update Key Facts header — add "Phase F complete: rank-2 BC for 2-boundary geometries"
- New equation labels: `peierls-rank-2-partial-current`, `peierls-hollow-rho-max`, `peierls-cavity-optical-depth`
- Cross-references to hollow-core case studies (BWR plenum, TRISO)

### 7.3 Tests

- Hollow sphere verification (mirrors hollow cyl tests in F.3)

---

## 8. F.5 — Verification suite

### 8.1 Primary gates (machine precision)

These are the L1 gates that mark Phase F as DONE:

1. **Slab k_eff = k_inf** at 1e-10 for homogeneous slab, white BC, any L — fixes the 16-40% rank-1 error. Test class: `TestSlabRank2KEffKInf`.
2. **Hollow cyl k_eff = k_inf** at 1e-10 for homogeneous hollow cylinder, white BC, any (r_0, R). Test class: `TestHollowCylRank2KEffKInf`.
3. **Hollow sph k_eff = k_inf** at 1e-10 for homogeneous hollow sphere, white BC, any (r_0, R). Test class: `TestHollowSphRank2KEffKInf`.
4. **Wigner-Seitz φ ≡ 1/Σ_t** pointwise for all three geometries (extends today's `TestSlabWhiteBCInfiniteMediumIdentity`). Test classes: `TestHollowCylWigner​Seitz`, `TestHollowSphWignerSeitz`.

### 8.2 Cross-checks

5. **Planar limit cross-check** (plan §3.4): hollow cylinder at `r_0 = 100 MFP, L = 1 MFP` (annular thickness) reproduces the slab homogeneous solution `φ = 1/Σ_t` at 1e-8 rel tol. Test: `TestPlanarLimitConvergence`.
6. **Solid-geometry regression** (all existing tests pass): `test_peierls_closure_operator.py`, `test_peierls_cylinder_white_bc.py`, `test_peierls_sphere_white_bc.py`, etc. Particularly critical: the rank-1 bit-exact pin tests must still pass (they gate the rank-1 → rank-2 reduction).

### 8.3 Physics tests

7. **BWR-style annular fuel**: `r_0 = 0.1 cm` (central plenum), `R = 0.5 cm`, single-region annular UO₂. Fission k-eigenvalue matches infinite-medium value (no leakage from white BC).
8. **TRISO-like multi-region**: concentric shells with different XS. Exercise the multi-region optical-depth walker with cavity traversal.

### 8.4 Rank-N extension (OPTIONAL in this session)

If time permits: per-face Marshak rank-N closure (n_bc_modes > 1 with n_surfaces = 2). Adds Gelbard DP_{N-1} moments per surface. Tests against Sanchez-McCormick tabulated values for rank-2 annular cylinder (if findable via `literature-researcher` + Zotero).

---

## 9. Risks and mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Rank-1 bit-exact regression breaks for solid cyl/sph | Medium | High | `TestRank2ReductionToRank1` as the earliest F.3 gate; pin rank-1 vs rank-2(n_surfaces=1) bit-exactly |
| Derivation error (history: commit `2538cfe` shipped a factor-of-2 bug that matched a similarly-buggy fixed-point diagnostic at 1e-39) | **High** | Critical | Cross-check rank-2 slab derivation via two INDEPENDENT routes: (a) per-face 2×2 partial-current balance solved in closed form; (b) direct numerical solution of the integral equation with convergence under mesh refinement. Demand agreement at 1e-10. |
| Hollow-core ray walker bug (cavity segment handling) | Medium | Medium | Test against solid-geometry paths with `sig_t_cavity = sig_t_material` (should recover solid); and against trivial cases (cavity spans full diameter → ray bypasses). |
| Scope creep — rank-N per-face + interface current | Low | Low | Explicitly out of scope for the primary Phase F; see §11. |
| Multigroup tangling | Low | Low | All Phase F work is 1-group; multigroup BC depends on #104. |

### 9.1 The "independent verification" discipline (learned today)

Commit `2538cfe` shipped an algebra bug. The fixed-point diagnostic "agreed at 1e-39" because it carried the same bug. Today's investigation caught it by comparing against the K_bc row-sum (a different code path). 

**Rule**: for any new analytical formula derived in this session, verify against TWO independent computational paths — not one. Specifically:

- Rank-2 k_eff = k_inf identity: verify via (a) full eigenvalue solve, (b) direct numerical solve of integral equation with discretized source iteration. Both must agree at 1e-10.
- Per-face P_esc / G_bc: verify via (a) closed-form E_n / Ki_n expressions, (b) adaptive mpmath.quad over the same integrand. Both must agree at 1e-10.
- Planar-limit reduction: verify via (a) hollow-cyl calculation at large r_0, (b) direct slab calculation. Both must converge to the same answer at 1e-8.

---

## 10. What NOT to do

1. **Do not break rank-1 solid cyl/sph** — all existing tests must pass unchanged. The `TestRank2ReductionToRank1` regression pin is the early warning.
2. **Do not attempt multigroup BC** — blocked on #104. Stay 1-group throughout Phase F.
3. **Do not introduce per-face modes for n_bc_modes > 1 without careful math** — rank-N per-face doubles the mode count and requires careful Legendre expansion on each surface. Optional in this session (§8.4); default scope is rank-2 Mark only.
4. **Do not retire `peierls_slab._build_system_matrices` yet** — that's Issue #111 (Phase G), which is a follow-up after F.3 lands. Retiring it mid-session risks breaking the multigroup slab path (which still lives there).
5. **Do not over-optimize** — compute_P_esc_inner etc. are verification-side, not hot path. Correctness over speed.
6. **Do not "verify" against a single route** (see §9.1). Always use two independent computational paths.

---

## 11. Out of scope (orthogonal work; don't do in Phase F)

- **Retire legacy rank-1 `build_white_bc_correction`** — scoped in an earlier audit as an independent cleanup item. Do AFTER Phase F (becomes easier once rank-2 is the default).
- **Peierls ↔ CP cross-check gating** — nice-to-have correctness check. Independent of Phase F; do separately.
- **Multigroup `solve_peierls_1g` generalization** — Issue #104. Precondition for #111 (legacy slab retirement).
- **Specular reflective BC** — trivial once per-face R supports arbitrary matrices (just set R = anti-diagonal identity for specular-in-mode-space). Can be done as a 1-line follow-up after F.3.
- **Albedo α ∈ (0, 1)** — `reflection_albedo(α) = α · reflection_mark`. Trivial follow-up.
- **Interface-current BC for multi-cell coupling** — different topology (cells talking via shared boundaries). Orthogonal to Phase F.

---

## 12. Reading order for the next session

1. **`.claude/lessons.md`** (unconditional per CLAUDE.md cardinal rule)
2. **`mcp__nexus__session_briefing()`** (unconditional)
3. **This plan** in full (§§0-8)
4. **Issue #110** — full text including the F.1-F.5 sequence and the topological unification motivation
5. **`.claude/plans/post-cp-topology-and-coordinate-transforms.md`** §§1-3 (Chapters 1-3): Class A/B topology, hollow-core implementation, planar limit
6. **`orpheus/derivations/peierls_slab.py`** L230-330 (`_build_system_matrices` with the legacy rank-2 E_2/E_3 BC — this is the MATH to port)
7. **`orpheus/derivations/peierls_geometry.py`** L1064-1770 (current BC machinery; F.3 modifies L1496-1770)
8. **`tests/derivations/test_peierls_reference.py`** L700-950 (today's Layer 6 slab BC tests; template for hollow-core test classes)
9. **`tests/derivations/test_peierls_closure_operator.py`** — rank-1 contract tests (the gate for bit-exact recovery)

Then start at §4 of this plan (F.1) and execute.

---

## 13. Acceptance criteria — when is Phase F done?

### Session 1 (F.1 + F.2) — preconditions

- [ ] `CurvilinearGeometry.inner_radius` field with validation
- [ ] `rho_inner_intersections` method with closed-form tests (tangent, radial inward, miss)
- [ ] `optical_depth_along_ray` handles cavity segment correctly (Σ_t = 0 interior)
- [ ] `compute_P_esc_outer`, `compute_P_esc_inner`, `compute_G_bc_outer`, `compute_G_bc_inner` implemented for slab + hollow cyl + hollow sph
- [ ] Factor-level tests at 1e-14 for slab per-face (closed-form E_2)
- [ ] Per-face sum recovers existing `compute_P_esc` / `compute_G_bc` at machine precision
- [ ] Solid-geometry regression: all existing BC tests pass unchanged
- [ ] Commit: `feat(derivations): inner_radius + per-surface BC primitives for hollow-core support (Phase F.1+F.2)`

### Session 2 (F.3) — the architectural payload

- [ ] `BoundaryClosureOperator` extended with `n_surfaces` field
- [ ] `reflection_{mark,marshak,vacuum}_rank2` builders
- [ ] `build_closure_operator` builds per-face K_bc for Class-A geometries
- [ ] Rank-1 bit-exact recovery for solid cyl/sph (test passes unchanged)
- [ ] **Slab k_eff = k_inf** at 1e-10 for homogeneous slab, white BC, L ∈ {0.5, 1, 3, 10} MFP (**THE headline gate**)
- [ ] Sphinx docs `:ref:peierls-rank-2-bc` subsection with full derivation
- [ ] Commit: `feat(derivations): per-face BoundaryClosureOperator — rank-2 white BC closes k_eff = k_inf identity for Class-A geometries (Phase F.3)`

### Session 3 (F.4 + F.5) — closure

- [ ] Hollow cylinder + hollow sphere full implementation
- [ ] Hollow cyl k_eff = k_inf at 1e-10
- [ ] Hollow sph k_eff = k_inf at 1e-10
- [ ] Wigner-Seitz φ ≡ 1/Σ_t pointwise for all 2-boundary geometries at 1e-10
- [ ] Planar limit cross-check: hollow cyl at r_0 = 100 ≈ slab at 1e-8
- [ ] BWR-style physics test (annular fuel with central plenum)
- [ ] Full Sphinx update (Phase F complete marker)
- [ ] Commit: `feat(derivations): hollow cyl/sph full verification suite + planar-limit cross-check (Phase F.4+F.5)`

### Final close-out

- [ ] Close Issue #110 with summary comment linking to the three commits
- [ ] Close Issue #100 (sphere rank-1 insufficient) — superseded by rank-2 for hollow sph
- [ ] Update `docs/theory/peierls_unified.rst` Key Facts header: "Phase F complete: machine-precision BC verification for Class-A geometries"
- [ ] Consider filing follow-ups for orthogonal work: legacy rank-1 retirement (independent), specular reflective / albedo (1-line adds), rank-N per-face (if needed)

---

## 14. File inventory (what to touch)

### Modified

- `orpheus/derivations/peierls_geometry.py` — major edits
  - `CurvilinearGeometry` dataclass + validation
  - `rho_inner_intersections` method
  - `optical_depth_along_ray` cavity handling
  - `compute_P_esc_outer` / `compute_P_esc_inner` (new)
  - `compute_G_bc_outer` / `compute_G_bc_inner` (new)
  - `compute_P_esc` / `compute_G_bc` routed through outer + inner (backwards compat)
  - `BoundaryClosureOperator` dataclass — `n_surfaces` field
  - `reflection_{mark,marshak,vacuum}_rank2` builders (new)
  - `build_closure_operator` — n_surfaces support

### Modified (tests)

- `tests/derivations/test_peierls_reference.py` — extend Layer 6 with hollow-core tests
- `tests/derivations/test_peierls_closure_operator.py` — add rank-2 structure tests; keep rank-1 bit-exact regression

### Added

- New singletons (optional): `CYLINDER_HOLLOW_1D`, `SPHERE_HOLLOW_1D` — parameterized by `inner_radius`. Probably NOT singletons since they carry geometry data; instead, factory functions.
- `derivations/diagnostics/diag_phase_f_rank_2_derivation.py` — independent verification of the rank-2 partial-current balance (per §9.1 discipline)
- `derivations/diagnostics/diag_planar_limit_hollow_cyl_to_slab.py` — convergence diagnostic for the plan's §3.4 planar limit

### Sphinx docs

- `docs/theory/peierls_unified.rst` — new `:ref:peierls-rank-2-bc` subsection + updated Key Facts
- `docs/theory/collision_probability.rst` — cross-reference for hollow-core CP (if CP side also updates)

---

## 15. Dependencies and ordering

```
F.1 (inner_radius) ──┐
                     ├─→ F.3 (per-face BoundaryClosureOperator) ──┐
F.2 (per-surface P_esc/G_bc) ─┘                                  ├─→ F.4 + F.5 (hollow sph + verification suite)
                                                                 │
                                                                 │
                                Orthogonal (any time): legacy rank-1 retirement, specular, albedo, rank-N per-face
```

F.1 and F.2 can be done in parallel (independent). F.3 requires both. F.4 and F.5 naturally follow F.3.

---

## 16. Estimated effort

- **Session 1 (F.1 + F.2)**: ~150 lines of code + ~300 lines of tests. Estimated 2-3 hours. Risk: low (additive).
- **Session 2 (F.3)**: ~200 lines of code + ~400 lines of tests. Estimated 3-4 hours. Risk: medium-high (architectural change, requires careful rank-1 regression + derivation cross-check).
- **Session 3 (F.4 + F.5)**: ~100 lines of code + ~500 lines of tests + ~200 lines of Sphinx. Estimated 2-3 hours. Risk: low-medium (mostly copy-pattern from cyl + verification work).

Total: ~8-10 hours across 3 focused sessions. Compare to the vacuum-BC milestone (similar scope, took 2 substantial sessions).

---

## End of plan

Phase F is the highest-leverage BC work remaining. F.3 alone closes the single biggest correctness gap (rank-1 Mark at finite L → 16-40% k_eff error for homogeneous cells) and simultaneously delivers hollow-core capability. The per-face mode space is a small architectural change with large verification payoff: the Wigner-Seitz identity φ = 1/Σ_t, currently analytical only, becomes a numerical pass at machine precision.

After Phase F lands:
- All three Class-A geometries (slab + hollow cyl + hollow sph) hit machine precision for white BC
- Solid cyl/sph stay at rank-1 (separate work, Issue #103, rank-N angular expansion)
- The slab's legacy rank-2 BC code in `peierls_slab.py` becomes retirable (Issue #111, Phase G)
- Specular / albedo / asymmetric BCs become 1-line additions on the per-face R

The hard part isn't the code. It's the derivation discipline (§9.1) — verify against two independent routes, always.
