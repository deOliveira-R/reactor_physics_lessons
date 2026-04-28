# Plan: specular reflective BC for Peierls (method-of-images foundation)

**Author**: Claude Opus 4.7, 2026-04-27.
**Issues**: #100 (sphere rank-1 insufficient), #103 (rank-N closure), #132 (Mark uniformity overshoot — the structural ceiling this plan tries to bypass).
**Branch (proposed)**: `feature/peierls-specular-bc`.
**Prerequisites**: branch `feature/rank-n-class-b-mr-mg` (current) merged to main, OR rebased onto.

## 1. Executive summary

- **What we are building**: a specular-reflective boundary condition for the Peierls integral framework, with TWO mathematically-independent implementations (mode-space `R_specular` operator + Davison method-of-images kernel). They cross-verify each other to machine precision; together they give the FIRST closure-approximation-free Peierls reference for Class B that bypasses the Mark uniformity assumption entirely.
- **Why now**: the Mark uniformity overshoot on heterogeneous Class B is the documented structural ceiling for the rank-N IC closure family (Issue #132 close-out). Three independent investigations (Davison/AugNystr/Sanchez 1977 PDF) converged on this verdict. **Specular reflection is a different BC class** — it preserves the angular distribution at the surface, so there's no "uniform isotropic re-entry" approximation. For a homogeneous cell, specular k_eff = white k_eff = k_inf by symmetry; for heterogeneous cells, **specular gives the angularly-exact pointwise eigenvalue**, which IS the L1 reference cp_sphere/cp_cylinder need.
- **Why specifically method-of-images**: the Davison investigation (commit 30335f2 / agent memory `issue_132_davison_image_series.md`) established that the image series `K_image = Σ_n (-1)^|n| · [E_1(τ|r-2nR-r'|) - E_1(τ|r-2nR+r'|)]` for sphere converges fast (~5 terms) but to the *wrong* eigenvalue under white BC because mirror sources can't reproduce the Mark averaging. **The same image series IS the correct K_bc kernel for SPECULAR reflection** — that's exactly what method-of-images solves. We have the infrastructure already; we just need to wire it as a specular BC, not (failing to) shoehorn it into white.
- **Verification mode change**: instead of "Peierls white-BC verifies CP white-BC k_inf" (same closure family, circular), the new posture is **"Peierls specular-BC verifies CP white-BC k_inf for homogeneous cells exactly, and bounds the closure error for heterogeneous cells independently"** — specular and white agree by symmetry on homogeneous; for heterogeneous, the divergence between specular and CP measures the CP Mark closure error directly.
- **Expected duration**: 4-6 sessions. The two-path implementation is core; verification of agreement and Sphinx writeup are the rest.
- **Acceptance**: at least one of {sphere, cylinder} ships specular BC + machine-precision agreement between mode-space and image-series paths + Class B 1G/2R k_eff at fraction-of-percent tolerance against an INDEPENDENT reference (MC if needed; or accepted as the new analytical reference itself).

## 2. Background and prior art

### 2.1 What's already in place

**Tensor-network architecture** (`peierls_geometry.py:3157-3284`):
- `BoundaryClosureOperator` factors `K_bc = G · R · P` where:
  - `P : V → A` (escape tensor, geometry physics)
  - `R : A → A` (reflection on mode space, BC physics)
  - `G : A → V` (response tensor, geometry physics)
- "Every BC flavour (vacuum, reflective, white-Mark, white-Marshak DP_N, albedo, interface current) is a CHOICE OF R" — explicit design statement. Specular fits as **a new R operator** without any geometry-side changes.
- Existing R helpers: `reflection_vacuum` (R=0), `reflection_mark` (rank-1 isotropic), `reflection_marshak` (Gelbard diagonal). Missing: `reflection_specular`.

**Existing closure dispatch** (`peierls_geometry.py:_build_full_K_per_group`, lines 4400-4540):
- Dispatches on `closure ∈ {"vacuum", "white_rank1_mark", "white_hebert", "white_f4"}`. Adding `"specular"` is a new branch with the standard `BoundaryClosureOperator` assembly using a new R.

**Davison method-of-images investigation** (commit 30335f2):
- Diagnostic scripts at `derivations/diagnostics/diag_sphere_davison_image_{01..04}_*.py`.
- Image positions for sphere: `{2nR + r', 2nR - r' : n ∈ ℤ}` with signs `(-1)^|n|`.
- Convergence: saturates at ~5 image terms (exponential decay).
- Falsification verdict was *for white BC*: image series converges to k = 0.704 vs cp_sphere k_inf = 1.5 for sphere 1G/1R fuel A (-53% off). **Diagnosed as: image series solves the SPECULAR-reflection problem instead.** This plan converts that diagnosis into a feature.

**Memory references**:
- `.claude/agent-memory/numerics-investigator/issue_132_davison_image_series.md` — has the image positions, sign convention, sanity-gate verification (vacuum-BC limit recovers correctly).
- `.claude/agent-memory/literature-researcher/sanchez_1977_nse64_canonical_ic.md` — confirms Sanchez 1977 has no K_∞ for white but discusses generic boundary operators.
- `.claude/agent-memory/literature-researcher/sanchez_mccormick_rank_n_per_face.md` — Sanchez-McCormick 1982 §III.F covers the per-face mode space; specular is a special case.

### 2.2 Why specular is a different BC class than white

For a surface point r_b with outward normal n_b:

- **Vacuum**: `ψ⁻(r_b, Ω) = 0` for all incoming Ω. No closure needed.
- **White (Mark)**: `ψ⁻(r_b, Ω) = J⁺(r_b)/π` for incoming Ω. *Averages* the angular distribution before reflecting. **Approximation** — the actual angular distribution at the surface is NOT necessarily isotropic.
- **Specular**: `ψ⁻(r_b, Ω) = ψ⁺(r_b, Ω̃)` where `Ω̃ = Ω - 2(Ω·n_b)n_b` is the mirror direction. **Exact** — no averaging, full angular distribution preserved.

For a **homogeneous** cell with rotational/translational symmetry, the eigenvector is uniform and the surface flux IS isotropic, so Mark = specular = white = exact. They all agree.

For a **heterogeneous** cell with non-trivial source distribution, the surface flux is anisotropic. Mark approximates this as isotropic; specular preserves the actual anisotropy. **Specular and white k_eff diverge in direct proportion to the Mark closure error.**

### 2.3 Why method-of-images is the natural specular implementation

For specular reflection on a flat surface (slab), method-of-images is exact: the reflected angular flux at observer r equals the unreflected angular flux at the mirror image of r across the surface. For a SPHERE with white BC, this fails (white is averaging; mirror sources can't reproduce). For a SPHERE with SPECULAR BC, method-of-images works AGAIN — each direction maps to its mirror direction across the local outward normal, and the spherical inversion `r' → R²/r'` (Kelvin transformation) provides the geometric image.

The image series for sphere with specular BC at outer surface r=R AND vacuum at center r=0 (Davison u=rφ=0 boundary):

```
K_image(r, r') = Σ_{n ∈ ℤ} (-1)^|n| · [E_1(Σ_t·|r - 2nR - r'|) - E_1(Σ_t·|r - 2nR + r'|)]
```

This is *exactly* the kernel I derived in the Davison investigation. The agent's negative result ("converges to wrong eigenvalue under white BC") becomes a positive result here: the converged eigenvalue IS the specular-BC eigenvalue, which is what we want.

**Important caveat from the investigator**: for multi-region σ_t, image points like `-r'` and `2R+r'` lie OUTSIDE the physical sphere. Material extension is needed — likely "specular-mirror also reflects σ_t across the surface" (since the image source represents a real source that *would have* been there if the cell were extended periodically). This is a subtle modeling choice that needs SymPy verification before implementation.

### 2.4 Why the mode-space path is also needed

Method-of-images works cleanly for sphere (radial symmetry → 1-D image series) but is harder to set up for cylinder (axial + azimuthal → 2-D image lattice) and was never derived in the Davison investigation for cylinder. The mode-space path (`R_specular` matrix) is **geometry-agnostic** — it works for any geometry where the surface modes are defined.

For the Marshak shifted-Legendre basis on the outgoing hemisphere `µ ∈ [0, 1]`:
- Mode 0: `P̃_0(µ) = 1` (isotropic component)
- Mode n: `P̃_n(µ) = P_n(2µ - 1)` (shifted Legendre)

Specular reflection maps outgoing direction `µ_out > 0` to incoming direction `µ_in = -µ_out < 0`. In the mode basis this is:
- `R_specular[m, n] = ∫_0^1 P̃_m(µ) · P̃_n(µ) dµ` (under the µ → −µ → µ identity for shifted Legendre on [0, 1]) — the inner product of mode-m and mode-n in the half-range basis. Actually trivially diagonal because shifted Legendre IS orthogonal under uniform half-range measure.
- More carefully: specular flips the sign of µ; for shifted Legendre on `[0, 1]` (where `µ` is the absolute cosine with the surface normal), the mode-n basis function evaluated at the SAME `µ` for both outgoing and incoming gives the mapping `R[n, n] = +1` for even n, `-1` for odd n (parity). So `R_specular = diag(+1, -1, +1, -1, ...)` in the Marshak basis.

This needs to be SymPy-verified for the actual basis convention used in `_shifted_legendre_eval`.

If the simple `diag(+1, -1, +1, -1, ...)` form is right, the implementation is **trivially small** — about 5 lines of code for `reflection_specular`.

The cross-verification: sphere mode-space specular k_eff should equal sphere image-series specular k_eff to machine precision. This is the load-bearing check.

## 3. The hypothesis under test

**The two specular implementations agree to machine precision, AND their common k_eff equals cp_sphere/cp_cylinder k_inf for homogeneous cells AND quantifies the CP closure error for heterogeneous cells.**

Specifically:

- **H_A**: For homogeneous Class B cells (1G/1R or any uniform-σ_t configuration), mode-space specular k_eff = image-series specular k_eff = cp k_inf to machine precision. Confirms (a) two implementations agree, (b) specular = white at homogeneous (by symmetry), (c) cp solver is bit-correct on homogeneous.
- **H_B**: For heterogeneous Class B cells (1G/2R fuel-mod), mode-space specular k_eff = image-series specular k_eff (still to machine precision — they're the same physics). The COMMON value diverges from cp k_inf by some amount X. **X is the CP Mark closure error**, measured directly.
- **H_C** (the verification payoff): the divergence X for heterogeneous Class B is in the range expected from the Sanchez 1977 LWR empirical bound (~1-3 % for typical geometries; up to ~10-50 % for extreme small-fuel test cells). If X agrees with this expected magnitude, **the specular Peierls is the correct L1 reference for cp solver verification**.

If H_A fails: bug in one of the specular paths. Fix and retry.
If H_B fails (the two paths disagree on heterogeneous): bug in one of them or the multi-region material extension is wrong. Investigate.
If H_C: X >> Sanchez bound (e.g., +100%): something else is wrong; investigate before declaring victory.

## 4. Architecture — what exists, what needs to change

### 4.1 What's already wired

- `BoundaryClosureOperator` framework (line 3184) — accepts arbitrary R matrix; `R_specular` plugs in directly.
- `_build_closure_operator_rank_n_white` and adjacent rank-N machinery — provide the P and G primitives that work with any R.
- Davison image-series helpers in `derivations/diagnostics/diag_sphere_davison_image_*.py` — sign convention and image positions already validated; need promotion to production form.
- `peierls_geometry.optical_depth_along_ray` — handles multi-region τ accumulation along arbitrary chord directions; reusable for image-source τ computation.

### 4.2 What needs to be built

**Phase 1 — `R_specular` mode-space derivation + implementation** (Session 1-2):

1. **SymPy derivation** at `derivations/peierls_specular_bc.py`:
   - Verify the parity-sign claim `R_specular = diag(+1, -1, +1, -1, ...)` for the `_shifted_legendre_eval` basis convention.
   - If the basis is normalised differently, derive the correct R_specular matrix. May be diagonal with parity factors, or include extra (2n+1) Gelbard normalisation depending on basis choice.
   - Sanity check: at rank-1 (R = scalar = +1), specular K_bc structure equals white K_bc (because mode-0 is the same and only the sign of higher modes differs).
2. **Implement `reflection_specular(N)`** in `peierls_geometry.py`:
   - Returns the N×N matrix derived above.
   - Located near `reflection_vacuum`, `reflection_mark`, `reflection_marshak`.
3. **Wire `closure="specular"`** into `_build_full_K_per_group`:
   - New branch that builds the closure operator with `reflection=reflection_specular(N)`.
   - For sphere, default rank N=1 should still give K_bc = K_bc^white_rank1_mark (sanity).
   - For cylinder, N=1 specular = N=1 white_mark (also by symmetry — single mode is isotropic).
4. **Foundation tests**:
   - At rank-1, specular k_eff = white_rank1_mark k_eff to machine precision (homogeneous).
   - At rank-N for HOMOGENEOUS cells, specular k_eff converges to k_inf as N → ∞.
   - At rank-N for HETEROGENEOUS cells, specular k_eff converges to a specific value (TBD by computation; that's the new reference).

**Phase 2 — Image-series K_bc primitive for sphere** (Session 2-3):

1. **SymPy derivation** at `derivations/peierls_specular_image_series.py`:
   - Verify the image positions `{2nR ± r' : n ∈ ℤ}` and signs `(-1)^|n|` for **specular** BC (NOT the white BC sign convention the investigator used and falsified).
   - Derive the multi-region material extension: when the image source at `r_image` lies outside the physical sphere `[0, R]`, what σ_t value does the τ accumulation use along the segment from r to r_image? Two candidates:
     - (a) Periodic extension: σ_t(r_image) = σ_t((r_image mod 2R) folded back into [0, R])
     - (b) Specular extension: σ_t reflects across the surface (image of a fuel pellet is a fuel pellet at the mirror position)
   - These two extensions give DIFFERENT k_eff values on heterogeneous cells. Which is "correct" for specular BC needs careful derivation — likely (b), the specular extension, because it preserves the physical interpretation that the specular reflection sees a virtual source on the other side of a mirror.
2. **Implement `compute_K_bc_specular_image_sphere(r_nodes, radii, sig_t, n_images)`** in `peierls_geometry.py`:
   - Sums the truncated image series with (b)-style σ_t extension.
   - n_images defaults to 10 (saturates by 5 per the investigator's convergence study).
3. **Wire as alternative path**: add `closure="specular_image"` (or extend `closure="specular"` with a `method="image"` keyword) that uses the image-series K_bc instead of the mode-space form.
4. **Cross-verification gate** (the core load-bearing test):
   - Mode-space specular k_eff vs image-series specular k_eff agreement to machine precision for sphere 1G/1R, 1G/2R, 2G/2R.
   - Disagreement → investigate; likely material-extension choice is wrong.
   - Agreement → BOTH are correct, BOTH are the new L1 reference.

**Phase 3 — Cylinder specular** (Session 3-4):

1. **Mode-space specular for cylinder** is straightforward — same `R_specular` parity-diagonal works regardless of geometry (it's a property of the basis, not the geometry). Wire through `boundary="specular"` for cylinder.
2. **Image-series for cylinder** is HARDER:
   - The cylinder has azimuthal AND axial symmetry → 2-D image lattice
   - Per the Davison investigator's reluctance: "for multi-region σ_t, the breakdown is even more severe" — even harder for cylinder
   - **Defer to Phase 4** if mode-space cylinder gives clean results; only pursue image-series for cylinder if mode-space disagrees with cp k_inf (would need cross-check)

**Phase 4 — Heterogeneous Class B verification** (Session 4):

1. **Run specular BC on canonical test cases**:
   - sphere 1G/1R, 1G/2R, 2G/1R, 2G/2R
   - cylinder 1G/1R, 1G/2R, 2G/1R, 2G/2R
2. **Compare against**:
   - cp_sphere/cp_cylinder k_inf (the current L1 reference, which has Mark closure error)
   - Hébert white_hebert k_eff (this branch's improvement, which has documented Mark uniformity overshoot on heterogeneous)
   - Optionally: ORPHEUS MC reference if available (truly independent)
3. **Document the verification verdict**:
   - Homogeneous: all four references should agree to machine precision (sanity)
   - Heterogeneous: specular gives the angularly-exact pointwise eigenvalue; cp k_inf and white_hebert each diverge from it by their respective closure errors; the divergence MEASURES the closure error

**Phase 5 — Sphinx + Issues + commit** (Session 5):

1. New section `:ref:peierls-specular-bc` in `peierls_unified.rst`:
   - Tensor-network framework: specular as another R choice
   - Mode-space derivation (SymPy results)
   - Image-series derivation (Davison construction with specular sign)
   - Cross-verification table
   - Heterogeneous Class B verification table
   - Connection to method-of-images literature (slab analog, sphere Kelvin transformation)
2. Issue #100, #103, #132 close with specular as the resolution path (or update with the new evidence)
3. Issue #114 may also benefit (specular gives a different sensitivity to the ρ-quadrature subdivision noise — would tighten constraints)
4. Foundation tests promoted from diagnostics to `tests/derivations/test_peierls_specular_bc.py`

## 5. Test plan

### 5.1 Test fixtures

Two parallel fixtures, one per implementation path:

- **`SphereSpecularModeSpaceFixture(ng_key, n_regions, n_bc_modes)`** — uses `boundary="specular"` (new mode-space branch).
- **`SphereSpecularImageSeriesFixture(ng_key, n_regions, n_images)`** — uses `boundary="specular_image"` (new image-series branch, sphere only).

### 5.2 Test classes

**A. Sanity baselines** (Session 1-2):

- `TestSpecularRank1EqualsWhiteMark` — at `n_bc_modes=1`, specular K_bc bit-equals white_rank1_mark K_bc (both use only mode 0 which is isotropic). Foundation gate.
- `TestSpecularHomogeneousConvergesToKinf` — sphere/cyl 1G/1R, 2G/1R: specular k_eff converges to k_inf as N → ∞ at rank ≤ 8. Foundation gate.

**B. Cross-verification** (Session 2-3, the load-bearing tests):

- `TestSpecularModeSpaceVsImageSeries[sphere_1G_1R]` — bit-equal at 1e-12 dps=30.
- `TestSpecularModeSpaceVsImageSeries[sphere_1G_2R]` — bit-equal at 1e-12 dps=30.
- `TestSpecularModeSpaceVsImageSeries[sphere_2G_2R]` — bit-equal at 1e-12 dps=30.

If these pass: BOTH paths verified, either is the new L1 reference.

**C. Heterogeneous Class B verification** (Session 4):

- `TestSpecularVsCPSphere[ng_1G_2R_heterogeneous]` — the divergence measures CP Mark closure error.
- `TestSpecularVsHebertSphere[ng_1G_2R_heterogeneous]` — divergence measures the Hébert overshoot quantitatively.
- (Optionally) `TestSpecularVsMonteCarlo[ng_1G_2R_heterogeneous]` — if MC L1 harness exists, specular vs MC at <0.1 % statistical bound.

### 5.3 Quadrature posture

Same as plan §5.3 of the Class B MR×MG plan: BASE for fast iteration, RICH for stability check. ULTRA off-suite.

For the image-series, n_images is the convergence parameter (saturates at ~5-10 per the investigator's study).

## 6. Decision tree

| Outcome | Trigger | Action |
|---------|---------|--------|
| **A — full success** | Mode-space specular = image-series specular bit-exact; both equal cp k_inf for homogeneous; both diverge from cp k_inf by ~Sanchez-bounded amount on heterogeneous | Ship as production specular BC; document as L1 reference for CP verification; close Issues #100/#103/#132. |
| **B — implementations disagree** | Mode-space and image-series specular k_eff differ by > machine ε on heterogeneous | Investigate via probe-cascade. Likely cause: material-extension convention for image series. Resolve before shipping; the disagreement is the bug signal. |
| **C — both paths agree but diverge wildly from cp** | Both specular k_eff agree to machine ε, but the common value differs from cp k_inf by >> Sanchez bound (e.g., +100 %) | Possible: my parity-sign R_specular is wrong (mode-space) AND material-extension is wrong (image-series), with errors that cancel between them. Investigate by direct mode-by-mode comparison against an independent specular implementation (Sn or MOC if available). |
| **D — partial success** | Sphere works (Phase 1+2 lands), cylinder doesn't (Phase 3 stuck on image-series 2-D lattice) | Ship sphere specular; document cylinder as future work; cylinder Class B verification falls back to mode-space specular (may have small residual since image-series cross-check unavailable). |

## 7. Acceptance criteria

The plan is complete when:

- (a) **`reflection_specular(N)`** shipped in `peierls_geometry.py` with SymPy-derived form (parity diagonal or other).
- (b) **`closure="specular"`** routes through `_build_full_K_per_group` for sphere AND cylinder; foundation tests pass.
- (c) **Image-series K_bc primitive** for sphere shipped (multi-region material extension SymPy-verified).
- (d) **Cross-verification gate**: mode-space and image-series specular agree to machine precision (1e-12 dps=30) on sphere 1G/1R, 1G/2R, 2G/2R. THIS IS THE LOAD-BEARING DELIVERABLE.
- (e) **Heterogeneous Class B verification table**: specular vs cp k_inf vs white_hebert vs (optionally) MC for sphere AND cylinder 1G/2R, 2G/2R.
- (f) **Sphinx section** `:ref:peierls-specular-bc` documents the tensor-network framework, the two implementations, the cross-verification proof, and the new L1 reference status.
- (g) **Tests**: `tests/derivations/test_peierls_specular_bc.py` with foundation + cross-verification + heterogeneous-verification gates.
- (h) **Issues #100, #103, #132 updated** with the new closure as the resolution path.

## 8. Risks and prerequisites

### 8.1 Prerequisites

- **Branch hygiene**: branch `feature/rank-n-class-b-mr-mg` (current) merged or rebased before starting. Otherwise the specular work piles 100+ commits ahead of main.
- **mpmath dps=30+ infrastructure**: needed for cross-verification at machine precision. Already available via `K_vol_element_adaptive` pattern.
- **`_shifted_legendre_eval` basis convention documented**: needed to derive R_specular correctly. Currently in `_kernels.py:383`, but the normalisation needs SymPy verification.

### 8.2 Risks

- **R1 — material-extension convention for image series is non-obvious**: the Davison investigator flagged this. Resolution requires careful first-principles derivation; SymPy verification helps. If wrong, the image-series gives wrong heterogeneous results, and cross-verification fails (Decision tree Outcome B).
- **R2 — R_specular may not be diagonal in the existing basis**: my "parity diagonal" guess depends on the shifted Legendre basis being orthogonal under the half-range µ measure with the conventional sign assignment. If the basis is normalised differently, R_specular has off-diagonal entries and the implementation grows. SymPy can resolve this in a few lines.
- **R3 — image-series for cylinder is genuinely harder**: the 2-D image lattice (azimuthal × axial) doesn't have the clean "1-D periodic structure" sphere has. May require a separate research session to derive. **Mitigation**: ship sphere first (Phase 1+2), defer cylinder image-series (Phase 3 part 2).
- **R4 — specular and cp k_inf may agree exactly** on heterogeneous cells: if so, the verification mode change is moot — cp was already correct, and the Hébert white_hebert overshoot was just a Peierls-Mark-closure artifact, not a CP error. This would be the BEST outcome for the codebase but the LEAST informative. Mitigation: it's still a positive result; document it cleanly.
- **R5 — cross-verification at 1e-12 may be too tight at rank-N high modes** due to Vandermonde conditioning in the Marshak basis. Mitigation: verify at rank-1 and rank-2 first; relax tolerance to 1e-8 at rank > 4 if needed.

### 8.3 Non-goals

- **Not** trying to make specular k_eff equal cp k_inf on heterogeneous cells. They're DIFFERENT physics (specular is exact angular preservation; cp is Mark-isotropic flat-flux). The divergence is the verification payload.
- **Not** implementing image-series for cylinder unless mode-space cylinder needs it for cross-verification.
- **Not** addressing Issue #114 (ρ-quadrature subdivision) directly — though specular's sensitivity to it may surface new evidence.
- **Not** building a MOC or Sn comparison — that's a separate verification harness (would be Path 1 from the previous discussion). This plan stays inside the Peierls framework.

## 9. Estimated budget

- **Best case (Outcome A)**: 4 sessions. S1 = R_specular SymPy + mode-space implementation + foundation tests. S2 = image-series K_bc derivation + sphere implementation + cross-verification gate. S3 = cylinder mode-space specular + heterogeneous verification table. S4 = Sphinx + Issues + commit.
- **Mid case (R2 or R3 fires)**: 5 sessions (add a derivation session for non-diagonal R_specular OR cylinder image-series).
- **Worst case (R1 fires + iteration on material extension)**: 6 sessions.
- **Outcome D (partial — sphere only)**: 3-4 sessions.

LoC delta:

- New `reflection_specular` + dispatch: ~40 LoC.
- Image-series K_bc primitive (sphere): ~80 LoC.
- SymPy derivation scripts (kept in `derivations/`): ~200 LoC × 2.
- Tests: ~300-500 LoC.
- Sphinx: ~200-400 lines depending on outcome.

Commit count: 4-6, scaled by outcome.

---

## Critical files for implementation

- `orpheus/derivations/peierls_geometry.py` — `BoundaryClosureOperator` framework + reflection helpers + `_build_full_K_per_group` dispatch; THE central file
- `orpheus/derivations/_kernels.py:_shifted_legendre_eval` (line 383) — basis convention for R_specular derivation
- `derivations/diagnostics/diag_sphere_davison_image_{01..04}_*.py` — image-series scaffolding to resurrect (with corrected sign convention for specular)
- `derivations/peierls_specular_bc.py` (NEW) — SymPy R_specular derivation
- `derivations/peierls_specular_image_series.py` (NEW) — SymPy image-series + material-extension derivation
- `tests/derivations/test_peierls_specular_bc.py` (NEW) — the test deliverable
- `docs/theory/peierls_unified.rst` — Sphinx update
- `.claude/agent-memory/numerics-investigator/issue_132_davison_image_series.md` — image-series sign convention + convergence reference
