# Numerical Notes — Discrete Ordinates Solver

Observations from verification and benchmarking of the SN solver.
To be incorporated into the Sphinx documentation.

---

## Quadrature on 1D meshes

The sweep dispatcher routes `ny=1` meshes with Gauss-Legendre quadrature
to a fast 1D cumprod path.  Lebedev quadrature on a `ny=1` mesh falls
through to the 2D wavefront sweep, which handles it correctly:

- Ordinates with `μ_x ≠ 0` stream along x; y-streaming terms cancel
  via reflective BCs on the single-cell y-dimension.
- Ordinates with `μ_x = μ_y = 0` (z-directed) are handled as pure
  collision: `ψ = Q·weight_norm / Σ_t`.

Both quadratures recover the analytical eigenvalue exactly on
homogeneous problems (verified to machine precision for 1G/2G/4G).

---

## keff sensitivity to solver choices (421-group heterogeneous PWR slab)

All cases: 10 cells, `δ = 0.2 cm`, material layout `[fuel×5, clad×1, cool×4]`,
P0 scattering, 421 energy groups.

| Configuration | keff | Notes |
|---|---|---|
| 1D GL S16, BiCGSTAB (FD operator) | 1.03882 | True 1D, 16 ordinates |
| 1D Lebedev 110, source iteration (DD sweep) | 1.04294 | 1D mesh, 2D quadrature |
| 2D (10×2) Lebedev 110, source iteration (DD sweep) | 1.04294 | Pseudo-2D, full volumes |
| 2D (10×2) Lebedev 110, BiCGSTAB (FD operator) | 1.04007 | Pseudo-2D, full volumes |
| 2D (10×2) Lebedev 110, BiCGSTAB (FD operator), half-volumes | 1.04192 | MATLAB convention |
| **MATLAB reference** | **1.04188** | 2D Lebedev, FD operator, half-volumes |

### Sources of variation

1. **Angular quadrature** (GL vs Lebedev): ~0.004 difference.
   GL S16 integrates 1D angular flux with 16 points on `[-1,1]`.
   Lebedev 110 integrates over the unit sphere — more angular resolution
   but different effective weights per `μ_x` direction.  On a coarse
   heterogeneous mesh, these give different eigenvalues.

2. **Spatial discretization** (DD sweep vs FD gradient): ~0.003 difference.
   Source iteration uses the diamond-difference wavefront sweep (`T⁻¹`).
   BiCGSTAB uses the explicit finite-difference transport operator (`T`).
   Both are O(h) on this mesh but with different truncation error constants.
   They converge to the same answer as h → 0.

3. **Boundary volume weighting**: ~0.002 difference (full vs half).
   The MATLAB code halves boundary cell volumes.  With `ny=2` and
   materials uniform in y, the y-direction halving (`vol[:,0] /= 2`,
   `vol[:,-1] /= 2`) scales all volumes equally and cancels in the
   keff ratio.  Only the x-direction halving (fuel edge, coolant edge)
   affects keff.  This is an artifact of the pseudo-2D implementation:
   a true 1D calculation has no y-volumes.

4. **Inner convergence**: source iteration with `max_inner=200`,
   `inner_tol=1e-8` does not fully converge for 421 groups
   (spectral radius ≈ 0.97).  BiCGSTAB fully converges the inner
   solve in ~100 Krylov iterations.  Both give the same outer-converged
   keff when the inner solve converges.

### Matching the MATLAB result

The MATLAB code uses:
- 2D Lebedev 110 quadrature on a 10×2 mesh
- Explicit FD transport operator with BiCGSTAB inner solve
- Boundary half-volumes on all edges
- P0 scattering, `bicgstab_tol = 1e-4`, `maxiter = 2000`

Our BiCGSTAB path with half-volumes reproduces 1.04192 vs MATLAB's 1.04188
(4×10⁻⁵ agreement).  The residual difference is from floating-point
details in the cross-section processing.

### True 1D reference

The cleanest reference is the 1D GL BiCGSTAB result (1.03882): no
pseudo-2D artifacts, well-conditioned angular quadrature, fully
converged inner solve.  The 2D Lebedev results on a 1D-invariant
mesh include angular quadrature effects from the unused y-direction.

---

## Normalization chain for the SN equation

The isotropic SN transport equation for ordinate n:

    μ_n · ∇ψ_n + Σ_t · ψ_n = Q / sum(w)

where `Q` is the total isotropic source (fission + scattering + n2n)
and `sum(w)` is the quadrature weight sum (4π for Lebedev, 2 for GL).

The normalization chain in the code:

1. **Fission source** (`compute_fission_source`):
   `Q_f = χ · (νΣ_f · φ) / k` — raw, un-normalized.

2. **Scattering source** (`_add_scattering_source`):
   `Q_s = Σ_s^T @ φ` — also un-normalized.

3. **Sweep** (`transport_sweep`):
   applies `Q_scaled = Q * weight_norm` where `weight_norm = 1/sum(w)`.
   This is the `1/sum(w)` division in the SN equation.

4. **Scalar flux** (inside sweep):
   `φ = Σ_n w_n · ψ_n` — standard quadrature integration.

5. **keff** (`compute_keff`):
   `k = (νΣ_f · φ · V) / (Σ_a · φ · V)` — volume-weighted ratio.

The `1/sum(w)` in step 3 and the `sum(w)` implicit in step 4 cancel:
`φ = Σ w_n · Q/(sum(w)·Σ_t) = Q/Σ_t` for uniform isotropic source.
This cancellation is verified by `test_homogeneous_scalar_flux_equals_Q_over_sigt`.

**Convention rule:** sources passed to the sweep must NOT include
`1/sum(w)` — the sweep applies it.  The BiCGSTAB path (direct operator)
must divide sources by `sum(w)` itself, since it solves `Tψ = b`
without the sweep.

---

## Scattering matrix convention

The `Mixture.SigS[l]` matrices use the convention `SigS[g_from, g_to]`:

    SigS[0] = [[Σ_{0→0}, Σ_{0→1}],
               [Σ_{1→0}, Σ_{1→1}]]

For the in-scatter source (total scattering into group g from all groups g'):

    Q_scatter[g] = Σ_{g'} Σ_{g'→g} · φ_{g'} = (SigS^T @ φ)[g]

The vectorized form for batched cells: `phi @ SigS` (equivalent to
`(SigS^T @ phi^T)^T` for row vectors).

The analytical eigenvalue problem uses:

    A = diag(Σ_t) - SigS^T    (removal matrix, NOTE: transposed)
    F = outer(χ, νΣ_f)        (fission matrix)
    k_inf = λ_max(A⁻¹F)

The transpose in A is because `SigS^T[g, g'] = SigS[g', g] = Σ_{g'→g}`
gives the in-scatter contribution, so `diag(Σ_t) - SigS^T` removes
the in-scatter from the total to get the net removal.

---

## Why 1-group verification is degenerate

For 1 energy group, the eigenvalue is:

    k = νΣ_f / Σ_a

This is a scalar ratio independent of the spatial or angular flux
distribution.  Consequences:

- Weight loss (z-ordinate bug) scales all flux equally → cancels in k
- Wrong scattering convention → no inter-group coupling to distort
- Wrong flux shape → doesn't matter, k is a material property

Only multi-group problems have a flux-shape-dependent eigenvalue:
`k = (νΣ_f · φ) / (Σ_a · φ)` where the dot product weights each
group differently.  A wrong group ratio (from angular errors,
normalization errors, or convergence failures) directly shifts keff.

**Rule:** Every transport solver must be verified on at least 2-group
problems.  1-group success gives false confidence.

---

## Two inner solver architectures

### Source iteration (sweep-based)

- **Operator:** `T⁻¹` (diamond-difference sweep)
- **Solution variable:** scalar flux `φ(x, y, g)`
- **Fixed-point:** `φ^{k+1} = T⁻¹(S·φ^k + Q_f)`
- **Convergence rate:** spectral radius of `T⁻¹S` (~0.97 for 421 groups)
- **Cost per iteration:** one transport sweep (~40 ms for 10×10×421)
- **Iterations needed:** ~200 for 6 decades (does not converge within
  `max_inner=200` at `tol=1e-8` for 421 groups)

### BiCGSTAB (direct operator)

- **Operator:** `T = μ·∇ + Σ_t` (finite-difference gradients)
- **Solution variable:** angular flux `ψ(x, y, n, g)` (much larger)
- **System:** `T·ψ = b` where `b` = fission + scattering + n2n
- **Convergence rate:** depends on condition number of T (well-conditioned)
- **Cost per iteration:** one matvec (comparable to one sweep)
- **Iterations needed:** ~100 at `tol=1e-4` (always converges)

The two architectures use **different spatial discretizations**
(diamond-difference vs finite-difference) that converge to different
keff on coarse meshes.  They agree in the limit h → 0.

---

## P1 scattering: available but unused

The 421-group cross-section library provides both P0 and P1 scattering
matrices (`fuel.SigS[0]` and `fuel.SigS[1]`, each 421×421).
The verification XS library has only P0.

The MATLAB code defaults to `L=0` (P0) but has the spherical harmonic
infrastructure for P1: Legendre moments of the angular flux are
computed via `fiL`, and the scattering source uses
`(2l+1) · Σ_s^l @ f_l / sum(w)` for each order l.

The current Python solver:
- Accepts `scattering_order` parameter (stored but not used)
- Only reads `SigS[0]` in `__init__`
- Source iteration path would need angular moment computation
  (not available from the sweep, which returns only scalar flux)
- BiCGSTAB path could support P1 more naturally since it works with
  angular flux directly

Implementing P1: the sweep-based path would need to store the angular
flux per ordinate (already returned by the sweep) and compute first
moments `f_1^m = Σ_n w_n · ψ_n · R_n^{1,m}` after each sweep.
The BiCGSTAB path would extend `build_rhs` to include the `l=1` term.

---

## RESOLVED: BiCGSTAB convergence on curvilinear geometries

**Status**: Fixed in commit `22723c8` (spherical) and `0ce2621` (cylindrical).

### Original problem

Multi-group BiCGSTAB on spherical geometry diverged (keff → NaN).

### Root cause (corrected)

The original hypothesis (central vs upwind inconsistency) was only
partially right.  The deeper issue was the **missing ΔA/w geometry
factor** in the explicit FD operator.  Without it, the angular
redistribution lacked per-ordinate flat-flux consistency, causing
the outer iteration to amplify errors.

### Fix applied

Added the same ΔA/w factor and Morel–Montry angular closure weights
to the BiCGSTAB operator that the sweep uses.  Both spherical and
cylindrical operators now read `redist_dAw` and `tau_mm` from SNMesh.

2G and 4G spherical BiCGSTAB converge to < 1e-6 of analytical.
Cylindrical BiCGSTAB matches source iteration to machine precision.

### Remaining improvements

See `IMPROVEMENTS.md` for the central tracker.  Related items:
- DO-00000000-001 — DSA (highest-value performance improvement)
- DO-00000000-002 — TSA (for anisotropic media)
- DO-00000000-007 — GMRES/preconditioned Krylov

---

## RESOLVED: Cylindrical azimuthal DD

**Status**: Fixed.  See `TODO_cylindrical_dd.md` for full details.

The root cause was NOT the sign convention as originally hypothesized.
It was a wrong α recursion (`cumsum(+w·ξ)` instead of `cumsum(−w·η)`)
combined with a missing ΔA/w geometry factor in the balance equation.
Both bugs broke per-ordinate flat-flux consistency.  Fixed using the
Bailey et al. (2009) formulation.
