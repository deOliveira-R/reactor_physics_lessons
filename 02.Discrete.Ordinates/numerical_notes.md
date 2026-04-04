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
