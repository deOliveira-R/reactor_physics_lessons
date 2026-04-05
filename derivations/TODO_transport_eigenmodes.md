# DO-00000000-005: 1D Transport Eigenmode Expansion (Case's Method)

> Tracked in `02.Discrete.Ordinates/IMPROVEMENTS.md`

## Background

The SN heterogeneous verification cases currently use Richardson
extrapolation from the 1D SN solver (O(h²) diamond-difference, S16).
This gives the correct **transport** eigenvalue but depends on running
the solver at multiple mesh refinements — it's a self-referencing test.

A truly independent reference requires solving the 1D multi-group
transport equation analytically (or semi-analytically). Two approaches
were explored:

### Diffusion transfer matrix (implemented, `sn_heterogeneous.py`)

Solves the multi-group diffusion equation with reflective BCs using
transfer matrices (matrix exponential per region, interface matching,
brentq root-finder). Gives results to 1e-12 precision, mesh-independent.

**Limitation:** Diffusion ≠ transport. The transport correction is
~0.003 for our benchmark cases (0.3% effect). This makes the diffusion
reference unsuitable as a precision target for the SN solver, but it
serves as a cross-check and an upper/lower bound.

### Comparison: diffusion vs SN Richardson

| Case | Diffusion | SN Richardson | Diff |
|------|-----------|---------------|------|
| 1G 2-region | 1.2646 | 1.2605 | +0.0041 |
| 1G 4-region | 1.1716 | 1.1675 | +0.0041 |
| 2G 2-region | 1.2338 | 1.2380 | -0.0042 |
| 2G 4-region | 1.0461 | 1.0497 | -0.0036 |
| 4G 2-region | 1.0312 | 1.0344 | -0.0032 |
| 4G 4-region | 0.8805 | 1.0831 | -0.0026 |

Note: diffusion can be either above or below transport depending
on the material configuration and energy group structure.

## Case's Eigenmode Method

The exact solution of the 1D monoenergetic transport equation with
isotropic scattering in a homogeneous medium is given by Case's
singular eigenfunctions (1960):

### The equation

    μ dψ/dx + Σ_t ψ = (c Σ_t / 2) ∫₋₁¹ ψ dμ'

where c = Σ_s / Σ_t is the scattering ratio (number of secondaries
per collision).

### Eigenmodes

The solution ψ(x, μ) = φ_ν(μ) exp(-x/ν) has eigenmodes:

1. **Discrete modes** (ν₀): exist for |ν₀| > 1/Σ_t. Found from the
   dispersion relation:

       1 = (c/2) ∫₋₁¹ dμ / (1 - μ/(ν₀ Σ_t))
         = c ν₀ Σ_t arctanh(1/(ν₀ Σ_t))

   For c < 1 (subcritical): two discrete modes ±ν₀
   For c > 1 (supercritical): no real ν₀

2. **Continuum modes** (ν ∈ [-1/Σ_t, 1/Σ_t]):
   
       φ_ν(μ) = (c/2) P.V.(1/(ν - μ/Σ_t)) + λ(ν) δ(ν - μ/Σ_t)

   where λ(ν) ensures the normalization.

### Multi-group extension

For G groups, the eigenmode expansion becomes a matrix problem.
The discrete eigenvalues ν₀ are roots of:

    det[I - (c Σ_t / 2) ∫₋₁¹ dμ (νΣ_t I - μ I)⁻¹] = 0

which for multi-group becomes:

    det[I - Σ_{s}^T · diag(1/Σ_t) · Λ(ν)] = 0

where Λ(ν) involves arctanh integrals.

### Interface matching

At each material interface, **all** angular moments must be matched
(half-range flux continuity):

    ∫₀¹ μⁿ ψ_left(x_i, μ) dμ = ∫₀¹ μⁿ ψ_right(x_i, μ) dμ

This is an infinite set of conditions, truncated at a finite order
for numerical solution. The matching gives a system whose determinant
must vanish — the eigenvalue condition for k.

### Reflective BCs

At x = 0: ψ(0, μ) = ψ(0, -μ) for μ > 0
At x = L: ψ(L, μ) = ψ(L, -μ) for μ > 0

These are half-range conditions on the angular flux.

### Implementation requirements

1. **Dispersion relation solver**: find discrete eigenvalues ν₀ for
   each region and each k-guess. For multi-group, this is a matrix
   root-finding problem.

2. **Continuum mode handling**: the singular eigenfunctions require
   Cauchy principal value integrals. Can be evaluated via Gauss-Legendre
   quadrature with subtraction of the singularity.

3. **Half-range moment matching**: truncate at order M, giving
   M+1 conditions per group per interface. Need M large enough for
   convergence.

4. **Nested root-finding**: outer loop on k (brentq), inner loop
   computes determinant of matching system for each k.

### Key references

- Case, K.M. (1960). "Elementary solutions of the transport equation
  and their applications." Annals of Physics 9(1):1-23.

- Case, K.M. and Zweifel, P.F. (1967). "Linear Transport Theory."
  Addison-Wesley.

- Siewert, C.E. (2000). "A concise and accurate solution to
  Chandrasekhar's basic problem in radiative transfer." JQSRT 64:109-130.

- Garcia, R.D.M. and Siewert, C.E. (various). Multi-group extensions
  and computational methods.

### Estimated effort

- 1-group, 1-region: straightforward (1-2 hours)
- 1-group, multi-region: moderate (half day)
- Multi-group, multi-region: significant (1-2 days)
- Full Pn scattering anisotropy: research-level

### Alternative: Method of Singular Eigenfunctions (F_N method)

The F_N method (Siewert, Garcia) provides a practical computational
framework for Case's method. It approximates the half-range angular
flux using a finite expansion in Chandrasekhar polynomials, then
matches at interfaces. This avoids explicit computation of the
continuum modes.

For N=20 the F_N method gives ~10 digits of accuracy, which is
sufficient for our verification needs.

## Diffusion reference (available now)

The `sn_heterogeneous.py` module provides the diffusion transfer
matrix eigenvalue for all 6 heterogeneous cases. This can be used:

1. As cross-check for the SN solver (expected ~0.3% difference)
2. As the primary reference for the diffusion solver verification
   (should match to machine precision)
3. As a consistency check: SN keff should be between diffusion keff
   and the homogeneous k_inf of the dominant region
