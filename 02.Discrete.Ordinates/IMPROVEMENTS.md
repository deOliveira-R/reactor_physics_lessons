# SN Solver Improvement Tracker

Central registry of all known improvements.  Each item is in one of
two states:

- **OPEN**: documented here with all available context for future work
- **DONE**: implemented and documented in Sphinx (`docs/theory/discrete_ordinates.rst`)

Items move from OPEN → DONE only when the Sphinx documentation is
complete (full derivation, rationale, gotchas, numerical evidence).

---

## DONE — implemented, needs Sphinx documentation

These items are implemented and tested but the Sphinx theory chapter
has not yet been updated with full context.  **This is blocking** —
the knowledge exists only in code comments and memory, not in the
permanent documentation.

### D1. Geometry-weighted balance equation (Bailey et al. 2009)

**Commits**: `fb3e976` through `5e00333`

The correct 1D curvilinear balance equation requires:
- α recursion: `α_{m+1/2} = α_{m-1/2} − w_m · η_m` (radial cosine, not azimuthal)
- ΔA/w geometry factor on the redistribution term
- Ordinates η-sorted within each level

Without these, per-ordinate flat-flux consistency is broken, causing
the Morel–Montry flux dip and heterogeneous divergence.

**Sphinx TODO**: full derivation of the balance equation from the
continuous PDE, why the old formulation failed (per-ordinate residual
analysis), Bailey's asymptotic expansion, contamination factor β,
numerical evidence tables.

### D2. Morel–Montry angular closure weights (Bailey Eq. 74)

**Commit**: `c54ad73`, `5e00333`

Weighted diamond difference with τ_m = (η_m − η_{m-1/2})/(η_{m+1/2} − η_{m-1/2})
replaces standard DD (τ = 0.5).  Cell edges at midpoints of consecutive
η values for cylindrical; weight-sum for spherical.  Clamped to [0.5, 1.0].

**Sphinx TODO**: derivation of τ from Bailey's asymptotic analysis,
why it forces β = 0, the cell-edge computation for both geometries,
the alternating τ pattern for Product quadrature.

### D3. BiCGSTAB operators for curvilinear geometries

**Commits**: `22723c8` (spherical), `0ce2621` (cylindrical)

Explicit transport operators with ΔA/w factor and M-M weights.
Multi-group spherical BiCGSTAB now converges (was unstable).
Cylindrical BiCGSTAB added (was missing entirely).

**Sphinx TODO**: operator formulation, FD vs DD face-flux approximation,
why the ΔA/w factor is needed in both implicit and explicit formulations.

### D4. Contamination analysis (derivations/sn_contamination.py)

**Commit**: `8ca174a`, `0ce2621`, `5e00333`

Computes Bailey's β factor and M-M τ weights for any quadrature.
Both geometries give β ≈ 0 (machine zero) with the corrected formulation.

**Sphinx TODO**: definition of β, its physical meaning (contaminated
diffusion equation), how to use the tool for verification.

---

## OPEN — documented for future implementation

### O1. Gauss-type azimuthal quadrature for cylindrical

**Priority**: Medium  
**Effort**: Moderate  
**References**: none specific — standard quadrature construction

The equally-spaced `ProductQuadrature` gives duplicate η values
(paired ±ξ ordinates), producing alternating M-M weights τ = [0.5, 1.0].
A Gauss-type azimuthal quadrature with non-uniform φ spacing would
give distinct η values and smoothly varying τ, potentially improving
angular accuracy for cylindrical sweeps.

**Context**: see `sn_geometry.py:278` and `TODO_cylindrical_dd.md`.

### O2. φ-based cell-edge computation for non-product quadratures

**Priority**: Low  
**Effort**: Small  
**References**: Bailey et al. (2009) Eq. 52

For quadratures where η values are distinct (not equally-spaced φ),
the η-midpoint cell-edge approach is approximate.  Transforming
actual φ cell boundaries to η-space via `η_edge = sin θ · cos(φ_edge)`
would give exact edges.

**Context**: see `sn_geometry.py:275`.

### O3. Diffusion Synthetic Acceleration (DSA)

**Priority**: HIGH  
**Effort**: Large  
**References**: Adams & Larsen (2002), Wareing et al. (various)

The highest-value performance improvement.  Reduces outer (power)
iterations from ~200 to ~20 for many-group problems by using a
diffusion-based correction after each source iteration.

Standard approach:
1. Perform one transport sweep → get angular flux ψ
2. Compute scalar flux residual: δφ = (scattering + fission) − Σ_t·φ
3. Solve a diffusion equation for the correction: −∇·(D∇δφ) + Σ_a·δφ = residual
4. Update φ ← φ + δφ

**Context**: see `numerical_notes.md` acceleration section.
Requires a 1D diffusion solver (already exists in `05.Diffusion.1D/`).

### O4. Transport Synthetic Acceleration (TSA)

**Priority**: Low  
**Effort**: Large  
**References**: Ramone et al. (1997)

Coarse-angle transport solve to accelerate fine-angle source iteration.
More robust than DSA for highly anisotropic media or voids.  Only
needed if DSA proves insufficient for specific problem types.

### O5. Linear Discontinuous (LD) angular finite elements

**Priority**: Low (M-M WDD is sufficient for now)  
**Effort**: Large  
**References**: Bailey, Morel & Chang (2009) — main topic of the paper

Second-order angular accuracy without the DD flux dip.  Solves a
2×2 system per cell-ordinate instead of one equation.  This is the
"gold standard" angular discretization.

**Why not now**: the M-M weighted DD (item D2) already forces β = 0,
eliminating the flux dip.  LD would give higher angular accuracy but
at 2× computational cost.  Worth considering for problems with strong
angular flux gradients.

### O6. Negative flux fixup

**Priority**: Low  
**Effort**: Small  
**References**: standard in production SN codes

Safety net: if WDD produces ψ^a_out < 0, clamp to zero and rebalance.
Currently not needed (zero negatives observed with the corrected
formulation), but good practice for extreme cases (voids, strong
absorbers, coarse angular meshes).

### O7. Transport eigenmodes — Case's method

**Priority**: Medium  
**Effort**: Large  
**References**: Case (1960), Siewert (2000), Garcia & Siewert (various)

Mesh-independent analytical/semi-analytical reference solutions for
1D multi-group transport.  Would provide truly independent verification
(current references are either diffusion-based or self-referencing via
Richardson extrapolation).

**Context**: full description in `derivations/TODO_transport_eigenmodes.md`.

### O8. Anisotropic scattering in curvilinear sweeps

**Priority**: Medium  
**Effort**: Moderate  
**References**: Lewis & Miller Ch. 4, Bell & Glasstone Ch. 5

P1+ anisotropic scattering is implemented for Cartesian 2D (spherical
harmonics on Lebedev quadrature) but NOT verified for curvilinear 1D.
The spherical harmonics expansion on GL (spherical) or Product
(cylindrical) quadrature needs verification and potentially different
moment computation.

### O9. Sphinx theory chapter update for curvilinear SN

**Priority**: HIGH — blocks items D1–D4 from being fully "DONE"  
**Effort**: Moderate  
**Where**: `docs/theory/discrete_ordinates.rst`

The current DO theory chapter was written before the Bailey et al.
formulation was implemented.  It needs a major update covering:
- The correct balance equation with ΔA/w factor
- The α recursion from the radial cosine (not azimuthal)
- The Morel–Montry flux dip analysis and WDD closure
- The contamination factor β and how to verify it
- Numerical evidence from the test suite
- The investigation history (what was tried, what failed, why)
