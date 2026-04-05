# SN Discrete Ordinates — Improvement Tracker

Central registry of ALL bugs, improvements, and features for the SN
solver.  **This is the single source of truth.**

## Tracking Number Format

`DO-YYYYMMDD-NNN` where:
- `DO` = Discrete Ordinates module
- `YYYYMMDD` = session date when the item was created
- `NNN` = sequential number within the session
- `00000000` = item predates the tracking system

## Status Legend

- **DONE**: implemented AND documented in Sphinx
- **IMPL**: implemented and tested, Sphinx documentation pending
- **OPEN**: not yet implemented, documented here with full context
- **WONT**: decided against, with rationale

## Where TODOs Live

TODOs exist in exactly TWO places:
1. **This file** — every item with its tracking number, status, and context
2. **In code** — at the exact location where the fix goes, with the
   matching tracking number (e.g., `# TODO DO-20260405-001: ...`)

No other files should contain TODOs.  If you find one, it must be
consolidated here and given a tracking number.

---

## IMPL — Implemented, Sphinx Documentation Pending

### DO-20260404-001 — Geometry-weighted balance equation (Bailey et al. 2009)

**Commits**: `fb3e976` through `981b87f`  
**Files**: `sn_geometry.py`, `sn_sweep.py`, `sn_quadrature.py`

The correct 1D curvilinear balance equation requires:
- α recursion: `α_{m+1/2} = α_{m-1/2} − w_m · η_m` (radial cosine)
- ΔA/w geometry factor on the redistribution term
- Ordinates η-sorted within each level

Without these, per-ordinate flat-flux consistency is broken, causing
the Morel–Montry flux dip and heterogeneous divergence.  The original
hypothesis (sign convention) was wrong — see `TODO_cylindrical_dd.md`
for the full investigation history.

**Sphinx needs**: full derivation from continuous PDE, per-ordinate
residual analysis, why the old formulation failed, numerical evidence
tables (before/after keff, fixed-source spike).

### DO-20260404-002 — Morel–Montry angular closure weights

**Commits**: `c54ad73`, `5e00333`  
**Files**: `sn_geometry.py` (`tau_mm`, `tau_mm_per_level`)

Weighted diamond difference with Bailey Eq. 74:
τ_m = (η_m − η_{m-1/2}) / (η_{m+1/2} − η_{m-1/2}).
Cell edges at midpoints of consecutive η values (cylindrical) or
weight-sum (spherical).  Clamped to [0.5, 1.0].

**Sphinx needs**: derivation of τ from asymptotic analysis, why it
forces contamination β = 0, cell-edge computation for both
geometries, alternating τ pattern for Product quadrature.

### DO-20260404-003 — BiCGSTAB operators for curvilinear geometries

**Commits**: `22723c8` (spherical), `0ce2621` (cylindrical)  
**Files**: `sn_operator.py`, `sn_solver.py`

Explicit transport operators with ΔA/w and M-M weights for both
geometries.  Multi-group spherical BiCGSTAB (previously unstable)
now converges.  Cylindrical BiCGSTAB added.

**Sphinx needs**: operator formulation, FD vs DD face-flux
approximation, why ΔA/w is needed in explicit operators too.

### DO-20260404-004 — Contamination analysis tool

**Commits**: `8ca174a`, `0ce2621`, `5e00333`  
**File**: `derivations/sn_contamination.py`

Computes Bailey's β factor and M-M τ weights.  Both geometries
give β ≈ 0 (machine zero) with the corrected formulation.

**Sphinx needs**: definition of β, physical meaning (contaminated
diffusion equation), usage as verification tool.

### DO-20260405-001 — Consolidated ΔA/w into SNMesh

**Commit**: `981b87f`  
**File**: `sn_geometry.py` (`redist_dAw`, `redist_dAw_per_level`)

Precomputed geometry factor eliminates duplication between sweep
and BiCGSTAB operator.  Future M-M improvements modify SNMesh only.

**Sphinx needs**: document the precomputed arrays and the single-source-
of-truth pattern for redistribution geometry.

---

## OPEN — Not Yet Implemented

### DO-20260405-002 — Gauss-type azimuthal quadrature for cylindrical

**Priority**: Medium | **Effort**: Moderate  
**Code location**: `sn_quadrature.py` (new quadrature class)

The equally-spaced `ProductQuadrature` gives duplicate η values
(paired ±ξ ordinates), producing alternating M-M weights
τ = [0.5, 1.0, ...].  A Gauss-type azimuthal quadrature with
non-uniform φ spacing would give distinct η values and smoothly
varying τ.

**References**: standard quadrature construction techniques.

### DO-20260405-003 — φ-based cell-edge computation for non-product quadratures

**Priority**: Low | **Effort**: Small  
**Code location**: `sn_geometry.py:275`

For quadratures where η values are distinct, transforming actual
φ cell boundaries to η-space would give exact M-M cell edges
instead of the midpoint approximation.

**References**: Bailey et al. (2009) Eq. 52.

### DO-00000000-001 — Diffusion Synthetic Acceleration (DSA)

**Priority**: HIGH | **Effort**: Large  
**Code location**: `sn_solver.py` (new acceleration method)

Reduces outer iterations from ~200 to ~20 for many-group problems.
Uses diffusion correction after each source iteration.
1D diffusion solver already exists in `05.Diffusion.1D/`.

**References**: Adams & Larsen (2002), Wareing et al.

### DO-00000000-002 — Transport Synthetic Acceleration (TSA)

**Priority**: Low | **Effort**: Large  
**Code location**: `sn_solver.py`

Coarse-angle transport acceleration for highly anisotropic media.
Only needed if DSA proves insufficient.

**References**: Ramone et al. (1997).

### DO-00000000-003 — Linear Discontinuous (LD) angular finite elements

**Priority**: Low | **Effort**: Large  
**Code location**: `sn_sweep.py` (new sweep variant)

Second-order angular accuracy without flux dip.  2×2 system per
cell-ordinate.  M-M WDD (DO-20260404-002) already eliminates the
flux dip, so LD is only needed for higher angular accuracy.

**References**: Bailey, Morel & Chang (2009) — main topic of paper.

### DO-00000000-004 — Negative flux fixup

**Priority**: Low | **Effort**: Small  
**Code location**: `sn_sweep.py` (inner loop guard)

If WDD produces ψ^a_out < 0, clamp to zero and rebalance.  Currently
not needed (zero negatives observed), but good practice for extreme
cases.

### DO-00000000-005 — Transport eigenmodes (Case's method)

**Priority**: Medium | **Effort**: Large  
**Code location**: `derivations/` (new module)

Mesh-independent analytical reference for 1D multi-group transport.
Full description in `derivations/TODO_transport_eigenmodes.md`.

**References**: Case (1960), Siewert (2000), Garcia & Siewert.

### DO-00000000-006 — Anisotropic scattering in curvilinear sweeps

**Priority**: Medium | **Effort**: Moderate  
**Code location**: `sn_sweep.py` (spherical/cylindrical branches)

P1+ anisotropic scattering implemented for Cartesian 2D but NOT
verified for curvilinear 1D.  Spherical harmonics on GL/Product
quadrature needs verification.

### DO-20260405-004 — Sphinx theory chapter for curvilinear SN

**Priority**: HIGH — blocks DO-20260404-001 through -004 from DONE  
**Effort**: Moderate  
**Code location**: `docs/theory/discrete_ordinates.rst`

Major update needed covering the Bailey et al. formulation:
- Balance equation with ΔA/w factor (derivation from PDE)
- α recursion from radial cosine
- Morel–Montry flux dip analysis and WDD closure
- Contamination factor β
- Investigation history (what failed, why)
- Numerical evidence from test suite

### DO-00000000-007 — GMRES/preconditioned Krylov for BiCGSTAB

**Priority**: Low | **Effort**: Moderate  
**Code location**: `sn_solver.py` (`_solve_bicgstab` methods)

BiCGSTAB can stagnate on non-normal operators.  GMRES(m) or
sweep-preconditioned BiCGSTAB may be more robust.

**References**: standard iterative methods literature.
