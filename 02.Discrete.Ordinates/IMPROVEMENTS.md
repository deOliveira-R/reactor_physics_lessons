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

## DONE — Implemented and Documented in Sphinx

### DO-20260404-001 — Geometry-weighted balance equation (Bailey et al. 2009)

Documented in ``docs/theory/discrete_ordinates.rst`` §5 (The Discrete Balance Equation).

### DO-20260404-002 — Morel–Montry angular closure weights

Documented in ``docs/theory/discrete_ordinates.rst`` §5.5–5.6.

### DO-20260404-003 — BiCGSTAB operators for curvilinear geometries

Documented in ``docs/theory/discrete_ordinates.rst`` §7.

### DO-20260404-004 — Contamination analysis tool

Documented in ``docs/theory/discrete_ordinates.rst`` §5.4.

### DO-20260405-001 — Consolidated ΔA/w into SNMesh

Documented in ``docs/theory/discrete_ordinates.rst`` §2.

### DO-20260405-004 — Sphinx theory chapter for curvilinear SN

Self-referential: this IS the Sphinx chapter (1276 lines, zero warnings).

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

### DO-00000000-007 — GMRES/preconditioned Krylov for BiCGSTAB

**Priority**: Low | **Effort**: Moderate  
**Code location**: `sn_solver.py` (`_solve_bicgstab` methods)

BiCGSTAB can stagnate on non-normal operators.  GMRES(m) or
sweep-preconditioned BiCGSTAB may be more robust.

**References**: standard iterative methods literature.
