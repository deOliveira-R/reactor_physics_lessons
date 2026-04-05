# Analytical Derivations — Improvement Tracker

Central registry of ALL bugs, improvements, and features for the
`derivations/` package (SymPy analytical references, XS library,
RST generation).

## Tracking Number Format

`DV-YYYYMMDD-NNN` where DV = Derivations package.

## Status Legend

- **DONE**: implemented AND documented in Sphinx
- **IMPL**: implemented and tested, Sphinx documentation pending
- **OPEN**: not yet implemented, documented here with full context

---

## DONE — Implemented and Documented

### DV-20260403-008 — Spherical CP derivation cases

`derivations/cp_sphere.py` exists and is registered in
`reference_values.py`.  9 verification cases ({1,2,4}eg × {1,2,4}rg)
with exponential kernel.  Created in session 2026-04-04.

### DV-20260403-009 — Documentation: regenerate RST with all current cases

`generate_rst.py` runs and produces 8 RST fragments covering all
registered cases.  Sortable HTML table in `verification.rst`.
Rebuilt in session 2026-04-05.

---

## IMPL — Implemented, Sphinx Documentation Pending

### DV-20260403-001 — XS library with P1 scattering anisotropy

4 abstract regions (A=fissile, B=moderator, C=cladding, D=gap) in
{1G, 2G, 4G}.  P1 scattering matrices added with physically motivated
μ̄ ratios (A=0.05, B=0.60, C=0.10, D=0.30).  Used by all derivation
modules and tests.

### DV-20260403-002 — Full 9-case CP matrices (slab + cylinder)

{1,2,4}eg × {1,2,4}rg = 9 semi-analytical eigenvalue cases per
geometry.  4-region layout: A + D + C + B (fuel + gap + clad + mod).
Documented in `docs/theory/verification.rst` via generated RST.

### DV-20260403-003 — Lazy registry for solver-computed cases

`reference_values.py` loads analytical cases at import time but
defers Richardson-extrapolated cases (SN/MOC heterogeneous) until
first access, avoiding solver imports at derivation-import time.

---

## OPEN — Not Yet Implemented

### DV-20260403-004 — Diffusion 2-group interface matching (proper transcendental equation)

**Priority**: Medium | **Effort**: Moderate
**Code location**: `derivations/diffusion.py`

The `dif_slab_2eg_2rg` case currently uses Richardson extrapolation
from the diffusion solver itself (self-referencing).  The proper
analytical solution requires solving the coupled 2-group interface
matching problem:

- Fuel: coupled cos/exponential modes from 2-group eigenvalue
- Reflector: coupled sinh/exponential decay modes (no fission)
- Interface: flux + current continuity for BOTH groups simultaneously
- Result: 4×4 determinantal condition → transcendental equation for k

The 1-group fast-group-only approximation was attempted and gave k=0.978
vs solver k=0.870 (12% error), confirming that the 2-group coupling
cannot be ignored.

**References**: Duderstadt & Hamilton §5.5 (multi-group interface matching).

### DV-20260403-005 — MC heterogeneous independent reference

**Priority**: Medium | **Effort**: Moderate
**Code location**: `derivations/mc.py`

MC heterogeneous cases use the CP cylinder eigenvalue as proxy
reference.  MC with periodic BCs and CP with white BC solve slightly
different problems.  A proper independent reference would be a
very-high-statistics MC run (~10⁵ active cycles) cached as a
reference value with known σ.

### DV-20260403-006 — MOC Richardson quality improvement

**Priority**: Low | **Effort**: Small
**Code location**: `derivations/moc.py`

MOC heterogeneous uses only 3 mesh levels (8, 12, 16 cells/side)
with non-uniform ratio (3/4 between finest two).  Should use 4 levels
with ratio 2 (e.g., 6, 12, 24, 48) for reliable O(h²) extrapolation.

### DV-20260403-007 — MMS (Method of Manufactured Solutions)

**Priority**: Medium | **Effort**: Moderate
**Code location**: `derivations/mms.py` (new file)

Fixed-source verification for SN, MOC, and diffusion.  Each solver's
transport operator is tested independently of the eigenvalue by:
1. Prescribing a flux shape φ(x)
2. Computing the source Q from the solver's own equations
3. Feeding Q into the solver's fixed-source sweep
4. Verifying O(h²) convergence to the prescribed flux
