# Thermal Hydraulics — Improvement Tracker

Central registry of ALL bugs, improvements, and features for Module 07 (Thermal Hydraulics).

## Tracking Number Format

`TH-YYYYMMDD-NNN` where TH = Thermal Hydraulics, YYYYMMDD = session date, NNN = sequence.

## Status Legend

- **DONE**: implemented AND documented in Sphinx
- **IMPL**: implemented and tested, Sphinx documentation pending
- **OPEN**: not yet implemented, documented here with full context

---

## DONE — Implemented and Documented

### TH-20260405-001 — IAPWS viscosity fallback for post-failure NaN

**Status:** DONE  
**Date:** 2026-04-05  
**Commit:** cc3ea85  
**Files:** `data/materials/h2o_properties.py` (lines 28–60)  
**Sphinx:** `docs/theory/thermal_hydraulics.rst` §IAPWS Viscosity Fallback

**Problem:** After clad failure, coolant node 2 reaches T > 900 °C during LOCA
blowdown.  pyXSteam's `my_ph` (dynamic viscosity) returns NaN above 900 °C due
to an artificial guard in the IAPWS correlation implementation — not a physical
singularity.  The NaN propagates through `_compute_pressure` (friction factor
needs viscosity) → NaN pressure → NaN everything → solver crash at t ≈ 395 s.

**Root cause chain:**
1. pyXSteam's `my_AllRegions_ph` checks `if T > 900 + 273.15` → returns NaN
2. The IAPWS 2008 viscosity correlation itself has no such limit; the formula
   is well-defined for the dilute-gas and finite-density terms at any T
3. XSteam has this same cutoff in both MATLAB and Python versions

**Fix:** Added `_iapws_viscosity(T_K, rho)` — the same IAPWS 2008 correlation
(H coefficient matrix, dilute-gas mu0, finite-density mu1) without the 900 °C
guard.  Used as NaN fallback in `h2o_properties()` for both subcooled and
superheated branches.  Validated bit-identical to pyXSteam below 900 °C
(ratio = 1.000000 at 500, 600, 700, 800, 850, 890, 895, 899 °C).

**Result:** Both `thermal_hydraulics.py` (correct physics) and
`thermal_hydraulics_dae.py` (MATLAB parity) now run to t = 600 s.

### TH-20260405-002 — MATLAB gap geometry bug identification

**Status:** DONE  
**Date:** 2026-04-05  
**Sphinx:** `docs/theory/thermal_hydraulics.rst` §MATLAB Gap Geometry Bug

**Finding:** MATLAB `funRHS.m` line 272: `gap.r_ = (clad.r(1) + fuel.dz)/2`
mixes a cladding inner radius (~4.22 mm) with the fuel axial node height
(~1.5 m), producing gap.r_ ≈ 0.752 m instead of the correct ~4.17 mm.
This makes the gap heat transfer area 180× too large, keeping MATLAB's fuel
332 °C cooler (808 °C vs 1140 °C at steady state).

**Impact on validation:**
- Python (correct physics): fuel center 1140 °C, clad failure at 287 s
- MATLAB (with bug): fuel center 808 °C, clad failure at 425.3 s
- Python DAE (replicating bug): fuel center 808 °C, clad failure at 379 s

The remaining 46 s gap between DAE (379 s) and MATLAB (425.3 s) is traced to
MATLAB's scalar indexing of `gap.T = (fuel.T(fuel.nr) + clad.T(1))/2` which
in column-major order returns element 20 of the flattened (2,20) array —
i.e., axial node 2, radial node 10 — not the fuel surface temperature.

### TH-20260405-003 — DAE version for MATLAB parity comparison

**Status:** DONE  
**Date:** 2026-04-05  
**Commit:** cc3ea85  
**Files:** `thermal_hydraulics_dae.py`, `run_thermal_hydraulics_dae.py`

Created alternative version with:
- Coolant pressure as state variable with stiff relaxation dp/dt = -K(p - p_target)
- MATLAB's gap geometry replicated for direct comparison
- Node 1 pressure used for inlet density (matches MATLAB's `cool.p(1)`)

Results match MATLAB within 1 kJ/kg for coolant enthalpy across all time points.

### TH-20260401-001 — Chunked integration for clad failure event detection

**Status:** DONE  
**Date:** 2026-04-01  
**Sphinx:** `docs/theory/thermal_hydraulics.rst` §Chunked Integration for Event Detection  
**Files:** `thermal_hydraulics.py` (lines 1024–1082)

Replaced scipy's built-in event detection (which fails with `ValueError` when
the event function has same sign at both ends of a step) with chunked integration
— one chunk per output step (1 s).  Dense-output bisection was tried and rejected
because interpolated states produce unphysical water property values.

### TH-20260401-002 — BDF solver switch (Radau → BDF)

**Status:** DONE  
**Date:** 2026-04-01  
**Sphinx:** `docs/theory/thermal_hydraulics.rst` §ODE Integration  
**Files:** `thermal_hydraulics.py` (all `solve_ivp` calls)

Switched from `method="Radau"` to `method="BDF"` to match MATLAB's `ode15s`.

### TH-20260401-003 — Graceful post-failure NaN handling

**Status:** DONE  
**Date:** 2026-04-01  
**Sphinx:** `docs/theory/thermal_hydraulics.rst` §Post-failure integration  
**Files:** `thermal_hydraulics.py` (Phase 2 integration)

Phase 2 uses chunked integration with try/except ValueError for graceful
degradation.  NaN root cause subsequently found and fixed (TH-20260405-001).

### TH-20260401-004 — Post-failure NaN at t>395 s

**Status:** DONE (fixed by TH-20260405-001)  
**Date:** 2026-04-01, resolved 2026-04-05  
**Sphinx:** `docs/theory/thermal_hydraulics.rst` §IAPWS Viscosity Fallback

Root cause was pyXSteam's viscosity cutoff at 900 °C.  Fixed via IAPWS viscosity
fallback.  See TH-20260405-001.

---

## OPEN — Not Yet Implemented

### TH-20260401-005 — Point-by-point LOCA validation against MATLAB

**Status:** OPEN  
**Date:** 2026-04-01  
**Priority:** Medium

MATLAB reference data available in `matlab_archive/09.Thermal.Hydraulics/results.m`.
Should compare at key times: t=1 s, 200 s (pre-LOCA), 210 s (blowdown), 242 s (deep LOCA),
287 s (clad failure).  At t=1 s, fuel center already matches (429.5 vs 428.0 °C).

### TH-20260401-006 — Dense-output bisection for precise clad failure time

**Status:** OPEN  
**Date:** 2026-04-01  
**Priority:** Low

Currently clad failure time has 1 s resolution (chunk endpoint).  A re-integration
bisection approach (re-running solve_ivp with smaller spans) could refine this to ~1 ms.
Tested but not implemented because:
- The bisection states from dense_output are unphysical (NaN in water properties)
- Re-integration bisection works but is expensive (~10 extra integrations)
- 1 s resolution is adequate for LOCA analysis

### TH-20260405-004 — Derivation scripts for TH equations

**Status:** OPEN  
**Date:** 2026-04-05  
**Priority:** Medium

No SymPy derivation scripts exist in `derivations/` for the TH module.
Key equations needing backing:
- Fuel FV discretisation (PDE → control volume integration → discrete form)
- Clad stress 15×15 linear system coefficient expressions
- IAPWS viscosity formula verification

### TH-20260405-005 — Add TH module to Sphinx autodoc path

**Status:** OPEN  
**Date:** 2026-04-05  
**Priority:** Medium

`07.Thermal.Hydraulics/` is not on the Sphinx autodoc path in `conf.py`.
Only one cross-reference (`:func:`) exists in 837 lines of RST.  Adding
autodoc would enable proper `:func:`, `:class:`, `:mod:` links throughout.

### TH-20260405-006 — Correct-physics version validation table

**Status:** OPEN  
**Date:** 2026-04-05  
**Priority:** Low

The "correct physics" version (`thermal_hydraulics.py`) has no validation
table in Sphinx because MATLAB's reference is based on wrong gap geometry.
An analytical steady-state estimate (ΔT ≈ LHGR/(4πk) ≈ 567 °C above fuel
surface) could serve as a sanity check.  Should also document the full
temperature evolution from the correct-physics run.
