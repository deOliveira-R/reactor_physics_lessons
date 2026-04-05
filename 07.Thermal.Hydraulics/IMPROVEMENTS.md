# Thermal Hydraulics — Improvement Tracker

Central registry of ALL bugs, improvements, and features for Module 07 (Thermal Hydraulics).

## Tracking Number Format

`TH-YYYYMMDD-NNN` where TH = Thermal Hydraulics, YYYYMMDD = session date, NNN = sequence.

## Status Legend

- **DONE**: implemented AND documented in Sphinx
- **IMPL**: implemented and tested, Sphinx documentation pending
- **OPEN**: not yet implemented, documented here with full context

---

## IMPL — Implemented, Sphinx Pending

### TH-20260401-001 — Chunked integration for clad failure event detection

**Status:** IMPL  
**Date:** 2026-04-01  
**Files:** `thermal_hydraulics.py` (lines 1024–1082)

**Problem:** scipy's `solve_ivp` event detection uses `brentq` root-finding internally.
When the event function (sigB - sigI) evaluates to the same sign at both ends of a solver
step (due to numerical noise or large step sizes), `brentq` raises
`ValueError: f(a) and f(b) must have different signs`.  This made the `_clad_failure_event`
unusable with scipy's built-in `events=` parameter.

**Solution:** Replaced single `solve_ivp` call with chunked integration — one chunk per
output step (1 s intervals).  After each chunk:
1. Evaluate `_clad_failure_event(t_next, y_end, p)` at chunk boundary
2. If sign changed from positive to negative → clad failure detected
3. Use chunk endpoint as failure state (1 s resolution)

**Why not dense-output bisection?**  Tried dense_output with bisection (40 iterations)
to refine the crossing time.  The interpolated states from `sol.sol(t_mid)` produced
unphysical intermediate values — specifically, water/steam properties (pyXSteam) returned
NaN for the interpolated pressure/enthalpy combinations.  The chunk endpoint is a fully
converged solver state and remains physical.

**Result:** Clad failure detected at t=287 s during LOCA blowdown.

### TH-20260401-002 — BDF solver switch (Radau → BDF)

**Status:** IMPL  
**Date:** 2026-04-01  
**Files:** `thermal_hydraulics.py` (all `solve_ivp` calls)

**Change:** Switched all `solve_ivp` calls from `method="Radau"` to `method="BDF"`.
BDF (Backward Differentiation Formula) is the same method family as MATLAB's `ode15s`.

**Validation:** Fuel center temperature at t=1 s: Python 429.5 °C vs MATLAB 428.0 °C (0.3% match).

### TH-20260401-003 — Graceful post-failure NaN handling

**Status:** IMPL  
**Date:** 2026-04-01  
**Files:** `thermal_hydraulics.py` (Phase 2 integration, lines 1083–1114)

**Problem:** After clad failure at t=287 s, the post-failure integration (Phase 2) hits
NaN at t=395 s.  The NaN originates from the Jacobian numerical differencing: the BDF/Radau
solver evaluates the RHS at slightly perturbed states, and these perturbations push
water/steam properties (pyXSteam) outside their valid pressure/temperature range.

**Solution:** Phase 2 uses chunked integration (1 s steps) with try/except ValueError
for graceful degradation.  The simulation stops cleanly at t=395 s instead of crashing.

**Note:** This is not a regression — the post-failure code path was never exercised
before TH-20260401-001 enabled event detection.

---

## OPEN — Not Yet Implemented

### TH-20260401-004 — Post-failure NaN at t>395 s

**Status:** OPEN  
**Date:** 2026-04-01  
**Priority:** Low

The root cause is pyXSteam's property evaluations failing for extreme LOCA conditions
during Jacobian numerical differencing.  Possible fixes:
1. Add clamping in the H2O property wrapper (return boundary values for out-of-range inputs)
2. Provide an analytical Jacobian to avoid numerical differencing entirely
3. Use the IAPWS viscosity fallback (already implemented in `h2o_properties.py` for the
   viscosity cutoff at 900 °C — extend to other properties)
4. Accept current behavior (395 s coverage is sufficient for most LOCA analyses)

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
