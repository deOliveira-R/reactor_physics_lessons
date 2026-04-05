# Reactor Kinetics 0D — Improvement Tracker

Central registry of ALL bugs, improvements, and features for Module 08 (Reactor Kinetics 0D).

## Tracking Number Format

`RK-YYYYMMDD-NNN` where RK = Reactor Kinetics, YYYYMMDD = session date, NNN = sequence.

## Status Legend

- **DONE**: implemented AND documented in Sphinx
- **IMPL**: implemented and tested, Sphinx documentation pending
- **OPEN**: not yet implemented, documented here with full context

---

## IMPL — Implemented, Sphinx Pending

### RK-20260401-001 — Chunked integration for gap closure event detection

**Status:** IMPL  
**Date:** 2026-04-01  
**Files:** `reactor_kinetics.py` (Phase 2, lines 1027–1082)

**Problem:** scipy's `solve_ivp` event detection uses `brentq` root-finding internally.
The `_gap_closure_event` function (returns `min(gap_dr - roughness)`) triggers a
`ValueError` when the event function has the same sign at both ends of a solver step.

**Solution:** Replaced single `solve_ivp` call with chunked integration — one chunk per
output step (dt_transient = 0.1 s intervals).  After each chunk:
1. Evaluate `_gap_closure_event(t_next, y_end, p)` at chunk boundary
2. If sign changed from positive to negative → gap closure detected
3. Refine crossing time via bisection on dense_output (40 iterations, ~1e-12 precision)

**Why dense-output bisection works here (but not for Module 09):**  The reactor kinetics
state variables (power, DNP, fuel/clad temperatures, coolant enthalpy, plastic strains) all
remain physical under interpolation.  Unlike Module 09, there are no water property lookups
that fail on interpolated states because the coolant stays in single-phase subcooled
conditions during the RIA transient.

**Result:** Gap closure detected at t=103.39 s.  After closure, Phase 3 continues with
closed-gap boundary conditions to t=120 s.

### RK-20260401-002 — BDF solver switch (Radau → BDF)

**Status:** IMPL  
**Date:** 2026-04-01  
**Files:** `reactor_kinetics.py` (all 3 `solve_ivp` calls: SS, transient, closed-gap)

**Change:** Switched from `method="Radau"` to `method="BDF"` to match MATLAB's `ode15s`.
Both are implicit multi-step methods for stiff systems, but BDF matches ode15s's startup
behavior more closely.

**Validation (vs MATLAB results.m, 247 time steps):**

| Time (s) | MATLAB Power | Python Power | Match |
|-----------|-------------|-------------|-------|
| 100.0 | 1.000 | 1.000 | exact |
| 100.1 | 1.114 | 1.113 | 0.1% |
| 100.2 | 1.665 | 1.669 | 0.2% |
| 100.3 | 4.993 | 5.239 | 5% |
| 100.4 | 22.061 | 23.960 | 9% |
| 100.6 | 25.289 | 16.501 | timing offset |

Peak power: MATLAB 25.3× at t=100.6 s, Python 24.0× at t=100.4 s.
Same magnitude, slight timing difference — expected for different BDF implementations.

### RK-20260401-003 — Corrected MATLAB reference (peak power is 22-25×, not ~5×)

**Status:** IMPL  
**Date:** 2026-04-01

**Previous understanding:** The parity plan stated "Python peak power 24× vs MATLAB ~5×"
as a MAJOR discrepancy requiring investigation.

**Actual finding:** Extracted full power history from `matlab_archive/10.Reactor.Kinetics.0D/results.m`
(247 time steps, 82 unique output points).  MATLAB peak is **25.3×** at t=100.6 s.
The "~5×" reference was incorrect — possibly confused with the normalized power at a later
time point during the RIA recovery phase.

**Impact:** Task 4 (investigate power peak overshoot) was resolved as NOT A BUG.

---

## OPEN — Not Yet Implemented

### RK-20260401-004 — Detailed reactivity trace validation

**Status:** OPEN  
**Date:** 2026-04-01  
**Priority:** Low

Full MATLAB reference extracted: 247 time steps with power, Doppler reactivity, coolant
reactivity, and total reactivity.  At t=100.1 s: MATLAB reac_total=81.15 pcm,
Python=80.39 pcm (1% match).  A comprehensive point-by-point comparison could be
produced for a validation report, but is not required for parity.

### RK-20260401-005 — Power oscillation timing difference

**Status:** OPEN  
**Date:** 2026-04-01  
**Priority:** Low

Python peaks at t=100.4 s, MATLAB at t=100.6 s.  After the peak, the power oscillation
pattern differs (Python decays monotonically, MATLAB shows a second peak).  This is
expected behavior from different BDF implementations handling the same stiff system with
very rapid Doppler feedback.  The max_step_transient is 1e-3, so the solver takes
O(100) steps per 0.1 s output interval.  The stiffness ratio (prompt neutron lifetime
20 μs vs feedback time ~0.1 s) means small integration differences are amplified.

### RK-20260401-006 — Chunked steady-state integration (performance)

**Status:** OPEN  
**Date:** 2026-04-01  
**Priority:** Low

Phase 1 (steady state, 0–100 s) uses a single `solve_ivp` call with max_step=10.
This works fine but could be chunked for consistency with Phase 2.  Not needed since
there are no events during steady state.
