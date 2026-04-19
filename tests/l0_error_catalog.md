# L0 Error Catalog

Errors caught during development by term-level verification.  Each
entry records the error, how it hid from higher-level tests, the L0
test that caught it, and the lesson learned.

This file is the primary QA publication artifact.  It supersedes
``gotchas.md`` (deleted).

## Error Classification

Errors are classified by the 6 AI Failure Modes taxonomy:

| # | Mode | Description |
|---|------|-------------|
| 1 | Sign flip | `(a − b)` vs `(b − a)` |
| 2 | Variable swap | `mu_x` vs `mu_y`, `h_sat_l` vs `h_sat_v` |
| 3 | Missing factor | Missing `2×`, `ΔA/w`, volume |
| 4 | Factor error | Wrong constant, hardcoded value |
| 5 | Index error | `face[i]` vs `face[i+1]` |
| 6 | Convention drift | Definition site vs usage site disagree |

---

## ERR-001 — Z-ordinate weight loss in 2D Lebedev sweep

**Failure mode:** #3 Missing factor — missing contribution  
**Date:** 2026-04-03  
**Solver:** SN (Cartesian 2D)

**Bug:** In `_sweep_2d_wavefront`, ordinates with `mu_x = mu_y = 0`
(z-directed) were skipped with `continue`.  Their quadrature weights
(0.77% of total) were lost from scalar flux integration.

**Impact:** Multi-group eigenvalue error of ~0.4% (2G: 1.867 vs 1.875).

**How it hid from higher-level tests:**
- 1-group keff = νΣ_f/Σ_a is independent of weight loss (cancels)
- Spatial convergence showed the scheme converged — to the wrong value
- "Reasonable numbers" — 0.4% looks like discretization error

**L0 test that catches it:** L0-SN-001 (streaming equilibrium) — the
volume-averaged φ would deviate from Q/Σ_t by the missing weight
fraction.

**Lesson:** 1-group eigenvalue tests are degenerate.  Always verify
with ≥2 groups.

---

## ERR-002 — Scattering matrix transpose in vectorization

**Failure mode:** #2 Variable swap — `SigS` vs `SigS^T`  
**Date:** 2026-04-03  
**Solver:** SN (all geometries)

**Bug:** Vectorized `_add_scattering_source` used `phi @ SigS^T`
(double-transposed) instead of `phi @ SigS`.

**Impact:** keff = 2.06 (catastrophically wrong) — caught immediately.

**How it could hide:** For symmetric scattering matrices (1-group
self-scatter), SigS = SigS^T and the bug is invisible.

**L0 test that catches it:** L0-SN-009 (scattering source magnitude)
— hand-calculated SigS^T @ φ vs code output with asymmetric 2G matrix.

**Lesson:** Always test with asymmetric inputs.  The identity
`(A^T v) = (v^T A)^T` means the transpose moves sides, not into
a pre-transposed matrix.

---

## ERR-003 — Octant batching breaks reflective BC ordering

**Failure mode:** #6 Convention drift — implicit ordinate ordering  
**Date:** 2026-04-03  
**Solver:** SN (Cartesian 2D)

**Bug:** Batching ordinates by sweep direction changed the order in
which reflective BC boundary fluxes were updated.  A group reads
boundary fluxes that its reflected group hasn't updated yet.

**Impact:** 2G convergence test failed (keff diffs grew).

**How it hid:** 1G converged; the optimization gave 2× speedup
on the passing test.

**L0 test that catches it:** Not directly an L0 problem (this is
a coupling issue between ordinates).  Caught by L1: eigenvalue
convergence with mesh refinement.

**Lesson:** Sequential processing order is part of the interface
contract for reflective BCs.  Don't parallelize without verifying
the data dependency.

---

## ERR-004 — Hardcoded 4π in BiCGSTAB RHS normalization

**Failure mode:** #4 Factor error — wrong normalization constant  
**Date:** 2026-04-03  
**Solver:** SN BiCGSTAB (spherical)

**Bug:** `build_rhs` hardcoded `4π` for angular normalization.
Correct for Lebedev (sum(w) = 4π), wrong for GL (sum(w) = 2).

**Impact:** BiCGSTAB on GL diverged (keff oscillating).

**How it hid:** All initial BiCGSTAB testing used Lebedev where
4π is correct.

**L0 test that catches it:** L0-SN-001 (streaming equilibrium)
with BiCGSTAB solver — φ would not converge to Q/Σ_t with wrong
normalization.

**Lesson:** Never hardcode quadrature-dependent constants.

---

## ERR-005 — DD recurrence rewrite breaks multi-group convergence

**Failure mode:** #4 Factor error — numerically unstable rewrite  
**Date:** 2026-04-04  
**Solver:** SN (Cartesian 1D)

**Bug:** Algebraically equivalent rewrite of `_solve_recurrence`
introduced catastrophic cancellation: `2*(…) − psi_in` subtracts
nearly-equal large numbers.

**Impact:** Multi-group scattering iteration diverged (~1e34/iter).

**How it hid:** 1-group unaffected; formulas are algebraically
identical; 2D wavefront (different code path) still passed.

**L0 test that catches it:** Not a single-term issue; caught by
L1 (multi-group eigenvalue convergence).

**Lesson:** "Algebraically equivalent" ≠ "numerically equivalent".
The stable form `0.5*(psi_in + psi_out)` averages known quantities;
the unstable form subtracts large numbers.

---

## ERR-006 — Wrong α recursion + missing ΔA/w in curvilinear sweep

**Failure mode:** #2 Variable swap + #3 Missing factor  
**Date:** 2026-04-04  
**Solver:** SN (cylindrical, spherical)

**Bug:** Two simultaneous bugs:
1. α recursion used `cumsum(+w·ξ)` with azimuthal cosine mu_y
   instead of `cumsum(−w·η)` with radial cosine mu_x
2. Missing `ΔA_i/w_m` geometry factor on the redistribution term

**Impact:** Heterogeneous keff diverged with mesh refinement
(1.15 → 0.90 → 0.52 → 0.25).  Spherical had flux spike 5.1× at r=0.

**How it hid from 20 passing tests:**
- Homogeneous eigenvalue: exact (redistribution cancels for flat flux)
- 1-group: degenerate (keff = material ratio)
- Particle balance: exact (telescoping is by construction)
- Conservation: exact (total is correct; per-ordinate is wrong)
- Flux non-negativity: no negatives produced
- Single sweep finite: fluxes are finite, just wrong

**L0 tests that catch it:**
- L0-SN-003 (per-ordinate flat-flux) — streaming + redistribution ≠ 0
  per ordinate without ΔA/w.  This is the definitive L0 diagnostic.
- L0-SN-001 (streaming equilibrium) — flux spike at r=0 visible
- L0-SN-008 (contamination β) — β ≈ 2.0 instead of ~0

**Investigation history:** 6 approaches failed before root cause found:
reverse sweep, step closure, starting direction, bidirectional sweep,
scaled α, zero redistribution.  Full details in
`docs/theory/discrete_ordinates.rst` §Investigation History.

**Lesson:** Per-ordinate flat-flux consistency (L0-SN-003) is the
FUNDAMENTAL correctness criterion for curvilinear SN.  It should be
the first test written for any curvilinear transport implementation.

---

## ERR-007 — Multi-group BiCGSTAB unstable for spherical geometry

**Failure mode:** #3 Missing factor — same as ERR-006  
**Date:** 2026-04-05  
**Solver:** SN BiCGSTAB (spherical, cylindrical)

**Bug:** The explicit FD transport operator was missing the same
ΔA/w geometry factor as the sweep (ERR-006).

**Impact:** Multi-group BiCGSTAB diverged for spherical (keff → NaN).
Previously documented as "BiCGSTAB is unreliable for curvilinear"
when the real issue was a missing geometry factor.

**L0 test that catches it:** L0-SN-003 applied to the operator output.

**Lesson:** When a bug is found in one code path (sweep), check ALL
code paths that implement the same physics (BiCGSTAB operator).

---

## ERR-008 — Boundary volume halving in SN keff computation

**Failure mode:** #6 Convention drift — geometry vs solver disagree  
**Date:** 2026-04-03  
**Solver:** SN (unified solver, CartesianMesh)

**Bug:** `CartesianMesh.volume` halved first/last cell volumes (matching
the old `PinCellGeometry` convention from the 2D MATLAB port).  The SN
solver used these volumes in the keff computation: `k = Σ(νΣf·φ·V) / Σ(Σa·φ·V)`.
With reflective BCs, boundary cells are at the symmetry plane and should
have full volume, not half.

**Impact:** ~1e-4 systematic error in keff for heterogeneous problems.
Homogeneous problems unaffected (uniform flux × uniform XS → volume
cancels in the ratio).

**How it hid:** Homogeneous verification tests all passed to machine
precision.  The error only appeared when comparing the unified solver
against the old `sn_1d.py` solver (which correctly used full `dx` as
volume, not the geometry's `volume` property).

**L0 test that catches it:** Direct comparison of keff between old and
new solver on the same heterogeneous problem.  A dedicated test would
verify `Σ(νΣf·φ·V) / Σ(Σa·φ·V) = keff_reported` with known volumes.

**Lesson:** Volume conventions (half-cell vs full-cell) must be
explicit and documented.  The same geometry object should not be used
for both "mesh for sweeping" (where boundary halving might make sense
for source normalization) and "volumes for integral quantities" (where
the full cell contributes).

---

### ERR-009 — CP neutron balance transpose

**Failure mode:** #2 Variable swap — P vs P.T  
**Date:** 2026-04-05  
**Solver:** CP (slab and cylindrical)

**Bug:** The CP power iteration computed `phi = P_inf @ source` instead
of `phi = P_inf.T @ source`.  With the convention `P[i,j]` =
P(birth_i → collision_j), the neutron balance for collision target `j`
sums over birth regions: `Σ_i P[i,j] · V_i · Q_i = P.T @ source`.

**Impact:** Wrong eigenvalue for ALL heterogeneous problems.  For the
1G 2-region slab benchmark: solver gave k=1.373 vs analytical k=1.272
(8% error).  keff was systematically too high because the flux
redistribution between fuel and moderator was incorrect.

**How it hid:** Homogeneous benchmarks passed to machine precision
because P is symmetric when all regions have identical cross sections
(P = P.T).  The 1-group homogeneous case is doubly degenerate: no
spatial redistribution AND k = νΣ_f/Σ_a regardless of P.  The bug
only appeared when running 2-region heterogeneous benchmarks with the
formal verification suite.

**L0 test that catches it:** The synthetic 1G 2-region slab benchmark
(`benchmark_1g_slab`) with analytical eigenvalue from the CP matrix.
The analytical k is computed independently by solving the 2×2 matrix
eigenvalue problem `det(A - (1/k)B) = 0`.  Any transpose error in the
solver causes immediate disagreement.

**Lesson:** The CP matrix convention (birth-first vs collision-first
indexing) propagates through the entire solver.  Document the convention
explicitly (now in `collision_probability.rst` §Flat-Source Approximation)
and verify with heterogeneous multi-region benchmarks.  Homogeneous
verification is *necessary but not sufficient* — it only tests the
diagonal of the CP matrix.

---

## ERR-010 — pyXSteam viscosity cutoff at 900 °C causes NaN cascade

**Failure mode:** #4 Factor error — library-imposed validity limit  
**Date:** 2026-04-05  
**Solver:** Thermal Hydraulics (Module 07)

**Bug:** pyXSteam's `my_AllRegions_ph` returns NaN for T > 900 °C due
to an artificial guard (`if T > 900 + 273.15: return NaN`).  The IAPWS
2008 viscosity correlation itself is well-defined beyond this limit.
During post-failure LOCA blowdown, coolant reaches ~901 °C in the
outlet node.  NaN viscosity → NaN kinematic viscosity → NaN friction
factor → NaN pressure → solver crash.

**Impact:** Integration stopped at t ≈ 395 s (of 600 s target).

**How it hid from higher-level tests:**
- Pre-failure phase (t < 287 s) coolant stays well below 900 °C
- Post-failure code path was never exercised until event detection was
  implemented (TH-20260401-001)
- The NaN manifested as `cool_p = [NaN, NaN]`, suggesting a pressure
  bug rather than a viscosity bug — required tracing through two levels
  of function calls to find the root cause

**L0 test that catches it:** Direct property evaluation:
`assert not np.isnan(h2o_properties(0.33, 4399e3)[0].mu)` — tests
that viscosity is returned for high-enthalpy states reachable during
LOCA.

**Fix:** `_iapws_viscosity(T_K, rho)` — same formula without cutoff.

**Lesson:** Third-party library validity limits are not always physical
limits.  When a library returns NaN, check whether the underlying
correlation is actually invalid or just conservatively guarded.

---

## ERR-011 — MATLAB gap geometry mixes radius with axial height

**Failure mode:** #2 Variable swap — `fuel.r` vs `fuel.dz`  
**Date:** 2026-04-05  
**Solver:** Thermal Hydraulics (MATLAB reference, `funRHS.m` line 272)

**Bug:** `gap.r_ = (clad.r(1) + fuel.dz)/2` adds a cladding inner
radius (~4.22 mm) to the fuel axial node height (~1.5 m), producing a
"gap radius" of 0.752 m instead of the correct ~4.17 mm.  The gap heat
transfer area becomes 180× too large.

**Impact:** MATLAB fuel centre temperature is 808 °C instead of the
correct 1140 °C at steady state.  This artificially keeps coolant below
pyXSteam's 900 °C viscosity limit and delays clad failure by ~138 s.

**How it hid:**
- MATLAB ran to completion because the wrong gap area prevented the
  coolant from ever reaching the viscosity cutoff
- All MATLAB results are self-consistent — temperatures, stresses, and
  failure time are plausible for a LOCA scenario, just based on wrong
  gap thermal resistance
- The bug is in a deformable geometry update section where `clad.r`,
  `fuel.r`, `fuel.dz`, `clad.dz` all appear nearby — easy to grab the
  wrong variable

**L0 test that catches it:** Steady-state fuel centre temperature check:
at 69 kW total power, T_fuel_centre should be ~1100-1200 °C, not ~800 °C.
A simple analytical estimate: ΔT ≈ LHGR/(4πk) ≈ 567 °C above the fuel
surface.

**Lesson:** Variable names that differ only in the last character
(`fuel.r` vs `fuel.dz`) are a maintenance hazard, especially in code
with mixed scalar/vector indexing (MATLAB's `fuel.dz` is a vector, but
`clad.r(1)` extracts a scalar via linear indexing of a 2D array).

---

## ERR-012 — Static heat transfer areas in deformable TH/RK modules

**Failure mode:** #3 Missing factor — missing geometry update  
**Date:** 2026-04-01  
**Solver:** Thermal Hydraulics (Module 07), Reactor Kinetics (Module 08)

**Bug:** Gap and clad radial heat transfer areas (`gap_a_bnd`, `clad_a_bnd`)
were computed once at initialization from fabrication geometry and never
updated with deformed radii/heights.  MATLAB's `funRHS.m` recomputes
`gap.a_` and `clad.a_` every RHS call from the current deformed geometry
(`clad.r`, `clad.dz`, `fuel.r`, `fuel.dz`).

**Impact:** During LOCA/RIA transients, fuel thermal expansion changes the
gap geometry by ~0.5–3%.  Using stale fabrication areas introduces a
systematic bias in the radial heat transfer.  The impact is small at steady
state but compounds during transients with large deformations.

**How it hid from higher-level tests:**
- At t=0 and during early transient, deformations are negligible (< 0.1%)
  so static areas produce identical results
- Steady-state eigenvalue tests don't exercise the deformable geometry path
- The `clad_a_bnd_def` variable WAS computed in the RHS (line 791 of TH)
  but never used — a classic "dead code" pattern

**L0 test that catches it:** Compare fuel surface temperature with static
vs deformed areas during a LOCA at t > 300 s.  The static version over-
estimates gap heat transfer when the gap narrows (area decreases with
thermal expansion).

**Fix:** Replace `p["gap_a_bnd"]` and `p["clad_a_bnd"]` with locally
computed deformed values.  Fuel areas kept static (MATLAB convention —
fuel boundary areas are not recomputed in `funRHS.m`).

**Lesson:** When initializing geometry parameters for an ODE RHS, document
explicitly which quantities are "frozen at fabrication" vs "updated each
call".  The MATLAB code doesn't distinguish these — it uses globals that
are silently overwritten.

---

## ERR-013 — Closed-gap stress BC uses fabrication gap width instead of roughness

**Failure mode:** #4 Factor error — wrong denominator  
**Date:** 2026-04-01  
**Solver:** Fuel Behaviour (Module 06)

**Bug:** In `_solve_stress()`, the closed-gap boundary conditions (BC3 and BC4)
divided stress/strain gradients across the gap by `params["gap_dr0"]` (fabrication
gap width = 100 μm) instead of the effective contact gap thickness (~6 μm roughness).
The MATLAB DAE uses the current deformed gap width, which converges to roughness
after closure.

**Impact:** Contact pressure was 40.5 MPa vs MATLAB's 39.8 MPa with the correct
fix, but was systematically wrong by a factor related to (100/6) ≈ 17× in the
stress gradient term before the fix.  The initial fix used `gap_dr0` → the contact
pressure was exactly 10× too high compared to the MATLAB reference value reported
at a different timestep.

**How it hid from higher-level tests:**
- The open-gap phase (before 2.85 years) was unaffected — BC3/BC4 use the
  pressure BCs, not the gap gradient form
- The closed-gap phase stress values were "reasonable" (~40 MPa) even though
  they were wrong — no analytical reference exists for the full closed-gap
  coupled system
- The 10× discrepancy was initially attributed to MATLAB's contact pressure
  being at a different timestep (it was — but the BC was also wrong)

**L0 test that catches it:** Compare contact pressure at a fixed time after
closure against an independent analytical estimate: for a thin-walled tube
under internal pressure (p_gas - p_cool) with contact, σ_r(inner) ≈ -p_contact
≈ -(p_gas - p_cool) × geometry_factor.

**Fix:** BC4 rewritten as a displacement-based gap constraint:
`r_clad_in_deformed - r_fuel_out_deformed = roughness`.  This is linear in
stresses, physically transparent, and avoids any division by gap width.

**Lesson:** When converting DAE residuals to algebraic equations for a linear
solve, the effective "thickness" used in finite-difference gradients across
interfaces must match the physical gap state.  A fabrication-time value is
wrong after gap closure because the physics has fundamentally changed.

---

## ERR-014 — sigT truncation inconsistency in .m data files

**Failure mode:** #4 Factor error — truncated intermediate vs stored total  
**Date:** 2026-04-05  
**Solver:** Cross-section data pipeline (all solvers affected via sigma-zero iteration)

**Bug:** The MATLAB `convertCSVtoM.m` script computes `sigT = sigC + sigF + sigL + sum(sigS)` from full-precision intermediates, then writes ALL quantities (sigC, sigF, sigS, sigT) independently truncated to `%13.6e` format.  The stored `sigT` is the once-truncated full-precision sum, while recomputing `sigT` from the stored (doubly-truncated) components gives a different value.  For U-238 at 600K: stored sigT[0,0] = 108.14, recomputed from components = 77.87, offset = 30.27 barns.

**Impact:** When the GXS→HDF5 converter computed sigT from components (the physically correct approach), the sigma-zero iterations converged to different values, shifting PWR k_inf by 0.4% (1.01771 vs 1.01357).

**How it hid from higher-level tests:**
- The aqueous reactor (H-1 + O-16 + U-235 at 294K) was unaffected because H-1 has only 1 sigma-zero (no interpolation needed) and O-16 has negligible offset
- All `.m` file components matched the GXS parser exactly (sigC, sigF, sigS diffs = 0) — the discrepancy only appeared in the stored sigT which was precomputed from higher-precision intermediates
- The 0.4% k_inf shift is within the range of "plausible" numerical differences between implementations

**L0 test that catches it:** Direct comparison: `assert max(|sigT_stored - (sigC + sigF + sigL + sigS_rowsum)|) < 1e-3` for each isotope.  This immediately reveals the 30-barn offset.

**Lesson:** When porting a data pipeline, verify not just individual components but also **derived quantities** that were computed from those components.  The same truncation format applied to inputs and outputs does not guarantee consistency between stored outputs and recomputed-from-stored-inputs.

---

## ERR-015 — compute_keff ignores (n,2n) net neutron production

**Failure mode:** #3 Missing factor — missing (n,2n) contribution in eigenvalue estimate  
**Date:** 2026-04-05  
**Solver:** CP (all geometries)

**Bug:** `CPSolver.compute_keff` computed `k = νΣf·φ·V / Σa·φ·V` where
`Σa = Σc + Σf + ΣL + Σ₂_out`.  Each (n,2n) reaction produces one
*extra* neutron (one in, two out), but neither the extra neutron
production nor the removal accounting reflected this.  The correct
eigenvalue balance is:

    k = νΣf·φ·V / (Σt − Σs − 2Σ₂)·φ·V

The denominator is the *net* removal after all scattering and (n,2n)
production is subtracted from the total.  When Σ₂ = 0, this reduces to
`νΣf / Σa` (no change to existing tests).

**Impact:** With `Sig2[0,0] = 0.01` on region A (2G), the solver
converged to k = 1.793 instead of the analytical k = 2.045 — a 12%
error.  The transport solve (`solve_fixed_source`) correctly included
`2·Σ₂·φ` in the source, so the flux shape was right, but the eigenvalue
estimate was biased low by the missing production term.

**How it hid from higher-level tests:**
- ALL test materials have `Sig2 = 0` (zero sparse matrix).  The formula
  is correct when Sig2 = 0 because `total - scatter - 0 = absorption`.
- The `make_mixture` function hardcoded `Sig2 = csr_matrix((ng, ng))`
  with no parameter to override it, so it was impossible to construct
  a test material with nonzero (n,2n) through the standard API.
- The error only appeared when a custom `Mixture` with nonzero `Sig2`
  was constructed directly and compared against the dense eigensolver.

**First wrong fix attempt:** Added `n2n_production` to the numerator:
`k = (νΣf + Σ₂_production) / Σa`.  This gives 1.808, still wrong.  The
issue is that `Σa` already includes `Σ₂_out` as removal, but for the
eigenvalue balance the (n,2n) appears as a *source* (2 neutrons out), not
a removal.  The correct denominator is `Σt - Σs - 2Σ₂`, not `Σa`.

**L0 test that catches it:** `test_cp_verification.py::TestN2N::
test_n2n_solver_keff_matches_analytical` — constructs a 2G material with
`Sig2[0,0] = 0.01`, computes the analytical eigenvalue via
`kinf_from_cp(..., sig_2_mats=[sig_2])`, and compares against the
solver.  Tolerance: 1e-5.

**Lesson:** The eigenvalue estimate formula `production / absorption`
hides the implicit assumption Σ₂ = 0.  When adding a new reaction type,
trace it through BOTH the transport solve (where it's a source) AND the
eigenvalue estimate (where it changes the production/removal balance).
The two must be consistent.  Testing with zero cross sections hides the
inconsistency — need at least one test with the term nonzero.

---

## ERR-016 — Tautological inner-iteration convergence residual

**Failure mode:** #6 Convention drift — convergence check tests identity, not convergence  
**Date:** 2026-04-05  
**Solver:** CP Gauss-Seidel mode

**Bug:** The GS inner convergence check computed:

    phi_g_new = transported_g / denom_g
    collision_rate_g = denom_g * phi_g_new
    res = ||collision_rate_g - transported_g||

Substituting the first line into the third: `denom_g * (transported_g /
denom_g) - transported_g = 0`.  The residual is identically zero
regardless of whether the within-group scattering has converged.  The
inner loop always exited after exactly 1 iteration.

**Impact:** The GS solver mode was functionally identical to a
sequential-group Jacobi update.  Groups with strong within-group
self-scatter (thermal groups where Σs(g→g)/Σt(g) is large) were NOT
iterating to convergence.  The outer power iteration still converged to
the correct eigenvalue (because it does its own convergence check), but
the GS mode provided no acceleration benefit from inner iterations.

**How it hid from higher-level tests:**
- All 27 eigenvalue tests passed because the outer iteration converged
  to the correct answer regardless of inner iteration count
- The diagnostic `n_inner` array showed values of 1 everywhere, which
  was interpreted as "fast convergence" rather than "broken residual"
- The `test_thermal_group_needs_more_inner_iterations` test used `>=`
  (thermal ≥ fast), which passed vacuously since all values were 1
- The AI QA review initially concluded that inner iterations are
  *fundamentally unnecessary* for the CP method (wrong — the source
  depends on the flux through self-scatter)

**L0 test that catches it:** `test_cp_verification.py::TestGSInnerIterations::
test_no_self_scatter_one_inner` — material with zero diagonal in Σs
should converge in ≤ 2 inner iterations (no self-consistency needed).
`test_thermal_needs_more_inner_than_fast` — with the corrected residual,
thermal groups genuinely need more inner iterations than fast groups.

**Fix:** Changed residual to relative flux change:
`||φ_new - φ_old|| / ||φ_new||`.  This is nonzero when within-group
self-scatter changes the source between iterations, and zero when the
source doesn't depend on the current group's flux.

**Lesson:** A convergence check that compares quantities derived from
each other by construction tests nothing.  The residual must compare
*independent* quantities: the old flux vs the new flux, or the old
source vs the recomputed source.  When a "convergence diagnostic" shows
all 1s, that's a red flag — it could mean instant convergence OR a
tautological check.  Distinguish the two by testing with a problem that
*should* require multiple iterations (strong self-scatter).

---

## ERR-017 — Wigner-Seitz pitch formula doubled in MC heterogeneous tests

**Failure mode:** #3 Missing factor — extra factor of 2  
**Date:** 2026-04-06  
**Solver:** Monte Carlo (test suite)

**Bug:** The MC heterogeneous test computed the square unit cell pitch as
`pitch = r_cell * sqrt(pi) * 2` instead of `pitch = r_cell * sqrt(pi)`.
The correct formula equates the square cell area to the Wigner-Seitz
circle area: `pitch^2 = pi * r_cell^2`, giving `pitch = r_cell * sqrt(pi)`.
The factor of 2 quadrupled the cell area.

**Impact:** The extra area was all moderator, which drastically changed the
neutron economy.  For the 1G 2-region case: k_mc = 0.757 vs k_ref = 0.990
(24% systematic error).  For the 2G 2-region case, the population collapsed
to zero (NaN) because the subcritical system with 4× moderator couldn't
sustain a neutron population at 200 neutrons/cycle.

**How it hid from higher-level tests:**
- All homogeneous tests passed (single material everywhere — pitch is
  irrelevant for delta-tracking in a homogeneous medium)
- The tests were all marked `@pytest.mark.slow` and may not have been
  run regularly
- The z-scores were NaN (0/0 from collapsed population) which fails
  the `< 5.0` assertion but doesn't indicate which direction the error is
- The error looked like "MC can't handle subcritical systems" rather than
  "the geometry is wrong"

**L0 test that catches it:** Direct comparison of pitch against the factory
convention: `pwr_pin_equivalent` uses `r_cell = pitch / sqrt(pi)`, so
inverting gives `pitch = r_cell * sqrt(pi)`.  A unit test asserting
`pitch**2 == pi * r_cell**2` would immediately flag the factor of 2.

**Lesson:** When constructing geometry for cross-method comparison, verify
the cell area/volume matches between the two methods.  A factor-of-2 error
in a linear dimension is a factor-of-4 in area — large enough to change
the qualitative physics (supercritical → subcritical), yet small enough to
be invisible in the code review because `* 2` looks like it "corrects for
a half-cell to full-cell conversion" or "accounts for the diameter vs
radius convention".

---

## ERR-018 — Direction sampling uses uniform theta instead of isotropic

**Failure mode:** #4 Factor error — wrong PDF for spherical sampling  
**Date:** 2026-04-06 (identified during L0 test design)  
**Solver:** Monte Carlo

**Bug:** The solver samples the polar angle as `theta = pi * rng.random()`
(uniform in [0, π]) instead of `theta = arccos(1 - 2*xi)` (uniform on the
unit sphere).  True isotropic sampling requires the PDF `p(theta) =
sin(theta)/2` to account for the solid angle Jacobian.

**Impact:** The uniform-theta sampling overweights the poles (theta ≈ 0 and
theta ≈ π) where `sin(theta)` is small.  For the 2D projection used by the
solver: `E[sin^2(theta)] = 1/2` (uniform) vs `2/3` (isotropic).  This
systematically shortens the average 2D step length by ~19%.

**Classification:** Known simplification, not a bug to fix.  The formula
matches the original MATLAB `monteCarloPWR.m` implementation.  Since the
solver only tracks 2D projections (x, y) and uses periodic BCs on a square
cell, the non-isotropic sampling affects the effective mean free path but
does not invalidate the eigenvalue calculation (it changes the effective
geometry scaling, which is absorbed into the keff estimate).

**L0 test that documents it:** `test_mc_properties.py::test_direction_sampling`
verifies `E[dir_x^2] = 1/4` (the formula's prediction, not the isotropic
1/3), confirming the code matches the INTENDED formula.

**Lesson:** When porting from MATLAB, document which simplifications are
intentional vs accidental.  A sampling formula that "looks wrong" may be
a deliberate approximation that the original author validated empirically.
The L0 test should verify the INTENDED formula, not the physically correct
one, and the documentation should explain the distinction.

---

## ERR-019 — Missing 4π·sin(θ) weight factor in MOC scalar flux update

**Failure mode:** #3 Missing factor — incomplete angular integration weight  
**Date:** 2026-04-06  
**Solver:** MOC (2D pin cell)

**Bug:** The MOC transport sweep accumulated `delta_phi` with weight
`omega_a * omega_p * t_s` instead of the correct
`4*pi * omega_a * omega_p * t_s * sin(theta_p)`.  Two factors were
missing: (1) the `4*pi` from the angular flux → scalar flux integral
(`phi = integral_{4pi} psi dOmega`), and (2) the `sin(theta_p)` that
arises because the 2D segment-averaged angular flux `bar_psi` relates
to the 3D path integral via `bar_psi = Q/Sig_t + delta_psi * sin(theta) / (Sig_t * ell)`.

The scalar flux update formula (Boyd et al. 2014, Eq. 45) is:

    phi_i = (4*pi / Sig_t_i) * [Q_i + (1/A_i) * sum omega_a * omega_p * t_s * sin(theta_p) * delta_psi]

The `4*pi` factor multiplies the entire bracket. When delta_phi is defined
as `sum(4*pi * omega_a * omega_p * t_s * sin_p * delta_psi)`, the update
becomes `phi = (4*pi*Q + delta_phi/A) / Sig_t`.

**Impact:** Heterogeneous keff was completely wrong: MOC gave 1.344 vs
CP reference of 0.902 for a 2-region fuel+coolant pin cell (1G).  The
homogeneous case was UNAFFECTED because `delta_psi = 0` when the angular
flux is spatially uniform (all boundary fluxes equal `Q/Sig_t`).

**How it hid from homogeneous tests:**
- For homogeneous material with converged boundary fluxes,
  `psi_in = Q/Sig_t` everywhere → `delta_psi = 0` → `delta_phi = 0`
- `phi = 4*pi*Q/Sig_t` regardless of the weight factor
- 1G: k = nu*SigF/SigA (weight-independent)
- 2G/4G: matrix eigenvalue (still weight-independent for uniform medium)
- All 3 homogeneous eigenvalue tests passed to machine precision

**L0 test that catches it:** `test_moc_verification.py::TestL0EquilibriumFlux::
test_pure_scatterer_equilibrium_single_sweep` — injects a non-trivial
boundary flux and checks that the resulting scalar flux matches the
analytical value.  With wrong weights, the correction term `delta_phi/A`
has the wrong magnitude and the flux deviates.  The heterogeneous
particle balance test also catches it immediately (production/absorption ≠ keff).

**Lesson:** The angular integration weight in MOC contains problem-specific
factors (`4*pi` from the full-sphere integral, `sin(theta_p)` from the
2D→3D projection) that cancel out for spatially uniform solutions.  This
makes the missing factor invisible to homogeneous tests.  ALWAYS test the
transport sweep with a heterogeneous problem before declaring the weight
formula correct.  The Boyd Eq. 45 formula should be verified term-by-term
against the derivation, not just checked for self-consistency on the
homogeneous case.

---

## ERR-020 — ULP-noisy cell volumes from `cbrt → **3` round trip

**Failure mode:** #3 Missing factor — numerical round-trip through a
non-bijective float64 operation destroys a structural invariant.
**Date:** 2026-04-13
**Solver:** `orpheus.geometry` (all consumers via `Mesh1D.volumes`)

**Bug:** `_subdivide_zone` constructed equal-volume spherical edges via
`r_k = cbrt(inner^3 + k/n * (outer^3 - inner^3))`, and `compute_volumes_1d`
then re-derived cell volumes as `(4/3) π · diff(edges**3)`. Because
`cbrt(x) ** 3 != x` at the ULP level in float64, the reconstructed
`edges**3` values drifted ~1 ULP per cell, so cells in a zone that were
*supposed* to have identical volume by construction drifted by up to
~2.2e-14 relative error. The cylindrical path (`sqrt → **2`) had the
same bug but at ~6.7e-15 — just under the common `rtol=1e-14` threshold,
which is why only the spherical case failed visibly.

The specific failing tests:

    tests/geometry/test_geometry.py::TestZoneSubdivision::test_equal_volume_single_zone[SPHERICAL]
    tests/geometry/test_geometry.py::TestZoneSubdivision::test_equal_volume_multi_zone[SPHERICAL]

reported volumes of `7.260569688296488` vs reference `7.260569688296414`
— a relative difference of `1.03e-14` against an `rtol=1e-14` assertion.

**Impact:** None observed in solver eigenvalues (the relative drift is
well below every physics tolerance in the repo), but the assertion
`"all cells in an equal-volume zone are bit-identical"` was broken,
masking an invariant a future bug could violate more seriously without
detection. Fixing it also tightens the cylindrical path from ~7e-15 to
bit-exact, eliminating a hidden source of noise in CP/SN/MOC
quadrature-weighted integrals over spherical/cylindrical meshes.

**How it hid:** Every downstream consumer tolerates ULP-level volume
drift (physics tolerances are ≥1e-10 for eigenvalues, ≥1e-8 for flux
shapes). The only test that asserted bit-exactness was the geometry
invariant test itself, and it was introduced late enough that the
spherical path had never been pushed to `rtol=1e-14`. The cylindrical
path accidentally passed.

**Fix:** Compute cell volumes **from the algebraic invariant** at
subdivision time, not from the edges after the fact. `_subdivide_zone`
now returns `(edges, volumes)`; for each coordinate system:

* Cartesian:   `V_cell = (outer - inner) / n`
* Cylindrical: `V_cell = π · (outer² - inner²) / n`
* Spherical:   `V_cell = (4/3) π · (outer³ - inner³) / n`

One scalar per zone, broadcast to every cell — no round trip through
`sqrt`/`cbrt`, so every cell in an equal-volume zone is bit-identical
by construction. `Mesh1D` gained an optional
`precomputed_volumes` field that overrides the edge-derived default;
`mesh1d_from_zones` populates it. Manually-constructed meshes with
arbitrary edges continue to fall back to `compute_volumes_1d` from
edges.

**L0 test that catches it:**
`tests/geometry/test_geometry.py::TestZoneSubdivision::test_equal_volume_{single,multi}_zone`
for every `CoordSystem` — enforces bit-equal volumes at `rtol=1e-14`.

**Lesson:** Non-bijective float operations (`sqrt`, `cbrt`, `exp`,
`log`) do not survive a round trip, and invariants that *should* hold
algebraically are not free — they must be preserved by design. When
an invariant of the form "X_i == X_j for all i, j" exists, compute X
once and broadcast, don't compute it N times and hope for the best.
Fishbone: whenever you see `op(op_inverse(x))` in the code, ask
whether the inverse is bit-exact, and if not, refactor to avoid the
round trip.

---

## ERR-021 — Degenerate ray tangent to pin-cell corner raises IndexError

**Failure mode:** #5 Index error — unchecked assumption that the
intersection list always has ≥2 entries.
**Date:** 2026-04-14
**Solver:** MOC (`orpheus.moc.geometry._ray_box_intersections`)

**Bug:** `_ray_box_intersections` walks the four walls of the square
pin cell `[0, pitch]^2`, collects every wall crossing whose hit point
lies inside the wall segment, deduplicates by `s`-tolerance, and then
indexed `s_vals[0]` and `s_vals[1]` as entry/exit. If the ray grazes a
*corner* of the box, the two adjacent wall solutions collapse to the
same `s` value; after the `1e-12` dedup pass only **one** entry
remains, and `s_vals[1]` raises `IndexError`. The same failure mode
triggers if the ray is parallel to one axis and offset so neither
orthogonal wall yields an in-range hit — the list can even start
empty. The half-step offset in the track generator makes an exact
corner hit vanishingly unlikely in normal use, but the guard was
missing.

**Impact:** None observed in production (no test ever seeded a ray
exactly through a corner). A seeded pitch or azimuthal angle that
aligned a ray with a corner would crash ray tracing before any flux
solve, masking the degeneracy as a random `IndexError` rather than a
skippable geometric edge case.

**Fix:** `_ray_box_intersections` now returns
`tuple | None`: `None` signals a degenerate ray (empty `s_vals`, or
fewer than two distinct entries after dedup). `_trace_single_ray`
short-circuits a `None` to `([], (x0, y0), (x0, y0), -1, -1)`, and
`MOCMesh._trace_all_rays` already skips tracks with `if not
segments: continue`, so a degenerate ray is now silently dropped
instead of aborting the trace.

**L0 test that catches it:**
`tests/moc/test_ray_tracing.py::test_degenerate_corner_ray` — seeds a
ray with `(x0, y0) = (0, 0)` and `phi = π/4` so the entry point is
exactly the `(0, 0)` corner, and asserts that
`_ray_box_intersections` returns `None` and `_trace_single_ray`
returns empty segments rather than raising.

**Lesson:** Whenever a function indexes a collection by a fixed
position (`list[0]`, `list[1]`), the *precondition* that the
collection has that many entries must be either enforced by
construction or checked explicitly. Geometric primitives in
particular must handle degenerate inputs (tangent, parallel,
coincident) as first-class cases, not crashes — in a ray tracer,
"this ray contributes nothing" is a valid outcome.

---

## ERR-022 — Negative lethargy bin width flips flux-per-lethargy sign

**Failure mode:** #6 Convention drift — sign of `du` depends on the
energy-grid ordering convention, which the callers silently relied on.
**Date:** 2026-04-14
**Solver:** MC (`orpheus.mc.solver.solve_monte_carlo`), homogeneous
spectrum solver (`orpheus.homogeneous.solver`), MOC plotting helper
(`orpheus.plotting.plot_moc_spectra`).

**Bug:** Group boundaries in ORPHEUS follow the standard nuclear-data
convention of *descending* energy (`eg[0]` = fast edge, `eg[-1]` =
thermal edge), so the lethargy widths
`du[g] = log(eg[g+1] / eg[g])` are **negative**. Three call sites
divided the (non-negative) group tally by this signed `du` to get
`flux_per_lethargy`, producing uniformly negative values:

* `orpheus.mc.solver.solve_monte_carlo`:
  `flux_per_lethargy = tally / xs.du`
* `orpheus.homogeneous.solver.HomogeneousResult.flux_per_lethargy`:
  `self.flux / self.du` (where `du` was stored signed)
* `orpheus.plotting.plot_moc_spectra`: `flux / du` for fuel, clad,
  coolant spectra

The homogeneous solver also stored `de = eg[1:] - eg[:-1]` as a
signed value, so `flux_per_energy` had the same sign flip.

**Impact:** None on eigenvalues — `flux_per_lethargy` is used only
for spectrum visualization, never fed back into a solver. The
visual output of every MC / MOC / homogeneous spectrum plot was
mirror-flipped through `y = 0`, which readers were silently
compensating for by reading the magnitudes and ignoring the sign.

**Fix:** Take the absolute value at the *definition* site of
`du` / `de`, not at the consumer site, so "lethargy bin width" and
"energy bin width" are non-negative by construction regardless of
grid ordering:

    du = np.abs(np.log(eg[1:] / eg[:-1]))
    de = np.abs(eg[1:] - eg[:-1])

Applied in `orpheus.mc.solver` (MCResult assignment),
`orpheus.homogeneous.solver._spectrum_result`, and
`orpheus.plotting.plot_moc_spectra`.

**L0 test that catches it:**
`tests/mc/test_gaps.py::test_flux_per_lethargy_nonnegative` — runs a
small MC with a descending two-group grid, asserts
`result.flux_per_lethargy >= 0` element-wise.

**Lesson:** "Width" quantities should be non-negative by convention,
and sign should be a property of a *direction*, not of a *measure*.
When the same quantity is computed at three different call sites, fix
it at the *definition* site — the code equivalent of normalizing at
the source rather than patching every consumer. Every consumer is an
opportunity for the bug to resurface.

This also reinforces ERR-006 / Meta-Lesson 6: convention-dependent
values (like "is `eg` ascending or descending?") must be pinned down
at a single source of truth, and every helper must be robust to the
convention, not assume it.

**Scope note:** This does *not* address the separate design question
in issue #25 about whether the MC tally should be a scattering
estimator (`w/Σ_s` on scatters, current behavior) or a collision
estimator (`w/Σ_t` on all collisions). That is a choice, not a bug —
both are unbiased estimators of the scalar flux. The sign-flip was
the genuine bug.

---

## ERR-023 — MC solver silently ignores Sig2 (n,2n) reactions

**Failure mode:** #6 Untested code path — every existing MC test
material had `Sig2 = 0`, so the missing branch was invisible.
**Date:** 2026-04-15
**Solver:** MC (`orpheus.mc.solver._random_walk`,
`orpheus.mc.solver._precompute_xs`).

**Bug:** The random walk only computed `sig_t = sig_a + sig_s_sum`
and used a two-way branch between absorption and scatter. The `mat.Sig2`
matrix was never touched — no reaction was sampled and no weight
doubling occurred. At the same time `_precompute_xs` seeded the
majorant with `mix.SigT`, which by the project's convention
(`orpheus.data.macro_xs.mixture._compute_mixture`, line 142) already
includes one copy of `Sig2.sum(axis=1)`. The mismatch meant the Σ_2n
fraction of the majorant was effectively *always* rejected as a
virtual collision: the particle free-flighted past (n,2n) sites
without ever sampling them. Net effect: zero (n,2n) contribution to
the scattering kernel, whereas the CP solver correctly includes
`2·Sig2·φ` as a source (anti-ERR-015).

**Impact:** Bias on `keff` whenever a material has nonzero (n,2n).
For the 2 G Region-A fixture with `Sig2[0,0] = 0.01`, the
analytical `k_inf` is 0.817 vs 0.800 with Sig2 = 0 — a 2 % shift
that the MC was unable to reproduce.

**Fix:** In `_random_walk`, compute `sig_2n_row = Sig2[ig, :]` and
`sig_2n_sum = sig_2n_row.sum()`, set
`sig_t = sig_a + sig_s_sum + sig_2n_sum`, and add a third branch in
the collision decision:

    r = rng.random() * sig_t
    if r < sig_s_sum:            ... # scatter
    elif r < sig_s_sum+sig_2n:   w *= 2.0; sample exit from Sig2 row
    else:                        ... # absorb / fission

The majorant stays as `mix.SigT` (unchanged) — because mixture.SigT
already carries `Σ_2n.sum` once, no `2·` factor is needed at the
majorant level. The weight doubling inside the (n,2n) branch is the
analog-MC convention for "one reaction, two neutrons emitted."

**L1 test that catches it:**
`tests/mc/test_gaps.py::test_mc_n2n_keff_matches_analytical` — builds
the 2 G Region-A mixture with `Sig2[0,0] = 0.01`, solves a scipy
generalised eigenvalue problem with effective loss
`SigT − Σ_s^T − 2·Σ_2n^T`, and checks that the MC keff matches to
`5σ + 5·10⁻³`. Also checks that the MC has moved at least halfway
from the Sig2 = 0 baseline toward the (n,2n) reference.

**Lesson:** Reinforces Meta-Lesson 6 (zero cross sections hide bugs).
A structurally correct-looking `sig_t = sig_a + sig_s_sum` is only
correct if *every* term in the project's total-XS definition is
accounted for. When a mixture field (here `Sig2`) is **never read**
inside a transport kernel, that field is silently dropped on the
floor — and every downstream test that happens to use a zero value
for it gives false confidence.

---

## ERR-024 — MC flux tally: scattering estimator instead of collision estimator

**Failure mode:** #3 Design error rather than term error — the tally
was a well-defined *scattering* estimator but the output field
`flux_per_lethargy` claimed to represent the scalar flux.
**Date:** 2026-04-15
**Solver:** MC (`orpheus.mc.solver._random_walk` tally accumulation).

**Bug:** On each real scattering event the solver accumulated
`tally[ig] += w / sig_s_sum`. This is the textbook *scattering*
estimator for a response-like integral weighted by Σ_s, but the
`MCResult.flux_per_lethargy` field was divided only by `|du|` and
treated as a scalar flux by plotting and by
`tests/mc/test_gaps.py::test_2g_flux_ratio_homogeneous`. Absorption
events contributed nothing, and the per-event weighting was
`1/Σ_s` instead of the `1/Σ_t` required by a collision estimator.
The existing spectral test had to be loosened to `ratio > 0.1` to
accommodate the bias (issue #25).

**Impact:** None on keff (the eigenvalue is computed from the
weight ratio in :eq:`keff-cycle` and never touches the tally). Bias
on every flux-spectrum plot, proportional to the relative shape
difference between Σ_s and Σ_t across groups. For Region A the
shape distortion is visible at the ~10 % level.

**Fix:** Move the tally inside the "real collision" branch
*before* the scatter-vs-(n,2n)-vs-absorb decision and use

    tally[ig] += w / sig_t

where `sig_t` is the real total (not the majorant). This is the
standard collision estimator and is unbiased for any combination of
reactions being sampled.

**L1 test that catches it:**
`tests/mc/test_gaps.py::test_2g_flux_ratio_homogeneous` — now
compares the MC flux shape (per-group, normalised) against the
analytical eigenvector from
`scipy.linalg.eig(F, diag(SigT) - Σ_s^T - 2·Σ_2n^T)` with
`rtol = 10 %`. The previous scattering-estimator bias cannot satisfy
this tolerance because the `Σ_s / Σ_t` ratio differs across groups.

**Lesson:** Unbiasedness alone is not a specification. A scattering
estimator and a collision estimator are *both* unbiased for their
respective integrals, but only one of them estimates the *flux*. If
a result field is labelled `flux_*`, the code that produces it must
be a flux estimator — and that contract should be enforced by a
spectral test, not a marginal `> 0.1` placeholder.

---

## ERR-025 — Diamond-difference cumprod recurrence: missing −Σ_t in numerator and missing 1/W source normalization

**Failure mode:** #3 Missing factor + #4 Factor error (two
compensating factor-of-two errors that cancel for homogeneous
problems)
**Date:** 2026-04-16
**Solver:** SN (`orpheus.sn.sweep._sweep_1d_cumprod`, 1D Cartesian
Gauss-Legendre fast path).

**Bug:** The precomputed face-flux recurrence coefficients were

    a = 2μ / (2μ + Δx·Σ_t)           # WRONG — missing −Σ_t in numerator
    b = 0.5·Δx·Q / (2μ + Δx·Σ_t)     # WRONG — missing 1/W, extra factor 0.5

instead of the canonical diamond-difference (DD) recurrence derived
symbolically in `orpheus.derivations.sn_balance.derive_cumprod_recurrence`:

    a = (2μ − Δx·Σ_t) / (2μ + Δx·Σ_t)
    b = 2·Δx·(Q/W) / (2μ + Δx·Σ_t)

where `W = Σ w_n` is the quadrature weight sum. The `1/W` factor is
needed because `SNSolver._add_scattering_source` produces `Q` in
scalar-flux units while the per-ordinate transport equation sees
`Q/W` on the right-hand side — the same normalization
`_sweep_2d_wavefront` already applied via its `weight_norm = 1/W`
factor (`Q_scaled = Q * weight_norm`). The 1D fast path had been
independently derived without that normalization, and its `a`
formula had an additional sign error in the numerator.

**Why the two errors cancel for homogeneous problems:** the fixed
point of the buggy recurrence is `ψ = Q/(2Σ_t)`, half the correct
`ψ = Q/Σ_t`. The missing `1/W = 1/2` for Gauss-Legendre on `[−1, 1]`
rescales by exactly 2, turning `Q/(2Σ_t)` back into `Q/Σ_t` per
ordinate. The resulting scalar flux is correct up to a uniform
rescaling by `Σ_t(x)`. For eigenvalue problems with a single
material this is invisible, because the Rayleigh quotient
`k = νΣ_f·φ / Σ_a·φ` is invariant under a uniform rescaling of φ.
At a material interface the rescale factor depends on which side
of the interface you are on, so the cancellation breaks and k_eff
shifts.

**Impact:** ~1.48 × 10⁻² error in k_eff on the ORPHEUS Phase 2.1b
2-region A+B reflective slab (fuel Σ_t=1, Σ_s=0.5, νΣ_f=0.75; mod
Σ_t=2, Σ_s=1.9, νΣ_f=0; reflective BCs). Case
singular-eigenfunction reference and CP slab E₃ kernel both give
k ≈ 1.27461, while the buggy solver converged to ≈ 1.25988.

**How it hid from higher-level tests:**
- Homogeneous single-region k_inf tests: exact to machine precision
  (uniform rescaling of φ is eigenvalue-invariant).
- Same-material two-region tests: exact for the same reason.
- Smooth-Σ MMS verification (Phase 2.1a): passed cleanly because
  the MMS consumer test uses `solve_sn_fixed_source`, which goes
  through the `_sweep_2d_wavefront` path with the correct
  `weight_norm = 1/W`.
- Self-referencing Richardson convergence tests on heterogeneous
  problems: saw clean O(h) convergence **to the wrong asymptote**,
  because the Richardson reference was built from the same buggy
  solver. This is exactly the T3 dead-end pattern documented in
  `docs/theory/diffusion_1d.rst` "Investigation history".

**Fix:** `orpheus/sn/sweep.py:119-140` — replaced the wrong
coefficients with the canonical DD recurrence, added a
source-of-truth comment pointing at `derive_cumprod_recurrence`.
One-formula correction; nothing downstream of the coefficients
needed changes.

**Evidence after fix:**

| Method                           | k_eff       |
|----------------------------------|-------------|
| Case singular-eigenfunction (S8) | 1.27461604  |
| CP slab E₃ kernel (converged)    | 1.27442847  |
| solve_sn S8 @ n_per=320 post-fix | 1.27461601  |

Case ↔ solve_sn agreement at matching quadrature order improved
from 1.48 × 10⁻² to 3.4 × 10⁻⁸.

**L1 test that catches it:**
`tests/sn/test_cartesian.py::test_heterogeneous_absolute_keff` — pins
the 2-region A+B reflective slab against the Case singular-eigenfunction
reference to 5 × 10⁻⁴. Without a material interface the bug is
invisible (the Rayleigh quotient's rescale invariance hides it), so
this test is the minimal configuration that exposes it.

---

## ERR-026 — Curvilinear sweep WDD angular closure converges to wrong fixed-source solution

**Failure mode:** #6 Wrong answer accepted — sweep converges to a
stable, balance-satisfying solution that is NOT the correct discrete
transport solution
**Date:** 2026-04-17
**Solver:** SN (`orpheus.sn.sweep._sweep_1d_spherical`,
`_sweep_1d_cylindrical`).

**Bug:** The curvilinear sweeps use a one-directional WDD angular
face-flux closure:

    ψ_{n+1/2} = (ψ_n − (1−τ)·ψ_{n−1/2}) / τ

while the BiCGSTAB transport operator
(`build_transport_linear_operator_spherical`) uses a symmetric closure:

    ψ_{n+1/2} = τ·ψ_{n+1} + (1−τ)·ψ_n

Both are consistent for flat flux analytically. But the sweep's
one-directional WDD, combined with the zero-area face at r=0
(which eliminates spatial coupling at the innermost cell), creates
a system where the iterative sweep converges to a NON-FLAT solution
that still satisfies the discrete balance equation.

**Impact:** `solve_sn_fixed_source` on spherical / cylindrical meshes
produces 35–50% error at cell 0, **growing** with mesh refinement
(divergent, not convergent). MMS verification (Phase 3.3–3.4) is
blocked. Global conservation is exact despite the wrong spatial profile.

**How it hid from higher-level tests:**
- Eigenvalue solver routes to BiCGSTAB for curvilinear geometry — the
  sweep is never exercised by the eigenvalue path
- 1-group k_eff is shape-independent (Rayleigh quotient invariance)
- Multi-group eigenvalue tests use 2% tolerance that absorbs the sweep
  error
- No fixed-source tests existed for curvilinear geometry until the
  Phase 3.3 MMS attempt

**Evidence:**

| Test (constant source, reflective BCs) | Sweep    | BiCGSTAB |
|-----------------------------------------|----------|----------|
| φ at cell 0 (nx=20)                    | 0.64     | 1.000    |
| Error vs refinement                    | Diverges | Zero     |
| Conservation (volume-weighted average)  | Exact    | Exact    |
| Cartesian sweep (same test)             | Exact    | N/A      |

**Fix:** Route `solve_sn_fixed_source` through BiCGSTAB for curvilinear
geometry, adding an external-source slot to `build_rhs_spherical` /
`build_rhs_cylindrical`. GitHub Issue #98 tracks the fix, Issue #99
tracks the blocked MMS verification.

**L1 test that catches it:**
`tests/sn/test_sweep_operator_inconsistency.py::test_spherical_sweep_vs_bicgstab_flat_flux`
— runs constant-source reflective-BC problem via both sweep and BiCGSTAB
and asserts BiCGSTAB is exact while documenting the sweep's deviation.

A cheaper L0 alternative — a direct unit test of the fixed-point
of the 1D cumprod recurrence on a 1-cell uniform-material slab
that would catch the factor-of-2 in milliseconds without needing
any reference solver — is tracked for a future commit as part of
the broader derivation-implementation audit (issue #95).

**Lesson:** When a symbolic derivation module exists
(`orpheus.derivations.sn_balance.derive_cumprod_recurrence`), its
output is the source of truth and the implementation must visibly
match. A comment in the implementation pointing back at the
derivation function would have caught this at review time. Two
opposite-sign factor-of-two errors cancelled exactly for eigenvalue
problems because the only thing that matters to
`k = νΣ_f·φ / Σ_a·φ` is the *shape* of φ, not its scale — and
factor-of-Σ_t(x) cancellations hide everywhere except at material
interfaces. Phase 1.2 learned the same lesson for diffusion
(hardcoded tolerances masking quadratic convergence); Phase 2.1b
is the same pattern repeated for SN. See GitHub issue #95 for the
follow-up audit work checking every solver implementation against
its derivation.

---

## ERR-027 — Peierls slab K-matrix: naive GL collocation for cross-panel entries

**Failure mode:** #3 Missing factor — missing quadrature resolution
(one-point rule where adaptive is required)
**Date:** 2026-04-19
**Solver:** CP / Peierls (`orpheus.derivations.peierls_slab._build_kernel_matrix`).

**Bug:** For cross-panel entries ``K[i, j]`` (observer node *i* and
source node *j* in different panels) the assembly used the one-point
collocation rule

    K[i, j] = (1/2) * E_1(σ_t * |x_i - x_j|) * w_j

i.e. the integral ∫_panel E_1(σ_t|x_i - x'|) L_j(x') dx' was
approximated by evaluating the kernel at the single node x_j and
multiplying by the GL weight w_j. This is only exact when the
integrand R(σ_t|x_i - x'|) × L_j(x') is polynomial of degree ≤ 2p−1.
E_1 is transcendental with a near-log spike at x' → x_i; the one-point
rule leaves ~1% quadrature error even for panel pairs with modest
optical separation, worst when x_i is within a small optical distance
of the source panel boundary (where the near-log spike sits just
outside the panel).

**Impact:** K[i, j] cross-panel entries wrong by 0.5–1.5% at 2 panels
× p=4; refining panels reduces this only at O(h) rate (non-spectral).
Downstream: slab k-eff tests showed ~0.4% tie-point offsets in the
Sanchez R≈1.98 case.

**How it hid from higher-level tests:**
- Row-sum K·[1, 1, …] is exact to O(1e-15) even with buggy cross-panel
  entries because ∑_j L_j(x') = 1 (partition of unity) kills the kink
  in the summed integrand. Every existing row-sum test was blind.
- k-eff test tolerance was 2% — absorbed the ~0.4% propagated error.
- 1-group and uniform-source tests also cancel the bug via partition
  of unity.

**L1 test that catches it:**
`tests/derivations/test_peierls_reference.py::TestSlabKMatrixElementwiseVsReference::test_cross_panel_boundary_neighbour_elementwise`
— element-wise `K[4, 3]` at n_panels=2, p=4 vs the adaptive
`slab_K_vol_element` reference at 1e-10.

**Fix:** Unified basis-aware Nyström assembly — every ``K[i, j]`` is
``(1/2) ∫_panel E_1(τ(x_i, x')) L_j(x') dx'`` evaluated via adaptive
``mpmath.quad``. Mirrors the adaptive reference
`peierls_reference.slab_K_vol_element`. See issue #113.

**Lesson:** "Row-sum conservation" tests are systematically blind to
basis-individual quadrature errors that happen to sum to zero under
partition-of-unity of the Lagrange basis. Element-wise K[i, j]
verification against an adaptive reference is the only reliable
L0/L1 gate for Nyström kernel assembly.

---

## ERR-028 — Peierls slab K-matrix: GL collocation of remainder R(τ) has unresolved kink at x'=x_i

**Failure mode:** #3 Missing factor — missing subdivision hint
**Date:** 2026-04-19
**Solver:** CP / Peierls (`orpheus.derivations.peierls_slab._build_kernel_matrix`).

**Bug:** The same-panel singularity-subtraction branch split the kernel
into ``E_1(τ) = R(τ) − ln τ − γ`` with ``R(τ) = E_1(τ) + ln τ + γ``
the smooth remainder, then used GL collocation for the ``R`` integral
and exact product-integration weights for the ``−ln`` integral. The
flaw: ``R(σ_t|x_i - x'|)`` is smooth in τ but its argument ``|x_i - x'|``
has a C⁰ kink in x' at x'=x_i. GL cannot integrate a function with a
derivative discontinuity in the interior of the interval — produces
~1% error on the diagonal panel.

**Impact:** Diagonal-panel entries (~40% of all K entries for p=4,
n_panels=8) wrong by ~1% relative. The row-sum identity survives by
partition of unity (see ERR-027), but element-wise K[i, i] deviates
by ~1.2e-2 at n=2, p=4.

**How it hid from higher-level tests:** Same cause as ERR-027 —
partition-of-unity of the Lagrange basis cancels the kink in the
row-sum, hiding from every conservation-style test.

**L1 test that catches it:**
`tests/derivations/test_peierls_reference.py::TestSlabKMatrixElementwiseVsReference::test_small_case_elementwise_agreement`
— element-wise K[i, j] including diagonal entries vs the adaptive
`slab_K_vol_element` reference at 1e-10.

**Fix:** Unified with ERR-027: adaptive `mpmath.quad` with the
subdivision hint ``[panel_a, x_i, panel_b]`` for same-panel entries.
mpmath handles both the log singularity and the derivative kink
natively. See issue #113.

**Lesson:** Singularity subtraction splits a kernel into "smooth" and
"singular" parts — but "smooth in τ" is not the same as "smooth in x'".
A change of variables that folds the singularity into a kink instead
of removing it merely trades one unresolved feature for another. The
adaptive-with-hint approach is more robust and unifies all cases.

---

## Meta-Lessons

1. **1-group is degenerate.** k = νΣ_f/Σ_a regardless of flux shape.
   Every solver MUST be verified at ≥2 groups.

2. **Homogeneous is degenerate.** Redistribution cancels for flat flux.
   Every curvilinear solver MUST be verified on heterogeneous problems.

3. **Per-ordinate consistency is fundamental.** L0-SN-003 would have
   caught ERR-006 (the most expensive bug) immediately.

4. **20 passing tests don't mean correct.** The cylindrical sweep
   passed 20 tests including homogeneous exact, particle balance, and
   conservation.  The bug survived because none tested what mattered:
   heterogeneous eigenvalue convergence with mesh refinement.

5. **Test with the pathological case.** Every physics feature has a
   regime where bugs are exposed.  For curvilinear: test near r=0 with
   mesh refinement.  For scattering: test with asymmetric matrices.
   For normalization: test with multiple quadrature types.

6. **Zero cross sections hide bugs.** If a reaction type (n,2n,
   upscatter) is zero in all test materials, every code path touching
   it is untested.  ERR-015 survived because `Sig2 = 0` everywhere.
   For every XS term, there must be at least one test where it's nonzero.

7. **A tautological residual proves nothing.** If the convergence
   check computes `f(x) - g(f(x))` where `g` is the inverse of the
   step that produced `f(x)`, the residual is identically zero.
   ERR-016 survived because "all inner iterations = 1" was mistaken
   for fast convergence.  Always verify with a problem that SHOULD
   require multiple iterations.

8. **Geometry area/volume must match across methods.** When
   comparing solvers with different geometry representations (e.g.,
   MC square cell vs CP Wigner-Seitz cylinder), verify that the
   cell area/volume is equal.  A factor-of-2 in a linear dimension
   is a factor-of-4 in area — enough to change supercritical to
   subcritical.  ERR-017 survived because all homogeneous tests
   passed and the heterogeneous tests were `@pytest.mark.slow`.

9. **Angular integration weights have hidden factors.** In MOC,
   the weight `omega_a * omega_p * t_s` is the spatial/angular
   discretization weight, but the scalar flux integral also requires
   `4*pi` (full-sphere normalization) and `sin(theta_p)` (2D→3D
   projection).  These factors cancel for spatially uniform solutions,
   making them invisible to homogeneous tests.  ERR-019 survived
   three homogeneous tests at machine precision because delta_psi = 0
   for uniform media.  Derive the weight formula from first principles
   and verify against a heterogeneous problem BEFORE trusting the
   homogeneous result.
