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
   step that produced `f(x)`, the residual is identically zero.
   ERR-016 survived because "all inner iterations = 1" was mistaken
   for fast convergence.  Always verify with a problem that SHOULD
   require multiple iterations.
