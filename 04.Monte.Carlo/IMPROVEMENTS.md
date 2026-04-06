# Monte Carlo — Improvement Tracker

Central registry of ALL bugs, improvements, and features for the
Monte Carlo solver.

## Tracking Number Format

`MT-YYYYMMDD-NNN` where MT = Monte Carlo Transport.

## Status Legend

- **DONE**: implemented AND documented in Sphinx
- **IMPL**: implemented and tested, Sphinx documentation pending
- **OPEN**: not yet implemented, documented here with full context

---

## IMPL — Implemented, Sphinx Documentation Pending

### MT-20260403-001 — MCGeometry protocol for delta-tracking

`MCGeometry` protocol: `material_id_at(x, y) -> int` + `pitch`.
Designed for delta-tracking (no distance-to-surface needed).
Extensible to future CSG implementations.

Two concrete implementations:
- `ConcentricPinCell(radii, mat_ids, pitch)` — annular regions
- `SlabPinCell(boundaries, mat_ids, pitch)` — 1D slab regions

### MT-20260403-002 — Homogeneous verification {1,2,4}G

Analytical derivation from random walk probability theory:
k = νΣf/Σa (1G), k = λ_max(A⁻¹F) (multi-group).
Statistical verification via z-score < 5σ.
1G homogeneous gives deterministic σ=0 (all neutrons see same XS).

### MT-20260406-001 — MCMesh augmented geometry

`MCMesh` class wrapping `Mesh1D` for MC delta-tracking, following the
same pattern as `CPMesh` and `SNMesh`:

    Mesh1D (from factories) → MCMesh(mesh, pitch) → solve_monte_carlo()

Supports Cartesian (x-lookup) and Cylindrical (radial distance).
Satisfies `MCGeometry` protocol — solver needs zero changes.
Verified: 10000-point agreement with `ConcentricPinCell` (zero mismatches).

### MT-20260406-002 — Full L0 verification suite (14 tests)

Term-level tests isolating each algorithmic component:
- MCMesh geometry lookup (Cartesian + Cylindrical)
- Majorant computation (max over materials per group)
- Delta-tracking virtual collision probability
- Scattering CDF sampling (catches SigS^T bug, ERR-002 equivalent)
- Fission weight adjustment (w *= SigP/sig_a)
- Chi spectrum sampling (fission group distribution)
- Periodic BC wrapping (x % pitch edge cases)
- Russian roulette weight conservation (statistical)
- Splitting weight conservation (exact)
- Scattering branching ratio (sig_s/sig_t)
- Direction sampling (E[dir_x^2] = 1/4 for uniform theta)
- Batch statistics formula (hand-calculated mean + sigma)
- Scattering convention anti-ERR-002 (asymmetric 2G, no upscatter)

### MT-20260406-003 — L1/L2/XV verification suite (9 tests)

Extended eigenvalue and convergence tests:
- L1: 2G/4rg and 4G/2rg heterogeneous cases (multi-group + multi-region)
- L1: 2G high-stats (non-degenerate, flux shape matters)
- L1: Weight ratio consistency (keff estimator internal consistency)
- L2: sigma ~ 1/sqrt(N) convergence rate
- L2: Bias decreases with more histories per cycle
- L2: Inactive cycles reduce source convergence bias
- XV: MC vs CP cylinder (2G 2-region)
- XV: MC vs CP slab (2G 2-region)

### MT-20260406-004 — Pitch formula bug fix (ERR-017)

Fixed pre-existing bug in heterogeneous MC tests: `pitch = r_cell *
sqrt(pi) * 2` → `pitch = r_cell * sqrt(pi)`.  The factor of 2
quadrupled the cell area, adding 4× moderator and causing k_mc to be
24% low (or NaN from population collapse in subcritical systems).
See `tests/l0_error_catalog.md` ERR-017.

---

## OPEN — Not Yet Implemented

### MT-20260403-003 — Sphinx theory chapter

**Priority**: HIGH | **Effort**: Large

No `docs/theory/monte_carlo.rst` exists.  Needs to document:
- Woodcock delta-tracking algorithm and derivation
- Analog absorption with fission weight adjustment
- Russian roulette and splitting (weight conservation proofs)
- Fission spectrum sampling from chi CDF
- keff estimator and statistical uncertainty (CLT)
- MCGeometry protocol design (CSG-extensible)
- MCMesh augmented geometry (base → augmented → solver pattern)
- Direction sampling (uniform theta simplification, not isotropic)
- Verification results (40 tests across L0/L1/L2/XV)
- Square-vs-circle geometry mismatch tolerance analysis

### MT-20260403-004 — Python neutron loop performance

**Priority**: HIGH | **Effort**: Large

The inner neutron random walk is a Python `while True` loop with
per-collision Python-level operations.  For 421-group problems with
many collisions per neutron, this dominates runtime.  Options:
- Batch neutrons: process all neutrons for one free-path step together
- Vectorize collision sampling across the neutron population
- Consider Cython/numba for the inner loop

### MT-20260403-005 — Heterogeneous independent reference

MC heterogeneous verification uses CP cylinder eigenvalue as proxy.
Should use high-statistics MC run (10⁵ active cycles) as independent
reference.  See DV-20260403-005.

### MT-20260403-006 — Majorant cross section per group

Current majorant is `max(SigT)` across all materials for each group.
For problems with highly variable cross sections, a more efficient
majorant (e.g., piecewise by region) would reduce virtual collision
rate.

### MT-20260406-005 — Solver ignores Sig2 (n,2n) reactions

**Priority**: MEDIUM | **Effort**: Small

The MC solver does not use `mat.Sig2` anywhere.  The scattering kernel
uses `SigS[0]` only — it does not include the `2*Sig2` contribution
that the CP and SN solvers include.  Invisible since all test materials
have Sig2 = 0.  Adding (n,2n) support requires:
- Including `2 * Sig2` in the scattering CDF
- Adjusting the total cross section: sig_t = sig_a + sig_s + 2*sig_2
- Adding at least one test with nonzero Sig2 (meta-lesson 6)

### MT-20260406-007 — Flux tally is not a proper estimator

**Priority**: MEDIUM | **Effort**: Small

The `detect_s` scattering detector accumulates `w / sig_s_sum` only on
scattering events (line 307).  A correct collision estimator would
accumulate `w / sig_t` on ALL real collisions.  Additionally,
`flux_per_lethargy = detect_s / du` produces negative values because the
energy grid is high-to-low (`du = log(E_upper/E_lower) < 0`).

The flux output is used only for visualization, not eigenvalue, so keff
is unaffected.  Fixing requires:
- Accumulate on all real collisions with `w / sig_t`
- Use `|du|` or reverse the energy grid for lethargy
- Add L1 test comparing MC flux spectrum against analytical

### MT-20260406-006 — Direction sampling is not isotropic (ERR-018)

**Priority**: LOW | **Effort**: Small

Documented in `tests/l0_error_catalog.md` ERR-018.  The solver uses
`theta = pi * rng.random()` (uniform theta) instead of
`theta = arccos(1 - 2*xi)` (isotropic on sphere).  This matches the
MATLAB original.  For a 2D MC with periodic BCs, the non-isotropic
sampling changes the effective mean free path but does not invalidate
keff.  Fixing this would break MATLAB compatibility and require
re-establishing all reference values.
