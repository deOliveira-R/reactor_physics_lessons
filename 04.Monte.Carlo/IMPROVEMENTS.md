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

## DONE — Implemented and Documented

### MT-20260403-001 — MCGeometry protocol for delta-tracking

`MCGeometry` protocol: `material_id_at(x, y) -> int` + `pitch`.
Designed for delta-tracking (no distance-to-surface needed).
Extensible to future CSG implementations.

Two concrete implementations:
- `ConcentricPinCell(radii, mat_ids, pitch)` — annular regions
- `SlabPinCell(boundaries, mat_ids, pitch)` — 1D slab regions

Documented in `docs/theory/monte_carlo.rst` §Solver Architecture, Layer 1.

### MT-20260403-002 — Homogeneous verification {1,2,4}G

Analytical derivation from random walk probability theory:
k = νΣf/Σa (1G), k = λ_max(A⁻¹F) (multi-group).
Statistical verification via z-score < 5σ.
1G homogeneous gives deterministic σ=0 (all neutrons see same XS).

Documented in `docs/theory/monte_carlo.rst` §Analytical Verification.

### MT-20260406-001 — MCMesh augmented geometry

`MCMesh` class wrapping `Mesh1D` for MC delta-tracking, following the
same pattern as `CPMesh` and `SNMesh`:

    Mesh1D (from factories) → MCMesh(mesh, pitch) → solve_monte_carlo()

Supports Cartesian (x-lookup) and Cylindrical (radial distance).
Satisfies `MCGeometry` protocol — solver needs zero changes.
Verified: 10000-point agreement with `ConcentricPinCell` (zero mismatches).

Documented in `docs/theory/monte_carlo.rst` §Solver Architecture, Layer 1.

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

Documented in `docs/theory/monte_carlo.rst` §Verification Suite.

### MT-20260406-003 — L1/L2/XV verification suite (9 tests)

Extended eigenvalue and convergence tests:
- L1: 2G/4rg, 4G/2rg, 4G/4rg heterogeneous cases
- L1: 2G high-stats (non-degenerate, flux shape matters)
- L1: Weight ratio consistency (keff estimator internal consistency)
- L2: sigma ~ 1/sqrt(N) convergence rate
- L2: Bias decreases with more histories per cycle
- L2: Inactive cycles reduce source convergence bias
- XV: MC vs CP cylinder (2G 2-region)
- XV: MC vs CP slab (2G 2-region)

Documented in `docs/theory/monte_carlo.rst` §Verification Suite.

### MT-20260406-004 — Pitch formula bug fix (ERR-017)

Fixed pre-existing bug in heterogeneous MC tests: `pitch = r_cell *
sqrt(pi) * 2` → `pitch = r_cell * sqrt(pi)`.  The factor of 2
quadrupled the cell area, adding 4× moderator and causing k_mc to be
24% low (or NaN from population collapse in subcritical systems).
See `tests/l0_error_catalog.md` ERR-017.

Documented in `docs/theory/monte_carlo.rst` §ERR-017 investigation.

### MT-20260406-008 — Restructure solver: Particle/Neutron/Bank architecture

Refactored the monolithic `solve_monte_carlo` (~220 lines) into 5 modular
layers: `Particle` → `Neutron` dataclasses, `NeutronBank` with array
storage, `_PrecomputedXS` cache, extracted functions (`_random_walk`,
`_russian_roulette`, `_split_heavy`), and `solve_monte_carlo` as ~60-line
orchestrator.  Verified: `test_seed_reproducibility` proves identical RNG
sequences.

Documented in `docs/theory/monte_carlo.rst` §Solver Architecture.

---

## OPEN — Not Yet Implemented

### MT-20260403-003 — Sphinx theory chapter remaining gaps

**Priority**: LOW | **Effort**: Small

`docs/theory/monte_carlo.rst` exists (1169 lines, archivist score 8/10).
Remaining gaps:
- Derivation scripts for hand-written proofs (delta-tracking equivalence,
  weight conservation, splitting E[N]=w)
- Autodoc path for MC module (`:class:`/`:func:` refs don't hyperlink)
- Convergence rate table with actual numerical results

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

### MT-20260406-006 — Direction sampling is not isotropic (ERR-018)

**Priority**: LOW | **Effort**: Small

Documented in `tests/l0_error_catalog.md` ERR-018.  The solver uses
`theta = pi * rng.random()` (uniform theta) instead of
`theta = arccos(1 - 2*xi)` (isotropic on sphere).  This matches the
MATLAB original.  For a 2D MC with periodic BCs, the non-isotropic
sampling changes the effective mean free path but does not invalidate
keff.  Fixing this would break MATLAB compatibility and require
re-establishing all reference values.

### MT-20260406-007 — Flux tally is not a proper estimator

**Priority**: MEDIUM | **Effort**: Small

The `detect_s` scattering detector accumulates `w / sig_s_sum` only on
scattering events.  A correct collision estimator would accumulate
`w / sig_t` on ALL real collisions.  Additionally,
`flux_per_lethargy = detect_s / du` produces negative values because the
energy grid is high-to-low (`du = log(E_upper/E_lower) < 0`).

The flux output is used only for visualization, not eigenvalue, so keff
is unaffected.  Fixing requires:
- Accumulate on all real collisions with `w / sig_t`
- Use `|du|` or reverse the energy grid for lethargy
- Add L1 test comparing MC flux spectrum against analytical

### MT-20260406-009 — Event-based vectorized transport

**Priority**: MEDIUM | **Effort**: Large

Replace the per-neutron serial random walk with event-based batch
processing: all neutrons advance one collision at a time using numpy
vectorized operations.  Each step processes the full population:

1. Sample all free paths (`-log(rng.random(n)) / sig_t_max`)
2. Move all positions (vectorized addition + periodic BC)
3. Lookup all materials (vectorized searchsorted)
4. Decide real/virtual for all (vectorized comparison)
5. Partition into scattered/absorbed/virtual sets
6. Process scattering CDF and absorption weight for each set

This is the approach used by GPU MC codes (OpenMC event-based mode).
Expected speedup: 10-50× over per-neutron Python loops.

### MT-20260406-010 — Parallel neutron transport

**Priority**: MEDIUM | **Effort**: Medium

Each neutron's random walk within a cycle is independent, enabling
embarrassingly parallel execution via `multiprocessing` or `concurrent.futures`.
Requires making the RNG state per-worker (e.g., `SeedSequence.spawn`).
Can be combined with event-based vectorization for hybrid parallelism.
