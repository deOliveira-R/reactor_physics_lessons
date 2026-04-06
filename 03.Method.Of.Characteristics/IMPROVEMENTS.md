# Method of Characteristics — Improvement Tracker

Central registry of ALL bugs, improvements, and features for the
MOC solver.

## Tracking Number Format

`MC-YYYYMMDD-NNN` where MC = Method of Characteristics.
(Note: Monte Carlo uses `MT` to avoid collision.)

## Status Legend

- **DONE**: implemented AND documented in Sphinx
- **IMPL**: implemented and tested, Sphinx documentation pending
- **OPEN**: not yet implemented, documented here with full context

---

## IMPL — Implemented, Sphinx Documentation Pending

### MC-20260406-001 — Proper 2D MOC solver with exact ray tracing

Replaced the pedagogical 8-direction Cartesian solver (MATLAB port) with
a production-quality 2D MOC:
- Exact ray-circle intersection through concentric annuli
- Tabuchi-Yamamoto polar quadrature (TY-1/2/3) × uniform azimuthal
- Configurable ray spacing, angular resolution
- Inverse Wigner-Seitz: `Mesh1D`(cylindrical) → `MOCMesh` → `MOCSolver`
- Satisfies `EigenvalueSolver` protocol with `power_iteration()`
- Reflective BC via track linking (vertical → same direction, horizontal → reversed)
- Boyd et al. (2014) scalar flux update: `phi = (4pi*Q + delta_phi/A) / sig_t`

Files: `moc_quadrature.py`, `moc_geometry.py`, `moc_solver.py`,
`method_of_characteristics.py` (rewritten).

### MC-20260406-002 — Inverse Wigner-Seitz geometry convention

`MOCMesh` wraps a cylindrical `Mesh1D` (from `pwr_pin_equivalent()`).
The pitch is recovered as `mesh.edges[-1] * sqrt(pi)`.  The outermost
annular region is reinterpreted as the square border.  Material IDs
flow directly from `mesh.mat_ids` — no extra parameters needed.

### MC-20260406-003 — 102-test verification suite

- 24 quadrature tests (weight sums, TY values, shapes)
- 20 ray tracing tests (intersection, region ID, segments, volume, links)
- 48 verification tests (L0 term isolation, L1 eigenvalue, L2 convergence, XV cross-verification)
- 6 + 4 eigenvalue + property tests from original suite (updated for new API)

### MC-20260406-004 — (n,2n) support in transport and eigenvalue

The source term includes `2 * Sig2^T @ phi`, and `compute_keff` uses
`production = (SigP + 2*Sig2_out) @ phi`.  Verified with analytical
eigenvalue for 1G material with nonzero Sig2.

### MC-20260406-005 — ERR-019 weight factor fix (4pi * sin_theta) — DONE

Bug: initial implementation used `omega_a * omega_p * t_s` as the
delta_phi accumulation weight.  Missing `4*pi * sin(theta_p)`.
Invisible to homogeneous tests (delta_psi = 0).  Caught by heterogeneous
cross-verification with CP solver.  See `tests/l0_error_catalog.md` ERR-019.
Sphinx documentation: `docs/theory/method_of_characteristics.rst` §ERR-019.

---

## OPEN — Not Yet Implemented

### MC-20260406-006 — Sphinx theory chapter — DONE

`docs/theory/method_of_characteristics.rst` (1277 lines, 76 directives).
Full derivations: characteristic ODE, flat-source solution, bar-psi from
ODE integral, Boyd Eq. 45 weight formula with sin(theta_p) derivation,
keff with (n,2n).  Investigation history (ERR-019).  Convergence tables
(ray spacing, azimuthal, polar) with real numerical data.  Design
decisions (flat-source vs linear-source, TY vs GL, inverse Wigner-Seitz).
API autodoc page at `docs/api/method_of_characteristics.rst`.  SymPy
verification in `derivations/moc_equations.py`.

### MC-20260406-007 — Pure Python loop bottleneck in transport sweep

**Priority**: HIGH | **Effort**: Medium

The transport sweep has a 5-deep Python loop:
`for a_idx / for t_idx / for p_idx / for seg / for g`.
For fine ray spacing + many groups, this dominates runtime.  Options:
- Vectorize the group loop (inner-most): segment attenuation is
  independent across groups, so `tau`, `exp(-tau)`, `delta_psi` can
  be computed as `(ng,)` arrays per segment
- Pre-flatten segment data into numpy arrays for batch processing
- Numba JIT for the sweep kernel
- Profile first to confirm bottleneck location

### MC-20260406-008 — Tighter heterogeneous tolerances

**Priority**: MEDIUM | **Effort**: Small

The `test_moc_heterogeneous` tests use 5e-2 tolerance.  With the new
solver at `ray_spacing=0.03, n_azi=16, n_polar=3`, the actual accuracy
is ~1e-3 vs CP.  Regenerate Richardson references using the new solver
(spacing refinement instead of cell refinement) and tighten to 1e-3.

### MC-20260406-009 — Cyclic track linking for faster BC convergence

**Priority**: LOW | **Effort**: Medium

Current approach: boundary fluxes persist between outer iterations,
converging gradually.  A cyclic track linking scheme (tracks form closed
loops under reflection) would allow exact BC propagation within a single
sweep.  Requires careful track spacing to ensure cycles close exactly.

### MC-20260406-010 — CMFD acceleration

**Priority**: LOW | **Effort**: Large

Coarse-Mesh Finite Difference acceleration would reduce outer iterations
from ~10-20 to ~5.  Requires:
- Overlaid coarse mesh (can be 1×1 for a single pin cell)
- Net current tallying during MOC sweep
- Modified diffusion solve with nonlinear correction factor
- Flux prolongation (ratio update)

### MC-20260406-011 — Degenerate ray guard in _ray_box_intersections

**Priority**: LOW | **Effort**: Small

If a ray is exactly tangent to a cell corner, `_ray_box_intersections`
may produce fewer than 2 unique intersection parameters, causing an
IndexError.  The half-step offset in ray placement makes this extremely
unlikely, but a guard clause would make it robust.  See QA finding PE-8.

### MC-20260406-012 — demo_moc.py and plotting.py update

**Priority**: MEDIUM | **Effort**: Small

The demo script and plotting module still reference the old `MoCGeometry`
class.  Need to update for the new `solve_moc()` API with `Mesh1D` input
and `MoCResult` with `moc_mesh` field.
