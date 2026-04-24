# Precision-Floor Tool Implementation Plan (T3.4 / G-P2.1)

**Author**: Plan agent (Claude Opus 4.7), 2026-04-24.

## 1. Executive Summary

- **What.** A per-reference quadrature-floor sweep for every `ContinuousReferenceSolution` enumerated by `peierls_cases.continuous_cases()`. Floor = the smallest `(n_panels_per_region, p_order, dps, n_angular, n_rho, n_surf_quad)` tuple for which the row-sum invariant `K · 1 ≈ Σ_t · φ_uniform(r)` holds to user-specified `ε ∈ {1e-4, 1e-6, 1e-8, 1e-10, 1e-12}`.
- **Why.** Today every shipped reference is hand-tuned. Below the (unknown) floor it silently fails as an L1 gate; above it, it wastes ~minutes/build. Recording the floor turns "internally consistent" from folklore into a numbered, regression-gated claim.
- **Where in the build.** Recommendation is **CI-sweep with committed JSON artefact + lightweight Sphinx include-rendering hook**. The Sphinx hook only re-renders the RST table from the committed JSON; it does NOT re-run the sweep.
- **Cost class.** One full sweep of all 13 currently-shipped references at 5 ε-targets, with bisection over a pruned 3-axis grid, is **~30–90 minutes wall-clock** on one core; **~10 minutes** parallelised across 8 cores. Run on PR-touch of `peierls_*.py` + nightly.
- **Expected adoption.** Two consumers land with it: (i) `@pytest.mark.at_precision_floor(reference_name)` regression gate; (ii) a stretch-goal `right_size_reference(name, eps)` builder that constructs references at the recorded floor instead of the over-conservative defaults — this is what makes the entire effort pay back in test-suite wall-clock.

---

## 2. Architecture

### 2.1 Module layout

```
tools/verification/
    generate_peierls_matrix.py            # existing — capability matrix
    generate_peierls_precision_floors.py  # NEW — sweep driver + RST renderer

orpheus/derivations/
    peierls_precision_floor.py            # NEW — pure-library invariant evaluator + sweep search

docs/theory/
    _peierls_precision_floors.inc.rst     # NEW — auto-generated include
    peierls_unified.rst                   # MODIFIED — add §theory-peierls-precision-floors with `.. include::`

docs/_data/                               # NEW directory (or use docs/theory/_data/)
    peierls_precision_floors.json         # NEW — committed sweep artefact (canonical source)

tests/derivations/
    test_peierls_precision_floor.py       # NEW — invariant-evaluator unit tests + at-floor regression marker
    test_peierls_precision_floor_artefact.py  # NEW — schema/staleness check on the committed JSON
```

Rationale for the `peierls_precision_floor.py` library/`generate_*` tool split: mirrors the existing `peierls_cases.capability_rows()` (library) ↔ `generate_peierls_matrix.py` (renderer) split. Keeps the invariant evaluator unit-testable without subprocess or RST plumbing, and lets the `@pytest.mark.at_precision_floor` marker import the evaluator directly.

### 2.2 Function signatures

```python
# orpheus/derivations/peierls_precision_floor.py

@dataclass(frozen=True)
class QuadratureSettings:
    n_panels_per_region: int
    p_order: int
    dps: int
    n_angular: int            # for curvilinear; ignored on slab-polar (adaptive)
    n_rho: int
    n_surf_quad: int          # only consumed by BC-correction assembly
    # Hash key for caching; ordered so monotone increases are well-defined.

@dataclass(frozen=True)
class FloorRecord:
    reference_name: str
    target_epsilon: float
    settings: QuadratureSettings
    measured_residual: float          # ||K·1 − Σ_t·φ_uniform||_∞ (relative)
    invariant: Literal["vacuum", "white_f4_total"]
    wall_time_s: float
    plateau_detected: bool            # True if sweep terminated on plateau, not on ε
    n_groups_evaluated: int           # 1 for scalar invariant; ng for per-group
    plateau_floor: float | None       # min residual reached if plateau_detected

def evaluate_invariant(
    ref: ContinuousReferenceSolution,
    settings: QuadratureSettings,
) -> tuple[float, dict]:
    """Build K at `settings`, evaluate the row-sum residual against
    `φ_uniform(r)` from `peierls_reference`. Returns (residual, diagnostics)."""

def sweep_floor(
    ref: ContinuousReferenceSolution,
    epsilon: float,
    *,
    base: QuadratureSettings = DEFAULT_BASE,
    max: QuadratureSettings = DEFAULT_MAX,
    parallel: bool = False,
) -> FloorRecord: ...

def sweep_all(
    references: Iterable[ContinuousReferenceSolution],
    epsilons: Sequence[float] = (1e-4, 1e-6, 1e-8, 1e-10, 1e-12),
    *,
    parallel_workers: int = 1,
) -> list[FloorRecord]: ...
```

```python
# tools/verification/generate_peierls_precision_floors.py

DEFAULT_JSON_OUT = REPO_ROOT / "docs" / "_data" / "peierls_precision_floors.json"
DEFAULT_RST_OUT  = REPO_ROOT / "docs" / "theory" / "_peierls_precision_floors.inc.rst"

def main(argv): ...
    # Modes:
    #   --sweep          run full sweep, write JSON
    #   --render-only    read committed JSON, regenerate RST (Sphinx-hook mode)
    #   --reference NAME --epsilon E   run a single (ref, ε) cell
    #   --workers N      parallel-across-references multiprocessing
```

### 2.3 Output schemas

**JSON artefact** — array of `FloorRecord` dicts, sorted lexically by `(reference_name, target_epsilon)`. Includes `schema_version: 1` at the top level so the test in §6 can detect stale-format artefacts.

**RST include file** — one `.. list-table::` per reference, columns: `target ε`, `n_panels`, `p_order`, `dps`, `n_angular`, `n_rho`, `n_surf_quad`, `residual`, `wall-time (s)`, `plateau?`. Renderer reuses `_list_table` from `generate_peierls_matrix.py` (extract to a shared helper in a third file if the duplication grows beyond two consumers).

---

## 3. Algorithm

### 3.1 Per-reference invariant dispatch

Map driven off `ref.problem.geometry_type` and `ref.problem.boundary_conditions`:

| Geometry | BC | Invariant | RHS source |
|---|---|---|---|
| `slab` | `vacuum` | `K · 1 ≈ Σ_t · φ_slab,vac(x_i)` | `slab_uniform_source_analytical` |
| `slab` | `white` (Mark) | `K_total · 1 ≈ Σ_t · (1/Σ_t) = 1` (pointwise) | `slab_uniform_source_white_bc_analytical` |
| `cylinder-1d` | `vacuum` | `K · 1 ≈ Σ_t · φ_cyl,vac(r_i)` | `cylinder_uniform_source_analytical` |
| `sphere-1d` | `vacuum` | `K · 1 ≈ Σ_t · φ_sph,vac(r_i)` | `sphere_uniform_source_analytical` |
| `cylinder-1d` / `sphere-1d` | F.4 white | `K_total · 1 ≈ Σ_t · φ_∞ = 1` | infinite-medium constant (no analytical curvilinear-white closed form yet) |

For F.4 closure references, evaluate `K_total = K_vol + K_bc` (the white-BC corrected matrix; see `peierls_geometry` lines ~4108–4125, `_K_total_one_group` builds exactly this); the white-BC invariant for a uniform pure-absorber cell collapses to `1` (Wigner-Seitz). For multi-region references the invariant is piecewise per region: assemble the global K, but evaluate `φ_uniform` row-by-row using the local `Σ_t(r_i)` of the region containing `r_i` — `slab_uniform_source_analytical` already accepts a uniform `sig_t`; multi-region requires a small extension that integrates the analytical RHS across region boundaries (see G-P1.1 dependency in §8).

The dispatch is keyed off `ref.name` regex (`peierls_slab_*`, `peierls_cyl1D_hollow_*`, `peierls_sph1D_hollow_*`). A registry table — not string parsing — should live next to `peierls_cases.capability_rows()`; new references registered there auto-pick up the floor sweep.

### 3.2 Multi-group handling

For a `ng > 1` reference, run the row-sum invariant **per group**: build `K_g` (the per-group K already exists — see `_K_total_one_group` group-local loop in `peierls_geometry.py:4292–4298`), evaluate `||K_g · 1 - Σ_{t,g} · φ_uniform,g||_∞` per group, take the **max over groups** as the residual the sweep targets. Do not aggregate by averaging — a group whose residual is 100× the others is a localised quadrature failure that aggregation would mask. The `FloorRecord.n_groups_evaluated` field records `ng`.

### 3.3 Sweep strategy — bisection on three axes

Linear sweep is wasteful. The 6-tuple decomposes into three correlated groups:

1. **Spatial** — `(n_panels_per_region, p_order)`. Quasi-monotone: increasing either reduces residual.
2. **Precision** — `dps`. Floor-of-floor below ~15; above ~30 hits a plateau set by the next bottleneck.
3. **Angular/ray** — `(n_angular, n_rho, n_surf_quad)`. Adaptive-mpmath path on slab makes these moot; for curvilinear non-adaptive paths these dominate.

**Bisection schedule (per ε):**

1. Start at `BASE = (n_panels=2, p=3, dps=15, n_ang=16, n_rho=16, n_surf=16)`. Evaluate.
2. If residual < ε: try `BASE/2` on each axis independently; take the smallest setting per axis that still satisfies ε.
3. If residual > ε: identify the limiting axis by single-axis upward steps (double `n_panels`; `p+=1`; `dps+=10`; `n_ang*=2`). Pick the axis with greatest residual reduction per unit cost (cost model: K-element wall-time ≈ `O(N² · n_ang · n_rho · ng)` on non-adaptive paths, `O(N² · ng)` adaptive-mpmath).
4. Iterate until either ε is met or the **plateau detector** fires.

**Plateau detector.** After 3 successive single-axis doublings reduce the residual by < 1.2× cumulatively, declare a structural floor. This indicates the residual is bounded by something OTHER than quadrature — bug in the K assembly, mismatch between invariant RHS and the actual operator, or an under-resolved physical regime (e.g. very small `r_0/R`). Record the plateau value and emit a clear warning in the RST output ("structural floor; quadrature cannot reach ε"). The ε-targets above the plateau still get rows, marked as `plateau_detected=True` with the actually-achievable residual.

### 3.4 Caching

`evaluate_invariant` memoises K by `(ref.name, settings_hash)` to a process-local dict. The bisection visits ~5–8 settings per (ref, ε) pair; without caching it would rebuild K up to 5× per ε-target across the ε-ladder. With caching, the cost across all 5 ε-targets for one reference ≈ cost of the largest setting visited (because smaller settings get cached during the descent of the smallest ε). **This is the biggest single cost-optimisation win.**

---

## 4. Cost Analysis

Inputs from `docs/theory/peierls_unified.rst:635, :838, :1191`:
- ~100 ms per K element on adaptive-mpmath at default dps;
- a representative full-reference build is ~1–2 minutes wall-time at default quadrature.

**Per-reference cost.** With caching, a full ε-ladder (5 targets) for one reference visits roughly the same K assemblies as a single build at the highest setting plus ~3–5 below-floor probe builds. Estimate **3–8× a single build = 3–15 minutes per reference**.

**Full sweep (13 references × 5 ε-targets, single core):** 13 × ~7 min ≈ **90 minutes**.

**Parallelised (8 workers, references are embarrassingly parallel — no shared state in the K assembly path):** ~13 minutes.

**Sphinx-hook cost (recommended: render-only mode):** `~50 ms` to read JSON and render RST. Negligible.

**Sphinx-hook cost (rejected: full sweep on every doc build):** 90 min. Doc builds become unusable; cache misses every CI run. This is why §5 recommends committed-JSON.

---

## 5. Integration: where does the sweep run?

| Option | Latency seen by developers | Maintenance | Risk of staleness |
|---|---|---|---|
| Sphinx hook (full sweep) | +90 min per `make html` | Zero (auto) | Zero |
| Sphinx hook (render-only from JSON) | +50 ms | JSON must be committed when refs change | Caught by §6 staleness test |
| CI sweep (PR-trigger on `peierls_*.py`) | 0 (async) | Bot or developer commits new JSON | Caught by §6 staleness test |
| Developer on-demand (`python -m tools.verification.generate_peierls_precision_floors --sweep`) | 90 min when invoked | Manual discipline | High without §6 staleness test |

**Recommendation: render-only Sphinx hook + CI-sweep + on-demand fallback.**

- The Sphinx hook (`docs/conf.py`) calls `generate_peierls_precision_floors.py --render-only`; it reads the committed JSON and writes the RST. Mirrors the existing `_regenerate_peierls_matrix` hook exactly.
- A CI workflow (`.github/workflows/peierls_precision_floor.yml`) runs the full sweep on PRs that touch `orpheus/derivations/peierls_*.py` or `tools/verification/generate_peierls_precision_floors.py`, parallelised across 8 workers. If the sweep produces a JSON different from `HEAD`'s, CI fails with a clear "commit the updated `peierls_precision_floors.json`" message.
- A scheduled nightly CI run does the same (catches drift from indirect dependencies — e.g. `_kernels.py` changes that don't touch the registered refs but shift K element values).
- The `@pytest.mark.at_precision_floor` marker (§6) reads the committed JSON in-process; tests don't trigger the sweep.

This matches the existing pattern used for the verification matrix and capability matrix: the data is the single source of truth and lives under version control; the doc build re-renders, never re-derives.

---

## 6. Test Plan

### 6.1 Invariant-evaluator unit tests (`test_peierls_precision_floor.py`)

- For each invariant type (slab vacuum / cyl vacuum / sph vacuum / slab white-Mark / curvilinear F.4-white), assert `evaluate_invariant(ref, very_high_settings)` returns residual `< 1e-12`. This is the "the invariant evaluator agrees with the existing `TestCylinderKernelRowSum` / `TestSphereKernelRowSum`" cross-check.
- For each invariant type, assert `evaluate_invariant(ref, intentionally_low_settings)` returns residual `> 1e-2`. Catches bogus "always-passes" implementations.
- Test caching: two calls at identical settings reuse K (introspect cache hit).
- Test plateau detector with a synthetic ref whose K assembly doesn't depend on quadrature (e.g. trivial constant matrix).

### 6.2 Floor regression marker (`@pytest.mark.at_precision_floor`)

```python
@pytest.mark.at_precision_floor("peierls_cyl1D_hollow_1eg_1rg_r0_10", epsilon=1e-8)
def test_my_solver_at_floor(...):
    ref = build_at_floor("peierls_cyl1D_hollow_1eg_1rg_r0_10", epsilon=1e-8)
    # ref is now the right-sized reference (§7).
```

The marker:
- Reads the committed JSON to find the floor settings for `(ref_name, ε)`;
- Builds the reference at exactly those settings (via the §7 right-sizer);
- A conftest fixture asserts the reference's invariant residual at-floor is within 2× of `ε` (the "floor regression" gate — if a refactor pushes the floor up, the marker fires before the test sees it).

### 6.3 Artefact staleness test (`test_peierls_precision_floor_artefact.py`)

Lightweight check: every name in `capability_rows()` appears in the JSON for every ε in the canonical ladder; no orphan rows in JSON; `schema_version` matches code. Runs in <100 ms; lives in normal CI. Does NOT re-run the sweep — that's the dedicated CI workflow's job.

---

## 7. Stretch goal — right-sized reference building

```python
# orpheus/derivations/peierls_precision_floor.py
def build_at_floor(
    reference_name: str,
    epsilon: float,
) -> ContinuousReferenceSolution:
    """Build the named reference at exactly the recorded floor for ε.
    Reads `docs/_data/peierls_precision_floors.json`; passes the
    `QuadratureSettings` through to the corresponding `_build_*_case` /
    `_build_peierls_*_hollow_f4_case` constructor."""
```

Replaces hand-picked quadrature in `_class_a_cases()` with `build_at_floor(name, ε=ref.required_epsilon)`. Each registered reference declares its required ε (a new `ContinuousReferenceSolution.required_epsilon` field defaulting to `1e-8`). At registration time the reference is built at the floor, not at the conservative default.

**Wall-time saving estimate.** Current conservative defaults (e.g. `n_panels_per_region=16, p_order=6, dps=30` for the unified-path slab) are typically ~3× over-resolved relative to the 1e-8 floor. A 3× reduction in `O(N²)` K assembly is **9× wall-time** per reference build at registration. For the 13 shipped references × ~1.5 min each = 19 min today → ~2 min after right-sizing. Compounds across every test session that imports `reference_values`.

---

## 8. Sequencing

| Dependency | Status | Effect on this tool |
|---|---|---|
| **G-P1.1** multi-region curvilinear refs | Not landed | Adds rows to the registry. The invariant evaluator needs a piecewise φ_uniform helper (`evaluate_piecewise_uniform_source(r_nodes, region_radii, sig_t_per_region)`) that assembles the per-region analytical RHS. **Can be designed now**, signature-locked, body added when G-P1.1 lands. |
| **G-P1.4** 4G refs | Not landed | Adds new `ng_key="4g"` rows. The per-group invariant of §3.2 is already group-agnostic, so the only change is adding `"4g"` to the ε-ladder sweep. **Zero blocker.** |
| **Issue #103** rank-N DP_N solid-cyl/sph | Not landed | Brings Class B references into the registry. The invariant for solid-cyl/sph at rank-N closure is the same row-sum identity. **Zero blocker.** |
| **Issue #101** chord-based Ki₁ analytical | Not landed | Provides a tighter analytical RHS for solid cyl. Optional improvement; current implementation works without it. |

**This tool can ship independently of all four**, against the 13 currently-shipped refs in `peierls_cases._class_a_cases()`. Each landed dependency adds rows to the JSON and corresponding RST table entries; no schema changes.

---

## 9. Estimated Budget

| Bucket | Estimate |
|---|---|
| **Sessions** | 3–4 working sessions: (1) library + invariant evaluators + unit tests; (2) sweep driver + caching + plateau detector + JSON artefact; (3) RST renderer + Sphinx hook + CI workflow + staleness test; (4) pytest marker + right-sizer + retrofit one reference as a smoke test. |
| **LoC** | ~600 production + ~400 tests. Renderer ~150 (mostly extracted from `generate_peierls_matrix.py`). |
| **Files touched (new)** | 5: `peierls_precision_floor.py`, `generate_peierls_precision_floors.py`, `_peierls_precision_floors.inc.rst` (auto-gen), `peierls_precision_floors.json` (auto-gen), 2 tests. |
| **Files touched (modified)** | 3: `docs/conf.py` (add hook), `docs/theory/peierls_unified.rst` (add include + section), `peierls_cases.py` (only if §7 stretch lands — add `required_epsilon` field). |
| **CI workflow** | 1 new `.github/workflows/peierls_precision_floor.yml`, ~50 lines. |

---

## Critical Files for Implementation

- `tools/verification/generate_peierls_matrix.py` — structural mirror for the new generator (including the `_list_table` helper and `main(argv)` shape).
- `orpheus/derivations/peierls_reference.py` — analytical `φ_uniform` for slab / cylinder / sphere; the row-sum invariant RHS source.
- `orpheus/derivations/peierls_cases.py` — registry (`continuous_cases()`, `capability_rows()`); the new tool iterates the same registry, and §7 modifies `_class_a_cases()` to use `build_at_floor`.
- `orpheus/derivations/peierls_geometry.py` — `build_volume_kernel` / `build_volume_kernel_adaptive` / `_K_total_one_group`; the K assembly the sweep parametrises and the per-group K loop the multi-group invariant calls into.
- `docs/conf.py` — the existing `_regenerate_peierls_matrix` hook the new render-only hook copies.
