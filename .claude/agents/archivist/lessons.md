# Archivist Lessons Learned

This file is the archivist agent's memory. Each entry records what
worked, what was missing, and what to do differently next time.

**Read this file at the START of every invocation.**

---

## 2026-04-04 — Directive 2+3: Documentation Gap Audit on `discrete_ordinates.rst`

### Gap Report

**File audited**: `docs/theory/discrete_ordinates.rst` (1678 lines)
**Code files**: `sn_sweep.py`, `sn_geometry.py`, `sn_solver.py`, `sn_operator.py`, `sn_quadrature.py`
**Derivation scripts**: `derivations/sn.py`, `derivations/sn_heterogeneous.py`, `derivations/sn_contamination.py`
**Improvement tracker**: `02.Discrete.Ordinates/IMPROVEMENTS.md`

#### A. Sections in the RST That Are Stale, Incomplete, or Missing Derivation Backing

1. **Stale module docstring in `sn_geometry.py` (line 9)**: Says "Currently only Cartesian coordinates are implemented." Spherical and cylindrical are fully implemented (lines 119-303). This docstring is stale and misleading. The RST itself is NOT stale -- it correctly documents all three geometries -- but the code docstring contradicts the docs.

2. **`sn_geometry.py` class docstring (line 33-34)**: Says "For future curvilinear geometries, additional curvature terms ... will be stored." This is stale; curvilinear IS implemented.

3. **RST "Heterogeneous Convergence" section (lines 1355-1385)**: Documents cylindrical convergence results but provides no multi-group (2G/4G) heterogeneous convergence tables for cylindrical or spherical. Only 1G cylindrical and 2G angular-convergence data are shown. Spherical heterogeneous convergence is completely absent.

4. **RST "Spatial and Angular Convergence" section (line 1412-1417)**: Very sparse (5 lines). Claims O(h^2) spatial and spectral angular convergence are "verified in `test_sn_1d.py`" but provides no convergence tables, rates, or numerical evidence in the RST itself. This violates the Archivist standard of "INCREDIBLY context-rich documentation."

5. **RST BiCGSTAB section (lines 998-1036)**: Discusses consistency between sweep and BiCGSTAB but does not document: (a) the explicit equation map filtering logic (which ordinates become unknowns), (b) the finite-difference gradient scheme with reflective BCs, (c) the specific approximation used for face fluxes in the curvilinear BiCGSTAB operator (arithmetic averaging of cell-centre values, lines 451/459 in `sn_operator.py`). These are non-trivial implementation details that differ from the DD sweep and affect eigenvalue consistency.

6. **RST References section**: [Bailey2009] citation appears to be for a PIFE paper on polyhedral grids, but it is actually used here for the curvilinear SN discretization. The full correct title should be verified -- the cited equations (50, 53-54, 74) suggest a different Bailey/Morel/Chang paper specifically on curvilinear SN. This may be a bibliographic error.

#### B. Topics in Code That Have NO Documentation in Sphinx

1. **`EquationMap` dataclass and equation map construction** (`sn_operator.py` lines 31-78): The logic for filtering which (ordinate, cell) pairs are unknowns (z-hemisphere filtering, boundary-incoming filtering) is completely undocumented in the RST. This is critical for understanding the BiCGSTAB path.

2. **`_compute_gradients` function** (`sn_operator.py` lines 149-192): The diamond-scheme finite-difference gradient with reflective BCs is not documented anywhere in the RST. The gradient formula differs from the DD sweep and this difference is the source of the coarse-mesh discrepancy mentioned in the RST.

3. **`_build_level_symmetric` construction** (`sn_quadrature.py` lines 178-270): The Level-Symmetric S_N construction algorithm (equal spacing in mu^2, Carlson-Lathrop formula, octant reflection) is not documented in the RST. The RST mentions the quadrature but does not derive or explain the construction.

4. **`ProductQuadrature.create` sorting convention** (`sn_quadrature.py` lines 403-409): The eta-sorting within each level for the cylindrical azimuthal sweep is mentioned briefly in the RST but the rationale (matching Bailey Eq. 50 recursion convention) is only in a code comment. Should be promoted to the RST.

5. **`_build_spherical_harmonics` function** (`sn_quadrature.py` lines 19-45): The Y_l^m convention (Y_0^0=1, Y_1^{-1}=mu_z, Y_1^0=mu_x, Y_1^1=mu_y) is documented in the RST (line 1172) but the derivation of WHY this convention was chosen (matching MATLAB) is only in a code comment. The RST mentions it but does not explain the choice or its consequences.

6. **`build_rhs` per-ordinate source construction for BiCGSTAB** (`sn_operator.py` lines 254-348): The detailed per-ordinate RHS assembly (including the Pn moment reconstruction within the operator) is not documented in the RST.

7. **Pure azimuthal ordinate (eta=0) branch** (`sn_sweep.py` lines 407-420): The cylindrical sweep has a special branch for `eta ~ 0` ordinates (no radial streaming, pure redistribution). This is not mentioned in the RST.

8. **`SNSolver.converged` method**: Convergence criteria (minimum 2 iterations, dual keff + flux tolerance) are not documented.

#### C. Derivations That Exist in Code but NOT in `derivations/` Scripts

1. **Cumprod recurrence derivation**: The RST (lines 830-883) derives the cumulative product solution to the DD recurrence, but there is NO `derivations/` script verifying this. The derivation is hand-written in RST, violating the "derivations come from code" principle. **Need**: `derivations/sn_cumprod.py` (or extend `derivations/sn.py`) with a SymPy verification that the cumprod formula correctly solves the recurrence.

2. **Flat-flux consistency proof**: The RST (lines 587-625) proves that the Delta_A/w factor ensures per-ordinate flat-flux consistency. This proof is hand-written in RST with no derivation script. **Need**: A SymPy or numerical verification in `derivations/sn_contamination.py` (or new script) that demonstrates the cancellation algebraically.

3. **Balance equation derivation** (lines 444-517): The step-by-step derivation from PDE to discrete balance equation is hand-written in RST. No derivation script produces this. This is the most important derivation in the entire chapter.

4. **WDD substitution into balance equation** (lines 740-773): The derivation of the c_out, c_in coefficients and final solve formula is hand-written. No derivation script verifies these algebraic manipulations.

5. **Semi-analytical heterogeneous eigenvalue**: `derivations/sn_heterogeneous.py` exists and provides transfer-matrix-based semi-analytical eigenvalues, but the RST does NOT reference this script or its results. The RST only shows Richardson-extrapolated references (from `derivations/sn.py`).

#### D. Cross-References That Are Missing or Broken

1. **`:class:`SNSolver`** (line 38): References `SNSolver` without module qualification. Sphinx will only resolve this if `sn_solver` is in the autodoc path. Need to verify whether `.. automodule:: sn_solver` is configured.

2. **`:func:`_find_reflections`** (line 288): Private function reference -- may not be visible in autodoc if private members are excluded.

3. **`:func:`_sweep_1d_cumprod`**, **`:func:`_sweep_1d_spherical`**, **`:func:`_sweep_1d_cylindrical`**, **`:func:`_sweep_2d_wavefront`** (lines 875, 773, 946, 981): All private functions -- may not resolve in Sphinx unless `automodule` includes private members.

4. **`:func:`transport_operator_matvec_spherical`**, **`:func:`transport_operator_matvec_cylindrical`** (lines 1538-1539): Not qualified with module path.

5. **`:mod:`derivations.sn_contamination`** (line 667): Module reference -- verify this resolves correctly.

6. **Missing cross-references**: The RST never references `build_equation_map`, `build_equation_map_spherical`, `build_equation_map_cylindrical`, `build_rhs`, `build_rhs_spherical`, `solution_to_angular_flux`, or any other `sn_operator.py` public functions. These are important API entry points for the BiCGSTAB path.

#### E. Equations in RST Without Derivation Script Backing

- :label:`transport-cartesian` (line 128): Standard textbook equation, acceptable without derivation script
- :label:`transport-cartesian-2d` (line 147): Standard textbook, acceptable
- :label:`transport-spherical` (line 163): Standard textbook, acceptable
- :label:`transport-cylindrical` (line 182): Standard textbook, acceptable
- :label:`multigroup` (line 206): Standard textbook, acceptable
- :label:`balance-general` (line 512): **Needs derivation script** -- this is the core discretized equation
- :label:`alpha-recursion` (line 536): Verified numerically in `sn_contamination.py` (dome construction)
- :label:`alpha-cylindrical` (line 556): Verified numerically in `sn_contamination.py`
- :label:`dd-cartesian-1d` (line 374): **Needs derivation script** (SymPy verification of DD closure algebra)
- :label:`dd-cartesian-2d` (line 413): **Needs derivation script**
- :label:`dd-solve` (line 757): **Needs derivation script** (WDD substitution algebra)
- :label:`dd-recurrence` (line 839): **Needs derivation script** (cumprod recurrence)
- :label:`wdd-closure` (line 679): Verified in `sn_contamination.py` (M-M weights force beta=0)
- :label:`wdd-face` (line 687): Algebraic rearrangement of wdd-closure, trivial
- :label:`mm-weights` (line 695): Verified in `sn_contamination.py`
- :label:`reflective-bc` (line 1057): Trivial definition
- :label:`pn-scatter` (line 1148): **Needs derivation script** (Pn expansion correctness)
- :label:`flux-moments` (line 1158): Standard definition

**Summary**: 5 non-trivial labeled equations lack derivation script backing.

#### F. Tracked Items in IMPROVEMENTS.md That Need Documentation Attention

1. **DO-20260405-002 (OPEN)**: Gauss-type azimuthal quadrature. Mentioned in the RST (line 727) with a forward reference to IMPROVEMENTS.md. No Sphinx documentation needed until implemented, but the RST mention is correct.

2. **DO-20260405-003 (OPEN)**: phi-based cell-edge computation. Referenced in the RST (via the cylindrical cell-edge discussion, lines 708-716). No Sphinx documentation needed until implemented.

3. **DO-00000000-001 (OPEN)**: DSA. Not mentioned in the RST at all. Should at least be listed as a planned enhancement in a "Future Work" section.

4. **DO-00000000-003 (OPEN)**: Linear Discontinuous angular finite elements. Not mentioned in the RST. Should be mentioned as an alternative to WDD in the angular closure section.

5. **DO-00000000-006 (OPEN)**: Anisotropic scattering in curvilinear sweeps. The RST documents Pn scattering for Cartesian but does NOT note the limitation that curvilinear Pn is unverified. This is a documentation gap -- the RST should warn readers.

6. All 6 DONE items correctly reference their RST sections. No issues there.

### Directive 3: Quality Score Self-Assessment

Assessment of the EXISTING `docs/theory/discrete_ordinates.rst`:

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Derivation depth** | 4 | Full step-by-step derivations for balance equations, WDD closure, cumprod. Missing only some intermediate algebra for the 2D DD. |
| **Cross-references** | 3 | Good coverage of sweep functions and SNSolver methods. Missing all sn_operator.py public API. Private function refs may not resolve. |
| **Numerical evidence** | 4 | Homogeneous verification table (3 geometries x 3 group counts), heterogeneous convergence, sensitivity table. Missing spherical heterogeneous and angular convergence data. |
| **Failed approaches** | 5 | Excellent. Full investigation history with 6 failed approaches and detailed rationale. This is exemplary. |
| **Code traceability** | 3 | Good for sweep and solver. Missing for sn_operator.py, sn_quadrature.py construction algorithms. |
| **Derivation source** | 2 | Only sn_contamination.py provides backing. The core balance equation derivations, cumprod recurrence, and WDD algebra are all hand-written in RST. This is the weakest dimension. |

**Overall**: 3.5/5. Strong on physics exposition and investigation history. Weak on derivation script backing (5 non-trivial equations without scripts) and sn_operator.py documentation coverage.

### Priority Actions (ranked by impact)

1. **HIGH**: Create `derivations/sn_balance.py` with SymPy verification of the DD balance equations (1D, 2D, curvilinear) and the WDD substitution algebra. This would bring "Derivation source" from 2 to 4.
2. **HIGH**: Add an "Explicit Operator" subsection to the BiCGSTAB section documenting `EquationMap`, gradient scheme, and the face-flux averaging approximation. This closes the biggest code-without-docs gap.
3. **MEDIUM**: Fix the two stale docstrings in `sn_geometry.py` (module and class level).
4. **MEDIUM**: Add a "Limitations and Future Work" section referencing OPEN items from IMPROVEMENTS.md (especially DO-00000000-006 re: curvilinear Pn being unverified).
5. **LOW**: Add numerical convergence tables (spatial O(h^2) rates, angular spectral convergence) to the "Spatial and Angular Convergence" section.

---

- **What worked**: Reading ALL code files and ALL derivation scripts in parallel before writing the report. This gave a complete picture of what exists vs what is documented. Systematic checking of every :label: against derivation scripts was effective.
- **What was missing**: I did not build Sphinx to verify cross-reference resolution. A future audit should run `python -m sphinx -b html docs docs/_build/html` and check for warnings on unresolved references.
- **Convention discovered**: The project has a strict "derivations come from code" principle but the RST has substantial hand-written derivations. This inconsistency is the single biggest quality gap.
- **Improvement for next time**: Build Sphinx as the first step to get a machine-verified list of broken cross-references, then focus the manual audit on content gaps.

---

## 2026-04-05 — Review of verification.rst and session 2026-04-03 documentation

- **What worked**: Reading ALL generated RST fragments (not just the parent RST) exposed that the quality bottleneck is in the derivation modules' `latex` output, not in `generate_rst.py`. Building Sphinx first (zero warnings) let me focus on content quality rather than structural issues. Systematic review of all 6 IMPROVEMENTS.md files caught the DO-20260403-001 status inconsistency.
- **What was missing**: I did not run `python -m derivations.generate_rst` to check if the generated RST is stale vs current derivation modules. A future review should regenerate and diff.
- **Convention discovered**: The `VerificationCase.latex` field is the quality bottleneck for generated RST. Enriching documentation means enriching the derivation modules' LaTeX output, not editing RST directly.
- **Improvement for next time**: When reviewing generated documentation, always trace back to the generator script AND the data source (derivation modules). The three-layer architecture (derivation -> generator -> RST) means quality issues can originate at any layer.

### Quality Score for `docs/theory/verification.rst`:

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Derivation depth** | 2 | Final values only in most generated fragments. No intermediate steps for CP/SN/MOC. |
| **Cross-references** | 1 | Zero :func:, :class:, :mod: directives in the entire file. |
| **Numerical evidence** | 3 | 40-case verification table is good. No convergence tables for Richardson cases. |
| **Failed approaches** | 2 | Tolerance rationale captures some "why", but no design rationale or failure history. |
| **Code traceability** | 2 | Architecture section describes the system. No links to specific code. |
| **Derivation source** | 4 | All values come from derivations/ package (the system works). Content is thin though. |

**Overall**: 2.3/5. The architecture is sound (single source of truth works), but the content at each layer is too thin to meet the Sphinx-as-brain standard.

---

## 2026-04-05 — Review of session documentation contributions (IMPROVEMENTS.md audit)

- **What worked**: Systematic cross-checking of the user's "key knowledge" list against both IMPROVEMENTS.md trackers AND Sphinx theory pages. Grepping for specific phrases (e.g., "angular bias", "Gauss-Legendre") in RST files quickly revealed which knowledge items were captured vs missing. Building Sphinx first (zero warnings) confirmed structural integrity before auditing content.
- **What was missing**: The user's knowledge list was the only source for "what this session produced." Without it, I would have had to reconstruct the session's contributions from git log. Future reviews should always start with `git log --since` to independently verify the scope.
- **Convention discovered**: DONE status for implementation items is justified by autodoc alone (API docs count as Sphinx documentation), but theory chapters are tracked as separate OPEN items. This two-tier documentation (autodoc for API, theory RST for physics) is an established pattern across NM, DA, CP, DO, and HO modules.
- **Improvement for next time**: When auditing a session's documentation, always check for "operational knowledge" (rules of thumb, gotchas, when-to-use-X-vs-Y) that lives only in memory/conversation but not in any persistent artifact. The GL-vs-Lebedev insight is exactly this kind of knowledge -- easy to lose between sessions.

### Quality Score (for this review task, not a documentation page)

N/A -- this was an audit, not a documentation creation task.

---

## 2026-04-05 — Review of session CP/NG/verification documentation contributions

- **What worked**: Cross-referencing commit `fa5732c` content against IMPROVEMENTS.md claims revealed the false Sphinx reference in DA-20260405-003. Grepping the RST for specific terms (NG, 421, Mixture.ng) quickly disproved the "mentioned in architecture" claim. Checking for `gotchas.md` existence caught the stale RST reference.
- **What was missing**: I should have run `git show fa5732c --stat` earlier to understand which files were actually committed vs which were only described by the user. The benchmark factory (`benchmarks.py`) was a 537-line contribution with no tracker entry -- easy to miss without checking the commit diff.
- **Convention discovered**: When a commit bundles multiple logical changes (P.T fix + NG refactoring + new files), each logical change needs its own IMPROVEMENTS.md entry even though they share a commit hash. The project already does this correctly (CP-001 and CP-002 both reference fa5732c). But NEW artifacts created in that commit (benchmarks.py) also need entries.
- **Improvement for next time**: Always run `git diff <commit>~1 <commit> --stat` to get the full list of files changed, then verify each significant new file has a tracker entry. Do not rely solely on the user's summary of session contributions.

### Quality Score (for this review task)

N/A -- audit, not documentation creation.

---

## 2026-04-05 — Review of thermal_hydraulics.rst and reactor_kinetics.rst

- **What worked**: Line count ratio (RST lines / code lines) is a fast heuristic for depth. TH=0.34 and RK=0.30 vs DO=1.2 immediately flags the pages as thin. Reading the actual code functions (_solve_clad_stress, _compute_pressure, _wall_heat_transfer) and comparing to RST coverage revealed exactly which sections are missing. Building Sphinx first (zero warnings) confirmed structural soundness so the review could focus entirely on content quality.
- **What was missing**: The modules are not on the autodoc path, so cross-references like `:func:` cannot resolve. This means even if cross-refs were added, they would cause Sphinx warnings. The autodoc configuration needs updating before cross-refs can be added.
- **Convention discovered**: TH and RK modules have a shared pattern: continuous PDE in RST but no FD discretization. This is the opposite of the DO page which derives the discrete balance equation in full. The "continuous PDE only" pattern is the primary quality gap for engineering simulation modules where the discretization IS the implementation.
- **Improvement for next time**: For simulation modules (vs transport solvers), the discretization section is MORE important than the PDE section -- any textbook has the PDE, but the specific FD stencil on a non-uniform mesh is what makes the code work. Future TH/RK documentation should start with the discrete form and derive the continuous form as context, not the other way around.

### Quality Score

| Dimension | TH | RK |
|-----------|----|----|
| Derivation depth | 2 | 2 |
| Cross-references | 1 | 1 |
| Numerical evidence | 1 | 2 |
| Failed approaches | 3 | 3 |
| Code traceability | 1 | 1 |
| Derivation source | 1 | 1 |
| **Overall** | **1.5** | **1.7** |

---

## 2026-04-05 — Review of thermal_hydraulics.rst (second pass, by request)

- **What worked**: Reading the error catalog (ERR-010, ERR-011) alongside the RST revealed content that exists in the catalog but not in the documentation (L0 test assertions, analytical estimates). Cross-checking IMPROVEMENTS.md status against RST content caught 3 IMPL items that should be DONE and 1 item under wrong header. Reading conf.py confirmed autodoc gap before suggesting cross-references.
- **What was missing**: No derivation scripts exist for TH at all -- this is a complete gap, not a partial one. The previous review (session same day) scored TH at 1.5; this deeper review confirms 2.2 with the viscosity/investigation sections pulling the score up.
- **Convention discovered**: The TH page has a "tale of two halves" pattern: debugging/investigation narrative is exemplary (score 5) while physics derivation foundations are weak (score 1). This suggests the page was written during/after debugging sessions rather than as a planned documentation effort. Future TH documentation should start with the FV discretization derivation, not the PDE.
- **Improvement for next time**: When reviewing a module's documentation, always check conf.py for autodoc inclusion FIRST. If the module is not on the autodoc path, all cross-reference suggestions are blocked until that is fixed. This should be step 0 of any review.

### Quality Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Derivation depth | 2 | PDEs shown, FV discretization not derived |
| Cross-references | 1 | Single :func: in 837 lines, module not on autodoc path |
| Numerical evidence | 3 | DAE parity + IAPWS validation good, no correct-physics table |
| Failed approaches | 5 | Exemplary 3-phase investigation history |
| Code traceability | 1 | Functions mentioned by name, not linked |
| Derivation source | 1 | Zero derivation scripts for TH |
| **Overall** | **2.2** | |

---

## 2026-04-05 — Review of cross_section_data.rst and DA/ERR tracker entries

- **What worked**: Building Sphinx first (zero warnings) confirmed structural soundness. Checking the actual `__init__.py` code against the RST claims caught the false fallback description immediately. Cross-checking whether `:func:` targets exist on the autodoc path (by reading `api/data.rst`) revealed that both cross-references are silently broken -- Sphinx does not warn about unresolvable `:func:` by default, making this a manual-verification-only catch.
- **What was missing**: I should have a systematic checklist for "data pipeline" documentation vs "physics solver" documentation. The Archivist standard is calibrated for physics derivations (SymPy scripts, analytical benchmarks), but data pipeline pages need different quality criteria: schema documentation, round-trip validation, format specification completeness. The derivation-script requirement is less critical here but a programmatic validation script would still add value.
- **Convention discovered**: IMPL status in IMPROVEMENTS.md is inappropriate for "intentionally excluded features" (DA-20260405-009). The status vocabulary (OPEN/IMPL/DONE) assumes features that will be implemented. Conscious exclusions should be DONE (documented decision) or OPEN (future work), not IMPL.
- **Improvement for next time**: For every `:func:` and `:class:` directive in an RST file, verify the target module appears in an `.. automodule::` directive somewhere in `docs/api/`. Sphinx's default behavior of silently rendering unresolvable cross-refs as plain text makes this a high-miss-rate issue.

### Quality Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| Derivation depth | 3 | Record layouts thorough, assembly algorithm clear, some gaps (sig2, chi) |
| Cross-references | 1 | 2 `:func:` refs, both unresolvable. No `:ref:` to other pages. |
| Numerical evidence | 4 | Two kinf matches + H-1/U-235 component tables. Missing U-238. |
| Failed approaches | 4 | sigT consistency section is excellent investigation history |
| Code traceability | 2 | Inline code shown, not linked via Sphinx |
| Derivation source | 1 | No derivation script exists |
| **Overall** | **2.5** | Good data reference, weak on cross-refs and derivation backing |

---

## 2026-04-05 — Update collision_probability.rst: GS solver, (n,2n), consolidated eigenvalues

- **What worked**: Reading the error catalog (ERR-015, ERR-016) alongside the IMPROVEMENTS.md entries gave complete context for documenting both the physics and the failure history. The error catalog's "first wrong fix attempt" and "how it hid from tests" sections were directly usable in the RST. Reading the actual code (`_solve_fixed_source_gs`, `compute_keff`) verified the formulas before writing them into the RST. Building Sphinx with a clean build directory caught the duplicate `keff-update` label conflict with `homogeneous.rst`.
- **What was missing**: The `derivations/_eigenvalue.py` module is not on the autodoc path, so `:func:` cross-references to `kinf_homogeneous` and `kinf_from_cp` will render but may not hyperlink correctly. This is an existing limitation (the derivations package is only partially on the autodoc path).
- **Convention discovered**: The label `keff-update` was already used in `homogeneous.rst`. Label names in ORPHEUS Sphinx must be page-prefixed (e.g., `cp-keff-update`, `ho-keff-update`) to avoid collisions across theory pages. This was not documented anywhere previously.
- **Improvement for next time**: Before adding any `:label:` to a math directive, grep all RST files for the proposed label name to catch duplicates before building. The Sphinx error message points to the duplicate but fixing it post-build wastes a rebuild cycle.

### Quality Score

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Derivation depth** | 4 | Full inner iteration algorithm with 4 numbered steps, fixed-point equation, why inner iterations are meaningful |
| **Cross-references** | 3 | :class:`CPParams`, :class:`CPResult`, :class:`CPSolver`, :func: to eigenvalue module. Private methods not linked. |
| **Numerical evidence** | 3 | 106-test count, tolerance table, ERR-015 12% error magnitude. No convergence rate tables for GS vs Jacobi. |
| **Failed approaches** | 5 | Full ERR-016 tautological residual history, ERR-015 first-wrong-fix, QA misdiagnosis |
| **Code traceability** | 4 | Code snippets for both Jacobi and GS, compute_keff pseudocode, parameter/diagnostic tables |
| **Derivation source** | 3 | _eigenvalue.py equations cited with full math. Inner iteration formulas are hand-written (no derivation script for GS convergence). |

**Overall**: 3.7/5. Strong on failure history and algorithm description. Could improve with a convergence comparison table (GS vs Jacobi iteration counts for a reference problem).

---

## 2026-04-05 — Round 2 forensic comparison of collision_probability.rst

- **What worked**: Extracting all 597 removed lines from `git diff` and systematically categorising each as reformulated/consolidated/lost. Checking `:eq:` cross-references via shell script (grep all refs, verify each label exists) was efficient and caught zero issues. Checking `:label:` uniqueness via `sort | uniq -d` was instant. Searching for specific phrases from round-1 items via grep gave immediate pass/fail per item.
- **What was missing**: Sphinx is not installed in this environment, so I could not verify cross-reference resolution or citation warnings via a real build. The `[Hebert2009]` vs `[Hébert2009]` mismatch would have been caught instantly by a build. Future forensic comparisons MUST have Sphinx available.
- **Convention discovered**: When a reference definition changes encoding (accent stripped), ALL citations must be updated. The committed version had `[Hébert2009]` consistently; the working copy changed the definition to `[Hebert2009]` but missed one citation at line 1530. This is a class of bug that only manifests across definition/citation pairs.
- **Improvement for next time**: For forensic comparisons, build a checklist of "cross-file consistency pairs" (citation def/use, label def/ref, substitution def/use) and verify each pair programmatically. Do not rely on grep for individual terms when the issue is consistency between two different spellings of the same identifier.

### Quality Score (for the forensic audit task itself)

N/A -- this was a verification audit, not documentation creation.

---

## 2026-04-06 — Re-review of monte_carlo.rst after solver restructuring (MT-20260406-008)

- **What worked**: Collecting actual test counts via `--collect-only` immediately caught the inflated "55 tests" claim. Checking the autodoc path (`docs/api/`) for `monte_carlo` confirmed that all ~40 cross-references render as plain text. Reading IMPROVEMENTS.md alongside the RST revealed multiple stale statuses (OPEN items that are now IMPL or DONE).
- **What was missing**: The previous review's "3 blocking fixes" were not explicitly listed in a machine-readable format -- I had to infer them from context. Future reviews should record blocking fixes as a numbered checklist in the lessons file for easy re-verification.
- **Convention discovered**: The test count in RST documentation is a high-risk claim because tests get added/removed/reorganized independently of the RST. Any test count claim should cite the specific test files and use `--collect-only` to verify before documenting.
- **Improvement for next time**: Always run `pytest --collect-only` as the FIRST step when reviewing any RST that claims a test count. This takes 1 second and catches inflated numbers immediately.

### Quality Score (for monte_carlo.rst)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Derivation depth | 4 | Delta-tracking proof, weight conservation proofs, splitting derivation -- all present but hand-written |
| Cross-references | 4 | ~40 :class:/:func: refs added for architecture, but none resolve (no autodoc) |
| Numerical evidence | 4 | Homogeneous + heterogeneous tables, tolerance analysis, specific XS values |
| Failed approaches | 5 | ERR-017 (24% error, NaN collapse), ERR-018 (13% step shortening) -- exemplary |
| Code traceability | 4 | Architecture pseudocode, annotated code snippets for all 5 layers |
| Derivation source | 3 | Eigenvalue refs to derivations/ good. Proofs hand-written (no derivations/mc_proofs.py) |
| **Overall** | **4.0** | Up from previous 4.2 (recalculated). Blocking: test count factual error. |

---

## 2026-04-06 — Review of method_of_characteristics.rst (new chapter)

- **What worked**: Running `pytest --collect-only` first (per lesson from MC review) confirmed the 102-test claim is accurate. Building Sphinx first confirmed zero warnings. Comparing directive counts against CP/DO reference chapters immediately quantified the depth gap. Reading `derivations/moc.py` revealed it only provides eigenvalue references, not equation verification -- this is the same pattern as MC.
- **What was missing**: I did not check for an IMPROVEMENTS.md for the MOC module. Future reviews should always check for tracked items. I also did not verify whether the `:func:` references to private functions (`_ray_box_intersections`, `_ray_circle_intersections`) actually resolve in autodoc.
- **Convention discovered**: Dead citations (referenced in bibliography but never cited in body text) are a recurring pattern. [Askew1972] and [KnottYamamoto2010] are defined but never referenced. This was also flagged in the DO review for [Bailey2009].
- **Improvement for next time**: Add a "dead citation check" to the review checklist: grep for `[Name20XX]_` in body text and verify each bibliography entry has at least one citation. This is trivially automatable.

### Quality Score (for method_of_characteristics.rst)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Derivation depth | 7 | Core chain solid (PDE->ODE->attenuation->boyd-eq-45). bar-psi not derived. keff-update not derived. |
| Cross-references | 7 | 26 directives, good class/func coverage. Zero :ref: to other pages. Dead citations. |
| Numerical evidence | 6 | Homogeneous + heterogeneous tables present. Angular/polar convergence qualitative only. No convergence rates. |
| Failed approaches | 9 | ERR-019 exemplary: symptoms, root cause, degeneracy explanation, lesson. Minor: no rejected hypotheses. |
| Code traceability | 7 | Key classes/methods linked. Missing _is_vertical, derivations.moc module ref. |
| Derivation source | 5 | derivations/moc.py exists but only does eigenvalue refs. Core equations hand-written in RST. |
| **Overall** | **6.9** | Solid first draft. Strongest: ERR-019. Weakest: numerical evidence, depth. |

### Blocking fixes (numbered for re-verification)

1. Add angular convergence table (n_azi sweep with keff values) -- DONE (lines 972-1003)
2. Add polar convergence table (TY-1/TY-2/TY-3 with keff values) -- DONE (lines 1007-1035)
3. Derive bar-psi from integral of ODE solution -- DONE (lines 535-598, excellent)
4. Add :ref: cross-references to CP, DO, verification pages -- PARTIAL (CP+DO done, verification missing)
5. Remove or cite [Askew1972] and [KnottYamamoto2010] in body text -- DONE (cited in overview, line 40-41)

---

## 2026-04-06 — Round 2 review of method_of_characteristics.rst (1274 lines)

- **What worked**: Checking for duplicate labels via `grep :label: | sort | uniq -d` immediately caught two collisions (keff-update, wigner-seitz) that Sphinx does not warn about. Verifying autodoc path existence before assessing cross-reference quality (per lesson from MC review) correctly identified the 35 non-resolving directives. Running `pytest --collect-only` first (per lesson from previous review) confirmed the 102-test claim.
- **What was missing**: I did not verify whether the convergence table values match what the actual tests produce. The numbers in the RST could be stale if the solver changed since the tables were written. Future reviews should run at least one convergence point to spot-check.
- **Convention discovered**: Math label collisions across RST files are silent in Sphinx (no warning). The previous lesson (CP review) established page-prefixed labels (e.g., `cp-keff-update`). This MOC page introduces two new collisions. This class of bug must be caught by manual `grep | uniq -d` since the build does not flag it.
- **Improvement for next time**: Add duplicate label check to the standard review checklist as step 0 (before even reading the file). It takes 1 second and catches a class of bug the build misses.

### Round 1 blocking fix verification

| # | Fix | Status |
|---|-----|--------|
| 1 | Angular convergence table | DONE |
| 2 | Polar convergence table | DONE |
| 3 | bar-psi derivation | DONE |
| 4 | :ref: cross-references | PARTIAL (CP+DO done, verification missing) |
| 5 | Dead citations | DONE |

### Quality Score (round 2)

| Dimension | R1 | R2 | Notes |
|-----------|----|----|-------|
| Derivation depth | 7 | 8.5 | bar-psi from first principles, keff-update with (n,2n). Missing: ODE integrating factor, angular discretization step. |
| Cross-references | 7 | 7.5 | 3 :ref: added, dead citations fixed. Blocking: no autodoc page (35 directives non-resolving). |
| Numerical evidence | 6 | 8.5 | 3 convergence tables with real numbers. Missing: theoretical order discussion, flux shape column. |
| Failed approaches | 9 | 9 | ERR-019 exemplary. Missing: rejected hypotheses during investigation. |
| Code traceability | 7 | 7.5 | Code snippet + variable mapping good. Blocked by autodoc gap. |
| Derivation source | 5 | 5 | derivations/moc.py only does eigenvalue refs. Core equations (bar-psi, Boyd-45, keff) all hand-written. |
| **Overall** | **6.9** | **7.7** | |

### Round 2 blocking fixes

1. Rename `keff-update` to `moc-keff-update` and `wigner-seitz` to `moc-wigner-seitz` (duplicate labels)
2. Create `docs/api/moc.rst` with automodule directives (activates 35 cross-refs)
3. Update IMPROVEMENTS.md: MC-20260406-005 should be DONE (ERR-019 fully documented)

---

## 2026-04-06 — Round 3 review of MOC theory chapter (8.3/10)

- **What worked**: Running Sphinx with `-W` (warnings-as-errors) confirmed zero build warnings. Running the derivation script confirmed all 3 proofs pass. Checking for `nitpicky` mode in conf.py revealed that unresolved cross-refs are silently swallowed -- critical finding for cross-reference scoring.
- **What was missing**: `nitpicky = True` in conf.py would have caught the `MoCGeometry` dead reference and unqualified `pwr_pin_equivalent` automatically. Without it, manual grep of `:func:/:class:/:meth:` against actual code symbols is required.
- **Convention discovered**: Private functions (`_ray_box_intersections`, etc.) referenced via `:func:` don't resolve unless `:private-members:` is added to the automodule directive. Best practice for private implementation details is to use code literals instead.
- **Improvement for next time**: Always check whether `nitpicky = True` is set. If not, manually verify every cross-reference against the codebase. The zero-warnings build gives false confidence about cross-ref resolution.

### Quality scores (round 3)

| Dimension | Score |
|-----------|-------|
| Derivation depth | 9 |
| Cross-references | 7 |
| Numerical evidence | 9 |
| Failed approaches | 9 |
| Code traceability | 8 |
| Derivation source | 8 |

### Round 3 fixes needed (cross-refs only)

1. `MoCGeometry` on line 1168 -> code literal (class no longer exists)
2. `_ray_box_intersections`, `_ray_circle_intersections`, `_is_vertical` -> code literals (private, won't resolve)
3. `pwr_pin_equivalent` on line 102 -> fully qualified `:func:`~geometry.factories.pwr_pin_equivalent``

---
