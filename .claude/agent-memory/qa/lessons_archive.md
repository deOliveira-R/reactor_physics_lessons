# QA Lessons

Behavioral corrections only. AGENT.md has the V&V hierarchy,
anti-patterns, and error catalog format -- never duplicate here.

> **Promoted to AGENT.md (2026-06-22):** the standing stance "a green
> gate is evidence of nothing until you have made it RED — mutation-verify
> every gate's teeth under `-O`, in-process, revert by re-editing" is now
> Enforcement #11 in `.claude/agents/qa/AGENT.md`. The lessons below keep
> the per-incident *mechanics* (which mutation point, which sentinel, which
> revert proof) — those stay here as recalled technique. The *rule* lives in
> the definition. Recurring instances: L-007, L-014, L-020, L-024, L-027,
> L-031, L-033, L-036, L-039, L-040, L-042, L-045..L-050.

---

## L-001 -- Test count is not coverage

20 passing tests (homogeneous exact, conservation, balance,
non-negativity) missed a fundamental 2-term bug in cylindrical DD.
Signature: keff diverging under mesh refinement (1.15 -> 0.90 -> 0.52).

**Rule**: When reviewing "all tests pass" for any solver, first ask:
"Is there a heterogeneous mesh-refinement convergence test?"
If not, the suite proves nothing about the transport operator.

---

## L-002 -- Orphan equation triage class-D before class-B

When closing orphan equations on a Sphinx theory page, scan the
**existing** test suite for evidence the equation is already verified
under a different label BEFORE writing a new test or marking it
documented. The Peierls/case-method/V_α suites have a long history of
adding equations to a theory page and adding tests in a test file
without wiring the `@pytest.mark.verifies("label")` connection. The
audit interprets these as orphans, but the test exists — adding the
label is a 1-line fix that closes the orphan with no new verification
work.

**Rule**: For a new orphan, the search order is D -> B -> A -> C:
1. Class D (existing test, just needs the label) — most common in
   ORPHEUS, scales to 25 percent or more of orphans on the trajectory-
   resolvent / peierls-Nystrom pages.
2. Class B (definitional / derivation step / governing equation) —
   bulk of the rest; mark `.. vv-status: <label> documented` with a
   rationale comment per Cardinal Rule 3.
3. Class A (write a real test) — only when no test covers and the
   equation is a verifiable claim.
4. Class C (stale, remove) — rare; the test suite usually catches
   stale labels via the audit's drop-into-orphan signal.

---

## L-003 -- The matrix.rst orphan list can lag the live RST

`docs/verification/matrix.rst` is auto-generated from
`tools/verification/generate_matrix.py` on every Sphinx build. Until
the build runs after a label rename, the matrix snapshot lists the OLD
labels. ALWAYS re-run `python -m tests._harness.audit --gaps` to get
the live orphan list before classifying — do not trust the matrix.rst
snapshot for label spelling. Observed during the 2026-05-03 78-orphan
sweep with `case-method-eqXX` -> `singular-eigenfunction-eqXX` rename:
the snapshot showed `case-method`, the audit showed `singular-
eigenfunction`. Mass-applying labels from the snapshot would have
created 5 self-inflicted orphans.

---

## L-004 -- vv-status rationale comments must NOT use [brackets]

The `:vv-status: documented` directive lives in the same RST file as
the labelled equation, conventionally as a top-level RST comment
(`.. vv-status: <label> documented`). When attaching a rationale
comment block with `..    [category] description` formatting,
docutils parses each [xxx] as a citation reference, producing
"duplicate citation" warnings under `-W`. Use (parens) instead of
[brackets] in rationale comments, and prefer a single-line `..`
comment over a multi-line `..  / .. / ..` continuation block.

---

## L-005 -- Locating slow/timeout tests in tests/derivations

The whole `tests/derivations` suite CANNOT be run in one bounded
process to find the `Timeout (>60.0s)` tests: with `--timeout=60
--timeout-method=signal` the per-test 60s stalls accumulate past any
sane `gtimeout` wall (even `-n 6` xdist hit the 600s cap mid-run and
junit-xml is NOT written when the process is SIGTERM-killed, so the
`-rfE` reason summary is lost). Working method:

1. Split into batches that each COMPLETE: the slow tests cluster
   entirely in `test_peierls_*` files. `test_fn*`/`*la13511*` (13
   unique files) and all 32 non-peierls/non-fn files are fast (0
   timeouts) — clear them as 2 group runs first.
2. For the peierls group, the 4 heavy files (`test_peierls_reference`,
   `_nystrom_verification`, `_convergence`, `_specular_bc`) plus
   `test_peierls_greens_function_cylinder_mr` hold ALL the timeouts;
   run each suspect file ALONE with `-n 8 -q -rfE --tb=no` so the
   per-test `FAILED ... - Failed: Timeout (>60.0s)` reason line is
   captured (only a COMPLETED run writes the short-summary reasons).
3. The other ~46 peierls files run clean as one group.

The 2026-06 sweep found exactly 20 timeout tests, all genuine
`Timeout (>60.0s)` (zero real assertion/exception failures): 5 in
`test_peierls_reference`, 11 in `_specular_bc`, 2 in
`_nystrom_verification`, 1 in `_convergence` (`cp_slab_1eg_2rg`
param), 1 in `_greens_function_cylinder_mr`.

**Param-level precision**: when only SOME params of a parametrized
test time out (sphere passes, cylinder+slab stall), mark just the
slow params with `pytest.param(..., marks=pytest.mark.slow)` rather
than the whole function/method — keeps the fast params in the
default lane. Verify with `--collect-only -m "not slow"`.

---

## L-006 -- Mode-8 (-O strip) classification: rewriter boundary + testpaths

Two facts gate whether a bare `assert` is a real `-O` false-green:

1. **Assertion-rewriter boundary.** pytest rewrites bare asserts ONLY
   in (a) collected test modules and (b) registered conftest/plugins.
   Asserts in `orpheus/` production modules (incl. `orpheus/derivations/`)
   are NEVER rewritten, so under `-O` they are inert NO-OPs. (`np.testing.*`
   / `pytest.fail` are function calls -> fire under `-O` regardless.)
2. **`testpaths = ["tests"]`.** The canonical suite collects ONLY `tests/`.
   `test_*` wrappers that live INSIDE an `orpheus/derivations/*.py` module
   (e.g. `balance.py:test_cartesian_1d`) are NOT collected -> their
   internal asserts run only on a manual `pytest orpheus/.../balance.py`
   (the docstring usage), never by `python -O -m pytest`. Those are class-D
   (dead w.r.t. the suite), NOT class-A.

**Classification recipe** for a bare assert in `orpheus/`:
- Nexus `callers` on the function node. Filter to callers whose id starts
  `tests.` (collected) vs `orpheus.derivations.` (in-module wrapper, dead).
- If a COLLECTED test calls it: read what the test independently asserts on
  the RETURN VALUE. If the test cross-checks the same property against a
  structurally-independent path (e.g. SymPy coeffs vs the DD sweep), the
  internal assert is class-B redundant (the H4 self-reference trap does NOT
  bite because producer and consumer are independent). If the test only
  consumes the return and the assert is the sole correctness gate -> class-A.
- `isinstance(...)` after a `case`/`if coord`-branch, and `x is not None`
  on an Optional the contract guarantees -> class-C type-narrowing (strip =
  designed; downstream AttributeError if ever violated).
- `assert row == n_unk` before `np.linalg.solve` (matrix-assembly row count),
  `if __debug__:` blocks, `assert <closed-form sanity>` guarding `[0]`
  indexing of a `sp.solve` result whose REAL verification is a returned
  `pass_*` boolean -> class-C.
- Import-time `validate_all()` on a HARDCODED data table (xs_library.py:307,
  `np.allclose(sig_t, sig_c+sig_f+sig_s.sum)`) with NO independent
  collected-test coverage and a constructor (`Mixture`) that does NOT
  re-validate -> class-A genuine false-green: the only consistency gate on
  the canonical XS library, silently inert under `-O`.

**The #228 audit** (56 sites): 1 class-A (`xs_library.py:307`), rest C/D.
The original premise (test_keff_2d bare asserts inert) was REFUTED -- those
ARE collected -> rewritten -> fire under `-O`.

---

## L-007 -- foundation + verifies(...) is silent level conflation

The harness (`tests/_harness/registry.py`) accepts `@pytest.mark.foundation`
stacked with `@pytest.mark.verifies("<eq>")`: `_existing_level` resolves
`level="foundation"` while `_collect_str_marker_args(item,"verifies")`
SEPARATELY records the equation, so Nexus writes a `tests` edge and the
audit credits the physics equation with the foundation test's parametrizations
(observed: a documented eq showed "6 test(s)" from one 6-param foundation test
that never touches a non-flat ψ). The registry docstring forbids it verbatim
("Foundation tests never carry a `verifies(...)` marker"), but NOTHING enforces
it — collection is silent.

**Rule**: a `foundation` test verifies a SOFTWARE invariant (data-structure /
factory / reflection-index contract); it MUST NOT carry `verifies(<physics-eq>)`.
The tell is a `documented`/representational-identity equation whose ONLY
"coverage" is a foundation test. Check the theory page's `.. vv-status:`
rationale: if it names the REAL verifiers (MMS operator-admission gate,
adjoint bit-identity) and the foundation test is not among them, the marker is
a misleading edge — drop `verifies(...)`, keep `foundation`, reference the eq
in prose via `:ref:` only. Fix is 1 line (delete the marker); re-run
`python -O -m tests._harness.audit` to confirm the eq drops out of the
coverage attribution. (Caught 2026-06 on
`test_coupled_pole_mu_level_invariant.py`, eq `sn-err-058-coupled-pole-continuity`.)

---

## L-008 -- False-xfail under a stale index: verify the FAILURE REASON, not just xfail status

A `xfail(strict=True)` test is "satisfied" the moment it fails for ANY
reason -- the strict gate only checks pass/fail, never WHY. A stale array
index (e.g. `(ng,nx)` flux read as `values[:,0]` -> length-1 cell-0 slice
broadcast against a length-nx reference -> garbage L2 ~14 that DIVERGES)
makes the test a FALSE xfail: green suite, but failing for a reason
unrelated to its documented xfail reason (#229 floor). The W1 review
(2026-06-13) caught the FIXED version: `values[:,0]`->`values[0,:]`.

**Rule**: when a diff touches an array index inside a strict-xfail test,
re-run it with `pytest --runxfail` to surface the REAL failure and confirm
it matches the xfail `reason=` (here sphere orders [1.995,1.999,1.407] +
err[-1]=1.4e-3 = the genuine #229 fine-mesh floor; cylinder orders ~0
err~1.9e-2 = structural floor). Then re-run WITHOUT --runxfail (`-rxX`) to
confirm it stays XFAIL (no strict-XPASS suite break). Same bug class as
the rank-d-carve fallout (`Q_numerical[:,0,:,0]->[:,0,:]`); `(ng,nx)`
1-group MMS is ALWAYS `[0,:]` for the radial profile, `[:,0]` is cell-0.

---

## L-009 -- "floor scales with quadrature" is a falsifiable gate, not a tautology

When a fix is claimed to clean a convergence RATE but NOT remove a floor,
the honest gate (vv anti-pattern #5/#17) pins the floor as a verified
quadrature-SCALING quantity rather than asserting "floor removed". The W1
`test_w1_aniso_sphere_floor_scales_with_quadrature` asserts
`err(S32,nx=160) < err(S16,nx=160)/2.0`. This is FALSIFIABLE: a fixed
closure-bug floor would be quadrature-independent -> ratio ~1.0 -> gate
FAILS. Only a #229-style interpolation floor (the half-angle thread is
interpolated, scales with angular order) passes (measured ratio 3.42).
A "floor scales" gate is a CLAIM about floor character, distinct from
both the rate gate and a (false) removal gate -- accept it.

---

## L-010 -- Mode-8 (-O) does NOT apply to bare asserts in collected tests/ modules

A bare `assert` in a test file UNDER `tests/` IS rewritten by pytest's
assertion rewriter at COLLECTION time and FIRES under `python -O` — the
interpreter `-O` flag strips asserts in NON-rewritten modules only
(production `orpheus/`, see L-006). The `PytestConfigWarning: assertions not
in test modules or plugins will be ignored` is about non-test-module asserts;
it does NOT mean the collected test's bare asserts are inert. Definitive
probe (run once if unsure): a collected `test_x(): assert 1==2` under
`-O` FAILS. So a W2/W3/W4-style file whose load-bearing gates are all bare
`assert np.all(orders > 1.9)` is SAFE under the canonical `-O` invocation.
Do NOT raise a Mode-8 flag for bare asserts in `tests/` — reserve Mode-8 for
bare asserts in `orpheus/` production paths (L-006 recipe).

**When a BRIEF hands you a Mode-8 hypothesis about a `tests/` subtree, settle it
in 2 min and PIVOT** (2026-07-30, boundary-machinery quadrant; the brief's
"highest priority" premise was FALSE and the headline came out `0/676 = 0.0 %`
inert). Two-step proof: (1) synthetic control — 3 failing tests (bare `assert` /
`np.testing` / `pytest.fail`) run WITH and WITHOUT `-O`, identical verdict;
(2) realistic — COPY two real files to `$CLAUDE_JOB_DIR/tmp/`, falsify one bare
assert each (`assert <expr> and False, 'MUT must FAIL'` preserves the original
expression), run both modes. The REAL Mode-8 surface to grep is then only:
bare asserts in the subsystem's `orpheus/` files + **non-collected helpers under
`tests/`** (a `_generate_*.py` snapshot generator matching no `python_files`
pattern IS `-O`-inert — low severity, but it is the honest hit).
Then pivot the quadrant to **what the asserts assert**: an AST census
(bare / `np.testing.*` / `pytest.raises` / `pytest.fail` / `assert_*` calls) plus
a bare-assert CONTENT classification (`structural` isinstance-hasattr-`is` /
`tag_equality` / `shape_len` / `predicate` / `membership` / `numeric`). Measured
on the boundary suite: 61.7 % bare, but only **29.1 % of bare asserts pin a
VALUE** — the `tag_equality` bucket is the test-side shadow of a production
stringly-typed-dispatch finding and is the migration surface if the plan types
that dispatch. Scripts worth re-deriving: `count_asserts.py`, `classify_asserts.py`.

---

## L-011 -- Replicate the test's OWN solve helper before judging a "value" claim

A prescribed-inflow / non-vacuum-BC MMS value-claim depends on `q.boundary`
being wired. A naive `solve_sn_fixed_source(...)` (vacuum default) reproduces
the WRONG number (measured 50% outer-cell error vs the test's 0.26%) — which
looks like the test is lying but is actually the test's own catcher firing
("vacuum inflow misses the A(R)>0 surface term"). ALWAYS import and call the
test module's internal `_solve(case, nc)` to reproduce a value claim; a
divergent hand-replication usually means YOU dropped the BC, not the test.
(Caught 2026-06-13 W3 prescribed_inflow review: test's `_solve` gave max-rel
2.637e-3 at nx=160, matching the docstring exactly.)

---

## L-012 -- "BC X is load-bearing because k=k_inf" holds ONLY for homogeneous

A directional-eigenvalue test's "vacuum BC is load-bearing — a reflective
sphere has k=k_inf, flux-shape independent, so P1 can't change k" reasoning
is TRUE only for a HOMOGENEOUS reflective medium. For a HETEROGENEOUS
reflective finite sphere the flux is NON-flat, so P1 DOES change k via
spectral/spatial coupling (measured Δ=2.4e-2 reflective vs 1.4e-2 vacuum —
reflective Δ was LARGER, not zero). The k=k_inf control fires cleanly only
on a homogeneous fissile sphere (measured reflective Δ=1.5e-12, machine
zero). When a docstring justifies a BC choice via k_inf, check the config:
if it is heterogeneous, the justification is wrong even though the test's
assertions (on the vacuum config) still hold — a Cardinal-Rule-3 (WHY must be
right) doc-correctness flag, not a test-validity failure.

---

## L-013 -- Verbatim-relocation claims: prove by NORMALIZED ast-diff, not by re-running gates alone

A "this code moved verbatim with deterministic substitutions X→Y" claim is
provable MECHANICALLY: extract the OLD body + NEW body, apply the claimed
substitutions to OLD (`.replace("self.sn_mesh","self.mesh")` etc.), strip
docstrings/imports/comments/blanks, and `difflib.unified_diff`. A TRUE
verbatim relocation reduces to ONLY: the signature line + the deliberately-
added fork (e.g. an `emit_angular`-guarded block) + the return-shape change.
Any other line in the diff is an unaudited edit hiding inside a "pure move"
claim. This is FAST (seconds) and far stronger than re-running a regression
gate, because the gate may admit drift (see L-014) or have a coverage hole
(see L-015). Used 2026-06-14 on #206 Phase C: the 1-D `_compute_LpC`→
`_apply_walk` move + `_compute_LpC_transpose`→`loss_action_transpose` move
both reduced to signature-only diffs (transpose: ZERO body lines changed).

---

## L-014 -- A regression gate's HARD floor and its STRICT (bit-identity) floor are different gates

A `kind="direct"` regression assert (`assert_array_almost_equal_nulp(
nulp=reduction_depth)`) HARD-tolerates up to `reduction_depth` ULP. "Bit-
identical (0 ULP)" for a pure-refactor PR is enforced ONLY by the
`-W error::DriftWarning` escalation layered ON TOP. So "the gate passes"
≠ "bit-identical" — verify WHICH invocation ran. Prove the strict floor is
LIVE (not a false gate): perturb the committed baseline `.npy` by 1 ULP
(`np.nextafter`), run with `-W error::DriftWarning` → MUST fail; without it
→ passes with a DriftWarning. Restore via `git checkout --` (np.save appends
`.npy` to a manual `.bak` name — don't hand-roll the backup).

---

## L-015 -- conftest filterwarnings overrides are SESSION-GLOBAL but do NOT cross-leak to sibling dirs (verify, don't assume)

`tests/sn/regression/conftest.py::pytest_configure` does
`config.addinivalue_line("filterwarnings","always::DriftWarning")`, which
makes `-W error::DriftWarning` INERT for that directory's own iterative DD
snapshots (they emit 100s–10000s ULP drift but never fail under `-W error`).
The fear: this leaks to a sibling gate (`tests/sn/sweep/core/`) co-collected
in the same session → false green. EMPIRICALLY DISPROVEN 2026-06-14: with a
1-ULP-perturbed `sweep/core` baseline AND `tests/sn/regression/` co-
collected, the `sweep/core` A-NEW gate STILL FAILED under `-W error::
DriftWarning` (per-item filterwarnings precedence: the `-W` CLI filter beats
the conftest `addinivalue_line` for items OUTSIDE regression/). So a "the
override leaks" worry is testable, not theoretical — perturb-and-run before
flagging it as a blind gate.

---

## L-016 -- "branch fires under quadrature Q" claims need a degeneracy probe, not faith

#206 claim 5 asserted the cylinder pure-azimuthal degenerate-ordinate branch
(`|mu_x|<1e-15`, `A_downstream=0`) "fires under the A-NEW matvec[CYL] leg".
FALSE: `Quadrature.level_symmetric(sn_order=2..8)` ALL have `min|mu_x| ≥
0.22` — ZERO degenerate ordinates. The branch is dead code under standard
LS cubature and is exercised by NO current test. Probe in 3 lines
(`np.count_nonzero(np.abs(q.mu_x)<1e-15)`) before accepting a branch-
coverage claim. (Not a regression — the branch was a verbatim relocation,
proven by L-013 — but the EVIDENCE for it is vacuous; flag the coverage gap.)

---

## L-017 -- Diffusion-limit silent-error: probe the THICK-CELL regime, not just refined mesh

A spatial scheme advertised "diffusion-limit-consistent" but shipped with a
FLAT source (slope source Q̂=0, e.g. LD Increment A #158) is O(h²) AND exact
on linears AND passes a sin-ansatz MMS -- yet SILENTLY loses the diffusion
limit on optically THICK cells. The MMS ladder hides it because every
refinement drives σ_t·h → thin where flat-source LD is fine; the failure
lives at σ_t·h ≫ 1 (coarse mesh on a diffusive medium), exactly where a
practical user runs.

**Probe recipe** (the discriminating oracle is DD, which IS interior-diffusion-
consistent via WDD): fixed coarse mesh, vacuum BC, eps-scaled diffusive
material (σ_t=1/eps, σ_a=eps, c→1, Q=eps), compare DD_mid vs scheme_mid as h
refines. Measured #158-A: DD holds ~0.950 at every refinement; flat-source LD
gives 0.401 at σ_t·h=100 (~58% deficit) and only recovers (0.884) at σ_t·h=12.5.
DO NOT trust a REFLECTIVE-BC infinite-medium probe at c≈1 -- it needs 1e5+
inner iters and both DD and LD read 81.9% wrong from non-convergence (a probe
artifact, NOT physics). Vacuum thick-cell head-to-head vs DD is the clean cut.

**The flag**: this is a SILENT wrong-answer exposure when (a) the docstring
headline claims "diffusion-limit-consistent" / "all four diffusion limits"
(true of full LD, FALSE of the flat-source cut that shipped), (b) the flat-Q̂
restriction is buried in code comments only ("flat (Q̂=0)", "Increment C"),
(c) NO user-facing warning and NO xfail/tripwire guards the interim. A
deferred-to-increment-C limitation needs EITHER a forward xfail tripwire
(strict=False, flips to xpass when C lands) OR a docstring user-warning that
the diffusion limit requires the moment source -- a staging note in a plan
file is NOT enough when the public entry (solve_sn_fixed_source cell_update=)
accepts the scheme NOW.

---

## L-018 -- "matvec path tested" needs an instrumented call-count, not a round-trip

A batched residual_kernel_batch round-trip test (residual(cell_kernel_batch(q))≈0)
is a SELF-consistency check (both arms share _kernel_terms) -- it is NOT the
L14 leg-2 (matvec correct) or leg-3 (matvec≡sweep). To know whether the
forward matvec is even EXERCISED end-to-end, monkeypatch-count the two kernels
during the solve: #158-A LD MMS solve = 1600 cell_kernel_batch (sweep/solve)
+ 0 residual_kernel_batch (matvec). SI sweeps never touch loss_action. The
matvec runs ONLY under inner_solver='krylov'. Probe it: LD-via-Krylov gave the
SAME flux as LD-via-SI to 4.1e-14 (matvec IS correct + matvec≡sweep holds) --
but NO committed test drives it, so the claim "matvec works" is true-but-
unverified (NEEDS-EVIDENCE: a 1-line inner_solver='krylov' MMS sibling closes it).

---

## L-019 -- A stress-ansatz flagged by the test-architect is a binding contract, not advice

When the test-architect memo's GATE spec mandates an angularly-non-trivial,
mixed-scale (k=1,3), heterogeneous-2G, a0>0-non-vanishing-at-boundary stress
ansatz AND the shipped MMS test instead uses build_1d_slab_mms_case() (the
canonical sin(πx/L), 1G, homogeneous -- the EXACT Mode-7 simplification bias
the spec said to override), that is a gate-downgrade, not a stylistic choice.
The sin ansatz: vanishes at both faces (BC handling untested), isotropic-in-μ
(no per-ordinate spatial variation in the moments), 1G (flux-shape degenerate
per H1), homogeneous (nulls redistribution per H2). It cannot stress LD's
slope-moment closure. The per-cell linear-exactness oracle (gate 1) IS
structurally independent and non-tautological (sign-flip breaks it by 1.88 vs
1e-12 tol -- verified), so LD is not unverified -- but the L1 MMS leg ran on
the weak ansatz. Cross-check the shipped test's case-builder against the
test-architect's ansatz spec; a mismatch is a flag even when all tests pass.

---

## L-020 -- "w=½ generic ops are byte-identical to DD's factored form" is TRUE (verify the IEEE micro-fact, not the docstring)

A coefficient-model refactor that replaces DD's factored closures with generic
affine ops parameterized by w=½ CAN be genuinely byte-identical, because mult-
by-0.5 is an exact power-of-2 scaling: `0.5*(a+b) == 0.5*a + 0.5*b` bit-for-bit
for ALL doubles (verified 2M random pairs, 0 differ) -- the single rounding in
`a+b` equals the single rounding in `0.5a+0.5b` (each summand exactly halved,
exponent-shifted). Likewise `QV*inv/0.5 == 2.0*QV*inv` (0 differ). So
`cell_average=(1-w)in+w*out` and `source_emission=QV*inv/w` at w=½ reproduce
DD's `0.5*(in+out)` / `2*QV*inv` EXACTLY. (#158 Inc B, 2026-06-14.)

**The trap**: the production docstring (`affine_closure.py`) CLAIMED the
opposite -- "principled-equivalent, not byte-identical, ~1 ULP, DD snapshots
re-baseline". That is STALE/WRONG: 0 `.npy` snapshots changed in the working
tree, and the sha256 byte-identity gate (`test_affine_carve_bit_identity.py`,
`si_slab_2g_het`) stayed GREEN. A Cardinal-Rule-3 doc-correctness flag, NOT a
numerics flag. ALWAYS resolve a "byte-identical?" dispute by (a) `git status
--short '**/*.npy'` (re-baseline tell) + (b) the sha256/array_equal gate +
(c) the IEEE micro-fact at the python prompt -- never by the docstring's claim.

**Liveness (L-014 applied to sha256)**: prove a sha256 gate is LIVE before
trusting its green -- monkeypatch the touched op to inject `np.nextafter(out,
inf)` (+1 ULP) and confirm the hash flips. Verified the slab psi-sha flips
under a 1-ULP perturbation of `source_emission`, so the green is real.

---

## L-021 -- Increment B closed the L-018 matvec-coverage gap for LD

L-018 flagged (Inc A) that LD's matvec was correct-but-UNVERIFIED (no committed
test drove `residual_kernel_batch`; SI sweeps only touch the solve kernel).
Inc B added `test_sn_1d_slab_ld_mms_krylov_matches_si` (inner_solver='krylov'),
which an instrumented call-count proves drives `residual_kernel_batch=640`,
`cell_kernel_batch=0` -- the matvec path is now genuinely exercised end-to-end
AND pinned ≡ the SI sweep. When re-reviewing a follow-up increment, re-check
whether it closes a prior increment's NEEDS-EVIDENCE item (the call-count probe
is the verification, not the round-trip self-consistency test).

---

## L-022 -- re-baseline masking-check: re-run the CONVERTED gate, prove the pre-existing red STILL hard-fails ≫ nulp; characterize drift via git-show OLD-vs-NEW .npy

When a commit converts a STRICT `assert_array_equal` snapshot gate to a nULP
`assert_regression(kind="direct")` AND deliberately leaves some baselines
untouched (claiming the untouched ones carry a SEPARATE pre-existing structural
red), the masking failure mode is: the looser gate silently SWALLOWS the real
red. The decisive check is NOT "the suite is green" -- it is re-running the
converted gate on the LEFT-UNTOUCHED arms and confirming they STILL HARD-FAIL
at a magnitude ≫ the nulp bound (#240 SPH bulk: ~1e15 ULP vs nulp=nx=5; the
conversion did not mask them). A re-baseline that silenced a real red would show
those arms flipping green.
Characterize the drift PRINCIPLEDLY (criterion c) by diffing the OLD vs NEW
snapshot bytes directly: `git show <commit>~1:path.npy > /tmp/old.npy` +
`git show <commit>:path.npy > /tmp/new.npy`, then ULP-diff. This shows the
EXACT re-baseline scope (#240: only 3 SLAB `_apply_bulk` keys changed; ALL
`_apply_boundary`, ALL `_solve_*`, ALL curvilinear, ALL 2-D keys byte-identical)
and proves the boundary-byte-identical claim from the binary itself, not prose.
The LIVE-code-vs-REGENERATED-snapshot ULP is necessarily 0 (snapshot was
regenerated at HEAD) -- it does NOT characterize the drift; only OLD-vs-NEW does.
A near-zero cancellation value can show a large ULP count (#240 seed=2: 64 ULP
at |val|=0.024, absΔ=2.22e-16, every other element exactly 1 ULP) -- inspect the
worst element's magnitude before calling it an algorithmic change; large-ULP at
small-magnitude is a ULP-metric artifact, not a non-associativity bound
violation. Criterion 2 (structural-independence) is the load-bearing one: run
the multi-group analytical k∞ recovery (`test_si_carve_recovers_analytical_kinf`
2eg/4eg) + LD MMS O(h²) -- old-vs-new ULP proximity is necessary-not-sufficient.
Cross-ref [[lessons-L020]] (git .npy status + ULP/sha256 over docstring),
[[lessons-L014]] (HARD nulp floor vs STRICT DriftWarning floor; the streaming
boundary gate is `assert_regression` not strict, but is exactly as strict as
`assert_array_equal` under the canonical `-W error::DriftWarning` -- prove it by
running the snapshot class under that flag: #240 = 18 passed / 0 escalations).

---

## L-023 -- convention-relocation re-baseline: scan the WHOLE tree for the OLD-convention literal, not just the diff's touched tests

When a kernel's input contract changes convention (#240: `SNMesh.streaming` /
`s_axes` went from pre-scaled `2|μ|/Δ` to RAW `g=|μ|/Δ`, with the scheme now
applying the diamond `2`), EVERY test that hand-feeds the kernel the OLD literal
is now passing physically-wrong input. The diff author re-baselined the
convention-encoding tests in the SAME directory as the code change
(`tests/sn/sweep/core/`) but MISSED 3 sites in a SIBLING dir
(`tests/sn/spatial/test_linear_discontinuous.py:272/303/340`, all
`s_axes=(2.0*mu/h,)`). 2 of the 3 broke (the geometry-cross-checks
`test_group1_equals_group2_flat` + `test_group3_equals_group2_scan_flat`: one
arm feeds the stale literal to `cell_kernel_batch`, the other derives `g` from
`abs_mu`/`V` correctly → divergence). The 3rd (`test_batched_round_trip`)
SURVIVED because it is a self-consistency round-trip (both arms share the stale
`s_axes`; residual at solved ψ̄ vanishes regardless of convention — the L-018
trap: a round-trip does NOT pin a convention).

**Recipe**: after confirming the touched-file gates pass, `grep -rn` the WHOLE
test tree for the OLD-convention literal (`2.0 *mu/ *h`, `2.0 *np.abs.*/widths`,
`s_axes=.*2\.0\*`, `streaming\(.\).*2\.0\*`). For each hit, classify: a
cross-check against a geometry-derived value WILL break (genuine missed
re-baseline → main goes red); a self-consistency round-trip survives but is now
feeding wrong physics (latent stale-convention test → fix for intent). Prove
the kernel is CORRECT (not buggy) by feeding the NEW literal locally and
confirming the cross-check passes — isolates "stale test input" from "code bug".
Prove the breaks are NEW (not pre-existing) by stash-pop: the 2 broke ONLY with
the diff applied; on clean HEAD they pass. (Caught 2026-06-15 #240 Step A.)

---

## L-024 -- affine-in-σ "value-correct leaf sum" carve: prove teeth bite by DISABLING the override

#240 Step B: `InvertibleOperator.apply` overrides the inherited
`OperatorSum.apply` (leaf sum `L.apply+C.apply`) to single-source σ from C via
`loss_action(self.sigma)`. The matvec is AFFINE in σ in the FORWARD direction
(`M(σ)ψ = streaming_action(ψ) + σ·ψ`), so the leaf sum is value-EQUAL to the
override to ≤2 ULP — a value-correct-by-coincidence twin source, NOT a bug (no
wrong value ever shipped). Verification consequences I confirmed:

1. **Teeth gate MUST be `array_equal` (0 ULP), not allclose.** Only bit-identity
   discriminates leak-vs-override (both are value-equal). PROVE the teeth bite
   by DISABLING the override (rename `apply`→`_DISABLED_apply` so
   `OperatorSum.apply` leaf-sum takes over) → all 7 teeth (4 fwd + 3 transpose)
   FAIL at exactly the predicted ULP (max 1.42e-14 / 7.99e-15 rel = ≤2 ULP).
   Restore byte-exact (sha256 match). This is the L-007/L-022 marker-removal
   masking-check applied to a strict-xfail→pass flip.
2. **NOT a `catches(ERR-NNN)`** — no wrong value shipped → `foundation` gate,
   `verifies(...)` only. The carve says so explicitly and is right.
3. **Migration loud-fail**: a missed caller passing an operator where the σ
   ARRAY is expected → `AttributeError` on `sig_t.shape[0]` / `[None]` (the
   dataclass operator has no `.shape`/`__getitem__`) — NOT a silent wrong-shaped
   array. Confirm by grepping the operator class for `shape`/`__getitem__`.
4. **Structurally-independent ref for a re-baselined Krylov golden = the SI
   golden for the SAME config.** SI rides `solve` (no apply override); Krylov
   rides matvec (override). They agree 3.9e-12 → the NEW Krylov value is CORRECT,
   not merely close-to-OLD (vv criterion 2). The SI golden stays UNCHANGED (apply-
   only blast radius) — that invariance IS the cross-check.
5. **seed0 46-ULP flag = large-ULP@small-mag artifact** (maxabs 3.55e-15 ≡ the
   CYL matvec order; rel ~7e-15). Masking-check (L-022): OLD CYL `.npy` HARD-FAILS
   under NEW code (`46 ULP ≫ nulp=reduction_depth=5`) — proves the re-baseline
   load-bearing; SPH untouched red STILL hard-fails (~1e15 ULP, #195/#209) —
   proves the gate not globally loosened.
6. **WATCH (fragility, not a blocker)**: an `array_equal` slab-apply value-pin is
   SEED-DEPENDENT (seed=7 → 0 ULP, seed=0 → 1 ULP via `TimedFullField.__add__`).
   Passes but brittle; acceptable because the teeth gate owns the structural
   distinction and 1 ULP is FP noise. (Reviewed 2026-06-15 #240 Step B.)

---

## L-025 -- "no missed site" for a dedup-carve: grep WHOLE tree + cross-ref the PLAN, not the closeout

A "route N inlined duplicates through one op" carve's missed-site check is NOT
satisfied by grepping the diff's touched files. Grep the WHOLE module subtree
(`orpheus/sn/`) for the OLD reconstruction literal (`= 2.0*psi... -`, the LD
`psi... + .../d2` form), then classify EACH residual hit as routed /
deferred-by-design / MISSED. Two failure modes a closeout memo can hide:
1. A deferral can belong to a THIRD category the closeout's bucket doesn't name.
   #240 D1 closeout listed only "scan-recurrence" deferrals (β-source `2ψ̄` at
   `loss_representation.py:1435`) but the genuinely-remaining direct DD `2ψ̄−in`
   at `loss_representation.py:2117` (`_OneDimScanWalk._sweep_direction`,
   curvilinear matvec) was NOT in that bucket. It is correctly deferred — but the
   authority is the PLAN (`issue_240_phase2_step_d_homing.md` scoped D1 to
   *Cartesian* inlined `2ψ̄−in`; the curvilinear-angular-fused thread is the NEXT
   campaign), NOT the closeout. ALWAYS read the plan's D-step scope line + `git
   blame` the deferral comment (here `fde76ac5`, pre-D1 → genuine standing
   rationale, not a paper-over).
2. A "missed" site can already be routed ONE LEVEL DEEP: the Cartesian arm of
   the same `_sweep_direction` (line 2056-2083) calls `residual_kernel_batch`,
   which D1 routed — so its reconstruction IS routed transitively. Check the
   call graph, not just the literal.

Verdict rule: a residual OLD-literal hit is a DEFECT only if (a) it is a direct
`ψ_out=reconstruct(ψ̄,in)` (not a scan-recurrence β-source), (b) not transitively
routed, AND (c) in the carve's DECLARED scope per the plan. Fail any of the
three → documented follow-up, not a blocker. (#240 D1, 2026-06-16: exactly ONE
residual direct DD recon tree-wide = line 2117, in-(c)-fail = deferred OK.)

---

## L-026 -- "scattering exercises the slope-source path" does NOT mean the MMS constrains its SIGN (Mode-10 honest-scope)

D5b-S4 = a vv-Mode-7 strengthened 2-D Cartesian LD MMS `ψ=[A+μ_x·B+μ_y·C]/W`
verifying the multi-D bilinear UBLD slope rows landed in S3. VERDICT SUPPORTED-WITH-
CONCERNS; numerics SOUND, no false-green, no blocker.

1. **L11 structural independence GENUINE.** The SymPy source is from the
   CONTINUOUS PDE (no `_LDCellTerms`/`_schur_terms`/`_ubld`). The FD-residual
   cross-check IS a genuine 2nd structural path PROVEN: corrupt `Q_closed`'s
   `μ_x·∂_xA` streaming sign → FD residual 0.047 ≫ 1e-7 tol (FD uses numpy
   central-diff of ψ, RHS embeds SymPy `diff` → a diff sign error breaks the
   equality). Branch2==Branch1 source ≤1e-13. Single-source `_LD2D_STRESS_COEFFS`
   `(num,den)` pairs (Rational∥float) = amplitudes can't drift.
2. **⭐ THE HONEST-SCOPE FINDING (sharpen point 2).** The closeout/docs say the
   slope-SOURCE half is "DEFERRED" because the EXTERNAL Q̂ is zeroed
   (`_lift_external_source_to_moments` → `lifted[...,AVERAGE_MOMENT]`, slopes=0,
   confirmed). BUT the SCATTERING source `Σ_s·φ̂` IS a genuine `(N,ng,nx,2^d)`
   moment source consumed through the SAME `Q_cells` slot as an external Q̂ would
   be (loss_rep:2814-2825 lifts BOTH into the same `Q_per_ord`→`QV_per_ord`; the
   slope-row sign code path `_reframe`+UBLD is source-AGNOSTIC). INSTRUMENTED the
   solve: iterate scalar-flux moments fed to `apply_p0_in_scatter` carry NON-ZERO
   slope rows (avg=1.31, x-slope=0.257, y-slope=0.129, xy=0.067), scattered
   `Σ_s⊗I_spatial` (`fg,fc...->gc...`) → the slope-source rows ARE populated +
   consumed. SO the slope-source CODE PATH is exercised. BUT — DECISIVE MUTATIONS
   on the slope-source rows (`_CellSolve.cell` Q_cells[...,1:]): SIGN-FLIP → order
   stays 1.97, finest in-band → NOT caught; ZERO the rows → order 1.99 NOT caught;
   ×3 magnitude → order 2.02 NOT caught. The scattering-slope source is an
   O(h)-small DG-internal forcing (slopes ~5× < average, c≤1.0) whose sign/
   magnitude affects the converged flux ABOVE O(h²) → absorbed in the floor. So
   the PRECISE honest claim is NEITHER of the brief's two options: it is
   "slope-source code path EXERCISED via scattering but the MMS is BLIND to a
   slope-source SIGN error (a sign flip is not caught) — genuinely UNVERIFIED for
   the sign convention; external-Q̂ plumbing also deferred." The docs' "DEFERRED"
   is substantively CORRECT (sign unverified) but the parenthetical "the only
   moment-valued source consumed is Σ_s·φ̂" UNDER-states that this consumed source
   does NOT verify the sign → CR3 doc-sharpen (follow-up, not blocker): the note
   should say the scattering channel exercises-but-does-not-constrain the
   slope-source sign.
3. **VALUE band REAL + tight (Mode-5 not rate-only).** Reproduced: errs
   [1.42e-2,3.54e-3,8.81e-4], orders [2.00,2.01], maxrelerr 1.78e-2→1.2e-3 (4×/
   halving), flux range matches ref. Band (1e-9,1e-2): upper 1e-2 is BELOW the
   coarsest error 1.42e-2 → a wrong-limit/non-converged flux WOULD exit the band.
   Genuine value gate.
4. **Mutation conclusion SOUND + I isolated what the closeout couldn't.** The
   slope-UNKNOWN half: sign-flip `_GRAD_1D[1,0]:-2→+2` → NaN (caught); a FINITE
   x↔y-symmetric 10% error (`_GRAD_1D` ×0.9) → order −0.06, finest 0.072 ≫ band
   (CAUGHT, NOT divergent — the missing subtle-finite discriminator). So the
   strengthening is non-vacuous + load-bearing for the slope-UNKNOWN half (catches
   both catastrophic AND finite). The strengthening's SPECIFIC x↔y-asymmetry value
   targets the slope-SOURCE same-sign trap — which (per finding 2) this MMS cannot
   reach AT ALL → the x↔y strengthening is defensive-correct-but-currently-
   untestable (ship per spec; its payoff arrives with the moment-source increment).
5. **Gate/marker integrity.** `ld-cartesian-2d` minted (1 unique `:label:`),
   audit `ld-cartesian-2d → 4 tests` exit 0, all verifies targets resolve
   (transport-cartesian-2d/multigroup/mg-balance exist). Quadrature exactness
   CONFIRMED: LS S4 `<μ_x²>=<μ_y²>=1/3`, `<μ_xμ_y>=<μ_x>=<μ_x³>=0`, ZERO pure-z →
   `φ=A` exact. Mode-8 clean (0 bare asserts new files + 0 in S4 prod additions).
   Mode-7 declaration present. ⚠ L-007 NIT: `test_v_ld2d_stress_substitution_
   identity` stacks `@foundation @verifies("ld-cartesian-2d")` = the conflation
   L-007 warns of — BUT (a) established convention (anisotropic_symbolic.py:69-70/
   147-148 do the same for algebra-of-record substitution gates), (b) the label is
   NOT solely foundation-backed (3 genuine L1 verifiers) → the L-007 tell ("ONLY
   coverage is foundation") does NOT bite → minor consistency nit, not blocker.
6. **Modes 1-6 defenses ALL present.** 2G, het (176 unique cell materials,
   spatially-varying σ_t), STRICTLY-asymmetric downscatter (SigS[0→1]=0.233,
   [1→0]=0.000 — pure downscatter, transpose-sensitive), non-square mesh 16×11 +
   domain 1.3×0.9.

VERDICT 2026-06-17 SUPPORTED-WITH-CONCERNS: 1 CR3 doc-sharpen (the scattering
channel exercises-but-doesn't-constrain the slope-source sign — the honest note
under-states this) + 1 L-007 marker nit (convention, non-biting) — both
follow-ups. The shipped scope (slope-UNKNOWN sign verified + average-moment
boundary + matvec twin + two-paths, mutation-verified non-vacuous) is honest. No
false-green, no blocker.

---

## L-027 -- prove a routing-predicate fix's negative-test teeth by REVERTING ONLY production

A "close-the-misroute" change (narrow a strategy `supports()` predicate so a
mesh stops selecting the wrong sweep rep) ships negative tests that assert the
misroute is GONE. Anti-pattern #11 demands the negative test could have FAILED
against the buggy code; for a routing predicate the cheapest proof is:
`git stash push -- <production files only>` (leave the NEW tests in the working
tree), then run the new tests against the reverted-to-pre-fix production. The
negative tests MUST go red AND the red message must NAME the original bug (not
just `AttributeError`); the strategy-free trait probes correctly go red with
`AttributeError: no attribute '<trait>'` (the trait did not exist pre-fix) —
that is the EXPECTED shape, not a flaw. `git stash pop` to restore. (#240
D5-0, 2026-06-16: all 7 new tests RED pre-fix; `test_2d_ld_sweep_raises_not_
silently_dd` red = "did NOT raise — silent return = ran inline DD" = the LIVE
silent-DD hole proven, not asserted.)

Bit-identity claim for a routing-only change: the load-bearing gate is
DESELECT-the-new-tests → pre-existing count UNCHANGED (the predicate touches no
computed flux). A directory-scoped strict gate's TOTAL legitimately grows by
+N_new (new tests live in the gated dir); the invariant is "no PRE-EXISTING
test's value moved", verified by deselection, NOT "total count unchanged"
(#240 D5-0: full 512/1/4, deselect-7 → 505/1/4 = the real proof).

Pyright adjudication for a docstring/ClassVar-only diff: prove ZERO net-new
diagnostics by capturing pyright on the touched files at pre-fix AND post-fix
(stash the production), line-number-STRIP (`sed -E 's/:[0-9]+:[0-9]+//'`), and
diff. Identical-modulo-line-shift = the inserted docstring block shifted every
pre-existing diagnostic by exactly its line count (e.g. +14 from a 14-line
ClassVar block); the diagnostic SET is unchanged → all pre-existing, no
regression. A file that contributes ZERO CLI diagnostics standalone means the
user's "import-unresolved" / "not accessed" items are IDE/cross-tree config
artifacts, not blockers.

---

## L-028 -- the ÷D vs ×(1/D) re-baseline: a "byte-identical" coefficient-model premise fails where the consumer still divides; verify the ORDERING before trusting a snapshot regen

When a fold routes a leftover-inline path onto an established coefficient model,
the fold is byte-identical ONLY where the consumer already uses the
`×inverse_denom` reciprocal form. A consumer still on `÷denom` DIVISION (a
leftover inline path) re-baselines ~1 ULP when it joins the model — division and
reciprocal-then-multiply are NOT IEEE-bit-identical (`2*X/D ≠ 2*X*(1/D)` at 1
ULP; verified `cartesian_scan_coefficients` reproduces the OLD inline alpha/beta
to max abs diff 2.22e-16 = exactly 1 ULP, NOT array_equal). #240 D5a: the spec's
"2-D SOLVE stays byte-identical" premise was WRONG — the pre-D5a
`ScanMarch._sweep_interior` was the ONE remaining `÷D_row` path (the 1-D
CumprodScan already rode `×inv`), so the SOLVE re-baselined too (BOTH
`si_2d_p1_aniso_het` AND `krylov_2d_p1_aniso_het` golden sha moved; the slab/1-D
sha UNCHANGED = the true negative control). The method-implementer's
load-bearing finding (caught a wrong spec premise) was CORRECT — confirm it by
direct algebraic inspection: replicate the OLD inline `alpha = 2*sx2/D - 1`,
`beta = 2*(Q+sy2*psi_y)/D` and the NEW `a, inv, w, (c_y,) =
cartesian_scan_coefficients(...)` + `source_emission(Q + c_y*psi_y, inv, w)` at
controlled input → max rel diff ~2e-16 (`c_y == 2*g_y` byte-exact, `w==0.5`).

**Snapshot-regen ORDERING masking-check** (don't let a re-baseline launder an
EARLIER untracked drift): the load-bearing claim is "pre-fold LIVE matvec ==
FROZEN snapshot at 0 ULP" — i.e. the frozen IS the correct pre-fold reference.
PROVE it: `git stash push -- <PRODUCTION + the regen'd snapshot>` (KEEP the new
test arms) → run the new arms vs the OLD code + OLD frozen snapshot under
`-W error::DriftWarning` → MUST pass at 0 ULP (proves frozen ≡ pre-fold live).
Then OLD-code + NEW(regen) snapshot → MUST hard-fail (proves the regen is real +
the gate is live). #240 D5a: 3 cart2d arms passed strict pre-fold, hard-failed
"6 ULP (max 256)" with the swap = ordering holds. OLD-vs-NEW snapshot ULP-diff:
relΔ ~1.2-3.6e-16 = 1 ULP of the O(1) field (maxabs 1.8-3.6e-15 @ |val|~17-75);
the 256-ULP metric is the L-022 large-ULP@small-mag near-zero-cancellation
artifact. Boundary trace + the `_LpC_` key + ALL non-cart2d (slab/curvilinear)
keys stayed byte-identical (0 ULP) — blast radius = the 2-D row-march ONLY.

**The STRICT-FROZEN docstring stales silently on a re-baseline** (CR3 / L-020):
`test_bc_extraction_2d.py::test_vacuum_bulk_bit_identical` uses
`np.testing.assert_array_equal` (STRICT) with a docstring claiming "must not move
a single bit" / "must stay frozen" / "E0-T1 proved bit-identical to the pre-carve
path". D5a regenerated its `.npy` baselines (relΔ ~1.5e-16) but did NOT touch the
test file → the "must stay frozen" WHY is now FALSE (the output is no longer
bit-identical to the pre-carve path; it is strict-against-the-POST-D5a value).
Gate functions correctly; rationale prose lags. Flag as a doc-correctness nit,
not a blocker. ALWAYS check: a silently-regenerated strict `.npy` baseline whose
CONSUMER test file is untouched → grep the consumer docstring for "frozen" /
"bit-identical to pre-carve" / "must not move a single bit" and flag the stale WHY.

**The two-paths oracle's analytical anchor is TRANSITIVE, not direct** (L14): the
D5a.1 `test_scan_march_equivalence` asserts `ScanMarch.sweep ≡ FullFieldWavefront`
via `assert_allclose` = TWO-PATHS-AGREE. The analytical `k_inf`/`φ=Q/Σ_t` ground
is reached SEPARATELY (`test_keff_2d::TestHomogeneousExact::test_homogeneous_exact`
pins the ScanMarch DEFAULT path to `νΣ_f/Σ_a` ≥2G; the G6 closed-form anchor in
`test_scan_march_end_to_end` runs WINDOW-forced, NOT ScanMarch). A closeout that
says "the oracle pins analytical k_inf=1.875" is LOOSE — the oracle is
transitively pinned; the direct anchor lives in a different file. Confirm the
anchor file is GREEN before crediting the oracle with analytical grounding.

---

## L-029 -- a re-encoded closed form is NOT L11-circular when compared against an INDEPENDENTLY-ASSEMBLED primitive (the d=1-reduction-to-production oracle)

The circularity principle: a token-for-token copy of a production formula is
circular as a VALUE check (a sign-flip in prod propagates into the test, stays
green). The DISTINCT case here (the SymPy UBLD): the test's RIGHT side re-encodes the
production `_schur_terms` S/eff_source/.slope (test lines 346-351, verbatim of
`linear_discontinuous.py:332-335,258-259`) — BUT the LEFT side is the symbolic
primitive's d=1 reduction obtained by `A⁻¹R` of a SEPARATELY-built Kronecker
matrix (`assemble_ubld([h],[mu],...)` → `LUsolve`), NOT a re-statement of
`_schur_terms`. So `diff_psi_bar==0` proves "the production Schur scalar EQUALS
the independently-assembled 2×2 solve" — one side is genuinely structurally
independent. A sign-flip in `_schur_terms` would NOT propagate into the Kronecker
assembly → the oracle WOULD catch it. The circularity test is: **does the bug
live on BOTH sides of the diff?** Re-encoded-formula-vs-re-encoded-formula =
circular; re-encoded-formula-vs-independent-construction = legitimate.

The genuinely-independent anchor in the same gate is `test_d1_symbolic_primitive_
matches_production_update`: evaluates the symbolic d=1 ψ̄/ψ_out at concrete 2-group
het numbers and asserts `LinearDiscontinuous().update` (the LIVE running algebra,
not a copied formula) reproduces them ≤1e-12 — closes the loop to production.

Sub-claims to ALSO check on a "the production X-view equals the d=1 reduction"
oracle: `diff_face` (`psi_out` vs `downstream_face_trace`) is BOTH sides the SAME
solve vector → a trace-operator-consistency check, NOT a structural-indep value
check (fine, it's a foundation closure-consistency claim, not a value claim — do
not credit it as independence). The ÷V `_kernel_terms` and ×V `affine_scan_
coefficients` views BOTH reduce to the same independently-assembled LEFT (verified
by transcribing prod lines 443-453 / 564-571 myself → diff 0); `a_source_indep`
= `Qbar not in a.free_symbols` is a real structural property of the transmission.

**Fast mutation-probe for a symbolically-SLOW gate**: when a foundation gate's
`sp.simplify(diff)` is pathologically slow on the MUTATED (garbage) expression
(#240 D5b: d2 exact-on-bilinear `simplify` of the |μ_axis|-dropped residual hit
the 400s pytest timeout — a `simplify` perf artifact, NOT gate evidence), DON'T
wait on the full pytest. Apply the mutation, then call the `derive_*` builder with
the params as CONCRETE RATIONALS from the start (no symbolic LUsolve blowup) and
read `diff` at concrete numbers: #240 mutated d2 residual = [0.596,-0.396,0.179,
-0.226] (manifestly non-zero on all 4 moments) → `is_zero_matrix` False →
`_require_zero_matrix`→`pytest.fail` → test FAILS. That is DECISIVE and seconds-
fast; the slow `simplify` only confirms the same non-zero. The d1 tests staying
GREEN under the mutation (proven: `test_d1_symbolic_primitive_matches_production_
update` passed 1.45s under mutation — d1 routes inflow inline via
`mu*fin_trace_weight()*psi_in`, never through `assemble_inflow_axis`) IS the
"d=1 oracle is blind to a per-axis factor" evidence (ERR-060 H2 multi-D analog).
ALWAYS revert the mutation (both return branches here) + re-run green (6 passed)
before closing. (Reviewed 2026-06-16 #240 D5b-S1 Branch 1 — VERDICT: all 6 claims
SUPPORTED.)

---

## L-030 -- Intrinsic-property gates: a PER-CELL invariant tested only with a SPATIALLY-UNIFORM fixture is untested for its per-cell-ness

`SpectrumField.__post_init__` enforces the simplex PER CELL (`values.sum(axis=0)` then
`np.allclose(col_sums,1)`), but every test fixture (`_chi` helper) builds a per-cell-UNIFORM
χ — so "per-cell sum==1" and "global mean==1" are INDISTINGUISHABLE in the suite. The code
is correct (probed: a χ summing to 1 in cell0 / 1.2 in cell1 IS rejected) but the per-cell
distinction is uncovered. **Rule**: when a validator's invariant is keyed on a specific AXIS
(per-cell, per-group, per-ordinate), at least one negative fixture MUST VARY along the
non-reduced axis so a global-collapse mis-implementation (`values.sum()` vs `.sum(axis=0)`)
would be caught. Uniform fixtures are blind to axis-collapse bugs — the spatial analogue of
the 1-group degeneracy (H1).

**Validator Mode-11 recipe for a `__post_init__`/`replace`-revalidating type** (verified live
on #257): (a) the `+`-routes-through-revalidation claim — a leaf that does NOT mix in `FluxRole`
inherits `Field.__add__` → `replace()` → re-runs `__post_init__`; PROVE by tracing
`__post_init__` call-count during `χ+χ` (fires ≥1, then raises simplex). (b) a negative test
that pins ONE branch in isolation (neg-entry with col_sum==1.0 exactly) genuinely isolates it
ONLY IF the OTHER branch (sum) would PASS that fixture AND runs SECOND — read `__post_init__`
ordering (neg-check before sum-check) + confirm the raised message names the intended branch,
not the masking one. (c) `mix`/convex-blend re-validation rides the same `replace` path.

**Doc-attribution drift (CR3, no phantom edge)**: `spectrum_field.py:32` frames a χ-drift /
depletion-feedback bug as "the ERR-039 normalization-bug class" — but ERR-039 is the moment-
projection `apply_transpose` (2ℓ+1) addition-theorem factor (nothing to do with χ or depletion).
`units.py:100` "ERR-039 normalization class" for a missing-`/4π` is a looser-but-OK framing
(ERR-039 IS an angular-norm factor). No `catches("ERR-039")` marker exists on these tests, so
no phantom coverage edge is written (contrast L-007) — pure stale-doc, flag don't block.
RULE: a prose "the ERR-NNN class" citation in a NEW docstring still warrants a 30-sec catalog
read; mis-citation in prose is a nit, but the same string in a `catches()` marker is a defect.

---

## L-031 -- a self-consistency round-trip + an A==A pin can BOTH be blind to the bug their docstring/marker claims; the genuine catcher is the continuous-PDE exact-on-bilinear oracle; and "matvec twin verified" ≠ end-to-end Krylov

D5b-S2 wires the d≥2 bilinear LD kernel onto the wavefront (`cell_kernel_batch`/
`residual_kernel_batch` route `len(s_axes)≥2` → `_ubld_system`+`per_cell_solve`;
`_ubld_outgoing_faces` sums the o_a=0,1 Kronecker blocks; faces in `2^{d-1}`
transverse order). DD/Step held byte-identical (513-strict gate UNCHANGED ==
S1 baseline; `tail=() if n_face_moments==1` → no length-1 axis appended; gate is
DIMENSION `len(s_axes)>1` AND trait `spatial_basis_per_axis>1`). Moment-ordering
out-face↔inflow consistency VERIFIED by hand (`_ubld_outgoing_faces` == manual
Kronecker trace; inflow consumer accepts same `2^{d-1}` object per axis, x↔y
non-square). d=1 closed form `==` dense `per_cell_solve` reduction (L29 anchor holds).

1. **MISATTRIBUTED `catches(ERR-NNN)` — the decisive finding.** The NEW
   `test_d2_assembled_matrices_match_symbolic` (entry-wise A/M/G/F_out numpy↔SymPy
   pin) carries `@catches("ERR-060")` but is BLIND to ERR-060: ERR-060 was the
   dropped `|μ_axis|` factor in `assemble_inflow_axis` (the INFLOW assembly), and
   the A==A pin checks `assemble_ubld` (the CELL matrices — A/M/G/F_out contain NO
   inflow factor). MUTATION-PROVEN: drop `|μ_axis|` in `_ubld.py:254` (`return out`
   instead of `mu_axis[...,None]*out`) → of the 3 `catches(ERR-060)` tests, ONLY
   the numpy `test_d2_exact_on_bilinear` fails; the A==A pin PASSES (matrix count
   2→3 inflated by a non-catcher). The pin IS a legit Mode-3 structural pin (a
   dropped streaming factor in G IS caught — verified by a hand-built buggy-G), but
   it does NOT catch the specific bug its marker claims. FIX = drop the
   `catches("ERR-060")` marker (keep `foundation` + the docstring's "catches a
   dropped/mis-scaled factor in the Kronecker assembly" claim, which is true);
   follow-up nit, NOT a blocker (the genuine catcher is correctly marked). RULE:
   for any `catches(ERR-NNN)` on a NEW test, MUTATE the exact documented bug and
   confirm THIS test (not just SOME test) goes red — a marker is a coverage CLAIM,
   L-007 applied to `catches`.
2. **Round-trip (D5b.1) is self-consistent → blind to inflow bugs (L-018/L-023
   reconfirmed in d≥2).** `test_residual_zero_at_solved_cell_avg_2d` PASSED under
   the |μ_axis| mutation (solve+apply share `_ubld_inflow` → the dropped factor
   cancels). It correctly feeds the FULL `psi_bar=psi_avg` (4 moments, not partial)
   + NON-flat per-axis `psi_in` (slope active) + ng∈{1,2} het → it pins solve≡apply
   SAME-system + the matvec-twin face reconstruction (both axes, x↔y-asymmetric
   replay: residual ~2e-16, out_x==rout_x/out_y==rout_y, matvec non-trivial at a
   different probe). But the VALUE-correctness of the multi-D kernel rests on
   `test_d2_exact_on_bilinear` (the L11/L-029-clean continuous-PDE oracle: source
   `Q=Ω·∇ψ+Σ_tψ` from a known bilinear ψ via SymPy, asserts solved moments == exact
   Legendre projections, cross-moment xy active d=0.9 — a kernel sign-flip would NOT
   propagate into the SymPy projections) + the MMS smoke. Sound: the indep oracle IS
   committed and IS the genuine ERR-060 catcher.
3. **"Matvec twin verified" is KERNEL-LEVEL only; end-to-end Krylov RAISES
   (deferred, honest).** Brief/spec D5b.4 wanted "Krylov≡SI on the 2-D LD path",
   but `_CellResidual.cell` RAISES `NotImplementedError` for d≥2 LD (the matvec walk
   needs the `2^d`-moment spatial iterate = S3). PROVEN: 2-D LD SI solve works
   end-to-end (finite flux); `inner_solver="krylov"` RAISES loudly (deferred). This
   is the CORRECT interim per L-017 (loud raise, NOT silent-wrong) — the kernel-level
   `residual_kernel_batch` matvec IS verified (round-trip + asymmetric replay), and
   the raise blocks an accidental wrong Krylov answer. L14 leg-3 (matvec≡sweep
   END-TO-END) is genuinely deferred; the kernel twin is the strongest claim S2 can
   make. NOT a blocker (loud-fail interim); the spec's D5b.4 "Krylov≡SI" wording
   over-reaches S2's actual scope — the SHIPPED scope (kernel twin + raise) is honest
   and the test file documents it.
4. **Smoke MMS honestly scoped (Brief Q2 = SUPPORTED).** `test_ld_2d_converges_
   second_order_smoke` is `@l1` NO `@verifies`, checks BOTH rate (`orders[-1]>1.85,
   all>1.7`) AND a value band (`1e-8<err[-1]<1e-2`) → NOT a Mode-5 rate-only false
   green. The absence of `verifies("ld-cartesian-2d")` is CORRECT: the isotropic
   sin·sin ansatz (`build_2d_cartesian_heterogeneous_mms_case`) under-stresses the
   bilinear slope (Mode-7) + S2 threads only the average source moment (Q̂=0); the
   real flux-shape claim (strengthened μ-non-trivial ansatz + Q̂≠0 moment source +
   non-vanishing boundary) is deferred to S4 — a SOUND deferral, NOT a hole, because
   the kernel value-correctness is independently pinned by the exact-on-bilinear
   oracle (the slope IS exercised there, cross-moment active). `ld-cartesian-2d`
   correctly NOT minted (S4/D6). DD≠LD routing-flip (D5b.5) is 1G but legit (a
   discrimination contract, not an eigenvalue claim; DD≠LD is structural).
5. **Pyright (Brief Q6): exactly ONE net-new diagnostic** (pre/post stash-diff,
   rule-level JSON, 51→52). `linear_discontinuous.py:556` `_ubld_inflow` returns
   `np.ndarray|None` (`R=None` seeded, accumulated in a `for a in range(d)` loop
   pyright can't prove non-empty). REAL but benign type nit (d≥1 ⇒ loop always runs
   ⇒ never None at return; SI solve confirmed live). Pattern-3 nit, follow-up: seed
   `R` with the first term. The other 51 (`AngularSourceSink`/`PoleAngularClosure`/
   `sweep_graph` None-subscript) are ALL pre-existing/rooting-noise. Mode-8 (Q7):
   clean — the 2 prod bare asserts in touched files are PRE-EXISTING (scheme.py:499
   is in a DOCSTRING; loss_rep:2238 pre-existing type-narrow); new tests use
   np.testing/pytest.fail; test-file bare asserts are in `tests/` (rewritten, L-010).
   VERDICT 2026-06-16: SUPPORTED-WITH-CONCERNS — all numerics sound; 1 marker
   misattribution (follow-up) + 1 pyright nit (follow-up) + spec-vs-shipped scope
   wording on D5b.4 (honest as shipped). NO false-green, NO blocker.

---

## L-032 -- "construct-general-only" capability-addition: the byte-id gate needs teeth on BOTH the auto-select AND the phantom-length-1-axis mistake

#240 D5b-S3-A0 minted a typed `SpatialMomentSpace` factor + an OPTIONAL
`spatial_moments: int = 1` param on the flux/source field-space factories
(`AngularField`/`ScalarField`/`HarmonicMomentField`), DEFAULT-OFF. The load-
bearing claim = byte-identical capability addition (DD/Step/LD all unchanged in
a live solve, no production field carries the axis yet). VERDICT: SUPPORTED (all
7 brief questions). The transferable verification pattern for a
construct-general / select-narrow capability addition:

1. **Two DISTINCT teeth-proofs, not one.** A "capability default-OFF" gate can
   leak two ways: (a) the factory AUTO-SELECTS the wider shape (auto-reads
   `mesh.scheme.spatial_basis_per_axis` → LD silently widens), (b) the gate
   appends a PHANTOM length-1 axis at default (re-associates a downstream
   reduction even though "nothing widened"). PROVE BOTH bite:
   - (a) MUTATION: force the helper to auto-read the scheme → ONLY the `[ld]`
     byte-id arms red (DD scheme reads 1, stays green) = the gate discriminates
     auto-select. (#240: `test_*_default_byte_identical_all_schemes[ld]` ×2 red.)
   - (b) MUTATION: make the "append iff >1" policy append at n==1 too
     (`return (n,)` instead of `() if n==1 else (n,)`) → byte-id arms red for
     ALL schemes (DD+LD) = the gate discriminates a phantom axis. (#240: 9 red.)
   The negative-control assertion that catches (b) is `not hasattr(field.space,
   "factors")` — a default field must be a BARE `FunctionSpace`, not a length-1
   `TensorProductSpace`. Independently confirm DD `(24,2,3,4)` == LD `(24,2,3,4)`
   at default DESPITE LD's `spatial_basis_per_axis==2` (the construct-general
   proof: the scheme SAYS 2 but the factory does not read it).

2. **The "append iff >1" policy MUST be single-sourced** (Pattern 7). #240's
   `spatial_moment_tail` delegates to `_ubld.face_moment_tail` (`() if n==1 else
   (n,)`); the cell-tail and face-cochain-tail can never disagree. Verify the
   delegation by reading both + the `AVERAGE_MOMENT=0` constant the space's
   `average_moment_index` surfaces (NOT a re-spelled `0`).

3. **Einsum "spectator-broadcast" lift (`fc->gc` ⇒ `fc...->gc...`) is provably
   byte-identical at rank-2-exact** AND correct as `Σ⊗I_spatial`: at the python
   prompt, `np.array_equal(einsum('fg,fc->gc',...), einsum('fg,fc...->gc...',...))`
   on rank-2 input = True (the `...` matches nothing → no axis); on rank-3 input,
   `einsum('fg,fc...->gc...')` == per-moment-independent stack (each spatial
   moment scattered independently). The IEEE micro-fact resolves the byte-id
   dispute, NOT the docstring (L-020). (#240: all 3 `material_xs_field.py`
   einsum lifts — apply_p0/apply_n2n/legendre_moments — array_equal at DD/Step.)

4. **The strict gate baseline is 513 (not 562).** The S3 crosswalk's "562/2skip"
   was a STALE-PLAN figure; the live S2/S3 baseline is 513P/1skip/4xf under
   `-W error::DriftWarning` (matches L-031). Re-confirmed; no golden moved
   (`git status --short '**/*.npy'` = empty). When a closeout and a plan
   disagree on a count, RUN the gate — the closeout's 513 was right.

5. **Adding a dataclass FIELD (not just a space factor) to a Flux leaf ripples
   to its Displacement sibling.** `FluxRole._mint_displacement` (`φ⊖φ`) copies
   EVERY init field → `MomentDisplacement` needed the same `spatial_moments`
   field or `φ⊖φ` raises TypeError. Verify the affine round-trip (`φ⊖φ→disp`,
   `φ+disp→φ` array_equal) at BOTH default AND widened. (#240: both exact.)

6. **Pyright net-new = 0** proven apples-to-apples (L-027): run the SAME
   5 touched files PRE (stash prod + hold the new untracked file) and POST,
   path+line-strip, `comm -23`. All 8 errors pre-existing (`DualSpace.of`
   return, `MaterialXSField` Optional, `from_face_arrays` Optional layout,
   `_check_partner` `other.L` on object); the NEW file alone = 0 errors; the new
   `find_factor` RAISES KeyError (not returns None) → type-clean. The brief's
   worry about "find_factor-returns-object at space.py:521" was a brief
   mis-attribution: :521 is `DualSpace.of`, pre-existing. (VERDICT 2026-06-16:
   SUPPORTED, no blocker, no follow-up.)

---

## L-033 -- a GENUINE structural-independence ground (L11 clean) can still ship a LATENT twin-path crash no committed gate drives (the d≥2 matvec pure_z gap)

D5b-S3 = the unified all-d LD moment matvec + ERR-061 frame fix (slope ψ̂_n
stored in per-ordinate SWEEP frame, summed by consumer as GLOBAL → backward
ordinates CANCEL the forward slope → φ̂ 6× under-driven → diffusion limit lost;
fixed by `octant_moment_frame_signs` = ∏_a sign_a^{o_a} involution via `_reframe`).

1. **The headline correctness claim is REAL (L11 clean, NOT L4).** The
   from-scratch LM-1989 solver (`_independent_ld_slab` in
   `tests/sn/spatial/test_ld_slope_frame.py`) is GENUINELY structurally
   independent: hand-built cell 2×2 `[[σh+μ,μ],[-μ/θ,σh+μ/θ]]`, hand SI, NO
   ORPHEUS kernel. Verified live: sweep-frame=1.4717 (== ORPHEUS pre-fix
   bit-for-bit), global-frame=2.3080 (rel 2.3% vs ANALYTICAL diffusion 2.362).
   Anchor is the closed-form diffusion VALUE (`@foundation`
   `test_independent_ld_global_frame_recovers_diffusion`), NOT "LD≈DD". The
   production LD value 2.30798 == the independent solver's global-frame value 4dp
   AND matches analytical diffusion (2.3%) BETTER than it matches DD (4.1%) → the
   chain prod-LD≡indep-LD≡analytical closes. The frame primitive verified at the
   prompt: ∏ closed form exact, DD→None, genuine involution (s·s=1 ∀octant), d=2
   x̂y flips iff ODD # axes reverse.
2. **`catches("ERR-061")` markers MUTATION-VERIFIED.** Neuter `_reframe`
   (`return arr` — single mutation, `loss_representation.py:147` imports the SAME
   helper so ALL paths hit) → the 3 prod-path catchers (slope-frame consistency,
   thick-diffusion 1G, thick-diffusion 2G) ALL go RED with the ERR-061 mechanism
   in the err_msg; the `@foundation` independent ground STAYS GREEN by design (no
   `catches`, doesn't ride `_reframe`). Correct per anti-pattern #11 / L-007.
   (catalog entry lists only the 2 thick tests; the slope-frame test ALSO carries
   `catches("ERR-061")` — minor catalog omission, not wrong.)
3. **Mode-7-at-primitive resolved** (the brief's load-bearing concern): the S1
   `assemble_ubld` exact-on-bilinear oracle nulled the diffusion slope; the NEW
   thick-cell tripwire (σ_t·h=10, c=0.99, COARSE nx=4 — NOT refined mesh, the
   L-017 thick-cell probe) + the 2G-het Mode-6 companion (asym SigS
   [[30,9.6],[0,39.6]], both groups recover, g0→g1=9.6/g1→g0=0 → transpose-
   sensitive) genuinely exercise the regime. ERR-061 catalog entry COMPLETE
   (mechanism + fix + #1+#6 classification + how-it-hid + lesson + bug-signature
   x-link).
4. **DD/Step byte-identity = TRUE** (negative control): GATE 4 513/1/4 under
   `-W error::DriftWarning` IDENTICAL pre/post; `git status --short '**/*.npy'`
   EMPTY (no golden moved). `face_moment_tail(1)==()` + `octant_moment_frame_signs(_,1)==None`
   → DD never grows the moment axis. The 7 spatial+operators reds are GENUINELY
   pre-existing (git-stash: at clean HEAD the same 7 + 2 fix-dependent tests fail;
   with diff exactly 7; the 7 = sphere matvec ×5 + 2-D mu_y BC ×2, none touch the
   slab scan / moment frame).

⚠ **THE CONCERN (the d≥2 matvec twin-path gap — a real follow-up).** "matvec≡sweep
for BOTH d=1 and d≥2" is NOT fully verified. d=1 has committed gates (scan≡DAG
+ Krylov≡SI). d≥2 has NO committed end-to-end Krylov≡SI gate (grep krylov in the
d≥2 MMS files = EMPTY; the closeout's "rel 4.99e-11" was an UNCOMMITTED manual
smoke). The d≥2 raise WAS retired (`_CellResidual.cell` comment "d≥2 raise is
RETIRED") so the path is live — but RUNNING it on the MMS case quad (N=110, **2
pure-z ordinates**) CRASHES: `loss_representation.py:742` matvec `pure_z` does
`LpC[oct_idx]=sigma*probe[oct_idx]` with NO moment-axis broadcast guard → ValueError
`(2,6,6) vs (1,2,6,6,4)`. The SWEEP `pure_z` (line 654-655) HAS the guard
(`if q.ndim>sig.ndim+1: sig=sig[...,None]`); the matvec twin does NOT — a Pattern-2
twin-path asymmetry. Latent because the d≥2 verification uses SI (smoke) + sweep-vs-
sweep two-paths on level_symmetric (ZERO pure-z); the matvec `pure_z`+moments combo
is untested. LOUD crash (ValueError), NOT silent-wrong → no false-green ships, but the
closeout's "2-D Krylov works end-to-end" is only true for no-pure-z quads. Fix = port
the sweep `pure_z` moment-broadcast guard to the matvec `pure_z` + add a committed 2-D
LD Krylov≡SI gate on a quad WITH pure-z (the L-018/L-021 "matvec needs a committed
call-count/end-to-end gate, not a round-trip" lesson, recurring a THIRD time).

OTHER follow-ups (non-blocking): (a) 4 net-new pyright nits (apples-to-apples
stash PRE 110 / POST 113 + comm: `D1ClosedForm` un-imported but used in a
`moment_scan_closure` return annotation — runtime-safe via `from __future__`;
`scheme.moment_scan_closure` LD-only method on a base-typed handle behind the
`is_moment` gate; `Q.spatial_moments_per_axis` narrowness; 1 reportReturnType) —
all Pattern-3 type-narrowness debt; (b) `verifies("ld-cartesian-1d","ld-slab")`
labels have NO `:label:` math block in docs/theory (pre-existing — already at HEAD,
audit tracks them with 6/4 tests, exit 0; `ld-cartesian-2d` correctly deferred per
L-031); minted in S4/archivist. Mode-8 clean (2 prod bare asserts in touched files
PRE-EXISTING: loss_rep:2376 type-narrow, solver:866 in `if __debug__:`). VERDICT
2026-06-17: SUPPORTED-WITH-CONCERNS. Numerics + the headline fix SOUND, byte-id real,
markers honest. The d≥2 matvec pure_z crash is a real latent defect (loud, narrow)
+ the missing d≥2 Krylov gate is a genuine coverage hole — both follow-ups, NOT
commit blockers (the SHIPPED scope is SI-verified d≥2 + matvec-verified d=1; no
false-green). No false-green found anywhere.

---

## L-034 -- Stale-snapshot triage: a HUGE-ULP bit-identity red is "live correct, frozen stale" until you find the apply-changing commit that did NOT re-capture

A frozen-snapshot bit-identity gate failing with a catastrophic metric
("not equal to 5 ULP, max is 8.8e15"; or `assert_allclose` 100% mismatched
few-%) on ONE geometry arm while sibling arms (slab/cylinder) PASS is the
fingerprint of **a stale snapshot left by an unrelated correctness fix**, NOT a
live solver bug. The live apply is usually the MORE-correct value; the frozen
reference is stale. Triage procedure (do NOT modify anything — produce verdict):

1. **Confirm the asymmetry** — run the failing arm AND a sibling arm (other
   geometry/seed). Sibling green + this red = geometry-scoped change, not a
   broad regression. (Here SLB matvec passes @1 ULP DriftWarning; cart2d+cyl
   streaming arms pass; only SPH fails → sphere-only-by-construction fix.)
2. **Blob-hash the fixture across refs** — `git rev-parse <ref>:<fixture>`.
   If the snapshot blob is byte-IDENTICAL since the last refresh commit but the
   code moved, the snapshot is the stale side. (SPH `.npy` `501fd29` unchanged
   since the ERR-058 refresh `798372f`; the `.npz` later changed at #240 but
   only its CARTESIAN arrays — sphere arrays rode the stale `798372f` values.)
3. **Find the diverging apply commit** — `git log <refresh>..<base> -- <prod
   apply files>` and grep the messages for the geometry + math-term keyword
   (curvilinear/sphere/clamp/seed/closure/weight). The culprit is a commit that
   (a) changed the geometry's apply value, single-sourced so the matvec
   inherits it, AND (b) touched NONE of the failing suites' fixtures (staleness
   slips in SILENTLY because the commit's own `-O` sweep didn't run them). Here:
   `b2d8a6d` "unclamp spherical Morel–Montry weight (Bailey Eq. 43)" — dropped a
   spurious `[½,1]` τ-clamp in `spherical_streaming`, regenerated only its OWN
   targeted snapshot, left these 2 suites stale.
4. **Verify the call path** — Nexus `callers(spherical_streaming)` →
   `SNMesh._init_core` → `StreamingOperator.apply`: BOTH the matvec test
   (`_LpC_apply`) and the streaming-operator test (`L.apply`) consume the same
   producer, so both inherit the change.
5. **Does the unmerged sibling branch fix it?** Git-archaeology beats a worktree
   run (editable `.venv` resolves to MAIN tree; worktree creation may be denied
   anyway). `merge-base --is-ancestor <fix> <branch>` (does it contain the fix?)
   + blob-hash the fixture on the branch vs base. If fixtures are byte-identical
   AND the branch changes the apply path FURTHER, the branch does NOT fix it —
   it inherits the same stale snapshot and moves the live value further away.
   (#236 forks off the clamp fix, leaves both fixtures byte-id, reworks
   `pole_angular_closure.py` +942 → would still fail, possibly differently.)

**Verdict mapping**: stale snapshot + NO open issue owning it = (B)
NEW/UNTRACKED — recommend a `tests(sn)`+`type:bug` re-baseline issue, NOT a
correctness regression. The fix = re-capture on main, validated against the
STRUCTURALLY-INDEPENDENT grounds (the matvec file's `Q/Σ_t` L0 streaming-
equilibrium row + the curvilinear L1 MMS/closed-form k_∞ the streaming class
cites) per vv bit-id criterion 2 — NOT just old-vs-new ULP. ⚠ A test docstring's
issue attribution can be STALE/wrong: this SPH red's docstring cited "#195/#209"
(both CLOSED, both different mechanisms — ERR-058 MMS-rate + cylindrical-pole
NaN); the REAL cause `b2d8a6d` is Refs #229. Trust the git-archaeology over the
docstring's cited issue number.

---

## L-035 -- "byte-identical EXCEPT a LATENT collision" claim: instrument BOTH branches on the gate suite + classify each divergent site as reached/correct

When a refactor claims "byte-identical to all paths EXCEPT one latent S4-style
collision," do NOT trust the latency claim — PROBE it. Patch the touched primitive
to compute BOTH the OLD and NEW branch on every call and assert array_equal,
running the FULL gate suite (a pytest plugin `pytest_configure` reassigns
`module._symbol` in EVERY importing module — `sg._reframe` AND `lr._reframe`;
attribute via `pytest_runtest_setup` → `item.nodeid`; pytest captures stderr so
read the probe under `-s`). #246's `_reframe` keyed on `is_moment_valued` (typed
origin) vs OLD `arr.shape[-1] != frame_signs.shape[0]` (size probe) — claimed the
d=2 `2^d==4` collision was "latent in production." The probe found **48
divergences, NEW≠OLD by 70%** (NOT FP, NOT zero), all at `_CellSolve.cell`'s
`Q_cells` reframe, in exactly TWO tests: `test_ld_2d_two_paths_ffw_equals_mfw` +
its `_stress_` sibling. So the collision is REACHED — not latent — by the
LOW-LEVEL `MovingFrontierWindow/FullFieldWavefront.sweep(Q_flat, ...)` API with a
flat source whose anti-diagonal level has exactly `2^d` cells (`n_diag==4`).

The decisive correctness call (which branch is RIGHT) needs a STRUCTURALLY-INDEPENDENT
reference, NOT the gate (the two-paths gate is Mode-11 BLIND: both FFW+MFW legs
share `_CellSolve`, both corrupted identically under OLD → agree to 0.0 EITHER way,
so it cannot distinguish OLD-wrong from NEW-correct). The independent reference:
moment-LIFT the same flat source onto slot 0 (`face_moment_tail`, slopes=0) and
sweep — a flat source and its zero-slope lift are the SAME physics. Result: NEW
flat-sweep == moment-lifted-sweep BYTE-IDENTICAL (0.0); OLD flat-sweep ≠ lifted by
70% (OLD was inconsistent with its OWN lift — the size-probe mis-classified the
4-cell anti-diagonal as a 4-slot moment vector and applied a spurious `[1,-1,-1,1]`
involution, scrambling cells). ⟹ NEW is CORRECT, OLD was a REACHED (test-path)
silent error.

⚠ The "production" qualifier is load-bearing: `solve_sn_fixed_source` ALWAYS
moment-lifts via `_lift_external_source_to_moments` (source reaches `_CellSolve`
as ndim+1 → rank test True → never the flat collision), so production USERS never
hit it — only the low-level `.sweep()` test API does. So "latent in production
(via the public solver)" is TRUE; "latent everywhere" is FALSE. Verdict =
SUPPORTED (the fix is correct and strictly better than OLD), but flag the framing:
it FIXES a reached test-path silent error, it is not purely prophylactic. The
rank discriminator `Q.ndim > sigt.ndim + 1` is genuinely S4-safe (a rank cannot
collide the way a trailing-size can) AND correctly classifies BOTH entry points
(flat `(N_oct,ng,n_diag)` ndim-3 → False; moment-lifted `(N_oct,ng,n_diag,2^d)`
ndim-4 → True), single-source-shared with `_moment_broadcast_sigma:515`. Gate-1
(`test_reframe_moment_intent.py`) is the genuine unit catcher (mutation-verified:
emulate OLD `_reframe` → Row-1 `out==arr` FAILS, sign-flips `[0,-1,2,-3,...]`);
Gate-3 `is_multi_moment` mutation-verified (const-True reddens DD-P2, const-False
reddens LD-P1). DD byte-id confirmed (regression suite: 0 divergences, all
short-circuit on `frame_signs is None`; the 13 within-tol DriftWarnings PRE-EXIST,
not escalated under `-W error`). 1-D scan + LD-kernel suites: 0 divergences (those
sites only ever reframe genuine moment arrays where size-probe≡intent agree).

---

## L-036 -- "MMS covers the retired term" claim: mutation-verify, but the deleted code may be a SWEEP↔MATVEC twin (mutate the SHARED coefficient source, not the deleted-apply method)

A retirement that deletes "verification machinery" (here the `M_spatial`/`M_angular_redist`
separately-applicable operator-leaf split + `loss_action_decomposed` + the `emit_angular`
arm of `_apply_walk`) defends "no correctness lost" by naming a surviving MMS. The
deleted decomposition tests pinned only STRUCTURE — `TestT4c…` asserts
`(m_full − m_ang) + m_ang == m_full`, a **TAUTOLOGY by the subtraction construction**
(`m_spat = m_cell − m_ang_cell` is literally the deleted code); `TestT4b…` are
`isinstance`/`direction_sign`/`capabilities`/`cached_property` type pins +
`L == M_spatial − σ_t·ψ` (self-referential, both sides one walk); `TestT5…` is
`from_geometry == from_geometry`. None pins an INDEPENDENT correctness invariant of
the production `m_full`. The deleted `m_ang` emission wrote ONLY a separate buffer no
production code read — the fused `m_full = (denom·ψ̄ − numer_upstream)/V` already
carries the redistribution (it lives INSIDE `denom`/`numer_upstream` via the closure's
`(ΔA/w)·c_out` / `(ΔA/w)·c_in·ψ`), so the production output is byte-identical (strict DD
regression passed at the SAME documented 6920-ULP baseline, 0 `.npy`/`.npz` regenerated,
607-del/48-ins).

⭐ THE TRAP when mutation-verifying the surviving MMS: the deleted code was in the
MATVEC/apply path (`_apply_walk`), but the MMS runs `solve_sn_fixed_source` = the
SWEEP/solve path. These are genuine TWINS that share only the precomputed coefficients.
My first 3 mutations (of `MorelMontryAngularSweep.cell_contribution`, then
`_redistribution_for_level`/`__call__`, then `cell_balance_terms`) ALL showed
call-count 0 on the sphere solve and byte-identical error ladders (GREEN-BLIND =
patching dead-for-this-path code). The sphere sweep is the `ScanMarch`/`CumprodScan`
1-D scan in `loss_representation.py:3106` (`ang_contrib = dA_w·c_in·ψ` into the source
`b` + `c_out` baked into `inverse_denom`); the cylinder routes through
`diamond.update`→`cell_balance_terms`. THE RIGHT MUTATION POINT = the SINGLE SHARED
source `GeometryCoefficients.from_mesh_and_quad` (`c_out=α_out/τ`, `c_in=(1−τ)/τ·α_out+α_in`,
sweep_cache.py:309-310) — `dataclasses.replace(gc, c_in=f·gc.c_in, c_out=f·gc.c_out)`.
Confirm the factory call-count > 0 + identity-reimpl (f=1) reproduces baseline FIRST,
THEN mutate. Result: c_in/c_out sign-flip, ×3, even ×1.5 → BOTH sphere AND cylinder MMS
go NaN or land orders of magnitude outside the gate bands (1e-8..5e-3 / 1e-3..5e-2).
The redistribution is an O(1) term in the curvilinear cell balance — the MMS STRONGLY
constrains it end-to-end. Verdict: SUPPORTED, no correctness coverage lost.

PROCEDURE for "deleted machinery covered by surviving test X": (1) read each deleted
test → classify STRUCTURE-pin vs CORRECTNESS-pin (tautology-by-construction = structure);
(2) confirm the production value is byte-identical (strict gate at documented baseline +
0 snapshot regen + diff is deletion-dominated); (3) mutation-verify X catches the term —
but FIRST instrument a call-count to find which of the SWEEP/MATVEC twins X exercises and
mutate the SHARED coefficient source, not the deleted-apply method (else GREEN-BLIND on
dead-for-this-path code mis-reads as "X is blind"). Mode-8 caveat: the MMS uses bare
`assert` under `-O`, but pytest's rewriter fires asserts in `tests/` modules (L-010) —
proved live by breaking a band to red. Baseline reds (5 stale SPH snapshots #250 + 2 mu_y
#232) are pre-existing: cylinder snapshot siblings walk the SAME modified curvilinear
`_apply_walk` and PASS; only SPH fails at ~1e15 ULP (stale-snapshot signature, L-034),
not the 1-ULP FP drift a real regression would show.

---

## L-037 -- Mode-10 closeout verification recipe (the activated-but-unconstrained slope source)

#247 Leg A closed the slope-SOURCE half of the LM-1989 trap for 2-D Cartesian LD
(the vv Mode-10 gap: a term genuinely CONSUMED yet a sign flip leaves the
converged flux at O(h²) + ~1.4×, sub-floor). VERDICT SUPPORTED, NO ERR (the
slope source was UNVERIFIED, not WRONG — the production lift correctly zeroed an
honest default q̂=0). Reusable recipe for adjudicating a Mode-10 closeout:

1. **The teeth are NOT the converged-flux value-band** (the §0 trap — the slope
   error is O(h²)-small). Demand TWO O(1) structural teeth instead: (a) the
   PRODUCER threads the projection through at machine precision
   (`array_equal(lifted, Qm)` — the production-change proof; under the bug, the
   dropped slope is O(1) e.g. 0.179); (b) a CONSUMED source-row sign flip moves
   the converged answer ≫ solver tol (the consumption proof).

2. **Prove the teeth bite by re-introducing the EXACT bug in-process** (throwaway
   conftest plugin monkeypatching the producer to re-zero slopes — NO production
   edit, L28). The sign-mutation gate's red message is the tell:
   `|Δφ|/|φ|=0.000e+00 ≤ tol — the slope row is NOT consumed` (flipping a
   re-zeroed row is a no-op). Confirm the 3 new gates RED.

3. **Prove the Mode-10 ASYMMETRY**: run the EXISTING flat scalar gate under the
   SAME buggy producer → it MUST stay GREEN (it feeds a flat source → slope row
   already zero → blind to the slope sign). GREEN-flat + RED-moment IS the gap.

4. **Calibrate the consumption-tol live**: a deterministic SI solve has noise
   floor EXACTLY 0.0 on an identical re-solve (measure it). Smallest signal
   (xy slot) ~5.8e-5 clears 1e-8 by ~5.8e3× → no false-green. The tol is
   defensible iff (noise ≪ tol ≪ smallest signal).

5. **L11 check the projector source**: `leggauss` + numpy + hand-laid algebra
   ONLY. Typed source CONTAINERS (`AngularSourceSink`/`TimedFullField`) imported
   to FEED the solve do NOT contaminate the reference — the LD cell op / the lift
   must not be called. The foundation sub-gate's reference is hand-derived
   polynomial coefficients, not a production echo.

6. **Confirm no latent CONSUMPTION bug** (else mint ERR): the now-consumed slope
   path must have no sign/magnitude error. The architecture is the proof when
   the producer change RIDES an EXISTING consumer path (external + scattering
   moment vectors SUMMED into ONE global-frame array → shared rank-gated
   involution reframe `octant_moment_frame_signs`, shared mass M=diag(h,θh),
   shared Kronecker order). Dispatch explorer to trace reframe/mass/Kronecker;
   if no separate external-vs-scattering branch (no extra/missing flip, no
   transpose), the consumed path is correct.

WATCH (non-blocking doc-nits this review surfaced): a "single-source shared by
fixed-source AND eigenvalue" docstring can be STALE (grep the lift's callers —
#247's lift has ONE prod caller; the eq path wraps its sweep OUTPUT, doesn't
call the lift); a d=2 cell-unknown prose label can transpose x↔y vs the
canonical [bar,y,x,xy] (axis0=x OUTER) — prose only, slots come from
moment_layout, no code path. Both CR3/stale-doc (L-020/L-028), no ERR.

---

## L-038 -- prove an xfail→live FLIP is non-vacuous TWO ways + Mode-10 with NO dominant regime (boundary transverse face-slope)

#251 widened the 2-D Cartesian LD boundary trace to CARRY the `2^{d-1}` transverse
face-slope (boundary twin of #247 Leg A's bulk slope). The boundary slot grows a
trailing moment axis at ONE lever (`geometry.boundary_face_layout` appends
`face_moment_tail(per_axis**(ndim-1))`); `_inflow_to_moments` rank-discriminates
(`is_moment_valued_by_flat_rank(face, mesh.ndim+1)`) → scalar arm seeds slot-0 only,
moment arm passes through; 4 outflow capture-collapse sites dropped so the outflow
moments land in the now-moment-shaped slot. VERDICT SUPPORTED, NO ERR, NO blocker.

1. **An xfail→live FLIP needs TWO red-proofs, not one.** A gate the closeout says
   "was xfail-strict, now passes via production" is only non-vacuous if it (a) goes
   RED against a re-introduced post-change bug AND (b) goes RED against the EMULATED
   PRE-change behavior. The re-zero mutation (`f[...,1:]=0` in the NEW moment arm)
   proves the consumed slope is constrained; but EMULATING the old unconditional
   zero-fill (treat the `(...,2)` moment face as scalar → spurious `(...,2,2)` axis +
   slot-1 zero) is what proves the gate genuinely REQUIRES the #251 change (threading
   gate red "did not RECOGNISE the moment-resolved inflow"; width-reject red "DID NOT
   RAISE"). Only (b) rules out a gate that was already green at HEAD. Do BOTH via a
   throwaway conftest plugin under `-O` (L28: no prod edit, no git stash).
2. **Mode-11 closure for a public-solve gate = INSTRUMENT the rewired arm + count.**
   When a closeout says "the surrogate monkeypatch was dropped and the gate re-targeted
   onto the public API," confirm the rewired production line is on the LIVE call graph:
   monkeypatch the method to COUNT which arm fires during the public solve. #251:
   `_inflow_to_moments` fired 344×, moment-resolved arm 688× (0 scalar/identity) on the
   public `+slope` `solve_sn_fixed_source` → Mode-11 CLOSED (the gate drives production,
   not a recompute-on-both-sides surrogate). The consumption gate's RED under the re-zero
   mutation is THROUGH that public path (|Δφ|/|φ|=0.000e+00 — flipping a zeroed slope is
   a no-op = the exact Mode-10 signature).
3. **Mode-10 with NO O(1)-dominant regime → structural teeth are the COMPLETE
   resolution (no value-improvement leg).** A boundary-trace slope is sub-floor for ANY
   value claim, not just the sign (probed: seeding the REAL slope makes near-bdy A-err
   WORSE 2.131e-2→2.163e-2, flipped is BETTER). So "improves-on-flat" is UNACHIEVABLE
   and dropping it is HONEST, not hiding a problem (keeping it would falsely RED a
   correctly-consumed slope). The companion-gate half of the Mode-10 recipe (isolate
   the term so its error is O(1)) is UNAVAILABLE — no fixed-source problem makes a
   boundary-trace slope the dominant forcing. Positive verification = TWO O(1)
   structural signals ONLY: machine-precision threading (`array_equal` slot-1, leggauss
   reference = L11, NON-circular b/c prod's arm is a pure pass-through) + consumed-flip
   ≫ TOL (4.101e-3, triple-agrees across my re-derivation / test-architect surrogate /
   public-path, above the deterministic 0.0 noise floor). This is a NEW Mode-10
   sub-case (neither #240 D5b-S4 nor #247 Leg A had a term with no dominant regime —
   both could improve-on-flat) → warrants the test-architect's one-line vv Mode-10 row
   addition.
4. **Reflective storage pass-through ≠ reflective SIGN.** A trace-widening's reflective
   path has TWO concerns: storage (the `PermutationOperator(axis=0)` broadcasts the new
   moment axis — verify NO corruption by seeding a random moment-shaped trace, running
   `_reflect_trace`, and checking slot-1 follows slot-0's permutation: #251 = 0
   corruption over 12 matched ordinates) and SIGN (the transverse-slope sign under a
   normal-flip reflection — UNVERIFIED b/c the vacuum-BC MMS nulls the reflective
   coupling, H2). Storage-correct is shippable; the SIGN is a Mode-1 trap the vacuum
   gates CANNOT see → MUST be a follow-up (#252, filed OPEN with correct labels), NOT a
   blocker. Confirm the follow-up issue actually exists (`gh issue view`) before
   crediting "filed as #NNN."
5. **Producer-rank carve: a widened slot needs the EXISTING SCALAR producers audited,
   not just the new moment one.** When a carve widens a trace/field slot, the existing
   scalar MMS callers feed the SAME widened slot → the producer (`prescribed_inflow`)
   must accept BOTH ranks (scalar→seed slot-0; moment→write full slot). The explorer's
   "1 real producer edit" under-scoped this by 1 (the scalar-onto-moment arm). Same
   class as Leg A's field-space layer: a rigid scalar contract above a widened slot
   needs a typed-union relaxation, not just an indexing fix.

⚠ Minor scope-note (NOT a defect): spec D6 said "DD rejects any moment trace" but
the impl early-returns IDENTITY at `n==1` (DD never receives a moment inflow —
`face_moment_tail(1)==()` makes the DD trace scalar-only), so the shipped reject-gate
only tests LD wrong-width. Correct by construction; flag for the spec author only.

---

## L-039 -- runtime_checkable Protocol gate: prove teeth by DROPPING/adding a member, not by a vacuous isinstance pass

Validating a `@runtime_checkable` structural Protocol as a REAL coverage gate
(vs a vacuous isinstance pass). VERDICT REAL-GATE. The recipe, consolidated
from two reviews (a `Vector` carrier Protocol and a `TransportState` state
Protocol):

(1) **Mode-8 is cleared by the mutation, not by inspection** -- bare asserts
in a COLLECTED test module ARE rewritten by pytest and DO fire under `-O`
(the `PytestConfigWarning` refers to NON-test modules; reaffirms L-010).
Proof: a Protocol mutation made the asserts raise a real `AssertionError`
under `-O`; `pytest.fail` likewise raises `Failed` under `-O` (a print after
it never runs).

(2) **Mutate the PRODUCTION Protocol, not the test object.** Two complementary
moves:
   - **Drop a required member** -- DROP only `__rmul__` -> ONLY
     `test_no_scalar_mul_rejected` reds while `test_string_is_not_vector` STAYS
     GREEN (str still lacks `__sub__`); the asymmetry proves each negative test
     OWNS a specific dunder. Reduce the Protocol to `__add__`-only -> ALL
     negatives red, positives stay green.
   - **Build an in-memory MUTANT Protocol and monkeypatch it in** -- a
     drop-all `class M(Vector, Protocol): ...` flips `np.ndarray` AND the leaf
     type to `isinstance==True`; run the REAL test fn with the production
     name monkeypatched to the mutant (patch BOTH the defining module's name
     AND the test module's import binding) -> the discriminating negative
     (`test_ndarray_..._not_a_transport_state`) fires with the exact message.
   The RED message names the wrongly-accepted object. Revert by RE-EDITING
   (untracked `??` files make `git diff` empty -> vacuous; the real revert
   proof is gate-green-again + grep zero MUTATION markers + dunder/line-count
   match).

(3) **`scalar * vector` is `__rmul__` NOT `__mul__` -- prove with the
Python micro-fact, not the docstring** (L-020 discipline): `0.0*OnlyRmul()`
fires `__rmul__`; `0.0*OnlyMul()` raises TypeError (`float.__mul__` returns
NotImplemented -> Python falls back to RHS `__rmul__`). So a carrier with
`__mul__`-but-no-`__rmul__` genuinely breaks inside `ScaledOperator`/
`ZeroOperator` (both do `scalar*op`). WATCH: ndarray*ndarray elementwise sites
(`DiagonalOperator`, `RankOne`) are NOT scalar*vector and don't bear on the
contract.

(4) **Coverage honesty: "every leaf satisfies the Protocol" is ONE shared
base, not N independent leaves** -- AngularFlux/ScalarFlux/BoundaryFlux/
HarmonicMomentField all inherit the dunders from the `Field` base. The
genuinely-independent positives are the FAMILIES (np.ndarray native /
Field-base / delegating subclass), not the leaf instances; don't over-credit
3 leaf cases as 3 proofs. A docstring that names a specific carrier as covered
when no test exercises it is a documented-but-untested gap (L-007-flavored) --
recommend +1 line, flag don't block.

(5) **runtime_checkable + @property data members**: `isinstance` checks
PRESENCE of all members (`__protocol_attrs__`; `hasattr`-style, ignores
property-vs-attr and the property's return type). Rule out "all-True/all-False
by accident" with a Partial duck (missing one member -> False, so each member
is individually load-bearing) + a complete Duck (all members, NO inheritance
-> True, structural). An `_is_a(candidate: object, protocol)` helper whose body
is literally `return isinstance(...)` does NOT mask: the `: object` annotation
only launders the STATIC type so pyright skips its unsafe-overlap warning on a
concrete literal -- which is the asserted FACT, not a hazard.

(6) **pyright "= baseline, NO offset" rooting** (the trap: a real +N masked by
a coincidental -N). Airtight no-checkout proof = THREE facts together: (a)
full-tree total EXACTLY the stated baseline; (b) the SUT file isolation 0/0;
(c) the SEAM file (the one a reverted-risky-part touched) has an EMPTY `git
diff --stat` + isolation 0/0. The masked-offset hazard REQUIRES a nonzero diff
on the seam or a new error somewhere -> empty diff + unchanged total rules it
out. Always demand the seam file's diff be empty when a closeout says "I
reverted the risky part."

---

## L-040 -- algebra-law-suite and broadcast-oracle have DISJOINT coverage; the law-suite is swap-INVARIANT (demand a separate nx≠ny oracle for the variable-swap mode)

A multiplier-algebra law-suite (M_1=I, M_0=ZeroOp, linearity, self-adjoint, spectrum→CAP_SOLVE,
homomorphism) on a DiagonalOperator broadcast engine **cannot catch a variable-swap mode #2
(axis-ordering) bug** — linearity `M[af+bg]=aM[f]+bM[g]` and homomorphism `M[f]M[g]=M[fg]` are
ALGEBRAICALLY swap-invariant (probed: a CONSISTENT group/spatial transpose applied to all three
operands preserves both laws → allclose stays True). The axis-ordering bug is caught ONLY by the
**broadcast oracle** (`engine.apply ≡ sigma[None]*psi`) in the discriminating regime: a 2-D carrier
`(N_ord, ng, nx, ny)` with **nx≠ny** makes a spatial-axis transpose `(ng,ny,nx)` RAISE on broadcast
(`(1,2,3,5)` vs `(12,2,5,3)`), whereas a SQUARE mesh silently agrees in shape (no discrimination).
So the two test families are NOT redundant — law-suite = intrinsic-property gate (verify the laws
hold, mutation-proven: nonlinearity reds linearity, additive-offset reds homomorphism, non-unit
scale reds M[1]=I), oracle = the variable-swap catcher. **Rule**: do NOT credit an algebra-law-suite
with axis-ordering coverage; demand a SEPARATE nx≠ny broadcast oracle for the variable-swap mode.
The ≥2G-asym-het requirement on linearity/homomorphism (anti-pattern #3/#4) is about NOT NULLING
the group/spatial structure (so the laws are exercised on real coupling), NOT about catching swaps.

**CAP_SOLVE behavioral-strengthening review (anti-pattern #11, BOTH-tested)**: the promotion adds
an honest spectrum gate (CAP_SOLVE iff min|f|>0) where the legacy CollisionOperator advertised it
always → silent IEEE NaN on σ=0. POSITIVE+NEGATIVE both present and teeth-proven: emulating the
legacy always-on bug (monkeypatch engine to force CAP_SOLVE) reds `test_spectrum_cap_solve...` with
the exact `-O`-firing message. Audit-confirmed safe: 3 prod sites all use σ_t (bounded away from 0
via S2 `total_cross_section_field`); `CollisionOperator.solve` has ZERO prod callers (the WDD sweep
is `InvertibleOperator.solve`, which has its OWN stricter construction-time `σ>0` check at
operator.py:784); the σ_r removal-fold path that COULD go ≤0 is issue #200/#215 — documented, NOT a
live code path. So nothing relied on the old always-on CAP_SOLVE.

**Mode-11 (gate-reaches-new-code) on a promotion**: the resolvent gates (kinf_homogeneous,
si_carve) CONSTRUCT the promoted C (`C_init` 458/22) + fire its `__post_init__` spectrum gate +
read `C.sigma` (19900/1705 — σ threaded into the WDD sweep), but `MultiplicationOperator.apply/solve`
are NEVER called (0/0 — `InvertibleOperator` OVERRIDES apply via loss-rep and solve is the sweep).
So the resolvent gates cover C's CONSTRUCTION+σ-threading; the apply/solve ARITHMETIC is covered by
the NEW broadcast oracle. Honest, complete, non-overlapping — but state it explicitly (don't claim
the kinf gate exercises `M.apply`). **Field-promotion is a label, 0 ULP**: `CrossSectionField.from_mesh(arr).values IS arr` (same object), so the S2 σ_t→CrossSectionField rewire is a pure retype;
broadcast oracle confirmed exactly 0 ULP (both forms are `expand_dims` on axis 0, reduction_depth=1).
Mode-8 clean (0 bare asserts in new test+prod; all gates via `_require`/`pytest.fail`/`np.testing`).

---

## L-041 -- cofree base-extraction "bit-identical carrier" claim: prove the polymorphic `_recombine` hook bites TWO mutation ways + the dedicated unit test must be byte-UNCHANGED vs HEAD

S4.5 extracts a TIMELESS `FullField` base (the 6 vector dunders + `to_flat`/`from_flat` + `copy`/`zeros` + validation, lifted ONCE) out of `TimedFullField`; the timed subclass keeps `_history`/`history_depth`/`advance`/`at_lag` and OVERRIDES `_recombine` (returns TimedFullField, empty history, preserved depth). The load-bearing claim is BIT-IDENTICAL `TimedFullField` behavior (pure carrier extraction, no math).

**Verification recipe for a base-extraction "behavior-unchanged" claim:**
1. **The dedicated unit test must be byte-UNCHANGED vs HEAD** — `git diff <HEAD> -- tests/.../test_timed_full_field.py` MUST be empty. A passing-but-EDITED test is weaker evidence (the edit could have relaxed a path). #257 S4.5: `test_timed_full_field.py` diff empty, 38 pass under `-O`. The NEW `test_full_field.py` is a SUPERSET (adds the recombine teeth + discriminating membership), not a replacement of coverage.
2. **The polymorphic-hook teeth bite TWO mutation ways** (do not stop at one):
   - (a) override returns the BARE base type → "type preserved" tooth reds (3 algebra tests: `type(out) is TimedFullField` fails "got FullField").
   - (b) override DROPPED → base `replace(self,...)` runs → KEEPS history (replace copies the class AND the field) → only the EMPTY-history tooth reds. This is the realistic "forgot to override" mutation and is caught ONLY because the empty-history test ADVANCES first (builds real history_length==1 as a precondition) — without that precondition the `out._history == ()` assertion is TAUTOLOGICAL (zeros()-input already has empty history). The advance-first precondition is the load-bearing non-tautology.
3. **`from_flat` made generic (`template: T -> T`, routes through `template._recombine`)** is pinned by AttributeError-teeth: mutate it to return a bare `FullField` → `from_flat_drops_history`/`_preserves_history_depth`/`iteration_protocol_detection` red with `'FullField' has no attribute history_depth/history_length` (the timed-only attrs are the discriminator).
4. **Discriminating type-check became NOMINAL** (was runtime-checkable Protocol `TransportState`, now concrete `@dataclass FullField` isinstance): confirm at runtime `isinstance(ndarray, FullField) is False` AND `issubclass(TimedFullField, FullField) is True` AND `isinstance(FullField, type) is True` (a real class, not Protocol-only) — anti-#11 positive+negative+timeless/timed all present, Mode-8-safe via `_require`/`pytest.fail` (0 bare asserts).
5. **type:ignore accounting** — count in BOTH files vs the HEAD original: net-new must be 0. S4.5: HEAD `timed_full_field.py` had 2 (`zeros_on` `[attr-defined]`); they MOVED to the new `full_field.py` (still 2), `timed_full_field.py` now 0. Net 0. The main-agent post-fixes (generic `from_flat` removing a `[override]` ignore; `zeros` delegating to base de-duping `zeros_on` ignores back to 2) are ignore-REDUCING, not -adding.

**Baseline-red triage**: the 7 regression reds (#250 SPHERE ×5 huge-ULP ~1e15 stale-snapshot per L-034/L-036 + #232 mu_y ×2 ValueError) are geometry-scoped — SLB/CYL/Cartesian arms in the SAME files (all using the `TimedFullField.zeros(...)` public API) PASS. A carrier-refactor break would fail ALL arms (type/attr error), not just SPH/mu_y (geometry-math/quadrature). Closeout's baseline-worktree (`93aa016` + symlinked `.venv`) independently saw the same 7. pyright EXACT `2295 errors, 19 warnings`.

**Pyright baseline-comparison gotcha (from closeout, worth keeping)**: a git worktree pyright count is ONLY comparable with the MAIN `.venv` symlinked into the worktree root (pyrightconfig `venv: .venv`); a worktree without it analyzes a different file set → bogus count (2922 vs 2295).

---

## L-042 -- additive-only SUT: prove the "baseline-invariant" claim by grepping for ANY importer (empty ⇒ cannot perturb), and the variable-swap teeth are the contraction AXIS not the operand order

S5 = additive only: 2 new untracked SUT files (`orpheus/numerics/functional.py`
runtime_checkable `Functional` Protocol; `orpheus/transport/production_rate_functional.py`)
+ 1 additive `numerics/__init__.py` export. NO edit to operator.py/fission.py/solver.py
(`git diff --stat HEAD` empty for all three). **Additive proof for "baseline-7 invariant"**:
`grep` the tracked tree for any importer of the SUT EXCLUDING the 4 new test files →
EMPTY ⇒ no pre-existing consumer ⇒ S5 cannot perturb any pre-existing outcome (stronger
than re-running the reds). transport+numerics dirs 915 passed/1 skipped; fission 18 passed.

**Bit-id premise must be checked, not assumed**: `CrossSectionField.from_mesh(nsf,sn).values`
is `array_equal` to raw nsf (the producer doesn't transform), so SUT
`(nu_sigma_f.values*phi).sum(axis=0,keepdims=True)` is 0-ULP `array_equal` to the legacy
`RankOneOperator.apply` `inner` line 1776 (`(right*x).sum(axis=axis,keepdims=True)`,
right=νΣf, axis=0). Correctness rides a GENUINELY structurally-independent ref
(`hand_derived_production_density` = explicit nested-Python double-loop, no numpy reduction,
no ORPHEUS algebra) — L11 clean. B.2 RankOne-equivalence is correctly DEMARCATED as de-risk
not correctness.

**Mode-2 framing trap (test author got it RIGHT)**: a literal νΣf↔φ swap is VALUE-INVARIANT
(pointwise product commutes — verified `array_equal`). The genuine Mode-2 hazard is the
CONTRACTION AXIS, discriminated by the nx≠ny (5≠3) mesh by SHAPE. Don't accept "swaps the
operands" as the swap teeth; the axis is the teeth.

**Mutation results (L12)**: `axis=0→1` reds exactly 6 production-rate gates (matches
test-architect claim); category gates stay green (assert no number). Shape-preserving Mode-3
`×1.5` magnitude factor reds 5 gates incl. dedicated `test_density_unweighted_by_cell_volume`
— real measure-fold teeth. **CAVEAT (novel)**: `keepdims=True→False` does NOT redden — the
`squeeze_density` helper collapses a leading length-1 axis, making the WHOLE suite agnostic to
keepdims (by design per helper docstring; even B.2 bit-id passes squeezed). So the user's
"confirm keepdims reddens" is the ONE unsupported sub-claim: the suite tolerates either output
rank, it does not ENFORCE keepdims. Not a correctness bug (value+bit-id hold either way) — but
the bit-identity-with-legacy claim is enforced only up to a squeeze, not literal rank.

**Category teeth (Claim 2) — the headline gate is the WEAK one**: runtime_checkable
`LinearOperator` checks 4 members (`apply,capabilities,domain,codomain`); `isinstance(func,
LinearOperator)` only goes True on a FULL 4-member leak. The realistic leak (a Functional that
grows `apply`+`capabilities`) leaves the headline `isinstance NOT LinearOperator` BLIND
(domain/codomain missing → still False) but reds the 3 surface gates `lacks_apply`/
`lacks_capabilities`/`disjoint`. Pytest-level leaky-probe mutation: all 3 surface gates red,
headline blind on partial leak. The 3 surface gates ARE the defense (test-architect flagged
this correctly). Mode-8 clean: 0 bare `assert` in any S5 test file; all route through
`require`(pytest.fail)/`np.testing.*` (fires under -O). pyright EXACT `2307 errors, 19 warnings`
(=user's b404ae1 baseline; plan's 2295 was stale-worktree); both SUT files individually 0/0/0
→ no masked offset. 1 skip = by-design (no estimator wrapper shipped).

---

## L-043 -- Mode-11 sentinel for a NEW PRIVATE adapter = install it as a PYTEST PLUGIN that wraps the internal call (prove scipy FIRED it, not just BUILT it)

Bit-identical refactor: 2 inline scipy-LinearOperator closures (`A_matvec`/`M_matvec`)
in `KrylovAcceleration.solve` → 1 named carrier `loss_minus_gains(psi)` + 2
`_as_scipy_linop(carrier, template, n)` calls; retired public `as_scipy_linop` (0 callers)
+ orphaned `spla` import + 5 tests + 3 doc xrefs.

**CLAIM-1 byte-id (Mode-2 A/M template-swap):** prove the non-swap TWO ways — (a) read the
2 call sites (A→`solution_template`, M→`q_ext`), (b) RUNTIME binding sentinel: monkeypatch
`_as_scipy_linop` + `KrylovAcceleration.solve` (stash `q_ext`/`solution_template` ids in
solve, compare template id in the adapter) → reported `carrier=loss_minus_gains
bind=solution_template` / `carrier=<lambda> bind=q_ext`. A swap would have inverted these.
`loss_minus_gains` reduction order char-identical to old `A_matvec` (L.apply first, then
`for g in self.gains: out=out-g.apply`); `(n,n)`/`dtype=float` preserved.

**CLAIM-2 Mode-11 sentinel for a NEW PRIVATE adapter — sharper than L-031/L-033 in-process
probes:** install the sentinel as a PYTEST PLUGIN (`-p <module>`, module must be on
PYTHONPATH — `-p /tmp/x` fails "No module named", copy to cwd + `PYTHONPATH=$(pwd)`), patch in
`pytest_configure`, restore + summarize in `pytest_unconfigure`. Wrap `linop._matvec` (the
internal scipy calls, NOT `.matvec`) with a counter to prove scipy FIRED it, not just BUILT
it. Tag A vs M by `carrier.__name__` (`loss_minus_gains` vs the precond lambda). Result on
identity-precond[slab] (non-None `lambda q:q` → M built): A built=2/fired=160, M
built=2/fired=161, both on TimedFullField → M-template wiring exercised on the REAL typed path.
This is the gold-standard Mode-11 evidence WITHOUT mutating any tracked production file (L28).

**CLAIM-3 retirement:** word-boundary grep (`[^_]as_scipy_linop`) = 0 hits in orpheus/tests/docs
(the `_as_scipy_linop` private hits are noise). 5 deleted tests pinned a now-gone
`LinearOperator`-taking public adapter; its only unique assertion (`MissingCapability` on
missing `CAP_APPLY`) maps to a behavior the NEW bare-callable adapter does NOT have — the
equivalent guard moved UP to `KrylovAcceleration.__init__:422` (composition-time, STRONGER),
covered by 14 `MissingCapability` refs in test_iteration.py + 3 surviving `NoApplyOperator`
negatives in test_operator.py. No non-redundant coverage lost.

**pyright net-new=0 PROOF without mutating tree:** the 3 `reportCallIssue` at the new line 228
(`spla.LinearOperator((n,n),matvec=...,dtype=float)`) are the scipy-stub false-positive that
existed at HEAD across `grep spla.LinearOperator HEAD:iteration.py` = 2 sites (756,766) + a 3rd
in the deleted public adapter → refactor CONSOLIDATED 3 sites → 1, which is why total dropped
2307→2297 (−10). `# type: ignore` delta −1 (removed `op.apply_transpose`), 0 added. Gates:
138 Krylov/round-trip pass (-O); broad regression 7 reds = EXACTLY #250 SPH×5 (huge-ULP ~1e15
while SLB sibling 1-ULP DriftWarning-pass = L-034 stale-snap) + #232 mu_y×2, all in
tests/sn/operators/ with 0 refs to the changed code (orthogonal). No ERR (no bug caught).

---

## L-044 -- "producer-now-emits-X, test-helper-decoupled" re-baseline integrity: prove the helper rebuilds the flat baseline TEST-SIDE or it silently inherits the new emission

S9 makes `SN2DCartesianLDStressMMSCase.prescribed_inflow` EMIT the moment-resolved
face slot (slot-0 transverse cell AVERAGE, slot-1 bare transverse P1 slope) via a
case-owned leggauss-only `_project_inflow_to_face_moments`, gated on
`face_moment_count>1` (DD/Step byte-identical). NO new field type, NO value gate
(slope is sub-floor for converged flux — vv Mode-10 companion-unavailable, 3rd recurrence).

**Re-baseline-integrity recipe for a "production-now-emits-X, test-helper-decoupled"
change** (the HIGH-PRIORITY trap): when a producer (MMS `prescribed_inflow`) gains a
new emission AND a test helper's "flat baseline" branch USED to route through that
producer, the helper MUST rebuild the flat baseline TEST-SIDE or it silently inherits
the new emission → toggle collapses. PROVE the decoupling kept teeth by probing the
helper's 4 legs directly: `None==zero` byte-id (slope-free baseline + no-op control
has teeth), `|mom−None|`/`|flip−None|`/`|mom−flip|` all ≫tol (slope consumed, sign
matters), `None≠mom` (toggle not vacuous). #257 S9: None==zero byte-id, |mom−flip|≈2.19e-2.

**Sign-mutation gate teeth proof (cheap, in-process, no prod edit):** monkeypatch the
SLOPE SOURCE (`_face_transverse_buffers`) to zero the slope, re-run the mom-vs-flip
comparison → `|mom−flip|/|φ|` goes 4.10e-3 (healthy, ~5.6 orders >1e-8 `_CONSUMPTION_TOL`)
→ 0.000e+00 (bug) → gate reds. Confirms the consumed-flip is genuine, not a tautology.

**Mode-11 producer-stamp NOT circular** (L-029 applied): GATE-B compares production
`case.prescribed_inflow` slot-1 vs `_face_transverse_buffers` (test-side leggauss),
NOT vs `case._project_inflow_to_face_moments` — two INDEPENDENT leggauss impls; GATE-C
separately pins their agreement (array_equal, maxdiff 0.0). A sign error in the prod
projector would NOT propagate into the test ref → GATE-B reds. Sentinel-instrument
`LR._LossRepresentation._inflow_to_moments` (the genuine prod consumer, reached 688×/solve):
flat/zero slot1==0, mom/flip slot1=1.9e-2, `|mom−flat|`phi_sum>1e-3 (consumed), zero==flat
byte-id. Mode-11 closed: producer IS exercised, not a surrogate.

**Verdict-pin teeth** = `improves`(mom<flat) check + `|mom−flat|/flat ≤ 0.30` band.
At bc_scale=20× (strongest amplification), mom monotonically WORSE (improves all False,
rel max 0.205 < 0.30, orders [1.7,2.4]) → sub-floor wall fundamental. Pin reds if slope
ever becomes above-floor. Coherent-promise gate teeth = flat first-cell-row order ≥1.85
(measured 1.99/2.00/2.00 — average alone delivers O(h²) at boundary, no asterisk).

**DD byte-identity proven 3 ways:** (a) `np.array_equal(prod_DD.values, pre-S9 face_coords
build)==True` (1344,); (b) GATE D strict `-W error::DriftWarning` 520/1/4 = baseline, NO
DriftWarning fired; (c) no LD-stress consumer in tests/sn/sweep/core or solve (grep) → no
value/snapshot pin could shift. Gates: G1 35pass / G2 590pass,1skip,4xfail / GATE-D 520/1/4
/ pyright 2282 = baseline 0 net-new. Mode-8 clean (0 bare assert in new file or prod).
NO blocker, NO false-green, NO ERR.

---

## L-045 -- a "behavior-neutral field-zeroing" claim is valid ONLY for the ONE fission/emission contract it was proven against; re-prove inertness for EVERY consumer (ERR-063)

SUT = `EmissionSpectrum(np.ndarray)` value-object + `Mixture/Isotope.__post_init__`
simplex/null χ guard (keyed `is_fissile = bool(np.any(SigF>0))`) + a "behavior-neutral"
precursor zeroing non-fissile χ on shared `xs_library` regions B/C/D. The TYPE + guard +
intrinsic-property gates are SOUND; the precursor is NOT behavior-neutral → BLOCK.

**What's SOUND (mutation-/byte-verified):** (1) intrinsic gates vv#11 BOTH legs, hand-laid
L11 refs; negativity clause INDEPENDENT of sum (mutate prod: drop `>=0` → ONLY
`test_negative_entry_raises_even_when_sum_is_one` reds; drop sum → 2 sum legs red, negativity
green; relax `assert_null` to atol=1e-6 → `test_any_nonzero_raises` reds, pinning STRICT
exact-zero). (2) Mode-8 clean (0 bare asserts; all `_require`/`pytest.fail`/`np.testing`/
`pytest.raises`; 28+13 pass under -O incl real-GENDF). (3) SN-path behavior-neutrality REAL:
direct re-solve of het 3-region DD (fuel A + non-fissile mod B) with mod.chi=[1,0] vs [0,0]
→ keff `1.2298233055738448` BYTE-identical + flux array_equal (max abs diff 0.0); confirmed
SigP≡0 on B/C/D so SN `FissionOperator` χ·(νΣf·φ)≡0. (4) is_fissile/SigP seam (item 5):
explorer audit + GENDF MF6/MT18-co-located-with-MF3/MT18 → NO real-data production path has
nonzero-χ-∧-zero-SigF; only the synthetic billiard fixture (reads SigP/chi, never SigF —
SigF=nu_sf injection inert, confirmed billiard.py:1031-1032 + tree-grep). DD reg 13pass / TA
full 107pass,2xfail (matches closeout).

**THE BLOCK (ERR-063):** "zeroing non-fissile χ is inert" assumed the SN/`compute_macro_xs`
contract (χ gated by SAME region's νΣf). FALSE for `solve_peierls_mg`: its MG fission op
`B[i,ge,j,gs] += K[i,j]·chi[i,ge]·nu_sf[j,gs]` weights SOURCE-region νΣf by SINK-region χ, so
χ on non-fissile region B (the emission spectrum of fission BORN in A but emitted INTO B) is
LOAD-BEARING. Direct probe: region-B χ [1,0]→[0,0] moves peierls k_eff `1.0985→0.5563` (1G/2R)
/ `1.1008→0.3856` (2G/2R) — O(1), not ULP. 7 L1 tests in
`tests/derivations/test_peierls_rank_n_class_b_mr_mg.py` (cylinder/sphere hebert overshoot +
recovers_kinf[2G_2R]+RICH + mark_floor[cyl/sph]) FAIL under S10a, PASS at clean HEAD (proven
via `git worktree add c6e21c0` + PYTHONPATH=worktree: 4+4 passed). Only RICH is @slow; other 6
plain @l1. Closeout MISSED it — it relied on "0 EmissionSpectrum reds" (counts only guard
ValueErrors, blind to silent accuracy regressions) + "DD snapshots didn't move" (DD = SN-only,
the consumer where χ IS inert) + never ran the 494s peierls suite. Test authors had ALREADY
flagged this χ-dependence (commit 76b11e8, Issue #132).

**RULE (new):** a "behavior-neutral field-zeroing" claim is only valid for the ONE
fission/emission contract it was proven against. When the field is a SHARED source feeding
consumers with DIFFERENT contracts (same-region χ·νΣf vs sink-region-χ × source-νΣf), re-prove
inertness for EVERY consumer with a DIRECT old-vs-new VALUE comparison (O(1) move = not neutral),
NOT a fast proxy ("snapshots didn't move" / "no guard errors"). Run the slow accuracy-band suites
that consume the edited field. L20 shared-source hazard + H5 (test count ≠ coverage). Recommended
fix: do NOT zero the shared library χ — decouple peierls cases' χ from the guarded library, OR
key the guard on production not SigF, OR restrict the guard off placeholder library regions.
Worktree-baseline recipe (clean-HEAD confirm): `git worktree add /tmp/x <HEAD>` + run with
`PYTHONPATH=/tmp/x` (editable .venv else imports MAIN tree — verify `orpheus.__file__`).

---

## L-046 -- for a WEIGHTED value-pin, L11-independence is not enough: the hand-ref must carry EVERY weight factor AND the fixture must make a factor-BLIND formula give a different answer

SUT = NEW `production_weighted_chi(isotopes,sigF,aDen,fissile_indices)` helper
(`χ_mix = weights @ fissile_spectra`, `w_i = aDen_i·Σ_g ν̄σf_i / Σ_j(…)`) replacing the
first-fissile χ shortcut in `compute_macro_xs`. The S10b consumer that produces the
multi-fissile χ_mix the [[L-045]] S10a guard validates for free (gate-1 interlock). ALL 7
review points SUPPORTED; clean, no blocker.

**⭐ THE #1 SCRUTINY — hand-ref structural-independence for a WEIGHTED value-pin.** When the
only value-pin is a hand-laid convex average, L11-independence is NOT enough: the hand-ref
must independently carry EVERY weight factor (here `aDen` AND the `Σ_g ν̄σf` production sum),
AND the fixture must make a factor-BLIND formula give a DIFFERENT answer or the factor is
untested (a vacuous pin). PROVE it two ways: (1) the fixture discriminates — gate 2 uses
unequal `aden=[2,1]`, so aDen-aware w=[0.4545,0.5455] vs aDen-blind w=[0.2941,0.7059] → χ
differs by 0.128 ≫ atol=1e-12 (compute by hand, don't eyeball); (2) MUTATE the production to
be aDen-BLIND (drop the `aDen[i]` factor) → gate 2 reds, gate-1 simplex + gate-3 byte-id STAY
green (blind formula is still a convex average of simplices = a simplex; single-fissile
unaffected). Hand-ref here is genuinely independent: explicit scalar `p_i = aden[i]*nubar_i*sigf_i`
term-by-term (single-nonzero-group fact), NOT `weights @ fissile_spectra` re-spelled. The
aDen-blind variant IS the "shares the code's weight-derivation / forgets aDen the same way"
failure the brief warned of — proven defeated.

**Other teeth (all mutation-verified live, not trusted from closeout — L12):** legacy
first-fissile shortcut → gate 2 + real-UO2 smoke red, gate1/gate3 green; unweighted mean →
gate 2 red (sole catcher); non-convex `2·(weights@spectra)` → gate-1 S10a `assert_normalized`
interlock fires (7 red) = the interlock is LIVE. Single-fissile collapse is EXACT byte-id
(w=[1], max abs diff 0.0). Mode-7 honest-scope ("flat-flux representative, NOT flux-exactness")
declared in helper docstring + gate file. Mode-8 clean (8/8 under -O, all `_require`/`np.testing`).

**Byte-identity scoping (re-baseline list EMPTY, independently confirmed):** DD regression
13pass (within-tol DriftWarnings pre-existing FP-noise); DD path never touches `compute_macro_xs`
(grep empty — builds Mixture via `xs_library.make_mixture`). Multi-fissile `compute_macro_xs`
callers = `uo2_fuel`/`pwr_like_mix` (`fissile_indices=[0,1]`); the ONLY pytest-collected
consumer is `test_solver_components.py::test_profile_421g` = a TIMING test (prints ms, asserts
NO k_eff/flux); `pwr_like_mix`/other `uo2_fuel` refs all in `examples/` (NOT in
`testpaths=["tests"]`). So no committed test pins a converged value off a multi-fissile mixture
→ the χ-value change rests entirely on gate 2's hand-ref. pyright net-new = 0 (mixture.py 3
errors WITH==WITHOUT change via stash; all 3 pre-existing `SigP/Sig2/SigT = sum(...)` int-noise,
#226; full project 2353==baseline). The closeout's `weights @ fissile_spectra` deviation (vs
brief's `sum(generator)` which costs +1 `reportReturnType`) verified principled — a convex
average IS a matvec.

---

## L-047 -- a runtime_checkable category Protocol's member-presence loophole is REAL; the direct `not hasattr(...)` negative gates are the defense (+ the partial-coverage `kernel ≠ full apply` caveat)

S6 is ADDITIVE + bit-identical: a §5.6 `IntegralKernelOperator` Protocol
(`orpheus/transport/integral_kernel_operator.py`, `@runtime_checkable`,
sole member `kernel`), a `FissionOperator.production_rate` property (S5
`ProductionRateFunctional` over νΣf), a `ScatteringOperator.kernel`
property (`OperatorProduct(R, OperatorProduct(Λ, M))`, `skip_l0=True`).
All 5 claims SUPPORTED; 2 caveats, no blocker, no false-green.

**Claim 1 (category teeth) -- `runtime_checkable` member-presence loophole
is REAL but the direct-attr gates close it.** Monkeypatch `ident.kernel =
"fake"` on `IdentityOperator()` → `isinstance(ident, IKO)` flips True
(the documented S5 loophole; isinstance only checks PRESENCE). The 3
negative gates that assert `not hasattr(..., "kernel")` directly
(`*_lacks_kernel`, `*_lacks_kernel_and_apply`) are the defense-in-depth
that does NOT depend on the Protocol machinery. Discriminator
(`IdentityOperator` IS a LinearOperator but NOT an IKO) proves a strict
refinement, not a `LinearOperator` alias. 20/20 green under -O.

**Claim 2 (fission) -- B.1 L11-clean, B.2 Mode-11-live.** B.1 reference
`hand_derived_fission_emission` = explicit Python double-loop, shares NO
numpy reduction with production (role-swap sensitive, verified
max-rel-diff 10×). B.2 reads `op.production_rate` OFF the live operator;
MUTATION-VERIFY (point production_rate at `total_cross_section_field`
instead of νΣf) reds BOTH B.2 gates @100% mismatch. `evaluate` =
`(nu_sigma_f.values*phi).sum(axis=0, keepdims=True)` = the RankOneOperator
`inner` line byte-for-byte → bit-id is structural. ⭐ ASYMMETRY:
`fission.kernel` IS the FULL F (production reads it at fission.py:454/471);
`scattering.kernel` is the aniso ℓ≥1 part ONLY.

**Claim 3 (scattering) -- Mode-11-live + the skip_l0 blind-spot CAVEAT.**
`S.kernel.apply == _aniso_source_from_moment_values(M·ψ)` @ 0 ULP, reads
live `S.kernel`. MUTATION: drop R → 2 gates red (value + shape moment-
tensor); `skip_l0 True→False` → value gate red (subtle flag IS load-
bearing). ⚠ CAVEAT: `S.kernel.apply` (L2≈0.98) is ~5% of the full
scattering source (L2≈21.9) -- ONLY the ℓ≥1 aniso redistribution, pre-1/W;
P0 in-scatter + n2n are NOT in it. DOCUMENTED honestly (module + property
docstrings, "genuinely-nonlocal-in-angle part", "P0/n2n are LOCAL/separate
components"; test docstring says "pre-1/W") but NO production reader and NO
test POSITIVELY asserts `kernel != full apply`. A future consumer mistaking
`kernel` for full S silently loses ~95% of the source. Recommend a 1-line
gate `require(not allclose(kernel.apply(ψ.values), S.apply(ψ).values))` to
pin the partial-ness; minor follow-up, not a blocker (the only current
consumer is the cross-check, which knows the semantics).

**Claim 4 (matvec arms byte-id) -- 17/17 green.** TestAnisoMomentSourcePath,
TestProtocolCompliance, TestP0AlgebraicIdentities,
TestRankOneTensorProductKernel, TestBitIdenticalToLegacyInlinedMath all
green; aniso MMS `test_curvilinear_aniso_scattering_p1.py` 2/2 green (the L1
physics reference for scattering).

**Claim 5 (pyright) -- CONFIRMED 0 net-new on production.** CLI `npx pyright`
= 2311 errors / 19 warnings (= user's number; S5 base 2307 + 4 from the new
test skeletons). fission 8, scattering 22 — proved 0 net-new by stash-tracked
+ hide-untracked-module → TRUE baseline (S6 reverted) = 30 = 8+22 unchanged.
The 3 `cast(LinearOperator, ...)` in scattering.kernel are a legit PEP-484
bridge (MomentProjection/LegendreMomentScattering/HarmonicMomentReconstruction
all carry `apply` at runtime; composition green @ 0 ULP) for the
unparametrised-LinearOperatorMixin generic gap (#226). 0 `# type: ignore`
added (the one match is inside an explanatory comment).

**Claim 6 (baseline reds = 7) -- CONFIRMED.** 5 #250 SPHERE stale-snapshot
reds (test_streaming_operator.py TestT4cPreT4RegressionSnapshotCurvilinear
×2 + test_bc_extraction_matvec.py SPH ×3, max ULP ~8.77e15 = L-034 stale-snap
signature) + 2 #232 mu_y. S6 touches none of streaming/matvec snapshot code
(additive only) → reds pre-exist.

**Mode-8 -- the test-architect's flag on `TestRankOneTensorProductKernel`
bare asserts is WRONG (re-confirms L-010).** The 4 NEW S6 test/helper files
have ZERO bare asserts (all route through `require`=pytest.fail / np.testing.*).
The existing `TestRankOneTensorProductKernel` (lines 365-411) DOES use bare
`assert isinstance/is/==` — BUT these FIRE under -O because pytest's assertion
REWRITER rewrites asserts in collected tests/ modules at import time, BEFORE
-O would strip them. PROVEN twice: (a) broke the kernel → bare-assert
`isinstance` red under -O with AssertionError; (b) `assert 1==2` probe in a
tests/_tmp_probe module FAILS under -O. So `TestRankOneTensorProductKernel`
is NOT a Mode-8 gap; the test-architect's "S6 should fix it" is unnecessary
(Mode-8 is a concern ONLY for bare asserts in `orpheus/` production, not in
collected `tests/`).

---

## L-048 -- "behavioral-neutral codomain re-point" = TWO legs: identical failing-test IDs vs a read-only baseline worktree AND the dropped field has ZERO production `.advance(` callers

#257 S8a: SN operator matvec leaves (`StreamingOperator/InvertibleOperator/
MultiplicationOperator/SNBoundaryOperator.apply`, the `S`/`F` TimedFullField-input
arm) re-typed to EMIT the timeless `FullField` instead of history-bearing
`TimedFullField` (cofree-comonad finding: an operator is a base arrow
`FullField→FullField`; only the iteration DRIVER carries the comonad). Claimed
value-neutral / bit-identical. VERDICT SUPPORTED — recipe:

**CLAIM-1 value-neutrality = TWO independent legs.** (a) Reconcile the
baseline-red set against a READ-ONLY `git worktree add -d HEAD~ /tmp/x` checkout
(L28 — never mutate the working tree): run the EXACT same `-O` gate on both;
S8a-tree and baseline must produce IDENTICAL failing test IDs (here 7: #250
SPHERE×5 + #232 mu_y×2; pass-count delta = +14/+1xfail = exactly the new C5
file, nothing else). ZERO non-baseline reds. (b) Prove the dropped `_history`
is genuinely unused in steady-state: grep + Nexus `context` for ALL production
`.advance(` callers — if ZERO `calls` edges (only docstrings/prose), the history
shift-register is test-only and dropping `_history=()` cannot perturb any
converged value. Here confirmed 0 production callers.

**The reattach mechanism (don't take on faith).** The driver re-attaches the
timed type via `TimedFullField.__add__`'s `_recombine` hook — CONFIRM the timed
operand is on the LEFT of the `+` (`rhs = q_ext + g.apply(psi)`, q_ext timed →
`rhs.__add__(FullField)` → `self._recombine` resolves to `TimedFullField._recombine`).
The reverse order (`timeless + timed`) would resolve to the BASE hook and yield
timeless — a silent history-drop. Also confirm the resolvent `L.solve` STILL
returns TimedFullField (re-mints the iterate each step). Krylov path is
`FullField−FullField→FullField` throughout (unravels to scipy flat, reconstructs
from solution_template — never relies on `__add__`).

**CLAIM-2 scope (no math drift).** For a "type-surface only" production diff,
PROVE it by filtering the diff: every `^[+-]` line must be a type annotation
(`"TimedFullField"`→`"FullField"`), a docstring/comment, or whitespace — ZERO
numerical expressions (`sigma`, `values`, `einsum`, `out_bulk`, the `(L+C)−C`
arithmetic `lpc.bulk.values - sigma_t[None]*psi.bulk.values`). Confirm the
INPUT dispatch (`@apply.register def _(self, psi: TimedFullField)`) is UNTOUCHED
(only the return annotation + the output construction `TimedFullField(...)→
FullField(...)` dropping `_history=()`/`history_depth=` changed). A drift into
the NEXT sub-stage's behavioral change (here S8b pure-L) would falsify neutrality.

**CLAIM-3 teeth + Mode-11 (the matvec leaf has ZERO graph callers — reached only
via OperatorSum/driver).** The codomain gate (C5) MUST call `L.apply`/`C.apply`/
`F.apply` DIRECTLY (not solve-only — solve routes through the sweep/loss-rep and
never touches the matvec emit path). Mutation-verify teeth by REVERTING the
re-point on ONE leaf (make `StreamingOperator.apply` emit `TimedFullField` again,
QA-MUTATION-SENTINEL comment) → the C5a `type(out) is FullField` checks go RED
across all geometries with a precise diagnostic (`got TimedFullField`); revert +
confirm green + `grep QA-MUTATION-SENTINEL` = 0 residue. The legacy snapshot
gates (`TestT4b/c`) DO reach the matvec leaf (`L.apply(state)` directly) — verify
slab/cyl arms reproduce frozen bulk (`assert_regression kind=direct`) + STRICT
0-ULP boundary; SPHERE arms are #250 stale-snap (O(1) value diff, L-034), red on
BOTH trees = pre-existing.

**CLAIM-4 re-pointed B-tests.** The ~41 re-pointed tests are clean type-surface
updates: `isinstance(out, TimedFullField)` → `isinstance(out, FullField)` +
`not isinstance(out, TimedFullField)`, DROPPING the now-meaningless
`out.history_depth==depth`/`out._history==()` assertions (they test the EXACT
attribute S8a removes), while PRESERVING all value assertions (`.bulk.mesh is`,
`isinstance(.bulk, AngularSourceSink)`, `.values.shape`, boundary `==0.0`). The
roundtrip tests (`test_removal_form_matvec_sweep`, `test_invertible_operator`
solve∘apply=id) re-wrap the timeless `op.apply` output into a TimedFullField
before feeding `solve` — mirrors the driver; byte-identical source `.bulk`/
`.boundary`. NIT (cosmetic, not blocker): a few function NAMES still say
`..._timed_full_field` despite now asserting timeless. Gate counts: pyright
2297/19 (=baseline, 0 net-new); regression 7 baseline reds only; L1/MMS 40pass/
2xfail (converged limit unmoved); C5 14pass/1xfail (sphere-krylov #200 xfail).

---

## L-049 -- when a value-moving carve preserves the COMPOSITE not the leaf, prove byte-id on the composite directly; pin a re-baselined `.npy` to a STRUCTURALLY-INDEPENDENT reference, not "whatever the leaf emits"

VERDICT SUPPORTED. The value-moving core of the streaming carve: production's
within-group matvec uses the COMPOSITE `(L+C).apply`=`InvertibleOperator.apply`
(rides `loss_action(σ_t)` UNCHANGED) — so the value-preservation story rests on
composite byte-identity, NOT the standalone pure-L leaf.

**CLAIM-1 (composite byte-id) is the load-bearing one — prove it directly.**
Emit `(L+C).apply(ψ)` on live vs a read-only `git worktree add -d 9316321`
baseline, per geometry (slab/sphere/cyl) ≥2G het, `PYTHONPATH=$PWD` to OVERRIDE
the editable .venv (confirm baseline ran baseline code via `inspect.getsource` +
dataclass `fields` — `sigma_t` present/absent is the tell). Result: 0 ULP,
absdiff=0.000e+00 all 3 geoms. The pure-L LEAF drifts (CART 32 / SPH 12 / CYL
117 canonical numpy nulp, boundary STRICT 0-ULP exactly) — "≤16 ULP" in a brief
can UNDERSTATE the leaf drift (CYL 117); it's still genuine FP-reassoc
(large-mag → moderate-ULP, rel ~1e-15) and the leaf has ZERO graph callers
(Nexus `callers` total:0), so inconsequential. Test bound `_BULK_NULP=256`
covers it; T4b snapshot DriftWarnings up to 192 ULP all within 256.

**CLAIM-4 re-baselined .npy (the headline laundering risk) — pin to the
STRUCTURALLY-INDEPENDENT composite, not "whatever pure-L emits".** The 3
`bc_extraction_2d` `.npy` were re-captured. The decisive check is NOT
"committed==pure-L" (circular) but "committed == `(L+C).apply.bulk − σ_t·ψ`"
(= the BYTE-IDENTICAL composite minus the collision diagonal) — measured ≤64
ULP (rel ~1e-16). Because the composite didn't move a single bit (CLAIM 1) AND
pure-L = composite − collision to ULP, the frozen `.npy` IS the genuine pure-L
value. MASKING-CHECK: OLD baseline `.npy` ≠ NEW `.npy` (absdiff 7.1e-15) ⟹ the
re-baseline was LOAD-BEARING (the strict gate would trip on the un-rebaselined
snap), not cosmetic. Three re-pointed test files (`TestSubtractiveDefinition`
→`array_equal`→`assert_array_almost_equal_nulp(256)` + boundary STRICT;
`test_apply_equals…`→`test_pure_L_plus_C_recovers_loss_action_het` with
composite==loss_action byte-exact + affine ULP; `TestResolutionADifferent…`
→`TestPureLIsLossActionAtZeroSigma` array_equal vs `loss_action(0)`) — all
structurally grounded.

**CLAIM-3 (C1 σ-freedom teeth + Mode-11).** Mutation-verify BOTH: (a) sentinel
on `loss_rep.streaming_action` FIRES (hits=1) when `L.apply` runs → C1 reaches
the rewired matvec leaf (Mode-11 — the leaf has zero callers, sweep routes
around it); (b) a σ-re-reading stub (`loss_action(σ_t)` not `loss_action(0)`)
makes `L.apply` σ-dependent → outputs DIFFER by O(1) maxdiff ~12 → C1's
`array_equal` reddens. The shipped teeth test asserts the leaking stub differs;
both confirmed all 3 geoms.

**Gates.** pyright full-tree 2297/19 = baseline (the AUTHORITATIVE oracle; the
baseline-WORKTREE pyright is mis-rooted (L-027 cross-tree-config artifact) — `numpy` unresolved → renders
the SAME #226 family with `Unknown` types, so the message-multiset diff is
noise; confirm instead that NO live diagnostic references `streaming_action`/
`_zero_sigma_for`/the new apply). 0 net-new `# type:ignore`. Broad regression
`-O`: 7 reds, ALL reconciled PRE-EXISTING on the baseline worktree (5 SPHERE
#250 = 3 `test_bc_extraction_matvec[*-SPH]` ~1e15 ULP L-034-stale + 2
`test_streaming_operator` T4c sphere; 2 mu_y #232). ⚠ the spec's route-around
`-k "not (sphere_1g/2g_apply)"` named only 2 of the 5 SPHERE reds — the 3
`test_bc_extraction_matvec` SPH are also #250-family (run WITHOUT `-k` +
reconcile all 7 vs baseline = stronger). CLAIM-5: `scattering.py`/`fission.py`/
`orpheus/transport/` byte-untouched (empty diff); `loss_action` body unedited
(only `streaming_action`/`_transpose`/`_zero_sigma_for` ADDED);
`InvertibleOperator.apply`/`.solve` NOT in diff. NO blocker, NO false-green.

---

## L-050 -- a singledispatch alias rename (`apply`→`_apply_impl`, `else: apply=_apply_impl`): prove runtime bit-id via `Cls.__dict__['apply'] is Cls.__dict__['_apply_impl']`, and a removed `NoReturn`-poisoned return UNMASKS every latent downstream pyright error

The change: `FissionOperator`/`ScatteringOperator.apply` dispatch on input CARRIER
type → DISTINCT output carrier (heteromorphic, not endomorphism). S8c renamed the
`@singledispatchmethod` dispatcher `apply`→`_apply_impl` (base `-> "Any"`, was no
annotation ⇒ pyright inferred `NoReturn`), kept all `.register` arms at natural
indent, and added `if TYPE_CHECKING: @overload def apply(...)->Carrier; def apply(self,x:Any,/)->Any else: apply=_apply_impl`.

**Runtime bit-identity is BY CONSTRUCTION + the alias-identity proof.** `apply` IS
`_apply_impl` at runtime — prove via `Cls.__dict__['apply'] is Cls.__dict__['_apply_impl']`
→ True (do NOT use `Cls.apply is Cls._apply_impl` — the singledispatchmethod
DESCRIPTOR returns a fresh `_singledispatchmethod_get` wrapper each class-attr access
→ False, a red herring). The `TimedFullField` arm's `self.apply(...)` still routes the
SAME dispatcher. Confirmed empirically: 111 operator-suite + C6 PASS, 77 Section-D
MMS backstops PASS (2 pre-existing xfails #195/#252) — bit-identity-sensitive
(convergence rates + ERR-026 catches would break on any runtime change).

**Mode-11 (rewired-path reached).** C6 `test_c6_apply_dispatch_parity` calls the
PUBLIC `apply`; alias-identity guarantees reach, but PROVE it: register a sentinel
arm on `Cls.__dict__['apply'].dispatcher` (3.14 API: `.dispatcher.dispatch(Carrier)`
to grab orig, `.dispatcher.register(Carrier, wrapper)`) then call `F.apply(phi)` →
sentinel fires count=1 + returns the right type. Mode-8 OK (`pytest.fail` not bare
assert; passed under `-O`).

**Static C6 gate has TEETH (mutation-verified).** `_c6_static_typing_pins` (no
`test_` prefix → never collected; pyright-only) carries `assert_type(F.apply(phi),
ScalarSourceSink)` per carrier. Mutate one overload (`ScalarFlux→ScalarSourceSink`
⇒ `→AngularSourceSink`) → `npx pyright <testfile>` reddens EXACTLY that
`assert_type` line (`reportAssertTypeFailure`) → revert (L28: edit-revert, NOT git
stash; verify exact via grep+sha256). Clean file shows only 3 pre-existing
`BC.reflective` `reportAttributeAccessIssue` (enum-stub quirk, not S8c).

**The −15 net pyright = −19 disappeared + 4 net-new (RECONCILE EXACTLY).** Method:
back up S8c files (sha256), `git show HEAD:<f> > <f>` to restore the clean
pre-change baseline (S8c uncommitted ⇒ HEAD IS baseline; NOT git stash per L28),
full `npx pyright --outputjson` baseline + current, diff on `(file,rule,msg)` key
(robust to line shifts) THEN confirm with PER-FILE before/after counts. ⚠ the
`(file,rule,msg)` global key gives FALSE net-new when a message's TYPE-RENDERING
text shifts at the same logical error (#257 S8c:
`test_krylov_curvilinear_precond_safety.py` L174 `gains` arg showed as both −1 and
+1 — SAME error, per-file count 4==4 = net ZERO; the `LinearOperator[V@Krylov…]`
render changed). The REAL net-new = +3, ALL in the standalone capture SCRIPT
`tests/sn/_fixtures/wave_t_t3/_capture_pre_t3_snapshots.py` (L191 `aniso.values`
×2 + L204 `np.savez allow_pickle`).

**Root cause = NoReturn→unreachable SUPPRESSION lift (PRE-EXISTING LATENT, not a
regression).** Baseline `apply` (no annotation) inferred `NoReturn`; pyright treats
statements AFTER a `Never`-returning call as UNREACHABLE → suppresses ALL
downstream diagnostics. So line-175 `out=p1_op.apply(psi)` poisoned the whole
`main()` body below it → the LATENT errors at L191/L204 were hidden. S8c's `Any`
base makes the body reachable again → the pre-existing under-typing surfaces.
CLASSIFY: pre-existing-latent, ZERO runtime defect — `build_aniso_source` declares
`np.ndarray | AngularSourceSink | None` but at runtime (scattering_order=1, non-None
psi) returns `AngularSourceSink` which HAS `.values`; `np.savez allow_pickle` is a
numpy-stub `**kwargs` quirk. File is a one-shot capture script (leading `_`, in
`_fixtures/`, `def main()`+`__main__`, no `test_`, NOT pytest-collected). NOT a
blocker; optional follow-up = tighten `build_aniso_source` return or annotate the
script. RULE: removing a `NoReturn`-poisoned dispatcher return UNMASKS every
pre-existing latent error in code downstream of the FIRST poisoned call — expect
net-new ≠ (per-file delta in the two changed files); reconcile globally + classify
each unmasked error as latent-vs-regression by checking it's an under-typed
accessor with a concrete correct runtime type, NOT a real defect.

**cast/ignore hygiene.** 0 new `# type:ignore` (scattering's lone grep hit @645 is
PROSE in the S6 docstring saying "NOT a type:ignore"). 3 honest casts: 1 production
(`scattering.py:1237` `cast("AngularFlux|HarmonicMomentField", psi.bulk)` — runtime
`psi.bulk` is `AngularFlux`, both union members dispatch to `AngularSourceSink`,
verified live) + 2 test sites (under-typed `integrate_angular()→object` / `state.bulk
→BulkField`). NO blocker, NO false-green, NO ERR.

---

## L-051 -- a mechanical API-migration rewire (deleted class → new face) is bit-id-VERIFIABLE not bit-id-ASSUMED: recompute the OLD einsum on a structurally-independent table; and brief-named symbols/files are CLAIMS (two phantoms here)

Task: `MomentProjection`/`HarmonicMomentReconstruction` (orpheus/numerics/projection.py)
DELETED; rewire tests to `quad.angular_frame(L).analysis.apply` / `.reconstruction.apply`
(or `op.frame.analysis` where a `ScatteringOperator` is in scope). The new faces delegate
to `SphericalHarmonicBasis.analyze`/`reconstruct`; brief CLAIMED 0-ULP bit-id.

**Don't ASSUME the brief's bit-id claim — PROVE it before trusting the unchanged asserts.**
The 3 `test_scattering_operator.py` sites carry `np.testing.assert_array_equal` against a
FROZEN `.npz` snapshot captured by the OLD `MomentProjection`. For those to pass unchanged,
`op.frame.analysis.apply == old M.apply` must be BYTE-identical, not just close. Verified
in-process (10-line script): `frame.table == quad.spherical_harmonics(L)` (np.array_equal),
`frame.analysis.apply(psi) == np.einsum("n,nlm,n...->lm...", w, Y, psi)`, and
`frame.reconstruction.apply(c) == np.einsum("nlm,l,lm...->n...", Y, 2l+1, c)` — all True.
The recomputed einsum is the STRUCTURALLY-INDEPENDENT reference (hand-written contraction,
not the production face) — this is the bit-id leg, NOT old-vs-new ULP. Frozen snapshot needs
NO regen: the value is byte-identical (L-049 inverse — here byte-id IS preserved, so the
re-baseline question doesn't arise).

**Two PHANTOMS in the brief — confirm every named symbol/file before editing.** (a)
`SphericalHarmonicBasis.from_quadrature(quad, L).values` (in the snapshot generator's
defensive block) does NOT exist — the brief flagged it; replaced with `quad.angular_frame
(L).analysis.apply`. (b) brief item 6 said "find `_s6_stub_plugin.py` (grep for it)" — it
does NOT exist anywhere (`find` + text-grep for the name both empty); the user's original
grep had conflated it with the `from ...projection import` matches. NOT every brief-named
artifact is real — a `find`/grep confirmation is one cheap call and prevents a fabricated edit.

**Scope discipline on a concurrent-edit task.** User was editing 4 sibling test files
(test_frame/test_spherical_harmonic_space/test_scattering_kernel_crosscheck/
test_projection_operators) — touched NONE. After rewiring, byte-compiled both no-test
generator SCRIPTS (`py_compile`) and exercised the rewired `_capture_legendre_moments`
helper end-to-end (shape `(L+1,2L+1,ng,nx,ny)`) since scripts aren't pytest-collected — a
broken script import is a latent breakage no test run would surface. 100 tests PASS under
`-O`. Pure mechanical rewire, no claim-pushback → no vv-principles/ERR update fires.

---

## L-052 -- a Hilbert-adjoint-via-metric-composition (`A.H = G_dom⁻¹·Aᵀ·G_cod`) is VERIFIABLE by a dense-matrix transpose + the DEFINING inner-product law; the "weight-free transpose" choice is provable, not faith

Frame carve Phase D added `R.H` (reconstruction Hilbert adjoint) to the discrete `Frame`
via a new `reconstruct_transpose` (`einsum("nlm,l,n...->lm...")`, weight-free) + a capability
flip so the generic `_AdjointOperator` composes `R.H = G_basis⁻¹·Rᵀ·G_measure`. VERDICT
SUPPORTED, math correct, all gates have teeth. The reusable adversarial recipe for a
normal↔adjoint operator-algebra change:

1. **Re-derive the adjoint identity by hand FIRST, then numerically prove it.** `R[n,(ℓm)]
   =(2ℓ+1)Yₗᵐ`; the matrix transpose `Rᵀ[(ℓm),n]=(2ℓ+1)Yₗᵐ` is weight-free BY DEFINITION
   (a representation transpose carries NO metric). Compose with metrics: `R* = g_C⁻¹·Rᵀ·w`
   `= ((2ℓ+1)/4π)·(2ℓ+1)·Σ_n w_n Yₗᵐ v_n = (2ℓ+1)²/4π·Σ w_n Y v`. The "weight-free
   reconstruct_transpose" choice is CORRECT, not a missing-factor bug: it is ASYMMETRIC with
   `analyze_transpose` (which DOES carry `w_n`) precisely because each transpose mirrors its
   OWN forward — `analyze` bakes `w_n` in, `reconstruct` does not. A spurious `w_n` in
   `reconstruct_transpose` would give `R*` a `w_n²` and break `⟨Rc,v⟩_W = ⟨c,R*v⟩_{g_C}`.
2. **The structurally-independent reference is a DENSE matrix built by LOOPS, transposed
   directly, composed with metrics by hand** — shares zero code with production's fused
   einsums. `R^T` agreed at 0 ULP, `R.H` at 0 ULP, the closed-form `(2ℓ+1)²/4π·Σw_n Y v`
   target at ~3e-15 (FP non-assoc on the reduction). The DEFINING law `⟨Rc,v⟩_W=⟨c,R*v⟩_{g_C}`
   (Riesz) is the strongest pin — calls `R.apply`/`R.H` on both sides but asserts their
   ALGEBRAIC consistency, NOT circular (it's the adjoint definition itself).
3. **Teeth: 4 mutations, all RED via in-process monkeypatch under `-O`.** drop `(2ℓ+1)` /
   bake a per-node factor / reverse the factor array (`[::-1]`) → all 3 new tests RED;
   wrong GRAM POWER (`metric_per_ell` squared, build a FRESH frame so it flows into the
   space) → both reconstruction-adjoint tests + the analysis `R.H` test RED. Restore → green.
4. **Mode-11 cleared by a sentinel that COUNTS entries into the rewired readers:**
   `R.H.apply(v)` calls `_FrameReconstruction.apply_transpose` ×1 → `basis.reconstruct_transpose`
   ×1 (the new path IS on the gate's call graph); reverting the capability to `{CAP_APPLY}`
   makes `R.H` raise `MissingCapability` (the cap flip is the load-bearing enabler). The
   capability-assert test has teeth: under pytest it REDs with "Extra items in the right set:
   'apply_transpose'" when the face is reverted to APPLY-only (inject the class-attr mutation
   via a `PYTHONPATH=/tmp` pytest-plugin `pytest_configure`, NOT a standalone script).
5. **⚠ METHODOLOGY TRAP I hit (L-010 self-application): a bare `assert` in MY OWN `python -O`
   probe SCRIPT is STRIPPED** — my throwaway `assert rec.capabilities == ...` printed "PASSED"
   while the values were visibly unequal (Mode-8 in the PROBE, not the test). A capability/value
   teeth-check MUST run through PYTEST (assertion-rewriter active in `tests/`) or use
   `np.testing.assert_*` / explicit `if x!=y: raise` in the script — NEVER a bare `assert`
   under `-O`. The test itself was fine; my probe lied.
6. **The bit-faithful-reference choice (rtol=1e-14 per-term fold) is PRINCIPLED, provable.**
   The test's `einsum("nlm,n->lm", Y*f, v)` (2-operand, pre-scaled table) is BIT-IDENTICAL to
   production's 3-operand `einsum("nlm,l,n...->lm...")` (0 ULP, `array_equal` True — rtol=1e-14
   is generous). The REJECTED post-scaled form `f·(S0ᵀv)` drifts 112 ULP (docstring said ~2;
   direction right, magnitude under-stated) because the Σ then runs at ×f-larger magnitude. A
   third independent dense matmul also agrees 0 ULP → the value is right, not just close to one
   reference. Choosing the per-term fold over post-scaling is a bit-FAITHFULNESS choice, NOT a
   tolerance relaxation. Cross-ref [[lessons-L011]] (per-term-fold = structural independence),
   [[lessons-L051]] (recompute the einsum on a structurally-independent table for bit-id).

## L-053 -- migration-review of a `.solve -> .inverse().apply` rewire: the keystone pins the WRAPPER, the migration rewires the LOOP; and the strong catcher may be slow-deselected

(Taxonomy re-evaluation 2026-07-01, branch refactor/inverse-as-operator @ 69ed531 -- F4/W5/W2
legs; distilled by the qa leg, persisted by the main agent.)

1. **Wrapper-surface vs loop-surface.** A keystone gate pinning `inverse().apply(b) == solve(b)`
   covers only the single-call WRAPPER delegation -- bit-identical BY CONSTRUCTION (both sides
   share the same, possibly-buggy, sweep), hence robust to / orthogonal to sweep-correctness
   bugs (#282). It does NOT cover the ITERATION-LOOP per-iterate seed threading -- the surface
   a `.solve -> .inverse().apply(rhs, initial_guess=psi_prev)` migration actually rewires.
2. **Test the loop surface by simulating the exact regression under the ACTUAL run config.**
   Patch the seed-threading helper to drop `initial_guess` in-process and run the canonical
   `-m "not slow"` invocation -- do NOT trust a plan's "test_X must red" claim: the strong
   end-to-end catcher here was `@pytest.mark.slow` and thus DESELECTED (a slow-marker sibling
   of Mode-8: a gate that cannot fire under the run config). The het-2G sphere seed-drop
   (|dk| = 3.46e-2) reddened only a fragile 1G monotone margin under `-m "not slow"`.
3. **A seed-insensitive geometry makes its seeded VALUE gate vacuous for seed-drop detection**
   (cylinder telescopes -- the seed cancels identically). The Mode-11 path-spy on the
   `initial_guess` threading is the load-bearing guarantee there, not any value assertion.
4. **A "fold-config identity" inverse realized as a SCHEDULE** (`B_lower` = octant-group edge
   set, not an algebra operator) **has NO exact round-trip pin** -- no forward `.apply` exists
   to invert against, and the converged-SI-equivalence fallback is Mode-9-DEGENERATE (the
   fixed point is splitting-invariant by construction: it cannot distinguish the fold, or even
   G-S from Jacobi). Minting the forward matvec of the SAME restricted operator (reify
   `M = (L+C-B_lower)`) is the only way to make the round-trip exact.

---

## L-054 -- triaging audit-MISSING `catches("ERR-NNN")`: grep the production RAISE SITE / invariant-enforcement FIRST -- three outcomes, and only one is "add the marker"

(Metadata-only marker-patch task 2026-07-11, branch refactor/sn-walk-unification; 9 MISSING
ERRs -> 5 tagged, 4 reported NO CATCHER.)

The audit lists an ERR as MISSING when no test carries its `catches` marker. That is NOT
automatically "add a marker" -- before tagging OR reporting a gap, grep whether the cataloged
bug is even *reachable*: find the production `raise <ErrorClass>` site (or the enforcement of
the invariant the ERR broke). Four things it discriminates:

1. **Genuine catcher present -> tag it, mutation-verified.** The exact bug re-introduced reds
   THIS test (L-007). Verified empirically for ERR-020 (bit-identity `np.all(vol==vol[0])`;
   edge-derived `**3` round-trip reds it), ERR-031 (positional arg-swap -> the swapped radii
   `[2.0,0.1]` trip the strictly-increasing guard -> `ValueError`), ERR-040 (tangential
   ordinate admitted to a selector -> `test_axis_aligned_ordinates_excluded_from_both_selectors`
   reds). Probe via a `/tmp` script with EXPLICIT boolean prints, never a bare `assert` (L-052).
2. **Catalog's "L0 test" names a RETIRED test class -> the marker didn't migrate.** ERR-020's
   entry named `TestZoneSubdivision::test_equal_volume_*` -- retired in Phase F, MOVED to
   `test_structured_geometry.py` (`test_equal_volume_{cyl,sph}_invariant`), and the marker was
   left behind in a now-STALE comment claiming the decorators "stay". Retirement-means-test-
   migration (L-022 family) applies to MARKERS too: re-tag the successor asserting the SAME
   invariant. This is the usual cause of a MISSING whose catalog "L0 test" still reads plausible.
3. **Typed error defined+exported but NEVER raised = dead scaffolding -> genuine unbuilt-invariant
   gap, report NO CATCHER.** ERR-041/045/047: the error classes ship + export + have a catalog
   entry ("TYPE SHIPPED Wave 3 / catching test ships Wave 7"), but `grep -rn "raise <Class>"
   orpheus/` is EMPTY and the `assert_*` invariant is a no-op default (no concrete override).
   The promissory "Wave 7 catching test" was never built; nothing can red on a recurrence.
   Do NOT invent a marker -- the MISSING is truthful.
4. **`assert_X` delegates to a WEAKER sibling invariant -> it catches the sibling's bug, NOT its
   own.** ERR-042 (`assert_geometry_map_measure_preserving`) `self.assert_is_involutive(quad)`
   and ASSUMES weight-symmetry "by construction" -- so it reds only on non-involution (ERR-044),
   never on the weight-measure drift ERR-042 documents; no Q4.x quadrature test checks
   `weights[ref]==weights` either. Tagging its test `catches("ERR-042")` is the exact L-007
   blind-marker trap (reds on a DIFFERENT class). Report NO CATCHER; the method's name over-
   claims its body (coding-elegance #20).

The pushback rationale for outcomes 3-4 is already covered by the vv-principles `catches`-marker
directive (mutation-verify the EXACT bug reds THIS test) -- no new anti-pattern; this is its
audit-triage application. Marker-only edits: confirm green under canonical `-O` AND that the
`git status` dirt in do-not-touch files is PRE-EXISTING (grep your own ERR numbers out of their
diffs) -- a shared working tree makes every dirty file look like yours.

---

## L-055 -- adjudicating campaign-narration staleness (#304 class): FIX bar = "provably lies about CURRENT code", verified against grep+`gh` BEFORE ruling; two over-fix guards

A doc-hygiene pass over campaign narration (`Phase X` / `Wave Y` / `(#N step` tags in test
comments/docstrings) is a Cardinal-Rule-3 correctness task, NOT a numerical review -- so it does
NOT touch the vv-principles anti-patterns (those are about verification EVIDENCE). The
three-way rule: KEEP-current (open issue's genuinely pending work), KEEP-provenance (truthful
backward attribution / retirement records / plan-pointers -- keep even if the plan file is
gone), FIX-stale (narration that LIES about the present). Conservative default = KEEP.

**The FIX bar is "provably lies about CURRENT code" -- and "provably" means VERIFIED, not
inferred from the tag.** Before ruling a forward-looking deferral ("future X", "blocked on
Phase C", "Phase 2 will land", "once #N lands") stale, verify the *future* against reality:
(1) `grep` the named symbol/closure/wiring/workaround tree-wide -- the strongest FIX signal is a
"future" thing the code now SHIPS + WIRES (a callsite proves it, e.g. `_build_white_hebert_op`
calls `compute_P_ss_cylinder` at a specific line) OR a "pending" workaround that is now 0-hits
(`cast(LinearOperator` gone tree-wide); (2) `gh issue view N --json state` -- an OPEN issue whose
*named sub-phase derivation* landed while a DIFFERENT residue stays open (#112: "Phase A/C
derivation landed, 3-D-normalization/rank-N residue open") means the tag's specific claim can be
stale even though the issue is open. Landed => rewrite to present-tense truth (name the shipped
fn + the still-open residue). Ambiguous / partial-landing (a phase that shipped INFRASTRUCTURE
but the usable capability still raises `NotImplementedError` pending an open follow-on; a bare
campaign phase with no issue number and no confirmable landing) => KEEP + report as orphan-TODO,
do NOT guess.

**Two guards against over-fixing** (both bit me as tempting-but-wrong FIX targets):
1. **A stale line inside a RUNTIME STRING is behavioral -> KEEP even when its text is stale.** A
   `description="... Wave 8 will switch ..."` dataclass field, an f-string diagnostic, an assert
   message -- these are data the code may write to a snapshot / test-ID / error, so editing them
   is a behavioral change. The "never touch runtime strings" constraint PROTECTS you here: the
   one genuinely-stale line in tests/geometry (`_generate_bc_equivalence_snapshots.py:159`) was a
   `description=` field -> untouchable despite the module docstring itself confirming Wave 8
   landed.
2. **A load-bearing-gate "failure here HALTs Phase X" banner is a characterization RECORD, not a
   pending-work lie -> KEEP after Phase X lands.** It states the test's structural importance
   ("this is THE exactness gate; its failure invalidates the whole closure"), which is durable;
   the `Phase X` is provenance, usually paired with a plan-pointer. Rewriting it churns truthful
   history into design vocabulary (the task explicitly forbids that).

Mechanics: comment/docstring-only, ZERO behavioral change; verify `pytest --collect-only` clean
after; a shared working tree means `git diff --stat` shows OTHER agents' production-file edits --
diff ONLY your own touched files to confirm your edits are comment-only (cross-ref L-054's
shared-tree note). (#304 surface-2, 2026-07-22: 277 hits in scope, 3 files FIXed -- a
future-closure-now-shipped, a workaround-now-0-hits, a Phase-2-constructors-now-shipped; the rest
KEEP.)

---

## L-056 -- reviewing a skill->Sphinx DISTILLATION: verify code-anchored specifics against CODE, never against the skill source (the source carries stale specifics that propagate, and the build gate is blind to them)

When a Sphinx page is authored as a faithful distillation of a `.claude/skills/*` doctrine
(e.g. `verification/principles.rst` from `vv-principles/{SKILL,reference}.md`), the DOCTRINE
(ladder/pillars/claim-layers/modes/anti-patterns) is almost always faithful -- reading it against
the preloaded skill confirms mechanism/instance/defense with no inversion (verified the whole
modes-7..12 highest-risk block clean this way in one pass). The yield is entirely in the
**code-anchored SPECIFICS the skill states but the build never checks**: module paths, an
"evaluated in mpmath" vs `scipy.optimize.brentq`-double impl detail, a test-count, an ERR
war-story's composition. Two structural reasons the build gate misses these:

1. **Python-domain roles are not `-W`-gated.** A `:mod:`/`:class:`/`:func:` pointing at a
   NON-EXISTENT target renders as plain text with NO warning unless the build runs `-n` (nitpicky)
   -- so an "exit 0 -W" gate is ZERO evidence a `:mod:` resolves. Verify every code-pointer by
   filesystem/`find`, AND grep the WHOLE corpus for the canonical spelling: the OUTLIER count is
   the bug (caught `orpheus.derivations.continuous.peierls` used 1x on the reviewed page vs
   `...peierls_nystrom` 240x everywhere else -- the bare form is a dead module).
2. **The skill source is not the code and is not build-gated, so its stale specifics propagate
   verbatim into the corpus.** The reviewed page inherited BOTH a dead module path AND a wrong
   impl detail from `reference.md`/`SKILL.md` -- both said the same wrong thing, so cross-checking
   the page against the skill would have PASSED them. "Code outranks doc" means CODE, not the
   trusted skill twin: confirm `mpmath`-vs-`scipy`, `brentq`-vs-`findroot`, `dtype=float`-vs-`mp.mpf`
   by reading the derivation, and read the CONSUMING test's docstring (it often states the truth
   the doctrine page fumbled -- here "double-precision transfer-matrix back-substitution").

A "worked example" whose stated purpose is "every coordinate is TRUE for this case" is a MUST-FIX
magnet: check each coordinate independently (claim-layer / ladder / pillar / tier / operator-form)
-- the classification can be right (semi-analytical, T2, `operator_form=="diffusion"` all held)
while ONE parenthetical mechanism token is false. None of these pushbacks is a vv-principles
anti-pattern (they are doc-accuracy, not evidence-reasoning) -> no skill anti-pattern addition;
this is the distillation-review application of Cardinal-Rule-3 + the retirement-audit "grep docs,
Python-roles silent under `-W`" rule. (#10 stage V5 principles.rst review, 2026-07-23: 2 MUST-FIX
[dead `:mod:` peierls path; "evaluated in mpmath" on a brentq/double reference] + 1 SHOULD-FIX
[carried "twenty one-group tests" where 3 sources say "20 passing tests", a mix]; the ~40-claim
doctrine body otherwise faithful.)

---

## L-057 -- reviewing a results-COMPILATION page (de-freeze + evidence-map + run-book): the count-de-freeze is CERTIFIABLE by live collect, a retitle can beat the test's OWN stale docstring, and a run-book that cites config for "operational detail" can point at a note that CONTRADICTS its headline

#231 task-#10 stage V6 = authoring `docs/theory/verification/summary.rst` (the V&V-part
results compilation) + de-freezing 4 frozen test-counts across the per-method chapters.
VERDICT PASS; the ~50-claim page was faithful end-to-end. Three reusable techniques:

1. **A count-DE-FREEZE is certifiable, not taken on faith.** When a diff removes a frozen
   "N tests across M files" and replaces it with "the auto-generated matrix carries the live
   counts", PROVE the old number was stale by `pytest <dir> --collect-only -q | tail -1`:
   CP 106/6→**154/11**, MC 55→**57**, MOC 102→**104** — all three old literals genuinely
   lied, so the de-freeze is warranted (not an invented rationale). The NEW page states NO
   count (counts-de-freeze doctrine), so a brief's own count claim ("48 rows") is NOT a
   doc-truth defect even when the live table is **47** — but report the delta so the parent
   doesn't propagate the wrong number. Structural counts that SURVIVE (the 27-case CP grid =
   3×3×3, the 4/3/4/1/5 cross-method list lengths) ARE verifiable: `.venv/bin/python -c` the
   list lengths + `ADAPTERS_BY_NAME` (6) at runtime, don't eyeball.
2. **A doc-RETITLE can be MORE accurate than the test's own name/docstring — verify against
   the live ASSERTION body.** V6 retitled the SN property "Flux symmetry"→"Flux flatness";
   `tests/sn/primitives/test_properties.py::test_flux_symmetry` is NAMED "symmetry" and its
   docstring says "must be symmetric about the center", but the LIVE assertion is
   `assert_allclose(flux, flux[0], rtol=1e-6)` ("homogeneous slab flux is exactly flat"). The
   retitle correctly describes the assertion, not the stale name. So a retitle-faithfulness
   check reads the `assert`, NEVER the test name or its (possibly-stale) docstring — same
   "code outranks doc" as L-056, applied to test-name vs test-body. (Diffusion vacuum→Marshak
   was the same shape: the doc states the ASSERTED framing `J⁻=0 @ 1e-12·scale AND
   boundary-cell flux>0`, matching the body, not the old "flux is small" scaffold.)
3. **A run-book that cites config for "operational detail" can point at a note that
   contradicts its own headline.** V6's run-book calls `python -O -m pytest -m "not slow"`
   "the pre-merge gate: the full tree ... single-process" and says the `[test]` extra's
   pyproject notes "carry the operational detail" — but those notes say "The SN suite OOMs
   when run as ONE single-process invocation ... **NO whole-tree single-process run**"
   (per-tier is the memory-safe default). A whole-REPO single-process run executes the whole
   SN tree in one process = exactly what the cited note warns against. RECONCILABLE (the
   pyproject note is inner-loop SN-iteration memory advice; the pre-merge gate genuinely IS
   the full-tree SERIAL run — `reference_test_execution_env` memory: completes 6391/0 in
   ~52 min, xdist UNSTABLE so serial is canonical), and the xdist "within-tier" statement IS
   faithful to pyproject — so it is a NIT (surface the pre-merge-gate vs inner-loop-per-tier
   distinction), not a falsehood. The lesson: when a run-book delegates to a config file for
   detail, READ that file and check the delegated-to text doesn't read as contradicting the
   delegating headline. Not a vv anti-pattern (doc internal-consistency, #231 prime directive)
   → qa-lessons only, no skill edit. Everything else — 6 evidence-map anchors resolving to
   CLAIMED content (not just resolving), the 8-case SN MMS ladder→files, the Mode-12
   homogeneous K=A⁻¹F matrix-object gate, the `compute_kinf_*`-vs-`kinf_homogeneous` footnote,
   the 10 matrix.rst headings, Peierls `precision_digits=30`, the `sentinel` marker "run
   WITHOUT -O", `generate_rst` runnable as `-m` + `reference_values` pkgutil auto-discovery —
   verified faithful. (#231 #10 V6, 2026-07-23.)

---

## L-058 -- a "k is designed-green / functional-blind to the mutation class" (Mode-12) claim is VERIFIABLE by running the mutation — a leaf-transpose-DROP is NOT similar to forward, so its k SHIFTS

#276 A4 phase-gate: the adjoint φ* certification. The test docstrings + the standalone
NOTE claimed "F†=F leaves k EXACTLY equal / k is designed-green on the entire adjoint
mutation class (eig(A†)=eig(A))" and used that to motivate the P1.3-k-vs-P1.4-spectrum
split. **The claim is FALSE and self-contradicted by a passing sibling test.** Verified two
ways:

1. **Direct closed-form check (10-line numpy):** under F†→F on ∞-medium 4G, the daggered
   operator becomes `(Aᵀ)⁻¹F` (F NOT transposed), whose dominant eig = `χ·A⁻¹νΣf = 0.153`
   vs forward `νΣf·A⁻¹χ = 1.488` — k SHIFTS 1.488→0.153 (|Δk|=1.33). The char poly is
   `det(A−Fᵀ/k) ≠ det(A−F/k)` for asymmetric A / χ∦νΣf. Only the CORRECT adjoint
   `(Aᵀ,Fᵀ)` is similar to forward `(A,F)`; a leaf-DROP mutation `(Aᵀ,F)` is a NON-transpose
   operator and its k is unconstrained.
2. **The self-contradiction tell:** `TestP13Mutations.test_fission_role_swap_shifts_k`
   applies the SAME F†=F on `_infinite_medium("4g")` and asserts `|Δk|>1e-6` — and PASSES.
   So the note's "the ∞ row stays GREEN under F†=F, which is why the k-tooth had to ride a
   shifted-k regime" is refuted by the very ∞-medium k-tooth it describes.

**The CORRECT Mode-12 framing** (the fix): `k_adj==k_fwd` is automatic FOR THE CORRECTLY-
BUILT adjoint (eig(A†)=eig(A)), so it confirms the eigenVALUE but NOT the adjoint FLUX SHAPE
(a right-k/wrong-ψ* solver — forward φ, or the νΣf degeneracy — passes the k-legs). THAT is
why the vector gates (spectrum, biorthogonality) are needed. The leaf-transpose-DROP
mutations DO shift k, so the k-legs carry real teeth too; the blind spot is the eigenVECTOR
identity, not "the machinery." **Behavioral rule: NEVER accept a "this functional is blind
to this mutation class" narrative by inspection — RUN the mutation (or the closed-form eig).
A k-blindness claim is right ONLY for mutations that keep the operator a valid transpose;
leaf-drops break the transpose and shift k.** This is the mirror of the #226 step-5b
OVERCLAIM (Mode-12): step-5b claimed a gate CATCHES when it's blind; this UNDERCLAIMS
(claims blind when it has teeth) — same defense (run it, don't narrate it). No coverage gap
(the genuine flux-shape blind spot IS closed by P1.4/P1.5); the finding is a wrong WHY =
should-fix (CR3), not a blocker.

**Skill refinement flagged (read-only task, so surfaced not applied):** the `vv-principles`
Mode-12 "Live application" text carries the same overstatement verbatim — "#276 A4 … 'k*
matches k' carries ZERO mutation coverage on the adjoint machinery." Sharpen "ZERO mutation
coverage on the adjoint machinery" → "cannot confirm the adjoint eigenVECTOR/flux-shape; the
leaf-transpose-drop mutations still shift k." When a review pushes back on the skill's OWN
example, flag the skill edit as a finding under a read-only constraint (main agent applies) —
honors both the read-only task instruction and the self-improvement trigger.

**Other durable micro-findings this review:** (a) an angle-flat-blindness justification is
checkable — `Σχ/W==Σwχ/W when angle-flat` holds IFF W==N; false for lebedev (N=110, W=4π:
8.75 vs 1.0) — but a conservative angle-varying `require` + an in-test wrong-spelling
discriminator make it harmless (doc-precision note, not a teeth failure). (b) `foundation`
stacked on a method under a module `pytestmark=l1` → `PytestUnknownMarkWarning: conflicting
V&V level markers … using 'l1'`; the intended foundation level is silently dropped — a
level-marker hygiene nit distinct from the L-007 foundation+verifies conflation. (c) a
k-only geometry leg with no closed-form (the coupled sphere daggered posing) is honest-scope:
it rides upstream-verified transpose machinery (#280/#310) + k-equality; flag the missing
vector-shape check as a scope boundary, don't credit k as flux-shape evidence.

**A6/ch15 RE-REVIEW SHARPENING (2026-07-25): 0.153 is the 0-D PROXY, not the SN-solve
k-tooth's value (0.171) — the metric carries the angular weight.** The `1.488→0.153
(|Δk|=1.33)` in point 1 above is the 0-D char-poly `eig((Aᵀ)⁻¹F)` (angular-COLLAPSED). The
ACTUAL `TestP13Mutations.test_fission_role_swap_shifts_k` SN daggered SOLVE gives
**k_mut = 0.171 (|Δk|=1.317)**, NOT 0.153 — reproduced 4× via the test module's own helpers,
stable + converged. WHY they differ: `.H`'s metric `G = V·w_n` carries the ANGULAR weight, so
the mutated (non-transpose) fission `F.H_mut = G⁻¹FG` is angularly non-trivial (the mutated
adjoint mode is 21% non-flat across ordinates); the 0-D reduction that yields 0.153 collapses
that angular structure. The QUALITATIVE claim (leaf-drop → k moves O(1)) is right; only the
MAGNITUDE 0.153 is the wrong (0-D) number for the SN k-tooth. **A6/ch15 propagated the 0-D
proxy everywhere it describes the SN-solve k-tooth:** `docs/theory/methods/sn/adjoint.rst:941`,
`docs/theory/verification/sn.rst:4893`, cert-test docstrings `:327`/`:359`, AND vv-principles
`SKILL.md:135` all cite `1.488→0.153` for the SN-solve mutation → should be `0.171` (MUST-FIX;
the number is NEVER asserted — the test only checks `|Δk|>1e-6` — so no gate fails, but the
cited magnitude is a plausible-substitution error: a real eig of a related operator).
**Rule (sharpens the L-058 verify-by-running rule): a cited mutation-MAGNITUDE for a
METRIC-adjoint SOLVE must be the full-solve value (RUN it), never the angular-collapsed 0-D
char-poly proxy — the metric conjugation on a MUTATED (non-transpose) operator is NOT
spectrum-preserving (only the CORRECT full-dagger is similar to forward), so `0-D ≠ SN-solve`
whenever the metric carries a reduced axis (`w_n`).**

---

## L-059 -- an accelerator's PRODUCTION rate can sit ABOVE the operator's spectral radius (a splitting/wall lag); certify the operator by building the FULLY-COUPLED matrix, don't credit "matrix says healthy" from a sibling-BC certificate

#2 DSA 3c rate tier. The reflective-BC gate (D12) split the rate claim: D11's Fourier
bound (ρ ≤ 0.2247c) runs VACUUM (production ρ_est ≈ 0.18 ≈ matrix ρ), while the fully-
reflective regime uses a LOOSER one-sided band (ρ ≤ 0.35) because production measures
0.28-0.31 — attributed to a "Jacobi wall lag" (the production splitting reads the iterate's
outgoing trace one iteration late), NOT a rate bug. The question: honest split or paper-over?

**The decisive check is to BUILD the fully-coupled matrix (error-iteration) operator and
read its ρ directly** — the operator ρ is the floor the splitting can approach; if the
matrix ρ ≪ production ρ_est, the gap IS a splitting artifact (fixable: wall ordering / trace
relaxation), not an operator/consistency bug. Confirmed here:
- production refl/VAC ρ_est = 0.279 vs its matrix certificate 0.154 (D2 report) — the lag is
  real and visible on a config that HAS a committed certificate.
- I built the refl/REFL matrix (both walls resolved IN-sweep by iterating the boundary
  partner fluxes to convergence, then composed with the production low-order via the test's
  own `_t_dsa`): ρ = 0.19-0.22 — HEALTHY. Production refl/refl 0.28-0.31 is the lag,
  confirmed. Split is HONEST.

**The evidence-completeness flag (IMPROVE, not blocker):** the committed instruments
(`_wd_sweep_matrix`, D2 Part C, rate-report Part D) only certify refl/**VAC** and vac/vac —
the `_wd_sweep_matrix` hardcodes a vacuum right wall (its `bc[0]=="reflective"` branch is
DEAD in every test; L-016). So the docstring's "the matrix certificates say the operator is
healthy" for the refl/**REFL** regime that D12 actually gates rested on INFERENCE (mechanism
+ the refl/vac certificate) until the reviewer supplied the refl/refl matrix. When a rate-
split's honesty argument cites "the matrix certifies the operator", verify the certificate
covers the EXACT BC/config the runtime gate exercises — a sibling-BC certificate + "same
mechanism" is inference, and the fully-coupled matrix for the gated config is cheap to build.

**Behavioral rule:** to adjudicate "elevated production rate = splitting artifact vs rate
bug", build the fully-coupled operator matrix for the GATED config and compare ρ_matrix to
ρ_production. ρ_production > ρ_matrix ⟹ splitting lag (honest, characterize + file the
improvement); ρ_matrix ≈ 1 ⟹ genuine consistency failure. Never accept "the matrix says
healthy" when the committed matrix is a different BC than the gate runs. Companion to the
numerical-bug-signatures Sig-8 (unconverged-inner-solve masquerade) and Sig-9 (ρ-blind stop)
— a THIRD "the rate looks off" mechanism: a within-iteration boundary lag elevating the
splitting's ρ above the operator's.

---

## L-060 -- a transpose/adjoint RECIPROCITY gate is Mode-12 blind to a SYMMETRIC completion-drop; a symmetric-completion (E_out diagonal) inverse-fix is VERIFIABLE by dense (Aᵀ)⁻¹=(A⁻¹)ᵀ, and mutation-tested BOTH ways

#2 ERR-071 root fix (the honest full-space composite sweep inverse). The forward outflow-row
is the defect `streamed − ψ_out`, so `(L+C)⁻¹` emits `ψ_out = streamed − rhs_out` via a
post-march restore `buf[out_rows] −= seed[out_rows]`. The transpose half claims: `E_out` (the
restore) is a diagonal partial identity ⟹ symmetric ⟹ `(Aᵀ)⁻¹ = S_oldᵀ − E_out` = the SAME
one-site restore on the SAME forward-sense outflow selector in `solve_transpose`.

**The symmetry argument is directly VERIFIABLE (don't trust the prose).** Build the composite
flatten/unflatten, then dense `A` (apply on unit cols), `A⁻¹` (solve on cols), `(Aᵀ)⁻¹`
(solve_transpose on cols); check `A·A⁻¹=I`, `Aᵀ·(Aᵀ)⁻¹=I`, and — the crux — `(A⁻¹)ᵀ=(Aᵀ)⁻¹`.
Measured 1e-16 on slab/ld_slab/cyl_product INCLUDING the cyl free-DOF subspace (where
`A·A⁻¹` shows `1.0` on the 8 μ_r≈0 rows — A genuinely rank-deficient there, the honest
free-DOF pair, NOT a bug). This proof is independent of the bilinear reciprocity gate.

**⭐ THE Mode-12 FINDING — a reciprocity gate pins the transpose RELATIONSHIP, not
correctness.** `⟨A.solve q,p⟩=⟨q,A.solve_transpose p⟩` (the g3 gate, the named transpose
catcher) is satisfied by ANY genuine transpose pair `(S,Sᵀ)` — so mutation-test it BOTH ways:
- **MUT-T** (fix forward, undo ONLY the transpose completion: additively add `E_out` back →
  `S_oldᵀ`): g3 REDS at O(1) (measured 3.9–7.4%) — the asymmetric catch. ✓
- **MUT-BOTH** (empty the outflow selector → the true pre-fix HEAD state, BOTH halves dropped):
  g3 stays GREEN (5.7e-17) — `(S_old, S_oldᵀ)` IS a transpose pair, so reciprocity is blind.
The forward/symmetric half is caught ONLY by the one-sided identity gate
`test_sweep_inverse_identity.py` (`(L+C)∘(L+C)⁻¹≡I`, reds at 1.8 under the symmetric drop).
So the two gates are NON-REDUNDANT partners; neither alone covers ERR-071. The catalog +
streaming.py note attribute each half to its gate correctly (transpose→g3, forward→identity)
— NO overclaim, but flag for maintainers: never delete the identity gate on "g3 covers it".
**Behavioral rule:** when a fix touches BOTH `solve` and `solve_transpose` symmetrically,
a reciprocity gate is Mode-12 blind to the symmetric regression; require a one-sided
`A∘A⁻¹=I` companion and mutation-test MUT-T AND MUT-BOTH separately.

**Package A (P1-DSA d₁ arm) — clean, SUPPORTED.** (28b) `f₁=−(D/h)Δf₀+a·d₁@ρ=0`:
`moment1_update` bit-exact vs an independent (23f) recompute (max|Δ|=0); `_dh`=D/h,
`_a_coef`=a are the SAME arrays that build a_low/g_map (single-source, so transitively pinned
by the entry-for-entry reference-builder gate `test_dsa_low_order.py`). The (28b) COMBINATION
is not pinned entry-for-entry (no reference builder for the updates) but is end-to-end
CONSTRAINED by the S2-exactness anchor (angular space = span{1,μ}, so one correction must
annihilate the ℓ=1 gain): sign-flip / drop-a·d1 / 3× all blow n from 2 to 49 (Mode-10
mutation-verified — the term is constrained, not merely exercised). Anti-mint confirmed:
`angular_frame(1).table[:,1,1]==mu_x` bit-exact (a CALLED frame row, not a `w·μ` twin).
P0-forced tooth: healthy n=2, forced n=33 (large margin). so=0 is a pure rename of the P0
path (verified vs HEAD). Trace arm ℓ=0 by theorem (derivation reflecting row f₁=0; vacuum
inert). Cross-refs [[lessons-L058]] (Mode-12 verify-by-running), [[lessons-L024]] (prove teeth
by disabling), [[lessons-L016]] (product-quad needed to exercise the μ_r≈0 free-DOF branch).

---

## L-061 -- YOUR OWN mutation needs a BITE CHECK: "the attribute was set" is a presence check, not a bite; and three NEW permanently-inert-gate classes

Enforcement #11 says make the gate RED. The corollary I violated (2026-07-30,
boundary quadrant) is that a mutation that DOESN'T CHANGE PRODUCTION BEHAVIOUR
manufactures a FALSE "no test catches this" finding — the exact Mode-11 trap,
turned inward. Two mechanical traps, both hit in one session:

1. **`__post_init__` install on a dataclass decorated WITHOUT one is a NO-OP.**
   `dataclasses` bakes the `self.__post_init__()` call into the generated
   `__init__` only if the hook exists at DECORATION time. I set
   `AlbedoBoundary.__post_init__ = <alpha -> 1-alpha>`; the plugin's guard
   ("`_APPLIED` is non-empty") passed, the run reported `303 passed` and I very
   nearly shipped "the albedo amplitude is ungated". Re-done at the REALIZER
   seam (`_orig_realize(self, dataclasses.replace(law, albedo=1-a), ms)`) the
   SAME mutation reddens 7. Check `hasattr(cls, "__post_init__")` before ever
   patching one.
2. **A capability REFUSAL is a TWO-part contract.** `LinearOperator.is_adjointable`
   is a DECLARED predicate defaulting `False`; `adjointable(op)` is just
   `return op.is_adjointable`. Adding `apply_transpose` alone does NOT lift the
   refusal (measured: still `False`) — you must ALSO override the predicate. Same
   for `is_invertible`/`is_assemblable`. A plan step "expose the transpose" that
   touches only the method is a no-op; say so.

**Recipe**: every mutation in the plugin registers a `bite()` callable run in
`pytest_configure` that exercises the production path and RAISES if the
observable is unchanged. For a driver-level bug (ERR-052 renorm-drop) the bite
must instrument a COUNTER + a magnitude (`renorm_calls 6->0`, `|phi|max
7.60->0.61`), because the observable the TEST reads can legitimately be unmoved
— which is the finding, not a failed bite.

**Three inert-gate classes to grep for, all found in one subsystem:**
- **Tautological raise**: `with pytest.raises(X): raise err` where `err` was
  built as an `X` two lines up. No input reds it. (`test_bc_errors.py`: 9 legs;
  confirmed by measurement — 0 of 14 guard-disabling mutations touched the file.)
- **`except Exception: pytest.skip(...)`**: converts ANY construction bug into a
  green skip forever. (`test_bc_extraction_matvec.py:445` — a self-described
  "SENTINEL", 3 rows, never run: the `try` builds a 1-D mesh then reads
  `spatial_shape[1]` -> `IndexError` swallowed. Tell: a skip REASON that is an
  exception message rather than prose. Always run `-rs` on a suite whose skip
  count is non-zero and READ the reasons.)
- **STALE `catches("ERR-NNN")`**: a marker that WAS a true catcher and drifted out
  of the failure regime. ERR-052 (power-iteration renorm) needs 30-60 outers to
  denormalise; the test's config now converges in **6**, and its assertion is an
  ORDERING with a 10x margin (`1.875 > 0.179`) — bite-verified bug re-introduced,
  test still green. Sharpens L-007/L-054: a `catches` marker is not verified ONCE,
  it decays; re-verify when reviewing the area.

**Adjacent level-conflation tell**: a snapshot file marked `l1` whose own header
records that the cross-implementation half was deleted and "the snapshots now
record realiser-path outputs" — a self-generated regression baseline wearing an
L1 label. It is still the WIDEST net (reddened 9 of 12 leaf mutations); the fix
is the marker, not the file. Same file: `_load_or_skip` SKIPS on a missing
snapshot instead of failing — tighten to a hard fail.

**The good news worth remembering**: guard-disabling is cheap and fast (neuter
each `assert_*` invariant / realizer refusal one at a time via a `-p` plugin,
~2 s per run over an 18-file set). The boundary subsystem scored 30/31 caught.
Cross-refs [[lessons-L024]] (prove teeth by DISABLING the override),
[[lessons-L007]] + [[lessons-L054]] (`catches` = coverage CLAIM),
[[lessons-L010]] (settle Mode-8 fast, then pivot to what the asserts assert).

---

## L-062 -- auditing a GROUP-THEORY / symmetry module: the sample-generates-the-group check, the partner-vs-bijection check, and the order-relation x predicate MONOTONICITY law (one loop, three defect classes)

`orpheus/numerics/symmetry.py` pre-carve audit (2026-08-02, ahead of widening
`_orbit_closure` to return its orbit permutation). Two CRITICAL false-certification
defects, both filed (ERR-072, ERR-073); both were unreachable by the shipped
182-test suite. The three reusable moves:

1. **Compute the group the SAMPLE generates, then compare.** A "representative
   orbit" check is sound iff the listed matrices GENERATE the claimed group (then
   per-generator closure implies closure under every product; weight-preservation
   composes). `_so2_representatives()` returned `{0,90,180,270}deg` about z ->
   generates `C_4`, NOT `SO(2)` -> every `product(n_mu, n_phi=4k)` certified
   `SO(2)`-invariant while being only `C_8`-invariant. Probe is 6 lines: BFS the
   group generated by the returned ops and print `len`. Did it for ALL 8
   generators in one table (orth err / dets / uniq / generated order / closed?) —
   `_octahedral_ops` (48, 24+/24-, closed) and `_icosahedral_ops` (120, 60+/60-,
   order census {1:1,2:31,3:20,5:24,6:20,10:24} = textbook `I_h`) came back
   CLEAN, which is itself a useful result to report. **Two tells travel with the
   defect**: (a) a docstring that PRE-AUTHORISES the gap ("necessary but not
   sufficient in general, sufficient by construction for the rules we ship") —
   check the enumerated rules one by one, the product family failed; (b) for a
   CONTINUOUS group the honest discrete predicate is a DIFFERENT question (finite
   set is `SO(2)`-invariant iff every node is ON the axis), so the declared tag
   describes the CONTINUUM being discretised — two claims sharing one predicate name.

2. **"Found a matching partner" != "is a bijection".** `_orbit_closure`'s
   docstring says "find a permutation pi"; the body finds SOME `j` within
   tolerance per `i` and never checks injectivity. Killer demo needs NO tolerance
   games: append a BIT-IDENTICAL duplicate of node 0 to an `O_h`-invariant LS(4)
   rule -> `is_invariant(O_h) = True` while `mass at p0 in mu = 1.047` vs
   `in M#mu = 0.524`; match map non-injective for 48/48 ops. Adding the bijection
   check is FREE for every shipped rule (verified 8 rules) -- so it is a pure win,
   not a tradeoff. Generalise: a docstring naming a STRUCTURE (permutation,
   bijection, partition, basis) that the body only implies -> assert it or weaken
   the prose; and prefer RETURNING the structure to a `bool` about it (a returned
   permutation makes its own bijectivity assertable; the carve's orbifold singular
   set is just `{i : perm[i]==i}`, underivable from the bool).

3. **The MONOTONICITY law is the single highest-yield gate for any module that
   ships an order relation AND a predicate respecting it.** `A <= B and P(B,x)
   => P(A,x)`, looped over every (edge x fixture): 68 violations over 11 measures
   x 19 groups, isolating THREE independent classes at once — a sampled-group
   checker (1), a FALSE lattice edge (`D_nh <= O(2)`, false under BOTH embeddings
   of O(2), and PINNED by a committed test at `test_symmetry.py:216`), and a
   realisation mismatch (`Z2 <= SO(3)` asserted while `Z2` is realised as an
   improper `det=-1` reflection). No per-relation and no per-predicate test can
   see any of them. Companion cheap invariants on the same object: reflexivity,
   antisymmetry (found `Trivial`/`Cn(1)` aliasing), transitivity (3 violations
   from the same alias) — all `itertools.permutations` one-liners.

**Method notes.** (a) A "which STRATEGY decides this tag?" table is built by
wrapping the shared kernel and recording `len(ops)` per call — it exposed that
the whole 1-D path NEVER calls `_orbit_closure` (so a permutation-returning carve
has a SECOND job it did not know about) and that `SO3` and `O3` pass the SAME
120-op set (`-I` is already inside `_icosahedral_ops()`), making
`SO3.is_invariant == O3.is_invariant` identically. (b) **A 0-call mutation counter
is a FINDING, not an inert mutation** — `_so2_representatives` called 0 times
across 182 tests means the path is unreachable by the suite; prove it by showing
the call count go 0 -> 1 when you hand it the right fixture. (c) The strongest
coverage result came from CRIPPLING a generator rather than a value: replacing
`O_h`'s 48 ops with its 8 diagonal sign-flips (= `D_2h`) left the suite 182-green;
bite proof = all sign combinations of `(0.6,0.8,0)` (closed under sign flips,
NOT under x<->y). (d) zsh does NOT word-split an unquoted `$VAR` — a `for M in
...; do pytest $SUITE -p $M; done` loop silently passed the whole path list as ONE
argument and every run produced no tests and a 0 counter; use a shell FUNCTION
with `"$@"` and always print the baseline `N passed` line in the same loop.
Cross-refs [[lessons-L024]], [[lessons-L058]] (Mode-12 verify-by-running),
[[lessons-L061]] (bite check on your own mutation).

## L-063 — a retired claim over "not-X" carries as many clauses as the partition has classes; and an over-powered mutation over-states coverage 60x

**Context.** `tests/sn/operators/test_native_matvec.py` pin 5 read "face residual
zero at NON-OUTFLOW ordinates (inflow ords get their value from the BC)". Wave O
#208 O.4a.2 inverted the inflow half — the live gate
`test_outer_face_inflow_slots_carry_the_identity` asserts `out[inflow] ==
psi.inflow`. Commit `b4984773` corrected the prose. Question posed: is the old
claim obsolete, or did it describe a still-true property that got silently dropped?

**Finding 1 — decompose the quantified set BEFORE ruling.** The SN face partition
is THREE-way (inflow ⊔ outflow ⊔ tangential, `|Ω·n| <= TANGENTIAL_EPS =
4*eps`), so "non-outflow" is TWO clauses with INDEPENDENT fates:
- inflow clause: **inverted** (obsolete). Mutation-verified live: emptying the
  inflow index set (= re-introducing the old claim verbatim) reds 4 gates; a
  mask swap reds the same 4.
- tangential clause: **still exactly true**. Production
  (`loss_representation/__init__.py:1162-1172` multi-D, `:3355-3378` 1-D)
  allocates `AngularBoundarySourceSink.zeros_on` and writes ONLY `out_idx` /
  `in_idx`, so tangential rows stay at zero by construction. Measured
  `array_equal(out[tan], 0)` True on cylinder `product(2,4)` (4/8 tangential),
  `product(4,8)` (8/32), sphere `product(4,4)` (8/16).
A prose fix that re-writes the whole sentence around the inverted clause silently
retires the surviving one. **When a retired claim quantifies over a complement
("non-X", "not inflow", "everything else"), enumerate the partition first.**

**Finding 2 — the pin file cannot express its own surviving clause.** All three
fixtures (`gauss_legendre(4)` slab + sphere, `level_symmetric(4)` cylinder) carry
**zero** tangential ordinates: the mutation bit 0 rows over 23 apply calls.
GL-at-even-order is the ONLY production quadrature with no tangential ordinate
(`numerics/operator.py:2533`). So the file's fixture set is exactly the blind spot.

**Finding 3 (the expensive one) — an over-powered mutation over-stated coverage
by 60x.** First mutation wrote a CONSTANT `SENTINEL=7.25` into the tangential
output rows: **60 new reds** over 5076 tests (kinf analytical, streaming
equilibrium, MMS, sweep-inverse, DD regression). Reading that as coverage would
have been wrong: a constant makes a LINEAR operator **affine**, and every one of
those 60 is a Krylov/SI solve that diverges when its operator stops being linear.
Re-run with the realistic LINEAR bug — `out[tan] = ±ψ[tan]`, what you get by
initialising the output block from the input, or from the documented "not inflow
== outflow" trap (`streamed[tan] - given[tan] = -given[tan]`) — the SAME 94 148
rows over 9 949 meshes reddened **exactly 1** test, for BOTH signs. → new
`vv-principles` anti-pattern **#18** (the dual of #17: #17's broken harness lies
"0 caught"; an over-strong mutation lies "richly caught", and an audit closes on it).

**Finding 4 — why only one catcher CAN exist.** The trace metric
`G = |Ω·n|·w_n` is *exactly* 0 on tangential rows: a `1e6` perturbation there
moves `⟨x,y⟩_G` by `0.0` — bit-identical. And the rows are decoupled: varying the
tangential INPUT by `1e3` leaves the bulk `(L+C)ψ` and every other face row
**bit-identical**. So every G-weighted gate (reciprocity, duality, norms,
pseudo-inverse round-trip) and every solver-level observable is designed-green
(Mode 12) — a direct array assertion is the only instrument that can exist.
`numerics/spaces/full_field_space.py:47-53` states the property and relies on it
for the Moore–Penrose adjoint being exact.

**The one catcher.** `tests/sn/operators/test_sweep_inverse_identity.py::
TestSweepInverseIdentity::test_forward_of_inverse_is_identity_on_a_random_composite[cyl_product]`
— asserts `back.boundary.face_view(face)[degenerate] == 0` where
`degenerate = setdiff1d(arange(N), union(inflow, outflow))`. Its `cyl_product`
fixture (`product(n_mu=4, n_phi=8)`) is documented as "the #280 MANDATORY cylinder
config: a PRODUCT quadrature carries degenerate pure-azimuthal ordinates".
Residual weakness: that branch is `if degenerate.size:` with NO non-vacuity guard
— the file's `pytest.fail("no live trace rows")` guards `n_live`, not
`degenerate`. A fixture drift to `level_symmetric` silently deletes the tree's
only catcher with zero signal.

**Pins 4 / 6 / 7 (same file, same question).** All three genuinely obsolete.
(4) `boundary.xmax_face` → `face_view("xmax")` is an accessor rename; the
attribute no longer exists anywhere and the shape contract it named is asserted
live in `TestOutputShape`. (6) the `eq_map` cross-check partner is gone from
`orpheus/` entirely (`EquationMap`/`face_outer_ordinate`: 0 hits in production,
prose-only in tests); the residual "does the matvec use the right mask" is
mutation-verified — both the empty-inflow and mask-swap mutations red 4 gates.
(7) the `NotImplementedError` guard's removal is the point of the change; real
2-D correctness lives in the MMS-2D + T4b snapshot suites, not here.

**Method notes.** Baseline carried 7 pre-existing reds (in-flight quadrature /
geometry campaigns) — always diff, never count. Two direct `loss_action` callers
bypass an `.apply`-level mutation (`test_one_octant_walk.py:149` discards the
result; `test_ld_adjoint_deferral.py:425` asserts only `interior.values`), so
neither hides a catcher. zsh `$SUITE` word-split bit AGAIN (L-062 (d)): the first
baseline run collected zero tests and reported `1 warning in 0.01s`.
Cross-refs [[lessons-L061]], [[lessons-L062]] (bite check, positive control),
[[lessons-L058]] (verify a blindness narrative by RUNNING it).

---

## L-064 — a single-sourcing retirement demotes a cross-check on the INPUT-RESOLUTION axis, invisibly; and the retirement's own CONCEPT grep missed the gutted package's sibling docstrings

**Dispatch.** Review of the two-commit retirement of `Quadrature.reflection_index`
on `refactor/operator-strategy-layers` — `c7ca338e` (test tier migrates onto
`tests/_harness/references.mirror_partner_indices` or onto
`quad.ordinate_permutation`, discriminated by claim class) and `b5ac130e` (the
table, `reflection_partners`, `_compute_sphere_reflection_partners`, the GL1D
closed-form dict and `_resolve_axis_to_index` all deleted; the Q4 gates re-posed;
the equation node renamed `quadrature-reflection-index` →
`quadrature-ordinate-permutation`).

### (a) The migration itself is sound, and the mutation battery says so

Baseline on the seven reference-class files: `331 passed / 1 xfailed`.

| mutation (in-process plugin, `python -O`) | result | reads as |
|---|---|---|
| **M1** `Quadrature.ordinate_permutation` returns a 0↔1-swapped (still bijective) π | 115 failed / 216 passed | OVER-POWERED (#18): the swap breaks measure-preservation, so most reds are `BoundaryGeometryMapNotMeasurePreservingError` at `assert_realizable` — the LAW I broke, not the partner map |
| **M2** `TraceRestrictionOperator.to_local` → naive `arange` (certification untouched, operator still a valid bound Γ₊→Γ₋ permutation of the right length) | **21 failed** | IN-CLASS. The reds are exactly the reference-class gates: `test_sn_boundary_realizer` α=1 + α=0.7 hand-computed rows, `test_reemission_closure::TestSpecularAgainstAnIndependentExpression` ×6, `test_boundary` ×3, `test_bound_compat` ×1, `test_snmesh_realizer_wiring` ×2, `test_b3_domain_narrowing::TestBitIdentityAgainstTheRetiredExpression` ×6 |
| **M1** on the PRODUCTION-DATUM tier (coupled-pole + azimuthal controls) | **16 of 17 failed** | the datum gates assert INVARIANTS of the datum, so a derivation drift reds them — the class is correctly chosen |
| **M4** `_ensure_pole_mirror` refusal removed (identity fallback) | new refusal pin RED, `DID NOT RAISE <ValueError>` | the new pin has teeth |
| **M5** deck kernel's `codomain=` binding dropped to `None` | 5 failed — ALL in `TestPeriodicIsBoundToThePartner`; the split gate stayed GREEN | see (c) |

`[M]` the new helper's structural independence is provable, not argued:
`dis.Bytecode(mirror_partner_indices).codeobj.co_names` =
`{AssertionError, _AXIS_INDEX, arange, argmin, asarray, axis_cosines,
column_stack, copy, float, int, isinstance, len, linalg, max, norm, np, range,
set, str, tolist}` and the module's only import is numpy. Every occurrence of
`ordinate_permutation` / `preserves` / `_orbit_closure` / `RigidMotion` in that
file is DOCSTRING prose. `[M]` reference-vs-production agree on 30/30
(rule × axis) pairs, and the match residual is **exactly 0.0** on every shipped
rule — a signed coordinate permutation is IEEE-exact — so the helper's `1e-12`
window (tighter than production's `1e-13 × 100 = 1e-11`) carries unbounded
headroom and is NOT the #16 latent-false-red it superficially resembles.

### (b) ⭐ THE FINDING — independence has TWO axes and single-sourcing closes one silently

`tests/geometry/test_specular_response_pins_to_geometry.py` says, in its module
docstring: *"the two sides are derived by genuinely independent routes … Neither
consults the other."*

- **Before**: geometric side = `SelfPairedDeck.mirror(axis).motion` applied to the
  nodes with a local argmin; response side = `quad.reflection_index(axis)`, which
  resolved the LETTER through the quadrature tier's own `_resolve_axis_to_index`
  → the table built from `RigidMotion.reflection(normal=np.eye(3)[axis])`. Two
  tiers each resolved "x" independently ⟹ the gate cross-checked the
  axis-letter → mirror-normal CONVENTION.
- **After**: the test builds ONE `SelfPairedDeck.mirror(axis).motion` and passes
  the SAME object to `quad.ordinate_permutation(motion)`. The two *derivations*
  are still independent; the *input resolution* is now shared.

`[M]` **M3** — `_mirror_motion` maps the letters x↔y:
`test_specular_response_pins_to_geometry.py` is **15/15 GREEN**, while **78** of
261 sibling gates red. The file is now exactly blind to a class it used to catch.

The class survives tree-wide only by a deliberate design choice in the new helper:
`_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}` is a LOCAL literal, with a comment
refusing to import `face_layout.AXIS_NAMES` ("a convention drift between the two
IS a defect these gates should surface, not absorb"). That instinct is what kept
the 78.

⟹ generalised as `vv-principles` anti-pattern **#22**. The
`coding-standards` rewire-demotion clause only covers the *caller* case (survivor
calls the other side); this is the *shared constructed input* case, where neither
calls the other and the gate is still demoted. Both are invisible in a diff.

### (c) A tombstone that names two carriers may be naming one carrier per LEG

`b5ac130e` deleted `TestTheSpecularArmInheritsTheRetiredTable` (2 rows) and left a
comment: row (b)'s binding claim "is asserted independently by the
self-paired/paired split gate at the end of this module
(`mirrored.domain is trace.outflow_space("xmin")`) and, for the load-bearing
off-diagonal case, by `TestPeriodicIsBoundToThePartner`."

Read as belt-and-braces. It is not. `[M]` **M5**: the split gate's codomain
assertion is `mirrored.codomain is realized.codomain` — **`None is None`-satisfiable**,
so it reds for a *dropped* codomain binding exactly never; the mirror arm's
codomain-by-`is` leg rides entirely on the transitive identity through
`TestPeriodicIsBoundToThePartner`. **When a tombstone lists N carriers, split the
retired claim into its LEGS and ask which carrier covers which leg** — a
`x is y` assertion between two SUT-produced values is vacuous when both can be
`None`, unlike the `x is <concrete object>` form beside it.

### (d) The CONCEPT grep must cover the gutted package's own SIBLING docstrings

`b5ac130e`'s message credits a CONCEPT grep for catching three present-tense
"reflection-index table" diagnostics in `_errors.py` ("the symbol grep cannot see
hyphenated prose"). It grepped the retired SYMBOL's hyphenation and not the
retired FIELD's — `reflection-partner map` / `reflection partners` — so four
present-tense-FALSE claims survived **inside `orpheus/numerics/quadrature/`, the
package the commit gutted**:

- `quadrature/__init__.py:38` — "the SN-side derived data (reflection partners,
  octant partition, level structure) **cached at construction time**"
- `rules_sphere.py:97` / `:576`, `rules_product.py:310` — "It wraps this measure
  **and precomputes the reflection-partner map at construction**" (the same
  commit rewrote all three factories to `cls(measure=…, level_structure=…)`)

Plus one missed site in a file the commit DID edit —
`curvilinear_numerics.rst:2280` "as a property of the quadrature's reflection
table itself" (lines 2213 and 2241 of the same page WERE updated: anti-pattern
#21's half-done-correction shape) — and two present-tense-false test docstrings:
`test_sn_boundary_realizer.py:316` (summary line still `psi[reflection_index]`
while its α=1 SIBLING 55 lines up was updated) and
`test_bc_equivalence_snapshot.py:435` ("never read from
`quadrature.reflection_index`, **which is the table production consults**" —
the negation is fine, the relative clause is false).

⚠ **No build of any severity could see the four package ones**: there is no
`automodule:: orpheus.numerics.quadrature*` anywhere in `docs/`, and
`tools/check_docstring_xrefs.py` (DEAD TARGETS 0, correctly) checks xref TARGETS,
not prose truth. **Grep was the only gate, and the grep's vocabulary was the
retirement's blind spot.**

### (e) Level-marker note, PRE-EXISTING, worth carrying

The renamed equation node has degree 29 / ~21 incoming `tests` edges, ALL from the
file-level `verifies(...)` list on `tests/sn/primitives/test_quadrature.py` — of
which only `TestReflectionIndices::test_x_reflection` and
`::test_reflection_involution` touch the permutation at all; the rest are
weight-sum, second-moment, α-dome and scattering-source rows. Meanwhile the
STRONGEST gates for that equation (`tests/numerics/test_quadrature_directional.py`
Q4.2/4.3/4.5/4.6/4.7/4.8) carry `pytestmark = [pytest.mark.foundation]` and
deliberately no `verifies` — correct level discipline (E1), but it means the audit
credits the wrong file. The rename inherited this; it did not create it. `#20`
(count CASES that read the varied thing, not rows) applied to `verifies` edges.

Cross-refs [[lessons-L063]] (over-powered mutation), [[lessons-L044]], [[lessons-L051]]
(two independent implementations IS independence), [[lessons-L056]] (Python-domain
roles are not `-W`-gated), `vv-principles` #21/#22.

---

## L-065 — an A-vs-B INVARIANCE gate's coverage is the set of production lines that READ the knob; the catastrophic positive control is INVALID for it; and MOVING a method to a sibling object can convert a self-consistency into an extent-only-guarded coupling

**Dispatch (2026-08-07).** Adversarial review of the SN **G6.5** carve —
`0d99140c` ("the half-trace SPACE owns its local↔global index map") +
`619a873d` ("the axis contract gets ONE name; the packing-order acceptance
gate"), branch `refactor/operator-strategy-layers`. Five review dimensions:
retirement completeness, the new deck-arm refusal's blast radius, the new
acceptance gate's teeth, the new `__post_init__` guard vs real producers, and
prose truth.

### Baselines measured (so a later session does not re-derive them)

| scope | result | time |
|---|---|---|
| `tests/numerics/test_angular_face_trace_space.py` + `test_trace_restriction_operator.py` + `tests/sn/operators/` | 1301 passed, **2 failed** (the declared cart2d pair), 1 skipped, 6 xfailed | 21 s |
| + `tests/geometry/` | 2074 passed, **3 failed** | 30 s |
| `tests/test_docstring_xrefs.py` + `tests/test_pyright_ratchet.py` | 4 passed | 77 s |

The **third** geometry red — `test_bc_equivalence_snapshot.py::
TestWhiteXminPartial03GLSnapshot::test_matches_the_frozen_scaled_lambertian`,
max rel `1.1e-15` at `rtol=8.88e-16` — is **PRE-EXISTING**: reproduced red at
`754d384e` in a read-only worktree (`git worktree add -d /tmp/g65_pre
754d384e`; resolution verified via `orpheus from:
/private/tmp/g65_pre/orpheus/__init__.py`, lessons H4). A brief that declares
only *some* baseline reds is declaring the reds of the batteries IT ran — widen
the scope and you inherit reds it never saw. Reconcile them against the parent
commit before attributing anything.

### 1 — the invariance gate: coverage = the knob's READERS

`TestFacePackingOrderIsBookkeeping` realizes `{specular, lambertian, periodic,
vacuum} × {gauss_legendre(8), product(4,4)}` on two `FaceLayout`s differing only
in face order and asserts `np.array_equal` on `apply` and `.H.apply`. It reads
like a broad acceptance criterion. It is not: **an A-vs-B invariance gate
compares two runs of the SAME code, so it is blind by construction to every
mutation that is not knob-dependent.** Its reachable coverage is exactly the set
of production lines that READ the knob — here, *one*:

```
orpheus/numerics/spaces/angular_trace_space.py:585
    face_metric = slot.slice_view(metric_flat).reshape(n_ordinates, -1)[:, 0]
```

(the matching write is :302; nothing else in the realization path touches a flat
offset — `grep '\.offset\|slice_view'` over `angular_trace_space.py`,
`realizer.py`, `sn/operators/boundary.py` returns those two plus `_face_row`).

**And the fixture annihilates it.** `_FORWARD = ("xmin","xmax")` puts both faces
on the SAME AXIS, so `|Ω·n| = |μ_x|` for both and the two slots' metric slices
are bit-identical. Measured with an in-process `_face_spaces` replacement that
reads the metric at flat offset 0 (the wrong slot — still a real face's metric,
right length, positive ⟹ in class):

| mutation | rows red / 10 | note |
|---|---|---|
| `vacuity` — `face_trace` ignores the face ORDER | 2 | both activation rows ⟹ leg live, non-vacuous, bare `assert` NOT `-O`-stripped |
| `perm_roll` — deck perm shifted iff the slot is not at offset 0 | **4** | `specular`×2 + `periodic`×2, via the bit-identity assertion. Real teeth. |
| `metric0` — wrong-slot metric read | **0** | bite log: the mutated read ran on EVERY face at EVERY offset, `changed=False` every time |
| `facerow0` — the ω·n ROW index read as the layout head | 9 | OVER-POWERED (#18): 6 are orientation-guard raises, 2 activation, only 1 bit-identity red |

Counterfactual, same mutation with a y-face in the layout
(`("xmin","xmax","ymin","ymax")` vs `("ymin","xmax","xmin","ymax")`,
`product(4,4)`): `Γ₊(xmax)`'s weights move by **max |Δw| = 0.963** — O(1).
`gauss_legendre(8)` structurally cannot carry y-faces (`build_omega_dot_n`
refuses: "every ordinate has mu_y == 0"), so the fix belongs on the
`product(4,4)` arm.

⭐ **The control trap, and it cost a run.** My first positive control was the
usual catastrophic one — replace the deck permutation with the identity. The
gate stayed **10/10 green, correctly**: both runs got the same wrong answer.
For an invariance gate the vv #17 "one mutation must redden many gates" control
is *invalid*; the control must itself be **knob-dependent** — neuter the knob so
A and B become the same object and require the ACTIVATION leg to red. A second
own-goal in the same family: `np.roll(perm, off % perm.size)` with offsets
`{0, 8}` and `perm.size == 4` folds to `0` in BOTH arms — a silent no-op. The
bite log (`changed=True/False` per call, written from inside the mutation) is
what caught both; A4 again.

⟹ `vv-principles` **#23** (written this dispatch).

### 2 — moving a method to a sibling object can create an extent-only-guarded coupling

`_deck_kernel` used to compute `local_perm = gamma_out_domain.to_local(…)` — the
**restriction operator's own `indices`**, which is precisely the array that
determines the row order `gamma_out.apply` emits. Post-carve it is
`gamma_plus.to_local(…)` — the **codomain SPACE's `ordinate_indices`**. And
`TraceRestrictionOperator._checked_space` → `checked_space_extent` compares only
the **extent**, never the index set elementwise.

On the trace's own cached pair they agree (measured: equal values, `is` False,
`shares_memory` False) and the carve's new
`test_the_restriction_and_the_space_carry_ONE_index_set` pins exactly that pair.
**But the deck kernel never consumes that pair** — it consumes the realizer's
LOCALLY-BUILT `_outflow_restriction` / `_partner_outflow_restriction`, whose
`indices` and `codomain` come from two different sources (and for periodic,
deliberately independent derivations, which the docstring calls a feature).

Measured, `product(4,4)`, hand-built method space with a trace and a
same-size-but-different `outflow_indices` (passes every guard):

```
operator (gather order): [0, 1, 4, 8]
space   (declared)     : [0, 4, 8, 12]
post-carve local_perm (from the SPACE)   : [0, 1, 2, 3]   <-- ACCEPTED, wrong
pre-carve  local_perm (from the OPERATOR): refusal — "row 12 is not in this
                                            restriction's index set"
```

The pre-carve code **refused** with its own documented crossed-index-set
diagnosis; the post-carve code **accepts and emits a permutation against a row
order the gather does not produce.** Not reachable from production (the
canonical `for_face` derives both from the trace), but reachable from the
hand-built method space the code explicitly supports (`realizer.py:246`).

⭐ And the one gate that LOOKS like it cross-checks the two cannot:
`test_deck_kernel.py::test_the_LOCAL_remap_is_not_arange`'s "independent"
reference is `searchsorted(sort(gamma_out.indices),
gamma_out.indices[kernel.perm])` — a **round-trip through the same array**,
which returns `perm` for any `perm` (lessons B2). Its real catch is the second,
activation-shaped assertion `not array_equal(perm, arange)`. Pre-carve the
round-trip was harmless (production shared the array); post-carve it is the only
place the two arrays meet and it is structurally blind. **A round-trip that was
harmless while one array existed becomes the gap the moment a refactor makes it
two.**

Fix proposed: make `_checked_space` compare `indices` elementwise against the
codomain's `ordinate_indices` when the space carries them — one place, every
call site, and it turns the tombstone's "the same array by construction" into a
checked fact.

### 3 — a campaign-step name in a forward-looking docstring is a self-expiring token

Two survivors, both verbatim in `754d384e`, both falsified by this carve's own
landing:

- `orpheus/numerics/operator.py:2475` (`checked_space_extent`, a SHARED
  production primitive): *"The redundancy is transitional: **G6.5 retires the
  lengths** in favour of the spaces (#330)."*
- `tests/numerics/test_angular_face_trace_space.py:741`: *"…until **G6.5 retires
  the former**."*

G6.5 shipped and deliberately did NOT retire them — and the carve KNEW, because
it rewrote the sibling claim **146 lines below in the same file**
(`operator.py:2621-2626`) to "*until the tree-wide mandate (#330) … G6.5,
2026-08-07, measured why the retirement cannot land sooner*". vv #21's
aggravator in its purest form: the stale claim and its correction now coexist in
ONE FILE, so a contributor can cite `operator.py` for either.

The cheap mechanical rule: **when a campaign step lands, grep the step's own
name.** `grep -rn 'G6\.5' orpheus/ tests/ docs/` filtered to the
forward-looking forms (drop `since G6.5` / `at G6.5` / `(G6.5)` / `— G6.5`)
returned exactly the two survivors out of 33 hits, in one command.

### 4 — what PASSED, with the measurement

- **Deck-arm refusal blast radius: zero consumers.** An in-process
  `_deck_kernel` wrapper logging every entry + the guard predicate over
  `tests/sn/operators` + `tests/geometry` + the two numerics batteries:
  **1460 entries, 3 with the predicate TRUE**, and all three are
  refusal-expecting rows. Bonus: one of them is spaceless AND lopsided and
  still gets the BIJECTION message ⟹ the "guard placed AFTER the size check"
  claim is empirically confirmed, with no mutation needed.
- **The `__post_init__` guard refuses no legitimate producer.** Only ONE
  producer exists (`AngularTraceSpace._face_spaces`); sweeping 8 quadrature
  families × 5 face-sets × 3 tiers = **240 canonical spaces, 0 refusals**, plus
  `SNMesh` end-to-end over 1-D Cartesian / 1-D spherical / 2-D Cartesian ×
  {reflective, vacuum}. No base `__post_init__` is shadowed; nothing
  `replace()`s the type; empty tiers are guarded (`if idx.size and …`).
  Teeth: disabling it reds **exactly one** test and nothing else.
- **The migrated `to_local` gates kept their teeth.** `to_local → arange` reds
  **48** tests — all 4 migrated space-battery gates plus `test_b3_domain_narrowing`
  ×6, `test_deck_kernel` ×~20, the realizer hand-computed rows,
  `test_snmesh_realizer_wiring` ×2, and the 2-D schedule-split partition. A
  claim-class-preserving migration.

### 5 — refuted candidates (first-class output)

- `SNMethodSpace.minimal(quad)` + a deck law in the docstring code-blocks at
  `geometry/boundary/_base.py:196` and `reflective.py:58` — **not** a G6.5 hit:
  measured, it already raises `BoundaryError … without outflow_indices` from
  `_outflow_restriction`, dead since B3.2.
- `derivations/diagnostics/diag_phase_g_step2_cyl_apply_internal.py:72` — dead
  at import (`from orpheus.sn.boundary_realizer import …`, a module path that no
  longer exists).
- Every `SNMethodSpace.minimal` test consumer — cannot reach `_deck_kernel`
  (0 logged entries); stopped earlier by the same refusal.
- `tests/geometry/test_reemission_closure.py:847`'s
  `TraceRestrictionOperator.to_local` mention — genuinely past tense ("Until
  this carve … fell through to"). The carve's tense judgment was right.
- The re-posed control in `test_specular_deck_chain.py` (a hand-built unbound
  twin rather than the old production A/B) — weaker, but NOT a hole: the module
  still carries the production halves (`test_the_mirror_squared_is_not_an_expression`
  asserts the bound square RAISES through the realizer;
  `test_the_realized_operator_carries_the_binding` pins the binding).

Cross-refs [[lessons-L063]] (over-powered mutation), [[lessons-L064]]
(single-sourcing closes the input-resolution axis — L-065 §2 is its PRODUCTION
twin), [[lessons-L022]] (worktree baselines), `vv-principles` #18/#20/#21/#23,
Mode 12.

---

## L-066 — an issue's blast-radius number is usually a NAME grep over a different type family (measured 43x over); and a hardcoded `converged=True` is only a defect if the producer ITERATES

**Context.** Issue #340 reconnaissance over `orpheus/derivations/`, 2026-08-09,
`main` @ `4bcce0bd`. The SN/numerics half had landed in `d9b027d7`
(`power_iteration` returns an outcome carrying `converged`; `IterationHistory.converged`
required, no optimistic default; 5 transcriptions → 1 predicate). The
`derivations/` family (`CriticalSolution`) was deliberately deferred with a filed
cost estimate: *"the cost is the ~87 reads feeding the cross-method comparison
dict, which wants its own pass."*

### 1. The estimate was a NAME grep over a DIFFERENT type family — and it inflated 43x

`[M]` `grep -rn '\.converged' tests/derivations/` = **87**, exactly the filed
number. Decomposed:

| family | hits |
|---|---|
| `test_peierls_*` (`PeierlsGreensFunction*Result.converged` — a different type, already honest) | **72** |
| `test_fn_*` (function-level FN result types) | 8 |
| `test_trajectory_*` | 4 |
| `test_singular_*` | 3 |
| `test_galerkin_*` | 0 |

The number is real; the *attribution* is not. `.converged` is a name shared by ~8
independent result dataclasses across three pillars, and a name grep cannot tell
them apart.

**The measurement that settles it** — an in-process pytest plugin wrapping the
exact class (throwaway, `-p` + `PYTHONPATH`, no tracked file edited):

```python
_orig_init = CS.__init__
def _init(self, *a, **k):
    caller = traceback.extract_stack()[-2]
    (CONSTRUCTED_WITH if ("converged" in k or len(a) > 4) else CONSTRUCTED_WITHOUT).append(
        (caller.filename, caller.lineno, caller.name))
    return _orig_init(self, *a, **k)

_orig_getattr = CS.__getattribute__
def _getattr(self, name):
    if name == "converged":
        READS.append(tuple(traceback.extract_stack()[-2][:3]))
    return _orig_getattr(self, name)

CS.__init__, CS.__getattribute__ = _init, _getattr
```

Run over every consumer suite (`tests/cross_method` + the 4 `tests/derivations`
facade modules), `python -O`, 140 passed / 250 s:

```
constructions WITH converged=   : 33
constructions WITHOUT converged=: 0     <-- removing the default breaks NOTHING
.converged READS on CriticalSolution: 2  (both in tests, both benign)
```

⟹ **87 → 2.** The deferred "expensive" half was a **zero-churn** edit. The error
direction is the dangerous one: an inflated cost defers a cheap fix, and the
deferral note then reads as a considered decision (`plan-authoring` §2).

**Instrument's positive control:** both counters non-zero ⟹ the wrap was live. A
`0/0` report means a dead plugin, not a clean tree — check that first.

**Completeness precondition, and it is what makes `0` an answer rather than a
sample:** a dynamic audit only sees the paths the suite runs. Pair it with a
static proof that no *other* construction path exists — `[M]` here: 9 literal
`CriticalSolution(` sites, no `**kwargs` splat, no `asdict`/`astuple` round-trip,
and the only `dataclasses.replace` calls in the family target
`Peierls*Solution.phi_values` and `CrossMethodCase`. Without that, the dynamic `0`
is merely "not observed".

**Nexus's role, stated precisely (it is NOT a failure):** `context` on
`py:class:...CriticalSolution` resolved every PRODUCER perfectly (9
`type_uses`/`calls` edges + the 4 rendering doc pages) and reported the attribute
node `...CriticalSolution.converged` with **`degree: 1`** — its only edge is
`contains` from the class. That is the graph correctly saying *a plain dataclass
attribute read is not an edge*. Route: **Nexus for producers, dynamic wrap for
readers, grep only to enumerate candidates.** Three tools, three questions.

### 2. A hardcoded `converged=True` is a defect ONLY if the producer iterates — 5 of 9 were legitimate

The tree carried **7** hardcoded `converged=True` at `CriticalSolution`
construction. Triaged one hop UP, by whether the producer has a loop + tolerance:

| site | producer's method | verdict |
|---|---|---|
| `singular_eigenfunction/spectrum.py:819`, `:872` | hand-rolled `while iters < max_bisect` + `break` on `d_hi-d_lo < bisect_tol*max(1,d_lo)` | ⛔ defect |
| `fn_method/moment_space.py:315`, `:347` | same / `minimize_scalar` + fallback | ⛔ defect |
| `fn_method/moment_space.py:424` | `compute_kinf_mg` = `float(nu_sigma_f @ np.linalg.solve(A, chi))` | ✅ nothing iterates |
| `galerkin_spectral/basis_space.py:756`, `:793` | `solve_eq4_eigenproblem` = direct `scipy.linalg.eig` | ✅ nothing iterates |

A grep-driven "fix every hardcode" pass would have minted **false honesty** at the
three direct-method sites — teaching readers that `True` there was *measured*. The
discriminator costs one hop and cannot be skipped. (Design residue for the carve,
not a defect: a boolean cannot distinguish *converged* from *not applicable*; the
five legitimate `True`s and the four lies are grep-identical, which is precisely
how the lies hid.)

### 3. `full_output=True` does NOT make a scipy status readable — `disp=False` is the load-bearing half

The subtree's one honest producer
(`singular_eigenfunction/cylinder/one_group.py:795-818`) passes
`full_output=True, **disp=False**` and records both `iterations` and
`converged`. `[M]` with scipy's default `disp=True`, a non-converged `brentq`
**raises `RuntimeError` instead of returning `converged=False`** — so
`res.converged` would be structurally unreachable-False even with
`full_output=True`. Measured on `f(x)=eˣ−3x−1` over `[1,3]`:
`maxiter ∈ {1,2,3}` → `converged=False`, `flag='convergence error'`;
`maxiter=100` → `True`. Reviewing a "we read scipy's flag" claim: check `disp`,
or the `False` leg is an unreachable branch wearing an honest name.

Corollary — the tree's *other* `brentq` sites (`.../core/dispersion.py:207`,
`cases/sn.py:650`, `cases/diffusion.py:597`) omit `full_output`, inherit
`disp=True`, and are **honest-by-raising**. Both policies are honest; only the
first is *readable*, and the readable one is what a "warn, CI-escalatable, not
raise" ruling requires.

### 4. Three discard shapes, ranked by how well they hide

Same defect class, worsening observability:

1. **Fact recorded partially** — `n_bisect_iters=iters` survives, the exit REASON
   does not. A consumer can *infer* from `iters == max_bisect`
   (`singular_eigenfunction/{slab,sphere}/one_group.py`).
2. **Fact computed, branched on, dropped at the `return`** —
   `converged = abs(k_new-k_val) < tol and iteration > 5` … `if converged: break`,
   and the result dataclass has no slot (`peierls_nystrom/slab.py:596`,
   `geometry.py:6450`). Literally the `power_iteration` defect, second family.
   (Also a **6th spelling** of the predicate — note the hardcoded `and iteration > 5`
   floor folded into the test; the first pass collapsed 5 spellings but only inside
   `orpheus/numerics` + `orpheus/sn`.)
3. **Dead local** — `converged = False`/`= True` assigned inside the loop and
   **never read anywhere after**, in the same function
   (`fn_method/slab/flux_reconstruction.py:854, 866`). Nothing downstream can ever
   recover it.

⭐ And the review-time tell for (1)/(2): **`iters < max_bisect` is the WRONG
predicate even when you go to fix it.** In all four hand-rolled loops the counter
increments at the BOTTOM and the tolerance test sits at the TOP, so the final
step's narrowing is never tested — a run exiting on `iters == max_bisect` may in
fact be inside tolerance. The correct spelling re-evaluates the bracket predicate
after the loop (`d_lo`/`d_hi` are still in scope), which also needs **no invented
tolerance**: `bisect_tol` is already a documented parameter at every site.

### 5. Measured consequences (so the carve has its RED gates ready)

All four defects starve silently through the public facade:

| path | knob | value moves | reported |
|---|---|---|---|
| `Spectrum.solve_critical` (SE slab) | `max_bisect=3` (vs 29 needed) | `d` 5.665505 → 5.679727 (**1.4e-2**) | `converged=True` |
| `MomentSpace.solve_critical` (FN slab) | `max_bisect=2` | `a` 0.9377198 → 0.94375 (**6.0e-3**) | `converged=True` |
| `MomentSpace.solve_critical` (FN sphere) | `max_bisect=1` | `R` 2.4248249 → 2.4317897 (**7.0e-3**) | `converged=True` |

⚠ **The positive control for a convergence-flag gate must itself be
budget-dependent** — a catastrophic kernel mutation leaves such a gate green by
construction (`vv-principles` #23's invariance-gate control, same shape).

**Secondary correctness bug found en route**, independent of the contract question:
`fn_method/sphere/one_group.py:358-362` reads `minimize_scalar(...).success`,
branches on it, and on `False` **falls back to the coarse bracket-scan guess** —
`[M]` at `max_bisect=5` scipy's own unconverged iterate was accurate to **8.4e-5**
while the fallback it chose is off by **7.0e-3**, an **83x** degradation, then
stamped `converged=True`. A fallback that is worse than the thing it replaces.

**Lesson.** An inherited blast-radius number is a `[M]`-less claim about a *name*,
not a *type*: re-measure it against the exact class with an in-process wrap before
letting it size (or defer) the work — here it was 43x too big and the direction
favoured deferral. Then triage every hardcoded status one hop UP: the producer's
method decides whether the constant is a lie or a fact, and grep cannot see the
difference.

**Cross-refs:** [[lessons-L039]], [[lessons-L043]], [[lessons-L052]] (in-process
mutation mechanics), `vv-principles` #23 (the control must match the knob),
`plan-authoring` §2 (a `[M]`-less number reads as measured), the
`feedback-lossy-return-type-is-the-root-cause` ruling (triage one hop UP).

---

## L-067 — the published escalation FLAG did not parse (so the "CI gate" was imaginary), the delta it produced was 100 % deliberate, and my own census plugin's decoder invented 44 of its 90 findings

**Context.** Issue #340 bullet 2 — "audit every SN gate for a missing
`history.converged` assertion". The instrument handed to me was the
`ConvergenceWarning` that landed in `d9b027d7`, escalated per the project's own
published recipe: `python -O -m pytest -W error::ConvergenceWarning`. HEAD
`4bcce0bd`, `tests/sn -m "not slow"`, SERIAL.

**F0 — the recipe cannot run.** `[M]` Python's `-W` parser resolves an
**undotted** category against `builtins`, so `warnings._setoption(
'error::ConvergenceWarning')` raises `_OptionError: unknown warning category`
and pytest exits `ERROR ... AttributeError: module 'builtins' has no attribute
'ConvergenceWarning'` with **zero tests collected**. Four sites publish that
spelling — `orpheus/numerics/convergence.py:70` and `:107`, the **emitted
warning message itself** at `orpheus/sn/solver.py:454`, and
`tests/sn/solve/test_convergence_contract.py:26`. The working form is
`-W error::orpheus.numerics.convergence.ConvergenceWarning`.

The sharp part is WHY no test caught it: `test_it_is_escalatable_to_an_error`
installs the filter **programmatically** (`simplefilter("error", ConvergenceWarning)`
+ `pytest.raises`) and passes. Category-escalatability and string-parseability
are two claims; the suite gated the first and published the second. → new
`vv-principles` Mode-8 **EIGHTH class** (gate the MECHANISM vs the published
INVOCATION), with the one-line gate:
`_pytest.config.parse_warning_filter(s, escape=False)` — `[M]` `UsageError` on
the bare name, `('error','',<class …ConvergenceWarning>,'',0)` on the dotted.

**The instrument's own positive control had to be built.** The obvious control —
the contract file — is useless: `[M]` all 9 of its tests PASS under the flag
because every starved call is wrapped in its own `catch_warnings`/`pytest.warns`.
The real control was an **unprotected** replica of its own `_fixed_source(
max_inner=50)` fixture in a throwaway module outside the repo tree: FAILS under
the flag, and the converged `max_inner=4000` leg stays green (anti-dud).

**The delta: 7, and every one DELIBERATE.** `16 failed, 2885 passed` vs the
9-red baseline (`2885+16 = 2901 = 2892+9`; all 9 baseline reds are
non-`ConvergenceWarning`, so the split needed no re-run).

| entry | budget | tol | distance | class |
|---|---|---|---|---|
| `test_dsa_acceleration::TestTeeth::test_sign_flipped_correction_breaks_convergence` | `max_inner=200` | 1e-11 | 8.378e+56 | divergence witness |
| `…::test_zeroed_trace_arm_breaks_the_reflective_case` | 120 | 1e-11 | 3.614e+20 | divergence witness |
| `test_krylov_curvilinear_precond_safety::test_g_d3_3…[eigenvalue]` | `max_outer=2` | 1e-3 | \|dk\| 4.663e-15 | structurally unconvergeable |
| `test_si_single_primitive_contract::…[slab]` / `[sphere]` | `max_inner=4` | 1e-12 | 5.275e-01 / 4.847e-01 | constructor spy |
| `test_sn_adjoint_certification::TestP13Mutations::test_streaming_no_reversal_shifts_k_heterogeneous` | `max_outer=500` | 1e-9 | \|dk\| 3.331e-16 | mutation-induced |
| `test_si_cyl_20cell_nan_regression::test_si_returns_finite_keff` | `max_outer=3` | 1e-10 | \|dk\| 8.628e-02 | finiteness-only |

Two of them are worth carrying:

* **`max_outer=2` can NEVER report converged** — `SNSolver.converged`
  (`solver.py:1559`) opens `if iteration <= 2: return False`. So that row's
  warning reports a *guard*, not a numerical shortfall: `|dk| = 4.66e-15` is 12
  orders INSIDE `keff_tol=1e-3`.
* **A NEGATIVE dominant eigenvalue makes a flux-increment criterion
  un-satisfiable.** `[M]` the mutated adjoint returns `k = -0.6519302852190432`
  **bit-identical** at `max_outer` 500 and 2000, `|dk|` pinned at the FP floor
  3.331e-16 (< `keff_tol=1e-9`) while `dphi` never falls below `flux_tol=1e-8`
  — the power iteration sign-alternates. `converged` needs BOTH (`:1564`).
  Raising the budget is futile; the fix is `pytest.warns`, never a bigger number.
  This also **REFUTED the issue's own attribution** ("the *sphere* run exhausts
  500 outers at |dk| = 3.3e-16"): the sphere row passes under escalation and
  `[M]` the unmutated adjoint converges in **5** outers. 3.3e-16 is just the FP
  floor of `|dk|` at `k ≈ O(1)` — a non-distinctive number, easy to misattribute.

**The instrument had three structural holes, so I built a second one.** The `-W`
sweep sees only a warning that ESCAPES the test body, so it is blind to (a)
suppressed warnings, (b) `xfail`-absorbed ones (`[M]` 61 xfails in the slice),
and — the big one — (c) **INNER truncation at the two eigenvalue entries**:
`solve_sn`/`solve_sn_adjoint` call `_warn_if_unconverged` with
`budget_name="max_outer"`, and their `converged` is `power_iteration`'s OUTER
fact (`solver.py:2344-2360`). A within-group solve that hits `max_inner` inside
a power iteration is invisible **by construction** — #340's defect class, one
level in. The second instrument was a pytest plugin wrapping
`_warn_if_unconverged` (entry census) and `_certify_within_group_exit` (inner
census); it reproduced the 9-red baseline exactly, so it is behaviour-neutral.

Entry census: **12 rows = the 7 delta + 5 correctly-suppressed contract-file
legs.** ⟹ nothing is hidden behind suppression or xfail anywhere in the slice —
the `-W` delta is complete at entry level. That negative is the whole reason the
second instrument was worth building.

**⚠ And then MY decoder invented 44 findings.** The inner census printed 90 rows.
The wrapper flagged "truncated" whenever `_claims_convergence(history, tol)` was
false — the same predicate production uses — but that predicate is ALSO false for
an **empty** history, which `KrylovAcceleration.solve`'s own docstring
(`iteration.py:937`) defines as *"GMRES returned in zero iterations"*, i.e.
converged on the initial guess. `[M]` all 44 print my `inf` sentinel and no
`KrylovAcceleration … info=` warning fired anywhere (scipy `info == 0`). **46
genuine.** The positive control had passed cleanly — it only ever exercised the
genuine branch. → new `vv-principles` METHOD-WARNING clause: for a CENSUS
instrument, enumerate every state the predicate maps to True and control each
one; a production predicate reused as a detector inherits its OTHER meanings.

**Second calibration, before ranking the 46:** an inner truncation on an EARLY
outer under a converged power iteration is an inexact-Newton posture, not a
defect. Census B is a SCREENING list. The two sharpest candidates measured
benign:

* `test_d3_admission::test_kinf_3d_equals_2d…[2g]` (worst shortfall in the whole
  census, 4.962e-03 vs `inner_tol=1e-11`, n=11): `[M]` `max_inner` 200→800→3000
  gives `|k − k_inf|` 2.917e-12 → 2.691e-12 → **1.998e-15** against the gate's
  `atol=1e-8` — 3400× inside. Structurally benign: the fixture is HOMOGENEOUS
  all-reflective, where `k_inf` is a material-property ratio independent of flux
  shape. The truncation shows up as *outer work* instead (8 → 3 outers).
* `test_dd_regression[sphere_2g_3reg_dd_n40]` (frozen snapshot): `[M]` shipped
  `max_inner=300` reproduces the snapshot **bit-identically**; at 1200 the drift
  is `dk = 1.446e-13` vs the pin's `10 × keff_tol = 1e-11` (69×) and
  `dφ = 1.640e-12` vs `10 × flux_tol = 1e-9` (610×). The baselines are NOT
  brittle to the truncation.

**Two more findings en route.** (1) `KrylovAcceleration.solve` surfaces scipy's
`info != 0` as a bare **`RuntimeWarning`**, not `ConvergenceWarning` — so the
escalation flag does not cover the Krylov half of the same defect class, and
ERR-053 (the precedent `convergence.py`'s docstring cites for its own design)
lives on that path. (2) `test_si_cyl_20cell_nan_regression.py:66-67` runs
`warnings.filterwarnings('ignore')` — unqualified, at module import; only
pytest's per-item filter reset keeps it from poisoning the run.

**Lesson.** Three, in order of transferability. (i) **A published command is a
separate claim from the API it wraps — gate the STRING.** (ii) **An audit
instrument needs one control PER STATE its predicate accepts**, not one control
per instrument; the positive control will happily pass while the decoder
mis-labels a different state, and it will do so in the flattering direction.
(iii) **An entry-level "converged" flag at an eigenvalue entry is the OUTER fact
only** — before crediting any convergence sweep as complete, ask which loop the
flag belongs to.

**Cross-refs:** `vv-principles` Mode-8 EIGHTH + NINTH classes and the census
METHOD-WARNING clause (all three added by this review), Mode-8 FOURTH class
(the marker-form xfail this NINTH one complements), `numerical-bug-signatures`
Signature 8 (the discarded-info-flag ancestor), [[lessons-L053]] (a `slow`
catcher is deselected — the same blindness applies here to the 114 deselected
rows), [[lessons-L061]] (`-rs` and read the reasons), issue #340.

---

## L-068

**Adversarially reviewing a design chain (SN cylindrical angular closure, Q5.6.4
attempt 2), 2026-08-11.** Branch `refactor/operator-strategy-layers`. Brief: refute
a 7-link chain C1→C7 if it can be refuted; the previous attempt "shipped a worse
answer by propagating an unchecked premise into a conclusion, so the premise audit
is the point". Deliverable `scratch/q64_attempt2_qa_review.md`.

### What the chain claimed, and what happened to each link

C1 conservative cylindrical form + `ξ` face coefficient → SURVIVED. C2 `α =
κ·w_gl·ξ(e_arc)` → SURVIVED. C3 "candidate (3) ill-posed at every order" → FALSE
AS STATED. C4 the published criteria are τ-blind → SURVIVED (strongest finding).
C5 `τ ≥ ½ ⟺ non-amplifying` → mis-scoped BOTH directions. C6 the "honest τ
instrument" → **BROKE**. C7 (the proposal) → does not follow.

### 1. C1: verified by a route touching nothing in the repo

Built `Ω` in LAB Cartesian, transported along a ray, chain rule: `dr/ds = η`,
`dω/ds = −ξ/r` (both residual `0`); then `(η/r)∂(rψ)/∂r − (1/r)∂(ξψ)/∂ω` minus the
non-conservative form `= 0`, while two plausible-wrong variants are non-zero. The
whole thing turns on `∂ξ/∂ω = η`. Face coefficient `W·ξ(ω_face)` by FTC, no κ.
κ itself derived symbolically as `Δω/(2 sin(Δω/2))` = midpoint-vs-exact ratio of
`∫_cell η dω`. Later corroborated by the literature agent: Hébert **Eq. (3.157)**
verbatim (his letters swapped), face term **Eq. (3.393)**, plus Bell & Glasstone
p. 58 + Table 1.2 and BMC Eq. (48).

**But two DEFECTS in how C1 was STATED**, and both mattered:
* *"so the half-angle faces **sit at** the geometric arc edges and the coefficient
  there is `w_gl·ξ`"* splices a DEFINITION (where the faces go — the very thing
  under debate) to a DERIVATION with a "so". `plan-authoring` §2's `[M]`-scope
  defect, and it was the clause §9bis.2 leaned on to reinstate the partition.
* C2's corroboration is **procedural, not structural** → digest C1.

### 2. C6 broke, and the mechanism is the reusable lesson

The instrument fed `η` and `ξ` through the closure, justified as *"a
P1/diffusion-limit flux is affine in the direction cosines"*.

I first attacked the **weighting** (the brief's own suspicion) and it held: five
weightings (unweighted max, `ξ(e)`-wtd, `|η(e)|`-wtd, uniform L2, sum-of-legs) all
rank `τ≡½` first. That felt like a validation. It validated nothing.

The defect is the **BASIS**:
* the `ξ→−ξ` reflection across the `(e_r,e_z)` plane leaves a 1-D cylinder
  invariant ⟹ `ψ` even in `ξ` ⟹ `J_φ ≡ 0` ⟹ the P1 limit at a level is `A + Bη`,
  `ξ` coefficient identically zero. BMC's own **Eq. (1)** is `φ/4π + 3J_r μ/4π` —
  ONE cosine. BMC **Eqs. (61)–(62)** write `Ω = μ e_r + ξ e_z`, no azimuthal
  component at all.
* on a σ_y-**folded** rule every node has `ξ > 0`, so `quad.mu_y` samples `|ξ|`:
  `[M]` `Σwξ = +6.703` folded vs `0.000` unfolded, `min ξ = +0.3125`. So "ψ affine
  in ξ" is not even a function on the rule the closure runs on.
* Fourier-cosine content on the arc: `cos ω` has `c_1 = 1` and nothing else (ONE
  harmonic = the P1 mode); `sin ω` spreads over `m = 0,2,4,6` with an `m⁻²` tail;
  `ω` (what `τ≡½` is exact on) over `m = 0,1,3,5`. In the `η` chart both are
  sqrt-/arccos-singular at the level endpoints.

Re-run on the realisable basis `{cos mω}` = Chebyshev in `η/sinθ`,
`folded_product(4,64)` level 0, `max|ψ̂(e) − f(e)|`:

| mode | chord | chord+absorber | **arc LANDED** | `τ≡½` |
|---|---|---|---|---|
| `cos 1` (P1) | 5.4e-15 | 3.605e-03 | **2.1e-15** | 2.412e-03 |
| `cos 2` | 4.887e-03 | 1.435e-02 | **4.854e-03** | 9.677e-03 |
| `cos 3` | 1.171e-02 | 3.205e-02 | **1.132e-02** | 2.188e-02 |
| `cos 4` | 2.185e-02 | 5.636e-02 | **1.994e-02** | 3.918e-02 |
| memo's `ξ` leg | 6.637e-01 | 1.631e-02 | 1.415e-01 | **6.131e-04** |

**Ranking INVERTED**, and not merely on the mode the arc convention is exact on —
it wins ≈2× on EVERY harmonic, which kills the symmetric circularity objection.

### 3. C3 was a PARITY artefact — the refinement-ladder trap

"exactly 1 of `M+1` edges has no real solution, every order" was measured at
`n_φ = 8/16/32/64` ⟹ `M = 4/8/16/32`, ALL EVEN. The failing edge is the one at
`ω = π/2`, an edge only when `M` is even. `[M]` at `n_φ = 6/10/14/18/26/34/66`
(`max κ·sin ω_arc = 0.9069/0.9669/0.9832/0.9898/0.9951/0.9972/0.9992 < 1`) →
**0 of `M+1`**. An 8× ladder read as "every order" and was one congruence class.

### 4. C5 mis-scoped in both directions

`[M]` end-to-end `Π|(1−τ)/τ|`: chord `1.000000`, arc `1.000000`, `τ≡½` `1.000000`,
chord+absorber `2.4549e-02` at `n_φ=64`. Both derived partitions satisfy
`τ(π−ω) = 1−τ(ω)`, so the product telescopes to exactly 1 — the memo's "worst
running product" is a TRANSIENT interior bulge (unit seed error peaks at `6.68` at
face 8/16 for arc, `20.36` chord, arriving at the level end at `1.000000`). So the
absorber and `τ≡½` do NOT tie: the absorber is the only DISSIPATIVE one, 40×.
Positivity: `(1−τ)/τ` grows as τ falls, so `τ≡½` is the SAFEST derived candidate —
`[M]` on a steep-shadow profile `min ψ̂ = −24.2` (`τ≡½`) vs `−77.2` (arc) vs `−230`
(chord). The stated caveat had it backwards. And no gate covers `ψ̂` positivity on
either arm; the two curvilinear positivity gates are both on the SPHERE's
converged SCALAR flux.

### 5. The decisive row — filled in, and it re-framed everything

Live solve of `cyl_2g_3reg_folded_4x8_dd_n40` per convention vs the
trajectory-resolvent reference, gate `1.2e-1`:

| convention | `k_eff` | overall | gate |
|---|---|---|---|
| chord | 1.2308955887 | 1.4409e-01 | FAIL |
| chord+absorber [OLD PROD] | 1.2302082296 | **6.5934e-02** | PASS |
| arc [LANDED] | 1.2310212586 | **1.2676e-01** | FAIL |
| `τ≡½` [C7] | 1.2313562779 | **1.0181e-01** | PASS |

The two bold anchors reproduce the memo's §4.6 numbers to the printed digits, and
the two `k_eff` are exactly the re-baseline pair recorded at
`test_phase_c_crosscheck.py:214` — that is what licensed reading the new `τ≡½` row.

⭐⭐ **Rank correlation**: the order `absorber < ½ < arc < chord` is EXACTLY the
transient-bulge order `{1.00, 1.00} < 9.44 < 40.7` (with the absorber's 40×
dissipation breaking the tie), and the REVERSE of the closure-accuracy order
(`arc 1.99e-02 … absorber 5.64e-02`). **The metric the whole campaign steered by
measures the RECURRENCE; the campaign was arguing about the CLOSURE.**

### 6. The literature (independent agent + my own sidecar spot-checks)

§9bis.9 landed at `8db88596` MID-REVIEW (memo 721→879 lines) with the strongest
defence of C7. Its OCR facts all confirmed by my own reading — §3.9.3 = cylinder,
§3.9.4 = sphere; Hébert's Eq. 3.406 (cyl) and 3.431 (sph) are BOTH "the diamond
differencing scheme"; 3.414/3.439 are both `2φ − φ_{−1/2}`; Hébert states no τ. So
the production docstring's *"Morel–Montry **weighted**-DD recurrence of Hébert
3.437/3.439"* is false three ways — a real, outcome-independent defect.

Its CONCLUSION refuted:
* BMC name `τ = ½` *"the diamond scheme"* under **Eq. (53), in their CYLINDER
  section**, and the paper's thesis is that the diamond preserves the diffusion
  limit only to LEADING order while Morel–Montry's weighted τ reaches FIRST order
  ("not as accurate").
* BMC's cylinder τ (**Eq. 74/75**) is barycentric in the **RADIAL COSINE**; ω
  never appears.
* ⭐ the predicate the sources STATE is chart-FREE — BMC under Eq. (43): *"will
  exactly relate the cell-edge and cell-center fluxes when the angular flux
  assumes the linear form defined by Eq. (1)"*. **L48 applied correctly ("take the
  PREDICATE, not the recipe") selects P2-in-η, i.e. what the tree ships.** The
  memo invoked L48 and substituted a different predicate ("barycentric in the
  variable the cells are equal in") that no source states.
* the Hébert appeal is **arm-asymmetric**: he prescribes the same diamond for the
  sphere, which the tree rejects (`[M]` sphere τ ∈ `[0.3897, 0.6103]`, never ½).
  The reason offered for the asymmetry (ω-midpoint NODES satisfy the diffusion
  moment condition exactly) is a NODE property that probe D already measured as
  τ-blind — level conflation.
* ⭐ the literature agent's best find, which dissolves §3's whole framing: **two
  different cosines live at every azimuthal face** — the azimuthal one (`α/W_p`,
  value fixed by the conservation recursion) is the STREAMING coefficient; the
  radial one (BMC Eq. 52) is what τ is barycentric in. C1(b) and BMC Eq. (74) name
  different numbers at the same face.
* ⛔ a published typo that would INVERT C1: BMC printed p. 156 writes
  `η = sinθ cos ω` (must be `sin ω`), in the paper the sphere arm cites.
* ⛔ Alcouffe & O'Dell (Hébert ref. [36]) — the primary source for ORPHEUS's
  cylinder cell-edge construction — **has never been read** (unresolved, 7 queries
  / 4 databases); Morel & Montry (1984) TTSP 13(5) 615–633 not local.
* ⚠ the closed-form τ FLIPS SIGN with march orientation (`½ ∓ ½cot ω tan(Δω/4)`);
  `[M]` the two agree as a SET to `8.9e-16`, differing only in order, so a
  level-symmetric fixture cannot see a flip. Ungated either way.

### 7. Consumer audit + mutation verification

`explorer` ran the three searches; I ran two in-process plugins (no tracked file
edited; revert proved by gate-green-again `136 passed`).
* Naive ω-swap → **21 of 136 red**, all cylinder, mostly the P3 guard
  (`τ_raw[0] = 4.598 ∉ [0,1]`) — loud. Zero sphere rows.
* **FAITHFUL C7** (partition AND P2 node in ω ⟹ `τ≡½`, plus C7's promised
  convert-at-the-consumer fix) → **only 2 of 136 red**, both the same value pin
  `test_cyl_tau_equals_the_ANALYTIC_closed_form_not_the_chord_convention[8,16]`,
  whose reference is the formula being retired. Blind: the `0.25 ≤ τ ≤ 0.75`
  wellposedness gate (½ is inside), per-ordinate flat-flux (τ-blind), the
  contamination-β gate, `test_alpha_closed_form` (α is τ-independent AND its edges
  are hand-rolled), the whole MMS ordering ladder, march-start structure.
  ⟹ **a first-order change to a production angular closure has exactly ONE
  catcher.**
* Three unguarded `angular_differencing` consumers each need hand conversion and
  **2 of 3 have no test consumer at all**: `alpha_defect_beta` → `π²−1 = 8.8696`
  garbage; `nu_closure_residual` → **`inf`** (`edge_omega[M] == 0.0` exactly, and
  the body's last line divides by `e[-1]`) — and that is the memo's own headline
  discriminator.
* C7 RE-CREATES a partition twin: `reduced_operator.py:871`'s
  `mu_start_per_level = −sinθ` IS `edges[p][0]`, consumed live by
  `_edge_seed_stencil:1414` — it desyncs silently.
* **The single-source partition producer has NO value gate**: no test anywhere
  asserts its returned values; the `[-1,1]` / `Σ Δμ = 2` / monotone / endpoint
  invariants live only in prose.
* `folded_product` is the ONLY cylinder-admissible factory and every instance is
  ω-equispaced ⟹ under C7 the cylinder τ is a CONSTANT on 100 % of shipped
  configurations, and the "derived, not hardcoded" defence rests on a hand-built
  arc no factory can emit (a signature-tautological gate waiting to be written).
  Probe E1's sphere `0.000e+00` is likewise a tautology — same variable, same
  unedited code.

### 8. What I concluded

The tree already ships the better closure. The empirical penalty is
seed-and-transient dominated, not closure-truncation dominated, and the one
unexamined option — hold the arc closure and fix the starting-direction SEED — is
the only candidate that could recover the accuracy without giving up BMC's
exactness property. Named the falsifiable experiment (vary only
`MorelMontryAngularSweep.psi_half_seed`, re-run the live probe) and the outcome
under each branch. C7 remains defensible ONLY as an explicit
stability-over-accuracy trade, argued at `1.0181e-01`, never as "the chart was
wrong" or "the literature says ½".

**Skill edits made:** new anti-pattern **#24** (validating an ADJUDICATING
instrument: basis / rank-correlation / cost-against-alternatives), and a
refinement-ladder-congruence-class sharpening on **#13**.

**Related:** `vv-principles` #24 (new), #13 (sharpened), #7 / ERR-032 (shared
upstream identity), Mode 7 (the ansatz-nulls dual of the basis check), Mode 12
(the algebraic form of the rank-correlation check), `plan-authoring` §2 (the
`[M]`-scope and quantifier/denominator rules — C1's "so" and C3's ladder are both
instances), `lessons-L048` (take the PREDICATE not the recipe — the memo invoked it
and substituted a different predicate), [[lessons-L029]] (circularity), Q5.6.4 /
`.claude/plans/archive/q64_tau_partition_memo.md`.

## L-069

**Task.** Judge whether 7 failing CYLINDER snapshot gates (3 modules,
`tests/sn/_data/affine_carve_baseline/`, `tests/sn/_data/bc_extraction_baseline/`,
`tests/sn/_fixtures/wave_t_t4/pre_t4_snapshots.npz`) may legitimately be
re-baselined. Deliverable `scratch/task51_cyl_snapshot_audit.md`.
Verdict: **RE-BASELINE all 7**, with 2 blocking doc repairs.

**The finding that reframed the whole audit.** The brief posed it as an open
judgment call. It was not: `39b46a31` — *"re-baseline the TWO cylinder
artifacts the ω-partition moved — and record why the other two did not"* —
is **already in the tree** (`git merge-base --is-ancestor` ⟹ YES), 6 commits
after the value-moving carve. It did the diligence properly *inside its
scope*: sha256 before/after, a per-artefact `τ := 0.7` sensitivity screen,
an in-place correction of a falsified prediction. Its scope was
`tests/sn/regression/snapshots/` — **one directory**. Its universal
*"Verified by sha256 over all 23 snapshots … these are the only two that
changed"* names its denominator honestly and is tree-wide FALSE: 7 more
frozen references moved in 3 other directories. `tests/sn` instead of
`tests/sn/regression` would have shown them in 0.2 s.
⟹ **Before auditing a re-baseline decision, `git log` the snapshot's own
directory for a commit that already made it.** The reds may be a
re-baseline's REMAINDER, and then the question is completeness, not
legitimacy.

**The bundled-mechanism false blindness (→ `vv-principles` #25).** The same
commit's case list, in `tests/sn/regression/_generate_snapshots.py`, carries
an `[M]` marker and says *"folded_2x4 has M = 2 … the ω-midpoint partition is
BIT-IDENTICAL to the retired η-midpoint one at M = 2 … **So this case's tau
did not change at all**, and no M = 2 fixture can ever see a partition
change."* `[M]` The partition half is true (interior edge `5.0e-17 ≈ 0`); τ
changed by **2.071e-01** on every level (`0.292893 → 0.5`) because
`3dda18ca` retired **two** things and at `M=2` the *absorber*
(`max(0.5, min(1.0, τ))`) was the binding one. Conclusion right, argument
void, certificate durable — and refuted empirically: the `n_φ=4` (`M=2`) row
of `test_cyl_tau_equals_the_ANALYTIC_closed_form_not_the_chord_convention`
is among the 32 gates that redden on old-τ.

**The instrument that made everything decidable — a whole-suite mutation
DIFFERENTIAL.** In-process plugin rebinding
`pole_angular_closure.morel_montry_tau_per_level` to the verbatim
pre-`3dda18ca` body (sole sweep/matvec consumer resolves it as a module
global, so one rebind covers the path). Bite check: the 7 go GREEN, plugin
reports `invoked 10 times`. Then `tests/sn -m "not slow"` in BOTH arms:
`MUT 41 failed / 3014 passed` vs `BASE 16 failed / 3039 passed`. The
symmetric difference was exact — 7 red only at HEAD, **32 red only under
old-τ**, 9 red in both (another agent's quadrature scope).
⟹ **Two arms of the same suite convert "does an external pin exist?" from an
argument into a list.** The 32 answered it: 8 rows of an analytic-closed-form
τ gate (with the retired chord as negative control, `n_φ=8` — the failing
fixtures' own config — covered), 8 rows pinning the *recurrence
amplification* closed-form one tier downstream, and 2 already-re-baselined
solve-tier snapshots (2G 3-region het). **My own earlier draft concluded "no
external pin exists" and was refuted by my own measurement.**

**The blindnesses, all confirmed by absence from the 32.** (a) The M-M
recurrence `ψ_{m+1/2} = (ψ_m − (1−τ)ψ_{m−1/2})/τ` on a flat field gives
`(ψ − (1−τ)ψ)/τ = ψ` **for every τ** — one line of algebra disqualifies every
flat-flux L0 anchor at every order, including the
`@verifies("streaming-equilibrium")` gate living in the same FILE as three of
the failing rows, and `test_streaming_equilibrium_curvilinear.py` (blind twice:
flat, and `n_phi=4`). (b) The anisotropic-cylinder MMS is kernel-blind +
out-of-regime (`vv-principles` #24(d)/(e), already on record for this fixture)
— no MMS row is among the 32. (c) The tree's sharpest pin, the `sha256`
golden `test_affine_carve_bit_identity.py`, has **0 of 3 cases cylindrical**.

**Bisect mechanics that held.** `git worktree add -f --detach` at
`3dda18ca~1` and `3dda18ca`, each verified by printing
`pole_angular_closure.__file__` AND checking for the post-carve symbol
`angular_cell_edges_per_level` (`False`/`True`) — a revision fingerprint
stronger than the path. 31 green vs exactly-7-red, with ULP fingerprints
identical to HEAD's ⟹ nothing since moved the value. Refuted the brief's two
named causes: `c33178ef` post-dates the move; the fold was re-captured onto
at `c39b7d44` and was green.

**Harness self-failure, mine, caught mid-run.** `grep -E "^FAILED"` on
COLOURED pytest output matches nothing (ANSI escapes precede the `F`) — my
first whole-suite extraction reported no failures beside a `41 failed`
summary line. The warnings-summary lines leaked through only because
`-W error::orpheus…` contains the substring `error`. Also: piping a
background command through `grep` writes only the FILTERED output to the
task file, so the evidence cannot be re-extracted — 17 min lost. Fix:
`--color=no`, redirect FULL output to a file, filter afterwards.

**Cross-refs.** `vv-principles` #25 (added by this review), #17 (harness lies
in the safe-looking direction), #24(d)/(e), Mode 12 / §H2,
`bug-signatures` Sig-10 (whose sibling-pass discriminator is VOID when the
changed code is single-geometry — SLB/SPH green carries no information),
`coding-standards` retirement 3-searches, `plan-authoring` §2,
[[lessons-L034]] (the deferred-SPH stale snapshot — same family), L-068.

---

## L-070 — the knowledge graph's V&V surface is a SEARCH relation wearing a PROOF relation's name; and the per-test evidence that would fix it is produced today and thrown away

**Dispatch, 2026-08-15.** Design the adversarial/audit half of a graph-grounded
test workflow (ORPHEUS #358, #334) and state the demand on Nexus. Memo:
`scratchpad/nexus_demand_qa.md` (fenced write). Branch `main` @ `a1c90aac`,
`sphinxcontrib-nexus 0.16.1`, graph `docs/_build/html/_nexus/graph.db`
(24530 nodes / 217667 edges).

### 1. What the V&V relation actually IS

`[M]` **All 2748 `tests` edges are `test → equation`.** Split
`method→equation` 1430 + `function→equation` 1318. **There is no `test → code`
edge in the graph at all.** `[M]` 2747 of 2748 carry
`source="pytest.mark.verifies"`, `confidence=1.0` uniformly — so the whole
relation is *declared*, and the confidence field carries zero information (a
blanket file-level marker scores identically to a single-purpose L0 gate).

`[M]` **#334 confirmed and it is 50.5 % of the relation.**
`quadrature-ordinate-permutation` → exactly **21** edges, all from
`tests/sn/primitives/test_quadrature.py`, whose `pytestmark` (`:26-38`) names
**9** equations. **34 files** have `edges == n_tests × n_equations` with
`n_eqs > 1` (the file-level-`pytestmark` signature) and emit **1388 of 2748**.
`tests/cp/test_verification.py` alone emits **575 = 20.9 %** of the entire V&V
relation from 23 tests × 25 equations.

### 2. The three false-ALIVE mechanisms, in ascending severity

**(a) `provenance` mislabels "same page" as `implemented_by`.** `[M]` the CLI
printed **10** `implemented_by` for the permutation equation; the graph holds
**1** `implements` edge into it, and **0** into its page. `query.py:1326-1334`
walks equation → containing page → every `documents`-edge target. 10× over-report
in the silent direction, from the tool both `nexus-verification` and
`nexus-debugging` name as primary.

**(b) `implements` is 100 % inferred on token overlap.** `[M]` **all 16624**
edges carry `source="inferred"`, `confidence=0.7`; `[M]` **13512 (81.3 %)** rest
on a **single** shared token. Worst generics: `operator` 470, `method` 257,
`case` 116, `solve` 105, `source` 98, `apply` 77. Worked case:
`sn-cell-flatten-roundtrip` is "implemented by" `data.macro_xs.cell_xs.CellXS`
on `shared_tokens=["cell"]`, so a **CP** `test_production_rate_shape_and_sum`
becomes its verifying test. `[M]` **2781 of 16624** `implements` edges have a
TEST source (nexus #49's family, reported CLOSED upstream, materially present
in this graph).

**(c) `verified` has no evidence floor.** `query.py:1479` sets
`status="verified"` iff `len(tests) > 0`, where `tests` falls back to
`heuristic-1hop` (conf 0.7) then `heuristic-multihop` (conf **0.5**, ≤3 hops).
No threshold anywhere. `[M]` equation statuses `verified 692 / implemented 47 /
documented 164` = 903 (every equation ⟹ **76.6 % read verified**), and **351 of
692 (50.7 %)** carry **no declared test at all**. `[M]`
`nexus audit --include-tests` → `tests_declared 2748`, `tests_inferred 74553`
— a **27:1** ratio behind the headline.

### 3. The decisive one — static `calls` has 0 % recall on the relation that matters

`[M]` for `quadrature-ordinate-permutation` (single true implementer,
`Quadrature.ordinate_permutation`):

| route | reaches the implementer |
|---|---|
| static `calls` closure, depth ≤ 8, from the 21 claimers | **0 / 21** |
| runtime execution (coverage dynamic contexts) | **7 / 21** |

`[M]` `nexus callers` on that method → `{"nodes": [], "total": 0}` while three
real production call sites exist —
`sn/boundary/realizer.py:606`, `sn/loss_representation/__init__.py:3784`,
`geometry/boundary/_specular.py:131` — **all annotation-mediated** (nexus #16).
Consequence: `[M]` `nexus dead-functions` flags that method as a dead-code
candidate (3144 candidates total, 1563 in `orpheus/`).

⚠ **Instrument checks I ran before believing the 0.** Depth SATURATES: reach is
identical at 6/8/12/20 hops (673/1772 corpus-wide), closure median 74 nodes,
max 363 — the traversal runs. Positive control: `ReflectiveBoundary` (sole
implementer of `reflective-bc`, 0/45 reach) has `[M]` **73** static callers, so
the machinery finds callers when edges exist.

⛔ **A headline I wrote and then refuted.** I first read the corpus-wide
"38.0 % of claiming tests reach an implementer" as "62 % over-credited". It is
not: the number mixes static blindness with genuine over-credit, and **the
static graph cannot separate them**. Only the single-implementer equation let
me ground-truth it. The inseparability is the demand.

### 4. The crux — coverage produces per-test attribution; nexus drops it

`[M]` (a) with `dynamic_context = test_function` + `[json] show_contexts = True`,
`coverage json` carries a per-file `contexts` block: **23** non-empty contexts
on a 50-test / 0.72 s slice, each a fully-qualified test id mapped to the exact
lines it ran. `[M]` (b) `grep -c -i "context"
sphinxcontrib/nexus/runtime.py` → **0**; `overlay_coverage` (`:314-357`) reads
only `executed_lines` / `missing_lines` / `executed_branches` /
`missing_branches` and writes a per-NODE record with no per-test dimension.

`[M]` **The extension is ~15 lines and I ran it.** Reusing nexus's own
`build_node_index` (797 files / 10207 spans) plus the ignored `contexts` block:
**23/23** contexts joined, **1353** (test, exercised-node) pairs from a 0.72 s
slice — and it answers #334 directly with the same 7, cross-checked by an
independent `ast.parse` span resolution. `RuntimeRun`'s own docstring
(`runtime.py:118-127`) calls itself *"a bag of orthogonal overlays, not a tagged
union"*, so an `exercises` family is the shape it was designed for.

### 5. ⚠ `runtime-ingest` reports a SILENT ZERO-JOIN (shipped L54 class)

`[M]` `nodes: 0 / edges: 0 / unresolved: 0`, **exit 0**, no warning, on a real
report. Cause: `coverage json` emits **339 of 339** file keys RELATIVE while
`build_node_index` keys ABSOLUTE, so every file is dropped at `runtime.py:332-336`
— **upstream of the `unresolved` counter**, which therefore cannot see it.
`[M]` with the keys rewritten to absolute the same artifact joins **2892** nodes.

### 6. The re-baseline adjudication query — prototyped, and it names task #51's answer

`[M]` `tests/sn/regression/test_dd_regression.py` +
`tests/sn/sweep/curvilinear/test_tau_producer_equivalence.py`: 27 passed,
**59.03 s bare → 85.58 s under coverage (1.45×)**. Joined:
881 nodes touched by the regression snapshot, 174 by the non-regression pins,
**174 by both**, **707 pinned by nothing else** (two-file denominator — state it).
The shared set contains `pole_angular_closure.morel_montry_tau_per_level`,
`angular_cell_edges_per_level`, `_assert_tau_within_unit_interval` — i.e. the
object lesson **A10/D14** records as having been hunted **by hand, twice**, after
two structurally-blind candidates.
⚠ It measures **CO-EXECUTION, not co-constraint**: the output is a candidate list
to mutation-verify, not a licence. `coding-standards` re-baseline step 2 is
unchanged; what improves is that its *hunt* becomes 174 ranked candidates.

### 7. What the graph CANNOT own — the honest ladder

| rung | for #334's equation | established by |
|---|---:|---|
| CLAIMED (a `verifies` marker) | 21 | graph `tests` edges |
| EXERCISED (execution entered the code) | 7 | coverage dynamic contexts |
| ASSERTED (the assertion can fail for it) | ≤2 | **judgement — not measurable from any trace** |
| MUTATION-VERIFIED | 0 | **mutation only** |

All 7 exercisers are *identically* connected to the implementer in any
exercised-coverage graph, yet 5 cannot fail for a permutation error. **No edge
quality separates them.** #334's own "~2 exercise it" is refuted in magnitude
(7, a 3.0× over-credit, not 10×) and is a good estimate of the ASSERTING rung —
a different question.

### 8. Marker surface — partial, inconsistent, un-traversable

`[M]` AST census over all 456 test files vs graph `node_attrs`:
`foundation` **1515 usages / 308 files → NO attribute** (the suite's largest
marker, and the one whose whole meaning is *"not a physics equation"* — E1's
conflation vector); `regression` 11 / 10 files → **NO attribute** (5 nodes, only
via the `decorators` string); `sentinel` 19 → NO. Present: `verifies` (896),
`vv_level` (1530), `catches` (239), `slow` (143). `[M]`
`_collect_pytestmark_assignments` (`ast_analyzer.py:267`) DOES handle
module-level `pytestmark` — so the gap is *which markers are lifted*, not the
mechanism.

`[M]` `catches` is an **attribute, not an edge** (`SELECT COUNT(*) … type='catches'`
→ 0) and no `ERR-NNN` node exists, so "which tests catch ERR-026" is a scan.
Phantom check ran nearly clean: 80 claimed ids vs 79 catalog ids, the only
phantom being `'M-SEED-DROP'` (a deliberate free-form tag); **0** catalog
entries lack a marker. ⟹ report it, do not inflate it.

`[M]` The graph has **no** node for any of the 73 `.npy`/`.npz` frozen
references; `type='file'` is 94 `doc:` + 94 `std:` only. A snapshot cannot be
named, let alone adjudicated.

### 9. ⛔ My own proposed decay-detector, refuted by measurement

I proposed reusing `body_shingles` as the "has the fixture moved since the
mutation was demonstrated?" fingerprint. `[M]` it is **bit-identical** under
`rtol 1e-12→1e-6`, `max_inner 1000→50`, expected `1.234567→9.876543`, and
`mesh→mesh64` — *every* Mode-8-class-7 decay cause — and moves only for
structural edits (assertion dropped, jaccard 0.375; `allclose→array_equal`,
0.714). `fingerprint.py:_token` normalizes `Constant→"C"`, `Name→"N"` **by
design** (Type-2 clone robustness). A ledger built on it reports every decayed
marker FRESH.

### 10. What DOES work today

`[M]` The L53 denominator is mechanizable and the graph beats the glob: test
files on disk **456**, graph knows **491**, `disk \ graph` = **0**, and the 35
extras are the non-collected `conftest.py` / `_harness/*` / `_generate_*.py`
modules where Mode-8's real surface lives. Directory level: 37 on disk, 36 in
graph, the 1 missing holding only `__init__.py`. `staleness` runs clean (188
checked). `discriminations`, `twin-paths`, `protocol-conformers`,
`dead-references`, `graph-query` all run. `bridges` did not finish inside 120 s.
`nexus impact` mixes doc pages, modules and tests in one depth bucket (no CLI
`--edge-types`).

### 11. The demand, ranked (full text in the memo §9)

**D1** per-test `exercises` runtime family (~15 lines in `overlay_coverage` +
a `RuntimeRun` field + a context-id→node-id mapper + a `runtime-exercises`
verb) — everything else is downstream. **D2** ingest must never report a silent
zero-join (normalize paths; per-reason drop breakdown; non-zero exit on
`nodes == 0`). **D3** configurable lifted-marker set + `catches` as a real edge
to `err:ERR-NNN` nodes, with module- vs function-SCOPE distinguishable. **D4**
mutation verdict as staleable data (literal-SENSITIVE fingerprint, per §9).
**D5** a `nexus inventory` verb.

### 12. Three novel rationales owed to `vv-principles` — NOT landed (write fence)

(N1) an INFERRED relation must not be consumed under a DECLARED relation's name;
(N2) Mode-8 DECODER clause's **dual** — a normalized fingerprint reused as a
change detector inherits its deliberate blindnesses; (N3) #17 sharpening — a
RECALL counter downstream of a FILTER cannot count what the filter dropped.
Drop-in text is in the memo §14. The skill files were already dirty in the
working tree (concurrent agent) — reconcile before writing.

**Cross-refs.** ORPHEUS #358, #334, #309; nexus #16 (open), #26 (closed, 0.15.0),
#49 (closed upstream, live in this graph). `vv-principles` Mode 8 (7th class,
METHOD WARNING), Mode 10, Mode 12, #17, #19, #24, #25;
`coding-standards` re-baseline step 2; lessons L-053 (denominator), L-054
(verify the instrument ran), L-067 (audit-instrument decoder), L-069 (A10's
symmetric-difference mutation).

---

## L-071 — a retirement census's three silent killers: the unquoted `$VAR`, the wrapper `grep`, and a tree that moves under you

**Dispatch.** CS3-R completeness census (2026-08-19): find torsor-era machinery
surviving campaign-1 CS3's cone carve (flux `A` → cone `K ⊂ V`) as
present-tense/live content across `orpheus/ tests/ docs/ .claude/{skills,agents,rules}`.
Deliverable `scratch/cs3r_census_qa.md`. Verdict: **12 MUST-FIX rows survive**.

### 1. ⛔ My first two sweeps reported ALL-CLEAN across every tree. Both were VOID.

```zsh
TREES="orpheus tests docs .claude/skills .claude/agents .claude/rules"
grep -rn -E "$pat" $TREES 2>/dev/null || echo "(0 hits)"     # ← reports 0. ALWAYS.
```

**zsh does not word-split an unquoted `$VAR`** (my own H9, learned on a pytest
`-p` loop, and it did NOT transfer to a grep loop). The whole string went in as
ONE nonexistent path; `2>/dev/null` ate the `No such file or directory`; the
`|| echo "(0 hits)"` rendered the failure as a *finding*. I published "P1: 0
hits, P2: 0 hits" and would have shipped a clean bill on a corpus I never read.

⭐ Caught only because I recognised `FluxDisplacement` from the `coding-elegance`
skill loaded into my own preamble, and its absence from my results was
**impossible**. That is not a method — it is luck. The method is the **positive
control**, which I then ran before every subsequent sweep:
`grep -rl flux <tree>` → `195/301/72/12/8/2` files. One line, and it makes a
dropped tree indistinguishable-from-clean impossible.

⟹ **Fix: `TREES=(a b c)` + `"${TREES[@]}"`; NEVER `2>/dev/null` on a census;
NEVER `|| echo "(0 hits)"` (it launders rc≠0 into a result).**

### 2. ⛔ `grep` in this shell is a FUNCTION wrapping `ugrep --ignore-files`.

`type grep` → a zsh function dispatching to `ARGV0=ugrep "$CLAUDE_CODE_EXECPATH"
-G --ignore-files --hidden …`. `--ignore-files` honours `.gitignore`, so the
interactive `grep` is **blind to ignored files by default** — fatal for a census
whose brief says "untracked files matter". Use **`command grep`** (real BSD
grep, `.gitignore`-blind) as the primary instrument and **`git grep`** as the
tracked-only cross-check; reconcile the two counts numerically. `[M]` here:
`75 = 75` on `[A-Za-z]*Displacement|FluxRole|⊖` once `git grep`'s path list was
restricted to the same six trees (it had read **257**, because `-- .claude`
sweeps `plans/` + `agent-memory/` too — a denominator trap, not a discrepancy).

### 3. ⛔ H12 fired: the tree moved MID-CENSUS, twice.

`f43758d8` → `755f99b5` (18:27, "CS3-R sweep 1", 16 files) → `a740d7ba` (18:34),
plus 5 files dirty in the shared tree at 18:28. **Caught by a `sed` reading
"increments" where my own grep 4 minutes earlier read "displacements" at the
same `file:line`** — i.e. by an accidental discrepancy. ⟹ **On any census in a
shared tree, stamp `git rev-parse --short HEAD` + `date` at the START and at
the END, and re-run every finding as a PREDICATE at the end.** I did: 10/10
predicates survived at `a740d7ba`. Report the remediated set separately — a
finding silently fixed by someone else, left in the list, reads as a false
positive and discredits the rest.

### 4. ⭐ The finding class the carve's own author cannot see: the SELF-CONTRADICTING FILE

Every survivor but one sits in a file whose *other* lines were correctly
migrated. The corrected line and the stale line coexist, and **the stale one is
usually FIRST** (docstring above body, module header above class body):

- `_bases.py` — `:18` and `:1134` record the retirement; `:1160`/`:1220` still
  name `RadialCharacteristic*Displacement` as **live concrete role leaves**.
- `test_operators_apply_typed.py` — the concurrent sweep upgraded the
  **assertion** and rewrote the **body comment**; the **docstring 20 lines
  above** still says the opposite.

This is `vv-principles` #21's aggravator (the file can now be cited for either),
and it is *created* by a partial correction pass — a pass that fixes the site it
was looking at. ⟹ **After any correction pass, re-audit the CORRECTED FILE
FIRST**; it is the likeliest home of a survivor, not the least.

### 5. ⭐⭐ `AGENT.md` is the highest-severity surface in a retirement audit, and nobody sweeps it

3 of the 12 survivors are agent briefs, and one is an **imperative to re-mint the
overturned design**:

- `explorer/AGENT.md:151-155` — *"The role grid **is** {Flux, Source/Sink,
  Residual, **Displacement**} … the SI iterate-delta **is** a `FluxDisplacement`
  … `flux+flux` **is** a TypeError"* — triple-false, present tense, in the
  standing brief of the project's **designated exploration delegate**.
- `cross-domain-attacker/AGENT.md:297-301` — *"**FIX: a difference-space /
  torsor displacement type.**"* Its own source skill
  (`cross-domain-frames/reference.md` Shape 3) already carries the dated ⛔
  re-pose. **The AGENT.md is the un-migrated twin of a corrected single source.**
- `elegance-enforcer/AGENT.md:185` — a standing ruling telling the elegance
  GATEKEEPER *not to flag* a mixin that no longer exists.

⟹ Why this ranks above a production docstring: **AGENT.md loads FRESH per
dispatch** (`reference_harness_context_snapshot_timing`), so a stale brief is
re-injected as *current fact* into every future sub-agent, and its output is
indistinguishable from a correct one. ⟹ **Put `.claude/agents/*/AGENT.md`,
`.claude/skills/*/`, AND `.claude/agent-memory/*/` in the blast radius of every
type/concept retirement.** `[M]` the memory surface here is **182 lines across
~20 files** — larger than the three in-scope `.claude` subtrees combined (75) —
and I was not briefed to sweep it; I named it in the memo so its absence could
not be read as a clean bill.

### 6. ⭐ Derivative staleness: a correction's own TODO note outlives the correction

`coding-elegance/SKILL.md:390` carries `⚠ … reference.md … **still cites**
FluxDisplacement … numerical-bug-signatures §479/§488 **still credits** the
retired type … **Both are stale as of 2026-08-19**`. `[M]` **both named targets
have since been fixed.** The clause now instructs readers to distrust two files
that are correct. A note of the form *"X is stale and owes a correction"* is a
**claim about another file** and rots the moment that file is repaired — nothing
in X's own repair prompts anyone to retire the note. ⟹ **grep for pointers AT a
file you just fixed, not only inside it.**

### 7. ⭐⭐ The torsor-shaped call pattern was a REAL Mode-12 gate downgrade — and I measured it in 10 lines

5 operator-linearity gates verified additivity through the **affine detour**
`op(ψ₁ + λ(ψ₂−ψ₁)) = (1−λ)op(ψ₁) + λop(ψ₂)`, a workaround for a restriction that
no longer exists. **Affine maps preserve affine combinations**, so the detour is
*exactly* blind to an affine regression `A(x) = Lx + q`. `[M]` pure-numpy probe
(`n=6`, random `L`, `λ=0.7`), no ORPHEUS import, **no file touched** — safe in a
tree under concurrent edit:

| form | `q = 0` control | `q ≠ 0` (affine bug) |
|---|---:|---:|
| retired detour | `4.440892e-16` | **`4.440892e-16`** — bit-identical to the control |
| CS3 direct `A(ψ₁+ψ₂)` | `8.881784e-16` | **`1.288361e+00`** |

⟹ Two transferable moves. **(a)** A retired type's *workaround idiom* outlives
the type and is a coverage question, not a style question — ask what error class
the detour's functional annihilates (Mode 12 at the *idiom*, not the fixture).
**(b)** When the SUT tree is being edited by someone else, a **10-line pure-numpy
model of the two functionals** settles the blindness decisively without touching
a file — strictly safer than a mutation battery and, here, equally decisive.
I then verified the concurrent fix moved the **assertion**, not just the prose,
by `git show <old>:<f>` vs working tree — `apply(psi1 + lam*(psi2-psi1))` →
`apply(psi1 + psi2)` in all five. **A prose-only fix would have left the
blindness with a corrected comment on top.**

### 8. Clean checks that are first-class output

`⊖`-mint in production (**0** — `_principal_bulk_leaf` at `iteration.py:412`
replaced `_flux_displacement_leaf`); the `Σλ=1` ceremony (**0** on the flux path
— all 40+ "partition of unity" hits are the unrelated energy-condensation
overlap table); **live Sphinx roles at a deleted target** (2 found in
`tests/`, both fixed concurrently → **0 tree-wide**; note no Sphinx severity
including `-n` could ever see them, since no `automodule` renders `tests/`);
retired-message test pins (**0** — the near hit
`match="boundary must be a BoundaryField"` cannot match
`"…RadialCharacteristicBoundaryField"`, which is what licensed recommending a
reword); and `affine-bc-form` **LIVE and intact** with many `:eq:` citers — the
ambiguity the brief warned about, confirmed unharmed, while the three genuinely
retired labels are gone with **0 dangling citers**.

⚠ And one finding found *while* checking a clean one: a retirement tombstone at
`test_radial_characteristic_field.py:146` names
`test_subtraction_mints_a_displacement_composite_per_block` as the successor
carrying a surviving claim — `[M]` that name exists **nowhere** in the tree
(CS3 renamed it). **A dead reference inside the very artefact whose job is
keeping coverage traceable** (F13's tombstone family, one level worse: not a
mis-split claim, a pointer to nothing).

**Skill/rule homes:** `vv-principles` #21 (self-contradicting file), Mode 12
(the detour's invariance group), #17 (positive control — here on a *grep*, not a
mutation); `coding-standards` retirement 3-search rule (extended: the three
`.claude/` surfaces); `plan-authoring` §2 (validate the FILTER, don't merely
write it down); lessons H9 (unquoted `$VAR`), H12 (the subject moves), H7
(shared tree), F13 (tombstones).

---

## L-072 — a design assembly's TELL is a gate: one was designed-RED by its own non-goals, and its central claim was true-in-the-arithmetic and false-at-the-codomain

**Context.** 2026-08-20, branch `feature/cs1-energy-space` @ `71515847`.
Adversarial Phase-1 review of ONE of three independent CS4a design assemblies
(`scratch/cs4a_assembly_physics.md`, the "physics-first" lens) against the
charter `.claude/plans/kernel_and_medium_objectives.md` (objectives O1–O9,
constraints C1–C10). Rivals: `cs4a_assembly_algebra.md`,
`cs4a_assembly_parsimony.md`. Verdict memo:
`scratch/cs4a_attack_physics.md`. **This was a design review, not a code
review — and the lesson is that the two need the same instruments.**

### 1. A design objective's "tell" is a coverage claim, and it can be DESIGNED-RED

The charter demanded a *falsifiable tell* per objective, which is the right
demand and is exactly `plan-authoring` §1's "done-when is a checkable
predicate". The assembly answered O7 ("one spelling per concept") with a tell
that is an instrument:

> "`grep -rn "SigS" orpheus/` finds the datum owned once (Mixture) and viewed
> once (ScatteringKernel)."

`[M]` I ran it: **70 hits**. Direct `Mixture.SigS` consumers:

| site | read |
|---|---|
| `orpheus/cp/solver.py:511` | `self._scat_mats = {mid: materials[mid].SigS[0] for mid in materials}` |
| `orpheus/moc/core.py:93` | `self.sig_s0.append(mix.SigS[0].toarray())` |
| `orpheus/mc/solver.py:369` | `sig_s_dense[mat_id] = np.array(mix.SigS[0].todense())` |
| `orpheus/sn/solver.py:1281` | `min(len(m.SigS) - 1 for m in materials.values())` |
| `orpheus/derivations/continuous/**` | 8 sites |

**Every one is inside the assembly's own §5 "Untouched" list**, and the
charter's non-goals bar re-routing solver entries. So the tell's answer is
pinned at FALSE the day the work lands, and no step in the slicing could move
it.

⭐ **This is `plan-authoring` §10's defect with a THIRD shape and the opposite
colour.** §10 names two: a metric keyed on a PROXY the work removes, and one
over a POPULATION the work removes members from. This is a population the work
is **forbidden** to touch — so the metric is *designed-red*, the mirror of
`vv-principles` #17's designed-green harness. And it survives longer than
designed-green would, because a permanently-red tell reads as *work remaining*:
a later session picks up the campaign, sees the tell failing, and chases a
target that was never reachable.

⚠ The aggravator, and the reason it is worth a rule: **naming an instrument
makes a done-when read as MORE rigorous than a prose one.** "Grep returns
past-tense only" looks unimpeachable next to "the concept has one owner". The
two rival assemblies both stated the same objective with tells scoped to the
CARVE (parsimony: "the kernel datum's grep-findable owner is `MaterialXSField`
(CS4a) / `ScatteringKernel` (CS4b)"; algebra: "3-search retirement audit per
symbol + `dead_references`") and neither is refutable by a tree-wide grep.

⟹ **Run a tell's own predicate at design time, over the whole tree, and
intersect the hits with the design's declared UNTOUCHED set.** Every hit inside
that set is a permanent counterexample. One command.

### 2. A locality premise can be TRUE and its architectural corollary FALSE — the discriminator is the CODOMAIN constructor, not the arithmetic

The assembly's central idea:

> "Collision, scattering, and fission are all *spatially local* (diagonal on the
> spatial axis) … **That physics fact dictates the whole binding signature — a
> spatial mesh is never data of the interaction, only of the pullback.**"

**Premise CONFIRMED, all four in-scope channels** (I checked the arithmetic, not
the prose): C = `DiagonalOperator(coefficient.values, broadcast_axes=(0,))`;
IsoS = `einsum("fg,fc...->gc...", sig_s0, phi[cells])` per material
(`material_xs_field.py:850`), `c` a spectator; IsoN2N the same × 2.0 (`:880`);
F = `outer(chi, ReactionRateFunctional(νΣf)) & IdentityOperator()`
(`fission.py:318`) whose functional contracts **only** the group axis
(`:320-368`) and whose `& IdentityOperator()` **is** the spatial broadcast. Even
the out-of-scope angular operator is diagonal (`einsum("mfc...,fg->mgc...")`,
`:998`).

**Corollary REFUTED.** Locality says the mesh is not needed to COMPUTE the
action. It says nothing about producing the CODOMAIN ELEMENT — and in this tree
every composite arm emits `FullField(interior=X…SourceSink.from_mesh(values,
mesh), boundary=X…BoundarySourceSink.zeros_on(mesh))`. Those constructors take a
mesh, and the only object in the frame that has one is the **operand**:

| operator | sites |
|---|---|
| `MultiplicationOperator` | `:432, :444` (apply, 2 arms), `:507, :517` (solve, 2 arms) |
| `IsotropicScattering` + `IsotropicN2N` | `isotropic_scattering.py:132` (shared `_scalar_composite_source`) |
| `FissionOperator` | `fission.py:443, 464, 465` (transpose), `:580, :615` (apply), `:649` (ScalarFlux arm) |

`[M]` **≥11 sites**; the assembly named **2**.

**And binding cannot remove them** — I measured what the bound space carries:

```
full_field_space: name='full_field' shape=(48,) .axes = None
  interior_space: name='sn_bulk'    shape=(4,2,4) .axes = None
  trace_space:    name='angular_trace'            .axes = None
```

`FullFieldSpace` = `name/shape/inner_product_weights/axes` (inherited,
`numerics/space.py:196`) + `interior_space/trace_space`
(`full_field_space.py:192-197`). **No mesh, on the composite or on either
block.** So "the realization is selected at construction from the space" leaves
the selected body still needing `bulk.mesh`.

⭐ **The tree states the OPPOSITE of the thesis, in a production guard**:
`sn/operators/streaming.py:589` — `if streaming.sn_mesh is not
diagonal.coefficient.mesh: raise ValueError(...)`, comment *"The diagonal
multiplier is mesh-free; **its mesh is carried by its CrossSectionField
coefficient**."*

⟹ **When a design says "X is not data of this operation", check the CODOMAIN
constructor, not the arithmetic.** A pure/local/diagonal kernel can still need
X to BUILD its result, and typed-carrier codomains are exactly where that bites.

### 3. Where the assembly's error came from — one word, two referents

`multiplication_operator.py:82`: *"the mesh is read off the carrier at apply
time (``mesh = psi.interior.mesh``)"*. In the **production docstring** "carrier"
means the FLUX carrier (the `FullField` operand); in the **charter** "carrier"
means `MaterialMesh`/`MaterialXSField`. The assembly inherited the docstring's
word and declared the read retired by a change that touches the *other* party.
The rival algebra assembly got it right by reading the code rather than the
prose (`§0 row 6`: "C reads `bulk.mesh` off the **OPERAND**"). ⟹ in a codebase
with two vocabularies for one word, a prose-sourced `[M]` inherits the wrong
referent silently.

### 4. The denominator nobody wrote: 4 of 13 bindings are axis-built

The assembly's O2: *"kernel `ng` == the space's `EnergyAxis` shape … **`ng` is
never passed — it is already IN the space**"* — flat, no denominator.

`[M]` per constructor call (13 calls at 10 sites, re-derived):

| space threaded | `.axes` | calls |
|---|---|---|
| `mat_xs.mesh.bulk_space` | `[('energy',(2,)),('spatial',(1,))]` — axis-built | **4** (`homogeneous/solver.py:152,155,157,204`) |
| `mesh.full_field_space` | **`None`**, both blocks `None` | **7** (`sn/coupled_system.py:419`; `sn/solver.py:1339,2804`; `diffusion/solver.py:236,242,243,247`) |
| none (space-anonymous) | — | **2** (`scattering.py:713`) |

Diffusion's interior block is hand-built with **no `axes=`**:
`FunctionSpace(name="scalar_bulk", shape=(ng,*spatial), inner_product_weights=V…)`
(`diffusion/augmented_mesh.py:362-366`), `space = mesh.full_field_space` at
`diffusion/solver.py:234`.

So the guard is axis-structural on **4 of 13** and degrades to a positional
shape-slot read on 7 — the majority path — i.e. a SECOND spelling of the
conformity rule, minted by a step whose own O7 says *"a silent second spelling
disqualifies"*. ⭐ **The parsimony rival publishes this exact defect as a
SELF-attack about its own design; the physics assembly does not state it at
all.** ⟹ when three independent assemblies answer one charter, **read every
rival's self-attacks as a checklist against your target** — a self-attack is a
measured weakness someone already did the work on, and the target that is silent
about it is the one that did not look.

### 5. Self-attacks as decoys — the shape

Both of the target's self-attacks were REAL and both were the objections the
**rival documents publish about themselves** (physics Attack 1 ≈ parsimony
Attack 2; physics Attack 2 ≈ parsimony Attack 1). Neither was manufactured. But
each lands on the *arguable* half of a seam whose *factual* half is worse:

- Attack 2 ("construction-time selection is dispatch relocated") argues
  instance- vs class-monomorphism — deflectable with "C2 says nothing about one
  class per space" plus the true fact that G1.2's carrier registry does not exist
  (#261). The factual half of the same seam is §2 above: the selected body still
  needs a mesh it can only get from the operand, and no registry changes that.
- Attack 1 ("the medium is a third spelling") is correct and stops before the
  conformity-guard denominator (§4) and before the reaction-rate fork (§6).

⟹ **A self-attack marks the seam, not the depth.** Take the seam it names and
push one level past the argument it answers — the prepared defence is the tell
that the author stopped there.

### 6. The §6b set of a design step is not its CONSTRUCTION sites

The assembly enumerated `[M]` 13 constructor calls (correct — I re-derived all
13) and called that its §6b completeness. Three signature changes its own
slicing performs have call-site sets it never enumerated:

- **`MultiplicationOperator.coefficient`** stops being uniformly a
  `CrossSectionField` under its K2 ("no `CrossSectionField.from_mesh`, no
  mesh") — production READERS at `streaming.py:589` (`.coefficient.mesh`),
  `:597`, `:659` (`.coefficient.values`).
- **`IntegratedReactionRate`** — `[M]` **7** production sites
  (`homogeneous:223,224`; `diffusion:337`; `sn/solver:1545,1596,1599,1716`),
  `evaluate` reads `self.cross_section.mesh.volume_measure`
  (`reaction_rate_functional.py:210,228`). The slicing re-poses 2 and leaves 5,
  and `reaction_rate_functional.py` appears **nowhere** in its blast radius ⟹
  two spellings of the volume-integrated reaction rate, which is O7's own
  disqualifier. (Parsimony makes it a dedicated step over all sites; algebra
  says the spelling "generalizes instead of forking". Physics alone forks it.)
- **The P1 xfail marker**, next section.

### 7. A partial ledger flip is not runnable until the marker is split (positive control run)

`[M]` `python -O -m pytest tests/sn/architecture/test_monomorphic_leaves.py -q`
→ `82 passed, **16 xfailed**`; `-rx` decomposes exactly as all three assemblies
claim: 5 R1-annotation `[L,C,S,F,B]`, 3 R2-anonymous `[C,S,F]`, 8 R6
`[B × 4 geometries × 2 carriers]`.

`_R1_XFAIL` is a **test-level** decorator at `:701` over all five params;
`_R2_XFAIL` at `:1026` over all three. Only `_R6_XFAIL` is per-row
(`_G13_ROWS`, `:735-747`, the `marks=[...] if leaf == "B"` pattern).

Positive control (scratchpad, throwaway module, `python -O`) emulating "C and F
fixed, L/S/B still broken" under ONE test-level `xfail(strict=True)`:

```
FAILED test_xfail_split_probe.py::test_rows[C] - [XPASS(strict)] ...
FAILED test_xfail_split_probe.py::test_rows[F] - [XPASS(strict)] ...
2 failed, 3 xfailed
```

⟹ "CS4a deletes 4 rows" is not executable until the two decorators become
per-`pytest.param` marks. The algebra rival flags it and schedules the split
inside its step; the target does not mention it. **Before crediting any
per-row ledger apportionment, check whether the marker is attached per-row or
per-test — the summary line reads identically either way.**

### 8. The type-vs-property rule, applied per channel — neither side survives whole

The three assemblies forked on which kernels to mint. Applying
`coding-standards`' rule honestly (mint **iff** ≥2 non-isomorphic realizations
AND a non-identity morphism is applied):

| channel | (a) realizations | (b) morphism | verdict |
|---|---|---|---|
| scattering | ℓ=0 scalar / moment space `SphericalHarmonicSpace.from_L(L)` / frame-conjugated ordinate | ℓ-restriction, `frame.conjugate`, `with_overridden_sig_s_and_n2n` | **MINT** (all three agree) |
| collision | one (diagonal coefficient) | identity only | **PROPERTY** (all three agree) |
| **(n,2n)** | **two — the tree says so**: `isotropic_scattering.py:32-38` "Both are the **scalar (ℓ=0) realization** of … `N2NMomentOperator`"; `N2NMomentOperator.domain = SphericalHarmonicSpace.from_L(L)` (`scattering.py:344-351`) | frame conjugation inside `full_scatter_kernel = frame.conjugate(Λ + N₂ₙ)` | **MINT** — physics+algebra right, **parsimony REFUTED** |
| **fission** | **one** — the angular arm is `integrate_angular()` → the ScalarFlux body → `AngularSourceSink.from_isotropic` (`fission.py:569-604`), i.e. the ℓ=0 realization wearing the ℓ=0 moment maps. Fission emission has no ℓ≥1 content. | the χ↔νΣf swap is `RankOneOperator`'s transpose, already owned | **PROPERTY** — parsimony right on the verdict, **wrong on the reason** |

Two residues worth keeping:

- **Parsimony's n2n fold silently overturns a documented physics ruling**, stated
  twice and emphatically: `isotropic_scattering.py:26-30` ("a DISTINCT
  *multiplication* channel … so it stays its own operator") and
  `scattering.py:300-310` ("**Keeping the multiplication reaction a visible
  distinct operator, rather than hidden in the scattering matmul, is the
  physics-faithful choice**"). Folding Σ₂ₙ in as a "channel view" IS that
  hiding — a C6-class silent overturn, executed in a §4 parenthesis.
- **The one real (b)-candidate for fission is CONDENSATION, and it argues
  against the mint.** `MaterialXSField.project_through` (`:343`, body
  `:500-540`) applies a genuinely non-identity, **χ↔νΣf-COUPLED** morphism — χ
  collapses through a `PetrovGalerkinFrame(WeightedIndicatorBasis(trial,
  iota*p))`, then νΣf's mixed fold consumes the *just-condensed* χ
  (`iota_tilde = (phi_star * chi[region_of_fine]).sum(...)`, `:517-520`). But it
  is owned **whole-mixture** and returns `dict[int, Mixture]`; per-channelising
  it to justify `FissionKernel` would fork the one condensation body four ways —
  the exact Pattern-2 violation the same assembly's §4 non-mint #3 cites as its
  reason to refuse per-channel FIELDS.

⟹ Honest ledger: scattering **mint**, n2n **mint**, fission **property**,
collision **property**. Each assembly is 3-of-4, on different rows. ⭐ **When a
review is asked to adjudicate a fork "per X", run the rule per X — the answer
was not a side.**

### 9. The Funk–Hecke / transpose question — no divergence, and two caveats owed

Asked whether the iso operator's `Σ_s0ᵀ` convention makes it a physically
different object from the ℓ=0 slice of the angular operator. `[M]` both
contractions:

| verb | forward | transpose |
|---|---|---|
| `apply_p0_in_scatter` (`:850`) | `einsum("fg,fc...->gc...")` = `Σ_s0ᵀφ` | `einsum("fg,gc...->fc...")` = `Σ_s0φ` (`:916`) |
| `apply_legendre_scattering_moments` (`:998`) | `einsum("mfc...,fg->mgc...")` = `Σ_sℓᵀ` per ℓ | `einsum("mfc...,gf->mgc...")` = `Σ_sℓ` (`:1054`) |

Both contract the SOURCE index against the first axis of the stored
`[g_from, g_to]` matrix (`mixture.py:32`). **The "ᵀ" is a property of the
STORAGE convention, shared identically by both realizations** — not a
distinguishing convention. And the tree ALREADY ships the decomposition:
`apply_legendre_scattering_moments` defaults `skip_l0=True` because "the P0
in-scatter goes through `apply_p0_in_scatter`" (`:970-973`), and the forward
`AngularFlux` arm assembles `iso(ℓ=0) ⊕ (1/W)·frame_conj(ℓ≥1)` with a
load-bearing perf ruling that the iso half **must not** route through the frame
(`scattering.py:1202-1208`).

⛔ Two caveats to demand before any CS2 "the frame IS the kernel's eigenbasis"
claim: (i) Funk–Hecke diagonalizes the **angular** factor only — the eigenvalue
is a dense `(ng,ng)` matrix per ℓ, so it is a partial diagonalization; (ii) the
eigenbasis is exact for the CONTINUUM and approximate for the discretization,
to the quadrature's moment-exactness — the `Π R = 4π·I` territory of ERR-039 /
`assert_galerkin_idempotency`. A "theorem" phrasing without (ii) is the next
ERR-051-shaped invariant claim.

### 10. Filter hygiene — definition-line contamination, twice, in one document

The assembly reported `[M]` "17 test sites" of `from_materials` (measured: **16**
— the 18-hit grep includes the definition at `material_mesh.py:242`) and "12 hit
lines minus 2 docstring hits" for a grep returning **13** non-definition lines.
Same cause both times. Both rivals got 16. ⟹ a census filter that does not
exclude `def `/`class ` lines is the filter that will also build the retirement
migration list.

### 11. What the target measured BETTER than its rivals (Phase-2 material)

Credit where it is owed, because a review that only subtracts is not calibrated:

- **Doc blast radius, exact.** `[M]` re-derived: exactly 2 `from_materials` refs
  in doc source (`docs/theory/foundations/spaces.rst:1030`,
  `infinite_medium.rst:1115`) and exactly the 5 named `.rst` files carrying
  `bulk_space`/"degenerate carrier" prose. Neither rival measured this.
- **The `DiffusionMesh` guard-ordering argument holds**: `ndim != 1` at
  `diffusion/augmented_mesh.py:203-210` fires BEFORE the `mesh is None` arm at
  `:211-219`, so that arm really does become dead code once the carrier retires.
- **The SN bare-assert unreachability claim is TRUE** (asserted without
  measurement; I closed it): the only surviving `mesh=None` producer is
  `SNMesh.from_axes`, which sets `mesh=None` **only when `len(axes) > 2`**
  (`sn/mesh/augmented_mesh.py:697-700`) ⟹ d≥3 Cartesian ⟹ `ndim != 1` ⟹ the
  `else: self.reduced = None` branch, never the `assert isinstance(mesh,
  Mesh1D)` at `:322,329,347`. Today's reachability is
  `SNMesh.from_material_mesh`, which passes `mesh=material_mesh.mesh` verbatim
  (`:755`) — that route dies with the carrier. ⚠ but its **`.areas` claim is
  FALSE**: `_areas is None` has THREE producers (`material_mesh.py:207-216`) —
  carrier, d≥3 axis-native, **and `isinstance(mesh, Mesh2D)`** — so a d≥3 carrier
  still raises `"not defined for 2-D meshes"` (`:521-525`), and d≥3 axis-native
  is the very meaning the assembly's own K3 says the sentinel collapses to. Its
  O8 row and its K3 row are mutually inconsistent.

---

## L-073 — a design that collapses runtime dispatch onto a construction-time KEY is asserting a claim about TRAFFIC, and an inventory of ARMS is not that claim

**Context.** ORPHEUS campaign-1 CS4a, round-1 adversarial review of three
independent design assemblies (`scratch/cs4a_assembly_{algebra,physics,parsimony}.md`)
against the charter `.claude/plans/kernel_and_medium_objectives.md`. Branch
`feature/cs1-energy-space` @ `71515847`, tree clean. My verdict memo:
`scratch/cs4a_attack_algebra.md`.

**The shared design move.** All three assemblies proposed the same central
mechanism, in near-identical words: the operator's constructor "selects the one
apply body the space's carrier family implies", after which "apply-time
isinstance ladders retire because the question they answered ('what arrived?')
is answered at construction ('what was bound?')". Each justified it with a
**static inventory of the dispatch ARMS** — the `singledispatchmethod` registry,
the `isinstance` branches, line-numbered and correct.

**Why the inventory is the wrong instrument.** An arm inventory enumerates what
the CLASS can receive. The design asserts what each bound INSTANCE actually does
receive. Those are different populations and only one of them is measurable
statically. Nothing type-errors when they diverge — the selected arm is *an* arm
the class supports — so the failure is silent, and silent in the flattering
direction (the code looks monomorphic; production just stops reaching the body
it needs).

**The census (~15 lines, one solve per family).** Wrap `cls.__dict__["apply"]`
through the descriptor protocol — `type(orig).__get__(orig, self, type(self))(x)`
— so `singledispatchmethod` still dispatches (naively re-binding `cls.apply`
breaks dispatch and raises, which reads as a code bug); log
`type(operand).__name__` keyed on `id(instance)`. **Positive control (#17):**
the workload's headline number must be bit-identical with and without the
wrapper. `[M]` SN keff `0.18764940308862563` both ways.

`[M]` 2026-08-20, one solve per family (SN k-eigen S4 2-region non-uniform slab
using the ledger file's own `_two_region_fissile()` / `_NONUNIFORM_EDGES`; 1-D
diffusion on the same mesh; homogeneous k∞):

| family | op | bound `space` | carriers arriving at `apply` |
|---|---|---|---|
| SN | **F** | `FullFieldSpace` | **`ndarray` ×17 and NOTHING else** |
| SN | S | `FullFieldSpace` | `TimedFullField` ×225 **AND** `AngularFlux` ×225 |
| SN | IsoS / IsoN2N | **`None`** | `ScalarFlux` ×225 (→ returns bare ndarray) |
| SN | C | `FullFieldSpace` | **never entered** (fused `L+C` owns the body) |
| DIFF | C | `FullFieldSpace` | `FullField` ×25 |
| DIFF | **F** | `FullFieldSpace` | `FullField` ×25 **AND** `ScalarFlux` ×25 |
| DIFF | **IsoS / IsoN2N** | `FullFieldSpace` | **`ndarray` ×27 AND `FullField` ×25** |
| HOMO | C, F, IsoS, IsoN2N | bulk `FunctionSpace` | `ndarray` ×2 |

**6 of 12 production instances refute the mechanism**, three distinct ways:
(a) **wrong arm** — `SNSolver.fission_op` is bound to `sn_mesh.full_field_space`
(`sn/solver.py:1339`) and its ONE production apply is
`self.fission_op.apply(flux_distribution)` with `flux_distribution: np.ndarray`
(`:1439`), reached from `numerics/eigenvalue.py:420`, i.e. the k-eigenvalue outer
iteration. Selecting from the space picks the composite arm and orphans the only
arm production uses. (b) **non-determination** — DIFF-IsoS/IsoN2N, one instance,
one space, two carrier families in one solve. (c) **asymmetric arrow** — the iso
"bare" arm is `_values_of(phi) = np.asarray(getattr(phi,"values",phi))`
(`isotropic_scattering.py:96-98`), so it takes a typed `ScalarFlux` and returns
a bare ndarray; no single carrier-family label names its domain AND codomain.

⭐ **The aggravator, and the reason this is a lesson rather than a finding.** One
assembly had listed exactly this census as its own **strongest self-attack**,
named the right suspect, and deferred it to execution time, describing its
absence as *"a plan, not a fact."* Run, it is not a precondition on the design —
it is the refutation of it. And its one measurable mitigation clause (*"the
scalar/bulk ndarray arm and the composite arm never coexist on ONE space"*) is
`[M]` false on the DIFF-IsoS row. ⟹ **when a document flags a missing
measurement as its own weakest point, RUN IT FIRST** — the author has already
localised the defect for you, and a self-attack is the cheapest place in a review
to convert a doubt into a verdict.

**The sibling, and why it is a different item.** `vv-principles` #28 (landed by a
peer in the same round) covers a key that is **ABSENT** — a guard on an operand's
optional metadata is inert where the field is `None`. This is a key that is
**present, correct, and non-determining**. Same review moment, different
measurement (build the object vs. run the workload), different repair (key on
what the object always carries vs. keep the operand read). Drop-in text for
**#29** is in the verdict memo §5.2 — **OWED and deliberately NOT landed**: the
dispatch charter was "change NOTHING except writing your own memo".

**Two further re-derivations worth keeping.**

1. ⛔ **#28's own `[M]` "8 of 13" is 7 of 13.** Split, construction by
   construction: `full_field_space` ×7 (`sn/coupled_system.py:419`,
   `diffusion/solver.py:236/242/243/247`, `sn/solver.py:1339/2804`);
   `bulk_space` ×4 (all `homogeneous/solver.py`); anonymous ×2
   (`scattering.py:713`). The slip is counting `homogeneous/solver.py:152` as
   `full_field_space` because `from_mesh`'s chain names it first
   (`multiplication_operator.py:343-346`) — `[M]` executed, `MaterialMesh` has
   **no** `full_field_space` attribute, so it falls through. #28's conclusion
   survives; its number needed the correction.
2. **The metric is INERT on all four energy-local leaves, on every bulk space.**
   `[M]` `.H` vs `apply_transpose` on a graded spherical bulk space with
   `V_cell` spread **3358×**: C `0.000e+00`, IsoN2N `0.000e+00`, IsoS
   `4.441e-16`, F `2.220e-16`. Mechanism in closed form: C/IsoS/IsoN2N/F are
   spatially diagonal (energy-only transfer per cell), `G = V_cell ⊗
   counting_energy` is diagonal, so `[G, Aᵀ] = 0`; and the energy half is
   counting **as a theorem** (`axis.py:226-239` — a weighted `EnergyAxis` is
   *refused*). ⟹ the R2 ledger row's stated harm ("`.H` silently degrades to a
   bare Euclidean transpose") is numerically **inert** for these four on any bulk
   space, so **no `.H` gate can witness their Optional→mandatory flip**; the
   honest CS4a witness is a construction refusal. One assembly proposed exactly
   such an adjoint gate as one of its four CS4a gates, having *written the
   blindness into the gate's own justification* (*"instead of the R2 silent
   degradation that today produces the same equality"*). Measured `0.0` under the
   defect, `0.0` under the fix. `vv-principles` #19 + Mode-12 commutator.

**The §6b half, for the next reviewer of any mandatory-parameter flip.** All
three assemblies enumerated the PRODUCTION call sites (10 sites / 13
constructions — I re-derived it; correct) and **none** counted the test side.
`[M]` regex over `tests/**/*.py`, comments dropped, `space=` in a 6-line window:
`MultiplicationOperator` 145 constructions of which **131** space-less across 43
files; IsoS 19/13/5; IsoN2N 14/11/5; F 15/10/6 — **165 space-less constructions
in ~50 files**. The step is 16× its planned size, and the under-count is in the
half that holds the coverage.

## L-074 — a guard hoisted to ONE home has as many arms as CALL SITES; and three "independent" comparisons that were one expression compared with itself

CS4a-R Phase-1 gate review, 2026-08-21, `feature/cs1-energy-space` @ `a9a2d55a`.
Gates: `tests/transport/test_kernels.py` (51 rows),
`tests/homogeneous/test_operator_spaces.py` (19),
`tests/sn/architecture/test_monomorphic_leaves.py` (85 + 14 xfail). All green
under `.venv/bin/python -O -m pytest -p no:randomly`; D5 byte gate 8/8.

**1. The hoisted-guard arm count (the digest rule A12).**
`orpheus/transport/operators/_energy_conformity.py` is one shared body called
from FOUR sites — `fission.py:201`, `multiplication_operator.py:214`,
`isotropic_scattering.py:263` and `:380`. Its gate,
`test_energy_conformity_guard_three_rows`, exercises **F only**. Per-site
no-op mutation (in-process plugin, `tests/transport/ + tests/homogeneous/ +
the ledger`, 655 rows): **F → 1 red; C → 0; IsoS+IsoN2N → 0**. `grep -rn
"energy extent" tests/` = **1 assertion**, in the F row. The C site is the one
that passes a DIFFERENT expression (`self.coefficient.values.shape[0]` vs
`self.mat_xs.ng` at the other three), i.e. the arm most able to be miswired is
the one with no witness. ⭐ The mechanism generalises: Pattern 2 hoists the
BODY, never the WIRING — so the elegance move *creates* the blind spot, and
vv#17's granularity rule (written for in-body early-return arms) has to be
re-read as *per call site*.

**2. Three "independent" comparisons that are one expression, measured.**
`ScatteringKernel.from_mixture` is `tuple(np.asarray(s.todense()) for s in
mixture.SigS)`; `MaterialXSField._build_dense_caches` is `[np.asarray(s.todense())
for s in mix.SigS]`. Same expression, same object. G1.3's docstring licenses
`array_equal` on the ground that the two sides are "independently assembled".
`[M]` transpose the kernel side ALONE → **2 reds** (G1.3 asymmetric row, G1.4);
transpose BOTH (the shared-source defect = `SigS` stored `[g_to,g_from]`, a
Mode-2/6 convention inversion) → **51/51 GREEN**, and the whole of
`tests/transport/` is green. The convention IS pinned — but only in
`tests/homogeneous/` (**17 reds**, incl. the L1 `test_kinf_exact` anchor and
the continuous reference). Same shape at G1.4: `dense_per_material` is
`sig_s_legendre(mid)[0].T`, so `p0 == iso[mid].T` cancels to an identity; what
G1.4 genuinely pins is the transpose CONVENTION between two named views and the
`(n,2n)` multiplicity, spelled independently as `ClassVar 2` vs a literal `2.0`.
And `chi_per_material` returns `materials[mid].chi` itself, so that leg is a
copy compared with its source. ⟹ `coding-standards`' single-sourcing-demotion
clause, arriving at BIRTH rather than at a retirement: the twin was minted and
immediately compared with its own source.

**3. `.H == apply_transpose` on a counting space is the wrapper calling the
callee.** `[M]` `space.apply_metric(x) **is** x` on the minted quotient space,
and `_AdjointOperator.apply` (`numerics/operator.py:1307-1313`) is
`G_dom⁻¹(apply_transpose(G_cod·y))`. So G2.7 asserts `f(x) == f(x)`. Proven:
under a dense AFFINE `MultiplicationOperator.apply_transpose`
(`A@y + 5`, non-diagonal, not a transpose of anything) the G2.7 equality reads
**True**. Its docstring's named falsifier — *"a leaf gaining a non-diagonal
energy coupling"* — is unreachable. Its ONE live falsifier is the wrapper
ceasing to delegate: mutating `_AdjointOperator.apply` to `1.5·y` reds G2.7 and
D4b. The gate's ⛔ metric-blindness block is otherwise exemplary — the defect is
that a correct disclaimer was paired with a falsifier that cannot fire.

**4. Two ATTACKS I ran and had to withdraw, both by my own probe.**
(a) *"G2.4's `MaterialMesh.volumes ×2` monkeypatch is the silent-no-op-that-
lies-safe"* — REFUTED: it is a plain `property`, and on the meshless
`from_materials` carrier (`mesh is None`) `volume_measure` takes the
`self.volumes.ravel()` arm; measured `weights 1.0 → 2.0` and `bulk_space`
changes. The instrument is live, the null is a real measurement.
(b) *"G2.3's frozen rate literals are derived from the code they pin"* —
REFUTED: all six reproduce EXACTLY from an independent
`float(np.sum(sigma*phi))` with no space, no operator, no
`IntegratedReactionRate`. ⟹ **run the probe before writing the finding**; both
attacks were plausible from the source alone.

**5. The self-auditing gate whose denominator is a hand-written list.**
`test_ledger_xfail_marks_are_strict` iterates `(*_R1_ROWS, *_R2_ROWS,
*_G13_ROWS)`. `[M]` a non-strict mark inside a covered list → **RED** (teeth
confirmed); the same mark in a NEW module-level list → **GREEN** (invisible);
a function-level `@pytest.mark.xfail` decorator is never in any row. The
docstring's own premise checks out (`grep -rn xfail_strict` = 0 across
`pyproject.toml` and every `conftest.py`), which is exactly why the evasion
matters. vv#13: the listed elements must GENERATE the audited set — walk the
module namespace / `request.session.items` instead of naming three globals.

**6. Population arithmetic worth carrying.** G2.1 says "on all 8 D5 cases";
`[M]` **1 of 8** (`homo_2eg_with_eg`) carries `eg`, and it is the sole case that
reds under a second-spelling-at-`_pose_space` mutation — the other 7 are
synthetic on both sides. G2.1 is also GREEN when the SHARED rule itself is
broken (correctly: G1.6 owns that, and reds 3 rows there). And the K1 module
docstring's fixture rationale is measurably wrong in one clause: `make_mixture`
DOES take `sig_s1`, and **all 12** shipped `get_mixture(region, ng_key)` pairs
ship `len(SigS) = 2` (order 1). What IS true of it: `SigL = np.zeros(ng)`
hardcoded, `Sig2` nnz 0 on all 12. The false clause is duplicated verbatim in
`tests/sn/architecture/_config.py:88-93` — one wrong claim, two homes.

---

## L-075 — read a mutation's verdict by the red set's IDENTITY, not its size: when the reds ARE the naming set, the pins are a mirror

**Context.** 2026-08-26, `refactor/unweld-phase-b` @ `226cc6ca`. Structural
specialization audit of the 1-D SN streaming / angular-redistribution corpus
(report-only; `orpheus/` and `tests/` off-limits; a wide suite in flight).
Deliverable `scratch/specialization_audit.md` (938 lines). The brief handed me
a suspicion to confirm or refute: *"`ReducedStreamingOperator.
requires_upstream_angular_state` and `.angular_marching_axis` are (I measured)
read by ZERO production sites and ~12 test sites."*

### 1. The measurement

Grep first (`grep -rn --include='*.py' -w <field> orpheus tests scripts`,
untruncated): **0 production readers**, **6 test assertions each** — 12 total,
across `tests/geometry/test_reduced_operator.py:318,319,353,354,362,363` and
`tests/sn/primitives/test_snmesh_consumes_reduced.py:88,89,97,98,115,116`.
The brief's estimate confirmed exactly. Remaining hits are 3 constructor
kwargs, 1 field decl, 2 docstrings and 1 comment.

Grep is weak evidence for "no consumer" (it cannot see dynamic or inherited
reads), so I made it RED. In-process, no tracked file touched (**A3**): a
throwaway `-p` plugin on `PYTHONPATH` wrapping all three streaming factories
and FLIPPING both fields on every operator produced.

| leg | outcome |
|---|---|
| CONTROL | `2585 passed, 6 skipped, 8 deselected, 32 xfailed` (293 s) |
| MUTATED | `6 failed, 2579 passed, …` (295 s) |

over `tests/{sn/sweep,geometry,sn/primitives,transport}` at `-m "not slow"`,
bite check **997 mutated operators** (`slab 532 / spherical 232 / cylindrical
233`). The six reds are **exactly the six assertions that name the fields**.

### 2. The lesson, and why #18 does not already cover it

The usual read of "6 gates went red" is *gated*. Here the count is noise; the
**composition** is the finding. `vv-principles` #18 asks *"by what mechanism
does THIS gate see THIS property?"* — and here that question returns a
perfectly good answer ("it reads the field directly"). The pin is not blind.
There is simply **nothing downstream of the value**: the pin asserts that a
producer wrote what a producer wrote.

⟹ the check is a set-diff: **red set vs `grep -rln "<symbol>" tests/`.**
Equality ⟹ no consumer. A red OUTSIDE the naming set is the thing that proves
one exists.

⚠ Two mechanics it rides on, both of which bit me:
- **Patch every rebinding site.** My first attempt patched
  `orpheus.geometry.reduced_operator` + `orpheus.sn.mesh.augmented_mesh` and
  got **3** reds instead of 6 — `orpheus/geometry/__init__.py` re-exports the
  three factories, so the test module that imports from the package got the
  ORIGINAL. A package `__init__` re-export is a rebinding site.
- **Carry a call counter.** Without the 997, a small red set is ambiguous
  between *no consumer* and *no bite* (**A4**, #17's positive control).

**LANDED** as a ⭐ sharpening on `vv-principles` **#17**, with this battery as
its worked example.

### 3. The concept was live, respelled twice — "dead or unwired" was a false dichotomy

The brief asked whether the CONCEPT is dead or merely unwired. Neither.
*"Does this chart need upstream angular state?"* is answered in production by
`upstream_state.angular_upstream is None` (stated by `StreamingTerms`' own
docstring, enforced by `_require_slab` in `linear_discontinuous.py:372,400`)
and by `SNMesh.is_cartesian` (`augmented_mesh.py:521`). The field is a THIRD
spelling. ⟹ when a zero-reader field names a real contract, ask *"how does
production answer this question today?"* before concluding the concept died —
the answer is usually a different spelling, and it changes the recommendation
from "wire it up" to "retire it".

### 4. The companion finding — a THREE-LINK dead chain, and the docstring that hid it

Same corpus, found by the same read-vs-write discipline:

```
AngularRedistribution.mu_start_per_level   <- the OWNER
  -> StreamingTerms.mu_start               (per cell x direction, replicated)
     -> GeometryCoefficients.mu_start      (per ordinate, re-gathered)
        -> nothing
```

`GeometryCoefficients.mu_start` is written at `cache.py:299`/`:356` and read by
**nothing** — so `StreamingTerms.mu_start`'s only production consumer is a
write into a dead array. Its docstring
(`reduced_operator.py:432-435`) says *"Consumed by `MorelMontryAngularSweep`
for the starting-direction seed"*; the closure reads
`angular.mu_start_per_level` (`pole_angular_closure.py:1538,1551`) — the owner.

⭐ **Two mechanics worth carrying:**
- **A field's test-hit count can be entirely WRITES.** All 5 test hits on
  `StreamingTerms.mu_start` are `mu_start=…` inside a **constructor call**. A
  grep-based coverage audit counts them; none can redden if the value is
  wrong. ⟹ split a field's census into READS and WRITES before calling it
  covered.
- **The completeness check for a FIELD is not the symbol grep.** I also had to
  rule out `getattr(geom, …)` (present in `test_cache.py:170,173` — but its two
  name loops omit `mu_start`), `asdict`/`fields`/`astuple`/`replace(geom,…)`
  (none), and the string `"mu_start"` (none). `plan-authoring` §6b's lesson —
  a contract's consumers are spelled without the symbol — applies to dead-field
  audits too, in the mirror direction.

### 5. Refuted attacks of mine (recorded per `process-discipline`)

- *"`dr / start_cosines[level]` has a sign bug"* (`mu_start_per_level` is
  negative) — REFUTED: `march_start_cosines` takes `abs()`
  (`radial_characteristic.py:181,184`).
- *"`mu_x` vs `eta` is a live cylindrical twin"* — REFUTED as a bug:
  `np.array_equal`, `max|Δ| = 0.0` on `folded_product(4,8)` and
  `gauss_legendre(4)`. Kept only as a Pattern-7 convention hazard.
- *"`_require_single_moment_gram` is an unwitnessed guard"* — HALF-REFUTED:
  `test_pole_angular_closure.py:440-486` gives it a positive leg plus three
  hand-built PER-ARM refusals, citing vv #17's granularity trap by name. What
  survives is only that no *production* producer can trip it — its sole
  producer `redistribution_gram` hardcodes `[:, None, None]` and holds no
  scheme handle (Mode-8 SIGNATURE-tautological).
- *"LD × Morel–Montry is a live silent scheme mismatch"* — REFUTED as live:
  `_require_slab` refuses curvilinear LD. Latent only — and the guard is in a
  DIFFERENT package on the OTHER tensor factor, which is why the DD-hardcoded
  seed march (`psi_half_angle_seed.py:180-185`, three `2.0 = 1/w_DD`) will not
  announce itself when #158's curvilinear LD lands.

### 6. The reusable adjudication move: a boundary Protocol's SIZE measures a misplacement

The brief asked whether `orpheus/geometry/reduced_operator.py`'s locally-declared
structural `AngularMeasure` Protocol (6 members, declared so geometry needs no
quadrature import) is a legitimate layer boundary or a workaround for a
misplaced object. **Both**, and the discriminator is mechanical:

⟹ **trace which Protocol members survive moving the suspect object out.**
Here, if the angular factor (`AngularRedistribution`, `angular_redistribution`,
`alpha_dome`, `_assert_alpha_dome_closes`) moves to the angular side, 4 of 6
members go with it and the remaining 2 exist only because `StreamingTerms`
bundles angular data (`mu`, `abs_mu`, `mu_start`, the `/w`) into a *geometry*
packet — and one of those, `mu_start`, is the dead field above. A Protocol that
would shrink from 6 to ~0–2 under a move is not a boundary; it is the **shadow**
of one object being in the wrong package. (The technique is still right — the
module documents a real payoff: the Protocol outlived the `AngularQuadrature`
type it was written against.)

The α-dome verdict itself, for the record: **angular**, decided by `[M]`
`alpha_dome(cosines, weights)` taking no geometry argument at all, and
`angular_redistribution(quad, coord)` needing no mesh — probed bit-identical to
the factories' values, and actually CALLED that way at
`augmented_mesh.py:417` on the d≥2 path where no `reduced` exists. The
chart-dependence is a *selection* (one enum choosing which cosine array to
march), not spatial data; the `1/r` lives in `ΔA`, the spatial Gram, not in α.

## L-076 — a runtime traffic census must count BODIES EXECUTED, not arms dispatched; and "zero applies" is not "no consumer"

**Context.** CS4c step 0 (ORPHEUS campaign-2 opening), 2026-08-30, HEAD
`2f44ed4e`. The chartered deliverable was a per-BINDING-SITE runtime feeding
census over the 13-site / 15-construction operator roster (SN / diffusion /
homogeneous), re-running and sharpening the 2026-08-20 CS4a round-2 census that
produced `vv-principles` #29. Instruments: `scratch/cs4c_step0_spy.py` +
`cs4c_step0_drive.py` (+ two focused probes), read-only, in-process only.

**Method that worked, and is reusable verbatim.**
1. Patch the `functools.singledispatch` **registry** of a `singledispatchmethod`
   — `cls.__dict__["_apply_impl"].dispatcher.register(typ, wrapper)` — one
   wrapper per registered arm INCLUDING the `object` fallback. `register` clears
   the dispatch cache, so it takes effect immediately.
2. `[M]` `cls.__dict__["apply"] is cls.__dict__["_apply_impl"]` → **True** when
   the class does `apply = _apply_impl` in an `else:` branch of
   `if TYPE_CHECKING:`. So A13's "patch EVERY rebinding site" is discharged **by
   identity**, not by two patches — but you must *check* the identity, not assume
   it.
3. Attribute traffic to the BINDING SITE by wrapping `cls.__init__` and walking
   `traceback.extract_stack()`; report `external_frame -> innermost_frame`, where
   external = innermost frame NOT in the operator's own module. Keep every
   instance alive in a list so `id()` cannot be recycled.
4. Capture the operand's **and the return's** `.space` at every call — that turns
   "which carrier" into "which domain and which codomain", which is the question
   a space-binding design actually asks.

**Controls — four tiers, and the third is the one that earns the zeros.**
* *instrument*: counter `0 → 5` on five direct calls.
* *installation*: after wrapping, every registry entry must carry the marker;
  raise at install time otherwise (12 arms, 0 unwrapped).
* ⭐ *per-ARM ACTIVATION*: fire **every** registered arm and every plain-method
  verb directly, so a `NOT-RUN(prod)` verdict is a fact about production rather
  than about a dead wrapper. Without this, 8 of 23 verbs would have read zero
  with no way to tell blindness from absence.
* *non-perturbation*: all 11 headline numbers bit-identical with and without the
  spy (`--control` flag on the same driver, same fixtures).

**The finding that is NOT in #29 (a) (b) (c) — the fourth way, (d) NO arm.**
`MultiplicationOperator` at `sn/coupled_system.py:446` (SN's C binding) is minted
**22 / 22 / 24 / 25 / 20** times per k-solve — once per outer, because
`build_within_group_system` is re-called per outer — and **every instance is
silent in all 9 SN scenarios, under BOTH `source_iteration` and `krylov`.**
Mechanism: `StreamingCollisionOperator` (`sn/operators/streaming.py:504`)
subclasses `OperatorSum` holding C as `b`, and **overrides** `apply` (`:723`) to
call `self.loss_representation.loss_action(self.sigma, psi)` where
`sigma = self.diagonal.coefficient.values` (`:712-719`); `apply_transpose` is
overridden identically (`:744`). So the parent reads C's **data** and never its
**body**.
Two consequences that point OPPOSITE ways, and a census owes both:
* *design*: there is no action body to select at construction — a
  bound-space-keyed collapse buys nothing at that binding, and the real question
  becomes whether the operand is an operator or a `CrossSectionField` in an
  operator's clothing.
* ⛔ *retirement*: **zero applies ≠ zero consumers.** The object is load-bearing
  through an attribute read one frame up. Inferring "dead" from the traffic is
  exactly backwards — the polarity-flipped twin of A13 (there, pins that only
  NAME a symbol prove no consumer; here, no apply traffic coexists with a live
  one).

**The second novel mechanic — an ARM can be a RE-DISPATCHER.**
`ScatteringOperator`'s `FullField` arm runs
`self.apply(cast("AngularFlux | HarmonicMomentFlux", psi.interior))`
(`scattering.py:1189`), so every composite apply produces TWO counted arm
entries with **exactly equal counts on every scenario**
(`FullField ×N` and `AngularFlux|HarmonicMomentFlux ×N`, 6/6 scenarios).
⟹ "4 arms" over-counts the bodies and under-counts the branching: selecting the
`FullField` body at construction *relocates* the runtime branch one frame in
rather than removing it. Tells: equal counts across every row (census side); a
`self.apply(...)` inside a registered arm (source side).

**Two predictions of my own, refuted by the run** (log them — the reason is what
stops the next attacker re-deriving them):
* *"the windowed `HarmonicMomentFlux` carrier is routed by
  `inner_schedule='gauss_seidel'`"* — **refuted** by a 12-row probe
  (6 configs × 2 schedules, every row's counter > 0): the discriminator is
  **spatial dimensionality**. 8/12 rows `AngularFlux` (slab/sphere/cylinder,
  P0 and P1, both schedules), 4/12 `HarmonicMomentFlux` (2-D Cartesian LS4, both
  schedules). The schedule changes only the iteration count.
  ⟹ the 2026-08-20 census's SN-S row (`TimedFullField ×225 AND AngularFlux ×225`)
  is **1-D-only**; its denominator was a held-fixed axis nobody wrote.
* *"SN C is silent only under source iteration; Krylov re-enters
  `OperatorSum.apply`"* — **refuted**: 22 instances / 22 silent under Krylov too.
  The override lives on the composite, not on the solve strategy.

**The scope caveat, made TIGHT instead of ritual.** "A run measures its workload
only" is usually an unfalsifiable disclaimer. Here it was bounded by an AST
census of every `Name` node in the 7 roster classes over `orpheus/**/*.py`:
**exactly 6 files** reference them (`diffusion/solver.py`,
`homogeneous/solver.py`, `sn/coupled_system.py`, `sn/solver.py`,
`transport/operators/scattering.py`, `sn/operators/streaming.py` — the last
holding no construction), and **`cp/`, `moc/`, `mc/` reference 0 each**, at both
the operator tier and the underlying array verb (`apply_p0_in_scatter` has
consumers in 3 files, all under `transport/`). So the residual risk is not
"another solver family" but "another configuration of a driven family".
⟹ **pair every runtime census with a static reference census** — the static one
supplies the denominator the runtime one cannot.

**Free by-catch (the census's own docstring).** `scattering.py:735-737`, on the
`isotropic_kernel` property whose body IS the space-anonymous mint, claims *"The
same energy operation is shared by every transport model (CP / MoC / diffusion /
homogeneous / MC)"*. `[M]` 2 of the 5 named models consume it. The neighbouring
clauses in the same docstring are both accurate and measured-true, and that is
the aggravator — accuracy on either side of a false clause removes the reader's
signal (vv #21's self-contradicting-file shape, at clause granularity).

**Landed:** `vv-principles` #29 gains the (d) NO-arm sharpening + the
BODIES-not-ARMS discipline, with this measurement as its worked example.
Deliverable: `scratch/cs4c_feeding_census.md`.

---

## L-077 — the #429 symmetry/quotient carve: a brute-force + mutation review

**Date** 2026-09-02. **Tree** branch `fix/angular-phantom-support`, HEAD
`c1fca8bd`, tracked tree clean. **Subject** `orpheus/numerics/symmetry.py`
(2455 L) + `orpheus/numerics/manifold.py` (1958 L) after `c1d53206`
(`SubgroupOfO3.O2(axis)`, the naming law) and `a7c8de6d` (the invariance
question moved onto the orbit space). Read-only: no tracked file edited;
`diff -q` against pristine copies + gate-green-again (670/670, 48.5 s) both
proved the revert.

### What the brute force CONFIRMED (so the review's negatives are trustworthy)

The core group theory is *right*, and measured against references built in
plain numpy, never through the module:

* `is_normalised_by` — **5103/5103** agree (27 tags × 189 motions = 168
  distinct group elements + 12 random O(3) + 9 partial rotations). The
  finite subset (3402/3402) is an EXACT conjugation reference.
* `normalises` — **729/729** ordered pairs (27²), self-normalisation 27/27,
  normaliser closed under product on every probed pair.
* `contains` — **575/576**; the one disagreement was MY probe (`Q *
  sign(diag(R))` from a QR is orthogonal, **77 of 200 improper**, so my
  "SO(3) sample" contained reflections). Instrument failure, not a finding —
  `plan-authoring` §4's VERIFY sharpening, live.
* `orbit_stabiliser` — **24/24** against a genuine maximum search. The search
  is finite by `g ∈ K ⟹ h⁻¹g ∈ Stab(p₀)`, so `K ⊆ H·O(2)_{p₀}`; sampled
  360 stabiliser elements × 17 test points. **σ_y's stabiliser really is σ_y
  alone** (0 extra), same for every C_n, D_nh, O_h, I_h.
* vv#15 compatibility law — **1260/1260** (edge × fixture) over 9 fixtures
  including two hand-built folds.
* the axial `lift` (barycentre) — equivariant on **183/183** normaliser
  pairs, worst 6.66e-16; `π∘lift = id` at **0.000e+00** on all three axes.
* the hemisphere section at the equator — `σ_y(section) = section` at
  **0.000e+00** on the ρ=1 rows: an equator node IS its own mate.
* tolerance margins — node-match and weight-match residual **0.000e+00** on
  all 15 certified (rule × mirror) cells; realized-element orthogonality
  1.33e-15 against a 1e-9 band (7.5e5× slack). No tolerance is load-bearing.

### The mutation table (in-process plugin, `pytest_configure`, caches cleared)

Scope: 6 files, 670 tests, ~50 s/arm. Positive control **111 reds**.

| arm | mutation | calls | reds |
|---|---|---:|---:|
| PC | every finite group collapses to {e} | 45 | **111** |
| M1 | `orbit_stabiliser` → `self` | 100 | 6 |
| M2 | axial normaliser `±ê_a` → `+ê_a` | 361 | 63 |
| M3 | `_identity_component_normalises` → True | 1085 | **1** |
| M3a/b/c/d/e | its five arms, separately | 723/90/**1**/197/675 | 7/27/**1**/**1**/**1** |
| M4 | `_fixes_every_point` → True (ERR-072) | 376 | 25 |
| M5 | drop kernel step 1, swallow the raise | 2290 | **0** |
| M5b | drop kernel step 1, raise escapes | 2025 | 16 |
| M6 | drop kernel step 2 (`H ⊇ G` short-circuit) | 2290 | **0** |
| M7 | ERR-073 relation-not-bijection | 6235 | 57 |
| M8 | hemisphere section → LOWER half | 200 | 3 |
| M9 | barycentre axis → 0 (ERR-080 forgery) | 495 | 27 |
| M10 | `_assert_named_by_stabiliser` off | 28 | 5 |

### The findings

1. **`identity_component` is FALSE on 12 of 22 members and has ZERO
   consumers.** A finite subgroup of O(3) is discrete, so its identity
   component is `{e}`; the property returns `self`. Worse, the docstring's
   own operative property — *"its orbits are connected, so it fixes every
   point of any finite invariant set"* — is violated by the returned value
   on **11 of 22**. `[M]` `grep "\.identity_component"` over `orpheus/`,
   `tests/`, `docs/theory/` = **0 hits**. A13's shape with the polarity of a
   *wrong* answer rather than a dead field.
2. **`is_invariant`'s O_h and I_h docstring bullets describe a RETIRED
   implementation.** Proven mechanically: unparse every function on the
   call chain and grep — `sorted`/`fingerprint`/`multiset`/`radii`/
   `icosahedron`/`vertex`/`representative orbit`/`np.abs` all **absent**.
   The I_h bullet advertises *"a 12-element representative orbit"* — the
   exact ERR-072 sampling defect the same docstring elsewhere says the
   module abolished.
3. **The "brute-conjugation CONTROL" is the production expression,
   α-renamed.** `ast.unparse` both, α-normalise the bound variables:
   **character-identical**, and `_closed(H)` returns `_group_elements` — the
   same list. The redeeming content is the *neighbouring* hand-derived
   `assert brute is (axis == mirror)`; the docstring credits the tautology.
4. **Registry stage 0 re-uses Γ for a second job with the opposite soundness
   direction.** `discrete_residual` is documented as a CLOSURE requirement on
   the RULE (*"the symmetry a reflecting x or y face needs"*); stage 0
   `Γ ⊇ spent_group(...)` reads it as a LICENCE TO FOLD. `[M]` the σ_y fold
   is admitted at both stages for `cartesian2d`, whose own docstring says
   *"2-D Cartesian (x-y) … never a symmetry of a z-uniform problem"* — the
   fold discards every `μ_y < 0` ordinate, emptying **2 of 4** sweep
   quadrants. Sound for the cylinder (σ_y IS the azimuthal reflection there),
   unsound for cartesian2d, and the two rows are byte-identical.
5. **Both kernel short-circuits are unwitnessed** (M5 0 reds, M6 0 reds).
   Step 1's refusal is duplicated one frame down in `Quotient.induced_action`
   (M5b: 16 reds ⟹ its real job is converting a raise into a `False`); step 2
   is a provable theorem-shaped optimisation (`orbit_coordinates` is
   H-invariant, so the fall-through returns the identity permutation
   bit-exactly).
6. **Three arms of `_identity_component_normalises` have ONE catcher each,
   and it is the SAME test.** Arm (c) is invoked **once** in 670 tests.
7. **`candidate_groups` branches on node STORAGE WIDTH** (`shape[1] < 3`)
   after the carve made width non-load-bearing (`ambient_representatives`
   accepts both). `[M]` ONE fold, two spellings: `maximal_invariance_groups`
   reports **`{D_2h}`** at ambient width and **`{σ_x, σ_y, σ_z}`** at chart
   width. `is_invariant` itself is identical on 7 of 7 groups, so the
   divergence is entirely in the candidate filter.
8. `Cn(1)` and `Trivial` are one group, two spellings, two behaviours:
   `SPHERE.quotient(Trivial)` builds, `SPHERE.quotient(Cn(1))` raises
   *"no catalogue entry for S^2/C_1"* — `_assert_named_by_stabiliser` passes
   it because `C_1.orbit_stabiliser == C_1`.

Deliverable: `scratch/_rev_qa_*.py` (9 probes) + `scratch/_rev_qa_arm_*.log`.

---

## L-078 — a reproduction that agrees to 9 decimals can still "disagree": the discrepancy was a UNIT, and the unit inverts the study's own conclusion

**Task** (2026-09-03, branch `fix/n2n-anisotropy`, HEAD `8707c53a`). Independently
reproduce a `#426` claim: restoring the ℓ=1 (n,2n) emission moment on a
Be-reflected U-235 slab moves k by "−413.55 pcm". Brief mandated structural
independence — write my own probe before reading the originating one.

**Outcome: REPRODUCED.** All three k values match to every published digit
(`A0 1.095322188`, `A1 1.091186690`, `A2 1.091199657`); C0 (my tape pipeline's
ℓ=0 product vs the shipped `Mixture.Sig2`) = **exactly 0.0** on both isotopes.

### The finding — three numbers, all called "pcm", and the ordering INVERTS

My Δk read **−377.56**, the claim **−413.55**. Neither is wrong:

```
Δk × 1e5     = -413.55   ABSOLUTE      <- the originating probe's `1e5*(k-k0)`
Δk/k0 × 1e5  = -377.56   RELATIVE      <- mine
Δρ × 1e5     = -346.01   REACTIVITY    <- (1/k0 - 1/k1)·1e5
```

They differ by exactly `k0 = 1.0953`. So far a nit. The bite is that the study
had **two fixtures at different k**, and the conventions do not preserve the
ORDERING between them:

| fixture | k0 | Δk·1e5 | Δk/k0·1e5 | Δρ·1e5 |
|---|---|---|---|---|
| fast_thin | 1.0953 | **−413.55** | −377.56 | −346.01 |
| fast_thick | 1.5262 | **−529.26** | −346.78 | **−228.00** |

*"A thicker Be reflector makes the P0 truncation worse"* is **TRUE** in the
absolute convention (529 > 414) and **FALSE** in reactivity (228 < 346) — a
**2.3×** spread at k = 1.53. The number was headed for a GitHub issue where the
natural physics reading is reactivity.

⟹ **A derived comparison quantity must carry its DEFINITION, not just its unit
name, whenever the unit name is overloaded** — and the tell that it matters is a
fixture set spanning a range of the normalising quantity. `pcm` = 10⁻⁵ and says
nothing about what was divided by what.

### What made the reproduction worth its cost (both probes used production code)

Probe-vs-probe agreement is blind to a SHARED convention, so I tested the two
shared premises **against physics** instead of against the other probe
(`scratch/_426_shared_premise_check.py`):

* **index convention** — (n,2n) can only lose energy, so the canonical
  fast-first matrix must be strictly upper-triangular. `[M]` **8195/8195** (Be)
  and **6067/6067** (U) nonzeros with `g_to ≥ g_from`; lower-triangle mass
  **0.000e+00**. A row/col swap would have put all of it below the diagonal.
* **Legendre normalisation** — for one `g→g'` transfer `Σ_ℓ/Σ_0 = ⟨P_ℓ(μ)⟩ ∈
  [−1,1]` is a HARD bound. `[M]` entrywise max **0.9603 / 0.8977** at ℓ=1,
  **0 entries > 1** at every ℓ ≤ 6. **A stray `(2ℓ+1)=3` would have shown as
  ratios up to ≈2.9.** Corroborated structurally: `gendf.py:518/545` writes raw
  MF=6 into `sigS[ℓ]` with no factor; the `(2ℓ+1)` is on
  `LegendreBasis.reconstruct` (`legendre_basis.py:228-232`).

Both closed. The residual shared premise, stated rather than closed: that
`_strip_transfer_yield`'s per-row scale is the right strip for ℓ≥1.

### Controls that earned their keep

* **no-op plumbing** (mine, not briefed): inject `0.0 × ℓ=1` → k **bit-identical**
  to A0. Proves the `replace(mix, SigS=…)` rebuild is inert, so Δk is the values.
* **linearity**: `Δk(+1)/Δk(+0.5) = 2.0031` ⟹ a factor-*n* convention slip shows
  as a factor *n* in Δk. This is what converts "the sign flipped" into a bound.
* **material attribution**: Be-only −377.30, U-only −0.26, **sum = A1 exactly**.
  99.93 % is the reflector — and the tape says why: μ̄ = rowsum(Σ₁)/rowsum(Σ₀) is
  **+0.303 mean** for Be (positive on 50/50 live groups) vs **+0.023** for U.
  Mechanism and magnitude agree, which no single k comparison could show.

### ⭐ The stretch leg FAILED to calibrate, and diagnosing whose failure it was is the lesson

Second route: the extended-transport ("outflow") correction reaches the same
physics through `SigT` + `SigS[0]` only, so it does NOT share the `(2ℓ+1)`
reconstruction convention. `[M]` −639.10 (unclamped) / −622.23 (clamped) vs the
direct −413.55: same sign, ratio 1.50–1.55.

I then tried to CALIBRATE that 1.5× by running the same correction on the
**elastic** ℓ=1 channel, where the P1 answer is what the tree ships. `[M]`
`ΔTR/ΔP1 = 0.6042`. Tempting conclusion: −639/0.60 = −1058, so the direct route
is 2.6× too small. **Wrong** — `[M]` **327 of 421** Be groups get a negative
corrected P0 diagonal in the elastic leg against **6** in the (n,2n) leg. The
two applications are in different regimes and the ratio is not transferable;
dividing by it is exactly the do-these-legs-share-a-population error.
⟹ honest verdict: corroborates SIGN decisively and order of magnitude within a
factor the approximation is itself *measured* to span (~1.7×); **cannot
adjudicate a factor of 2**. Corroboration, not verification.

I did NOT run the adjoint perturbation leg: the moment/`/W` convention risk was
high enough that a wrong reproduction of mine would have impeached a correct
result (the §4 VERIFY trap).

### Findings in the originating probe (`scratch/_426_be_reflected_probe.py`)

* **D1 ⛔** `dk_pcm = 1e5*(k-k0)` labelled "pcm" (`:244`, `:247`, `:261`, md
  header `:257`) — the inversion above.
* **D2 ⚠** C0, the LICENSING control, is a bare `assert` (`:228`), as are the NL
  (`:92`), `lmax ≤ L_solve` (`:134`) and padding (`:237`) checks. A script, not a
  collected module, so pytest's rewriter does not cover it — Mode 8's measured
  domain exactly. Promotion into `tests/` silently voids its own licence.
* **D3 ⚠** the `.md` deliverable is written only after ALL fixtures (`:264`) —
  and **it never was**: the run died inside `thermal`, the JSON holds 2 of 3
  fixtures, and nothing announces the truncation but a missing `DONE`.
* **D4 ⚠** `warnings.simplefilter("ignore", RuntimeWarning)` wraps the whole
  `compute_macro_xs` (`:118-120`) — would swallow a genuine σ₀-convergence
  warning on the very fixture (`thermal`) most likely to raise one.
* **D5 ⚠** reporting scope: A4/A6 raise `L_solve` while ELASTIC stays P2 (ingest
  keeps `range(3)`), and elastic anisotropy is ~15× the (n,2n) effect on this
  fixture — so "A6 = the converged anisotropy answer" is a mis-read waiting.

### ⭐ Two attacks WITHDRAWN at Phase 2 — recorded so they are not re-attacked

* *"`A0_shipped_L2` is not a clean baseline, it runs through
  `with_n2n_moments(lmax=0)`"* — **wrong, it is the BETTER design**: baseline and
  arms share the rebuild, so the delta is purely the injected values. My separate
  no-op control and my raw-mixture A0 both confirm it.
* *"A2→A4 conflates more moments with higher solver order"* — **closed by the
  probe's own `C_pad_L6_shipped`**: padding to L=6 with zero moments reproduces
  A0 bit-identically on both fixtures, so raising `L_solve` is provably inert.

**Artifacts**: `scratch/_426_repro_probe.py` (stages `tape|c0|solve|controls|pt|
pt2|cal|cal2`), `scratch/_426_shared_premise_check.py`, `scratch/_426_repro.md`.
No tracked project file edited; ≈240 s of compute.

---

## L-079 — the #428 four-solver (n,2n) census: a family HANDLED with zero witnesses, an ERR whose only catcher the canonical gate deselects, and a census whose control validated the wrong stage

**Date** 2026-09-03. **Brief:** establish, `[M]` with `file:line`, how each of
six solver families treats (n,2n) at HEAD, so an archivist can split
`docs/theory/foundations/cross_section_data.rst` §"Reactions Not Included".
READ-ONLY (no tracked file edited; `git status --porcelain -- orpheus/ tests/
docs/` empty throughout). Memo: `scratch/_428_four_solver_check.md`.

### The verdict, in one line

**All six families HANDLE MT=16** — removal counted once in `Σ_t`/`Σ_a`,
emission `2Σ₂ᵀ` from the one home `N2NKernel.multiplicity`, and the channel in
the k balance. The doc's *"every transport solver assumes a 1-in-1-out
scattering model"* is present-tense-false for every one of them. **ERR-023 (MC
ignores Sig2) is FIXED**, and MC is unbiased (`1.63 σ`).

### The method that made it decisive, and it is reusable verbatim

A *reading* census answers "does the code touch `Sig2`" — the wrong question,
because a family can touch it and be untested, or be right and read as wrong.
The instrument that answered the real question was **one mutation at the datum's
single home**, applied by an in-process throwaway pytest plugin:

```python
def pytest_configure(config):
    from orpheus.transport.kernels import N2NKernel
    N2NKernel.multiplicity = 1          # or 0 for the stronger arm
    import orpheus.mc.solver as mc
    mc._N2N_MULTIPLICITY = 1.0          # ⚠ a module-level float()'d COPY
```

Three mechanics generalise:
1. **The mutation home is one attribute, so every family sees it** — which is
   only true because CS4c step 3 single-sourced it. A datum with one home is a
   free cross-family battery.
2. ⚠ **A `float(CONST)` taken at import is a SECOND home for the mutation** —
   `mc/solver.py:36` copies the ClassVar at import, so patching the class alone
   leaves MC unmutated and reports MC as blind. Grep the constant's name for
   module-scope assignments before believing any zero.
3. **The reds are read as a TABLE, per tree**, not as a count — and the
   composition is the finding (A13).

### Finding 1 — diffusion HANDLES the channel and has ZERO witnesses

`[M]` `tests/diffusion` = **113 passed / 0 red** under ν₂ₙ: 2→1 **and** under
the stronger 2→0, while `tests/homogeneous` reddens 7 in the same process (the
positive control). Cause, measured by instrumenting `Mixture.__post_init__` for
the run: **625 mixtures constructed, 1 with nonzero `Sig2`** — and that one is
`homo_2eg_n2n`, built as a side effect of the derivations registry, never handed
to a diffusion solve. `_fixture_materials()`
(`tests/diffusion/test_operators.py:103`) calls `make_mixture(...)` with no
`sig_2=`.

⭐ The shape worth carrying: `IsotropicN2N` appears in **4** diffusion tests, so
a grep-based coverage read says "covered". Every one of them applies it to a
**zero kernel** — carrier arms, assembly refusal, composite-vs-bare matching —
i.e. Mode-10 *exercised-but-unconstrained* at FAMILY scale. ⟹ **for a coverage
question about a DATUM, census the fixture's VALUE, not the operator's
appearances**; a `Mixture.__post_init__` (or any producer `__post_init__`) spy
answers it in ten lines and cannot be fooled by a name.

Second-order tell found in the same file: the test-side `_loss()`
(`:169`) assembles `A = L + C − S − B` **without** the N2N arm while production
assembles `L + C − (S + N2N) − B`. Bit-identical on `Sig2 = 0` — a twin the day
a Σ₂≠0 fixture lands.

### Finding 2 — an ERR whose only catcher the canonical gate deselects

ERR-023's sole catcher is `tests/mc/test_gaps.py:718`, carrying
`@pytest.mark.slow` **and** `@pytest.mark.catches("ERR-023")`. `[M]` at
`-m "not slow"` the MC tree is 39 passed / 0 red under the mutation; run alone
the same test FAILS in 84 s. So the test has real teeth and the gate that
decides "green" never runs it.

⭐ This is a NEW class beside Mode 8's nine. All nine describe a gate that
cannot **fail**; this is a gate that cannot **run**, and it is invisible to
every existing check — `nexus errors` counts the catcher, a mutation run scoped
to that file reddens it, and only the marker set says otherwise. → digest **E7**.

### Finding 3 — SN's ν₂ₙ is pinned at the operator tier, not end-to-end

`[M]` `tests/sn/operators` = 8 red; `tests/sn/verification/analytical` = 57
passed / **0** red; `tests/sn/eigenvalue` = 67 passed / **0** red. Two unrelated
causes:
* `test_kinf_homogeneous` parametrizes `{"1eg","2eg","4eg"}` — the shipped
  derivation registry HAS a Σ₂≠0 member (`homo_2eg_n2n`) and the SN ladder skips
  it (#13's finite-roster corollary: probe every member of a shipped roster).
* `test_reflective_n2n_convention` (`catches("ERR-065")`) compares the reported
  `keff` against `_map_ratio_kstar(solver, phi)` — a ratio built from the
  **solver's own operators**, so both sides move together under a multiplicity
  mutation. ⭐ That is **correct for what it claims** (it gates *estimator ≡
  posed problem*, ERR-065's actual defect class) and structurally blind to a
  wrong ν₂ₙ VALUE. ⟹ two different claims can wear one `catches` marker; when a
  mutation leaves an ERR's catcher green, ask whether the marker names the
  defect class the gate sees, before calling the gate blind.

### Finding 4 — the census whose control validated the wrong stage → digest A17

`tests/transport/test_n2n_multiplicity_census.py` claims *"a thirteenth literal
home is unspellable without reddening this census"*. Its filter is two-stage:
a **name-net** over the function body, then a **literal pattern**. `[M]` the net
`("n2n","sig2","sig_2n","_2n")` misses `sig_2` — `derivations/`'s spelling — so
widening it by that one token yields 2 hits
(`derivations/common/eigenvalue.py:61, :290`). Those two literals are correct by
design (the reference tree must not read the SUT's constant — and that
independence is precisely why the CP and homogeneous mutations reddened), but
they are excluded by a **filter gap**, not a named exclusion, so a *production*
literal spelled `sig_2` escapes identically. Its positive control (`:91`) is a
synthetic source whose function is named `n2n_source_assembly` — every arm
clears stage 1 for free.

### Refuted candidates (recorded so nobody re-attacks them)

* *"the SN adjoint drops (n,2n)"* — plausible from `solve_sn_adjoint`'s own
  docstring (`sn/solver.py:2983` writes `A_loss = L+C-S-B`). **Refuted**:
  `_adjoint_posing_parts` calls `build_within_group_system` without `n2n_op`,
  which **defaults to minting one**, and folds `gain = S + N2N + B_a`. `[M]`
  k_adj matches `k_inf(Σ₂≠0)` to 1.7e-14 on **both** arms (slab =
  non-carrying, sphere/cyl = carrying / System B). The docstring is the defect.
* *"MC is biased on (n,2n)"* — **refuted**: `1.655710 ± 0.001525` vs
  `1.653226` = **1.63 σ**. The `w *= 2` + one-exit-group-from-the-row treatment
  is expectation-preserving because the row's normalisation IS the emission
  spectrum.
* *"`xs_library` mixtures might carry Σ₂"* — `[M]` **0 of 12** do, so the doc's
  sentence about the synthetic library is TRUE (the factory does take `sig_2=`).

### Two probe bugs of MY OWN, both caught by the §4 VERIFY move

* I printed `k.emission_matrix` (a METHOD, not a property) and read `False` for
  `== 2Σ₂ᵀ`. It is `True`. ⟹ a bound-method repr in a comparison is the tell.
* I transcribed the CP `jacobi` row as identical to `gauss_seidel` because a
  `tail` had clipped it. Re-measured: `1.6532258064618464` vs
  `1.6532258064612666` — different. ⟹ never transcribe a row a viewport clipped.

### Denominator facts worth reusing

`[M]` over the 13 shipped `.GXS` tapes: **MF=6 MT=16 in 11**, **MT=17 in 6**,
**MT=37 in 2** — so 17/37 are on the tapes and genuinely unread. The library
carries **no Pu**, and MT=17 rides four **zirconium** isotopes, so "heavy
isotopes (U-235, U-238, Pu-239, …)" is not an accurate scoping sentence.

### ⚠ TWO skill items OWED and NOT landed (the brief forbade tracked-file edits)

Both belong in `vv-principles` §Anti-patterns; drop-in text is digest **E7** and
**A17** above. Neither rationale is currently in the skill (`[M]` grepped).

---

## L-080 — #448: the SN eigenvalue finalize as ONE `fixed_point_step`

**2026-09-06, branch `fix/sn-eigenvalue-finalize-448`, UNCOMMITTED working-tree
review. READ-ONLY on `orpheus/` + `tests/` (proof: `git status --porcelain`
md5 `d4c6fb10…` identical at mid- and end-session).**

### The change
The finalize hand-built `Fφ/k + S₀ᵀφ + 2Σ₂ₙ,₀ᵀφ`, lifted it isotropically, and
swept — so at every `scattering_order ≥ 1` the ℓ≥1 half of BOTH channels was
missing from the returned ψ. Replaced by
`fixed_point_step(driven.implicit.inverse(), driven.gains, _eigenvalue_driver_source(...), ψ_conv)`
— one application of `G(ψ) = M⁻¹(q_F + Σ Nᵢψ)` through the splitting the LAST
inner solve DROVE, recorded as a new `DrivenSplitting(system, implicit, gains)`
NamedTuple beside `_psi_typed`.

### What I measured (probes `scratch/_448_qa/probe{1,2,3,4}*.py`)

**All four finalize arms are structurally right** (probe1, 20 solves):

| arm | `driven.implicit` | `driven.gains` | iterate bulk | returned bulk | G1 |
|---|---|---|---|---|---|
| 1-D SI | `StreamingCollisionOperator` | S, N2N, `SNBoundaryOperator` | AngularFlux | AngularFlux | 1.4e-11…6.2e-11 |
| 2-D windowed SI | `StreamingCollisionOperator` | S, N2N (**moment-bound**: domain interior == `spherical_harmonic_space ⊗ …`, probe4), B angular | **HarmonicMomentFlux** | **AngularFlux** | 2.0e-11 / 6.6e-11 |
| coupled sph/cyl | `CoupledOperator` | ONE coupled gain grid | AngularFlux | AngularFlux | 5.2e-11…6.5e-11 |
| Krylov | `system.implicit_operator` | `system.explicit_gains` | AngularFlux | AngularFlux | 1.0e-11…5.5e-11 |
| **G-S (opt-in)** | **`ScheduledInvertibleOperator`** | S, N2N, **`SNMaskedBoundaryOperator`** | HarmonicMomentFlux | AngularFlux | 1.9e-11 / 6.5e-11 |

Coupled ψ½ (declared blind): returned vs converged ray `2.5e-11` interior /
`1.6e-12` boundary, `min ψ½ > 0` — the "fold FISSION only into q½" design is
measured correct on both coupled arms, both orders.

Returned trace (declared blind): `inflow == B·outflow` to `≤ 2.0e-11`
(exactly `0.0` on vacuum), and `== converged trace` to `≤ 1.8e-11` on all 5 arms.

### The mutation battery (in-process plugin `scratch/_448_qa/_mut448.py`)
Finalize-SCOPED by construction: it rewrites `self._driven` AFTER the inner
solve returns, so the SI driver keeps its own real gains and only the
reconstruction is mutated — the `vv#18` over-power problem solved structurally
rather than by a phase flag.

* `dropB` (drop the boundary gain from `_driven`): **8 of 14** G1 rows red
  (`slab_refl`, `sphere_refl`, `cylinder_vac`, `cart2d` × both orders); the
  3 vacuum-slab arms + `slab_krylov` correctly stay green. Positive control OK.
* Census under `dropB`: **161 inner solves, 0 `ScheduledInvertibleOperator`
  splittings** ⟹ the gate module never reaches the G-S arm.
* `wrongM_gs` (record the UN-split `L+C` while keeping the re-split gains —
  `M − N ≠ A`, the ERR-056 incoherent-splitting shape, and exactly what the
  Krylov arm records): **15/15 G-S splittings mutated**, red set =
  `test_the_repair_is_AUDIBLE_and_names_the_root_fix` +
  `test_it_is_SILENT_when_the_answer_was_already_canonical` — the GAUGE pair,
  on a fully-reflective singular box. Real detection, wrong subject
  (`vv#18`: by what mechanism does THIS gate see THIS property? — it sees the
  trace's kernel content moving, not the reconstruction).

### The band's stated MECHANISM was wrong (the novel lesson → skill #13)
The memo's §1.2 table read *"four decades of `flux_tol` move it not at all;
the empirical driver is `inner_tol` alone"* — `[M]` `1e-6/1e-7/1e-8/1e-9` all
give `n_outer = 10`. Extending the sweep to `flux_tol = 1e-11` takes
`n_outer → 12` and the polish falls `3.43e-11 → 6.96e-13` (**49×**); same for
`keff_tol 1e-10 → 1e-12`. So at the gate's own `inner_tol = 1e-11` the OUTER
term dominates. A tolerance is a **discrete** knob — it acts only through the
iteration count — so a sweep whose count never moves has not tested it.
Landed in `vv-principles` #13 as the fourth disguise.

### Findings the tree still owes
1. **G-S finalize arm has no self-consistency witness** (measured above).
2. **The trace-provenance gate the module DECLARES it needs** was not written
   (option B was taken).
3. **ERR-083 does not exist** — 4 gates carry `catches("ERR-083")`; `[M]`
   `merge.py:641-660` `adopted=True` on ORPHEUS ⟹ a per-marker
   `logger.warning`, so `sphinx -W` fails. (My first hypothesis — that the
   `not adopted` guard silences it — was REFUTED by reading the code before
   publishing; the skill's sentence is right.)
4. **34 doc sites across 12 pages** still name the 6 retired/moved symbols,
   including a DECLARED `.. implements:: :by:` edge at
   `operator_algebra.rst:4043` pointing at `TransferOperator.build_aniso_source`.
   2 of the 34 (`error_catalog.rst:59,:1058`) are past-tense ERR-002 history
   and STAY.
5. **13 duplicate/mangled import lines in 5 test files** from the automated
   rewrite (`test_keff_curvilinear.py:527-533` has three imports of one name,
   one with column-0 continuation lines inside a function body — parses, ugly).
6. `fixed_point_step` / `lagged_source` are in `iteration.__all__` with **no
   direct test** — only exercised through `solve_sn`'s finalize and
   `SourceIteration.solve`.
7. The retired `build_aniso_source` guard is SUPERSEDED, not lost:
   `admit_composite` (`lift.py:182-202`) checks interior space **and** trace
   space **and** carrier class — strictly stronger. Production reaches
   `_redistribute_ordinates` only through `apply`. No refutation.
