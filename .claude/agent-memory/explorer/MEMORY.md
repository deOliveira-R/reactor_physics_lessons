# Explorer memory index

One line per entry. Behavioral lessons live in `lessons.md` (read FIRST each
dispatch); durable subsystem SHAPE lives in `AGENT.md`. This index holds only
(2) git-true active-state and (3) durable convention/units reference. Per-campaign
`file:line` carve maps are archaeology — they stale in days and are re-derivable
in seconds via Nexus. **Never restate a lesson's content here** — `lessons.md`
carries its own headings, so a summary line is a second copy that drifts
(2026-08-11: that block had grown to 47 lines and pushed this file to 165).

## 1. Lessons (read first)

- [lessons.md](lessons.md) — the exploration lessons, L-001…L-038, each headed by
  its own one-line rule (L-039 2026-09-05: Nexus `callers` on METHODS ⟹ read the `unresolved` count; L-040 2026-09-07: a retirement's PROSE census = xrefs / PROMISSORY claims / CONVENTION statements — sort before counting). The spine (blast-radius = graph+grep+constructors+doc-nodes;
  verify-premise-first; durable-shape vs line-map; git-is-authoritative-for-merge-status)
  is PROMOTED to AGENT.md Operating Principles 4–7; L1/L2/L3/L5 remain there as the
  forensic war-stories behind them. Skim the headings, read the one that matches your
  question shape.

## 2. Active / in-flight state

Merge-status in memory goes STALE (L5). ALWAYS reconcile any "resume X" against
`git merge-base --is-ancestor <hash> HEAD` before acting; never trust a frozen
"NOT pushed". Landed milestones live in the SN theory page's development-history
changelog, not here; the open backlog is GitHub issues.

- **Every SN campaign this agent has audited is MERGED to main** (git-verified
  through 2026-07-21): operator-algebra unification, Wave-O / role-typing /
  g-adjoint, the matvec carve onto `_OneDimScanWalk`, LD-on-the-DAG, the
  foundation cleanup, the field-typed algebra map, #236, #280 walk-unification
  (`b23e972e`), #34 ray-leg retirement, task #54 `sn/spatial`→`sn/sweep`
  (`588f2429`). Only surviving local SN branch: `feature/sn-adjoint-transport`
  (the paused #276 campaign).
- Durable post-#280 facts: A_BB = `RadialCharacteristicOperator`
  (`orpheus/sn/operators/radial_characteristic.py`) WRAPS the ψ½ march, and its
  `.solve` is the sole production caller of `carlson_inward_sweep_from_source`;
  the walk executors (`_OneDimScanWalk`/`_loop_walk`/`_dag_legs`) live in
  `sn/loss_representation/__init__.py`.

## 3. Durable reference (survives code churn)

Convention/units facts a line-number drift cannot invalidate — they pin WHY a
quantity carries the units it does.

- [CP equation→code truth (nexus#82)](nexus82_cp_implementers.md) — the CP construction has exactly
  TWO implementations (production `CPMesh` 4-step pipeline; derivations `build_cp_matrix` single body)
  and the three `_*_cp_matrix` are **zero-arithmetic facades** the page still cites as independent.
  ⭐ Method correction: the page's `vv-status rationale` comments answered **0 of 15** — the
  **test `pytestmark` COMMENTS** did, naming the exact symbol whose breakage reds the gate.
  `[M]` all 15 had **zero** `implements` edges (not even guesses) ⟹ measure the stand-down hazard
  before letting it shape breadth.
- [Which equations can be IMPLEMENTED at all (nexus #82)](nexus82_operator_algebra_implementers.md)
  — 21 of 40 declarable on `operator_algebra.rst`, 19 NONE. ⭐ The transferable half is the
  KIND split: `{identity, law, canonical-form}` → NONE always; `{typing-rule, definition}` →
  check for a DECLARATION SITE (a Protocol/class/TypeVar can carry a typing rule; nothing can
  carry a `≠`). The three verb labels resolve by the page's own base-hosting rule. + 5 doc-drift
  finds (⛔ `keff-as-integrated-rates` is present-tense-false vs `compute_keff`).
- [Spatial transform category](spatial_transform_category_durable.md) — the spatial
  layer's mirrors are **E(d) = O(d) ⋉ ℝ^d**, not O(3); the gap is the TRANSLATION
  (⛔ corrected 2026-08-20: `RigidMotion` ELEMENTS express it; the TAG layer +
  `close_group` still cannot hold a translation GROUP). Two genuine group objects: `octant_moment_frame_signs` (character
  rep of (Z₂)^d) and `reflection_index("x")` (the r=0 quotient's deck transformation).
  Sweep reversal spelled 11×, done right once (the adjoint).
- [Angular layer — hidden transformations](angular_layer_hidden_transformations.md) —
  the SH basis's polar axis is `μ_x` ⟹ every `Y_ℓ^m` carries an untaggable 120° `O_h`
  rotation; "octants" is really the `(Z₂)³=D_2h` orbit stratification; `_orbit_closure`
  is one of SIX partner-map engines; + two measured defect leads and the #325 sites.
- [Quadrature landscape](quadrature_landscape_durable.md) — which of {range, spacing,
  rule-on-circle/interval, exactness-space, node-generation} has ≥2 realizations; MoC's
  `[0,π)`+Σω=1 vs SN's `[0,2π)`+Σw=4π; `level_symmetric_sn` is EQUAL-weight at degree
  **3 for every N** (tag says N−1) ⟹ discrete SH Gram 25–45 % off at L≥2; every shipped
  `invariance_group` is a DECLARED tag (the checker's only caller is unreachable).
- [Convergence-knob semantics (#364)](convergence_knob_semantics_durable.md) — which knobs are
  the SAME quantity across the 5 iterating families and which are not. `dk`/`max_outer` same;
  `dphi` differs only cosmetically (`[M]` l-inf/l2 = 1.13–3.4, under the 10x tolerance gap);
  ⛔ `inner_tol` is a RESIDUAL in SN and an INCREMENT in CP/MoC, `[M]` differing by exactly
  `1/(1-rho)` — 1000x at c=0.999. CP's inner knobs are DEAD on its Jacobi default path.
- [SN solve exit + the reflective default](sn_solve_exit_and_reflective_default.md) — the exit is
  **THREE** `Solution`-construction sites (the forward fixed-source arms BYPASS `_package_solution`,
  whose docstring claims it is the only one); an **unset-BC** `Mesh2D`/axis-tuple resolves to
  **ALL-REFLECTIVE** (+ DD) on every route, and a PARTIAL declaration keeps the reflective default on
  the rest; the **tangential** ordinate bucket is what full-face functionals see and half-range ones
  do not; ZERO production consumers of the returned trace, and `Solution.compare` is bulk-only.
- [MoC equation→code truth](moc_equation_implementers.md) — the MoC kernel lives in
  exactly **TWO** places (production `solve_fixed_source` + the independent MMS `mms_sweep`);
  the page carries **NO** `vv-status rationale` comments (the authored knowledge is in the
  claiming test modules' pytestmark comments); + 2 measured defect leads (dual `t_s^eff`
  denominator, MMS trig-chart directions losing the #325 exact mirror) and the
  ⚠ never-`grep -v derivations` exhaustiveness trap.
- [loss_representation equation→code truth (nexus#82)](nexus82_loss_representation_implementers.md) —
  which of the page's 17 equations are LAWS with **NONE** (leaf-sum, removal-σ, facewise-separable)
  vs computed; ⚠ `loss-rep-resolution-a`'s stated mechanism is STALE — since #257 S8b `L` is
  `loss_action(σ=0)`, the subtraction exists nowhere; and the LpC guess pool was **disjoint** from
  the truth (`StreamingCollisionOperator` absent).
- [Monte Carlo equation→code truth (nexus#82)](nexus82_monte_carlo_implementers.md) — **22 of 22
  DECLARABLE**. ⭐ The transferable half is the KIND PRIOR: a page of ALGORITHMIC RULES
  (sample/estimate/split/wrap) is ~100 % declarable, an ALGEBRAIC-LAW page ~50 % — so skip the
  triage effort and spend it on the two real judgement shapes (an *expectation identity of a
  procedure*; a *test-tolerance* equation, whose only site is the test that claims it). ⛔ `[M]`
  **0 of 4** MC private kernels is imported by any test — four L0 gates replicate the solver inline.
- [SN multigroup axis structure](sn_multigroup_axis_structure.md) — three-tier
  group-blindness; NO group loop ("within-group" = fission-external); τ/c are ANGULAR
  closure weights, not optical thicknesses; `_within_group_triple` →
  `build_within_group_system` (coupled_system.py).
- [GENDF ingest truncation + (n,2n) probe traps](gendf_ingest_truncation_and_n2n_probe.md) — EVERY channel is
  Legendre-truncated at ingest (elastic P2, (n,2n) P0); `load_isotope` reads the .h5, tape moments need
  `_parse_gendf`/`_extract_mf6`; the solver's min-over-materials L clamp; never re-strip yield on ℓ≥1;
  `[M]` 421g 1-D k-solve ≈ 0.3 s/outer — no condensation needed.
- [HarmonicMomentField UNITS](harmonic_moment_field_units_convention.md) — why a stored
  SH moment carries SCALAR-flux units (no-prefactor SH, Y₀⁰=1, Σw=4π ⟹ sr cancels);
  R≠M*; the ERR-039/ERR-051 history.
- [Harmonic frame + folded-quadrature plumbing](harmonic_frame_folded_quadrature_plumbing.md)
  — NO computed Gram on the kernel path (exactness ASSUMED ⟹ the folded ξ-odd garbage);
  3 shape-contract tiers; σ_y parity per slot; the cylinder-P1 gate is folded-blind.
- [Flux torsor vs cone inventory](flux_torsor_vs_cone_inventory.md) — the #208 torsor is ONE mixin +
  7×7 leaves with exactly 2 production displacement-TYPE consumers (SI diagnostics duck-typed; DSA);
  the stop rides source-typed norms; `affine_combination` 0 production callers; superposition is
  SOURCE-level only; cone footholds already ship (`is_positivity_preserving` UNREAD, realizer cone
  refusal, coefficient cone battery, ray normalization); NO Step/SC realization ⟹ §6c witness gap.
  ⚠ snapshot 2026-08-19 pre-ruling — if the cone campaign landed since, this maps the RETIRED side.
- [No affine operator — the SPLIT convention](affine_operator_split_convention.md) — the tree
  has ZERO affine operator types and that is a RULING: an affine map in a linear slot measured
  `|B(0)| = q` and raised `ConvergenceCertificateError` (broke GMRES's Arnoldi relation), and
  delivered `q` twice. Every affine law = linear operator + typed SOURCE on the rhs; the
  `block_role` stamp is NOT the fence. Read before designing any "affine operator".
- [SN α ends + the ψ½ two-leg block](sn_alpha_and_psi_half_ends.md) — α is ONE-SIDED
  (far end an FP residual, never hard-set; its only production check is a bare `assert`
  that `-O` strips); the ψ½ block DOES march both decoupled ends but only `cells(p,−1)`
  seeds the angular recurrence; `[M]` the two ends' answers differ **8 %** (vacuum
  sphere) with zero gates comparing them — every flat/reflective fixture is blind.
- [Cylindrical SN level-order sensitivity (#326)](cylindrical_sn_level_order_sensitivity.md)
  — α IS `−W·ξ` at a half-angle boundary (Hébert 3.399, exact via the Dirichlet kernel);
  the recursion is a cumulative integral in ω ⟹ the level must be a HALF range, but
  ORPHEUS spans `[0,2π)`; every existing α gate is telescoping-blind; the ξ-mirror
  invariant (not the MMS) is the adjudicator. The 1-D cyl sweep reads ONLY (η, w) — ξ
  enters solely via the source. **+ the HALF-RANGE fix map**: the fold belongs in
  `LevelStructure`, NEVER in the `DiscreteMeasure`; (A) fold-existing-nodes vs (B)
  Hébert-midpoint are separated by the **R12a predicate**, not by α; the ONE real break
  is the ξ-odd SH moment, and it vanishes if you fold the ALGORITHM rather than the STATE.
- [Non-SN geometric-transform census](nonsn_geometric_transform_census.md) — zero
  hand-built rotation/reflection matrices outside `numerics/symmetry.py`; MoC owns the
  only 2 hand-rolled `_orbit_closure` clones (both guard-free); MC has periodic only;
  CP's images are an ADDITION; 4 spellings of one `(I−TR)⁻¹` orbit sum.
- [Phase 5 µ-resolved primitive inventory](phase5_mu_resolved_primitive_inventory.md) —
  µ-resolved vs µ-integrated primitives in `peierls_geometry.py`.
- [Pyright ignored-package measurement](reference_pyright_ignored_package_measurement.md)
  — true error count for a `[tool.pyright]`-ignored package: CLI file args + a /tmp
  ignore-free config; discount the editable-install import artifacts.
- [Census predicates: bound-method reference + activation traceback](census_predicates_bound_reference_and_activation_traceback.md) — a CALL census reads a BOUND verb as dead (0/1/0 = bound, say so); receiver-name regexes miss `head * bulk`; wrap-and-run with a 3-frame traceback finds the HOTTEST client (the R6 guard line = 58/118 densifier mints).
