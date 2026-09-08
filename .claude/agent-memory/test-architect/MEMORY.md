# Test-Architect Memory Index

One line per entry — detail lives in the linked file, NEVER inlined here (the
index is loaded whole every dispatch; keep it small). Four sections: (1) lessons
— READ `lessons.md` FIRST every dispatch; (2) active/in-flight state — git-true
(reconcile "unmerged" claims against git before acting); (3) durable reference
recipes; (4) design idioms. The failure-mode taxonomy lives in `vv-principles`;
the reference inventory + XS mixtures in `AGENT.md`. No campaign play-by-play
here — it is merged archaeology.

## 1. Lessons — a HOT digest over a COLD archive (READ the digest at START)

- **[Lessons — hot digest](lessons.md)** — ~1700 lines. One imperative rule per
  entry, grouped by behavioral family (gates that cannot red · harness discipline ·
  config blindness · reference & claim layer · tolerance · carve archetypes ·
  snapshots & exactness · pure-math primitives). **Read this file whole, every
  dispatch.** Every entry ends with a `→ LNN` pointer into the archive.
  ⚠ **A digest entry recording a GAP needs a LANDING note the moment the gap
  closes** — `L3`'s "no SN MMS exercises `q.boundary ≠ 0`" was true when written,
  the §4.6 fix landed, and the stale entry then generated a whole phase brief for
  work already done (`L40a`).
- **[Lessons — cold archive](lessons_archive.md)** — ~8300 lines, sections L1–L72,
  append-ordered. The war stories, measured numbers, `file:line` detail and
  per-fixture tables. **Open ONE section at a time, only when the digest's pointer
  says the detail matters.** Never read it whole — that is ~48K tokens.
- NO lesson content is inlined here. The digest is the index over the archive;
  this file is the index over everything else. New lessons: add the RULE to the
  digest (with its `→ LNN`) and the war story as a new archive section.

## 2. Active / in-flight verification work

**Detail → [active campaigns](active_campaigns.md) and `lessons_archive.md` §LNN.** ONE line
each here — name, terminal status, pointer. Reconcile every "unmerged" claim against git
first. Cross-transferable rulings are already in `lessons.md`; nothing else belongs here.

- **CS4c step 6 — the CS2 residue: identity, metric, carrier guard, reflect verbs** — ⏹ PLAN + 4 ANCHOR FILES DELIVERED 2026-09-07, PRE-carve (`scratch/_step6/test_architect_verification_plan.md`, 1077 lines; **55 rows landed green**, pyright 0; 10-arm battery over 5550 rows). ⛔⛔ TWO BLOCKING rulings: the F1 `require_member` ruling names a SHAPE and a SEMANTICS no one signature satisfies; the factored-metric arm is UNREACHABLE from the factors (`axes=None` + dense slot + `metric=None` on 8 of 8 rows). 10 hazards, 6 open rulings. → **`L79`**
- **#448 — the eigenvalue finalize returns a flux that solves the equation it reports** — ⏹ **R2 COMPLETE 2026-09-06, POST-carve** (carve landed as option B, uncommitted on `f75a9e59`). Gates: `tests/sn/solve/test_eigenvalue_finalize_reconstruction.py` **86 rows** (+`cart2d_gs` G-S arm, +`TestTheReturnedTrace` = R1's owed gate, +the G-S splitting class) and **NEW** `tests/numerics/test_fixed_point_step.py` (8 rows) — `[M]` **94 passed / 52.9 s**, pyright 0; 32 anchors in `tests/sn/_data/finalize_reconstruction_448/` (28 pre-carve + 4 post-carve) ⚠ **STILL UNTRACKED**. Battery `scratch/_448/battery_r2.{py,sh}` 12 arms RUN; memo `scratch/_448_verification_plan.md` §12. ERR-083 MINTED. ⭐⭐ `φ = ∫ψ dΩ` is the DEFINITION — a claim layer that needs no reference. ⛔⛔ Four R2 findings: the phase hook LEAKED (its own positive control caught it); a declared blindness named the wrong symbol AND its mutation diverged the solve; a two-ended defect needs two arms; `dead_references` on an uncommitted tree reports graph staleness. OPEN: R3 (G3c) with the user; System-B ψ½ still ungated. → **`L78`**
- **CS4c step 5 — each binding acts through the body its ends select** — ⏹ PLAN DELIVERED 2026-09-04, PRE-carve (`scratch/cs4c_step5_verification_plan.md`; extends the older plan's §11.6 3-arm skeleton into **11 gate classes + 18 battery arms**; scopes `[M]` core **92.75 s** / +windowed **153.77 s** / adjoint+ERR-082 **86.2 s**; predicted **11142 → ≈11206**, xf 66 unchanged). ⛔⛔ **BLOCKING O-1**: the moment composite's interior has `axes is None`, so the flagship moment-domain sibling is UNCONSTRUCTIBLE as designed (`_scalar_interior_space` raises; read it off the CODOMAIN). ⛔⛔ **O-2/F-2**: the done-when AST predicate is LEXICAL and 3 carrier parses live one frame out. 6 open rulings, 8 findings. → **`L77`**
- **#426 — the (n,2n) channel keeps its angular distribution** — ⏹ PLAN + 5 DRAFTS DELIVERED, PRE-carve (`scratch/_426_{verification_plan.md,draft_test_be_reflected.py,draft_test_tape_pin.py,draft_test_h5_roundtrip.py,draft_test_role_ast.py,draft_test_diffusion_n2n.py}`; **41 rows, 22 RED today** = the §6c red-before, pyright 0; predicted +89 → 11096, 54 MEASURED / 34 design / 1 corpus-coupled). ⛔⛔ **BLOCKING ruling O-1**: 2 of 13 isotopes have `sig2.nnz = 0`, so a two-list clamp forces P0 on every water-bearing solve (`[M]` +5787 pcm-relative). 5 open rulings, 17 hazards. → **`L76`**
- **#434 R3 — the registry names what it spends, what it leaves, and what it owes** — ⏹ PLAN + GATES + BATTERY DELIVERED (`scratch/_r3_{verification_plan.md,gates_draft.py,test_migration.md,mut.py,battery.sh}`; **42 rows / 8 classes**, 42 passed 1.84 s, pyright 0; **battery 16 arms, 16 BIT, 14 red, 2 DECLARED NULLS at 0**, pristine clean; predicted +42 → 11007). ⚠⚠ the carve landed **three times** mid-dispatch (production, tests, then an elegance pass that re-signatured the predicate). 6 brief premises REFUTED — incl. `quotient_group` is `None` not `Trivial`, the D∞h slab fold is UNCONSTRUCTIBLE, and `TestR2SelectionIsUnchanged` RED on 4 of 4. → **`L75`**
- **#434 R2 — invariance is the measure's question; groups import geometry only** — ⏹ PLAN + GATES DELIVERED (`scratch/_r2_{verification_plan.md,gates_draft.py,test_migration.md,shim.py}`; 121 rows dry-run 103/121 in 5.92 s, 19-arm battery, 5 open rulings, 2 frozen baselines; **battery SHIPPED** `scratch/_r2_{mut.py,battery.sh}` — 23 arms, pre-landing 23 UNINSTALLABLE / 0 red / pristine clean, 15 of 23 BITE-validated through the shim). ⛔⛔ BLOCKING: the carve does not IMPORT as planned — `AXIS_INDEX`/`AXIS_LETTER` must leave `manifold` first (`[M]` 6 of 9 entry points die). → **`L74`**
- **#434 R4 — the lift is a derivation output, an orbit space's dimension is a theorem** — ⏹ PLAN + GATES + BATTERY DELIVERED (102 rows landed; **22-arm battery, 22/22 bit, pristine diff clean**, `scratch/_r4_{mut.py,battery.sh,battery.log}`). ⚠⚠ the kernel is Mode-12 BLIND (`[M]` 0 of 9925) and `dim_law_reads_dim_H` reds **0 of 4597** — witness delivered (`scratch/_r4_dimH_witness.py`, 2 pass / 2 fail). → **`L73`**
- **#434 R1 — every question about a group is computed from its REALIZATION** — ⏹ PLAN + GATES DELIVERED (`scratch/_r1_{verification_plan.md,gates_draft.py,test_migration.md,mut.py}`; 28 gates, 20-arm battery, 26/28 rows red). ⚠⚠ the carve LANDED mid-dispatch ⟹ every gate MEASURED. → **`L72`**
- **#429 tracker 2.2b — the Γ-slot** — ⏹ PLAN DELIVERED (`scratch/_22b_verification_plan.md`, 14 gate classes, battery NOT run). → **`L71`**
- **#432 / #429 tracker 1.9 — the orbit space is named by its STABILISER `O(2)_a`** — ⏹ GATES SHIPPED (43 tests / 5 files, 17-arm battery, 3558 passed). → **`L70`**
- **#429 2.1-W — the σ-even QUOTIENT sub-basis a folded rule binds** — ⏹ GATE SHIPPED (3 rows, 6-arm battery). → **`L68`**
- **#429 FUSED step (Landing A + B)** — ⏹ PLAN DELIVERED (`scratch/_fused_verification_plan.md`; 20 gates, 16 hazards, 7 open rulings). → **`L69`**
- **CS4c binding ladder — every operator receives its two spaces** — ⏹ PLAN DELIVERED (`scratch/cs4c_verification_plan.md`; 10 contradictions, 8 gate specs, 7 open rulings). → **`L67`**
- **P4-remainder step 2 — the producer binds the angular axis** — ⏹ PLAN DELIVERED (`scratch/p4rem_step2_verification.md`). → **`L66`**
- **CS5 — an axis can name the generator that made it** — ⏹ PLAN DELIVERED (`scratch/cs5_verification_plan.md`; 9 gates, 5 rulings). → **`L65`**
- **P4.9b — the operator is posed with its two closures** — ⏹ PLAN DELIVERED (`scratch/p4_9b_verification_plan.md`; 11-arm battery, 7 blocking rulings). → **`L64`**
- **P4.9a — per-cell angular un-weld** — ⏹ PLAN DELIVERED (`scratch/p4_9a_verification_plan.md`; 9-arm battery, 5 rulings). → **`L63`**
- **CS4b fields-are-space-elements** — ⏹ PLAN DELIVERED (`scratch/cs4b_verification_plan.md`). → **`L62`**
- **CS4a kernel core** — ⏹ PLAN DELIVERED (`scratch/cs4a_verification_plan.md`). → **`L61`**
- **CS1.5 Medium un-weld** — ⏹ PLAN DELIVERED (`scratch/cs15_verification_plan.md`). → **`L60`**
- **CS1 Energy axis / axis-composed spaces** — ⏹ PLAN DELIVERED (`scratch/cs1_verification_plan.md`). → **`L59`**
- **CS3 cone overturn** — ⏹ PLAN DELIVERED + gate SHIPPED (`tests/numerics/test_si_diagnostic_trajectory.py`). → **`L58`**
- **#358 forward half — graph-grounded test workflow** — ⏹ MEMO DELIVERED (scratchpad; 8 demands on Nexus). → **`L55`**
- **#344 step 7 — the reflective-box `ker A` characterization** — ⏹ SHIPPED (25 rows, 13-arm battery). → **`L49`**
- **#340 R5 / N4.7 / N6 — the convergence-contract trio** — ⏹ DELIVERED (`scratch/n4_7_verification_plan.md`; N6 shipped 21 gates). → **`L44`–`L46`**
- **#235 angular-differencing — design the ranking INSTRUMENT** — ⏹ DELIVERED (`scratch/q68_angular_instrument_design.md`). → **`L48`**
- **SN curvilinear angular-closure seam** — ⏹ SHIPPED (3 files, 89 passed, 13-arm battery). → **`L47`**
- **Q5.6 "6.3 flip" — `SNMesh(CYLINDRICAL)` admits only CARRYING quadratures** — ⏹ DELIVERED (`scratch/q5_6_3_gate_design.md`). → **`L43`**
- **G2 geometric-transformation machinery** — ⏹ SHIPPED 2026-08-03 (42 gates / 96 cases, 32/32 mutations caught). → **`L35`**
- **#337 moment-matched level-symmetric node seed** — ⏹ PLAN DELIVERED (`scratch/issue_337_verification_plan.md`). → **`L42`**
- **G6.3 step 7 — the deck-transformation uplift** — ⏹ SHIPPED (141 gates over 2 files). → **`L41`**
- **P4 non-trivial MMS through the DECLARED inflow channel** — ⏹ DESIGN DELIVERED (`scratch/p4_mms_design.md`). → **`L40`**
- **P3 affine boundary source channel** — ⏹ PLAN DELIVERED (`scratch/p3_verification_plan.md`); ⚠ 11-item residual gap list. → **`L39`**
- **G6 every operator knows its spaces · G6.3 step 5 · G5 self-paired deck collapse** — ⏹ PLANS DELIVERED (`scratch/g6*_verification_plan.md`, `scratch/g5_verification_plan.md`); OPEN rulings C1/C3/C4, Q1, guard A-vs-B. → **`L36`–`L38`**
- **#325 symmetry-exact circle nodes** — ⏹ SHIPPED (17 gates); ⛔ BLOCKED on a USER physics ruling (exactness manufactures `argsort` ties worth 1.008 % flux). → **`L34`**
- **B3.4c periodic → partner face** — ⏹ PLAN + dry-run gate module (95/95). → **`L33`**
- **Boundary machinery B3.2 → B3.4b + #21** — ⏹ ALL DELIVERED; six open follow-ups. → **`L29`–`L32`**
- **Three-DOF separation (operator ∥ splitting ∥ realization)** — ⏹ P0 SHIPPED (`tests/sn/architecture/`). → **`L24`–`L26`**
- **#280 RESIDUE · DSA for SN (#2)** — ⏹ SPECS in `.claude/plans/archive/`; PRE-carve, DSA runs after #280. → **`L23`**
- **Coupled Block Operator step-5 (#41)** — ⏹ SPEC delivered; awaiting user rulings D2a/D3/D5. → [spec](coupled_operator_step5_solve_verification.md)
- **Diffusion #290** — ⏹ MERGED @ `3a19133`. → [plan](diffusion_integration_290_verification_plan.md)
- **A3/#280 Phase-2.5 walk-unification** — ⏹ gate files on `main`. → `L17`–`L19`, [recipe](a3_reverse_scan_transpose_verification.md)
- **Prior SN campaigns** (#206/#208/#236/#240/#247/#251/#257/#18/#19/#20) — ⏹ MERGED to `main`, NOT open work.

## 3. Durable reference (reusable verification-design recipes)

Reusable RECIPEs / cited by `AGENT.md`. Core lessons in `lessons.md`; these keep the worked method.

- [Convergence-RATE verification](si_convergence_rate_verification.md) — AGENT.md §5. Iterations-to-converge vs analytic SI ρ=c; measurand `history.n_inner`; the OPEN eigenvalue-path `n_inner=None` gap; rate-claims flux-shape-independent → 1G-OK.
- [Snapshot migration when production goes BARE](snapshot_migration_when_production_goes_bare.md) — AGENT.md §7. Shared-driver SoT; schema=persisted∩compared; VACUUM-bit-id gate; snapshot-inheritance-needs-anchor; false-`@catches` retirement; term-activation re-verify.
- [SN sentinel harness](sn_sentinel_harness.md) — `@pytest.mark.sentinel` one-cheap-test-per-capability-node; cosmic-ray mutation-validation (`git checkout` after each run); per-NODE-sentinel-leaves-interior-uncovered gap.
- [SOTP separability verification](sotp_separability_verification.md) — separable ⟺ Cartesian-product per-axis; coupled physics → OperatorSum fallback; Route-A array_equal vs Route-B nulp; slab degenerate.
- [Operator space-guard only bites OperatorSum](operator_space_guard_only_bites_operatorsum.md) — the domain/codomain guard is INVISIBLE to SI/Krylov matvec; bites only actually-composed sums; `FunctionSpace.__eq__` by `(name,shape)`; activation-gate the composed sum.
- [Cross-layer relocation carve](cross_layer_relocation_carve_verification.md) — relocate-down + registry-dispatch. H1 registration-timing MASKED by process-global state → fresh-process subprocess gate mandatory. H2 `TYPE_CHECKING` sn import trips `test_layer_imports`. Layer-inversion usually doc-only at runtime.
- [A3/#280 reverse-scan transpose-solve](a3_reverse_scan_transpose_verification.md) — reverse-DAG `apply_transpose`; retired-CAP→typed-predicate reconciliation; assembled-Mᵀ (Cartesian-only) vs dense-apply SPHERE keystone; 1-D loop spy + orientation-OBJECT AST tripwire. §7 CYLINDER arm: mandatory `product(n_mu=4,n_phi=8)` (LS nulls both hard terms=control); G1/G2-dense-Mᵀ-keystone/G3-full-field-recip(#284)/G4/G5; ERR-066 degenerate-drop tooth.
- [A_BA ψ½ Schur-fold un-weld](aba_schur_fold_unweld_verification.md) — lessons L22. Welded-fold un-weld (N sites→ONE source). 7 gate types: manufactured-anisotropic fold contract, Mode-11 wrap-counter EXACT `2·n_levels`, bit-id INHERITS + independent `½·emission`, two transpose gates, F-non-vacuity, cyl/slab None-ray control.
- [CoupledOperator Step-4 verification](coupled_operator_step4_verification.md) — N-general block machinery (ψ½=instance #1). 4d.0 `FullField`→`System[I,B]` structure-only (multi-instantiation synthetic CRUX). 4d.1 assemble≡probe principled-equiv + block-`.H` Mode-12. 4d.2 presence=block-existence. 4d.3 block-apply WRAPs fused walk + two-anchor.
- [CoupledOperator B.2b re-type](coupled_operator_b2b_retype_verification.md) — pure re-labeling → `array_equal` EVERY row (any rtol/nulp = RED FLAG). b1 split SourceSink + role-preserving bridge (role⊕values split-blind). b2 family-blind `from_blocks` + presence-dispatch. b3 A_BA/B_b onto ray composite + adapter-delegation sentinel.
- [A_AB seed-injection](a_ab_seed_injection_verification.md) — cell-local rectangular coupling (ray→bulk, σ-indep) = `A_bs` block. Equivalence gates SHARE closure methods → blind; the ONE catcher = gate-3 Euclidean fwd↔transpose adjoint-consistency + `test_radial_characteristic_metric` anchor. Sphere ONE level → multi-level untestable.
- [A_BB forward shared-kernel EXTRACT](radial_characteristic_forward_extract_verification.md) — Step 4b. Round-trip PRINCIPLED-EQUIV ~3 ULP not 0-ULP; `solve∘apply=id` only on CONSISTENT subspace; transpose seam adds seed_cells_bar A_AB term; EUCLIDEAN not V_cell metric; Mode-11 anti-twin routing sentinel.

## 4. Durable design idioms (feedback)

- [Regression tolerance design](feedback_regression_tolerance_design.md) — iterative→`SAFETY(10)×conv_tol` off run-config SoT, direct→`nulp(reduction_depth)`; `DriftWarning` tripwire; `-O`-safe.
- [Eigen on non-fissile mixture is malformed](feedback_eigen_on_nonfissile_mixture.md) — k=0/abs→nan dead gate; reformulate fixed-source; corroborate vs `(diagΣ_t−Σ_s0ᵀ)⁻¹Q`.
- [Diagnostic→test promotion](feedback_diagnostic_promotion.md) — verify-diag-runs-first; reproduce via public API; 3 foundation classes; delete-after-pass. (SoT: `tests/derivations/_promotion_policy.md`.)
- [V&V tagging idioms](feedback_vv_tagging.md) — module `pytestmark` vs per-test `verifies()`; foundation carries NO `verifies()`; xfail `strict=False`+`reason=`.
- [Cross-method protocol design](feedback_cross_method_protocol.md) — reuse registry schema; `max(tol_a,tol_b)` agreement; L1-not-L4; verify truth values vs literature memos first.
