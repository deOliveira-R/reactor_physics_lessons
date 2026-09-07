# Archivist — Memory Index

Slim index. Behavioral lessons live in `lessons.md` (read FIRST each
dispatch). The mechanical build-gating / cross-ref / venv-worktree /
close-out-arc procedure lives in `AGENT.md` ("Build-Gating & Cross-Ref
Reality", "Close-Out Narrative Arc"). The V&V vocabulary lives in the
`vv-principles` / `algebra-of-record` skills. This index holds only
(1) the lessons pointer, (2) git-true active/doc-debt state, (3) durable
doc-architecture reference. Campaign play-by-play is retired — its
behavioral lesson is in `lessons.md`; its landed milestones are in the
SN theory page's "Development history" section.

## 1. Lessons — a HOT digest over a COLD archive

Same hot/cold split as this index: read the digest always, page the archive on
demand. **Never re-summarize a lesson here or in the digest — each layer points
down, it does not copy up.** Counts are deliberately NOT quoted (a frozen number
rots; `grep -c '^## L-0'` answers it).

- [lessons.md](lessons.md) — **HOT digest, read FIRST every dispatch** (one `Read`
  fits it). Every lesson as one imperative rule + its failure→correction core, in
  9 themes: (1) verify against the LIVE tree · (2) the build is blind, grep is the
  gate · (3) a `:label:` is a V&V edge · (4) retirement & staleness · (5) page
  surgery · (6) doc SHAPE per event class · (7) V&V vocabulary curation · (8)
  code-prose rebalance · (9) gates & tooling. Each entry carries a `→ L-0NN`
  pointer into the archive.
- [lessons_archive.md](lessons_archive.md) — **COLD, load on demand** (~250 KB —
  never open whole). One `## L-0NN` section per lesson: war stories, evidence
  tables, `file:line` detail. Open ONLY the section a pointer sent you to; new
  lessons are appended HERE first, then distilled into the digest.

## 2. Active / doc-debt state — git-true

**No owed Sphinx pass on merged work.** Landed SN milestones live in the "Development
history" changelog at `docs/theory/methods/sn/history.rst` (⚠ NOT the pre-split
`discrete_ordinates.rst`, and orphaned July HTML survives in `_build`, so a stale-ref
grep must `test -f` the SOURCE). Active track = **#231** (§3); main agent commits, I
stage + gate.

**In flight — ONE line each. The STATE is in git (`git status --porcelain -- docs/`), never in
this list; the evidence is in the lesson.** Branch `fix/sn-eigenvalue-finalize-448`, docs
UNCOMMITTED (I stage + gate; the main agent commits).

- **#448 — a solver's RETURN is a claim, and the two guards' complements did not cover the exit**
  (2026-09-06; 16 `.rst`, 0 `orpheus/`/`tests/`). The eigenvalue finalize hand-built a **P0-only**
  reconstruction source, so at every `scattering_order ≥ 1` the returned ψ solved a different
  equation from the one the power iteration converged. Landed: **ERR-083** (minted, all numbers
  MINE on both trees — `git worktree add HEAD --detach` IS the pre-carve tree while the carve is
  uncommitted); a new H2 `sn-finalize-one-step` in `solver.rst` + eq-label `sn-finalize-map`
  (sentinels 597→**598**, predicted) + the `sn-exit-balance-projection` label and its ⛔ warning;
  the SN changelog row; 30 retired-symbol sites re-pointed across 16 files. Gate: **EXIT=1 / 1
  WARNING + 4 nexus `catches` lines → EXIT=0, 102-byte log**; `dead_references` **0/67**; my role
  probe **0/7056**; `:by:` **0/416**; HTML backtick gate **0** in all 8 new slices. ⚠ Three brief
  claims refuted (26 sites → 30; Protocol "3 members" → **5**; "7 arms" → **8**, the module grew
  45→86 rows mid-task). ⛔ 4 `orpheus/` docstring lies reported, NOT fixed (report §6)
  → [[lessons-L95]]

⏹ **MERGED 2026-09-03 — branch `fix/angular-phantom-support` is gone (`[M]` its commits are
ancestors of HEAD); every entry that used to sit here is landed.** #428, #434 R1–R4, #432 and
#429 (2.5 / 3.1 / 2.3 / 2.1b / 2.0a / the fused commit) are archived as
[[lessons-L80]]…[[lessons-L92]], each naming its own commits. `git log --oneline -- docs/` is
the index. ⚠ This list had frozen "in flight" on landed work **seven** times before this
collapse — reconcile with git BEFORE reading it, every time.

⚠ **ERR-026 history block — status CHANGED, not confirmed.** Its branch
`docs/err026-history-is-not-a-crossref` no longer exists locally or on the remote, so the
2026-08-24 "still OPEN, unlanded" claim is void. `[M]` the gate's guard now reads
`bare_module_guess = "." not in target and role != "mod" and …`
(`tools/check_docstring_xrefs.py:549`), i.e. it keys on the TARGET being undotted — so the
L-062/L-067 dotted-target blindness **appears repaired**. NOT directly re-probed (a standalone
import of the gate module raises inside `dataclasses`); corroborating evidence only: the stock
gate and my own import probe, two independently-vocabularied instruments, both read 0 dead.

⚠ **Every entry above is a snapshot.** Reconcile with git FIRST — this list has frozen
"in flight" on landed work SIX times. `git merge-base --is-ancestor <hash> HEAD`, and
`git branch --list <branch>` (a vanished branch means merged).

**⏹ MERGED — the durable record is the lesson + the tree; this is only a pointer.**
Every docs pass from 2026-08 back to the Boundary/DSA work is archived as
[[lessons-L39]]…[[lessons-L79]] (one `## L-0NN` section each, with its commits).
Nothing here needs re-listing: `git log --oneline -- docs/` is the index, and each
lesson names its own hashes. Their CODE-side reports are GitHub's to track; the
corpus-wide RST-nested-markup finding lives on **#379**.

## 3. Durable reference (reusable doc-architecture)

Each entry is a ONE-LINE pointer; the full recipe lives in the linked `feedback_*.md`.

- **Landed-milestone record:** `docs/theory/methods/sn/history.rst`. POINT here instead of
  re-listing campaigns. (This line named the pre-split `discrete_ordinates.rst` until 2026-08-18,
  contradicting §2 four lines up — an index can go stale against ITSELF.)
- **Ontology-overturn rewrite** (a page whose THESIS was refuted): the recipe is
  [[lessons-L63]] — argument-unit not symbol-unit, the 4-way eq-label fate rubric, the
  unlabelled-history-equation trick, the two-sided illegal-states rule. Instance:
  `field_algebra.rst` affine → cone (CS3).
- [canonical-convention-page](feedback_canonical_convention_page.md) — 13-section anatomy
  for a multi-PR migration's canonical theory page + keep/flip rubric (`index_convention.rst`).
- [canonical-axis-convention SSOT section](feedback_canonical_axis_convention_ssot_section.md)
  — SSOT section for an axis-flip enforced at a data-ingest boundary (`cross_section_data.rst`).
- [double-category architecture insight](feedback_double_category_architecture_insight.md) —
  documenting a categorical framing of a SHIPPED type system; impossibility as an
  obstruction table. Instance: (Rep×Role) carrier grid (#268/#261).
- [orientation-axis two-frames doc](feedback_orientation_axis_two_frames_doc.md) — 2×2-face
  operator unification with ORIENTATION as the coherence axis (#280 P2.5e).
- [carrier-grid-typed-seam-layering](feedback_carrier_grid_typed_seam_layering.md) — NxM typed
  grid + seam one layer up; completing one path silently stales a sibling claim (Frame P4).
- [capstone-architecture-page](feedback_capstone_architecture_page.md) — a NEW page for the
  LAYER above per-method pages (cross-ref, don't duplicate). `loss_representations.rst`.
- [capstone-completion-status-reaudit](feedback_capstone_completion_status_reaudit.md) — the
  COMPLETION phase: re-audit ship-state claims; document an unbuilt sibling as a SEAM (P7).
- [capstone-root-cause-ruling](feedback_capstone_root_cause_ruling.md) — retrofitting the
  structural WHY (a theorem) behind a split the docs only ASSERTED (#268 `frame.rst`).
- [operator-classes→frame-faces re-homing](feedback_operator_classes_to_frame_faces_rehoming.md)
  — sweep when standalone operator classes retire into two FACES of one frame (#268 P1).
- [operator-reification/retype doc pattern](feedback_operator_reification_retype_doc_pattern.md)
  — reifying a duck-typed operator; block coisometry `= 4π·I`, never `= I` (#226 step 2).
- [named-family-member theory section](feedback_named_family_member_theory_section.md) — a NEW
  § for a named member of an invariant-keyed operator family (#226 step 4 GreenOperator).
- [step-5b first-consumer close-the-loop](feedback_issue_138_step5b_first_consumer_closeloop.md)
  — wiring the FIRST consumer of a verified-but-unwired type; → vv Mode 12 (#226 step 5b).
- [consumption-mode + capability-axis](feedback_consumption_mode_and_capability_axis.md) — a
  NEW consumption mode on an operator algebra (solve/apply/ASSEMBLE) (#272/#284/#282).
- [algebra-of-record stub→narrative](feedback_stub_to_rich_narrative_expansion.md) —
  SymPy-module-as-canonical-source; stub/expand separation (also lessons L5).
- [solver-replacement campaign close-out](feedback_solver_replacement_campaign_closeout.md) —
  a legacy island solver replaced by the operator-algebra family; LIVE/MOOT split (#290 P8).
- [type-confinement docstring sync](feedback_type_confinement_docstring_sync.md) — code-final
  sync when a carve confines a subtype to one role (P4.5 W-C).
- [Petrov-Galerkin homogenization reframe](feedback_petrov_galerkin_homogenization_reframe.md)
  — THE LIVE recipe: flux-weighting is a TEST weight, not a measure (#268 P3). Supersedes
  [Galerkin-natural-metric](feedback_galerkin_natural_metric_reframe.md) (why-it-was-tried only).
- [domain-op + L2-promotion + asymmetry-law](feedback_domain_op_l2_promotion_asymmetry_law.md)
  — section shape for a domain OPERATION born from an L2 promotion (#267).
- [orbit-space terminology sweep](feedback_orbit_space_terminology.md) — add-aside-then-bridge-then-sweep for a precise math term.
- [auto-generated tables](feedback_autogen_tables.md) — registry-as-SSOT: metadata fn +
  generator + `builder-inited` hook (also lessons L8).
- [audit-then-edit partitions](feedback_audit_partition.md) — the KEEP/RELOCATE/TRIM/REMOVE
  partition table for a read-only doc-cleanup audit.
- [cross-solver unified-law doc architecture](feedback_cross_solver_unified_law_doc_architecture.md)
  — ONE law spanning N solver families: canonical derivation + short sibling spellings (#259/#291).
- **Doc-architecture redesign (#231, OPEN):** the standing target for any "modernize a theory
  page" task — template, machine header, prose rebalancing, V&V slices, bibtex (spec in the
  issue). Phase 1a–1c DONE; **Phase 2 code-prose rebalancing ACTIVE** — P2-A/B/C/D/G done
  (maps in `.claude/plans/phase2_code_prose/`). Five file-classes calibrated:
  teaching-operator = aggressive TWIN-cut; machinery/driver/mesh = small, COMMENTS dominate;
  ABC = leanest; contract-heavy-operator = small. Main agent commits; I stage + gate.
