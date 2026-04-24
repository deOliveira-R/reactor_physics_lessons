# Diagnostic → Permanent Test Promotion Policy

`derivations/diagnostics/diag_*.py` is a scratchpad for the
**numerics-investigator** agent and human debuggers. Scripts land
there during an investigation and pile up fast. This policy decides,
once an investigation is over, which scripts graduate into the
permanent V&V harness and which are deleted.

The rule is intentionally simple — the git history is the
investigation record; the test harness is the regression record.

## Triage: one of three outcomes per script

### 1. DELETE — bug is now covered by a permanent test

If a diag script exercised a real bug that is now gated by a test in
`tests/`, delete the diag. Its value lives in the permanent test and
in `git log` / the linked GitHub issue. Keeping the diag duplicates
the gate and invites drift.

**Example.** Issue #131 Probe F ran the 2G 2-region white_f4 fixture.
After the fix, the permanent
`TestSlabViaUnifiedDiscrepancyDiagnostic::test_2eg_2rg_parity_bit_exact`
gates that exact fixture at `rel_diff < 1e-10`. Probe F was
deleted.

### 2. PROMOTE — covers a regime not yet gated

If a diag exercises a regime, axis, or invariant **not** yet covered
by a permanent test, promote it. Move the assertion into
`tests/derivations/` (or the appropriate module), add the proper
markers, and delete the diag.

Promotion markers, pick one:

- `@pytest.mark.l0` / `l1` / `l2` + `@pytest.mark.verifies("label")`
  where `label` is a `:label:` in `docs/theory/*.rst`. For physics
  equations.
- `@pytest.mark.foundation` for software invariants that do not
  correspond to a theory equation (reduction properties, factory
  outputs, data-structure immutability). No `verifies(...)`.
- `@pytest.mark.slow` for tests above ~30 s wall time.

Make sure the test's docstring cites the originating Issue and says
*what bug it would catch* — not just *what it computes*.

**Example.** Issue #131 Probes A and B ran vacuum-BC parity on 1G and
2G 2-region slabs — regimes that *isolate* multi-region handling from
the F.4 closure (the permanent flagship test runs `white_f4` only).
Both were promoted into `TestSlabMultiRegionVacuumParity`. Probe D
exercised a software invariant (multi-region branch with σ_t uniform
must reduce to single-region closed form) and was promoted into
`TestMultiRegionEscapeReduction` as a `@pytest.mark.foundation`
test.

### 3. LEAVE — investigation still active

If the script is part of an ongoing investigation (the target bug is
not yet fixed, or the script is a scan/characterization being
iterated on), leave it. Diagnostics under active use *should* pile
up; that is the folder's purpose.

Rule of thumb for "still active": the referenced GitHub issue is
still open, or the commit that added the diag is within the last
session's work. If neither, reconsider (the investigation may have
concluded and nobody triaged the scratchpad).

## When to run the triage

- Immediately after landing a fix for the Issue that spawned the
  diagnostics. The investigator knows what each probe proved; do the
  triage while context is fresh.
- Opportunistically, when starting work on a module — if the diag
  folder has stale artifacts from a closed Issue, triage them first.

## What not to promote

- Scripts whose only assertion is a print / plot.
- Scripts that hard-code a failing value as the expected answer
  (these reflect the bug, not the spec).
- Scripts that need external data not under version control.

Convert these into real tests or delete — do not pretend they are
regression gates by adding a pytest marker.

## What lives in `derivations/diagnostics/` permanently

Nothing. Files named `_*.json`, `_*.py`, and other leading-underscore
artifacts are investigation *state* (scan results, cached derivations)
and may persist, but the `diag_*.py` runnable scripts are all
short-lived by policy.
