---
name: census-predicates-bound-reference-and-activation-traceback
description: A CALL census is structurally blind to a bound-method REFERENCE (`reflect=self.lower.reflect_rows_inplace`) and reads a live verb as dead; a receiver-name regex on `*` products misses `head * mesh.bulk_space`; and an activation counter on the SPACE MINT (`FunctionSpace.__mul__`) with a 3-frame traceback census located the densifier's hottest production client — a GUARD line — in one run. Learned on the step-6 boundary re-census, 2026-09-07.
metadata:
  type: feedback
---

**Rule.** When a verb can be PASSED as a callable, a call census (`ast.Call`
whose `func.attr == verb`) must be paired with a bare-attribute census
(`ast.Attribute(attr == verb)` NOT under a `Call`), and the positive control
must be chosen on the spelling you are least sure of. And when the question
is "which production site is the hot consumer of X", do not census callers —
wrap X with a counter that records `traceback.extract_stack(limit=4)` and RUN
the production arms.

**Why (measured 2026-09-07, `scratch/_step6/explorer_boundary_recensus.md`).**

- `SNMaskedBoundaryOperator.reflect_rows_inplace` is the live inter-group
  reflect of the boundary Gauss-Seidel resolvent. `[M]` the AST call census
  over `orpheus/` returned **0** — and its own positive control ("found in
  `orpheus/sn`") FAILED, which is what made the blindness visible: production
  BINDS it (`sn/operators/scheduled_invertible.py:274
  reflect=self.lower.reflect_rows_inplace`) and the call happens through the
  parameter name (`reflect(boundary_flux, group.reflect_faces)` in
  `_sweep_scheduled`). Nexus `callers` returned `nodes: []` with NO unresolved
  block (a bound reference mints no call edge and no receiver phantom), so
  the graph and the AST agreed on the wrong answer. Only a bare `grep -rn
  <verb>` filtered to non-`def`, non-prose lines found the reference. This
  is §6b's "spelled without the symbol" family one tier further: not a
  variable-call, a NON-call.
- A receiver-name regex for space products (`[a-z_]*space[a-z_]* \* …`)
  found 2 production `*` mints and missed the one that fires 113× per
  windowed solve — `head * mesh.bulk_space` (`transport/fields/_bases.py:911`)
  — because the receiver is spelled `head`. The runtime census found it.
- The activation census: wrap `FunctionSpace.__mul__` (and the densifier
  helpers) with a counter keyed on a 3-frame traceback, run one 2-outer
  2-D SI `solve_sn`. `[M]` 58 of 118 product mints came from
  `sn/operators/boundary.py:714 _apply_faces` → `space_on` → the mint — i.e.
  the R6 carrier-guard line is the densifier's busiest client, and the
  §7.3 "densifier-native" item and the R6 item are ONE line. No static
  census (callers, grep, AST) can return "hottest"; only the run can, and
  it costs one script (`scratch/_step6/…` describes it; the probe itself is
  ~40 lines).
- Sibling refutation from the same run, worth carrying: the P7
  `_tensor_product_factored_metric` arm the plan named as the "native"
  client fired **0×** on 11 SN runs; the LEGACY dense arm is the live one.
  A plan can name the wrong client with a correct symbol — the counter is
  what discriminates "exists" from "runs" (vv#29 mode).

**How to apply.**
1. For every verb in a census that is a METHOD (not a module function), run
   three predicates and state all three: literal calls, bound references
   (`Attribute` not under `Call`), and Nexus `callers` + its `unresolved`
   count (L-039). A verb with 0/0/0 is dead; 0/1/0 is BOUND — say so.
2. Put the positive control on the spelling the population is most likely
   to use, not on the one you already saw (`SNBoundaryOperator(` in
   `coupled_system.py` was a control that could not fail; the
   `reflect_rows_inplace` control was the one that carried information).
3. For "hottest / which arm / does it run" questions, wrap-and-run over the
   production arms (1-D SI, Krylov, 2-D SI/G-S/Krylov) BEFORE pricing a
   retirement; report per-arm counts and the traceback keys. A grep that
   filters by receiver NAME is a §2 FILTER defect on a product/dunder
   operation — the receiver of `*` is arbitrary.

Related: L-039 (Nexus `callers` on a METHOD), L-013 (swap-it-and-run beats
grep-classification), [[sn-solve-exit-and-reflective-default]].
