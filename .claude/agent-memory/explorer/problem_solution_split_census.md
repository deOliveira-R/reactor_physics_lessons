---
name: problem-solution-split-census
description: Where the consumers-campaign opener census lives (scratch/_consumers/explorer_problem_solution_census.md, untracked) and the four durable findings about SNMesh vs Solution that a re-census should re-verify first
metadata:
  type: reference
---

**Memo:** `scratch/_consumers/explorer_problem_solution_census.md` (UNTRACKED `scratch/`; written 2026-09-08 at HEAD `7e9b6210`, graph @ `82c0d441`). Probe scripts under the session scratchpad are gone; every count carries its command so it re-runs.

Durable shape (re-verify, don't trust the numbers):
- The SN hub owns SPACES + realized BCs + the bound closure + chart data and **zero bound bulk operators**; every operator (L, C, S, N2N, F, B_a, System B, MaterialXSField, DSA, collision cache) is constructed on the SOLVER side (`SNSolver.__init__` + `build_within_group_system` per outer step). `HomogeneousProblem` is the inverse shape (17 cached operators/fields on the hub, a thin reader solver). Diffusion mirrors SN (operators on `DiffusionSolver`).
- Solution-side state living on the hub: the 1-D walk-ORDER family (`dag_walk` + helpers) and three memo slots WRITTEN from outside (`_coll_cache`, `_pole_mirror_cache`, `_geom_cache`) — pinned by `tests/sn/sweep/core/test_cache.py`.
- The k-outer iterate is a bare `(ng, *spatial)` ndarray on `EigenvalueSolver[Carrier=np.ndarray]`; the trace is smuggled via `SNSolver._phi_of_trace`. `scattering_order` (a truncation order) lives on the solver, not the hub.
- `is_same_phase_space` is NOT a content identity (see lessons L-041).

**Why:** the consumers campaign opens on this census; a re-census that starts from these four claims saves the discovery pass.
**How to apply:** on any "Problem → Solution", `SNProblem`, `Solution.mesh`, or hub-ownership brief, read the memo's §1.1 table and §3 hatch table first, then re-run `git log --since=2026-09-08 -- orpheus/sn/mesh orpheus/sn/solver.py orpheus/sn/solution.py` to see what moved.
