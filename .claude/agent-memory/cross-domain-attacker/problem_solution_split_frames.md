---
name: problem-solution-split-frames
description: The SNProblem → Solution split (CS4c §22.5) — a posed discrete problem is a point in a parameter space determining (V,G,K,A,F); a Solution is a point in a NON-singleton fiber; the split's boundary runs through SNSolver (12/6/2), not between mesh and Solution.
metadata:
  type: project
---

Attack of 2026-09-08 on the user-ruled Problem → Solution concept (plan
`.claude/plans/cs4c_binding_design.md` §22.5 + ADDENDUM, §22.6 F2, §26 R-c1). Full memo
(untracked): `scratch/_consumers/attacker_problem_solution_frames.md`. Re-ground every
`file:line` before acting — git is the authority.

**Why:** the consumers campaign opens on this split; it renames the graph's #2 node
(`SNMesh`, `[M]` Nexus degree **1016**, graph @ `82c0d441`, stale-flagged) and re-homes every
consumer.

**How to apply:** fire the four rulings below on any Problem/Solution/hub/save-state question in
this family (`SNProblem`, `DiffusionProblem`, `CPProblem`, `MoCProblem`, `HomogeneousProblem`).

---

## R1 — the native object, and what a Solution is

A posed discrete problem is a point `p` in `Geometry × Quadrature × Scheme × Closure ×
Materials × BoundaryLaws × Truncations` together with the assignment
`p ⟼ (V(p), G(p), K(p), A(p), F(p))` — a Hilbert space with Gram and cone, plus a pencil.
`[M]` the pencil has ONE construction site, `build_within_group_system(sn_mesh, mat_xs,
scattering_order)` (`sn/coupled_system.py:456`).

The metric needs no design: `[M]` after 6.2c a space's metric is inside its identity
(`numerics/space.py:328-355` structural `__eq__`), so it is a CONSEQUENCE of the generating data.

A Solution has **three** parts, the tree types two: a REPRESENTATIVE of the solution SET
(⛔ NOT "an element of the cone" — DD is not positivity-preserving), the ORBIT
(`IterationRecord`), and the GAUGE that picked the representative. `[M]` the coset half is done
right (`LossKernelGauge` on the hub with a MEASURED σ-independence argument,
`sn/mesh/augmented_mesh.py:1041-1072`; `gauge_correction` on the history with a
"None means not measured" discipline, `sn/solution.py:181-214`); the RAY half is not
(`homogeneous/solver.py:397` applies `phi * 100/rate` and records nothing).

## R2 — ⭐ the plan's discriminating rule INVERTS on splittings; re-phrase it on the solution SET

The chartered rule ("changes the OPERATOR ⟹ Problem; changes only HOW it is inverted ⟹
Solution") classifies by *what the code constructs*, and a splitting manufactures operator
OBJECTS without moving `A`. Replace with:

> **P-clause** — Problem-side iff changing it moves `{ψ : Aψ = q}` (i.e. moves `A` or `V`).
> **S-clause** — Solution-side iff the set is fixed and only the PATH to it changes.

`[M]` it inverts on 3 of 12 shipped straddlers: `inner_schedule` (constructs a
`ScheduledInvertibleOperator` yet `solver.py:1424-1428` records the converged `k_eff` shift as
`~inner_tol` — same fixed point), the SI splitting, and — the other way — the angular closure
(constructs nothing at solve time, yet deletes the redistribution term ⟹ different `A`).

The acceptance gate falls out: per Solution-side coordinate, solve one Problem at two values and
assert the limits agree to a tolerance that SHRINKS with the tolerances.

## R3 — ⭐⭐ the carve cuts THROUGH `SNSolver`, not between mesh and Solution — 12 / 6 / 2

`[M]` sorting every attribute `SNSolver.__init__` sets (`sn/solver.py:1418-1589`):
**12 PROBLEM** (`sn_mesh`, `quad`, `scattering_order` — clamped at :1473, `ng`, `mat_xs` :1454,
`weight_norm` :1476, `volume`, `scattering_op` :1527, `n2n_op` :1534, `fission_op` :1545,
`geom_cache` :1578, `coll_cache` :1585) / **6 STRATEGY** (`inner_solver`, `inner_schedule`,
`keff_tol`, `flux_tol`, `max_inner`, `inner_tol`) / **2 SOLUTION-IN-PROGRESS** (`_inner`,
`inner_records`). 60 % of the solver is Problem-side.

⟹ the campaign owes **three** types, not two: `SNProblem`, `SolveStrategy`, `Solution`. The
fiber is `[M]` non-singleton (7 free kwargs on `solve_sn`, `solver.py:2347-2359`), so no
canonical section exists (L-022(b)); `[M]` a returned `Solution` records NEITHER coordinate —
`SolutionBase` is `(angular_flux, scalar_flux, mesh, keff, history, radial_characteristic)`
(`solution.py:421-426`) and `IterationHistory` is `(record, keff_history, balance_defect,
gauge_correction)` (:231-234).

## R4 — the identity gradient runs BACKWARDS (L-019 inverted), and that is the save state's blocker

The plan's *"its identity is the identity of everything it induces"* is today **false in the only
checkable direction**. `[M]` everything the hub induces has CONTENT identity
(`FunctionSpace.__eq__` structural since 6.1); `[M]` `SNMesh`/`MaterialMesh` define **no**
`__eq__`/`__hash__`/`__getstate__`/`__reduce__` at all — identity is `is`, plus one hand-written
5-clause predicate `is_same_phase_space` (`augmented_mesh.py:545-597`).

⟹ **derive the Problem's `__eq__` from its induced objects** rather than hand-writing it — the
claim made operational, reusing step 6's landed structural identity.

`[M]` the shipped small Problem cannot answer either: `HomogeneousProblem` is
`@dataclass(frozen=True)` over `mixture: Mixture` (`homogeneous/solver.py:162, 215`) and
`Mixture` is a NON-frozen dataclass of ndarrays + `list[csr_matrix]` (`data/macro_xs/mixture.py:49,
86-94`) ⟹ `[R]` `__eq__` raises `ValueError` (ambiguous array truth) and `__hash__` raises
`TypeError`. **Fix that class first** — it is the model the coda says `SNProblem` follows.

## The four highest-value measured findings (each is a fail-able test)

1. ⭐ **`scattering_order` is homeless and the pairing guard cannot see it.** `[M]` 0 occurrences
   under `orpheus/sn/mesh/`; clamped inside `SNSolver.__init__` (`solver.py:1469-1473`);
   `is_same_phase_space` compares mesh/quad/materials/scheme-TYPE only. ⟹ a P0 forward and a P3
   adjoint compare EQUAL, and `Solution.condense(adjoint=…)` (`solution.py:1045`) proceeds —
   pairing a P0 flux with a P3 importance. Its single source exists but on the wrong side:
   `[M]` `ScatteringOperator.legendre_order` IS the solve's `scattering_order`
   (`transport/operators/transfer.py:551-560`), and `S` is cached on the SOLVER (:1527).
2. **The Solution writes σ-dependent state onto the Problem.** `[M]` 4 stamp sites
   (`solver.py:1589`, `:1633`, `loss_representation/__init__.py:3970`, `:3943`) and
   `_ensure_coll_cache` (`:3946-3971`) NEVER validates σ. `_pole_mirror_cache` is legitimately
   Problem-side (σ-free); `_coll_cache` is not.
3. **A Problem→Problem morphism returns only its codomain.** `[M]`
   `Solution.homogenize -> MaterialMesh` (`solution.py:737-742`), `condense -> dict[int, Mixture]`
   (:948-954) — neither returns the `(R, P)` frame pair the producer derived (:753-757). A coarse
   solution cannot be prolonged ⟹ a two-level scheme over these arrows is unspellable.
   (The user's own `feedback_lossy_return_type_is_the_root_cause`, at the morphism tier.)
4. **`mat_xs` minted 6× independently.** `[M]` `material_xs_field()` is an uncached METHOD
   (`transport/mesh/material_mesh.py:556-569`) with 6 production call sites — Problem-side data
   re-derived per consumer (Smell #16 shape 2).

## Rulings on the brief's named straddlers (structural reason, not preference)

* **windowed sweep** = Solution (a representation ISO conjugating `A`), **conditioned** on
  `L_window ≥ scattering_order`; the space stays Problem-side. ⚠ `[M]` the sweep's return
  re-derives `L` from the buffer shape (`sn/operators/streaming.py:1006-1011`) — a second
  spelling of the truncation order.
* **the SI splitting** = Solution; `WithinGroupSystem` (`coupled_system.py:335-382`) FUSES it
  with the equation (4 fields = `loss` + `space` | `implicit_operator` + `explicit_gains`), and
  its docstring's invariance claim (:340-341) is `[M]` refuted by `_select_si_splitting`
  (`solver.py:1149-1177`), which returns a different `implicit_operator` per schedule.
* **the angular closure** = **Problem** (it supplies the curvilinear redistribution ⟹ part of L).
  `[M]` two live sites disagree: `is_same_phase_space` EXCLUDES it in writing (:571-577) while
  `geometry_cache_for(sn_mesh, sn_mesh.angular_closure)` KEYS on it (`solver.py:1578-1580`).
  The exclusion's stated reason (fields stay contractible) is about LAYOUT — strictly weaker than
  same-problem. ⟹ the carve owes TWO predicates: `==` (identity) and `layout_compatible`
  (contractibility, 3 consumers: `solution.py:634, 917, 1045`).
* **fission normalisation** = Solution-side GAUGE on a Problem-side RAY (Krein–Rutman).
  ⛔ NOT a torsor (canonical zero + physical superposition ⟹ vector space + cone predicate).
* **the adjoint `.H`** = **NOT a different Problem — a Solution-side ROLE.** `[M]`
  `solve_sn_adjoint` builds the same hub and the same operators and daggers at the call site
  (`solver.py:2889-2897`). `†=G⁻¹AᵀG` is determined by `(A,G)`, both Problem-side ⟹ `A†` is
  derived like `A⁻¹`. **Do not mint an `AdjointProblem`.**
* **`homogenize`** = a Problem MORPHISM parameterized by a Solution (the solution supplies only
  the TEST WEIGHT; `[M]` the `adjoint=` parameter makes this explicit — `solution.py:791-793`).
  ⟹ it belongs on the Problem: `problem.homogenized(coarse, weight=sol[, adjoint])`.

## Cross-method

`MaterialMesh` IS the Problem's data half already (`_init_data` = the method-agnostic block);
`SNProblem`/`DiffusionProblem` are its method-augmented completions — `[M]`
`DiffusionMesh._init_core` takes 4 params, `SNMesh._init_core` 7, and the 3 extra ARE the angular
discretization ⟹ diffusion is `MethodDiscretization = ()`, the degenerate member, so unifying
DELETES content (L-020's criterion). **The infinite medium is NOT in that family**: `[M]` its pose
has `generator=None` on the spatial axis (`homogeneous/solver.py:153-159`), no chart, no BCs, and
`L ≡ 0` is EMPTY content not trivial content; the family is a coproduct `P_meshed ⊔ P_∞` whose
shared factor is `EnergyAxis.from_materials` (already the one energy-arm rule, and it survived
the coda's `from_materials` retirement deliberately).

Borrow the **digest**, not MC's architecture: `[M]` the tree's only serialization primitive is
`_canonicalize` + `_hash_params` (SHA-256 over canonical JSON, handles ndarray/Mapping/set) at
`derivations/continuous/sood_registry/cache.py:154-195` — a `SolveStrategy` hash in all but name,
sitting in the wrong tier.

## Refuted, with the QUESTION each was refuted for

* **Sheaf** — refuted FOR the Problem→Solution split: no topology on the parameter base, no
  restriction, no gluing. FACT: it fires over the SPATIAL domain (MoC's trajectory bundle), so it
  is live for domain decomposition.
* **Functor `P → Operators`** — refuted as a LAW (reduce-discrete ≠ discretize-reduce;
  homogenization is an approximation), retained as a DIAGNOSTIC. FACT: the coarsening is a natural
  transformation on the **rate functional** (`solution.py:759-761`), which IS the adjunction
  frame's first test.
* **Galois connection** — refuted: Problems are not ordered (space- and energy-coarsening act on
  orthogonal factors, no lattice). The real structure is `R ⊣ P` on the two Problems' SPACES.
* **Category theory / Grothendieck fibration** — low-signal: the concrete win (indexed family,
  projection, non-singleton fiber) is fully captured by the parameter-space + fiber frame, which
  produced a fail-able test.
* **State machine / IFS** — already realized (`fixed_point_step`, `IterationRecord`); re-proposing
  it re-derives a landed design.
* **Torsor** — refuted by the two-question test; the ray/coset is a SUBSET of `V` and the gauge is
  a scalar on the record.
* **MPO / tensor networks** — no bond-dimension knob in a 7-factor finite product.
