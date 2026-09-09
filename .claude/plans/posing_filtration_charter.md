# The posing filtration — problem posing is a monotone refinement, and every guard sits at its earliest decidable point

**Status: RATIFIED 2026-08-25** (user proposal, 2026-08-24 → three adversarial
rounds → surviving form ratified with its amendments). This file is the
**charter AND the derivation record** for the architecture that governs the
operator/mesh un-weld arc and re-shapes the landing surface of CS1.5′ and CS2.
It supersedes the *shape* of the CS1.5 Medium charter (campaign plan §2.5 and
`cs15_medium_unweld_design.md` — no `Medium` class exists in the ratified
form) while preserving that charter's surviving physics objectives (§8 maps
them). The **step decomposition for the arc is deliberately NOT here** — steps
are the next design round's output, designed against this charter, with the
user steering (surgical posture stands).

**Why this file is long.** It records the *reasoning*, not only the
conclusions. The archivist will later write the Sphinx theory page from it,
and a fresh session must be able to re-derive every guard placement from the
principle rather than trust a table. Per the user's instruction: for every
discharge point we state why that place is RIGHT — the positive reason is
near-singular — and we do not enumerate why other places are wrong (there are
many wrong places for many reasons; there is usually one right one for one
reason).

---

## 1. The concept this architecture serves

Everything in the problem layer is a map on spaces (user, 2026-08-24):

- a **field** is a map *space → values* — cross-sections map space to a
  cross-section value, flux maps space to a flux value;
- an **operator** is a map *space → space*;
- therefore the organizing object of the codebase is the **space with labeled
  axes**, and nothing fundamental requires a mesh object.

`SNMesh`'s diagnosis under this lens: a god object produced by **identity
scarcity** (things accreted to it because it was the only object with strong
identity while `FunctionSpace.__eq__` was weak — campaign plan §2.5's §8
mechanism, repaired by CS4b's axes-content identity) and by **naming
preceding machinery** (user: "we tried to give everything mathematical names
but their roles was more naming than machinery and a lot of things were hand
rolled"). The named objects are now *becoming* machinery and absorbing the
mesh's roles; `SNMesh` trends toward "a save state" — a pure data aggregate
with no machinery, which is a legitimate terminal role (organization +
persistence, #406) and NOT a god object, precisely because it would carry no
behavior. Whether that aggregate survives as a class is **an open user fork**
(ruled OPEN 2026-08-24); this charter resolves the *admissibility-placement*
half of the old "funnel" question (§5) and leaves the *container* half open.

**Scoping ruling (user, 2026-08-24): the arc builds the CONSUMED objects
first — space, fields, operators.** The consumers — solvers, strategies,
traversal (`dag_walk`, schedules, cumprod machinery) — are deferred to a later
arc whose consumption "will drastically change to have a lazy realization."
The filtration gives that later arc its criterion for free (§3, "lazy
realization").

## 2. The ontology: posing is a filtration

**The chain of problem-posing commitments is a filtration: a monotone
refinement of partitions of phase space. Each stage commits exactly one
refinement, and each commitment is only *statable* on the partition the
previous stage built.** That last clause is why the order is principled
rather than arbitrary: you cannot assign materials finer than regions exist,
cannot mesh what has no extent, cannot pick DOFs before a method says what a
DOF is. The linearization is not a convention; it is the dependency structure
of statability.

The stages, with the partition each commits:

| stage | declares / commits | partition refined to | measure that appears |
|---|---|---|---|
| **Materials** | `{id → Mixture}`, raw grid provenance per material | per-MATERIAL; energy per-GROUP (the library's partition of the energy axis) | the energy discrete measure's *data* (formal construction is method-time — §4.1) |
| **Geometry** (overlay) | region partition; deck identifications; boundary data; region→material assignment | per-REGION | none new (identifications *define* the quotient domain) |
| **Mesh** | cell partition conforming to regions | per-CELL | the spatial measure (volumes) |
| **State fields** | state data (T, ρ, burnup, …) on cells | none (data on the cell partition) | none |
| **Method head(s)** | the method + its discretization scheme | per-DOF — the **terminal refinement** of every axis | angular measure (if the method has one); spatial DOF measure (nodal/modal); energy possibly *refined* (unionization) |

Three structural theorems fall out of the ontology — each was independently
forced during the adversarial rounds and then recognized as a consequence,
which is the evidence the ontology is native rather than imposed:

**(T1) All axes formally construct at the method** (user's counter-9,
sharpened by the MC-unionization example — a Monte-Carlo head with a
unionized energy grid *refines* the energy axis, so even the energy axis is
not formally constructible before the method). The filtration derives this:
the method commits the last refinement of every axis, so only there is the
full axis datum present. Constructing an axis earlier would mint an object a
later commitment could invalidate — a mutable or twice-derived space, both
worse. Earlier stages **accumulate measures and data; axes resolve at the
head** (no half-axis object exists — `[M]` `Axis` requires `kind` at every
mint, `orpheus/numerics/axis.py:103-141`: "the basis character is physics and
must be spelled at every mint").

**(T2) The leak principle is the adaptedness axiom.** The tree's already-
ratified law — *"the mint consults exactly its defining data … a spectator
with `eg=None` must not flip the axis identity of a problem it does not
touch"* (`[M]` `orpheus/transport/mesh/material_mesh.py` `bulk_space`
docstring, ~:379-410) — is precisely the statement "every stage-k object is
measurable with respect to stage k's partition." The architecture's central
law was ratified before the architecture was named. Consequence used in §5:
declared-but-unassigned materials are **inert by construction** — no
spectator-warning machinery is needed, ever, because the method-time mint
reads reachable materials only.

**(T3) Coarsening is the same chain walked backward.** Condensation and
homogenization are conditional expectations onto coarser stages of the same
filtration. The landed machinery already implements the projections: the
collapse pair's `retraction` with its Parseval divisor IS the projection onto
a coarser partition (CS4b S6, `orpheus/numerics/frame.py` `_collapse_pair`),
and the condensation ruling "fractional-overlap downsampling" is the measure
pushforward between partitions. Posing walks forward; coarsening walks
backward; one structure. Corollary: **the filtration mints a space at every
stage** (each stage's partition with its measure is a space at that
resolution); the method's space is merely the terminal one, and the
inter-stage maps are the retraction/section family.

**Data spaces vs solution spaces — the precision T1 needs.** T1 governs the
*solution* (flux/DOF) space. **Data lives at the stage where it is declared**
(T2), on that stage's space: cross-sections are cell-wise data on the
cell×group space, constructible at the mesh stage with no method in sight —
which is why `MaterialMesh.bulk_space` legitimately exists today. A method
head whose solution space differs **re-poses** data onto its pose; the
measured precedent is S7's EE-1 (`[M]` the homogeneous solver re-poses XS
fields onto its pose — `orpheus/homogeneous/solver.py`, commit `2e054bfc` —
forced by G2.5's measure-authority mutation gate). Without this distinction
the charter would forbid an XS field before a method exists, which is both
false to the tree and wrong.

**Symmetry's exact role** (resolving the round-2/3 exchange): the group acts
on phase space; its orbit partition **bounds how coarse an admissible pose
may be**; among admissible refinements the good ones respect orbit structure.
So **refinement is the flow; symmetry is the admissibility bound and quality
criterion on refinements** — the user's monotone-symmetry-breaking argument
(each stage's object bounds the maximum theoretical symmetry; monotone
NON-INCREASING, not strictly decreasing) is the constraint surface of the
filtration, not its engine. This division also covers the case symmetry
alone cannot order: a generic heterogeneous problem's group goes trivial at
the overlay, yet mesh and method stages still refine meaningfully. The
per-stage symmetry-group *machinery* is aspirational (the ingredients ship:
`RigidMotion` in `orpheus/geometry/` — placed there for exactly this reason
(user) — `SubgroupOfO3`/invariance lattice #152, invariant tests #166, and
the constructive kernel measured in §5 guard 4); declared-vs-computed groups
is an open design choice, and the "mesher scores symmetry preservation"
criterion is the user's recorded aspiration. Until built, the symmetry story
justifies; it does not yet execute.

## 3. What each stage is, and what exists today

**Library vs Materials.** The library is all data you own; **`Materials` is a
declaration for THIS problem** — `{id → Mixture}` with each material's raw
grid provenance. The distinction is load-bearing (round-2 attack 9): it is
what makes stage-1 data lawful under T2. Naming ruled 2026-08-25: the word
"Medium" is **not** used for this object — "admission-previewed material
declaration with raw grid provenance" *is* a material list, and `Materials`
is its honest name. The word *medium* survives in exactly one place, where it
is physically exact: **`InfiniteMedium(mixture)`** — a single material
filling all space. Placement: the **problem layer**, never a method on
`Mixture` (round-2 attack 12, conceded: `mixture.as_infinite_medium` would
make `orpheus.data` import the problem layer — dependency inversion).
(Placement refined 2026-08-25, R23: the concrete home is
`orpheus/homogeneous/` — the aggregate's own method-family package; the
never-on-Mixture half is unchanged.)

- Two rulings bind Materials' content (round-3, both fully accepted):
  **(R-mint)** *the mint is the law; admission is a preview.* The method-time
  mint re-reads the raw data (single source); the Materials-time coherence
  guard (the XD-4 three-outcome shape: agree / no grid / refuse) exists for
  fail-early ergonomics ONLY, and its correctness criterion is that its
  refusals are **implied by** the mint's — a preview may never disagree with
  the law, in either direction. **(R-raw)** *no early collapse.* Because a
  head may unionize, Materials carries each material's **full grid
  provenance**, never a pre-reconciled common grid — collapsing early is the
  lossy-return-type defect at the root, and it would throw away exactly what
  the MC head needs. Consequence: two heads over one Materials may
  legitimately realize **different** energy axes.
- ✅ **Shape + home RULED 2026-08-25, and LANDED the same day @
  `c6964299` (Phase A item 5 — annotations widened at `5c64c78f`;
  `MaterialMesh` parses at the boundary via `Materials.of`, guard 2
  discharges through `restrict()`, `is_same_phase_space` moved to
  per-mixture identity):** `Materials` lives at
  **`orpheus/data/materials.py`** — the
  incumbent `data/materials/` property-correlation package (h2o/matpro
  thermophysics) renames to **`data/material_properties/`** ("long
  overdue"; `[M]` 6 consumers, all in the undecided-fate
  TH/kinetics/fuel zone — mechanical sweep). The class is a frozen
  `eq=False` (identity — content identity joins the CS2 family later,
  #403 the precedent) wrapper over `Mapping[int, Mixture]` re-bound to a
  `MappingProxyType`; admission refuses ONLY the empty declaration;
  `restrict(ids)` is guard 2's mechanism (assigned-but-undeclared
  refuses in the declaration's vocabulary). **No `ng` property, no
  energy-axis preview** — both withdrawn at the exchange: the preview
  had zero consumers (and a preview that does not exist cannot disagree
  with the law — R7's cleanest form), and a scalar `ng` on the
  declaration is the wrong object entirely (see the taxonomy bullet).
  Zero imports beyond `data`-internal. The Library-vs-Materials
  distinction survives as class semantics, not a package boundary.
- ⭐ **The data-kind taxonomy (user, 2026-08-25 — recorded for the
  future data overhaul, priced §9):** macro data arises **at least three
  ways** — (1) nuclide concentrations × micro library of **GENDF**
  class (multigroup); (2) the same with **PENDF/ACE** class
  (pointwise/continuous); (3) **collapsed** data from a PRIOR solve
  (condensation/homogenization — the cross-problem provenance loop).
  PENDF-class data feeds Monte Carlo **exclusively**; GENDF-class and
  collapsed data must produce ONE final object with the same available
  data, consumable by ANY method **including multigroup Monte Carlo**.
  Each kind carries its OWN consistency check (a GENDF-consistency
  check — this is what supersedes the scalar-`ng` idea — a PENDF
  variant, a collapsed variant); the check family lands with the data
  overhaul, not this arc. Today's mesh-time `InconsistentMaterialsError`
  stays untouched this arc (downstream, working; superseded by the
  per-kind family later). #395's regime scoping is this taxonomy's
  energy-grid shadow.
- ⭐ **Lazy resolution of concrete properties (user, 2026-08-25):**
  resolution of concrete numbers (`ng`, …) is **delayed as needed** —
  each consuming stage resolves what it consumes (T2 applied to
  data-reads; §3's lazy-realization criterion applied to DATA). The
  first heavy consumer of concrete energy-structure numbers is
  **Campaign 2's partitioning machinery** (`GeneralizedEigenPencil`,
  resolvent, partitioning), where — user's stated direction, hedged as
  "I think" — a significant chunk of the **spectral radius** becomes
  predictable and the general objective is to LOWER it.
- The infinite path: `InfiniteMedium(mixture)` is a **complete problem**
  (the slowing-down problem — the energy sub-algebra posed alone; not
  transport at degenerate geometry). It runs the SAME generative primitives
  (axis mint, kernel binding) at trivial stage values; ergonomic entries are
  sugar over shared primitives, which is what dissolves the twin-path hazard
  (round-1 attack 1, withdrawn on exactly this condition — it is a
  condition, not a given: **zero downstream `if infinite:` arms; if one
  appears, the architecture has failed its own thesis**). It is also the
  in-solver evaluation tool (per-material spectra, k∞, condensation
  weighting) — every transport method holds a Materials and can spin up the
  cheap head per material. Note the cycle this creates across problems: a
  condensed Materials' provenance is a *solve* on a prior Materials — a DAG
  per problem, a loop across problems; the #406 save-state story should
  record that provenance.
  ⭐ **REFRAMED 2026-08-25 (user; main-agent concurred): `InfiniteMedium`
  is the homogeneous family's AGGREGATE — `SNMesh`'s analog on the
  infinite path** (organization + shared objects, the save-space role).
  It lives in `orpheus/homogeneous/` and takes a `mixture` DIRECTLY,
  never a `Materials` — clean because the shared law
  `EnergyAxis.from_materials` accepts `Iterable[Mixture]`, so both paths
  feed ONE law and the infinite path never touches a declaration map
  (`Materials` is the heterogeneous path's declaration; the mixture IS
  the infinite path's). This is the container fork R3's **first data
  point**: the infinite path keeps an aggregate, by design. Its
  absorption of the degenerate `MaterialMesh.from_materials` carrier is
  governed by the twin-path condition above (same primitives, zero
  `if infinite:` arms) — the tree already leans this way: `[M]`
  `_pose_space`'s docstring (homogeneous/solver.py) demotes the carrier
  to "supplies cross sections, not the posing". Design timing follows
  the O-4 sequencing logic: aggregates AFTER the operator shape
  crystallizes.

  ✅ REMEDIED 2026-09-08 by the CS4c coda — C1 `5caad3d6` / C2
  `39e7f32f`. The aggregate ships as **`HomogeneousProblem`** (ruling
  R-c1's name), in `orpheus/homogeneous/solver.py`, frozen, taking one
  `Mixture` directly. "Absorption of the carrier" resolved by
  RETIREMENT, not absorption: the hub mints the material and
  cross-section fields itself, and `MaterialMesh.from_materials` is
  deleted — zero `if infinite:` arms anywhere, and the twin-path
  condition is met the strong way (there is no second path left to be a
  twin of). ⚠ The `[M]` four lines up is now stale as a description:
  `_pose_space`'s docstring no longer demotes a carrier, it records that
  nothing on the path builds one, and names a genuine unit-cell
  `Mesh1D`'s `bulk_space` as an `==` REFERENCE kept honest by the
  identity-bridge gate.

**Geometry (the overlay).** Declares: the region partition; the
**deck identifications**; the genuine **boundary data**; and the
**region → material assignment** (direction matters: region→material is the
total function; material→regions is its multivalued inverse). The
unification ratified in round 2/3: **reflective, periodic, and rotational
"boundary conditions" are not boundary conditions — they are stage-2
geometry.** The domain is a quotient; the deck group is part of the
problem's symmetry group (feeding §2's bound a computable contribution); and
only vacuum / albedo / prescribed-inflow remain as boundary *data* — declared
here because they describe the domain's physical continuation
(method-independent), realized only at a head (no method ⇒ nothing to
resolve; the infinite problem has no boundary and no method-head boundary
work, consistently). What ships: `StructuredGeometry`
(`orpheus/geometry/structured_geometry.py`) IS the 1-D overlay — interface
positions + per-segment materials. What does not: **the d≥2 overlay object**
(today 2-D goes straight to a raw per-cell `mat_map` with no region concept)
— priced work the chain demands, an attack the user redirected onto the
current architecture rather than the proposal (round-1 attack 5).

**Mesh.** Commits the cell partition; the **spatial measure** (volumes)
appears; the spatial *axis* does not (its kind — nodal/modal — is a scheme
commitment; §2 T1). `mat_map` becomes **derived**: region map ∘ (cell →
region), well-defined exactly because of the conformity guard (§5 guard 3).
`MaterialMesh` reads correctly as this stage's object — Materials × mesh
through the pullback.

**State fields** (round-2 attack 13, fully conceded — the multiphysics
slot, empty today). Multiphysics state is a **field** — a map space→(T, ρ,
…) — on the cell partition. Cross-section resolution then becomes a lazy
pointwise map `(nuclides, library) × state → XS`, making resolved XS
*derived fields*. The placement's singular reason: state is spatial data, so
it needs the spatial partition (after mesh), and operators consume resolved
XS, so it precedes operator binding. Named now, one sentence, so that
statefulness never has to be retrofitted into the root: `Mixture` stays the
parametrization (nuclide mixture × library); the *binding of state to
space* is a chain stage.

**Method head(s).** The terminal refinement, per head: the scheme fixes the
spatial axis's kind (and any moment tail); the angular measure and axis
appear **if the method has one** (SN/PN yes; CP pre-integrates — no angular
axis ever; MC samples continuously — tally grids are a different overlay;
round-1 attack 2, conceded and folded in: the chain is a universal prefix +
per-method measure stacks); energy is formally constructed and possibly
refined (unionization). ALL axes and the solution space construct here (T1).
All remaining obligations discharge here, exact-or-refuse (§5). **Multiple
heads share one prefix** — the shipped witness is DSA: SN's head builds a
diffusion low-order system over the same meshed problem. Heads must not
mutate the prefix (sharing discipline), and per-head axes may differ
(R-raw). Cheap re-instantiable heads over a fixed prefix is also exactly
what convergence studies and the future lazy-realization solver arc want:
**realize each operator at the coarsest stage where it is measurable** — the
filtration hands the lazy arc its criterion.

## 4. The guards — where each is discharged, and why that place is right

**The schema (the near-singular reason, stated once):** a guard's home is
the **earliest stage at which its predicate is decidable** — the stage where
its last-arriving operand exists. Earlier, the predicate cannot be stated;
later, an inadmissible object has already existed. Each row below names the
last-arriving operand; that is the whole justification, per the schema.

| # | guard | discharged at | last-arriving operand | status in tree |
|---|---|---|---|---|
| 1 | energy-grid coherence **preview** (agree / no-grid / refuse; regime-scoped per #395) | Materials admission | the declared mixtures' grids | XD-4 amendment recorded (campaign plan §2.5); **preview only — implied-by-the-mint is its correctness criterion (R-mint)**. ⭐ Sharpened 2026-08-25 (R21/R22): **no preview machinery is BUILT** — admission refuses only assignment-independent trivia (the empty declaration; a cross-material refusal at admission would let a spectator flip admissibility, violating R11/T2); coherence lives wholly in the mint; the future check is per-DATA-KIND (GENDF / PENDF / collapsed — R22), landing with the data overhaul; today's mesh-time `InconsistentMaterialsError` stays untouched this arc |
| 2 | assigned-but-undeclared material → refuse | overlay construction | the assignment (declaration already exists) | new with the chain (user-spotted, 2026-08-25) |
| 3 | mesh conforms to regions (pullback well-defined) | mesh construction | the cells | chartered in the old CS1.5 design (§6c witness: hand-built non-conforming `Mesh1D` refused with a region-naming reason) — survives verbatim |
| 4 | deck motion is a symmetry of the discrete rule | head (BC realization) | the quadrature | **SHIPS, exact-or-refuse**: `[M]` `Quadrature.ordinate_permutation` (`directional.py:337`) — every image matches a node (no bare argmin, ERR-074), bijection (ERR-073), equal weights (ERR-042); `None` = "not a symmetry of this rule", caller refuses in the law's vocabulary; consumed by `_deck_kernel` (`realizer.py:452`), the ONE body every deck law realizes through. **No interpolation arm exists** — no silent approximation ships |
| 5 | quadrature × geometry admissibility | head | the quadrature | **the live gap**: `[M]` #398 — `SNMesh(slab_1D, lebedev)` constructs while `tests/numerics/test_registry.py:1210` asserts inadmissibility. The chain names its discharge point; the arc lands it there |
| 6 | scheme × geometry capability | head | the scheme | exists today as `supports()` machinery on the loss-representation family; re-homes with the head |
| 7 | boundary-data realization (vacuum/albedo/inflow onto trace spaces) | head | the method's trace machinery | ships (`SNBoundaryRealizer` takes the `SNMethodSpace` bundle — already one level un-welded); the assembled table's *storage* is the container question, OPEN |
| 8 | state-field × mesh conformity | state stage | the state field | future (slot named, empty) |
| 9 | energy formal resolution incl. unionization | head, per head | the method | future (R-raw makes it possible; #395's unionization arm) |
| — | spectator inertness | *nowhere — a theorem, not a guard* | — | T2: the mint reads reachable materials; a declared-unassigned material is weightless by construction. **Do not build warning machinery for it** |

**The funnel fork, resolved-in-half.** The old question "is the space mint
the unique construction funnel?" dissolves into: **admissibility is
distributed — the chain is the funnel, staged** (each guard at its earliest
decidable point, rows above). What remains genuinely open is the
**container**: whether a stage-4 aggregate class (`SNDiscretization`, or
`SNMesh` demoted to it) survives for organization + persistence (#406: no
save/restore story exists; a single dumpable object is a real argument FOR)
and as the home of caches like the realized-BC table. Ruled OPEN by the user
(2026-08-24): "We don't know yet if SNMesh or SNDiscretization (or whatever
name we give it) will dissolve." #398 is the admissibility witness whenever
that fork is ruled; the latent `BoundaryOperator`-factory option (like the
other operator factories) is on the table.

## 5. What this re-reads in the existing evidence base

**The operator mesh-independence census**
(`scratch/operator_mesh_independence_census.md`, `[M]` @ `55bb47b9` — 13
operators already mesh-free in two shapes, 7 one-step un-weldable, three
chokepoints) re-reads under the chain as follows:

- **`pole_angular_closure`'s `cls(sn_mesh)` contract** → head-side binding
  data for L; re-contract to its actual needs `(quad, reduced, coord,
  levels, ng)` — all mesh-free-available; the gap was the contract, not the
  data.
- **The L-binding bundle** (`_streaming_axes` per-axis stencils; the 1-D /
  curvilinear `dag_walk` visit iterators) → the stencils are head-side
  operator-binding data (in scope); the **traversal is strategy** (deferred
  arc) — the seam cuts through `LossRepresentation`, which holds `mesh`
  whole for both halves. The arc frees its *construction* (bundle =
  stencils + closure + spaces) and leaves traversal as the one solve-time
  handoff — which is the shape lazy realization wants anyway. The 2-D
  `SweepDependencyGraph.for_shape` (pure shape) is the proof traversal
  needs no mesh when its turn comes.
- **The realized-BC table** (`sn_mesh.bc`, assembled at
  `augmented_mesh.py:380`) → ⛔ REFUTED 2026-08-25 (user): the original
  text here read "declarations move to the overlay (already on `Axis1D.bc`
  today — one stage too late under the chain)". They do NOT move — the
  declaration already lives at the geometry stage: `[M]`
  `StructuredGeometry` carries the endpoint `BC` tuple
  (`orpheus/geometry/structured_geometry.py:215-227`) and `Region.mat_id`
  the assignment (`:146-161`); `Axis1D.bc`
  (`orpheus/transport/mesh/axis.py:172/:210`) is a LATER stage carrying
  the commitment forward, which is chain-consistent, not late. The work is
  realization discipline only: head-side (guard 7); the *table object's
  home* rides the container fork.

**The `MaterialXSField` verdict** (adversarial round, 2026-08-25 — main
agent's verdict at the user's invitation; unopposed and consistent with the
chain, ratify formally at the arc design): the class is `MaterialMesh`'s XS
**facade** — `[M]` `from_mesh` is `cls(materials=mesh.materials, mesh=mesh)`
(`material_xs_field.py:194-209`, adds no data); its 1155 lines decompose as
Materials content + the expansion machine (wrapping the already-free
`assemble_cell_xs`) + typed field mints + coarsening projections
(`project_through`/`_bilinear`, `:253/:317`) + **eight `apply_*` operator
kernels (~400 lines, `:741-1021`)** whose docstring confesses the mechanism
("Encapsulates the per-material loop that previously lived at
`scattering.py:405`"). Dissolution map: content → Materials; expansion →
the field-minting path at its stage (data spaces, §2); mints → honest
`CrossSectionField`s (which stay); coarsening → Materials/Mixture morphisms;
apply arms → the bound operators (CS4c's chartered "S → kernel shell").
Blast radius `[M]` 18 production + 32 test files — §6b-scale; CP/MoC/MC/
diffusion get mechanical rewiring only (sharpening-order law).

**Shape-metadata reads**: `ng`/`spatial_shape` re-point to the space's axes
— `[M]` the axes exist (`material_mesh.py:385` mints
`of_axes(energy_axis, SpaceFactorAxis("spatial", …))`;
`augmented_mesh.py:1129` mints `of_axes(angular, *scalar.axes)`); the only
missing convenience is a public axis-by-label accessor (the lookup already
lives inside `retraction`/`section`).

## 5b. The consumables inventory — improvements identified BEFORE this charter

> ✅ **PHASE A LANDED 2026-08-25, merged @ `5c64c78f`** (ff-only,
> branch deleted, pushed): S-1 (`bdb4bfc6`), O-1 + O-5 (`76ee98ce`),
> the Materials step (`c6964299`; R20/R21 executed), consumer fix
> (`ca1bb92b`), annotation widening (`5c64c78f`). `[M]` full-tree fast
> gate **9806 passed / 0 failed** (1:00:55 serial `-O`, 227 deselected;
> +15 tests over the Campaign-1 merge population); `npx pyright
> orpheus/` 0; sphinx `-W` 0; `dead_references` 0 after both renames.
> F-2 adjudicated INTO O-3 (see its bullet). Per-item ✅ stamps below.
> ✅ **PHASE B COMPLETE 2026-08-26** — the discussion concluded (§5d), then
> B.1 (`27576937`: `AngularRedistribution` minted, six `Optional` fields +
> the fused `redist_dAw` cache retired) and B.2 (`6859ca05`: the closure
> takes `cls(angular, gram)`, its two TENSOR FACTORS; the gram carries
> `(n_mom, n_thread)`; `n_mom > 1` refuses, naming #158). ⚠ On branch
> `refactor/unweld-phase-b`, **NOT merged**; the canonical fast gate was
> IN FLIGHT at the compaction point (`scratch/_phaseb_merge_gate.log`).
> ⛔ The bug B.2 caught is the arc's own thesis: `sn_mesh.reduced` is
> `None` for multi-D, and `cls(sn_mesh)` had hidden that because Identity
> reached for `quad.N`. The neutral factors build from `(quad, coord)`
> alone — the mesh that HAS no reduced operator can still supply them.
>
> ▶ **NEXT: `.claude/plans/streaming_path_says_what_it_is.md`** — the
> successor plan (names / homes / welds / strata / the scheme mint, plus
> §5b on making `L`'s three factors first-class). ⛔ **FOUR FORKS AT ITS
> §9 ARE UNRULED**; two of them change what a phase DOES. It carries a
> ▶ RESUME STATE header — read that first.

The arc's raw work items for the consumed objects (Space / Fields /
Operators), identified during the 2026-08-24/25 census + design-surface
discussion — i.e. *before* the filtration was proposed — and preserved here
so the design round inherits the inventory and not only the architecture.
**This is an inventory, not a step order**: the design round shapes these
into steps AGAINST §2–§4 (each item's landing stage is now derivable from
the chain), with the user steering. Forks are marked; nothing here is
implicitly ruled.

### Space

- **S-1 — `FunctionSpace.axis(label)` public accessor.** ✅ LANDED
  `bdb4bfc6`: the resolution hoisted into `_axis_index` (one home; the
  collapse-pair mint routes through it, pinned fragments preserved);
  public `axis(label)` returns the tuple member itself.
- **S-2 — re-point `ng`/`spatial_shape` read-throughs to the space's
  axes.** `[M]` the axes exist on the production mints
  (`material_mesh.py:385`; `augmented_mesh.py:1129`). Known consumers:
  `MaterialXSField.ng/.spatial_shape` (`material_xs_field.py:727/:737`) —
  note these dissolve with F-1 anyway, so the re-point may land AS PART OF
  the dissolution rather than before it.
- **S-3 — fork ✅ RULED 2026-08-25 (user accepted the recommendation):**
  this arc takes minted space OBJECTS as given; the mint-as-free-function
  + axes non-Optional + name-bridge retirement stay CS2. Under the charter
  this sharpens: the solution-space mint's home is the METHOD HEAD (T1),
  so CS2's landing surface is the head.
- **S-4 — recorded, not this arc's item:** `axes: Optional` — legacy
  name-built spaces remain constructible (the collapse pair refuses them);
  completion is CS2's identity work.

### Fields

- **F-1 — the `MaterialXSField` dissolution (R13; anatomy + map in §5).**
  The arc's largest fields item. Blast radius `[M]` 18 production + 32
  test files; CP/MoC/MC/diffusion receive mechanical rewiring only
  (sharpening-order law). Receiving objects: `Materials` (content),
  stage-native field mints (expansion via `assemble_cell_xs`), the bound
  operators (arms — see O-6).
- **F-2 — the mesh-keyed moment-space mint re-points to its space-level
  spelling.** `[M]` census S4: `_space_for_mesh_and_L`
  (`transport/fields/_bases.py:687`), reached from
  `HarmonicMomentFlux.from_mesh_and_L(..., sn_mesh, ...)`
  (`streaming.py:1001` call site); the spelling
  `SphericalHarmonicSpace.from_L(L) * bulk_space` already exists.
  ⭐ ADJUDICATED AT PHASE A (2026-08-25, main agent): **deferred INTO the
  O-3 step.** `[M]` the mint's mesh read is not shape metadata — it is
  `_compose_spatial_moments(space, mesh, per_axis)` reading the mesh's
  bound SCHEME for the within-cell moment tail (`_bases.py:195-216`:
  "the tail is the scheme-owned `moment_axis` … the scheme binds at
  transport-method augmentation"), i.e. §5c's space-side induction
  exactly. A shallow re-point now would churn the precise seam the
  scheme carve redesigns; the honest landing is with the mint's inputs
  becoming the scheme's (R14/R19). The streaming call-site's mesh reach
  is legitimately transitive until then (L still binds the mesh until
  O-3 lands).
- **F-3 — `CrossSectionField` and kin STAY** (honest `(values, space)`
  fields); the dissolution mints *more* of them per channel — the
  kernel-as-operator-valued-field chain (§5).

### Operators

- **O-1 — the one-step-un-weldable seven** ✅ LANDED `76ee98ce` (+
  `ca1bb92b`): all seven bind spaces + values at construction;
  `require_member` re-keyed `mesh=` → `space=`; `march_start_cosines`
  public; B_b's outer law construction-bound; the `WindowedSweep`
  guard is space-content. ⚠ Census corrections found in execution:
  `gauss_seidel` ALSO needs the BC-derived reflective set (the S1 row
  under-listed it — signature is `(ndim, octants, reflective)` with
  `reflective_faces(sn_mesh)` public); and the §6b sets were three
  times larger than the head-clipped greps showed (plan-authoring
  surprise-log row, 2026-08-25 — enumerate UNTRUNCATED).
- **O-2 — the pole-closure re-contract**: the family's `cls(sn_mesh)`
  contract (`orpheus/sn/sweep/pole_angular_closure.py:211/:310/:1382`;
  back-reference bound at `augmented_mesh.py:394-401`) re-contracts —
  ⭐ ARGUMENTS RETHOUGHT (user, 2026-08-25): the earlier list `(quad,
  reduced, coord, levels, ng)` is redundant, not minimal. `[M]` the
  family's actual `sn_mesh.` attribute reads: `quad` ×4, `ng` ×2,
  `reduced` ×1, `radial_characteristic_levels` ×1, `coord` ×1 (plus one
  self-dispatch read). And `[M]` `Quadrature` already carries the level
  machinery (`n_levels`/`level_indices`/`level_mu`,
  `numerics/quadrature/directional.py:565-590`); `ng` is the space's
  energy axis (the S-1 accessor); a space→quadrature accessor is a live
  design option (today the angular axis mint carries only the WEIGHTS,
  not the `Quadrature` — `augmented_mesh.py` `angular_bulk_space`). Step
  design derives the minimal non-derivable set and shapes the
  re-contract for the family's ruled future as the **`AngularClosure`
  candidate member** (R15).
  ⭐ The two derivability questions are MEASURED (2026-08-25, compaction
  prep): **`radial_characteristic_levels` IS `(quad, coord)`-derivable**
  — its body is `march_start_structure_per_level(self.quad,
  self.reduced.coord)` filtered on `consumes_independent_seed`
  (`augmented_mesh.py:862-869`; the producer is a free function in
  `pole_angular_closure`); and **`reduced` is NOT quad-only-derivable**
  — it is the reduced STENCIL, minted from `(legacy Mesh1D, Quadrature)`
  at `_init_core` (`augmented_mesh.py:328/:335/:353` —
  `slab/cylindrical/spherical_streaming(mesh, quadrature)`), so it stays
  an independent operand (a mesh-free TYPE, per the census). ⟹ the
  minimal set trends to `(quad, reduced)` with `coord = reduced.coord`
  and ng from the space; the Phase B DISCUSSION rules: whether the
  closure takes `reduced` whole or only the fields it reads; the
  `AngularClosure` family shaping (R15); the `cls(sn_mesh)` +
  back-reference dispatch contract (`closure_cls(self)`,
  `augmented_mesh.py:394-401`, PR-TYPED-6.5 Phase 2.3); the
  space→quadrature accessor option; and Phase B's own BOUNDARY (what of
  the L-binding cache family — `sn/sweep/cache.py`'s
  `StreamingCoefficientCache` (`GeometryCoefficients` until 2026-08-26)/`CollisionCache` — is O-2's vs O-3's).
  Unblocks `RadialCharacteristicSeeding`, the curvilinear walks, and
  the sweep cache.
- **O-3 — the L-binding bundle — ⭐ REFINED AND ITS FORK RESOLVED (user,
  2026-08-25): see §5c.** The scheme is a stage-2 generator on the Frame
  pattern; `StreamingOperator` binds `(domain, codomain,
  DiscretizationScheme)` exactly as `ScatteringOperator` binds its frame;
  the "bundle" IS the scheme's minted package (closure + trace descriptor +
  basis kind + positivity predicate); the `LossRepresentation` family
  reorganizes into the `DiscretizationSchemeBase` family and drops
  `mesh: "SNMesh"` (`loss_representation/__init__.py:462/:980`); the
  `_streaming_axes` stencils (`augmented_mesh.py:1769`) are the evaluated
  layer of the closure's structure; traversal stays a solve-time handoff
  (deferred arc; `SweepDependencyGraph.for_shape` the mesh-free precedent)
  — and is CONTRABAND in the scheme by the §5c hard guard.
- **O-4 — B under the chain**: ⛔ the "declarations move / one stage late"
  claim is REFUTED (2026-08-25, user; `[M]` in the §5 realized-BC bullet)
  — declarations already live at the geometry stage (`StructuredGeometry`'s
  BC tuple; `Axis1D.bc` carries them forward); nothing moves. The work is
  realization discipline only, head-side through the realizer chain
  (already takes the `SNMethodSpace` bundle). **Fork deliberately unruled
  — sequencing RULED 2026-08-25 (user): operators first; once the
  StreamingOperator shape crystallizes, apply that concept to the
  BoundaryOperator.** The fork's two arms stay as recorded: a
  `BoundaryOperator` FACTORY minting the operator directly vs a
  first-class realized-law table object (whose *storage* home rides the
  container fork R3).
- **O-5 — drift repairs** ✅ LANDED `76ee98ce` for its executable half:
  the `dsa.py` widths read re-pointed to the axis-primary spelling;
  `WindowedSweep`'s `is`-check converted to space-content AT O-1 (CS2's
  identity work arrived early for that one site, forced by the un-weld);
  the `solver.py` throwaway-`SNBoundaryOperator` ergonomics stay
  RECORDED as O-4 design input (B is unchanged this phase);
  `ScheduledInvertibleOperator`'s `is`-check stays CS2.
- **O-6 — the eight `apply_*` arms** on `MaterialXSField`
  (`material_xs_field.py:741-1021`) move to the bound operators — CS4c's
  chartered "S → kernel shell"; may phase with or before F-1.
- **O-7 — factories still bridging through the mesh** (legitimate today,
  re-read at the design round): `build_within_group_system(sn_mesh,
  mat_xs, …)` (`orpheus/sn/coupled_system.py:446`) and
  `LossKernelGauge.for_mesh` — factory-level mesh consumption is the
  current bridging pattern; under the chain these become head-side
  assemblies over stage objects.

## 5c. The O-3 refinement (RULED 2026-08-25): the scheme is a stage-2 generator, and L binds the way S does

**The ruling.** `DiscretizationScheme` reorganizes on the Frame-machinery
pattern (a `DiscretizationSchemeBase` family): it is a **stage-2 generator**
— a factory whose *induced parts survive the forgetting*. `StreamingOperator`
is instantiated with **`(domain, codomain, DiscretizationScheme)`** — the
same discipline `ScatteringOperator` uses with its frame. The scheme mints;
the operator retains the minted objects; the scheme is forgotten behind a
**transitional accessor** — *declared scaffolding* — whose retirement
criterion is the **behavioral-identity test**: two operators with equal
minted arrows and different provenance must be behaviorally
indistinguishable, and the day that test passes with the accessor
unconsulted anywhere in apply/adjoint/solve, the accessor goes. Until then
it stays, honestly, because we do not yet know exactly what must be
extracted from the scheme to live inside L.

**Why L needs a scheme at all — the "full operator" fact.** L is a *full
operator*: it acts on bulk AND boundary, and the scheme is what makes the
bulk–boundary connection. And why the generator lives OUTSIDE its consumers
while each consumer forgets it: **sharing justifies interning** — one frame
per axis pair, many arrows per consumer (Windowing minting its own
differently-bound arrows from the same frame is the use case). The cache
triad extends: **kernel per cross-section set, frame per axis pair, arrows
per space** — and the scheme joins it (scheme per closure choice; evaluated
closures per mesh × quadrature × shift).

**The induced package is richer than the closure alone — the scheme induces
on BOTH sides:**

- **Operator side, in two layers that must stay separate**: the *closure
  function* a(·) — the scheme's mathematical content, structural,
  shift-invariant (DD's ψ̄ = (ψ_in + ψ_out)/2; step's ψ_out = ψ̄; SC's
  exponential; LD's local 2×2 — one function family, the Padé table row) —
  and the *evaluated coefficient table* a(τᵢ) with τ = Σ_t·Δs/|μ| —
  instance data, because τ depends on the collision diagonal. The geometric
  chord table Δs/|μ| is shift-invariant; the τ values are not. This
  recipe/instance split one level down is load-bearing for the pencil:
  `Pencil.at(σ)` rebinds the diagonal → τ re-evaluates → closure
  coefficients rebuild cheaply against static structure. It is also exactly
  the JAX static/traced boundary (#394's substrate), and it **locates the
  α-admissibility guard physically**: an inadmissible shift manifests as
  τ ≤ 0 at closure evaluation — the guard lives where the closure is
  evaluated, which is why `StreamingCollisionOperator.__init__` was always
  the right site.
- **Space side**: the **trace descriptor** and the **cell basis kind**.
  DD's edge unknowns ARE the spatial axis's trace content; LD enriches the
  per-cell representation itself (the spatial axis goes Modal — two dofs
  per cell in 1-D — plus its face traces). The scheme co-determines what
  the solution space IS, not just how the operator acts on it. (This is
  §2 T1's mechanism for the spatial axis: the head resolves its kind
  because the head holds the scheme.)
- **Realization flags**: the **cone-preservation predicate** — a predicate
  over τ, not a constant (DD is positive iff τ ≤ 2 per cell), so the flag
  is honestly *mesh-dependent*, evaluated against the actual mesh; the
  max-τ-at-setup diagnostic is this predicate's evaluation report. The
  CONE-PRESERVE gate consumes the predicate, never a boolean (#390/#400
  territory).

**The doctrine this instantiates** is already canon (the stage-2 generator
discipline, crystallized 2026-08-24 — memory
`feedback_stage2_generator_discipline.md`): *a stage-2 generator induces
structure on both the space and the operator, and the two inductions must
be minted together, at one site; forgetting = retaining the induced parts;
accessors are provenance.* Frame and Scheme are its two worked instances
(Frame: HarmonicAxis metric + Analysis/Synthesis, consistency = the
tightness gate; Scheme: trace descriptor + basis kind + closure,
consistency = ONE closure serving both apply and solve — **ERR-026's
structural closing**); Mesh and Quadrature are the degenerate space-side-
only cases. ERR-026's shape — two closures on one object, `A.inverse()`
not the inverse of `A` — is precisely what happens when the two inductions
occur at different sites; minting together makes the bug class
unconstructable, the same move as the Frame factory closing ERR-039.

**The hard guard, inherited from a corpse**: **the scheme must not carry
traversal.** CumprodScan vs wavefront vs KBA ordering is cost-side
(deferred strategy arc); the scheme is answer-side only. The clean test,
to be written into the scheme's docstring at birth as a constructor-guard
sentence: *everything the scheme provides changes the ANSWER; if a
candidate datum changes only the COST, it is contraband.* This is what
keeps the design from re-growing the `LossRepresentation` it replaces —
the scheme object is that class's successor, and its original sin is
structurally excluded, not merely avoided.

**The specified mechanism** (the ruling's plan deltas; labels key into the
CS2 / Phase-S ledger):
`DiscretizationScheme.mint(mesh_axis, quad_axis) → (Closure,
TraceDescriptor, basis-kind, positivity-predicate)`, with the solution
space's WithTrace content AND the streaming binding drawn from **one mint
call** (S2/1.3). The closure's two-layer split is an explicit dependency
of the shift-rebind path (3.2: rebinds re-evaluate the table only,
against static structure). C1's CONE-PRESERVE consumes the predicate. The
doctrine paragraph lands in D8 — the spaces chapter — beside the
forgetful-functor section, with Frame and Scheme as its two worked
instances (archivist: add to §10's cross-link list).

⛔ **SUPERSEDED IN SIGNATURE (user, 2026-08-25 — the mint spelling above
was "just a conceptual suggestion"; the paragraph stays per plan-authoring
§3):** **(a)** `mint(mesh_axis, quad_axis)` cannot be the family
signature — it is SN-welded and *would not work for diffusion* (no
quadrature axis); the family must span heads. **(b)** `mesh_axis` and
`quad_axis` live IN the space, so **the space suffices as the mint
input**. **(c)** the minted package must be DECOMPOSED BY DESTINATION —
"what goes into StreamingOperator? what into the Trace?" — and if the
decomposition is clean, the right place to mint may be OUTSIDE the
StreamingOperator, passing only the `SpatialClosure` (or so) in. **(d)**
the counter-pressure, recorded verbatim: *"StreamingOperator should have
all information it can leverage to be tested and diagnosed."* The
one-mint-call principle, the two-layer split, and the doctrine stand; the
signature and the mint-inside-vs-outside binding are the O-3 design task.
`[M]` measured input to that task (2026-08-25): the family already EXISTS
— `orpheus/transport/spatial/scheme.py` (1496 lines) holds the
`DiscretizationScheme` Protocol (:426) + `DiscretizationSchemeBase
(RegistryMixin, ABC)` (:689) with `moment_axis(ndim)` (:1375 — space-side
induction ALREADY minted here, consumed by `SNMesh.angular_trial_space`),
kernel batches, scan-coefficient surfaces, and
`CellVisit`/`UpstreamState`/`CellResult` vocabulary (:84/:205/:234) that
the traversal hard guard must adjudicate; `LossRepresentation`
(`orpheus/sn/loss_representation/__init__.py`, 4955 lines) is the
mesh-welded strategy layer (`mesh: "SNMesh"` frozen field ~:462 +
`supports(mesh)`). The two-layer split thus PARTIALLY EXISTS — scheme.py
is the structural layer, loss_representation the welded instance+strategy
layer — and O-3's design measures both surfaces and performs the
decomposition.

**Naming (ruled + one proposal to ratify at the design round).** Ruled:
keep **"closure"** for the retained object (the corpus reserves it), and
resist scheme-flavored names anywhere downstream — the scheme name
describes the GENERATOR; the minted closure is the invariant-bearing
object; the operator it binds into needs no scheme-flavored name at all
(the rule that kept `SweepOperator` from becoming `TrackSweepOperator`).
Proposal (main agent): qualify the class as **`CellClosure`** — the corpus
already carries an ANGULAR closure (`pole_angular_closure`), so an
unqualified `Closure` class is ambiguous between the angular strategy and
this cell-local relation; "cell" states the locality that is the object's
defining property (DD/step/SC/LD are all cell-local relations
(inflow, source) → (average, outflow) per (cell, ordinate)), and the two
layers then read naturally — the `CellClosure` carries the function family
and shift-invariant structure; its evaluated coefficient table is instance
state rebuilt on shift rebinds. Alternate if "cell" reads too
finite-volume: `SpatialClosure`.

✅ **RULED 2026-08-25 (user): `SpatialClosure`** (the `CellClosure`
proposal not taken); and the pole angular closure family
(`orpheus/sn/sweep/pole_angular_closure.py`) is the candidate member of
an **`AngularClosure`** concept — the closure family pattern is
`<Axis-role>Closure`, one closure concept per axis the scheme closes
(R15).

✅ **THE BINDING HALF OF THE O-3 DESIGN TASK IS RULED (user, 2026-08-28 —
recorded at `streaming_path_says_what_it_is.md` §5b "THE MECHANISM" and its
phase P4.9).** Supersession item **(c)** above asked where the mint sits;
ruled: **outside the operator** — `StreamingOperator` is constructed from
**(domain, codomain, spatial closure, angular closure)**; discretization
schemes are factories returning a `SpatialClosure`, the angular scheme
returns an `AngularClosure` (R15's family — P4.9 makes both
constructor-real). The transitional-accessor question resolves with it:
*what must be extracted from the scheme to live inside L* is the spatial
closure — and each factor already ships its own adjoint
(`streaming_cell_transpose`; the polymorphic `angular_adjoint` family), so
`L.H` composes. ✅ *2026-08-28: the un-weld half (P4.9a) is LANDED — the
M-M relation is single-homed in the closure, the L2 protocol is purely
spatial, and the closure mints its scan constants; the 4-arg ctor that
makes both closures constructor-real is P4.9b.*
✅✅ *2026-08-29: **P4.9b is LANDED and MERGED** — the ctor is the
transitional `(sn_mesh, spatial_closure, angular_closure)` (three
required fields, no defaults, NO guards — the no-guard ruling survived a
four-attack exercise; `.pose(sn_mesh)` is the intermediate posing
surface; the literal cross-method 4-tuple stays the recorded end state
riding O-3/CS5). ⭐ Supersession item (c) sharpened by ruling: the hub
(SNMesh — the save-state/data hub) KEEPS the generator (DSA consistency
+ nodal/modal space induction); the operator holds the two CLOSURES.*
Item **(d)**'s
diagnosability counter-pressure is NOT discharged: whether the operator
keeps a provenance accessor to its generator is now a **P4.9b**
design-time question. ✅ **DISCHARGED at P4.9b (user's own words in the
Q2 ruling — "leaves just an accessor for provenance", realized with
ZERO aliases): the hub retains the generator and the operator reaches it
through its transitional `sn_mesh` field; today `spatial_closure` IS the
generator instance (extraction = identity until O-3 splits the
closure/factory family), so every input the operator computes from is a
readable field. No new accessor was minted; O-3 revisits at the split.** **O-3 retains** the cross-head
mint signature, the (TraceDescriptor, basis-kind, positivity-predicate)
package, the two-layer closure-function/evaluated-table split, and
`cell_balance.py`'s reorganisation under the scheme family; the streaming
plan's P5 still rides O-3. ⚠ The `[M]` 2026-08-25 measurement above
cites `scheme.py` line numbers (`:426`/`:689`/`:1375`, "1496 lines") that
predate two waves of growth — the file is **1798 lines** after P4.3 moved
`StreamingTerms` in at `:107` (2026-08-28; Protocol now `:646`, Base
`:910`, `moment_axis` `:1677`). Re-measure at the design round; the
structural claims stand. ⚠ The 2026-08-25 paragraph above cites the
closure at `sn/sweep/pole_angular_closure.py`; it moved to
`sn/angular/closure.py` at that plan's P2 (`dcd6a9f6`), byte-identical.

## 5d. The Phase B opening discussion — the record (2026-08-25, IN PROGRESS)

⚠ **Epistemic status of this whole section**: the census/measurement rows are
`[M]`; the *design* rows are the main agent's **proposals**, opened for the
user's steer and **NOT ruled**. Nothing here has been executed. The one
LANDED item is the falsified-claim repair (`7433f7b3`), which was a
correctness fix, not a design act.

### 5d.1 The naming re-derivation — `AngularClosure` stands (R15 unchanged)

The user asked whether **`CurvilinearClosure`** describes the family better.
`[M]` **it collides with established in-tree vocabulary**: *"curvilinear cell
closure"* already means the SPATIAL closure's curvilinear arm —
`diamond.py:167` ("DD **has** a curvilinear cell closure"), `scheme.py:846`
(`supports_curvilinear` = "whether the scheme has a curvilinear
(sphere/cylinder) **cell closure**"), + 5 LD sites. Naming the angular
family `CurvilinearClosure` would give the angular object its sibling's
name, and one grep would return both concepts.

The structural reason behind the collision: **curvature is the REGIME, not
the axis, and it modifies both axes' treatment** — face areas / ΔA on the
spatial side, α / τ on the angular side. So "Curvilinear" cannot
discriminate *what is being closed*, whereas `<Axis-role>Closure` names the
closed unknowns. The Cartesian `IdentityAngularClosure` member confirms the
family spans regimes: the axis always exists; the closure is trivial when
curvature vanishes.

⚠ Recorded weakness of `AngularClosure` (not a defect today): in the wider
field the phrase can also mean the moment-hierarchy closure (P_N / M_N). If
ORPHEUS ever types THAT concept, it closes the harmonic-moment expansion,
not the angular-cell differencing — `MomentClosure` / `HarmonicClosure`
names it without contention. Reserve those spellings; do not rename this one.

### 5d.2 The smuggle audit — what is DD-specific and has escaped DD

The user's question: *is anything actually DD-specific and smuggled outside
of DD?* `[M]` four findings, one of them live:

1. ⛔ **DD's blend inverse `2.0` is hard-coded in the scheme-neutral balance
   module.** `transport/spatial/cell_balance.py:248` (`streaming_denom_term =
   2.0 * abs_mu * A_downstream`) and `:343` (the full denominator). That
   `2.0` is exactly `1/w` at DD's `w = ½`. Function consumers are NOT
   DD-only: `diamond.py:87` **and** the streaming matvec bodies
   (`loss_representation/__init__.py:3121`, `:3472`). Harmless **by
   coincidence** — `supports_curvilinear` gates every curvilinear mesh to DD,
   so the containment is a CAPABILITY FENCE, not the algebra. The principled
   form already exists one class over: `affine_scan_coefficients` RETURNS the
   blend weight (`face_blend_weight`) rather than inlining it.
   ⟹ **filed as #407**; the carve belongs to O-3 (§5c), where `cell_balance.py`
   reorganizes under the scheme family anyway.
2. **The mirror smuggle, same family, other axis**: the M-M angular
   recurrence is re-spelled inline in the matvec/transpose kernel bodies
   (`loss_representation:4346/:4651/:4693`) from `GeometryCoefficients`'
   stored constants, instead of routed through the closure's kernels — two
   spellings of one recurrence, mitigated only by the constants being
   single-sourced (`cache.py:332-337`). Recorded in #407; carve both axes
   together.
3. `CollisionCache`'s class docstring states DD's formulas as the *cache's*
   contract while the code correctly delegates to `scheme.affine_scan_
   coefficients` — stale over-specific doc, repair when O-3 touches it.
4. The `affine_scan_coefficients` operand MENU is DD-drawn (`A_total` present
   because DD needs it — LD's override marks it unused; `c_in` absent because
   DD's triple does not need it). Not wrong; drawn around the first occupant.
   Re-derive the menu when a second curvilinear occupant arrives.

✅ **The properly-fenced counter-example, worth keeping as the model**:
`sn/acceleration/dsa.py:214-219` declares and GATES its DD-consistency
(`scheme_key != "diamond_difference"` → refusal citing the missing WDD
stability theory). Scheme-specific consumption done right.

### 5d.3 The LD probe — why a second member was brought in

⭐ **The user's framing, and it is the methodological point of this round**:
LD was brought in NOT as scope creep but because *"having 2 schemes with
angular closure allows us to precisely pin-point the shape of an angular
closure base class and machinery"* — the `defer-until-≥2` rule used as a
DESIGN INSTRUMENT. The second member is derived to triangulate the ABC, not
to be built.

Two probes were run **blind to each other** (structural independence,
`vv-principles` L11): a literature sweep of the local corpus, and a
from-scratch SymPy derivation.

**⛔ REFUTED — "the curvilinear LD cell closure is unpublished"** (the tree
asserted this at 16 sites). `[M]` **Adams & Martin 1992**, NSE 111(2), App. A
(LOCAL, page-verified pp. 160-161) carries the complete 1-D spherical LD
moment-balance system: (A.1a)/(A.1b) moments, **(A.2a)/(A.2b) weighted-diamond
angular closure applied PER SPATIAL MOMENT**, (A.4a-d) mass integrals. Plus
⛔ MWS 1996 (JCP 128) was listed here on 2026-08-25 as the spherical-LD
lumping primary — **REFUTED the same day** by the agent's own round 2: it is
**1-D SLAB** (Eq. 53 is `μ ∂ψ/∂z`, no angular-derivative term anywhere, and
its §8 names 1-D spherical as FUTURE work). The round-1 classification came
from a citing paper's context rather than the paper. The *published* verdict
is unaffected — it rests on Adams-Martin + Hill + Machorro + Palmer-Adams.
Also: MWS's `τ = σ_t/μ` is NOT the Morel-Montry weight (a FOURTH overloading
of τ beyond the three the closure module already warns about).
⭐ The real sphere-lumping source is **Palmer-Adams 1993**. Morel-González-Aller-Warsa 2007 (r-z
lumped LD — now local), Hill 1975 ONETRAN, Wu-Xie-Fischer 1999,
Lathrop 2000 §III.D. Root cause of the false negative: a ONE-QUERY
denominator in a prior extraction while the refuting paper sat in
`scratch/literature/`. ✅ **REPAIRED `7433f7b3`** — all 16 sites re-scoped to
"not yet implemented (#158)". Full record: **#158's 2026-08-25 comment**.

**The two probes agree entry-by-entry** — `[M]` all three independent ratios
of the redistribution Gram: A-M's `{r_kΔr_k, Δr_k²/6, r_kΔr_k/3}` vs the
derivation's `R = ΔA·[[1, h/(6r_c)], [h/(6r_c), 1/3]]` ⟹ `R₀₁/R₀₀ =
h/(6r_c)` ✓, `R₁₁/R₀₀ = 1/3` ✓, `1/w` placement ✓, per-moment τ ✓,
non-diagonal collision Gram ✓, no-lumping-on-the-sphere ✓ (the derivation
measured that lumping `R` BREAKS the flat-flux L0 identity; A-M use the exact
mass matrix). ⚠ One disagreement, **sign only**: A-M's printed slope-coupled
terms carry a minus where the derivation gives plus; the convention lives in
the paper BODY (Sec. III.A), not the appendix. Magnitudes cross-confirmed,
sign to spot-resolve at transcription — the ERR-032-class hazard the
two-probe design exists to catch.

### 5d.4 ⭐⭐ What the second member actually pinned — the axes are ORTHOGONAL

The finding that reshapes the family, and it was NOT the expected one:

> **There is no separate "LD angular closure" to write.** The angular closure
> is ONE body (weighted diamond / M-M τ); LD's entire injection is the
> **Gram `R`** — the one-measure-down Gram of the scheme's own basis.

`[M]` the member split (76 SymPy checks,
`scratch/ld_curvilinear_shape_derivation.md` §7):

- **Member-INDEPENDENT** (one body, no LD arm): the τ producer, the angular
  cell edges, `alpha_dome` + its `α_{M+1/2}=0` contract, the `c_in`/`c_out`
  algebra, the P3 τ∈[0,1] guard, `march_start_structure_per_level` / the
  carrying-levels predicate, and the recurrence step itself. (τ eliminates
  componentwise with the **same scalars**: `∂(redist)/∂ψ_vec = c_out·R/w_n`
  exactly — no row-dependent `c`.)
- **Member-DEPENDENT**: carrier shape (`(ng,M+1,nx)` → `(ng,M+1,nx,2)`;
  `2^d` in general), `R`, the collision/source Gram `M`, the weak gradient
  `G`, the cell solve (scalar denom → 2×2), the coefficient triple
  (`(a, 1/denom, w)` → `(a, A⁻¹, w_vec)`, `a` stays SCALAR), the
  starting-direction solve, the sweep-frame⇄global-frame map, lumping.

⟹ **PROPOSED contract** (main agent; not ruled): the closure's cell
contribution widens to a trailing moment axis —
`denom_term (n_mask, n_mom, n_mom)`, `upstream_numer_term (ng, n_mask,
n_mom)` — with DD the `n_mom == 1` case, byte-identical after a squeeze; and
the scheme's ONLY injection is one hook,
`redistribution_gram(cell) -> (n_mom, n_mom)` (DD `[[ΔA]]`; LD the 2×2 above,
**diagonal** on the cylinder).

⟹ **This REFINES the user's proposed pattern** (*"a scheme is a factory of
spatial closure, angular closure, and other things; whatever scheme cannot
provide an angular closure fails loud"*). The measurement says the scheme is
a factory of *(spatial closure, the moment-space data the shared angular
closure needs)* — the loud-failure discipline is right and already shipped in
weaker form (`supports_curvilinear` + a value sniff + a per-visit raise); what
moves is WHAT is minted. The genuinely SECOND `AngularClosure` member is not
LD-in-space at all — it is **angular-LD** (Walters-Morel 1991, per Lathrop
2000 §III.D, recorded there as *less* accurate than weighted diamond). That is
the axis the ABC must keep open.

⚠ **Risk to the contract, unsettled (memo §10, "Q7")**: `R₀₁` mediates an
average↔slope coupling of size `h/(6r_c)`, which is **`1/3` at the pole
cell** — exactly where the M-M flux dip lives. If a redone BMC first-order
expansion carrying both moments gives `β^LD ≠ 0` at the DD-derived τ, then
**τ becomes cell-dependent** and today's `tau_per_ordinate` contract widens
from per-ordinate to per-(ordinate, cell). Settling procedure: memo §10.4.
⟹ **the ABC must not freeze `tau_per_ordinate`'s arity until Q7 is closed.**

### 5d.5 Two gates that do not exist, and one that is doubly blind

`[M]` from the derivation, recorded for `test-architect` (see #158 §5):

1. **Two inequivalent starting-direction discretizations**, both flat-flux
   exact and `O(h)` apart at the pole cell. ⛔ **Double blindness**: the
   canonical L0 flat-flux gate cannot discriminate them, AND the gap is
   proportional to the **slope source moment**, which ORPHEUS zeroes for the
   external source (#247). A `vv-principles` Mode-12 stabiliser finding.
2. **No catcher exists** for a global-vs-sweep-frame **slope sign** error on
   the half-angle thread (the slope flips sign across `μ=0` where the sweep
   reverses; DD never meets this because the average is sign-invariant).
   Invisible to flat flux (`ψ̂=0`) and to any single-sweep-direction fixture.
3. ⛔ **A per-moment-row gate written on the CYLINDER is vacuous** (the slope
   row is trivially `0=0` under flat flux, and `R` is diagonal there) — such a
   gate must run on the **sphere**.

### 5d.6 Where the discussion stands — the open forks

| # | fork | main agent's lean |
|---|---|---|
| B-a | `reduced`'s angular family: take it whole / take loose fields / **derive at the consumer** (α, τ, μ_start are all quadrature-derivable; `redist_dAw = ΔA ⊗ 1/w` is a stored PRODUCT of a quadrature factor and a geometry×basis factor — `R₀₀ = ΔA` exactly) | derive-at-consumer; the minimal operand set trends to `(quad, coord)` with `R` arriving from the scheme |
| B-b | does Phase B land the constructor re-contract, or wait and open with the uncontested retirements? | ⚠ **the Q7 risk now argues for the retirements first** — re-contracting before τ's arity is settled would freeze the wrong shape |
| B-c | who mints the closure (scheme-mints vs registry) | scheme supplies `R`; the mint question is O-3's (§5c), where the space-as-mint-input ruling (R19) lives |
| B-d | Phase B's boundary at `sn/sweep/cache.py` | `[M]` `CollisionCache` is already operand-shaped (`from_geometry(geom, sig_t, scheme)`); `GeometryCoefficients`' welds are spatial-chart + traversal (the deferred arc). ⟹ Phase B touches neither |

**Retirement list accumulated so far** (uncontested, all `[M]`): the two
production-dead flags on `ReducedStreamingOperator`
(`requires_upstream_angular_state`, `angular_marching_axis` — 0 production
reads, 12 test references); the two dead `SNMesh` shims (`face_areas`,
`delta_A` — 0 callers); `redist_dAw` as a stored product; and two stale
comments (the closure binding site's over-listed operands at
`augmented_mesh.py:391` — it names `_volumes`/`axis_widths`, `[M]` zero
volume/width reads in the closure file; and the kernel comment claiming the
closure reads `V` from `self`).

### 5d.7 The literature round 2 (the four acquired papers) — three results that move the design

The user supplied the four requested papers; the same agent was resumed (its
round-1 context intact) and asked six questions. Record:
`scratch/lit_ld_curvilinear_sources.md` (694 lines, round-1 sections intact
per plan-authoring §3).

**(a) ✅ The sign disagreement is SETTLED — the printed minus is a published
typo, and the derivation was right.** Adams-Martin Sec. III.B defines `ψ^x`
as the linear-basis coefficient on `P = 2(r−r_k)/Δr` with a Galerkin test
space (`v = b`), and `γ` only for the average (A.4d). ⟹ the four
redistribution weights are ONE symmetric positive-definite Gram
`[[r_kΔr, Δr²/6], [Δr²/6, r_kΔr/3]]`, **all plus**: the printed magnitudes
match exactly and only the two `ψ^x` signs differ. `[M]` five independent
confirmations (Machorro's single-signed weak form; ONETRAN's positive
`ΔA_i`/`z_5`; Palmer-Adams's FL row sums + printed BLD `R`).
⚠ **The typo is invisible to a conservation check** — both terms telescope
over `m` for either sign. A balance-only gate cannot catch a
redistribution-sign error; only a slope-exciting fixture can.

**(b) ⛔⛔ PLAIN SPHERICAL LD FAILS THE THICK DIFFUSION LIMIT** — Palmer-Adams
1993's complete curvilinear verdict, and it is the most consequential fact
in this round: three-point removal term, unphysical boundary conditions,
interior scalar flux low by **~2×**. Fully-lumped (FL) and corner-balance
(CB) **pass**; mass-lumped (ML) partial. (r-z: BLD / MLBLD / SLBLD fail,
FLBLD / CB pass.) ⟹ **#158's curvilinear arm must implement the LUMPED
form, not bare (A.1)** — and note this is Adams refuting, one year later,
the naive reading of his own 1992 appendix (not a formal contradiction:
different questions, equations vs asymptotics).
⭐ And it CORROBORATES the derivation's independent lumping finding from the
other side: `[M]` the derivation measured that lumping `R` breaks the
flat-flux L0 identity; Palmer-Adams state that under lumping `L_k` "is
redefined **to preserve the infinite-medium solution**" — the same identity,
repaired by construction rather than left broken. ⟹ lumping is admissible
**only** with that compensating redefinition; recording the pair is what
makes the constraint legible.
⛔ **REFUTED 2026-08-26 (§5d.8, probe 03) — the paragraph above CONFLATES TWO
DIFFERENT OPERATIONS, and the synthesis "the same identity" is FALSE.**
`[M]` Palmer-Adams's FL is **nodal ROW-SUM** lumping, which **preserves
`R₀₁` EXACTLY** (only `R₁₁` moves, `ΔA/3 → ΔA`). The operation the earlier
derivation condemned is **Legendre-DIAGONAL** lumping — a different map. So
"⛔ `R` may not be lumped on the sphere" does **not** apply to FL, and FL
needs no compensating repair of that identity. ⚠ The error is mine (main
agent): two operations sharing the word "lumping" were merged into one
claim, and the merge read as a cross-confirmation because both halves were
independently true. The sharper fact that replaces it: `[M]` the
infinite-medium identity pins only `rowsum(L) = (−1, +1)`, leaving **one
free parameter per row** ⟹ **the accuracy/positivity trade is a CHOICE of
that parameter, not a property of "lumping"**.

**(c) ⭐⭐ THE RANK CONTRADICTION — and it splits an index the design had
CONFLATED.** Two published families disagree on what the angular closure
acts on:

| family | angular device | acts on | redistribution coupling |
|---|---|---|---|
| Adams-Martin (A.2a/b), Palmer-Adams (9), Wu-Xie-Fischer (27) | weighted / plain diamond | **every spatial moment** | full `2×2` Gram |
| **ONETRAN** (Hill 1975, Eq. 32) | plain diamond (Eq. 30) | the spatial **average only** | **rank-1**: `(α/w)·[ΔA_i; z_5] ⊗ [1,1]` |

Both are published and shipped; **ORPHEUS must choose explicitly** rather
than inherit one by accident.

⟹ **The consequence for the ABC, and it SHARPENS §5d.4 rather than
refuting it.** §5d.4 proposed one index (`n_mom`) and a scheme-owned hook
`redistribution_gram(cell) -> (n_mom, n_mom)`. The ONETRAN datum shows
`n_mom` was two indices wearing one name:

- **`n_mom`** — how many spatial moments the SCHEME carries (DD 1, LD 2);
- **`n_thread`** — how much of the spatial representation the ANGULAR
  DEVICE propagates through its half-angle recurrence (ONETRAN 1, A-M 2).

The coupling is the **rectangular** pairing `R_kj = ∫ b_k^scheme · b_j^thread ·
r dr`, shape `(n_mom, n_thread)`: `1×1 = [ΔA]` for DD; `2×2` for LD +
per-moment closure; `2×1 = [r_kΔr; Δr²/6]` for LD + ONETRAN's average-only
closure (`[M]` matching ONETRAN's own `[ΔA_i; z_5]`).

⟹ ⭐ **`R` is owned by NEITHER side alone** — it is the pairing of the
scheme's spatial basis with the angular thread's spatial basis under the
one-measure-down geometry. The orthogonality claim of §5d.4 SURVIVES and is
sharpened: each axis contributes exactly one index to `R`. The proposed hook
therefore takes the thread's basis as an argument rather than being a pure
scheme property. **Still PROPOSED, not ruled.**

**(d) Q6 — a clear NEGATIVE, which leaves the arity risk OPEN.** `[M]`
Palmer-Adams's curvilinear verdict is driven **entirely by spatial
locality**; a grep of it finds **zero** occurrences of diamond / weighted /
τ / flux-dip. **No source analyzes retuning τ for LD.** The one qualitative
coupling is Machorro's: the dip "involves angular discretization, spatial
discretization and boundary conditions at the origin", and a joint
linear-in-`(r, μ)` order removes it *without* τ at all. ⟹ §5d.4's arity
warning **stands unresolved** — settling it is original work (re-run
Palmer-Adams's matrix analysis with τ symbolic; their `R_k`/`α` formalism
supports it).

**(e) Q2 — the starting-direction fork, answered and complicated.** ONETRAN
takes variant **(b)** mechanically (bulk rows, same test functions), with
Eq. (38) replacing the curvature term by `−μ_m ×` the two-point **average**
— flat-flux exact, `O(h)` otherwise, **no justification given**.
Machorro/DG takes **neither**: there is no starting-direction flux at all,
since `(1−μ²)` vanishes at `μ = −1`. ⭐ And the round's highest-value
sentence (Machorro, printed p. 79): Walters & Morel found the
bilinear-DG/"SLD" **origin error** under fine-radial / coarse-angular meshes
and attributed it to *insufficient starting-direction information*; both
they and Machorro repair it with **quadratic-in-angle functions in the cells
bordering `μ = −1`** — the same hybrid Lathrop 2000 adopts. ⟹ the
starting-direction treatment may demand a LOCALLY HIGHER-ORDER ANGULAR
representation, which is `AngularClosure` territory, not the scheme's.

**(f) ⛔ Round 1's MWS classification was REFUTED by its own author-agent**
— see the ⛔ block in §5d.3. Worth carrying as method: the correction came
from reading the paper the user supplied, i.e. **the acquisition itself was
the instrument that caught it**.

**New acquisition asks** (both M&C proceedings, neither local): **Palmer &
Adams 1991**, ANS M&C Pittsburgh Vol. 5 §21.1 pp. 4-1..4-11 — writes the
spherical LD `R_k` **unlumped**, and would settle Adams-Martin's intent
directly; and **Walters & Morel 1991**, M&C Vol. III p. 13.2 3-1 — the
primary for BOTH the LD-in-angle scheme (the genuinely second
`AngularClosure` member) and the starting-direction remedy in (e).

### 5d.8 ✅ THE ARITY QUESTION IS SETTLED — `tau_per_ordinate` KEEPS its arity, by THEOREM

`[M]` 106 SymPy checks / 6 probes / 0 failures. Memo:
`scratch/tau_under_ld_dip_analysis.md`. Original derivation (no source in
the literature treats this — §5d.7(d)), run after the user's steer to
*"lean heavily on the mathematical properties and theorems … identify what
is the invariant, and what physics needs preserved."* **⟹ the ABC may be
frozen as it stands; §5d.4's arity warning is DISCHARGED.**

**Theorem A (the decision).** A scalar convex combination commutes with
every linear map, so `P(blend) = blend(P)`. Both of τ's defining conditions
— cone membership `τ ∈ [0,1]` (the shipped P3 guard) and the
barycentric/`span{1,μ}` condition — are therefore *the same scalar statement
in every moment component*. A per-(ordinate, cell) τ is an **overdetermined
system whose every row returns the same per-ordinate value** (`[M]` solved
independently at `n_mom = 4`). Hypotheses checked and none mentions a basis:
τ is `r`-independent by construction; the moment projection is linear; `K`
is convex (an intersection of half-spaces).

**Theorem B (why β cannot acquire spatial content).** The redistribution
operator is the **TENSOR PRODUCT** `R_i ⊗ A_ang(τ, α, w)` ⟹ the
diffusion-limit condition is the *identical* angular scalar `S` — `[M]` free
symbols `{μ, w, τ}`, no spatial symbol — and `S = 0` annihilates the
contamination in every moment row **for an arbitrary symmetric `R`**.
⭐⭐ **This is §5d.4's orthogonality claim PROVEN at the operator level**: the
two axes do not merely contribute one index each, the operator FACTORS.

**Blast radius, verified:** `CellVisit.tau: float` (`scheme.py:197`) is
already stamped per cell visit from `tau_per_ordinate[global_ordinate]`
(`augmented_mesh.py:1644`) — even a WIDENS verdict would have been
producer-side only. *(✅ tense note 2026-08-28: P4.9a retired both the
field and the stamp — the visit family is purely spatial and τ is read
from the closure's own accessors; the verdict above is unaffected, it
was about the producer side, which is now even more so.)*

#### ⭐ The LEAD — the risk relocates to the SEED, and harder than expected

`[M]` the starting-direction equation carries **no** angular redistribution
(`α_½ = 0`; `(1−μ²) = 0` at `μ = −1`), so it is a purely SPATIAL solve
inheriting the spatial scheme's cone behaviour **where nothing angular can
damp it**. Transmission ladder (one probe, all three): **DD flips sign at
`τ_opt = 2`** (Padé(1,1)), **bare LD at `τ_opt = 3`** (Padé(1,2), with
`a·τ_opt → −2`), **lumped LD never** (Padé(0,2), discriminant `1−2 < 0`).
In the thick limit `τ_opt = σ_t h → ∞`, so a **bare-LD seed march ALTERNATES
IN SIGN cell to cell**.

⭐⭐ And the quantified punchline: `β_eff(μ_s) = S + (μ_s + 1)·S_e` is
**exactly affine**, so `[M]` with τ AT the Morel-Montry value, a
starting-cosine error of **1.6 % (S4) → 0.05 % (S32)** reproduces the
*entire* diamond-scheme contamination. ⟹ **τ buys nothing a sloppy seed does
not give back, and the leverage GROWS with N.**

#### The refutations (8 in memo §10) — three matter to this charter

1. ⛔ **My H1 ARGUMENT was INVALID** (its conclusion survives). *"The
   signature admits no spatial argument"* is `vv-principles` **Mode 8
   SIGNATURE-tautological**: the signature reflects the *derivation's scope*
   (M-M and BMC both hold space continuous, and say so), not the joint
   problem. Had LD introduced spatial dependence, the answer would have been
   a NEW function with a spatial argument — the existing one's arity says
   nothing. ⟹ **a type signature is evidence about an author's assumptions,
   never about a theorem.** (Durable; belongs in `lessons.md`.)
2. ⛔ **Morel-Montry's OWN summary rule is refuted** — *"the dip can be
   expected to be eliminated with any spatial scheme as long as the starting
   flux is not seriously UNDERestimated"*. `[M]` `dβ/dμ_s = S_e` **flips
   sign** between `N = 2` (`+9.1e-1` — their own Gauss-S₂ case, the only one
   they computed) and `N ≥ 4` (`−1.1e-1 … −1.7e-5`): the SAFE DIRECTION
   INVERTS while the stakes collapse 5 orders. A 1984 universal generalised
   from `N = 2` (`vv-principles` #13 — a sample promoted to a population, in
   the literature rather than in our tree).
3. ⛔ **My steer's "lumping ⟹ M-matrix" is FALSE** — MWS lump the mass and
   deliberately *not* the gradient, so `A_LR = +½` and `A⁻¹_LR < 0` (their
   own §8 caveat). **What survives** is strictly weaker and is the useful
   half: the **transmission** becomes unconditionally sign-preserving.

#### Two design answers this delivers

- **The seed fork of §5d.5 DISSOLVES.** `[M]` memo variants (a) the slab-LD
  march and (b) the `μ = −1` bulk row are **asymptotically IDENTICAL** —
  both discrete gradients equal `[[0, 2/h], [0, 0]]`, so `e⃗ = 0` exactly and
  `μ̃_½ = −1` holds per component for both. The M-A origin-values half is
  **vacuous** under LD (DG derives its traces; there is nothing at `r = 0`
  to supply). ⟹ the criterion with teeth is the **CONE**, not the
  contamination: **ship the seed march with the LUMPED cell.** (Zeroing the
  seed slope gives an `O(1)` mesh-independent defect; borrowing it gives
  `(1+μ₁)ĝ`.)
- ⛔ **`is_positivity_preserving` CONFLATES THREE PROPERTIES** and must not
  read `True` for lumped LD (`[M]` counterexample `A⁻¹_LR =
  −2/(τ_opt² + 2τ_opt + 2)`). The honest split:
  **`transmission_is_sign_preserving`** (`False` DD / `False` bare LD /
  **`True` lumped LD** — the first `True` among affine-scannable schemes,
  and the property the seed analysis actually needs) and **`is_monotone`**
  (`A⁻¹ ≥ 0`; `False` for all three shipped). ⟹ filed as **#408**;
  it is the first real consumer-side finding for #390.

⚠ **The scope statement the theory page owes** (memo §8(iii)): the
asymptotic expansion settles the *arity* question and is **structurally
incapable** of settling the *positivity* question — a sign-alternating
cell-to-cell mode is EXCLUDED BY THE ANSATZ, which is why Palmer-Adams carry
*"stable and reasonable"* as a SEPARATE acceptance criterion. **Two
questions, two instruments** (`vv-principles` Mode 12).

**Not settled** (memo §13): the cylinder is asserted, not measured (the
theorems transfer — `R` is μ-independent and diagonal there, so the `R₀₁`
channel does not exist — but every numeric leg is spherical; re-run probe
06's cone legs with `nu_closure_residual`, ⛔ never β on the folded arc); the
oscillatory mode's end-to-end flux error is unquantified (needs a
pole-resolved fixed-source sphere at `σ_t h > 3`, fixture OUTSIDE
`span{1,μ}` per `vv-principles` #24(d)); whether Palmer-Adams's actual `L_k`
is one of the memo's `(λ,ν)` members (needs the 1991 M&C paper).

## 6. The rulings ledger (all user, this session, unless marked)

| # | ruling | date |
|---|---|---|
| R1 | Operators must stop requiring `SNMesh` at construction (transitive reach fine) — the arc's charter | 2026-08-24 |
| R2 | Consumed objects first (space, fields, operators); solver/strategy/traversal deferred to a lazy-realization arc | 2026-08-24 |
| R3 | The container fork is OPEN — the aggregate may survive as organization + persistence (#406); the BoundaryOperator-factory option is live | 2026-08-24 |
| R4 | The posing chain is a filtration; the architecture of §2–§4 — ratified through three adversarial rounds | 2026-08-25 |
| R5 | All axes formally construct at the method (T1); stages accumulate measures and data; no half-axis objects | 2026-08-25 |
| R6 | Naming: `Materials` (a declaration), not `Medium`; `InfiniteMedium(mixture)` in the problem layer is where the word survives; `mixture.as_infinite_medium` rejected (dependency inversion). *Refined by R23: the concrete home is `orpheus/homogeneous/`* | 2026-08-25 |
| R7 | R-mint: the method-time mint is the law; Materials-time admission is a preview whose refusals must be implied by the mint's | 2026-08-25 |
| R8 | R-raw: Materials carries raw per-material grid provenance; no early collapse; per-head axes may differ | 2026-08-25 |
| R9 | The state-fields stage exists (named now, empty today); `Mixture` = parametrization; state binds to space as a chain stage | 2026-08-25 |
| R10 | Deck identifications are stage-2 geometry, not BCs; only vacuum/albedo/inflow are boundary data; realization is head-side, exact-or-refuse | 2026-08-25 |
| R11 | Assigned-but-undeclared refuses at the overlay; declared-but-unassigned is legal and inert by T2 — no warning machinery | 2026-08-25 |
| R12 | Symmetry is the admissibility bound and quality criterion on refinements, not the flow; per-stage group machinery is aspirational (family: #152, #166) | 2026-08-25 |
| R13 | `MaterialXSField` dissolution verdict — ✅ RATIFIED at the arc design round's opening (was: proposed-unopposed) | 2026-08-25 |
| R14 | O-3 resolved (§5c): `DiscretizationScheme` is a stage-2 generator (`DiscretizationSchemeBase` family, successor of `LossRepresentation`); `StreamingOperator` binds `(domain, codomain, scheme)`; the closure splits function/evaluated-table; transitional accessor under the behavioral-identity retirement test; **the scheme must not carry traversal** (answer/cost constructor guard); "closure" names the retained object, scheme-flavored names forbidden downstream (class-name proposal `CellClosure`, to ratify) | 2026-08-25 |
| R15 | Closure naming: **`SpatialClosure`** (the `CellClosure` proposal not taken); the pole angular closure family is the **`AngularClosure` candidate member**; family pattern `<Axis-role>Closure` | 2026-08-25 |
| R16 | S-3 ruled as recommended: the arc takes minted space objects as given; mint-as-free-function / axes non-Optional / name-bridge retirement stay CS2, whose landing surface is the head (T1) | 2026-08-25 |
| R17 | The `Materials` class IS minted this arc (stage 1); concrete shape ruled on the main agent's proposal — *resolved same day by R20/R21* | 2026-08-25 |
| R18 | O-4 sequencing: the factory-vs-table fork stays unruled until the operator shape crystallizes — operators first, then the crystallized concept applies to the BoundaryOperator; and its "declarations move" premise is refuted (declarations already live at the geometry stage) | 2026-08-25 |
| R19 | §5c mint correction: the SPACE suffices as the scheme's mint input (the axes live in the space); the minted package decomposes by destination; the family must serve diffusion (not SN-welded); the StreamingOperator retains all information it can leverage for test/diagnosis — mint-inside-vs-outside is the O-3 design fork | 2026-08-25 |
| R20 | `Materials` home: `orpheus/data/materials.py`; the incumbent property-correlation package renames `data/materials/` → `data/material_properties/` ("long overdue"; `[M]` 6 consumers, all TH/kinetics/fuel zone) | 2026-08-25 |
| R21 | `Materials` final shape: frozen `eq=False` identity wrapper over `Mapping[int, Mixture]` (MappingProxyType); admission refuses only the empty declaration; `restrict(ids)` = guard 2; **no `ng` property, no preview** — concrete-property resolution is LAZY at the consuming stage; today's mesh-time `InconsistentMaterialsError` untouched this arc | 2026-08-25 |
| R22 | The data-kind taxonomy: macro data arises ≥3 ways (GENDF-class / PENDF-class / collapsed-from-solve); per-kind consistency checks (a GENDF check supersedes scalar-ng); PENDF → MC exclusively; GENDF + collapsed → one method-agnostic final object (multigroup MC included); the data-layer overhaul is priced future work (§9.6); first heavy consumer of concrete energy-structure numbers = Campaign 2 partitioning (spectral-radius direction, hedged) | 2026-08-25 |
| R23 | `InfiniteMedium` reframed: the homogeneous family's AGGREGATE (`SNMesh`'s analog — organization + shared objects), in `orpheus/homogeneous/`, taking `mixture` directly (never a Materials); R3's first data point; design after the operator shape crystallizes — ✅ REMEDIED 2026-09-08 by the CS4c coda — C1 `5caad3d6` / C2 `39e7f32f`, and it landed exactly as ruled EXCEPT for the name: the aggregate ships as **`HomogeneousProblem`** (ruling R-c1's word, `orpheus/homogeneous/solver.py`), a frozen dataclass taking one `Mixture` directly, owning the pose + the mixture-direct fields + the bound operators + the rate co-vectors as per-instance `cached_property` state. The timing condition was honoured (it was designed after the CS4c operator shape crystallized). ⚠ Its home is INTERIM: the carve into a standalone module with a thin Problem → Solution solver is the consumers campaign's, alongside `SNMesh` → `SNProblem`. | 2026-08-25 |

## 7. The adversarial record (distilled; refuted candidates are first-class output)

Round 1 (main agent attacks → user counters): **A1** twin-path bypass →
WITHDRAWN: the infinite path is the energy sub-algebra (slowing-down), a
different problem family sharing primitives; survives as the
shared-primitives condition (§3). **A2** chain overfits SN → CONCEDED by
user; universal prefix + per-method measure stacks. **A3** stage-1 "Medium"
is an inventory → resolved by the user's stronger form (a tower of *total*
problems, each complete at its symmetry) + the round-3 naming (R6). **A4**
false total order → DEFEATED by the symmetry-monotonicity argument; spawned
R12's machinery demand. **A5** d≥2 overlay missing → redirected by the user
onto the current tree; stands as priced work. **A6** energy-kind freedom →
main agent WRONG; kind is data-fixed (multigroup = indicator-modal),
determination at declaration. **A7** reflective demands mirror-symmetric
quadrature → WITHDRAWN as stated; transformed and then **measured** (guard
4: exact-or-refuse, no interpolation arm — both parties partially right,
tree cleaner than either claimed). **A8** multiphysics at the root →
transformed into the state-fields stage (R9).

Round 2: **A9** stage-1 axis mint vs the leak principle → user sharpened to
T1 (all axes at the method); forced the Library/Materials distinction. **A10**
symmetry filtration is prose → conceded aspirational (issue-tracked); the
user's challenge "find a principled alternative flow criterion" was answered
by the filtration ontology (§2), which SUBSUMES symmetry as the bound.
**A11** deck realization owes a join obligation → measured, guard 4. **A12**
`mixture.as_infinite_medium` layering → conceded, `InfiniteMedium(mixture)`.
**A13** lazy XS defers the state question → fully conceded, R9. **A14**
half-axis → no half-object was intended; the measures-accumulate/
axes-resolve spelling ratified (R5).

Round 3: **A15** mint + preview twin-check drift → fully accepted, R7.
**A16** early grid collapse (lossy-return-type at the root) → fully
accepted, R8. User's closing refinement: the amended stage-1 object is a
declaration; name it `Materials` (R6); the assigned-undeclared guard
appears (R11).

## 8. Supersessions and survivals (plan-authoring §3 — edited in place, nothing dropped)

- **Campaign plan §2.5 (CS1.5 Medium charter)**: SUPERSEDED IN SHAPE — no
  `Medium` class; no `from_medium` arms. Its surviving physics objectives
  (`kernel_and_medium_objectives.md`) map onto the chain: *medium
  expressible* → Materials + overlay; *generator lattice* → the stages as
  generator inputs (§2); *conformity* → guard 3, verbatim; *retirement
  honesty* → binds the arc's every step. The XD-4 three-outcome amendment
  survives as guard 1's shape under R7's preview discipline. The grounding
  census (`scratch/cs15_grounding_census.md`) remains the `[M]` fact base
  where still current — its site counts were already flagged STALE
  (re-censused 2026-08-24; re-census again before designing).
- **The CS1.5′ resume pointer** (campaign plan §5 tail): carries a
  supersession banner pointing here; its pickup duties survive where they
  apply (re-census; #398 as witness; surgical posture).
- **`SNMesh`/`SNDiscretization`**: not dissolved by this charter (R3 open);
  demoted-by-trajectory toward the save-state aggregate; every *machinery*
  role has a named destination stage.
- **CS2**: unchanged in substance — the mint-as-free-function, axes
  non-Optional, the angular axis — but its landing surface is now the
  method head (T1), and its identity-completion work is what converts the
  remaining `is`-checks to space-content checks.
- **CS4c**: unchanged — the apply-arm migration (R13's arms), Riesz legs,
  `DualSpace`, the `dual()` functor.

## 9. Priced open work (not designed here)

1. The d≥2 overlay object (region geometry above raw `mat_map`).
2. Per-stage symmetry groups, declared-vs-computed (aspiration; machinery
   family #152/#166; the mesher's symmetry-preservation score).
3. The container fork + #406 save-state story (+ the cross-problem
   provenance loop: condensed Materials ← a solve).
4. #398's discharge at the head (guard 5) — the arc's natural first
   admissibility landing.
5. The deferred consumer arc: lazy realization with the §3 criterion;
   traversal objects (the `for_shape` precedent generalized).
6. **The data-layer overhaul** (user, 2026-08-25 — the §3 taxonomy):
   the three-way macro-data provenance (GENDF-class / PENDF-class /
   collapsed-from-solve) with a per-kind consistency-check family;
   PENDF → Monte Carlo exclusively; GENDF + collapsed → one
   method-agnostic final object (multigroup MC included). Concrete
   properties resolve lazily; the first heavy consumer of concrete
   energy-structure numbers is Campaign 2's partitioning
   (spectral-radius prediction/reduction, hedged).

## 10. Note to the archivist (when the arc lands — not before)

The Sphinx home is a foundations theory page on problem posing (sibling of
`spaces.rst`); its spine is §2 of this file — the filtration, T1–T3, the
data-vs-solution-space distinction, and the guards schema of §4 with the
worked table. The pages it must cross-link: `spaces.rst` (the collapse pair
IS the backward walk — T3), the condensation/homogenization theory (the
same walk at the physics level), `frame.rst` (the Parseval divisor as the
projection's metric), and the boundary-conditions page (deck
identifications as geometry — R10, with guard 4's exact-or-refuse
certification and its ERR-042/073/074 lineage). The leak principle's
docstring (`material_mesh.py`) should gain its T2 name when touched. Write
the page from the *reasoning*, not the table — the guards schema ("earliest
decidable point; name the last-arriving operand") is one sentence and
regenerates every row.
