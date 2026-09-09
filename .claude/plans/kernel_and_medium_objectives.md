# Kernel + medium — objectives for the joint design (assembly-free)

**STATUS: DRAFT v2 for user red-line (2026-08-20).** v1 (`cs15_objectives.md`,
retired into this file) was scoped to the medium/carrier half. The premise
correction (user, same day) widened it: the carrier's demands are the
OPERATORS' artifact — apply-time polymorphism reading a carrier — so the
design space is the full path **from material data + spatial description to a
posed problem**, and the order is RE-RULED kernel-core-first (campaign plan §0
ruling 3's ⚠⚠). This file states WHAT must be true — outcomes, invariants,
constraints — with no commitment to any type lattice, class name, or
construction mechanism beyond what standing rulings fix. It is the shared
brief for independent design assemblies.

Round-1 designers assemble from THIS FILE + THE TREE ONLY (independence is
the exercise's integrity condition — the census and design records argue
directions in places). Round-2 material (adversaries + synthesis):
`scratch/cs15_grounding_census.md`; `cs15_medium_unweld_design.md` (its §2
`[M]` fact table is authoritative); `scratch/cs15_verification_plan.md`
(parked); the incumbent CONTESTANTS — the charter's §B2 round-5 lattice, the
round-7 medium design, campaign §5's own CS4 means-sketch.

Vocabulary note: "medium" and "kernel" below are lowercase DOMAIN words —
what fills space; the representation-free physics datum of an interaction.
Whether either is a class, a pair, a Protocol, or nothing nominal is what
assemblies decide.

---

## The problem, in domain terms

The infinite-homogeneous eigenproblem — physics whose spatial symmetry is
everything, so after the quotient no spatial structure remains — is posed
today by FABRICATING a discretization (edges `[0,1]`, a coordinate system, an
integration node at `0.5`), because the operator stack decides its behavior at
APPLICATION time by reading mesh data off a carrier, and "uranium everywhere"
had no way to be said to a stack that interrogates every operand for cells.
The fabrication radiates: an ambiguous sentinel (`mesh is None` meaning two
unrelated things), asymmetric method-layer guards (one typed refusal keyed on
the ambiguous sentinel; one bare assert `-O` strips into a deep crash), an
error message naming the wrong case. Beside it, two structural debts of the
same origin: the pair solvers actually consume — (spatial description,
materials-by-id) — has validity conditions nothing checks at its birth; and
the scattering CONCEPT is spelled as two operators (isotropic-energy, angular)
over one underlying transfer datum, a twin at the physics level.

## The demand→home mapping — `[M]` the premise's evidence

Every carrier demand of the homogeneous path, against the home ruling 2
assigns it (measured 2026-08-20; filters in the census + design note §2):

| carrier demand today | true home under construction-time binding |
|---|---|
| `materials` (via `MaterialXSField`) | the kernel's own data (the Mixture's Σ matrices) |
| `ng` | **the space's energy axis** — shape, ALL of it, is the space's; the kernel's data must CONFORM at binding (an agreement guard, construction-time — the user's sharpening: ng is already IN the space) |
| `mat_map` | the assignment — trivial/absent for "uranium everywhere"; genuine pullback data only when a mesh exists |
| `spatial_shape` | the space's |
| `bulk_space` | the space IS the binding argument |
| `volume_measure` (via `IntegratedReactionRate`) | the space's measure — `⟨Σx, φ⟩` in the space's inner product (CS1's `_diagonal_inner_product` already computes exactly this) |

Salvaged `[M]` rows from the parked verification plan
(`scratch/cs15_verification_plan.md` — assembly-independent facts):
(i) interface-position arithmetic over 4,902 shipped interfaces: slab/uniform
bit-exact vs mesh edges; CYL/SPH equal-volume differ by 1 ULP (10 of 4,902,
the `sqrt`/`cbrt` round-trip) — any conformity check is a ~4-ULP-band
question, not `==` and not a loose tolerance. (ii) The D5 byte gate is BLIND
to space weights (bit-identical end-to-end) and LOADED on cell volume (flux
397.946→198.973); `k_inf` is blind to both — the gate-sensitivity split any
assembly's verification leans on.

## Objectives — each with its falsifiable tell

- **O1 — Honest pose.** The homogeneous eigenproblem is posed with ZERO
  fabricated data: everything reaching the operators is a true property of
  the posed physics — Energy ⊗ the spatial quotient point, counting measure,
  per-unit-volume convention. *Tell:* no `[0,1]` edges, no invented node, no
  coordinate system on the path; results bit-identical (D5, 8/8).
  ✅ REMEDIED 2026-09-08 by the CS4c coda — C1 `5caad3d6` / C2 `39e7f32f`. The tell is MET, in both halves: `[M]` the homogeneous path
  constructs no `MaterialMesh` at all (a construction spy over
  `MaterialMesh._init_data` — the one body every surface funnels into —
  counts zero while every consumed object is touched), so there are no
  `[0,1]` edges, no node and no chart; and the byte gate reads 8 of 8 on
  `k_inf`, the flux bytes and both rates, against a fixture captured
  before the campaign and never regenerated. The pose's data is minted
  by `HomogeneousProblem` from the `Mixture` alone (C1); the fabricated
  carrier's factory is deleted (C2). Record:
  `docs/theory/foundations/infinite_medium.rst`, "Development history".

- **O2 — Operators are decided at construction.** How material data becomes
  an operator is decided when the operator is CONSTRUCTED (what was bound),
  never at application (what shape arrived). A bound operator has one domain,
  one codomain, one apply. "Uranium everywhere" is a legal binding with no
  spatial mesh, no carrier, no assignment. The binding VALIDATES kernel-data ↔
  space-axes conformity (ng disagreement is a construction refusal, not an
  apply-time crash). *Tell:* the homogeneous operators build from (material
  data, Energy ⊗ point) alone; for every operator the core rebinds, its P1
  strict-xfail rows delete and no apply-time shape dispatch remains (the
  ledger apportionment CS4a vs CS4b is stated by the assembly, since L/B and
  S-angular's frame ride CS2).

- **O3 — The medium is expressible without a discretization.** "What fills
  space" — a material assignment over a spatial description (full-symmetry
  case; 1-D layered case) — can be stated, validated, consumed with no mesh
  in existence. Its validity conditions (assignment covers its regions; one
  group count; in the coarse-multigroup regime one energy-grid content — C5)
  are established at its birth. Designed AGAINST the landed binding
  signature: for meshed poses the assignment is a binding input, so its home
  follows the binding's shape, not the other way around. *Tell:* the
  homogeneous solve consumes the statement + nothing else; each invalid
  pairing has a typed construction refusal with its own witness.

- **O4 — Leaks are impossible in both directions.** Physics values come from
  kernels; shape and measure from spaces; spatial assignment from the
  medium×mesh pairing — and each is UNASKABLE of the wrong party:
  discretization data is type-absent (the user's "not there, not an error")
  on anything discretization-free; no operator interrogates its operand's
  provenance at apply time. *Tell:* the census's five geometry attributes are
  unreachable on the quotient-posed path; no consumer reads mesh data that
  the space or kernel already owns.

- **O5 — The generator lattice is CS2-ready.** The stages — spatial
  description (geometry/symmetry) → material assignment → discretization →
  method data — stay structurally distinct and separately addressable, so
  CS2 derives EnergyAxis from material content, SpatialAxis from a mesh, the
  quotient spatial axis from the medium's SURVIVING symmetry (C4). *Tell:*
  when CS2 lands, no stage's constructor and no binding signature needs
  re-pointing.

- **O6 — Conformity has a home.** Whether a discretization conforms to a
  medium's physical interfaces is checkable: by construction on the canonical
  path (F2), refusable-with-a-reason where a hand-built mesh meets a stated
  medium; no path CLAIMING the pairing admits silent nonconformity. *Tell:*
  a non-conforming hand-built mesh is refused with a region-naming reason —
  a committed red-capable witness (the `[M]` ULP row above fixes the
  comparison discipline's scale).

- **O7 — One spelling per concept; the fakes retire completely.** The
  scattering concept has ONE representation-free datum — every scattering
  operator (isotropic-energy, angular) is a binding of it; same for fission
  (the χ⊗νΣf dyad) and collision (the coefficient). Regions/interfaces reuse
  or explicitly supersede the shipped spellings (`Region`,
  `StructuredGeometry`, `from_geometry`) — a silent second spelling
  disqualifies. The fabricated path AND its adapters (the degenerate carrier,
  `MaterialXSField`'s meshless admission) retire when their last consumer
  dissolves: tests migrated, doc xrefs re-pointed (`:meth:` refs die
  silently — F5), the ambiguous sentinel gone. *Tell:* retirement greps +
  `dead_references` return only past-tense history; one grep-findable owner
  per datum.
  ✅ REMEDIED 2026-09-08 by the CS4c coda — C1 `5caad3d6` / C2 `39e7f32f` — for the FABRICATED-PATH clause only; the one-spelling-per-datum
  and Region/StructuredGeometry clauses are untouched and O7 stays open on
  them. `[M]` at C2: `MaterialMesh.from_materials` deleted (the homonym
  `EnergyAxis.from_materials` survives and is a different object), the
  facade's meshless-admission prose retired, the ambiguous `mesh is None`
  sentinel now carries ONE meaning (d≥3 axis-native) and is gated as the
  singleton law `mesh is None ⟹ ndim ≥ 3`, 29 test call sites in 11 files
  migrated to one shared genuine-carrier fixture or to the hub, and a
  `not hasattr(cls, "from_materials")` gate over the three-class hierarchy
  makes the retirement unspellable rather than merely done.
  `dead_references` 0 dead / 68 checked at C2.

- **O8 — Refusals are honest.** Whatever refuses — a binding on
  non-conforming data, a method layer on an un-discretized statement — does
  so TYPED, naming the true reason, symmetric across methods, alive under
  `python -O`. *Tell:* the SN bare-assert crash and the wrong-reason
  `.areas` message are unspellable; positive and negative legs per vv#11.

- **O9 — Behavior is preserved.** Solver ENTRY signatures unchanged this
  arc; homogeneous results bit-identical (D5 8/8); SN/diffusion suites green.
  Internal operator call sites MAY re-spell (the CS1-known ~10 construction
  sites of C/IsoS/IsoN2N/F) — §6b completeness per step. *Tell:* scoped
  suites + byte gate; the re-spelled call-site set enumerated in the step
  that moves it.

## Constraints — standing rulings that bound every assembly

- **C1** The CS1 substrate is the vocabulary: spaces own measures;
  `Axis`/`EnergyAxis`/`of_axes` + derived-name identity mint every space.
- **C2** Campaign ruling 2 (2026-08-19, VERBATIM the end-state): kernels are
  representation-free data; a constructor binds Kernel × Frame/Space → the
  fully-bound operator; apply-time overloading retires. Assemblies REALIZE
  this; they do not debate it. C stays a multiplier (space-only, no frame).
- **C3** The collapse doctrine: the quotient point carries the counting
  measure and the per-unit-volume convention.
- **C4** Symmetry monotonicity (round 6): every arrow down the lattice
  preserves or shrinks symmetry; the quotient is licensed above the mesh.
- **C5** The grid-coherence invariant is REGIME-SCOPED (user, 2026-08-20):
  refusal is right for coarse-multigroup (GENDF-class), NOT universal —
  fine data (ACE-class/MC) legitimately unequal; unionization is the modern
  reconciliation. No assembly bakes the refusal as a universal law of media.
  Issue #395 tracks the fine-data arm; the refusal's docstring cites it.
- **C6** `StructuredGeometry` documents "no infinite-medium kind" as a design
  ruling — overturnable only explicitly, never silently.
- **C7** Layer contract: `geometry`/`data` below `transport`; method packages
  above. Everything minted needs a legal home.
- **C8** The CS2 fence, sharpened: no Spatial/Angular axis classes, no ⊕;
  **S-angular's frame mint and the L/B xfail rows ride CS2** (campaign §5's
  own done-when carve-out); the angular `ScatteringOperator`'s design review
  is chartered POST-kernel (user, 2026-08-20: "we will review its design once
  that which currently exists as ScatteringOperator becomes
  ScatteringKernel").
- **C9** Inherited and binding (operator-strategy campaign): the 12
  P1 strict-xfails are the ledger; paired construction; shape symmetry.
- **C10** Scope + process: nothing minted without a consumer; lands on the
  held branch; per-step gates + copy-aside batteries; pyright terminal-1;
  Sphinx `-W` clean.

## Facts still to measure (assemblies may demand these; first one measures)

- The §6b call-site set for re-posing C/IsoS/IsoN2N/F construction (CS1's
  census: ~10 production sites across homogeneous/sn/diffusion — re-verify on
  the day of the edit).
- What `LegendreMomentScattering` + `mat_xs` already provide as kernel-shaped
  data (campaign §5's means says kernels "wrap what already exists" —
  unverified there).
- `ScatteringOperator`'s cross-method arms (the `ScalarFlux` entry, the
  ndarray hatches — keep-rulings #205/#276 at `scattering.py:1172` per §5)
  — what each arm's consumer set is, before any assembly proposes their
  fate.

## Non-goals (doors left open, not built)

Re-routing solver entries; symmetry-group machinery; multi-D/CSG media;
unifying homogenize/condense; the fine-data/unionized-grid arm (an issue,
nothing more); any CS1-substrate change; Campaign 2's pencil/partitioning.

## What a submission must contain

1. The conceptual assembly — concepts, homes, relationships — ≤1 page,
   lossless to a fresh reader.
2. A per-objective discharge table (how each O becomes true; which tell).
3. **The internal slicing**: which operators the kernel core rebinds first,
   where the medium/assignment enters, what of the demoted CS1.5 design
   survives at which point, and the P1-xfail ledger apportionment (CS4a vs
   CS4b). The TOP-LEVEL order is ruled; the slicing is the assembly's.
4. What it deliberately does NOT mint, and why.
5. Blast radius against the shared facts (what moves, what is untouched,
   which existing spellings it reuses/absorbs/supersedes).
6. Self-attack: the two strongest objections, stated as an adversary would.
