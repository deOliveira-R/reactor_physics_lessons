.. _manifolds:

=================================================================
Manifolds: the Point Set, the Orbit Space, and What a Basis Eats
=================================================================

.. contents:: Contents
   :local:
   :depth: 2


.. Machine header — the ``nexus-meta`` schema for this page (PROVISIONAL).
.. Seeded 2026-08-31 at tracker 2.0a of the angular-spaces campaign (#429),
.. under user ruling D0.7. This page owns LEVEL 1 of the three-level stack;
.. ``spaces`` owns levels 2 and 3 and ``discrete_measures`` owns the measure
.. that lives on level 1. The schema is provisional pending a full re-audit
.. of the corpus.

.. dropdown:: Machine header — ``nexus-meta`` schema (PROVISIONAL)
   :color: muted

   .. code-block:: yaml

      module: numerics
      concept: manifolds
      role: "the point-set layer — the manifold M a measure is supported on and a basis function is defined over, its algebra (product, orbit space, membership), the invariant-theoretic derivation that produces an orbit space, the TWO coordinate systems an orbit space honestly has (the invariant chart's codomain and a section's image), and the three-level separation (manifold / fields on it / coefficients) that keeps a FunctionSpace from being mistaken for a domain"
      depends_on: []
      related: [discrete_measures, spaces, frame, spherical_harmonics]
      status: "MINTED, gated, WIRED, and CONSUMED. Two catalogued derivations ship (S^2/O(2)_a for the three AXIAL entries and S^2/<sigma_a> for the three MIRROR axes — six keys, two procedures — plus the derived identity quotient), and a Quotient carries BOTH coordinate systems after the 2026-08-31 two-slot ruling. `Space = str` and its six SPACE_* tags are RETIRED (tracker 2.0c, 2026-09-01): `DiscreteMeasure.support`, `GeneratingMeasure.support`, `UniformMeasure.support`, `ProductMeasure.support`, the `ReferenceMeasure` Protocol and `AngularSymmetry.support` all carry a Manifold, and `Basis.domain` does too (2.1). Tracker 2.1b (2026-09-01) read a SECOND answer off that same slot: `Basis.invariance_group` is DERIVED from `domain` by a match on its TYPE (a Quotient of the sphere -> its `by`; the sphere -> Trivial; anything else -> None), so a basis declares the symmetry its functions HAVE by naming the manifold they EAT. `[M]` 6 of 6 shipped bases answer, the property is @final, and it cost zero subclass edits and no new field. ERR-080's pairing therefore has BOTH operands and is a computable lattice verdict — `[M]` `Trivial contains O2('x')` is False for the slab (it read SO2('x') until #432), while the shipped fold's two halves are literally ONE group object. Nothing CONSUMES that verdict yet: the frame's pairing gate is tracker 2.2, and a gate written on the FRAME's measure would be inert today because that measure still carries the forged S^2. Tracker 2.4 (2026-09-01) gave the axial rotation group its AXIS — `SO2(axis)` beside `Mirror(axis)` — and made the slab's polar rule DECLARE its orbit space: `Quadrature.gauss_legendre(8).measure.support.name == 'S^2/O2_x'`, via the new verb `DiscreteMeasure.on_orbit_space`. That is this page's first PRODUCTION consumer, and it collapsed the registry twin (`AngularSymmetry.support` now calls `SPHERE.quotient`). Tracker 2.3 (2026-09-02) gave the category its ARROWS: `ManifoldMap(domain, codomain, apply)` is a frozen value type, composition `psi @ phi` is refused across mismatched endpoints, and `DiscreteMeasure.pushforward` now READS its target off `phi.codomain` (`new_space=` retired) and refuses a map out of the wrong point set — by manifold VALUE, so the slab's `S^2/O2_x` rule and the chart rule on `[-1,1]`, whose nodes are `np.array_equal`, are told apart. Three arrows are typed: `archimedes(axis)` ([-1,1] x S^1 -> S^2, Archimedes' hat-box, `[M]` the product rule is bit-identical to its retired hand loop on 60 of 60 configurations and its support IS the chart's codomain); the orbit retraction inside `quotient()`; and `barycentre(orbit_space)` (S^2/O(2)_a -> Ball(3), since 1 - norm(mu e_a)^2 = 1 - mu^2 = det P / 4). ERR-080 restated in that vocabulary: it is the barycentre map with a FORGED codomain — `[M]` the forgery's nodes are np.array_equal to the honest map's image and differ only in the type claimed. 2.3 is an ENABLER and repairs nothing: no membership check runs inside a map (that refusal is tracker 2.0b, at measure construction), the forgery arm stays a raw constructor BY DESIGN until 3.4, and `[M]` the ERR-080 gate still declares three xfail(strict=True) rows. Neither the entry's chart nor its section ships. Tracker 3.1 (2026-09-02) gave the CATALOGUE ENTRY its own arrow and the measure that arrow pushes forward: `orbit_coordinates` stores the quotient map's action on the base's ambient coordinates and `Quotient.quotient_map` derives the typed arrow, whose CODOMAIN IS THE ENTRY and never the realization (user-ruled; reading it onto [-1,1] is the axis-blind reading 2.4 made refusable) -- `[M]` H-invariant with a negative leg, pi_a . phi_a = pr_1 bit-exact on 12 of 12, beta_a . pi_a the axial projection on 3 of 3, and the change of variables on level_symmetric(4) reading 4.18879020478639, 1 ULP from 4pi/3. `Quotient.reference` carries pi_* dOmega: LEGENDRE on the three axial entries by Archimedes' hat-box, None on the three mirrors (the weighted disk measure 2 du dv / sqrt(1-u^2-v^2), which no shipped ReferenceMeasure realization spells) and on M/{e} (Lebesgue on the BASE, whose orthogonal system a Manifold does not carry) -- both None user-ruled 2026-09-02. `AngularSymmetry.reference` now READS that field, collapsing the campaign's SECOND Pattern-2 twin after `support` at 2.4; its bare-sphere arm stays, deliberately, because a geometry that spends nothing is handed the BASE. The engine seed is therefore complete: `[M]` 9 of 9 procedure outputs are slots over TWELVE fields (was 6 of 8, then 7 of 9), though all SEVEN quotients of S^2 the catalogue produces still read derived_by='hand'. `[M]` the value arrives by a FUNCTION-scope import of generating_measure (alive 7 of 7 import orders; every module-scope placement dead 7 of 7, at the top and at the bottom of the file alike) while the TYPE rides TYPE_CHECKING -- a guard defers a name and can never carry a value. 3.1 is an ENABLER too: `[M]` `reference` has ONE production reader and `quotient_map` has ZERO, the entry's SECTION still does not ship, and ERR-080 itself is still OPEN with its three xfail(strict=True) rows untouched. Tracker 2.5 (2026-09-02) is the campaign's FRAME-side pre-step and is documented on theory/foundations/frame (frame-moment-space-single-home), not here: both HarmonicFrame doors stopped naming SphericalHarmonicBasis and now demand the two-member TruncatedBasis surface (L + space) — the same key-on-what-it-declares move this page makes for invariance_group — and the SEVEN production sites that re-minted the angular coefficient space from the integer L now READ it off the bound basis. So 3.4's Legendre basis on S^2/O2_a is bindable and its space propagates to every operator end and every moment field by construction. `[M]` 33 of 33 (rule, L) rows are metric-identical to the from_L(L) mint they replace, and the converged slab flux on ERR-080's own gate fixture is array_equal pre/post at L = 0, 1, 2 and 3 — bit-identical even where the answer is wrong, which is what a pre-step owes an xfail(strict=True) gate. A capability, not a repair. THE FUSED COMMIT (2026-09-02, trackers 0.1b + 0.6 + 2.2 + 3.4 + 3.4b) IS THE REPAIR, and ERR-080 is CLOSED. Three objects land on this page. (1) `Quotient.descending_slots` — the isotypic probe, user-ruled onto the ENTRY because fibre-constancy is a theorem about pi and it has two readers; it samples SO(2) at INCOMMENSURATE angles because four right angles generate C_4 and falsely admit m = +-4 at L >= 4 (vv #13), a control that is blind below L = 4. `[M]` about x at L=4 it returns exactly {(l,0)}, 5 real slots of 25; about y/z only 2, because the invariant subspace is one-dimensional per degree and slot-ALIGNED only about the harmonics own polar axis x. The fold sigma-even mask now READS it, `[M]` bit-identical on 15 of 15 (axis, L) rows, and its retired five-direction probe had norms 0.83-0.998 (off the sphere, and refusable after 0.6). (2) `Descent` — the two realizations of Funcs(M/H) as ONE object with the discriminator (downstairs iff the quotient has a classical named basis) as `frame_basis`, which `Quadrature._harmonic_basis` binds; `[M]` the isomorphism is array_equal, max|D| = 0.0, on 7 of 7 sphere rules at L=4, and that BIT tier is a measured constraint on the polynomial SPELLING (no single scipy routine reproduces the harmonics m=0 column: lpmv differs at l=1 by 8.3e-17..1.1e-16, eval_legendre at l>=2 by up to 4.8e-16). The upstairs refusal is AXIS-keyed, not alignment-keyed, because about y/z alignment holds at l <= 1 and L=0 is where every solve mints. (3) `quotient_onto` — G0, ONE predicate: a frame is admissible iff a quotient map measure.support -> basis.domain EXISTS, and its table is the basis pulled back along it. `[M]` all seven shipped pairings measured: identity for slab+Legendre, sphere+harmonics and fold+sigma-even; the entry own pi for Legendre-on-a-sphere-rule; REFUSED for slab+full-harmonics (ERR-080), fold+full-harmonics, and — mathematically admissible, over-refused because invariance_group is a LOWER bound with no axis-parameterised O(2) to declare — Legendre-on-a-sigma_y-fold (GitHub #432). `[M]` end to end on ERR-080 own fixture, against a pinned pre-repair tree: phi is array_equal at L = 0 and 1 (max|D| = 0.0) and moves 7.765 / 3.546 at L = 2, 3 to +4.000000000000; gauss_legendre(16) at scattering_order=4 raised a DenseMetric Penrose ValueError before and returns +4.000000000000 after. NOT closed by it: the membership predicate still does not run at measure construction (2.0b) — a forged measure is still CONSTRUCTIBLE, and what is gone is every path from one to a basis; `fundamental_domain` is still None on every axial entry with zero readers. #432 LANDED the same day (see below). Trackers 2.5 / 3.1 / 2.3 / 2.4 / 2.1b were correctly recorded as enablers, and every "still OPEN / still declares three xfail rows" clause in this page dated before 2026-09-02 was repealed by this commit. #432 (2026-09-02) IS THE NAMING LAW: an orbit space is named by its STABILISER, the largest subgroup of O(3) with its orbits. `SubgroupOfO3.O2(axis)` — the pointwise stabiliser O(2)_a = C_inf,v of a coordinate axis, rotations about it AND mirrors through it — joins the lattice beside SO2(axis) and Mirror(axis); `SubgroupOfO3.orbit_stabiliser` names which group an orbit space is recorded under and moves exactly TWO lattice members (SO2(a) -> O2(a) because R[x]^SO(2)_a = R[x_a, x_b^2+x_c^2] = R[x]^O(2)_a, and SO3 -> O3 because both fix only the radius); every other member is its own stabiliser. `Quotient.__post_init__` refuses a non-maximal `by` (a construction invariant, so `dataclasses.replace(entry, by=SO2("x"))` is refused too) and `_catalogued_quotient` refuses at the door with the theorem, so `_sphere_mod_o2` is a pure derivation with one function-scope import and the catalogue keeps SIX keys for SIX entries. Consequences, all `[M]` 2026-09-02: the three axial entries are named `S^2/O2_a`; `gauss_legendre(8).measure.support.name == 'S^2/O2_x'`, `.quotient_group == O2('x')`, `.space.name == 'L2[S^2/O2_x]'`; `LegendreBasis(L).domain.name == 'S^2/O2_x'` and `.invariance_group == O2(axis)` — the FULL group the P_l have, no longer a strict lower bound — with `LegendreSpace.from_L` READING the name off the basis's domain (`legendre_space(S^2/O2_x)`); GEOMETRY_ANGULAR_SYMMETRY["slab"/"sphere"].continuous_isotropy = O2("x") (user-ruled: with O(2)_x spent the recorded residual Mirror("x") is exactly G/G^0, where under SO(2)_x the true residual is the Klein four-group and the recorded mirror is half of it). Every axial relation against a FINITE group is COMPUTED from that group's realization (from the realization; one absolute element band `_ELEMENT_ATOL` = 1e-9; SO(2)_a = O(2)_a intersect SO(3) by composition, no proper_only flag) rather than tabulated — `[M]` the tabulated arm it replaced answered `SO2('x') not-contains C_1` while `SO2('x') contains Trivial`, one group under two spellings and two answers, with a committed test pinning the wrong one. THE OVER-REFUSAL AT #432 IS GONE: `[M]` `GalerkinFrame(LegendreBasis(L), folded_product(4,8).measure)` constructs at L = 0,2,4,6 with a (16, L+1) table and the arrow S^2/sigma_y -> S^2/O2_x, the isotropic field reading exactly 4pi at l=0 (bit-identical to sum(weights)) and <= 1.42e-15 at l >= 1; the NEGATIVE leg `axis="y"` on the same fold is still refused, and on a sigma_x-folded rule `axis="x"` is refused while `axis="z"` is admitted. NOTHING NUMERICAL MOVED: `[M]` stage 0 of quadrature selection is identical on 24 of 24 (geometry x rule) rows against a pinned pre-change tree, and the invariance/containment compatibility law re-runs at 0 violations over 450 (edge x fixture) pairs over 18 groups, with the 15-group control reproducing its recorded 342/0 exactly. The walk's report SIMPLIFIES rather than grows: `[M]` gauss_legendre(8) reports {O2_x, sigma_x} where it reported {SO2_x, sigma_x, sigma_y, sigma_z}, because sigma_y and sigma_z are absorbed by O(2)_x while sigma_x flips the axis and is absorbed by nothing. TRACKER 2.2b (2026-09-02, user-ruled) MOVES THE INVARIANCE QUESTION ONTO THE ORBIT SPACE and gives the registry its Gamma slot. An isometry descends to M/H iff it NORMALISES H, decided exactly per family (`SubgroupOfO3.is_normalised_by` / `normalises`): a finite H by conjugating its element set, SO(2)_a / O(2)_a / D_inf_h by g e_a = +-e_a, and a CONTINUOUS G through the Lie condition on its identity component plus its coset representatives -- never sampled, and `[M]` the four-right-angle sample over-certifies on 2 of 8 (G, H) pairs, ERR-072 recurring in a new predicate. `Quotient.lift` is a right inverse of the quotient map, at this step derived per family (the orbit barycentre for the axial entries -- EQUIVARIANT, not a section, which is all an induced action needs; the hemisphere section for a mirror; the identity for the trivial entry -- ONE formula since R4, see below), `[M]` pi . lift = id to 0.000e+00 on all three; `Quotient.induced_action(motion)` is the arrow [p] -> [g p] and REFUSES a motion outside the normaliser (`[M]` C_4 about z on S^2/sigma_y). Invariance has ONE kernel: a bare support is asked on the trivial orbit space R^3/{e} (`[M]` `_ambient_orbit_space().name == 'spatial_R3/Trivial'`, dim 3 -- the ambient space and not the sphere, because every barycentre and every zero-padded node lands OFF S^2), and `[M]` that reduction agrees with the ambient reading on 144 of 144 (sphere rule x candidate group) rows (recorded as 150 of 150 on 2026-09-02; only the DENOMINATOR moved, it being the size of a COMPUTED candidate set). Since R2 of #434 (2026-09-03) that kernel lives in `orpheus.numerics.invariance` and the verbs live on `DiscreteMeasure` -- `is_invariant_under`, `certificate_under`, `permutation_under`, `singular_set_under`, `symmetry_groups` -- and `SubgroupOfO3.is_invariant` is DELETED, no facade. `_polar_axis_of` and `_invariance_on_points` are RETIRED; `_embedded_nodes` reads the entry via `ambient_representatives` (RENAMED from `section_coordinates` at the elegance review: the axial arm returns a BARYCENTRE, not a representative, so the old name promised ERR-080's own forged codomain; renamed again to `orbit_barycentres` at R4, see below) and is still array_equal to `barycentre` on 12 of 12 rows. The registry stage 0 is now ONE expression -- Gamma contains `spent_group(D, X)`, what the descent arrow `quotient_onto(D, X)` SPENDS ({e} for the identity, target.by for a fold of the base, refused naming the missing work for the induced map between two quotients) -- over the SAME arrow a frame G0 reads, and stage 1 asks Gamma-invariance ON X. Reading `X.by` instead would have REFUSED the geometry's own domain (`[M]` the slab's sigma_x does not contain O(2)_x). `[M]` the shipped cylindrical fold (folded_product(4,8), S^2/sigma_y) is ADMITTED at both stages for cylinder and cartesian2d where it was refused at both; stage-0 refusals move 12 -> 10 of 20 (constructor x geometry) pairs with no pair moving the other way; the FOLD arm has NO witness on slab/sphere, since `[M]` no shipped orbit space is a proper further quotient of S^2/O(2)_x (the only arrow out of it is the identity, which spends {e}). NOTHING ELSE MOVED: `[M]` gauss_legendre(8) 0 of 15 candidate groups change, product(4,8) 0 of 23, folded_product(4,8) 4 of 21 (sigma_y, C_2, D_1h, D_2h, all False -> True), walk(fold) {sigma_x, sigma_z} -> {D_2h} with walk(slab) and walk(product) unchanged and brute-force agreement 6 of 6 both sides, and the compatibility law re-runs at 0 violations over 342 and 450 (edge x fixture) pairs. The spent-group door refuses (M/H)/G for G contained in H with the theorem, with ONE exception it names: the trivial group is admitted on every base as the identity ENTRY (`[M]` at this step `S^2/sigma_y/Trivial` -- a SECOND object for one orbit space, which R4 collapsed to the entry itself, see below -- and the fold measure's quotient by Trivial returns all 16 nodes on it) -- acting trivially is not the same defect as being spelled twice. `certificate_under` follows the same route and the plan's section II.11 lead is CLOSED: `[M]` certificate_under(gauss_legendre(8), sigma_x) AND the same call on the BARE chart rule gauss_legendre_on_mu(8) both return 2 permutations where both were refused by SHAPE. What survives is the MESSAGE at measure.py/symmetry.py, now a THREE-arm disjunction wearing two-arm text -- its new third arm is 'the group does not normalise the spent group' (reported, not repaired; a production edit). `Quadrature.ordinate_permutation` moved onto the same kernel, so the tree has ONE notion of 'does this isometry permute the ordinates': `[M]` sigma_y on the fold now yields the IDENTITY permutation where it yielded None, and every other (rule x mirror) cell over four rules is unchanged. `is_normalised_by` asks the motion's LINEAR part rather than refusing a translated motion -- a point group acts on directions and a translation does not move one, which is the convention ordinate_permutation already ran. GitHub #370 gap 2 (stage 0 cannot match a quotient support) is CLOSED, the way that issue demanded -- by the lattice arrow, not by widening a tag; gap 1 stands, `[M]` folded_product(4,8).measure.exactness is still None. The fold measure invariance_group stays None BY DESIGN: the stored slot is a statement about the representatives, the computed predicate is about the orbits. #434 R1 (2026-09-03, UNCOMMITTED in the working tree when this was written — trust git) MAKES EVERY QUESTION ABOUT A GROUP A COMPUTATION ON ITS REALIZATION. so(3) is simple and 3-dimensional, so its subalgebras are {0}, one line R[a]_x per axis, and so(3) itself — never dimension 2 — and a closed subgroup of O(3) is therefore exactly (identity component, one representative per connected component): `IdentityComponent` (a tuple of skew generators) and `Realization` (component + representatives, identity first). `contains`, `is_normalised_by`, `normalises`, `identity_component`, `dim`, `generic_images` and "does G0 fix these nodes" are each ONE body on that pair, and NO relation between two groups is written down anywhere: `_NAMED_LATTICE` (8 hand edges), `_named_contains`, `_contains` (109 lines, 28 tag-dispatch sites by AST: 24 isinstance + 4 `is _NamedSubgroup.X`; module-wide the same predicate falls 86 -> 31), `_finite_contains`, `_fixes_axis`, `_axial_contains`, `_rotation_generator`, `_maps_axis_to_itself`, `_continuous_decomposition`, `_fixes_every_point`, `_identity_component_normalises` (five per-family arms), `_is_axis_supported`, `_is_origin_supported`, `SubgroupTag`, `is_subgroup_of` and `_GROUP_CACHE` are RETIRED. `[M]` against a pinned pre-carve tree over 27 spellings (26 distinct groups): `contains` 0 of 729 ordered pairs moved, `normalises` 0 of 729, `is_invariant` 0 of 270 (10 rules x 27 groups), the walk 0 of 10 rules, and the vv-#15 compatibility law 0 violations on both trees at every denominator it has been run at (57 edges/342 pairs, 75/450, and the widened 175/1750). THREE answers move, all intended: `identity_component` is now Trivial for every finite member (17 of 27 spellings; it returned the group ITSELF before, contradicting its own docstring's "its orbits are connected" on O_h, and invisible because the property had ZERO readers — the two sites needing it destructured `_continuous_decomposition`); `Cn(1)` normalises to the `Trivial` tag on the type, so one group has one spelling (before: each contained the other, they compared unequal, `_maximal` dropped BOTH, hash differed, and `SPHERE.quotient(Cn(1))` answered "no catalogue entry"); and `dim` in {0,1,3} is NEW, with no production consumer until R4 reads it for the orbit-space dimension law. `SubgroupOfO3` is a frozen dataclass (`g._tag = ...` raised nothing before and moved `hash(quotient)` under three memos); `orbit_stabiliser` is structural (finite -> self; dim 3 -> O(3); torus about a -> O(2)_a if it contains self, else self if self contains it, else a NAMED refusal no shipped member reaches). `_MEMBERSHIP_ATOL` is renamed `_ELEMENT_ATOL` (1e-9, the one element-level band) so the identically-spelled 1e-12 POINT band in manifold.py no longer makes one grep answer twice. #434 R4 (2026-09-03, UNCOMMITTED in the working tree when this was written — trust git) MAKES THE LIFT A DERIVATION OUTPUT AND AN ORBIT SPACE'S DIMENSION A THEOREM. Every catalogued entry's lift is ONE formula, the Reynolds projector P_H = int_H rho(g) dg onto H's fixed subspace read from the chart's side: mu -> mu e_a for S^2/O2_a (the centre of the constant-mu circle, today's barycentre, unchanged), (x_b, x_c) -> (0, x_b, x_c) for S^2/sigma_a (NEW - the midpoint of {p, sigma_a p}, inside the ball, on the sphere only on the equator; until now the mirror entry lifted through a HEMISPHERE SECTION with a square root and a rho > 1 refusal, both retired), the identity for M/{e}. `Quotient.lift_coordinates` and `lift_codomain` are REQUIRED fields beside `orbit_coordinates` (a seventh entry cannot forget the lift; until now `lift` was a three-arm branch on the group's tag whose own message read 'add it to Quotient.lift'), and ONE helper `_coordinate_chart(columns, ambient)` returns the chart and its lift as a pair, so embed . select IS the projector by construction. `[M]` against a reference built from the group's REALIZED matrices (SVD null space of {X; r-I}, then B B^T) and never from a column index: array_equal on 8 of 8 constructible entries x 41 seeded unit vectors, max|D| = 0.000e+00, with P_H a 0/1 DIAGONAL on every one (hence bit-exactly idempotent and symmetric); against the finite group's own MEAN, array_equal on all three mirror entries; against a 16-point trapezoid over the orbit circle, 3.331e-16 on the three axial ones (and MORE points is worse - 2.831e-14 at n=1024, since the trapezoid is exact for n >= 3). `ambient_representatives` -> `orbit_barycentres`: ONE concept on both coordinate widths, where the ambient arm used to pass a fold's nodes through AS representatives - `[M]` a fold's (x, y, z) now comes back as (x, 0, z) - and `barycentre(entry)` is defined on EVERY entry, its old 'a mirror orbit has no axis to lie on' refusal narrowed to a manifold that is not a Quotient at all. The DIMENSION LAW dim(M/H) = dim M - dim(generic orbit), with the orbit's dimension the rank of {X p : X in h} at a generic point and NOT dim H (O(3) on S^2 has dim 3 and a 2-dimensional generic orbit, so dim S^2/O(3) = 0 - GitHub #440; SO(3) on R^3 likewise gives 1), is enforced in `__post_init__`, which now carries FOUR clauses (stabiliser, dimension, the lift codomain's ambient width, the fundamental domain), each with an input only IT rejects (the fd clause's witness is now a half-meridian against the disk, because its old one - a hemisphere against [-1,1] - is caught one clause earlier; and `[M]` SO2('x') PASSES the dimension clause, since a group and its stabiliser have the same orbits, which is why clause 1 cannot fold into clause 2). The generic orbit dimension is the MAXIMUM over a probe SET (9 seeded rows for S^2, 4 for a flat base), not the value at one point: orbit dimension is upper semicontinuous, and `[M]` with a single probe row placed ON the axis the one-point spelling both REFUSED the honest S^2/O2_z and ADMITTED the disk forgery. `lift_codomain` is COMPARED where the two coordinate maps are not - `[M]` with it excluded, replace(entry, lift_codomain=SPHERE) compared EQUAL to the catalogue entry and barycentre's memo then answered for both, ERR-080's shape re-minted by the field built to refuse it. `[M]` before it a forged S^2/O2_z realized on the DISK and a forged S^2/sigma_x realized on [-1,1] both CONSTRUCTED and compared unequal to the entry they claim to be - ERR-080's defect class one field over from the one #432 closed. (M/H)/{e} IS M/H: `[M]` until now SPHERE.quotient(Mirror('y')).quotient(Trivial).name was 'S^2/sigma_y/Trivial', a second object for one orbit space inside the spent-group door's own exception, and no test pinned the string; it is now the entry BY IDENTITY. `SubgroupOfO3.is_trivial` replaces five `name == "Trivial"` string compares (3 in manifold.py, 2 in basis/descent.py). THE INVARIANCE KERNEL IS MODE-12 BLIND TO ALL OF IT - every downstream answer is read through `orbit_coordinates`, which is exactly the column selection P_H re-writes, so `[M]` 0 of 9925 answers of R1's behaviour grid moved and chart(g.P p) is array_equal to chart(g.p) on every normalising motion (200 seeded vectors x 3 motions x 2 entries). The gates therefore live at the AMBIENT tier, where the discriminator between the retired section and the projector is O(1) (its supremum is EXACTLY 1, attained at the pole; `[M]` 9.748e-01 / 9.932e-01 / 9.953e-01 sampled over 41 vectors), and the round trip pi . lambda = id ships as a DECLARED BLIND leg because it holds for the section and the projector alike. ONE answer moves and it is a strengthening: the reference harness's mirror partner map on a sigma_y-folded rule returns the IDENTITY permutation where it used to raise ('a node's mirror image misses the node set by 1.155e+00 / 1.189e+00'), agreeing with `ordinate_permutation` where the two contradicted each other - `[M]` 31 of 33 (rule x axis) rows unchanged over 11 shipped rules, both folded call sites in the tree pass axis='x', and the projection is injective on every fold (min chart separation 1.155e+00 / 4.403e-01 / 2.751e-01 / 1.510e-01 at folded_product (2,4)/(4,6)/(4,8)/(8,8)). The engine-seed compliance moves to `[M]` 10 of 10 procedure outputs over FOURTEEN fields (was 9 of 9 over twelve). `[M]` tests/numerics/test_manifold.py collects 240 rows, 102 of them the nine TestR4* classes, all @pytest.mark.foundation and none carrying verifies(...) R2 OF #434 (2026-09-03) REVERSED THE IMPORT DIRECTION: this module now imports `symmetry` at MODULE scope (AXIS_INDEX, AXIS_LETTER, SubgroupOfO3 -- the axis table moved BACK to symmetry.py, its home until 2026-09-02, because reversing the other edge without it closes a 2-cycle: `[M]` shadow-package probe, one fresh interpreter per (variant, entry point), 10 of 10 clean shipped vs 3 of 10 with the old axis home, and `import orpheus` alone stays GREEN under the broken variant). `Quotient.by` is annotated with the real class, `_trivial_group` is RETIRED, and every group member this module reads is an ordinary read -- `[M]` 10 distinct over 24 sites at the pre-carve commit, by AST on a `group`/`.by` receiver (the count is predicate-bound and has been reported as three, seven and nine at different dates; the pre-carve module docstring's own enumeration is ten). The kernel's step 2 (`G subseteq H => True`) is DELETED as an optimisation the closure re-proves: `[M]` it would have fired on 28 (rule x group) rows over 11 shipped rules with 0 disagreements, and the predicate moved 0 of 330 rows on a fixed 30-spelling group list against a pinned pre-carve tree. The candidate set is read off the orbit BARYCENTRES, not the stored node width: `[M]` the walk on gauss_legendre(2/8/16) moves {O2_x, sigma_x} -> {D_2h, O2_x} and on folded_product(4,6) {D_1h, sigma_x} -> {D_2h}, both strengthenings (D_2h contains both dropped answers), and folded_product(4,8)'s candidate set falls 20 -> 18. #434 R3 (2026-09-03, UNCOMMITTED in the working tree when this was written -- trust git) SPLITS THE REGISTRY LEDGER INTO THREE FACTS AND MAKES STAGE 0 TOTAL. `AngularSymmetry(spent, unspent, owed)`: K the stabiliser the dimensional reduction integrates away (it alone derives `support` and `reference`), Gamma the FINITE symmetry the solution still HAS in the geometry's local frame (NEW -- the fold licence), R the reflection closure a reflecting face is owed. The `G^0` / `Gamma = G/G^0` language is RETIRED: the pair was never a factorisation, `[M]` the slab's spent O(2)_x is DISCONNECTED. Stage 0 is the descent arrow S^2/K -> X PLUS a coverage test H subseteq Gamma K on the group H the rule's support was folded by, decided by `SubgroupOfO3.is_subset_of_product(gamma=, kappa=)` (keyword-only; body on `Realization`) -- Gamma K is a finite union of closed cosets, so H^0 lands in the coset containing e, which IS K, and each representative r needs some gamma with gamma^-1 r in K; both implications reverse. TOTAL: no join, no coset space, no arm that raises, so `manifold.spent_group` and its NotImplementedError are RETIRED (D3) and the equality short circuit is unnecessary (`[M]` the slab's own rule is O(2)_x subseteq {e}.O(2)_x). `__post_init__` refuses a continuous `unspent`/`owed`, which is what makes totality a theorem. `AngularSymmetry.domain_refusal(measure) -> str | None` carries the ONE failing clause and `admits_domain` is its `is None` -- `[M]` over 4 geometries x 7 rules the 17 stage-0 refusals split 14 arrow-only / 3 coverage-only / 0 both, so the retired disjunctive message named a SATISFIED fact on all 17. The table: slab and sphere (O2_x, {e}, sigma_x); cylinder (Trivial, D_1h, D_2h) -- `[M]` Dnh(1) realizes as exactly {e, sigma_y, sigma_z, C_2^x}, the plan's [R] row said Mirror('y') and the opener refuted it on the local frame eta = Omega.rhat, xi = Omega.phihat, mu = Omega.zhat (columns 0,1,2): an axisymmetric cylinder is even under sigma_y (xi -> -xi, phi -> -phi) AND under sigma_z (mu -> -mu, z-uniform, mu entering only through |eta| = sqrt(1-mu^2)); cartesian2d (Trivial, sigma_z, D_2h). `[M]` admissions: the sigma_y fold is REFUSED on cartesian2d (D1 / ERR-081 -- 2 of 4 (sign mu_x, sign mu_y) sweep quadrants empty, all 16 nodes at mu_y >= +0.194) and admitted on the cylinder; a sigma_z fold is admitted on both; the slab's own rule is admitted; a 1-D rule is refused for both 2-D geometries by the identity-component conjunct. NOTHING SELECTED MOVED: `[M]` 48 of 48 (geometry x target degree) rows in the frozen selection baseline keep their chosen spec name, parameters and node count, while 96 `domain mismatch` strings were re-worded and 8 of 8 `symmetry mismatch` strings are byte-identical. `[M]` the product is not either factor: O(2)_x subseteq O_h.SO(2)_x while neither factor contains it (197 such triples), and with Gamma = {e} the predicate is bit-equal to `contains` on 441 of 441 pairs -- but NO SHIPPED GEOMETRY exercises the product structure (every row has K or Gamma trivial), so the gate uses the geometry-free witness. The two R2 frozen baselines moved from untracked scratch/ into tests/numerics/data/. `vv-principles` gained anti-patterns #33 (a fact recorded for one job spent on another) and #34 (a brute-force control credited on its name -- check independence by alpha-normalised AST)."


This page develops the **point-set layer** — the thing a measure is
supported *on*, the thing a basis function is defined *over*, and the
thing a quotient by a symmetry group produces. It is the layer *below*
:doc:`/theory/foundations/spaces` (which types the vector space a
discrete field lives in) and *below*
:doc:`/theory/foundations/discrete_measures` (which types the weighted
atom list). Until 2026-08 the corpus had no object for it at all: level
1 was a ``str``.

The organizing claim is one sentence: **a manifold is a first-class
value with an algebra — product, orbit space, membership — and it is a
different object from the function space of fields defined on it.**
Everything on this page follows from taking that seriously: why a
1-dimensional quadrature declaring its nodes to live on :math:`S^2` is
a forgery that a type can refuse, why ``f"{a} × {b}"`` was already
performing a product, why an orbit space is an **orbifold** and not a
quotient manifold, and why :math:`\det P = 4(1-\mu^2)` is the same
polynomial three times over.

.. warning::

   **"Manifold" is the second thing in this corpus with that name, and
   the older one is unrelated.** `[M]` counted on the tree as it stood
   before this page: **21** occurrences of the word in
   ``docs/theory/``, of which **14** are the
   S\ :sub:`N` *solution manifold* — the affine set
   :math:`\psi^\star + \ker A` that a singular loss operator admits
   instead of a unique solution
   (:doc:`/theory/foundations/field_algebra`,
   :doc:`/theory/methods/sn/cartesian_multid`). That object is a coset
   in a *vector* space, reached by a splitting's gauge freedom. *This*
   page's manifold is a point set in **Man**, reached by a chart. They
   share no machinery and no type; only the word.

   The discriminator when reading: a *solution* manifold is always
   qualified by the word "solution" or by :math:`\ker A`, and it is
   never an argument to anything. A :class:`Manifold
   <orpheus.numerics.manifold.Manifold>` is always the thing something
   else is *defined over*.

.. note::

   **Three words for the same level-1 object, all standard, all kept.**
   A measure calls it its **support**
   (:attr:`DiscreteMeasure.support
   <orpheus.numerics.measure.DiscreteMeasure.support>`); a basis
   function calls it its **domain**; the quadrature registry calls it
   the angular **domain** :math:`S^2/K`, the orbit space of the
   stabiliser its dimensional reduction SPENT
   (:attr:`AngularSymmetry.support
   <orpheus.numerics.quadrature.registry.AngularSymmetry.support>`).
   These are not twins — *support of a measure* and *domain of a
   function* are both correct usage for the same manifold, and
   category-theoretically ``dom`` is the source of a morphism in both
   cases (in **Man** for a basis function, in **Vect** for an
   operator). What was missing was not a word; it was the object all
   three words name.

   ⚠ ``support`` is nonetheless a **misnomer of a different kind**, and
   the corpus already says so: `[M]`
   ``gauss_legendre(8).measure.support`` is ``'[-1,1]'`` while
   :math:`\operatorname{supp}(\mu)` is 8 points. The tag names the
   ambient manifold, not the support of the measure. Renaming it is
   tracked with the migration (:ref:`manifold-seams`); this page uses
   *manifold* for the object and quotes ``support`` when it means the
   slot.

.. admonition:: Key Facts (the point-set layer)
   :class: tip

   - **Three levels, and the tree named two.** The manifold :math:`M`;
     the fields on it :math:`L^2(M)`, which is what a
     :class:`~orpheus.numerics.space.FunctionSpace` is; and the
     coefficients :math:`\mathbb{R}^K`, which is what a
     :class:`~orpheus.numerics.basis.base.Basis`'s shape is. A basis
     function **eats a point of** :math:`M` — that is why a
     ``FunctionSpace`` cannot be a basis's domain
     (:ref:`manifold-three-levels`).
   - **The level-2 check passed; the level-1 check had nothing to
     check.** That is why :ref:`ERR-080 <manifold-err-080>` survived.
     On a slab the frame's arrow ``measure.space → basis.space``
     was between two well-formed spaces — `[M]` ``L2[S^2]`` of shape
     :math:`(8,)` into ``spherical_harmonic_space`` of shape
     :math:`(3,5)` — while the nodes it carried were not points of the
     manifold the basis needed. ✅ Since 2026-09-02 the level-1 check
     exists and is ONE predicate: *does a quotient map*
     ``measure.support -> basis.domain`` *exist?* `[M]` the same frame
     now reads ``L2[S^2/O2_x]`` :math:`(8,)` into
     ``legendre_space(S^2/O2_x)`` :math:`(3,)`, with the arrow the
     identity (:ref:`manifold-g0-descent-arrow`).
   - **Membership is what makes a support claim falsifiable.**
     :meth:`Manifold.contains
     <orpheus.numerics.manifold.Manifold.contains>` is the defining
     equation of the member evaluated on the candidate nodes. `[M]` on
     ``gauss_legendre(8).angular_frame(2)`` the production measure
     declares ``support='S^2'`` over rows whose norms are
     :math:`0.1834 \ldots 0.9603` — **0 of 8** within :math:`10^{-12}`
     of 1 — and :class:`Sphere <orpheus.numerics.manifold.Sphere>`
     refuses them, three hops upstream of the wrong answer
     (:ref:`manifold-err-080`).
   - ✅ **The type is WIRED** (tracker 2.0c, 2026-09-01): ``Space = str``
     and its six ``SPACE_*`` tags are retired, and every measure in the
     tree carries a ``Manifold``. ⛔ **The membership PREDICATE is still
     not enforced at construction** (tracker 2.0b): :meth:`contains`
     ships and is gated, but nothing calls it on the way in, so `[M]` a
     forged :math:`(\mu, 0, 0)` measure declaring ``support=SPHERE`` is
     **still constructible today**. ✅ **ERR-080 is nonetheless CLOSED
     (2026-09-02)**, because the refusal that closed it is not this one:
     a basis rejects a non-unit direction and a frame rejects the
     pairing, so a forged measure is *unusable* rather than
     *unspellable*. This bullet read *"ERR-080 is open — its refusal is
     tracker 2.0b plus the fused fix step"* until then; the fused fix
     step is what landed, and 2.0b is still owed
     (:ref:`manifold-seams`).
   - ✅ **…and CONSUMED** (tracker 2.4, 2026-09-01): the slab's polar
     quadrature now *declares* the orbit space it lives on. `[M]`
     ``Quadrature.gauss_legendre(8).measure.support.name`` is
     ``'S^2/O2_x'`` and its induced space is ``'L2[S^2/O2_x]'``, with
     nodes and weights **bit-identical** to the chart-level rule
     (``np.array_equal`` on both). The declaration is a repair, not
     wiring: `[M]` before it, an 8-node slab **angular** space and an
     8-node **spatial** rule on :math:`[-1,1]` compared ``==`` *and*
     hash-equal (:ref:`manifold-orbit-space-declaration`).
   - ⭐ **A basis declares the symmetry its functions HAVE by naming the
     manifold they EAT** (tracker 2.1b, 2026-09-01). A function on
     :math:`M/H`, pulled back to :math:`M`, *is* an
     :math:`H`-invariant function — so for FUNCTIONS the group a basis
     HAS and the group its domain SPENT are one property, and
     :attr:`Basis.invariance_group
     <orpheus.numerics.basis.base.Basis.invariance_group>` is a
     ``match`` on ``domain``: stored nowhere, ``@final``, `[M]` **6 of
     6** shipped bases answering with **0** subclass edits. For a POINT
     SET they come apart, which is why a measure needs **two** slots:
     `[M]` the slab's rule HAS ``Mirror('x')`` while it SPENT
     ``O2('x')``, and the :math:`\sigma_y` fold HAS **nothing**,
     because folding destroys the closure it spends
     (:ref:`manifold-has-versus-spent`).
   - ✅ **The pairing is a REFUSAL** (tracker 2.2, 2026-09-02) — and it
     is one predicate, not the containment it was first stated as. A
     frame is admissible iff a quotient map
     ``measure.support -> basis.domain`` EXISTS
     (:func:`~orpheus.numerics.manifold.quotient_onto`), and its table
     is the basis pulled back along that arrow. `[M]` all seven shipped
     pairings measured: the four legitimate ones admit (three by
     identity, one by the entry's own :math:`\pi` — a Legendre basis on
     a full-sphere rule), and ERR-080's slab-plus-full-harmonics is
     exactly what it refuses. ⚠ It also refuses one *mathematically
     ⛔ This bullet read *"⚠ It also refuses one mathematically
     admissible pairing — Legendre on a* :math:`\sigma_y` *fold — because
     the derived* ``invariance_group`` *is a strict LOWER bound and no
     axis-parameterised* :math:`O(2)` *exists to declare (GitHub #432)"*
     until 2026-09-02. **#432 landed the same day** and the pairing is
     now ADMITTED: the entry is named by its stabiliser
     :math:`O(2)_a`, so ``invariance_group`` is the FULL group and
     :math:`\sigma_y \in O(2)_x` gives the induced arrow. `[M]`
     ``GalerkinFrame(LegendreBasis(L), folded_product(4,8).measure)``
     constructs with a :math:`(16, L{+}1)` table
     (:ref:`manifold-orbit-space-stabiliser`).
     ⭐ Why an arrow and not the containment: the containment IS the
     :math:`K \subseteq H` arm, and it cannot express the other two
     cases (:ref:`manifold-g0-descent-arrow`).
   - ⭐ **What DESCENDS is decidable, and the entry decides it**
     (2026-09-02). :meth:`Quotient.descending_slots
     <orpheus.numerics.manifold.Quotient.descending_slots>` asks which
     slots of a basis on the base are constant on :math:`\pi`'s fibres,
     sampled at generic points and at their images under the group's
     *generic* elements — for :math:`SO(2)` at INCOMMENSURATE angles,
     because `[M]` four right angles generate :math:`C_4` and falsely
     admit the :math:`m = \pm4` slots at :math:`L \ge 4`
     (``vv-principles`` #13), and the control for that is blind below
     :math:`L = 4`. `[M]` about :math:`x` at :math:`L = 4` it returns
     exactly :math:`\{(\ell, 0)\}` — **5 real slots of 25**; about
     :math:`y`/:math:`z` only 2, because the invariant subspace is
     one-dimensional per degree but not slot-ALIGNED off the harmonics'
     own polar axis. Two readers share it, which is why it lives on the
     entry: the descent, and the fold's :math:`\sigma`-even sub-basis
     (`[M]` bit-identical through it, **15 of 15** rows)
     (:ref:`manifold-descending-slots`).
   - ⭐⭐ **The descended space has TWO honest realizations, and the
     ruling names which one a frame binds.** Upstairs is the
     :math:`H`-invariant subspace of a basis on the base (the
     :math:`m = 0` column; the :math:`\sigma`-even slots); downstairs is
     the quotient's own classical basis when it has one
     (:math:`\{P_\ell(\mu)\}` on :math:`S^2/O(2)_a`).
     :class:`~orpheus.numerics.basis.descent.Descent` is the pair as ONE
     object with the discriminator on it, and the isomorphism is
     checkable **at the bit** — `[M]` :math:`\max\lvert\Delta\rvert =
     0.0` on **7 of 7** shipped sphere rules at :math:`L = 4`, which is
     a measured constraint on how the polynomial is SPELLED and not an
     accident (:ref:`manifold-descent`).
   - ⭐ **The rotation axis is a PARAMETER, not a convention — because
     the tree carries two poles.** :math:`SO(2)` left the parameter-free
     enum on 2026-09-01 and became ``SO2(axis)``, exactly as the
     reflection had on 2026-08-02. `[M]` the real spherical-harmonic
     basis takes ``cos θ = μ_x`` while a product rule's polar factor is
     :math:`\mu_z`, and **one** Gauss–Legendre rule serves both roles —
     so the group a marginal was quotiented by cannot be spelled without
     naming its axis (:ref:`manifold-so2-axis-is-a-parameter`).
   - ⭐ **A map carries its two point sets, so a codomain cannot be
     forged at the call site** (tracker 2.3, 2026-09-02).
     :class:`~orpheus.numerics.manifold.ManifoldMap` gives the
     category its **arrows**; ``pushforward`` reads its target off
     ``φ.codomain`` (``new_space=`` retired) and refuses a map whose
     ``domain`` is not the measure's support — by manifold VALUE, so
     `[M]` the slab's rule and the chart rule, whose nodes are
     ``np.array_equal``, are told apart. Three arrows ship: the
     Archimedes chart, whose codomain the product rule now READS
     (`[M]` bit-identical to the retired hand loop on **60 of 60**
     configurations); the orbit retraction inside ``quotient()``; and
     the orbit **barycentre**
     :math:`\mu \mapsto \mu\,\hat e_a`, whose honest codomain is
     ``Ball(3)`` because :math:`1-\lVert\mu\hat e_a\rVert^2 = 1-\mu^2
     = \tfrac14\det P`. ⟹ **ERR-080 is that map with a forged
     codomain** — `[M]` the forgery's nodes are ``np.array_equal`` to
     the honest map's image and differ only in the type claimed
     (:ref:`manifold-arrows`).
   - ⭐ **The catalogue entry gets its OWN arrow, and the measure that
     arrow pushes forward** (tracker 3.1, 2026-09-02). The quotient map
     :math:`\pi : M \to M/H` is the invariant tuple read as a function
     of a point — ``orbit_coordinates`` stores its action,
     :attr:`Quotient.quotient_map
     <orpheus.numerics.manifold.Quotient.quotient_map>` derives the
     typed arrow, and its **codomain is the ENTRY, never the**
     ``realization`` (user ruling): reading it onto :math:`[-1,1]` is
     the axis-blind reading tracker 2.4 made refusable. Four laws, all
     `[M]` bit-exact — :math:`H`-invariance with a negative leg;
     :math:`\pi_a\circ\varphi_a = \mathrm{pr}_1` on **12 of 12**;
     :math:`\beta_a\circ\pi_a` the axial projection on **3 of 3**; and
     the change of variables on a real rule, ``level_symmetric(4)``
     pushed along :math:`\pi_x` giving
     :math:`\int\mu^2 d(\pi_*\mu) = 4.18879020478639`, **1 ULP** from
     :math:`4\pi/3` (:ref:`manifold-quotient-map`). And
     :attr:`Quotient.reference
     <orpheus.numerics.manifold.Quotient.reference>` carries
     :math:`\pi_*\,d\Omega` — ``LEGENDRE`` on the axial entries by
     Archimedes' hat-box, ``None`` on the mirrors (whose pushforward is
     the weighted disk measure
     :math:`2\,du\,dv/\sqrt{1-u^2-v^2}`, which no shipped realization
     spells) and on :math:`M/\{e\}`; the registry now READS it, which
     collapses the campaign's **second** Pattern-2 twin
     (:ref:`manifold-pushforward-reference`,
     :ref:`manifold-second-twin-reference`).
   - ⛔ **2.3 and 3.1 are ENABLERS: they repair nothing, and 3.1's two
     halves differ in CONSUMPTION.** No membership check runs inside a
     map (that refusal is tracker 2.0b, at measure construction); the
     forgery arm stays a raw constructor **by design**, because routing
     it through ``pushforward`` would force it to tell the truth; and
     `[M]` the ERR-080 gate still declares **three**
     ``xfail(strict=True)`` rows, untouched by either. `[M]` over
     ``orpheus/``: ``reference`` has **one** production reader (the
     registry) while ``quotient_map`` and ``orbit_coordinates`` had
     **zero** outside their own module — `[M]` ten occurrences of the
     first and three of the second, all in
     ``tests/numerics/test_manifold.py``. ✅ ``quotient_map`` acquired
     its production readers on 2026-09-02: :meth:`LegendreBasis.evaluate
     <orpheus.numerics.basis.legendre_basis.LegendreBasis.evaluate>`
     pulls a full-sphere rule's directions back along it, and
     :func:`~orpheus.numerics.manifold.quotient_onto` returns it as a
     frame's G0 arrow. ``orbit_coordinates`` is still read only inside
     this module. The entry's **section** still does not ship at all:
     `[M]` ``fundamental_domain`` is ``None`` on every
     :math:`S^2/O(2)_a` entry and has zero readers anywhere, because a
     section is a *choice* and every field the entry carries is a
     derivation *output* (:ref:`manifold-arrows-not-built`).
   - **The algebra was already running, spelled as string
     interpolation.** `[M]` ``measure.py:588`` was
     :meth:`__mul__ <orpheus.numerics.manifold.Manifold.__mul__>`
     (``f"{self.support} × {other.support}"``), ``:1022`` was
     :meth:`quotient <orpheus.numerics.manifold.Manifold.quotient>`
     (``f"{self.support}/{group.name}"``) and ``:802`` was the
     pushforward's codomain (``f"φ_*({self.support})"``). The
     interpolation *was* the operation, unnamed
     (:ref:`manifold-string-tag`).
   - **The type is a CLOSED SUM split by TOTALITY**, not a polymorphic
     hierarchy. ``dim`` / ``name`` / ``contains`` / ``__mul__`` are
     answerable by every manifold and live on the abstract base; the
     derivation fields are answerable only by a quotient and live on
     :class:`~orpheus.numerics.manifold.Quotient` alone, so asking a
     sphere for a syzygy ideal is a type error rather than a ``None``
     (:ref:`manifold-closed-sum`).
   - **An orbit space is derived, by invariant theory, once per
     entry** — minimal invariants of :math:`\mathbb{R}[x]^H`, the
     syzygy ideal by elimination, then the Procesi–Schwarz condition
     :math:`P \succeq 0` with
     :math:`P_{ij} = \langle \nabla p_i, \nabla p_j\rangle`
     (:eq:`manifold-procesi-schwarz`). For :math:`S^2/O(2)_a` this gives
     :math:`P = \operatorname{diag}(1, 4p_2)`,
     :math:`\det P = 4p_2 = 4(1-\mu^2)` and the orbit space
     :math:`[-1,1]` (:eq:`manifold-s2-mod-so2`); for the shipped
     cylindrical fold :math:`S^2/\langle\sigma_a\rangle` it gives
     :math:`P = \operatorname{diag}(1,1,4p_3)` and the **closed unit
     disk** :math:`D^2` (:eq:`manifold-s2-mod-mirror`). **Both families
     are one derivation reading the axis off the group**, so the
     catalogue holds `[M]` **six** keys served by **two** procedures.
   - ⭐ **An orbit space has TWO honest coordinate systems, and a
     Quotient carries both** (user ruling, 2026-08-31). ``realization``
     is the **invariant chart's codomain** — the same language as
     ``generators`` / ``gram`` / ``det_gram``; ``fundamental_domain`` is
     a **section's image**, in the BASE's coordinates, which is what
     :meth:`DiscreteMeasure.quotient
     <orpheus.numerics.measure.DiscreteMeasure.quotient>` actually
     emits. They answer different questions and neither subsumes the
     other: `[M]` the chart is **Mode-12 blind** to the ERR-080 forgery
     while the section refuses it (:ref:`manifold-two-coordinate-systems`).
   - ⭐ **ERR-080's level-1 half is a botched SECTION of**
     :math:`S^2/O(2)_x`. The realization is a chart; a consumer needed a
     section; the tree fabricated one by zero-padding to
     :math:`(\mu,0,0)`, which is off :math:`S^2` — `[M]` norms
     :math:`0.183\ldots0.960`, while an honest :math:`\varphi = 0`
     half-meridian is on the sphere to :math:`0.0`. ⚠ That names the
     **level-1** half only; the level-2 repair is still the trivial
     isotypic sub-basis :math:`\{Y_\ell^0\}\cong\{P_\ell\}`
     (:ref:`manifold-err-080-is-a-section`). Since tracker 2.4 the slab
     at least *names* the orbit space it needs a section of; declaring
     the section itself is tracker 2.3, and it is a CHOICE
     (:ref:`manifold-the-axis-convention-for-a-section`).
   - ⚠ **An orbit space is an ORBIFOLD, not a quotient manifold.**
     Where the action is not free, the image of the fixed-point set is
     a **singular stratum**. For :math:`S^2/O(2)_a` that locus is
     :math:`\mu = \pm 1`, the poles — exactly where
     :math:`\det P` vanishes, and exactly where the curvilinear
     S\ :sub:`N` :math:`\alpha`-dome closes. A design that assumes a
     quotient is a smooth submersion is wrong there and only there
     (:ref:`manifold-singular-stratum`). For the :math:`\sigma_a` fold
     the stratum is the disk's **boundary circle** — a locus, not a
     point set, which is why ``singular_stratum`` is a symbolic locus
     and not a ``tuple[float, ...]``
     (:ref:`manifold-stratum-is-a-locus`).
   - **One polynomial, three appearances, three epistemic statuses.**
     :math:`(1-\mu^2)` is the squared :math:`SO(2)_a`-orbit radius
     (**derived**, this page); the redundant harmonic
     :math:`Y_2^{+2}` that makes the slab Gram rank-deficient
     (**measured**: `[M]` ratio spread :math:`8.9\times10^{-16}`); and
     the curvilinear angular-redistribution coefficient
     (**an identity of polynomials, with the mechanism unproved** —
     the reduction has not been derived, and the ruling that must
     settle it is Phase 1.3 of #429).
   - **The catalogue is the engine's SEED, not its rival.** A general
     orbit-space engine is **deferred, not refused**; the binding
     requirement is on the DATA MODEL — a catalogue entry must *be* the
     derivation procedure's output, so an engine ships by *computing*
     these fields instead of reading them, introducing no new type. The
     falsifiable check, and `[M]` **9 of 9** of the procedure's outputs
     are slots as of tracker 3.1, over **twelve** fields — it read
     6 of 8, then 7 of 9 (:ref:`manifold-engine-seed`). ⚠ A complete
     seed is not a shipped engine: `[M]` all **seven** quotients of
     :math:`S^2` the catalogue produces still read ``derived_by="hand"``.
   - ⭐ **This module imports** :mod:`orpheus.numerics.symmetry` **at
     MODULE scope, and until 2026-09-03 it could not.** A quotient is a
     manifold and a group, so the group arrives as an ordinary
     dependency: ``AXIS_INDEX``, ``AXIS_LETTER`` and
     :class:`~orpheus.numerics.symmetry.SubgroupOfO3` at
     ``manifold.py:78``, and :attr:`Quotient.by
     <orpheus.numerics.manifold.Quotient.by>` annotated with the real
     class. `[M]` 10 of 10 entry points import cleanly in fresh
     interpreters (:ref:`manifold-import-cycle`).
     ⛔ **This bullet read** *"imports nothing from* ``numerics`` *at
     MODULE scope, and that is load-bearing"* **until 2026-09-03**, and
     every word of it was true: the invariance kernel lived in
     ``symmetry``, so ``symmetry`` imported ``measure``, ``measure``
     imports this module, and the edge would have closed a cycle — a
     three-hop one, and after tracker 2.4 a two-hop one as well. R2 of
     #434 moved the kernel to :mod:`orpheus.numerics.invariance`, below
     ``measure``, and the loop is gone rather than deferred around. The
     module still carries **one** runtime edge into a higher layer,
     ``manifold → generating_measure`` at *function* scope inside a
     derivation, to put the ``LEGENDRE`` **value** in a field: `[M]` it
     is alive on 7 of 7 fresh import orders and every module-scope
     placement of the same line is dead on 7 of 7, because a
     ``TYPE_CHECKING`` guard defers a *name* and can never carry a
     *value* (:ref:`manifold-value-at-function-scope`).


.. _manifold-three-levels:

Three levels, and why a function space is not a domain
======================================================

A basis function is a map, and the question *"what is its domain?"* is
a question about what it **eats**:

.. math::

   Y_\ell^m : S^2 \longrightarrow \mathbb{R},
   \qquad
   P_\ell : [-1,1] \longrightarrow \mathbb{R},
   \qquad
   \mathbf{1}_{C_i} : \mathbb{R}^d \longrightarrow \{0,1\} .

The argument is a **point** — a unit direction, a cosine, a position —
not a vector of degrees of freedom. `[M]`
:class:`~orpheus.numerics.space.FunctionSpace` is documented as *"a
finite-dimensional vector space of discrete fields"* and carries a
``shape``: the tensor shape of the DOFs. So :math:`L^2(S^2)` is the
space the harmonics are **elements of**, and it is never the space they
are **maps from**. ``FunctionSpace`` answers *"what do these live in"*;
the domain question is *"what do these consume"*. Different arrows.

Separating them gives three levels. The tree named two:

.. list-table:: The three levels
   :header-rows: 1
   :widths: 14 30 28 28

   * - Level
     - The object
     - What a ``Basis`` carried
     - What a ``DiscreteMeasure`` carried
   * - 1 — the manifold
     - :math:`M`: :math:`S^2`, :math:`[-1,1]`, :math:`M/H`,
       :math:`\mathbb{R}^d`, energy, an index set
     - ⛔ **nothing**
     - ``support`` — a bare ``str``
   * - 2 — fields on :math:`M`
     - :math:`L^2(M)`, discretized at :math:`N` nodes
     - —
     - ``.space`` (a
       :class:`~orpheus.numerics.space.FunctionSpace`)
   * - 3 — coefficients
     - :math:`\mathbb{R}^K`
     - ``.space`` (a
       :class:`~orpheus.numerics.space.FunctionSpace`)
     - —

Read the table's last two columns across row 3: a basis and a measure
both have a ``.space``, and they are spaces of **different levels**.
That is not a naming accident to be tidied — a frame's whole job is the
pair of arrows between them
(:doc:`/theory/foundations/frame`). What is missing is row 1, on both
sides.

⭐ **And the level-2 check already passes, which is why the defect
survived.** On a slab the frame's analysis arrow
``measure.space → basis.space`` is between two perfectly well-formed
spaces: `[M]` on ``gauss_legendre(8).angular_frame(2)`` the domain is
``L2[S^2]`` of shape :math:`(8,)` and the codomain is
``spherical_harmonic_space`` of shape :math:`(3,5)`, the metric
resolves, and the pairing computes. Nothing is wrong at level 2.
**The check that fails is one level down**, on a manifold where no
object existed — so there was nothing to compare and nothing to refuse.

⚠ And note what the level-2 name is: `[M]` ``L2[S^2]``. It is
*derived* — ``measure.py:331`` builds it as ``f"L2[{self.support}]"``
— so the forged level-1 tag propagates upward verbatim, and a reader
inspecting the space sees a confident, wrong statement about the
manifold. A derived name is only as true as what it derives from.

.. note::

   **Why the operator-vocabulary collision is not one.** The proposal
   to spell a basis's level-1 slot ``support`` — to dodge
   :class:`~orpheus.numerics.operator.LinearOperator`'s existing
   ``domain`` / ``codomain`` vocabulary — was raised and **refuted** on
   2026-08-31, on two independent grounds. First, ``support`` is
   *mathematically false for a basis*: :math:`\operatorname{supp}(f)`
   means *where* :math:`f` *is non-zero*, and for an indicator basis
   that is exactly ONE cell per function — so
   ``IndicatorBasis.support = "spatial_R1"`` would state something
   untrue. (For the spherical-harmonic basis it is accidentally
   near-right, almost all of :math:`S^2`, which is what let the
   proposal past.) Second, the collision is a *word*, not a type:
   ``dom`` is the source of a morphism in both readings, in **Man** for
   a basis function and in **Vect** for an operator. Same functor,
   different categories. The slot is therefore ``domain``, and it
   :ref:`landed 2026-09-01 <manifold-seams>` (tracker 2.1) as an
   abstract property of the ABC — `[M]` answered by **6 of 6** shipped
   subclasses, the denominator being ``Basis.__subclasses__()`` walked
   recursively at runtime. ⛔ This sentence read *"it is not yet built"*
   until that day.


.. _manifold-string-tag:

What the string tag could not do
================================

Level 1 **was** ``Space = str`` (``measure.py:111``, retired 2026-09-01 by
tracker 2.0c), with the module's own comment on the alias set reading,
verbatim:

   *"Common aliases used across the project. These are recommendations,
   not constraints; user code may pass arbitrary strings."*
   — ``measure.py:113-114``, as it stood

That is an honest description of a slot with no semantics. Three
consequences followed from it, each measured; the type answers all three,
and the argument for minting it was these sites, not the size of the
migration. **All three are now closed in the code** — the sections below are
kept in the present tense of the tag they describe, because they are the
*reason* the type exists and re-tensing them would erase the evidence.

.. _manifold-err-080:

(a) Nothing could be refused — the ERR-080 forgery
--------------------------------------------------

✅ **REPAIRED 2026-09-02 by #429's fused commit. This section is the
record of the defect and is written in the past tense throughout; what
replaced it is** :ref:`manifold-what-descends`\ **, and the closed
catalogue entry is** :doc:`ERR-080 </theory/verification/error_catalog>`\ **.**

A 1-dimensional angular quadrature carries no azimuthal information.
:meth:`Quadrature.angular_frame
<orpheus.numerics.quadrature.directional.Quadrature.angular_frame>`
nonetheless built its measure by ``column_stack``\ ing three
axis-cosine arrays — two of which were a zero *fallback*, not data — and
declared the result ``support=SPACE_SPHERE`` (later ``support=SPHERE`` —
the forgery survived the retype; only its spelling changed). The rows were then
:math:`(\mu, 0, 0)` with
:math:`\lVert\Omega\rVert = |\mu| \neq 1`: points of :math:`[-1,1]`,
not of :math:`S^2`.

.. note::

   **Scoped, 2026-09-01; RETIRED 2026-09-02.** The paragraph above became a
   statement about the **1-D arm alone**, which was the whole of that
   change: the ``column_stack`` had run for *every* rule, so a Lebedev or
   level-symmetric frame also rebuilt a measure it had been handed. From
   2026-09-01 a rule whose nodes already were three-component directions
   handed the frame **its own measure**, and the construction survived only
   where there was genuinely nothing honest to build — `[M]` 10 of the 12
   shipped rules routed, 2 did not. It lived in
   ``Quadrature._harmonic_frame_measure`` with its retirement trigger written
   beside it, and that trigger fired: the method is **deleted**, and `[M]`
   ``frame.measure is q.measure`` on **12 of 12** shipped rules.

   ⭐ The repair also reversed two losses this page did not record, because
   they are not what the forgery is *about*: the rebuilt measure carried three
   of :class:`~orpheus.numerics.measure.DiscreteMeasure`'s five fields, so it
   dropped ``invariance_group`` and ``exactness`` as well as falsifying
   ``support``. `[M]` 10 of 12 rules carry a group, **0 of 12 frames did** —
   and at that moment the frame's forged ``support`` was still a *string*
   tag, so :attr:`DiscreteMeasure.phase
   <orpheus.numerics.measure.DiscreteMeasure.phase>` matched none of its
   manifold arms and fell through to the ``invariance_group`` fallback,
   which the rebuild had just dropped. *The angular frame's own measure
   could not say it was angular*: it raised ``NotImplementedError`` on all
   twelve. ⟹ the transferable form, worth more than the instance: **"the
   rebuild loses X" is a completeness claim over the source type's FIELD
   LIST**, and its denominator is ``dataclasses.fields(T)`` — not the
   concept you happen to be chasing.

   ⚠ Re-measured after the 2.0c retype: that same frame measure carried a
   real :class:`~orpheus.numerics.manifold.Sphere` support and `[M]`
   answered ``phase == 'angular'`` from the manifold arm, with
   ``invariance_group`` and ``exactness`` still ``None`` — the forgery was
   unchanged, so that reading was the raise having moved, not the defect.
   ✅ Since 2026-09-02 there is no rebuild at all: the frame IS handed the
   rule's measure, so all five fields are the rule's own and the losses
   above cannot recur by construction.

`[M]` reproduced 2026-08-31 on
``Quadrature.gauss_legendre(8).angular_frame(2)``, reading the
production measure's own nodes:

.. list-table:: The declared support against the nodes
   :header-rows: 1
   :widths: 46 54

   * - Reading
     - Value
   * - ``measure.support``
     - ``'S^2'``
   * - :math:`\lVert\Omega_n\rVert`, sorted
     - ``0.1834 0.1834 0.5255 0.5255 0.7967 0.7967 0.9603 0.9603``
   * - rows within :math:`10^{-12}` of the sphere
     - **0 of 8**
   * - ``Sphere().contains(nodes)``
     - ``False``
   * - ``Sphere().contains(nodes / ‖nodes‖)`` — positive leg
     - ``True``
   * - ``Interval(-1, 1).contains(nodes[:, 0])``
     - ``True`` — the manifold the nodes *actually* belong to

The positive leg is not decoration. A contract predicate exercised only
against a broken instance validates the *raising*, not the *claim*
(``vv-principles`` #11); the third and fourth rows together are what
make the refusal a statement about :math:`S^2` rather than about the
function.

That forgery is the first link of **ERR-080** — a live wrong-answer
defect in :math:`P_L` scattering, publicly reachable through
``solve_sn(..., scattering_order >= 2)`` on any 1-D chart, with a
second symptom (a crash whose message blames a layer three hops
downstream) at higher :math:`(N, L)`. The full mechanism, its
:math:`\ell = 1` masking, the census over 105 slab rows and the
proposed repair are in the
:doc:`error catalogue </theory/verification/error_catalog>` (search
``ERR-080``) and in
:doc:`/theory/foundations/spherical_harmonics`; this page carries only
the part that is a **manifold** claim.

.. warning::

   ⛔ **This warning read** *"The predicate exists; it is not wired"*
   **until 2026-09-02, and every clause of it was true when written.**
   `[M]` 2026-08-31, ``grep`` over ``orpheus/`` and ``tests/``: the only
   importers of :mod:`orpheus.numerics.manifold` were its own test
   module; ``angular_frame`` still wrote a string; ``DiscreteMeasure``
   still took a ``str``; and ERR-080 was open, held by an
   ``xfail(strict=True)`` regression gate.

   ✅ **ERR-080 is CLOSED (2026-09-02).** What closed it is not the
   construction-time membership refusal this warning was waiting for —
   `[M]` that is still absent, and a forged measure remains
   *constructible* (tracker 2.0b). It is the two refusals one level up,
   which together make a forged measure unusable rather than
   unspellable: :meth:`SphericalHarmonicBasis.evaluate
   <orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis.evaluate>`
   rejects a non-unit direction (tracker 0.6), and the frame's G0
   rejects the *pairing* that produced the forgery in the first place
   (:ref:`manifold-g0-descent-arrow`). ⭐ The transferable reading: the
   repair did not come from the predicate this page minted, but it could
   not have been *stated* without the point-set layer the predicate
   belongs to — the two refusals are both sentences about manifolds.

.. _manifold-string-algebra:

(b) The algebra ran as string concatenation
-------------------------------------------

`[M]` re-measured by AST over ``orpheus/`` + ``tests/`` (2026-08-31,
independently of the plan that first reported it, and agreeing with
it): of **62** ``.support`` attribute reads — 31 in ``orpheus/``, 31 in
``tests/`` — **18** feed a string operation (an f-string, a
``.startswith``, or a ``+``), and **all 18 are in** ``orpheus/``, none
in ``tests/``. Four of the 18 are the ones that matter here:

.. list-table:: Four sites, three of them verbs, spelled as interpolation
   :header-rows: 1
   :widths: 22 40 38

   * - Site (as of 2026-08-31)
     - What it computed, until the verb replaced it
     - The verb it *is*
   * - ``measure.py:588``
     - ``new_space = f"{self.support} × {other.support}"``
     - :meth:`Manifold.__mul__
       <orpheus.numerics.manifold.Manifold.__mul__>` — the product
       manifold :math:`M \times N`
   * - ``measure.py:1022``
     - ``new_space = f"{self.support}/{group.name}"``
     - :meth:`Manifold.quotient
       <orpheus.numerics.manifold.Manifold.quotient>` — the orbit
       space :math:`M/H`
   * - ``measure.py:802``
     - ``f"φ_*({self.support})"``
     - the codomain of a pushforward :math:`\varphi_* \mu`
   * - ``measure.py:331``
     - ``name = f"L2[{self.support}]"``
     - the *derived* level-2 name — the level-1 object smuggled inside
       a level-2 label

⛔ **Every cell of the middle column is now history, and each of the four
died on a different date.** The column header read *"What it computes
today"* until 2026-09-02; it was true when written and the campaign
repealed it row by row. Rows 1 and 2 became :meth:`Manifold.__mul__
<orpheus.numerics.manifold.Manifold.__mul__>` and
:meth:`Manifold.quotient
<orpheus.numerics.manifold.Manifold.quotient>` at tracker 2.0c
(2026-09-01); row 4's ``str`` became a typed
:class:`~orpheus.numerics.manifold.Manifold` the same day (the
interpolation survives, but of a point set rather than a tag — see the
✅ below); and row 3 is the last to go, at tracker 2.3 (2026-09-02),
where the pushforward's codomain stopped being *named at the call site*
and became a field of the map (:ref:`manifold-arrow-type`). The rows
stay because the *argument* is what they carry: the mint added no
algebra, and this is the evidence.

This is the strongest available form of the project's own type-minting
criterion (``coding-standards``, *Type vs property*: mint **iff** there
are :math:`\ge 2` non-isomorphic realizations **and** a non-identity
morphism is applied). The morphisms are not merely *applicable* — they
are *shipped*. The mint adds no algebra; it gives a name to one that
already runs.

⚠ **And the last row was not a naming quibble — it shipped as a
falsehood.** ``measure.py:331`` at least *derives* its name, from a
``str``; ``basis/indicator_basis.py`` **hard-coded** it as
``f"L2[coarse_cells_R{self.ndim}]"``. And
:meth:`EnergyGrid.as_basis <orpheus.data.energy_grid.EnergyGrid.as_basis>`
builds an :class:`~orpheus.numerics.basis.indicator_basis.IndicatorBasis`
over an **energy index** partition
(``edges = arange(n_groups + 1) - 0.5``), so that basis named its own
coefficient space after a *spatial* manifold it has nothing to do
with. `[M]` reproduced 2026-08-31 on a two-group grid, and again
2026-09-01 immediately before the repair:

.. code-block:: python

   >>> eg = EnergyGrid(edges=np.array([1e6, 1.0, 1e-5]))    # 2 GROUPS
   >>> mesh = Mesh1D(edges=np.array([0.0, 0.5, 1.0]), ...)  # 2 CELLS
   >>> eg.as_basis().space.name              # BEFORE #429 tracker 2.1
   'L2[coarse_cells_R1]'
   >>> mesh.indicator_basis().space.name
   'L2[coarse_cells_R1]'                     # ...the very same value
   >>> eg.as_basis().space == mesh.indicator_basis().space
   True

The two compared ``==`` **and hash-equal**, so a 2-group energy space and
a 2-cell spatial space were one value: both are hand-named, axes-less
:class:`~orpheus.numerics.space.FunctionSpace` mints, whose identity is
``(name, shape)`` — before AND after the 2026-09-07 identity flip, which
made identity structural only where a space declares its ``axes``
(:ref:`spaces-identity-bridge`). A false name is therefore not cosmetic
but an illegal state that IS representable, and staying axes-less is
exactly why the flip could not have repaired this one.

✅ **REMEDIED 2026-09-01 by #429 tracker 2.1.** The
:class:`~orpheus.numerics.basis.base.Basis` ABC now asks every basis what
its functions EAT —
:attr:`~orpheus.numerics.basis.base.Basis.domain`, a
:class:`Manifold` — and an
:class:`~orpheus.numerics.basis.indicator_basis.IndicatorBasis` takes the
manifold it partitions as a required constructor field, so the name
derives:

.. code-block:: python

   >>> eg.as_basis().space.name
   'L2[coarse_cells(energy)]'
   >>> mesh.indicator_basis().space.name
   'L2[coarse_cells(spatial_R1)]'
   >>> eg.as_basis().space == mesh.indicator_basis().space
   False

⭐ **What made the defect invisible for as long as it lived is worth
recording, because it is not carelessness.** At four of the five
production sites the basis and its
:class:`~orpheus.numerics.measure.DiscreteMeasure` are built in the SAME
function, three to five lines apart — and the *measure* named the
manifold correctly the whole time (``support="energy"``,
``"spatial_R1"``, ``f"index({label})"``). The answer was never
unavailable, only unasked; a hard-coded f-string is exactly the shape
that cannot be contradicted by the object sitting beside it. The durable
gate is therefore not *"the name is right"* — which any self-consistent
lie satisfies — but *"the two halves of one frame name ONE manifold"*
(``tests/numerics/test_basis_domain.py::test_d6``).

⭐ And assigning the type was itself a census: `[M]` it immediately
separated two manifolds the string tag ``"energy"`` had conflated — the
continuous energy axis in eV (:class:`Interval`, partitioned by
``tests/data/test_energy_grid.py``) from the multigroup *index* axis
(:class:`EnergyGroups`, what production partitions). Both have ambient
dimension 1, so no dimensional check could have found it; only naming the
point set does.

⭐ **And the slot answers a SECOND question, which is what tracker 2.1b
collected the same day.** Once a basis names the manifold its functions
eat, it has already declared the symmetry those functions *have*: a
function on an orbit space :math:`M/H` is, pulled back to :math:`M`,
exactly an :math:`H`-invariant function. So
:attr:`Basis.invariance_group
<orpheus.numerics.basis.base.Basis.invariance_group>` is **derived from**
``domain`` and stored nowhere — the second operand of the ERR-080 pairing,
obtained for no new field and no subclass edit
(:ref:`manifold-basis-invariance-group`).

The remaining half is the *measure's* side, and since tracker 2.0c it is a
**name** rather than a type: ``support`` became a :class:`Manifold` that
same day, so ``measure.py:371`` now derives ``f"L2[{self.support.name}]"``
from a typed point set. What is still missing is that
:class:`~orpheus.numerics.space.FunctionSpace` records only the resulting
string; one that carried its own manifold would collapse both spellings
into one — the level-2 half of this repair, tracked at
:ref:`manifold-seams`. ⛔ This paragraph read *"``support`` is still a
``str``, so ``measure.py:331`` derives a correct name from an untyped
tag"* until 2026-09-01, and it was true when written: 2.1 and 2.0c landed
hours apart, and the second one repealed the first one's premise.

.. _manifold-string-drift:

(c) The vocabulary drifted, and a nonsense quotient is spellable
----------------------------------------------------------------

`[M]` **2026-08-31, on the tree as it then stood** — read this
subsection in the past tense throughout — both ``'S^2/<sigma_y>'`` (a
hand-written ``new_space=`` in ``tests/numerics/test_measure.py``) and
``'S^2/sigma_y'`` (two further sites, and what the production
:meth:`DiscreteMeasure.quotient
<orpheus.numerics.measure.DiscreteMeasure.quotient>` emitted, since
``SubgroupOfO3.Mirror("y").name == "sigma_y"``) shipped. They named
**one** quotient and were **unequal under** ``==``. Also shipped as
support literals: ``'img'``, ``'probe'``, ``'[-1,1]^slab'``.

.. note::

   ⛔ **The mechanism this subsection describes is retired, and its two
   halves died separately.** The *tag* went at tracker 2.0c
   (2026-09-01), when ``support`` became a
   :class:`~orpheus.numerics.manifold.Manifold` and two strings naming
   one quotient stopped being expressible. The *hand-written target*
   went at tracker 2.3 (2026-09-02): ``new_space=`` is retired and a
   pushforward reads its codomain off the map
   (:ref:`manifold-arrow-type`). The subsection stays because the
   demonstration below — a nonsense quotient accepted by one verb and
   refused by the other — is the argument that the mint was a repair
   and not a re-spelling. ⚠ Line numbers in the original `[M]` are
   deliberately dropped here: which line of a test file carried a
   spelling is not a durable fact, and three of the four cited files
   have since been edited by this campaign.

The sharper demonstration is what the quotient verb accepts. `[M]`
2026-08-31, folding a :math:`\sigma_y`-invariant 4-node set by
``Mirror("y")`` under three different tags:

.. list-table:: What the two quotient verbs accept
   :header-rows: 1
   :widths: 34 33 33

   * - Declared ``support``
     - ``DiscreteMeasure.quotient`` result
     - :meth:`Manifold.quotient
       <orpheus.numerics.manifold.Manifold.quotient>`
   * - ``'S^2'``
     - ``'S^2/sigma_y'`` — accepted
     - ``NotImplementedError``, naming the missing derivation
   * - ``'probe'``
     - ``'probe/sigma_y'`` — accepted
     - (not a manifold; unspellable)
   * - ``'not_a_manifold_at_all'``
     - ``'not_a_manifold_at_all/sigma_y'`` — accepted
     - (not a manifold; unspellable)

⚠ **State the scope of that exactly, because the obvious summary is
false.** :meth:`DiscreteMeasure.quotient
<orpheus.numerics.measure.DiscreteMeasure.quotient>` is **not**
unchecked: it calls
:func:`~orpheus.numerics.invariance.certificate_under` and refuses a
measure that is not :math:`G`-invariant. What is unchecked is the
**tag** — the new support is minted by interpolation from whatever
string the old one held, for any group, with no claim that :math:`M/H`
is a manifold anyone has derived. Two different objects are being
gated and un-gated: the *nodes* are certified, the *manifold* is
asserted.

⭐ And the same run shows the constructor's other blind spot: the four
nodes used above have :math:`\lVert x\rVert = \sqrt2`, and
``DiscreteMeasure(nodes=..., support="S^2")`` accepted them without
complaint. That is ERR-080's mechanism in four lines, with no
quadrature involved.


.. _manifold-closed-sum:

The type: a closed sum, split by TOTALITY
=========================================

The ruled shape (user, 2026-08-31) is a **closed sum**, and the axis of
the split is **totality**:

- the operations **every** manifold answers — ``dim``, ``name``,
  ``contains``, ``__mul__`` — are abstract on the base
  :class:`~orpheus.numerics.manifold.Manifold`, so no variant may
  abstain;
- the operations only a **quotient** can answer — the invariant
  generators, the syzygy ideal, the Procesi–Schwarz matrix and its
  determinant, the singular stratum, the provenance — live on
  :class:`~orpheus.numerics.manifold.Quotient` alone.

A sphere has no syzygy ideal, and under this shape it cannot be asked
for one: the question is a :exc:`TypeError`, not a ``None``.

.. _manifold-shape-evidence:

The evidence, not the taste
---------------------------

Two shipped precedents were **read**, not recalled, and the deciding
measurement is a field census.

.. list-table:: The two precedents, measured 2026-08-31
   :header-rows: 1
   :widths: 26 74

   * - Precedent
     - What it actually is
   * - :class:`~orpheus.geometry.boundary.BoundaryTraceLaw`
     - A **registered sibling hierarchy**: `[M]` 7 direct subclasses
       (``AlbedoBoundary``, ``PeriodicBoundary``, ``PrescribedInflow``,
       ``ReflectiveBoundary``, ``VacuumInflow``, ``WhiteBoundary``,
       ``ZeroFluxBoundary``) and 7 registry keys. The right shape when
       every member answers the same questions differently.
   * - :class:`~orpheus.numerics.symmetry.SubgroupOfO3`
     - **ONE class over a** ``_tag`` **sum**: `[M]` 13 methods
       dispatching with **10** ``isinstance`` calls and **0** ``match``
       statements; not a dataclass, and **not frozen** —
       ``g._tag = 'MUTATED'`` succeeds. The right *data model* for a
       small stable member set, with the dispatch it would get today.

The ruled shape is the second precedent's data model with the first's
class-per-variant realization — the two halves of :math:`M/H` kept
structurally parallel — and the reason it is not a polymorphic
hierarchy over a nullable base is measurable:

.. list-table:: Where the derivation fields belong
   :header-rows: 1
   :widths: 40 30 30

   * - Field
     - Answerable by a quotient?
     - Answerable by a sphere / interval?
   * - invariant generators :math:`p_1 \ldots p_k`
     - yes
     - **no**
   * - syzygy ideal :math:`I`
     - yes
     - **no**
   * - :math:`P_{ij} = \langle\nabla p_i, \nabla p_j\rangle`
     - yes
     - **no**
   * - :math:`\det P`
     - yes
     - **no**
   * - the singular stratum
     - yes
     - **no**
   * - ``derived_by`` provenance
     - yes
     - **no**

`[M]` **every** derivation field is in the left column. On a
polymorphic hierarchy they would therefore sit on the *base*, returning
``None`` for every non-quotient — which is exactly the tax
:attr:`SubgroupOfO3.mirror_axis
<orpheus.numerics.symmetry.SubgroupOfO3.mirror_axis>` already pays
(`[M]` ``None`` for ``SO2('x')``, ``Dinfh``, ``Oh`` and ``O3``, ``1``
for ``Mirror('y')``) and which ``directional.py:522`` already branches
on to raise :exc:`NotImplementedError`. Repeating that tax on a
brand-new type, with six fields instead of one, is the design the
closed sum refuses.

⭐ **And tracker 2.4 doubled the tax rather than repaying it**, which
is worth knowing before reading it as an argument against
axis-parameterisation. The axial rotation group's axis needed the same
accessor, so :attr:`SubgroupOfO3.rotation_axis
<orpheus.numerics.symmetry.SubgroupOfO3.rotation_axis>` joined it as
the **continuous dual**: `[M]` 2026-09-02, ``SO2('x') → 0``,
``O2('x') → 0``, ``SO2('z') → 2``, ``O2('z') → 2``, and ``None`` for
everything else — *including* the groups that CONTAIN axial rotations
without **fixing** the axis (:math:`D_{\infty h}` flips it by
:math:`\sigma_h`; :math:`SO(3)` and :math:`O(3)` move it), because the
accessor identifies the axis whose polar interval the group's orbit
space on :math:`S^2` **is**. ⛔ It read *"the group whose elements are
exactly the rotations about one coordinate axis"* until 2026-09-02: true
of the one axial family that then existed, and repealed by #432, which
added the stabiliser :math:`O(2)_a` above it
(:ref:`manifold-orbit-space-stabiliser`). The two accessors are still
**mutually exclusive** on the shipped family: `[M]` over **21** groups
(the six named entries, three mirrors, three axial rotations, three
axial stabilisers, :math:`C_{2,3,4,6}` and :math:`D_{2h}, D_{6h}`),
``mirror_axis`` is non-``None`` on exactly the three :math:`\sigma_a` and
``rotation_axis`` on exactly the **six** axial groups — **zero** groups
answer non-``None`` to both.

.. note::

   **Why not a phantom type parameter.** ``.claude/plans/sn_reshape.md``
   Issue 2 — quoted verbatim in ``measure.py:106-109`` — rules *"don't
   try to enforce* ``Space`` *types via Python generics; not expressive
   enough without runtime overhead"*. That ruling stands and **does not
   cover this**: it rejects ``Generic[Tag]`` *phantom* parameters,
   which ``coding-standards`` also rejects, because they are erased at
   runtime and do not specialize dunders — one ``__mul__`` body would
   serve every instantiation. A first-class **value** with real methods
   is a different proposal, and the one the three shipped morphisms
   above require.

.. _manifold-members:

The members
-----------

`[M]` **ten** variants — ``Manifold.__subclasses__()`` restricted to
this module, which is also what the exhaustiveness gate compares
against. They fall in three families (curved, flat, discrete) plus
**four** constructors that take another manifold as an argument:
``Product``, ``Ball``, ``FundamentalDomain`` and ``Quotient``. Two
dimensions travel with each member and are easy to conflate: the
**topological** dimension ``dim`` (what the manifold *is*) and the
**ambient** coordinate count (how many columns ``contains`` consumes).
They differ for every curved member.

⛔ **Correction (2026-08-31).** The first version of this paragraph read
*"nine variants … plus two recursive constructors"*. Both halves were
wrong when written: the mint shipped **eight** concrete variants
(``git show b8c05d16:orpheus/numerics/manifold.py | grep -c '^class .*(Manifold)'``
:math:`\to` 8) and the table below listed eight rows, so the prose and
its own table disagreed. The two-slot ruling then added
:class:`~orpheus.numerics.manifold.Ball` and
:class:`~orpheus.numerics.manifold.FundamentalDomain`, taking the count
to ten. The count is now stated with the command that produces it,
because a member roster is a universal and the shipped
``test_every_variant_is_reachable_from_this_modules_list`` is the only
thing that keeps it honest.

.. list-table:: The shipped members
   :header-rows: 1
   :widths: 16 8 8 20 48

   * - Variant
     - ``dim``
     - ambient
     - ``name``
     - Membership predicate
   * - ``Sphere``
     - 2
     - 3
     - ``S^2``
     - :math:`\bigl|\lVert x\rVert - 1\bigr| \le \varepsilon`
   * - ``Circle``
     - 1
     - 2
     - ``S^1``
     - :math:`\bigl|\lVert x\rVert - 1\bigr| \le \varepsilon`
   * - ``Interval(a, b)``
     - 1
     - 1
     - ``[a,b]``
     - finite, and :math:`a - \varepsilon \le x \le b + \varepsilon`
   * - ``RealSpace(d)``
     - :math:`d`
     - :math:`d`
     - ``spatial_Rd``
     - finite
   * - ``EnergyGroups(ng)``
     - 0
     - 1
     - ``energy``
     - integral, and :math:`0 \le g < n_g`
   * - ``IndexSet(label, n)``
     - 0
     - 1
     - ``index(label)``
     - integral, and :math:`0 \le i < n`
   * - ``Product(L, R)``
     - :math:`\dim L + \dim R`
     - sum
     - ``L × R``
     - each factor admits its own coordinate block
   * - ``Ball(d)``
     - :math:`d`
     - :math:`d`
     - ``D^d``
     - finite, and :math:`\lVert p\rVert^2 \le 1 + \varepsilon`
   * - ``FundamentalDomain(M, n⃗, ℓ)``
     - :math:`\dim M -` #antipodal pairs
     - :math:`M`'s
     - ``M|ℓ``
     - :math:`M` admits it, **and**
       :math:`\langle p, n_i\rangle \ge -\varepsilon` for every normal
   * - ``Quotient(M, H, …)``
     - :math:`\dim` of the realization
     - realization's
     - ``M/H``
     - **either** coordinate system — the chart's, or the
       fundamental domain's, dispatched on the ambient width
       (:ref:`manifold-two-coordinate-systems`)

`[M]` the names reproduce the retired ``SPACE_*`` string tags
**verbatim** — ``S^2``, ``S^1``, ``[-1,1]``, ``[0,1]``, ``[0,inf)``,
``R``, ``energy``, ``spatial_R1``, ``index(angular)`` — so the
migration off ``Space = str`` could not silently re-word an error message
or an ``L2[...]`` space name that a test pins. That is a deliberate
constraint on the type, gated by
``tests/numerics/test_manifold.py::TestTypeLaws::test_the_names_reproduce_the_retired_string_tags``.

✅ **The constraint paid, and it is now a fact rather than a design
intent.** `[M]` 2026-09-01, immediately before tracker 2.0c touched a call
site: **10 of 10** tag constants reproduce exactly, and so do the two
*derived* spellings the tag vocabulary built by hand — ``S^2/sigma_y`` via
:attr:`Quotient.name` (against ``f"{support}/{group.name}"``) and
``[-1,1] × [-1,1]`` via :attr:`Product.name` (against ``f"{a} × {b}"``). The
single divergence in the whole migration is the affine remap
``LEGENDRE.on(0, 1)``, whose f-string over floats read ``"[0.0,1.0]"`` where
``Interval(0.0, 1.0).name`` normalises to ``"[0,1]"`` — a re-baseline of two
test pins, and an improvement: the object is the same interval whatever the
float repr of its endpoints, which the string could not say.

Two consolidations are visible in that table and were the reason the
member list in the plan's own row was incomplete:

- ``Interval`` **is ONE family, not three tags.** The retired
  ``SPACE_INTERVAL_M11``, ``SPACE_INTERVAL_01``, ``SPACE_HALF_LINE``
  and ``SPACE_R`` are four *members* of it — which is what
  ``generating_measure.py:420``'s ``support=f"[{a},{b}]"`` was already
  saying, one interpolation at a time.
- ``IndexSet`` **was minted twice, under incompatible spellings.**
  `[M]` ``frame.py:759`` builds ``f"index({axis_label})"`` and
  ``sn/operators/loss_kernel_gauge.py:1169`` builds
  ``f"sn_trace_orbit{orbit}_g{group}"``, whose "points" are trace DOF
  indices cast to float. One family; ``label`` is what distinguishes
  the instances.

⚠ ``EnergyGroups`` is deliberately **not** a bare ``IndexSet``. A group
index and any other integer-noded counting rule are indistinguishable
from their nodes alone, which is precisely why the tag had to supply
the physical identity in the first place; the measure's derived
``phase`` depends on it.

⭐ **The last two members were minted by a DERIVATION, not by a survey**
— which is the healthiest reason for a variant to exist, and worth
recording as such. Neither ``Ball`` nor ``FundamentalDomain`` was on
anyone's list of manifolds the tree might want. They arrived because
the second catalogue entry produced objects the shipped member set
could not name:

- ``Ball`` — :math:`S^2/\langle\sigma_a\rangle` **is** the closed
  2-disk in invariant coordinates (:ref:`manifold-s2-sigma-y` derives
  it), and the nearest shipped 2-D member,
  ``Product(COSINE_INTERVAL, COSINE_INTERVAL)``, is the bounding
  **square**. The discriminator is measured, not stylistic: `[M]`
  :math:`(0.9, 0.9)` is in the square and **not** in the disk, and it
  is the image of no direction at all, since
  :math:`\eta^2 + \mu^2 = 1.62 > 1` forces :math:`\xi^2 = -0.62 < 0`.
  Adopting the square because it already ships would have admitted
  :math:`(\eta,\mu)` pairs that no :math:`\Omega` maps to
  (:ref:`manifold-realization-refuted`).
- ``FundamentalDomain`` — the section's image, the *other* of a
  quotient's two coordinate systems, and the one every measure the
  tree emits through ``.quotient(...)`` actually speaks
  (:ref:`manifold-two-coordinate-systems`). One rule covers both a
  half-space and a hyperplane: an **antipodal pair** of normals
  :math:`\{n, -n\}` spells the equality :math:`\langle p, n\rangle = 0`,
  so `[M]` ``FundamentalDomain(SPHERE, (e_y,), …).dim == 2`` (the
  :math:`\sigma_y` hemisphere) while
  ``FundamentalDomain(SPHERE, (e_y, -e_y, e_x), …).dim == 1``
  (an :math:`SO(2)` half-meridian) — from the same field, with no
  second slot and no flag.

.. note::

   **The membership tolerance is a construction tolerance, not a
   physics one.** ``_MEMBERSHIP_ATOL`` is :math:`10^{-12}`, chosen to
   match the construction quality of the shipped quadrature rules
   (whose nodes are exact to a few ULP). `[M]` a node at
   :math:`1 + 10^{-9}` is refused; at :math:`1 + 10^{-13}` and
   :math:`1 + 10^{-14}` it is admitted. No caller should widen it to
   make a measure fit — the whole value of the predicate is that a
   forged support has nowhere to hide, and the ERR-080 forgery misses
   by :math:`4\times10^{-2}` to :math:`8\times10^{-1}`, not by ULPs.
   ``contains`` is a **universal**, not a mean: one bad row is enough.

   ⚠ **The name was ambiguous until 2026-09-03, and this note owns the
   surviving half.** ``symmetry.py`` carried a constant of the SAME
   spelling at :math:`10^{-9}`, asking a different question — whether
   two realized *operators* are the same ELEMENT of a group, not whether
   a *point* lies on a manifold — so a grep for ``_MEMBERSHIP_ATOL``
   returned two bands and two meanings. #434 R1 renamed that one
   ``_ELEMENT_ATOL``; :math:`10^{-12}` here is the point band, and it is
   now the only ``_MEMBERSHIP_ATOL`` in the tree.


.. _manifold-orbit-space:

The orbit space, derived
========================

An orbit space :math:`M/H` is not declared and not guessed. It is
**computed**, by a standard construction from real invariant theory,
once per (manifold, group) pair. This section states the procedure,
runs it in full on the **first** pair the tree catalogued, and draws
the four consequences that pay for it downstream. The **second** pair —
the shipped cylindrical fold :math:`S^2/\langle\sigma_y\rangle` — is
run in the section after it, because its answer forced a change to the
data model and so belongs after the consequences rather than beside
them (:ref:`manifold-second-entry`).

The catalogue holds **six** keys today — ``Sphere/O2_x``,
``Sphere/O2_y``, ``Sphere/O2_z``, ``Sphere/sigma_x``,
``Sphere/sigma_y``, ``Sphere/sigma_z`` — served by **two** procedures,
because each family shares one derivation that reads the axis off the
group. Asking for an orbit space by a NON-MAXIMAL group never reaches
the table: the door refuses it first, with the theorem
(:ref:`manifold-orbit-space-stabiliser`). The identity quotient :math:`M/\{e\} = M` is a seventh answer and
is *not* a table row: it is derived on the spot, for every manifold,
because it is a theorem (:ref:`manifold-twin-lookup`).

⛔ This sentence read *"**four** keys … ``Sphere/SO2``"* until
2026-09-01, when tracker 2.4 gave the axial rotation group its axis and
the single ``SO2`` key became three. The count did not move again at
#432 (2026-09-02): the three axial entries were RE-KEYED onto their
stabiliser, ``SO2_a`` → ``O2_a``, not added to. The **procedure** count
has never moved, and that is the point: parameterising a family costs
keys, never derivations (:ref:`manifold-so2-axis-is-a-parameter`).

.. _manifold-derivation-procedure:

The procedure
-------------

Let :math:`G` be a compact group acting orthogonally on
:math:`\mathbb{R}^n`, and let :math:`X \subseteq \mathbb{R}^n` be a
:math:`G`-stable real algebraic set. Five steps:

**1. Minimal generators of the invariant ring.** Find
:math:`p_1, \ldots, p_k` generating :math:`\mathbb{R}[x]^G`. That a
finite generating set exists is Hilbert–Weyl; finding it is the step a
Gröbner engine would automate.

**2. The orbit map separates orbits.** The polynomial map
:math:`p = (p_1, \ldots, p_k) : \mathbb{R}^n \to \mathbb{R}^k` is
proper and separates :math:`G`-orbits, so

.. math::

   \mathbb{R}^n / G \;\cong\; p(\mathbb{R}^n) \;\subseteq\; \mathbb{R}^k ,

and the image is a **closed semialgebraic** set. The orbit space is
therefore describable by finitely many polynomial equalities and
inequalities — which is what makes the whole construction finite.

**3. The syzygy ideal gives the equalities.** The generators need not
be algebraically independent; the relations among them are

.. math::

   I \;=\; \ker\bigl(\mathbb{R}[y] \to \mathbb{R}[x],\;
                     y_i \mapsto p_i \bigr),

computed by elimination. :math:`V(I)` — the variety of that ideal — is
the algebraic set the image lies in. When the invariants *are*
algebraically independent, :math:`I = (0)` and there are no equalities.

**4. Procesi–Schwarz gives the inequalities.** The image is cut out of
:math:`V(I)` by one positive-semidefiniteness condition on the
**gradient Gram matrix** of the invariants, re-expressed in the
invariants themselves:

.. math::
   :label: manifold-procesi-schwarz

   p(\mathbb{R}^n) \;=\;
   \bigl\{\, y \in V(I) \;:\; P(y) \succeq 0 \,\bigr\},
   \qquad
   P_{ij} \;=\; \bigl\langle \nabla p_i,\, \nabla p_j \bigr\rangle .

This is the theorem the whole construction turns on: Procesi and
Schwarz, *Inequalities defining orbit spaces*, **Inventiones
Mathematicae 81** (1985), 539–554. :math:`V(I)` supplies the
equalities; :math:`P \succeq 0` supplies the inequalities, and together
they are a complete description.

**5. Intersect with the ideal of** :math:`X`. Steps 1–4 quotient the
whole ambient space. Adjoining the defining ideal of :math:`X` — for
:math:`S^2`, the single relation :math:`\lVert x\rVert^2 = 1` —
restricts the answer to :math:`X/G`.

.. (vv-status rationale) manifold-procesi-schwarz is a
   LITERATURE-TRANSCRIBED theorem statement (Procesi & Schwarz 1985):
   it has no implementing ORPHEUS function to verify against and makes
   no solver claim. What IS verifiable is its INSTANCE at
   :eq:`manifold-s2-mod-so2`, whose P-matrix, determinant and stratum
   are recomputed symbolically by the foundation gates in
   tests/numerics/test_manifold.py::TestQuotient
   (test_the_procesi_schwarz_matrix_matches_the_hand_derivation,
   test_det_P_vanishes_exactly_on_the_singular_stratum). Those tests
   carry @pytest.mark.foundation and deliberately NO verifies(...),
   per vv-principles' foundation-tier rule.
.. vv-status: manifold-procesi-schwarz documented

.. _manifold-s2-so2:

The worked entry: :math:`S^2 / O(2)_a`
----------------------------------------

This is the **first** entry the tree catalogued, and it is the entry
every 1-dimensional angular discretisation is secretly using. Every
line below was **re-derived and re-run** in this session, independently
of the catalogue, for **all three axes**, and then compared against it.

Write :math:`a` for the **rotation axis** and :math:`b, c` for the other
two — the same convention the mirror entry uses for its mirrored axis
(:ref:`manifold-s2-sigma-y`). The shipped slab and sphere geometries
spend :math:`a = x`; a product rule's polar factor is about
:math:`a = z`. **Why the axis is a parameter at all, rather than a
convention fixed once, is** :ref:`manifold-so2-axis-is-a-parameter` —
read it before treating the three entries as redundant.

**Step 0 — the group.** The derivation starts from the **rotations**
— :math:`SO(2)_a`, the proper rotations fixing :math:`\hat e_a` and
mixing the other two coordinates,
acting on :math:`\mathbb{R}^3` by (shown for :math:`a = z`, and
:math:`\det R_\theta = +1` for every :math:`a`)

.. math::

   R_\theta =
   \begin{pmatrix}
     \cos\theta & -\sin\theta & 0 \\
     \sin\theta & \phantom{-}\cos\theta & 0 \\
     0 & 0 & 1
   \end{pmatrix}.

It is compact, **connected**, and :math:`\dim = 1` — every structural
difference from the mirror entry follows from those three words
(:ref:`manifold-chart-section-asymmetry`).

⭐ **The ENTRY, however, is named by** :math:`O(2)_a` — the full
stabiliser of :math:`\hat e_a`, whose invariants and orbits are the
rotations' (:ref:`manifold-orbit-space-stabiliser`, #432). Nothing below
changes: every step from the invariants on is the same, and asking the
catalogue for :math:`S^2/SO(2)_a` is refused at the door, naming
:math:`O(2)_a`, before any derivation runs. Read :math:`SO(2)_a` below as the
group the derivation is *written from*, and :math:`O(2)_a` as the group
the answer is *recorded under*.

**Step 1 — the invariants.** Two invariants generate:

.. math::

   p_1 = x_a, \qquad p_2 = x_b^2 + x_c^2 .

`[M]` verified symbolically for general :math:`\theta`, **on all three
axes** — both satisfy :math:`p(R_\theta x) - p(x) = 0` after
``simplify``, with the non-invariant control :math:`x_b` correctly
reported **not** invariant in each case. The control matters: a check
that passes on everything is not a check.

**Step 3 — the syzygy ideal is empty.** The Jacobian (shown for
:math:`a = z`)

.. math::

   \frac{\partial (p_1, p_2)}{\partial (x, y, z)}
   =
   \begin{pmatrix} 0 & 0 & 1 \\ 2x & 2y & 0 \end{pmatrix}

has `[M]` generic rank **2** on every axis, equal to the number of
invariants, so :math:`p_1` and :math:`p_2` are algebraically independent
and :math:`I = (0)`. There are no equalities; the orbit space is cut out
by inequalities alone.

**Step 4 — the Procesi–Schwarz matrix.** With
:math:`\nabla p_1 = \hat e_a` and
:math:`\nabla p_2 = 2(x_b \hat e_b + x_c \hat e_c)`,

.. math::
   :label: manifold-s2-mod-so2

   P \;=\;
   \begin{pmatrix} 1 & 0 \\ 0 & 4 p_2 \end{pmatrix},
   \qquad
   \det P \;=\; 4 p_2 ,
   \qquad\text{so}\qquad
   \mathbb{R}^3 / O(2)_a \;=\; \{\, p_2 \ge 0 \,\}.

.. note::

   ⚠ **The label** ``manifold-s2-mod-so2`` **keeps its** ``so2``
   **prefix, and that is a historical artefact rather than a claim.**
   The equation's content is the orbit space, which #432 renamed onto
   the stabiliser :math:`O(2)_a` (:ref:`manifold-orbit-space-stabiliser`)
   without moving one symbol of the derivation. The label is an **API**
   — `[M]` ``tests/numerics/test_slab_orbit_space.py:258`` carries
   ``@pytest.mark.verifies("manifold-s2-mod-so2")`` and
   :doc:`spherical_harmonics` cites it — so it is kept and only the body
   was re-spelled, per the rule that a stale NAME is not a false CLAIM.

The two invariants have orthogonal gradients everywhere — the first
points along :math:`a`, the second lies in the :math:`bc`-plane — which
is why :math:`P` is diagonal and the condition collapses to a single
inequality. Note that :math:`P` itself carries **no** trace of which
axis was chosen: the axis lives entirely in the *generators*, which is
exactly why the three entries are one derivation and three keys.

**Step 5 — restrict to the sphere.** Adjoining
:math:`p_1^2 + p_2 = 1` and writing
:math:`\mu = p_1 = \hat\Omega \cdot \hat e_a`:

.. math::

   \det P \big|_{S^2} \;=\; 4\,(1 - \mu^2),
   \qquad
   S^2 / O(2)_a \;=\; \{\, \mu \in \mathbb{R} : 1 - \mu^2 \ge 0 \,\}
   \;=\; [-1, 1].

`[M]` SymPy's ``solve_univariate_inequality`` on
:math:`4 - 4\mu^2 \ge 0` returns ``Interval(-1, 1)`` on every axis — the
orbit space is *computed*, not asserted — and the zero set is exactly
:math:`\{-1, +1\}`.

⚠ **The three orbit spaces are isometric and their realizations are
identical, and they are still three different quotients.** `[M]`
``SPHERE.quotient(SubgroupOfO3.O2('x')).realization ==
SPHERE.quotient(SubgroupOfO3.O2('z')).realization`` is ``True`` (both
are ``Interval(-1.0, 1.0)``) while the two ``Quotient`` values compare
``False``. That is the correct answer and it is the whole point:
:math:`\mu` is the cosine against a *different* direction in each, so a
rule on :math:`S^2/O(2)_x` and a rule on :math:`S^2/O(2)_z` can carry
byte-identical nodes and mean different functions of direction. A type
that identified them would re-admit exactly the confusion
:ref:`manifold-so2-axis-is-a-parameter` describes.

.. (vv-status note) manifold-s2-mod-so2 is the INSTANCE of
   :eq:`manifold-procesi-schwarz` for the axial-rotation catalogue
   family. Its content IS checked, and tightly: the P-matrix and its
   determinant are recomputed symbolically and compared with
   sp.simplify, and the singular stratum is SOLVED for rather than
   compared to a literal, by
   tests/numerics/test_manifold.py::TestQuotient::{test_the_procesi_schwarz_matrix_matches_the_hand_derivation,
   test_det_P_vanishes_exactly_on_the_singular_stratum}.
   .
   ⛔ UN-SENTINELED 2026-09-01 (tracker 2.4), and the reason is worth
   keeping because the sentinel carried its own exit condition and the
   condition FIRED. This block used to read `.. vv-status:
   manifold-s2-mod-so2 documented`, arguing that the gates above are
   @pytest.mark.foundation and that vv-principles' foundation tier
   carries no verifies(...) marker by rule -- so a verifies edge would
   assert a claim class the gates do not make. Two things falsified it.
   (a) [M] tests/numerics/test_slab_orbit_space.py:258,
   test_d1_three_axes_three_quotients_one_derivation, now carries
   @pytest.mark.verifies("manifold-s2-mod-so2") and asserts THIS
   equation's content per axis -- the invariants p1 = x_a and
   p2 = x_b^2 + x_c^2 against the shipped generators, det P = 4 p_2,
   the realization, and the pairwise distinctness of the three
   quotients. (b) [M] the foundation/verifies exclusion is not this
   project's practice: 65 tests tree-wide carry BOTH markers, the
   algebra-of-record SymPy-identity shape (tests/derivations/*_symbolic.py
   is full of them), so the combination produces a real edge here as it
   does there.
   .
   Keeping the sentinel would have made the generated matrix contradict
   itself -- the label listed with 1 verifying test AND in the
   Documented-only set, which is excluded from the orphan gate. The
   directive is therefore removed rather than re-argued, and this note
   stays so a future auditor does not re-add one. The equation is now
   verifies-covered; the two TestQuotient gates above remain its
   tightest checks and carry no marker, by the same 65-test convention.

.. list-table:: My re-derivation against the shipped catalogue entry, all three axes
   :header-rows: 1
   :widths: 44 56

   * - Check
     - Result, `[M]` 2026-09-01, for :math:`a \in \{x, y, z\}`
   * - :math:`P` (mine) :math:`-` ``entry.gram``
     - ``simplify`` :math:`\to` the zero :math:`2\times2` matrix, **3 of
       3 axes**
   * - :math:`\det P` (mine) :math:`-` ``entry.det_gram``
     - ``simplify`` :math:`\to 0`, **3 of 3**
   * - :math:`(p_1 - x_a,\; p_2 - (x_b^2+x_c^2))` (mine)
       :math:`-` ``entry.generators``
     - ``simplify`` :math:`\to (0, 0)`, **3 of 3** — and this is the row
       that *sees* the axis: the shipped generators read
       ``(p1 - x, p2 - y**2 - z**2)`` for :math:`a = x` and
       ``(p1 - z, p2 - x**2 - y**2)`` for :math:`a = z`
   * - ``entry.realization``
     - ``Interval(a=-1.0, b=1.0)`` — the **same value** on all three
   * - ``entry.dim`` / ``Sphere().dim``
     - ``1`` / ``2``
   * - ``entry.syzygy``
     - ``()`` — empty, as derived
   * - ``entry.singular_stratum`` / ``entry.is_free``
     - ``1 - u0**2`` / ``False``. The stratum is a **locus**, and
       solving it gives :math:`\{-1, +1\}` — the poles
       (:ref:`manifold-stratum-is-a-locus`)
   * - ``entry.fundamental_domain``
     - ``None`` — and that is an answer, not a gap: **no** section of
       :math:`S^2 \to S^2/O(2)_a` is canonical, since every
       half-meridian is one and none is distinguished
       (:ref:`manifold-two-coordinate-systems`)
   * - ``entry.name`` / ``entry.derived_by``
     - ``'S^2/O2_x'``, ``'S^2/O2_y'``, ``'S^2/O2_z'`` / ``'hand'``.
       ⛔ These cells read ``'S^2/SO2_x'``/``_y``/``_z`` until
       2026-09-02, when #432 named the entry by its stabiliser; the
       derivation, the gram, the stratum and the realization are
       unchanged (:ref:`manifold-orbit-space-stabiliser`)
   * - ``type(entry.gram)``
     - ``ImmutableDenseMatrix``, so the ``Quotient`` is **hashable** —
       required by the memo below, and by any ``set``/``dict`` keyed on
       a measure's support

⚠ One reproduction note, kept because it is the cheap trap in step 4.
Re-expressing :math:`P` in the invariants is a substitution, and
``sp.Matrix(...).subs(x**2 + y**2, p2)`` **silently fails** on
:math:`4x^2 + 4y^2`, which does not literally contain the node
:math:`x^2+y^2`. My first run therefore produced
:math:`\det P = 4x^2 + 4y^2`, an empty solution set for the stratum,
and a spurious disagreement with the catalogue. The failure was mine,
not the entry's; ``factor`` before substituting (or, as the shipped
builder does, substitute :math:`x_b \to \sqrt{p_2},\, x_c \to 0` —
legal because the expression is constant on an orbit) fixes it. A
disagreement with a reference is not a refutation until you have
diagnosed whose it is.

.. _manifold-orbit-space-stabiliser:

An orbit space is named by its STABILISER, so it has ONE spelling
------------------------------------------------------------------

The derivation above is written for the **rotations**, and until
2026-09-02 the entry was keyed and named by them — ``S^2/SO2_x``. That
was one spelling of an object with two groups, and it was the smaller
one. The ruling that replaced it (**GitHub #432**, user-ruled
2026-09-02) is a naming law for *every* entry the catalogue will ever
hold, not a fact about :math:`SO(2)`, so it is stated here once and
cross-referenced from every other page that names an orbit space.

.. admonition:: The naming law
   :class: important

   **An orbit space is named by its STABILISER** — the largest subgroup
   of :math:`O(3)` whose orbits are its orbits, equivalently the largest
   one fixing every invariant. Two groups with the same orbits do not
   give two entries; they give **one entry and two spellings**, and only
   the maximal spelling is admitted. A smaller orbit-equivalent group is
   refused *at the derivation*, with the theorem, rather than silently
   producing a second name for one point set.

Which group that is, for the axial family
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The orbits of :math:`SO(2)_a` on :math:`S^2` are the circles of constant
:math:`\mu = \Omega\cdot\hat e_a`. A reflection in a plane **containing**
the axis carries each such circle onto itself — it reverses the
traversal and fixes the latitude — so it preserves the orbit partition
while not being a rotation. Adjoining every such :math:`\sigma_v` gives

.. math::
   :label: manifold-axial-stabiliser

   O(2)_a \;=\; \bigl\{\, g \in O(3) \;:\; g\,\hat e_a = \hat e_a \,\bigr\}
   \;=\; SO(2)_a \;\sqcup\; SO(2)_a\,\sigma_v ,

the **pointwise stabiliser of the axis vector**: two connected
components, whose identity component is :math:`SO(2)_a`. Embedded in
three dimensions it is the point group :math:`C_{\infty v}`. It ships as
:class:`~orpheus.numerics.symmetry.O2`, axis-parameterised for exactly
the reason :class:`~orpheus.numerics.symmetry.SO2` is
(:ref:`manifold-so2-axis-is-a-parameter`) — the tree carries two poles.

.. (vv-status rationale) manifold-axial-stabiliser is a group-theoretic
   DEFINITION — the pointwise stabiliser of a coordinate axis in O(3),
   written as the disjoint union of its two components. It states no
   flux, eigenvalue or convergence claim and has no implementing
   function to verify against: what a test can check is the CONSEQUENCE
   (the lattice edges of the table below, the exact invariance
   criterion, SubgroupOfO3.orbit_stabiliser's answers, and the
   Quotient.__post_init__ refusal), and
   tests/numerics/test_symmetry.py + tests/numerics/test_manifold.py do
   check those under @pytest.mark.foundation, which carries no
   verifies(...) marker by vv-principles' foundation-tier rule. Landing
   an edge here would mint a coverage claim of a class those gates do
   not make.
.. vv-status: manifold-axial-stabiliser documented

**The proof is one line of the derivation above, and it is symbolic.**
The invariants recorded in :eq:`manifold-s2-mod-so2` are
:math:`p_1 = x_a` and :math:`p_2 = x_b^2 + x_c^2`; a vertical mirror
fixes :math:`x_a` and negates one of :math:`x_b, x_c`, which
:math:`p_2` squares away. So

.. math::
   :label: manifold-axial-invariant-rings

   \mathbb{R}[x]^{SO(2)_a} \;=\; \mathbb{R}[x]^{O(2)_a}
   \;=\; \mathbb{R}\bigl[\, x_a,\; x_b^2 + x_c^2 \,\bigr],

and every step of the Procesi–Schwarz procedure downstream of the
invariants — the syzygy ideal, :math:`P`, :math:`\det P`, the
restriction to :math:`S^2` — reads *only* the invariants. **One
procedure, one output, two groups.**

.. (vv-status rationale) manifold-axial-invariant-rings is an IDENTITY
   BETWEEN TWO INVARIANT RINGS, and neither side is a quantity any
   ORPHEUS function computes: the tree stores the generators of the
   right-hand side once, on the catalogue entry, and never forms either
   ring. Declaring an implementer would assert that some symbol IS one
   of the two sides. The verifiable consequence is
   SubgroupOfO3.SO2(a).orbit_stabiliser == SubgroupOfO3.O2(a), the
   refusal it induces (SPHERE.quotient(SO2(a)) raises, naming O2(a)),
   and the derived equality of the entries' fields — all gated under
   @pytest.mark.foundation in tests/numerics/test_manifold.py.
.. vv-status: manifold-axial-invariant-rings documented

`[M]` 2026-09-02, re-derived in this session independently of the tree
(SymPy, general :math:`\theta`, all three axes; the axis-permuting
convention :math:`(x_a, x_b, x_c)` of the section above):

.. list-table:: :math:`p_1, p_2` under each candidate element
   :header-rows: 1
   :widths: 30 34 36

   * - element
     - :math:`p_1(gx) - p_1(x)`
     - :math:`p_2(gx) - p_2(x)`
   * - a rotation :math:`R_\theta` about :math:`a`
     - :math:`0`
     - :math:`0`
   * - a vertical mirror :math:`\sigma_v`
       (normal :math:`\hat e_c`)
     - :math:`0`
     - :math:`0`
   * - :math:`R_\theta\,\sigma_v` — a generic element of the OTHER
       component
     - :math:`0`
     - :math:`0`
   * - ⛔ :math:`\sigma_h` (normal :math:`\hat e_a`) — **not** in
       :math:`O(2)_a`
     - :math:`-2x_a`
     - :math:`0`
   * - ⛔ the control :math:`x_b`, under a rotation
     - :math:`x_b(\cos\theta - 1) - x_c\sin\theta`
     - —

Read the last two rows before the first three: they are what makes the
first three informative. :math:`\sigma_h` moves :math:`p_1`, so
:math:`D_{\infty h}` — which contains it — does **not** share these
invariants, so it is a *different* orbit space and not a third spelling
of this one. And :math:`x_b` is not invariant at all, which is the
control without which a check comparing an expression to itself would
pass.

What :math:`O(2)_a` is NOT — three near neighbours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - group
     - what it fixes
     - why it is not the stabiliser of :math:`\mu`
   * - :math:`SO(2)_a`
     - :math:`\hat e_a`, properly
     - the identity **component** of the answer — right orbits, not the
       largest group with them
   * - :math:`C_{\infty h}`
     - :math:`\hat e_a` up to :math:`\sigma_h`
     - rotations plus the mirror **perpendicular** to the axis, which
       flips :math:`\mu \to -\mu` and so merges two orbits
   * - :math:`D_{\infty h}`
     - the axis as a **LINE**
     - :math:`D_{\infty h} = O(2)_a \times \{e, \sigma_h\}` — the
       stabiliser of :math:`\pm\hat e_a` as a set. Its invariant ring is
       :math:`\mathbb{R}[x_a^2,\, x_b^2 + x_c^2]`, one degree up, so its
       orbit space is the HALF interval :math:`\mu^2 \in [0,1]` — a
       different point set, and one the catalogue does not hold

⚠ **The name has history, and the history is the reason to state this
distinction rather than assume it.** ``O2`` was the name of the
:math:`D_{\infty h}` entry until 2026-08-02 and was retired because that
entry is REALIZED as :math:`C_{\infty h}` — rotations plus
:math:`\sigma_h` — while true :math:`O(2)` embedded in three dimensions
is :math:`C_{\infty v}`. The class that ships under the name today is
the group the old name should have meant; the retired one lives on as
:attr:`SubgroupOfO3.Dinfh
<orpheus.numerics.symmetry.SubgroupOfO3.Dinfh>` and the lattice records
``Dinfh.contains(O2("z"))`` as the edge that spells
:math:`D_{\infty h} = O(2)_z \times \{e, \sigma_h\}`.

Invariance is decided exactly, and it does NOT ask for :math:`\mu \to -\mu`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A finite point set is :math:`O(2)_a`-closed **iff every node lies ON the
axis** — the same criterion as :math:`SO(2)_a`, decided and never
sampled (ERR-072; :ref:`manifold-so2-axis-lattice`). Axis support is
forced by the :math:`SO(2)_a` half; and a point *on* the axis is fixed by
every vertical mirror, so the second component adds no condition. `[M]`
2026-09-02, hand-built measures on :math:`S^2`:

.. list-table::
   :header-rows: 1
   :widths: 42 12 12 12 12

   * - node set
     - :math:`O(2)_x`
     - :math:`SO(2)_x`
     - :math:`O(2)_y`
     - :math:`\sigma_x`
   * - :math:`\mu \in \{\pm 0.3, \pm 0.8\}` on the :math:`x`-axis
     - ✅
     - ✅
     - ⛔
     - ✅
   * - :math:`\mu \in \{0.3, 0.5, 0.9\}` — **asymmetric**, same axis
     - ✅
     - ✅
     - ⛔
     - ⛔
   * - a single node :math:`\mu = 0.3` on the axis
     - ✅
     - ✅
     - ⛔
     - ⛔
   * - one node at :math:`45^\circ` — off the axis
     - ⛔
     - ⛔
     - ⛔
     - ⛔

⭐ Row 2 is the one to read. An asymmetric :math:`\mu`-set is
:math:`O(2)_x`-invariant and **not** :math:`\sigma_x`-invariant:
:math:`O(2)_a` does not contain the reflection that reverses
:math:`\mu`, because :math:`\sigma_x` flips :math:`\hat e_x` and the
group fixes it. The forward/backward pairing a slab owes its two sweep
senses is a *separate* group, and it is exactly the residual the
geometry table records beside the spent one
(:ref:`manifold-has-versus-spent`).

The law is an ACCESSOR, and the refusal is a construction invariant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The naming law is not prose the catalogue is trusted to honour. It is
one property on the group and one guard on the type, and between them a
mis-named quotient is **unspellable** rather than merely refused at one
door.

**(1) The group answers which group it would be named by.**
:attr:`SubgroupOfO3.orbit_stabiliser
<orpheus.numerics.symmetry.SubgroupOfO3.orbit_stabiliser>` returns the
largest subgroup of :math:`O(3)` with this group's orbits. Since #434 R1
(2026-09-03) it reads the answer off the group's **realization**
(:ref:`manifold-realization`) rather than matching its tag: a finite
group is its own stabiliser; a component of dimension 3 answers
:math:`O(3)`; a torus about :math:`\hat a` answers :math:`O(2)_a` when
:math:`O(2)_a` contains it and itself when it contains :math:`O(2)_a`.
That third clause is where :math:`D_{\infty h}` lands, and the case
neither containment covers raises a NAMED refusal rather than falling
through to a wrong answer — `[M]` no shipped member reaches it, and
`[M]` the accessor is still total on the lattice, its answers unchanged
by the carve except for the ``C_1`` merge. It moves exactly two
FAMILIES:

.. list-table:: ``orbit_stabiliser`` over the shipped lattice
   :header-rows: 1
   :widths: 24 22 54

   * - group
     - answers
     - why
   * - :math:`SO(2)_a`
     - :math:`O(2)_a`
     - :eq:`manifold-axial-invariant-rings`: the vertical mirrors fix
       both invariants, so the orbits are the same circles.
   * - :math:`SO(3)`
     - :math:`O(3)`
     - :math:`\mathbb{R}[x]^{SO(3)} = \mathbb{R}[\lVert x\rVert^2] =
       \mathbb{R}[x]^{O(3)}` — the antipodal map fixes the radius, so
       both groups' orbits are the spheres about the origin.
   * - every other member
     - **itself**
     - `[R]` a *generic* orbit of a finite :math:`G` is a free orbit of
       :math:`\lvert G\rvert` points, and any :math:`H` with the same
       orbits permutes each of them, so :math:`\lvert H\rvert =
       \lvert G\rvert` and :math:`H = G`. Holds for ``Trivial``,
       :math:`C_n`, :math:`D_{nh}` (including :math:`D_{1h}`),
       :math:`\sigma_a`, :math:`O_h`, :math:`I_h`; and for the
       continuous :math:`D_{\infty h}`, :math:`O(2)_a`, :math:`O(3)`,
       whose orbits already determine them.

⚠ Read the table's shape, not only its rows: **two families** in the
lattice are not their own orbit stabiliser, and those are exactly the
two whose orbit space a caller might reasonably ask for by the wrong
name. That is why the property is worth minting rather than a
special case in the sphere derivation. ⚠ *Families*, not members —
`[M]` 2026-09-03 over the 27 spellings this page's fixtures build, the
accessor moves **4** of them, because the axial family has three axes
(:math:`SO(2)_x`, :math:`SO(2)_y`, :math:`SO(2)_z`, :math:`SO(3)`). The
denominator matters here for the same reason the axis is a parameter.

**(2) The type refuses to hold a non-maximal** ``by``.
:class:`~orpheus.numerics.manifold.Quotient`'s ``__post_init__``
asserts ``by == by.orbit_stabiliser``, so the refusal is a property of
the VALUE and not of the route that built it — ``dataclasses.replace``
on a live entry is refused by the same guard as a fresh derivation, and
no catalogue builder can produce a mis-named entry however it is
registered. This is *illegal-states-unrepresentable* on the naming law:
a quotient object that names a non-maximal group cannot exist.

**(3) The door answers before the lookup.** ``_catalogued_quotient``
checks the same predicate *before* consulting ``_ORBIT_CATALOGUE``, so a
caller who asks for :math:`S^2/SO(2)_x` is told the theorem — the
orbit space exists and is spelled :math:`S^2/O(2)_x` — rather than the
generic *"no catalogue entry — derive it"* message
(:ref:`manifold-refusal-names-the-work`), which would send them off to
derive a space that already ships. Diagnosis at the door is the whole
reason the check is not left to the type alone.

⟹ ``_ORBIT_CATALOGUE`` therefore holds **six** keys for **six** entries
— ``Sphere/O2_x``, ``Sphere/O2_y``, ``Sphere/O2_z`` and the three
``Sphere/sigma_*`` — served by **two** procedures, and
``_sphere_mod_o2`` is a pure derivation again: it validates nothing and
carries the module's single function-scope runtime import
(:ref:`manifold-value-at-function-scope`).

.. note::

   ⛔ **A rejected first design, kept because it is the tempting one.**
   The refusal originally lived *inside* ``_sphere_mod_o2``, reached
   through three decoy ``Sphere/SO2_*`` catalogue keys registered
   against the same procedure. It works and it was measured working —
   but it puts a *validation* inside a *derivation*, needs a second
   function-scope import (``SubgroupOfO3``, to compare by value) so the
   module's own "one runtime edge" paragraph stops being true, spends
   three catalogue keys on rows that derive nothing, and leaves
   ``dataclasses.replace(entry, by=SO2('x'))`` **accepted** — the guard
   was on one route, not on the value. Reading it as a construction
   invariant plus a door check fixes all four at once, and the accessor
   is what makes the invariant one line.

   ⛔ **One of those four costs EXPIRED on 2026-09-03, and the other
   three did not.** R2 of #434 reversed the import direction, so
   :class:`~orpheus.numerics.symmetry.SubgroupOfO3` is a module-scope
   name here and comparing a group by value costs no import at all
   (:ref:`manifold-import-cycle`). The clause is kept rather than
   deleted because it is the *reason* the ruling was reached, and
   because the ruling does not depend on it: the three surviving costs —
   a validation inside a derivation, three keys deriving nothing, and a
   ``replace``-shaped hole the route guard cannot see — are each
   sufficient on their own.

What the ruling buys, and it is not cosmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One entry with one spelling is what lets
:attr:`Basis.invariance_group
<orpheus.numerics.basis.base.Basis.invariance_group>` — DERIVED by
reading ``domain.by`` (:ref:`manifold-basis-invariance-group`) — be the
**full** group a basis's functions have rather than a lower bound on it.
`[M]` 2026-09-02:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - reading
     - before #432
     - after
   * - ``LegendreBasis(3).domain.name``
     - ``'S^2/SO2_x'``
     - ``'S^2/O2_x'``
   * - ``LegendreBasis(3).invariance_group.name``
     - ``'SO2_x'`` — a strict LOWER bound
     - ``'O2_x'`` — the group the :math:`P_\ell(\mu)` actually have
   * - ``LegendreBasis(3).space.name``
     - ``'legendre_space(S^2/SO2_x)'``
     - ``'legendre_space(S^2/O2_x)'``
   * - ``gauss_legendre(8).measure.support.name`` / ``.space.name``
     - ``'S^2/SO2_x'`` / ``'L2[S^2/SO2_x]'``
     - ``'S^2/O2_x'`` / ``'L2[S^2/O2_x]'``
   * - ``GEOMETRY_ANGULAR_SYMMETRY['slab'].spent``
     - ``SO2('x')``
     - ``O2('x')``

⚠ The last row's SLOT was called ``continuous_isotropy`` on both sides
of this table until R3 of #434 (2026-09-03) renamed it ``spent``; the
row is re-spelled rather than tombstoned because what #432 moved is the
VALUE, and running the reading as it was written would now raise
``AttributeError``.

and the consequence a user can see is at
:ref:`manifold-g0-descent-arrow`: because :math:`\sigma_b \in O(2)_a`
for :math:`b \ne a`, the induced map :math:`S^2/\sigma_b \to
S^2/O(2)_a` exists, so a frame may bind the Legendre basis on a
:math:`\sigma_y`-folded rule — the pairing #432 was filed for, which was
over-refused while the declaration was a lower bound.

⚠ **The orbit space did not move, and no number did.** `[M]` stage 0 of
quadrature selection — the ``geometry.support == rule.support`` compare
— is IDENTICAL on **24 of 24** (geometry × rule) rows before and after,
measured against a pinned pre-change tree
(``git archive HEAD``, editable finder stripped, ``orpheus.__file__``
asserted): the slab admits ``gauss_legendre(8)`` and refuses the other
five on both sides, and the cylinder and 2-D Cartesian rows do not move
at all. Both halves of the compare were re-spelled together, which is
the whole content of the change at that tier.

.. _manifold-so2-axis-is-a-parameter:

Why the rotation axis is a PARAMETER — the tree carries two poles
------------------------------------------------------------------

The derivation above is written in :math:`a`, and the catalogue holds
three keys where it held one. That is not generality for its own sake.
It is forced, and the forcing is a *measured* property of this codebase
rather than a mathematical nicety: **ORPHEUS has two polar axes in
simultaneous use, and one Gauss–Legendre rule serves both.**

Until 2026-09-01 the axial rotation group was a parameter-free enum
member ``SO2``, realized about :math:`z` — its exactness criterion asked
whether every node had :math:`\rho = \sqrt{x^2+y^2} = 0`. The three
facts that make that untenable are all in the tree today:

.. list-table:: The two poles, `[M]` 2026-09-01 on the live tree
   :header-rows: 1
   :widths: 30 12 58

   * - Site
     - Pole
     - What fixes it there
   * - The slab / sphere polar marginal
     - :math:`x`
     - The 1-D embedding is :math:`(\mu, 0, 0)`, so the residual mirror
       is :math:`\sigma_x` and the marginal's nodes lie on the
       :math:`x`-axis.
   * - ``_evaluate_real_sh``
       (:mod:`orpheus.numerics.basis.spherical_harmonic_basis`)
     - :math:`x`
     - ``cos_theta = mu_x``. The real spherical-harmonic pole **is**
       :math:`x`, so :math:`Y_\ell^0 = P_\ell(\mu_x)` and the Legendre
       polynomials are the :math:`m = 0` members *of that* basis.
   * - Every product rule's polar factor
       (:func:`~orpheus.numerics.quadrature.product_mu_phi`)
     - :math:`z`
     - `[M]` on ``Quadrature.product(4, 8)``, the :math:`z` column takes
       exactly **4** distinct values and they are the ``leggauss(4)``
       nodes, while :math:`x` and :math:`y` take **9** each. The
       azimuth winds about :math:`z`.
   * - :math:`C_n`, :math:`D_{nh}`, :math:`D_{\infty h}`
     - :math:`z`
     - The standard setting for the finite families, which the lattice
       already assumed and this page already recorded.

⟹ **the same function**,
:func:`~orpheus.numerics.quadrature.gauss_legendre_on_mu`, is the raw
material of *both* rows: it is the slab's rule on :math:`S^2/O(2)_x`
and the polar factor of a product rule on :math:`S^2/O(2)_z`. A group
tag on that rule that did not name its axis would be a claim about the
wrong pole in one of the two uses, whichever way it was fixed.

**The symptom the bare tag actually produced.** The retired criterion
was ``all(hypot(x, y) <= atol)`` — :math:`\rho` measured **about**
:math:`z`, unconditionally. `[M]` run verbatim against the slab
marginal's own embedded nodes, whose first row is
:math:`(-0.9602898564975362,\, 0,\, 0)`, it returns **False**: the
:math:`\rho` it computes is :math:`|\mu|`, not :math:`0`. So the one
shipped rule whose orbit space *is* :math:`S^2/O(2)_x` reported that it
was **not** invariant under the group it had been quotiented by. The
campaign plan for #429 records this as its "Part IV obstacle 1". With
the axis named, the same criterion is exact *and* discriminating: `[M]`
on ``Quadrature.gauss_legendre(8).measure``, ``SO2('x')`` is **True**
while ``SO2('y')`` and ``SO2('z')`` are **False** — and `[M]` 2026-09-02
the stabilisers answer identically, ``O2('x')`` **True** /
``O2('y')``, ``O2('z')`` **False**, because a point ON the axis is fixed
by the vertical mirrors too and the criterion is the same one
(:ref:`manifold-orbit-space-stabiliser`).

**Two alternatives were available and both were refused.**

*Fix the axis by fiat* — pick one pole and standardise. That is exactly
what the retired bare tag did, and it re-ships the defect shape the
reflection family had already been cured of on 2026-08-02: an unnamed
group realized on one axis while a consumer needs another
(:doc:`/theory/foundations/discrete_measures` records the ``Z_2`` case,
where ``product(4, 3)`` is :math:`\sigma_z`-closed and **not**
:math:`\sigma_x`-closed while the tag answered ``True``). One curable
family per repair is not a pattern; two is.

*Move the slab to the* :math:`z` *pole* — make one pole true. There is
no single pole to standardise **on**: the move would have to touch the
sweep, the real spherical-harmonic basis, every ``Mirror('x')`` row in
the geometry table, and every frozen slab snapshot — to buy a
convention, not a capability. And it would still leave the product
rule's factor and the slab's rule as *different* objects that a shared
function returns, which is the thing the axis parameter states directly.

⭐ **The general rule this instance is the second witness for.** A group
realized on a coordinate axis carries that axis as **data**, not as a
setting: :class:`~orpheus.numerics.symmetry.Mirror` reached this ruling
for the plane on 2026-08-02 and
:class:`~orpheus.numerics.symmetry.SO2` for the rotation axis on
2026-09-01, both after shipping a wrong answer, both by leaving the
parameter-free enum for a frozen dataclass — and
:class:`~orpheus.numerics.symmetry.O2`, the axis's full stabiliser,
was born axis-parameterised on 2026-09-02 for the same reason, without
ever having to ship a wrong answer first
(:ref:`manifold-orbit-space-stabiliser`). The enum is now exactly the
groups that have **no** axis to name: `[M]` **six** members —
``Trivial``, :math:`D_{\infty h}`, :math:`O_h`, :math:`I_h`,
:math:`SO(3)`, :math:`O(3)` — beside **five** parameterised families
:math:`C_n`, :math:`D_{nh}`, :math:`\sigma_a`, :math:`SO(2)_a` and
:math:`O(2)_a`. ⛔ That count read **four** families until 2026-09-02.

.. note::

   ⚠ **Why** :math:`C_n` **did not follow.** It is cyclic about
   :math:`z` and is *not* axis-parameterised, and that asymmetry is
   deliberate rather than an oversight: no consumer has yet needed a
   :math:`C_n` about another axis. The project's own rule is to unify
   after the **second** instance, not before it
   (``coding-standards``), and both axis families crossed that line by
   shipping a wrong answer. When a :math:`C_n` about :math:`x` first
   appears, this is the paragraph that says what to do.

.. _manifold-realization:

A closed subgroup of :math:`O(3)` IS (identity component, coset representatives)
---------------------------------------------------------------------------------

Every question this page asks about a group — *does it contain that
one?*, *does it normalise it, so it acts on the orbit space?*, *does its
connected part fix these nodes?*, *what is its dimension?* — used to be
answered by a different piece of code per family of group, and two of
those pieces were hand-written tables of relations between NAMES. #434
R1 (2026-09-03) replaced all of them with one representation and one
body per question. The representation is not a data-structure choice; it
is a classification theorem, and the theorem is what makes the
replacement possible.

The theorem — :math:`\mathfrak{so}(3)` has no two-dimensional subalgebra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A compact subgroup :math:`G \le O(3)` is a closed Lie subgroup. Write
:math:`G^0` for its identity component — the connected part — and recall
that :math:`G/G^0` is finite, so

.. math::
   :label: manifold-group-as-component-and-cosets

   G \;=\; \bigsqcup_{r \in R} r\,G^0 ,
   \qquad R \ \text{a set of coset representatives},\ e \in R,

with :math:`G^0 = \exp\mathfrak g` determined by the Lie subalgebra
:math:`\mathfrak g \subseteq \mathfrak{so}(3)`. That decomposition is
only *useful* because the second factor is small and the first has just
three possibilities:

.. math::
   :label: manifold-so3-subalgebras

   \mathfrak g \;\in\;
   \bigl\{\, \{0\},\ \ \mathbb R\,[\hat a]_\times\ (\hat a \in S^2),\ \
   \mathfrak{so}(3) \,\bigr\},
   \qquad \dim\mathfrak g \in \{0, 1, 3\} \ \text{— never } 2 .

.. (vv-status rationale) manifold-group-as-component-and-cosets and
   manifold-so3-subalgebras: two structural facts of Lie theory,
   transcribed here because the shipped representation IS this
   decomposition and every predicate on it is one case table over
   ``dim``. Neither states a flux, eigenvalue or convergence claim, so
   neither has an L0..L3 ladder slot. Their CODE content is verified by
   the foundation gates on ``SubgroupOfO3.contains`` /
   ``is_normalised_by`` / ``normalises`` / ``dim`` in
   ``tests/numerics/test_symmetry.py``, which carry
   ``@pytest.mark.foundation`` and therefore no ``verifies(...)`` marker
   by ``vv-principles``' foundation-tier rule.
.. vv-status: manifold-group-as-component-and-cosets documented
.. vv-status: manifold-so3-subalgebras documented

**The one-line reason, and it is a complete proof.** As a Lie algebra
:math:`\mathfrak{so}(3)` is :math:`(\mathbb R^3, \times)`: the map
:math:`v \mapsto [v]_\times` (the skew matrix with
:math:`[v]_\times w = v \times w`) is an isomorphism onto
:math:`\mathfrak{so}(3)`, and it carries the cross product onto the
bracket,

.. math::
   :label: manifold-so3-is-the-cross-product

   \bigl[\,[\hat a]_\times,\ [\hat b]_\times\,\bigr]
   \;=\; [\,\hat a \times \hat b\,]_\times .

.. (vv-status rationale) manifold-so3-is-the-cross-product: the standard
   Lie-algebra isomorphism (R^3, x) ~ so(3), transcribed because it is
   the whole content of the dimension law above and because the shipped
   ``symmetry._skew`` / ``symmetry._axis_of`` pair IS this map and its
   inverse. Not a solver claim; verified as an identity on the shipped
   generators (the foundation gates over ``_realize``'s output), and
   ``[M]`` 2026-09-03 by direct evaluation over the nine coordinate
   pairs, max deviation exactly 0.
.. vv-status: manifold-so3-is-the-cross-product documented

Now suppose :math:`\mathfrak g` is a subalgebra with
:math:`\dim\mathfrak g = 2`, spanned by independent :math:`u, v`. Then
:math:`u \times v \in \mathfrak g` by closure, and :math:`u \times v` is
perpendicular to both :math:`u` and :math:`v` and nonzero — a **third**
independent direction inside a two-dimensional space. Contradiction. So
a subalgebra has dimension 0, 1 or 3, and in the middle case it is the
line through a single :math:`[\hat a]_\times`, whose exponential is the
circle of rotations about :math:`\hat a`. `[M]` 2026-09-03,
:eq:`manifold-so3-is-the-cross-product` evaluated over the nine
coordinate pairs :math:`(\hat a, \hat b)`: maximum deviation
:math:`0.000\times10^{0}` — exactly zero, the identity being a
rearrangement of the same six signed entries — and the membership test
:math:`[\hat x \times \hat y]_\times \in
\operatorname{span}\{[\hat x]_\times\}` answers ``False``, which is the
contradiction realized.

⟹ **a closed subgroup of** :math:`O(3)` **is exactly the pair (identity
component, coset representatives)**, and the identity component is
exactly one of: the trivial group, a circle about a named axis, or
:math:`SO(3)`. That is
:class:`~orpheus.numerics.symmetry.IdentityComponent` (a tuple of
skew generators, and nothing else) and
:class:`~orpheus.numerics.symmetry.Realization` (a component plus a
tuple of :class:`~orpheus.geometry.transformation.RigidMotion`
representatives, identity first). For a FINITE group the identity
component is trivial and the representatives are every element, so the
finite and continuous cases are one type rather than two arms.

.. list-table:: `[M]` 2026-09-03 — what the lattice realizes, over the
   27 spellings this page's fixtures build (26 distinct groups)
   :header-rows: 1
   :widths: 12 14 30 44

   * - :math:`\dim`
     - spellings
     - :math:`\mathfrak g`
     - members
   * - 0
     - 18
     - :math:`\{0\}`; :math:`G` is finite and its representatives are
       its elements
     - ``Trivial``, :math:`O_h`, :math:`I_h`, :math:`C_n`,
       :math:`D_{nh}`, :math:`\sigma_a` (:math:`C_1` is ``Trivial``,
       so 18 spellings are 17 groups)
   * - 1
     - 7
     - :math:`\mathbb R\,[\hat a]_\times` — a circle about :math:`\hat a`
     - :math:`SO(2)_a` (1 component), :math:`O(2)_a` (2),
       :math:`D_{\infty h}` (4)
   * - 3
     - 2
     - :math:`\mathfrak{so}(3)`
     - :math:`SO(3)` (1 component), :math:`O(3)` (2, the second
       generated by :math:`-I`)
   * - 2
     - **0**
     - —
     - :eq:`manifold-so3-subalgebras`: there is no such subalgebra, so
       there is no such subgroup to represent

:attr:`SubgroupOfO3.dim <orpheus.numerics.symmetry.SubgroupOfO3.dim>`
reads that number off the component. It is new at R1 and has no
production consumer yet: the dimension law on an orbit space —
:math:`\dim(M/H) = \dim M - \dim H` on the principal stratum,
:ref:`manifold-dimension-drop` — **will** read it when #434 R4 lands,
which is the carve that makes the lift a derivation output. Stating the
consumer as future tense is deliberate; today ``dim`` is a capability.

.. _manifold-one-body-per-question:

One body per question
~~~~~~~~~~~~~~~~~~~~~

Each row below was, before R1, between two and five branches on the
group's tag, plus (for the first two) a table of relations between
names. Each is now one expression on the realization.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - question
     - the one body
   * - :math:`g \in G`
     - :math:`g` lies in some coset: :math:`r^{-1}g \in G^0` for some
       representative :math:`r`. And :math:`h \in G^0` is decided by
       :math:`\dim`: the identity alone; a PROPER motion fixing
       :math:`\hat a`; any proper motion.
   * - :math:`H \subseteq G`
     - :math:`\mathfrak h \subseteq \mathfrak g` (a torus in a torus iff
       the axes are parallel; anything in :math:`\mathfrak{so}(3)`;
       :math:`\{0\}` in everything) **and** every representative of
       :math:`H` is an element of :math:`G`.
   * - :math:`H \subseteq \Gamma K` (a product SET)
     - the same two conjuncts, generalised on the second:
       :math:`\mathfrak h \subseteq \mathfrak k` **and** every
       representative :math:`r` of :math:`H` has SOME
       :math:`\gamma \in \Gamma` with :math:`\gamma^{-1} r \in K`.
       Added at #434 R3 for the registry's stage-0 coverage test; the
       derivation, and why :math:`\Gamma` must be finite, are at
       :ref:`manifold-coverage-by-a-product-section`. `[M]` with
       :math:`\Gamma = \{e\}` it is bit-equal to the row above on
       **441 of 441** ordered pairs.
   * - :math:`gGg^{-1} = G`
     - :math:`\mathrm{Ad}_g\,\mathfrak g = \mathfrak g` — for a torus,
       :math:`g\hat a = \pm\hat a`, since
       :math:`Q[\hat a]_\times Q^{\mathsf T} = \det Q\,[Q\hat a]_\times`
       and a span cannot see the sign — **and** every representative
       conjugates back into :math:`G`.
   * - :math:`G \subseteq N(H)`
     - the component exactly, through the Lie algebra
       (:eq:`manifold-normaliser-lie-criterion`), and each representative
       of :math:`G` one by one.
   * - :math:`G^0` fixes these points
     - :math:`Xp = 0` for every generator :math:`X` and every point
       :math:`p`. For a torus :math:`|\hat a \times p| = 0`, i.e.
       :math:`p` is ON the axis; for :math:`SO(3)`, :math:`p = 0`.
       This is one body where the tree carried two
       (``_is_axis_supported`` and ``_is_origin_supported``), and it is
       the criterion ERR-072 is about
       (:ref:`manifold-normaliser-sampling-control`).
   * - a generic set of elements
     - every element (finite), or the incommensurate rotations of the
       component composed with each representative (a torus) —
       :meth:`Realization.generic_images
       <orpheus.numerics.symmetry.Realization.generic_images>`. `[M]`
       2026-09-03: 1 / 2 / 4 / 48 images for ``Trivial`` /
       :math:`\sigma_z` / :math:`C_4` / :math:`O_h`; 6 / 12 / **24** for
       :math:`SO(2)_a` / :math:`O(2)_a` / :math:`D_{\infty h}` — six
       incommensurate angles times the number of components, so
       :math:`D_{\infty h}` is answerable for the first time, having had
       no arm of its own before. :math:`SO(3)` and :math:`O(3)` refuse
       by name: no axis to sample about, and no consumer yet.

⭐ **The tag survives, and survives as exactly one thing: the group's
IDENTITY and its name.** :func:`~orpheus.numerics.symmetry._realize` is
the single place it is read for structure, memoised on the tag because
the lattice walk asks :math:`O(n^3)` questions per measure and would
otherwise re-close the same group — `[M]` a single walk once rebuilt
:math:`I_h` 41 times, 9.3 s of a 9.4 s walk. Everywhere else the tag is
read only by ``name`` and ``__repr__`` and by the constructors.

⭐ **Computing rather than tabulating costs nothing here, because the
lattice walk's queries are overwhelmingly REPEATS and both layers are
memoised on immutable tags.** `[M]` 2026-09-03, cleared caches, one
:func:`~orpheus.numerics.invariance.symmetry_groups` walk per
rule, counted off ``functools.cache``'s own ``cache_info``:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - walk on
     - ``_tags_contain`` (queries → distinct)
     - ``_realize`` (reads → groups built)
   * - ``gauss_legendre(8)``
     - 420 → 203
     - 433 → **15**
   * - ``product(4, 8)``
     - 880 → 427
     - 995 → **22**
   * - ``lebedev(9)``
     - 1152 → 523
     - 1193 → **24**

Roughly half of every walk's containment questions are literally the
same question asked again, and a walk that asks a thousand structural
questions builds **fewer than 25 groups**. A hand-written table is not
buying speed over this; it is buying a second, unverifiable copy of the
answer.

.. _manifold-realization-one-spelling:

One group, one spelling — :math:`C_1` IS the trivial group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~orpheus.numerics.symmetry.SubgroupOfO3` is a frozen dataclass
whose ``__post_init__`` rewrites the tag ``Cn(1)`` to ``Trivial``. It is
the naming law of :ref:`manifold-orbit-space-stabiliser` applied one
tier down — *an object has one spelling* — and it is Pattern 4: the
second spelling is not refused, it is **unrepresentable**. `[M]`
2026-09-03: ``SubgroupOfO3.Cn(1) == SubgroupOfO3.Trivial`` is ``True``,
``repr`` reads ``SubgroupOfO3.Trivial``, and the hashes agree, so the
27 spellings this page probes are 26 groups.

Two spellings for one group is not a cosmetic defect. `[M]` before the
merge each ``contains`` the other and the two compare **unequal**, so
:func:`~orpheus.numerics.invariance._maximal` — which keeps a group only
when nothing else strictly contains it — dropped **both**, and a walk
that found :math:`\{e\}` reported the empty tuple; ``hash`` differed, so
a memo keyed on a :class:`~orpheus.numerics.manifold.Quotient` missed;
and ``SPHERE.quotient(Cn(1))`` answered *"no catalogue entry"*, the one
message the door promises never to give for a group it holds.

⚠ **Frozen, and that is load-bearing rather than hygiene.** `[M]`
2026-09-03 ``g._tag = "MUTATED"`` now raises
``FrozenInstanceError``; before R1 it SUCCEEDED, and because a
:class:`~orpheus.numerics.manifold.Quotient` carries the group as a
field, mutating it moved ``hash(quotient)`` out from under three memos
keyed on it. Every tag class was already frozen; the wrapper was the
one mutable link in the chain.

The identity component was wrong, and nothing read it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:attr:`SubgroupOfO3.identity_component
<orpheus.numerics.symmetry.SubgroupOfO3.identity_component>` is now read
off the realization: ``Trivial`` for every finite member (a finite
subgroup of :math:`O(3)` is discrete, so its identity component is
:math:`\{e\}`), :math:`SO(2)_a` for the axial families and
:math:`D_{\infty h}`, :math:`SO(3)` for :math:`SO(3)` and
:math:`O(3)`.

⛔ **Until 2026-09-03 it returned the group ITSELF for every finite
member.** `[M]` measured against a pinned pre-carve tree (``git archive
HEAD``, editable finder stripped, ``orpheus.__file__`` asserted), the
answer moves on **17 of the 27 spellings** — every finite one, from
:math:`O_h` and :math:`I_h` through every :math:`C_n`, :math:`D_{nh}`
and :math:`\sigma_a`. It contradicted its own docstring, which said
*"its orbits are connected, so it fixes every point"* — false of
:math:`O_h`, whose orbits are 48 points.

⭐ **Why nothing noticed, and it is the more useful half.** The property
had **zero readers**. The two places that genuinely needed the identity
component destructured a private helper instead
(``_continuous_decomposition``, now retired), so the public accessor was
a claim nobody could falsify by using it — ``vv-principles`` #17's
guard-with-no-witness, wearing a ``@property`` instead of a ``raise``.
Its first reader arrives with #434 R4. That ordering is the argument for
fixing it now rather than when the consumer lands: a wrong accessor with
no reader is indistinguishable from a right one until the day it is
read, and on that day the failure is attributed to the new consumer.

What the carve moved, and what it did not
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The behaviour contract, measured against the pinned pre-carve tree. Both
runs enumerate the same 27 spellings; the ``contains`` and ``normalises``
tables are the full :math:`27 \times 27` grids.

.. list-table:: `[M]` 2026-09-03 — pre-carve (``git archive HEAD``) vs the
   realization
   :header-rows: 1
   :widths: 34 22 44

   * - reading
     - moved
     - what that says
   * - ``A.contains(B)``
     - **0 of 729**
     - 109 lines of dispatch and two hand tables replaced with no
       change to a single answer
   * - ``A.normalises(B)``
     - **0 of 729**
     - the five per-family arms and the one Lie criterion agree
       everywhere they are both defined
   * - ``m.is_invariant_under(g)``
     - **0 of 270**
     - 10 shipped rules × 27 groups, the selection-facing predicate.
       ⛔ Spelled ``g.is_invariant(m)`` when this row was measured; R2 of
       #434 moved the verb onto the measure and deleted the group's
       method with no façade. The predicate is the same one, so the row
       stands; only the receiver and the argument changed places
       (:ref:`discrete-measure-invariance-module`).
   * - ``symmetry_groups``
     - **0 of 10 rules**
     - the walk's report is unchanged rule by rule
   * - the compatibility law
     - **0 violations, both sides**
     - 175 edges × 10 fixtures = 1750 (edge × fixture) pairs; the
       recorded 57/342 and 75/450 readings reproduce EXACTLY on both
       trees, which is what licenses reading 1750 as a widening of the
       same instrument rather than a different one
   * - ``g.identity_component``
     - **17 of 27**
     - the intended repair above
   * - ``g.orbit_stabiliser``
     - **1 of 27**
     - and only because ``C_1`` merged into ``Trivial``; the accessor's
       answers are otherwise untouched
   * - ``g.dim``
     - **new**
     - ``AttributeError`` on the pre-carve tree

⭐ The two zeros are the finding. A carve that deletes a hand-written
relation table is exactly the kind that can silently change an answer
nobody is looking at, and the honest instrument is the FULL grid rather
than the edges anyone thought to name — the same reason the
compatibility law is run at every denominator it has ever been run at
rather than only at the newest one.

.. _manifold-so2-axis-lattice:

What the axis buys in the containment lattice
----------------------------------------------

The parameter is not decoration on a name — it changes the **order**.
The two AXIAL families, :math:`SO(2)_a` and its stabiliser
:math:`O(2)_a` (:ref:`manifold-orbit-space-stabiliser`), are the
continuous ones *with* a parameter, so their edges depend on which axis
each side names and no enumeration over group NAMES can carry them.

⭐ **Every relation in the lattice is COMPUTED from the two groups'
realizations, and no relation between two groups is written down
anywhere** (:ref:`manifold-realization`, #434 R1, 2026-09-03).
:math:`H \subseteq G` holds iff :math:`\mathfrak h \subseteq
\mathfrak g` and every coset representative of :math:`H` is an ELEMENT
of :math:`G`; for the axial pair that unfolds to exactly the two
statements the retired arm spelled by hand — a finite group lies inside
:math:`O(2)_a` iff each of its elements fixes :math:`\hat e_a`, and
inside :math:`SO(2)_a` iff it is also proper, since
:math:`SO(2)_a = O(2)_a \cap SO(3)`.

⛔ **This paragraph named an arm, and the arm is gone.** It read, until
2026-09-03 and verbatim, that the axial families' relations "are
neither in the enum-to-enum table nor decidable finite-vs-finite, and
they live in their own arm (``symmetry._axial_contains``)", computed
through ``symmetry._fixes_axis``. True when written, and #434 R1
dissolved the
distinction it rests on: there is no enum-to-enum table, no
finite-vs-finite arm and no axial arm, because there is no per-family
dispatch left to arm. What survives verbatim is the tolerance — one
absolute band, :math:`10^{-9}`, for every element-level comparison a
realization makes (``symmetry._ELEMENT_ATOL``, renamed from
``_MEMBERSHIP_ATOL`` in the same carve so that the identically-spelled
:math:`10^{-12}` band in ``manifold.py`` — a POINT on a manifold, a
different question — stops making one grep answer twice).

⭐ **The history is why the paragraph exists at all, and it is now
twice over.** The hand-maintained table this section used to describe
had already contradicted itself: `[M]` it answered
``SO2('x') ⊉ C_1`` while ``SO2('x') ⊇ Trivial`` — **one group under two
spellings, two answers** — and a committed test pinned the wrong one.
#432 computed the axial edges and that contradiction went; #434 R1
computed the rest and removed the second spelling as well, since
``Cn(1)`` now normalises to the ``Trivial`` tag on the type
(:ref:`manifold-realization-one-spelling`). A computed relation cannot
disagree with itself, and a group with one spelling cannot be asked
twice.

`[M]` 2026-09-02, re-measured 2026-09-03 on the live tree over all
three axes:

.. list-table:: The two axial families' edges
   :header-rows: 1
   :widths: 34 16 50

   * - Relation
     - Holds
     - Why
   * - :math:`SO(2)_a \subseteq O(2)_a`, and not conversely
     - **every** :math:`a`
     - The identity component sits inside the stabiliser; the vertical
       mirrors are improper, so the reverse fails.
   * - :math:`SO(2)_a,\, O(2)_a \subseteq O(3)`
     - **every** :math:`a`
     - Both are subgroups of the full orthogonal group.
   * - :math:`SO(2)_a \subseteq SO(3)`; :math:`O(2)_a \not\subseteq SO(3)`
     - **every** :math:`a`
     - Proper rotations inside the proper rotations — and
       :math:`O(2)_a` carries :math:`\det = -1` elements.
   * - :math:`SO(2)_a,\, O(2)_a \subseteq D_{\infty h}`
     - :math:`a = z` **only**
     - :math:`D_{\infty h} = O(2)_z \times \{e, \sigma_h\}`; a rotation
       about :math:`x` does not preserve the :math:`z` axis.
   * - :math:`C_n \subseteq SO(2)_a,\, O(2)_a`, every :math:`n`
     - :math:`a = z` **only**
     - :math:`SO(2)_z = \bigcup_n C_n` in the standard setting. On
       :math:`x` and :math:`y` only :math:`C_1 = \{e\}` fits.
   * - :math:`\sigma_b \subseteq O(2)_a`
     - :math:`b \ne a` **only**
     - A mirror whose plane CONTAINS :math:`\hat e_a` fixes it;
       :math:`\sigma_a` flips it. (No mirror is inside any
       :math:`SO(2)_a` — reflections are improper.)
   * - :math:`D_{1h} \subseteq O(2)_x`
     - :math:`a = x` **only**
     - `[M]` :math:`D_{1h}` is realized as the order-4 Klein group
       :math:`\{e, \sigma_y, \sigma_z, C_2^x\}`, every element of which
       fixes :math:`\hat e_x` — a computed row a per-family table would
       have had to spell by hand.
   * - :math:`D_{nh} \subseteq O(2)_a`, :math:`n \ge 2`
     - **never**
     - :math:`D_{nh}` carries :math:`C_2` axes lying IN the plane, and
       :math:`\sigma_h`; each flips :math:`\hat e_a`.
   * - :math:`\{e\} \subseteq SO(2)_a,\, O(2)_a`
     - **every** :math:`a`
     - The identity fixes every axis and is proper.
   * - :math:`O(2)_a \subseteq O(2)_b` (and the :math:`SO(2)` pair),
       :math:`a \ne b`
     - **never**
     - Two distinct axial groups meet only in a set fixing two
       independent vectors.
   * - :math:`D_{\infty h},\, SO(3),\, O(3) \subseteq O(2)_a`
     - **never**
     - None of them fixes an axis: no finite or larger continuous group
       is inside an axial stabiliser.

⭐ **The lattice and the invariance predicate were re-checked against
each other, which is the gate that catches a wrong edge.** For every
asserted edge :math:`A \subseteq B` and every measure :math:`m`,
:math:`B\text{-invariant}(m) \Rightarrow A\text{-invariant}(m)` must
hold — the compatibility law (``vv-principles`` #15), which is the loop
that exposed ERR-072. `[M]` re-run 2026-09-01 over **15** groups (the
six named entries, three mirrors, three axial rotations, :math:`C_2`,
:math:`C_4`, :math:`D_{2h}`) × **6** fixtures (the declared slab
marginal, the chart-level :math:`\mu`-rule, a marginal declared about
:math:`z`, ``product(4,8)``, ``level_symmetric(4)``,
``folded_product(4,8)``): **0 violations over 342 (edge × fixture)
pairs.**

✅ **Re-run at 2026-09-02 with the three stabilisers** :math:`O(2)_a`
**added, and the denominator is the number that moved.** `[M]` over
**18** groups (the fifteen above plus ``O2('x')``, ``O2('y')``,
``O2('z')``) × the same **6** fixtures: **57 → 75 edges**, and **0
violations over 450 (edge × fixture) pairs.** The 15-group control was
re-run in the same script and reproduced its recorded reading exactly
(57 edges, 342 pairs, 0 violations), which is what licenses reading the
new number as a widening rather than a different instrument.

✅ **Re-run again at 2026-09-03, after #434 R1 made every edge a
computation on the realizations, at three denominators and on BOTH
trees.** `[M]` the 15-group and 18-group readings above reproduce
EXACTLY on the pinned pre-carve tree and on the carved one — 57/342/0
and 75/450/0, four readings, all identical — and the same law widened
to **27** group spellings × **10** shipped rules
(``gauss_legendre`` 8 and 16, ``product`` (4,8) and (4,4),
``folded_product`` (4,8) and (4,6), ``level_symmetric`` 4 and 8,
``lebedev`` 9 and 17) reads **175 edges** and **0 violations over 1750
(edge × fixture) pairs**, again on both trees. The old readings
reproducing is what makes 1750 a widening; the two trees agreeing is
what makes it a regression gate on the carve.

The most informative row is the one that could not exist before: `[M]`
``gauss_legendre_on_polar_orbit(8, "z")`` — the same eight nodes,
declared about :math:`z` — is invariant under :math:`SO(2)_z`,
:math:`O(2)_z`, :math:`D_{\infty h}` **and** :math:`C_4`, while the
slab's :math:`x`-declared rule is invariant under :math:`SO(2)_x`,
:math:`O(2)_x` and none of those three. Same floats, different groups,
because the axis says where the nodes are embedded.

⭐ **What the stabiliser does to the WALK's answer, and it is a
simplification.** :func:`~orpheus.numerics.invariance.symmetry_groups`
returns the MAXIMAL invariant candidates, so adding a group *above*
three existing answers removes them from the report. `[M]` 2026-09-02 on
``Quadrature.gauss_legendre(8).measure``, walking the same candidate set
with and without the three :math:`O(2)_a`:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - candidate set
     - maximal invariance groups
     - reading
   * - without :math:`O(2)_a` (pre-#432)
     - ``{SO2_x, sigma_x, sigma_y, sigma_z}`` — **four**
     - Three of the four are *incomparable* only because the group that
       contains :math:`\sigma_y` and :math:`\sigma_z` could not be
       spelled.
   * - with :math:`O(2)_a` (ships)
     - ``{O2_x, sigma_x}`` — **two**
     - :math:`\sigma_y, \sigma_z \in O(2)_x` and are absorbed;
       :math:`\sigma_x` survives because it FLIPS :math:`\hat e_x` and
       so lies in no :math:`O(2)_x`. The pair is exactly *what the rule
       spent* beside *what it still has*.

`[M]` the walk agrees with brute force — every invariant candidate
enumerated, then reduced to maximals by hand — on **6 of 6** shipped
rules (the slab marginal, the chart :math:`\mu`-rule,
``product(4,8)``, ``level_symmetric(4)``, ``folded_product(4,8)``,
``lebedev(11)``), and the three extra candidates cost **11.3 – 26.2 %**
per walk (min over 15 interleaved repeats, host ``.venv``, CPython 3.14,
a loaded machine: slab 5.0 → 5.6 ms, ``folded_product(4,8)`` 103.5 →
115.2 ms, ``product(4,8)`` 107.5 → 135.6 ms, ``lebedev(11)`` 283.8 →
329.9 ms, ``level_symmetric(4)`` 320.8 → 367.1 ms). ⚠ The walk is not
on any solve's hot path — it is a construction-time report — but the
cost is written here rather than left to a reader's estimate, because a
per-candidate cost is what a fourth family would pay again.

.. _manifold-quotient-is-memoised:

The derivation runs once — ``Manifold.quotient`` is memoised
-------------------------------------------------------------

An orbit space is **derived once and recorded**; that is the
catalogue's whole philosophy, and until 2026-09-01 nothing enforced it,
because nothing on a hot path asked for a quotient. Tracker 2.4 changed
that: *every slab quadrature now carries one*, so an unmemoised
:meth:`Manifold.quotient
<orpheus.numerics.manifold.Manifold.quotient>` would put a SymPy
derivation on the construction path of every slab solve.

`[M]` 2026-09-01, ``SPHERE.quotient(SubgroupOfO3.SO2('x'))`` on this
machine (host ``.venv``, CPython 3.14), cache cleared before the cold
reading — the call is spelled ``O2('x')`` since #432 and the entry it
returns is the same object:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Reading
     - Cost
     - Note
   * - cold (the SymPy derivation)
     - **6.43 ms**
     - Gradients, the ``simplify`` chain, the determinant.
   * - warm (mean over 1000 lookups)
     - **0.76 µs**
     - ``CacheInfo(hits=1000, misses=1)``.
   * - ratio
     - :math:`\approx 8500\times`
     -

Two properties make the memo *legal*, and both are worth stating
because a memo on the wrong object is a shared-mutable-state bug:
every :class:`~orpheus.numerics.manifold.Manifold` is a **frozen value
type** and hashable — which is why tracker 2.4 also retyped every
builder's ``gram`` to :class:`sympy.ImmutableMatrix`, since a mutable
``Matrix`` field makes the whole ``Quotient`` unhashable — and the
returned :class:`~orpheus.numerics.manifold.Quotient` is itself frozen,
so sharing one object across callers is safe for the same reason.

⭐ **A second memo landed in the same step, and it is the larger
number.** The containment machinery asks
:func:`~orpheus.numerics.symmetry.SubgroupOfO3.contains` for a *realized
operator set* before it consults any lattice table, so every question
involving :math:`I_h` rebuilt the icosahedral group's 120-element
closure. That was tolerable while the invariance walk offered one axial
rotation group; offering **three** multiplied the traffic. `[M]` on this
machine: ``_icosahedral_ops`` costs **155 ms** cold and returns 120
elements; a single ``symmetry_groups`` walk calls it **33**
times on the slab marginal and **48** times on ``product(4, 8)`` — so
without the memo the slab walk alone would spend :math:`33 \times 155\
\text{ms} \approx 5.1\ \text{s}` rebuilding a constant. Memoised, `[M]`
the same walk is **5.3 ms** warm (197.5 ms on the first call of a fresh
process, which is the one cold build).

.. _manifold-dimension-drop:

Consequence 1 — the dimension drops by the GENERIC ORBIT's, and that is a construction invariant
--------------------------------------------------------------------------------------------------

:math:`\dim S^2 = 2`, :math:`\dim O(2)_a = 1`, and the quotient has
dimension 1. That is the generic count: the orbits are the
constant-latitude circles, each 1-dimensional, and the quotient records
one number per orbit. It is *only* generic — at the two poles the orbit
is a single point, and the drop there is 2, not 1. Which is the next
consequence.

⛔ **And the group's dimension is a COINCIDENCE here, not the law.**
This heading read *"the dimension drops by the group's"* until
2026-09-03, which is true of :math:`O(2)_a` on :math:`S^2` and false in
general. What the dimension of an orbit space is equal to is

.. math::
   :label: manifold-orbit-dimension-law

   \dim (M/H) \;=\; \dim M \;-\; \dim\bigl(\text{generic }H\text{-orbit}\bigr),
   \qquad
   \dim\bigl(H \cdot p\bigr) \;=\;
   \operatorname{rank}\,\bigl\{\, X p \;:\; X \in \mathfrak h \,\bigr\} ,

where :math:`\mathfrak h` is :math:`H`'s Lie algebra — the tangent space
of the orbit through :math:`p` is the image of :math:`\mathfrak h` under
:math:`X \mapsto Xp`, so its dimension is that map's rank. A finite group
has :math:`\mathfrak h = \{0\}` and zero-dimensional orbits, and is
answered without asking about a point at all; a positive-dimensional one
need not act with orbits of its own dimension, because the ACTION can
have a stabiliser.

⛔ **"At a generic point" is a computation over a SET, and the
single-point spelling is measurably wrong.** Orbit dimension is upper
semicontinuous — it can only DROP on the singular stratum — so the
generic value is the **maximum** over a probe set, never the value at one
chosen row. `[M]` 2026-09-03 (elegance review) with one probe row placed
ON the axis, a single-point spelling of the law *refused* the honest
:math:`S^2/O(2)_z` (the orbit there is a point, rank 0, so the law read
:math:`2-0 = 2` against a 1-D realization) and *admitted* the disk
forgery — both errors at once, from one unlucky row. The shipped
:func:`~orpheus.numerics.manifold._generic_points` therefore returns a
SET and
:meth:`SubgroupOfO3.generic_orbit_dimension
<orpheus.numerics.symmetry.SubgroupOfO3.generic_orbit_dimension>` takes
the maximum over it: `[M]` **9** seeded rows for :math:`S^2`, **4** for a
flat ambient base, and on every shipped pair the per-row values are
already unanimous (:math:`O(2)_a` reads ``1`` on 9 of 9,
:math:`O(3)` reads ``2`` on 9 of 9) — so the maximum is not *needed* by
any shipped entry and is what makes "generic" a computation rather than a
claim about a lucky draw.

⭐ **Two counter-examples, both of which** :math:`\dim M - \dim H`
**would get wrong, and one of them is a live issue.**

.. list-table:: `[M]` 2026-09-03 — where the two readings part
   :header-rows: 1
   :widths: 26 14 20 20 20

   * - :math:`H` acting on :math:`M`
     - :math:`\dim H`
     - :math:`\dim` generic orbit
     - :math:`\dim M - \dim` orbit
     - :math:`\dim M - \dim H`
   * - :math:`O(2)_a` on :math:`S^2`
     - 1
     - 1
     - **1** ✅
     - 1 ✅
   * - :math:`\langle\sigma_a\rangle` on :math:`S^2`
     - 0
     - 0
     - **2** ✅
     - 2 ✅
   * - :math:`O(3)` on :math:`S^2`
     - 3
     - 2
     - **0** ✅
     - ⛔ :math:`-1`
   * - :math:`SO(3)` on :math:`\mathbb{R}^3`
     - 3
     - 2
     - **1** ✅
     - ⛔ :math:`0`

:math:`S^2` is a SINGLE :math:`O(3)`-orbit, so :math:`S^2/O(3)` is a
point — dimension :math:`0`, not :math:`-1`; the stabiliser of a
direction is a whole :math:`O(2)`, and the orbit is the sphere, not a
copy of the group. And :math:`\mathbb{R}^3/SO(3)` is the ray
:math:`[0,\infty)` — the radius survives — dimension :math:`1`, not
:math:`0`. The first of those is `GitHub #440
<https://github.com/deOliveira-R/ORPHEUS/issues/440>`_'s entry, which is
why the law had to be stated on the ORBIT before that entry could be
catalogued: a law written on :math:`\dim H` would have refused it at
construction.

⭐ **It is a construction invariant, not a remark.** Since 2026-09-03
(#434 R4) :meth:`Quotient.__post_init__
<orpheus.numerics.manifold.Quotient>` evaluates
:eq:`manifold-orbit-dimension-law` and refuses a mismatch, so a chart
REALIZES an orbit space's dimension rather than declaring one, and
``dataclasses.replace`` re-runs the check like every other invariant on
this type. The rank is read off the group's own realization
(:ref:`manifold-realization`) — its identity component's skew generators
applied to a generic point of the base — so the law is computed from
what the group IS, and #434 R1's new
:attr:`~orpheus.numerics.symmetry.SubgroupOfO3.dim` is its first
production consumer.

⛔ **What it caught, measured before it landed.** `[M]` 2026-09-03: with
the stabiliser law of :ref:`manifold-orbit-space-stabiliser` in place but
no dimension check, a **forged** :math:`S^2/O(2)_z` realized on the
DISK and a **forged** :math:`S^2/\sigma_x` realized on
:math:`[-1,1]` both CONSTRUCTED, and each compared unequal to the
catalogue entry it claims to be — ERR-080's defect class (one orbit
space, two objects, one of them lying about where its points live)
reopened one field over from the one #432 closed. Both are refused now,
naming the law:

.. code-block:: text

   S^2/O2_z: dim(M/H) = dim S^2 - dim(generic O2_z-orbit) = 1, but the
   realization 'D^2' has dim 2. A chart of an orbit space realizes its
   dimension; check the elimination against S^2's own ideal.

   S^2/sigma_x: dim(M/H) = dim S^2 - dim(generic sigma_x-orbit) = 2, but
   the realization '[-1,1]' has dim 1. A chart of an orbit space realizes
   its dimension; check the elimination against S^2's own ideal.

⭐ **FOUR clauses, in this order, each with an input only IT rejects** —
which is what stops the later ones being decoration certified by an
earlier one's witness (``vv-principles`` #17's per-arm rule: a guard has
as many claims as it has early returns).
:meth:`~orpheus.numerics.manifold.Quotient.__post_init__` asks, in turn:
is ``by`` the orbits' full stabiliser
(:ref:`manifold-orbit-space-stabiliser`); does the realization have the
dimension :eq:`manifold-orbit-dimension-law` forces; does the lift land
in the base's **ambient** space; and — when a section ships — does the
fundamental domain agree with the realization
(:ref:`manifold-two-coordinate-systems`). `[M]` 2026-09-03, one input
per clause, each run against the live tree:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Clause
     - An input only IT rejects
     - The message it names
   * - 1. stabiliser
     - ``by=SO2("x")`` on the axial entry. `[M]` it PASSES clause 2 —
       :math:`SO(2)_x` and :math:`O(2)_x` have the same orbits, so both
       report generic orbit dimension ``1`` — which is precisely why
       clause 1 cannot be folded into it
     - *"S^2/SO2_x is the orbit space S^2/O2_x: SO2_x and its
       stabiliser O2_x … have the same orbits"*
   * - 2. dimension
     - :math:`\sigma_x` against ``realization=[-1,1]``
     - *"dim(M/H) = … = 2, but the realization '[-1,1]' has dim 1"*
   * - 3. lift codomain
     - the axial entry with ``lift_codomain=Ball(2)`` or ``[-1,1]``
     - *"the lift lands in S^2's ambient space (3 columns), but
       lift_codomain 'D^2' has 2"*
   * - 4. fundamental domain
     - :math:`\sigma_y` on the disk with a **half-meridian** domain
       (normals :math:`\{\hat e_y, -\hat e_y\}`, `[M]` ``dim 1``)
     - *"the fundamental domain … has dim 1 but the realization 'D^2'
       has dim 2"*

⭐ **Clause 3 is the one the lift's own field made necessary, and it is
about a CONSUMER rather than about geometry.** :func:`_act_through`
hands the lift's output straight to
:meth:`RigidMotion.on_points
<orpheus.geometry.transformation.RigidMotion.on_points>`, so a codomain
of the wrong width is a codomain a consumer would read and be misled by
— which is ERR-080's shape with the arrow pointing the other way. The
same reasoning is why
:attr:`~orpheus.numerics.manifold.Quotient.lift_codomain` is **compared**
while the two coordinate maps beside it are not: a manifold HAS value
equality where a function does not, and `[M]` 2026-09-03 with the field
excluded ``dataclasses.replace(entry, lift_codomain=SPHERE)`` compared
**EQUAL** to the catalogue entry — and since :func:`barycentre` is
memoised on the entry, whichever of the two was asked first answered for
both. Two orbit spaces disagreeing about where their lift lands are two
objects, and the field that exists to refuse ERR-080 had re-minted it.
`[M]` today that ``replace`` compares **unequal**, and the width gate
refuses it outright.

⚠ **The fd clause's discriminating input MOVED when the dimension
clause landed**, and this is exactly the trap of ordering guards.  Until
2026-09-03 the fd clause was shown with *"a hemisphere offered against a
1-D realization"* — and `[M]` that input is now rejected one clause
earlier, by the dimension law, because a :math:`\sigma_a` entry's
realization must be 2-dimensional whatever domain is offered. The fd
clause still bites; the witness is now a domain whose dimension is
**too low** rather than a realization whose dimension is too low.

⚠ And the reason the fd clause needs a fresh input is *ordering*, not
absence: `[M]` 2026-09-03 the :math:`\sigma_x` forgery ships a real
``fundamental_domain`` — the hemisphere ``S^2|x>=0``, ``dim`` 2 — which
against a 1-D realization violates clause 4 as well. Clause 2 simply runs
first. (The other forgery, :math:`O(2)_z` on the disk, does carry
``fundamental_domain=None`` and would reach clause 4's early return.) A
guard's arms are as many claims as it has early returns, and an arm whose
only witness is another arm's input is certified by nothing.

.. (vv-status rationale) manifold-orbit-dimension-law is a STRUCTURAL
   theorem of Lie group actions, transcribed here so the catalogue's
   construction invariant has a name and its two counter-examples have a
   home. It makes no solver claim — no flux, no eigenvalue, no
   convergence order. What IS verifiable is the invariant it induces on
   Quotient.__post_init__ (every shipped entry satisfies it; the two
   forgeries above are refused), gated by
   tests/numerics/test_manifold.py::TestR4AnOrbitSpacesDimensionIsATheorem,
   whose rows carry @pytest.mark.foundation and deliberately NO
   verifies(...) per vv-principles' foundation-tier rule.
.. vv-status: manifold-orbit-dimension-law documented

.. _manifold-singular-stratum:

Consequence 2 — the action is not free, so the quotient is an ORBIFOLD
-----------------------------------------------------------------------

The poles :math:`\mu = \pm 1` lie on the rotation axis, so their
stabilizer is the whole of :math:`SO(2)`: the action there has a fixed
point, the orbit collapses from a circle to a point, and the
pushforward of the uniform measure vanishes. The image of the
fixed-point set is the **singular stratum**.

Concretely, :math:`[-1,1]` is a manifold *with boundary* — an orbifold
— not a quotient manifold, and

.. math::

   \det P \big|_{S^2} \;=\; 4\,(1-\mu^2) \;=\; 0
   \quad\Longleftrightarrow\quad \mu = \pm 1 :

**the stratum is exactly where** :math:`\det P` **vanishes**, which is
why the shipped entry *derives* it rather than declaring it (the
foundation gate solves :math:`\det P = 0` and compares against
``singular_stratum``). Anything designed on the assumption that a
quotient is a smooth submersion is wrong there — and *only* there,
which is what makes the stratum worth carrying as a field rather than
a caveat.

⭐ Two shipped objects already live on that stratum, from opposite
directions:

- the curvilinear S\ :sub:`N` :math:`\alpha`-dome **closes** at
  :math:`\mu = \pm 1`, because the redistribution coefficient
  :math:`(1-\mu^2)` vanishes there
  (:doc:`/theory/methods/sn/curvilinear_one_group`);
- the spherical-harmonic evaluator's ``on_axis`` guard fires when
  :math:`\sin\theta \approx 0`, i.e. on directions *along* the polar
  axis — the same locus, detected numerically and named nothing.

⭐ The second catalogued entry has an exact cylindrical analogue of the
first bullet, measured on production data: the fold's stratum is the
disk's boundary circle, the shipped quadrature nodes sit strictly
inside it, and the march seeds sit exactly **on** it, where the
:math:`\alpha`-dome closes (:ref:`manifold-orbifold-discretised`).

.. _manifold-one-polynomial:

Consequence 3 — one polynomial, three appearances
--------------------------------------------------

:math:`(1-\mu^2)` shows up three times in this corpus. They are the
same polynomial; whether they are the same *object* is three different
questions, with three different answers, and conflating them is the
error this subsection exists to prevent.

.. list-table:: :math:`(1-\mu^2)`, by epistemic status
   :header-rows: 1
   :widths: 26 44 30

   * - Appearance
     - What it is
     - Status
   * - the squared orbit radius
     - a point of :math:`S^2` at latitude :math:`\mu` sits at distance
       :math:`\sqrt{1-\mu^2}` from the rotation axis, so its
       :math:`SO(2)`-orbit is a circle of that radius and
       :math:`\det P = 4\,r_{\rm orbit}^2`
     - **DERIVED** on this page, from
       :eq:`manifold-procesi-schwarz`
   * - the redundant harmonic
     - on a 1-D rule the retained :math:`Y_2^{+2}` column is
       :math:`\propto (1-\mu^2)`, which is what makes the discrete
       Gram rank-deficient
     - **MEASURED** — see below
   * - the angular-redistribution coefficient
     - the :math:`(1-\mu^2)/r \cdot \partial_\mu \psi` term in the
       spherical streaming operator
     - ⚠ **an identity of polynomials only.** The *mechanism* —
       that the redistribution term is the connection of the
       phase-space quotient — is **unproved**

**The measured one.** `[M]` 2026-08-31, reading the live frame's own
basis and measure at :math:`L = 2`. The coefficient array has shape
:math:`(3,5) = 15` slots, of which **9** are actual
:math:`(\ell, m)` harmonics and 6 are padding. On a genuine 3-D rule
all 9 light up: ``lebedev(11)`` gives **9 live of 15**. On the slab
``gauss_legendre(8)`` only **5** do —
:math:`\{(0,0),(1,0),(2,0),(2,1),(2,2)\}` — and the discrete Gram
over those five has ``matrix_rank`` **4**, on live singular values
:math:`2.70755,\; 1.41922,\; 4.92450\times10^{-1},\;
4.74468\times10^{-2}`. So one of the five is a linear combination of
the others.

Which one is the point: dividing the :math:`(2,2)` column by
:math:`(1-\mu^2)` gives a **constant** :math:`0.866025`, with a spread
of :math:`8.9\times10^{-16}` across all eight nodes. The column *is*
that polynomial, to round-off — and :math:`(1-\mu^2)` is
:math:`\det P / 4`, which on the quotient is a function of
:math:`\mu` alone and therefore already spanned. The rank deficiency
is a theorem about :math:`S^2/O(2)_x`, not a conditioning accident.

⚠ Do not quote a fifth singular value. It is a noise-floor reading and
`[M]` does not reproduce between runs; the reproducible statements are
the four live values, the counts **9 of 15** (3-D) versus **5 of 15**
(slab), and the rank **4**.

⚠ **And do not read the third row as settled.** The polynomial identity
is real and it is suggestive — the same expression appears as the
orbit-space boundary and as the coefficient that vanishes at the same
locus — but "the curvilinear redistribution term *is* the quotient's
connection" is a claim about a **reduction that has not been carried
out**. It is Phase 1.3 of #429, whose stated done-when admits exactly
two outcomes: a derivation, or an explicit ruling that the coincidence
is accidental in 1-D spherical geometry. Until one of those lands, the
honest statement is the one in the table: three occurrences of one
polynomial, one of them derived here, one of them measured, one of them
open.

⭐ The second catalogued entry produces the **cylindrical twin** of that
open row — :math:`1 - \eta^2 - \mu^2` is simultaneously the fold's
:math:`\det P` on :math:`S^2` and the locus on which the cylindrical
:math:`\alpha`-dome closes and the march seeds sit, measured exactly
(:ref:`manifold-orbifold-discretised`). It is the same *kind* of
statement as the third row here — an identity of loci with an unproved
mechanism — so closing one does not close the other, and neither may be
cited for the other.

.. _manifold-gelfand:

Consequence 4 — the quotient is a Gelfand pair, so :math:`\Lambda` is forced
-----------------------------------------------------------------------------

:doc:`/theory/foundations/spherical_harmonics` and
:doc:`/theory/foundations/frame` already own the statement that the
P\ :math:`_\ell` scattering kernel factors as :math:`R\,\Lambda\,M` with
:math:`\Lambda` diagonal, and the derivation (Funk–Hecke plus Schur's
lemma, read as the spectral theorem :math:`A = U\Sigma U^{*}`) lives at
:eq:`sh-funk-hecke-eigenvalue` and :ref:`frame-eigenbasis-ownership`.
**Edited there, consumed here.** What this page adds is the *quotient*
register — the reason the same factorization is forced by the orbit
space, stated in the vocabulary of :math:`S^2/O(2)_a` rather than of the
scattering operator.

:math:`S^2` is itself a homogeneous space, :math:`S^2 = SO(3)/SO(2)`,
and :math:`(SO(3), SO(2))` is a **Gelfand pair** — the convolution
algebra of :math:`SO(2)`-bi-invariant functions on :math:`SO(3)` is
commutative. So the object under discussion is really the double coset
space

.. math::

   SO(2) \,\backslash\, SO(3) \,/\, SO(2),

whose bi-invariant functions are the **zonal spherical functions** of
the pair — for :math:`(SO(3), SO(2))` exactly the Legendre polynomials
:math:`P_\ell`. Commutativity of that algebra is what forces every
:math:`SO(3)`-equivariant zonal operator to be simultaneously
diagonalised with a multiplier depending on :math:`\ell` alone. Read
this way, :math:`R\,\Lambda\,M` with :math:`\Lambda` diagonal is a
**theorem of harmonic analysis on the quotient**, not a chosen
factorization that happens to work — which is the same conclusion the
two pages above reach from the operator side, by a different route.

⭐ The same lens names the repair ERR-080 needs. The sub-basis that
fixes a 1-D chart is not a "zonal special case" bolted on for slabs; it
is the **trivial isotypic component** of the :math:`SO(2)` action —
:math:`\{Y_\ell^0\} \cong \{P_\ell\}` — and that says *why those
slots and not others*. It is also the reason the repaired Gram is
expected to be exactly diagonal rather than merely better conditioned:
Gauss–Legendre integrates :math:`P_\ell P_{\ell'}` exactly to degree
:math:`2N-1`, so the Gram becomes exactly :math:`2/(2\ell+1)` on the
diagonal. The falsifiable form of that prediction is recorded with
ERR-080, not here.


.. _manifold-second-entry:

The second entry, and the two coordinate systems it forced
==========================================================

The first catalogued entry answered the question *what is*
:math:`M/H` with a single object, and
:class:`~orpheus.numerics.manifold.Quotient` stored it in a single
field, ``realization``, which both
:meth:`contains <orpheus.numerics.manifold.Quotient.contains>` and the
ambient-width helper read. The second entry — the **shipped cylindrical
fold** :math:`S^2/\langle\sigma_y\rangle`, which
:meth:`Quadrature.folded_product
<orpheus.numerics.quadrature.directional.Quadrature.folded_product>`
performs on every curvilinear rule — cannot be stored that way, and the
reason is not a detail of the entry. It is that an orbit space has
**two** honest coordinate systems, and the tree produces data in both.

⚠ **Why the first entry could not expose the fork — and it is not the
dimensions.** The chart codomain and a section have the *same* ``dim``
in **both** entries, so dimension cannot discriminate them: `[M]`
``Interval(-1,1).dim`` is :math:`1` and an :math:`SO(2)` half-meridian
written as ``FundamentalDomain(SPHERE, (e_y, -e_y, e_x), …).dim`` is
also :math:`1`; ``Ball(2).dim`` is :math:`2` and the
:math:`\sigma_y` hemisphere ``FundamentalDomain(SPHERE, (e_y,), …).dim``
is also :math:`2`. Indeed the agreement is now a **construction law**,
gated in
:meth:`Quotient.__post_init__ <orpheus.numerics.manifold.Quotient>` —
a quantity that must always agree cannot tell two cases apart. Two
measured facts hid the fork instead:

1. **No section of** :math:`S^2 \to S^2/O(2)_a` **is canonical.** Every
   half-meridian is one and none is distinguished, so there was nothing
   to put in a second slot even had one existed. That is the normal
   situation for a positive-dimensional group.
2. **The tree's** :math:`SO(2)` **data is already in chart
   coordinates.** `[M]` ``gauss_legendre(8).measure.nodes`` has shape
   :math:`(8,)` and holds the invariant :math:`\mu` itself, so the
   realization and the data speak the same language and nobody had to
   choose. `[M]` ``folded_product(4,8).measure.nodes`` has shape
   :math:`(16,3)` — the base's ambient columns — so for the second
   entry the same slot cannot even *see* the tree's own nodes.

.. _manifold-s2-sigma-y:

The derivation — :math:`S^2/\langle\sigma_a\rangle` is the closed disk
-----------------------------------------------------------------------

The procedure of :ref:`manifold-derivation-procedure`, run in full.
Write :math:`a` for the mirrored axis and :math:`b, c` for the other
two; the shipped fold is :math:`a = y`. Every line below was re-derived
in this session, independently of the catalogue entry, and then
compared against it.

**Step 0 — the group.**
:math:`\sigma_a : x_a \mapsto -x_a`, with
:math:`H = \langle\sigma_a\rangle = \{e, \sigma_a\}` of order 2. `[M]`
:math:`\det\sigma_y = -1` and :math:`\sigma_y^2 = I`. That
determinant is not decoration: :math:`\sigma_a` is an **improper**
element, and specifically a *reflection* — it fixes a hyperplane
pointwise — which predicts steps 1 and 3 before either is run.

**Step 1 — the invariants.** A polynomial is :math:`\sigma_a`-invariant
iff it is **even in** :math:`x_a`, so

.. math::

   p_1 = x_b, \qquad p_2 = x_c, \qquad p_3 = x_a^2 .

`[M]` verified symbolically, with two non-invariant controls: for
:math:`x` and :math:`z` and :math:`y^2` the difference
:math:`p(\sigma_y x) - p(x)` is :math:`0`, while for the controls
:math:`y` and :math:`xy` it is :math:`-2y` and :math:`-2xy`. A check
that passes on everything is not a check.

*Completeness*, by Molien's series — not by eyeballing. Molien's
formula gives the Hilbert series of :math:`\mathbb{R}[x]^H` from the
group alone, and `[M]`

.. math::

   M(t) \;=\; \tfrac12 \sum_{g \in H} \frac{1}{\det(I - t g)}
         \;=\; \frac{1}{(1-t)^2 (1-t^2)}
         \;=\; 1 + 2t + 4t^2 + 6t^3 + 9t^4 + 12t^5 + O(t^6),

which is *exactly* the Hilbert series of a free polynomial algebra on
generators of degrees :math:`1, 1, 2`. `[M]` the difference of the two
series simplifies to :math:`0`. That single equality carries two facts
at once: **completeness** — the subalgebra
:math:`\mathbb{R}[x_b, x_c, x_a^2]` has the same graded dimension as
the whole invariant ring in every degree, so it *is* the whole ring —
and **freeness**, i.e. an empty syzygy ideal, which step 3 then
re-derives independently.

*Minimality*, by counting :math:`\dim(\mathfrak{m}/\mathfrak{m}^2)`
degree by degree, where :math:`\mathfrak{m}` is the ideal of
positive-degree invariants: `[M]` degree 1 contributes **2** new
generators, degree 2 contributes **1**, and degrees 3–5 contribute
**0** — so the minimal generating set has exactly three members.

.. warning::

   **A trap in that count, recorded because it produced a
   self-consistent wrong answer.** A "decomposable" is a product of
   **two or more** positive-degree invariants. Counting products of
   :math:`k \ge 1` factors instead includes the generators themselves,
   and then every degree reports *"0 new generators needed"* — i.e. the
   empty set generates the invariant ring. The output is internally
   consistent and completely wrong. The predicate is :math:`k \ge 2`.

**Step 3 — the syzygy ideal is empty, and predictably so.** `[M]` the
elimination
:math:`\langle u_1 - x_b,\, u_2 - x_c,\, u_3 - x_a^2\rangle \cap
\mathbb{R}[u]` returns a Gröbner basis with **no** :math:`u`-only
generator, so :math:`I = (0)`; the Jacobian
:math:`\partial(p_1,p_2,p_3)/\partial(x,y,z)` has determinant
:math:`\pm 2 x_a` — `[M]` exactly :math:`-2y` for the shipped
:math:`a = y` ordering, the sign being an artefact of which two
coordinates are kept first — and generic rank **3**, equal to the
number of invariants. It is the **rank** that carries the argument.

⭐ **But the theorem is better than the computation here.** A
reflection generates a *reflection group*, so by
**Chevalley–Shephard–Todd** its invariant ring is a polynomial ring —
hence free, hence :math:`I = (0)`. The elimination is the mechanical
route; the answer was never in doubt. This is a real structural
contrast with :math:`S^2/O(2)_a`, whose syzygy ideal is *also* empty but
for the unrelated reason that its two invariants happen to be
algebraically independent: neither :math:`SO(2)_a` nor its
stabiliser :math:`O(2)_a` is a **finite** reflection group — the
Chevalley–Shephard–Todd theorem is about finite groups — so no such
theorem applies to either.

**Step 4 — Procesi–Schwarz.** The three gradients
:math:`\nabla p_1 = e_b`, :math:`\nabla p_2 = e_c`,
:math:`\nabla p_3 = 2 x_a e_a` are mutually orthogonal, so :math:`P` is
diagonal and :math:`P \succeq 0` collapses to a single inequality:

.. math::
   :label: manifold-s2-mod-mirror

   P \;=\; \operatorname{diag}\bigl(1,\, 1,\, 4 p_3\bigr),
   \qquad
   \det P \;=\; 4 p_3 ,
   \qquad\text{so}\qquad
   \mathbb{R}^3 / \langle\sigma_a\rangle \;=\; \{\, p_3 \ge 0 \,\}.

.. (vv-status rationale) manifold-s2-mod-mirror is the second INSTANCE
   of :eq:`manifold-procesi-schwarz`, for the S^2/<sigma_a> catalogue
   entry, and is classified exactly as its sibling
   :eq:`manifold-s2-mod-so2` is, for the same structural reason. Its
   content IS checked, and tightly: the P-matrix and its determinant are
   recomputed symbolically and compared with sp.simplify, the syzygy is
   asserted empty, and the stratum is SOLVED for (and shown to be a
   one-parameter family, i.e. a curve) rather than compared to a
   literal, by tests/numerics/test_manifold.py::
   TestTheSigmaYFoldIsExpressibleAndDiscriminating::{
   test_the_derivation_reproduces_procesi_schwarz,
   test_the_stratum_is_a_CIRCLE_not_a_point_set}. Those gates carry
   @pytest.mark.foundation and deliberately NO verifies(...): they
   assert an intrinsic law of a data type, with no flux, eigenvalue or
   convergence claim behind them, and vv-principles' foundation tier
   carries no verifies marker by rule. A verifies edge here would mint a
   coverage claim of a class the gates do not make.
.. vv-status: manifold-s2-mod-mirror documented

**Step 5 — restrict to the sphere.** Adjoining the sphere's ideal
:math:`p_1^2 + p_2^2 + p_3 = 1` makes :math:`p_3` *eliminable*, and
what remains is two-dimensional:

.. math::

   \det P \big|_{S^2} \;=\; 4\,\bigl(1 - p_1^2 - p_2^2\bigr),
   \qquad
   S^2/\langle\sigma_a\rangle \;\cong\;
   \bigl\{\, (p_1, p_2) : p_1^2 + p_2^2 \le 1 \,\bigr\} \;=\; D^2 .

`[M]` **the re-derivation agrees with the shipped entry exactly.** The
construction, so it regenerates from this page: form the three
gradients of :math:`(x, z, y^2)` symbolically, build
:math:`P_{ij} = \langle\nabla p_i, \nabla p_j\rangle`, substitute
:math:`y^2 \to p_3` to re-express it in the invariants, then compare
against ``SPHERE.quotient(SubgroupOfO3.Mirror("y"))`` with the entry's
own generator symbol mapped onto :math:`p_3`. ``sympy.simplify`` of the
difference is the zero :math:`3\times3` matrix, and of the
determinants, :math:`0`.

⚠ The substitution step has a trap the sibling entry records
(:ref:`manifold-s2-so2`), and it does **not** bite here: ``subs``
matches syntactic nodes, so ``subs(x**2 + y**2, p2)`` silently fails on
:math:`4x^2+4y^2` — whereas ``4*y**2`` literally contains the node
``y**2``, so ``subs(y**2, p3)`` succeeds. A derivation that "happens to
work" is still worth asserting: check that no free :math:`x, y, z`
remains in :math:`P` after the substitution.

⭐ **Step 5 supplies the equality that the syzygy ideal did not**, and
that is worth stating before an engine is written. In *both* catalogued
entries ``syzygy`` is honestly ``()``, and in both the real equality
arrives at step 5 — :math:`p_1^2 + p_2 = 1` for :math:`O(2)_a`,
:math:`p_1^2 + p_2^2 + p_3 = 1` here. An engine that emits only
:math:`I` has emitted only half the equalities.

In transport coordinates the invariants are the direction cosines
themselves: :math:`p_1 = \mu_x = \eta` (the radial cosine),
:math:`p_2 = \mu_z = \mu` (the axial cosine), and the eliminated
:math:`p_3 = \mu_y^2 = \xi^2`. So the orbit space of the shipped
cylindrical fold is

.. math::

   S^2/\langle\sigma_y\rangle
   \;=\; \{\, (\eta, \mu) : \eta^2 + \mu^2 \le 1 \,\},
   \qquad \xi^2 = 1 - \eta^2 - \mu^2 \ \text{recovered from it}.

⚠ **The dimension does NOT drop**, and that one line is the source of
everything in the rest of this section. :math:`H` is **finite**, so
:math:`\dim H = 0` and :math:`\dim(S^2/H) = 2 - 0 = 2`; generic orbits
are two points, not curves. Contrast :math:`S^2/O(2)_a`, where
:math:`2 - 1 = 1`. With no dimensional reduction the invariant chart
buys nothing as a *data* representation — :math:`3 \to 2` floats with
the third recoverable — while for :math:`O(2)_a` it buys a genuine
:math:`3 \to 1` reduction (:ref:`manifold-chart-section-asymmetry`).

.. _manifold-stratum-is-a-locus:

The stratum is a LOCUS, and that retyped a field
-------------------------------------------------

:math:`\det P = 4 x_a^2` vanishes exactly on the mirror's own
fixed-point set — the great circle :math:`S^2 \cap \{x_a = 0\}`, which
in the realization's coordinates is the disk's **boundary**:

.. math::

   \operatorname{Fix}(\sigma_a) \cap S^2
   \;=\; \{\, \xi = 0 \,\}
   \;\longleftrightarrow\;
   \{\, \eta^2 + \mu^2 = 1 \,\} \;=\; \partial D^2 .

Every point of it is fixed by :math:`\sigma_a`, so its stabilizer is
all of :math:`H`, the orbit collapses from two points to one, and the
quotient is an **orbifold with boundary** — the same conclusion as
:ref:`manifold-singular-stratum` reaches for :math:`O(2)_a`, by the same
route. In transport terms the stratum is :math:`\xi = \mu_y = 0`: the
purely **meridional** directions.

⛔ **And this is what retyped a field.** ``singular_stratum`` was
``tuple[float, ...]`` and the first entry stored ``(-1.0, 1.0)``. A
**circle is not a tuple of floats**. The first catalogued entry's
*shape* had silently become the field's *type*: a stratum is a locus,
and two poles are merely a locus that happens to be finite. The field
is now a SymPy expression in the realization's coordinates whose
vanishing set is the stratum, with ``None`` for a free action — so
`[M]` :math:`O(2)_a` stores ``1 - u0**2`` (solving to
:math:`\{-1,+1\}`, unchanged in content) and :math:`\sigma_a` stores
``1 - u0**2 - u1**2``, whose solution set is a one-parameter family.
:attr:`is_free <orpheus.numerics.manifold.Quotient.is_free>` reads
``is None`` rather than ``== ()``.

⭐ **Why the stratum is STORED at all, when** ``det_gram`` **already
determines it.** This looks like a Pattern-2 twin and is not, and the
distinction is worth the paragraph because it is the criterion for
every future field: *a value that its owner cannot recompute from its
own state is derivation output, and storing it is right.* Recovering
the locus needs the **base's** defining ideal —
:math:`\det P = 4p_2` becomes :math:`4(1-\mu^2)` only after
substituting :math:`p_1^2 + p_2 = 1`, and :math:`\det P = 4p_3` becomes
:math:`4(1 - p_1^2 - p_2^2)` only after
:math:`p_1^2 + p_2^2 + p_3 = 1` — and a
:class:`~orpheus.numerics.manifold.Quotient` does not carry that ideal.
Step 5 of the procedure is exactly where that ideal enters, which is
why the stratum is a *fifth-step* output and not a property of
:math:`\det P` alone.

.. _manifold-two-coordinate-systems:

Two honest coordinate systems: a chart codomain AND a section's image
----------------------------------------------------------------------

The ruled shape (user, 2026-08-31) is **two slots**, and the two answer
different questions. State them that way when populating a new entry,
because the failure mode is putting the right object in the wrong slot:

.. list-table:: The two slots, and the question each answers
   :header-rows: 1
   :widths: 22 39 39

   * -
     - ``realization``
     - ``fundamental_domain``
   * - The question
     - *"What does the invariant chart of* :math:`M/H` *map ONTO?"*
     - *"Which points of* :math:`M` *are the chosen orbit
       representatives?"*
   * - Whose coordinates
     - the **invariants'** — the same language as ``generators``,
       ``gram`` and ``det_gram``
     - the **base's** ambient coordinates
   * - :math:`S^2/O(2)_a`
     - ``Interval(-1, 1)`` — the polar cosine :math:`\mu`
     - ``None``: no section is canonical
   * - :math:`S^2/\langle\sigma_a\rangle`
     - ``Ball(2)`` — the disk :math:`(\eta, \mu)`
     - ``FundamentalDomain(SPHERE, ((0.0, 1.0, 0.0),), 'y>=0')`` — the
       closed hemisphere, named ``S^2|y>=0``
   * - Who produces data in it
     - a rule *born* in the chart, e.g.
       ``gauss_legendre(8).measure.nodes``, shape :math:`(8,)`
     - :meth:`DiscreteMeasure.quotient
       <orpheus.numerics.measure.DiscreteMeasure.quotient>`, **always** —
       e.g. ``folded_product(4,8).measure.nodes``, shape :math:`(16,3)`
   * - Can it see the ERR-080 forgery?
     - ⛔ **no** — Mode-12 blind, measured below
     - ✅ yes — the forged rows are not on :math:`S^2`

⭐ **The producer had already chosen, and it chose the section.** `[M]`
:meth:`DiscreteMeasure.quotient
<orpheus.numerics.measure.DiscreteMeasure.quotient>` computes orbit
representatives and then pushes forward along
``lambda nodes: nodes[representative]`` — a **selection** of parent
nodes, applying no chart. So *every* measure the tree emits through
``.quotient(...)`` carries the base's ambient columns, by construction;
``folded_product``'s :math:`(16,3)` is not a stylistic choice but the
only thing that method can produce. Under a chart-only reading,
``Quotient.contains`` could validate **none** of them.

**What each half of the type does.**

- ``Quotient.contains`` accepts **either** language and dispatches on
  the ambient width. This is the one place in the type where the
  distinction is a genuine local split rather than a repeated tag test,
  and it is the reason the fork is resolved in one method instead of at
  every call site.
- ``_ambient`` still reports the **realization's** width. That is
  deliberate and is not a compromise: a
  :class:`~orpheus.numerics.manifold.Product` factor must have one
  canonical width, or a product's coordinate split would be ambiguous.
  ``contains`` is deliberately the wider of the two.
- ``Quotient.__post_init__`` gates the pair: the two views describe one
  object, so their ``dim`` must agree. `[M]` this is a real check and
  not a tautology — the fundamental domain *derives* its ``dim`` from
  the base less one per antipodal normal pair, while the realization
  *states* its own, so a domain of the wrong dimension is refused where
  it is written. `[M]` 2026-09-03, offering the :math:`\sigma_y` entry a
  **half-meridian** (normals :math:`\{\hat e_y, -\hat e_y\}`, an
  antipodal pair, ``dim 1``) against its 2-dimensional disk:

.. code-block:: text

   S^2/sigma_y: the fundamental domain 'S^2|half-meridian' has dim 1
   but the realization 'D^2' has dim 2 — the two must describe the same
   orbit space. Check the normals: an antipodal PAIR spells an
   equality and drops a dimension; a lone normal does not.

  ⛔ This example read *"a hemisphere offered against a 1-D
  realization"* until 2026-09-03, quoting the message for
  ``realization=[-1,1]``. `[M]` that input is now rejected one clause
  EARLIER, by :eq:`manifold-orbit-dimension-law` — a
  :math:`\sigma_a` entry's realization must be 2-dimensional whatever
  domain is offered — so the quoted message is no longer the one the
  tree emits for it. The clause is unmoved and still bites; only its
  witness had to change direction, from a realization too small to a
  domain too small (:ref:`manifold-dimension-drop`).

.. warning::

   ⛔ **The half-spaces must be CLOSED, and the witness is production
   data.** `[M]` the cylindrical march seeds each polar level at
   :math:`\xi = 0` **exactly** — the seed of level :math:`p` is
   :math:`(-\sqrt{1-\mu_p^2},\, 0,\, \mu_p)`, on :math:`S^2` to
   :math:`0.0` and on the stratum to :math:`0.0` — so a strict
   :math:`\langle p, n\rangle > 0` would refuse a direction the
   production sweep actually marches from. This is
   ``coding-elegance`` anti-pattern #18's half (ii) — *every legal
   value must be admitted*, which is a claim about the **producers**
   and is the half that gets skipped. Gated by
   ``test_the_half_space_is_CLOSED_because_production_marches_from_it``.

   ⚠ The shipped folded *rule* cannot witness this: `[M]` its 16 nodes
   have :math:`\mu_y \in [0.1945,\, 0.8688]`, strictly positive,
   because the even-:math:`n_\varphi` staggering makes the fold **free**
   on the nodes. The closure's edge data is the only witness available,
   which is why the gate is built on the seeds rather than on the
   quadrature.

.. _manifold-realization-refuted:

Five single-slot candidates, measured and refused
--------------------------------------------------

Before the two-slot ruling, five single-object candidates were put to
the shipped data. **All five fail**, and they fail in two disjoint
ways, which is itself the argument for two slots: the two that admit
the tree's nodes are blind to the chart, and the three that admit chart
points are blind to the nodes. The matrix below is **measured**, cell
by cell — and
"REFUSE (shape)" is a raised :exc:`ValueError` from the ambient-width
check, not a ``False`` return, which is a behavioural difference a
caller must handle (:ref:`manifold-gotcha-shape-vs-verdict`).

The five inputs: the shipped folded nodes :math:`(16,3)`; the **orbit
twins**, the same nodes with :math:`\mu_y \to -\mu_y`, i.e. the wrong
representative; the **ERR-080 forgery** :math:`(\mu, 0, 0)`,
:math:`(8,3)`, not unit-norm; and the chart images of the first and
third, :math:`(16,2)` and :math:`(8,2)`.

.. list-table:: Every candidate against every input, measured 2026-08-31
   :header-rows: 1
   :widths: 26 15 15 15 15 14

   * - Candidate
     - shipped
     - twins
     - forgery
     - shipped charted
     - forgery charted
   * - ``SPHERE``
     - ADMIT
     - ⛔ **ADMIT**
     - refuse
     - REFUSE (shape)
     - REFUSE (shape)
   * - ``RealSpace(2)``
     - REFUSE (shape)
     - REFUSE (shape)
     - REFUSE (shape)
     - ADMIT
     - ⛔ **ADMIT**
   * - ``COSINE_INTERVAL × COSINE_INTERVAL``
     - REFUSE (shape)
     - REFUSE (shape)
     - REFUSE (shape)
     - ADMIT
     - ⛔ **ADMIT**
   * - ``Ball(2)`` alone
     - REFUSE (shape)
     - REFUSE (shape)
     - REFUSE (shape)
     - ADMIT
     - ⛔ **ADMIT**
   * - the hemisphere alone
     - ADMIT
     - refuse
     - refuse
     - REFUSE (shape)
     - REFUSE (shape)
   * - ⭐ **SHIPPED: both slots**
     - ADMIT
     - refuse
     - refuse
     - ADMIT
     - ⚠ ADMIT

Reading the rows:

**The sphere itself.** ``realization = SPHERE`` is the convenient
placeholder, and the one to refuse hardest. It does *not* regress
ERR-080 (the forgery is still
refused, since :math:`\lVert(\mu,0,0)\rVert = |\mu| \ne 1`). What it
loses is **the fold itself**: the orbit twins are admitted, and more
sharply, `[M]` under this choice ``Quotient.contains`` becomes the
*same function* as ``SPHERE.contains`` — **no input exists that the
quotient refuses and its base admits**. A predicate that cannot
distinguish :math:`M/H` from :math:`M` is ``vv-principles`` #17's *arm
with no witness*, decidable at design time with no mutation needed.
⛔ And it is topologically false, not merely weak: :math:`D^2 \ncong
S^2` — the disk is contractible with boundary :math:`S^1` and
:math:`\chi = 1`, the sphere has no boundary and :math:`\chi = 2`. That
``dim`` happens to agree (both 2) carries no information here either,
precisely because :math:`H` is finite.

**The two shipped 2-D members**, ``RealSpace(2)`` and the square. Both
buy nothing the disk does not, and both are strictly weaker:
``RealSpace(2)`` drops the disk inequality entirely, and
``COSINE_INTERVAL × COSINE_INTERVAL`` is the bounding **square**, whose
discriminating witness is measured — `[M]` :math:`(0.9, 0.9)` is in the
square and not in the disk, and corresponds to **no direction at all**,
since :math:`\eta^2 + \mu^2 = 1.62 > 1` forces :math:`\xi^2 = -0.62`.
Reusing a shipped member because it ships is how a type acquires a
predicate that admits impossible points.

**The disk alone.** ``Ball(2)`` is what the *documented* meaning of
``realization`` requires, and the sharpest refusal of the five, because
it fails on the
very defect the type was minted for. ⛔ `[M]` **the chart is Mode-12
blind to ERR-080.** The chart :math:`(x,y,z) \mapsto (x,z)` discards
:math:`\mu_y`, and :math:`\mu_y` is precisely what the forgery
falsifies, so the forged row :math:`(\mu, 0)` is a **perfectly legal**
point of the disk — it is the orbit of the real direction pair
:math:`(\mu, \pm\sqrt{1-\mu^2}, 0)`. Measured:
:math:`\max |(\mu,0)|^2` over the eight forged rows is
:math:`0.9221566084920586 < 1`, so *every* forged row lands inside the
closed disk. The mechanism is exact, not statistical
(``vv-principles`` Mode 12 — the measured functional's stabiliser
contains the error class): no tolerance, refinement or fixture choice
can expose it.

**The hemisphere alone** — the only single object that admits the
shipped nodes and refuses both wrong inputs. Its cost is that it
**redefines** the field rather than adding to it: ``realization`` would
stop meaning *chart codomain*, the type would then answer in the base's
language for :math:`\sigma_a` and in the chart's for :math:`O(2)_a` —
the exact vocabulary drift the mint exists to end
(:ref:`manifold-string-drift`) — and the derivation fields
(``generators``, ``gram``, ``det_gram``) would be speaking a coordinate
system the ``realization`` beside them no longer names.

⚠ **And read the shipped row honestly: it does not dominate every
cell.** The two-slot design still admits the *charted* forgery, and
that is correct rather than a residual defect — in chart coordinates
:math:`(\mu, 0)` genuinely **is** a point of the orbit space, and no
predicate over the chart can know it was built by zero-padding. What
the second slot buys is that the data the tree actually produces
arrives in *section* coordinates, where the predicate that can see the
forgery is the one that runs.

.. _manifold-err-080-is-a-section:

ERR-080's level-1 half is a botched section of :math:`S^2/O(2)_x`
------------------------------------------------------------------

The chart-versus-section question is not new with :math:`\sigma_y`. It
arises for :math:`O(2)_a` too — the moment any consumer of a
1-dimensional rule needs a 3-D direction — and the tree has been
answering it, silently and wrongly, for as long as :ref:`ERR-080
<manifold-err-080>` has existed.

The realization :math:`[-1,1]` is the **chart's** codomain,
unambiguously: a section of :math:`S^2 \to S^2/O(2)_a` is a half-meridian
*inside* :math:`S^2 \subset \mathbb{R}^3`, ambient 3, and
``Interval(-1, 1)`` is ambient 1. So when
:meth:`Quadrature.angular_frame
<orpheus.numerics.quadrature.directional.Quadrature.angular_frame>`
needs three columns it is not asking for the chart at all — it is
asking for a **section**, which the tree never had. It fabricated one
by zero-padding:

.. list-table:: The fabricated section against an honest one
   :header-rows: 1
   :widths: 40 60

   * - Construction
     - `[M]` 2026-08-31, on ``gauss_legendre(8)``
   * - what the tree builds — ``column_stack`` of the three
       axis-cosine arrays, two of them a zero *fallback*
     - rows :math:`(\mu, 0, 0)`; norms
       :math:`0.1834 \ldots 0.9603`; ``Sphere().contains`` → ``False``
   * - an honest :math:`\varphi = 0` half-meridian. ⛔ this row spelled
       it :math:`\mu \mapsto (\sqrt{1-\mu^2},\, 0,\, \mu)` — a
       :math:`z`-pole section — until 2026-09-01. With the axis a
       parameter it is written in the axis's own language,
       :math:`\mu \mapsto \mu\,\hat e_a + \sqrt{1-\mu^2}\,\hat e_b`, and
       the slab's :math:`a` is :math:`x`:
       :math:`\mu \mapsto (\mu,\, \sqrt{1-\mu^2},\, 0)`
     - `[M]` on :math:`S^2` to :math:`0.0` (max
       :math:`\bigl|\lVert v\rVert - 1\bigr|`);
       ``Sphere().contains`` → ``True``. ⭐ Note what the comparison now
       shows: the fabrication is this map with the :math:`\hat e_b` term
       **dropped**, which is precisely why its rows fall short of the
       unit sphere
       (:ref:`manifold-the-axis-convention-for-a-section`)

⟹ **ERR-080's first link is not "a wrong tag". It is a section
fabricated where none was declared** — the realization is a chart, a
consumer needed a section, and zero-padding is what a codebase does
when the object it needs has no slot. With ``fundamental_domain`` in
the type, that padding has nowhere to live: an entry either declares a
section or declares that it has none, and :math:`S^2/O(2)_a` honestly
declares ``None``.

.. warning::

   ⚠ **This names the level-1 half only. Do not read it as the
   repair.** Making the section land on :math:`S^2` makes the nodes
   *points of the manifold*; it does **not** fix the level-2 half. On
   any :math:`\varphi = 0` section every :math:`Y_\ell^{m \ne 0}` is
   evaluated at a *chosen* azimuth that carries no information, and the
   corpus's recorded repair for that is unchanged: the **trivial
   isotypic sub-basis** :math:`\{Y_\ell^0\} \cong \{P_\ell\}`
   (:ref:`manifold-gelfand`, and the falsifiable form with ERR-080 in
   the :doc:`error catalogue </theory/verification/error_catalog>`).
   Both halves are needed; this section establishes only the first.
   ⛔ This sentence ended, verbatim, "and **ERR-080 remains open**
   (:ref:`manifold-seams`)" until 2026-09-02: true when written, and
   repealed the same day by #429's fused commit, which landed the
   second half as :class:`~orpheus.numerics.basis.legendre_basis.LegendreBasis`
   and CLOSED ERR-080. What is still owed is tracker 2.0b, the
   membership check at measure construction — a *different* seam.

   ⚠ Declaring a section for :math:`O(2)_a` would also be a **choice**,
   not a derivation — the :math:`\varphi = 0` half-meridian is one of a
   continuum — so it belongs to the step that makes a slab declare its
   quotient, not to the derivation. The shipped entry therefore carries
   ``fundamental_domain=None`` on purpose.

.. _manifold-chart-section-asymmetry:

Why the two entries diverge: a structural asymmetry, not a style choice
-------------------------------------------------------------------------

This is the transferable half of the whole section, and it is what a
future entry's author needs before populating either slot. The two
shipped entries make **opposite** choices, and both are locally
correct; what could not serve both was the *type*.

.. list-table:: The asymmetry, term by term
   :header-rows: 1
   :widths: 26 37 37

   * -
     - :math:`S^2/O(2)_a`
     - :math:`S^2/\langle\sigma_a\rangle`
   * - the group
     - compact **connected**, :math:`\dim = 1`
     - **finite**, :math:`\dim = 0`, and a *reflection*
   * - :math:`\dim(S^2/H)`
     - :math:`2 - 1 = 1` — **drops**
     - :math:`2 - 0 = 2` — **does not drop**
   * - the invariant chart's codomain
     - :math:`[-1,1] \subset \mathbb{R}^1`
     - the closed disk :math:`D^2 \subset \mathbb{R}^2`
   * - the chart *as data*
     - :math:`3 \to 1` floats: a genuine **reduction**
     - :math:`3 \to 2` floats, the third recoverable: **no reduction**
   * - a canonical section?
     - ⛔ **no** — every half-meridian is one; :math:`\varphi = 0` is an
       arbitrary pick
     - ✅ **yes** — the mirror determines the closed half-space, and
       being a *reflection* makes it **strict**
   * - what the tree ships as ``measure.nodes``
     - the **chart**, :math:`(8,)`
     - the **section**, :math:`(16,3)`
   * - `[M]` ``measure.support`` (2026-09-02)
     - ``'S^2/O2_x'`` — the *quotient's* name. ⛔ This cell read
       ``'[-1,1]'`` — the *realization's* name — until 2026-09-02; it
       was true when written and tracker 2.4 repealed it on
       2026-09-01, when the slab's rule began *declaring* its orbit
       space (:ref:`manifold-orbit-space-declaration`). The row's
       point survives intact: what the rule ships as ``nodes`` is
       still the **chart**, one column, and only the *support* now
       says which orbit space those chart coordinates are for.
     - ``'S^2/sigma_y'`` — the *quotient's* name

⟹ **For a positive-dimensional group the chart is strictly cheaper and
no section is canonical, so the chart wins and the tree ships it. For a
finite reflection the chart is no cheaper and the section IS canonical,
so the section wins and the tree ships that.** Neither branch was
wrong; the single-slot type was.

.. warning::

   ⛔ **"Canonical because it is a reflection" does not generalise to a
   rotation — leave** ``fundamental_domain=None`` **for** :math:`C_n`.
   The closed half-space is a *strict* fundamental domain (it meets
   every orbit exactly once) only because :math:`\sigma_a`'s
   fixed-point set lies **in** it: a free orbit
   :math:`\{(x,y,z), (x,-y,z)\}` with :math:`y > 0` meets
   :math:`\{y \ge 0\}` once, and a stratum orbit :math:`\{(x,0,z)\}` is
   a single point that also lies in it. For a rotation :math:`C_n` the
   closed sector's two meridian edges are identified **with each
   other** by the group, so the closed sector maps 2-to-1 on its
   boundary and is *not* homeomorphic to the orbit space. A
   fundamental-domain slot filled for a :math:`C_n` entry would be
   stating something false, and the type cannot catch it — the ``dim``
   gate would pass.

.. note::

   **The hemisphere IS a legitimate realization set-theoretically, and
   is NOT a diffeomorphic one — both halves matter.** `[M]` the chart
   :math:`c : H^+ \to D^2`, :math:`(x,y,z) \mapsto (x,z)`, with inverse
   :math:`(p_1,p_2) \mapsto (p_1, \sqrt{1-p_1^2-p_2^2}, p_2)`, is a
   continuous bijection from a **compact** set onto a **Hausdorff**
   one, hence a homeomorphism (no separate inverse-continuity argument
   needed). It is *not* a diffeomorphism: :math:`\partial y/\partial
   p_i = -p_i/\sqrt{1-p_1^2-p_2^2}` blows up on the stratum, and from
   the forward side :math:`\mathrm{d}c` annihilates :math:`e_y` there —
   rank 1 on the boundary circle, 2 in the interior. That is a
   **Whitney fold**, and it shows up in the measure as an integrable
   singularity in :math:`1/\lvert y \rvert` — that is, in
   :math:`1/\lvert\xi\rvert`, the coordinate the fold quotients:
   `[M]` :math:`\int_{D^2} \mathrm{d}p_1\, \mathrm{d}p_2 /
   \lvert y\rvert = 2\pi`, the area of a hemisphere — finite.

   ⚠ **The fold does not bite** ``contains``. Membership is a level-1,
   set-theoretic question and the homeomorphism settles it; the fold
   bites at **levels 2 and 3** — what a basis function eats, and what a
   derivative operator differentiates. So it must not be cited as an
   argument against a disk realization *for membership purposes*. The
   arguments against the disk alone are the ones in
   :ref:`manifold-realization-refuted`, and they are entirely
   different.

.. _manifold-orbifold-discretised:

The orbifold is already realized in the shipped cylindrical sweep
------------------------------------------------------------------

:ref:`manifold-singular-stratum` records that two shipped objects live
on the :math:`O(2)_a` stratum from opposite directions. The
:math:`\sigma_y` entry has an exact cylindrical analogue, and it is
measured:

.. list-table:: The fold's stratum against the shipped cylindrical data
   :header-rows: 1
   :widths: 46 54

   * - Object
     - `[M]` 2026-08-31, ``folded_product(4, 8)`` on a cylinder
   * - the 16 quadrature nodes
     - :math:`1 - \eta^2 - \mu^2 \in [0.0378,\, 0.7549]` — **strictly
       interior**; the fold is free on them
   * - the four march seeds — the starting angular edge per level,
       ``AngularRedistribution.mu_start_per_level``
     - the seed direction is
       :math:`(-\sqrt{1-\mu_p^2},\, 0,\, \mu_p)`, on :math:`S^2` to
       :math:`0.0` and with
       :math:`1 - \eta^2 - \mu^2 = 0.0` **exactly**, on all four —
       i.e. **on the stratum**
   * - the azimuthal cell edges of each level
     - the nodes sit at :math:`\omega/\pi \in \{0.125, 0.375, 0.625,
       0.875\}`, the staggered midpoints of four cells partitioning
       :math:`(0,\pi)`, so the edges are
       :math:`\omega/\pi \in \{0, \tfrac14, \tfrac12, \tfrac34, 1\}`
       and the two **outer** edges are :math:`\omega = 0, \pi`, where
       :math:`\xi = \sin\theta \sin\omega = 0`
   * - the :math:`\alpha`-dome per level
     - five edge values per level, closing at both ends. Levels 0 and 3
       read :math:`[0,\, 0.2566,\, 0.3629,\, 0.2566,\, 0]`; levels 1
       and 2 read
       :math:`[0,\, 0.8900,\, 1.2587,\, 0.8900,\, 1.1\times10^{-16}]`

⟹ **That is what an orbifold looks like when you discretise it:** the
*interior* of the fundamental domain carries the nodes, and its
*boundary* — the stratum — carries the closure's degenerate data, the
:math:`\alpha`-dome's zeros and the march seed.

⚠ Two naming traps in that table, both `[M]` and both worth knowing
before quoting it. ``mu_start_per_level`` holds a **radial** cosine
:math:`\eta = -\sin\theta_p`, not a polar :math:`\mu` — the name is the
half-angle thread's, not this page's; and the field's own docstring
spells the level's polar cosine :math:`\xi_p`, while :math:`\xi`
everywhere on this page is :math:`\mu_y`, the *azimuthal* cosine that
the fold quotients. The values are unambiguous — `[M]`
``mu_start_per_level`` equals :math:`-\sqrt{1-\mu_p^2}` on the level
cosines exactly — but the symbols are not.

⚠ **This is an identity of LOCI, established by arithmetic — the
mechanism is unproved.** Whether the cylindrical redistribution term
*is* the :math:`\sigma_y` quotient's connection is the exact
cylindrical twin of the open spherical question recorded in the third
row of :ref:`manifold-one-polynomial`, and it is not closed here. It
has the same two admissible outcomes: a derivation, or an explicit
ruling that the coincidence is geometric bookkeeping. Cite this
subsection for the measured coincidence, never for the mechanism.


.. _manifold-engine-seed:

The catalogue is the engine's SEED, not its rival
=================================================

An orbit space *can* be computed from scratch: steps 1–4 above are
mechanical, and a Gröbner-basis engine would run them. The project has
ruled that it will **not build that engine yet** — and the ruling is
worth quoting exactly, because the obvious paraphrase ("we refused the
engine") is the wrong one:

   *"We're not outright ruling out building the engine. We're ruling
   that we will not prematurely build the engine. The embryo should be
   such that if the day ever arises that we decide building the engine
   is the right step, it should be a development of the embryo, instead
   of having to do the entire engine from scratch for a code base that
   was not ready to receive it."* — user, 2026-08-31 (decision D0.1)

The groups that occur in transport number about a dozen. A Gröbner
engine for them is abstraction without a consumer, and its failure mode
is debugging elimination orderings instead of transport. So each entry
is derived once by the procedure and recorded — **deferred, not
refused.**

.. _manifold-engine-data-model:

The binding requirement is on the DATA MODEL, not the signature
----------------------------------------------------------------

⛔ The first version of this ruling said *"the catalogue and the engine
have the same interface — a second backend behind an unchanged
signature"*, and it was **rejected as too weak**, in the ruling's own
words as *"the twin-path risk wearing a compliment"*. A shared
signature guarantees only that the *call site* survives. The engine
would still arrive with its own representation of polynomials, ideals
and PSD conditions, plus a translation layer to whatever the catalogue
happened to store — a from-scratch build with a seam, which is exactly
what the ruling forbids.

⟹ **A catalogue entry must BE the derivation procedure's output, not a
human summary of its answer.** The procedure emits, per entry: the
invariant generators; the syzygy ideal; the matrix :math:`P` and
:math:`\det P`; the quotient map :math:`\pi : M \to M/H` and the
codomain it realizes onto; its **lift** :math:`\lambda` back into the
base's ambient space, and the manifold that lands on; a section, when
one is canonical; the pushforward of the base's measure along
:math:`\pi`; the stratum where :math:`\det P` vanishes; and its own
provenance. **Those are the entry's fields.** An engine then ships by *computing* them instead of
reading them — a development, with no new vocabulary and no seam.

⭐ **The list above has grown TWICE, and both growths are the ruling
working rather than the ruling slipping.** The section was not on the
procedure's output list until the second entry produced one
(:ref:`manifold-two-coordinate-systems`); the **lift** was not on it
until 2026-09-03, when #434 R4 found it living as a three-arm branch on
the group's tag inside ``Quotient.lift`` — a *derivation output* being
re-derived at read time from the key the catalogue had already
dispatched on. In both cases the output was added to the *procedure*
and a slot was added to match, which is exactly the direction the check
below permits. What it forbids is the reverse — a field the procedure
does not emit, or an output the procedure emits that the entry has to
summarise in prose, or **re-compute from its inputs**.

⛔ **The word** ``chart`` **left this list on 2026-09-02**, and the
correction is worth more than the word. The list read *"…the matrix*
:math:`P` *and* :math:`\det P`\ *; the chart; a section…"* — but the
map the procedure emits is the **quotient map**, and a quotient map is
by construction *not* injective, while a chart is. Tracker 2.3 ruled
the naming (:ref:`manifold-arrow-type`) and tracker 3.1 shipped the
map (:ref:`manifold-quotient-map`); what ``realization`` has always
been is the codomain the invariants land in, which the strict chart —
the *inverse* of the Archimedes parametrisation — maps onto as well.

The ruling comes with its own falsifiable check, and it is the question
to ask of any future edit here:

   *Given a catalogue entry, could an engine populate its fields
   without introducing a single new type?*

If the answer is no, the embryo has drifted from being a seed and the
ruling has been violated — however clean the interface looks.

.. list-table:: The procedure's outputs against the shipped slots
   :header-rows: 1
   :widths: 34 22 44

   * - Procedure output
     - Slot on :class:`~orpheus.numerics.manifold.Quotient`
     - Note
   * - invariant generators :math:`p_1 \ldots p_k`
     - ``generators``
     - SymPy expressions in the ambient coordinates
   * - syzygy ideal :math:`I`
     - ``syzygy``
     - ``()`` when the invariants are independent
   * - :math:`P_{ij}`
     - ``gram``
     - re-expressed in the invariants
   * - :math:`\det P`
     - ``det_gram``
     - its zero locus is the orbit-space boundary
   * - the singular stratum
     - ``singular_stratum``
     - a **locus** in the realization's coordinates — derivation
       output that a ``Quotient`` cannot recompute, because recovering
       it needs the base's own ideal
       (:ref:`manifold-stratum-is-a-locus`)
   * - provenance
     - ``derived_by``
     - ``"hand"`` / ``"engine"``
   * - a section of :math:`M \to M/H`, when canonical
     - ``fundamental_domain``
     - its **image**, in the base's coordinates; ``None`` is an
       answer, not a gap
       (:ref:`manifold-two-coordinate-systems`)
   * - the quotient map :math:`\pi : M \to M/H`
     - ``orbit_coordinates``, plus the derived ``quotient_map``
     - the map's **action** on the base's ambient coordinates —
       ``field(compare=False, repr=False)``, because a function has no
       value equality — with the typed arrow derived on top of it,
       because a frozen dataclass cannot store an arrow whose codomain
       is itself (:ref:`manifold-quotient-map`). ⛔ This row read
       *"the chart* :math:`M/H \to N` *— not a slot; only its codomain
       ships, as* ``realization``\ *"* until 2026-09-02. Two things
       were wrong with it and one right: the map is a slot now
       (tracker 3.1), and it was never a **chart** — a chart is
       injective and :math:`\Omega \mapsto \Omega\cdot\hat e_a` is
       not, which is the naming ruling tracker 2.3 made. The
       ``realization`` is still the codomain of the *chart*, and the
       chart is still not a value anywhere.
   * - the lift :math:`\lambda : M/H \to \mathbb{R}^n`, and where it
       lands
     - ``lift_coordinates``, ``lift_codomain``, plus the derived
       ``lift``
     - the lift's **action** on the chart's coordinates, and the
       manifold that action lands on — the orbit BARYCENTRE, which is
       the Reynolds projector :math:`P_H` read from the chart's side
       (:eq:`manifold-reynolds-projector`, :ref:`manifold-lift`). The
       map is ``field(compare=False, repr=False)`` for the reasons
       ``orbit_coordinates`` is; the **codomain is COMPARED**, because a
       :class:`Manifold` has value equality where a function does not
       and two entries that agree on :math:`(M, H)` and disagree on
       where their lift lands are two objects
       (:ref:`manifold-dimension-drop`). ⭐ **New at #434 R4,
       2026-09-03.** Until then ``lift`` was a property branching on
       ``by``'s tag, whose fall-through read *"add the entry's section
       (or its equivariant barycentre) to ``Quotient.lift``"* — the
       ruling's own forbidden direction, an output the entry
       re-derived instead of storing, and one a seventh entry could
       forget. The fields are REQUIRED, so it cannot be forgotten now.
   * - the pushforward measure :math:`\pi_*\,d\Omega`
     - ``reference``
     - the measure a degree of exactness on this orbit space is
       **against**, as a
       :class:`~orpheus.numerics.exactness.ReferenceMeasure`
       (:eq:`manifold-quotient-pushforward`). ``LEGENDRE`` on the
       three axial entries, by Archimedes' hat-box; ``None`` on the
       three mirror entries and on :math:`M/\{e\}`, and both
       ``None``\ s are answers rather than gaps
       (:ref:`manifold-pushforward-reference`). ⛔ This row read
       *"not a slot … it cannot be added by importing —* `[M]` *a
       module-scope* ``manifold → exactness`` *edge closes a two-hop
       cycle and kills 5 of 5 fresh import orders"* until 2026-09-02.
       The cycle is real and unchanged; what the row got wrong is
       that it treated the cycle as blocking the *slot* rather than
       one *mechanism* for filling it. The shipped answer splits the
       problem: the **type** rides a
       :data:`typing.TYPE_CHECKING` import and the **value** a
       function-scope one, `[M]` alive on 7 of 7 import orders where
       every module-scope placement dies on 7 of 7
       (:ref:`manifold-value-at-function-scope`).

`[M]` **10 of 10, re-measured 2026-09-03** — by
``dataclasses.fields``, ``Quotient`` declares **fourteen** fields:
``base``, ``by``, ``realization``, ``orbit_coordinates``,
``lift_coordinates``, ``lift_codomain``, ``fundamental_domain``,
``generators``, ``syzygy``, ``gram``, ``det_gram``, ``derived_by``,
``reference``, ``singular_stratum`` — of which the first two are the
entry's *inputs*, and ``quotient_map`` and ``lift`` are derived
properties on top of the fourth and the fifth. So every one of the
procedure's ten outputs is now a slot, and the seed is complete.
Stating the fraction is the point: a ruling whose compliance is claimed
but not counted is not checkable. (`[M]` it read **6 of 8** until the
two-slot ruling on 2026-08-31, **7 of 9** until tracker 3.1, and
**9 of 9 over twelve fields** until #434 R4 on 2026-09-03; the
denominator has moved twice and the numerator three times, which is why
the two numbers are given together and not as a percentage.)

⚠ **A complete seed is not a shipped engine, and the distinction is the
whole point of the ruling.** What 9 of 9 says is that an engine can now
populate every field without introducing a type — the falsifiable check
above passes. It says nothing about the entries: `[M]` all of them still
read ``derived_by="hand"`` — all **seven** quotients of :math:`S^2` the
catalogue can produce, its six keys plus the derived identity — six keys
ship out of the expected dozen, and the engine itself is deferred
(:ref:`manifold-seams`).

⭐ **Why the provenance field exists at all.** ``derived_by`` is read by
nothing today, and a reviewer could reasonably call it speculative. It
is not: a mixed hand/engine state must be *expressible*, or an
incremental engine rollout would have to be all-or-nothing — and an
incremental rollout is exactly what the ruling anticipates. The field
is the difference between a migration that can be staged and one that
cannot.

.. _manifold-refusal-names-the-work:

The refusal names the missing WORK, not the gap
------------------------------------------------

The engine's absence must be a work item a fresh session can pick up,
not a wall. `[M]` ``SPHERE.quotient(SubgroupOfO3.OctahedralOh)``
raises, verbatim:

.. code-block:: text

   no catalogue entry for S^2/Oh: derive it once (minimal invariants
   p_1..p_k of R[x]^H; syzygy ideal I by elimination; Procesi-Schwarz
   P_ij = <grad p_i, grad p_j> with P >= 0; intersect with the ideal of
   S^2) and register it in orpheus/numerics/manifold.py's
   _ORBIT_CATALOGUE, or implement the derivation engine. Catalogued
   today (manifold CLASS / group): ['Sphere/O2_x', 'Sphere/O2_y',
   'Sphere/O2_z', 'Sphere/sigma_x', 'Sphere/sigma_y',
   'Sphere/sigma_z'].

Four things are in that message and all four are load-bearing: which
pair was asked for, the **procedure** to answer it, **where** to put
the answer, and what is catalogued already. The last is spelled
*manifold CLASS* on purpose — the request is named by manifold
*instance* name (``S^2``) while the catalogue is keyed by class, and a
message that silently switched vocabularies would send its reader
looking for a key that does not exist.

The catalogue is keyed on the **pair** ``(manifold class, group
name)``, because a quotient is binary dispatch: it is a property of
neither operand alone.

.. _manifold-tests-are-the-spec:

The regression tests are the engine's acceptance suite, written first
---------------------------------------------------------------------

Every catalogue entry ships a **symbolic** regression test that
reproduces its own derivation. Because the fields *are* the
procedure's outputs, those tests are not merely regression pins: they
are the engine's **specification**, and the engine ships on the day it
reproduces them by computation instead of by lookup. A specification
written before the implementation cannot be shaped to flatter it
(``vv-principles`` #17).

Concretely, for the two shipped entries, the assertions are on the
**symbolic value** rather than on a string or a float:

.. code-block:: python

   # tests/numerics/test_manifold.py::TestQuotient
   assert s2_mod_so2.syzygy == ()
   assert sp.simplify(s2_mod_so2.det_gram - 4 * p2) == 0
   assert sp.simplify(
       s2_mod_so2.gram - sp.Matrix([[1, 0], [0, 4 * p2]])
   ) == sp.zeros(2, 2)

   # ...and the stratum is DERIVED, not compared to a literal:
   det_on_sphere = sp.simplify(s2_mod_so2.det_gram.subs(p2, 1 - mu**2))
   roots = sorted(sp.solve(sp.Eq(det_on_sphere, 0), mu))
   assert [float(r) for r in roots] == [-1.0, 1.0]
   assert sp.simplify(s2_mod_so2.singular_stratum - (1 - u0**2)) == 0

   # tests/numerics/test_manifold.py
   #   ::TestTheSigmaYFoldIsExpressibleAndDiscriminating
   assert sp.simplify(fold.det_gram - 4 * u2) == 0
   assert sp.simplify(fold.gram - sp.diag(1, 1, 4 * u2)) == sp.zeros(3, 3)
   assert fold.realization == Ball(2)
   assert fold.dim == 2                     # H is FINITE: no drop

   # ...and the stratum is shown to be a CURVE, not a point set:
   assert sp.simplify(fold.singular_stratum - (1 - u0**2 - u1**2)) == 0
   sols = sp.solve(sp.Eq(fold.singular_stratum, 0), u1)
   assert len(sols) == 2 and all(u0 in s.free_symbols for s in sols)

⭐ **The last two lines are the shape of the assertion that a retyped
field needed.** Solving the locus and checking that the solutions
*retain a free symbol* is what distinguishes a curve from a point set,
and it is a claim no comparison against a literal could make. The
:math:`SO(2)` gate's own stratum assertion was likewise rewritten to
solve rather than compare — `[M]` it read
``s2_mod_so2.singular_stratum == (-1.0, 1.0)`` until 2026-08-31 — and
it survived the retyping **without weakening**, because it had already
been written to solve :math:`\det P = 0` rather than to trust the
stored value.

.. _manifold-twin-lookup:

⭐ The tree already performs this lookup, one level up
------------------------------------------------------

The mint was not introducing a new idea. `[M]`
:attr:`AngularSymmetry.support
<orpheus.numerics.quadrature.registry.AngularSymmetry.support>` — which
predates it — already computed :math:`S^2/K` from the *spent* group by
catalogue lookup, in the string vocabulary, and already raised
:exc:`NotImplementedError` on an uncatalogued quotient with the same
shape of message.

✅ **The twin is COLLAPSED as of tracker 2.4 (2026-09-01), in the
direction reading (iv) below predicted.** ``AngularSymmetry.support`` no
longer holds a table: it *calls* ``SPHERE.quotient(spent)``, so the two
lookups are one call and cannot disagree. `[M]` for a slab,
``GEOMETRY_ANGULAR_SYMMETRY['slab'].support is
SPHERE.quotient(SubgroupOfO3.O2('x'))`` — object **identity**, not
merely equality, because the catalogue memoises
(:ref:`manifold-quotient-is-memoised`). The section is kept as the
argument that got there.

.. list-table:: Two lookups of :math:`S^2/H`, re-measured 2026-09-01 after the tracker-2.4 collapse
   :header-rows: 1
   :widths: 16 30 30 24

   * - :math:`H`
     - ``AngularSymmetry(...).support.name``
     - ``SPHERE.quotient(H).name``
     - Reading
   * - ``O2('x')``
     - ``'S^2/O2_x'``
     - ``'S^2/O2_x'``
     - **Identical object.** ⛔ This row read ``'[-1,1]'`` on both sides
       until 2026-09-01 — the registry returned the CHART, which is the
       axis-blind spelling a slab rule could share with a spatial
       interval — and was keyed ``SO2('x')`` until 2026-09-02, when #432
       named the entry by its stabiliser.
   * - ``O2('z')``
     - ``'S^2/O2_z'``
     - ``'S^2/O2_z'``
     - Identical object. A row that could not be *spelled* before the
       axis was a parameter.
   * - ``Trivial``
     - ``'S^2'``
     - ``'S^2/Trivial'``
     - ⚠ **The one surviving divergence, and it is deliberate.** The
       registry short-circuits to the base, because a geometry that
       spends nothing discretises the sphere and ``'S^2'`` is the name
       every 2-D/3-D rule declares. The catalogue reports the
       *derivation's own output*, whose ``realization`` **is** that
       sphere. Same point set, two registers; a committed row pins them.
   * - ``sigma_y``
     - ``'S^2/sigma_y'``
     - ``'S^2/sigma_y'``
     - Now answered, by delegation — but see (iii): it is a row the
       registry has no *business* being asked, not one it was missing.
   * - ``Oh``
     - :exc:`NotImplementedError`
     - :exc:`NotImplementedError`
     - The refusal is now literally the same refusal
       (:ref:`manifold-refusal-names-the-work`).

Four readings, all useful — and all four survive the collapse, because
what collapsed is the *implementation*, not the distinction between the
two questions.

**(i)** On the overlapping rows the typed catalogue reproduced the
registry's answer exactly, which was the cheapest available evidence
that the type is a *re-typing* of an existing fact and not a rival one.
The pin is
``tests/numerics/test_manifold.py::TestQuotient::test_the_derived_orbit_space_agrees_with_the_hand_written_table``,
and reading its docstring is worth more than reading this paragraph,
because it records what the collapse did to its own claim class — twice,
in opposite directions.

* ⛔ **The axial row was DEMOTED by the collapse**, and correctly. Once
  the registry derives its domain *through* ``SPHERE.quotient``, asking
  whether the two agree is asking one call whether it equals itself —
  ``coding-standards``' single-sourcing clause exactly. The fix stays;
  the gate's *description* is what had to move. What survives on that
  row is a different and still-real claim: that the registry hands out
  the **orbit space**, carrying its spent group, rather than the chart.
* ⭐ **The same gate was PROMOTED by tracker 2.0c**, with no line of its
  body changing. It used to compare ``mine.name == theirs`` because
  ``theirs`` was a *string* — the strongest claim the string vocabulary
  admitted. Both sides are ``Manifold`` values now, so the assertion is
  **object equality**: not that two producers spell the orbit space the
  same way, but that they produce the same point set. A name gate is
  satisfied by any self-consistent lie; this one is not.
* ✅ **What is load-bearing on every row after the collapse** is the
  hand-written ``expected`` column, which is authored independently of
  both producers: it is a genuine external pin on the Procesi–Schwarz
  derivation, and the one input that could still redden the row is a
  wrong derivation.

The *do not re-baseline* note stands: if it reddens, one of the two is
wrong about a quotient, and which one is a mathematical question, not a
test-maintenance one.

**(ii)** ⛔ **The** ``Trivial`` **row read** :exc:`NotImplementedError`
**in the first version of this page, and it was already false when the
page landed.** Comparing the two lookups is what exposed the gap —
:math:`S^2/\{e\} = S^2` is legal and trivially derivable — and the same
commit that published the table (``fba4205a``) closed it, by
**deriving** the answer rather than tabulating it: the trivial group's
invariant ring is the whole polynomial ring, so :math:`p_i = x_i`,
:math:`P = I`, :math:`\det P = 1`, which vanishes nowhere, hence no
stratum, hence a free action — right vacuously, the only element being
the identity. That doubles as a **positive control on the machinery**:
the procedure reproduces a known answer. The row is corrected here as
history rather than deleted, because a gap reported into the corpus has
the shortest shelf life of anything on a page — the report is what
triggers the repair.

**(iii)** ⭐ **The** ``sigma_y`` **row is not a gap in the registry, and
reading it as one would send its repair in the wrong direction.** The
two lookups quotient by **different parts of the symmetry**.
:attr:`AngularSymmetry.support
<orpheus.numerics.quadrature.registry.AngularSymmetry.support>` is
:math:`S^2/K` — the stabiliser a dimensional reduction SPENDS — while a
mirror is a member of the finite parts the reduction did *not* spend.
`[M]` 2026-09-03 the shipped geometry table says so directly:
``GEOMETRY_ANGULAR_SYMMETRY["cylinder"]`` is ``spent=Trivial,
unspent=Dnh(1), owed=Dnh(2)`` — nothing continuous is spent on a
cylinder, so its declared angular domain is the whole sphere.
:meth:`Manifold.quotient
<orpheus.numerics.manifold.Manifold.quotient>` has no such restriction:
it quotients by any subgroup, which is why it can answer
:math:`S^2/\langle\sigma_y\rangle` at all.

⛔ **This paragraph read** *"a mirror is a member of the discrete
residual* :math:`\Gamma`\ *"* **and quoted the row as**
``continuous_isotropy=Trivial, discrete_residual=Dnh(2)``, **until
2026-09-03.** True of the two-entry ledger; R3 of #434 split it into
three and re-bound :math:`\Gamma` onto the middle one
(:ref:`manifold-gamma-slot`). The reading (iii) makes — the two lookups
answer about different parts, so the ``sigma_y`` row is not a registry
gap — is unchanged, and is in fact sharper under three fields: a mirror
can now be a member of TWO of them at once, and the cylinder's is both
``unspent`` and inside ``owed``.

**(iv)** Two lookups of one fact is a Pattern-2 twin by construction,
and reading (iii) sharpens what the collapse *is*. The registry's
``support`` is not a rival catalogue; it is the **special case**
:math:`H = K`, the group the geometry SPENT, so the collapse is
``support = base.quotient(spent)`` rather than a merge of two tables.
(The slot was called ``continuous_isotropy`` and this paragraph wrote
:math:`G^0` for it until #434 R3 renamed it ``spent`` on 2026-09-03; the
argument is untouched.)

✅ **LANDED, tracker 2.4, 2026-09-01** — and the shipped form differs
from the one predicted here in one instructive respect. This paragraph
predicted ``base.quotient(G⁰).realization.name``, i.e. the *chart's
name*, because at the time ``support`` was a ``str``. What shipped is
``base.quotient(G⁰)`` — the **orbit space itself**, dropping both the
``.realization`` and the ``.name``. That is not a detail: taking the
realization is exactly the axis-blind step
(:ref:`manifold-so2-axis-is-a-parameter`), since all three
:math:`S^2/O(2)_a` realize onto the *same* interval. The prediction
was correct about which object is the special case and wrong about how
much of it to keep, and the intervening retype (2.0c) is what made the
better answer expressible.

.. admonition:: ✅ RESOLVED 2026-09-02 (#429 tracker 2.2b) — stage 0 reads
                the descent arrow
   :class: tip

   `[M]` 2026-09-02, against a pinned pre-change tree:
   ``GEOMETRY_ANGULAR_SYMMETRY["cylinder"].admits_domain`` on
   ``folded_product(4, 8).measure`` is now **True**, and so is
   ``admits_symmetry``. Over the five shipped ``Quadrature`` factories ×
   the four geometries the stage-0 refusal count went **12 → 10 of 20**,
   with no pair moving ``True`` → ``False``. The predicate is
   :eq:`manifold-gamma-slot-stage-zero`: a descent arrow
   :math:`\mathcal{D} \to X` must EXIST — equality being its identity
   case — and a containment must hold on the group the rule's orbit space
   was quotiented by (:ref:`manifold-gamma-slot`).

   ⛔ **The second conjunct read** *"…must lie in the residual*
   :math:`\Gamma` *the geometry still OWES"* **until 2026-09-03, and ONE
   of the two verdicts above went back.** R3 of #434 re-asked the
   containment against what the solution keeps UNSPENT rather than what a
   reflecting face is owed. `[M]` 2026-09-03 on the live tree the
   cylinder row is unchanged — the shipped fold is still admitted at both
   stages, which is what 2.2b was for — while
   ``GEOMETRY_ANGULAR_SYMMETRY["cartesian2d"].admits_domain(folded_product(4,
   8).measure)`` is **False** again, because a z-uniform plane's solution
   is even in :math:`\mu_z` alone (ERR-081). The refusal count above is a
   measurement of the 2.2b step against its own pre-change tree and is
   left as it was; the current grid is tabulated at
   :ref:`manifold-gamma-slot`, where it reads **11 admitted of 28** over a
   wider rule set.

   ⭐ **The paragraph below diagnosed this correctly and mis-called the
   remedy in one word, which is worth preserving.** *The mismatch was
   never the vocabulary; it was the claim* is exactly right, and it is
   why the fix is not a looser comparison: :math:`S^2/\sigma_y` and
   :math:`S^2` really are different orbit spaces and the gate really must
   not equate them. What the paragraph could not see is that they are
   related by an ARROW, and that the arrow — plus a requirement on what a
   fold may spend — is a *different* predicate rather than a weakened
   one. It is nonetheless weaker in effect, and the measurement says by
   how much: two of twenty pairs (one of which R3 took back, above).

   ⛔ **The block as it stood, kept verbatim** (``plan-authoring`` §3):

      ⚠ **A live consequence of (iii), re-measured 2026-09-01: still
      latent, and its stated CAUSE is now wrong.** ``admits_domain`` is
      ``measure.support == self.support``. `[M]` the cylinder declares
      ``S^2`` (a :class:`~orpheus.numerics.manifold.Sphere`) while the
      shipped cylindrical rule carries
      ``folded_product(4,8).measure.support == S^2/sigma_y`` (a
      :class:`~orpheus.numerics.manifold.Quotient`), so
      ``GEOMETRY_ANGULAR_SYMMETRY["cylinder"].admits_domain(...)`` is
      **False** — stage 0 would still reject the tree's own fold.

      ⛔ This block read "the gate is a string comparison … two different
      quotients that the string vocabulary cannot tell apart", verbatim,
      until 2026-09-01. Both halves are void: tracker 2.0c made it a
      ``Manifold`` value comparison, and the two quotients are now
      perfectly distinguishable — which is the point. **The mismatch was
      never the vocabulary; it was the claim.** A rule folded by a member
      of :math:`\Gamma` lives on :math:`S^2/\Gamma'` while the geometry
      declares :math:`S^2/G^0`, and those are two genuinely different
      orbit spaces. Typing them made the disagreement *legible* instead
      of removing it, which is the correct outcome and the reason the fix
      is still not to loosen the comparison.

      It bites nothing today for one reason only: `[M]` ``folded_product``
      is **not in** ``quadrature_registry`` (four specs ship —
      ``GaussLegendre1D``, ``LebedevSphere``, ``LevelSymmetricSN``,
      ``ProductQuadrature``), so the selector never presents it to stage
      0. The day it is registered, this is the first thing that fires.
      Recorded as a seam (:ref:`manifold-seams`).

   ⚠ That last paragraph's fact survives and its inference does not.
   ``folded_product`` is still unregistered — `[M]` 2026-09-02 the same
   four specs ship — so nothing in the SELECTOR exercises the new arm
   yet; what changed is that the gate no longer refuses the rule when it
   is asked directly. Registration is still blocked, now on stage 2
   rather than stage 0: `[M]` ``folded_product(4, 8).measure.exactness``
   is ``None`` (`GitHub #370
   <https://github.com/deOliveira-R/ORPHEUS/issues/370>`_, gap 1).


.. _manifold-second-twin-reference:

⭐ The SECOND twin on the same object: the reference measure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``support`` was not the only thing the registry answered twice.
:attr:`AngularSymmetry.reference
<orpheus.numerics.quadrature.registry.AngularSymmetry.reference>`
tabulated ``LEGENDRE`` for *any* axial rotation while the catalogue
entry — which carries every other output of the same derivation —
carried nothing. That is the identical shape, one level down: **the
support says which space, the reference says which measure on that
space**, and both are functions of the spent group alone.

✅ **Collapsed at tracker 3.1 (2026-09-02),** by the same move: the
property now reads :attr:`Quotient.reference
<orpheus.numerics.manifold.Quotient.reference>` off the entry
``support`` already returns, and the ``LEGENDRE`` import is gone from
``registry.py``. `[M]` re-measured on the shipped table after the
collapse — all four geometries, and the arm that raises:

.. list-table:: What the registry answers, and where the answer comes from
   :header-rows: 1
   :widths: 16 18 20 22 24

   * - Geometry
     - Spent :math:`K`
     - ``support.name``
     - ``reference.name``
     - Where it comes from
   * - ``slab``
     - ``O2('x')``
     - ``'S^2/O2_x'``
     - ``'legendre'``
     - the entry's field, **by** ``is`` **identity**
   * - ``sphere``
     - ``O2('x')``
     - ``'S^2/O2_x'``
     - ``'legendre'``
     - the same entry — the catalogue memoises, so these two rows are
       one object (:ref:`manifold-quotient-is-memoised`)
   * - ``cylinder``
     - ``Trivial``
     - ``'S^2'``
     - ``'uniform(S^2)'``
     - ⚠ **not** a catalogue read — the bare-sphere arm, below
   * - ``cartesian2d``
     - ``Trivial``
     - ``'S^2'``
     - ``'uniform(S^2)'``
     - the same arm
   * - *a spent* ``Mirror('y')``
     - ``Mirror('y')``
     - ``'S^2/sigma_y'``
     - :exc:`NotImplementedError`
     - the entry's ``reference`` is ``None``, and the message names
       the missing **work**
       (:ref:`manifold-pushforward-reference`)

⚠ **The** ``Trivial`` **row is the surviving divergence, and 3.1
widened it rather than closing it — user-ruled, 2026-09-02.** Reading
(ii) above already records why ``support`` short-circuits to the bare
sphere: a geometry that spends nothing discretises :math:`S^2`, so
stage 0's descent arrow must have the BASE as its source — which is what
lets a fold of it be admitted (:ref:`manifold-gamma-slot`). ⛔ This
clause read *"and 'S^2' is the name every 2-D/3-D rule declares, so
stage 0 must compare against the base"* until 2026-09-02: `[M]`
``folded_product(4, 8)`` is a 3-D-node rule declaring
``'S^2/sigma_y'``, and it is exactly the rule the arrow exists for. The reference arm inherits that: since the
domain handed out is the **base**, the measure on it is Lebesgue on
:math:`S^2`, which is
:data:`~orpheus.numerics.exactness.UNIFORM_ON_SPHERE` — and the
identity entry ships ``reference=None`` precisely because *that* answer
is a property of the base, not of :math:`M/\{e\}`
(:ref:`manifold-pushforward-reference`). ⟹ two producers on one row,
deliberately, in both columns; and the honest reading is that this arm
is not a twin at all, because the two are answering about **different
manifolds**.

⭐ **The mirror row is the reference collapse's own version of reading
(iii).** ``support`` answers it — a mirror is a legal argument to
:meth:`Manifold.quotient
<orpheus.numerics.manifold.Manifold.quotient>` — while ``reference``
raises, and the split is not an inconsistency. It is the difference
between *which orbit space* (derivable for any subgroup) and *which
measure on it* (derivable only where a shipped
:class:`~orpheus.numerics.exactness.ReferenceMeasure` realization
spells the pushforward). `[M]` it bites no shipped geometry, since no
geometry spends a mirror — reading (iii) again — so it is a witness
rather than a defect.


.. _manifold-orbit-space-declaration:

The first production consumer: a quadrature DECLARES its orbit space
=====================================================================

Everything above is about the type. This section is about the day it
was first *used*, which is tracker 2.4 (2026-09-01): the slab's polar
quadrature stopped naming the interval a chart happens to map onto and
started naming the orbit space it lives on.

.. list-table:: The slab's polar rule, before and after
   :header-rows: 1
   :widths: 30 32 38

   * -
     - Before (the chart)
     - After, `[M]` 2026-09-01
   * - ``measure.support.name``
     - ``'[-1,1]'``
     - ``'S^2/O2_x'``
   * - ``measure.space.name``
     - ``'L2[[-1,1]]'``
     - ``'L2[S^2/O2_x]'``
   * - ``measure.quotient_group``
     - ``None``
     - ``SubgroupOfO3.O2('x')`` (``SO2('x')`` until #432, 2026-09-02)
   * - ``measure.phase``
     - ``'angular'``, via a **fallback** on the :math:`O(3)` tag
     - ``'angular'``, from the **manifold alone**
   * - nodes / weights
     - —
     - **bit-identical** to ``gauss_legendre_on_mu(8)``
       (``np.array_equal`` on both)

⭐ **This is a repair, not wiring, and the measurement says so.** Take
the eight nodes and weights of ``Quadrature.gauss_legendre(8)`` and
build a *spatial* rule from them on ``Interval(-1, 1)``. Before the
declaration the two induced function spaces were `[M]` ``==`` **and
hash-equal** — an 8-node slab angular space and an 8-node spatial rule
were the same value, so any cache, ``set`` or operator-domain check
keyed on the space would have conflated them. After it, `[M]` ``==`` is
``False`` and the hashes differ. That is the energy/spatial collision of
tracker 2.1 recurring one level up, and 2.0c could not close it: while
both supports were honestly ``Interval(-1, 1)``, there was nothing to
tell apart.

.. _manifold-on-orbit-space:

``on_orbit_space`` — the third verb, and why it is neither of the others
------------------------------------------------------------------------

The declaration is performed by a new measure verb,
:meth:`DiscreteMeasure.on_orbit_space
<orpheus.numerics.measure.DiscreteMeasure.on_orbit_space>`. Its
semantics are equation-free, which is the whole of its content:
**the same atoms, re-read as chart coordinates of an orbit space.**
Same nodes, same weights — `[M]` the *same array objects*, not copies —
and only what the measure KNOWS about its support changes, from "an
interval" to "the polar marginal of a sphere, about this axis".

It is easy to mistake for the two verbs the corpus already has, and it
is neither:

.. list-table::
   :header-rows: 1
   :widths: 20 26 26 28

   * -
     - ``pushforward(φ)``
     - ``quotient(G)``
     - ``on_orbit_space(M/H)``
   * - Starts from
     - a measure on :math:`\mathcal{X}`
     - a measure on the **base** :math:`M`
     - a measure on the **chart** :math:`C`
   * - Does it move a node?
     - yes, applies :math:`\varphi`
     - no — selects orbit representatives
     - **no — applies nothing at all**
   * - Node count
     - unchanged
     - one per orbit (drops)
     - unchanged
   * - Mass
     - preserved
     - preserved (orbit-stabilizer weights)
     - **untouched**
   * - What changes
     - the points
     - the points and the support
     - **only the support**

⟹ a :math:`\mu`-rule was never on :math:`S^2` to begin with, so there
is no fold to perform; and no map is applied, so there is no
pushforward. The verb exists because "this list of numbers is a
coordinate list *for* :math:`S^2/O(2)_x`" is a fact about the measure's
type that no arithmetic can supply.

**The one precondition, refused where the declaration is written.**
``on_orbit_space`` raises unless the orbit space's ``realization`` **is**
this measure's current support — the chart it was built on. `[M]`
handing a :math:`\mu`-rule the mirror quotient
``SPHERE.quotient(SubgroupOfO3.Mirror('y'))``, whose chart is the disk
:math:`D^2`, raises verbatim:

.. code-block:: text

   a measure on '[-1,1]' cannot be read on 'S^2/sigma_y': that orbit
   space's chart is 'D^2'. A rule declares the orbit space whose CHART
   it was built on; to fold a rule on the base manifold, use quotient()

The message names the mismatch *and* the other verb, because the two
failure modes ("wrong orbit space" and "you meant to fold") are exactly
the two ways a caller gets here.

**What the metadata does, and why.**

* ``invariance_group`` is **DROPPED**. A subgroup of :math:`O(3)` is a
  claim about an *embedding*, and the orbit space fixes an embedding —
  its axis — that the chart did not. The adopter re-tags: `[M]`
  :func:`~orpheus.numerics.quadrature.gauss_legendre_on_polar_orbit`
  immediately stamps ``Mirror(axis)`` back, for the *named* axis, which
  is a strictly more specific claim than the chart's could be. This is
  the same discipline the metadata-propagation table on
  :doc:`/theory/foundations/discrete_measures` applies everywhere: the
  field becoming ``None`` is correct behaviour, and the caller who knows
  the residual re-establishes it.
* ``exactness`` **survives**. The reference measure is a measure on the
  chart and the chart is unchanged, so the claim is untouched: `[M]`
  ``exact to algebraic degree 15 against legendre`` before and after,
  on ``n = 8``.

.. _manifold-polar-orbit-rule:

Two objects on one interval: the chart rule and the orbit rule
---------------------------------------------------------------

The declaration could not simply be pushed into
:func:`~orpheus.numerics.quadrature.gauss_legendre_on_mu`, and the
reason is the two-poles fact of
:ref:`manifold-so2-axis-is-a-parameter` in its operational form. That
function serves **two** roles, and only one of them has an axis to name:

* it is the raw material of the slab's rule, whose :math:`\mu` is the
  cosine against :math:`x`;
* it is the **polar factor** of every product rule in
  :mod:`orpheus.numerics.quadrature.rules_product`, whose :math:`\mu` is
  the cosine against :math:`z`.

A factor that declared :math:`S^2/O(2)_x` while sitting inside a
product about :math:`z` would be a false claim about the object it is
part of. So the tree carries two functions:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * -
     - ``gauss_legendre_on_mu(n)``
     - ``gauss_legendre_on_polar_orbit(n, axis)``
   * - Support
     - ``COSINE_INTERVAL`` — the chart, naming no axis
     - ``SPHERE.quotient(O2(axis))``
   * - ``invariance_group``
     - ``Mirror('x')`` (the canonical :math:`(\mu,0,0)` embedding)
     - ``Mirror(axis)``
   * - Consumed by
     - the product rules' polar factor; the raw material below
     - :meth:`Quadrature.gauss_legendre
       <orpheus.numerics.quadrature.directional.Quadrature.gauss_legendre>`
       and the ``GaussLegendre1D`` registry spec
   * - In the registry?
     - **no**, deliberately
     - **yes**, as ``partial(..., axis="x")``

⭐ **The chart-level rule is deliberately NOT registered, and stage 0 is
what enforces it.** `[M]` re-measured 2026-09-02 on the slab's
``AngularSymmetry``, and every reading is unchanged by tracker 2.2b:
``admits_domain`` is **False** for ``gauss_legendre_on_mu(8)`` (support
``[-1,1]``), **False** for a marginal declared about :math:`y` or
:math:`z`, and **True** only for the :math:`x`-declared rule — `[M]`
all four re-measured unchanged on 2026-09-03, after #434 R3 gave the
second conjunct a different right operand. The
geometry names the group it spends and the rule names the group its
orbit space was quotiented by, so a rule about the wrong pole is refused
by the same predicate that refuses a sphere cubature.

⛔ This paragraph said the predicate compares *"one fact on both sides"*
until 2026-09-02. That was equality, and stage 0 is no longer equality:
it asks for a descent ARROW plus a containment
(:eq:`manifold-gamma-slot-stage-zero`, :ref:`manifold-gamma-slot`). The
three readings above are unaffected, for a reason worth stating: `[M]`
``quotient_onto`` finds **no** arrow from :math:`S^2/O(2)_x` onto
:math:`[-1,1]`, onto :math:`S^2/O(2)_y` or onto :math:`S^2/O(2)_z` — a
chart is not an orbit space and two distinct axial quotients are
incomparable — so all three still fail the FIRST conjunct.

`[M]` re-measured 2026-09-03, the live rejection text a slab selection
emits for the three :math:`S^2` rules reads:

.. code-block:: text

   domain mismatch: geometry 'slab' discretises S^2/O2_x, but the rule's
   nodes live on S^2, and S^2/O2_x has no descent arrow onto it

⛔ That text ended *"— no descent arrow onto it, or a fold by a group
outside the owed sigma_x"* from 2026-09-02 until #434 R3 the next day.
It was a DISJUNCTION over the two conjuncts, printed whichever bit;
``domain_refusal`` now returns the one failing clause and the selector
appends it (:ref:`manifold-gamma-slot`).

.. note::

   ⚠ **The** ``phase`` **fallback arm did NOT become unreachable, and a
   pre-flight predicted it would.** :attr:`DiscreteMeasure.phase
   <orpheus.numerics.measure.DiscreteMeasure.phase>` classifies a
   measure by the TYPE of its support manifold, with one fallback: a
   rule on a bare :class:`~orpheus.numerics.manifold.Interval` carrying
   an :math:`O(3)` invariance tag is angular. Tracker 2.4 was expected
   to close that arm by making the slab declare a sphere quotient. It
   did — for the slab. `[M]` over eight shipped rules
   (``gauss_legendre_on_mu``, ``gauss_legendre_on_polar_orbit``,
   ``Quadrature.gauss_legendre`` / ``product`` / ``folded_product`` /
   ``level_symmetric`` / ``lebedev``, ``periodic_trapezoid``) the arm is
   reached by exactly **one**: ``gauss_legendre_on_mu`` itself, which for
   the reason above must keep a bare interval. Every other rule now
   answers from its manifold's type. The honest statement is that the
   fallback is **scoped to the chart-level rule**, not retired — and it
   is a live example of the plan hazard where an "unreachable after this
   step" prediction is falsified by the same step's own design
   constraint.

.. _manifold-the-axis-convention-for-a-section:

What a section will have to choose, in the axis's language
------------------------------------------------------------

:ref:`manifold-err-080-is-a-section` establishes that ERR-080's level-1
half is a fabricated **section** of :math:`S^2 \to S^2/O(2)_x`, and that
``fundamental_domain=None`` is the honest entry because no section is
canonical. Tracker 2.4 does not change that — it changes the *language*
the eventual choice has to be made in, and that is worth writing down
before someone makes it.

Now that the axis is a parameter, the :math:`\varphi = 0` half-meridian
has an axis-general spelling. Writing :math:`a` for the rotation axis
and :math:`b, c` for the other two, the candidate section is

.. math::

   \mu \;\longmapsto\;
   \mu\,\hat e_a \;+\; \sqrt{1-\mu^2}\,\hat e_b \;+\; 0\cdot\hat e_c ,

which is on :math:`S^2` by construction. `[M]` for :math:`a = x` this is
:math:`\mu \mapsto (\mu, \sqrt{1-\mu^2}, 0)` and it is the object the
tree fabricated as :math:`(\mu, 0, 0)` — the fabrication is the same map
with the :math:`\hat e_b` term dropped, which is exactly why its rows
have norms :math:`|\mu| < 1` rather than :math:`1`.

⚠ **It is still a CHOICE.** Every half-meridian is a section;
:math:`\varphi = 0` merely names one, and which one you pick is a
convention about where azimuth zero sits, not a derivation. The shipped
catalogue entry therefore keeps ``fundamental_domain=None`` on purpose,
and the choice belongs to whichever step declares a section — not to
the orbit-space derivation, which would then be smuggling a convention
into a theorem.

.. note::

   ⛔ **This paragraph read "and it belongs to tracker 2.3 … the step
   that mints the typed** ``Chart`` **— not to the orbit-space
   derivation" until 2026-09-02.** Tracker 2.3
   landed on that date and the prediction is half right, in the way
   worth recording (``coding-standards``, the *falsified prediction*
   tense class): the **phase** was right and a typed map did land, but
   it is :class:`~orpheus.numerics.manifold.ManifoldMap` rather than a
   ``Chart``, and **the choice was not made**. `[M]` 2026-09-02 the
   three arrows 2.3 types are a parametrisation of the *base*
   (``archimedes``), a per-measure orbit *retraction*, and the orbit
   *barycentre*, which lands off :math:`S^2` by construction — none of
   them is a section, and
   :attr:`Quotient.fundamental_domain
   <orpheus.numerics.manifold.Quotient.fundamental_domain>` is still
   read by nothing outside :mod:`orpheus.numerics.manifold` itself.
   The naming ruling is the reason: *a chart is* :math:`M \supset U \to
   \mathbb{R}^n`, and only the **inverse** of the Archimedes map is
   one, so a type called ``Chart`` would have mis-described two of its
   own three instances (:ref:`manifold-arrows`). The section is still
   owed, and it is still a choice.


.. _manifold-arrows:

Maps between manifolds: the ARROWS
==================================

Everything above this point is about **objects**. A category needs
arrows too, and the tree had been drawing three of them freehand:
whenever a construction wanted to move a point set somewhere else it
applied a callable and then *named the destination by hand*, at the
call site, in whatever vocabulary was locally convenient. That is the
shape a forged claim takes — the same shape :ref:`ERR-080
<manifold-err-080>` has — because a destination named at the call site
is a claim nobody else made and nothing can contradict.

Tracker 2.3 (2026-09-02) gives the arrows a type. It adds no
mathematics: every one of the three maps below was already being
computed, correctly, in production. What it adds is that the
**codomain travels with the map** instead of being supplied by the
caller, so *"apply this map and declare the result to live on*
:math:`S^2`\ *"* stops being a sentence anyone can write.


.. _manifold-arrow-type:

The type: a map carries its two point sets
-------------------------------------------

:class:`~orpheus.numerics.manifold.ManifoldMap` is a frozen value with
three fields — ``domain``, ``codomain``, ``apply`` — and it is the
point-level analogue of a
:class:`~orpheus.numerics.operator.LinearOperator`: where an operator
carries the two *function spaces* it maps between, a map carries the
two *point sets*. The design ruling (user, 2026-09-02) was for **one**
value type with named maps as factories — ``archimedes(axis)``,
``barycentre(orbit_space)`` — exactly as
:data:`~orpheus.numerics.manifold.SPHERE` and ``LEGENDRE`` are values
of their own types rather than subclasses of them.

Two arrows are *induced* by every :math:`\varphi : M \to N`, and they
are the reason the type exists rather than being decoration:

- the **pushforward of a measure**,
  :meth:`DiscreteMeasure.pushforward
  <orpheus.numerics.measure.DiscreteMeasure.pushforward>` — the image
  measure :math:`\varphi_*\mu` lives on :math:`N` *because*
  :math:`\varphi` says so (:eq:`discrete-measure-pushforward`);
- the **pullback of a function**, :math:`f \mapsto f \circ \varphi`,
  which is what the change-of-variables identity evaluates on the
  pushed measure. ⛔ Nothing in the tree applies a pullback through
  this type yet: the planned consumer is the map that restricts a
  basis to an orbit space (tracker 3.4b), and until it lands the
  second arrow is a statement about what the type *is*, not about
  what ships.

⭐ **The verb that changed is the pushforward, and its three states are
the campaign in miniature.** The same operation, spelled three ways
across five weeks:

.. list-table:: ``pushforward`` — who names the target
   :header-rows: 1
   :widths: 16 40 44

   * - When
     - Spelling
     - Who names :math:`\mathcal{Y}`, and what could contradict them
   * - until 2026-09-01
     - ``μ.pushforward(f)``
     - **Nobody.** The support was *fabricated* as
       ``f"φ_*({self.support})"`` — a manifold nobody has derived,
       wearing a name that makes it look like one
       (:ref:`manifold-string-algebra`).
   * - 2026-09-01 (2.0c)
     - ``μ.pushforward(f, new_space=Y)``
     - **The call site.** An improvement — only :math:`\varphi`'s
       author knows :math:`\mathcal{Y}`, and now somebody had to say
       it — but the caller is not always that author, and a caller who
       is wrong is unopposed.
   * - 2026-09-02 (2.3)
     - ``μ.pushforward(φ)``
     - **The map.** ``new_space=`` is retired; the target is *read*
       off ``φ.codomain``, and the verb additionally **refuses** a map
       whose ``domain`` is not this measure's support.

The refusal is by manifold **value**, not by array shape, and the
sharpest witness for that is a pair of measures carrying *literally the
same numbers*. `[M]` 2026-09-02:
``gauss_legendre_on_polar_orbit(4, "x").nodes`` is ``np.array_equal``
to ``gauss_legendre_on_mu(4).nodes`` — the slab's rule is the chart
rule with a declared orbit space (:ref:`manifold-polar-orbit-rule`) —
and handing the first to the product embedding raises where the second
is accepted:

.. code-block:: text

   ValueError: cannot push a measure on 'S^2/O2_x × S^1' forward
   along a map out of '[-1,1] × S^1': the map's domain must be the
   measure's support. Build the map out of 'S^2/O2_x × S^1', or hand
   this verb a measure on '[-1,1] × S^1' — the same numbers on a
   different manifold are a different measure.

Note where the difference surfaces: not on the polar factor but on the
**product**, because :meth:`Manifold.__mul__
<orpheus.numerics.manifold.Manifold.__mul__>` carried it there. The
algebra of objects is what makes the arrow's guard discriminating.

.. note::

   **No membership check runs inside the map**, deliberately. A map
   whose ``apply`` lands outside its declared codomain is a real
   defect, and its ruled home is
   :meth:`~orpheus.numerics.manifold.Manifold.contains` at *measure
   construction* (tracker 2.0b) — one refusal, on the object that
   actually escapes, rather than two half-refusals. ⛔ That is still
   **not built**: nothing calls ``contains`` on the way in, so a
   measure whose nodes are not points of its support remains
   constructible (:ref:`manifold-seams`). ⛔ The clause, verbatim,
   "and **ERR-080 remains open**" stood here until 2026-09-02 and is
   repealed: the defect was closed that day by refusing the *pairing*
   rather than the forged measure, so a forgery is still spellable and
   is no longer reachable by any basis.

   The **Jacobian** is likewise not this type's business. ``pushforward``
   is the :math:`\varphi`-image with weights preserved verbatim; a
   change of variables against a target *reference* measure is the
   caller's, and that asymmetry is documented where the identity lives
   (:doc:`/theory/foundations/discrete_measures`).


.. _manifold-three-arrows:

The three maps the tree was already spelling
---------------------------------------------

`[M]` 2026-09-02, at 2.3's opener the tree drew three arrows around the
quotient and none of them was typed. One of them it drew **twice** —
once honestly and once as ERR-080.

.. list-table:: Three arrows, four spellings
   :header-rows: 1
   :widths: 22 30 24 24

   * - Arrow
     - How it was spelled
     - Codomain, before
     - Codomain, after
   * - the orbit retraction
       :math:`M \to M/H`
     - a ``lambda`` plus a hand-written ``new_space=``, inside
       :meth:`DiscreteMeasure.quotient
       <orpheus.numerics.measure.DiscreteMeasure.quotient>`
     - named at the call site
     - ``self.support.quotient(group)`` — the **catalogue's own
       object**, by identity
   * - the Archimedes chart
       :math:`[-1,1]\times S^1 \to S^2`
     - a hand-written double loop plus the literal
       ``support=SPHERE``, inside ``spherical_product``
     - a literal
     - :data:`~orpheus.numerics.manifold.SPHERE`, **read** off
       ``archimedes("z").codomain``
   * - the orbit barycentre
       :math:`S^2/O(2)_a \to D^3`, **honest** spelling
     - inline in ``invariance._embedded_nodes``, which embeds a polar
       marginal for an invariance check
     - (untyped — a bare array)
     - :class:`~orpheus.numerics.manifold.Ball`\ ``(3)``, via
       :func:`~orpheus.numerics.manifold.barycentre` — and since
       2026-09-02 reached through
       :attr:`Quotient.lift
       <orpheus.numerics.manifold.Quotient.lift>`, which at #434 R4
       (2026-09-03) stopped being a three-family branch and became ONE
       formula on a stored field, the Reynolds projector
       (:ref:`manifold-lift`)
   * - the **same** map, **forged** spelling
     - inline in ``Quadrature._harmonic_frame_measure``'s 1-D arm
       (**deleted** 2026-09-02)
     - ⛔ ``support=SPHERE`` — **a lie**
     - ✅ **RETIRED** with #429's fused commit; unchanged **by design**
       until then (see below)

⭐ **The fourth row is the whole argument.** Rows three and four compute
*the same image* — `[M]` 2026-09-02 on ``gauss_legendre(8)``, the
forgery's nodes are ``np.array_equal`` to
``barycentre(measure.support)(measure.nodes)`` — and differ only in
what they claim about where that image lives. The honest one says
:math:`D^3` and is right (`[M]` ``Ball(3).contains`` → ``True``); the
forged one says :math:`S^2` and is wrong (`[M]` ``Sphere().contains``
→ ``False``, norms :math:`0.1834 \ldots 0.9603`). A codomain that is a
**field of the map** cannot be forged at the call site; that is the
entire purchase of this type.

The forgery therefore **stayed a raw constructor** until tracker 3.4
retired it (2026-09-02), and that was not an oversight: it *cannot* be
re-spelled through ``pushforward`` without telling the truth about its
codomain, so re-spelling it would have silently repaired ERR-080's
level-1 half in a step whose subject was the type system. The arm
carried a comment naming the map it was a forgery of.

⭐ **And the repair was not "give it the right codomain", which is worth
recording because that is the reading this table invites.** Nothing on
the 1-D side ever *wanted* a point of :math:`S^2`: the barycentre was
being computed to feed a basis, and the basis was the wrong one. The fix
changes the BASIS — a 1-D rule binds the Legendre family its own orbit
space admits — and the map has nothing left to do
(:ref:`manifold-what-descends`). A forged codomain is a defect you can
see from the type system; *which* repair it calls for is a question the
type system does not answer.


.. _manifold-archimedes:

Archimedes: :math:`[-1,1] \times S^1 \to S^2`
----------------------------------------------

Write :math:`a` for an axis and :math:`b, c` for its two cyclic
successors (:math:`z \to x, y`; :math:`x \to y, z`; :math:`y \to z, x`
— a right-handed frame in every case). The map is

.. math::

   \varphi_a(\mu, (\cos\varphi, \sin\varphi))
   \;=\; \mu\,\hat e_a
   \;+\; \sqrt{1-\mu^2}\,\bigl(\cos\varphi\,\hat e_b
                               + \sin\varphi\,\hat e_c\bigr),

which for :math:`a = z` is the direction-cosine triple every product
rule has always used —
:math:`\mu_x = \sin\theta\cos\varphi`,
:math:`\mu_y = \sin\theta\sin\varphi`,
:math:`\mu_z = \mu`, with :math:`\sin\theta = \sqrt{1-\mu^2}` — stated
as an equation in the module docstring of
``orpheus.numerics.quadrature.rules_product`` and pinned against the
map by hand in
``tests/numerics/test_manifold.py::test_archimedes_about_z_is_the_labelled_equation_verbatim``.

**Its relation to the orbit-space chart is exact, and it is a
projection.** The chart of :math:`S^2/O(2)_a` is
:math:`\pi(\Omega) = \Omega\cdot\hat e_a`
(:ref:`manifold-s2-so2`, step 5), and composing gives

.. math::

   \pi \circ \varphi_a \;=\; \mathrm{pr}_1 ,

`[M]` bit-exactly: over 500 random :math:`(\mu, \varphi)` pairs per
axis, :math:`\max\lvert \pi(\varphi_a(\mu,\varphi)) - \mu \rvert =`
**0.0** for :math:`a \in \{x,y,z\}`, with
:math:`\max\bigl\lvert\lVert\varphi_a\rVert - 1\bigr\rvert \le`
**2.22e-16**. So the polar factor of a product rule *is* the orbit-space
coordinate, and the circle factor is the fibre the quotient forgets —
which is why one Gauss–Legendre rule can serve both a slab marginal and
a product rule's polar factor
(:ref:`manifold-so2-axis-is-a-parameter`).

.. admonition:: Why it is named for Archimedes
   :class: note

   **Archimedes' hat-box theorem** is the statement about this map that
   transport actually needs: the pushforward of the uniform measure
   :math:`d\Omega` along :math:`\mu = \Omega\cdot\hat e_a` is
   :math:`2\pi\,d\mu` — *uniform* on :math:`[-1,1]`, for every axis. A
   Gauss–Legendre rule in :math:`\mu` times any circle rule is therefore
   exact on the sphere against Lebesgue measure, which is the theorem
   :func:`~orpheus.numerics.quadrature.rules_product.spherical_product_claim`
   composes the two factors' claims through.

   ⚠ **Edited elsewhere, consumed here — and the theorem is now spelled
   in THREE registers, one per page, on purpose.**
   :doc:`/theory/foundations/discrete_measures` owns it in the
   *selection* register (which reference measure a rule's exactness
   claim must match, and why a degree without one is meaningless); this
   page owns it in the *map* register, where the hat-box is a statement
   about :math:`\varphi_a`; and since tracker 3.1 it is a **derivation
   output** with a labelled equation and a field to sit in
   (:eq:`manifold-quotient-pushforward`,
   :ref:`manifold-pushforward-reference`). ⛔ This paragraph read
   *"…and why it keys on* ``rotation_axis`` *being non-*``None`` *rather
   than on one group"* until 2026-09-02. That is history: the registry
   keys on nothing — it *reads*
   :attr:`Quotient.reference
   <orpheus.numerics.manifold.Quotient.reference>` off the entry, and
   the axis-generality the old predicate bought is now a property of
   the derivation, which reads its axis off the group and returns
   ``LEGENDRE`` for all three.

⚠ **It is a parametrisation, not a chart in the strict sense**, and the
place it fails is not incidental — it is the stratum. The circle factor
**collapses** at :math:`\mu = \pm 1`: a whole fibre maps to one pole.
That is exactly the singular locus of :math:`S^2/O(2)_a`
(:ref:`manifold-singular-stratum`), so the map is injective off the
stratum and the stratum is precisely where it is not. Its inverse on
:math:`S^2 \setminus \{\pm\hat e_a\}` is the :math:`(\mu, \varphi)`
chart. The collapse is gated directly: `[M]` on a
:math:`7 \times 8` grid the eight images at :math:`\mu = +1` are
``np.array_equal`` to :math:`\hat e_a` repeated, and likewise at
:math:`\mu = -1`.

**What the product rule now reads.**
:func:`~orpheus.numerics.quadrature.rules_product.spherical_product`
was a hand-written double loop that filled a
:math:`(n_\mu n_\varphi, 3)` array and then declared
``support=SPHERE``. It is now the algebra it always was:

.. code-block:: python

   measure = (polar * azimuthal).pushforward(archimedes("z")).with_metadata(
       invariance_group=group,   # DERIVED from the factors' generators
       exactness=claim,          # DERIVED through the product theorem
   )

— the tensor product is the measure's own
:meth:`~orpheus.numerics.measure.DiscreteMeasure.__mul__` on the
product manifold :math:`[-1,1]\times S^1`, the embedding is the typed
chart, and the support is the chart's codomain. `[M]` 2026-09-02,
re-run independently of the gate: over **60** configurations
(:math:`n_\mu \in \{2,3,4,5,6\}` :math:`\times` :math:`n_\varphi \in
\{6,8,10,16,24,32\}` :math:`\times` both circle shifts), **0** differ
from the transcribed pre-2.3 loop — ``np.array_equal`` on nodes and on
weights — and ``measure.support`` **is** ``archimedes("z").codomain``,
one object rather than a literal that happens to agree.

⭐ Two derivations of the same rule now meet: the *claim* side composes
the factors' exactness through the product theorem, and the *measure*
side composes their atoms through this chart. Neither reads the other,
and both refuse a mismatched factor pair — the claim side on the
exactness system, the measure side on the manifold.


.. _manifold-barycentre:

The orbit barycentre — the Reynolds projector :math:`P_H`, and why it is not a section
----------------------------------------------------------------------------------------

An orbit of the axial group :math:`O(2)_a` — equivalently of its
rotation half, which has the same orbits
(:ref:`manifold-orbit-space-stabiliser`) — is the circle
:math:`\{\Omega : \Omega\cdot\hat e_a = \mu\}`, of radius
:math:`\sqrt{1-\mu^2}` about the point :math:`\mu\,\hat e_a` on the
axis. That point is the orbit's **barycentre** — its mean under the
fibre's uniform measure — and the map

.. math::

   \beta_a : \mu \;\longmapsto\; \mu\,\hat e_a

is what a consumer wanting *one representative point per orbit* keeps
reaching for. Where it lands is a one-line computation, and it is the
whole story:

.. math::

   1 - \lVert \mu\,\hat e_a \rVert^2 \;=\; 1 - \mu^2
   \;=\; \tfrac14 \det P ,

the **squared orbit radius**, which is the quantity the catalogue
already records as :attr:`Quotient.det_gram
<orpheus.numerics.manifold.Quotient.det_gram>` — `[M]` the shipped
entry's ``det_gram`` is :math:`4 p_2` (:eq:`manifold-s2-mod-so2`),
which restricted to the sphere by :math:`p_1^2 + p_2 = 1` is
:math:`4(1-\mu^2)` (:ref:`manifold-s2-so2`, step 5); the identity
reproduces to **0.0** over nine :math:`\mu` values. So :math:`\beta_a` lands **on**
:math:`S^2` exactly where the orbit radius vanishes — the two poles,
i.e. the singular stratum — and strictly **inside** the ball
everywhere else. Its codomain is
:class:`~orpheus.numerics.manifold.Ball`\ ``(3)`` and can be nothing
else: `[M]` ``Ball(3).contains`` → ``True`` on the whole image,
``Sphere().contains`` → ``False`` on the interior and ``True`` on the
two poles.

.. _manifold-reynolds-projector-section:

:math:`\beta_a` is one instance of ONE map — the Reynolds projector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paragraphs above derive :math:`\beta_a` for the axial family, which
is how the tree met it. It is not a family-specific formula. For a
compact group :math:`H` acting orthogonally on :math:`\mathbb{R}^n`, the
**mean of an orbit** is the image of the point under the **Reynolds
operator** — the average of the representation over the group's Haar
measure —

.. math::
   :label: manifold-reynolds-projector

   P_H \;=\; \int_H \rho(g)\, \mathrm{d}g ,
   \qquad
   P_H^2 = P_H = P_H^{\mathsf T} ,
   \qquad
   \operatorname{ran} P_H = (\mathbb{R}^n)^H ,

the orthogonal projector onto :math:`H`'s **fixed subspace**. Three
one-line proofs, and each is a property the code relies on. Idempotence:
:math:`P_H P_H = \int_H\!\int_H \rho(gh)\,\mathrm{d}g\,\mathrm{d}h =
P_H` by invariance of Haar measure. Symmetry: :math:`\rho` is orthogonal
and Haar measure is inversion-invariant, so :math:`P_H^{\mathsf T} =
\int_H \rho(g)^{-1}\mathrm{d}g = P_H`. Range: :math:`\rho(h) P_H = P_H`
for every :math:`h`, so the image is fixed, and :math:`P_H` is the
identity on anything already fixed.

⭐ **And the projector is what a COORDINATE chart already computes.**
When the surviving invariants of an entry are *coordinate functions* —
which is true of both shipped sphere families — the chart is a column
selection and its lift is the scatter of those columns back into a zero
vector, so the composite is the orthogonal projector onto the span of
those axes. That span *is* :math:`(\mathbb{R}^3)^H`, so

.. math::

   \underbrace{\lambda}_{\text{embed}} \circ
   \underbrace{\pi}_{\text{select}} \;=\; P_H .

One helper spells the pair —
``_coordinate_chart(columns, ambient) -> (select, embed)`` — and the two
sphere builders call it with their invariant columns, so an entry cannot
carry a chart and a lift that disagree.

.. list-table:: `[M]` 2026-09-03 — :math:`P_H` on the eight entries the tree constructs
   :header-rows: 1
   :widths: 22 20 24 34

   * - Entry
     - :math:`\operatorname{diag} P_H`
     - The lift, in coordinates
     - Its fixed subspace :math:`(\mathbb{R}^3)^H`
   * - :math:`S^2/O(2)_a` (axial, 3 entries)
     - one :math:`1`, on :math:`a`
     - :math:`\mu \mapsto \mu\,\hat e_a`
     - the axis :math:`\mathbb{R}\hat e_a` — the centre of the
       constant-:math:`\mu` circle
   * - :math:`S^2/\sigma_a` (mirror, 3 entries)
     - two :math:`1`\ s, off :math:`a`
     - :math:`(x_b, x_c) \mapsto (0, x_b, x_c)`
     - the mirror plane :math:`\{x_a = 0\}` — the midpoint of
       :math:`\{p, \sigma_a p\}`
   * - :math:`M/\{e\}` (2 constructible)
     - all :math:`1`\ s
     - the identity
     - everything — :math:`P_{\{e\}} = I`

`[M]` 2026-09-03, over **eight** entries — the six catalogue keys, plus
:math:`S^2/\{e\}` and the ambient :math:`\mathbb{R}^3/\{e\}` the
invariance kernel is asked on; :math:`M/\{e\}` is derivable for every
manifold, so eight is the shipped population and not a closed set — ×
41 seeded unit vectors, against a reference built from the group's
**realized matrices** and never from a column index — an orthonormal
basis of :math:`\bigcap_{X \in \mathfrak h} \ker X \cap
\bigcap_{r} \ker(r - I)` by SVD, then :math:`B B^{\mathsf T}`:
``embed ∘ select`` reproduces :math:`P_H` with ``np.array_equal`` on
**8 of 8**, ``max|Δ| = 0.000e+00``, and :math:`P_H` is a **0/1 diagonal**
on every one, hence bit-exactly idempotent and symmetric.

⚠ **The bit tier is a property of the SHIPPED entries, not of the
construction.** It holds because every shipped :math:`H` is
axis-aligned, so :math:`P_H` is diagonal and ``embed`` re-writes the
same floats ``select`` read. An entry whose :math:`H` is not
axis-aligned gives a dense :math:`P_H`, and the comparison then belongs
at ``assert_array_almost_equal_nulp`` (``vv-principles`` #31 — a
bit-exactness claim is a property of the fixture until a measurement
makes it a property of the construction).

⭐ **Two independent constructions confirm that** ``embed ∘ select``
**really is the orbit mean, one per family, and neither of them looks
at a column.** For a FINITE :math:`H` the mean is a finite average over
the group's own element list: `[M]` :math:`\lambda(\pi(p)) =
\tfrac12\sum_{g\in\langle\sigma_a\rangle} g\,p` is ``array_equal``
on all three mirror entries, exactly — :math:`(x + (-x))/2` is
:math:`0.0` and :math:`(y+y)/2` is :math:`y` in IEEE-754. For the axial
family the orbit is a circle and the mean is an integral, taken here by
an :math:`n`-point trapezoid over :math:`R_\theta p`: `[M]` the residual
against the shipped lift is ``2.220e-16`` at :math:`n = 8`,
``3.331e-16`` at 16, ``6.661e-16`` at 32, ``1.554e-15`` at 64 and
``2.831e-14`` at 1024. ⚠ **More points is worse**, not better: the
trapezoid integrates :math:`\cos\theta` and :math:`\sin\theta`
*exactly* for :math:`n \ge 3`, so everything past that is summation
error. A gate on this instrument wants :math:`n = 16`, and its
docstring must say so, or a future session will "strengthen" it into a
false red.

.. (vv-status rationale) manifold-reynolds-projector is a
   LITERATURE-TRANSCRIBED definition from representation theory (the
   Reynolds operator of a compact group), stated here so the catalogue's
   ``lift_coordinates`` field has a name for what it computes. It makes
   no solver claim and there is no ORPHEUS function that evaluates the
   Haar integral. What IS verifiable is its INSTANCE — that
   ``embed ∘ select`` equals P_H on every shipped entry, against the
   SVD/group-mean/trapezoid references measured in the paragraphs above
   — and that instance is gated by
   tests/numerics/test_manifold.py::TestR4TheCoordinateChartPairIsTheReynoldsProjector
   and ::TestR4TheLiftIsTheOrbitBarycentre, whose rows carry
   @pytest.mark.foundation and deliberately NO verifies(...) per
   vv-principles' foundation-tier rule.
.. vv-status: manifold-reynolds-projector documented

.. _manifold-barycentre-not-a-section:

Why it is still not a section
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

⛔ **It is not a section, and the distinction is the one ERR-080 got
wrong.** A section of :math:`M \to M/H` lands **on** :math:`M` by
picking a representative; :math:`P_H` lands on the FIXED SUBSPACE, and a
mean of unit vectors is not a unit vector. For the axial family it meets
:math:`S^2` only at the two poles, where the orbit is a point; for a
mirror entry only on the equator :math:`x_a = 0`, which is the mirror's
own fixed locus — and in both cases that is precisely the **singular
stratum** (:ref:`manifold-singular-stratum`), which is the general
statement: :math:`P_H p \in M` iff the orbit of :math:`p` is a single
point.

⭐ For a positive-dimensional group no representative is canonical at
all, which is why every :math:`S^2/O(2)_a` entry carries
``fundamental_domain=None`` on purpose
(:ref:`manifold-err-080-is-a-section`). The barycentre is canonical
*precisely because it is not a representative*: an orbit has exactly one
mean, and a section has as many spellings as there are half-meridians.

.. _manifold-barycentre-equivariance:

Being canonical is exactly what an induced action needs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A section would be *more* than :meth:`Quotient.induced_action
<orpheus.numerics.manifold.Quotient.induced_action>` asks for. What it
asks for is **equivariance**: the arrow :math:`[p] \mapsto [g\,p]` is
well defined only if the point the chart is read off moves with
:math:`g`, and the projector does, whenever :math:`g` normalises
:math:`H`. The proof is one substitution in
:eq:`manifold-reynolds-projector`, using invariance of Haar measure:

.. math::

   g\,P_H\,g^{-1}
   \;=\; \int_H g\,\rho(h)\,g^{-1}\,\mathrm{d}h
   \;=\; \int_{gHg^{-1}} \rho(k)\,\mathrm{d}k
   \;=\; P_{gHg^{-1}} .

So for :math:`g` in the normaliser :math:`gHg^{-1} = H` and
:math:`P_H g = g P_H` — the barycentre of the image orbit **is** the
image of the barycentre — while for :math:`g` outside it the two sides
are projectors onto *different* subspaces and the identity fails by
:math:`O(1)`. That is the same criterion :eq:`manifold-normaliser-descent`
states for the descent itself (:ref:`manifold-normaliser-criterion`),
arrived at from the lift's side, which is why one guard covers both.

`[M]` 2026-09-03, 200 seeded unit vectors, comparing
:math:`P_H(g\,p)` against :math:`g\,(P_H p)` and, draw-free, the
operator gap :math:`\lVert g P_H g^{-1} - P_H\rVert_\infty`:

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 20 18

   * - Entry
     - :math:`g`
     - normalises?
     - :math:`\lVert gP_Hg^{-1}\!-\!P_H\rVert_\infty`
     - :math:`\max\lvert P_H(gp) - g(P_Hp)\rvert`
   * - :math:`S^2/O(2)_x`
     - :math:`\sigma_x,\sigma_y,\sigma_z`
     - ✅
     - ``0.000e+00``
     - ``0.000e+00``
   * - :math:`S^2/O(2)_x`
     - :math:`C_4` about :math:`z`
     - ⛔
     - ``1.000e+00``
     - ``9.918e-01``
   * - :math:`S^2/O(2)_x`
     - a generic rotation
     - ⛔
     - ``4.661e-01``
     - ``6.798e-01``
   * - :math:`S^2/\sigma_y`
     - :math:`\sigma_x,\sigma_y,\sigma_z`, :math:`C_4` about :math:`y`
     - ✅
     - ``0.000e+00``
     - ``0.000e+00``
   * - :math:`S^2/\sigma_y`
     - :math:`C_4` about :math:`z`
     - ⛔
     - ``1.000e+00``
     - ``9.918e-01``

⭐ **Read the operator column, not the pointwise one.** The pointwise
gap is a property of the DRAW — it is the worst of 200 random unit
vectors and moves with the seed — while
:math:`\lVert gP_Hg^{-1} - P_H\rVert_\infty` is a property of the
group pair alone: for a quarter turn taking one coordinate axis to
another it is exactly :math:`1`, because two distinct 0/1 diagonals
differ by :math:`1` in some entry. A gate that pins the seed-dependent
number pins a seed; a gate that pins the operator gap pins the theorem
(``vv-principles`` #31, and ``lessons`` L-071's three-flavours rule).

⚠ **The** :math:`C_4`-about-:math:`y` **row of the mirror entry is the
one that looks wrong and is not.** A quarter turn about :math:`y` maps
:math:`\hat e_y` to itself, so it normalises :math:`\langle\sigma_y
\rangle` and must commute with :math:`P_{\sigma_y}` — which is a
reminder that the *stabiliser of the projector* is strictly larger than
the group being quotiented by, and that a negative leg must be chosen
outside the projector's own stabiliser rather than merely outside
:math:`H` (``vv-principles`` #17's null-control trap).

⟹ **ERR-080, restated in one sentence of this section's vocabulary:
the forgery is** :math:`\beta_a` **with its codomain declared as**
:math:`S^2` **instead of** :math:`D^3`. Everything else about the
computation is right. That is why the defect is invisible to any
arithmetic check and why no tolerance reaches it: the numbers are the
correct barycentres, and what is false is a *type*.

.. list-table:: The two spellings of :math:`\beta_a`, measured on ``gauss_legendre(8)``
   :header-rows: 1
   :widths: 30 35 35

   * -
     - ``invariance._embedded_nodes``
     - ``Quadrature._harmonic_frame_measure`` (1-D arm, retired
       2026-09-02)
   * - what it computes
     - :math:`\mu \mapsto \mu\,\hat e_a`, axis read off the support's
       group
     - :math:`\mu \mapsto (\mu, 0, 0)` by zero-padding
   * - `[M]` are the images equal?
     - **yes** — ``np.array_equal``, both directions
     - **yes**
   * - declared codomain
     - ``Ball(3)`` — via
       :func:`~orpheus.numerics.manifold.barycentre` since 2.3, read off
       :attr:`Quotient.lift
       <orpheus.numerics.manifold.Quotient.lift>` since 2.2b
     - ⛔ ``SPHERE``
   * - `[M]` is the declaration true?
     - ✅ ``Ball(3).contains`` → ``True``
     - ⛔ ``Sphere().contains`` → ``False``; norms
       :math:`0.1834\ldots0.9603`
   * - why it wants the barycentre
     - an invariance check *should* use it: a rotation about :math:`a`
       **fixes** the barycentre, so the point is genuinely
       :math:`SO(2)_a`-invariant
     - it wants a **direction**, which the barycentre is not — so it
       is the wrong map, honestly applied

⭐ **The honest spelling reads the map** (Pattern 2 — one spelling of
one concept), and since 2026-09-02 it reads it through the ENTRY.
``invariance._embedded_nodes`` no longer names :func:`barycentre
<orpheus.numerics.manifold.barycentre>` at all: it asks
:meth:`Quotient.orbit_barycentres
<orpheus.numerics.manifold.Quotient.orbit_barycentres>`, which reads the
entry's own :attr:`~orpheus.numerics.manifold.Quotient.lift_coordinates`
(:ref:`manifold-lift`). `[M]` 2026-09-03 the identity is unmoved by
either re-routing: ``_embedded_nodes`` is ``np.array_equal`` to
``barycentre(measure.support)(measure.nodes)`` on **12 of 12** rows
(``gauss_legendre_on_polar_orbit(n, axis)``,
:math:`n \in \{2,4,8,16\}` × all three axes) on the pre-2.2b tree, on
the 2.2b tree and on this one.

⛔ **The name** ``barycentre`` **is now defined on EVERY entry, and the
refusal it used to carry is gone.** Until 2026-09-03 the function
refused anything that was not an axial orbit space — *"a mirror orbit is
a pair of points with no axis to lie on"* — which was true of the
formula it then implemented and false of the concept: a pair of points
has a mean, and that mean is :eq:`manifold-reynolds-projector` at
:math:`H = \langle\sigma_a\rangle`. `[M]` 2026-09-03 ``barycentre(e)``
answers on all **eight** entries the tree constructs — the six
catalogue keys and the two trivial quotients — and equals ``e.lift`` on
every one; what it still refuses is a manifold that is not a
:class:`~orpheus.numerics.manifold.Quotient` at all —

.. code-block:: text

   the barycentre map is defined on an orbit space M/H; got '[-1,1]',
   whose points are not orbits.

— which is the refusal the old message was *about* all along, with the
family restriction that had been fused into it removed. The function is
:func:`functools.cache`\ d, so `[M]` ``barycentre(e) is barycentre(e)``
is ``True`` while ``e.lift is e.lift`` is ``False`` (the property
assembles a fresh :class:`~orpheus.numerics.manifold.ManifoldMap` each
call, as :attr:`~orpheus.numerics.manifold.Quotient.quotient_map` does);
that memo is what lets a derivation agreement be stated by *identity*
rather than by value.

.. note::

   ⚠ **A brief for this step reported that** :class:`Ball`
   **had zero production consumers before it.** `[M]` that is not what
   the tree says, and the correct statement is sharper. ``Ball(2)`` has
   been production since 2026-08-31 as the :math:`\sigma_y` entry's
   ``realization`` (``manifold.py``, the mirror derivation) and is
   matched in the ambient-dimension table. What had **never** been
   constructed anywhere — `[M]` ``git grep "Ball("`` over ``orpheus/``
   and ``tests/`` at the pre-2.3 commit returns **six** lines — the
   class definition, one ``match`` pattern, and **four** constructions,
   every one of them ``Ball(2)`` — is ``Ball(3)``, and what is new
   in kind is that a :class:`Ball` is now the **codomain of an arrow**
   rather than a field of a catalogue entry. The production docstring
   states the weaker claim; it is reported rather than edited here.


.. _manifold-arrow-composition:

Composition, functoriality, and the fold as a two-arrow chain
---------------------------------------------------------------

Composition is ``psi @ phi`` for :math:`\psi \circ \varphi`, refused
unless ``phi.codomain == psi.domain`` — the same guard, at the same
tier, as an operator product over unequal spaces:

.. code-block:: text

   ValueError: cannot compose: the inner map lands on 'S^2' but the
   outer map is defined on '[-1,1] × S^1'. A composition psi @ phi
   needs phi.codomain == psi.domain.

The one law worth stating — and, for a map of finite point sets whose
membership :meth:`~orpheus.numerics.manifold.Manifold.contains`
already governs, essentially the only intrinsic thing such a map can
get wrong — is that the pushforward is a **functor**:

.. math::
   :label: manifold-map-functoriality

   (\psi \circ \varphi)_* \mu \;=\; \psi_*\bigl(\varphi_* \mu\bigr).

.. (vv-status rationale) manifold-map-functoriality: A structural law
   of the arrow type, not a solver claim, so it carries no L0..L3
   ladder slot and no ``verifies(...)`` marker. It is gated by the
   ``foundation`` test
   ``tests/numerics/test_manifold.py::TestManifoldMap::test_functoriality_the_pushforward_of_a_composite_is_the_composite_of_pushforwards``
   (nodes and weights by ``np.array_equal``, and the support by
   value), and re-measured on production objects by the shipped-fold
   table in this section.
.. vv-status: manifold-map-functoriality documented

⭐ **This is not an abstract law here — the shipped cylindrical fold is
literally a chain of two of these arrows.**
:meth:`Quadrature.folded_product
<orpheus.numerics.quadrature.directional.Quadrature.folded_product>`
builds a product rule and then folds it, and after 2.3 both halves are
typed maps:

.. math::

   [-1,1] \times S^1
   \;\xrightarrow{\ \varphi_z\ }\; S^2
   \;\xrightarrow{\ \rho\ }\; S^2/\langle\sigma_y\rangle ,

with :math:`\rho` the orbit retraction that
:meth:`DiscreteMeasure.quotient
<orpheus.numerics.measure.DiscreteMeasure.quotient>` builds from the
invariance certificate. Because :math:`\varphi_z` lands on
:math:`S^2` and :math:`\rho` is defined there, ``rho @ chart``
type-checks, and :eq:`manifold-map-functoriality` says the one-shot
route must agree with the two-step one.

`[M]` 2026-09-02, measured — the composite built with ``@``, pushed in
one step and consolidated, against the two-step route, against the
shipped rule:

.. list-table:: The fold, three ways
   :header-rows: 1
   :widths: 18 14 24 22 22

   * - :math:`(n_\mu, n_\varphi)`
     - :math:`N` after the fold
     - one-shot ``ρ @ φ`` vs two-step
     - vs shipped ``folded_product``
     - ``support``
   * - (2, 8)
     - 8
     - ``array_equal`` ✅
     - ``array_equal`` ✅
     - ``'S^2/sigma_y'``, by **identity** with the catalogue entry
   * - (4, 8)
     - 16
     - ✅
     - ✅
     - ✅
   * - (4, 16)
     - 32
     - ✅
     - ✅
     - ✅
   * - (6, 10)
     - 30
     - ✅
     - ✅
     - ✅
   * - (3, 24)
     - 36
     - ✅
     - ✅
     - ✅

⚠ **Read the fixture, not just the ticks.** The reconstruction uses the
**staggered** circle rule, which is what ``folded_product`` selects
(:math:`\Sigma = \varnothing`, the fold's well-posedness condition);
with the node-aligned shift the same code agrees with itself on both
routes but does *not* reproduce the shipped rule, because it is then a
different rule. `[M]` at :math:`(2, 8)`: node-aligned puts **4** nodes
on :math:`\Sigma = \{\xi = 0\}`, so its 16 atoms fold into **10**
orbits — sizes ``[1,1,1,1,2,2,2,2,2,2]``, the four singletons being
the fixed points — while staggered puts **0** there and folds into
**8**, all of size 2. The functoriality half is fixture-independent;
the third column is a statement about which circle rule ships, and the
singleton orbits are exactly the well-posedness condition the fold's
own :math:`\Sigma = \varnothing` requirement names.


.. _manifold-arrows-not-built:

What 2.3 did NOT build
------------------------

Three things, stated so the next phase does not re-derive a decision
already taken and so no reader mistakes an arrow for a repair.

**(1) Neither of the catalogue entry's own two maps ships.**

.. note::

   ⛔ **Half REMEDIED 2026-09-02 by tracker 3.1 — kept in place because
   the enumeration below is what made the remedy possible.** The
   entry's own map now ships, as
   :attr:`Quotient.quotient_map
   <orpheus.numerics.manifold.Quotient.quotient_map>` over the stored
   :attr:`~orpheus.numerics.manifold.Quotient.orbit_coordinates`
   (:ref:`manifold-quotient-map`) — and the paragraph below is *why*
   3.1 could not simply promote one of 2.3's three arrows: each is
   structurally the wrong map, for three different reasons, so the
   entry needed a fourth. ⭐ It also mis-named it: what ships is the
   **quotient map**, not a *chart* (a chart is injective and this is
   not), which is the ruling tracker 2.3 had already made about the
   type's own name.

   ⟹ **The SECTION half stands entirely unchanged.** `[M]` 2026-09-02
   ``fundamental_domain`` is still ``None`` on every
   :math:`S^2/O(2)_a` entry, still has **zero** production readers
   outside :mod:`orpheus.numerics.manifold`, and 3.1 declined it for
   the same reason 2.3 did: a section is a **choice**, and a quotient
   map is a derivation *output*
   (:ref:`manifold-the-axis-convention-for-a-section`).

The :ref:`engine data model <manifold-engine-data-model>` lists the
entry's *chart* :math:`M/H \to N` and its *section* as procedure
outputs that are not slots, and 2.3 does not change that — it changes
only whether such a thing could be *expressed*. None of the three
arrows above is either of them:

.. list-table::
   :header-rows: 1
   :widths: 30 32 38

   * - Arrow
     - Type
     - Why it is not the entry's chart or section
   * - :math:`\varphi_a`, ``archimedes``
     - :math:`[-1,1]\times S^1 \to S^2`
     - a parametrisation of the **base**, not a chart of the orbit
       space. Its first component *is* the chart, in the sense
       :math:`\pi \circ \varphi_a = \mathrm{pr}_1` measured above, but
       the chart :math:`\pi : S^2 \to [-1,1]` itself is still not a
       value anywhere.
   * - :math:`\rho`, the retraction
     - :math:`M \to M/H`
     - built **per measure** from an invariance certificate, so it
       depends on the atoms and not only on :math:`(M, H)` — it cannot
       be a field of an entry. Its image stays in the base's
       coordinates, which is the section's coordinate system, but it
       is not a section: it is the quotient map with a chosen
       representative per *realized* orbit.
   * - :math:`\beta_a`, ``barycentre``
     - :math:`S^2/O(2)_a \to D^3`
     - a map **out of** the orbit space, landing off :math:`S^2`. A
       section is a map **into** the base. See above.

⟹ ``fundamental_domain=None`` on every :math:`S^2/O(2)_a` entry is
still the honest answer, and :attr:`Quotient.fundamental_domain
<orpheus.numerics.manifold.Quotient.fundamental_domain>` still has
`[M]` **zero production readers**. The section remains a *choice*
(:ref:`manifold-the-axis-convention-for-a-section`), and 2.3 declined
to make it.

⭐ **What 3.1 shipped instead, and why the table above is the argument
for it.** The entry's arrow is a *fourth* map — a map **into** the
orbit space, out of the base, depending only on :math:`(M, H)`. Read
the three rows as a set of exclusions and that is exactly the gap they
leave: :math:`\varphi_a` is out of a product and into the base;
:math:`\rho` is into the orbit space but built per *measure*;
:math:`\beta_a` is out of the orbit space. Only the fourth is a field
of an entry, and it is the one the derivation already computes
(:ref:`manifold-quotient-map`).

**(2) The pushforward reference measure is deferred to tracker 3.1 —
and the reason is a measured import cycle.**

.. note::

   ✅ **DISCHARGED 2026-09-02 by tracker 3.1**, and the deferral's
   *reason* survives its own discharge, which is why this item is kept
   rather than deleted. The cycle it measured is real and unchanged;
   what the item got wrong is that it read a cycle blocking one
   **mechanism** as a cycle blocking the **slot**. The shipped answer
   splits the need in two — the *type* under
   :data:`typing.TYPE_CHECKING`, the *value* through a function-scope
   import inside the derivation function, which is the very idiom the
   last sentence below prescribes.

   `[M]` 2026-09-02, on a **renamed shadow copy of the real package**
   (not a throwaway three-module one) so the editable install cannot
   serve the production tree by accident, over **seven** entry points:
   the shipped function-scope import is alive on **7 of 7**; the same
   import at the *top* of the module dies on **7 of 7**
   (``ImportError: cannot import name 'Manifold'``); and at the
   *bottom*, the most favourable module-scope position there is, it
   dies on **7 of 7** as well, one hop further along
   (``ImportError: cannot import name 'DiscreteMeasure'``). The full
   table, and why this cycle is *not* order-dependent, is at
   :ref:`manifold-value-at-function-scope`.

   ⚠ The 5-of-5 figure below is a different measurement of a different
   edge — ``manifold → exactness``, on a throwaway package — and it
   stands as written. 3.1's shipped runtime edge is
   ``manifold → generating_measure`` (the module that owns the
   ``LEGENDRE`` **value**); ``exactness`` supplies only the *type*, and
   that one really is carried by the ``TYPE_CHECKING`` guard.

An orbit space's
pushforward reference (the :math:`2\pi\,d\mu` of the hat-box) is a
field of the catalogue **entry**, not of the map, because it is a
property of :math:`(M, H)` rather than of any one arrow. ⛔ This
sentence continued *"it is answered today by a twin on the registry —*
:attr:`AngularSymmetry.reference
<orpheus.numerics.quadrature.registry.AngularSymmetry.reference>`\ *,*
``LEGENDRE`` *for any axial rotation and* ``UNIFORM_ON_SPHERE`` *for
the trivial group — and collapsing that twin onto the entry is the same
move tracker 2.4 made for* ``support``\ *"* until 2026-09-02. That is
now history: the twin **is** collapsed, in exactly that direction —
`[M]` the registry's slab answer *is* the entry's field, by ``is``
identity, and its ``LEGENDRE`` import is gone. What survived the
collapse is the arm the prediction did **not** anticipate, and the
asymmetry is instructive: the ``UNIFORM_ON_SPHERE`` half is *not* a
catalogue read and was user-ruled to stay one
(:ref:`manifold-twin-lookup`).

⛔ It cannot be done by adding an import. `[M]` 2026-09-02 by AST with
relative imports resolved, :mod:`orpheus.numerics.exactness` imports
:mod:`orpheus.numerics.manifold` at **module scope, twice** — once for
:class:`~orpheus.numerics.manifold.Manifold` and once for
:data:`~orpheus.numerics.manifold.CIRCLE` and
:data:`~orpheus.numerics.manifold.SPHERE`. A module-scope
``manifold → exactness`` edge therefore closes a **two-hop** cycle,
with no import order that survives it: demonstrated on a throwaway
package carrying exactly that topology — no production file touched —
**5 of 5** entry points die with

.. code-block:: text

   ImportError: cannot import name 'Manifold' from partially
   initialized module 'pkg.manifold' (most likely due to a circular
   import)

and **5 of 5** import cleanly with the ``TYPE_CHECKING`` guard
restored (the positive control, without which a clean reading carries
no information). ⟹ the viable mechanism is a
:class:`~orpheus.numerics.manifold.Quotient` field populated **inside**
the derivation function through a function-scope import — the idiom
``_sphere_mod_o2`` already uses for SymPy — never at module scope.
This is the same guard as the ``manifold → exactness`` row of
:ref:`manifold-import-cycle`, on a second pair of modules — and it is the
guard that OUTLIVED the one that section used to be named after.

✅ **That prediction held verbatim, and it is the rare kind that names
its own mechanism rather than its phase.** `[M]` 2026-09-02
``manifold.py:1194`` (``:1679`` after #434 R1 and R4) is
``from orpheus.numerics.generating_measure import LEGENDRE``, at
function scope inside ``_sphere_mod_o2``, three lines below the
``import sympy as sp`` it was modelled on. ⚠ Note what it got *right*
by not being specific: it prescribed a function-scope import without
naming which module, and the module that shipped is
``generating_measure`` rather than ``exactness`` — because ``LEGENDRE``
is a **value** and ``exactness`` owns only the *type*
(:ref:`manifold-value-at-function-scope`).

**(3) ERR-080 is not repaired, and 2.3 moves neither of its gates.**
Nothing here calls
:meth:`~orpheus.numerics.manifold.Manifold.contains` on the way into a
measure, so the forged :math:`(\mu, 0, 0)` measure is still
constructible; the forgery arm is still a raw
:class:`~orpheus.numerics.measure.DiscreteMeasure` constructor by
design; and the level-2 half — the trivial isotypic sub-basis
:math:`\{Y_\ell^0\} \cong \{P_\ell\}` — is untouched. `[M]` by AST
over ``tests/sn/solve/test_pl_order_does_not_move_the_infinite_medium_flux.py``
the module still declares **three** ``@pytest.mark.xfail(strict=True)``
rows and 2.3 edits none of them. What 2.3 buys ERR-080 is a
*sentence*: the defect now has a name in the type system's own
vocabulary — :math:`\beta_a` with a forged codomain — and one honest
implementation of that map to point at.


.. _manifold-quotient-map:

The entry's OWN arrow: the quotient map :math:`\pi : M \to M/H`
----------------------------------------------------------------

Tracker 2.3 gave the *category* its arrows. Every one of its three maps,
though, is a map drawn **around** the quotient rather than by it: a
parametrisation of the base, a retraction built per *measure*, and a map
**out of** the orbit space. The one arrow the entry itself owns —
:math:`\pi`, the map that *makes* the orbit space — was the row the
:ref:`engine data model <manifold-engine-data-model>` marked ⛔ *not a
slot*. Tracker 3.1 (2026-09-02) fills it — and, beside it, the
**pushforward measure**, the other output the same table marked ⛔. With
those two the seed is complete at
:ref:`9 of 9 <manifold-engine-data-model>`.

The map is not new mathematics. It is the invariant tuple, read as a
function of a point of the base — **the invariants that survive
eliminating the base's own ideal**:

.. math::

   \pi_a(\Omega) = \Omega\cdot\hat e_a = p_1
   \qquad &\text{on } S^2/O(2)_a, \\
   \pi_a(\Omega) = (x_b,\, x_c) = (p_1,\, p_2)
   \qquad &\text{on } S^2/\langle\sigma_a\rangle, \\
   \pi(x) = x
   \qquad &\text{on } M/\{e\}.

For :math:`S^2/O(2)_a` the derivation's minimal invariants are
:math:`p_1 = x_a` and :math:`p_2 = x_b^2 + x_c^2`, and the sphere's ideal
:math:`p_1^2 + p_2 = 1` eliminates the second — so **one** coordinate
survives and the orbit space is 1-dimensional
(:eq:`manifold-s2-mod-so2`). For :math:`S^2/\langle\sigma_a\rangle` the
invariants are :math:`x_b`, :math:`x_c`, :math:`x_a^2`, and the same
ideal eliminates the *third* — so **two** survive and the dimension does
not drop (:eq:`manifold-s2-mod-mirror`). In both cases the surviving
invariants happen to be coordinate functions, which is why the shipped
:attr:`~orpheus.numerics.manifold.Quotient.orbit_coordinates` is a
**column selection**; an entry whose *surviving* invariants included a
higher-degree polynomial — :math:`S^2/C_n` about an axis needs one of
degree :math:`n`, :math:`\mathrm{Re}\,(x_b + i x_c)^n`, since a rotation
by :math:`2\pi/n` multiplies :math:`x_b + i x_c` by an :math:`n`-th root
of unity — would carry a genuine polynomial map in the same slot, and
nothing about the field's type would have to change.

⭐ **The codomain is the ENTRY, never the** ``realization`` **(user
ruling, 2026-09-02),** and that is the whole reason this is a slot rather
than a convenience. Read as :math:`\pi : S^2 \to [-1,1]` the map lands on
the *chart's codomain* — precisely the axis-blind reading tracker 2.4 made
refusable, since all three :math:`S^2/O(2)_a` realize onto the *same*
interval and a rule on :math:`[-1,1]` is not a rule on
:math:`S^2/O(2)_x` (:ref:`manifold-so2-axis-is-a-parameter`). Read as
:math:`\pi : S^2 \to S^2/O(2)_x` it lands on the orbit space, carrying
the axis and the spent group with it — which is what makes the pushforward
of a rule along it a rule *on an orbit space* rather than a bag of numbers
on an interval.

⚠ **It is not a chart, and the corpus already ruled why.** A chart is
:math:`M \supset U \to \mathbb{R}^n`, and :math:`\Omega \mapsto
\Omega\cdot\hat e_a` is not injective on :math:`S^2` — a whole orbit maps
to one value, which is the *point* of a quotient map. Only the **inverse**
of the Archimedes parametrisation is a chart in the strict sense
(:ref:`manifold-archimedes`). That ruling is what named the type
:class:`~orpheus.numerics.manifold.ManifoldMap` rather than ``Chart``
at tracker 2.3, and it is why the field here is called
``orbit_coordinates``.


Stored ``apply``, derived arrow — the 2.1b pattern, forced
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The entry stores the map's **action** and derives the typed arrow:

.. code-block:: python

   orbit_coordinates: Callable[[NDArray], NDArray] = field(
       compare=False, repr=False
   )

   @property
   def quotient_map(self) -> ManifoldMap:
       return ManifoldMap(
           domain=self.base, codomain=self, apply=self.orbit_coordinates
       )

This is the shape
:attr:`Basis.invariance_group
<orpheus.numerics.basis.base.Basis.invariance_group>` took one step
earlier (:ref:`manifold-basis-invariance-group`) — *derive what the
fields already determine* — but here it is not a preference, it is
**forced**. An arrow whose ``codomain`` is the instance being
constructed cannot be an ordinary field of it: the arrow needs ``self``,
and ``self`` is not complete until the last field is assigned. Storing it
would mean either reaching around the type's own frozenness with an
``object.__setattr__`` in ``__post_init__``, or carrying a second object
that can silently disagree with ``base`` and ``orbit_coordinates``.
Deriving it makes both unspellable, and costs nothing: the property is a
three-argument constructor call over fields the entry already holds.

⭐ **Why** ``compare=False`` **— and the measurement is sharper than the
docstring's reason.** A function has no value equality, which is the
stated reason and is true. The consequence worth publishing is what the
exclusion *buys*: `[M]` 2026-09-02,
``pickle.loads(pickle.dumps(q)) == q`` is **True on 7 of 7** shipped
quotients of :math:`S^2`, precisely *because* the callable is excluded
from ``__eq__``.

⛔ **A** ``functools.partial`` **is picklable but does NOT round-trip
equal**, and the two claims are easy to conflate — the first is what
the spelling was chosen for and it holds; the second does not. `[M]`
over the same seven: the callable survives ``pickle`` with no
``PicklingError`` — which a ``lambda`` *would* raise — and its output
is bit-identical to the original's, **7 of 7**. But it compares equal
to the original only **1 of 7**. :func:`functools.partial` inherits
``object.__eq__``, so every axial and mirror entry's
``partial(_ambient_columns, …)`` round-trips **unequal**; only the
trivial entry's plain module-level ``_all_coordinates`` compares equal,
and for a reason that does not generalise — pickling a function stores
it *by reference*, so unpickling returns the **same object**.

⟹ a :class:`~orpheus.numerics.manifold.Quotient` that compared its
``orbit_coordinates`` would fail to round-trip on 6 of 7 entries, and
the entry is memoised into a cache and used as a dictionary key
(:ref:`manifold-quotient-is-memoised`). The exclusion is load-bearing
for serialisation, not merely tidy — which is a stronger argument for
it than *"a function has no value equality"*, and one a future reader
can falsify in three lines.


Four laws, all measured, and the negative leg on each
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A quotient map is over-determined by the objects already on the page, so
every one of its laws is checkable against something the entry did not
produce. All four are `[M]` 2026-09-02 on the working tree.

.. list-table:: The quotient map's laws
   :header-rows: 1
   :widths: 22 40 38

   * - Law
     - What it says
     - Measured
   * - :math:`H`-invariance
     - :math:`\pi(h\cdot\Omega) = \pi(\Omega)` for every
       :math:`h \in H` — the defining property, and the only one that
       is about :math:`H` at all
     - bit-exact on **7 of 7** shipped quotients of :math:`S^2` (the
       six catalogue keys plus the derived identity), 1000 random
       directions each — rotations by 0.3 / 1.7 / 4.1 rad about the
       axis for each ``O2(a)``, the reflection for each ``Mirror(a)``,
       the identity for ``Trivial``. **Negative leg on 7 of 7:** a
       rotation about a *different* axis moves the image. ⚠ Since #432
       the axial rows' generic set is TWICE that size — `[M]`
       ``O2('x').generic_images`` returns **12** images (each of the six
       incommensurate rotations, and each composed with one vertical
       mirror) against ``SO2('x')``'s **6** — because the second
       component must be sampled too. The law is unchanged: a vertical
       mirror fixes :math:`\pi_a` for the same reason it fixes the
       orbits (:ref:`manifold-orbit-space-stabiliser`).
   * - :math:`\pi_a \circ \varphi_a = \mathrm{pr}_1`
     - the polar factor of a product rule **is** the orbit-space
       coordinate — the chart and the parametrisation agree
       (:ref:`manifold-archimedes`)
     - bit-exact on **12 of 12** — three axes × Gauss orders
       :math:`n \in \{2,4,8,16\}`, random azimuths — composed through
       the typed ``@``, so the composition's own endpoint guard is
       exercised as well
   * - :math:`\beta_a \circ \pi_a` is the axial projection
     - the barycentre of the orbit through :math:`\Omega` is
       :math:`(\Omega\cdot\hat e_a)\,\hat e_a`
       (:ref:`manifold-barycentre`)
     - bit-exact on **3 of 3** axes, 1000 random directions, with
       ``Q.contains(π(v))`` ``True``; :math:`\beta_a`'s codomain reads
       ``D^3``, so the chain :math:`S^2 \to S^2/O(2)_a \to D^3` type-checks
       end to end
   * - the pushforward identity
     - :math:`\int f\,d(\pi_*\mu) = \int (f\circ\pi)\,d\mu` — the
       change of variables the whole slot exists for
     - on ``level_symmetric(4)`` pushed along :math:`\pi_x`:
       ``support is`` the catalogue entry, nodes ``array_equal`` to
       the rule's :math:`\mu_x` column, weights unchanged, and
       :math:`\int \mu^2 \, d(\pi_*\mu) = \int(\Omega\cdot\hat
       e_x)^2\,d\mu` **bit-exact** at ``4.18879020478639`` — `[M]`
       **1 ULP** from :math:`4\pi/3`

⭐ **And the refusal is the fourth law's other half.** `[M]` a rule on
:math:`[-1,1]` handed to :math:`\pi_x` is REFUSED, because
:meth:`~orpheus.numerics.measure.DiscreteMeasure.pushforward` compares the
map's ``domain`` against the measure's ``support`` by manifold *value*:

.. code-block:: text

   ValueError: cannot push a measure on '[-1,1]' forward along a map
   out of 'S^2': the map's domain must be the measure's support.

That is the 2.3 guard doing exactly the work 3.1 needs. Without it the
quotient map would be a callable anyone could point at any array, which
is the shape :ref:`ERR-080 <manifold-err-080>` has.

.. warning::

   ⛔ **The map is a CAPABILITY, not a repair — and 3.1's two halves
   have OPPOSITE consumption status, which is easy to read past.**
   `[M]` 2026-09-02 over ``orpheus/``:
   :attr:`~orpheus.numerics.manifold.Quotient.reference` has **one**
   production reader (``registry.py``, and the collapse is
   :ref:`measured there <manifold-second-twin-reference>`), while
   ``quotient_map`` and ``orbit_coordinates`` have **zero** outside
   :mod:`orpheus.numerics.manifold` itself — their only consumers are
   in ``tests/numerics/test_manifold.py``, where `[M]` ``quotient_map``
   occurs **ten** times and ``orbit_coordinates`` three. Nothing in
   production pushes a measure along :math:`\pi` yet.

   ⟹ **ERR-080 is unchanged by any of this.** No membership check runs
   on the way into a measure, the forgery arm is still a raw
   :class:`~orpheus.numerics.measure.DiscreteMeasure` constructor by
   design, and `[M]` its gate still declares **three**
   ``xfail(strict=True)`` rows. What the entry now owns is the honest
   map the forgery is a forgery *of* the other side of
   (:ref:`manifold-barycentre`) — a reader who meets the refusal
   predicate without this clause will conclude the defect is repaired.


The numeric map IS the recorded symbolic invariants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The engine ruling (:ref:`manifold-engine-data-model`) says an entry's
fields must *be* the derivation's output. A stored callable is the field
where that is easiest to violate: nothing about
``partial(_ambient_columns, 0)`` looks like
:attr:`~orpheus.numerics.manifold.Quotient.generators`, and a hand entry
could pick the wrong column without any other field noticing.

So the tie is measured, not asserted: `[M]` 2026-09-02, on **7 of 7**
shipped quotients of :math:`S^2` the numeric ``orbit_coordinates``
agrees **bit-exactly** with the column selection the entry's own
recorded generators name — ``O2_x/y/z`` → column 0 / 1 / 2;
``sigma_x/y/z`` → the two columns that are *not* the mirror axis,
:math:`(1,2)`, :math:`(0,2)`, :math:`(0,1)`; ``Trivial`` → all three.
The committed gate does the stronger thing and ``lambdify``\ s the
surviving generators directly, which is what an engine would do; the
hand entry spells the columns. The two must agree, and the test is the
specification of that agreement — one more row in the acceptance suite
that is :ref:`written before the engine <manifold-tests-are-the-spec>`.

.. note::

   ⚠ **The map's own stabiliser is BIGGER than** :math:`H` **for the
   axial family, so** :math:`H`\ **-invariance cannot recover** ``by``
   **— it is a declaration, not a computed stabiliser.** `[M]`
   2026-09-02, bit-exactly on all three axes: :math:`\pi_a` is
   unchanged under the mirror :math:`\sigma_b` for
   :math:`b \ne a`, and :math:`\sigma_b \notin SO(2)_a`. The reason is
   not a bug and is worth carrying: a reflection in a plane
   *containing* the axis maps each constant-\ :math:`\mu` circle to
   itself, so :math:`O(2)_a` and :math:`SO(2)_a` induce the **same
   orbit partition** of :math:`S^2` — and therefore the same orbit
   space, the same invariants and the same map.

   ⟹ a quotient map determines the **partition**, and the partition
   does not determine the group. The entry *declares* which group it
   quotients by, :attr:`Quotient.name
   <orpheus.numerics.manifold.Quotient.name>` spells it, and
   :attr:`Basis.invariance_group
   <orpheus.numerics.basis.base.Basis.invariance_group>` reads it —
   none of them derives it. ⭐ The mirror family is the contrast that
   makes the point checkable rather than decorative: `[M]` there the
   stabiliser is exactly :math:`\langle\sigma_a\rangle`, and
   :math:`\sigma_x` genuinely moves :math:`\pi_y`'s image. ⭐ And the
   shape is one the corpus already knows in a *weaker* form: ERR-072
   is a group predicate that under-determines its group because it was
   **sampled** (:ref:`manifold-so2-axis-lattice`). This one
   under-determines it while being **exact** — no refinement of the
   check can fix it, because :math:`SO(2)_a` and :math:`O(2)_a` are
   genuinely indistinguishable by their orbits on :math:`S^2`.

   ✅ **And that is exactly why the DECLARATION is pinned to the
   maximal group** (2026-09-02, #432). Written the day before, this
   note read as a caveat: two groups are indistinguishable by the map,
   so ``by`` is whatever the entry says. Read once more, it is the
   argument for a rule — *if the map cannot tell them apart, do not let
   the catalogue offer two names for one point set* — and the rule is
   the naming law: an orbit space is named by its **stabiliser**, the
   type refuses a non-maximal ``by`` at construction, and
   :math:`S^2/SO(2)_a` is refused at the catalogue door naming
   :math:`O(2)_a` (:ref:`manifold-orbit-space-stabiliser`). Every
   measurement above stands unchanged; what moved is the conclusion
   drawn from it. ⛔ This note ended at *"genuinely indistinguishable by
   their orbits"* until 2026-09-02, and a reader who stopped there would
   conclude that ``by`` is free — it is not; it is **determined**, by
   the orbit partition plus maximality.


Its image is in the CHART's coordinates; the retraction's is in the SECTION's
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two maps in the tree now land on the same codomain and emit **different
numbers**, and both are right. This is the :ref:`two-coordinate-systems
ruling <manifold-two-coordinate-systems>` seen from the arrow side, and
it is the thing to hold on to when reading either.

.. list-table::
   :header-rows: 1
   :widths: 24 26 26 24

   * - Map
     - Codomain
     - Coordinates of the image
     - Width
   * - ``Quotient.quotient_map``
     - the entry
     - the ``realization``'s — the **invariants**
     - ``ambient_dim(realization)`` (1 for :math:`S^2/O(2)_a`,
       2 for the disk)
   * - the retraction inside
       :meth:`DiscreteMeasure.quotient
       <orpheus.numerics.measure.DiscreteMeasure.quotient>`
     - the entry (the same object, by identity)
     - the ``fundamental_domain``'s — a **representative** in the
       base
     - the base's ambient width (3 on :math:`S^2`)

:meth:`Quotient.contains
<orpheus.numerics.manifold.Quotient.contains>` accepts **both**, and
dispatches on the ambient width — which is why it is deliberately wider
than :func:`~orpheus.numerics.manifold.ambient_dim`, which reports the
chart's. A design that normalised the two to one language would have to
pick, and picking is what ERR-080 did: the tree needed a *section*, had a
*chart*, and fabricated the missing one by zero-padding
(:ref:`manifold-err-080-is-a-section`).

⚠ **The two are not interchangeable even where their widths agree.** On
:math:`M/\{e\}` the realization *is* the base and the fundamental domain
is all of it, so both languages have width 3 and
``contains`` never needs to dispatch. On :math:`S^2/\langle\sigma_a\rangle`
they are 2 and 3 and it does. On :math:`S^2/O(2)_a` there is only one
language, because `[M]` ``fundamental_domain`` is ``None`` on every axial
entry and stays so — a section is a *choice*, not a derivation output
(:ref:`manifold-the-axis-convention-for-a-section`).


.. _manifold-pushforward-reference:

The pushforward reference: Archimedes' hat-box as a catalogue field
--------------------------------------------------------------------

A degree of exactness is meaningless on its own: it is an **index into
the orthogonal system of a measure**, so the same integer means different
things against different measures — a rule can agree on space, on
orthogonal system *and* on degree while integrating the wrong thing
(:doc:`/theory/foundations/discrete_measures` measures the gap at
**0.696** on :math:`\int_{-1}^1 x^6`). The measure a degree on an orbit
space is *against* is therefore a fact about the orbit space, and it has
exactly one honest definition: the **pushforward** of the base's own
Lebesgue measure along the quotient map.

.. math::
   :label: manifold-quotient-pushforward

   \underbrace{(\pi_a)_*\,d\Omega \;=\; 2\pi \, d\mu}
     _{\text{on } S^2/O(2)_a \,\cong\, [-1,1]}
   \qquad\qquad
   \underbrace{(\pi_a)_*\,d\Omega \;=\;
     \frac{2 \, dx_b \, dx_c}{\sqrt{1 - x_b^2 - x_c^2}}}
     _{\text{on } S^2/\langle\sigma_a\rangle \,\cong\, D^2}

.. (vv-status rationale) manifold-quotient-pushforward: A derivation
   output of the orbit-space procedure — the image of the base's
   Lebesgue measure under the entry's own quotient map — and therefore
   a statement about a catalogue entry's field, not a solver claim, so
   it carries no L0..L3 ladder slot and no ``verifies(...)`` marker.
   The axial half is what ``Quotient.reference = LEGENDRE`` encodes and
   is gated by the ``foundation`` module ``tests/numerics/test_manifold.py``
   (the pushforward-identity rows, which check the image measure's
   support, nodes and weights against the rule they came from) together
   with ``tests/numerics/test_registry.py`` (the registry's answer read
   off the entry by identity). The mirror half is gated by its
   CONSEQUENCE — the ``NotImplementedError`` the registry raises for a
   spent mirror, whose message names this measure as the missing work.
.. vv-status: manifold-quotient-pushforward documented

**The axial half, derived.** Parametrise :math:`S^2` by
:math:`(\mu, \varphi)` through the Archimedes map
(:ref:`manifold-archimedes`); the surface measure is
:math:`d\Omega = d\mu \, d\varphi`, with no Jacobian factor — that
*is* Archimedes' hat-box theorem, the statement that the sphere and its
circumscribed cylinder have the same area element under the axial
projection. The quotient map keeps :math:`\mu` and forgets
:math:`\varphi`, so pushing forward integrates the fibre out:

.. math::

   (\pi_a)_*\,d\Omega
   \;=\; \Bigl(\int_0^{2\pi} d\varphi\Bigr)\, d\mu
   \;=\; 2\pi\,d\mu .

⟹ the image is **uniform in the invariant**. A degree of exactness on
:math:`S^2/O(2)_a` is therefore a degree against *Lebesgue measure on*
:math:`[-1,1]`, up to a constant no exactness claim carries — which is
:data:`~orpheus.numerics.generating_measure.LEGENDRE`. `[M]` its mass is
exactly ``2.0`` and its orthogonal system is ``ALGEBRAIC``; the name
records the polynomial family the measure *generates*, not a weighting,
and its weight is :math:`w(x) = 1`. The hat-box constant :math:`2\pi` is
no claim's business. Total mass checks out on both sides:
:math:`2\pi \cdot 2 = 4\pi`, the area of :math:`S^2`.

**The mirror half, derived — and it is why that entry ships**
``None``. Write the disk's coordinates as :math:`(u, v) = (x_b, x_c)`
and put :math:`f(u,v) = \sqrt{1 - u^2 - v^2}`, so that each hemisphere of
:math:`S^2` is the graph :math:`x_a = \pm f(u,v)` and its area element is

.. math::

   \sqrt{1 + f_u^2 + f_v^2}\;du\,dv
   \;=\; \sqrt{1 + \frac{u^2 + v^2}{1 - u^2 - v^2}}\;du\,dv
   \;=\; \frac{du\,dv}{\sqrt{1 - u^2 - v^2}} ,

with :math:`f_u = \partial f/\partial u = -u/f` and
:math:`f_v = -v/f`. And :math:`\pi_a` identifies the two hemispheres, so
the pushforward carries **twice** that.

`[M]` 2026-09-02, re-derived symbolically and mass-checked:
:math:`\int_{D^2} 2(1-r^2)^{-1/2}\, r\,dr\,d\theta = 4\pi` exactly, the
area of :math:`S^2` again. And measured on a real rule — ``lebedev(11)``
pushed along :math:`\pi_y` — the image carries total weight
``12.566370614359172``, bit-identical to ``4*np.pi``, with
:math:`\int p_1^2\,d(\pi_*\mu) = \int(\Omega\cdot\hat e_x)^2\,d\mu`
bit-exact. ⚠ Read the image as what
:meth:`~orpheus.numerics.measure.DiscreteMeasure.pushforward` promises:
the :math:`\varphi`-image with weights preserved verbatim and **no**
consolidation, so `[M]` its 50 atoms occupy only **29** distinct points
of the disk — a mirror-symmetric rule folds two-to-one off the fixed
plane. Consolidating orbits is
:meth:`~orpheus.numerics.measure.DiscreteMeasure.quotient`'s job, and
it is a different verb (:ref:`manifold-arrow-composition`).

That measure is a perfectly good measure. What it is **not** is a
:class:`~orpheus.numerics.exactness.ReferenceMeasure` any shipped
realization spells: its weight :math:`(1-u^2-v^2)^{-1/2}` is a genuine
Jacobian on a 2-dimensional domain, which is neither a
``UniformMeasure`` nor any 1-D three-term recurrence. ⟹
``reference=None`` on all three :math:`\sigma_a` entries, **user-ruled
2026-09-02**, and the registry's refusal names the missing *work* rather
than the gap (:ref:`manifold-refusal-names-the-work`):

.. code-block:: text

   NotImplementedError: the catalogue entry for S^2/sigma_y carries no
   exactness reference: no shipped ReferenceMeasure realization spells
   the pushforward of dOmega along its quotient map. Add one to
   orpheus/numerics/exactness.py and populate `reference` in the entry's
   derivation in orpheus/numerics/manifold.py.

`[M]` re-measured 2026-09-03, that refusal is reachable and was
exercised: ``AngularSymmetry(spent=Mirror('y'), unspent=Trivial,
owed=Trivial)`` answers ``support`` fine — ``'S^2/sigma_y'`` — and raises
on ``reference``. (The slot was called ``continuous_isotropy`` and the
ledger had two entries until R3 of #434 renamed it and added a third;
the witness is unchanged, and a reader following the old spelling gets a
``TypeError``.) It bites no
shipped geometry, because no geometry spends a mirror
(:ref:`manifold-twin-lookup`, reading (iii)); it is the *witness* that
the ``None`` is an answer rather than an omission.

**The second honest** ``None`` **is** :math:`M/\{e\}`, and its reason is
different in kind. The pushforward of Lebesgue along the *identity* is
Lebesgue **on the base** — a perfectly spellable measure, and for
:math:`S^2` the tree already ships it as
:data:`~orpheus.numerics.exactness.UNIFORM_ON_SPHERE`. What the generic
derivation cannot do is *name* it: the orthogonal system that Lebesgue on
:math:`M` indexes — spherical harmonics on :math:`S^2`, Fourier on
:math:`S^1`, polynomials on an interval — is a property of the **base**,
and :class:`~orpheus.numerics.manifold.Manifold` does not carry one.
:meth:`Manifold.quotient
<orpheus.numerics.manifold.Manifold.quotient>` accepts the trivial group
on *every* member, not just the sphere — `[M]`
``COSINE_INTERVAL.quotient(Trivial).name`` is ``'[-1,1]/Trivial'`` — so
an answer the derivation could give for :math:`S^2` it could not give
for an interval, and a field populated on one member only would be a
special case wearing a general name. ⟹ ``None``, and the registry keeps
a bare-sphere arm of its own
(:ref:`manifold-second-twin-reference`).

.. warning::

   ⚠ **The reference lives in the CHART's coordinates; the entry lives
   on the orbit space. Nothing gates the pair, and that is not a
   defect — it is the two-coordinate-systems asymmetry, one register
   down.** `[M]` 2026-09-02:
   ``GEOMETRY_ANGULAR_SYMMETRY['slab'].support.name`` is
   ``'S^2/O2_x'`` while its ``reference.support.name`` is
   ``'[-1,1]'``, and ``grep`` finds **no** read of
   ``reference.support`` anywhere in ``orpheus/``.

   The pushforward :math:`2\pi\,d\mu` is naturally written in the
   invariant :math:`\mu`, which is exactly the coordinate system
   ``quotient_map``'s image lands in — so the two are consistent, and a
   future gate should assert
   ``entry.reference.support == entry.realization``, **not**
   ``== entry``. Asserting the second would be the axis-blind mistake
   :ref:`in reverse <manifold-so2-axis-is-a-parameter>`: it would demand
   that a measure carry an axis that the *measure* genuinely does not
   know. Only the space does.


.. _manifold-value-at-function-scope:

Why the TYPE arrives under ``TYPE_CHECKING`` and the VALUE at function scope
-----------------------------------------------------------------------------

The ``reference`` field needs two things from
:mod:`orpheus.numerics.exactness` and
:mod:`orpheus.numerics.generating_measure`, and they need **different
mechanisms**, for a reason worth stating once because it recurs:

* the **type**, :class:`~orpheus.numerics.exactness.ReferenceMeasure`,
  which is only ever an annotation — and an annotation is erased at
  runtime under ``from __future__ import annotations``, so a
  :data:`typing.TYPE_CHECKING` import carries it for free
  (``manifold.py:96``);
* the **value**, ``LEGENDRE``, which is a real object that the axial
  derivation must *put in the field*. **No guard can carry a value** —
  a ``TYPE_CHECKING`` block is erased, so a name bound only there is
  ``NameError`` at runtime. It needs a real import, and `[M]` the two
  module-scope placements that could conceivably work — the top of the
  import block, and the very bottom with every name already bound — are
  **both** fatal, so what ships is **function scope**
  (``manifold.py:1679``).

⭐ It is worth noticing that the type is *narrow on purpose* and that is
what makes half of this cheap: ``ReferenceMeasure`` is a
``@runtime_checkable`` ``Protocol`` with three members (``name``,
``support``, ``orthogonal_system``), so ``LEGENDRE`` — a
``GeneratingMeasure`` — satisfies it **structurally**, by having the
attributes rather than by inheriting. `[M]`
``isinstance(LEGENDRE, ReferenceMeasure)`` is ``True``. A nominal base
class here would have forced a runtime import of the base into the
generator's own module and bought nothing.

**Why the value cannot be hoisted, measured.** This is the hazard
:ref:`manifold-import-cycle` documents, on the pair that OUTLIVED it —
``manifold`` ⇄ ``generating_measure``, where the loop runs through
``exactness`` and ``measure`` and R2 did not touch it. Here what survives
the hazard is not a ``TYPE_CHECKING`` guard but the function-scope
import, because the thing needed is a value. It was measured rather than
argued — on a **renamed shadow copy** of the
package (``shadowpkg``), so the editable install's ``sys.meta_path``
finder cannot serve the real tree by accident and every subprocess
prints the ``__file__`` it actually loaded. No production file was
touched.

.. list-table:: The ``LEGENDRE`` import, three placements × seven entry points
   :header-rows: 1
   :widths: 34 20 46

   * - Placement
     - Import orders alive
     - What fails, and where
   * - **function scope**, inside
       ``_sphere_mod_o2`` — the shipped shape
     - **7 of 7**
     - nothing. The positive control, without which the two rows
       below carry no information.
   * - module scope, at the **top** of the file
     - **0 of 7**
     - ``ImportError: cannot import name 'Manifold' from partially
       initialized module`` — ``exactness`` is reached first and asks
       ``manifold`` for a class it has not defined yet
   * - module scope, at the **bottom**, every name already bound —
       the most *favourable* module-scope position there is
     - **0 of 7**
     - ``ImportError: cannot import name 'DiscreteMeasure' from
       partially initialized module`` — one hop further along, in
       ``generating_measure``'s own import of ``measure``

The seven entry points are ``manifold``, ``exactness``,
``generating_measure``, ``measure``, ``symmetry``, ``quadrature.registry``
and the package root ``numerics``, each imported first in a fresh
interpreter.

⭐ **And the reason the answer is the same for all seven — this cycle is
NOT order-dependent, unlike the one the guard was written for.**
``orpheus/numerics/__init__.py`` eagerly imports ``.measure`` (and much
else) at module scope, so *every* ``import orpheus.numerics.X`` runs the
package body first and the effective import order is fixed before the
entry point has any say. The pre-2.4 three-hop
``measure → manifold → symmetry → measure`` cycle was order-dependent —
which is exactly what let a smoke test report green on a broken façade
(:ref:`manifold-import-cycle`) — and this one cannot be, at any
placement. Worse to introduce; cheaper to detect.

⟹ **the general rule, and it is the transferable half:** a
``TYPE_CHECKING`` guard defers a *name*, so it solves the annotation
problem completely and the value problem not at all. When a low-level
module must hold a value minted by a higher-level one, the mechanism is a
function-scope import at the site that mints the entry — and its safety
condition is that the function is never called during module
initialisation. `[M]` 2026-09-02, by AST over ``orpheus/`` with call
depth tracked: **8** calls that can mint a quotient (**7**
``.quotient(...)`` plus one ``.on_orbit_space(...)``) and **0 of 8** at
module scope — every one is inside a function or a method body. The
count is the positive control: a filter that found zero *calls* would
report the same safe-looking zero as one that found zero *module-scope*
calls, and only the first number distinguishes them. So the first
quotient is derived at *rule construction*, long after every module has
loaded.

⚠ **That safety condition is a property of the CALL SITES, not of this
module, so it can be broken from outside it.** A future module-scope
``SPHERE.quotient(...)`` anywhere in ``orpheus/`` — a pre-built
constant, a registry populated at import — would run the derivation
during initialisation and re-open the cycle from the other end. Nothing
gates it today; the cheap check is the AST census above, and its
predicate is *call depth zero*, not the call's spelling.


.. _manifold-basis-invariance-group:

The second operand: a basis declares the symmetry its functions HAVE
====================================================================

The section above gave the pairing's **measure** side: a rule that says
which orbit space its atoms live on. This section is the **basis** side
— tracker 2.1b, 2026-09-01 — and the whole of its content is that the
basis side needed *no new field*. It follows from one elementary fact
about functions on a quotient:

.. math::

   \mathcal{F}(M/H) \;\;\xrightarrow[\;\cong\;]{\;\;\pi^*\;\;}\;\;
   \bigl\{\, f \in \mathcal{F}(M) \;:\; f \circ h = f \ \ \forall h \in H \,\bigr\}
   \;=\; \mathcal{F}(M)^H ,

with :math:`\pi : M \to M/H` the orbit projection and
:math:`\mathcal{F}` any function class the projection respects —
continuous, measurable, :math:`L^2`, smooth off the singular stratum.
Nothing here needs regularity; the statement is set-theoretic. Pulling a
function back along :math:`\pi` produces an :math:`H`-invariant function
on :math:`M`, and every :math:`H`-invariant function descends to the
quotient, because it is constant on orbits and the orbits *are* the
points of :math:`M/H`. Being a function on :math:`M/H` **is** being
:math:`H`-invariant, spelled two ways. So a basis that has
already named :math:`M/H` as its :attr:`domain
<orpheus.numerics.basis.base.Basis.domain>` has already declared its
group: the group is :attr:`Quotient.by
<orpheus.numerics.manifold.Quotient.by>`, sitting inside the slot
tracker 2.1 minted.

⭐ **The tracker asked for the wrong object, and the phase opener said
so.** Its row read, verbatim, *"``Basis.invariance_group`` — absent;
derivable for every shipped basis"*, and the plan's own census measured
*"0 of 6 subclasses answer it"*. Both true — and the design they invited
was a second abstract property with six overrides, kept in agreement
with ``domain`` by hand: exactly the two-tags-that-drift shape this page
exists to argue against. `[M]` **0 of 6** shipped
bases carried the name before this step — ``git show HEAD:`` over every
module of :mod:`orpheus.numerics.basis` and over
:mod:`orpheus.sn.operators.loss_kernel_gauge`, which is where
:class:`~orpheus.sn.operators.loss_kernel_gauge.LossKernelBasis` lives,
returns zero occurrences in each. After it, **6 of 6** answer (the
denominator is ``Basis.__subclasses__()`` walked recursively at runtime,
not a hand-list) and the basis-side diff is **one** file,
``orpheus/numerics/basis/base.py``: one concrete ``@final`` property on
the ABC, **zero** subclass edits. That is ``coding-standards``' *clean
before extending* landing as a no-op extension through a single generic
body, and it is the same dissolution tracker 2.0d's ``quotient_group``
**field** underwent at 2.0c, one level over: the fact was already in the
type, so the field would have been its second home.

.. _manifold-has-versus-spent:

HAS and SPENT: one slot for a function, two for a point set
------------------------------------------------------------

A measure carries **two** group slots and a basis carries **one**, and
that asymmetry is not an oversight on either side — it is the difference
between a point set and a function.

* :attr:`DiscreteMeasure.invariance_group
  <orpheus.numerics.measure.DiscreteMeasure.invariance_group>` — what the
  atom list **HAS**: a stored field recording a subgroup under which the
  nodes, weights included, are closed. It is a *declaration*, not a
  computed stabiliser, and ``None`` means unspecified rather than
  trivial.
* :attr:`DiscreteMeasure.quotient_group
  <orpheus.numerics.measure.DiscreteMeasure.quotient_group>` — what it
  **SPENT**: the group its support was folded by, derived from
  :attr:`Quotient.by <orpheus.numerics.manifold.Quotient.by>` and stored
  nowhere (tracker 2.0c).

For a POINT SET these come apart in both directions, and the shipped
rules realise three of the four combinations. The table is **exhaustive
over the family**, not a sample: its denominator is every
``classmethod`` factory on
:class:`~orpheus.numerics.quadrature.directional.Quadrature`,
enumerated by ``vars(Quadrature)`` — `[M]` **five of five**
(``vv-principles`` #31's finite-roster corollary: for an enumerable
shipped set, probe every member, because the one you skip is where the
counterexample lives).

.. list-table:: `[M]` 2026-09-01 — HAS and SPENT on all five shipped rules
   :header-rows: 1
   :widths: 26 20 27 27

   * - Rule
     - ``support.name``
     - HAS (``invariance_group``)
     - SPENT (``quotient_group``)
   * - ``lebedev(17)``
     - ``'S^2'``
     - ``OctahedralOh``
     - ``None``
   * - ``level_symmetric(8)``
     - ``'S^2'``
     - ``OctahedralOh``
     - ``None``
   * - ``product(4, 8)``
     - ``'S^2'``
     - ``Dnh(8)``
     - ``None``
   * - ``gauss_legendre(8)``
     - ``'S^2/O2_x'``
     - ``Mirror('x')``
     - ``O2('x')``
   * - ``folded_product(4, 8)``
     - ``'S^2/sigma_y'``
     - ``None``
     - ``Mirror('y')``

⚠ The missing combination is (**no** HAS, **no** SPENT) — an untagged
rule on the bare sphere. Nothing shipped is in that state, which is a
fact about the rules, not about the type.

Read the last two rows. The slab's polar rule HAS :math:`\sigma_x` and
SPENT :math:`O(2)_x` — **two different groups in two slots on one
measure**, so no single field could carry both. And the
:math:`\sigma_y`-folded product rule HAS **nothing**, precisely because
it spent :math:`\sigma_y`: folding keeps one representative per orbit,
and a set with one point of each mirror pair is no longer closed under
the mirror **in the base**. *Spending a symmetry destroys having it* —
which is why reading either slot as the other is ``plan-authoring`` §3's
ambiguous-name hazard, and why
:attr:`~orpheus.numerics.measure.DiscreteMeasure.quotient_group`'s own
docstring says so.

⚠ **Three words were added to that sentence on 2026-09-02, and they are
load-bearing.** *In the base* is where the fold's node set fails to be
closed; on the ORBIT SPACE :math:`\sigma_y` acts trivially and every
orbit is its own image, so `[M]` since tracker 2.2b
``folded_product(4, 8).measure.is_invariant_under(SubgroupOfO3.Mirror("y"))``
is **True** where it was ``False``
(:ref:`manifold-one-invariance-kernel`). The stored slot did NOT move —
`[M]` ``invariance_group`` is still ``None`` on that rule — because it
is a declaration about the representatives, which is the object the
sentence above is about. The computed predicate now answers a THIRD
question, *does the group act on the orbit space and permute it*, and
the fold's answer to that is yes for the whole of :math:`D_{2h}`
(:ref:`manifold-2-2b-what-moved`).

For FUNCTIONS the two collapse, by the isomorphism above. There is no
"folded away" to lose: a basis on :math:`M/H` *is* a set of
:math:`H`-invariant functions on :math:`M`, so what it has and what its
domain spent are one property. A basis therefore carries **one** slot,
named for what it HAS and read off what its domain SPENT:

.. list-table:: `[M]` 2026-09-01 — the one slot, over all six shipped bases
   :header-rows: 1
   :widths: 36 30 34

   * - Basis
     - ``domain.name``
     - ``invariance_group``
   * - ``SphericalHarmonicBasis(L)``, :math:`L \in \{0,1,3,7\}`
     - ``'S^2'``
     - ``Trivial``
   * - ``MirrorEvenSphericalHarmonicBasis(L=2, mirror_axis=a)``,
       :math:`a \in \{x,y,z\}`
     - ``'S^2/sigma_a'``
     - ``Mirror(a)`` — and ``is domain.by``
   * - ``IndicatorBasis`` from
       :meth:`EnergyGrid.as_basis
       <orpheus.data.energy_grid.EnergyGrid.as_basis>`
     - ``'energy'`` (an :class:`EnergyGroups`)
     - ``None``
   * - ``IndicatorBasis`` from
       :meth:`Mesh1D.indicator_basis
       <orpheus.geometry.mesh.Mesh1D.indicator_basis>`
     - ``'spatial_R1'`` (a :class:`RealSpace`)
     - ``None``
   * - ``WeightedIndicatorBasis``, ``OverlapBasis`` — both **delegate**
       ``domain`` to the basis they wrap
     - the wrapped basis's
     - ``None``, by delegation
   * - ``LossKernelBasis``
     - ``'index(sn_trace_orbit(...)_g)'`` (an :class:`IndexSet`)
     - ``None``

The mirror-even row is the one that carries the design: `[M]` the
answer is not merely *equal* to ``domain.by``, it **is** it —
``basis.invariance_group is basis.domain.by`` — so there is no second
object that could drift. A stored copy that happened to be right would
pass ``==`` and fail ``is``, which is what
``test_e2b_a_mirror_even_harmonic_HAS_its_mirror_read_off_its_domain``
asserts.

.. _manifold-invariance-three-arms:

Three arms, and why the answer off the sphere is ``None``, not ``Trivial``
---------------------------------------------------------------------------

The derivation is a ``match`` on the **type** of ``domain``, with three
arms:

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - ``domain``
     - Answer
     - Why
   * - ``Quotient(base=Sphere(), by=H)``
     - :math:`H`
     - the functions descend from :math:`M/H`, so they are exactly the
       :math:`H`-invariant ones
   * - ``Sphere()``
     - ``Trivial``
     - :math:`O(3)` **acts** on the domain and the basis has spent none
       of it — a domain of :math:`S^2` promises no invariance, whatever
       the individual functions happen to have (see the lower bound
       below)
   * - anything else
     - ``None``
     - no subgroup of :math:`O(3)` acts at all

⚠ **The third arm is a category answer, and** ``Trivial`` **would be a
lie.** No subgroup of :math:`O(3)` acts on a spatial mesh, an
energy-group index or a trace-DOF index set — there is no rotation of a
list of group boundaries. ``Trivial`` names the subgroup :math:`\{e\}`
**of** :math:`O(3)`, so writing it asserts that :math:`O(3)` acts on
this domain at all; ``None`` says the question does not arise. The
distinction is exactly the one
:attr:`DiscreteMeasure.phase <orpheus.numerics.measure.DiscreteMeasure.phase>`
already draws for the *same* manifolds when it refuses to classify a
non-angular support as angular, and it is why the two spellings of
"nothing" on the two sides mean opposite things: a full-sphere rule's
``quotient_group`` is ``None`` because it **spent nothing**, while
full-sphere harmonics' ``invariance_group`` is ``Trivial`` because they
**have** the trivial group. Same word in English, different lattice
elements — :math:`\{e\}` on one side, *no answer* on the other.

⭐ And the arms are decided by the domain and by nothing else, which is
a testable claim rather than a description:
``test_e4_the_group_is_decided_by_the_domain_and_by_nothing_else`` runs
all three on **one** class shape whose instances differ only in
``domain``, so an implementation keyed on the subclass — an
``isinstance`` on ``SphericalHarmonicBasis``, say — would give every
stub the same answer and fail.

.. _manifold-invariance-lower-bound:

The reading is a LOWER BOUND, and that is why the property is ``@final``
-------------------------------------------------------------------------

The domain gives the symmetry the basis is *guaranteed* to have, not the
largest one it happens to have. `[M]` ``SphericalHarmonicBasis(L=0)`` is
a single constant function, invariant under all of :math:`O(3)` — and it
answers ``Trivial``, at :math:`L \in \{0,1,3,7\}` alike, because its
domain says :math:`S^2` and a domain of :math:`S^2` promises nothing
more. The property is a **declaration read off a type**, never a
computed stabiliser.

Under-declaring is therefore *legal and lossy*: a basis invariant under
more than its domain shows will be refused pairings it could have
admitted, once the frame checks the two halves
(:ref:`manifold-invariance-pairing`). The remedy is to **declare the
finer domain** — a Legendre basis on :math:`S^2/O(2)_x` rather than the
full harmonics on :math:`S^2` — which is tracker 3.4, landed 2026-09-02,
and which is the level-2 half of ERR-080's repair.

.. important::

   ⭐ **The reading is still a lower bound in general — and since #432
   the ONE case where the gap bit is closed, by making the domain name
   the maximal group.** Both halves matter and they are different
   claims. The general one is unchanged: ``SphericalHarmonicBasis(L=0)``
   is :math:`O(3)`-invariant and answers ``Trivial``, because its
   domain is :math:`S^2` and a domain of :math:`S^2` promises nothing.
   The specific one is that the axial orbit space used to be *keyed by
   the rotation half*, so a Legendre basis on it derived
   :math:`SO(2)_a` while its :math:`P_\ell(\mu)` are invariant under
   the whole :math:`O(2)_a` — a gap that cost a real, mathematically
   admissible pairing (the :math:`\sigma_b`-fold). Naming an orbit
   space by its **stabiliser** (:ref:`manifold-orbit-space-stabiliser`)
   removes that gap *without* touching the property's derivation: the
   reading is the same ``match`` on ``domain.by``; what changed is that
   ``by`` is now maximal by construction.

   ⟹ the remedy for under-declaration remains **declare a finer
   domain**, never *widen the reading*, which is why the property stays
   ``@final``.

✅ **The frame side of that remedy landed first, 2026-09-02 (tracker
2.5), and it landed for the reason the paragraph above gives.** A basis
that declares the finer domain is not a
:class:`~orpheus.numerics.basis.SphericalHarmonicBasis`, and the
harmonic frame had **two** ``isinstance`` doors on exactly that class —
so 3.4's basis could not have been bound at all. Both doors now demand a
*surface* instead, the two-member
:class:`~orpheus.numerics.basis.base.TruncatedBasis` ``Protocol``
(``L`` + ``space``), which is the same move this section argues for one
object over: **key on what the object declares, never on which subclass
it is.** The same step made the seven production sites that re-minted
the angular coefficient space from the integer :math:`L` READ it off the
bound basis instead, so the family a quadrature chooses now propagates
to every operator end and every moment field by construction rather than
by coincidence. `[M]` 2026-09-02, the surface's shipped implementors are
**2 of the 5** :class:`~orpheus.numerics.basis.Basis` subclasses (the
full harmonics and their σ-even restriction); the third — 3.4's Legendre
basis — does not exist yet, so this is a cleared path and **not** a
repair: ERR-080 stays open with its three ``xfail(strict=True)`` rows
untouched. See :ref:`frame-moment-space-single-home`.

⛔ **The remedy is never to override the property**, and the type
enforces that: an override lets ``domain`` and ``invariance_group``
disagree, which is precisely the two-homes-for-one-fact state the
derivation exists to make unspellable. Hence ``@final`` — `[M]`
``Basis.__dict__['invariance_group'].fget.__final__`` is ``True``. It is
the same argument that keeps
:attr:`~orpheus.numerics.measure.DiscreteMeasure.quotient_group` derived
rather than stored, applied to the object on the other side of the
frame.

.. _manifold-invariance-pairing:

The pairing, measured: ERR-080 as a lattice verdict
-----------------------------------------------------

With both operands in hand, the check ERR-080 needs can finally be
*written down*. The rule is a containment in the subgroup lattice:

.. math::

   \text{admissible}
   \quad\Longleftrightarrow\quad
   \underbrace{G_{\text{spent}}}_{\texttt{measure.quotient\_group}}
   \;\subseteq\;
   \underbrace{G_{\text{have}}}_{\texttt{basis.invariance\_group}} ,

read as: *the symmetry a rule folded away must be one the basis's
functions are blind to.* If the rule kept one representative per
:math:`H`-orbit, then any function that distinguishes points within an
orbit has been handed a sample that cannot see the distinction, and the
pairing is a forgery whatever its shapes say.

Measured on the objects that ship, and on the pairing the tree
**actually forms**: each rule against the basis its own
``angular_frame(2)`` binds. The denominator is again all five
``Quadrature`` factories, and `[M]` **exactly one of the five fails**:

.. list-table:: `[M]` 2026-09-01 — ``rule.measure`` vs ``rule.angular_frame(2).basis``
   :header-rows: 1
   :widths: 26 22 18 34

   * - Rule
     - Basis its frame binds
     - SPENT / HAVE
     - ``have.contains(spent)``
   * - ``folded_product(4, 8)``
     - ``MirrorEvenSphericalHarmonicBasis`` (``mirror_axis=1``)
     - ``Mirror('y')`` / ``Mirror('y')``
     - ✅ **True** — and the two are the *same object*
       (``have is spent``), because ``basis.domain is measure.support``
       is one memoised :class:`Quotient`
   * - ``gauss_legendre(8)``
     - ``SphericalHarmonicBasis``
     - ``O2('x')`` / ``Trivial``
     - ⛔ **False** — ERR-080, as a lattice verdict. ⛔ The SPENT cell
       read ``SO2('x')`` until 2026-09-02; the verdict is unchanged,
       since ``Trivial`` contains neither
   * - ``lebedev(17)``
     - ``SphericalHarmonicBasis``
     - ``None`` / ``Trivial``
     - spent nothing, so nothing to contain — admitted
   * - ``level_symmetric(8)``
     - ``SphericalHarmonicBasis``
     - ``None`` / ``Trivial``
     - the same
   * - ``product(4, 8)``
     - ``SphericalHarmonicBasis``
     - ``None`` / ``Trivial``
     - the same

The first row is the whole design in one line: the fold's two halves do
not merely *agree*, they read one group object out of one manifold, so
no drift between them is representable. The second row is ERR-080 —
stated, for the first time in this corpus, as a verdict a predicate
could return rather than as a story about zero-padded columns. And the
shape of the table matters as much as the verdict in it: the defect is
**not** a general weakness of the harmonic frame — `[M]` **1 of the 5**
shipped rules fails its own frame's pairing, and it is the 1-D one.
⚠ That is a *different* denominator from ERR-080's own scope census,
which counts ``(constructor, order)`` rows (`[M]` 7 of 15 non-zero, 5 of
them this defect); the two are not comparable and neither implies the
other.

.. note::

   ⛔ **Nothing refuses on this verdict yet, and the reason is worth
   stating precisely.** The frame's pairing gate — the plan's **G2**,
   fused with its siblings at tracker 2.2 — is not written. And a gate
   written naively on the *frame's* measure would be **inert on the very
   row it is for**: `[M]` 2026-09-01,
   ``Quadrature.gauss_legendre(8).angular_frame(2).measure.support.name``
   is still ``'S^2'`` (the surviving 1-D forgery,
   :ref:`manifold-err-080`) while
   ``Quadrature.gauss_legendre(8).measure.support.name`` is
   ``'S^2/O2_x'``. That is why the gate below reads the verdict off the
   **quadrature's** measure — the object that knows what it spent — and
   why the negative leg is a *measurement made spellable*, not a
   refusal (``plan-authoring`` §6c: a step that adds a gate must land
   with the case the gate catches, and this step deliberately adds the
   operand rather than the gate).

   ⛔ This block read, verbatim, "**ERR-080 is OPEN.** It is held by
   the ``xfail(strict=True)`` gate … three rows red by design. Nothing
   on this page repairs it." until 2026-09-02. Both halves were true when
   written and both were repealed the same day: #429's fused commit
   landed the pairing gate this section calls **G2**, the three
   ``xfail(strict=True)`` rows flipped, and **ERR-080 is CLOSED**. The
   verdict this section measures is what the gate now reads — see
   :ref:`manifold-g0-descent-arrow` for the predicate that consumes it.

**The campaign plan's "Part IV" lattice table, as a test.** That
four-row admissibility table — the same section of the #429 plan quoted
at :ref:`manifold-err-080-is-a-section` — was the *done-when* for this
step, and it now runs on real objects rather than on names
(``test_e5_part_IV_lattice_table_runs_on_the_objects_that_ship``):

.. list-table::
   :header-rows: 1
   :widths: 10 30 26 34

   * - Row
     - Basis space (HAVE)
     - Rule (SPENT)
     - Verdict
   * - 1
     - full :math:`S^2` harmonics — ``Trivial``
     - slab — ``O2('x')``
     - ⛔ refused: this is ERR-080's pairing, and the refusal is
       categorical — no tolerance is involved
   * - 2
     - :math:`S^2/O(2)_x` — ``O2('x')``
     - slab — ``O2('x')``
     - ✅ the repair
   * - 3
     - :math:`S^2/O(2)_x` — ``O2('x')``
     - full sphere — ``None``
     - ✅ a smaller space on a full rule is legal
   * - 4
     - :math:`S^2/\langle\sigma_y\rangle` — ``Mirror('y')``
     - fold — ``Mirror('y')``
     - ✅ the shipped fold

⚠ Two readings of that table are worth pinning. Row 3's measure side is
spelled ``None`` and not ``Trivial``, for the reason
:ref:`manifold-invariance-three-arms` gives: a full-sphere rule has SPENT
nothing, and the lattice element ``None`` stands for on that side is
:math:`\{e\}`, which every group contains. And rows 2 and 3 need a basis
on :math:`S^2/O(2)_x` — tracker 3.4's Legendre basis. ⛔ This clause
read *"which* `[M]` *does not ship … the gate stands one in with a
test-local stub … a fixture with an expiry date"* until 2026-09-02;
✅ :class:`~orpheus.numerics.basis.legendre_basis.LegendreBasis` **ships**,
the expiry date arrived, and the stub is owed retirement.

⭐ **And the table itself was superseded in KIND, not merely in
tense.** The containment :math:`G_{\text{spent}} \subseteq
G_{\text{have}}` it tabulates is the *third* of three arms of the
predicate that shipped, and the other two are pairings it cannot
express — in particular row 3's converse, a Legendre basis on a
FULL-SPHERE rule, which is legitimate and which a bare lattice test
refuses. Read this table as the pairing's derivation and
:ref:`manifold-g0-descent-arrow` as what the frame checks.

**The gates.** Section E of ``tests/numerics/test_basis_domain.py``,
`[M]` **six** functions and **eleven** collected rows — the module went
13 rows → 24, and the V&V matrix's ``numerics/test_basis_domain`` row
reads the same +11 independently — all ``@pytest.mark.foundation``
(the property is a type law, not a solver claim, and carries no theory
equation label):

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Gate
     - What it pins
   * - ``test_e1``
     - ⭐⭐ the keystone, both legs: the fold's two halves read ONE
       group object (``==``, then ``is``), and the slab's pairing is
       **refusable** — asserted on the quadrature's measure, for the
       reason in the note above.
   * - ``test_e2``
     - the full-sphere harmonics HAVE ``Trivial`` at
       :math:`L \in \{0,1,3,7\}` — :math:`L = 0` included on purpose,
       as the lower-bound witness.
   * - ``test_e2b``
     - the mirror-even basis HAS its mirror **by identity**
       (``is domain.by``), over all three axes, with a negative leg
       showing a different mirror is incomparable — so the answer moves
       with the axis rather than being one constant.
   * - ``test_e3``
     - the category leg: ``None`` on every non-angular basis, including
       both delegating wrappers — with a positive control (the same
       class shape on a sphere answers ``Trivial``), so the arm is not
       "everything is ``None``" (``vv-principles`` #11).
   * - ``test_e4``
     - the three arms on one class shape differing only in ``domain``,
       with the quotient arm exercised through a **second group family**
       (``O2``), since the shipped fold basis only ever brings a
       ``Mirror``. ⚠ `[M]` 2026-09-02 the test's BODY builds
       ``SPHERE.quotient(SubgroupOfO3.O2("x"))`` while its own docstring
       still says ``SO2('x')`` — reported, not edited here.
   * - ``test_e5``
     - Part IV's four-row table above, on shipping objects.


.. _manifold-what-descends:

What descends — the isotypic probe, the descent, and the frame's G0 arrow
=========================================================================

Everything above builds the orbit space and its arrows. This chapter is
the payoff: **given a basis on the base, which of its functions are
functions on the orbit space, what are the two honest ways to spell
them, and what does a frame have to check before binding one to a
rule?** The three answers landed together on 2026-09-02 in #429's fused
commit, which is the repair for :ref:`ERR-080 <manifold-err-080>`.

The organizing identity is one line of representation theory. Pulling a
function back along the quotient map :math:`\pi : M \to M/H` is an
isomorphism onto the :math:`H`-invariant functions on the base,

.. math::
   :label: manifold-descent-isomorphism

   \pi^{*} : \operatorname{Funcs}(M/H) \;\xrightarrow{\ \sim\ }\;
             \operatorname{Funcs}(M)^{H},
   \qquad
   (\pi^{*} f)(x) = f(\pi(x)),

because :math:`\pi` is surjective and its fibres are exactly the orbits.
So "the functions on the orbit space" and "the invariant functions on
the base" are the same vector space wearing two coordinate systems —
which is the function-side twin of the fact that the orbit space itself
has two (:ref:`manifold-two-coordinate-systems`).

.. (vv-status rationale) A representational identity: it says the two
   realizations name one space, not what any solver computes. Its
   verifiable content is the bit-identity gate on the shipped rules
   (``Descent.is_isomorphism``) and the foundation tests of
   ``Quotient.descending_slots``; it makes no claim about a flux or an
   eigenvalue.
.. vv-status: manifold-descent-isomorphism documented


.. _manifold-descending-slots:

Which slots descend — the entry's isotypic probe
-------------------------------------------------

:eq:`manifold-descent-isomorphism` is an existence statement. The
operational question a basis asks is narrower and decidable: *of the
slots of MY table, which ones are* :math:`H`\ *-invariant?* A function
:math:`f` on the base descends iff it is constant on the fibres of
:math:`\pi`, i.e. iff

.. math::
   :label: manifold-fibre-constancy

   f(g\,x) = f(x)
   \qquad\text{for every } x \in M,\ g \in H .

:meth:`Quotient.descending_slots
<orpheus.numerics.manifold.Quotient.descending_slots>` asks exactly
that, and it lives **on the entry** by user ruling (2026-09-02) for a
reason worth stating: :eq:`manifold-fibre-constancy` is a theorem about
:math:`\pi`, which the entry owns, and it has **two** readers — the
descent below and the :math:`\sigma`-even harmonic sub-basis. Spelled
twice it would be a Cardinal-Rule-2 twin; spelled on the entry it is
one predicate that both read.

The implementation is the predicate transcribed. Tabulate the basis at
generic base points, tabulate it again at their images under a generic
set of the group's elements, and keep the slots that agree to ``atol``
at every image. It is deliberately duck-typed on ``basis`` — the module
imports nothing from :mod:`orpheus.numerics.basis` (:ref:`manifold-import-cycle`)
— so it accepts anything with ``evaluate(points) -> (N, *modes)`` and
returns a boolean mask over ``*modes``.

.. warning::

   ⛔ **A finite sample of a CONTINUOUS group generates a finite
   SUBGROUP, and the subgroup admits slots the real group does not.**
   This is ``vv-principles`` #13 in its sharpest form, and it bites here
   at a specific order. `[M]` 2026-09-02: sampling :math:`SO(2)_x` at
   the four right angles generates :math:`C_4`, and every
   :math:`C_4`-invariant slot passes — so the :math:`m = \pm 4`
   harmonics are **falsely admitted** at :math:`L \ge 4`.
   :meth:`SubgroupOfO3.generic_images
   <orpheus.numerics.symmetry.SubgroupOfO3.generic_images>` therefore
   rotates by angles pairwise **incommensurate** with :math:`\pi` and
   with each other (:math:`1`, :math:`\sqrt2`, :math:`e`, :math:`2.5`,
   :math:`\sqrt7`, :math:`\pi/3 + 0.1`), where no finite subgroup can
   hide. For a FINITE group the generic set is every element of the
   memoised closure, and no sampling question arises.

   ⚠ **And the negative control for this is BLIND below** :math:`L = 4`.
   `[M]` right angles and incommensurate angles select **the same**
   slots at :math:`L = 1, 2, 3` about :math:`x`; the first divergence is
   at :math:`L = 4`. A gate that exercises the trap only at the orders a
   solve typically uses would read green under the broken sampling —
   which is why the probe's own tests reach :math:`L = 4`.

   ⭐ Contrast :meth:`DiscreteMeasure.is_invariant_under
   <orpheus.numerics.measure.DiscreteMeasure.is_invariant_under>`, which
   decides
   continuous groups **exactly** (ERR-072 is the record of what happens
   when it does not). A probe of FUNCTIONS cannot do that — there is no
   closed form for "is this arbitrary tabulated slot axisymmetric?" — so
   it samples where a finite subgroup cannot masquerade.

.. note::

   ⭐ **The probe points are NORMALISED, and that is a repair the fix
   forced.** The nine generic directions are drawn from a fixed seed and
   divided by their norms, so they are points of :math:`S^2`. The
   five-direction probe this replaced (the retired
   ``_PARITY_PROBE_DIRECTIONS``) was **not**: `[M]` its norms are
   :math:`0.8307 \ldots 0.9980`, i.e. every one of them is off the
   sphere, and after tracker 0.6 wired the membership refusal into
   :meth:`SphericalHarmonicBasis.evaluate
   <orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis.evaluate>`
   they would have been **refused** — the same refusal that closes
   ERR-080's level-1 half, applied to a probe nobody had thought of as
   data. `[M]` normalising them left every mask **bit-identical**, on
   all **15** (mirror axis, :math:`L`) rows for
   :math:`L \in \{0,1,2,3,4\}` × three axes, so nothing about the fold
   moved.

**What it measures.** `[M]` 2026-09-02, the degree-4 real harmonics
(a :math:`45`-slot rectangular table of which :math:`25` are live —
the :math:`|m| > \ell` padding is identically zero and therefore
descends *vacuously*, which is why every count below is over **live**
slots):

.. list-table:: Descending live slots of :math:`\{Y_\ell^m\}_{\ell\le4}` per axial entry
   :header-rows: 1
   :widths: 22 20 58

   * - Entry
     - live descending
     - which slots, and what it means
   * - :math:`S^2/O(2)_x`
     - **5 of 25**
     - exactly :math:`\{(\ell, m{=}0)\}` for
       :math:`\ell = 0\ldots4` — one per degree, the :math:`m = 0`
       column. The trivial isotypic component is one-dimensional in
       every degree (Schur), and about :math:`x` it is a set of SLOTS.
   * - :math:`S^2/O(2)_y`
     - **2 of 25**
     - :math:`(0,0)` and :math:`(1,{+}1)` only
   * - :math:`S^2/O(2)_z`
     - **2 of 25**
     - :math:`(0,0)` and :math:`(1,{-}1)` only

⭐ **The asymmetry between the rows is not a defect in the probe; it is
the two-pole convention showing up as a rank statement.** The real
spherical harmonics of this corpus take :math:`\cos\theta = \mu_x`, so
:math:`x` is the harmonics' own polar axis and the invariant subspace
lines up with slots. About :math:`y` and :math:`z` the invariant
subspace is still one-dimensional in every degree — Schur does not care
which axis you picked — but from :math:`\ell \ge 2` it is a *linear
combination* of several slots rather than one of them, so a
slot-mask has nothing to return there. The probe answers honestly about
whatever slots the basis has; deciding what to DO about the misalignment
is the descent's business, below.

**The consumer that already existed.**
:attr:`MirrorEvenSphericalHarmonicBasis.even_slot_mask
<orpheus.numerics.basis.spherical_harmonic_basis.MirrorEvenSphericalHarmonicBasis.even_slot_mask>`
now READS this probe. It used to classify each slot as :math:`\sigma`-even
or :math:`\sigma`-odd with its own five-direction parity test; "even
under :math:`\sigma_a`" and "constant on the orbits of
:math:`\langle\sigma_a\rangle`" are the same predicate, because a mirror
orbit is the pair :math:`\{\Omega, \sigma\Omega\}`. `[M]` the two agree
**bit-identically on 15 of 15** (axis, :math:`L`) rows, so the
collapse is a pure Pattern-2 single-sourcing with no numerical
consequence.


.. _manifold-descent:

The two realizations of the descended space, and the discriminator
--------------------------------------------------------------------

:eq:`manifold-descent-isomorphism` gives the descended space two honest
spellings, and a codebase that ships both without a witness has a twin:

* **upstairs** — the :math:`H`-invariant *subspace of a basis on the
  base*, kept in that basis's own layout. For the real harmonics and
  :math:`O(2)_x` that is the :math:`m = 0` column
  :math:`\{Y_\ell^0\}`; for a coordinate mirror it is the
  :math:`\sigma`-even slots, which is exactly what
  :class:`~orpheus.numerics.basis.spherical_harmonic_basis.MirrorEvenSphericalHarmonicBasis`
  realizes.
* **downstairs** — the quotient's OWN classical basis, when it has one.
  For :math:`S^2/O(2)_a` that is
  :math:`\{P_\ell(\mu)\}`,
  :math:`\mu = \Omega\cdot\hat e_a`
  (:class:`~orpheus.numerics.basis.legendre_basis.LegendreBasis`): a
  FLAT head of :math:`L+1` coefficients with no slots to zero.

:class:`~orpheus.numerics.basis.descent.Descent` is that pair as ONE
object, and it carries the ruling that says which one a frame binds
(user-ruled 2026-08-31):

   **downstairs when the quotient has a classical named basis**
   (:math:`S^2/O(2)_a \to \{P_\ell\}`)\ **, upstairs otherwise**
   (:math:`S^2/\sigma_a` has no classical family — its
   :math:`\sigma`-even harmonics are the only spellable realization).

That sentence is :attr:`Descent.frame_basis
<orpheus.numerics.basis.descent.Descent.frame_basis>`, and it is what
:meth:`Quadrature._harmonic_basis
<orpheus.numerics.quadrature.directional.Quadrature._harmonic_basis>`
binds. ⭐ **The point is that the basis a frame carries is DERIVED from
the entry, never chosen at the call site.** Until 2026-09-02 that
dispatch read the quadrature's ``folded_by`` TAG and knew nothing of the
1-D case, which is the whole of ERR-080; it now reads
``measure.support`` and asks the entry.

**The isomorphism is checkable, and at the BIT tier.** `[M]`
2026-09-02, ``downstairs.evaluate(π(Ω)) == upstairs_columns(Ω)`` is
``np.array_equal`` — :math:`\max\lvert\Delta\rvert = 0.0` exactly — on
**7 of 7** shipped full-sphere rules at :math:`L = 4`
(``level_symmetric(4)``, ``level_symmetric(8)``, ``lebedev(5)``,
``lebedev(11)``, ``lebedev(17)``, ``product(4,4)``, ``product(8,8)``).

.. warning::

   ⛔ **That bit tier is a MEASURED CONSTRAINT on how the polynomial is
   spelled, not a happy accident — and no single scipy routine meets
   it.** `[M]` 2026-09-02 on ``gauss_legendre(2,4,8,16)`` at
   :math:`L \le 4`, against the harmonics' own :math:`m = 0` column:

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - spelling of :math:`P_\ell(\mu)`
        - :math:`\max\lvert\Delta\rvert` against the column
      * - ``lpmv(0, ℓ, μ)`` for every :math:`\ell`
        - :math:`0` at :math:`\ell \ne 1`; **8.3e-17 … 1.1e-16** at
          :math:`\ell = 1`
      * - ``eval_legendre(ℓ, μ)`` for every :math:`\ell`
        - :math:`0` at :math:`\ell \le 1`; **up to 4.8e-16** at
          :math:`\ell \ge 2`
      * - ``1.0`` / :math:`\mu` (the input array) / ``lpmv`` — the
          shipped branching
        - ``array_equal``, **4 of 4** rules

   The branching is what
   :func:`~orpheus.numerics.basis.legendre_basis.legendre_table` ships,
   and the reason is a bit-identity claim one layer up: the repair must
   not move the slab flux where the old basis was already right. `[M]`
   with the shipped spelling the converged flux at :math:`L = 0, 1` is
   ``array_equal`` to the pre-repair answer; with pure ``lpmv`` the
   :math:`L = 1` row is not, and moves by **2.753e-14** on ERR-080's own
   fixture — a :math:`10^{-16}` table perturbation amplified by the
   Krylov solve. That would have traded a bit-identity claim for a
   tolerance, on the two rows that are the gate's positive controls.

.. warning::

   ⚠ **The upstairs face is slot-ALIGNED only about the harmonics' own
   polar axis, and the refusal is keyed on the AXIS rather than on
   measured alignment.** :meth:`Descent.upstairs_columns
   <orpheus.numerics.basis.descent.Descent.upstairs_columns>` refuses an
   an axial group about :math:`y` or :math:`z` outright. The tempting
   alternative — measure the alignment and refuse when it fails — is a
   ``vv-principles`` #17 trap: `[M]` about :math:`y` and :math:`z` the
   invariant subspace IS slot-aligned at :math:`\ell \le 1` and only
   spreads from :math:`\ell \ge 2`, and :math:`L = 0` is the order at
   which **every** solve mints its fission and :math:`(n,2n)` moment
   ends — isotropic ones included. An alignment-keyed refusal would
   therefore be silently inert exactly where the traffic is. The
   downstairs face has no such restriction and is available at every
   axis, which is why it is the one every consumer binds.


.. _manifold-g0-descent-arrow:

G0 — a frame's two halves must name ONE orbit space
-----------------------------------------------------

A frame binds a basis to a measure. The level-2 check — do the
*spaces* compose? — has always passed on ERR-080's pairing
(:ref:`manifold-three-levels`), because both sides are well-formed
vector spaces. The level-1 check is the one that was missing, and it
now has a single predicate:

   **a frame binding functions on** ``basis.domain`` **to a rule on**
   ``measure.support`` **is admissible iff a quotient map**
   ``measure.support -> basis.domain`` **EXISTS; the frame's table is
   the basis pulled back along it.**

:func:`~orpheus.numerics.manifold.quotient_onto` returns that arrow or
``None``, in three honest cases:

#. ``source == target`` — the identity (the special case :math:`K = H`;
   the slab's rule on :math:`S^2/O(2)_x` with the Legendre basis on the
   same entry);
#. ``target`` is a quotient of ``source`` itself — the entry's own
   :attr:`~orpheus.numerics.manifold.Quotient.quotient_map` (a Legendre
   basis on a full-sphere rule: :math:`P_\ell(\Omega\cdot\hat e_a)` is a
   perfectly good function on a Lebedev or level-symmetric rule);
#. both are quotients of one base and the group ``source`` SPENT is
   contained in the group ``target`` was quotiented BY
   (:math:`K \subseteq H`) — the induced map
   :math:`M/K \to M/H`.

⭐ **Why one predicate, and not the containment**
:math:`G_{\text{spent}} \subseteq G_{\text{have}}` **the pairing was
first stated as** (:ref:`manifold-invariance-pairing`): the containment
is case 3, and cases 1 and 2 are pairings it cannot express — case 2 in
particular is the Legendre-on-a-full-sphere-rule binding, which is
legitimate and which a bare lattice test refuses. Asking for the ARROW
is the same question asked in the category, and it answers all three.

`[M]` 2026-09-02, the shipped verdicts, every row constructed and run:

.. list-table:: G0 on the shipped pairings
   :header-rows: 1
   :widths: 26 26 12 36

   * - rule (``measure.support``)
     - basis (``domain``)
     - verdict
     - the arrow, or the reason
   * - slab GL — :math:`S^2/O(2)_x`
     - Legendre on :math:`S^2/O(2)_x`
     - ✅
     - identity (case 1) — **the repair**
   * - sphere rule — :math:`S^2`
     - full harmonics on :math:`S^2`
     - ✅
     - identity (case 1)
   * - sphere rule — :math:`S^2`
     - Legendre on :math:`S^2/O(2)_x`
     - ✅
     - the entry's :math:`\pi` (case 2) — a full-sphere rule may
       carry :math:`P_\ell(\Omega\cdot\hat e_x)`
   * - :math:`\sigma_y` fold — :math:`S^2/\sigma_y`
     - :math:`\sigma`-even harmonics on the same entry
     - ✅
     - identity (case 1)
   * - slab GL — :math:`S^2/O(2)_x`
     - full harmonics on :math:`S^2`
     - ⛔
     - **ERR-080's pairing.** No map :math:`S^2/O(2)_x \to S^2` exists
       — the arrow runs the other way
   * - :math:`\sigma_y` fold — :math:`S^2/\sigma_y`
     - full harmonics on :math:`S^2`
     - ⛔
     - same shape: a fold cannot carry the unfolded family
   * - :math:`\sigma_y` fold — :math:`S^2/\sigma_y`
     - Legendre on :math:`S^2/O(2)_x`
     - ✅
     - the induced :math:`S^2/\sigma_y \to S^2/O(2)_x` (case 3), since
       :math:`\sigma_y \in O(2)_x`. ⛔ This row read **⛔ ⚠
       mathematically admissible, refused** until 2026-09-02 — see the
       warning below
   * - :math:`\sigma_y` fold — :math:`S^2/\sigma_y`
     - Legendre on :math:`S^2/O(2)_y`
     - ⛔
     - the NEGATIVE leg of the row above, and the one that makes it
       falsifiable: :math:`\sigma_y \notin O(2)_y` (a mirror in the
       :math:`y`-plane flips :math:`\hat e_y`), so no arrow exists

The refusal message names both operands and both groups, and points at
:meth:`Quadrature.angular_frame
<orpheus.numerics.quadrature.directional.Quadrature.angular_frame>` as
the surface that derives the right basis, so a caller who reaches it is
told what to do rather than what happened.

.. warning::

   ⛔ **This warning read as follows until 2026-09-02, and the
   diagnosis was right — it was the DECLARATION that was too weak, not
   the pairing:**

      *⚠ The last row is a KNOWN over-refusal, and its cause is the
      lower-bound property this page already documents.*
      :math:`P_\ell(\Omega\cdot\hat e_x)` *is invariant under the*
      **full** :math:`O(2)_x` — *including* :math:`\sigma_y`, *a
      reflection in a plane containing the axis, which does not move*
      :math:`\mu_x`. *But* ``Basis.invariance_group`` *is DERIVED from
      the domain as* ``SO2('x')``, *a strict lower bound, and*
      ``SubgroupOfO3`` *has* **no axis-parameterised** :math:`O(2)`
      *member to declare instead* (:math:`D_{\infty h}` *is
      parameter-free). So the verdict is* ``SO2('x') ⊇ Mirror('y')``
      :math:`=` **False**, *and the honest description is "the
      declaration is too weak", not "the pairing is wrong". It is inert
      today — no dispatch selects it, and it is tracked as* **GitHub
      #432**.

   ✅ **#432 landed 2026-09-02 and the row is now ADMITTED.** The
   missing member exists — :class:`~orpheus.numerics.symmetry.O2`, the
   axis's stabiliser — and the orbit space is NAMED by it, so the
   derived ``invariance_group`` is the full :math:`O(2)_x` the
   old warning already knew the functions had
   (:ref:`manifold-orbit-space-stabiliser`). `[M]` 2026-09-02:
   ``GalerkinFrame(LegendreBasis(L=L, axis="x"),
   Quadrature.folded_product(4, 8).measure)`` constructs at
   :math:`L = 0, 2, 4, 6` with a :math:`(16, L{+}1)` table and the arrow
   :math:`S^2/\sigma_y \to S^2/O(2)_x`; the isotropic field's
   :math:`\ell = 0` moment reads :math:`4\pi = 12.566370614359172`
   **bit-identically to** ``sum(weights)`` and its :math:`\ell \ge 1`
   moments :math:`\le 1.42\times10^{-15}` — no aliasing on this fold,
   because the azimuthal rule is exact to trigonometric degree 7 and the
   fold is :math:`\sigma_y`-even. ⚠ The admission is **not** blanket:
   `[M]` the same fold with ``LegendreBasis(axis="y")`` is still refused
   (:math:`\sigma_y \notin O(2)_y`), and on a :math:`\sigma_x`-folded
   rule ``axis="x"`` is refused while ``axis="z"`` is admitted — the
   predicate is the arrow, and the arrow is the lattice.

**Where it fires.** ``FrameBase.__post_init__`` checks the TRIAL half at
construction, and
:attr:`~orpheus.numerics.frame.FrameBase.test_descent` checks the TEST
half on first use (the Petrov-Galerkin subclass binds the test basis,
so it is not available in the base's ``__post_init__``);
:class:`~orpheus.numerics.frame.GalerkinFrame`'s hand-written
constructor calls the same helper explicitly, because it bypasses the
dataclass ``__init__`` that would otherwise run it. The arrows
themselves are cached and are what
:attr:`FrameBase.table <orpheus.numerics.frame.FrameBase.table>` pulls
the nodes back along — so the check and the tabulation read ONE object,
and a frame that passed G0 cannot then tabulate through a different map.
See :ref:`frame-g0-descent-arrow` for the frame-side account.


.. _manifold-normaliser:

Who ACTS on an orbit space — the normaliser, the lift, the induced action
=========================================================================

Everything above this point builds orbit spaces and maps *into* and *out
of* them. This chapter asks the question that has to be answered before
a symmetry can be tested on one at all: **which isometries act on**
:math:`M/H` **in the first place**, and what does a point of
:math:`M/H` have to be handed to them as.

It exists because the tree was asking an invariance question in the
wrong place. A fold's nodes are **representatives** — one point of each
:math:`\sigma_y`-orbit — and a polar marginal's nodes are the *chart
coordinate* :math:`\mu` of :math:`S^2/O(2)_x`. Asking whether the
AMBIENT action of a group permutes those points is a different question
from asking whether the group permutes the *orbits*, and the two
disagree on exactly the object the shipped cylinder uses. `[M]` 2026-09-02,
against a pinned pre-change tree (``git archive HEAD orpheus`` at
``4b7d24c3``, imported with the editable finder stripped and
``orpheus.__file__`` asserted): :math:`\sigma_y` read *not invariant* on
``folded_product(4, 8)`` — it maps a :math:`y \ge 0` representative onto
its absent mate — while it acts **trivially** on :math:`S^2/\sigma_y`,
where every orbit is its own image. Every OWED-symmetry admission of
the shipped cylinder configuration failed on that reading
(:ref:`manifold-gamma-slot`).

#429 tracker 2.2b (user-ruled 2026-09-02, three rulings) makes the
orbit-space reading the ONLY reading: one invariance kernel, one
descent arrow shared by the frame and the selector, and the axial
entries taking the same route as the folds.

.. _manifold-normaliser-criterion:

The criterion — an isometry descends iff it NORMALISES the group
-----------------------------------------------------------------

Let :math:`H \le O(3)` and let :math:`[p]_H` be the :math:`H`-orbit of
:math:`p \in M`. The obvious definition of an action on :math:`M/H` is
:math:`g \cdot [p]_H := [g\,p]_H`, and it is a definition only when the
right-hand side does not depend on which representative arrived:

.. math::
   :label: manifold-normaliser-descent

   g\cdot[p]_H := [g\,p]_H \ \text{ is well defined on } M/H
   \quad\Longleftrightarrow\quad
   g\,H\,g^{-1} = H .

.. (vv-status rationale) manifold-normaliser-descent: A definition-level
   criterion of group theory (an isometry descends to an orbit space iff
   it lies in the normaliser), transcribed here because the tree now
   implements it as a refusal. It states no solver claim and has no
   L0..L3 ladder slot. Its CODE content is verified by the foundation
   gates on ``SubgroupOfO3.is_normalised_by`` / ``normalises`` and by
   ``Quotient.induced_action``'s refusal leg; its consequence for
   selection is measured at :ref:`manifold-gamma-slot`.
.. vv-status: manifold-normaliser-descent documented

If :math:`p' = h\,p` is another representative then
:math:`g\,p' = (g h g^{-1})\,g\,p`, so :math:`[g\,p']_H = [g\,p]_H`
for every :math:`h` exactly when :math:`gHg^{-1} \subseteq H`; the
reverse inclusion follows by applying the same argument to
:math:`g^{-1}`. Outside the normaliser :math:`[g\,p]_H` is not a
function of the orbit, and there is no action to test invariance
against — which is why
:meth:`Quotient.induced_action
<orpheus.numerics.manifold.Quotient.induced_action>` **refuses** rather
than evaluating on whatever representative it was handed.

:meth:`SubgroupOfO3.is_normalised_by
<orpheus.numerics.symmetry.SubgroupOfO3.is_normalised_by>` decides
:eq:`manifold-normaliser-descent` in ONE body on the realization
(:ref:`manifold-realization`): :math:`g` must carry the Lie algebra onto
itself, :math:`\mathrm{Ad}_g\,\mathfrak h = \mathfrak h`, and it must
conjugate every coset representative back into :math:`H`. No family is
sampled (ERR-072). Read at each family that unfolds to the statements
below, which is what the table records — the *answers*, not five pieces
of code:

.. list-table:: `[M]` 2026-09-02, re-measured 2026-09-03 — one positive
   and one negative leg per family
   :header-rows: 1
   :widths: 22 40 19 19

   * - Family of :math:`H`
     - What the one body reduces to
     - Positive leg
     - Negative leg
   * - FINITE (:math:`\sigma_a`, :math:`C_n`, :math:`D_{nh}`,
       :math:`O_h`, :math:`I_h`)
     - the conjugated element set equals the element set, element by
       element, through
       :meth:`RigidMotion.conjugated_by
       <orpheus.geometry.transformation.RigidMotion.conjugated_by>`
     - ✅ ``True``
     - ✅ ``False``
   * - :math:`SO(2)_a`, :math:`O(2)_a`
     - :math:`g\hat e_a = \pm\hat e_a` — :math:`g` carries the LINE of
       :math:`a` onto itself, so the rotations about :math:`a` are
       conjugated to the rotations about :math:`a` and the mirrors
       through :math:`a` to the mirrors through :math:`a`
     - ✅ ``True``
     - ✅ ``False``
   * - :math:`D_{\infty h}`
     - the same, for :math:`z`
     - ✅ ``True``
     - ✅ ``False``
   * - :math:`\{e\}`, :math:`SO(3)`, :math:`O(3)`
     - normal in :math:`O(3)`, so **everything** normalises them
     - ✅ ``True``
     - ⚠ ``True`` — and that IS the answer, not a gap: these three
       families have no negative leg to find

⚠ Read the last row as a measurement, not as an omission. `[M]` over
the eleven families probed, **8 have a genuine negative leg and 3 do
not**, because :math:`\{e\}`, :math:`SO(3)` and :math:`O(3)` are normal
subgroups of :math:`O(3)`. A probe that reported ``False`` there would
be the bug.

⚠ **A TRANSLATED motion is projected, not refused** — the question is
asked of the motion's LINEAR part. A point group acts on *directions*
and a translation does not move a direction, so a translated deck
element normalises exactly what its rotation does; `[M]` 2026-09-03 a
pure translation answers ``True`` on **26 of 26** distinct members of
the lattice, which is the same
convention :meth:`Quadrature.ordinate_permutation
<orpheus.numerics.quadrature.directional.Quadrature.ordinate_permutation>`
already runs (a periodic wrap's ordinate permutation is the identity).
Refusing would have made an isometry that plainly acts on the sphere
unaskable.

.. _manifold-normaliser-lie-criterion-section:

The group-level criterion — one Lie condition, not five family arms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`SubgroupOfO3.normalises
<orpheus.numerics.symmetry.SubgroupOfO3.normalises>` lifts the question
from an ELEMENT to a GROUP — *does* :math:`G` *act on* :math:`M/H`?
Writing :math:`G = \bigsqcup_r r\,G^0`
(:eq:`manifold-group-as-component-and-cosets`), :math:`G \subseteq N(H)`
splits along that decomposition and along nothing else: each
representative :math:`r` is one element, asked by
:eq:`manifold-normaliser-descent`; and :math:`G^0` cannot be asked
element by element, because it has infinitely many. This is exactly
where ERR-072's trap lives — sample the connected group and conjugate —
so the connected half is decided through its Lie algebra:

.. math::
   :label: manifold-normaliser-lie-criterion

   \operatorname{Lie} N(H) \;=\;
   \bigl\{\, X \in \mathfrak{so}(3) \;:\;
   [X, \mathfrak h] \subseteq \mathfrak h
   \ \ \text{and}\ \
   X - \mathrm{Ad}_s X \in \mathfrak h \ \ \forall\, s \in R_H \,\bigr\},

.. (vv-status rationale) manifold-normaliser-lie-criterion: the Lie
   algebra of the normaliser of a closed subgroup, transcribed with its
   proof because the shipped ``IdentityComponent.normalises`` IS this
   expression and every per-family arm it replaced is a corollary of it.
   A structural criterion of Lie theory, not a solver claim, with no
   L0..L3 ladder slot. Its CODE content is verified by the foundation
   gates on ``SubgroupOfO3.normalises`` and by the sampling control
   below (:ref:`manifold-normaliser-sampling-control`), both under
   ``@pytest.mark.foundation``.
.. vv-status: manifold-normaliser-lie-criterion documented

so that :math:`G^0 \subseteq N(H)` iff :math:`\mathfrak g \subseteq
\operatorname{Lie} N(H)` — a finite number of matrix conditions,
:math:`\dim\mathfrak g \le 3` generators against
:math:`\dim\mathfrak h \le 3` generators and :math:`|R_H|` coset
representatives of :math:`H`.

**The proof, both directions.** *Necessity*: if :math:`\exp(tX)`
normalises :math:`H` then for each :math:`s \in H` the curve
:math:`f(t) = \exp(tX)\,s\,\exp(-tX)\,s^{-1}` lies in :math:`H`, starts
at the identity, and is continuous, so it lies in :math:`H^0`;
differentiating at :math:`t = 0` gives
:math:`X - \mathrm{Ad}_s X \in \mathfrak h`, and taking :math:`s` inside
:math:`H^0` gives :math:`[X, \mathfrak h] \subseteq \mathfrak h`.
*Sufficiency*: put :math:`Y_s = X - \mathrm{Ad}_s X \in \mathfrak h`;
then :math:`f'(t)f(t)^{-1} = \mathrm{Ad}_{\exp(tX)} Y_s`, which lies in
:math:`\mathrm{Ad}_{\exp(tX)}\mathfrak h = \mathfrak h` because the
first condition makes :math:`\mathfrak h` invariant under
:math:`\mathrm{ad}_X` and hence under :math:`\mathrm{Ad}_{\exp(tX)}`. So
:math:`f` never leaves :math:`H^0`, and :math:`\exp(tX)` normalises
:math:`H`.

**Why the coset representatives of** :math:`H` **suffice.** For
:math:`s h` with :math:`h = \exp Y`, :math:`Y \in \mathfrak h`,

.. math::

   X - \mathrm{Ad}_{sh} X
   \;=\; (X - \mathrm{Ad}_s X)
   \;+\; \mathrm{Ad}_s\,(X - \mathrm{Ad}_h X),

and :math:`X - \mathrm{Ad}_h X = -(\mathrm{ad}_Y X +
\tfrac12\mathrm{ad}_Y^2 X + \dots) \in \mathfrak h` by the first
condition, while :math:`\mathrm{Ad}_s` preserves :math:`\mathfrak h`
because :math:`H^0` is normal in :math:`H`. So the second condition on
:math:`R_H` implies it on all of :math:`H`.

⭐ **The five per-family arms are corollaries, and this is where they
went.** Read :eq:`manifold-normaliser-lie-criterion` at each family and
the retired arms fall out of it, verbatim:

.. list-table:: `[M]` 2026-09-03 — the one criterion, evaluated per family
   :header-rows: 1
   :widths: 24 46 30

   * - the case
     - what :eq:`manifold-normaliser-lie-criterion` reduces to
     - measured
   * - :math:`H` FINITE (:math:`\mathfrak h = 0`)
     - the bracket condition is vacuous and
       :math:`X - \mathrm{Ad}_s X = 0` says :math:`s` COMMUTES with
       :math:`X` — one matrix, one commutator per element, which is
       precisely the retired arm
     - agrees with a direct commutator test on **51 of 51** (axis ×
       finite-family) rows
   * - :math:`H = O(2)_b` or :math:`SO(2)_b`
     - :math:`[[\hat a]_\times, [\hat b]_\times] = [\hat a \times
       \hat b]_\times` lies in the LINE of :math:`[\hat b]_\times` only
       when :math:`\hat a \times \hat b = 0`, i.e.
       :math:`\hat a \parallel \hat b`; then
       :math:`X - \mathrm{Ad}_{\sigma_v} X = 2X \in \mathfrak h`
       (an improper :math:`g` sends :math:`[v]_\times` to
       :math:`-[gv]_\times`)
     - ``normalises`` equals *"the axes agree"* on **9 of 9**
       (:math:`a`, :math:`b`) pairs, for both :math:`SO(2)_b` and
       :math:`O(2)_b`
   * - :math:`H = D_{\infty h}`
     - the same, with the family's axis fixed at :math:`z`
     - :math:`SO(2)_a` normalises it for :math:`a = z` only
       (``True``/``False``/``False`` on :math:`z`/:math:`x`/:math:`y`)
   * - :math:`G^0 = SO(3)`, :math:`H` finite
     - all three generators must commute with every :math:`s`, and only
       :math:`\pm I` commutes with all of :math:`\mathfrak{so}(3)` —
       so :math:`H \subseteq \{e, -I\}`, the centre of :math:`O(3)`
     - ``False`` on all 16 non-trivial finite members probed, ``True``
       on ``Trivial``, and ``True`` on :math:`\{e, -I\}` built directly
       as a :class:`~orpheus.numerics.symmetry.Realization` — the
       positive leg, which has no named member to be asked through
   * - :math:`H = \{e\}`, :math:`SO(3)`, :math:`O(3)`
     - normal in :math:`O(3)`; every condition holds identically
     - ``True`` for every :math:`G`

⛔ **This subsection listed those five as the criterion until
2026-09-03**, in the shape the code then had — a five-arm dispatch in
``symmetry._identity_component_normalises``, each arm resting on one or
two tests, one of them (`[M]` qa's per-arm census) invoked exactly ONCE
across 670 tests. Every arm's *answer* was right, which is why the
prose survived the check that mattered: `[M]` over the full
:math:`27\times27` grid, the one Lie criterion and the five arms agree
on **729 of 729** ordered pairs
(:ref:`manifold-realization`). What changed is that four of the five
were unfalsifiable-in-practice branches and are now one expression with
a proof above it.

Finally the disconnected families are closed by their **coset
representatives** —
:math:`O(2)_a = SO(2)_a \sqcup SO(2)_a\sigma_v`,
:math:`D_{\infty h} = SO(2)_z \sqcup SO(2)_z\sigma_h \sqcup
SO(2)_z\sigma_v \sqcup SO(2)_z C_2'`,
:math:`O(3) = SO(3) \sqcup SO(3)(-I)`. Exact on the identity component
and finite on the representatives is exact on the whole group.

.. list-table:: `[M]` 2026-09-02 — ``A.normalises(B)``, a selection
   :header-rows: 1
   :widths: 16 16 12 56

   * - :math:`A`
     - :math:`B`
     - Answer
     - Why
   * - :math:`D_{2h}`
     - :math:`\sigma_y`
     - ``True``
     - every element of :math:`D_{2h}` sends :math:`\hat e_y` to
       :math:`\pm\hat e_y`
   * - :math:`C_4` (about :math:`z`)
     - :math:`\sigma_y`
     - ``False``
     - the quarter turn conjugates :math:`\sigma_y` to
       :math:`\sigma_x`
   * - :math:`SO(2)_x`
     - :math:`\sigma_x`
     - ``True``
     - :math:`\sigma_x` commutes with :math:`[\hat e_x]_\times`
   * - :math:`SO(2)_x`
     - :math:`\sigma_y`
     - ``False``
     - it does not
   * - :math:`SO(2)_x`
     - :math:`O(2)_x` / :math:`O(2)_y`
     - ``True`` / ``False``
     - the axial pair agrees only on a shared axis
   * - :math:`D_{\infty h}`
     - :math:`O(2)_z` / :math:`O(2)_x`
     - ``True`` / ``False``
     - the same, with :math:`z` fixed by the family
   * - :math:`SO(3)`
     - :math:`\{e\}` / :math:`\sigma_x`
     - ``True`` / ``False``
     - a finite :math:`H` must sit inside :math:`\{e, -I\}`
   * - :math:`O_h`
     - :math:`\sigma_x` / :math:`C_4`
     - ``False`` / ``False``
     - :math:`O_h` contains the :math:`x \leftrightarrow y` exchange,
       which moves both

.. _manifold-normaliser-sampling-control:

⭐ The control — and the sampled criterion over-certifies HERE too
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two probes, on the same eight :math:`(G, H)` pairs with :math:`G`
continuous, both run 2026-09-02 on the working tree.

**The positive control.** Conjugate by ten rotations of :math:`G`'s
identity component at angles pairwise incommensurate with :math:`\pi`
(:math:`1`, :math:`\sqrt 2`, :math:`e`, :math:`2.5`, :math:`\sqrt 7`,
:math:`\pi/3 + 0.1`, :math:`0.37`, :math:`1.913`, :math:`e/3`,
:math:`5.1`) and ask :meth:`is_normalised_by
<orpheus.numerics.symmetry.SubgroupOfO3.is_normalised_by>` of each; the
conjunction must equal the exact answer. `[M]` **8 of 8 pairs agree**,
:math:`10` angles each. Without this the exact criterion's ``False``
answers would be indistinguishable from a criterion that is merely
strict.

**The negative control, and it is the interesting one.** Replace the ten
angles by the four RIGHT angles :math:`\{0, \pi/2, \pi, 3\pi/2\}` — the
sample ERR-072 was built from. `[M]` the answer changes on **2 of the 8
pairs**, and in the flattering direction both times:

.. list-table:: `[M]` 2026-09-02 — the four-right-angle sample vs the exact criterion
   :header-rows: 1
   :widths: 26 22 22 30

   * - :math:`(G, H)`
     - right-angle sample
     - exact
     - reading
   * - :math:`(SO(2)_x,\ D_{2h})`
     - ``True``
     - ``False``
     - the quarter turns permute
       :math:`\{\sigma_y, \sigma_z\}` and land back in
       :math:`D_{2h}`; a generic angle does not
   * - :math:`(SO(2)_z,\ D_{2h})`
     - ``True``
     - ``False``
     - the same, one axis over
   * - the other six pairs
     - agree
     - agree
     - :math:`\sigma_x, \sigma_z, O(2)_a, C_4` under their own axes

⟹ ERR-072's mechanism is not confined to ``is_invariant_under``: it recurs in
every predicate about a continuous group, and it recurs with the SAME
signature — a sample that happens to generate a finite subgroup
containing the answer. That is why
:meth:`IdentityComponent.normalises
<orpheus.numerics.symmetry.IdentityComponent.normalises>` evaluates the
Lie condition :eq:`manifold-normaliser-lie-criterion` — a bracket and a
finite set of :math:`X - \mathrm{Ad}_s X` memberships — rather than
conjugating by any sample of group elements at all. On a finite
:math:`H` that expression *is* the commutator against
:math:`[\hat e_a]_\times`; the point is that it is derived from the
criterion rather than written down as a family rule, so the same
protection extends to the cases nobody enumerated.

⚠ This paragraph named ``symmetry._identity_component_normalises`` until
2026-09-03, when #434 R1 dissolved that helper's five per-family arms
into the one criterion above. The reasoning is unchanged, the measured
2-of-8 over-certification is unchanged, and `[M]` the two
implementations agree on 729 of 729 ordered pairs
(:ref:`manifold-realization`) — only the name and the derivation moved.

.. _manifold-lift:

The lift — a right inverse of the quotient map, and it is ONE formula
-----------------------------------------------------------------------

:eq:`manifold-normaliser-descent` is about points of the BASE, and a
measure's nodes are not always points of the base: a fold carries
representatives (base coordinates, width 3) while a polar marginal
carries :math:`\mu` (chart coordinates, width 1). The arrow that closes
the gap is :attr:`Quotient.lift
<orpheus.numerics.manifold.Quotient.lift>`, a right inverse of the
quotient map — :math:`\pi \circ \lambda = \mathrm{id}` on the
:attr:`~orpheus.numerics.manifold.Quotient.realization`.

⭐ **It is a FIELD, for the reason** ``orbit_coordinates`` **is one: the
derivation emits it.** Since 2026-09-03 (#434 R4) an entry carries
:attr:`~orpheus.numerics.manifold.Quotient.lift_coordinates` — the
lift's action on chart coordinates — beside
:attr:`~orpheus.numerics.manifold.Quotient.lift_codomain`, the manifold
it lands on, and ``lift`` assembles the typed arrow on top of them
exactly as :attr:`~orpheus.numerics.manifold.Quotient.quotient_map`
does. Both fields are **required**, so a seventh entry cannot forget the
lift. The MAP is ``field(compare=False, repr=False)`` — a function has no
value equality — while the CODOMAIN is compared like every other
:class:`Manifold`-valued slot, for a measured reason
(:ref:`manifold-dimension-drop`): with it excluded,
``dataclasses.replace(entry, lift_codomain=SPHERE)`` compared EQUAL to
the catalogue entry, and :func:`barycentre`'s memo then answered for both
— ERR-080's own shape re-minted by the field built to refuse it.

.. list-table:: `[M]` 2026-09-03 — one formula, eight entries, and the law
   :header-rows: 1
   :widths: 22 36 22 20

   * - Entry
     - ``lift_coordinates``
     - Typed arrow
     - :math:`\max\lvert \pi(\lambda(x)) - x\rvert`
   * - :math:`S^2/O(2)_a` (axial, 3)
     - :math:`\mu \mapsto \mu\,\hat e_a` — the centre of the
       constant-:math:`\mu` circle
     - ``S^2/O2_a → D^3``
     - ``0.000e+00``
   * - :math:`S^2/\sigma_a` (mirror, 3)
     - :math:`(x_b, x_c) \mapsto (0, x_b, x_c)` — the midpoint of
       :math:`\{p, \sigma_a p\}`
     - ``S^2/sigma_a → D^3``
     - ``0.000e+00``
   * - :math:`M/\{e\}` (2 constructible)
     - the identity — the chart IS the base
     - ``M/Trivial → M``
     - ``0.000e+00``

Every row is the same map: :math:`P_H`, the Reynolds projector onto
:math:`H`'s fixed subspace, read from the chart's side
(:eq:`manifold-reynolds-projector`). Where a family used to supply its
own arm, ``_coordinate_chart(columns, ambient)`` now returns the chart
and its lift **as a pair**, and the two sphere builders call it with
their invariant columns — so the entry's two coordinate maps are spelled
once each and cannot disagree.

⭐ **Every arm's DOMAIN is the entry, and the codomain is a field.** A
lift consumes chart coordinates, so its domain is the orbit space itself
rather than the :attr:`~orpheus.numerics.manifold.Quotient.realization`
the chart coordinates are *valued in* — the same ruling that made
:attr:`~orpheus.numerics.manifold.Quotient.quotient_map`'s codomain the
entry and never the interval (:ref:`manifold-quotient-map`). What
changed at R4 is that the codomain stopped being a per-arm literal and
became :attr:`~orpheus.numerics.manifold.Quotient.lift_codomain`, which
a consumer READS to learn which it was handed: :math:`D^3` for both
sphere families, the base itself where the lift is the identity.

.. note::

   ⛔ **The mirror entry's lift was a hemisphere SECTION until
   2026-09-03, and retiring it is the load-bearing half of R4.** Until
   then ``Quotient.lift`` was a three-arm branch on the group's tag —
   barycentre for an axial group, section for a mirror, identity for the
   trivial one — whose fall-through raised *"add the entry's section (or
   its equivariant barycentre) to Quotient.lift"*: a second dispatch
   over the very key the catalogue had already used to choose the
   builder, and one a seventh entry could silently omit. The mirror arm
   computed

   .. math::

      (x_b, x_c) \;\longmapsto\;
      \sqrt{1 - x_b^2 - x_c^2}\,\hat e_a + x_b\hat e_b + x_c\hat e_c ,

   the representative on the closed hemisphere :math:`x_a \ge 0`, and it
   carried the machinery that a square root forces: a
   :math:`\rho^2 > 1 + 10^{-12}` refusal for a chart point outside the
   closed disk, and a ``np.maximum(0.0, ·)`` clamp under the root. All
   of it is gone — ``_hemisphere_section``, its literal and its
   ``sqrt`` — because the projector needs none of it and the induced
   action never wanted a representative.

   ⟹ **What was lost, stated precisely.** The section lands ON
   :math:`S^2` and the barycentre does not, so a consumer that needs a
   *direction* out of a mirror entry's chart no longer gets one from the
   lift. `[M]` 2026-09-03 no shipped consumer does: the lift's readers
   are :meth:`~orpheus.numerics.manifold.Quotient.orbit_barycentres` and
   :func:`~orpheus.numerics.manifold.barycentre`, and every consumer
   downstream of those asks an ORBIT question. The entry's
   :attr:`~orpheus.numerics.manifold.Quotient.fundamental_domain` still
   ships and still carries the hemisphere — it is what
   :meth:`Quotient.contains
   <orpheus.numerics.manifold.Quotient.contains>` validates a fold's
   representatives against (:ref:`manifold-two-coordinate-systems`) — so
   the section's *image* survives as a membership test; what retired is
   the *map into it*.

⭐ **A section is more than an induced action asks for, and the
barycentre has exactly what it asks for.** The requirement is
equivariance under the normaliser, and :math:`P_H` is equivariant
because it is canonical:
:math:`g P_H g^{-1} = P_{gHg^{-1}}`, so :math:`g` normalising
:math:`H` gives :math:`P_H g = g P_H` — an isometry carrying
:math:`H`-orbits onto :math:`H`-orbits carries their means onto their
means (:ref:`manifold-barycentre-equivariance`, with the measured
negative leg). No choice of representative enters, which is why the
axial family — where no representative is canonical at all — is served
by the same formula as the mirror family, where one would have been.

:meth:`Quotient.orbit_barycentres
<orpheus.numerics.manifold.Quotient.orbit_barycentres>` is the entry
point: it tells the two honest coordinate systems apart by WIDTH (the
same discrimination :meth:`Quotient.contains
<orpheus.numerics.manifold.Quotient.contains>` already makes, see
:ref:`manifold-two-coordinate-systems`) and returns the orbit barycentre
for **both** — a base-width point through
:math:`\lambda \circ \pi = P_H`, a chart-width one through
:math:`\lambda`.

.. note::

   ⚠ **This method has been renamed twice in two days, and both renames
   are the same lesson.** It shipped on 2026-09-02 as
   ``section_coordinates``, was renamed within about two hours to
   ``ambient_representatives`` at that step's elegance review, and became
   ``orbit_barycentres`` at #434 R4 on 2026-09-03.

   * ``section_coordinates`` was **ERR-080's own defect one level up**: a
     *section* is a choice of representative — a point OF the base — and
     the axial arm cannot return one, so the method promised a codomain
     it does not land in.
   * ``ambient_representatives`` fixed the codomain claim and kept a
     weaker version of the same fault. It promised a *representative*,
     and it delivered one only on the ambient-width path, where it
     **passed the points through unchanged**; on the chart-width path it
     lifted to a barycentre. Two things under one name, with a ⚠ in its
     own docstring retracting the promise for the axial entry — the tell
     that the name was carrying a disjunction.
   * ``orbit_barycentres`` names what every path returns. R4 also made
     that true: the ambient-width path no longer passes through, it
     projects, so `[M]` a fold's :math:`(x, y, z)` nodes come back as
     :math:`(x, 0, z)`.

   ⟹ **A name that must be qualified per argument is a disjunction
   wearing a noun.** Both renames were caught on the name alone, before
   any consumer read it.

.. _manifold-lift-is-mode-12-blind:

The kernel cannot see this carve — so the gates are at the AMBIENT tier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

⛔ **Everything the invariance machinery computes is invariant under the
change R4 makes**, and this is a design constraint rather than a
footnote. ``_orbit_space_closure``, ``is_invariant_under``,
``symmetry_groups``, ``ordinate_permutation`` and
``permutation_under`` all read ``orbit_coordinates(...)`` of whatever
the lift returns — and ``orbit_coordinates`` is exactly the column
selection that :math:`P_H` re-writes. Since :math:`\pi` annihilates the
columns :math:`\lambda` zeroes,

.. math::

   \pi\bigl(g \cdot P_H p\bigr) \;=\; \pi\bigl(g \cdot p\bigr)
   \qquad\text{for every } g \text{ in the normaliser,}

so no answer downstream of the chart can move. `[M]` 2026-09-03, the R4
lift semantics installed and the full behaviour grid of #434 R1
re-captured: **0 of 9925 answers moved** (31 groups; ``contains`` 961,
``normalises`` 961, ``is_normalised_by`` 31×240, ``orbit_stabiliser`` /
``identity_component`` / ``dim`` 93, ``is_invariant_under`` 31×15, the walk
5). And `[M]` directly, on 200 seeded unit vectors × three normalising
motions × the axial and mirror entries, ``chart(g·P p)`` is
``array_equal`` to ``chart(g·p)`` on every row, ``max|Δ| = 0.000e+00``.

⟹ **Three consequences, each of which shapes what R4 can be verified
by.** (i) No end-to-end kernel row may be credited as a catcher for this
carve — there are no reds to credit, and a green there is
``vv-principles`` #12 by construction, not evidence. (ii) The gates must
assert on the lift's OWN output, in the base's ambient coordinates,
never through the chart. The discriminator between the retired
hemisphere section and the projector is :math:`O(1)` there, and it is
exactly computable rather than sampled: the two maps agree on the kept
columns and differ in column :math:`a` alone, where the section writes
:math:`\sqrt{1-\rho^2}` and the projector writes :math:`0`, so the gap
IS :math:`\lvert x_a\rvert` of the original direction — supremum
**exactly 1**, attained at the pole. `[M]` 2026-09-03 over 41 seeded
unit vectors the sampled maximum is ``9.748e-01 / 9.932e-01 /
9.953e-01`` on :math:`\sigma_x/\sigma_y/\sigma_z` (a draw, approaching
the bound), while through the chart the two are ``np.array_equal`` on
all three — **exactly zero**, not small. (iii) The round-trip
:math:`\pi \circ \lambda = \mathrm{id}` is a **declared-blind** leg:
it holds for the hemisphere section and for the projector alike, so it
ships labelled as such, with the teeth in
:math:`\lambda \circ \pi = P_H`.

⭐ **One answer DOES move, and it is a strengthening.** The reference
harness ``tests/_harness/references.py`` builds a mirror partner map by
negating a column of ``_embedded_nodes(quad.measure)``. On a fold, that
array is now the barycentres, whose :math:`y` column is identically
zero, so :math:`\sigma_y` maps the set to itself and the harness returns
the **identity permutation** where it used to RAISE — `[M]` 2026-09-03,
with the pre-R4 pass-through re-installed over
:meth:`~orpheus.numerics.manifold.Quotient.orbit_barycentres` in the
same interpreter, ``folded_product(2,4)`` and ``(4,8)`` raise *"a node's
mirror image misses the node set by 1.155e+00 / 1.189e+00"* and return
``Permutation([0 … N−1])`` after. That is the answer
:meth:`~orpheus.numerics.quadrature.directional.Quadrature.ordinate_permutation`
has given since tracker 2.2b
(:ref:`manifold-ordinate-permutation-orbit-space`), so the reference and
production now AGREE about a fold's spent mirror where they used to
contradict each other. `[M]` 2026-09-03, no committed assertion moves with it, measured two
ways. Over 11 shipped rules × 3 axes (both 1-D rules, three product
rules, two Lebedev, two level-symmetric, two folds), the harness's
answer — permutation or refusal, compared cell by cell against the
pre-R4 pass-through re-installed in the same interpreter — is unchanged
on **31 of 33** rows, and the two that move are exactly the two folds
at ``axis="y"``. And a runtime census over the harness's six consumer
test modules (a spy wrapping ``mirror_partner_indices``, 327 passed /
1 xfailed) shows **62** calls over four ``(support, axis)`` cells —
:math:`S^2` × x (19), :math:`S^2` × y (15), :math:`S^2/O(2)_x` × x
(24), and :math:`S^2/\sigma_y` × **x** (4). So no consumer asks a fold
about :math:`\sigma_y` at all: the one answer R4 moves is one the
committed suite never reads. The
projection is injective on every fold's node set — `[M]` minimum chart
separation ``1.155e+00 / 4.403e-01 / 2.751e-01 / 1.510e-01`` at
``folded_product`` (2,4)/(4,6)/(4,8)/(8,8) — so no two orbits collapse
into one.

⚠ **And it costs the harness one independence claim, per support.** That
module's ``vv-principles`` #22 note says its reference *"no longer
cross-checks the EMBEDDING convention — it cross-checks the
PERMUTATION"*. On the **fold** rows that is now more true than it was:
``_embedded_nodes`` routes through ``orbit_coordinates`` and
``lift_coordinates``, which is the same pair ``_orbit_space_closure``
reads, so the two sides share the chart. `[M]` on the 9 non-fold rules
nothing is shared that was not shared before (their supports are
:math:`S^2` or a chart-width :math:`S^2/O(2)_x`). The note is therefore
**support-scoped**, not global, and R4 re-worded it that way in the
harness itself — a fold row is not an independent check of the lift, and
saying so is what stops the identity permutation being read as
corroboration. A sibling comment in
``tests/sn/sweep/curvilinear/test_coupled_pole_mu_level_invariant.py``
records the ``vv-principles`` #20 consequence on the same change: on the
two folded rows of that gate the :math:`\mu_y` column is identically
zero (`[M]` :math:`\lvert\mu_y\rvert_{\max}` ``8.7e-01`` /
``9.1e-01`` → ``0.0``), so *"and* :math:`\mu_y` *is held"* is vacuous
there and is carried by the :math:`\mu_z` leg and the six unfolded
rows.

.. _manifold-induced-action:

``Quotient.induced_action`` — the arrow, and the refusal
----------------------------------------------------------

:meth:`Quotient.induced_action(motion)
<orpheus.numerics.manifold.Quotient.induced_action>` returns the typed
arrow :math:`M/H \to M/H` of :eq:`manifold-normaliser-descent`: section
coordinates in (chart-width input is lifted first), chart coordinates
out — the same convention as
:attr:`~orpheus.numerics.manifold.Quotient.quotient_map`. When the
motion is outside the normaliser it raises, naming the theorem.

.. list-table:: `[M]` 2026-09-02 — ``SPHERE.quotient(Mirror("y")).induced_action(g)``
   :header-rows: 1
   :widths: 22 20 58

   * - :math:`g`
     - Verdict
     - What it does on the disk chart :math:`(x, z)`
   * - :math:`\sigma_x`
     - admitted
     - :math:`(x, z) \mapsto (-x, z)` — `[M]` on the representative
       :math:`(0.6, 0, 0.8)`, image ``[-0.6, 0.8]``
   * - :math:`\sigma_y`
     - admitted
     - the IDENTITY on the chart — it is the group that was spent
   * - :math:`\sigma_z`
     - admitted
     - :math:`(x, z) \mapsto (x, -z)`
   * - :math:`C_2` about :math:`z`
     - admitted
     - :math:`(x, z) \mapsto (-x, z)`
   * - :math:`C_4` about :math:`z`
     - ⛔ **refused**, ``ValueError``
     - it conjugates :math:`\sigma_y` to :math:`\sigma_x`, so
       :math:`[g\,p]` depends on which of the two representatives
       arrived

The refusal is the section's §6c witness: an input that exists in the
tree today and that the new guard rejects. It is also the reason the
guard is asked of the GROUP
(:meth:`~orpheus.numerics.symmetry.SubgroupOfO3.is_normalised_by`) rather
than re-derived here: a group's normaliser is the group's own business,
and re-deriving it in this module would be a Cardinal-Rule-2 twin.
⛔ **A second reason stood beside that one until 2026-09-03 and has
expired**: the call was also *duck-typed*, because this module could not
name :class:`~orpheus.numerics.symmetry.SubgroupOfO3` at runtime at all.
It can now (:ref:`manifold-import-cycle`), so the ownership argument is
the whole of the argument — which is the stronger position, since it does
not depend on an import direction that a later carve can reverse.

.. _manifold-one-invariance-kernel:

ONE invariance kernel — a measure is asked ON its own support
---------------------------------------------------------------

The question is the MEASURE's, and since R2 of #434 (2026-09-03) so is
the verb: :meth:`DiscreteMeasure.is_invariant_under
<orpheus.numerics.measure.DiscreteMeasure.is_invariant_under>` routes
every measure through a single kernel,
:func:`~orpheus.numerics.invariance.is_invariant_under`, in **three**
conjuncts, cheapest and most decisive first. (Why the verb sits on the
measure and the kernel in a module of its own is
:ref:`discrete-measure-invariance-module`; what follows is the
mathematics it evaluates.)

1. :math:`G` must NORMALISE :math:`H` (:meth:`normalises
   <orpheus.numerics.symmetry.SubgroupOfO3.normalises>`) — otherwise it
   does not act on :math:`M/H` at all, and the answer is ``False``
   because the question is ill-posed, not because the nodes fail it.
2. For a CONTINUOUS :math:`G` the elements are infinite, so its identity
   component :math:`G^0` is decided by structure: a connected orbit
   inside a FINITE set is a point, so :math:`G^0` must FIX every node's
   orbit barycentre. That is :meth:`IdentityComponent.fixes
   <orpheus.numerics.symmetry.IdentityComponent.fixes>` — the condition
   :math:`Xp = 0` for every generator, which reads *"on the axis"* for a
   torus and *"at the origin"* for :math:`SO(3)` — the axis-support (or
   origin) rule the ambient arm has applied since ERR-072, stated once in
   one body rather than twice in two (``_is_axis_supported`` and
   ``_is_origin_supported`` retired at #434 R1, 2026-09-03). It is exact
   on the barycentre because the chart is injective on orbits, so
   :math:`G^0` fixes :math:`P_H p` iff it fixes :math:`[p]`.
3. Every element (finite :math:`G`) or coset representative (continuous
   :math:`G`) must permute the weighted node set **in chart coordinates**
   through its induced action
   (:func:`~orpheus.numerics.invariance._orbit_closure`, with ERR-073's
   bijection guard and ERR-074's no-bare-``argmin`` guard). ⭐ Since R1
   those are the SAME slot — the kernel iterates
   :attr:`Realization.representatives
   <orpheus.numerics.symmetry.Realization.representatives>` and a finite
   group's representatives ARE its elements
   (:ref:`manifold-realization`), so conjunct 3 has one arm where the
   prose reads as two cases.

.. _manifold-kernel-deleted-short-circuit:

The step that was deleted, and the step that had to stay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

⛔ **This list read FOUR steps until 2026-09-03, and step 2 was**
:math:`G \subseteq H \Rightarrow \texttt{True}`. The reasoning was
correct — a group inside :math:`H` acts trivially on :math:`M/H`, so
every one of its elements fixes every orbit — and that is exactly why it
could go: **the closure re-proves it.** An element that acts trivially
induces the identity permutation, which the closure finds like any other.
The branch was an OPTIMISATION wearing the shape of a guard, and a branch
that can only ever agree with the body beneath it is a second place for
one answer to live.

`[M]` 2026-09-03, over the 11 shipped rules × their own candidate sets:
the short circuit would have fired on **28** (rule × group) rows, and on
all 28 the general body returns ``True`` through the closure —
**0 disagreements**. Widened to a fixed 30-spelling group list × the same
11 rules, the predicate moved **0 of 330** rows against a pinned
``git archive HEAD`` tree (probe: build both trees' answers for the same
``(rule, group.name)`` keys and diff them). That is what licenses reading
the deletion as a *theorem being re-proved* rather than as a behaviour
change nobody measured.

⚠ **Do not run the same argument on conjunct 1 — the asymmetry is the
whole point, and it points at ERR-072.** For a FINITE :math:`G` the
closure's own per-motion guard — "does this isometry normalise
:math:`H`?", asked before any node is touched — really does re-prove
normalisation, because the representatives it iterates ARE the elements.
For a CONTINUOUS :math:`G` they are not: the representatives cover the
components, never :math:`G^0` itself, so nothing downstream would ever
ask whether :math:`G^0` normalises :math:`H`. Deleting conjunct 1 would
therefore be a silent wrong ``True`` on exactly the family ERR-072 is
about — a continuous group certified from a finite sample of it. The
``component`` half of :meth:`SubgroupOfO3.normalises
<orpheus.numerics.symmetry.SubgroupOfO3.normalises>` is the only check
that :math:`G^0` acts at all.

⭐ **And the kernel still has no fast path.** A ``Trivial`` / ``Cn(1)``
short-circuit stood in front of the whole list until #434 R1; removing it
was the same theorem one tier down. :math:`\{e\}` normalises every
stabiliser, so conjunct 1 passes; it is finite, so conjunct 2 is skipped;
and its single element is the identity, whose induced action is the
identity permutation, so conjunct 3 passes. `[M]` 2026-09-03 the trivial
group answers ``True`` on **11 of 11** shipped rules through the general
body, with no fast path in front of it.

.. _manifold-kernel-position-test:

The position test runs for EVERY continuous group now
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

⛔ **This paragraph read** *"the POSITION test runs only when* :math:`H`
*is finite; on an axial entry it would be a tautology"* **until
2026-09-03, and the guard that implemented it is gone.** The old body
short-circuited the test with
:math:`\mathfrak h \supseteq \mathfrak g`, on the argument that an axial
entry's barycentre :math:`\mu\,\hat e_a` lies on axis :math:`a` by
construction, so *"is every node on the axis?"* could not answer anything
there. The observation is true and the guard was unnecessary: a test that
cannot fail **passes**, which is the answer the theorem says it must give.
Deleting it removes a second place for the :math:`G^0 \subseteq H` case
to be decided, at the cost of a dot product per node.

What survives from that paragraph is the reading it was written to
protect, and it is worth keeping in view: on an axial entry conjunct 2
carries **no information** — ``vv-principles`` #19 at the kernel rather
than at a gate — so the row that decides such a case is conjunct 3, and a
gate that exercises only axial entries is not a gate on conjunct 2.
`[M]` on ``gauss_legendre_on_polar_orbit(8, "z")`` (support
:math:`S^2/O(2)_z`), :math:`H \supseteq SO(2)_z = G^0` while
:math:`H \not\supseteq D_{\infty h}`, so :math:`D_{\infty h}` reaches
conjunct 2, passes it by construction, and is admitted by conjunct 3 —
``True``, unchanged across the carve.

⚠ **The window changed with the guard, and in the direction that makes
the two windows agree.** The position test now runs at the NODE window,
:data:`~orpheus.numerics.invariance._NODE_WINDOW_FACTOR` (= 100) times
the weight window ``atol``; until R2 it ran at ``atol`` itself while the
node match one conjunct later already used 100× it, so one kernel asked
the same *"is this point where I think it is?"* question at two
tolerances. `[M]` 2026-09-03 the change is inert on every shipped rule: the three
polar marginals are the only ones the test even reaches, and their
off-axis residual :math:`|\hat a \times p|` is ``0.000e+00`` on all
three, so nothing production builds lies between :math:`10^{-13}` and
:math:`10^{-11}`. That is why the gate for it carries a MANUFACTURED
witness rather than a shipped one — a window change with no shipped
input between the two windows is unfalsifiable by the roster
(``vv-principles`` #17's granularity trap, at a tolerance).

**A measure that names no orbit space is asked on the TRIVIAL one.** A
bare :class:`~orpheus.numerics.manifold.Sphere` support, or a
chart-level :class:`~orpheus.numerics.manifold.Interval`, is handed
:math:`\mathbb{R}^3/\{e\}` — `[M]` 2026-09-03
``_ambient_orbit_space().name`` is ``'spatial_R3/Trivial'``, ``dim`` 3.
Its chart is the base, its lift is the identity, and every isometry
descends to it, so the kernel *reduces* on it to the ambient question the
tree asked before, with the same zero-padding convention for
lower-dimensional nodes
(:func:`~orpheus.numerics.invariance._embedded_nodes`). There is one
kernel and there are not two readings that could disagree.

⚠ **The base is** :math:`\mathbb{R}^3` **and not the sphere, on
purpose** — a zero-padded interval or planar rule, and every barycentre,
lands OFF :math:`S^2`, and a container must honestly contain what is put
in it. That is this chapter's own subject applied to itself: naming
:math:`S^2` there would be the ERR-080 forgery one more time, in the
kernel that exists to make it unspellable.

⭐ **The reduction is measured, not asserted.** `[M]` 2026-09-03, over
six shipped sphere rules (``product(4,8)``, ``product(3,5)``,
``lebedev(5)``, ``lebedev(11)``, ``level_symmetric(4)``,
``level_symmetric(8)``) × every group
:func:`~orpheus.numerics.invariance.candidate_groups` offers each of them
(22 / 18 / 22 / 24 / 26 / 32), replacing ``support`` with the trivial
quotient leaves the answer identical on **144 of 144** (rule × group)
rows. ⛔ This reading was recorded as *"150 of 150"* on 2026-09-02; the
finding is unchanged and only the DENOMINATOR moved, because it is the
size of the candidate set and the candidate set has been re-derived twice
since (R1's one-spelling merge, and R2 reading the azimuth count off the
orbit barycentres). A denominator that is a *computed* set is a number
with a shelf life; the row it summarises is not.

.. _manifold-gamma-slot:

The registry's ledger — a geometry admits a rule on a FOLD of its domain
--------------------------------------------------------------------------

.. note::

   **The anchor's name is a fossil, kept deliberately.**
   ``manifold-gamma-slot`` was minted on 2026-09-02, when the registry
   recorded TWO facts per geometry and called the second one
   :math:`\Gamma`. R3 of #434 (2026-09-03) split the ledger into THREE
   and moved :math:`\Gamma` onto a *different* one — see the warning
   below. The anchor stays because a cross-document ``:ref:`` that
   dangles renders as plain text with no warning at any severity, and
   this one has citers on three pages; the LETTERS used from here on are
   the current ones.

**The consumer, and the three facts it records.** A geometry's angular
symmetry is not one undifferentiated thing, and since R3 of #434 the
registry does not store it as one.
:class:`~orpheus.numerics.quadrature.registry.AngularSymmetry` records
three groups, because a quadrature is asked three different questions
and each is answered by a different one:

.. list-table::
   :header-rows: 1
   :widths: 10 12 38 40

   * - Symbol
     - Field
     - What it is
     - What it decides
   * - :math:`K`
     - ``spent``
     - the stabiliser the dimensional reduction integrated away
     - the angular DOMAIN :math:`\mathcal{D} = S^2/K`
       (:attr:`AngularSymmetry.support
       <orpheus.numerics.quadrature.registry.AngularSymmetry.support>`)
       **and** the reference measure on it — one fact, read off one
       catalogue entry (:ref:`manifold-second-twin-reference`)
   * - :math:`\Gamma`
     - ``unspent``
     - the FINITE symmetry the solution still HAS, in the geometry's own
       local frame
     - which FOLDS stage 0 admits: a fold may spend only what the
       solution has
   * - :math:`R`
     - ``owed``
     - the reflection closure a reflecting face still needs
     - stage 1 — the node set must be closed under it, realized as a
       permutation of the ordinates

The per-geometry DERIVATION of those three — why a cylinder's is
:math:`(\{e\},\ D_{1h},\ D_{2h})` and a z-uniform Cartesian plane's is
:math:`(\{e\},\ \sigma_z,\ D_{2h})`, read off each geometry's own
transport equation — belongs to the selection algorithm and lives at
:ref:`quadrature-selection-algorithm`, with the ledger table itself in
:mod:`orpheus.numerics.quadrature.registry`'s module docstring. This
page owns the LATTICE half: what the two stages ask of the point-set
layer, and why each question is decided rather than declared.

.. warning::

   ⚠ **One letter, two bindings, and the older one is the OPPOSITE
   half.** Until 2026-09-03 this page and the registry both wrote
   :math:`\Gamma` for the closure a reflecting face is OWED, and paired
   it with :math:`G^0` for the spent half as though the two were the
   factorisation :math:`G = G^0 \cdot (G/G^0)`. Both halves of that were
   wrong, and each in its own way.

   * The pair was never a factorisation. `[M]` 2026-09-03 the slab and
     the sphere spend :math:`O(2)_x`, which is DISCONNECTED
     (:math:`\dim = 1`, two components), so it is not any group's
     identity component and there is no :math:`G` for which the two
     recorded entries are :math:`G^0` and :math:`G/G^0`.
   * The owed closure was doing a SECOND job it was never a statement
     about — licensing folds. That is the defect R3 repaired, and it is
     catalogued as **ERR-081**
     (:doc:`/theory/verification/error_catalog`).

   From 2026-09-03, :math:`\Gamma` is the UNSPENT symmetry and the owed
   closure is :math:`R`. A sentence on any page that pairs
   :math:`\Gamma` with :math:`G^0`, or that calls :math:`\Gamma` a
   *residual*, predates that split and describes the two-entry ledger.

.. _manifold-gamma-slot-stage-zero-section:

Stage 0 — the descent arrow, and the coverage test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A rule need not live on :math:`\mathcal{D}` itself. It may live on a
further quotient :math:`X = \mathcal{D}/H` — a FOLD — and the shipped
cylindrical rule does: ``folded_product`` halves the sphere by
:math:`\sigma_y`. So stage 0 asks two things, and neither is declared
anywhere; both are read off the orbit-space catalogue and the
containment lattice:

.. math::
   :label: manifold-gamma-slot-stage-zero

   \text{stage 0} \iff
   \bigl(S^2/K \twoheadrightarrow X\bigr) \ \text{exists}
   \ \wedge\ H \subseteq \Gamma K,
   \qquad
   \text{stage 1} \iff X \ \text{is } R\text{-invariant} ,

.. (vv-status rationale) manifold-gamma-slot-stage-zero: A statement of
   an admission PREDICATE — which rules a geometry's angular domain
   admits — not a solver claim, and it has no L0..L3 ladder slot. Its
   three conjuncts are the shipped bodies of
   ``AngularSymmetry.domain_refusal`` (whose ``is None`` is
   ``admits_domain``) and ``admits_symmetry``; the verifiable content is
   the registry's own selection gates in
   ``tests/numerics/test_registry.py`` —
   ``TestStageZeroIsTheDescentArrowPlusTheUnspentSymmetry`` for the two
   stage-0 conjuncts and
   ``TestStageOneOnAFoldAsksTheOrbitSpaceNotTheRepresentatives`` for
   stage 1 — plus the measured (rule x geometry) grid recorded under this
   label.
.. vv-status: manifold-gamma-slot-stage-zero documented

where :math:`X` is the manifold the rule's nodes live on and :math:`H`
is the group :math:`X` was quotiented by
(:attr:`DiscreteMeasure.quotient_group
<orpheus.numerics.measure.DiscreteMeasure.quotient_group>`, which is
:math:`\{e\}` for a bare rule).

.. admonition:: ⛔ The second conjunct read something else until
                2026-09-03, and the difference is ERR-081
   :class: warning

   From 2026-09-02 (#429 tracker 2.2b) to 2026-09-03 (#434 R3) the
   equation's second conjunct was

   .. math::

      \operatorname{spent}(\mathcal{D} \to X) \subseteq \Gamma_{\text{owed}} ,

   read through a module-level helper ``manifold.spent_group(source,
   target)`` that named what the descent arrow spends — :math:`\{e\}`
   for the identity, ``target.by`` for a fold of the base, and a
   ``NotImplementedError`` naming the missing work for the induced map
   between two quotients of one base, because what THAT arrow spends is
   a coset, not a subgroup.

   Two things were wrong with it, and only the first is a question of
   spelling.

   1. It asked about the OWED closure. A geometry owes
      :math:`R` so that a reflecting face is an exact ordinate
      permutation; whether the SOLUTION is even under a mirror is a
      different question with a different answer, and on a z-uniform
      Cartesian plane the two differ. `[M]` 2026-09-03 that plane owes
      :math:`D_{2h}` and its solution is even in :math:`\mu_z` only, so
      the criterion above admitted ``folded_product(4, 8)`` — a
      :math:`\sigma_y` fold — and `[M]` 2 of the 4
      :math:`(\operatorname{sign}\mu_x, \operatorname{sign}\mu_y)` sweep
      quadrants of that rule are EMPTY (all 16 nodes carry
      :math:`\mu_y \ge +0.194`). See ERR-081.
   2. It needed an equality short circuit in front of it, and the
      short circuit was load-bearing rather than an optimisation.
      Reading the fold group against the geometry's OWN domain refuses
      it: `[M]` the slab's own Gauss-Legendre rule lives on
      :math:`S^2/O(2)_x` with :math:`H = O(2)_x`, and
      :math:`\sigma_x \not\supseteq O(2)_x` — an infinite group cannot
      sit inside a finite one. Asking what the ARROW spends dodged that
      (the identity arrow spends :math:`\{e\}`), at the cost of a
      predicate with an arm that raises.

   ⭐ **The coverage test needs neither dodge**, which is the argument
   for it: `[M]` the slab's own rule reads
   :math:`O(2)_x \subseteq \{e\}\cdot O(2)_x` — TRUE, no special case —
   and every arm answers, so ``admits_domain`` is total. `[M]`
   2026-09-03 over 4 geometries × the 7 rules tabulated below there are
   **0** raises in 28 rows. ``spent_group`` is RETIRED with its
   ``NotImplementedError``; the *facts* its table recorded survive as the
   descent-arrow half of the equation above and are measured in the grid
   below.

.. _manifold-coverage-by-a-product-section:

The coverage theorem — a containment in a product SET is decided in two steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\Gamma K = \{\gamma k\}` is a **set**, not a group — :math:`\Gamma`
need not normalise :math:`K` — so "is :math:`H` inside it?" is not a
lattice query and :eq:`subgroup-of-o3-containment` does not answer it.
It is nonetheless *exactly* decidable, in two conjuncts, and the
derivation is the reason the predicate is total.

Write :math:`\Gamma = \{\gamma_1, \dots, \gamma_m\}` (finite, by the
construction invariant on
:class:`~orpheus.numerics.quadrature.registry.AngularSymmetry`) and
decompose both :math:`H` and :math:`K` as
:eq:`manifold-group-as-component-and-cosets` prescribes.

**Step 1 — the identity components.** :math:`\Gamma K =
\bigcup_{i} \gamma_i K` is a finite union of cosets of :math:`K`, each
of them closed. :math:`H^0` is connected and contains :math:`e`, so it
cannot meet two disjoint closed cosets and must lie inside the single
coset that contains :math:`e`. And if :math:`e \in \gamma_i K` then
:math:`\gamma_i \in K`, whence :math:`\gamma_i K = K`. So that coset IS
:math:`K`, and

.. math::

   H \subseteq \Gamma K \;\Longrightarrow\; H^0 \subseteq K^0 ,

a statement about Lie algebras, decided by
:meth:`IdentityComponent.contains
<orpheus.numerics.symmetry.IdentityComponent.contains>` — a torus in a
torus iff the axes are parallel, anything in :math:`\mathfrak{so}(3)`,
:math:`\{0\}` in everything.

**Step 2 — the coset representatives.** Given
:math:`H^0 \subseteq K^0 \subseteq K`, write
:math:`H = \bigsqcup_{r} r H^0` over :math:`H`'s representatives. Then
:math:`r H^0 \subseteq r K`, so

.. math::

   H \subseteq \Gamma K
   \;\Longleftrightarrow\;
   \forall r:\ r \in \Gamma K
   \;\Longleftrightarrow\;
   \forall r\ \exists \gamma \in \Gamma:\ \gamma^{-1} r \in K ,

decided by :meth:`Realization.contains_element
<orpheus.numerics.symmetry.Realization.contains_element>` over
:math:`\Gamma`'s :math:`m` elements. Both implications reverse at every
step, so the pair of conjuncts is the DEFINITION of the containment and
not an approximation of it — which is what makes
:meth:`SubgroupOfO3.is_subset_of_product
<orpheus.numerics.symmetry.SubgroupOfO3.is_subset_of_product>` a fifth
row of :ref:`the one-body table <manifold-one-body-per-question>`
rather than a new kind of predicate.

.. note::

   ⚠ **The unspent group must be FINITE, and that is a construction
   invariant rather than a runtime check at the call site.** Step 2
   enumerates :math:`\Gamma`'s elements, so a continuous :math:`\Gamma`
   has no decision procedure here.
   ``AngularSymmetry.__post_init__`` refuses one at construction —
   `[M]` 2026-09-03,
   ``AngularSymmetry(spent=Trivial, unspent=SO2("z"), owed=Trivial)``
   raises ``ValueError: AngularSymmetry.unspent must be a finite group
   (its elements are enumerated by the fold licence); SO2_z is
   continuous`` — which is what makes the totality of ``admits_domain``
   a THEOREM rather than a hope (Pattern 4: the illegal state is
   unrepresentable). :math:`K` may be continuous and routinely is: `[M]`
   the slab and sphere spend :math:`O(2)_x`.

⭐ **The product is not either factor**, which is why the general form
is not ceremony. `[M]` 2026-09-03,
:math:`O(2)_x \subseteq O_h \cdot SO(2)_x` is **True** while
:math:`O_h \supseteq O(2)_x` and :math:`SO(2)_x \supseteq O(2)_x` are
both **False** — because :math:`O(2)_x = SO(2)_x \sqcup \sigma_z
SO(2)_x` and :math:`\sigma_z \in O_h`. It is not an isolated witness:
`[M]` over the 21-member set
:math:`\{` ``Trivial``, :math:`SO(3)`, :math:`O(3)`, :math:`O_h`,
:math:`I_h`, :math:`D_{\infty h}`, :math:`\sigma_{x,y,z}`,
:math:`SO(2)_{x,y,z}`, :math:`O(2)_{x,y,z}`, :math:`C_{2,3,4}`,
:math:`D_{1h}, D_{2h}, D_{3h}` :math:`\}` — 12 of them finite, so
:math:`21 \times 12 \times 21 = 5292` admissible
:math:`(H, \Gamma, K)` triples — **217** have
:math:`H \subseteq \Gamma K` with neither factor containing
:math:`H`. ⚠ That COUNT is a property of the member set and not of the
lattice: `[M]` swapping :math:`D_{3h}` for :math:`C_6` reads **181**
and swapping :math:`C_3` for :math:`D_{4h}` reads **255**, at the same
denominator. Quote the witness, not the count. And with
:math:`\Gamma = \{e\}` the predicate degenerates to :meth:`contains
<orpheus.numerics.symmetry.SubgroupOfO3.contains>` exactly: `[M]`
bit-equal on **441 of 441** ordered pairs over that same set.

⚠ **No shipped geometry exercises the product structure**, and that is
a property of the table rather than of the design: `[M]` 2026-09-03
every one of the four rows has :math:`K` trivial or :math:`\Gamma`
trivial, so :math:`H \subseteq \Gamma K` degenerates to a containment on
all four. The witness above is geometry-free and is what the gate uses;
a mutation replacing the body with
``kappa.contains(self) or gamma.contains(self)`` would otherwise be
green tree-wide (``plan-authoring`` §6c). The first geometry with both
factors non-trivial is what would activate it.

Stage 0, measured — the whole grid, and which conjunct decides each row
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`[M]` 2026-09-03 on the live tree: 4 geometries × 7 rules = 28 rows,
**11 admitted** and **17 refused**, and the refusals split
**14 arrow-only / 3 coverage-only / 0 both**. The seventh "rule" is a
:math:`\sigma_z` fold of the product rule, constructed rather than
shipped, because it is the only input that separates the cylinder's
:math:`\Gamma` from the plane's.

.. list-table:: `[M]` 2026-09-03 — stage 0, and which conjunct decides
   :header-rows: 1
   :widths: 26 16 14 14 14 16

   * - Rule (support, :math:`H`)
     - slab
     - sphere
     - cylinder
     - cartesian2d
     - Why
   * - ``gauss_legendre(8)``
       (:math:`S^2/O(2)_x`, :math:`O(2)_x`)
     - ✅
     - ✅
     - ⛔ coverage
     - ⛔ coverage
     - Its own geometries admit it through the identity arrow, as
       :math:`O(2)_x \subseteq \{e\}O(2)_x`. For the 2-D geometries the
       ARROW exists (:math:`S^2 \to S^2/O(2)_x` is the entry's own
       :math:`\pi`) and step 1 refuses: :math:`SO(2)_x \not\subseteq
       \{0\}`.
   * - ``gauss_legendre_on_mu(8)``
       (:math:`[-1,1]`, bare)
     - ⛔ arrow
     - ⛔ arrow
     - ⛔ arrow
     - ⛔ arrow
     - The chart-level rule, deliberately unregistered: a chart is not
       an orbit space and nothing descends onto it
       (:ref:`manifold-polar-orbit-rule`).
   * - ``product(4, 8)`` /
       ``lebedev(5)`` /
       ``level_symmetric(4)``
       (:math:`S^2`, bare)
     - ⛔ arrow
     - ⛔ arrow
     - ✅
     - ✅
     - A full-sphere rule for a geometry that spends nothing. For the
       1-D geometries :math:`S^2/O(2)_x` has no arrow ONTO :math:`S^2`.
   * - ``folded_product(4, 8)``
       (:math:`S^2/\sigma_y`, :math:`\sigma_y`)
     - ⛔ arrow
     - ⛔ arrow
     - ✅
     - ⛔ **coverage**
     - ⭐ **The R3 row.** :math:`\sigma_y \subseteq D_{1h}\{e\}` on the
       cylinder and :math:`\sigma_y \not\subseteq \sigma_z\{e\}` on the
       plane — the ERR-081 repair, and the only verdict R3 moved.
   * - :math:`\sigma_z` fold of ``product(4, 8)``
       (:math:`S^2/\sigma_z`, :math:`\sigma_z`)
     - ⛔ arrow
     - ⛔ arrow
     - ✅
     - ✅
     - Both 2-D geometries are z-uniform, so both admit a
       :math:`\sigma_z` fold: :math:`\sigma_z \subseteq D_{1h}\{e\}` and
       :math:`\sigma_z \subseteq \sigma_z\{e\}`.

.. warning::

   ⚠ **The FOLD arm has no witness on the SLAB, and that is a property
   of the catalogue rather than of the design.** `[M]` 2026-09-03,
   ``quotient_onto(S^2/O2_x, X)`` is non-``None`` for **exactly one** of
   the seven catalogued spaces of :math:`S^2` (the base, the three
   axial quotients, the three mirror quotients) — :math:`X = S^2/O(2)_x`
   itself, the identity arrow. No shipped orbit space is a proper further
   quotient of a slab's domain, so on ``"slab"`` and ``"sphere"`` stage 0
   admits exactly what equality admitted. It bites on ``"cylinder"`` and
   ``"cartesian2d"``, whose domain is the whole sphere. A future entry
   :math:`(S^2/O(2)_x)/K'` would activate it — and, unlike under the
   retired ``spent_group``, it would be DECIDED rather than refused,
   because the coverage test never asks what an arrow spends. Stated
   here so a reader does not infer coverage from the predicate's
   generality (``plan-authoring`` §6c).

   ⭐ **The COVERAGE leg, by contrast, has an end-to-end witness in the
   selector — one that is easy to miss, because it is not a fold of a
   reduced domain.** ``GaussLegendre1D`` is registered and its support
   IS a quotient, :math:`S^2/O(2)_x` with :math:`H = O(2)_x`, so for the
   two 2-D geometries the arrow exists and step 1 of the coverage
   theorem refuses it (:math:`SO(2)_x \not\subseteq \{0\}`). `[M]`
   2026-09-03, neutering ``is_subset_of_product`` in-process moves that
   rejection from stage 0 to stage 2's V conjunct — *"exact against
   legendre, but geometry 'cylinder' integrates against
   uniform(S^2)"* — and leaves the CHOSEN rule unchanged
   (``LebedevSphere(order=5)`` either way). So the leg changes the
   REASON at the selector tier and no selection; its behavioural gate
   belongs at the ``admits_domain`` tier, and no end-to-end selector row
   may be credited to it.

Stage 1 is asked on :math:`X`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``admits_symmetry`` asks the same question it always asked, and it is
correct for a fold only because the predicate itself moved to the orbit
space (:ref:`manifold-one-invariance-kernel`). ⛔ **This paragraph read**
*"its text did not change — it is still* ``Γ.is_invariant(measure)``\ *"*
**until 2026-09-03.** R2 of #434 did change the text, and only the text:
the body is now ``measure.is_invariant_under(self.owed)``, receiver and
argument swapped, because the question is the measure's
(:ref:`discrete-measure-invariance-module`); R3 then renamed the field it
reads from ``discrete_residual`` to ``owed``. `[M]` 2026-09-03 the stage-1
verdict is unmoved on every shipped (rule × geometry) pair — R3 touched
neither the message nor the group, and `[M]` **8 of 8** ``symmetry
mismatch`` strings in the frozen selection baseline are byte-identical
across the carve.

The refusal names ONE clause
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`AngularSymmetry.domain_refusal
<orpheus.numerics.quadrature.registry.AngularSymmetry.domain_refusal>`
returns the one failing clause or ``None``, and ``admits_domain`` is its
``is None``; the selector appends the returned reason to its stage
prefix. `[M]` 2026-09-03, the live text for the two refusals that matter:

.. code-block:: text

   domain mismatch: geometry 'slab' discretises S^2/O2_x, but the rule's
   nodes live on S^2, and S^2/O2_x has no descent arrow onto it

   domain mismatch: geometry 'cartesian2d' discretises S^2, but the
   rule's nodes live on S^2/sigma_y, a fold by sigma_y, which is not a
   symmetry the solution has in this geometry's frame (unspent sigma_z,
   spent Trivial)

⛔ **Until R3 the message was a DISJUNCTION naming both causes on every
refusal** — *"no descent arrow onto it, or a fold by a group outside the
owed sigma_x"* — while the predicate had already decided which conjunct
bit and thrown the answer away. `[M]` 2026-09-03 over the same 28-row
grid, **14 of the 17** refusals fail the arrow only and the coverage
clause is TRUE for all 14; **3** fail coverage only and the arrow exists
for all 3; **0** fail both. So the disjunction named a SATISFIED fact on
17 of 17, and the carve's own headline finding — the ``cartesian2d``
fold — was invisible in its own error message. The repair is one
predicate returning ``str | None`` rather than two call sites re-asking
``quotient_onto``, which would have re-created the twin R3 removed.

.. _manifold-orbit-certificate-orbit-space:

``certificate_under`` follows — and the II.11 lead is CLOSED
--------------------------------------------------------------

:func:`~orpheus.numerics.invariance.certificate_under` takes the same
route, so the certificate and the predicate cannot disagree about
whether a fold is closed under its owed residual. On a
:class:`~orpheus.numerics.manifold.Quotient` support it refuses a group
that does not normalise the spent group, and otherwise builds the
permutations of the CHART points under each element's induced action.

That CLOSES a defect lead the campaign recorded and did not schedule
(§II.11 of the plan): the certificate used to refuse a 1-D measure by
SHAPE, before invariance was ever asked, and its refusal message said
something false when it did. The shape test is gone — a bare support is
asked on the trivial orbit space like any other, and a 1-D node set is
a legitimate point set there.

.. list-table:: `[M]` 2026-09-02 — ``certificate_under``, before → after
   :header-rows: 1
   :widths: 26 14 16 20 24

   * - Measure
     - Group
     - ``is_invariant_under``
     - Certificate, before → after
     - Why, after
   * - ``gauss_legendre(8)`` (``'S^2/O2_x'``)
     - ``True``
     - :math:`\sigma_x`
     - ⛔ ``None`` → ✅ 2 permutations
     - ✅ **the II.11 case, closed**
   * - ``gauss_legendre_on_mu(8)`` (``'[-1,1]'``, BARE)
     - ``True``
     - :math:`\sigma_x`
     - ⛔ ``None`` → ✅ 2 permutations
     - ✅ **the shape refusal is gone for a bare support too**
   * - ``folded_product(4, 8)`` (``'S^2/sigma_y'``)
     - ``False`` → ``True``
     - :math:`\sigma_y`
     - ⛔ ``None`` → ✅ 2 permutations
     - the group that was spent, acting trivially
   * - ``folded_product(4, 8)``
     - ``False`` → ``True``
     - :math:`D_{2h}`
     - ⛔ ``None`` → ✅ 8 permutations
     - the cylinder's owed residual, realized
   * - ``gauss_legendre(8)``
     - ``True``
     - :math:`O(2)_x`
     - ``None`` → ``None``
     - correct, and unchanged: a continuous group has no finite node
       permutation
   * - ``gauss_legendre(8)`` / ``folded_product(4, 8)``
     - ``False``
     - :math:`C_4`
     - ``None`` → ``None``
     - ⚠ a **new** arm — :math:`C_4` normalises neither
       :math:`O(2)_x` nor :math:`\sigma_y`

⛔ **What of II.11 remains — the MESSAGE, and it grew an arm.** The
refusal text at ``measure.py`` and ``symmetry.py`` still reads *"this
measure is not X-invariant (or X is continuous, and a continuous group
has no finite node permutation)"*, and it is now a **three**-arm
disjunction wearing two-arm text: the third arm — the group does not
NORMALISE the spent group — arrived with this step. `[M]` on
``gauss_legendre(8)`` under :math:`C_4` the parenthetical is false
(:math:`C_4` is finite) and the true reason, *"it does not act on this
orbit space at all"*, is not among the two the sentence offers. This is
``vv-principles`` #17's multi-arm granularity trap in a production
guard. It is REPORTED here rather than repaired, because the repair is a
message and a guard in :mod:`orpheus.numerics.measure` /
:mod:`orpheus.numerics.symmetry`, and this pass touches only
``docs/theory``.

.. _manifold-spent-group-door:

The spent-group door — :math:`(M/H)/G` for :math:`G \subseteq H` is refused
-----------------------------------------------------------------------------

A group inside the group already spent acts trivially on the orbit
space, so quotienting by it is the identity and
:math:`S^2/\sigma_y/\sigma_y` would be a SECOND SPELLING of
:math:`S^2/\sigma_y`. That is the same disease the naming law diagnoses
(:ref:`manifold-orbit-space-stabiliser`), so it gets the same treatment:
``_catalogued_quotient`` refuses at the door, with the theorem, before
any lookup.

.. code-block:: text

   S^2/sigma_y/sigma_y is S^2/sigma_y: sigma_y lies in the spent group
   sigma_y, so it acts trivially on the orbit space and the quotient is
   the identity — there is nothing to quotient. The orbit space is
   S^2/sigma_y already.

`[M]` 2026-09-02: ``SPHERE.quotient(Mirror("y")).quotient(g)`` raises
that ``ValueError`` for :math:`g = \sigma_y` and falls through to the
catalogue's own ``NotImplementedError`` for :math:`g = \sigma_x`
(*"no catalogue entry for S^2/sigma_y/sigma_x"*). The refusal reaches
the measure-level verb too:
``folded_product(4,8).measure.quotient(Mirror("y"))`` raises the door's
message where it used to raise ``certificate_under``'s
*"this measure is not sigma_y-invariant"* — which was the AMBIENT
reading, and is exactly the sentence 2.2b made false.

⭐ **The door names ONE exception, and it is a derivation output rather
than a special case.** The trivial group is admitted on every base,
including an orbit space, because :math:`\{e\}` acting trivially is not
the same defect as a group being spelled twice: quotienting by it is
what asking for the identity fold MEANS, and refusing it would have been
the door mistaking *"acts trivially"* for *"is spelled twice"*. On a
bare base :math:`M/\{e\}` is the identity ENTRY, derived by
``_mod_trivial`` — the catalogue's positive control on its own machinery
(:ref:`manifold-twin-lookup`). `[M]` 2026-09-03
``folded_product(4,8).measure.quotient(Trivial)`` returns all **16**
nodes on its support, unchanged.

⛔ **But on an orbit space the answer is the ORBIT SPACE ITSELF, and it
was a second object until 2026-09-03.** :math:`(M/H)/\{e\}` is
:math:`M/H` — the door's own theorem for every :math:`G \subseteq H`,
applied at :math:`G = \{e\}`, which is contained in every group there
is. `[M]` until #434 R4 this arm built a fresh
:class:`~orpheus.numerics.manifold.Quotient` whose ``base`` was the fold
and whose ``by`` was ``Trivial``, so
``SPHERE.quotient(Mirror("y")).quotient(Trivial).name`` was
``'S^2/sigma_y/Trivial'`` and the value compared **unequal** to
``SPHERE.quotient(Mirror("y"))``: one orbit space, two objects, two
names — exactly the two-spellings disease this door exists to refuse,
inside the door's own exception. `[M]` 2026-09-03 the answer is now the
entry, **by identity**:

.. code-block:: pycon

   >>> fold = SPHERE.quotient(SubgroupOfO3.Mirror("y"))
   >>> fold.quotient(SubgroupOfO3.Trivial) is fold
   True
   >>> fold.quotient(SubgroupOfO3.Trivial).name
   'S^2/sigma_y'

⚠ **No ASSERTION pinned the old string, and the paragraph above was its
only carrier.** `[M]` 2026-09-03, a Python-``re`` census of
``sigma_y/Trivial`` over ``tests/`` returns **2 hits, both in
docstrings** — and both belong to the R4 gate that replaces the
behaviour
(``TestR4TheTrivialQuotientOfAnOrbitSpaceIsThatOrbitSpace``), recording
the pre-carve name as ``[M]`` history exactly as this paragraph does. So
the correction is a docs edit plus a new gate rather than a re-key, which
is the shape to expect from a naming defect no consumer had reached yet:
the corpus was the regression surface, and a claim nothing asserts is the
one that rots loudest.

⚠ **A brief for this step reported that a second fold by** :math:`\sigma_x`
**"works today and stays".** `[M]` it does not, on either tree: both
before and after, ``folded_product(4,8).measure.quotient(Mirror("x"))``
raises ``NotImplementedError`` because the catalogue has no
:math:`S^2/\sigma_y/\sigma_x` entry. What 2.2b changed on that verb is
only the :math:`\sigma_y` row, and it changed the REASON, not the
refusal.

.. _manifold-ordinate-permutation-orbit-space:

One notion of "does this isometry permute the ordinates"
----------------------------------------------------------

:meth:`Quadrature.ordinate_permutation
<orpheus.numerics.quadrature.directional.Quadrature.ordinate_permutation>`
is the tree's single source for that question — the boundary realizer
and the specular deck both read it — and it embedded a rule's nodes in
:math:`\mathbb{R}^3` and asked the AMBIENT question, exactly as
``is_invariant_under`` did. It now takes the orbit-space route too, so
the two cannot answer differently about one rule. Since R2 of #434 they
are the same call: ``ordinate_permutation`` is
:meth:`self.measure.permutation_under(...)
<orpheus.numerics.measure.DiscreteMeasure.permutation_under>`, the
single-motion face of the one closure.

.. list-table:: `[M]` 2026-09-02 — ``ordinate_permutation(sigma_a)``, before → after
   :header-rows: 1
   :widths: 26 24 24 26

   * - Rule
     - :math:`\sigma_x`
     - :math:`\sigma_y`
     - :math:`\sigma_z`
   * - ``folded_product(4, 8)`` (:math:`S^2/\sigma_y`)
     - 16-cycle, 0 fixed → unchanged
     - ⛔ ``None`` → ✅ **the IDENTITY permutation**
     - 16-cycle, 0 fixed → unchanged
   * - ``gauss_legendre(8)`` (:math:`S^2/O(2)_x`)
     - 8 nodes, 0 fixed → unchanged
     - IDENTITY → unchanged
     - IDENTITY → unchanged
   * - ``product(4, 8)``, ``lebedev(11)`` (:math:`S^2`)
     - unchanged
     - unchanged
     - unchanged

⭐ **The one cell that moves is the right one, and it is the fold's own
spent group.** :math:`\sigma_y` acts trivially on :math:`S^2/\sigma_y`,
so the permutation it induces on the stored ordinates is the identity —
where the ambient reading answered ``None`` (*"not a symmetry of this
rule"*) because a :math:`y \ge 0` representative maps to a point the
rule does not carry. The slab's two ``IDENTITY`` cells are the same fact
one entry over and were already right: a polar marginal's nodes lift to
the :math:`x`-axis, which :math:`\sigma_y` and :math:`\sigma_z` fix
pointwise.

.. _manifold-2-2b-what-moved:

What moved, and what did not
------------------------------

Everything below was measured on 2026-09-02 against a pinned
pre-change tree, one probe per row.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Reading
     - Before
     - After
   * - ``is_invariant_under`` over
       :func:`~orpheus.numerics.invariance.candidate_groups`,
       ``folded_product(4, 8)``
     - —
     - **4 of 21** flip, all ``False`` → ``True``:
       :math:`\sigma_y`, :math:`C_2`, :math:`D_{1h}`, :math:`D_{2h}`
   * - the same, ``gauss_legendre(8)``
     - —
     - **0 of 15** change
   * - the same, ``product(4, 8)``
     - —
     - **0 of 23** change
   * - :func:`~orpheus.numerics.invariance.symmetry_groups`
       on the fold
     - ``{sigma_x, sigma_z}``
     - ``{D_2h}`` — the two mirrors are absorbed into the group that
       contains them, and :math:`\sigma_y` joins because it acts
       trivially
   * - the walk on ``gauss_legendre(8)`` / ``product(4, 8)``
     - ``{O2_x, sigma_x}`` / ``{D_8h}``
     - unchanged
   * - the walk against brute force over 6 shipped rules
     - 6 of 6 agree
     - 6 of 6 agree
   * - the compatibility law
       :math:`A \subseteq B \wedge P(B) \Rightarrow P(A)`
       (``vv-principles`` #15)
     - 0 violations over 342 and over 450 (edge × fixture) pairs
     - **0 violations, both denominators, unchanged** — and unchanged
       again over #434 R1 at a third, widened denominator (175 edges,
       1750 pairs, 27 groups × 10 rules), on the pre-carve and carved
       trees alike
   * - ``_embedded_nodes`` against
       :func:`~orpheus.numerics.manifold.barycentre`, 12 rows
     - ``array_equal`` 12 of 12
     - **12 of 12**, now reached through
       :meth:`~orpheus.numerics.manifold.Quotient.orbit_barycentres`
       (called ``ambient_representatives`` from this step until #434 R4
       on 2026-09-03) and :attr:`Quotient.lift
       <orpheus.numerics.manifold.Quotient.lift>`
   * - ``ordinate_permutation`` on the fold under
       :math:`\sigma_y`
     - ``None``
     - **the IDENTITY permutation**
       (:ref:`manifold-ordinate-permutation-orbit-space`); every other
       (rule × mirror) cell over four rules is unchanged
   * - ``certificate_under`` on a BARE 1-D support
     - ``None`` — refused by SHAPE (§II.11)
     - **2 permutations**; the shape test is gone
   * - the ``is_invariant_under`` answers on
       ``gauss_legendre_on_polar_orbit(8, "z")`` and on the chart rule
       ``gauss_legendre_on_mu(8)``
     - 7 and 6 candidate groups ``True``
     - **unchanged**, group for group
   * - walk cost, min of 9 repeats, host ``.venv``, loaded machine
     - fold 115.7 ms, slab 5.8 ms, ``product(4,8)`` 134.4 ms
     - fold 106.9 ms, slab 5.8 ms, ``product(4,8)`` 136.5 ms — within
       the noise of a loaded machine; no cost claim is made either way

⚠ **The declaration and the computation now answer differently about a
fold, deliberately.** ``folded_product(4, 8).measure.invariance_group``
is still ``None`` — the STORED tag is a statement about the
representatives, and folding really does destroy having
:math:`\sigma_y` as a symmetry OF THE SECTION — while
``measure.is_invariant_under(Mirror("y"))`` is now ``True``, because it
asks about
the ORBITS and every orbit is its own image. Those are two questions,
and :ref:`manifold-has-versus-spent` is where the slots are told apart. A
reader who reads either as the other gets ``plan-authoring`` §3's
ambiguous-name hazard.

⭐ **What this unblocks, and what it does not.** `GitHub #370
<https://github.com/deOliveira-R/ORPHEUS/issues/370>`_ records two
structural gaps that stop ``folded_product`` being REGISTERED. Its
second — *"stage 0 cannot match a quotient support"* — is closed here,
and closed the way that issue demanded: not by widening
:attr:`AngularSymmetry.support
<orpheus.numerics.quadrature.registry.AngularSymmetry.support>` to
accept a quotient tag, but by reading the lattice's own descent arrow.
Its first is untouched: `[M]` 2026-09-02
``folded_product(4, 8).measure.exactness`` is still ``None``, and a rule
with no exactness claim is refused at stage 2. Registration remains
blocked, on a derivation rather than on a predicate.

.. _manifold-gotchas:

Gotchas
=======

.. _manifold-import-cycle:

The import graph reads like the mathematics — and it did not, until R2
------------------------------------------------------------------------

A quotient is a manifold **and** a group, so this module imports
:class:`~orpheus.numerics.symmetry.SubgroupOfO3` (and the axis table)
like any other dependency, at module scope. That is new — it is R2 of
#434, 2026-09-03 — and it is the whole of what that carve bought:

.. code-block:: text

   geometry.transformation  <-  symmetry     (groups)
                            <-  manifold     (an orbit space is a manifold AND a group)
                            <-  measure      (a measure lives on a manifold)
                            <-  invariance   (a measure X a group)

Each arrow points from a thing to the thing it is defined in terms of,
and no arrow runs back **at runtime** — the one back-edge in the table
below, ``invariance → measure``, is a ``TYPE_CHECKING`` import, which is
erased. `[M]` 2026-09-03, by an AST census over the
six modules with relative imports resolved and ``TYPE_CHECKING`` bodies
separated (the census is validated against a known relative import
before its zeros are read — see the two traps below):

.. list-table:: Every edge among ``symmetry`` / ``manifold`` / ``measure`` / ``invariance`` / ``exactness`` / ``generating_measure``
   :header-rows: 1
   :widths: 30 26 20 24

   * - Site
     - Edge
     - Scope
     - Note
   * - ``symmetry.py:102``
     - ``symmetry → geometry.transformation``
     - module, **runtime**
     - The only package this module now reaches, plus the numpy-only
       ``roots_of_unity`` beside it. A group is a set of rigid motions;
       nothing about a group needs a measure, a manifold or a
       quadrature.
   * - ``manifold.py:78``
     - ``manifold → symmetry``
     - module, **runtime**
     - ⭐ **The reversal.** ``AXIS_INDEX``, ``AXIS_LETTER`` and
       :class:`~orpheus.numerics.symmetry.SubgroupOfO3`, so
       :attr:`Quotient.by <orpheus.numerics.manifold.Quotient.by>` is
       annotated with the real class instead of a string, and every
       member this module used to duck-type is an ordinary read
       (:ref:`manifold-import-cycle-history` counts them).
   * - ``manifold.py:81``, ``:82``
     - ``manifold → geometry.transformation``,
       ``manifold → exactness``
     - ``TYPE_CHECKING``
     - Annotations only. ``exactness`` imports THIS module at module
       scope (two edges, ``:115`` and ``:116``), so that one is a real
       two-hop cycle and the guard on it is load-bearing.
   * - ``manifold.py:1684``
     - ``manifold → generating_measure``
     - **function** scope
     - The module's only runtime edge into a higher layer — the
       ``LEGENDRE`` *value*, inside ``_sphere_mod_o2``, because a
       ``TYPE_CHECKING`` guard defers a *name* and can never carry a
       *value* (:ref:`manifold-value-at-function-scope`).
   * - ``measure.py:96``, ``:97``, ``:98``
     - ``measure → invariance``, ``→ symmetry``, ``→ manifold``
     - module, **runtime**
     - The verbs' host reaches all three. ``invariance`` is bound as a
       MODULE (``from orpheus.numerics import invariance as
       _invariance``) so each verb resolves its kernel at call time —
       the delegation a counting spy can see.
   * - ``invariance.py:54``, ``:60``, ``:61``
     - ``invariance → geometry.transformation``, ``→ manifold``,
       ``→ symmetry``
     - module, **runtime**
     - A measure × a group: both operands, at the bottom of the stack.
   * - ``invariance.py:64``
     - ``invariance → measure``
     - ``TYPE_CHECKING``
     - The back-edge, and it is type-only **by construction**: this
       module READS measures and never builds one, so the name is
       needed for an annotation and the value never is.
   * - ``generating_measure.py:163``–``:165``
     - ``→ exactness`` / ``→ measure`` / ``→ manifold``
     - module, **runtime**, three times
     - Unchanged by R2, and what makes ``manifold.py:1684`` have to be
       function-scoped.

⚠ **A census keyed on a package name is not a census of edges.** The
table above needed two fixes to its own filter before its zeros meant
anything, and both fail silently in the reassuring direction. (1) An
``ImportFrom`` with ``level > 0`` carries an **unqualified**
``.module``, so a filter on ``node.module.startswith("orpheus")`` drops
every relative import — a census written that way reports
``invariance → measure`` as ABSENT and concludes there is no back-edge
at all. (2) ``from orpheus.numerics import invariance as _invariance``
has ``node.module == "orpheus.numerics"``: it is an edge to the
**submodule**, and a filter that only compares ``node.module`` against
the module set reports ``measure → invariance`` as absent too — which is
the single most load-bearing runtime edge in the table. `[M]` both were
live in the first draft of this census.

⭐ **10 of 10 entry points import cleanly**, measured the only way this
question can honestly be measured — one fresh interpreter per entry
point, ``orpheus``, ``orpheus.numerics``, ``.symmetry``, ``.manifold``,
``.measure``, ``.invariance``, ``.quadrature.registry``,
``orpheus.geometry``, ``orpheus.geometry.transformation`` and
``orpheus.sn.solver`` — with a positive control (importing a module that
does not exist must return a non-zero code, or a clean reading carries no
information). The live witness is
``tests/test_layer_imports.py``'s fresh-interpreter gate, which R2
widened by the four entries the old cycle killed and the previous list
could not see.

.. _manifold-import-cycle-history:

The cycle this section used to be about
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

⛔ **Until 2026-09-03 this section was titled** *"The module imports
nothing from* ``numerics`` *at MODULE scope — on purpose"*, **and every
claim under it was true.** The invariance kernel lived in ``symmetry``,
so ``symmetry`` imported ``measure``, and ``measure`` imports
``manifold`` — a three-hop cycle
``measure → manifold → symmetry → measure`` that a module-scope
``manifold → symmetry`` edge would close. Tracker 2.4 (2026-09-01) then
added ``symmetry → manifold`` directly, making it a **two-hop** cycle as
well: two independent loops, one guard. The costs the guard imposed are
the reason R2 exists, and they are worth listing because none of them
looked like a cost at the time:

* :class:`~orpheus.numerics.symmetry.SubgroupOfO3` could be referenced
  here only under :data:`typing.TYPE_CHECKING`, so every member of it
  this module read was duck-typed at runtime — the group always had to
  arrive as an *argument*, never as a name this module could resolve. A
  design that wanted to *construct* a group here (to normalise a caller's
  tag, say) was simply not available. `[M]` 2026-09-03, by AST over
  ``manifold.py`` at the pre-carve commit, counting attribute reads whose
  receiver is spelled ``group`` or ``.by``: **10 distinct members over 24
  sites** — ``name`` (13), ``contains`` (2), ``is_trivial`` (2), and one
  each of ``dim``, ``generic_images``, ``generic_orbit_dimension``,
  ``is_normalised_by``, ``mirror_axis``, ``orbit_stabiliser`` and
  ``rotation_axis``. ⚠ The count is predicate-bound and has been reported
  as **three**, **seven** and **nine** at different dates under different
  receiver sets, which is why the predicate is written into the sentence
  rather than left to the reader; what never changed, and what the
  argument rests on, is that the group always arrived as an argument.
* :meth:`DiscreteMeasure.quotient
  <orpheus.numerics.measure.DiscreteMeasure.quotient>` reached its
  invariance certificate through a **function-scope** import.
* The identity entry's group was fetched by a function-scope import of
  its own, inside a helper called ``_trivial_group`` — retired at R2,
  because :class:`~orpheus.numerics.symmetry.SubgroupOfO3` is now an
  ordinary name here.
* And the kernel that the cycle existed to support carried **two**
  copies of the orbit closure — an identical lambda inlined in a second
  function — while three docstrings claimed there was one.

⭐ **The axis table had to move BACK, and that is the part a plan cannot
guess.** R2 as first written said only *"* ``manifold`` *imports*
``symmetry`` *at module scope"*, and as written **it does not import**:
``symmetry`` read ``AXIS_INDEX`` / ``AXIS_LETTER`` from ``manifold`` at
six sites that do not move, so reversing the other edge closes a 2-cycle
in the opposite direction. The ruling was to move the axis table back to
``symmetry.py`` — its home until 2026-09-02, so a restoration rather than
a novelty — and let ``manifold``, ``basis/descent.py`` and
``basis/spherical_harmonic_basis.py`` read it from there.

`[M]` 2026-09-03, on a **renamed shadow copy** of the package
(``shadowpkg``, so the editable install's ``sys.meta_path`` finder cannot
serve the real tree by accident, and every subprocess prints the
``__file__`` it actually loaded), one fresh interpreter per (variant,
entry point), no production file touched:

.. list-table::
   :header-rows: 1
   :widths: 46 18 36

   * - Variant
     - Clean imports
     - What fails
   * - **V0** — the shipped direction
     - **10 of 10**
     -
   * - **V1** — ``symmetry`` reads one name from ``manifold`` at module
       scope (the axis table's old home, the rest of R2 unchanged)
     - **3 of 10**
     - ``ImportError: cannot import name 'Quotient' from partially
       initialized module`` on ``orpheus.numerics``, ``.symmetry``,
       ``.manifold``, ``.measure``, ``.invariance``,
       ``.quadrature.registry`` and ``orpheus.sn.solver``

⚠ **and the three survivors are the trap.** ``import orpheus`` alone is
among them: it returns ``rc=0`` under V1. A smoke test on the package
root reports GREEN on a façade that cannot serve a single one of its
numerics entry points — the order-dependence ``plan-authoring`` §6d
warns about, reproduced here rather than argued.

⚠ **Two further ways this is easy to get wrong, both measured earlier
and both still live.** The relative-import filter above is the first.
The second is a **tense** trap: this section once read, verbatim, "the
cycle is not live today — ``measure`` does not import ``manifold``
yet", written
2026-08-31 when the guard was purely prophylactic and false within a
day. A prophylactic guard stops being prophylactic without anything
editing it, so *"is this still just precaution?"* is a question with a
shelf life — and the mirror now applies to this section itself: the
cycle is GONE, so the guard that removed it is a layering fact rather
than a precaution, and the way to break it is to add an arrow, not to
delete a guard.

.. _manifold-gotcha-ambient:

Topological dimension is not ambient dimension
-----------------------------------------------

``dim`` is what the manifold *is*; the ambient count is how many
columns :meth:`contains <orpheus.numerics.manifold.Manifold.contains>`
consumes. They differ for every curved member — a sphere is
``dim == 2`` in ``3`` ambient coordinates — and a
:class:`~orpheus.numerics.manifold.Product` needs the *ambient* count
to know where to split a point's coordinates, not the topological one.
The module keeps the ambient count in a deliberately **exhaustive**
``match`` with a raising fall-through, so a new member that forgets it
fails loudly rather than silently mis-splitting a product's
coordinates; a foundation gate walks every shipped variant through it.

⭐ **A** :class:`~orpheus.numerics.manifold.Quotient` **is where the two
counts come apart hardest, and the type answers them differently on
purpose.** Its ambient count — what ``_ambient`` reports, and therefore
what a :class:`~orpheus.numerics.manifold.Product` uses to split a
point's coordinates — is the **realization's**, because a product
factor must have one canonical width or the split is ambiguous. Its
``contains``, by contrast, accepts **either** coordinate system and
dispatches on the width it is handed
(:ref:`manifold-two-coordinate-systems`). So for the shipped fold `[M]`
``_ambient`` is :math:`2` (the disk) while ``contains`` also accepts
the :math:`(16,3)` section points the tree's own measures carry.

⚠ That asymmetry is deliberate and it is the one place on this page
where a single object answers two questions with two different numbers.
Read it as: *the canonical coordinate for composition is the chart's;
the predicate is as wide as the honest languages the object has.* An
earlier version of this paragraph stated only the first half — *"the
ambient count is the realization's, because membership is decided in
the coordinates the chart lands in"* — which was true of the
single-slot type and became false for ``contains`` on 2026-08-31.

.. _manifold-gotcha-shape-vs-verdict:

A wrong ambient dimension is a refusal, not a ``False``
--------------------------------------------------------

``SPHERE.contains(np.zeros((4, 2)))`` raises :exc:`ValueError` naming
the expected ambient dimension. It does not return ``False``. The
distinction is the difference between *"these are not points of this
manifold"* — a verdict about the data — and *"you handed me something
that cannot be points of anything with this ambient dimension"* — an
error in the caller. Collapsing the two would let a shape bug read as a
membership failure, and a membership predicate that returns ``False``
for malformed input is a predicate whose ``False`` means nothing.

.. _manifold-gotcha-not-a-manifold:

``EnergyGroups`` and ``IndexSet`` are 0-dimensional, and that is honest
-----------------------------------------------------------------------

A finite index set carries no metric structure and no smooth structure;
calling it a manifold is a stretch that the type makes deliberately.
The justification is that the *algebra* is what level 1 is for — an
energy axis composes with a spatial one under :math:`\times`, and a
measure on the pair is the tensor product — and a "manifold" that
refused to admit the counting factors would force every composite
support back into a string. ``dim`` is ``0`` for both, which is the
correct topological dimension of a finite discrete set, and neither
carries a chart.


.. _manifold-seams:

What is NOT built (the standing seams)
======================================

Stated explicitly so no reader mistakes a shipped *type* for a shipped
*migration*, and so the next phase does not re-derive a decision
already taken. The whole of the first row is what makes this page a
description of a capability rather than of a repair.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Not built
     - Where it lands, and what stands in for it today
   * - ⛔ ~~**Any production consumer at all**~~
     - ✅ **REMEDIED 2026-09-01 (tracker 2.0c).** *(Recorded as written —
       `[M]` 2026-08-31: "the only importers of
       :mod:`orpheus.numerics.manifold` are its own test module;
       ``Space = str`` is still live at ``measure.py:111`` with its six
       ``SPACE_*`` aliases, ``DiscreteMeasure.support`` is still a
       ``str``".)* The alias and all six tags are retired; ``support`` is
       a ``Manifold`` on all six implementors, the tensor product and the
       fold route through :meth:`Manifold.__mul__` and
       :meth:`Manifold.quotient`, and :attr:`DiscreteMeasure.phase`
       dispatches on the manifold's TYPE instead of on string prefixes.

       ⚠ This row read, verbatim, "**ERR-080 is still open** — held by
       the same ``xfail(strict=True)`` gate … retyping the slot is what
       makes the refusal *spellable*; it does not make it *fire*", until
       2026-09-02.
       ✅ **ERR-080 is CLOSED**: the fused fix step landed and the gate's
       three strict-xfail markers self-retired. ⛔ The second half stands
       exactly as written — nothing calls :meth:`contains` on the way in,
       so `[M]` a forged measure is still **constructible**; what closed
       the defect is the refusal at the BASIS and at the FRAME, not at
       the measure (:ref:`manifold-g0-descent-arrow`). Construction-time
       membership is still owed, and is the row below.
   * - ``Basis.domain``
     - ✅ **LANDED 2026-09-01 (2.1).** No
       :class:`~orpheus.numerics.basis.base.Basis` could state the
       manifold its functions consume, which is why the ERR-080 pairing
       had nothing to check (:ref:`manifold-three-levels`). It is now an
       abstract property, so a basis that cannot say what it eats
       refuses to be constructed, and all six shipped subclasses answer.
       :class:`~orpheus.numerics.basis.indicator_basis.IndicatorBasis`
       takes it as a **constructor field** (``partition_of``) rather
       than deriving it, and the reason was measurable in advance:
       `[M]` by AST, of 18 ``IndicatorBasis(...)`` construction sites
       tree-wide **4 are in** ``orpheus/``, and those four partition
       **three different manifold families** — a finite index set
       (``frame.py``, paired with ``support=f"index({axis_label})"``
       three lines below), :math:`\mathbb{R}^d` at two ranks
       (``geometry/mesh.py``), and the energy counting set
       (``data/energy_grid.py``). Any value the class *derived* from its
       own fields would hard-code one of the three. ⭐ The prediction
       held, and execution added a fourth family the string tag had
       hidden: `[M]` a partition by energy **VALUE** in eV is an
       :class:`Interval`, not the :class:`EnergyGroups` **index** axis
       production partitions — both ambient dimension 1, so only naming
       the point set separates them.
   * - ``Basis.invariance_group``
     - ✅ **LANDED 2026-09-01 (2.1b), and DERIVED — this row exists to
       record that no slot was added.** The pairing ERR-080 needs has two
       operands and the basis's was missing; the tracker recorded the
       property as *absent and derivable*, which invited a second
       abstract property with six overrides, and the phase opener found
       the answer already sitting in ``domain.by``.
       A function on :math:`M/H` *is* an :math:`H`-invariant function, so
       the group is read by a ``match`` on the domain's TYPE — `[M]`
       **6 of 6** shipped bases answer, with **0** subclass edits and one
       ``@final`` property on the ABC. This is tracker 2.0d's
       ``quotient_group`` FIELD dissolving into :attr:`Quotient.by
       <orpheus.numerics.manifold.Quotient.by>` at 2.0c, replayed one
       level over (:ref:`manifold-basis-invariance-group`).

       ⛔ This row read, verbatim, "**Still not built: the CONSUMER.**
       … nothing refuses on it … ERR-080 remains open" until 2026-09-02,
       and it was true when written. ✅ **The consumer landed the same day**
       (tracker 2.2, fused): :func:`~orpheus.numerics.manifold.quotient_onto`
       is the predicate, it reads the verdict off the QUADRATURE's
       measure — which is what the old row's second `[M]` predicted a
       frame-side gate could not do — and the table it returns is the
       frame's (:ref:`manifold-g0-descent-arrow`). The verdict itself is
       now spelled ``Trivial ⊇ O2('x')``, still ``False``
       (:ref:`manifold-invariance-pairing`).
   * - ``FunctionSpace.manifold``, and the derived ``L2[...]`` name
     - **Still open, and the reason narrowed.** Two sites build a level-2
       name by interpolating a level-1 tag, and both now interpolate a
       typed :class:`Manifold`: `[M]` ``measure.py:371`` is
       ``f"L2[{self.support.name}]"`` (2.0c retyped ``support``) and
       ``basis/indicator_basis.py:355`` is
       ``f"L2[coarse_cells({self.domain.name})]"`` (2.1 gave the basis a
       ``domain``). ⛔ This row read *"one of them
       (``basis/indicator_basis.py:284``) **hard-codes** it and `[M]` is
       already **false** for the energy-grid basis"* until 2026-09-01 —
       true when written, repaired by 2.1
       (:ref:`manifold-string-algebra`). What remains is the seam
       itself: a ``FunctionSpace`` still records the **string**, so the
       two producers agree by discipline rather than by construction, and
       a space that carried its own manifold would collapse both
       spellings into one.
   * - The **SECTION** :math:`M/H \to M`, as an entry field
     - **Still open, deliberately, and it is now the ONLY one of the
       three.** ⛔ This row read *"The two MAPS — the* ``chart`` *and
       the section — and the pushforward measure, as entry fields …*
       `[M]` *7 of the derivation procedure's 9 outputs are slots"*
       until 2026-09-02. Two of its three subjects were discharged by
       tracker 3.1: the entry's own map ships as
       :attr:`~orpheus.numerics.manifold.Quotient.quotient_map` over
       the stored
       :attr:`~orpheus.numerics.manifold.Quotient.orbit_coordinates`
       (:ref:`manifold-quotient-map`) and the pushforward measure ships
       as :attr:`~orpheus.numerics.manifold.Quotient.reference`
       (:ref:`manifold-pushforward-reference`), and #434 R4 discharged
       a FOURTH subject the row never named — the **lift**, which was
       not on the procedure's output list at all until it was found
       living as a tag branch (:ref:`manifold-lift`). So `[M]`
       2026-09-03 the count is now **10 of 10** over **fourteen**
       fields (:ref:`manifold-engine-data-model`).

       ⭐ **The row also mis-named its own first subject**, and the
       correction outlives it: the map an entry emits is the
       **quotient map**, not a chart — a chart is injective and
       :math:`\Omega \mapsto \Omega\cdot\hat e_a` is not
       (:ref:`manifold-arrow-type`). What ``realization`` has always
       been is the *codomain* that map lands in.

       ⟹ what stands is the **section**, and it stands for a reason
       rather than for want of a phase. A section is a **choice** — for
       a positive-dimensional group no half-meridian is distinguished —
       while every field the entry carries is a derivation *output*.
       `[M]` 2026-09-03 ``fundamental_domain`` is ``None`` on all three
       axial entries and has **zero** readers outside
       :mod:`orpheus.numerics.manifold`; it is populated only on the
       three mirror entries and on :math:`M/\{e\}`, where a canonical
       section exists. That is why ``Quotient.contains`` must accept
       both languages rather than normalising to one
       (:ref:`manifold-two-coordinate-systems`), and why ERR-080's
       level-1 half — a fabricated section — is not closed by 3.1
       (:ref:`manifold-err-080-is-a-section`).

       ⭐ **#434 R4 made the seam narrower AND more clearly a seam.**
       Until 2026-09-03 the mirror entries had a section *as a map* —
       ``lift`` returned the hemisphere representative — while the
       axial ones had none, so the type answered in two languages
       depending on the family. R4 retired that arm: every entry's
       lift is now the orbit barycentre, which lands on the base only
       where the orbit is a point (:ref:`manifold-lift`). What survives
       of the mirror section is its **image**, in
       ``fundamental_domain``, which is what ``contains`` validates a
       fold's representatives against. So the seam is exactly **a map**
       :math:`M/H \to M` **for any entry**, and `[M]` **no shipped
       consumer wants one**: every reader of the lift asks an ORBIT
       question, and an orbit question is answered by a canonical point
       of the ambient space, not by a chosen point of the base.

       ⚠ **Where a section-like object DOES return is `GitHub #436
       <https://github.com/deOliveira-R/ORPHEUS/issues/436>`_.** That
       issue's subject is the datum
       :class:`~orpheus.numerics.manifold.FundamentalDomain` does not
       carry — Poincaré's **face pairings**, one isometry per normal —
       and the operation it buys is a ``retract(p)``: apply the pairing
       while a normal is violated, which is total, idempotent and needs
       no square root. That is the shape a section wants, and it is
       *derived from the pairing* rather than chosen, which is what
       makes it admissible on this page's own ruling. Two subsystems
       already hand-roll the missing datum (the specular deck's
       ``_mirror_motion``, MoC's ``_reflected_azi_index``), so the
       consumer exists before the field does — the opposite of the
       situation the lift was in.
   * - A ``ManifoldMap`` for the ERR-080 forgery arm
     - ✅ **NEVER BUILT, and the arm it would have wrapped is now
       GONE** (2026-09-02, tracker 3.4). Recorded as written, verbatim:
       "⛔ Deliberately NOT built, and it is the point.
       ``Quadrature._harmonic_frame_measure``'s 1-D arm computes the
       orbit barycentre and declares it on :math:`S^2`; `[M]`
       2026-09-02 its nodes are ``np.array_equal`` to
       :func:`~orpheus.numerics.manifold.barycentre`'s image. It stays
       a raw :class:`~orpheus.numerics.measure.DiscreteMeasure`
       constructor because routing it through
       :meth:`~orpheus.numerics.measure.DiscreteMeasure.pushforward`
       would force it to name ``Ball(3)``." The retirement landed the
       same day, and the arrow was never needed
       (:ref:`manifold-barycentre`).
   * - The remaining catalogue entries
     - **Phase 1.1.** `[M]` **six keys** ship — ``(Sphere,
       "O2_x"/"O2_y"/"O2_z")`` and ``(Sphere,
       "sigma_x"/"sigma_y"/"sigma_z")`` — served by **two**
       procedures, since each family shares one derivation that reads
       the axis off the group. The identity quotient is a seventh
       answer, derived rather than tabulated. ⛔ This row read **four
       keys**, with a single ``(Sphere, "SO2")``, until tracker 2.4
       parameterised the axial rotation group on 2026-09-01; and the
       three axial keys were spelled ``"SO2_a"`` until #432 re-keyed
       them onto the axis's stabiliser on 2026-09-02. Note the
       *procedure* count has not moved at either step
       (:ref:`manifold-so2-axis-is-a-parameter`,
       :ref:`manifold-orbit-space-stabiliser`). The expected
       remainder covers :math:`\mathbb{Z}_2` antipodal, :math:`C_n` /
       :math:`D_n` about an axis, the :math:`O_h` sublattice for octant
       symmetry, :math:`SO(3)`, and :math:`SO(2)\times\mathbb{R}_z` for
       the 1-D cylinder. ⚠ Whoever adds a :math:`C_n` entry must leave
       ``fundamental_domain=None``: a closed sector is **not** a
       strict fundamental domain, and the ``dim`` gate cannot catch a
       wrong one (:ref:`manifold-chart-section-asymmetry`).
   * - Collapsing the twin lookup
     - ✅ **DONE at tracker 2.4, 2026-09-01.**
       :attr:`AngularSymmetry.support
       <orpheus.numerics.quadrature.registry.AngularSymmetry.support>`
       no longer holds a table: it calls ``SPHERE.quotient(spent)``, so
       `[M]` for a slab its answer *is* the catalogue's object, by
       identity. The shipped collapse keeps the **orbit space**, where
       this row predicted ``…​.realization.name`` — a difference that
       matters, since taking the realization is the axis-blind step
       (:ref:`manifold-twin-lookup`, reading (iv)). The ``Trivial`` row
       remains two producers, deliberately.
   * - The ``support`` tag's own vocabulary split
     - ✅ **DONE at trackers 2.0c + 2.4.** `[M]` today
       ``gauss_legendre(8).measure.support.name`` is ``'S^2/O2_x'``
       and ``folded_product(4,8).measure.support.name`` is
       ``'S^2/sigma_y'`` — **both** the quotient's name, both typed
       :class:`~orpheus.numerics.manifold.Quotient` values. ⛔ This row
       read *"``gauss_legendre(8).measure.support`` is ``'[-1,1]'`` —
       the realization's name … the registry's stage-0 gate compares it
       by string equality"*. The register split is closed and the gate
       is a value comparison. ⛔ This row then read *"What SURVIVES is
       the disagreement it exposed … admits_domain on the shipped fold
       is still False, because a rule folded by a member of Γ genuinely
       does not live on the geometry's S^2/G^0"*, verbatim, until
       2026-09-02. The premise stays true and the conclusion is gone:
       the two orbit spaces really are different, and stage 0 now asks
       for the ARROW between them rather than for their equality. `[M]`
       2026-09-02 that call is **True**, and so is ``admits_symmetry``,
       and `[M]` re-measured 2026-09-03 both are still **True** after R3
       of #434 re-posed the arrow's companion conjunct — the fold is a
       symmetry a cylinder's solution HAS, which is what R3 asks
       (:ref:`manifold-gamma-slot`). ✅ **This seam is DISCHARGED** —
       what still blocks registration is stage 2, not stage 0
       (:ref:`manifold-2-2b-what-moved`).
   * - The derivation ENGINE
     - **Deferred, not refused** — the ruling, the falsifiable
       compliance check and the acceptance suite that is already
       written are at :ref:`manifold-engine-seed`.
   * - Renaming ``DiscreteMeasure.support``
     - **With the migration.** The slot names the ambient manifold,
       not :math:`\operatorname{supp}(\mu)`; the corpus already
       records the misnomer
       (:doc:`/theory/foundations/discrete_measures`). Renaming a slot
       that `[M]` **87** ``support=`` keyword arguments pass — 29 in
       ``orpheus/``, 58 in ``tests/`` — is a migration act, not a docs
       one.
   * - An ``automodule`` for this module
     - **Not scheduled, deliberately.** `[M]` 2026-08-31: of 48
       ``automodule`` directives in the doc source, **6** are
       ``orpheus.numerics.*`` — ``axis``, ``convergence``,
       ``coupled_system``, ``eigenvalue``, ``field``, ``functional``.
       ``manifold``'s two closest siblings in the three-level stack,
       :mod:`orpheus.numerics.measure` and
       :mod:`orpheus.numerics.space`, are **not** among them, so
       surfacing level 1 alone would make
       :class:`~orpheus.numerics.manifold.Manifold` a live link while
       :class:`~orpheus.numerics.space.FunctionSpace` beside it in the
       same sentence stays plain text. Surfacing the package is its
       own task. ⚠ Consequence for editors: the Python-domain
       cross-references on this page render as **plain text with no
       warning at any severity**, so a stale one is invisible to
       ``sphinx -W`` and must be caught by an import-resolution grep.


.. _manifold-verification:

Verification
============

The gates live in ``tests/numerics/test_manifold.py``: `[M]` 2026-09-02,
**91 test functions, 143 collected rows** (by AST for the first, by the
generated matrix for the second), run under the canonical
``python -O -m pytest`` invocation. Several functions are parametrized
— over the shipped-variant list, over three bases, and (since tracker
2.3) over the three rotation axes — and the two counts are given
separately because they move for different reasons: adding a *variant*
moves the second and not the first.

⚠ The row count here is the one the generated V&V matrix reports for
this module, which is a second, independent reading of the same tree
(``docs/theory/verification/matrix.rst``, ``numerics/test_manifold``
row). `[M]` it read **44** before the two-slot ruling, **56** after,
**70** after tracker 2.3 added ``TestManifoldMap``, **108** after the
fused commit added the isotypic probe's gates, and **143** after tracker
2.2b added the lift, the induced action and the spent-group door. An
earlier version of this paragraph said *"30 test functions, 40 collected
rows"*; both numbers were wrong when written — the module had 32
functions and 44 rows — which is why the count is now stated with the
instrument that produces it.

⭐ **Tracker 2.2b's gates are spread over three modules, and the two
siblings moved further than this one.** `[M]` 2026-09-02 from the same
generated matrix: ``numerics/test_symmetry`` **133 → 230** rows (the
normaliser's per-family legs, the sampling controls and the one-kernel
equivalence) and ``numerics/test_registry`` **80 → 107** (the ledger slot's admission
grid). All three modules are
``@pytest.mark.foundation`` for this page's standing reason: they gate
the type's own laws, not an L0–L3 claim about a flux.

⚠ **The symmetry module's row is FALLING under #434 R1 — read its value
from the matrix, never from here.** The carve retires the gates that
pinned the per-family arms it dissolved, so the count goes DOWN, which
is the expected shape when the thing under test stops having five
spellings and is why R1's acceptance evidence is a behaviour contract
rather than a count (below). `[M]` 2026-09-03 the row was observed at
230, 212 and 215 within one afternoon as the carve landed, which is the
reason this paragraph names a direction and a mechanism instead of a
number. ``numerics/test_manifold`` (**143**) and
``numerics/test_registry`` (**107**) are unmoved by it.

The fused commit's other two objects have their own modules, both
``@pytest.mark.foundation``: `[M]` 2026-09-03
``tests/numerics/test_legendre_basis.py`` (**34** rows) and
``tests/numerics/test_descent.py`` (**20** rows) — the intrinsic laws of
the new basis and the bit-identity of the two realizations respectively
(:ref:`manifold-descent`). ⛔ The first read *"15 functions, 32 rows"*
until 2026-09-03; the matrix reported 34 on the day it was written, so
the function count and the row count were taken from different readings
of the same tree. Only the row count is quoted now, because it has an
instrument that re-measures itself.

⭐ **#434 R4's verification lives at the AMBIENT tier, and that is
forced rather than chosen.** The invariance kernel is
:ref:`structurally blind <manifold-lift-is-mode-12-blind>` to the change
R4 makes — every downstream answer is read through
``orbit_coordinates``, which is exactly the column selection the
projector re-writes — so no end-to-end row can be a catcher and a green
one is ``vv-principles`` #12 by construction. The gates therefore assert
on the lift's OWN output, against references built from the group's
realized matrices (an SVD null space, the group mean, a trapezoid over
the orbit circle) and never from a column index. `[M]` 2026-09-03,
``tests/numerics/test_manifold.py`` collects **240** rows, of which
**102** are the nine ``TestR4*`` classes; five pre-R4 rows retired with
the behaviour they pinned (the mirror lift's hemisphere-section row ×3,
the pass-through width dispatch ×2) and three were re-keyed in place.
Every one carries ``@pytest.mark.foundation`` and none carries
``verifies(...)``, for this page's standing reason.

⭐ **Two of R4's rows are declared BLIND, in their own names, which is
the honest half.** :math:`\pi \circ \lambda = \mathrm{id}` holds for
the retired hemisphere section and for the projector alike, so it ships
as a round-trip leg *labelled* blind with the teeth in
:math:`\lambda \circ \pi = P_H`; and
``test_the_CHART_is_blind_to_all_of_this_and_the_two_tiers_say_why``
pins the blindness as a **property** rather than merely avoiding it —
the same move the fold's Mode-12 companion row makes one chapter up. A
gate set that did not say which of its rows cannot fail would read as
more coverage than it has.

⭐ **#434 R1's verification is a BEHAVIOUR CONTRACT, not a test-count
delta, and it is the shape a carve of this kind owes.** The carve
deletes two hand-written relation surfaces and thirteen per-family functions
while promising that no answer changes; the honest instrument is
therefore the full answer grid, captured on a pinned pre-carve tree
(``git archive HEAD``, the editable finder stripped and
``orpheus.__file__`` asserted in each subprocess) and compared cell by
cell. `[M]` 2026-09-03: ``contains`` and ``normalises`` over the full
:math:`27\times27` grids, **0 of 729** each; ``is_invariant_under`` over 10
shipped rules × 27 groups, **0 of 270**;
``symmetry_groups`` over the same 10 rules, **0**; the
compatibility law at three denominators on both trees, **0 violations
throughout**. Three readings move and each is named in advance
(:ref:`manifold-realization`). A grid is the right denominator here for
the reason :ref:`manifold-normaliser-sampling-control` gives one tier
down: the edges anyone thinks to name are the ones a hand table already
got right.

**Every one carries** ``@pytest.mark.foundation`` **and none carries**
``verifies(...)``, and that is the correct tier rather than an
omission. ``foundation`` is the V&V ladder's *orthogonal* category —
software invariants with no theory-page equation label behind them:
data-structure laws, factory outputs, algebraic reduction invariants.
The assertions here are the intrinsic laws of the type (dimension
additivity, membership, the quotient's dimension drop, the recorded
derivation), not an L0–L3 claim about a solver, a flux or an
eigenvalue. Tagging them ``verifies`` would mint a coverage edge that
an audit would then trust.

Seven groups, and what each is for:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Group
     - What it pins
   * - The type's own laws
     - The base is uninstantiable (a sum, not a member); every variant
       answers the three total operations; variants are frozen and
       compare by value; ``dataclasses.replace`` **re-runs the
       construction invariant** rather than being a hole in it; the
       names reproduce the retired string tags verbatim.
   * - The product algebra
     - :math:`\dim(M\times N) = \dim M + \dim N` over **every ordered
       pair** of the shipped members; the name reproduces the retired
       interpolation; multiplying a non-manifold is refused;
       membership splits the coordinate blocks.
   * - Membership, **both legs**
     - `[M]` **8 of the 9** tests in the ``TestMembership`` class
       assert a positive verdict beside their negative one
       (``vv-principles`` #11 — a contract predicate tested only
       against a broken instance validates the *raising*, not the
       *claim*); the ninth is
       ``test_a_wrong_ambient_dimension_is_a_typed_refusal``, which
       asserts a **raise** rather than a verdict and so has no positive
       leg to carry (:ref:`manifold-gotcha-shape-vs-verdict`). The
       load-bearing row is the ERR-080 forgery:
       the negative leg refuses :math:`(\mu,0,0)`, the positive leg
       admits the same nodes normalised, and a third assertion places
       them on :math:`[-1,1]` where they belong. (The count is scoped
       to that class on purpose — the fold's own membership gates,
       below, carry their legs differently.)
   * - The recorded derivation
     - The symbolic regression tests of
       :ref:`manifold-engine-seed` — for **both** entries, the
       :math:`P` matrix, the determinant, the empty syzygy, and the
       stratum **solved for** rather than compared to a literal.
   * - ⭐ The :math:`\sigma_y` fold, on production data
     - The load-bearing gate of the two-slot ruling carries **three**
       legs on the shipped
       ``Quadrature.folded_product(4, 8).measure.nodes``: the section
       ADMITS them, and REFUSES both wrong inputs — the orbit twins
       (which ``realization = SPHERE`` would have admitted) and the
       ERR-080 forgery. A single-leg gate could not tell those two
       candidate designs apart, since ``SPHERE`` also refuses the
       forgery. A companion row asserts the chart is **Mode-12 blind**
       to the same forgery while the section is not — i.e. the
       blindness is pinned as a *property*, not merely avoided; and
       ``test_the_half_space_is_CLOSED_because_production_marches_from_it``
       pins the non-strict inequality against the march seed, the only
       witness available since the shipped rule never populates the
       stratum.
   * - The two coordinate systems agree, and the orbit space's
       dimension is a THEOREM
     - ``__post_init__``'s FOUR clauses, each with both legs
       (``vv-principles`` #11) and — since #434 R4 — each with an input
       only IT rejects, so none can be certified by another's witness.
       The **stabiliser** clause is at
       :ref:`manifold-orbit-space-stabiliser`; the **lift-codomain**
       clause refuses a codomain whose ambient width is not the base's
       (`[M]` ``Ball(2)`` and ``[-1,1]`` on an axial entry). The
       **fundamental-domain** clause: a half-meridian
       offered against the 2-D disk is **refused at construction**,
       every shipped entry **satisfies** it, and a third row pins the
       rule that makes one field express both a half-space and a
       hyperplane — :math:`\dim = 2` for a lone normal, :math:`\dim = 1`
       once its antipode joins it. The **orbit-dimension** clause
       (:eq:`manifold-orbit-dimension-law`): the two forgeries of
       :ref:`manifold-dimension-drop` are refused naming the law, every
       constructible entry satisfies it, and a companion row measures
       that :math:`\dim M - \dim H` would be WRONG on :math:`O(3)`
       acting on :math:`S^2` and on :math:`SO(3)` acting on
       :math:`\mathbb{R}^3`. ⛔ The first bullet read *"a hemisphere
       offered against a 1-D realization"* until 2026-09-03; `[M]` that
       input is now rejected by the orbit-dimension clause instead, one
       step earlier, so quoting it would have credited the wrong arm.
   * - ⭐ The ARROWS (``TestManifoldMap``, tracker 2.3)
     - The type's own laws first — a map is a frozen value with two
       endpoints; composition is refused across mismatched endpoints
       (**both** legs, plus the ``TypeError`` on a non-map); the
       pushforward is **functorial**
       (:eq:`manifold-map-functoriality`), nodes and weights by
       ``np.array_equal``. Then the two named maps, each with a
       positive leg *and* a refusal leg: ``archimedes`` lands on
       :math:`S^2` for **all three** axes and collapses the fibre onto
       the stratum, and equals the direction-cosine triple spelled by
       hand; ``barycentre`` lands **inside** the ball and on the sphere
       **only at the poles** — which is the ERR-080 discriminator
       stated as a property of a *map* rather than of a quadrature —
       and satisfies :math:`1-\lVert b\rVert^2 = 1-\mu^2`. ⛔ Its
       refusal leg named *"a mirror quotient, the trivial quotient and
       a bare interval, with an* :math:`O(2)_a` *entry as the positive
       control"* until 2026-09-03; after #434 R4 the first two are
       **answers**, and what the map refuses is a manifold that is not a
       :class:`~orpheus.numerics.manifold.Quotient` at all
       (:ref:`manifold-lift`) — the row was re-keyed onto that and its
       positive control widened to all eight entries. A final row pins
       the Pattern-2 collapse: ``invariance._embedded_nodes`` is
       ``np.array_equal`` to the map it reads
       (:ref:`manifold-barycentre`) — `[M]` 12 of 12 rows, still 12 of
       12 after tracker 2.2b re-routed the read through
       :attr:`Quotient.lift
       <orpheus.numerics.manifold.Quotient.lift>`, and still 12 of 12
       after R4 made that lift a stored field (:ref:`manifold-lift`).

⭐ Two exhaustiveness gates are worth naming separately, because they
are what make "closed sum" a checkable claim rather than a description.
``test_every_variant_is_reachable_from_this_modules_list`` compares
``Manifold.__subclasses__()`` against the module's own exercised list,
so a member added to the module but not to the tests fails **there**;
``test_ambient_dimension_is_defined_for_every_variant`` walks every
shipped variant through the exhaustive ``match``. The benefit of a
closed sum is that an operation can be checked against every member,
and that benefit is only real if the member list is itself pinned.

.. note::

   **A second module carries the CONSUMER-side gates**, and it is where
   this page's basis-facing claims are pinned:
   ``tests/numerics/test_basis_domain.py``, `[M]` **24** collected rows
   (was 13 before tracker 2.1b; the count is the generated V&V matrix's
   ``numerics/test_basis_domain`` row, the same independent instrument
   used above). Section D pins ``domain`` — every shipped basis answers,
   a basis that cannot say what it eats **cannot be constructed**, and
   the flagship ``test_d6`` pins *"the two halves of one frame name ONE
   manifold"*. Section E pins ``invariance_group``
   (:ref:`manifold-basis-invariance-group`). Every row there is
   ``@pytest.mark.foundation`` for the same reason as this module's:
   these are the type's own laws, not an L0–L3 claim about a flux.


.. _manifold-development-history:

Development history
===================

Reverse-chronological changelog of the architectural milestones of
*this page's* subject — the point-set layer. The space layer's own
changelog is at :ref:`spaces-development-history`. Entries marked
*(in development)* live on an unmerged feature branch and have no
landed merge-to-``main`` hash yet; **trust** ``git`` **over this table
for merge status.**

.. list-table::
   :header-rows: 1
   :widths: 10 50 12 28

   * - When
     - Architectural milestone
     - Issue
     - Where
   * - 2026-09-03
     - ⭐⭐ **Invariance is the MEASURE's question, and groups import
       geometry only.** The invariance kernel left
       :mod:`~orpheus.numerics.symmetry` for a module of its own,
       :mod:`~orpheus.numerics.invariance`, and the five verbs moved onto
       :class:`~orpheus.numerics.measure.DiscreteMeasure`
       (``is_invariant_under``, ``certificate_under``,
       ``permutation_under``, ``singular_set_under``, ``symmetry_groups``);
       ``SubgroupOfO3.is_invariant`` is DELETED, with no façade, because a
       façade would have kept the deferred import it existed to justify.
       The arrows now run one way — ``geometry.transformation ← symmetry ←
       manifold ← measure ← invariance`` — so this module imports
       :class:`~orpheus.numerics.symmetry.SubgroupOfO3` at **module**
       scope, :attr:`Quotient.by
       <orpheus.numerics.manifold.Quotient.by>` is annotated with the real
       class, ``_trivial_group`` is retired, and the group members this
       module read — `[M]` 10 distinct over 24 sites at the pre-carve
       commit, by AST on a ``group``/``.by`` receiver — are ordinary reads
       (:ref:`manifold-import-cycle`). Three deletions inside the kernel:
       the ``G ⊆ H ⟹ True`` short circuit (an optimisation the closure
       re-proves — `[M]` it would have fired on **28** (rule × group) rows
       over 11 shipped rules with **0** disagreements, and the predicate
       moved **0 of 330** rows on a fixed 30-spelling list against a
       pinned pre-carve tree), the second inlined copy of the orbit
       closure (three docstrings had claimed "one closure" while two
       functions carried an identical lambda), and the guard that skipped
       the position test when :math:`G^0 \subseteq H` — a test that cannot
       fail passes (:ref:`manifold-kernel-deleted-short-circuit`). The
       candidate set is read off the orbit **barycentres** rather than the
       stored node width, so one fold no longer has two candidate sets by
       spelling: `[M]` the walk on ``gauss_legendre(2/8/16)`` moves
       :math:`\{O(2)_x, \sigma_x\} \to \{D_{2h}, O(2)_x\}` and on
       ``folded_product(4, 6)``
       :math:`\{D_{1h}, \sigma_x\} \to \{D_{2h}\}` — both strengthenings,
       since :math:`D_{2h}` contains every answer it replaces — while
       ``folded_product(4, 8)``'s candidate set falls 20 → 18 (the two
       dropped, :math:`C_4` and :math:`D_{4h}`, are `[M]` not invariances
       of it). ⚠ The axis table had to move BACK to ``symmetry.py`` first,
       or the reversal does not import at all: `[M]` on a renamed shadow
       package, one fresh interpreter per (variant, entry point), **10 of
       10** clean shipped against **3 of 10** with the old axis home — and
       ``import orpheus`` alone stays green under the broken variant.
     - `#434 <https://github.com/deOliveira-R/ORPHEUS/issues/434>`_
     - *(in development)* ``fix/angular-phantom-support``; carve R2 of
       #434
   * - 2026-09-03
     - ⭐⭐ **The lift is a derivation OUTPUT, and an orbit space's
       dimension is a THEOREM.** Two objects an entry was re-deriving at
       read time became fields, and a third became a construction
       invariant. (1) :attr:`Quotient.lift
       <orpheus.numerics.manifold.Quotient.lift>` was a three-arm branch
       on the group's tag whose fall-through said *"add the entry's
       section … to* ``Quotient.lift``\ *"* — a second dispatch over the
       key the catalogue had already used to choose the builder. It is
       now assembled from
       :attr:`~orpheus.numerics.manifold.Quotient.lift_coordinates` and
       :attr:`~orpheus.numerics.manifold.Quotient.lift_codomain`, both
       REQUIRED, and every catalogued entry's lift is ONE formula: the
       **Reynolds projector** :math:`P_H = \int_H \rho(g)\,dg` onto
       :math:`H`'s fixed subspace, read from the chart's side
       (:eq:`manifold-reynolds-projector`). ``_coordinate_chart(columns,
       ambient)`` returns the chart and its lift as a pair, so
       ``embed ∘ select = P_H`` by construction —
       `[M]` ``np.array_equal`` against an SVD reference built from the
       group's realized matrices on **8 of 8** constructible entries ×
       41 seeded unit vectors, ``max|Δ| = 0.000e+00``, with :math:`P_H`
       a 0/1 diagonal on every one; against the finite group's own MEAN,
       ``array_equal`` on all three mirror entries; against a 16-point
       trapezoid over the orbit circle, ``3.331e-16`` on the three axial
       ones. ``_hemisphere_section`` retires with its :math:`\sqrt{\cdot}`
       and its :math:`\rho > 1` refusal, and
       ``ambient_representatives`` becomes
       :meth:`~orpheus.numerics.manifold.Quotient.orbit_barycentres` —
       ONE concept on both coordinate widths, where the ambient arm used
       to pass points through as representatives. (2) The **dimension
       law** :math:`\dim(M/H) = \dim M - \dim(\text{generic orbit})`
       (:eq:`manifold-orbit-dimension-law`), with the orbit's dimension
       the rank of :math:`\{Xp : X \in \mathfrak h\}` and **not**
       :math:`\dim H` — :math:`O(3)` on :math:`S^2` has :math:`\dim H =
       3` and a 2-dimensional generic orbit, so :math:`S^2/O(3)` is a
       point (`GitHub #440
       <https://github.com/deOliveira-R/ORPHEUS/issues/440>`_), and
       :math:`SO(3)` on :math:`\mathbb{R}^3` likewise — is enforced in
       ``__post_init__``, which now carries **four** clauses (stabiliser,
       dimension, the lift's codomain width, the fundamental domain),
       each with an input only IT rejects. The orbit dimension is the
       MAXIMUM over a probe SET, not the value at one point: `[M]`
       orbit dimension is upper semicontinuous, and with a single probe
       row placed ON the axis the one-point spelling both refused the
       honest :math:`S^2/O(2)_z` and admitted the disk forgery. `[M]` before it, a forged
       :math:`S^2/O(2)_z` realized on the DISK and a forged
       :math:`S^2/\sigma_x` realized on :math:`[-1,1]` both CONSTRUCTED
       and compared unequal to the entry they claim to be — ERR-080's
       defect class one field over from the one #432 closed
       (:ref:`manifold-dimension-drop`). (3) :math:`(M/H)/\{e\}` **IS**
       :math:`M/H`: `[M]` until this carve
       ``SPHERE.quotient(Mirror("y")).quotient(Trivial).name`` was
       ``'S^2/sigma_y/Trivial'``, a second object for one orbit space
       inside the spent-group door's own exception; it is now the entry
       by IDENTITY (:ref:`manifold-spent-group-door`). And
       :attr:`SubgroupOfO3.is_trivial
       <orpheus.numerics.symmetry.SubgroupOfO3.is_trivial>` replaces
       five ``name == "Trivial"`` string compares. ⛔ **The invariance
       kernel cannot see any of it** — every downstream answer is read
       through ``orbit_coordinates``, the column selection :math:`P_H`
       re-writes, so `[M]` **0 of 9925** answers of the #434 R1
       behaviour grid moved and ``chart(g·P p)`` is ``array_equal`` to
       ``chart(g·p)`` on every normalising motion. The gates are
       therefore at the AMBIENT tier, with the round trip
       :math:`\pi\circ\lambda = \mathrm{id}` shipped as a *declared
       blind* leg (:ref:`manifold-lift-is-mode-12-blind`). One answer
       moves, and it is a strengthening: the reference harness's mirror
       partner map on a :math:`\sigma_y`-folded rule returns the
       IDENTITY permutation where it used to raise, agreeing with
       :meth:`~orpheus.numerics.quadrature.directional.Quadrature.ordinate_permutation`
       where the two contradicted each other — `[M]` 31 of 33 (rule ×
       axis) rows unchanged, and both folded call sites in the tree pass
       ``axis="x"``.
     - `#434 <https://github.com/deOliveira-R/ORPHEUS/issues/434>`_
     - *(in development)* ``fix/angular-phantom-support``; carve R4 of
       the symmetry-machine review. ⚠ Uncommitted in the working tree
       when this row was written — trust ``git`` over this cell.
       ``numerics/manifold.py``, ``numerics/symmetry.py``,
       ``numerics/basis/descent.py``,
       ``numerics/quadrature/directional.py``
   * - 2026-09-03
     - ⭐⭐ **Every question about a group is COMPUTED from its
       realization, and no relation between two groups is written down
       anywhere.** :math:`\mathfrak{so}(3)` is simple and
       three-dimensional, so its subalgebras are :math:`\{0\}`, one line
       :math:`\mathbb R\,[\hat a]_\times` per axis, and
       :math:`\mathfrak{so}(3)` itself — never dimension **2**
       (:eq:`manifold-so3-subalgebras`, proved in one line from
       :math:`\mathfrak{so}(3) \cong (\mathbb R^3, \times)`). A closed
       subgroup of :math:`O(3)` is therefore exactly (identity
       component, one representative per connected component):
       :class:`~orpheus.numerics.symmetry.IdentityComponent` and
       :class:`~orpheus.numerics.symmetry.Realization`, and
       ``contains`` / ``is_normalised_by`` / ``normalises`` /
       ``identity_component`` / ``dim`` / ``generic_images`` / *"does*
       :math:`G^0` *fix these nodes"* are each ONE body on that pair
       (:ref:`manifold-realization`). `[M]` **thirteen** functions
       retire — twelve private per-family helpers plus the public
       ``is_subgroup_of`` — together with the module's TWO hand-written
       relation surfaces (``_NAMED_LATTICE``'s 8 enum-to-enum edges and
       ``_axial_contains``'s hand-spelled axial rows, each of which had
       shipped a false edge before it went), a group cache and a type
       alias;
       ``_contains`` alone was **109 lines** carrying **28** tag-dispatch
       sites (by AST: 24 ``isinstance`` calls + 4 ``is
       _NamedSubgroup.X`` comparisons), and module-wide the same
       predicate falls from **86** sites to **31**. The normaliser's five
       arms
       collapse into the Lie criterion
       :eq:`manifold-normaliser-lie-criterion`, now carried with its
       proof (:ref:`manifold-normaliser-lie-criterion-section`).
       ⛔ **Nothing observable moved except three intended answers**:
       `[M]` against a pinned pre-carve tree over 27 spellings,
       ``contains`` **0 of 729** ordered pairs, ``normalises`` **0 of
       729**, ``is_invariant_under`` **0 of 270**, the walk **0 of 10** rules,
       and the compatibility law 0 violations on both trees at every
       denominator (57/342, 75/450, and the widened 175/1750). The three
       that move: ``identity_component`` is ``Trivial`` for every finite
       member (**17 of 27** spellings — it returned the group ITSELF,
       contradicting its own docstring, and had **zero readers**);
       ``Cn(1)`` normalises to the ``Trivial`` tag so one group has one
       spelling; and ``dim`` :math:`\in \{0,1,3\}` is NEW, with no
       consumer until R4 reads it for the orbit-space dimension law.
       :class:`~orpheus.numerics.symmetry.SubgroupOfO3` becomes a frozen
       dataclass — before, ``g._tag = …`` succeeded and moved
       ``hash(quotient)`` under three memos keyed on it.
     - `#434 <https://github.com/deOliveira-R/ORPHEUS/issues/434>`_
     - *(in development)* ``fix/angular-phantom-support``; carve R1 of
       the symmetry-machine review. ⚠ Uncommitted in the working tree
       when this row was written — trust ``git`` over this cell.
       ``numerics/symmetry.py``, ``numerics/quadrature/registry.py``
   * - 2026-09-02
     - ⭐⭐ **A symmetry is asked ON the orbit space, and a geometry
       admits a rule on a FOLD of its domain.** The invariance question
       was being asked in the ambient space, where a fold's nodes are
       one representative per orbit and a polar marginal's nodes are a
       chart coordinate; `[M]` :math:`\sigma_y` therefore read *not
       invariant* on the shipped ``folded_product(4, 8)`` while acting
       TRIVIALLY on :math:`S^2/\sigma_y`, and every OWED-symmetry
       admission of the shipped cylinder configuration
       failed on that reading. Three objects land.
       :meth:`SubgroupOfO3.is_normalised_by
       <orpheus.numerics.symmetry.SubgroupOfO3.is_normalised_by>` and
       :meth:`normalises
       <orpheus.numerics.symmetry.SubgroupOfO3.normalises>` decide
       :eq:`manifold-normaliser-descent` — an isometry descends to
       :math:`M/H` iff it normalises :math:`H` — EXACTLY for every
       family, the continuous ones through their identity component and
       their coset representatives rather than by sampling (this row
       read *"through the rotation generator*
       :math:`[\hat e_a]_\times`\ *"* until 2026-09-03, naming the
       finite-:math:`H` case of what is now one Lie criterion,
       :eq:`manifold-normaliser-lie-criterion`); `[M]` the
       four-right-angle sample
       over-certifies on **2 of 8** :math:`(G, H)` pairs, which is
       ERR-072's mechanism recurring in a NEW predicate
       (:ref:`manifold-normaliser-sampling-control`).
       :attr:`Quotient.lift <orpheus.numerics.manifold.Quotient.lift>`
       gives every catalogued family a right inverse of its quotient map
       — the orbit BARYCENTRE for the axial entries (equivariant, not a
       section, which is all an induced action needs), the hemisphere
       section for a mirror, the identity for the trivial entry; `[M]`
       :math:`\pi \circ \lambda = \mathrm{id}` to ``0.000e+00`` on all
       three (:ref:`manifold-lift`). And
       :meth:`Quotient.induced_action
       <orpheus.numerics.manifold.Quotient.induced_action>` is the
       arrow :math:`[p] \mapsto [g\,p]`, refusing a motion outside the
       normaliser (`[M]` :math:`C_4` about :math:`z` on
       :math:`S^2/\sigma_y`). Invariance — spelled ``is_invariant`` on
       the GROUP at that step, and
       ``DiscreteMeasure.is_invariant_under`` since R2 the next day —
       then has ONE kernel: a
       bare support is asked on the trivial orbit space
       :math:`\mathbb{R}^3/\{e\}` — the ambient space and not the
       sphere, because every barycentre and every zero-padded node lands
       off :math:`S^2` — and `[M]` that reduction is identical to the
       old ambient reading on **150 of 150** (sphere rule × candidate
       group) rows — `[M]` re-measured 2026-09-03 the same finding reads
       **144 of 144**, the denominator having moved with the computed
       candidate set (:ref:`manifold-one-invariance-kernel`). The consumer is the registry's
       ledger slot: stage 0 became ONE expression at that step, *what the
       descent arrow SPENDS lies in* :math:`\Gamma`, over the SAME
       arrow a frame's G0 reads — read through a module-level helper
       ``manifold.spent_group``, which R3 of #434 retired the next day
       when the second conjunct moved onto the UNSPENT symmetry
       (:eq:`manifold-gamma-slot-stage-zero`,
       :ref:`manifold-gamma-slot`) — and `[M]` the shipped cylindrical fold
       is admitted at both stages where it was refused at both, with the
       stage-0 refusal count moving **12 → 10 of 20** (constructor ×
       geometry) pairs and no pair moving the other way. `[M]` nothing
       else did:
       the slab rule's answers are unchanged on **0 of 15** candidate
       groups, ``product(4, 8)`` on **0 of 23**, the walk on both is
       unchanged and agrees with brute force **6 of 6**, the
       compatibility law re-runs at **0 violations** over 342 and 450
       (edge × fixture) pairs, and ``_embedded_nodes`` stays
       ``array_equal`` to :func:`~orpheus.numerics.manifold.barycentre`
       on **12 of 12** rows. ``_polar_axis_of`` and
       ``_invariance_on_points`` are RETIRED; the spent-group door
       refuses :math:`(M/H)/G` for :math:`G \subseteq H` with the
       theorem, naming the trivial group as its one exception (an
       identity ENTRY, not a second spelling);
       :meth:`Quadrature.ordinate_permutation
       <orpheus.numerics.quadrature.directional.Quadrature.ordinate_permutation>`
       takes the same route, so `[M]` :math:`\sigma_y` on the fold now
       yields the IDENTITY permutation where it yielded ``None``
       (:ref:`manifold-ordinate-permutation-orbit-space`); and
       ``certificate_under`` follows it too, CLOSING the §II.11 lead —
       `[M]` ``certificate_under(gauss_legendre(8), sigma_x)`` and the
       same call on the BARE chart rule both return a certificate where
       both were refused by SHAPE, while the refusal MESSAGE grows a
       third unnamed arm and is reported rather than repaired
       (:ref:`manifold-orbit-certificate-orbit-space`).
     - `#429 <https://github.com/deOliveira-R/ORPHEUS/issues/429>`_
     - *(in development)* ``fix/angular-phantom-support``; tracker 2.2b,
       user-ruled 2026-09-02 (three rulings). ⚠ Uncommitted in the
       working tree when this row was written — trust ``git`` over this
       cell.
   * - 2026-09-02
     - ⭐⭐ **An orbit space is named by its STABILISER, so it has ONE
       spelling — and the naming law is an accessor plus a construction
       invariant.**
       :class:`~orpheus.numerics.symmetry.O2` joins the lattice: the
       pointwise stabiliser :math:`O(2)_a = C_{\infty v}` of a
       coordinate axis, every rotation about it and every reflection in
       a plane containing it, axis-parameterised beside
       :class:`~orpheus.numerics.symmetry.SO2` and
       :class:`~orpheus.numerics.symmetry.Mirror`.
       :attr:`SubgroupOfO3.orbit_stabiliser
       <orpheus.numerics.symmetry.SubgroupOfO3.orbit_stabiliser>` names
       which group an orbit space is recorded under — it moves exactly
       two lattice members, :math:`SO(2)_a \mapsto O(2)_a` and
       :math:`SO(3) \mapsto O(3)` — and
       :class:`~orpheus.numerics.manifold.Quotient`'s ``__post_init__``
       refuses a non-maximal ``by``, so a mis-named quotient is
       *unspellable* rather than refused at one door. The three axial
       entries are re-keyed ``SO2_a`` → ``O2_a`` (six keys, six entries,
       two procedures, unchanged); the slab and sphere geometries spend
       :math:`O(2)_x`; the Legendre basis's ``invariance_group`` is the
       **full** group its :math:`P_\ell(\mu)` have rather than a lower
       bound; and every axial relation against a finite group is now
       COMPUTED from that group's realization rather than tabulated —
       `[M]` the tabulated arm it replaced answered
       ``SO2('x') ⊉ C_1`` while ``SO2('x') ⊇ Trivial``, one group under
       two spellings and two answers. ⟹ **the over-refusal at #432 is
       gone**: `[M]` the frame admits the Legendre basis on a
       :math:`\sigma_b`-folded rule, :math:`(16, L{+}1)` table with the
       arrow :math:`S^2/\sigma_y \to S^2/O(2)_x` and the isotropic
       field's :math:`\ell \ge 1` moments :math:`\le 1.42\times10^{-15}`,
       while ``axis="y"`` on the same fold stays refused. ⛔ **Nothing
       numerical moved**: `[M]` stage 0 of quadrature selection is
       identical on **24 of 24** (geometry × rule) rows against a pinned
       pre-change tree, and the compatibility law re-runs at **0
       violations over 450** (edge × fixture) pairs over 18 groups (the
       15-group control reproducing its recorded 342/0 exactly)
       (:ref:`manifold-orbit-space-stabiliser`).
     - `#432 <https://github.com/deOliveira-R/ORPHEUS/issues/432>`_
     - ``numerics/symmetry.py``, ``numerics/manifold.py``,
       ``numerics/quadrature/``, ``numerics/basis/``,
       ``numerics/spaces/``
   * - 2026-09-02
     - ⭐⭐ **What DESCENDS is decidable, the descended space has two
       realizations with a ruling between them, and a frame must name
       ONE orbit space — ERR-080 is CLOSED.** #429's fused commit
       (trackers 0.1b + 0.6 + 2.2 + 3.4 + 3.4b), and the first entry in
       this changelog that repairs a wrong answer rather than enabling
       one. Three objects land here. :meth:`Quotient.descending_slots
       <orpheus.numerics.manifold.Quotient.descending_slots>` asks
       fibre-constancy of a basis on the base, on the ENTRY because the
       predicate is a theorem about :math:`\pi` and has two readers;
       for a continuous group it samples INCOMMENSURATE angles, since
       `[M]` four right angles generate :math:`C_4` and falsely admit
       :math:`m = \pm4` at :math:`L \ge 4` (``vv-principles`` #13, with
       a control blind below :math:`L = 4`). `[M]` about :math:`x` at
       :math:`L = 4`: exactly :math:`\{(\ell,0)\}`, **5 real slots of
       25**. :class:`~orpheus.numerics.basis.descent.Descent` carries the
       two realizations and the discriminator — *downstairs iff the
       quotient has a classical named basis* — with the isomorphism
       checkable **at the bit** (`[M]` ``array_equal``, **7 of 7**
       sphere rules at :math:`L = 4`), which is what forces the
       polynomial's SPELLING (no single scipy routine reproduces the
       harmonics' :math:`m = 0` column). And
       :func:`~orpheus.numerics.manifold.quotient_onto` is G0: **one**
       predicate — the arrow ``measure.support -> basis.domain`` exists —
       subsuming the containment as its :math:`K \subseteq H` arm and
       additionally admitting a Legendre basis on a full-sphere rule.
       `[M]` all **seven** shipped pairings measured; ERR-080's is
       exactly the one refused. End to end against a pinned pre-repair
       tree, on the catalogue entry's own fixture: :math:`\phi` is
       ``array_equal`` at :math:`L = 0, 1` and moves **7.765** /
       **3.546** at :math:`L = 2, 3` onto ``+4.000000000000``, and
       ``gauss_legendre(16)`` at :math:`L = 4` stops raising.
       ⛔ **Not closed by it**: construction-time membership (2.0b) — a
       forged measure is still *constructible*, and what is gone is
       every path from one to a basis; and the entry's section. ⛔ This
       list also named *"the over-refusal at* **#432**\ *"* until later
       the same day, when #432 landed (the row above)
       (:ref:`manifold-what-descends`).
     - #429
     - ``numerics/manifold.py``, ``numerics/symmetry.py``,
       ``numerics/frame.py``, ``numerics/basis/``
   * - 2026-09-02
     - **The catalogue entry gets its OWN arrow, and the measure that
       arrow pushes forward — the engine seed closes at 9 of 9.** Two
       fields, and between them they finish the
       :ref:`data-model ruling <manifold-engine-data-model>`.
       ``orbit_coordinates`` stores the **quotient map's** action on
       the base's ambient coordinates — the invariants that survive
       eliminating the base's own ideal, which for every shipped entry
       is a column selection — and :attr:`Quotient.quotient_map
       <orpheus.numerics.manifold.Quotient.quotient_map>` derives the
       typed arrow on top of it, because a frozen dataclass cannot
       *store* an arrow whose codomain is itself. ⭐ **That codomain is
       the ENTRY, never the** ``realization`` (user ruling): read onto
       :math:`[-1,1]` the map is axis-blind, which is exactly the
       reading tracker 2.4 made refusable. Four laws, `[M]` all
       bit-exact — :math:`H`-invariance with a negative leg,
       :math:`\pi_a\circ\varphi_a = \mathrm{pr}_1` on **12 of 12**,
       :math:`\beta_a\circ\pi_a` the axial projection on **3 of 3**,
       and the change of variables on ``level_symmetric(4)`` at
       ``4.18879020478639``, **1 ULP** from :math:`4\pi/3`
       (:ref:`manifold-quotient-map`). :attr:`Quotient.reference
       <orpheus.numerics.manifold.Quotient.reference>` carries
       :math:`\pi_*\,d\Omega` (:eq:`manifold-quotient-pushforward`):
       ``LEGENDRE`` on the three axial entries by Archimedes' hat-box;
       ``None`` on the three mirrors, whose pushforward is the
       **weighted disk measure**
       :math:`2\,du\,dv/\sqrt{1-u^2-v^2}` that no shipped
       :class:`~orpheus.numerics.exactness.ReferenceMeasure`
       realization spells, and on :math:`M/\{e\}`, whose answer is a
       property of the *base* — both ``None``\ s user-ruled. ⭐ The
       registry now **reads** that field, collapsing the campaign's
       **second** Pattern-2 twin after ``support`` at 2.4, with its
       bare-sphere arm deliberately kept because a geometry that spends
       nothing is handed the base (:ref:`manifold-second-twin-reference`).
       ⭐ The mechanism is itself the lesson: the **type** rides a
       ``TYPE_CHECKING`` import and the **value** a *function-scope*
       one, `[M]` alive on **7 of 7** fresh import orders where every
       module-scope placement — top of the file and bottom alike — dies
       on **7 of 7**, because a guard defers a *name* and can never
       carry a *value* (:ref:`manifold-value-at-function-scope`).
       ⛔ **An enabler, not a repair, and its two halves differ:**
       `[M]` ``reference`` has **one** production reader and
       ``quotient_map`` **zero**, its ten occurrences all in one test
       module; the entry's **section** still does
       not ship (a section is a choice, not a derivation output); and
       ERR-080 keeps its three ``xfail(strict=True)`` rows.
     - `#429 <https://github.com/deOliveira-R/ORPHEUS/issues/429>`_
     - *(in development)* ``fix/angular-phantom-support``; tracker 3.1.
       ⚠ The code was **uncommitted in the working tree** when this row
       was written — trust ``git log`` over this cell for its hash.
   * - 2026-09-02
     - **The category gets its ARROWS, and a codomain stops being
       something a caller can assert.** Every construction that moved
       a point set had been applying a callable and then *naming the
       destination by hand* — which is the exact shape of
       :ref:`ERR-080 <manifold-err-080>`, since a destination named at
       the call site is a claim nothing can contradict.
       :class:`~orpheus.numerics.manifold.ManifoldMap` makes it a
       field: one frozen value type (``domain``, ``codomain``,
       ``apply``) with named maps as **factories**, ruled that way
       (user, 2026-09-02) for the same reason
       :data:`~orpheus.numerics.manifold.SPHERE` is a value and not a
       subclass. :meth:`DiscreteMeasure.pushforward
       <orpheus.numerics.measure.DiscreteMeasure.pushforward>` retires
       ``new_space=`` and **reads** its target, and additionally
       refuses a map out of the wrong point set — by manifold VALUE,
       which is what makes it discriminating: `[M]` the slab's
       :math:`S^2/O(2)_x` rule and the chart rule on :math:`[-1,1]`
       have ``np.array_equal`` nodes and only one of them is accepted.
       Three arrows are typed — the **Archimedes** chart
       :math:`[-1,1]\times S^1 \to S^2` (named for the hat-box
       theorem; `[M]` :math:`\pi \circ \varphi_a = \mathrm{pr}_1`
       bit-exactly, and the product rule is now
       ``(polar * azimuthal).pushforward(archimedes("z"))``,
       bit-identical to its retired hand loop on **60 of 60**
       configurations with its support **identical** to the chart's
       codomain); the orbit **retraction** inside ``quotient()``, now
       landing on the catalogue's own object; and the orbit
       **barycentre** :math:`\mu \mapsto \mu\,\hat e_a`, whose honest
       codomain is ``Ball(3)`` because
       :math:`1 - \lVert\mu\hat e_a\rVert^2 = 1-\mu^2 = \tfrac14\det P`
       — it lands ON the sphere only at the stratum, so it is
       **canonical precisely because it is not a section**. ⭐ That
       gives ERR-080 a one-sentence statement in the type system's own
       vocabulary — *the barycentre map with a forged codomain* —
       and `[M]` the forgery's nodes are ``np.array_equal`` to the
       honest map's image, so what is false about it is a **type** and
       nothing else. The honest spelling
       (``invariance._embedded_nodes``) read the map from this step on,
       collapsing a Pattern-2 twin (`[M]` bit-identical on 12 rows; the
       read moved behind :attr:`Quotient.lift
       <orpheus.numerics.manifold.Quotient.lift>` at tracker 2.2b and
       `[M]` is still bit-identical on 12 rows).
       ⛔ **An enabler, not a repair**: no membership check runs inside
       a map, the forgery arm stays a raw constructor **by design**
       until tracker 3.4, and the gate still declares three
       ``xfail(strict=True)`` rows (:ref:`manifold-arrows`).
     - `#429 <https://github.com/deOliveira-R/ORPHEUS/issues/429>`_
     - *(in development)* ``fix/angular-phantom-support``
       (``5ec3a00a``); tracker 2.3. (⛔ This cell read *"the code was
       **uncommitted in the working tree** when this row was written"*
       until the hash landed the same day — the hedge was honest and is
       superseded, which is the state a hedge is supposed to reach.)
   * - 2026-09-01
     - **A basis learns what it EATS — and therefore what symmetry it
       HAS.** :class:`~orpheus.numerics.basis.base.Basis` gained the
       level-1 slot the three-level table had listed as ⛔ *nothing*
       (:ref:`manifold-three-levels`): ``domain``, a :class:`Manifold`,
       abstract on the ABC so a basis that cannot say what it consumes
       **cannot be constructed**. That closed a live falsehood — `[M]`
       ``basis/indicator_basis.py`` hard-coded its coefficient space's
       name as ``f"L2[coarse_cells_R{ndim}]"``, so a 2-group **energy**
       basis and a 2-cell **spatial** basis compared ``==`` *and*
       hash-equal; they do not now
       (:ref:`manifold-string-algebra`). ⭐ Assigning the type was itself
       a census: it separated the continuous energy axis in eV
       (:class:`Interval`) from the multigroup *index* axis
       (:class:`EnergyGroups`), which the tag ``"energy"`` had conflated
       at equal ambient dimension. **Then the same slot answered a
       second question for free.** A function on :math:`M/H` *is* an
       :math:`H`-invariant function, so
       :attr:`~orpheus.numerics.basis.base.Basis.invariance_group` is a
       ``match`` on ``domain`` — `[M]` **6 of 6** shipped bases answer,
       ``@final``, **0** subclass edits, no new field. The tracker had
       recorded the property as *absent and derivable*, which invited six
       overrides; the phase opener dissolved them, exactly as tracker
       2.0d's ``quotient_group`` **field** had dissolved into
       :attr:`Quotient.by <orpheus.numerics.manifold.Quotient.by>` one
       step earlier. With both operands in hand
       the ERR-080 pairing became a **lattice verdict** — `[M]`
       ``Trivial ⊇ SO2('x')`` is ``False`` for the slab, while the
       shipped fold's two halves are literally one group object
       (:ref:`manifold-basis-invariance-group`). ⛔ **Nothing refuses on
       that verdict**: the frame's pairing gate is tracker 2.2, and
       ERR-080 stays open, held by its ``xfail(strict=True)`` gate.
     - `#429 <https://github.com/deOliveira-R/ORPHEUS/issues/429>`_
     - *(in development)* ``fix/angular-phantom-support``
       (``c461fe8d`` for 2.1, ``9b4a4d9c`` for 2.1b). (⛔ This cell read
       *"2.1b was **uncommitted in the working tree**"* until its hash
       landed.)
   * - 2026-09-01
     - **The axial rotation group gets its AXIS, and the type gets its
       first production consumer.** :math:`SO(2)` left the parameter-free
       enum and became ``SO2(axis)``, beside ``Mirror(axis)`` and for
       the same reason one month later: `[M]` the tree carries **two
       poles** — the real spherical-harmonic basis and the slab's polar
       marginal are about :math:`x`, every product rule's polar factor
       and the finite families are about :math:`z` — and one
       Gauss–Legendre rule serves both roles, so the group a marginal
       was quotiented by cannot be spelled without its axis
       (:ref:`manifold-so2-axis-is-a-parameter`). The catalogue went
       from four keys to **six**, still two procedures, because the
       :math:`SO(2)` derivation now reads its axis off the group exactly
       as the mirror one does (:ref:`manifold-s2-so2`). Downstream, the
       slab's rule **declares** its orbit space through a new measure
       verb, :meth:`on_orbit_space
       <orpheus.numerics.measure.DiscreteMeasure.on_orbit_space>` — same
       atoms, new support — so `[M]`
       ``gauss_legendre(8).measure.support.name`` is ``'S^2/O2_x'`` and
       an 8-node angular space no longer compares equal to an 8-node
       spatial rule on the same interval, which it did before
       (:ref:`manifold-orbit-space-declaration`). ⭐ That collapsed the
       registry twin this page had listed as a seam: ``AngularSymmetry``
       now *calls* ``SPHERE.quotient`` (:ref:`manifold-twin-lookup`).
       ⚠ Two memos landed with it, and they are not optimisation
       polish: the catalogue derivation is ~6 ms of SymPy and every slab
       quadrature now carries one, and the icosahedral operator set was
       being rebuilt tens of times per invariance walk once the axial
       family offered three axes
       (:ref:`manifold-quotient-is-memoised`). ⛔ **ERR-080 remains
       open**; what this bought it is the *vocabulary* for its section,
       not the section
       (:ref:`manifold-the-axis-convention-for-a-section`).
     - `#429 <https://github.com/deOliveira-R/ORPHEUS/issues/429>`_
     - *(in development)* ``fix/angular-phantom-support``
       (``17501245``); tracker 2.4. (⛔ This cell read *"the code was
       **uncommitted in the working tree**"* until the hash landed.)
   * - 2026-08-31
     - **An orbit space gets its second coordinate system, and the
       catalogue its second derivation.** Deriving the shipped
       cylindrical fold :math:`S^2/\langle\sigma_y\rangle` produced an
       object the single-slot type could not hold: because :math:`H` is
       **finite** the dimension does not drop, so the invariant chart
       buys no reduction and a *section* is canonical — while every
       measure the tree emits through ``.quotient(...)`` already speaks
       the section's coordinates. Ruled: **two slots**. ``realization``
       keeps its documented meaning (the chart's codomain, in the
       invariants' language) and ``fundamental_domain`` carries the
       section's image, in the base's; ``contains`` accepts both and
       dispatches on ambient width, ``_ambient`` still reports the
       chart's, and ``__post_init__`` gates their dimensions
       (:ref:`manifold-two-coordinate-systems`). Two variants were
       minted *by the derivation*, ``Ball`` and ``FundamentalDomain``,
       and ``singular_stratum`` was retyped from ``tuple[float, ...]``
       to a symbolic **locus** — the :math:`\sigma_y` stratum is a
       circle, and the first entry's shape had become the field's type
       (:ref:`manifold-stratum-is-a-locus`). Four candidate single-slot
       realizations were measured and refused, including the disk
       alone, which is **Mode-12 blind** to ERR-080
       (:ref:`manifold-realization-refuted`). ⭐ The finding that pays
       for the rest: **ERR-080's level-1 half is a botched section of**
       :math:`S^2/O(2)_a` — a chart where a section was needed, faked by
       zero-padding off the sphere
       (:ref:`manifold-err-080-is-a-section`). Three mirror keys ship,
       one procedure.
     - `#429 <https://github.com/deOliveira-R/ORPHEUS/issues/429>`_
     - *(in development)* ``fix/angular-phantom-support``
       (``b55bba56``); tracker 1.1 for the entry, user ruling of
       2026-08-31 for the two slots
   * - 2026-08-31
     - **The manifold becomes an OBJECT.** Level 1 of the three-level
       stack stops being ``Space = str`` — an opaque tag whose own
       comment called the entries *"recommendations, not
       constraints"* — and becomes a closed sum with an algebra:
       ``dim`` / ``name`` / ``contains`` / ``__mul__`` total on the
       base, and the invariant-theoretic derivation fields on
       :class:`~orpheus.numerics.manifold.Quotient` alone. `[M]`
       **eight** variants at this commit (ten after the ruling above),
       two of them consolidating families the string
       vocabulary had spelled several ways
       (:ref:`manifold-members`). The three morphisms the mint names
       were already running as string interpolation at
       ``measure.py:588`` / ``:1022`` / ``:802``
       (:ref:`manifold-string-algebra`). The first catalogue entry
       ships, ``S^2/O(2)_a = [-1,1]``, derived by the Procesi–Schwarz
       procedure and carrying its own symbolic regression tests
       (:ref:`manifold-s2-so2`) — which, because the entry's fields
       *are* the procedure's outputs, are the deferred derivation
       engine's acceptance suite, written before it
       (:ref:`manifold-engine-seed`). The identity quotient
       :math:`M/\{e\} = M` is **derived, not tabulated**, and doubles as
       a positive control on the machinery
       (:ref:`manifold-twin-lookup`). ⛔ **Type only: zero production
       consumers, and ERR-080 remains open**
       (:ref:`manifold-seams`).
     - `#429 <https://github.com/deOliveira-R/ORPHEUS/issues/429>`_
     - *(in development)* ``fix/angular-phantom-support``
       (``b8c05d16``); tracker 2.0a, user ruling D0.7 for the mint and
       2.0a-R for the shape


References
==========

* Procesi, C. and Schwarz, G. (1985). "Inequalities defining orbit
  spaces." *Inventiones Mathematicae* **81**, no. 3, 539–554,
  doi:10.1007/BF01388581. The theorem :eq:`manifold-procesi-schwarz`
  transcribes: the image of the orbit map is cut out of the syzygy
  variety by the single condition
  :math:`P \succeq 0` on the gradient Gram matrix of the invariants.
  ⚠ **Not held locally.** `[M]` 2026-08-31 the paper is absent from
  ``scratch/literature/`` (78 items, all reactor-physics and transport
  literature) and has no OCR sidecar, so no page- or equation-level
  verification against the scan was possible. The statement above is
  the standard form of the theorem and the volume/year are
  over-determined and consistent (*Inventiones* **81** is 1985); a
  session that acquires the paper should verify the theorem's own
  numbering and record it here.
* Schwarz, G. (1975). "Smooth functions invariant under the action of a
  compact Lie group." *Topology* **14**, 63–68. The result behind step
  2 — that the orbit map is proper and separates orbits, so smooth
  invariants factor through it.
* Weyl, H. (1946). *The Classical Groups: Their Invariants and
  Representations*, 2nd ed. Princeton University Press. Finite
  generation of the invariant ring (step 1), for the classical groups
  this corpus uses.
* Satake, I. (1956). "On a generalization of the notion of manifold."
  *Proceedings of the National Academy of Sciences* **42**, 359–363.
  The original definition of a V-manifold — what is now called an
  **orbifold** — which is what :math:`S^2/O(2)_a` is and a quotient
  manifold is not (:ref:`manifold-singular-stratum`).
* Helgason, S. (1984). *Groups and Geometric Analysis*. Academic Press.
  Chapter IV (spherical functions on a Gelfand pair; the Funk–Hecke
  theorem as the :math:`(SO(3), SO(2))` instance), which is the
  literature behind :ref:`manifold-gelfand`.
* Hamermesh, M. (1962). *Group Theory and Its Application to Physical
  Problems*. Addison-Wesley. §2.5 (finite point groups) — the source
  :doc:`/theory/foundations/discrete_measures` already cites for the
  group side of the same construction.
