.. _function-spaces:

==========================================================
Function Spaces: Axes, Measure, and the Collapse Doctrine
==========================================================

.. contents:: Contents
   :local:
   :depth: 2


.. Machine header — the ``nexus-meta`` schema for this page (PROVISIONAL).
.. Seeded at campaign-1 CS1 (2026-08-20) as ``field_algebra``'s sibling:
.. that page owns the ELEMENT algebra, this one owns the SPACES those
.. elements live in. The schema is provisional pending a full re-audit of
.. the corpus.

.. dropdown:: Machine header — ``nexus-meta`` schema (PROVISIONAL)
   :color: muted

   .. code-block:: yaml

      module: numerics
      concept: function_spaces
      role: "the space layer — a function space as the ordered product of its AXES (index shape, factor measure, basis kind, the generator that minted it, and the structural identity that deliberately excludes that generator), the counting-measure theorem on the energy axis, and the collapse doctrine that decides which axes survive a degeneracy and why"
      depends_on: [field_algebra, frame]
      related: [manifolds, discrete_measures, operator_algebra, operator_adjoint]
      status: "seeded at campaign-1 CS1 (the Energy axis); the generator slot landed at CS5 (2026-08-29); the densifying half of the `*` product retired at CS4c step 6 item 6.2a (2026-09-07); the Harmonic axis subclass (with the Legendre / spatial-moment ones) is CS4c step 6 item 6.2c, ruled 2026-09-07 and sequenced after 6.2b; the Spatial / Quadrature subclasses stay CS2 and unscheduled"


This page develops the **space layer**: what a discrete function space
*is*, what it is made of, and what happens to it when a physical
degeneracy collapses one of its factors. It is the companion to
:doc:`/theory/foundations/field_algebra`, which types the **elements**
(a flux is a point of the positive cone :math:`K \subset V`); this page
types :math:`V` itself.

The organizing claim is one sentence: **a function space is the ordered
product of its axes, and an axis carries exactly five things — an index
shape, a factor measure, a basis kind, the generator object that minted
it, and a structural identity that deliberately EXCLUDES that
generator.** Everything else on this page follows from taking that
seriously: why the energy metric is the identity as a *theorem* rather
than a default, why the homogeneous solver's spatial factor survives its
own collapse while the angular factor of a scalar space does not, why an
axis can hand a consumer back the direction cosines it forgot, and why
"two copies of :math:`\mathbb{R}^n` with different inner products are
the same space" is a claim this corpus has **overturned**.

.. warning::

   **"Generator" means two different things one paragraph apart, and
   the page used to run them together.** Until 2026-08-29 the sentence
   above read *"…and the identity of the generator that produced it"*,
   which conflated (a) the axis's structural identity being **typed per
   subclass** — an :class:`~orpheus.numerics.axis.EnergyAxis` is not a
   generic :class:`~orpheus.numerics.axis.Axis`, so identity records
   *what KIND of generator* produced the factor — with (b) the axis
   recording *WHICH generator INSTANCE* produced it. Only (a) existed
   before campaign-1 phase CS5. (b) now exists as the
   :attr:`~orpheus.numerics.axis.Axis.generator` slot, and its governing
   ruling is the exact opposite of what the old wording implied:
   **provenance is never identity** (:ref:`spaces-axis-generator`).

.. note::

   **Three unrelated things in this corpus are called an "axis". Keep
   them apart.**

   - A **space-factor axis** (:class:`~orpheus.numerics.axis.Axis`,
     :class:`~orpheus.numerics.axis.EnergyAxis`) — *this page's*
     subject. One tensor factor of a function space; it carries a
     measure and a basis kind, and its own structural slots know nothing
     about geometry. ⚠ Since CS5 it also records the object that minted
     it, and *that* object may well be geometric — the spatial factor's
     generator is the carrier's volume measure, whose nodes are cell
     centres. Geometry is therefore reachable *through* the axis without
     being *part of* it (:ref:`spaces-axis-generator`); the identity key
     is unmoved, which is the whole point of the exclusion.
   - A **geometric axis** (``Axis1D`` / ``AxisMesh`` /
     ``RadialAxisMesh`` in :mod:`orpheus.transport.mesh.axis`) — one
     coordinate DIRECTION of a structured mesh, carrying edges, face
     labels and a coordinate system. The name is a known misnomer
     (it declares per-axis geometry and creates no mesh); the rename is
     tracked as issue **#393**, and the two vocabularies are being kept
     deliberately distinct in the meantime. Where both are in play the
     code spells the space-factor one out: ``from orpheus.numerics.axis
     import Axis as SpaceFactorAxis``
     (:mod:`orpheus.transport.mesh.material_mesh`).
   - A **symmetry axis** — the invariant line of a rotation or mirror,
     as used throughout :doc:`/theory/foundations/discrete_measures`
     and the quadrature machinery.

   ⚠ The first two collide on an attribute NAME, on one object. `[M]`
   for a carrier ``mm``, ``mm.axes`` is the GEOMETRIC tuple
   (``(AxisMesh,)`` — the mesh's coordinate directions) while
   ``mm.bulk_space.axes`` is the SPACE-FACTOR tuple
   (``(EnergyAxis, Axis)``). They are different attributes of different
   objects that happen to share a spelling; neither is derived from the
   other.

.. note::

   **Two symbols on this page look alike.** :math:`V` (italic) is the
   **function space** — the object this page is about — and
   :math:`V_{\rm cell}` is a **cell volume**, a weight on the spatial
   axis. They meet constantly here (the whole point of clause 1 is that
   a spatial factor's weight is what distinguishes two spaces), so the
   volume is always written out in full. This follows
   :doc:`/theory/foundations/field_algebra`, which draws the same
   distinction for the same reason.

.. admonition:: Key Facts (the space layer)
   :class: tip

   - **A space IS its axis tuple.** Shape is the concatenation of the
     axes' shapes; the metric is the tensor product of the factor
     measures, stored PER AXIS and never densified
     (:eq:`spaces-axis-product`;
     :meth:`FunctionSpace.of_axes
     <orpheus.numerics.space.FunctionSpace.of_axes>`).
   - ⚠ **…and a space is NOT a domain.** There are three levels — the
     manifold :math:`M`, the fields on it :math:`L^2(M)`, and the
     coefficients :math:`\mathbb{R}^K` — and a
     :class:`~orpheus.numerics.space.FunctionSpace` is the **second**.
     A basis function eats a *point* of :math:`M`, so this page's
     object can never be a basis's domain, and the frame's level-2
     arrow type-checking says nothing about the level-1 pairing (which
     is how :ref:`ERR-080 <manifold-err-080>` survived). Level 1 was a
     bare ``str`` until 2026-08-31 and now has its own type and its own
     page: :doc:`/theory/foundations/manifolds`.
   - **An axis is (index shape, factor measure, basis kind, generator)
     plus a structural identity over the first three.**
     ``weights=None`` **is** the counting measure — deliberately and
     always; an axis has no "unbound" state, so the legacy two-state
     ambiguity of ``inner_product_weights`` cannot arise on this type.
   - **The metric is an OBJECT, and a space has exactly ONE source for
     it.** Since campaign 1 P7 a space may carry a typed
     :class:`~orpheus.numerics.metric.HilbertMetric` — the Hadamard
     weight is its diagonal special case — so a form with off-diagonal
     structure is expressible at last; resolution is
     ``metric`` > ``inner_product_weights`` > Euclidean, with axis-built
     spaces routing through their axes instead, and a construction guard
     refuses any two sources at once. ⚠ Consequently
     ``inner_product_weights is None`` **no longer implies Euclidean** —
     read a space's pairing through
     :meth:`~orpheus.numerics.space.FunctionSpace.inner_product`, never
     through the slot (:ref:`spaces-metric-object`).
   - **An axis is a FORGETFUL MAP from its generator, and the mint is
     its section.** The axis keeps the weights and drops the nodes; since
     CS5 the mint routes THROUGH the generator
     (:meth:`measure.axis(label) <orpheus.numerics.measure.DiscreteMeasure.axis>`
     / :meth:`quad.axis(label) <orpheus.numerics.quadrature.directional.Quadrature.axis>`),
     which records itself in
     :attr:`~orpheus.numerics.axis.Axis.generator`, so the forgetting is
     **recoverable**: ``a.generator.axis(a.label) == a``
     (:eq:`spaces-axis-generator-section`). A consumer holding the SPACE
     recovers ``mu_x`` / ``eta`` / ``mu_z`` / ``level_indices`` without
     being handed the quadrature separately
     (:ref:`spaces-axis-generator`).
   - ⛔ **The generator is provenance, and its exclusion from identity is
     STRUCTURALLY MANDATORY, not taste.** A
     :class:`~orpheus.numerics.quadrature.directional.Quadrature` is
     unhashable and a
     :class:`~orpheus.numerics.measure.DiscreteMeasure` is
     un-``==``-able, so an identity key containing either makes
     ``Axis.__eq__`` and ``hash(Axis)`` **RAISE** — measured, not
     conjectured (:ref:`spaces-generator-identity-exclusion`). Because
     the ``of_axes`` name digest rides the same key, an inclusion would
     also split the space identity of every carrier that builds its own
     rule instance.
   - **The two one-line discriminators of the collapse doctrine.**
     *(i)* **Can the admissible fields be integrated over the collapsed
     domain?** — symmetry-forced constancy on infinite-measure orbits
     means no, so the collapse must NORMALIZE. *(ii)* **Is the
     surviving convention consulted INSIDE the family, or only at
     re-embedding?** — inside ⟹ the axis persists; only at re-embedding
     ⟹ the convention lives on the arrow and the axis DROPS.
   - **A collapse that DROPS an axis is realized by two typed arrows,
     never one.** :meth:`FunctionSpace.retraction
     <orpheus.numerics.space.FunctionSpace.retraction>` is fiber
     integration :math:`R = \pi_*`; :meth:`~orpheus.numerics.space.FunctionSpace.section`
     is its right inverse :math:`E` with :math:`R\circ E = \mathrm{id}`.
     They differ by exactly the axis's total mass
     (:math:`R^\dagger = \Sigma w\,E`, the *plain* broadcast versus the
     normalized one), and they carry different TYPES so that scalar
     cannot be dropped at a call site. Both are induced by a rank-one
     indicator frame and memoized on the space
     (:ref:`spaces-collapse-pair`).
   - **EnergyGrid is a 1-D mesh in energy** — groups are its cells,
     group boundaries are its faces, and condensation is the
     mesh-overlap map
     (:meth:`EnergyGrid.overlap_to
     <orpheus.data.energy_grid.EnergyGrid.overlap_to>`). The one-group
     member is the one-CELL energy mesh; it keeps its edges because they
     define :math:`\bar\sigma`.
   - **The quotient point records the DENSITY CONVENTION.** The
     homogeneous solver's spatial factor is not absent and not
     measureless: it is a one-point axis whose unit weight *is* the
     normalized "per unit volume" convention, and the pairing consumes
     it. A genuine one-cell mesh with :math:`V_{\rm cell} \neq 1` is a
     **different** space.
   - **The energy metric is the identity as a THEOREM.** Multigroup
     flux components are group INTEGRALS (covariant, extensive) and
     cross sections are flux-weighted group AVERAGES (contravariant,
     intensive), so :math:`\int\sigma\varphi\,\mathrm{d}E =
     \sum_g \sigma_g\varphi_g` exactly and no group widths appear
     (:eq:`energy-condensation-counting-measure`). Consequence:
     :math:`V \cong V^*` isometrically along energy and the adjoint
     there is the plain transpose. A weighted
     :class:`~orpheus.numerics.axis.EnergyAxis` is REFUSED at
     construction.
   - **NODAL vs MODAL is the coordinate-cone question.** A nodal factor
     has one (components are cell/point values); a modal factor does not
     (components are expansion coefficients). ``has_coordinate_cone`` is
     three-valued — ``True`` / ``False`` / ``None`` for legacy spaces —
     and :meth:`Field.cone_violations
     <orpheus.numerics.field.Field.cone_violations>` consults it,
     REFUSING on ``False`` rather than manufacturing violations out of a
     basis choice.
   - **Identity is STRUCTURAL for an axis-built space** — the identity
     flip (structural ``__eq__``), CS4c step 6, 2026-09-07. The chartered
     doctrine is *identity = the axes' structural content*, so **metric
     differences imply space differences**, and since the flip that holds
     DIRECTLY rather than through a bridge: an axis-built space compares
     and hashes by its ``axes`` tuple, and
     :class:`~orpheus.numerics.axis.Axis` equality is structural content
     (type, label, shape, kind, measure bytes) with ``generator``
     deliberately excluded. An **axes-less** space keeps the nominal
     ``(name, shape)`` identity: for the digest-named composites and
     traces (five classes, four digest-folding factories) that IS
     content identity because the factory folds content into the name
     (CS4b S3); for the family-tagged moment heads
     (``'spherical_harmonic_space'``, ``'legendre_space(…)'``,
     ``'spatial_moment_space'``) it is family + dimension, deliberately
     metric-blind until the heads become axis-built (step 6 items
     6.2b/6.2c); a hand-built legacy space carries whatever its author
     wrote. An axis-built space is never equal to a hand-named one
     wearing its label. Until the flip the doctrine flowed through a
     BRIDGE — the injectively derived name
     (:ref:`spaces-identity-bridge`).


.. _spaces-the-axis:

The axis: five slots, and what each one decides
===============================================

An **axis** is one tensor factor of a function space — the value object
recording *(index shape, factor measure, basis kind, generator)*, plus a
structural identity derived from the first three. It is the unit the
composition machinery reasons about: partitions, collapses, frames and
(later) :math:`\oplus`-lifts act **per axis**, never on an anonymous
position of a monolithic shape tuple.

.. list-table:: The five slots
   :header-rows: 1
   :widths: 16 30 54

   * - Slot
     - Type
     - What it decides
   * - ``shape``
     - ``tuple[int, ...]``, rank :math:`\ge 1`
     - The factor's index set. Rank :math:`> 1` is admissible and
       deliberate: a spherical-harmonic axis is :math:`(L+1, 2L+1)`, and
       a rank-:math:`d` spatial axis is a legal CS2 design choice. Rank
       0 is refused — a factor with no index set has nothing to measure.
   * - ``weights``
     - ``NDArray | None`` over exactly ``shape``
     - The **factor measure**. ``None`` IS the counting measure. This is
       the slot the pairing consumes, and (through identity) the slot
       that makes a quotient point and a one-cell mesh different spaces.
   * - ``kind``
     - :class:`~orpheus.numerics.axis.BasisKind`
     - ``NODAL`` ⟹ the factor carries a **coordinate cone**;
       ``MODAL`` ⟹ it does not. Keyword-only with **no default** — the
       basis character is physics and must be spelled at every mint.
   * - ``generator``
     - ``DiscreteMeasure | Basis | Quadrature | None``, keyword-only,
       default ``None``
     - **Provenance, never identity** (CS5). The object that minted this
       factor, kept so the axis's forgetting is recoverable: an axis
       drops its generator's NODES, and this slot is how a consumer
       holding only the space gets them back. ``None`` is an honest
       reading wherever no generator object exists.
       :ref:`spaces-axis-generator` develops it.
   * - identity
     - structural, **per subclass**
     - :math:`(\text{type}, \text{label}, \text{shape}, \text{kind},
       \text{weights bytes})` plus each subclass's own data. Two axes
       differing only in measure are DIFFERENT axes; two axes differing
       only in ``generator`` are the SAME axis.

Composition, and the metric that comes with it
----------------------------------------------

A space is the ordered product of its axes. Two things are determined by
that sentence alone, and both are what
:meth:`FunctionSpace.of_axes
<orpheus.numerics.space.FunctionSpace.of_axes>` implements:

.. math::
   :label: spaces-axis-product

   V \;=\; \bigotimes_{a} V_a,
   \qquad
   \mathrm{shape}(V) \;=\; \mathrm{shape}(V_1) \frown \cdots \frown
   \mathrm{shape}(V_n),
   \qquad
   G_V \;=\; \bigotimes_{a} G_a,
   \quad
   G_a = \operatorname{diag}(w_a) \ \text{ or } \ I,

.. (vv-status rationale) Structural/representational identity: it STATES
   what ``FunctionSpace.of_axes`` constructs (shape = concatenation of
   the factors' shapes; metric = tensor product of the per-axis factor
   measures, applied factor-by-factor and never materialized). Not a
   solver claim — no flux, no eigenvalue, no discretization error. The
   verifiable content is the CS1 foundation battery
   (``tests/numerics/test_space_of_axes.py``: shape concatenation, the
   per-axis metric against an independently-built dense reference, the
   no-densification proof, and the derived name's determinism across
   processes) plus ``tests/numerics/test_axis.py`` for the factor laws.
.. vv-status: spaces-axis-product documented

where :math:`\frown` is tuple concatenation and :math:`w_a` is axis
:math:`a`'s weight array. The second half is the operational content:
the measure lives **per axis**, so composing two 2000-point weighted
axes stores :math:`2 \times 2000` weights and never the
:math:`4{,}000{,}000`-entry outer product. The metric is *applied*
factor by factor — each axis's weights broadcast into their own slot of
the element, with leading and trailing ones for the neighbouring
factors — so an interior weighted axis works exactly like a leading one.
That is a position the legacy prefix-broadcast convention could not
reach.

.. warning::

   **The legacy twin survives; its DENSIFYING half retired 2026-09-07**
   (CS4c step 6 item 6.2a). ``V * W`` (the ``*`` dunder →
   :class:`~orpheus.numerics.space.TensorProductSpace`) is still the
   PRE-axis composition mechanism — it takes whole *spaces* as factors
   and derives its name by joining theirs — but it no longer builds a
   dense weight array. It never populates ``inner_product_weights`` at
   all. Two arms:

   * **Every factor axis-built** → the product's ``axes`` is the
     concatenation of the factors' axes, and there is no metric object:
     the per-axis path of :eq:`spaces-axis-product` applies unchanged.
     (This arm is untouched by 6.2a.)
   * **Any factor axes-less** → the product's ``axes`` stays ``None``
     (an axis is never fabricated for a space that did not declare one)
     and the metric is a lazy
     :class:`~orpheus.numerics.metric.FactoredMetric` carrying ONE
     positioned entry per **axis** of an axis-built factor (a
     :class:`~orpheus.numerics.metric.DiagonalMetric` on that axis's
     block, or ``None`` for a counting-measure axis), one entry per
     dense-slot leaf factor, and a factor's own metric object riding
     verbatim — a nested ``FactoredMetric`` flattens, a
     :class:`~orpheus.numerics.metric.DenseMetric` positions on its
     block. Every block Euclidean ⟹ no metric at all.

   ⛔ **This block read** *"The legacy twin is still live, and CS2
   retires it … on all-diagonal factors it DENSIFIES the metric into an
   outer-product* ``inner_product_weights`` *… CS1 keeps it — it threads
   the* ``axes`` *record when both sides carry one, and bridges
   axis-borne measures into its dense weights on mixed products … and
   CS2 collapses the live mints onto axis concatenation and retires the
   densifier"* **until 2026-09-07**, and that was an accurate
   description of the tree until item 6.2a. The product formed
   :math:`w_V \otimes w_W` once and stored it; a *mixed* product (one
   axis-built factor, one not) therefore had to BRIDGE the axis-borne
   measure into that dense slot, because a weighted axis-built factor
   read through the dense slot alone would have been treated as
   Euclidean — a value bug, not a representation choice. The bridge and
   the outer-product builder retired together with the slot they fed.

   What survives of the old rule is the half that was never about the
   densifier: **new axis-aware code composes with** ``of_axes``, because
   ``of_axes`` is the only ROOT producer of an ``axes`` record — ``*``
   and :meth:`dual <orpheus.numerics.space.FunctionSpace.dual>` merely
   thread one through, so both need an axis-built ancestor (the same
   closure argument the nodal/modal refusal rests on, below).
   ``*`` is for a product whose factors are **not** all
   axis-built. `[M]` 2026-09-07, by AST over ``orpheus/**/*.py``, that is
   the harmonic/moment family and nothing else — **four** production
   sites: the angular head :math:`\otimes` the axis-built cell group, in
   :meth:`HarmonicFrame.moment_space_on
   <orpheus.transport.frames.harmonic_frame.HarmonicFrame.moment_space_on>`
   and in its content-equal field-side twin
   :meth:`MomentField._space_for_mesh_and_L
   <orpheus.transport.fields._bases.MomentField._space_for_mesh_and_L>`,
   each of which then appends a
   :class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`
   factor for a widened angular space (that factor's inner product is
   Euclidean, so it contributes a ``None`` entry). Item **6.2c**
   axis-ifies the head; until then see
   :ref:`spaces-metric-propagation`, where dropping a factor's metric is
   measured as a value bug rather than a representation choice.

Canonical storage: one measure, one spelling, one identity
-----------------------------------------------------------

Because the measure is part of the identity, two spellings of the *same*
measure would be two identities of the same space — the exact twin the
axis layer exists to prevent. Three construction rulings close it
(2026-08-20), and each is enforced in
:meth:`Axis.__post_init__ <orpheus.numerics.axis.Axis>`:

#. **All-ones weights collapse to** ``None`` **at construction.** The
   counting measure has ONE spelling. Without this,
   ``weights=None`` and ``weights=np.ones(shape)`` would be the same
   measure with unequal identities. `[M]` the two axes compare equal and
   hash equal after canonicalization.
#. **Weights are canonicalized as** ``w + 0.0`` **and stored
   read-only.** :math:`-0.0` and :math:`+0.0` are one measure and must
   be one byte pattern — the identity key reads ``weights.tobytes()``,
   so an un-normalized :math:`-0.0` would mint a second name for one
   measure. The addition also forces a fresh allocation, so mutating the
   caller's array can never move an axis's hash after it has been used
   as a dictionary key.
#. **Non-finite weights are REFUSED.** A factor measure has finite
   weights.

There is deliberately **no non-negativity guard**. CS2's quadrature axes
legally carry signed weights (level-symmetric families with negative
weights are real, shipped objects), and the axis is the wrong layer to
outlaw them. `[M]` ``Axis("x", (2,), weights=[-1.0, 2.0], kind=NODAL)``
constructs.

Identity is structural, and it is per subclass
-----------------------------------------------

Equality and hashing read the structural content — never object
identity, and never a subset of the fields. Two consequences are
load-bearing and both are gated:

- **An** :class:`~orpheus.numerics.axis.EnergyAxis` **never equals a
  generic** :class:`~orpheus.numerics.axis.Axis` **carrying the same
  field tuple.** The identity of an axis is *what kind of generator
  produced this factor*, not a bag of fields. `[M]` the comparison is
  ``False``.
- **A synthetic axis never equals a** ``from_grid`` **axis of the same**
  ``ng``. Same index set, no partition data — a different axis. `[M]`
  ``EnergyAxis.synthetic(2) == EnergyAxis.from_grid(grid_2g)`` is
  ``False``.

.. _spaces-identity-bridge:

The derived name — once the identity bridge, now the readable label
-------------------------------------------------------------------

⛔ **The bridge reading was retired on 2026-09-07** by the identity flip
(structural ``__eq__``, CS4c step 6). The section keeps its label and its
history, because the bridge is why the chartered doctrine was already
true for axis-built spaces between CS1 (2026-08-20) and the flip, and
because the name derivation it describes still ships — what changed is
what that derivation is FOR.

:meth:`of_axes <orpheus.numerics.space.FunctionSpace.of_axes>` computes
the name deterministically and injectively from the axes' structural
content — a length-prefixed, type-tagged content digest, never Python's
``hash()``, so it is stable across processes.

**Until the flip that injectivity WAS the identity.** Space identity was
``(name, shape)`` for every space, so the only way an axis-built space
could carry *metric differences imply space differences* was to mint a
different NAME for every different axis tuple: different axes → different
names → different spaces, with no flag day. That is the bridge.

**Since the flip the axes are read directly.**
:class:`~orpheus.numerics.space.FunctionSpace`'s :meth:`__eq__` compares
the ``axes`` tuple when both sides carry one, so the doctrine holds
without passing through a string. The derived name survives as three
other things, each load-bearing: the readable LABEL a human reads in a
traceback; what keeps ``repr`` and the space-content guard messages
content-distinguishing; and the identity carrier of the axes-less
COMPOSITES that fold member names —
:meth:`FullFieldSpace.from_blocks
<orpheus.numerics.spaces.full_field_space.FullFieldSpace.from_blocks>`
folds each member's ``(name, shape)`` pair into its own
``full_field#<digest>``, so an injective member name is still what makes
the composite content-keyed.

The mechanism is visible in one construction. The homogeneous carrier's
quotient point and a genuine one-cell mesh with :math:`V_{\rm cell} = 2`
have the same shape :math:`(2, 1)` and the same readable prefix, and are
**unequal** spaces:

.. code-block:: text

   quotient point (volumes = [1.0])  ->  energy(2,)*spatial(1,)#<digest A>
   one-cell mesh  (volumes = [2.0])  ->  energy(2,)*spatial(1,)#<digest B>
   before the flip:  digest A != digest B  ->  UNEQUAL  (via the names)
   after  the flip:  axes differ in measure bytes  ->  UNEQUAL  (directly)

`[M]` 2026-09-07 on the landed carve, building both spaces as
``FunctionSpace.of_axes(EnergyAxis.synthetic(2), Axis("spatial", (1,),
weights=w, kind=BasisKind.NODAL))`` with ``w = [1.0]`` and ``w = [2.0]``
(the all-ones measure canonicalizes to ``None``, so the two axes differ
in their measure bytes):

.. list-table:: What the flip changed, on one pair of mints
   :header-rows: 1
   :widths: 46 18 18 18

   * - comparison
     - before
     - after
     - reads
   * - the two mints above (different measures)
     - ``False``
     - ``False``
     - unchanged — the doctrine held through the bridge, and now holds
       directly
   * - two independent mints of the SAME axes
     - ``True``
     - ``True``
     - unchanged; the hashes agree too
   * - an axis-built space vs a hand-named
       :class:`~orpheus.numerics.space.FunctionSpace` carrying its exact
       ``name`` and ``shape``
     - ``True``
     - ``False``
     - **flipped** — a label is no longer a way to spell somebody
       else's space
   * - ``A * B`` (both factors axis-built) vs
       ``FunctionSpace.of_axes(*A.axes, *B.axes)``
     - ``False``
     - ``True``
     - **flipped** — Q-T4 realized: an axis product is not a different
       *kind* of space, and the two mints derive DIFFERENT names
       (``…#a ⊗ …#b`` against one concatenated digest) while carrying
       the same axes

The last two rows are the whole content of the flip. The third says the
derived name stopped being a *credential* — before, anything that could
spell the string was that space. The fourth says the name stopped being
the *identity* — two spellings of one axis tuple were two spaces, and are
now one. `[R]` the "before" column is derivable rather than re-run: the
pre-flip body is exactly ``self.name == other.name and self.shape ==
other.shape`` (``git show 823f97dd:orpheus/numerics/space.py``), so it is
decided by the names printed above.

.. note::

   **What was overturned.** Until this campaign the
   :class:`~orpheus.numerics.space.FunctionSpace` docstring taught that
   *two copies of* :math:`\mathbb{R}^n` *are "the same" space regardless
   of which inner product is installed*. That is false in the only sense
   the operator algebra cares about: the metric defines ``.H``, so two
   spaces differing only in metric are spaces where the same symbol
   denotes **different operators**, and composing across them must be
   refused rather than silently accepted. The docstring now carries both
   halves — the chartered doctrine and the current nominal realization —
   because stating only the target would lie forward and stating only
   the present would lie backward.

.. _spaces-nodal-modal:

NODAL and MODAL: which factors have a coordinate cone
------------------------------------------------------

:doc:`/theory/foundations/field_algebra` establishes that flux lives in
the positive cone :math:`K \subset V` and that cone membership is an
**element predicate**. The space layer supplies the missing half: *on
which spaces is that predicate even meaningful?*

- A **NODAL** factor has components that are point or cell VALUES (an
  indicator-like basis). Pointwise nonnegativity of the coefficients
  **is** nonnegativity of the function, so the coordinate cone
  :math:`K = \{x \ge 0\}` is the physical positive cone. The
  discrete-ordinates axis, the spatial cell axis and the energy group
  axis are all nodal.
- A **MODAL** factor has components that are expansion COEFFICIENTS. A
  positive function may have negative coefficients, so a per-component
  sign test answers a question about the *basis*, not about the
  function. The spherical-harmonic moment axis is modal.

This dichotomy is the ray-effect / negative-source dichotomy seen from
the algebra side: positivity is native to the quadrature axis,
rotational equivariance is native to the harmonic axis, and no angular
basis has both.

:attr:`FunctionSpace.has_coordinate_cone
<orpheus.numerics.space.FunctionSpace.has_coordinate_cone>` is
**three-valued**, and the third value is the point:

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Value
     - When
     - What consumers do
   * - ``True``
     - axis-built, ALL factors ``NODAL``
     - Answer. The sign test is meaningful; the arithmetic is unchanged
       from the legacy path.
   * - ``False``
     - axis-built, ANY factor ``MODAL``
     - **Refuse**, with a typed error naming the space and the reason.
       Answering would manufacture violations out of a basis choice.
   * - ``None``
     - ``axes is None`` — legacy, not migrated
     - Pre-CS1 behavior, unchanged. The question cannot be answered
       structurally, and collapsing ``None`` into ``False`` would fire
       the refusal on every legacy space in the tree.

⛔ **This paragraph claimed** *"`[M]` the refusal has no production
witness yet, deliberately. The only axis mint in* ``orpheus/`` *today is*
``MaterialMesh.bulk_space``\ *, whose factors are both* ``NODAL``\ *…
The arm becomes production-reachable when CS2 mints the harmonic axis"*
**until 2026-09-07**, and both halves had gone false — not by item
6.2a, which touches no axis declaration, but by the CS4b axis migration
that preceded it and was never reconciled here.

`[M]` re-measured 2026-09-07, by AST over ``orpheus/**/*.py``:
:meth:`of_axes <orpheus.numerics.space.FunctionSpace.of_axes>` has
**seven** production call sites, not one. And the ``False`` row now has
a production occupant: :attr:`SNMesh.angular_trial_space
<orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space>` appends the
scheme-owned ``moment_axis`` — declared ``BasisKind.MODAL``, because a
DG cell moment is an expansion coefficient — to an axis-built base
whenever the bound scheme is multi-moment, and `[M]`
``LinearDiscontinuous.is_multi_moment`` is ``True`` while
``DiamondDifference``'s is ``False``. So an LD carrier's trial space is
an axis-built space with a MODAL factor. ⚠ Scope: that measures the
*occupant*, not the *firing* — whether any consumer asks
``has_coordinate_cone`` of that space is a separate census this page
does not carry. The named test witness stands either way
(``tests/numerics/test_field.py``, gates E1/E2 — the refusal and the
same values answered on an all-nodal space, the positive-and-negative
pair ``vv-principles`` anti-pattern #11 requires).

What survived intact is the CLOSURE ARGUMENT and its consequence for
the harmonic family. Since ``of_axes`` is the only ROOT producer of an
``axes`` record (``*`` and :meth:`dual
<orpheus.numerics.space.FunctionSpace.dual>` merely thread one through,
so both need an axis-built ancestor), and the angular head declares no
axes, every harmonic-moment space in the tree is still legacy
(``axes is None``) and therefore takes the ``None`` arm — `[M]`
2026-09-07 the moment product's metric is a
:class:`~orpheus.numerics.metric.FactoredMetric` while its ``axes`` is
``None``. Item **6.2a** did not move that: retiring the densifier
changed how a product carries its measure, not whether its factors
declare axes. Item **6.2c** does, by minting the harmonic axis (this
sentence said "CS2" until the 2026-09-07 ruling).


.. _spaces-axis-generator:

The generator slot: an axis is a forgetful map, and the mint is its section
---------------------------------------------------------------------------

An axis keeps the weights of the object that produced it and nothing
else about it. That is exactly the right amount of structure for a
*measure* — the pairing consumes weights, not nodes
(:eq:`spaces-axis-product`) — and it is one datum short for every
consumer that needs the geometry back. The discrete-ordinates axis is
the sharp case: its factor measure is the quadrature weights
:math:`w_n`, and a sweep needs the direction cosines
:math:`\hat\Omega_n` that the weights were attached to. Before campaign-1
phase CS5 the only way to supply them was to hand the consumer the
:class:`~orpheus.numerics.quadrature.directional.Quadrature`
*alongside* the space — which is the reach-past this campaign exists to
retire, because a consumer taking two arguments that must agree is a
consistency obligation the type system is not carrying.

This section is the doctrine's home: the forgetful-map framing, the
section law, the refuted alternative, the identity exclusion, the
``None`` inventory (:ref:`spaces-generator-none-inventory`), the seams
and the gates (:ref:`spaces-generator-gates`). The two mint accessors'
own API narrative lives one page over, at
:ref:`discrete-measures-quadrature-axis`.

**The forgetful map.** Write a generator :math:`g` for the discrete
measure it presents,

.. math::
   :label: spaces-axis-forgetful-map

   \mu_g \;=\; \sum_{i=1}^{n} w_i\,\delta_{x_i},
   \qquad
   \mathcal{F}_\ell(g) \;=\;
   \bigl(\,\ell,\ (n,),\ w,\ \texttt{NODAL}\,\bigr) ,

.. (vv-status rationale) Structural/representational identity: it STATES
   what ``DiscreteMeasure.axis`` constructs from a measure's atoms — the
   label, the rank-1 index shape ``(n_points,)``, the weight vector, and
   the NODAL kind implied by the generator's TYPE. Not a solver claim —
   no flux, no eigenvalue, no discretization error. The verifiable
   content is the CS5 foundation battery
   (``tests/numerics/test_axis_generator.py`` G3, the mint-fidelity
   roster over all five shipped ``Quadrature`` factories, comparing the
   minted axis's label/shape/kind/weight BYTES against the literal
   construction it replaced).
.. vv-status: spaces-axis-forgetful-map documented

where :math:`\ell` is the factor's label. :math:`\mathcal{F}_\ell` keeps
:math:`w` and drops the nodes :math:`x` — hence *forgetful*, and hence
the ``NODAL`` kind is not a parameter of the mint but a **consequence of
the generator's type**: a discrete measure's components are point values
with a coordinate cone (:ref:`spaces-nodal-modal`), so the
mis-declaration "a nodal basis carrying no nodes" is unspellable on this
path. It was spellable on the literal path, and CS5 retired that path at
**two** of the tree's three nodal mint sites — the angular factor of
:attr:`SNMesh.angular_bulk_space
<orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space>` and the
rank-1 spatial factor of :attr:`MaterialMesh.bulk_space
<orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>`. The
third, the homogeneous pose's counting point, keeps its literal because
it has no generator to mint through; what changed there is that the
``None`` became a documented reading rather than an unexamined default
(:ref:`spaces-generator-none-inventory`).

**Recoverable forgetting.** The repair is not to stop forgetting — the
axis genuinely should not carry a node array it never pairs against, and
:eq:`spaces-axis-product` is the reason. The repair is to make the
forgetting *recoverable*: the mint

.. math::

   m_\ell(g) \;=\; \mathcal{F}_\ell(g)\ \text{ with }\
   \texttt{generator} = g

records its own input, and the projection ``a.generator`` (written :math:`p`)
reads it back. The two compose to identities in both directions:

.. math::
   :label: spaces-axis-generator-section

   p\bigl(m_\ell(g)\bigr) \;=\; g \quad\text{for every generator } g,
   \qquad
   m_{\ell(a)}\bigl(p(a)\bigr) \;=\; a \quad\text{for every minted axis } a .

.. (vv-status rationale) Structural identity (the SECTION law): it
   STATES that the generator-minted axis is a two-sided inverse pair
   with the generator projection — the mint records its own input
   (``a.generator is g``, an identity check) and re-minting from the
   recorded generator reproduces the axis (``a.generator.axis(a.label)
   == a``, a structural-equality check). Not a solver claim — no flux,
   no eigenvalue, no discretization error. The verifiable content is the
   CS5 foundation battery: ``tests/numerics/test_axis_generator.py``
   G8 (``TestG8TheMintIsASectionOfTheForgetfulMap``, the angular roster
   plus the NODAL-measure leg) and the spatial leg beside G6a in
   ``tests/numerics/test_space_of_axes.py``.
.. vv-status: spaces-axis-generator-section documented

The right-hand law is the operational one and it is gated by name:
``a.generator.axis(a.label) == a``. Read it as *the mint is a section of
the forgetful map* — every generator-ful axis can be regenerated from
what it remembers, so nothing the axis dropped is unreachable from the
axis.

.. warning::

   **The law's domain is MINTED axes, and a hand-passed generator can
   lie.** ``generator`` is an ordinary keyword field with no
   cross-check against the rest of the axis — nothing refuses
   ``Axis("angular", (2,), weights=[1.0, 3.0], kind=NODAL,
   generator=Quadrature.gauss_legendre(4))``, an axis of shape
   :math:`(2,)` claiming to have come from a 4-ordinate rule. `[M]` it
   constructs, and :eq:`spaces-axis-generator-section` reads ``False``
   on it while reading ``True`` on the accessor-minted sibling. That
   asymmetry is the whole reason both the class docstring and this page
   say **prefer minting through the generator**: a generator-minted axis
   cannot forget its provenance and cannot misstate it, whereas a
   hand-passed one is an unchecked assertion. The section law is the
   predicate that detects the lie — which is why it is gated rather than
   assumed.

.. _spaces-generator-why-quadrature:

Why the QUADRATURE and not the bare measure (an opener's refutation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The design as chartered proposed typing the accessor
:class:`~orpheus.numerics.measure.DiscreteMeasure` — the measure is the
mathematical object, the quadrature is an SN-side convenience wrapper
over it, and the layering argument says reach for the lower one. The
phase's opening ground measurement **refuted** it, and the refutation is
worth carrying because the layering argument is otherwise correct.

The done-when named four things a consumer must be able to recover
through the space: ``mu_x``, ``eta``, ``mu_z``, ``level_indices``. A
bare measure can supply **three of the four** — as columns of its node
array rather than under those names — and cannot supply the fourth at
all. The reason is a field list:

.. list-table:: What each candidate generator can answer
   :header-rows: 1
   :widths: 22 30 48

   * - Name
     - On a ``DiscreteMeasure``?
     - Where it actually lives
   * - ``mu_x`` / ``mu_z``
     - the DATA is there
     - columns of ``measure.nodes`` (shape :math:`(N, d)`), read by
       :meth:`Quadrature.axis_cosines
       <orpheus.numerics.quadrature.directional.Quadrature.axis_cosines>`.
       `[M]` ``q.mu_x`` is bit-equal to ``measure.nodes[:, 0]`` and
       ``q.mu_z`` to ``measure.nodes[:, 2]``.
   * - ``eta``
     - the DATA is there
     - the same column 0 under its cylindrical-frame name
       (:math:`\eta = \hat\Omega\cdot\hat r`) — an alias, not a second
       array. `[M]` ``q.eta`` is bit-equal to ``q.mu_x``.
   * - ``level_indices``
     - ⛔ **no**
     - the :class:`~orpheus.numerics.quadrature.rules_sphere.LevelStructure`
       side-channel, carried by the
       ``Quadrature``. `[M]` a ``DiscreteMeasure`` has exactly five
       fields — ``nodes``, ``weights``, ``support``,
       ``invariance_group``, ``exactness`` — and
       ``hasattr(measure, "level_indices")`` is ``False``.

So the level fibration is not a *name* the measure lacks, it is
**structure the measure does not contain**: the polar-level partition of
the ordinate set is a datum the curvilinear :math:`\alpha`-recursion
needs and the atom list cannot express. Typing the accessor at the
measure would therefore have shipped an accessor that answers the easy
three-quarters of its own contract, with the one name that motivated the
design still requiring the reach-past.

The ruling is consequently: **the angular axis's generator is the
Quadrature**, and
:meth:`Quadrature.axis <orpheus.numerics.quadrature.directional.Quadrature.axis>`
delegates the *structural* mint down to
:meth:`DiscreteMeasure.axis <orpheus.numerics.measure.DiscreteMeasure.axis>`
— one home for the shape/weights/kind logic, no twin — and upgrades only
the provenance, through ``dataclasses.replace``. The upgrade re-runs the
axis's ``__post_init__``, so it cannot bypass a construction invariant:
the all-ones collapse, the :math:`-0.0` normalization and the read-only
flag all survive the replace, which is asserted rather than assumed.

.. note::

   **The generalisation, and it is the reusable half.** The refuted
   proposal is an instance of a pattern worth naming: *choosing a
   provenance type by LAYER rather than by CONTRACT*. The layer argument
   ("the measure is lower, so record the measure") is a statement about
   the dependency graph; the contract argument ("record the object that
   answers everything the axis forgot") is a statement about what the
   consumer needs. When the lower object is a **projection** of the
   higher one — and a measure is exactly the quadrature minus its
   side-channels — the layer argument silently narrows the contract.
   The decidable check is the one run here: enumerate the names the
   consumer must recover and ask which candidate answers **all** of
   them.

.. _spaces-generator-identity-exclusion:

Provenance is never identity — and here the exclusion is mandatory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``generator`` is absent from :meth:`Axis._identity_key
<orpheus.numerics.axis.Axis>`, so equality, hash and the ``of_axes``
name digest all ignore it. The *doctrinal* reason is one sentence — two
axes with identical structural content are the same axis whatever
instance produced them, exactly as two identical measures are the same
measure — but the doctrinal reason is not the load-bearing one. **The
exclusion is structurally mandatory**: an identity key containing a
generator does not merely change the answer, it makes the operation
raise.

.. list-table:: Why an inclusion is not an option (`[M]` 2026-08-29, this tree)
   :header-rows: 1
   :widths: 34 30 36

   * - Reading
     - Result
     - Mechanism
   * - ``hash(Quadrature.gauss_legendre(4))``
     - ``TypeError: unhashable type: 'Quadrature'``
     - a ``dataclass`` with ``eq=True`` and ``frozen=False`` gets
       ``__hash__ = None``. `[M]`
       ``Quadrature.__dataclass_params__.frozen`` is ``False``.
   * - ``q1.measure == q2.measure``
     - ``ValueError: The truth value of an array with more than one
       element is ambiguous``
     - ``DiscreteMeasure`` is ``frozen=True, eq=True`` over ndarray
       fields, so the generated ``__eq__`` compares arrays and the
       tuple comparison collapses them to a ``bool``.
   * - the simulated inclusion — a subclass appending
       ``self.generator`` to ``_identity_key`` — then ``a1 == a2``
     - **RAISES** ``ValueError`` (same message)
     - the key is compared as a tuple, so the generator's own
       un-comparability propagates straight to ``Axis.__eq__``.
   * - the same subclass, then ``hash(a1)``
     - **RAISES** ``TypeError`` (same message)
     - ``Axis.__hash__`` hashes the key tuple, so the generator's
       unhashability propagates straight through.

Reproduce it by subclassing rather than by editing the shipped key —
the simulation is four lines and needs no mutation of production code:

.. code-block:: python

   from orpheus.numerics.axis import Axis, BasisKind
   from orpheus.numerics.quadrature.directional import Quadrature

   class _WithGeneratorInKey(Axis):
       def _identity_key(self):
           return (*super()._identity_key(), self.generator)

   q1, q2 = Quadrature.gauss_legendre(4), Quadrature.gauss_legendre(4)
   a1 = _WithGeneratorInKey("angular", (q1.N,), weights=q1.weights,
                            kind=BasisKind.NODAL, generator=q1)
   a2 = _WithGeneratorInKey("angular", (q2.N,), weights=q2.weights,
                            kind=BasisKind.NODAL, generator=q2)
   a1 == a2      # ValueError: truth value of an array ... is ambiguous
   hash(a1)      # TypeError: unhashable type: 'Quadrature'

Three consequences follow, and each is separately load-bearing:

#. **Issue #403's content-equal-measure hazard never reaches axis
   identity.** Two carriers built from equal inputs hold *distinct*
   rule instances; because the generator is out of the key, their axes
   are equal and their spaces are equal. `[M]` on two twin 1-D
   :math:`S_N` carriers built from equal edge arrays with
   ``gauss_legendre(4)``: ``a.quad is not b.quad`` is ``True`` while
   ``a.angular_bulk_space == b.angular_bulk_space`` and the hashes
   agree; a third carrier differing only in a cell edge is
   ``False``. That is the same row the field algebra's fiber table
   reports (:doc:`/theory/foundations/field_algebra`), and it is
   unmoved by CS5 **because** of the exclusion.
#. **The derived space NAME cannot drift — and since the identity flip
   the exclusion carries space identity directly.**
   ``_structural_bytes`` iterates ``_identity_key``, so any field
   admitted to the key enters every derived space name
   (:ref:`spaces-identity-bridge`). Until the identity flip (structural
   ``__eq__``, CS4c step 6, 2026-09-07) that name WAS the identity, so an
   inclusion would not merely perturb a digest — it would split one space
   into as many spaces as there are rule instances in the process. Since
   the flip an axis-built space compares its ``axes`` tuple directly, and
   ``Axis`` equality IS ``_identity_key``: the exclusion of ``generator``
   is now what keeps provenance out of SPACE identity, with no digest in
   between. The conclusion is unchanged, and the argument for it got
   one link shorter — which is why consequence 1 above is still measured
   the same way.
#. **The three CS5 consumer re-points are bit-identical by
   construction.** `[M]`
   :attr:`SNMesh.angular_bulk_space
   <orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space>`'s
   generator-minted angular axis compares equal to the literal
   ``Axis("angular", (quad.N,), weights=quad.weights, kind=NODAL)`` it
   replaced, and mints the same digest — so no snapshot, no cached
   space, and no equality-keyed consumer could observe the change.

.. _spaces-generator-not-a-reverse-accessor:

What this is NOT: the refused axis-to-measure accessor still stands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CS4b phase S6.0b considered and **refused** giving
:class:`~orpheus.numerics.axis.Axis` a public accessor that *produces* a
:class:`~orpheus.numerics.measure.DiscreteMeasure`
(:ref:`spaces-collapse-pair-refuted`). That refusal is not overturned,
and the distinction is the whole design:

- the refused thing points **axis → measure** and would have had to
  **manufacture** its output. A pre-CS5 axis had dropped the nodes, so
  the only node set it could synthesise is the index set — which is
  precisely what the collapse-pair mint builds *locally*
  (``nodes = arange(n)``, ``support = f"index({label})"``,
  :ref:`spaces-collapse-pair-frame`). Exposing that as an accessor
  would have published a *synthetic* measure under a name readers
  would take for the generating one.
- CS5 points **generator → axis** and manufactures nothing. It records
  the real object at the one moment it is in scope — the mint — so the
  recovered data is the generator's own, not a reconstruction.

The consequence for the collapse pair is: nothing changed. The rank-one
indicator frame is deliberately built over the **index set**, not over
the generator's physical nodes, because a marginal over an axis is a sum
over its indices; that site still builds its own measure and does not
consult ``axis.generator``. Both facts are worth stating together,
because "the axis can now reach a measure" invites exactly the wrong
inference at that call site.

.. _spaces-generator-none-inventory:

The honest-``None`` inventory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``generator=None`` is a reading, not a gap. Measured on a shipped 1-D
slab carrier (edges ``0|1|3|6``, ``gauss_legendre(4)``, 2 groups):

.. list-table:: `[M]` 2026-08-29 — every axis of every shipped space on that carrier
   :header-rows: 1
   :widths: 26 18 18 38

   * - Space
     - Axis
     - ``generator``
     - Why
   * - :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space`
     - ``energy``
     - ``None``
     - an :class:`~orpheus.numerics.axis.EnergyAxis` is the counting
       measure as a **theorem**
       (:ref:`spaces-counting-measure-theorem`), so there is no measure
       object to name; the group structure it does carry is its own
       ``from_grid`` data.
   * - :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space`
     - ``spatial``
     - ``DiscreteMeasure``
     - the carrier's own
       :attr:`volume_measure
       <orpheus.transport.mesh.material_mesh.MaterialMesh.volume_measure>`
       — its one documented data path. `[M]` nodes are the cell centres
       ``[0.5, 2.0, 4.5]`` and weights the volumes ``[1.0, 2.0, 3.0]``,
       both hand-derivable from the edge list.
   * - :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space`
     - ``angular``
     - ``Quadrature``
     - the rule itself, so ``mu_x`` / ``eta`` / ``mu_z`` /
       ``level_indices`` answer through the space.
   * - :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space`
     - all three
     - as above
     - byte-identical to ``angular_bulk_space`` for a slopeless
       closure, so it inherits the same axes.
   * - the homogeneous pose
     - ``spatial``
     - ``None``
     - the infinite-medium pose has **no mesh**, so no spatial measure
       object exists. The one-point counting axis IS the normalized
       per-unit-volume convention (:ref:`spaces-quotient-family`);
       ``None`` is the record, not an omission.

The reading to carry: a ``None`` generator says *no generator object
exists for this factor*, never *nobody bothered*. Two of the five rows
above read ``None``, and each for a stated structural reason rather
than for lack of attention — which is what makes the reading
falsifiable: if a factor that HAS a generator object ever reads
``None``, that is a defect, and this table is where to check.

.. warning::

   **The generator is a LIVE REFERENCE, not a snapshot — the axis's own
   weights copy is the authority.** `[M]`
   ``Quadrature.__dataclass_params__.frozen`` is ``False``, so a
   quadrature is mutable and ``axis.generator``'s arrays can in
   principle be moved after the axis was minted. The axis's own
   ``weights`` cannot: ``__post_init__`` stores ``w + 0.0`` with
   ``setflags(write=False)``, so `[M]` ``axis.weights.flags.writeable``
   is ``False`` and ``axis.weights is quad.weights`` is ``False`` — a
   fresh, read-only, defensively-copied array. Consequently **the
   factor measure of record is** ``axis.weights``, and a consumer that
   reads ``axis.generator.weights`` instead is reading a mutable alias
   of what the axis's identity was computed from. Read the generator
   for what the axis FORGOT (nodes, cosines, level structure); read the
   axis for what it KEPT.

.. _spaces-generator-seams:

The seams: rank-:math:`d`, MODAL, and the solve-time consumer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three arms of the design were named here when CS5 landed, each deliberately
not built for a stated reason rather than for lack of time. **Two remain**:
the third — a solve-time consumer that reads a generator — was discharged
on 2026-08-29 by the streaming campaign's P4-remainder, and its entry below
is kept in place, past-tensed, because the *reason a gate was withheld* is
the transferable content.

**Rank-d spatial axes are generator-less BY CONTRACT (CS2).**
A :class:`~orpheus.numerics.measure.DiscreteMeasure` is a **flat atom
list** — nodes :math:`(N, d)`, weights :math:`(N,)` — so
:meth:`DiscreteMeasure.axis
<orpheus.numerics.measure.DiscreteMeasure.axis>` mints at rank 1, by
construction. A rank-:math:`d` spatial axis has shape
:math:`(n_x, n_y, n_z)`, and there is no rank-:math:`d` measure-to-axis
pairing yet. Minting flat anyway is not merely inelegant, it is
observable: `[M]` on a :math:`2\times3\times2` Cartesian carrier the
axis-built name is ``spatial(2, 3, 2)#4833f3fd3f12352b`` while the flat
mint gives ``spatial(12,)#c9919057f3e8687d`` — a different space, for
every :math:`d \ge 2` carrier in the tree. So
:attr:`MaterialMesh.bulk_space
<orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>`
branches on rank: the rank-1 arm mints through ``volume_measure``, the
rank-:math:`d` arm stays literal and generator-less. ⛔ The day CS2
mints the rank-:math:`d` pairing, that row must be inverted
**deliberately** — it is a seam's witness, not a permanent truth, and
its gate says so in its own name
(``test_the_rank_d_spatial_axis_is_generator_less_BY_CONTRACT``).

**No MODAL axis has a generator yet, and no ``Basis`` can mint one.**
The :attr:`~orpheus.numerics.axis.Axis.generator` annotation admits a
:class:`~orpheus.numerics.basis.base.Basis` — that is the chartered
MODAL arm — but `[M]` ``hasattr(Basis, "axis")`` is ``False`` and no
subclass defines it, so the arm is **declared and unbuilt**. Two
consequences worth stating: the section law
:eq:`spaces-axis-generator-section` currently ranges over
measure-generated and quadrature-generated axes only, and a
hand-constructed ``generator=<some Basis>`` axis would fail it with an
``AttributeError`` rather than a ``False``. The arm becomes real when
CS2 mints the harmonic axis — the same phase that gives the ``False``
arm of ``has_coordinate_cone`` its first production witness
(:ref:`spaces-nodal-modal`).

**No solve-time consumer read a generator, and that is why two gates
were withheld** — ✅ **discharged 2026-08-29.** CS5 landed the machinery
and the nodal mint sites; it did **not** re-point the streaming producer,
which then took a space and a quadrature as separate arguments. Two
specified gates — a refusal when a consumer is handed a generator-less
axis, and the route keystone proving the consumer reads *through* the
space — therefore did not land with it, because a gate that ships before
the case it catches exists is green and unfalsifiable by construction
(``plan-authoring`` §6c).

The streaming campaign's **P4-remainder** landed the re-point and both
gates with it, in the same change (``ad04e236``):
:class:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator` gained
an :attr:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.angular_axis`
field, the angular-closure family's construction contract widened to
``cls(angular, pairing, angular_axis)``, and every re-pointed read narrows
through one accessor,
:meth:`Axis.generator_as <orpheus.numerics.axis.Axis.generator_as>`. The
reason that accessor is load-bearing rather than decorative is the same
observation that chose the generator's type in the first place
(:ref:`spaces-generator-why-quadrature`): the bare union cannot answer the
consumers' reads, because a
:class:`~orpheus.numerics.measure.DiscreteMeasure` has no ``mu_x`` and no
``level_indices``. So the narrow is a **type claim**, not a ``None``
check — parse, don't validate — and the refusal it raises names both the
axis and the asking consumer. What makes proving that re-point *hard* —
and what the withheld keystone had to be — is
:ref:`spaces-generator-route-gate`.

.. _spaces-generator-route-gate:

Proving a re-point when every value gate is ``X == X``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A re-point that changes *where a consumer gets a datum* while leaving the
datum itself alone is the case in which ordinary value gates carry **zero
information**. Before the P4-remainder the streaming producer read the
quadrature off a courier field on the angular factor; after it, it reads
the generator off the bound axis — and those are the **same object**:
`[M]` on a shipped 1-D operator ``op.angular_axis.generator is quad``
for the very ``quad`` its factory was handed, which is the object the
retired courier held. So every before/after comparison of a
produced value is literally ``X == X``: green under a correct re-point,
and green under a re-point that silently kept reading the old path. This
is ``vv-principles`` #19 at the level of a data *route* rather than a
metric — the reading you naturally take is the one that cannot
discriminate.

The instrument that *can* discriminate is a **decoy generator**: an axis
whose generator carries different angular data while the axis itself is
identity-equal to the true one. Identity-equality is what makes it a
route probe — the axis's own record is its weights
(:ref:`spaces-generator-none-inventory`), so a decoy that preserves
weights is **invisible to the space**, and any movement in the produced
value can only have come from a read that went *through* the axis to the
generator. `[M]` on the keystone's own fixtures ``quad.axis() ==
decoy.axis()`` and ``hash`` agree, so the decoy cannot be caught by
identity.

⚠ **The decoy catalogue is not free — production's own admission guards
refuse most of it, and the refusals come from two different tiers.** This
is the finding worth carrying, because the obvious decoys (roll the
nodes, negate them, reverse them) all fail, and they fail for reasons that
are easy to misattribute. Measured on the keystone's own curvilinear
configurations — sphere ``gauss_legendre(4)``, cylinder
``folded_product(4, 6)``; the α-dome tier admits everything on the
Cartesian chart, whose dome is the neutral zero:

.. list-table:: `[M]` 2026-08-29 — which decoys production admits, and where the others die
   :header-rows: 1
   :widths: 22 16 31 31

   * - Decoy
     - Axis-blind?
     - α-dome tier
       (:func:`~orpheus.sn.angular.redistribution.angular_redistribution`)
     - Closure-mint tier
       (:func:`~orpheus.sn.angular.closure.morel_montry_tau_per_level`)
   * - nodes :math:`\times\,0.9`
     - yes
     - **admitted** — scaling preserves the antisymmetry
       :math:`\sum_n w_n \mu_n = 0` (`[M]` :math:`\pm 5.6\!\times\!10^{-17}`
       before and after), so the dome still closes at
       :math:`\alpha_{M+1/2} = 0`
     - **admitted**
   * - nodes rolled by 1
     - yes
     - **refused** — *"the alpha dome does not close"*. `[M]` the roll
       breaks the antisymmetry outright, :math:`\sum_n w_n \mu_n =
       -0.366` (sphere) / :math:`+0.239` (cylinder), and the guard
       reports exactly that residue as :math:`\alpha_{M+1/2}`
     - refused (never reached)
   * - nodes negated
     - yes
     - admitted — a sign flip preserves the antisymmetry
     - **refused** — P3, :math:`\tau \notin [0,1]`
   * - nodes reversed
     - yes
     - admitted — the shipped weight vectors are symmetric, so a
       reversal preserves the antisymmetry too
     - **refused** — P3 / the partition producer
   * - weights :math:`\times\,0.9`
     - **no**
     - admitted
     - **refused at this order** (:math:`\tau = 1.195`); admitted only
       at :math:`N = 2` — see the note below

⭐ **The attribution matters.** It is tempting to write "the α-dome guard
refuses rolled, negated and reversed nodes"; `[M]` the dome refuses only
the **roll**. Negation and reversal sail through the dome — its admission
contract is an antisymmetry, and both operations preserve it — and die one
tier later at the Morel–Montry closure's **P3** membership guard
(:math:`\tau_m` must lie in its own angular cell). A decoy catalogue is
therefore a statement about *two* contracts, and citing only the first
would send the next session looking for the refusal in the wrong file.

That leaves the **node-scale decoy as the only member admitted at both
tiers on a curvilinear chart**, which is why it is the keystone's decoy
rather than a stylistic choice. Its discriminating power is measurable and
structural: `[M]` on a 4-cell mesh it moves **4 of 4** packets on the slab,
**4 of 4** on the sphere and **8 of 12** on the cylinder. The cylinder's
shortfall is not a weakness of the probe but an identity of the rule — the
four packets that do not move are the :math:`\eta = 0` azimuthal member of
each of its four :math:`\mu`-levels, and :math:`0.9 \times 0 = 0`. A floor
with a reason is worth more than a floor with a number: this one cannot
drift without the rule changing.

.. note:: **Two decoys, because one decoy moves two reads together.**
   The scale decoy moves ``mu_x`` and ``level_indices`` *at once*, so on
   its own it cannot tell a complete re-point from one that re-pointed the
   cosine read and left the index read on the old path
   (``vv-principles`` #17's per-arm discipline). The isolator is a
   **level-roll** decoy — same measure, level list rolled by one, so
   ``mu_x`` is untouched — and `[M]` it moves **4 of 12** packets on
   ``folded_product(4, 6)``. The floor is 4 rather than 8 because the
   per-level radial cosines are a **palindrome**
   (:math:`0.440,\,0.814,\,0.814,\,0.440`), so a roll of one fixes half
   the levels; the :math:`\eta = 0` members never move either way. On the
   closure side the third probe is a **weight** decoy — the only read
   whose datum is :math:`w_n` rather than a cosine — and it is deliberately
   *not* axis-blind: it proves the :math:`\Delta A/w` mint goes through
   the axis, at the cost of being visible to space identity, so it is a
   separate row rather than a leg of the first. `[M]` on the sphere,
   weights :math:`\times\,0.9` is **admitted** at the closure gate's
   ``gauss_legendre(2)`` fixture and **refused** at
   ``gauss_legendre(4)``, ``(6)`` and ``(8)`` — with
   :math:`\tau = 1.195`, :math:`1.047`, :math:`1.059` respectively —
   because shrinking every weight shrinks the cumulative partition and
   pushes the outermost :math:`\tau` past 1. A decoy's admissibility is
   therefore a property of the *order* as well as of the operation, and
   a decoy catalogue quoted without its fixture is not re-runnable.

.. _spaces-generator-protocol:

The sibling repair: a contract that declares what its consumer reads
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same ``level_structure`` side-channel that decided the generator's
type exposed an under-declared contract one package over. The
``AngularMeasure`` :class:`~typing.Protocol` in
:mod:`orpheus.sn.angular.redistribution` — the structural contract the
:math:`\alpha`-dome recursion consumes — declared six members while the
cylindrical streaming factory admission-probed a seventh, spelled
``getattr(angular_measure, "level_structure", None)``.

That spelling is doubly invisible. It is a **string-form attribute
read**, so no symbol grep over the Protocol's members can see it; and it
is a **defaulted** ``getattr`` inside a guard's condition, so the day
the attribute is renamed the guard does not raise — it silently takes
the ``None`` branch and the admission check evaporates
(``vv-principles`` #28, the temporal twin). The repair is to declare the
member: the contract now states ``level_structure: LevelStructure |
None``, and ``None`` is the honest reading for a slab or 2-D rule rather
than an absent attribute. `[M]` ``Quadrature.level_symmetric(4)``
carries a
:class:`~orpheus.numerics.quadrature.rules_sphere.LevelStructure`;
``Quadrature.gauss_legendre(4)``
reads ``None`` while still answering ``level_indices`` with a single
degenerate level.

The tolerant ``getattr`` at the probe deliberately survived CS5 — a
structural conformer could predate the declaration — with the hardening
scheduled for the re-point that unblocks the two withheld gates above.
✅ **Both landed on 2026-08-29** (``1fb70c15``): the probe now reads
``angular_measure.level_structure`` directly, and the change is `[M]`
**unobservable** — no shipped or test carrier is in the absent-attribute
state the defaulted ``getattr`` distinguished, so the refusal's own
witness (a slab quadrature rejected by *"level structure"*) is unchanged
either way. That is the honest shape of this repair: the value of a
direct read is not a behaviour delta, it is that the guard can no longer
evaporate silently the day the member is renamed.

The same re-point widened the contract once more, for the same reason
and by the same test. The three streaming factories **mint the axis** from
the measure they are handed, so ``axis()`` is now something a consumer of
this contract calls — and a Protocol that omitted it would be
under-declaring exactly as before. The contract therefore declares
``axis(label="angular") -> Axis`` as well, with
:meth:`Quadrature.axis <orpheus.numerics.quadrature.directional.Quadrature.axis>`
as its concrete implementer. ⭐ The label **defaults at the generator**,
which is a single-source move rather than an ergonomic one: a directional
quadrature *is* the angular generator, so its axis's role is intrinsic,
the literal ``"angular"`` disappears from all three mint sites, and a
label-twin across mint sites becomes **unspellable** rather than merely
unlikely.

.. _spaces-generator-gates:

Verification — the gates, by name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cite these rather than copying their numbers; each re-measures itself.
`[M]` before they landed, a mutation battery over the 184-test anchor
set around the touched sites found **zero** genuine catchers for any
property on this page — the machinery was landing unwitnessed, which is
why the gate module is part of the same change.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Claim
     - Gate
     - What a mutation does to it
   * - provenance is not identity
     - ``tests/numerics/test_axis_generator.py``
       ``::TestG1GeneratorIsProvenanceNotIdentity``
     - putting ``generator`` in ``_identity_key`` makes ``==`` and
       ``hash`` RAISE — the sub-case ``G1c`` pins the two properties
       (unhashable / un-``==``-able) the exclusion rests on, so a
       future "tidy the field into the key" is refuted by a red rather
       than discovered by a traceback.
   * - the digest is blind to provenance
     - ``tests/numerics/test_space_of_axes.py``
       ``::test_of_axes_name_is_BLIND_to_the_generator``
     - the same inclusion moves the derived space name; the gate is
       stated at the tier where the damage would occur
       (:ref:`spaces-identity-bridge`).
   * - the mint reproduces the literal
     - ``tests/numerics/test_axis_generator.py``
       ``::TestG3MintFidelity``
     - minting ``MODAL``, or dropping the ``replace`` upgrade so the
       generator stays the bare measure, both red. Honest scope: the
       two sides are not independent on the weight VALUES (both read
       ``measure.weights``); what this pins is the THREADING and the
       surviving canonicalization.
   * - the four names answer through the space
     - ``tests/numerics/test_axis_generator.py``
       ``::TestG4TheFourNamesAnswerThroughTheSpace``
     - parametrized over **all five** ``Quadrature`` classmethod
       factories — ``gauss_legendre``, ``level_symmetric``, ``product``,
       ``folded_product``, ``lebedev`` — probed whole rather than
       laddered (``vv-principles`` #31's finite-roster corollary).
       ⭐ The roster shipped with **four** and the fifth was added the
       same day (``cb3cd15b``), which is the corollary demonstrating
       itself: the absent member was ``folded_product``, the
       :math:`\sigma_y`-folded cylindrical *carrying* rule the
       curvilinear MMS case builders default to and the one with the
       richest ``level_indices`` structure — i.e. the omission sat on
       exactly the axis the roster exists to gate, while the roster's
       own prose called itself exhaustive. A finite shipped family is
       enumerable (``vars(cls)`` + ``isinstance(v, classmethod)``); a
       roster that names its members by hand must be checked against
       that enumeration, not against its own adjective. Carries the
       refutation leg: a bare measure must NOT answer
       ``level_indices``, and if one starts to, this design must be
       re-ruled.
   * - the mint is a section
     - ``tests/numerics/test_axis_generator.py``
       ``::TestG8TheMintIsASectionOfTheForgetfulMap``
     - minting at a shape other than ``(generator.n_points,)``, or
       hand-passing a generator that did not produce the axis, reds it.
   * - the spatial chain, anchored independently
     - ``tests/numerics/test_space_of_axes.py``
       ``::test_the_spatial_axis_is_minted_through_the_carriers_own_measure``
     - its volumes and cell centres are HAND-DERIVED from the edge
       list, not read back from the mesh — the one structurally
       independent pin in the mesh → measure → axis chain.
   * - the rank-:math:`d` seam
     - ``tests/numerics/test_space_of_axes.py``
       ``::test_the_rank_d_spatial_axis_is_generator_less_BY_CONTRACT``
     - the contract row; inverting it is a deliberate CS2 act.
   * - the widened Protocol
     - ``tests/sn/angular/test_redistribution.py``
       ``::TestG9TheProtocolDeclaresWhatItsConsumersRead``
     - removing the ``level_structure`` declaration reds the first
       assertion. `[M]` nothing else in the tree observes the
       contract, so this row is its only witness.

The **P4-remainder** added the two gates CS5 withheld, plus the rows that
make them discriminate. They are listed separately because their evidence
class is different: CS5's rows are *value* gates on a mint, these are
*route* gates on a read (:ref:`spaces-generator-route-gate`), so a green
reading of one says nothing about the other.

.. list-table:: The re-point's own gates (P4-remainder, ``ad04e236``)
   :header-rows: 1
   :widths: 24 30 46

   * - Claim
     - Gate
     - What a mutation does to it
   * - the narrow is the ONE refusal home
     - ``tests/numerics/test_axis_generator.py``
       ``::TestG5GeneratorAsIsTheOneRefusalHome``
     - the Axis-tier rows: positive first (the narrow returns the
       generator itself), then generator-less and **wrong-kind**
       refusals — a measure-minted axis narrowed to ``Quadrature``
       refuses, which is what makes it a type claim rather than a
       ``None`` check. Two further rows are structural: the refusal's
       message fragments are **disjoint** from the neighbouring
       space-lookup refusal (so a pin on either can never match the
       other's raise), and an **AST** row asserts the accessor's
       production call sites are exactly the declared consumers — it
       reds the moment the refusal is wired onto a path a shipped
       generator-less axis travels (the homogeneous pose, the energy
       family).
   * - the refusal names both parties
     - ``tests/sn/mesh/test_reduced_operator.py``
       ``::…::test_a_generator_less_axis_refuses_naming_streaming_terms``;
       ``tests/sn/sweep/curvilinear/test_angular_closure.py``
       ``::…::test_G5_a_generator_less_axis_refuses_naming_the_closure``
     - three-fragment match per consumer — the axis label, the
       consumer's own name, and the ``"minted through"`` remedy. A
       generic message keeps a *wrong* reason true, so each fragment is
       pinned separately rather than as one regex.
   * - the courier is dead, structurally
     - ``tests/sn/mesh/test_reduced_operator.py``
       ``::…::test_the_courier_is_dead_by_field_set``
     - ``dataclasses.fields`` equality — never ``hasattr``, which a
       defaulted field or a ``getattr`` fallback would still answer —
       so a **re-addition** reds too. This is the row that keeps a
       partial re-point unspellable: with no courier on the angular
       factor, a consumer that still wants the quadrature has exactly
       one place to get it.
   * - the two mints agree
     - ``tests/sn/mesh/test_reduced_operator.py``
       ``::…::test_the_two_mints_agree_on_the_1d_arm`` /
       ``…_on_the_d2_cartesian_arm``
     - the only gate that reds on a wrong **label** at either mint site,
       and therefore the witness for the defaulted label: the
       :math:`d \ge 2` Cartesian arm has no reduced operator, so the hub
       mints the closure's axis itself — a second mint site that was one
       typo away from a label twin until the default killed the
       spelling. ``==``, never ``is`` (the mint is fresh per call).
   * - **KEYSTONE** — the packet reads through the axis
     - ``tests/sn/mesh/test_reduced_operator.py``
       ``::…::test_K1_the_packet_reads_THROUGH_the_axis`` (per chart)
     - four legs, because a route gate's own precondition must be
       gated too: an anti-dud **control** (the axis-built operator
       reproduces the factory-built one exactly), the **route** (the
       node-scale decoy moves :math:`|\eta|`), the decoy's
       **invisibility** to space identity (``==`` and ``hash`` agree —
       ``vv-principles`` #19), and the documentation leg that the
       angular factor was *not* reached. `[M]` 4 / 4 slab, 4 / 4
       sphere, 8 / 12 cylinder packets moved.
   * - the index read is a SEPARATE route
     - ``tests/sn/mesh/test_reduced_operator.py``
       ``::…::test_K2_the_cylinder_index_read_is_a_separate_route``
     - the level-roll decoy leaves ``mu_x`` untouched, so it isolates
       the ``level_indices`` read from the cosine read — without it a
       half-re-pointed producer is indistinguishable from a complete
       one (``vv-principles`` #17's per-arm discipline). `[M]` 4 / 12.
   * - the closure mint reads through the axis
     - ``tests/sn/sweep/curvilinear/test_angular_closure.py``
       ``::TestP4RemTheClosureMintReadsThroughTheAxis``
     - three decoys for three reads: nodes (:math:`\tau` and
       :math:`\mu_x` move), weights (:math:`\Delta A/w` moves — its own
       row, because the node decoy preserves weights by construction),
       and a **different-**\ :math:`N` axis for the identity member,
       whose only mint read is the ordinate count and for which a
       same-\ :math:`N` decoy is a structural non-catcher.

.. _spaces-counting-measure-theorem:

The counting-measure theorem on the energy axis
===============================================

The energy factor is the one axis whose measure is not a modelling
choice. It is forced — by the multigroup convention itself — to be the
counting measure, and *that* is why the energy metric is the identity.
This section derives it, states what the derivation buys at the space
layer, and says where the claim's single source of truth lives.

.. important::

   **Single source of truth.** The counting-measure claim
   (:math:`w_g = 1`, not :math:`w_g = \Delta u_g`) is stated, derived
   from Hébert's continuous formulation, and gated on
   :doc:`/theory/foundations/frame` — see
   :eq:`energy-condensation-counting-measure` inside
   :ref:`sn-energy-condensation`, whose verifiable content is the
   rate-preservation gate (a :math:`\Delta u_g` weight breaks it). This
   page does **not** restate that equation. What it owns is the claim's
   *space-layer consequence* — the energy metric is
   :math:`G_E = I`, hence :math:`V \cong V^*` isometrically along energy
   and the adjoint there is the plain transpose — and the fact that the
   consequence is now enforced at construction. Edited there, consumed
   here.

Covariant and contravariant: why no width appears
--------------------------------------------------

Write the continuous pairing that every reaction rate is:

.. math::

   r \;=\; \int_0^\infty \sigma(E)\,\varphi(E)\,\mathrm{d}E .

Discretizing splits into two *different* kinds of object, and the whole
theorem is that ORPHEUS discretizes each one in its own natural
variance:

- **The flux components are group INTEGRALS** — covariant, extensive
  quantities:

  .. math::

     \varphi_g \;=\; \int_{E_{g+1}}^{E_g} \varphi(E)\,\mathrm{d}E .

  The bin width is *inside* :math:`\varphi_g`; the stored number is
  "eV-free" and is a member of :math:`V`.

- **The cross sections are flux-weighted group AVERAGES** —
  contravariant, intensive quantities:

  .. math::

     \sigma_g \;=\;
     \frac{\displaystyle\int_{E_{g+1}}^{E_g} \sigma(E)\,\varphi(E)\,
     \mathrm{d}E}
          {\displaystyle\int_{E_{g+1}}^{E_g} \varphi(E)\,\mathrm{d}E} .

  This is a co-vector: it is a functional *on* fluxes, and it lives in
  :math:`V^*`.

Substituting the second definition into a group-by-group sum and
collapsing the denominator against :math:`\varphi_g` gives the pairing
back, **exactly** and with no measure factor left over:

.. math::

   \sum_g \sigma_g\,\varphi_g
   \;=\;
   \sum_g \frac{\int_g \sigma\varphi\,\mathrm{d}E}{\int_g
   \varphi\,\mathrm{d}E}\;\int_g \varphi\,\mathrm{d}E
   \;=\;
   \sum_g \int_g \sigma\varphi\,\mathrm{d}E
   \;=\;
   \int_0^\infty \sigma(E)\varphi(E)\,\mathrm{d}E
   \;=\; r .

That identity is the theorem. The two variances were *chosen* so that
the discrete pairing is the continuous integral with **weight one**;
introducing a lethargy width :math:`\Delta u_g` would double-count the
width and break rate preservation
(:eq:`energy-condensation-counting-measure`). Lethargy is the node
*coordinate*, never the *weight*.

.. note::

   **This is the spatial axis's exact opposite, and the contrast is the
   fastest way to remember which is which.** A spatial flux
   :math:`\phi_i` **is a density** — it was never integrated over the
   cell — so pairing it against a cross section requires the geometric
   volume measure :math:`V_i`
   (:eq:`sn-homogenization-fine-rate`). An energy flux
   :math:`\varphi_g` **is already an integral**, so pairing it requires
   nothing. Same equation, opposite measures, and the difference is
   entirely in which variance the discretization chose. Getting this
   backwards is the classical missing-width / double-counted-width bug
   in group-constant generation.

What the theorem buys at the space layer
-----------------------------------------

Three statements, each a direct consequence:

#. **The energy metric is** :math:`G_E = I_{n_g}`. Not "defaults to",
   not "is conventionally taken as" — *is*, by the argument above. In
   axis vocabulary: the energy factor's ``weights`` is ``None``, which
   IS the counting measure.
#. :math:`V \cong V^*` **isometrically along energy.** The Riesz map on
   that factor is the identity, so the distinction between a flux
   (:math:`V`) and a cross section (:math:`V^*`) is carried entirely by
   the *role*, never by a metric that would have to be applied to move
   between them.
#. **The adjoint along energy is the plain transpose.** Since
   :math:`A^\dagger = G^{-1} A^{\mathsf T} G`
   (:doc:`/theory/foundations/operator_adjoint`) and :math:`G_E = I`,
   the Hilbert adjoint and the Euclidean transpose coincide *on that
   factor*. This is why the energy-only operators of the homogeneous
   solver could be bound to a real space with **no value motion at
   all** — see :ref:`spaces-development-history`.

Construction enforces it
-------------------------

The theorem is not left as prose for a future contributor to violate.
:class:`~orpheus.numerics.axis.EnergyAxis` **refuses weights at
construction**, with a message that states the reason; a deliberately
non-physical weighted toy must use a generic
:class:`~orpheus.numerics.axis.Axis`, which is exactly what the
``.H``-sensitivity control in the CS1 battery does. Both constructors —
:meth:`EnergyAxis.from_grid <orpheus.numerics.axis.EnergyAxis.from_grid>`
and :meth:`EnergyAxis.synthetic
<orpheus.numerics.axis.EnergyAxis.synthetic>` — mint ``weights=None``.
The axis is also ``NODAL`` and rank-1 by construction, both refused
otherwise.

.. _spaces-energy-grid-is-a-mesh:

EnergyGrid is a 1-D mesh in energy
-----------------------------------

The reading that makes the energy axis a *member of the same family* as
the spatial one, rather than a special case:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Mesh concept
     - In energy
     - Carrier
   * - cells
     - the groups
     - ``shape = (ng,)``
   * - faces
     - the group boundaries
     - ``edges`` (``ng + 1`` values)
   * - the mesh-overlap map
     - condensation (fine → coarse)
     - :meth:`EnergyGrid.overlap_to
       <orpheus.data.energy_grid.EnergyGrid.overlap_to>`
   * - a one-cell mesh
     - the one-group member
     - :math:`\bar\sigma` still needs its edges and its weighting
       spectrum

Edges follow the canonical fast-first convention — strictly
DESCENDING, group ``0`` the fastest
(:ref:`canonical-group-convention`). The invariant is checked once, in
:class:`~orpheus.data.energy_grid.EnergyGrid`'s own construction, and
deliberately not re-checked on the axis.

Two consequences of the faces reading are worth stating explicitly,
because both were design forks:

- **Identity is** :math:`n_g` **plus the edges' CONTENT.** Not the
  ``EnergyGrid`` object. ⚠ ``Mixture.energy_grid`` mints a FRESH
  ``eq=False`` grid on every access, so ``is`` and ``==`` are both
  ``False`` for two reads of one mixture; an axis identity keyed on grid
  object identity would make two mints from one mixture disagree inside
  a single legitimate solve. The axis reads ``edges.tobytes()``.
- **A synthetic axis is NOT a from-grid axis.** ``synthetic(ng)`` is the
  honest spelling for a fixture or library that declares a group COUNT
  with no boundary energies — `[M]` every shipped ``get_mixture`` pair
  has ``eg is None``, so this is the common case, not the exotic one.
  Same index set, no partition data: a different axis.

.. _spaces-vv-collapse-hook:

The :math:`V` / :math:`V^*` collapse hook (declared, not yet built)
-------------------------------------------------------------------

The counting theorem has a sequel the axis is *recording data for*, and
this page declares it now so the eventual implementation is checked
against a stated contract rather than invented:

- **Condensation acts as a plain SUM on** :math:`V`. Integrals add:
  :math:`\Phi_G = \sum_{g \in G}\varphi_g`
  (:eq:`energy-condensation-coarse-flux`).
- **Condensation acts as a flux-weighted AVERAGE on** :math:`V^*`.
  Averages re-weight:
  :math:`\Sigma_G = \sum_{g\in G}\varphi_g\Sigma_g \big/
  \sum_{g\in G}\varphi_g`
  (:eq:`energy-condensation-vector-collapse`).
- **Collapse adjoint-consistency IS precisely that pair being mutually
  adjoint** under the counting pairing. Rate preservation
  (:eq:`energy-condensation-rate-preservation`) is what that
  adjointness says in physics vocabulary — which is why the corpus
  reads condensation as a Petrov-Galerkin projection rather than an
  averaging recipe (:ref:`sn-energy-condensation`).

.. warning::

   **This is a DECLARATION, not shipped machinery.** The pair above is
   what the group structure recorded on
   :class:`~orpheus.numerics.axis.EnergyAxis` exists to feed; the
   morphisms themselves are scheduled for a later phase / campaign 2
   (⚠ this row read "S7" until 2026-08-24: a bare plan-internal step
   number, and it COLLIDES with campaign 1 CS4b's own step S7, which
   landed that day and built none of this — trust the tree, not the
   number), and the condensation that ships today
   (:meth:`Mixture.condense
   <orpheus.data.macro_xs.mixture.Mixture.condense>`,
   :meth:`Solution.condense <orpheus.sn.solution.Solution.condense>`)
   does not route through an axis. Do not read this section as a
   description of a code path.


.. _spaces-metric-object:

The metric is an OBJECT: three sources, one resolution
======================================================

Everything above treats a space's metric as a **measure** — a weight per
index, stored per axis, applied by broadcasting. That is the right object
for a *measure*, and it is not the general object. A discrete Hilbert
space's metric is a symmetric positive-semi-definite **bilinear form**

.. math::

   \langle x, y\rangle_G \;=\; y^{\mathsf T} G\, x ,
   \qquad G = G^{\mathsf T},\quad G \succeq 0 ,

and nothing in that definition makes :math:`G` diagonal. Until campaign 1
phase **P7** (2026-08-30) the tree could spell exactly two realizations,
both of them a thing that is *multiplied* into the element — a broadcast
weight array (the Hadamard product :math:`G\odot x`) and the per-axis
factor measures of an axis-built space — so a form with off-diagonal
structure was **unspellable at every level**, not merely unimplemented.
P7 makes the metric a thing that is **applied**: a typed
:class:`~orpheus.numerics.metric.HilbertMetric` family owned by the
space, of which the Hadamard weight is the diagonal special case it
always was.

.. admonition:: Why this is a foundations concern and not a frame detail
   :class: tip

   Two independent consumers demanded the same missing primitive, which
   is what promoted it from a loose end to a layer:

   - the **slab spherical-harmonic frame**, whose measured discrete Gram
     carries live off-diagonals at :math:`0.93` of the Cauchy–Schwarz
     scale, so no diagonal metric satisfies Parseval on it
     (:ref:`frame-parseval-dense-arm`); and
   - the **curvilinear multi-moment cell mass** :math:`M/V`, whose true
     value at a spherical pole cell is
     ``[[1, 0.5], [0.5, 0.4]]``
     — dense *and* cell-dependent.

   Only the first is closed by P7, and the reason is measured rather
   than asserted: see :ref:`spaces-metric-not-on-the-axis` below.

.. _spaces-metric-three-sources:

Three sources, and the order that resolves them
-----------------------------------------------

A :class:`~orpheus.numerics.space.FunctionSpace` now has three places a
metric can come from, and **exactly one of them may be occupied**:

.. list-table:: The metric sources, in resolution order
   :header-rows: 1
   :widths: 20 34 46

   * - Source
     - Realization
     - When it is the right one
   * - ``axes``
     - the per-axis path
       (``FunctionSpace._apply_axes_weights``) — each factor's weight
       vector broadcast into its own index block
     - An axis-built space (:eq:`spaces-axis-product`). The axes **are**
       the metric source; the space never consults the other two.
   * - ``metric``
     - the object itself —
       :class:`~orpheus.numerics.metric.DiagonalMetric`,
       :class:`~orpheus.numerics.metric.DenseMetric`, or a
       :class:`~orpheus.numerics.metric.FactoredMetric` product of them
     - A form no weight array can spell. Today's founding occupant is
       the frame's matrix Parseval metric.
   * - ``inner_product_weights``
     - resolved to a
       :class:`~orpheus.numerics.metric.DiagonalMetric`, whose
       arithmetic is operation-for-operation the arms that used to live
       inline
     - The legacy dense weight array — every pre-P7 metric, and still
       the ordinary case.
   * - *(none)*
     - the verbs short-circuit without constructing an object
     - Euclidean, :math:`\sum_i x_i y_i`.

Resolution happens once per space, in
``FunctionSpace._resolved_metric``, and the three metric verbs —
:meth:`~orpheus.numerics.space.FunctionSpace.apply_metric`,
:meth:`~orpheus.numerics.space.FunctionSpace.apply_inverse_metric` and
:meth:`~orpheus.numerics.space.FunctionSpace.inner_product` — delegate
to whatever it returns. Composite spaces
(:class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`,
:class:`~orpheus.numerics.coupled_system.CoupledSpace`) override the
three verbs to dispatch **per block** and therefore inherit the family
for free: a dense-metric interior block flows through a composite with
no composite-level change.

**Exclusivity is a construction guard with three pairwise arms**
(``axes``/``weights``, ``axes``/``metric``, ``weights``/``metric``), and
the restructuring is not cosmetic. Before P7 the check sat *behind* an
``if self.axes is None: return`` early exit, so the
``(weights, metric)`` arm was **structurally unreachable** — a guard arm
with no possible witness (``vv-principles`` #17's granularity trap, in
its purest form: the guard would have been mutated as a unit and
certified by whichever arm the suite happened to reach). Each arm now
carries its own gate in ``tests/numerics/test_space_of_axes.py``.

.. warning::

   **The flip found a whole subclass family standing outside the
   construction contract.** All three
   :class:`~orpheus.numerics.space.FunctionSpace` subclasses that
   define their own ``__post_init__`` —
   :class:`~orpheus.numerics.spaces.spherical_harmonic_space.SphericalHarmonicSpace`,
   :class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`
   and
   :class:`~orpheus.numerics.spaces.angular_trace_space.AngularFaceTraceSpace`
   — **overrode without chaining**, so *every* base construction guard
   (not only the new one) had been silently opted out of since each
   subclass was written. All three now call ``super().__post_init__()``
   first. It surfaced only because the first dressing attempt left the
   spherical-harmonic basis's continuum weights in the field *beside*
   the new metric object and nothing refused it — the two-source illegal
   state was representable exactly where the guard was supposed to live.
   This is the standing argument for making a guard's own arms
   individually falsifiable: the arm that has no witness is the arm that
   is not running.

.. important::

   **⚠ An empty weights slot no longer means "Euclidean".**
   ``inner_product_weights is None`` describes the *diagonal source*
   only; a dense-metric space reads ``None`` there while carrying a
   real, non-Euclidean metric.
   Read the space's behaviour through
   :meth:`~orpheus.numerics.space.FunctionSpace.inner_product`, never
   through the slot. `[M]` re-measured 2026-09-07 after item 6.2a, and
   the production exposure is still small — worth stating so the hazard
   is not over-read: of the **221** lines across **57** files in
   ``orpheus/`` + ``tests/`` that name the slot, **28** carry an
   ``is None`` / ``is not None`` spelling (prose included) and only
   **four** are production BRANCHES — all four inside ``space.py``
   itself: the two arms of the exclusivity guard, the resolution in
   ``_resolved_metric``, and — fourth since 6.2a — the dense-slot-leaf
   arm of the tensor product's factored builder, which positions a
   hand-built factor's weights as a
   :class:`~orpheus.numerics.metric.DiagonalMetric` on that factor's
   block. (This paragraph read *"198 lines across 52 files … 20 are*
   ``is None`` / ``is not None`` *branches and only three are
   production"* until then. The fourth is a change of SPELLING, not of
   exposure: the pre-6.2a builder funnelled both a dense-slot leaf and
   an axis-built factor through one local ``w is not None`` test — the
   latter via the densifier — where the arms are now separate and the
   densifier is gone.)

.. _spaces-metric-moore-penrose:

The Moore–Penrose doctrine, extended from a reciprocal to a matrix
-------------------------------------------------------------------

:meth:`HilbertMetric.apply_inverse
<orpheus.numerics.metric.HilbertMetric.apply_inverse>` is the
**pseudo**-inverse :math:`G^{+}` — the reciprocal on the metric's range,
zero on its kernel — for every realization. On the diagonal arm this is
the doctrine the tree has carried since the trace-metric work (a
tangential ordinate with :math:`|\Omega\cdot n| = 0` has weight zero and
must map to zero, not to infinity). The dense arm extends it unchanged,
and there it is **not a convenience: it is the only thing that exists.**

`[M]` 2026-08-30 on ``Quadrature.gauss_legendre(8).angular_frame(2)`` —
the flagship consumer — the discrete Gram is :math:`15\times15` with
**5 live diagonal slots and rank 4**: its singular values are
:math:`2.708,\ 1.419,\ 4.925\times10^{-1},\ 4.745\times10^{-2}` and then
the round-off floor (:math:`\sim10^{-17}`, fifteen orders below the
smallest genuine mode). :func:`numpy.linalg.inv` raises there;
:func:`numpy.linalg.pinv` returns the object Parseval needs.

Two construction choices in
:class:`~orpheus.numerics.metric.DenseMetric` are worth their reasons:

#. **The cutoff is PINNED, not left to the library default.** An
   implicit default is a silent dependency on a NumPy version, and a
   mutation arm that over-truncates is only meaningful against a fixed
   value. The shipped ``rcond`` is :math:`10^{-12}`, and the cliff it
   must sit below is computable rather than guessed: NumPy's ``rcond``
   is relative to the largest singular value, so truncation begins at
   :math:`\sigma_{\min}^{\rm live}/\sigma_{\max} = 4.745\times10^{-2} /
   2.708 = 1.75\times10^{-2}`. `[M]` scanned on the flagship Gram, the
   Parseval ratio reads ``1.000000000`` at every scanned ``rcond`` in
   :math:`[10^{-15},\,10^{-2}]` — as it must, since no truncation can
   occur below the cliff — and drops to ``0.991414787`` from
   :math:`3\times10^{-2}` upward, where the genuine
   :math:`4.745\times10^{-2}` mode is discarded. :math:`10^{-12}` sits
   ten orders below the cliff and five above the noise floor.
#. **The pseudo-inversion declares hermiticity.** `[M]` the same call
   *without* ``hermitian=True`` returns a matrix whose asymmetry is
   :math:`\max|M - M^{\mathsf T}| = 4.74\times10^{-14}`, against
   :math:`1.1\times10^{-16}` with it — three orders, and enough to trip
   the type's own symmetry admission on a re-wrap.

The type refuses two illegal states outright rather than repairing them:
an **asymmetric** matrix (an asymmetric form is not an inner product —
producers symmetrize, the type does not), and an explicitly supplied
inverse face that fails the **Penrose identity** (an inconsistent
:math:`(G, G^{+})` pair is not a representation choice). Both thresholds
are module constants, so a gate on the type quotes the type's own
number (``vv-principles`` #16).

.. _spaces-metric-parseval-theorem:

Parseval is a THEOREM for any Gram, singular or not
----------------------------------------------------

The pseudo-inverse is not a graceful degradation of the invertible
story; it is what makes the story *general*. For a band-limited field
:math:`\psi = S_0 c` the analysis face returns the **covariant** moments
:math:`\varphi = M\psi = Gc` identically
(:eq:`frame-analysis-is-the-gram`, pure algebra). Measuring those
moments under :math:`G^{+}` and using the first Penrose identity
:math:`G G^{+} G = G`:

.. math::
   :label: spaces-pseudo-inverse-parseval

   \|M\psi\|^2_{G^{+}}
   \;=\; (Gc)^{\mathsf T} G^{+} (Gc)
   \;=\; c^{\mathsf T}\bigl(G G^{+} G\bigr) c
   \;=\; c^{\mathsf T} G\, c
   \;=\; \|S_0 c\|^2_W .

.. (vv-status rationale) A representational identity: it derives the
   Parseval isometry (:eq:`frame-parseval-isometry`, whose SSOT is
   :ref:`frame-parseval-metric` on the frame page) for a SINGULAR Gram
   by substituting the first Penrose identity. No flux, no eigenvalue,
   no discretization error — the metric-object layer's statement of a
   claim the frame page owns in the frame register. The verifiable
   content is the DenseMetric law battery
   (``tests/numerics/test_dense_metric.py``, hand-derived literals in
   exact binary fractions plus the range-projector leg) and, on the
   frame side, the four-mechanism dressing gate, the isometry gate's
   slab row, and the wrong-metric discriminator
   (``tests/numerics/test_frame.py``).
.. vv-status: spaces-pseudo-inverse-parseval documented

.. no-implementation:: spaces-pseudo-inverse-parseval
   :kind: identity

   **Nothing implements this.** Both sides are computed — the left by
   :meth:`FunctionSpace.inner_product
   <orpheus.numerics.space.FunctionSpace.inner_product>` on the frame's
   dressed coefficient space, the right by the same verb on the measure
   space — and no line in production forms the comparison. That is the
   point: the equality is what lets the dressing be a one-line
   ``replace`` rather than a bespoke correction. It is *measured* by the
   Parseval isometry gate.

The identity holds because :math:`G G^{+} G = G` is one of the four
Penrose conditions, true by definition of the pseudo-inverse — no
hypothesis about rank, conditioning or invertibility enters. `[M]` on
the flagship slab Gram the residual is
:math:`\max|G G^{+} G - G| = 1.55\times10^{-15}` absolute, which is
:math:`7.8\times10^{-16}` relative to :math:`\max|G| = 2` (and
:math:`7.7\times10^{-16}` in the Frobenius ratio — the norm is worth
writing down, because three reasonable norms of the same residual differ
by an order).

.. _spaces-metric-one-spelling:

The pairing has ONE spelling, and that is a bit-identity contract
------------------------------------------------------------------

:meth:`HilbertMetric.pairing
<orpheus.numerics.metric.HilbertMetric.pairing>` is
``float(np.sum(self.apply(x) * y))`` — :math:`\sum (Gx)\odot y` — for
**every** realization, inherited rather than overridden. Two reasons,
both load-bearing:

**Bit-identity with the legacy path.** The pre-P7 diagonal spelling was
``np.sum(w * x * y)``, which Python evaluates left-to-right as
``(w*x)*y``; ``DiagonalMetric.apply(x)`` *is* ``w_b * x``, so routing
the shipped diagonal path through the family preserves the reduction
tree exactly. The obvious alternative — densify and matmul,
:math:`y^{\mathsf T}(\operatorname{diag}(w)\,x)` — is **not** the same
program. `[M]` 40 seeds :math:`\times` 500 draws at :math:`n = 15`, the
two spellings disagree bitwise on **60.4 %–69.8 %** of draws, with the
worst gap ranging over **46–16 384 ULP** (relative
:math:`9.2\times10^{-15}`–:math:`2.1\times10^{-12}`) across seeds.
Note which half of that is publishable: the *fraction* is a stable
property of the arithmetic (two different reduction trees disagree on
about two draws in three), while the *ULP gap* is unbounded wherever the
sum nearly cancels and must never be frozen into a docstring or a gate
band. What the measurement licenses is the structural claim: a matmul
pairing would move pinned numbers tree-wide, so it is deliberately
unspellable here.

**One source of truth for the adjoint.**
``AdjointOperator`` builds :math:`A^{\dagger} = G^{-1}A^{\mathsf T}G`
from ``apply_metric``/``apply_inverse_metric`` while the pairing that
*judges* it comes from ``inner_product``. Deriving the pairing from
``apply`` makes those two agree by construction — the ERR-067 family
(two spellings of one metric diverging silently) becomes unspellable
rather than merely untested.

.. _spaces-metric-propagation:

Composition: a dropped metric is a VALUE bug wearing a representation costume
-------------------------------------------------------------------------------

A new representation is invisible to every consumer that propagates the
old one by copying an array. Three such sites exist, and all three were
measured *silently wrong* on a dense-metric space before P7 taught them
— each failing in the flattering direction (reverting to Euclidean, or
to a plausible-looking value, with no error and no warning):

.. list-table:: The three propagation sites
   :header-rows: 1
   :widths: 26 37 37

   * - Site
     - Pre-P7 behaviour on a dense-metric operand
     - What it does now
   * - :meth:`DualSpace.of
       <orpheus.numerics.space.DualSpace.of>`
     - copied ``inner_product_weights`` only, so the dual of a
       dense-metric space read the **plain Euclidean** pairing.
       `[M]` on the ``test_dense_metric`` fixture the dual read
       :math:`4.5` where the primal reads :math:`23.25`
     - threads the metric object (L²-Riesz: the dual carries the SAME
       metric), so both read :math:`23.25`
   * - :meth:`TensorProductSpace.from_factors
       <orpheus.numerics.space.TensorProductSpace.from_factors>`
       (the legacy ``*``)
     - densified the factor measures into one outer-product weight
       array — which has no slot for a matrix factor, so the dense
       factor was **dropped** and its block went Euclidean
     - **since 2026-09-07 (CS4c step 6 item 6.2a) the factored arm is
       the ONLY non-axis arm**: a lazy
       :class:`~orpheus.numerics.metric.FactoredMetric` positioned per
       **axis** of an axis-built factor and per dense-slot leaf factor,
       Kronecker never materialized, and ``inner_product_weights`` never
       populated by a product at all.
       ⛔ This cell read *"grows a dense-factor arm: a lazy*
       ``FactoredMetric``\ *, one positioned entry per factor"* until
       then. That was true of P7 and it described an arm that was
       **inert**: `[M]` the step-6 activation census counted **0**
       entries across 11 SN runs (~450 product mints), because no factor
       reaching ``*`` on an SN path carried a metric OBJECT — the only
       production installs are the frame's
       ``DenseMetric.inverse_of(discrete_gram)`` and the two ψ½ ray
       spaces, none of which enters a product. What 6.2a changed is the
       arm's REACHABILITY, not its existence: the dense arm it shared the
       dispatch with is gone, so every axes-less product now takes it
   * - :attr:`FrameBase.gram
       <orpheus.numerics.frame.FrameBase.gram>`
     - ``replace(test_space, inner_product_weights=diagonal)`` on a
       dressed test space handed the **row-sum probe** a space whose
       ``apply_inverse_metric`` ignores that slot and applies its own
       matrix
     - **strips** the metric object while installing the probe
       diagonal — the cross-Gram machinery must never inherit the
       trial-side Parseval dressing

The third is the sharpest, because it is not a missing feature but a
live projection error on a production path
(:meth:`FrameBase.project <orpheus.numerics.frame.FrameBase.project>`
is what homogenisation and condensation call). `[M]` on the overlap
frame — Gram
``[[1.25, 0.25], [0.25, 1.25]]``,
condition number :math:`1.50` — the pre-P7 spelling computes
:math:`G\,(Mf)` instead of :math:`\operatorname{diag}^{+}(Mf)` and
returns ``[7.0, 11.0]`` where the true projection is
:math:`[8/3,\,16/3]`: **162.5 %** wrong on the first component,
**106.3 %** on the second. Post-P7 that spelling is not merely wrong,
it is **unrepresentable** — the exclusivity guard refuses the
two-source space at construction, `[M]` with
``ValueError: space 'L2[coarse_cells(spatial_R1)]' carries BOTH dense
inner_product_weights and a metric object`` (the space name gained its
manifold at #429 tracker 2.1; the guard is unchanged). The illegal state that
produced a 162 % value error is now a state that cannot be built.

.. note::

   **The tensor product's laziness is the same discipline the axis path
   already follows.** :class:`~orpheus.numerics.metric.FactoredMetric`
   stores ``(block_shape, factor)`` pairs and applies them in sequence,
   each to its own index block; the Kronecker product
   :math:`G_1\otimes G_2\otimes\cdots` is never formed. The pairing is
   nonetheless exact against a densified reference: `[M]` a
   :math:`(3,)` dense-metric factor times a :math:`(2,)` weighted
   factor reads ``141.5``, bit-equal to
   :math:`y^{\mathsf T}\bigl(G\otimes\operatorname{diag}(w)\bigr)x`
   built with :func:`numpy.kron`, where dropping the dense factor would
   have read ``45.0``. On separable probes the pairing factorizes
   exactly, which is what makes a hand-derived pin possible:
   :math:`23.25 \times 2.5 = 58.125` in exact binary fractions.

.. warning::

   **This does NOT mean an axis-built product can carry a matrix.**
   :eq:`spaces-axis-product` says the metric of an ``of_axes`` product
   is :math:`\bigotimes_a G_a` with
   :math:`G_a = \operatorname{diag}(w_a)` or :math:`I`, and that
   remains exactly true — see :ref:`spaces-metric-not-on-the-axis`. The
   :class:`~orpheus.numerics.metric.FactoredMetric` arm belongs to the
   **legacy** ``*`` composition, which takes whole *spaces* as factors
   and therefore can meet one that carries a metric object. ⛔ This
   paragraph closed *"The two paths were not unified by P7 and CS2 still
   retires the second"* until **2026-09-07**. The retirement it promised
   landed as **CS4c step 6 item 6.2a** — and what retired is the
   *densifier*, not ``*``: the product's dense weights array and the
   bridge that fed it are gone, so both paths now apply their measure
   factor-by-factor. They are still not one path, and the residue is
   narrow and named: an axes-less factor makes the product axes-less, so
   ``*`` cannot answer an axis query. `[M]` 2026-09-07 the only production
   occupants are the four harmonic/moment mints; item **6.2c** axis-ifies
   the angular head and closes it.

.. _spaces-metric-dressing-evidence:

The dense dressing, and the only evidence that can adjudicate a metric
------------------------------------------------------------------------

The founding consumer installs it:
:attr:`FrameBase.basis_space
<orpheus.numerics.frame.FrameBase.basis_space>` on a ``DENSE`` verdict
now returns the basis's space dressed with
``DenseMetric.inverse_of(discrete_gram)`` — the pseudo-inverse at the
pinned cutoff, with the exact symmetrized Gram kept as the inverse face
(:math:`(G^{+})^{+} = G` for a symmetric PSD form, so this is exact and
strictly better conditioned than a second pseudo-inversion) — and
**strips** the basis's continuum weights, because the dressing
*replaces* the metric on that arm exactly as the diagonal arm
overwrites it. Full account, including the slab Gram's own table:
:ref:`frame-parseval-dense-arm`.

⛔ **Reciprocity cannot adjudicate a metric, and it never will.** The
Hilbert-adjoint identity :math:`A^{\dagger} \equiv G^{-1}A^{\mathsf T}G`
holds to :math:`1.4\times10^{-16}` for **every** invertible :math:`G` —
Euclidean, random, :math:`(V\!\cdot\! w)^3` — because ``.H`` is *built
from* the stored metric and a sandwich always reproduces the pairing it
was assembled from (ERR-039's first shield, at this layer). A
reciprocity gate therefore proves *loadedness* and carries **zero**
information about *which* metric is installed. The instrument that can
fail is one that compares the metric against a quantity defined without
it — Parseval, i.e. the field's own norm — plus hand-derived literals
with no solver, frame or quadrature in the chain.

.. list-table:: The wrong-metric discriminator — `[M]` 2026-08-30, one band-limited :math:`\psi` on ``gauss_legendre(8).angular_frame(2)``
   :header-rows: 1
   :widths: 40 24 36

   * - Metric on the coefficient space
     - Parseval ratio
     - Verdict
   * - the basis's **continuum** Gram (pre-F-0, undressed)
     - :math:`25.53`
     - the ERR-039 defect
   * - the best **diagonal** candidate, :math:`1/\operatorname{diag}(G)`
     - :math:`1.806`
     - a diagonal metric is *provably insufficient* here, not merely
       unavailable
   * - the **matrix** pseudo-inverse :math:`G^{+}` (shipped)
     - :math:`0.999999999999999`
     - the theorem, :eq:`spaces-pseudo-inverse-parseval`

The middle row is the whole justification for the phase, and it is the
row a diagonal-only design could never have produced. ⚠ Its **floor is
frame-dependent** and only the slab's is gated: `[M]` on the same
construction the diagonal candidate reads :math:`1.066` on
``product(4,4)`` at :math:`L=2` and :math:`0.996` on
``level_symmetric(4)`` at :math:`L=3` — close enough to 1 that a gate
pinned there would be pinning noise.

**The dressing moves a production adjoint, and nothing observed it.**
This is not plumbing. The scattering operator builds
``quadrature.angular_frame(scattering_order)`` in production, so
dressing a ``DENSE`` frame changes the spaces every ``.H`` through the
adjoint sandwich reads. `[M]` comparing the dressed and undressed
analysis adjoints **as operators** (column by column on the coefficient
basis, so the number carries no random draw):

.. list-table:: Movement of ``frame.analysis.H`` under the dressing
   :header-rows: 1
   :widths: 34 22 22 22

   * - Frame
     - :math:`\max|\Delta|`
     - relative (max-norm)
     - relative (Frobenius)
   * - ``product(4,4)``, :math:`L=2`
     - :math:`12.49`
     - :math:`0.994`
     - :math:`0.985`
   * - ``gauss_legendre(8)``, :math:`L=2`
     - :math:`12.39`
     - :math:`0.986`
     - :math:`0.980`
   * - ``level_symmetric(4)``, :math:`L=3`
     - :math:`12.49`
     - :math:`0.994`
     - :math:`0.980`

The two operators are not a small correction apart; they are
essentially unrelated, which is the correct reading of a repair whose
"before" state the tree's own docstring called *"the stored-metric
sandwich, NOT the physical Hilbert adjoint"*. ⚠ Do not freeze a
single-vector figure for this: a random-probe relative movement on
``product(4,4)`` is `[M]` **0.879–0.986** over 200 seeds, so a quoted
point value inside that band is one draw's reading, not a property of
the frame.

.. _spaces-metric-frame-square:

What does NOT ride along: the frame square's scalar closure
-------------------------------------------------------------

Installing the right metric does **not** make every identity that
mentions a metric come true, and the distinction is worth a subsection
because it is exactly where a reader will over-read the repair.

The spherical-harmonic frame square closes on one scalar,
:math:`M^{*} = R/W` and :math:`R^{*} = W M`
(:eq:`frame-square-closure-sh`), and that identity needs the per-degree
Gram :math:`G_\ell` to be **a single number for each** :math:`\ell`.
Writing :math:`d` for the addition-theorem factors
(:math:`d_\ell = 2\ell+1`) and :math:`Y` for the synthesis table, the
closure is the operator statement

.. math::

   Y\Bigl(G^{+} - \tfrac{1}{W}\operatorname{diag}(d)\Bigr) \;=\; 0 ,

i.e. the metric and the reconstruction weights need agree only **modulo
the kernel of** :math:`Y`. That makes the relation decidable and it
splits the shipped frames three ways rather than two:

.. list-table:: `[M]` 2026-08-30 — the closure residual is not implied by the metric being right
   :header-rows: 1
   :widths: 30 14 22 34

   * - Frame
     - Verdict
     - rel :math:`\|M^{*}y - Ry/W\|`
     - Why
   * - the six ``DIAGONAL`` sphere frames — LS\ :sub:`4` at
       :math:`L{=}1` and :math:`L{=}2`, and LS\ :sub:`8`, product
       :math:`8\times8`, folded :math:`8\times8` and Lebedev-13 at
       :math:`L{=}2`
     - ``DIAGONAL``
     - :math:`\le 1\times10^{-15}`
     - each live :math:`\ell` block is one constant; per-degree
       diagonal spread :math:`\le 4\times10^{-15}`
   * - ⛔ ``gauss_legendre(8)``, :math:`L=2` — **this row is HISTORY**
     - was ``DENSE``
     - was :math:`0.30`–:math:`10.2` (200 seeds)
     - the live :math:`\ell{=}2` diagonal was
       :math:`[0.4,\,0.8,\,0.8]` — no :math:`G_\ell` existed, at
       **any** metric. ⭐ Both :math:`0.8`\ s were ERR-080's fabricated
       :math:`m \ne 0` slots; since #429's fused commit (2026-09-02) a
       1-D rule binds a FLAT Legendre head and this frame is
       ``DIAGONAL`` with :math:`G_\ell = 2/(2\ell+1)` — one number per
       degree — so it moves into the **first** row of this table.
       `[M]` 2026-09-02, 200 seeds: rel :math:`\le 5.1\times10^{-16}`,
       and the same at ``gauss_legendre(4)``, ``(16)`` and at
       :math:`L = 3`
   * - ``product(4,4)`` :math:`L{=}2`; ``level_symmetric(4)``
       :math:`L{=}3`; ``folded_product(2,4)`` :math:`L{=}3`
     - ``DENSE``
     - `[M]` 2026-09-02, 200 seeds:
       :math:`7.6\times10^{-4}`–:math:`0.47`,
       :math:`4.3\times10^{-2}`–:math:`0.15`,
       :math:`0.26`–:math:`1.65`
     - same cause, milder: only the top degree's block is non-constant.
       (The third is new here — it takes over the slab's role as the
       loudest breaker, on a SHIPPED rule and a basis the repair does
       not touch)
   * - ``folded_product(4,6)``, :math:`L=3`; ``gauss_legendre(2)``,
       :math:`L=2`
     - ``DENSE``
     - :math:`\le 3\times10^{-15}` / `[M]`
       :math:`\le 1.6\times10^{-15}` (200 seeds each)
     - their coupling is a pure **rank deficiency**, so the
       disagreement lives entirely in :math:`\ker Y`. ⭐ The second is
       a 2026-09-02 addition and it is the stronger exhibit, because it
       is a **Legendre** frame and its rank deficiency has a closed
       form: :math:`P_2` vanishes identically at ``GL_2``'s two nodes,
       which ARE its roots (the dead-slot theorem —
       :math:`G_{22} \sim 10^{-33}`). Two families, one mechanism

The last row is the one that makes the statement a theorem rather than a
correlation, and it was found by measuring rather than by reasoning
(and it now has a second, independently-caused member — see the
``gauss_legendre(2)`` note in that row).
That frame's only live off-diagonal couples two :math:`\ell = 3` slots
whose :math:`2\times2` block is
``[[0.6732, 0.8691], [0.8691, 1.1220]]``
— `[M]` determinant :math:`-8.7\times10^{-17}`, **rank 1**: the two
harmonics are linearly *dependent* on that folded node set. So
:math:`G^{+}` and :math:`\operatorname{diag}(d)/W` differ by an
:math:`O(1)` amount that :math:`Y` annihilates: `[M]`
:math:`\|Y D\|_\infty = 4.4\times10^{-16}` with
:math:`\|D\|_\infty = 0.557`, against
:math:`\|Y D\|_\infty = 6.30` on the slab.

⟹ **the** ``DIAGONAL`` **verdict is sufficient for the scalar
closure, and** ``DENSE`` **does not decide it.** That is why the
frame-square gate keeps only the
diagonal parameters — not because a dense frame always breaks the
closure (one shipped frame does not), but because the verdict does not
*imply* it, and a gate must not assert what its population cannot
guarantee. The discriminator is Gram structure, never geometry: a
*sphere* rule (``product(4,4)``) is among the frames that break it.

.. _spaces-metric-not-on-the-axis:

Where the metric may NOT live, and what stays refused
-------------------------------------------------------

Three rulings bound the phase. Each is a decision with a reason, not a
scheduling accident.

**1. The axis keeps a MEASURE; it does not grow a form.**
:attr:`Axis.weights <orpheus.numerics.axis.Axis.weights>` stays a 1-D
weight vector, and :eq:`spaces-axis-product` stays literally true. The
distinction is not notation: *a measure is diagonal by nature* — it
assigns a number to each atom of an index set — whereas a Gram is a
**bilinear form** on the space the atoms span, and the two coincide only
when the basis is orthogonal on that measure. The generator induces, the
space holds: a frame that measures its own Gram dresses the *space* it
mints, and no axis is asked to carry a matrix it has no way to produce.

**2. Identity stays metric-blind.** Space identity is
:math:`(\text{name}, \text{shape})`, so a dressed and an undressed space
of the same name compare equal and hash equal, and the frame's
``basis_space == basis.space`` invariant survives the dressing
untouched. The ``metric`` field is declared ``compare=False`` for the
same **structurally mandatory** reason the weights slot is: an ndarray
inside a dataclass-generated ``__eq__`` makes the comparison return an
array and ``hash()`` raise. (The chartered doctrine that *metric
differences imply space differences* is unaffected — it flows through
the axis-derived name, :ref:`spaces-identity-bridge`, and an axis-built
space cannot carry a metric object at all.)

⚠ **The consequence for any GATE over these two spellings**: a
``==`` assertion is structurally blind to which of them a producer
bound, so it cannot adjudicate a choice between them. `[M]` 2026-09-02
over 33 shipped (rule, :math:`L`) rows the frame's ``basis_space`` and
its ``basis.space`` are ``(name, shape)``-equal on **33 of 33** and
metric-different on **33 of 33**. When #429 tracker 2.5 bound the
angular operator ends to one of the two, the gate that pins the choice
therefore had to assert the metric ARRAY with the other spelling as its
negative control (``vv-principles`` #19) —
:ref:`frame-moment-space-single-home`.

**3. Expressible is not known — the curvilinear moment mass stays
refused.** The multi-moment cell mass on a curved chart was blocked on
**two** independent things, deliberately named because they need
opposite repairs: the *machinery* (a non-Hadamard space metric,
`GitHub #409 <https://github.com/deOliveira-R/ORPHEUS/issues/409>`_) and
the *value* (which :math:`G` is right is pinned by physical functionals
and needs the cell-solve consumer of
`#158 <https://github.com/deOliveira-R/ORPHEUS/issues/158>`_). P7 lands
the machinery only. The refusal in
``DiscretizationSchemeBase._assert_moment_mass_is_expressible`` therefore
re-derives to name **#158 alone**, and nothing installs a guess: a
metric that can now be *spelled* is not a metric anyone has *chosen*.

.. note::

   **Why the axis ruling and the curvilinear refusal are the same
   measurement read twice.** The two consumers need different strengths,
   and that asymmetry is the strongest argument for an
   operator-valued metric over a "matrix weights slot on ``Axis``":

   - the slab frame needs **one cell-independent** dense
     :math:`K\times K` coefficient Gram — a per-axis matrix could carry
     that;
   - the curvilinear mass is
     :math:`G_{(i,k),(i,j)} = V_i\,M_{kj}(r_i)` — dense in the moment
     index **and** varying along the spatial index, i.e.
     **non-separable** across (spatial :math:`\times` moment). No
     per-axis object of any shape, vector or matrix, can carry it.

   A design that closed the first consumer by widening ``Axis`` would
   have closed exactly one of the two and made the second harder, by
   putting the metric on the factor that cannot express it. The
   space-held object closes the first and leaves the second's *machinery*
   available for whatever chooses its value.


.. _spaces-moment-head:

The angular HEAD of a moment space — two families, one surface
-----------------------------------------------------------------

The metric sections above are about *what a factor measures with*. This
one is about a question that had never been asked because it had only
one answer: **what SHAPE does the leading factor of a moment space
have, and who is allowed to know?**

Landed 2026-09-02, #429 tracker 3.4/3.4b, inside the fused commit that
repaired :doc:`ERR-080 </theory/verification/error_catalog>`.

A moment field's space is ``<head> ⊗ cells``. Until 2026-09-02 the head
was always
:class:`~orpheus.numerics.spaces.spherical_harmonic_space.SphericalHarmonicSpace`
— the rectangular :math:`(L+1, 2L+1)` table with the
addition-theorem-shifted :math:`[\ell + m]` column and zero padding
outside :math:`|m| \le \ell`. The ERR-080 repair adds a second family:
:class:`~orpheus.numerics.spaces.legendre_space.LegendreSpace`, the
coefficient space of :math:`\{P_\ell(\mu)\}_{\ell \le L}` on the orbit
space :math:`S^2/O(2)_a`, which is **FLAT** — :math:`(L+1,)`, one
coefficient per degree, because the trivial isotypic component of the
:math:`O(2)_a` action — equivalently of its rotation half, which has the
same orbits (:ref:`manifold-orbit-space-stabiliser`) — is
one-dimensional in every degree
(:ref:`manifold-descending-slots`).

Two families with different ranks means the layout is a *variable*, so
it needs an owner. :class:`~orpheus.numerics.spaces.moment_head.MomentHead`
is a ``runtime_checkable`` ``Protocol`` carrying exactly what a consumer
must not assume: ``L``, ``shape``, ``isotropic_slot``,
``degree_block(l)``, ``truncated(L_new)``. Both classes satisfy it
structurally — no base class, no registration — and a consumer holding
``space.factors[0]`` narrows with ``isinstance``, the same
key-on-what-it-declares idiom as
:class:`~orpheus.numerics.basis.base.TruncatedBasis` on the basis side.

.. list-table:: What the head answers, per family
   :header-rows: 1
   :widths: 30 34 36

   * - Question
     - ``spherical_harmonic_space``
     - ``legendre_space(S^2/O2_a)``
   * - ``shape``
     - :math:`(L+1,\ 2L+1)`
     - :math:`(L+1,)`
   * - ``isotropic_slot``
     - ``(0, 0)``
     - ``(0,)``
   * - ``degree_block(l)``
     - ``(l, 0:2l+1)``
     - ``(l,)``
   * - rank ⟹ where the group axis is
     - 2 ⟹ ``values.shape[2]``
     - 1 ⟹ ``values.shape[1]``
   * - continuum metric
     - :math:`4\pi/(2\ell+1)` per degree, spread over the
       :math:`2\ell+1` live columns
     - :math:`4\pi/(2\ell+1)` per degree

⭐ **The two metrics agreeing is not a coincidence and it is the reason
the descent is an ISOMETRY rather than merely an isomorphism.** The
Legendre Gram is taken against the **pushforward** of :math:`d\Omega`
along the quotient map — :math:`\pi_*\,d\Omega = 2\pi\,d\mu` by
Archimedes' hat-box, which is what
:attr:`Quotient.reference <orpheus.numerics.manifold.Quotient.reference>`
carries — so

.. math::
   :label: spaces-legendre-pushforward-gram

   \int_{S^2/O(2)_a} P_\ell^2 \; \mathrm{d}(\pi_*\Omega)
   \;=\; \int_{-1}^{1} P_\ell(\mu)^2 \, 2\pi \,\mathrm{d}\mu
   \;=\; \frac{4\pi}{2\ell+1},

exactly the harmonics'
:attr:`metric_per_ell <orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis.metric_per_ell>`.

.. warning::

   ⚠ **NOT the bare Legendre mass-2 normalisation**
   :math:`2/(2\ell+1)`, which is the Gram against :math:`\mathrm{d}\mu`
   with no :math:`2\pi`. The two differ by a factor :math:`2\pi` and the
   wrong one would move **every operator end's metric** on every 1-D
   solve. The discriminator is which measure the orbit space carries,
   and the orbit space carries the pushforward of the sphere's — a
   1-D angular rule is a rule on :math:`S^2/O(2)_a`, not on an
   abstract interval. ⛔ That orbit space was spelled
   :math:`S^2/SO(2)_a` here until 2026-09-02; #432 renamed it onto the
   axis's full stabiliser, which changes no integral on this page —
   :math:`\pi_*\,d\Omega` is the pushforward along the SAME map
   (:ref:`manifold-orbit-space-stabiliser`).

   ⚠ And do not confuse either with the **discrete** Gram a frame
   measures, which is against the RULE's weights: `[M]` a
   Gauss–Legendre rule sums to :math:`W = 2`, so its discrete Legendre
   Gram is :math:`2/(2\ell+1)` and its Parseval-dressed ``basis_space``
   carries the inverse, :math:`(2\ell+1)/2`. Continuum, discrete and
   Parseval-dressed are three different arrays on this space exactly as
   they are on the harmonic one (:ref:`spaces-metric-three-sources`).

.. (vv-status rationale) A normalisation identity: it states which Gram
   the Legendre head carries and why, not what a solver computes. Its
   verifiable content is the constructor gate on
   ``LegendreSpace.from_L`` (the metric is sourced from
   ``LegendreBasis.metric_per_ell``, so the formula has one home) and
   the bit-identity of the descent, which would fail under the mass-2
   normalisation.
.. vv-status: spaces-legendre-pushforward-gram documented

⛔ **Why the Protocol is a repair and not decoration: on a flat head the
pre-2026-09-02 reads returned the wrong array and raised NOTHING.** A
consumer indexing ``values[0, 0]`` on an :math:`(L+1, n_g, n_x)` tensor
gets *group 0's spatial slice* — well-shaped, silently wrong. The full
list of sites that read the rectangular layout as if it were the
contract: ``scalar_flux``, ``isotropic_part``, ``anisotropic_part``,
``l_block``, the fission :math:`\ell = 0` dyad, ``ng`` (which located
the group axis at a hard-coded index 2), ``zeros_for_mesh_and_L``, and
the material field's per-degree group contraction, whose ``einsum``
spelled the :math:`m` axis into its subscripts and would have contracted
the GROUP axis as if it were :math:`m`. All of them read the head now;
the harmonic specs are the former inline ones **verbatim**, so that path
is bit-identical by construction, and a head of an unshipped rank is
refused by name rather than contracted wrongly. The frame-side account
is :ref:`frame-g0-descent-arrow`.


.. _spaces-collapse-doctrine:

The collapse doctrine: which axes survive a degeneracy, and why
===============================================================

This is the page's load-bearing content, and it is presented
**dialectically** — the question, then the two answers that were tried
and refuted *together with the questions that refuted them*, then the
doctrine that stands. That ordering is deliberate pedagogy, not
archaeology: the refuted versions are each *almost* right, and a reader
who meets only the final statement will re-derive one of them within a
week. The record was produced by an extended design dialogue on
2026-08-19/20 and is preserved at
``.claude/plans/cs1_energy_space_design.md`` Appendix A.


.. _spaces-collapse-the-question:

The question: two prior doctrines in tension
---------------------------------------------

The homogeneous infinite-medium solver, diffusion, S\ :sub:`N` and
P\ :sub:`N` differ in which tensor factors their spaces carry. The
architecture record that preceded this campaign stated two rules about
rank-one collapses, and they disagreed.

- **The retract rule.** Every canonical relation between spaces —
  the outflow half of a face slot, scalar inside angular, :math:`\ell=0`
  inside moments — has one shape: a projection :math:`\pi: V \to V'`, an
  embedding :math:`\iota: V' \to V`, with :math:`\pi\circ\iota =
  \mathrm{id}` and, under the right metrics, :math:`\iota = \pi^{H}` up
  to a scalar. On this reading a retract's codomain **DROPS** the
  collapsed axis: *scalar space is not angular space with a trivial
  axis*, and conflating the two is precisely the source of the classical
  :math:`4\pi` bookkeeping errors. What *is* a subspace of angular space
  is :math:`\iota(V_{\rm scal})`, the isotropic functions — a different
  object from :math:`V_{\rm scal}` itself.
- **The quotient rule.** The homogeneous solver is not measureless — it
  is maximally *quotiented*. Translation invariance quotients the
  spatial axis to a **one-point axis with unit measure**, and the "per
  unit volume" intensive convention *is* that normalized quotient
  measure. On this reading the collapsed axis **PERSISTS**.

Both collapses are rank-one. Both arguments are correct about their own
case. So the question the space layer has to answer is exactly:
**which rank-one collapses leave an axis behind, and what decides it?**


.. _spaces-collapse-version-1:

Version 1 — compactness (REFUTED)
----------------------------------

*Proposed.* A collapse over a **compact** domain integrates: the measure
is consumed by the integration, nothing problem-specific is left, and
the axis drops. That is angle — :math:`S^2` is compact, the total
:math:`4\pi` is universal. A collapse over a **non-compact** domain
cannot integrate (the integral diverges), so it must normalize, and the
normalization constant is problem data — the axis persists as the
quotient point. That is space — :math:`\mathbb{R}^d` under translations.

It reproduces both prior rules, it is a single criterion, and it is
wrong.

.. admonition:: ⛔ The refuting question — energy, and Bateman
   :class: error

   *Where does ENERGY sit?* Energy is collapsed by integration over
   intervals, so compactness would put it on the "integrate ⟹ drop"
   side. But the one-group flux manifestly **keeps** its axis — the
   shipped layout is :math:`(1, *\mathrm{spatial})`, not
   :math:`(*\mathrm{spatial})` — and it *must*: the group-averaged cross
   section :math:`\bar\sigma` is defined only relative to an interval
   and a weighting spectrum, and the Bateman / depletion pairing
   :math:`\langle\bar\sigma, \phi\rangle` consumes exactly that datum.
   Drop the axis and :math:`\bar\sigma` loses the thing that defines it.

   ⟹ **Compactness decides integrate-versus-normalize, at best. It
   cannot decide persist-versus-drop.** Those are two different
   questions, and Version 1 conflated them.


.. _spaces-collapse-version-2:

Version 2 — "energy is effectively finite-measure" (REFUTED)
-------------------------------------------------------------

*Proposed.* Patch energy in as a third case: it is *effectively*
finite-measure — the group structure is topped by a highest energy
:math:`E_0`, and the flux is integrable below it — so it behaves like a
compact domain for integration purposes while still carrying
problem-specific partition data.

.. admonition:: ⛔ The refuting question — what is the measure of :math:`(0, \infty)`?
   :class: error

   The energy domain has **infinite** Lebesgue measure. There is no
   upper limit on neutron energy; the grid top :math:`E_0` is a
   *library's* practicality, not a fact about the space. Calling energy
   "effectively finite-measure" smuggles a data-layer truncation into a
   space-layer definition, and it would make the doctrine depend on
   which library you loaded.

   The correct question is not about the domain at all. It is:
   **does the integral of that infinite tail CONVERGE?** — that is,
   is the admissible field class in :math:`L^1` over the collapsed
   domain?

   ⟹ **The discriminator is a property of the FIELDS the physics
   admits, never of the bare domain.** That reformulation is what
   unlocked the standing doctrine, and it is why the doctrine below
   talks about an *admissible field class* rather than about measures of
   sets.


.. _spaces-collapse-doctrine-standing:

The standing doctrine: two forks
---------------------------------

Version 1's real error was answering one question when there are two.
The doctrine separates them.

**Fork 1 — does the collapse INTEGRATE or NORMALIZE?** Decided by the
integrability of the **admissible field class** over the collapsed
domain. The mechanism is structural, and it runs through symmetry:

- A quotient **by a symmetry group** forces the fields to be CONSTANT
  along group orbits. If the group is **non-compact** (translations of
  :math:`\mathbb{R}^d`), its orbits carry infinite Haar measure, and a
  nonzero constant on infinite measure is never :math:`L^1`. Integration
  is impossible *structurally* — whatever the physics — and the only
  surviving functional is the normalized average, "per unit orbit
  measure".
- If the group is **compact** (rotations; :math:`S^2` is an orbit), Haar
  measure is finite, integration is always available, and the total
  (:math:`4\pi`) is canonical.
- With **no symmetry acting** — energy is the case — nothing forces
  constancy, and the *physics* decides integrability. For neutron
  spectra it holds: the fission spectrum :math:`\chi` decays
  super-exponentially above source energies and the thermal Maxwellian
  :math:`\to 0` as :math:`E \to 0`, so the tail integral converges. The
  practical grid top :math:`E_0` is the library's assertion that the
  neglected tail is below tolerance — a data-layer truncation
  statement, not a space-layer fact.

**Fork 2 — does the AXIS PERSIST?** Decided by **consultation**: the
axis survives iff the collapse leaves data that the surviving family
still consults — in its own pairing, or in its identity and guards. A
collapse that leaves only *re-embedding* conventions puts them on the
:math:`(\pi, \iota)` arrows and drops the axis; the family itself
changes.

Three clauses cover every collapse in the corpus:

.. list-table:: The three clauses
   :header-rows: 1
   :widths: 6 30 34 30

   * - #
     - Situation
     - Verdict
     - Instance
   * - **1**
     - collapse along **non-compact group orbits**
     - **NORMALIZE**; the axis **persists** as the quotient point with
       unit weight — the density convention is consulted by the
       member's OWN pairing
     - the homogeneous spatial slot (translations of
       :math:`\mathbb{R}^d`). In the modal branch the surviving datum is
       the mode parameter — buckling's :math:`B`.
   * - **2**
     - **partition-integration** of an :math:`L^1` class, no symmetry
       acting
     - **INTEGRATE** per cell; the **nodal mesh-axis persists** all the
       way down to its one-cell member — the partition (boundaries,
       weighting spectrum) is problem data consulted by identity, by
       guards and by the :math:`V`/:math:`V^*` pairing
     - energy. The one-group member is the one-CELL energy mesh, and
       :math:`\langle\bar\sigma, \phi\rangle` is exactly what consumes
       its edges. Likewise a genuine one-cell slab keeps its axis with
       weight :math:`V_{\rm cell} \neq 1`.
   * - **3**
     - **whole-domain integration** over a **compact canonical orbit**
     - the axis **DROPS** — the total is universal, so nothing
       problem-specific survives on the axis; the rebroadcast convention
       lives on the embedding :math:`\iota`, consulted only when LEAVING
       the family
     - angle. :math:`S^2` is a compact-group orbit and :math:`4\pi` is
       universal, so scalar spaces carry NO angular slot and are a
       different family.

.. admonition:: The two one-line tests
   :class: tip

   #. *Can the admissible fields be integrated over the collapsed
      domain?* — symmetry-forced constancy on infinite-measure orbits
      ⟹ **no** ⟹ normalize.
   #. *Is the surviving convention consulted INSIDE the family, or only
      at re-embedding?* — inside ⟹ **axis**; only at re-embedding ⟹
      **arrow**.

Notice what the doctrine does to the tension it was built to settle: it
does not pick a winner. **The retract rule and the quotient rule are
both right, about different clauses** — clause 3 is the retract, clause
1 is the quotient — and what was missing was the second fork, which
neither rule states.


.. _spaces-collapse-retrodictions:

Retrodictions: the doctrine against the shipped tree
------------------------------------------------------

A doctrine invented to settle one dispute is worth little if it only
settles that dispute. The rows below are layouts the doctrine was NOT
built from, and the clause column is its *prediction* of each.

⚠ **One row is not a retrodiction — read the third column.** The
buckling member is a design conclusion of the same dialogue, not
something the tree ships; it is listed because the doctrine's clause-1
modal branch is what predicts its shape, and a table of confirmations
that quietly includes an aspiration is the exact defect this corpus
otherwise catches in plans.

.. list-table::
   :header-rows: 1
   :widths: 62 14 24

   * - Fact
     - Clause
     - Status
   * - the homogeneous shape :math:`(n_g, 1)` — the spatial point is
       PRESENT, and its unit volume is consumed by the reaction-rate
       functionals inside the solve
     - 1
     - `[M]` **ships**
   * - the scalar family :math:`(n_g, *\mathrm{spatial})` — **no**
       angular slot; the :math:`4\pi` conventions live at the embeddings
     - 3
     - `[M]` **ships**
   * - the one-group layout :math:`(1, *\mathrm{spatial})` — the energy
       axis persists, with its edges
     - 2
     - `[M]` **ships**
   * - one-cell meshes keep weight :math:`V_{\rm cell} \neq 1`, and are
       therefore NOT the quotient point
     - 2
     - `[M]` **ships** (gated; see below)
   * - partial currents: a hemisphere collapse parameterized by
       :math:`\hat n` — the parameter survives on the FACE structure,
       the angular content on the arrow
     - 3 (+ a face axis)
     - `[M]` **ships**
   * - the buckling member: a size-1 **MODAL** spatial axis carrying
       :math:`B`, live angle, field :math:`\mathbb{F} = \mathbb{C}`
     - 1 (modal branch)
     - ⛔ **NOT built** — a prediction (campaign 2)

The fourth row is the one worth pausing on, because it is where the
doctrine becomes *mechanically* enforced rather than merely stated. A
quotient point and a genuine one-cell mesh have the same shape and
differ only in measure — the measure is part of axis identity, and since
the identity flip (structural ``__eq__``, CS4c step 6, 2026-09-07) an
axis-built space IS its axis tuple, so they are **unequal spaces**.
Until the flip the same conclusion arrived one step later, through the
axis-derived name (:ref:`spaces-identity-bridge`). Nothing else in the
tree can see the difference: `[M]` a scalar metric commutes with every
operator, so
:math:`A^\dagger = G^{-1}A^{\mathsf T}G = A^{\mathsf T}` whenever
:math:`G = cI` — the quotient-vs-one-cell distinction is provably
invisible to ``.H``, to every norm and to every value gate. **Space
identity is the only instrument that carries it**, which is why the
CS1 battery guards it with an identity gate and pairs that with an
explicit must-stay-green proof that no value gate can.


.. _spaces-quotient-family:

Consequences: the quotient family, and what a degenerate axis stores
---------------------------------------------------------------------

Clause 1 says the homogeneous solver's spatial axis persists. The
obvious follow-up is *so what does it store, and why is that useful?*
The answer is the **density convention** — the quotient's unit
normalization weight, made a first-class, composable object rather than
an unwritten agreement. Three payoffs, in increasing order of how much
they hurt when the convention is implicit:

#. **The pairing consumes it, so rates follow automatically.** If the
   convention ever changes — per lattice cell of volume
   :math:`V_{\rm cell}`
   instead of per unit volume, say — every functional follows by
   arithmetic, because the weight is the thing they integrate against.
   `[M]` **this is live, and it is measurable — since CS4a K2, of the
   SPACE's own measure.**
   :func:`~orpheus.homogeneous.solver.solve_homogeneous_infinite`
   normalises its flux through the space's pairing
   (``space.inner_product(Σx, φ)``), which contracts against the
   posing's per-axis weights — so the quotient point's weight is
   genuinely consumed, not decorative. The separated measurement (each
   measure varied ALONE, 2026-08-21): minting the point with weight 2.0
   moves the flux and both specific rates by exactly the measure ratio
   while :math:`k_\infty` is unchanged, and doubling the CARRIER's cell
   volumes moves **nothing** — the carrier supplies cross sections, not
   the measure. (Until K2 the rates read ``mesh.volume_measure``
   instead: the same experiment then read :math:`0.225` vs :math:`0.450`
   between the quotient carrier, weights ``[1.0]``, and a one-cell slab
   of width 2, weights ``[2.0]`` — a true measurement whose two measures
   were varied TOGETHER, so it could not distinguish which one the rate
   consumed; the pre-K2 value path was in fact bit-identically inert to
   the space weight.)
#. **Family coherence.** The :math:`B = 0` fiber and the :math:`B \neq
   0` members share the slot, so fiberwise machinery — Fourier
   convergence analysis, :math:`\rho(B)` — reads a uniform signature
   across the family instead of special-casing the degenerate member.
#. **Boundary-of-family maps get a home and a Jacobian.** Pushing
   homogenized constants into a meshed context is a map between two
   members of the family; with the measure on the axis, the conversion
   is the **measure ratio** — an object with a name — instead of an
   invisible unit-convention shift. That invisible shift is the
   classical missing-volume-factor bug class in homogenization.

The quotient family, and where the modes live
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The homogeneous solver is the **terminal** member of a family, not an
isolated special case:

- **The infinite-medium member.** Translation invariance quotients the
  spatial axis to the one-point axis with unit measure; isotropy
  quotients angle to the :math:`\ell = 0` retract (clause 3 — the
  angular axis is GONE, not trivial). The spectrum problem lives on the
  energy axis tensored with the quotient point, and everything transfers
  with no special cases: the energy axis is nodal, so the coordinate
  cone, the Hilbert-metric brackets, ray normalization and the
  irreducibility gate all apply verbatim
  (:doc:`/theory/foundations/infinite_medium`).
- **The buckling / B\ :sub:`1` member is the INTERMEDIATE quotient** — a
  one-dimensional **MODAL** spatial axis parameterized by :math:`B`, on
  which streaming is multiplication by :math:`iB\mu`. It is modal
  because a Fourier mode is a coefficient, not a cell value; the field
  is :math:`\mathbb{C}`; and Fourier convergence analysis is exactly
  *the solver diagonalized over this quotient family*, computed
  fiberwise. The :math:`B = 0` fiber is the infinite-medium member, and
  it is the same slot — which is payoff 2 above, stated concretely.
- **P**\ :sub:`N` **is the ANGULAR buckling.** The two hierarchies are
  the same construction on two different groups: irreps of translations
  give the buckling ladder on the spatial axis, irreps of rotations give
  the :math:`P_N` ladder on the angular axis. The parallel also
  *predicts* clause 3's asymmetry: the trivial angular slot stays
  ABSENT on scalar-family members, exactly as the trivial spatial slot
  would if space were compact.

.. note::

   **CS1 does not build the buckling member; it only refuses to
   foreclose it.** The complex field :math:`\mathbb{F} = \mathbb{C}` and
   the modal spatial axis are scheduled work (campaign 2). What CS1
   guarantees is that ``MODAL`` exists as a first-class basis kind from
   day one, so the member can be minted later without re-typing the
   axis, and that the consumers which cannot answer on a modal factor
   (the cone predicate) refuse rather than silently answer.

.. _spaces-symmetry-monotonicity:

The symmetry-monotonicity law
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clause 1 quotients "by the symmetry group" — but *whose* symmetry?
Symmetry lives at the **geometry**, and every arrow down the modelling
lattice at best preserves it and usually reduces it. It never increases
it:

.. math::

   G_{\rm medium} = G_{\rm geom} \cap \operatorname{Stab}(\text{material
   assignment}),
   \qquad
   G_{\rm mesh} = G_{\rm geom} \cap \operatorname{Stab}(\text{cells}),
   \qquad
   G_{\rm pullback} \subseteq G_{\rm medium} \cap G_{\rm mesh}.

A material assignment breaks the geometry's symmetry unless the
assignment is itself invariant; a uniform mesh keeps discrete
translations while an unstructured one keeps nothing.

The consequence for the collapse doctrine is precise and it settles a
"where does this belong?" question that the mesh vocabulary could not
answer: **clause 1's quotient consumes the MEDIUM's surviving symmetry,
not raw geometry's.** The infinite homogeneous medium is exactly the
member whose assignment stabilizer is *everything* — a uniform
assignment breaks nothing — so the full translation group survives to be
quotiented. Buckling then restricts to an irrep of that surviving group.

.. warning::

   **The** ``Medium`` **layer this law argues for is CHARTERED, not
   shipped.** The lattice the law is stated over —
   geometry :math:`\to` {medium, mesh} :math:`\to` the pullback that
   carries materials on cells — is a design conclusion of the same
   dialogue, scheduled as its own phase (CS1.5) between CS1 and CS2.
   Today the homogeneous path still constructs a degenerate
   :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` through
   ``from_materials``, which is the pullback wearing a constructor story
   it does not have a mesh for. Read this subsection as *why the
   quotient is licensed one level above the mesh*, not as a description
   of a class that exists.



.. _spaces-collapse-pair:

The realized machinery: the axis collapse pair
==============================================

The doctrine above decides **which** axes survive a degeneracy. This
section is the machinery that performs the collapse when the verdict is
"drop", and it is where the doctrine stops being a rule and becomes two
typed arrows a call site can hold.

The bridge is the retract rule's own sentence
(:ref:`spaces-collapse-the-question`): *a projection*
:math:`\pi: V \to V'`, *an embedding* :math:`\iota: V' \to V`, *with*
:math:`\pi\circ\iota = \mathrm{id}` *and, under the right metrics,*
:math:`\iota = \pi^{H}` *up to a scalar*. Everything below is that
sentence made precise and made executable:

- :math:`\pi` is :class:`~orpheus.numerics.operator.AxisRetractionOperator`,
  minted by :meth:`FunctionSpace.retraction
  <orpheus.numerics.space.FunctionSpace.retraction>`;
- :math:`\iota` is :class:`~orpheus.numerics.operator.AxisSectionOperator`,
  minted by :meth:`FunctionSpace.section
  <orpheus.numerics.space.FunctionSpace.section>`;
- **the scalar is** :math:`\Sigma w`, the axis's total mass, and it is
  not a convention: it is the :math:`1\times 1` Gram of the rank-one
  frame that mints the pair (:ref:`spaces-collapse-pair-frame`);
- the two are **different types** precisely so that the "up to a
  scalar" cannot be silently dropped at a call site — which is the
  ERR-051 failure class made unspellable.

.. code-block:: python

   from orpheus.numerics.axis import Axis, BasisKind
   from orpheus.numerics.space import FunctionSpace

   V = FunctionSpace.of_axes(
       Axis("angular", (4,), weights=w, kind=BasisKind.NODAL),
       Axis("energy",  (2,),            kind=BasisKind.NODAL),
       Axis("spatial", (5,), weights=V_cell, kind=BasisKind.NODAL),
   )
   R = V.retraction("angular")   # V -> energy (x) spatial   (fiber integration)
   E = V.section("angular")      # energy (x) spatial -> V   (the section)

   R.apply(E.apply(phi))         # == phi
   R.H.apply(phi)                # the plain broadcast — NOT E

Both verbs are **memoized on the space**, one mint per space per axis
label, so a carrier that caches its spaces (every solver carrier does)
gets warm operators for free: ``sn.angular_bulk_space.retraction("angular")``
builds the generator once and returns the same object thereafter
(`[M]` identity, gated).


.. _spaces-collapse-pair-two-arrows:

Two arrows, not one — and the scalar between them
--------------------------------------------------

Write :math:`V = V_{\rm ax} \otimes V'` for the product whose first
factor is the axis being collapsed, with the product metric
:math:`G_V = \operatorname{diag}(w) \otimes G_{V'}`
(:eq:`spaces-axis-product`). The **retraction** is fiber integration
over that factor — the pushforward :math:`\pi_*` along the projection
that forgets the axis:

.. math::
   :label: spaces-collapse-retraction

   (R\,\psi)(\cdot) \;=\; \sum_{n} w_n\,\psi(n,\cdot),
   \qquad
   R : V \longrightarrow V' .

.. (vv-status rationale) Structural/representational identity: it STATES
   what ``AxisRetractionOperator.apply`` computes (the axis's factor
   measure contracted over the axis's ndarray dims). Not a solver claim
   — no flux, no eigenvalue, no discretization error. The verifiable
   content is the CS4b S6 foundation battery
   (``tests/numerics/test_axis_marginal.py``): the tightness row pins
   this contraction against the mint frame's own analysis content, and
   G6.5 pins it bit-identically against a hand-spelled einsum on the
   real S\ :sub:`N` carrier.
.. vv-status: spaces-collapse-retraction documented

.. implements:: spaces-collapse-retraction
   :by: py:method:orpheus.numerics.operator.AxisRetractionOperator.apply

   **Implemented by** 2 sites. The kernel is the operator's; the
   canonical field-level consumer re-spells nothing — since CS4b S6.2
   :meth:`AngularField._integrate_angular_values
   <orpheus.transport.fields._bases.AngularField._integrate_angular_values>`
   IS this ``apply``, so the tree has one realization of the angular
   reduction and not two.

.. implements:: spaces-collapse-retraction
   :by: py:method:orpheus.transport.fields._bases.AngularField._integrate_angular_values

Its Hilbert adjoint is where the two-arrow discipline is forced. Under
the product metric the axis weights **cancel exactly**:

.. math::
   :label: spaces-collapse-adjoint-is-pullback

   R^{\dagger}
   \;=\; G_V^{-1}\,R^{\mathsf T}\,G_{V'}
   \;=\; \bigl(\operatorname{diag}(w)\otimes G_{V'}\bigr)^{-1}
         \bigl(\operatorname{diag}(w)\otimes G_{V'}\bigr)\,\pi^{*}
   \;=\; \pi^{*},

.. (vv-status rationale) Structural identity of the metric sandwich: the
   Euclidean transpose of a weighted contraction is the WEIGHTED
   scatter, and the product metric's own axis block is exactly what
   removes those weights again — so the Hilbert adjoint of fiber
   integration is the UNWEIGHTED broadcast. Representational, not a
   solver claim (no physics enters; it is true for every axis measure).
   The verifiable content is the CS4b S6 foundation battery's G6.3 row
   in ``tests/numerics/test_axis_marginal.py``, which pins the
   adjunction on the physical metrics and carries the vv #19 NEGATIVE
   leg (the same pairing under a deliberately stripped spatial measure
   must break at O(1)).
.. vv-status: spaces-collapse-adjoint-is-pullback documented

.. no-implementation:: spaces-collapse-adjoint-is-pullback
   :kind: identity

   **Nothing implements this.** It is an identity between two things
   that are each computed elsewhere and never equated in production:
   the left side is produced by the generic metric-aware adjoint
   wrapper (``R.H``, which knows nothing about axes), the right side by
   :meth:`AxisSectionOperator.apply
   <orpheus.numerics.operator.AxisSectionOperator.apply>` scaled by
   :math:`\Sigma w`. No line forms the comparison — that is the point:
   the cancellation is what lets the adjoint be free rather than
   bespoke. It is *measured* by the G6.4 row of
   ``tests/numerics/test_axis_marginal.py``.

where :math:`\pi^{*}` is the **pullback** — the plain, unweighted
broadcast of :math:`\varphi` across the axis. So
:math:`(R, R^{\dagger}) = (\pi_*, \pi^{*})` is the discrete realization
of the fiber-integration / pullback adjunction, and the pairing

.. math::

   \langle R\psi,\ \varphi\rangle_{V'}
   \;=\;
   \langle \psi,\ \pi^{*}\varphi\rangle_{V}

holds on the *physical* metrics with no correction factor at all
(`[M]` bounded by :math:`3.7\times10^{-13}` relative over 200 draws on
the shipped S\ :sub:`N` carrier and :math:`1.6\times10^{-13}` on the
synthetic three-axis fixture — a scalar-valued identity whose relative
residual is set by cancellation in the inner products, not by the
operators; see :ref:`spaces-collapse-pair-evidence`).

**The pullback is not the section.** :math:`\pi^{*}` broadcasts
:math:`\varphi` unchanged, so :math:`R\pi^{*}\varphi = (\Sigma w)\varphi`
— it is a right inverse of :math:`R` only after division by the axis's
total mass. That division is the second arrow:

.. math::
   :label: spaces-collapse-section

   (E\,\varphi)(n,\cdot) \;=\; \frac{\varphi(\cdot)}{\Sigma w},
   \qquad
   E \;=\; \frac{\pi^{*}}{\Sigma w}
   \;=\; R^{\dagger}\,(R\,R^{\dagger})^{-1},
   \qquad
   R\circ E \;=\; \mathrm{id}_{V'} .

.. (vv-status rationale) Structural/representational identity: it STATES
   what ``AxisSectionOperator.apply`` computes (divide by the axis's
   total mass, then broadcast) and identifies it as the Moore-Penrose
   pseudo-inverse of the retraction in the two spaces' own metrics. Not
   a solver claim. The verifiable content is the CS4b S6 foundation
   battery in ``tests/numerics/test_axis_marginal.py``: G6.1 (the
   section law), G6.2 (idempotence of the composite projector), G6.6
   (bit-identity with the shipped isotropic-source kernel on the real
   S\ :sub:`N` carrier) and the gram-derivation row that pins the
   divisor against the mint frame's own Gram entry.
.. vv-status: spaces-collapse-section documented

.. implements:: spaces-collapse-section
   :by: py:method:orpheus.numerics.operator.AxisSectionOperator.apply

   **Implemented by** 3 sites. The kernel is the operator's; the mint
   supplies the divisor from the frame's Gram, and the canonical
   producer-side consumer re-spells nothing — since CS4b S6.2
   :meth:`AngularSourceSink.from_isotropic
   <orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink.from_isotropic>`
   IS this ``apply``.

.. implements:: spaces-collapse-section
   :by: py:function:orpheus.numerics.frame._collapse_pair

.. implements:: spaces-collapse-section
   :by: py:method:orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink.from_isotropic

Since :math:`R` is rank-deficient by construction, *many* right inverses
exist — putting all the mass on one ordinate, :math:`(\iota_0
\varphi)(n,\cdot) = \delta_{n0}\,\varphi/w_0`, is one. :math:`E` is
distinguished among them by being the **minimum-norm** one, which is
what :math:`R^{\dagger}(RR^{\dagger})^{-1}` says: it is the
Moore–Penrose pseudo-inverse in the two spaces' own metrics. `[M]` on
the synthetic fixture and one draw (``default_rng(21)``),
:math:`\lVert E\varphi\rVert_V = 1.700` against
:math:`\lVert \iota_0\varphi\rVert_V = 4.389` for the one-ordinate
right inverse — a factor 2.6 — with both satisfying
:math:`R\circ(\cdot) = \mathrm{id}` at the round-off floor.

The four arrows close into a square, and every entry is one of the two
operators or one of their adjoints:

.. list-table:: The collapse square (:math:`W \equiv \Sigma w`)
   :header-rows: 1
   :widths: 22 30 48

   * - Arrow
     - What it is
     - Identity
   * - :math:`R = \pi_*`
     - fiber integration (the weighted axis sum)
     - :math:`R\,R^{\dagger} = W\,\mathrm{id}`
   * - :math:`R^{\dagger} = \pi^{*}`
     - the pullback (the **plain** broadcast)
     - :math:`R^{\dagger} = W\,E` — the anti-ERR-051 scalar
   * - :math:`E`
     - the section (broadcast **after** dividing by :math:`W`)
     - :math:`R\circ E = \mathrm{id}`, and :math:`E` is minimum-norm
   * - :math:`E^{\dagger}`
     - the :math:`w`-mean (average over the axis)
     - :math:`E^{\dagger} = R/W`, and :math:`E^{\dagger}\circ R^{\dagger} = \mathrm{id}`

and the composite in the other order,

.. math::

   P \;\equiv\; E\circ R,
   \qquad
   (P\psi)(n,\cdot) \;=\; \frac{1}{\Sigma w}\sum_m w_m\,\psi(m,\cdot),

is the **conditional expectation onto axis-constant functions** — the
:math:`w`-mean projector. It is idempotent and, because
:math:`P^{\dagger} = R^{\dagger}E^{\dagger} = (WE)(R/W) = P`,
self-adjoint in :math:`G_V`: an *orthogonal* projector, not merely an
oblique one (`[M]` self-adjointness bounded by
:math:`4.0\times10^{-14}` relative over 200 draws — again a
cancellation-limited scalar identity, see
:ref:`spaces-collapse-pair-evidence`).
That is the precise sense in which "the isotropic part of
:math:`\psi`" is a well-defined object rather than a convention.

.. warning::

   **An axis whose SIGNED measure sums to zero has NO section, and the
   retraction over it is still legal.** The asymmetry is structural, not
   defensive: :math:`R` is a contraction and needs no division, while
   :math:`E` divides by :math:`\Sigma w` — and at :math:`\Sigma w = 0`
   the rank-one Gram is singular, so the mint frame has no canonical
   dual and no section EXISTS to hand back. The mint therefore leaves
   that arm unminted and :meth:`FunctionSpace.section
   <orpheus.numerics.space.FunctionSpace.section>` refuses at access,
   naming the cause. Signed axis weights are deliberately legal on
   :class:`~orpheus.numerics.axis.Axis` (a :math:`\sigma`-folded
   quadrature can carry them), so this is a reachable state and not a
   theoretical one.

.. note::

   **This is where the doctrine's "up to a scalar" gets its value.**
   The retract rule said :math:`\iota = \pi^{H}` *up to a scalar* and
   left the scalar unnamed, which is exactly the gap the classical
   :math:`4\pi` bookkeeping errors live in. Here the scalar is
   :math:`\Sigma w`, it is read off the mint frame's Gram, and the two
   arrows carry different **types** — so a call site that reaches for
   ``R.H`` where it wanted ``E`` does not silently rescale a source by
   :math:`4\pi`; it holds an object of the wrong class. `[M]` the gate
   asserts precisely that ``R.H`` is *not* an
   :class:`~orpheus.numerics.operator.AxisSectionOperator`.


.. _spaces-collapse-pair-naming:

Why "retraction" and "section" — and why "embedding" was rejected
------------------------------------------------------------------

The names are canonical, and the reason is worth stating because the
first spelling shipped and was replaced within a day.

:math:`R\circ E = \mathrm{id}` makes :math:`(R, E)` a **split
epi/mono pair** in the categorical sense (Mac Lane, *Categories for the
Working Mathematician*, §I.5): a morphism with a right inverse is a
**split epimorphism** and the right inverse is called a **section**;
dually the left inverse of a split monomorphism is called a
**retraction**. The collapse doctrine's own prose already used "the
retract rule", so :math:`R` inherited the right word immediately.

The first implementation (S6.0) called the other arrow
``AxisEmbeddingOperator``. That name was retired the same day for two
independent reasons:

#. **It cannot discriminate the pair it exists to discriminate.** Any
   injective structure-preserving map is an embedding — and
   :math:`\pi^{*} = R^{\dagger}` is one too. So "embedding" names a
   property both arrows have, in a design whose entire purpose is to
   keep :math:`R^{\dagger}` and :math:`E` apart. The
   :math:`\Sigma w` weld the ERR-051 class is made of was hiding in
   the *name*.
#. **The object is defined relative to** :math:`R`. :math:`E` is not "a
   map into :math:`V` that happens to be injective"; it is *the* right
   inverse of a specific retraction — :math:`R\circ E = \mathrm{id}`
   IS its definition. The categorical name for that is **section**, and
   nothing weaker carries the relation.

"Embedding" survives in this corpus only as a generic adjective (the
doctrine's :math:`\iota`, the :math:`S^2` embedding of
:doc:`/theory/foundations/spherical_harmonics`). It is not the name of
an operator.

.. note::

   **The naming rule this instantiates** is the same one
   :ref:`the frame hierarchy <frame-discipline-as-a-type>` follows: *a
   reader of a type name knows its properties without reading the
   docstring.* ``AxisSectionOperator`` tells a reader that composing it
   after the retraction is the identity; ``AxisEmbeddingOperator`` told
   them only that it was injective, which is true of the wrong arrow
   too.


.. _spaces-collapse-pair-frame:

The pair is frame-induced — a stage-2 generator at rank one
------------------------------------------------------------

The kernels above could be written by hand: an einsum and a broadcast,
six lines each. They are not, and the reason is Cardinal Rule 2. The
hand-written pair is a **concept-level twin** of machinery this corpus
already ships — the discrete frame
(:doc:`/theory/foundations/frame`) — specialized to rank one. Two
places would have had to agree, forever, about what "the axis measure"
means and what the normalization divisor is.

So the pair is *induced*. At the mint site
(:func:`orpheus.numerics.frame._collapse_pair`) a literal frame is
built over the axis's index set, read for its induced data, and
discarded:

.. code-block:: python

   frame = GalerkinFrame(
       basis=IndicatorBasis(
           edges_per_axis=(np.array([-0.5, n - 0.5]),),
           partition_of=IndexSet(label=axis_label, n=n),   # #429 tracker 2.1
       ),
       measure=DiscreteMeasure(
           nodes=np.arange(n, dtype=float),
           weights=flat_weights,              # weights=None IS the counting measure
           support=f"index({axis_label})",
       ),
   )
   kernel_weights = frame.measure.weights          # the analysis face's content
   total_weight   = float(frame.discrete_gram[0, 0])   # the rank-one Parseval metric

The basis is a **single-region indicator** covering every index
:math:`\{0,\dots,n-1\}` — one region, so exactly one coefficient, so a
:math:`1\times1` Gram. Under that basis every table entry is
:math:`1`, and the frame's Gram
(:math:`G_{jk} = \sum_n w_n \phi_j(x_n)\phi_k(x_n)`) collapses to the
axis's total mass:

.. math::
   :label: spaces-collapse-rank-one-gram

   G \;=\; \bigl[\textstyle\sum_n w_n\bigr] \;=\; [\,\Sigma w\,],
   \qquad
   E \;=\; R_{\rm frame}\circ G^{-1},

.. (vv-status rationale) Literature-transcribed / structural: it states
   the rank-one instance of the Parseval theorem already derived at
   :ref:`frame-parseval-metric` (the frame's codomain metric is the
   INVERSE discrete Gram), specialized to a single-region indicator
   basis where the Gram is 1x1 and its entry is the measure's total
   mass. Not a solver claim. The verifiable content is the
   gram-derivation row of ``tests/numerics/test_axis_marginal.py``,
   which asserts the section's divisor IS the literal frame's
   ``discrete_gram[0, 0]``, and the tightness row, which pins the
   minted kernels against that frame's own face contents.
.. vv-status: spaces-collapse-rank-one-gram documented

.. implements:: spaces-collapse-rank-one-gram
   :by: py:function:orpheus.numerics.frame._collapse_pair

   **Implemented by** 1 site. The mint is where the identity is
   *used* — it reads ``frame.discrete_gram[0, 0]`` and stores it as the
   section's divisor. The Gram itself is the frame's generic
   :math:`O(NK^2)` einsum, not a rank-one special case, which is the
   whole point: nothing in the collapse pair re-derives what a frame
   already computes.

so the section's divisor **is** the Parseval metric of
:ref:`frame-parseval-metric` at :math:`K = 1`. "Divide by
:math:`4\pi`" is therefore not a convention this code chose; it is the
inverse Gram of a frame, obtained the same way every other metric in
the corpus is obtained.

The frame is discarded on return. That is deliberate, and it is the
**stage-2 generator discipline** — the ruling this section exists to
realize (user, 2026-08-24):

   A stage-2 generator induces structure on both the space and the
   operator, and the two inductions must be minted together, at one
   site. Frame: induces the HarmonicAxis metric (space side), mints
   Analysis/Synthesis (operator side) — consistency is the tightness
   gate. Scheme: induces the trace descriptor and basis kind (space
   side), mints the closure (operator side) — consistency is one
   closure serving both apply and solve, which is ERR-026's structural
   closing. Mesh and Quadrature are the degenerate cases (space side
   only). Forgetting = retaining the induced parts; accessors are
   provenance.

Three consequences, each visible in the code:

**Both inductions at one site.** The mint constructs the retraction and
the section together and returns them as a pair. There is no path that
produces one without the other, so they cannot disagree about the axis,
the dims, or the marginal space — `[M]` the two arrows share ONE
marginal-space instance (``R.codomain is E.domain``), gated.

**Forgetting means copying the induced parts out.** The operators
retain the bound spaces, the ndarray dims the axis occupies, the flat
weights, and the scalar divisor — and nothing else. In particular they
do **not** retain a frame *face*: a face is a view holding
``frame:`` (:class:`~orpheus.numerics.frame.FrameBase`'s
``_FrameAnalysis`` / ``_FrameReconstruction``), so keeping one would
keep the generator alive through it. `[M]` read the face
dataclasses — this is why the mint copies arrays rather than storing
``frame.analysis``.

**Consistency is a gate, not an instance.** Because the generator is
thrown away, nothing at runtime *forces* the minted kernels to agree
with the frame that produced them. The **tightness gate** supplies
that: it rebuilds the literal frame independently and pins all three
correspondences — :math:`R` against the analysis content,
:math:`R^{\mathsf T}` against ``analyze_transpose``, and :math:`E`
against reconstruction composed with :math:`G^{-1}`. `[M]` all three
are bit-exact on **200 of 200** draws, which is a *stronger* statement
than the section law two sections up: there the exactness is a property
of the draw, here it is a property of the construction, because the two
sides evaluate the same reduction in the same order. The operator
kernels are hand einsums and the frame path runs the basis's table
einsums, so these are two different float programs and agreement is a
real claim, not a tautology.

.. note::

   **The latent generalization, recorded and not built.** The mint is
   parameterized by exactly one choice — the basis. Swap the
   single-region ``IndicatorBasis`` for a single-region
   ``WeightedIndicatorBasis`` and the same site produces a *profiled*
   collapse: the Petrov-Galerkin test side of a
   :math:`\chi`-class emission collapse, where the axis is not
   averaged uniformly but against a spectrum. That is the same
   machinery, a different basis, and it is built when its consumer
   lands (CS4c) — not before. Recording it here is what stops a future
   session minting a second, parallel mechanism for it.


.. _spaces-collapse-pair-clause-gate:

The clause gate: which axes admit, and why energy refuses
-----------------------------------------------------------

The mint refuses an axis the doctrine says must persist. This is the
one place in the tree where the collapse doctrine is **enforced** rather
than described, so it is worth reading the admission table against
:ref:`spaces-collapse-doctrine-standing` clause by clause.

.. list-table:: Admission at the mint
   :header-rows: 1
   :widths: 26 12 62

   * - Axis
     - Clause
     - Verdict, and why
   * - **angle** (an untyped ``Axis`` today)
     - 3
     - **ADMIT.** Whole-domain integration over a compact canonical
       orbit: the total is universal, nothing problem-specific survives
       on the axis, so the drop-form marginal is exactly right and the
       re-broadcast convention lives on the arrows :math:`E` /
       :math:`\pi^{*}`.
   * - a typed :class:`~orpheus.numerics.axis.EnergyAxis`
     - 2
     - **REFUSE**, with a pointer. Partition-integration of an
       :math:`L^1` class: the energy axis PERSISTS at its one-cell
       member, because :math:`\langle\bar\sigma,\varphi\rangle` consumes
       the partition. A drop-form marginal here would be a second
       mechanism for a collapse the tree already implements as
       **condensation**.
   * - a ``MODAL`` axis
     - —
     - **REFUSE.** Contracting expansion COEFFICIENTS with the basis
       mass is not an integral of the represented function. The modal
       average is the coefficient at the average slot; slice it.
   * - an untyped generic axis, whatever its label
     - 3
     - **ADMIT.** The gate reads the axis's TYPE, never its label
       string.
   * - the only axis of a single-axis space
     - —
     - **REFUSE.** Its marginal would be a bare scalar, which is not a
       :class:`~orpheus.numerics.space.FunctionSpace`. Contract with
       the space's inner product instead.
   * - a space with ``axes is None``
     - —
     - **REFUSE.** An axes-less space — a hand-named legacy space, or a
       product with an axes-less factor — has no named factors to
       marginalise over. (This row read *"a densified legacy product"*
       until 2026-09-07; item 6.2a retired the densification, not the
       refusal, which stays reachable for every space that declares no
       axes.)

The energy row is the load-bearing one, and it is the doctrine's
Version-1 refutation cashed out in code
(:ref:`spaces-collapse-version-1`). Energy is collapsed *by
integration*, so a compactness-flavoured reading puts it on the "drop"
side; the shipped one-group layout is :math:`(1, *\mathrm{spatial})`
and keeps its axis, because :math:`\bar\sigma` is defined only relative
to an interval and a weighting spectrum. The machinery that performs an
energy collapse correctly therefore already exists — it is
:meth:`EnergyGrid.overlap_to
<orpheus.data.energy_grid.EnergyGrid.overlap_to>` and the
Petrov-Galerkin condensation frames of
:ref:`sn-energy-condensation` — and the refusal message names it, so
a caller who reaches for the wrong tool is handed the right one rather
than a wrong answer.

The clause gate is not hypothetical on a shipped carrier. `[M]` on the
:math:`S_N` fixture above, whose ``angular_bulk_space`` axes are
``(Axis, EnergyAxis, Axis)``:

.. code-block:: text

   sn.angular_bulk_space.retraction("angular")  -> OK, codomain (2, 5)
   sn.angular_bulk_space.retraction("spatial")  -> OK, codomain (4, 2)
   sn.angular_bulk_space.retraction("energy")   -> TypeError: ... is a typed
       EnergyAxis, which PERSISTS at its one-cell member (collapse doctrine
       clause 2 ...). The energy collapse is condensation: use
       EnergyGrid.overlap_to / the Petrov-Galerkin condensation frame, not a
       drop-form marginal.

Two of the carrier's three factors marginalise; the one the doctrine
says must persist refuses, by TYPE, with the successor named in the
message.

.. warning::

   **The gate reads the TYPE, and today only energy has one.** A
   generic ``Axis(label="energy", ...)`` — a synthetic test factor — is
   ADMITTED, and correctly so: refusing on the label string would be
   stringly-typed dispatch, and it would refuse a legitimate synthetic
   fixture that carries none of energy's physics. The consequence is
   that the clause gate is **structural for energy and permissive for
   everything else** until CS2 lands the typed spatial / quadrature /
   harmonic axes; at that point the verdict becomes axis-family
   polymorphism and each family answers for itself. Until then, do not
   read "the mint admitted it" as "the doctrine says it drops".


.. _spaces-collapse-pair-refuted:

What was tried, and what refuted it
------------------------------------

**The hand-derived pair (S6.0) — superseded within a day.** The first
implementation minted the two operators from the space with
hand-spelled kernels and a hand-chosen divisor
(``weights.sum()``). It was correct and it was a twin: the design
dialogue that followed established the exact correspondence with the
rank-one frame, which is Cardinal Rule 2's stop condition. The
re-carve (S6.0b) kept the operator shells verbatim — admission, dims
bookkeeping, bound spaces, the einsum/broadcast kernels — and changed
only where the two retained numbers come from. That is why the re-carve
is bit-identity-safe by construction and why the equivalence gates
survive it unchanged.

**An** ``Axis`` **→ measure accessor — refused.** The mint needs a
:class:`~orpheus.numerics.measure.DiscreteMeasure` built from the axis.
The obvious move is to give :class:`~orpheus.numerics.axis.Axis` a
public accessor that produces one. It was rejected under the same
ruling that governs the frame: *accessors are provenance*. An accessor
would make the generator reachable from the axis forever, so the axis
would carry a permanent dependence on frame machinery it does not
need. The mint builds the measure with a **local** helper instead; the
axis stayed four slots and nothing more.

.. note::

   **Read with CS5 (2026-08-29), which added a fifth slot and did NOT
   overturn this.** The paragraph above is preserved as written; only
   its closing tense moved, because
   :attr:`~orpheus.numerics.axis.Axis.generator` now exists. The two are
   compatible, and the reason is the arrow's direction. What S6.0b
   refused points **axis → measure** and would have had to *manufacture*
   its output — a pre-CS5 axis had dropped the nodes, so the only node
   set it could synthesise is the index set, which is exactly what the
   local helper builds (``nodes = arange(n)``,
   ``support = f"index({label})"``, :ref:`spaces-collapse-pair-frame`).
   Publishing that as an accessor would have handed readers a
   **synthetic** measure under a name they would take for the generating
   one. CS5 points **generator → axis** and manufactures nothing: it
   records the real object at the one moment it is in scope, the mint.

   The collapse pair is unchanged by it, deliberately. Its rank-one
   indicator frame is built over the axis's **index set** — a marginal
   over an axis is a sum over its indices — not over the generator's
   physical nodes, and that site still builds its own measure rather
   than consulting ``axis.generator``. Both halves are worth stating
   together, because "the axis can now reach a measure" invites exactly
   the wrong inference here. Full treatment:
   :ref:`spaces-generator-not-a-reverse-accessor`.

**Caching the pair on the** ``Axis`` **— refused for the same reason,
and it would have been wrong anyway.** The pair is not a function of
the axis alone: its domain is the whole product and its codomain is the
product minus that factor, so two spaces sharing an axis have
*different* collapse pairs. The cache belongs to the space, and it is
there (memoized in the frozen dataclass's ``__dict__``, one entry per
axis label).

**Retaining the frame on the operators — refused.** Keeping the frame
would make consistency automatic instead of gated, which sounds
strictly better. It is not: it retains a whole generator (basis, table,
measure, two spaces) on every collapse operator in the tree to secure a
property that a two-line gate already secures, and it re-opens the
question of whether two operators built from *equal* frames are the
same operator. Consistency here is carried by content-determinism plus
the tightness gate, not by instance sharing. The rule that would flip
this: a second consumer needing the *identical frame instance* for
measure-consistency or anti-aliasing. None exists today.


.. _spaces-collapse-pair-evidence:

Numerical evidence
-------------------

`[M]` 2026-08-24, measured against the tree at HEAD. **The construction,
so the tables regenerate from this page.** The *synthetic* fixture is
the three-axis product ``angular(4, w=[0.3, 0.7, 0.5, 0.5]) ⊗
energy(2, counting) ⊗ spatial(5, V=[0.2, 0.3, 0.4, 0.7, 1.4])`` built
with :meth:`FunctionSpace.of_axes
<orpheus.numerics.space.FunctionSpace.of_axes>`; the weights are
non-uniform on purpose so that no cancellation flatters a law. The
:math:`S_N` fixture is the shipped carrier —
``SNMesh(Mesh1D(edges=[0, 0.2, 0.5, 0.9, 1.6, 3.0], cartesian, vacuum),
Quadrature.gauss_legendre(4), 2 groups)`` — and the operators come from
``sn.angular_bulk_space.retraction("angular")`` / ``.section("angular")``.
Inputs are ``numpy.random.default_rng(seed).standard_normal(shape)``.

Every entry below is a **bound over 200 independent draws**, not a
single reading: the residual of an exact-in-real-arithmetic identity is
a property of the numbers that happen to be involved, so one seed's
value is not reusable (:math:`\max_k \lVert a-b\rVert_\infty /
\lVert b\rVert_\infty` over ``default_rng(1000+k)``,
:math:`k = 0..199`).

.. list-table:: The square, measured (bound over 200 draws)
   :header-rows: 1
   :widths: 40 30 30

   * - Identity
     - synthetic 3-axis
     - :math:`S_N` carrier (GL4 slab)
   * - :math:`R\circ E = \mathrm{id}`
     - :math:`1.5\times10^{-16}`; ``array_equal`` on **123 of 200**
     - :math:`0.0` — ``array_equal`` on **200 of 200**
   * - :math:`P = E\circ R` idempotent
     - :math:`1.5\times10^{-16}`; ``array_equal`` on **130 of 200**
     - :math:`0.0` — ``array_equal`` on **200 of 200**
   * - :math:`R^{\dagger} = \pi^{*}` (the plain broadcast)
     - :math:`2.4\times10^{-16}`
     - :math:`2.3\times10^{-16}`
   * - :math:`R^{\dagger} = (\Sigma w)\,E`
     - :math:`2.4\times10^{-16}`
     - :math:`2.3\times10^{-16}`
   * - :math:`E^{\dagger} = R/\Sigma w`
     - :math:`3.6\times10^{-16}`
     - :math:`3.6\times10^{-16}`
   * - :math:`R\,R^{\dagger} = (\Sigma w)\,\mathrm{id}`
     - :math:`2.2\times10^{-16}`
     - :math:`2.2\times10^{-16}`
   * - the adjunction :math:`\langle R\psi,\varphi\rangle_{V'} = \langle\psi,\pi^{*}\varphi\rangle_V`
     - :math:`1.6\times10^{-13}`
     - :math:`3.7\times10^{-13}`
   * - :math:`P` self-adjoint in :math:`G_V`
     - :math:`4.0\times10^{-14}`
     - :math:`2.1\times10^{-14}`
   * - :math:`R \equiv` the shipped angular reduction
     - —
     - **bit-exact** (``np.array_equal``)
   * - :math:`E \equiv` the shipped isotropic-source kernel
     - —
     - **bit-exact** (``np.array_equal``)

⚠ The two **pairing** rows sit three orders above the others, and that
is arithmetic rather than a defect: an inner product of two random
fields can very nearly cancel, so the *relative* residual of a
scalar-valued identity is bounded by the conditioning of that
cancellation, not by the operators. Their gate row is written at
``rtol=1e-13`` for exactly this reason, and a *tighter* tolerance there
would be a latent false red (``vv-principles`` #16).

.. warning::

   **Do not read "bit-exact" as a law — it is a property of the draw.**
   :math:`R\circ E = \mathrm{id}` is exact in real arithmetic and holds
   at the round-off floor in IEEE-754; whether the floor is *zero*
   depends on how :math:`\sum_n w_n(\varphi/\Sigma w)` happens to
   re-associate for the particular numbers involved. `[M]` on the
   synthetic fixture — the one the gate uses — ``np.array_equal``
   FAILS on **844 of 2000** seeds (worst relative deviation
   :math:`1.5\times10^{-16}`, i.e. about one ULP), and the idempotence
   row fails on **57 of 200**. The originally-shipped G6.1/G6.2 rows
   pinned ``array_equal`` on seeds that happened to land in the exact
   set — seed-fragile — and were re-pinned at ``nulp=1`` the same day
   this audit measured the fragility (their docstrings carry the
   sweep). On the shipped :math:`S_N` carrier the identity
   is bit-exact on 200 of 200 seeds — there :math:`\Sigma w = 2`
   exactly *and* the symmetric Gauss–Legendre weights re-associate
   cleanly — which is why the production-facing rows can be pinned at
   ``np.array_equal`` honestly. Two consequences worth carrying: a
   fixture whose weights are chosen "non-uniform so no cancellation
   flatters a law" buys angular discrimination at the cost of exact
   re-association, so the general tier there is
   ``assert_array_almost_equal_nulp(..., nulp=4)``; and a multi-dim
   axis is one ULP by construction, because a flattened 2-D measure
   sums more terms.

**The divisor is the Gram entry, and that is stronger than
"the divisor is** ``weights.sum()``\ **".** The two agree on most
fixtures and not on all — the Gram is an einsum reduction
(``einsum("n,nj,nk->jk", w, table, table)``) and ``ndarray.sum`` is a
pairwise reduction, so they can differ by a ULP:

.. list-table:: Divisor vs. the naive total, on shipped quadratures
   :header-rows: 1
   :widths: 16 32 32 20

   * - rule
     - divisor (frame Gram entry)
     - ``quad.weights.sum()``
     - identical?
   * - ``gauss_legendre(4)``
     - ``2.0``
     - ``2.0``
     - yes
   * - ``gauss_legendre(8)``
     - ``1.9999999999999998``
     - ``2.0``
     - **no** (1 ULP — ``nextafter(2.0, 0)``; the gate's GL8 row pins
       the bound at ``nulp=1``)
   * - ``gauss_legendre(16)``
     - ``2.0``
     - ``2.0``
     - yes
   * - ``gauss_legendre(32)``
     - ``2.0``
     - ``2.0``
     - yes
   * - ``gauss_legendre(64)``
     - ``2.0000000000000004``
     - ``2.0000000000000004``
     - yes

At ``gauss_legendre(8)`` the shipped
:meth:`AngularSourceSink.from_isotropic
<orpheus.transport.source_sinks.angular_source_sink.AngularSourceSink.from_isotropic>`
therefore differs from a hand-written :math:`Q/\Sigma w` by
:math:`2.0\times10^{-16}` relative — one ULP, and the *induced* value
is the principled one. The lesson for a future gate: pin the divisor
against ``frame.discrete_gram[0, 0]`` (exact, always), never against
``weights.sum()`` (a fixture-dependent coincidence).

**The section is the harmonic frame's isotropic column — at the
measure level.** A natural question is whether the collapse pair
duplicates the spherical-harmonic frame's :math:`\ell = 0` channel.
Measured on the :math:`S_N` carrier above with
``HarmonicFrame.from_galerkin(sn.quad.angular_frame(L))``, feeding a
moment field that is zero except at :math:`(\ell, m) = (0,0)`:

.. list-table:: The isotropic column against the section
   :header-rows: 1
   :widths: 26 18 28 28

   * - frame
     - Gram max off-diagonal
     - :math:`\text{face}^{\dagger}(e_0\varphi)` vs :math:`E\varphi`
     - :math:`\text{reconstruction}(e_0\varphi)/W` vs :math:`E\varphi`
   * - slab, :math:`L=1`
     - :math:`5.6\times10^{-17}`
     - :math:`5.6\times10^{-17}`
     - **0.0** (``array_equal``)
   * - sphere, :math:`L=1`
     - :math:`5.6\times10^{-17}`
     - :math:`1.1\times10^{-16}`
     - **0.0** (``array_equal``)
   * - slab, :math:`L=2`
     - :math:`1.155`
     - :math:`16.17`
     - **0.0** (``array_equal``)
   * - sphere, :math:`L=2`
     - :math:`1.155`
     - :math:`16.17`
     - **0.0** (``array_equal``)

Two readings, and the second is the one to carry:

#. The **adjoint** correspondence — the section is the isotropic column
   of the harmonic frame's *physical* adjoint — holds exactly when the
   measured Gram is DIAGONAL, i.e. when the Parseval metric exists at
   all (:ref:`frame-parseval-metric`). ⚠ The discriminator is the
   **Gram**, not the geometry: the :math:`L=2` rows read identically on
   slab and sphere, because the angular frame is built from
   ``sn.quad`` and knows nothing about the spatial coordinate system.
   A 1-D polar Gauss–Legendre rule has no azimuthal nodes, so the
   :math:`m \ne 0` modes are not orthogonal under it and the Gram is
   dense at :math:`L\ge2` — `[M]` refining the polar order does not fix
   it (``gauss_legendre(8)`` at :math:`L=2` reads the same
   :math:`1.155` off-diagonal and the same :math:`16.17`).
#. The **measure-level** correspondence —
   :math:`E = \text{reconstruction}(e_0\,\cdot)/W` — is bit-exact in
   *every* configuration, dense Gram included, because it never touches
   a metric. That is the honest statement of the relationship, and it
   is also the reason the collapse pair is minted from an **indicator**
   frame rather than lifted out of the harmonic one: the collapse rides
   the measure, and it must keep working where the harmonic frame's
   metric does not exist.


.. _spaces-collapse-pair-gates:

Verification — cite the gate, never copy its numbers
------------------------------------------------------

The battery is ``tests/numerics/test_axis_marginal.py``, ``foundation``
-tagged throughout: these are software and mathematical invariants of a
*construction*, not equation claims, so no row carries
``verifies(...)``. Each row's docstring names the mutation that reddens
it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Row family
     - What it pins
   * - ``TestSectionLaws``
     - :math:`R\circ E = \mathrm{id}`, idempotence of :math:`P`, and
       that the marginal space is the remaining axes **verbatim**
       (measures intact, so the marginal's metric stays physical).
   * - ``TestAdjointPairing``
     - the adjunction on the physical metrics, **with** its vv #19
       negative leg — the same pairing under a deliberately stripped
       spatial measure must break at O(1), because a positive reading
       alone cannot discriminate metric-loaded from metric-blind.
   * - ``TestTwoArrows``
     - :math:`R^{\dagger} = (\Sigma w)E` (the anti-ERR-051 row) and the
       *type* discrimination — ``R.H`` must not be an
       :class:`~orpheus.numerics.operator.AxisSectionOperator`.
   * - ``TestShippedKernelEquivalence``
     - bit-identity with the canonical angular reduction and the
       isotropic-source kernel on the real :math:`S_N` carrier, and
       that the angular marginal **is** the carrier's scalar bulk
       space. ⚠ Both equivalence rows are pinned against kernels
       hand-spelled **in the test**: since S6.2 the production targets
       route through the very operators under test, so a production
       comparison would be tautological.
   * - ``TestAxisGeneric``
     - the verbs are not angular-only — an untyped axis is admitted
       whatever its label, and a multi-dimensional axis contracts all
       of its dims with its own measure.
   * - ``TestAdmission``
     - every refusal in the clause table above, plus the shape guards
       in both directions, and the zero-total-weight asymmetry.
   * - ``TestFrameInduction``
     - the generator discipline itself: **tightness** (the minted
       kernels against an independently rebuilt literal frame's face
       contents), the **gram-derivation** of the divisor, the clause-2
       energy refusal, and that both verbs share one memoized mint.

.. _spaces-fences:

What is NOT built (the standing seams)
======================================

Stated explicitly so no reader mistakes a chartered design for shipped
machinery, and so the next phase does not re-derive a decision already
taken.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Not built
     - Where it lands, and what stands in for it today
   * - ``SpatialAxis`` / ``QuadratureAxis`` / ``HarmonicAxis``
     - **Split owners since 2026-09-07.** ``HarmonicAxis`` (rank-2
       :math:`(L+1, 2L+1)`, MODAL) and the Legendre / spatial-moment
       subclasses are **CS4c step 6 item 6.2c**, the step that makes the
       angular head axis-built — ruled 2026-09-07, sequenced after
       6.2b, and owing its own verification round before code because
       the identity and adjoint metrics move across frame, fields and
       operators. ``SpatialAxis`` / ``QuadratureAxis`` remain **CS2**
       and unscheduled. Today there are still no axis subclasses beyond
       :class:`~orpheus.numerics.axis.EnergyAxis`; the spatial factor of
       :attr:`MaterialMesh.bulk_space
       <orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>`
       is a generic :class:`~orpheus.numerics.axis.Axis` labelled
       ``"spatial"``. The quotient point is a generic instance today and
       gets re-homed as ``SpatialAxis.quotient_point()`` when the
       subclass lands.
   * - Axis-built COMPOSITE and TRACE spaces
     - **CS2.** ⚠ Re-measured 2026-08-24: the *bulk* half of this
       fence has fallen. Campaign 1 CS4b moved the angular family onto
       axis-built carrier mints, so `[M]` on a shipped 1-D
       :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` the scalar bulk
       ``(energy, spatial)``, the angular bulk
       ``(angular, energy, spatial)`` and the scheme-widened
       ``angular_trial_space`` are ALL axis-built and all report
       ``has_coordinate_cone is True``. What is still legacy
       (``axes is None``, ``has_coordinate_cone is None``) is the
       **composite** :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
       and the flat **trace** buffers — the block/direct-sum structure
       the axis layer does not yet compose (see the
       :math:`\oplus` row below). This is also why the axis collapse
       pair refuses a non-axis-built space: it has no named factors to
       marginalise over.
   * - :math:`\oplus` composition (direct sums of spaces)
     - **CS2's opener.** The axis layer composes with :math:`\otimes`
       only; block/composite structure still rides
       :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
       and the coupled-block machinery.
   * - The rank-:math:`d` measure-to-axis pairing
     - **CS2.** A :class:`~orpheus.numerics.measure.DiscreteMeasure` is
       a flat atom list, so :meth:`DiscreteMeasure.axis
       <orpheus.numerics.measure.DiscreteMeasure.axis>` mints at rank 1;
       the rank-:math:`d` arm of :attr:`MaterialMesh.bulk_space
       <orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>`
       therefore stays literal and **generator-less by contract**,
       gated as such. Inverting that row is a deliberate CS2 act
       (:ref:`spaces-generator-seams`).
   * - A ``Basis`` that mints its own MODAL axis
     - **CS2.** :attr:`~orpheus.numerics.axis.Axis.generator`'s
       annotation admits a
       :class:`~orpheus.numerics.basis.base.Basis`, but `[M]`
       ``hasattr(Basis, "axis")`` is ``False`` and no subclass defines
       it — the MODAL arm is declared and unbuilt. Both shipped mints
       are ``NODAL`` by construction, so no MINT can produce a modal
       generator-ful axis (:ref:`spaces-generator-seams`).
   * - ✅ A solve-time consumer that READS a generator — **BUILT
       2026-08-29**
     - **Discharged by the streaming campaign's P4-remainder**
       (``ad04e236``), and kept here past-tensed rather than deleted so
       the *reason* survives. CS5 landed the slot and the two
       generator-minted nodal sites; no production consumer read
       ``axis.generator``, so the two gates that would witness one (a
       generator-less refusal, and the route keystone) were withheld
       rather than shipped green and unfalsifiable. The re-point landed
       them with it: the streaming producer binds an
       :attr:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.angular_axis`
       and recovers the direction cosines and the level fibration through
       its generator, and the angular-closure family's construction
       contract takes the axis as a third operand
       (:ref:`spaces-generator-seams`,
       :ref:`spaces-generator-route-gate`).
   * - ``FunctionSpace.manifold`` — the level-1 slot
     - **#429 tracker 2.0c.** A space records the index shape of its
       DOFs and not the point set those DOFs discretise, so the
       manifold is smuggled through a NAME STRING. ✅ **Half discharged
       2026-09-01 by tracker 2.1**: ``basis/indicator_basis.py`` used to
       hard-code ``f"L2[coarse_cells_R{self.ndim}]"`` — **false** for the
       energy-grid basis, an index partition calling itself spatial, and
       ``==``/hash-equal to a same-sized spatial space — and now derives
       ``f"L2[coarse_cells({self.domain.name})]"`` from a
       :class:`~orpheus.numerics.manifold.Manifold` the caller declares.
       ⚠ **What remains is that the SPACE records only the string.**
       `[M]` both producers now interpolate a typed
       :class:`~orpheus.numerics.manifold.Manifold` — ``measure.py:371``
       is ``f"L2[{self.support.name}]"`` since tracker 2.0c retyped
       ``support``, and the indicator basis is the line above — but a
       ``FunctionSpace`` still holds the resulting *name*, so the two
       producers agree by discipline rather than by construction.
       ``tests/numerics/test_basis_domain.py::test_d6`` pins that
       agreement, and pins the one pair that does NOT yet agree in
       spelling (``LossKernelBasis``'s bare label against ``IndexSet``'s
       ``index(...)``). ⛔ This clause read *"`[M]` ``measure.py:331``
       still derives ``f"L2[{self.support}]"`` from a ``str``"* until
       2026-09-01: true when written, and repealed hours later by 2.0c,
       which is the campaign's own step. The level-1 view of this seam,
       and the rest of that migration, is at :ref:`manifold-seams`.
   * - The identity flip (an axis-built space IS its axis tuple)
     - ✅ **LANDED 2026-09-07 — the identity flip, CS4c step 6**
       (structural ``__eq__``). ``__eq__`` and ``__hash__`` read
       ``axes`` directly when both sides carry one; an axes-less space
       keeps ``(name, shape)``, which for the digest-named leaf classes
       IS content identity; one of each is never the same space.
       ⛔ This row read *"S3. Until then the derived name is the bridge,
       and axes is declared compare=False"* until 2026-09-07. Its first
       half is now history
       (:ref:`spaces-identity-bridge`); its second was never the whole
       mechanism — the field stays ``compare=False`` so a subclass's
       dataclass-generated ``__eq__`` never reaches an ndarray, while the
       manual ``__eq__`` reads it. ⚠ The "S3" of that retired sentence is
       this plan-internal step, NOT the landed **CS4b S3** content-digest
       re-key that gave the composites and trace spaces their names —
       which is why the landed thing is named *the identity flip* here
       and everywhere after it.
   * - Retiring the densifying ``__mul__``
     - ✅ **LANDED 2026-09-07 — CS4c step 6 item 6.2a.** ``*`` survives
       and stopped densifying: the dense outer-product weights builder
       and the mixed-product bridge that fed it are gone, and a product
       never populates ``inner_product_weights``. What remains of the
       twin is *axes-lessness*, not densification — item **6.2c**
       axis-ifies the angular head and closes that.
       ⛔ This row read *"CS2. The legacy* ``*`` *path is documented
       above and kept working; its own gates live in a separate test
       module so the retirement is a file-level move"* until then. The
       first clause is now history; the **second was wrong about the
       mechanism**, and the landing says so. A gate that pins
       *behaviour* migrates with the behaviour, it does not travel with
       a file: ``test_space_algebra.py``'s two dense-slot rows
       (``test_inner_product_factorises_weighted``,
       ``test_inner_product_mixed_euclidean_and_weighted``) and
       ``test_space_of_axes.py``'s mixed-product third leg were
       **re-keyed in place** onto the factored metric's applied VALUES —
       each still asserts the same outer product, now as
       ``tp.apply_metric(...)`` rather than as a stored tensor — and the
       new arm-agreement band lives in
       ``tests/numerics/test_tensor_product_metric_is_factored.py``. No
       file moved.
   * - The condensation morphisms on :math:`V` / :math:`V^*`
     - **Campaign 2.** Declared at
       :ref:`spaces-vv-collapse-hook`; the axis records the group
       structure they will consume. (This row said "S7" until
       2026-08-24 — a colliding plan-internal step number, see the
       warning at that anchor.)
   * - ``Medium`` and the mesh-conformity guard
     - **CS1.5.** See :ref:`spaces-symmetry-monotonicity`.
   * - Making the operators' ``space`` slot mandatory
     - **CS4.** Every operator's ``space`` is still
       ``FunctionSpace | None``; a ``None`` operand makes the
       composition guard skip rather than validate.
   * - A CHOSEN curvilinear moment mass
     - **Not scheduled here — and the reason is a split, not a delay.**
       P7 landed the *machinery* for a non-Hadamard metric; which
       :math:`G` a curved chart's multi-moment cell wants is a
       **value** question, pinned by physical functionals and needing
       the cell-solve consumer of `#158
       <https://github.com/deOliveira-R/ORPHEUS/issues/158>`_. The
       scheme family still refuses, now naming that one blocker.
       Expressible is not known
       (:ref:`spaces-metric-not-on-the-axis`).
   * - The Riesz legs (``riesz_lower`` / ``riesz_raise``)
     - **CS4c.** P7's metric family is what those legs will wrap —
       the metric arithmetic has one home, so retiring
       ``AdjointOperator`` into
       :math:`A^{*} = A.\mathrm{domain.riesz\_raise}\circ A.\mathrm{dual}()
       \circ A.\mathrm{codomain.riesz\_lower}` needs no third spelling
       of it. `[M]` neither method is defined anywhere in the tree
       today (``hasattr`` is ``False`` on
       :class:`~orpheus.numerics.space.FunctionSpace`,
       :class:`~orpheus.numerics.space.DualSpace`,
       :class:`~orpheus.numerics.space.TensorProductSpace` and
       :class:`~orpheus.numerics.operator.LinearOperator`); the only
       occurrence of either name is the metric module's own docstring,
       naming the compatibility target.


.. _spaces-development-history:

Development history
===================

Reverse-chronological (latest first) changelog of the architectural
milestones of *this page's* subject — the space layer. The field
algebra's own changelog is on
:doc:`/theory/foundations/field_algebra`, the operator algebra's on
:doc:`/theory/foundations/operator_algebra`, and the S\ :sub:`N`
solver's is :ref:`sn-development-history`. Entries marked *(in
development)* live on an unmerged feature branch and have no landed
merge-to-``main`` hash yet; trust ``git`` over this table for merge
status.

.. list-table::
   :header-rows: 1
   :widths: 10 50 12 28

   * - When
     - Architectural milestone
     - Issue
     - Where
   * - 2026-08-30
     - **The metric becomes an OBJECT — a space can carry a
       non-diagonal** :math:`G` (campaign 1, phase P7). The metric stops
       being a thing that is *multiplied* into the element (a broadcast
       weight array, or the per-axis factor measures) and becomes a
       thing that is **applied**: a typed
       :class:`~orpheus.numerics.metric.HilbertMetric` family —
       :class:`~orpheus.numerics.metric.DiagonalMetric` (the Hadamard
       realization, bit-identical to the arithmetic it replaces),
       :class:`~orpheus.numerics.metric.DenseMetric` (a symmetric matrix
       on the flattened leading block, Moore–Penrose inverse face,
       refusing an asymmetric matrix or a Penrose-inconsistent pair) and
       :class:`~orpheus.numerics.metric.FactoredMetric` (a lazy
       per-block tensor product; the Kronecker product is never
       materialized). The space gains a third metric SOURCE and a
       three-arm exclusivity guard — of whose arms the
       ``(weights, metric)`` one had been structurally unreachable — and
       the three propagation sites that used to copy a weight array are
       taught: :meth:`DualSpace.of
       <orpheus.numerics.space.DualSpace.of>` threads the object,
       ``TensorProductSpace.from_factors`` grows a dense-factor arm, and
       :attr:`FrameBase.gram <orpheus.numerics.frame.FrameBase.gram>`
       **strips** it while installing the row-sum probe (`[M]` the
       pre-P7 spelling returned ``[7.0, 11.0]`` for a projection whose
       true value is :math:`[8/3,\,16/3]` — a silent value error, and
       now an unconstructible state). The founding consumer is the
       frame: a ``DENSE`` verdict is no longer a refusal but the matrix
       pseudo-inverse dressing, so **Parseval is a theorem on every
       frame** (:eq:`spaces-pseudo-inverse-parseval`), the recorded F-0
       limitation is repaired, and `[M]` a production analysis adjoint
       moves by :math:`98\,\%` in Frobenius relative — a correctness
       repair, not plumbing. What did NOT change:
       :attr:`Axis.weights <orpheus.numerics.axis.Axis.weights>` stays a
       1-D measure (a measure is diagonal by nature; a Gram is a form),
       identity stays metric-blind, and the curvilinear moment mass
       stays refused on its VALUE alone
       (:ref:`spaces-metric-not-on-the-axis`). Found by the flip: all
       three :class:`~orpheus.numerics.space.FunctionSpace` subclasses
       that override ``__post_init__`` did so **without chaining**, and
       had been outside every base construction guard since they were
       written.
     - `#409 <https://github.com/deOliveira-R/ORPHEUS/issues/409>`_
     - ``6a0e0473`` (the family), ``bae73fa7`` (exclusivity + the
       factored product), ``f1f30cea`` (the dense dressing),
       ``af9f95f1`` (the curvilinear refusal names one blocker) —
       *(in development,* branch ``feature/p7-nondiagonal-metric``\ *;*
       ``git`` *is the merge-status authority)*
   * - 2026-08-29
     - **A producer binds the space factor, and reads the angular
       geometry through it** (the streaming campaign's P4-remainder) —
       the first solve-time consumer of
       :attr:`~orpheus.numerics.axis.Axis.generator`, which discharges
       CS5's third seam one day after it was declared.
       :class:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator`
       gains an
       :attr:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.angular_axis`
       field, minted by each of the three streaming factories from the
       measure they were already handed, and
       :meth:`~orpheus.sn.mesh.reduced_operator.ReducedStreamingOperator.streaming_terms`
       recovers the radial cosine and the level fibration **through** it.
       The recovery goes via one accessor,
       :meth:`Axis.generator_as <orpheus.numerics.axis.Axis.generator_as>`
       — a typed narrow whose refusal names both the axis and the asking
       consumer, and which is load-bearing rather than defensive because
       the declared union's other arm cannot answer either read. The
       angular-closure family's construction contract widens to
       ``cls(angular, pairing, angular_axis)``, so both concrete members
       mint through the same narrow. In the other direction the
       **courier dies** — the field whose docstring said it was *"held
       so a consumer that needs the weights, the level partition or the
       cosines does not have to be handed the quadrature separately"*,
       which is a weld's own confession:
       :class:`~orpheus.sn.angular.redistribution.AngularRedistribution`
       sheds its ``quadrature`` field and is pure :math:`\alpha` data —
       ``coord``, ``alpha_per_level``, ``mu_start_per_level`` — pinned
       structurally by field-set equality so a re-addition reds. Two
       supporting single-source moves: ``Quadrature.axis``'s label
       **defaults**, making a label twin across mint sites unspellable,
       and the cylinder admission probe's defaulted ``getattr`` on
       ``level_structure`` hardens to a direct read now that the contract
       declares the member. Every value comparison over this re-point is
       ``X == X``, so the acceptance evidence is a **route** gate with a
       decoy generator, not a value gate
       (:ref:`spaces-generator-route-gate`).
     - —
     - ``ac485104`` (the dead ``_weight_of`` retires), ``ad04e236`` (the
       binding, the courier's death, and the gates), ``1fb70c15`` (the
       admission probe reads the declared contract) — *(in development,*
       branch ``feature/p4rem-producer-binds-axis``\ *;* ``git`` *is the
       merge-status authority)*
   * - 2026-08-29
     - **An axis can name the generator that made it** (campaign 1,
       phase CS5). The axis gains a fifth slot,
       :attr:`~orpheus.numerics.axis.Axis.generator` — **provenance,
       never identity** — and the mint routes THROUGH the generator:
       :meth:`DiscreteMeasure.axis
       <orpheus.numerics.measure.DiscreteMeasure.axis>` (the
       axis-composed sibling of :attr:`~orpheus.numerics.measure.DiscreteMeasure.space`,
       ``NODAL`` by construction) and :meth:`Quadrature.axis
       <orpheus.numerics.quadrature.directional.Quadrature.axis>`
       (delegating the structural mint down and upgrading only the
       provenance). An axis is a **forgetful map** from its generator —
       it keeps the weights and drops the nodes — and the mint is that
       map's **section**, so the forgetting is recoverable
       (:eq:`spaces-axis-generator-section`): a consumer holding only
       the SPACE recovers ``mu_x`` / ``eta`` / ``mu_z`` /
       ``level_indices``. The generator is EXCLUDED from
       ``_identity_key`` — structurally mandatory, not taste, since an
       inclusion makes ``Axis.__eq__`` and ``hash(Axis)`` RAISE
       (:ref:`spaces-generator-identity-exclusion`). **Two** of the
       tree's three nodal mint sites consume it, bit-identically:
       :attr:`SNMesh.angular_bulk_space
       <orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space>`
       collapses to ``self.quad.axis(...)``, and the rank-1 arm of
       :attr:`MaterialMesh.bulk_space
       <orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>`
       mints through the carrier's own ``volume_measure``; the third,
       the homogeneous pose's counting point, keeps its literal and is
       documented as honestly generator-less. The rank-:math:`d` spatial
       arm likewise stays literal, by contract. The chartered ``DiscreteMeasure``-typed accessor
       was **refuted** by the phase's opening ground measurement — a
       bare measure answers three of the done-when's four names
       (:ref:`spaces-generator-why-quadrature`). Sibling repair: the
       ``AngularMeasure`` Protocol in
       :mod:`orpheus.sn.angular.redistribution` now declares the
       ``level_structure`` member its cylinder-factory consumer was
       already reading past the contract
       (:ref:`spaces-generator-protocol`).
     - —
     - merged @ ``cb3cd15b`` — ``4e7b8977`` (the slot, the two mints,
       the three consumers, the gates), ``b0bfc06c`` (the Protocol
       declaration), ``cb3cd15b`` (the gate-quality repairs the docs
       pass's own census found: the roster's fifth factory, the
       unmarked gate class, the module head still saying "four slots")
   * - 2026-08-24
     - **The space becomes the construction key, and it mints the
       collapse pair** (campaign 1, phase CS4b, steps S5–S7).
       *Construction goes space-primary* (S5): the carrier gains
       :attr:`SNMesh.angular_trial_space
       <orpheus.sn.mesh.augmented_mesh.SNMesh.angular_trial_space>`
       (the scheme-widened angular mint), the composite allocators go
       space-keyed, and the mesh-keyed leaf **sugar tier is deleted** —
       every call site now names a space, not a carrier.
       *The axis collapse pair is minted on the space* (S6): the verbs
       :meth:`FunctionSpace.retraction
       <orpheus.numerics.space.FunctionSpace.retraction>` /
       :meth:`~orpheus.numerics.space.FunctionSpace.section` return the
       split epi/mono pair, memoized per axis label, and S6.0b re-carved
       their REALIZATION so both are the induced output of a
       single-region indicator frame built and discarded at one site —
       the **stage-2 generator discipline**, with the section's divisor
       read off the frame's :math:`1\times1` Parseval metric rather
       than chosen by hand (:ref:`spaces-collapse-pair`). The angular
       reduction and the isotropic-source projection are re-keyed onto
       that pair, so each has ONE realization tree-wide, and
       ``AxisEmbeddingOperator`` is renamed
       :class:`~orpheus.numerics.operator.AxisSectionOperator` on
       canonical-naming grounds (:ref:`spaces-collapse-pair-naming`).
       *The mesh-less carrier's two meanings un-weld* (S7): promoting
       the infinite-medium 1-cell carrier to an :math:`S_N` phase space
       raises a typed :class:`ValueError` (pre-repair a messageless
       bare ``assert`` that ``-O`` stripped into a deep
       ``AttributeError``), and
       :attr:`MaterialMesh.areas
       <orpheus.transport.mesh.material_mesh.MaterialMesh.areas>` names
       its own three cases instead of blaming a 2-D mesh for all of
       them. The homogeneous solver's reaction rates become the typed
       integrated co-vector, **re-posed onto the solver's own pose** so
       the pose stays the measure authority
       (:doc:`/theory/foundations/infinite_medium`).
     - —
     - merged @ ``55bb47b9`` —
       ``b00bf2d7`` … ``2690a434`` (space-primary construction),
       ``048144db`` (the pair), ``19b85775`` (the frame induction +
       the rename), ``ffb8f286`` (space-derived truncation),
       ``78925753`` / ``53e7d207`` (the re-keys and the packer
       re-home), ``1f8e0323`` (the S7 repairs), ``2e054bfc`` (the
       typed rate co-vector)
   * - 2026-08-20
     - **The space layer gains AXES, and the energy axis is the first
       one** (campaign 1, phase CS1). A new
       :mod:`orpheus.numerics.axis` mints
       :class:`~orpheus.numerics.axis.Axis` (frozen; structural
       per-subclass identity; canonical measure storage — all-ones
       collapses to ``None``, :math:`-0.0` normalized, non-finite
       refused, signed weights deliberately legal) and
       :class:`~orpheus.numerics.axis.EnergyAxis`
       (``from_grid`` / ``synthetic``; identity = :math:`n_g` + edges
       CONTENT; weighted axes refused by the counting-measure theorem).
       :meth:`FunctionSpace.of_axes
       <orpheus.numerics.space.FunctionSpace.of_axes>` composes a space
       as the ordered product of its axes with a **per-axis metric
       path** (no densification) and a deterministic, injective
       **derived name** — the identity bridge that made "metric
       differences imply space differences" true from that day, three
       weeks before the identity flip made it direct
       (:ref:`spaces-identity-bridge`).
       :attr:`FunctionSpace.has_coordinate_cone
       <orpheus.numerics.space.FunctionSpace.has_coordinate_cone>` makes
       the NODAL/MODAL dichotomy machine-readable, and
       :meth:`Field.cone_violations
       <orpheus.numerics.field.Field.cone_violations>` consults it.
       :attr:`MaterialMesh.bulk_space
       <orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space>`
       mints the scalar bulk through ONE uniform formula, so the
       degenerate carrier's quotient point, a genuine one-cell mesh and
       a meshed carrier all fall out of the same body; the homogeneous
       solver then poses :math:`A = C - K_{\rm iso}` and
       :math:`F` on that real space, retiring both hand-written
       **production** ``basis_shape=(ng, 1)`` spellings and turning the
       ``OperatorSum`` guard from *skipped* into *validating* — `[M]`
       zero ``basis_shape=(ng, 1)`` call sites remain anywhere in
       ``orpheus/``, though the keyword survives as an explicit
       override. The four
       ``test_monomorphic_leaves`` strict-xfail rows are deleted and
       succeeded by a positive floor. The **counting-measure theorem is
       why this moved no values**: identity metrics along both factors,
       and guards that compare spaces rather than values.
     - —
     - merged @ ``55bb47b9`` —
       ``1afff47b`` (the axis), ``f4876354`` (``of_axes`` + per-axis
       metric + cone metadata), ``e8769897`` / ``24a991ba`` (the
       operators' space slot renamed and widened), ``6bd782ab`` (the
       homogeneous posing), ``6da1b23c`` (the cone consult)
   * - 2026-08-19
     - **Flux is typed into the positive cone** :math:`K \subset V`,
       which is what made *this* page's question askable: once cone
       membership is an element predicate rather than a constructor
       invariant, "on which spaces is the predicate meaningful?" becomes
       a question about the SPACE, and the answer is the basis kind of
       its factors (:ref:`spaces-nodal-modal`). See
       :doc:`/theory/foundations/field_algebra`.
     - #331
     - merged ``f9d571b5``

.. admonition:: Verification — cite the gate, never copy its numbers
   :class: note

   The CS1 battery is ``foundation``-tagged throughout: these are
   software and mathematical invariants of a *type*, not equation
   claims, so no gate carries ``verifies(...)``.

   - ``tests/numerics/test_axis.py`` — the intrinsic laws of the axis
     concept: rank, measure canonicalization, the refusals, structural
     identity per subclass, and the ``synthetic`` / ``from_grid``
     inequality.
   - ``tests/numerics/test_space_of_axes.py`` — composition: shape
     concatenation, the per-axis metric against an independently built
     reference, the no-densification proof, the derived name's
     determinism across processes, and ``has_coordinate_cone``.
   - ``tests/numerics/test_field.py`` (gates E1/E2) — the cone
     consult's **positive and negative pair**: a MODAL space REFUSES
     with a typed error naming the space, and the same values on an
     all-NODAL space answer exactly what the legacy path answers.
   - ``tests/homogeneous/test_operator_spaces.py`` — the positive floor
     (all five homogeneous operators plus :math:`K` report the SAME
     space), the refusal witnesses (a 2g-vs-4g sum; :math:`M^{-1}(2g)
     \circ F(4g)`), the energy arm's ``from_grid``-vs-``synthetic``
     discrimination, and the ``.H`` **loaded/blind pair** required by
     ``vv-principles`` anti-pattern #19 — a bit-identity leg for the
     shipped scalar-metric case, paired with a deliberately
     non-physical per-group-weighted axis on which ``.H`` demonstrably
     MOVES.
   - ``tests/homogeneous/test_byte_stability.py`` — the migration gate
     that measured the theorem. It pins the homogeneous solve
     bit-exactly (``np.array_equal`` and exact ``==``, never
     ``allclose``) against a baseline captured immediately before the
     wiring, over every producing mixture the tree ships. `[M]` it held
     bit-exactly across the rewiring on 2026-08-20 — which is the
     evidence for the "no value motion" claim above. It is a CS1
     migration gate by design and retires after the merge cycle,
     subsumed by the L1 correctness anchor and the materialization byte
     pin.
