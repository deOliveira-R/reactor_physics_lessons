r"""The composite carrier: a generic ``interior ⊕ boundary`` block field.

L2 (transport, method-agnostic). This module holds TWO things:

* :class:`Composite` — the **generic** two-block composite
  ``Composite[Interior, Boundary]``: an ``interior`` field paired with its
  ``boundary`` partner, carrying the vector-space algebra ONCE. It is the
  structural object the within-group transport operator
  :math:`A = L + C - S - B` acts on (the fission gain :math:`F` sits on the
  right-hand side, never inside :math:`A`) — every operator leaf maps a
  composite to a composite (the inner role may change, flux → source, but the
  carrier type does not). The name is **structural, not domain-role**: a
  domain meaning arises from the *specialization* — ``SN`` is
  ``Composite[AngularFlux, AngularBoundaryFlux]``, diffusion / CP are
  ``Composite[ScalarFlux, ScalarBoundaryFlux]``, MoC its own leaves.
* :class:`FullField` — the **SN specialization**
  ``Composite[BulkField, BoundaryField]``: the generic base with the SN
  concrete-locus guards (interior is a ``BulkField``, boundary a
  ``BoundaryField``). A pure two-block composite — the curvilinear
  starting-direction (ψ½) state that transiently rode here as an optional
  third block was evicted by the coupled-block campaign's Phase B.2d into
  its own **System-B** composite
  (:class:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField`);
  on a carrying mesh the driver iterate is the coupled pair
  ``CoupledField[FullField, RadialCharacteristicField]``, never a
  wider ``FullField``.

Why the interior + boundary split is the right L2 abstraction
=============================================================

The pre-D-H ``orpheus.sn.angular_flux.AngularFlux`` conflated volumetric
flux values with boundary trace values. Per Cardinal Rule 2 (shared concepts →
shared abstraction), the interior + boundary split is NOT SN-specific — every
transport method has the same pair:

* **SN**: interior =
  :class:`~orpheus.transport.fields.angular_flux.AngularFlux` on
  ``(N, ng, *spatial)``; boundary =
  :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
  on the flat face-trace layout.
* **CP / MoC / diffusion**: their own interior + boundary leaves
  (diffusion / CP already build ``Composite[ScalarFlux, ScalarBoundaryFlux]``).

Where the curvilinear ψ½ state lives (#282 route (a) → Phase B.2d)
==================================================================

On a mesh whose Morel–Montry thread genuinely consumes independent
starting-direction state (the R12a predicate — the 1-D sphere; see
:mod:`orpheus.numerics.spaces.radial_characteristic_space`), the ψ½ ray is
**System B** — its own ``interior ⊕ boundary``
:class:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField`,
coupled to System A (this ``FullField``) through the named blocks of the
within-group grid (``A_AB`` Seeding / ``A_BA`` Emission / ``B_b``; see
:func:`orpheus.sn.coupled_system.build_within_group_system`). The 2.5d
interim — ψ½ as an optional third block ON this class, with mesh-keyed
presence and a mixed-presence law — is retired: presence is now
*existence of System B* (a carrying mesh builds a 2×2 coupled system, a
seedless one a 1×1), so a seed block on System A's composite is not merely
illegal, it is unrepresentable.

The cofree-comonad framing (the #217 split)
===========================================

:class:`~orpheus.transport.timed_full_field.TimedFullField` is the **cofree
comonad** ``Cofree(FullField, depth=d)`` over the base: it pairs the *current*
timeless frame with a rotating history buffer of prior timeless frames. Only the
iteration / time-stepping drivers see the comonad (the history); the operator
algebra is blind to it — it reads a timeless frame in and writes a timeless
frame out.

This is why the split is structurally FORCED rather than aesthetic:

* A **static source** ``q = q_int ⊕ q_∂`` has no time — it never ``advance``\ s.
  Typing it as a history-bearing ``TimedFullField`` hands it verbs (``advance`` /
  ``at_lag``) it must never use ("a type error of altitude").
* An **iterating flux state** ``ψ^n`` DOES advance through iterations / time
  steps — it is the comonad, ``TimedFullField``.

So the operator-algebra carrier is the timeless :class:`FullField`; the
driver-level iterate is the timed :class:`TimedFullField`. This module holds the
algebra ONCE (DRY) on :class:`Composite` — every subclass inherits it.

Algebra contract
================

Same-class arithmetic propagates to every block:

.. code-block:: python

    a + b = Composite(interior=a.interior + b.interior,
                      boundary=a.boundary + b.boundary)

The six vector-space dunders (``+``, ``-``, unary ``-``, scalar ``*``,
``scalar *``, ``/ scalar``) live ONCE on :class:`Composite` and route through two
small per-shape hooks — :meth:`Composite._map_binary` (elementwise over two
operands' blocks) and :meth:`Composite._map_unary` (elementwise over one
operand's blocks) — then rebuild via the polymorphic :meth:`Composite._recombine`
hook. A subclass needing a different rebuild overrides ONLY the hooks; the
dunders themselves are never duplicated.
:class:`~orpheus.transport.timed_full_field.TimedFullField` overrides
:meth:`_recombine` alone to rebuild a ``TimedFullField`` with empty history
(#217: "algebra results carry empty history"). One definition of the algebra, the
correct concrete return type for each subclass.

Cross-class arithmetic is rejected at two layers: :meth:`Composite._check_partner`
rejects a partner that is not a :class:`Composite`; the member-level leaf dunders
enforce role / units / space content (the leaf-type match
``AngularFlux + ScalarFlux → TypeError``) by delegation, so the composite does
NOT pre-check member types — that would be a second spelling of the members'
own law. (Until campaign 1 CS3, 2026-08-19, this delegation also carried the
#208 affine torsor gate; flux lives in V now and same-typed ``±`` are plain
vector ops.)

Grep signal
===========

``Composite`` — the generic *structural* interior ⊕ boundary carrier.
``FullField`` — the SN specialization (the *full* SN domain, *timeless*).
Its history-bearing subclass keeps the strong three-token grep signal
``TimedFullField`` (Timed + Full + Field).

References
==========

* GH **issue #217** — the timeless-``FullField`` extraction (the composite source
  is the first timeless consumer).
* ``.claude/plans/archive/coupled_block_operator_campaign.md`` — the ``Composite``
  generalization (Phase A2) + the ψ½ eviction into System B (Phase B.2d).
* Grand Report v3 §5.5 (Field hierarchy), §5.3 (``DirectSumSpace`` /
  :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`).
* ``coding-elegance`` Pattern 2 (the algebra lives ONCE in the base via the
  recombine + block-map hooks) + Pattern 5 (``Composite`` is the right primitive;
  ``FullField`` narrows its loci, ``TimedFullField`` composes it with history).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Generic, Self, TypeVar

import numpy as np

from orpheus.numerics.field import Field
from orpheus.numerics.spaces.full_field_space import FullFieldSpace
from orpheus.transport.fields._bases import (
    BulkField,
    BoundaryField,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from orpheus.transport.mesh.material_mesh import MaterialMesh


__all__ = ["Composite", "FullField"]

#: The interior (first-block) leaf type — any
#: :class:`~orpheus.numerics.field.Field` leaf. The bound is the generic
#: ``Field`` base (NOT ``BulkField``): the two composite slots are the
#: structural *interior* and *boundary* blocks, and the SPECIALIZATION fixes
#: the concrete locus — System A (:class:`FullField`) narrows interior to a
#: :class:`~orpheus.transport.fields._bases.BulkField` (``AngularFlux``);
#: System B (the ψ½ ray) to a codim-1 ``RadialCharacteristicInteriorFlux``
#: (a ``FaceField`` sibling, NOT a ``BulkField``). The generic parameter makes
#: ``Composite[AngularFlux, ...].interior`` read as the PRECISE leaf type.
Interior = TypeVar("Interior", bound=Field)
#: The boundary (second-block) leaf type — any
#: :class:`~orpheus.numerics.field.Field` leaf (the specialization narrows it:
#: System A to a :class:`~orpheus.transport.fields._bases.BoundaryField`
#: spatial trace, System B to the ψ½ ``RadialCharacteristicBoundaryFlux`` r = R
#: corner).
Boundary = TypeVar("Boundary", bound=Field)
#: A composite flavor — the template type of :meth:`Composite.from_flat` (its
#: return follows the template, so the reconstruction preserves the concrete
#: subclass: a ``TimedFullField`` template yields a ``TimedFullField``).
CompositeT = TypeVar("CompositeT", bound="Composite")


@dataclass(frozen=True, kw_only=True)
class Composite(Generic[Interior, Boundary]):
    r"""The generic ``interior ⊕ boundary`` composite carrier.

    A structural two-block composite: an ``interior`` volumetric field paired
    with its ``boundary`` trace partner. Generic over the two leaf types, so the
    specialization carries the domain meaning
    (``Composite[AngularFlux, AngularBoundaryFlux]`` = an SN frame,
    ``Composite[ScalarFlux, ScalarBoundaryFlux]`` = a diffusion / CP frame),
    while THIS type is method-agnostic. Holds the vector-space algebra ONCE (the
    six dunders route through the :meth:`_map_binary` / :meth:`_map_unary` /
    :meth:`_recombine` hooks a subclass overrides to add extra blocks).

    Parameters
    ----------
    interior : Interior
        The interior (first-block) leaf — any
        :class:`~orpheus.numerics.field.Field`. The specialization narrows it
        to the concrete locus (a ``BulkField`` volumetric flux for System A, a
        codim-1 ``RadialCharacteristicInteriorFlux`` for the ψ½ ray).
    boundary : Boundary
        The boundary (second-block) partner leaf — any
        :class:`~orpheus.numerics.field.Field` (a ``BoundaryField`` spatial
        trace for System A, the ψ½ ``RadialCharacteristicBoundaryFlux`` r = R
        corner for System B).

    Notes
    -----
    NOT a :class:`~orpheus.numerics.field.Field` subclass at the typed-class
    level — its natural backing is a
    :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
    (direct-sum) Field. Ships as a structured composite with delegate-style
    dunders that propagate to every block.
    """

    interior: Interior
    boundary: Boundary

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def zeros(
        cls,
        *,
        interior: "type[Interior]",
        boundary: "type[Boundary]",
        space: "FullFieldSpace",
    ) -> "Self":
        r"""Allocate a zero composite from the leaf TYPES + the composite SPACE.

        Generic over the method's leaf types: the caller passes the interior and
        boundary :class:`~orpheus.numerics.field.Field` *subclasses* (SN passes
        :class:`~orpheus.transport.fields.angular_flux.AngularFlux` /
        :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`;
        diffusion / CP pass their scalar leaves) and the carrier's cached
        :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
        (``sn.full_field_space`` / ``diffusion_mesh.full_field_space``);
        each block is zero-allocated on the matching block space via the one
        :meth:`~orpheus.numerics.field.Field.zeros` primitive. A composite is
        an ELEMENT of its direct-sum space (CS4b S4), so its allocator is
        space-keyed exactly like the leaves' (S5 — the mesh-keyed
        ``zeros_on`` delegation retired with the sugar tier). This keeps the
        cross-method-generic container free of any hard-wired leaf type.
        """
        interior_space = space.interior_space
        trace_space = space.trace_space
        if interior_space is None or trace_space is None:
            raise ValueError(
                f"{cls.__name__}.zeros: the composite space must carry BOTH "
                f"block spaces (interior_space / trace_space) — a zero "
                f"composite is an element of the full direct sum; got "
                f"{space!r}."
            )
        return cls(
            interior=interior.zeros(interior_space),
            boundary=boundary.zeros(trace_space),
        )

    # ── Construction validation ──────────────────────────────────────

    def __post_init__(self) -> None:
        # The generic base enforces only that both blocks are Field leaves; the
        # CONCRETE locus — BulkField / BoundaryField for System A, the ψ½ ray
        # leaves for System B — is narrowed by each specialization's own
        # __post_init__. The base relaxed off BulkField / BoundaryField in
        # Phase B (the coupled-block campaign) so a ray leaf, which is neither,
        # can fill a slot.
        if not isinstance(self.interior, Field):
            raise TypeError(
                f"{type(self).__name__}: interior must be a Field leaf; got "
                f"{type(self.interior).__name__}"
            )
        if not isinstance(self.boundary, Field):
            raise TypeError(
                f"{type(self).__name__}: boundary must be a Field leaf; got "
                f"{type(self.boundary).__name__}"
            )
        # Cross-slot coherence at CONSTRUCTION retired (CS4b S4, the F4
        # ruling): the pre-S4 ``interior.mesh is boundary.mesh`` check was
        # the mesh-identity doctrine's composite arm, and the question it
        # asked — "do these two blocks belong to one carrier?" — is not
        # spellable from block CONTENT alone (a bulk space cannot derive
        # its carrier's trace space; the digests are opaque folds of
        # different data). The reference lives where a carrier does: every
        # operator/solution admission seam compares each block against the
        # carrier-minted space (the S3-landed ``space_on`` arms), which is
        # the currency doctrine — no refusal machinery at a corner no
        # workflow legitimately reaches (the S3 witnesses had to
        # MANUFACTURE mixed composites). Twin-carrier blocks, which the
        # retired check refused, are content-equal and mix by the F2 law.

    # ── The composite as a space element (CS4b S4 — the F4 ruling) ────

    @cached_property
    def space(self) -> "FullFieldSpace":
        r"""The direct-sum space this composite is an element of.

        DERIVED from the blocks — ``FullFieldSpace.from_blocks(
        interior.space, boundary.space)`` — never stored: a composite IS
        its two block elements, so its space is determined by theirs, and
        a stored copy would be a twin datum needing a coherence gate that
        checks bookkeeping instead of physics. ``from_blocks`` derives
        the name from member content (the of_axes rule), so this property
        compares ``==`` (content) with the carrier's cached mint
        (``SNMesh.full_field_space`` / ``DiffusionMesh.full_field_space``
        / ``SNMesh.radial_characteristic_field_space``) whenever the
        blocks ride carrier-minted spaces — which post-S2a they always
        do. Not ``is``: the wrapper is minted per composite (cached per
        instance); the MEMBERS are the carrier's cached objects.
        """
        return FullFieldSpace.from_blocks(
            self.interior.space, self.boundary.space,
        )

    @property
    def principal_bulk_leaf(self) -> "Interior":
        r"""The leaf whose space norm carries this iterate's convergence
        diagnostics — for a two-block composite, the ``interior``.

        The composite's own answer to the campaign-1 CS3-R relocation (each
        carrier names its principal leaf; the iteration layer reads only
        this property): the diagnostic is the INTERIOR leaf's space-induced
        ``l2``, deliberately NOT the whole-composite flat norm, which
        additionally ravels the boundary trace block (``[M]`` 4.71e-3 apart
        on the c→1 pin fixture; the convention is pinned by
        :mod:`tests.numerics.test_si_diagnostic_trajectory`).
        """
        return self.interior

    # ── Polymorphic recombine + block-map hooks (Pattern 2) ───────────

    def _recombine(
        self, *, interior: "Field", boundary: "Field",
    ) -> "Self":
        r"""Rebuild a composite of the SAME concrete type from recombined blocks.

        The polymorphic hook the block-map methods route their result through.
        The base spelling is ``replace(self, ...)`` — provably ``Self`` (and
        ``replace`` re-runs ``__post_init__`` so the block invariants re-fire for
        free). A subclass needing a different rebuild OVERRIDES this;
        :class:`~orpheus.transport.timed_full_field.TimedFullField` overrides it to
        rebuild a ``TimedFullField`` with an EMPTY history (#217).
        """
        return replace(self, interior=interior, boundary=boundary)  # type: ignore[arg-type]

    def _map_binary(
        self, other: "Composite", op: "Callable[[object, object], object]",
    ) -> "Self":
        r"""Apply a binary elementwise ``op`` across the blocks of ``self`` and
        ``other``, then recombine.

        The single place a two-operand dunder (``+`` / ``-``) spells its block
        propagation. A subclass with a different block structure overrides
        this to combine its blocks.
        """
        return self._recombine(
            interior=op(self.interior, other.interior),  # type: ignore[arg-type]
            boundary=op(self.boundary, other.boundary),  # type: ignore[arg-type]
        )

    def _map_unary(
        self, op: "Callable[[object], object]",
    ) -> "Self":
        r"""Apply a unary elementwise ``op`` across the blocks of ``self``, then
        recombine.

        The single place a one-operand transform (unary ``-``, scalar ``*`` / ``/``,
        :meth:`copy`) spells its block propagation. A subclass with a different
        block structure overrides this to transform its blocks.
        """
        return self._recombine(
            interior=op(self.interior),  # type: ignore[arg-type]
            boundary=op(self.boundary),  # type: ignore[arg-type]
        )

    # ── Algebra (propagates to every block via the map hooks) ─────────

    def _check_partner(self, other: object) -> None:
        r"""Reject a partner that is not a :class:`Composite`.

        Layer 1 at the CONTAINER level only. The member-level leaf algebra
        (class identity, space equality, mesh binding — the fiber discipline;
        since campaign 1 CS3, 2026-08-19, flux lives in V and the leaf ``±``
        are the plain vector ops, the retired affine gate and displacement
        mint included) is the SINGLE SOURCE OF TRUTH on the leaves —
        ``__add__`` / ``__sub__`` delegate to ``self.interior ±
        other.interior`` and ``self.boundary ± other.boundary``, where the
        leaf dunders enforce it. Member pre-checks here would be a second
        spelling of the members' own law, so they are intentionally absent.

        The accepted partner is ANY :class:`Composite` flavor — a timeless base or
        a timed subclass. This is load-bearing for the time-derivative stencil
        ``state.at_lag(0) - state.at_lag(1)``: the current frame is a
        :class:`~orpheus.transport.timed_full_field.TimedFullField` while a
        historical frame is a timeless :class:`FullField` snapshot, and the two
        must subtract. The CONCRETE result type is governed by :meth:`_recombine`,
        so accepting a base partner does not weaken the "algebra results carry
        empty history" guarantee. ``state + 42`` (a non-``Composite`` partner) is
        still rejected here; the leaf gate then rejects any leaf/mesh/units
        mismatch (including a cross-method ``AngularFlux + ScalarFlux``) by
        delegation.
        """
        if not isinstance(other, Composite):
            raise TypeError(
                f"{type(self).__name__} arithmetic requires a same-class "
                f"partner; got {type(other).__name__}."
            )

    # ``other`` is deliberately the BASE ``Composite`` (not ``Self``): the partner
    # rule is "any Composite flavor" (see ``_check_partner`` — load-bearing for the
    # timed − timeless time-derivative stencil), and the RESULT flavor is governed
    # by ``self``'s ``_recombine`` hook alone.
    def __add__(self, other: "Composite") -> "Self":
        self._check_partner(other)
        return self._map_binary(other, lambda a, b: a + b)  # type: ignore[operator]

    def __sub__(self, other: "Composite") -> "Self":
        self._check_partner(other)
        return self._map_binary(other, lambda a, b: a - b)  # type: ignore[operator]

    def __neg__(self) -> "Self":
        return self._map_unary(lambda a: -a)  # type: ignore[operator]

    def __mul__(self, scalar: float) -> "Self":
        return self._map_unary(lambda a: a * float(scalar))  # type: ignore[operator]

    def __rmul__(self, scalar: float) -> "Self":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Self":
        return self._map_unary(lambda a: a / float(scalar))  # type: ignore[operator]

    # ── Flat-vector protocol (Krylov / scipy.gmres adapter) ──────────
    #
    # Direct-sum flat representation: ``concat(interior.values.ravel(),
    # boundary.values)``. The boundary values are already a flat 1-D ndarray (per
    # the trace leaf's flat-backing storage); the interior values are reshaped via
    # :meth:`ndarray.ravel`. The Krylov adapter at
    # :mod:`orpheus.numerics.iteration` consumes this flat representation as the
    # GMRES iterate vector; round-trip exactness is the load-bearing invariant.

    def to_flat(self) -> "NDArray":
        r"""Pack the composite blocks into a flat 1-D vector.

        The packed layout is ``[interior.values.ravel(), boundary.values]``
        — the direct-sum representation, its ordered blocks supplied by
        :meth:`_flat_parts`. Lives ONCE here; :meth:`from_flat` is its
        inverse.
        """
        return np.concatenate(self._flat_parts())

    def _flat_parts(self) -> "list[NDArray]":
        r"""The ordered 1-D block arrays :meth:`to_flat` concatenates.

        ``[interior.values.ravel(), boundary.values]`` — the hook exists so
        the flat protocol stays defined ONCE on :class:`Composite`.
        """
        return [
            self.interior.values.ravel(),
            self.boundary.values,  # already 1-D (flat trace storage)
        ]

    @classmethod
    def from_flat(cls, flat: "NDArray", template: "CompositeT") -> "CompositeT":
        r"""Reconstruct a composite from a flat 1-D vector + template.

        The ``template`` provides the shapes, types, AND the concrete composite
        class: reconstruction is delegated to the template's :meth:`_from_flat`
        instance hook, so the result is the SAME concrete type as ``template`` (a
        :class:`~orpheus.transport.timed_full_field.TimedFullField` template yields
        a ``TimedFullField`` with preserved ``history_depth`` and an EMPTY history).

        Parameters
        ----------
        flat : np.ndarray
            1-D vector matching ``template.to_flat()`` in size.
        template : Composite
            Source of structural metadata (shapes, spaces, meshes) AND the
            concrete return type.
        """
        return template._from_flat(flat)

    def _from_flat(self, flat: "NDArray") -> "Self":
        r"""Rebuild a composite of ``self``'s type from a flat vector (self = template).

        The instance-hook inverse of :meth:`_flat_parts`: slices
        ``interior | boundary`` from the template layout, rebuilds each leaf with
        the template's ``space`` / ``mesh``, and routes the pair through
        :meth:`_recombine`. Instance method (not classmethod) so the ``Self``
        override is Liskov-clean.
        """
        n_interior = self.interior.values.size
        n_boundary = self.boundary.values.size
        expected_total = n_interior + n_boundary
        if flat.size != expected_total:
            raise ValueError(
                f"{type(self).__name__}.from_flat: flat.size = {flat.size} does "
                f"not match template total size {n_interior} + {n_boundary} = "
                f"{expected_total}"
            )
        interior_values = flat[:n_interior].reshape(self.interior.values.shape)
        boundary_values = flat[n_interior : n_interior + n_boundary]
        return self._recombine(
            interior=replace(self.interior, values=interior_values),
            boundary=replace(self.boundary, values=boundary_values),
        )

    # ── Diagnostics ──────────────────────────────────────────────────

    def copy(self) -> "Self":
        r"""Return a deep copy with owned ndarrays.

        Snapshots every carried block with owned ndarrays (via :meth:`_map_unary`,
        so a subclass's extra block is copied too). Used by callers that need a
        stable iterate without aliasing. Routes through :meth:`_recombine`, so a
        :class:`~orpheus.transport.timed_full_field.TimedFullField` copy is a
        ``TimedFullField`` with EMPTY history (the existing ``copy`` drops history
        — bit-identical behaviour).
        """
        return self._map_unary(lambda a: a.copy())  # type: ignore[attr-defined]


@dataclass(frozen=True, kw_only=True)
class FullField(Composite[BulkField, BoundaryField]):
    r"""The SN composite: ``interior ⊕ boundary`` with the SN locus guards.

    The :class:`Composite` specialization the SN operator algebra acts on —
    ``interior`` an :class:`~orpheus.transport.fields.angular_flux.AngularFlux`,
    ``boundary`` an
    :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`.
    Bound to the ``BulkField`` / ``BoundaryField`` ABCs (not a single leaf) so the
    same class serves every SN geometry. The whole 2-block algebra — the six
    dunders, the flat protocol, ``zeros``, ``copy`` — is inherited from the
    generic base; this class adds ONLY the concrete-locus construction guards.

    A curvilinear carrying mesh's ψ½ ray is **System B**, a sibling
    :class:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField`
    coupled through the within-group grid — never a block on this class (the
    Phase B.2d eviction; see the module docstring).

    Parameters
    ----------
    interior : BulkField
        The volumetric / bulk field (typically ``AngularFlux`` for SN).
    boundary : BoundaryField
        The boundary partner field on the trace of ``interior``'s domain.
    """

    # ── Construction validation (override — the SN locus narrows) ─────

    def __post_init__(self) -> None:
        # System A narrows the generic Composite slots to the SN locus types:
        # interior = the volumetric bulk (AngularFlux), boundary = the spatial
        # trace (AngularBoundaryFlux). These guards live HERE, not on the
        # generic Composite base — the base's slots relaxed to Field in Phase B
        # so System B's ψ½ ray leaves (a FaceField sibling, neither a BulkField
        # nor a BoundaryField) can fill them; the concrete-locus guard belongs
        # with the concrete specialization. Messages are verbatim (bit-identical
        # System-A behaviour: they fire BEFORE the base's generic Field check).
        if not isinstance(self.interior, BulkField):
            raise TypeError(
                f"{type(self).__name__}: bulk must be a BulkField; got "
                f"{type(self.interior).__name__}"
            )
        if not isinstance(self.boundary, BoundaryField):
            raise TypeError(
                f"{type(self).__name__}: boundary must be a BoundaryField "
                f"(an AngularBoundaryField / ScalarBoundaryField family leaf); "
                f"got {type(self.boundary).__name__}"
            )
        super().__post_init__()

    # ── The shared System-A matvec input parse ─────────────────────────
    @classmethod
    def require_member(
        cls, x: object, *, mesh: "MaterialMesh", context: str,
    ) -> "FullField":
        r"""Parse ``x`` as a System-A composite on the carrier ``mesh`` — the
        ONE matvec input contract of the SN leaves (CS4c step 6 item 6.3).

        Five consumers, one body (``coding-elegance`` Pattern 2 / Pattern 4
        — illegal inputs unrepresentable at one place, not re-validated per
        leaf): ``StreamingOperator.apply`` / ``apply_transpose``,
        ``StreamingCollisionOperator.apply`` / ``apply_transpose`` and
        ``SNBoundaryOperator._apply_faces`` (the R6 row of the
        monomorphic-leaves ledger, which until this item read
        ``psi.interior`` unguarded and leaked a raw ``AttributeError``).
        Mirrors the SHAPE of
        :meth:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField.require_member`
        — a ``TypeError`` for a foreign carrier that names the refusing
        surface and the carrier it wanted, a ``ValueError`` for a content
        mismatch carrying the greppable ``space-content invariant``
        vocabulary — but NOT its signature: System B's parse is keyed on a
        caller-supplied bound space, this one on the CARRIER, because the
        reference is derived from the operand itself
        (``x.interior.space_on(mesh)`` — *"does your space agree with what
        your family would be on MY carrier?"*): ``B_a`` is fed the moment
        iterate (`[M]` 59/58/47 ``HarmonicMomentFlux``-interior composites
        per 2-D windowed solve) as well as the angular composite its bound
        end names, and a bound-end comparison would refuse them.

        The input contract is the TIMELESS composite (the matvec leaves
        are base arrows ``FullField -> FullField``; only the iteration
        driver carries the history-bearing ``TimedFullField`` comonad). A
        ``TimedFullField`` IS a ``FullField``, so the SI / Krylov drivers
        that pass a timed iterate satisfy the parse; a bare ndarray does
        not.

        **ELEGANCE-DEBT[guard] #457** — a runtime guard is a protection,
        not the target state: it retires when every leaf is bound on the
        end it acts on (R18's ``B`` reshape — ``B`` bound on its own trace
        end, a moment-bound sibling for the windowed iterate, ``L``'s ends
        typed ``FullFieldSpace``), at which point the admission is the
        ordinary composability guard on the BOUND end and an alien carrier
        cannot be typed at all.

        Parameters
        ----------
        x : object
            The matvec input (``psi`` for apply, ``phi`` for the transpose).
        mesh : MaterialMesh
            The operator's carrier — the interior's space must content-equal
            (today: BE) its family's cached mint on it.
        context : str
            The refusing surface, e.g. ``"StreamingOperator.apply"`` —
            ``_apply_faces`` serves both directions, so the caller names
            the method rather than the parse guessing it.
        """
        if not isinstance(x, cls):
            raise TypeError(
                f"{context}: expected FullField, got "
                f"{type(x).__name__}.  D-I.3d (2026-05-29) retired the "
                "bare-ndarray packed-vector contract; construct a timeless "
                "composite via ``FullField(interior=AngularFlux(...), "
                "boundary=AngularBoundaryFlux(...))`` (or the timed "
                "``TimedFullField(interior=..., boundary=...)`` for an iterate)."
            )
        if x.interior.space != x.interior.space_on(mesh):
            raise ValueError(
                f"{context}: operator and composite must agree in space "
                "content (space-content invariant)."
            )
        return x
