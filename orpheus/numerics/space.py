r"""Function spaces for the operator-algebra framework.

A *function space* is the domain or codomain of a linear operator. In
the matrix-free transport algebra, every operator :math:`A : V \to W`
acts on discrete flux distributions that live in a definite
:class:`FunctionSpace` — angular-flux space, scalar-flux space, a
boundary-trace space, and so on. Tagging operators with their
:attr:`domain` and :attr:`range` lets the composition machinery in
:mod:`orpheus.numerics.operator` reject ill-formed compositions at
construction time (raising :class:`IncompatibleOperatorComposition`),
preventing the harmful-stub anti-pattern where a downstream Krylov
consumer hits a shape mismatch mid-iteration.

The *Hilbert-adjoint* refinement (:meth:`LinearOperator.adjoint`,
``A.H``) further requires the domain and range to carry an inner
product. :class:`FunctionSpace` stores the inner-product weights as
metadata: the L² inner product
:math:`\langle x, y \rangle_w = \sum_i w_i \, x_i \, y_i` reduces to
the Euclidean :math:`\sum_i x_i \, y_i` when the weights are absent,
which is the natural default for spaces that have no canonical
quadrature (cell-flux, region-flux). For angular-flux and
boundary-trace spaces the quadrature weights ARE the canonical
inner-product weights, so the adjoint identity
:math:`\langle A x, y \rangle_W = \langle x, A^* y \rangle_V`
becomes a non-trivial consistency check (see test
``tests/numerics/test_operator.py::test_hilbert_adjoint_weighted_identity``).

Future direction (Grand Report v3 §5.3 + §6.1)
==============================================

The Grand Report v3 anticipates a richer Space ontology with these
specialisations layered on top of :class:`FunctionSpace`:

* **MeshFunctionSpace** — functions on a structured mesh; carries a
  reference to the :class:`~orpheus.geometry.mesh.Mesh1D` /
  :class:`~orpheus.geometry.mesh.Mesh2D` instance.
* **AngularTraceSpace** — functions on the boundary; the domain/range for
  :class:`~orpheus.geometry.boundary.BoundaryTraceLaw`. ONE
  whole-boundary space (see
  :mod:`orpheus.numerics.spaces.angular_trace_space`); inflow / outflow are
  selectors over its signed :math:`\Omega\cdot\hat n`, not directional
  tags.
* **RegionSpace** — region-piecewise constant fields (one value per
  homogenised region); used by region-collapsed CP / homogenisation.
* **EnergyGroupSpace** — multi-group flux space; tensored with a
  spatial space to form the full state.
* **DiscreteAngularSpace** — quadrature-tagged angular space carrying
  invariance-group metadata and the underlying
  :class:`~orpheus.numerics.measure.DiscreteMeasure`.

And these compositional dunders:

* ``S * T`` — tensor product; produces a :class:`FunctionSpace` whose
  shape is the concatenation of factor shapes.
* ``S + T`` — direct sum; concatenated dimension on a shared abstract
  type tag.
* ``S.dual()`` — dual space; for inner-product-bearing spaces this is
  isomorphic to ``S`` itself but carries a covariance tag for
  bra-ket-style composition checks.

These are NOT shipped in 9.6 — the file is structured so they can be
slotted in additively without disturbing the :class:`FunctionSpace`
base.

References
----------

* Trefethen, L.N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM.
  §1 — vector spaces, inner products, the Hilbert adjoint vs.
  representation transpose distinction.
* Reed, M. & Simon, B. (1980). *Methods of Modern Mathematical
  Physics I: Functional Analysis*, §III.6 (Hilbert spaces, Riesz
  representation, the adjoint operator).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TYPE_CHECKING, TypeVar

import numpy as np
from numpy.typing import NDArray

from .axis import Axis, BasisKind
from .metric import (
    DenseMetric,
    DiagonalMetric,
    FactoredMetric,
    HilbertMetric,
)

if TYPE_CHECKING:
    from orpheus.numerics.frame import _AxisCollapsePair
    from orpheus.numerics.operator import (
        AxisRetractionOperator,
        AxisSectionOperator,
        LinearOperator,
        RieszLowerOperator,
        RieszRaiseOperator,
    )

__all__ = [
    "DualSpace",
    "FunctionSpace",
    "TensorProductSpace",
    "angular_flux_space",
    "scalar_flux_space",
]

#: The space's element type (what a member of :math:`V` IS): a bare
#: ``NDArray`` for the leaf spaces, a bulk ⊕ boundary composite field for
#: :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`. The
#: PEP-696 default is ``Any`` because a bare ``FunctionSpace`` annotation
#: (the operator layer's ``domain`` / ``codomain`` slots) genuinely holds
#: EITHER realization today — specializing those slots to the operator's
#: own carrier (``FunctionSpace[Domain]``, the #65 two-param discipline
#: extended to the space layer) is the follow-on step, not this one.
Carrier = TypeVar("Carrier", default=Any)


@dataclass(frozen=True)
class FunctionSpace(Generic[Carrier]):
    r"""A finite-dimensional vector space of discrete fields.

    Parameters
    ----------
    name : str
        Human-readable identifier. Used by :meth:`__repr__` and by
        :class:`IncompatibleOperatorComposition` error messages. Two
        spaces with the same ``name`` and ``shape`` compare equal even
        if they were constructed via different factory functions —
        ``name`` is the **identity** of the space, not a description
        of its contents.
    shape : tuple[int, ...]
        Tensor shape of the elements. A ``shape=(n_cells,
        n_ordinates, n_groups)`` space contains 3-D arrays; a
        ``shape=(n_cells,)`` space contains 1-D arrays.
    inner_product_weights : NDArray | None, default None
        Diagonal weights for the L² inner product
        :math:`\langle x, y \rangle = \sum_i w_i \, x_i \, y_i`. The
        weights array MUST be broadcast-compatible with ``shape``;
        most commonly a 1-D array along one axis (e.g. quadrature
        weights along the ordinate axis of an angular-flux space).
        ``None`` selects the Euclidean inner product
        :math:`\sum_i x_i \, y_i` — unless a ``metric`` object is
        installed (below), which this slot then does NOT describe: a
        dense-metric space reads ``inner_product_weights is None``
        while carrying a real, non-Euclidean metric.
    metric : HilbertMetric | None, keyword-only, default None
        The metric **object** (campaign 1, P7) — a
        :class:`~orpheus.numerics.metric.HilbertMetric` realization for
        forms no diagonal array can spell (a
        :class:`~orpheus.numerics.metric.DenseMetric` Gram inverse being
        the founding occupant). On an AXIS-BUILT space it is the
        positioned OVERLAY of forms (CS4c step 6 item 6.2c-i, ruling
        R-6.2c-2): a :class:`~orpheus.numerics.metric.FactoredMetric` with
        one entry per axis, a form only on a block whose axis carries NO
        measure (a dense-Gram head), ``None`` elsewhere — the space's
        metric is DERIVED from its axes with that overlay merged in, so
        every axis keeps supplying its own diagonal block. Resolution
        order: the axes-derived object (overlay merged) > an explicit
        object on an axes-less space > ``inner_product_weights`` >
        Euclidean. Validated at construction via
        :meth:`HilbertMetric.validate_for` and, beside axes, by the
        positioning guard.
    axes : tuple[Axis, ...] | None, default None
        The generator record of an axis-composed space (campaign 1,
        CS1): the ordered tensor factors this space is the product of.
        ``None`` means *legacy / not axis-built* — every pre-CS1
        construction path. Populated by :meth:`of_axes` (and threaded
        through ``*``); when present, the factor measures live PER AXIS
        (``inner_product_weights`` stays ``None`` — never both, enforced
        at construction) and the metric is DERIVED from them as one
        :class:`~orpheus.numerics.metric.FactoredMetric` (item 6.2c-i).
        Since the identity flip (CS4c step 6,
        2026-09-07) it IS the identity of an axis-built space —
        ``__eq__``/``__hash__`` read it directly; the dataclass field
        stays ``compare=False`` only so a subclass's generated ``__eq__``
        never reaches an ndarray (see the identity paragraph below).

    Notes
    -----
    The class is **frozen**. A :class:`FunctionSpace` encodes pure
    *geometry* — the discrete degrees of freedom (``shape``), the
    inner-product metric, and the composition algebra (``*`` /
    :meth:`dual`). It is **role- and dimension-agnostic** (the
    "View-G" decision, issues #205 / #207): a flux ``ψ`` and a
    reaction-rate density ``Lψ`` live on the *same* geometric space
    even though they carry different units. **Units do NOT live on the
    space** — they are a property of the *quantity*, carried by the
    :class:`~orpheus.numerics.field.Field` role-leaf (as a class
    constant ``UNITS``) and, for maps, by the operator's unit-gain
    (issue #208). This keeps one space per grid (no ``flux_space`` vs
    ``ratedensity_space`` duplication) and lets ``L`` / ``L⁻¹`` type as
    geometric endomorphisms on the bulk grid with a dimensional gain.

    **Identity — the chartered doctrine, realized (campaign 1, item 0.8;
    the identity flip landed at CS4c step 6, 2026-09-07).** The doctrine
    (ruled 2026-08-19/20): a space's identity is the **structural content
    of its axes** plus its tags — and since the metric lives on the axes,
    *metric differences imply space differences* (a quotient point with
    unit weight and a genuine one-cell mesh with :math:`V \neq 1` are
    DIFFERENT spaces; the old reading "two copies of :math:`\mathbb{R}^n`
    are 'the same' space regardless of which inner product is installed"
    is OVERTURNED). The realization, per class: an **axis-built** space
    compares and hashes by its ``axes`` tuple directly (``Axis.__eq__``
    is structural and excludes provenance), so an all-axes ``*`` product
    and the :meth:`of_axes` mint of the same axes are one space, and an
    axis-built space is never equal to a hand-named space wearing its
    label. An **axes-less** space keeps the nominal ``(name, shape)``
    identity, and what that means depends on the name: the digest-named
    composites and traces
    (:class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`,
    the two trace spaces, the two radial-characteristic sub-spaces —
    five classes, four digest-folding factories) fold their content into
    the name, so ``(name, shape)`` IS content identity there (CS4b S3);
    the family-tagged heads
    (:class:`~orpheus.numerics.spaces.spherical_harmonic_space.SphericalHarmonicSpace`
    — ``'spherical_harmonic_space'`` is a tag, not a digest —
    :class:`~orpheus.numerics.spaces.legendre_space.LegendreSpace`,
    :class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`)
    carried only their family and, through ``shape``, their truncation
    order — deliberately metric-blind, which is what let a frame's
    Parseval-dressed head still equal the field's continuum head (the
    admission seam CS4c step 6 items 6.2b/6.2c re-posed: since 6.2c-ii the
    two harmonic heads are axis-built and the tree binds the frame's; the
    spatial-moment tail follows at 6.2c-iii); a hand-built legacy space carries whatever
    its author wrote.
    Until the flip the same doctrine flowed through a BRIDGE: identity
    was ``(name, shape)`` for every space and :meth:`of_axes` derived the
    name injectively from the axes' content. The derived name survives
    as the readable label, as what keeps ``repr`` and guard messages
    content-distinguishing, and as the identity carrier of the axes-less
    composites that fold member names; its injectivity is no longer
    load-bearing for ``==`` on axis-built spaces.
    """

    name: str
    shape: tuple[int, ...]
    # Metadata, NOT identity (see class docstring): on an axes-less space
    # identity is (name, shape) — two such spaces with the same (name, shape)
    # are equal regardless of the installed dense metric (the leaf classes
    # derive their names from content); on an axis-built space the measures
    # live ON the axes, so they enter the identity through ``axes`` and this
    # slot is ``None``. ``compare=False`` keeps the weights out of the
    # dataclass-generated ``__eq__``/``__hash__`` of subclasses (e.g.
    # ``AngularTraceSpace``, ``SphericalHarmonicSpace``) that regenerate them — an
    # array-valued metric would otherwise make ``==`` raise on the ambiguous
    # element-wise truth value. The base class's manual ``__eq__`` already
    # ignores it; this makes every subclass agree by construction.
    inner_product_weights: Optional[NDArray] = field(
        default=None, repr=False, compare=False,
    )
    # The axis-composition generator record (campaign 1, CS1) — THE
    # identity of an axis-built space since the identity flip (CS4c step
    # 6): ``__eq__``/``__hash__`` read it directly (``compare=False`` only
    # keeps a subclass's generated ``__eq__`` from re-deriving equality
    # over every field, same rationale as the weights above) — and
    # LOAD-BEARING for the metric: when present, the factor measures live
    # per axis and the metric is DERIVED from them as one FactoredMetric
    # (never densified; item 6.2c-i retired the inline per-axis loop).
    axes: Optional[tuple[Axis, ...]] = field(
        default=None, repr=False, compare=False,
    )
    # The metric OBJECT (campaign 1, P7): the third metric source — a
    # first-class HilbertMetric realization (today: DenseMetric, the
    # non-Hadamard case no array-or-axes source can spell). Resolution
    # order in _resolved_metric: axes-derived (with this object merged in
    # as the positioned OVERLAY of forms) > an object on an axes-less
    # space > dense weights > None; beside axes the object must be
    # POSITIONED over them (one entry per axis, a form only where the
    # axis carries no measure) — the guard checks it.
    # ``compare=False`` is structurally MANDATORY, not taste: [M] a
    # compared metric field makes the dataclass-generated ``__eq__`` of
    # subclasses return an ndarray and ``hash()`` raise — the same
    # mechanism recorded for the weights above, re-measured for P7.
    # ``kw_only`` so subclass positional fields keep their order.
    metric: Optional[HilbertMetric] = field(
        default=None, repr=False, compare=False, kw_only=True,
    )

    def __post_init__(self) -> None:
        if self.metric is not None:
            # The object knows its own admission (a DenseMetric must span
            # the flattened leading block; a DiagonalMetric is as
            # permissive as the legacy array slot).
            self.metric.validate_for(self.shape)
        # ONE metric source, enforced pairwise over all three (P7 S2 —
        # illegal states unrepresentable; these are construction bugs,
        # not user input). ⚠ Until P7 the check lived under the
        # axes-early-return below, so the (dense, metric) arm was
        # structurally unreachable; each arm now has its own witness
        # (battery arms M10a/b/c).
        if self.axes is not None and self.inner_product_weights is not None:
            raise ValueError(
                f"space {self.name!r} carries BOTH per-axis measures and "
                f"dense inner_product_weights — one metric source only "
                f"(the axes own the measure on an axis-built space)"
            )
        if self.axes is not None and self.metric is not None:
            # CS4c step 6 item 6.2c-i: an axis-built space's metric is a
            # DERIVED object over its axes; an explicit object is admitted
            # only as the positioned OVERLAY of forms (one entry per axis,
            # in order, a form only on a measure-less axis's block) — the
            # home of a dense Gram on an axis-built head (ruling
            # R-6.2c-2). Anything else is two metric sources.
            _require_positioned_over(self.metric, self.axes, name=self.name)
        if self.inner_product_weights is not None and self.metric is not None:
            raise ValueError(
                f"space {self.name!r} carries BOTH dense "
                f"inner_product_weights and a metric object — one metric "
                f"source only (install the object OR the array, never both)"
            )
        if self.axes is None:
            return
        # The remaining axis-built guard: a shape that IS the axes'
        # concatenation.
        concat: tuple[int, ...] = ()
        for ax in self.axes:
            concat = concat + ax.shape
        if concat != self.shape:
            raise ValueError(
                f"space {self.name!r} shape {self.shape} does not equal "
                f"the concatenation {concat} of its axes' shapes"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctionSpace):
            return NotImplemented
        if other is self:
            return True
        # The identity flip (structural ``__eq__``, CS4c step 6, 2026-09-07):
        # an axis-built space IS its axis tuple. ``Axis.__eq__`` compares
        # (type, label, shape, kind, weights bytes) — never ``generator`` —
        # so two mints agree iff their axes' content agrees, and the derived
        # name stops being the carrier of that fact. An axis-built space is
        # never the same space as a hand-named one wearing its label; two
        # axes-less spaces keep the nominal ``(name, shape)`` identity —
        # content identity where the factory folds content into the name
        # (the composites and traces, CS4b S3), family + dimension on the
        # remaining family-tagged head (SpatialMomentSpace — metric-blind
        # by design until it becomes axis-built, item 6.2c-iii; the two
        # harmonic heads are axis-built since 6.2c-ii).
        if self.axes is not None or other.axes is not None:
            if self.axes is None or other.axes is None:
                return False
            return self.axes == other.axes
        return self.name == other.name and self.shape == other.shape

    def __hash__(self) -> int:
        # Consistent with ``__eq__`` by construction: the structural arm
        # hashes the axes tuple it compares, the nominal arm ``(name, shape)``.
        if self.axes is not None:
            return hash(self.axes)
        return hash((self.name, self.shape))

    def __repr__(self) -> str:
        return f"FunctionSpace({self.name!r}, shape={self.shape})"

    # ------------------------------------------------------------------
    # Axis composition (campaign 1, CS1)
    # ------------------------------------------------------------------

    @classmethod
    def of_axes(cls, *axes: Axis) -> "FunctionSpace":
        r"""Compose a space as the ordered product of its axes.

        The ONE composition mechanism of the axis doctrine: a space IS
        its axis tuple. Shape is the concatenation of the axes' shapes;
        the factor measures stay PER AXIS (``inner_product_weights`` is
        never populated — **no densification**, structurally: composing
        two 2000-point weighted axes stores 2 × 2000 weights, never the
        4 000 000-entry outer product); the metric machinery routes
        through the per-axis path.

        **The derived name — the readable label, and the carrier the
        axes-less composites fold.** The name is derived
        DETERMINISTICALLY and INJECTIVELY from the axes' structural
        content (label, shape, kind, measure bytes, subclass identity —
        via a content digest, never ``hash()``, so it is stable across
        processes). Until the identity flip (CS4c step 6, 2026-09-07)
        that injectivity was the identity BRIDGE: space identity was
        ``(name, shape)`` for every space, so different axis tuples had
        to mint different names to be different spaces. Since the flip
        an axis-built space compares by its ``axes`` tuple directly
        (:meth:`__eq__`), and the digest's injectivity is load-bearing
        only where a derived name is folded into an axes-less composite's
        own digest
        (:meth:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace.from_blocks`)
        and for what ``repr`` and the guard messages can tell apart. Two
        same-``ng`` energy axes with different partitions, or two
        same-shape spatial axes with different measures, compose into
        UNEQUAL spaces either way — the chartered "metric differences
        imply space differences" doctrine, now directly.

        Always returns a plain :class:`FunctionSpace` — an axis product
        is not a different *kind* of space (ruled Q-T4); invoking
        through a subclass does not change the return type.

        **The second composition, ``V * W``** (the ``*`` dunder →
        :class:`TensorProductSpace`), threads ``axes`` when every factor
        is axis-built — then it IS this composition, with ``factors``
        metadata on top — and otherwise positions the factor measures in
        a lazy :class:`~orpheus.numerics.metric.FactoredMetric`, per
        axis for an axis-built factor. ⚠ Until CS4c step 6 item 6.2a
        (2026-09-07) that second arm DENSIFIED the metric into an
        outer-product ``inner_product_weights`` tensor, bridging
        axis-borne measures through a densifier so no value was lost —
        `[M]` 18 state-sized tensors per 2-D windowed solve. The
        densifier retired with the dense arm; what keeps ``*`` alive is
        the axes-less angular head of the moment products, which item
        6.2c makes axis-built. New axis-aware code composes with
        ``of_axes``.
        """
        if not axes:
            raise ValueError(
                "of_axes needs at least one axis — a space with no "
                "factors has no index set"
            )
        shape: tuple[int, ...] = ()
        for ax in axes:
            shape = shape + ax.shape
        payload = b"".join(
            len(chunk).to_bytes(8, "little") + chunk
            for chunk in (ax._structural_bytes() for ax in axes)
        )
        digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
        readable = "*".join(f"{ax.label}{ax.shape}" for ax in axes)
        return FunctionSpace(
            name=f"{readable}#{digest}",
            shape=shape,
            inner_product_weights=None,
            axes=tuple(axes),
        )

    @property
    def has_coordinate_cone(self) -> bool | None:
        r"""Whether per-component positivity is meaningful on this space.

        Three-valued, deliberately:

        * ``True`` — axis-built, ALL factors ``NODAL``: components are
          point/cell values, the coordinate cone :math:`K = \{x \ge 0\}`
          is the physical positive cone, and per-component sign tests
          (``Field.cone_violations``) are meaningful.
        * ``False`` — axis-built, ANY factor ``MODAL``: components are
          expansion coefficients; a positive function may have negative
          coefficients, so a per-component sign test is MEANINGLESS and
          must be refused, not answered.
        * ``None`` — ``axes is None`` (legacy / not migrated): the
          question cannot be answered structurally; consumers keep their
          pre-CS1 behavior. Collapsing ``None`` into ``False`` would
          make the cone refusal fire on every legacy space in the tree.
        """
        if self.axes is None:
            return None
        return all(ax.kind is BasisKind.NODAL for ax in self.axes)

    # ------------------------------------------------------------------
    # Axis lookup — the public by-label accessor (un-weld arc S-1)
    # ------------------------------------------------------------------

    def _axis_index(self, label: str) -> int:
        """Position of the unique axis labeled ``label`` in :attr:`axes`.

        The one home of by-label axis resolution: the public
        :meth:`axis` accessor and the collapse-pair mint
        (:func:`orpheus.numerics.frame._collapse_pair`) both route
        through here, so the refusal vocabulary cannot drift between
        them.
        """
        if self.axes is None:
            raise TypeError(
                f"axis lookup {label!r}: {self!r} is not axis-built "
                f"(axes is None) — no named factors to look up. Compose "
                f"the space with FunctionSpace.of_axes."
            )
        hits = [i for i, ax in enumerate(self.axes) if ax.label == label]
        if len(hits) != 1:
            raise ValueError(
                f"axis lookup: label {label!r} names {len(hits)} axes "
                f"of {self!r} (have {[ax.label for ax in self.axes]}) — "
                f"need exactly one."
            )
        return hits[0]

    def axis(self, label: str) -> Axis:
        r"""The unique axis labeled ``label`` — by-label factor access.

        The public spelling of reads that used to route through carrier
        shape metadata: the space carries its axes, so the axis IS the
        metadata's home — ``space.axis("energy").shape[0]`` is the group
        count, ``space.axis("spatial").shape`` the spatial shape.
        Refuses a legacy name-built space (``axes is None``, TypeError)
        and a label naming zero or several axes (ValueError, naming the
        inventory) — the same vocabulary the collapse pair refuses in
        (shared resolver).
        """
        k = self._axis_index(label)
        axes = self.axes
        assert axes is not None  # narrowing only: _axis_index refused the None case
        return axes[k]

    # ------------------------------------------------------------------
    # Axis collapse — the retraction / section pair (CS4b S6.0b)
    # ------------------------------------------------------------------

    def _axis_collapse_pair(self, axis_label: str) -> "_AxisCollapsePair":
        r"""Memoized mint of the axis collapse pair (one mint per axis label).

        The pair is frame-induced at ONE site
        (:func:`orpheus.numerics.frame._collapse_pair` — the stage-2
        generator discipline: both inductions minted together, generator
        discarded) and cached in the frozen dataclass's ``__dict__`` (the
        F-0 ``basis_space`` pre-seed pattern), so both verbs share one
        mint and carriers that cache their spaces get warm operators for
        free — ``sn.angular_bulk_space.retraction("angular")`` costs the
        frame build once per carrier.
        """
        cache: dict[str, "_AxisCollapsePair"] = self.__dict__.setdefault(
            "_collapse_pairs", {}
        )
        if axis_label not in cache:
            from orpheus.numerics.frame import _collapse_pair

            cache[axis_label] = _collapse_pair(self, axis_label)
        return cache[axis_label]

    def retraction(self, axis_label: str) -> "AxisRetractionOperator":
        r"""Mint (memoized) the retraction :math:`R = \pi_*` over the named axis.

        The measure contraction of the axis's factor —
        :math:`(R\psi)(\cdot) = \sum_n w_n \psi(n, \cdot)`: fiber
        integration (the pushforward :math:`\pi_*` along the projection
        that forgets the axis) — the angular flux reduction when the
        axis is ``"angular"`` (`[M]` bit-identical with the shipped
        einsum, G6.5), the volume integral on ``"spatial"``.

        **Frame-induced**: the pair is the single-region indicator
        frame's output, minted once per axis at
        :func:`orpheus.numerics.frame._collapse_pair` and memoized on
        this space — both arrows come from that one mint (the stage-2
        generator's two-inductions clause; the frame itself is
        discarded, per the forgetful-map discipline). Born bound: domain
        is THIS space, codomain the same product with the axis dropped
        (remaining measures intact), so ``.H`` is the Hilbert adjoint
        out of the box — the pullback :math:`\pi^*`, the plain
        broadcast.

        The section satisfying :math:`R \circ E = \mathrm{id}` is
        :meth:`section` — a DIFFERENT arrow
        (:math:`R^\dagger = \Sigma w \cdot E`, `[M]` exact); the
        split epi/mono pair carries the canonical names (retraction /
        section, Mac Lane CWM §I.5) so the :math:`\Sigma w` convention
        is unspellable to swap (ERR-051).

        A typed :class:`~orpheus.numerics.axis.EnergyAxis` is REFUSED
        (collapse doctrine clause 2 — condensation owns that collapse);
        see the mint's docstring for the full admission table.
        """
        return self._axis_collapse_pair(axis_label).retraction

    def section(self, axis_label: str) -> "AxisSectionOperator":
        r"""Mint (memoized) the section :math:`E` of the axis retraction.

        The measure-normalized right inverse — the constant-along-the-
        axis field whose retraction reproduces the input,
        :math:`(E\phi)(n, \cdot) = \phi(\cdot)/\Sigma w`; DEFINED
        by :math:`R \circ E = \mathrm{id}` (`[M]` bit-exact; the
        canonical name for the right inverse of a retraction — split
        monomorphism). On the ``"angular"`` axis this is the
        isotropic-source projection :math:`Q/\Sigma w` broadcast across
        the ordinates. The divisor is frame-induced — the rank-one
        frame's 1×1 ``discrete_gram`` entry (the Parseval metric), read
        at the shared mint (see :meth:`retraction`). Born bound: domain
        is the marginal space, codomain THIS space.
        :math:`E \circ R` is the conditional expectation onto
        axis-constant functions.

        Refuses an axis whose SIGNED measure sums to zero: the rank-one
        Gram is singular — the frame has no canonical dual, so no
        section exists (the retraction over the same axis stays legal).
        """
        pair = self._axis_collapse_pair(axis_label)
        if pair.section is None:
            raise ValueError(
                f"FunctionSpace.section: axis {axis_label!r} of {self!r} "
                f"has zero total weight (a signed measure summing to 0) "
                f"— the section divides by Σw, so none exists. The "
                f"retraction over this axis is still legal."
            )
        return pair.section

    # ------------------------------------------------------------------
    # Inner product / norm
    # ------------------------------------------------------------------

    def inner_product(self, x: Carrier, y: Carrier) -> float:
        r"""Return :math:`\langle x, y \rangle`.

        The **surface** is carrier-generic: a space defines the inner
        product on its own element type. The base **realization**
        (:meth:`_diagonal_inner_product`) is the diagonal-weight form for
        the bare-array carrier; a space with a structured carrier
        overrides this surface with its own realization
        (:class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
        dispatches per direct-sum block).

        Parameters
        ----------
        x, y : Carrier
            Elements of this space (arrays of shape :attr:`shape` for
            the default realization).

        Returns
        -------
        float
            Scalar inner product.
        """
        return self._diagonal_inner_product(x, y)

    def _diagonal_inner_product(self, x: Any, y: Any) -> float:
        r"""The bare-array realization of :meth:`inner_product`.

        With diagonal weights ``w`` the inner product is the weighted
        sum :math:`\sum_i w_i \, x_i \, y_i`. Without weights it
        reduces to the Euclidean :math:`\sum_i x_i \, y_i`. The
        weights array is broadcast against ``x * y`` through the resolved
        :class:`~orpheus.numerics.metric.DiagonalMetric`
        (:func:`orpheus.numerics.metric._broadcast_leading` — the SAME
        leading-axis convention :meth:`_diagonal_apply_metric` uses, one
        home) — so a 1-D weight vector
        along (say) the ordinate axis acts on the full
        ``(n_ordinates, n_groups, *spatial)`` tensor without manual
        reshaping. Valid only for an ``NDArray``-carried space — the
        ``Any`` parameters are the realization/surface seam, not an
        open contract.

        Notes
        -----
        ⛔ **This routed through numpy's default (TRAILING) broadcast
        until 2026-08-04, while :meth:`_diagonal_apply_metric` used the
        LEADING convention — the same metric applied along different
        axes by two methods of one space.** The divergence is invisible
        whenever ``w.ndim >= x.ndim`` (the leading broadcast is then a
        no-op), which is every case the tree exercised, so it shipped.
        It bites the moment a space carries a leading-axis metric over
        an element with trailing axes:

        * non-square shapes — :meth:`inner_product` raised
          ``ValueError`` while :meth:`apply_metric` worked. `[M]`
          ``SphericalHarmonicSpace.from_L(3)`` on its PRODUCTION layout
          ``(L+1, 2L+1, ng, *spatial)`` did exactly this.
        * square shapes — both succeeded and **silently disagreed**
          (`[M]` 456 vs 552 on a ``(3, 3)`` probe), which is the
          dangerous half.

        Either way :math:`\langle Ax, y\rangle = \langle x, A^\dagger
        y\rangle` is false by construction, since ``AdjointOperator``
        builds :math:`A^\dagger = G_V^{-1}A^{\mathsf T}G_W` from
        :meth:`apply_metric` while the pairing that judges it came from
        here — the ERR-067 family, one layer down. The fix is
        bit-identical wherever the old path did not raise: it only ever
        pads ``w`` with trailing 1s, which is a no-op when ``w`` already
        spans every axis.
        """
        m = self._resolved_metric
        if m is None:
            return float(np.sum(x * y))
        # Σ (G⊙x)·y through the realization — for the resolved diagonal
        # case this is np.sum((w_b*x)*y), the SAME reduction tree as the
        # legacy np.sum(w*x*y) (left-to-right), so the reroute is
        # bit-identical; the matmul spelling y@(diag(w)@x) is NOT ([M] up
        # to 1792 ULP at n=15) and is deliberately unspellable here.
        return m.pairing(x, y)

    def norm(self, x: Carrier) -> float:
        r"""Return the induced :math:`L^2` norm
        :math:`\sqrt{\langle x, x \rangle}`.

        Carrier-generic through :meth:`inner_product` — valid unchanged
        for a structured-carrier space that overrides the inner product.
        """
        return float(np.sqrt(self.inner_product(x, x)))

    # ------------------------------------------------------------------
    # Metric application (Hilbert adjoint building blocks, Wave O / O.2b)
    # ------------------------------------------------------------------

    def apply_metric(self, x: Carrier) -> Carrier:
        r"""Apply the Hilbert metric :math:`G\odot x` (identity if Euclidean).

        Carrier-generic surface; the base realization
        (:meth:`_diagonal_apply_metric`) delegates to the space's resolved
        :class:`~orpheus.numerics.metric.HilbertMetric` — a diagonal
        weight broadcast against the leading axes of a bare-array ``x``,
        or a dense matrix on its flattened leading block. This is the
        building block :class:`~orpheus.numerics.operator.AdjointOperator`
        applies to the codomain before the transpose. Composite spaces
        (bulk :math:`\oplus` trace) OVERRIDE this to apply a per-block metric
        to a structured field (the Wave-O direct-sum adjoint).
        """
        return self._diagonal_apply_metric(x)

    def _diagonal_apply_metric(self, x: Any) -> Any:
        r"""The bare-array realization of :meth:`apply_metric`."""
        m = self._resolved_metric
        if m is None:
            return x
        return m.apply(x)

    def apply_inverse_metric(self, x: Carrier) -> Carrier:
        r"""Apply the Moore–Penrose pseudo-inverse metric :math:`G^{+}\odot x`.

        Carrier-generic surface (see :meth:`apply_metric`); the base
        realization is :meth:`_diagonal_apply_inverse_metric`, delegating
        to the resolved :class:`~orpheus.numerics.metric.HilbertMetric`.
        For the diagonal realization this is
        ``(1/G)⊙x`` where ``G ≠ 0``, and ``0`` on the metric's null space
        (``G = 0`` — e.g. the tangential partial-current trace slots where
        ``|Ω·n| = 0``). The pseudo-inverse is exact for the Hilbert adjoint:
        the null-space components carry zero ``⟨·,·⟩_G`` weight and are zero
        in any matvec output by construction. Identity if Euclidean. Applied
        to the domain after the transpose. For a strictly-positive metric
        (e.g. the angular quadrature weights ``w_n``) this is plain ``x/G``,
        so the spherical-harmonic adjoint path stays bit-identical.
        """
        return self._diagonal_apply_inverse_metric(x)

    def _diagonal_apply_inverse_metric(self, x: Any) -> Any:
        r"""The bare-array realization of :meth:`apply_inverse_metric`."""
        m = self._resolved_metric
        if m is None:
            return x
        return m.apply_inverse(x)

    @property
    def _resolved_metric(self) -> Optional[HilbertMetric]:
        r"""The metric SOURCE resolved to its realization, PER CALL.

        Resolution order (CS4c step 6 item 6.2c-i, ruling R-6.2c-2): an
        axis-built space's metric is DERIVED from its axes as a
        :class:`~orpheus.numerics.metric.FactoredMetric` — one
        :class:`~orpheus.numerics.metric.DiagonalMetric` per weighted axis
        on that axis's block, ``None`` on a counting-measure axis — with
        the space's positioned OVERLAY of forms merged in (a dense Gram's
        :class:`~orpheus.numerics.metric.DenseMetric` on the block of an
        axis that carries no measure; the guard admitted it), no object at
        all when every block is Euclidean (:func:`_axes_metric`); else an
        explicit ``metric`` object on an axes-less space; else a legacy
        ``inner_product_weights`` array resolves to a ``DiagonalMetric``;
        ``None`` means Euclidean and the verbs short-circuit without an
        object. ONE realization for every source:
        :meth:`DiagonalMetric.apply_block` is operation-for-operation the
        per-axis reshape-and-multiply the retired ``_apply_axes_weights``
        performed inline (`[M]` bit-identical on every axis-built space —
        gated in ``tests/numerics/test_axis_metric_is_a_derived_object.py``),
        and the legacy diagonal arm is operation-for-operation the arms
        that used to live inline in the three ``_diagonal_*`` realizations.

        ⛔ Deliberately NOT a ``cached_property``, and the reason is a
        measured red: the mutation-battery idiom mutates a frozen
        space's weight FIELD in place (``object.__setattr__``), and the
        adjoint-certification propagation probes assert the mutation
        reaches the metric surface at the next read. `[M]` 2026-08-30
        (P7 exit gate): a cached resolution served the stale
        ``DiagonalMetric`` and
        ``test_gsd_metric_drop_is_k_blind_but_vector_red``'s own
        ``after == before`` probe reddened — the pre-P7 semantics read
        the field per call, and every such battery relies on it. The
        wrapper is a tiny frozen object; the numpy work dominates.
        """
        if self.axes is not None:
            return _axes_metric(self.axes, self.metric)
        if self.metric is not None:
            return self.metric
        if self.inner_product_weights is not None:
            return DiagonalMetric(self.inner_product_weights)
        return None

    # ------------------------------------------------------------------
    # Space algebra (Depth B step D-B)
    # ------------------------------------------------------------------

    def __mul__(self, other: "FunctionSpace") -> "TensorProductSpace":
        r"""Return the tensor product :math:`V \otimes W` of this space
        with ``other``.

        Implements ``V * W`` per grand-report §6.1: the resulting
        :class:`TensorProductSpace` carries the concatenated shape
        ``self.shape + other.shape``, the factor measures positioned per
        block (concatenated axes when every factor is axis-built, a lazy
        :class:`~orpheus.numerics.metric.FactoredMetric` otherwise —
        never a stored outer product, since CS4c step 6 item 6.2a), and
        the multiplied units. Associative on its inputs:
        ``(A * B) * C`` and ``A * (B * C)`` both produce a flat
        3-factor :class:`TensorProductSpace`.

        Loadbearing for the Wave T tensor-network rewires per the
        grand report §15.1 (streaming as
        :math:`L = \sum_{\text{axis}} D_{\text{axis}} \otimes \Omega_{\text{axis}} \otimes I_g`)
        and §16A.10 (boundary as
        :math:`B = G_{\text{patch}} \otimes K_\omega \otimes K_g`).
        See ``.claude/plans/depth_b_field_on_function_space.md`` §6
        step D-B for the design.
        """
        if not isinstance(other, FunctionSpace):
            return NotImplemented
        self_factors = (
            self.factors if isinstance(self, TensorProductSpace) else (self,)
        )
        other_factors = (
            other.factors if isinstance(other, TensorProductSpace) else (other,)
        )
        return TensorProductSpace.from_factors(self_factors + other_factors)

    @property
    def riesz_lower(self) -> "RieszLowerOperator":
        r"""The Riesz LOWERING leg :math:`\flat : V \to V^*` — ``G x``.

        The inner product's isomorphism onto the dual, as a first-class
        arrow (CS4c R2): delegates to :meth:`apply_metric`, so the metric
        arithmetic stays single-sourced; what the leg adds is the
        bookkeeping (``domain = V``, ``codomain = V.dual()``) and an
        individually-testable, individually-MUTABLE seam — the
        codomain-side factor of the Hilbert adjoint
        ``A* = domain.riesz_raise ∘ A.dual() ∘ codomain.riesz_lower``.

        PRIMAL spaces only — the leg's constructor refuses a
        :class:`DualSpace` (which deliberately carries its primal's
        metric, so a dual-side ♭ would be the G² trap; see the leg class).
        """
        from orpheus.numerics.operator import RieszLowerOperator

        return RieszLowerOperator(self)

    @property
    def riesz_raise(self) -> "RieszRaiseOperator":
        r"""The Riesz RAISING leg :math:`\sharp : V^* \to V` — ``G⁺ f``.

        The mirror of :attr:`riesz_lower` (Moore–Penrose by the metric
        family's doctrine): delegates to :meth:`apply_inverse_metric`;
        ``domain = V.dual()``, ``codomain = V``. The domain-side factor
        of the Hilbert adjoint. The round trip
        ``riesz_raise ∘ riesz_lower`` is :math:`P_{\mathrm{range}(G)}` —
        the identity iff the metric is strictly positive (a singular
        trace block projects its tangential slots to zero).
        """
        from orpheus.numerics.operator import RieszRaiseOperator

        return RieszRaiseOperator(self)

    def dual(self) -> "FunctionSpace":
        r"""Return the dual space :math:`V^*`.

        Under L²-Riesz identification (the standard ORPHEUS setting
        where every :class:`FunctionSpace` carries an inner product),
        :math:`V^*` is isomorphic to :math:`V` itself with a covariance
        tag for bra-ket-style composition tracking. The dual carries
        the same shape, weights, and units as the primal; its
        ``primal`` attribute holds a reference back.

        The return type is :class:`FunctionSpace`, not :class:`DualSpace`,
        because ``dual`` is reflexive: applied to a :class:`DualSpace` it
        returns the primal (:math:`V^{**} = V`), which is any space.

        Used by the Hilbert-adjoint machinery
        (:meth:`~orpheus.numerics.operator.LinearOperator.adjoint`,
        ``A.H``) to track which spaces are codomain-sourced vs
        domain-sourced through operator composition.
        """
        return DualSpace.of(self)


# ---------------------------------------------------------------------------
# Tensor-product and dual space (Depth B step D-B)
# ---------------------------------------------------------------------------


def _axis_entries(
    axes: tuple[Axis, ...],
    positioned: Optional[HilbertMetric] = None,
) -> list[tuple[tuple[int, ...], DiagonalMetric | DenseMetric | None]]:
    r"""One positioned metric entry per AXIS — the block form of the axis
    doctrine's "the measure lives per axis" — with the space's positioned
    OVERLAY of forms merged in.

    A weighted axis contributes a
    :class:`~orpheus.numerics.metric.DiagonalMetric` carrying its measure on
    its own index block; a counting-measure axis (``weights is None``, the
    canonical spelling) contributes ``None`` — no pass over that block —
    unless ``positioned`` (the space's admitted overlay: one entry per
    axis, a form only where the axis carries no measure) places a FORM on
    that block, which then IS the entry: the dense Gram no diagonal
    measure can spell (ruling R-6.2c-2, 2026-09-08 — *each axis supplies
    its diagonal block from its measure; a dense-Gram head supplies a
    positioned DenseMetric on its block*). The guard makes shadowing
    unspellable, so the merge never has to choose.
    :meth:`DiagonalMetric.apply_block` performs exactly the explicit
    reshape-and-multiply (leading 1s for the preceding axes' ranks, the
    axis shape, trailing 1s for the rest — exact for interior axes, never
    materializing the outer product; the ERR-067-family divergence is
    unspellable here) that ``FunctionSpace._apply_axes_weights`` performed
    inline until CS4c step 6 item 6.2c-i retired it into this one
    realization.
    """
    forms: tuple[DiagonalMetric | DenseMetric | None, ...] = (
        (None,) * len(axes)
        if not isinstance(positioned, FactoredMetric)
        else tuple(form for _, form in positioned.entries)
    )
    return [
        (
            ax.shape,
            form if form is not None
            else (None if ax.weights is None else DiagonalMetric(ax.weights)),
        )
        for ax, form in zip(axes, forms)
    ]


def _axes_metric(
    axes: tuple[Axis, ...], positioned: Optional[HilbertMetric] = None,
) -> Optional[FactoredMetric]:
    r"""The metric an axis-built space DERIVES from its axes (CS4c step 6
    item 6.2c-i): a lazy :class:`~orpheus.numerics.metric.FactoredMetric`
    over :func:`_axis_entries` (the positioned overlay merged in), or
    ``None`` when every block is Euclidean (the default — no object, no
    allocation).
    Rebuilt per resolution (a handful of frozen wrappers around the axes'
    own arrays; nothing is copied), so an in-place weight mutation reaches
    the metric surface at the next read exactly as the dense-slot arm's
    does.
    """
    entries = _axis_entries(axes, positioned)
    if all(entry is None for _, entry in entries):
        return None
    return FactoredMetric(tuple(entries))


def _require_positioned_over(
    metric: HilbertMetric, axes: tuple[Axis, ...], *, name: str,
) -> None:
    r"""Admit an explicit metric object on an AXIS-BUILT space only as the
    positioned OVERLAY of forms — a
    :class:`~orpheus.numerics.metric.FactoredMetric` with exactly one entry
    per axis, in order, each on that axis's block, a form ONLY on a block
    whose axis carries no measure, and never a diagonal one.

    That is the one shape in which a form no diagonal measure can spell —
    a dense Gram's :class:`~orpheus.numerics.metric.DenseMetric` on a
    moment head's block — lives beside per-axis measures without becoming
    a second metric source (ruling R-6.2c-2, 2026-09-08: *a Gram is a FORM,
    never on* ``Axis.weights``; the space's metric is the derived object
    and a dense block is positioned ON it). Everything else is refused by
    name, each a second spelling of one measure: a diagonal array spanning
    the whole shape or a factored object whose blocks do not follow the
    axes (the derived object would silently disagree with it — `[M]` a
    relaxed guard alone left the per-axis short-circuit reading the axes
    and ignoring the object by ``max|Δ| = 14.6`` on a dense-Gram head); a
    positioned :class:`~orpheus.numerics.metric.DiagonalMetric` (a diagonal
    measure is the axis's own to carry); a form on an axis that ALSO
    carries a measure (two sources on one block — the overlay merge would
    have to pick one and the loser would be dead data).
    """
    if not isinstance(metric, FactoredMetric):
        raise ValueError(
            f"space {name!r} carries per-axis measures and a "
            f"{type(metric).__name__} — an axis-built space's metric is "
            f"DERIVED from its axes; an explicit object must be a "
            f"FactoredMetric positioned over them (one entry per axis)."
        )
    blocks = tuple(block for block, _ in metric.entries)
    axis_blocks = tuple(ax.shape for ax in axes)
    if blocks != axis_blocks:
        raise ValueError(
            f"space {name!r}: the metric object's blocks {blocks} do not "
            f"follow the axes' blocks {axis_blocks} — a positioned object "
            f"carries exactly one entry per axis, in order."
        )
    for k, ((_, form), ax) in enumerate(zip(metric.entries, axes)):
        if form is None:
            continue
        if isinstance(form, DiagonalMetric):
            raise ValueError(
                f"space {name!r}: entry {k} (axis {ax.label!r}) positions a "
                f"DiagonalMetric — a diagonal measure is the axis's own to "
                f"carry (Axis.weights); only a FORM no measure can spell (a "
                f"dense Gram) is positioned on the derived object."
            )
        if ax.weights is not None:
            raise ValueError(
                f"space {name!r}: axis {ax.label!r} carries a measure AND "
                f"entry {k} positions a {type(form).__name__} on its block — "
                f"two metric sources on one block; a form REPLACES the "
                f"measure (mint that axis with weights=None)."
            )


def _positioned_forms(
    factors: tuple["FunctionSpace", ...],
) -> Optional[FactoredMetric]:
    r"""The positioned OVERLAY an all-axis-built product carries beside its
    concatenated axes: each factor's admitted overlay (one entry per axis)
    in order, ``None`` on every axis of a factor that carries none — or no
    object at all when no factor carries a form. The product then derives
    exactly the entries its factors derive (item 6.2c-i).
    """
    entries: list[
        tuple[tuple[int, ...], DiagonalMetric | DenseMetric | None]
    ] = []
    for f in factors:
        assert f.axes is not None  # narrowing only: the caller's arm
        if isinstance(f.metric, FactoredMetric):
            entries.extend(f.metric.entries)
        else:
            entries.extend((ax.shape, None) for ax in f.axes)
    if all(form is None for _, form in entries):
        return None
    return FactoredMetric(tuple(entries))


def _tensor_product_factored_metric(
    factors: tuple["FunctionSpace", ...],
) -> Optional[FactoredMetric]:
    r"""Assemble a tensor product's metric as a lazy per-BLOCK product —
    the one non-axis arm of ``*`` (CS4c step 6 item 6.2a, 2026-09-07).

    One positioned entry per block, and the Kronecker product is never
    materialized:

    * an **axis-built** factor contributes one entry per AXIS — a
      :class:`~orpheus.numerics.metric.DiagonalMetric` carrying the
      axis measure on that axis's block, ``None`` for a counting-measure
      axis — so the product applies exactly the arithmetic
      :meth:`FunctionSpace._apply_axes_weights` applies on the factor
      itself (:meth:`~orpheus.numerics.metric.DiagonalMetric.apply_block`
      is operation-for-operation that reshape-and-multiply: bit-identical
      on an axis block by construction);
    * a factor carrying a metric **object** rides verbatim — a nested
      :class:`~orpheus.numerics.metric.FactoredMetric` flattens (a
      product of products is one product), a
      :class:`~orpheus.numerics.metric.DenseMetric` positions on the
      factor's block (the Parseval-dressed head of a dense-Gram frame);
    * a **dense-slot leaf** (``inner_product_weights`` set, no axes — the
      spherical-harmonic / Legendre heads, the trace spaces, hand-built
      legacy spaces) becomes a ``DiagonalMetric`` on its block;
    * a **Euclidean** factor contributes ``None`` (no pass over its block).

    Returns ``None`` when every block is Euclidean, preserving the
    Euclidean default (no object, no allocation).

    **History.** Until 6.2a ``*`` had a DENSE arm that formed
    :math:`w_1 \otimes w_2 \otimes \cdots` as one weights tensor on every
    mint with an axes-less factor — `[M]` 18 ``(L+1, 2L+1, ng, nx, ny)``
    tensors per 2-D windowed solve at ``max_inner = 6``,
    ``2·max_inner + 6`` in general — and this builder (the P7
    dense-factor arm) was unreachable on every SN path (no shipped factor
    carried a metric object) *and* itself densified an axis-built factor's
    measure. The factored application agrees with the retired outer
    product to `[M]` 2 ULP worst over 1600 draws (200 seeds × 8
    (geometry × L) rows — one extra rounding, reduction depth +1; the
    ``vv-principles`` bit-identity criterion 3), gated at ``nulp = 4`` in
    ``tests/numerics/test_tensor_product_metric_is_factored.py``.
    """
    entries: list[
        tuple[tuple[int, ...], DiagonalMetric | DenseMetric | None]
    ] = []
    for f in factors:
        m = f.metric
        if f.axes is not None:
            # Per AXIS, never per factor: an axis-built factor's measure
            # stays exactly where the axis doctrine put it, its positioned
            # overlay of forms merged in (the same entries the factor
            # DERIVES for itself — one spelling).
            entries.extend(_axis_entries(f.axes, m))
        elif m is not None:
            if isinstance(m, FactoredMetric):
                entries.extend(m.entries)
            elif isinstance(m, (DiagonalMetric, DenseMetric)):
                entries.append((f.shape, m))
            else:
                # Type-narrowing refusal: the positioned application is
                # defined on the leaf realizations; anything else would
                # die one call later with a worse message.
                raise TypeError(
                    f"a tensor product can position only diagonal/dense "
                    f"factor metrics, got {type(m).__name__}"
                )
        elif f.inner_product_weights is not None:
            entries.append(
                (
                    f.shape,
                    DiagonalMetric(
                        np.ascontiguousarray(
                            np.broadcast_to(f.inner_product_weights, f.shape)
                        )
                    ),
                )
            )
        else:
            entries.append((f.shape, None))
    if all(entry is None for _, entry in entries):
        return None
    return FactoredMetric(tuple(entries))


@dataclass(frozen=True)
class TensorProductSpace(FunctionSpace):
    r"""A function space that decomposes as
    :math:`V = V_1 \otimes V_2 \otimes \cdots \otimes V_k`.

    The tensor-product structure makes algebraic identities of operators
    on this space (adjoint distributivity, composition distributivity,
    representation polymorphism) checkable at the type level. See grand-
    report §5.3, §15, §32.4 for the L1 motivation; see
    ``.claude/plans/wave_t_tensor_network.md`` for the production
    consumers being wired in Wave T.

    Construction
    ------------
    Two equivalent paths:

    * **Operator-algebra dispatch** — ``A * B`` where ``A`` and ``B``
      are :class:`FunctionSpace` instances returns a
      :class:`TensorProductSpace`. The dunder is associative on its
      inputs (``(A * B) * C`` flattens to a 3-factor product, never
      nests).
    * **Explicit factory** — :meth:`from_factors` with a tuple of
      factor spaces.

    Notes
    -----
    The class is **frozen**. Identity is the inherited
    ``(name, shape)`` tuple, where ``name`` and
    ``shape`` are derived from the factors. Two
    :class:`TensorProductSpace` instances with the same factor sequence
    compare equal even if reached via different composition paths.

    The ``factors`` field is metadata that supports introspection (e.g.,
    operator-algebra factor matching for ``(A & B) ∘ (C & D)`` rewriting)
    and is not part of the identity.

    Parameters
    ----------
    factors : tuple[FunctionSpace, ...]
        The factor spaces, in order. Should have ``len >= 2`` for a
        meaningful tensor product; trivial 1-factor TensorProductSpaces
        are permitted by the dataclass but produced only via the
        :meth:`__mul__` flattening edge cases.
    """

    factors: tuple["FunctionSpace", ...] = field(default=(), compare=False, repr=False)

    # ── Equality / hashing inherited from FunctionSpace ───────────────
    #
    # The @dataclass(frozen=True) decorator would otherwise auto-generate
    # __eq__ that compares every field — including the ndarray
    # ``inner_product_weights`` which raises "truth value ambiguous" at
    # use time. Explicit delegation restores the base identity (the ``axes``
    # tuple when axis-built, ``(name, shape)`` otherwise — one body, eight
    # delegating names across the space family). ``factors`` is already
    # excluded from compare via the field-level ``compare=False``.

    def __eq__(self, other: object) -> bool:
        return FunctionSpace.__eq__(self, other)

    def __hash__(self) -> int:
        return FunctionSpace.__hash__(self)

    def find_factor[T: "FunctionSpace"](self, factor_type: type[T]) -> T:
        r"""Return the (first) tensor factor that is an instance of ``factor_type``.

        The tree query the moment-carrier fields rely on to recover their
        typed factor from a composed space — e.g.
        ``space.find_factor(SphericalHarmonicSpace).L`` (on a full-sphere
        rule; the head factor's own ``L`` in general) recovers the
        angular truncation order, and
        ``space.find_factor(SpatialMomentSpace).per_axis`` recovers the
        spatial-moment basis size — without the consumer having to know
        the factor's position in the product (issue #207). The factories
        compose factors in a fixed order, but consumers query by TYPE, not
        index, so the layout can change without breaking the query.

        Parameters
        ----------
        factor_type : type[T]
            The :class:`FunctionSpace` subclass to search for among
            :attr:`factors`. The return is typed AS this class (generic),
            so ``find_factor(SphericalHarmonicSpace).L`` /
            ``find_factor(SpatialMomentSpace).per_axis`` type-resolve — the
            method's reason to exist is the typed bridge from a composed
            space back to its factor's metadata.

        Returns
        -------
        T
            The first factor that ``isinstance(factor, factor_type)``, typed
            as ``factor_type``.

        Raises
        ------
        KeyError
            If no factor matches ``factor_type`` — an explicit failure
            (the query is a structural assertion: the caller believes the
            composed space carries this factor).
        """
        for f in self.factors:
            if isinstance(f, factor_type):
                return f
        raise KeyError(
            f"TensorProductSpace {self.name!r} has no factor of type "
            f"{factor_type.__name__}; factors are "
            f"{[type(f).__name__ for f in self.factors]!r}."
        )

    @classmethod
    def from_factors(
        cls, factors: tuple["FunctionSpace", ...],
    ) -> "TensorProductSpace":
        r"""Construct a :class:`TensorProductSpace` from a tuple of factor
        spaces.

        Derives:
        * ``name`` from ``" ⊗ ".join(f.name for f in factors)``
        * ``shape`` from concatenated factor shapes
        * the metric: concatenated ``axes`` when every factor is
          axis-built, else a lazy
          :class:`~orpheus.numerics.metric.FactoredMetric` positioned
          per block (``None`` if every block is Euclidean) —
          ``inner_product_weights`` is never populated by a product
        """
        if len(factors) < 2:
            raise ValueError(
                f"TensorProductSpace.from_factors requires at least 2 "
                f"factors; got {len(factors)}"
            )
        name = " ⊗ ".join(f.name for f in factors)
        shape: tuple[int, ...] = ()
        for f in factors:
            shape = shape + f.shape
        # Two arms, and neither materializes a weights tensor (CS4c step
        # 6 item 6.2a retired the dense arm and its densifier bridge):
        # * axis threading (CS1, gate B7) — when EVERY factor carries an
        #   axes record, the product's record is the concatenation and
        #   the measure rides the per-axis path; a factor's positioned
        #   OVERLAY of forms (a dense-Gram head, 6.2c-i) rides beside it,
        #   concatenated per axis, so the product derives the same
        #   entries its factors do;
        # * the factored arm — an axes-less factor on either side leaves
        #   ``axes=None`` (never fabricate an axis for a space that did
        #   not declare one) and the product carries a lazy
        #   FactoredMetric positioned per block: per AXIS for an
        #   axis-built factor, per factor for a dense-slot leaf, verbatim
        #   for a metric object (P7 S2: dropping an object would silently
        #   revert the product to Euclidean on that factor, a VALUE bug —
        #   [M] 33.0 where G ⊗ w gives 109.0, on the harmonic-frame mint
        #   path).
        factor_axes = [f.axes for f in factors]
        metric: Optional[HilbertMetric] = None
        if all(fa is not None for fa in factor_axes):
            axes: Optional[tuple[Axis, ...]] = tuple(
                ax for fa in factor_axes if fa is not None for ax in fa
            )
            metric = _positioned_forms(factors)
        else:
            axes = None
            metric = _tensor_product_factored_metric(factors)
        return cls(
            name=name,
            shape=shape,
            inner_product_weights=None,
            axes=axes,
            metric=metric,
            factors=factors,
        )

    def __repr__(self) -> str:
        return f"TensorProductSpace({self.name!r}, shape={self.shape})"


@dataclass(frozen=True)
class DualSpace(FunctionSpace):
    r"""The dual :math:`V^*` of a :class:`FunctionSpace`.

    Under L²-Riesz identification (the standard ORPHEUS setting where
    every :class:`FunctionSpace` carries an inner product),
    :math:`V^*` is isomorphic to :math:`V` itself but carries a
    covariance tag that the operator-algebra layer reads through to
    track which spaces participate as bras vs. kets in composition
    chains. The dual carries the same ``shape`` and
    ``inner_product_weights`` as the primal; the ``primal`` field is
    the introspection link.

    Used by the Hilbert-adjoint machinery
    (:meth:`~orpheus.numerics.operator.LinearOperator.adjoint`,
    ``A.H``) — taking ``A.H`` swaps domain ↔ codomain AND flips both
    to their duals, so the adjoint's domain is the original's codomain
    dual.

    Notes
    -----
    ``V.dual().dual() == V`` is enforced by :meth:`of` recognising a
    :class:`DualSpace` argument and returning its primal (idempotency).

    Parameters
    ----------
    primal : FunctionSpace
        The primal space :math:`V` of which this is the dual.
    """

    # Required (a DualSpace WITHOUT its primal is an illegal state);
    # kw_only sidesteps the inherited-defaults field-ordering rule.
    primal: "FunctionSpace" = field(kw_only=True, compare=False, repr=False)

    # ── Equality / hashing inherited from FunctionSpace ───────────────
    #
    # Same rationale as :class:`TensorProductSpace.__eq__` — auto-
    # generated dataclass __eq__ would compare ndarray weights and raise
    # "truth value ambiguous". Explicit delegation to FunctionSpace.

    def __eq__(self, other: object) -> bool:
        return FunctionSpace.__eq__(self, other)

    def __hash__(self) -> int:
        return FunctionSpace.__hash__(self)

    @classmethod
    def of(cls, primal: "FunctionSpace") -> "FunctionSpace":
        r"""Construct the dual of ``primal``.

        Idempotent: ``of(of(V)) == V`` (returns the primal of a passed
        :class:`DualSpace`, never wraps twice).
        """
        if isinstance(primal, DualSpace):
            return primal.primal
        return cls(
            name=f"{primal.name}*",
            shape=primal.shape,
            inner_product_weights=primal.inner_product_weights,
            # The dual carries the SAME metric as the primal (L²-Riesz),
            # so an axis-built primal's dual threads the axes record —
            # dropping it would silently strip the measure (CS1) — and a
            # metric OBJECT threads too (P7 S2): [M] before this line the
            # dual of a dense-metric space read the plain Euclidean
            # pairing (4.5 where the primal reads 23.3).
            axes=primal.axes,
            metric=primal.metric,
            primal=primal,
        )

    def __repr__(self) -> str:
        return f"DualSpace({self.name!r}, shape={self.shape})"


# ---------------------------------------------------------------------------
# Pre-populated common-space factories
# ---------------------------------------------------------------------------
#
# These are *factory functions*, not module-level singletons, because
# the shape of an angular-flux space depends on the mesh dimensions —
# every solver instance carries its own (n_cells, n_ordinates, n_groups)
# triple. Factories give the caller named, well-typed construction
# sites without forcing premature commitment to a global instance.


def angular_flux_space(
    n_cells: int,
    n_ordinates: int,
    n_groups: int,
    *,
    quadrature_weights: NDArray | None = None,
) -> FunctionSpace:
    r"""Construct the angular-flux space for an SN solve.

    Shape is ``(n_cells, n_ordinates, n_groups)``. When
    ``quadrature_weights`` is provided, it is broadcast along the
    ordinate axis as the inner-product metadata so the canonical
    angular inner product
    :math:`\langle \psi, \varphi \rangle_\Omega = \sum_n w_n \,
    \psi_n \, \varphi_n` (summed over cells / groups too) becomes
    :meth:`FunctionSpace.inner_product`.

    Parameters
    ----------
    n_cells, n_ordinates, n_groups : int
        Tensor dimensions.
    quadrature_weights : NDArray, optional
        Shape ``(n_ordinates,)`` quadrature weights along the
        ordinate axis. Reshaped to ``(1, n_ordinates, 1)`` so it
        broadcasts against the full angular-flux tensor.
    """
    weights: NDArray | None = None
    if quadrature_weights is not None:
        w = np.asarray(quadrature_weights)
        if w.shape != (n_ordinates,):
            raise ValueError(
                f"quadrature_weights must have shape ({n_ordinates},), "
                f"got {w.shape}"
            )
        weights = w.reshape(1, n_ordinates, 1)
    return FunctionSpace(
        name="angular_flux",
        shape=(n_cells, n_ordinates, n_groups),
        inner_product_weights=weights,
    )


def scalar_flux_space(n_cells: int, n_groups: int) -> FunctionSpace:
    r"""Construct the scalar-flux space.

    Shape is ``(n_cells, n_groups)``. No inner-product weights — the
    canonical inner product on scalar-flux space is the Euclidean
    sum (or, equivalently, the volume-weighted L² inner product when
    the cell volumes are absorbed into the operator that produced
    :math:`\phi`; see :meth:`Mesh1D.volume_measure` for the
    volume-weighted variant).
    """
    return FunctionSpace(
        name="scalar_flux",
        shape=(n_cells, n_groups),
    )


