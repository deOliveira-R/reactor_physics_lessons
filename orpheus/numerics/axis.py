r"""Space-factor axes — the generators of axis-composed function spaces.

An **axis** is one tensor factor of a function space: the value object
recording *(index shape, factor measure, basis kind)* as its structural
identity, plus — since CS5 — *generator provenance* (which object minted
it), deliberately outside that identity.
Spaces are ordered products of axes (``FunctionSpace.of_axes``); the axis is
the unit the composition machinery reasons about — partitions, collapses,
frames, and (later) ⊕-lifts act **per axis**, never on an anonymous
position of a monolithic shape tuple.

The five slots, precisely
=========================

* ``shape`` — the index set of this factor, rank ≥ 1. Rank > 1 is
  admissible (a spherical-harmonic axis is ``(L+1, 2L+1)``; a rank-``d``
  spatial axis is a legal design choice for CS2).
* ``weights`` — the **factor measure** over exactly ``shape``.
  ``None`` **is the counting measure, deliberately and always**: an axis
  has no "unbound" state, so the two-state ambiguity of the legacy
  ``FunctionSpace.inner_product_weights`` (``None`` = "no canonical
  quadrature" *or* "Euclidean") cannot arise on this type.
* ``kind`` — :class:`BasisKind`. ``NODAL`` factors carry a coordinate
  cone (per-component positivity is meaningful); ``MODAL`` factors do not
  (a spectral coefficient may be negative for a positive function).
* ``generator`` — **provenance, never identity** (CS5): the object that
  minted this axis — a :class:`~orpheus.numerics.measure.DiscreteMeasure`
  or :class:`~orpheus.numerics.quadrature.directional.Quadrature` for
  NODAL factors, a :class:`~orpheus.numerics.basis.base.Basis` for MODAL
  ones — or, on a moment HEAD a frame has dressed with its Parseval
  metric, the :class:`~orpheus.numerics.frame.FrameBase` that dressed it
  (the Stage-2 generator: the pairing basis ⊗ measure induces the
  coefficient space's metric, and only the frame can re-dress the head at
  another order; CS4c step 6 item 6.2c-ii) — or ``None`` where no
  generator object exists (the counting axis). An axis is a **forgetful
  map** from its generator (it keeps the
  weights and drops the nodes); the accessor lets a consumer recover the
  un-forgotten data (direction cosines, level structure) THROUGH the
  space instead of being handed the generator separately. Deliberately
  EXCLUDED from :meth:`_identity_key`: two axes with identical
  structural content are the same axis whatever instance produced them,
  so content-equal but distinct-instance generators (the #403 hazard)
  never reach axis equality — and never perturb the ``of_axes``
  space-name digest, whose injectivity rides the same key.
* identity — **structural, per subclass** (see below).

Canonical storage — one spelling per measure (ruled 2026-08-20)
===============================================================

Two construction rulings make the measure's identity unambiguous:

* **All-ones weights collapse to ``None`` at construction.** The counting
  measure has ONE spelling and therefore one identity; without this,
  ``weights=None`` and ``weights=np.ones(shape)`` would be the same
  measure with unequal identities — a twin exactly of the shape the
  fresh-``EnergyGrid``-per-access trap takes at the grid layer.
* **Weights are canonicalized as ``w + 0.0`` and stored read-only.**
  ``-0.0`` and ``+0.0`` are one measure and must be one identity/byte
  pattern; the addition also guarantees a defensive copy, so mutating the
  caller's array can never move an axis's hash after it has been used as
  a dict key.

Non-finite weights are **refused** (a measure's weights are finite
numbers). There is deliberately **no non-negativity guard**: CS2's
quadrature axes legally carry signed weights (e.g. level-symmetric
families with negative weights), and the axis is the wrong layer to
outlaw them.

Identity — structural per subclass, from day one
================================================

Equality and hash compare the *structural content* — ``(type, label,
shape, kind, weights bytes)`` plus each subclass's own identity data —
never object identity and never a subset. Two axes that differ only in
measure are **different axes** (the collapse doctrine's "a genuine
one-cell slab keeps its axis with weight V ≠ 1, distinguished from the
quotient point by measure"); an :class:`EnergyAxis` never equals a
generic :class:`Axis` carrying the same field tuple. This identity is
what ``FunctionSpace.of_axes`` derives *space names* from, and — since the
identity flip (CS4c step 6, 2026-09-07) — what ``FunctionSpace.__eq__`` reads
DIRECTLY on an axis-built space: axis identity IS space identity there.

Layering note: this module is model-independent numerics. It consumes an
:class:`~orpheus.data.energy_grid.EnergyGrid` only through its surface
(``edges``, ``n_groups``) under ``TYPE_CHECKING`` — the runtime
dependency direction stays ``data → numerics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Iterable

    from orpheus.data.energy_grid import EnergyGrid
    from orpheus.data.macro_xs.mixture import Mixture
    from orpheus.numerics.basis.base import Basis
    from orpheus.numerics.frame import FrameBase
    from orpheus.numerics.measure import DiscreteMeasure
    from orpheus.numerics.quadrature.directional import Quadrature

__all__ = ["Axis", "BasisKind", "EnergyAxis"]

#: The generator kinds :meth:`Axis.generator_as` can narrow to.
_G = TypeVar("_G")


@unique
class BasisKind(Enum):
    """The basis character of a space factor.

    ``NODAL`` — components are point/cell VALUES (indicator-like basis):
    the factor carries a coordinate cone, so per-component positivity is a
    meaningful predicate. ``MODAL`` — components are expansion
    COEFFICIENTS (spectral basis): no coordinate cone; a positive function
    may have negative coefficients, so per-component sign tests are
    meaningless and must be refused, not answered.
    """

    NODAL = "nodal"
    MODAL = "modal"


@dataclass(frozen=True, eq=False)
class Axis:
    r"""One tensor factor of a function space — a frozen value object.

    Parameters
    ----------
    label : str
        The factor's role name (``"energy"``, ``"spatial"``, …). Part of
        the identity: two same-shaped factors with different roles are
        different axes.
    shape : tuple[int, ...]
        The factor's index shape, rank ≥ 1 (refused otherwise).
    weights : NDArray | None, default None
        The factor measure over exactly ``shape``. ``None`` IS the
        counting measure (identity metric) — deliberately, always; an
        all-ones array is CANONICALIZED to ``None`` at construction so
        the counting measure has one spelling and one identity. Stored
        canonicalized (``w + 0.0``, killing ``-0.0``), defensively
        copied, read-only. Non-finite entries are refused; signed
        weights are legal (quadrature families need them).
    kind : BasisKind
        Keyword-only, no default — the basis character is physics and
        must be spelled at every mint.
    generator : DiscreteMeasure | Basis | Quadrature | FrameBase | None, default None
        Keyword-only provenance: the object that minted this axis (see
        the module docstring's slot table). NOT part of the identity —
        equality, hash and the ``of_axes`` name digest ignore it. Prefer
        minting through the generator (``measure.axis(label)`` /
        ``quad.axis(label)``) over passing this by hand: a
        generator-minted axis cannot forget its provenance, and its
        ``kind`` is implied by the generator's type (a measure mints
        NODAL — its components are point values with a coordinate cone).
        ⚠ A LIVE REFERENCE, not a snapshot: ``Quadrature`` is a mutable
        dataclass, so ``axis.generator.weights`` can be moved under the
        axis while the axis's own ``weights`` copy (read-only, hashed)
        cannot — the axis's copy is the authoritative factor measure.

    Notes
    -----
    Frozen and hashable with **structural equality per subclass**:
    ``__eq__``/``__hash__`` read ``(type, label, shape, kind, weights
    bytes)``. Subclasses extend the key via :meth:`_identity_key`; the
    class check keeps an ``EnergyAxis`` and a field-identical generic
    ``Axis`` unequal (the identity is *what kind of generator produced
    this factor*, not a bag of fields).
    """

    label: str
    shape: tuple[int, ...]
    weights: NDArray | None = field(default=None, repr=False)
    kind: BasisKind = field(kw_only=True)
    generator: DiscreteMeasure | Basis | Quadrature | FrameBase | None = field(
        default=None, kw_only=True, repr=False
    )

    def __post_init__(self) -> None:
        shape = tuple(int(n) for n in self.shape)
        object.__setattr__(self, "shape", shape)
        if len(shape) < 1:
            raise ValueError(
                f"an Axis needs rank >= 1, got shape {shape!r} — a rank-0 "
                f"factor has no index set to measure"
            )
        if self.weights is not None:
            w = np.ascontiguousarray(self.weights, dtype=float)
            if w.shape != shape:
                raise ValueError(
                    f"axis weights must live over exactly the axis shape: "
                    f"weights shape {w.shape} != axis shape {shape}"
                )
            if not bool(np.isfinite(w).all()):
                raise ValueError(
                    f"axis weights must be finite (a factor measure has "
                    f"finite weights); got {w!r}"
                )
            if bool((w == 1.0).all()):
                # Canonicalization: the counting measure has ONE spelling.
                object.__setattr__(self, "weights", None)
            else:
                # ``+ 0.0`` canonicalizes -0.0 -> +0.0 AND forces a fresh
                # allocation (defensive copy even when the input was
                # already a contiguous float array).
                w = w + 0.0
                w.setflags(write=False)
                object.__setattr__(self, "weights", w)

    def generator_as(self, kind: type[_G], *, consumer: str) -> _G:
        r"""The generator, NARROWED to the type ``consumer`` needs — or a
        refusal naming both parties.

        The one home of the generator-less refusal (P4-remainder G5): a
        consumer that must recover forgotten data (direction cosines, the
        level fibration) calls this instead of touching
        :attr:`generator` bare, because the bare union cannot answer the
        reads (a :class:`~orpheus.numerics.measure.DiscreteMeasure` has
        no ``mu_x``) and a bare ``None`` dereference names neither the
        axis nor the asker. ``consumer`` is the caller's own name; the
        message pins BOTH, so a refusal is diagnosable from the text
        alone.

        Raises
        ------
        ValueError
            If the axis carries no generator of ``kind`` — i.e. it was
            not minted through one (``measure.axis(...)`` /
            ``quad.axis()``).
        """
        g = self.generator
        if not isinstance(g, kind):
            got = type(g).__name__ if g is not None else "None"
            raise ValueError(
                f"axis '{self.label}': {consumer} needs the generating "
                f"{kind.__name__}, but this axis was not minted through "
                f"one (generator={got}). Mint the axis via its generator "
                f"(e.g. quad.axis()) so consumers can recover what the "
                f"axis forgot."
            )
        return g

    def _identity_key(self) -> tuple[Any, ...]:
        """The structural content equality/hash read (subclasses extend).

        ``generator`` is deliberately ABSENT: provenance is not identity
        (module docstring, slot table). Any future field added here also
        enters :meth:`_structural_bytes` and therefore every derived
        space NAME — the blast radius of an inclusion is space identity.
        """
        w = self.weights
        return (
            self.label,
            self.shape,
            self.kind,
            None if w is None else w.tobytes(),
        )

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, Axis)  # type narrowing only
        return self._identity_key() == other._identity_key()

    def __hash__(self) -> int:
        return hash((self.__class__, self._identity_key()))

    def _structural_bytes(self) -> bytes:
        """An INJECTIVE byte encoding of the structural identity.

        Consumed by ``FunctionSpace.of_axes``'s derived-name digest.
        Injectivity was load-bearing for space identity until the identity
        flip (CS4c step 6, 2026-09-07 — space identity was ``(name,
        shape)``, so a name collision between different axis tuples would
        have collapsed two different spaces into one); it stays
        load-bearing where a derived name is folded into an axes-less
        composite's own digest (``FullFieldSpace.from_blocks``) and for the
        label's diagnostic value, hence the belt-and-braces encoding: every chunk is TYPE-TAGGED
        (``T``/``N``/``B``/``R``) and LENGTH-PREFIXED, so no
        concatenation of different identity keys can share a byte
        stream. Deterministic across processes by construction — no
        ``hash()``, no dict order, only content bytes and ``repr`` of
        primitives.
        """
        chunks = [b"T" + type(self).__qualname__.encode()]
        for part in self._identity_key():
            if part is None:
                chunks.append(b"N")
            elif isinstance(part, bytes):
                chunks.append(b"B" + part)
            else:
                chunks.append(b"R" + repr(part).encode())
        return b"".join(len(c).to_bytes(8, "little") + c for c in chunks)


@dataclass(frozen=True, eq=False)
class EnergyAxis(Axis):
    r"""The multigroup energy factor — a 1-D mesh in energy.

    **The faces reading.** An :class:`~orpheus.data.energy_grid.EnergyGrid`
    is a one-dimensional MESH in energy: the group boundaries are its
    FACES, the groups are its CELLS, and condensation is the mesh-overlap
    map (``EnergyGrid.overlap_to`` — fractional re-binning, fine → coarse
    only). The one-group member is the one-CELL energy mesh — its edges
    and weighting spectrum survive because they define :math:`\bar\sigma`,
    which is exactly what the Bateman/depletion pairing
    :math:`\langle\bar\sigma, \phi\rangle` consumes. The axis therefore
    persists down to its terminal one-cell member (collapse doctrine,
    clause 2: partition-integration of an L¹ field class).

    **The counting-measure theorem.** Multigroup flux components are group
    INTEGRALS (covariant, extensive): :math:`\phi_g = \int_g \phi(E)\,dE`.
    Cross sections are flux-weighted group AVERAGES (contravariant,
    intensive). The convention is chosen so that
    :math:`\int \sigma(E)\,\phi(E)\,dE = \sum_g \sigma_g\,\phi_g`
    EXACTLY — no group widths appear. The energy metric is therefore the
    COUNTING measure **as a theorem, not a default**: metric = I,
    :math:`V \cong V^*` isometrically, and the adjoint along energy is
    the plain transpose. Construction enforces it: a weighted
    ``EnergyAxis`` is refused (use a generic :class:`Axis` for
    deliberately non-physical toys), and both constructors mint
    ``weights=None``.

    **The V/V\* collapse hook (declared now, built at S7/Campaign 2).**
    Condensation acts as plain SUM on V (integrals add) and as
    flux-weighted AVERAGE on V* (averages re-weight); collapse
    adjoint-consistency is precisely that pair being mutually adjoint
    under the counting pairing. This axis records the group structure
    those morphisms will consume.

    **Identity (ruled Q2): ng + edges CONTENT.** ``from_grid`` axes carry
    the boundary energies; equality reads their BYTES, never
    ``EnergyGrid`` object identity — ``Mixture.energy_grid`` mints a
    FRESH ``eq=False`` grid per access, so two mints from one mixture
    must (and do) yield equal axes. ``synthetic(ng)`` axes carry no
    edges; identity is ``ng`` alone, and a synthetic axis NEVER equals a
    ``from_grid`` axis of the same ``ng`` (same index set, different
    partition — different axis).

    Edges follow the canonical fast-first convention (strictly
    DESCENDING, group 0 = fastest; ``EnergyGrid`` refuses anything
    else — the invariant is checked there, once, not re-checked here).
    """

    edges: NDArray | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.weights is not None:
            raise ValueError(
                "an EnergyAxis cannot carry weights: the multigroup "
                "convention makes the energy measure COUNTING as a theorem "
                "(group integrals x group averages pair without widths). "
                "For a deliberately non-physical weighted toy, use a "
                "generic Axis."
            )
        if self.kind is not BasisKind.NODAL:
            raise ValueError(
                "an EnergyAxis is NODAL by construction: groups are the "
                "CELLS of a 1-D energy mesh (components are per-group "
                "integrals, not spectral coefficients)"
            )
        if len(self.shape) != 1:
            raise ValueError(
                f"an EnergyAxis is rank 1 (groups index a 1-D energy "
                f"mesh); got shape {self.shape}"
            )
        if self.edges is not None:
            edges = np.ascontiguousarray(self.edges, dtype=float) + 0.0
            edges.setflags(write=False)
            object.__setattr__(self, "edges", edges)
            (ng,) = self.shape
            if edges.shape != (ng + 1,):
                raise ValueError(
                    f"EnergyAxis edges must be the {ng + 1} group "
                    f"boundaries of its {ng} groups; got shape "
                    f"{edges.shape}"
                )

    @classmethod
    def from_grid(cls, grid: "EnergyGrid") -> "EnergyAxis":
        """The axis of a real group structure — identity = ng + edges bytes.

        Axis-from-mesh: symmetric with the (CS2) ``SpatialAxis.from_mesh``
        generator. Consumes the grid's surface only (``edges``,
        ``n_groups``); the descending-edges invariant is the grid's own
        construction contract.
        """
        return cls("energy", (grid.n_groups,), kind=BasisKind.NODAL, edges=grid.edges)

    @classmethod
    def synthetic(cls, ng: int) -> "EnergyAxis":
        """The axis of a grid-less ``ng``-group problem — identity = ng only.

        The honest spelling for fixtures/libraries that declare a group
        COUNT with no boundary energies (every shipped ``get_mixture``
        pair has ``eg is None``). Deliberately UNEQUAL to any
        ``from_grid`` axis of the same ``ng``: same index set, no
        partition data — a different axis.
        """
        return cls("energy", (int(ng),), kind=BasisKind.NODAL, edges=None)

    @classmethod
    def from_materials(cls, materials: "Iterable[Mixture]") -> "EnergyAxis":
        """THE energy-arm rule, in its one home (hoisted at CS4a K1).

        Content-equal ``eg`` edges across ALL the given materials ⟹
        :meth:`from_grid` on the first one's grid; any absent or
        differing edge set ⟹ :meth:`synthetic` on the first one's
        ``ng``. Refuses an empty iterable (no material, no axis).

        The CALLER chooses the denominator, and that choice is part of
        the rule's meaning: a carrier passes exactly its REACHABLE
        materials (the leak principle — a spectator entry with
        ``eg=None`` must not flip the axis identity of a problem it
        does not touch), and the homogeneous pose passes its one
        mixture. Both spellings of the energy arm
        (``MaterialMesh.bulk_space`` and the mixture-minted homogeneous
        space) route through here, so they cannot diverge — the second
        spelling this hoist exists to make unspellable.

        Consumes the materials' surface only (``eg``, ``energy_grid``,
        ``ng``) under ``TYPE_CHECKING`` — the module's layering note
        applies unchanged.
        """
        mats = list(materials)
        if not mats:
            raise ValueError(
                "EnergyAxis.from_materials needs at least one material "
                "(an empty carrier has no energy structure to read)"
            )
        egs = [m.eg for m in mats]
        present = [eg for eg in egs if eg is not None]
        if len(present) == len(egs) and all(
            np.array_equal(eg, present[0]) for eg in present[1:]
        ):
            return cls.from_grid(mats[0].energy_grid)
        return cls.synthetic(mats[0].ng)

    def _identity_key(self) -> tuple[Any, ...]:
        e = self.edges
        return (*super()._identity_key(), None if e is None else e.tobytes())


@dataclass(frozen=True, eq=False)
class HarmonicAxis(Axis):
    r"""The angular MOMENT factor of the real spherical-harmonic family — the
    rectangular ``(L+1, 2L+1)`` coefficient table of
    :class:`~orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis`
    (the addition-theorem-shifted ``[l + m]`` column, zero-padded outside
    :math:`|m| \le \ell`), ``MODAL`` — a spectral coefficient may be negative
    for a positive function, so the factor carries no coordinate cone
    (CS4c step 6 item 6.2c-ii, ruled 2026-09-07/08).

    **The measure IS the head's metric.** Minted by the BASIS
    (:meth:`SphericalHarmonicSpace.from_L
    <orpheus.numerics.spaces.spherical_harmonic_space.SphericalHarmonicSpace.from_L>`)
    the weights are the CONTINUUM Gram :math:`4\pi/(2\ell+1)` broadcast to
    the padded layout; re-dressed by a FRAME
    (:attr:`~orpheus.numerics.frame.FrameBase.basis_space`) they are the
    Parseval inverse of the discrete Gram's diagonal — or ``None`` with the
    matrix pseudo-inverse POSITIONED on the space's derived metric object
    when that Gram is dense (item 6.2c-i) — and the frame becomes the
    axis's :attr:`generator`, the object that can re-dress the head at
    another order (:func:`~orpheus.numerics.spaces.moment_head.truncated_head`).

    **Identity** is the class plus ``(label, shape, kind, weights)``: the
    family is the class, the order is the shape, and — since the identity
    flip (CS4c step 6, 2026-09-07) — the METRIC enters the identity through
    the weights, so a frame-dressed head and a continuum head of the same
    order are two spaces (the metric-blind ``(name, shape)`` seam that let
    them pass for one is gone; the tree carries ONE moment space per
    carrier, the frame's, ruling R-6.2c-1).
    """


@dataclass(frozen=True, eq=False)
class LegendreAxis(Axis):
    r"""The angular MOMENT factor of the Legendre family on
    :math:`S^2/O(2)_a` — the FLAT ``(L+1,)`` coefficient axis of
    :class:`~orpheus.numerics.basis.legendre_basis.LegendreBasis` (one
    coefficient per degree: the trivial isotypic component of
    :math:`O(2)_a` is one-dimensional in every degree), ``MODAL``.

    Everything :class:`HarmonicAxis` says about the measure and the
    generator holds here; what this family ADDS to the identity is the
    axis of the spent stabiliser, ``spent_axis`` — `[M]` 2026-09-08 (the
    6.2c verification round, hazard H-10): ``LegendreSpace.from_L(1, "x")``
    and ``from_L(1, "z")`` carry ``array_equal`` weights and one shape, so
    a family-generic identity would COLLAPSE two physically different
    spaces (the tree carries two poles). The spent axis is therefore part
    of :meth:`_identity_key`, exactly as an :class:`EnergyAxis` carries its
    group edges.

    Parameters
    ----------
    spent_axis : str
        Keyword-only. The axis of the spent :math:`O(2)_a` stabiliser —
        ``"x"``, ``"y"`` or ``"z"``.
    """

    spent_axis: str = field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.spent_axis not in ("x", "y", "z"):
            raise ValueError(
                f"LegendreAxis: spent_axis must be x/y/z, got {self.spent_axis!r}."
            )

    def _identity_key(self) -> tuple[Any, ...]:
        return (*super()._identity_key(), self.spent_axis)
