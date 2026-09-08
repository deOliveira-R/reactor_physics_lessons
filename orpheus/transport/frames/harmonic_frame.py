r"""The :class:`HarmonicFrame` — the shared angular operator FACTORY and its
minted, carrier-typed faces.

The frame is the shared ``(basis, measure)`` pairing — its identity is the
table's identity — and it MINTS precisely-typed bound operator faces, taking
at mint time the one input it cannot know: the angular FIELD space of the
posed problem. The minted faces are the transport realisations of the
:class:`~orpheus.numerics.projection.AnalysisOperator` /
:class:`~orpheus.numerics.projection.ReconstructionOperator` roles (whose own
docstrings anticipate exactly this shape:
``AnalysisOperator[AngularFlux, HarmonicMomentFlux]``): the frame square's
four faces are instances of the TWO classes below over the (flux ⊗ source)
carrier pairs — primal/dual lives in the carrier GENERICS, never in class
names.

The frame square::

    M_flux : AngularFlux              → HarmonicMomentFlux        flux_analysis_on
    R_src  : HarmonicMomentSourceSink → AngularSourceSink         source_reconstruction_on
    M_src  : AngularSourceSink        → HarmonicMomentSourceSink  (mint-ready, unminted)
    R_flux : HarmonicMomentFlux       → AngularFlux               (mint-ready, unminted)

Only the faces with a production consumer are minted (the defer-until-consumer
discipline): the windowed bulk projection
(:class:`~orpheus.sn.operators.windowing.BulkAnalysisOperator`) and the S6
adjoint gates consume the flux analysis; the windowed in-scatter arm consumes
the source reconstruction. The other two are one mint verb away — the face
classes are carrier-generic.

Why minted faces instead of bound-frame verbs (the A1 → F-1 re-carve)
=====================================================================

The S4-amendment's A1 phase bound the field spaces to the FRAME — a proxy: it
bound the *factory* when the ruling ("an operator is not an operator without
its two spaces") was about the *products*. That made the frame unshareable
(its identity carried a field space) and outlawed the space-less kernel path.
F-1 (``.claude/plans/frame_square_recarve.md``) completes the amendment where
it was aimed: the FACES are the bound operators — domain, codomain, and
carrier pair demanded at construction — and the frame reverts to the shared
factory two consumers can hold in common. Frames over the same ``(basis,
measure)`` are the same projection, and every face minted from one frame
shares its cached table (the "derived, not independent" guarantee at the
array level). A1's derivation logic, content-equality admission text, and
gates carried over into the mint — completion, not revert.

Kernels delegate to the frame's numerics faces
(:attr:`~orpheus.numerics.frame.FrameBase.analysis` /
:attr:`~orpheus.numerics.frame.FrameBase.reconstruction`), so a minted face is
bit-identical to the raw face on values — the typed seam adds carriers and
binding, never a number. ``apply`` ADMITS exactly its carrier on its bound
domain (content equality — the space-content invariant); ``apply_transpose``
also accepts raw ``ndarray`` values, the seam the metric-aware
``AdjointOperator`` drives, so a face's ``.H`` is the PHYSICAL Hilbert
adjoint on the F-0 Parseval metrics — ``M* = R/W`` and ``R* = W·M`` (see
:attr:`FrameBase.basis_space
<orpheus.numerics.frame.FrameBase.basis_space>`).

The moment codomain is derived at mint time — moment = f(angular, L), never
the reverse, because the angular space carries the quadrature axis and, on a
widened iterate, the scheme's mass-bearing spatial-moment axis, which no
moment operand determines (an earlier revision made ``reconstruct`` take a
per-call ``space=``; that parameter was an unbound operator's missing
codomain leaking into the apply signature — user diagnosis, 2026-08-22). Its
spherical-harmonic factor is the frame's OWN F-0-dressed
:attr:`~orpheus.numerics.frame.FrameBase.basis_space`, so the Parseval metric
rides into the product codomain with a single source: the frame's codomain IS
the SH factor. Since CS4c step 6 item 6.2c-ii (ruling R-6.2c-1) the
carrier's own cached moment space —
:meth:`SNMesh.moment_space <orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space>`,
the object every moment field and admission guard on that carrier holds —
reads the SAME dressed head, so the derived product is STRUCTURALLY equal
to it (one space, two owners, ruling O-5); until then the admission seam
was metric-blind: the two were one ``(name, shape)`` and two metrics, the
frame's Parseval-dressed and
the carrier's continuum one (#429 tracker 2.5, Landing A). Item 6.2c, which
makes the head axis-built and its weights part of the identity, is where
that seam is decided.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from numpy.typing import NDArray

from orpheus.numerics.basis.base import Basis, TruncatedBasis
from orpheus.numerics.frame import GalerkinFrame
from orpheus.numerics.measure import DiscreteMeasure
from orpheus.numerics.projection import AnalysisOperator, ReconstructionOperator
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.transport.source_sinks.angular_source_sink import AngularSourceSink
from orpheus.transport.source_sinks.harmonic_moment_source_sink import (
    HarmonicMomentSourceSink,
)

if TYPE_CHECKING:
    from orpheus.numerics.space import FunctionSpace


__all__ = [
    "HarmonicAnalysisOperator",
    "HarmonicFrame",
    "HarmonicReconstructionOperator",
]


#: The per-ordinate (angular) carrier axis of the (flux ⊗ source) grid.
AngularFieldT = TypeVar("AngularFieldT", AngularFlux, AngularSourceSink)
#: The spherical-harmonic-moment carrier axis of the same grid.
MomentFieldT = TypeVar("MomentFieldT", HarmonicMomentFlux, HarmonicMomentSourceSink)


def _admit(
    face: object, operand: object, carrier: type, space: "FunctionSpace", end: str,
) -> None:
    r"""The bound-face admission: exactly the minted carrier, on the bound space.

    Content equality (``(name, shape)`` — the space-content invariant): a
    wrong carrier or a content-different space is a loud ``TypeError`` naming
    both sides. Shared by both face classes (one spelling of the admission).
    """
    if not isinstance(operand, carrier):
        raise TypeError(
            f"{type(face).__name__}: unsupported carrier "
            f"{type(operand).__name__}; this face is minted for "
            f"{carrier.__name__}."
        )
    if operand.space != space:  # type: ignore[attr-defined]
        raise TypeError(
            f"{type(face).__name__}: operand rides space "
            f"{operand.space.name!r} but this face is bound to {end} "  # type: ignore[attr-defined]
            f"{space.name!r} — a bound face admits only elements of its "
            f"bound spaces (the space-content invariant)."
        )


class HarmonicAnalysisOperator(
    AnalysisOperator, Generic[AngularFieldT, MomentFieldT],
):
    r"""A minted analysis face :math:`M \otimes I_{\text{cells}}` — bound and
    carrier-typed.

    The role base is inherited UNPARAMETERIZED (the
    :class:`~orpheus.numerics.operator.ZeroMorphism` precedent): the transport
    carriers deliberately do not satisfy the numerics ``Vector`` protocol's
    endomorphic-arithmetic shape (a source's ``+`` legitimately returns a
    union), so the per-carrier precision rides this class's OWN generics —
    the mint verbs return ``HarmonicAnalysisOperator[AngularFlux,
    HarmonicMomentFlux]`` etc., and callers get the precise pair.

    An :class:`~orpheus.numerics.projection.AnalysisOperator` member whose two
    spaces AND two carriers are demanded at construction — the S4-amendment's
    base demand realised at the transport layer (the
    :class:`~orpheus.numerics.operator.ZeroMorphism` binding precedent). The
    kernel delegates to the minting frame's numerics analysis face (shared
    cached table); the carrier wrap parameters — the truncation order ``L``
    and the within-cell spatial-moment width — are read ONCE, at mint time,
    from the frame's basis and the bound angular domain.

    ``apply`` admits exactly ``domain_carrier`` on the bound domain and its
    output rides the bound codomain. ``apply_transpose`` maps the codomain
    carrier back to the domain carrier (the representation transpose), and
    also accepts raw ``ndarray`` values — the seam the metric-aware
    ``AdjointOperator`` drives, so ``face.H`` is the physical Hilbert
    adjoint on the F-0 Parseval metrics (:math:`M^* = R/W`).
    """

    def __init__(
        self,
        *,
        frame: "HarmonicFrame",
        domain: "FunctionSpace",
        codomain: "FunctionSpace",
        domain_carrier: type[AngularFieldT],
        codomain_carrier: type[MomentFieldT],
    ) -> None:
        from orpheus.transport.fields._bases import BulkField

        self.frame = frame
        self._domain = domain
        self._codomain = codomain
        self.domain_carrier = domain_carrier
        self.codomain_carrier = codomain_carrier
        # The wrap parameters, read once at mint time (moment = f(angular, L)).
        self._L = frame.truncation_order
        self._spatial_moments = BulkField.spatial_moments_per_axis_of(domain)

    @property
    def domain(self) -> "FunctionSpace":
        return self._domain

    @property
    def codomain(self) -> "FunctionSpace":
        return self._codomain

    def apply(self, field: AngularFieldT) -> MomentFieldT:
        r"""Project the bound per-ordinate carrier to its SH moments (:math:`M`)."""
        _admit(self, field, self.domain_carrier, self._domain, "angular domain")
        values = self.frame.analysis.apply(field.values)
        return self.codomain_carrier(
            values=values,
            space=self._codomain,
            L=self._L,
            spatial_moments=self._spatial_moments,
        )

    def apply_transpose(
        self, moment: MomentFieldT | NDArray,
    ) -> AngularFieldT | NDArray:
        r"""The representation transpose :math:`M^\top` — carrier → carrier,
        or raw values → raw values (the ``AdjointOperator`` seam)."""
        if isinstance(moment, self.codomain_carrier):
            _admit(self, moment, self.codomain_carrier, self._codomain, "moment codomain")
            values = self.frame.analysis.apply_transpose(moment.values)
            return self.domain_carrier(values=values, space=self._domain)
        return self.frame.analysis.apply_transpose(moment)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.domain_carrier.__name__} → "
            f"{self.codomain_carrier.__name__}, L={self._L})"
        )


class HarmonicReconstructionOperator(
    ReconstructionOperator, Generic[MomentFieldT, AngularFieldT],
):
    r"""A minted reconstruction face :math:`R \otimes I_{\text{cells}}` — bound
    and carrier-typed.

    The synthesis sibling of :class:`HarmonicAnalysisOperator` (one docstring
    for the shared design; here only the mirror facts). The kernel delegates
    to the frame's numerics reconstruction face — the addition-theorem
    (canonical-dual) synthesis. The role ABC is apply-only; this face ADDS the
    transpose, so ``is_adjointable`` is overridden ``True`` (mirroring the
    numerics ``_FrameReconstruction``) and ``face.H`` is the physical
    :math:`R^* = W\,M` on the F-0 Parseval metrics.
    """

    def __init__(
        self,
        *,
        frame: "HarmonicFrame",
        domain: "FunctionSpace",
        codomain: "FunctionSpace",
        domain_carrier: type[MomentFieldT],
        codomain_carrier: type[AngularFieldT],
    ) -> None:
        from orpheus.transport.fields._bases import BulkField

        self.frame = frame
        self._domain = domain
        self._codomain = codomain
        self.domain_carrier = domain_carrier
        self.codomain_carrier = codomain_carrier
        self._L = frame.truncation_order
        self._spatial_moments = BulkField.spatial_moments_per_axis_of(codomain)

    @property
    def is_adjointable(self) -> bool:
        # The face adds apply_transpose on top of the apply-only role, so
        # ``R.H`` is free here — the override lives on the face (as on the
        # numerics ``_FrameReconstruction``).
        return True

    @property
    def domain(self) -> "FunctionSpace":
        return self._domain

    @property
    def codomain(self) -> "FunctionSpace":
        return self._codomain

    def apply(self, moment: MomentFieldT) -> AngularFieldT:
        r"""Synthesise the bound per-ordinate carrier from SH moments (:math:`R`)."""
        _admit(self, moment, self.domain_carrier, self._domain, "moment domain")
        values = self.frame.reconstruction.apply(moment.values)
        return self.codomain_carrier(values=values, space=self._codomain)

    def apply_transpose(
        self, field: AngularFieldT | NDArray,
    ) -> MomentFieldT | NDArray:
        r"""The representation transpose :math:`R^\top` — carrier → carrier,
        or raw values → raw values (the ``AdjointOperator`` seam)."""
        if isinstance(field, self.codomain_carrier):
            _admit(self, field, self.codomain_carrier, self._codomain, "angular codomain")
            values = self.frame.reconstruction.apply_transpose(field.values)
            return self.domain_carrier(
                values=values,
                space=self._domain,
                L=self._L,
                spatial_moments=self._spatial_moments,
            )
        return self.frame.reconstruction.apply_transpose(field)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.domain_carrier.__name__} → "
            f"{self.codomain_carrier.__name__}, L={self._L})"
        )


#: Instance-dict key under which :meth:`HarmonicFrame.from_galerkin`
#: interns the upgrade on the upstream :class:`GalerkinFrame`.
_UPGRADE_SLOT = "_harmonic_frame_upgrade"


def _admit_truncated(basis: Basis, door: str) -> TruncatedBasis:
    r"""The door's ONE demand: a trial basis that carries a truncation order.

    The mints read ``L`` and the operator ends read ``space`` — the
    :class:`~orpheus.numerics.basis.base.TruncatedBasis` surface — so that
    is what the door asks for, TYPED, at the door (a frame built over an
    indicator trial fails HERE with a message naming the surface, not three
    frames later with an ``AttributeError``). Until #429 tracker 2.5
    (2026-09-02) this door named ONE class, ``SphericalHarmonicBasis``,
    which would have refused the slab's Legendre basis on every 1-D solve
    at :math:`L = 0`.
    """
    if not isinstance(basis, TruncatedBasis):
        raise TypeError(
            f"{door} requires a trial basis carrying a truncation order L "
            f"(the harmonic family: the real spherical harmonics, their "
            f"sigma-even restriction, the Legendre basis on S^2/O(2)_a); "
            f"got {type(basis).__name__}, which carries none. The mints read "
            f"L and the operator ends read the basis's coefficient space."
        )
    return basis


@dataclass(frozen=True, init=False)
class HarmonicFrame(GalerkinFrame):
    r"""The angular spherical-harmonic :class:`GalerkinFrame` — the shared
    operator factory that mints the carrier-typed faces.

    Constructed from ``(basis, measure)`` alone — the constructor demands a
    trial basis carrying a truncation order
    (:class:`~orpheus.numerics.basis.base.TruncatedBasis`: the real
    spherical harmonics, their σ-even restriction, the Legendre basis on
    :math:`S^2/O(2)_a`; a harmonic frame over an indicator trial is an
    illegal state, refused at the door) — or upgraded from a generic
    ``quadrature.angular_frame(L)`` via :meth:`from_galerkin`. WHICH family
    the frame binds is the quadrature's decision, derived from the point
    set its measure lives on; this frame reads the basis it is handed. ⭐
    The FRAME is the single source of the angular coefficient space: the
    operator ends and the moment fields minted downstream read its
    Parseval-dressed ``basis_space`` (CS4c step 6 item 6.2c-ii, ruling
    R-6.2c-1; #429 tracker 2.5 had bound the basis's own ``basis.space``
    on 2026-09-02 — until then seven production sites re-minted it from
    ``L`` as the full-sphere family, which is exactly the family a 1-D
    rule must NOT bind). Identity is the
    table's identity: two frames over the same pairing are the same
    projection, and every face minted here shares this frame's cached table
    and F-0 Parseval codomain.

    The A1 revision bound ``angular_space``/``moment_space`` fields and
    carrier verbs ``analyse``/``reconstruct`` on the frame itself; F-1
    retired them (2026-08-23) — the binding lives on the MINTED FACES
    (:meth:`flux_analysis_on` / :meth:`source_reconstruction_on`), which is
    where the S4-amendment's spaces-demand was aimed.
    """

    # The inherited frozen (read-only) field, un-narrowed: the door guarantees
    # the TruncatedBasis surface, and the mints read it through
    # :attr:`truncation_order`.
    basis: Basis

    def __init__(self, basis: Basis, measure: DiscreteMeasure) -> None:
        _admit_truncated(basis, "HarmonicFrame")
        super().__init__(basis, measure)

    @property
    def truncation_order(self) -> int:
        r"""The truncation order :math:`L` of the bound family — what the mints read."""
        return _admit_truncated(self.basis, "HarmonicFrame").L

    @classmethod
    def from_galerkin(cls, frame: GalerkinFrame) -> "HarmonicFrame":
        r"""Upgrade a generic angular :class:`GalerkinFrame`, reusing its basis
        + measure (no rebuild — the table / numerics spaces / faces are
        bit-identical). The ONLY job left here after F-1 is the door: a
        frame over a trial basis carrying no truncation order (an indicator
        basis) is rejected at the upgrade boundary, not later when a mint
        first reads ``L``. The family itself is NOT narrowed — the fold's
        σ-even harmonics and the slab's Legendre basis pass exactly as the
        full harmonics do (#429 tracker 2.5).

        INTERNED per upstream frame object (CS4c §14.4): upgrading the same
        :class:`GalerkinFrame` twice returns the SAME :class:`HarmonicFrame`
        — combined with :meth:`Quadrature.angular_frame
        <orpheus.numerics.quadrature.directional.Quadrature.angular_frame>`'s
        per-``(rule, L)`` interning, "one frame per (axis content, L)" is an
        object identity, and the cached projection table is shared by every
        consumer rather than re-evaluated per operator.
        """
        # The intern rides the upstream frame's own instance dict — the
        # cached_property idiom one level out (a frozen dataclass has no
        # __slots__ here, and writing __dict__ directly bypasses the
        # frozen __setattr__ exactly as cached_property does).  Lifetime
        # is thereby correct for free: the upgrade dies with its
        # quadrature-interned upstream, and no global registry exists to
        # leak or to key (a GalerkinFrame hashes its ndarray fields, so
        # it cannot key a dict).
        inst_dict = cast("dict[str, Any]", frame.__dict__)
        cached = inst_dict.get(_UPGRADE_SLOT)
        if cached is not None:
            return cached
        basis = frame.basis
        _admit_truncated(basis, "HarmonicFrame.from_galerkin")
        upgraded = cls(basis, frame.measure)
        inst_dict[_UPGRADE_SLOT] = upgraded
        return upgraded

    @classmethod
    def for_space(
        cls, angular_space: "FunctionSpace", L: int,
    ) -> "HarmonicFrame":
        r"""THE blessed frame chain (CS4c §14.4) — the one spelling every
        consumer uses to reach the shared angular frame from a space.

        ``angular_space → leading (angular) axis →``
        :meth:`Axis.generator_as <orpheus.numerics.axis.Axis.generator_as>`
        ``(Quadrature) →``
        :meth:`Quadrature.angular_frame
        <orpheus.numerics.quadrature.directional.Quadrature.angular_frame>`
        ``(L) →`` :meth:`from_galerkin` — every hop single-sourced, both
        cache tiers interned, so S, F, and the windowing method minting
        from the same posed space share ONE frame object and ONE metric
        (the quadrature's weights; no copy exists to drift).

        Parameters
        ----------
        angular_space : FunctionSpace
            An axis-built per-ordinate space whose LEADING axis is the
            angular factor (the module's ``axes[0]`` convention, same as
            :meth:`moment_space_on`).
        L : int
            The spherical-harmonic truncation order of the wanted frame.
        """
        from orpheus.numerics.quadrature.directional import Quadrature

        axes = angular_space.axes
        if axes is None:
            raise TypeError(
                "HarmonicFrame.for_space: the angular field space must be "
                "axis-built (an S2 per-ordinate space); a shape-only space "
                "carries no generator channel to reach the quadrature."
            )
        quadrature = axes[0].generator_as(
            Quadrature, consumer="HarmonicFrame.for_space",
        )
        return cls.from_galerkin(quadrature.angular_frame(L))

    # ── the moment-codomain derivation (the PUBLIC single source) ─────

    def moment_space_on(self, angular_space: "FunctionSpace") -> "FunctionSpace":
        r"""The moment codomain derived from an angular domain (+ this frame's ``L``) — the SINGLE SOURCE of the moment-space derivation (CS4c §14.4: public, so field mints consume it instead of re-deriving; drift between a face's codomain and a minted moment field's space is unspellable).

        ``basis_space * of_axes(<cell axes>)`` — the angular HEAD factor is
        the frame's OWN F-0-dressed
        :attr:`~orpheus.numerics.frame.FrameBase.basis_space` (the bound
        basis's coefficient space, whatever family the quadrature bound —
        the spherical-harmonic space on a full-sphere rule; single source:
        the Parseval metric rides into the product), the cell group is the
        angular space's own energy/spatial axes (the same instances the
        carrier's mints share, so the product content-equals the carrier's
        cached ``SNMesh.moment_space(L, width)`` — the object the moment
        fields hold since CS4c step 6 item 6.2b — which since #429 tracker
        2.5 reads the SAME basis through the mesh's quadrature) — with the
        ``SpatialMomentSpace`` factor appended for a widened angular space.
        Runs once per mint: the derivation direction is moment = f(angular,
        L), never the reverse.
        """
        from orpheus.numerics.moment_layout import SPATIAL_MOMENT_AXIS_LABEL
        from orpheus.numerics.quadrature.directional import Quadrature
        from orpheus.numerics.space import FunctionSpace
        from orpheus.numerics.spaces.spatial_moment_space import (
            SpatialMomentSpace,
        )
        from orpheus.transport.fields._bases import BulkField

        axes = angular_space.axes
        if axes is None:
            raise TypeError(
                "HarmonicFrame mint: the angular field space must be "
                "axis-built (an S2 per-ordinate space); a shape-only space "
                "cannot name the cell axes the moment codomain is derived "
                "from."
            )
        # The leading axis must be the QUADRATURE's per-ordinate axis — a
        # moment space is axis-built too since CS4c step 6 item 6.2c-ii
        # (its leading axis is a MODAL head minted by a basis or a frame),
        # so the axes-less refusal above no longer catches it; this does,
        # by name (hazard H-6: a guard must not silently lose its subject).
        axes[0].generator_as(Quadrature, consumer="HarmonicFrame.moment_space_on")
        cell_axes = [
            ax for ax in axes[1:] if ax.label != SPATIAL_MOMENT_AXIS_LABEL
        ]
        base = self.basis_space * FunctionSpace.of_axes(*cell_axes)
        per_axis = BulkField.spatial_moments_per_axis_of(angular_space)
        if per_axis == 1:
            return base
        ndim = next(
            len(ax.shape) for ax in cell_axes if ax.label == "spatial"
        )
        return base * SpatialMomentSpace.from_per_axis(per_axis, ndim)

    # ── the mint verbs (one per face-with-a-consumer) ─────────────────

    def flux_analysis_on(
        self, angular_space: "FunctionSpace",
    ) -> HarmonicAnalysisOperator[AngularFlux, HarmonicMomentFlux]:
        r"""Mint the FLUX analysis face :math:`M \otimes I` bound to
        ``angular_space`` (``AngularFlux → HarmonicMomentFlux``).

        Consumers: the windowed bulk projection
        (:class:`~orpheus.sn.operators.windowing.BulkAnalysisOperator`) and
        the S6 adjoint gates. The source-analysis sibling
        (``AngularSourceSink → HarmonicMomentSourceSink``) is mint-ready,
        unminted — add its verb with its first consumer.
        """
        return HarmonicAnalysisOperator(
            frame=self,
            domain=angular_space,
            codomain=self.moment_space_on(angular_space),
            domain_carrier=AngularFlux,
            codomain_carrier=HarmonicMomentFlux,
        )

    def source_reconstruction_on(
        self, angular_space: "FunctionSpace",
    ) -> HarmonicReconstructionOperator[HarmonicMomentSourceSink, AngularSourceSink]:
        r"""Mint the SOURCE reconstruction face :math:`R \otimes I` landing in
        ``angular_space`` (``HarmonicMomentSourceSink → AngularSourceSink``).

        Consumer: the MOMENT end of the transfer lift
        (:class:`~orpheus.transport.operators.transfer.TransferOperator`'s
        typed :math:`\ell \ge 1` route, selected at construction when the
        binding's domain is the moment composite — the explicit typed
        grid path); every lift admits the face against the angular space
        it emits on. The flux-reconstruction sibling
        (``HarmonicMomentFlux → AngularFlux``) is mint-ready, unminted.
        """
        return HarmonicReconstructionOperator(
            frame=self,
            domain=self.moment_space_on(angular_space),
            codomain=angular_space,
            domain_carrier=HarmonicMomentSourceSink,
            codomain_carrier=AngularSourceSink,
        )
