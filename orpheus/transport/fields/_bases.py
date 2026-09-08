r"""Storage-base ABCs for the typed transport field vocabulary (B.1).

This module is the single source of truth for the machinery that every
typed transport field used to redeclare leaf-by-leaf (Cardinal Rule 2).
Before B.1, ``AngularFlux`` and ``AngularSourceSink`` each independently
carried an identical ``mesh`` field, ``(N, ng, nx, ny)`` shape-check,
mesh-binding ``_check_partner``, ``from_mesh``/``from_ndarray``, and
``N/ng/nx/ny`` read-throughs — and ``ScalarFlux`` / ``ScalarSourceSink``
mirrored the same pattern on ``(ng, nx, ny)``. The repetition IS the
architectural smell; these bases are the consolidation.

The storage × role × locus hierarchy
=====================================

The field vocabulary (issues #205 / #201; the #290 P2.5 axis-coherence
ruling) is a grid of THREE orthogonal axes: **locus** {Bulk,
Boundary(field) / Trace(space)} × **family** {Angular, Scalar, Moment}
× **role** {Flux, SourceSink, Residual}. (A fourth role, Displacement,
existed until campaign 1 CS3, 2026-08-19 — flux lives in V now, so
differences are same-typed and the sibling family retired.) A bulk leaf is
named ``<Family><Role>``, a boundary leaf ``<Family>Boundary<Role>`` —
"Boundary" is the locus qualifier, never a fourth family. This module
provides the *locus + family* axes as ABCs; the *role* leaves
(``AngularFlux``, ``AngularSourceSink``, ...) sit beneath them::

    Field (numerics, L1 — values + space + dunder algebra)
     ├─ BulkField (ABC)           codim-0 (cell centres): ng + the spatial-moment tail reads
     │   ├─ AngularField (ABC)    + N + the carrier's cached space via _space_for_mesh (space_on)
     │   │   ├─ AngularFlux           role leaf  (flux)
     │   │   └─ AngularSourceSink     role leaf  (source; renamed from PerOrdinateSource in B.2)
     │   ├─ ScalarField (ABC)     + the carrier's cached space via _space_for_mesh (space_on)
     │   │   ├─ ScalarFlux            role leaf  (flux)
     │   │   └─ ScalarSourceSink       role leaf  (source; renamed from IsotropicSource in B.2)
     │   └─ MomentField (ABC)     + L + the carrier's cached space via SNMesh.moment_space(L, width) (space_on)
     │       └─ HarmonicMomentFlux   role leaf  (flux-only for now)
     └─ FaceField[K] (ABC)        codim-1 (faces/edges): flat single-buffer + FaceLayout[K]
         │                        slice-views + layout guards + space_on via _face_space_of. STRUCTURE only — the metric descends PER LEAF
         │                        (spatial |Ω·n̂|·w; pole V_cell), never on this ABC (ERR-067).
         ├─ BoundaryField (ABC, FaceField[str])   SPATIAL faces (keyed by name) + from_face_arrays;
         │   │                    the FullField boundary-slot discriminator (the pole is NOT one)
         │   ├─ AngularBoundaryField (ABC)   mesh: SNMesh + AngularTraceSpace (mesh.angular_trace)
         │   │   ├─ AngularBoundaryFlux          role leaf  (flux)
         │   │   ├─ AngularBoundarySourceSink    role leaf  (source; B.3 — orpheus.transport.source_sinks)
         │   │   ├─ AngularBoundaryResidual      role leaf  (residual; B.3 — orpheus.transport.residuals)
         │   └─ ScalarBoundaryField (ABC)    ScalarTraceSpace (DiffusionMesh.scalar_trace; #290 P2/P7a)
         │       ├─ ScalarBoundaryFlux           role leaf  (flux — the per-face (J⁺, J⁻) pair)
         ├─ RadialCharacteristicInteriorField (ABC, FaceField[(level,sign)])  ANGULAR edge — the ψ½
         │   │                    marched cells (μ = μ_start; #282 route (a); System B's interior);
         │   │                    a FaceField SIBLING of BoundaryField, never a child
         │   ├─ RadialCharacteristicInteriorFlux          role leaf  (flux — the ψ½ state)
         │   ├─ RadialCharacteristicInteriorSourceSink    role leaf  (source — q½ cells)
         │   ├─ RadialCharacteristicInteriorResidual      role leaf  (residual)
         └─ RadialCharacteristicBoundaryField (ABC, FaceField[(level,sign)])  the r = R ψ½ corner
             │                    (System B's boundary; the unified 3-tuple base retired at 4e)
             ├─ RadialCharacteristicBoundaryFlux          role leaf  (flux — corner data/defect)
             ├─ RadialCharacteristicBoundarySourceSink    role leaf  (source — corner datum)
             ├─ RadialCharacteristicBoundaryResidual      role leaf  (residual)

Parametrization (no twin paths)
===============================

The per-family phase-space shape lives on the SPACE — the shared
``__post_init__`` validator is :class:`~orpheus.numerics.field.Field`'s
own values-vs-space check (CS4b S4 collapsed the per-family
``_phase_space_shape`` hook into it). Every family sources its space from the
CARRIER's cached mints (campaign 1 CS4b): the Angular/Scalar families
read ``mesh.angular_bulk_space`` / ``mesh.bulk_space``, ``MomentField``
composes ``<the quadrature frame's basis space at L> * mesh.bulk_space``
(the spherical-harmonic space on a full-sphere rule — READ off
``mesh.quad.angular_frame(L)``, never minted from ``L``; #429 tracker
2.5), and the
``BoundaryField`` families read the cached traces
(``mesh.angular_trace`` / ``mesh.scalar_trace`` via
:meth:`FaceField._face_space_of`). Role is CLASS identity — the leaves
of one family share one space instance, and the class arm of the
partner gate is the sole role enforcement. (The per-leaf ``_SPACE_NAME``
role tags retired with this move; until CS4b each leaf minted its own
role-named tag space.)

References
----------

* ``.claude/plans/field_role_typing_view_g.md`` — Phase B (field
  vocabulary), step B.1 (storage-base ABCs).
* Grand Report v3 §5.5 (Field hierarchy), §32.5 (Field primitive spec).
* ``coding-elegance`` Pattern 2 (single source of truth), Pattern 4
  (illegal states unrepresentable), Pattern 5 (build the primitive).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from typing import TYPE_CHECKING, Generic, Mapping, Self, TypeVar, Protocol, cast, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from orpheus.numerics.field import Field
from orpheus.numerics.axis import EnergyAxis
from orpheus.numerics.moment_layout import (
    SPATIAL_MOMENT_AXIS_LABEL,
    cell_moment_count,
)
from orpheus.numerics.space import FunctionSpace
from orpheus.numerics.spaces.spatial_moment_space import (
    SpatialMomentSpace,
    spatial_moment_tail,
)
from orpheus.numerics.spaces.scalar_trace_space import ScalarTraceSpace
from orpheus.numerics.spaces.angular_trace_space import AngularTraceSpace
from orpheus.numerics.spaces.radial_characteristic_space import (
    RadialCharacteristicBoundarySpace,
    RadialCharacteristicInteriorSpace,
)

if TYPE_CHECKING:
    from orpheus.numerics.spaces.moment_head import MomentHead
    from orpheus.diffusion.augmented_mesh import DiffusionMesh
    from orpheus.numerics.face_layout import FaceLayout
    from orpheus.sn.mesh.augmented_mesh import SNMesh
    from orpheus.transport.mesh.material_mesh import MaterialMesh


#: The face-key type of a :class:`FaceField` and its
#: :class:`~orpheus.numerics.face_layout.FaceLayout`: ``str`` face names for
#: the spatial :class:`BoundaryField`, the ``(level, sign)`` tuple for
#: the ψ½ split loci. Bounded by
#: :class:`~collections.abc.Hashable` — the face key is a mapping key.
K = TypeVar("K", bound=Hashable)


__all__ = [
    "FieldRole",
    "RolePair",
    "spatial_moments_per_axis_of",
    "BulkField",
    "AngularField",
    "ScalarField",
    "MomentField",
    "FaceField",
    "BoundaryField",
    "AngularBoundaryField",
    "ScalarBoundaryField",
]


# ═══════════════════════════════════════════════════════════════════════
# Role pairs — the flux ↔ source/sink partnership, declared ONCE per pair
# ═══════════════════════════════════════════════════════════════════════


class FieldRole(Enum):
    r"""The two roles a transport field can play on one carrier space.

    A *flux* is a state; a *source/sink* is a signed rate density — the
    same space, a different physical kind (units differ; the
    :class:`~orpheus.numerics.field.Field` class-identity gate keeps them
    from mixing). Every operator of the transport algebra maps one role
    to the other (a gain ``S`` reads a flux and emits a source; an inverse
    ``(L + C)⁻¹`` reads a source and returns a flux), so the operator
    tier needs to name the role it EMITS without naming the leaf class —
    the leaf is the OPERAND's business (CS4c step 5, ruling R-2: the
    carriers declare their partners; the lift verb reads the declaration).
    """

    FLUX = "flux"
    SOURCE_SINK = "source_sink"


class RolePair:
    r"""Mixin: a role leaf that knows its partner across the flux ↔
    source/sink pair.

    The pair is declared ONCE, on the source/sink leaf's class statement —
    ``class AngularSourceSink(AngularField, flux=AngularFlux)`` — and
    :meth:`__init_subclass__` registers BOTH directions, so the map is a
    bijection by construction: a second source/sink naming the same flux
    is refused at import time, and neither half can be re-pointed later
    (``coding-elegance`` Pattern 4 — the illegal state is unspellable,
    not validated). Why the source/sink side declares: the source/sink
    leaves already import the flux leaves for their named compositions
    (``from_isotropic``, ``from_balance``); the flux leaves import no
    source/sink, so the dependency runs one way and the package init of
    :mod:`orpheus.transport.fields` completes the registration by
    importing :mod:`orpheus.transport.source_sinks` at its tail (a bare
    ``import``, never a name — see that file).

    Consumers:

    * :meth:`role_partner` — the leaf CLASS of the other role on the same
      carrier (``AngularFlux.role_partner(FieldRole.SOURCE_SINK) is
      AngularSourceSink``, and back);
    * :meth:`into_role` — the ONE spelling of *"same space, same family
      fields, the other role's class"*: the operator tier's typed output
      (an emission rides the operand's space; an inverse returns the
      flux of the source it was handed) without an ``isinstance`` on the
      operand — the carrier parse the CS4c step-5 census counted **12
      times** across three verbs, retired by this verb;
    * :meth:`role` — which half this leaf is.

    A residual (the defect of a balance) and a coefficient field are
    neither half of any pair; they carry no partner, and asking is a
    ``TypeError`` naming them.
    """

    # Set by ``__init_subclass__`` on BOTH halves of a declared pair — the
    # same mapping object, so the two directions cannot drift apart.
    _role_partners: "Mapping[FieldRole, type[RolePair]]"
    _field_role: FieldRole

    def __init_subclass__(
        cls, *, flux: "type[RolePair] | None" = None, **kwargs: object,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if flux is None:
            return
        if not (isinstance(flux, type) and issubclass(flux, RolePair)):
            raise TypeError(
                f"{cls.__name__}: flux= must name a role leaf class; got "
                f"{flux!r}."
            )
        if "_role_partners" in vars(flux):
            other = flux._role_partners[FieldRole.SOURCE_SINK].__name__
            raise TypeError(
                f"{flux.__name__} already has the source/sink partner "
                f"{other}; {cls.__name__} cannot be a second one — a role "
                f"pair is a bijection."
            )
        partners: dict[FieldRole, type[RolePair]] = {
            FieldRole.FLUX: flux,
            FieldRole.SOURCE_SINK: cls,
        }
        cls._role_partners = partners
        cls._field_role = FieldRole.SOURCE_SINK
        flux._role_partners = partners
        flux._field_role = FieldRole.FLUX

    @classmethod
    def role(cls) -> FieldRole:
        r"""Which half of the pair this leaf is (``TypeError`` if neither)."""
        role = getattr(cls, "_field_role", None)
        if role is None:
            raise TypeError(
                f"{cls.__name__} is not one half of a flux ↔ source/sink "
                f"pair (a residual, a coefficient, or an abstract base) — "
                f"it has no role."
            )
        return role

    @classmethod
    def role_partner(cls, role: FieldRole) -> "type[Self]":
        r"""The leaf class playing ``role`` on this leaf's carrier.

        Asking for this leaf's OWN role returns this leaf (the identity
        half of the pair), so an operator can spell its output role
        without branching on the operand's.
        """
        partners = getattr(cls, "_role_partners", None)
        if partners is None:
            raise TypeError(
                f"{cls.__name__} is not one half of a flux ↔ source/sink "
                f"pair (a residual, a coefficient, or an abstract base) — "
                f"it has no role partner."
            )
        return partners[role]  # type: ignore[return-value]

    def into_role(
        self, role: FieldRole, values: NDArray, *,
        space: "FunctionSpace | None" = None,
    ) -> "Field":
        r"""A field of ``role`` carrying ``values`` on this field's space
        (or ``space=``), with this field's family fields (``L``,
        ``spatial_moments``, …) carried across — the role transition,
        spelled once.

        The family fields ride via :func:`dataclasses.fields`, so a new
        family field is carried without this verb learning its name.
        """
        target = cast("type[Field]", type(self).role_partner(role))
        carried = {
            f.name: getattr(self, f.name)
            for f in dataclass_fields(cast("Field", self))
            if f.name not in ("values", "space")
        }
        return target(
            values=values,
            space=cast("Field", self).space if space is None else space,
            **carried,
        )



def spatial_moments_per_axis_of(space: FunctionSpace) -> int:
    r"""The per-axis spatial-moment width a SPACE carries (``1`` if none) —
    the module-level name of :meth:`BulkField.spatial_moments_per_axis_of`,
    for the operator and frame tiers that hold only a space (`[M]` 4 of its
    5 call sites are outside ``fields/``; a leading underscore misdescribed
    its audience — the elegance review's N1)."""
    return BulkField.spatial_moments_per_axis_of(space)

# ═══════════════════════════════════════════════════════════════════════
# Bulk locus
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class BulkField(RolePair, Field):
    r"""Bulk-locus storage base — a :class:`Field` on the grid's cell centres.

    Carries the machinery shared by every bulk transport field: the
    per-family carrier-cached space mint (:meth:`_space_for_mesh`, read by
    the factories AND the operator admission guards' :meth:`space_on`
    reference), the optional within-cell spatial-moment factor, and the
    ``ng`` read-through. A bulk field is an ELEMENT of its space (CS4b S4
    — the campaign's name executed): ``(values, space)`` is the whole
    state, the space is carrier-minted and content-named, and every
    structural question (shape, ``ng``, moment width, ordinate count) is
    answered by the space. The pre-S4 ``mesh`` binding retired with its
    reads — a field no longer knows its carrier; the carrier's knowledge
    enters through its cached space mints, read AT the call site
    (``mesh.angular_bulk_space`` / ``bulk_space`` / the traces / the
    scheme-widened ``angular_trial_space`` — the S5 space-primary
    spelling), and through the seams that hold one (the ``space_on``
    admission guards).

    Shape validation is :class:`~orpheus.numerics.field.Field`'s own
    ``values.shape == space.shape`` — the pre-S4 ``_phase_space_shape``
    cross-check re-derived the same shape from the mesh, a twin of the
    space's own content that died with the binding.

    Abstract — instantiate a concrete role leaf (``AngularFlux``,
    ``ScalarFlux``, ...).
    """

    # ── Algebra extension (over Field) ───────────────────────────────

    # CS4b S3 (F2 re-key): the mesh-identity override RETIRED. Partner
    # identity is the base gate's space CONTENT equality
    # (``Field._check_partner``): the carrier-cached axis-built spaces
    # carry the per-cell geometry the retired provenance arm guarded
    # (volumes as the spatial measure, ``ng`` and quadrature as axis
    # content), so two fields mix iff their spaces agree in CONTENT —
    # twin and BC-only-differing carriers now legitimately mix (a
    # boundary law changes neither DOFs nor Gram; laws are operator
    # data), while a moved cell edge, a different group structure, or a
    # different quadrature refuses exactly as before.

    # ── Optional spatial-moment factor (#240 D5b-S3-A0) ──────────────

    @staticmethod
    def _compose_spatial_moments(
        space: FunctionSpace, mesh: "MaterialMesh", spatial_moments_per_axis: int,
    ) -> FunctionSpace:
        r"""Append the optional within-cell spatial-moment factor to ``space``.

        Two arms, discriminated by the base space's composition mechanism
        (campaign 1 CS4b, crosswalk B5):

        * **Axis-built base** (the carrier-cached bulk spaces): the tail is
          the scheme-owned MODAL
          :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`,
          carrying ``moment_mass_diagonal`` as the axis measure — basis ↔
          mass single-sourced at the scheme (``θ`` enters the mass, so a
          carrier that binds no scheme cannot host a moment tail; the
          scheme binds at transport-method augmentation).
        * **Axes-less base** (the harmonic family's
          ``<angular head> * cell_group`` — until CS4c step 6 item 6.2c
          axis-ifies the angular head factor): NOT composed here. Since
          item 6.2b the carrier composes that product's tail itself —
          :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space`
          appends a
          :class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`
          factor via the tensor-product ``*`` (a factored metric, never a
          densified one, since item 6.2a) — and an axes-less input here is
          refused by name, so the tail rule for that product has one home.

        The factor is the Linear-Discontinuous closure's spatial-slope
        carrier that travels between source-iteration sweeps (the
        diffusion-limit-consistent scattering source
        :math:`\Sigma_s \otimes I_{\rm spatial}`).

        **Gated on ``spatial_moments_per_axis > 1`` (construct-general /
        select-narrow, #240 D5b-S3-A0).** The total within-cell moment
        count is ``spatial_moments_per_axis ** mesh.ndim``. When the count
        is ``1``, the ``spatial_moment_tail`` policy returns ``()`` and
        this method returns ``space`` UNCHANGED — the field space stays
        BYTE-IDENTICAL to its pre-S3 shape (the backward-compat invariant,
        single-sourced from
        :func:`orpheus.numerics.moment_layout.face_moment_tail` via
        :func:`~orpheus.numerics.spaces.spatial_moment_space.spatial_moment_tail`).

        ``spatial_moments_per_axis`` is an EXPLICIT parameter (the
        ``spatial_moments`` factory parameter, default ``1`` everywhere),
        NOT auto-read from ``mesh.scheme.spatial_basis_per_axis``. This is
        the construct-general / select-narrow discipline: the CALLER
        selects. Since S3-A landed, the iterate / cell-emit / source seams
        (the SI cold starts, ``coupled_system``, ``windowing``) thread the
        scheme's ``spatial_basis_per_axis`` here explicitly, so LD
        production fields DO carry the axis while DD/Step (per_axis == 1)
        get no factor. Reading the scheme by DEFAULT here would still be
        wrong: the widening must remain the decision of the seams that
        also FILL the axis — a Pattern-4 concern (an axis no producer
        fills is an illegal state).
        """
        n_moments = cell_moment_count(spatial_moments_per_axis, mesh.ndim)
        # "append iff > 1" — single-sourced; () at n==1 → no factor, byte-id.
        if spatial_moment_tail(n_moments) == ():
            return space
        if space.axes is not None:
            # The width thread behind space_on's admission re-mint (and,
            # until S6, the moment family's keyed factories). The public
            # sugar tier retired at S5 — call sites read the carrier
            # mints (angular_bulk_space / angular_trial_space); this
            # private derivation re-mints a FIELD's own width on a
            # carrier for comparison.
            scheme = getattr(mesh, "scheme", None)
            if scheme is None:
                raise TypeError(
                    "a within-cell moment tail needs the discretization "
                    "scheme's mass (θ enters moment_mass_diagonal), and "
                    f"this carrier ({type(mesh).__name__}) binds no scheme "
                    "— moment-tailed fields live on a transport-method "
                    "mesh (the scheme binds at augmentation)."
                )
            axis = scheme.moment_axis(mesh.axes)
            if axis.shape != (n_moments,):
                raise ValueError(
                    f"spatial_moments={spatial_moments_per_axis} requests "
                    f"{n_moments} cell moments, but {type(scheme).__name__} "
                    f"masses {axis.shape[0]} — the moment tail is the "
                    "scheme's basis, so only its own width is mintable."
                )
            return FunctionSpace.of_axes(*space.axes, axis)
        # An axes-less space composes its tail where it is minted — the
        # carrier's ``SNMesh.moment_space`` for the (still axes-less until
        # item 6.2c) harmonic-moment product (CS4c step 6 item 6.2b); this
        # family-side composer serves the axis-built angular/scalar mints
        # only, so an axes-less input here is a caller error, not a case.
        raise TypeError(
            f"the within-cell moment tail is appended to an AXIS-BUILT "
            f"space; {space!r} declares no axes — the harmonic-moment "
            f"product composes its tail at the carrier (SNMesh.moment_space)."
        )

    @staticmethod
    def _spatial_moment_tail_of(space: FunctionSpace) -> tuple[int, ...]:
        r"""The trailing spatial-moment shape suffix carried by ``space``, or ``()``.

        Reads the optional
        :class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`
        factor OFF a composed space — the space is the single source of
        truth for the moment width, so the shape validation (Field's
        values-vs-space check — the pre-S4 ``_phase_space_shape``
        hook's successor) derives
        the expected widened shape from here rather than re-threading the
        factory's ``spatial_moments`` parameter into a stored field
        (Angular/Scalar leaves carry no such field — only the windowed
        :class:`HarmonicMomentFlux` does, because its ``L`` field already
        breaks the uniform-signature contract).

        Returns ``()`` for a non-composed / DD-default space (no factor →
        byte-identical validation prefix), and ``(per_axis ** ndim,)`` when
        a moment factor is present — as the
        :data:`~orpheus.numerics.moment_layout.SPATIAL_MOMENT_AXIS_LABEL`
        axis on an axis-built space (CS4b), or as a
        :class:`SpatialMomentSpace` factor on an axes-less one.
        """
        if space.axes is not None:
            for ax in space.axes:
                if ax.label == SPATIAL_MOMENT_AXIS_LABEL:
                    return ax.shape
            return ()
        find_factor = getattr(space, "find_factor", None)
        if find_factor is None:
            return ()  # a bare FunctionSpace (DD default) — no factor.
        try:
            factor = find_factor(SpatialMomentSpace)
        except KeyError:
            return ()
        return factor.shape

    @property
    def spatial_moments_per_axis(self) -> int:
        r"""The within-cell spatial-moment count per axis carried by this field.

        Reads the ``per_axis`` parameter OFF the optional
        :class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`
        factor on this field's space (the single source of truth for the moment
        width, #240 D5b-S3-A0).  Returns ``1`` for a non-composed / DD-default
        space.  Producers that derive a moment-carrying child field (e.g.
        :meth:`AngularFlux.integrate_angular`,
        :meth:`HarmonicMomentFlux.scalar_flux`) pass this as the child's
        ``spatial_moments`` so the moment axis is propagated as a TYPED factor,
        not an opaque widened ndarray."""
        return type(self).spatial_moments_per_axis_of(self.space)

    @staticmethod
    def spatial_moments_per_axis_of(space: FunctionSpace) -> int:
        r"""The per-axis spatial-moment width a SPACE carries (``1`` if none).

        The static body of :attr:`spatial_moments_per_axis`, hoisted so a
        consumer holding only the space — the S4-amendment's bound
        :class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame`,
        which derives its moment codomain from its bound angular domain at
        construction — reads the SAME rule the fields do (single source;
        the alternative was a second copy of the tail-inversion in the
        frame)."""
        from orpheus.numerics.spaces.spatial_moment_space import SpatialMomentSpace

        tail = BulkField._spatial_moment_tail_of(space)
        if space.axes is not None:
            if tail == ():
                return 1
            # The axis stores the CELL count (per_axis ** ndim); invert it.
            # ndim is the spatial axis's rank — the ONE "spatial" axis
            # carries the whole spatial shape (CS4b S4: the space answers
            # every structural question; the mesh read died with the
            # binding).
            ndim = next(
                len(ax.shape)
                for ax in space.axes
                if ax.label == "spatial"
            )
            per_axis = round(tail[0] ** (1.0 / ndim))
            if per_axis ** ndim != tail[0]:
                raise ValueError(
                    f"moment axis carries {tail[0]} cell moments, which is "
                    f"not a per-axis power for ndim={ndim}"
                )
            return per_axis
        find_factor = getattr(space, "find_factor", None)
        if find_factor is None:
            return 1
        try:
            factor = find_factor(SpatialMomentSpace)
        except KeyError:
            return 1
        return factor.per_axis

    @classmethod
    def _space_for_mesh(
        cls, mesh: "MaterialMesh", *, spatial_moments: int = 1,
    ) -> FunctionSpace:
        r"""The family's space mint for ``mesh`` — the per-family hook.

        Implemented by :class:`AngularField` / :class:`ScalarField` (the
        carrier-cached reads); :class:`MomentField` keys on ``(mesh, L,
        width)`` instead and overrides :meth:`space_on` directly — since
        CS4c step 6 item 6.2b also a carrier-cached read
        (:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space`).
        """
        raise NotImplementedError(
            f"{cls.__name__} declares no per-mesh space mint — instantiate "
            "a concrete family (AngularField/ScalarField subclasses), or "
            "the moment family's keyed read (MomentField.space_on)."
        )

    def space_on(self, mesh: "MaterialMesh") -> FunctionSpace:
        r"""The space THIS field's family (and moment width) mints on ``mesh``.

        The polymorphic reference of the operator admission guards
        (CS4b S3): *"does your space agree with what your family would be
        on MY carrier?"* — spelled once per family hierarchy, single-
        sourced through the same derivation the factories use, so a guard
        can compare content without knowing which role family it holds
        (angular, scalar, moment, face — each answers with its own mint).
        """
        return type(self)._space_for_mesh(
            mesh, spatial_moments=self.spatial_moments_per_axis,
        )

    # ── Metadata read-throughs ───────────────────────────────────────

    @property
    def ng(self) -> int:
        r"""Number of energy groups — read off the SPACE's energy axis.

        CS4b S3: the space answers every structural question (sizing is
        space data — XD-10); the mesh delegation retired with the reads
        S4 deleted. The moment family (axes-less TensorProduct until CS4c
        step 6 item 6.2c) overrides with its own read.
        """
        axes = self.space.axes
        if axes is not None:
            for ax in axes:
                if isinstance(ax, EnergyAxis):
                    return int(ax.shape[0])
        raise TypeError(
            f"{type(self).__name__}.ng: no EnergyAxis on this space — an "
            "axes-less family must override ng with its own space read "
            "(MomentField does)."
        )

    # C5.2 (#225): the ``nx``/``ny`` read-throughs are RETIRED — a
    # field keyed on ``(nx, ny)`` silently truncates a 3-D tensor.
    # Spatial shape reads are rank-generic: ``mesh.spatial_shape``.


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class AngularField(BulkField):
    r"""Per-ordinate bulk family on ``(N, ng, nx, ny)``.

    The storage base for the angular role leaves (``AngularFlux``,
    ``AngularSourceSink``, ``AngularResidual``). The family shares ONE
    space — the carrier's cached
    :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space`
    (campaign 1 CS4b: role is class identity, never space identity; the
    class arm of :meth:`~orpheus.numerics.field.Field._check_partner` is
    the sole role gate). Abstract — instantiate a concrete leaf.
    """

    @classmethod
    def _space_for_mesh(  # type: ignore[override] — the family narrows its
        # carrier (SNMesh), the same #267 covariant-override doctrine as the
        # ``mesh`` field above; every caller passes this family's carrier.
        cls, mesh: "SNMesh", *, spatial_moments: int = 1,
    ) -> FunctionSpace:
        r"""The leaf's :class:`FunctionSpace` for ``mesh``.

        Reads the carrier's cached, axis-built
        :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.angular_bulk_space`
        (campaign 1 CS4b — the carrier is the ONE mint; every angular
        leaf on one carrier shares the SAME space instance, carrying the
        physical Hilbert metric ``w_n × V_cell`` per axis). The private
        derivation behind :meth:`space_on`'s admission re-mint (the
        public sugar tier retired at CS4b S5 — call sites read the
        carrier mints directly).

        ``spatial_moments`` (default ``1``) is the optional within-cell
        spatial-moment basis size per axis (#240 D5b-S3-A0). At the
        default ``1`` the space IS the cached instance; at ``> 1`` the
        scheme-owned MODAL moment axis is composed on (see
        :meth:`BulkField._compose_spatial_moments`).
        """
        return cls._compose_spatial_moments(
            mesh.angular_bulk_space, mesh, spatial_moments,
        )

    def _integrate_angular_values(self) -> "NDArray":
        r"""The ONE moment-0 reduction body :math:`\sum_n w_n\,(\cdot)_n`.

        Contracts the leading ``N`` axis with the quadrature weights
        ``w_n`` — ``(N, ng, nx, ny[, 2^d]) → (ng, nx, ny[, 2^d])``. The
        ``ng...`` einsum is spatial-moment-axis-agnostic, so a
        φ̂-carrying field reduces to a φ̂-carrying scalar (#240 D5b-S3);
        the moment width propagates as a TYPED factor read off this
        field's space, not an opaque axis.

        Role leaves wrap this body in their own scalar type
        (:meth:`AngularFlux.integrate_angular
        <orpheus.transport.fields.angular_flux.AngularFlux.integrate_angular>`
        → ``ScalarFlux``; until campaign 1 CS3 the displacement sibling
        wrapped the same body — a linear reduction is its own tangent
        map). Values-level, role-blind, single source of truth for the
        canonical angular reduction (the DSA restriction ``R`` rides it,
        #2).

        Since CS4b S6.2 the realization IS the space's memoized
        frame-induced retraction (:meth:`FunctionSpace.retraction
        <orpheus.numerics.space.FunctionSpace.retraction>` — the
        rank-one indicator frame's analysis content; `[M]` bit-identical
        with the pre-S6.2 einsum spelling, G6.5), so the canonical
        reduction has ONE realization tree-wide and admission
        (axis-built space, NODAL angular axis) lives at the mint.
        """
        return self.space.retraction("angular").apply(self.values)

    @property
    def N(self) -> int:  # noqa: N802 — matches Quadrature.N
        r"""Number of angular ordinates — the space's leading angular axis.

        CS4b S3: sizing is space data; the axis order convention
        (angular, energy, spatial[, moment]) makes this ``axes[0]``.
        """
        axes = self.space.axes
        if axes is not None:
            return int(axes[0].shape[0])
        raise TypeError(
            f"{type(self).__name__}.N: no axes on this space — every "
            "shipped angular space is axis-built (S2)."
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class ScalarField(BulkField):
    r"""Scalar bulk family on ``(ng, nx, ny)``.

    The storage base for the scalar role leaves (``ScalarFlux``,
    ``ScalarSourceSink``, ``ScalarResidual``, ``CrossSectionField``). The
    family shares ONE space — the carrier's cached
    :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space`
    (campaign 1 CS4b: role is class identity, never space identity).
    Abstract — instantiate a concrete leaf.
    """

    @classmethod
    def _space_for_mesh(
        cls, mesh: "MaterialMesh", *, spatial_moments: int = 1,
    ) -> FunctionSpace:
        r"""The leaf's :class:`FunctionSpace` for ``mesh``.

        Reads the carrier's cached, axis-built
        :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space`
        (campaign 1 CS4b — the carrier is the ONE mint; every scalar leaf
        on one carrier shares the SAME space instance, carrying the
        cell-volume measure on the spatial axis). The private
        derivation behind :meth:`space_on`'s admission re-mint (the
        public sugar tier retired at CS4b S5 — call sites read the
        carrier mints directly).

        ``spatial_moments`` (default ``1``) is the optional within-cell
        spatial-moment basis size per axis (#240 D5b-S3-A0); at ``> 1``
        the scheme-owned MODAL moment axis is composed on (see
        :meth:`BulkField._compose_spatial_moments`). The
        :class:`ScalarSourceSink` scattering-source accumulator is the
        carrier that selects ``> 1`` at S3-A so the slope rows can hold
        :math:`\Sigma_s \cdot \hat\phi`.
        """
        return cls._compose_spatial_moments(
            mesh.bulk_space, mesh, spatial_moments,
        )


@runtime_checkable
class _CarriesMomentSpace(Protocol):
    """A carrier that OWNS its harmonic-moment spaces — the SN hub's surface
    (:meth:`orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space`, CS4c step 6
    item 6.2b): one cached space per ``(L, spatial_moments)``, the angular
    head READ off the carrier's quadrature frame (#429 tracker 2.5), the
    cell group its own ``bulk_space``."""

    def moment_space(
        self, L: int, *, spatial_moments: int = 1,
    ) -> FunctionSpace: ...


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class MomentField(BulkField):
    r"""Real-spherical-harmonic moment-space bulk family (storage base).

    The storage base for the moment role leaves —
    :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
    (the moment-space flux state) and
    :class:`~orpheus.transport.source_sinks.harmonic_moment_source_sink.HarmonicMomentSourceSink`
    (the bare source/sink). It carries the construction machinery the two
    share: the ``L`` truncation-order + ``spatial_moments`` fields, the
    ``(L+1, 2L+1, ng, *spatial[, …])`` shape contract (validated by
    Field's values-vs-space check, the pre-S4 ``_phase_space_shape``
    hook's successor), the
    ``L``-match :meth:`_check_partner`, and the
    :class:`~orpheus.numerics.space.TensorProductSpace`-building
    :meth:`from_mesh_and_L` / :meth:`zeros_for_mesh_and_L` factories
    (keyed, S6 re-home pending; the positional ``from_ndarray`` alias
    retired with the S5 sugar tier).

    A moment field is a moment field on the spherical-harmonic ⊗
    scalar-bulk phase space, keyed on the truncation order ``L``; its
    space is the CARRIER's cached
    :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space` at
    ``(L, width)`` — a TensorProductSpace whose angular head is read off
    the carrier's quadrature frame and whose cell-group factor IS the
    carrier's cached
    :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.bulk_space`
    (campaign 1 CS4b — one mint; CS4c step 6 item 6.2b — one OBJECT per
    key, so the two role leaves, the admission guards and the sweep's
    iterate wrap all hold the same instance; role is class identity,
    exactly as in the Angular/Scalar families). Abstract — instantiate a
    concrete role leaf.

    This lift happened when the second moment representation arrived
    (``feedback_unify_after_two_instances``): the machinery used to live on
    the lone ``HarmonicMomentField`` leaf; the Frame-campaign P4
    source/sink sibling triggered the "clean before extending" pass.
    """

    #: Maximum harmonic order retained. Determines the angular HEAD's axes —
    #: ``values.shape[:2] == (L+1, 2L+1)`` for the harmonic family the
    #: full-sphere rules bind (a flat head, one axis, once a 1-D rule binds
    #: its Legendre basis — #429 tracker 3.4). Encoded in ``space.shape``
    #: AND kept as a top-level field for ergonomic hot-path read access
    #: (avoids a per-read composition-tree traversal of the head factor's
    #: own ``L``).
    L: int

    #: Optional within-cell spatial-moment basis size per axis (#240
    #: D5b-S3-A0). Default ``1`` — the cell-average closure, byte-identical
    #: to the pre-S3 ``(L+1, 2L+1, ng, *spatial)`` shape. At ``> 1`` (the
    #: LD windowed iterate) a trailing ``spatial_moments ** ndim`` axis rides
    #: on every moment so the in-sweep ``moment_buf`` can carry the
    #: within-cell slopes the diffusion-limit-consistent scattering source
    #: needs between sweeps. Single-sourced "append iff > 1" via
    #: :func:`~orpheus.numerics.spaces.spatial_moment_space.spatial_moment_tail`.
    spatial_moments: int = 1

    # ── Metadata read-through (the axes-less family's own ng) ────────

    @property
    def head(self) -> "MomentHead":
        r"""The angular HEAD factor of this field's space — the family that
        says which index tuple is the isotropic slot, which selects a degree
        block, and how many leading axes it owns (#429, 2026-09-02:
        ``(L+1, 2L+1)`` for the real harmonics, ``(L+1,)`` for the Legendre
        basis a 1-D rule binds). Every layout read on the carrier goes
        through it; none assumes the rectangular family."""
        from orpheus.numerics.space import TensorProductSpace
        from orpheus.numerics.spaces.moment_head import MomentHead

        if not isinstance(self.space, TensorProductSpace):
            raise TypeError(
                f"{type(self).__name__}: a moment field's space is a tensor "
                f"product <angular head> ⊗ cells; got "
                f"{type(self.space).__name__}."
            )
        head = self.space.factors[0]
        if not isinstance(head, MomentHead):
            raise TypeError(
                f"{type(self).__name__}: the leading factor {head.name!r} "
                f"is not an angular head (no isotropic slot / degree block "
                f"surface)."
            )
        return head

    @property
    def ng(self) -> int:
        r"""Number of energy groups — the axis right after the angular head's.

        The moment family's space is a TensorProductSpace (axes-less until
        CS4c step 6 item 6.2c axis-ifies the angular factor), so the base's
        EnergyAxis read has
        nothing to find; the family's OWN shape contract
        ``(<head>, ng, *spatial[, …])`` locates ``ng`` right after the
        head's axes — index 2 for the rectangular harmonics, 1 for a flat
        Legendre head.
        """
        return int(self.space.shape[len(self.head.shape)])

    # ── Algebra extension (over BulkField) ───────────────────────────

    def _check_partner(self, other: Field) -> Self:
        r"""Add the ``L``-match on top of the base class/space-content gate.

        :meth:`Field._check_partner` already rejects on class identity and
        space CONTENT equality (the S3 re-key; the mesh-identity arm is
        retired). This override adds an explicit ``L`` match for a clearer
        error message at the truncation-mismatch site (the space check
        also catches it via shape mismatch, but the message is less
        specific).
        """
        partner = super()._check_partner(other)
        if self.L != partner.L:
            raise ValueError(
                f"{type(self).__name__} arithmetic requires matching L; "
                f"got self.L={self.L}, other.L={partner.L}."
            )
        return partner

    # ── Construction factories ───────────────────────────────────────

    @classmethod
    def from_mesh_and_L(
        cls, values: NDArray, mesh: "SNMesh", L: int, *, spatial_moments: int = 1,
    ):
        r"""Construct from raw values + mesh + L on the carrier's own
        moment space.

        The space is READ off the carrier —
        :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space`
        (CS4c step 6 item 6.2b): ``<angular head> * mesh.bulk_space``, the
        angular head read off the carrier's quadrature frame at ``L``
        (``mesh.quad.angular_frame(L).basis.space``; the spherical-harmonic
        space on a full-sphere rule — #429 tracker 2.5, never minted from
        ``L``), the cell-group factor the carrier's cached scalar bulk
        (campaign 1 CS4b: one mint, metric-carrying), ONE object per
        ``(L, width)`` per carrier; the moment-axis structure is
        type-visible through the composition tree (queryable via
        ``space.find_factor(...)`` per Issue #207).

        ``spatial_moments`` (default ``1``, byte-identical #240 D5b-S3-A0)
        optionally composes a within-cell
        :class:`~orpheus.numerics.spaces.spatial_moment_space.SpatialMomentSpace`
        factor on AFTER the cell-group space — EXACTLY the same ``*``
        composition that adds the angular head ("append iff > 1",
        single-sourced with the space's own shape contract).
        """
        return cls(
            values=values,
            space=cls._space_for_mesh_and_L(
                mesh, L, spatial_moments=spatial_moments,
            ),
            L=L,
            spatial_moments=spatial_moments,
        )

    @classmethod
    def _space_for_mesh_and_L(
        cls, mesh: "MaterialMesh", L: int, *, spatial_moments: int = 1,
    ) -> FunctionSpace:
        r"""The moment family's space for ``(mesh, L, width)`` — READ off the
        carrier, which owns it (CS4c step 6 item 6.2b, 2026-09-07).

        The carrier's :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space`
        is a keyed cache: every read of one ``(carrier, L, width)`` returns
        the SAME object, so the factory (:meth:`from_mesh_and_L`), the
        admission-guard reference (:meth:`space_on`) and the sweep's
        iterate wrap share one instance — ``is``, not merely ``==`` — and
        nothing is re-minted per call (`[M]` until 6.2b this method
        minted ``<head> * bulk_space`` on every call: 113 of the 118
        ``*`` products per 2-D windowed solve, 58 from the boundary
        leaf's guard and 55 from the sweep's iterate wrap).

        A carrier that owns no moment space (a transport ``MaterialMesh``
        alone — no quadrature, no angular head to read) cannot host a
        moment field, and says so.
        """
        if not isinstance(mesh, _CarriesMomentSpace):
            raise TypeError(
                f"a moment field's space is READ off the SN carrier that "
                f"owns it (SNMesh.moment_space), and {type(mesh).__name__} "
                f"carries no quadrature and owns no moment space; build the "
                f"moment field on the SN phase-space carrier (an SNMesh)."
            )
        return mesh.moment_space(L, spatial_moments=spatial_moments)

    def space_on(self, mesh: "MaterialMesh") -> FunctionSpace:
        r"""The moment family's space on ``mesh`` (see BulkField.space_on) —
        the carrier's own cached object, never a re-mint."""
        return type(self)._space_for_mesh_and_L(
            mesh, self.L, spatial_moments=self.spatial_moments,
        )

    @classmethod
    def zeros_for_mesh_and_L(
        cls, mesh: "SNMesh", L: int, *, spatial_moments: int = 1,
    ):
        r"""Construct a zero moment field at order ``L`` sized to ``mesh`` (B.5.A).

        Mirrors :meth:`from_mesh_and_L` with a zero buffer. The extra
        ``L`` makes the signature keyed rather than uniform — a moment
        field is never a
        :class:`~orpheus.transport.timed_full_field.TimedFullField`
        composite slot, so it never needed the (since-retired, S5)
        uniform mesh-keyed allocator surface; its own re-home is S6.

        ``spatial_moments`` (default ``1``, byte-identical #240 D5b-S3-A0)
        sizes the optional within-cell spatial-moment axis to match
        :meth:`from_mesh_and_L`.
        """
        space = cls._space_for_mesh_and_L(
            mesh, L, spatial_moments=spatial_moments,
        )
        return cls(
            values=np.zeros(space.shape),
            space=space,
            L=L,
            spatial_moments=spatial_moments,
        )


# ═══════════════════════════════════════════════════════════════════════
# Face locus (codim-1) — the shared flat-buffer discipline
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class FaceField(RolePair, Field, Generic[K]):
    r"""Codim-1 face storage base — a mesh-bound flat-buffer :class:`Field`
    on a layout-bearing face space (method- and locus-agnostic).

    The single parent of every **codim-1** transport field — the faces and
    edges bounding a phase-space cell, as opposed to :class:`BulkField`'s
    codim-0 cell centres. It owns, ONCE, the flat-buffer discipline the two
    codim-1 loci share: the single-buffer storage contract
    (``values.shape == (layout.total_size,)``), per-face slice-view access
    keyed by the layout key ``K`` (a ``str`` face name for the spatial trace,
    a ``(level, sign, part)`` tuple for the ψ½ pole edge), the cross-mesh /
    cross-layout arithmetic guards, the read-through :attr:`layout` property,
    and the single :meth:`_face_space_of` space hook — the family's cached
    face-space read behind :meth:`space_on` and the family diagnoses (the
    mesh-keyed factories retired at CS4b S5; call sites read the carrier's
    cached face space directly).

    **STRUCTURE only — no metric on this ABC.** Exactly as :class:`BulkField`
    carries no metric (each bulk leaf's ``V·w`` lives on its own space), a
    face field's Hilbert metric descends **per leaf**, on the leaf's face
    space: the spatial trace carries the partial-current ``|Ω·n̂|·w`` metric
    (:class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`),
    the ψ½ pole the SPD radial cell-volume STATE metric :math:`V_{\rm cell}`
    (the split ψ½ spaces in :mod:`orpheus.numerics.spaces.radial_characteristic_space`).
    The through-flux coefficient is NOT the state metric (ERR-067, vv Mode 12)
    — so this ABC deliberately unifies the *layout*, never the *measure*.

    Two codim-1 loci realize the base as **siblings** (NOT parent/child — the
    :class:`~orpheus.transport.full_field.FullField` composite discriminates
    its boundary slot by ``isinstance(·, BoundaryField)``, a test the ψ½ pole
    must FAIL):

    * :class:`BoundaryField` — the **spatial** faces (``FaceField[str]``): the
      boundary of the SPATIAL domain, keyed by face name.
    * the ψ½ split loci (:class:`RadialCharacteristicInteriorField` /
      :class:`RadialCharacteristicBoundaryField`) — the **angular** edge
      (``FaceField[tuple[int, int, str]]``): the ψ½ pole seed, the boundary of
      the ANGULAR domain (:math:`\mu = \mu_{\rm start}`), keyed by
      ``(level, sign, part)``.

    A face field is an ELEMENT of its layout-bearing face space (CS4b S4
    — ``(values, space)`` is the whole state; the pre-S4 ``mesh`` binding
    retired with its reads, and the carrier's knowledge enters through
    the factories' mesh ARGUMENT and the ``space_on`` admission seams).
    Storage is a SINGLE flat backing buffer; per-face access is
    slice-view, no copies. Inherits Field's same-class/same-space dunder
    algebra. Abstract — instantiate a concrete role leaf of one of the loci.
    """

    # ── The per-family face-space source ─────────────────────────────

    @classmethod
    @abstractmethod
    def _face_space_of(cls, mesh: "MaterialMesh") -> FunctionSpace:
        r"""Return ``mesh``'s cached face space for this family.

        The single hook the factories construct through: the spatial
        families read ``mesh.angular_trace`` (raising on the trace-less 2-D
        cylindrical mesh) / ``mesh.scalar_trace`` (a :class:`DiffusionMesh`
        member, raising on a bare MaterialMesh — #290 P7a); the
        starting-direction family reads the split ψ½ mesh spaces
        (R12a-keyed). MUST return a layout-bearing space or raise
        :class:`ValueError` with the family's own diagnosis.
        """
        raise NotImplementedError

    # ── Construction validation ──────────────────────────────────────

    def __post_init__(self) -> None:
        super().__post_init__()  # Field: values.shape == space.shape.
        # The space IS the face space and carries the FaceLayout
        # (illegal-states-unrepresentable): a face field on a layout-less
        # space cannot do face_view. Families ADD their space-type narrowing
        # (AngularTraceSpace / ScalarTraceSpace / the split ψ½ spaces) on
        # top of this structural floor.
        layout = getattr(self.space, "layout", None)
        if layout is None:
            raise TypeError(
                f"{type(self).__name__} requires a layout-bearing face "
                f"space (a trace / starting-direction space built via its "
                f"factory); got space={self.space!r}. Construct on the "
                f"carrier's cached face space (mesh.angular_trace / "
                f"mesh.scalar_trace / the split ψ½ spaces)."
            )
        expected = (layout.total_size,)
        if self.values.shape != expected:
            raise ValueError(
                f"{type(self).__name__}: values.shape {self.values.shape!r} "
                f"does not match (layout.total_size,) = {expected!r}"
            )
        if self.space.shape != expected:
            raise ValueError(
                f"{type(self).__name__}: space.shape {self.space.shape!r} "
                f"does not match (layout.total_size,) = {expected!r}"
            )

    # ── Algebra extension (over Field) ───────────────────────────────

    def _check_partner(self, other: Field) -> Self:
        partner = super()._check_partner(other)
        # CS4b S3 (F2 re-key): the mesh-identity arm RETIRED. The base
        # gate's space CONTENT equality carries the discrimination: since
        # S3 the trace-space names fold a content digest (layout identity,
        # quadrature weights/directions, face areas), so ``(name, shape)``
        # equality IS content equality — same-boundary carriers mix
        # whatever their interiors or boundary LAWS; a different layout,
        # quadrature, or face geometry refuses.
        # The pre-CS4b layout arm (an ``is``/structural fallback) RETIRED
        # with the content digest: the trace-space name now folds the
        # layout's structural identity, so ``space ==`` already implies
        # layout-content equality — no input can reach a layout mismatch
        # past the base's space arm ([M] the S3 battery measured the arm
        # blind; keeping it would be a gate that cannot fail).
        return partner

    def space_on(self, mesh: "MaterialMesh") -> FunctionSpace:
        r"""The face family's mint on ``mesh`` (see BulkField.space_on)."""
        return type(self)._face_space_of(mesh)

    # ── Per-face access (slice views into the flat buffer) ───────────

    @property
    def layout(self) -> "FaceLayout[K]":
        r"""The per-geometry :class:`FaceLayout`, read off the space.

        The layout lives on the face space (``space.layout``), not as a
        separate field attribute. This read-through property preserves the
        ``field.layout`` access surface (``field.layout.faces``,
        ``.total_size``).
        """
        return self.space.layout  # type: ignore[attr-defined]

    def face_view(self, key: K) -> NDArray:
        r"""Return a per-face slice view into the flat backing buffer.

        The returned ndarray shares memory with :attr:`values`.

        Raises
        ------
        KeyError
            If ``key`` is not a face in this layout.
        """
        if key not in self.layout.faces:
            raise KeyError(
                f"{type(self).__name__}: no face keyed {key!r} in layout; "
                f"available: {list(self.layout.faces)!r}"
            )
        return self.layout.faces[key].slice_view(self.values)

    @property
    def face_views(self) -> Mapping[K, NDArray]:
        r"""Mapping ``{face_key: slice_view}`` for every face in the layout.

        All views memory-shared with :attr:`values`.
        """
        return {key: self.face_view(key) for key in self.layout.faces}


# ═══════════════════════════════════════════════════════════════════════
# Boundary locus (spatial faces)
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class BoundaryField(FaceField[str]):
    r"""Spatial-face storage base — the SPATIAL locus of :class:`FaceField`.

    The boundary of the SPATIAL domain: codim-1 faces keyed by name
    (``"xmin"`` / ``"xmax"`` / ...), under the partial-current
    ``|Ω·n̂|·w`` metric. The flat-buffer discipline (storage, slice views,
    mesh/layout guards, the :meth:`~FaceField.space_on` hook) is
    inherited from :class:`FaceField`; this
    intermediate (a) adds the spatial-only :meth:`from_face_arrays` per-face
    packer, and (b) is the type the
    :class:`~orpheus.transport.full_field.FullField` composite discriminates
    its boundary slot by — the ψ½ pole (the RadialCharacteristic split loci, a
    :class:`FaceField` SIBLING) is deliberately NOT a ``BoundaryField``.

    Two storage families realize the SPATIAL locus (#290 P2; named per the
    P2.5 axis-coherence ruling — family-qualified, uniform role tokens):

    * :class:`AngularBoundaryField` — the ANGULAR family (``mesh`` narrowed to
      :class:`SNMesh`, space to the quadrature-coupled
      :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`).
    * :class:`ScalarBoundaryField` — the SCALAR family (``mesh`` narrowed to
      :class:`DiffusionMesh`, space to
      :class:`~orpheus.numerics.spaces.scalar_trace_space.ScalarTraceSpace`;
      per-face ``(J⁺, J⁻)`` partial-current pairs).

    A boundary trace is method BEHAVIOR (#290 P7a): the bare MaterialMesh
    data carrier owns no trace, so a boundary field on one is
    unrepresentable (:meth:`~FaceField._face_space_of` raises). Abstract —
    instantiate a concrete role leaf of one of the families.
    """

    # ── The spatial-only per-face packer (str-keyed) ─────────────────

    @classmethod
    def from_face_arrays(
        cls, mesh: "MaterialMesh", face_arrays: Mapping[str, NDArray],
    ):
        r"""Construct from per-face ndarrays, packing into the flat layout.

        ``face_arrays`` must cover EVERY face in the mesh's layout; each
        per-face ndarray's shape must match the layout slot's shape. Spatial
        faces only — the ψ½ pole builds through its role factories, not a
        per-face dict.

        The packing loop is the layout's own
        :meth:`~orpheus.numerics.face_layout.FaceLayout.pack` (native
        place, CS4b S6.2) — this factory contributes exactly what the
        layout cannot know: WHICH space the mesh mints for this family
        (the space-level admission below) and the typed field wrap.

        Raises
        ------
        ValueError
            If ``face_arrays`` keys differ from the mesh's layout faces,
            or any per-face ndarray's shape mismatches the layout slot.
        """
        space = cls._face_space_of(mesh)
        layout = getattr(space, "layout", None)
        if layout is None:
            raise ValueError(
                f"{cls.__name__}.from_face_arrays: the trace space carries "
                f"no FaceLayout (a bare-constructor space, not one built "
                f"via its factory). A trace field cannot be packed "
                f"without a face layout."
            )
        return cls(values=layout.pack(face_arrays), space=space)


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class AngularBoundaryField(BoundaryField):
    r"""Angular boundary-trace storage base — the SN family of the
    :class:`BoundaryField` locus.

    Carries what is ANGULAR about the locus: the :class:`SNMesh`
    binding (covariant narrowing of the base ``mesh: MaterialMesh``)
    and the space narrowing to the unified quadrature-coupled
    :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace` (the
    ``|Ω·n̂|⊙w_n`` partial-current metric + the ``omega_dot_n``
    selector table). All storage/factory/guard machinery is inherited
    from :class:`BoundaryField`. The concrete role leaves are
    ``AngularBoundaryFlux``, ``AngularBoundarySourceSink``, ``AngularBoundaryResidual``.
    Abstract — instantiate a role leaf.
    """

    # The static twin of the __post_init__ isinstance gate below (the
    # ``mesh: SNMesh`` covariant-narrowing idiom): an angular boundary
    # field's space IS the quadrature-coupled AngularTraceSpace, so
    # consumers of its atoms (``omega_dot_n``, the face layout, the
    # partial-current metric) type-check without re-narrowing.
    space: AngularTraceSpace

    # ── Construction validation (angular narrowing) ──────────────────

    def __post_init__(self) -> None:
        # A.5: the space IS the unified angular AngularTraceSpace. The
        # isinstance narrowing runs FIRST so the family-specific
        # message fires for the common misuse (a bare FunctionSpace or
        # a scalar trace passed where the angular trace belongs).
        if not isinstance(self.space, AngularTraceSpace):
            raise TypeError(
                f"{type(self).__name__} requires an AngularTraceSpace carrying a "
                f"FaceLayout (A.5 re-home); got space={self.space!r}. Build "
                f"via {type(self).__name__}.from_face_arrays, or pass "
                f"mesh.angular_trace as the space."
            )
        super().__post_init__()

    # ── The angular trace-space source (``mesh.angular_trace``) ──────────────

    @classmethod
    def _face_space_of(cls, mesh: "MaterialMesh") -> AngularTraceSpace:
        space = getattr(mesh, "angular_trace", None)
        if space is None:
            raise ValueError(
                f"{cls.__name__}: mesh has no AngularTraceSpace (mesh.angular_trace is "
                f"None — only trace-less 2-D cylindrical meshes, which "
                f"have no SN sweep, hit this). A boundary field cannot "
                f"be built without a boundary trace."
            )
        return space


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class ScalarBoundaryField(BoundaryField):
    r"""Scalar boundary-trace storage base — the ``(J⁺, J⁻)`` family of
    the :class:`BoundaryField` locus (#290 P2).

    Carries what is SCALAR about the locus: the space narrowing to
    :class:`~orpheus.numerics.spaces.scalar_trace_space.ScalarTraceSpace`
    (per-face partial-current pairs under the face-AREA metric), the
    :class:`DiffusionMesh` binding (covariant narrowing of the base
    ``mesh: MaterialMesh`` — the exact :class:`AngularBoundaryField` /
    ``SNMesh`` discipline; #290 P7a), and the trace-space source
    ``mesh.scalar_trace``. A scalar trace lives on the DIFFUSION phase
    space — when DSA (#2) restricts an SN solve, the SN mesh promotes
    (``DiffusionMesh.from_material_mesh(sn_mesh)`` — an ``SNMesh`` IS a
    ``MaterialMesh``) and :math:`A_{\rm diff}`'s fields bind to the
    promoted mesh. The concrete role leaves are
    :class:`~orpheus.transport.fields.scalar_boundary_flux.ScalarBoundaryFlux`
    (the state — and, since campaign 1 CS3, its own iterate increments:
    differences are same-typed); source/residual siblings join when their
    operator codomains demand them (#290 P4). Abstract — instantiate a
    role leaf.
    """

    # ── Construction validation (scalar narrowing) ────────────────────

    def __post_init__(self) -> None:
        # The family's space narrowing runs FIRST so the family-specific
        # message fires for the common misuse (an angular AngularTraceSpace or
        # a bare FunctionSpace passed where the scalar trace belongs).
        if not isinstance(self.space, ScalarTraceSpace):
            raise TypeError(
                f"{type(self).__name__} requires a ScalarTraceSpace (the "
                f"(J⁺, J⁻) partial-current trace); got space="
                f"{self.space!r}. Build via "
                f"{type(self).__name__}.from_face_arrays, or pass "
                f"mesh.scalar_trace as the space."
            )
        super().__post_init__()

    # ── The scalar trace-space source (``mesh.scalar_trace``) ─────────

    @classmethod
    def _face_space_of(cls, mesh: "MaterialMesh") -> ScalarTraceSpace:
        space = getattr(mesh, "scalar_trace", None)
        if space is None:
            raise ValueError(
                f"{cls.__name__}: mesh carries no scalar trace "
                f"(mesh.scalar_trace) — a bare MaterialMesh is the "
                f"method-agnostic DATA carrier; the (J⁺, J⁻) boundary "
                f"trace is diffusion method BEHAVIOR (#290 P7a). Build "
                f"the field on a DiffusionMesh "
                f"(DiffusionMesh.from_material_mesh promotes)."
            )
        return space


# ═══════════════════════════════════════════════════════════════════════
# The SPLIT ψ½ loci — System B's interior ⊕ boundary (Phase B)
# ═══════════════════════════════════════════════════════════════════════
#
# The coupled-block campaign poses the ψ½ ray as System B — its OWN
# interior ⊕ boundary composite — split into two INDEPENDENT loci, exactly
# as the spatial domain splits into BulkField (interior) and BoundaryField
# (boundary). They are FaceField SIBLINGS, not parent/child of each other,
# and each keys by (level, sign) on its own split space. The concrete role
# leaves (…Flux state / …SourceSink emission) live in fields/ and
# source_sinks/ (the …Displacement increment family retired at CS3). The historical
# UNIFIED base (cells ⊕ corner interleaved on one
# FaceField[(level, sign, part)] buffer) retired at 4e, when the fused
# (L+C) walk went split-native — System B's composite, which took the
# freed ``RadialCharacteristicField`` name at 4e-e1b, is now the ONLY ψ½
# representation.


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class RadialCharacteristicInteriorField(FaceField[tuple[int, int]]):
    r"""System B's INTERIOR locus — the ψ½ cells (Phase B split).

    The ``(ng, nx)`` half-angle flux at every radial cell, per seed-carrying
    ``(level, sign)`` leg — the marched interior state that
    :class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator`
    (A_BB) evolves (μ∂_r + σ_t), that A_AB reads (inward leg) and A_BA writes.
    The interior half of the ψ½ split; a ``FaceField[tuple[int, int]]`` keyed by
    ``(level, sign)`` on the
    :class:`~orpheus.numerics.spaces.radial_characteristic_space.RadialCharacteristicInteriorSpace`
    (SPD ``G_sd = V_cell`` state metric). A :class:`FaceField` **sibling** of the
    boundary locus :class:`RadialCharacteristicBoundaryField`, NOT a child (like
    :class:`BulkField` vs :class:`BoundaryField`). ``mesh`` is :class:`SNMesh` and
    the space source is the R12a-keyed ``mesh.radial_characteristic_interior_space``
    — construction on a non-carrying mesh is unrepresentable (the factory raises;
    the composite spells absence as ``None``). The concrete role leaves are
    ``RadialCharacteristicInteriorFlux`` (state — same-class signed differences
    carry the iterate increment since campaign 1 CS3, 2026-08-19, when the
    displacement sibling retired) and
    ``RadialCharacteristicInteriorSourceSink`` (emission). Abstract
    — instantiate a role leaf.
    """

    space: RadialCharacteristicInteriorSpace

    def __post_init__(self) -> None:
        if not isinstance(self.space, RadialCharacteristicInteriorSpace):
            raise TypeError(
                f"{type(self).__name__} requires a "
                f"RadialCharacteristicInteriorSpace (read off "
                f"mesh.radial_characteristic_interior_space); got "
                f"space={self.space!r}."
            )
        super().__post_init__()  # FaceField: Field shape + layout-bearing check.

    @classmethod
    def _face_space_of(
        cls, mesh: "MaterialMesh",
    ) -> RadialCharacteristicInteriorSpace:
        r"""Return ``mesh``'s cached ψ½ interior space, or raise (R12a)."""
        space = getattr(mesh, "radial_characteristic_interior_space", None)
        if space is None:
            raise ValueError(
                f"{cls.__name__}: mesh carries no "
                f"radial_characteristic_interior_space — no μ-level consumes "
                f"independent starting-direction state (R12a; Cartesian and the "
                f"production cylinder rules land here). System B's interior block "
                f"is spelled absent, never a zero-DOF field."
            )
        return space

    @property
    def levels(self) -> tuple[int, ...]:
        r"""The seed-carrying μ-level indices (read off the space)."""
        return self.space.levels

    def cells(self, level: int, sign: int) -> NDArray:
        r"""The ``(ng, nx)`` ψ½ cells view for ``(level, sign)`` — memory-shared."""
        return self.space.slot_view(self.values, level, sign)


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)
class RadialCharacteristicBoundaryField(FaceField[tuple[int, int]]):
    r"""System B's BOUNDARY locus — the ψ½ r = R corner (Phase B split).

    The ``(ng,)`` half-angle flux at the outer radius r = R, per seed-carrying
    ``(level, sign)`` leg — System B's BC locus, on which
    :class:`~orpheus.sn.operators.boundary.RadialCharacteristicBoundaryOperator`
    (B_b) acts. Inflow corner (``sign = -1``) is the given BC data; outflow corner
    (``sign = +1``) is the defect row (ruling R13). The boundary half of the split
    of the ψ½ split; a ``FaceField[tuple[int, int]]`` keyed
    by ``(level, sign)`` on the
    :class:`~orpheus.numerics.spaces.radial_characteristic_space.RadialCharacteristicBoundarySpace`
    (``G = V(r = R)`` corner gauge). A :class:`FaceField` **sibling** of the
    interior locus :class:`RadialCharacteristicInteriorField`, NOT a child.
    ``mesh`` is :class:`SNMesh` and the space source is the R12a-keyed
    ``mesh.radial_characteristic_boundary_space``. The concrete role leaves are
    ``RadialCharacteristicBoundaryFlux`` (state — same-class signed differences
    carry the iterate increment since campaign 1 CS3, 2026-08-19, when the
    displacement sibling retired) and
    ``RadialCharacteristicBoundarySourceSink`` (emission). Abstract
    — instantiate a role leaf.
    """

    space: RadialCharacteristicBoundarySpace

    def __post_init__(self) -> None:
        if not isinstance(self.space, RadialCharacteristicBoundarySpace):
            raise TypeError(
                f"{type(self).__name__} requires a "
                f"RadialCharacteristicBoundarySpace (read off "
                f"mesh.radial_characteristic_boundary_space); got "
                f"space={self.space!r}."
            )
        super().__post_init__()  # FaceField: Field shape + layout-bearing check.

    @classmethod
    def _face_space_of(
        cls, mesh: "MaterialMesh",
    ) -> RadialCharacteristicBoundarySpace:
        r"""Return ``mesh``'s cached ψ½ boundary space, or raise (R12a)."""
        space = getattr(mesh, "radial_characteristic_boundary_space", None)
        if space is None:
            raise ValueError(
                f"{cls.__name__}: mesh carries no "
                f"radial_characteristic_boundary_space — no μ-level consumes "
                f"independent starting-direction state (R12a). System B's "
                f"boundary block is spelled absent, never a zero-DOF field."
            )
        return space

    @property
    def levels(self) -> tuple[int, ...]:
        r"""The seed-carrying μ-level indices (read off the space)."""
        return self.space.levels

    def corner(self, level: int, sign: int) -> NDArray:
        r"""The ``(ng,)`` r = R corner view for ``(level, sign)`` — memory-shared.

        Inflow corner (``sign = -1``): the given-data / identity row;
        outflow corner (``sign = +1``): the defect row (ruling R13).
        """
        return self.space.slot_view(self.values, level, sign)

