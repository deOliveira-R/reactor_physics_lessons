r"""Augmented geometry for S\ :sub:`N` discrete ordinates transport.

:class:`SNMesh` is axis-primary (C5.1, #225): its canonical spatial
representation is a tuple of :class:`~orpheus.transport.mesh.axis.Axis1D`, and it
precomputes the coordinate-specific streaming stencil used by the
transport sweep. Two construction surfaces funnel into one body — the
axis-native :meth:`SNMesh.from_axes`, and the legacy
:class:`~geometry.mesh.Mesh1D` / :class:`~geometry.mesh.Mesh2D`
constructor (converted to axes once at the boundary).

Three coordinate systems are supported: Cartesian (1D/2D), spherical
(1D), and cylindrical (1D).  Curvilinear geometries precompute angular
redistribution coefficients (:math:`\alpha`), the geometry factor
:math:`\Delta A/w`, and Morel--Montry angular closure weights.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from typing import ClassVar, Iterator, TYPE_CHECKING

import numpy as np

from orpheus.geometry import CoordSystem, Mesh1D, Mesh2D
from orpheus.geometry.boundary import (
    BoundaryTraceLaw,
    ReflectiveBoundary,
    VacuumInflow,
)
from orpheus.geometry.boundary._bound_compat import _BoundBoundaryOperator
from orpheus.transport.spatial.scheme import StreamingTerms
from orpheus.sn.angular.redistribution import angular_redistribution
from orpheus.transport.method import resolve_boundary_conditions
from orpheus.transport.mesh.axis import (
    Axis1D,
    FaceLabel,
    axes_from_legacy_mesh,
    face_labels as _axis_face_labels,
    face_outflow_ordinates as _axis_face_outflow_ordinates,
    face_shape as _axis_face_shape,
    legacy_mesh_from_axes,
    n_unknowns_flat as _axis_n_unknowns_flat,
)
from orpheus.transport.mesh.material_mesh import (
    InconsistentMaterialsError,
    MaterialMesh,
)
from ..boundary.realizer import SNBoundaryRealizer
from .method_space import SNMethodSpace
from .reduced_operator import (
    ReducedStreamingOperator,
    cylindrical_streaming,
    slab_streaming,
    spherical_streaming,
)
from orpheus.numerics.quadrature import Quadrature
from orpheus.transport.spatial.scheme import DiscretizationSchemeBase, CellVisit
from orpheus.transport.spatial.diamond import DiamondDifference
from ..angular.closure import (
    IdentityAngularClosure,
    MorelMontryAngularSweep,
    AngularClosureBase,
    assert_carrying_quadrature,
    default_angular_closure_class,
    march_start_structure_per_level,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from orpheus.data.materials import Materials
    from orpheus.data.macro_xs.mixture import Mixture
    from orpheus.numerics.face_layout import FaceLayout
    from orpheus.numerics.space import FunctionSpace
    from orpheus.numerics.spaces.angular_trace_space import AngularTraceSpace
    from orpheus.numerics.spaces.full_field_space import FullFieldSpace
    from orpheus.numerics.spaces.radial_characteristic_space import (
        RadialCharacteristicBoundarySpace,
        RadialCharacteristicInteriorSpace,
    )
    from orpheus.sn.operators.loss_kernel_gauge import LossKernelGauge
    # NOTE (B.5.A): the mesh provides SPACE data only and does NOT import
    # transport-field types — zero-allocation lives on the field types
    # (``Field.zeros(space)`` / ``TimedFullField.zeros(..., space=)``,
    # reading this carrier's cached space mints).
    # The ``AngularBoundaryFlux`` / ``AngularFlux`` mentions below are docstring
    # cross-references (Sphinx resolves them by full path, no import needed).


# ``InconsistentMaterialsError`` moved to
# :mod:`orpheus.transport.mesh.material_mesh` (it is raised by
# ``MaterialMesh.ng``, the method-agnostic group-consistency check) and is
# re-exported here for the SN-side consumers / tests that import it from
# ``orpheus.sn.mesh.augmented_mesh``.


# ═══════════════════════════════════════════════════════════════════════
# SNMesh
# ═══════════════════════════════════════════════════════════════════════

class SNMesh(MaterialMesh):
    """Augmented geometry for the discrete ordinates method.

    Axis-primary (C5.1, #225): the canonical spatial representation is
    :attr:`axes` — a tuple of :class:`~orpheus.transport.mesh.axis.Axis1D` — from
    which all shape metadata derives. Constructed either axis-natively
    via :meth:`from_axes` or from a legacy
    :class:`~geometry.mesh.Mesh1D` / :class:`~geometry.mesh.Mesh2D`
    (converted to axes once at the inbound boundary; the legacy object
    is retained as :attr:`mesh` for the consumers still reading through
    it). Precomputes the streaming stencil (diamond-difference
    coefficients that depend only on geometry and angular quadrature,
    not on cross sections).

    For Cartesian geometry the stencil stores one per-axis array, read via
    :meth:`streaming`:

    * ``streaming(a)[n, i]`` = :math:`2|\\mu_{a,n}| / \\Delta a_i`
      for every axis ``a < ndim`` (built over ``range(ndim)`` from
      ``quad.axis_cosines(a)`` — no hand-listed x/y pair).

    Parameters
    ----------
    mesh : Mesh1D or Mesh2D
        Base geometry.
    quadrature : Quadrature
        Angular quadrature (Gauss–Legendre, Lebedev, etc.).
    materials : dict mapping material id to Mixture
        Macroscopic cross sections keyed by the integer ids appearing
        in ``mesh.mat_ids`` / ``mesh.mat_map``.  Required (Issue #197
        PR-TYPED-0).  The authoritative source of truth for both
        cross sections and the group count :attr:`ng`; every operator
        that consumes ``sn_mesh`` (L, C, S, F) reads materials from
        here, not from a parallel argument.  All materials must agree
        on ``ng`` — heterogeneous group structures are a
        homogenization-step concern that must precede SNMesh
        construction.

    Attributes
    ----------
    materials : dict mapping material id to Mixture
        The materials dict passed at construction (single source of
        truth).
    ng : int
        Number of energy groups, derived from materials and validated
        for consistency.
    BOUNDARY_OPERATOR_REGISTRY : dict[str, type[BoundaryTraceLaw]]
        Supported boundary-condition kinds (Wave 8 / C8.3) — the SN
        law-admission table read by the shared
        :func:`~orpheus.transport.method.resolve_boundary_conditions`
        body (#290 P7b). Values are :class:`BoundaryTraceLaw`
        subclasses (``VacuumInflow``, ``ReflectiveBoundary``) realized
        per face by :meth:`realize_boundary_law` via
        :class:`SNBoundaryRealizer` for every supported mesh
        (1-D Cartesian, 1-D spherical, 1-D cylindrical, 2-D
        Cartesian) and wrapped in :class:`_BoundBoundaryOperator`
        for compatibility with the SN-side call surface.
    bc : dict[str, _BoundBoundaryOperator]
        Resolved BC operator per boundary face, keyed by the face
        name — the SAME keys as :attr:`boundary_face_layout` /
        ``angular_trace.layout.faces``, both derived from :attr:`face_labels`
        through the single-sourced
        :attr:`~orpheus.transport.mesh.axis.FaceLabel.face_name` crosswalk (C4,
        #220). Each value is a :class:`_BoundBoundaryOperator` shim
        pairing the realized 1-arg :class:`LinearOperator` with the
        **law** it was realized from — so a consumer can ask what the
        face's law DOES (``bc[face].law.geometry_map``,
        ``bc[face].law.response_kernel``) and not only what it was
        declared as. Its ``kind`` tag reads that law's registry key,
        keeping ``sn_mesh.bc["xmin"] == "vacuum"`` style comparisons
        working. The face inventory IS the BC
        inventory: slab ``{"xmin", "xmax"}``; **a solid sphere /
        cylinder has only ONE entry** (``"xmax"``, the outer radius —
        the pole r=0 is the angular closure's regularity condition,
        not a BC face, so it has NO entry rather than a ``None``);
        2-D Cartesian all four faces.
    """

    BOUNDARY_OPERATOR_REGISTRY: ClassVar[dict[str, type[BoundaryTraceLaw]]] = {
        "vacuum": VacuumInflow,
        "reflective": ReflectiveBoundary,
    }
    # Values are the LAW CLASSES themselves (not factory functions), looked up
    # by the shared TransportMethod resolve body (#290 P7b —
    # ``resolve_boundary_conditions`` owns the face loop and the tag → law
    # parse; ``realize_boundary_law`` below dispatches
    # :class:`SNBoundaryRealizer`), applied uniformly for 1-D Cartesian, 1-D
    # spherical, 1-D cylindrical, and 2-D Cartesian meshes.
    #
    # The 4 other kinds ``SNBoundaryRealizer`` dispatches today (``white``,
    # ``periodic``, ``albedo``, ``prescribed_inflow``) are NOT registered here
    # — so they are declarable only by constructing the law directly, never
    # from a ``BC(...)`` tag; admitting them requires SN-sweep-side wiring
    # (sweep cycles for periodic, etc.) and is issue #189.  ``zero_flux`` is
    # the seventh law and is NOT dispatchable at all: the SN realizer REFUSES
    # it (a negative angular inflow is unrepresentable — use ``vacuum``).
    # Future expansion is mechanical: add the law class as a value here,
    # ensure the realizer dispatch handles it, and add an SN-side test that
    # the sweep behaves correctly.
    #
    # There is no ``mixed`` kind: ``MixedBoundaryOperator`` was deleted in
    # Wave 11 and rank-N boundaries are expressed through the descriptor-tree
    # algebra (``LawSum`` / ``LawScaled`` + ``realize_recursively``) instead.

    def __init__(
        self,
        mesh: Mesh1D | Mesh2D,
        quadrature: Quadrature,
        materials: "Materials | Mapping[int, Mixture]",
        scheme: DiscretizationSchemeBase | None = None,
        angular_closure: "type[AngularClosureBase] | None" = None,
    ) -> None:
        # The legacy inbound surface: convert the Mesh1D / Mesh2D declaration
        # to the canonical axis tuple ONCE at the boundary, extract the one
        # payload the axes cannot carry (the material assignment — named
        # ``mat_ids`` on Mesh1D, ``mat_map`` on Mesh2D), and run the same
        # construction body as :meth:`from_axes`. Everything downstream derives
        # from ``self.axes``; ``self.mesh`` survives as inbound provenance for
        # the consumers still reading through it (1-D reduced streaming
        # construction, trace build, realizer metadata, MMS helpers).
        self._init_core(
            axes=axes_from_legacy_mesh(mesh),
            mesh=mesh,
            mat_map=mesh.mat_ids if isinstance(mesh, Mesh1D) else mesh.mat_map,
            quadrature=quadrature,
            materials=materials,
            scheme=scheme,
            angular_closure=angular_closure,
        )

    def _init_core(
        self,
        *,
        axes: tuple[Axis1D, ...],
        mesh: Mesh1D | Mesh2D | None,
        mat_map: np.ndarray | None,
        quadrature: Quadrature,
        materials: "Materials | Mapping[int, Mixture]",
        scheme: DiscretizationSchemeBase | None,
        angular_closure: "type[AngularClosureBase] | None",
    ) -> None:
        # The ONE construction body both surfaces funnel into (C5.1).
        #
        # ── Method-agnostic DATA block → MaterialMesh base ──
        # :meth:`MaterialMesh._init_data` sets ``self.mesh`` / ``self.materials``
        # / ``self.axes`` / ``self.axis_widths`` / ``self.mat_map`` /
        # ``self._volumes`` / ``self._areas`` / ``self.nx`` / ``self.coord`` and
        # runs the materials-consistency validation.  ``materials`` is REQUIRED:
        # SNMesh IS the SN phase space (mesh × quadrature × material group
        # structure); without materials ``.ng`` is undefined (Pattern 4 —
        # illegal states unrepresentable).
        MaterialMesh._init_data(
            self,
            axes=axes,
            mesh=mesh,
            mat_map=mat_map,
            materials=materials,
        )

        # ── SN method layer (BEHAVIOR atop the MaterialMesh data) ──
        self.quad = quadrature
        # Cell-update strategy. Defaults to :class:`DiamondDifference`, which
        # reproduces the inlined sweep math bit-identically (every regression
        # snapshot at ``tests/sn/regression/snapshots/`` was generated with DD
        # and matches bit-for-bit through ``self.scheme.update(...)``).  Pass
        # ``scheme=LinearDiscontinuous()`` etc. to select another.
        self.scheme: DiscretizationSchemeBase = (
            scheme if scheme is not None else DiamondDifference()
        )
        # Angular-redistribution closure.  The default is
        # :class:`MorelMontryAngularSweep` for curvilinear (the canonical
        # per-cell Morel--Montry weighted-DD angular recurrence — BMC 2010
        # Eqs. (42)/(43), NOT Hébert, who ships the plain angular diamond —
        # with the Carlson coupled-pole seed of Hébert §3.9.4) and
        # :class:`IdentityAngularClosure` for Cartesian (flat geometry has
        # no angular-redistribution term at all).  Derivation + the
        # ERR-026 closure: curvilinear_numerics.rst
        # §sn-phase-d-carlson-coupled-pole-sweep.
        #
        # Instantiation is DEFERRED until after the coord dispatch below
        # populates ``self.reduced`` / ``self._volumes`` / ``self.axis_widths``
        # (the data the strategies bind to) — see the ``self.angular_closure
        # = …`` line after the BC resolution.  The override is a CLASS, not an
        # instance: a closure binds to its mesh at construction (``cls(self)``),
        # and this mesh does not exist yet when the caller assembles the
        # constructor arguments (Pattern 4 — an unbound / foreign-bound closure
        # is now unspellable).
        self._user_supplied_closure = angular_closure

        # (``self.axes`` / ``self.axis_widths`` / ``self.mat_map`` /
        # ``self._volumes`` / ``self._areas`` / ``self.nx`` /
        # ``self.coord`` and the materials-consistency validation are all
        # set by the ``MaterialMesh._init_data`` call above — the
        # method-agnostic data block.)

        # Dispatch stencil setup by coordinate system.
        #
        # Curvilinear connection-coefficient math (sphere / cylinder) lives
        # in :mod:`orpheus.sn.mesh.reduced_operator` (Wave B Issue 6 placed
        # it in ``geometry/``; the 2026-08 un-weld arc brought it home)
        # because it is CHART data, not solver data: one object serves the
        # sphere and the cylinder, and Cardinal Rule 2 forbids duplicating
        # it on each solver-side mesh class.  (⛔ The reason recorded here
        # until 2026-08-27 was "so MoC and CP can consume the same
        # primitive".  They cannot and will not — neither forms an angular
        # redistribution term; see structured_geometry.rst "Who needs a
        # connection coefficient — and who does not".  The placement is
        # still right, for the chart-data reason above.)  The
        # Cartesian per-axis streaming stencils are SN-specific (DD
        # denominator precomputation) and stay local to ``_setup_cartesian``.
        #
        # ``self.reduced`` is the canonical accessor every downstream
        # consumer should bind to: ``sn_mesh.reduced.streaming_terms(
        # cell_idx, dir_idx, mu_level_idx)`` returns the per-(cell,
        # direction) packet a sweep cell update needs (the deprecated
        # ``@property`` accessors below still preserve the legacy names).
        # ``self.coord`` was derived from the axes by
        # ``MaterialMesh._init_data`` (the whole-mesh coordinate system;
        # multi-axis tuples are all-Cartesian by construction). The 1-D
        # arms hand the legacy ``Mesh1D`` to the reduced streaming
        # constructors (the genuine remaining Mesh1D consumers; ⛔ "shared
        # with MoC/CP via :mod:`orpheus.geometry.reduced_operator`" until
        # 2026-08-27 — measurably false, see above).
        match self.coord:
            case CoordSystem.CARTESIAN:
                self._setup_cartesian()
                # Presence contract (P4.5): ``reduced`` is populated iff
                # the mesh is 1-D (``is_1d``) — the chain scan is a 1-D
                # construct, and the slab mints the ONE carrier's
                # zero-curvature case exactly like the curvilinear arms
                # (P4.1b: "the slab is not a special case" — real
                # widths/volumes, neutral angular element, zero curvature
                # couplings).  The d≥2 Cartesian wavefront has no chain
                # scan and carries ``None``.
                if self.ndim == 1:
                    assert isinstance(mesh, Mesh1D)
                    self.reduced: ReducedStreamingOperator | None = (
                        slab_streaming(mesh, quadrature)
                    )
                else:
                    self.reduced = None
            case CoordSystem.CYLINDRICAL:
                assert isinstance(mesh, Mesh1D)
                self.reduced = cylindrical_streaming(mesh, quadrature)
                # Q5.6 step 6.3 — cylindrical quadrature ADMISSION: every
                # mu-level must be CARRYING (the R12a march-start
                # predicate), so the psi-half seed is honest independent
                # state marched by route (a).  Deliberately AFTER
                # ``cylindrical_streaming``: its structure-less guard
                # ("level structure") keeps ownership of slab/sphere
                # cubatures with the more specific message; this guard
                # refuses rules that HAVE levels but whose levels cannot
                # carry (node-aligned products, unfolded staggered
                # products, level-symmetric rules).
                assert_carrying_quadrature(quadrature, self.coord)
                # Cartesian-style per-axis streaming arrays not used here
                # (curvilinear streaming lives in reduced.streaming_terms).
                self._streaming_axes = None
            case CoordSystem.SPHERICAL:
                assert isinstance(mesh, Mesh1D)
                self.reduced = spherical_streaming(mesh, quadrature)
                self._streaming_axes = None

        # ── Boundary trace + realized laws ──
        # Build ONE unified trace space per SNMesh, keyed on the mesh's
        # TRUE boundary faces (``boundary_face_layout``): slab
        # ``xmin``/``xmax``, curvilinear ``xmax`` only (the pole at r=0
        # is the angular closure's regularity condition, not a BC
        # face), multi-D Cartesian all ``2·ndim`` faces. Inflow /
        # outflow are selectors over the signed Ω·n it carries.
        # UNCONDITIONAL — every constructible SNMesh builds its trace
        # (geometry-blind: quadrature + face names); built HERE, in the
        # construction body, as phase-space substrate — not inside BC
        # resolution.
        from orpheus.numerics.spaces.angular_trace_space import AngularTraceSpace
        self._trace = AngularTraceSpace.from_quadrature_and_layout(
            self.quad, self.boundary_face_layout,
        )

        # Resolve the per-axis BC declarations through the ONE shared
        # TransportMethod body (#290 P7b): the face loop over
        # ``face_labels``, the ``BC("reflective")`` infinite-lattice /
        # eigenvalue default, and the tag → law parse are method-
        # generic; :meth:`realize_boundary_law` below is the SN arm.
        # The face inventory IS the BC inventory by construction (C4,
        # #220): a face that exists has exactly one entry; the curvilinear
        # pole has none (Pattern 4 — a pole-BC is unrepresentable).
        # Consumers key into :attr:`bc` by the same face names the trace
        # layout carries.
        self.bc: dict[str, _BoundBoundaryOperator] = (
            resolve_boundary_conditions(self)
        )

        # (Materials-consistency validation — every ``mat_map`` id has a
        # ``materials`` entry, and all materials agree on ``ng`` — plus
        # the eager ``ng`` trigger run inside ``MaterialMesh._init_data``
        # above, so a bad materials dict raises at construction time.)

        # ── Pole-angular closure binding (PR-TYPED-6.5 Phase 2.9) ──
        # The closure takes the TWO TENSOR FACTORS of the redistribution
        # operator, not the mesh (the un-weld arc's Phase B): the angular
        # factor (dome, starting direction, measure) and the spatial pairing.
        # The mesh's job here is to hand over two values it already holds —
        # not to be captured.  The user-supplied closure CLASS, or the
        # default-by-coord-system, is constructed through the family's
        # ``cls(angular, pairing, angular_axis)`` contract; every mesh
        # carries a BOUND closure.
        closure_cls = (
            self._user_supplied_closure
            if self._user_supplied_closure is not None
            else default_angular_closure_class(self.coord)
        )
        if self.reduced is not None:
            angular = self.reduced.angular
            pairing = self.reduced.redistribution_pairing
            angular_axis = self.reduced.angular_axis
        else:
            # Multi-D Cartesian: there is NO reduced streaming operator
            # (the chain scan is a 1-D construct; d ≥ 2 rides the
            # wavefront schedule) — and there is no curvature either, so
            # both tensor factors are the NEUTRAL element and neither
            # needs it.  That they are buildable from ``(quad, coord)``
            # alone is the un-weld's own point: the closure's operands
            # were never mesh facts.  The axis mint is the same one the
            # 1-D factories run (``quad.axis()``, label defaulted at the
            # generator so the two arms cannot spell it differently).
            angular = angular_redistribution(self.quad, self.coord)
            pairing = np.zeros((int(np.prod(self.spatial_shape)), 1, 1))
            angular_axis = self.quad.axis()
        self.angular_closure: AngularClosureBase = closure_cls(
            angular, pairing, angular_axis,
        )
        # Drop the temporary attribute now that the closure is bound.
        del self._user_supplied_closure

    # ── Boundary condition resolution ─────────────────────────────────
    #
    # The face loop, the reflective default, and the tag → law parse
    # live in the ONE shared TransportMethod body,
    # :func:`~orpheus.transport.method.resolve_boundary_conditions`
    # (#290 P7b — it replaced the twin ``SNMesh._resolve_bcs`` /
    # ``DiffusionMesh._resolve_bcs`` loops). Only the genuinely
    # SN-specific arm remains here:

    def realize_boundary_law(
        self,
        law: BoundaryTraceLaw,
        face: str,
    ) -> "_BoundBoundaryOperator":
        r"""Realize one typed boundary law on ``face`` — the SN arm of the
        :class:`~orpheus.transport.method.TransportMethod` hook.

        Called per face by the shared
        :func:`~orpheus.transport.method.resolve_boundary_conditions`
        body. Build an :class:`SNMethodSpace` carrying the precomputed
        unified :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`,
        hand the law to :class:`SNBoundaryRealizer.realize`, pair the
        1-arg result back with ``law`` in :class:`_BoundBoundaryOperator`
        so the SN-side call surface sees a uniform 1-arg ``apply(psi)``
        contract that can still be asked what law it realizes.

        **The pairing is the point (campaign phase B2.0).** Until B2.0
        this line kept only ``kind=law.key`` and dropped the law, so
        ``sn_mesh.bc[face]`` was a realized operator plus a *string* —
        which is why five production sites answer structural questions
        (*does my ``G`` permute ordinates? is my ``R`` zero?*) by
        comparing that string against literals: a string was the only
        thing that survived realization. Handing the law through makes
        those questions answerable at the object. The ``kind`` tag lives
        on as a read-through of the law's registry key until B2.2 retires
        the string surface.

        Issue #188 / C188.3: every supported mesh (1-D Cartesian,
        1-D spherical, 1-D cylindrical, 2-D Cartesian) routes
        through the realizer here. The pre-C188.3 curvilinear
        bypass — which wrapped the raw 2-arg
        :class:`BoundaryTraceLaw` with a bound quadrature — is
        gone, made redundant by the unified trace's curvilinear
        support. ``face`` must name a face present in the trace;
        curvilinear's inner pole has no label and is handled by the
        angular closure, not here.
        """
        method_space = SNMethodSpace.for_face(
            mesh=self.mesh,
            quadrature=self.quad,
            face=face,
            trace=self._trace,
        )
        realized = SNBoundaryRealizer().realize(law, method_space)
        return _BoundBoundaryOperator(realized, law)

    # ── Properties ────────────────────────────────────────────────────
    #
    # ``_validate_materials`` and the data properties ``ng`` / ``volumes``
    # / ``volume_measure`` / ``areas`` / ``ndim`` / ``spatial_shape`` —
    # plus the ``material_xs_field()`` builder — are inherited from
    # :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` (the
    # method-agnostic data carrier).  SNMesh adds only the SN-method
    # behavior (quadrature / streaming stencil / boundary trace / closures)
    # on top.

    @property
    def is_1d(self) -> bool:
        """True if this is a genuine 1-D mesh (``ndim == 1``).

        Reads the genuine spatial dimensionality, NOT the phantom
        ``ny == 1`` shim: a :class:`Mesh2D` with a single y-cell is 2-D
        (``ndim == 2``) and is NOT 1-D. The old ``ny == 1`` test
        misclassified that degenerate case and was the root of #214; the
        genuine-dimensionality test is the phantom-axis-elimination
        invariant (R-1 Phase A). This is the single source of truth for
        the 1-D-vs-multi-D dispatch in the streaming operators
        (``not sn_mesh.is_1d`` selects the multi-D Cartesian path).
        """
        return self.ndim == 1

    @property
    def is_cartesian(self) -> bool:
        """True if the mesh carries no curvature (Cartesian slab / 2-D / 3-D).

        The genuine coordinate-system criterion, read off the
        :class:`~orpheus.geometry.coord.CoordSystem` this mesh was posed with.
        This is ORTHOGONAL to :attr:`is_1d`: a slab is Cartesian AND 1-D; a
        2-D Cartesian mesh is Cartesian AND not 1-D; a cylinder is 1-D AND
        not Cartesian.  Sweep-strategy selection
        (``orpheus.sn.loss_representation.default_for``) keys on BOTH axes —
        the anti-hyperplane DAG family requires ``is_cartesian``, the chain
        scan requires ``is_1d`` — so neither alone is a sufficient
        discriminator.

        **Reads the ENUM.**  Until 2026-08-26 this was ``curvature is None``,
        where ``curvature`` was a stringly-typed re-encoding of exactly this
        three-valued :class:`CoordSystem` (``None`` / ``'cylindrical'`` /
        ``'spherical'``) assigned inside the very ``match self.coord`` that
        already knew the answer.  Equivalent on all three arms by
        construction, so the swap is bit-identical — but this property is the
        CONTRACT its consumers speak (including two duck-typed test
        surrogates that stub the property and never saw the field), and the
        field was only ever an implementation of it.  Re-basing here is what
        lets ``curvature`` be retired without touching those consumers.
        """
        return self.coord is CoordSystem.CARTESIAN

    def is_same_phase_space(self, other: "SNMesh") -> bool:
        r"""True iff ``other`` realizes the SAME discrete SN phase space.

        Two :class:`SNMesh` instances pose identical discrete problems when
        they were built from the same CONSTITUENT OBJECTS — the geometry
        mesh and quadrature by ``is`` identity, the materials by
        per-entry MIXTURE identity (same id-set, same ``Mixture`` object
        per id — the tier that guarantees bit-identical re-derivation of
        the data block, per :meth:`from_material_mesh`'s contract; the
        declaration wrapper itself is parsed per carrier since the
        un-weld arc, so wrapper identity is not the constituent) — with
        the same discretization-scheme TYPE.  This is the pairing guard for
        consumers that combine fields from TWO solutions (the P6 #281
        adjoint-weighted collapse, :meth:`SolutionBase.compare
        <orpheus.sn.solution.SolutionBase.compare>`): a forward and an
        adjoint solve share the constituents when the caller passes the
        same ``(materials, mesh, quadrature)`` to both entries, even though
        each entry constructs its own ``SNMesh`` wrapper.

        Deliberately CONSTITUENT-identity, not value-equality: two
        equal-shaped meshes built from different edge arrays are different
        problems as far as this predicate can prove — the L29 discipline
        (never relax an invariant to a weaker downstream check) keeps the
        strong tier; callers wanting cross-problem comparisons must
        construct shared constituents.

        The angular closure is deliberately EXCLUDED: it is a
        solve-time sweep strategy (how ψ is computed near the pole), not a
        phase-space constituent — it changes neither the field layout nor
        the quadrature the pairings contract, so fields from two closures
        remain contractible (do not "strengthen" the guard by adding it).
        The scheme is compared by TYPE (not ``is``) because schemes are
        parameter-free strategy singletons constructed per solve; a future
        LAYOUT-parametrized scheme would need a stronger comparison here.
        """
        return self is other or (
            self.mesh is other.mesh
            and self.quad is other.quad
            # Constituent identity at the MIXTURE tier: the declaration
            # wrapper is parsed per carrier (un-weld arc R20/R21 — a
            # ``Materials`` per mesh), so "same materials dict object"
            # is no longer spellable; what the old ``is`` was a proxy
            # FOR is that the data block re-derives bit-identically,
            # and the mixtures ARE that data. Same id-set + same
            # Mixture object per id keeps every previously-true case
            # true (same dict ⟹ same entries) at the honest tier.
            and self.materials.ids == other.materials.ids
            and all(
                self.materials[i] is other.materials[i]
                for i in self.materials
            )
            and type(self.scheme) is type(other.scheme)
        )

    def streaming(self, axis: int) -> np.ndarray:
        r"""Per-axis RAW down-face streaming ``g = |μ_axis|·face_area_downstream/V = |μ_axis|/Δ_axis``, ``(N, n_axis)``.

        The dimension-generic accessor the anti-hyperplane DAG walk reads as
        ``str_axes[axis]`` — the **scheme-agnostic** geometric streaming.  Each
        spatial scheme applies its OWN closure factor: DD contributes
        :math:`\sum_a 2g_a` to the cell-balance denominator (the
        :math:`2 = 1/w_{\rm DD}` is DD's diamond closure, applied in its kernel,
        NOT here — #240); Linear-Discontinuous reads the same raw ``g``.
        Indexes the per-axis stencil tuple ``_setup_cartesian`` builds over
        ``range(ndim)`` (since C3.6 there is no hand-listed
        ``(streaming_x, streaming_y)`` pair to drift out of axis order — the
        tuple IS positional-by-axis from birth).

        Cartesian-only (the anti-hyperplane lattice is a Cartesian object);
        curvilinear meshes carry their streaming in
        ``reduced.streaming_terms`` (the chain-scan substrate) and are swept by
        the ``CumprodScan`` strategy, not the DAG walk.
        ``axis`` must satisfy ``0 <= axis < ndim``.
        """
        # The None-ness IS the Cartesian-only gate: ``_setup_cartesian``
        # builds the stencil tuple, the curvilinear arms assign ``None`` —
        # so checking the attribute directly both guards and narrows.
        streaming_axes = self._streaming_axes
        if streaming_axes is None:
            raise AttributeError(
                "SNMesh.streaming(axis) is Cartesian-only; curvilinear meshes "
                "carry streaming in reduced.streaming_terms (the chain-scan "
                "substrate), not the anti-hyperplane DAG."
            )
        if not 0 <= axis < self.ndim:
            raise IndexError(
                f"streaming axis {axis} out of range for ndim={self.ndim}"
            )
        return streaming_axes[axis]

    # ── Dim-agnostic geometry primitives (R-1 Phase A C1) ─────────────
    #
    # ``ndim`` / ``spatial_shape`` are inherited from
    # :class:`~orpheus.transport.mesh.material_mesh.MaterialMesh` (the
    # method-agnostic data carrier).

    @property
    def face_labels(self) -> tuple[FaceLabel, ...]:
        r"""Canonical boundary-face inventory.

        Each :class:`~orpheus.transport.mesh.axis.FaceLabel` carries an
        ``axis_index`` and an ``endpoint`` label, derived from the
        per-axis endpoints. Cartesian 1-D returns 2 labels; spherical
        / cylindrical 1-D returns 1 label (the pole is NOT a face —
        see :class:`~orpheus.transport.mesh.axis.Axis1D` docstring); 2-D Cartesian
        returns 4 labels; synthetic 3-D Cartesian would return 6.

        The iteration order is the canonical concatenation order for
        :meth:`AngularFlux.to_flat` (C3) and the canonical iteration
        order for :attr:`AngularBoundaryFlux.face_buffers` (C4).
        """
        return _axis_face_labels(self.axes)

    def face_shape(self, label: FaceLabel) -> tuple[int, ...]:
        r"""Spatial shape of the boundary face identified by ``label``.

        The face lies in the codimension-1 hyperplane spanned by the
        axes other than ``label.axis_index``; its shape is the
        per-axis cell count of those axes in axis-index order.
        """
        return _axis_face_shape(self.axes, label)

    def face_outflow_ordinates(self, label: FaceLabel) -> np.ndarray:
        r"""Ordinate indices whose direction-cosine is OUTWARD at face ``label``.

        At the ``max`` / ``outer`` endpoint of an axis, outflow is
        :math:`\mu_{axis} > +10^{-15}`; at ``min``, outflow is
        :math:`\mu_{axis} < -10^{-15}`. Ordinates exactly tangent to
        the face contribute neither inflow nor outflow.

        This method is the canonical producer for the per-face
        outflow mask used by the pack convention (C3),
        :class:`AngularBoundaryFlux.face_buffers` (C4), and the sweep DAG
        face-trace state (C5).
        """
        return _axis_face_outflow_ordinates(self.axes, label, self.quad)

    @property
    def n_unknowns_flat(self) -> int:
        r"""Total flat-vector size for typed :class:`AngularFlux`.

        The pack convention (C3) is the direct-sum decomposition
        :math:`V = V_\text{cells} \oplus \bigoplus_\ell V_{\text{face}, \ell}`;
        ``n_unknowns_flat`` is the dimension of that vector space.
        Cells contribute :math:`N \cdot n_g \cdot \prod_i n_i`; each
        face :math:`\ell` contributes
        :math:`|\text{outflow}_\ell| \cdot n_g \cdot \prod_{i \ne \text{axis}(\ell)} n_i`.
        """
        return _axis_n_unknowns_flat(self.axes, self.quad, self.ng)

    @classmethod
    def from_axes(
        cls,
        axes: tuple[Axis1D, ...],
        quadrature: "Quadrature",
        materials: "Materials | Mapping[int, Mixture]",
        *,
        mat_map: np.ndarray | None = None,
        scheme: DiscretizationSchemeBase | None = None,
        angular_closure: "type[AngularClosureBase] | None" = None,
    ) -> "SNMesh":
        r"""Build an :class:`SNMesh` from an axis tuple — the axis-native surface.

        C5.1 (axis-primary inversion, #225): the caller's axes ARE the
        mesh's axes — stored verbatim and never round-tripped through a
        legacy mesh and re-derived (the pre-C5.1 round-trip silently
        reset custom endpoint labels to ``min``/``max``/``outer``). A
        legacy :class:`Mesh1D` / :class:`Mesh2D` ADAPTER is still
        synthesized at d≤2 for the consumers that read through
        ``self.mesh`` (1-D reduced streaming construction, trace build,
        realizer metadata) — each dissolves across C5.2–C5.5.

        Endpoint labels must be canonical (``min``/``max``/``outer``):
        the :attr:`bc` dict is keyed by
        :attr:`~orpheus.transport.mesh.axis.FaceLabel.face_name`, which fails loud
        on a custom label (C4 doctrine — overridable labels cannot
        silently desync the face-name crosswalk). Custom labels are for
        standalone axis use, not SNMesh construction.

        Parameters
        ----------
        axes : tuple of :class:`~orpheus.transport.mesh.axis.Axis1D`
            Per-axis 1-D mesh descriptors. Length 1 → 1-D mesh;
            length 2 → 2-D Cartesian mesh; length ≥3 → d-D Cartesian
            (C5.5, #225 — all-Cartesian required, mesh-adapter-free
            from birth, swept by the d-generic ``FullFieldWavefront``
            spine).
        quadrature : :class:`Quadrature`
            Angular quadrature.
        materials : Materials or mapping of material id to Mixture
            Materials dict keyed by material id; same contract as the
            legacy constructor.
        mat_map : ndarray or None
            Material-id assignment. Shape ``spatial_shape``. Defaults
            to all-zeros (single material with id 0).
        scheme : DiscretizationSchemeBase or None
            Cell-update strategy. Defaults to :class:`DiamondDifference`.
        angular_closure : type[AngularClosureBase] or None
            Override the default angular closure CLASS
            (curvilinear → :class:`MorelMontryAngularSweep`,
            Cartesian → :class:`IdentityAngularClosure`).  A class, not
            an instance: closures bind to their mesh at construction
            (``cls(sn_mesh)``), and the mesh does not exist yet.
        """
        axes = tuple(axes)
        # C5.5 (#225): d≥3 is mesh-adapter-free from birth — every
        # consumer that read through ``self.mesh`` was dissolved across
        # C5.2–C5.4 (volume measure, trace, windowing gates) or is
        # d≤2-only (the 1-D reduced streaming constructors, the MMS
        # helpers). d≤2 still synthesizes the legacy adapter for those
        # remaining consumers.
        mesh = (
            legacy_mesh_from_axes(axes, mat_map=mat_map)
            if len(axes) <= 2 else None
        )
        obj = cls.__new__(cls)
        obj._init_core(
            axes=axes,
            mesh=mesh,
            mat_map=mat_map,
            quadrature=quadrature,
            materials=materials,
            scheme=scheme,
            angular_closure=angular_closure,
        )
        return obj

    @classmethod
    def from_material_mesh(
        cls,
        material_mesh: MaterialMesh,
        quadrature: "Quadrature",
        *,
        scheme: DiscretizationSchemeBase | None = None,
        angular_closure: "type[AngularClosureBase] | None" = None,
    ) -> "SNMesh":
        r"""Promote a :class:`MaterialMesh` to a solvable SN phase space.

        The data/behavior join: a :class:`MaterialMesh` carries the
        method-agnostic data (axes + materials + mat_map + volumes); this
        classmethod adds the SN method layer (angular quadrature + sweep
        stencil + boundary trace + closures) to make it solvable.

        It is the natural consumer of cross-section homogenization: a
        homogenized :class:`MaterialMesh` (from
        :meth:`~orpheus.sn.solution.Solution.homogenize`) is promoted
        here to re-solve the coarsened problem on the same outer geometry
        (the "re-solve the homogenized problem" path).  The
        material-mesh's axes / mesh / mat_map / materials are passed
        through verbatim — ``_init_core`` re-derives the data block
        bit-identically from them.

        Parameters
        ----------
        material_mesh : MaterialMesh
            The mesh+materials carrier to promote.
        quadrature : Quadrature
            Angular quadrature for the SN method.
        scheme : DiscretizationSchemeBase or None
            Cell-update strategy.  Defaults to :class:`DiamondDifference`.
        angular_closure : type[AngularClosureBase] or None
            Override the default angular closure CLASS
            (curvilinear → :class:`MorelMontryAngularSweep`,
            Cartesian → :class:`IdentityAngularClosure`).  A class, not
            an instance: closures bind to their mesh at construction
            (``cls(sn_mesh)``), and the mesh does not exist yet.

        Every carrier in the hierarchy promotes: ``mesh is None`` has ONE
        meaning (the d≥3 axis-native carrier, which promotes normally —
        ``tests/transport/test_material_mesh_admission.py`` pins that every
        d≤2 constructor carries a mesh). The mesh-less infinite-medium
        1-cell carrier this method once refused with a typed
        ``ValueError`` (S7 G7.1) retired at the CS4c coda, 2026-09-08 —
        the infinite-medium problem builds no carrier, so the refusal
        had no reachable input left.
        """
        obj = cls.__new__(cls)
        obj._init_core(
            axes=material_mesh.axes,
            mesh=material_mesh.mesh,
            mat_map=material_mesh.mat_map,
            quadrature=quadrature,
            materials=material_mesh.materials,
            scheme=scheme,
            angular_closure=angular_closure,
        )
        return obj

    @property
    def angular_trace(self) -> "AngularTraceSpace":
        r"""The unified boundary :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`.

        One concrete trace space for the whole boundary :math:`\Gamma`,
        built (A.2/A.3) from this mesh's quadrature +
        :attr:`boundary_face_layout` (geometry-blind since C5.3, #225).
        It is the single source of truth for the signed projection
        :math:`\Omega\cdot\hat n_f` per face; the inflow / outflow
        *selectors*
        (:meth:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace.outflow_indices_for_face`)
        replace the inline ``sign(Ω·n)`` masks that the streaming matvec
        and the boundary realizer previously each recomputed.

        ALWAYS non-``None`` (C5.3): the only mesh the pre-C5.3 gate
        excluded — a cylindrical :class:`~orpheus.geometry.mesh.Mesh2D`
        — cannot become an SNMesh at all, so every constructible SNMesh
        carries a trace.
        """
        return self._trace

    @cached_property
    def radial_characteristic_levels(self) -> tuple[int, ...]:
        r"""μ-level indices that consume INDEPENDENT starting-direction state (R12a).

        The seed-presence predicate of #282 route (a), posed on the two
        structural facts of the level's march-start edge
        (:class:`~orpheus.sn.angular.closure.MarchStart`,
        Q5.4/T26): a level carries a ψ½ block iff the M-M half-angle
        recurrence genuinely consumes a seed value — i.e. the start
        edge is NOT itself an ordinate (``on_edge_node``: an η-minimum
        node on :math:`\Sigma`, cylinder product NODE_ALIGNED rules —
        the #229 fact) AND the start is NOT η-degenerate
        (``degenerate``: a double-cover tie killing the recurrence's
        :math:`(1-\tau_0)` thread weight, cylinder level-symmetric
        rules — measured 0.0-bit solve insensitivity). Sphere-GL is
        the carrying instance (one level, the whole quadrature);
        Cartesian never carries. A σ_y-FOLDED product rule carries on
        every level — the arc's start is genuinely off-node (T22b).
        Since Q5.6.3 the cylindrical ADMISSION
        (:func:`~orpheus.sn.angular.closure.assert_carrying_quadrature`
        in ``_init_core``) refuses any cylinder rule with a
        non-carrying level, so on a constructed cylindrical SNMesh
        this property is always the FULL level range; the non-carrying
        cylinder families above are the refusal battery's negatives,
        never live meshes.

        R12a refines the R12 letter ("μ_start ∉ the level's μ-nodes"),
        which conflated the two facts: the letter fires on
        level-symmetric cylinder rules where the seed is nonetheless
        dead. Until Q5.4 the predicate read the raw M-M float
        (``τ_raw,0 ∈ (0,1)`` exclusive, plus an FP-noise guard); the
        first-ordinate trichotomy is now a bit-exact gated CONSEQUENCE
        of the two facts, not the predicate. Level indexing matches
        the closure's: the sphere's single M-M level is index ``0``;
        cylinder levels index ``quad.level_indices``.
        """
        if self.is_cartesian:            # "Cartesian never carries", above
            return ()
        assert self.reduced is not None  # curvilinear ⇒ reduced populated
        starts = march_start_structure_per_level(self.quad, self.coord)
        return tuple(
            p for p, start in enumerate(starts)
            if start.consumes_independent_seed
        )

    def _radial_characteristic_for_levels_args(
        self,
    ) -> "tuple[tuple[int, ...], int, int, np.ndarray] | None":
        r"""The shared ``for_levels`` args for the ψ½ ray spaces, or ``None``.

        Single-sources the R12a levels gate + the ``(ng, nx, cell_volumes)``
        sourcing across the split :attr:`radial_characteristic_interior_space`
        / :attr:`radial_characteristic_boundary_space` (Phase B — the
        coupled-block campaign poses the ψ½ ray as System B, its own
        interior ⊕ boundary composite; the unified space retired at 4e), so
        the ψ½ spaces are built from ONE set of inputs. ``None`` ⟺
        :attr:`radial_characteristic_levels` is empty (absence is spelled
        ``None``, never a zero-DOF space). ``cell_volumes`` is the
        ``G_sd = V_cell`` state metric — the SAME radial cell-volume measure the
        bulk metric ``G_bulk = V_cell·w_n`` reads (:attr:`full_field_space`).
        """
        levels = self.radial_characteristic_levels
        if not levels:
            return None
        return (
            levels,
            self.ng,
            int(self.spatial_shape[0]),
            np.asarray(self.volumes, dtype=float).ravel(),
        )

    @cached_property
    def radial_characteristic_interior_space(
        self,
    ) -> "RadialCharacteristicInteriorSpace | None":
        r"""The ψ½ INTERIOR (cells) space — System B's interior block, or ``None``.

        The ``(ng, nx)`` cells legs under the SPD ``G_sd = V_cell`` metric, on
        which
        :class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator`
        (A_BB) marches — the seed sibling of :attr:`angular_trace`, paired
        with :attr:`radial_characteristic_boundary_space` (Phase B; the
        historical unified space retired at 4e). ``None`` on non-carrying
        meshes (R12a). Cached (one per mesh), built from the shared
        :meth:`_radial_characteristic_for_levels_args`.
        """
        args = self._radial_characteristic_for_levels_args()
        if args is None:
            return None
        from orpheus.numerics.spaces.radial_characteristic_space import (
            RadialCharacteristicInteriorSpace,
        )

        levels, ng, nx, cell_volumes = args
        return RadialCharacteristicInteriorSpace.for_levels(
            levels, ng=ng, nx=nx, cell_volumes=cell_volumes,
        )

    @cached_property
    def radial_characteristic_boundary_space(
        self,
    ) -> "RadialCharacteristicBoundarySpace | None":
        r"""The ψ½ BOUNDARY (corner) space — System B's boundary block, or ``None``.

        The corner sibling of :attr:`radial_characteristic_interior_space`
        (Phase B): the
        ``(ng,)`` r = R corner legs under the ``G = V(r = R)`` corner gauge, on
        which
        :class:`~orpheus.sn.operators.boundary.RadialCharacteristicBoundaryOperator`
        (B_b) acts (inflow = given data; outflow = the defect row). ``None`` on
        the same non-carrying meshes (R12a). Cached (one per mesh), built from
        the shared :meth:`_radial_characteristic_for_levels_args`.
        """
        args = self._radial_characteristic_for_levels_args()
        if args is None:
            return None
        from orpheus.numerics.spaces.radial_characteristic_space import (
            RadialCharacteristicBoundarySpace,
        )

        levels, ng, nx, cell_volumes = args
        return RadialCharacteristicBoundarySpace.for_levels(
            levels, ng=ng, nx=nx, cell_volumes=cell_volumes,
        )

    @cached_property
    def reflective_axes(self) -> tuple[int, ...]:
        r"""WHICH axes are reflective at BOTH endpoints — the closable loops.

        The geometry half of the gauge-freedom predicate; the closure
        half is
        :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.face_transmission_spectrum`.
        An undamped face mode only closes into a null vector if it can
        return to itself, which needs a **closed** reflective loop —
        one axis reflective at both ends gives a there-and-back path,
        and `[M]` (#344) two such axes are what the loss operator's
        kernel actually requires: at ``d = 2`` a single vacuum face
        collapses ``dim ker A`` from 12 to 0.

        Reports closed PAIRS, not faces, and that distinction is
        load-bearing: a *mixed* axis (one face reflective, one vacuum)
        contributes nothing, because the outbound leg escapes.  Derived
        from :attr:`bc` through :attr:`FaceLabel.face_name` — the same
        single-sourced crosswalk the trace layout and the sweep
        schedule key on — so a face inventory that grows a dimension is
        handled correctly with no edit here.

        Returns the axis INDICES, ascending, because that is what the
        consumers need: :attr:`reflective_axis_pairs` wants only how
        many, but
        :func:`~orpheus.sn.operators.loss_kernel_gauge._reflection_orbits`
        must know *which* axes generate the mirror group.  Both read
        this one body — until 2026-08-15 the gauge carried a
        line-for-line twin of it, and `[M]` widening either copy alone
        was **inert** (0 of 25 gates red) because the survivor guarded
        the gate.

        ⚠ Reads the *realized law*, not the tag a caller passed:
        ``resolve_boundary_conditions`` fills unset faces with
        ``BC("reflective")``, so a bare ``SNMesh(mesh, quad, mats)`` is
        all-reflective and this returns every axis.
        """
        from orpheus.geometry.boundary.reflective import ReflectiveBoundary

        by_axis: dict[int, list[bool]] = {}
        for label in self.face_labels:
            bound = self.bc.get(label.face_name)
            by_axis.setdefault(label.axis_index, []).append(
                bound is not None
                and isinstance(bound.law, ReflectiveBoundary)
            )
        return tuple(
            axis for axis, faces in sorted(by_axis.items())
            if len(faces) == 2 and all(faces)
        )

    @cached_property
    def reflective_axis_pairs(self) -> int:
        r"""How many axes are reflective at BOTH endpoints.

        The count of :attr:`reflective_axes`, which owns the criterion
        and the reasoning behind it.  Kept as its own name because the
        gauge-freedom predicate asks a *cardinality* question
        (``pairs >= 2``) and reads better spelled that way.
        """
        return len(self.reflective_axes)

    @cached_property
    def loss_kernel_gauge(self) -> "LossKernelGauge":
        r"""The :math:`G`-orthogonal projector onto :math:`\ker(L + C - S - B)`.

        Cached HERE, on the mesh, because the kernel is a **Stratum-1**
        (geometry-only) object: it is determined by the boundary laws, the
        quadrature and the cell edges, and never reads a cross-section — ``[M]``
        (#344) an absorber and a fissile mixture on the same box give
        bit-identical residuals (:math:`2.799\times10^{-16}`).  So one build
        serves every group, every outer and every eigenvalue iterate, which is
        what amortises the setup to nothing (``[M]`` 4.0 ms at d=2 ``(3,4)``,
        22.3 ms at d=3 ``(3,4,5)``, single-process).

        The mesh is the right owner rather than
        :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`
        (which is geometry-blind since C5.3 and cannot see the boundary laws)
        or :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
        (whose ``__eq__`` is ``(name, shape)``, so two meshes with different
        BCs and the same DOF count compare equal — a cache keyed there would be
        keyed on a size).  It sits beside
        :attr:`reflective_axis_pairs`, which is the geometry half of its own
        applicability predicate.

        ⚠ **Zero blocks when there is nothing to fix** — a non-singular
        configuration yields the zero projector, so :meth:`LossKernelGauge.gauge`
        is the identity and no consumer needs a ``None`` branch.  Ask
        :func:`~orpheus.sn.operators.loss_kernel_gauge.gauge_freedom` for the
        reason (and for the warning owed when a closure is UNDETERMINED).
        """
        from orpheus.sn.operators.loss_kernel_gauge import LossKernelGauge

        return LossKernelGauge.for_mesh(self)

    @cached_property
    def radial_characteristic_field_space(self) -> "FullFieldSpace | None":
        r"""System B's member space — the ψ½ ``interior ⊕ boundary`` composite, or ``None``.

        The :class:`~orpheus.numerics.space.FunctionSpace` the re-typed
        coupling blocks declare (B.2b DP1): ``A_BA``'s codomain and ``B_b``'s
        domain/codomain — the carrier space of
        :class:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField`.
        REUSES the family-blind
        :class:`~orpheus.numerics.spaces.full_field_space.FullFieldSpace`
        (the same direct-sum member-wise metric dispatch System A's
        :attr:`full_field_space` uses — zero new space classes; this IS the
        post-eviction end-state, one composite-space class with instances
        differing in members), instantiated over the two split ψ½ spaces:

        * **interior** — :attr:`radial_characteristic_interior_space`
          (``G_sd = V_cell`` cells state metric);
        * **trace slot** — :attr:`radial_characteristic_boundary_space`
          (the ``G = V(r = R)`` corner gauge).

        Identity is the family rule (CS4b S4): ``full_field#<digest>``
        with the digest folded from the two ψ½ member spaces' content —
        the role tag the pre-S4 mint carried (``"radial_characteristic"``)
        was role-flavoured space naming (G2.3: role is CLASS identity —
        the role lives on ``RadialCharacteristicField``, the field class).
        ``None`` on non-carrying meshes (R12a; System B does not exist
        there). Cached: one space per mesh, so every block shares one
        identity instance.
        """
        interior = self.radial_characteristic_interior_space
        boundary = self.radial_characteristic_boundary_space
        if interior is None or boundary is None:
            return None
        from orpheus.numerics.spaces.full_field_space import FullFieldSpace

        return FullFieldSpace.from_blocks(interior, boundary)

    @cached_property
    def angular_bulk_space(self) -> "FunctionSpace":
        r"""The angular-bulk function space of this carrier — axis-built (CS4b).

        The per-ordinate bulk phase space ``(N, ng, *spatial)`` as the ordered
        axis product

        .. math::

            V_{\rm ang} \;=\; V_\Omega \otimes V_E \otimes V_r
            \qquad
            \bigl(\text{angular } w_n\bigr) \otimes
            \bigl(\text{energy}\bigr) \otimes
            \bigl(\text{spatial } V_{\rm cell}\bigr),

        i.e. literally ``of_axes(angular, *bulk_space.axes)`` — the angular
        factor prepended to the SCALAR bulk. Three conventions live here
        (CS4b crosswalk B1, ``.claude/plans/cs4b_crosswalk.md``):

        * **Axis order is** ``(angular, energy, spatial)``, matching the bulk
          tensor layout ``(N, ng, *spatial)`` and
          :attr:`full_field_space`'s dense interior. The scalar bulk is this
          product minus axis 0, so the angular retract is "drop axis 0" and
          the two carriers cannot disagree on the shared factors.
        * **The energy and spatial arms are** :attr:`bulk_space`'s **axes,
          reused verbatim** — the reachable-materials energy-arm rule and
          the cell-volume spatial measure are spelled ONCE, there (Pattern
          2). This property adds exactly one fact: the quadrature measure
          ``w_n`` on the ordinate axis (NODAL — ordinates are point
          samples; per-component positivity is meaningful, so
          ``has_coordinate_cone`` reads ``True``).
        * **The derived space name is not an API** — it is a content digest
          that changes when CS2 mints typed axis subclasses. Consumers pin
          per-axis ``label/shape/kind/weights`` or relative ``is``/``==``,
          never the name literal.

        This is the WIDTH-1 base: the within-cell spatial-moment tail
        (LD) is the SIBLING mint :attr:`angular_trial_space`, which
        appends the scheme's
        :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`
        (construct-general / select-narrow, #240 D5b-S3-A0) — a call
        site widens by SELECTING that property; this one never reads
        the scheme, and the slopeless closures collapse the two mints
        to one instance.

        Carries the physical Hilbert metric per axis (``w_n``, ``V_cell``)
        — and :attr:`full_field_space`'s interior IS this mint (or the
        widened trial mint, ``is``-shared since CS4b S5), so ``G_bulk =
        V·w_n`` has one spelling in production; the ≤ 1 ULP gate that
        reproduces it is against a HAND-densified oracle
        (``tests/sn/mesh/test_angular_bulk_space.py``), not a second
        production copy. Cached: every
        consumer of one carrier reads the SAME instance; equal carriers
        mint ``==`` spaces through the derived name.
        """
        from orpheus.numerics.space import FunctionSpace

        scalar = self.bulk_space
        assert scalar.axes is not None  # of_axes-built by construction
        # The generator mints its own axis (CS5): identical structural
        # content to the literal ``Axis("angular", (quad.N,), weights=
        # quad.weights, kind=NODAL)`` this replaced — the space name and
        # identity are unchanged — plus provenance: the axis carries the
        # quadrature, so a consumer holding the SPACE can recover the
        # forgotten angular geometry (``mu_x``/``eta``/``mu_z``/
        # ``level_indices``) without being handed the quadrature
        # separately.
        angular = self.quad.axis()
        return FunctionSpace.of_axes(angular, *scalar.axes)

    @cached_property
    def angular_trial_space(self) -> "FunctionSpace":
        r"""The angular TRIAL space — :attr:`angular_bulk_space` in the
        scheme's within-cell basis (CS4b S5).

        The space the discrete angular solution actually lives in: the
        width-1 bulk product extended by the bound scheme's within-cell
        spatial-moment factor,

        .. math::

            V_{\rm trial} \;=\; V_\Omega \otimes V_E \otimes V_r
            \;\otimes\; V_{\rm moment}({\rm scheme}),

        where the trailing factor is the scheme-owned MODAL
        :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`
        (the tensor-Legendre cell basis). For the slopeless closures
        (Diamond Difference / Step —
        :attr:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.is_multi_moment`
        ``False``) the trial basis IS the cell average and this property
        returns :attr:`angular_bulk_space` **itself**: the same cached
        instance, byte-identical, so slopeless consumers pay nothing and
        the two mints cannot drift.

        **The metric fact that makes this a correctness mint, not sugar
        (#310 C2 ruling 3).** The moment axis carries the scheme's mass
        ``M_ii/V = ∏_a θ^{o_a}`` as its measure (basis ↔ mass
        single-sourced at the scheme), so the trial Gram is
        ``G = V·w_n ⊗ diag(1, θ, …)``. An average-only ``V·w_n``
        broadcast over the moment axis would mis-weight the slope DOF:
        ``.H`` becomes a WRONG adjoint on the slope rows AND reciprocity
        goes Mode-12-blind to a slope-row transpose.

        **The construct-general / select-narrow seam (#240 D5b-S3-A0),
        post S5.** Which of the two mints a call site reads IS the
        widening decision: the seams that FILL the moment axis (the SI
        cold starts, the LD emissions, ``coupled_system``) allocate on
        THIS space; width-1 cell-average consumers read
        :attr:`angular_bulk_space`. This property replaces the retired
        ``spatial_moments=`` factory parameter (the int was a lossy
        proxy for the scheme's basis — crosswalk B5); the composition
        rule "widen ⟺ append the scheme's moment axis" is spelled HERE,
        once, not at the ~19 call sites that used to thread the int.

        :attr:`full_field_space` builds its interior on this mint, so
        the composite's interior, this property, and every trial-space
        field allocation share ONE cached instance at every scheme
        width.
        """
        from orpheus.numerics.space import FunctionSpace

        if not self.scheme.is_multi_moment:
            return self.angular_bulk_space
        base = self.angular_bulk_space
        assert base.axes is not None  # of_axes-built by construction
        return FunctionSpace.of_axes(
            *base.axes, self.scheme.moment_axis(self.axes),
        )

    def moment_space(
        self, L: int, *, spatial_moments: int = 1,
    ) -> "FunctionSpace":
        r"""The harmonic-moment space this carrier induces at truncation
        order ``L`` — ONE object per ``(L, spatial_moments)`` (CS4c step 6
        item 6.2b, 2026-09-07).

        The carrier owns its spaces the way it owns
        :attr:`angular_bulk_space` and :attr:`angular_trial_space`: the
        quadrature frame at ``L`` is read for the angular HEAD
        (``quad.angular_frame(L).basis_space`` — the frame's
        Parseval-dressed coefficient space, the ONE moment space the tree
        binds since CS4c step 6 item 6.2c-ii, ruling R-6.2c-1 2026-09-08:
        *the carrier's norm is the field's energy* — ``‖Mψ‖² = ‖ψ‖²_W``
        holds on 33 of 33 shipped (rule, L) rows under it and on 0 under
        the basis's continuum Gram that #429 Landing A had bound here;
        never minted from ``L``. The frame's
        :meth:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame.moment_space_on`
        derives a structurally EQUAL object from the angular space — two
        owners, one space, ruling O-5), the cell group IS
        :attr:`bulk_space` (the same instance the scalar family holds),
        and the within-cell spatial-moment tail is composed onto the cell
        group iff ``spatial_moments > 1`` THROUGH the fields' own composer
        (:meth:`~orpheus.transport.fields._bases.BulkField.compose_spatial_moments`,
        CS4c step 6 item 6.2c-iii): the scheme's mass-weighted MODAL
        :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`,
        the SAME factor the widened angular space carries — so the widened
        moment product is axis-built like the un-widened one and its tail
        has one spelling (until 6.2c-iii a separate Euclidean
        ``SpatialMomentSpace`` class was appended here instead).

        **Why the hub, and why a keyed cache.** Every moment field on this
        carrier (:meth:`HarmonicMomentFlux.from_mesh_and_L
        <orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux.from_mesh_and_L>`
        / ``zeros_for_mesh_and_L``), every admission guard's reference
        (``space_on``) and the sweep's iterate wrap read THIS method, so
        they hold the same object — identity is ``is``, not a content
        comparison — and nothing is re-minted per call. `[M]` until 6.2b
        the moment family minted its own space on every read: 113 of the
        118 ``*`` products per 2-D windowed SI solve (58 from the boundary
        leaf's carrier guard, 55 from the sweep's iterate wrap; the count
        grew as ``2·max_inner + 6``). With the hub owning it the count is
        INVARIANT in the iteration budget — gated in
        ``tests/sn/mesh/test_hub_owns_the_moment_space.py``. The cache is
        keyed, not a bare property, because ``L`` and the width are the
        posing's truncation orders, chosen per binding.

        ``spatial_moments`` is the CALLER's selection (construct-general /
        select-narrow, #240 D5b-S3-A0): the seams that FILL the moment
        axis pass the scheme's ``spatial_basis_per_axis``; width-1
        consumers pass nothing.
        """
        key = (int(L), int(spatial_moments))
        cached = self._moment_spaces.get(key)
        if cached is not None:
            return cached
        from orpheus.transport.fields._bases import BulkField

        head = self.quad.angular_frame(L).basis_space
        bulk = BulkField.compose_spatial_moments(
            self.bulk_space, self, spatial_moments,
        )
        space = head * bulk
        self._moment_spaces[key] = space
        return space

    @cached_property
    def _moment_spaces(self) -> "dict[tuple[int, int], FunctionSpace]":
        """The per-``(L, spatial_moments)`` cache behind :meth:`moment_space`."""
        return {}

    @cached_property
    def full_field_space(self) -> "FullFieldSpace":
        r"""The composite carrier :math:`V_{\rm bulk} \oplus V_{\rm trace}` (Wave O / O.2b).

        The function space of the FULL streaming operator
        (:class:`~orpheus.sn.operators.streaming.StreamingOperator`) and every bulk
        :math:`\oplus` boundary composite — the domain/codomain under which
        ``L.H`` and ``(L + C - B).H`` are the **metric-correct G-adjoint**
        :math:`A^\dagger = G^{-1} A^{\mathsf T} G` (Issue #208). The
        block-diagonal Hilbert metric :math:`G` is

        * **bulk** :math:`G_{\rm bulk} = V_{\rm cell}\,w_n` — the full
          phase-space measure :math:`\mathrm{d}V\,\mathrm{d}\Omega`,
          carried PER AXIS by :attr:`angular_trial_space` (the interior
          IS that cached mint since CS4b S5, and at slopeless widths that
          mint IS :attr:`angular_bulk_space` — ``w_n`` on the ordinate
          axis, ``V`` on the spatial axis, the scheme's moment mass on
          the LD tail; pre-S2b this property densified the same Gram to
          a broadcast ``(N, 1, nx, ny)`` array, reproduced by the axis
          form at ≤ 1 ULP on every face);
        * **trace** :math:`G_{\rm trace} = |\Omega\cdot\hat n_f|\,w_n` — the
          partial-current surface metric already carried by
          :attr:`angular_trace`.

        The two factors carry :math:`w_n`; they differ only in the
        spatial measure (cell volume vs. oriented face). On a carrying
        mesh the ψ½ ray's state metric (``G_sd = V_cell``) lives on
        **System B's own composite space**,
        :attr:`radial_characteristic_field_space` — never as a third
        block here (the B.2d eviction; the coupled DOF count is the
        honest two-system sum). Cached: the composite is immutable for a
        given mesh + quadrature.
        """
        from orpheus.numerics.spaces.full_field_space import FullFieldSpace

        # The interior IS the carrier's trial space (CS4b S5 — one mint,
        # ``is``-shared with :attr:`angular_trial_space` at every scheme
        # width: DD/Step read the width-1 cached angular bulk itself; a
        # multi-moment closure (LD) appends the scheme's moment axis
        # carrying the moment mass, so ``.H`` stays the metric-correct
        # adjoint on the slope rows — the mass fact and its #310 C2
        # rationale live on the trial mint's docstring).  The Gram
        # ``G_bulk = V_cell·w_n [⊗ mass]`` lives per axis on that space,
        # reproducing the retired dense spelling at ≤ 1 ULP on every
        # metric face — gated by the hand-built dense oracle in
        # ``tests/sn/mesh/test_angular_bulk_space.py``.
        return FullFieldSpace.from_blocks(
            self.angular_trial_space,
            self.angular_trace,
        )

    @property
    def boundary_face_layout(self) -> "FaceLayout[str]":
        r"""Flat :class:`~orpheus.numerics.face_layout.FaceLayout` of boundary faces.

        Depth B step D-G primitive. Returns the per-geometry boundary
        face descriptor: which faces exist, their per-face shapes, and
        the flat-buffer offsets that pack them. The post-D-G pure-Field
        :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
        consumes this layout to lay out its flat backing buffer.

        Derived from :attr:`face_labels` (C4): one slot per label, named
        by the single-sourced :attr:`FaceLabel.face_name` crosswalk,
        shaped ``(N, ng, *face_shape(label))`` — axis-count generic, no
        per-geometry hand-list. The geometry mapping falls out:

        * 1-D slab — two faces ``xmin`` / ``xmax``, each ``(N, ng)``.
        * 1-D curvilinear sphere / cylinder — one face ``xmax``, shape
          ``(N, ng)`` (the single ``"outer"`` endpoint renders as
          ``xmax``; the geometric pole at r=0 is a regularity
          condition, not a BC face, so it has no label and no slot).
        * 2-D Cartesian — four faces: ``xmin`` / ``xmax`` shape
          ``(N, ng, ny)``; ``ymin`` / ``ymax`` shape ``(N, ng, nx)``.
        * A 3-axis mesh (C5) would yield six slots ``xmin`` … ``zmax``
          with codimension-1 shapes — no edit needed here.

        Spatial-moment tail (#251 — Leg B of #247)
        ------------------------------------------
        A multi-moment closure (LD's bilinear UBLD face) carries a
        trailing ``2^{d-1}``-transverse-moment axis per face slot, so a
        moment-resolved prescribed inflow can carry the along-face
        (transverse) Legendre slope and the sweep outflow can STORE
        those ``2^{d-1}`` moments instead of collapsing to the average
        (slot 0).  The width is the scheme's per-face count
        ``per_axis^{d-1}``, appended via the single-source
        :func:`~orpheus.numerics.moment_layout.face_moment_tail` (the
        same "append iff > 1" policy the cell-cochain
        :attr:`_LossRepresentation._n_face_moments` /
        :attr:`_spatial_moment_tail` key on).  DD/Step
        (``per_axis == 1`` → ``per_axis^{d-1} == 1`` → ``face_moment_tail
        == ()``) leaves every slot shape untouched, so the trace stays
        byte-identical — the negative control.  A 1-D slab face is a
        point (``face_shape == ()`` → ``per_axis^0 == 1`` → no tail even
        for LD-1D), so the transverse face-moment is a 2-D-and-higher
        concern by construction.

        Returns
        -------
        FaceLayout
            Per-geometry face descriptor. Total flat size = sum of
            ``prod(shape)`` over all faces. Slot order = the canonical
            :attr:`face_labels` order (axis ascending, endpoint in axis
            order), which reproduces the historical hand-listed order.

        Notes
        -----
        The layout contains ONLY boundary face slots. Interior
        wavefront cache cells (pre-D-G stored in AngularBoundaryFlux's 2-D
        ``xmin_xmax_buf`` / ``ymin_ymax_buf`` interior positions) are
        explicitly excluded — post-D-G they are **ephemeral local
        arrays inside the sweep**, allocated per sweep and never
        persisted (the boundary trace is the sole persistent face-state
        carrier). A short-lived ``SweepScratch`` type was introduced to
        hold them and deleted in the same step: a sweep-private
        persistent type re-created exactly the boundary/interior
        conflation D-G was dissolving.
        """
        from orpheus.numerics.face_layout import FaceLayout
        from orpheus.numerics.moment_layout import face_moment_count, face_moment_tail

        N = self.quad.N
        # Per-face transverse moment count per_axis^{d-1} (#251) — the FACE
        # tail (the CELL tail is per_axis^d).  DD/Step → () → byte-identical.
        # Single-sourced with the cochain's ``_n_face_moments`` via
        # ``face_moment_count`` so the producer and the consumer cannot drift.
        n_face_moments = face_moment_count(self.scheme.spatial_basis_per_axis, self.ndim)
        moment_tail = face_moment_tail(n_face_moments)
        return FaceLayout.from_named_shapes([
            (label.face_name, (N, self.ng, *self.face_shape(label), *moment_tail))
            for label in self.face_labels
        ])

    # ── Sweep DAG traversal ───────────────────────────────────────────

    _DEGENERATE_ABS_ETA_THRESHOLD: ClassVar[float] = 1e-15

    def dag_walk(
        self,
        *,
        ordinate_idx: int | None = None,
        direction_sign: int | None = None,
        mu_level_idx: int | None = None,
    ) -> Iterator[CellVisit]:
        r"""Walk the per-ordinate cell DAG in topological order.

        Issue #196 Phase G Step 2.6 (Q3): the single canonical iteration
        primitive for 1-D sweeps.  Yields visits either for a single
        ordinate or for all ordinates of a sweep direction under one
        XOR signature.

        The SN sweep on a given ordinate is forward substitution on
        the block-triangular streaming + collision operator under the
        ordinate's DAG ordering.  This method yields the per-cell
        visit packets in that DAG order; the consumer folds over the
        packets, threading the spatial-upstream face flux through the
        accumulator and writing the per-cell angular state into a
        persistent array.

        Exactly one of ``ordinate_idx`` or ``direction_sign`` must be
        supplied (XOR):

        * ``ordinate_idx=n`` — yields visits for a single ordinate.
          For slab/sphere: ``n`` is the global ordinate index.  For
          cylindrical: ``n`` is the within-level azimuthal index
          :math:`m \in [0, M)` and ``mu_level_idx`` MUST also be
          supplied; the signed :math:`\eta` resolves through
          ``quad.level_indices[mu_level_idx][n]``.
        * ``direction_sign=±1`` — yields visits for the sweep
          direction (``+1`` outward, ``-1`` inward).  Cell ordering
          depends ONLY on the direction sign (and level for
          cylindrical), so any ordinate in the correct sign class
          yields the same cell sequence; this branch picks a
          non-degenerate representative.

        SN-specific by design.  MoC will not consume this method —
        its mathematical structure is fiber bundles + solution
        sheaves, a different DAG shape.  Premature abstraction
        avoided per Cardinal Rule 2.

        Parameters
        ----------
        ordinate_idx : int | None
            See above; mutually exclusive with ``direction_sign``.
        direction_sign : int | None
            See above; mutually exclusive with ``ordinate_idx``.
        mu_level_idx : int | None
            For cylindrical geometry: which :math:`\mu`-level the
            ordinate (subset) belongs to.  ``None`` for slab/sphere;
            required for cylindrical.

        Yields
        ------
        CellVisit
            One per cell, in topological order.  The packet's
            :attr:`face_area_downstream` is float: ``1.0`` for slab,
            ``0.0`` for cylindrical pure-azimuthal degenerate,
            physical face area for sphere / non-degenerate cylinder.

        Raises
        ------
        ValueError
            If neither or both of ``ordinate_idx`` / ``direction_sign``
            are supplied; if ``direction_sign not in (+1, -1)``; if
            called on a 2-D Cartesian mesh (no
            :class:`ReducedStreamingOperator`); if a cylindrical
            mesh is queried without ``mu_level_idx``; or if no
            non-degenerate representative ordinate exists for
            ``direction_sign``.

        Notes
        -----
        2-D Cartesian wavefront scheduling is intentionally not
        encapsulated here — its anti-diagonal vectorisation
        operates on cell slices, not per-cell visits.
        """
        if (ordinate_idx is None) == (direction_sign is None):
            raise ValueError(
                "dag_walk requires exactly one of `ordinate_idx` or "
                "`direction_sign`."
            )
        if not self.is_1d:
            # The honest predicate (P4.5): the chain scan is a 1-D
            # construct; ``reduced`` presence is its ctor-guaranteed
            # realization (populated iff ``is_1d``).
            raise ValueError(
                "dag_walk is only defined for meshes with a "
                "ReducedStreamingOperator (1-D Cartesian, spherical, "
                "or cylindrical).  2-D Cartesian wavefront sweeps "
                "use anti-diagonal scheduling, not per-cell visits."
            )
        coord = self.coord

        # Direction-keyed branch: resolve a non-degenerate representative
        # ordinate, then delegate to the ordinate-keyed branch (single source
        # of truth — Pattern 2). Cylindrical ``mu_level_idx`` is required by
        # both ``_representative_ordinate`` and the cylindrical visit iterator,
        # each via ``_require_mu_level`` (fail-loud at point of use).
        if direction_sign is not None:
            if direction_sign not in (+1, -1):
                raise ValueError(
                    f"direction_sign must be +1 or -1; got "
                    f"{direction_sign}"
                )
            ordinate_idx = self._representative_ordinate(
                direction_sign, mu_level_idx,
            )
        # Exactly one of ordinate_idx / direction_sign was supplied (the XOR
        # guard above); the direction-keyed arm resolved it, so the ordinate
        # is now concrete — narrow for the type checker.
        if ordinate_idx is None:  # pragma: no cover — unreachable per XOR guard
            raise ValueError(
                "dag_walk: ordinate_idx unresolved after mode dispatch."
            )

        # Ordinate-keyed branch.
        if coord is CoordSystem.CARTESIAN:
            yield from self._iter_cartesian_visits(ordinate_idx)
            return
        if coord is CoordSystem.SPHERICAL:
            yield from self._iter_spherical_visits(ordinate_idx)
            return
        if coord is CoordSystem.CYLINDRICAL:
            yield from self._iter_cylindrical_visits(
                ordinate_idx, self._require_mu_level(mu_level_idx),
            )
            return
        raise ValueError(  # pragma: no cover — exhaustive match above
            f"Unknown coord system: {coord!r}"
        )

    def dag_walk_cell_indices(
        self,
        *,
        direction_sign: int,
        mu_level_idx: int | None = None,
    ) -> Iterator[int]:
        r"""Lightweight twin of :meth:`dag_walk` — yields just cell indices.

        Consumers that build their own per-cell algebra from primitives
        (the loss-representation walk in
        :mod:`orpheus.sn.loss_representation` — the former unified matvec
        ``transport_operator_matvec_unified`` was its Depth-B predecessor,
        deleted at the walk unification) only need the cell traversal
        order, not the full
        :class:`~orpheus.transport.spatial.scheme.CellVisit` packet.

        Eliminates per-cell-per-call ``ReducedStreamingOperator.streaming_terms()``
        construction + frozen-dataclass overhead.  PR-TYPED-6c profiling
        showed this was ~14% of matvec time on slab, ~18% on cylinder
        — all building a packet the matvec discards.

        Cell-iteration order matches :meth:`dag_walk`:

        * Slab, sphere, cylinder non-degenerate: ``range(nx)`` for
          :math:`\mu_n \ge 0`, ``range(nx-1, -1, -1)`` for :math:`\mu_n < 0`.
        * Cylindrical pure-azimuthal degenerate
          (:math:`|\eta_n| < 10^{-15}`): ``range(nx)`` regardless of
          ``direction_sign`` — same as :meth:`dag_walk`.
        """
        if not self.is_1d:
            # Same honest predicate as :meth:`dag_walk` (P4.5).
            raise ValueError(
                "dag_walk_cell_indices is only defined for meshes with a "
                "ReducedStreamingOperator (1-D Cartesian, spherical, "
                "or cylindrical)."
            )
        coord = self.coord
        if direction_sign not in (+1, -1):
            raise ValueError(
                f"direction_sign must be +1 or -1; got {direction_sign}"
            )

        # Resolve the representative ordinate's signed primary cosine.
        # (Cylindrical mu_level_idx is required by _representative_ordinate
        # and the global-ordinate lookup below, each via _require_mu_level.)
        ordinate_idx = self._representative_ordinate(
            direction_sign, mu_level_idx,
        )
        if coord is CoordSystem.CYLINDRICAL:
            mu_level = self._require_mu_level(mu_level_idx)
            level_indices = self.quad.level_indices  # type: ignore[attr-defined]
            global_n = int(level_indices[mu_level][ordinate_idx])
            mu_n = float(self.quad.mu_x[global_n])
        else:
            mu_n = float(self.quad.mu_x[ordinate_idx])

        # Cylindrical degenerate ordinates iterate forward regardless of sign.
        if (
            coord is CoordSystem.CYLINDRICAL
            and abs(mu_n) < self._DEGENERATE_ABS_ETA_THRESHOLD
        ):
            yield from range(self.nx)
            return

        if mu_n >= 0:
            yield from range(self.nx)
        else:
            yield from range(self.nx - 1, -1, -1)

    def _require_mu_level(self, mu_level_idx: int | None) -> int:
        """Narrow ``mu_level_idx`` to ``int`` for a cylindrical sweep.

        Cylindrical 1-D radial sweeps are organised by μ-level (a subset of
        azimuthal ordinates at one polar cosine), so every cylindrical
        traversal needs ``mu_level_idx``; slab/sphere pass ``None``. Single
        source of truth for the "cylindrical requires mu_level_idx" contract
        (Pattern 2) — fails loudly (``-O``-safe, not ``assert``) and returns
        the narrowed ``int`` so callers index ``level_indices`` cleanly.
        """
        if mu_level_idx is None:
            raise ValueError(
                "cylindrical sweep requires mu_level_idx (which μ-level the "
                "ordinate subset belongs to); slab/sphere pass None."
            )
        return mu_level_idx

    def _representative_ordinate(
        self,
        direction_sign: int,
        mu_level_idx: int | None,
    ) -> int:
        """Pick a non-degenerate ordinate matching the direction sign.

        Cell ordering in :meth:`dag_walk` depends only on
        ``direction_sign`` (and the level for cylindrical), so any
        non-degenerate ordinate in the correct sign class produces
        the same cell sequence.  The degenerate :math:`|\\eta| <
        10^{-15}` ordinates are excluded because they iterate forward
        regardless of sign and would not match the bulk direction's
        signed iteration.
        """
        assert self.reduced is not None
        coord = self.coord
        eps = self._DEGENERATE_ABS_ETA_THRESHOLD
        if coord is CoordSystem.CYLINDRICAL:
            mu_level = self._require_mu_level(mu_level_idx)
            level_indices = self.quad.level_indices  # type: ignore[attr-defined]
            level_ords = np.asarray(level_indices[mu_level])
            eta_at_level = self.quad.eta[level_ords]
            if direction_sign == +1:
                cand = np.where(eta_at_level > +eps)[0]
            else:
                cand = np.where(eta_at_level < -eps)[0]
            if cand.size == 0:
                raise ValueError(
                    f"No non-degenerate ordinate in cylindrical level "
                    f"{mu_level} satisfies "
                    f"direction_sign={direction_sign}."
                )
            return int(cand[0])
        mu_x = self.quad.mu_x
        if direction_sign == +1:
            cand = np.where(mu_x > +eps)[0]
        else:
            cand = np.where(mu_x < -eps)[0]
        if cand.size == 0:
            raise ValueError(
                f"No non-degenerate ordinate satisfies "
                f"direction_sign={direction_sign} in this quadrature."
            )
        return int(cand[0])

    def _make_cell_visit(
        self,
        *,
        cell_idx: int,
        face_area_downstream: float,
        st: StreamingTerms,
    ) -> CellVisit:
        r"""Assemble one :class:`CellVisit` — purely spatial (P4.9a).

        ALL four ``dag_walk`` yield paths (slab / sphere / cylinder /
        cylindrical-degenerate) funnel through here (Pattern 2 — no
        per-site divergence).  The former Morel--Montry stamp
        (``tau`` / ``c_in`` / ``c_out``, #236 Phase 2 B2/B3) left with
        the un-weld: a MESH copying closure data onto visits was the
        smell — the closure's contributions now reach a scheme as
        assembled arguments the SN walk builds from the closure's own
        minted constants.
        """
        return CellVisit(
            cell_idx=cell_idx,
            streaming_terms=st,
            face_area_downstream=face_area_downstream,
        )

    def _iter_cartesian_visits(
        self,
        ordinate_idx: int,
    ) -> Iterator[CellVisit]:
        """Yield slab (1-D Cartesian) visits in sweep direction.

        Order: forward (cell 0 → nx-1) for :math:`\\mu \\ge 0`,
        backward for :math:`\\mu < 0`.  Slab carries
        ``face_area_downstream = 1.0`` (neutral curvature; Issue
        #196 Phase G Step 2.5) so the unified cell-balance helper
        consumes one geometry-blind number.
        """
        assert self.reduced is not None
        mu_n = float(self.quad.mu_x[ordinate_idx])
        cell_indices = (
            range(self.nx) if mu_n >= 0 else range(self.nx - 1, -1, -1)
        )
        for i in cell_indices:
            st = self.reduced.streaming_terms(
                cell_idx=i, direction_idx=ordinate_idx,
            )
            # Slab: ``direction_idx`` IS the global ordinate.
            yield self._make_cell_visit(
                cell_idx=i,
                face_area_downstream=1.0,
                st=st,
            )

    def _iter_spherical_visits(
        self,
        ordinate_idx: int,
    ) -> Iterator[CellVisit]:
        """Yield spherical visits in sweep direction.

        Outward (:math:`\\mu \\ge 0`): cell 0 → nx-1, downstream face
        is the outer face ``A[i+1]``.  Inward (:math:`\\mu < 0`):
        cell nx-1 → 0, downstream face is the inner face ``A[i]``.
        """
        assert self.reduced is not None
        mu_n = float(self.quad.mu_x[ordinate_idx])
        if mu_n >= 0:
            cell_indices = range(self.nx)
            select_outer = True
        else:
            cell_indices = range(self.nx - 1, -1, -1)
            select_outer = False
        for i in cell_indices:
            st = self.reduced.streaming_terms(
                cell_idx=i, direction_idx=ordinate_idx,
            )
            face_downstream = (
                st.face_area_outer if select_outer else st.face_area_inner
            )
            # Sphere: ``direction_idx`` IS the global ordinate.
            yield self._make_cell_visit(
                cell_idx=i,
                face_area_downstream=face_downstream,
                st=st,
            )

    def _iter_cylindrical_visits(
        self,
        ordinate_idx: int,
        mu_level_idx: int,
    ) -> Iterator[CellVisit]:
        """Yield cylindrical visits in sweep direction for one level.

        ``ordinate_idx`` is the within-level azimuthal index
        :math:`m \\in [0, M)`.  The global ordinate is resolved via
        ``quad.level_indices[mu_level_idx][ordinate_idx]``.

        * :math:`\\eta_n \\ge 0` outward: cell 0 → nx-1, downstream
          face is the outer face.
        * :math:`\\eta_n < 0` inward: cell nx-1 → 0, downstream
          face is the inner face.
        * :math:`|\\eta_n| < 10^{-15}` pure-azimuthal degenerate:
          forward iteration (so the angular M-M closure runs in a
          natural order) but ``face_area_downstream`` is ``None`` —
          no spatial face flow.
        """
        assert self.reduced is not None
        level_indices = self.quad.level_indices  # type: ignore[attr-defined]
        global_n = int(level_indices[mu_level_idx][ordinate_idx])
        eta_n = float(self.quad.eta[global_n])
        abs_eta = abs(eta_n)

        if abs_eta < self._DEGENERATE_ABS_ETA_THRESHOLD:
            # Pure-azimuthal degenerate: no spatial flow.  Iterate
            # forward so the angular M-M closure runs in a natural
            # order; ``face_area_downstream = 0.0`` signals "no
            # spatial flow" to the strategy (geometric truth — the
            # cell has no radial face on this ordinate).  Issue #196
            # Phase G Step 2.5: replaced ``None`` with the
            # geometrically-correct float ``0.0``.
            for i in range(self.nx):
                st = self.reduced.streaming_terms(
                    cell_idx=i,
                    direction_idx=ordinate_idx,
                    mu_level_idx=mu_level_idx,
                )
                # Cylinder: the global ordinate is resolved through the
                # level partition (``global_n`` above) — the SAME index
                # ``streaming_terms`` resolves.
                yield self._make_cell_visit(
                    cell_idx=i,
                    face_area_downstream=0.0,
                    st=st,
                )
            return

        if eta_n >= 0:
            cell_indices = range(self.nx)
            select_outer = True
        else:
            cell_indices = range(self.nx - 1, -1, -1)
            select_outer = False
        for i in cell_indices:
            st = self.reduced.streaming_terms(
                cell_idx=i,
                direction_idx=ordinate_idx,
                mu_level_idx=mu_level_idx,
            )
            face_downstream = (
                st.face_area_outer if select_outer else st.face_area_inner
            )
            # Cylinder: global ordinate via the level partition.
            yield self._make_cell_visit(
                cell_idx=i,
                face_area_downstream=face_downstream,
                st=st,
            )

    # ── Stencil setup ─────────────────────────────────────────────────

    def _setup_cartesian(self) -> None:
        r"""Precompute the raw per-axis down-face streaming coefficient ``g``.

        The **scheme-agnostic** geometric streaming, one array per spatial
        axis:

        .. math::

            g_a \;=\; \frac{|\mu_a|\,A_{\rm down}}{V}
                \;=\; \frac{|\mu_a|}{\Delta a}
                \qquad(\text{Cartesian tensor-product: } A_{\rm down}/V = 1/\Delta a)

        This is NOT the DD denominator term.  The diamond-difference closure
        contributes :math:`\Sigma_t + \sum_a 2g_a` to the cell-balance
        denominator — the factor :math:`2 = 1/w_{\rm DD}` is DD's diamond
        closure (:math:`\psi_{\rm out} = 2\bar\psi - \psi_{\rm in}`), owned by
        the *scheme* and applied inside its cell kernel, NOT baked into this
        geometric accessor (#240).  Linear-Discontinuous reads the same raw
        ``g`` without DD's factor.

        Precomputing avoids per-ordinate per-cell divisions in the inner
        sweep loop.  Built over ``range(ndim)`` from the canonical per-axis
        accessors (``quad.axis_cosines(a)`` — the legacy ``mu_x`` / ``mu_y``
        names are property views of exactly these columns), with NO phantom
        axis: a 1-D mesh carries one streaming array, not an ``ny=1`` second.
        """
        # _streaming_axes[a][n, i] = |μ_a[n]| / Δa[i] — the RAW down-face
        # streaming g (shape (N_ord, n_a)); the scheme owns its closure factor.
        self._streaming_axes: tuple[np.ndarray, ...] | None = tuple(
            np.abs(self.quad.axis_cosines(a))[:, None]
            / widths[None, :]
            for a, widths in enumerate(self.axis_widths)
        )


    # ⛔ The backward-compat accessors ``face_areas`` and ``delta_A``
    # retired here on 2026-08-27 (P4.1c).  They routed to
    # ``self.reduced`` and emitted a ``DeprecationWarning``; they dated
    # from Wave E Round 2 (#164), against ``coding-standards``' rule that
    # a deprecation alias lives for ONE merge cycle.
    #
    # `[M]` by AST over ``orpheus/`` + ``tests/`` at retirement: **11
    # reads, 0 of them in ``orpheus/``** — every consumer was a test, and
    # the tests were the ones written to verify the shims.  A shim kept
    # alive by its own coverage.  Read ``self.reduced.face_areas`` /
    # ``self.reduced.delta_A``, which is what these forwarded to.
