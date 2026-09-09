r"""Mesh + materials — the method-agnostic transport data carrier.

:class:`MaterialMesh` is the "mesh + materials" middle type the codebase
was missing.  Between the geometry :class:`~orpheus.geometry.mesh.Mesh1D`
/ :class:`~orpheus.geometry.mesh.Mesh2D` (which carry material *ids* but
no :class:`~orpheus.data.macro_xs.mixture.Mixture` cross sections) and a
method-specific phase space such as
:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` (mesh + materials + *quadrature* +
sweep machinery) there was no carrier for *just* mesh + materials.

The abstraction axis is **data vs behavior**:

* :class:`MaterialMesh` is the **method-agnostic transport state / data**
  — the axes/mesh, the per-material :class:`Mixture` dict, the material
  map, cell volumes, the uniform group count :attr:`ng`, the natural
  volume measure, and the macroscopic XS field built from them.
* The **method layer** (angular quadrature + sweep/streaming stencil +
  boundary trace + closures) is *behavior*.  A method-specific mesh
  **is a** :class:`MaterialMesh` that adds that behavior:
  ``SNMesh(MaterialMesh)`` (quadrature + sweep machinery + angular
  trace) and ``DiffusionMesh(MaterialMesh)`` (scalar trace + realized
  boundary laws; #290 P7a), each conforming **structurally** to the
  :class:`~orpheus.transport.method.TransportMethod` Protocol (minted
  at #290 P7b over both witnesses; the shared
  :func:`~orpheus.transport.method.resolve_boundary_conditions` body
  resolves every method-mesh's BC declarations through it).

This is the layer where cross-section **homogenization** lands: a
fine-mesh :class:`~orpheus.sn.solution.Solution` plus a coarse
:class:`~orpheus.geometry.mesh.Mesh1D` / :class:`~orpheus.geometry.mesh.Mesh2D`
produce a homogenized
:class:`MaterialMesh` (flux·volume-weighted collapse), which a transport
method can then *promote* back to a solvable phase space
(:meth:`SNMesh.from_material_mesh`).

Layer (``tests/test_layer_imports.py``): L2 ``transport``.  It imports
only ``geometry`` (legacy mesh shapes), ``numerics`` (the volume
measure), ``data`` (the :class:`Mixture` type, ``TYPE_CHECKING``), and
its sibling :mod:`~orpheus.transport.mesh.axis` /
:mod:`~orpheus.transport.mesh.material_xs_field` modules.  It imports no
L3 method package — which is exactly what let it be promoted out of
``orpheus.sn``.
"""

from __future__ import annotations

from functools import cached_property, reduce
from typing import TYPE_CHECKING

from collections.abc import Mapping

import numpy as np

from orpheus.data.materials import Materials
from orpheus.geometry import Mesh1D, Mesh2D
# The SPACE-FACTOR axis vocabulary (campaign 1, CS1) — aliased because this
# module's own ``Axis1D``/``self.axes`` are GEOMETRIC axes (a different
# concept; the naming coordination is the Q3 rename issue).
from orpheus.numerics.manifold import RealSpace
from orpheus.numerics.axis import Axis as SpaceFactorAxis
from orpheus.numerics.axis import BasisKind, EnergyAxis
from orpheus.numerics.space import FunctionSpace
from orpheus.transport.mesh.axis import (
    Axis1D,
    AxisMesh,
    axes_from_legacy_mesh,
    coord_system as _axis_coord_system,
    spatial_shape as _axis_spatial_shape,
)

if TYPE_CHECKING:
    from orpheus.data.macro_xs.mixture import Mixture
    from orpheus.geometry import CoordSystem
    from orpheus.transport.mesh.material_xs_field import MaterialXSField


__all__ = ["InconsistentMaterialsError", "MaterialMesh"]


class InconsistentMaterialsError(ValueError):
    """Raised when a materials dict has inconsistent metadata.

    Currently triggered when materials disagree on ``ng`` (number of
    energy groups).  A :class:`MaterialMesh` requires a uniform group
    structure across all materials in its ``mat_map`` because every
    transport operator that consumes the mesh assumes one ``ng``.  A
    homogenization / energy-condensation step must precede method-mesh
    construction if the input materials carry different group
    structures.
    """


class MaterialMesh:
    r"""Method-agnostic mesh + materials carrier.

    Axis-primary (C5.1, #225): the canonical spatial representation is
    :attr:`axes` — a tuple of :class:`~orpheus.transport.mesh.axis.Axis1D`
    — from which all shape metadata derives.  Constructed either from a
    legacy :class:`~orpheus.geometry.mesh.Mesh1D` /
    :class:`~orpheus.geometry.mesh.Mesh2D` (converted to axes once at the
    inbound boundary; the legacy object is retained as :attr:`mesh` for
    the consumers still reading through it) or axis-natively via
    :meth:`from_axes`.

    Parameters
    ----------
    mesh : Mesh1D or Mesh2D
        Base geometry.  Its material assignment (``mat_ids`` on
        :class:`Mesh1D`, ``mat_map`` on :class:`Mesh2D`) keys into
        ``materials``.
    materials : Materials or mapping of material id to Mixture
        The stage-1 declaration (a bare ``{id: Mixture}`` dict is parsed
        into :class:`~orpheus.data.materials.Materials` at this
        boundary). The authoritative source of truth for both cross
        sections and the group count :attr:`ng`.  All materials must
        agree on ``ng`` — heterogeneous group structures are a
        homogenization-step concern that must precede method-mesh
        construction.

    Attributes
    ----------
    mesh : Mesh1D or Mesh2D or None
        Inbound provenance / legacy adapter (``None`` for axis-native
        d≥3 meshes).
    axes : tuple of Axis1D
        Canonical spatial representation.
    materials : Materials
        The stage-1 declaration (parsed at construction; single source
        of truth — its mapping surface serves every dict-shaped read).
    mat_map : np.ndarray
        Material-id assignment, shape :attr:`spatial_shape`.
    ng : int
        Number of energy groups, derived from materials and validated
        for consistency.
    """

    def __init__(
        self,
        mesh: Mesh1D | Mesh2D,
        materials: "Materials | Mapping[int, Mixture]",
    ) -> None:
        # Legacy inbound surface (C5.1 axis-primary inversion, #225):
        # convert the Mesh1D / Mesh2D declaration to the canonical axis
        # tuple ONCE at the boundary, extract the material assignment the
        # axes cannot carry (``mat_ids`` on Mesh1D, ``mat_map`` on
        # Mesh2D), and run the same data-construction body as
        # :meth:`from_axes`.
        self._init_data(
            axes=axes_from_legacy_mesh(mesh),
            mesh=mesh,
            mat_map=mesh.mat_ids if isinstance(mesh, Mesh1D) else mesh.mat_map,
            materials=materials,
        )

    def _init_data(
        self,
        *,
        axes: tuple[Axis1D, ...],
        mesh: Mesh1D | Mesh2D | None,
        mat_map: np.ndarray | None,
        materials: "Materials | Mapping[int, Mixture]",
    ) -> None:
        r"""The ONE data-construction body both surfaces funnel into.

        Subclasses (``SNMesh``) call this from their own ``_init_core``
        to populate the method-agnostic data block, then layer their
        behavior (quadrature, sweep stencil, boundary trace) on top.
        Every line here is bit-for-bit the data block formerly inlined
        in ``SNMesh._init_core`` (C5.1) — the split is a pure
        relocation, not a semantic change.
        """
        # ``materials`` is REQUIRED: a MaterialMesh without materials has
        # no ``ng`` and no XS field — an illegal state (coding-elegance
        # Pattern 4).  Parsed at THIS boundary into the stage-1
        # :class:`~orpheus.data.materials.Materials` declaration (un-weld
        # arc, R20/R21): a bare ``{id: Mixture}`` dict is admitted and
        # normalized once; downstream the type is ``Materials`` (its
        # mapping surface keeps every existing read working). The single
        # source of truth for the declaration lives here; every operator
        # reads materials + ``ng`` from the mesh, not from a parallel
        # argument.
        self.mesh = mesh
        self.materials: "Materials" = Materials.of(materials)
        # The axis tuple is the PRIMARY representation (C5.1): stored
        # verbatim — never round-tripped through a legacy mesh and
        # re-derived — it is the canonical dim-agnostic ground truth for
        # spatial_shape / coord / axis_widths.
        self.axes: tuple[Axis1D, ...] = tuple(axes)
        # ``np.diff(edges)`` is bitwise identical to the legacy spellings
        # it replaces (``Mesh1D.widths`` / ``Mesh2D.dx`` / ``Mesh2D.dy``).
        # ``axis_widths`` is THE single spelling of per-axis cell widths,
        # positional-by-axis.
        self.axis_widths: tuple[np.ndarray, ...] = tuple(
            np.diff(ax.edges) for ax in self.axes
        )
        # Material assignment: the one construction payload the axes do
        # not carry. ``None`` (axis-native default) → single material
        # with id 0; shape MUST match spatial_shape (parse, don't
        # validate downstream).
        if mat_map is None:
            mat_map = np.zeros(self.spatial_shape, dtype=int)
        else:
            mat_map = np.asarray(mat_map, dtype=int)
            if mat_map.shape != self.spatial_shape:
                raise ValueError(
                    f"MaterialMesh: mat_map shape {mat_map.shape} must "
                    f"match spatial_shape={self.spatial_shape}"
                )
        self.mat_map: np.ndarray = mat_map
        # Cell volumes / radial face areas stay dataclass-owned while the
        # adapter is present (preserves the Mesh1D curvilinear formulas +
        # the ``precomputed_volumes`` ULP escape hatch bit-identically).
        # Axis-native (mesh-less, d≥3 — all-Cartesian by construction):
        # the cell volume is the tensor-product cell measure, the
        # iterated outer product of the per-axis widths. 2-D per-face
        # areas have a different shape and feed no matvec caller — None.
        if mesh is not None:
            self._volumes: np.ndarray = mesh.volumes
            # The type carries the dimensionality fact the old ``ndim == 1``
            # int-guard only implied: per-face areas exist on Mesh1D only.
            self._areas: np.ndarray | None = (
                mesh.areas if isinstance(mesh, Mesh1D) else None
            )
        else:
            self._volumes = reduce(np.multiply.outer, self.axis_widths)
            self._areas = None
        # ``nx`` = spatial_shape[0] sugar.
        self.nx: int = self.spatial_shape[0]
        # The whole-mesh coordinate system derives from the axes
        # (multi-axis tuples are all-Cartesian by construction).
        self.coord: "CoordSystem" = _axis_coord_system(self.axes)

        # ── Materials consistency validation (Issue #197 PR-TYPED-0) ──
        # Two checks at construction time:
        #   (1) every material id used in ``mat_map`` must have an entry
        #       in ``materials`` — otherwise downstream code would key
        #       into an undefined material.
        #   (2) all materials must agree on ``ng`` — one uniform group
        #       structure; heterogeneous ``ng`` is a homogenization-step
        #       concern that must precede method-mesh construction.
        # Both surface at construction, NOT lazily — the failure mode
        # (operators built on a bad mesh) is action-at-a-distance
        # otherwise.
        self._validate_materials()
        # Trigger ``ng`` property's consistency check eagerly so
        # mismatched-ng materials raise at construction time.
        _ = self.ng

    # NOTE: a GENERAL axis-native ``MaterialMesh.from_axes`` (arbitrary
    # cell count / coordinate system) is still intentionally NOT provided.
    # ``SNMesh.from_axes`` already exists with a different
    # (quadrature-bearing) signature, so a base ``from_axes`` here would be
    # an incompatible override; and the base class has no axis-native
    # consumer today — the infinite-medium problem poses on its own space
    # (``HomogeneousProblem.space``, Energy ⊗ the counting point) and
    # builds no carrier at all since the CS4c coda (until then a
    # ``from_materials`` factory fabricated a mesh-less one-cell carrier
    # here whose ``[0, 1]`` edges, node and chart nothing consumed;
    # retired 2026-09-08). ``.homogenize`` still builds via the legacy
    # ``MaterialMesh(coarse_mesh, materials)`` ctor. Defer the general
    # form until a real N-cell consumer exists (defer-until-≥2-instances).

    # ── Materials validation ──────────────────────────────────────────

    def _validate_materials(self) -> None:
        """Validate the assignment against the declaration.

        Every material id referenced in ``self.mat_map`` MUST appear in
        the declaration — discharged through the declaration's OWN
        reachable-subset constructor
        (:meth:`~orpheus.data.materials.Materials.restrict`, the
        assigned-but-undeclared guard at its one home; un-weld arc). The
        empty-declaration refusal fires even earlier, at the
        ``Materials`` admission inside ``_init_data``'s parse. Failure
        surfaces at construction time, not lazily inside a solver step.

        Raises
        ------
        ValueError
            If any ``mat_map`` id is missing from the declaration; the
            error message shows both sets so the user can see the gap.
        """
        self.materials.restrict(int(x) for x in np.unique(self.mat_map))

    # ── Properties ────────────────────────────────────────────────────

    @property
    def ng(self) -> int:
        """Number of energy groups; uniform across all materials.

        Derived from ``self.materials``; the single source of truth for
        the group count.  All materials must agree on ``ng`` — a
        method-mesh requires one uniform group structure across the mesh.

        Raises
        ------
        InconsistentMaterialsError
            If materials disagree on ``ng``.  A homogenization /
            condensation step must precede method-mesh construction in
            that case.
        ValueError
            Never for emptiness here — the empty declaration is refused
            at the ``Materials`` admission inside ``_init_data``'s
            parse, so a constructed carrier always has ≥1 material.
        """
        ngs = {m.ng for m in self.materials.values()}
        if len(ngs) != 1:
            raise InconsistentMaterialsError(
                f"MaterialMesh requires uniform ng across all materials; "
                f"got ng values {sorted(ngs)} in materials dict with keys "
                f"{sorted(self.materials.keys())}.  Homogenize / condense "
                f"to a common group structure before method-mesh "
                f"construction."
            )
        return ngs.pop()

    @property
    def volumes(self) -> np.ndarray:
        """Cell volumes, shape ``spatial_shape`` (rank ``ndim``)."""
        return self._volumes

    @cached_property
    def cells_by_material(self) -> dict[int, tuple[np.ndarray, ...]]:
        r"""Per-material cell-index arrays — the mesh's material LAYOUT.

        For each declared material id, the ``np.where`` index tuple such
        that ``mat_map[indices] == mid`` everywhere — one index array PER
        MESH AXIS (``(ix,)`` on 1-D, ``(ix, iy)`` on 2-D).  A pure
        function of :attr:`mat_map`, so it lives HERE (the mesh owns its
        machinery), cached once, and is SHARED by every per-material
        field built over this mesh
        (:class:`~orpheus.transport.material_field.MaterialField` — the
        CS4c kernel fields — and, transitionally, the
        :class:`~orpheus.transport.mesh.material_xs_field.MaterialXSField`
        facade's read-through) — no consumer recomputes the ``where``
        partition.

        A material declared but UNUSED on the map yields empty index
        arrays — harmless to every accumulation verb (empty fancy
        index, empty einsum).

        Returns
        -------
        dict[int, tuple[np.ndarray, ...]]
            ``{mid: indices}`` for every id in :attr:`materials`; tuple
            arity is the mesh ndim.
        """
        return {
            mid: np.where(self.mat_map == mid)
            for mid in self.materials
        }

    # CS1.5 re-point: ``bulk_space`` moves to ``Medium`` (the carrier concept
    # that genuinely owns it); the property rides that carve unchanged.
    @cached_property
    def bulk_space(self) -> FunctionSpace:
        r"""The scalar-bulk function space of this carrier — axis-built (CS1).

        The UNIFORM formula, honest on EVERY member of the hierarchy
        (single generic body, Cardinal Rule 2)::

            of_axes(energy_axis, SpaceFactorAxis("spatial", spatial_shape,
                                                 weights=volumes, NODAL))

        * **The quotient point** — the space the infinite-medium problem
          poses on (``HomogeneousProblem.space``: Energy ⊗ a counting
          point, shape ``(ng, 1)``) — is minted from the Mixture, not from
          a carrier, since the CS4c coda. A genuine UNIT-width one-cell
          mesh reaches the same space through this formula: ``[M]`` its
          volumes are ``[1.0]``, so the spatial factor canonicalizes to
          the COUNTING weight — the normalized "per unit volume" density
          convention (collapse doctrine, clause 1) — and its
          ``bulk_space`` is ``==`` the pose (G2.1 pins it). Until the
          coda a fabricated ``from_materials`` carrier played this role.
        * **A genuine one-cell mesh of width ≠ 1** keeps ``V ≠ 1`` BY THE
          DATA — distinguished from the quotient point by MEASURE, hence
          (through the derived name) by space identity. ⚠ Provably
          invisible to ``.H`` (a scalar metric commutes with every
          operator — the F2 measurement); identity is the only
          instrument that carries it.
        * **A meshed carrier** (``SNMesh``/``DiffusionMesh`` inherit this)
          gets the honest scalar bulk ``(ng, *spatial)`` with cell-volume
          weights — the seed of CS2's single scalar-bulk mint. It is NOT
          the angular composite: ``SNMesh.bulk_space`` and
          ``SNMesh.full_field_space`` are different spaces with different
          jobs.

        **The energy arm** reads only materials REACHABLE from ``mat_map``
        (the leak principle: the mint consults exactly its defining data —
        a declaration may carry SPECTATOR materials no cell references
        (``Materials ⊋ reachable``), and a spectator with
        ``eg=None`` must not flip the axis identity of a problem it does
        not touch). The reachable set then goes through the ONE energy-arm
        rule, :meth:`~orpheus.numerics.axis.EnergyAxis.from_materials`
        (hoisted there at CS4a K1 so the mixture-minted homogeneous space
        and this property cannot spell the rule twice). Deterministic per
        carrier; deliberately NO new construction-time refusal — grid
        coherence across materials is a per-data-kind consistency
        concern (the data-layer overhaul, charter R22), and the mint's
        synthetic fallback is the shipped law.

        Cached: every consumer of one carrier reads the SAME instance
        (and equal carriers mint ``==`` spaces through the derived name).
        """
        reachable = self.materials.restrict(
            sorted(int(i) for i in np.unique(self.mat_map))
        )
        energy = EnergyAxis.from_materials(reachable.values())
        if len(self.spatial_shape) == 1:
            # The carrier's volume measure generates the spatial axis
            # (CS5, user ruling: "the mesh is able to generate a Discrete
            # Measure of space"). ``self.volume_measure`` is the ONE
            # documented data path — nodes = cell centres, weights = THIS
            # carrier's ``volumes`` (delegated to the legacy mesh when
            # present, bit-identically) — so the minted axis has exactly
            # the structural content of the literal it replaced, plus its
            # generator.
            spatial = self.volume_measure.axis("spatial")
        else:
            # Rank-d axes have no rank-d measure->axis mint: the measure
            # is FLAT (nodes ``(N, d)``, weights ``(N,)``) while the axis
            # is rank-d — the CS2 rank-d seam, gated as a CONTRACT (G6b):
            # this arm's axis stays generator-less until CS2 mints the
            # rank-d pairing, and inverting that row must be deliberate.
            spatial = SpaceFactorAxis(
                "spatial",
                self.spatial_shape,
                weights=self.volumes,
                kind=BasisKind.NODAL,
            )
        return FunctionSpace.of_axes(energy, spatial)

    def integrate_per_group(self, density: np.ndarray) -> np.ndarray:
        r"""Volume-integrate a per-group cell density into a per-group rate.

        .. math::

            R_g \;=\; \int_V d_g(\mathbf{r})\, dV \;=\; \sum_i V_i \, d[g, i]

        ``density`` is a principled ``(ng, *spatial_shape)`` field; the
        result is ``(ng,)``.  The integral IS :attr:`volume_measure` — this
        method owns only the axis bookkeeping that measure needs (it
        consumes a flat ``(N_cells, ng)`` view, Issue 9.6 wiring), so the
        cell weights have exactly one source.

        ⭐ **It is the INTEGRAL, deliberately not "the reaction rate".**
        :meth:`~orpheus.sn.solver.SNSolver.compute_group_production_rate`
        and its absorption sibling integrate a reaction-rate *density*
        :math:`\Sigma_x \phi` — they are the composition of a
        cross-section weighting with this.  Other consumers integrate a
        density with no cross section in it at all (the per-group balance
        defect of a residual field, #340 N6b).  Folding them onto one
        "reaction rate" name would be the [[lessons-L30]] error — same
        data, different operation.

        ⛔ Lives on :class:`MaterialMesh`, NOT on a solver, because it
        needs exactly :attr:`ng` and :attr:`volume_measure` and both are
        the mesh's.  It shipped on ``SNSolver`` for one commit
        (`b0137171`) and moved here the same day: three of the five #340
        N6b call sites hold a mesh and no solver, and
        :class:`~orpheus.diffusion.augmented_mesh.DiffusionMesh` is the
        same base, so a solver-side home would have been a twin the
        moment a second method wanted it.
        """
        ng = self.ng
        return self.volume_measure(
            np.moveaxis(np.asarray(density, dtype=float), 0, -1).reshape(-1, ng)
        )

    @property
    def volume_measure(self):
        r"""Cell-volume :class:`~orpheus.numerics.measure.DiscreteMeasure`.

        The natural integration measure :math:`\mu_V = \sum_i V_i\,
        \delta_{c_i}` for volume-integrated rates (keff
        production/absorption; the flux·volume weighting of
        homogenization).  Consumers read THIS property, not
        ``mesh.mesh.volume_measure`` — the legacy mesh adapter is a
        construction provenance detail, not a data path.

        Delegates to the legacy dataclass's measure while the adapter is
        present (bit-identity: same atoms, same construction — including
        the ``precomputed_volumes`` escape hatch and the curvilinear
        volume formulas the dataclass owns).  Axis-native
        (``self.mesh is None``, d≥3): the rank-d analogue — atoms are the
        cell-centre tuples ordered with ``np.meshgrid(..., indexing='ij')``
        (the same layout ``volumes.ravel()`` exposes), weights the
        flattened cell volumes.
        """
        if self.mesh is not None:
            return self.mesh.volume_measure
        from orpheus.numerics.measure import DiscreteMeasure
        centers = [0.5 * (ax.edges[:-1] + ax.edges[1:]) for ax in self.axes]
        grids = np.meshgrid(*centers, indexing="ij")
        nodes = np.stack([g.ravel() for g in grids], axis=-1)
        return DiscreteMeasure(
            nodes=nodes,
            weights=self.volumes.ravel(),
            support=RealSpace(self.ndim),
        )

    @property
    def areas(self) -> np.ndarray:
        """Face areas at each radial edge, shape (nx+1,) (1-D meshes).

        Sourced from :attr:`Mesh1D.areas`.  Cartesian slab returns an
        array of ones; cylinder returns :math:`2\\pi r`; sphere returns
        :math:`4\\pi r^2`.

        Raises
        ------
        AttributeError
            If the carrier holds no per-face areas — two DISTINCT arms,
            each naming its own case (S7 G7.2; pre-repair one message
            claimed "2-D meshes" for both, false on one): the 2-D legacy
            mesh (areas live on the ``Mesh2D``) and the d≥3 axis-native
            carrier (no legacy mesh at all). ``mesh is None`` has ONE
            meaning — the d≥3 axis-native carrier: every d≤2 constructor
            carries a mesh (``tests/transport/test_material_mesh_admission.py``
            pins the theorem). Until the CS4c coda a third arm served the
            mesh-less infinite-medium 1-cell carrier that shared the
            sentinel (S7 G7.3 discriminated the two by ``ndim``); that
            carrier retired with its factory, 2026-09-08.
        """
        if self._areas is None:
            if self.mesh is None:
                raise AttributeError(
                    f"MaterialMesh.areas: the {self.ndim}-D axis-native "
                    "carrier holds no per-face areas (radial face areas "
                    "are a Mesh1D curvilinear concept; no matvec "
                    "consumes Cartesian face areas today)."
                )
            raise AttributeError(
                "MaterialMesh.areas is not defined for 2-D meshes; "
                "face-area data lives in the underlying Mesh2D directly."
            )
        return self._areas

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions; equals ``len(self.axes)``."""
        return len(self.axes)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        r"""Per-axis cell counts ``(n_0, n_1, ...)``.

        The canonical dim-agnostic shape descriptor.  Every dim-agnostic
        shape reader (typed-field factories, pack convention, sweep DAG)
        reads from here. ``self.nx`` is sugar for ``spatial_shape[0]``.
        """
        return _axis_spatial_shape(self.axes)

    # ── Macroscopic XS field ──────────────────────────────────────────

    def material_xs_field(self) -> "MaterialXSField":
        """Build the macroscopic XS field from this mesh's materials.

        Returns a
        :class:`~orpheus.transport.mesh.material_xs_field.MaterialXSField`
        wrapping the per-material :class:`Mixture` data plus this mesh's
        ``mat_map`` — the single source of truth for both per-cell and
        per-material XS access used by every transport operator.

        Lazy import of :mod:`.material_xs_field` to avoid a circular
        dependency at module import time.
        """
        from orpheus.transport.mesh.material_xs_field import MaterialXSField
        return MaterialXSField.from_mesh(self)
