"""Shared test helpers for the SN test suite.

Issue #197 PR-TYPED-0 introduced ``materials`` as a REQUIRED parameter
on :class:`SNMesh`.  Many geometry-only tests don't consume cross-
section values — they exercise sweep DAGs, BC realisation,
quadrature, cache structure, etc.  ``placeholder_materials`` provides
a minimal-but-valid :class:`Mixture` dict that those tests can hand
to :class:`SNMesh` so the construction succeeds without inviting any
real cross-section semantics into the test.

Issue #197 PR-TYPED-2 introduced :class:`AngularBoundaryFlux` as the typed
replacement for the stringly-typed ``psi_bc: dict``.  Test fixtures
that previously passed ``{}`` to :func:`transport_sweep` should now
build a zero-initialised :class:`AngularBoundaryFlux` via
``AngularBoundaryFlux.zeros(sn_mesh.angular_trace)`` (or :func:`make_boundary_flux_zero`
below for non-SNMesh callers).

Tests that DO need realistic cross sections continue to use
``orpheus.derivations.common.xs_library.get_mixture`` etc. — this
helper is for the geometry-only call sites.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.fields.scalar_flux import ScalarFlux
from orpheus.sn import solve_sn_fixed_source

if TYPE_CHECKING:
    from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
    from orpheus.transport.mesh.material_xs_field import MaterialXSField
    from orpheus.sn.mesh.augmented_mesh import SNMesh
    from orpheus.transport.fields.scalar_flux import ScalarFlux
    from orpheus.transport.timed_full_field import TimedFullField


# Anchor for shared, sn-root-relative test data (regression snapshots,
# the sweep reference vector, the Wave-T fixture .npz files).  The
# capability-taxonomy reorg nests tests several directories deep
# (e.g. ``sweep/curvilinear/``); a ``Path(__file__).parent``-relative
# data lookup would break on every move.  Every consumer resolves data
# through this single anchor so the data store stays at the sn-root
# regardless of where the test that reads it lives.
SN_TESTS_ROOT = Path(__file__).resolve().parent
"""Absolute path to ``tests/sn/`` — the anchor for shared test data."""


def volume_weighted_l2(
    values: np.ndarray, reference: np.ndarray, volumes: np.ndarray
) -> float:
    r"""Discrete measure-weighted L2 norm of a field error,
    :math:`\lVert v - v_{\rm ref}\rVert_{2,V} = \sqrt{\sum_i V_i\,(v_i - v_{{\rm ref},i})^2}`.

    The single source of truth for the MMS / convergence gates' error norm.
    ``volumes`` is the cell measure of the mesh: cell widths for a slab,
    radial shell volumes for curvilinear 1-D, cell areas/volumes in 2-D — in
    every case the discrete measure that makes the sum a quadrature of
    :math:`\int (v-v_{\rm ref})^2\,\mathrm{d}V`. Issue #249 retired the
    byte-identical private copies that the ``tests/sn/verification/mms/`` gates
    each used to re-mint (``_l2_1d`` / ``_l2`` / ``_l2_2d`` / ``_l2_error`` /
    ``_cell_l2``); this is now the sole implementation — new gates consume it.
    """
    diff = values - reference
    return float(np.sqrt(np.sum(volumes * diff * diff)))


# MMS convergence-ladder solver knobs — the fixed-source inner-iteration budget
# the mms/ gates share, tight enough that the discretisation error (not the
# iteration residual) sets the measured L2.
MMS_MAX_INNER = 500
MMS_INNER_TOL = 1e-13


def scalar_flux_l2_ladder(case, n_cells) -> np.ndarray:
    """Volume-weighted scalar-flux L2 error ladder for an MMS ``case`` over the
    mesh-refinement sequence ``n_cells``.

    The single source of truth for the "build mesh → solve fixed-source →
    measure :func:`volume_weighted_l2` of the scalar flux vs ``case.phi_exact``"
    recipe the curvilinear / space–angle convergence gates share.  ``case`` is
    any MMS case exposing ``build_mesh`` / ``external_source`` / ``materials`` /
    ``quadrature`` / ``phi_exact``.  Returns the error ladder aligned with
    ``n_cells``.
    """
    errors = []
    for nc in n_cells:
        mesh = case.build_mesh(nc)
        Q = case.external_source(mesh)
        result = solve_sn_fixed_source(
            case.materials, mesh, case.quadrature, Q,
            max_inner=MMS_MAX_INNER, inner_tol=MMS_INNER_TOL,
        )
        phi_num = result.scalar_flux.values[0, :]
        phi_ref = case.phi_exact(mesh.centers)
        errors.append(volume_weighted_l2(phi_num, phi_ref, mesh.volumes))
    return np.asarray(errors)


def stamp_capability_marker(items, conftest_file: str, capability: str) -> None:
    """Apply ``@pytest.mark.cap(<capability>)`` to every test under a dir.

    The capability-taxonomy reorg encodes the SN-capability tier of a
    test as the *directory* it lives in (single source of truth). Rather
    than decorate every test file with a ``cap(...)`` marker — which can
    drift from the directory it documents — each capability directory
    carries a one-line ``conftest.py`` that delegates here. Every item
    collected at or below the conftest's directory gets the marker; the
    existing ``l0/l1/l2/foundation/verifies/catches`` markers on each
    test are untouched (``cap`` is orthogonal and composable).

    Parameters
    ----------
    items
        The collected items passed to ``pytest_collection_modifyitems``.
    conftest_file
        ``__file__`` of the calling capability-directory conftest.
    capability
        The capability name (one of the DAG nodes; see the ``cap``
        marker description in ``pyproject.toml``).
    """
    import pytest

    here = Path(conftest_file).resolve().parent
    marker = pytest.mark.cap(capability)
    for item in items:
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:
            continue
        # ``here`` is the item's parent OR an ancestor — covers both a
        # flat capability dir and a nested one (e.g. sweep/core under
        # sweep). The sweep/ conftest does NOT stamp because each leaf
        # (core, slab, curvilinear, cartesian_2d) carries its own.
        if item_path.parent == here:
            item.add_marker(marker)



def placeholder_materials(
    ng: int = 1, mat_ids: tuple[int, ...] = (0,),
) -> dict:
    """Build a placeholder ``{mat_id: Mixture}`` dict for SNMesh tests.

    Parameters
    ----------
    ng
        Number of energy groups.  All Mixtures will report this value
        via :attr:`Mixture.ng`.
    mat_ids
        Material ids to include in the dict.  Default ``(0,)`` covers
        the common case where the mesh's ``mat_ids`` / ``mat_map`` is
        zeros.

    Returns
    -------
    dict[int, Mixture]
        Each entry has ``SigT = ones(ng)`` and all other cross sections
        zero.  Suitable for SNMesh tests that don't compute physical
        quantities from the materials.
    """
    from orpheus.data.macro_xs.mixture import Mixture
    z = np.zeros(ng)
    z_mat = csr_matrix(np.zeros((ng, ng)))
    return {
        int(mid): Mixture(
            SigC=z.copy(),
            SigL=z.copy(),
            SigF=z.copy(),
            SigP=z.copy(),
            SigT=np.ones(ng),
            SigS=[z_mat],
            Sig2=[z_mat],
            chi=z.copy(),
        )
        for mid in mat_ids
    }


def _as_stack(channel) -> "list":
    """A single matrix is the P0 stack; a list/tuple is the stack as given."""
    return list(channel) if isinstance(channel, (list, tuple)) else [channel]


def material_xs_from_raw(
    *,
    sig_s: "dict[int, list[np.ndarray]]",
    sig2: "dict[int, np.ndarray | list[np.ndarray]] | None" = None,
    cells_by_mat: "dict[int, tuple[np.ndarray, np.ndarray]]",
    ng: int,
    nx: int,
    ny: int = 1,
) -> "MaterialXSField":
    """Build a REAL ``MaterialXSField`` from raw per-material scattering data.

    The production-path replacement for the retired
    ``MaterialXSField._synthetic_for_tests`` (campaign 1 CS4b, Q7 ruling —
    option 3): real :class:`Mixture` objects carry the given Legendre
    lists and (n,2n) matrices — or (n,2n) Legendre STACKS, a list per
    material, since #426 step 2 — a real ``Mesh2D`` paints ``cells_by_mat``
    into its ``mat_map``, and the field is built by the production
    ``MaterialXSField.from_mesh`` — so the per-material dispatch tests
    exercise the true admission + lazy dense-cache path (including the
    EE-4 read-only cache freeze) instead of pre-populated caches on a
    duck-typed mesh stub, which the space-carrying accessors of CS4b
    cannot serve.

    The grid is always rank-2 ``(nx, ny)`` (the stub's convention — the
    consuming fixtures shape their arrays accordingly), unit-pitch
    Cartesian. Non-scattering channels are zero and ``SigT = 1``: these
    fixtures assert scattering dispatch, never a physical balance.
    Every cell must be painted — a real carrier has no unassigned cells.
    """
    from orpheus.geometry import Mesh2D
    from orpheus.transport.mesh.material_mesh import MaterialMesh
    from orpheus.transport.mesh.material_xs_field import MaterialXSField
    from orpheus.data.macro_xs.mixture import Mixture

    mat_map = np.full((nx, ny), -1, dtype=int)
    for mid, (ix, iy) in cells_by_mat.items():
        mat_map[np.asarray(ix), np.asarray(iy)] = mid
    if (mat_map < 0).any():
        raise ValueError(
            "material_xs_from_raw: cells_by_mat must paint every cell of "
            f"the ({nx}, {ny}) grid — a real carrier has no unassigned "
            "cells."
        )

    z = np.zeros(ng)
    materials = {
        int(mid): Mixture(
            SigC=z.copy(),
            SigL=z.copy(),
            SigF=z.copy(),
            SigP=z.copy(),
            SigT=np.ones(ng),
            SigS=[csr_matrix(np.asarray(s)) for s in mats],
            Sig2=[
                csr_matrix(np.asarray(block))
                for block in _as_stack(
                    sig2[mid] if sig2 is not None and mid in sig2
                    else np.zeros((ng, ng))
                )
            ],
            chi=z.copy(),
        )
        for mid, mats in sig_s.items()
    }
    mesh = Mesh2D(
        edges_x=np.arange(nx + 1, dtype=float),
        edges_y=np.arange(ny + 1, dtype=float),
        mat_map=mat_map,
    )
    return MaterialXSField.from_mesh(MaterialMesh(mesh, materials))


# ── B3.2: face-ful method spaces for mesh-less realizer tests ────────


def face_trace(quadrature, faces=("xmin", "xmax")):
    r"""An :class:`AngularTraceSpace` over ``faces`` for a bare quadrature.

    Campaign phase **B3.2** typed the SN boundary law :math:`\Gamma_+ \to
    \Gamma_-`, so realizing one needs the face's OUTFLOW ordinates (its
    domain) as well as its inflow ordinates (its codomain) — face-orientation
    data a quadrature alone cannot supply. Every mesh-less realizer test
    therefore needs a face, and this is the ONE place the trace is stood up
    (Pattern 2: a per-file copy of the layout incantation is a drift habitat).

    ``AngularTraceSpace.from_quadrature_and_layout`` is geometry-blind — it
    derives the outward normals from the ``"{axis}{min|max}"`` face-name
    convention — so no mesh is needed.
    """
    from orpheus.numerics.face_layout import FaceLayout
    from orpheus.numerics.spaces.angular_trace_space import AngularTraceSpace

    layout = FaceLayout.from_named_shapes(
        [(f, (int(quadrature.N),)) for f in faces]
    )
    return AngularTraceSpace.from_quadrature_and_layout(quadrature, layout)


def local_positions(global_rows, index_set) -> np.ndarray:
    r"""Positions of ``global_rows`` within ``index_set`` — by LINEAR SCAN.

    The test-side reference for B3.2's ``sel → position-within-Γ_S`` remap.
    Deliberately a DIFFERENT algorithm from production's
    :meth:`~orpheus.numerics.spaces.angular_trace_space.AngularFaceTraceSpace.to_local`
    (``np.searchsorted``, a binary search that additionally requires a sorted
    haystack; owned by the half-trace space since G6.5): a reference computed
    by calling ``to_local`` would share the very code the remap gates exist to
    check, making the cross-check procedural rather than structural.

    Raises ``KeyError`` when a requested row is not in ``index_set`` — the
    "two index sets were crossed" condition production reports as a
    ``ValueError``. Callers that probe for it should catch ``KeyError``.
    """
    lookup = {int(k): i for i, k in enumerate(np.asarray(index_set))}
    return np.array([lookup[int(k)] for k in np.asarray(global_rows)], dtype=int)


def face_method_space(quadrature, face="xmax", faces=("xmin", "xmax")):
    r"""A face-ful :class:`SNMethodSpace` carrying BOTH half-traces.

    The B3.2 successor of ``SNMethodSpace.minimal(quadrature)`` for tests that
    realize a narrowed law: ``minimal`` is faceless, so it can name neither
    :math:`\Gamma_+` nor :math:`\Gamma_-` and the realizer refuses it (loudly,
    by design).

    Which laws need a face, as of **B3.4a**:

    * **vacuum**, **reflective** (narrowed at B3.2), **white** and
      **prescribed_inflow** (narrowed at B3.4a) all require it. White
      additionally requires the face to MATCH the law's declared
      ``axis``/``outward_sign`` — ``"xmax"`` ⇔ ``axis="x", outward_sign=+1``,
      ``"ymin"`` ⇔ ``axis="y", outward_sign=-1``, and so on — or the
      realizer's orientation cross-check refuses it (the ERR-041 pattern; a
      mismatched white law averages the wrong hemisphere).
    * **albedo** (with a closure, B3.4b) and **periodic** (B3.4c) require it
      too — periodic doubly so, since its domain is a *different* face's
      :math:`\Gamma_+` and a faceless space cannot even name the partner.
      ``minimal`` remains correct only for dispatch-failure tests and for
      probing the refusals themselves.

    (Before B3.4a this docstring listed white among the full-face laws; until
    G6.3 step 7 it listed albedo and periodic. Both claims were true when
    written and became false with the narrowings — the second was caught
    late, in step 7's audit, after B3.4c had already made it false.)
    """
    from orpheus.sn.mesh.method_space import SNMethodSpace

    return SNMethodSpace.for_face(
        quadrature=quadrature, face=face,
        trace=face_trace(quadrature, faces),
    )


# ── Shared curvilinear / slab mesh builders ──────────────────────────
#
# Lifted verbatim from the legacy ``test_cylindrical.py`` /
# ``test_spherical.py`` modules during the SN taxonomy reorg so the
# cylinder + sphere split files (eigenvalue/, sweep/curvilinear/,
# verification/analytical/) share ONE definition rather than each
# carrying a copy.

_COORD_TO_TAG = {
    "CARTESIAN": "SLB",
    "CYLINDRICAL": "CYL",
    "SPHERICAL": "SPH",
}


def _bcs_for(tag: str, bc):
    """BC tuple matching the geometry tag's endpoint count."""
    if tag == "SLB":
        return (bc, bc)
    return (bc,)


def curvilinear_homogeneous_mesh(
    n_cells: int,
    total_width: float,
    mat_id: int = 0,
    coord=None,
    bc=None,
):
    """Single-region uniform mesh in any coordinate system.

    SN tests default to ``BC.reflective`` (the eigenvalue / lattice
    convention). CP tests must override to ``BC.white`` because CP
    only supports ``"vacuum"`` / ``"white"``.
    """
    from orpheus.geometry import (
        BC, CoordSystem, Mesh1D, Region, RegionMesh, StructuredGeometry,
    )
    if coord is None:
        coord = CoordSystem.CARTESIAN
    if bc is None:
        bc = BC.reflective
    tag = _COORD_TO_TAG[coord.name]
    geom = StructuredGeometry(
        geometry=tag,
        regions=(Region(mat_id=mat_id, outer_thickness_cm=total_width),),
        bcs=_bcs_for(tag, bc),
    )
    return Mesh1D.from_geometry(geom, region_meshes=(RegionMesh(n_cells=n_cells),))


def curvilinear_two_region_mesh(
    outers,
    mat_ids,
    n_cells,
    coord,
    bc=None,
):
    """Two-region mesh with absolute outer-edge convention."""
    from orpheus.geometry import (
        BC, Region, RegionMesh, StructuredGeometry,
    )
    from orpheus.geometry import Mesh1D
    if bc is None:
        bc = BC.reflective
    tag = _COORD_TO_TAG[coord.name]
    geom = StructuredGeometry(
        geometry=tag,
        regions=(
            Region(mat_id=mat_ids[0], outer_thickness_cm=outers[0]),
            Region(mat_id=mat_ids[1], outer_thickness_cm=outers[1] - outers[0]),
        ),
        bcs=_bcs_for(tag, bc),
    )
    return Mesh1D.from_geometry(geom, region_meshes=(
        RegionMesh(n_cells=n_cells[0]),
        RegionMesh(n_cells=n_cells[1]),
    ))


def make_tiny_spherical_sn_mesh(n_cells: int = 2, sn_order: int = 2) -> "SNMesh":
    """Minimal bound-closure host: an ``n_cells``-cell reflective sphere.

    The cheapest SNMesh satisfying the angular-closure family's
    ``cls(sn_mesh)`` construction contract (C5, 2026-07-03, retired the
    unbound ``MorelMontryAngularSweep()`` legacy mode) — for foundation
    tests of strategy construction, registry ``create``, repr, and seed
    wiring that need a real bound instance but never read its
    coefficients.
    """
    from orpheus.geometry import CoordSystem
    from orpheus.numerics.quadrature import Quadrature
    from orpheus.sn.mesh.augmented_mesh import SNMesh

    mesh = curvilinear_homogeneous_mesh(
        n_cells, 1.0, coord=CoordSystem.SPHERICAL,
    )
    return SNMesh(
        mesh, Quadrature.gauss_legendre(sn_order), placeholder_materials(),
    )


def cart2d_2g_nonsquare(nx: int = 5, ny: int = 7) -> "SNMesh":
    """2-D Cartesian, reflective, 2G, NON-SQUARE (the x↔y-swap moat).

    The discriminating config for structural operator/representation
    tests: the octant frame, the pure-z branch, and the interior
    recurrence all degenerate on a 1G flat square box (vv §H1/§H2).
    Promoted from ``test_one_octant_walk.py`` when the S6.5
    one-instance tests became its second consumer.
    """
    from orpheus.geometry import BC, CoordSystem, Mesh2D
    from orpheus.numerics.quadrature import Quadrature
    from orpheus.sn.mesh.augmented_mesh import SNMesh

    mesh = Mesh2D(
        edges_x=np.linspace(0.0, 2.0, nx + 1),
        edges_y=np.linspace(0.0, 3.0, ny + 1),
        mat_map=np.zeros((nx, ny), dtype=int),
        coord=CoordSystem.CARTESIAN,
        bc_xmin=BC("reflective"), bc_xmax=BC("reflective"),
        bc_ymin=BC("reflective"), bc_ymax=BC("reflective"),
    )
    return SNMesh(mesh, Quadrature.level_symmetric(4), placeholder_materials(ng=2))


def random_radial_characteristic_field(sn: "SNMesh", rng):
    """A random ψ½ FLUX composite whose per-slot values reproduce the retired
    unified leaf's single-buffer draw BIT-IDENTICALLY (``None`` on non-carrying).

    Phase C 4e retired the unified single-buffer leaf (which then held the
    ``RadialCharacteristicField`` name) in favour of the split
    ``interior ⊕ boundary`` composite — reminted as
    :class:`RadialCharacteristicField` at 4e-e1b. The unified leg walk
    (:func:`~orpheus.numerics.spaces.radial_characteristic_space._radial_characteristic_legs`)
    interleaves ``cells`` then ``corner`` PER ``(level, sign)`` leg, and the old
    fill was one ``rng.standard_normal(unified.values.shape)`` over that flat
    buffer. Drawing per-leg in the SAME order — the ``cells`` slot then the
    ``corner`` slot, each an ``rng.standard_normal(slot.shape)`` — consumes the
    identical rng stream and reshapes C-order into the identical slots, so the
    frozen walk / affine-carve baselines hold at ``nulp=1``. Returns ``None`` on
    a non-carrying mesh (R12a): System B does not exist there.
    """
    from orpheus.transport.radial_characteristic_field import (
        RadialCharacteristicField,
    )

    if sn.radial_characteristic_field_space is None:
        return None
    comp = RadialCharacteristicField.flux_zeros(sn.radial_characteristic_field_space)
    for level in sn.radial_characteristic_levels:
        for sign in (-1, +1):
            cells = comp.interior.cells(level, sign)
            cells[...] = rng.standard_normal(cells.shape)
            corner = comp.boundary.corner(level, sign)
            corner[...] = rng.standard_normal(corner.shape)
    return comp


def het_operands(sn: "SNMesh"):
    """Heterogeneous σ_t + a non-flat random state (≥2G, non-degenerate).

    Returns ``(sig_t, psi, seed)``: a random per-group per-cell total
    cross section, a 2-block
    :class:`~orpheus.transport.timed_full_field.TimedFullField` with
    random bulk AND boundary values, and — on a carrying mesh (R12a) —
    the random ψ½ FLUX composite (System B's split-native member) for the
    walk's explicit ``radial_characteristic_flux`` leg (B.2d / 4e;
    ``None`` on slab/cyl).  The rng draw ORDER (bulk → faces → seed) matches
    the pre-eviction 3-block builder exactly (the per-leg seed draw is
    bit-faithful to the retired unified single-buffer fill — see
    :func:`random_radial_characteristic_field`), so the frozen walk
    baselines hold bit-identically — every term of the loss action stays
    activated (nothing nulled by a flat or zero state).
    """
    from orpheus.transport.fields.angular_flux import AngularFlux
    from orpheus.transport.timed_full_field import TimedFullField

    rng = np.random.default_rng(20260611)
    sig_t = rng.uniform(0.3, 3.0, size=(sn.ng, *sn.spatial_shape))
    psi = TimedFullField.zeros(
        interior=AngularFlux, boundary=AngularBoundaryFlux, space=sn.full_field_space,
    )
    psi.interior.values[...] = rng.standard_normal(psi.interior.values.shape)
    for face in psi.boundary.layout.faces:
        fv = psi.boundary.face_view(face)
        fv[...] = rng.standard_normal(fv.shape)
    seed = random_radial_characteristic_field(sn, rng)
    return sig_t, psi, seed


def legacy_proxy_matvec(
    psi_view: "np.ndarray", sn_mesh: "SNMesh", sigma_t: "np.ndarray",
    *, bc_outer=None, angular_closure=None,
) -> "np.ndarray":
    """Call :func:`_transport_operator_matvec_unified` with the
    cell-centre-proxy boundary fill semantics (pre-B1'' convention).

    Tests that compare against L0 hand-derived references built BEFORE
    the B1'' face-aware architecture (i.e. references constructed
    around "no face state, fall back to cell-centre proxy" semantics)
    feed bare ``psi_view`` ``(N, ng, nx, ny)`` ndarrays and expect a
    bare ``(N, ng, nx, ny)`` cell-output ndarray.  This helper bridges:

    1. Build a :class:`AngularBoundaryFlux` whose face buffers carry
       ``psi_view``'s cell-centre value at the outer (and slab-inner)
       face — the cell-centre-proxy fill.
    2. Wrap into a :class:`TimedFullField`.
    3. Call :func:`_transport_operator_matvec_unified`.
    4. Return ``result.interior.values`` as a bare ndarray.

    The "legacy" prefix refers to the BOUNDARY-FILL CONVENTION (the
    pre-B1'' cell-centre proxy), not to retired code.  Production
    code uses the B1'' face-aware path via
    :class:`StreamingCollisionOperator` (= ``L + C``); this helper exists only
    for L0 tests that pin the legacy convention's behaviour against
    closed-form hand references.
    """
    from orpheus.transport.fields.angular_flux import (
        AngularFlux,
    )
    from orpheus.transport.fields.angular_boundary_flux import (
        AngularBoundaryFlux,
    )
    from orpheus.transport.timed_full_field import TimedFullField
    from orpheus.sn.operators.streaming import (
        StreamingOperator,
    )
    from orpheus.transport.operators.multiplication_operator import MultiplicationOperator

    # Wave T T.5 close-out (matvec retirement): route through the
    # public operator-algebra path `(L + C).apply`.  The legacy
    # `_transport_operator_matvec_unified` helper was DELETED; the
    # canonical 1-D matvec body is now `_OneDimScanWalk._apply_walk`
    # (the fused `(L+C)ψ`), and `(L + C).apply` = `L.apply + C.apply`
    # = `((L+C)ψ - sigma_t * psi) + sigma_t * psi` = (L+C)
    # bit-exact for the legacy semantic.  The ``bc_outer`` /
    # ``angular_closure`` override parameters are not used by
    # any production caller of this helper today (all call sites pass
    # `bc_outer=None, angular_closure=None`) — kept in the
    # function signature for legacy back-compat but ignored.
    del bc_outer, angular_closure  # explicitly mark unused
    boundary = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
    boundary.face_view("xmax")[:] = psi_view[:, :, -1]
    if "xmin" in boundary.layout.faces:
        boundary.face_view("xmin")[:] = psi_view[:, :, 0]
    # #282 route (a): the "legacy proxy" pins the PRE-route-(a) matvec
    # convention (seed = the input field extrapolated in μ to the level's
    # edge — the retired AngularEdgeExtrapolation-of-the-iterate).  On a
    # carrying mesh (sphere, R12a) route (a) reads the seed as STATE, so
    # to reproduce the old convention bit-exactly for the L0 hand-
    # reference tests, fill the ψ½ block with the closure's edge
    # extrapolation of ``psi_view`` (the cells legs; corners = the same
    # edge value so a constant field telescopes to σ_t·ψ).  Non-carrying
    # meshes → None, byte-identical to the pre-2.5d helper.  Since Q5.6.3
    # the Cartesian charts are the only admitted non-carrying ones: the
    # ADMITTED cylinder's folded rule carries on every level.
    radial_characteristic = radial_characteristic_edge_seed(psi_view, sn_mesh)
    composite = TimedFullField(
        interior=AngularFlux(values=psi_view, space=sn_mesh.angular_bulk_space),
        boundary=boundary,
        _history=(),
        history_depth=2,
    )
    L_op = StreamingOperator.pose(sn_mesh)
    C_op = MultiplicationOperator.from_mesh(sigma_t, sn_mesh)
    result = _LC_matvec(
        composite, sigma_t, sn_mesh=sn_mesh, LC=(L_op + C_op),
        radial_characteristic_flux=radial_characteristic,
    )
    return result.interior.values


def radial_characteristic_edge_seed(psi_view, sn_mesh):
    """The pre-route-(a) ψ½ seed: the input field extrapolated in μ to each
    carrying level's starting-direction edge (the retired
    ``AngularEdgeExtrapolation``-of-the-iterate convention), so an
    augmented matvec fed THIS seed reproduces the old seed-from-iterate
    value.  ``None`` on non-carrying meshes.

    Shared by :func:`legacy_proxy_matvec` (the L0 hand-reference bridge)
    and the phase-C composite builder — a CONSISTENT, LINEAR-in-``psi``
    seed (a constant field extrapolates to the same constant, so
    ``(L+C)·const = σ_t·const`` still holds, and the augmented apply
    stays a linear operator)."""
    if sn_mesh.radial_characteristic_field_space is None:
        return None
    from orpheus.transport.radial_characteristic_field import (
        RadialCharacteristicField,
    )

    closure = sn_mesh.angular_closure
    psi_g_first = psi_view[..., 0].swapaxes(0, 1) if psi_view.ndim == 4 else psi_view.swapaxes(0, 1)
    seed = RadialCharacteristicField.flux_zeros(sn_mesh.radial_characteristic_field_space)
    for p in sn_mesh.radial_characteristic_levels:
        level_idx = closure.level_indices[p]
        psi_level = psi_g_first[:, level_idx, :]          # (ng, M_p, nx)
        edge = closure.edge_extrapolated_seed(psi_level, p)  # (ng, nx)
        for sign in (-1, +1):
            seed.interior.cells(p, sign)[...] = edge
            seed.boundary.corner(p, sign)[...] = edge[:, -1]
    return seed


def _LC_matvec(
    psi: "TimedFullField", sigma_t: "np.ndarray",
    *,
    sn_mesh=None,
    LC=None,
    radial_characteristic_flux=None,
) -> "TimedFullField":
    r"""Test-helper shim: returns ``(L + C).apply(psi)`` as a composite.

    Step 6 (the two-channel collapse): a bare call is the ray-decoupled
    ``(A,A)`` block matvec (the walk's only channel); a seed-carrying
    call routes the JOINT row-A action ``LC·ψ_A + Seeding·ψ_B`` through
    THE GRID (:func:`joint_m_grid` — the production joint spelling;
    presence is structural, never a kwarg channel).  ``LC`` optionally
    injects a pre-built ``(L+C)`` composite.

    Wave T T.5 close-out (matvec retirement, post-T.5.2): the module-
    level helper ``_transport_operator_matvec_unified`` was DELETED;
    the canonical 1-D matvec body is now
    :meth:`~orpheus.sn.loss_representation._OneDimScanWalk._apply_walk`
    (the ``(L+C)ψ`` walk).  This shim constructs the canonical
    ``(L + C)`` operator-algebra composite and delegates to its public
    :meth:`apply` — the migration target for tests that previously
    called the deleted helper directly.
    """
    from orpheus.sn.operators.streaming import StreamingOperator
    from orpheus.transport.operators.multiplication_operator import MultiplicationOperator
    # CS4b S4: fields no longer carry the mesh — the caller passes the
    # carrier (required unless a pre-built LC is injected AND the call is
    # seedless; the carrying arm builds the joint grid off the carrier).
    if sn_mesh is None and (LC is None or radial_characteristic_flux is not None):
        raise TypeError("_LC_matvec: pass sn_mesh= (or a pre-built LC=)")
    if LC is None:
        L = StreamingOperator.pose(sn_mesh)
        C = MultiplicationOperator.from_mesh(sigma_t, sn_mesh)
        LC = L + C
    if radial_characteristic_flux is None:
        return LC.apply(psi)
    from orpheus.numerics.coupled_system import CoupledField

    grid, _space = joint_m_grid(sn_mesh, LC)
    joint = grid.apply(
        CoupledField(systems=(psi, radial_characteristic_flux)),
    )
    return joint.systems[0]


def make_boundary_flux_zero(sn_mesh: "SNMesh") -> "AngularBoundaryFlux":
    """Build a zero-initialised :class:`AngularBoundaryFlux` for ``sn_mesh``.

    Issue #197 PR-TYPED-2 — typed replacement for ``psi_bc = {}``.
    Allocates only the buffers the mesh's geometry consumes (slab gets
    two 1-D faces; curvilinear gets one outer face; 2-D Cartesian gets
    the persistent ``(N, ng, nx+1, ny)`` / ``(N, ng, nx, ny+1)``
    buffers).  Per-geometry dispatch lives inside the mesh's cached
    ``angular_trace`` layout; this helper is a clean alias so test
    fixtures don't have to chain through ``sn_mesh``.
    """
    return AngularBoundaryFlux.zeros(sn_mesh.angular_trace)


def make_scalar_flux_zero(sn_mesh: "SNMesh") -> "ScalarFlux":
    """Build a zero-initialised :class:`ScalarFlux` for ``sn_mesh``."""
    return ScalarFlux.zeros(sn_mesh.bulk_space)


def redistribution_via_live_path(
    psi_level: "np.ndarray",    # (ng, M, nx)
    alpha: "np.ndarray",        # (M+1,)
    dAw: "np.ndarray",          # (nx, M)
    tau: "np.ndarray",          # (M,)
    V: "np.ndarray",            # (nx,)
    *,
    psi_half_seed=None,         # (ng, nx) starting-direction seed array | None
) -> "np.ndarray":
    r"""Single-level M-M redistribution :math:`R_{m,i,g}` via the LIVE surface.

    Issue #248 — the dead legacy ``MorelMontryAngularSweep.__call__`` bundle
    (and its private ``_weighted_angular_recurrence_single_level`` kernel) was
    retired.  This helper reconstructs the SAME redistribution that bundle
    returned for one level, but through the production algebra:

    * the half-angle ψ-thread :math:`\phi_{m\pm 1/2,i,g}` comes from the
      module-level
      :func:`~orpheus.sn.angular.closure.compute_psi_half_per_level`
      — the pure-algebra surface whose recurrence kernel
      (``_psi_half_grid_single_level``) is the SAME one the matvec's
      :meth:`~MorelMontryAngularSweep.precompute_psi_state` consumes (the C5
      unbound-mode retirement, 2026-07-03, moved this surface off the class;
      no closure instance is needed).  The optional ``psi_half_seed`` is a
      plain ``(ng, nx)`` starting-direction seed array (``None`` ⇒ the Phase B
      zero seed); #282 route (a) (2026-07-04) retired the seed-strategy zoo
      and its context object, so the
      seed is now the raw array the closure would itself compute (the
      angular-edge extrapolation on non-carrying levels, the composite ψ½
      state on carrying levels);
    * the geometry redistribution fold
      :math:`R_m = (\Delta A/w)_{i,m}/V_i\,(\alpha_{m+1/2}\phi_{m+1/2}
      - \alpha_{m-1/2}\phi_{m-1/2})` is applied here explicitly (the caller
      owns ``α``, ``ΔA/w``, ``V``).

    The reconstruction is byte-faithful to the retired
    ``_weighted_angular_recurrence_single_level``: that kernel called the SAME
    ``_psi_half_grid_single_level`` (via ``compute_psi_half_per_level`` here)
    with the SAME ``psi_half_seed`` and applied the IDENTICAL α-weighted fold
    loop.

    Single source of truth (Cardinal Rule 2): the foundation closure test
    (``tests/sn/sweep/curvilinear/test_angular_closure.py``) imports this
    helper for the α/ΔA/w/τ/V redistribution fold under hand-built coefficient
    arrays.  (The L0 cylinder hand-reference in
    ``test_unified_matvec_cylinder.py`` was migrated off this helper in #282
    route (a) — it now consumes the mesh-bound closure's ``precompute_psi_state``
    per-level half-angle grid directly.)
    """
    from orpheus.sn.angular.closure import (
        compute_psi_half_per_level,
    )

    ng, M, nx = psi_level.shape
    grid = compute_psi_half_per_level(
        psi_level, tau, psi_half_seed=psi_half_seed,
    )
    faces = grid.faces  # (ng, M+1, nx); faces[:, m, :] = φ_{m-1/2}
    redist = np.empty((ng, M, nx))
    for m in range(M):
        redist[:, m, :] = (
            dAw[:, m].reshape(1, nx)
            * (alpha[m + 1] * faces[:, m + 1, :]
               - alpha[m] * faces[:, m, :])
            / V.reshape(1, nx)
        )
    return redist


# The B.2b/B.2c FusedRay*Gain test oracles (the retired production
# adapters' verbatim bodies, kept through d1 as the fused-composition
# reference) DISSOLVED at B.2d d2 with the 3-block carrier: the fused
# spelling they embedded into is unrepresentable (``FullField`` is
# 2-block), and their consumer gates re-expressed onto the record's
# named splitting (grid ≡ M − N; N ≡ the pieces; the walk's explicit
# leaf-kwarg legs).


def sweep_once(source, sig_t, sn_mesh, boundary_flux):
    """One full physical transport sweep — the typed successor of the
    retired operator-free ``transport_sweep`` (step 6).

    Routes the SAME physics through the production operator surfaces:
    the ``(L+C).solve`` WDD sweep on a seedless mesh; the joint
    within-group M grid (:func:`joint_m_grid` — ``[[LC, Seeding],
    [None, march]]``) on a carrying mesh, with the q½ member folded from
    the source by the ONE fold factory exactly as the retired
    self-derivation did.  Consumes/mutates ``boundary_flux`` in place
    (the old in-out buffer contract: inflow slots read as the seed, the
    marched trace written back) and returns ``(angular_values,
    scalar_values)`` — the scalar recomputed by the quadrature
    contraction (tolerance-equivalent to the walk's per-leg
    accumulation, not bit-pinned).

    ERR-071 role conversion: the buffer's stale OUTFLOW rows are iterate
    state, not rhs data — the exact inverse honours rhs outflow rows as
    the defect rhs (``ψ_out = streamed − rhs_out``), so the rhs boundary
    is built through :meth:`AngularBoundarySourceSink.prescribed_inflow`
    (inflow slots only; outflow rows unrepresentable), preserving the
    in-out buffer contract exactly.
    """
    from orpheus.numerics.coupled_system import CoupledField
    from orpheus.sn.operators.streaming import StreamingOperator
    from orpheus.transport.operators.multiplication_operator import (
        MultiplicationOperator,
    )
    from orpheus.transport.radial_characteristic_field import (
        RadialCharacteristicField,
    )
    from orpheus.transport.source_sinks import AngularBoundarySourceSink
    from orpheus.transport.timed_full_field import TimedFullField

    LC = StreamingOperator.pose(sn_mesh) + MultiplicationOperator.from_mesh(
        sig_t, sn_mesh,
    )
    rhs = TimedFullField(
        interior=source,
        boundary=AngularBoundarySourceSink.prescribed_inflow(
            sn_mesh,
            {
                face: boundary_flux.face_view(face)
                for face in boundary_flux.layout.faces
            },
        ),
        _history=(),
        history_depth=2,
    )
    if sn_mesh.radial_characteristic_field_space is not None:
        q_half = RadialCharacteristicField.source_from_angular(
            np.asarray(source.values), sn_mesh,
        )
        grid, _space = joint_m_grid(sn_mesh, LC)
        psi_a = grid.solve(CoupledField(systems=(rhs, q_half))).systems[0]
    else:
        psi_a = LC.solve(rhs)
    boundary_flux.values[...] = psi_a.boundary.values
    values = np.asarray(psi_a.interior.values)
    scalar = np.einsum("n,ng...->g...", sn_mesh.quad.weights, values)
    return values, scalar


def joint_m_grid(sn_mesh: "SNMesh", LC):
    """The step-5 joint ``M`` — the honest upper-triangular grid
    ``[[LC, Seeding], [None, march]]`` over the given (possibly variant)
    ``L + C`` — returning ``(grid, space)``.

    The ONE test-fixture spelling of the joint resolvent (the successor of
    the retired ``CoupledInvertibleOperator`` fused bridge, deleted at
    step-5d): the march re-uses the SAME σ_t field object the ``LC`` sum's
    multiplication member carries (``LC.b.coefficient`` — mesh-identity
    intact, no reconstruction), exactly as ``build_within_group_system``
    shares one field between ``C`` and the march. ``grid.solve`` is the
    numerics block back-substitution, ``grid.inverse()`` the
    ``CoupledSubstitutionOperator``.
    """
    from orpheus.numerics.coupled_system import CoupledOperator, CoupledSpace
    from orpheus.sn.operators.radial_characteristic import (
        RadialCharacteristicSeeding,
    )

    space = CoupledSpace.from_systems(
        (sn_mesh.full_field_space, sn_mesh.radial_characteristic_field_space),
    )
    march = rc_march(sn_mesh, LC.b.coefficient)
    grid = CoupledOperator(
        [[LC, RadialCharacteristicSeeding(sn_mesh)], [None, march]],
        domain=space, codomain=space,
    )
    return grid, space


# ═══════════════════════════════════════════════════════════════════════
# The independent G-metric oracle + spectrum reduction (#276 A4 sweep)
# ═══════════════════════════════════════════════════════════════════════
# Shared across tests/sn/operators/ (the reciprocity gates) and
# tests/sn/solve/ (the adjoint entry/certification batteries).  These
# are VERIFICATION ORACLES: built directly from raw mesh/quadrature
# data, structurally independent of the production metric machinery
# (anti-R1) — never re-spell them locally, import from here.


def g_bulk_measure(sn: "SNMesh") -> np.ndarray:
    r"""G_bulk = V_cell · w_n [⊗ moment mass] — built from raw mesh data.

    On a multi-moment closure (LD) the bulk field carries the trailing
    ``2^d`` spatial-moment axis, and its Hilbert measure carries the moment
    mass ``∏_a θ^{o_a}`` (#310 C2 ruling 3): ``G_bulk = V·w_n ⊗ diag(1, θ,
    …)``.  Rebuilt HERE from the raw ``sn.scheme.theta`` scalar with a
    plain kron loop — structurally independent of the production
    ``moment_mass_diagonal`` helper (anti-R1), so the metric cross-checks
    pin the production θ-weighting against an independent spelling.
    """
    w_n = np.asarray(sn.quad.weights, dtype=float)
    V = np.asarray(sn.volumes, dtype=float)  # (*spatial,)
    # (N, 1) ordinate+group axes ⊗ (*spatial,) volume axes — rank-generic.
    w_b = w_n.reshape((w_n.shape[0], 1) + (1,) * V.ndim)
    base = w_b * V[None, None]
    if sn.scheme.spatial_basis_per_axis > 1:
        from orpheus.transport.spatial.linear_discontinuous import (
            LinearDiscontinuous,
        )

        # Explicit raise, not a bare assert: this module is not
        # assertion-rewritten, so ``python -O`` (the canonical
        # invocation) would strip an assert to a no-op (vv Mode 8).
        if not isinstance(sn.scheme, LinearDiscontinuous):
            raise TypeError(
                "multi-moment bulk measure: scheme must be "
                "LinearDiscontinuous (the θ carrier); got "
                f"{type(sn.scheme).__name__}."
            )
        theta = float(sn.scheme.theta)
        mm = np.array([1.0])
        for _ in range(sn.ndim):
            mm = np.kron(mm, np.array([1.0, theta]))
        return base[..., None] * mm
    return base


def g_trace_cosine_weight(
    sn: "SNMesh", face_idx: int, *, with_cosine: bool,
) -> np.ndarray:
    r"""Per-ordinate trace weight for a face: ``|Ω·n|·w_n`` (true) or ``w_n`` (wrong)."""
    w_n = np.asarray(sn.quad.weights, dtype=float)
    if with_cosine:
        return np.abs(sn.angular_trace.omega_dot_n[face_idx]) * w_n
    return w_n  # the L11 wrong metric: drops |Ω·n|


def g_inner(a: "TimedFullField", b: "TimedFullField", sn: "SNMesh", *,
            with_cosine: bool = True) -> float:
    r"""``⟨a,b⟩_G = Σ_bulk a·b·(V·w_n) + Σ_trace a·b·(|Ω·n|·w_n)``.

    Built directly from ``omega_dot_n`` / ``quad.weights`` / ``volumes`` —
    the structurally-independent reference inner product on System A's
    2-block composite. ``with_cosine=False`` drops the ``|Ω·n|`` factor
    (the L11 wrong-metric control).

    B.2d: the ψ½ seed is System B's own composite — its ``G_sd = V_cell``
    reciprocity lives on the COUPLED space (the grid ``.H`` gate,
    ``test_psi_half_coupling::TestCoupledBuilder``), never as a third term
    here.
    """
    bulk = float(np.sum(g_bulk_measure(sn) * a.interior.values * b.interior.values))
    trace = 0.0
    for f_idx, face in enumerate(sn.angular_trace.layout.faces):
        af = a.boundary.face_view(face)
        bf = b.boundary.face_view(face)
        w_face = g_trace_cosine_weight(sn, f_idx, with_cosine=with_cosine)
        w_b = w_face.reshape((w_face.shape[0],) + (1,) * (af.ndim - 1))
        trace += float(np.sum(af * bf * w_b))
    return bulk + trace


def g_coupled_diagonal(sn: "SNMesh") -> np.ndarray:
    r"""The COUPLED G-metric diagonal from raw mesh data, in flat order.

    The full solution metric of a carrying (System-B) mesh, as ONE flat
    diagonal aligned with ``CoupledField.to_flat()`` order
    (``system-A interior ⊕ system-A trace ⊕ ray interior ⊕ ray trace``):

    * bulk — ``V_cell · w_n`` (⊗ moment mass on LD) via
      :func:`g_bulk_measure`;
    * trace — per-face ``|Ω·n| · w_n`` via :func:`g_trace_cosine_weight`,
      written through the boundary field's ``face_view`` (a view into the
      flat backing buffer, so the face layout is never hand-derived);
    * ray — the ψ½ STATE metric ``G_sd = V_cell`` (interior slots) and
      ``V[-1]`` (the boundary gauge slot), the hand gauge of
      ``test_radial_characteristic_carrier``.

    Built ENTIRELY from raw data (``sn.volumes`` / ``quad.weights`` /
    ``omega_dot_n`` / ``slot_view`` layout) — never from a space's stored
    ``inner_product_weights`` — so a production metric bug is CAUGHT by a
    gate using this diagonal rather than inherited into its reference
    (anti-R1; the coupled sibling of :func:`g_inner`).
    """
    from orpheus.transport.fields.angular_boundary_flux import (
        AngularBoundaryFlux,
    )
    from orpheus.transport.fields.angular_flux import AngularFlux

    interior = AngularFlux.zeros(sn.angular_trial_space)
    bulk = np.broadcast_to(
        g_bulk_measure(sn), interior.values.shape,
    ).ravel().astype(float)

    bfield = AngularBoundaryFlux.zeros(sn.angular_trace)
    for f_idx, face in enumerate(sn.angular_trace.layout.faces):
        w_face = g_trace_cosine_weight(sn, f_idx, with_cosine=True)
        view = bfield.face_view(face)
        view[:] = np.broadcast_to(
            w_face.reshape((w_face.shape[0],) + (1,) * (view.ndim - 1)),
            view.shape,
        )
    trace = np.asarray(bfield.values, dtype=float).copy()

    ii = sn.radial_characteristic_interior_space
    bb = sn.radial_characteristic_boundary_space
    # Explicit raises, not bare asserts: this module is not
    # assertion-rewritten, so ``python -O`` would strip asserts (Mode 8).
    if ii is None or bb is None:
        raise TypeError(
            "g_coupled_diagonal: carrying (System-B) mesh required — the "
            "coupled metric has a ray block by definition."
        )
    V = np.asarray(sn.volumes, dtype=float).ravel()
    iw = np.zeros(int(ii.shape[0]))
    bw = np.zeros(int(bb.shape[0]))
    for p in ii.levels:
        for sign in (-1, +1):
            ii.slot_view(iw, p, sign)[:] = V[None, :]
            bb.slot_view(bw, p, sign)[:] = V[-1]

    g = np.concatenate([bulk, trace, iw, bw])
    if not np.all(g > 0.0):
        raise ValueError(
            "g_coupled_diagonal: the assembled metric must be SPD "
            "(strictly positive) — a zero slot means a face/slot the "
            "fill loops did not cover (layout drift) or a ghost metric."
        )
    return g


def energy_spectrum(sol) -> np.ndarray:
    r"""L2-normalised per-group spatial-MEAN spectrum of a Solution's scalar flux.

    The one spelling of the spectrum-extraction convention the adjoint
    batteries compare against closed-form eigenvectors: group axis first,
    unweighted spatial mean over every remaining axis, ℓ² normalisation
    (matching :func:`orpheus.numerics.eigenvalue.dominant_eigenpair`'s
    convention).  On a spatially-flat iterate the mean is exact; on a
    structured one it is the flat-weight projection — the batteries only
    apply it where flatness holds (∞-medium legs) or where the reference
    shares the same reduction.
    """
    sf = np.asarray(sol.scalar_flux.values)
    spec = sf.mean(axis=tuple(range(1, sf.ndim)))
    return spec / np.linalg.norm(spec)


# ══════════════════════════════════════════════════════════════════════
# Issue #326 — the per-level ordinate ORDERING as a controllable input
# ══════════════════════════════════════════════════════════════════════
#
# ``rules_product.py`` orders each mu-level by ``np.argsort(mu_x)``.  The key
# ``eta = mu_x = sin(theta) cos(phi)`` is 2-to-1 over ``phi in [0, 2pi)``, so
# the azimuthal mirror pair ``(phi, 2pi - phi)`` ties and the level is NOT
# totally ordered; the tie-break is a free input that measurably moves the
# cylindrical answer (issue #326).  Three test modules need to VARY that input
# with node VALUES held bit-identical, so the swap lives here once rather than
# being re-spelled in each (Pattern 2 — a twin here would be a convention that
# can drift between the gates that compare against each other).

#: The per-level orderings #326 adjudicates.  ``None`` is production (no patch).
PRODUCT_LEVEL_ORDERINGS = ("lexsort", "stable", "azimuthal")

_PRODUCT_RULE_IMPORT_SITES = (
    "orpheus.numerics.quadrature.rules_product",
    "orpheus.numerics.quadrature.directional",
)


def _product_rule_with_ordering(tie_break: str, *, exact_nodes: bool):
    """Build a ``product_mu_phi`` replacement with a chosen level ordering.

    ``exact_nodes`` selects the node generator, and it MATTERS: with
    trig-evaluated ``np.cos(np.linspace(...))`` nodes the mirror pair's ``eta``
    differ by ~1 ULP, so the "tie" is resolved by ROUNDING NOISE before any
    tie-break rule can act and lexsort / stable / quicksort all agree.  The
    tie-break only becomes a reachable free variable once the nodes are
    algebraically exact (``roots_of_unity``, issue #325).

    ⚠ ``exact_nodes=False`` is now the HISTORICAL arm, not production.  As of
    2026-08-02 ``rules_product`` builds its azimuths with ``periodic_trapezoid``
    (roots of unity), so ``exact_nodes=True`` — the default here — is what
    production does.  Keep the False arm: it is what makes the noise-resolved
    regime measurable rather than merely asserted.
    """
    from orpheus.numerics.exactness import UNIFORM_ON_SPHERE, ExactnessClaim
    from orpheus.numerics.manifold import SPHERE
    from orpheus.numerics.measure import DiscreteMeasure
    from orpheus.numerics.quadrature.rules_sphere import (
        LevelStructure,
        PolarInvariant,
    )
    from orpheus.numerics.symmetry import SubgroupOfO3

    if tie_break not in PRODUCT_LEVEL_ORDERINGS:
        raise ValueError(
            f"unknown ordering {tie_break!r}; expected one of "
            f"{PRODUCT_LEVEL_ORDERINGS}"
        )

    def build(n_mu: int, n_phi: int):
        mu_gl, w_gl = np.polynomial.legendre.leggauss(n_mu)
        if exact_nodes:
            from orpheus.numerics.roots_of_unity import roots_of_unity
            cos_phi, sin_phi = roots_of_unity(np.arange(n_phi), n_phi)
        else:
            phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
            cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        w_phi = 2.0 * np.pi / n_phi

        n_total = n_mu * n_phi
        eta, xi, mu_z, w = (np.empty(n_total) for _ in range(4))
        level_indices: list[np.ndarray] = []
        i = 0
        for p in range(n_mu):
            sin_theta = np.sqrt(1.0 - mu_gl[p] ** 2)
            first = i
            for m in range(n_phi):
                eta[i], xi[i] = sin_theta * cos_phi[m], sin_theta * sin_phi[m]
                mu_z[i], w[i] = mu_gl[p], w_gl[p] * w_phi
                i += 1
            level = np.arange(first, first + n_phi)
            if tie_break == "lexsort":       # eta ascending, ties broken by xi
                order = np.lexsort((xi[level], eta[level]))
            elif tie_break == "stable":      # eta ascending, ties by phi order
                order = np.argsort(eta[level], kind="stable")
            else:                            # "azimuthal": omega increasing
                order = np.arange(n_phi)
            level_indices.append(level[order])

        measure = DiscreteMeasure(
            nodes=np.column_stack([eta, xi, mu_z]), weights=w,
            # The product rule's TRUE symmetry is the finite D_{n_phi h}
            # (tests/numerics/test_symmetry.py pins it); this field read the
            # bare `SO2` tag until 2026-09-01, when SO2 became axis-
            # parameterised and the false constant lost its spelling.
            support=SPHERE, invariance_group=SubgroupOfO3.Dnh(n_phi),
            exactness=ExactnessClaim(
                reference=UNIFORM_ON_SPHERE,
                degree=min(2 * n_mu - 1, n_phi - 1),
            ),
        )
        structure = LevelStructure(
            n_levels=n_mu, level_indices=level_indices, level_mu=mu_gl,
            polar_invariant=PolarInvariant.SIGNED_MU_Z,
            azimuth=np.mod(np.arctan2(xi, eta), 2.0 * np.pi),
            hemisphere=np.sign(mu_z).astype(np.int64),
        )
        return measure, structure

    return build


@contextmanager
def product_level_ordering(tie_break: str, *, exact_nodes: bool = True):
    """Run the body with ``Quadrature.product``'s level ordering replaced.

    In-process monkeypatch of ``product_mu_phi`` at every import site — no
    tracked file is touched.  Build the ``Quadrature`` (and anything holding
    one, e.g. an MMS case) INSIDE the block; objects constructed outside keep
    the ordering they were born with.

    Parameters
    ----------
    tie_break :
        One of :data:`PRODUCT_LEVEL_ORDERINGS`.
    exact_nodes :
        ``True`` (default) uses the algebraically-exact ``roots_of_unity``
        generator, which is what makes the tie-break reachable at all.
    """
    import importlib

    replacement = _product_rule_with_ordering(tie_break, exact_nodes=exact_nodes)
    saved = []
    try:
        for name in _PRODUCT_RULE_IMPORT_SITES:
            module = importlib.import_module(name)
            saved.append((module, module.product_mu_phi))
            module.product_mu_phi = replacement
        yield
    finally:
        for module, original in saved:
            module.product_mu_phi = original


def seam_quad(n_mu: int, n_phi: int, shift, *, folded: bool):
    """A quad-like over the seam's ``(measure, structure)``, optionally σ_y-folded.

    Builds ``spherical_product(gauss_legendre_on_mu(n_mu),
    periodic_trapezoid(n_phi, shift=shift))`` and exposes the attribute
    surface the pole-closure producers read (``mu_x``/``mu_y``/``mu_z``,
    ``weights``, ``level_indices``).  With ``folded=True`` the measure is
    quotiented by ``Mirror("y")`` and the level structure descends by
    selection (Q5.1/Q5.3) — each level is then an ARC in march order
    (T22b).  Shared by the march-start classification gates and the τ
    arc-well-posedness gates.
    """
    from types import SimpleNamespace

    from orpheus.numerics.quadrature import (
        gauss_legendre_on_mu,
        periodic_trapezoid,
        spherical_product,
    )
    from orpheus.numerics.symmetry import SubgroupOfO3

    measure, structure = spherical_product(
        gauss_legendre_on_mu(n_mu), periodic_trapezoid(n_phi, shift=shift)
    )
    if folded:
        quotient = measure.quotient(SubgroupOfO3.Mirror("y"))
        structure = structure.quotient(parent=measure, onto=quotient)
        measure = quotient
    return SimpleNamespace(
        mu_x=measure.nodes[:, 0],
        mu_y=measure.nodes[:, 1],
        mu_z=measure.nodes[:, 2],
        weights=measure.weights,
        level_indices=structure.level_indices,
    )


def rc_march(sn_mesh, total_cross_section):
    """Assemble A_BB from a carrying mesh — the un-weld arc's assembly read,
    spelled ONCE for tests (mirrors ``build_within_group_system``'s spelling;
    the operator itself binds spaces + values, never the mesh)."""
    from orpheus.sn.operators.radial_characteristic import (
        RadialCharacteristicOperator,
        march_start_cosines,
    )

    reduced = sn_mesh.reduced
    assert reduced is not None  # carrying fixture; narrowing only
    return RadialCharacteristicOperator(
        sn_mesh.radial_characteristic_field_space,
        total_cross_section,
        bulk_space=sn_mesh.bulk_space,
        dr=sn_mesh.axis_widths[0],
        start_cosines=march_start_cosines(
            reduced, sn_mesh.radial_characteristic_levels,
        ),
    )


def reflect_outflow_into_inflow(boundary_flux, sn_mesh: "SNMesh") -> None:
    r"""In-place: fill each face's inflow ordinate slots with the realized
    boundary law applied to that face's outflow trace — the ``−B`` reflective
    coupling, externalised for a BARE sweep (Wave O #208 O.4a.2).

    **A test-tree helper since #448.** The bare sweep reads the inflow
    ordinate slots of its boundary buffer as the inflow seed and does not
    re-apply ``bc`` to the outflow internally; PRODUCTION delivers the
    coupling as the ``B`` gain of every within-group driver (Wave O O.2a —
    since B.2d the :func:`~orpheus.sn.coupled_system.build_within_group_system`
    record's ``explicit_gains``), and since #448 the eigenvalue finalize is
    one :func:`~orpheus.numerics.iteration.fixed_point_step` of that same
    map, so no production path sets inflow slots by hand any more.  The
    sweep-tier gates that drive ``sweep_once`` / ``_sweep_jacobi`` in a loop
    (the 2-D octant equivalence suite, the curvilinear sweep regressions,
    the ng=2 layout guard, the iteration primitive's SN fixture) still need
    the inter-sweep ``ψ.inflow = B·ψ.outflow``, which is what this helper is:
    spelled on production's live reflect (CS4c step 6 item 6.5): the Jacobi
    split's ``upper`` half IS the full-inflow mask (`[M]` 4/4 geometries —
    ``SweepSchedule.jacobi`` reflects no face in-sweep, so ``lower`` is empty
    and ``upper`` carries every inflow row of every face), and *zero the
    inflow rows, then*
    :meth:`~orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace`
    (ADDITIVE on the mask's rows) reproduces the whole-face ASSIGNMENT
    ``ψ.inflow ← B·ψ.outflow`` bit-for-bit (`[M]` ``array_equal`` on 4/4
    geometries on a non-zero-inflow buffer; dropping the zeroing moves the
    answer by 1.1–2.6 — the assignment/additive difference is real). The
    retired assignment verb ``reflect_inflow_inplace`` was this helper's
    last consumer. ``B_a`` is the SAME core the matvec / SI driver consume
    as the boundary gain, so the two routes cannot drift.  `[M]` its last
    production call site — the pre-#448 finalize's reflect of the converged
    trace — was inert on a converged exit (2.0e-13 / 2.3e-15 / bit-identical
    on a vacuum arm), which is why moving it cost no gate.

    For vacuum ``B = 0`` so the inflow slots stay zero; for
    reflective/white/albedo it is the same ``R·G`` reflection the
    pre-extraction sweep applied at entry, relocated to the caller.  The
    ψ½ ray corner (System B's ``B_b``) is NOT wrapped here — `[M]` 0 of the
    14 consumer sites ever passed a ray, and its in-place verb
    ``reflect_corner_inplace`` retired at #448 with the finalize that was its
    only caller; on a carrying mesh the corner is the coupled gain grid's
    business.  The helper carries exactly the arm the gates drive.
    """
    from orpheus.sn.loss_representation.sweep_schedule import SweepSchedule
    from orpheus.sn.operators.boundary import SNBoundaryOperator

    operator = SNBoundaryOperator(sn_mesh)
    full_inflow = operator.split(
        SweepSchedule.jacobi(sn_mesh.ndim, sn_mesh.quad.octants),
    ).upper
    trace = sn_mesh.angular_trace
    faces = tuple(boundary_flux.layout.faces)
    for face in faces:
        boundary_flux.face_view(face)[trace.inflow_indices_for_face(face)] = 0.0
    full_inflow.reflect_rows_inplace(boundary_flux, faces)
