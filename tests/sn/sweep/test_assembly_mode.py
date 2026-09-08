r"""The SN assembly mode — per-ordinate blocks vs the production sweep + matvec.

Stencil-assembly campaign 2b (the L16 gate spec):

* **G1** ``assembled @ x ≡ apply(x)`` per ordinate × group × geometry —
  the object-level matvec equivalence (never a scalar functional, Mode
  12), on het + non-uniform-h + ≥2G fixtures with non-flat seeded x.
* **G2** ``scipy.linalg.solve_triangular(PᵀMP, Pᵀq) ≡ (L+C).solve(q)``
  — LAPACK ``dtrtrs`` is a structurally-INDEPENDENT forward
  substitution vs the ORPHEUS sweep recurrence, which EARNS this gate
  its L2 cross-check status and **discharges #284's sweep-inverse
  contract question at the object level**: on the source subspace
  (bulk source, zero trace — every production rhs today), the sweep IS
  forward substitution on the assembled walk-order-triangular matrix.
  The triangularity leg ``triu(PᵀMP, 1) == 0`` is a structural EXACT
  zero (the one 0-tolerance assertion in the family).
* **One-source teeth**: a sign flip monkeypatched into the SHARED
  Cartesian coefficient source (``_cartesian_streaming_diagonal`` —
  the fold ``cell_kernel_batch`` / ``residual_kernel_batch`` /
  ``cartesian_scan_coefficients`` all consume) must move the sweep,
  the matvec, AND the assembly TOGETHER: both absolute values leave
  their baselines O(1) while the cross-mode equivalences PERSIST —
  divergence under a shared-source mutation is precisely the twin-path
  signature (the assembler owns no stencil spelling: it extracts every
  coefficient by unit probes of the production ``residual_kernel_batch``).
  The tooth lives on the 2-D fixture, where the production SOLVE
  (ScanMarch → ``cartesian_scan_coefficients``) rides the same fold as
  the matvec; the 1-D solve rides the ×V ``affine_scan_coefficients``
  form instead — the documented #242 dual-form seam, NOT a twin (the
  two are the same equation in different volume conventions, pinned
  against each other by the existing sweep suites and by G2 here).
* **#282 route (a)** (curvilinear assembly is OUT of 2b — blocked on
  exactly this): the spherical within-group operator, probed from the
  PRODUCTION matvec, HAD a walk-order back edge — the Morel–Montry
  half-angle seed row read later-ordinate columns — and this gate
  asserted that defect positively until route (a) (2.5d d3) made the
  ψ½ seed block first-class STATE; it then flipped RED as designed
  (L16) and was rewritten as today's augmented-triangularity
  certificate. ⛔ The cylinder row was documented as "the non-carrying
  (R12a) control" — TRUE at birth (2026-07-04, a ``product(4,8)``
  rule) and FALSE since ``384d62e4`` (6.3 leg 2b) swapped the fixture
  to the always-carrying fold; `[M]` 2026-08-29 (§5b P0) it binds the
  field space with all 4 levels carrying, i.e. it is a SECOND carrying
  case. The empty-seed arm's constructible witness is a Gauss-Lobatto
  SPHERE rule — #415. (The α-dome "telescoping the seed away"
  misreading stays corrected per #280 Phase 2.5b; it was never why the
  cylinder is triangular.)

The degenerate all-zero octant branch (pure-z ordinates over a lower-D
mesh — the ``Q/Σ_t`` diagonal) has no fixture here: no shipped
quadrature factory produces an exactly-grazing ordinate over these
meshes; the branch is exercised when a product-cubature consumer
arrives.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve_triangular

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import BC, CoordSystem, Mesh2D
from orpheus.geometry.mesh import Mesh1D
from orpheus.transport.spatial.linear_discontinuous import LinearDiscontinuous
from orpheus.numerics.operator import MissingAssembly
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.loss_representation.assembly import (
    assemble_ordinate_blocks,
    ordinate_walk_order,
)
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.operators.streaming import StreamingOperator
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.radial_characteristic_field import RadialCharacteristicField
from orpheus.transport.full_field import FullField
from orpheus.transport.operators.multiplication_operator import (
    MultiplicationOperator,
)
from orpheus.transport.source_sinks import (
    AngularBoundarySourceSink,
    AngularSourceSink,
)

_RTOL = 1e-11   # L16: sparse-order ≠ apply-order ⇒ never 0-ULP (measured ~6e-16)


# ── Fixtures: het, non-uniform h, ≥2G, vacuum (zero-inflow posing) ─────


def _slab_mesh() -> SNMesh:
    mesh1d = Mesh1D(
        edges=np.array([0.0, 0.5, 1.5, 3.0, 5.0]),       # non-uniform
        mat_ids=np.array([0, 1, 1, 0]),                  # heterogeneous
        bc_left=BC("vacuum"), bc_right=BC("vacuum"),
    )
    quad = Quadrature.gauss_legendre(n_ordinates=4)
    return SNMesh(
        mesh1d, quad, {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")},
    )


def _cartesian_2d_mesh() -> SNMesh:
    geom = Mesh2D(
        edges_x=np.array([0.0, 0.4, 1.1, 2.1, 3.0]),     # non-uniform
        edges_y=np.array([0.0, 0.7, 1.5, 2.0]),
        mat_map=np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0]]),
        bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("vacuum"), bc_ymax=BC("vacuum"),
    )
    quad = Quadrature.level_symmetric(sn_order=4)
    return SNMesh(
        geom, quad, {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")},
    )


_MESH_BUILDERS = {"slab": _slab_mesh, "cartesian_2d": _cartesian_2d_mesh}


def _loss(sn_mesh: SNMesh):
    """The production within-group resolvent ``L + C`` (the solver's own
    spelling — StreamingOperator + M[σ_t] on the composite space)."""
    mat_xs = sn_mesh.material_xs_field()
    return StreamingOperator.pose(sn_mesh) + MultiplicationOperator(
        coefficient=mat_xs.total_cross_section_field,
        domain=sn_mesh.full_field_space, codomain=sn_mesh.full_field_space,
    )


def _bulk_impulse_state(sn_mesh: SNMesh, n: int, g: int, x: np.ndarray):
    """A composite that is ``x`` on bulk row (n, g) and zero elsewhere."""
    state = FullField.zeros(
        interior=AngularFlux, boundary=AngularBoundaryFlux, space=sn_mesh.full_field_space,
    )
    state.interior.values[n, g] = x.reshape(sn_mesh.spatial_shape)
    return state


def _bulk_source(sn_mesh: SNMesh, n: int, g: int, q: np.ndarray):
    """A source composite with ``q`` on bulk row (n, g), zero trace —
    the #284 source subspace the sweep inverts exactly."""
    rhs = FullField(
        interior=AngularSourceSink.zeros(sn_mesh.angular_bulk_space),
        boundary=AngularBoundarySourceSink.zeros(sn_mesh.angular_trace),
    )
    rhs.interior.values[n, g] = q.reshape(sn_mesh.spatial_shape)
    return rhs


# ── G1 + block-diagonality ─────────────────────────────────────────────


@pytest.mark.l0
@pytest.mark.verifies("loss-rep-affine-kernel-maps")
@pytest.mark.parametrize("geometry", list(_MESH_BUILDERS))
def test_g1_assembled_matvec_equals_apply(geometry):
    """G1: per (ordinate, group), ``M @ x`` reproduces the production
    matvec's bulk row — and the Cartesian ``L + C`` is exactly
    per-ordinate-per-group block-diagonal (other rows untouched)."""
    sn_mesh = _MESH_BUILDERS[geometry]()
    A = _loss(sn_mesh)
    n_cells = int(np.prod(sn_mesh.spatial_shape))
    ng = 2
    rng = np.random.default_rng(20260704)
    for n in range(sn_mesh.quad.n_ordinates):
        blocks = assemble_ordinate_blocks(sn_mesh, n)
        for g in range(ng):
            x = rng.random(n_cells) + 0.5              # non-flat, positive
            out = A.apply(_bulk_impulse_state(sn_mesh, n, g, x))
            bulk_out = np.asarray(out.interior.values)
            np.testing.assert_allclose(
                blocks[g].apply(x), bulk_out[n, g].ravel(),
                rtol=_RTOL, atol=1e-14,
                err_msg=f"{geometry}: G1 broke at ordinate {n}, group {g}",
            )
            rest = bulk_out.copy()
            rest[n, g] = 0.0
            np.testing.assert_array_equal(
                rest, 0.0,
                err_msg=f"{geometry}: (n={n}, g={g}) leaked off-block",
            )


# ── G2: triangularity (exact) + LAPACK ≡ sweep (the #284 discharge) ───


@pytest.mark.l0
@pytest.mark.verifies("loss-rep-walk-order-rows")
@pytest.mark.parametrize("geometry", list(_MESH_BUILDERS))
def test_g2_walk_order_triangularity_is_exact(geometry):
    """``triu(PᵀMP, 1) == 0`` EXACTLY — the walk order is a certificate
    of triangularity, structurally (no tolerance: an entry is either
    emitted above the diagonal or it is not)."""
    sn_mesh = _MESH_BUILDERS[geometry]()
    for n in range(sn_mesh.quad.n_ordinates):
        order = ordinate_walk_order(sn_mesh, n)
        # A true permutation of the bulk cells…
        np.testing.assert_array_equal(np.sort(order), np.arange(order.size))
        for g, block in enumerate(assemble_ordinate_blocks(sn_mesh, n)):
            M = block.as_matrix()
            permuted = M[np.ix_(order, order)]
            np.testing.assert_array_equal(
                np.triu(permuted, k=1), 0.0,
                err_msg=(
                    f"{geometry}: ordinate {n} group {g} emitted above "
                    f"the walk-order diagonal"
                ),
            )


@pytest.mark.l2
@pytest.mark.verifies("loss-rep-walk-order-rows")
@pytest.mark.parametrize("geometry", list(_MESH_BUILDERS))
def test_g2_lapack_forward_substitution_equals_sweep(geometry):
    """The #284 discharge, object-level: on the source subspace the
    production sweep ``(L+C).solve`` IS forward substitution on the
    assembled walk-order-triangular matrix — verified against LAPACK's
    ``dtrtrs`` (scipy ``solve_triangular``), a structurally-independent
    realization of the same substitution (L2 cross-check status)."""
    sn_mesh = _MESH_BUILDERS[geometry]()
    A = _loss(sn_mesh)
    n_cells = int(np.prod(sn_mesh.spatial_shape))
    ng = 2
    rng = np.random.default_rng(20260705)
    for n in range(sn_mesh.quad.n_ordinates):
        blocks = assemble_ordinate_blocks(sn_mesh, n)
        order = ordinate_walk_order(sn_mesh, n)
        for g in range(ng):
            q = rng.random(n_cells) + 0.5
            psi = A.solve(_bulk_source(sn_mesh, n, g, q))
            psi_row = np.asarray(psi.interior.values)[n, g].ravel()
            permuted = blocks[g].as_matrix()[np.ix_(order, order)]
            via_lapack = solve_triangular(permuted, q[order], lower=True)
            np.testing.assert_allclose(
                via_lapack, psi_row[order],
                rtol=_RTOL, atol=1e-14,
                err_msg=(
                    f"{geometry}: sweep ≠ LAPACK forward substitution at "
                    f"ordinate {n}, group {g} (#284)"
                ),
            )


@pytest.mark.l0
def test_g3_dd_slab_probed_column_pin():
    """The SN family's ONE probed≡assembled oracle (L16: the diffusion
    loss + one DD slab block): every column of the assembled slab block
    equals the production apply on the corresponding basis field — the
    exhaustive object pin, kept permanently as the fuller-view oracle
    (probing goes through apply's arithmetic; emission through the
    kernel probes — structurally distinct consumptions of one kernel)."""
    sn_mesh = _slab_mesh()
    A = _loss(sn_mesh)
    n_cells = int(np.prod(sn_mesh.spatial_shape))
    n, g = 1, 1                                    # one block suffices
    M = assemble_ordinate_blocks(sn_mesh, n)[g].as_matrix()
    probed = np.empty_like(M)
    for j in range(n_cells):
        basis = np.zeros(n_cells)
        basis[j] = 1.0
        out = A.apply(_bulk_impulse_state(sn_mesh, n, g, basis))
        probed[:, j] = np.asarray(out.interior.values)[n, g].ravel()
    np.testing.assert_allclose(M, probed, rtol=_RTOL, atol=1e-15)


@pytest.mark.foundation
def test_streaming_only_block_differs_by_the_collision_diagonal():
    """``include_collision=False`` emits the pure-L block: the L+C and
    L blocks differ by exactly the collision diagonal diag(σ_t) (to
    fold rounding — the two denominators are separate left folds)."""
    sn_mesh = _slab_mesh()
    sigma_t = np.asarray(
        sn_mesh.material_xs_field().total_cross_section_field.values
    )
    for n in (0, 3):
        with_c = assemble_ordinate_blocks(sn_mesh, n)
        without_c = assemble_ordinate_blocks(
            sn_mesh, n, include_collision=False,
        )
        for g in range(2):
            delta = with_c[g].as_matrix() - without_c[g].as_matrix()
            np.testing.assert_allclose(
                delta, np.diag(sigma_t[g]), rtol=1e-13, atol=1e-15,
            )


# ── One-source teeth (the campaign's whole point) ─────────────────────


@pytest.mark.foundation
def test_teeth_shared_kernel_sign_flip_moves_all_three_modes(monkeypatch):
    """Flip the diamond couplings in the ONE Cartesian fold
    (``_cartesian_streaming_diagonal``): sweep, matvec, and assembly
    must all leave their baselines O(1) — TOGETHER — while G1 and G2
    persist. If assembly (or the sweep) stayed at its baseline, a twin
    coefficient spelling would exist → stop, fix, catalog (L16)."""
    from orpheus.transport.spatial.diamond import DiamondDifference

    sn_mesh = _cartesian_2d_mesh()
    n, g = 5, 1
    n_cells = int(np.prod(sn_mesh.spatial_shape))
    rng = np.random.default_rng(3)
    x = rng.random(n_cells) + 0.5
    q = rng.random(n_cells) + 0.5

    def snapshot():
        A = _loss(sn_mesh)
        M = assemble_ordinate_blocks(sn_mesh, n)[g].as_matrix()
        apply_row = np.asarray(
            A.apply(_bulk_impulse_state(sn_mesh, n, g, x)).interior.values
        )[n, g].ravel()
        sweep_row = np.asarray(
            A.solve(_bulk_source(sn_mesh, n, g, q)).interior.values
        )[n, g].ravel()
        return M, apply_row, sweep_row

    baseline_M, baseline_apply, baseline_sweep = snapshot()

    original = DiamondDifference._cartesian_streaming_diagonal

    def flipped(reaction_xs, s_axes):
        denom, couplings = original(
            reaction_xs, tuple(-s for s in s_axes),
        )
        return denom, couplings

    monkeypatch.setattr(
        DiamondDifference,
        "_cartesian_streaming_diagonal",
        staticmethod(flipped),
    )

    mutated_M, mutated_apply, mutated_sweep = snapshot()

    scale = float(np.abs(baseline_M).max())
    assert np.abs(mutated_M - baseline_M).max() > 1e-2 * scale, (
        "assembly blind to the shared-fold flip — twin stencil"
    )
    assert (
        np.abs(mutated_apply - baseline_apply).max()
        > 1e-2 * np.abs(baseline_apply).max()
    ), "matvec blind to the shared-fold flip"
    assert (
        np.abs(mutated_sweep - baseline_sweep).max()
        > 1e-2 * np.abs(baseline_sweep).max()
    ), "sweep blind to the shared-fold flip — the fold is not its source"
    # The one-source signature: the equivalences PERSIST under mutation.
    np.testing.assert_allclose(
        mutated_M @ x, mutated_apply, rtol=_RTOL, atol=1e-14,
        err_msg="G1 broke under the shared mutation — twin path",
    )
    order = ordinate_walk_order(sn_mesh, n)
    permuted = mutated_M[np.ix_(order, order)]
    np.testing.assert_allclose(
        solve_triangular(permuted, q[order], lower=True),
        mutated_sweep[order],
        rtol=_RTOL, atol=1e-14,
        err_msg="G2 broke under the shared mutation — twin path",
    )


# ── Scope guards ───────────────────────────────────────────────────────


def _ld_mesh(geometry: str) -> SNMesh:
    """LD fixtures — het, non-uniform h, ≥2G (the DD fixtures' twins,
    with the bilinear closure selected)."""
    materials = {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")}
    if geometry == "slab":
        mesh1d = Mesh1D(
            edges=np.array([0.0, 0.5, 1.5, 3.0, 5.0]),
            mat_ids=np.array([0, 1, 1, 0]),
            bc_left=BC("vacuum"), bc_right=BC("vacuum"),
        )
        quad = Quadrature.gauss_legendre(n_ordinates=4)
        return SNMesh(mesh1d, quad, materials, scheme=LinearDiscontinuous())
    geom = Mesh2D(
        edges_x=np.array([0.0, 0.4, 1.1, 2.1, 3.0]),
        edges_y=np.array([0.0, 0.7, 1.5, 2.0]),
        mat_map=np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0]]),
        bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("vacuum"), bc_ymax=BC("vacuum"),
    )
    quad = Quadrature.level_symmetric(sn_order=4)
    return SNMesh(geom, quad, materials, scheme=LinearDiscontinuous())


def _block_upper_mask(n_cells: int, cm: int, order: np.ndarray) -> np.ndarray:
    """Boolean mask of entries ABOVE the cell-block diagonal after the
    walk-order block permutation (``kron(triu(1,1), ones(cm×cm))``)."""
    upper_cells = np.triu(np.ones((n_cells, n_cells), dtype=bool), k=1)
    return np.kron(upper_cells, np.ones((cm, cm), dtype=bool))


def _dof_order(order: np.ndarray, cm: int) -> np.ndarray:
    """Expand a cell walk order to the DOF layout (cell-major,
    moment-minor — the bulk C-ravel of ``(*spatial, cm)``)."""
    return (order[:, None] * cm + np.arange(cm)[None, :]).ravel()


@pytest.mark.l0
@pytest.mark.verifies("loss-rep-sweep-global-conjugation")
@pytest.mark.parametrize("geometry", ["slab", "cartesian_2d"])
def test_g1_ld_assembled_matvec_equals_apply(geometry):
    """G1 for the bilinear closure: the block walk emits LD's UBLD
    coefficients (extracted through LD's OWN residual kernel) and the
    assembled matvec reproduces the production apply on the
    moment-valued bulk row."""
    sn_mesh = _ld_mesh(geometry)
    A = _loss(sn_mesh)
    cm = 2 ** len(sn_mesh.spatial_shape)
    n_cells = int(np.prod(sn_mesh.spatial_shape))
    rng = np.random.default_rng(20260706)
    ng = 2
    N = sn_mesh.quad.n_ordinates
    for n in range(0, N, 3):                          # a representative stride
        blocks = assemble_ordinate_blocks(sn_mesh, n)
        for g in range(ng):
            x = rng.random(n_cells * cm) + 0.5
            # A MOMENT-VALUED iterate (the production spelling — the
            # spatial_moments factor selects the scheme's moment axis so
            # the bilinear closure's trailing 2^d axis is carried).
            state = FullField(
                interior=AngularFlux.zeros(sn_mesh.angular_trial_space),
                boundary=AngularBoundaryFlux.zeros(sn_mesh.angular_trace),
            )
            state.interior.values[n, g] = x.reshape(
                sn_mesh.spatial_shape + (cm,)
            )
            out = A.apply(state)
            np.testing.assert_allclose(
                blocks[g].apply(x),
                np.asarray(out.interior.values)[n, g].ravel(),
                rtol=_RTOL, atol=1e-14,
                err_msg=f"LD {geometry}: G1 broke at ordinate {n}, group {g}",
            )


@pytest.mark.l2
@pytest.mark.parametrize("geometry", ["slab", "cartesian_2d"])
def test_g2_ld_block_triangular_and_lapack_solve_equals_sweep(geometry):
    """G2 for the bilinear closure: the assembled block is
    BLOCK-lower-triangular in walk order (exact — the moment blocks are
    dense within a cell, so the structural zero lives ABOVE the cell
    blocks), and LAPACK's dense LU solve of the assembled matrix
    reproduces the production sweep on a flat source (the #284
    discharge's LD arm — the sweep inverts the assembled object
    exactly, verified through a structurally-independent solver)."""
    sn_mesh = _ld_mesh(geometry)
    A = _loss(sn_mesh)
    d = len(sn_mesh.spatial_shape)
    cm = 2 ** d
    n_cells = int(np.prod(sn_mesh.spatial_shape))
    rng = np.random.default_rng(20260707)
    from orpheus.numerics.moment_layout import AVERAGE_MOMENT
    from scipy.linalg import lu_factor, lu_solve

    for n in (0, sn_mesh.quad.n_ordinates - 1):
        blocks = assemble_ordinate_blocks(sn_mesh, n)
        order = ordinate_walk_order(sn_mesh, n)
        dof_order = _dof_order(order, cm)
        upper = _block_upper_mask(n_cells, cm, order)
        for g in range(2):
            M = blocks[g].as_matrix()
            permuted = M[np.ix_(dof_order, dof_order)]
            np.testing.assert_array_equal(
                permuted[upper], 0.0,
                err_msg=(
                    f"LD {geometry}: ordinate {n} group {g} emitted above "
                    f"the walk-order cell-block diagonal"
                ),
            )
            # Moment source on (n, g): the sweep's ψ must satisfy the
            # assembled system M ψ = q_lifted (average row = q, slope
            # rows = 0 — the raw per-moment matvec convention).
            q = rng.random(n_cells) + 0.5
            src_values = np.zeros(
                (sn_mesh.quad.n_ordinates, 2) + sn_mesh.spatial_shape + (cm,)
            )
            src_values[n, g, ..., AVERAGE_MOMENT] = q.reshape(
                sn_mesh.spatial_shape
            )
            rhs = FullField(
                interior=AngularSourceSink(
                    values=src_values, space=sn_mesh.angular_trial_space,
                ),
                boundary=AngularBoundarySourceSink.zeros(sn_mesh.angular_trace),
            )
            psi = A.solve(rhs)
            psi_row = np.asarray(psi.interior.values)[n, g].reshape(-1)
            q_lifted = np.zeros(n_cells * cm)
            q_lifted[AVERAGE_MOMENT::cm] = q
            via_lapack = lu_solve(lu_factor(M), q_lifted)
            np.testing.assert_allclose(
                via_lapack, psi_row,
                rtol=1e-10, atol=1e-13,
                err_msg=(
                    f"LD {geometry}: sweep ≠ LAPACK solve of the assembled "
                    f"block at ordinate {n}, group {g} (#284 LD arm)"
                ),
            )


@pytest.mark.foundation
def test_curvilinear_refuses_the_cartesian_walk():
    """Curvilinear streaming lives on the chain-scan substrate — the
    Cartesian accessor's own gate refuses, and this assembler inherits
    that honesty (the per-ordinate factorization does not exist there;
    see the #282 characterization below for WHY)."""
    mesh1d = Mesh1D(
        edges=np.linspace(0.0, 1.0, 5),
        mat_ids=np.zeros(4, dtype=int),
        bc_left=BC("reflective"), bc_right=BC("vacuum"),
        coord=CoordSystem.SPHERICAL,
    )
    quad = Quadrature.gauss_legendre(n_ordinates=4)
    sn_mesh = SNMesh(mesh1d, quad, {0: get_mixture("A", "2g")})
    with pytest.raises(AttributeError, match="Cartesian-only"):
        assemble_ordinate_blocks(sn_mesh, 0)


# ── #282 route (a): the AUGMENTED walk-order triangularity certificate ──
#
# The #280 Phase 2.5d fix retired the lagged Morel–Montry ψ½ pole seed
# (a walk-order BACK edge that made the spherical operator non-triangular
# in sweep order) and replaced it with a first-class ψ½ STATE block whose
# rows the augmented (L+C) emits.  The pre-fix characterization asserted
# the spherical back edge EXISTED (`above > 0`); route (a) makes the
# augmented one-group matrix EXACTLY block-lower-triangular in the
# augmented sweep order — the transpose analog of the #284 discharge and
# the acceptance evidence for #282 (L16 loud-flip: this replaces, not
# relaxes, the RED characterization).


def _probe_augmented_matrix_one_group(sn_mesh: SNMesh, g: int) -> np.ndarray:
    r"""The one-group AUGMENTED matrix of the production matvec by column
    probes — the ψ½ seed DOFs stacked BEFORE the ordinate DOFs.

    DOF order (rows == cols): per carrying level, the seed leg
    ``[corner_in(−1), cells⁻ (nx−1..0), cells⁺ (0..nx−1), corner_out(+1)]``
    (the seed's own march order), then the ``N·nx`` ordinate-bulk DOFs.
    On a NON-carrying mesh the seed block is empty and this reduces to
    the plain ordinate-bulk probe (the pre-fix
    ``_probe_bulk_matrix_one_group``). ⚠ That arm currently has NO
    witness in this gate — `[M]` 2026-08-29 (§5b P0): both parametrized
    charts are CARRYING (the cylinder stopped being non-carrying when
    ``384d62e4`` swapped its fixture to the fold), and since Q5.6.3
    every admitted cylinder rule is carrying. The
    constructible witness for this arm is a Gauss-Lobatto SPHERE rule —
    #415.
    """
    A = _loss(sn_mesh)
    N = sn_mesh.quad.n_ordinates
    nx = int(np.prod(sn_mesh.spatial_shape))
    levels = sn_mesh.radial_characteristic_levels
    carrying = sn_mesh.radial_characteristic_field_space is not None

    def _seed_leg_view(rows, level):
        # The emitted seed leg (a source composite) in march order, one group.
        return np.concatenate([
            [rows.boundary.corner(level, -1)[g]],
            rows.interior.cells(level, -1)[g][::-1],
            rows.interior.cells(level, +1)[g],
            [rows.boundary.corner(level, +1)[g]],
        ])

    def _read(out, rows) -> np.ndarray:
        bulk = np.asarray(out.interior.values)[:, g].ravel()   # (N·nx,)
        if not carrying:
            return bulk
        seed = np.concatenate([_seed_leg_view(rows, p) for p in levels])
        return np.concatenate([seed, bulk])

    def _fresh():
        # Scheme-aware bulk (LD carries the trailing 2^d moment axis; DD's
        # spatial_moments=1 is the byte-identical default).
        return FullField(
            interior=AngularFlux.zeros(sn_mesh.angular_trial_space),
            boundary=AngularBoundaryFlux.zeros(sn_mesh.angular_trace),
        )

    def _apply(st, seed_leaf):
        # step 6: the joint row action is THE GRID's block matvec —
        # systems[0] = LC·ψ_A + Seeding·ψ_B (the bulk), systems[1] =
        # A_BB·ψ_B (the self-contained emitted seed rows).
        if not carrying:
            return _read(A.apply(st), None)
        from orpheus.numerics.coupled_system import CoupledField

        from tests.sn._test_helpers import joint_m_grid

        grid, _space = joint_m_grid(sn_mesh, A)
        out = grid.apply(CoupledField(systems=(st, seed_leaf)))
        return _read(out.systems[0], out.systems[1])

    def _zero_seed():
        return (
            None if not carrying
            else RadialCharacteristicField.flux_zeros(sn_mesh.radial_characteristic_field_space)
        )

    n_seed_per_level = 2 * nx + 2
    columns: list[np.ndarray] = []
    # ── seed columns (per level, in the same march order as _read) ──
    for p in levels:
        for local in range(n_seed_per_level):
            st = _fresh()
            seed_leaf = _zero_seed()
            if local == 0:
                seed_leaf.boundary.corner(p, -1)[g] = 1.0
            elif local <= nx:
                seed_leaf.interior.cells(p, -1)[g][nx - local] = 1.0
            elif local <= 2 * nx:
                seed_leaf.interior.cells(p, +1)[g][local - nx - 1] = 1.0
            else:
                seed_leaf.boundary.corner(p, +1)[g] = 1.0
            columns.append(_apply(st, seed_leaf))
    # ── ordinate-bulk columns (generic over the spatial-moment tail:
    # probe every raveled per-group bulk DOF in the same C-order _read
    # ravels — N·nx for DD, N·nx·2^d for LD) ──
    probe_shape = _fresh().interior.values[:, g].shape
    for flat in range(int(np.prod(probe_shape))):
        st = _fresh()
        idx = np.unravel_index(flat, probe_shape)
        st.interior.values[(idx[0], g, *idx[1:])] = 1.0
        columns.append(_apply(st, _zero_seed()))
    return np.array(columns).T


def _augmented_sweep_order(sn_mesh: SNMesh) -> np.ndarray:
    """The augmented walk order: seed DOFs first (their march order, as
    stacked by the probe), then the ordinate-bulk DOFs in increasing μ
    (cells marching WITH each ordinate's direction — inward for μ<0)."""
    mu = np.asarray(sn_mesh.quad.mu_x)
    nx = int(np.prod(sn_mesh.spatial_shape))
    # The spatial-moment tail rides innermost in the probe's C-order ravel
    # (2^d for LD, 1 for DD — the DD order is byte-identical to the
    # pre-moment spelling); within a cell the moments stay contiguous, so
    # the walk order is block-wise with 2^d-wide cell blocks.
    tail = sn_mesh.scheme.spatial_basis_per_axis ** sn_mesh.ndim
    composite_space = sn_mesh.radial_characteristic_field_space
    n_seed = (
        0 if composite_space is None
        else composite_space.shape[0] // sn_mesh.ng
    )
    order: list[int] = list(range(n_seed))   # seed DOFs already march-ordered
    for n in np.argsort(mu, kind="stable"):
        cells = range(nx - 1, -1, -1) if mu[n] < 0.0 else range(nx)
        order.extend(
            n_seed + int(n) * nx * tail + i * tail + m
            for i in cells for m in range(tail)
        )
    return np.asarray(order, dtype=np.intp)


@pytest.mark.foundation
@pytest.mark.parametrize("coord", [
    CoordSystem.SPHERICAL,
    CoordSystem.CYLINDRICAL,
])
def test_282_augmented_walk_order_is_triangular(coord):
    r"""The #282 route-(a) acceptance certificate (L16 loud-flip of the
    former back-edge characterization): the one-group AUGMENTED (L+C)
    matrix, permuted to the augmented sweep order, is EXACTLY
    block-lower-triangular (``triu == 0``).

    * **sphere** — the FLIP: pre-fix the spherical operator had a
      walk-order back edge (the lagged ψ½ pole seed read later-ordinate
      columns); route (a) makes the ψ½ block first-class STATE, so the
      augmented matrix ``[[A_ss, 0], [A_bs, A_bb]]`` is triangular in
      ``[seed⁻ march, seed⁺ march, ordinates↑μ]`` order — a genuine
      forward-substitution certificate (the 2.5b LAPACK-≡-sweep leg
      builds on it).
    * **cylinder** — a SECOND carrying case, taking the SAME augmented
      path as the sphere: `[M]` 2026-08-29 (§5b P0) this fixture binds
      ``radial_characteristic_field_space`` with ``_carrying_levels ==
      [0, 1, 2, 3]``. ⛔ It was documented as "the non-carrying (R12a)
      control" — true for the ``product(4,8)`` rule it was born with,
      false since ``384d62e4`` (6.3 leg 2b) swapped the fixture to the
      fold. What this row adds over the sphere is the chart ×
      quadrature family (folded rule, degenerate pure-azimuthal
      ordinates), not the empty-seed branch — that branch's witness is
      a Gauss-Lobatto sphere rule, #415. (The α-dome "telescoping the
      seed away" misreading stays corrected per #280 Phase 2.5b.)
    """
    mesh1d = Mesh1D(
        edges=np.array([0.0, 0.3, 0.8, 1.0]),
        mat_ids=np.array([0, 1, 0]),
        bc_left=BC("reflective"), bc_right=BC("vacuum"),
        coord=coord,
    )
    quad = (
        Quadrature.gauss_legendre(n_ordinates=4)
        if coord is CoordSystem.SPHERICAL
        else Quadrature.folded_product(n_mu=4, n_phi=8)
    )
    sn_mesh = SNMesh(
        mesh1d, quad, {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")},
    )
    M = _probe_augmented_matrix_one_group(sn_mesh, g=0)
    order = _augmented_sweep_order(sn_mesh)
    permuted = M[np.ix_(order, order)]
    # np.testing (fires under -O — the g_adjoint discipline), not bare assert.
    np.testing.assert_array_equal(
        np.triu(permuted, k=1), 0.0,
        err_msg=(
            f"[{coord}] the augmented walk order is NOT lower-triangular — "
            "a back edge survives the #282 route-(a) seed carve "
            "(sphere: the ψ½ block leaked a later-column read; "
            "cylinder: the α-dome cancellation broke)."
        ),
    )


@pytest.mark.foundation
def test_282_teeth_coupling_direction_swap_reds():
    r"""§16.E teeth #1 — feed the LAST ordinate (not the μ=−1 starting
    direction) into the seed row → the triangularity leg REDS.

    The certificate has teeth: monkeypatch the closure's seed read to
    pull from an ordinate column that comes AFTER the seed in the walk
    order (a back edge), and confirm ``triu != 0``.  Reverts by
    in-process monkeypatch (never git checkout).
    """
    from orpheus.sn.angular.closure import MorelMontryAngularSweep

    mesh1d = Mesh1D(
        edges=np.array([0.0, 0.3, 0.8, 1.0]),
        mat_ids=np.array([0, 1, 0]),
        bc_left=BC("reflective"), bc_right=BC("vacuum"),
        coord=CoordSystem.SPHERICAL,
    )
    sn_mesh = SNMesh(
        mesh1d, Quadrature.gauss_legendre(n_ordinates=4),
        {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")},
    )
    # Mutant: the recurrence seed reads the LAST ordinate's cells (a
    # walk-order back edge) instead of the starting-direction STATE.
    orig = MorelMontryAngularSweep.precompute_psi_state

    def _mutant(self, psi_view, *, radial_characteristic=None):
        # Inject a bulk read: overwrite the given seed state with a slice
        # of psi_view's last ordinate (the back-edge coupling under test).
        if radial_characteristic is not None:
            import numpy as _np
            psi_g_first = psi_view.swapaxes(0, 1)
            for p in self._carrying_levels:
                cells = radial_characteristic.cells(p, -1)
                cells += psi_g_first[:, -1, :]   # + last-ordinate column
        return orig(self, psi_view, radial_characteristic=radial_characteristic)

    import pytest as _pytest
    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            MorelMontryAngularSweep, "precompute_psi_state", _mutant,
        )
        M = _probe_augmented_matrix_one_group(sn_mesh, g=0)
    order = _augmented_sweep_order(sn_mesh)
    above = np.abs(np.triu(M[np.ix_(order, order)], k=1)).max()
    if above <= 1e-12 * np.abs(M).max():
        _pytest.fail(
            "§16.E teeth #1 has NO teeth: a back-edge seed read left the "
            "augmented matrix triangular — the certificate cannot catch "
            "a coupling-direction swap."
        )
