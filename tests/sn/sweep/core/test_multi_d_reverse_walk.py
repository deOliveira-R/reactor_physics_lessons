r"""#310 C3/C4/C5 — the multi-D reverse walk gates (spec §5 + §6, R2a/R2b/R2c).

The multi-D Cartesian adjoint matvec ``(L+C)ᵀφ`` exists as the ORACLE arm:
:meth:`FullFieldWavefront.loss_action_transpose` routes through the shared
:class:`_OctantWalk` apply-transpose frame — the UNCHANGED
:meth:`SweepDependencyGraph.walk_full` over each octant's MIRROR graph
(``−signs_eff``: reversed levels + swapped face roles, the discrete face of
"the adjoint streams along −Ω") × the :class:`_CellResidualTranspose` level
operation, bottoming in the SAME scheme kernel VJP as the 1-D reverse arms
(``residual_kernel_batch_transpose``, #310 C2).

This file is the multi-D sibling of ``test_one_dim_loop_walk.py`` (spec
§5.2) plus the §5.3 structurally-independent object oracles:

* **runtime spy** — both matvec orientations exercise the ONE
  ``walk_full`` frame; direction is the level-op OBJECT + mirror-label
  DATA, observed at runtime;
* **AST tripwire** — no boolean orientation flags in the shared frames;
* **dense-``Mᵀ`` column probe** (the NEW 2-D oracle artifact) — ``M``
  probed off the FORWARD apply over the full composite basis, the reverse
  probed the same way, ``M_rev == M_fwdᵀ`` pinned as an OBJECT (the Mode-12
  stabiliser escape: a matrix equality, outside every spectral invariance
  group);
* **Euclidean pairing identity** — ``⟨Fx, w⟩ = ⟨x, Fᵀw⟩`` over the full
  composite (bulk ⊕ trace; covers the boundary defect/identity algebra);
* **d=1 cross-realization** — the SAME slab operator through TWO
  independent reverse realizations (the C2 reverse legs vs the C3
  mirror-DAG walk) is BIT-identical (same kernel, same elementwise ops,
  batching-order-free);
* **assembled-``Mᵀ``** — the CSR ``M.T @ x`` of the forward-probed
  per-ordinate bulk blocks (LAPACK-side artifact, structurally independent
  of the walk);
* **reverse ``window ≡ full``** (#310 C4, spec §5.1) — the rolling-frontier
  PRODUCTION reverse (``MovingFrontierWindow``) is BIT-identical to the
  full-cochain oracle (same mirror graph, same level order, same kernel —
  different storage), plus the M-R2-WINDOWDRIFT seed-drop tooth;
* **the ScanMarch-2D row-march reverse** (#310 C4, spec §5 R2b) — the
  reversed row march (mirror label → reversed rows + reversed x-scan;
  ``_x_scan_faces_transpose`` + the β-pullback + backwards transverse
  chaining), pinned principled-equivalent against the oracle reverse and
  through the parametrized dense-``Mᵀ``/pairing gates, with the
  transverse-chain and scan-seed teeth;
* **axis equivariance + mutation teeth** (M-R2-ADDRESSING + the
  M-R2-AXISSWAP partial-swap tooth) — the committed value teeth of spec
  §11, plus a MEASURED design finding each way: (a) M-R2-LEVELORDER is
  *unrepresentable* in the mirror-graph realization — the level order and
  the face roles are ONE graph object, so "reverse the addressing but not
  the levels" cannot be spelled; (b) the TOTAL axis conjugation is an
  exact no-op on het σ (2e-16) — the reverse interior is genuinely
  d-generic (no hidden per-axis code), so the representable axis-swap bug
  class is the PARTIAL swap (one tuple crossed, the Mode-2 variable
  swap), which reds the pairing O(1) where it types (square) and cannot
  even shape-check on the rectangular primary configs (L16 — the
  shape-guard is the reason nx≠ny is mandated);
* **the LD-2D reverse** (#310 C5, spec §6 R2c) — the moment-tailed face
  cochain reverses through the SAME mirror-octant frame (the C2 LD batch
  VJP is d-generic; C5 is the gate battery + the flip): dense-``Mᵀ`` and
  pairing on the wavefront pair (ScanMarch construction-refuses LD-2D —
  its own pinned row), the assembled-``Mᵀ`` keystone (the LAPACK/CSR
  artifact emitted from the shared UBLD source), reverse
  ``window ≡ full`` bit-identity, plus the two §6 teeth: the
  moment-drop mutation with its EXACT slope-free blindness control
  (Mode 7 — every committed LD-2D gate is anisotropic by construction)
  and the cross-moment ``x̂ŷ`` frame-sign mutation whose deviation
  splits EXACTLY by octant backward-count parity (the involution's
  group theory made visible — §3.3(c) at d=2, the ERR-066 family's
  likeliest sign-error site);
* **loud deferrals** — the wavefront ``sweep_transpose`` (G-S
  reverse-solve, out of scope R7) and the Pattern-4 moment-tail
  backstop.

``-O``-safe (vv Mode 8): ``pytest.fail`` / ``np.testing`` only.
``foundation`` — software/algebra invariants (no theory ``:label:``).
"""
from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import BC, Mesh2D
from orpheus.numerics.quadrature import Quadrature
from orpheus.sn.loss_representation import (
    CumprodScan,
    FullFieldWavefront,
    IncompatibleRepresentation,
    MovingFrontierWindow,
    ScanMarch,
    _OctantWalk,
    _reverse_octant_traversal,
)
from orpheus.sn.loss_representation.assembly import assemble_ordinate_blocks
from orpheus.sn.loss_representation.sweep_graph import (
    SweepDependencyGraph,
    _CellResidualTranspose,
)
from orpheus.sn.loss_representation import _ApplyOperands
from orpheus.sn.loss_representation.sweep_schedule import _octant_sweep
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.full_field import FullField
from orpheus.transport.spatial.linear_discontinuous import LinearDiscontinuous
from tests.sn.operators.test_g_adjoint_reciprocity import (
    _make_ld_slab,
    _make_slab,
    _random_composite,
)

pytestmark = pytest.mark.foundation


# ═══════════════════════════════════════════════════════════════════════
# Builders — het σ + non-uniform h + rectangular nx≠ny (spec §5.5 / §9)
# ═══════════════════════════════════════════════════════════════════════


def _cart2d_probe_mesh() -> SNMesh:
    """Small rectangular (nx=3 ≠ ny=2) NON-UNIFORM 2-material cart2d mesh —
    the dense-probe config.  Rectangular + non-uniform h + het σ makes
    ``A ≠ Aᵀ`` observable (L16: a square-uniform-symmetric config is
    transpose-BLIND to axis-swap/DOF-transposition classes)."""
    geom = Mesh2D(
        edges_x=np.array([0.0, 0.4, 1.1, 2.0]),
        edges_y=np.array([0.0, 0.7, 1.5]),
        mat_map=np.array([[0, 1], [1, 0], [0, 0]]),
        bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("vacuum"), bc_ymax=BC("vacuum"),
    )
    return SNMesh(
        geom, Quadrature.level_symmetric(2),
        {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")},
    )


def _cart2d_square_uniform_mesh() -> SNMesh:
    """Square (3×3) UNIFORM mesh for the axis-conjugation tooth — the config
    on which the x↔y swap is a symmetry of the GEOMETRY (the quadrature's S2
    ordinates have ``μ_x = μ_y`` exactly), isolating what the mutation
    breaks: the octant-identity coupling, not the mesh."""
    geom = Mesh2D(
        edges_x=np.linspace(0.0, 2.0, 4),
        edges_y=np.linspace(0.0, 2.0, 4),
        mat_map=np.zeros((3, 3), dtype=int),
        bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("vacuum"), bc_ymax=BC("vacuum"),
    )
    return SNMesh(
        geom, Quadrature.level_symmetric(2), {0: get_mixture("A", "2g")},
    )


def _het_sigma(sn: SNMesh, rng: np.random.Generator) -> np.ndarray:
    """Heterogeneous (space × group) positive σ_t for the rep-level calls."""
    return 0.4 + rng.random((2, *sn.spatial_shape))


# ═══════════════════════════════════════════════════════════════════════
# Composite ⇄ flat codec (local, used consistently on BOTH directions —
# the transpose claim is codec-invariant under any fixed DOF bijection)
# ═══════════════════════════════════════════════════════════════════════


def _zero_composite(sn: SNMesh) -> FullField:
    # Scheme-aware: a multi-moment closure (LD) carries the (…, 2^d) bulk
    # tail, selected by the mesh's own per-axis basis size; the boundary
    # auto-sizes from the moment-resolved trace layout.  per_axis == 1
    # (DD) reproduces the moment-free composite byte-identically.
    return FullField(
        interior=AngularFlux.zeros(sn.angular_trial_space),
        boundary=AngularBoundaryFlux.zeros(sn.angular_trace),
    )


def _flatten(field, faces: tuple[str, ...]) -> np.ndarray:
    parts = [np.asarray(field.interior.values).ravel()]
    parts += [np.asarray(field.boundary.face_view(f)).ravel() for f in faces]
    return np.concatenate(parts)


def _basis_size(sn: SNMesh, faces: tuple[str, ...]) -> int:
    z = _zero_composite(sn)
    n = int(np.asarray(z.interior.values).size)
    for f in faces:
        n += int(np.asarray(z.boundary.face_view(f)).size)
    return n


def _basis_composites(sn: SNMesh, faces: tuple[str, ...]):
    """Unit composites in the SAME DOF order ``_flatten`` reads."""
    z = _zero_composite(sn)
    bulk_shape = np.asarray(z.interior.values).shape
    for idx in np.ndindex(*bulk_shape):
        e = _zero_composite(sn)
        e.interior.values[idx] = 1.0
        yield e
    for f in faces:
        view_shape = np.asarray(z.boundary.face_view(f)).shape
        for idx in np.ndindex(*view_shape):
            e = _zero_composite(sn)
            e.boundary.face_view(f)[idx] = 1.0
            yield e


def _pairing(a, b, faces: tuple[str, ...]) -> float:
    tot = float(np.sum(np.asarray(a.interior.values) * np.asarray(b.interior.values)))
    for f in faces:
        tot += float(np.sum(
            np.asarray(a.boundary.face_view(f)) * np.asarray(b.boundary.face_view(f))
        ))
    return tot


def _pairing_defect(sn: SNMesh, rep, sig: np.ndarray, rng) -> float:
    """Relative defect of ``⟨Fx, w⟩ − ⟨x, Fᵀw⟩`` on random full composites."""
    faces = tuple(sn.angular_trace.face_names)
    x = _random_composite(sn, rng)
    w = _random_composite(sn, rng)
    lhs = _pairing(rep.loss_action(sig, x), w, faces)
    rhs = _pairing(x, rep.loss_action_transpose(sig, w), faces)
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)


def _probe_dense(sn: SNMesh, sig: np.ndarray, action) -> np.ndarray:
    """Column-probe ONE direction's dense matrix over the FULL composite
    basis (bulk ⊕ trace) — the shared dense-object artifact.  ``action`` is
    a rep's ``loss_action`` or ``loss_action_transpose`` bound method."""
    faces = tuple(sn.angular_trace.face_names)
    n = _basis_size(sn, faces)
    M = np.empty((n, n))
    for k, e in enumerate(_basis_composites(sn, faces)):
        M[:, k] = _flatten(action(sig, e), faces)
    return M


def _assert_dense_mt_pins_object(sn: SNMesh, rep, sig: np.ndarray, label: str):
    """The ONE dense-``Mᵀ`` object pin — ``M_rev == M_fwdᵀ`` as a MATRIX,
    plus the anti-vacuous asymmetry check.  Shared by the d=2 DD gate, the
    d=3 gate, and the LD-2D gate (one spelling of the pin; Mode-12: a
    matrix equality sits outside every spectral invariance group)."""
    M_fwd = _probe_dense(sn, sig, rep.loss_action)
    M_rev = _probe_dense(sn, sig, rep.loss_action_transpose)
    scale = float(np.max(np.abs(M_fwd)))
    np.testing.assert_allclose(
        M_rev, M_fwd.T, rtol=1e-12, atol=1e-13 * scale,
        err_msg=(
            f"[{label}] the reverse walk is NOT the transpose of the "
            "forward walk (dense column-probe object mismatch)"
        ),
    )
    # The config genuinely discriminates: the operator must not be
    # accidentally symmetric (a symmetric M would null the whole gate).
    if np.allclose(M_fwd, M_fwd.T, rtol=1e-6, atol=1e-9 * scale):
        pytest.fail(
            f"[{label}] probe config produced a symmetric M — the dense-Mᵀ "
            "gate is vacuous on this config (L16: pick het/non-uniform/"
            "rectangular)"
        )


# ═══════════════════════════════════════════════════════════════════════
# §5.2 — the runtime spy + the AST tripwire (the one-walk claim)
# ═══════════════════════════════════════════════════════════════════════


def test_reverse_matvec_routes_through_the_shared_frame(monkeypatch):
    """[L0 structural] the multi-D reverse routes through the SAME
    ``walk_full`` + ``_interior_walk`` frames the forward uses, with the
    ``_CellResidualTranspose`` level op — and the FORWARD control leg never
    touches the transpose op (the two-direction one-walk matrix, Mode-11
    wrap sentinel: an in-process counter, not a green-only claim)."""
    walk_hits: list[str] = []
    op_hits: list[str] = []
    frame_hits: list[str] = []
    real_walk = SweepDependencyGraph.walk_full
    real_cell = _CellResidualTranspose.cell
    real_frame = _OctantWalk._interior_walk

    def walk_spy(self, **kwargs):
        walk_hits.append("walked")
        return real_walk(self, **kwargs)

    def cell_spy(self, cell_idx, **kwargs):
        op_hits.append("cell")
        return real_cell(self, cell_idx, **kwargs)

    def frame_spy(self, sweeps, **kwargs):
        frame_hits.append("frame")
        return real_frame(self, sweeps, **kwargs)

    monkeypatch.setattr(SweepDependencyGraph, "walk_full", walk_spy)
    monkeypatch.setattr(_CellResidualTranspose, "cell", cell_spy)
    monkeypatch.setattr(_OctantWalk, "_interior_walk", frame_spy)

    rng = np.random.default_rng(20260724)
    sn = _cart2d_probe_mesh()
    sig = _het_sigma(sn, rng)
    rep = FullFieldWavefront.pose(sn)
    phi = _random_composite(sn, rng)

    # (1) the REVERSE matvec: all three sentinels fire.
    _ = rep.loss_action_transpose(sig, phi)
    if not (walk_hits and op_hits and frame_hits):
        pytest.fail(
            "the reverse matvec did NOT route through the shared frames "
            f"(walk_full hits={len(walk_hits)}, "
            f"_CellResidualTranspose.cell hits={len(op_hits)}, "
            f"_interior_walk hits={len(frame_hits)}) — the C3 carve "
            "regressed into a private reverse walk."
        )

    # (2) the FORWARD control: the same walk fires, the transpose op NEVER.
    walk_hits.clear(); op_hits.clear(); frame_hits.clear()
    _ = rep.loss_action(sig, phi)
    if not (walk_hits and frame_hits):
        pytest.fail(
            "the forward matvec left the shared frames — the one-walk "
            "claim broke on the forward side."
        )
    if op_hits:
        pytest.fail(
            f"_CellResidualTranspose.cell fired {len(op_hits)}× during the "
            "FORWARD matvec — the direction objects leaked across "
            "orientations."
        )


def test_reverse_walk_is_orientation_object_not_boolean():
    """[L0 structural] the shared multi-D frames fork on OBJECTS (mirror
    labels + level-op) — never a boolean orientation flag (the
    ``test_one_dim_loop_walk`` rule's multi-D sibling; spec §5.2 M-R2-SPY).
    Source-inspection via AST identifiers, so docstrings NAMING the
    anti-pattern don't trip it.  ``-O``-safe."""
    smells = {
        "is_solve", "is_apply", "is_matvec",
        "is_adjoint", "is_forward", "is_transpose", "is_reverse",
    }
    offenders: set[str] = set()
    for obj in (
        SweepDependencyGraph,
        _CellResidualTranspose,
        _OctantWalk,
        _reverse_octant_traversal,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
        identifiers = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
        } | {
            node.arg for node in ast.walk(tree)
            if isinstance(node, (ast.arg, ast.keyword))
            and node.arg is not None
        }
        offenders |= identifiers & smells
    if offenders:
        pytest.fail(
            f"the multi-D walk frames carry boolean orientation flag(s) "
            f"{sorted(offenders)} — orientation MUST be carried by the "
            "mirror-label data + the level-op objects (coding-elegance "
            "Smell #3)."
        )


def test_reverse_traversal_grazing_and_pure_z_labels():
    """[L0 structural] the corner rows of the mirror map, on HAND-BUILT
    labels — no quadrature in the tree produces grazing (single-axis-zero)
    or pure-z ordinates in 2-D, so these branches are otherwise unreachable
    (enforcer C3 SHOULD-FIX #3: new subtle reversal logic wants a tooth
    before a grazing-producing quadrature ever lands):

    * GRAZING: a ``0`` axis rides ``+1`` forward, so its reversal is
      ``−1`` — and the ``0`` must NOT survive into the mirror label (the
      walk's own effective map would re-flip it to ``+1``, un-walking the
      WRONG chain; the tempting ``-s`` spelling yields exactly that bug).
      The physical recovery ``−signs_addr`` then gives ``+1`` — the
      forward's effective sign.
    * PURE-Z: the all-zero in-plane label is its own mirror (the collision
      diagonal is self-transposed) and passes through untouched, so the
      frame's ``pure_z`` branch still keys on it.
    """
    from orpheus.sn.loss_representation.sweep_graph import OctantLabel
    from orpheus.sn.loss_representation.sweep_schedule import OctantSweep

    grazing = OctantSweep(label=OctantLabel((0, -1)), indices=(3, 5))
    pure_z = OctantSweep(label=OctantLabel((0, 0)), indices=(7,))
    mirrored = _reverse_octant_traversal((grazing, pure_z))

    g_label = mirrored[0].label.signs
    if g_label != (-1, +1):
        pytest.fail(
            f"grazing (0, −1) mirrored to {g_label}; expected (−1, +1) — "
            "the grazing axis must mirror AFTER the effective map "
            "(0 → +1 → −1), never survive as 0"
        )
    physical = tuple(-s for s in g_label)
    if physical != (+1, -1):
        pytest.fail(
            f"physical recovery −signs_addr gave {physical}; expected "
            "(+1, −1) — the forward's EFFECTIVE signs of (0, −1): the "
            "grazing axis rides +1, the genuine −1 axis stays −1"
        )
    if mirrored[0].indices != (3, 5):
        pytest.fail("mirroring must not touch the physical ordinate indices")
    if mirrored[1] is not pure_z:
        pytest.fail(
            "the pure-z (all-zero) sweep must pass through UNTOUCHED — it "
            "is its own mirror and the frame's pure_z branch keys on it"
        )


# ═══════════════════════════════════════════════════════════════════════
# §5.3 — the structurally-independent object oracles
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "rep_cls", [FullFieldWavefront, MovingFrontierWindow, ScanMarch],
    ids=["ffw-oracle", "window", "scanmarch"],
)
def test_dense_mt_2d_column_probe_pins_the_object(rep_cls):
    """[L0 object] the NEW 2-D dense-``Mᵀ`` oracle: ``M`` column-probed off
    the FORWARD apply over the FULL composite basis (bulk ⊕ trace), the
    reverse probed identically — ``M_rev == M_fwdᵀ`` as a MATRIX equality,
    for EVERY 2-D representation's own reverse (the C3 oracle arm + the C4
    windowed and row-march production arms — each rep's transpose is the
    transpose of ITS OWN forward).

    Mode-12: this pins the OBJECT, outside every spectral invariance group
    (``eig(Mᵀ) = eig(M)`` makes any spectral functional designed-green on
    the whole transpose mutation class).  Config per spec §5.5: rectangular
    nx≠ny + non-uniform h + het σ, so ``M ≠ Mᵀ`` and addressing bugs are
    observable.  The full composite is mandatory — the domain-boundary
    in↔out swap lives in the trace rows (a bulk-only probe is blind to it).
    """
    rng = np.random.default_rng(20260726)
    sn = _cart2d_probe_mesh()
    _assert_dense_mt_pins_object(sn, rep_cls.pose(sn), _het_sigma(sn, rng), "DD d=2")


@pytest.mark.parametrize(
    "rep_cls", [FullFieldWavefront, MovingFrontierWindow, ScanMarch],
    ids=["ffw-oracle", "window", "scanmarch"],
)
def test_pairing_identity_full_composite(rep_cls):
    """[L0 object] ``⟨Fx, w⟩ = ⟨x, Fᵀw⟩`` at machine precision on random
    full composites — the whole-surface Euclidean pairing (bulk residual +
    chain + boundary defect/identity algebra in one identity), for every
    2-D representation's own (forward, reverse) pair, on both the vacuum
    probe mesh and the reflective nonsquare helper mesh."""
    from tests.sn._test_helpers import cart2d_2g_nonsquare

    rng = np.random.default_rng(20260727)
    for name, sn in (
        ("vacuum 3x2 het", _cart2d_probe_mesh()),
        ("reflective 5x7", cart2d_2g_nonsquare()),
    ):
        sig = _het_sigma(sn, rng)
        rel = _pairing_defect(sn, rep_cls.pose(sn), sig, rng)
        if rel > 1e-12:
            pytest.fail(
                f"[{name}] {rep_cls.__name__} Euclidean pairing identity "
                f"broke: rel defect {rel:.3e} (exact-transpose claim is "
                "machine-precision)"
            )


def test_d1_cross_realization_bit_identical():
    """[L0 cross] the SAME slab operator through TWO independent reverse
    realizations — the C2 reverse legs (``CumprodScan``) vs the C3
    mirror-DAG ``walk_full`` (``FullFieldWavefront``) — is BIT-identical,
    DD and LD.

    Byte-equality is structural, not luck: both bottom in the same
    ``residual_kernel_batch_transpose`` with the same per-cell operand
    values, and every op is elementwise over the ordinate axis (the leg
    batching vs octant batching difference cannot move a bit).  If this
    ever drifts to ULP, a kernel gained a cross-ordinate reduction —
    re-derive before relaxing."""
    rng = np.random.default_rng(20260728)
    for name, (sn, sig) in (
        ("DD slab", _make_slab(ng=2)),
        ("LD slab het non-uniform", _make_ld_slab(ng=2)),
    ):
        x = _random_composite(sn, rng)
        z_scan = CumprodScan.pose(sn).loss_action_transpose(sig, x)
        z_dag = FullFieldWavefront.pose(sn).loss_action_transpose(sig, x)
        np.testing.assert_array_equal(
            np.asarray(z_dag.interior.values),
            np.asarray(z_scan.interior.values),
            err_msg=f"[{name}] bulk cotangents differ between realizations",
        )
        for f in sn.angular_trace.face_names:
            np.testing.assert_array_equal(
                np.asarray(z_dag.boundary.face_view(f)),
                np.asarray(z_scan.boundary.face_view(f)),
                err_msg=f"[{name}] trace cotangent differs on face {f}",
            )


def test_assembled_mt_2d_per_ordinate_block():
    """[L0 object] assembled-``Mᵀ`` (2-D): the CSR ``M.T @ x`` of each
    forward-probed per-ordinate bulk block equals the reverse walk's bulk
    output on a bulk-impulse cotangent — and the transpose stays exactly
    per-(ordinate, group) block-diagonal (no off-block leak).

    Structurally independent of the walk under test: the blocks are
    assembled by forward-kernel unit probes and transposed by scipy CSR.
    σ is the mesh's own material field — the SAME source the assembly
    reads."""
    sn = SNMesh(
        Mesh2D(
            edges_x=np.array([0.0, 0.4, 1.1, 2.1, 3.0]),
            edges_y=np.array([0.0, 0.7, 1.5, 2.0]),
            mat_map=np.array([[0, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 0]]),
            bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
            bc_ymin=BC("vacuum"), bc_ymax=BC("vacuum"),
        ),
        Quadrature.level_symmetric(sn_order=4),
        {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")},
    )
    sigma = np.asarray(
        sn.material_xs_field().total_cross_section_field.values, float,
    )
    rep = FullFieldWavefront.pose(sn)
    rng = np.random.default_rng(20260729)
    n_cells = int(np.prod(sn.spatial_shape))
    for n in range(sn.quad.n_ordinates):
        blocks = assemble_ordinate_blocks(sn, n)
        for g in range(2):
            r = rng.standard_normal(n_cells)
            w = _zero_composite(sn)
            w.interior.values[n, g] = r.reshape(sn.spatial_shape)
            z = rep.loss_action_transpose(sigma, w)
            bulk = np.asarray(z.interior.values)
            np.testing.assert_allclose(
                bulk[n, g].ravel(), blocks[g].apply_transpose(r),
                rtol=1e-12, atol=1e-13,
                err_msg=f"assembled-Mᵀ broke at ordinate {n}, group {g}",
            )
            rest = bulk.copy()
            rest[n, g] = 0.0
            np.testing.assert_array_equal(
                rest, 0.0,
                err_msg=(
                    f"reverse walk leaked off-block at (n={n}, g={g}) — "
                    "the transpose must stay per-ordinate block-diagonal"
                ),
            )


# ═══════════════════════════════════════════════════════════════════════
# §5.1 — the reverse ``window ≡ full`` storage-policy pin (#310 C4, R2b)
# ═══════════════════════════════════════════════════════════════════════


def test_reverse_window_equals_full():
    """[L0 storage] the rolling-frontier PRODUCTION reverse
    (``MovingFrontierWindow.loss_action_transpose``) is BIT-identical to the
    full-cochain oracle — same mirror graph, same reversed level order, same
    ``_CellResidualTranspose`` kernel calls, different storage — so
    ``np.array_equal`` is the RIGHT contract (the L16 ``window ≡ full``
    sibling; anything looser would license a genuinely different reverse).
    Both the het/non-uniform/rectangular vacuum config and the reflective
    nonsquare helper mesh (boundary-cotangent algebra live on both)."""
    from tests.sn._test_helpers import cart2d_2g_nonsquare

    rng = np.random.default_rng(20260803)
    for name, sn in (
        ("vacuum 3x2 het", _cart2d_probe_mesh()),
        ("reflective 5x7", cart2d_2g_nonsquare()),
    ):
        sig = _het_sigma(sn, rng)
        phi = _random_composite(sn, rng)
        full = FullFieldWavefront.pose(sn).loss_action_transpose(sig, phi)
        window = MovingFrontierWindow.pose(sn).loss_action_transpose(sig, phi)
        np.testing.assert_array_equal(
            np.asarray(window.interior.values),
            np.asarray(full.interior.values),
            err_msg=(
                f"[{name}] reverse window ≠ full on the bulk cotangent — "
                "the storage-policy claim broke (same math, different "
                "storage must be BIT-identical)"
            ),
        )
        for f in sn.angular_trace.face_names:
            np.testing.assert_array_equal(
                np.asarray(window.boundary.face_view(f)),
                np.asarray(full.boundary.face_view(f)),
                err_msg=f"[{name}] reverse window ≠ full on face {f}",
            )


def test_window_seed_drop_mutation_reds(monkeypatch):
    """[Mode-10 tooth] M-R2-WINDOWDRIFT, realized as the representable bug:
    dropping the frontier's boundary-cotangent seed (the mirror "inflow" =
    the physical out-face cotangents) moves the windowed reverse O(1) off
    the full-cochain oracle, so the ``window ≡ full`` gate REDS.  The
    frontier-ORDER class itself is unrepresentable at this layer — the
    mirror graph's ``window_plan`` and its levels are ONE object (the same
    finding shape as M-R2-LEVELORDER on the full arm)."""
    import orpheus.sn.loss_representation as lr

    rng = np.random.default_rng(20260804)
    sn = _cart2d_probe_mesh()
    sig = _het_sigma(sn, rng)
    phi = _random_composite(sn, rng)
    reference = np.asarray(
        FullFieldWavefront.pose(sn).loss_action_transpose(sig, phi).interior.values
    )

    orig = lr.MovingFrontierWindow._loss_action_transpose_interior

    def seedless_interior(self, operands, oct_idx, signs_addr, out_bars):
        return orig(
            self, operands, oct_idx, signs_addr,
            tuple(np.zeros_like(b) for b in out_bars),
        )

    monkeypatch.setattr(
        lr.MovingFrontierWindow, "_loss_action_transpose_interior",
        seedless_interior,
    )
    mutated = np.asarray(
        MovingFrontierWindow.pose(sn).loss_action_transpose(sig, phi).interior.values
    )
    rel = float(
        np.max(np.abs(mutated - reference))
        / max(float(np.max(np.abs(reference))), 1e-300)
    )
    if rel < 1e-3:
        pytest.fail(
            f"dropping the frontier boundary-cotangent seed moved the "
            f"windowed reverse only {rel:.3e} off the oracle — the "
            "window ≡ full gate has no bite on the seed wiring"
        )


# ═══════════════════════════════════════════════════════════════════════
# §5.3b — the ScanMarch-2D row-march reverse (#310 C4, R2b scan slice)
# ═══════════════════════════════════════════════════════════════════════


def test_scanmarch_reverse_matches_oracle():
    """[L0 cross] the row-march reverse ≈ the full-cochain oracle reverse —
    principled-equivalent (same kernel VJP, different association order:
    the scan-form x-chain + β-pullback vs the DAG-form face chain), the
    reverse sibling of the forward's row-march-vs-oracle pin.  Tight
    tolerance: any addressing/chaining bug is O(1), association noise is
    ~1e-15."""
    from tests.sn._test_helpers import cart2d_2g_nonsquare

    rng = np.random.default_rng(20260805)
    for name, sn in (
        ("vacuum 3x2 het", _cart2d_probe_mesh()),
        ("reflective 5x7", cart2d_2g_nonsquare()),
    ):
        sig = _het_sigma(sn, rng)
        phi = _random_composite(sn, rng)
        oracle = FullFieldWavefront.pose(sn).loss_action_transpose(sig, phi)
        march = ScanMarch.pose(sn).loss_action_transpose(sig, phi)
        scale = float(np.max(np.abs(np.asarray(oracle.interior.values))))
        np.testing.assert_allclose(
            np.asarray(march.interior.values),
            np.asarray(oracle.interior.values),
            rtol=1e-12, atol=1e-13 * scale,
            err_msg=f"[{name}] row-march reverse bulk ≠ oracle reverse bulk",
        )
        for f in sn.angular_trace.face_names:
            np.testing.assert_allclose(
                np.asarray(march.boundary.face_view(f)),
                np.asarray(oracle.boundary.face_view(f)),
                rtol=1e-12, atol=1e-13 * scale,
                err_msg=f"[{name}] row-march reverse ≠ oracle on face {f}",
            )


def test_scanmarch_transverse_chain_mutation_reds(monkeypatch):
    """[Mode-10 tooth] the reversed transverse chaining: zeroing the kernel
    VJP's ``in_y_bar`` (the cotangent the row march must thread to the
    previous physical row) reds the ScanMarch pairing O(1) — the
    recursion-direction bug class (Mode 4) for the row-march reverse."""
    from orpheus.transport.spatial.diamond import DiamondDifference

    rng = np.random.default_rng(20260806)
    sn = _cart2d_probe_mesh()
    sig = _het_sigma(sn, rng)

    real_vjp = DiamondDifference.residual_kernel_batch_transpose

    def chainless_vjp(self, **kwargs):
        psi_bar_cot, (in_x_bar, in_y_bar) = real_vjp(self, **kwargs)
        return psi_bar_cot, (in_x_bar, np.zeros_like(in_y_bar))

    monkeypatch.setattr(
        DiamondDifference, "residual_kernel_batch_transpose", chainless_vjp,
    )
    rel = _pairing_defect(sn, ScanMarch.pose(sn), sig, rng)
    if rel < 1e-3:
        pytest.fail(
            f"zeroed transverse cotangent chain moved the pairing defect "
            f"only {rel:.3e} — the row-march chaining tooth has no bite"
        )


def test_scanmarch_scan_seed_mutation_reds(monkeypatch):
    """[Mode-10 tooth] the x-chain seed cotangent: dropping the
    ``ordinate_scan_transpose`` ψ̄0 chain term from ``_x_scan_faces_transpose``
    (keeping only the direct ``in_x[0]`` consumption) reds the ScanMarch
    pairing O(1) through the trace rows — the chain-endpoint bug class
    (Mode 5) for the scan-form reverse."""
    import orpheus.sn.loss_representation as lr
    from orpheus.sn.sweep.scan import _x_scan_faces_transpose as real_t

    rng = np.random.default_rng(20260807)
    sn = _cart2d_probe_mesh()
    sig = _het_sigma(sn, rng)

    def seedless_t(alpha, in_x_bar, x_outflow_bar, x_reverse):
        beta_bar, _x_seed_bar = real_t(alpha, in_x_bar, x_outflow_bar, x_reverse)
        direct_only = (in_x_bar[..., ::-1] if x_reverse else in_x_bar)[..., 0]
        return beta_bar, direct_only

    monkeypatch.setattr(lr, "_x_scan_faces_transpose", seedless_t)
    rel = _pairing_defect(sn, ScanMarch.pose(sn), sig, rng)
    if rel < 1e-3:
        pytest.fail(
            f"dropped scan-chain seed term moved the pairing defect only "
            f"{rel:.3e} — the x-chain seed tooth has no bite"
        )


# ═══════════════════════════════════════════════════════════════════════
# d=3 — the family flip's d≥3 face, VERIFIED not assumed (#310 C4-c)
# ═══════════════════════════════════════════════════════════════════════


def test_d3_dense_mt_and_pairing_on_the_spine():
    """[L0 object, d=3] the scheme-aware family trait flips the d-generic
    FullFieldWavefront spine True at d≥3 too — so the d=3 reverse gets its
    own object evidence (a trait claiming capability the gates never ran
    would be capability-claimed-unverified): the mirror-octant reverse on
    a rectangular nx≠ny≠nz NON-UNIFORM d=3 mesh satisfies the Euclidean
    pairing at machine precision AND the dense-``Mᵀ`` full-composite
    matrix equality (the same two objects that pin d=2)."""
    from orpheus.transport.mesh.axis import AxisMesh

    rng = np.random.default_rng(20260808)
    sn = SNMesh.from_axes(
        (
            AxisMesh(edges=np.array([0.0, 0.6, 1.0])),             # nx=2
            AxisMesh(edges=np.array([0.0, 0.4, 1.1, 2.0])),        # ny=3
            AxisMesh(edges=np.array([0.0, 0.3, 0.7, 1.2, 2.0])),   # nz=4
        ),
        Quadrature.level_symmetric(2),
        {0: get_mixture("A", "2g")},
    )
    rep = FullFieldWavefront.pose(sn)
    if rep.has_transpose_walk is not True:
        pytest.fail(
            "the d=3 DD spine must carry the flipped family trait "
            "(#310 C4-c) — has_transpose_walk read False"
        )
    sig = 0.4 + rng.random((2, *sn.spatial_shape))

    rel = _pairing_defect(sn, rep, sig, rng)
    if rel > 1e-12:
        pytest.fail(
            f"d=3 Euclidean pairing identity broke: rel defect {rel:.3e}"
        )

    _assert_dense_mt_pins_object(sn, rep, sig, "DD d=3")


# ═══════════════════════════════════════════════════════════════════════
# §6 — the LD-2D reverse (#310 C5, R2c): the moment-tailed face cochain
# ═══════════════════════════════════════════════════════════════════════


def _ld2d_probe_mesh() -> SNMesh:
    """LD sibling of the DD probe config: rectangular (nx=3 ≠ ny=2)
    NON-UNIFORM 2-material vacuum mesh, LinearDiscontinuous — the bulk
    carries the ``(…, 4)`` ``[avg, ŷ, x̂, x̂ŷ]`` Kronecker tail (axis-0
    outer) and every face a trailing 2-moment ``[avg, transverse-slope]``
    axis (#251)."""
    geom = Mesh2D(
        edges_x=np.array([0.0, 0.4, 1.1, 2.0]),
        edges_y=np.array([0.0, 0.7, 1.5]),
        mat_map=np.array([[0, 1], [1, 0], [0, 0]]),
        bc_xmin=BC("vacuum"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("vacuum"), bc_ymax=BC("vacuum"),
    )
    return SNMesh(
        geom, Quadrature.level_symmetric(2),
        {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")},
        scheme=LinearDiscontinuous(),
    )


def _ld2d_reflective_mesh() -> SNMesh:
    """Reflective nonsquare non-uniform LD sibling — the boundary-cotangent
    algebra live on the MOMENT-RESOLVED trace (reflection threads the
    transverse face-slope; its transpose must too)."""
    geom = Mesh2D(
        edges_x=np.array([0.0, 0.5, 1.3, 2.0]),
        edges_y=np.array([0.0, 0.9, 2.0]),
        mat_map=np.array([[0, 1], [1, 0], [0, 0]]),
        bc_xmin=BC("reflective"), bc_xmax=BC("reflective"),
        bc_ymin=BC("reflective"), bc_ymax=BC("reflective"),
    )
    return SNMesh(
        geom, Quadrature.level_symmetric(2),
        {0: get_mixture("A", "2g"), 1: get_mixture("B", "2g")},
        scheme=LinearDiscontinuous(),
    )


_LD2D_WAVEFRONT_REPS = [FullFieldWavefront, MovingFrontierWindow]
_LD2D_IDS = ["ffw-oracle", "window"]


def test_ld_2d_scanmarch_is_construction_refused():
    """[L0 structural] WHY the LD-2D gates parametrize over the wavefront
    pair only: ScanMarch's facewise supports-gate refuses an LD 2-D mesh
    at CONSTRUCTION (either orientation — there is no ScanMarch LD-2D
    forward to transpose), so the family's LD-2D reverse claim lives
    entirely on the wavefront frame.  Pins the parametrization's honesty:
    if ScanMarch ever admits LD-2D, this reds and the gates here gain a
    row."""
    with pytest.raises(IncompatibleRepresentation):
        ScanMarch.pose(_ld2d_probe_mesh())


@pytest.mark.parametrize("rep_cls", _LD2D_WAVEFRONT_REPS, ids=_LD2D_IDS)
def test_ld_2d_dense_mt_column_probe_pins_the_object(rep_cls):
    """[L0 object] the LD-2D reverse IS the transpose of the LD-2D forward
    — the SAME dense pin as d=2 DD / d=3, now with the ``(…, 4)`` bulk
    moment tail and 2-moment faces in the probe basis.  Anisotropy is
    STRUCTURAL here (spec §6.3): the basis spans every slope DOF, so a
    dropped/mis-signed slope row cannot hide behind an all-flat input."""
    rng = np.random.default_rng(20260812)
    sn = _ld2d_probe_mesh()
    _assert_dense_mt_pins_object(sn, rep_cls.pose(sn), _het_sigma(sn, rng), "LD-2D")


@pytest.mark.parametrize("rep_cls", _LD2D_WAVEFRONT_REPS, ids=_LD2D_IDS)
def test_ld_2d_pairing_identity_full_composite(rep_cls):
    """[L0 object] ``⟨Fx, w⟩ = ⟨x, Fᵀw⟩`` at machine precision on random
    moment-tailed composites (random slope moments ⟹ anisotropic, §6.3),
    vacuum het AND reflective nonsquare (the moment-resolved trace's
    reflection transpose live)."""
    rng = np.random.default_rng(20260813)
    for name, sn in (
        ("vacuum 3x2 het", _ld2d_probe_mesh()),
        ("reflective 3x2", _ld2d_reflective_mesh()),
    ):
        sig = _het_sigma(sn, rng)
        rel = _pairing_defect(sn, rep_cls.pose(sn), sig, rng)
        if rel > 1e-12:
            pytest.fail(
                f"[LD-2D {name}] {rep_cls.__name__} Euclidean pairing "
                f"identity broke: rel defect {rel:.3e}"
            )


def test_ld_2d_reverse_window_equals_full():
    """[L0 storage] reverse ``window ≡ full`` BIT-identical on LD-2D — the
    ``n_face_moments = 2`` frontier slabs carry the transverse moment axis
    through the mirror walk with zero storage-policy drift (the §5.1
    contract at the moment-tailed face width; anisotropic inputs by
    construction)."""
    rng = np.random.default_rng(20260814)
    for name, sn in (
        ("vacuum 3x2 het", _ld2d_probe_mesh()),
        ("reflective 3x2", _ld2d_reflective_mesh()),
    ):
        sig = _het_sigma(sn, rng)
        phi = _random_composite(sn, rng)
        full = FullFieldWavefront.pose(sn).loss_action_transpose(sig, phi)
        window = MovingFrontierWindow.pose(sn).loss_action_transpose(sig, phi)
        np.testing.assert_array_equal(
            np.asarray(window.interior.values),
            np.asarray(full.interior.values),
            err_msg=(
                f"[LD-2D {name}] reverse window ≠ full on the bulk "
                "cotangent — the storage-policy claim broke at "
                "n_face_moments = 2"
            ),
        )
        for f in sn.angular_trace.face_names:
            np.testing.assert_array_equal(
                np.asarray(window.boundary.face_view(f)),
                np.asarray(full.boundary.face_view(f)),
                err_msg=f"[LD-2D {name}] reverse window ≠ full on face {f}",
            )


def test_ld_2d_assembled_mt_per_ordinate_block():
    """[L0 object] the §6.1 KEYSTONE: the CSR ``M.T @ x`` of each
    forward-probed per-ordinate LD block (the ``cell·4 + moment`` DOF
    layout, emitted from the shared UBLD source through
    ``assemble_ordinate_blocks``'s kernel probing + the
    ``octant_moment_frame_signs`` conjugation) equals the reverse walk's
    bulk output on a bulk-impulse cotangent — and the transpose stays
    exactly per-(ordinate, group) block-diagonal (no off-block leak).
    Structurally independent of the walk under test (forward unit probes
    + scipy CSR transpose); σ is the mesh's own material field — the SAME
    source the assembly reads."""
    sn = _ld2d_probe_mesh()
    sigma = np.asarray(
        sn.material_xs_field().total_cross_section_field.values, float,
    )
    rep = FullFieldWavefront.pose(sn)
    rng = np.random.default_rng(20260815)
    n_cells = int(np.prod(sn.spatial_shape))
    cm = sn.scheme.spatial_basis_per_axis ** sn.ndim
    for n in range(sn.quad.n_ordinates):
        blocks = assemble_ordinate_blocks(sn, n)
        for g in range(2):
            r = rng.standard_normal(n_cells * cm)
            w = _zero_composite(sn)
            w.interior.values[n, g] = r.reshape(*sn.spatial_shape, cm)
            z = rep.loss_action_transpose(sigma, w)
            bulk = np.asarray(z.interior.values)
            np.testing.assert_allclose(
                bulk[n, g].ravel(), blocks[g].apply_transpose(r),
                rtol=1e-12, atol=1e-13,
                err_msg=f"LD-2D assembled-Mᵀ broke at ordinate {n}, group {g}",
            )
            rest = bulk.copy()
            rest[n, g] = 0.0
            np.testing.assert_array_equal(
                rest, 0.0,
                err_msg=(
                    f"LD-2D reverse walk leaked off-block at (n={n}, g={g}) "
                    "— the transpose must stay per-ordinate block-diagonal"
                ),
            )


def test_ld_2d_moment_drop_mutation_asymmetry(monkeypatch):
    """[Mode-10 tooth + Mode-7 control] M-R2c-MOMENTDROP: zeroing the
    transverse-slope face-cotangent chain (the ``n_face_moments → 1``
    collapse of the reverse's face algebra) reds the pairing O(1) on
    ANISOTROPIC composites — and is EXACTLY invisible on slope-free
    (isotropic) composites: the mutated and clean defects agree to
    machine precision there.  The asymmetry pair IS the §6.3 mandate's
    proof: every committed LD-2D gate must (and does) drive anisotropic
    inputs, because an all-flat suite cannot see this bug class."""
    rng = np.random.default_rng(20260816)
    sn = _ld2d_probe_mesh()
    sig = _het_sigma(sn, rng)
    rep = FullFieldWavefront.pose(sn)
    faces = tuple(sn.angular_trace.face_names)

    def defect(x, w):
        lhs = _pairing(rep.loss_action(sig, x), w, faces)
        rhs = _pairing(x, rep.loss_action_transpose(sig, w), faces)
        return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-300)

    def slope_free_composite(rng_):
        c = _zero_composite(sn)
        vals = rng_.standard_normal(np.asarray(c.interior.values).shape)
        vals[..., 1:] = 0.0
        c.interior.values[...] = vals
        for f in faces:
            v = c.boundary.face_view(f)
            rv = rng_.standard_normal(v.shape)
            rv[..., 1:] = 0.0
            v[...] = rv
        return c

    x_a, w_a = _random_composite(sn, rng), _random_composite(sn, rng)
    x_i, w_i = slope_free_composite(rng), slope_free_composite(rng)
    clean_iso = defect(x_i, w_i)

    real_vjp = LinearDiscontinuous.residual_kernel_batch_transpose

    def facechain_slope_dropped(self, **kw):
        psi_bar_cot, psi_in_cots = real_vjp(self, **kw)
        dropped = tuple(c.copy() for c in psi_in_cots)
        for c in dropped:
            c[..., 1:] = 0.0
        return psi_bar_cot, dropped

    monkeypatch.setattr(
        LinearDiscontinuous, "residual_kernel_batch_transpose",
        facechain_slope_dropped,
    )
    mut_aniso = defect(x_a, w_a)
    mut_iso = defect(x_i, w_i)
    if mut_aniso < 1e-3:
        pytest.fail(
            f"dropped transverse-slope face chain moved the anisotropic "
            f"pairing defect only {mut_aniso:.3e} — the moment-drop tooth "
            "has no bite"
        )
    if not np.isclose(mut_iso, clean_iso, rtol=0.0, atol=1e-13):
        pytest.fail(
            f"slope-free composites SAW the moment drop (mutated "
            f"{mut_iso:.3e} vs clean {clean_iso:.3e}) — the §6.3 "
            "config-blindness control broke; re-derive before trusting "
            "any isotropic LD-2D snapshot as slope coverage"
        )


def test_ld_2d_cross_moment_frame_sign_octant_asymmetry(monkeypatch):
    """[Mode-10 tooth + Mode-12 asymmetry] M-R2c-FRAMESIGN-2D: dropping
    the cross-moment ``x̂ŷ`` sign from the REVERSE's octant frame
    conjugation (``s[3] → +1`` unconditionally — the "use ∏ wrong"
    spelling) moves the reverse O(1) off ``M_fwdᵀ`` EXACTLY on the
    ordinates of one-backward-axis octants (where the true cross sign is
    ``s_x·s_y = −1``) and NOT AT ALL on both-forward / both-backward
    octants (``+1`` already correct — the mutation sits in those octants'
    stabiliser).  The per-octant parity split in ONE run is the
    involution's ∏-group theory made visible — the §3.3(c) FRAMESIGN
    discipline at d=2 (the ERR-066 family's likeliest sign-error site).
    The forward reference is probed CLEAN before the patch, so the
    corruption is one-sided (a both-sides frame error conjugates away —
    the involution is self-transpose)."""
    import orpheus.sn.loss_representation as lr

    rng = np.random.default_rng(20260817)
    sn = _ld2d_probe_mesh()
    sig = _het_sigma(sn, rng)
    rep = FullFieldWavefront.pose(sn)
    M_fwd = _probe_dense(sn, sig, rep.loss_action)      # clean reference

    orig_mfs = lr._LossRepresentation._moment_frame_signs

    def cross_sign_dropped(self, signs_eff):
        s = orig_mfs(self, signs_eff)
        if s is not None:
            s = s.copy()
            s[3] = 1.0          # the x̂ŷ Kronecker slot (axis-0-outer layout)
        return s

    monkeypatch.setattr(
        lr._LossRepresentation, "_moment_frame_signs", cross_sign_dropped,
    )
    M_rev_mut = _probe_dense(sn, sig, rep.loss_action_transpose)

    dev = np.abs(M_rev_mut - M_fwd.T)
    scale = float(np.max(np.abs(M_fwd)))
    bulk_per_ord = sn.ng * int(np.prod(sn.spatial_shape)) * 4
    for entry in sn.quad.octants:
        sweep = _octant_sweep(entry, sn.ndim)
        odd = sum(1 for s in sweep.label.signs if s < 0) % 2 == 1
        for o in sweep.indices:
            d_bulk = float(np.max(
                dev[o * bulk_per_ord:(o + 1) * bulk_per_ord, :]
            ))
            if odd and d_bulk < 1e-1 * scale:
                pytest.fail(
                    f"one-backward-axis octant {sweep.label.signs} ordinate "
                    f"{o}: cross-sign drop moved its rows only "
                    f"{d_bulk:.3e} — the frame-sign tooth has no bite"
                )
            if not odd and d_bulk > 1e-12 * scale:
                pytest.fail(
                    f"even-backward octant {sweep.label.signs} ordinate "
                    f"{o}: rows moved {d_bulk:.3e} under a mutation inside "
                    "its stabiliser — the parity asymmetry proof is broken"
                )


# ═══════════════════════════════════════════════════════════════════════
# §11 — the committed mutation teeth (M-R2-ADDRESSING + the axis swap)
# ═══════════════════════════════════════════════════════════════════════


def test_addressing_mutation_reds(monkeypatch):
    """[Mode-10 tooth] M-R2-ADDRESSING: dropping the mirror (forward labels
    in the reverse traversal — gather at ``face_in``, seed at the physical
    in-edge, i.e. the whole transposed addressing gone) reds the pairing
    identity O(1).  Fires under ``-O``; reverted by monkeypatch."""
    import orpheus.sn.loss_representation as lr

    rng = np.random.default_rng(20260730)
    sn = _cart2d_probe_mesh()
    sig = _het_sigma(sn, rng)

    monkeypatch.setattr(lr, "_reverse_octant_traversal", lambda sweeps: sweeps)
    rel = _pairing_defect(sn, FullFieldWavefront.pose(sn), sig, rng)
    if rel < 1e-3:
        pytest.fail(
            f"un-mirrored reverse traversal moved the pairing defect only "
            f"{rel:.3e} — the addressing tooth has no bite (the gate would "
            "miss a forward-addressed reverse walk)"
        )


def test_reverse_interior_is_axis_equivariant(monkeypatch):
    """[L0 invariance] the reverse interior is AXIS-EQUIVARIANT: the total
    x↔y conjugation (crossed labels, face tuples, streaming axes, spatial
    axes — inputs AND outputs) is an exact no-op, on HET σ (the
    discriminating config: any hidden per-axis special case would break
    equivariance there).

    This is the symmetry-group pin of d-genericity — the reverse path
    contains no hand-rolled x/y code, so conjugating its whole world by
    the axis permutation reproduces it bit-for-machine-precision.  It is
    NOT a mutation tooth (a total conjugation is a strict symmetry of a
    d-generic FUNCTION, any config); the representable bug class is the
    PARTIAL swap — the companion tooth below."""
    rng = np.random.default_rng(20260731)
    sn = _cart2d_square_uniform_mesh()

    orig = FullFieldWavefront._loss_action_transpose_interior

    def _swap_spatial(a: np.ndarray) -> np.ndarray:
        return np.swapaxes(a, -2, -1)

    def conjugated_interior(self, operands, oct_idx, signs_addr, out_bars):
        ops2 = _ApplyOperands(
            probe=_swap_spatial(operands.probe),
            sig_t=_swap_spatial(operands.sig_t),
            str_axes=operands.str_axes[::-1],
            Q_zero=_swap_spatial(operands.Q_zero),
        )
        psi_cot, capture = orig(
            self, ops2, oct_idx, signs_addr[::-1], out_bars[::-1],
        )
        return _swap_spatial(psi_cot), capture[::-1]

    monkeypatch.setattr(
        FullFieldWavefront, "_loss_action_transpose_interior",
        conjugated_interior,
    )
    sig_het = 0.4 + rng.random((2, 3, 3))       # transpose-ASYMMETRIC
    rel = _pairing_defect(sn, FullFieldWavefront.pose(sn), sig_het, rng)
    if rel > 1e-12:
        pytest.fail(
            f"total axis conjugation moved the reverse by {rel:.3e} on het "
            "σ — the reverse interior gained axis-specific (non-d-generic) "
            "code"
        )


def test_axis_swap_partial_mutation_reds(monkeypatch):
    """[Mode-10 tooth] M-R2-AXISSWAP as the representable bug: the PARTIAL
    swap — crossing ONE per-axis tuple (the out-face cotangents) against
    uncrossed addressing, the Mode-2 variable swap — reds the pairing
    O(1).  Demonstrated on the SQUARE mesh because that is where it TYPES:
    on the rectangular primary configs (dense-Mᵀ, assembled-Mᵀ) the same
    mutation cannot even shape-check — the L16 reason nx≠ny is mandated
    (the whole silent-value-bug class becomes a loud shape error)."""
    rng = np.random.default_rng(20260732)
    sn = _cart2d_square_uniform_mesh()
    sig = 0.4 + rng.random((2, 3, 3))

    orig = FullFieldWavefront._loss_action_transpose_interior

    def crossed_faces_interior(self, operands, oct_idx, signs_addr, out_bars):
        return orig(self, operands, oct_idx, signs_addr, out_bars[::-1])

    monkeypatch.setattr(
        FullFieldWavefront, "_loss_action_transpose_interior",
        crossed_faces_interior,
    )
    rel = _pairing_defect(sn, FullFieldWavefront.pose(sn), sig, rng)
    if rel < 1e-3:
        pytest.fail(
            f"crossed out-face cotangent tuple moved the pairing defect "
            f"only {rel:.3e} — the partial axis-swap tooth has no bite"
        )


# ═══════════════════════════════════════════════════════════════════════
# Loud deferrals (spec §12.2 — out-of-scope stays typed and RED-loud)
# ═══════════════════════════════════════════════════════════════════════


def test_wavefront_solve_transpose_still_raises():
    """[deferral pin] the multi-D ``sweep_transpose`` (the G-S reverse-SOLVE,
    out-of-scope R7) stays a typed raise — C3 lands the matvec transpose
    ONLY, and must not silently un-defer the solve arm."""
    sn = _cart2d_probe_mesh()
    with pytest.raises(NotImplementedError, match="reverse-scan"):
        FullFieldWavefront.pose(sn).sweep_transpose(
            np.zeros((sn.quad.N, 2, *sn.spatial_shape)),
            np.full((2, *sn.spatial_shape), 0.5),
            _random_composite(sn, np.random.default_rng(1)).boundary,
        )


def test_tail_mismatch_refuses_loudly():
    """[Pattern-4 pin] a cotangent whose spatial-moment tail does not match
    the scheme's raises the typed backstop instead of broadcasting silently
    through the batch VJP (the multi-D mirror of the C2 1-D backstop)."""
    sn = _cart2d_probe_mesh()
    rng = np.random.default_rng(20260802)
    phi = _random_composite(sn, rng)
    # Graft a bogus trailing moment axis onto the DD (tail-less) cotangent
    # (width 2^d = 4, the LD-2D layout the DD scheme does not carry). The
    # widened FACTORY mint refuses on a DD mesh since CS4b S4 (the scheme
    # has no moment axis to mint), so the bogus carrier is built RAW on a
    # hand-composed widened space — exactly the off-scheme input the VJP
    # backstop exists to refuse.
    from orpheus.numerics.axis import Axis, BasisKind
    from orpheus.numerics.moment_layout import SPATIAL_MOMENT_AXIS_LABEL
    from orpheus.numerics.space import FunctionSpace

    assert sn.angular_bulk_space.axes is not None
    bad_interior = AngularFlux(
        values=np.asarray(phi.interior.values)[..., None].repeat(4, axis=-1),
        # a hand-built OFF-SCHEME tail axis (the DD scheme mints none; since
        # CS4c step 6 item 6.2c-iii the tail is always an axis, never a class)
        space=FunctionSpace.of_axes(
            *sn.angular_bulk_space.axes,
            Axis(SPATIAL_MOMENT_AXIS_LABEL, (4,), kind=BasisKind.MODAL),
        ),
    )
    bad = FullField(interior=bad_interior, boundary=phi.boundary)
    with pytest.raises(ValueError, match="spatial-moment tail"):
        FullFieldWavefront.pose(sn).loss_action_transpose(_het_sigma(sn, rng), bad)
