r"""``FunctionSpace.of_axes`` — composition, naming, per-axis metric, cone
(campaign 1, CS1 step 2).

Gate ids B1–B8 refer to the CS1 battery of record
(``scratch/cs1_verification_plan.md`` §2); B9–B11 land with step 3b in this
same module.

⚠ Deliberately NOT hosted in ``test_space.py`` / ``test_space_algebra.py``:
those pinned the densifying ``__mul__`` path (``test_space_algebra.py``
asserted ``inner_product_weights`` IS the dense ``np.outer``) until CS4c
step 6 item 6.2a (2026-09-07) retired it — their dense-slot rows were
re-keyed in place onto the factored metric's VALUES (a behavioural row
migrates, it is not deleted). Two composition mechanisms, two files —
``of_axes`` (the pure axis path) and ``*`` (axis threading, else a
factored metric) stay two until item 6.2c makes the angular head
axis-built.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from orpheus.numerics.axis import Axis, BasisKind, EnergyAxis
from orpheus.numerics.space import FunctionSpace

pytestmark = pytest.mark.foundation

_EDGES_2G = np.array([1.0e7, 1.0e3, 1.0e-3])


def _require(condition: bool, message: str) -> None:
    """A ``-O``-firing assertion (NOT a bare ``assert``)."""
    if not condition:
        pytest.fail(message)


def _point() -> Axis:
    """The quotient spatial point (counting weight — the density convention)."""
    return Axis("spatial", (1,), kind=BasisKind.NODAL)


def _reachable_arrays(space: FunctionSpace) -> list[np.ndarray]:
    """Every ndarray reachable from the space's own state (B4's walker)."""
    found: list[np.ndarray] = []
    stack: list[object] = list(vars(space).values())
    while stack:
        obj = stack.pop()
        if isinstance(obj, np.ndarray):
            found.append(obj)
        elif isinstance(obj, (tuple, list)):
            stack.extend(obj)
        elif isinstance(obj, Axis):
            stack.extend(vars(obj).values())
    return found


def test_of_axes_shape_is_the_concatenation() -> None:
    """B1 — shape by concatenation, over a RANK-2 axis too.

    The rank-2 member (the harmonic axis's shape, in-fence as a generic
    ``Axis``) is what makes a ``shape + (n,)`` vs ``shape + axis.shape``
    slip visible; a battery of rank-1 axes only cannot see it.
    """
    space = FunctionSpace.of_axes(
        EnergyAxis.synthetic(2),
        Axis("harmonic", (3, 5), kind=BasisKind.MODAL),
        _point(),
    )
    _require(space.shape == (2, 3, 5, 1), f"shape {space.shape} != (2, 3, 5, 1)")
    axes = space.axes
    assert axes is not None
    _require(len(axes) == 3, "the axes record must carry all three factors")


def test_of_axes_name_is_INJECTIVE_on_structural_content() -> None:
    r"""B2 ⭐ — distinct axis tuples ⟹ distinct names.

    Load-bearing until the identity flip (CS4c step 6, 2026-09-07) because
    space identity was ``(name, shape)`` (Q2): a NAME COLLISION between
    different axis tuples made two different spaces compare EQUAL, so the
    composition guard passed an ill-posed sum — the exact defect CS1 exists
    to make unspellable, reintroduced one layer down. Since the flip an
    axis-built space compares by its axes tuple; the name's injectivity is
    what keeps an axes-less composite's folded digest content-keyed and the
    label diagnostic, so the gate stays.

    The population deliberately contains two SAME-SHAPE pairs, where shape
    carries no information and only the name can discriminate:

    * ``synthetic(2)`` vs ``from_grid(<2-group edges>)``  (A5)
    * spatial point with weight ``1.0`` vs weight ``2.0``  (A12 / B9)
    """
    from orpheus.data.energy_grid import EnergyGrid

    tuples: list[tuple[Axis, ...]] = [
        (EnergyAxis.synthetic(2), _point()),
        (EnergyAxis.from_grid(EnergyGrid(_EDGES_2G)), _point()),  # same shapes
        (EnergyAxis.synthetic(2), Axis("spatial", (1,), weights=np.array([2.0]), kind=BasisKind.NODAL)),  # same shapes
        (EnergyAxis.synthetic(3), _point()),
        (EnergyAxis.synthetic(2),),
        (EnergyAxis.synthetic(2), Axis("spatial", (1,), kind=BasisKind.MODAL)),  # kind differs
    ]
    spaces = [FunctionSpace.of_axes(*t) for t in tuples]
    names = {s.name for s in spaces}
    _require(
        len(names) == len(tuples),
        f"name collision: {len(names)} names for {len(tuples)} distinct axis tuples",
    )
    _require(
        len(set(spaces)) == len(tuples),
        "space identity collision — two different axis tuples compare equal",
    )


def test_of_axes_name_is_BLIND_to_the_generator() -> None:
    """B12 (CS5) — provenance never perturbs the identity bridge.

    Until the identity flip (CS4c step 6) space identity was ``(name,
    shape)`` and the derived name WAS the identity; since the flip
    ``__eq__`` reads ``_identity_key`` directly through ``Axis.__eq__``, so
    provenance stays out of the identity at both tiers for one reason.
    ``_structural_bytes`` iterates ``_identity_key``,
    which excludes ``Axis.generator``; this gate is the observable
    statement of that, at the tier where an inclusion would do damage.
    Mutation: append ``self.generator`` to ``_identity_key`` → the digest
    moves ([M] ``angular(4,)#a1259d874905e50e`` → ``#a56a82a93fac074b``),
    and ``==``/``hash`` RAISE outright (the generator objects are
    un-comparable / unhashable — the exclusion is structurally mandatory,
    see ``test_axis_generator.py`` G1c).

    ⚠ Do NOT count B3 as this gate's catcher: its subprocess leg runs an
    unmutated interpreter, so it reds for ANY in-process digest change —
    a cross-process differential, not a provenance gate.
    """
    from orpheus.numerics.quadrature.directional import Quadrature

    q = Quadrature.gauss_legendre(4)
    minted = q.axis("angular")
    literal = Axis(
        "angular", (q.N,), weights=np.asarray(q.weights, float),
        kind=BasisKind.NODAL,
    )
    s_m = FunctionSpace.of_axes(minted)
    s_l = FunctionSpace.of_axes(literal)
    _require(s_m.name == s_l.name, f"digest moved: {s_m.name!r} != {s_l.name!r}")
    _require(s_m == s_l and hash(s_m) == hash(s_l), "space identity moved")
    _require(
        FunctionSpace.of_axes(minted, _point()).name
        == FunctionSpace.of_axes(literal, _point()).name,
        "the shipped composite's name moved",
    )


def test_of_axes_name_is_deterministic_across_processes() -> None:
    """B3 — the derived name does not depend on ``PYTHONHASHSEED``.

    A name built from ``hash(...)`` of a str/tuple is per-process random.
    The in-process leg cannot see that; the subprocess leg can, and it is
    the only leg that can.
    """
    space = FunctionSpace.of_axes(EnergyAxis.synthetic(2), _point())
    twin = FunctionSpace.of_axes(EnergyAxis.synthetic(2), _point())
    _require(space.name == twin.name, "same content, two constructions, one name")

    snippet = (
        "from orpheus.numerics.axis import Axis, BasisKind, EnergyAxis\n"
        "from orpheus.numerics.space import FunctionSpace\n"
        "s = FunctionSpace.of_axes(EnergyAxis.synthetic(2), "
        "Axis('spatial', (1,), kind=BasisKind.NODAL))\n"
        "print(s.name)\n"
    )
    import os

    env = dict(os.environ, PYTHONHASHSEED="271828")
    child = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, env=env,
    )
    _require(child.returncode == 0, f"subprocess failed: {child.stderr}")
    _require(
        child.stdout.strip() == space.name,
        f"name differs across processes: {child.stdout.strip()!r} != {space.name!r} "
        f"— a per-process hash leaked into the identity bridge",
    )


def test_of_axes_never_densifies_the_metric() -> None:
    r"""B4 ⭐ — the factor measures stay per-axis; no outer product is
    materialized.

    Three legs, because each is blind alone:

    1. **exact/structural** — ``space.inner_product_weights is None`` (the
       dense slot is not populated) and no ndarray reachable from the
       space has ``size == prod(space.shape)``;
    2. **memory-shaped** — total reachable ndarray bytes <= the per-axis
       bytes + slack, at a shape with a ``[M]`` 1000x separation
       (``(2000,) x (2000,)``: dense 32 000 000 B vs per-axis 32 000 B);
    3. **behavioural** — the metric still APPLIES correctly at that shape
       (a "never densify" implemented by dropping the metric would pass
       legs 1-2; this is that leg's control).

    ⛔ NOT done by asking a densifier to ``MemoryError``: ``[M]`` a 550 GB
    ``np.multiply.outer`` did not raise, it got the process OOM-KILLED
    (exit 137), which fails the RUN, not the TEST.
    """
    n = 2000
    w_a = np.linspace(0.5, 4.0, n)
    w_b = np.linspace(0.25, 8.0, n)
    space = FunctionSpace.of_axes(
        Axis("a", (n,), weights=w_a, kind=BasisKind.NODAL),
        Axis("b", (n,), weights=w_b, kind=BasisKind.NODAL),
    )
    # Leg 1: structural.
    _require(space.inner_product_weights is None, "dense slot populated")
    full = int(np.prod(space.shape))
    _require(
        all(arr.size != full for arr in _reachable_arrays(space)),
        "an ndarray of the FULL product size is reachable from the space",
    )
    # Leg 2: memory-shaped (per-axis 2 x 16 000 B; dense would be 32 MB).
    total_bytes = sum(arr.nbytes for arr in _reachable_arrays(space))
    _require(
        total_bytes <= 10 * (2 * n * 8),
        f"reachable ndarray bytes {total_bytes} exceed per-axis budget "
        f"— something densified",
    )
    # Leg 3: behavioural control — the metric ACTS at this shape.
    x = np.ones((n, n))
    gx = space.apply_metric(x)
    sample = np.random.default_rng(7).integers(0, n, size=(20, 2))
    for i, j in sample:
        _require(
            bool(gx[i, j] == w_a[i] * w_b[j]),
            f"metric wrong at ({i},{j}): {gx[i, j]} != {w_a[i] * w_b[j]}",
        )


def test_per_axis_metric_equals_an_INDEPENDENT_broadcast_reference() -> None:
    r"""B5 — on a weighted toy, the per-axis metric equals a reference
    built in this test.

    ⚠ The reference is written HERE from an explicit reshape, NOT from
    ``orpheus.numerics.metric._broadcast_leading`` (the one home of the
    leading broadcast since ``FunctionSpace._broadcast_metric`` retired at
    CS4c step 6 item 6.2a) — routing both sides through the
    production helper would make this a tautology on the very convention
    (LEADING vs trailing padding) that has already shipped a bug once.
    Non-square ``(3, 4)``, because a square toy cannot see an axis swap.
    Power-of-two weights, so every product is IEEE-exact and the
    comparison is bit-level.
    """
    w_a = np.array([2.0, 4.0, 8.0])
    w_b = np.array([0.5, 1.0, 2.0, 4.0])
    space = FunctionSpace.of_axes(
        Axis("a", (3,), weights=w_a, kind=BasisKind.NODAL),
        Axis("b", (4,), weights=w_b, kind=BasisKind.NODAL),
    )
    rng = np.random.default_rng(42)
    x = rng.standard_normal((3, 4))
    # Independent reference: explicit reshape, each weight on ITS axis.
    reference = w_a.reshape(3, 1) * (w_b.reshape(1, 4) * x)
    _require(
        bool(np.array_equal(space.apply_metric(x), reference)),
        "per-axis metric disagrees with the independent placement reference",
    )
    # Pseudo-inverse roundtrip (exact for power-of-two weights).
    _require(
        bool(np.array_equal(space.apply_inverse_metric(space.apply_metric(x)), x)),
        "inverse metric does not invert the metric",
    )


def test_metric_and_inner_product_agree_on_a_NONSQUARE_axis_space() -> None:
    r"""B6 ⭐ — the inner product equals the independently-built weighted
    pairing on a non-square space.

    The ERR-067 family, one layer up. ``FunctionSpace._diagonal_inner_product``'s
    own Notes record that ``inner_product`` and ``apply_metric`` DIVERGED
    in the tree until 2026-08-04 (trailing vs leading broadcast; ``[M]``
    456 vs 552 on a ``(3,3)`` probe, invisible whenever
    ``w.ndim >= x.ndim``). ``A.H = G^-1 A^T G`` is built from
    ``apply_metric`` while the pairing that judges it comes from
    ``inner_product``: if they disagree, the adjoint identity is false by
    construction and every reciprocity gate downstream is meaningless.

    On the per-axis path the production pairing is SINGLE-SOURCED through
    ``apply_metric`` (one spelling — the divergence is unspellable by
    construction), so the literal pair-agreement check would be a
    tautology; the gate therefore compares against a reference built HERE
    from an explicit reshape, which keeps teeth on the whole path (an
    ``apply_metric`` mutation reddens this gate through the inherited
    pairing).
    """
    w_a = np.array([2.0, 4.0, 8.0])
    w_b = np.array([0.5, 1.0, 2.0, 4.0])
    space = FunctionSpace.of_axes(
        Axis("a", (3,), weights=w_a, kind=BasisKind.NODAL),
        Axis("b", (4,), weights=w_b, kind=BasisKind.NODAL),
    )
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, 4))
    y = rng.standard_normal((3, 4))
    reference = float(np.sum(w_a.reshape(3, 1) * w_b.reshape(1, 4) * x * y))
    got = space.inner_product(x, y)
    _require(
        bool(np.isclose(got, reference, rtol=1e-13, atol=0.0)),
        f"inner product {got} != independent weighted pairing {reference}",
    )


def test_mul_threads_axes_and_does_not_fabricate_them() -> None:
    """B7 — ``(A * B).axes == A.axes + B.axes``; a legacy space on either
    side leaves the product's ``axes`` ``None``.

    The negative half is the point: inventing an axis for a space that
    never declared one would make ``has_coordinate_cone`` answer for
    spaces that have not been migrated (a false True, in the direction
    that silently ENABLES the step-4 cone consult).

    Third leg — the mixed product: an axis-built factor's measure must
    survive, positioned in the product's factored metric (until CS4c step
    6 item 6.2a it rode a dense slot through a densifier bridge), never
    be silently dropped (treating a weighted axis-built factor as
    Euclidean would be a value bug wearing a representation label).
    """
    a = FunctionSpace.of_axes(EnergyAxis.synthetic(2), _point())
    b = FunctionSpace.of_axes(Axis("x", (3,), kind=BasisKind.NODAL))
    product = a * b
    a_axes, b_axes = a.axes, b.axes
    assert a_axes is not None and b_axes is not None
    _require(product.axes == a_axes + b_axes, "axes must thread through *")
    _require(
        product.inner_product_weights is None,
        "an all-axes product must not densify",
    )

    legacy = FunctionSpace("legacy", (5,))
    _require((a * legacy).axes is None, "axes fabricated for a legacy right factor")
    _require((legacy * a).axes is None, "axes fabricated for a legacy left factor")

    # The mixed product: weighted axis-built x legacy — the measure survives,
    # FACTORED (CS4c step 6 item 6.2a): no dense slot, and the applied metric
    # is the outer product it never stores.
    weighted = FunctionSpace.of_axes(
        Axis("w", (2,), weights=np.array([2.0, 4.0]), kind=BasisKind.NODAL)
    )
    mixed = weighted * legacy
    _require(mixed.axes is None, "mixed product must not carry a partial axes record")
    _require(mixed.inner_product_weights is None, "a mixed product must not densify")
    _require(
        mixed.metric is not None,
        "the axis-borne measure was dropped by the mixed product",
    )
    expected = np.multiply.outer(np.array([2.0, 4.0]), np.ones(5))
    _require(
        bool(np.array_equal(mixed.apply_metric(np.ones((2, 5))), expected)),
        "the axis-borne measure was distorted by the mixed product",
    )


@pytest.mark.parametrize(
    "kinds,expected",
    [
        ((BasisKind.NODAL, BasisKind.NODAL), True),
        ((BasisKind.NODAL, BasisKind.MODAL), False),
        ((BasisKind.MODAL, BasisKind.MODAL), False),
    ],
)
def test_has_coordinate_cone_follows_the_basis_kinds(
    kinds: tuple[BasisKind, BasisKind], expected: bool
) -> None:
    """B8a — all-nodal ⟹ True, any-modal ⟹ False."""
    space = FunctionSpace.of_axes(
        Axis("p", (2,), kind=kinds[0]), Axis("q", (3,), kind=kinds[1])
    )
    _require(
        space.has_coordinate_cone is expected,
        f"kinds {kinds} => {space.has_coordinate_cone}, expected {expected}",
    )


def test_has_coordinate_cone_is_None_on_a_legacy_space() -> None:
    """B8b — ``axes is None`` ⟹ ``None``, the third state.

    Three-valued deliberately: ``False`` means "provably no coordinate
    cone", ``None`` means "not migrated". Collapsing them would make the
    step-4 refusal fire on every legacy space in the tree.
    """
    _require(
        FunctionSpace("legacy", (5,)).has_coordinate_cone is None,
        "a legacy space must answer None (not migrated), never True/False",
    )


def test_dual_of_an_axis_built_space_keeps_the_measure() -> None:
    """The dual carries the SAME metric as the primal (L²-Riesz) — for an
    axis-built primal that means the axes record threads through
    ``dual()``; dropping it would silently strip the measure from every
    adjoint built against the dual."""
    w = np.array([2.0, 4.0])
    space = FunctionSpace.of_axes(
        Axis("a", (2,), weights=w, kind=BasisKind.NODAL)
    )
    dual = space.dual()
    _require(dual.axes == space.axes, "dual() must thread the axes record")
    x = np.array([1.0, 1.0])
    _require(
        bool(np.array_equal(dual.apply_metric(x), w)),
        "the dual's metric must equal the primal's",
    )


def test_quotient_point_and_a_genuine_one_cell_mesh_are_DIFFERENT_spaces() -> None:
    r"""B9 ⭐ — the collapse doctrine's retrodiction (A.5 row 4), mechanized.

    ``MaterialMesh.from_materials`` mints the QUOTIENT carrier — volumes
    ``[1.0]``, the normalized "per unit volume" density convention, whose
    spatial weight canonicalizes to counting. A genuine one-cell slab of
    width 2 keeps ``V = 2`` BY THE DATA. Both spaces have shape
    ``(ng, 1)`` — shape carries nothing; the derived NAME (hence space
    identity) is the only discriminator.

    ⚠ Per F2 (measured): this distinction is provably INVISIBLE to
    ``.H`` — a scalar metric commutes with every operator, so ``V = 2``
    reads bit-identical to ``V = 1`` on every adjoint — which is exactly
    why this identity gate exists and why no adjoint-flavoured gate can
    replace it (mutation M17's MUST-STAY-GREEN column is the proof).

    # CS1.5 re-point: ``bulk_space`` moves to Medium; this gate re-points
    # with it.
    """
    from orpheus.derivations.common.xs_library import get_mixture
    from orpheus.geometry import Mesh1D
    from orpheus.transport.mesh.material_mesh import MaterialMesh

    mix = get_mixture("A", "2g")
    quotient = MaterialMesh.from_materials({0: mix}).bulk_space
    one_cell_mesh = MaterialMesh(
        Mesh1D(edges=np.array([0.0, 2.0]), mat_ids=np.array([0])), {0: mix}
    )
    _require(
        bool(np.array_equal(one_cell_mesh.volumes, [2.0])),
        "precondition lost: the one-cell slab no longer has V = 2",
    )
    one_cell = one_cell_mesh.bulk_space
    _require(quotient.shape == one_cell.shape == (2, 1), "precondition: same shape")
    _require(
        quotient != one_cell,
        "the quotient point and a genuine one-cell mesh collapsed to ONE "
        "space — the measure fell out of the identity",
    )


def test_bulk_space_on_a_MESHED_carrier_is_the_honest_scalar_bulk() -> None:
    """B10 — ``(ng, *spatial)`` with cell-volume weights; and DISTINCT
    from the method mesh's angular composite.

    ``bulk_space`` is inherited by ``SNMesh``/``DiffusionMesh``, so the
    uniform formula must be honest on EVERY member (the seed of CS2's
    single scalar-bulk mint). The second half
    (``mesh.bulk_space != mesh.full_field_space``) is what makes D7's
    chain-ordering claim non-vacuous.

    # CS1.5 re-point: ``bulk_space`` moves to Medium; this gate re-points
    # with it.
    """
    from orpheus.derivations.common.xs_library import get_mixture
    from orpheus.diffusion import DiffusionMesh
    from orpheus.geometry import BC, Mesh1D
    from orpheus.transport.mesh.material_mesh import MaterialMesh

    mix = get_mixture("A", "2g")
    # Non-uniform edges, so the volumes are NOT all-ones and the weights
    # survive canonicalization (a uniform unit mesh would collapse to the
    # counting spelling and this gate would assert nothing).
    mesh1d = Mesh1D(edges=np.array([0.0, 1.0, 3.0, 6.0]), mat_ids=np.array([0, 0, 0]))
    carrier = MaterialMesh(mesh1d, {0: mix})
    space = carrier.bulk_space
    _require(space.shape == (2, 3), f"scalar bulk shape {space.shape} != (2, 3)")
    axes = space.axes
    assert axes is not None
    spatial_weights = axes[1].weights
    assert spatial_weights is not None
    _require(
        bool(np.array_equal(spatial_weights, carrier.volumes)),
        "the spatial factor measure must BE the cell volumes",
    )

    diffusion_mesh = DiffusionMesh(
        Mesh1D(
            edges=np.array([0.0, 1.0, 3.0, 6.0]),
            mat_ids=np.array([0, 0, 0]),
            bc_left=BC("reflective"),
            bc_right=BC("reflective"),
        ),
        {0: mix},
    )
    _require(
        diffusion_mesh.bulk_space != diffusion_mesh.full_field_space,
        "the scalar bulk and the composite carrier collapsed — D7's "
        "chain-ordering claim would be vacuous",
    )


def test_the_spatial_axis_is_minted_through_the_carriers_own_measure() -> None:
    """G6a (CS5) — the mesh generates the spatial measure; the measure
    mints the axis; the axis is TODAY'S axis (same identity, same digest)
    plus its generator.

    Independent literal anchor (verification plan Q4): the volumes and
    centres asserted below are HAND-DERIVED from the edge list — not read
    back from the mesh — so the mesh→measure→axis chain has a pin that
    is structurally independent of every array it threads.
    """
    from orpheus.derivations.common.xs_library import get_mixture
    from orpheus.geometry import Mesh1D
    from orpheus.transport.mesh.material_mesh import MaterialMesh

    mix = get_mixture("A", "2g")
    # Cartesian slab, edges 0|1|3|6 ⟹ BY HAND: widths (=volumes) 1,2,3;
    # centres 0.5, 2.0, 4.5. Non-uniform, so canonicalization keeps them.
    carrier = MaterialMesh(
        Mesh1D(edges=np.array([0.0, 1.0, 3.0, 6.0]), mat_ids=np.array([0, 0, 0])),
        {0: mix},
    )
    space = carrier.bulk_space
    sp_ax = space.axis("spatial")
    g = sp_ax.generator
    _require(g is not None, "the rank-1 spatial axis must carry its measure")
    assert g is not None
    npt = np.testing
    npt.assert_array_equal(g.weights, np.array([1.0, 2.0, 3.0]),
                           err_msg="volumes, hand-derived")
    npt.assert_array_equal(g.nodes, np.array([0.5, 2.0, 4.5]),
                           err_msg="centres, hand-derived")
    literal = Axis(
        "spatial", carrier.spatial_shape, weights=carrier.volumes,
        kind=BasisKind.NODAL,
    )
    _require(sp_ax == literal, "minted spatial axis != today's literal")
    _require(
        FunctionSpace.of_axes(literal).name == FunctionSpace.of_axes(sp_ax).name,
        "the spatial digest moved",
    )
    # G8-spatial — the section law on the rank-1 arm (the law's domain):
    _require(g.axis(sp_ax.label) == sp_ax, "spatial mint: not a section")


def test_the_rank_d_spatial_axis_is_generator_less_BY_CONTRACT() -> None:
    """G6b (CS5) — the CS2 rank-d seam, pinned as a CONTRACT.

    A ``DiscreteMeasure`` is a flat atom list, so ``measure.axis`` mints
    rank 1; a rank-d spatial axis (shape ``(nx, ny, nz)``) has no rank-d
    measure→axis pairing yet. [M] minting it flat would change every
    d≥2 space name (``spatial(12,)#3712…`` vs ``spatial(3, 4)#1dcb…`` —
    verification plan R1), so the rank-d arm deliberately stays
    generator-less. ⛔ The day CS2 mints the rank-d pairing, THIS row
    must be inverted DELIBERATELY — it is the seam's witness, not a
    permanent truth.
    """
    from orpheus.numerics.quadrature.directional import Quadrature
    from orpheus.sn.mesh.augmented_mesh import SNMesh
    from orpheus.transport.mesh.axis import AxisMesh
    from tests.sn._test_helpers import placeholder_materials

    axes = tuple(
        AxisMesh(edges=np.linspace(0.0, ext, n + 1))
        for ext, n in zip((1.0, 2.0, 3.0), (2, 3, 2))
    )
    sn = SNMesh.from_axes(
        axes, Quadrature.level_symmetric(sn_order=4), placeholder_materials(ng=2)
    )
    sp_ax = sn.bulk_space.axis("spatial")
    _require(sp_ax.shape == (2, 3, 2), f"fixture rank moved: {sp_ax.shape}")
    _require(
        sp_ax.generator is None,
        "the rank-d arm must stay generator-less until CS2 mints the "
        "rank-d measure→axis pairing (inverting this is a deliberate act)",
    )


def test_bulk_space_is_cached_and_content_stable() -> None:
    """B11 — one carrier, ONE instance (``is``); equal carriers, EQUAL
    spaces (the derived name is content, not identity).

    # CS1.5 re-point: ``bulk_space`` moves to Medium; this gate re-points
    # with it.
    """
    from orpheus.derivations.common.xs_library import get_mixture
    from orpheus.transport.mesh.material_mesh import MaterialMesh

    mix = get_mixture("A", "2g")
    carrier = MaterialMesh.from_materials({0: mix})
    _require(carrier.bulk_space is carrier.bulk_space, "must be cached")
    twin = MaterialMesh.from_materials({0: mix})
    _require(
        carrier.bulk_space == twin.bulk_space
        and hash(carrier.bulk_space) == hash(twin.bulk_space),
        "equal carriers must mint equal spaces",
    )


def test_axis_built_construction_guards() -> None:
    """The two illegal states of an axis-built space are refused, and
    ``of_axes`` refuses an empty factor list.

    One metric source only (per-axis measures XOR dense weights), and the
    shape must BE the axes' concatenation — both are construction bugs
    when violated, so both raise typed errors.
    """
    ax = Axis("a", (2,), weights=np.array([2.0, 4.0]), kind=BasisKind.NODAL)
    with pytest.raises(ValueError, match="one metric source"):
        FunctionSpace(
            "bad", (2,), inner_product_weights=np.array([1.0, 2.0]), axes=(ax,)
        )
    with pytest.raises(ValueError, match="concatenation"):
        FunctionSpace("bad", (3,), axes=(ax,))
    with pytest.raises(ValueError, match="at least one axis"):
        FunctionSpace.of_axes()


class TestThreeSourceExclusivity:
    """P7 S2 (battery of record ``scratch/p7_verification_plan.md`` §2
    group B): a space takes exactly ONE metric source — per-axis
    measures XOR dense weights XOR a metric object — with each pairwise
    arm its own witness (vv-principles #17's granularity trap: the
    ``(dense, metric)`` arm was structurally UNREACHABLE before the P7
    guard restructure, hidden behind the axes early-return).
    """

    def test_a_space_takes_exactly_one_metric_source(self) -> None:
        """B1 — the three positive legs: each source ALONE constructs,
        and its metric ACTS (the matching realization, read through the
        public verb rather than a private attribute)."""
        from orpheus.numerics.metric import DenseMetric

        x = np.array([1.0, 2.0])
        ax = Axis("a", (2,), weights=np.array([2.0, 4.0]), kind=BasisKind.NODAL)
        by_axes = FunctionSpace.of_axes(ax)
        _require(
            bool(np.array_equal(by_axes.apply_metric(x), [2.0, 8.0])),
            "the axes source must act per axis",
        )
        by_weights = FunctionSpace(
            "w", (2,), inner_product_weights=np.array([2.0, 4.0])
        )
        _require(
            bool(np.array_equal(by_weights.apply_metric(x), [2.0, 8.0])),
            "the dense-weights source must act as the Hadamard metric",
        )
        by_object = FunctionSpace(
            "m", (2,), metric=DenseMetric(np.array([[2.0, 0.5], [0.5, 4.0]]))
        )
        _require(
            bool(np.array_equal(by_object.apply_metric(x), [3.0, 8.5])),
            "the metric object must act as the dense form",
        )

    def test_axes_and_a_metric_object_are_refused(self) -> None:
        """B2 — the (axes, metric) arm (battery arm M10b), re-posed at CS4c
        step 6 item 6.2c-i (ruling R-6.2c-2, 2026-09-08): an axis-built
        space's metric is DERIVED from its axes, so a bare object beside
        them is still two sources and is refused by name — while an object
        POSITIONED over the axes (a FactoredMetric with one entry per axis,
        a form only on an axis that carries NO measure) is ADMITTED,
        because that is where a dense Gram lives on an axis-built head; a
        form beside an axis that also carries a measure is two sources on
        one block and stays refused. Until then ANY object beside axes was
        refused ("one metric source only")."""
        from orpheus.numerics.metric import DenseMetric, FactoredMetric

        ax = Axis("a", (2,), weights=np.array([2.0, 4.0]), kind=BasisKind.NODAL)
        dense = DenseMetric(np.array([[1.0, 0.0], [0.0, 1.0]]))
        with pytest.raises(ValueError, match="positioned over them"):
            FunctionSpace("bad", (2,), axes=(ax,), metric=dense)
        with pytest.raises(ValueError, match="two metric sources on one block"):
            FunctionSpace("bad", (2,), axes=(ax,), metric=FactoredMetric((((2,), dense),)))
        counting = Axis("a", (2,), kind=BasisKind.NODAL)
        admitted = FunctionSpace(
            "good", (2,), axes=(counting,), metric=FactoredMetric((((2,), dense),)),
        )
        _require(admitted.metric is not None, "the positioned object was dropped")

    def test_dense_weights_and_a_metric_object_are_refused(self) -> None:
        """B3 — the (dense, metric) arm: the one the pre-P7 structure
        could not reach (``if self.axes is None: return`` short-circuited
        before any check). Battery arm M10c is its teeth."""
        from orpheus.numerics.metric import DenseMetric

        with pytest.raises(ValueError, match="one metric source"):
            FunctionSpace(
                "bad",
                (2,),
                inner_product_weights=np.array([1.0, 2.0]),
                metric=DenseMetric(np.array([[1.0, 0.0], [0.0, 1.0]])),
            )


class TestAxisAccessor:
    """S-1 (un-weld arc): ``FunctionSpace.axis(label)`` — the public
    by-label factor accessor, sharing the collapse pair's resolver so the
    refusal vocabulary cannot drift (``_axis_index`` is the one home).

    Positive AND negative per vv-principles #11: the axis returned IS the
    tuple member (identity, not a copy), and each structural refusal
    carries its typed class + pinned fragment.
    """

    def test_returns_the_tuple_member_identically(self) -> None:
        eg = EnergyAxis("energy", (2,), kind=BasisKind.NODAL, edges=_EDGES_2G)
        sp = Axis("spatial", (3, 4), kind=BasisKind.NODAL)
        space = FunctionSpace.of_axes(eg, sp)
        _require(space.axis("energy") is eg, "energy axis must be the member itself")
        _require(space.axis("spatial") is sp, "spatial axis must be the member itself")
        _require(space.axis("energy").shape == (2,), "ng reads off the axis")
        _require(space.axis("spatial").shape == (3, 4), "spatial shape reads off the axis")

    def test_unknown_label_refuses_naming_the_inventory(self) -> None:
        space = FunctionSpace.of_axes(_point(), Axis("angular", (4,), kind=BasisKind.NODAL))
        with pytest.raises(ValueError, match="names 0 axes"):
            space.axis("energy")

    def test_duplicate_label_refuses(self) -> None:
        space = FunctionSpace.of_axes(
            Axis("a", (2,), kind=BasisKind.NODAL),
            Axis("a", (3,), kind=BasisKind.NODAL),
        )
        with pytest.raises(ValueError, match="names 2 axes"):
            space.axis("a")

    def test_legacy_name_built_space_refuses(self) -> None:
        legacy = FunctionSpace("legacy", (2, 3))
        with pytest.raises(TypeError, match="not axis-built"):
            legacy.axis("energy")
