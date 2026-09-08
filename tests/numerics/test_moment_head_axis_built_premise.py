r"""Pre-carve anchors for CS4c step 6 item 6.2c — the angular moment HEAD becomes axis-built.

**What this file is.** Item 6.2c makes the angular head of every moment space
an AXIS-BUILT space: one rank-``r`` :class:`~orpheus.numerics.axis.Axis`
(``(L+1, 2L+1)`` for the rectangular harmonics, ``(L+1,)`` for the flat
Legendre family), ``BasisKind.MODAL``, carrying the head's own measure, with
``inner_product_weights = None`` (the "never both" construction guard).  It is
the last axes-less factor on every production tensor product.

Every row below is measured on the **unmodified tree** (`main` @ ``79d2944a``)
and is GREEN today.  Three of them are RECORDS of a state the carve re-poses —
they are marked ``⛔ RE-POSED BY 6.2c`` in their own docstrings, so a session
reading a red here knows whether it is a regression or the carve arriving.

**Why the head can be simulated without a production edit.**  The head classes
already inherit ``axes`` from :class:`~orpheus.numerics.space.FunctionSpace`,
and ``__post_init__`` accepts a single axis whose shape IS the head's shape.
So ``replace(head, inner_product_weights=None, axes=(Axis(...),))`` builds
exactly the object 6.2c will mint — with the SAME class, name, shape and ``L``
— and the whole equivalence is measurable before any code moves.

**The measured headline `[M]` 2026-09-07 (`scratch/_step6_2c/p12_axis_head_feasible.py`).**

===========================================================  ==========================
claim                                                        measured
===========================================================  ==========================
head ``apply_metric``  axis route vs dense-slot route        **bit-identical, 0.0**
head ``inner_product``                                       **exactly equal**
product ``apply_metric`` axis-threaded vs ``FactoredMetric``  **bit-identical, 0 ULP**
two axis-built mints of one head                             ``==`` and hash-equal
axis-built head ``==`` its name-built twin                   **False** (the identity flip)
``head * bulk`` with an axis-built bulk                       takes the AXIS arm: 3 axes,
                                                             ``metric is None``
``factors`` / ``factors[0]`` / ``MomentHead`` narrowing        survive
``has_coordinate_cone``                                       ``None`` → **False**
===========================================================  ==========================

⟹ 6.2c's metric unification is a **bit-identical re-expression** on every
diagonal-Gram row, not a ``nulp``-banded one (contrast item 6.2a, whose dense →
factored move measured 2 ULP).  That is the acceptance tier this file pins.

**Activation evidence.**  Each gate names, in its own docstring, the mutation
that reddens it.  The two structural RECORDS (``truncated`` and the
one-metric-source guard) are red-BEFORE facts for 6.2c and are asserted here in
their TODAY form so the carve's own red is attributable.

Foundation mark: software invariants (identity, metric arithmetic, layout
surface); no physics claim rides here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from orpheus.numerics.axis import Axis, BasisKind
from orpheus.numerics.metric import DenseMetric, FactoredMetric
from orpheus.numerics.quadrature import Quadrature
from orpheus.numerics.space import FunctionSpace, TensorProductSpace
from orpheus.numerics.spaces.legendre_space import LegendreSpace
from orpheus.numerics.spaces.moment_head import MomentHead
from orpheus.numerics.spaces.spherical_harmonic_space import SphericalHarmonicSpace

pytestmark = pytest.mark.foundation

_HARMONIC_AXIS_LABEL = "harmonic"


# ── the head's two spellings: the axis-built production one (since 6.2c-ii)
#    and the RETIRED dense-slot twin, re-built here as the comparison oracle ──


#: The two shipped head families.  Typed as their union so the ``MomentHead``
#: members (``L``, ``truncated``, ``degree_block``) resolve statically —
#: ``FunctionSpace`` alone does not carry them (``coding-elegance`` #19: find
#: the principled spelling before reaching for ``# type: ignore``).
Head = SphericalHarmonicSpace | LegendreSpace


def _dense_slot_twin[T: Head](head: T) -> T:
    """The SAME head in its RETIRED spelling — the measure in the legacy
    ``inner_product_weights`` slot, no axes (what ``from_L`` minted until CS4c
    step 6 item 6.2c-ii, 2026-09-08).

    ``replace`` preserves every other subclass field (``L``, and
    ``LegendreSpace.spent_axis``), so the only thing that moves is WHERE the
    measure lives — the twin is the independent oracle for the bit-identity
    rows and the name-built partner for the identity rows.
    """
    assert head.axes is not None and len(head.axes) == 1, "a production head is single-axis"
    w = head.axes[0].weights
    return replace(head, axes=None, inner_product_weights=None if w is None else np.asarray(w))


def _axis_built[T: Head](head: T) -> T:
    """The production head IS axis-built since item 6.2c-ii — identity, kept
    so the rows below read as they were written (the premise landed)."""
    assert head.axes is not None
    return head


def _bulk(ng: int = 2, spatial: tuple[int, ...] = (4, 3)) -> FunctionSpace:
    """An axis-built cell group, as every SN carrier's ``bulk_space`` is."""
    return FunctionSpace.of_axes(
        Axis("energy", (ng,), None, kind=BasisKind.NODAL),
        Axis("spatial", spatial, np.full(spatial, 0.5), kind=BasisKind.NODAL),
    )


_HEADS: dict[str, Callable[[], Head]] = {
    "sh_L0": lambda: SphericalHarmonicSpace.from_L(0),
    "sh_L1": lambda: SphericalHarmonicSpace.from_L(1),
    "sh_L2": lambda: SphericalHarmonicSpace.from_L(2),
    "sh_L3": lambda: SphericalHarmonicSpace.from_L(3),
    "legendre_L0": lambda: LegendreSpace.from_L(0, "x"),
    "legendre_L1": lambda: LegendreSpace.from_L(1, "x"),
    "legendre_L2": lambda: LegendreSpace.from_L(2, "x"),
    "legendre_L3_z": lambda: LegendreSpace.from_L(3, "z"),
}


# ═══════════════════════════════════════════════════════════════════════
# G1 — the metric arithmetic is a BIT-IDENTICAL re-expression
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("label", sorted(_HEADS))
def test_the_head_metric_is_bit_identical_on_the_axis_route(label: str) -> None:
    r"""Moving the head's measure from the dense slot onto ONE axis changes no bit.

    The axes-derived :class:`~orpheus.numerics.metric.FactoredMetric`'s
    ``DiagonalMetric.apply_block`` (since CS4c step 6 item 6.2c-i; the
    inline ``_apply_axes_weights`` loop before it) reshapes the axis's
    weights to the head's own leading block and multiplies — the SAME
    single elementwise product the dense-slot
    :class:`~orpheus.numerics.metric.DiagonalMetric` performs, so
    there is no re-association and ``array_equal`` is the honest assertion
    (``vv-principles`` §bit-identity criterion 3: zero reductions reordered).

    Reddens under: the battery's ``A5`` arm (the per-axis metric skips an
    axis) — `[M]` 2026-09-07 it reddens all 8 rows of this gate, 17 rows of
    this file and 43 of the 4501-row battery scope.
    """
    axis_head = _HEADS[label]()
    head = _dense_slot_twin(axis_head)           # the retired spelling, the oracle
    assert axis_head.axes is not None
    assert axis_head.inner_product_weights is None
    assert axis_head.metric is None
    assert head.axes is None and head.inner_product_weights is not None

    rng = np.random.default_rng(20260907)
    x = rng.standard_normal(head.shape + (2, 4))
    np.testing.assert_array_equal(
        np.asarray(axis_head.apply_metric(x)), np.asarray(head.apply_metric(x)),
        err_msg="the axis route must be bit-identical to the dense-slot route",
    )
    np.testing.assert_array_equal(
        np.asarray(axis_head.apply_inverse_metric(x)),
        np.asarray(head.apply_inverse_metric(x)),
        err_msg="the pseudo-inverse must agree bit-for-bit too (dead slots → 0)",
    )
    assert axis_head.inner_product(x, x) == head.inner_product(x, x)


@pytest.mark.parametrize("label", sorted(_HEADS))
def test_the_product_metric_is_bit_identical_when_the_head_gains_axes(label: str) -> None:
    r"""``head * bulk`` flips from the FactoredMetric arm to the AXIS arm — 0 ULP.

    With an axes-less head the product takes ``from_factors``' factored arm and
    carries a :class:`~orpheus.numerics.metric.FactoredMetric`; with an
    axis-built head EVERY factor is axis-built, so the product threads axes and
    carries ``metric is None``.  The two apply the same per-block diagonals in
    the same order, so the values are bit-identical — this is what makes 6.2c's
    metric unification a re-expression rather than a re-baseline.

    Reddens under: the battery's ``A5`` arm (a per-axis measure dropped) —
    `[M]` all 8 rows of this gate.
    """
    axis_head = _HEADS[label]()
    bulk = _bulk()
    today = _dense_slot_twin(axis_head) * bulk    # the retired spelling's product: the factored arm
    axis_product = axis_head * bulk

    assert isinstance(today.metric, FactoredMetric) and today.axes is None
    assert axis_product.metric is None
    assert axis_product.axes is not None and len(axis_product.axes) == 3
    assert axis_product.inner_product_weights is None
    assert today.shape == axis_product.shape

    rng = np.random.default_rng(5)
    x = rng.standard_normal(today.shape)
    np.testing.assert_array_equal(
        np.asarray(axis_product.apply_metric(x)), np.asarray(today.apply_metric(x)),
        err_msg="the axis-threaded product must reproduce the FactoredMetric bit-for-bit",
    )
    assert axis_product.inner_product(x, x) == today.inner_product(x, x)


def test_no_reachable_array_on_the_axis_product_is_state_sized() -> None:
    r"""The axis arm stores per-AXIS measures, never their outer product.

    The structural half of the memory claim (``L59c``): the separation leg
    below is a synthetic ratio, this one is exact and free — with three axes of
    shapes ``(3, 5)``, ``(2,)``, ``(4, 3)`` the product's index set has 360
    entries and NO array the space can reach has that size.
    """
    head = SphericalHarmonicSpace.from_L(2)
    product = head * _bulk()
    assert product.axes is not None
    n = int(np.prod(product.shape))
    assert n == 360
    reachable = [ax.weights for ax in product.axes if ax.weights is not None]
    assert reachable, "the fixture must carry at least one non-counting axis"
    assert all(w.size < n for w in reachable)
    assert sum(w.nbytes for w in reachable) < 8 * n


def test_the_axis_route_never_densifies_a_large_product() -> None:
    r"""SEPARATION, sized on a synthetic pair so the leg is fixture-honest.

    ``L59c``: a small production fixture gives only an 8× ratio, which is too
    weak to be a keystone.  Two 2000-point axes: per-axis storage is 2 × 2000
    weights (32 000 B), the outer product would be 4 000 000 entries (32 MB).
    """
    w = np.linspace(0.5, 1.5, 2000)
    a = Axis("a", (2000,), w, kind=BasisKind.NODAL)
    b = Axis("b", (2000,), w, kind=BasisKind.NODAL)
    space = FunctionSpace.of_axes(a, b)
    assert space.inner_product_weights is None and space.metric is None
    assert a.weights is not None and b.weights is not None
    stored = a.weights.nbytes + b.weights.nbytes
    dense = 8 * int(np.prod(space.shape))
    assert dense // stored >= 100, f"separation {dense}/{stored} is too weak to be a keystone"


# ═══════════════════════════════════════════════════════════════════════
# G4 — the MomentHead surface survives the axis-building
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("label", sorted(_HEADS))
def test_the_moment_head_surface_survives(label: str) -> None:
    r"""An axis-built head is still a :class:`MomentHead`, with the same layout.

    The head keeps its CLASS (``replace`` on a frozen dataclass), so the five
    production narrowing sites — ``_moment_head_of`` (`transfer.py:128`),
    ``fission.py:177``, ``_bases.py:817``, ``_block_contraction``
    (`material_field.py:176`, keyed on ``len(head.shape)``) and
    ``material_field.py:360`` (``head.degree_block``) — read the same answers.

    Reddens under: minting the head as a bare ``FunctionSpace.of_axes(...)``
    (the design alternative §6 O-4 rejects) — the protocol members vanish.
    """
    axis_head = _HEADS[label]()
    head = _dense_slot_twin(axis_head)
    assert isinstance(axis_head, MomentHead)
    assert axis_head.L == head.L
    assert axis_head.shape == head.shape
    assert axis_head.name == head.name
    assert axis_head.isotropic_slot == head.isotropic_slot
    for l in range(head.L + 1):
        assert axis_head.degree_block(l) == head.degree_block(l)


@pytest.mark.parametrize("label", sorted(_HEADS))
def test_the_product_still_exposes_its_head_as_factor_zero(label: str) -> None:
    r"""``factors[0]`` and ``find_factor`` survive the axis-threading arm.

    ``TensorProductSpace.from_factors`` stores ``factors`` on BOTH arms, so
    the tree query the moment carriers rely on (``space.factors[0].L``,
    ``find_factor(<head class>)``) is untouched — the head-shaped worry
    only arises if the mint moves from ``head * bulk`` to ``of_axes(...)``.
    """
    axis_head = _axis_built(_HEADS[label]())
    product = axis_head * _bulk()
    assert isinstance(product, TensorProductSpace)
    assert product.factors[0] is axis_head
    assert isinstance(product.factors[0], MomentHead)
    assert product.find_factor(type(axis_head)) is axis_head


# ═══════════════════════════════════════════════════════════════════════
# The identity flip on the head — the FORK's mechanism
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("label", sorted(_HEADS))
def test_two_axis_built_mints_of_one_head_are_one_space(label: str) -> None:
    """Structural identity: same axes ⟹ same space, and hash-consistent."""
    a, b = _HEADS[label](), _HEADS[label]()
    assert a is not b
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


@pytest.mark.parametrize("label", sorted(_HEADS))
def test_an_axis_built_head_is_never_equal_to_its_name_built_twin(label: str) -> None:
    r"""The identity flip (6.1): an axis-built space is not a name-built one.

    This is why the head's §6b set is the whole COMPARISON population and not
    just the construction sites — every partner has to move in the same commit
    (``plan-authoring`` §6b).
    """
    axis_head = _HEADS[label]()
    head = _dense_slot_twin(axis_head)
    assert axis_head.name == head.name and axis_head.shape == head.shape
    assert axis_head != head
    assert head != axis_head


@pytest.mark.parametrize("L", [0, 1, 2])
def test_the_metric_enters_the_identity_once_the_head_is_axis_built(L: int) -> None:
    r"""⭐ **THE FORK, in one assertion — RULED and LANDED (R-6.2c-1, item
    6.2c-ii).**  Two heads that differ ONLY in their measure — the basis's
    continuum head and the frame's Parseval-dressed head — are UNEQUAL now
    that the head is axis-built (``Axis.__eq__`` compares weights bytes);
    their RETIRED dense-slot twins are ``==`` (the metric-blind
    ``(name, shape)`` identity of a name-built space), which is the seam the
    tree carried until 6.2c-ii and the reason the fork had to be ruled
    before the head could be axis-built. The tree binds the dressed one.
    """
    frame = Quadrature.lebedev(11).angular_frame(L)
    continuum = frame.basis.space
    dressed = frame.basis_space
    assert isinstance(continuum, SphericalHarmonicSpace) and isinstance(dressed, SphericalHarmonicSpace)
    assert continuum.axes is not None and dressed.axes is not None
    assert not np.array_equal(
        np.asarray(continuum.axes[0].weights), np.asarray(dressed.axes[0].weights),
    )
    assert continuum != dressed, "axis-built: the measure IS the identity — the two heads separate"
    assert _dense_slot_twin(continuum) == _dense_slot_twin(dressed), (
        "the retired name-built spelling was metric-BLIND — that was the seam"
    )


# ═══════════════════════════════════════════════════════════════════════
# The two RED-BEFORE structural records 6.2c must repair
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("label", sorted(_HEADS))
def test_truncated_re_mints_through_the_generator_and_KEEPS_the_axes(label: str) -> None:
    r"""P1 (was the ⛔ RECORD of the pre-6.2c-ii behaviour — ``truncated``
    re-minted through ``from_L`` and LOST the axes, `[M]` 2026-09-07).

    A basis-generated head truncates through its BASIS: the lower head is
    axis-built and equal to the ``L_new`` mint of the same family and spent
    axis (:func:`~orpheus.numerics.spaces.moment_head.truncated_head`).
    Negative control: the retired dense-slot twin of the lower mint is NOT
    what comes back.
    """
    head = _HEADS[label]()
    if head.L == 0:
        pytest.skip("L = 0 has no lower order to truncate to")
    lower = head.truncated(head.L - 1)
    assert isinstance(lower, MomentHead)
    assert lower.axes is not None, "truncated() keeps the axes (re-mints through the generator)"
    assert lower.L == head.L - 1 and lower.name == head.name
    minted = type(head).from_L(head.L - 1, head.spent_axis) if isinstance(head, LegendreSpace) else type(head).from_L(head.L - 1)
    assert lower == minted
    assert lower != _dense_slot_twin(minted)
    assert minted.axes is not None
    assert lower.axes[0].generator == minted.axes[0].generator


@pytest.mark.parametrize("label", ["gauss_legendre(8)", "lebedev(11)", "folded_product(4,8)"])
def test_a_frame_dressed_head_truncates_to_the_frames_dressed_head_at_the_lower_order(label: str) -> None:
    r"""P1, the FRAME arm: the head a moment field carries is the frame's
    Parseval-dressed one, and it truncates through its FRAME — the lower
    head is the frame at ``L_new``'s dressed space (re-dressed, never
    sliced: the Gram verdict can flip with ``L``), structurally equal to the
    quadrature's own interned frame's head at that order — on both families.
    Negative control: the basis's continuum head at ``L_new`` is another space.
    """
    quad = _RULES[label]()
    L = 2
    dressed = quad.angular_frame(L).basis_space
    assert isinstance(dressed, MomentHead)
    assert dressed.axes is not None and dressed.axes[0].generator is quad.angular_frame(L)
    lower = dressed.truncated(L - 1)
    assert isinstance(lower, MomentHead)
    assert lower.axes is not None and lower.L == L - 1 and lower.name == dressed.name
    assert lower == quad.angular_frame(L - 1).basis_space
    assert lower != quad.angular_frame(L - 1).basis.space
    assert lower.axes[0].generator == quad.angular_frame(L - 1)


def test_a_dense_gram_lives_positioned_on_the_axis_built_heads_own_object() -> None:
    r"""RE-POSED at CS4c step 6 item 6.2c-i (ruling R-6.2c-2, 2026-09-08) —
    the construction guard an axis-built head used to collide with.

    Until 6.2c-i ``FunctionSpace.__post_init__`` refused ``axes`` + ``metric``
    together ("one metric source only"), and on a DENSE-Gram frame the
    Parseval-dressed head is exactly ``inner_product_weights=None`` + a
    :class:`DenseMetric` — no diagonal measure to put on an axis, so the
    axis-built head was UNCONSTRUCTIBLE for that row (`[M]` 17 of 75 shipped
    (rule, L) rows are DENSE, 2 of them at ``L ≤ 2``). Now the space's
    metric is a DERIVED object over its axes and the dense form is a block
    POSITIONED on it: admitted, and APPLIED.

    The control the old record carried survives with its teeth: dropping
    the object for a counting axis silently changes the value.
    """
    from orpheus.numerics.metric import FactoredMetric

    G = np.eye(3) + 0.25
    dense = DenseMetric.inverse_of(G)
    axis = Axis(_HARMONIC_AXIS_LABEL, (3,), None, kind=BasisKind.MODAL)
    with pytest.raises(ValueError, match="positioned over them"):
        FunctionSpace(name="probe", shape=(3,), axes=(axis,), metric=dense)
    head = FunctionSpace(
        name="probe", shape=(3,), axes=(axis,),
        metric=FactoredMetric((((3,), dense),)),
    )
    x = np.array([1.0, 2.0, 3.0])
    # The positioned block applies through DenseMetric.apply_block (a
    # reshape-and-matmul on the block) while the standalone object applies
    # through DenseMetric.apply (the leading matmul): the SAME form, two
    # associations — [M] 1 ULP apart on this probe, so the honest tier is
    # a nulp band, not bit-equality (vv-principles §bit-identity criterion 3).
    np.testing.assert_array_almost_equal_nulp(
        head.apply_metric(x), np.asarray(dense.apply(x)), nulp=4,
    )

    counting = FunctionSpace(name="probe", shape=(3,), axes=(axis,))
    np.testing.assert_array_equal(counting.apply_metric(x), x)
    assert not np.allclose(np.asarray(dense.apply(x)), x), (
        "the control must MOVE, or dropping the object would be inert"
    )


def test_the_cone_answer_flips_from_unanswerable_to_refusing() -> None:
    r"""⛔ RE-POSED BY 6.2c — ``has_coordinate_cone`` goes ``None`` → ``False``.

    A MODAL factor makes the per-component sign test meaningless, so
    :meth:`~orpheus.numerics.field.Field.cone_violations` REFUSES rather than
    answering.  `[M]` 2026-09-07 the moment path has **zero production callers**
    of ``cone_violations`` (47 sites tree-wide, all definitions/prose in
    ``orpheus/`` and 35 test calls, none on a moment field), so the flip is
    predicted inert — battery arm ``C1`` measures it rather than asserting it.
    """
    axis_head = SphericalHarmonicSpace.from_L(1)
    head = _dense_slot_twin(axis_head)            # the retired spelling answered None
    assert head.has_coordinate_cone is None
    assert axis_head.has_coordinate_cone is False
    assert (axis_head * _bulk()).has_coordinate_cone is False
    assert (head * _bulk()).has_coordinate_cone is None


# ═══════════════════════════════════════════════════════════════════════
# The production heads, not only the hand-built ones
# ═══════════════════════════════════════════════════════════════════════


_RULES = {
    "gauss_legendre(8)": lambda: Quadrature.gauss_legendre(8),
    "level_symmetric(4)": lambda: Quadrature.level_symmetric(4),
    "lebedev(11)": lambda: Quadrature.lebedev(11),
    "product(4,6)": lambda: Quadrature.product(4, 6),
    "folded_product(4,8)": lambda: Quadrature.folded_product(4, 8),
}


@pytest.mark.parametrize("label", sorted(_RULES))
@pytest.mark.parametrize("L", [0, 1, 2])
def test_the_production_head_of_every_family_is_axis_buildable(label: str, L: int) -> None:
    r"""The head a shipped rule's frame actually binds — built through the
    production chain, not hand-minted (``vv`` #28: build the operand).

    Both heads are covered: the **continuum** head (``frame.basis.space``,
    the basis its generator) and the frame's **dressed** head
    (``frame.basis_space``, the frame its generator) — each axis-built, each
    bit-identical to its own retired dense-slot twin on the metric verbs; on
    a DENSE-Gram row the dressed head carries no axis measure and its dense
    form is POSITIONED on the space's own metric object (hazard H-1, ruling
    R-6.2c-2), so the twin comparison is made on the continuum head only there.
    """
    from orpheus.numerics.frame import GramStructure
    from orpheus.transport.frames import HarmonicFrame

    frame = HarmonicFrame.from_galerkin(_RULES[label]().angular_frame(L))
    rng = np.random.default_rng(11)
    for head, generator in ((frame.basis.space, frame.basis), (frame.basis_space, frame)):
        assert isinstance(head, (SphericalHarmonicSpace, LegendreSpace)), (
            "every shipped rule binds one of the two head families"
        )
        assert head.axes is not None and head.inner_product_weights is None
        assert head.axes[0].generator is generator
        assert isinstance(head, MomentHead) and head.L == L
        if head.axes[0].weights is None:
            assert frame.discrete_gram_structure is GramStructure.DENSE and head is frame.basis_space
            assert isinstance(head.metric, FactoredMetric)
            continue
        twin = _dense_slot_twin(head)
        x = rng.standard_normal(head.shape + (2, 3))
        np.testing.assert_array_equal(
            np.asarray(head.apply_metric(x)), np.asarray(twin.apply_metric(x)),
        )
