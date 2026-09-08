r"""An axis-built space's metric is a DERIVED object over its axes — and a
dense Gram is a block POSITIONED on it (CS4c step 6 item 6.2c-i, ruling
R-6.2c-2, 2026-09-08).

**The doctrine, stated once.** The axes ARE the metric source of an
axis-built space (a measure is diagonal by nature); the space REALIZES that
source as ONE :class:`~orpheus.numerics.metric.FactoredMetric` — a
:class:`~orpheus.numerics.metric.DiagonalMetric` on each weighted axis's
block, ``None`` on a counting axis — and applies it through
:meth:`DiagonalMetric.apply_block`, the explicit leading-1s / block /
trailing-1s reshape-and-multiply.  A form no diagonal measure can spell
(a moment head whose discrete Gram is DENSE — `[M]` 17 of 75 shipped
(rule, L) rows) is a :class:`~orpheus.numerics.metric.DenseMetric`
POSITIONED on that axis's block of the space's own object — an OVERLAY
merged into the derived entries, never a replacement of them: the
construction guard admits ``axes`` + a metric object iff the object is a
``FactoredMetric`` with exactly one entry per axis, in order, a form only
on an axis that carries no measure, and never a diagonal one (a diagonal
measure is the axis's own to carry). Every other axis keeps supplying its
own block, and a product of axis-built factors carries its factors'
overlays concatenated beside its axes.

**The pair this file is (ruling O-2's gate pair):**

* the dense block IS APPLIED — negative control: the counting-axis route
  with no object, `[M]` the two disagree by :math:`O(1)` (the 6.2c
  verification round measured ``max|Δ| = 14.6`` on a dense-Gram head and
  68.7 % / 1784 % on the two ``L ≤ 2`` dense rows — a VALUE bug, which is
  why a merely relaxed guard was refused: the old per-axis short-circuit
  read the axes FIRST and ignored the object);
* with NO dense block the derived object is **bit-identical** to the
  retired inline per-axis loop — reimplemented HERE from its own
  reshape-and-multiply spelling as the oracle, never routed through the
  production helper (``vv`` #22: an oracle that shares the SUT's body is a
  tautology), on synthetic axes (a rank-2 head-shaped axis among them)
  and on the four ledger carriers' trial spaces.

Foundation mark: software invariants of the metric realization, no physics
claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.axis import Axis, BasisKind, EnergyAxis
from orpheus.numerics.metric import DenseMetric, DiagonalMetric, FactoredMetric
from orpheus.numerics.space import FunctionSpace
from tests.sn.architecture.test_monomorphic_leaves import (
    _cart2d,
    _cylinder,
    _slab,
    _sphere,
)

pytestmark = pytest.mark.foundation

_GEOMETRIES = {"slab": _slab, "sphere": _sphere, "cylinder": _cylinder, "cart2d": _cart2d}


# ── the ORACLE: the retired inline loop, spelled here ────────────────────

def _inline_per_axis(axes, x: np.ndarray, *, inverse: bool) -> np.ndarray:
    """The reshape-and-multiply the retired ``_apply_axes_weights`` performed
    (leading 1s for the preceding axes' ranks, the axis shape, trailing 1s
    for the rest; the Moore–Penrose pseudo-inverse per axis) — an oracle
    independent of the production realization."""
    out = np.asarray(x)
    ndim = out.ndim
    start = 0
    for ax in axes:
        rank = len(ax.shape)
        w = ax.weights
        if w is not None:
            wb = np.asarray(w).reshape((1,) * start + tuple(ax.shape) + (1,) * (ndim - start - rank))
            if inverse:
                nonzero = wb != 0.0
                out = np.where(nonzero, out / np.where(nonzero, wb, 1.0), 0.0)
            else:
                out = out * wb
        start += rank
    return out


def _synthetic_axes() -> tuple[Axis, ...]:
    """A rank-2 head-shaped MODAL axis (zero-padded weights, like a
    spherical-harmonic head at L = 1), a counting energy axis, a weighted
    spatial axis — every shape the realization must position."""
    head_w = np.array([[4.0 * np.pi, 0.0, 0.0], [4.0 * np.pi / 3.0] * 3])
    return (
        Axis(label="harmonic_probe", shape=(2, 3), weights=head_w, kind=BasisKind.MODAL),
        EnergyAxis(label="energy", shape=(2,), kind=BasisKind.NODAL),
        Axis(label="spatial", shape=(4,), weights=np.array([0.5, 1.0, 1.5, 2.0]), kind=BasisKind.NODAL),
    )


# ═════════════════════════════════════════════════════════════════════════
# Leg 1 — with NO dense block, the derived object is bit-identical to the
#         retired inline loop (synthetic axes, then the four carriers)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_derived_object_is_bit_identical_to_the_inline_loop_on_synthetic_axes(seed):
    """``apply_metric`` / ``apply_inverse_metric`` / ``inner_product`` on an
    axis-built space ≡ the inline per-axis oracle, ``array_equal`` — including
    a rank-2 axis with zero-padded weights (the pseudo-inverse's kernel) and
    an extra trailing element axis (the leading-aligned convention)."""
    axes = _synthetic_axes()
    space = FunctionSpace.of_axes(*axes)
    if not isinstance(space._resolved_metric, FactoredMetric):
        pytest.fail("the axis-built space did not DERIVE a FactoredMetric from its axes")
    rng = np.random.default_rng(seed)
    for shape in (space.shape, space.shape + (5,)):
        x = rng.standard_normal(shape)
        y = rng.standard_normal(shape)
        if not np.array_equal(space.apply_metric(x), _inline_per_axis(axes, x, inverse=False)):
            pytest.fail(f"apply_metric differs from the inline loop on shape {shape}")
        if not np.array_equal(space.apply_inverse_metric(x), _inline_per_axis(axes, x, inverse=True)):
            pytest.fail(f"apply_inverse_metric differs from the inline loop on shape {shape}")
        want = float(np.sum(_inline_per_axis(axes, x, inverse=False) * y))
        if space.inner_product(x, y) != want:
            pytest.fail(f"inner_product differs from the inline loop's reduction on shape {shape}")


def test_an_all_counting_space_derives_no_object():
    """Every axis counting ⟹ no metric object at all (the Euclidean default,
    no allocation) and the verbs are the identity."""
    space = FunctionSpace.of_axes(
        EnergyAxis(label="energy", shape=(3,), kind=BasisKind.NODAL),
        Axis(label="spatial", shape=(2,), kind=BasisKind.NODAL),
    )
    if space._resolved_metric is not None:
        pytest.fail("an all-counting axis-built space derived a metric object")
    x = np.arange(6.0).reshape(3, 2)
    assert np.array_equal(space.apply_metric(x), x)
    assert space.inner_product(x, x) == float(np.sum(x * x))


@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_the_carriers_trial_space_reads_bit_identically(geometry):
    """On the four ledger carriers' ``angular_trial_space`` (the interior of
    every composite: ``V_cell × w_n`` per axis, a moment axis on LD) the
    derived object reproduces the inline loop bit-for-bit, both directions
    and the pairing."""
    sn_mesh = _GEOMETRIES[geometry]()
    space = sn_mesh.angular_trial_space
    axes = space.axes
    assert axes is not None
    rng = np.random.default_rng(7)
    x = rng.standard_normal(space.shape)
    y = rng.standard_normal(space.shape)
    if not np.array_equal(space.apply_metric(x), _inline_per_axis(axes, x, inverse=False)):
        pytest.fail(f"[{geometry}] apply_metric differs from the inline loop")
    if not np.array_equal(space.apply_inverse_metric(x), _inline_per_axis(axes, x, inverse=True)):
        pytest.fail(f"[{geometry}] apply_inverse_metric differs from the inline loop")
    if space.inner_product(x, y) != float(np.sum(_inline_per_axis(axes, x, inverse=False) * y)):
        pytest.fail(f"[{geometry}] inner_product differs from the inline loop")


# ═════════════════════════════════════════════════════════════════════════
# Leg 2 — a POSITIONED dense block is admitted and APPLIED
# ═════════════════════════════════════════════════════════════════════════

def _dense_gram_on_the_head() -> tuple[tuple[Axis, ...], np.ndarray]:
    """Axes whose head block carries NO diagonal measure (a dense Gram has
    none) plus a symmetric positive-definite 6×6 form for that block."""
    axes = (
        Axis(label="harmonic_probe", shape=(2, 3), kind=BasisKind.MODAL),
        Axis(label="spatial", shape=(4,), weights=np.array([0.5, 1.0, 1.5, 2.0]), kind=BasisKind.NODAL),
    )
    rng = np.random.default_rng(11)
    a = rng.standard_normal((6, 6))
    gram = a @ a.T + 6.0 * np.eye(6)
    return axes, gram


def test_a_positioned_dense_block_is_admitted_and_applied():
    """The guard admits ``axes`` + a FactoredMetric positioned over them
    (the dense form on the measure-less head block, ``None`` on the
    spatial block), and ``apply_metric`` APPLIES the dense block on the
    head's flattened block (oracle: an explicit matmul on the reshaped
    array) while the spatial axis's OWN measure — never restated in the
    object — still applies on its block: the overlay is merged into the
    derived entries, not substituted for them.

    NEGATIVE CONTROL: the same axes with NO object (the counting-axis route
    the old short-circuit would have taken) differ at O(1)."""
    axes, gram = _dense_gram_on_the_head()
    dense = DenseMetric(gram)
    spatial_w = axes[1].weights
    assert spatial_w is not None
    positioned = FactoredMetric((((2, 3), dense), ((4,), None)))
    space = FunctionSpace(name="dense_head_probe", shape=(2, 3, 4), axes=axes, metric=positioned)

    x = np.random.default_rng(3).standard_normal((2, 3, 4))
    got = space.apply_metric(x)
    # Oracle: G on the flattened head block, then the spatial diagonal.
    want = (gram @ x.reshape(6, 4)).reshape(2, 3, 4) * spatial_w.reshape(1, 1, 4)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)
    inv = space.apply_inverse_metric(got)
    np.testing.assert_allclose(inv, x, rtol=1e-11, atol=1e-13)

    counting_route = FunctionSpace.of_axes(*axes).apply_metric(x)
    separation = float(np.abs(got - counting_route).max()) / float(np.abs(got).max())
    if separation < 1e-2:
        pytest.fail(
            f"CONTROL FAILED: the positioned dense block moved apply_metric by only "
            f"{separation:.3e} relative against the counting-axis route"
        )


def test_a_product_of_axis_built_factors_carries_the_overlay_beside_its_axes():
    """``head * bulk`` with a dense-Gram head (the 6.2c-ii occupant: the
    moment space of a DENSE-Gram frame): the product is axis-built (axes
    concatenated), carries the head's positioned form beside them, and
    applies it — the dense block on the head, the spatial measure on its
    own block, identity on the counting energy block. Oracle: the explicit
    matmul on the reshaped array."""
    axes, gram = _dense_gram_on_the_head()
    spatial_w = axes[1].weights
    assert spatial_w is not None
    head_axis = axes[0]
    head = FunctionSpace(
        name="dense_head", shape=(2, 3), axes=(head_axis,),
        metric=FactoredMetric((((2, 3), DenseMetric(gram)),)),
    )
    bulk = FunctionSpace.of_axes(
        EnergyAxis(label="energy", shape=(2,), kind=BasisKind.NODAL), axes[1],
    )
    product = head * bulk
    assert product.axes is not None and len(product.axes) == 3
    if not isinstance(product.metric, FactoredMetric):
        pytest.fail("the product dropped the head's positioned form")
    assert [form is None for _, form in product.metric.entries] == [False, True, True]

    y = np.random.default_rng(5).standard_normal((2, 3, 2, 4))
    got = product.apply_metric(y)
    want = (gram @ y.reshape(6, 8)).reshape(2, 3, 2, 4) * spatial_w.reshape(1, 1, 1, 4)
    np.testing.assert_allclose(got, want, rtol=1e-13, atol=0.0)
    # And the product reads the same pairing its factors do — the head's
    # form and the bulk's measures, one spelling.
    x_head = y[:, :, 0, 0]
    np.testing.assert_allclose(
        head.inner_product(x_head, x_head), float(x_head.ravel() @ gram @ x_head.ravel()),
        rtol=1e-13,
    )


def test_the_guard_refuses_an_object_not_positioned_over_the_axes():
    """``axes`` + a DiagonalMetric spanning the whole shape, a FactoredMetric
    whose blocks do not follow the axes, a positioned DiagonalMetric (a
    diagonal measure is the axis's own to carry), or a form on an axis
    that also carries a measure (two sources on one block) — each refused
    by name; the derived object would otherwise silently disagree with it
    or shadow it."""
    axes, gram = _dense_gram_on_the_head()
    whole = DiagonalMetric(np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="positioned over them"):
        FunctionSpace(name="bad_whole", shape=(2, 3, 4), axes=axes, metric=whole)
    # Blocks that concatenate to the SPACE shape (so the object's own
    # validate_for passes) but do not follow the AXES' blocks.
    misaligned = FactoredMetric((((2,), None), ((3, 4), DenseMetric(np.eye(12)))))
    with pytest.raises(ValueError, match="do not follow the axes"):
        FunctionSpace(name="bad_blocks", shape=(2, 3, 4), axes=axes, metric=misaligned)
    # A diagonal measure spelled OFF its axis.
    off_axis = FactoredMetric((((2, 3), DiagonalMetric(np.full((2, 3), 2.0))), ((4,), None)))
    with pytest.raises(ValueError, match="the axis's own to carry"):
        FunctionSpace(name="bad_diag", shape=(2, 3, 4), axes=axes, metric=off_axis)
    # A form on the WEIGHTED spatial axis — two sources on one block.
    shadowing = FactoredMetric((((2, 3), None), ((4,), DenseMetric(np.eye(4)))))
    with pytest.raises(ValueError, match="two metric sources on one block"):
        FunctionSpace(name="bad_shadow", shape=(2, 3, 4), axes=axes, metric=shadowing)
    # The weights slot stays refused beside axes — unchanged.
    with pytest.raises(ValueError, match="one metric source only"):
        FunctionSpace(name="bad_weights", shape=(2, 3, 4), axes=axes, inner_product_weights=np.ones((2, 3, 4)))
