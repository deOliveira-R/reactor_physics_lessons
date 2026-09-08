r"""Foundation suite for :class:`SpatialMomentSpace` (#240 D5b-S3-A0).

The within-cell tensor-Legendre DG (spatial) moment space — the spatial
sibling of :class:`SphericalHarmonicSpace` (the angular moment space).
These are software-invariant (``foundation``) tests: they pin the type's
shape / metadata / factory / composition contract, NOT a theory-page
equation. Per ``vv-principles`` § "V&V level taxonomy", a data-structure
+ factory-output invariant carries ``@pytest.mark.foundation`` and NEVER
``verifies(...)``.

Structural independence (``vv-principles`` L11): every claim is checked
against an INDEPENDENTLY-computed expectation — the moment count is
``per_axis ** ndim`` recomputed inline (not read from the space under
test), the composed shape is the hand-concatenated factor shapes, and
the slot-0 convention is cross-checked against the ``_ubld`` single-source
DIRECTLY (not via the space's own property).

Mode-8 / L26: every assertion is a FUNCTION CALL (``np.testing.*`` /
``pytest.fail`` / ``pytest.raises``) — bare ``assert`` is a NO-OP under the
canonical ``-O`` invocation, so it is banned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.space import FunctionSpace, TensorProductSpace
from orpheus.numerics.spaces import (
    SpatialMomentSpace,
    SphericalHarmonicSpace,
    spatial_moment_tail,
)


def _check(cond: bool, msg: str) -> None:
    """Mode-8-safe boolean assertion (a function call, fires under ``-O``)."""
    if not cond:
        pytest.fail(msg)


# ─────────────────────────────────────────────────────────────────────
# (a) shape / metadata / from_* factory + find_factor round-trip
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
@pytest.mark.parametrize(
    "per_axis, ndim, expected_n",
    [
        pytest.param(1, 0, 1, id="degenerate_default_1^0"),
        pytest.param(1, 1, 1, id="dd_step_1d"),
        pytest.param(1, 2, 1, id="dd_step_2d"),
        pytest.param(1, 3, 1, id="dd_step_3d"),
        pytest.param(2, 1, 2, id="ld_1d"),
        pytest.param(2, 2, 4, id="ld_2d"),
        pytest.param(2, 3, 8, id="ld_3d"),
    ],
)
def test_from_per_axis_shape_and_metadata(per_axis, ndim, expected_n):
    r"""``from_per_axis`` encodes ``per_axis ** ndim`` in ``shape`` + metadata.

    The factory mirrors :meth:`SphericalHarmonicSpace.from_L` — the SIZE
    lives in ``shape`` and the descriptive ``(per_axis, ndim)`` factorisation
    is metadata. The expected count ``per_axis ** ndim`` is recomputed
    independently (the parametrize column), not read off the space.
    """
    space = SpatialMomentSpace.from_per_axis(per_axis, ndim)
    independent_n = per_axis ** ndim  # independent recomputation of the count
    _check(independent_n == expected_n, "parametrize column self-consistency")
    np.testing.assert_equal(space.shape, (expected_n,))
    np.testing.assert_equal(space.n_moments, expected_n)
    np.testing.assert_equal(space.per_axis, per_axis)
    np.testing.assert_equal(space.ndim, ndim)
    _check(space.name == "spatial_moment_space", f"name={space.name!r}")
    # Euclidean inner product (no within-cell diagonal metric here, #207).
    _check(space.inner_product_weights is None, "expected Euclidean (no weights)")


@pytest.mark.foundation
def test_find_factor_round_trip_through_composition():
    r"""A composed space recovers the :class:`SpatialMomentSpace` factor by TYPE.

    The query mechanism the moment-carrier fields rely on (issue #207):
    ``space.find_factor(SpatialMomentSpace).per_axis`` recovers the basis
    size without the consumer knowing the factor's position. Mirrors
    ``space.find_factor(SphericalHarmonicSpace).L``.
    """
    sm = SpatialMomentSpace.from_per_axis(2, 2)
    cell_group = FunctionSpace(name="cell_group", shape=(2, 5, 7))
    composed = cell_group * sm
    _check(isinstance(composed, TensorProductSpace), "expected TensorProductSpace")

    recovered = composed.find_factor(SpatialMomentSpace)
    _check(recovered is sm, "find_factor returns the SAME factor object")
    np.testing.assert_equal(recovered.per_axis, 2)
    np.testing.assert_equal(recovered.ndim, 2)

    # Co-existing with the angular factor: a space carrying BOTH moment
    # kinds finds each by type (the orthogonal-axes invariant).
    sh = SphericalHarmonicSpace.from_L(1)
    both = sh * cell_group * sm
    np.testing.assert_equal(both.find_factor(SphericalHarmonicSpace).L, 1)
    np.testing.assert_equal(both.find_factor(SpatialMomentSpace).per_axis, 2)


@pytest.mark.foundation
def test_find_factor_raises_when_absent():
    r"""``find_factor`` raises ``KeyError`` for an absent factor type.

    The query is a structural assertion (the caller believes the space
    carries the factor); an absent factor is an explicit failure, not a
    silent ``None`` (Pattern 4 — illegal-states-unrepresentable). The
    SH-only composition has no spatial-moment factor.
    """
    sh = SphericalHarmonicSpace.from_L(2)
    cell_group = FunctionSpace(name="cell_group", shape=(2, 5, 7))
    composed = sh * cell_group
    with pytest.raises(KeyError):
        composed.find_factor(SpatialMomentSpace)


@pytest.mark.foundation
def test_rejects_inconsistent_shape_and_negative_metadata():
    r"""``__post_init__`` rejects shape/metadata inconsistency (Pattern 4).

    The space is a typed cross-check: ``shape == (per_axis ** ndim,)`` must
    hold by construction, and ``per_axis >= 1`` / ``ndim >= 0``. These are
    production invariants → real ``raise`` (fire under ``-O``).
    """
    with pytest.raises(ValueError):
        SpatialMomentSpace(name="spatial_moment_space", shape=(3,), per_axis=2, ndim=2)
    with pytest.raises(ValueError):
        SpatialMomentSpace(name="spatial_moment_space", shape=(1,), per_axis=0, ndim=1)
    with pytest.raises(ValueError):
        SpatialMomentSpace(name="spatial_moment_space", shape=(1,), per_axis=1, ndim=-1)


# ─────────────────────────────────────────────────────────────────────
# (b) tensor-product composition has the right shape
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
@pytest.mark.parametrize(
    "cell_group_shape, per_axis, ndim",
    [
        pytest.param((2, 3, 4), 2, 2, id="cellgroup_2g_3x4_ld2d"),
        pytest.param((1, 6), 2, 1, id="cellgroup_1g_6_ld1d"),
        pytest.param((3, 5, 5, 5), 2, 3, id="cellgroup_3g_5x5x5_ld3d"),
    ],
)
def test_tensor_product_composition_shape(cell_group_shape, per_axis, ndim):
    r"""``cell_group * SpatialMomentSpace`` concatenates factor shapes.

    The composed shape is the hand-concatenated factor shapes
    ``cell_group_shape + (per_axis ** ndim,)`` — computed independently of
    the ``__mul__`` machinery under test.
    """
    cell_group = FunctionSpace(name="cell_group", shape=cell_group_shape)
    sm = SpatialMomentSpace.from_per_axis(per_axis, ndim)
    composed = cell_group * sm
    independent_shape = cell_group_shape + (per_axis ** ndim,)
    np.testing.assert_equal(composed.shape, independent_shape)
    # an allocation on the composed space has the same shape (Field gate proxy)
    np.testing.assert_equal(np.zeros(composed.shape).shape, independent_shape)


# ─────────────────────────────────────────────────────────────────────
# (c) byte-identity at default — per_axis == 1 appends NO factor
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_per_axis_1_size_is_unit_no_widening(ndim):
    r"""``per_axis == 1`` gives a size-1 space and the tail policy says ``()``.

    The backward-compat invariant: a cell-average closure (DD/Step) widens
    NOTHING. ``spatial_moment_tail(per_axis ** ndim) == ()`` is the
    "append iff > 1" gate every consumer rides; here it is checked DIRECTLY
    against the single-source policy.
    """
    sm = SpatialMomentSpace.from_per_axis(1, ndim)
    np.testing.assert_equal(sm.shape, (1,))
    np.testing.assert_equal(sm.n_moments, 1)
    # The policy single-source returns () at count 1 — no trailing axis.
    np.testing.assert_equal(spatial_moment_tail(sm.n_moments), ())
    # And > 1 returns the genuine trailing axis (the contrast).
    ld = SpatialMomentSpace.from_per_axis(2, ndim)
    np.testing.assert_equal(spatial_moment_tail(ld.n_moments), (2 ** ndim,))


# ─────────────────────────────────────────────────────────────────────
# (e) moment-ordering / size consistency with _ubld single-sources
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_average_moment_index_matches_moment_layout_single_source():
    r"""``average_moment_index`` equals the ``moment_layout.AVERAGE_MOMENT`` single-source.

    Structural independence (L11): the space's slot-0 convention is checked
    against the canonical ``moment_layout`` constant DIRECTLY (the production
    single-source every moment consumer reduces on, #245), NOT against a
    literal ``0`` re-spelled in the test. If the Kronecker layout's slot-0
    convention ever moves, BOTH the production reductions and this property
    track it together.
    """
    from orpheus.numerics.moment_layout import AVERAGE_MOMENT

    sm = SpatialMomentSpace.from_per_axis(2, 2)
    np.testing.assert_equal(sm.average_moment_index, AVERAGE_MOMENT)
    # the average index is a valid slot of every non-degenerate space
    _check(
        0 <= sm.average_moment_index < sm.n_moments,
        "average_moment_index must be a valid slot",
    )


@pytest.mark.foundation
@pytest.mark.parametrize(
    "per_axis, ndim",
    [(2, 1), (2, 2), (2, 3), (1, 2)],
)
def test_n_moments_matches_spatial_basis_per_axis_power_ndim(per_axis, ndim):
    r"""``n_moments == spatial_basis_per_axis ** ndim`` (the UBLD count law).

    The same derivation the production scheme uses
    (:attr:`DiscretizationSchemeBase.spatial_basis_per_axis` raised to the
    mesh ``ndim``). Pinned against an independent recomputation so the
    space and the scheme can never disagree on the cell-moment count.
    """
    sm = SpatialMomentSpace.from_per_axis(per_axis, ndim)
    independent = per_axis ** ndim
    np.testing.assert_equal(sm.n_moments, independent)
    np.testing.assert_equal(sm.shape, (independent,))


@pytest.mark.foundation
def test_equality_by_size_identity():
    r"""Equality is by ``(name, shape)`` — size-identity, per the abstract frame.

    ``shape == (per_axis ** ndim,)`` encodes the SIZE but not the
    ``(per_axis, ndim)`` factorisation, so a ``(per_axis=4, ndim=1)`` space
    and a ``(per_axis=2, ndim=2)`` space (both size 4) compare equal — the
    same convention :class:`SphericalHarmonicSpace` /
    :class:`TensorProductSpace` follow (identity = type tag + dimension).
    The two never coexist on one mesh, so size-identity is correct.
    """
    a = SpatialMomentSpace.from_per_axis(2, 2)  # shape (4,)
    b = SpatialMomentSpace.from_per_axis(4, 1)  # shape (4,)
    c = SpatialMomentSpace.from_per_axis(2, 1)  # shape (2,)
    _check(a == b, "size-identity: (pa=2,d=2) == (pa=4,d=1) (both shape (4,))")
    np.testing.assert_equal(hash(a), hash(b))
    _check(a != c, "distinct size → not equal")
    _check(len({a, c}) == 2, "distinct size → distinct spaces (container separation; hash inequality is not a law)")
    # cross-class equality with a bare FunctionSpace of the same (name, shape)
    bare = FunctionSpace(name="spatial_moment_space", shape=(4,))
    _check(a == bare, "equal-(name,shape) bare FunctionSpace compares equal")
