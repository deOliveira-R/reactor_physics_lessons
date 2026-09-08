r"""The two moment-head AXES — :class:`~orpheus.numerics.axis.HarmonicAxis`
and :class:`~orpheus.numerics.axis.LegendreAxis` (CS4c step 6 item 6.2c-ii,
2026-09-08; the memo's gates P2 and P3).

A moment head is an axis-built space with ONE MODAL axis whose measure is the
head's metric and whose generator is the object that minted it — the
:class:`~orpheus.numerics.basis.base.TruncatedBasis` for the continuum head
(``from_L`` / ``basis.space``), the :class:`~orpheus.numerics.frame.GalerkinFrame`
for the Parseval-dressed one (``frame.basis_space``). Two things this file
pins that nothing else does:

* **P2 — the spent axis is part of the Legendre head's IDENTITY.** `[M]`
  (the 6.2c verification round, hazard H-10) ``LegendreSpace.from_L(1, "x")``
  and ``from_L(1, "z")`` carry ``array_equal`` measures and one shape, so an
  identity of ``(label, shape, kind, weights)`` alone COLLAPSES two physically
  different spaces (the tree carries two poles). The positive leg (same axis
  ⟹ equal) and the negative leg (different spent axis ⟹ unequal) are asserted
  with the measures ``array_equal`` on both sides, so the separation cannot be
  attributed to the measure.
* **P3 — the axis's slots.** ``MODAL``; the measure IS the head's metric (the
  continuum Gram on the basis-minted head, the Parseval inverse on the
  frame-dressed one); the generator narrows to the basis or the frame and
  REFUSES the quadrature, naming both parties (the frame mints keep refusing a
  moment space by exactly that channel).

Foundation mark: software invariants of the axis doctrine; no physics claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.axis import Axis, BasisKind, HarmonicAxis, LegendreAxis
from orpheus.numerics.basis.base import Basis, TruncatedBasis
from orpheus.numerics.basis.legendre_basis import LegendreBasis
from orpheus.numerics.basis.spherical_harmonic_basis import SphericalHarmonicBasis
from orpheus.numerics.frame import GalerkinFrame
from orpheus.numerics.quadrature import Quadrature
from orpheus.numerics.spaces.legendre_space import LegendreSpace
from orpheus.numerics.spaces.spherical_harmonic_space import SphericalHarmonicSpace

pytestmark = pytest.mark.foundation


def _head_axis(space) -> Axis:
    assert space.axes is not None and len(space.axes) == 1, "a moment head is a single-axis space"
    return space.axes[0]


# ═════════════════════════════════════════════════════════════════════════
# P2 — the spent axis separates the Legendre heads
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("L", [0, 1, 3])
def test_the_spent_axis_is_part_of_the_legendre_heads_identity(L: int) -> None:
    """Positive leg: two mints about the same axis are one axis and one space.
    Negative leg: two different spent axes are two axes and two spaces — with
    the measures ``array_equal`` on both sides (the separation is the spent
    axis's, not the measure's)."""
    x1, x2, z = LegendreSpace.from_L(L, "x"), LegendreSpace.from_L(L, "x"), LegendreSpace.from_L(L, "z")
    ax1, ax2, az = _head_axis(x1), _head_axis(x2), _head_axis(z)
    assert isinstance(ax1, LegendreAxis) and isinstance(az, LegendreAxis)
    assert ax1.spent_axis == "x" and az.spent_axis == "z"
    assert ax1 == ax2 and hash(ax1) == hash(ax2) and x1 == x2
    # the measures agree bit-for-bit across the two poles — the family's Gram
    # does not know the axis — so the inequality below is the spent axis's alone
    assert ax1.weights is not None and az.weights is not None
    np.testing.assert_array_equal(ax1.weights, az.weights)
    assert ax1.shape == az.shape and ax1.kind is az.kind
    assert ax1 != az and x1 != z and hash(x1) != hash(z)
    assert len({x1, x2, z}) == 2


def test_a_family_generic_axis_would_collapse_the_two_poles() -> None:
    """The counterfactual the subclass exists for (`[M]` hazard H-10): the SAME
    slots on a plain :class:`Axis` — label, shape, kind, weights — compare
    EQUAL across the two poles, and a space built on such axes cannot tell
    ``x`` from ``z``. The subclass's extended identity key is what separates
    them."""
    ax_x, ax_z = _head_axis(LegendreSpace.from_L(1, "x")), _head_axis(LegendreSpace.from_L(1, "z"))
    plain_x = Axis(ax_x.label, ax_x.shape, ax_x.weights, kind=ax_x.kind)
    plain_z = Axis(ax_z.label, ax_z.shape, ax_z.weights, kind=ax_z.kind)
    assert plain_x == plain_z, "the generic identity is blind to the spent axis — that is the collapse"
    assert ax_x != ax_z
    # and a LegendreAxis is never equal to a plain Axis wearing its slots
    assert ax_x != plain_x and plain_x != ax_x


def test_the_legendre_axis_refuses_an_unknown_spent_axis() -> None:
    with pytest.raises(ValueError, match="spent_axis must be x/y/z"):
        LegendreAxis("legendre", (2,), None, kind=BasisKind.MODAL, spent_axis="w")


# ═════════════════════════════════════════════════════════════════════════
# P3 — the axis's slots: MODAL, the head's measure, the generator
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("L", [0, 2])
def test_the_basis_minted_head_axis_carries_the_continuum_gram_and_the_basis(L: int) -> None:
    """``from_L`` mints ONE MODAL axis: the SH head's measure is the padded
    ``4π/(2ℓ+1)`` (zero outside ``|m| ≤ ℓ``), the Legendre head's the flat
    ``4π/(2ℓ+1)``; the generator narrows to the basis and refuses the
    quadrature, naming both parties."""
    sh = SphericalHarmonicSpace.from_L(L)
    ax = _head_axis(sh)
    assert isinstance(ax, HarmonicAxis) and ax.kind is BasisKind.MODAL
    assert ax.shape == (L + 1, 2 * L + 1)
    w = ax.weights
    assert w is not None
    per_ell = 4.0 * np.pi / (2.0 * np.arange(L + 1) + 1.0)
    for ell in range(L + 1):
        np.testing.assert_allclose(w[ell, : 2 * ell + 1], per_ell[ell], rtol=1e-15)
        np.testing.assert_array_equal(w[ell, 2 * ell + 1 :], 0.0)
    basis = ax.generator_as(Basis, consumer="test")
    assert isinstance(basis, SphericalHarmonicBasis) and isinstance(basis, TruncatedBasis) and basis.L == L
    with pytest.raises(ValueError, match="needs the generating Quadrature"):
        ax.generator_as(Quadrature, consumer="the-asker")

    leg = LegendreSpace.from_L(L, "z")
    lax = _head_axis(leg)
    assert isinstance(lax, LegendreAxis) and lax.kind is BasisKind.MODAL and lax.shape == (L + 1,)
    assert lax.weights is not None
    np.testing.assert_allclose(lax.weights, per_ell, rtol=1e-15)
    lbasis = lax.generator_as(Basis, consumer="test")
    assert isinstance(lbasis, LegendreBasis) and lbasis.axis == "z" and lbasis.L == L
    # the space's metric verbs read the axis's measure — the head has no other source
    assert sh.inner_product_weights is None and sh.metric is None
    assert leg.inner_product_weights is None and leg.metric is None


@pytest.mark.parametrize("label", ["gauss_legendre(8)", "lebedev(11)", "level_symmetric(4)"])
def test_the_frame_dressed_head_axis_carries_the_parseval_measure_and_the_frame(label: str) -> None:
    """``frame.basis_space`` re-weights the SAME axis class with the Parseval
    inverse of the discrete Gram's diagonal (zero on dead slots) and makes the
    FRAME the generator — the object that can re-dress the head at another
    order; the basis is no longer reachable through the axis, the quadrature
    never was."""
    rules = {
        "gauss_legendre(8)": lambda: Quadrature.gauss_legendre(8),
        "lebedev(11)": lambda: Quadrature.lebedev(11),
        "level_symmetric(4)": lambda: Quadrature.level_symmetric(4),
    }
    frame = rules[label]().angular_frame(2)
    dressed = frame.basis_space
    ax = _head_axis(dressed)
    cont = _head_axis(frame.basis.space)
    assert type(ax) is type(cont) and ax.kind is BasisKind.MODAL and ax.shape == cont.shape
    assert ax.generator_as(GalerkinFrame, consumer="test") is frame
    with pytest.raises(ValueError, match="needs the generating Basis"):
        ax.generator_as(Basis, consumer="the-asker")
    with pytest.raises(ValueError, match="needs the generating Quadrature"):
        ax.generator_as(Quadrature, consumer="the-asker")
    assert ax.weights is not None and cont.weights is not None
    diag = np.diagonal(frame.discrete_gram).reshape(ax.shape)
    live = diag > 0.0
    np.testing.assert_allclose(ax.weights[live], 1.0 / diag[live], rtol=1e-15)
    np.testing.assert_array_equal(ax.weights[~live], 0.0)
    assert not np.array_equal(ax.weights, cont.weights)
    assert ax != cont and dressed != frame.basis.space


def test_the_harmonic_axis_is_never_a_plain_axis_wearing_its_slots() -> None:
    """The class is the family: a generic ``Axis`` with identical slots is a
    different axis (the identity is *what kind of generator produced this
    factor*), so a hand-built head can never pass for the family's."""
    ax = _head_axis(SphericalHarmonicSpace.from_L(1))
    plain = Axis(ax.label, ax.shape, ax.weights, kind=ax.kind)
    assert ax != plain and plain != ax
    assert hash(ax) != hash(plain) or ax != plain
