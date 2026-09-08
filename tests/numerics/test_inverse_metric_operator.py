r"""``InverseMetricOperator`` — a space's inverse metric AS an operator.

The adapter lets a SPACE's inverse metric enter the operator algebra —
the trace metrics' :math:`G^{+}`, a degenerate metric's Moore–Penrose face.
:func:`test_the_frame_can_now_spell_its_own_projector` was its founding
reason: the projection algebra :math:`\Pi = R \circ G^{-1} \circ M` was
written down in the tree but the frame's probe was a ``FunctionSpace``
while :meth:`~orpheus.numerics.frame.FrameBase.conjugate` wants a
``LinearOperator``. Since CS4c step 6 item 6.2c-ii the frame owns that
factor as a typed arrow (:attr:`~orpheus.numerics.frame.FrameBase.gram_inverse`,
``test_space → basis_space``): an endomorphism of a metric-twin space
cannot compose with the faces once the metric enters space identity, so the
projector row below spells the frame's own arrow, and this adapter keeps
the space-side rows (the degenerate trace metrics) as its justification.

⚠ The degenerate-metric rows are the load-bearing ones.  `[M]` the SN
trace metric :math:`G = |\Omega\cdot\hat n| w_n` is EXACTLY zero on
tangential ordinates — 192 of 384 rows under ``product(4,4)`` — so the
pseudo-inverse masking is the normal case here, not an edge case, and a
suite that only ever exercised ``level_symmetric`` (0 zero rows) would
be blind to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import BC, Mesh2D
from orpheus.numerics.basis import SphericalHarmonicBasis
from orpheus.numerics.frame import GalerkinFrame
from orpheus.numerics.operator import InverseMetricOperator
from orpheus.numerics.quadrature import Quadrature, lebedev_sphere
from orpheus.sn.mesh.augmented_mesh import SNMesh

_REFLECTIVE = BC("reflective")


def _trace_space(quadrature: Quadrature):
    """The SN boundary-trace space of a small all-reflective 2-D box."""
    edges = np.linspace(0.0, 1.0, 4)
    mesh = Mesh2D(
        edges_x=edges, edges_y=edges,
        mat_map=np.zeros((3, 3), dtype=int),
        bc_xmin=_REFLECTIVE, bc_xmax=_REFLECTIVE,
        bc_ymin=_REFLECTIVE, bc_ymax=_REFLECTIVE,
    )
    sn_mesh = SNMesh(mesh, quadrature, {0: get_mixture("B", "2g")})
    space = sn_mesh.full_field_space.trace_space
    assert space is not None, "a 2-D box always has a boundary-trace space"
    return space


#: ``level_symmetric`` has NO tangential ordinate, ``product`` has many —
#: the pair is the point (a nondegenerate and a degenerate metric).
_QUADRATURES = {
    "level_symmetric(4)": Quadrature.level_symmetric(4),
    "product(4,4)": Quadrature.product(n_mu=4, n_phi=4),
}


@pytest.mark.foundation
@pytest.mark.parametrize("name", sorted(_QUADRATURES))
def test_the_operator_is_bound_to_the_space_on_both_ends(name):
    """``domain`` and ``codomain`` are the space itself.

    The metric re-weights a carrier; it does not move between spaces.
    Binding BOTH ends is what lets ``OperatorProduct``'s compatibility
    guard check a composition that contains it — an unbound factor
    poisons that end of the product.
    """
    space = _trace_space(_QUADRATURES[name])
    operator = InverseMetricOperator(space)
    assert operator.domain is space
    assert operator.codomain is space
    assert operator.space is space


@pytest.mark.foundation
@pytest.mark.parametrize("name", sorted(_QUADRATURES))
def test_it_is_self_adjoint(name):
    """A real diagonal weight is its own transpose."""
    space = _trace_space(_QUADRATURES[name])
    operator = InverseMetricOperator(space)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(np.shape(space.inner_product_weights))
    assert operator.is_adjointable
    np.testing.assert_array_equal(
        operator.apply(x), operator.apply_transpose(x),
    )


@pytest.mark.foundation
def test_no_solve_and_it_is_spelled_by_ABSENCE():
    """``is_invertible`` is ``False`` AND there is no ``inverse`` method.

    On a degenerate metric this is a Moore–Penrose pseudo-inverse, so
    ``G⁺G ≠ I`` and "invert me" is not a question it can answer.  The
    house spelling for that is the ABSENCE of the method
    (``TraceRestrictionOperator``), never a raising stub — so misuse is
    a static error rather than a runtime one.
    """
    operator = InverseMetricOperator(_trace_space(_QUADRATURES["product(4,4)"]))
    assert operator.is_invertible is False
    assert not hasattr(operator, "inverse")


@pytest.mark.foundation
def test_the_pseudo_inverse_masks_the_metrics_null_space():
    """`[M]` 192 of 384 ``product(4,4)`` trace rows carry ``G == 0`` exactly.

    On those rows the pseudo-inverse returns 0 — it does NOT divide.
    The ``level_symmetric`` control has zero such rows, which is exactly
    why it cannot witness this and the pair must both be exercised.
    """
    degenerate = _trace_space(_QUADRATURES["product(4,4)"])
    weights = np.asarray(degenerate.inner_product_weights)
    null_rows = np.abs(weights.ravel()) == 0.0
    assert null_rows.any(), "fixture no longer has a degenerate metric"

    ones = np.ones(weights.shape)
    out = InverseMetricOperator(degenerate).apply(ones).ravel()
    assert np.all(out[null_rows] == 0.0)
    assert np.all(np.isfinite(out)), "a masked row must not divide by zero"

    control = np.asarray(
        _trace_space(_QUADRATURES["level_symmetric(4)"]).inner_product_weights
    )
    assert not np.any(np.abs(control.ravel()) == 0.0), (
        "level_symmetric is the NONDEGENERATE control; if it grows a "
        "tangential ordinate the pair stops discriminating."
    )


@pytest.mark.foundation
@pytest.mark.parametrize("name", sorted(_QUADRATURES))
def test_it_inverts_the_metric_off_the_null_space(name):
    """``G⁺ G x == x`` wherever the metric is nonzero — the actual contract."""
    space = _trace_space(_QUADRATURES[name])
    weights = np.asarray(space.inner_product_weights)
    live = np.abs(weights.ravel()) > 0.0
    x = np.ones(weights.shape)
    round_trip = space.apply_metric(InverseMetricOperator(space).apply(x))
    np.testing.assert_allclose(round_trip.ravel()[live], x.ravel()[live])


@pytest.mark.foundation
def test_the_frame_can_now_spell_its_own_projector():
    r"""⭐ ``conjugate(G⁻¹)`` is the projector — spelled with the frame's own arrow.

    ``frame.conjugate(frame.gram_inverse)`` composes
    :math:`R \circ G^{-1} \circ M`, which for a Galerkin frame (test ==
    trial) is the orthogonal projector onto ``span(basis)``.  Asserted
    the way a projector is defined — idempotence — plus agreement with
    the frame's own ``project``/``reconstruction`` pair, so the operator
    spelling and the array spelling cannot drift.

    Until item 6.2c-ii the composition went through this adapter over the
    frame's probe SPACE; the arrow (:class:`~orpheus.numerics.frame.CrossGramInverse`)
    is the same :math:`G^{-1}` with the faces' ends as its type.
    """
    frame = GalerkinFrame(SphericalHarmonicBasis(L=3), lebedev_sphere(13))
    projector = frame.conjugate(frame.gram_inverse)

    rng = np.random.default_rng(0)
    # The field lives on the MEASURE's nodes (``table`` is ``(n_nodes, …)``),
    # which is the analysis face's domain.
    field = rng.standard_normal(frame.table.shape[0])

    once = projector.apply(field)
    twice = projector.apply(once)
    np.testing.assert_allclose(twice, once, atol=1e-12, err_msg="not idempotent")

    # Same answer as the hand-rolled chain the frame already supports.
    np.testing.assert_allclose(
        once, frame.reconstruction.apply(frame.project(field)), atol=1e-12,
    )

    # An exact SH frame at L=3 reproduces an in-span field exactly, so the
    # projector is not merely idempotent-by-being-zero.
    in_span = frame.reconstruction.apply(frame.project(field))
    np.testing.assert_allclose(
        projector.apply(in_span), in_span, atol=1e-12,
    )
    assert np.linalg.norm(once) > 1e-6, "degenerate fixture: nothing projected"
