r"""One moment space, two owners — the hub and the frame agree STRUCTURALLY
(CS4c step 6 item 6.2c-ii; the memo's gates P5, P7 and P10).

Since item 6.2b the carrier (:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh`)
owns the moment space and every moment field / admission guard on it holds
ONE object per ``(L, width)``; since item 6.2c-ii the head it composes is the
frame's Parseval-dressed ``basis_space`` (ruling R-6.2c-1), and the frame's
own derivation, :meth:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame.moment_space_on`,
builds the same product from the angular space's own axes. The two owners
cannot share an object (`[M]` ruling O-5: the frame has no carrier to reach),
so the reachable target — and the one asserted here — is structural equality
ACROSS owners and ``is``-identity WITHIN each.

* **P5** — hub ``==`` frame, ``is`` within each, on every ledger carrier;
* **P7** — the three product mechanisms are ONE space by identity — the
  hub's ``head * bulk``, ``FunctionSpace.of_axes(head_axis, *bulk.axes)``,
  and the frame's derivation — while their NAMES differ (`[M]` the ``*``
  product keeps the head's family tag, the ``of_axes`` mint is a pure
  digest), so the gate asserts identity and NOT the name;
* **P10** — the structural / memory leg: on every (geometry × L) row the
  moment space is axis-built, ``inner_product_weights is None``, the metric
  is derived (no object on a DIAGONAL-Gram frame), and no reachable array
  has the product's size.

Foundation mark: software invariants (identity, structure); no physics claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.frame import GramStructure
from orpheus.numerics.metric import FactoredMetric
from orpheus.numerics.space import FunctionSpace, TensorProductSpace
from orpheus.transport.frames import HarmonicFrame
from tests.sn.architecture.test_monomorphic_leaves import (
    _cart2d,
    _cylinder,
    _slab,
    _sphere,
)

pytestmark = pytest.mark.foundation

_GEOMETRIES = {"slab": _slab, "sphere": _sphere, "cylinder": _cylinder, "cart2d": _cart2d}
_ORDERS = [0, 1, 2]


@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
@pytest.mark.parametrize("L", _ORDERS)
def test_P5_the_hub_and_the_frame_derive_one_space_by_equality_and_each_owner_by_identity(
    geometry: str, L: int,
) -> None:
    sn_mesh = _GEOMETRIES[geometry]()
    hub = sn_mesh.moment_space(L)
    frame = HarmonicFrame.for_space(sn_mesh.angular_bulk_space, L)
    derived = frame.moment_space_on(sn_mesh.angular_bulk_space)
    assert hub == derived and derived == hub, "two owners, ONE space (structural equality)"
    assert hash(hub) == hash(derived)
    # is-identity WITHIN each owner
    assert sn_mesh.moment_space(L) is hub
    assert frame.flux_analysis_on(sn_mesh.angular_bulk_space).codomain == hub
    # the head both owners hold is the frame's Parseval-dressed one
    assert isinstance(hub, TensorProductSpace)
    assert hub.factors[0] == frame.basis_space
    assert hub.factors[0] != frame.basis.space, "the continuum head is another space"


@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
@pytest.mark.parametrize("L", _ORDERS)
def test_P7_the_three_product_mechanisms_are_one_space_by_identity_not_by_name(
    geometry: str, L: int,
) -> None:
    sn_mesh = _GEOMETRIES[geometry]()
    hub = sn_mesh.moment_space(L)
    frame = sn_mesh.quad.angular_frame(L)
    head = frame.basis_space
    bulk = sn_mesh.bulk_space
    assert head.axes is not None and bulk.axes is not None
    star = head * bulk
    of_axes = FunctionSpace.of_axes(*head.axes, *bulk.axes)
    derived = HarmonicFrame.from_galerkin(frame).moment_space_on(sn_mesh.angular_bulk_space)
    assert hub == star == of_axes == derived
    assert len({hub, star, of_axes, derived}) == 1
    # the names DIFFER — identity is structural, never nominal
    assert star.name == hub.name
    assert of_axes.name != hub.name, "the of_axes mint is a pure digest; the * product keeps the head's tag"
    # the DENSE-Gram overlay (if any) rides both mechanisms alike
    if frame.discrete_gram_structure is GramStructure.DENSE:
        assert isinstance(star.metric, FactoredMetric)
    else:
        assert star.metric is None
    x = np.random.default_rng(L).standard_normal(hub.shape)
    np.testing.assert_array_equal(star.apply_metric(x), of_axes.apply_metric(x))
    np.testing.assert_array_equal(star.apply_metric(x), derived.apply_metric(x))


@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
@pytest.mark.parametrize("L", _ORDERS)
def test_P10_the_moment_space_is_axis_built_and_never_densified(geometry: str, L: int) -> None:
    sn_mesh = _GEOMETRIES[geometry]()
    space = sn_mesh.moment_space(L)
    frame = sn_mesh.quad.angular_frame(L)
    assert space.axes is not None, "the moment space is axis-built"
    assert space.inner_product_weights is None
    if frame.discrete_gram_structure is GramStructure.DENSE:
        assert isinstance(space.metric, FactoredMetric)
        forms = [f for _, f in space.metric.entries]
        assert forms[0] is not None and all(f is None for f in forms[1:])
    else:
        assert space.metric is None
    n = int(np.prod(space.shape))
    reachable = [ax.weights for ax in space.axes if ax.weights is not None]
    assert all(w.size < n for w in reachable), "no reachable array is state-sized"
    assert sum(w.nbytes for w in reachable) < 8 * n
