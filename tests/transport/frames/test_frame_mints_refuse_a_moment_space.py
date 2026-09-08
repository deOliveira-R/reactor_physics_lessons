r"""The harmonic frame's mints refuse a MOMENT space — by name, on both doors
(CS4c step 6 item 6.2c-ii; the memo's gate P6, hazard H-6).

Until 6.2c-ii a moment space was axes-less, so ``moment_space_on`` refused it
through its *"must be axis-built"* check — a guard that silently lost its
subject the moment the head became axis-built (`[M]` the 6.2c verification
round: an axis-built moment space was ACCEPTED and returned a plausible-looking
product). Both doors now refuse through the same channel the blessed frame
chain reads — the leading axis's generator must be the QUADRATURE — so the
refusal's TYPE and MESSAGE are the ones asserted here, on the shipped moment
space of every ledger carrier.

Foundation mark: software invariant; no physics claim.
"""

from __future__ import annotations

import pytest

from orpheus.transport.frames import HarmonicFrame
from tests.sn.architecture.test_monomorphic_leaves import (
    _cart2d,
    _cylinder,
    _slab,
    _sphere,
)

pytestmark = pytest.mark.foundation

_GEOMETRIES = {"slab": _slab, "sphere": _sphere, "cylinder": _cylinder, "cart2d": _cart2d}


@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_P6_both_doors_refuse_a_moment_space_naming_the_missing_quadrature(geometry: str) -> None:
    sn_mesh = _GEOMETRIES[geometry]()
    L = 1
    moment = sn_mesh.moment_space(L)
    assert moment.axes is not None, "the moment space IS axis-built — the old axes-less refusal cannot catch it"
    frame = HarmonicFrame.for_space(sn_mesh.angular_bulk_space, L)
    with pytest.raises(ValueError, match="HarmonicFrame.moment_space_on needs the generating Quadrature"):
        frame.moment_space_on(moment)
    with pytest.raises(ValueError, match="HarmonicFrame.for_space needs the generating Quadrature"):
        HarmonicFrame.for_space(moment, L)
    # the positive leg: the angular space passes both doors
    assert frame.moment_space_on(sn_mesh.angular_bulk_space) == moment
    assert HarmonicFrame.for_space(sn_mesh.angular_bulk_space, L) is frame
