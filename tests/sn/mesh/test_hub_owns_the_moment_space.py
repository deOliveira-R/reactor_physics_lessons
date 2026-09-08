r"""CS4c step 6 item 6.2b — the hub owns the harmonic-moment space.

**What this file gates.** :meth:`SNMesh.moment_space
<orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space>` is the ONE producer of
the moment space on a carrier: a keyed cache over ``(L, spatial_moments)``
whose every read returns the SAME object. The moment field family
(``from_mesh_and_L`` / ``zeros_for_mesh_and_L`` / ``space_on``), the boundary
leaf's carrier guard and the sweep's iterate wrap all READ it, so identity
is ``is`` — not a content comparison — and nothing is re-minted per call.

**Red-before, measured 2026-09-07 on the pre-carve tree** (``[M]``
``scratch/_step6/probes/p7_mintlaw.py`` / ``p19_mintlaw_post.py``):

* ``space_on(mesh) is space_on(mesh)`` was ``False`` on every read (a fresh
  ``TensorProductSpace`` per call; ``==`` was ``True`` — which is why an
  ``==`` gate here would be signature-tautological and is not written);
* the ``FunctionSpace.__mul__`` count per 2-D windowed SI solve was
  ``2·max_inner + 6`` — 12 at ``max_inner = 3``, 18 at 6, 30 at 12, 54 at
  24 — because the guard (``boundary.py``'s ``_apply_faces``) re-minted on
  every apply and the sweep re-minted on every iterate wrap (58 + 55 of 118
  products per solve at ``max_inner = 6``).

Each row below states which of those two facts it flips. ``pytestmark
foundation``: software invariants (object identity, a call count), no physics
claim.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from orpheus.numerics.space import FunctionSpace
from orpheus.sn.operators.boundary import SNBoundaryOperator
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from tests.sn.architecture.test_monomorphic_leaves import (
    _cart2d,
    _cylinder,
    _slab,
    _sphere,
    _two_region_fissile,
)

pytestmark = pytest.mark.foundation

_GEOMETRIES = {"slab": _slab, "sphere": _sphere, "cylinder": _cylinder, "cart2d": _cart2d}
_ORDERS = (0, 1)


# ═════════════════════════════════════════════════════════════════════════
# G2.4 — identity: one object per key, read by every consumer
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
@pytest.mark.parametrize("L", _ORDERS, ids=[f"L{n}" for n in _ORDERS])
def test_g2_4a_one_object_per_key_read_by_the_factory_and_the_guard(geometry, L):
    """``mesh.moment_space(L, w)`` returns ONE object; the field factory's
    space and the admission reference ``space_on`` ARE that object; a
    different key is a different object.

    Flips the first red-before: ``space_on(mesh) is space_on(mesh)`` was
    ``False`` on every read.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    width = sn_mesh.scheme.spatial_basis_per_axis
    first = sn_mesh.moment_space(L, spatial_moments=width)
    second = sn_mesh.moment_space(L, spatial_moments=width)
    if first is not second:
        pytest.fail(f"[{geometry} L={L}] the hub minted twice for one key")

    field = HarmonicMomentFlux.zeros_for_mesh_and_L(
        sn_mesh, L, spatial_moments=width,
    )
    if field.space is not first:
        pytest.fail(f"[{geometry} L={L}] the factory minted its own space instead of reading the hub's")
    if field.space_on(sn_mesh) is not first:
        pytest.fail(f"[{geometry} L={L}] the admission reference `space_on` is a re-mint, not the hub's object")
    if field.values.shape != first.shape:
        pytest.fail(f"[{geometry} L={L}] the zero field's shape {field.values.shape} is not the space's {first.shape}")

    other_order = sn_mesh.moment_space(L + 1, spatial_moments=width)
    if other_order is first:
        pytest.fail(f"[{geometry} L={L}] two truncation orders share one object")
    if other_order.shape == first.shape:
        pytest.fail("CONTROL INVALID: the two orders have the same shape, so the key test is vacuous")


def test_g2_4b_a_carrier_that_owns_no_moment_space_is_refused_by_name():
    """A carrier without ``moment_space`` cannot host a moment field, and the
    refusal names the hub verb (the same typed sentence the retired
    quadrature-carrier protocol raised, re-keyed on what the family now
    READS)."""

    class _NoHub:
        """No quadrature, no moment space — a transport MaterialMesh's shape."""
        ndim = 1

    with pytest.raises(TypeError, match="owns no moment space"):
        HarmonicMomentFlux.zeros_for_mesh_and_L(_NoHub(), 0)  # type: ignore[arg-type]


def _windowed_driver(max_inner: int):
    """ONE 2-D Cartesian windowed SI solve (the probe fixture the pre-carve
    counts were measured on): level_symmetric(4), 4×3 cells, ng=2, L=1,
    Jacobi schedule. Returns the Solution."""
    from orpheus.geometry import BC, CoordSystem
    from orpheus.geometry.mesh import Mesh2D
    from orpheus.numerics.quadrature import Quadrature
    from orpheus.sn.solver import solve_sn_fixed_source

    mat_map = np.zeros((4, 3), dtype=int)
    mat_map[2:, :] = 1
    mesh = Mesh2D(
        edges_x=np.array([0.0, 0.1, 0.35, 0.9, 1.6]),
        edges_y=np.array([0.0, 0.2, 0.5, 1.4]),
        mat_map=mat_map,
        coord=CoordSystem.CARTESIAN,
        bc_xmin=BC("reflective"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("reflective"), bc_ymax=BC("vacuum"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_sn_fixed_source(
            _two_region_fissile(), mesh, Quadrature.level_symmetric(sn_order=4),
            np.ones((24, 2, 4, 3)), scattering_order=1, inner_tol=1e-14,
            max_inner=max_inner, inner_schedule="jacobi",
        )


def test_g2_4c_the_boundary_guard_sees_the_hubs_object_on_a_windowed_solve(monkeypatch):
    """On a real windowed solve, every moment-interior composite the boundary
    leaf's ``_apply_faces`` receives rides the carrier's cached object — the
    space the guard compares against IS the space the sweep wrapped the
    iterate in (``is``, read through ``space_on``).

    ACTIVATION asserted: at least one moment operand must reach the guard,
    else the row is vacuous (the windowed arm feeds ``B_a`` the moment
    iterate — `[M]` 59/58/47 per solve pre-carve).
    """
    seen: list[tuple[Any, Any]] = []
    original = SNBoundaryOperator._apply_faces

    def spy(self, psi, method, *, rows=None):
        seen.append((self.sn_mesh, psi.interior))
        return original(self, psi, method, rows=rows)

    monkeypatch.setattr(SNBoundaryOperator, "_apply_faces", spy)
    _windowed_driver(max_inner=6)

    moment_operands = [
        (mesh, interior) for mesh, interior in seen
        if isinstance(interior, HarmonicMomentFlux)
    ]
    if not moment_operands:
        pytest.fail(
            f"ACTIVATION FAILED: the boundary leaf saw {len(seen)} operands "
            f"and none carried a moment interior — the row has no subject"
        )
    for mesh, interior in moment_operands:
        hub = mesh.moment_space(interior.L, spatial_moments=interior.spatial_moments)
        if interior.space is not hub:
            pytest.fail(
                "a moment operand reached the boundary guard on a space that "
                "is not the hub's object — the sweep (or a factory) re-minted"
            )
        if interior.space_on(mesh) is not hub:
            pytest.fail("the guard's reference `space_on` is not the hub's object")


# ═════════════════════════════════════════════════════════════════════════
# G2.5 — the memory assertion, RATE leg: the mint count does not scale
#        with the iteration budget
# ═════════════════════════════════════════════════════════════════════════

def _count_products(max_inner: int, monkeypatch) -> tuple[int, int]:
    """``(FunctionSpace.__mul__ activations, products carrying a dense slot)``
    during one windowed solve at ``max_inner``."""
    counts = {"mul": 0, "dense": 0}
    original = FunctionSpace.__mul__

    def counting(self, other):
        counts["mul"] += 1
        out = original(self, other)
        if out.inner_product_weights is not None:
            counts["dense"] += 1
        return out

    monkeypatch.setattr(FunctionSpace, "__mul__", counting)
    try:
        _windowed_driver(max_inner=max_inner)
    finally:
        monkeypatch.setattr(FunctionSpace, "__mul__", original)
    return counts["mul"], counts["dense"]


def test_g2_5_the_product_count_per_windowed_solve_is_invariant_in_max_inner(monkeypatch):
    """The number of ``*`` products minted during a windowed solve is the
    SAME at ``max_inner = 3`` and ``max_inner = 12``, and none carries a
    dense weights slot — both legs of the charter's memory assertion, in the
    form no fixture retuning can decay.

    Flips the second red-before: `[M]` pre-carve the counts were 12 and 30
    (``2·max_inner + 6``). The row also refuses the pre-carve LAW at 12 as a
    positive control on its own instrument: a count that still read 30
    would mean the spy is counting and the carve did not land.
    """
    at_3, dense_3 = _count_products(3, monkeypatch)
    at_12, dense_12 = _count_products(12, monkeypatch)
    if at_3 == 0 or at_12 == 0:
        pytest.fail("INSTRUMENT DEAD: no `*` product was counted on a windowed solve")
    if at_12 >= 2 * 12 + 6:
        pytest.fail(
            f"the pre-carve mint law still holds at max_inner=12 "
            f"({at_12} products; the law predicts 30) — the hub is not "
            f"being read"
        )
    if at_3 != at_12:
        pytest.fail(
            f"the `*` count scales with the iteration budget: {at_3} at "
            f"max_inner=3 vs {at_12} at 12 — a space is still re-minted per "
            f"iterate"
        )
    if dense_3 or dense_12:
        pytest.fail(
            f"a product carried a dense weights slot ({dense_3} / {dense_12}) "
            f"— the densifier is back"
        )
