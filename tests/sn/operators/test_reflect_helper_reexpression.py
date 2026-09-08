r"""Pre-carve anchor for the CS4c step-6 reflect-verb retirement (F4 / item 6.5).

**What this file is.**  CS4c step 6 item 6.5 (LANDED 2026-09-07) retired
``SNBoundaryOperator.reflect_into_inflow`` (a literal, not a role — the
symbol is gone) and its mutating façade ``reflect_inflow_inplace`` (`[M]`
**0** production
callers each), and re-expressed the sweep-tier helper
``tests/sn/_test_helpers.py::reflect_outflow_into_inflow`` on the live G-S
verb :meth:`~orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace`.

The two verbs are NOT interchangeable, and the plan row does not say so:

* ``reflect_inflow_inplace`` is a whole-face **ASSIGNMENT**
  ``bf[f][inflow] = (B·bf)[f][inflow]`` (``boundary.py:816-828``);
* ``reflect_rows_inplace`` is **ADDITIVE on ``rows``**
  ``bf[f][rows] += (B·bf)[f][rows]`` (``boundary.py:1240-1243``) — because a
  forward-substitution row completes ``z_in = y_row + (Bz)_row`` on top of a
  seed.

So the re-expression is *zero the inflow slots, then add through a full-inflow
mask*, and this file pins BOTH halves of that on unmodified production:

1. the full-inflow mask ALREADY EXISTS — no new production factory is needed;
2. the zero-then-add body is bit-identical to the assignment body on a buffer
   whose inflow slots are NON-ZERO.

⛔ **The non-zero-inflow precondition is asserted in the row, not assumed.**
On a zero-inflow buffer assignment ≡ additive and the gate is inert by
construction (``plan-authoring`` §6c; the 2026-09-05 "0 == 0" row).

**Activation evidence (measured 2026-09-07 at ``b889089e``).**  Dropping the
zeroing — the one step that makes the two bodies agree — reddens every row at
O(1):

==========  =============================  =============================
geometry    zero-then-add (the claim)      CONTROL: add without zeroing
==========  =============================  =============================
slab        ``array_equal``, max|Δ| 0.0    **differs, max|Δ| 1.766e+00**
sphere      ``array_equal``, max|Δ| 0.0    **differs, max|Δ| 1.078e+00**
cylinder    ``array_equal``, max|Δ| 0.0    **differs, max|Δ| 1.457e+00**
cart2d      ``array_equal``, max|Δ| 0.0    **differs, max|Δ| 2.592e+00**
==========  =============================  =============================

**Fixture scope, stated rather than left to inference (``vv`` #20).**  The
four ledger meshes span {slab, sphere, cylinder, cart2d} × {reflective,
vacuum}, which is the whole law family reachable here: `[M]` ``SNMesh``
accepts only reflective/vacuum face laws, so white / albedo are structurally
ABSENT from this surface (they are constructed on ``B_b`` directly), and the
helper's own docstring records that `[M]` **0** of its consumer sites ever
passes a ψ½ ray.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.sn.loss_representation.sweep_schedule import SweepSchedule
from orpheus.sn.operators.boundary import SNBoundaryOperator
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.fields.angular_flux import AngularFlux
from orpheus.transport.full_field import FullField
from tests.sn._test_helpers import reflect_outflow_into_inflow
from tests.sn.architecture.test_monomorphic_leaves import (
    _cart2d,
    _cylinder,
    _slab,
    _sphere,
)

pytestmark = pytest.mark.foundation

_GEOMETRIES = {"slab": _slab, "sphere": _sphere, "cylinder": _cylinder, "cart2d": _cart2d}


def _random_trace(sn_mesh, *, seed: int) -> AngularBoundaryFlux:
    """A fixed-seed random FLUX trace, filled on every face.

    Built directly as an ``AngularBoundaryFlux`` rather than read off a
    composite's role-erased ``boundary`` slot: both verbs under test declare
    the flux role in their signature, and threading the erased slot would ask
    the reader (and the type checker) to take the role on trust.
    """
    trace = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
    rng = np.random.default_rng([seed, 7])
    for face in trace.layout.faces:
        view = trace.face_view(face)
        view[...] = rng.standard_normal(view.shape)
    return trace


def _full_inflow_mask(operator: SNBoundaryOperator):
    """The full-inflow ``SNMaskedBoundaryOperator`` — production's OWN factory.

    ``SweepSchedule.jacobi`` builds ONE octant group with ``reflect_faces=()``,
    so ``lower_inflow_rows`` returns ``{}`` and ``SNBoundaryOperator.split``
    puts EVERY inflow row of EVERY ``_face_laws`` face into ``upper``.  Nothing
    new is minted here; the mask is read off the shipped split.
    """
    mesh = operator.sn_mesh
    schedule = SweepSchedule.jacobi(mesh.ndim, mesh.quad.octants)
    return operator.split(schedule)


# ═════════════════════════════════════════════════════════════════════════
# G5.2 — the full-inflow mask already exists (no new production surface)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_g5_2_the_jacobi_split_upper_is_the_full_inflow_mask(geometry):
    r"""``B.split(jacobi).upper.rows`` IS the per-face full inflow row set, and
    ``.lower`` is empty.

    This is the row that answers item 6.5's open question — *does the carve
    need a new ``full-inflow`` mask factory?* — with **no**.  If a future
    schedule change makes ``jacobi`` reflect a face, ``lower`` stops being
    empty and this row reds, which is exactly when the helper's re-expression
    would silently start dropping rows.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    parts = _full_inflow_mask(operator)
    trace = sn_mesh.angular_trace

    want = {face: trace.inflow_indices_for_face(face) for face in operator._face_laws}
    if not want:
        pytest.fail(f"[{geometry}] the mesh declares no boundary face laws — row vacuous")
    if set(parts.upper.rows) != set(want):
        pytest.fail(
            f"[{geometry}] the jacobi upper mask covers faces "
            f"{sorted(parts.upper.rows)}, the mesh's law faces are {sorted(want)}"
        )
    for face, rows in want.items():
        got = np.asarray(parts.upper.rows[face])
        if not np.array_equal(got, np.asarray(rows)):
            pytest.fail(
                f"[{geometry}] face {face!r}: the jacobi upper mask is not the "
                f"full inflow row set (got {got.tolist()}, want {np.asarray(rows).tolist()})"
            )
        if got.size == 0:
            pytest.fail(f"[{geometry}] face {face!r} has an EMPTY inflow row set — row vacuous")
    for face, rows in parts.lower.rows.items():
        if np.asarray(rows).size:
            pytest.fail(
                f"[{geometry}] the jacobi LOWER half is non-empty on face "
                f"{face!r} — the schedule now reflects in-sweep and the "
                f"'upper == full inflow' identity no longer holds"
            )


# ═════════════════════════════════════════════════════════════════════════
# G5.1 — the re-expressed helper body is bit-identical to today's
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_g5_1_zero_then_add_reproduces_the_assignment_bit_for_bit(geometry):
    r"""The helper (*zero the inflow rows, then add through the jacobi-upper
    mask*) ≡ the operator's PUBLIC forward action ``(B·ψ).boundary`` on the
    inflow rows, ``np.array_equal`` per face — and it leaves the outflow rows
    untouched.

    Until CS4c step 6 item 6.5 the reference leg was the whole-face
    ASSIGNMENT verb ``reflect_inflow_inplace`` (`[M]` bit-identical on 4/4
    geometries); that verb retired with 0 production callers and the
    reference is now ``apply`` on a zero-bulk composite — two routes into
    ``_reflect_trace`` through DIFFERENT verbs (the lift and the masked
    additive), so the row still discriminates the helper's wiring (the
    zeroing, the mask, the face loop) even though the law is shared.
    Bit-identity is the right tier here and is EARNED, not assumed: neither
    body reorders a reduction — both write ``(B·ψ)[f][inflow]`` computed by
    the same ``_reflect_trace`` call — so a tolerance would admit exactly the
    class of bug the row exists to catch (lessons §5: bit-exactness is earned
    per law).

    ⛔ ACTIVATION, asserted in the row: the buffer's inflow slots must be
    NON-ZERO before the call.  On a zero-inflow buffer the two bodies are
    trivially equal and the row proves nothing.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    trace = sn_mesh.angular_trace

    old = _random_trace(sn_mesh, seed=101)
    new = _random_trace(sn_mesh, seed=101)
    if not np.array_equal(old.values, new.values):
        pytest.fail("the two fixed-seed buffers differ — the comparison is not old-vs-new")

    inflow_magnitudes = {
        face: float(np.abs(old.face_view(face)[trace.inflow_indices_for_face(face)]).max())
        for face in old.layout.faces
    }
    if min(inflow_magnitudes.values(), default=0.0) <= 0.0:
        pytest.fail(
            f"[{geometry}] ACTIVATION FAILED: some face's inflow slots are all "
            f"zero before the call, so assignment and accumulation agree "
            f"trivially there — {inflow_magnitudes}"
        )

    # The INDEPENDENT spelling — the operator's public forward action on a
    # zero-bulk composite: ``(B·ψ).boundary``'s inflow rows are ``B·ψ.outflow``
    # (the retired assignment verb was ``apply`` without the zero-bulk
    # carrier; since item 6.5 this is the reference leg).
    probe = FullField(
        interior=AngularFlux(
            values=np.zeros(sn_mesh.angular_trial_space.shape),
            space=sn_mesh.angular_trial_space,
        ),
        boundary=old,
    )
    via_apply = operator.apply(probe).boundary

    # The helper — zero the inflow rows, then ADD through the full-inflow mask.
    reflect_outflow_into_inflow(new, sn_mesh)

    for face in new.layout.faces:
        rows = trace.inflow_indices_for_face(face)
        if not np.array_equal(via_apply.face_view(face)[rows], new.face_view(face)[rows]):
            pytest.fail(
                f"[{geometry}] face {face!r}: the helper's inflow rows are not "
                f"bit-identical to the operator's forward action: max|Δ| = "
                f"{float(np.abs(via_apply.face_view(face)[rows] - new.face_view(face)[rows]).max()):.6e}"
            )
        outflow = np.setdiff1d(np.arange(new.face_view(face).shape[0]), rows)
        if outflow.size and not np.array_equal(new.face_view(face)[outflow], old.face_view(face)[outflow]):
            pytest.fail(f"[{geometry}] face {face!r}: the helper touched OUTFLOW rows")


@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_g5_1b_dropping_the_zeroing_is_the_positive_control(geometry):
    r"""The CONTROL for the row above: omit the zeroing and the two bodies MUST
    differ at O(1).

    Without this leg, ``test_g5_1`` is compatible with a buffer whose inflow
    happened to be zero, or with an accumulation that silently behaved like an
    assignment — a green reading alone cannot discriminate loaded from blind
    (``vv`` #19).  MEASURED spread at seed 101: 1.078e+00 … 2.592e+00 across the four
    geometries — ONE draw; `[M]` an archivist's 40-seed × 4-geometry sweep
    (2026-09-07) reads 0.515 … 5.198, with the bit-identity row above at
    40/40, so the floor asserted here (1e-3) is what is draw-stable, not the
    spread.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    mask = _full_inflow_mask(operator).upper

    old = _random_trace(sn_mesh, seed=101)
    unzeroed = _random_trace(sn_mesh, seed=101)
    via_apply = operator.apply(
        FullField(
            interior=AngularFlux(
                values=np.zeros(sn_mesh.angular_trial_space.shape),
                space=sn_mesh.angular_trial_space,
            ),
            boundary=old,
        )
    ).boundary
    mask.reflect_rows_inplace(unzeroed, tuple(unzeroed.layout.faces))

    trace = sn_mesh.angular_trace
    delta = max(
        float(np.abs(
            via_apply.face_view(face)[trace.inflow_indices_for_face(face)]
            - unzeroed.face_view(face)[trace.inflow_indices_for_face(face)]
        ).max())
        for face in unzeroed.layout.faces
    )
    if delta < 1e-3:
        pytest.fail(
            f"[{geometry}] CONTROL FAILED: dropping the zeroing moved the "
            f"answer by only {delta:.3e} — the bit-identity row above cannot "
            f"be crediting the zeroing step"
        )


# ═════════════════════════════════════════════════════════════════════════
# G5.4 — the unknown-face refusal, on the live verb (moved at item 6.5)
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_g5_4_the_unknown_face_refusal_lives_on_the_live_verb(geometry):
    r"""A bogus face RAISES through ``reflect_rows_inplace`` — the refusal
    MOVED into the live verb at CS4c step 6 item 6.5 (ruling O-5).

    Until then this row was a RECORD of the opposite: ``reflect_rows_inplace``
    filtered ``faces`` against its rows BEFORE calling ``_reflect_trace`` and
    was SILENT on a bogus face (4/4 geometries), while the retired trace-only
    ``reflect_into_inflow`` raised through ``_reflect_trace``'s guard — a
    guard reachable from no other public surface, retired with its callers.
    Its message vocabulary ("… are not boundary faces of this mesh") is kept
    on the live verb, and the buffer is untouched when it raises.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    mask = _full_inflow_mask(operator).upper
    buffer = _random_trace(sn_mesh, seed=7)
    before = buffer.values.copy()
    with pytest.raises(ValueError, match="boundary faces"):
        mask.reflect_rows_inplace(buffer, ("bogus_face",))
    if not np.array_equal(before, buffer.values):
        pytest.fail(f"[{geometry}] the refused call mutated the buffer")
