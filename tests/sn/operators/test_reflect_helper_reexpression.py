r"""Pre-carve anchor for the CS4c step-6 reflect-verb retirement (F4 / item 6.5).

**What this file is.**  Step 6 retires
:meth:`~orpheus.sn.operators.boundary.SNBoundaryOperator.reflect_into_inflow`
and its mutating façade ``reflect_inflow_inplace`` (`[M]` **0** production
callers each), and re-expresses the sweep-tier helper
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
    r"""``reflect_inflow_inplace`` (assignment) ≡ *zero the inflow, then add
    through the jacobi-upper mask*, ``np.array_equal`` on the WHOLE trace.

    Bit-identity is the right tier here and is EARNED, not assumed: neither
    body reorders a reduction — both write ``(B·ψ)[f][inflow]`` computed by the
    same ``_reflect_trace`` call — so a tolerance would admit exactly the class
    of bug the row exists to catch (lessons §5: bit-exactness is earned per
    law).

    ⛔ ACTIVATION, asserted in the row: the buffer's inflow slots must be
    NON-ZERO before the call.  On a zero-inflow buffer the two bodies are
    trivially equal and the row proves nothing.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    mask = _full_inflow_mask(operator).upper
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

    # OLD body — the whole-face ASSIGNMENT the helper carries today.
    operator.reflect_inflow_inplace(old)

    # NEW body — zero the inflow slots, then ADD through the full-inflow mask.
    for face in new.layout.faces:
        new.face_view(face)[trace.inflow_indices_for_face(face)] = 0.0
    mask.reflect_rows_inplace(new, tuple(new.layout.faces))

    if not np.array_equal(old.values, new.values):
        pytest.fail(
            f"[{geometry}] the re-expressed helper body is not bit-identical: "
            f"max|Δ| = {float(np.abs(old.values - new.values).max()):.6e}"
        )


@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_g5_1b_dropping_the_zeroing_is_the_positive_control(geometry):
    r"""The CONTROL for the row above: omit the zeroing and the two bodies MUST
    differ at O(1).

    Without this leg, ``test_g5_1`` is compatible with a buffer whose inflow
    happened to be zero, or with an accumulation that silently behaved like an
    assignment — a green reading alone cannot discriminate loaded from blind
    (``vv`` #19).  MEASURED spread: 1.078e+00 … 2.592e+00 across the four
    geometries.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    mask = _full_inflow_mask(operator).upper

    old = _random_trace(sn_mesh, seed=101)
    unzeroed = _random_trace(sn_mesh, seed=101)
    operator.reflect_inflow_inplace(old)
    mask.reflect_rows_inplace(unzeroed, tuple(unzeroed.layout.faces))

    delta = float(np.abs(old.values - unzeroed.values).max())
    if delta < 1e-3:
        pytest.fail(
            f"[{geometry}] CONTROL FAILED: dropping the zeroing moved the "
            f"answer by only {delta:.3e} — the bit-identity row above cannot "
            f"be crediting the zeroing step"
        )


# ═════════════════════════════════════════════════════════════════════════
# G5.4-pre — the guard the retirement ORPHANS, recorded before it happens
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_g5_4pre_the_unknown_face_refusal_lives_only_on_the_RETIRING_verb(geometry):
    r"""RECORD: today a bogus face RAISES through ``reflect_into_inflow`` and is
    SILENTLY DROPPED by ``reflect_rows_inplace``.

    ``_reflect_trace`` refuses an unknown face (``boundary.py:596-601``,
    message *"… are not boundary faces of this mesh"*), and its only public
    route is the verb item 6.5 retires: ``reflect_rows_inplace`` filters
    ``faces`` against ``self.rows`` BEFORE calling ``_reflect_trace``
    (``:1252-1256``), and ``_apply_faces`` always passes ``faces=None``.

    ⟹ retiring both verbs makes that ``ValueError`` **reachable from no public
    surface**, and its sole witness
    (``test_sn_boundary_operator.py::TestFaceRestrictedReflect::test_unknown_face_raises``)
    has no successor.  The carve owes an explicit ruling — move the refusal
    into ``reflect_rows_inplace``, or delete the guard with its witness and say
    so (``lessons`` §1: retiring a guard makes the replacement's teeth
    NET-NEW, never migrated).

    ⚠ RECORD, not THEOREM.  When 6.5 lands, this row is expected to change
    shape; do not "fix" it by relaxing the silent-drop half.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    mask = _full_inflow_mask(operator).upper
    buffer = _random_trace(sn_mesh, seed=7)

    with pytest.raises(ValueError, match="boundary faces"):
        operator.reflect_into_inflow(buffer, faces=["bogus_face"])

    before = buffer.values.copy()
    mask.reflect_rows_inplace(buffer, ("bogus_face",))
    if not np.array_equal(before, buffer.values):
        pytest.fail(
            f"[{geometry}] reflect_rows_inplace acted on a bogus face — the "
            f"recorded contrast (raise vs silent drop) has changed"
        )
