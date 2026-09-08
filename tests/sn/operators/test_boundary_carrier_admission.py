r"""Pre-carve anchor for the CS4c step-6 R6 carrier guard (§8 row 6 / F1).

**What this file is.**  Step 6 item 6.3 replaces ``SNBoundaryOperator``'s
unguarded ``psi.interior`` read (``sn/operators/boundary.py:714``) with a
typed ``FullField.require_member`` parse, mirroring the SHAPE of
:meth:`orpheus.transport.radial_characteristic_field.RadialCharacteristicField.require_member`
(``TypeError`` for a foreign carrier, ``ValueError`` for a space-content
mismatch) — **keeping today's ``space_on`` semantics**, i.e. comparing the
operand's interior space against the space that operand's OWN class would mint
on the operator's mesh, NOT against the operator's bound end.

That distinction is the whole design fork, and it has a measured consequence
this file pins: ``B_a`` is the one gain still fed the MOMENT iterate.  On every
2-D windowed SI / Gauss-Seidel solve, ``_apply_faces`` receives
``HarmonicMomentFlux``-interior composites whose space is NOT the bound end's
angular interior; today's guard ACCEPTS them, and after 6.3 it MUST STILL
accept them.  This is the ``plan-authoring`` §8 acceptance witness — landed
BEFORE the carve, green on unmodified production, so the carve inherits a
measured baseline rather than a hoped-for one.

**Activation evidence (measured 2026-09-07 at ``b889089e``, in-process
monkeypatch of ``SNBoundaryOperator._apply_faces``).**  The control installs
the *other* candidate semantics — the bound-end (``admit_composite``) shape —
and the two carrier families partition exactly:

=================================  ==========================  ==========================
carrier                            TODAY (``space_on``)        CONTROL (bound-end guard)
=================================  ==========================  ==========================
``HarmonicMomentFlux`` interior    **accepted 8 / 8**          **refused 8 / 8**
(4 geometries × L ∈ {0, 1})
``AngularFlux`` interior           accepted 4 / 4              accepted 4 / 4
(4 geometries)
=================================  ==========================  ==========================

The angular rows are the ANTI-DUD leg (lessons L31): they prove the control is
not "refuse everything", so the moment rows' green is attributable.  Output is
non-vacuous — ``|out.boundary|max`` runs **8.58e-01 … 2.93e+00** across the
eight moment rows.

⚠ **Not 2-D-only.**  The explorer's 59 / 58 / 47 figures are production
TRAFFIC on the windowed arm; the moment CARRIER is constructible on slab,
sphere, cylinder and cart2d alike, with no solve.  Building it by hand gives a
4× wider witness in milliseconds — and a §6c red-before that a windowed
end-to-end solve could not make cheaper.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.spaces.full_field_space import FullFieldSpace
from orpheus.sn.operators.boundary import SNBoundaryOperator
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.transport.timed_full_field import TimedFullField
from tests.sn.architecture.test_monomorphic_leaves import (
    _cart2d,
    _cylinder,
    _random_composite,
    _slab,
    _sphere,
)

pytestmark = pytest.mark.foundation

_GEOMETRIES = {"slab": _slab, "sphere": _sphere, "cylinder": _cylinder, "cart2d": _cart2d}
_ORDERS = (0, 1)


def _bound_end_of(operator: SNBoundaryOperator) -> FullFieldSpace:
    """``B_a``'s bound domain, parsed as the BLOCK-BEARING composite it must be.

    The narrowing is a real precondition, not typing ceremony: the rows below
    read ``.interior_space`` off it, and a domain that is not a
    ``FullFieldSpace`` carries no blocks — in which case the whole
    ``space_on``-vs-bound-end contrast this file measures is unreadable and
    the row must say so rather than crash one line later.
    """
    domain = operator.domain
    if not isinstance(domain, FullFieldSpace):
        pytest.fail(
            f"B_a's domain is {type(domain).__name__}, not a block-bearing "
            f"FullFieldSpace — the bound-end half of this file's contrast "
            f"cannot be read"
        )
    return domain


def _moment_composite(sn_mesh, L: int, *, seed: int = 4) -> TimedFullField:
    """A ``FullField`` whose INTERIOR is a ``HarmonicMomentFlux`` at order ``L``.

    Built through the production factory (``zeros_for_mesh_and_L``) so the
    interior's space is production's own mint, then filled with a fixed-seed
    random state in BOTH blocks — a zero trace would null ``B`` entirely and
    make the row measure a smaller operator than it names.
    """
    interior = HarmonicMomentFlux.zeros_for_mesh_and_L(
        sn_mesh, L, spatial_moments=sn_mesh.scheme.spatial_basis_per_axis,
    )
    interior.values[...] = np.random.default_rng(seed).standard_normal(interior.values.shape)
    boundary = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
    for face in boundary.layout.faces:
        view = boundary.face_view(face)
        view[...] = np.random.default_rng(seed + 5).standard_normal(view.shape)
    return TimedFullField(interior=interior, boundary=boundary, _history=(), history_depth=0)


# ═════════════════════════════════════════════════════════════════════════
# G3.1a — the MOMENT carrier is accepted, and the two candidate semantics
#         genuinely disagree on it
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
@pytest.mark.parametrize("L", _ORDERS, ids=[f"L{n}" for n in _ORDERS])
def test_g3_1a_b_a_accepts_a_moment_interior_composite(geometry, L):
    r"""``B_a.apply`` accepts a ``HarmonicMomentFlux``-interior composite and
    returns a non-zero reflected trace.

    The two PRECONDITIONS are asserted IN the row (lessons L58d — a row that
    omits its discriminator degrades silently the day the two spaces
    coincide):

    1. ``interior.space == interior.space_on(mesh)``  — today's guard's
       comparison, which must hold or the row is testing the refusal path;
    2. ``interior.space != operator.domain.interior_space`` — the bound end's
       interior, which must DIFFER or the row cannot distinguish ``space_on``
       semantics from ``admit_composite`` semantics and is vacuous.

    ⛔ If (2) ever stops holding, this row has lost its subject: say so rather
    than relaxing it — the moment iterate would then BE the bound end, which
    is R18's B reshape having landed.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    composite = _moment_composite(sn_mesh, L)

    interior_space = composite.interior.space
    if interior_space != composite.interior.space_on(sn_mesh):
        pytest.fail(
            f"[{geometry} L={L}] PRECONDITION 1 failed: the moment interior's "
            f"space is not its own mint on this mesh — the row is exercising "
            f"the refusal path, not the acceptance path"
        )
    bound_end = _bound_end_of(operator)
    if interior_space == bound_end.interior_space:
        pytest.fail(
            f"[{geometry} L={L}] PRECONDITION 2 failed: the moment interior's "
            f"space EQUALS the bound end's interior, so `space_on` and "
            f"bound-end admission agree here and the row cannot discriminate "
            f"the two candidate guard semantics"
        )

    out = operator.apply(composite)
    magnitude = float(np.abs(out.boundary.values).max())
    if not magnitude > 0.0:
        pytest.fail(
            f"[{geometry} L={L}] B_a returned an all-zero trace on a "
            f"non-zero-trace moment composite — the row is vacuous "
            f"(`|out.boundary|max` = {magnitude:.3e})"
        )


# ═════════════════════════════════════════════════════════════════════════
# G3.1b — the ANTI-DUD leg: the angular carrier is accepted too
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_g3_1b_b_a_accepts_the_angular_interior_composite(geometry):
    r"""``B_a.apply`` accepts an ``AngularFlux``-interior composite — the leg
    that makes the moment rows attributable.

    Without it, a control that refuses EVERY carrier would look like a clean
    partition (lessons L31: pair every refusal claim with a positive control,
    else an arm that refuses everything also "passes").  MEASURED: the
    bound-end control leaves these 4 rows green while reddening all 8 moment
    rows.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    composite = _random_composite(sn_mesh, seed=3)

    bound_end = _bound_end_of(operator)
    if composite.interior.space != bound_end.interior_space:
        pytest.fail(
            f"[{geometry}] CONTROL INVALID: the angular interior's space does "
            f"NOT equal the bound end's interior, so this row no longer "
            f"contrasts with the moment rows"
        )

    out = operator.apply(composite)
    if not float(np.abs(out.boundary.values).max()) > 0.0:
        pytest.fail(f"[{geometry}] the angular row returned an all-zero trace — vacuous")


# ═════════════════════════════════════════════════════════════════════════
# G3.2 — the two clauses of the landed parse are ORDERED (post-carve rows;
#        item 6.3 landed 2026-09-07 — the R6 RECORD row this file carried,
#        test_g3_1c, was deleted with the ledger's _R6_XFAIL in that commit)
# ═════════════════════════════════════════════════════════════════════════

class _AlienCarrier:
    """A carrier no leaf's arrow accepts, and that is all it is."""


_CARRIER_FRAGMENT = "expected FullField"
_CONTENT_FRAGMENT = "space-content"


@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
def test_g3_2a_an_alien_carrier_is_refused_by_the_carrier_clause_alone(geometry):
    r"""``B_a.apply(<alien>)`` raises a ``TypeError`` naming the operator and
    the carrier it wanted — and NOT the content fragment: the carrier clause
    fires FIRST, before any ``.interior`` read (until item 6.3 this leaked a
    raw ``AttributeError: 'X' object has no attribute 'interior'``).

    Asserting the ABSENT fragment is what pins the clause ORDER (`L43c`): a
    parse that read the content first would raise its own error, or the
    old AttributeError, before naming the carrier.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    operator = SNBoundaryOperator(sn_mesh)
    with pytest.raises(TypeError) as excinfo:
        operator.apply(_AlienCarrier())  # type: ignore[arg-type]
    message = str(excinfo.value)
    if "SNBoundaryOperator.apply" not in message:
        pytest.fail(f"[{geometry}] the refusal does not name the refusing surface: {message!r}")
    if _CARRIER_FRAGMENT not in message:
        pytest.fail(f"[{geometry}] the refusal does not name the expected carrier: {message!r}")
    if _CONTENT_FRAGMENT in message:
        pytest.fail(
            f"[{geometry}] the CONTENT clause fired on an alien carrier — the "
            f"clause order is wrong: {message!r}"
        )


def test_g3_2b_a_content_mismatch_is_refused_by_the_content_clause_alone():
    r"""On the width-stretched slab composite ``TestO13BoundaryOperator`` uses
    (the ONE pre-existing pin of ``B_a``'s content refusal), the parse raises
    a ``ValueError`` carrying the ``space-content`` vocabulary and NOT the
    carrier fragment — the operand IS a ``FullField``, so the carrier clause
    must pass it through to the content clause.
    """
    from tests.sn.operators.test_space_content_witnesses import _composite, _slab as _slab_of_width

    sn_mesh = _slab_of_width()
    stretched = _slab_of_width(width=2.0)
    operator = SNBoundaryOperator(sn_mesh)
    with pytest.raises(ValueError) as excinfo:
        operator.apply(_composite(stretched))
    message = str(excinfo.value)
    if _CONTENT_FRAGMENT not in message:
        pytest.fail(f"the content refusal lost its vocabulary: {message!r}")
    if _CARRIER_FRAGMENT in message:
        pytest.fail(f"the CARRIER clause fired on a genuine FullField: {message!r}")
    if "SNBoundaryOperator.apply" not in message:
        pytest.fail(f"the content refusal does not name the refusing surface: {message!r}")


# ═════════════════════════════════════════════════════════════════════════
# G3.4 — the parse is ONE body with FIVE call sites (RECORD: a census)
# ═════════════════════════════════════════════════════════════════════════

_EXPECTED_CALL_SITES = 5     # L apply/transpose, LC apply/transpose, B_a _apply_faces


def test_g3_4_the_carrier_parse_has_one_body_and_five_call_sites():
    r"""RECORD (`L55i`): ``FullField.require_member(`` is called at exactly
    five sites under ``orpheus/`` and the retired ``_require_typed_composite``
    at none — the §6b set of the parse, made mechanical so a sixth consumer
    (or a re-hand-rolled clause) shows up here rather than in a review.

    Per ``vv-principles`` #17's Pattern-2-hoist clause the body is one
    but the WIRING is five: each site passes its own ``context`` and its own
    carrier, so a per-site mutation battery is the teeth; this row is the
    denominator that battery needs.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3] / "orpheus"
    call = re.compile(r"FullField\.require_member\(")
    retired = re.compile(r"_require_typed_composite\(")
    calls: list[str] = []
    dead: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root.parent).as_posix()
        for i, line in enumerate(text.splitlines()):
            if line.lstrip().startswith("#"):
                continue
            if call.search(line):
                calls.append(f"{rel}:{i + 1}")
            if retired.search(line):
                dead.append(f"{rel}:{i + 1}")
    if dead:
        pytest.fail(f"the retired guard is still spelled at: {dead}")
    if len(calls) != _EXPECTED_CALL_SITES:
        pytest.fail(
            f"FullField.require_member has {len(calls)} call sites, expected "
            f"{_EXPECTED_CALL_SITES}: {calls} — a consumer joined or left the "
            f"parse; re-baseline this record WITH the per-site battery"
        )
