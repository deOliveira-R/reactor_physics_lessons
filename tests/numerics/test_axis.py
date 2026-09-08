r"""Intrinsic laws of the space-factor axis (campaign 1, CS1 step 1).

Every DEFINING law of the ``Axis`` / ``EnergyAxis`` concept gets a direct
test here, not merely a usage test — the project's "test a math concept's
intrinsic properties" standard. The concept: an axis is
(index shape, factor measure, basis kind, generator identity), frozen,
structurally identified PER SUBCLASS, with ``weights=None`` meaning the
COUNTING measure deliberately (an axis has no "unbound" state), and
weights stored CANONICALLY (all-ones collapses to ``None``; ``-0.0``
normalized; non-finite refused) so one measure has one spelling and one
identity.

Gate ids A1–A12 refer to the CS1 battery of record
(``scratch/cs1_verification_plan.md`` §2); the canonicalization gates
realize the Q-T1/Q-T3 rulings recorded in
``.claude/plans/cs1_energy_space_design.md`` §T-R.

No theory ``:label:`` — these are software/math invariants of a type, so
``foundation``, never ``verifies(...)``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from orpheus.data.energy_grid import EnergyGrid
from orpheus.derivations.common.xs_library import get_mixture
from orpheus.numerics.axis import Axis, BasisKind, EnergyAxis

pytestmark = pytest.mark.foundation

#: Two groups, strictly DESCENDING (the canonical fast-first convention,
#: ``EnergyGrid`` refuses anything else).
_EDGES_2G = np.array([1.0e7, 1.0e3, 1.0e-3])


def _require(condition: bool, message: str) -> None:
    """A ``-O``-firing assertion (NOT a bare ``assert``)."""
    if not condition:
        pytest.fail(message)


def test_axis_is_frozen() -> None:
    """A1 — the axis is a VALUE: rebinding a field raises.

    LOADED leg (there is no control): the mutation is the witness. An axis
    that is hashable and mutable is an illegal state — its identity could
    change after it has been used as a dict key or compared inside a
    space's ``__eq__``.
    """
    ax = Axis("x", (3,), kind=BasisKind.NODAL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ax.label = "y"  # type: ignore[misc]


def test_weights_none_is_the_counting_measure() -> None:
    """A2 — ``weights=None`` IS the counting measure, deliberately.

    Sharpened per the Q-T3 ruling: the counting measure has ONE spelling,
    enforced by CANONICALIZATION — an all-ones weight array collapses to
    ``None`` at construction, so the None-spelled and ones-spelled
    counting axes are not merely equivalent, they are EQUAL (one
    identity), and no consumer can ever observe two spellings of the
    identity metric.
    """
    none_spelled = Axis("x", (3,), kind=BasisKind.NODAL)
    ones_spelled = Axis("x", (3,), weights=np.ones(3), kind=BasisKind.NODAL)
    _require(none_spelled.weights is None, "None spelling must store None")
    _require(
        ones_spelled.weights is None,
        "Q-T3 canonicalization: all-ones weights must collapse to None at "
        "construction (one spelling per measure)",
    )
    _require(
        none_spelled == ones_spelled and hash(none_spelled) == hash(ones_spelled),
        "the two spellings of the counting measure must be ONE identity",
    )


def test_weights_are_stored_canonically_and_read_only() -> None:
    """Q-T1/Q-T3 — the stored measure is canonical bytes, immutable.

    ``-0.0`` normalizes to ``+0.0`` (one measure, one byte pattern — the
    name digest and the hash read these bytes), and the stored array
    refuses in-place writes.
    """
    neg_zero = Axis("x", (2,), weights=np.array([-0.0, 2.0]), kind=BasisKind.NODAL)
    pos_zero = Axis("x", (2,), weights=np.array([0.0, 2.0]), kind=BasisKind.NODAL)
    _require(
        neg_zero == pos_zero and hash(neg_zero) == hash(pos_zero),
        "-0.0 and +0.0 spell ONE measure and must be ONE identity",
    )
    w = neg_zero.weights
    assert w is not None  # narrowing for the type checker
    _require(
        w.tobytes() == np.array([0.0, 2.0]).tobytes(),
        "stored weights must be the canonical (+0.0) byte pattern",
    )
    _require(not w.flags.writeable, "stored weights must be read-only")
    with pytest.raises((ValueError, RuntimeError)):
        w[0] = 5.0


def test_non_finite_weights_are_refused() -> None:
    """Q-T1 — a factor measure has FINITE weights; nan/inf are refused.

    (No non-negativity guard, deliberately: CS2's quadrature axes legally
    carry signed weights.)
    """
    for bad in (np.array([np.nan, 1.0]), np.array([np.inf, 1.0])):
        with pytest.raises(ValueError, match="finite"):
            Axis("x", (2,), weights=bad, kind=BasisKind.NODAL)
    # The signed-weight door stays OPEN (the anti-claim of the guard):
    signed = Axis("x", (2,), weights=np.array([-0.5, 1.5]), kind=BasisKind.NODAL)
    w = signed.weights
    assert w is not None
    _require(bool(w[0] == -0.5), "signed weights are legal (quadrature axes)")


def test_energy_axis_equality_is_edges_CONTENT_not_grid_identity() -> None:
    r"""A3 ⭐ — two DISTINCT ``EnergyGrid`` objects with equal edges give
    EQUAL axes.

    ``EnergyGrid`` is ``frozen=True, eq=False`` (identity equality) and
    ``Mixture.energy_grid`` mints a FRESH one per access, so
    ``[M]`` (2026-08-20, ``4e11731b``)::

        mix.energy_grid is mix.energy_grid   -> False
        mix.energy_grid == mix.energy_grid   -> False
        np.array_equal(a.edges, b.edges)     -> True

    If ``from_grid`` compared by grid identity, two ``bulk_space`` mints
    from ONE mixture would be UNEQUAL spaces and ``_agreed_space`` would
    raise inside a legitimate homogeneous solve. This is the cheapest
    gate in the battery with an end-to-end consequence.
    """
    mix = dataclasses.replace(get_mixture("A", "2g"), eg=_EDGES_2G)
    grid_a, grid_b = mix.energy_grid, mix.energy_grid
    _require(grid_a is not grid_b, "precondition lost: energy_grid now caches")
    _require(grid_a != grid_b, "precondition lost: EnergyGrid gained value equality")
    axis_a, axis_b = EnergyAxis.from_grid(grid_a), EnergyAxis.from_grid(grid_b)
    _require(axis_a == axis_b, f"{axis_a!r} != {axis_b!r} — identity leaked from the grid")
    _require(hash(axis_a) == hash(axis_b), "equal axes must hash equal")


def test_synthetic_equality_is_ng_only() -> None:
    """A4 — ``synthetic(ng)`` identity is exactly ``ng``."""
    _require(EnergyAxis.synthetic(2) == EnergyAxis.synthetic(2), "same ng must be equal")
    _require(
        hash(EnergyAxis.synthetic(2)) == hash(EnergyAxis.synthetic(2)),
        "equal synthetic axes must hash equal",
    )
    _require(EnergyAxis.synthetic(2) != EnergyAxis.synthetic(3), "ng differs => unequal")


def test_synthetic_and_from_grid_at_the_same_ng_are_UNEQUAL() -> None:
    r"""A5 ⭐ — the sharp pair: same ``ng``, same shape, different axes.

    ``synthetic(2)`` and ``from_grid(<2-group edges>)`` describe the same
    INDEX SET and different PARTITIONS. Q2 rules identity = ``ng`` + edges
    CONTENT, so they must differ; if identity collapsed to ``ng`` the
    derived space NAME would collide and the axes tuples would compare
    equal — since the identity flip (CS4c step 6) the axes tuple IS the
    space identity, and until it the derived name was — so two physically
    different spaces would compare equal and compose silently either way.
    """
    synthetic = EnergyAxis.synthetic(2)
    gridded = EnergyAxis.from_grid(EnergyGrid(_EDGES_2G))
    _require(synthetic.shape == gridded.shape, "precondition: same index set")
    _require(synthetic != gridded, "identity collapsed to ng — the partition was dropped")


def test_energy_axis_hash_agrees_with_eq() -> None:
    """A6 — the A3/A4/A5 population round-trips through a ``set`` and a
    ``dict`` with exactly the expected cardinality."""
    mix = dataclasses.replace(get_mixture("A", "2g"), eg=_EDGES_2G)
    population = [
        EnergyAxis.from_grid(mix.energy_grid),
        EnergyAxis.from_grid(mix.energy_grid),  # equal to the previous (A3)
        EnergyAxis.synthetic(2),
        EnergyAxis.synthetic(2),  # equal to the previous (A4)
        EnergyAxis.synthetic(3),
        EnergyAxis.from_grid(EnergyGrid(np.array([2.0e7, 1.0, 1.0e-5]))),
    ]
    distinct = {population[0], population[2], population[4], population[5]}
    _require(
        len(set(population)) == 4,
        f"expected 4 distinct axes in the population, got {len(set(population))}",
    )
    _require(len({ax: None for ax in population}) == 4, "dict keying must agree with set")
    _require(len(distinct) == 4, "the four representatives must stay distinct")


def test_axis_is_immune_to_mutation_of_the_source_array() -> None:
    r"""A7 — the axis defensively copies its edges/weights.

    ``Mixture`` is a plain (NON-frozen) ``@dataclass`` and ``eg`` is a bare
    ndarray, so the caller's array is live. A hashable value object that
    aliases a mutable buffer is an illegal state: its ``__hash__`` silently
    changes after it has been stored in a dict.
    """
    edges = _EDGES_2G.copy()
    axis_before = EnergyAxis.from_grid(EnergyGrid(edges))
    reference = EnergyAxis.from_grid(EnergyGrid(_EDGES_2G))
    hash_before = hash(axis_before)
    edges[0] = 9.9e9  # mutate the caller's buffer
    _require(hash(axis_before) == hash_before, "hash moved under source mutation")
    _require(axis_before == reference, "equality moved under source mutation")

    weights_src = np.array([2.0, 5.0])
    weighted = Axis("x", (2,), weights=weights_src, kind=BasisKind.NODAL)
    weighted_ref = Axis("x", (2,), weights=np.array([2.0, 5.0]), kind=BasisKind.NODAL)
    weights_src[0] = 7.0
    _require(weighted == weighted_ref, "weights identity moved under source mutation")


def test_rank_zero_shape_is_refused() -> None:
    """A8 — rank >= 1 is a construction invariant, refused with a typed
    error (not a bare ``assert`` — ``-O`` would strip that in production)."""
    with pytest.raises(ValueError, match="rank"):
        Axis("x", (), kind=BasisKind.NODAL)


def test_weights_shape_must_match_shape() -> None:
    """A9 — the factor measure lives over exactly ``shape``."""
    with pytest.raises(ValueError, match="shape"):
        Axis("x", (3,), weights=np.ones(4) * 2.0, kind=BasisKind.NODAL)


def test_basis_kind_participates_in_identity() -> None:
    """A10 — NODAL and MODAL are different axes.

    The cone metadata reads ``kind``; if kind were metadata-only, a modal
    space and a nodal space would compare equal and the CS1 step-4 cone
    refusal would be reachable through the wrong space.
    """
    nodal = Axis("x", (3,), kind=BasisKind.NODAL)
    modal = Axis("x", (3,), kind=BasisKind.MODAL)
    _require(nodal != modal, "basis kind must participate in identity")


def test_identity_is_structural_PER_SUBCLASS() -> None:
    """A11 — an ``EnergyAxis`` never equals a generic ``Axis`` carrying the
    same tuple of fields.

    "Structural per subclass, from day one" (§A). Free if ``EnergyAxis`` is
    its own dataclass (dataclass ``__eq__`` requires ``other.__class__ is
    self.__class__``); NOT free if ``EnergyAxis`` is a factory returning a
    bare ``Axis`` — which is exactly the design decision this pins.
    """
    energy = EnergyAxis.synthetic(2)
    generic = Axis("energy", (2,), kind=BasisKind.NODAL)
    _require(energy != generic, "subclass identity collapsed (EnergyAxis == Axis)")
    _require(generic != energy, "subclass identity must be symmetric")


def test_weights_are_part_of_identity() -> None:
    r"""A12 — same label, same shape, different measure ⟹ different axis.

    The §B clause-2 retrodiction at the axis tier ("a genuine one-cell slab
    keeps its axis with weight V != 1, distinguished from the quotient point
    by MEASURE") and the Q8 migration doctrine ("metric differences imply
    space differences"). ⚠ Per F2 this distinction is INVISIBLE to ``.H``
    on a one-cell space (a scalar metric commutes with everything), so
    identity is the only instrument that can carry it.
    """
    counting = Axis("spatial", (1,), kind=BasisKind.NODAL)  # quotient point, weight 1
    one_cell = Axis("spatial", (1,), weights=np.array([2.0]), kind=BasisKind.NODAL)
    other_cell = Axis("spatial", (1,), weights=np.array([3.0]), kind=BasisKind.NODAL)
    _require(counting != one_cell, "V != 1 must differ from the quotient point")
    _require(one_cell != other_cell, "different measures must be different axes")


def test_energy_axis_refuses_weights_and_modal_kind() -> None:
    """The counting theorem and the faces reading, enforced at construction.

    A weighted ``EnergyAxis`` would spell the non-physical state the
    counting-measure theorem forbids (the deliberately non-physical D4b
    control toy uses a generic ``Axis`` instead); a MODAL ``EnergyAxis``
    would contradict the groups-are-cells reading. All-ONES weights are
    accepted — canonicalization collapses them to ``None``, because ones
    IS the counting measure.
    """
    with pytest.raises(ValueError, match="[Cc]ounting|theorem"):
        EnergyAxis("energy", (2,), weights=np.array([2.0, 5.0]), kind=BasisKind.NODAL)
    with pytest.raises(ValueError, match="NODAL"):
        EnergyAxis("energy", (2,), kind=BasisKind.MODAL)
    ones_spelled = EnergyAxis("energy", (2,), weights=np.ones(2), kind=BasisKind.NODAL)
    _require(
        ones_spelled == EnergyAxis.synthetic(2),
        "ones-spelled counting EnergyAxis must canonicalize to the synthetic axis",
    )
