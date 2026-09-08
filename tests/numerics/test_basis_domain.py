r"""Every basis states the manifold its functions EAT — #429 tracker 2.1.

A basis function is a **map**, and a map is not defined until its source is:
:math:`Y_\ell^m : S^2 \to \mathbb R` takes a POINT of :math:`S^2`;
:math:`\mathbf 1_R` takes a point of whatever was partitioned. Until 2.1 the
:class:`~orpheus.numerics.basis.base.Basis` ABC had no way to ask, so the
answer was smuggled through a coefficient-space NAME STRING — and one of the
two producers hard-coded it.

The defect that forced this
===========================

:meth:`~orpheus.numerics.basis.indicator_basis.IndicatorBasis.space` built
``f"L2[coarse_cells_R{ndim}]"``, asserting a SPATIAL manifold whatever the
edges actually partitioned. `[M]` 2026-09-01, pre-fix: a 2-group **energy**
indicator space and a 2-cell **spatial** indicator space were ``==``-equal
**and hash-equal**, both ``FunctionSpace('L2[coarse_cells_R1]', shape=(2,))``.
:class:`~orpheus.numerics.space.FunctionSpace` identity is ``(name, shape)``,
so a false name is not cosmetic — it is an illegal state that IS
representable, and every composability guard downstream reads it as truth.

⭐ **The sharpest form of it, and the reason this file's keystone is
``test_d6``:** at four of the five production sites the basis and its measure
are built in the SAME function, three to five lines apart, and the *measure*
named the manifold correctly the whole time (``support="energy"``,
``"spatial_R1"``, ``f"index({label})"``). The answer was never unavailable —
only unasked. So the durable gate is not "the name is right" but **"the two
halves of one frame name ONE manifold"**, which goes red the day either side
re-invents it.

What was NOT claimed when this file landed, and is now
=====================================================

At 2.1 ``DiscreteMeasure.support`` was still a ``str``, so ``test_d6`` could
only pin that a frame's two halves SPELLED the manifold the same way.
✅ Tracker 2.0c (2026-09-01) retyped it, and ``test_d6`` asserts the halves
ARE one manifold (its own docstring records the promotion).

...and the symmetry its functions HAVE — #429 tracker 2.1b (section E)
======================================================================

A basis states the group its functions are invariant under, and it does so
by naming its domain: a function on an orbit space :math:`M/H` is an
:math:`H`-invariant function, so
:attr:`~orpheus.numerics.basis.base.Basis.invariance_group` is READ off
:attr:`~orpheus.numerics.basis.base.Basis.domain` — ``Quotient.by``;
``Trivial`` on the bare sphere; ``None`` where no subgroup of :math:`O(3)`
acts. The tracker asked for a field answered by six subclasses; the phase
opener found the answer already sitting in the fold basis's domain, the way
2.0d's ``quotient_group`` field dissolved into ``Quotient.by`` at 2.0c.

The keystone is ``test_e1``: the fold's two halves read ONE group object,
and the slab's pairing with the full-sphere harmonics — ERR-080's — reads as
the lattice verdict False. ⚠ Nothing REFUSES on that verdict yet: the frame
gate is tracker 2.2 (fused with 0.1b + 0.6 + 3.4), so ``test_e1``'s negative
leg is a measurement made spellable, not a refusal (``plan-authoring`` §6c —
the witness a field with no consumer can honestly carry is an AGREEMENT on a
shipped pairing, and the fold ships).

See :doc:`/theory/foundations/manifolds` §(b) and
:doc:`/theory/foundations/spaces`.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from orpheus.data.energy_grid import EnergyGrid
from orpheus.geometry import BC, CoordSystem, Mesh1D
from orpheus.numerics.basis import (
    Basis,
    IndicatorBasis,
    OverlapBasis,
    SphericalHarmonicBasis,
    WeightedIndicatorBasis,
)
from orpheus.numerics.basis.spherical_harmonic_basis import (
    MirrorEvenSphericalHarmonicBasis,
)
from orpheus.numerics.manifold import (
    ENERGY,
    SPHERE,
    EnergyGroups,
    IndexSet,
    Manifold,
    Quotient,
    RealSpace,
    ambient_dim,
)
from orpheus.numerics.frame import GalerkinFrame
from orpheus.numerics.measure import DiscreteMeasure
from orpheus.numerics.quadrature import Quadrature
from orpheus.numerics.space import FunctionSpace
from orpheus.numerics.symmetry import SubgroupOfO3
from orpheus.sn.operators.loss_kernel_gauge import LossKernelBasis

pytestmark = [pytest.mark.foundation]


_TWO_CELLS = (np.array([0.0, 0.5, 1.0]),)
_TWO_GROUPS = (np.arange(3, dtype=float) - 0.5,)
_THREE_CELLS = (np.array([0.0, 0.3, 0.6, 1.0]),)


class _AbstractStub(Basis):
    """Every abstract member of :class:`Basis` EXCEPT ``domain``.

    Full signatures, and bodies that RAISE rather than ``...``: a stub whose
    members silently return ``None`` is not a faithful stand-in for the ABC,
    and the type checker says so. Nothing here is ever reached. ``test_d3``
    uses it bare (it must refuse to construct); section E subclasses it with
    a ``domain`` alone, so that the group a basis HAS is shown to be decided
    by the DOMAIN and by nothing else about the class.
    """

    def evaluate(self, points: NDArray, /) -> NDArray:
        raise NotImplementedError

    def synthesize(self, c: NDArray, table: NDArray, /) -> NDArray:
        raise NotImplementedError

    def analyze(self, v: NDArray, table: NDArray, w: NDArray, /) -> NDArray:
        raise NotImplementedError

    def analyze_transpose(
        self, c: NDArray, table: NDArray, w: NDArray, /,
    ) -> NDArray:
        raise NotImplementedError

    def reconstruct(self, c: NDArray, table: NDArray, /) -> NDArray:
        raise NotImplementedError

    def reconstruct_transpose(
        self, v: NDArray, table: NDArray, /,
    ) -> NDArray:
        raise NotImplementedError

    def mass_matrix(self, measure: DiscreteMeasure, /) -> NDArray:
        raise NotImplementedError

    @property
    def space(self) -> FunctionSpace:
        raise NotImplementedError


def test_d1_an_energy_space_and_a_spatial_space_are_not_the_same_space() -> None:
    """⭐⭐ KEYSTONE — the defect 2.1 exists to make unspellable.

    Two partitions with the same cell COUNT of two entirely different
    manifolds must not produce equal coefficient spaces. `[M]` before 2.1 they
    did, ``==`` and hash-equal alike.

    Both legs matter (``vv-principles`` #11). The negative control — a 3-cell
    spatial partition against the 2-cell one — is what shows the inequality is
    not a degenerate always-false: if ``__eq__`` were broken outright, the
    first assertion would pass for the wrong reason and this file would
    certify nothing.
    """
    energy = IndicatorBasis(_TWO_GROUPS, EnergyGroups(2)).space
    spatial = IndicatorBasis(_TWO_CELLS, RealSpace(1)).space

    assert energy.shape == spatial.shape == (2,)     # the collision's precondition
    assert energy != spatial
    assert len({energy, spatial}) == 2   # separation through the container

    # negative control: inequality is decided by the MANIFOLD, and shape still
    # separates two partitions of the SAME manifold.
    three = IndicatorBasis(_THREE_CELLS, RealSpace(1)).space
    assert spatial != three
    assert IndicatorBasis(_TWO_CELLS, RealSpace(1)).space == spatial


def test_d2_every_shipped_basis_answers_what_its_functions_eat() -> None:
    """The completeness claim, enumerated by RUNTIME introspection.

    ⚠ Deliberately **not** an AST census and not a hand-written list. `[M]`
    2026-09-01 an AST pass over this tree reported 3 direct / 5 recursive
    ``Basis`` subclasses where the runtime answer is **4 / 6** — inheritance is
    a runtime relation, and a static pass cannot see re-exports, aliased bases
    or qualified base names. Enumerating at runtime also means a basis added
    tomorrow is in scope without editing this file.

    Filtered to ``orpheus.``-defined classes so a test-local subclass declared
    by whichever module pytest imported first cannot make the gate
    order-dependent.
    """

    def shipped(cls: type) -> list[type]:
        out: list[type] = []
        for sub in cls.__subclasses__():
            if sub.__module__.startswith("orpheus."):
                out.append(sub)
            out += shipped(sub)
        return out

    subclasses = shipped(Basis)
    assert len(subclasses) >= 6, (
        f"expected at least the 6 shipped bases, found {len(subclasses)}: "
        f"{[c.__name__ for c in subclasses]} — if a basis was retired, retire "
        f"its row here too; if the count COLLAPSED, an import is missing and "
        f"this gate is measuring nothing."
    )
    for sub in subclasses:
        assert "domain" not in getattr(sub, "__abstractmethods__", frozenset()), (
            f"{sub.__name__} does not say what its functions eat"
        )


def test_d3_a_basis_that_cannot_say_what_it_eats_cannot_be_built() -> None:
    """The refusal is STRUCTURAL — at construction, not at first call.

    The stub (:class:`_AbstractStub`, module scope since 2.1b) implements
    every other abstract member, so ``domain`` is the only thing missing and
    the refusal cannot be credited to the wrong arm (``vv-principles`` #17's
    granularity trap). The positive leg is the same stub WITH ``domain``,
    which must construct.
    """

    # The ignore DOCUMENTS the claim rather than hiding a defect: pyright
    # reports exactly the refusal this line asserts ("Basis.domain is not
    # implemented"), which is the established idiom for a negative leg here
    # (e.g. tests/transport/test_timed_full_field.py:126).
    with pytest.raises(TypeError, match="domain"):
        _AbstractStub()                              # type: ignore[abstract]

    class _Complete(_AbstractStub):
        @property
        def domain(self) -> Manifold:
            return SPHERE

    assert _Complete().domain == SPHERE                  # positive leg


def test_d4_a_partition_must_have_one_axis_per_coordinate() -> None:
    """The construction invariant, both legs.

    A ``d``-axis tensor partition partitions a ``d``-coordinate manifold. The
    negative leg is what stops a 2-axis spatial partition from claiming the
    energy axis; the positive legs are every shipped combination, so the guard
    cannot pass by refusing everything.

    ⚠ The invariant is the AMBIENT width, NOT
    :meth:`~orpheus.numerics.manifold.Manifold.contains` on the cell centres.
    The stronger check reads better and is wrong: `[M]` the single-region index
    partition ``[-0.5, n-0.5]`` that ``frame.py``'s axis marginal ships has
    centre :math:`(n-1)/2`, not an integer, and ``IndexSet`` admits only
    integers — so it would refuse a correct production caller
    (``vv-principles`` #16).
    """
    ok = [
        (_TWO_CELLS, RealSpace(1)),
        (_TWO_GROUPS, EnergyGroups(2)),
        (_TWO_GROUPS, IndexSet("g", 2)),
        ((np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0])), RealSpace(2)),
    ]
    for edges, manifold in ok:
        basis = IndicatorBasis(edges, manifold)
        assert ambient_dim(basis.domain) == basis.ndim

    with pytest.raises(ValueError, match="partition axis"):
        IndicatorBasis((np.array([0.0, 1.0]), np.array([0.0, 1.0])), ENERGY)
    with pytest.raises(ValueError, match="partition axis"):
        IndicatorBasis(_TWO_CELLS, RealSpace(2))
    with pytest.raises(ValueError, match="partition axis"):
        IndicatorBasis(_TWO_CELLS, SPHERE)               # 1 axis, 3 coordinates


@pytest.mark.parametrize("L", [0, 1, 3, 7])
def test_d5_a_harmonic_eats_a_direction_at_every_degree(L: int) -> None:
    """:math:`S^2` is constant in :math:`L` — truncation changes the SPAN, not
    the source.

    Swept over four degrees rather than asserted once, because "the domain does
    not depend on ``L``" is the actual claim and a single degree cannot carry
    it.
    """
    assert SphericalHarmonicBasis(L=L).domain == SPHERE


@pytest.mark.parametrize("axis,name", [(0, "S^2/sigma_x"), (1, "S^2/sigma_y"), (2, "S^2/sigma_z")])
def test_d5b_a_mirror_even_harmonic_eats_the_QUOTIENT(axis: int, name: str) -> None:
    """⭐ The class's whole content, stated as a type.

    Every σ-even harmonic takes the same value at :math:`\\Omega` and
    :math:`\\sigma_a\\Omega`, and a function constant on the orbits of
    :math:`H` is a function on :math:`M/H`. So this basis's domain is the
    quotient — and it must NOT compare equal to the sphere, which is the
    discrimination that had no operands before 2.1.

    `[M]` the derived name matches the support a ``folded_product`` rule's
    angular frame already reports (landed 0.1c), so the two halves of a folded
    frame agree by DERIVATION rather than by two tags that happen to match.
    """
    basis = MirrorEvenSphericalHarmonicBasis(L=2, mirror_axis=axis)
    expected = SPHERE.quotient(SubgroupOfO3.Mirror("xyz"[axis]))

    assert basis.domain == expected
    assert basis.domain.name == name
    assert basis.domain != SPHERE                    # the discriminating leg
    assert SphericalHarmonicBasis(L=2).domain == SPHERE   # the parent is unmoved


def test_d6_a_frames_two_halves_name_ONE_manifold() -> None:
    """⭐⭐ THE FLAGSHIP. The basis and the measure of one frame agree.

    This is the property whose absence WAS the defect: at every site below the
    two objects are built within five lines of each other, and before 2.1 the
    measure named the manifold correctly while the basis hard-coded a spatial
    one. A gate on the basis's name alone would have been satisfied by any
    self-consistent lie; this one cannot be, because the two names come from
    independently-authored halves.

    ⭐⭐ **STRENGTHENED at tracker 2.0c (2026-09-01), and by the same change
    it demanded.** Until then ``support`` was a ``str``, so the strongest
    claim this gate could make was ``basis.domain.name == measure.support`` —
    the two halves SPELL the manifold the same way. Both halves are now
    ``Manifold``, so it asserts they ARE the same manifold, which is what the
    test's own name has always said. (``coding-standards``' mirror clause: a
    retirement can silently PROMOTE a gate's claim class; the description
    moves with it or it goes on advertising the weaker claim.)

    ⚠ One known limit, stated rather than hidden: the fifth production pair
    (``frame.py``'s axis marginal, a private ``_collapse_pair`` whose frame is
    deliberately forgetful) is pinned instead by
    ``tests/numerics/test_axis_marginal.py``'s independent re-spelling of the
    same construction.

    ✅ The ``LossKernelBasis`` divergence this test used to pin — its measure
    tagging the bare label while ``IndexSet`` wrapped it as ``index(...)`` —
    was **discharged at 2.0c**, and not by agreement: the measure now reads
    ``support=basis.domain``, so the two cannot differ. It is asserted below as
    a third pair rather than as an exception.
    """
    mesh = Mesh1D(
        edges=np.array([0.0, 0.5, 1.0]), mat_ids=np.array([0, 1]),
        coord=CoordSystem.CARTESIAN,
        bc_left=BC("vacuum"), bc_right=BC("vacuum"),
    )
    grid = EnergyGrid(edges=np.array([2.0e7, 1.0, 1.0e-5]))

    pairs = [
        ("Mesh1D", mesh.indicator_basis(), mesh.volume_measure),
        ("EnergyGrid", grid.as_basis(), grid.as_measure()),
    ]
    for who, basis, measure in pairs:
        assert basis.domain == measure.support, (
            f"{who}: the frame's basis says it lives on "
            f"{basis.domain!r} and its measure says {measure.support!r} — "
            f"one frame, two manifolds."
        )

    # ...and the spaces those agreeing halves mint stay distinguishable.
    assert mesh.indicator_basis().space != grid.as_basis().space

    # The loss-kernel pair, now ONE point set with ONE spelling. Its measure
    # is built inside `_build_gauge_blocks`, so this pins the basis's half and
    # the production site reads it directly (`loss_kernel_gauge.py`,
    # `support=basis.domain`) — the divergence is unspellable, not merely
    # absent.
    block = LossKernelBasis(table=np.eye(2), orbit=(0, 1), group=3)
    assert isinstance(block.domain, IndexSet)
    assert block.domain.label == "sn_trace_orbit(0, 1)_g3"
    assert block.domain.name == "index(sn_trace_orbit(0, 1)_g3)"


def test_d7_a_wrapping_basis_cannot_drift_from_the_basis_it_wraps() -> None:
    """Delegation, not duplication (Cardinal Rule 2).

    ``WeightedIndicatorBasis`` reweights the trial indicator, and ``OverlapBasis``
    decorates it with a fractional table; neither moves the support, so neither
    may carry its own answer. Asserted by IDENTITY (``is``), which a copied
    literal would fail even when the copy happens to be right today.
    """
    trial = IndicatorBasis(_TWO_GROUPS, EnergyGroups(2))

    weighted = WeightedIndicatorBasis(trial, np.ones(2))
    assert weighted.domain is trial.domain
    assert weighted.space == trial.space

    table = np.array([[1.0, 0.0], [0.0, 1.0]])
    overlap = OverlapBasis.from_indicator(trial, table, fine=EnergyGroups(2))
    # ⚠ Since #429's fused commit (2026-09-02) the overlap's DOMAIN is the
    # FINE partition it EATS (the table's rows), deliberately NOT the wrapped
    # basis's — that inherited `domain = partition_of` was the defect the
    # frame's G0 caught on landing (every `Mixture.condense` refused). What
    # cannot drift is the COARSE side it spans: the partition it decorates.
    assert overlap.domain == EnergyGroups(2)
    assert overlap.partition_of is trial.partition_of
    assert overlap.space == trial.space


# ═══════════════════════════════════════════════════════════════════════════
# E. ...and the symmetry its functions HAVE — #429 tracker 2.1b
# ═══════════════════════════════════════════════════════════════════════════


def test_e1_the_folds_two_halves_read_ONE_group_and_the_slab_now_does_too() -> None:
    r"""⭐⭐ KEYSTONE — a frame's two halves name ONE orbit space, and the slab now does.

    **The fold (unchanged).** Its measure SPENT ``Mirror('y')`` (folded onto
    :math:`S^2/\sigma_y`) and its frame basis HAS ``Mirror('y')`` (the
    σ_y-even harmonics on the same orbit space). Asserted three ways,
    strongest last: the lattice verdict, equality, and **identity** — the two
    halves are the ``by`` of one memoised :class:`Quotient`, so there is no
    second tag that could drift.

    **The slab — ⛔ INVERTED 2026-09-02 (#429/ERR-080).** This leg used to
    read the DEFECT as a verdict: ``gauss_legendre(8)`` spent ``SO2('x')``
    while the harmonics its frame bound HAVE only ``Trivial``, so
    ``Trivial ⊇ SO2('x')`` was **False** and nothing refused it (tracker 2.2
    did not exist; the frame's measure still carried the forged
    :math:`S^2`). The fused commit closed both halves: the frame binds the
    Legendre basis on :math:`S^2/O(2)_x`, so the verdict is now **True** by
    the same three assertions the fold gets.

    ⭐ **The refusal did not disappear — it MOVED, and it is asserted here.**
    A verdict that merely flipped to True would be a weaker gate than the one
    it replaced (the old negative leg was the only place the lattice's
    discriminating power was visible). So the refused pairing is constructed
    directly, from shipped classes: a raw ``GalerkinFrame`` binding the full
    harmonics to the slab's own measure — exactly what production built
    before the repair — and G0 rejects it. That is row 5 of the G0 pairing
    table, whose other six rows live in
    ``tests/numerics/test_frame.py::TestG0TheFrameBindsAlongAQuotientMap``.
    """
    fold = Quadrature.folded_product(n_mu=4, n_phi=8)
    fold_basis = fold.angular_frame(2).basis
    spent = fold.measure.quotient_group
    has = fold_basis.invariance_group
    assert spent is not None and has is not None
    assert has.contains(spent)                             # the lattice verdict
    assert has == spent == SubgroupOfO3.Mirror("y")
    assert has is spent                                    # ONE object, not two agreeing tags...
    assert fold_basis.domain is fold.measure.support       # ...because the manifold is one object

    slab = Quadrature.gauss_legendre(8)
    slab_basis = slab.angular_frame(2).basis
    spent = slab.measure.quotient_group
    has = slab_basis.invariance_group
    assert spent is not None and has is not None
    assert has.contains(spent)                             # ⭐ was False (ERR-080)
    assert has == spent == SubgroupOfO3.O2("x")
    assert has is spent
    assert slab_basis.domain is slab.measure.support

    # …and the pairing that WAS built is now refused at construction (G0's
    # row 5) — the negative leg this gate would otherwise have lost.
    with pytest.raises(ValueError, match="no quotient map"):
        GalerkinFrame(SphericalHarmonicBasis(L=2), slab.measure)


@pytest.mark.parametrize("L", [0, 1, 3, 7])
def test_e2_the_full_sphere_harmonics_HAVE_the_trivial_group(L: int) -> None:
    """``Trivial``, and not ``None``: :math:`O(3)` ACTS on :math:`S^2`, and the
    basis spent none of it.

    Swept over four degrees because "constant in ``L``" is the claim —
    and ``L = 0`` is included on purpose: :math:`P_0` is
    :math:`O(3)`-invariant and still answers ``Trivial``, because the reading
    is the lower bound the DOMAIN guarantees, not a computed stabiliser
    (under-declaration is legal and lossy; see the property's docstring).
    """
    assert SphericalHarmonicBasis(L=L).invariance_group == SubgroupOfO3.Trivial


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_e2b_a_mirror_even_harmonic_HAS_its_mirror_read_off_its_domain(axis: int) -> None:
    """The group is DERIVED: it is the domain's ``by``, by identity.

    A stored copy that happened to be right would pass ``==`` and fail ``is``
    (`[M]` ``Mirror('y') is Mirror('y')`` is False for two constructions), so
    the identity leg is what separates *derived* from *duplicated*. The
    negative leg — a different mirror is a different, incomparable group —
    shows the answer moves with the axis rather than being one constant.
    """
    basis = MirrorEvenSphericalHarmonicBasis(L=2, mirror_axis=axis)
    assert basis.invariance_group == SubgroupOfO3.Mirror("xyz"[axis])
    assert isinstance(basis.domain, Quotient)
    assert basis.invariance_group is basis.domain.by

    other = MirrorEvenSphericalHarmonicBasis(L=2, mirror_axis=(axis + 1) % 3)
    mine, theirs = basis.invariance_group, other.invariance_group
    assert mine is not None and theirs is not None
    assert not mine.contains(theirs)


def test_e3_no_subgroup_of_O3_acts_on_a_mesh_a_group_index_or_a_trace_index() -> None:
    """The category leg: ``None``, not ``Trivial``, on every non-angular basis.

    ``Trivial`` would assert an :math:`O(3)` action that does not exist on a
    spatial partition, an energy-group index or a trace-DOF index set; the
    measure side gives the same category answer for the same manifolds
    (:attr:`DiscreteMeasure.phase`). Wrappers answer by DELEGATION (they
    delegate ``domain``, and the group is read off it), so a wrapped energy
    basis cannot answer differently from the basis it wraps.

    The positive control is in the same test (``vv-principles`` #11): the
    ``None`` is decided by the DOMAIN, so the same class shape with a sphere
    domain answers ``Trivial`` — the arm is not "everything is None".
    """
    energy = IndicatorBasis(_TWO_GROUPS, EnergyGroups(2))
    spatial = IndicatorBasis(_TWO_CELLS, RealSpace(1))
    cases = {
        "energy indicator": energy,
        "spatial indicator": spatial,
        "weighted (delegates)": WeightedIndicatorBasis(energy, np.ones(2)),
        "overlap (delegates)": OverlapBasis.from_indicator(energy, np.eye(2), fine=EnergyGroups(2)),
        "loss kernel": LossKernelBasis(table=np.eye(2), orbit=(0, 1), group=3),
    }
    for who, basis in cases.items():
        assert basis.invariance_group is None, who

    class _OnSphere(_AbstractStub):
        @property
        def domain(self) -> Manifold:
            return SPHERE

    assert _OnSphere().invariance_group == SubgroupOfO3.Trivial   # positive control


def test_e4_the_group_is_decided_by_the_domain_and_by_nothing_else() -> None:
    """The three arms of the derivation on ONE class shape, differing only in
    ``domain``.

    This is the gate that separates *reads the domain* from *knows the
    class*: an implementation keyed on the subclass (an ``isinstance`` on
    ``SphericalHarmonicBasis``, say) would give every stub the same answer.
    The quotient arm is exercised with ``O2('x')`` (the stabiliser the
    entry is named by since #432; ``SO2('x')`` until 2026-09-02) — a second
    group FAMILY through the same arm, since the shipped fold basis only
    ever brings a ``Mirror``.
    """

    class _OnEnergy(_AbstractStub):
        @property
        def domain(self) -> Manifold:
            return ENERGY

    class _OnSphere(_AbstractStub):
        @property
        def domain(self) -> Manifold:
            return SPHERE

    class _OnPolarOrbit(_AbstractStub):
        @property
        def domain(self) -> Manifold:
            return SPHERE.quotient(SubgroupOfO3.O2("x"))

    assert _OnEnergy().invariance_group is None
    assert _OnSphere().invariance_group == SubgroupOfO3.Trivial
    assert _OnPolarOrbit().invariance_group == SubgroupOfO3.O2("x")


def test_e5_part_IV_lattice_table_runs_on_the_objects_that_ship() -> None:
    r"""Part IV's four-row table as a test — tracker 2.1b's done-when.

    Each row pairs a basis's HAS with a measure's SPENT and asserts the
    verdict Part IV states. Rows 2 and 3 need a basis on
    :math:`S^2/O(2)_x`, which is tracker 3.4's ``LegendreBasis`` and does
    not ship; a test-local stub declaring that DOMAIN stands in for it, and
    ⚠ **must be replaced by the real basis when 3.4 lands** — a retirement
    trigger, not a permanent fixture.

    Row 3's measure side is spelled ``None``: a full-sphere rule has SPENT
    nothing, and ``quotient_group`` answers ``None`` rather than ``Trivial``
    for a point set (HAS ≠ SPENT there). The lattice element that ``None``
    stands for on the SPENT side is :math:`\{e\}`, which every group
    contains — asserted explicitly rather than through a local spelling of
    G2's ``None`` rule, because the predicate that reads these operands is
    tracker 2.2's to write once, in production.
    """

    class _OnPolarOrbit(_AbstractStub):
        @property
        def domain(self) -> Manifold:
            return SPHERE.quotient(SubgroupOfO3.O2("x"))

    slab = Quadrature.gauss_legendre(8).measure            # spent O2('x')
    sphere = Quadrature.lebedev(17).measure                # spent nothing
    fold = Quadrature.folded_product(4, 8).measure         # spent Mirror('y')
    full_sh = SphericalHarmonicBasis(L=2).invariance_group
    legendre_like = _OnPolarOrbit().invariance_group
    fold_basis = MirrorEvenSphericalHarmonicBasis(L=2, mirror_axis=1).invariance_group
    assert full_sh is not None and legendre_like is not None and fold_basis is not None
    assert slab.quotient_group is not None and fold.quotient_group is not None

    # row 1 — Trivial (full SH) vs SO2 (slab): refuses the Part I bug, categorically
    assert not full_sh.contains(slab.quotient_group)
    # row 2 — SO2 (trivial isotypic) vs SO2 (slab): the repair
    assert legendre_like.contains(slab.quotient_group)
    # row 3 — SO2 vs Trivial (sphere): a smaller space on a full rule is legal
    assert sphere.quotient_group is None
    assert legendre_like.contains(SubgroupOfO3.Trivial)
    # row 4 — Mirror(y) vs Mirror(y): the shipped fold
    assert fold_basis.contains(fold.quotient_group)
