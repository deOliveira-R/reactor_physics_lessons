r"""CS5 — an axis can name the generator that made it (the provenance doctrine).

One doctrine, one file (the ``test_space_of_axes.py`` file-level-doctrine
precedent): an :class:`~orpheus.numerics.axis.Axis` is a **forgetful map**
from its generator — it keeps the weights and drops the nodes — and since
CS5 (2026-08-29) the mint routes THROUGH the generator
(``measure.axis(label)`` / ``quad.axis(label)``), which records itself as
``Axis.generator``: provenance, **never identity**.

The gates here are the landed machinery's ONLY witnesses — ``[M]`` a
three-arm in-process mutation battery over the 184-test anchor set found
**0 genuine catchers** before this module landed (the one red was a
cross-process harness artefact; ``scratch/cs5_verification_plan.md`` §1).

G5 (the generator-less refusal) and G7 (the route keystone) deliberately do
NOT live here yet: no solve-time consumer reads a generator until the
streaming plan's P4-remainder phase re-points the producer, and a gate that
lands before the case it catches ships green and unfalsifiable
(plan-authoring §6c). Their specs are in the verification plan.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from orpheus.numerics.manifold import RealSpace
from orpheus.numerics.axis import Axis, BasisKind
from orpheus.numerics.quadrature.directional import Quadrature
from orpheus.numerics.space import FunctionSpace

pytestmark = pytest.mark.foundation


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _point() -> Axis:
    """The quotient spatial point (counting weight — the density convention)."""
    return Axis("spatial", (1,), kind=BasisKind.NODAL)


# The shipped angular family — EXHAUSTIVE by construction, not a ladder
# (vv-principles #31's finite-roster corollary): these are the FIVE
# ``Quadrature`` classmethod factories — [M] 2026-08-29 (archivist census)
# ``[n for n, v in vars(Quadrature).items() if isinstance(v, classmethod)]``
# = gauss_legendre, lebedev, level_symmetric, product, folded_product; the
# roster's first cut omitted folded_product, the sigma_y-folded CARRYING
# rule the curvilinear MMS builders default to — i.e. the member with the
# richest ``level_indices``, on exactly the axis this roster gates. Only
# the cylindrical-compatible rules carry a LevelStructure; on the others
# ``level_indices`` is ``[arange(N)]`` — which is the point: the accessor
# answers the fourth name on every rule, degenerately where the rule has
# one level.
_RULES = [
    ("gauss_legendre(4)", lambda: Quadrature.gauss_legendre(4)),
    ("level_symmetric(4)", lambda: Quadrature.level_symmetric(4)),
    ("product(4,8)", lambda: Quadrature.product(4, 8)),
    ("folded_product(4,8)", lambda: Quadrature.folded_product(4, 8)),
    ("lebedev(5)", lambda: Quadrature.lebedev(5)),
]


class TestG1GeneratorIsProvenanceNotIdentity:
    """G1 — the exclusion from ``_identity_key``, and WHY it is mandatory.

    The negative legs (a DIFFERENT weights/kind is still unequal) already
    ship as ``test_axis.py::test_weights_are_part_of_identity`` and
    ``::test_basis_kind_participates_in_identity`` — cited, not
    duplicated; this class is the positive-exclusion half of that pair.
    """

    @pytest.mark.parametrize("name,factory", _RULES)
    def test_distinct_instance_content_equal_generators_are_ONE_axis(
        self, name, factory
    ):
        """G1a — provenance is not identity, and #403 never reaches the axis.

        Mutation: put ``generator`` in ``_identity_key`` → ``a1 == a2``
        RAISES ``ValueError`` (ndarray truth ambiguity through
        ``DiscreteMeasure.__eq__``) — measured, not merely a digest move.
        """
        q1, q2 = factory(), factory()
        _require(q1 is not q2, "the two rules must be distinct instances")
        a1, a2 = q1.axis("angular"), q2.axis("angular")
        _require(a1 == a2, f"{name}: content-equal generators must not split identity")
        _require(hash(a1) == hash(a2), f"{name}: hash must agree with eq")
        _require(len({a1, a2}) == 1, f"{name}: one axis, one set member")

    def test_a_generator_ful_axis_equals_the_generator_LESS_one(self):
        """G1b — the accessor is additive: adding provenance moves nothing."""
        q = Quadrature.gauss_legendre(4)
        minted = q.axis("angular")
        bare = Axis(
            "angular", (q.N,), weights=np.asarray(q.weights, float),
            kind=BasisKind.NODAL,
        )
        _require(
            minted == bare and hash(minted) == hash(bare),
            "a generator-ful axis must equal the same axis without one",
        )
        _require(len({minted, bare}) == 1, "one axis, one set member")

    def test_the_exclusion_is_MANDATORY_not_a_taste_ruling(self):
        """G1c ⭐ — WHY the generator may never enter the key.

        A ``Quadrature`` is unhashable (``frozen=False`` dataclass with
        ``eq=True`` sets ``__hash__ = None``) and a ``DiscreteMeasure``
        is un-``==``-able (frozen ``eq=True`` over ndarrays raises the
        truth-value ambiguity); an identity key containing either would
        make ``Axis.__eq__`` RAISE and ``hash(Axis)`` RAISE, not merely
        disagree. This gate pins the two properties the exclusion rests
        on, so a future "tidy the field into the key" is refuted by a
        red rather than discovered by a traceback.
        """
        q1, q2 = Quadrature.gauss_legendre(4), Quadrature.gauss_legendre(4)
        with pytest.raises(TypeError, match="unhashable"):
            hash(q1)
        with pytest.raises(ValueError, match="ambiguous"):
            _ = q1.measure == q2.measure


class TestG3MintFidelity:
    """G3 — ``quad.axis(label)`` ≡ the literal mint, plus provenance.

    Independence, stated honestly: the two sides are NOT independent on
    the weight VALUES (``q.weights`` is ``measure.weights``, which is
    what ``DiscreteMeasure.axis`` reads). What this gate genuinely pins
    is the THREADING — label, shape spelling (``(n_points,)`` vs
    ``(quad.N,)``: same value, different producer chain), ``kind``, the
    ``weights=`` wiring, and that ``replace`` preserves the canonicalized
    bytes. The independent anchor for the weight VALUES is the
    quadrature weight-sum / moment-exactness suite
    (``tests/numerics/test_quadrature_directional.py``).
    """

    @pytest.mark.parametrize("name,factory", _RULES)
    def test_quad_axis_is_the_literal_construction_plus_provenance(
        self, name, factory
    ):
        q = factory()
        minted = q.axis("angular")
        literal = Axis(
            "angular", (q.N,), weights=np.asarray(q.weights, float),
            kind=BasisKind.NODAL,
        )
        _require(minted.label == literal.label == "angular", f"{name}: label")
        _require(minted.shape == literal.shape == (q.N,), f"{name}: shape")
        # the MODAL mutation's catcher:
        _require(minted.kind is BasisKind.NODAL, f"{name}: kind must be NODAL")
        _require(
            (minted.weights is None) == (literal.weights is None),
            f"{name}: canonicalization diverged between mint and literal",
        )
        if minted.weights is not None:
            assert literal.weights is not None
            _require(
                minted.weights.tobytes() == literal.weights.tobytes(),
                f"{name}: the weight BYTES feed the digest — array_equal "
                f"is not enough",
            )
            _require(
                not minted.weights.flags.writeable,
                f"{name}: the read-only canonicalization must survive replace()",
            )
        # the drop-the-upgrade mutation's catcher:
        _require(
            minted.generator is q,
            f"{name}: the generator is the QUADRATURE, not the measure "
            f"(got {type(minted.generator).__name__})",
        )


class TestG4TheFourNamesAnswerThroughTheSpace:
    """G4 ⭐ — the done-when's core, verbatim: ``mu_x`` / ``eta`` / ``mu_z``
    / ``level_indices`` are reachable from a SPACE, with no quadrature in
    the caller's hand. A numerics-layer gate because the CONTRACT is
    numerics-layer — it builds a ``FunctionSpace``, not an ``SNMesh``.

    ⚠ ``eta`` rows are CONTRACT rows, not re-point coverage: [M] ``eta``
    has no solve-time consumer outside the mesh (verification plan R6).
    """

    @pytest.mark.parametrize("name,factory", _RULES)
    @pytest.mark.parametrize("attr", ["mu_x", "eta", "mu_z"])
    def test_direction_cosines_answer_through_the_space(self, name, factory, attr):
        q = factory()
        space = FunctionSpace.of_axes(q.axis("angular"), _point())
        g = space.axis("angular").generator
        assert g is not None
        npt.assert_array_equal(
            getattr(g, attr), getattr(q, attr),
            err_msg=f"{name}.{attr} lost through the space",
        )

    @pytest.mark.parametrize("name,factory", _RULES)
    def test_level_indices_answers_through_the_space(self, name, factory):
        """⛔ The name a bare DiscreteMeasure CANNOT answer.

        ``level_indices`` lives on the LevelStructure side-channel, not
        in ``(nodes, weights, support, invariance_group, exactness)`` —
        which is exactly why the angular generator is the Quadrature and
        not the measure (``scratch/cs5_ground_measure.md`` §L). This row
        is that ruling's witness.
        """
        q = factory()
        g = (
            FunctionSpace.of_axes(q.axis("angular"), _point())
            .axis("angular")
            .generator_as(Quadrature, consumer="G4 gate")
        )
        got, want = g.level_indices, q.level_indices
        _require(len(got) == len(want), f"{name}: level count")
        for k, (a, b) in enumerate(zip(got, want)):
            npt.assert_array_equal(a, b, err_msg=f"{name} level {k}")

    @pytest.mark.parametrize("name,factory", _RULES)
    def test_the_full_generator_surface_answers_through_the_space(
        self, name, factory
    ):
        """The done-when's four names are the HARD ones, not the whole
        reach-past set — [M] ``N`` (12 reads) and ``weights`` (12) reach
        past too (ground memo §I). The generator answers the FULL
        ``AngularMeasure`` surface, ``level_structure`` included, so the
        P4-remainder re-point never finds a name this contract lacks.
        """
        q = factory()
        g = (
            FunctionSpace.of_axes(q.axis("angular"), _point())
            .axis("angular")
            .generator_as(Quadrature, consumer="G4 gate")
        )
        _require(g.N == q.N, f"{name}: N")
        npt.assert_array_equal(g.weights, q.weights, err_msg=f"{name}: weights")
        _require(
            g.level_structure is q.level_structure,
            f"{name}: level_structure must be the rule's own side-channel "
            f"(None on slab/2-D rules is the honest reading)",
        )

    def test_the_bare_measure_cannot_answer_the_fourth_name(self):
        """The refutation leg — do NOT weaken the done-when to three.

        A ``DiscreteMeasure``-generated axis answers three of four; the
        fourth is the whole reason ``Quadrature.axis`` exists. If a
        measure starts answering ``level_indices``, this design must be
        re-ruled — that is a signal, not a convenience.
        """
        q = Quadrature.level_symmetric(4)
        measure_axis = q.measure.axis("angular")
        _require(
            not hasattr(measure_axis.generator, "level_indices"),
            "a DiscreteMeasure must NOT answer level_indices — if it "
            "starts to, Quadrature.axis's raison d'etre changed and this "
            "design must be re-ruled",
        )


class TestG8TheMintIsASectionOfTheForgetfulMap:
    """G8 ⭐ — the intrinsic law: for every generator-ful axis ``a``,
    ``a.generator.axis(a.label) == a`` (the mint is a SECTION of the
    forgetful map, so the forgetting is recoverable).

    ⚠ Scoped honestly: the law ranges over GENERATOR-FUL axes only. A
    generator-less axis (the homogeneous counting point, every
    EnergyAxis, the MODAL moment axis) is outside the domain, not a
    counterexample. The spatial round-trip rows live beside G6 in
    ``test_space_of_axes.py`` — the rank-d arm is generator-less by
    contract (the CS2 seam), so only the rank-1 arm is in the law's
    domain.

    Mutation: mint at a shape other than ``(generator.n_points,)``, or
    hand-pass a ``generator=`` that did not produce the axis — this is
    the gate that catches the rank-d mismatch (verification plan R1).
    """

    @pytest.mark.parametrize("name,factory", _RULES)
    def test_angular_round_trip(self, name, factory):
        a = factory().axis("angular")
        assert a.generator is not None
        _require(a.generator.axis(a.label) == a, f"{name}: not a section")
        _require(
            a.generator.axis(a.label).generator is a.generator,
            f"{name}: the round-trip must preserve the generator itself",
        )

    def test_measure_round_trip(self):
        """The NODAL-measure leg of the same law (spatial-shaped)."""
        from orpheus.numerics.measure import DiscreteMeasure

        m = DiscreteMeasure(
            nodes=np.array([0.5, 1.5, 2.5]),
            weights=np.array([0.2, 0.3, 0.5]),
            support=RealSpace(1),
        )
        a = m.axis("spatial")
        _require(a.generator is m, "the mint must record its generator")
        _require(m.axis(a.label) == a, "measure mint: not a section")


class TestG5GeneratorAsIsTheOneRefusalHome:
    """P4-remainder G5 — the typed narrow with the by-name refusal.

    The accessor is load-bearing, not decorative: the bare union cannot
    answer the consumers' reads (a DiscreteMeasure has no ``mu_x`` /
    ``level_indices``), so every re-pointed read narrows here — one home,
    and the consumer's own name rides into the message. Per-consumer
    refusal rows live beside their consumers
    (``tests/sn/mesh/test_reduced_operator.py``,
    ``tests/sn/sweep/curvilinear/test_angular_closure.py``).
    """

    def test_the_narrow_returns_the_generator_itself(self):
        """G5.1's Axis-tier half — positive first (vv #11)."""
        q = Quadrature.gauss_legendre(4)
        a = q.axis()
        assert a.generator_as(Quadrature, consumer="test") is q

    def test_a_generator_less_axis_refuses_with_both_names(self):
        q = Quadrature.gauss_legendre(4)
        bare = Axis(
            "angular", (q.N,), weights=np.asarray(q.weights, float),
            kind=BasisKind.NODAL,
        )
        with pytest.raises(ValueError, match="angular"):
            bare.generator_as(Quadrature, consumer="somebody")
        with pytest.raises(ValueError, match="somebody"):
            bare.generator_as(Quadrature, consumer="somebody")
        with pytest.raises(ValueError, match="minted through"):
            bare.generator_as(Quadrature, consumer="somebody")

    def test_a_wrong_KIND_generator_refuses_too(self):
        """A measure-minted axis narrowed to Quadrature refuses — the
        narrow is a type claim, not a None check (parse, don't validate).
        The message names what it got."""
        from orpheus.numerics.measure import DiscreteMeasure

        m = DiscreteMeasure(
            nodes=np.array([0.5, 1.5]), weights=np.array([0.4, 0.6]),
            support=RealSpace(1),
        )
        a = m.axis("spatial")
        with pytest.raises(ValueError, match="DiscreteMeasure"):
            a.generator_as(Quadrature, consumer="test")

    def test_fragment_disjointness_with_the_space_refusal(self):
        """G5.4 (L43c) — the neighbouring refusal on the same consumer
        path spells 'axis lookup:'; this message must not, so a pin on
        either can never match the other's raise."""
        q = Quadrature.gauss_legendre(4)
        bare = Axis(
            "angular", (q.N,), weights=np.asarray(q.weights, float),
            kind=BasisKind.NODAL,
        )
        try:
            bare.generator_as(Quadrature, consumer="x")
        except ValueError as e:
            assert "axis lookup:" not in str(e)
        else:  # pragma: no cover
            pytest.fail("must refuse")

    def test_G5_6a_the_shipped_generator_less_axes_ARE_generator_less(self):
        """The inventory row: the honest-None sites answer None and the
        accessor WOULD bite them — proof the guard has teeth wherever it
        is wired. (Shelf life: the MODAL moment axis retires from this
        list when §5.5 item 3 lands; the counting point and every
        EnergyAxis remain.)"""
        from orpheus.numerics.axis import EnergyAxis

        counting = Axis("spatial", (1,), kind=BasisKind.NODAL)
        energy = EnergyAxis.synthetic(3)
        for ax in (counting, energy):
            assert ax.generator is None
            with pytest.raises(ValueError, match="minted through"):
                ax.generator_as(Quadrature, consumer="inventory")

    def test_G5_6b_the_call_site_set_is_the_declared_consumer_set(self):
        """The 'not on their path' row (AST, the tree's own idiom): the
        accessor's production call sites are exactly the declared
        consumers — it reds the moment someone wires the refusal onto a
        path a shipped generator-less axis travels (the homogeneous
        pose, the energy family, the MODAL moment axis)."""
        import ast
        import pathlib

        declared = {
            ("orpheus/sn/mesh/reduced_operator.py", "streaming_terms"),
            ("orpheus/sn/angular/closure.py", "__init__"),  # MM + Identity
            # CS4c §14.4 — the blessed frame chain's one production hop:
            # space → angular axis → generator_as(Quadrature) → interned
            # angular_frame(L).  Every frame consumer (S, F, windowing)
            # reaches the quadrature THROUGH this classmethod, so the
            # refusal fires only where a space genuinely lost its
            # generator (a hand-built axis fed to a frame mint).
            ("orpheus/transport/frames/harmonic_frame.py", "for_space"),
            # CS4c step 6 item 6.2c-ii (hazard H-6): the moment-codomain
            # derivation refuses a MOMENT space by the same channel — a
            # moment space is axis-built too since then, so the axes-less
            # refusal no longer catches it; its leading axis is a MODAL head
            # whose generator is a basis or a frame, never the quadrature.
            ("orpheus/transport/frames/harmonic_frame.py", "moment_space_on"),
            # (An n2n.py from_solver_data entry lived here §14.1–§16.4:
            # the tier-2 mint recovered bare WEIGHTS through this channel
            # until CS4c step 4's harmonization retired the weights field
            # — the frame now arrives through for_space above, so the
            # site DISSOLVED and this gate caught the stale declaration
            # at the step-4 exit gate, exactly as designed.)
        }
        found = set()
        for f in pathlib.Path("orpheus").rglob("*.py"):
            tree = ast.parse(f.read_text())
            encl: dict[int, str] = {}

            def walk(node, name):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    name = node.name
                    for line in range(
                        node.lineno, (node.end_lineno or node.lineno) + 1
                    ):
                        encl.setdefault(line, name)
                for ch in ast.iter_child_nodes(node):
                    walk(ch, name)

            walk(tree, "<module>")
            for n in ast.walk(tree):
                if (
                    isinstance(n, ast.Attribute)
                    and n.attr == "generator_as"
                ):
                    found.add(
                        (str(f), encl.get(n.lineno, "<module>"))
                    )
        assert found == declared, (
            f"generator_as call-site set drifted:\n"
            f"  extra: {sorted(found - declared)}\n"
            f"  missing: {sorted(declared - found)}\n"
            f"a new consumer is fine — declare it here; a consumer on a "
            f"generator-less production path is not."
        )
