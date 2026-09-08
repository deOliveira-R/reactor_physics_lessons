r"""Pre-carve anchors for the CS4c step-6 per-class space-identity carve (§7.2 / F3).

**What this file is.** Campaign CS4c step 6 item 6.1 retires the *S3 identity
bridge* on ``of_axes``-built spaces: today space identity is ``(name, shape)``
and the name is a blake2b digest of the axes' structural content
(:meth:`orpheus.numerics.space.FunctionSpace.of_axes`), so structural equality
holds *through* a nominal comparison.  After the carve the ``of_axes`` family
compares its axes tuple directly and the axes-less classes keep their own
digests.

Every row below states a law that is TRUE ON TODAY'S UNMODIFIED TREE, so the
carve inherits a measured baseline instead of a hoped-for one
(``plan-authoring`` 2026-08-28: build the acceptance artefact PRE-carve and
prove it ACTIVATES).  Rows the carve must *introduce* are specified in the
verification memo and are deliberately NOT here — a gate that cannot fail
today and cannot fail after is not an anchor.

**Activation evidence (mutation battery, 2026-09-07, monkeypatch-only,
scope = tests/numerics + tests/transport + tests/sn/architecture +
tests/sn/operators = 5550 passed / 2 skipped / 16 xfailed in 117.3 s).**  Red
counts are over that whole scope, not over this file:

===========================================  ======  ==================================
arm                                          reds    what it models
===========================================  ======  ==================================
``FunctionSpace.__eq__ -> True``                85+1E POSITIVE CONTROL
``FunctionSpace.__eq__`` drops ``name``          39   identity moved to shape only
``of_axes`` digest -> ``CONST<shape>``           20   every axis product collides
``Axis._identity_key`` drops ``weights``         20   measure-blind axis identity
``FunctionSpace.__hash__ -> 0``                   6   eq/hash consistency (see below)
===========================================  ======  ==================================

⚠ The ``__hash__ -> 0`` arm is the one to read carefully.  A constant hash is
LEGAL Python (``a == b`` only implies ``hash(a) == hash(b)``, never the
converse), so those 6 reds are tests asserting ``hash(a) != hash(b)`` as a
"these are different spaces" leg — ``.claude/agent-memory/test-architect``
lesson L70a, which says that leg reds on correct code.  This file therefore
asserts separation THROUGH THE CONTAINER (``len({a, b}) == 2``) and never by
comparing hashes, and pins the eq⟹hash implication in the only direction that
is a law.

⛔ **Deliberately absent: "≠ across subclasses with equal ``(name, shape)``".**
MEASURED today: a hand-built ``DualSpace(name="X", shape=(4,), primal=…)``
compares EQUAL to ``FunctionSpace("X", (4,))``, and so does a
``TensorProductSpace`` with the same derived name.  That is a behaviour the
carve *introduces*; asserting it here would ship a red.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from orpheus.numerics.axis import Axis, BasisKind, EnergyAxis
from orpheus.numerics.quadrature.rules_1d import gauss_legendre_on_mu
from orpheus.numerics.space import DualSpace, FunctionSpace, TensorProductSpace

pytestmark = pytest.mark.foundation


# ── Fixture family — two axis KINDS, two shapes, weighted and unweighted ──
#
# A single (label, shape) pair would leave the digest's injectivity untested
# on the only dimension that matters (lessons L59e: an injectivity gate needs
# at least one pair whose SHAPES are identical, else `shape` carries the
# discrimination and the NAME is never tested).

def _spatial(weights: "list[float] | None") -> Axis:
    return Axis(
        label="spatial",
        shape=(3,),
        weights=None if weights is None else np.asarray(weights, dtype=float),
        kind=BasisKind.NODAL,
    )


def _energy(ng: int = 2) -> EnergyAxis:
    return EnergyAxis(label="energy", shape=(ng,), kind=BasisKind.NODAL)


_W = [1.0, 2.0, 3.0]
_W_PERTURBED = [1.0, 2.0, 3.5]          # SAME shape, different measure


def _axis_families() -> "list[tuple[str, tuple[Axis, ...], tuple[Axis, ...]]]":
    """``(label, axes_a, axes_b)`` where ``a`` and ``b`` are content-equal
    tuples built through two INDEPENDENT mints (never one object reused —
    ``vv`` #22's shared-object axis)."""
    return [
        ("energy2-spatial3-weighted", (_energy(), _spatial(_W)), (_energy(), _spatial(_W))),
        ("energy4-spatial3-euclidean", (_energy(4), _spatial(None)), (_energy(4), _spatial(None))),
        ("spatial3-only", (_spatial(_W),), (_spatial(_W),)),
    ]


_FAMILIES = _axis_families()
_FAMILY_IDS = [f[0] for f in _FAMILIES]


# ═════════════════════════════════════════════════════════════════════════
# G1.1 — the five laws that hold TODAY
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(("_label", "axes_a", "axes_b"), _FAMILIES, ids=_FAMILY_IDS)
def test_g1_1a_two_independent_mints_of_equal_axes_are_the_same_space(
    _label, axes_a, axes_b,
):
    """Reflexivity, symmetry and content equality across two mints.

    ``axes_a`` and ``axes_b`` are built by separate calls, so nothing is
    shared between the two sides but the CONTENT — the property the carve
    must preserve when the digest stops being the identity.
    """
    a = FunctionSpace.of_axes(*axes_a)
    b = FunctionSpace.of_axes(*axes_b)
    if a is b:
        pytest.fail("the two mints returned one object — the row compares a space with itself")
    if not (a == a and b == b):
        pytest.fail(f"reflexivity failed: {a!r}")
    if not (a == b and b == a):
        pytest.fail(f"content-equal axis tuples minted unequal spaces: {a!r} vs {b!r}")


@pytest.mark.parametrize(("_label", "axes_a", "axes_b"), _FAMILIES, ids=_FAMILY_IDS)
def test_g1_1b_equality_implies_equal_hash(_label, axes_a, axes_b):
    """``a == b ⟹ hash(a) == hash(b)`` — the ONLY direction that is a law.

    The converse (``a != b ⟹ hash(a) != hash(b)``) is NOT asserted anywhere in
    this file: a constant hash is legal and would make such a row a false red
    on correct code (lesson L70a).  Separation is asserted through the
    container instead — see :func:`test_g1_2_distinct_spaces_survive_a_set`.
    """
    a = FunctionSpace.of_axes(*axes_a)
    b = FunctionSpace.of_axes(*axes_b)
    assert a == b, "precondition: the two mints must be equal for this law to apply"
    if hash(a) != hash(b):
        pytest.fail(f"equal spaces hash differently: {hash(a)} vs {hash(b)} ({a!r})")


def test_g1_1c_a_different_measure_on_one_axis_makes_a_different_space():
    """The measure is part of the identity — SAME label, SAME shape, one
    weight changed.

    ``spatial(3,)`` with weights ``[1, 2, 3]`` vs ``[1, 2, 3.5]``: the shapes
    are identical, so the discrimination CANNOT come from ``shape`` and must
    come from the axis content (today: through the digest; after the carve:
    through the axes tuple).  This is the row the
    ``Axis._identity_key`` drops-``weights`` arm reddens (20 reds tree-wide).
    """
    a = FunctionSpace.of_axes(_energy(), _spatial(_W))
    b = FunctionSpace.of_axes(_energy(), _spatial(_W_PERTURBED))
    if a.shape != b.shape:
        pytest.fail("CONTROL INVALID: the two spaces differ in shape, so the row is vacuous")
    if a == b:
        pytest.fail(
            f"two spaces differing only in one axis's MEASURE compared equal: "
            f"{a!r} == {b!r} — the measure has dropped out of space identity"
        )


def test_g1_1d_axis_ORDER_is_part_of_the_identity():
    """Two axes of the same shape, composed in the two orders, are different
    spaces — the total shape is identical, so only the ORDER discriminates."""
    u = Axis(label="u", shape=(2,), weights=None, kind=BasisKind.NODAL)
    v = Axis(label="v", shape=(2,), weights=None, kind=BasisKind.NODAL)
    a = FunctionSpace.of_axes(u, v)
    b = FunctionSpace.of_axes(v, u)
    if a.shape != b.shape:
        pytest.fail("CONTROL INVALID: the shapes differ, so the row does not test ORDER")
    if a == b:
        pytest.fail(f"axis order dropped out of the identity: {a!r} == {b!r}")


def test_g1_1e_comparison_against_a_non_space_is_notimplemented():
    """The raw dunder returns ``NotImplemented`` (so Python's fallback runs),
    and the ``==`` surface is therefore ``False`` rather than an exception.

    Asserted on the DUNDER, not on ``==``: ``a == 42`` is ``False`` for any
    ``__eq__`` that raises ``TypeError``, returns ``False``, or returns
    ``NotImplemented``, so the ``==`` reading alone cannot see the contract.
    """
    space = FunctionSpace.of_axes(_energy(), _spatial(_W))
    if FunctionSpace.__eq__(space, 42) is not NotImplemented:
        pytest.fail(
            "FunctionSpace.__eq__ against a non-space did not return "
            "NotImplemented — Python's reflected-comparison fallback is bypassed"
        )
    assert (space == 42) is False


# ═════════════════════════════════════════════════════════════════════════
# G1.2 — separation THROUGH THE CONTAINER (never hash(a) != hash(b))
# ═════════════════════════════════════════════════════════════════════════

def test_g1_2_distinct_spaces_survive_a_set():
    """Distinct spaces stay distinct as dict keys / set members, and equal
    ones deduplicate.

    Lesson L70a: ``hash(a) != hash(b)`` is not a legal separation leg (a
    constant hash is a valid implementation).  The container is: it consults
    ``__eq__`` on collision, so ``len({a, b}) == 2`` is a claim about
    equality, not about hashing.
    """
    a = FunctionSpace.of_axes(_energy(), _spatial(_W))
    a_twin = FunctionSpace.of_axes(_energy(), _spatial(_W))
    b = FunctionSpace.of_axes(_energy(), _spatial(_W_PERTURBED))
    c = FunctionSpace.of_axes(_energy(4), _spatial(_W))
    if len({a, a_twin}) != 1:
        pytest.fail("two content-equal spaces did not deduplicate in a set")
    if len({a, b, c}) != 3:
        pytest.fail(
            f"three distinct spaces collapsed in a set (len={len({a, b, c})}): "
            f"{a.name!r} / {b.name!r} / {c.name!r}"
        )
    keyed = {a: "first"}
    keyed[a_twin] = "second"
    if keyed != {a: "second"} or len(keyed) != 1:
        pytest.fail("a content-equal space did not address the same dict slot")


# ═════════════════════════════════════════════════════════════════════════
# G1.4 — WHY the nine delegations exist: no ndarray ever reaches __eq__
# ═════════════════════════════════════════════════════════════════════════

def test_g1_4_comparing_spaces_that_differ_only_in_dense_weights_does_not_raise():
    r"""Two spaces whose only difference is the dense ``inner_product_weights``
    array compare WITHOUT raising.

    This is the reason the eight subclass ``__eq__`` bodies exist at all
    (``space.py:1005-1011``): a frozen dataclass's generated ``__eq__`` would
    compare the ndarray field and raise *"truth value of an array with more
    than one element is ambiguous"*.  The carve rewrites the base body, so
    this row is the PREMISE it must not break (lessons L61a / L73a: gate the
    premise, not only the corollary).
    """
    weights_a = np.array([2.0, 3.0])
    weights_b = np.array([5.0, 7.0])
    a = FunctionSpace(name="same", shape=(2,), inner_product_weights=weights_a)
    b = FunctionSpace(name="same", shape=(2,), inner_product_weights=weights_b)
    try:
        verdict = a == b
    except Exception as exc:                       # noqa: BLE001 — the class IS the measurement
        pytest.fail(
            f"comparing two spaces that differ only in dense weights raised "
            f"{type(exc).__name__}: {exc} — the (name, shape) delegation has "
            f"been lost and a dataclass __eq__ is reaching the ndarray"
        )
    assert verdict is True, "the two spaces share (name, shape); today's identity says equal"

    # The same claim on the two subclasses that carry their own delegation.
    tp_a = TensorProductSpace.from_factors(
        (FunctionSpace(name="A", shape=(2,), inner_product_weights=weights_a),
         FunctionSpace(name="B", shape=(3,))),
    )
    tp_b = TensorProductSpace.from_factors(
        (FunctionSpace(name="A", shape=(2,), inner_product_weights=weights_b),
         FunctionSpace(name="B", shape=(3,))),
    )
    for pair, label in ((( tp_a, tp_b), "TensorProductSpace"),
                        ((DualSpace.of(a), DualSpace.of(b)), "DualSpace")):
        try:
            same_name_and_shape = pair[0] == pair[1]
        except Exception as exc:                   # noqa: BLE001
            pytest.fail(f"{label}.__eq__ raised {type(exc).__name__}: {exc}")
        assert same_name_and_shape is True, f"{label}: the two share (name, shape)"


# ═════════════════════════════════════════════════════════════════════════
# G1.5 — the REASON identity may not move onto measure content
# ═════════════════════════════════════════════════════════════════════════

def test_g1_5_a_discrete_measure_cannot_be_compared_but_an_axis_can():
    r"""``DiscreteMeasure.__eq__`` RAISES; ``Axis.__eq__`` does not.

    ``Axis._identity_key`` deliberately excludes ``generator`` and encodes the
    weights as ``.tobytes()``, so a structural space identity spelled through
    ``Axis`` never reaches measure equality.  This row pins that REASON — it
    is what makes "the carve does not need ``DiscreteMeasure.__eq__`` fixed"
    a checkable statement rather than an assumption (lesson L65b: check
    whether a doctrinal exclusion is also MANDATORY; here it is).
    """
    measure = gauss_legendre_on_mu(4)
    with pytest.raises(ValueError, match="truth value of an array"):
        _ = measure == gauss_legendre_on_mu(4)

    a = Axis(
        label="mu", shape=(4,), weights=measure.weights,
        kind=BasisKind.NODAL, generator=measure,
    )
    b = Axis(
        label="mu", shape=(4,), weights=np.array(measure.weights, copy=True),
        kind=BasisKind.NODAL, generator=gauss_legendre_on_mu(4),
    )
    if a.generator is b.generator:
        pytest.fail("CONTROL INVALID: both axes carry the SAME measure object")
    if not (a == b):
        pytest.fail(
            "two axes with equal content but DIFFERENT generator objects "
            "compared unequal — provenance has leaked into identity"
        )
    assert hash(a) == hash(b)


# ═════════════════════════════════════════════════════════════════════════
# G1.7 — the §6b census: how many __eq__ bodies the space family carries
# ═════════════════════════════════════════════════════════════════════════

_SPACE_FAMILY = (
    "orpheus/numerics/space.py",
    "orpheus/numerics/axis.py",
    "orpheus/numerics/spaces/full_field_space.py",
    "orpheus/numerics/spaces/spherical_harmonic_space.py",
    "orpheus/numerics/spaces/legendre_space.py",
    "orpheus/numerics/spaces/scalar_trace_space.py",
    "orpheus/numerics/spaces/radial_characteristic_space.py",
    "orpheus/numerics/spaces/spatial_moment_space.py",
    "orpheus/numerics/spaces/angular_trace_space.py",
)

#: MEASURED 2026-09-07 at `b889089e`, by the AST census below: TEN ``__eq__``
#: definitions over the nine files, of which NINE delegate to
#: ``FunctionSpace.__eq__`` and one (``Axis``) is already structural.
#: ``angular_trace_space.py`` defines none (it inherits).
_EXPECTED_EQ_SITES = 10
_EXPECTED_DELEGATIONS = 8       # the eight SUBCLASS one-liners (base + Axis excluded)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _eq_census() -> "tuple[list[str], list[str]]":
    """``(every __eq__ site, the sites whose body is the one-line delegation)``."""
    root = _repo_root()
    sites: list[str] = []
    delegations: list[str] = []
    for rel in _SPACE_FAMILY:
        path = root / rel
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source)
        for node in ast.walk(module):
            if not (isinstance(node, ast.FunctionDef) and node.name == "__eq__"):
                continue
            tag = f"{rel}:{node.lineno}"
            sites.append(tag)
            body = [s for s in node.body if not isinstance(s, ast.Expr)]
            if (
                len(body) == 1
                and isinstance(body[0], ast.Return)
                and "FunctionSpace.__eq__" in (ast.unparse(body[0].value) if body[0].value else "")
            ):
                delegations.append(tag)
    return sites, delegations


def test_g1_7_the_space_family_carries_ten_eq_bodies_of_which_eight_delegate():
    """RECORD (not a theorem): the ``__eq__`` population the carve re-keys.

    A RECORD row says *something changed*, never which side is right
    (lessons L55i).  Its value here is that it makes the §6b denominator
    mechanical: the carve is ONE base body plus a per-class decision, not ten
    independent bodies — mutating any one of the eight identical delegations
    is indistinguishable from mutating another.

    The row is expected to be RE-BASELINED by the carve; the count belongs in
    the commit message, and a NEW hand-rolled ``__eq__`` must show up here.
    """
    sites, delegations = _eq_census()
    if not sites:
        pytest.fail("CENSUS BROKEN: no __eq__ found in the space family at all")
    if len(sites) != _EXPECTED_EQ_SITES or len(delegations) != _EXPECTED_DELEGATIONS:
        pytest.fail(
            f"the space family's __eq__ population moved: "
            f"{len(sites)} sites (expected {_EXPECTED_EQ_SITES}), "
            f"{len(delegations)} one-line delegations (expected "
            f"{_EXPECTED_DELEGATIONS}).\n  sites: {sites}\n  delegations: {delegations}"
        )
