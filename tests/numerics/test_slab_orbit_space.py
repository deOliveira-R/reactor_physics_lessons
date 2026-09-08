r"""Tracker 2.4 — the slab says what space its ordinates live on.

The keystone of ``.claude/plans/angular_spaces_derived_from_symmetry.md``
tracker **2.4** (issue #429), and the reason it is a defect FIX rather
than wiring. Until 2026-09-01 the slab's angular rule declared its support
as ``Interval(-1, 1)`` — the CHART a quotient map happens to land on — and
so did every 1-D spatial rule on the same interval. ``FunctionSpace``
compares by ``(name, shape)``, so `[M]` an 8-node slab ANGULAR space and an
8-node SPATIAL space were ``==`` **and hash-equal**: an angular flux and a
spatial coefficient vector were the same object to every composability
check in the tree. That is tracker 2.1's energy/spatial collision
(``L2[coarse_cells_R1]``) one level up, and 2.0c could not close it — the
retype made the supports honest, and they were honestly identical.

The repair is a DECLARATION: the slab's rule is read on the orbit space
:math:`S^2/O(2)_x` (:meth:`DiscreteMeasure.on_orbit_space`, adopted by
:func:`gauss_legendre_on_polar_orbit`), whose chart is the same interval.
No coordinate changes; what the measure knows about itself does, and with
it the space name, the phase, the spent group, and the registry's stage-0
admission — which now compares an orbit space that names its group against
a geometry that names the group it spends.

Sections:

* **A** — the keystone: the collision is unspellable, and the declaration
  changes no number.
* **B** — the axis is load-bearing (Part IV obstacle 1, resolved by a
  derivation rather than a prediction): stage 0 refuses a chart-level rule
  AND a rule about the wrong axis.
* **C** — ``on_orbit_space``'s contract, both legs (``vv-principles`` #11).
* **D** — three axes, three orbit spaces, one derivation, memoised.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from orpheus.numerics.manifold import (
    COSINE_INTERVAL,
    SPHERE,
    UNIT_INTERVAL,
    Interval,
    Quotient,
)
from orpheus.numerics.measure import DiscreteMeasure
from orpheus.numerics.quadrature import (
    Quadrature,
    gauss_legendre_on_mu,
    gauss_legendre_on_polar_orbit,
    select_quadrature,
)
from orpheus.numerics.quadrature.registry import GEOMETRY_ANGULAR_SYMMETRY
from orpheus.numerics.symmetry import SubgroupOfO3

_AXES = ("x", "y", "z")


def _spatial_rule_on(interval: Interval, n: int) -> DiscreteMeasure:
    """An ``n``-node SPATIAL rule on ``interval`` — a cell-centre rule with
    the same node count as the angular one it must not be confused with."""
    edges = np.linspace(interval.a, interval.b, n + 1)
    return DiscreteMeasure(
        nodes=0.5 * (edges[1:] + edges[:-1]),
        weights=np.diff(edges),
        support=interval,
    )


# ============================================================================
# A. The keystone
# ============================================================================


@pytest.mark.foundation
@pytest.mark.parametrize("n", [2, 8, 16])
def test_a1_a_slab_angular_space_is_not_a_spatial_space_on_its_chart(
    n: int,
) -> None:
    r"""The live defect, closed: same interval, same node count, DIFFERENT
    spaces — because the slab's rule names the orbit space it lives on
    rather than the chart it was built on.

    The positive control is the half that makes this a measurement rather
    than a tautology: two SPATIAL rules on the interval with ``n`` nodes
    ARE equal, so shape and point set are not what separates the two —
    the ROLE is, and the role is now spelled in the support.
    """
    angular = Quadrature.gauss_legendre(n).measure
    spatial = _spatial_rule_on(Interval(-1.0, 1.0), n)

    assert angular.space.shape == spatial.space.shape == (n,)
    assert angular.space != spatial.space
    assert len({angular.space, spatial.space}) == 2   # separation through the container
    assert angular.space.name == f"L2[S^2/O2_x]"
    assert spatial.space.name == "L2[[-1,1]]"

    # positive control: role, not shape, is the discriminator
    other_spatial = _spatial_rule_on(Interval(-1.0, 1.0), n)
    assert spatial.space == other_spatial.space
    assert hash(spatial.space) == hash(other_spatial.space)


@pytest.mark.foundation
@pytest.mark.parametrize("n", [2, 8, 16])
def test_a2_the_declaration_changes_no_coordinate(n: int) -> None:
    """Nodes and weights are bit-identical to the chart-level rule's, the
    orbit space contains them, and the exactness claim (a claim about the
    chart's reference measure) survives the declaration."""
    declared = Quadrature.gauss_legendre(n).measure
    chart = gauss_legendre_on_mu(n)

    np.testing.assert_array_equal(declared.nodes, chart.nodes)
    np.testing.assert_array_equal(declared.weights, chart.weights)
    assert isinstance(declared.support, Quotient)
    assert declared.support.realization == chart.support == COSINE_INTERVAL
    assert declared.support.contains(declared.nodes.reshape(-1, 1))
    assert declared.exactness == chart.exactness
    assert declared.exactness is not None
    assert declared.exactness.reference.name == "legendre"


@pytest.mark.foundation
def test_a3_the_slab_answers_from_its_manifold_alone() -> None:
    """The ``phase`` and the spent group are DERIVED from the support: strip
    the residual-symmetry tag and the slab still knows it is angular and
    what it was quotiented by. Discharges the 2.0c pre-flight's question
    about the ``invariance_group`` fallback arm for the slab — and records,
    with a witness, that the arm is NOT dead: the chart-level μ-rule (the
    product rules' polar factor, which cannot declare an axis) still
    reaches it and is still angular there."""
    slab = Quadrature.gauss_legendre(8).measure
    stripped = replace(slab, invariance_group=None)
    assert stripped.phase == "angular"
    assert stripped.quotient_group == SubgroupOfO3.O2("x")
    assert slab.invariance_group == SubgroupOfO3.Mirror("x")

    chart = gauss_legendre_on_mu(8)
    assert chart.invariance_group == SubgroupOfO3.Mirror("x")
    assert chart.phase == "angular"                  # via the fallback arm
    assert chart.quotient_group is None              # it names no axis
    with pytest.raises(NotImplementedError, match="phase is undetermined"):
        _ = replace(chart, invariance_group=None).phase


# ============================================================================
# B. The axis is load-bearing
# ============================================================================


@pytest.mark.foundation
@pytest.mark.parametrize("axis", _AXES)
def test_b1_the_marginal_is_invariant_under_the_group_it_was_quotiented_by(
    axis: str,
) -> None:
    r"""Part IV obstacle 1, answered by DERIVATION: a finite point set is
    :math:`SO(2)_a`-closed iff every node lies on axis :math:`a`; a
    marginal declared on :math:`S^2/SO(2)_a` embeds along :math:`a`; hence
    it is invariant under its own spent group and under no other axis's.
    `[M]` the retired bare ``SO2`` answered ``False`` on the slab, because it
    was realized about :math:`z` while the slab embeds along :math:`x`."""
    m = gauss_legendre_on_polar_orbit(8, axis)
    assert m.quotient_group == SubgroupOfO3.O2(axis)
    assert m.quotient_group is not None
    assert m.is_invariant_under(m.quotient_group)
    for other in _AXES:
        assert m.is_invariant_under(SubgroupOfO3.SO2(other)) is (other == axis)
    # the residual the adopter re-tags is the mirror normal to the SAME axis
    assert m.invariance_group == SubgroupOfO3.Mirror(axis)
    assert m.is_invariant_under(SubgroupOfO3.Mirror(axis))


@pytest.mark.foundation
@pytest.mark.parametrize("geometry", ["slab", "sphere"])
def test_b2_stage_0_refuses_the_chart_level_rule_and_the_wrong_axis(
    geometry: str,
) -> None:
    """The §6c witness for the axis parameter: two inputs that exist in the
    tree today and that the gate must reject. A rule that names no axis
    (the chart-level ``gauss_legendre_on_mu``) and a rule about the WRONG
    axis are both refused at stage 0; only the declared x-marginal is
    admitted. Before tracker 2.4 the first of these was the registered
    slab rule."""
    symmetry = GEOMETRY_ANGULAR_SYMMETRY[geometry]
    assert symmetry.support == SPHERE.quotient(SubgroupOfO3.O2("x"))

    assert symmetry.admits_domain(Quadrature.gauss_legendre(8).measure)
    assert symmetry.admits_domain(gauss_legendre_on_polar_orbit(8, "x"))
    assert not symmetry.admits_domain(gauss_legendre_on_mu(8))
    assert not symmetry.admits_domain(gauss_legendre_on_polar_orbit(8, "z"))
    assert not symmetry.admits_domain(gauss_legendre_on_polar_orbit(8, "y"))


@pytest.mark.foundation
def test_b3_the_selector_still_hands_the_slab_its_declared_rule() -> None:
    """End to end: selection for a slab returns a measure on the slab's
    orbit space, with the reference the geometry asks for."""
    measure, log = select_quadrature("slab", target_degree=7)
    assert log.chosen_spec is not None
    assert log.chosen_spec.name == "GaussLegendre1D"
    assert measure.support == SPHERE.quotient(SubgroupOfO3.O2("x"))
    assert measure.space.name == "L2[S^2/O2_x]"
    assert measure.exactness is not None
    assert measure.exactness.reference == GEOMETRY_ANGULAR_SYMMETRY["slab"].reference


# ============================================================================
# C. on_orbit_space — both legs
# ============================================================================


@pytest.mark.foundation
def test_c1_a_rule_on_the_chart_is_read_on_the_orbit_space() -> None:
    """Positive leg: the atoms are unchanged, the support becomes the orbit
    space, the embedding-dependent tag is dropped, the chart-level exactness
    claim is kept."""
    chart = gauss_legendre_on_mu(6)
    orbit = SPHERE.quotient(SubgroupOfO3.O2("y"))
    read = chart.on_orbit_space(orbit)

    assert read.support is orbit
    np.testing.assert_array_equal(read.nodes, chart.nodes)
    np.testing.assert_array_equal(read.weights, chart.weights)
    assert read.invariance_group is None
    assert read.exactness == chart.exactness
    assert read.phase == "angular"
    assert read.quotient_group == SubgroupOfO3.O2("y")


@pytest.mark.foundation
def test_c2_the_wrong_chart_is_refused_where_the_declaration_is_written() -> None:
    """Negative leg: a rule can only be read on an orbit space whose chart
    it was built on. A rule on the unit interval is not a rule on
    :math:`S^2/SO(2)`, and a rule on the SPHERE is not read on
    :math:`S^2/\\sigma_y` this way — that is a fold, and the fold is
    :meth:`DiscreteMeasure.quotient`."""
    on_unit = DiscreteMeasure(
        nodes=np.array([0.25, 0.75]), weights=np.array([0.5, 0.5]),
        support=UNIT_INTERVAL,
    )
    o2_x = SPHERE.quotient(SubgroupOfO3.O2("x"))
    with pytest.raises(ValueError, match=r"\[0,1\].*S\^2/O2_x.*\[-1,1\]"):
        on_unit.on_orbit_space(o2_x)

    on_sphere = Quadrature.lebedev(order=5).measure
    with pytest.raises(ValueError, match="use quotient\\(\\)"):
        on_sphere.on_orbit_space(SPHERE.quotient(SubgroupOfO3.Mirror("y")))


# ============================================================================
# D. Three axes, three orbit spaces, one derivation
# ============================================================================


@pytest.mark.foundation
@pytest.mark.verifies("manifold-s2-mod-so2")
def test_d1_three_axes_three_quotients_one_derivation() -> None:
    r"""The catalogue derives :math:`S^2/O(2)_a` for each axis from ONE
    procedure that reads the axis off the group: the invariants are
    :math:`p_1 = x_a` and :math:`p_2 = x_b^2 + x_c^2`, the realization is
    :math:`[-1,1]` in every case, and the three results are three different
    quotients — unequal, differently named, differently derived."""
    import sympy as sp

    x, y, z = sp.symbols("x y z", real=True)
    p1, p2 = sp.symbols("p1 p2", real=True)
    coord = {"x": x, "y": y, "z": z}

    quotients = {a: SPHERE.quotient(SubgroupOfO3.O2(a)) for a in _AXES}
    for a, q in quotients.items():
        assert q.name == f"S^2/O2_{a}"
        assert q.realization == COSINE_INTERVAL
        assert q.by == SubgroupOfO3.O2(a)
        others = [coord[b] for b in _AXES if b != a]
        assert sp.simplify(q.generators[0] - (p1 - coord[a])) == 0
        assert sp.simplify(
            q.generators[1] - (p2 - (others[0] ** 2 + others[1] ** 2))
        ) == 0
        assert sp.simplify(q.det_gram - 4 * p2) == 0
    for a in _AXES:
        for b in _AXES:
            assert (quotients[a] == quotients[b]) is (a == b)


@pytest.mark.foundation
def test_d2_the_catalogue_memoises_and_the_orbit_space_is_hashable() -> None:
    """An orbit space is derived once and recorded — the catalogue's own
    philosophy, and since 2.4 every slab quadrature carries one, so a
    symbolic derivation per construction would sit on every slab solve's
    path. A frozen value type that cannot be hashed is a contradiction; the
    Gram matrix is immutable so it can."""
    a = SPHERE.quotient(SubgroupOfO3.O2("x"))
    b = SPHERE.quotient(SubgroupOfO3.O2("x"))
    assert a is b
    assert hash(a) == hash(b)
    # a fresh, un-memoised derivation agrees with the recorded one
    from orpheus.numerics.manifold import _sphere_mod_o2

    fresh = _sphere_mod_o2(SPHERE, SubgroupOfO3.O2("x"))
    assert fresh is not a and fresh == a and hash(fresh) == hash(a)
    # and the derivation REFUSES the rotation half, naming the stabiliser
    # (#432): S^2/SO(2)_x IS this entry, so it is not a second one.
    with pytest.raises(ValueError, match="is the orbit space S\\^2/O2_x"):
        _sphere_mod_o2(SPHERE, SubgroupOfO3.SO2("x"))
    with pytest.raises(ValueError, match="is the orbit space S\\^2/O2_x"):
        SPHERE.quotient(SubgroupOfO3.SO2("x"))
