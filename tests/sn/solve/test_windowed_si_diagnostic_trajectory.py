r"""The WINDOWED arm's SI diagnostic trajectory — the pin the tree did not have.

**Why this file exists.**  On the windowed (moment) arm the SI increment norm
is the ONLY production consumer of the angular moment space's metric —
`[M]` 2026-09-07, a vv#29 instance census over one 2-D windowed solve
(``scratch/_step6_2c/p6_traffic.py``): **6 `norm` + 6 `inner_product` calls on
the moment `TensorProductSpace`, and ZERO `apply_metric` calls**, reached from
``solver.py:3823 ← iteration.py:801 ← field.py:379``.  And the tree's one pin of
that diagnostic, ``tests/numerics/test_si_diagnostic_trajectory.py``, REFUSES a
windowed fixture by construction (its ``:245-247`` raises *"the 1-D slab
windowed the SI iterate — windowing is 2-D Cartesian"*).  So the moment
metric's effect on the recorded ρ trajectory was seen by **nothing**.

`[M]` the scale of what was unseen: re-running this exact driver with the
frame's Parseval-dressed head in place of the hub's continuum one
(``scratch/_step6_2c/p7_shim_forward.py``) leaves ``scalar_flux``
**bit-identical** and the residual (stopping) trajectory **bit-identical**,
and moves ``increment_norms`` by **91.6 % relative** and ρ by **3.85 %
relative** — a diagnostic-only movement, five orders above this pin's ``rtol``
and invisible to every other gate in the tree.

**The three legs, and why each is needed.**

1. **ROUTE** (``vv`` Mode 11) — a counting spy proves the recorded norms came
   from the MOMENT space and not from some other leaf: the count is asserted
   EXACT against ``len(record.increment_norms)``, never ``> 0``.
2. **VALUE** — the frozen trajectory, at ``rtol = 1e-9``.  Justified below.
3. **DISCRIMINATION** — the frozen numbers are metric-DEPENDENT, and the test
   says so with a measurement rather than a comment: the same vector's norm
   under the two candidate head metrics differs by ≈ ``4π``.  Without this leg
   the value pin is compatible with "the metric is inert", which is false.

**The tolerance is a claim.**  `[M]` the trajectory is bit-identical across two
runs in one process (``p16_windowed_pin.py``: ``array_equal`` True, max rel
0.0), so the floor is FP-reproducibility, not the solve.  ``rtol = 1e-9`` sits
~3 orders above a plausible cross-platform reduction-order drift and **7 orders
below** the 3.85 % ρ movement / 91.6 % norm movement it exists to catch.  It is
NOT read off the solve tolerance, because the pinned quantity is not what the
solve converges (the STOP rides the residual — ``iteration.py:795-801``).

⛔ **This file freezes a RECORD, not a reference** (``lessons`` §4): a red says
*something moved*, with zero information about which side is right.  Its
companion THEOREM is the Parseval identity in
``tests/transport/frames/test_moment_metric_fork_premise.py``.

⛔ **RE-POSED BY CS4c step 6 item 6.2c.**  When the moment head becomes
axis-built, ONE metric is chosen for the ONE moment space.  If the ruling is
the Parseval metric, leg 2's numbers move by the measured 91.6 % / 3.85 % and
leg 3's assertion flips sides — both loudly, which is the point of freezing
them now.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.geometry import BC, CoordSystem
from orpheus.geometry.mesh import Mesh2D
from orpheus.numerics.quadrature import Quadrature
from orpheus.numerics.space import FunctionSpace, TensorProductSpace
from orpheus.numerics.spaces.moment_head import MomentHead
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.sn.solver import solve_sn_fixed_source
from orpheus.transport.frames import HarmonicFrame
from tests.sn.architecture.test_monomorphic_leaves import _two_region_fissile

pytestmark = pytest.mark.foundation

_SN_ORDER = 4
_L = 1
_NG = 2
_INNER_TOL = 1e-6
_MAX_INNER = 60

#: `[M]` 2026-09-07, `main` @ 79d2944a, serial, one process: bit-identical
#: across repeated runs.  22 recorded increments; the head carries the
#: CONTINUUM Gram 4π/(2ℓ+1) (which metric is asserted directly, below).
_INCREMENT_NORMS = (
    4.179894982724e+01, 2.148977886811e+01, 9.603944803076e+00,
    4.341126296059e+00, 2.163202902533e+00, 1.072425668726e+00,
    5.389857653439e-01, 2.744171773514e-01, 1.399923053592e-01,
    7.170869223686e-02, 3.681085282022e-02, 1.891803619665e-02,
    9.732067621689e-03, 5.009504656272e-03, 2.579631202657e-03,
    1.328776432601e-03, 6.845953559090e-04, 3.527602742530e-04,
    1.817907968915e-04, 9.369090752327e-05, 4.828892524695e-05,
)

#: The first three contraction ratios — the shape of the transient.  The tail
#: settles to ≈ 0.5154 (the fixture's scattering ratio); pinning the head is
#: what carries the metric information, the tail is the physics.
_RHO_HEAD = (0.514122458983, 0.446907567640, 0.452014915232)
_RHO_TAIL = 0.515406740350

_RTOL = 1e-9


def _mesh() -> Mesh2D:
    mat_map = np.zeros((4, 3), dtype=int)
    mat_map[2:, :] = 1
    return Mesh2D(
        edges_x=np.array([0.0, 0.1, 0.35, 0.9, 1.6]),
        edges_y=np.array([0.0, 0.2, 0.5, 1.4]),
        mat_map=mat_map, coord=CoordSystem.CARTESIAN,
        bc_xmin=BC("reflective"), bc_xmax=BC("vacuum"),
        bc_ymin=BC("reflective"), bc_ymax=BC("vacuum"),
    )


def _is_moment_space(space: object) -> bool:
    return (
        isinstance(space, TensorProductSpace)
        and bool(space.factors)
        and isinstance(space.factors[0], MomentHead)
    )


@pytest.fixture(scope="module")
def solved():
    """ONE 2-D windowed SI solve, shared by every row (0.6 s cold)."""
    return solve_sn_fixed_source(
        _two_region_fissile(), _mesh(),
        Quadrature.level_symmetric(sn_order=_SN_ORDER),
        np.ones((24, _NG, 4, 3)),
        scattering_order=_L, inner_tol=_INNER_TOL, max_inner=_MAX_INNER,
        inner_schedule="jacobi",
    )


# ═══════════════════════════════════════════════════════════════════════
# Leg 1 — ROUTE: the recorded norms came from the MOMENT space
# ═══════════════════════════════════════════════════════════════════════


def test_every_recorded_increment_norm_is_taken_on_the_moment_space(monkeypatch) -> None:
    r"""A counting spy on ``FunctionSpace.norm``, asserted EXACT.

    ``vv`` Mode 11: a green end-to-end pin proves nothing about WHICH space's
    metric produced it.  The count is pinned to ``len(increment_norms)``, not
    to ``> 0`` — so a solve that stops windowing (or an iterate that stops
    being a moment field) reddens here rather than silently re-pointing the
    value pin below at a different leaf.
    """
    calls: list[tuple[int, ...]] = []
    original = FunctionSpace.norm

    def spy(self, x):
        if _is_moment_space(self):
            calls.append(self.shape)
        return original(self, x)

    monkeypatch.setattr(FunctionSpace, "norm", spy)
    solution = solve_sn_fixed_source(
        _two_region_fissile(), _mesh(),
        Quadrature.level_symmetric(sn_order=_SN_ORDER),
        np.ones((24, _NG, 4, 3)),
        scattering_order=_L, inner_tol=_INNER_TOL, max_inner=_MAX_INNER,
        inner_schedule="jacobi",
    )
    history = solution.history
    assert history is not None, "a windowed SI solve must carry its record"
    record = history.record
    assert len(calls) == len(record.increment_norms) == len(_INCREMENT_NORMS)
    assert set(calls) == {(_L + 1, 2 * _L + 1, _NG, 4, 3)}, (
        "the windowed iterate's space must be the (head ⊗ energy ⊗ spatial) moment product"
    )


def test_the_recorded_trajectory_was_taken_under_the_CONTINUUM_head_metric() -> None:
    r"""WHICH metric the frozen numbers belong to — asserted on the ARRAY.

    This is the fork's own gate (``plan-authoring`` §2: a ruling must be
    visible in the test, not implied by a green).  The hub's moment space
    (:meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.moment_space`) binds the
    basis's CONTINUUM coefficient space, #429 tracker 2.5 Landing A — its head
    carries ``4π/(2ℓ+1)`` in the live slots and 0 in the ``|m| > ℓ`` padding.

    ⛔ RE-POSED BY 6.2c if the fork rules the Parseval metric.
    """
    sn_mesh = SNMesh(
        _mesh(), Quadrature.level_symmetric(sn_order=_SN_ORDER), _two_region_fissile(),
    )
    space = sn_mesh.moment_space(_L)
    assert isinstance(space, TensorProductSpace) and _is_moment_space(space)
    head = space.factors[0]
    assert isinstance(head, MomentHead)
    weights = head.inner_product_weights
    assert weights is not None and head.metric is None
    live = np.asarray(weights) > 0.0
    expected = 4.0 * np.pi / (2.0 * np.arange(_L + 1) + 1.0)
    np.testing.assert_allclose(
        np.asarray(weights)[live],
        np.repeat(expected, [2 * l + 1 for l in range(_L + 1)]),
        rtol=0.0, atol=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════
# Leg 2 — VALUE: the frozen trajectory
# ═══════════════════════════════════════════════════════════════════════


def test_the_windowed_increment_norm_trajectory_reproduces(solved) -> None:
    r"""The frozen ``‖Δφ‖`` trajectory, ``rtol = 1e-9`` (module docstring)."""
    history = solved.history
    assert history is not None
    record = history.record
    assert record.converged and record.iterations_run == len(_INCREMENT_NORMS) + 1
    np.testing.assert_allclose(
        np.asarray(record.increment_norms), np.asarray(_INCREMENT_NORMS),
        rtol=_RTOL, atol=0.0,
        err_msg=(
            "the windowed SI increment-norm trajectory moved. Check WHICH metric "
            "the moment space carries before re-baselining — a head-metric change "
            "moves this by ~92 % and moves NOTHING else (the converged flux and "
            "the residual trajectory are bit-identical either way)."
        ),
    )


def test_the_windowed_contraction_ratio_trajectory_reproduces(solved) -> None:
    r"""The ρ transient, and its settled tail.

    ρ is a RATIO, so it is blind to a UNIFORM rescaling of the norm
    (``lessons`` L58b) — which is exactly why the ratio pin alone is not
    enough and the un-normalised ``increment_norms`` row above is its
    partner.  The head-metric fork is NOT uniform (it is per-ℓ), so ρ does
    move: `[M]` 3.85 % relative.
    """
    history = solved.history
    assert history is not None
    ratios = np.asarray(history.record.contraction_ratios)
    assert len(ratios) == len(_INCREMENT_NORMS) - 1
    np.testing.assert_allclose(
        ratios[:3], np.asarray(_RHO_HEAD), rtol=_RTOL, atol=0.0,
    )
    np.testing.assert_allclose(ratios[-1], _RHO_TAIL, rtol=_RTOL, atol=0.0)


def test_the_stop_rides_the_residual_not_the_increment(solved) -> None:
    r"""The claim that makes this file a DIAGNOSTIC pin rather than a value gate.

    ``iteration.py:795-801`` records the increment for diagnostics and stops on
    the residual; ``convergence.py:1088`` says ρ is *"a DIAGNOSTIC, not a
    verdict — deliberately NOT a StoppingCriterion"*.  Asserted structurally so
    a future change that promotes the increment to a criterion cannot land
    quietly — it would make the moment metric a CONVERGENCE input.
    """
    history = solved.history
    assert history is not None
    record = history.record
    assert [c.name for c in record.criteria] == ["residual"]
    assert record.increment_norms  # recorded, and not among the criteria


# ═══════════════════════════════════════════════════════════════════════
# Leg 3 — DISCRIMINATION: the frozen numbers are metric-dependent
# ═══════════════════════════════════════════════════════════════════════


def test_the_pin_discriminates_the_head_metric_choice() -> None:
    r"""The two candidate head metrics give norms ≈ ``4π`` apart on ONE vector.

    Without this leg the frozen trajectory is compatible with *"the head's
    metric does not reach the recorded norm"*, which is false — and a future
    audit could delete the file as inert.  The separation is computed here, on
    a deterministic vector, so the pin's discriminating power is a measurement
    inside the test rather than a claim in its docstring (``vv`` #19).
    """
    sn_mesh = SNMesh(
        _mesh(), Quadrature.level_symmetric(sn_order=_SN_ORDER), _two_region_fissile(),
    )
    continuum_space = sn_mesh.moment_space(_L)
    frame = HarmonicFrame.from_galerkin(sn_mesh.quad.angular_frame(_L))
    dressed_space = frame.basis_space * sn_mesh.bulk_space
    assert continuum_space.shape == dressed_space.shape
    assert continuum_space == dressed_space, (
        "the two candidates are (name, shape)-equal today — that IS the seam 6.2c removes"
    )

    rng = np.random.default_rng(20260907)
    x = rng.standard_normal(continuum_space.shape)
    # zero the |m| > l padding, as every real moment field has
    assert isinstance(continuum_space, TensorProductSpace)
    head = continuum_space.factors[0]
    assert isinstance(head, MomentHead)
    mask = np.zeros(head.shape, dtype=bool)
    for l in range(_L + 1):
        mask[head.degree_block(l)] = True
    x *= mask.reshape(head.shape + (1,) * (x.ndim - len(head.shape)))

    continuum_norm = continuum_space.norm(x)
    dressed_norm = dressed_space.norm(x)
    assert continuum_norm > 0.0 and dressed_norm > 0.0
    ratio = continuum_norm / dressed_norm
    assert 3.0 < ratio < 13.0, (
        f"the two head metrics must be separated on this pin's own quantity; "
        f"ratio={ratio:.4f} (expected ≈ sqrt(4π·4π/(2ℓ+1)) per ℓ, ~4-12×)"
    )
    assert abs(ratio - 1.0) > 1e3 * _RTOL
