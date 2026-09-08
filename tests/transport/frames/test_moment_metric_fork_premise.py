r"""Pre-carve anchors for CS4c step 6 item 6.2c — the moment-metric FORK.

**The fork, stated once.**  Two moment-head spaces exist in the tree today and
they carry DIFFERENT metrics:

=========================================  =========================  ===================
object                                     head factor                metric on the head
=========================================  =========================  ===================
``SNMesh.moment_space(L, w)`` (6.2b)       ``frame.basis.space``      CONTINUUM ``4π/(2ℓ+1)``
``HarmonicFrame.moment_space_on(space)``   ``frame.basis_space``      PARSEVAL ``1/diag(G)``
                                                                      (or ``G⁺``, DENSE rows)
=========================================  =========================  ===================

They coexist only because ``FunctionSpace.__eq__`` is ``(name, shape)`` on a
name-built space, i.e. metric-BLIND: the analysis face STAMPS the returned
:class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux` with
its dressed codomain (`harmonic_frame.py:200-205`) while the sweep's iterate
wrap and the boundary guard compare against the hub's continuum space, and
``HarmonicFrame._admit`` (`harmonic_frame.py:128`) waves both through.  Item
6.2c makes the head AXIS-BUILT, which puts the measure into the identity — so
the two stop being interchangeable and ONE metric must be chosen.

**What this file pins.**  The three facts a ruling has to be taken against,
each with the OTHER candidate as its negative control, all GREEN today:

1. the two spaces exist, are ``==``, and are metric-different on **33 of 33**
   shipped (rule, L) rows;
2. **Parseval** — ``‖Mψ‖²_{basis_space} = ‖ψ‖²_W`` holds exactly under the
   dressed metric on 33 of 33 and FAILS by a factor **3.41 … 157.91** under the
   continuum one (this is what the continuum choice would cost);
3. **the Λ-adjoint objection does not bite on physical data** — under the
   dressed metric ``Λ.H`` differs from ``Λᵀ`` on 5 of 33 rows for an arbitrary
   head draw and on **0 of 33** for a genuine covariant moment ``φ = Mψ``
   (this is what the dressed choice does NOT cost).

`[M]` all three re-measured 2026-09-07 on `main` @ ``79d2944a``
(``scratch/_step6_2c/p1_fork_ground.py``, ``p8_parseval.py``, ``p9_on_range.py``).

⛔ **Two inherited claims this file REFUTES, so they cannot be re-quoted.**
``tests/transport/frames/test_moment_space_is_read_off_the_frame.py`` records,
in prose, that the dressed end *"would move [Λ's Hilbert adjoint] on 10 of 33
rows (the dense-Gram rows)"* and that the dressed metric *"would move
``apply_metric`` by 96–161 %"*.  Re-measured: the adjoint moves on **5** rows,
**3 of which are DIAGONAL-Gram** (the mechanism is the Parseval metric's
Moore–Penrose PROJECTION of the σ-odd columns a folded rule cannot see, not
Gram density); and **no statistic reproduces 161 %** — the draw-free
per-element movement spans 0.5 %…100.0 % over the 33 rows and 0.5 %…222 % over
60 rows at ``L ≤ 4``.  Those sentences carry no statistic, no fixture and no
denominator, and must be rewritten with one whichever way the fork is ruled.

Foundation mark: software invariants + one closed-form identity (Parseval);
no eigenvalue claim rides here.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.frame import GramStructure
from orpheus.numerics.quadrature import Quadrature
from orpheus.transport.frames import HarmonicFrame
from orpheus.transport.material_field import TransferMaterialField
from orpheus.transport.operators.transfer import LegendreMomentTransfer
from tests.sn._test_helpers import material_xs_from_raw

pytestmark = pytest.mark.foundation

#: The shipped angular rules the moment path binds a frame for.  Built in the
#: test BODY, never in a ``parametrize`` argument list (``vv`` Mode 8, third
#: pipeline class: a production call in the argument list dies at COLLECTION
#: under a mutation and the run reports ``FAILED = 0``).
_RULES = {
    "gauss_legendre(2)": lambda: Quadrature.gauss_legendre(2),
    "gauss_legendre(8)": lambda: Quadrature.gauss_legendre(8),
    "gauss_legendre(16)": lambda: Quadrature.gauss_legendre(16),
    "level_symmetric(4)": lambda: Quadrature.level_symmetric(4),
    "level_symmetric(8)": lambda: Quadrature.level_symmetric(8),
    "lebedev(11)": lambda: Quadrature.lebedev(11),
    "lebedev(17)": lambda: Quadrature.lebedev(17),
    "product(4,6)": lambda: Quadrature.product(4, 6),
    "product(8,8)": lambda: Quadrature.product(8, 8),
    "folded_product(2,4)": lambda: Quadrature.folded_product(2, 4),
    "folded_product(4,8)": lambda: Quadrature.folded_product(4, 8),
}

_ORDERS = [0, 1, 2]

#: `[M]` 2026-09-07, ``p11_dense_and_guard.py``: the (rule, L) rows in this
#: file's grid whose measured trial Gram is DENSE — the two rows where the
#: Parseval metric is a matrix and NO diagonal measure exists.  Hazard H-1
#: for item 6.2c: an axis-built head cannot carry them (the construction guard
#: refuses ``axes`` beside a ``metric`` object).
_DENSE_ROWS = {("gauss_legendre(2)", 2), ("folded_product(2,4)", 2)}

_NX, _NG = 4, 2
_SIGS = [
    np.array([[0.20, 0.00], [0.05, 0.18]]),
    np.array([[0.02, 0.00], [0.01, 0.015]]),
    np.array([[0.004, 0.000], [0.002, 0.003]]),
]


def _transfer(L: int) -> TransferMaterialField:
    """A per-material Legendre stack over a 4-cell 2-group layout."""
    cells = {0: (np.arange(_NX), np.zeros(_NX, dtype=int))}
    return TransferMaterialField.scattering(
        material_xs_from_raw(
            sig_s={0: _SIGS[: L + 1]}, sig2={0: np.zeros((_NG, _NG))},
            cells_by_mat=cells, ng=_NG, nx=_NX, ny=1,
        )
    )


def _adjoint_gap(space, transfer: TransferMaterialField, L: int, x: np.ndarray) -> float:
    r"""``‖Λ.H x − Λᵀ x‖ / ‖Λᵀ x‖`` with both ends bound to ``space``."""
    lam = LegendreMomentTransfer(
        transfer.at_order(L), skip_l0=False, domain=space, codomain=space,
    )
    t = np.asarray(lam.apply_transpose(x))
    h = np.asarray(lam.H.apply(x))
    denominator = float(np.linalg.norm(t))
    assert denominator > 0.0, "the fixture must make Λᵀ non-trivial"
    return float(np.linalg.norm(h - t) / denominator)


# ═══════════════════════════════════════════════════════════════════════
# F1 — the fork's PREMISE: two spaces, ==, metric-different
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("label", sorted(_RULES))
@pytest.mark.parametrize("L", _ORDERS)
def test_the_two_moment_heads_are_equal_and_metric_different(label: str, L: int) -> None:
    r"""⛔ RE-POSED BY 6.2c — the seam the axis-built head removes.

    The continuum head (what the hub and the operator ends bind, #429 tracker
    2.5 Landing A) and the frame's Parseval-dressed head (what the analysis
    face's codomain is) are ``==`` — because ``(name, shape)`` identity is
    metric-blind — and carry DIFFERENT metrics on every shipped row.

    Reddens under: making either producer read the other's space (the fork's
    resolution), or the identity flip reaching the head (item 6.2c itself).
    """
    frame = HarmonicFrame.from_galerkin(_RULES[label]().angular_frame(L))
    continuum, dressed = frame.basis.space, frame.basis_space

    assert continuum == dressed, "today: identity is metric-blind on the head"
    assert continuum.name == dressed.name and continuum.shape == dressed.shape

    assert continuum.inner_product_weights is not None
    assert continuum.metric is None
    if (label, L) in _DENSE_ROWS:
        assert frame.discrete_gram_structure is GramStructure.DENSE
        assert dressed.metric is not None and dressed.inner_product_weights is None
    else:
        assert frame.discrete_gram_structure is GramStructure.DIAGONAL
        assert dressed.metric is None and dressed.inner_product_weights is not None
        assert not np.array_equal(
            np.asarray(continuum.inner_product_weights),
            np.asarray(dressed.inner_product_weights),
        )

    # the movement, on the DRAW-FREE statistic (a per-element ratio, not an
    # L2 residual whose value depends on the draw — ``vv`` #31)
    rng = np.random.default_rng(20260907)
    x = rng.standard_normal(continuum.shape)
    a = np.asarray(continuum.apply_metric(x))
    b = np.asarray(dressed.apply_metric(x))
    live = np.abs(a) > 0.0
    assert live.any()
    movement = np.abs(b[live] - a[live]) / np.abs(a[live])
    assert float(movement.max()) > 1e-3, (
        "the two metrics must be distinguishable, or this file has no subject"
    )


def test_the_dense_gram_inventory_is_the_one_this_file_declares() -> None:
    r"""The ``_DENSE_ROWS`` constant is a MEASUREMENT, re-derived here.

    A hard-coded exclusion list is a claim about the corpus; deriving it in the
    test means a rule whose Gram structure moves reddens this row instead of
    silently taking the wrong branch above (``vv`` #13's finite-roster rule:
    for a shipped finite family, probe every member).
    """
    measured = {
        (label, L)
        for label in _RULES
        for L in _ORDERS
        if HarmonicFrame.from_galerkin(
            _RULES[label]().angular_frame(L)
        ).discrete_gram_structure is GramStructure.DENSE
    }
    assert measured == _DENSE_ROWS, (
        f"the DENSE-Gram inventory moved: {sorted(measured)} != {sorted(_DENSE_ROWS)}"
    )
    assert len(_RULES) * len(_ORDERS) == 33


# ═══════════════════════════════════════════════════════════════════════
# F2 — what the CONTINUUM choice would cost: Parseval
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("label", sorted(_RULES))
@pytest.mark.parametrize("L", _ORDERS)
def test_parseval_holds_under_the_dressed_metric_and_FAILS_under_the_continuum_one(
    label: str, L: int,
) -> None:
    r"""``‖Mψ‖²_{G⁺} = ‖ψ‖²_W`` for band-limited ψ — with the CONTINUUM metric
    as the negative control (``vv`` #19: the positive reading alone cannot
    tell a loaded gate from a blind one).

    The physics: an analysed moment ``φ = M ψ`` is COVARIANT (``φ = G c``), so
    the inner product under which it carries the field's own L²(dΩ) energy is
    ``G⁻¹`` — ``‖ψ‖²_W = cᵀGc = φᵀG⁻¹φ``.  The continuum Gram is the metric of
    the CONTRAVARIANT coefficient space; installed on ``φ`` it computes
    ``cᵀ G G_c G c``, wrong by ``(4π/(2ℓ+1))²`` per ℓ.

    `[M]` 2026-09-07 over the 33 rows: dressed **33 of 33** at 1.0000000000;
    continuum **0 of 33**, ratio spanning **3.41 … 157.91**.

    Reddens under: reverting the F-0/P7 dressing (the frame returning
    ``basis.space`` undressed) — which is exactly the continuum arm of the fork.
    """
    frame = HarmonicFrame.from_galerkin(_RULES[label]().angular_frame(L))
    rng = np.random.default_rng(1234)
    c = rng.standard_normal(frame.basis_space.shape)
    psi = frame.basis.synthesize(c, frame.table)
    phi = frame.analysis.apply(psi)
    energy = frame.measure_space.inner_product(psi, psi)
    assert energy > 0.0

    np.testing.assert_allclose(
        frame.basis_space.inner_product(phi, phi), energy, rtol=1e-12,
        err_msg="the dressed (Parseval) metric must make analysis an isometry",
    )
    # the NEGATIVE control: the same pairing under the continuum Gram
    ratio = float(frame.basis.space.inner_product(phi, phi)) / float(energy)
    assert ratio > 3.0, (
        f"the continuum metric must be measurably NOT the Parseval one; "
        f"ratio={ratio:.4f}"
    )


# ═══════════════════════════════════════════════════════════════════════
# F3 — what the DRESSED choice does NOT cost: the Λ adjoint
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("label", sorted(_RULES))
@pytest.mark.parametrize("L", _ORDERS)
def test_the_lambda_adjoint_is_the_transpose_on_PHYSICAL_moments_under_both_metrics(
    label: str, L: int,
) -> None:
    r"""``Λ.H == Λᵀ`` on a covariant moment ``φ = Mψ``, under EITHER candidate.

    ``Λ`` is ``(ℓ, m)``-diagonal (it mixes only groups), so ``Λ.H = G⁺ΛᵀG``
    collapses to ``Λᵀ`` for any metric diagonal in ``(ℓ, m)``.  The dressed
    metric additionally ZEROES the slots a rule cannot see (a folded rule's
    σ-odd columns, a rank-deficient Gram's null space) — and those slots carry
    identically-zero moments for every field the rule can analyse, so the
    pseudo-inverse's projection is invisible on the range of ``M``.

    `[M]` 2026-09-07: **0 of 33** rows move on a physical moment under either
    metric; **5 of 33** move under the dressed metric on an arbitrary head
    draw (``test_..._only_off_the_range_of_analysis`` pins that separation).
    ⟹ the Λ-adjoint objection to the Parseval metric is a claim about inputs
    production cannot produce.
    """
    frame = HarmonicFrame.from_galerkin(_RULES[label]().angular_frame(L))
    transfer = _transfer(L)
    rng = np.random.default_rng(20260907)
    n_nodes = frame.measure.weights.shape[0]
    psi = rng.standard_normal((n_nodes, _NG, _NX, 1))
    phi = np.asarray(frame.analysis.apply(psi))

    for name, space in (("continuum", frame.basis.space), ("dressed", frame.basis_space)):
        gap = _adjoint_gap(space, transfer, L, phi)
        assert gap < 1e-12, f"{name} end moved Λ.H off Λᵀ on a physical moment: {gap:.3e}"


def test_the_dressed_adjoint_moves_ONLY_off_the_range_of_analysis() -> None:
    r"""The separation that makes the row above informative rather than vacuous.

    On the three folded / rank-deficient rows an ARBITRARY head draw DOES move
    ``Λ.H`` away from ``Λᵀ`` under the dressed metric — so the previous gate's
    green is a statement about the range of ``M``, not about the metric being
    inert.  Without this leg the ``0 of 33`` reading is compatible with "the
    two metrics are indistinguishable", which is false.

    Reddens under: making the dressed metric strictly positive (dropping the
    dead-slot zeros), which would also break the Moore–Penrose contract the
    frame's own ``test_parseval_dressing_installed_on_diagonal_frames`` pins.
    """
    movers = [
        ("folded_product(2,4)", 1),
        ("folded_product(4,8)", 1),
        ("folded_product(4,8)", 2),
        ("folded_product(2,4)", 2),
        ("gauss_legendre(2)", 2),
    ]
    rng = np.random.default_rng(20260907)
    for label, L in movers:
        frame = HarmonicFrame.from_galerkin(_RULES[label]().angular_frame(L))
        transfer = _transfer(L)
        x = rng.standard_normal(frame.basis.space.shape + (_NG, _NX, 1))
        continuum_gap = _adjoint_gap(frame.basis.space, transfer, L, x)
        dressed_gap = _adjoint_gap(frame.basis_space, transfer, L, x)
        assert continuum_gap < 1e-12, (
            f"{label} L={L}: the continuum end must be transpose-exact off-range too"
        )
        assert dressed_gap > 1e-3, (
            f"{label} L={L}: expected an off-range separation, got {dressed_gap:.3e}"
        )
