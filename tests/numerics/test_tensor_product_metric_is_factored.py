r"""Pre-carve anchors for the CS4c step-6 tensor-product metric carve (§7.3 / F2).

**What this file is.** Step 6 item 6.2a (LANDED 2026-09-07) stopped
``FunctionSpace.__mul__`` from DENSIFYING a tensor product's metric into one
``(L+1, 2L+1, ng, nx[, ny])`` weights tensor and has it carry a per-BLOCK
:class:`~orpheus.numerics.metric.FactoredMetric` instead (per axis for an
axis-built factor).  Pre-carve, both arms existed in production (the dense
``_tensor_product_inner_weights`` — the live one — and the P7
``_tensor_product_factored_metric``, which fired 0× on every measured SN path
because no shipped factor carried a metric object), so the equivalence was
measured BEFORE the carve and the carve inherited a measured band instead of
adopting one.  Post-carve the dense arm is gone from production; its oracle is
hand-built here (``_dense_arm``) so the band stays measured against an
INDEPENDENT spelling of the outer product.

**The band is 2 ULP, and it is DRAW-STABLE.**  `[M]` 2026-09-07, 200 seeds ×
8 (geometry × L) rows = 1600 draws, dense arm vs factored arm on the SAME
production factors:

==================  ==========  ==========  ==========  =========
row                 bit-equal   max abs     max rel     max ULP
==================  ==========  ==========  ==========  =========
slab   L=0 / L=1    8/200 1/200 3.55e-15    2.53e-16    **2.0**
sphere L=0 / L=1    4/200 0/200 1.14e-13    3.03e-16    **2.0**
cyl    L=0 / L=1    2/200 0/200 5.68e-14    3.15e-16    **2.0**
cart2d L=0 / L=1    0/200 0/200 3.55e-15    3.46e-16    **2.0**
==================  ==========  ==========  ==========  =========

⛔ ``np.array_equal`` would be a FALSE RED — **0 of 8 rows are bit-equal over a
full seed sweep** (``vv`` #31: "bit-exact" is a property of the DRAW until a
sweep makes it a property of the fixture).  An absolute ``atol`` is
fixture-dependent (3.5e-15 … 1.1e-13, tracking the data magnitude).  The ULP
distance is the draw-stable statistic and it is exactly **2** on every row and
every draw, so the shipped band is ``nulp=4`` — 2× the measured worst.

**Mechanism, so the band is a reason and not a constant.**  The dense arm
forms ``w_head ⊗ w_bulk`` once and multiplies; the factored arm multiplies by
each factor's diagonal in turn.  One extra rounding, reduction depth +1 —
``vv-principles`` §bit-identity criterion 3.

**Activation evidence (mutation battery, 2026-09-07, scope 5550 rows).**

=========================================  =====  ====================================
arm                                        reds   note
=========================================  =====  ====================================
drop one factor's measure                    6    the Euclidean-factor mutation
position the FactoredMetric entries         2    the wrong-block mutation
in REVERSED factor order
install the factored arm globally (shim)     3    the 3 gates the carve must re-key
=========================================  =====  ====================================

The wrong-block arm's 2 reds are ``test_dense_metric.py::…does_not_go_silently
_euclidean`` and ``test_harmonic_frame.py::…carries_the_dense_parseval_metric``
— neither is on this file's claim, so **G2.1's teeth are net-new** for the
per-mint equivalence.
"""

from __future__ import annotations

import numpy as np
import pytest

import orpheus.numerics.space as _spacemod
from orpheus.numerics.metric import DiagonalMetric, FactoredMetric, _broadcast_leading
from orpheus.numerics.space import FunctionSpace, TensorProductSpace
from tests.sn.architecture.test_monomorphic_leaves import (
    _cart2d,
    _cylinder,
    _slab,
    _sphere,
)

pytestmark = pytest.mark.foundation

#: MEASURED worst ULP distance between the two arms over 1600 draws: 2.0.
#: The shipped band is 2× that (never a bare `array_equal` — 0 of 8 rows are
#: bit-equal over a seed sweep).
_ARM_AGREEMENT_NULP = 4

_GEOMETRIES = {"slab": _slab, "sphere": _sphere, "cylinder": _cylinder, "cart2d": _cart2d}
_ORDERS = (0, 1)
_SEEDS = (0, 1, 2, 3, 4)


def _capture_production_factor_tuples(sn_mesh, L: int) -> "list[tuple[FunctionSpace, ...]]":
    """Every factor tuple PRODUCTION hands to ``TensorProductSpace.from_factors``
    while minting the moment family's space on ``sn_mesh`` at order ``L``.

    Read off production rather than re-derived here (``coding-elegance``
    Pattern 2): the hub's mint (``head * mesh.bulk_space``, the cell group
    widened by the scheme's moment axis when asked) and the ``harmonic_frame`` mint
    (``basis_space * of_axes(*cell_axes)``) all funnel through this one
    classmethod, so a transcription of any of them into the test could drift.
    """
    from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux

    captured: list[tuple[FunctionSpace, ...]] = []
    original = TensorProductSpace.from_factors.__func__

    def recording(cls, factors):
        captured.append(tuple(factors))
        return original(cls, factors)

    TensorProductSpace.from_factors = classmethod(recording)   # type: ignore[method-assign]
    try:
        HarmonicMomentFlux.zeros_for_mesh_and_L(
            sn_mesh, L, spatial_moments=sn_mesh.scheme.spatial_basis_per_axis,
        )
        frame = sn_mesh.quad.angular_frame(L)
        # ⚠ `moment_space_on` is HarmonicFrame's mint; a GalerkinFrame (what a
        # 1-D rule returns) does not carry it.  Guarded so the row reports the
        # mints it REACHED rather than dying on a frame-family difference.
        mint = getattr(frame, "moment_space_on", None)
        if mint is not None:
            mint(sn_mesh.angular_trial_space)
    finally:
        TensorProductSpace.from_factors = classmethod(original)  # type: ignore[method-assign]
    return captured


def _dense_arm(factors) -> "np.ndarray | None":
    """The RETIRED dense arm, hand-built: the outer product of every block's
    weights (ones for a counting-measure block), positioned exactly as the
    factored realization positions its entries — the ORACLE the factored
    metric is measured against, independent of production now that
    production no longer forms it (CS4c step 6 item 6.2a, 2026-09-07).
    """
    blocks: "list[np.ndarray | None]" = []
    shapes: "list[tuple[int, ...]]" = []
    for f in factors:
        if f.axes is not None:
            for ax in f.axes:
                blocks.append(
                    None if ax.weights is None else np.asarray(ax.weights, dtype=float)
                )
                shapes.append(tuple(ax.shape))
        else:
            w = f.inner_product_weights
            blocks.append(None if w is None else np.asarray(w, dtype=float))
            shapes.append(tuple(f.shape))
    if all(b is None for b in blocks):
        return None
    result: "np.ndarray | None" = None
    for b, shape in zip(blocks, shapes):
        w = np.ones(shape) if b is None else np.broadcast_to(b, shape)
        result = w if result is None else np.multiply.outer(result, w)
    return result


def _factored_arm(factors):
    return _spacemod._tensor_product_factored_metric(factors)


# ═════════════════════════════════════════════════════════════════════════
# G2.1 — the two arms agree to a DRAW-STABLE 2 ULP on production's factors
# ═════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("geometry", list(_GEOMETRIES), ids=list(_GEOMETRIES))
@pytest.mark.parametrize("L", _ORDERS, ids=[f"L{n}" for n in _ORDERS])
def test_g2_1_dense_and_factored_metric_arms_agree_to_the_measured_band(geometry, L):
    r"""``dense-weights ⊙ x`` ≡ ``FactoredMetric.apply(x)`` on every factor
    tuple production mints, to ``nulp=4``.

    ACTIVATION, asserted in the row: at least one captured tuple must produce
    NON-``None`` dense weights, else the comparison is ``x`` against ``x``
    (``vv`` Mode 8's tautological class — a metric-free product makes both
    arms the identity).

    The row is parametrized over FIVE seeds because a single draw's reading is
    a property of the draw, not of the fixture (``vv`` #31): `[M]` 0 of 8
    (geometry × L) rows are bit-equal over 200 seeds, so a one-seed
    ``array_equal`` row would be green today and red on any fixture edit.
    """
    sn_mesh = _GEOMETRIES[geometry]()
    tuples = _capture_production_factor_tuples(sn_mesh, L)
    if not tuples:
        pytest.fail(
            f"[{geometry} L={L}] production minted NO tensor product — the "
            f"row's subject does not exist on this mesh"
        )

    weighted = [f for f in tuples if _dense_arm(f) is not None]
    if not weighted:
        pytest.fail(
            f"[{geometry} L={L}] every captured factor tuple is Euclidean "
            f"(dense weights are None on all {len(tuples)}) — both arms are "
            f"the identity and the comparison is vacuous"
        )

    for factors in weighted:
        dense = _dense_arm(factors)
        assert dense is not None                       # narrowing; `weighted` filtered
        metric = _factored_arm(factors)
        if metric is None:
            pytest.fail(
                f"[{geometry} L={L}] the factored builder returned None on a "
                f"factor tuple whose hand-built oracle is weighted — the "
                f"measure was DROPPED"
            )
        shape = tuple(int(n) for f in factors for n in f.shape)
        for seed in _SEEDS:
            x = np.random.default_rng(seed).standard_normal(shape)
            via_dense = x * _broadcast_leading(dense, x.ndim)
            via_factored = metric.apply(x)
            np.testing.assert_array_almost_equal_nulp(
                via_dense, via_factored, nulp=_ARM_AGREEMENT_NULP,
            )


def test_g2_2_the_moment_product_carries_no_dense_slot_and_threads_its_axes():
    r"""Item 6.2c-ii LANDED (2026-09-08): on every shipped SN mint the angular
    head is AXIS-BUILT, so ``head * bulk`` takes ``from_factors``' axis arm —
    the product's ``axes`` is the concatenation, ``inner_product_weights`` is
    ``None``, and the metric is DERIVED from the axes (no object at all on a
    DIAGONAL-Gram frame; on a DENSE-Gram frame the head's positioned
    pseudo-inverse rides beside the axes as the product's overlay of forms,
    item 6.2c-i).

    History, kept apart so the suite says which claim moved: until 6.2a
    (then ``test_g2_1b``) the product took the DENSE arm and formed a
    ``(L+1, 2L+1, ng, nx[, ny])`` weights tensor per mint; from 6.2a to
    6.2c-ii (then this row's previous body) it took the FACTORED arm —
    one ``DiagonalMetric`` entry for the axes-less head's dense slot plus
    one per bulk axis — with ``axes is None``.

    Value leg: the product's pairing equals the head's pairing times the
    bulk's, on the Parseval-dressed head every consumer now holds.
    """
    from orpheus.numerics.frame import GramStructure

    findings: list[str] = []
    for geometry, factory in _GEOMETRIES.items():
        sn_mesh = factory()
        bulk_axes = sn_mesh.bulk_space.axes
        assert bulk_axes is not None, "the carrier's bulk space is of_axes-built"
        for L in _ORDERS:
            frame = sn_mesh.quad.angular_frame(L)
            head = frame.basis_space
            product = head * sn_mesh.bulk_space
            findings.append(
                f"{geometry} L={L}: head.axes={head.axes is not None} "
                f"product.axes={product.axes is not None} "
                f"product.ipw={product.inner_product_weights is not None} "
                f"product.metric={type(product.metric).__name__}"
            )
            if head.axes is None or product.axes is None:
                pytest.fail(
                    "the angular head / the moment product is not axis-built — "
                    "item 6.2c-ii regressed.\n  " + "\n  ".join(findings)
                )
            assert len(product.axes) == 1 + len(bulk_axes)
            assert product.axes[0] is head.axes[0]
            if product.inner_product_weights is not None:
                pytest.fail(
                    f"[{geometry} L={L}] the moment product DENSIFIES "
                    f"({product.inner_product_weights.nbytes} B).\n  "
                    + "\n  ".join(findings)
                )
            if frame.discrete_gram_structure is GramStructure.DENSE:
                assert isinstance(product.metric, FactoredMetric)
                forms = [f for _, f in product.metric.entries]
                assert forms[0] is not None and all(f is None for f in forms[1:])
            else:
                assert product.metric is None
            # the value leg: the product pairing factorises over the head and the bulk
            rng = np.random.default_rng(L + 7)
            x = rng.standard_normal(head.shape)
            y = rng.standard_normal(sn_mesh.bulk_space.shape)
            xy = np.multiply.outer(x, y)
            want = float(head.inner_product(x, x)) * float(sn_mesh.bulk_space.inner_product(y, y))
            np.testing.assert_allclose(product.inner_product(xy, xy), want, rtol=1e-12)



# ═════════════════════════════════════════════════════════════════════════
# G2.3 — the SEPARATION the memory assertion is about, sized honestly
# ═════════════════════════════════════════════════════════════════════════

def test_g2_3_dense_versus_per_axis_storage_separates_by_three_orders():
    r"""A dense tensor-product metric costs ``prod(shape)`` doubles; the
    per-axis form costs ``sum(shape)``.

    Sized on a SYNTHETIC pair rather than on a ledger mesh on purpose: `[M]`
    the ledger's 2-D moment product separates by only **8×** (1152 B dense vs
    144 B per-axis), which is too weak to be a keystone.  ``(2000,) ⊗ (2000,)``
    separates by **1000×** in a few milliseconds, so the leg is
    fixture-honest (lessons L59c — never gate "does not allocate" by asking a
    densifier to ``MemoryError``; size it for SEPARATION on reachable
    ``ndarray.nbytes``).

    ⚠ This row does NOT say production avoids the dense form — it says the two
    forms are separable, i.e. that the memory assertion item 6.2 makes is a
    measurable claim.  The assertion itself is a post-carve gate.
    """
    n = 2000
    w = np.linspace(1.0, 2.0, n)
    a = FunctionSpace(name="A", shape=(n,), inner_product_weights=w)
    b = FunctionSpace(name="B", shape=(n,), inner_product_weights=w.copy())

    dense = _dense_arm((a, b))
    assert dense is not None, "CONTROL INVALID: the synthetic factors are Euclidean"
    per_axis_bytes = a.inner_product_weights.nbytes + b.inner_product_weights.nbytes  # type: ignore[union-attr]
    ratio = dense.nbytes / per_axis_bytes
    if dense.nbytes != n * n * 8 or ratio < 100.0:
        pytest.fail(
            f"the dense/per-axis separation is {ratio:.1f}× "
            f"({dense.nbytes} B vs {per_axis_bytes} B) — expected ≥ 100× at "
            f"n={n}; the memory claim item 6.2 makes is not measurable here"
        )
