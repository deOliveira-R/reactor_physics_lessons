r"""Intrinsic-property tests for the frame faces + the discipline-type hierarchy.

The angular :class:`~orpheus.numerics.frame.GalerkinFrame` binds a :class:`Basis` to a
:class:`DiscreteMeasure` and exposes the analysis / reconstruction faces. These tests pin:

* the **adjoint-for-free** — BOTH ``frame.analysis.H`` and ``frame.reconstruction.H``
  fall out of the frame's swapped spaces with no bespoke code (each pinned against an
  INDEPENDENT closed-form einsum: :math:`M^* = S_0 \circ G^{-1} = R/W` for analysis,
  :math:`R^* = W\,M` for reconstruction — the F-0 Parseval metric);
* the **F-0 Parseval metric** (``frame_square_recarve.md``) — the codomain metric is
  the INVERSE discrete trial Gram, so analysis is an isometry onto its image, the
  frame square closes with one scalar, and the slab's non-diagonal Gram is refused
  VISIBLY (the ``TestParseval``-prefixed suite at the end of this module);
* the symmetric **space pairing** (basis → ``basis_space``; measure → ``measure_space``);
* the faces **compose through ``OperatorProduct`` with real spaces** (no ``cast``) —
  the enabler for the Phase-C cast retirement;
* the structural Galerkin invariant :math:`\Pi R = 4\pi I` routed through the frame.

The full SH-space law suite (:math:`\Pi R = 4\pi I`, :math:`\Pi^* = g_C S_0`,
:math:`R = (2\ell+1) S_0`) lives in ``test_spherical_harmonic_space.py``,
constructed on the same frame faces.
"""

from __future__ import annotations

import numpy as np
import pytest

from orpheus.numerics.basis import (
    GramStructure,
    IndicatorBasis,
    OverlapBasis,
    SphericalHarmonicBasis,
    WeightedIndicatorBasis,
)
from dataclasses import replace

from scipy.linalg import eigh as scipy_eigh

from orpheus.numerics.basis.legendre_basis import LegendreBasis
# NOT re-exported from ``orpheus.numerics.basis`` (its ``__all__`` carries
# Basis/Descent/GramStructure/Indicator/Legendre/Overlap/SH/Truncated/
# WeightedIndicator) — imported from its module, as every other consumer does.
from orpheus.numerics.basis.spherical_harmonic_basis import (
    MirrorEvenSphericalHarmonicBasis,
)
from orpheus.numerics.manifold import COSINE_INTERVAL, SPHERE, RealSpace
from orpheus.numerics.symmetry import SubgroupOfO3
from orpheus.numerics.frame import FrameBase, GalerkinFrame, PetrovGalerkinFrame
from orpheus.numerics.measure import DiscreteMeasure
from orpheus.numerics.metric import _DENSE_METRIC_RCOND, DenseMetric, FactoredMetric
from orpheus.numerics.operator import NotInvertible
from orpheus.numerics.quadrature import Quadrature, lebedev_sphere
from orpheus.numerics.spaces import SphericalHarmonicSpace
from tests._harness.predicates import (
    STRUCTURAL_ABSENT,
    assert_inverse_adjoint_contract,
)


@pytest.fixture
def sh_frame():
    """An exact spherical-harmonic frame: Lebedev(13) ⋈ SH(L=3)."""
    measure = lebedev_sphere(13)
    L = 3
    basis = SphericalHarmonicBasis(L=L)
    return GalerkinFrame(basis, measure), L


@pytest.mark.foundation
def test_frame_faces_two_axis_contract(sh_frame):
    r"""The carve keystone (#226, P4 keystone v2) for the numerics frame faces.

    Both faces carry a working ``apply_transpose`` — so ``.H`` falls out of
    the metric-aware ``AdjointOperator`` — hence ``is_adjointable`` is True
    (and the eager ``.H`` returns the wrapper); neither face is invertible,
    STRUCTURALLY (a projection face declares no ``inverse()`` — misuse is a
    static error, Design C). The reconstruction face's ``is_adjointable`` is
    the OVERRIDE that lifts it above the bare
    :class:`~orpheus.numerics.projection.ReconstructionOperator` default
    (``is_adjointable`` False); the analysis face inherits ``True`` from
    :class:`~orpheus.numerics.projection.AnalysisOperator`.
    """
    frame, _ = sh_frame
    for face in (frame.analysis, frame.reconstruction):
        assert_inverse_adjoint_contract(
            face,
            invertible=False,
            adjointable=True,
            inverse_contract=STRUCTURAL_ABSENT,
        )


def _band_limited(rng, L, *trailing):
    """A random ``(L+1, 2L+1, *trailing)`` moment array with |m|>ℓ slots zeroed."""
    c = rng.standard_normal((L + 1, 2 * L + 1, *trailing))
    for ell in range(L + 1):
        c[ell, 2 * ell + 1 :] = 0.0
    return c


# ── the adjoint-for-free ──────────────────────────────────────────────────

@pytest.mark.foundation
def test_analysis_hilbert_adjoint_falls_out_of_the_frame_spaces(sh_frame):
    r"""``frame.analysis.H`` is the PHYSICAL Hilbert adjoint :math:`M^* = S_0 \circ G^{-1}`.

    No bespoke adjoint code — the frame's swapped ``(measure_space, basis_space)``
    metrics feed the generic ``AdjointOperator``, and the F-0 Parseval dressing
    (the codomain metric is the INVERSE discrete Gram — see
    :attr:`FrameBase.basis_space`) makes the sandwich the physical adjoint. Pinned
    against an INDEPENDENT reference: the direct :math:`S_0(G^{-1}c)` einsum with
    the closed-form inverse SH Gram :math:`(2\ell+1)/4\pi` (exact for the
    degree-exact Lebedev rule; NOT the frame's own contraction). Equivalently
    :math:`M^* = R/W` — the closure pinned family-wide by
    ``test_parseval_frame_square_closes`` below.

    ⛔ Pre-F-0 this gate pinned :math:`g_C \cdot S_0` with the CONTINUUM Gram
    :math:`g_C = 4\pi/(2\ell+1)` as the codomain metric — the WRONG side for
    the carried covariant moments (`[M]` Parseval ratio 118.7 vs 1.000;
    ``scratch/probe_f1_parseval.py``, 2026-08-24). The loaded-not-blind negative
    leg is ``test_parseval_reds_under_the_pre_repair_continuum_metric``.
    """
    frame, L = sh_frame
    rng = np.random.default_rng(14)
    c = _band_limited(rng, L, 4, 2)
    g_inv = (2.0 * np.arange(L + 1) + 1.0) / (4.0 * np.pi)  # closed-form G⁻¹ diag
    expected = np.einsum("nlm,l,lm...->n...", frame.table, g_inv, c)
    np.testing.assert_allclose(
        frame.analysis.H.apply(c), expected, rtol=1e-12, atol=1e-14,
    )


@pytest.mark.foundation
def test_reconstruction_hilbert_adjoint_falls_out_of_the_frame_spaces(sh_frame):
    r"""``frame.reconstruction.H`` is the PHYSICAL Hilbert adjoint :math:`R^* = W\,M`.

    Symmetric with the analysis face: the F-0-dressed domain metric
    (:math:`G^{-1}`) enters the sandwich through its pseudo-inverse :math:`G`,
    giving :math:`(R^* v)_\ell^m = d_\ell\,G_\ell \sum_n w_n
    Y_\ell^m(\hat\Omega_n)\, v_n` — and the SH identity :math:`d_\ell G_\ell =
    (2\ell+1)\cdot 4\pi/(2\ell+1) = 4\pi = W` collapses the per-:math:`\ell`
    factor to the ONE scalar :math:`W`. Pinned against that INDEPENDENT
    closed-form einsum (NOT the frame's own contraction).
    ``R : \text{basis} \to \text{measure}``, so ``R.H`` maps nodal values →
    coefficients. (Pre-F-0 this pinned :math:`(2\ell+1)^2/4\pi\cdot Y^{\mathsf T}W`
    — the continuum-metric sandwich.)
    """
    frame, L = sh_frame
    rng = np.random.default_rng(17)
    n = frame.measure.weights.shape[0]
    v = rng.standard_normal((n, 4, 2))
    expected = 4.0 * np.pi * np.einsum(
        "n,nlm,n...->lm...", frame.measure.weights, frame.table, v,
    )
    np.testing.assert_allclose(
        frame.reconstruction.H.apply(v), expected, rtol=1e-12, atol=1e-14,
    )


# ── the space pairing ─────────────────────────────────────────────────────

@pytest.mark.foundation
def test_basis_space_is_the_spherical_harmonic_space(sh_frame):
    frame, L = sh_frame
    # the frame's codomain is the SH space of order L — the basis's own space
    # re-DRESSED with the Parseval measure (its head axis re-weighted, the
    # frame its generator), so since the identity flip (CS4c step 6, item
    # 6.2c-ii) it is a DIFFERENT space from the continuum mint, same family,
    # same order, same name; the Frame caches it and shares it across the faces.
    assert isinstance(frame.basis_space, SphericalHarmonicSpace)
    assert frame.basis_space.L == L and frame.basis_space.shape == (L + 1, 2 * L + 1)
    assert frame.basis_space.name == frame.basis.space.name == SphericalHarmonicSpace.from_L(L).name
    assert frame.basis_space != frame.basis.space
    assert frame.basis_space != SphericalHarmonicSpace.from_L(L)
    assert frame.basis.space == SphericalHarmonicSpace.from_L(L)
    assert frame.basis_space.axes is not None and frame.basis_space.axes[0].generator is frame
    # the analysis codomain / reconstruction domain are that same space
    assert frame.analysis.codomain is frame.basis_space
    assert frame.reconstruction.domain is frame.basis_space


@pytest.mark.foundation
def test_measure_space_carries_the_quadrature_weights_as_its_metric(sh_frame):
    frame, _ = sh_frame
    ms = frame.measure_space
    assert ms.shape == (frame.measure.weights.shape[0],)
    np.testing.assert_array_equal(ms.inner_product_weights, frame.measure.weights)
    # analysis domain / reconstruction codomain are that same space
    assert frame.analysis.domain is ms
    assert frame.reconstruction.codomain is ms


# ── composition through OperatorProduct (the cast-retirement enabler) ─────

@pytest.mark.foundation
def test_faces_compose_through_operator_product_with_real_spaces(sh_frame):
    """``reconstruction @ analysis`` builds an ``OperatorProduct`` (no ``cast``).

    Both faces carry real ``domain``/``codomain``, so the composition's space-
    compatibility check is live (not skipped on ``None``) and passes — which is what
    let Phase C drop the ``cast(LinearOperator, …)`` workarounds in the scattering
    kernel (now retired).
    """
    frame, _ = sh_frame
    rng = np.random.default_rng(15)
    psi = rng.standard_normal((frame.measure.weights.shape[0], 4, 2))
    product = frame.reconstruction @ frame.analysis
    expected = frame.reconstruction.apply(frame.analysis.apply(psi))
    np.testing.assert_array_equal(product.apply(psi), expected)


@pytest.mark.foundation
def test_pi_R_is_4pi_identity_through_the_frame(sh_frame):
    """``analysis ∘ reconstruction = 4π·I`` on band-limited coefficients (via the frame).

    Inherited from the structural ``test_spherical_harmonic_space`` invariant by the
    faces' bit-identity; pinned here on the frame-routed path.
    """
    frame, L = sh_frame
    rng = np.random.default_rng(16)
    c = _band_limited(rng, L)
    out = frame.analysis.apply(frame.reconstruction.apply(c))
    np.testing.assert_allclose(out, 4.0 * np.pi * c, rtol=1e-10, atol=1e-12)


# ── caching + capability surface ──────────────────────────────────────────

@pytest.mark.foundation
def test_table_and_faces_are_cached(sh_frame):
    frame, _ = sh_frame
    assert frame.table is frame.table
    assert frame.analysis is frame.analysis
    assert frame.reconstruction is frame.reconstruction


# (The former ``test_face_capabilities`` API-smoke — caps == {apply,
# apply_transpose} on both faces — was retired with the frozenset at carve
# P4; its surviving claim, both faces adjointable-not-invertible, is pinned
# in FULL by ``test_frame_faces_two_axis_contract`` above.)


# ── the discipline-type hierarchy (P1) ─────────────────────────────────────

@pytest.mark.foundation
def test_galerkin_is_a_petrov_galerkin_is_a_frame_base(sh_frame):
    """``GalerkinFrame ⊂ PetrovGalerkinFrame ⊂ FrameBase`` — discipline is the TYPE.

    Liskov-correct: a Galerkin frame IS-A Petrov-Galerkin frame (with ``test is
    trial``), strengthening — never weakening — the base promises.
    """
    frame, _ = sh_frame
    assert isinstance(frame, GalerkinFrame)
    assert isinstance(frame, PetrovGalerkinFrame)
    assert isinstance(frame, FrameBase)


@pytest.mark.foundation
def test_galerkin_frame_test_is_the_trial_basis(sh_frame):
    """The Galerkin specialisation fixes ``test = trial`` and reuses the trial caches.

    Reusing :attr:`table`/:attr:`basis_space` (not re-evaluating) is what keeps the
    Galerkin analysis 0-ULP-identical to the single-discipline frame this hierarchy
    replaced, and preserves the analysis-codomain ``is`` identity.
    """
    frame, _ = sh_frame
    assert frame.test is frame.basis
    assert frame.test_table is frame.table
    assert frame.test_space is frame.basis_space
    assert frame.analysis.codomain is frame.basis_space


@pytest.mark.foundation
def test_galerkin_frame_takes_no_test_basis():
    """``GalerkinFrame`` binds test=trial; its constructor takes no ``test_basis``.

    The ``test ≠ trial`` freedom is exactly what a :class:`PetrovGalerkinFrame` is for,
    so a distinct test basis is forbidden on a :class:`GalerkinFrame` by the constructor
    SIGNATURE itself (a ``TypeError`` on the extra argument), not a runtime guard.
    """
    measure = lebedev_sphere(13)
    # ``*args`` so the arity violation is a runtime TypeError, not a static type error.
    args = [SphericalHarmonicBasis(L=3), measure, SphericalHarmonicBasis(L=2)]
    with pytest.raises(TypeError):
        GalerkinFrame(*args)


@pytest.mark.foundation
def test_petrov_galerkin_degenerate_equals_galerkin_bit_identically(sh_frame):
    """A ``PetrovGalerkinFrame`` with ``test_basis = trial`` is the Galerkin degenerate.

    Passing the trial basis itself as the test basis resolves test→trial, so the GENERAL
    Petrov-Galerkin analysis (which reads the TEST table) must reduce BIT-IDENTICALLY to
    the Galerkin analysis. This pins the PG analysis machinery in the degenerate case
    here; the genuine ``test ≠ trial`` instance (flux-weighted homogenisation) lands
    with its consumer in a later phase.
    """
    galerkin, L = sh_frame
    pg = PetrovGalerkinFrame(galerkin.basis, galerkin.measure, galerkin.basis)
    assert pg.test is pg.basis
    rng = np.random.default_rng(24)
    psi = rng.standard_normal((galerkin.measure.weights.shape[0], 4, 2))
    np.testing.assert_array_equal(
        pg.analysis.apply(psi), galerkin.analysis.apply(psi),
    )
    c = _band_limited(rng, L)
    np.testing.assert_array_equal(
        pg.reconstruction.apply(c), galerkin.reconstruction.apply(c),
    )


# ── the project verb: G⁻¹ M (homogenise / condense) — P3 ────────────────────

def _indicator_frame(edges, centres, weights, test_weight=None):
    """A small hand-built indicator frame on an explicit measure.

    ``test_weight=None`` → :class:`GalerkinFrame` (test=trial=plain indicator);
    an array → :class:`PetrovGalerkinFrame` with a flux-weighted indicator test.
    """
    trial = IndicatorBasis((np.asarray(edges, dtype=float),), RealSpace(1))
    measure = DiscreteMeasure(
        nodes=np.asarray(centres, dtype=float),
        weights=np.asarray(weights, dtype=float),
        support=RealSpace(1),
    )
    if test_weight is None:
        return GalerkinFrame(trial, measure)
    return PetrovGalerkinFrame(
        trial, measure, WeightedIndicatorBasis(trial, np.asarray(test_weight, float)),
    )


@pytest.mark.foundation
def test_project_is_gram_inverse_times_analysis():
    r"""``frame.project(f) = G⁻¹ M f`` on a hand-known diagonal-Gram frame.

    Galerkin indicator frame (test=trial), 4 fine nodes / non-uniform volumes, 3
    coarse cells the LAST of which is EMPTY (no fine node). The diagonal Gram is the
    region volume :math:`m_R = \sum_{i\in R} V_i`; ``project`` is the volume-weighted
    average :math:`(\sum_R V_i f_i)/m_R`. The empty region (``m_R = 0``) must project
    to ``0`` (the Moore–Penrose pseudo-inverse), NOT ``nan``/``inf`` — the verb-level
    pin of the zero-flux-region law.
    """
    centres = [0.5, 1.5, 2.5, 3.5]
    V = [1.0, 1.0, 2.0, 1.0]
    f = np.array([10.0, 20.0, 30.0, 40.0])
    frame = _indicator_frame([0.0, 2.0, 4.0, 5.0], centres, V)  # R2=[4,5] empty
    out = frame.project(f)
    expected = np.array([
        (1.0 * 10 + 1.0 * 20) / (1.0 + 1.0),    # R0: nodes 0,1
        (2.0 * 30 + 1.0 * 40) / (2.0 + 1.0),    # R1: nodes 2,3
        0.0,                                     # R2: empty → 0 (pseudo-inverse)
    ])
    np.testing.assert_allclose(out, expected, rtol=1e-14)
    assert np.isfinite(out).all(), "empty region produced nan/inf, not 0"


@pytest.mark.foundation
def test_petrov_galerkin_project_is_cross_gram_extraction():
    r"""``PetrovGalerkinFrame.project`` extracts coefficients against the CROSS Gram.

    A genuine ``test ≠ trial`` frame (test = a flux-weighted indicator ``w·1_R``,
    trial = ``1_R``): the diagonal cross Gram is :math:`G_R = \langle\chi_R,
    \mathbf 1_R\rangle_W = \sum_{i\in R} w_i V_i` and :math:`(M f)_R = \sum_{i\in R}
    w_i V_i f_i`, so ``project = M f / G``. Pinned against the independent hand
    arithmetic (NOT a re-call of the production einsum).
    """
    centres = [0.5, 1.5, 2.5, 3.5]
    V = np.array([0.5, 1.5, 1.0, 1.0])
    w = np.array([2.0, 3.0, 5.0, 7.0])
    f = np.array([10.0, 20.0, 30.0, 40.0])
    frame = _indicator_frame([0.0, 2.0, 4.0], centres, V, test_weight=w)  # 2 cells
    regions = [[0, 1], [2, 3]]
    expected = np.array([
        sum(w[i] * V[i] * f[i] for i in sel) / sum(w[i] * V[i] for i in sel)
        for sel in regions
    ])
    np.testing.assert_allclose(frame.project(f), expected, rtol=1e-13)


@pytest.mark.foundation
def test_petrov_galerkin_degenerate_project_equals_galerkin_project(sh_frame):
    r"""``PetrovGalerkinFrame(b, m, test=b).project ≡ GalerkinFrame(b, m).project``.

    The ``project``-verb analogue of the face-level degenerate test: when ``test is
    trial`` the general PG ``project`` (which reads the TEST Gram) must reduce to the
    SAME numpy chain BIT-IDENTICALLY (``array_equal``, the 0-ULP discipline).
    """
    galerkin, _ = sh_frame
    pg = PetrovGalerkinFrame(galerkin.basis, galerkin.measure, galerkin.basis)
    rng = np.random.default_rng(31)
    psi = rng.standard_normal((galerkin.measure.weights.shape[0], 4, 2))
    np.testing.assert_array_equal(pg.project(psi), galerkin.project(psi))


@pytest.mark.foundation
def test_petrov_galerkin_project_differs_from_galerkin_when_test_neq_trial():
    r"""The PG type is LOAD-BEARING: ``test ≠ trial`` gives a DIFFERENT answer.

    The same geometry projected (a) flux-weighted (the PG test ``w·1_R``) and (b)
    plain Galerkin (test=trial=``1_R``, the volume average) gives materially distinct
    coefficients — the type carries real information, it is not ceremony. Both match
    their respective independent hand references; the discrimination is asserted to
    have actually fired (no silent same-answer pass).
    """
    centres = [0.5, 1.5, 2.5, 3.5]
    V = np.array([0.5, 1.5, 1.0, 1.0])
    w = np.array([2.0, 3.0, 5.0, 7.0])
    f = np.array([10.0, 20.0, 30.0, 40.0])
    edges = [0.0, 2.0, 4.0]
    regions = [[0, 1], [2, 3]]

    pg = _indicator_frame(edges, centres, V, test_weight=w).project(f)
    galerkin = _indicator_frame(edges, centres, V).project(f)

    pg_ref = np.array([
        sum(w[i] * V[i] * f[i] for i in sel) / sum(w[i] * V[i] for i in sel)
        for sel in regions
    ])
    gal_ref = np.array([
        sum(V[i] * f[i] for i in sel) / sum(V[i] for i in sel) for sel in regions
    ])
    np.testing.assert_allclose(pg, pg_ref, rtol=1e-13)
    np.testing.assert_allclose(galerkin, gal_ref, rtol=1e-13)
    assert not np.allclose(pg, galerkin, rtol=1e-6), (
        "PG (flux-weighted) and Galerkin (volume) projections coincided — the test "
        "weight does not discriminate; the PG type would be ceremony here"
    )


# ── Gram-structure: the projection-validity declaration (P5.5a) ───────────


@pytest.mark.foundation
def test_basis_gram_structure_declarations():
    r"""Each trial basis declares the Gram structure that makes ``project`` valid.

    Intrinsic-property pin: ``IndicatorBasis`` (disjoint cells) and
    ``SphericalHarmonicBasis`` (orthogonal) are DIAGONAL; ``OverlapBasis`` (fractional
    rows summing to 1) is PARTITION_OF_UNITY — it MUST override the DIAGONAL it inherits
    from ``IndicatorBasis`` (a straddling row shares ≥2 columns ⟹ non-diagonal Gram).
    The base ``Basis`` default (here via the test-only ``WeightedIndicatorBasis``) is
    DENSE — the safe refusal a new basis inherits until it consciously declares.
    """
    ib = IndicatorBasis((np.array([0.0, 1.0, 2.0]),), RealSpace(1))
    ob = OverlapBasis(
        edges_per_axis=(np.array([-0.5, 0.5, 1.5]),),
        partition_of=RealSpace(1),
        overlap_table=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        fine=RealSpace(1),   # the three fine rows live on the same line the two coarse cells partition
    )
    assert ib.gram_structure is GramStructure.DIAGONAL
    assert SphericalHarmonicBasis(L=1).gram_structure is GramStructure.DIAGONAL
    assert ob.gram_structure is GramStructure.PARTITION_OF_UNITY
    # OverlapBasis IS-A IndicatorBasis but MUST NOT inherit its DIAGONAL claim.
    assert ob.gram_structure is not ib.gram_structure
    # The base default (a test-only basis never used as a trial) is the safe DENSE.
    assert WeightedIndicatorBasis(ib, np.ones(2)).gram_structure is GramStructure.DENSE


@pytest.mark.foundation
def test_project_refuses_dense_gram_trial():
    r"""``project`` / ``.gram_inverse`` REFUSE a DENSE-Gram trial — illegal state unrepresentable.

    The row-sum probe is wrong for a trial that is neither disjoint nor a partition of
    unity; rather than return a silently-wrong coarsening, the frame raises
    :class:`NotInvertible` (the dense ``(MR)⁻¹M`` solve is unbuilt — #275). Mutation
    gate: a trial declaring ``GramStructure.DENSE`` reddens BOTH ``.gram_inverse`` and
    ``.project``, while the otherwise-identical DIAGONAL trial succeeds — proving the
    refusal keys on the declaration, not on some unrelated failure.
    """

    class _DenseTrial(IndicatorBasis):
        @property
        def gram_structure(self) -> GramStructure:
            return GramStructure.DENSE

    edges = np.array([0.0, 1.0, 2.0])
    measure = DiscreteMeasure(
        nodes=np.array([0.5, 1.5]), weights=np.ones(2), support=RealSpace(1),
    )
    dense = PetrovGalerkinFrame(
        _DenseTrial((edges,), RealSpace(1)), measure,
        WeightedIndicatorBasis(IndicatorBasis((edges,), RealSpace(1)), np.ones(2)),
    )
    with pytest.raises(NotInvertible, match="DENSE"):
        _ = dense.gram_inverse
    with pytest.raises(NotInvertible, match="DENSE"):
        dense.project(np.array([3.0, 5.0]))
    # Control: the SAME geometry with the honest DIAGONAL trial projects fine.
    ok = PetrovGalerkinFrame(
        IndicatorBasis((edges,), RealSpace(1)), measure,
        WeightedIndicatorBasis(IndicatorBasis((edges,), RealSpace(1)), np.ones(2)),
    )
    np.testing.assert_allclose(ok.project(np.array([3.0, 5.0])), [3.0, 5.0])


# ── the F-0 Parseval metric — the metric truth (frame_square_recarve F-0) ──
#
# THE THEOREM (exact, unconditional — algebra, not quadrature exactness): for a
# band-limited field ψ = S₀c the analysis output is φ = Mψ = Gc IDENTICALLY,
# with G the discrete TRIAL Gram of the pairing (basis ⊗ measure). So the
# codomain inner product under which analysis is an isometry onto its image is
# the INVERSE discrete Gram:
#
#     ‖φ‖²_{G⁻¹} = cᵀG c = ‖ψ‖²_W                     (Parseval)
#
# and with that metric both faces' .H are the PHYSICAL Hilbert adjoints. Two
# consequences, split by precondition:
#
#   * Parseval needs only a DIAGONAL discrete Gram — any values (LS4 at L=2 is
#     dressed with its true discrete inverse and closes exactly, whatever its
#     relation to the continuum Gram);
#   * the SH closure M* = R/W, R* = W·M additionally needs the per-ℓ identity
#     d_ℓ·G_ℓ = W (degree-exactness; [M] 2026-08-24 every shipped sphere
#     family measures exact to ~1e-15, incl. LS4/LS8 at L=2).
#
# [M] scratch/probe_f1_parseval.py + probe_f1_parseval_slab.py (2026-08-24):
# the pre-F-0 stored metric (continuum 4π/(2ℓ+1)) was the WRONG side — ratio
# 118.7 vs 1.000 on that probe's seed (the ratio is a moment-energy-weighted
# average of the per-ℓ factors, so it is draw-dependent; the draw-independent
# statement is the (4π/(2ℓ+1))² per-ℓ factor); closure ≤1e-15 once dressed;
# the slab GL live Gram has off-diagonals at 0.93 of the Cauchy–Schwarz
# scale, so NO diagonal Parseval metric exists there (the DENSE arm below).

def _overlap_frame() -> GalerkinFrame:
    """The PoU overlap frame — measured-DENSE with an INVERTIBLE Gram
    ([M] ``[[1.25, .25], [.25, 1.25]]``, cond 1.50): the non-angular
    member of the DENSE population."""
    ob = OverlapBasis(
        edges_per_axis=(np.array([-0.5, 0.5, 1.5]),),
        partition_of=RealSpace(1),
        overlap_table=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        fine=RealSpace(1),   # the three fine rows live on the same line the two coarse cells partition
    )
    measure = DiscreteMeasure(
        nodes=np.array([0.0, 0.5, 1.0]), weights=np.ones(3), support=RealSpace(1),
    )
    return GalerkinFrame(ob, measure)


#: The six sphere families whose discrete Gram measures DIAGONAL — the
#: population of the DIAGONAL-arm dressing gate and of the scalar
#: frame-square collapse (a sphere-family property; see D3).
_DIAGONAL_FRAME_CASES = [
    pytest.param(lambda: Quadrature.level_symmetric(4).angular_frame(1), id="LS4-L1"),
    pytest.param(lambda: Quadrature.level_symmetric(4).angular_frame(2), id="LS4-L2"),
    pytest.param(lambda: Quadrature.level_symmetric(8).angular_frame(2), id="LS8-L2"),
    pytest.param(lambda: Quadrature.product(8, 8).angular_frame(2), id="product8x8-L2"),
    pytest.param(
        lambda: Quadrature.folded_product(8, 8).angular_frame(2), id="folded8x8-L2",
    ),
    pytest.param(
        lambda: GalerkinFrame(SphericalHarmonicBasis(L=2), lebedev_sphere(13)),
        id="lebedev13-L2",
    ),
]

#: Every Parseval-capable frame — the diagonal six PLUS the slab, whose
#: dressing is the matrix pseudo-inverse (P7). Until P7 the slab param
#: carried a skip mark (*"NON-DIAGONAL discrete Gram … no diagonal
#: Parseval metric exists; the matrix-metric home is the CS4c Riesz-leg
#: machinery"* — [M] 2026-08-23): the isometry now RUNS there through
#: the DenseMetric dressing, and the mark retired with the refusal.
_PARSEVAL_FRAME_CASES = _DIAGONAL_FRAME_CASES + [
    pytest.param(
        lambda: Quadrature.gauss_legendre(8).angular_frame(2), id="slab-GL8-L2",
    ),
]

#: The measured-DENSE population (D1) — four MECHANISMS, never one
#: family (vv-principles #13): a slab measure, a coarse product at L=2,
#: a coarse level-symmetric at L=3, and a non-angular partition-of-unity
#: basis.
#: ⛔ RE-KEYED 2026-09-02 (#429). The ``slab-GL8-L2`` param DIED with the
#: ERR-080 repair: a 1-D rule now binds the LEGENDRE basis on its own orbit
#: space, and `[M]` that Gram is **DIAGONAL** — offdiag ``8.808e-17``, diag
#: ``[2, 2/3, 0.4] = 2/(2l+1)`` exactly. Its replacements are chosen so the
#: mechanism COUNT rises rather than falls (six, up from four):
#:
#: * ``folded_product(2,4)-L3`` — a σ-fold quotient basis, `[M]` live-block
#:   relative off-diagonal **1.000**, untouched by #429;
#: * ``equispaced(8)-L3`` — the LEGENDRE family's only dense-AND-full-rank
#:   witness. ⭐ A theorem forces this: `[M]` 12 of 12 rows, a ``GL_n`` rule's
#:   Legendre Gram is diagonal-and-exact for ``L <= n-1`` and has a
#:   structurally DEAD slot at ``l = n`` (the nodes ARE ``P_n``'s roots), so
#:   **no 1-D Gauss frame can be dense and full rank** — the dense Legendre
#:   arm has to come from a non-Gauss 1-D measure;
#: * ``LS4-L4-Legendre`` — the Legendre basis on a FULL-SPHERE rule (the G0
#:   row-4 pairing), so the dense arm is exercised on both charts of the
#:   orbit space, not only the realization's.
_DENSE_FRAME_CASES = [
    pytest.param(lambda: Quadrature.product(4, 4).angular_frame(2), id="product4x4-L2"),
    pytest.param(lambda: Quadrature.level_symmetric(4).angular_frame(3), id="LS4-L3"),
    pytest.param(
        lambda: Quadrature.folded_product(2, 4).angular_frame(3),
        id="folded2x4-L3",
    ),
    pytest.param(lambda: _equispaced_legendre_frame(3), id="equispaced8-L3"),
    pytest.param(
        lambda: GalerkinFrame(
            LegendreBasis(L=4), Quadrature.level_symmetric(4).measure
        ),
        id="LS4-L4-Legendre",
    ),
    pytest.param(_overlap_frame, id="overlap-R1"),
]


def _equispaced_legendre_frame(L: int, n: int = 8):
    r"""An equispaced-equal-weight 1-D rule, declared on the orbit space it lives on.

    ⭐ The ONLY dense-and-full-rank Legendre witness that exists, by the dead-slot
    theorem above (`[M]` ``n = 8, L = 3``: offdiag ``2.222e-01``, **0 dead slots**,
    rank 4/4; ``L = 4``: rank 5/5). Its support is
    ``SPHERE.quotient(SO2("x"))`` — the honest declaration a 1-D angular rule
    makes — which is also what lets G0 admit it; a twin declaring the raw
    ``COSINE_INTERVAL`` is G0's constructed refusal witness and lives in
    ``tests/transport/frames/test_binding_tightness.py``.
    """
    mu = np.linspace(-1.0, 1.0, n + 2)[1:-1]
    weights = np.full(mu.size, 2.0 / mu.size)
    return Quadrature(
        measure=DiscreteMeasure(
            nodes=mu,
            weights=weights,
            support=SPHERE.quotient(SubgroupOfO3.O2("x")),
        )
    ).angular_frame(L)


@pytest.mark.foundation
@pytest.mark.parametrize("make_frame", _DIAGONAL_FRAME_CASES)
def test_parseval_dressing_installed_on_diagonal_frames(make_frame):
    r"""The verdict is DIAGONAL and ``basis_space`` carries the inverse discrete Gram.

    The dressed metric equals ``1/G_kk`` on live slots and EXACTLY ``0.0`` on
    dead ones (layout padding; the folded frame's σ-odd columns) — the dead-slot
    zeros are what make the Moore–Penrose inverse-metric path exact. The
    Galerkin override keeps the analysis codomain the SAME dressed object.
    The DENSE population's sibling is
    ``test_dense_frames_are_dressed_with_the_pseudo_inverse_gram`` (D1).
    """
    frame = make_frame()
    assert frame.discrete_gram_structure is GramStructure.DIAGONAL
    diag = np.diagonal(frame.discrete_gram).reshape(frame.basis.space.shape)
    live = diag > 0.0
    metric = _dressed_measure(frame)
    assert metric is not None
    np.testing.assert_allclose(metric[live], 1.0 / diag[live], rtol=1e-15)
    np.testing.assert_array_equal(metric[~live], 0.0)
    assert frame.test_space is frame.basis_space


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
@pytest.mark.parametrize("make_frame", _PARSEVAL_FRAME_CASES)
def test_parseval_analysis_is_an_isometry_onto_its_image(make_frame):
    r"""``‖Mψ‖_{basis_space} = ‖ψ‖_W`` for band-limited ψ — Parseval, rtol 1e-12.

    The coefficient draw is deliberately UNMASKED (garbage in dead slots): a
    dead table column annihilates its coefficient in ψ = S₀c AND zeroes both
    its moment and its metric slot, so the identity must hold regardless —
    pinning the Moore–Penrose dead-slot handling for free.
    """
    frame = make_frame()
    rng = np.random.default_rng(1234)
    c = rng.standard_normal(frame.basis_space.shape)
    psi = frame.basis.synthesize(c, frame.table)
    phi = frame.analysis.apply(psi)
    np.testing.assert_allclose(
        frame.basis_space.inner_product(phi, phi),
        frame.measure_space.inner_product(psi, psi),
        rtol=1e-12,
    )


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
@pytest.mark.verifies("hilbert-adjoint-equals-metric-times-S0")
@pytest.mark.parametrize("make_frame", _DIAGONAL_FRAME_CASES)
def test_parseval_frame_square_closes(make_frame):
    r"""``M.H = R/W`` and ``R.H = W·M`` — the frame square closes with ONE scalar.

    The SH-specific collapse of the general adjoints (:math:`M^* = S_0\circ
    G^{-1}`, :math:`R^* = d\,G\cdot(Y^{\mathsf T}W)`): per ℓ,
    :math:`d_\ell G_\ell = (2\ell+1)\cdot 4\pi/(2\ell+1) = 4\pi = W`, so the
    whole per-ℓ dressing collapses to the single scalar :math:`W = \sum_n w_n`
    — which IS the shipped scattering kernel's :math:`1/W` prefactor.
    `[M]` closure 5.6e-17 on the probe; every shipped sphere family measures
    degree-exact to ~1e-15.

    ⛔ The slab param is deliberately ABSENT, and D3
    (``test_diagonal_gram_suffices_for_the_collapse_and_dense_does_not_decide_it``)
    states why as a claim rather than a silent removal: `[M]` 2026-08-30,
    under the CORRECT dense Parseval metric the slab's collapse still
    fails (rel 2.65 on this file's seed) — its live ℓ=2 Gram diagonal
    ``[0.4, 0.8, 0.8]`` is not a per-ℓ scalar, so no :math:`G_\ell`
    exists there at ANY metric. ⚠ The honest quantifier (archivist,
    200-seed census): DIAGONAL is SUFFICIENT for the collapse — this
    gate's population — while DENSE does not decide it either way
    (``folded_product(4,6)`` L=3 is DENSE and satisfies it; the sphere
    rule ``product(4,4)`` L=2 breaks it).
    """
    frame = make_frame()
    W = float(frame.measure.weights.sum())
    rng = np.random.default_rng(4321)
    y = rng.standard_normal(frame.basis_space.shape)
    v = rng.standard_normal(frame.measure.weights.shape)
    np.testing.assert_allclose(
        frame.analysis.H.apply(y), frame.reconstruction.apply(y) / W,
        rtol=1e-12, atol=1e-13,
    )
    np.testing.assert_allclose(
        frame.reconstruction.H.apply(v), W * frame.analysis.apply(v),
        rtol=1e-12, atol=1e-13,
    )


@pytest.mark.foundation
def test_parseval_reds_under_the_pre_repair_continuum_metric():
    r"""The §6c witness: the Parseval isometry gate is LOADED on the metric, not blind.

    In-process pre-repair mutation (process-discipline: monkeypatch, never a git
    checkout): pre-seed the frame's cached-property slots with the UNDRESSED
    continuum-metric space (``SphericalHarmonicSpace.from_L`` — exactly what the
    pre-F-0 ``basis_space`` returned), and Parseval FAILS by the measured
    margin: the ratio is a weighted average of the per-ℓ factors
    :math:`(4\pi/(2\ell+1))^2 \ge 17.5` at L=1 (`[M]` 118.7 on the probe's
    seed). A BLIND gate would read 1.0 here (vv-principles #19: only the
    wrong-structure reading discriminates loaded from blind).
    """
    frame = Quadrature.level_symmetric(4).angular_frame(1)
    undressed = SphericalHarmonicSpace.from_L(1)
    # cached_property stores in the instance __dict__ — pre-seeding it IS the
    # pre-repair frame; no production code is touched and nothing needs undoing.
    frame.__dict__["basis_space"] = undressed
    frame.__dict__["test_space"] = undressed
    rng = np.random.default_rng(1234)
    c = rng.standard_normal(undressed.shape)
    psi = frame.basis.synthesize(c, frame.table)
    phi = frame.analysis.apply(psi)
    ratio = frame.basis_space.inner_product(phi, phi) / (
        frame.measure_space.inner_product(psi, psi)
    )
    assert ratio > 10.0, (
        f"pre-repair continuum metric read Parseval ratio {ratio:.3g} ≈ 1 — "
        f"the isometry gate would be BLIND to the wrong-side metric"
    )


@pytest.mark.foundation
@pytest.mark.catches("ERR-080")
def test_the_slab_frame_is_DIAGONAL_after_the_err080_repair():
    r"""⛔ **INVERTED 2026-09-02 (#429/ERR-080).** This gate pinned DENSE on the slab; the slab is now DIAGONAL.

    Read the history, because deleting it would lose the measurement that
    makes the inversion checkable. Until the repair a 1-D rule's frame bound
    the FULL real spherical harmonics to a measure forged onto :math:`S^2` as
    :math:`(\mu, 0, 0)`. `[M]` 2026-08-23 that Gram measured DENSE — total
    weight 2 not :math:`4\pi`, live slots ``[1, 1, 3]`` per degree,
    off-diagonals at 0.93 of the Cauchy–Schwarz scale — and P7 (2026-08-30)
    dressed it with the Moore–Penrose ``DenseMetric``. **That density was the
    defect's own signature**: the degenerate :math:`m > 0` harmonics were
    linearly dependent on the slab nodes, which is why the discrete Gram was
    rank-deficient and the per-mode scattering multiplier stopped being a
    function of the flux.

    After the repair a 1-D rule binds the Legendre basis on
    :math:`S^2/O(2)_x`. `[M]` 2026-09-02: the Gram is **DIAGONAL** —
    off-diagonal ``8.808e-17``, diagonal ``[2, 2/3, 0.4] = 2/(2\ell+1)``
    exactly, rank 3/3 — and the dressed metric is the plain reciprocal.

    ⭐ **The stronger claim this gate now carries, and the reason it is not
    merely a weakened re-pin.** The fabricated :math:`m \ne 0` slots are
    **unspellable**: the coefficient space is FLAT, shape ``(L+1,)``, so
    there is no slot for them to live in. The old pin asserted the fabricated
    :math:`\ell = 2` row ``[0.4, 0.8, 0.8]`` — `[M]` two-thirds honest, since
    ``0.4 = 2/5`` IS the correct :math:`m = 0` entry and only the two
    ``0.8``\ s were fabricated. This row pins the honest three and asserts the
    other two cannot be indexed at all.
    """
    frame = Quadrature.gauss_legendre(8).angular_frame(2)

    assert isinstance(frame.basis, LegendreBasis)
    assert frame.basis.gram_structure is GramStructure.DIAGONAL       # declared
    assert frame.discrete_gram_structure is GramStructure.DIAGONAL    # measured
    assert frame.table.shape == (8, 3), "the Legendre head is FLAT"

    gram = frame.discrete_gram
    diag = np.diagonal(gram)
    np.testing.assert_allclose(diag, [2.0, 2.0 / 3.0, 0.4], rtol=1e-12)
    assert float(np.max(np.abs(gram - np.diag(diag)))) < 1e-14
    assert np.linalg.matrix_rank(gram) == 3

    # the fabricated slots are UNSPELLABLE — not zeroed, absent
    assert frame.basis.space.shape == (3,)
    with pytest.raises(IndexError):
        _ = gram[2, 3]

    # the DIAGONAL dressing is the plain reciprocal, and no DenseMetric is
    # installed (the arm the old body asserted)
    assert frame.basis_space.metric is None
    weights = _dressed_measure(frame)
    assert weights is not None
    np.testing.assert_allclose(weights, 1.0 / diag, rtol=1e-12)

    # ⭐ and a NEW true fact the repair creates: the sphere collapse
    # d_l G_l = W holds EXACTLY on the slab now — (2l+1) * 2/(2l+1) = 2 = W
    # — where before it was unspellable (see D3 below).
    np.testing.assert_allclose(
        frame.basis.addition_theorem_factor * diag,
        float(frame.measure.weights.sum()),
        rtol=1e-12,
    )


@pytest.mark.foundation
def test_the_dense_matrix_parseval_dressing_rides_a_quotient_basis_frame():
    r"""The DENSE arm's flagship, re-keyed off the slab (#429 §5.5).

    ``folded_product(2, 4).angular_frame(3)`` binds the σ-even harmonic
    sub-basis on :math:`S^2/\sigma_y` and is untouched by the ERR-080 repair.
    `[M]` 2026-09-02: verdict DENSE, **live-block relative off-diagonal
    1.000** (a genuine coupling, not a rank artefact), rank 4/28 — and the
    draw-free separation is **2.7× today's retired flagship**
    (see :func:`test_no_diagonal_metric_can_satisfy_parseval_on_a_dense_frame`).

    ⚠ **Not** ``angular_frame(2)`` on the same rule, though it also reads
    DENSE: `[M]` its live-block relative off-diagonal is ``8.6e-17``, i.e. it
    is numerically diagonal and its DENSE verdict is driven by rank
    deficiency alone. A dense gate keyed there would be pinning a label.
    """
    frame = Quadrature.folded_product(2, 4).angular_frame(3)
    assert frame.discrete_gram_structure is GramStructure.DENSE

    gram = frame.discrete_gram
    live = np.diagonal(gram) > 1e-14
    block = gram[np.ix_(live, live)]
    scale = np.sqrt(np.outer(np.diagonal(block), np.diagonal(block)))
    relative_offdiag = float(
        np.max(np.abs(block - np.diag(np.diagonal(block))) / scale)
    )
    assert relative_offdiag > 0.5, (
        f"the dense flagship must be genuinely COUPLED, not merely "
        f"rank-deficient; live-block relative offdiag {relative_offdiag:.3e}"
    )

    metric = _dense_metric_of(frame)
    np.testing.assert_allclose(
        metric.matrix,
        np.linalg.pinv(
            (gram + gram.T) / 2.0, hermitian=True, rcond=_DENSE_METRIC_RCOND
        ),
        rtol=1e-12, atol=1e-15,
    )
    assert frame.basis_space.inner_product_weights is None
    assert frame.test_space is frame.basis_space


@pytest.mark.foundation
def test_indicator_frame_parseval_metric_is_the_inverse_region_mass():
    r"""The SAME theorem on the indicator frame: the Parseval metric is ``1/m_R``.

    :math:`G_{RR} = \sum_{i\in R} w_i = m_R` (the region mass), so the dressed
    metric is ``1/m_R`` on occupied regions and EXACTLY ``0.0`` on the empty one
    (the dead-slot arm — matching ``project``'s Moore–Penrose convention).
    Parseval on a region-wise-constant (band-limited) field:
    :math:`\|Mf\|^2_{1/m} = \sum_R m_R \bar f_R^2 = \|f\|^2_V` exactly.
    """
    frame = _indicator_frame(
        [0.0, 2.0, 4.0, 5.0], [0.5, 1.5, 2.5, 3.5], [1.0, 1.0, 2.0, 1.0],
    )  # R2 = [4, 5] is empty
    assert frame.discrete_gram_structure is GramStructure.DIAGONAL
    np.testing.assert_allclose(
        frame.basis_space.inner_product_weights, [0.5, 1.0 / 3.0, 0.0],
    )
    f = np.array([10.0, 10.0, 30.0, 30.0])  # constant per region — band-limited
    phi = frame.analysis.apply(f)
    np.testing.assert_allclose(
        frame.basis_space.inner_product(phi, phi),
        frame.measure_space.inner_product(f, f),
        rtol=1e-13,
    )


@pytest.mark.foundation
def test_overlap_frame_measures_dense_while_declaring_partition_of_unity():
    r"""The declared/measured Gram facts are INDEPENDENT — the overlap witness.

    :class:`OverlapBasis` DECLARES PARTITION_OF_UNITY (the cross-Gram row-sum
    probe is valid for ``project`` — :math:`R\mathbf 1 = \mathbf 1`), while its
    TRIAL Gram MEASURES DENSE (a straddling row makes two columns share
    support). Since P7 the DENSE verdict means the Parseval dressing is
    INSTALLED (a matrix metric on ``basis_space``) while ``project``
    keeps working through the row-sum probe — which never inherits the
    dressing (the stripped ``gram``, pinned by the C3 gate below).
    """
    frame = _overlap_frame()
    assert frame.basis.gram_structure is GramStructure.PARTITION_OF_UNITY  # declared
    assert frame.discrete_gram_structure is GramStructure.DENSE    # measured
    assert isinstance(frame.basis_space.metric, DenseMetric)       # dressed (P7)
    assert frame.basis_space.inner_product_weights is None         # ≠ Euclidean
    np.testing.assert_allclose(
        frame.project(np.array([2.0, 4.0, 6.0])), [8.0 / 3.0, 16.0 / 3.0],
    )


@pytest.mark.foundation
def test_the_gram_row_sum_probe_survives_a_dense_dressed_test_space():
    r"""C3 (P7 S2, battery arm M14): ``gram_inverse``/``project`` are
    CROSS-Gram machinery and must never inherit the test space's Parseval
    dressing.

    The pre-P7 spelling ``replace(self.test_space,
    inner_product_weights=diagonal)`` carried a dense-dressed test
    space's metric OBJECT into the probe — [M] 2026-08-30 (pre-flight,
    ``scratch/p7/preflight.log``): ``frame.project([2,4,6])`` read
    ``[7.0, 11.0]`` against the true ``[8/3, 16/3]`` (rel 1.625), a
    silent VALUE error, with no guard involved. Since CS4c step 6 item
    6.2c-ii the normalisation is an ARROW (:class:`CrossGramInverse`,
    ``test_space → basis_space``) whose action reads the probe diagonal
    and no space's metric — the leak is UNSPELLABLE, and this row asserts
    the arrow's ends, its diagonal and its value on the pre-seeded
    dense-dressed frame (the same idiom as the pre-repair-metric red gate).
    """
    from dataclasses import replace as _replace

    from orpheus.numerics.frame import CrossGramInverse
    from orpheus.numerics.metric import DenseMetric

    ob = OverlapBasis(
        edges_per_axis=(np.array([-0.5, 0.5, 1.5]),),
        partition_of=RealSpace(1),
        overlap_table=np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        fine=RealSpace(1),   # the three fine rows live on the same line the two coarse cells partition
    )
    measure = DiscreteMeasure(
        nodes=np.array([0.0, 0.5, 1.0]), weights=np.ones(3), support=RealSpace(1),
    )
    frame = GalerkinFrame(ob, measure)
    dressed = _replace(
        frame.basis.space,
        metric=DenseMetric.inverse_of(frame.discrete_gram),
    )
    vars(frame)["basis_space"] = dressed  # the cached_property pre-seed idiom
    assert frame.test_space is dressed  # the Galerkin identity holds on the seed
    probe = frame.gram_inverse
    assert isinstance(probe, CrossGramInverse)
    assert probe.domain is frame.test_space and probe.codomain is frame.basis_space
    np.testing.assert_allclose(probe.diagonal, [1.5, 1.5])   # the row sums of MR on this overlap frame
    np.testing.assert_array_equal(
        frame.project(np.array([2.0, 4.0, 6.0])), [8.0 / 3.0, 16.0 / 3.0],
    )
    # and the frame's own projector composes through the arrow
    projector = frame.conjugate(probe)
    np.testing.assert_allclose(
        projector.apply(np.array([2.0, 4.0, 6.0])),
        frame.reconstruction.apply(frame.project(np.array([2.0, 4.0, 6.0]))),
    )


@pytest.mark.foundation
@pytest.mark.parametrize("make_frame", _DENSE_FRAME_CASES)
def test_dense_frames_are_dressed_with_the_pseudo_inverse_gram(make_frame):
    r"""D1 — the DENSE arm's counterpart of the diagonal dressing gate.

    Four mechanisms, one law: verdict DENSE ⟹ ``basis_space`` carries a
    :class:`DenseMetric` whose matrix is the Moore–Penrose pseudo-inverse
    of the measured (symmetrized) Gram at the module's pinned ``rcond``,
    and the Galerkin identity ``test_space is basis_space`` survives the
    dressing.
    """
    frame = make_frame()
    assert frame.discrete_gram_structure is GramStructure.DENSE
    metric = _dense_metric_of(frame)
    g = frame.discrete_gram
    expected = np.linalg.pinv(
        (g + g.T) / 2.0, hermitian=True, rcond=_DENSE_METRIC_RCOND
    )
    np.testing.assert_allclose(metric.matrix, expected, rtol=1e-12, atol=1e-15)
    assert frame.test_space is frame.basis_space


def _dressed_measure(frame) -> np.ndarray | None:
    """The dressed head's DIAGONAL metric where the axis doctrine put it (CS4c
    step 6 item 6.2c-ii): the single head axis's measure on an axis-built
    coefficient space, the legacy weights slot on an axes-less one (the
    indicator bases, CS2)."""
    space = frame.basis_space
    if space.axes is not None:
        assert len(space.axes) == 1, "a coefficient space is single-axis"
        return space.axes[0].weights
    return space.inner_product_weights


def _continuum_measure(frame) -> np.ndarray:
    """The basis's own CONTINUUM Gram — the head axis's measure on ``basis.space`` (axis-built since item 6.2c-ii)."""
    space = frame.basis.space
    measure = space.axes[0].weights if space.axes is not None else space.inner_product_weights
    assert measure is not None
    return np.asarray(measure)


def _dense_metric_of(frame) -> DenseMetric:
    """The installed DenseMetric — POSITIONED on the derived object of an
    axis-built head (item 6.2c-i), bare on an axes-less coefficient space —
    narrowed, so a diagonal-dressed frame fails HERE, loudly."""
    metric = frame.basis_space.metric
    if isinstance(metric, FactoredMetric):
        assert len(metric.entries) == 1, f"{frame}: a coefficient space is single-axis"
        metric = metric.entries[0][1]
    assert isinstance(metric, DenseMetric), f"{frame} carries no DenseMetric (metric={metric!r})"
    return metric


def _dense_matrix_of(frame) -> np.ndarray:
    """The installed DenseMetric's matrix — narrowed, so a diagonal-dressed frame fails HERE, loudly."""
    return np.asarray(_dense_metric_of(frame).matrix)


def _parseval_ratio_range(gram: np.ndarray, metric: np.ndarray) -> tuple[float, float]:
    r"""The DRAW-FREE range of the Parseval ratio under a candidate metric.

    For a band-limited :math:`\psi = S_0 c` the analysis coefficients are
    :math:`\phi = Gc` and the discrete norm is :math:`c^\top G c`, so the
    Parseval ratio is the generalized Rayleigh quotient

    .. math:: r(c) \;=\; \frac{(Gc)^\top M (Gc)}{c^\top G c},

    whose RANGE over :math:`c \in \mathrm{range}(G)` is the generalized
    eigenvalue range of the pencil :math:`(G M G,\; G)` restricted there.
    Closed form, exact, and — unlike a random draw — a property of the FRAME.

    ⛔ **Why this replaces the seeded statistic.** `[M]` 2026-09-02, 400
    seeds on the frame D2/D4 used to pin: the same ratio ranged
    **0.2327 … 1.9975**. A committed floor of ``1.5`` there pinned a SEED,
    not a frame (``vv-principles`` #31), and would go red on any innocent
    fixture edit.
    """
    symmetric = (gram + gram.T) / 2.0
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    keep = eigenvalues > 1e-10 * max(float(eigenvalues.max()), 1.0)
    basis = vectors[:, keep]
    left = basis.T @ (symmetric @ metric @ symmetric) @ basis
    right = basis.T @ symmetric @ basis
    spectrum = scipy_eigh(left, right, eigvals_only=True)
    return float(np.min(spectrum)), float(np.max(spectrum))


def _diagonal_candidate_metric(gram: np.ndarray) -> np.ndarray:
    """The best diagonal candidate ``1/G_kk`` (0.0 on dead slots — Moore–Penrose)."""
    diagonal = np.diagonal(gram)
    live = diagonal > 0.0
    return np.diag(np.where(live, 1.0 / np.where(live, diagonal, 1.0), 0.0))


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
def test_no_diagonal_metric_can_satisfy_parseval_on_a_dense_frame():
    r"""D2 — THE WRONG-METRIC DISCRIMINATOR (``vv-principles`` #19's loaded reading).

    ⛔ **RE-KEYED 2026-09-02 (#429).** This gate rode ``slab-GL8-L2``, whose
    Gram is DIAGONAL after the ERR-080 repair — the separation it measured
    was a property of the fabrication. Its replacement is
    ``folded_product(2, 4).angular_frame(3)``, a σ-fold quotient-basis frame
    untouched by the repair.

    ⭐ And the replacement is STRICTLY stronger, in the exact sense the claim
    needs. `[M]` 2026-09-02, draw-free ranges of the Parseval ratio:

    ==========================  =====================  ==================
    metric                      range                  worst :math:`|r-1|`
    ==========================  =====================  ==================
    dense Moore–Penrose         ``[1.000, 1.000]``     ``0.000``
    best diagonal candidate     ``[1.000, 3.000]``     **2.000**
    undressed continuum         ``[10.53, 157.9]``     ``156.9``
    ==========================  =====================  ==================

    The diagonal candidate's range never drops BELOW 1 here, so *"no diagonal
    metric can satisfy Parseval"* is witnessed **strictly** rather than
    on-average — the retired slab witness had range ``[0.065, 2.000]``, i.e.
    it under- and over-shot, and a draw could land near 1.

    This family is the only correctness evidence the metric has: reciprocity
    holds to 1e-16 for EVERY invertible :math:`G` (#409) and can never
    adjudicate one.
    """
    frame = Quadrature.folded_product(2, 4).angular_frame(3)
    gram = np.asarray(frame.discrete_gram)

    dense = _parseval_ratio_range(gram, _dense_matrix_of(frame))
    diagonal = _parseval_ratio_range(gram, _diagonal_candidate_metric(gram))
    continuum = _parseval_ratio_range(
        gram,
        np.diag(np.asarray(_continuum_measure(frame), dtype=float).reshape(-1)),
    )

    assert dense == pytest.approx((1.0, 1.0), abs=1e-12), f"dense range {dense}"
    assert diagonal[1] > 2.5, f"diagonal candidate range {diagonal}"
    assert diagonal[0] >= 1.0 - 1e-12, (
        f"the diagonal candidate must be STRICTLY insufficient (never below 1) "
        f"on this frame; range {diagonal}"
    )
    assert continuum[0] > 10.0, f"continuum range {continuum}"


@pytest.mark.foundation
def test_diagonal_gram_suffices_for_the_collapse_and_dense_does_not_decide_it():
    r"""D3 — a DIAGONAL verdict SUFFICES for the sphere collapse; a DENSE one decides nothing.

    The collapse is :math:`M^\dagger = R/W`, which holds iff a per-degree
    scalar :math:`G_\ell` exists with :math:`d_\ell G_\ell = W`.

    ⛔ **RE-KEYED and INVERTED 2026-09-02 (#429).** This gate's failing
    witness was the slab, whose live :math:`\ell = 2` Gram diagonal was the
    fabricated ``[0.4, 0.8, 0.8]`` — three numbers where the theory has one,
    so no :math:`G_\ell` existed. `[M]` after the repair the slab's Gram is
    ``diag(2/(2\ell+1))`` and the Legendre dual factor is :math:`2\ell+1`, so
    :math:`d_\ell G_\ell = 2 = W` **exactly** and the collapse HOLDS there —
    the inversion is asserted below, because it is the sharpest statement the
    repair licenses and it would otherwise be lost.

    The DENSE failing witness moves to ``folded_product(2, 4).angular_frame(3)``:
    `[M]` matrix-level ``max|M† − R/W| / max|R/W| = 0.6564``, draw-free (both
    faces swept as matrices, no random probe).

    ⛔ The gate's NAME survives the re-key because the decidable statement is
    unchanged and was itself a 2026-08-30 refutation of an earlier name
    ("…is a sphere-family property"): `[M]` ``product(4,4)`` L=2 IS a sphere
    rule and BREAKS the collapse, while ``folded_product(4,6)`` L=3 measures
    DENSE and SATISFIES it — so DENSE does not decide it either way.
    """
    slab = Quadrature.gauss_legendre(8).angular_frame(2)
    assert slab.discrete_gram_structure is GramStructure.DIAGONAL
    assert _collapse_residual(slab) < 1e-12, (
        "the repaired slab frame must SATISFY the collapse: its Legendre Gram "
        "is diag(2/(2l+1)) and its dual factor is (2l+1), so d_l G_l = 2 = W"
    )

    dense = Quadrature.folded_product(2, 4).angular_frame(3)
    assert dense.discrete_gram_structure is GramStructure.DENSE
    residual = _collapse_residual(dense)
    assert residual > 0.5, (
        f"the DENSE witness must BREAK the collapse (rel {residual:.4g})"
    )

    # the isometry still holds there under the correct dense dressing — the
    # two properties are independent, which is the whole point of the name.
    assert _parseval_ratio_range(
        np.asarray(dense.discrete_gram),
        _dense_matrix_of(dense),
    ) == pytest.approx((1.0, 1.0), abs=1e-12)


def _collapse_residual(frame) -> float:
    r"""``max|M† − R/W| / max|R/W|`` with both faces swept as MATRICES — draw-free."""
    shape = frame.basis_space.shape
    n_modes = int(np.prod(shape))

    def sweep(operator):
        columns = []
        for index in range(n_modes):
            unit = np.zeros(n_modes)
            unit[index] = 1.0
            columns.append(np.asarray(operator.apply(unit.reshape(shape))).ravel())
        return np.array(columns).T

    adjoint = sweep(frame.analysis.H)
    reconstruction = sweep(frame.reconstruction) / float(frame.measure.weights.sum())
    return float(
        np.max(np.abs(adjoint - reconstruction)) / np.max(np.abs(reconstruction))
    )


@pytest.mark.foundation
def test_the_dense_dressing_reds_under_the_diagonal_and_the_pre_repair_metrics():
    r"""D4 — the DENSE arm's loadedness witness (``vv-principles`` #19).

    A green isometry reading is compatible with a LOADED gate and with a
    BLIND one; only the WRONG-metric reading discriminates. So this row
    installs two wrong metrics and requires each to break Parseval by a
    measured margin.

    ⛔ **RE-POSED DRAW-FREE 2026-09-02.** The retired body pinned floors of
    ``10`` and ``1.5`` on a single seeded draw; `[M]` the ``1.5`` floor was a
    SEED — the same statistic ranged **0.2327 … 1.9975** over 400 draws on
    the very frame it pinned. The floors below are on
    :func:`_parseval_ratio_range`, which is the exact range of that ratio
    over the whole band-limited subspace and so cannot move with a draw.

    `[M]` 2026-09-02 on ``folded_product(2,4).angular_frame(3)``: continuum
    ``[10.53, 157.9]``, best diagonal candidate ``[1.000, 3.000]``, dressed
    ``[1.000, 1.000]``.
    """
    frame = Quadrature.folded_product(2, 4).angular_frame(3)
    gram = np.asarray(frame.discrete_gram)

    dressed = _parseval_ratio_range(gram, _dense_matrix_of(frame))
    assert dressed == pytest.approx((1.0, 1.0), abs=1e-12), (
        f"the CONTROL leg: the honest dressing must read Parseval, else a "
        f"wrong-metric red carries no information; range {dressed}"
    )

    for label, metric, floor in (
        (
            "continuum",
            np.diag(np.asarray(_continuum_measure(frame), dtype=float).reshape(-1)),
            10.0,
        ),
        ("diagonal", _diagonal_candidate_metric(gram), 2.5),
    ):
        low, high = _parseval_ratio_range(gram, metric)
        assert max(abs(low - 1.0), abs(high - 1.0)) > floor - 1.0, (
            f"{label} metric read Parseval range [{low:.4g}, {high:.4g}] — "
            f"the gate would be blind to it"
        )


@pytest.mark.l1
@pytest.mark.catches("ERR-039")
def test_the_dressing_lands_parseval_on_the_production_anisotropic_frame():
    r"""D5 — the tier-of-observability gate for the production adjoint move.

    ``product(4,4).angular_frame(2)`` is a production-reachable
    ScatteringOperator configuration (``scattering.py`` builds
    ``quadrature.angular_frame(scattering_order)``) and its Gram measures
    DENSE. `[M]` 2026-08-30 (design pre-flight): dressing it moves
    ``frame.analysis.H`` by ``max|Δ| = 8.246`` — **rel 0.8995** on that
    probe's draw; ⚠ draw-banded 0.879–0.986 over 200 seeds, with the
    draw-free operator-level Frobenius relative at **0.980–0.985**
    across three DENSE frames (archivist census) — the recorded F-0
    limitation repaired (the undressed ``.H`` was the stored-metric
    sandwich, NOT the physical Hilbert adjoint), and NOTHING else in
    the 4371-test pre-flight scope observed it (plan-authoring §8,
    measured). This gate is where the change is visible: Parseval holds
    post-dressing (`[M]` 1.000000000000 this seed) and the pre-repair
    continuum metric reads 65.66.
    """
    frame = Quadrature.product(4, 4).angular_frame(2)
    assert frame.discrete_gram_structure is GramStructure.DENSE
    rng = np.random.default_rng(1234)
    c = rng.standard_normal(frame.basis.space.shape)
    psi = frame.basis.synthesize(c, frame.table)
    phi = frame.analysis.apply(psi)
    norm_w = frame.measure_space.inner_product(psi, psi)
    assert frame.basis_space.inner_product(phi, phi) / norm_w == pytest.approx(
        1.0, rel=1e-12
    )
    pre_repair = frame.basis.space.inner_product(phi, phi) / norm_w
    assert pre_repair > 10.0, f"pre-repair metric read {pre_repair:.4f}"


# ══════════════════════════════════════════════════════════════════════
# G0 (#429 tracker 2.2) — a frame's two halves must name ONE orbit space
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.foundation
@pytest.mark.catches("ERR-080")
class TestG0TheFrameBindsAlongAQuotientMap:
    r"""The construction-time arrow ``measure.support -> basis.domain``.

    A frame binds functions on ``basis.domain`` to a rule on
    ``measure.support``; that is well-posed **iff** the functions can be
    evaluated at the rule's nodes, i.e. iff a quotient map exists between the
    two. ONE predicate — the lattice one — decides all seven shipped
    pairings, and it is the frame-level statement of ERR-080: a basis eats
    points of its own orbit space or of a FINER one, never of a coarser one.

    ⛔ **§6c — G0 lands with no shipped production refusal.** `[M]`
    2026-09-02: after the fused commit every support the dispatch selects
    picks a basis G0 admits, so nothing production BUILDS is rejected. Its
    refusal witnesses therefore have to be CONSTRUCTED, and they are, below —
    from shipped classes, exactly the pairing the repair removed. A gate
    whose rejected input does not exist ships green and unable to fail.
    """

    def test_a_rule_on_the_CHART_is_refused_for_a_basis_on_the_orbit_space(self) -> None:
        """G0 compares the ENTRY, never its realization (tracker 2.4's ruling,
        at the frame): the same eight cosines declared on the chart ``[-1,1]``
        are refused for a Legendre basis on ``S^2/O2_x`` — no quotient map
        ``[-1,1] -> S^2/O2_x`` exists, the chart is not the orbit space —
        and admitted once declared on the entry. Built DIRECTLY on the frame,
        because the quadrature's dispatch refuses a chart-level rule one guard
        earlier and would otherwise mask a G0 that compared realizations
        (`[M]` 2026-09-02, battery arm m10 was BLIND without this row)."""
        rule = Quadrature.gauss_legendre(8).measure
        on_chart = DiscreteMeasure(
            nodes=rule.nodes, weights=rule.weights, support=COSINE_INTERVAL,
        )
        with pytest.raises(ValueError, match="no quotient map"):
            GalerkinFrame(LegendreBasis(L=2), on_chart)
        on_entry = DiscreteMeasure(
            nodes=rule.nodes, weights=rule.weights,
            support=SPHERE.quotient(SubgroupOfO3.O2("x")),
        )
        frame = GalerkinFrame(LegendreBasis(L=2), on_entry)
        assert frame.table.shape == (8, 3)

    def test_the_four_shipped_pairings_are_admitted(self) -> None:
        r"""Rows 1–4 of the pairing table, each a frame production actually builds (or was asked to).

        =====  =====================  ==========  =======================
        row    measure support        basis       spent :math:`\subseteq` has
        =====  =====================  ==========  =======================
        1      :math:`S^2`            SH          ``Trivial ⊆ Trivial``
        2      :math:`S^2/\sigma_y`   MirrorEven  ``σ_y ⊆ σ_y``
        3      :math:`S^2/O(2)_x`    Legendre    ``O2_x ⊆ O2_x``
        4 ⭐   :math:`S^2`            Legendre    ``Trivial ⊆ SO2_x``
        =====  =====================  ==========  =======================

        Row 4 is the pairing the user asked to be buildable:
        :math:`P_\ell(\Omega\cdot\hat e_x)` on a Lebedev or level-symmetric
        rule, reached by the entry's own quotient map.
        """
        sphere_rule = Quadrature.level_symmetric(4)
        fold_rule = Quadrature.folded_product(4, 8)
        slab_rule = Quadrature.gauss_legendre(8)

        # row 1 — the full harmonics on a full-sphere rule
        row1 = GalerkinFrame(SphericalHarmonicBasis(L=2), sphere_rule.measure)
        assert row1.table.shape == (24, 3, 5)

        # row 2 — the sigma-even sub-basis on the fold (dispatch-selected)
        row2 = fold_rule.angular_frame(2)
        assert isinstance(row2.basis, MirrorEvenSphericalHarmonicBasis)

        # row 3 — the Legendre basis on the slab's own orbit space
        row3 = slab_rule.angular_frame(2)
        assert isinstance(row3.basis, LegendreBasis)
        assert row3.table.shape == (8, 3)

        # row 4 — the Legendre basis on a FULL-SPHERE rule
        row4 = GalerkinFrame(LegendreBasis(L=2), sphere_rule.measure)
        assert row4.table.shape == (24, 3)

        for frame in (row1, row2, row3, row4):
            assert frame.descent is not None
            assert frame.descent.domain == frame.measure.support
            assert frame.descent.codomain == frame.basis.domain
            assert frame.test_descent is frame.descent  # Galerkin: test IS trial

    def test_the_two_refusals_and_the_admitted_fold_pairing_are_constructed_from_shipped_classes(self) -> None:
        r"""Rows 5–7 — including **the Part I bug**, which is ERR-080's own pairing — and row 7 ADMITS since #432.

        Row 5 is the frame the tree built for every 1-D solve until
        2026-09-02: the FULL harmonics (``Trivial``) on a rule whose measure
        lives on :math:`S^2/O(2)_x`. ``Trivial ⊉ O(2)_x``, so no arrow
        exists — a coarser orbit space cannot map onto a finer one, which is
        exactly the direction the forged :math:`(\mu, 0, 0)` nodes pretended
        to travel.
        """
        slab = Quadrature.gauss_legendre(8).measure
        fold = Quadrature.folded_product(4, 8).measure

        # row 5 — THE PART I BUG
        with pytest.raises(ValueError, match="no quotient map"):
            GalerkinFrame(SphericalHarmonicBasis(L=2), slab)

        # row 6 — the full harmonics on a mirror fold
        with pytest.raises(ValueError, match="no quotient map"):
            GalerkinFrame(SphericalHarmonicBasis(L=2), fold)

        # row 7 — the Legendre basis on a sigma_y fold: ADMITTED since
        # 2026-09-02 (#432). The entry is named by its stabiliser O(2)_x
        # (rotations about x AND the mirrors through it), so the basis
        # declares the FULL group its functions have and `O2('x') ⊇
        # Mirror('y')` gives the arrow S^2/sigma_y -> S^2/O2_x. Until then
        # Basis.invariance_group could only be derived as the lower bound
        # SO2('x') and this mathematically admissible pairing was refused —
        # recorded here as a row for the same reason the refusal was.
        admitted = GalerkinFrame(LegendreBasis(L=2), fold)
        assert admitted.descent.domain == fold.support
        assert admitted.descent.codomain == LegendreBasis(L=2).domain
        assert admitted.table.shape == (fold.n_points, 3)
        # and the mirror that FLIPS the axis is still outside O(2)_x: the
        # Legendre basis about x on an x-FOLDED rule is refused, while the
        # same basis about z (sigma_x fixes the z-axis) is admitted.
        x_fold = Quadrature.product(4, 8).quotient(SubgroupOfO3.Mirror("x")).measure
        with pytest.raises(ValueError, match="no quotient map"):
            GalerkinFrame(LegendreBasis(L=2, axis="x"), x_fold)
        GalerkinFrame(LegendreBasis(L=2, axis="z"), x_fold)

    def test_the_message_names_both_halves_and_both_groups(self) -> None:
        r"""The refusal is a DIAGNOSIS, not a wall — it says which spaces, which groups, and where to look."""
        with pytest.raises(ValueError) as excinfo:
            GalerkinFrame(
                SphericalHarmonicBasis(L=2), Quadrature.gauss_legendre(8).measure
            )
        message = str(excinfo.value)
        for fragment in ("S^2", "S^2/O2_x", "spent O2_x", "has Trivial", "ERR-080"):
            assert fragment in message, f"{fragment!r} missing from: {message}"

    def test_g0_fires_on_BOTH_construction_paths(self) -> None:
        r"""``GalerkinFrame`` has a hand-written ``__init__``, so the dataclass ``__post_init__`` is not enough.

        ``plan-authoring`` §6b in miniature: one guard, two doors. A gate on
        one door certifies the other by accident.
        """
        slab = Quadrature.gauss_legendre(8).measure
        with pytest.raises(ValueError, match="no quotient map"):
            GalerkinFrame(SphericalHarmonicBasis(L=2), slab)          # ctor path
        with pytest.raises(ValueError, match="no quotient map"):
            PetrovGalerkinFrame(                                       # dataclass path
                basis=SphericalHarmonicBasis(L=2),
                measure=slab,
                test_basis=SphericalHarmonicBasis(L=2),
            )

    def test_the_table_is_the_pullback_along_the_descent_arrow(self) -> None:
        r"""B9 — ``frame.table == basis.evaluate(π(nodes))``, and the raw-node tabulation RAISES.

        The arrow is not decoration: the table is the basis pulled back along
        it. On a Legendre-on-a-full-sphere frame the pullback is the entry's
        quotient map, and `[M]` ``π(nodes) == axis_cosines(0)`` bit-exactly on
        every sphere rule — so the frame's table is the Legendre table at the
        rule's own polar cosines.

        NEGATIVE leg: hand the flat basis the RAW ``(N, 3)`` nodes and it
        must REFUSE, rather than silently broadcasting three columns into a
        width the basis would accept.
        """
        entry = SPHERE.quotient(SubgroupOfO3.O2("x"))
        for build in (
            lambda: Quadrature.level_symmetric(4),
            lambda: Quadrature.lebedev(11),
            lambda: Quadrature.product(4, 6),
        ):
            rule = build()
            basis = LegendreBasis(L=3)
            frame = GalerkinFrame(basis, rule.measure)
            nodes = np.asarray(rule.measure.nodes, dtype=float)

            cosines = entry.quotient_map(nodes)
            assert np.array_equal(
                cosines, np.asarray(rule.axis_cosines(0), dtype=float)
            )
            assert np.array_equal(frame.table, basis.evaluate(cosines))
            assert np.array_equal(frame.table, basis.evaluate(frame.descent(nodes)))

            # the raw-node tabulation is refused, not broadcast
            with pytest.raises(ValueError, match="not all lie on it|expected points"):
                basis.evaluate(np.column_stack([nodes[:, 0], nodes[:, 1]]))

    def test_g0_respects_the_subgroup_order_relation(self) -> None:
        r"""``vv-principles`` #15 — the predicate and the lattice cross-check each other.

        G0 is a predicate on a pair of orbit spaces and ``SubgroupOfO3.contains``
        is the order relation it must respect. Gated over every edge among the
        shipped angular groups, neither half can be wrong alone without this
        reddening — which is what makes it worth more than either a
        per-pairing table or a per-lattice-edge table.
        """
        nodes = np.asarray(Quadrature.level_symmetric(4).measure.nodes, dtype=float)
        weights = np.asarray(Quadrature.level_symmetric(4).weights, dtype=float)
        # ⚠ The unfolded case is the BARE sphere, not ``SPHERE.quotient(Trivial)``
        # — that is what every shipped unfolded rule declares (`[M]` LS4,
        # lebedev(11), product(4,6) all carry ``support=Sphere()``), and the two
        # are NOT the same object here: ``S^2/Trivial`` is a ``Quotient`` whose
        # realization is ``S^2``, and ``quotient_onto(S^2/Trivial, S^2)`` returns
        # ``None`` although the two spaces are isomorphic. Unreachable from any
        # shipped producer, so it is recorded rather than gated.
        supports = {
            SubgroupOfO3.Trivial: SPHERE,
            SubgroupOfO3.O2("x"): SPHERE.quotient(SubgroupOfO3.O2("x")),
            SubgroupOfO3.Mirror("y"): SPHERE.quotient(SubgroupOfO3.Mirror("y")),
        }
        checked = 0
        for spent, support in supports.items():
            measure = DiscreteMeasure(nodes=nodes, weights=weights, support=support)
            for has, basis in (
                (SubgroupOfO3.Trivial, SphericalHarmonicBasis(L=2)),
                (SubgroupOfO3.O2("x"), LegendreBasis(L=2)),
                (
                    SubgroupOfO3.Mirror("y"),
                    MirrorEvenSphericalHarmonicBasis(L=2, mirror_axis=1),
                ),
            ):
                admissible = has.contains(spent)
                if admissible:
                    GalerkinFrame(basis, measure)
                else:
                    with pytest.raises(ValueError, match="no quotient map"):
                        GalerkinFrame(basis, measure)
                checked += 1
        assert checked == 9


# ══════════════════════════════════════════════════════════════════════
# G0 on a MIRROR FOLD — the pairing #432 turned from refused into admitted
# ══════════════════════════════════════════════════════════════════════


class TestTheStabiliserDecidesAdmissionOnAMirrorFold:
    r"""``LegendreBasis(axis=a)`` on a :math:`\sigma_b`-folded rule is
    admitted **iff** :math:`b \ne a` — the lattice fact
    :math:`\sigma_b \in O(2)_a \iff b \ne a`, read at the frame tier.

    Before #432 the Legendre basis could only DERIVE the lower bound
    :math:`SO(2)_a` from its domain, so :math:`\sigma_b \notin SO(2)_a` for
    every :math:`b` and **all six** off-axis pairings were over-refused —
    a mathematically admissible frame the tree could not build. Naming the
    orbit space by its stabiliser moved three of them per axis, and left the
    on-axis one refused, which is the discriminating half: :math:`\sigma_a`
    FLIPS :math:`\hat e_a`, so it is the one coordinate mirror outside
    :math:`O(2)_a`.

    Claim layer: **term-level (L0)**, closed form (the lattice edge) plus a
    **flux-shape**-free numerical witness (the isotropic moments below). No
    eigenvalue claim is made anywhere in this class.
    """

    @staticmethod
    def _mirror_folds() -> "dict[str, DiscreteMeasure]":
        r"""The three :math:`\sigma_b`-folded measures, and the honest note
        about where each comes from.

        ⚠ **Only two of the three are SHIPPED.** ``product(4, 8)`` folds
        under :math:`\sigma_x` and :math:`\sigma_y` (the fold acts within a
        polar level), and `[M]` 2026-09-02 refuses to fold under
        :math:`\sigma_z` — *"the quotient does not act on the fiber: level 0
        (invariant value -0.861136…)"* — because :math:`\sigma_z` permutes
        the polar LEVELS rather than acting inside them. The z row is
        therefore CONSTRUCTED here (a hand-built half-sphere measure declared
        on ``S^2/sigma_z``), and it is labelled as such: a §6c denominator
        that silently counted it as shipped would over-report the gate's
        production reach.
        """
        rule = Quadrature.product(n_mu=4, n_phi=8)
        nodes = np.column_stack([rule.mu_x, rule.mu_y, rule.mu_z])
        upper = nodes[:, 2] >= 0.0
        return {
            "x": rule.quotient(SubgroupOfO3.Mirror("x")).measure,   # SHIPPED
            "y": rule.quotient(SubgroupOfO3.Mirror("y")).measure,   # SHIPPED
            "z": DiscreteMeasure(                                   # CONSTRUCTED
                nodes=nodes[upper],
                weights=2.0 * rule.weights[upper],
                support=SPHERE.quotient(SubgroupOfO3.Mirror("z")),
            ),
        }

    def test_the_nine_axis_by_fold_pairings_split_exactly_on_the_lattice_edge(
        self,
    ) -> None:
        r"""The full :math:`3 \times 3` table — **9 pairings, 6 admitted, 3
        refused**, and the refused ones are exactly the diagonal.

        `[M]` 2026-09-02: every admitted frame tabulates ``(n_points, L+1)``
        and its descent runs ``S^2/sigma_b -> S^2/O2_a``; every refused one
        raises ``ValueError`` naming *"no quotient map"*. The verdict is
        compared against ``SubgroupOfO3.O2(a).contains(Mirror(b))`` computed
        in the same loop, so the frame tier and the lattice tier cross-check
        each other (``vv-principles`` #15) rather than the test restating a
        table.
        """
        folds = self._mirror_folds()
        admitted = refused = 0
        for fold_axis, measure in folds.items():
            assert measure.support.name == f"S^2/sigma_{fold_axis}"
            for basis_axis in ("x", "y", "z"):
                basis = LegendreBasis(L=2, axis=basis_axis)
                edge = SubgroupOfO3.O2(basis_axis).contains(
                    SubgroupOfO3.Mirror(fold_axis)
                )
                assert edge is (basis_axis != fold_axis)
                if edge:
                    frame = GalerkinFrame(basis, measure)
                    assert frame.table.shape == (measure.n_points, 3)
                    assert frame.descent.domain == measure.support
                    assert frame.descent.codomain == basis.domain
                    admitted += 1
                else:
                    with pytest.raises(ValueError, match="no quotient map"):
                        GalerkinFrame(basis, measure)
                    refused += 1
        assert (admitted, refused) == (6, 3)

    @pytest.mark.parametrize("L", [2, 4, 6])
    def test_the_admitted_fold_frame_reproduces_the_isotropic_moments(
        self, L: int
    ) -> None:
        r"""The numerical witness for the newly-admitted pairing, on the
        SHIPPED fold ``folded_product(4, 8)`` (:math:`\sigma_y`).

        Analysing the constant field :math:`\psi \equiv 1` must give
        :math:`\phi_0 = \int_{S^2} \mathrm{d}\Omega = 4\pi` and
        :math:`\phi_\ell = 0` for :math:`\ell \ge 1` — a closed form
        (:math:`\int_{-1}^{1} P_\ell \,\mathrm{d}\mu = 2\delta_{\ell 0}`,
        pushed forward by Archimedes' hat box) that needs no solver, no
        reference implementation and no tolerance beyond round-off.

        `[M]` 2026-09-02 on ``folded_product(4, 8)`` (16 nodes,
        :math:`\sum w = 4\pi`): :math:`\phi_0` reproduces
        :math:`4\pi = 12.566370614359172` and
        :math:`\max_{\ell \ge 1} |\phi_\ell|` reads **8.465e-16** at
        :math:`L = 2` and **1.388e-15** at :math:`L = 4` and :math:`L = 6`,
        against the ``1e-13`` absolute band asserted here — two orders of
        margin, and the band is absolute because the quantity being bounded
        is a difference of :math:`O(4\pi)` sums.

        ⚠ Non-vacuity: :math:`\phi_0` is asserted at the SAME band, so a
        frame that returned zeros everywhere would fail rather than pass the
        :math:`\ell \ge 1` leg vacuously.
        """
        fold = Quadrature.folded_product(n_mu=4, n_phi=8).measure
        assert fold.support == SPHERE.quotient(SubgroupOfO3.Mirror("y"))

        frame = GalerkinFrame(LegendreBasis(L=L), fold)
        moments = frame.analysis.apply(np.ones(fold.n_points))
        assert moments.shape == (L + 1,)
        np.testing.assert_allclose(moments[0], 4.0 * np.pi, atol=1e-13)
        assert float(np.max(np.abs(moments[1:]))) < 1e-13, moments
        # the fold's own mass, so the phi_0 row is not reading a coincidence
        np.testing.assert_allclose(fold.weights.sum(), 4.0 * np.pi, atol=1e-13)

    def test_the_refusal_on_the_ON_AXIS_fold_names_both_orbit_spaces(self) -> None:
        r"""The refused diagonal is a DIAGNOSIS: the message names the
        basis's orbit space, the rule's, and both spent groups — so a reader
        can see that :math:`\sigma_a \notin O(2)_a` rather than only that
        something was rejected.
        """
        x_fold = Quadrature.product(n_mu=4, n_phi=8).quotient(
            SubgroupOfO3.Mirror("x")
        ).measure
        with pytest.raises(ValueError) as excinfo:
            GalerkinFrame(LegendreBasis(L=2, axis="x"), x_fold)
        message = str(excinfo.value)
        for fragment in ("S^2/O2_x", "S^2/sigma_x", "no quotient map"):
            assert fragment in message, f"{fragment!r} missing from: {message}"
