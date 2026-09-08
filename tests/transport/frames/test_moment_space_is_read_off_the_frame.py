r"""#429 tracker 2.5 — the angular moment space is READ off the frame, never minted from ``L``.

The defect this gates (`[M]` 2026-09-02, the fused step's opener): the
angular moment space had EIGHT homes. The frame carried it
(``frame.basis.space``) and seven production sites re-minted it from the
integer ``L`` as ``SphericalHarmonicSpace.from_L(L)`` — the scattering
operator's :math:`\Lambda` ends (three spellings), the fission and (n,2n)
:math:`\ell = 0` ends, the moment-flux field's head, and ``truncate`` —
while two ``isinstance`` doors on :class:`HarmonicFrame` admitted one basis
class only. Every copy silently chose the full-sphere family; the day a
1-D rule binds its Legendre basis, every copy mismatches the frame at the
``(name, shape)`` composability guards, and the FIRST mismatch is at
:math:`L = 0` on every solve (fission and (n,2n) mint there).

Three gates, each with the input it rejects (``plan-authoring`` §6c):

* **the ROUTE gate** — bind a FOREIGN truncated basis (not a spherical
  harmonic subclass at all, with a RENAMED coefficient space) into the
  quadrature's frame and require every operator end and every moment
  field space to MOVE with it. A reverted producer (one that still mints
  from ``L``) fails the composability guard loudly, which is the red.
  The mutant is unconstructible before the door widened — so the door
  and the producers are ONE step (``plan-authoring`` §6b);
* **the METRIC gate** — ⛔ RE-POSED at CS4c step 6 item 6.2c-ii (ruling
  R-6.2c-1, 2026-09-08). Landing A (2026-09-02) bound the basis's own
  CONTINUUM space (``basis.space``) and recorded as its reason that *"Λ's
  Hilbert adjoint under the continuum end is its transpose exactly while
  the dressed end would move it on 10 of 33 rows (the dense-Gram rows)"*
  and that the dressed metric *"would move apply_metric by 96–161 %"*.
  `[M]` re-measured (``scratch/_step6_2c/p3_scan_161.py``,
  ``p4_lambda_adjoint.py``, ``p9_on_range.py``, 33 shipped (rule, L)
  rows): the adjoint moves on **5** rows for an ARBITRARY head draw,
  **3 of them DIAGONAL-Gram** (the mechanism is the Parseval metric's
  Moore–Penrose projection of the slots a folded rule cannot see, not
  Gram density), and on **0 of 33** for a physical moment ``φ = Mψ``;
  and NO statistic reproduces 161 % (the draw-free per-element movement
  spans 0.5 %…100 % over the 33 rows). What the continuum binding COST
  was measured too: Parseval ``‖Mψ‖² = ‖ψ‖²_W`` fails on 33 of 33 rows
  under it (ratio 3.41…157.91) and holds on 33 of 33 under the dressed
  metric. The tree therefore binds the FRAME's Parseval-dressed
  ``basis_space`` everywhere — the operator ends, the moment fields, the
  carrier's cached moment space — and the two heads are structurally
  UNEQUAL (the head is axis-built; its measure enters the identity), so
  this gate asserts the ends ARE the frame's space and are NOT the
  continuum mint, with the continuum head as its negative control;
* **the DOOR gate** — the frame asks for the
  :class:`~orpheus.numerics.basis.base.TruncatedBasis` SURFACE (a
  truncation order), typed, at the door: an indicator trial is refused
  there, and a foreign truncated basis is admitted where the old door
  refused it.

Foundation mark: software invariants (route, metric identity, typing),
mutation-proven; no physics claim rides here.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from orpheus.geometry import BC, Mesh1D
from orpheus.numerics.basis.base import Basis, GramStructure, TruncatedBasis
from orpheus.numerics.basis.indicator_basis import IndicatorBasis
from orpheus.numerics.metric import DenseMetric, FactoredMetric
from orpheus.numerics.basis.spherical_harmonic_basis import SphericalHarmonicBasis
from orpheus.numerics.frame import GalerkinFrame
from orpheus.numerics.manifold import SPHERE, Manifold, RealSpace
from orpheus.numerics.measure import DiscreteMeasure
from orpheus.numerics.operator import IncompatibleOperatorComposition, OperatorProduct
from orpheus.numerics.quadrature import Quadrature
from orpheus.numerics.space import FunctionSpace, TensorProductSpace
from orpheus.numerics.spaces.legendre_space import LegendreSpace
from orpheus.numerics.spaces.spherical_harmonic_space import SphericalHarmonicSpace
from orpheus.sn.mesh.augmented_mesh import SNMesh
from orpheus.transport.fields.harmonic_moment_flux import HarmonicMomentFlux
from orpheus.transport.frames import HarmonicFrame
from orpheus.transport.operators.fission import FissionOperator
from orpheus.transport.operators.n2n import N2NOperator
from orpheus.transport.operators.scattering import ScatteringOperator
from orpheus.transport.operators.transfer import LegendreMomentTransfer
from orpheus.transport.material_field import TransferMaterialField
from tests.sn._test_helpers import material_xs_from_raw, placeholder_materials

pytestmark = pytest.mark.foundation

_MUTANT_NAME = "mutated_moment_space"
_SIGS0 = np.array([[0.20, 0.00], [0.05, 0.18]])
_SIGS1 = np.array([[0.02, 0.00], [0.01, 0.015]])
_SIG2 = np.array([[0.00, 0.03], [0.01, 0.00]])


# ── the FOREIGN truncated basis: carries L and a renamed space, is NOT an SH subclass ──


@dataclass(frozen=True)
class _ForeignTruncatedBasis(Basis):
    """The same functions as the rule's OWN basis, delegated — with a RENAMED
    coefficient space, and NOT a ``SphericalHarmonicBasis`` subclass. It
    carries a truncation order, which is the ONE thing the door asks for;
    its renamed space is ``(name, shape)``-unequal to the honest mint, which
    is what the route gate reads.

    ⛔ **The parent is a FIELD since 2026-09-02 (#429).** It used to hard-code
    ``SphericalHarmonicBasis(L)``, hence ``domain = SPHERE`` — and the frame's
    G0 arrow now refuses a basis on :math:`S^2` bound to a rule whose measure
    lives on :math:`S^2/O(2)_x`, which is the whole of ERR-080. Delegating
    the domain keeps the mutant admissible **wherever the honest basis is**,
    so the ONE thing that differs between the two runs stays the coefficient
    SPACE — which is what this module's route gate is about (``vv-principles``
    #18: a mutation that also breaks a structural law over-states its own
    coverage).
    """

    L: int
    parent: Basis

    @classmethod
    def like(cls, basis: Basis) -> "_ForeignTruncatedBasis":
        """A foreign twin of ``basis``: same functions, same domain, renamed space."""
        order = getattr(basis, "L", None)
        assert isinstance(order, int)
        return cls(L=order, parent=basis)

    def _parent(self) -> Basis:
        # the parent FOLLOWS this twin's order, so a re-mint at another order
        # (``at_order``, item 6.2c-ii — the head truncates through it) still
        # delegates to the honest family at THAT order
        return self.parent.at_order(self.L) if getattr(self.parent, "L", None) != self.L else self.parent  # type: ignore[attr-defined]

    def at_order(self, L_new: int, /) -> "_ForeignTruncatedBasis":
        return replace(self, L=L_new)

    def evaluate(self, points, /):
        return self._parent().evaluate(points)

    def synthesize(self, coefficients, table, /):
        return self._parent().synthesize(coefficients, table)

    def analyze(self, values, table, weights, /):
        return self._parent().analyze(values, table, weights)

    def analyze_transpose(self, coefficients, table, weights, /):
        return self._parent().analyze_transpose(coefficients, table, weights)

    def reconstruct(self, coefficients, table, /):
        return self._parent().reconstruct(coefficients, table)

    def reconstruct_transpose(self, values, table, /):
        return self._parent().reconstruct_transpose(values, table)

    def mass_matrix(self, measure, /):
        return self._parent().mass_matrix(measure)

    @property
    def gram_structure(self) -> GramStructure:
        return self._parent().gram_structure

    @property
    def domain(self) -> Manifold:
        return self._parent().domain

    @property
    def space(self) -> FunctionSpace:
        return replace(self._parent().space, name=_MUTANT_NAME)


def _slab(nx: int = 4, n_ord: int = 4, ng: int = 2) -> SNMesh:
    mesh = Mesh1D(
        edges=np.linspace(0.0, 4.0, nx + 1), mat_ids=np.zeros(nx, dtype=int),
        bc_left=BC("reflective"), bc_right=BC("reflective"),
    )
    return SNMesh(mesh, Quadrature.gauss_legendre(n_ordinates=n_ord), placeholder_materials(ng=ng))


def _mat_xs(nx: int = 4):
    cells = {0: (np.arange(nx), np.zeros(nx, dtype=int))}
    return material_xs_from_raw(
        sig_s={0: [_SIGS0, _SIGS1]}, sig2={0: _SIG2},
        cells_by_mat=cells, ng=2, nx=nx, ny=1,
    )


def _inner_factor_domain_name(kernel: OperatorProduct) -> str:
    """``frame.conjugate(X)`` is ``R ∘ (X ∘ M)``; the domain of ``X`` is the end under test."""
    inner = kernel.b
    assert isinstance(inner, OperatorProduct)
    domain = inner.a.domain
    assert domain is not None
    return domain.name


def _bind_foreign(sn: SNMesh, L: int) -> HarmonicFrame:
    """Install the foreign basis as the quadrature's frame at ``L`` through
    the production chain's own cache — so ``HarmonicFrame.for_space`` (the
    ONE spelling every consumer uses) hands the foreign frame out."""
    # the foreign frame rides the SAME measure the production frame carries
    # (today the forged 3-D padding of a 1-D rule; after #429's fix the rule's
    # own) — the test is about the SPACE, not the nodes
    honest = sn.quad.angular_frame(L)
    cache = sn.quad._angular_frames or {}
    cache[L] = GalerkinFrame(_ForeignTruncatedBasis.like(honest.basis), honest.measure)
    object.__setattr__(sn.quad, "_angular_frames", cache)
    frame = HarmonicFrame.for_space(sn.angular_bulk_space, L)
    assert frame.basis.space.name == _MUTANT_NAME, "the route through for_space did not reach the foreign frame"
    return frame


# ═══════════════════════════════════════════════════════════════════════
# A1 — the ROUTE gate
# ═══════════════════════════════════════════════════════════════════════


def test_swapping_the_frames_basis_moves_every_operator_end_and_field_space() -> None:
    """Every producer that used to mint ``SphericalHarmonicSpace.from_L(L)``
    now reads the frame's basis: with a FOREIGN basis bound, the
    scattering, fission and (n,2n) kernels COMPOSE (a reverted producer
    would fail the ``(name, shape)`` guard), the moment field's head is the
    foreign space, and the tier-2 mints take the basis, not the integer."""
    sn = _slab()
    L = 1
    _bind_foreign(sn, L)
    _bind_foreign(sn, 0)          # fission mints at ℓ = 0 on every solve
    mat = _mat_xs()
    composite = sn.full_field_space

    S = ScatteringOperator.from_solver_data(mat_xs=mat, space=composite, scattering_order=L)
    lam = S._moment_transfer(skip_l0=False)
    assert lam.domain.name == _MUTANT_NAME and lam.codomain.name == _MUTANT_NAME
    assert S.full_transfer_kernel is not None      # R∘Λ∘M composes on the foreign ends
    assert S.kernel is not None

    F = FissionOperator.from_solver_data(mat_xs=mat, space=composite)
    assert _inner_factor_domain_name(F.full_fission_kernel) == _MUTANT_NAME   # R∘(F₀∘M)

    N = N2NOperator.from_solver_data(mat_xs=mat, space=composite, scattering_order=L)
    assert _inner_factor_domain_name(N.full_transfer_kernel) == _MUTANT_NAME

    field = HarmonicMomentFlux.zeros_for_mesh_and_L(sn, L)
    assert isinstance(field.space, TensorProductSpace)
    assert field.space.factors[0].name == _MUTANT_NAME
    # truncation stays in the head's OWN family, one order down — the head
    # keeps its IDENTITY (name) and only its order moves; a truncate that
    # re-minted the family from an integer would hand back the default name
    truncated = field.truncate(0).space
    assert isinstance(truncated, TensorProductSpace)
    # the truncated head keeps its own FAMILY's L = 0 layout — ``(1, 1)`` for
    # the rectangular harmonics, ``(1,)`` for the flat Legendre family the
    # slab now binds. Read off the honest head, so this line cannot drift.
    assert truncated.factors[0].shape == sn.quad.angular_frame(0).basis.space.shape
    assert truncated.factors[0].name == _MUTANT_NAME

    foreign = _ForeignTruncatedBasis.like(sn.quad.angular_frame(L).basis)
    assert LegendreMomentTransfer.on_basis(
            TransferMaterialField.scattering(mat), foreign,
        ).domain.name == _MUTANT_NAME
    assert LegendreMomentTransfer.on_basis(
            TransferMaterialField.n2n(mat), foreign, skip_l0=False,
        ).codomain.name == _MUTANT_NAME

    # the negative leg: an end minted from L alone does NOT compose with the
    # foreign frame — this is the red every reverted producer produces
    minted = SphericalHarmonicSpace.from_L(L)
    stale = LegendreMomentTransfer(
        S.transfer, skip_l0=False, domain=minted, codomain=minted,
    )
    with pytest.raises(IncompatibleOperatorComposition, match=r"A\.domain == B\.codomain"):
        S.flux_analysis.frame.conjugate(stale)


# ═══════════════════════════════════════════════════════════════════════
# A2 — the METRIC-IDENTITY gate (and the fork control)
# ═══════════════════════════════════════════════════════════════════════


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


def _head_measure(space: FunctionSpace) -> np.ndarray | None:
    """The head's diagonal metric where the axis doctrine put it — the single head axis's measure."""
    assert space.axes is not None and len(space.axes) == 1, "a moment head is a single-axis space"
    return space.axes[0].weights


@pytest.mark.parametrize("label", sorted(_RULES))
@pytest.mark.parametrize("L", [0, 1, 2])
def test_the_tree_binds_the_frames_parseval_space_and_not_the_continuum_head(label: str, L: int) -> None:
    """RE-POSED at CS4c step 6 item 6.2c-ii (ruling R-6.2c-1): the space the
    operator ends and field heads read is the FRAME's Parseval-dressed
    ``basis_space`` — axis-built, the frame its head axis's generator — and
    it is structurally UNEQUAL to the basis's own continuum head (the
    ``from_L(L)`` mint, ``basis.space``), which is this gate's negative
    control: same family, same order, same name, different measure ⟹ a
    different space (the metric-blind ``(name, shape)`` seam Landing A
    leaned on is gone).

    The metric is asserted on the ARRAY per row: on a DIAGONAL-Gram row the
    head axis's measure is the Moore–Penrose reciprocal of the discrete Gram's
    diagonal (zero on dead slots); on a DENSE-Gram row (`[M]` 2 of these 33:
    ``gauss_legendre(2)`` and ``folded_product(2,4)`` at L = 2) the axis
    carries no measure and the matrix pseudo-inverse is POSITIONED on the
    space's derived metric object (item 6.2c-i)."""
    q = _RULES[label]()              # built in the body, never in the parametrize list
    frame = HarmonicFrame.from_galerkin(q.angular_frame(L))
    dressed = frame.basis_space
    # ⛔ RE-KEYED 2026-09-02 (#429): the family to compare against is the one
    # the rule's ORBIT SPACE carries — a 1-D rule's frame binds the flat
    # Legendre head, a sphere rule's the rectangular harmonics.
    continuum = (
        LegendreSpace.from_L(L, "x") if label.startswith("gauss_legendre")
        else SphericalHarmonicSpace.from_L(L)
    )
    assert frame.basis.space == continuum, "the basis's own space IS the continuum mint"
    assert dressed.name == continuum.name and dressed.shape == continuum.shape
    assert dressed != continuum and continuum != dressed, (
        "the dressed head and the continuum head must be two spaces — the metric is the identity"
    )
    assert dressed.axes is not None and len(dressed.axes) == 1
    assert dressed.axes[0].generator is frame, "the frame dresses the head and becomes its generator"
    assert dressed.inner_product_weights is None
    diag = np.diagonal(frame.discrete_gram).reshape(dressed.shape)
    live = diag > 0.0
    if frame.discrete_gram_structure is GramStructure.DENSE:
        assert (label, L) in {("gauss_legendre(2)", 2), ("folded_product(2,4)", 2)}
        assert _head_measure(dressed) is None, "a dense Gram leaves the axis measure-less"
        assert isinstance(dressed.metric, FactoredMetric) and len(dressed.metric.entries) == 1
        assert isinstance(dressed.metric.entries[0][1], DenseMetric)
    else:
        assert dressed.metric is None
        measure = _head_measure(dressed)
        assert measure is not None
        np.testing.assert_allclose(measure[live], 1.0 / diag[live], rtol=1e-15)
        np.testing.assert_array_equal(measure[~live], 0.0)
        # the negative control: the continuum head's measure is the OTHER array
        cont = _head_measure(continuum)
        assert cont is not None and not np.array_equal(cont, measure)


def test_the_operator_ends_carry_the_frames_parseval_space_not_the_continuum_one() -> None:
    """On a real posed composite the scattering factor's ends ARE the frame's
    Parseval-dressed space (structurally equal to the carrier's cached moment
    head — one space, two owners) and NOT the basis's continuum head: `[M]`
    (2026-09-08, the fork's ground) the two metrics differ by exactly
    ``[(2ℓ+1)/4π]²`` per degree on every shipped row, Parseval holds on 33
    of 33 rows under the dressed one and on 0 under the continuum one, and
    the converged flux and the residual trajectory are bit-identical under
    either — the SI increment norm (diagnostics) is the only reader."""
    sn = _slab()
    L = 1                      # the synthetic data carries P0 and P1
    S = ScatteringOperator.from_solver_data(mat_xs=_mat_xs(), space=sn.full_field_space, scattering_order=L)
    ends = S._moment_transfer(skip_l0=False).domain
    frame = sn.quad.angular_frame(L)
    assert ends == frame.basis_space
    assert ends != frame.basis.space
    hub_space = sn.moment_space(L)
    assert isinstance(hub_space, TensorProductSpace)
    hub_head = hub_space.factors[0]
    assert ends == hub_head, "the operator ends and the carrier's cached moment head are one space"
    measure = _head_measure(ends)
    assert measure is not None
    # the slab's Legendre frame is DIAGONAL: the Parseval measure is the
    # reciprocal of the discrete Gram's diagonal — (2l+1)/2 on a GL rule
    # exact through degree L, i.e. NOT the continuum 4π/(2l+1)
    np.testing.assert_allclose(measure, 1.0 / np.diagonal(frame.discrete_gram), rtol=1e-12)
    cont = _head_measure(frame.basis.space)
    assert cont is not None
    np.testing.assert_array_equal(np.asarray(cont).reshape(-1), 4.0 * np.pi / (2.0 * np.arange(L + 1) + 1.0))
    assert not np.array_equal(measure, cont)


# ═══════════════════════════════════════════════════════════════════════
# A3 — the DOOR gate
# ═══════════════════════════════════════════════════════════════════════


def test_the_door_asks_for_a_truncation_order_not_for_one_class() -> None:
    """An indicator trial (no truncation order) is refused at BOTH doors
    with a message naming the surface; a foreign truncated basis — which
    the old ``isinstance(basis, SphericalHarmonicBasis)`` door refused —
    is admitted, and the frame reads its order."""
    indicator = IndicatorBasis((np.array([0.0, 1.0, 2.0]),), RealSpace(1))
    measure = DiscreteMeasure(
        nodes=np.array([0.5, 1.5]), weights=np.ones(2), support=RealSpace(1),
    )
    assert not isinstance(indicator, TruncatedBasis)
    with pytest.raises(TypeError, match="truncation order"):
        HarmonicFrame(indicator, measure)
    with pytest.raises(TypeError, match="truncation order"):
        HarmonicFrame.from_galerkin(GalerkinFrame(indicator, measure))

    # ⚠ The foreign basis is built as a twin of the RULE'S OWN basis
    # (2026-09-02, #429): the frame's G0 arrow refuses a basis on
    # :math:`S^2` bound to a 1-D rule's :math:`S^2/O(2)_x`, so a mutant
    # hard-coding the harmonics would red for the WRONG reason and this gate
    # would stop measuring the door.
    rule = Quadrature.gauss_legendre(4)
    foreign = _ForeignTruncatedBasis.like(rule.angular_frame(1).basis)
    assert isinstance(foreign, TruncatedBasis)
    assert not isinstance(foreign, SphericalHarmonicBasis)
    frame = HarmonicFrame(foreign, rule.measure)
    assert frame.truncation_order == 1
    upgraded = HarmonicFrame.from_galerkin(GalerkinFrame(foreign, rule.measure))
    assert upgraded.truncation_order == 1

    # …and the same twin on a SPHERE rule, so the door gate is not keyed to
    # one family either (the foreign basis there wraps the harmonics).
    sphere = Quadrature.level_symmetric(4)
    foreign_sphere = _ForeignTruncatedBasis.like(sphere.angular_frame(1).basis)
    assert HarmonicFrame(foreign_sphere, sphere.measure).truncation_order == 1
