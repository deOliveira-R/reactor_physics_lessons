r"""The :class:`FrameBase` hierarchy — a discrete frame binding a :class:`Basis`
to a measure, with the **projection discipline carried by the type**.

A *frame* (harmonic-analysis frame theory) is a ``(basis, measure)`` pairing that
emits the operational analysis/reconstruction pair the rest of the algebra consumes:

* **analysis** :math:`M = T` — sampled values → coefficients (``frame.analysis``);
* **reconstruction** :math:`R` — coefficients → values, the canonical-dual synthesis
  (``frame.reconstruction``).

The naked synthesis :math:`T^* = S_0` is the shared :class:`Basis` primitive
(:meth:`~orpheus.numerics.basis.base.Basis.synthesize`) the weighted faces are each
one diagonal away from; it lives one level below the faces.

The pairing fixes BOTH spaces:

* the **basis** is the synthesis (**trial**) side — fixes the codomain
  (``frame.basis_space`` ``= basis.space`` dressed with the frame's **Parseval
  metric**, the inverse of the discrete trial Gram — a property of the
  *pairing*, not of the basis alone; see :attr:`FrameBase.basis_space`);
* the **measure** fixes the domain — ``frame.measure_space``, the nodal space + the
  quadrature weights.

So ``coefficient_space`` is never a third parameter: it is derived from the basis. The
hierarchy is **layout-agnostic** — the index layout (the :math:`(\ell, m)` axes for
spherical harmonics; a flat cell axis for the indicator basis) lives entirely in the
basis, which owns the weighted contractions; the frame caches the table ONCE
(``frame.table``) and the faces delegate.

Discipline IS a type — the Petrov-Galerkin base / Galerkin specialisation
====================================================================

Every Galerkin-style discretisation factors as an analysis/reconstruction pair
:math:`(M, R)`. The **discipline** — whether the *test* functions equal the *trial*
functions — is a genuine *kind of object*, not a flag, so it is carried by the **type**
(GitHub #268; this REVERSES the earlier ``projection.py`` discipline-ABC design, where
the discipline was a marker mixin on the operator role):

.. code-block:: text

   FrameBase                 abstract; the discipline-FREE mechanics
   │                         (table, spaces, reconstruction do NOT depend on the test side)
   └─ PetrovGalerkinFrame    explicit TEST basis (test ≠ trial in general); M = ⟨test, ·⟩_W
      └─ GalerkinFrame       test IS trial — STRENGTHENS the promise to Π* = R (a
                             symmetric, here-diagonal Gram). The angular spherical-
                             harmonic projection is the canonical pure-Galerkin frame.

``GalerkinFrame`` *is-a* ``PetrovGalerkinFrame`` with ``test is trial`` — Liskov-correct,
strengthening (never weakening) the base promises. A genuine ``test ≠ trial`` instance —
flux-weighted spatial homogenisation, spectrum-weighted energy condensation — preserves
a **bilinear** functional :math:`\langle\varphi^*, \Sigma\varphi\rangle` and so cannot be
posed as a Galerkin projection without *folding the solution into the metric*; that fold
is legitimate only for forward-flux, reaction-rate-only reduction and breaks under the
eigenvalue-consistent (adjoint-weighted) homogenisation reactor physics requires. Hence
the test side is a first-class **basis** (the test *space*), NEVER a weight smuggled onto
the measure: **the measure carries the axis + the fixed** :math:`L^2` **metric, never the
discipline**; the solution-weighting (:math:`\varphi`, :math:`\varphi^*`) lives on the
test side = the frame TYPE.

The trial side owns reconstruction; the test side owns analysis
---------------------------------------------------------------

``reconstruction`` (:math:`R`) synthesises a fine field from coefficients — it is purely
**trial**-side (``basis.reconstruct``), identical across disciplines. ``analysis``
(:math:`M`) measures a field against the **test** functions — :math:`(M f)_k =
\sum_n w_n\, \chi_k(x_n)\, f(x_n)` for test functions :math:`\chi_k` — so it reads the
test basis tabulated at the nodes (``frame.test_table``). For a :class:`GalerkinFrame`
``test is trial``, ``test_table is table``, and the analysis is bit-identical to the
single-discipline frame this hierarchy replaced (the 0-ULP scattering-kernel canary).

Iso vs non-iso (a capability, not a separate path)
==================================================

The frame is the single mechanism for ALL choice-dependent change-of-basis (GitHub
#263): whether the analysis/reconstruction are mutually inverse (``R∘M = I`` — an
invertible Vandermonde, e.g. nodal-DG; the analysis face would become invertible)
or band-limiting (``R∘M`` = projector ≠ I, ``N > (L+1)²`` — spherical harmonics; a
section/retraction) is a *capability of the given frame*, not a reason for a second
mechanism. The spherical-harmonic frame is the non-iso case.

References
----------

* Grand Report v3 §5.4 / §19 — bases and harmonic projection.
* Christensen, O. (2016). *An Introduction to Frames and Riesz Bases*, 2nd ed. —
  the analysis operator :math:`T`, synthesis operator :math:`T^*`, frame operator
  :math:`S = T^*T`, and canonical dual.
* Brenner, S. C. and Scott, L. R. (2008). *The Mathematical Theory of Finite Element
  Methods*, 3rd ed. Springer. §3.4 — Galerkin vs Petrov-Galerkin (test vs trial space).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import cached_property
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from orpheus.numerics.axis import BasisKind, EnergyAxis
from orpheus.numerics.basis.base import Basis, GramStructure
from orpheus.numerics.basis.indicator_basis import IndicatorBasis
from orpheus.numerics.manifold import IndexSet, ManifoldMap, quotient_onto
from orpheus.numerics.measure import DiscreteMeasure
from orpheus.numerics.metric import DenseMetric
from orpheus.numerics.operator import (
    AxisRetractionOperator,
    AxisSectionOperator,
    LinearOperator,
    NotInvertible,
    OperatorProduct,
)
from orpheus.numerics.projection import AnalysisOperator, ReconstructionOperator
from orpheus.numerics.space import FunctionSpace


__all__ = ["FrameBase", "GalerkinFrame", "PetrovGalerkinFrame"]


#: Relative off-diagonal threshold for the :attr:`FrameBase.discrete_gram_structure`
#: diagonality verdict. The measured separation is wide open: an SH-degree-exact
#: sphere cubature's off-diagonals sit at ~1e-16 of the Cauchy–Schwarz scale
#: :math:`\sqrt{G_{jj}G_{kk}}`, while the slab GL live Gram's sit at 0.93 of
#: it (0.58 of the largest diagonal; `[M]` 2026-08-23) — any threshold in
#: (1e-12, 1e-2) draws the same verdict; 1e-10 leaves two orders of headroom
#: for accumulated roundoff at high mode counts.
_DISCRETE_GRAM_DIAGONALITY_RTOL = 1e-10


def _descent_arrow(basis: Basis, measure: DiscreteMeasure, role: str) -> ManifoldMap:
    r"""G0 — the frame's two halves must name ONE orbit space, up to a quotient map.

    A frame binds functions on ``basis.domain`` to a rule on
    ``measure.support``; that is well-posed iff the functions can be
    evaluated at the rule's nodes, i.e. iff a quotient map
    ``measure.support -> basis.domain`` EXISTS
    (:func:`~orpheus.numerics.manifold.quotient_onto`: equality, the entry's
    own map, or the induced :math:`M/K \to M/H` for :math:`K \subseteq H`).
    The frame's table is the basis pulled back along that arrow. ONE
    predicate (user-ruled 2026-09-02, #429 tracker 2.2): it refuses the Part
    I bug — the full harmonics (``Trivial``) on the slab's :math:`S^2/O(2)_x`
    — and admits the slab's Legendre basis, the fold's σ-even harmonics, the
    full harmonics on a full-sphere rule, AND the Legendre basis on a
    full-sphere rule. `[M]` 2026-09-02, before this gate, every one of the
    21 frames three real solves built passed it; after the forgery's
    retirement the slab-harmonics pairing is exactly the one it refuses.
    """
    arrow = quotient_onto(measure.support, basis.domain)
    if arrow is None:
        spent = getattr(measure, "quotient_group", None)
        has = getattr(basis, "invariance_group", None)
        raise ValueError(
            f"a frame binds {role} functions on {basis.domain.name!r} to a "
            f"rule on {measure.support.name!r}, and no quotient map "
            f"{measure.support.name} -> {basis.domain.name} exists (the rule "
            f"spent {spent.name if spent is not None else None}, the basis "
            f"has {has.name if has is not None else None}). A basis eats "
            f"points of its own orbit space or of a finer one; give the rule "
            f"the basis its orbit space carries "
            f"(Quadrature.angular_frame derives it) — #429, ERR-080."
        )
    return arrow


@dataclass(frozen=True)
class FrameBase(ABC):
    r"""A discrete frame: a :class:`Basis` bound to a :class:`DiscreteMeasure`.

    The abstract base carries the **discipline-free** mechanics — the trial table, the
    two spaces, the reconstruction face, and the analysis face's wiring. The single
    abstract hook is :attr:`test` (the test basis), which the discipline subclasses
    fix: :class:`GalerkinFrame` (``test is trial``) and :class:`PetrovGalerkinFrame`
    (an explicit, generally distinct, test basis).

    Parameters
    ----------
    basis : Basis
        The synthesis (trial) side — fixes the codomain (:attr:`basis_space`).
    measure : DiscreteMeasure
        Fixes the domain (:attr:`measure_space`) and the quadrature weights.
    """

    basis: Basis
    measure: DiscreteMeasure

    def __post_init__(self) -> None:
        # G0 at construction, on BOTH halves (the trial here; the test basis
        # through :attr:`test_descent` on first use, since the discipline
        # subclass binds it) — a frame over an incompatible pairing is
        # unspellable, loudly, early.
        _descent_arrow(self.basis, self.measure, "trial")

    # ── the G0 arrows: what the tables are pulled back along ──────────────
    @cached_property
    def descent(self) -> ManifoldMap:
        r"""The quotient map ``measure.support -> basis.domain`` the trial table is pulled back along (G0)."""
        return _descent_arrow(self.basis, self.measure, "trial")

    @cached_property
    def test_descent(self) -> ManifoldMap:
        r"""The same arrow for the TEST basis (``test is trial`` on a Galerkin frame)."""
        if self.test is self.basis:
            return self.descent
        return _descent_arrow(self.test, self.measure, "test")

    # ── the discipline hook ───────────────────────────────────────────────
    @property
    @abstractmethod
    def test(self) -> Basis:
        r"""The **test** basis — the analysis (measured) side of the frame.

        :class:`GalerkinFrame` returns the trial :attr:`basis` (``test is trial``);
        :class:`PetrovGalerkinFrame` returns its explicit test basis. The analysis
        face reads this basis tabulated at the nodes (:attr:`test_table`).
        """
        ...

    # ── trial side (reconstruction + the Galerkin analysis) ───────────────
    @cached_property
    def table(self) -> NDArray:
        r"""The TRIAL basis tabulated at the measure's nodes — :math:`\Phi(\text{node}, \text{mode})`.

        Evaluated ONCE and cached; the reconstruction face and (for a Galerkin frame)
        the analysis face read this rather than re-tabulating (the L16 perf guard).
        The nodes are pulled back along the G0 arrow first (:attr:`descent`):
        the identity for a basis on the rule's own orbit space, the quotient
        map for a Legendre basis on a full-sphere rule.
        """
        return self.basis.evaluate(self.descent(self.measure.nodes))

    @cached_property
    def discrete_gram(self) -> NDArray:
        r"""The TRIAL basis's discrete Gram over the measure — flattened ``(K, K)``.

        :math:`G_{jk} = \sum_n w_n\,\phi_j(x_n)\,\phi_k(x_n)`, computed from the
        CACHED :attr:`table` (the L16 perf guard — never re-tabulates), with the
        basis's mode layout flattened to one axis (``K =`` the coefficient count,
        row-major — the same order as ``table.reshape(N, K)``). The same
        mathematical object as :meth:`Basis.mass_matrix
        <orpheus.numerics.basis.base.Basis.mass_matrix>` in the basis's own
        layout; that one is the measure-based *diagnostic*, this one is the
        frame's cached *operational* copy — the Parseval-metric source
        (:attr:`basis_space`) and the diagonality verdict
        (:attr:`discrete_gram_structure`) both read it. O(N·K²), once per frame.
        """
        weights = self.measure.weights
        flat = self.table.reshape(weights.shape[0], -1)
        return np.einsum("n,nj,nk->jk", weights, flat, flat)

    @cached_property
    def discrete_gram_structure(self) -> GramStructure:
        r"""The MEASURED diagonality of :attr:`discrete_gram` — DIAGONAL or DENSE.

        The measured counterpart of the trial's DECLARED
        :attr:`~orpheus.numerics.basis.base.Basis.gram_structure`, and a genuinely
        different fact: the declaration states which structure the *cross* Gram
        :math:`MR` has for :meth:`project`'s row-sum probe; this verdict states
        whether the *trial* Gram :math:`G` is diagonal **on this measure** — the
        precondition for the Parseval metric (:attr:`basis_space`). The two can
        disagree: the SH basis declares DIAGONAL (continuum-orthogonal) yet
        measures DENSE on a slab GL measure (total weight 2, live off-diagonals
        at 0.93 of the Cauchy–Schwarz scale — `[M]` 2026-08-23; discovery record
        ``scratch/probe_f1_parseval_slab.py``),
        and an :class:`~orpheus.numerics.basis.overlap_basis.OverlapBasis`
        declares PARTITION_OF_UNITY while its trial Gram measures DENSE.

        DIAGONAL iff no diagonal entry is negative (a negative-weight quadrature
        can make :math:`G` indefinite, and an indefinite form is not a metric)
        AND every live off-diagonal is below
        ``_DISCRETE_GRAM_DIAGONALITY_RTOL`` of the Cauchy–Schwarz scale
        :math:`\sqrt{G_{jj}G_{kk}}`. Structurally dead slots (:math:`G_{kk}=0` —
        layout padding, σ-odd folded columns, empty indicator regions) are
        exempt, but any coupling INTO a dead slot is DENSE.
        """
        gram = self.discrete_gram
        diag = np.diagonal(gram)
        if np.any(diag < 0.0):
            return GramStructure.DENSE
        off = np.abs(gram - np.diag(diag))
        scale = np.sqrt(np.outer(diag, diag))
        live = scale > 0.0
        if np.any(off[~live] > 0.0):
            return GramStructure.DENSE
        if np.any(live) and (
            float(np.max(off[live] / scale[live]))
            > _DISCRETE_GRAM_DIAGONALITY_RTOL
        ):
            return GramStructure.DENSE
        return GramStructure.DIAGONAL

    @cached_property
    def basis_space(self) -> FunctionSpace:
        r"""The codomain — the trial coefficient space, carrying the PARSEVAL metric.

        ``basis.space`` dressed with the inverse of the discrete trial Gram's
        diagonal (zero on dead slots). The analysis face's outputs are the
        COVARIANT moments :math:`\varphi = G c` of a band-limited field
        :math:`S_0 c`, so the inner product that makes analysis an isometry
        onto its image — Parseval, :math:`\|\varphi\|_{G^{-1}} = \|S_0 c\|_W`
        — is :math:`G^{-1}`, the **inverse of the discrete Gram**. That is a
        property of the *pairing* (basis ⊗ measure), so the frame owns it: the
        basis's own ``space`` keeps the continuum Gram (for the SH basis
        :math:`4\pi/(2\ell+1)` — the cross-Gram vocabulary of
        :attr:`gram`/:meth:`project`), and the frame REPLACES the metric with
        the measured inverse. With this metric the faces' ``.H`` is the
        physical Hilbert adjoint: for the SH frame :math:`M^{*} = R/W` and
        :math:`R^{*} = W\,M` with :math:`W = \sum_n w_n` — the frame square
        closes with one scalar (`[M]` closure ≤1e-15 across every shipped
        sphere family; the pre-F-0 stored continuum metric was the WRONG
        side — off the physical adjoint by exactly :math:`(4\pi/(2\ell+1))^2`
        per ℓ. Live witnesses: ``tests/numerics/test_frame.py``'s
        ``test_parseval_*`` suite, whose negative leg re-installs the
        pre-repair metric in-process; discovery record
        ``scratch/probe_f1_parseval.py`` — note the probe reads the frame's
        NOW-DRESSED space, so its "stored" row prints 1.000 post-repair).

        For a frame whose discrete Gram is NOT diagonal
        (:attr:`discrete_gram_structure` DENSE — e.g. the slab GL measure,
        where NO diagonal metric satisfies Parseval), the metric is the
        matrix pseudo-inverse :math:`G^{+}` installed as a
        :class:`~orpheus.numerics.metric.DenseMetric` (campaign 1, P7):
        Parseval is then a THEOREM for any Gram, singular or not
        (:math:`\|Mc\|^2_{G^{+}} = c^{\mathsf T} G G^{+} G c =
        \|S_0 c\|^2_W`), and the faces' ``.H`` is the physical Hilbert
        adjoint on every frame, not only the diagonal ones. `[M]`
        2026-08-30, the wrong-metric discriminator on the slab GL8/L=2
        frame: the dense dressing reads the Parseval ratio
        1.000000000000 where the best diagonal candidate ``1/diag(G)``
        reads 1.806 and the undressed continuum metric reads 25.53 — a
        diagonal metric there is not merely unavailable but provably
        insufficient. (Until P7 the DENSE arm returned the space
        UNDRESSED — *"Parseval is unavailable"* was the recorded F-0
        limitation, with the matrix-metric home deferred to the CS4c
        Riesz-leg machinery; P7 landed the metric object CS4c's legs will
        wrap, and the refusal era's record lives in this file's history
        and the error catalog's ERR-039 entry.)

        Equality is untouched either way: ``(name, shape)`` identity is
        metric-blind, so ``basis_space == basis.space`` still holds and no
        consumer's ``==`` moves.
        """
        space = self.basis.space
        if self.discrete_gram_structure is not GramStructure.DIAGONAL:
            # DENSE verdict (P7): the Parseval metric exists — it is just
            # not diagonal. Install the matrix pseudo-inverse of the
            # measured Gram (DenseMetric.inverse_of keeps the exact
            # symmetrized Gram as the inverse face) and STRIP the basis's
            # continuum weights — the dressing REPLACES the metric on
            # this arm exactly as the diagonal arm overwrites it, and a
            # space carrying both sources is the illegal state the
            # exclusivity guard refuses.
            return replace(
                space,
                inner_product_weights=None,
                metric=DenseMetric.inverse_of(self.discrete_gram),
            )
        diag = np.diagonal(self.discrete_gram).reshape(space.shape)
        live = diag > 0.0
        inverse = np.where(live, 1.0 / np.where(live, diag, 1.0), 0.0)
        return replace(space, inner_product_weights=inverse)

    # ── test side (the analysis face) ─────────────────────────────────────
    @cached_property
    def test_table(self) -> NDArray:
        r"""The TEST basis tabulated at the measure's nodes (the analysis contraction).

        Defaults to the test basis evaluated at the nodes; :class:`GalerkinFrame`
        overrides it to *reuse* :attr:`table` (``test is trial`` ⟹ the same array,
        no re-evaluation, 0-ULP-identical analysis).
        """
        return self.test.evaluate(self.test_descent(self.measure.nodes))

    @cached_property
    def test_space(self) -> FunctionSpace:
        r"""The analysis codomain — the test basis's coefficient space (``= test.space``).

        :class:`GalerkinFrame` overrides it to *reuse* :attr:`basis_space` (``test is
        trial`` ⟹ the same cached space object, preserving the analysis-codomain ``is``
        identity of the single-discipline frame this hierarchy replaced).
        """
        return self.test.space

    # ── the measure (domain) side ─────────────────────────────────────────
    @cached_property
    def measure_space(self) -> FunctionSpace:
        r"""The domain — the measure's induced discrete-:math:`L^2` space.

        Read straight off :attr:`DiscreteMeasure.space` (per-node values with the
        quadrature weights as the metric): the measure OWNS its domain space,
        symmetric with the basis owning its codomain — neither is fabricated here.
        """
        return self.measure.space

    # ── the two operator faces ────────────────────────────────────────────
    @cached_property
    def analysis(self) -> "_FrameAnalysis":
        r"""The analysis face :math:`M = T` (``measure_space → test_space``)."""
        return _FrameAnalysis(self)

    @cached_property
    def reconstruction(self) -> "_FrameReconstruction":
        r"""The reconstruction face :math:`R` (``basis_space → measure_space``)."""
        return _FrameReconstruction(self)

    # ── composed operators (the "define Frame, compose, done" production path) ──
    def conjugate(self, operator: LinearOperator, /) -> OperatorProduct:
        r"""Frame-conjugate a coefficient-space operator: :math:`R \circ A \circ M`.

        THE production composition for a method whose action is "project to
        coefficients, act there, reconstruct" — e.g. SN anisotropic scattering
        :math:`S_{\ell\ge 1} = R\,\Lambda\,M` (the per-ordinate redistribution). Returns
        the typed :class:`~orpheus.numerics.operator.OperatorProduct`
        ``OperatorProduct(R, OperatorProduct(A, M))``, whose
        :meth:`~orpheus.numerics.operator.OperatorProduct.apply` is ``R(A(M·x))`` — the
        SAME numpy chain a hand-rolled ``reconstruction.apply(A.apply(analysis.apply(x)))``
        runs, now ONE named operator (Cardinal Rule 2: the composition IS the production
        path, not a parallel "semantic" reading of it).

        **The double-category 2-cell — and, for an eigenbasis frame, the spectral
        theorem.** :meth:`conjugate` is the 2-cell of the (Representation × Role)
        carrier grid: a vertical Role-morphism ``A`` (e.g. scattering's :math:`\Lambda`,
        the diagonal :math:`\Sigma_{s,\ell}` multiply) conjugated by the horizontal
        Representation-adjoint pair :math:`(M, R)`. When the Frame is an operator's
        EIGENBASIS — the SH angular frame is the scattering kernel's, by Funk–Hecke —
        :math:`R\circ\Lambda\circ M` IS the spectral theorem :math:`A = U\Sigma U^{*}`
        written out (:math:`M=U` analysis into the eigenbasis, :math:`\Lambda=\Sigma`
        the spectrum :math:`\Sigma_{s,\ell}`, :math:`R=U^{*}` synthesis). The frame is
        then OWNED by that operator (scattering owns its angular frame), not by the
        phase space — see :ref:`frame-eigenbasis-ownership`. The 0-ULP
        ``test_scattering_kernel_crosscheck`` is this 2-cell's interchange-law witness.

        ``operator`` must compose between the faces — its ``domain`` is the analysis
        codomain (:attr:`test_space`) and its ``codomain`` the reconstruction domain
        (:attr:`basis_space`); the :class:`OperatorProduct` space-compatibility guard
        enforces it (an endomorphism on the coefficient space when ``test == trial``).
        """
        return OperatorProduct(
            self.reconstruction, OperatorProduct(operator, self.analysis),
        )

    def reconstruct_after(self, operator: LinearOperator, /) -> LinearOperator:
        r"""Reconstruct after a coefficient-space operator: :math:`R \circ A`.

        The :meth:`conjugate` sub-operator for inputs ALREADY in coefficient space (the
        analysis :math:`M` already applied) — e.g. the angular-windowed SN moment
        iterate, whose bulk IS :math:`M\psi`, so only :math:`R\,\Lambda` remains.
        Returns ``OperatorProduct(R, A)`` (``apply`` = ``R(A·c)``). Wiring a windowed
        consumer to :meth:`conjugate` instead would erroneously re-apply :math:`M` (a
        double-projection).
        """
        return OperatorProduct(self.reconstruction, operator)

    # ── the coefficient-extraction verb (homogenise / condense) ──────────
    @cached_property
    def gram(self) -> FunctionSpace:
        r"""The coefficient space carrying the frame-Gram diagonal :math:`G_R = \langle\chi_R, \phi_R\rangle_W`.

        :meth:`project` normalises by the cross Gram :math:`G_{kj} = \langle\chi_k,
        \phi_j\rangle_W = (M R)_{kj}` of the analysis :math:`M` and reconstruction
        :math:`R`. This property takes a **single** ``analysis ∘ reconstruction``
        probe of the all-ones coefficient vector — the **row sum** of :math:`M R` —
        and installs it as the coefficient space's metric, so :meth:`project`'s
        normalisation is the reciprocal
        :meth:`~orpheus.numerics.space.FunctionSpace.apply_inverse_metric` (whose
        Moore–Penrose pseudo-inverse zeroes empty / zero-weight regions for free).
        The diagonal acquires the test weight's trailing (group, …) shape from the
        analysis face.

        The row-sum probe is the weight :meth:`project` needs under EITHER of two
        sufficient conditions — distinguish them, because the second does NOT imply a
        diagonal Gram:

        * **disjoint-support trial** (orthogonal harmonics, nested cell / group
          indicators): :math:`M R` is **diagonal**, off-diagonals structurally zero,
          so the row sum simply IS the diagonal :math:`(M R\,\mathbf 1)_R = \int_R
          w\,\mathrm{d}V` (:math:`w` the test weight).
        * **partition-of-unity trial** (the fractional
          :class:`~orpheus.numerics.basis.OverlapBasis`: a straddling fine cell lands
          in ≥2 coarse columns, so :math:`M R` is **NOT** diagonal): here the rows of
          the membership table sum to 1, i.e. :math:`R\,\mathbf 1 = \mathbf 1`, so the
          probe collapses to :math:`(M\,\mathbf 1)_R = \sum_i T_{iR}\,w_i = \Phi_R` —
          exactly the per-region weight, *even though the Gram is non-diagonal*. The
          conservative re-binning is correct because of this PoU collapse, NOT because
          the off-diagonals vanish.

        **A trial that is neither disjoint NOR a partition of unity** (a tapered
        weight, a higher-rank GEC moment — #275) makes the row-sum probe ≠ the true
        projection :math:`(M R)^{-1} M f`; such a basis needs the dense cross-Gram solve
        (the unbuilt least-squares seam), not this diagonal probe. That precondition is
        now carried by the **type**: the trial declares its
        :attr:`~orpheus.numerics.basis.base.Basis.gram_structure`, and this property
        **refuses** a :attr:`~orpheus.numerics.basis.base.GramStructure.DENSE` trial
        (raising :class:`~orpheus.numerics.operator.NotInvertible` — the projection
        normalisation :math:`G^{-1}` is not realizable as posed) rather than
        return a silently-wrong probe.
        """
        if self.basis.gram_structure is GramStructure.DENSE:
            raise NotInvertible(
                f"FrameBase.project / .gram needs a row-sum-collapsible trial Gram, but "
                f"the trial {type(self.basis).__name__} declares GramStructure.DENSE: "
                f"its cross Gram MR is neither diagonal nor a partition of unity, so the "
                f"row-sum probe is NOT the projection normalisation G⁻¹. The dense "
                f"(MR)⁻¹M least-squares solve is not built (GitHub #275 — higher-rank "
                f"GEC moments / tapered weights); build it before projecting through "
                f"this basis."
            )
        ones = np.ones(self.basis_space.shape)
        diagonal = self.analysis.apply(self.reconstruction.apply(ones))
        # The probe is a CROSS-Gram object: its row-sum diagonal IS the
        # projection normalisation, and it must never inherit the test
        # space's own PARSEVAL dressing — a dense-dressed test_space
        # would otherwise hand this replace() two metric sources, and
        # the pre-P7 spelling silently applied the dense matrix inside
        # project() instead of the probe reciprocal ([M] 2026-08-30:
        # rel 1.625 on the overlap frame's [8/3, 16/3]). Strip the
        # object, install the diagonal.
        return replace(
            self.test_space, inner_product_weights=diagonal, metric=None
        )

    def project(self, field: NDArray, /) -> NDArray:
        r"""Extract coefficients: :math:`G^{-1} M f` — the homogenise / condense verb.

        The Petrov-Galerkin coefficient extraction: analyse the field against the
        test functions (:math:`(M f)_k = \langle\chi_k, f\rangle_W`), then normalise
        by the cross :attr:`gram` :math:`G`. For flux-weighted spatial homogenisation
        (``test`` :math:`= \varphi\cdot\mathbf 1_R`, ``trial`` :math:`= \mathbf 1_R`)
        this is the rate-preserving effective cross section :math:`\Sigma_R = \int_R
        \varphi\Sigma\,\mathrm{d}V / \int_R\varphi\,\mathrm{d}V`; for a
        :class:`GalerkinFrame` (``test = trial``) it is the orthogonal projection onto
        the coarse space. The normalisation is a reciprocal (the diagonal :attr:`gram`
        probe) for both the disjoint-indicator / orthogonal-harmonic consumers (Gram
        genuinely diagonal) AND the partition-of-unity
        :class:`~orpheus.numerics.basis.OverlapBasis` (non-diagonal Gram, but the
        row-sum probe is still the right per-region weight — see :attr:`gram`); the
        dense solve is the (unbuilt) least-squares seam only. Trailing (group, …)
        axes ride the analysis and broadcast against the diagonal Gram (so a vector
        channel divides by :math:`\Phi_{R,g}` and a ``[g_from, g_to]`` matrix channel
        by its source-group :math:`\Phi_{R,g_{\mathrm{from}}}`).
        """
        return self.gram.apply_inverse_metric(self.analysis.apply(field))


@dataclass(frozen=True)
class PetrovGalerkinFrame(FrameBase):
    r"""A frame with an explicit **test** basis (the Petrov-Galerkin discipline).

    The analysis measures against test functions :math:`\chi_k` that need NOT equal the
    trial functions :math:`\phi_k`, so the coefficient extraction :math:`G^{-1} M` (the
    ``project`` verb, built in a later phase) uses the *cross* Gram
    :math:`G_{kj} = \langle \chi_k, \phi_j\rangle` and the Hilbert adjoint
    :math:`M^* \ne R`.

    Flux-weighted spatial homogenisation and spectrum-weighted energy condensation are
    the headline consumers: the test basis is the trial cell/group indicator weighted
    by the solution (:math:`\varphi\cdot\mathbf 1_R` forward, or the bilinear PAIR
    weight :math:`\varphi^*\!\odot\varphi\cdot\mathbf 1_R` for the eigenvalue-consistent
    adjoint-weighted collapse — the product, NOT a bare :math:`\varphi^*` swap; theorem
    T1 of :mod:`orpheus.derivations.common.homogenization`) — a genuinely different
    basis, NOT a metric on the measure.

    The test basis is **required** — a Petrov-Galerkin frame is *defined* by carrying an
    explicit test space, so there is no implicit "``None`` means trial" default. Passing
    the trial :attr:`basis` itself as the test basis is the legal **degenerate**: the
    frame then behaves like a :class:`GalerkinFrame` (a useful "PG reduces to Galerkin"
    equivalence test) but does NOT advertise the strengthened :math:`M^* = R` promise.
    To get ``test = trial`` *without* naming a test basis, construct a
    :class:`GalerkinFrame` instead.

    Parameters
    ----------
    test_basis : Basis
        The explicit test basis. Generally distinct from the trial ``basis``; passing
        ``basis`` itself is the Galerkin degenerate.
    """

    test_basis: Basis

    @property
    def test(self) -> Basis:
        return self.test_basis


@dataclass(frozen=True, init=False)
class GalerkinFrame(PetrovGalerkinFrame):
    r"""The Galerkin specialisation: the test basis IS the trial basis.

    A :class:`GalerkinFrame` is constructed from ``(basis, measure)`` alone — it binds
    the inherited (required) :attr:`test_basis` to the trial ``basis``, so
    ``test is trial``. That coincidence strengthens the base Petrov-Galerkin promises:
    the Gram is symmetric (here diagonal — SH-orthogonal / disjoint indicators), so
    :math:`M^* = R` up to the basis's dual factor and the coefficient extraction is a
    reciprocal, not a solve. The angular spherical-harmonic projection
    (``quadrature.angular_frame(L)``) is the canonical pure-Galerkin frame; the in-sweep
    moment accumulation AND the §5.6 scattering kernel share THIS object.

    It is a genuine subtype of :class:`PetrovGalerkinFrame` (Liskov: a Galerkin frame
    IS a Petrov-Galerkin frame whose test basis equals its trial basis). The constructor
    takes **no** ``test_basis`` argument — that ``test ≠ trial`` freedom is exactly what
    distinguishes a :class:`PetrovGalerkinFrame`, so a distinct test basis is forbidden
    here by the constructor signature itself (it is spelled by building one of those).
    """

    def __init__(self, basis: Basis, measure: DiscreteMeasure) -> None:
        # test = trial: bind the (required) test_basis to the trial basis. A frozen
        # dataclass forbids attribute assignment, so set the three fields directly.
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "measure", measure)
        object.__setattr__(self, "test_basis", basis)
        _descent_arrow(basis, measure, "trial")   # G0 (the dataclass path runs __post_init__; this ctor must too)

    @cached_property
    def test_table(self) -> NDArray:
        # test is trial → reuse the trial table (same array; 0-ULP-identical analysis).
        return self.table

    @cached_property
    def test_space(self) -> FunctionSpace:
        # test is trial → reuse the cached trial space (preserves the codomain `is`).
        return self.basis_space


@dataclass(frozen=True)
class _FrameAnalysis(AnalysisOperator):
    r"""The analysis face :math:`M = T`: ``measure_space → test_space``.

    A frame-backed :class:`AnalysisOperator` view; the math lives on the TEST basis
    (:meth:`Basis.analyze` / :meth:`Basis.analyze_transpose`) tabulated at the nodes
    (``frame.test_table``). Carries the swapped spaces and a working
    ``apply_transpose`` (``is_adjointable=True``), so the metric-aware
    ``AdjointOperator`` gives ``.H`` (the W-weighted Hilbert adjoint)
    for free. For a :class:`GalerkinFrame` the test basis is the trial basis, so this is
    the :math:`Y^* W` projection bit-identical to the single-discipline frame.
    """

    frame: FrameBase

    @property
    def domain(self) -> FunctionSpace:
        return self.frame.measure_space

    @property
    def codomain(self) -> FunctionSpace:
        return self.frame.test_space

    def apply(self, values: NDArray, /) -> NDArray:
        return self.frame.test.analyze(
            values, self.frame.test_table, self.frame.measure.weights,
        )

    def apply_transpose(self, coefficients: NDArray, /) -> NDArray:
        return self.frame.test.analyze_transpose(
            coefficients, self.frame.test_table, self.frame.measure.weights,
        )


@dataclass(frozen=True)
class _FrameReconstruction(ReconstructionOperator):
    r"""The reconstruction face :math:`R`: ``basis_space → measure_space``.

    A frame-backed :class:`ReconstructionOperator` view delegating to the TRIAL basis's
    :meth:`Basis.reconstruct` (the canonical-dual synthesis) and its representation
    transpose :meth:`Basis.reconstruct_transpose` — reconstruction is purely trial-side,
    identical across disciplines. Carries the swapped spaces and a working
    ``apply_transpose`` (``is_adjointable=True``), so the metric-aware
    ``AdjointOperator`` gives ``R.H`` for free — symmetric with the
    analysis face.
    """

    frame: FrameBase

    @property
    def is_adjointable(self) -> bool:
        # The frame's reconstruction FACE adds apply_transpose on top of
        # the bare ReconstructionOperator role (apply-only), so ``R.H``
        # is free here. The override lives on the face.
        return True

    @property
    def domain(self) -> FunctionSpace:
        return self.frame.basis_space

    @property
    def codomain(self) -> FunctionSpace:
        return self.frame.measure_space

    def apply(self, coefficients: NDArray, /) -> NDArray:
        return self.frame.basis.reconstruct(coefficients, self.frame.table)

    def apply_transpose(self, values: NDArray, /) -> NDArray:
        return self.frame.basis.reconstruct_transpose(values, self.frame.table)


# ── The axis collapse pair — the rank-one frame's induced output (S6.0b) ──


class _AxisCollapsePair(NamedTuple):
    """One axis's minted collapse pair (``section`` is ``None`` iff Σw = 0)."""

    retraction: AxisRetractionOperator
    section: AxisSectionOperator | None


def _collapse_pair(space: FunctionSpace, axis_label: str) -> _AxisCollapsePair:
    r"""Mint the axis collapse pair — the single-region indicator frame's output.

    THE stage-2 generator discipline, applied at rank one (user, ruled
    2026-08-24): *"A stage-2 generator induces structure on both the space
    and the operator, and the two inductions must be minted together, at
    one site. … Forgetting = retaining the induced parts; accessors are
    provenance."* Here the generator is the frame over the axis's index
    set — the single-region :class:`IndicatorBasis` covering synthetic
    index nodes :math:`\{0, \ldots, n-1\}`, bound to the axis's measure
    in a :class:`GalerkinFrame` — built eagerly HERE, read for its induced
    data, and DISCARDED (the forgetful map: neither the axis nor the
    operators keep the generator, and a frame FACE is a view holding
    ``frame:``, so no face is retained either). What is read off it:

    * the kernel weights — the frame measure's diagonal (the analysis
      face's content IS the retraction's contraction);
    * the section's divisor — the 1×1 :attr:`FrameBase.discrete_gram`
      entry :math:`\Sigma w`, i.e. the rank-one **Parseval metric**
      (F-0's inverse-discrete-Gram theorem at :math:`K = 1`):
      :math:`E = R_{\text{frame}} \circ G^{-1}`. A zero entry (a signed
      measure summing to zero) means the Gram is singular — the frame has
      no canonical dual, so no section exists: that arm is ``None`` and
      :meth:`FunctionSpace.section` refuses at access.

    Both operators are constructed together at this one site (the
    two-inductions clause); the tightness gate
    (``tests/numerics/test_axis_marginal.py``) pins the minted kernels
    against the literal frame's face contents, and the gram-derivation
    gate pins the divisor against :attr:`FrameBase.discrete_gram`.

    Admission (typed refusals, all at this mint — the public path is
    :meth:`FunctionSpace.retraction` / :meth:`FunctionSpace.section`,
    which memoize this mint per axis label):

    * the space must be **axis-built** (``axes is not None``) — an
      axes-less space (a hand-named legacy space, or a product with an
      axes-less head) has no named factors to marginalise over;
    * ``axis_label`` must name **exactly one** axis;
    * the axis must be **NODAL**: a "marginal" over a MODAL axis would
      contract expansion COEFFICIENTS with the basis mass, which is not
      an integral of the represented function — the modal average is the
      coefficient at the average slot, not a weighted sum;
    * the axis must not be a typed :class:`~orpheus.numerics.axis.EnergyAxis`
      — collapse doctrine clause 2 (partition-integration of an
      :math:`L^1` class): the energy axis PERSISTS at its one-cell
      member, because the one-group limit keeps its edges and spectrum
      (:math:`\langle\bar\sigma, \phi\rangle` consumes the
      partition). The clause-2 collapse already ships as condensation
      (``EnergyGrid.overlap_to``, the Petrov-Galerkin condensation
      frames), and a drop-form marginal here would twin it (Cardinal
      Rule 2). Untyped generic axes stay admitted whatever their label —
      the clause gate reads the TYPE, never the label string (stringly
      dispatch rejected); it becomes fully structural axis-family
      polymorphism when CS2's typed axes land;
    * a single-axis space is refused — its marginal would be a bare
      scalar, which is not a :class:`FunctionSpace` (contract with the
      space's inner product instead).
    """
    # Arms 1-2 (axis-built; exactly-one label) discharge in the shared
    # resolver FunctionSpace._axis_index (un-weld arc S-1) — one home for
    # the by-label refusal vocabulary.
    k = space._axis_index(axis_label)
    axes = space.axes
    assert axes is not None  # narrowing only: _axis_index refused the None case
    axis = axes[k]
    if axis.kind is BasisKind.MODAL:
        raise TypeError(
            f"collapse pair: axis {axis_label!r} is MODAL — contracting "
            f"expansion coefficients with the basis mass is not an "
            f"integral of the represented function. The modal average is "
            f"the coefficient at the average slot (slice it), not a "
            f"weighted sum."
        )
    if isinstance(axis, EnergyAxis):
        raise TypeError(
            f"collapse pair: axis {axis_label!r} is a typed EnergyAxis, "
            f"which PERSISTS at its one-cell member (collapse doctrine "
            f"clause 2 — the one-group limit keeps its edges and "
            f"spectrum, because ⟨σ̄,φ⟩ consumes the partition). The "
            f"energy collapse is condensation: use EnergyGrid.overlap_to "
            f"/ the Petrov-Galerkin condensation frame, not a drop-form "
            f"marginal."
        )
    if len(axes) == 1:
        raise ValueError(
            f"collapse pair: {space!r} has only the {axis_label!r} axis "
            f"— its marginal would be a bare scalar, which is not a "
            f"FunctionSpace. Contract with the space's inner product "
            f"instead."
        )

    # The ndarray dims this axis occupies: axes map to dims by
    # cumulative rank (an axis's shape may span several dims).
    start = sum(len(ax.shape) for ax in axes[:k])
    dims = tuple(range(start, start + len(axis.shape)))
    marginal_space = FunctionSpace.of_axes(
        *(ax for i, ax in enumerate(axes) if i != k)
    )

    # The generator, eagerly: the axis's measure on synthetic index
    # nodes, under the single-region indicator covering them all.
    w = axis.weights
    flat_weights = (
        np.ones(int(np.prod(axis.shape)))
        if w is None
        else np.asarray(w, dtype=float).ravel()
    )
    n = int(flat_weights.shape[0])
    # ⭐ One frame, one manifold — and now literally one expression. Both
    # halves used to say it separately (the basis as an ``IndexSet``, the
    # measure as ``f"index({axis_label})"``), which is two spellings of one
    # fact and exactly what a frame's two halves must not have.
    points = IndexSet(label=axis_label, n=n)
    frame = GalerkinFrame(
        basis=IndicatorBasis(
            edges_per_axis=(np.array([-0.5, n - 0.5]),),
            partition_of=points,
        ),
        measure=DiscreteMeasure(
            nodes=np.arange(n, dtype=float),
            weights=flat_weights,
            support=points,
        ),
    )

    # The induced reads — everything the operators retain comes off the
    # frame; the frame goes out of scope at return (forgetful).
    kernel_weights = frame.measure.weights
    total_weight = float(frame.discrete_gram[0, 0])

    retraction = AxisRetractionOperator(
        full_space=space,
        marginal_space=marginal_space,
        axis_shape=axis.shape,
        dims=dims,
        flat_weights=kernel_weights,
    )
    section = (
        None
        if total_weight == 0.0
        else AxisSectionOperator(
            full_space=space,
            marginal_space=marginal_space,
            axis_shape=axis.shape,
            dims=dims,
            flat_weights=kernel_weights,
            total_weight=total_weight,
        )
    )
    return _AxisCollapsePair(retraction, section)
