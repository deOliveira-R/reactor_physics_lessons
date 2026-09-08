r"""Hilbert metrics — a space's inner-product structure as an OBJECT.

The metric of a discrete Hilbert space is a symmetric positive-semi-definite
bilinear form :math:`G`: :math:`\langle x, y\rangle = y^{\mathsf T} G x`.
Until P7 (campaign 1) the tree could spell exactly two realizations, both
*multiplied* into the element: a broadcast diagonal weight array (the
Hadamard product :math:`G \odot x`) and the per-axis factor measures of an
axis-built space. Neither can express a metric with off-diagonal structure —
and the tree carries one that provably needs it: the slab spherical-harmonic
frame's discrete Gram has live off-diagonals at 0.93 of the Cauchy–Schwarz
scale, so **no diagonal metric satisfies Parseval on it** (`[M]` 2026-08-30:
a diagonal ``1/diag(G)`` dressing reads the Parseval ratio 1.806 where the
dense pseudo-inverse reads 1.000000000000 on the same band-limited field —
the wrong-metric discriminator gate in ``tests/numerics/test_frame.py``).

This module makes the metric a thing that is **applied** rather than a thing
that is multiplied. The family owns the arithmetic; the Hadamard weight is
the diagonal special case it always was:

* :class:`DiagonalMetric` — the Hadamard realization: a weight array
  broadcast against the **leading** axes of the element (the tree's metric
  convention — :func:`_broadcast_leading`, the one home since
  ``FunctionSpace._broadcast_metric`` retired at CS4c step 6 item 6.2a).
* :class:`DenseMetric` — a dense symmetric matrix acting on the flattened
  leading block (row-major — the same flattening as
  :attr:`~orpheus.numerics.frame.FrameBase.discrete_gram`, whose
  pseudo-inverse is the realization's flagship occupant).

A :class:`~orpheus.numerics.space.FunctionSpace` resolves its metric
*source* (the ``metric`` field, the legacy ``inner_product_weights`` array,
or nothing) into one of these realizations exactly once and delegates its
three metric verbs — ``apply_metric`` / ``apply_inverse_metric`` /
``inner_product`` — to it. Axis-built spaces keep their per-axis path
(``_apply_axes_weights``): the axes ARE the metric source there, and a
measure is diagonal by nature — a Gram is a *form*, a different concept,
which is why :class:`~orpheus.numerics.axis.Axis` never grows a matrix slot
(the P7 design ruling: the generator induces, the space holds).

The inverse face is Moore–Penrose, everywhere, by doctrine
==========================================================

``apply_inverse`` is the **pseudo**-inverse :math:`G^{+}`: the reciprocal on
the metric's range, zero on its kernel. This is not a numerical convenience
— on the flagship consumer it is the only thing that exists: `[M]` the slab
``gauss_legendre(8).angular_frame(2)`` Gram is :math:`15\times 15` with 5
live diagonal slots and **rank 4** (one noise-level mode at ``~1e-16`` —
its last digit is solver-dependent: ``eigvalsh`` reads ``8.2e-17``, SVD
``6.0e-17``), so ``np.linalg.inv`` raises where :func:`np.linalg.pinv`
returns the exact object Parseval needs: :math:`G G^{+} G = G` (`[M]`
max-abs residual ``1.6e-15``, relative ``7.8e-16`` — Frobenius-relative
the same) makes :math:`\|M\psi\|^2_{G^{+}} = c^{\mathsf T} G G^{+} G c
= \|S_0 c\|^2_W` a **theorem for any** :math:`G`, singular or not. The
diagonal realization has carried the same doctrine since the trace-metric
work (zero-weight tangential ordinates map to zero); the dense realization
extends it unchanged.

The pairing has ONE spelling: :math:`\sum (Gx) \odot y`
=======================================================

:meth:`HilbertMetric.pairing` is ``float(np.sum(self.apply(x) * y))`` for
every realization — the single home of the pairing arithmetic. Two reasons,
both load-bearing:

* **Bit-identity.** The legacy diagonal spelling ``np.sum(w * x * y)``
  evaluates left-to-right as ``(w*x)*y``, and ``apply(x)`` IS ``w_b * x``,
  so the reduction tree is preserved exactly. The matmul spelling
  ``y @ (diag(w) @ x)`` is NOT equivalent: `[M]` it differs on 60–70 %
  of draws at ``n = 15`` (the design scan's draw read 1360 of 2000),
  worst per-seed deviation banded 46–16384 ULP / rel ``9.2e-15`` to
  ``2.1e-12`` over 40 seeds (archivist census) — routing the shipped
  diagonal path through a densified matmul would move pinned numbers
  tree-wide. The bit-exact witness is
  ``tests/numerics/test_dense_metric.py``'s pairing-spelling gate.
* **Single source.** ``AdjointOperator`` builds the Hilbert adjoint from
  ``apply_metric``/``apply_inverse_metric`` while the pairing that judges
  it comes from ``inner_product``; deriving the pairing FROM ``apply``
  makes the two agree by construction (the ERR-067 family — two metric
  spellings diverging silently — is unspellable here).

CS4c compatibility (the leg target)
===================================

The recorded CS4c debt (``docs/theory/foundations/frame.rst``, the
recorded-debt admonition) intends the Riesz legs
``A* = A.domain.riesz_raise ∘ A.dual() ∘ A.codomain.riesz_lower``. This
family's two faces are exactly what those legs will wrap: the metric
arithmetic lives here once, so retiring ``AdjointOperator`` into the leg
composition later needs no third spelling of it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "HilbertMetric",
    "DiagonalMetric",
    "DenseMetric",
    "FactoredMetric",
]


#: Relative cutoff handed to :func:`np.linalg.pinv` by
#: :meth:`DenseMetric.inverse_of` (and by :class:`DenseMetric` when it must
#: derive its own inverse face). Pinned rather than left to numpy's default:
#: an implicit default is a silent dependency on a numpy version, and the
#: mutation battery's over-truncation arm is only meaningful against a
#: pinned value.
#:
#: **What it is FOR: structural RANK DETERMINATION, not noise suppression.**
#: ⛔ REFUTED 2026-08-31 — this block called ``σ₅`` a "``~1e-16`` noise
#: mode" and placed ``1e-12`` "~4 orders above the noise floor". Every
#: number it quoted reproduces; the READING does not, and "noise floor"
#: names something that does not exist. An angular frame on a 1-D chart has
#: a Gram that is **rank-deficient by construction** (ERR-080, #429):
#: :meth:`Quadrature.angular_frame` column-stacks three
#: ``axis_cosines`` of which two are the documented ZERO FALLBACK on a slab,
#: tags the measure ``support=S^2``, and ``_evaluate_real_sh`` then reads
#: ``arctan2(0, 0) = 0`` — a fabricated azimuth, under which every ``m > 0``
#: harmonic is a LIVE column instead of being absent. The redundancy is
#: closed-form: ``Y_2^{+2} = (√3/2)(1−μ²)`` and ``1−μ² = (2/3)(Y_00−Y_20)``
#: give ``Y_00 − Y_20 − √3·Y_22 ≡ 0``. `[M]` 2026-08-31 on
#: ``gauss_legendre(8).angular_frame(2)``: the predicted null vector
#: ``v = (1, 0, −1, 0, −√3)/√5`` over the live slots
#: ``(0,0),(1,0),(2,0),(2,1),(2,2)`` IS the SVD-measured one
#: (``|1 − |cos θ|| = 2.2e-16``) and it annihilates the **table**
#: (``‖A v‖∞ = 1.4e-16``), not merely the Gram — so ``σ₅`` is the
#: floating-point residue of an EXACT zero, not a small-but-real mode.
#: ``rcond`` separates a genuine range from a structural kernel.
#:
#: The admissible window has TWO measured edges. Configuration for every
#: number below: the shipped ``15×15``
#: :attr:`~orpheus.numerics.frame.FrameBase.discrete_gram` of
#: ``gauss_legendre(8).angular_frame(2)``, numpy 2.4.4, `[M]` 2026-08-31,
#: probes ``scratch/probe_rcond_0{3,4,10,11,14}_*.py`` (memo
#: ``scratch/rcond_rederivation.md``).
#:
#: * **Upper — over-truncation**, at ``1.752390e-02``, which is
#:   ``|λ₄|/|λ₁| = 4.744684e-02 / 2.707550`` to 7 s.f. Construction cannot
#:   see it (``G⁺ G G⁺ = G⁺`` holds for a TRUNCATED pinv too, so
#:   :data:`_DENSE_METRIC_PENROSE_RTOL` is blind); the sole catcher is
#:   ``test_parseval_analysis_is_an_isometry_onto_its_image``
#:   (``rtol=1e-12``). `[M]` 200-seed census astride the edge: at
#:   ``1.75e-2`` **0/200** seeds fail (``|r−1| ≤ 2.4e-15``); at ``1.76e-2``
#:   **200/200** do (``|r−1|`` spanning ``2.8e-07 … 4.1e-01``). The
#:   LOCATION is a spectrum property and may be quoted; the failure
#:   MAGNITUDE is draw-dependent and must not be. The old "5e-2, then 3e-2"
#:   history was scan resolution: the transition is a STEP across ``2e-6``
#:   in ``rcond``, not a cliff with a slope.
#: * **Lower — a construction REFUSAL**, at ``8.696754e-17``, equal to 7
#:   s.f. to the largest relative round-off residue as ``eigh`` reports it
#:   (the decomposition ``pinv(hermitian=True)`` actually cuts on). Below it
#:   :meth:`DenseMetric.__post_init__` raises — ``max|G⁺ G G⁺ − G⁺| =
#:   7.9e+16`` at ``rcond = 1e-18``. So **no ADMISSIBLE rcond can admit the
#:   kernel**: ``‖G⁺‖₂ = 21.076 = 1/4.744684e-02`` is bit-constant across
#:   the whole window, and the corrupt band is unreachable, not merely
#:   distant.
#:
#: ``1e-12`` sits ``10.24`` decades below the upper edge and ``4.06`` above
#: the lower. ⚠ The lower margin must stay wide **because the number it is
#: measured from is round-off with no stable value**: `[M]` the same
#: matrix's largest residue reads ``8.70e-17`` (``eigh``), ``9.71e-18``
#: (``eigvalsh`` and ``svd(hermitian=True)``) and ``2.27e-17`` (general
#: ``svd``) — 9.0× spread over four numpy routines, so it moves with the
#: LAPACK driver, the BLAS and the machine.
#:
#: ⚠ **The 15-decade gap is a property of the GL8 / L=2 frame, not of this
#: constant — the pin is NOT comfortable everywhere.** `[M]` census at this
#: rcond (``scratch/probe_rcond_16_census.py``; ``REFUSED`` =
#: ``basis_space`` raises, ``BREACH`` = builds but ``|Parseval − 1| >
#: 1e-12``, seed 1234). Over **105 slab rows** (``gauss_legendre`` orders
#: ``2,3,4,5,6,7,8,9,11,12,16,17,24,32,33`` × ``L ∈ {0,1,2,3,4,5,7}``):
#: **20 REFUSED, 11 BREACH — 31 affected (30 %), minimum affected L = 3**,
#: including the DEFAULT order: accessing ``basis_space`` on
#: ``gauss_legendre(16).angular_frame(4)`` raises. Over **196 3-D rows**
#: (``level_symmetric(2..16)``, ``product`` incl. ``3×6``/``5×7``/``9×13``,
#: ``folded_product``, ``lebedev(5..29)``, same ``L`` grid): **0 affected**,
#: worst headroom 6.5 decades. The mechanism is not
#: the kernel rising to meet the pin — on a 1-D chart the fabricated
#: azimuth mints ``~L²/2`` phantom columns against a fixed node count, the
#: odd-``m`` ones carry a non-polynomial ``√(1−μ²)``, and the LIVE spectrum
#: descends continuously THROUGH ``1e-12`` (``GL24``/``L=7``: smallest kept
#: mode ``3.73e-12``, 0.57 decades above the pin; ``GL17``/``L=7``: largest
#: dropped mode ``3.44e-13``, 0.46 below). There, no cutoff is right.
#:
#: When ERR-080 lands, the kernel disappears and this constant reverts to
#: what it was mis-described as — a pure conditioning guard. `[R]`+`[M]`:
#: with the azimuth no longer fabricated a slab carries only its ``m = 0``
#: columns, the Legendre polynomials, which Gauss–Legendre integrates
#: exactly to degree ``2N−1``; so the Gram becomes DIAGONAL, exactly
#: ``2/(2ℓ+1)``, with ``cond = 2L+1`` (`[M]`
#: ``scratch/probe_rcond_17_post_repair.py``: max off-diagonal ``≤1.3e-15``,
#: ``σ_min/σ_max = 1/(2L+1)`` on ``GL{5,8,9,16,24,32}`` ×
#: ``L ∈ {2,3,4,5,7}``). The frame then takes its DIAGONAL arm, no
#: :class:`DenseMetric` is built on a slab at all, and all 31 affected rows
#: retire. The falsifiable check on the day it lands:
#: ``gauss_legendre(8).angular_frame(2).discrete_gram_structure is
#: GramStructure.DIAGONAL``.
_DENSE_METRIC_RCOND: float = 1e-12

#: Symmetry admission threshold for :class:`DenseMetric`, relative to the
#: matrix's own scale. An asymmetric form is not an inner product, so an
#: asymmetric matrix is REFUSED rather than silently symmetrized — the
#: producer symmetrizes (as :meth:`DenseMetric.inverse_of` does). `[M]` the
#: quantities this guards against are real: ``np.linalg.pinv`` WITHOUT
#: ``hermitian=True`` returns ``max|M − Mᵀ| = 4.74e-14`` on the slab Gram,
#: while the honest spellings sit at ``~1e-16``.
_DENSE_METRIC_SYMMETRY_RTOL: float = 1e-12

#: Penrose-consistency admission threshold for an explicitly supplied
#: inverse face: ``M @ M⁺ @ M`` must reproduce ``M`` to this relative
#: tolerance, or the pair is refused — an inconsistent (matrix, inverse)
#: pair is an illegal state, not a representation choice.
_DENSE_METRIC_PENROSE_RTOL: float = 1e-10


def _broadcast_leading(w: NDArray, target_ndim: int) -> NDArray:
    r"""Pad ``w`` with trailing singleton axes up to ``target_ndim``.

    The tree's metric convention is **leading-aligned**: a metric spanning
    the first axes of an element acts on the full tensor by broadcasting
    over the trailing (element) axes. This is the arithmetic that lived in
    ``FunctionSpace._broadcast_metric`` until P7, which delegated here until
    it retired at CS4c step 6 item 6.2a (2026-09-07) — one home, one door;
    a no-op whenever ``w`` already spans every axis.
    """
    w = np.asarray(w)
    if w.ndim >= target_ndim:
        return w
    return w.reshape(w.shape + (1,) * (target_ndim - w.ndim))


class HilbertMetric(ABC):
    r"""A symmetric positive-semi-definite bilinear form, as an object.

    The contract is three verbs and one derived pairing:

    * :meth:`apply` — :math:`x \mapsto Gx` (the forward face).
    * :meth:`apply_inverse` — :math:`x \mapsto G^{+}x`, the Moore–Penrose
      pseudo-inverse face (reciprocal on the range, zero on the kernel).
    * :meth:`validate_for` — the admission check a
      :class:`~orpheus.numerics.space.FunctionSpace` runs at construction
      when this object is installed as its metric source.
    * :meth:`pairing` — :math:`\langle x, y\rangle = \sum (Gx) \odot y`,
      **final in spirit**: realizations inherit it so the pairing
      arithmetic has exactly one spelling (see the module docstring's
      bit-identity note).
    """

    @abstractmethod
    def apply(self, x: NDArray) -> NDArray:
        r"""Return :math:`G x` (leading-aligned on the element's axes)."""

    @abstractmethod
    def apply_inverse(self, x: NDArray) -> NDArray:
        r"""Return :math:`G^{+} x` — pseudo-inverse on the range, 0 on the kernel."""

    @abstractmethod
    def validate_for(self, shape: tuple[int, ...]) -> None:
        r"""Raise if this metric cannot serve a space of element ``shape``."""

    def pairing(self, x: NDArray, y: NDArray) -> float:
        r"""Return :math:`\langle x, y\rangle = \sum (Gx) \odot y` — the one spelling."""
        return float(np.sum(self.apply(x) * y))


@dataclass(frozen=True)
class DiagonalMetric(HilbertMetric):
    r"""The Hadamard realization: a diagonal weight array, leading-aligned.

    The special case every pre-P7 metric was. ``apply`` broadcasts the
    weights against the leading axes (:func:`_broadcast_leading`) and
    multiplies; ``apply_inverse`` is the masked reciprocal — the
    Moore–Penrose doctrine's diagonal form, operation-for-operation the
    arithmetic that lived in ``FunctionSpace._diagonal_apply_metric`` /
    ``_diagonal_apply_inverse_metric`` (bit-identity is a contract here,
    not an accident: the resolved legacy ``inner_product_weights`` path
    routes through this class).
    """

    weights: NDArray

    def apply(self, x: NDArray) -> NDArray:
        return _broadcast_leading(self.weights, np.ndim(x)) * x

    def apply_inverse(self, x: NDArray) -> NDArray:
        wb = _broadcast_leading(self.weights, np.ndim(x))
        nonzero = wb != 0.0
        return np.where(nonzero, x / np.where(nonzero, wb, 1.0), 0.0)

    def validate_for(self, shape: tuple[int, ...]) -> None:
        # The legacy dense-array contract: any array broadcast-compatible
        # with the element is admissible, and broadcast errors surface at
        # first application exactly as they always did. Nothing to refuse
        # at construction beyond what numpy will say more precisely later.
        return None

    def apply_block(
        self,
        x: NDArray,
        *,
        start: int,
        block_shape: tuple[int, ...],
        inverse: bool,
    ) -> NDArray:
        r"""Apply to ONE interior index block of ``x`` (``start`` leading
        ranks precede it) — the positioned form
        :class:`FactoredMetric` composes. Operation-for-operation the
        per-axis arithmetic of ``FunctionSpace._apply_axes_weights``
        (explicit reshape: leading 1s, the block shape, trailing 1s), so
        a factored diagonal block and an axis-borne measure are the same
        arithmetic by construction."""
        out = np.asarray(x)
        rank = len(block_shape)
        w = np.ascontiguousarray(
            np.broadcast_to(np.asarray(self.weights), block_shape)
        )
        wb = w.reshape(
            (1,) * start + block_shape + (1,) * (out.ndim - start - rank)
        )
        if inverse:
            nonzero = wb != 0.0
            return np.where(nonzero, out / np.where(nonzero, wb, 1.0), 0.0)
        return out * wb


@dataclass(frozen=True)
class DenseMetric(HilbertMetric):
    r"""A dense symmetric matrix metric on the flattened leading block.

    ``matrix`` is :math:`G`, ``(K, K)`` with ``K`` the row-major flattening
    of the metric's block — the same layout as
    :attr:`~orpheus.numerics.frame.FrameBase.discrete_gram`
    (``table.reshape(N, K)`` order), so a frame's Gram algebra and this
    realization agree on what "slot :math:`k`" means by construction.
    Elements with extra **trailing** axes are handled leading-aligned, like
    every metric in the tree: ``apply`` reshapes to ``(K, -1)``, matmuls,
    and reshapes back.

    Construction refuses two illegal states (the guard thresholds are the
    module constants — a gate on this type must quote them, per
    ``vv-principles`` #16):

    * an **asymmetric** matrix (an asymmetric form is not an inner
      product) — producers symmetrize, this type does not;
    * an explicitly supplied inverse face that fails the **Penrose
      identity** :math:`G G^{+} G = G` (an inconsistent pair is an illegal
      state, not a representation choice).

    When no inverse face is supplied it is derived here, once, as
    ``np.linalg.pinv(matrix, hermitian=True, rcond=_DENSE_METRIC_RCOND)``.
    Prefer :meth:`inverse_of` when what you hold is the GRAM whose inverse
    the metric should BE (the Parseval case): it keeps the exact Gram as
    the inverse face instead of a second pseudo-inversion.
    """

    matrix: NDArray
    inverse_matrix: Optional[NDArray] = field(default=None)

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(
                f"DenseMetric requires a square 2-D matrix, got shape "
                f"{np.shape(self.matrix)}"
            )
        scale = max(1.0, float(np.max(np.abs(m)))) if m.size else 1.0
        asym = float(np.max(np.abs(m - m.T))) if m.size else 0.0
        if asym > _DENSE_METRIC_SYMMETRY_RTOL * scale:
            raise ValueError(
                f"DenseMetric requires a symmetric matrix: "
                f"max|G - G^T| = {asym:.3e} exceeds "
                f"{_DENSE_METRIC_SYMMETRY_RTOL:.0e} of the matrix scale "
                f"{scale:.3e}. An asymmetric form is not an inner product — "
                f"symmetrize at the producer."
            )
        object.__setattr__(self, "matrix", m)
        if self.inverse_matrix is None:
            inv = np.linalg.pinv(
                m, hermitian=True, rcond=_DENSE_METRIC_RCOND
            )
            object.__setattr__(self, "inverse_matrix", inv)
        else:
            inv = np.asarray(self.inverse_matrix, dtype=float)
            if inv.shape != m.shape:
                raise ValueError(
                    f"DenseMetric inverse face shape {inv.shape} does not "
                    f"match the matrix shape {m.shape}"
                )
            residual = float(np.max(np.abs(m @ inv @ m - m)))
            if residual > _DENSE_METRIC_PENROSE_RTOL * scale:
                raise ValueError(
                    f"DenseMetric was handed an inconsistent inverse face: "
                    f"max|G G+ G - G| = {residual:.3e} exceeds "
                    f"{_DENSE_METRIC_PENROSE_RTOL:.0e} of the matrix scale "
                    f"{scale:.3e} (the Penrose identity fails)."
                )
            object.__setattr__(self, "inverse_matrix", inv)

    @classmethod
    def inverse_of(cls, gram: NDArray) -> "DenseMetric":
        r"""The metric whose MATRIX is the pseudo-inverse of ``gram``.

        The Parseval constructor: the frame's discrete Gram :math:`G` goes
        in, the metric :math:`G^{+}` comes out — with the (symmetrized)
        Gram itself installed as the inverse face, which is EXACT for a
        symmetric PSD form (:math:`(G^{+})^{+} = G` globally: both share
        the range, and :math:`G` is already zero on its own kernel) and
        strictly better conditioned than a second pseudo-inversion.
        """
        g = np.asarray(gram, dtype=float)
        g_sym = (g + g.T) / 2.0
        matrix = np.linalg.pinv(
            g_sym, hermitian=True, rcond=_DENSE_METRIC_RCOND
        )
        return cls(matrix=matrix, inverse_matrix=g_sym)

    @property
    def dim(self) -> int:
        r"""``K`` — the flattened slot count the matrix acts on."""
        return int(self.matrix.shape[0])

    def _matmul_leading(self, m: NDArray, x: NDArray) -> NDArray:
        flat = np.asarray(x).reshape(self.dim, -1)
        return (m @ flat).reshape(np.shape(x))

    def apply(self, x: NDArray) -> NDArray:
        return self._matmul_leading(self.matrix, x)

    def apply_inverse(self, x: NDArray) -> NDArray:
        inv = self.inverse_matrix
        assert inv is not None  # established by __post_init__; narrowing only
        return self._matmul_leading(inv, x)

    def validate_for(self, shape: tuple[int, ...]) -> None:
        size = int(np.prod(shape)) if shape else 1
        if size != self.dim:
            raise ValueError(
                f"DenseMetric of dimension {self.dim} cannot serve a space "
                f"of shape {shape} ({size} slots) — the matrix must span "
                f"the element's flattened leading block."
            )

    def apply_block(
        self,
        x: NDArray,
        *,
        start: int,
        block_shape: tuple[int, ...],
        inverse: bool,
    ) -> NDArray:
        r"""Apply to ONE interior index block of ``x`` — the positioned
        form :class:`FactoredMetric` composes: the element is reshaped to
        ``(lead, K, trail)`` around the block's row-major flattening and
        the matrix acts on the middle axis."""
        m = self.inverse_matrix if inverse else self.matrix
        assert m is not None  # established by __post_init__; narrowing only
        out = np.asarray(x)
        lead_shape = out.shape[:start]
        lead = int(np.prod(lead_shape)) if lead_shape else 1
        x3 = out.reshape(lead, self.dim, -1)
        return np.einsum("kl,alb->akb", m, x3).reshape(out.shape)


@dataclass(frozen=True)
class FactoredMetric(HilbertMetric):
    r"""A lazy tensor product of per-block metrics — :math:`G = G_1 \otimes G_2 \otimes \cdots`.

    Each entry is ``(block_shape, factor)``, the factor one of the
    POSITIONED leaf realizations (:class:`DiagonalMetric` /
    :class:`DenseMetric`) or ``None`` for a Euclidean block; the blocks'
    concatenation must equal the space shape (:meth:`validate_for`).
    Factors apply in sequence, each to its own index block — the
    Kronecker product is **never materialized**, the same discipline the
    per-axis path already follows ("never materializes the outer
    product"), so the pairing is exact and the cost is one pass per
    non-Euclidean factor.

    The occupant this realization exists for: a tensor product with a
    dense-metric factor — e.g. the harmonic frame's moment space, the
    Parseval-dressed spherical-harmonic coefficient block ⊗ the spatial
    cell measure — where the dense weights array ``*`` carried until CS4c
    step 6 item 6.2a could not carry the off-diagonal block and dropping
    it would silently revert the
    product to Euclidean on that factor (`[M]` 2026-08-30, the
    pre-repair behaviour: the probe pairing read 33.0 where
    :math:`G \otimes w` gives 109.0 — a value bug wearing a
    representation costume).
    """

    entries: tuple[
        tuple[tuple[int, ...], "DiagonalMetric | DenseMetric | None"], ...
    ]

    def __post_init__(self) -> None:
        for block_shape, factor in self.entries:
            if isinstance(factor, DenseMetric):
                size = int(np.prod(block_shape)) if block_shape else 1
                if size != factor.dim:
                    raise ValueError(
                        f"FactoredMetric entry: a DenseMetric of dimension "
                        f"{factor.dim} cannot serve a factor block of shape "
                        f"{block_shape} ({size} slots)."
                    )

    def _walk(self, x: NDArray, *, inverse: bool) -> NDArray:
        out = np.asarray(x)
        start = 0
        for block_shape, factor in self.entries:
            if factor is not None:
                out = factor.apply_block(
                    out, start=start, block_shape=block_shape, inverse=inverse
                )
            start += len(block_shape)
        return out

    def apply(self, x: NDArray) -> NDArray:
        return self._walk(x, inverse=False)

    def apply_inverse(self, x: NDArray) -> NDArray:
        return self._walk(x, inverse=True)

    def validate_for(self, shape: tuple[int, ...]) -> None:
        concat: tuple[int, ...] = ()
        for block_shape, _ in self.entries:
            concat = concat + block_shape
        if concat != shape:
            raise ValueError(
                f"FactoredMetric blocks {concat} do not concatenate to the "
                f"space shape {shape}."
            )
