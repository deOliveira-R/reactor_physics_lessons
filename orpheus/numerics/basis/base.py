r"""The :class:`Basis` ABC — a discrete spectral basis on a measure space.

A *basis* is the **synthesis (trial) side** of a discrete frame
(:class:`orpheus.numerics.frame.FrameBase`): a collection of functions that can be
tabulated at sample points, reconstructed from coefficients, and weighed against
a quadrature to form a discrete Gram. It is the choice-free, measure-free half of
the analysis/synthesis pair — the :class:`~orpheus.numerics.measure.DiscreteMeasure`
supplies the analysis (test) side, and the :class:`FrameBase` binds the two.

Why the ABC is un-deferred on a single concrete basis
=====================================================

The package previously deferred a formal ``Basis`` ABC "until a second concrete
basis arrives" (``feedback_unify_after_two_instances``). It is promoted now — with
:class:`~orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis`
still the only concrete member — because the **forcing consumer has arrived**: the
generic :class:`~orpheus.numerics.frame.FrameBase` binds an *abstract* basis to a
measure and applies a non-identity morphism (analysis / reconstruction) across the
interface. The Frame needs the interface, not a second basis; the interface itself
is math-rigid (Grand Report v3 §5.4 lists the nine eventual bases — real spherical
harmonics, Chebyshev, Lagrange, FE shape functions, ...) and every one of them
tabulates, reconstructs, and has a discrete Gram. Deferring the ABC would force the
Frame to name a concrete basis, which is exactly the coupling the abstraction exists
to break.

The contract — tabulate, then contract a cached table
=====================================================

:meth:`evaluate` is the only method that takes sample *points*; it produces the
``Φ(point, mode)`` **table** (the layout-bearing object — ``(N, ℓ, m)`` for spherical
harmonics, ``(N, K)`` for a flat-mode basis). Every other operation **consumes that
table**, so the :class:`FrameBase` evaluates ONCE (``frame.table``) and the per-apply hot
path never re-tabulates. The basis owns these contractions because the index layout is
the basis's own; the :class:`FrameBase` stays layout-agnostic and merely delegates.

The shared naked synthesis :math:`S_0`
--------------------------------------

:meth:`synthesize` is the naked synthesis :math:`S_0(c) = \sum_k \phi_k\, c_k` (the
frame-theory *synthesis operator* :math:`T^*` — NO weights, NO dual factor). The four
weighted operations are each :math:`S_0` (or its transpose) post-multiplied by ONE
diagonal weight family:

* :meth:`analyze` :math:`M = w_n \cdot (\text{analysis contraction})` — the analysis
  operator :math:`T`;
* :meth:`analyze_transpose` :math:`M^\top = w_n \cdot S_0` — its representation transpose;
* :meth:`reconstruct` :math:`R = d_k \cdot S_0` — the canonical-dual synthesis (for the
  SH basis :math:`d_\ell = 2\ell+1`);
* :meth:`reconstruct_transpose` :math:`R^\top = d_k \cdot S_0^\top` — its representation
  transpose (measure-free, like :meth:`reconstruct`).

The two analysis-side weightings bake in the quadrature weight :math:`w_n`; the two
synthesis-side ones carry only the dual factor :math:`d_k` — synthesis is measure-free.

They are kept as **fused contractions** (not ``weight ⊙ synthesize``) because FP
non-associativity makes the factored form drift at the ULP level, and the scattering
kernel is pinned at 0 ULP. The shared :math:`S_0` is the *conceptual* unity, documented
here; the implementation keeps the fused einsum.

References
----------

* Grand Report v3 §5.4 — Basis hierarchy.
* Christensen, O. (2016). *An Introduction to Frames and Riesz Bases*, 2nd ed.
  Birkhäuser — the analysis/synthesis operator pair this ABC is the trial side of.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

from numpy.typing import NDArray

from orpheus.numerics.manifold import Manifold, Quotient, Sphere
from orpheus.numerics.symmetry import SubgroupOfO3

if TYPE_CHECKING:
    from orpheus.numerics.measure import DiscreteMeasure
    from orpheus.numerics.space import FunctionSpace


__all__ = ["Basis", "GramStructure", "TruncatedBasis"]


class GramStructure(Enum):
    r"""The structure of a TRIAL basis's frame-Gram — the projection-validity declaration.

    When a basis is the **trial** side of a :class:`~orpheus.numerics.frame.FrameBase`,
    the coefficient extraction :meth:`~orpheus.numerics.frame.FrameBase.project`
    (:math:`G^{-1}M`) normalises by the cross Gram :math:`G = MR`. The frame computes
    :math:`G` with a single **row-sum probe** (``analysis(reconstruction(ones))``) — but
    that probe equals the projection's required normalisation ONLY under one of two
    structural conditions. This enum is the basis's declaration of which (if any) holds,
    so the wrong combination cannot be spelled silently (a precondition the **type**
    enforces, not a docstring):

    * :attr:`DIAGONAL` — disjoint-support trial (orthogonal harmonics, nested cell /
      group indicators): :math:`MR` is diagonal, the row sum IS the diagonal.
    * :attr:`PARTITION_OF_UNITY` — overlapping trial whose membership rows sum to 1
      (the fractional :class:`~orpheus.numerics.basis.overlap_basis.OverlapBasis`):
      :math:`MR` is NOT diagonal, but :math:`R\mathbf 1 = \mathbf 1` collapses the probe
      to the per-region weight anyway.
    * :attr:`DENSE` — neither (a tapered weight, a higher-rank GEC moment — GitHub #275):
      the row-sum probe is wrong; the true projection needs the dense :math:`(MR)^{-1}M`
      solve, which is NOT built. :meth:`~orpheus.numerics.frame.FrameBase.project`
      **refuses** a DENSE-Gram trial rather than return a silently-wrong coarsening.

    The base :class:`Basis` defaults to :attr:`DENSE` — the safe refusal: a new basis
    must consciously declare it is row-sum-collapsible (having checked its Gram) before
    ``project`` will use the shortcut on it.
    """

    DIAGONAL = "diagonal"
    PARTITION_OF_UNITY = "partition_of_unity"
    DENSE = "dense"


@runtime_checkable
class TruncatedBasis(Protocol):
    r"""A basis indexed by a truncation ORDER :math:`L` — the harmonic family's shared surface.

    The real spherical harmonics, their σ-even restriction and the Legendre
    basis on :math:`S^2/O(2)_a` are each *one family of functions truncated
    at a degree*, and every consumer that spells "the coefficient space of
    order :math:`L`" — an operator's endomorphic ends, a moment field's
    angular head, a frame's mint — reads it off such a basis
    (:attr:`space`), never mints it from the integer. The family the frame
    binds is DERIVED from the point set its measure lives on
    (:meth:`~orpheus.numerics.quadrature.directional.Quadrature._harmonic_basis`),
    so a consumer that re-minted the space from ``L`` would silently choose
    the full-sphere family on every rule — which is exactly how the angular
    moment space had eight homes until #429 tracker 2.5 (2026-09-02):
    seven production ``SphericalHarmonicSpace.from_L(L)`` mints beside the
    frame that already carried it.

    Structural (``Protocol``) and ``runtime_checkable``: the
    :class:`~orpheus.transport.frames.harmonic_frame.HarmonicFrame` door
    asks for THIS surface — *does the trial basis carry a truncation
    order?* — not for one class, because the fold's σ-even restriction and
    the slab's Legendre basis are as much harmonic-family members as the
    full harmonics, and a door that named one class refused the others
    (`[M]` 2026-09-02, the two ``isinstance`` narrowings at that door were
    the first thing a 1-D solve would have hit after the fix, at
    :math:`L = 0`).
    """

    @property
    def L(self) -> int:
        """The truncation order — the degree the family is cut at."""
        ...

    @property
    def space(self) -> "FunctionSpace":
        """The coefficient space this truncated family spans."""
        ...

    def at_order(self, L_new: int, /) -> "TruncatedBasis":
        """THIS family at the truncation order ``L_new`` — the same functions
        cut at another degree (same domain, same spent axis), the verb a
        head truncates THROUGH (:func:`~orpheus.numerics.spaces.moment_head.truncated_head`,
        CS4c step 6 item 6.2c-ii)."""
        ...


class Basis(ABC):
    r"""Abstract discrete spectral basis — the synthesis side of a :class:`FrameBase`.

    Concrete bases (real spherical harmonics today; Legendre, Chebyshev,
    Lagrange/FE shape functions to come) implement the operations below. A basis
    is *choice-free*: it knows its functions and their convention, but not which
    quadrature samples them — that choice is the
    :class:`~orpheus.numerics.measure.DiscreteMeasure`, bound to the basis by a
    :class:`~orpheus.numerics.frame.FrameBase`.

    The contraction methods take the ``table`` from :meth:`evaluate` (positional,
    so concrete bases may name their arguments in their own domain vocabulary —
    e.g. ``directions``), so the :class:`FrameBase` tabulates once and the per-apply
    hot path never re-evaluates.
    """

    # ── Tabulation (the only points-consuming method) ─────────────────────
    @abstractmethod
    def evaluate(self, points: NDArray, /) -> NDArray:
        r"""Tabulate the basis functions at ``points`` → the ``Φ(point, mode)`` table.

        For the spherical-harmonic basis, ``Y[n, ℓ, ℓ+m] = Y_ℓ^m(Ω̂_n)`` at the
        ``(N, 3)`` direction cosines; a flat-mode basis returns ``(N, K)``.
        """
        ...

    # ── Table contractions (the Frame caches the table and delegates here) ──
    @abstractmethod
    def synthesize(self, coefficients: NDArray, table: NDArray, /) -> NDArray:
        r"""Naked synthesis :math:`S_0(c) = \sum_k \phi_k\, c_k` against a cached ``table``.

        The pure (frame-theory *synthesis operator* :math:`T^*`) reconstruction —
        NO measure weights, NO dual-frame factor. The shared kernel the three
        weighted contractions below are each one diagonal away from.
        """
        ...

    @abstractmethod
    def analyze(
        self, values: NDArray, table: NDArray, weights: NDArray, /,
    ) -> NDArray:
        r"""Analysis :math:`M = T`: sampled values → coefficients.

        :math:`(M f)_k = \sum_n w_n\, \phi_k(x_n)\, f(x_n)` — the W-weighted
        projection onto the basis. For spherical harmonics
        :math:`\phi_\ell^m = \sum_n w_n Y_\ell^m(\hat\Omega_n)\,\psi_n`. The
        ``weights`` are the measure's (analysis is the *measured* / test side).
        """
        ...

    @abstractmethod
    def analyze_transpose(
        self, coefficients: NDArray, table: NDArray, weights: NDArray, /,
    ) -> NDArray:
        r"""Representation transpose :math:`M^\top = w_n \cdot S_0`: coefficients → values.

        The matrix transpose of :meth:`analyze` (NOT its Hilbert adjoint): the
        naked synthesis weighted by the quadrature weight on each node. The
        metric-aware ``AdjointOperator`` machinery combines this with the
        frame's metrics — the measure weights on the domain, the F-0 PARSEVAL
        metric :math:`G^{-1}` (the inverse discrete Gram) on the codomain — to
        form the physical Hilbert adjoint :math:`M^* = S_0 \circ G^{-1}`, so
        the :class:`FrameBase`'s analysis face gets ``.H`` for free.
        """
        ...

    @abstractmethod
    def reconstruct(self, coefficients: NDArray, table: NDArray, /) -> NDArray:
        r"""Reconstruction :math:`R`: coefficients → values (the dual-frame synthesis).

        :math:`S_0` weighted by the canonical-dual factor (intrinsic to the basis;
        for spherical harmonics :math:`R = \sum_\ell (2\ell+1) \sum_m Y_\ell^m\,
        \phi_\ell^m` — the addition-theorem reconstruction). **Measure-free** (the
        dual factor needs no quadrature) — synthesis is the choice-free / trial side.
        """
        ...

    @abstractmethod
    def reconstruct_transpose(
        self, values: NDArray, table: NDArray, /,
    ) -> NDArray:
        r"""Representation transpose :math:`R^\top = d_k \cdot S_0^\top`: values → coefficients.

        The matrix transpose of :meth:`reconstruct` (NOT its Hilbert adjoint):
        :math:`(R^\top v)_k = d_k \sum_n \phi_k(x_n)\, v_n`, the naked
        analysis weighted by the canonical-dual factor :math:`d_k` (for spherical
        harmonics :math:`d_\ell = 2\ell+1`). **Measure-free**, symmetric with
        :meth:`reconstruct` — the quadrature weights are NOT baked in (unlike
        :meth:`analyze_transpose`, whose forward :meth:`analyze` carries them). The
        metric-aware ``AdjointOperator`` combines this with the codomain/domain
        Gram to form the W-weighted Hilbert adjoint :math:`R^*`, so the
        :class:`FrameBase`'s reconstruction face gets ``.H`` for free — symmetric with
        the analysis face.
        """
        ...

    # ── The Gram structure (the projection-validity declaration) ──────────
    @property
    def gram_structure(self) -> GramStructure:
        r"""How this basis's frame-Gram is structured when it is the **trial** side.

        Declares whether :meth:`~orpheus.numerics.frame.FrameBase.project`'s row-sum
        probe is a valid coefficient normalisation for this basis (:class:`GramStructure`).
        Defaults to :attr:`GramStructure.DENSE` — the safe refusal: a new basis must
        consciously override this to :attr:`~GramStructure.DIAGONAL` or
        :attr:`~GramStructure.PARTITION_OF_UNITY` (having verified its Gram) before
        ``project`` will use the shortcut; a DENSE trial is refused rather than
        silently mis-projected.
        """
        return GramStructure.DENSE

    # ── The discrete Gram + the coefficient space ─────────────────────────
    @abstractmethod
    def mass_matrix(self, measure: "DiscreteMeasure", /) -> NDArray:
        r"""Discrete Gram :math:`\sum_n w_n\, \phi_j(x_n)\, \phi_k(x_n)` over ``measure``.

        The frame operator :math:`S = T^* T` in discrete form. Equals the
        continuous Gram (the basis's intrinsic metric) when the quadrature is
        exact to the basis's degree; the residual is a quadrature-exactness
        diagnostic. (A one-off diagnostic, so it is naturally measure-based and
        evaluates its own table.)
        """
        ...

    # ── The manifold the basis functions are defined ON ───────────────────
    @property
    @abstractmethod
    def domain(self) -> Manifold:
        r"""The :class:`~orpheus.numerics.manifold.Manifold` these functions EAT.

        A basis function is a **map**, and a map is not defined until its
        source is: :math:`Y_\ell^m : S^2 \to \mathbb R` takes a POINT of
        :math:`S^2`. This is that source.

        ⚠ Not to be confused with :attr:`space`, which is the other end of a
        different arrow. There are three levels and the two properties name two
        of them:

        =====================================  ==========================
        level                                  named by
        =====================================  ==========================
        the manifold :math:`M`                 :attr:`domain` (here)
        fields on :math:`M`, discretized       ``measure.space``
        coefficients :math:`\mathbb R^K`       :attr:`space`
        =====================================  ==========================

        So :attr:`space` answers *what do these live in*; :attr:`domain`
        answers *what do these eat*. The distinction is not academic — it is
        what makes a space name falsifiable. Before this property existed, an
        :class:`~orpheus.numerics.basis.indicator_basis.IndicatorBasis` over
        an ENERGY-group partition and one over a 2-cell SPATIAL partition both
        named their coefficient space ``L2[coarse_cells_R1]``, so they compared
        ``==`` **and** hash-equal: an illegal state that was representable,
        because the manifold was smuggled through a hard-coded name string
        (:doc:`/theory/foundations/manifolds`).

        ⭐ Abstract rather than defaulted, and a ``@property`` rather than a
        field. Abstract because a basis that cannot say what it eats is not a
        basis, so the refusal belongs at construction. A property because a
        dataclass FIELD does not satisfy an abstract property in Python —
        ``ABCMeta`` re-checks ``getattr(cls, name)`` and an annotation-only
        field puts nothing in ``__dict__``, so the subclass stays abstract —
        and because every abstract property in this tree
        (:attr:`space`, ``FrameBase.test``, ``LinearOperator.domain``,
        ``Manifold.dim``) is already answered that way.
        """
        ...

    # ── The symmetry the basis functions HAVE — read off the domain ───────
    @property
    @final
    def invariance_group(self) -> SubgroupOfO3 | None:
        r"""The subgroup of :math:`O(3)` every one of these functions is invariant under.

        ⭐ **Derived, never stored, and ``@final``** — the basis-side twin of
        :attr:`DiscreteMeasure.quotient_group
        <orpheus.numerics.measure.DiscreteMeasure.quotient_group>`, for the
        same reason it is derived there: a basis declares the symmetry its
        functions HAVE by naming the manifold they EAT. A function on an
        orbit space :math:`M/H` is, pulled back to :math:`M`, exactly an
        :math:`H`-invariant function — that is what a function on a quotient
        *is* — so the group is already in :attr:`domain`, as
        :attr:`Quotient.by <orpheus.numerics.manifold.Quotient.by>`, and a
        second slot for it would be a second home for one fact, kept in
        agreement by hand. (#429 tracker 2.1b, 2026-09-01: the tracker asked
        for a FIELD answered by six subclasses; the phase opener found the
        answer sitting in ``domain.by``, exactly as tracker 2.0d's
        ``quotient_group`` field dissolved into ``Quotient.by`` at 2.0c.)

        Three answers, by the TYPE of :attr:`domain`:

        * a :class:`~orpheus.numerics.manifold.Quotient` of the sphere — the
          group it was quotiented BY. The σ-even sub-basis
          (:class:`~orpheus.numerics.basis.spherical_harmonic_basis.MirrorEvenSphericalHarmonicBasis`)
          answers ``Mirror(axis)``; the Legendre basis on
          :math:`S^2/O(2)_a` (tracker 3.4) answers ``O2(a)`` — the FULL
          stabiliser, since the entry is named by it (#432).
        * the :class:`~orpheus.numerics.manifold.Sphere` itself —
          :attr:`SubgroupOfO3.Trivial <orpheus.numerics.symmetry.SubgroupOfO3.Trivial>`.
          :math:`O(3)` acts on the domain and the basis has spent none of it:
          the full degree-:math:`L` real harmonics have no common symmetry.
        * anything else — ``None``. No subgroup of :math:`O(3)` acts on a
          spatial mesh, an energy-group index or a trace-DOF index set, so
          the question has no answer there rather than a trivial one: it is
          the category answer :attr:`DiscreteMeasure.phase
          <orpheus.numerics.measure.DiscreteMeasure.phase>`'s dispatch gives
          for the same manifolds, and the reason ``Trivial`` would be a lie —
          it asserts an action that does not exist.

        ⚠ **HAS versus SPENT.** A measure carries two group slots because the
        two come apart for a POINT SET: a rule can be *closed under* a mirror
        (``invariance_group``, HAS) without having been *folded by* it
        (``quotient_group``, SPENT), and once folded it is no longer closed.
        For FUNCTIONS the two coincide — being :math:`H`-invariant and
        descending to :math:`M/H` are one property — so a basis carries one
        slot, named for what it HAS, and read off what its domain SPENT. The
        ``None`` of the two properties mean different things for the same
        reason: a full-sphere rule has spent nothing (``quotient_group`` is
        ``None``), while the full-sphere harmonics HAVE the trivial group.

        ⚠ **The reading is a LOWER BOUND, and under-declaring is legal but
        lossy.** ``SphericalHarmonicBasis(L=0)`` is :math:`O(3)`-invariant
        and answers ``Trivial``, because its domain says :math:`S^2`. A basis
        invariant under more than its domain shows will be refused pairings
        it could have admitted once the frame checks its two halves (Part
        IV's G2, tracker 2.2: the group a measure SPENT must be one the basis
        HAS — ``measure.quotient_group ⊆ basis.invariance_group``). The
        remedy is to declare the finer domain, never to override this
        property: an override lets :attr:`domain` and this answer disagree,
        which is precisely the two-tags-that-drift state the derivation
        exists to make unspellable — hence ``@final``.

        `[M]` 2026-09-01: the fold's two halves read ONE object —
        ``folded_product(4, 8).measure.quotient_group`` and its frame basis's
        ``invariance_group`` are both ``Mirror('y')``, the ``by`` of the one
        memoised :class:`~orpheus.numerics.manifold.Quotient` — while the
        slab's rule (``S^2/O2_x``, spent ``O2('x')``) against the
        full-sphere harmonics (``Trivial``) reads ``Trivial ⊇ O2('x')``
        **False**: ERR-080's pairing is a lattice verdict now, and nothing
        yet refuses on it (``tests/numerics/test_basis_domain.py::test_e1``).
        """
        match self.domain:
            case Quotient(base=Sphere(), by=group):
                return group
            case Sphere():
                return SubgroupOfO3.Trivial
            case _:
                return None

    @property
    @abstractmethod
    def space(self) -> "FunctionSpace":
        r"""The coefficient :class:`FunctionSpace` this basis spans.

        A harmonic family's space is AXIS-BUILT (CS4c step 6 item 6.2c-ii):
        one MODAL head axis carrying the basis's CONTINUUM Gram as its
        measure, this basis its generator. A measure-free basis like the
        indicators carries no metric at all. The frame's codomain
        (:attr:`~orpheus.numerics.frame.FrameBase.basis_space`) re-DRESSES
        that space with the discrete PARSEVAL inverse (the head axis
        re-weighted, the frame its generator) — the dressed space, not this
        one, is the ONE moment space the tree binds (the operator ends, the
        moment fields, the Hilbert-adjoint machinery — ruling R-6.2c-1,
        2026-09-08); this space's own metric stays the continuum /
        cross-Gram vocabulary of ``project``/``gram``. The
        basis owns exactly one space (the nodal/domain space comes from the
        measure), so the unqualified name is unambiguous — matching the
        ``Field.space`` convention. The :class:`FrameBase` re-exposes it provenance-
        qualified as ``frame.basis_space`` (beside ``frame.measure_space``).
        """
        ...
