r"""Real spherical-harmonic basis on :math:`S^2` truncated at degree :math:`L`.

Naming: for :math:`Y_\ell^m`, :math:`\ell` is the **degree** and :math:`m` the
**order** — so the truncation parameter :math:`L` is the maximum *degree*, not
the order. (The transport idiom "P_L scattering order" is a separate concept —
there "order" qualifies the Legendre expansion, not the SH azimuthal index.)

The canonical home of the :math:`Y_\ell^m` evaluator and of the Gram matrix
:math:`g_C = \mathrm{diag}(4\pi/(2\ell+1))` of ERR-039 fame. Pre-frame the
evaluator lived as a free ``evaluate_real_sh`` function and the Gram literal
:math:`(2\ell+1)` was carried as a raw array on the since-retired
harmonic-reconstruction operator. Both move here under a single typed class —
the SH convention, the addition-theorem factor, and the discrete Gram now have
one home (the frame's reconstruction face reads the factor live from
:attr:`addition_theorem_factor`).

Convention
==========

The project uses the **no-:math:`4\pi/(2\ell+1)`-prefactor** normalisation
of the real spherical harmonics, in which the addition theorem reads

.. math::
   :label: real-sh-addition-theorem

   \sum_{m=-\ell}^{\ell} Y_\ell^m(\hat\Omega)\,Y_\ell^m(\hat\Omega')
   \;=\; P_\ell(\hat\Omega \cdot \hat\Omega'),

so the :math:`P_\ell`-scattering reconstruction takes the form

.. math::

   q_n \;=\; \sum_{\ell=0}^{L} (2\ell+1) \sum_m Y_\ell^m(\hat\Omega_n)\,
            \phi^{\ell m},

and the discrete inner product against an exact quadrature satisfies

.. math::
   :label: sh-mass-matrix-diagonal

   \sum_n w_n \, Y_\ell^m(\hat\Omega_n) \, Y_{\ell'}^{m'}(\hat\Omega_n)
   \;=\; \frac{4\pi}{2\ell+1} \, \delta_{\ell\ell'} \delta_{m m'}.

The polar axis is :math:`\mu_x` (so :math:`\cos\theta = \mu_x`,
:math:`\sin\theta = \sqrt{1-\mu_x^2}`); azimuth is measured in the
:math:`(\mu_y,\mu_z)` plane:

.. math::

   \cos\phi \;=\; \frac{\mu_y}{\sin\theta}, \qquad
   \sin\phi \;=\; \frac{\mu_z}{\sin\theta}.

The :math:`\ell\le 1` branch is hard-coded to bit-identical values for
the legacy :math:`P_0/P_1` regression tests:

.. math::

   Y_0^0 = 1, \quad Y_1^{-1} = \mu_z, \quad Y_1^0 = \mu_x,
   \quad Y_1^{+1} = \mu_y.

For :math:`\ell \ge 2` the formula uses :func:`scipy.special.lpmv` with
the Condon–Shortley phase :math:`(-1)^m` removed and norm
:math:`\sqrt{2(\ell-m)!/(\ell+m)!}` for :math:`m \ne 0`.

Cross-method consumers
======================

This module is generic infrastructure: every method that integrates an
angular field against the spherical-harmonic basis consumes it.

* **SN scattering** (:mod:`orpheus.transport.operators.scattering`) — the
  :math:`Y^* W` projection that builds the per-ordinate :math:`P_\ell`
  source.
* **PN solver** (future, §10 of the architecture report) —
  spherical-harmonic moment basis is the native space.
* **MC adjoint moments** — variance reduction with response moments
  built against :math:`Y_\ell^m`.
* **Energy-condensation diagnostics** — when within-group anisotropy
  needs an angular characterisation.

References
----------

* Bell, G. I. and Glasstone, S. (1970). *Nuclear Reactor Theory*.
  Van Nostrand Reinhold. §1.6 (real spherical harmonics in transport).
* Lewis, E. E. and Miller, W. F. Jr. (1993). *Computational Methods
  of Neutron Transport*. ANS. §4.7 (:math:`P_\ell` scattering Galerkin
  reconstruction with the :math:`(2\ell+1)` factor).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from math import factorial, sqrt
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import lpmv

from orpheus.numerics.basis.base import Basis, GramStructure

if TYPE_CHECKING:
    from orpheus.numerics.manifold import Manifold, Quotient
    from orpheus.numerics.measure import DiscreteMeasure
    from orpheus.numerics.spaces.spherical_harmonic_space import (
        SphericalHarmonicSpace,
    )


__all__ = ["MirrorEvenSphericalHarmonicBasis", "SphericalHarmonicBasis"]


# ─────────────────────────────────────────────────────────────────────
# Basis class
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SphericalHarmonicBasis(Basis):
    r"""Real spherical harmonics on :math:`S^2`, truncated at degree :math:`L`.

    The first concrete :class:`~orpheus.numerics.basis.base.Basis` — the
    synthesis (trial) side of the spherical-harmonic frame. Implements the
    three fundamental basis operations (:meth:`evaluate`, :meth:`synthesize`,
    :meth:`mass_matrix`) and carries the SH convention (the no-prefactor
    normalisation, the addition-theorem factor :math:`2\ell+1`, the continuous
    Gram diagonal :math:`4\pi/(2\ell+1)`).

    Parameters
    ----------
    L : int
        Maximum harmonic degree retained. ``L == 0`` returns the :math:`P_0`
        table only. Negative ``L`` is rejected.

    Notes
    -----
    Frozen dataclass — equality and hashing are by ``L`` alone (the basis
    IS its truncation degree; two ``SphericalHarmonicBasis(L=2)`` instances
    are the same basis).

    The basis carries the convention (the no-prefactor normalisation
    documented in the module docstring) and the addition-theorem factor
    :math:`2\ell+1`. The continuous (theoretical) mass-matrix diagonal
    :math:`4\pi/(2\ell+1)` is the :meth:`metric_per_ell` property; the
    DISCRETE mass matrix against a quadrature is computed by
    :meth:`discrete_mass_matrix`. The two agree to within the
    quadrature's degree of exactness.
    """

    L: int

    def __post_init__(self) -> None:
        if self.L < 0:
            raise ValueError(
                f"SphericalHarmonicBasis: L must be non-negative, got L={self.L}"
            )

    # ── Convention-bearing properties ────────────────────────────────

    @cached_property
    def addition_theorem_factor(self) -> NDArray:
        r"""The :math:`(2\ell+1)` array, shape ``(L+1,)``.

        Used by the addition-theorem reconstruction
        :math:`R = (2\ell+1) Y` and equal to
        :math:`4\pi \cdot g_C^{-1}` where :math:`g_C` is the SH Gram matrix.
        """
        return 2.0 * np.arange(self.L + 1) + 1.0

    @cached_property
    def live_slot_mask(self) -> NDArray:
        r"""``(L+1, 2L+1)`` bool — ``True`` on the :math:`|m| \le \ell` slots, ``False`` on the padding.

        The rectangular table's own layout, stated once: column ``l + m``
        holds :math:`Y_\ell^m`, so row :math:`\ell` is live on columns
        ``0 .. 2l`` and identically zero beyond. Consumers that restrict a
        table or a mask to REAL slots (the descent's upstairs face, a
        slot count) read this rather than re-deriving the rule.
        """
        cols = np.arange(2 * self.L + 1)
        rows = np.arange(self.L + 1)
        return cols[None, :] <= 2 * rows[:, None]

    @cached_property
    def metric_per_ell(self) -> NDArray:
        r"""The Gram-matrix diagonal :math:`4\pi/(2\ell+1)` per :math:`\ell`, shape ``(L+1,)``.

        This is the THEORETICAL (continuous-:math:`L^2`-on-:math:`S^2`)
        metric under the project's no-prefactor SH convention. The
        DISCRETE counterpart against a quadrature is the diagonal of
        :meth:`discrete_mass_matrix`; the two agree iff the quadrature
        is exact for :math:`Y_\ell^m Y_{\ell'}^{m'}` of degree
        :math:`\ell + \ell' \le 2L`.
        """
        return 4.0 * np.pi / self.addition_theorem_factor

    # ── Tabulation ────────────────────────────────────────────────────

    def evaluate(self, directions: NDArray) -> NDArray:
        r"""Tabulate :math:`Y_\ell^m(\hat\Omega_n)` at the given direction set.

        Parameters
        ----------
        directions : NDArray, shape ``(N, 3)``
            Direction cosines :math:`(\mu_x, \mu_y, \mu_z)` per ordinate.

        Returns
        -------
        NDArray, shape ``(N, L+1, 2L+1)``
            ``Y[n, l, l+m]`` is :math:`Y_\ell^m(\hat\Omega_n)` under the
            no-prefactor convention; entries outside :math:`|m| \le \ell`
            are zero.
        """
        directions = np.asarray(directions)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(
                f"SphericalHarmonicBasis.evaluate expects directions of shape "
                f"(N, 3); got {directions.shape}."
            )
        if not self.domain.contains(directions):
            # 0.6 (#429): a real spherical harmonic eats a POINT of S^2. The
            # forged (mu, 0, 0) ordinates of a 1-D rule — ERR-080's
            # construction — have |Omega| = |mu| < 1 off the poles and read a
            # fabricated azimuth as a real one; they are refused here, in
            # kind, rather than tabulated wrong.
            off = int(np.sum(np.abs(np.linalg.norm(directions, axis=1) - 1.0) > 1e-12))
            raise ValueError(
                f"SphericalHarmonicBasis.evaluate: {off} of "
                f"{directions.shape[0]} directions are not points of S^2 "
                f"(|Omega| off 1 by more than 1e-12). A real spherical "
                f"harmonic eats a unit direction; a 1-D rule's mu is a point "
                f"of S^2/O(2)_a, whose basis is the Legendre basis "
                f"(orpheus.numerics.basis.legendre_basis) — ERR-080."
            )
        return _evaluate_real_sh(
            self.L, directions[:, 0], directions[:, 1], directions[:, 2],
        )

    def evaluate_from_components(
        self,
        mu_x: NDArray,
        mu_y: NDArray,
        mu_z: NDArray,
    ) -> NDArray:
        r"""Tabulate from separated direction-cosine arrays.

        Equivalent to :meth:`evaluate` but accepts three :math:`(N,)`
        arrays instead of one :math:`(N, 3)` array. Provided for
        :class:`~orpheus.numerics.quadrature.Quadrature` and other
        per-component consumers (the SN quadratures historically expose
        ``mu_x`` / ``mu_y`` / ``mu_z`` as separate attributes).

        Parameters
        ----------
        mu_x, mu_y, mu_z : NDArray, shape ``(N,)``
            Direction-cosine arrays.

        Returns
        -------
        NDArray, shape ``(N, L+1, 2L+1)``
            Same layout as :meth:`evaluate`.
        """
        return self.evaluate(np.column_stack([mu_x, mu_y, mu_z]))

    # ── Mass matrix ───────────────────────────────────────────────────

    def mass_matrix(self, measure: "DiscreteMeasure") -> NDArray:
        r"""Discrete Gram matrix :math:`\sum_n w_n Y_\ell^m Y_{\ell'}^{m'}` over a quadrature.

        Computes the :math:`(L+1, 2L+1, L+1, 2L+1)` 4-tensor

        .. math::

            g[\ell, m, \ell', m']
            \;=\; \sum_n w_n \, Y_\ell^m(\hat\Omega_n) \, Y_{\ell'}^{m'}(\hat\Omega_n).

        For an exact quadrature of degree :math:`\ge 2L` this equals

        .. math::

            \mathrm{diag}\!\left(\frac{4\pi}{2\ell+1}\right)
            \delta_{\ell\ell'} \delta_{m m'}

        per :eq:`sh-mass-matrix-diagonal`, agreeing with
        :attr:`metric_per_ell` along the diagonal and vanishing
        off-diagonal. The match-to-:attr:`metric_per_ell` is the
        ERR-039 test gate pinned by
        ``tests/numerics/test_spherical_harmonic_space.py``.

        Parameters
        ----------
        measure : DiscreteMeasure
            Angular quadrature on :math:`S^2`. ``measure.nodes`` MUST be
            a ``(N, 3)`` array of direction cosines; ``measure.weights``
            the :math:`(N,)` quadrature weights.

        Returns
        -------
        NDArray, shape ``(L+1, 2L+1, L+1, 2L+1)``
            Full 4-tensor Gram matrix. Off-diagonal entries that vanish
            under an exact quadrature are bit-zero only at exact arithmetic;
            small FP residuals are expected and used by tests as a
            quadrature-exactness diagnostic.
        """
        Y = self.evaluate(measure.nodes)
        return np.einsum("n,nlm,nLM->lmLM", measure.weights, Y, Y)

    # ── Table contractions (the Frame caches the table, delegates here) ───

    def synthesize(self, coefficients: NDArray, table: NDArray) -> NDArray:
        r"""Naked synthesis :math:`S_0(c)_n = \sum_{\ell, m} Y_\ell^m(\hat\Omega_n)\, c_\ell^m`.

        The bare reconstruction with NO :math:`(2\ell+1)` factor and NO
        :math:`w_n` weight — the pure synthesis :math:`S_0` (the frame-theory
        synthesis operator :math:`T^*`). The three weighted contractions
        (:meth:`analyze`, :meth:`analyze_transpose`, :meth:`reconstruct`) are each
        this kernel post-multiplied by ONE diagonal, but kept as separate fused
        einsums for 0-ULP bit-identity (FP non-associativity).

        Parameters
        ----------
        coefficients : NDArray, shape ``(L+1, 2L+1, ...)``
            Moment-space input; entries outside :math:`|m| \le \ell` are zero by
            construction (dotted with zero ``table`` entries).
        table : NDArray, shape ``(N, L+1, 2L+1)``
            The :math:`Y_\ell^m(\hat\Omega_n)` table from :meth:`evaluate`.

        Returns
        -------
        NDArray, shape ``(N, ...)`` — :math:`S_0(c)` per ordinate.
        """
        return np.einsum("nlm,lm...->n...", table, coefficients)

    def analyze(
        self, values: NDArray, table: NDArray, weights: NDArray,
    ) -> NDArray:
        r"""Analysis :math:`(M\psi)_\ell^m = \sum_n w_n\, Y_\ell^m(\hat\Omega_n)\,\psi_n`.

        The W-weighted Galerkin projection of a per-ordinate field onto harmonic
        moments — the frame's *analysis operator* :math:`T`. ONE fused
        :func:`numpy.einsum` over the ordinate axis; trailing axes broadcast.
        """
        return np.einsum("n,nlm,n...->lm...", weights, table, values)

    def analyze_transpose(
        self, coefficients: NDArray, table: NDArray, weights: NDArray,
    ) -> NDArray:
        r"""Representation transpose :math:`(M^\top c)_n = w_n \sum_{\ell m} Y_\ell^m(\hat\Omega_n)\, c_\ell^m`.

        The matrix transpose of :meth:`analyze` (:math:`= w_n \cdot S_0`) — NOT the
        Hilbert adjoint. The metric-aware ``AdjointOperator`` combines it with the
        frame's metrics (measure :math:`w_n` on the domain; the F-0 Parseval
        :math:`G^{-1}` on the codomain) to give the physical
        :math:`M^* = S_0 \circ G^{-1} = R/W`, so the Frame's analysis face gets
        ``.H`` for free.
        """
        return np.einsum("n,nlm,lm...->n...", weights, table, coefficients)

    def reconstruct(self, coefficients: NDArray, table: NDArray) -> NDArray:
        r"""Reconstruction :math:`(R\phi)_n = \sum_\ell (2\ell+1) \sum_m Y_\ell^m(\hat\Omega_n)\, \phi_\ell^m`.

        The addition-theorem (canonical-dual) synthesis — :math:`S_0` weighted by
        the dual factor :math:`2\ell+1 = 4\pi\, g_C^{-1}`, read live from
        :attr:`addition_theorem_factor` (no stored copy). Measure-free.
        """
        return np.einsum(
            "nlm,l,lm...->n...", table, self.addition_theorem_factor, coefficients,
        )

    def reconstruct_transpose(self, values: NDArray, table: NDArray) -> NDArray:
        r"""Representation transpose :math:`(R^\top v)_\ell^m = (2\ell+1) \sum_n Y_\ell^m(\hat\Omega_n)\, v_n`.

        The matrix transpose of :meth:`reconstruct` (:math:`= 2\ell+1` times the
        naked analysis :math:`S_0^\top`) — NOT the Hilbert adjoint, and **measure-free**:
        no :math:`w_n` is baked in (symmetric with :meth:`reconstruct`, asymmetric
        with :meth:`analyze_transpose`, whose forward bakes the weights in). The
        metric-aware ``AdjointOperator`` combines it with the codomain (measure
        :math:`w_n`) and domain (the F-0 Parseval :math:`G^{-1}`, entering the
        sandwich through its pseudo-inverse :math:`G`) metrics to give the
        physical Hilbert adjoint :math:`(R^* v)_\ell^m = d_\ell G_\ell \sum_n w_n
        Y_\ell^m(\hat\Omega_n)\, v_n = W\,(M v)_\ell^m` (the SH identity
        :math:`d_\ell G_\ell = 4\pi = W`), so the Frame's reconstruction face gets
        ``.H`` for free.
        """
        return np.einsum(
            "nlm,l,n...->lm...", table, self.addition_theorem_factor, values,
        )

    # ── Gram structure: orthogonal harmonics ⟹ diagonal Gram ──────────────
    @property
    def gram_structure(self) -> GramStructure:
        r"""DIAGONAL — the real spherical harmonics are orthogonal (:math:`g_C` diagonal)."""
        return GramStructure.DIAGONAL

    # ── The domain (what they eat) and the coefficient space (what they span) ──

    @property
    def domain(self) -> "Manifold":
        r""":math:`S^2` — a real spherical harmonic eats a unit DIRECTION.

        The constant answer for every degree :math:`L`: truncating the degree
        changes what the basis SPANS, never what its functions are defined on.

        ⭐ This is the property ERR-080 needed and did not have.  The defect is
        a 1-D quadrature handing :meth:`evaluate` the forged ordinates
        :math:`(\mu, 0, 0)`, which satisfy :math:`\lVert\Omega\rVert = 1`
        only at :math:`\mu = \pm 1` — so they are not points of this domain,
        and ``domain.contains(directions)`` says so
        (:doc:`/theory/verification/error_catalog`).  Since 2026-09-02 that
        refusal IS wired into :meth:`evaluate` (#429 tracker 0.6), and the
        frame's G0 refuses the pairing one level up.
        """
        from orpheus.numerics.manifold import SPHERE

        return SPHERE

    @property
    def space(self) -> "SphericalHarmonicSpace":
        r"""The :class:`SphericalHarmonicSpace` of degree :math:`L` this basis spans.

        Axis-built (CS4c step 6 item 6.2c-ii): one
        :class:`~orpheus.numerics.axis.HarmonicAxis` whose measure is the
        continuum Gram :math:`g_C = \mathrm{diag}(4\pi/(2\ell+1))` on the
        padded layout, this basis its generator. The spherical-harmonic
        :class:`~orpheus.numerics.frame.GalerkinFrame` re-dresses it with the
        discrete Parseval inverse (:attr:`~orpheus.numerics.frame.FrameBase.basis_space`)
        — that dressed head is the moment space the tree binds. Lazy import:
        ``SphericalHarmonicSpace`` imports this basis, so a top-level import
        would cycle.
        """
        from orpheus.numerics.spaces.spherical_harmonic_space import (
            SphericalHarmonicSpace,
        )
        return SphericalHarmonicSpace.for_basis(self)

    def at_order(self, L_new: int, /) -> "SphericalHarmonicBasis":
        r"""This family (the same class — the σ-even restriction stays σ-even)
        cut at degree ``L_new``."""
        return replace(self, L=L_new)




@dataclass(frozen=True)
class MirrorEvenSphericalHarmonicBasis(SphericalHarmonicBasis):
    r"""The σ-EVEN subspace of the degree-:math:`L` real SH basis — the
    QUOTIENT's basis (Q5.6).

    On a measure folded by a coordinate mirror :math:`\sigma_a`, the
    σ-odd harmonics are *not in the quotient's function space*: their
    discrete moments are garbage, not zero (`[M]` the ξ-carrying
    :math:`l = 1` slot reads :math:`+6.49` for a FLAT flux on the
    σ_y-folded product rule, where the full rule cancels to
    :math:`10^{-16}` — the scattering kernel's raw :math:`Y^{\mathsf T}W`
    analysis has no Gram division anywhere to absorb it).  The σ-EVEN
    sub-basis stays **exactly orthogonal on the quotient** (even × even
    products are even, which the fold integrates exactly), so the
    restriction is the whole fix.

    The restriction keeps the parent's rectangular
    :math:`(L+1, 2L+1)` layout and structurally ZEROES the σ-odd table
    columns — the same mechanism that already zeroes the
    :math:`|m| > \ell` padding, so every consumer (the ``nlm``
    einsums, the m-blind :math:`\Lambda`, the moment shapes, the
    fixed-slot reads) flows through unchanged with the odd moments
    coming out as EXACT ``0.0``.

    The per-slot parity is **DERIVED**, never hand-listed — and since
    2026-09-02 (#429 tracker 3.4, user-ruled) it is derived by the
    ENTRY: :meth:`Quotient.descending_slots
    <orpheus.numerics.manifold.Quotient.descending_slots>` asks which
    slots of the parent are constant on the fibres of the fold's own
    quotient map (σ-even = constant on the orbit :math:`\{\Omega,
    \sigma\Omega\}`), the same probe :class:`~orpheus.numerics.basis.descent.Descent`
    reads for every entry; the private five-direction probe this class
    carried is retired (`[M]` the mask is bit-identical through the entry,
    15 of 15 (axis, L) rows). The hand rule is chart-subtle — this basis
    measures its azimuth FROM :math:`\mu_y`, so the σ_y-odd set mixes the
    cos and sin branches ({cos, m odd} ∪ {sin, |m| even}) and any "mask the
    sin branch" shortcut would zero the wrong functions (ERR-072's
    declared-not-computed family).

    Parameters
    ----------
    L : int
        Maximum harmonic degree (inherited).
    mirror_axis : int
        The coordinate index the fold's mirror negates (0 = x, 1 = y,
        2 = z); ``SubgroupOfO3.mirror_axis`` supplies it.
    """

    mirror_axis: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.mirror_axis not in (0, 1, 2):
            raise ValueError(
                f"MirrorEvenSphericalHarmonicBasis: mirror_axis must be "
                f"0, 1 or 2; got {self.mirror_axis!r}."
            )

    @property
    def domain(self) -> "Quotient":
        r""":math:`S^2/\sigma_a` — the QUOTIENT, not the sphere.

        This is the whole content of the class stated as a type.  The parent's
        functions eat a direction; these eat a direction **modulo the mirror**,
        because every one of them takes the same value at :math:`\Omega` and at
        :math:`\sigma_a\Omega` (that is what σ-even MEANS).  A function
        constant on the orbits of :math:`H` is a function on :math:`M/H`.

        Derived through :meth:`~orpheus.numerics.manifold.Manifold.quotient`
        rather than tagged, so it is the *same* object the folded measure
        carries — `[M]` ``S^2/sigma_y``, matching the support a
        ``folded_product`` rule's angular frame reports.  ⭐ That agreement is
        the point: before the two sides could name a manifold, a fold basis on
        an unfolded measure and a fold basis on a folded one were
        indistinguishable, and the G0 well-posedness gate the plan has carried
        since it was written had no operands to compare.
        """
        from orpheus.numerics.manifold import SPHERE
        from orpheus.numerics.symmetry import AXIS_LETTER, SubgroupOfO3

        return SPHERE.quotient(SubgroupOfO3.Mirror(AXIS_LETTER[self.mirror_axis]))

    @cached_property
    def even_slot_mask(self) -> NDArray:
        r"""``(L+1, 2L+1)`` float mask — 1.0 on σ-even slots, 0.0 on σ-odd.

        READ off the fold's catalogue entry — the slots of the parent
        harmonics that DESCEND to :math:`S^2/\sigma_a` (constant on the
        fibres of its quotient map; :meth:`Quotient.descending_slots
        <orpheus.numerics.manifold.Quotient.descending_slots>`). Padding
        slots (:math:`|m| > \ell`, identically zero) descend vacuously and
        stay unmasked — inert, the parent already zeroes them.
        """
        return self.domain.descending_slots(
            SphericalHarmonicBasis(L=self.L)
        ).astype(float)

    def evaluate(self, directions: NDArray) -> NDArray:
        table = super().evaluate(directions)
        return table * self.even_slot_mask[None, :, :]

    def evaluate_from_components(
        self,
        mu_x: NDArray,
        mu_y: NDArray,
        mu_z: NDArray,
    ) -> NDArray:
        table = super().evaluate_from_components(mu_x, mu_y, mu_z)
        return table * self.even_slot_mask[None, :, :]


# ─────────────────────────────────────────────────────────────────────
# Algorithm body (private — preserved bit-identical from
# legacy spherical_harmonics.evaluate_real_sh)
# ─────────────────────────────────────────────────────────────────────


def _evaluate_real_sh(
    L: int,
    mu_x: NDArray,
    mu_y: NDArray,
    mu_z: NDArray,
) -> NDArray:
    r"""Implementation body of :meth:`SphericalHarmonicBasis.evaluate`.

    A free function because it is class-free math (an
    :math:`(L, \Omega) \mapsto Y` table): it was carved out so the
    now-retired back-compat shim ``orpheus.numerics.spherical_harmonics``
    could re-export it without instantiating a class, and it stays free
    because :meth:`SphericalHarmonicBasis.evaluate` and
    :meth:`SphericalHarmonicBasis.evaluate_from_components` are both
    thin callers of it. Algorithm preserved bit-identical to the legacy
    ``evaluate_real_sh`` so the snapshots at
    ``tests/sn/regression/snapshots/`` continue to pass.

    See the module docstring for the convention and citations.
    """
    mu_x = np.asarray(mu_x)
    mu_y = np.asarray(mu_y)
    mu_z = np.asarray(mu_z)
    N = len(mu_x)
    if L < 0:
        return np.zeros((N, 0, 0))
    Y = np.zeros((N, L + 1, 2 * L + 1))

    Y[:, 0, 0] = 1.0
    if L == 0:
        return Y

    Y[:, 1, 0] = mu_z   # m = -1
    Y[:, 1, 1] = mu_x   # m =  0
    Y[:, 1, 2] = mu_y   # m = +1
    if L == 1:
        return Y

    cos_theta = mu_x
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))
    on_axis = sin_theta < 1e-15
    safe_st = np.where(on_axis, 1.0, sin_theta)
    cos_phi = np.where(on_axis, 1.0, mu_y / safe_st)
    sin_phi = np.where(on_axis, 0.0, mu_z / safe_st)
    phi = np.arctan2(sin_phi, cos_phi)

    for l in range(2, L + 1):
        Y[:, l, l] = lpmv(0, l, cos_theta)  # m = 0: P_l(μ_x)
        for m in range(1, l + 1):
            P_lm = lpmv(m, l, cos_theta)
            sign = (-1.0) ** m   # remove Condon–Shortley phase
            norm = sqrt(2.0 * factorial(l - m) / factorial(l + m))
            cos_mphi = np.cos(m * phi)
            sin_mphi = np.sin(m * phi)
            Y[:, l, l + m] = sign * norm * P_lm * cos_mphi   # m > 0
            Y[:, l, l - m] = sign * norm * P_lm * sin_mphi   # m < 0
    return Y
