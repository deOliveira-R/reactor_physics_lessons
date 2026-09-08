r"""The Legendre basis :math:`\{P_\ell\}` on the orbit space :math:`S^2/O(2)_a` — THE FIX for ERR-080.

The defect (#429, ERR-080)
==========================

``solve_sn(scattering_order >= 2)`` on any 1-D chart returned a wrong answer
(`[M]` an infinite medium read :math:`-3.7647` against an analytic
:math:`+4.0`) because a 1-D rule's frame bound the FULL real spherical
harmonics to a measure whose nodes had been forged onto :math:`S^2` as
:math:`(\mu, 0, 0)`: a point of :math:`[-1, 1]` is an ORBIT of the
:math:`SO(2)` action, not a point of the sphere, and the basis read the
fabricated azimuth as a real one. The numbers were never wrong — a MEAN (the
orbit barycentre, `[M]` :math:`1.1\times10^{-16}` from :math:`(\mu,0,0)`) was
handed to a basis that needs a POINT.

The repair, in the domain's own terms
=====================================

A 1-D rule's measure lives on :math:`S^2/O(2)_a` (tracker 2.4 declares it;
named by its stabiliser since #432), and the functions on an orbit space are
the :math:`H`-invariant functions on the base — for :math:`O(2)_a`
(equivalently its rotation half, which has the same orbits) the **trivial
isotypic component** of the
degree-:math:`\ell` harmonics, which is one-dimensional in every degree
(Schur; `[M]` 2026-09-02, a rank test about every axis) and is spanned
downstairs by :math:`P_\ell(\mu)`, :math:`\mu = \Omega\cdot\hat e_a`. So the
basis a 1-D rule binds is THIS one: :math:`L+1` members, a FLAT coefficient
space, no fabricated slots to zero. The isomorphism with the upstairs
realization — the :math:`m = 0` column of the spherical-harmonic table — is
:class:`~orpheus.numerics.basis.descent.Descent`'s witness, and it is exact
at the BIT: :math:`Y_\ell^0(\Omega) = P_\ell(\Omega\cdot\hat e_x)` under the
no-prefactor convention, and :meth:`LegendreBasis.evaluate` spells the
polynomial exactly as ``_evaluate_real_sh`` spells that column.

⚠ **The spelling is a measured constraint, not a taste** (`[M]` the
verification memo's H-1, re-derived by the archivist 2026-09-02): no single
``scipy`` routine reproduces the column bit-for-bit — ``lpmv(0, 1, μ)``
differs from the input array at :math:`\ell = 1` by :math:`8\times10^{-17}`
(GL8; :math:`1.1\times10^{-16}` on GL16, LS4, Lebedev) and ``eval_legendre``
differs at :math:`\ell \ge 2` by up to :math:`4.8\times10^{-16}` over
``gauss_legendre(2, 4, 8, 16)`` at :math:`L \le 4`. The column is
:math:`P_0 = 1`, :math:`P_1 = \mu` (the input), :math:`P_\ell = ` ``lpmv(0, ℓ, μ)``
for :math:`\ell \ge 2`, and so is this table. With it the slab's flux at
:math:`L \le 1` is ``array_equal`` across the repair; with pure ``lpmv`` the
TABLE moves by :math:`4.4\times10^{-16}` and the converged FLUX on
ERR-080's own fixture by :math:`2.8\times10^{-14}` at :math:`L = 1` — a
tolerance where a bit-identity claim was available.

Conventions
===========

* the addition-theorem (canonical-dual) factor is :math:`2\ell+1`, as for the
  harmonics — the reconstruction :math:`R = (2\ell+1)\,P_\ell` restricted to
  the descended column IS the spherical-harmonic reconstruction restricted to
  :math:`m = 0`;
* the continuum Gram is :math:`4\pi/(2\ell+1)`: the Gram against the
  **pushforward** :math:`\pi_* d\Omega = 2\pi\,d\mu`
  (:attr:`~orpheus.numerics.manifold.Quotient.reference`), which coincides
  with the harmonics' — the descent is an isometry. ⚠ NOT the bare Legendre
  mass-2 value :math:`2/(2\ell+1)`;
* the discrete Gram over a rule is DIAGONAL wherever the rule is exact to
  degree :math:`2L`. ⭐ A theorem for the Gauss–Legendre family (`[M]` 12 of
  12 rows): ``GL_n``'s Legendre Gram is diagonal-and-exact for
  :math:`L \le n-1` and has a structurally DEAD slot at :math:`\ell = n`
  (:math:`P_n` vanishes at its own roots), so a 1-D Gauss frame at
  :math:`L \ge n` is rank-deficient — the frame's DENSE arm installs the
  pseudo-inverse metric there exactly as for an over-resolved sphere frame
  (user-ruled 2026-09-02).

Both coordinate systems of the orbit space are accepted by :meth:`evaluate`:
the realization's (``(N,)`` / ``(N, 1)`` values of :math:`\mu` — the slab
rule's own nodes) and the base's (``(N, 3)`` unit directions — a full-sphere
rule, pulled back along the entry's
:attr:`~orpheus.numerics.manifold.Quotient.quotient_map`, so
:math:`P_\ell(\Omega\cdot\hat e_a)` is a legitimate basis on a Lebedev or
level-symmetric rule — user-ruled 2026-09-02).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import lpmv

from orpheus.numerics.basis.base import Basis, GramStructure

if TYPE_CHECKING:
    from orpheus.numerics.manifold import Manifold, Quotient
    from orpheus.numerics.measure import DiscreteMeasure
    from orpheus.numerics.spaces.legendre_space import LegendreSpace


__all__ = ["LegendreBasis", "legendre_table"]


def legendre_table(L: int, mu: NDArray) -> NDArray:
    r"""The ``(N, L+1)`` table :math:`P_\ell(\mu_n)`, spelled as the spherical-harmonic
    :math:`m = 0` column spells it (module docstring): ``1.0``, the input, then
    ``lpmv(0, ℓ, μ)``. Class-free math, kept free so the descent witness can
    call the same spelling the basis does."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    table = np.zeros((mu.size, L + 1))
    table[:, 0] = 1.0
    if L >= 1:
        table[:, 1] = mu
    for ell in range(2, L + 1):
        table[:, ell] = lpmv(0, ell, mu)
    return table


@dataclass(frozen=True)
class LegendreBasis(Basis):
    r"""Legendre polynomials :math:`\{P_\ell\}_{\ell \le L}` on :math:`S^2/O(2)_a`.

    Parameters
    ----------
    L : int
        Maximum degree retained. Negative ``L`` is rejected.
    axis : str, default ``"x"``
        The axis of the spent stabiliser :math:`O(2)_a` — the polar axis
        the cosine :math:`\mu` is measured against. The slab and sphere
        geometries spend :math:`O(2)_x`.

    Notes
    -----
    Frozen dataclass — equality and hashing are by ``(L, axis)``. Its
    :attr:`domain` is the catalogue ENTRY :math:`S^2/O(2)_a`, so
    :attr:`~orpheus.numerics.basis.base.Basis.invariance_group` answers
    ``O2(axis)`` by derivation (tracker 2.1b) — the FULL group these
    functions have, since a mirror through the polar axis does not move
    :math:`\mu`. That is what lets the frame's G0 admit this basis on a
    :math:`\sigma_b`-folded rule, :math:`b \ne a` (#432, 2026-09-02; until
    the entry was named by its stabiliser the derived answer was the lower
    bound ``SO2(axis)`` and that pairing was over-refused). The table is
    FLAT: ``(N, L+1)``.
    """

    L: int
    axis: str = "x"

    def __post_init__(self) -> None:
        if self.L < 0:
            raise ValueError(
                f"LegendreBasis: L must be non-negative, got L={self.L}"
            )
        if self.axis not in ("x", "y", "z"):
            raise ValueError(
                f"LegendreBasis: axis must be x/y/z, got {self.axis!r}."
            )

    # ── Convention-bearing properties ────────────────────────────────

    @cached_property
    def addition_theorem_factor(self) -> NDArray:
        r"""The :math:`(2\ell+1)` array, shape ``(L+1,)`` — the canonical-dual factor."""
        return 2.0 * np.arange(self.L + 1) + 1.0

    @cached_property
    def metric_per_ell(self) -> NDArray:
        r"""The pushforward Gram :math:`4\pi/(2\ell+1)` per degree, shape ``(L+1,)`` (module docstring)."""
        return 4.0 * np.pi / self.addition_theorem_factor

    # ── Tabulation ────────────────────────────────────────────────────

    def evaluate(self, points: NDArray, /) -> NDArray:
        r"""Tabulate :math:`P_\ell` at points of the orbit space — in EITHER of its coordinate systems.

        Parameters
        ----------
        points : NDArray
            ``(N,)`` or ``(N, 1)`` — values of :math:`\mu`, the realization's
            coordinate (a 1-D rule's own nodes); or ``(N, 3)`` — unit
            directions of the base, pulled back along the entry's quotient
            map (a full-sphere rule). Any other width is refused, naming
            both.

        Returns
        -------
        NDArray, shape ``(N, L+1)``
            ``table[n, ℓ]`` is :math:`P_\ell(\mu_n)`.
        """
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            mu = arr.reshape(-1)
            if not self.domain.contains(mu):
                raise ValueError(
                    f"LegendreBasis.evaluate: the points are not on the orbit "
                    f"space {self.domain.name} (a cosine must lie in [-1, 1])."
                )
        elif arr.ndim == 2 and arr.shape[1] == 3:
            base = self.domain.base
            if not base.contains(arr):
                raise ValueError(
                    f"LegendreBasis.evaluate: {arr.shape[0]} directions were "
                    f"offered as points of {base.name} and not all lie on it "
                    f"(|Ω| off 1 by more than the membership tolerance)."
                )
            mu = self.domain.quotient_map(arr)
        else:
            raise ValueError(
                f"LegendreBasis.evaluate: expected points of "
                f"{self.domain.name} in one of its two coordinate systems — "
                f"(N,) / (N, 1) values of mu, or (N, 3) unit directions of "
                f"the base pulled back along the quotient map; got shape "
                f"{arr.shape}."
            )
        return legendre_table(self.L, mu)

    # ── Table contractions (the Frame caches the table, delegates here) ───

    def synthesize(self, coefficients: NDArray, table: NDArray, /) -> NDArray:
        r"""Naked synthesis :math:`S_0(c)_n = \sum_\ell P_\ell(\mu_n)\, c_\ell`."""
        return np.einsum("nl,l...->n...", table, coefficients)

    def analyze(self, values: NDArray, table: NDArray, weights: NDArray, /) -> NDArray:
        r"""Analysis :math:`(M\psi)_\ell = \sum_n w_n P_\ell(\mu_n)\,\psi_n`."""
        return np.einsum("n,nl,n...->l...", weights, table, values)

    def analyze_transpose(
        self, coefficients: NDArray, table: NDArray, weights: NDArray, /,
    ) -> NDArray:
        r"""Representation transpose :math:`(M^\top c)_n = w_n \sum_\ell P_\ell(\mu_n)\, c_\ell`."""
        return np.einsum("n,nl,l...->n...", weights, table, coefficients)

    def reconstruct(self, coefficients: NDArray, table: NDArray, /) -> NDArray:
        r"""Reconstruction :math:`(R\phi)_n = \sum_\ell (2\ell+1) P_\ell(\mu_n)\, \phi_\ell`."""
        return np.einsum(
            "nl,l,l...->n...", table, self.addition_theorem_factor, coefficients,
        )

    def reconstruct_transpose(self, values: NDArray, table: NDArray, /) -> NDArray:
        r"""Representation transpose :math:`(R^\top v)_\ell = (2\ell+1) \sum_n P_\ell(\mu_n)\, v_n`."""
        return np.einsum(
            "nl,l,n...->l...", table, self.addition_theorem_factor, values,
        )

    # ── Gram ──────────────────────────────────────────────────────────

    @property
    def gram_structure(self) -> GramStructure:
        r"""DIAGONAL — the Legendre polynomials are orthogonal on the orbit space."""
        return GramStructure.DIAGONAL

    def mass_matrix(self, measure: "DiscreteMeasure", /) -> NDArray:
        r"""Discrete Gram :math:`\sum_n w_n P_\ell(\mu_n) P_{\ell'}(\mu_n)` over a rule, shape ``(L+1, L+1)``."""
        table = self.evaluate(measure.nodes)
        return np.einsum("n,nl,nL->lL", measure.weights, table, table)

    # ── The domain (what they eat) and the coefficient space (what they span) ──

    @property
    def domain(self) -> "Quotient":
        r"""The catalogue entry :math:`S^2/O(2)_a` — a Legendre polynomial eats an ORBIT of the axis's stabiliser, a constant-:math:`\mu` circle."""
        from orpheus.numerics.manifold import SPHERE
        from orpheus.numerics.symmetry import SubgroupOfO3

        return SPHERE.quotient(SubgroupOfO3.O2(self.axis))

    @property
    def space(self) -> "LegendreSpace":
        r"""The :class:`~orpheus.numerics.spaces.legendre_space.LegendreSpace` of degree :math:`L` this basis spans (lazy import: the space imports this basis)."""
        from orpheus.numerics.spaces.legendre_space import LegendreSpace

        return LegendreSpace.for_basis(self)

    def at_order(self, L_new: int, /) -> "LegendreBasis":
        r"""This family about the same spent axis, cut at degree ``L_new``."""
        return replace(self, L=L_new)
