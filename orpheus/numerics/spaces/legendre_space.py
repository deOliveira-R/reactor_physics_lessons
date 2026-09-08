r"""Function space for Legendre moment coefficients on an :math:`O(2)_a` orbit space.

The coefficient space of :class:`~orpheus.numerics.basis.legendre_basis.LegendreBasis`
— :math:`\{P_\ell(\mu)\}_{\ell \le L}` on :math:`S^2/O(2)_a` — a **FLAT** head of
shape ``(L+1,)``. The sibling of
:class:`~orpheus.numerics.spaces.spherical_harmonic_space.SphericalHarmonicSpace`
in the :class:`~orpheus.numerics.spaces.moment_head.MomentHead` family, and the
second member of it: the first family has a rectangular head, this one has
one coefficient per degree, and the moment carriers read which through the
protocol rather than assuming either (#429 tracker 3.4, 2026-09-02).

The continuum metric it carries is :math:`4\pi/(2\ell+1)` — the Gram of
:math:`P_\ell` against the **pushforward** of :math:`d\Omega` along the
quotient map, :math:`\pi_* d\Omega = 2\pi\,d\mu` (Archimedes' hat-box;
:attr:`~orpheus.numerics.manifold.Quotient.reference`):
:math:`\int_{-1}^{1} P_\ell^2 \, 2\pi\,d\mu = 4\pi/(2\ell+1)`. It coincides
exactly with the spherical-harmonic
:attr:`~orpheus.numerics.basis.spherical_harmonic_basis.SphericalHarmonicBasis.metric_per_ell`,
which is what makes the descent :math:`\{Y_\ell^0\} \cong \{P_\ell\}` an
ISOMETRY and not merely an isomorphism. ⚠ Not the bare ``LEGENDRE`` mass-2
normalisation :math:`2/(2\ell+1)`: that is a factor :math:`2\pi` away and
would move every operator end's metric (`[M]` the verification memo §3).
"""
from __future__ import annotations

from dataclasses import dataclass

from orpheus.numerics.axis import BasisKind, LegendreAxis
from orpheus.numerics.basis.legendre_basis import LegendreBasis
from orpheus.numerics.space import FunctionSpace
from orpheus.numerics.spaces.moment_head import truncated_head

__all__ = ["LegendreSpace"]


@dataclass(frozen=True)
class LegendreSpace(FunctionSpace):
    r"""Function space of Legendre moment coefficients up to degree :math:`L` on :math:`S^2/O(2)_a`.

    Parameters
    ----------
    name : str
        Inherited. Convention: ``"legendre_space(S^2/O2_<axis>)"`` — the
        orbit space's own name, READ off the basis's domain by
        :meth:`from_L` rather than spelled a second time, because two axes
        are two spaces (the tree carries two poles; tracker 2.4).
    shape : tuple[int, ...]
        Inherited. MUST equal ``(L + 1,)``; ``__post_init__`` checks.
    axes : tuple[LegendreAxis], optional
        ONE :class:`~orpheus.numerics.axis.LegendreAxis` (CS4c step 6 item
        6.2c-ii) whose measure is the per-degree continuum Gram
        :math:`4\pi/(2\ell+1)` (module docstring) and whose identity
        carries the spent axis; a frame's dressed ``basis_space``
        re-weights it with the Parseval inverse exactly as for the
        spherical-harmonic space. The legacy ``inner_product_weights`` slot
        stays ``None``.
    L : int, default 0
        Maximum degree retained.
    spent_axis : str, default ``"x"``
        The axis of the spent stabiliser :math:`O(2)_a` (``axis`` is a
        :class:`FunctionSpace` method, hence the longer name).

    Notes
    -----
    Equality and hashing are STRUCTURAL, inherited from
    :class:`FunctionSpace` (the identity flip, CS4c step 6): the head axis
    — family, order, measure AND the spent axis — is the identity, so
    ``from_L(1, "x")`` and ``from_L(1, "z")`` are two spaces although
    their measures are ``array_equal`` (`[M]` hazard H-10).
    """

    L: int = 0
    spent_axis: str = "x"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.spent_axis not in ("x", "y", "z"):
            raise ValueError(
                f"LegendreSpace: spent_axis must be x/y/z, got {self.spent_axis!r}."
            )
        expected = (self.L + 1,)
        if self.shape != expected:
            raise ValueError(
                f"LegendreSpace: shape={self.shape} inconsistent with "
                f"L={self.L}; expected shape={expected} (one coefficient "
                f"per degree — the flat head)."
            )

    def __eq__(self, other: object) -> bool:
        return FunctionSpace.__eq__(self, other)

    def __hash__(self) -> int:
        return FunctionSpace.__hash__(self)

    @classmethod
    def for_basis(cls, basis: LegendreBasis) -> "LegendreSpace":
        r"""The coefficient space ``basis`` spans — THE mint of the Legendre
        head (CS4c step 6 item 6.2c-ii): one :class:`~orpheus.numerics.axis.LegendreAxis`
        carrying the basis's continuum Gram and its spent axis, ``basis``
        itself as the axis's generator. :attr:`LegendreBasis.space`
        delegates here; :meth:`from_L` is the ``(L, axis)`` sugar.
        """
        L = basis.L
        head_axis = LegendreAxis(
            "legendre",
            (L + 1,),
            basis.metric_per_ell,
            kind=BasisKind.MODAL,
            generator=basis,
            spent_axis=basis.axis,
        )
        return cls(
            name=f"legendre_space({basis.domain.name})",
            shape=(L + 1,),
            axes=(head_axis,),
            L=L,
            spent_axis=basis.axis,
        )

    @classmethod
    def from_L(cls, L: int, axis: str = "x") -> "LegendreSpace":
        r"""The canonical Legendre space for degree :math:`L` about ``axis``
        — :meth:`for_basis` over ``LegendreBasis(L=L, axis=axis)``.

        The metric is sourced from :class:`LegendreBasis` so the
        :math:`4\pi/(2\ell+1)` formula lives in exactly one place.
        """
        return cls.for_basis(LegendreBasis(L=L, axis=axis))

    # ── the MomentHead surface ───────────────────────────────────────

    @property
    def isotropic_slot(self) -> tuple[int, ...]:
        r"""``(0,)`` — the :math:`\ell = 0` coefficient of a flat head."""
        return (0,)

    def degree_block(self, l: int, /) -> tuple[int | slice, ...]:
        r"""``(l,)`` — the degree-:math:`\ell` block of a flat head is one coefficient."""
        if not 0 <= l <= self.L:
            raise ValueError(
                f"LegendreSpace.degree_block: l={l} out of range [0, {self.L}]."
            )
        return (l,)

    def truncated(self, L_new: int, /) -> "FunctionSpace":
        r"""This family's space at the lower order ``L_new``, about the same
        axis, under this head's own name — re-minted by the head axis's
        generator (:func:`~orpheus.numerics.spaces.moment_head.truncated_head`)."""
        return truncated_head(self, L_new)

    # ── delegated convention (single source in the basis) ──────────

    @property
    def basis(self) -> LegendreBasis:
        """The associated :class:`LegendreBasis`, determined by ``(L, axis)``."""
        return LegendreBasis(L=self.L, axis=self.spent_axis)
