r"""The angular HEAD of a moment space — the surface every moment carrier reads.

A moment field's space is ``<head> ⊗ cells``: the head is the coefficient
space of the basis the quadrature's frame bound (#429 tracker 2.5, 2026-09-02).
Two families ship. The real spherical harmonics' head is the rectangular
``(L+1, 2L+1)`` table with the addition-theorem-shifted ``[l + m]`` column and
zero padding outside :math:`|m| \le \ell`; the Legendre basis on
:math:`S^2/O(2)_a` has a FLAT head, ``(L+1,)`` — one coefficient per degree,
because the trivial isotypic component of :math:`O(2)_a` (equivalently of its
rotation half, which has the same orbits) is one-dimensional in every degree
(`[M]` 2026-09-02, a rank test about every axis).

Until #429's fused commit every consumer that indexed ``values[0, 0]`` or
sliced ``values[l, :2l+1]`` read the FIRST family's layout as if it were the
contract. It was the family's: on a flat head ``values[0, 0]`` is group 0's
spatial slice and raises nothing (`[M]` the verification memo's H-15 —
``scalar_flux``, ``isotropic_part``, ``anisotropic_part``, ``l_block``, the
fission ℓ=0 dyad, and the material field's per-degree contraction, which
spelled the m-axis into its einsum). So the LAYOUT is the head's to say —
which index tuple is the isotropic slot, which selects the degree-:math:`\ell`
block, how many leading axes the head owns (``len(shape)``), and how it
truncates within its own family — and a consumer READS it rather than
assuming it.

Structural (``Protocol``) and ``runtime_checkable``: the two space classes
satisfy it by carrying the members, and a consumer holding ``space.factors[0]``
narrows with ``isinstance`` — the same idiom as
:class:`~orpheus.numerics.basis.base.TruncatedBasis` on the basis side.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from orpheus.numerics.space import FunctionSpace

__all__ = ["MomentHead", "truncated_head"]


@runtime_checkable
class MomentHead(Protocol):
    r"""The head factor of a moment space: a truncated family that knows its own layout."""

    @property
    def L(self) -> int:
        """The truncation order — the degree the family is cut at."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """The head's own axes (``(L+1, 2L+1)`` rectangular, ``(L+1,)`` flat)."""
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def isotropic_slot(self) -> tuple[int, ...]:
        r"""The index tuple of the :math:`\ell = 0` coefficient within the head's axes."""
        ...

    def degree_block(self, l: int, /) -> tuple[int | slice, ...]:
        r"""The index tuple selecting the degree-:math:`\ell` block within the head's axes."""
        ...

    def truncated(self, L_new: int, /) -> "FunctionSpace":
        """This family's space at the lower order ``L_new``, under this head's own name."""
        ...


def truncated_head(head: "FunctionSpace", L_new: int, /) -> "FunctionSpace":
    r"""This family's head at the lower order ``L_new``, under ``head``'s own
    name — re-minted by the head AXIS's GENERATOR (CS4c step 6 item 6.2c-ii,
    ruling O-3: *re-mint AND re-axis, never slice*).

    An axis-built head has one axis, and that axis remembers what minted it:

    * a :class:`~orpheus.numerics.basis.base.TruncatedBasis` (the head is
      the basis's own continuum coefficient space, :attr:`Basis.space
      <orpheus.numerics.basis.base.Basis.space>`) re-spans the family at
      ``L_new`` — same functions, same spent axis — and hands back ITS
      space;
    * a :class:`~orpheus.numerics.frame.GalerkinFrame` (the head is the
      frame's Parseval-dressed :attr:`~orpheus.numerics.frame.FrameBase.basis_space`,
      the ONE moment space the tree binds) re-poses itself at ``L_new`` and
      hands back its dressed space at that order — the metric is the
      frame's to install at EVERY order, because the discrete Gram's
      verdict can FLIP with :math:`L` (`[M]` ``folded_product(2,4)`` is
      DENSE at :math:`L = 2` and DIAGONAL at :math:`L = 1`), so a slice of
      the parent's dressing is undefined there.

    The head keeps its identity (its name) and only its order moves; a
    truncated moment field therefore lands on the SAME space its carrier
    mints at ``L_new`` (structurally equal — ruling O-5), never on a
    second-metric twin.
    """
    from orpheus.numerics.basis.base import TruncatedBasis
    from orpheus.numerics.frame import GalerkinFrame

    if not isinstance(head, MomentHead):
        raise TypeError(
            f"truncated_head: {type(head).__name__} is not a moment head "
            f"(no truncation order / layout surface)."
        )
    if not 0 <= L_new <= head.L:
        raise ValueError(
            f"{type(head).__name__}.truncated: L_new={L_new} must lie in "
            f"[0, {head.L}]."
        )
    axes = head.axes
    if axes is None or len(axes) != 1:
        raise TypeError(
            f"truncated_head: a moment head is a single-axis space; "
            f"{head!r} carries axes={axes!r}."
        )
    generator = axes[0].generator
    if isinstance(generator, GalerkinFrame):
        lower = generator.at_order(L_new).basis_space
    elif isinstance(generator, TruncatedBasis):
        lower = generator.at_order(L_new).space
    else:
        got = type(generator).__name__ if generator is not None else "None"
        raise ValueError(
            f"truncated_head: the head axis of {head.name!r} was minted "
            f"through neither a truncated basis nor a Galerkin frame "
            f"(generator={got}), so nothing can re-mint the family at "
            f"L_new={L_new}."
        )
    return replace(lower, name=head.name)
