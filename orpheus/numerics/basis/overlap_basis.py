r"""Fractional-overlap trial basis — the non-nested generalization of IndicatorBasis.

:class:`OverlapBasis` carries a precomputed **fractional** membership table
:math:`T[g,G]\in[0,1]` (a partition of unity, rows summing to 1) instead of the
point-sampled one-hot table of
:class:`~orpheus.numerics.basis.indicator_basis.IndicatorBasis`. A fine cell that
straddles a coarse boundary belongs *fractionally* to each coarse cell it overlaps —
the conservative re-binning that a NON-nested condensation needs (see
:meth:`orpheus.data.energy_grid.EnergyGrid.overlap_to`). The one-hot
:class:`IndicatorBasis` is the **nested degenerate** (no straddles → every
:math:`T[g,G]\in\{0,1\}`).

Why a subclass (and why this is a no-op extension, not a new arm)
================================================================

Every table contraction on :class:`IndicatorBasis` — :meth:`analyze`,
:meth:`reconstruct`, :meth:`synthesize`, the transposes, and the diagonal Gram a
:class:`~orpheus.numerics.frame.FrameBase` builds via
``analysis(reconstruction(ones))`` — is a **pure function of the membership table**
and never assumes it was one-hot. So the fractional generalization reuses ALL of
them unchanged; only the table *production* differs. :class:`OverlapBasis` overrides
exactly one method, :meth:`evaluate`, to return the precomputed overlap table rather
than bucketing points with ``searchsorted``. The rate-preserving projection then
falls out of the existing ``frame.project = G⁻¹ M`` because a partition-of-unity
table gives ``reconstruction(ones) = 1`` → the diagonal Gram :math:`\Phi_G =
\sum_g T[g,G]\,\varphi_g`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from orpheus.numerics.basis.base import GramStructure
from orpheus.numerics.basis.indicator_basis import IndicatorBasis
from orpheus.numerics.manifold import Manifold

__all__ = ["OverlapBasis"]


@dataclass(frozen=True, eq=False)
class OverlapBasis(IndicatorBasis):
    r"""Cell-indicator basis carrying a precomputed FRACTIONAL membership table.

    Parameters
    ----------
    edges_per_axis : tuple[NDArray, ...]
        The coarse-index partition (inherited from :class:`IndicatorBasis`), used
        only for the coarse cell count :attr:`n_cells` and the coefficient
        :attr:`space` — the membership table, not these edges, is the source of
        truth (:meth:`evaluate` is overridden). For the energy axis this is a
        length-1 tuple of the coarse-group index edges.
    overlap_table : NDArray, shape ``(n_fine, n_coarse)``
        The precomputed partition-of-unity table :math:`T[g,G]\in[0,1]`,
        ``rows.sum(axis=1) == 1``. One-hot recovers :class:`IndicatorBasis`.

    Notes
    -----
    Frozen, ``eq=False`` (identity equality; the fields are NumPy arrays).

    The inherited :meth:`~orpheus.numerics.basis.indicator_basis.IndicatorBasis.mass_matrix`
    docstring claims a **diagonal** Gram ("the indicators have disjoint support") —
    that claim does **not** hold for a fractional table (a straddling row makes two
    columns share support, so the cross Gram is non-diagonal). It is latent: no
    consumer calls ``mass_matrix`` (the frame's :meth:`~orpheus.numerics.frame.FrameBase.project`
    uses the partition-of-unity row-sum probe, not the full Gram — see
    :attr:`~orpheus.numerics.frame.FrameBase.gram_inverse`). A future least-squares consumer
    that needs the dense Gram must compute it for the fractional case, not trust the
    inherited diagonal claim.
    """

    overlap_table: NDArray
    #: The manifold the fine ROWS of :attr:`overlap_table` index — what these
    #: functions EAT (the fine partition a condensation projects FROM), as
    #: opposed to :attr:`partition_of`, the coarse partition they SPAN. ⭐
    #: Until 2026-09-02 (#429, the frame's G0) this class inherited
    #: ``domain = partition_of`` and so declared that it ate the COARSE
    #: nodes while :meth:`evaluate` validates the FINE row count — a basis
    #: whose domain named the wrong end of its own map. Nothing compared the
    #: two until G0 did: `[M]` every non-degenerate ``Mixture.condense``
    #: raised on the first G0 (45 tests in 3 files), the day a frame checked
    #: that its two halves name one point set. ``kw_only`` because
    #: :class:`IndicatorBasis`'s positional fields precede it.
    fine: Manifold = field(kw_only=True)

    @property
    def domain(self) -> Manifold:
        r"""The FINE partition — a fractional-overlap function eats a fine-group node
        (its table's rows) and spans the coarse groups (:attr:`partition_of`, its
        columns). The two ends of the map are two manifolds, and this is the source."""
        return self.fine

    @classmethod
    def from_indicator(
        cls, indicator: IndicatorBasis, overlap_table: NDArray, /, *, fine: Manifold,
    ) -> "OverlapBasis":
        r"""Decorate a (nested) :class:`IndicatorBasis` with a fractional membership table.

        The fractional trial **is** the target grid's basis-view (e.g.
        :meth:`~orpheus.data.energy_grid.EnergyGrid.as_basis`) — the coarse-cell geometry
        (:attr:`edges_per_axis` → :attr:`n_cells`, :attr:`space`) and its
        :attr:`~orpheus.numerics.basis.indicator_basis.IndicatorBasis.partition_of`
        manifold, both taken from ``indicator`` so the decorated basis cannot
        drift from the grid it decorates — carrying a precomputed
        partition-of-unity ``overlap_table`` in place of the one-hot membership
        :meth:`evaluate` would compute from those edges. A one-hot table recovers the plain
        ``indicator`` (the nested degenerate). This is the canonical constructor — the
        binary :meth:`~orpheus.data.energy_grid.EnergyGrid.overlap_to` calls it — so a call
        site reads "the trial is the target basis-view, mismatch-corrected" rather than
        reaching into the indicator's ``edges_per_axis``. ``fine`` is the
        manifold the table's ROWS index — the partition the functions EAT —
        which is not the indicator's (that is the coarse one they span).
        """
        return cls(
            edges_per_axis=indicator.edges_per_axis,
            partition_of=indicator.partition_of,
            overlap_table=overlap_table,
            fine=fine,
        )

    # ── Gram structure: a straddling row shares ≥2 columns ⟹ NOT diagonal ──
    @property
    def gram_structure(self) -> GramStructure:
        r"""PARTITION_OF_UNITY — fractional rows sum to 1, so :math:`R\mathbf 1=\mathbf 1`.

        Overrides the inherited :attr:`IndicatorBasis.gram_structure` DIAGONAL: a fine
        row straddling a coarse boundary populates ≥2 columns, so the cross Gram
        :math:`MR` is non-diagonal. The :meth:`~orpheus.numerics.frame.FrameBase.project`
        row-sum probe stays valid because the rows are a partition of unity (see
        :attr:`~orpheus.numerics.frame.FrameBase.gram_inverse`), NOT because the Gram is diagonal.
        """
        return GramStructure.PARTITION_OF_UNITY

    def evaluate(self, points: NDArray, /) -> NDArray:
        r"""Return the precomputed ``(n_fine, n_coarse)`` partition-of-unity table.

        Interval overlap is computed once when the table is built (it depends on the
        fine AND coarse partitions plus the within-group weight, not on sample
        points), so this returns it directly. ``points`` (the fine-group nodes) are
        accepted for the :class:`~orpheus.numerics.basis.base.Basis` contract and
        validated against the table's fine-row count.
        """
        pts = np.asarray(points)
        n_rows = int(self.overlap_table.shape[0])
        if pts.shape[0] != n_rows:
            raise ValueError(
                f"OverlapBasis table has {n_rows} fine rows but got "
                f"{pts.shape[0]} sample points."
            )
        return self.overlap_table

    # ── Table diagnostics (the provenance of the non-nested re-binning) ────
    @cached_property
    def dominant_column(self) -> NDArray:
        r"""The DOMINANT coarse column of each fine row — ``argmax`` of :attr:`overlap_table`.

        ``(n_fine,)``. For a NESTED (one-hot) table this is the exact containing-coarse
        map; for a straddling fine row it is the coarse cell receiving the largest
        fraction (a reporting view — the real apportionment is the full table). For the
        energy axis this is the dominant coarse group of each fine group (the former
        ``GroupCondensation.coarse_of_fine``).
        """
        return self.overlap_table.argmax(axis=1).astype(int)

    @cached_property
    def fractional_columns(self) -> NDArray:
        r"""Coarse columns that received a STRICTLY-FRACTIONAL contribution.

        ``(k,)`` indices of coarse cells/groups where some fine row straddled the
        boundary and contributed a weight strictly between 0 and 1 — i.e. where the
        re-binning leaned on the within-group model rather than a clean (one-hot)
        nesting. Empty for a nested table (pure rate-preserving collapse, no
        assumption); non-empty where the coarse partition is locally finer than the fine
        one. For the energy axis this is the former ``GroupCondensation.locally_interpolated``
        — the data-vs-assumption provenance of the downsampling/interpolation asymmetry.
        """
        frac = (self.overlap_table > 1e-12) & (self.overlap_table < 1.0 - 1e-12)
        return np.nonzero(frac.any(axis=0))[0]
