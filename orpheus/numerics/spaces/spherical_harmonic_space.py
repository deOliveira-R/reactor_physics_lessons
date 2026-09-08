r"""Function space for spherical-harmonic moment coefficients.

This module ships the typed home of the ERR-039 Gram matrix
:math:`g_C = \mathrm{diag}(4\pi/(2\ell+1))`. Pre-frame the SH Gram lived as
a free literal (the ``two_l_plus_one`` array, which is
:math:`4\pi \cdot g_C^{-1}` wearing a disguise) on the since-retired
harmonic-reconstruction operator, and as a prose warning on the moment
projection's representation transpose. It now lives here: the space is
AXIS-BUILT (CS4c step 6 item 6.2c-ii) — one
:class:`~orpheus.numerics.axis.HarmonicAxis` whose MEASURE is the metric —
and the metric formula itself is sourced from
:class:`~orpheus.numerics.basis.SphericalHarmonicBasis` so the
:math:`(2\ell+1)` literal exists in exactly one place.

What this enables
=================

The discrete spherical-harmonic :class:`~orpheus.numerics.frame.GalerkinFrame`'s
codomain is this space **re-dressed by the frame** (F-0,
``frame_square_recarve.md``): :meth:`SphericalHarmonicSpace.from_L` carries the
CONTINUUM Gram :math:`g_C = 4\pi/(2\ell+1)` as its head axis's measure, and
:attr:`FrameBase.basis_space <orpheus.numerics.frame.FrameBase.basis_space>`
re-weights that axis with the PARSEVAL metric — the inverse of the frame's
discrete Gram, :math:`(2\ell+1)/4\pi` on a degree-exact sphere rule (a
positioned matrix pseudo-inverse where the Gram is dense) — because the
carried moments are COVARIANT (:math:`\varphi = Gc`) and only :math:`G^{-1}`
makes analysis an isometry onto its image (`[M]` ``scratch/probe_f1_parseval.py``,
2026-08-24: continuum-side Parseval ratio 118.7 vs 1.000). Since CS4c step 6
item 6.2c-ii (ruling R-6.2c-1, 2026-09-08: *the carrier's norm is the
field's energy*) that dressed head is the ONE moment space the tree binds
— the carrier's cached moment space, every moment field, every operator
end — and the two heads are structurally UNEQUAL (the measure enters the
identity), so the metric-blind seam that let them pass for one is gone.
The generic ``AdjointOperator`` machinery then computes ``frame.analysis.H``
as the physical :math:`S_0 \circ G^{-1} = R/W` with no bespoke code.

ERR-039 in one sentence: the addition-theorem reconstruction
:math:`R = (2\ell+1) S_0` and the analysis face's Hilbert adjoint
:math:`\Pi^* = R/W` differ by exactly the total weight :math:`W = 4\pi`. With
the metric carried on the space and the basis providing the convention, they
fall out as two derived expressions from a common ground.

References
----------

* Grand Report v3 §5.3 — Space hierarchy.
* Grand Report v3 §6.3 — ``.T`` (representation transpose) vs ``.H``
  (Hilbert adjoint).
* Grand Report v3 §19 — Harmonic projection.
* :mod:`orpheus.numerics.basis.spherical_harmonic_basis` — the basis,
  the SH convention, and the source of
  :attr:`SphericalHarmonicSpace.metric_per_ell`.
* ERR-039 entry: ``docs/theory/verification/error_catalog.rst``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from orpheus.numerics.axis import BasisKind, HarmonicAxis
from orpheus.numerics.basis.spherical_harmonic_basis import SphericalHarmonicBasis
from orpheus.numerics.space import FunctionSpace
from orpheus.numerics.spaces.moment_head import truncated_head


__all__ = ["SphericalHarmonicSpace"]


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _padded_metric_tensor(L: int, metric_per_ell: NDArray) -> NDArray:
    r"""Broadcast the per-:math:`\ell` metric to the padded ``(L+1, 2L+1)`` storage layout.

    The :class:`~orpheus.transport.fields.harmonic_moment_flux.HarmonicMomentFlux`
    storage convention is to allocate ``(L+1, 2L+1)`` slots with the
    addition-theorem-shifted :math:`m`-index ``[l + m]`` — entries
    outside :math:`|m| \le \ell` are zero by construction. The
    inner-product weights MUST share that layout so the broadcast
    :math:`\sum_{\ell m} w_{\ell m} \, x_{\ell m} \, y_{\ell m}` reduces
    correctly to :math:`\sum_\ell (4\pi/(2\ell+1)) \sum_{m=-\ell}^{\ell}
    x_\ell^m y_\ell^m` (the padded slots contribute zero either way).

    Parameters
    ----------
    L : int
        Maximum harmonic degree.
    metric_per_ell : NDArray, shape ``(L+1,)``
        The :math:`4\pi/(2\ell+1)` diagonal per :math:`\ell`. Sourced
        from :attr:`SphericalHarmonicBasis.metric_per_ell` so the formula
        lives in exactly one place.

    Returns
    -------
    NDArray, shape ``(L+1, 2L+1)``
        Padded metric: row :math:`\ell` holds ``metric_per_ell[ell]`` in
        the :math:`2\ell+1` valid slots ``[0, 2*ell]``, zero elsewhere.
    """
    cols = np.arange(2 * L + 1)
    rows = np.arange(L + 1)
    valid_mask = cols[None, :] <= 2 * rows[:, None]   # (L+1, 2L+1) bool
    return np.where(valid_mask, metric_per_ell[:, None], 0.0)


# ─────────────────────────────────────────────────────────────────────
# Space class
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SphericalHarmonicSpace(FunctionSpace):
    r"""Function space of real-spherical-harmonic moment coefficients up to degree :math:`L`.

    Parameters
    ----------
    name : str
        Inherited from :class:`FunctionSpace`. Convention:
        ``"spherical_harmonic_space"``.
    shape : tuple[int, ...]
        Inherited from :class:`FunctionSpace`. MUST equal
        ``(L + 1, 2 * L + 1)``; ``__post_init__`` checks.
    axes : tuple[HarmonicAxis], optional
        Inherited from :class:`FunctionSpace`. ONE
        :class:`~orpheus.numerics.axis.HarmonicAxis` (CS4c step 6 item
        6.2c-ii) whose measure is the padded ``(L+1, 2L+1)`` metric tensor,
        zero in the :math:`|m| > \ell` padding. WHICH metric depends on the
        instance's provenance: :meth:`from_L` installs the CONTINUUM Gram
        (row :math:`\ell` holds :math:`4\pi/(2\ell+1)`, the basis its
        generator), while a frame's dressed ``basis_space`` — built from
        this same class via :func:`dataclasses.replace` — carries the
        PARSEVAL inverse (:math:`(2\ell+1)/4\pi` on a degree-exact rule;
        the frame its generator; F-0, :attr:`FrameBase.basis_space
        <orpheus.numerics.frame.FrameBase.basis_space>`). The legacy
        ``inner_product_weights`` slot stays ``None``.
    L : int, default 0
        Maximum harmonic degree retained. Must satisfy
        ``shape == (L + 1, 2 * L + 1)``.

    Notes
    -----
    Frozen dataclass with the same subclassing pattern as
    :class:`~orpheus.numerics.spaces.angular_trace_space.AngularTraceSpace`. The default
    ``L = 0`` is required by dataclass-inheritance rules: it follows
    :attr:`FunctionSpace.inner_product_weights` which has a default,
    so every subsequent field must also have one.

    Equality and hashing are STRUCTURAL (inherited from
    :class:`FunctionSpace` since the identity flip, CS4c step 6): the head
    axis — family, order, measure — is the identity, so a frame-dressed
    head and the continuum head of the same order are two spaces, and two
    dressed heads over one pairing are one.
    """

    L: int = 0

    def __post_init__(self) -> None:
        # The base guards first (one-metric-source exclusivity + metric
        # admission — P7): an override that skips them silently opts the
        # whole subclass out of the construction contract.
        super().__post_init__()
        expected = (self.L + 1, 2 * self.L + 1)
        if self.shape != expected:
            raise ValueError(
                f"SphericalHarmonicSpace: shape={self.shape} inconsistent with "
                f"L={self.L}; expected shape={expected}."
            )

    # ── Equality / hashing inherited from FunctionSpace ───────────────
    #
    # FunctionSpace defines explicit structural __eq__ and __hash__ (the
    # axes when both sides carry them). The @dataclass(frozen=True)
    # decorator on this subclass would otherwise auto-generate its own
    # __eq__ / __hash__ that compare ALL fields (including ndarray-bearing
    # ones) — and ndarray equality returns an array, raising at use time.
    # Explicitly delegating keeps the structural identity; ``L`` is
    # already encoded in the head axis's shape.

    def __eq__(self, other: object) -> bool:
        return FunctionSpace.__eq__(self, other)

    def __hash__(self) -> int:
        return FunctionSpace.__hash__(self)

    # ── Constructor ──────────────────────────────────────────────────

    @classmethod
    def for_basis(cls, basis: SphericalHarmonicBasis) -> "SphericalHarmonicSpace":
        r"""The coefficient space ``basis`` spans — THE mint of the harmonic
        head (CS4c step 6 item 6.2c-ii): one :class:`~orpheus.numerics.axis.HarmonicAxis`
        carrying the basis's continuum Gram on the padded layout, and
        ``basis`` itself as the axis's generator (provenance is the object
        that spans the head — the σ-even restriction's head is generated by
        the σ-even basis, not by a plain-harmonic re-mint of its order).
        :attr:`SphericalHarmonicBasis.space` delegates here; :meth:`from_L`
        is the order-only sugar over the plain family.
        """
        L = basis.L
        shape = (L + 1, 2 * L + 1)
        head_axis = HarmonicAxis(
            "harmonic",
            shape,
            _padded_metric_tensor(L, basis.metric_per_ell),
            kind=BasisKind.MODAL,
            generator=basis,
        )
        return cls(
            name="spherical_harmonic_space",
            shape=shape,
            axes=(head_axis,),
            L=L,
        )

    @classmethod
    def from_L(cls, L: int) -> "SphericalHarmonicSpace":
        r"""Construct the canonical SH space for truncation degree :math:`L`
        — the plain real-harmonic family's head, :meth:`for_basis` over
        ``SphericalHarmonicBasis(L=L)``.

        Builds the metric tensor from :class:`SphericalHarmonicBasis` so
        the :math:`(2\ell+1)` / :math:`4\pi/(2\ell+1)` formulas live in
        exactly one place (the basis).

        Parameters
        ----------
        L : int
            Maximum harmonic degree. Must be non-negative.

        Returns
        -------
        SphericalHarmonicSpace
            With ``name="spherical_harmonic_space"``,
            ``shape=(L+1, 2L+1)``, and ONE :class:`~orpheus.numerics.axis.HarmonicAxis`
            whose measure is the padded :math:`4\pi/(2\ell+1)` CONTINUUM
            Gram and whose generator is the basis. (A frame's
            ``basis_space`` is this object re-dressed with the discrete
            Parseval inverse — :attr:`FrameBase.basis_space
            <orpheus.numerics.frame.FrameBase.basis_space>`, F-0 — the
            moment space the tree binds.)
        """
        return cls.for_basis(SphericalHarmonicBasis(L=L))

    # ── Delegated properties (single source of truth in the basis) ───

    @property
    def basis(self) -> SphericalHarmonicBasis:
        r"""The associated :class:`SphericalHarmonicBasis`.

        Uniquely determined by :attr:`L`; constructed on access.
        """
        return SphericalHarmonicBasis(L=self.L)

    @property
    def metric_per_ell(self) -> NDArray:
        r"""The :math:`4\pi/(2\ell+1)` diagonal per :math:`\ell`, shape ``(L+1,)``.

        Delegated to :attr:`SphericalHarmonicBasis.metric_per_ell` so the
        SH convention's formula has a single canonical home.
        """
        return self.basis.metric_per_ell

    def truncated(self, L_new: int) -> "FunctionSpace":
        r"""This family's space at the lower order ``L_new`` — the head a moment field truncates TO.

        The moment carrier truncates by asking its angular HEAD factor for
        the same family one order down, never by re-minting a family from
        an integer (#429 tracker 2.5): a Legendre head truncates to a
        Legendre head, a spherical-harmonic head to this. Re-minted by the
        head axis's GENERATOR (:func:`~orpheus.numerics.spaces.moment_head.truncated_head`,
        CS4c step 6 item 6.2c-ii): the basis re-spans the continuum head at
        ``L_new``; a frame re-DRESSES its Parseval head at ``L_new`` — the
        metric is the frame's to install at every order, and the truncated
        head is structurally the carrier's own space at ``L_new``. The head
        keeps its identity (its name) and only its order moves.
        """
        return truncated_head(self, L_new)

    # ── the MomentHead surface (orpheus.numerics.spaces.moment_head) ──

    @property
    def isotropic_slot(self) -> tuple[int, ...]:
        r"""``(0, 0)`` — the :math:`(\ell, m) = (0, 0)` slot of the rectangular head."""
        return (0, 0)

    def degree_block(self, l: int, /) -> tuple[int | slice, ...]:
        r"""``(l, 0:2l+1)`` — the degree-:math:`\ell` block: row :math:`\ell`, its :math:`2\ell+1` live columns."""
        if not 0 <= l <= self.L:
            raise ValueError(
                f"SphericalHarmonicSpace.degree_block: l={l} out of range "
                f"[0, {self.L}]."
            )
        return (l, slice(0, 2 * l + 1))

    @property
    def addition_theorem_factor(self) -> NDArray:
        r"""The :math:`(2\ell+1)` factor per :math:`\ell`, shape ``(L+1,)``.

        Delegated to :attr:`SphericalHarmonicBasis.addition_theorem_factor` —
        the canonical source of the addition-theorem factor that the discrete
        spherical-harmonic :class:`~orpheus.numerics.frame.GalerkinFrame`'s
        reconstruction face reads.
        """
        return self.basis.addition_theorem_factor
