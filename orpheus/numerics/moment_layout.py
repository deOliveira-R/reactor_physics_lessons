r"""Physics-free spatial-moment layout policy (tensor-Legendre Kronecker).

L1 primitive (mathematics, knows no neutrons) — the moment-axis sibling of
:mod:`orpheus.numerics.face_layout`. The physics-free moment-axis primitives
every spatial-moment consumer keys on live HERE, in exactly one place: the
slot-0 cell/face **average** index (:data:`AVERAGE_MOMENT`), the "append a
trailing moment axis iff there is more than one moment" **tail** policy
(:func:`face_moment_tail`), and the rank-based "is this buffer moment-valued?"
discriminator (:func:`is_moment_valued_by_rank`).

Why ``numerics`` and not ``sn.sweep`` (#245)
==============================================

These conventions describe the Kronecker ordering of a tensor-Legendre
DG basis; they carry no transport physics (no :math:`\Sigma`, no
:math:`\mu`). Both halves of the spatial-moment iterate need them:

* the scheme-owned spatial-moment AXIS
  (:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`,
  the CAPABILITY half — the typed factor every widened space composes,
  labelled :data:`SPATIAL_MOMENT_AXIS_LABEL`, its cell tail sized by
  :func:`spatial_moment_tail`; until CS4c step 6 item 6.2c-iii the
  harmonic-moment product carried a separate Euclidean
  ``SpatialMomentSpace`` class here instead — two spellings of one
  factor, retired), and
* the UBLD cell assembler :mod:`orpheus.transport.spatial._ubld` (the
  REALIZATION half — buffers, sweeps, the face cochain).

``numerics`` sits *below* the transport/method layers, so homing the
policy here lets both import it in the correct (downward) direction. Previously the constants
lived in ``sn.spatial._ubld`` and ``SpatialMomentSpace`` reached UP into
``sn.spatial`` for them via a *deferred* (call-time) import — a band-aid
over a layering inversion (a ``numerics`` space depending on the SN
package). Relocating here removes the deferral and the inversion; the
SN module re-exports these names downward (the
:data:`orpheus.numerics.face_layout.AXIS_NAMES` precedent) so SN
consumers keep importing them next to the UBLD primitives they name.

This module is leaf — no ``orpheus`` imports (numpy enters only under
``TYPE_CHECKING``, for the rank-predicate type hints) — so importing it can
never re-introduce a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


__all__ = [
    "AVERAGE_MOMENT",
    "SPATIAL_MOMENT_AXIS_LABEL",
    "spatial_moment_tail",
    "cell_moment_count",
    "face_moment_count",
    "face_moment_tail",
    "is_moment_valued_by_flat_rank",
    "is_moment_valued_by_rank",
]


#: Index of the cell/face AVERAGE moment in the tensor-Legendre Kronecker
#: layout (``[bar, …]``): the all-``P₀`` moment is first (d=2 cell order
#: ``[ψ̄, ψ̂_y, ψ̂_x, ψ̂_xy]``; per-axis face order ``[bar, slope]``).  Single
#: source for the slot-0 convention every moment consumer reduces on (#240 D5b)
#: — change the layout here, not at the scattered ``[..., 0]`` call sites.
AVERAGE_MOMENT = 0

#: The ``label`` of the within-cell spatial-moment factor when it rides an
#: axis-built space as a typed :class:`~orpheus.numerics.axis.Axis`
#: (campaign 1 CS4b): minted by
#: :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`,
#: read back by the field layer's tail/width accessors.  One spelling, here,
#: so the mint and the readers cannot drift.
SPATIAL_MOMENT_AXIS_LABEL = "spatial_moment"


def cell_moment_count(per_axis: int, ndim: int) -> int:
    r"""Per-cell spatial-moment count :math:`(\text{per\_axis})^{d}`.

    A cell is codimension-0, so it carries the full tensor-Legendre product
    over all ``d`` axes: ``1`` for the cell-average closures (DD/Step) and
    ``2^d`` for the bilinear UBLD Linear-Discontinuous closure (d=2: ``4`` —
    ``[ψ̄, ψ̂_y, ψ̂_x, ψ̂_xy]``).  Single source of the cell-count policy —
    the ``d`` exponent — shared by the scheme's typed moment axis
    (:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.moment_axis`),
    the field shape builders, the UBLD cell assembler, the SN
    solver's lift/reduce helpers, and the loss representation (#253: these
    were ~20 open-coded ``per_axis ** ndim`` spellings — a layout-policy
    change now lands HERE, not at scattered call sites).  ``per_axis``
    itself single-sources at the scheme trait
    ``DiscretizationScheme.spatial_basis_per_axis``; the codimension-1 FACE
    sibling is :func:`face_moment_count` (``d-1`` exponent).
    """
    return per_axis ** ndim


def face_moment_count(per_axis: int, ndim: int) -> int:
    r"""Per-face transverse spatial-moment count :math:`(\text{per\_axis})^{d-1}`.

    A face is codimension-1, so it carries the tensor-Legendre moments of the
    ``d-1`` along-face (transverse) axes: ``1`` for the cell-average closures
    (DD/Step → scalar faces, byte-identical) and ``2^{d-1}`` for the bilinear
    UBLD Linear-Discontinuous closure (d=2: ``2`` — ``[face-bar, face-slope]``).
    Single source of the "face is codimension-1" policy (the ``d-1`` exponent)
    shared by the trace producer
    (:meth:`orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout`) and the interior
    face cochain (``orpheus.sn.loss_representation._LossRepresentation._n_face_moments``),
    which MUST agree on the face width or the capture↔shed seam mis-shapes (#251).
    The CELL count is the sibling :func:`cell_moment_count` (no ``-1``).
    """
    return per_axis ** (ndim - 1)


def face_moment_tail(n_face_moments: int) -> tuple[int, ...]:
    r"""Trailing moment-axis shape suffix for a face-cochain buffer.

    A multi-moment closure (LD's bilinear face, ``n_face_moments > 1``) carries a
    trailing ``2^{d-1}``-moment axis; a cell-average closure (DD/Step,
    ``== 1``) leaves the face rank untouched (``()`` — NO length-1 axis appended)
    so its buffers stay byte-identical (#240 D5b — the backward-compat invariant).
    Single source for both storage policies (``_MovingFrontier`` window +
    ``FullFieldWavefront`` full cochain), which must agree on the tail shape.
    """
    return () if n_face_moments == 1 else (n_face_moments,)


def is_moment_valued_by_flat_rank(array: "np.ndarray", flat_ndim: int) -> bool:
    r"""Does ``array`` carry the trailing spatial-moment axis, judged against the
    FLAT (moment-free) rank ``flat_ndim``?

    ``True`` iff ``array`` has MORE axes than a flat ``(N…, ng, *spatial)`` buffer
    (whose rank is ``flat_ndim``) — i.e. it carries the trailing ``2^d`` moment
    axis. This is the rank CORE that both moment-valued discriminators reduce to:
    :func:`is_moment_valued_by_rank` (which derives ``flat_ndim`` from a
    per-ordinate-stripped reference array) and the external-source lift
    (:func:`orpheus.sn.solver._lift_external_source_to_moments`, which knows only
    the flat ``(N, ng, *spatial)`` rank, not a reference array). Homing the rank
    test here keeps it a single source (#246 / #247) — the layout-shift hazard is
    that an open-coded ``array.ndim == flat_ndim`` third spelling silently
    diverges from the primitive its sibling consumers track.

    RANK, not trailing-size: a coincidental ``n_diag == 2^d`` (a d=2 anti-diagonal
    of exactly 4 cells) mis-fires a ``shape[-1] == 2^d`` probe, but a flat
    buffer's rank never collides with a moment buffer's.
    """
    return array.ndim > flat_ndim


def is_moment_valued_by_rank(array: "np.ndarray", reference: "np.ndarray") -> bool:
    r"""Does ``array`` carry the trailing spatial-moment axis, judged by RANK?

    ``True`` iff ``array`` has more than one axis beyond ``reference`` — the
    S4-safe discriminator for "is this a moment-valued buffer". A moment buffer
    is ``(N…, ng, *spatial, 2^d)`` while its scalar reference (``Σ_t`` /
    ``reaction_xs``) is the per-ordinate-stripped ``(ng, *spatial)``, so a genuine
    moment buffer carries one MORE leading (ordinate) axis PLUS the trailing
    ``2^d`` moment axis — net ``> reference.ndim + 1`` — whereas a flat
    ``(N…, ng, *spatial)`` buffer (a matvec-zero / flat external source) sits at
    exactly ``reference.ndim + 1`` (the flat ``(N…, ng, *spatial)`` rank).

    Delegates to :func:`is_moment_valued_by_flat_rank` against that flat rank
    (``reference.ndim + 1`` — the per-ordinate-stripped reference plus the leading
    ordinate axis), the single rank core. The single source for the matvec
    moment-broadcast
    (:func:`orpheus.sn.loss_representation._moment_broadcast_sigma`) and the
    cell-solve source-reframe gate (``orpheus.sn.loss_representation.sweep_graph._CellSolve``).
    """
    return is_moment_valued_by_flat_rank(array, reference.ndim + 1)


def spatial_moment_tail(n_cell_moments: int) -> tuple[int, ...]:
    r"""Trailing spatial-moment-axis shape suffix for a bulk-field buffer.

    The CELL analogue of :func:`face_moment_tail` (which sizes the per-FACE
    transverse cochain). A multi-moment closure (LD, ``n_cell_moments ==
    per_axis**ndim > 1``) carries a trailing spatial-moment axis on the bulk
    field; a cell-average closure (DD/Step, ``== 1``) leaves the field rank
    untouched (``()`` — NO length-1 axis appended) so its buffers / spaces
    stay byte-identical (#240 D5b — the backward-compat invariant).

    Delegates to :func:`face_moment_tail` so the "append iff > 1" decision
    lives in EXACTLY ONE place — the cell-moment tail and the face-cochain
    tail must never disagree on the policy (``coding-elegance`` Pattern 7:
    normalise the convention at one site). Homed here since CS4c step 6
    item 6.2c-iii (it lived beside the retired ``SpatialMomentSpace``).
    """
    return face_moment_tail(n_cell_moments)
