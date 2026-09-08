"""SN operator algebra leaves — typed :class:`TimedFullField` contract.

Provides the four-operator algebra leaves consumed by the within-group
equation :math:`A_{\\rm wg} = L + C - S_{\\rm foldable}`:

* :class:`StreamingOperator` — :math:`L = \\Omega\\cdot\\nabla
  + \\text{angular redistribution}` (the curvilinear pole term lives
  here for sphere / cylinder).
* the collision multiplier :math:`C = M[\\sigma_t]` (a plain
  :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
  — diagonal in position, group, and direction; #261 retired the former
  ``CollisionOperator`` thin subclass).
* :class:`StreamingCollisionOperator` — the sweep-invertible specialisation
  :math:`(L + C)` returned by ``L + C``; ``is_invertible=True``
  via the WDD sweep.

All three operators consume and emit
:class:`~orpheus.transport.timed_full_field.TimedFullField` — the
typed composite carrier (bulk = :class:`~orpheus.transport.fields.angular_flux.AngularFlux`,
boundary = :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`).
Producer-side normalisation (Pattern 7): the typed contract is
enforced at every operator entry; no bare-ndarray packed-vector
adapter.  The matvec kernel is NOT on this operator — it lives in the
loss representation (``orpheus.sn.loss_representation``): the fused
``(L+C)ψ`` single-emission body (the apply-direction twin of the sweep)
rides ``_OneDimScanWalk._apply_walk`` for 1-D, and the multi-D Cartesian
``loss_action`` walk uses the ``ScanMarch`` default
(``MovingFrontierWindow`` a selectable peer).  The curvilinear
Morel–Montry angular redistribution is computed IN-SWEEP there — not a
separable typed leaf.

Three geometries are supported:

* **Cartesian 2D** — ``L = μ_x ∂/∂x + μ_y ∂/∂y + Σ_t``
* **Spherical 1D** — ``L = μ (A ∂/∂r)/V + (α ∂/∂μ)/V + Σ_t``
* **Cylindrical 1D** — per-level azimuthal redistribution

.. note:: Symmetric-closure invariant

   ``L.apply`` (the matvec) uses a **symmetric** closure — Cartesian
   upwind cell-center finite differences, curvilinear arithmetic
   spatial-face averages with τ-weighted angular interpolation —
   DISTINCT from the WDD asymmetric closure the sweeps (``.solve``)
   use.  The two converge in the fine-mesh limit; on curvilinear the
   sweep's WDD closure carries the ERR-026 closure-bias self-consistent
   fixed point that the Krylov-on-:meth:`apply` path bypasses.  Full
   comparison + the closure table:
   ``docs/theory/methods/sn/loss_representation.rst §loss-rep-history``,
   ``curvilinear_numerics.rst §sn-phase-d-err-026-closure-narrative``.

.. note:: Boundary-condition handling (Issue #208)

   The realized boundary law ``B`` is a **first-class sibling
   operator** of ``L``, NOT re-applied inside this matvec.  The
   canonical SN loss is ``(L_full + C - S - F - B)`` on the direct-sum
   state ``V = V_bulk ⊕ V_inflow ⊕ V_outflow``.  The representation's
   bare ``loss_action`` reads ``psi.boundary.inflow`` as a GIVEN, keeps
   the outflow self-consistency defect ``streamed - psi.outflow`` on the
   outflow trace row, and adds the inflow identity ``I·psi.inflow`` —
   with NO ``bc.apply``.  The reflective coupling
   ``psi.inflow = B·psi.outflow`` is delivered by the sibling ``-B``
   (:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`); the
   outer Krylov / SI loop drives the boundary consistency residual
   ``psi.inflow - B·psi.outflow - q.inflow → 0``.  Both the 1-D and the
   multi-D Cartesian paths are bare (matvec ≡ sweep by construction,
   L21) — post-extraction the outflow trace is the explicit solved
   unknown ``psi.outflow`` that ``-B`` reads, closing ERR-026 by
   construction for the 1-D curvilinear path.  Full block-matrix
   derivation, the three design corrections, the two ``-B`` delivery
   routes, and the O.2 forcing: :ref:`bc-extraction`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

import numpy as np

from functools import cached_property

from orpheus.numerics.operator import (
    BlockRole,
    LinearOperator,
    OperatorSum,
)

from orpheus.numerics.quadrature import Quadrature
from orpheus.transport.full_field import FullField
from orpheus.transport.operators.multiplication_operator import MultiplicationOperator

if TYPE_CHECKING:
    from collections.abc import Callable

    from orpheus.transport.fields.angular_flux import AngularFlux
    from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
    from orpheus.transport.fields.cross_section_field import CrossSectionField
    from orpheus.transport.timed_full_field import TimedFullField
    from orpheus.numerics.space import FunctionSpace
    from ..mesh.augmented_mesh import SNMesh
    from ..angular.closure import AngularClosureBase
    from orpheus.transport.spatial.scheme import DiscretizationSchemeBase
    from orpheus.numerics.frame import FrameBase
    from orpheus.transport.source_sinks import (
        AngularSourceSink,
        ScalarSourceSink,
    )
    from ..loss_representation.sweep_schedule import SweepSchedule
    from ..loss_representation import LossRepresentation
    # Type-only (the runtime constructions are late imports inside ``inverse``
    # / ``__sub__`` to break the operator <-> composite import cycles).
    from .boundary import SNMaskedBoundaryOperator
    from .scheduled_invertible import ScheduledInvertibleOperator
    from .sweep_operator import SweepOperator

__all__ = [
    "StreamingOperator",
]


@dataclass
class StreamingOperator(LinearOperator["FullField"]):
    r"""Pure streaming + angular-redistribution operator :math:`L` as a
    :class:`~orpheus.numerics.operator.LinearOperator` leaf.

    The "L" of the Phase G four-operator algebra
    :math:`A_{\rm wg} = L + C - S_{\rm foldable}`. Carries the spatial
    streaming math plus the curvilinear angular redistribution — the
    cell-collision term lives in the separate collision multiplier
    :math:`C = M[\sigma_t]` (a
    :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`).
    The split lets the within-group operator be pure algebra
    (``L + C - S.foldable_part()``); no ``WithinGroupOperator`` wrapper.

    Pure σ-free streaming ``apply`` (#257 S8b)
    ------------------------------------------

    .. math::

        L.{\rm apply}(\psi) \;:=\; \Omega\cdot\nabla\psi \;=\;
            \text{streaming\_action}(\psi)

    :math:`L` computes pure streaming **directly, with no
    :math:`\sigma`**: :meth:`apply` calls the
    :attr:`loss_representation`'s named σ-free
    :meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action`
    leaf (the ONE streaming discretization, single-sourced through
    ``loss_action`` at :math:`\sigma = 0`).  The collision diagonal
    :math:`C = M[\sigma_t]` is the separate shared collision multiplier
    (a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
    and the composition

    .. math::

        (L + C).{\rm apply}(\psi) \;=\; \text{streaming\_action}(\psi)
            \;+\; \sigma_t \odot \psi \;=\; M(\psi;\;\sigma_t)

    recovers the full within-group loss (the WDD matvec is affine in
    :math:`\sigma` in the forward direction — see
    :meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action`).

    Why :math:`L` carries NO σ (Pattern 4 — #257 S8b)
    -------------------------------------------------

    The discrete curvilinear matvec threads :math:`\sigma_t` through the
    Carlson coupled-pole seed (Hébert §3.9.4), but that
    :math:`\sigma`-dependence is *exactly the collision diagonal* the seed
    injects — it cancels into :math:`\sigma\cdot\psi` and belongs to
    :math:`C`, not :math:`L` (the continuous :math:`L` is σ-independent, so
    is the discrete leaf; σ-freedom is probe-verified byte-stable).
    Pattern 4 (illegal states unrepresentable): the constructor takes NO
    σ — a σ on :math:`L` would be a parameter the leaf never reads.

    Capability set
    --------------

    Pure streaming alone is **not
    invertible** (the streaming operator is rank-deficient without a
    collision term to make the within-group cell balance non-singular).
    The ``solve`` capability appears at the
    :class:`~orpheus.numerics.operator.OperatorSum` level: ``(L + C
    - S_foldable).solve(q)`` would route to the within-group sweep via a
    σ_r fusion hook — but ⚠ that σ_r-sweep is exact ONLY for isotropic flux
    (it removes the diagonal-in-angle ``Σ_s0·I``, not the isotropic-projection
    ``Σ_s0·P_iso``); wiring it as a within-group accelerator ships 46–56 %
    errors on anisotropic problems (issue #215; the stable+correct fold is
    consistent DSA #2 / Krylov). ``apply_transpose`` IS available
    (Wave O / O.2b) — the analytic reverse-direction adjoint matvec
    :math:`L^{\mathsf T}` (see :meth:`apply_transpose`), so the operator
    carries a working ``apply_transpose`` and ``L.H`` is the physical G-adjoint.

    Posing (P4.9b) — the operator holds its two closures
    ----------------------------------------------------

    :math:`L` is POSED with the two objects that close its two axes: the
    spatial discretization (one cell-local closure per spatial scheme)
    and the bound angular closure (the ordinate march).  The production
    surface is :meth:`pose`, which reads BOTH off the hub
    (:class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` — the save-state /
    data hub that owns the generator so SN and DSA stay consistent) — on
    that path the operator's slots ARE the hub's instances, by
    construction, which is why this constructor carries **no guards**.

    The raw constructor is the declared EXPERT SEAM (doctored diagnostic
    probes build through it — better than a monkeypatch).  What it does
    not check, measured 2026-08-28: a wrong-FAMILY closure (e.g. the
    Cartesian identity closure on curvilinear factors) constructs and
    then raises at the FIRST sweep (typed on the sphere, untyped
    ``IndexError`` on the cylinder — the walk's family dispatch refuses
    it); a closure smuggled from a DIFFERENT hub of equal shape is the
    one genuinely silent arm (wrong pairing, plausible-wrong answers) —
    reachable only by deliberately crossing two hubs.

    Parameters
    ----------
    sn_mesh : SNMesh
        The geometric substrate: quadrature, BCs (the face-name-keyed
        ``sn_mesh.bc`` dict), and (for curvilinear) the precomputed
        connection coefficients.  Transitional — the end state (rides
        O-3/CS5) is the cross-method ``(domain, codomain,
        spatial-discretization[, angular-discretization])`` constructor
        with no mesh argument.
    spatial_closure : DiscretizationSchemeBase
        The spatial axis's closure.  Today this receives the hub's
        discretization-scheme INSTANCE — the extraction of a closure
        from its generator is the identity until O-3 splits the
        closure/factory family; the slot names the ROLE it consumes.
    angular_closure : AngularClosureBase
        The angular axis's closure — the hub's bound instance (the
        Morel–Montry march on curvilinear charts, the identity closure
        on Cartesian).  Pure :math:`L` reads no :math:`\sigma`.
    """

    sn_mesh: "SNMesh"
    spatial_closure: "DiscretizationSchemeBase"
    angular_closure: "AngularClosureBase"

    # Streaming is the sole FULL operator — it couples bulk ↔ boundary
    # (reads the inflow trace to seed the sweep, writes the outflow
    # trace). Issue #208. Class-level constant (unannotated so the
    # dataclass does not treat it as a field).
    block_role = BlockRole.FULL

    @classmethod
    def pose(cls, sn_mesh: "SNMesh") -> "StreamingOperator":
        r"""Pose :math:`L` from the hub's own method objects (P4.9b).

        The INTERMEDIATE posing surface while the operator migrates to
        its explicit-argument constructor: reads the hub's
        discretization scheme and bound angular closure and passes them
        — the operator's slots are the hub's own instances, BY
        CONSTRUCTION (one generator and one bound closure per solve;
        the ERR-026 two-inductions shape is unspellable on this path,
        which is why the raw constructor carries no guards).

        The end state (recorded in the campaign plan, rides O-3/CS5) is
        the cross-method ``(domain, codomain, spatial-discretization[,
        angular-discretization])`` constructor with no mesh argument;
        this classmethod is that migration's lever and retires with it.
        """
        return cls(sn_mesh, sn_mesh.scheme, sn_mesh.angular_closure)

    @property
    def is_adjointable(self) -> bool:
        # Two-factor honest: the KERNEL factor (scheme.has_transpose_kernel
        # — the cell relation has a REGISTERED transpose realization; DD and
        # LD both derive True since #310 C2) AND the ORIENTATION factor
        # (representation.has_transpose_walk — the walk reverses; every
        # registered representation since #310 C4/C5). Both factors pass on
        # the whole registered scheme × representation grid today; the
        # predicate stays so a FUTURE scheme without a registered kernel
        # pair (or a representation without a reverse walk) raises
        # MissingAdjoint eagerly at ``.H`` construction rather than
        # reaching a raising reverse walk at apply time. is_invertible
        # inherits base False — pure streaming L is not sweep-invertible;
        # only (L+C) is. (Two-factor derivation: loss_representation.rst
        # §loss-rep-orientation-two-frames.)
        return (
            type(self.spatial_closure).has_transpose_kernel
            and self.loss_representation.has_transpose_walk
        )

    @property
    def domain(self) -> "FunctionSpace":
        r"""The composite carrier :math:`V_{\rm bulk}\oplus V_{\rm trace}` (Wave O / O.2b).

        :math:`L` is the sole FULL operator — it couples bulk :math:`\leftrightarrow`
        boundary (seeds the sweep from the inflow trace, emits the outflow
        trace). Advertising :attr:`~orpheus.sn.mesh.augmented_mesh.SNMesh.full_field_space`
        is what lets :class:`~orpheus.numerics.operator.AdjointOperator`
        read the **block-diagonal G-adjoint metric** (bulk :math:`V\,w_n`
        :math:`\oplus` trace :math:`|\Omega\cdot\hat n|\,w_n`) for ``L.H`` —
        without it the adjoint silently reduces to the metric-blind Euclidean
        transpose (Issue #208 risk R5).

        ``C`` / ``S`` / ``F`` advertise the SAME composite domain, so the
        within-group ``(L + C) - S - B``
        :class:`~orpheus.numerics.operator.OperatorSum` guard VALIDATES the
        build (no ``None``-spaced summand is silently skipped), and the
        transpose-closed sub-sums G-conjugate every bulk leaf via the
        op-level :math:`G^{-1}(\sum \text{leaf}^{\mathsf T})G`.  Every
        within-group leaf transposes, so ``.H`` reachability extends to the
        full loss ``(L + C - S - B)`` (pinned by
        ``test_g_adjoint_reciprocity``); it stays predicate-gated per leaf
        (loud :class:`~orpheus.numerics.operator.MissingAdjoint`, never
        silently Euclidean) via the two-factor :attr:`is_adjointable`.
        Two-factor derivation:
        ``docs/theory/methods/sn/loss_representation.rst §loss-rep-orientation-two-frames``.
        """
        return self.sn_mesh.full_field_space

    @property
    def codomain(self) -> "FunctionSpace":
        # Endomorphism on the composite (see :meth:`domain`).
        return self.sn_mesh.full_field_space

    def apply(self, psi: "FullField") -> "FullField":
        r"""Pure σ-free forward streaming :math:`L\,\psi = \Omega\cdot\nabla\psi`.

        Computes pure streaming via the :attr:`loss_representation`'s σ-free
        :meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action`
        leaf; :math:`L` reads NO :math:`\sigma` (the ``L + C`` composition
        recovers the full loss — see the class docstring).

        ``L`` is the ONLY operator that emits a non-zero face
        residual on its output ``.boundary`` — the collision multiplier
        :math:`C = M[\sigma_t]`
        (a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
        :class:`~orpheus.transport.operators.scattering.ScatteringOperator`, and
        :class:`~orpheus.transport.operators.fission.FissionOperator` all leave the
        output boundary at the auto-allocated zero.

        Parameters
        ----------
        psi : FullField
            Composite carrier with bulk
            (:class:`~orpheus.transport.fields.angular_flux.AngularFlux`)
            and boundary
            (:class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`).
            Operator and ``psi.interior.mesh`` MUST be the same
            :class:`~orpheus.sn.mesh.augmented_mesh.SNMesh` instance.

        Returns
        -------
        FullField
            ``L·ψ`` as a timeless composite — bulk carries the pure
            streaming cell action, boundary carries the face residual at
            the layout-assigned trace slots (non-zero at outer face for
            curvilinear; non-zero at outer + inner faces for slab).
            History-free (#257 S8a — the matvec leaf is a base arrow
            ``FullField -> FullField``; the comonad lives on the driver).
        """
        FullField.require_member(
            psi, mesh=self.sn_mesh, context="StreamingOperator.apply",
        )
        return self.loss_representation.streaming_action(psi)

    def apply_transpose(self, phi: "FullField") -> "FullField":
        r"""Euclidean transpose :math:`L^{\mathsf T}\,\phi` (#208).

        The σ-free adjoint streaming leaf: :math:`L^{\mathsf T}\phi` via the
        :attr:`loss_representation`'s
        :meth:`~orpheus.sn.loss_representation.LossRepresentation.streaming_action_transpose`
        (single-sourced through ``loss_action_transpose`` at :math:`\sigma = 0`;
        implemented on the full registered scheme × representation grid
        since #310 C5 — the 1-D reverse walks, the mirror-octant wavefront
        reverses on DD and LD at any ``d``, the row-march reverse; an
        unregistered-kernel scheme still raises typed, never a silent
        wrong answer).  Since :math:`C = \sigma_t\odot` is a self-adjoint
        diagonal, the full adjoint loss factors as
        :math:`(L + C)^{\mathsf T} = L^{\mathsf T} + C`.

        This returns the **plain Euclidean transpose** :math:`L^{\mathsf T}`.
        The metric conjugation :math:`G^{-1}\!\cdot^{\mathsf T}\!\cdot G` of the
        physical **G-adjoint** ``L.H`` is applied AROUND this by
        :class:`~orpheus.numerics.operator.AdjointOperator`, which reads the
        ``domain`` / ``codomain`` ``inner_product_weights`` (bulk volume on the
        cell block, the ``|Ω·n|·w`` partial-current metric on the trace block).

        Verified by the G-adjoint reciprocity gate
        ``test_g_adjoint_reciprocity`` (slab / sphere / cylinder, -O-firing) +
        its L11 wrong-trace-metric negative control.
        """
        FullField.require_member(
            phi, mesh=self.sn_mesh, context="StreamingOperator.apply_transpose",
        )
        return self.loss_representation.streaming_action_transpose(phi)

    # ── LossRepresentation carve (S2) — the polymorphic matvec dispatch ─────

    @cached_property
    def loss_representation(self) -> "LossRepresentation":
        r"""THE loss-operator representation for this operator's mesh (S6.5).

        The ONE first-class ``LossRepresentation``
        (``orpheus.sn.loss_representation``) carrying BOTH actions of
        :math:`(L+C)`: :meth:`apply` routes through
        ``representation.loss_action`` / ``loss_action_transpose`` (the
        matvec), and :meth:`StreamingCollisionOperator.solve` runs the forward
        substitution on the SAME object via
        :attr:`StreamingCollisionOperator.loss_representation` — L21 ("matvec ≡
        sweep") as a type fact.  Selection is by geometry
        (``default_for``): 1-D → ``CumprodScan``; multi-D Cartesian →
        ``ScanMarch`` (the S6.9 Fork-B2 default).  ``cached_property`` because
        the selection is fixed by the mesh, stable across the operator's
        lifetime; the lazy import breaks the operator ↔ loss_representation
        module cycle.
        """
        from ..loss_representation import default_for

        return default_for(
            self.sn_mesh, self.spatial_closure, self.angular_closure,
        )

    # ── Algebra dispatch — sweep-invertible composite (R-1 Step C) ────

    @overload
    def __add__(self, other: "MultiplicationOperator") -> "StreamingCollisionOperator": ...
    @overload
    def __add__(
        self, other: "LinearOperator[FullField]",
    ) -> "OperatorSum[FullField, FullField]": ...
    def __add__(
        self, other: "LinearOperator[FullField]",
    ) -> "OperatorSum[FullField, FullField]":
        r"""Compose :math:`L + X`.

        When ``X`` is a
        :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
        (the collision diagonal :math:`C = M[\sigma_t]`), returns the
        sweep-invertible specialisation :class:`StreamingCollisionOperator`
        carrying the algebraic identity :math:`(L + C)^{-1} \approx
        \text{WDD sweep}` — the typed ``@overload`` spells the fusion
        rule, so ``L + C`` reads as a ``StreamingCollisionOperator`` statically
        (C4: the covariant summand legs make the specialisation
        assignable to the generic contract).  Otherwise falls through to
        the generic :class:`OperatorSum` via the mixin.

        #261: ``L + C`` is the canonical (and only) ordering — the dispatch
        lives here on the SN-specific streaming leaf, because the
        transport-level multiplier cannot dispatch back onto ``StreamingOperator``
        (that would be a ``transport → sn`` upward import).
        """
        if isinstance(other, MultiplicationOperator):
            return StreamingCollisionOperator(self, other)
        return super().__add__(other)


# ─────────────────────────────────────────────────────────────────────────
# StreamingCollisionOperator — sweep-invertible composite (L + C)
# ─────────────────────────────────────────────────────────────────────────


class StreamingCollisionOperator(
    OperatorSum[
        "FullField", "FullField", "StreamingOperator", "MultiplicationOperator",
    ],
):
    r"""Sweep-invertible composite :math:`L + C` carrying ``.solve`` = WDD sweep.

    The SN-specific algebraic identity

    .. math::

        (L_{\rm streaming} + C_{\rm diagonal})^{-1} \;\approx\;
        \text{WDD sweep}

    has no generic ``(A+B)^{-1}`` formula — a plain :class:`OperatorSum`
    can only invert ITERATIVELY (the preconditioned-splitting
    :class:`~orpheus.numerics.green_operator.GreenOperator` its generic
    ``.inverse()`` returns).  The WDD sweep IS the DIRECT inverse algorithm
    for this specific composite — the algebraic foundation of the entire SN
    method (Lewis & Miller §3.2; Adams & Larsen 2002 §III) — and this
    subclass's ``.inverse()`` override
    (→ :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`) shadows
    the generic Green by MRO (the type IS the structure).  :meth:`apply` /
    :meth:`apply_transpose` OVERRIDE the :class:`OperatorSum` leaf-sum to
    return the within-group loss :math:`(L+C)\psi = M(\sigma)\psi` (and its
    transpose) DIRECTLY via :attr:`loss_representation`, single-sourcing
    :math:`\sigma` from the diagonal — the SAME :math:`\sigma` ``solve``
    threads into the WDD sweep, so matvec, adjoint, and sweep are three
    actions of ONE operator (L21).

    Construction
    ============

    Two equivalent paths:

    * **Operator algebra dispatch** — ``L + C`` where ``L`` is a
      :class:`StreamingOperator` and ``C`` is the collision multiplier
      :math:`M[\sigma_t]`
      (a :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`)
      returns this class automatically.  ``L + C`` is the canonical (and
      only) ordering: the dispatch lives one-directionally on
      :meth:`StreamingOperator.__add__`, because a transport-level
      multiplier cannot dispatch back onto an ``sn`` operator (#261).  The
      composite reads as math.
    * **Explicit construction** — ``StreamingCollisionOperator(L, C)``.  Useful
      when composing variants such as
      ``StreamingCollisionOperator(L_leaf, MultiplicationOperator.from_mesh(σ_r, mesh))``
      where
      ``σ_r = σ_t - Σ_{s,0}^{g→g}`` is the removal cross-section that
      lets one fold the within-group self-scatter into the diagonal
      collision term (Adams & Larsen 2002 §III; tracked by issue
      `#200 <https://github.com/deOliveira-R/ORPHEUS/issues/200>`_).

    The two paths produce structurally identical objects — the choice
    only changes the call-site readability.

    Capability set
    ==============

    ``is_invertible=True`` — adds
    ``solve`` (the WDD sweep) to the parent :class:`OperatorSum`'s set;
    ``apply_transpose`` propagates by the :class:`OperatorSum` closure
    law (both :math:`L` and :math:`C` advertise it) and is OVERRIDDEN to
    the composite's own :math:`M(\sigma)^{\mathsf T}` action (Wave O #208 /
    #240 Step B).  The adjoint matvec is complete on the registered
    scheme × representation grid since #310 C4 (multi-D Cartesian DD) /
    C5 (LD-2D); an unregistered-kernel scheme still raises typed —
    never a silent wrong answer.

    The ``.solve`` API
    ==================

    The ``rhs`` parameter is the timeless composite
    :class:`~orpheus.transport.full_field.FullField` (W-C, P4.5).  The
    history-bearing :class:`~orpheus.transport.timed_full_field.TimedFullField`
    iterate passes through by inheritance (it IS a ``FullField``), and a
    bare ``FullField`` is admitted as the ``history_depth = 0``
    degenerate.  ``rhs`` carries:

    * ``rhs.interior.values`` — per-ordinate source ``(N, ng, nx, ny)``.
      This is treated as the per-ordinate anisotropic source
      :math:`Q^{\rm aniso}` that the sweep consumes (the isotropic
      source is zero).
    * ``rhs.boundary`` — face source / BC inflow trace.  Typically
      zero for volumetric SI/Krylov sources (which carry no face
      contribution); the persistent reflective-BC state lives on the
      :class:`SNMesh` and is handled inside the sweep.  It seeds the
      sweep's mutable boundary buffer (per-face copy).

    The curvilinear starting-direction :math:`\psi_{1/2}` is computed
    DIRECTLY from the source (the #282 route (a) direct seed, 2.5d) — the
    WDD sweep is an EXACT direct inverse with no previous-iterate seed. A
    warm start, when useful, lives at the ITERATION layer
    (:meth:`~orpheus.numerics.iteration.SourceIteration.solve`'s
    ``initial_guess`` :math:`x_0`) or the ITERATIVE
    :class:`~orpheus.numerics.green_operator.GreenOperator`, never on this
    direct sweep.

    Parameters
    ----------
    streaming : StreamingOperator
        :math:`L = \Omega\cdot\nabla + \text{angular redistribution}`.
        σ-free since #257 S8b: :meth:`StreamingOperator.apply` IS
        ``loss_action(0, ψ)`` — the same walk read at :math:`\sigma = 0`,
        NOT a subtraction.  The Resolution-A identity
        ``L.apply(ψ) + σ_t ⊙ ψ = M(ψ; σ_t)`` still holds (the forward
        matvec is affine in :math:`\sigma`); no shipped code evaluates
        its subtractive form.
    diagonal : MultiplicationOperator
        :math:`C = M[\sigma]`.  Its ``.coefficient.values`` is the
        per-cell per-group coefficient used by the sweep (canonically
        ``σ_t``; can be ``σ_r`` for the foldable variant).

    Notes
    -----
    The validation ``σ > 0`` everywhere guards against the
    ``σ_r < 0`` case that can arise when within-group self-scatter
    exceeds total cross-section (rare; not physically meaningful but
    mathematically possible for ill-conditioned multi-group sets).
    The sweep would emit NaN at those cells — surfacing the
    inconsistency at construction is friendlier.
    """

    def __init__(
        self,
        streaming: "StreamingOperator",
        diagonal: "MultiplicationOperator",
    ) -> None:
        if not isinstance(streaming, StreamingOperator):
            raise TypeError(
                f"StreamingCollisionOperator: 'streaming' must be a "
                f"StreamingOperator; got {type(streaming).__name__}."
            )
        if not isinstance(diagonal, MultiplicationOperator):
            raise TypeError(
                f"StreamingCollisionOperator: 'diagonal' must be a "
                f"MultiplicationOperator; got {type(diagonal).__name__}."
            )
        # Mesh-identity invariant (#261 re-anchor): the WDD sweep threads the
        # diagonal's σ against the STREAMING geometry, so the two must act on
        # the SAME mesh object — geometric consistency, strictly stronger than
        # the (name, shape) shape-equality the OperatorSum composition guard
        # checks. The diagonal multiplier is mesh-free; its mesh is carried by
        # its CrossSectionField coefficient.
        if diagonal.coefficient.space != streaming.sn_mesh.bulk_space:
            raise ValueError(
                "StreamingCollisionOperator: the diagonal multiplier's σ "
                "must agree with the streaming geometry's scalar bulk in "
                "content — the space-content invariant "
                "(diagonal.coefficient.space == streaming.sn_mesh.bulk_space): "
                "the WDD sweep pairs the diagonal's σ with the streaming "
                "geometry."
            )
        if not np.all(diagonal.coefficient.values > 0):
            min_sigma = float(np.min(diagonal.coefficient.values))
            raise ValueError(
                f"StreamingCollisionOperator: diagonal coefficient must be "
                f"strictly positive everywhere for the WDD sweep to be "
                f"well-defined; got min(sigma) = {min_sigma:.3e}.  If "
                f"sigma_r = sigma_t - Sigma_(s,0)^(g->g) is dipping "
                f"negative, the multi-group cross-section set is "
                f"physically inconsistent."
            )
        super().__init__(streaming, diagonal)
        # block_role is now DERIVED by OperatorSum.__init__ (Wave O / O.2b 4.5):
        # join(L=FULL, C=BULK) = FULL. The former hand-stamp here was the
        # twin-path retired in 4.5 — the role is carried by construction.

    @property
    def is_invertible(self) -> bool:
        # (L+C) is sweep-invertible: the WDD forward-substitution sweep IS its
        # inverse operator (in
        # __init__). This is the SOLE invertible OperatorSum — the base
        # OperatorSum.is_invertible is False. is_adjointable inherits the
        # OperatorSum a∧b law (both L and C advertise apply_transpose).
        return True

    # ── Convenience accessors ─────────────────────────────────────────

    @property
    def streaming(self) -> "StreamingOperator":
        """The streaming operand (alias for ``self.a``)."""
        return self.a

    @property
    def diagonal(self) -> "MultiplicationOperator":
        """The diagonal-multiplier operand :math:`C = M[\\sigma]` (alias for ``self.b``)."""
        return self.b

    @property
    def loss_representation(self) -> "LossRepresentation":
        r"""The ONE :class:`LossRepresentation` for this operator (S6.5, #222).

        Delegates to the streaming leaf's cached instance — the SAME
        object :meth:`StreamingOperator.apply` consumes for the matvec
        :math:`(L+C)\psi`.  :meth:`solve` runs the forward substitution
        :math:`(L+C)^{-1}q` on it, so "matvec ≡ sweep — two actions of
        ONE operator" (L21) is a type fact enforced by construction,
        not a coincidence of two ``default_for`` calls agreeing.
        """
        return self.streaming.loss_representation

    @property
    def sn_mesh(self) -> "SNMesh":
        """The shared :class:`SNMesh` (validated mesh-identity at init)."""
        return self.streaming.sn_mesh

    @property
    def sigma(self) -> np.ndarray:
        r"""The diagonal coefficient used by ``solve`` (σ_t or σ_r).

        Single-sources :math:`\sigma` off the diagonal multiplier's
        :class:`~orpheus.transport.fields.cross_section_field.CrossSectionField`
        coefficient (``coding-elegance`` Pattern 2 — no duplicate storage).
        """
        return self.diagonal.coefficient.values

    # ── apply / apply_transpose: the composite's OWN matvec (#240 Step B) ──

    def apply(self, psi: "FullField") -> "FullField":
        r"""Matvec :math:`(L+C)\,\psi = M(\sigma)\,\psi` — the composite OWNS it.

        Both the matvec and the sweep are actions of the ONE :math:`(L+C)`
        operator (L21 "matvec ≡ sweep"), realised with THIS composite's
        diagonal :math:`\sigma` (``self.sigma`` — the SAME array :meth:`solve`
        threads into the WDD sweep); the representation's :meth:`loss_action`
        returns the full within-group loss :math:`M(\sigma)\psi` directly.
        This OVERRIDES the inherited leaf-sum :meth:`OperatorSum.apply`, which
        is value-equal only by the forward-direction affine-in-:math:`\sigma`
        coincidence — the override single-sources :math:`\sigma` from the
        diagonal (Pattern 2), removing that latent coupling.

        On a carrying mesh this IS the ray-decoupled ``(A,A)`` block action
        (step 6 — presence is structural): the joint ``M`` matvec is the
        within-group grid's :meth:`~orpheus.numerics.coupled_system.CoupledOperator.apply`,
        never a kwarg channel on this surface.
        """
        FullField.require_member(
            psi, mesh=self.sn_mesh, context="StreamingCollisionOperator.apply",
        )
        return self.loss_representation.loss_action(self.sigma, psi)

    def apply_transpose(self, phi: "FullField") -> "FullField":
        r"""Adjoint matvec :math:`(L+C)^{\mathsf T}\,\phi = M(\sigma)^{\mathsf T}\,\phi`.

        The adjoint sibling of :meth:`apply` (#240 Phase 2 Step B): the
        representation's :meth:`loss_action_transpose` realises
        :math:`(L+C)^{\mathsf T}\phi = M(\sigma)^{\mathsf T}\phi` directly with
        THIS composite's diagonal :math:`\sigma`, overriding the inherited
        :meth:`OperatorSum.apply_transpose` leaf sum.  Complete on the
        registered grid since #310 C4/C5 (the multi-D Cartesian reverses,
        DD and LD); an unregistered-kernel scheme still raises typed —
        never a silent wrong answer.
        The plain Euclidean transpose; the metric conjugation of the physical
        G-adjoint ``.H`` is applied AROUND this by
        :class:`~orpheus.numerics.operator.AdjointOperator` (pinned by
        ``test_g_adjoint_reciprocity``).

        On a carrying mesh this is the ray-decoupled ``(A,A)`` block
        transpose (step 6): the joint ``Mᵀ`` action is the grid's
        :meth:`~orpheus.numerics.coupled_system.CoupledOperator.apply_transpose`.
        """
        FullField.require_member(
            phi, mesh=self.sn_mesh, context="StreamingCollisionOperator.apply_transpose",
        )
        return self.loss_representation.loss_action_transpose(self.sigma, phi)

    # ── Algebra dispatch — schedule-folded composite (#226 step 2) ────

    @overload
    def __sub__(
        self, other: "SNMaskedBoundaryOperator",
    ) -> "ScheduledInvertibleOperator": ...
    @overload
    def __sub__(
        self, other: "LinearOperator[FullField]",
    ) -> "OperatorSum[FullField, FullField]": ...
    def __sub__(
        self, other: "LinearOperator[FullField]",
    ) -> "OperatorSum[FullField, FullField]":
        r"""Compose :math:`(L+C) - X`.

        When ``X`` is the strictly-lower boundary half ``B_lower`` (an
        :class:`~orpheus.sn.operators.boundary.SNMaskedBoundaryOperator`
        from :meth:`~orpheus.sn.operators.boundary.SNBoundaryOperator.split`),
        returns the sweep-invertible schedule-folded specialisation
        :class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`
        — the reified splitting matrix :math:`M = (L+C-B_{\rm lower})` whose
        ``solve`` is the octant-group forward substitution (§17 W2).
        The typed ``@overload`` spells the fusion rule (C4, as
        :meth:`StreamingOperator.__add__`).
        Otherwise falls through to the generic difference via the mixin.

        ``(L+C) - B_lower`` is the canonical spelling — the dispatch lives
        here on the SN composite, mirroring :meth:`StreamingOperator.__add__`
        (#261: one-directional, the operand cannot dispatch back).
        """
        from .boundary import SNMaskedBoundaryOperator

        if isinstance(other, SNMaskedBoundaryOperator):
            from .scheduled_invertible import ScheduledInvertibleOperator

            return ScheduledInvertibleOperator(self, other)
        return super().__sub__(other)

    # ── solve: WDD sweep ─────────────────────────────────────────────

    def inverse(self) -> "SweepOperator":
        r"""Return the inverse OPERATOR :math:`(L+C)^{-1}` (the carve, #226).

        ``A.inverse().apply(b)`` is the WDD forward-substitution sweep,
        BIT-IDENTICAL to ``A.solve(b)`` — the returned
        :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` delegates to
        :meth:`solve`. This is the operator normal form
        ``K = A_loss.inverse() @ F`` (Grand Report v3 §1): the forward view
        :meth:`apply` and the inverse view ``inverse().apply`` are the two views
        of ONE operator, the way ``A`` and ``A.H`` are.

        **Forward-side back-half twin (collapse trigger).**  The
        ``is_invertible``/``inverse``/``solve`` back-half here is deliberately
        coextensive with
        :class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`'s
        (the schedule-folded sibling, #226 step 2) — two witnesses, kept
        twinned per defer-until-≥2.  TRIGGER: at the 3rd sweep-invertible
        FORWARD composite, extract a shared mixin; do not hand-re-derive it.
        (Distinct from the INVERSE-side twin noted on ``SweepOperator`` —
        Green/Matrix inverses grow that shape, not this one.)
        """
        from orpheus.sn.operators.sweep_operator import SweepOperator

        return SweepOperator(self)

    def solve(
        self,
        rhs: "FullField",
        *,
        initial_guess: "FullField | None" = None,
    ) -> "TimedFullField":
        r"""Invert :math:`(L + C)\,\psi = \text{rhs}` via the WDD sweep.

        The cell-balance equation
        :math:`(\Omega\cdot\nabla + \sigma)\,\psi = Q` is integrated
        cell-by-cell in inflow-to-outflow order; the angular closure
        (Cartesian → identity, curvilinear → Morel-Montry) is bound
        on the mesh.

        Parameters
        ----------
        rhs : FullField
            The timeless composite source (W-C, P4.5).  The
            history-bearing :class:`TimedFullField` iterate passes
            through by inheritance (it IS a ``FullField``), and a bare
            :class:`~orpheus.transport.full_field.FullField` is admitted
            as the ``history_depth = 0`` degenerate.  Carries:

            * ``bulk.values`` — per-ordinate source
              :math:`Q^{\rm aniso}`, shape ``(N, ng, nx, ny)``.
            * ``boundary`` — BC inflow trace (typically zero for
              SI/Krylov volumetric sources; seeds the sweep's mutable
              boundary buffer).

            The legacy :class:`AngularFlux` arm is retired (it is NOT a
            ``FullField``, so the guard rejects it).
        Returns
        -------
        TimedFullField
            The angular flux satisfying :math:`(L + C)\,\psi =
            \text{rhs}`, with the sweep's outflow face state in
            ``.boundary``.  The WDD sweep emits a
            :class:`TimedFullField` iterate (the genuine driver-side
            comonad carrier); its ``history_depth`` matches
            ``rhs.history_depth`` (0 when ``rhs`` is a bare
            ``FullField``), and ``_history`` is empty — the outer
            SI / Krylov loop owns history threading.

        The uniform inverse-family
        :class:`~orpheus.numerics.iteration.SupportsSeededApply` keyword
        ``initial_guess`` (#285) is ACCEPTED and DROPPED — the WDD sweep is an
        EXACT direct inverse with nothing to seed (the curvilinear ψ½ starting
        direction is the direct #282 seed, 2.5d). It is kept so the driver's
        uniform threading and the seed-INDEPENDENCE gates express the
        accept-and-drop contract; a warm start lives at the iteration layer
        (:class:`~orpheus.numerics.iteration.SourceIteration` /
        :class:`~orpheus.numerics.green_operator.GreenOperator`), never here.

        On a carrying mesh this IS the ray-decoupled ``(L+C)``
        diagonal-block solve (zero-seed closure — the leg the M grid's
        block substitution consumes, ``ψ_A = LC⁻¹(q_A − Seeding·ψ_B)``;
        step 6 — presence is structural): the JOINT direct inverse ``M⁻¹``
        is the within-group grid's
        :meth:`~orpheus.numerics.coupled_system.CoupledOperator.solve`,
        never a kwarg channel on this surface.
        """
        del initial_guess  # accept-and-drop: exact direct inverse, nothing to seed (#280 2.5c)
        return self._solve_timed_full_field(rhs)

    # The moment-emitting entry is the typed windowed product ``P @ A.inverse()``
    # (:class:`~orpheus.sn.operators.windowing.WindowedSweep`), whose fused
    # ``apply`` calls :meth:`_solve_timed_full_field` with ``moment_frame`` — the
    # ONE private application-context body.  (A former public ``solve_moments``
    # whose output-mode argument silently changed the codomain was a composition
    # wearing a config — retired.)

    def _solve_timed_full_field(
        self,
        rhs: "FullField",
        *,
        moment_frame: "FrameBase | None" = None,
        schedule: "SweepSchedule | None" = None,
        reflect: "Callable[[AngularBoundaryFlux, tuple[str, ...]], None] | None" = None,
    ) -> "TimedFullField":
        r"""Composite :class:`TimedFullField` body of :meth:`solve`.

        Runs the WDD forward substitution on :attr:`loss_representation`
        — the operator's ONE
        :class:`~orpheus.sn.loss_representation.LossRepresentation` instance
        — and handles the field plumbing at the public-entry boundary: the
        sweep's mutable write-through ``boundary_buf`` is seeded per-face
        from ``rhs.boundary`` (works for slab, curvilinear, and 2-D
        Cartesian), the sweep mutates it in place, and the result is
        re-wrapped as a composite at the end.

        Parameters
        ----------
        rhs : FullField
            Per-ordinate source on the composite carrier (the timed
            iterate passes via inheritance; a bare ``FullField`` is the
            ``history_depth = 0`` degenerate).

        Returns
        -------
        TimedFullField
            Solve output with ``bulk`` = ``(L + C)^{-1} rhs.interior`` and
            ``boundary`` = the sweep's outflow face state.
            ``history_depth`` matches ``rhs.history_depth``; ``_history``
            is empty (solver outputs carry no iteration history — the
            outer SI / Krylov loop owns history threading).
        """
        from orpheus.transport.fields.angular_flux import (
            AngularFlux,
        )
        from orpheus.transport.fields.angular_boundary_flux import (
            AngularBoundaryFlux,
        )
        from orpheus.transport.fields.harmonic_moment_flux import (
            HarmonicMomentFlux,
        )
        from orpheus.transport.full_field import FullField
        from orpheus.transport.timed_full_field import TimedFullField

        # W-C (P4.5): the operator boundary speaks the timeless
        # :class:`FullField` composite; the timed iterate passes via
        # inheritance (``TimedFullField`` IS a ``FullField``), and a bare
        # ``FullField`` is admitted as the ``history_depth=0`` degenerate.
        # Legacy :class:`AngularFlux` stays retired — it is NOT a
        # ``FullField``, so the guard still rejects it.  Single guard site
        # for both :meth:`solve` and :meth:`solve_moments`.
        if not isinstance(rhs, FullField):
            raise TypeError(
                f"StreamingCollisionOperator: 'rhs' must be FullField; "
                f"got {type(rhs).__name__}.  Legacy AngularFlux retired "
                f"in D-H.2-C3."
            )
        sn_mesh = self.sn_mesh
        if rhs.interior.space != rhs.interior.space_on(sn_mesh):
            raise ValueError(
                "StreamingCollisionOperator.solve(FullField): rhs and "
                "operator must agree in space content "
                "(space-content invariant)."
            )

        # ── Boundary buffer for the sweep ─────────────────────────────
        #
        # The sweep mutates ``boundary_buf`` (mutable write-through:
        # ``frozen=True`` freezes field rebinding but the underlying flat
        # ndarray stays writable through :meth:`face_view`).
        #
        # BARE SWEEP (#208): the inflow seed is the boundary SOURCE
        # ``rhs.boundary`` (the inflow slots carry ``q.boundary + B·ψ.outflow``
        # — the SI driver folds ``S + B`` so the ``Bψ`` reflective inflow rides
        # in ``rhs.boundary``).  The bare sweep no longer re-applies ``bc`` to
        # any iterate's outflow; the curvilinear ψ½ starting direction is the
        # sweep's OWN direct computation from the source (#282), not a threaded
        # previous-iterate seed — the WDD sweep is an exact direct inverse.
        boundary_buf = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)
        seed_boundary = rhs.boundary
        # Per-face copy via L2 face_view — works for slab (xmin, xmax),
        # curvilinear (xmax only), and 2-D Cartesian (all 4).
        for face_name in boundary_buf.layout.faces:
            if face_name in seed_boundary.layout.faces:
                boundary_buf.face_view(face_name)[:] = (
                    seed_boundary.face_view(face_name)
                )

        # ── Sweep on the operator's ONE representation — the SAME
        # :class:`LossRepresentation` instance the matvec
        # (:meth:`StreamingOperator.apply`) consumes, so L21 ("matvec ≡
        # sweep") is a type fact, not two ``default_for`` calls agreeing.
        # ``rhs.interior.values`` IS the per-ordinate source by producer
        # contract (typed at the ``rhs`` guard above, so no wrap-unwrap round
        # trip through :class:`AngularSourceSink`).
        #
        # ONE sweep for BOTH output modes — the moment frame rides as an
        # optional kwarg (the 1-D representation raises on it; moment output
        # is 2-D Cartesian only).  Only the OUTPUT WRAP differs.
        bulk_values, _scalar = self.loss_representation.sweep(
            rhs.interior.values,
            self.sigma,
            boundary_buf,
            moment_frame=moment_frame,
            schedule=schedule,
            reflect=reflect,
        )

        # ── The outflow defect rows of the rhs (ERR-071) ──────────────
        # The forward's outflow-trace row is the DEFECT ``streamed −
        # ψ_out`` (sign pinned by the round-trip identity gate,
        # ``tests/sn/operators/test_sweep_inverse_identity.py``), so
        # the EXACT inverse emits ``ψ_out = streamed − rhs_out``.  The
        # march writes ``streamed`` into the buffer's outflow slots —
        # clobbering the seeded rhs copy — so the rhs's outflow-row
        # content is restored here.  Every physical rhs carries ZERO
        # there (the builders populate inflow slots only; outflow rows
        # are 0 = 0 identities at the fixed point), so this is bit-inert
        # on all SI/eigenvalue paths.  It is load-bearing for the
        # DSA-preconditioned GMRES posture (#2): Krylov residual
        # vectors exercise the FULL composite space, and the dropped
        # term made the preconditioner ``M = (I + 𝒞)∘(L+C)⁻¹`` SINGULAR
        # on the outflow-trace subspace (measured ‖M q‖/‖q‖ = 1e-15 on
        # a pure outflow-row vector — GMRES stalled at an O(1) true
        # residual and the exit certificate refused the claim).
        # Tangential rows (excluded from both selectors) keep their
        # seeded copy untouched — the identity-row inverse.
        trace_space = sn_mesh.angular_trace
        for face_name in boundary_buf.layout.faces:
            if face_name not in seed_boundary.layout.faces:
                continue
            out_rows = trace_space.outflow_indices_for_face(face_name)
            if out_rows.size:
                boundary_buf.face_view(face_name)[out_rows] -= (
                    seed_boundary.face_view(face_name)[out_rows]
                )
        # The sweep output carries the trailing 2^d spatial-moment axis at a
        # multi-moment closure (the φ̂ iterate, #240 D5b-S3); the typed wrap
        # selects the SpatialMomentSpace factor so the iterate is a legal typed
        # state.  DD/Step (per_axis == 1) → no factor, byte-identical.
        per_axis = sn_mesh.scheme.spatial_basis_per_axis
        if moment_frame is None:
            bulk = AngularFlux(
                values=bulk_values, space=sn_mesh.angular_trial_space,
            )
        else:
            # In moment mode the sweep returns the (L+1, 2L+1, ...) moment
            # tensor, so its own leading axis fixes L (no basis-specific read).
            bulk = HarmonicMomentFlux.from_mesh_and_L(
                bulk_values, sn_mesh, bulk_values.shape[0] - 1,
                spatial_moments=per_axis,
            )

        # ── L2 direct return — no adapter needed (D-H.2-C2).  On a
        # carrying mesh this is the ray-decoupled (A,A) block; the marched
        # ψ½ state is the M grid's OWN block member (step 6), never on
        # this composite. ──
        return TimedFullField(
            interior=bulk,
            boundary=boundary_buf,
            _history=(),
            history_depth=(
                rhs.history_depth if isinstance(rhs, TimedFullField) else 0
            ),
        )

    def solve_transpose(self, b: "FullField") -> "FullField":
        r"""Invert :math:`(L + C)^{\mathsf T}\,x = b` via the REVERSE-SCAN.

        The transpose-solve :math:`(L+C)^{-\mathsf T}` (#280 Phase 2.5b): the
        adjoint sibling of :meth:`solve`, exactly as :meth:`apply_transpose`
        (the matvec transpose :math:`(L+C)^{\mathsf T}`) is the sibling of
        :meth:`apply`.  Delegates to the representation's
        :meth:`~orpheus.sn.loss_representation._OneDimScanWalk.sweep_transpose`
        (the reverse-mode adjoint of the forward sweep-scan) and packs the
        transposed composite, completing the outflow defect rows of ``b``
        exactly as :meth:`solve` does (ERR-071: ``E_out`` is symmetric, so
        the transpose-inverse carries the SAME one-site restore — see the
        inline note below).

        **Duality typing (#276 A4):** the input ``b`` is the dual of the
        solve's codomain — dual-of-flux, i.e. an adjoint SOURCE
        (source-classed, the same composite geometry ``apply_transpose``
        emits); the output is the dual of its domain — dual-of-source under
        the G-pairing, i.e. the adjoint FLUX
        (:class:`~orpheus.transport.fields.angular_flux.AngularFlux` bulk +
        :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
        trace).  This is what lets the daggered fixed-point iteration close
        over the SAME class pattern as the forward (flux iterate → source
        gains → transpose-solve → flux iterate): the adjoint flux ψ* is a
        first-class importance field, not a bookkeeping cotangent.  (Until
        A4 the output was wrapped in the source-sink family — a matvec-gate
        era spelling that no class-sensitive consumer had ever exercised;
        the typed SI loop is the first, and the cross-class ``_check_partner``
        guard caught the mis-classing on first contact.)

        This is the reverse-scan primitive behind the ``.H.inverse()`` swap
        law (``A.H.inverse() ≡ A.inverse().H``).  1-D-scan-family only —
        DD since #280 2.5b, LD since #310 C2 (the ``_run_transpose``
        moment arm); the multi-D reverse-scan is the wavefront G-S
        schedule-reverse arm (R7, #310 deferred-out — no consumer; the
        representation raises).

        On a carrying mesh this is the ray-DECOUPLED ``(L+C)⁻ᵀ``
        diagonal-block transpose-solve (the M grid's transposed-
        substitution (A,A) leg; step 6 — presence is structural): the
        JOINT inverse ``M⁻ᵀ`` is the grid's
        :meth:`~orpheus.numerics.coupled_system.CoupledOperator.solve_transpose`,
        never a kwarg channel on this surface.
        """
        from orpheus.transport.fields.angular_boundary_flux import (
            AngularBoundaryFlux,
        )
        from orpheus.transport.fields.angular_flux import AngularFlux
        from orpheus.transport.full_field import FullField

        sn_mesh = self.sn_mesh
        if b.interior.space != b.interior.space_on(sn_mesh):
            raise ValueError(
                "StreamingCollisionOperator.solve_transpose(FullField): b and "
                "operator must agree in space content "
                "(space-content invariant)."
            )
        q_bar, m_boundary = self.loss_representation.sweep_transpose(
            b.interior.values,
            self.sigma,
            b.boundary,
        )
        # Duality typing (#276 A4, docstring above): dual-of-source = the
        # adjoint FLUX — wrap flux-classed so the daggered iteration closes.
        # The rep layer's ``m_boundary`` carries the values; the class ROLE
        # is this operator boundary's decision.
        boundary_out = AngularBoundaryFlux(
            values=np.asarray(m_boundary.values),
            space=sn_mesh.angular_trace,
        )
        # ── The outflow defect rows of ``b`` (ERR-071, transpose half) ──
        # The solve half establishes the EXACT inverse as ``A⁻¹ = S_old −
        # E_out`` with ``E_out`` the diagonal partial identity on the
        # FORWARD-sense outflow-trace rows (the post-march restore in
        # :meth:`_solve_timed_full_field`).  ``E_out`` is diagonal ⟹
        # symmetric, so the exact transpose-inverse is ``(Aᵀ)⁻¹ =
        # (A⁻¹)ᵀ = S_oldᵀ − E_out`` — the SAME one-site restore, on the
        # SAME forward-outflow selector, applied to the reverse-scan's
        # output boundary.  (The reverse-scan IS ``S_oldᵀ`` exactly: the
        # G3 full-composite reciprocity gates pinned that identity on
        # random boundaries pre-fix, and red the completion's absence.)
        # Physical adjoint paths are bit-inert here for the same reason
        # the forward is: their sources carry zero on these rows.
        trace_space = sn_mesh.angular_trace
        for face_name in boundary_out.layout.faces:
            if face_name not in b.boundary.layout.faces:
                continue
            out_rows = trace_space.outflow_indices_for_face(face_name)
            if out_rows.size:
                boundary_out.face_view(face_name)[out_rows] -= (
                    b.boundary.face_view(face_name)[out_rows]
                )
        return FullField(
            interior=AngularFlux(
                values=q_bar, space=sn_mesh.angular_trial_space,
            ),
            boundary=boundary_out,
        )
