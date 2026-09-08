r"""Selectable representations of the S\ :sub:`N` loss operator :math:`(L+C)`.

The within-group transport solve :math:`\psi = (L+C)^{-1} q` (the *sweep*)
and its operator twin :math:`(L+C)\,\psi` (the *matvec*) admit several
distinct *algorithms*, each natural for a different mesh:

* a **1-D parallel-prefix scan** (Blelloch 1990 §1.5) — the geometry-blind
  chain recurrence (slab + sphere + cylinder), :meth:`._OneDimScanWalk.sweep`;
* a **multi-D wavefront walk** over the per-octant anti-hyperplane DAG
  (:meth:`SweepDependencyGraph.walk_full` /
  :meth:`~SweepDependencyGraph.walk_windowed`), in two buffer policies — a
  full-field buffer (the slow, readable verification oracle) and a rolling
  :math:`(d{-}1)`-frontier window (the fast production path).

Historically the choice between them was a *scattered, procedural* branch
spelled three different ways (an operator-free ``transport_sweep`` entry, five
matvec operator gates, hand-built oracle adapters), so adding a method or a
dimensionality meant touching every call site.

This module replaces that with a first-class :class:`LossRepresentation`: each
algorithm is an object that carries **both** the forward ``sweep`` and (from
Phase S2) the ``loss_action`` matvec twin, plus a **declared, queryable**
:meth:`~LossRepresentation.supports` predicate.  The operator selects one
via :func:`default_for` and then calls it branchlessly.  (The module-level
``transport_sweep`` wrapper retired at step 6 with the walk's ψ½ joint
channel: the typed surfaces are
:meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.solve` for the
``(L+C)`` block and the within-group M grid's
:meth:`~orpheus.numerics.coupled_system.CoupledOperator.solve` for the
joint march.)

The hierarchy
=============

.. code-block:: text

    LossRepresentation (Protocol: sweep, loss_action, supports)
    ├── _DAGWavefront            ── Cartesian anti-hyperplane DAG family
    │   ├── FullFieldWavefront     buffer = full field     · the ORACLE
    │   └── MovingFrontierWindow   buffer = rolling frontier · production opt
    └── CumprodScan             ── 1-D chain prefix scan, any geometry

``FullFieldWavefront`` and ``MovingFrontierWindow`` consume the **same**
per-octant DAG (the family-owned ``sweep_graphs`` accessor, cached per
mesh shape since S6.4(c)) — they are two *buffer policies* over one
anti-hyperplane walk, already pinned bit-identical by the C3.2b
``window ≡ full`` oracle.  ``CumprodScan`` builds no DAG: a 1-D chain is a
total order, the Blelloch closed form needs no graph.

The governing principle
========================

    *Construct each strategy as general as its algorithm naturally allows;
    select narrow; specialize the implementation only on a measured internal
    performance cost.*

So an algorithm's **capability** (what it CAN express — e.g. the moving frontier
is naturally d-general, ``CumprodScan`` intrinsically 1-D) is kept separate from
**policy** (what :meth:`~LossRepresentation.supports` / :func:`default_for`
*recommend* at a given ``(geometry, ndim)``): "don't pick the window at d=1" is a
selection recommendation, never a reason to make the window unable to express d=1.
Only a *measured* hot-path regression justifies restricting an implementation's
d-range.  The full three-layer rationale is on the theory page —
``docs/theory/methods/sn/loss_representation.rst §loss-rep-selection``.

Selection is a single source of truth
======================================

:meth:`~LossRepresentation.supports` returns :class:`Compatibility` — an
``(ok, reason)`` pair.  The same predicate serves three consumers:

#. a (future) teaching frontend — ``[S for S in LOSS_REPRESENTATIONS if
   S.supports(mesh).ok]`` grays-out an inapplicable method *and explains
   why* (pedagogically load-bearing — ORPHEUS teaches reactor physics);
#. the factory :func:`default_for` — picks the best *available* production
   optimization, falling back to the full-field spine when no optimization
   exists, so it is never stuck;
#. the construction guard — :meth:`_LossRepresentation.__post_init__` raises
   :class:`IncompatibleRepresentation` on an illegal pairing, so even a bypassed
   UI cannot build one.

The compatibility signal is the genuine criterion — the coordinate system
(:attr:`SNMesh.is_cartesian`) and the dimensionality (:attr:`SNMesh.ndim`)
— NOT the ``sweep_graphs is None`` substrate proxy.

Carve history
=============

The S0–S6.9 arc that produced this module (the protocol + thin-wrapper
strategies, the ``loss_action`` matvec twin, the d-generic ``FullFieldWavefront``
oracle, the ``frontier_dim = d-1`` window, the one-walk unification, and the
S6.9 Fork-B2 default flip window → ``ScanMarch``) is recorded on the theory page
— ``docs/theory/methods/sn/loss_representation.rst §loss-rep-history``.

See also
========

* :doc:`/theory/methods/sn/loss_representation` — the capstone architecture page
  (the native lower-triangular frame, the four schedules, the selection
  SSOT, the one-walk/one-instance theorems, the Fork-B2 evidence).
* ``.claude/plans/sn_sweep_strategy.md`` — the authoritative design (the
  locked decisions, the verification strategy, phases S0–S6.9).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from weakref import WeakKeyDictionary

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

# S6.4(f): the orchestration (the schedule loop + the 1-D unified body)
# lives IN this module — ``sweep.py`` dissolved; selector and bodies share
# one home, so the historical load-time import cycle is gone.
from orpheus.geometry import CoordSystem
from orpheus.geometry.boundary import SelfPairedDeck
from orpheus.numerics.moment_layout import (
    AVERAGE_MOMENT,
    cell_moment_count,
    face_moment_count,
    face_moment_tail,
    is_moment_valued_by_flat_rank,
    is_moment_valued_by_rank,
)

from orpheus.transport.spatial._ubld import octant_moment_frame_signs
from orpheus.transport.spatial.scheme import UpstreamState
from ..sweep.scan import (
    _scanmarch_row,
    _x_scan_faces,
    _x_scan_faces_transpose,
    ordinate_scan,
    ordinate_scan_transpose,
)
from ..sweep.cache import CollisionCache, StreamingCoefficientCache
from orpheus.numerics.face_layout import face_name, face_opposite
from .sweep_graph import (
    OctantLabel,
    SweepDependencyGraph,
    _CellResidual,
    _CellResidualTranspose,
    _CellSolve,
    _CellSolveAngular,
    _CellSolveMoment,
    _reframe,
)
from .sweep_schedule import SweepSchedule

#: Source-free ``Q_cells`` for the matvec apply (the operator action ``(L+C)ψ̄``
#: carries no volumetric source).  Shared read-only ``(1,1,1)`` zero broadcast
#: into ``residual_kernel_batch`` — the kernel only READS ``Q_cells`` (it never
#: mutates it), so one shared instance is safe and avoids a per-cell allocation.
_MATVEC_ZERO_SOURCE = np.zeros((1, 1, 1))


def frame_signs_for(
    scheme: "DiscretizationSchemeBase", signs: tuple[int, ...],
) -> "np.ndarray | None":
    r"""Sweep⇄global moment-frame sign vector bound to a scheme — or ``None``.

    The ONE binding of an octant's ``signs`` and the scheme's
    ``spatial_basis_per_axis`` to the single-source involution
    :func:`~orpheus.transport.spatial._ubld.octant_moment_frame_signs` (#240 D5b-S3 —
    the diffusion-limit root cause; the primitive owns the ``None``-for-DD/Step
    no-op convention).  Hoisted to module level so BOTH the
    :class:`_LossRepresentation` cell-op frames (via
    :meth:`_LossRepresentation._moment_frame_signs`) AND the
    :class:`_OneDimScanWalk` 1-D scan/matvec sites call ONE binding —
    ``_OneDimScanWalk`` does not inherit ``_LossRepresentation`` and so cannot
    reach the method (Pattern 2 — the last binding-dup the per-axis read was
    repeated across).
    """
    return octant_moment_frame_signs(signs, scheme.spatial_basis_per_axis)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from orpheus.numerics.frame import FrameBase
    from orpheus.transport.fields._bases import BoundaryField
    from orpheus.transport.fields.angular_flux import AngularFlux
    from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
    from orpheus.transport.full_field import FullField
    from orpheus.transport.source_sinks import (
        AngularSourceSink,
        AngularBoundarySourceSink,
    )
    from orpheus.transport.timed_full_field import TimedFullField

    from ..angular.closure import AngularClosureBase
    from ..mesh.augmented_mesh import SNMesh
    from ..operators.streaming import StreamingOperator
    from orpheus.transport.spatial.scheme import DiscretizationSchemeBase
    from .sweep_schedule import OctantSweep, OctantSweepGroup


# ═══════════════════════════════════════════════════════════════════════
# Selection vocabulary
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Compatibility:
    """Whether a strategy applies to a mesh, with a human-readable reason.

    ``ok`` is the machine-checkable verdict; ``reason`` is the explanation a
    frontend shows when graying-out an inapplicable method ("Moving-frontier
    window — requires Cartesian geometry, d = 2") and the message the
    construction guard raises.  ``reason`` is the empty string when
    ``ok is True`` (no explanation needed).
    """

    ok: bool
    reason: str


class IncompatibleRepresentation(ValueError):
    """A :class:`LossRepresentation` was constructed for a mesh it cannot sweep.

    Raised by the construction guard (:meth:`_LossRepresentation.__post_init__`)
    so that an illegal ``(strategy, mesh)`` pairing is unrepresentable —
    even if a caller bypasses :func:`default_for`.
    """


def _curvilinear_capability(
    mesh: "SNMesh", spatial_closure: "DiscretizationSchemeBase",
) -> Compatibility:
    r"""The (scheme × geometry) curvilinear-capability gate — single source.

    A curvilinear (sphere/cylinder) mesh needs a scheme whose cell closure
    handles the Morel–Montry angular-redistribution thread; a slab/Cartesian-
    only scheme (Linear-Discontinuous today — the curvilinear LD closure is
    not yet implemented, #158/#6) must be rejected at SELECTION, not raised mid-sweep.
    Cartesian meshes are unconstrained (every scheme handles :math:`\mu\partial_x`).

    Consumed by the 1-D scan selectors (:meth:`CumprodScan.supports` /
    :meth:`ScanMarch.supports`) and :func:`default_for` so that
    ``mesh.scheme.is_affine_scannable`` — a geometry-blind 1-D trait — cannot
    license a curvilinear sweep the scheme has no closure for (#236 ST2; the
    dishonest-selection fix).
    """
    if mesh.is_cartesian or spatial_closure.supports_curvilinear:
        return Compatibility(True, "")
    return Compatibility(
        False,
        f"{type(spatial_closure).__name__} has no curvilinear cell closure "
        "(slab/Cartesian only); the curvilinear (sphere/cylinder) closure for "
        "this scheme is not yet implemented (Issue #158 curvilinear arm / #6)",
    )


@runtime_checkable
class LossRepresentation(Protocol):
    r"""One algorithm for the within-group transport solve and its twin.

    A strategy is constructed *for a mesh* (``Strategy(mesh)``); the
    construction guard rejects an incompatible pairing.  It then exposes:

    * :meth:`sweep` — one forward substitution :math:`\psi = (L+C)^{-1} q`;
    * :meth:`loss_action` — the matvec twin :math:`(L+C)\,\psi` *(added in
      Phase S2)*;
    * :meth:`supports` — the (classmethod) selection predicate.
    """

    def sweep(
        self,
        Q: "np.ndarray",
        sig_t: "np.ndarray",
        boundary_flux: "AngularBoundaryFlux",
        *,
        moment_frame: "FrameBase | None" = None,
        schedule: "SweepSchedule | None" = None,
        reflect: "Callable[[AngularBoundaryFlux, tuple[str, ...]], None] | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray | None]":
        """Perform one within-group transport sweep on this strategy's mesh.

        Returns the mode-keyed pair: ``(angular_flux, scalar_flux)`` for the
        full-angular default, ``(moment_buf, None)`` when a ``moment_frame``
        is given (the windowed-SI moment path; the scalar is subsumed as
        ``moment_buf[0, 0]``).  Strategies that reject moment output
        (``CumprodScan``) declare the narrower always-angular pair.

        ``schedule``/``reflect`` (#226 step 2): ``None`` (default) is the
        bare Jacobi sweep; a given schedule runs the SAME uniform
        sweep-and-reflect loop with the inter-group ``reflect`` — the
        forward substitution of the reified ``M = (L+C−B_lower)``.
        Multi-D only; the 1-D scan raises (not a wavefront).

        On a carrying mesh (R12a) this is the ray-DECOUPLED ``(L+C)``
        block solve (zero-seed closure; step 6 — presence is structural):
        the JOINT ψ½ march is the within-group M grid's
        :meth:`~orpheus.numerics.coupled_system.CoupledOperator.solve`,
        never a leg channel on this surface.
        """
        ...

    def sweep_transpose(
        self,
        bulk_cot: "np.ndarray",
        sigma: "np.ndarray",
        boundary_cot: "BoundaryField",
    ) -> "tuple[np.ndarray, AngularBoundarySourceSink]":
        r"""The transpose-solve :math:`(L+C)^{-\mathsf T}` — the REVERSE-SCAN.

        The solve-scan frame's adjoint (#280 Phase 2.5b): the transpose sibling
        of :meth:`sweep`, exactly as :meth:`loss_action_transpose` is the adjoint
        sibling of :meth:`loss_action`.  Consumes the composite cotangent
        (``bulk_cot`` on :math:`\bar\psi`, ``boundary_cot``) and returns
        ``(Q_bar, m_boundary)`` — the reverse-mode adjoint of the
        forward sweep-scan, sharing its ``ordinate_scan`` substrate via
        :func:`~orpheus.sn.sweep.scan.ordinate_scan_transpose`.  On a
        carrying mesh this is the ray-decoupled block's reverse-scan (step
        6); the joint ``M⁻ᵀ`` is the grid's transposed substitution.  Raises
        :class:`NotImplementedError` for representations / geometries whose
        reverse-scan is deferred (the multi-D wavefront G-S
        schedule-reverse — R7, the #280 sibling deferral, no consumer;
        the lagged-seed cylinder until its forward fundamental fix).
        Distinct from the matvec transpose
        :meth:`loss_action_transpose`, complete on the registered grid
        since #310 C4/C5; the 1-D LD moment-tailed reverse-scan is LIVE
        since #310 C2.  Never a silent wrong answer.
        """
        ...

    def loss_action(
        self, sigma: "np.ndarray", psi: "FullField",
    ) -> "FullField":
        r"""The forward within-group loss action :math:`(L+C)\,\psi` for this geometry.

        The sweep's operator-twin (L21 — sweep and matvec are different
        applications of the SAME operator): the sweep solves
        :math:`(L+C)^{-1} q`, this APPLIES :math:`(L+C)`.  **Return the FULL loss
        :math:`(L+C)\psi` for the given** ``sigma``, **NOT** :math:`L\psi`.  The
        two operator doors are two readings of THIS action:
        :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.apply`
        passes the composite's own :math:`\sigma` and returns the result
        directly, while
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` passes
        :math:`\sigma = 0` (via ``streaming_action``) and so gets bare
        :math:`L\psi` — the Resolution-A identity :math:`L = (L+C) - C`
        realised as the :math:`\sigma`-free reading, NOT as a subtraction
        (#257 S8b).  A leaf that ignored ``sigma`` and returned :math:`L\psi`
        would therefore drop :math:`C` out of the composite's matvec while
        ``solve`` kept it — matvec and sweep no longer the same operator
        (L21 broken).  ``sigma`` is the ``(ng, ...)`` diagonal coefficient the matvec
        realises (#240 Phase 2 Step B — passed EXPLICITLY, symmetric with
        :meth:`sweep`'s ``sig_t``, so the composite — not the leaf — single-sources
        :math:`\sigma`); the per-geometry walk machinery is on ``self.mesh``.

        On a carrying mesh this is the ray-decoupled ``(A,A)`` BLOCK action
        (bit-identical to a zero seed — the walk's welded feed reads zeros;
        step 6 — presence is structural): the JOINT action ``M·[ψ_A, ψ_B]``
        is the within-group M grid's
        :meth:`~orpheus.numerics.coupled_system.CoupledOperator.apply`.
        """
        ...

    def streaming_action(self, psi: "FullField") -> "FullField":
        r"""The pure σ-free streaming action :math:`L\,\psi = \Omega\cdot\nabla\psi`.

        The genuine pure-:math:`L` leaf — spatial streaming + curvilinear
        angular redistribution, NO collision diagonal.  Single-sourced through
        :meth:`loss_action` at :math:`\sigma = 0`, because the within-group WDD
        matvec is AFFINE in :math:`\sigma`
        (:math:`(L+C)\psi = \text{streaming\_action}(\psi) + \sigma\cdot\psi`; the
        σ-affine decomposition — including why the curvilinear Carlson coupled-pole
        seed's :math:`\sigma`-dependence cancels into :math:`\sigma\cdot\psi` — is
        derived at
        ``docs/theory/methods/sn/loss_representation.rst §loss-rep-removal-form-matvec``).
        Pattern 2 — the streaming walk lives ONCE in ``loss_action``; there is no
        twin σ-free discretization.
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` calls this directly
        (#257 S8b) so :math:`L` reads no :math:`\sigma`: the collision diagonal
        :math:`C = M[\sigma_t]` is the separate shared multiplier leaf, and the
        composition :math:`L + C` recovers the full loss.
        """
        ...

    def loss_action_transpose(
        self, sigma: "np.ndarray", phi: "FullField",
    ) -> "FullField":
        r"""The adjoint loss action :math:`(L+C)^{\mathsf T}\,\phi` for this geometry.

        Return the FULL adjoint loss :math:`(L+C)^{\mathsf T}\phi` for the given
        ``sigma``.  :math:`C = \sigma\odot` is a self-adjoint diagonal, so the
        adjoint matvec is affine in :math:`\sigma` exactly as the forward one
        is, and
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply_transpose`
        recovers bare :math:`L^{\mathsf T}\phi` by calling this at
        :math:`\sigma = 0` (via ``streaming_action_transpose``) — not by
        subtracting (#257 S8b).
        Implemented by EVERY registered representation on EVERY registered
        scheme since #310 C5 (the 1-D reverse walks, the mirror-octant
        wavefront reverses — DD and LD, any ``d`` — and the row-march
        reverse); a scheme that registers no transpose kernel pair still
        refuses with a typed :class:`NotImplementedError` (the
        ``has_transpose_kernel`` covering law) — never a silent wrong
        answer.  ``sigma`` is the ``(ng, ...)`` diagonal coefficient,
        passed EXPLICITLY (#240 Phase 2 Step B).

        On a carrying mesh this is the ray-decoupled ``(A,A)ᵀ`` block action
        (step 6 — presence is structural): the JOINT transposed action
        ``Mᵀ`` — including the seed pullback ``Seedingᵀ·φ_A`` — is the
        within-group M grid's
        :meth:`~orpheus.numerics.coupled_system.CoupledOperator.apply_transpose`;
        the ``A_ABᵀ`` pullback belongs to the explicit grid block
        (:meth:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicSeeding.apply_transpose`).
        """
        ...

    def streaming_action_transpose(self, phi: "FullField") -> "FullField":
        r"""The pure σ-free adjoint streaming action :math:`L^{\mathsf T}\,\phi`.

        The transpose sibling of :meth:`streaming_action` (#257 S8b): the σ-free
        :math:`L^{\mathsf T}` leaf, single-sourced through
        :meth:`loss_action_transpose` at :math:`\sigma = 0`.  Used by
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply_transpose`.  Inherits
        that method's contract (implemented on the full scheme ×
        representation grid since #310 C5; an unregistered-kernel scheme
        raises typed — never a silent wrong answer).
        """
        ...

    @property
    def has_transpose_walk(self) -> bool:
        r"""Whether THIS representation can walk its traversal in reverse.

        The ORIENTATION factor of the adjoint-reachability predicate (#280,
        Phase 2.5a): ``is_adjointable = scheme.has_transpose_kernel ∧
        representation.has_transpose_walk`` — the scheme trait says the
        per-cell relation has a transpose realization (the KERNEL axis);
        this trait says the walk itself reverses (the ORIENTATION axis).
        Every REGISTERED representation answers ``True`` since #310 C4/C5
        (the 1-D reverse walks, the mirror-octant wavefront reverses, the
        row-march reverse); the trait remains the honest gate for a FUTURE
        representation without a reverse walk: ``False`` makes the eager
        ``.H`` refuse at construction
        (:class:`~orpheus.numerics.operator.MissingAdjoint`), with the
        representation's own typed raise as the loud backstop for direct
        Euclidean calls.
        """
        ...

    @classmethod
    def supports(
        cls, mesh: "SNMesh", spatial_closure: "DiscretizationSchemeBase",
    ) -> Compatibility:
        """Whether this strategy can sweep ``mesh`` with ``spatial_closure``.

        Selection consumes the HANDED closure (P4.9b Q4), never
        ``mesh.scheme``.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════
# Common base — the construction guard (illegal pairings unrepresentable)
# ═══════════════════════════════════════════════════════════════════════


#: P4.9b Q1 ruling — the strategy layer OWNS the interned Stratum-1 table.
#: Keyed weakly on the hub (one entry per mesh), VALIDATED against the
#: angular-closure identity (a doctored pair handed at pose gets its own
#: build — the keystone's surviving-cache trap dissolves).  The mechanism
#: lives in THIS retirement-bound layer deliberately: when Campaign 2
#: replaces the consumer side with a lazy solution strategy, the interim
#: interning dies with the layer it serves (the user's
#: survives-the-lazy-strategy criterion; `scratch/p4_9b_design.md` §9).
#: The COUNT gate (`tests/sn/sweep/core/test_cache.py`) pins builds-per-
#: solve == 1 — the F2-measured hazard (6-10 operators/solve × 8.78 ms).
_GEOM_CACHE_INTERN: "WeakKeyDictionary[SNMesh, tuple[AngularClosureBase, StreamingCoefficientCache]]" = (
    WeakKeyDictionary()
)


def geometry_cache_for(
    mesh: "SNMesh", angular_closure: "AngularClosureBase",
) -> StreamingCoefficientCache:
    """The lazily-resolved, hub-interned geometry table (Stratum 1).

    σ-free (geometry × quadrature — since P4b the table carries no
    closure algebra at all), so its lifetime is the hub's; the
    closure-identity validation rebuilds for a different handed closure
    (post-P4b the rebuilt table is bit-identical — the validation is
    retained as the intern's declared key, and it dissolves with this
    layer at Campaign 2).  Consumers: the walk's ensure path (lazy, first
    sweep) and the solver's σ-stratum posing (which needs Stratum 1 to
    build :class:`CollisionCache`).
    """
    entry = _GEOM_CACHE_INTERN.get(mesh)
    if entry is not None and entry[0] is angular_closure:
        return entry[1]
    cache = StreamingCoefficientCache.from_mesh_and_quad(mesh)
    _GEOM_CACHE_INTERN[mesh] = (angular_closure, cache)
    return cache


@dataclass(frozen=True)
class _LossRepresentation:
    """Base for every concrete strategy: the mesh + the two closures + the guard.

    A frozen dataclass carrying the state every strategy needs — the
    :class:`SNMesh` (the geometric substrate) and, since P4.9b step 2, the
    TWO CLOSURES the posed operator holds: the walk consumes the closure
    pair it is HANDED, never the hub's attributes (the keystone route gate
    ``tests/sn/operators/test_operator_feeds_the_walk.py`` pins it; the
    read-set gate bounds the residual hub route to the two space facts).
    """

    mesh: "SNMesh"
    spatial_closure: "DiscretizationSchemeBase"
    angular_closure: "AngularClosureBase"

    @classmethod
    def pose(cls, mesh: "SNMesh") -> "_LossRepresentation":
        """Pose from the hub's own objects — the test-side intermediate.

        Production hands the pair explicitly (the posed operator's
        fields, ``streaming.py`` ``loss_representation``); tests posing a
        bare representation read the hub here, mirroring
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.pose`.
        """
        return cls(mesh, mesh.scheme, mesh.angular_closure)

    @classmethod
    def supports(
        cls, mesh: "SNMesh", spatial_closure: "DiscretizationSchemeBase",
    ) -> Compatibility:
        """The selection predicate — every concrete strategy implements it.

        Selection consumes the HANDED spatial closure (P4.9b Q4 ruling:
        strategy-selection predicates are operator-side), never
        ``mesh.scheme``.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement supports()"
        )

    # ── pure-L streaming primitive (#257 S8b) ────────────────────────────
    # The σ-free streaming leaf, single-sourced through loss_action at σ = 0.
    # The within-group WDD matvec is affine in σ (the curvilinear Carlson
    # coupled-pole seed's σ-dependence is exactly the collision diagonal it
    # injects, so it cancels into σ·ψ), hence loss_action(0, ψ) == Ω·∇ψ.  The
    # streaming discretization lives ONCE in loss_action; there is no twin
    # σ-free walk (coding-elegance Pattern 2).  These concrete defaults work
    # for every strategy because each subclass overrides loss_action /
    # loss_action_transpose, and Python method resolution dispatches the
    # self.loss_action call to the right walk.

    if TYPE_CHECKING:
        # Abstract signatures for the type checker only — every concrete
        # strategy implements these (the base is never instantiated directly).
        # Declared under TYPE_CHECKING so they do NOT create a runtime method
        # that the subclass overrides would "obscure" (reportRedeclaration);
        # at runtime ``streaming_action``'s ``self.loss_action`` resolves to the
        # concrete subclass via normal MRO.
        def loss_action(
            self, sigma: "np.ndarray", psi: "FullField",
        ) -> "FullField": ...

        def loss_action_transpose(
            self, sigma: "np.ndarray", phi: "FullField",
        ) -> "FullField": ...

    def streaming_action(self, psi: "FullField") -> "FullField":
        r"""Pure σ-free forward streaming :math:`L\,\psi = \Omega\cdot\nabla\psi`.

        See the :meth:`LossRepresentation.streaming_action` protocol docstring.
        """
        return self.loss_action(self._zero_sigma_for(psi), psi)

    def streaming_action_transpose(self, phi: "FullField") -> "FullField":
        r"""Pure σ-free adjoint streaming :math:`L^{\mathsf T}\,\phi`.

        See :meth:`LossRepresentation.streaming_action_transpose`.  Inherits the
        deferral contract of :meth:`loss_action_transpose`.
        """
        return self.loss_action_transpose(self._zero_sigma_for(phi), phi)

    def sweep_transpose(
        self,
        bulk_cot: "np.ndarray",
        sigma: "np.ndarray",
        boundary_cot: "BoundaryField",
    ) -> "tuple[np.ndarray, AngularBoundarySourceSink]":
        r"""Reverse-scan default — DEFERRED (the #280 2.5b kernel-pair contract).

        The transpose-solve :math:`(L+C)^{-\mathsf T}` is realised only by the
        1-D scan family (:class:`CumprodScan` overrides this).  Every other
        representation (multi-D Cartesian wavefront / windowed) inherits this
        loud deferral — the reverse-scan of a wavefront is the G-S
        schedule-reverse arm (#310 R7, no consumer); never a silent wrong
        answer.  (Distinct from the matvec transpose
        :meth:`loss_action_transpose`, which every representation
        implements on every registered scheme since #310 C4/C5.)
        """
        raise NotImplementedError(
            f"{type(self).__name__}.sweep_transpose: the transpose-solve "
            "(L+C)⁻ᵀ reverse-scan is 1-D-scan-only (#280 Phase 2.5b); the "
            "multi-D wavefront reverse-scan is a deferred kernel-pair arm."
        )

    def _zero_sigma_for(self, field: "FullField") -> "np.ndarray":
        r"""The zero diagonal coefficient :math:`\sigma = 0` matching ``field``'s
        ``(ng, *spatial)`` shape — the σ-free streaming probe.

        The group count is read off the carrier (``field.interior.values`` is the
        ``(N, ng, *spatial)`` angular flux); the spatial shape is the mesh's.
        """
        ng = int(field.interior.values.shape[1])
        return np.zeros((ng, *self.mesh.spatial_shape))

    @property
    def _n_face_moments(self) -> int:
        r"""Per-face transverse moment count :math:`(\text{per\_axis})^{d-1}`.

        The interior face cochain (full or windowed) carries this many moments
        per face for the selected scheme: ``1`` for the slopeless cell-average
        closures (DD/Step → byte-identical scalar faces) and ``2^{d-1}`` for the
        bilinear UBLD Linear-Discontinuous closure (#240 D5b — d=2: 2).  Reads
        the multi-moment face-cochain width from the single-source
        :func:`~orpheus.numerics.moment_layout.face_moment_count` (shared with the
        trace producer :meth:`~orpheus.sn.mesh.augmented_mesh.SNMesh.boundary_face_layout`)."""
        return face_moment_count(self.spatial_closure.spatial_basis_per_axis, self.mesh.ndim)

    def _moment_frame_signs(
        self, signs_eff: tuple[int, ...],
    ) -> "np.ndarray | None":
        r"""Octant sweep⇄global moment-frame sign vector — or ``None`` for DD/Step.

        The cell kernel works in the per-ordinate SWEEP frame; the iterate
        ``φ̂`` + its scattering source ``Σ_s·φ̂`` live in the GLOBAL frame, so the
        ``_CellSolve`` / ``_CellResidual`` level operations re-sign the slope
        moments at the octant boundary (#240 D5b-S3 — the diffusion-limit root
        cause).  Thin delegate to the module-level :func:`frame_signs_for`
        binding (this octant's ``signs_eff`` + the scheme's ``per_axis``), the
        ONE site that binds the involution to a scheme — shared with the
        :class:`_OneDimScanWalk` 1-D scan/matvec sites (which cannot reach this
        method).
        """
        return frame_signs_for(self.spatial_closure, signs_eff)

    @property
    def _spatial_moment_tail(self) -> tuple[int, ...]:
        r"""Trailing per-CELL spatial-moment axis shape :math:`(\text{per\_axis})^d`.

        The bulk-field analogue of :attr:`_n_face_moments` (the FACE tail is
        ``per_axis^{d-1}``; the CELL tail is ``per_axis^d``).  A multi-moment
        closure (LD, ``per_axis > 1``) carries this trailing axis on the
        iterate / source / probe / residual / angular-octant buffers so the
        spatial-moment iterate ``φ̂`` travels between sweeps (#240 D5b-S3 — the
        unified moment matvec).  DD/Step (``per_axis == 1``) → ``()`` (no axis;
        every buffer stays byte-identical — the negative control).  Single
        source via :func:`~orpheus.numerics.moment_layout.face_moment_tail` (the same
        "append iff > 1" policy ``spatial_moment_tail`` delegates to), fed the
        per-CELL count ``per_axis^d``."""
        per_axis = self.spatial_closure.spatial_basis_per_axis
        return face_moment_tail(cell_moment_count(per_axis, self.mesh.ndim))

    def _inflow_to_moments(
        self, inflow: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", ...]:
        r"""Carry the per-face domain inflow as a ``2^{d-1}``-moment object.

        A multi-moment closure's cochain consumes a ``2^{d-1}``-transverse-moment
        domain inflow per face (per-axis Legendre order ``[bar, slope, …]``); the
        boundary trace supplies it.  This is the boundary twin of Leg A's bulk
        :func:`~orpheus.sn.solver._lift_external_source_to_moments` — it
        rank-DISCRIMINATES the incoming face against the flat (moment-free) face
        rank ``d + 1`` (a scalar face ``(N_oct, ng, *transverse)`` has rank
        ``2 + (d−1) = d + 1``; transverse carries ``d−1`` axes):

        * **single-moment closure** (DD/Step, ``n == 1``): identity — the trailing
          axis is absent, every buffer byte-identical.
        * **scalar inflow** (a vacuum face / the existing scalar prescribed
          inflow): widen — zeros buffer, the AVERAGE moment (slot 0) seeded by
          the scalar, the transverse slopes ZERO (a scalar trace carries no
          along-face variation; the Leg-B asymmetry — the scalar default is
          correctly blind to the transverse slope).
        * **moment-resolved inflow** (the #251 moment-resolved boundary trace,
          ``2^{d-1}``-valued): PASS THROUGH — the producer (the widened trace,
          ``geometry.boundary_face_layout``) already carries the projected
          transverse face-slope; this method threads it unchanged.  Validates the
          trailing transverse-moment width == ``2^{d-1}`` (a clear ValueError
          otherwise — the moment-resolved relaxation must not swallow a real shape
          bug; ``coding-elegance`` Pattern 4).

        The projection that fills slot 1 lives at the call site (the MMS / the
        trace producer), NEVER here — production accepts the moment-resolved face,
        does not compute it (Pattern 6, L11 structural independence, exactly Leg A).
        """
        n = self._n_face_moments
        if n == 1:
            return inflow
        # A scalar face is (N_oct, ng, *transverse); transverse carries d−1 axes,
        # so its flat rank is 2 + (d − 1) = ndim + 1.  A moment-resolved face
        # carries one MORE (the trailing 2^{d-1}-moment) axis (#251).
        flat_face_ndim = self.mesh.ndim + 1
        widened = []
        for face in inflow:
            if is_moment_valued_by_flat_rank(face, flat_face_ndim):
                _assert_face_moment_width(
                    face, n, where=f"{type(self).__name__}._inflow_to_moments",
                )
                widened.append(face)            # thread the projected slope through
            else:
                buf = np.zeros((*face.shape, n))
                buf[..., AVERAGE_MOMENT] = face  # average moment ← scalar inflow
                widened.append(buf)
        return tuple(widened)

    def sweep(
        self,
        Q: "np.ndarray",
        sig_t: "np.ndarray",
        boundary_flux: "AngularBoundaryFlux",
        *,
        moment_frame: "FrameBase | None" = None,
        schedule: "SweepSchedule | None" = None,
        reflect: "Callable[[AngularBoundaryFlux, tuple[str, ...]], None] | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray | None]":
        """One within-group sweep — every concrete strategy implements it."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement sweep()"
        )

    def loss_action(
        self, sigma: "np.ndarray", psi: "FullField",
    ) -> "FullField":
        """The forward loss action ``(L+C)ψ`` — every concrete leaf implements it.

        Returns the FULL within-group loss ``(L+C)ψ`` for the given ``sigma``
        (NOT bare ``Lψ``).  ``StreamingCollisionOperator.apply`` passes the
        composite's own σ and returns this directly; ``StreamingOperator.apply``
        passes σ = 0 and so gets ``Lψ`` — the Resolution-A identity
        ``L = (L+C) − C`` realised as the σ-free reading, not a subtraction
        (#257 S8b).  ``sigma`` is the diagonal coefficient, passed explicitly
        (#240 Step B).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement loss_action()"
        )

    def loss_action_transpose(
        self, sigma: "np.ndarray", phi: "FullField",
    ) -> "FullField":
        """The adjoint loss action ``(L+C)ᵀφ`` — 1-D implemented or a deferral raise.

        Returns the FULL adjoint loss ``(L+C)ᵀφ`` for the given ``sigma``;
        ``StreamingOperator.apply_transpose`` recovers bare ``Lᵀφ`` by calling
        this at σ = 0 (``C = σ⊙`` is self-adjoint, so the adjoint matvec is
        affine in σ too — #257 S8b, not a subtraction).
        ``sigma`` is the diagonal coefficient (#240 Step B).
        The transposed ψ½ leg kwargs are the B.2d explicit-leaf protocol (see
        :meth:`LossRepresentation.loss_action_transpose`).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement loss_action_transpose()"
        )

    @property
    def has_transpose_walk(self) -> bool:
        """Reverse-walk capability — opt-in ``False`` (the orientation factor).

        See :meth:`LossRepresentation.has_transpose_walk`.  Concrete leaves
        whose transpose walk EXISTS override — since #310 C4/C5 that is
        EVERY registered leaf (the scan and wavefront families both
        unconditionally) — so this base default is the honest floor a
        FUTURE representation inherits until its reverse lands: the eager
        ``.H`` refuses at construction rather than reaching a raising
        ``loss_action_transpose`` at apply time.
        """
        return False

    def __post_init__(self) -> None:
        compat = type(self).supports(self.mesh, self.spatial_closure)
        if not compat.ok:
            raise IncompatibleRepresentation(
                f"{type(self).__name__} cannot sweep this mesh "
                f"(ndim={self.mesh.ndim}, coord={self.mesh.coord.value!r}): "
                f"{compat.reason}."
            )


# ═══════════════════════════════════════════════════════════════════════
# _OctantWalk — THE in-plane octant traversal (S6.4, #222)
# ═══════════════════════════════════════════════════════════════════════

def _assert_face_moment_width(face: "np.ndarray", n: int, *, where: str) -> None:
    r"""Guard a moment-resolved face inflow's trailing transverse-moment width.

    A moment-resolved boundary inflow MUST carry exactly ``n = 2^{d-1}``
    transverse moments, or the relaxation silently MIS-BROADCASTS: a width-1
    trailing axis fans the single moment across all ``n`` slots — a wrong-physics
    seed numpy does NOT reject.  Single source of the face-moment-width contract,
    shared by the windowed :meth:`_LossRepresentation._inflow_to_moments` and the
    FFW-oracle :meth:`FullFieldWavefront._octant_face_cochain` seed — the twin
    face-cochain entry points must validate identically (#251).
    """
    if face.shape[-1] != n:
        raise ValueError(
            f"{where}: moment-resolved face inflow trailing width "
            f"{face.shape[-1]} != the per-face transverse-moment count "
            f"2^(d-1) = {n}.  A moment-resolved boundary inflow must carry "
            f"exactly 2^(d-1) transverse moments."
        )


def _inflow_faces(signs_eff: tuple[int, ...]) -> tuple[str, ...]:
    """Per-axis domain faces an octant's streaming ENTERS through.

    An octant streaming in the ``+a`` direction enters at the ``a``-min face
    (``("xmin", "ymin")`` for the ``(+1, +1)`` octant); a ``−a`` octant at
    the ``a``-max face.  ``signs_eff`` carries the EFFECTIVE signs (grazing
    ``0`` already mapped to ``+1`` — the streaming coefficient is zero, so
    the WDD result is sign-independent).
    """
    return tuple(
        face_name(a, -1 if s >= 0 else +1)
        for a, s in enumerate(signs_eff)
    )


def _outflow_faces(signs_eff: tuple[int, ...]) -> tuple[str, ...]:
    """Per-axis domain faces an octant's streaming EXITS through — the
    face-by-face OPPOSITE of :func:`_inflow_faces`.

    Since **B3.4c** that sentence is the implementation rather than a comment
    beside a second transcription. The two functions had parallel bodies with
    ``min``/``max`` swapped, which is a twin whose whole content is one word:
    an octant entering at ``a``-min exits at ``a``-max, always, because those
    are the two faces normal to ``a``. A sign flip in either body was a live
    edit away from silently disagreeing with the other.
    """
    return tuple(face_opposite(face) for face in _inflow_faces(signs_eff))


def _reverse_octant_traversal(
    sweeps: "tuple[OctantSweep, ...]",
) -> "tuple[OctantSweep, ...]":
    r"""The reverse-mode octant traversal — the multi-D sibling of
    :func:`_reverse_traversal` (#310 C3).

    Reverse-mode retraces each octant's walk backwards.  For the wavefront
    family the whole reversal is DATA: mirroring a streaming octant's
    EFFECTIVE signs selects the MIRROR graph, whose levels are the forward
    levels reversed and whose ``face_in``/``face_out`` roles are the
    forward's swapped (``face_in(−o) == face_out(o)``) — reversed program
    order + transposed addressing + the domain-boundary in↔out swap, with
    :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full`
    itself UNCHANGED.  The mirror is applied AFTER the grazing map (a
    grazing ``0`` axis rides ``+1`` forward, so its reversal is ``−1`` —
    the reverse must un-walk the chain the forward actually addressed, and
    ``0`` must NOT survive into the label for the walk's own effective map
    to re-flip).  Pure-z degenerate labels (all-zero in-plane) are their
    own mirror — the collision-diagonal branch is self-transposed.

    The ordinate ``indices`` stay PHYSICAL: the label is the march's
    ADDRESSING octant (the discrete μ→−μ — exact for the DAG topology and
    face roles), and the physical octant is recovered per octant as
    ``−signs`` where the cell algebra needs it (the level op's frame
    signs).  Octant ORDER is untouched: the Cartesian matvec has no
    inter-octant edge (the reflective coupling is the sibling ``−B``
    operator), unlike the 1-D pole handoff that forces
    :func:`_reverse_traversal` to reverse the leg order.
    """
    return tuple(
        sweep if not any(sweep.label.signs) else replace(
            sweep,
            label=OctantLabel(tuple(
                -(+1 if s == 0 else s) for s in sweep.label.signs
            )),
        )
        for sweep in sweeps
    )


@dataclass(frozen=True)
class _ApplyOperands:
    r"""Problem data of the APPLY-frame matvecs :math:`(L+C)\,\psi` /
    :math:`(L+C)^{\mathsf T}\varphi`.

    What every apply-frame interior kernel consumes, bundled once per
    :meth:`_OctantWalk.loss_action` / :meth:`_OctantWalk.loss_action_transpose`
    call — ONE bundle for BOTH orientations (#310 C3).  ``probe`` is the
    matvec's DRIVING bulk field: the forward's apply target
    :math:`\bar\psi`, the reverse's residual cotangent :math:`\bar r`
    (same structural role, same σ-epoch data; the semantic role is
    resolved at the typed ``FullField`` boundary).  ``Q_zero`` is the zero
    volumetric source — the matvec evaluates the loss *action*, not a
    balance; kernels whose walk signature requires a source slot (the
    graph walks) pass it through (the transpose op ignores it — the VJP
    is source-free by contract).  ``str_axes`` is the per-axis streaming
    tuple :math:`2|\mu_a|/\Delta a` over ``range(ndim)`` —
    positional-by-axis like every kernel tuple, so axis ``a``'s
    coefficients pair with axis ``a``'s faces by construction.
    """

    probe: "np.ndarray"                  # (N, ng, *spatial[, 2^d]) — the driving field (fwd ψ̄ / rev r̄)
    sig_t: "np.ndarray"                  # (ng, *spatial)
    str_axes: tuple["np.ndarray", ...]   # d arrays, each (N, n_a)
    Q_zero: "np.ndarray"                 # (1, ng, *spatial)


@dataclass(frozen=True)
class _SolveOperands:
    r"""Problem data of the SOLVE direction :math:`(L+C)^{-1} q`.

    The sweep's mirror of :class:`_ApplyOperands`: the solve direction is
    driven by the GIVEN per-ordinate volumetric source ``Q`` (the unknown is
    :math:`\bar\psi`), where the apply direction is driven by the given probe
    :math:`\bar\psi` (no source).  Same positional-by-axis ``str_axes``
    convention.
    """

    Q: "np.ndarray"                      # (N, ng, *spatial) — per-ordinate source
    sig_t: "np.ndarray"                  # (ng, *spatial)
    str_axes: tuple["np.ndarray", ...]   # d arrays, each (N, n_a)


def _moment_broadcast_sigma(
    sig: "np.ndarray", moment_valued: "np.ndarray",
) -> "np.ndarray":
    r"""``σ_t`` reshaped to broadcast over a trailing spatial-moment axis.

    The pure-z degenerate ordinates have no in-plane streaming, so the cell is
    collision-only and the loss couples to ``σ_t`` alone — the SOLVE balance is
    ``ψ̄ = Q / σ_t`` and its matvec twin is ``(L+C)ψ̄ = σ_t · ψ̄``.  At a
    multi-moment closure (LD, #240 D5b-S3) ``Q`` / ``ψ̄`` carry a trailing
    ``2^d`` spatial-moment axis that ``σ_t`` ``(ng, *spatial)`` lacks; each
    moment is scaled by the SAME scalar ``1/σ_t`` (resp. ``σ_t``), so ``σ_t``
    gains a length-1 trailing axis to broadcast.  DD/Step (no moment axis) →
    ``sig`` unchanged, byte-identical.

    The SINGLE source of the pure-z moment-broadcast convention: both the
    sweep arm (:math:`Q/σ_t`) and the matvec arm (:math:`σ_t·ψ̄`) call this, so
    the L21 twin (sweep ≡ matvec are two applications of the same collision-only
    operator) CANNOT diverge on the moment-axis reshape.  The moment-axis
    discriminator is :func:`~orpheus.numerics.moment_layout.is_moment_valued_by_rank`
    (single-sourced with the ``_CellSolve`` cell-solve source-reframe gate).
    """
    return sig[..., None] if is_moment_valued_by_rank(moment_valued, sig) else sig


@dataclass(frozen=True)
class _SweepEmit(ABC):
    r"""Solve-direction OUTPUT mode — angular field XOR harmonic moments.

    The Phase 5c output DI (which buffers are given selects the mode —
    mirroring the windowed walk's historical output contract), made a
    closed TYPE family (C5, 2026-07-03 — the ``_CellSolve`` precedent):
    each mode is a subclass with REQUIRED buffers, so a mixed or
    half-wired output is unrepresentable by construction (the former
    Optional fields + exactly-one runtime guard + ``*_buffers()``
    narrowing accessors all retired with the split).

    * :class:`_SweepEmitAngular` — ``angular_flux`` ``(N, ng, *spatial)``
      written per octant + ``scalar_flux`` ``(ng, *spatial)`` accumulated
      :math:`\sum_n w_n \psi_n`.
    * :class:`_SweepEmitMoment` — ``moment_buf``
      ``(L+1, 2L+1, ng, *spatial)`` accumulated
      :math:`\phi_\ell^m \mathrel{+}= \sum_n w_n Y_\ell^m \psi_n` with the
      octant harmonics ``Y`` ``(N, L+1, 2L+1)``; the full angular field is
      never materialized (the ~3× peak-memory win; the scalar is subsumed,
      ``moment_buf[0, 0]``).

    The pure-z volumetric balance emits through the polymorphic
    :meth:`pure_z`; the interior kernels accumulate at their own
    granularity (per anti-hyperplane for the window, per row for the
    scan-march) dispatching on the emit's TYPE.
    """

    weights: "np.ndarray"                       # (N,)

    @abstractmethod
    def pure_z(self, oct_idx: "np.ndarray", psi_avg: "np.ndarray") -> None:
        """Emit the pure-z volumetric balance ``ψ = Q/Σ_t`` (no faces).

        The accumulations use ``buf[...] +=`` (item-level in-place add, the
        same ufunc as a bare ``+=``) — a bare ``self.buf +=`` would rebind
        the attribute and trip the frozen dataclass.
        """


@dataclass(frozen=True)
class _SweepEmitAngular(_SweepEmit):
    """Angular-field output mode — per-octant ψ + accumulated scalar."""

    angular_flux: "np.ndarray"                  # (N, ng, *spatial)
    scalar_flux: "np.ndarray"                   # (ng, *spatial)

    def pure_z(self, oct_idx: "np.ndarray", psi_avg: "np.ndarray") -> None:
        self.angular_flux[oct_idx] = psi_avg
        self.scalar_flux[...] += np.einsum(
            "ng...,n->g...", psi_avg, self.weights[oct_idx],
        )


@dataclass(frozen=True)
class _SweepEmitMoment(_SweepEmit):
    """Harmonic-moment output mode — accumulated φ_ℓ^m, no angular field."""

    moment_buf: "np.ndarray"                    # (L+1, 2L+1, ng, *spatial)
    Y: "np.ndarray"                             # (N, L+1, 2L+1)

    def pure_z(self, oct_idx: "np.ndarray", psi_avg: "np.ndarray") -> None:
        self.moment_buf[...] += np.einsum(
            "nlm,ng...,n->lmg...", self.Y[oct_idx], psi_avg,
            self.weights[oct_idx],
        )


@dataclass(frozen=True)
class _OctantWalk:
    r"""THE in-plane octant traversal of the Cartesian loss operator.

    The sweep (forward substitution :math:`(L+C)^{-1} q`) and the matvec
    (:math:`(L+C)\,\psi`) traverse the SAME octant decomposition: project the
    quadrature octant to its in-plane signs, branch the pure-z degenerate
    octants, derive the per-axis in/out domain faces, read the octant's
    inflow, run the interior traversal, shed the outflow.  The two
    directions fork ONLY at

    * the **cell kernel** — the per-octant interior traversal the calling
      representation supplies: the window's frontier walk, the scan-march's
      row-march, the oracle's full cochain — each in its solve
      (:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`)
      or apply
      (:meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`)
      direction; and
    * the **emit policy** — what the direction accumulates: the sweep's
      angular/moment output; the matvec's :math:`(L+C)\psi` bulk + the O.4b
      boundary defect.

    NEVER a boolean ``is_solve`` flag — the direction is carried by the
    kernel/emit OBJECTS (the anti-degradation tripwire in
    ``tests/sn/operators/test_one_octant_walk.py`` enforces this shape).

    Dimension-generic from birth: signs / faces / inflow / captures are
    per-axis tuples over ``mesh.ndim`` — at d = 2 byte-identical to the
    legacy x/y spelling (pinned by the ``window ≡ full`` oracles).

    S6.4 staging: sub-step (a) routes the window + scan-march MATVEC frames
    through this walk; (b) brings the sweep frames in (the one-walk spy
    test flips xfail → xpass); (d) folds the full-field oracle.
    """

    mesh: "SNMesh"
    spatial_closure: "DiscretizationSchemeBase"
    angular_closure: "AngularClosureBase"

    def _interior_walk(
        self,
        sweeps: "tuple[OctantSweep, ...]",
        *,
        inflow_of: "Callable[[str], np.ndarray]",
        shed: "Callable[[str, np.ndarray, np.ndarray], None]",
        pure_z: "Callable[[np.ndarray], None]",
        interior: "Callable[[np.ndarray, tuple[int, ...], tuple[np.ndarray, ...]], tuple[np.ndarray, ...]]",
    ) -> None:
        r"""THE shared octant frame (the one-walk seam, S6.4).

        For each octant sweep unit: project the label to the in-plane signs,
        dispatch pure-z degenerates to ``pure_z``, derive the effective signs
        and per-axis domain faces, read the octant's inflow via
        ``inflow_of(face)[oct_idx]``, run ``interior`` (returning the
        per-axis outflow captures), and ``shed`` each capture into its
        outflow face.  Both public directions route through here — the
        matvec since sub-step (a), the sweep from sub-step (b) — so
        "matvec ≡ sweep is one walk" is a code fact, not a test-maintained
        coincidence.
        """
        for sweep in sweeps:
            oct_idx = np.asarray(sweep.indices)
            # The schedule's ``_octant_sweep`` is the SOLE in-plane projection
            # site, so the label carries exactly ``mesh.ndim`` signs — no
            # re-truncation here (a second silent projection could mask a
            # mis-sized label; a wrong length now fails loud at the face zips).
            signs = sweep.label.signs
            if not any(signs):
                # Pure-z degenerate octant: no in-plane streaming — no
                # faces, no boundary interaction. The direction's policy
                # handles the volumetric balance.
                pure_z(oct_idx)
                continue
            # Grazing (sign 0) ordinates ride the +1 sweep direction: the
            # streaming coefficient is zero, the WDD result sign-independent
            # (matches the legacy sx_eff/sy_eff mapping).
            signs_eff = tuple(+1 if s == 0 else s for s in signs)
            inflow = tuple(
                inflow_of(face)[oct_idx] for face in _inflow_faces(signs_eff)
            )
            capture = interior(oct_idx, signs_eff, inflow)
            for face, capture_a in zip(_outflow_faces(signs_eff), capture):
                shed(face, oct_idx, capture_a)

    def sweep_group(
        self,
        group: "OctantSweepGroup",
        *,
        operands: _SolveOperands,
        emit: _SweepEmit,
        boundary_flux: "AngularBoundaryFlux",
        interior: "Callable[[_SolveOperands, _SweepEmit, np.ndarray, tuple[int, ...], tuple[np.ndarray, ...]], tuple[np.ndarray, ...]]",
    ) -> None:
        r"""The SOLVE-direction frame for ONE octant group (S6.4 sub-step (b)).

        One forward-substitution pass over the group's octants on the SAME
        :meth:`_interior_walk` frame the matvec uses — the L21 unification.
        The Jacobi / Gauss-Seidel splitting lives one level up (the schedule
        loop's inter-group reflect, :func:`~orpheus.sn.loss_representation._sweep_scheduled`);
        this frame is the bare per-group sweep, blind to the boundary
        coupling.  The calling representation supplies ONLY its interior
        kernel::

            interior(operands, emit, oct_idx, signs_eff, inflow) -> capture

        Boundary coupling via the LIVE ``boundary_flux``: each octant reads
        its inflow off the trace and sheds its outflow back into it as the
        walk advances.  Distinct octants own DISJOINT ordinate slices of a
        face, so an octant's outflow write never clobbers another octant's
        inflow — the Jacobi single-group call is bit-identical to the legacy
        per-octant loop, and the Gauss-Seidel schedule reflects the
        just-shed outflow between groups so a later group reads the fresh
        current-iterate inflow off the SAME trace (the
        :math:`(L+C-B_{\rm lower})^{-1}` forward substitution).

        The pure-z degenerate octants take the volumetric balance
        :math:`\psi = Q_n / \Sigma_t` straight into the emit policy — no
        faces, no boundary interaction.
        """
        def pure_z(oct_idx: "np.ndarray") -> None:
            # ψ = Q/Σ_t for the in-plane-degenerate ordinates.  The trailing 2^d
            # spatial-moment axis (#240 D5b-S3) is broadcast on Σ_t by the SHARED
            # helper (the SAME convention the matvec twin's σ_t·ψ̄ rides — L21).
            q = operands.Q[oct_idx]
            emit.pure_z(oct_idx, q / _moment_broadcast_sigma(operands.sig_t, q))

        def run_interior(
            oct_idx: "np.ndarray",
            signs_eff: tuple[int, ...],
            inflow: tuple["np.ndarray", ...],
        ) -> tuple["np.ndarray", ...]:
            return interior(operands, emit, oct_idx, signs_eff, inflow)

        def shed(face: str, oct_idx: "np.ndarray", capture_a: "np.ndarray") -> None:
            boundary_flux.face_view(face)[oct_idx] = capture_a

        self._interior_walk(
            group.sweeps,
            inflow_of=boundary_flux.face_view,
            shed=shed,
            pure_z=pure_z,
            interior=run_interior,
        )

    def loss_action(
        self,
        sigma: "np.ndarray",
        psi: "FullField",
        interior: "Callable[[_ApplyOperands, np.ndarray, tuple[int, ...], tuple[np.ndarray, ...]], tuple[np.ndarray, tuple[np.ndarray, ...]]]",
    ) -> "FullField":
        r"""The APPLY-direction frame :math:`(L+C)\,\psi` (S6.4 sub-step (a)).

        Owns everything the matvec frames previously duplicated in lockstep:
        the probe / accumulator setup, the pure-z branch
        (:math:`(L+C)\bar\psi = \Sigma_t\,\bar\psi` — no in-plane streaming,
        so :math:`L\bar\psi = 0` after the operator's :math:`-C`), the
        per-octant inflow read from the GIVEN trace, the outflow capture,
        the O.4b active-trace boundary residual, and the typed assembly.
        The calling representation supplies ONLY its interior kernel::

            interior(operands, oct_idx, signs_eff, inflow)
                -> (LpC_octant, capture)

        Boundary semantics (BARE — O.4b Phase E): each octant reads its
        inflow from the GIVEN trace ``psi.boundary`` (NO ``bc.apply`` — the
        reflective coupling is the sibling ``−B``); the domain-edge outflow
        is captured into ``streamed`` (OUTFLOW slots only).  The output
        boundary is the O.4b active-trace residual: OUTFLOW slots → defect
        ``streamed − given``; INFLOW slots → identity ``given``.

        Returns the FULL loss :math:`(L+C)\bar\psi` for the given ``sigma``
        (NOT bare :math:`L\bar\psi`);
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` recovers
        :math:`L\bar\psi` by calling this frame at :math:`\sigma = 0`, not by
        subtracting (Resolution A as the σ-free reading, #257 S8b).
        """
        from orpheus.transport.full_field import FullField
        from orpheus.transport.source_sinks import (
            AngularSourceSink, AngularBoundarySourceSink,
        )

        sn_mesh = self.mesh
        ndim = sn_mesh.ndim
        ng = sigma.shape[0]
        spatial = sigma.shape[1:]
        probe = psi.interior.values
        # The matvec is intrinsically moment-valued (#240 D5b-S3): a multi-moment
        # closure (LD) carries a trailing 2^d spatial-moment axis on the probe
        # / accumulators so the apply returns the full moment residual.  DD/Step
        # (per_axis == 1) → ``()`` tail, every buffer byte-identical.  The tail
        # is read OFF the probe (its space already carries the scheme's
        # spatial-moment axis — the iterate is the single source of truth for the width).
        per_axis = self.spatial_closure.spatial_basis_per_axis
        moment_tail = face_moment_tail(cell_moment_count(per_axis, ndim))
        operands = _ApplyOperands(
            probe=probe,
            sig_t=sigma,
            str_axes=tuple(sn_mesh.streaming(a) for a in range(ndim)),
            Q_zero=np.zeros((1, ng, *spatial, *moment_tail)),
        )

        # (L+C)·ψ̄ accumulator; ``L.apply`` reaches bare-streaming Lψ̄ by running
        # this SAME frame at σ = 0 (#257 S8b), never by subtracting Σ_t·ψ̄.
        LpC = np.zeros((sn_mesh.quad.N, ng, *spatial, *moment_tail))
        trace = sn_mesh.angular_trace
        boundary = psi.boundary
        streamed = {
            face: np.zeros_like(boundary.face_view(face))
            for face in trace.face_names
        }

        def pure_z(oct_idx: "np.ndarray") -> None:
            # (L+C)·ψ̄ = σ·ψ̄ for the in-plane-degenerate ordinates.  The trailing
            # 2^d spatial-moment axis (#240 D5b-S3) is broadcast on σ by the SAME
            # shared helper the sweep twin's Q/σ_t rides (L21 — the qa Concern A
            # blocker: a quadrature WITH pure-z ordinates + a moment-valued probe
            # broadcast-crashed here without the guard the sweep already had).
            probe_oct = probe[oct_idx]
            LpC[oct_idx] = _moment_broadcast_sigma(sigma, probe_oct) * probe_oct

        def run_interior(
            oct_idx: "np.ndarray",
            signs_eff: tuple[int, ...],
            inflow: tuple["np.ndarray", ...],
        ) -> tuple["np.ndarray", ...]:
            LpC_oct, capture = interior(operands, oct_idx, signs_eff, inflow)
            LpC[oct_idx] = LpC_oct
            return capture

        def shed(face: str, oct_idx: "np.ndarray", capture_a: "np.ndarray") -> None:
            streamed[face][oct_idx] = capture_a

        (jacobi_group,) = SweepSchedule.jacobi(sn_mesh.ndim, sn_mesh.quad.octants).groups
        self._interior_walk(
            jacobi_group.sweeps,
            inflow_of=boundary.face_view,
            shed=shed,
            pure_z=pure_z,
            interior=run_interior,
        )

        # Boundary-block residual (O.4b — the active trace).
        out_boundary = AngularBoundarySourceSink.zeros(sn_mesh.angular_trace)
        for face in trace.face_names:
            given = boundary.face_view(face)
            out_idx = trace.outflow_indices_for_face(face)
            in_idx = trace.inflow_indices_for_face(face)
            if out_idx.size:
                out_boundary.face_view(face)[out_idx] = (
                    streamed[face][out_idx] - given[out_idx]
                )
            if in_idx.size:
                out_boundary.face_view(face)[in_idx] = given[in_idx]

        return FullField(
            interior=AngularSourceSink(values=LpC, space=sn_mesh.angular_trial_space),
            boundary=out_boundary,
        )

    def loss_action_transpose(
        self,
        sigma: "np.ndarray",
        phi: "FullField",
        interior: "Callable[[_ApplyOperands, np.ndarray, tuple[int, ...], tuple[np.ndarray, ...]], tuple[np.ndarray, tuple[np.ndarray, ...]]]",
    ) -> "FullField":
        r"""The APPLY-TRANSPOSE frame :math:`(L+C)^{\mathsf T}\varphi` (#310 C3).

        The reverse-mode adjoint of :meth:`loss_action`, on the SAME
        :meth:`_interior_walk` octant frame — orientation is carried by DATA
        (the :func:`_reverse_octant_traversal` mirror labels) + the injected
        interior kernel (bottoming in :class:`_CellResidualTranspose`),
        never a flag.  Per octant the mirror label drives the ADDRESSING
        (mirror graph = reversed levels + swapped face roles) while the
        ordinate rows stay physical — the multi-D sibling of the 1-D
        reverse's ``_reverse_traversal(_dag_legs())`` through the one
        ``_loop_walk``.

        Boundary semantics — the transpose of the O.4b active-trace
        residual (the 1-D reverse's algebra, per face):

        * the forward's OUTFLOW defect rows ``b_out = streamed − ψ_out``
          pull back as ``streamed̄ = b̄_out`` — seeding the reverse
          cochain's out-edge slots (rows NOT classified outflow stay ZERO:
          the forward assembly discarded them, so every discarded path's
          pullback vanishes structurally) — and ``ψ_out† = −b̄_out``;
        * the forward's INFLOW identity rows ``b_in = ψ_in`` pull back as
          ``ψ_in† = b̄_in`` PLUS the walked chain's in-edge capture (the
          forward consumed ψ_in twice: the identity row and the cochain
          seed).

        The pure-z transpose is the diagonal itself (``σ_t·r̄`` —
        collision-only, self-transposed, same moment broadcast as the
        forward twin, L21).  Returns the FULL ``(L+C)ᵀφ`` for the given
        ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply_transpose`
        recovers bare ``Lᵀφ`` by calling this frame at σ = 0 (#257 S8b, the
        mirror of ``apply`` — not a subtraction).
        """
        from orpheus.transport.full_field import FullField
        from orpheus.transport.source_sinks import (
            AngularSourceSink, AngularBoundarySourceSink,
        )

        sn_mesh = self.mesh
        ndim = sn_mesh.ndim
        ng = sigma.shape[0]
        spatial = sigma.shape[1:]
        scheme = self.spatial_closure
        if not type(scheme).has_transpose_kernel:
            # The trait DERIVES from the transpose-kernel registrations
            # (#310 ruling 2).  The honest front door is
            # StreamingOperator.is_adjointable (eager ``.H`` raises
            # MissingAdjoint); this guard is the backstop for direct
            # Euclidean apply_transpose calls that bypass ``.H``.
            raise NotImplementedError(
                "_OctantWalk.loss_action_transpose: scheme "
                f"{type(scheme).__name__} registers no transpose kernel "
                "pair (residual_kernel_batch_transpose) — the adjoint "
                "matvec on this scheme is a typed deferral (#310)."
            )
        per_axis = scheme.spatial_basis_per_axis
        moment_tail = face_moment_tail(cell_moment_count(per_axis, ndim))
        res_bar = phi.interior.values             # (N, ng, *spatial[, 2^d])
        if res_bar.shape[2 + ndim:] != tuple(moment_tail):
            # Pattern-4 backstop (mirror of the 1-D reverse): a cotangent
            # whose spatial-moment tail does not match the scheme's would
            # BROADCAST silently through the batch VJP — refuse loudly.
            raise ValueError(
                "_OctantWalk.loss_action_transpose: cotangent interior "
                f"shape {res_bar.shape} does not carry the scheme's "
                f"spatial-moment tail {tuple(moment_tail)} "
                f"({type(scheme).__name__})."
            )

        # The apply frame's operand bundle serves BOTH orientations:
        # ``probe`` is the matvec's driving bulk field — the forward's ψ̄,
        # the reverse's residual cotangent r̄ (same role, same σ-epoch data).
        operands = _ApplyOperands(
            probe=res_bar,
            sig_t=sigma,
            str_axes=tuple(sn_mesh.streaming(a) for a in range(ndim)),
            Q_zero=np.zeros((1, ng, *spatial, *moment_tail)),
        )

        psi_cot = np.zeros((sn_mesh.quad.N, ng, *spatial, *moment_tail))
        trace = sn_mesh.angular_trace
        b_bar = phi.boundary

        # ── reverse the boundary writeback (see the docstring's algebra) ──
        streamed_bar: dict[str, "np.ndarray"] = {}
        trace_cot: dict[str, "np.ndarray"] = {}
        for face in trace.face_names:
            given_bar = b_bar.face_view(face)
            seeded = np.zeros_like(given_bar)
            cot = np.zeros_like(given_bar)
            out_idx = trace.outflow_indices_for_face(face)
            in_idx = trace.inflow_indices_for_face(face)
            if out_idx.size:
                seeded[out_idx] = given_bar[out_idx]
                cot[out_idx] = -given_bar[out_idx]
            if in_idx.size:
                cot[in_idx] = given_bar[in_idx]
            streamed_bar[face] = seeded
            trace_cot[face] = cot

        def pure_z(oct_idx: "np.ndarray") -> None:
            # (L+C)ᵀ = σ_t for the in-plane-degenerate ordinates — the
            # collision-only diagonal is self-transposed (same shared
            # broadcast helper as the forward twin, L21).
            rb_oct = res_bar[oct_idx]
            psi_cot[oct_idx] = _moment_broadcast_sigma(sigma, rb_oct) * rb_oct

        def run_interior(
            oct_idx: "np.ndarray",
            signs_addr: tuple[int, ...],
            out_bars: tuple["np.ndarray", ...],
        ) -> tuple["np.ndarray", ...]:
            psi_cot_oct, capture = interior(
                operands, oct_idx, signs_addr, out_bars,
            )
            psi_cot[oct_idx] = psi_cot_oct
            return capture

        def shed(face: str, oct_idx: "np.ndarray", capture_a: "np.ndarray") -> None:
            # ACCUMULATE onto the identity rows (mirror of the 1-D
            # ``fi_bar[leg.ordinates] += f_bar.T``); distinct octants own
            # disjoint ordinate slices of a face, so each row receives
            # exactly one walked deposit.
            trace_cot[face][oct_idx] += capture_a

        (jacobi_group,) = SweepSchedule.jacobi(sn_mesh.ndim, sn_mesh.quad.octants).groups
        self._interior_walk(
            _reverse_octant_traversal(jacobi_group.sweeps),
            inflow_of=lambda face: streamed_bar[face],
            shed=shed,
            pure_z=pure_z,
            interior=run_interior,
        )

        out_boundary = AngularBoundarySourceSink.zeros(sn_mesh.angular_trace)
        for face in trace.face_names:
            out_boundary.face_view(face)[...] = trace_cot[face]

        return FullField(
            interior=AngularSourceSink(
                values=psi_cot, space=sn_mesh.angular_trial_space,
            ),
            boundary=out_boundary,
        )


# ═══════════════════════════════════════════════════════════════════════
# CumprodScan — the 1-D chain prefix scan (any geometry)
# ═══════════════════════════════════════════════════════════════════════


class CumprodScan(_LossRepresentation):
    r"""1-D parallel-prefix scan — slab, sphere, cylinder via one body.

    Intrinsically 1-D: a prefix scan needs a total order (a chain).  The
    geometry difference is absorbed by the two-stratum cache, so slab +
    sphere + cylinder share THE SAME scan expression
    (:meth:`._OneDimScanWalk.sweep` → :func:`~orpheus.sn.sweep.scan.ordinate_scan`).
    The default production path for every 1-D mesh.
    """

    @classmethod
    def supports(
        cls, mesh: "SNMesh", spatial_closure: "DiscretizationSchemeBase",
    ) -> Compatibility:
        if not mesh.is_1d:
            return Compatibility(False, "requires a 1-D mesh")
        # #236 ST2: a curvilinear mesh needs a curvilinear-capable scheme —
        # ``is_affine_scannable`` (a geometry-blind 1-D trait) is NOT sufficient
        # (LD is affine-scannable in slab but has no curvilinear closure).
        geometry = _curvilinear_capability(mesh, spatial_closure)
        if not geometry.ok:
            return geometry
        return Compatibility(
            spatial_closure.is_affine_scannable,
            "requires an affine-scannable cell-update scheme on a 1-D mesh",
        )

    def sweep(
        self,
        Q: "np.ndarray",
        sig_t: "np.ndarray",
        boundary_flux: "AngularBoundaryFlux",
        *,
        moment_frame: "FrameBase | None" = None,
        schedule: "SweepSchedule | None" = None,
        reflect: "Callable[[AngularBoundaryFlux, tuple[str, ...]], None] | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray]":
        if moment_frame is not None:
            # Moment output is the 2-D windowed-SI peak-memory optimization;
            # 1-D / curvilinear meshes stay full-angular (the Morel–Montry
            # Carlson seed reads the per-ordinate iterate; lesson L21).
            raise ValueError(
                "CumprodScan.sweep: moment output (moment_frame given) "
                "is 2-D Cartesian only — 1-D/curvilinear meshes stay "
                "full-angular (the Morel–Montry Carlson seed reads the "
                "per-ordinate iterate; lesson L21)."
            )
        if schedule is not None:
            # The octant-group schedule is a multi-D wavefront concern; the
            # 1-D scan is not a wavefront (boundary G-S is a no-op there).
            raise ValueError(
                "CumprodScan.sweep: a sweep schedule is multi-D only — "
                "the 1-D scan is not a wavefront."
            )
        return _OneDimScanWalk(self.mesh, self.spatial_closure, self.angular_closure).sweep(Q, sig_t, boundary_flux)

    def sweep_transpose(
        self,
        bulk_cot: "np.ndarray",
        sigma: "np.ndarray",
        boundary_cot: "BoundaryField",
    ) -> "tuple[np.ndarray, AngularBoundarySourceSink]":
        r"""The transpose-solve ``(L+C)⁻ᵀ`` — the REVERSE-SCAN (#280 2.5b).

        The solve-scan frame's adjoint: the transpose sibling of :meth:`sweep`
        (as :meth:`loss_action_transpose` is of :meth:`loss_action`).  The
        reverse-scan LIVES in :meth:`._OneDimScanWalk.sweep_transpose` (the
        reverse-mode adjoint of :meth:`._OneDimScanWalk._run`), sharing
        ``_run``'s ``ordinate_scan`` substrate via
        :func:`~orpheus.sn.sweep.scan.ordinate_scan_transpose`.
        """
        return _OneDimScanWalk(self.mesh, self.spatial_closure, self.angular_closure).sweep_transpose(
            bulk_cot, sigma, boundary_cot,
        )

    def loss_action(
        self, sigma: "np.ndarray", psi: "FullField",
    ) -> "FullField":
        r"""1-D forward loss action ``(L+C)ψ`` — the geometry-blind spatial sum.

        S6.3 / #206 Phase C: returns the FULL within-group loss ``(L+C)ψ`` for
        the given ``sigma``; ``StreamingOperator.apply`` recovers bare ``Lψ`` by
        calling this at σ = 0 (#257 S8b), never by subtracting ``C = σ⊙``.
        The matvec walk LIVES in :meth:`._OneDimScanWalk.loss_action` (the
        apply-direction twin of the sweep — L21 "matvec ≡ sweep"); the angular
        Morel–Montry redistribution + Carlson pole seed ride through
        ``angular_closure`` there (NOT re-inlined).
        """
        return _OneDimScanWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action(sigma, psi)

    def loss_action_transpose(
        self, sigma: "np.ndarray", phi: "FullField",
    ) -> "FullField":
        r"""1-D adjoint loss action ``(L+C)ᵀφ`` — the reverse spatial sum.

        S6.3 / #206 Phase C: returns the FULL ``(L+C)ᵀφ`` for the given
        ``sigma``; ``StreamingOperator.apply_transpose`` recovers bare ``Lᵀφ``
        by calling this at σ = 0 (#257 S8b), never by subtracting ``C``.
        The transpose walk LIVES in
        :meth:`._OneDimScanWalk.loss_action_transpose`, which carries the
        curvilinear angular SECOND triangular factor (``closure.angular_adjoint``)
        — so the spatial reverse NEVER silently drops the angular adjoint
        (pinned by ``test_g_adjoint_reciprocity`` sphere/cyl, -O-firing).
        """
        return _OneDimScanWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action_transpose(sigma, phi)

    @property
    def has_transpose_walk(self) -> bool:
        """``True`` — the 1-D scan family walks in reverse (#280 2.5a).

        ``supports`` confines this leaf to 1-D, where the shared
        ``_OneDimScanWalk`` frame traverses ``_reverse_traversal`` of the
        same legs the forward marches.
        """
        return True


# ═══════════════════════════════════════════════════════════════════════
# _DAGWavefront — the Cartesian anti-hyperplane DAG family
# ═══════════════════════════════════════════════════════════════════════


class _DAGWavefront(_LossRepresentation):
    r"""Base for the two buffer policies over the per-octant DAG walk.

    ``FullFieldWavefront`` (full-field buffer; the oracle) and
    ``MovingFrontierWindow`` (rolling :math:`(d{-}1)`-frontier; the
    production optimization) both walk the **same** per-octant
    anti-hyperplane DAG (:attr:`sweep_graphs`) with the same
    diamond-difference cell kernel.  They differ only in *how much* of the
    interior face cochain they retain — a storage policy, pinned
    bit-identical by the ``window ≡ full`` oracle.

    S6.4(c) — the family OWNS the DAG: ``sweep_graphs`` is THIS base's
    accessor over the per-shape cache
    :meth:`SweepDependencyGraph.for_shape`, NOT a mesh attribute.  The mesh
    is pure geometry; DAG-free representations (:class:`CumprodScan`,
    :class:`ScanMarch`) never mention the substrate — the historical
    curvilinear ``mesh.sweep_graphs = None`` slot (an illegal state) is
    unrepresentable.

    The DAG walk is naturally d-general (the oracle admits any-d Cartesian,
    the window ``d ≥ 2``); each strategy's ``supports`` states its current
    selection scope.
    """

    @classmethod
    def supports(
        cls, mesh: "SNMesh", spatial_closure: "DiscretizationSchemeBase",
    ) -> Compatibility:
        return Compatibility(
            mesh.is_cartesian and mesh.ndim == 2,
            "requires Cartesian geometry, d = 2",
        )

    @property
    def sweep_graphs(self) -> "Mapping[OctantLabel, SweepDependencyGraph]":
        r"""The per-octant DAG family for this representation's mesh shape.

        Routes to the per-shape cache (the graphs depend only on cell
        topology + octant signs, so same-shape meshes share byte-identical
        graphs); treat the mapping as immutable.
        """
        return SweepDependencyGraph.for_shape(self.mesh.spatial_shape)

    @property
    def has_transpose_walk(self) -> bool:
        r"""Family trait — the wavefront reverse is scheme- and d-complete (#310 C5).

        Since C3/C4 BOTH storage policies own the mirror-octant reverse
        (:meth:`FullFieldWavefront.loss_action_transpose` the full-cochain
        oracle, :meth:`MovingFrontierWindow.loss_action_transpose` the
        windowed production), and since C5 the multi-moment (LD) face
        cochain reverses through the same frame — so the family's
        orientation factor is unconditionally True wherever the shared
        frame runs.  The SCHEME factor lives where it belongs, in
        ``type(scheme).has_transpose_kernel`` (the registration-coupled
        covering law), read by
        :attr:`~orpheus.sn.operators.streaming.StreamingOperator.is_adjointable`
        as the other conjunct — an unregistered-kernel scheme still
        refuses loudly at both faces.
        """
        return True


class MovingFrontierWindow(_DAGWavefront):
    r"""Wavefront sweep — rolling :math:`(d{-}1)`-frontier buffer.

    The anti-diagonal (level-scheduled) sweep over the per-octant DAG,
    carrying only the rolling frontier of interior face fluxes (a 2-diagonal
    at d=2) — the ~30 % peak-memory win over the full-field oracle.
    Generalized to ``frontier_dim = d-1`` in S4 — the windowed WALK is
    d=3-pinned at the graph layer (``walk_windowed ≡ walk_full`` bit-id,
    ``test_sweep_graph_window_equivalence``), while ``supports`` stays
    conservatively d=2 (select narrow) until a d≥3 compute path + mesh
    exist; widen it WITH a measured d=3 profile, not before.

    A SELECTABLE PEER since the S6.9 Fork-B2 flip (#222): the multi-D Cartesian
    production default is now :class:`ScanMarch` (measured faster at identical
    peak memory — evidence at
    ``docs/theory/methods/sn/loss_representation.rst §loss-rep-fork-b2``), and
    this representation is kept as a genuinely different schedule over the same
    lower-triangular operator (user decision: multiple proper methods ARE the
    point of selectability).  Its end-to-end coverage rides the forced-window
    gates in ``tests/sn/solve/test_scan_march_end_to_end.py`` + the explicit
    window≡full oracles.
    """

    def sweep(
        self,
        Q: "np.ndarray",
        sig_t: "np.ndarray",
        boundary_flux: "AngularBoundaryFlux",
        *,
        moment_frame: "FrameBase | None" = None,
        schedule: "SweepSchedule | None" = None,
        reflect: "Callable[[AngularBoundaryFlux, tuple[str, ...]], None] | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray | None]":
        if schedule is None:
            return _sweep_jacobi(
                Q, sig_t, self.mesh, boundary_flux,
                spatial_closure=self.spatial_closure,
                angular_closure=self.angular_closure,
                moment_frame=moment_frame,
                interior=self._sweep_interior,
            )
        # #226 step 2: the scheduled (Gauss-Seidel) walk on THIS
        # representation's own interior kernel — the reified
        # ``M = (L+C−B_lower)`` forward substitution.  Same uniform loop,
        # different schedule (the splitting is the schedule; S6.4(b)).
        return _sweep_scheduled(
            Q, sig_t, self.mesh, boundary_flux,
            spatial_closure=self.spatial_closure,
            angular_closure=self.angular_closure,
            schedule=schedule,
            reflect=reflect,
            moment_frame=moment_frame,
            interior=self._sweep_interior,
        )

    def _sweep_interior(
        self,
        operands: _SolveOperands,
        emit: _SweepEmit,
        oct_idx: "np.ndarray",
        signs_eff: tuple[int, ...],
        inflow: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", ...]:
        r"""Rolling-frontier interior kernel, SOLVE direction, one octant.

        Drives
        :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`
        with the ``_CellSolve`` level operation (the windowed walk of the
        solve cell kernel
        :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.cell_kernel_batch`)
        over this octant's DAG, emitting per anti-hyperplane into the
        :class:`_SweepEmit` mode buffers.  Returns the per-axis domain-edge
        outflow ``capture``.
        """
        graph = self.sweep_graphs[OctantLabel(signs_eff)]
        ng = operands.sig_t.shape[0]
        spatial = operands.sig_t.shape[1:]
        # Multi-moment closures (LD's bilinear UBLD) carry a trailing
        # 2^{d-1}-moment face axis on the cochain; the domain inflow + capture
        # widen to match (#240 D5b).  Single-moment (DD/Step) → identity.
        n_face_moments = self._n_face_moments
        inflow = self._inflow_to_moments(inflow)
        capture = tuple(np.empty_like(face) for face in inflow)
        # Angular mode allocates a per-octant angular buffer (scattered into
        # the global field below); moment mode accumulates directly into the
        # shared moment tensor per anti-hyperplane, so NO per-octant angular
        # field is materialized (the Phase 5c peak-memory win).  At a
        # multi-moment closure (LD) the angular buffer carries the trailing 2^d
        # spatial-moment axis (the φ̂ iterate, #240 D5b-S3); DD/Step → ``()``.
        # The emit's TYPE selects the level-op subclass (C4/C5 — the mode is
        # a type on both ends; the isinstance dispatch pins the walker/emit
        # mode match by construction).
        frame_signs = self._moment_frame_signs(signs_eff)
        if isinstance(emit, _SweepEmitAngular):
            angular_flux = emit.angular_flux
            angular_flux_oct = np.zeros(
                (oct_idx.size, ng, *spatial, *self._spatial_moment_tail)
            )
            level_op: _CellSolve = _CellSolveAngular(
                scheme=self.spatial_closure,
                weights_octant=emit.weights[oct_idx],
                angular_flux_octant=angular_flux_oct,
                scalar_flux_buf=emit.scalar_flux,
                moment_frame_signs=frame_signs,
            )
        elif isinstance(emit, _SweepEmitMoment):
            angular_flux, angular_flux_oct = None, None
            level_op = _CellSolveMoment(
                scheme=self.spatial_closure,
                weights_octant=emit.weights[oct_idx],
                moment_buf=emit.moment_buf,
                Y_octant=emit.Y[oct_idx],
                moment_frame_signs=frame_signs,
            )
        else:
            raise TypeError(f"unknown _SweepEmit mode: {type(emit).__name__}")
        graph.walk_windowed(
            level_op=level_op,
            inflow=inflow,
            Q_octant=operands.Q[oct_idx],
            sig_t=operands.sig_t,
            str_axes_octant=tuple(s[oct_idx] for s in operands.str_axes),
            capture=capture,
            n_face_moments=n_face_moments,
        )
        if angular_flux is not None:
            angular_flux[oct_idx] = angular_flux_oct
        # The domain-edge capture carries the trailing 2^{d-1}-transverse-moment
        # axis at a multi-moment closure; the moment-resolved boundary trace
        # (#251 — geometry.boundary_face_layout) STORES it whole (the outflow
        # moments land in the now-moment-shaped slot, no longer collapsed to the
        # average).  DD/Step (n_face_moments == 1) → no moment axis, byte-identical.
        return capture

    def loss_action(
        self, sigma: "np.ndarray", psi: "FullField",
    ) -> "FullField":
        r"""2-D Cartesian forward loss action ``(L+C)ψ`` via the rolling-frontier window.

        S6.4 sub-step (a): routes through the shared :class:`_OctantWalk`
        apply frame (the ONE octant traversal — octant projection, pure-z
        branch, boundary I/O, the O.4b boundary residual), supplying only the
        rolling-frontier interior kernel :meth:`_loss_action_interior`
        (:meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`
        × ``_CellResidual`` — the apply-direction walk of the SAME per-octant
        wavefront DAG and the SAME diamond-difference closure the 2-D sweep
        uses; matvec ≡ sweep, ONE discretization, L21).  Returns ``(L+C)ψ̄`` for
        the given ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` recovers
        the bare-streaming ``Lψ̄`` by calling this walk at σ = 0 (#257 S8b), not
        by subtracting the collision diagonal ``C``.
        """
        return _OctantWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action(
            sigma, psi, self._loss_action_interior,
        )

    def _loss_action_interior(
        self,
        operands: _ApplyOperands,
        oct_idx: "np.ndarray",
        signs_eff: tuple[int, ...],
        inflow: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", tuple["np.ndarray", ...]]:
        r"""Rolling-frontier interior kernel, APPLY direction, one octant.

        Drives
        :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`
        with the ``_CellResidual`` level operation (the windowed walk of the
        apply cell kernel
        :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`)
        over this octant's DAG.  Returns ``(LpC_octant, capture)`` — the
        octant's ``(L+C)ψ̄`` block and the per-axis domain-edge outflow.
        """
        graph = self.sweep_graphs[OctantLabel(signs_eff)]
        ng = operands.sig_t.shape[0]
        spatial = operands.sig_t.shape[1:]
        # The unified moment matvec (#240 D5b-S3): a multi-moment closure carries
        # the trailing 2^d spatial-moment axis on the residual + the 2^{d-1}-moment
        # face cochain, exactly as the SOLVE direction (_sweep_interior).  The
        # probe already carries the moment axis (operands.probe).  DD/Step
        # (per_axis == 1) → ``()`` tail + n_face_moments == 1, byte-identical.
        n_face_moments = self._n_face_moments
        LpC_oct = np.zeros((oct_idx.size, ng, *spatial, *self._spatial_moment_tail))
        inflow = self._inflow_to_moments(inflow)
        capture = tuple(np.empty_like(face) for face in inflow)
        graph.walk_windowed(
            level_op=_CellResidual(
                scheme=self.spatial_closure,
                psi_avg_probe_octant=operands.probe[oct_idx],
                residual_octant=LpC_oct,
                moment_frame_signs=self._moment_frame_signs(signs_eff),
            ),
            inflow=inflow,
            Q_octant=operands.Q_zero,
            sig_t=operands.sig_t,
            str_axes_octant=tuple(s[oct_idx] for s in operands.str_axes),
            capture=capture,
            n_face_moments=n_face_moments,
        )
        # Domain-edge capture carries the trailing 2^{d-1}-transverse-moment axis
        # at a multi-moment closure; the moment-resolved boundary trace (#251)
        # STORES it whole into the now-moment-shaped ``streamed`` slot (which
        # inherits the widened face_view shape) — the B-residual emit below then
        # carries the outflow moments.  DD/Step → no moment axis, byte-identical.
        return LpC_oct, capture

    def loss_action_transpose(
        self, sigma: "np.ndarray", phi: "FullField",
    ) -> "FullField":
        r"""Adjoint loss action ``(L+C)ᵀφ`` — the reversed rolling-frontier walk (PRODUCTION).

        The reverse-mode adjoint of :meth:`loss_action` (#310 C4), routed
        through the shared :class:`_OctantWalk` apply-transpose frame with
        the rolling-frontier interior kernel
        :meth:`_loss_action_transpose_interior` — the UNCHANGED
        :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`
        over each octant's MIRROR graph × the
        :class:`_CellResidualTranspose` level operation.  The mirror graph's
        own ``window_plan`` IS the reversed frontier (built for every graph
        at construction), so the reverse pays the same
        :math:`(d{-}1)`-frontier peak memory as the forward — and the
        storage-policy claim is pinned bit-identical to the full-cochain
        oracle (:meth:`FullFieldWavefront.loss_action_transpose`) by the
        reverse ``window ≡ full`` gate
        (``test_multi_d_reverse_walk.test_reverse_window_equals_full``).
        Returns ``(L+C)ᵀφ`` for the given ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply_transpose`
        recovers bare ``Lᵀφ`` by calling this walk at σ = 0 (#257 S8b), not by
        subtracting ``σ_t·φ``.
        """
        return _OctantWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action_transpose(
            sigma, phi, self._loss_action_transpose_interior,
        )

    def _loss_action_transpose_interior(
        self,
        operands: _ApplyOperands,
        oct_idx: "np.ndarray",
        signs_addr: tuple[int, ...],
        out_bars: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", tuple["np.ndarray", ...]]:
        r"""Rolling-frontier interior kernel, APPLY-TRANSPOSE direction, one octant.

        The windowed sibling of
        :meth:`FullFieldWavefront._loss_action_transpose_interior` (the same
        mirror-octant realization, frontier storage instead of the full
        cochain): ``signs_addr`` is the MIRROR label
        (:func:`_reverse_octant_traversal`), so
        :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_windowed`
        over the mirror graph seeds its frontier "inflow" = the physical
        OUT-faces with the outflow cotangents ``out_bars``, gathers at the
        physical out-faces and scatters at the physical in-faces in
        reversed level order, and sheds its "capture" = the physical
        IN-face cotangents.  Only the level op knows the physical
        orientation (``operands.probe`` = the residual cotangent ``r̄``;
        the frame signs are the PHYSICAL octant's, recovered as
        ``−signs_addr`` — the mirror is an involution).  Returns
        ``(psi_cot_octant, capture)``.
        """
        graph = self.sweep_graphs[OctantLabel(signs_addr)]
        ng = operands.sig_t.shape[0]
        spatial = operands.sig_t.shape[1:]
        n_face_moments = self._n_face_moments
        psi_cot_oct = np.zeros(
            (oct_idx.size, ng, *spatial, *self._spatial_moment_tail)
        )
        inflow = self._inflow_to_moments(out_bars)
        capture = tuple(np.empty_like(face) for face in inflow)
        physical_signs = tuple(-s for s in signs_addr)
        graph.walk_windowed(
            level_op=_CellResidualTranspose(
                scheme=self.spatial_closure,
                res_bar_octant=operands.probe[oct_idx],
                psi_bar_cot_octant=psi_cot_oct,
                moment_frame_signs=self._moment_frame_signs(physical_signs),
            ),
            inflow=inflow,
            Q_octant=operands.Q_zero,
            sig_t=operands.sig_t,
            str_axes_octant=tuple(s[oct_idx] for s in operands.str_axes),
            capture=capture,
            n_face_moments=n_face_moments,
        )
        return psi_cot_oct, capture


class FullFieldWavefront(_DAGWavefront):
    r"""Verification-oracle wavefront sweep — the dimension-generic SPINE.

    Walks the same per-octant DAG as :class:`MovingFrontierWindow` but
    retains the FULL interior face cochain (the fuller view).  Slower and
    more memory-hungry — its purpose is verification: ONE body for d=1 (slab)
    and d=2 (Cartesian), and the reference the d-specific production
    optimizations are cross-checked against — the 1-D :class:`CumprodScan`
    (principled-equivalence at nulp) and the 2-D :class:`MovingFrontierWindow`
    (``window ≡ full`` bit-identity).  Never the production default (the
    window wins at d=2, the scan at d=1); selected explicitly by oracle tests.

    Since S6.4(d) every direction routes through the shared
    :class:`_OctantWalk` frame (sweep via the kernel-parameterized schedule
    loop, matvec via the apply frame, adjoint matvec via the
    apply-transpose frame — #310 C3) — this class supplies only the
    full-cochain interior kernels (:meth:`_sweep_interior` /
    :meth:`_loss_action_interior` /
    :meth:`_loss_action_transpose_interior`, walking the d-generic
    :meth:`SweepDependencyGraph.walk_full` × the ``_CellSolve`` /
    ``_CellResidual`` / ``_CellResidualTranspose`` level operations).
    ``supports`` is any-d Cartesian (S3) — the spine is genuinely
    dimension-generic, unlike the d=2-only window.
    """

    @classmethod
    def supports(
        cls, mesh: "SNMesh", spatial_closure: "DiscretizationSchemeBase",
    ) -> Compatibility:
        # Override the _DAGWavefront family's d=2-only predicate: the spine is
        # the genuine d-generic oracle (it walks the per-octant DAG for any
        # Cartesian d via the d-generic ``graph.residual``).
        return Compatibility(mesh.is_cartesian, "requires Cartesian geometry")

    # ── The full-cochain boundary embedding (shared by both kernels) ──

    @staticmethod
    def _octant_face_cochain(
        spatial: tuple[int, ...],
        signs_eff: tuple[int, ...],
        inflow: tuple["np.ndarray", ...],
        n_face_moments: int = 1,
    ) -> tuple["np.ndarray", ...]:
        r"""Allocate one octant's FULL per-axis interior face cochain, with the
        domain in-edges seeded from the octant's inflow.

        Axis ``a``'s buffer carries ``n_a + 1`` face slots along its own axis
        (the fuller view — every interior face is retained, vs the window's
        rolling frontier).  Only the IN-edge slot is seeded: by the upwind
        invariant every other slot (interior + out-edge) is written before any
        read, so a zero initialization is byte-identical to the historical
        whole-trace ι_* seed.

        ``n_face_moments`` is the per-face transverse moment count
        :math:`(\text{per\_axis})^{d-1}` (DD/Step: 1; LD-2D: 2 — #240 D5b).  At
        ``> 1`` each face buffer carries a trailing moment axis; the IN-edge is
        seeded from the domain inflow.  The inflow is rank-DISCRIMINATED against
        the flat face rank ``d + 1`` (a scalar face ``(N_oct, ng, *transverse)``
        has rank ``2 + (d−1) = d + 1``), exactly as the windowed twin's
        :meth:`_inflow_to_moments`:

        * a SCALAR inflow (a vacuum face) seeds the AVERAGE moment (slot 0); the
          transverse slope moments stay zero (the scalar default is blind to the
          along-face variation — the Leg-B asymmetry);
        * a MOMENT-RESOLVED inflow (the #251 moment-resolved boundary trace,
          ``2^{d-1}``-valued) seeds ALL ``2^{d-1}`` moments — the projected
          transverse face-slope is threaded into the cochain unchanged.

        For the S2 foundation gates the domain inflow is VACUUM / zero, so the
        whole moment object is zero.  At ``n_face_moments == 1`` the trailing axis
        is ABSENT — DD's rank-r face buffers are byte-identical.
        """
        N_oct, ng = inflow[0].shape[0], inflow[0].shape[1]
        ndim = len(spatial)
        tail = face_moment_tail(n_face_moments)
        # A scalar face is (N_oct, ng, *transverse) — rank 2 + (d−1) = ndim + 1;
        # a moment-resolved face carries one MORE trailing axis (#251).
        flat_face_ndim = ndim + 1
        faces = []
        for a in range(ndim):
            face_shape = list(spatial)
            face_shape[a] += 1
            buf = np.zeros((N_oct, ng, *face_shape, *tail))
            in_edge: "list[slice | int]" = [slice(None)] * (2 + ndim)
            in_edge[2 + a] = 0 if signs_eff[a] >= 0 else spatial[a]
            if n_face_moments == 1:
                buf[tuple(in_edge)] = inflow[a]
            elif is_moment_valued_by_flat_rank(inflow[a], flat_face_ndim):
                # Moment-resolved inflow (#251): seed ALL 2^{d-1} moments — the
                # projected transverse face-slope threads into the cochain.  Guard
                # the width as the windowed twin does (else a width-1 trailing axis
                # silently broadcasts the single moment across all n slots).
                _assert_face_moment_width(
                    inflow[a], n_face_moments,
                    where="FullFieldWavefront._octant_face_cochain",
                )
                buf[tuple(in_edge)] = inflow[a]
            else:
                # Scalar inflow: seed the average moment (slot 0); the transverse
                # slope moments stay zero (vacuum → all zero; the scalar default
                # is blind to the along-face variation — the Leg-B asymmetry).
                buf[(*in_edge, AVERAGE_MOMENT)] = inflow[a]
            faces.append(buf)
        return tuple(faces)

    @staticmethod
    def _edge_outflow(
        psi_faces_oct: tuple["np.ndarray", ...],
        spatial: tuple[int, ...],
        signs_eff: tuple[int, ...],
    ) -> tuple["np.ndarray", ...]:
        """Extract the per-axis domain OUT-edge slots (the octant's shed
        outflow) from the walked cochain.

        At a multi-moment closure the captured slot carries the trailing
        ``2^{d-1}`` moment axis; the domain-edge consumer (the boundary trace)
        takes the average moment.  For S2 the captured outflow feeds only the
        per-axis ``capture`` arrays the walk discards on a vacuum domain edge.
        """
        ndim = len(spatial)
        capture = []
        for a in range(ndim):
            out_edge: "list[slice | int]" = [slice(None)] * (2 + ndim)
            out_edge[2 + a] = spatial[a] if signs_eff[a] >= 0 else 0
            capture.append(psi_faces_oct[a][tuple(out_edge)])
        return tuple(capture)

    # ── The two directions' interior kernels ──────────────────────────

    def sweep(
        self,
        Q: "np.ndarray",
        sig_t: "np.ndarray",
        boundary_flux: "AngularBoundaryFlux",
        *,
        moment_frame: "FrameBase | None" = None,
        schedule: "SweepSchedule | None" = None,
        reflect: "Callable[[AngularBoundaryFlux, tuple[str, ...]], None] | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray | None]":
        if moment_frame is not None:
            raise ValueError(
                "FullFieldWavefront.sweep: the full-field oracle does not "
                "implement moment output — use MovingFrontierWindow for the "
                "windowed-SI moment path."
            )
        # S6.4(d): the oracle sweep = the Jacobi schedule × the full-cochain
        # kernel on the SAME schedule loop + walk frame as production (the
        # former private ``_sweep_full_field`` frame dissolved).  A given
        # ``schedule`` (#226 step 2) composes for free — the inter-group
        # reflect is kernel-agnostic (S6.4(b)).
        if schedule is None:
            return _sweep_jacobi(
                Q, sig_t, self.mesh, boundary_flux,
                spatial_closure=self.spatial_closure,
                angular_closure=self.angular_closure,
                moment_frame=None,
                interior=self._sweep_interior,
            )
        return _sweep_scheduled(
            Q, sig_t, self.mesh, boundary_flux,
            spatial_closure=self.spatial_closure,
            angular_closure=self.angular_closure,
            schedule=schedule,
            reflect=reflect,
            moment_frame=None,
            interior=self._sweep_interior,
        )

    def _sweep_interior(
        self,
        operands: _SolveOperands,
        emit: _SweepEmit,
        oct_idx: "np.ndarray",
        signs_eff: tuple[int, ...],
        inflow: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", ...]:
        r"""Full-cochain interior kernel, SOLVE direction, one octant.

        Drives :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full`
        with the ``_CellSolve`` level operation over the octant's
        complete per-axis face cochain — the fuller view the window replaces
        with a rolling frontier (the ``window ≡ full`` bit-identity anchor).
        Angular emit only (the oracle has no moment mode — guarded in
        :meth:`sweep`).
        """
        sig_t = operands.sig_t
        ng = sig_t.shape[0]
        spatial = sig_t.shape[1:]
        n_face_moments = self._n_face_moments
        graph = self.sweep_graphs[OctantLabel(signs_eff)]
        psi_faces_oct = self._octant_face_cochain(
            spatial, signs_eff, inflow, n_face_moments,
        )
        # The angular octant buffer carries the trailing 2^d spatial-moment axis
        # at a multi-moment closure (the φ̂ iterate, #240 D5b-S3); DD/Step →
        # ``()`` tail, byte-identical.  The isinstance parse pins the
        # angular-only contract loudly (a moment-mode emit here is a caller
        # error — the oracle has no moment mode, guarded in :meth:`sweep`).
        if not isinstance(emit, _SweepEmitAngular):
            raise TypeError(
                "full-field oracle _sweep_interior emits angular only; got "
                f"{type(emit).__name__}."
            )
        angular_flux, scalar_flux = emit.angular_flux, emit.scalar_flux
        angular_oct = np.zeros((oct_idx.size, ng, *spatial, *self._spatial_moment_tail))
        graph.walk_full(
            level_op=_CellSolveAngular(
                scheme=self.spatial_closure,
                weights_octant=emit.weights[oct_idx],
                angular_flux_octant=angular_oct,
                scalar_flux_buf=scalar_flux,
                moment_frame_signs=self._moment_frame_signs(signs_eff),
            ),
            psi_faces_octant=psi_faces_oct,
            Q_octant=operands.Q[oct_idx],
            sig_t=sig_t,
            str_axes_octant=tuple(s[oct_idx] for s in operands.str_axes),
        )
        angular_flux[oct_idx] = angular_oct
        capture = self._edge_outflow(psi_faces_oct, spatial, signs_eff)
        # The domain-edge capture carries the trailing 2^{d-1}-transverse-moment
        # axis at a multi-moment closure; the moment-resolved boundary trace
        # (#251) STORES it whole (oracle twin of the windowed _sweep_interior).
        # DD/Step → no moment axis, byte-identical.
        return capture

    def loss_action(
        self, sigma: "np.ndarray", psi: "FullField",
    ) -> "FullField":
        r"""Forward loss action ORACLE ``(L+C)ψ`` — the full-field DAG walk (d-generic).

        S6.4(d): routes through the shared :class:`_OctantWalk` apply frame,
        supplying the full-cochain interior kernel
        :meth:`_loss_action_interior`
        (:meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full` ×
        ``_CellResidual`` — the full-field walk sharing the SAME cell kernel
        as the windowed walk,
        so the MATH cannot drift from
        :meth:`MovingFrontierWindow.loss_action` — only storage).  Returns
        ``(L+C)ψ̄`` for the given ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` recovers
        bare ``Lψ̄`` by calling this walk at σ = 0 (#257 S8b), not by subtracting
        ``σ·ψ̄``.  Sole purpose: verification (production is the
        window / the 1-D scan).
        """
        return _OctantWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action(
            sigma, psi, self._loss_action_interior,
        )

    def _loss_action_interior(
        self,
        operands: _ApplyOperands,
        oct_idx: "np.ndarray",
        signs_eff: tuple[int, ...],
        inflow: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", tuple["np.ndarray", ...]]:
        r"""Full-cochain interior kernel, APPLY direction, one octant.

        Drives :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full`
        with the ``_CellResidual`` level operation (the full-field walk of
        the apply cell kernel
        :meth:`~orpheus.transport.spatial.diamond.DiamondDifference.residual_kernel_batch`)
        over the octant's complete face cochain.  Returns
        ``(LpC_octant, capture)``.
        """
        sig_t = operands.sig_t
        ng = sig_t.shape[0]
        spatial = sig_t.shape[1:]
        graph = self.sweep_graphs[OctantLabel(signs_eff)]
        # The unified moment matvec oracle (#240 D5b-S3): the full cochain carries
        # the 2^{d-1}-moment faces + the residual the 2^d spatial-moment axis.
        n_face_moments = self._n_face_moments
        psi_faces_oct = self._octant_face_cochain(
            spatial, signs_eff, inflow, n_face_moments,
        )
        LpC_oct = np.zeros((oct_idx.size, ng, *spatial, *self._spatial_moment_tail))
        graph.walk_full(
            level_op=_CellResidual(
                scheme=self.spatial_closure,
                psi_avg_probe_octant=operands.probe[oct_idx],
                residual_octant=LpC_oct,
                moment_frame_signs=self._moment_frame_signs(signs_eff),
            ),
            psi_faces_octant=psi_faces_oct,
            Q_octant=operands.Q_zero,
            sig_t=sig_t,
            str_axes_octant=tuple(s[oct_idx] for s in operands.str_axes),
        )
        capture = self._edge_outflow(psi_faces_oct, spatial, signs_eff)
        # The domain-edge capture carries the trailing 2^{d-1}-transverse-moment
        # axis at a multi-moment closure; the moment-resolved boundary trace
        # (#251) STORES it whole (oracle twin of the windowed _loss_action_interior).
        # DD/Step → no moment axis, byte-identical.
        return LpC_oct, capture

    def loss_action_transpose(
        self, sigma: "np.ndarray", phi: "FullField",
    ) -> "FullField":
        r"""Adjoint loss action ORACLE ``(L+C)ᵀφ`` — the reversed full-field DAG walk (d-generic).

        The reverse-mode adjoint of :meth:`loss_action` (#310 C3), routed
        through the shared :class:`_OctantWalk` apply-transpose frame with
        the full-cochain interior kernel
        :meth:`_loss_action_transpose_interior` — the UNCHANGED
        :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full`
        over each octant's MIRROR graph × the
        :class:`_CellResidualTranspose` level operation, bottoming in the
        SAME scheme kernel VJP as the 1-D reverse arms
        (:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch_transpose`,
        #310 C2) — so the adjoint MATH cannot drift from the forward's, only
        the orientation data.

        Verification-oracle arm of the flipped family (#310 C4/C5): the
        family predicate (:meth:`_DAGWavefront.has_transpose_walk`) is
        True wherever the shared frame runs (DD and LD, any ``d``, since
        C5), and the PRODUCTION reverses (the window / the row-march)
        are pinned against THIS walk — bit-identical (window) and
        principled-equivalent (scan-march) — while this walk itself is
        pinned by the 2-D dense-``Mᵀ`` forward-probe + the
        assembled-``Mᵀ`` cross-check + the d=1 cross-realization against
        the 1-D scan reverse (``tests/sn/sweep/core/test_multi_d_reverse_walk.py``).
        Returns ``(L+C)ᵀφ`` for the given ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply_transpose`
        recovers bare ``Lᵀφ`` by calling this walk at σ = 0 (#257 S8b), not by
        subtracting ``σ_t·φ``.
        """
        return _OctantWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action_transpose(
            sigma, phi, self._loss_action_transpose_interior,
        )

    def _loss_action_transpose_interior(
        self,
        operands: _ApplyOperands,
        oct_idx: "np.ndarray",
        signs_addr: tuple[int, ...],
        out_bars: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", tuple["np.ndarray", ...]]:
        r"""Full-cochain interior kernel, APPLY-TRANSPOSE direction, one octant.

        ``signs_addr`` is the octant's ADDRESSING label — the MIRROR of the
        physical effective signs (:func:`_reverse_octant_traversal`) — so
        the forward helpers realize the transposed roles VERBATIM:
        :meth:`_octant_face_cochain` seeds its "in-edge" = the physical
        OUT-edge with the outflow cotangents ``out_bars``;
        :meth:`~orpheus.sn.loss_representation.sweep_graph.SweepDependencyGraph.walk_full`
        over the mirror graph gathers at the physical out-faces and
        scatters at the physical in-faces in reversed level order;
        :meth:`_edge_outflow` extracts its "out-edge" = the physical
        IN-edge — the domain-inflow cotangent capture.  Only the level op
        knows the physical orientation (``operands.probe`` = the residual
        cotangent ``r̄``; the frame signs are the PHYSICAL octant's,
        recovered as ``−signs_addr`` — the mirror is an involution).
        Returns ``(psi_cot_octant, capture)``.
        """
        sig_t = operands.sig_t
        ng = sig_t.shape[0]
        spatial = sig_t.shape[1:]
        graph = self.sweep_graphs[OctantLabel(signs_addr)]
        n_face_moments = self._n_face_moments
        faces_bar = self._octant_face_cochain(
            spatial, signs_addr, out_bars, n_face_moments,
        )
        psi_cot_oct = np.zeros(
            (oct_idx.size, ng, *spatial, *self._spatial_moment_tail)
        )
        physical_signs = tuple(-s for s in signs_addr)
        graph.walk_full(
            level_op=_CellResidualTranspose(
                scheme=self.spatial_closure,
                res_bar_octant=operands.probe[oct_idx],
                psi_bar_cot_octant=psi_cot_oct,
                moment_frame_signs=self._moment_frame_signs(physical_signs),
            ),
            psi_faces_octant=faces_bar,
            Q_octant=operands.Q_zero,
            sig_t=sig_t,
            str_axes_octant=tuple(s[oct_idx] for s in operands.str_axes),
        )
        capture = self._edge_outflow(faces_bar, spatial, signs_addr)
        return psi_cot_oct, capture


# ═══════════════════════════════════════════════════════════════════════
# ScanMarch — the row-march + x-scan schedule (1-D scan ∘ transverse march)
# ═══════════════════════════════════════════════════════════════════════


class ScanMarch(_LossRepresentation):
    r"""Scan-march sweep — ``scan(x)`` marched over the transverse axes (#222).

    Reframes the d-D diamond-difference sweep as forward substitution along the
    sweep axis — the first-order linear scan
    :func:`~orpheus.sn.sweep.scan.ordinate_scan` — marched over the transverse
    axes: ``scan(x)`` at d=1, ``scan(x) ∘ march(y)`` at d=2.  ONE primitive that
    **unifies** the 1-D :class:`CumprodScan` (its degenerate ``s_y = 0`` case)
    and the 2-D row-march: the within-row x-face recurrence is the SAME Blelloch
    scan, the transverse coupling rides the affine source (the row-march
    interior kernel :meth:`_sweep_interior`, S6.4(b) — the former private
    ``_sweep_2d_scanmarch`` frame dissolved into the shared
    :class:`_OctantWalk` + the Jacobi schedule).

    A different *schedule* from the :class:`_DAGWavefront` family (row-march vs
    anti-diagonal) over the SAME lower-triangular solve — principled-equivalent
    at nulp, pinned against the :class:`FullFieldWavefront` oracle (issue #222).
    Its production value: it reuses the conditioning-robust ``ordinate_scan``
    per line (the ERR-054 pole reset + the ERR-057 denormal underflow handled
    for free) and is the natural home for the flux-independent ``a_attenuation``
    cache the wavefront lacks (#206).

    Selection — ``is_1d OR (is_cartesian AND ndim == 2)``: 1-D any geometry
    (the chain scan; the curvilinear Morel–Montry angular thread folds into
    the source) AND 2-D Cartesian (the row-march).  The d≥3 row-march
    (``scan(x)∘march(y, z)`` — a raster march over the transverse
    hyperplane) is the algorithm's natural generalization but the interior
    kernels unpack d=2 today, so ``supports`` tells the truth (C3.6:
    construct general, SELECT NARROW) and a d≥3 Cartesian mesh
    (constructible since C5.5/#225 via the mesh-less
    ``SNMesh.from_axes``) falls through ``default_for`` to the genuinely
    d-generic :class:`FullFieldWavefront` spine instead of misrouting
    here (pinned LIVE by ``TestD3SupportsMatrix``; the d≥3 kernel
    generalization is #227).  Widen this predicate WITH the kernel generalization,
    never before it.

    **The 2-D Cartesian PRODUCTION DEFAULT since the S6.9 Fork-B2 flip (#222)**
    — measured faster than the window at identical peak memory (the full
    sweep/matvec/end-to-end basis is at
    ``docs/theory/methods/sn/loss_representation.rst §loss-rep-fork-b2``).
    1-D still selects ``CumprodScan`` (registered first; same scan primitive,
    no march shell).  Mode-9 FP-invariance vs the window is pinned end-to-end
    by ``tests/sn/solve/test_scan_march_end_to_end.py``.
    """

    @classmethod
    def supports(
        cls, mesh: "SNMesh", spatial_closure: "DiscretizationSchemeBase",
    ) -> Compatibility:
        # The 1-D arm reads ``is_affine_scannable`` (single-axis prefix
        # scannability — LD's 1-D scan IS valid here).  The d≥2 arm reads the
        # DISTINCT ``transverse_coupling_is_facewise`` (cross-axis
        # separability): the row-march ``scan(x) ∘ march(y)`` is exact ONLY
        # when the transverse coupling folds into the scan source as a
        # 0th-order face value (DD/Step), NOT when it is a 1st-order slope
        # moment (LD's bilinear multi-D closure).  Conflating the two — a 1-D
        # trait licensing a multi-D schedule — silently misroutes a 2-D LD
        # mesh into the inline-DD row-march (#240 D5-0); the split keeps the
        # selection honest.
        if mesh.is_1d:
            # #236 ST2: the same curvilinear-capability gate as CumprodScan —
            # a slab-only scheme on a curvilinear mesh is rejected here, not
            # raised mid-sweep.
            geometry = _curvilinear_capability(mesh, spatial_closure)
            if not geometry.ok:
                return geometry
            return Compatibility(
                spatial_closure.is_affine_scannable,
                "requires an affine-scannable cell-update scheme on a "
                "1-D mesh (any geometry)",
            )
        return Compatibility(
            mesh.is_cartesian
            and mesh.ndim == 2
            and spatial_closure.transverse_coupling_is_facewise,
            "2-D scan-march requires a scheme whose transverse coupling is "
            "facewise (separable into independent per-axis 1-D scans) — the "
            "slopeless cell-average closures (Diamond Difference, Step); "
            "Linear-Discontinuous's bilinear slope coupling needs the "
            "wavefront (the d≥3 row-march kernels are deferred — the "
            "full-field spine serves d≥3)",
        )

    def sweep(
        self,
        Q: "np.ndarray",
        sig_t: "np.ndarray",
        boundary_flux: "AngularBoundaryFlux",
        *,
        moment_frame: "FrameBase | None" = None,
        schedule: "SweepSchedule | None" = None,
        reflect: "Callable[[AngularBoundaryFlux, tuple[str, ...]], None] | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray | None]":
        if self.mesh.is_1d:
            # d=1 ⇒ ``scan(x)`` with no transverse march: the unified 1-D body
            # (slab + curvilinear via the two-stratum cache; the Morel–Montry
            # Carlson angular thread folds into the scan's affine source).  This
            # is the ``s_y = 0`` degeneration of the 2-D scan-march.
            if moment_frame is not None:
                raise ValueError(
                    "ScanMarch.sweep: moment output (moment_frame given) "
                    "is 2-D Cartesian only — 1-D/curvilinear meshes stay "
                    "full-angular (the Morel–Montry Carlson seed reads the "
                    "per-ordinate iterate; lesson L21)."
                )
            if schedule is not None:
                raise ValueError(
                    "ScanMarch.sweep: a sweep schedule is multi-D only — "
                    "the 1-D scan is not a wavefront."
                )
            return _OneDimScanWalk(self.mesh, self.spatial_closure, self.angular_closure).sweep(Q, sig_t, boundary_flux)
        # multi-D ⇒ the row-march sweep = the schedule × the scan-march
        # interior kernel on the SAME schedule loop the window uses (S6.4(b):
        # the former private ``_sweep_2d_scanmarch`` frame dissolved into the
        # shared walk — and the Gauss-Seidel schedule composes for free, the
        # inter-group reflect being kernel-agnostic; #226 step 2 threads it
        # through this door so the reified ``M`` runs the operator's ONE
        # representation instance).
        if schedule is None:
            return _sweep_jacobi(
                Q, sig_t, self.mesh, boundary_flux,
                spatial_closure=self.spatial_closure,
                angular_closure=self.angular_closure,
                moment_frame=moment_frame,
                interior=self._sweep_interior,
            )
        return _sweep_scheduled(
            Q, sig_t, self.mesh, boundary_flux,
            spatial_closure=self.spatial_closure,
            angular_closure=self.angular_closure,
            schedule=schedule,
            reflect=reflect,
            moment_frame=moment_frame,
            interior=self._sweep_interior,
        )

    def _sweep_interior(
        self,
        operands: _SolveOperands,
        emit: _SweepEmit,
        oct_idx: "np.ndarray",
        signs_eff: tuple[int, ...],
        inflow: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", ...]:
        r"""Row-march interior kernel, SOLVE direction, one octant.

        Marches the y-rows in the octant's y-sweep order: within each row the
        x-face recurrence is the first-order linear scan
        (:func:`~orpheus.sn.sweep.scan._scanmarch_row`) whose coefficients
        come from the scheme's
        :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.cartesian_scan_coefficients`
        — the row body carries NO inline diamond ``2`` or blend ``w`` (the
        scheme owns them; #240 D5a / #239 the coefficient-model lift).  The
        transverse-y coupling ``c_y · ψ_{y,in}`` rides the affine source via
        :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.source_emission`.
        Emits per row into the :class:`_SweepEmit` mode buffers.  Returns the
        per-axis domain-edge outflow ``(capture_x, out_y)``.

        The flux-independent scan coefficients are computed PER LINE (the
        :math:`(d{-}1)`-slab working set; mesh-memoising them — the #206
        single-source cache — is the measured follow-on).
        """
        sig_t = operands.sig_t                      # (ng, nx, ny)
        ng, nx, ny = sig_t.shape
        sx_eff, sy_eff = signs_eff
        inflow_x, inflow_y = inflow                 # (N_oct, ng, ny) / (N_oct, ng, nx)
        s_x, s_y = (s[oct_idx] for s in operands.str_axes)
        Q_oct = operands.Q[oct_idx]                 # (N_oct, ng, nx, ny)
        w_oct = emit.weights[oct_idx]               # (N_oct,)
        N_oct = oct_idx.size

        x_reverse = sx_eff < 0
        capture_x = np.empty((N_oct, ng, ny))       # domain x-outflow, per y-row

        # Per-mode row emission, bound ONCE before the march (C5 — the emit
        # mode is a TYPE; the row loop below is mode-blind).  Angular mode
        # stages a per-octant buffer and scatters it into the global field
        # in ``finish_octant`` (one block store instead of ny per-row
        # advanced-index scatters); moment mode accumulates directly into
        # the shared tensor per row (no angular field materialized).
        if isinstance(emit, _SweepEmitAngular):
            angular_oct = np.zeros((N_oct, ng, nx, ny))
            angular_flux, scalar_flux = emit.angular_flux, emit.scalar_flux

            def emit_row(j: int, psi_avg_row: "np.ndarray") -> None:
                angular_oct[:, :, :, j] = psi_avg_row
                scalar_flux[:, :, j] += np.einsum(
                    "ngi,n->gi", psi_avg_row, w_oct,
                )

            def finish_octant() -> None:
                angular_flux[oct_idx] = angular_oct

        elif isinstance(emit, _SweepEmitMoment):
            moment_buf, Y_oct = emit.moment_buf, emit.Y[oct_idx]

            def emit_row(j: int, psi_avg_row: "np.ndarray") -> None:
                moment_buf[:, :, :, :, j] += np.einsum(
                    "nlm,ngi,n->lmgi", Y_oct, psi_avg_row, w_oct,
                )

            def finish_octant() -> None:
                return None

        else:
            raise TypeError(f"unknown _SweepEmit mode: {type(emit).__name__}")

        # March the y-rows in the octant's y-sweep order, threading ψ_y.
        scheme = self.spatial_closure
        psi_y_in = inflow_y                          # (N_oct, ng, nx) — row-0 inflow
        out_y = psi_y_in                             # last-row out_y (ny ≥ 1 → set below)
        y_rows = range(ny) if sy_eff >= 0 else range(ny - 1, -1, -1)
        for j in y_rows:
            # #239 coefficient-model lift: the row-march asks the SCHEME for its
            # affine x-scan coefficients (the diamond ``2 = 1/w_DD`` + the
            # transverse coupling ``c_y = 2 g_y`` live in the scheme, NOT inline
            # here), so the scan-march is generic over every facewise closure
            # (DD today; Step rides it free once Step exists).  The transverse-y
            # ``g_y`` is the KNOWN-upstream 0th-order face value the row absorbs
            # into the affine source (``transverse_coupling_is_facewise``).
            a_scan, inverse_denom, w_row, (c_y,) = scheme.cartesian_scan_coefficients(
                s_scan=s_x[:, None, :],              # (N_oct, 1, nx) RAW g_x
                s_transverse=(s_y[:, j][:, None, None],),  # (N_oct, 1, 1) RAW g_y
                reaction_xs=sig_t[None, :, :, j],    # (1, ng, nx) Σ_t on this row
            )
            # Affine source b = source_emission(Q + c_y·ψ_y, inverse_denom, w):
            # the transverse-y direct term folds into the effective source.
            beta = scheme.source_emission(
                Q_oct[:, :, :, j] + c_y * psi_y_in, inverse_denom, w_row,
            )
            psi_avg_row, out_y, x_outflow = _scanmarch_row(
                a_scan, beta, inflow_x[:, :, j], psi_y_in, w_row, x_reverse,
            )
            psi_y_in = out_y
            capture_x[:, :, j] = x_outflow
            emit_row(j, psi_avg_row)

        finish_octant()
        # x-outflow is each row's last x-scan value (captured above); the
        # y-outflow is the LAST-marched row's out_y.
        return (capture_x, out_y)

    def loss_action(
        self, sigma: "np.ndarray", psi: "FullField",
    ) -> "FullField":
        r"""Forward loss action ``(L+C)ψ`` — the row-march apply (L21: sweep & matvec, ONE operator).

        S6.3: the matvec ``(L+C)ψ`` walk LIVES here.  1-D → the geometry-blind
        :meth:`_OneDimScanWalk._apply_walk` (the ``s_y = 0`` degeneration of the
        2-D scan-march).  2-D Cartesian → the row-march reconstruction of
        the interior faces from the KNOWN probe via
        :func:`~orpheus.sn.sweep.scan._x_scan_faces` with the apply coefficients
        ``α = −1``, ``β = 2 ψ̄`` (a pure-reflection scan: since ψ̄ is known the WDD
        closure ``out_x = 2ψ̄ − in_x`` IS a first-order recurrence).  The per-cell
        residual is ``(σ + s_x + s_y)·ψ̄ − s_x·in_x − s_y·in_y`` (``= (L+C)ψ̄`` at
        zero source) for the given ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` reaches
        ``Lψ̄`` by calling this walk at σ = 0 (#257 S8b), not by subtracting.

        Principled-equivalent (NOT bit-identical) to
        :meth:`MovingFrontierWindow.loss_action`: the row-march and the
        anti-diagonal reconstruct the SAME faces in a different order, so the
        residual agrees to FP-association.  The :class:`FullFieldWavefront`
        oracle pins it (G2.c).  S6.4 sub-step (a): the octant frame + the
        O.4b boundary-residual block are the shared :class:`_OctantWalk`
        apply frame (the former Fork-B1 lockstep duplication of
        :meth:`MovingFrontierWindow.loss_action` is GONE) — this class
        supplies only the row-march interior kernel
        :meth:`_loss_action_interior`.
        """
        if self.mesh.is_1d:
            # d=1 ⇒ scan(x) with no transverse march: the 1-D apply-direction
            # walk (#206 Phase C — the s_y = 0 degeneration of the 2-D
            # scan-march; the matvec walk lives in _OneDimScanWalk.loss_action).
            return _OneDimScanWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action(sigma, psi)
        return _OctantWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action(
            sigma, psi, self._loss_action_interior,
        )

    def _loss_action_interior(
        self,
        operands: _ApplyOperands,
        oct_idx: "np.ndarray",
        signs_eff: tuple[int, ...],
        inflow: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", tuple["np.ndarray", ...]]:
        r"""Row-march interior kernel, APPLY direction, one octant.

        Marches the y-rows in the octant's y-sweep order, reconstructing the
        x-faces from the KNOWN probe via :func:`~orpheus.sn.sweep.scan._x_scan_faces`
        with the apply coefficients ``α = −1``, ``β = 2ψ̄`` (a pure-reflection
        scan) and threading the transverse-y face ``out_y = 2ψ̄ − ψy_in`` row
        to row.  Returns ``(LpC_octant, capture)`` with the per-axis
        domain-edge outflow ``(cap_x, out_y)`` — the x-outflow captured per
        row, the y-outflow the LAST-marched row's ``out_y``.
        """
        sig_t = operands.sig_t                      # (ng, nx, ny)
        ng, nx, ny = sig_t.shape
        sx_eff, sy_eff = signs_eff
        inflow_x, inflow_y = inflow                 # (N_oct, ng, ny) / (N_oct, ng, nx)
        s_x, s_y = (s[oct_idx] for s in operands.str_axes)  # (N_oct, nx) / (N_oct, ny)
        probe_oct = operands.probe[oct_idx]         # (N_oct, ng, nx, ny)
        N_oct = oct_idx.size

        x_reverse = sx_eff < 0
        scheme = self.spatial_closure
        LpC_oct = np.empty((N_oct, ng, nx, ny))
        cap_x = np.empty((N_oct, ng, ny))            # domain x-outflow, per y-row
        s_x_row = s_x[:, None, :]                    # (N_oct, 1, nx) RAW g_x — row-invariant

        # March the y-rows in the octant's y-sweep order.  #239 coefficient-model
        # lift: the scheme owns the diamond ``2`` and the blend ``w`` — the
        # x-faces reconstruct off the probe via the scheme's reflection scan
        # (``α = −1``, ``β = 2ψ̄`` for DD), and the per-cell residual + the
        # transverse-y outflow ride the scheme's ``residual_kernel_batch`` (the
        # ÷V matvec kernel every facewise closure shares).  No inline DD.
        psi_y_in = inflow_y                          # (N_oct, ng, nx) — row-0 inflow
        out_y = psi_y_in                             # last-row out_y (ny ≥ 1 → set below)
        y_rows = range(ny) if sy_eff >= 0 else range(ny - 1, -1, -1)
        for j in y_rows:
            psi_bar_row = probe_oct[:, :, :, j]      # (N_oct, ng, nx)
            # Reconstruct the x-faces from the KNOWN probe (scheme reflection
            # scan ``ψ_out = α·ψ_in + β``; for DD ``out_x = 2ψ̄ − in_x``).
            alpha_reflect, beta_reflect = scheme.reflect_scan_coefficients(
                psi_bar_row,
            )
            in_x_row, _out_x_row, x_outflow = _x_scan_faces(
                alpha_reflect, beta_reflect, inflow_x[:, :, j], x_reverse,
            )
            # Per-cell residual (L+C)ψ̄ + the transverse-y outflow via the
            # scheme's ÷V matvec kernel: residual = (Σ_t + Σ 2g_a)ψ̄ − Σ 2g_a·in_a
            # at zero source; psi_out is the (out_x, out_y) face pair.
            residual_row, (_out_x_face, out_y) = scheme.residual_kernel_batch(
                psi_bar=psi_bar_row,
                psi_in=(in_x_row, psi_y_in),
                s_axes=(s_x_row, s_y[:, j][:, None, None]),
                reaction_xs=sig_t[:, :, j],
                Q_cells=np.zeros((1, ng, nx)),
            )
            LpC_oct[:, :, :, j] = residual_row
            psi_y_in = out_y
            cap_x[:, :, j] = x_outflow
        return LpC_oct, (cap_x, out_y)

    def loss_action_transpose(
        self, sigma: "np.ndarray", phi: "FullField",
    ) -> "FullField":
        r"""Adjoint loss action ``(L+C)ᵀφ`` — the reversed row-march (#310 C4).

        The reverse-mode adjoint of :meth:`loss_action`, direction by
        direction: 1-D → the geometry-blind
        :meth:`_OneDimScanWalk.loss_action_transpose` (the reversed leg
        walk, #310 C2); 2-D Cartesian → the shared :class:`_OctantWalk`
        apply-transpose frame with the row-march reverse interior
        :meth:`_loss_action_transpose_interior` — reversed rows +
        the x-face-chain transpose
        (:func:`~orpheus.sn.sweep.scan._x_scan_faces_transpose`, ONE
        :func:`~orpheus.sn.sweep.scan.ordinate_scan_transpose` per row) +
        the transverse cotangent chained BACKWARDS, bottoming in the same
        scheme kernel VJP
        (:meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch_transpose`)
        as every other reverse arm.  Principled-equivalent (NOT
        bit-identical) to
        :meth:`FullFieldWavefront.loss_action_transpose` — the reverse
        sibling of the forward's row-march-vs-oracle pin.  Returns
        ``(L+C)ᵀφ`` for the given ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply_transpose`
        recovers bare ``Lᵀφ`` by calling this walk at σ = 0 (#257 S8b), not by
        subtracting ``σ_t·φ``.
        """
        if self.mesh.is_1d:
            # #206 Phase C: the 1-D transpose walk lives in _OneDimScanWalk.
            return _OneDimScanWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action_transpose(sigma, phi)
        return _OctantWalk(self.mesh, self.spatial_closure, self.angular_closure).loss_action_transpose(
            sigma, phi, self._loss_action_transpose_interior,
        )

    def _loss_action_transpose_interior(
        self,
        operands: _ApplyOperands,
        oct_idx: "np.ndarray",
        signs_addr: tuple[int, ...],
        out_bars: tuple["np.ndarray", ...],
    ) -> tuple["np.ndarray", tuple["np.ndarray", ...]]:
        r"""Row-march interior kernel, APPLY-TRANSPOSE direction, one octant.

        The honest reverse-mode of :meth:`_loss_action_interior`'s own
        program, with the MIRROR label driving the schedule exactly as in
        the wavefront reverse (``signs_addr`` = the mirror of the physical
        effective signs, so the forward's sign-reading spellings —
        ``x_reverse``, the ``y_rows`` order — produce the REVERSED physical
        march for free, and ``out_bars`` arrives at the mirror in-faces =
        the physical OUT-faces).  Per row, in mirror-y order, threading the
        transverse cotangent backwards:

        1. the batched kernel VJP with a ZERO x-out cotangent — the forward
           DISCARDED its kernel ``out_x`` (the scan owns the x-chain), so
           that pullback vanishes structurally; ``in_y_bar`` becomes the
           previous physical row's ``out_y`` cotangent (the reversed
           transverse chaining);
        2. the x-face-chain transpose
           (:func:`~orpheus.sn.sweep.scan._x_scan_faces_transpose` — the
           VJP of the forward's reflection scan, same multiplier ``α``,
           opposite direction), whose seed cotangent is the row's domain
           x-INFLOW cotangent (→ the capture);
        3. the β-pullback ``ψ̄† += β̄·β_pullback`` — the scan's faces were
           reconstructed FROM the probe, so the face-chain cotangent flows
           back onto it (the scheme's ψ̄-independent
           :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.reflect_scan_coefficients_transpose`).

        Returns ``(psi_cot_octant, capture)`` with the per-axis physical
        IN-face cotangents ``(cap_x_cot, in_y_bar_final)``.
        """
        sig_t = operands.sig_t                      # (ng, nx, ny)
        ng, nx, ny = sig_t.shape
        sxm_addr, sym_addr = signs_addr             # MIRROR label (±1, never 0)
        out_x_bar, out_y_bar = out_bars             # (N_oct, ng, ny) / (N_oct, ng, nx)
        s_x, s_y = (s[oct_idx] for s in operands.str_axes)
        res_bar_oct = operands.probe[oct_idx]       # (N_oct, ng, nx, ny) — r̄
        N_oct = oct_idx.size

        # The scan-transpose primitive takes the PHYSICAL forward orientation
        # (physical sx = −sxm_addr); the mirror label orders the ROW march.
        x_reverse_physical = sxm_addr > 0
        scheme = self.spatial_closure
        psi_cot_oct = np.empty((N_oct, ng, nx, ny))
        cap_x_cot = np.empty((N_oct, ng, ny))       # physical x-IN cotangent, per row
        s_x_row = s_x[:, None, :]                   # (N_oct, 1, nx) — row-invariant
        zero_out_x_bar = np.zeros((N_oct, ng, nx))

        out_y_bar_run = out_y_bar                   # final physical row's out_y cotangent
        y_rows = range(ny) if sym_addr >= 0 else range(ny - 1, -1, -1)
        for j in y_rows:
            res_bar_row = res_bar_oct[:, :, :, j]   # (N_oct, ng, nx)
            psi_cot_kernel, (in_x_bar, in_y_bar) = (
                scheme.residual_kernel_batch_transpose(
                    res_bar=res_bar_row,
                    psi_out_bar=(zero_out_x_bar, out_y_bar_run),
                    s_axes=(s_x_row, s_y[:, j][:, None, None]),
                    reaction_xs=sig_t[:, :, j],
                )
            )
            alpha, beta_pullback = scheme.reflect_scan_coefficients_transpose(
                res_bar_row,
            )
            beta_bar, x_seed_bar = _x_scan_faces_transpose(
                alpha, in_x_bar, out_x_bar[:, :, j], x_reverse_physical,
            )
            psi_cot_oct[:, :, :, j] = psi_cot_kernel + beta_bar * beta_pullback
            out_y_bar_run = in_y_bar                # reversed transverse chaining
            cap_x_cot[:, :, j] = x_seed_bar
        return psi_cot_oct, (cap_x_cot, out_y_bar_run)

    @property
    def has_transpose_walk(self) -> bool:
        """True — both arms reverse (#310 C4).

        1-D: the shared reverse loop walk (#280 2.5a / #310 C2); 2-D
        Cartesian: the row-march reverse (#310 C4-b).  ``supports`` already
        narrows selection to exactly those arms (1-D affine-scannable /
        2-D facewise), and the KERNEL factor of ``is_adjointable``
        (``scheme.has_transpose_kernel``) carries the scheme narrowing —
        so the orientation factor is unconditionally honest here.
        """
        return True


# ═══════════════════════════════════════════════════════════════════════
# Registry + factory — the single selection source of truth
# ═══════════════════════════════════════════════════════════════════════

#: The CONCRETE strategy leaves (never the abstract ``_LossRepresentation`` /
#: ``_DAGWavefront`` bases) — :func:`default_for` constructs whichever it
#: picks, so only buildable strategies belong here.
#:
#: Selection priority order: the 1-D scan, the 2-D Cartesian scan-march
#: (the production default since the S6.9 Fork-B2 flip, 2026-06-11), the
#: wavefront window (a selectable peer, d=2), then the full-field oracle
#: (any-d Cartesian — the spine a d≥3 mesh falls through to, C3.6).
#: :func:`default_for` returns the FIRST that applies — the registry ORDER
#: is the default-selection policy, single-sourced here.
LOSS_REPRESENTATIONS: tuple[type[_LossRepresentation], ...] = (
    CumprodScan,
    ScanMarch,
    MovingFrontierWindow,
    FullFieldWavefront,
)


def default_for(
    mesh: "SNMesh",
    spatial_closure: "DiscretizationSchemeBase",
    angular_closure: "AngularClosureBase",
) -> LossRepresentation:
    """Select the default sweep strategy for ``mesh``.

    Returns the first strategy in :data:`LOSS_REPRESENTATIONS` whose
    :meth:`~LossRepresentation.supports` admits ``mesh`` — the best *available*
    production optimization, falling back to the spine: 1-D →
    :class:`CumprodScan`; multi-D Cartesian → :class:`ScanMarch` (the S6.9
    Fork-B2 flip, 2026-06-11 — measured 0.57–0.84× the window's sweep time at
    identical peak memory, issue #222).  :class:`MovingFrontierWindow` stays a
    selectable peer (a genuinely different schedule — anti-diagonal wavefront
    vs row-march — kept by user decision: multiple proper methods ARE the
    point of selectability), pinned end-to-end by the forced-window gates in
    ``test_scan_march_end_to_end.py``.

    Raises
    ------
    IncompatibleRepresentation
        If no strategy applies.  Reachable for a curvilinear (sphere/cylinder)
        mesh paired with a slab-only scheme (Linear-Discontinuous; #236 ST2):
        the curvilinear-capability gate rejects it up front with a specific
        reason.  Otherwise unreachable for a constructible mesh whose scheme is
        geometry-capable (every 1-D mesh → ``CumprodScan``; every 2-D Cartesian
        mesh → ``ScanMarch``; a d≥3 Cartesian mesh — axis-native via
        ``SNMesh.from_axes`` since C5.5 (#225) — → ``FullFieldWavefront``, the
        never-stuck any-d spine).
    """
    # #236 ST2: reject a (scheme × geometry) pairing the scheme has no closure
    # for UP FRONT, with the SPECIFIC reason.  NOT redundant with the loop: the
    # loop's supports() reject it via the SAME gate, but exhaust to the GENERIC
    # "no sweep strategy supports this mesh (ndim=…)" fall-through — which reads
    # as a dimensionality problem, not a scheme-geometry one.  Do NOT drop this
    # as a "DRY cleanup"; the specific reason is pinned by
    # ``test_unified_sweep_dispatch.py::TestHonestCurvilinearSchemeSelection``.
    geometry = _curvilinear_capability(mesh, spatial_closure)
    if not geometry.ok:
        raise IncompatibleRepresentation(geometry.reason)
    for cls in LOSS_REPRESENTATIONS:
        if cls.supports(mesh, spatial_closure).ok:
            return cls(mesh, spatial_closure, angular_closure)
    raise IncompatibleRepresentation(
        f"no sweep strategy supports this mesh "
        f"(ndim={mesh.ndim}, coord={mesh.coord.value!r}, "
        f"is_cartesian={mesh.is_cartesian})."
    )


# ═══════════════════════════════════════════════════════════════════════
# ORCHESTRATION — the transport-sweep entry + the schedule loop + the 1-D
# unified scan body (relocated from the dissolved ``sweep.py`` at S6.4(f):
# the walks live with their owner, and the historical lazy-import cycle is
# gone).
#
# References (carried with the 1-D body):
#
# * Hébert, A. (2009). *Applied Reactor Physics*. Ch. 3 §3.9.3 (cylinder,
#   pp. 137-141) / §3.9.4 (sphere, pp. 141-144) — curvilinear SN
#   cell-balance, sweep ordering, Carlson starting direction.  NOT the
#   source of the weighted tau (he defines none; he ships the plain
#   angular diamond).
# * Morel, J. E., & Montry, G. R. (1984). TTSP 13(5):615-633 — the
#   weighted angular closure tau, PRIMARY.
# * Bailey, Morel & Chang (2010). NSE 165(2):149-169 — Eqs. (42)/(43),
#   the form of tau this code implements (their Eq. (41), beta = 0, is
#   what determines it).
# * Lewis & Miller (1984). *Computational Methods of Neutron Transport.*
#   §4.5 (curvilinear DD); §5.3 (DD/WDD/Step/LD); §6.4 (sweep ordering).
# * Blelloch (1990). CMU-CS-90-190 §1.5 — first-order linear recurrence
#   closed form (the cumprod/cumsum scan the 1-D body evaluates).
# ═══════════════════════════════════════════════════════════════════════



#: The one μ-direction trichotomy threshold of the 1-D walk: an ordinate is a
#: FORWARD leg member at ``μ > +eps``, a BACKWARD leg member at ``μ < −eps``,
#: and DEGENERATE (pure-azimuthal — no radial streaming, not on any leg) at
#: ``|μ| ≤ eps``.  Single source for the leg masks (:meth:`_OneDimScanWalk.
#: _dag_legs`) and the degenerate set (:meth:`_OneDimScanWalk.
#: _degenerate_positions`) so the three classes PARTITION the quadrature by
#: construction — two thresholds could silently drop or double-count an
#: ordinate.  (``_run``'s slab ``μ >= 0`` split is a separate micro-seam no
#: GL ordinate hits; it re-poses at 2.5b.)
_MU_DIRECTION_EPS = 1e-15


@dataclass(frozen=True)
class _WalkLeg:
    r"""One (μ-level × direction) chain of the 1-D walk DAG.

    The 1-D sweep DAG's node set factorizes into LEGS: for each quadrature
    μ-level ``p`` and each direction class (``μ > +eps`` outward,
    ``μ < −eps`` inward) the member ordinates march the SAME cell chain in
    the SAME order, carrying one face flux (forward) or one face cotangent
    (adjoint).  A leg is that chain with its ordinate bundle — the unit
    :meth:`_OneDimScanWalk._loop_walk` traverses.

    ``cells`` is in TRAVERSAL order for the orientation at hand: the
    builder (:meth:`_OneDimScanWalk._dag_legs`) materializes the DOWNWIND
    (``dag_walk``) order; :func:`_reverse_traversal` reverses it for the
    adjoint (reverse-mode is reverse program order).  ``within`` (positions
    inside the level) and ``ordinates`` (global indices) are the SAME
    selection in the two indexing vocabularies the kernels consume
    (``angular_closure.cell_contribution`` is level-positional; the
    flux buffers are global-ordinate-indexed).
    """

    mu_level_idx: int
    direction_sign: int
    within: "np.ndarray"       # (K,) positions within the μ-level
    ordinates: "np.ndarray"    # (K,) global ordinate indices
    abs_mu: "np.ndarray"       # (K,) |μ| per leg ordinate
    cells: tuple[int, ...]     # the cell chain, in traversal order


def _reverse_traversal(legs: "tuple[_WalkLeg, ...]") -> "tuple[_WalkLeg, ...]":
    r"""The exact-reverse traversal of a leg schedule — reverse-mode order.

    Reverse-mode AD retraces the primal program backwards: the legs in
    reverse schedule order AND each leg's cell chain reversed.  The pole
    handoff reverses with it — the primal pole edge (inward legs seed their
    mirror outward legs, ERR-058a) becomes the outward-legs-feed-inward-legs
    cotangent edge, which is exactly why every reversed ``+1`` leg precedes
    every reversed ``−1`` leg in the output.
    """
    return tuple(
        replace(leg, cells=leg.cells[::-1]) for leg in reversed(legs)
    )


@dataclass(frozen=True)
class _OneDimScanWalk:
    r"""The shared 1-D-scan frame — the 1-D analogue of :class:`_OctantWalk`.

    Owns the geometry-blind 1-D SN sweep (the SOLVE direction), shared by
    :meth:`CumprodScan.sweep` and the :class:`ScanMarch` 1-D branch.  Like
    ``_OctantWalk`` it is a frozen ``mesh`` holder; the per-ordinate cache stash,
    the slab joint-batch + curvilinear per-ordinate bodies, and the two-stratum
    cache ensure/stash live here.

    Two frames, each shared across ORIENTATION (#280, Phase 2.5):

    * the **apply-loop frame** — :meth:`loss_action` (forward matvec
      :math:`(L+C)\psi`) and :meth:`loss_action_transpose` (adjoint matvec
      :math:`(L+C)^{\mathsf T}\varphi`) are ONE orientation-parametrized
      per-cell loop over the one DAG: both route their marches through
      :meth:`_loop_walk` on :meth:`_dag_legs`-built legs, forking only at
      the per-cell kernel closures (the ``_OctantWalk._interior_walk``
      cell-kernel-injection discipline, realized here at 2.5a).  The
      adjoint traverses :func:`_reverse_traversal` of the SAME legs —
      reverse-mode is reverse program order, never a twin frame.
    * the **solve-scan frame** — :meth:`sweep` → :meth:`_run` rides the
      Blelloch affine scan, NOT the per-cell loop; its transpose (2.5b)
      is the REVERSE-SCAN coherent with ``_run``, not a reverse loop.

    Execution {scan (solve) / cell loop (apply)} is thus a non-free third
    axis keyed by the kernel, while orientation (fwd ↔ adj) is the
    coherence axis each frame shares — the #280 shape.
    """

    mesh: "SNMesh"
    spatial_closure: "DiscretizationSchemeBase"
    angular_closure: "AngularClosureBase"

    def _dag_legs(self) -> "tuple[_WalkLeg, ...]":
        r"""Every non-empty leg of the 1-D walk DAG, in DEPENDENCY order.

        Dependency order = all ``−1`` (inward) legs, then all ``+1``
        (outward) legs, μ-levels ascending within each class.  The ONLY
        inter-leg edge of the 1-D DAG is the curvilinear pole continuation
        :math:`\psi(0, +\mu) = \psi(0, -\mu)` (ERR-058a — the Carlson
        coupled-pole seed), which makes every inward leg a predecessor of
        its mirror outward leg; slab legs are independent (both faces are
        given traces).  The FORWARD orientation traverses this order; the
        ADJOINT traverses :func:`_reverse_traversal` of it.

        The ±eps direction masks and the ``dag_walk_cell_indices`` order
        are materialized HERE ONCE for both orientations — the leg
        decomposition can no longer drift between the forward and adjoint
        walks (the pre-2.5a lockstep duplication).  Ordinates with
        :math:`|\mu| \le` ``_MU_DIRECTION_EPS`` are on NO leg — they are
        the :meth:`_degenerate_positions` set (volumetric balance, no
        face march).
        """
        sn_mesh = self.mesh
        mu_x = sn_mesh.quad.mu_x
        level_indices = self.angular_closure.level_indices
        legs: list[_WalkLeg] = []
        for direction_sign in (-1, +1):
            for p, level_idx in enumerate(level_indices):
                level_idx_arr = np.asarray(level_idx)
                mu_level = mu_x[level_idx_arr]
                within_mask = (
                    mu_level > +_MU_DIRECTION_EPS if direction_sign > 0
                    else mu_level < -_MU_DIRECTION_EPS
                )
                if not np.any(within_mask):
                    continue
                ordinates = level_idx_arr[within_mask]
                cells = tuple(sn_mesh.dag_walk_cell_indices(
                    direction_sign=direction_sign, mu_level_idx=p,
                ))
                if not cells:
                    continue
                legs.append(_WalkLeg(
                    mu_level_idx=p,
                    direction_sign=direction_sign,
                    within=np.where(within_mask)[0],
                    ordinates=ordinates,
                    abs_mu=np.abs(mu_x[ordinates]),
                    cells=cells,
                ))
        return tuple(legs)

    def _loop_walk(
        self,
        legs: "tuple[_WalkLeg, ...]",
        *,
        open_leg: "Callable[[_WalkLeg], np.ndarray]",
        visit: "Callable[[_WalkLeg, int, np.ndarray], np.ndarray]",
        close_leg: "Callable[[_WalkLeg, np.ndarray], None]",
    ) -> None:
        r"""THE shared 1-D per-cell loop frame (#280 2.5a — the one-loop seam).

        For each leg: bind the marching carry off the leg's upwind endpoint
        (``open_leg`` — the forward's face-flux seed, the adjoint's outflow
        cotangent), advance it through the leg's cells (``visit`` — the
        per-cell kernel, returning the updated carry), and deposit it at
        the downwind endpoint (``close_leg`` — the forward's outflow shed,
        the adjoint's seed-cotangent routing).  Both matvec orientations
        route through here, so "the adjoint walks the SAME DAG, reversed"
        is a code fact, not a test-maintained coincidence (spy + AST
        tripwire: ``tests/sn/sweep/core/test_one_dim_loop_walk.py``).

        Orientation is carried by the DATA, never by a flag: the leg
        schedule and each leg's ``cells`` arrive already in traversal
        order (forward = :meth:`_dag_legs` as built, downwind; adjoint =
        :func:`_reverse_traversal` of it), and the endpoint bindings are
        the injected callables — the ``_ApplyOperands`` /
        ``_SolveOperands`` / ``_SweepEmit`` object discipline's sibling.
        """
        for leg in legs:
            carry = open_leg(leg)
            for i in leg.cells:
                carry = visit(leg, i, carry)
            close_leg(leg, carry)

    def _degenerate_positions(self) -> "tuple[np.ndarray, list[int], list[int]]":
        r"""The degenerate pure-azimuthal ordinates + their level positions.

        Global indices ``global_deg`` with :math:`|\mu_x| \le`
        ``_MU_DIRECTION_EPS`` (no radial streaming — e.g. the
        :math:`\varphi = \pi/2,\,3\pi/2` samples of an equispaced product
        quadrature), each resolved to its ``(level, within-level position)``
        pair for the closure's positional API.  These ordinates sit on NO
        leg of the walk DAG: their cell balance is volumetric (zero face
        coupling, ``A_downstream = 0``), so both orientations handle them
        in a dedicated per-cell block OUTSIDE :meth:`_loop_walk` — the 1-D
        sibling of ``_OctantWalk``'s ``pure_z`` branch.
        """
        mu_x = self.mesh.quad.mu_x
        level_indices = self.angular_closure.level_indices
        global_deg = np.where(np.abs(mu_x) < _MU_DIRECTION_EPS)[0]
        deg_level: list[int] = []
        deg_within: list[int] = []
        for n_global in global_deg:
            for p, lvl in enumerate(level_indices):
                lvl_arr = np.asarray(lvl)
                pos = np.where(lvl_arr == n_global)[0]
                if pos.size > 0:
                    deg_level.append(p)
                    deg_within.append(int(pos[0]))
                    break
        return global_deg, deg_level, deg_within

    # ── The starting-direction (ψ½) block — retired from the walk (step 6).
    #
    # The ψ½ rows are the M grid's OWN blocks now: the forward/transpose
    # residual kernels live single-sourced in
    # ``orpheus/sn/sweep/psi_half_angle_seed.py`` and are consumed by the
    # standalone ``A_BB`` operator
    # (``RadialCharacteristicOperator.apply``/``.apply_transpose``); the
    # seed→bulk M-M recurrence coupling's transpose is the explicit grid
    # block ``RadialCharacteristicSeeding.apply_transpose``.  The walk's
    # fused ``_seed_rows_forward``/``_seed_rows_transpose`` glue retired
    # with the joint channel.

    def sweep(
        self,
        Q: np.ndarray,
        sig_t: np.ndarray,
        boundary_flux: "AngularBoundaryFlux",
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Geometry-blind 1-D SN sweep — three numpy tensor ops per ordinate.

        Replaces ``_sweep_1d_cartesian`` and ``_sweep_1d_curvilinear`` with
        one body driven by the two-stratum precomputed cache.  Slab,
        sphere, and cylinder share THE SAME scan expression; the
        cache abstracts the geometry difference (slab carries neutral
        curvature values; the M-M angular thread and Carlson seed run on
        the curvilinear arms, keyed on the mesh's ``coord``).

        Per-ordinate hot path
        ---------------------

        1. ``b = 2 · (QV_chain + ang_contrib) · coll.inverse_denom[n]``
           — per-cell (in chain order) affine additive coefficient.
        2. ``psi_face = ordinate_scan(coll.a_attenuation[n], b, psi_in)``
           — the Blelloch closed form, three numpy ops internally.
        3. ``psi_avg = 0.5 · (psi_in_chain + psi_face)``
           — DD spatial closure.

        For the rare degenerate cylindrical pure-azimuthal ordinate
        (``geom.is_degenerate[n] == True``, ``|η| < 10^{-15}``), the scan
        is meaningless and the slow per-cell ``scheme.update`` path
        runs instead.

        Cache provenance
        ----------------

        The cache is interned in the strategy layer
        (:func:`geometry_cache_for`'s ``WeakKeyDictionary``) — keyed weakly
        BY ``self.mesh`` and validated against the handed closure's
        identity, never stashed ON the mesh (only the σ stratum
        ``_coll_cache`` and ``_pole_mirror_cache`` remain mesh attributes).
        :class:`SNSolver.__init__` resolves it eagerly; a sweep invoked
        outside the solver (e.g. ad-hoc tests) resolves it lazily through
        the same intern on first call.

        Bit-identity contract
        ---------------------

        The cache-driven path produces algebraically the SAME values as the
        per-cell ``scheme.update`` reference iteration (the Pattern 2
        dual-view contract) — the cache precomputes once at solver
        construction what the reference rebuilds every sweep.  The dual-view
        test (``tests/sn/sweep/core/test_cache.py``) pins this at
        ``rtol=1e-13`` across the parametrised geometry × ng × source grid;
        slab regression snapshots stay bit-identical at ``rtol=1e-12``.
        """
        geom = self._ensure_geom_cache()
        coll = self._ensure_coll_cache(sig_t, geom)
        return self._run(Q, sig_t, boundary_flux, geom, coll)

    def sweep_transpose(
        self,
        bulk_cot: np.ndarray,
        sigma: np.ndarray,
        boundary_cot: "BoundaryField",
    ) -> "tuple[np.ndarray, AngularBoundarySourceSink]":
        r"""The transpose-solve :math:`(L+C)^{-\mathsf T}` — the REVERSE-SCAN.

        The solve-scan frame's adjoint (#280 Phase 2.5b): the transpose sibling
        of :meth:`sweep`, exactly as :meth:`loss_action_transpose` is the
        adjoint sibling of :meth:`loss_action` in the apply-loop frame.  Drives
        the two-stratum cache into :meth:`_run_transpose` (the reverse-mode
        adjoint of :meth:`_run`), so the transposed recurrence shares ``_run``'s
        ``ordinate_scan`` substrate (via :func:`ordinate_scan_transpose`) rather
        than duplicating a reverse loop.
        """
        geom = self._ensure_geom_cache()
        coll = self._ensure_coll_cache(sigma, geom)
        return self._run_transpose(bulk_cot, sigma, boundary_cot, geom, coll)

    def loss_action(
        self, sigma: "np.ndarray", psi: "FullField",
    ) -> "FullField":
        r"""1-D forward loss action ``(L+C)ψ`` — the matvec, apply direction.

        #206 Phase C: ``(L+C)ψ`` via the shared :meth:`_apply_walk` (the
        apply-direction twin of :meth:`sweep` — L21 "matvec ≡ sweep"). Returns
        the FULL ``(L+C)ψ`` for the given ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply` recovers
        ``Lψ`` by calling this walk at σ = 0 (#257 S8b) — the Resolution-A
        identity as the σ-free reading, not a subtraction.  ``sigma`` is the
        diagonal coefficient, passed explicitly (#240 Step B).

        On a carrying mesh this is the ray-decoupled ``(A,A)`` block action
        (the walk's welded seed feed reads zeros; step 6 — presence is
        structural): the joint ``M`` action is the within-group grid's
        :meth:`~orpheus.numerics.coupled_system.CoupledOperator.apply`.
        """
        from orpheus.transport.full_field import FullField
        from orpheus.transport.source_sinks import AngularSourceSink
        m_cell, m_boundary = self._apply_walk(sigma, psi)
        # The matvec output carries the trailing 2^d spatial-moment axis at a
        # multi-moment closure (the φ̂ iterate, #240 D5b-S3); the typed wrap
        # selects the scheme's spatial-moment axis.  DD/Step → no factor, byte-id.
        return FullField(
            interior=AngularSourceSink(
                values=m_cell, space=self.mesh.angular_trial_space,
            ),
            boundary=m_boundary,
        )

    def _apply_walk(
        self, sigma: "np.ndarray", psi: "FullField",
    ) -> "tuple[np.ndarray, AngularBoundarySourceSink]":
        r"""The 1-D apply-direction walk — the fused ``(L+C)ψ`` single emission.

        The apply direction is the structural twin of :meth:`sweep` (L21 "matvec
        ≡ sweep"). The sweep SOLVES ``(L+C)⁻¹q``; this APPLIES ``(L+C)ψ`` to a
        KNOWN probe ψ̄.  Since the apply has a concrete ψ̄ it rides a
        scheme-specific density residual (#158 the coefficient model):

        * **Cartesian (every affine scheme)** —
          ``scheme.residual_kernel_batch`` (the ÷V ``g=|μ|/Δ`` kernel)
          returns the density residual AND the outgoing face in one call.  DD and
          LD route through it UNIFORMLY (#158/#240) — DD reproduces its diamond
          march, LD its Schur residual, with no scheme branch.  ``s = 2|μ|/Δ`` is
          scheme-agnostic (LD's kernel halves it to ``g = |μ|/Δ`` internally).
          (DD's Cartesian matvec re-associates ~1 ULP vs the pre-#240
          ``cell_balance`` path on non-power-of-2 widths — a deliberate
          principled-equivalence re-baseline, not a special case.)
        * **Curvilinear (DD-only)** — the ``cell_balance_for_streaming`` density
          path (``m_full = (denom·ψ̄ − numer_upstream)/V``) carrying the
          Morel–Montry angular redistribution thread IN-SWEEP (NOT a pure
          ``(a, inverse_denom, w)`` coefficient), with DD's diamond march
          ``out = 2ψ̄ − in`` inlined.  Curvilinear SN is DD-only (the LD
          curvilinear closure is not yet implemented, #158), so this is a single-occupant
          geometry, not a polymorphism gap.  The angular-redistribution term is
          verified end-to-end by the anisotropic curvilinear MMS
          (``tests/sn/verification/mms/test_curvilinear_aniso_convergence.py``,
          ``catches("ERR-026")``); #238 retired the separately-applicable
          ``M_angular_redist`` leaf that re-walked here only to isolate this
          term (it had no production consumer).

        Returns ``(m_full_cell, m_boundary)`` — the bulk buffer in cell-first
        ``(N, ng, nx)`` layout + the boundary trace block (OUTFLOW =
        self-consistency defect, INFLOW = identity; NO BC reflection — the
        sibling ``−B`` carries it). The Morel–Montry angular redistribution +
        the Carlson coupled-pole seed (curvilinear) ride through
        ``angular_closure`` (ERR-058 / #195 — NEVER re-inlined). ``sigma``
        is the ``(ng, nx)`` group diagonal cross-section, passed directly — the
        frame needs no operator handle (it may be ``σ_t`` OR the removal ``σ_r``
        — the caller single-sources it; the frame never assumes ``σ_t``).  Since #240 Phase 2 Step B the protocol
        :meth:`loss_action` / :meth:`loss_action_transpose` ALSO take ``sigma``
        directly (symmetric with :meth:`sweep`'s ``sig_t``), so the diagonal is
        single-sourced by the CALLER (``StreamingOperator.apply`` passes its
        ``sigma_t``; ``StreamingCollisionOperator.apply`` passes the composite's
        diagonal ``self.sigma``) — the frame never reads it off an operator
        handle.
        """
        from orpheus.transport.source_sinks import AngularBoundarySourceSink
        from orpheus.transport.spatial.cell_balance import cell_balance_for_streaming

        sn_mesh = self.mesh
        psi_view = psi.interior.values
        quad = sn_mesh.quad
        N = quad.N
        ng = psi_view.shape[1]
        nx = sn_mesh.nx
        # The chart is an ENUM on the mesh.  Until 2026-08-26 this frame
        # re-derived a string from it through a defaulted ``getattr`` and
        # re-validated that string's domain at runtime -- a domain NO
        # consumer here reads: [M] all six uses below ask only
        # ``== "cartesian"`` / ``!= "cartesian"``, never sphere-vs-cylinder.
        # A three-valued string used as a boolean, with a guard for values
        # nothing branches on.
        is_cartesian = sn_mesh.is_cartesian
        if is_cartesian and not sn_mesh.is_1d:
            raise NotImplementedError(
                "_OneDimScanWalk._apply_walk: multi-D Cartesian is not "
                "handled by the 1-D scan walk; multi-D Cartesian routes "
                "through ScanMarch / _OctantWalk (the production default), "
                "not this frame."
            )

        # The operator hands its bound closure through the representation
        # (P4.9b — the walk consumes the HANDED pair, never the hub's attrs).
        angular_closure = self.angular_closure

        mu_x = quad.mu_x
        A = sn_mesh.areas

        psi_g_first = psi_view.swapaxes(0, 1)
        # The unified moment matvec (#240 D5b-S3): a multi-moment closure (LD)
        # carries a trailing 2^d spatial-moment axis on the iterate.  In 1-D
        # d=1, 2^d = per_axis (2 for LD) and the FACE cochain is 2^{d-1} = 1
        # (scalar), so only the cell probe / residual carries the axis.  DD/Step
        # (per_axis == 1) → ``()`` tail, every buffer byte-identical.  The width
        # is read OFF the iterate's space (the single source of truth).
        per_axis = self.spatial_closure.spatial_basis_per_axis
        moment_tail = face_moment_tail(cell_moment_count(per_axis, sn_mesh.ndim))
        out_g_first = np.zeros((ng, N, nx, *moment_tail))

        V = sn_mesh.volumes
        sigma_gx = sigma

        boundary = psi.boundary
        trace = sn_mesh.angular_trace
        has_inner_face = "xmin" in boundary.layout.faces
        face_outer = boundary.face_view("xmax")
        face_inner = boundary.face_view("xmin") if has_inner_face else None

        if is_cartesian and face_inner is None:
            raise ValueError(
                "Slab geometry requires psi.boundary.xmin_face to be "
                "populated."
            )

        # #282 route (a) → B.2d → step 6: on a carrying mesh this walk IS
        # the ray-decoupled (A,A) BLOCK action — a ZERO seed feeds the
        # closure (bit-identical to the retired dead-slot convention: the
        # welded feed reads zeros).  The pre-2.5d
        # extrapolate-from-the-iterate treatment (the #282 back edge)
        # stays retired, never silently reproduced.  The joint M action
        # (a live ψ½ seed feeding the recurrence) is the within-group
        # grid's block matvec; the closure consumes the INTERIOR member
        # only (its ``.cells(p, -1)`` read — the M-M recurrence seed
        # lives on the marched cells).
        seed_field = None
        if sn_mesh.radial_characteristic_field_space is not None:
            from orpheus.transport.radial_characteristic_field import (
                RadialCharacteristicField,
            )

            seed_field = RadialCharacteristicField.flux_zeros(sn_mesh.radial_characteristic_field_space)
        psi_state = angular_closure.precompute_psi_state(
            psi_view,
            radial_characteristic=(
                seed_field.interior if seed_field is not None else None
            ),
        )

        # The d=1 matvec probe (the iterate) is GLOBAL-frame; the residual
        # kernel works in the SWEEP frame, so for a BACKWARD sweep
        # (``direction_sign < 0``) the LD slope moment must be re-signed on the
        # probe IN and the residual OUT — the d=1 counterpart of the d≥2
        # ``_CellResidual`` frame map (#240 D5b-S3 root cause).  ``direction_sign``
        # IS the d=1 ``signs_eff``, so this rides the SAME single-source
        # binding (:func:`frame_signs_for`) as the d≥2 sites (``None`` for
        # DD/Step → byte-identical).

        spatial_closure = self.spatial_closure
        dag_legs = self._dag_legs()

        def _sweep_direction(
            direction_sign: int,
            psi_face_in_init: np.ndarray,
        ) -> np.ndarray:
            frame_signs = frame_signs_for(spatial_closure, (direction_sign,))
            # The d=1 scan probe + residual are genuine moment buffers at a
            # multi-moment closure (LD); scalar at DD/Step (frame_signs None →
            # the reframe is a short-circuit no-op regardless).
            is_moment_valued = spatial_closure.is_multi_moment
            outflow_at_end = np.zeros((ng, N))

            def open_leg(leg: _WalkLeg) -> np.ndarray:
                return psi_face_in_init[leg.ordinates, :].T

            def visit(
                leg: _WalkLeg, i: int, psi_face_in: np.ndarray,
            ) -> np.ndarray:
                psi_cell = psi_g_first[:, leg.ordinates, i]
                if is_cartesian:
                    # Coefficient model (#158/#240): the Cartesian matvec rides
                    # the scheme's group-2 ÷V kernel ``residual_kernel_batch``
                    # UNIFORMLY — DD reproduces its diamond march, LD its
                    # bilinear UBLD residual, with NO scheme branch (the kernel
                    # returns BOTH the moment residual and the outgoing face,
                    # the apply twin of the scan solve).  ``s_axes`` is the RAW
                    # down-face streaming ``g = |μ|·face_area_downstream/V = |μ|/Δ`` (slab
                    # face_area_downstream=1, V=Δ).  Source-free apply (``Q_cells = 0``).
                    # Cartesian has no Morel–Montry angular redistribution
                    # thread (the curvilinear arm below carries it).
                    # NOTE(#240): ``leg.abs_mu / V[i]`` re-derives the raw ``g``
                    # that ``SNMesh.streaming(0)`` already produces — a Pattern-2
                    # dup; single-sourcing it (``streaming(0)[leg.ordinates, i]``)
                    # is a deferred follow-up pending a widths-vs-volumes bit-id
                    # check.
                    #
                    # The unified moment matvec (#240 D5b-S3): the probe carries
                    # the trailing 2^d moment axis (``moment_tail``); the d=1
                    # FACE is scalar (2^{d-1} = 1).  ``swapaxes(0, 1)[:, :, None]``
                    # maps ``(ng, K[, 2^d]) → (K, ng, 1[, 2^d])`` (the ``[None]``
                    # inserts the n_diag axis BEFORE the moment axis) — agnostic
                    # over the trailing pack (DD scalar → byte-identical).
                    probe_cell = _reframe(
                        np.swapaxes(psi_cell, 0, 1)[:, :, None], frame_signs,
                        is_moment_valued=is_moment_valued,
                    )
                    resid, (psi_out_cell,) = (
                        spatial_closure.residual_kernel_batch(
                            psi_bar=probe_cell,
                            psi_in=(psi_face_in.T[:, :, None],),
                            s_axes=((leg.abs_mu / V[i])[:, None, None],),
                            reaction_xs=sigma_gx[:, i][None, :, None],
                            Q_cells=_MATVEC_ZERO_SOURCE,   # source-free apply
                        )
                    )
                    resid = _reframe(
                        resid, frame_signs, is_moment_valued=is_moment_valued,
                    )
                    # resid (K, ng, 1[, 2^d]) → (ng, K[, 2^d]); the outgoing
                    # face is scalar (K, ng, 1) → (ng, K).
                    out_g_first[:, leg.ordinates, i] = np.swapaxes(
                        resid[:, :, 0], 0, 1,
                    )
                    return psi_out_cell[:, :, 0].T                    # (ng, K)
                # Curvilinear matvec — the ``cell_balance`` density path
                # carrying the Morel–Montry angular thread (NOT a pure
                # (a, inverse_denom, w) coefficient, so it cannot ride the
                # coefficient-model kernel above).  Curvilinear SN is DD-only
                # (the LD curvilinear closure is not yet implemented — guarded in
                # ``LinearDiscontinuous.affine_scan_coefficients``), so DD's
                # diamond march ``out = 2ψ̄ − in`` inlined here is a
                # single-occupant geometry, NOT a polymorphism gap.
                angular_denom_term, angular_numer_upstream = (
                    angular_closure.cell_contribution(
                        psi_state, i, leg.mu_level_idx, leg.within,
                    )
                )
                A_downstream = A[i + 1] if direction_sign > 0 else A[i]
                denom, numer_upstream = cell_balance_for_streaming(
                    abs_mu=leg.abs_mu,
                    A_downstream=A_downstream,
                    face_area_total=A[i] + A[i + 1],
                    total_xs=sigma_gx[:, i],
                    volume=V[i],
                    psi_face_in=psi_face_in,
                    angular_denom_term=angular_denom_term,
                    angular_numer_upstream=angular_numer_upstream,
                )
                m_full = (denom * psi_cell - numer_upstream) / V[i]
                out_g_first[:, leg.ordinates, i] = m_full
                return 2.0 * psi_cell - psi_face_in   # DD diamond march

            def close_leg(leg: _WalkLeg, psi_face_in: np.ndarray) -> None:
                outflow_at_end[:, leg.ordinates] = psi_face_in

            self._loop_walk(
                tuple(
                    leg for leg in dag_legs
                    if leg.direction_sign == direction_sign
                ),
                open_leg=open_leg,
                visit=visit,
                close_leg=close_leg,
            )
            return outflow_at_end

        # Wave O O.4a.2 — KEYSTONE DELETED. The backward sweep seeds from the
        # GIVEN outer inflow trace (``face_outer``'s μ<0 / inward ordinates),
        # NOT from the forward sweep's own reflected outflow
        # (``inflow_full = bc_outer.apply(outflow_at_boundary.T)``). This
        # decouples bulk ↔ boundary inside one matvec call: the reflective
        # coupling moves to the sibling −B, and the outer Krylov/SI loop drives
        # the inflow consistency ``ψ.inflow − B·ψ.outflow → 0``.
        outflow_at_inner = _sweep_direction(-1, face_outer)

        if not is_cartesian:
            # Carlson coupled-pole spatial seed (ERR-058 a, Issue #195):
            # at r = 0 the outward characteristic is the CONTINUATION of
            # the inward one — ψ(0, +μ) = ψ(0, −μ) — so the +1 sweep's
            # pole-face seed is the −1 sweep's pole-face outflow at the
            # mirror ordinate (already computed above: data, propagated
            # from the outer boundary, lower-triangular).  The historical
            # innermost-CELL-CENTRE read ψ(Δr/2) was O(h)-wrong on
            # non-flat profiles (exact on flat ψ — which is why every
            # flat-flux gate stayed green) and is retired; this is the
            # #192-deferred "inward-determines-outward" pole condition.
            pole_face_seed = outflow_at_inner.T[self._ensure_pole_mirror()]
        else:
            # Slab: read the GIVEN inner inflow trace (the forward sweep's
            # μ>0 seed at xmin) directly. Wave O O.4a.2 — the BC reflection
            # is NOT re-derived here; it moves to the sibling −B.
            assert face_inner is not None  # guaranteed by the cartesian guard
            pole_face_seed = face_inner
        outflow_at_boundary = _sweep_direction(+1, pole_face_seed)

        # Degenerate-ordinate branch (cylinder) — the pure-azimuthal set, on
        # no leg of the DAG (volumetric balance, zero face coupling; the 1-D
        # sibling of ``_OctantWalk``'s pure-z branch).
        global_deg, deg_level, deg_within = self._degenerate_positions()
        if global_deg.size:
            n_deg = global_deg.size
            abs_mu_deg = np.abs(mu_x[global_deg])
            zero_face = np.zeros((ng, n_deg))
            for i in range(nx):
                angular_denom_term = np.empty(n_deg)
                angular_numer_upstream = np.empty((ng, n_deg))
                for col_idx in range(n_deg):
                    denom_one, numer_one = angular_closure.cell_contribution(
                        psi_state, i, deg_level[col_idx],
                        np.array([deg_within[col_idx]]),
                    )
                    angular_denom_term[col_idx] = denom_one[0]
                    angular_numer_upstream[:, col_idx] = numer_one[:, 0]

                psi_cell = psi_g_first[:, global_deg, i]
                denom, numer_upstream = cell_balance_for_streaming(
                    abs_mu=abs_mu_deg,
                    A_downstream=0.0,
                    face_area_total=A[i] + A[i + 1],
                    total_xs=sigma_gx[:, i],
                    volume=V[i],
                    psi_face_in=zero_face,
                    angular_denom_term=angular_denom_term,
                    angular_numer_upstream=angular_numer_upstream,
                )
                m_full = (denom * psi_cell - numer_upstream) / V[i]
                out_g_first[:, global_deg, i] = m_full

        m_cell = out_g_first.swapaxes(0, 1)

        # Wave O O.4a.2 — the boundary block of (L+C) carries the two trace
        # DIAGONALS of the block matrix; the off-diagonal −B is a sibling
        # operator (so this matvec contains NO BC reflection):
        #   * OUTFLOW slots — the self-consistency defect
        #     ``ψ.outflow − streamed`` (the r_outflow row's I·ψ.outflow
        #     diagonal minus L_out,b·ψ.interior). UNCHANGED from pre-extraction;
        #     kept as ``computed − stored`` so the vacuum path is bit-identical
        #     (the per-row sign is free — q.outflow ≡ 0, the outflow trace is a
        #     pure definition with no source).
        #   * INFLOW slots — the identity ``ψ.inflow`` (the r_inflow row's
        #     I·ψ.inflow diagonal). NEW at O.4a.2. The sibling −B adds
        #     −B·ψ.outflow, so the full (L+C−S−F−B) inflow residual is
        #     ``ψ.inflow − B·ψ.outflow`` (the consistency the outer loop drives
        #     to q.inflow, the prescribed inflow / zero for vacuum+reflective).
        # The outflow / inflow ordinate sets are the disjoint sign(Ω·n)
        # partitions read from the unified AngularTraceSpace selector (single source
        # of truth) — A.4 retired the inline ``mu_x > ±eps`` masks.
        m_boundary = AngularBoundarySourceSink.zeros(sn_mesh.angular_trace)
        outer_outflow = trace.outflow_indices_for_face("xmax")
        if outer_outflow.size:
            m_boundary.face_view("xmax")[outer_outflow, :] = (
                outflow_at_boundary[:, outer_outflow].T
                - face_outer[outer_outflow, :]
            )
        outer_inflow = trace.inflow_indices_for_face("xmax")
        if outer_inflow.size:
            m_boundary.face_view("xmax")[outer_inflow, :] = (
                face_outer[outer_inflow, :]
            )
        if face_inner is not None:
            inner_outflow = trace.outflow_indices_for_face("xmin")
            if inner_outflow.size:
                m_boundary.face_view("xmin")[inner_outflow, :] = (
                    outflow_at_inner[:, inner_outflow].T
                    - face_inner[inner_outflow, :]
                )
            inner_inflow = trace.inflow_indices_for_face("xmin")
            if inner_inflow.size:
                m_boundary.face_view("xmin")[inner_inflow, :] = (
                    face_inner[inner_inflow, :]
                )

        return m_cell, m_boundary

    def loss_action_transpose(
        self, sigma: "np.ndarray", phi: "FullField",
    ) -> "FullField":
        r"""1-D adjoint loss action ``(L+C)ᵀφ`` — the matvec transpose.

        The reverse-mode adjoint of :meth:`loss_action`.  The forward matvec is a
        forward-substitution sweep (lower-triangular in cell-visit order, with
        the Morel–Montry angular recurrence + Carlson pole seed forming a SECOND
        triangular factor in the ordinate index); its Euclidean transpose is the
        reverse-substitution sweep:

        * reversed cell traversal (the DD face-flux chain
          ``ψ_face_in ← 2·ψ_cell − ψ_face_in`` transposed);
        * the boundary block SWAPPED — the forward FULL operator reads the
          inflow trace and writes the outflow trace, so the transpose reads
          OUTFLOW cotangents and writes INFLOW cotangents;
        * the angular factor reversed, delegated to ``closure.angular_adjoint``
          (zero for the slab identity closure) — NEVER re-inlined; the
          Carlson coupled-pole seed adjoint routes through the mirror
          permutation.

        Every coefficient is ψ-independent (geometry + σ_t): ``denom`` is
        recomputed through the SAME ``cell_balance_for_streaming`` /
        ``cell_contribution`` the forward uses (Pattern 2 — no twin algebra).
        Since 2.5a (#280) the reversal itself is STRUCTURAL: the adjoint
        marches :func:`_reverse_traversal` of :meth:`_dag_legs` through the
        SAME :meth:`_loop_walk` frame the forward matvec uses — the leg
        decomposition and traversal topology cannot drift from the
        forward's; only the per-cell kernels and the endpoint bindings
        (outflow cotangent in, seed cotangent out) differ.  The visit
        mirrors ``_apply_walk``'s TWO ARMS (#310 C2): the CARTESIAN arm
        rides the scheme-uniform batch VJP
        :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.residual_kernel_batch_transpose`
        (DD + LD, no scheme branch, moment-tail + frame conjugation
        mirrored); the CURVILINEAR arm rides the registered cell-balance
        VJP
        :meth:`~orpheus.transport.spatial.scheme.DiscretizationSchemeBase.streaming_cell_transpose`
        (#310 C1, DD-only single-occupant geometry, Morel–Montry thread
        with the walk).
        Returns ``(L+C)ᵀφ`` for the given ``sigma``;
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.apply_transpose`
        recovers bare ``Lᵀφ`` by calling this walk at σ = 0 (#257 S8b) — ``C``
        is a self-adjoint diagonal, so the adjoint matvec is affine in σ too.
        Pinned by the G-adjoint reciprocity gate ``test_g_adjoint_reciprocity``
        (slab / sphere / cylinder, -O-firing) + its L11 wrong-trace-metric
        negative control.
        """
        from orpheus.transport.full_field import FullField
        from orpheus.transport.source_sinks import AngularSourceSink, AngularBoundarySourceSink
        from orpheus.transport.spatial.cell_balance import cell_balance_for_streaming

        sn_mesh = self.mesh
        quad = sn_mesh.quad
        N = quad.N
        ng = phi.interior.values.shape[1]
        nx = sn_mesh.nx
        # See _apply_walk's note: the twin of that prelude, and already
        # DRIFTED -- this copy never carried the domain re-validation.
        is_cartesian = sn_mesh.is_cartesian
        if is_cartesian and not sn_mesh.is_1d:
            # Structural 1-D-only guard (NOT a deferral since #310 C4): the
            # multi-D scan-family reverse is the row-march
            # (ScanMarch._loss_action_transpose_interior via the shared
            # _OctantWalk frame); this walk's leg decomposition is 1-D by
            # construction, so a multi-D mesh reaching it is a routing bug.
            raise NotImplementedError(
                "_OneDimScanWalk.loss_action_transpose is 1-D-only — the "
                "multi-D Cartesian adjoint of the scan family is the "
                "row-march reverse (ScanMarch.loss_action_transpose)."
            )
        if not type(self.spatial_closure).has_transpose_kernel:
            # The trait DERIVES from the transpose-kernel registrations
            # (#310 ruling 2: the Cartesian batch VJP, plus the curvilinear
            # cell-balance VJP iff the scheme claims curvilinear).  The
            # honest front door is StreamingOperator.is_adjointable (eager
            # ``.H`` raises MissingAdjoint); this guard is the backstop for
            # direct Euclidean apply_transpose calls that bypass ``.H``.
            raise NotImplementedError(
                "_OneDimScanWalk.loss_action_transpose: scheme "
                f"{type(self.spatial_closure).__name__} registers no transpose "
                "kernel pair (residual_kernel_batch_transpose; plus "
                "streaming_cell_transpose for a curvilinear scheme) — the "
                "LD/UBLD Schur-residual adjoint (cell-moment cotangents + "
                "the reverse moment-frame involution) lands at #310 C2."
            )

        closure = self.angular_closure
        scheme = self.spatial_closure
        # Mirror-ordinate permutation for the coupled-pole seed adjoint
        # (curvilinear only; the mesh-stashed derivation makes the
        # unconditional read a cache hit).
        mirror = self._ensure_pole_mirror()
        A = sn_mesh.areas
        V = sn_mesh.volumes
        sgx = sigma                                  # (ng, nx)
        trace = sn_mesh.angular_trace
        has_inner_face = "xmin" in phi.boundary.layout.faces

        # #282 route (a) → B.2d → step 6: this is the ray-decoupled (A,A)ᵀ
        # block action on a carrying mesh — the bulk/trace reversal is
        # chi-independent (M_BA = 0), and the seed pullback belongs to the
        # explicit A_ABᵀ grid block
        # (``RadialCharacteristicSeeding.apply_transpose``); the joint Mᵀ is
        # the within-group grid's ``CoupledOperator.apply_transpose``.

        out_bar = phi.interior.values.swapaxes(0, 1)   # (ng, N, nx[, 2^d])
        fo = phi.boundary.face_view("xmax")                       # (N, ng)

        # The unified moment adjoint (#310 C2 — mirror of _apply_walk): a
        # multi-moment closure (LD) carries the trailing 2^d spatial-moment
        # axis on the incoming cotangent and the ψ̄-cotangent buffer; the d=1
        # FACE cochain is scalar (2^{d-1} = 1) for every closure.  DD/Step
        # (per_axis == 1) → ``()`` tail, every buffer byte-identical.
        per_axis = scheme.spatial_basis_per_axis
        moment_tail = face_moment_tail(cell_moment_count(per_axis, sn_mesh.ndim))
        is_moment_valued = scheme.is_multi_moment
        frame_signs_by_dir = {
            ds: frame_signs_for(scheme, (ds,)) for ds in (+1, -1)
        }
        if phi.interior.values.shape[3:] != tuple(moment_tail):
            # Pattern-4 backstop: a cotangent whose spatial-moment tail does
            # not match the scheme's would BROADCAST silently through the
            # batch VJP (a tail-less field against LD's (…, 2^d) mass) —
            # refuse loudly instead.
            raise ValueError(
                "_OneDimScanWalk.loss_action_transpose: cotangent interior "
                f"shape {phi.interior.values.shape} does not carry the "
                f"scheme's spatial-moment tail {tuple(moment_tail)} "
                f"({type(scheme).__name__})."
            )

        psi_bar = np.zeros((ng, N, nx, *moment_tail))
        fo_bar = np.zeros((N, ng))
        # xmin-face cotangent: written by the slab arms below; on curvilinear
        # the pole is structurally NOT a face (#220), so it stays zero and is
        # never written back — same discard idiom as ``outflow_inner_bar``.
        fi_bar = np.zeros((N, ng))
        numer_bar = [
            np.zeros((ng, np.asarray(li).size, nx))
            for li in closure.level_indices
        ]

        # ── reverse the boundary writeback (mirror _compute_LpC m_boundary) ──
        # m.outflow = (swept outflow) − ψ.outflow;  m.inflow = ψ.inflow.
        outflow_boundary_bar = np.zeros((ng, N))    # +1 sweep outflow → xmax
        outflow_inner_bar = np.zeros((ng, N))       # −1 sweep outflow → xmin (slab); pole-discarded (curv)
        oo = trace.outflow_indices_for_face("xmax")
        oi = trace.inflow_indices_for_face("xmax")
        if oo.size:
            outflow_boundary_bar[:, oo] += fo[oo].T
            fo_bar[oo] += -fo[oo]
        if oi.size:
            fo_bar[oi] += fo[oi]
        if has_inner_face:
            fi = phi.boundary.face_view("xmin")                   # (N, ng)
            io = trace.outflow_indices_for_face("xmin")
            ii = trace.inflow_indices_for_face("xmin")
            if io.size:
                outflow_inner_bar[:, io] += fi[io].T
                fi_bar[io] += -fi[io]
            if ii.size:
                fi_bar[ii] += fi[ii]

        # ── ψ-independent angular_denom_term source (dummy state) ──
        # Coefficient-only use: ``cell_contribution`` reads ONLY the
        # denom leg off this state, so the zero seed of the ``None``
        # radial_characteristic is never consumed (documented on the ABC).
        psi_state_coef = closure.precompute_psi_state(np.zeros((N, ng, nx)))

        # ── reverse the spatial DD marches — the SAME legs, exact-reverse
        # order (#280 2.5a).  Reverse-mode retraces the primal walk
        # backwards: the adjoint marches ``_reverse_traversal(_dag_legs())``
        # through the SAME ``_loop_walk`` frame the forward matvec uses, so
        # every reversed +1 leg (whose close routes the pole-seed cotangent)
        # precedes every reversed −1 leg (whose open reads it), and each
        # leg's cells run upwind.  Leg slots are disjoint and the mirror
        # handoff is level-local, so this linearization is value-identical
        # to the pre-2.5a per-level (+1, −1) nesting — pinned by the frozen
        # ``walk_matvec_*`` adjoint baselines. ──

        def open_leg(leg: _WalkLeg) -> np.ndarray:
            return (
                outflow_boundary_bar[:, leg.ordinates]
                if leg.direction_sign > 0
                else outflow_inner_bar[:, leg.ordinates]
            ).copy()

        def visit(leg: _WalkLeg, i: int, f_bar: np.ndarray) -> np.ndarray:
            if is_cartesian:
                # ── Cartesian arm: the scheme-uniform ÷V batch VJP (#310 C2
                # — the exact mirror of _apply_walk's kernel arm).  DD and LD
                # route through ``residual_kernel_batch_transpose`` with NO
                # scheme branch; the frame conjugation transposes as itself
                # (the involution is diagonal), so the residual cotangent
                # reframes IN and the ψ̄ cotangent reframes OUT — the adjoint
                # of ``x ↦ D·K(D·x)`` is ``y ↦ D·Kᵀ(D·y)``.  Cartesian has
                # no Morel–Montry thread (mirror of the forward arm), so no
                # ``numer_bar`` accumulation here.
                frame_signs = frame_signs_by_dir[leg.direction_sign]
                ob = out_bar[:, leg.ordinates, i]          # (ng, K[, 2^d])
                res_bar_cell = _reframe(
                    np.swapaxes(ob, 0, 1)[:, :, None], frame_signs,
                    is_moment_valued=is_moment_valued,
                )                                          # (K, ng, 1[, 2^d])
                psi_bar_cot, (f_in_cot,) = (
                    scheme.residual_kernel_batch_transpose(
                        res_bar=res_bar_cell,
                        psi_out_bar=(f_bar.T[:, :, None],),
                        s_axes=((leg.abs_mu / V[i])[:, None, None],),
                        reaction_xs=sgx[:, i][None, :, None],
                    )
                )
                psi_bar_cot = _reframe(
                    psi_bar_cot, frame_signs, is_moment_valued=is_moment_valued,
                )
                # Each (ordinate, cell) slot is visited by exactly one leg,
                # so the single scatter-add is bit-identical to in-place
                # accumulation.
                psi_bar[:, leg.ordinates, i] += np.swapaxes(
                    psi_bar_cot[:, :, 0], 0, 1,
                )
                return f_in_cot[:, :, 0].T                 # (ng, K)
            # ── Curvilinear arm (DD-only, single-occupant geometry — the
            # mirror of _apply_walk's cell_balance arm): the registered
            # cell-balance VJP carrying the Morel–Montry thread (#310 C1).
            A_downstream = A[i + 1] if leg.direction_sign > 0 else A[i]
            face_area_total = A[i] + A[i + 1]
            angular_denom_term, _ = closure.cell_contribution(
                psi_state_coef, i, leg.mu_level_idx, leg.within,
            )
            denom, _ = cell_balance_for_streaming(
                abs_mu=leg.abs_mu,
                A_downstream=A_downstream,
                face_area_total=face_area_total,
                total_xs=sgx[:, i],
                volume=V[i],
                psi_face_in=np.zeros((ng, leg.within.size)),
                angular_denom_term=angular_denom_term,
                angular_numer_upstream=np.zeros((ng, leg.within.size)),
            )                                       # (ng, n_mask)
            ob = out_bar[:, leg.ordinates, i]
            # The scheme's registered spatial VJP (#310 C1): transposes the
            # cell relation {residual, face chain}; each (ordinate, cell)
            # slot is visited by exactly one leg, so the single scatter-add
            # below is bit-identical to in-place accumulation.
            psi_bar_cot, f_bar = scheme.streaming_cell_transpose(
                res_bar=ob,
                psi_out_bar=f_bar,
                denom=denom,
                abs_mu_A_total=leg.abs_mu * face_area_total,
                volume=V[i],
            )
            psi_bar[:, leg.ordinates, i] += psi_bar_cot
            # The Morel–Montry angular-numerator cotangent is the WALK's
            # (spatial-only kernel contract, #310 ruling 1); the downstream
            # angular thread reverses in ``angular_adjoint`` below.
            numer_bar[leg.mu_level_idx][:, leg.within, i] += -ob / V[i]
            return f_bar

        def close_leg(leg: _WalkLeg, f_bar: np.ndarray) -> None:
            # reverse the sweep seed
            if leg.direction_sign > 0:
                if not is_cartesian:
                    # adjoint of the Carlson coupled-pole seed: the
                    # forward +1 seed reads the −1 sweep's pole-face
                    # outflow at the mirror ordinate, so the seed
                    # cotangent routes into the −1 reversal's INITIAL
                    # outflow cotangent (mirror partners live in the
                    # same level; the reversed −1 legs read it AFTER
                    # every +1 leg has closed — the reversed pole edge).
                    outflow_inner_bar[:, mirror[leg.ordinates]] += f_bar
                else:
                    # slab +1 seed = ψ.inflow[xmin]
                    fi_bar[leg.ordinates] += f_bar.T
            else:
                fo_bar[leg.ordinates] += f_bar.T    # −1 seed = ψ.inflow[xmax]

        self._loop_walk(
            _reverse_traversal(self._dag_legs()),
            open_leg=open_leg,
            visit=visit,
            close_leg=close_leg,
        )

        # ── reverse the degenerate-ordinate branch (cylinder) — the
        # pure-azimuthal set, on no leg (volumetric balance, zero face
        # coupling, ``A_downstream = 0``).  The forward writes
        # ``m[deg, i] = (denom·ψ_cell − angular_numer)/V`` with NO face
        # march, so its transpose is slot-local: the ψ-diagonal lands on
        # ``psi_bar``, the angular-numerator cotangent on ``numer_bar``
        # (``angular_adjoint`` below reverses the M-M thread's
        # ψ-dependence).  Slot order vs the leg walk is free — degenerate
        # ordinates sit on no leg (the trichotomy partition pin).
        # MISSING pre-2.5a: an equispaced product quadrature
        # (φ = π/2, 3π/2 ⇒ |μ_x| ≈ 6e-17) silently DROPPED these rows
        # from the adjoint — every ``level_symmetric`` reciprocity row was
        # blind; caught by the ``cyl_product_2g`` G-reciprocity row that
        # landed with this block (red pre-fix, green post-fix).
        global_deg, deg_level, deg_within = self._degenerate_positions()
        if global_deg.size:
            n_deg = global_deg.size
            abs_mu_deg = np.abs(quad.mu_x[global_deg])
            zero_face = np.zeros((ng, n_deg))
            for i in range(nx):
                angular_denom_term = np.empty(n_deg)
                for col_idx in range(n_deg):
                    denom_one, _ = closure.cell_contribution(
                        psi_state_coef, i, deg_level[col_idx],
                        np.array([deg_within[col_idx]]),
                    )
                    angular_denom_term[col_idx] = denom_one[0]
                denom, _ = cell_balance_for_streaming(
                    abs_mu=abs_mu_deg,
                    A_downstream=0.0,
                    face_area_total=A[i] + A[i + 1],
                    total_xs=sgx[:, i],
                    volume=V[i],
                    psi_face_in=zero_face,
                    angular_denom_term=angular_denom_term,
                    angular_numer_upstream=zero_face,
                )
                ob = out_bar[:, global_deg, i]
                # reverse m = (denom·ψ − angular_numer)/V  (no face terms)
                psi_bar[:, global_deg, i] += denom * ob / V[i]
                for col_idx in range(n_deg):
                    numer_bar[deg_level[col_idx]][
                        :, deg_within[col_idx], i,
                    ] += -ob[:, col_idx] / V[i]

        # ── reverse the angular factor (delegated; zero for the slab closure) ──
        # #282 route (a) → step 6: the reverse M-M recurrence STOPS at the
        # seed on carrying levels — ``seed_cells_bar`` (the per-level
        # seed-cells cotangent) is the (A,A)ᵀ block's DISCARD: that pullback
        # is the explicit A_ABᵀ grid block
        # (``RadialCharacteristicSeeding.apply_transpose``), never this
        # walk's; non-carrying levels were scattered onto ``psi_ang_bar``
        # inside the closure.
        # Cartesian carries no angular thread (the Cartesian arm accumulates
        # no ``numer_bar``, and IdentityAngularClosure.angular_adjoint is the
        # structural zero-map) — the delegated reversal is curvilinear-only,
        # which also keeps the scalar ``psi_ang_bar`` off the moment-tailed
        # LD buffer (curvilinear ⟹ DD ⟹ no tail).
        if not is_cartesian:
            psi_ang_bar, _seed_cells_bar_discarded = closure.angular_adjoint(
                tuple(numer_bar),
            )
            psi_bar += psi_ang_bar

        # ── assemble the typed composite ──
        m_boundary = AngularBoundarySourceSink.zeros(sn_mesh.angular_trace)
        m_boundary.face_view("xmax")[...] = fo_bar
        if has_inner_face:
            m_boundary.face_view("xmin")[...] = fi_bar
        return FullField(
            interior=AngularSourceSink(
                values=psi_bar.swapaxes(0, 1),
                space=sn_mesh.angular_trial_space,
            ),
            boundary=m_boundary,
        )

    def _ensure_geom_cache(self) -> StreamingCoefficientCache:
        """The interned Stratum-1 table for THIS walk's handed closure.

        Lazily resolved on first need through the strategy layer's
        intern (:func:`geometry_cache_for`) — the retired mesh-attr
        ``_geom_cache`` memo's successor (P4.9b step 2c; the memo-
        retirement gate pins its absence).
        """
        return geometry_cache_for(self.mesh, self.angular_closure)

    def _ensure_pole_mirror(self) -> np.ndarray:
        r"""The r = 0 coupled-pole mirror pairing, derived on first use.

        The Carlson coupled-pole continuation :math:`\psi(0, +\mu) =
        \psi(0, -\mu)` IS the r = 0 quotient's :math:`\sigma_x` deck
        transformation, so its ordinate pairing is derived from the mirror
        MOTION through the one source the boundary tier also reads
        (:meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`,
        G6.3 step 7d — until then these sites read the precomputed
        ``reflection_index`` table, a second path for the same concept).
        Derived ONCE and stashed on the mesh, the same idiom as
        :meth:`_ensure_geom_cache`: the sweep bodies consume it
        10²–10⁴ times per solve, and the O(N²) match is a
        construction-time cost, not a per-iteration one.

        Raises
        ------
        ValueError
            When the quadrature is not closed under the x-mirror (no
            bijective, weight-preserving match — e.g. an odd-``n_phi``
            product rule): the pole continuation is then unrealizable,
            and refusing HERE — at first use, before any march — replaces
            the retired table's lookup miss with the pairing's own
            diagnosis.
        """
        mirror = getattr(self.mesh, "_pole_mirror_cache", None)
        if mirror is None:
            pi = self.mesh.quad.ordinate_permutation(
                SelfPairedDeck.mirror(axis="x").motion
            )
            if pi is None:
                raise ValueError(
                    "the coupled-pole continuation ψ(0,+μ) = ψ(0,−μ) "
                    "needs the x-mirror ordinate pairing, and this "
                    "quadrature is not closed under the x-mirror (no "
                    "bijective, weight-preserving match of the ordinates "
                    "onto their mirror images — e.g. an odd-n_phi product "
                    "rule). A curvilinear sweep cannot seed the r = 0 "
                    "pole on it."
                )
            mirror = pi.indices
            self.mesh._pole_mirror_cache = mirror  # type: ignore[attr-defined]
        return mirror

    def _ensure_coll_cache(
        self,
        sig_t: np.ndarray,
        geom: StreamingCoefficientCache,
    ) -> CollisionCache:
        """Return the collision cache, building it on first use if absent.

        The expected invariant (per cache-invariance test #4) is that the
        cache is constructed by :class:`SNSolver.__init__` and consumed by
        every sweep without rebuild.  Ad-hoc test callers may bypass the
        solver — in that case the cache is built lazily here.

        No bridge needed under PR-INDEX-3: ``sig_t`` arrives as principled
        ``(ng, nx, ny=1)`` and the cache consumes ``(ng, nx)`` — a single
        slice on the degenerate ``ny`` axis suffices.
        """
        cache = getattr(self.mesh, "_coll_cache", None)
        if cache is None:
            # 1-D meshes: sig_t is the principled (ng, nx) layout the cache
            # expects natively (rank-d (N, ng, *spatial); no phantom ny axis).
            sig_t_1d = sig_t  # (ng, nx)
            cache = CollisionCache.from_geometry(
                geom, sig_t_1d, self.spatial_closure, self.angular_closure
            )
            self.mesh._coll_cache = cache  # type: ignore[attr-defined]
        return cache

    def _run(
        self,
        Q: np.ndarray,
        sig_t: np.ndarray,
        boundary_flux: "AngularBoundaryFlux",
        geom: StreamingCoefficientCache,
        coll: CollisionCache,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inner body of the unified 1-D sweep.

        Internal arrays carry the principled ``(N, ng, nx, ny=1)`` layout
        (energy ``g`` is the *second* axis, NOT trailing; see
        :ref:`theory-sn-index-convention`), so no entry/exit transposes are
        needed — caller-side principled-layout inputs flow directly through the
        body.  :class:`CollisionCache` fields are ``(N, ng, nx)`` natively;
        :class:`StreamingCoefficientCache` stays on ``(N, nx)`` / ``(N,)`` (no group
        axis).

        Splits cleanly into setup (BC inflow, source pre-scale, Carlson
        seed when curvilinear) and a per-direction or per-ordinate scan:

        * **SLAB** (joint-batch): ordinates within a chain direction are
          independent (no M-M angular thread), so one
          :func:`ordinate_scan` call per chain handles the entire chain's
          ordinates at once with shape ``(nx, K, ng)`` where ``K`` is the
          number of ordinates in that direction (``N/2`` for symmetric GL).
          Exactly 2 scan calls per sweep regardless of ``N`` or ``ng``.

        * **CURVILINEAR** (sphere/cylinder, per-ordinate): the M-M angular
          thread couples ordinates sequentially within a μ-level (the
          Morel--Montry weighted recurrence reads ``psi_angle[chain]``
          updated by the *previous* ordinate in the level).  One ``ordinate_scan`` per ordinate per level — unchanged
          from PR-INDEX-1's pre-state.  A future parallel-prefix
          reformulation of the M-M recurrence could unlock joint-batch for
          curvilinear too (research-level; deferred per plan §7).
        """
        quad = self.mesh.quad
        N = quad.N
        nx = self.mesh.nx
        ng = Q.shape[1]                                          # (N, ng, nx, ny=1)
        weights = quad.weights
        mu = quad.mu_x

        # ── #282 route (a) → step 6: the starting-direction contract ──
        # This walk is the ray-DECOUPLED ``(L+C)`` diagonal-block solve on
        # a carrying mesh (the M-M thread starts at ZERO — the leg the M
        # grid's block substitution consumes); the JOINT march M⁻¹ (System
        # B solved up front from the TRUE q½ source) is the within-group
        # grid's ``CoupledOperator.solve`` substitution, never an in-walk
        # engine.
        seed_levels = frozenset(self.mesh.radial_characteristic_levels)

        # ── Entry layout — the public contract is the principled
        # (N, ng, *spatial) = (N, ng, nx) for 1-D (no phantom ny axis).
        Q_per_ord = Q                                            # (N, ng, nx)
        sig_t_p = sig_t                                          # (ng, nx)
        V = self.mesh.volumes                                      # (nx,) — no group axis
        scheme = self.spatial_closure
        # The angular-closure block, read from the walk's own handed
        # closure (P4b: the geometry table sheds the closure copies —
        # one durable home, the closure's read-only per-ordinate cache).
        angular = self.angular_closure
        c_in_per_ordinate = angular.c_in_per_ordinate            # (N,)
        c_out_per_ordinate = angular.c_out_per_ordinate          # (N,)
        tau_inv_per_ordinate = angular.tau_inv_per_ordinate      # (N,)
        march_a_in_per_ordinate = angular.march_a_in_coeff_per_ordinate  # (N,)

        coord = self.mesh.coord
        is_slab = coord is CoordSystem.CARTESIAN
        is_sphere = coord is CoordSystem.SPHERICAL

        # ── Spatial-moment width (the unified moment matvec, #240 D5b-S3;
        # convention: docs/theory/methods/sn/cartesian_multid.rst
        # §ld-ubld-unified-moment-matvec) ──
        # A multi-moment closure (LD, ``per_axis > 1``) carries a trailing 2^d
        # spatial-moment axis on the iterate / source / output so the within-cell
        # slope iterate ``φ̂`` travels between sweeps.  DD/Step (``per_axis == 1``)
        # → ``()`` tail, every buffer + every recurrence byte-identical (the
        # negative control).  Single source via ``face_moment_tail``.
        per_axis = scheme.spatial_basis_per_axis
        moment_tail = face_moment_tail(cell_moment_count(per_axis, self.mesh.ndim))
        is_moment = moment_tail != ()

        # A multi-moment closure lifts a FLAT scalar source onto the average
        # moment (slope 0), exactly as the DAG's ``_ubld_system`` does (#240
        # D5b-S3): the scattering-slope source ``Σ_s·φ̂`` arrives as a genuine
        # ``(N, ng, nx, 2^d)`` moment source, but an external / flat source (the
        # two-paths oracle, a manufactured Q̄) is rank-3 and lifts here so the
        # scan and the DAG consume the SAME source convention.  Discriminated by
        # RANK (a genuine moment source carries exactly ONE extra axis), the same
        # contract the DAG kernel uses.
        if is_moment and Q_per_ord.ndim == 3:
            lifted = np.zeros((N, ng, nx, *moment_tail))
            lifted[..., AVERAGE_MOMENT] = Q_per_ord
            Q_per_ord = lifted

        # ── Common pre-scale ──────────────────────────────────────────────
        # R-1 Step 4 A1 — single per-ordinate source.  The producer applied
        # ``1/W`` already; the sweep multiplies by cell volume V only.
        # No iso/aniso distinction internally — every WDD recurrence
        # consumes the same ``QV_per_ord``.  When the source carries the
        # spatial-moment axis (LD's ``[Q̄, Q̂]``), V broadcasts over it (each
        # moment scaled by the cell volume — the ×V source-moment convention
        # ``s_bar = Q̄·V`` / ``s_hat = Q̂·V`` the d=1 closed form consumes).
        V_b = V[None, None, :, None] if is_moment else V[None, None, :]
        QV_per_ord = Q_per_ord * V_b                             # (N, ng, nx[, 2^d])

        # Internal principled layout — angular flux (N, ng, nx[, 2^d]),
        # scalar flux (ng, nx[, 2^d]) working buffer (ny added at return).
        angular_flux = np.zeros((N, ng, nx, *moment_tail))
        scalar_flux = np.zeros((ng, nx, *moment_tail))

        # ── BC inflow + per-level Carlson seed (curvilinear only) ─────────
        #
        # Wave O (#208) O.4a.2 — BARE SWEEP: the entry ``bc_*.apply`` is GONE.
        # The reflective coupling ``ψ.inflow = B·ψ.outflow`` is no longer
        # re-applied inside the sweep; it is supplied by the CALLER as the
        # ``−B`` source term (the SI driver folds ``S + B`` into the source;
        # the direct fixed-source loops + the final reconstruction reflect the
        # persisted outflow into the inflow slots via ``SNBoundaryOperator``
        # before each sweep — see ``solver.py``). The sweep now reads the
        # SEEDED inflow trace DIRECTLY: the incoming-ordinate slots of the
        # face view ARE the inflow seed, and the outgoing-ordinate slots are
        # persisted in place after the sweep. Reading the inflow ords (before)
        # and writing the outflow ords (after) touch DISJOINT ordinate sets,
        # so aliasing the face view is safe.  (Each geometry arm below binds
        # its OWN face views — the two arms consume disjoint variable sets,
        # so no cross-arm Optionals exist; C5, 2026-07-03.)

        # ── SLAB joint-batch fast path ────────────────────────────────────
        #
        # Slab has no M-M angular thread, no degenerate ordinates, and one
        # chain per direction.  Group ordinates by chain direction and run
        # ONE ordinate_scan per chain.  Exactly 2 scan calls per sweep.
        if is_slab:
            # D-H.2-C2: L2 :class:`AngularBoundaryFlux` provides writable per-face
            # views via :meth:`face_view`.  Slab layout has both ``xmin``
            # and ``xmax`` slots (shape ``(N, ng)`` each); writes through
            # the view propagate to the flat backing buffer.  Per-cell-call
            # outflow persistence below (``xmin_face[ords] = ...``)
            # mutates these views in place.
            xmin_face = boundary_flux.face_view("xmin")   # (N, ng)
            xmax_face = boundary_flux.face_view("xmax")  # (N, ng)
            inflow_left = xmin_face    # incoming-ord slots = seeded inflow
            inflow_right = xmax_face  # incoming-ord slots = seeded inflow

            # Partition ordinates by direction sign (μ ≥ 0 → forward chain).
            forward_mask = mu >= 0
            forward_ords = np.where(forward_mask)[0]
            backward_ords = np.where(~forward_mask)[0]

            for direction_sign, ords in ((+1, forward_ords), (-1, backward_ords)):
                if ords.size == 0:
                    continue
                K = ords.size

                # Chain order is identical across ordinates in one direction
                # for slab — pick from the first ordinate.
                chain = geom.chain_idx[ords[0]]                   # (nx,)
                inv = geom.chain_idx_inv[ords[0]]                 # (nx,)

                # Per-ordinate inflow (cells degenerate, group axis full).
                psi_in_chain = (
                    inflow_left[ords] if direction_sign == +1
                    else inflow_right[ords]
                )                                                  # (K, ng)

                # Per-ordinate source in chain order — R-1 Step 4 A1's
                # single-source convention: ``QV_per_ord`` already encodes
                # per-ordinate magnitude × cell volume.  Slice the K
                # ordinates and reorder along the chain axis.
                QV_full_chain = QV_per_ord[ords][:, :, chain]      # (K, ng, nx[, 2^d])

                # Cache fields are (N, ng, nx) natively under PR-INDEX-2.
                # Indexed slice [ords] yields (K, ng, nx) — no transpose.
                inv_denom_chain = coll.inverse_denom[ords]         # (K, ng, nx)
                a_atten_chain = coll.a_attenuation[ords]           # (K, ng, nx)
                w_chain = coll.face_blend_weight[ords]             # (K, ng, nx)

                if not is_moment:
                    # ── Slopeless (DD/Step) flat-source scan — byte-identical ──
                    # Affine source emission b = QV·inverse_denom/w (#158
                    # coefficient model — DD's 2·QV·inv is the w=½ case).
                    # (K, ng, nx); ordinate_scan wants the cell axis leading.
                    b_chain = self.spatial_closure.source_emission(
                        QV_full_chain, inv_denom_chain, w_chain,
                    )
                    a_scan = np.transpose(a_atten_chain, (2, 0, 1))   # (nx, K, ng)
                    b_scan = np.transpose(b_chain, (2, 0, 1))         # (nx, K, ng)
                    w_scan = np.transpose(w_chain, (2, 0, 1))         # (nx, K, ng)

                    # ONE scan call per chain — joint-batched over (K, ng).
                    psi_face_chain_scan = ordinate_scan(
                        a_scan, b_scan, psi_in_chain,
                    )                                                  # (nx, K, ng)

                    # Spatial closure ψ̄ = (1−w)ψ_in + w·ψ_out (DD's ½-mean is
                    # w=½) — face-in shifts upstream by 1.
                    psi_face_in_chain = np.empty_like(psi_face_chain_scan)
                    psi_face_in_chain[0] = psi_in_chain
                    psi_face_in_chain[1:] = psi_face_chain_scan[:-1]
                    psi_avg_scan = self.spatial_closure.cell_average(
                        psi_face_in_chain, psi_face_chain_scan, w_scan,
                    )
                    # (nx, K, ng) → per-ordinate (ng, nx) via reorder.
                    psi_avg_per_ord = np.transpose(psi_avg_scan, (1, 2, 0))  # (K,ng,nx)
                    # Scatter back to cell-index order + write angular_flux,
                    # accumulate scalar_flux.
                    psi_avg_cell_order = psi_avg_per_ord[:, :, inv]   # (K, ng, nx)
                    angular_flux[ords, :, :] = psi_avg_cell_order
                    w_ords = weights[ords]                            # (K,)
                    scalar_flux += np.einsum(
                        "k,kgx->gx", w_ords, psi_avg_cell_order,
                    )
                    psi_face_out = psi_face_chain_scan[-1]            # (K, ng)
                else:
                    # ── Multi-moment (LD) slope-source scan ─────────────────
                    # The full LD with the threaded scattering-slope source
                    # ``Σ_s·φ̂`` (#240 D5b-S3 OWED-2): the scan propagates the
                    # downstream FACE (scalar — the d=1 face is 2^{d-1}=1) along
                    # the chain with a slope-augmented affine source, then
                    # reconstructs the (ψ̄, ψ̂) cell moments per cell — both
                    # single-sourced through the d=1 closed form (Pattern 2).
                    #
                    # FRAME (the diffusion-limit root cause, ERR-061): the cell
                    # kernel works in the per-ordinate SWEEP frame, but the
                    # iterate ``φ̂`` + the scattering source ``Σ_s·φ̂`` live in the
                    # GLOBAL frame, so for a BACKWARD sweep the slope moment must
                    # be re-signed — global→sweep IN on the source moments,
                    # sweep→global OUT on the (ψ̄, ψ̂) result — exactly the SAME
                    # single-source binding ``_CellSolve``/``_apply_walk`` ride
                    # at d≥1.  The scalar OUTGOING FACE stays sweep-frame (it
                    # propagates along the chain, never crossing into the iterate).
                    frame_signs = frame_signs_for(scheme, (direction_sign,))  # (2^d,)
                    # Inside the multi-moment (LD) branch: the source ``QV`` and
                    # the reconstructed ``(ψ̄, ψ̂)`` are genuine moment buffers
                    # (``is_moment`` ≡ ``scheme.is_multi_moment``).
                    # Source moments in chain order; global→sweep IN.
                    QV_chain_sweep = _reframe(
                        QV_full_chain, frame_signs, is_moment_valued=is_moment,
                    )
                    s_bar = QV_chain_sweep[..., AVERAGE_MOMENT]        # (K, ng, nx)
                    s_hat = QV_chain_sweep[..., 1]                     # (K, ng, nx)

                    # The per-cell d=1 closed form — the ONE LD algebra handle
                    # (slope fold shared with the matvec/per-cell Schur).
                    abs_mu_c = geom.abs_mu[ords][:, None, None]        # (K, 1, 1)
                    A_down_c = geom.face_area_downstream[ords][:, None, :]           # (K, 1, nx)
                    V_c = geom.volume[ords][:, None, :]                # (K, 1, nx)
                    # Σ_t chain-ordered (ng, nx) → broadcast (1, ng, nx) over K.
                    sig_t_chain = sig_t_p[:, chain][None, :, :]        # (1, ng, nx)
                    cf = scheme.moment_scan_closure(
                        abs_mu=abs_mu_c, face_area_downstream=A_down_c,
                        volume=V_c,
                        reaction_xs=sig_t_chain,
                    )
                    # Face-chain affine source b = flat emission + slope term.
                    b_chain = (
                        scheme.source_emission(s_bar, inv_denom_chain, w_chain)
                        + cf.scan_slope_face_source(
                            V_c, s_hat, inv_denom_chain, w_chain,
                        )
                    )                                                  # (K, ng, nx)
                    a_scan = np.transpose(a_atten_chain, (2, 0, 1))   # (nx, K, ng)
                    b_scan = np.transpose(b_chain, (2, 0, 1))         # (nx, K, ng)

                    # ONE scan call per chain — the scalar downstream FACE.
                    psi_face_chain_scan = ordinate_scan(
                        a_scan, b_scan, psi_in_chain,
                    )                                                  # (nx, K, ng)
                    psi_face_in_chain = np.empty_like(psi_face_chain_scan)
                    psi_face_in_chain[0] = psi_in_chain
                    psi_face_in_chain[1:] = psi_face_chain_scan[:-1]
                    # (nx, K, ng) → (K, ng, nx): the per-cell upstream face.
                    psi_in_cell = np.transpose(psi_face_in_chain, (1, 2, 0))

                    # Reconstruct the (ψ̄, ψ̂) cell moments — the scan twin of the
                    # per-cell Schur (sweep frame).
                    psi_bar, psi_hat = cf.scan_reconstruct(
                        V_c, s_bar, s_hat, psi_in_cell,
                    )                                                  # (K, ng, nx)
                    mom_sweep = np.stack([psi_bar, psi_hat], axis=-1)  # (K,ng,nx,2)
                    # sweep→global OUT on the moment (the involution's inverse =
                    # itself).
                    mom_global = _reframe(
                        mom_sweep, frame_signs, is_moment_valued=is_moment,
                    )
                    # Scatter back to cell-index order; write moment angular flux,
                    # accumulate the moment scalar flux φ̂ = Σ_n w_n ψ̂_n.
                    mom_cell_order = mom_global[:, :, inv, :]          # (K,ng,nx,2)
                    angular_flux[ords] = mom_cell_order
                    w_ords = weights[ords]                            # (K,)
                    scalar_flux += np.einsum(
                        "k,kgxp->gxp", w_ords, mom_cell_order,
                    )
                    psi_face_out = psi_face_chain_scan[-1]            # (K, ng) scalar

                # Persist outflow at the appropriate boundary face — the
                # last chain output is the (scalar) outgoing-face flux on that
                # side (the d=1 face cochain is moment-free for both closures).
                if direction_sign == +1:
                    xmax_face[ords] = psi_face_out             # (K, ng)
                else:
                    xmin_face[ords] = psi_face_out             # (K, ng)

        # ── CURVILINEAR per-ordinate path ─────────────────────────────────
        #
        # M-M angular thread couples ordinates sequentially within a level
        # (psi_angle[chain] is updated by ordinate m → consumed by m+1).
        # Joint-batch over ordinates is blocked; loop stays per-ordinate.
        else:
            # D-H.2-C2: 1-D curvilinear layout has only the outer radial
            # ``xmax`` face (the geometric pole at r=0 is a regularity
            # condition, not a BC face).  Writable view into the L2 flat
            # backing buffer.
            bc_outer = boundary_flux.face_view("xmax")  # (N, ng)
            inflow_full = bc_outer  # incoming-ord slots = seeded inflow (bare sweep)
            sigma_t_gx = sig_t_p                                  # (ng, nx)
            if is_sphere:
                # Sphere is the single-level case with NO μ-level index
                # (``mu_level_idx=None`` on the walk below).
                levels: "list[int | None]" = [None]
                level_ordinates_list = [list(range(N))]
            else:
                level_indices = quad.level_indices
                levels = list(range(len(level_indices)))
                level_ordinates_list = [list(li) for li in level_indices]

            # The whole curvilinear scan IS the Morel–Montry thread (the
            # per-level ψ½ seed + the in-sweep angular recurrence), so
            # the closure is parsed to the M-M type loudly — a different
            # curvilinear closure would need its own scan choreography.
            #
            # Per-level seed dispatch — the iterate plays NO role: every
            # ADMITTED curvilinear level is CARRYING (R12a; the Q5.6.3
            # cylindrical admission refuses non-carrying rules at SNMesh
            # construction), and the ψ½ legs are solved DIRECTLY, up
            # front (before the level loop), by System B's NAMED
            # resolvent ``A_BB.solve`` — the Hébert (3.434)-(3.435) DD
            # march on the TRUE q½ source, inward from the seeded inflow
            # corner, pole-continued, outward to the outflow corner
            # (4e-e2: the walk routes THROUGH
            # ``RadialCharacteristicOperator``) — and the marched inward
            # cells ARE the recurrence seed.  So the curvilinear
            # (L+C).solve is a direct inverse for every geometry: the
            # sphere via #282 route (a), the folded cylinder via the
            # same route on its multi-level carrier.
            #
            # HISTORY: until Q5.6.3, non-carrying NODE_ALIGNED-product
            # cylinders were admitted and their t = 0 self-referential
            # seed (ψ½ ≡ ψ̄_{m0}) was handled by the #280 2.5b
            # "direct-seed fold" (c_out → c_out − c_in on the m0
            # diagonal).  The admission flip made that configuration
            # unconstructible, and the fold was RETIRED with its family
            # (Q5.6.3 leg 5, user ruling on gate_design §6 Q3).
            from ..angular.closure import MorelMontryAngularSweep

            closure = self.angular_closure
            if not isinstance(closure, MorelMontryAngularSweep):
                raise TypeError(
                    "_OneDimScanWalk curvilinear scan requires the "
                    "Morel-Montry closure (its Carlson coupled-pole seed "
                    f"thread); got {type(closure).__name__}."
                )
            # Carlson coupled-pole spatial seed (ERR-058, Issue #195): each
            # inward (μ<0) ordinate's pole-face outflow is captured here and
            # consumed as the spatial seed of its MIRROR outward (μ>0)
            # ordinate — the r=0 continuity ψ(0, +μ) = ψ(0, −μ).  Mirror
            # partners share a level, and the M-M thread sweeps inward
            # ordinates first, so the captured value is always data.
            mirror = self._ensure_pole_mirror()
            pole_outflow = np.zeros((mu.size, ng))

            for p_idx, level in enumerate(levels):
                ordinates_in_level = level_ordinates_list[p_idx]
                ords_arr = np.asarray(ordinates_in_level)
                # ── the ZERO thread (step 6 — the walk IS the (A,A) block) ──
                # ⛔ Until Q5.6.3 a NON-carrying level (ψ½ ≡ ψ̄_{m0} —
                # product, t = 0) was resolved by the #280 2.5b diagonal
                # fold when m0 was swept: no pre-loop seed, no iterate
                # read, ``psi_angle`` the M-M thread buffer that m0
                # (swept first) filled with its own average before any
                # downstream ordinate read it.  The admission flip made
                # that level unconstructible and the fold was retired
                # with it (the HISTORY note above), so EVERY level
                # reaching this loop is CARRYING: the
                # zero thread IS the ray-decoupled (L+C) closure — no fold
                # entry exists for carrying levels, so the recurrence
                # starts at exactly ψ½ = 0 (the LC diagonal-block
                # semantics); the joint march (A_BB solved up front, the
                # marched cells read as the seed) is the M grid's
                # substitution, never an in-walk engine.
                psi_angle = np.zeros((ng, nx))                    # (ng, nx)

                for m_local, global_n in enumerate(ordinates_in_level):
                    mu_n = mu[global_n]
                    w_n = weights[global_n]
                    chain = geom.chain_idx[global_n]

                    # Per-ordinate source assembly (R-1 Step 4 A1):
                    # ``QV_per_ord[global_n]`` is the per-ordinate source ×
                    # cell volume for ordinate ``global_n``, shape (ng, nx).
                    QV_full = QV_per_ord[global_n]                  # (ng, nx)
                    QV_chain = QV_full[:, chain]                    # (ng, nx)

                    # Per-ordinate spatial-upstream inflow (ng,).
                    if mu_n < 0:
                        psi_in = inflow_full[global_n]
                    elif geom.is_degenerate[global_n]:
                        # Degenerate (μ_r = 0) ordinate: no radial streaming —
                        # the spatial-upstream slot is inert.
                        psi_in = np.zeros(ng)
                    else:
                        # Coupled-pole seed (ERR-058 a): the mirror inward
                        # ordinate's pole-face outflow (captured below in
                        # this level's earlier M-M steps) — the r = 0
                        # continuity ψ(0, +μ) = ψ(0, −μ).  Mirrors the
                        # matvec's seed (Pattern 2 — the sweep/matvec pair
                        # stays ONE discrete system).  The historical
                        # pole-CELL-centre read of the previous iterate was
                        # O(h)-wrong on non-flat profiles (exact on flat ψ —
                        # invisible to every flat-flux gate).
                        psi_in = pole_outflow[mirror[global_n]]

                    # Degenerate cyl-axis ordinate: slow per-cell path.
                    if geom.is_degenerate[global_n]:
                        ordinate_idx = global_n if is_sphere else m_local
                        visits = list(self.mesh.dag_walk(
                            ordinate_idx=ordinate_idx,
                            mu_level_idx=level,
                        ))
                        # P4.9a: the walk (L3, the composition site)
                        # assembles the closure's balance contributions
                        # from the MINTED constants and applies the
                        # owner's march itself — the scheme closes the
                        # SPATIAL axis only.  ``geom.c_*`` are the
                        # closure's own per-ordinate arrays (stored
                        # unchanged by the cache), so the values are
                        # bit-identical to the retired visit stamp.
                        c_in_n = c_in_per_ordinate[global_n]
                        c_out_n = c_out_per_ordinate[global_n]
                        # ΔA/w from its two factors (P4.7 — the packet no
                        # longer carries the fusion): same operands and op
                        # as the retired per-packet copy, bit-identical.
                        reduced_op = self.mesh.reduced
                        assert reduced_op is not None  # curvilinear => minted
                        w_n = float(np.asarray(
                            self.mesh.quad.weights)[global_n])
                        for visit in visits:
                            i = visit.cell_idx
                            dAw_vi = float(
                                np.asarray(reduced_op.delta_A)[i]) / w_n
                            # scheme.update expects per-cell (ng,)
                            # arrays — sig_t / source slice on the cell axis.
                            result = scheme.update(
                                visit=visit,
                                total_xs=sig_t_p[:, i],
                                source=QV_full[:, i],
                                upstream_state=UpstreamState(
                                    spatial_upstream=psi_in,
                                ),
                                angular_denom_term=(
                                    dAw_vi * c_out_n
                                ),
                                angular_numer_upstream=(
                                    dAw_vi * c_in_n
                                    * psi_angle[:, i]
                                ),
                            )
                            psi = result.cell_average_flux           # (ng,)
                            # The angular axis is closed by ITS closure;
                            # the τ index is the GLOBAL ordinate, exactly
                            # what the retired visit stamp used.
                            psi_angle[:, i] = closure.advance_psi_half(
                                psi, psi_angle[:, i], ordinate=global_n,
                            )
                            angular_flux[global_n, :, i] = psi
                            scalar_flux[:, i] += w_n * psi
                        continue

                    # Non-degenerate fast path: per-ordinate scan (ng, nx).
                    # psi_angle on (ng, nx); chain reorders the nx axis.
                    psi_a_in_chain = psi_angle[:, chain].copy()      # (ng, nx)
                    ang_contrib = (
                        geom.delta_A_over_w[global_n] * c_in_per_ordinate[global_n]
                    )[None, :] * psi_a_in_chain                       # (ng, nx)
                    # Cache fields are (N, ng, nx) natively under PR-INDEX-2.
                    # Indexed slice [global_n] yields (ng, nx) — no transpose.
                    inv_denom_p = coll.inverse_denom[global_n]       # (ng, nx)
                    a_atten_p = coll.a_attenuation[global_n]         # (ng, nx)
                    w_p = coll.face_blend_weight[global_n]           # (ng, nx)
                    # Affine source emission b = (QV + angular)·inverse_denom/w
                    # (#158 coefficient model — the Morel–Montry angular
                    # redistribution rides the volumetric source; DD's
                    # 2·(QV+ang)·inv is the w=½ case).
                    b = self.spatial_closure.source_emission(
                        QV_chain + ang_contrib, inv_denom_p, w_p,
                    )  # (ng, nx)

                    # ordinate_scan: leading axis is the scan/cell axis.
                    # Pass (nx, ng) — transpose from (ng, nx).
                    psi_face_chain = ordinate_scan(
                        a_atten_p.T, b.T, psi_in,
                    )                                                 # (nx, ng)
                    if mu_n < 0:
                        # Coupled-pole capture: an inward chain ends at the
                        # pole; its final face value seeds the mirror outward
                        # ordinate's chain (consumed above).
                        pole_outflow[global_n] = psi_face_chain[-1]

                    # Spatial closure ψ̄ = (1−w)ψ_in + w·ψ_out (w_p.T → (nx, ng)
                    # to match the scan layout) — vectorised cell-average.
                    psi_face_in_chain = np.empty_like(psi_face_chain)
                    psi_face_in_chain[0] = psi_in
                    psi_face_in_chain[1:] = psi_face_chain[:-1]
                    psi_avg_chain = self.spatial_closure.cell_average(
                        psi_face_in_chain, psi_face_chain, w_p.T,
                    )
                    # Principled view: (ng, nx).
                    psi_avg_chain_p = psi_avg_chain.T                # (ng, nx)

                    # M-M angular thread (curvilinear-only): every ordinate
                    # reads the previous ordinate's outgoing edge threaded
                    # through ``psi_angle`` (the level's first ordinate reads
                    # the route-(a) marched ψ½ seed placed on the thread).
                    psi_angle_out_chain_p = (
                        tau_inv_per_ordinate[global_n] * psi_avg_chain_p
                        - march_a_in_per_ordinate[global_n] * psi_a_in_chain
                    )                                                 # (ng, nx)
                    psi_angle[:, chain] = psi_angle_out_chain_p

                    # Scatter back to cell-index order + writes.
                    inv = geom.chain_idx_inv[global_n]
                    psi_avg_p = psi_avg_chain_p[:, inv]              # (ng, nx)
                    angular_flux[global_n, :, :] = psi_avg_p
                    scalar_flux += w_n * psi_avg_p

                    # Persist outflow at the outer face for outward ordinates.
                    if mu_n >= 0 and abs(mu_n) >= self.mesh._DEGENERATE_ABS_ETA_THRESHOLD:
                        bc_outer[global_n] = psi_face_chain[-1]      # (ng,)

        # ── Exit — PR-INDEX-5: caller consumes principled layout ──────────
        # NO iteration-cache write-back: the sweep is a stateless EXACT
        # inverse.  Any warm start (the previous iterate) lives at the
        # ITERATION layer (SourceIteration / GreenOperator), never threaded
        # into this direct sweep (the vestigial seed retired, #280 2.5c).
        # angular_flux is (N, ng, nx); scalar_flux is (ng, nx) — the principled
        # (N, ng, *spatial) / (ng, *spatial) public contract (no phantom ny).
        return angular_flux, scalar_flux

    def _run_transpose(
        self,
        bulk_cot: np.ndarray,
        sigma: np.ndarray,
        boundary_cot: "BoundaryField",
        geom: StreamingCoefficientCache,
        coll: CollisionCache,
    ) -> "tuple[np.ndarray, AngularBoundarySourceSink]":
        r"""The REVERSE-SCAN — :math:`(L+C)^{-\mathsf T}` on the composite cotangent.

        The Euclidean transpose-solve (#280 Phase 2.5b): the reverse-mode
        adjoint of :meth:`_run` (which realises :math:`(L+C)^{-1}`).  By
        linearity the adjoint of the solve IS the transpose-solve
        (:math:`(M^{-1})^{\mathsf T} = M^{-\mathsf T}`), so every linear step
        of :meth:`_run` transposes in reverse program order — the affine
        :func:`ordinate_scan` becoming the REVERSE
        :func:`ordinate_scan_transpose` (coherent with ``_run``'s Blelloch
        scan; never a reverse loop bolted onto :meth:`loss_action_transpose`).

        The augmented :math:`(L+C)` is block-lower-triangular
        ``[[A_ss, 0], [A_bs, A_bb]]`` in ``[seed, bulk]`` order (the #282
        route-(a) certificate), so its transpose is block-UPPER-triangular
        and the transpose-solve marches **bulk-first (the reverse scan) then
        seed** — the reverse of ``_run``'s seed-first march.

        Consumes the SAME ``geom`` / ``coll`` caches ``_run`` does (the
        ``×V`` ``affine_scan`` coefficient form — NOT the matvec's ``÷V``
        ``residual_kernel_batch``; the #242 two-denom seam, pinned by G1).
        Returns ``(Q_bar, m_boundary)`` — the bulk source cotangent + the
        trace cotangent.  On a carrying mesh this is the ray-decoupled
        ``(L+C)⁻ᵀ`` block (step 6): the M-M thread cotangent is DISCARDED
        (the zero thread is a FIXED input of the decoupled map, so its
        cotangent propagates nowhere); the joint ``M⁻ᵀ`` is the M grid's
        transposed substitution.  Two slab arms since #310 C2 (mirroring
        ``_run``): DD/Step take the slopeless reverse-scan; LD takes the
        moment arm — the reverse of ``_run``'s slope-source scan
        (``scan_reconstruct_transpose`` + the diagonal self-transposes of
        the face-source folds), gated by the SAME derived
        ``has_transpose_kernel`` trait the reverse walk checks.
        """
        from orpheus.transport.source_sinks import AngularBoundarySourceSink

        quad = self.mesh.quad
        N = quad.N
        nx = self.mesh.nx
        ng = bulk_cot.shape[1]
        mu = quad.mu_x
        scheme = self.spatial_closure
        # The angular-closure block, read from the walk's own handed
        # closure (P4b: the geometry table sheds the closure copies —
        # one durable home, the closure's read-only per-ordinate cache).
        angular = self.angular_closure
        c_in_per_ordinate = angular.c_in_per_ordinate            # (N,)
        c_out_per_ordinate = angular.c_out_per_ordinate          # (N,)
        tau_inv_per_ordinate = angular.tau_inv_per_ordinate      # (N,)
        march_a_in_per_ordinate = angular.march_a_in_coeff_per_ordinate  # (N,)

        # ── the R12a starting-direction contract (mirror _run, step 6) ──
        # This is the ray-DECOUPLED ``(L+C)⁻ᵀ`` diagonal-block transpose on
        # a carrying mesh — the M-M thread cotangent is DISCARDED (the zero
        # thread is a FIXED input of the decoupled map, so its cotangent
        # propagates nowhere); the joint ``M⁻ᵀ`` is the M grid's transposed
        # substitution.
        seed_levels = frozenset(self.mesh.radial_characteristic_levels)

        # ── The trait guard (#310 C2 — the R5/R6 two-guard lift, unified):
        # the reverse-scan is available exactly when the scheme registers
        # its transpose kernels — the SAME derived trait the reverse walk
        # checks.  The old moment-count probe is GONE: a spatial-moment
        # tail no longer means "deferred", it means the LD moment arm
        # below (the reverse of ``_run``'s slope-source scan).
        if not type(scheme).has_transpose_kernel:
            raise NotImplementedError(
                "_OneDimScanWalk._run_transpose: scheme "
                f"{type(scheme).__name__} registers no transpose kernel "
                "pair (residual_kernel_batch_transpose; plus "
                "streaming_cell_transpose for a curvilinear scheme) — the "
                "reverse-scan is a typed deferral (#310)."
            )
        per_axis = scheme.spatial_basis_per_axis
        moment_tail = face_moment_tail(cell_moment_count(per_axis, self.mesh.ndim))
        is_moment = moment_tail != ()
        if bulk_cot.shape[3:] != tuple(moment_tail):
            # Pattern-4 backstop (mirror of loss_action_transpose's): a
            # tail-mismatched cotangent would broadcast silently.
            raise ValueError(
                "_OneDimScanWalk._run_transpose: bulk cotangent shape "
                f"{bulk_cot.shape} does not carry the scheme's "
                f"spatial-moment tail {tuple(moment_tail)} "
                f"({type(scheme).__name__})."
            )

        V = self.mesh.volumes
        coord = self.mesh.coord
        is_slab = coord is CoordSystem.CARTESIAN

        Q_bar = np.zeros((N, ng, nx, *moment_tail))
        m_boundary = AngularBoundarySourceSink.zeros(self.mesh.angular_trace)

        # ── SLAB reverse-scan (no angular thread, no seed) ────────────────
        if is_slab:
            xmin_cot = boundary_cot.face_view("xmin")            # (N, ng)
            xmax_cot = boundary_cot.face_view("xmax")
            xmin_bar = m_boundary.face_view("xmin")
            xmax_bar = m_boundary.face_view("xmax")
            forward_ords = np.where(mu >= 0)[0]
            backward_ords = np.where(mu < 0)[0]

            for direction_sign, ords in ((+1, forward_ords), (-1, backward_ords)):
                if ords.size == 0:
                    continue
                chain = geom.chain_idx[ords[0]]                  # (nx,)
                inv = geom.chain_idx_inv[ords[0]]                # (nx,)
                inv_denom_chain = coll.inverse_denom[ords]       # (K, ng, nx)
                w_chain = coll.face_blend_weight[ords]           # (K, ng, nx)
                a_atten_chain = coll.a_attenuation[ords]         # (K, ng, nx)

                a_scan = np.transpose(a_atten_chain, (2, 0, 1))  # (nx, K, ng)
                out_cot = xmax_cot if direction_sign > 0 else xmin_cot

                if not is_moment:
                    # ── Slopeless (DD/Step) reverse-scan — byte-identical ──
                    # ψ̄ cotangent, cell → chain order (transpose of the
                    # [:,:,inv] scatter — ``_run``'s ``psi_avg_cell_order``).
                    psi_avg_per_ord_bar = bulk_cot[ords][:, :, chain]  # (K,ng,nx)
                    psi_avg_scan_bar = np.transpose(
                        psi_avg_per_ord_bar, (2, 0, 1),
                    )                                            # (nx, K, ng)
                    w_scan = np.transpose(w_chain, (2, 0, 1))    # (nx, K, ng)

                    # cell_averageᵀ: ψ̄ = (1−w)·face_in + w·face_out.
                    psi_face_in_bar = (1.0 - w_scan) * psi_avg_scan_bar
                    psi_face_bar = w_scan * psi_avg_scan_bar
                    # shiftᵀ: face_in[0]=ψ_in; face_in[i]=face_out[i−1].
                    psi_in_bar = psi_face_in_bar[0].copy()       # (K, ng)
                    psi_face_bar[:-1] += psi_face_in_bar[1:]
                    # outflow persistence: face_out[−1] is the boundary
                    # outflow slot — its cotangent enters the last face.
                    psi_face_bar[-1] += out_cot[ords]            # (K, ng)

                    # reverse the affine scan (the 2.5b keystone op).
                    b_scan_bar, psi_in_bar_scan = ordinate_scan_transpose(
                        a_scan, psi_face_bar,
                    )
                    psi_in_bar += psi_in_bar_scan                # (K, ng)

                    # b = source_emission(QV, inv, w) is diagonal → self-Tᵀ;
                    # QV = Q·V → Q_bar = QV_bar·V.
                    b_bar = np.transpose(b_scan_bar, (1, 2, 0))  # (K, ng, nx)
                    QV_bar = scheme.source_emission(
                        b_bar, inv_denom_chain, w_chain,
                    )
                    Q_bar_chain = QV_bar * V[chain][None, None, :]
                    Q_bar[ords] = Q_bar_chain[:, :, inv]         # scatter → cells
                else:
                    # ── Multi-moment (LD) reverse slope-source scan (#310
                    # C2) — the exact reverse of ``_run``'s LD branch, in
                    # reverse program order.  The moment cotangent reframes
                    # sweep-ward with the SAME involution the forward uses
                    # (conjugation commutes with transpose —
                    # derive_octant_frame_sign_is_involution).
                    frame_signs = frame_signs_for(scheme, (direction_sign,))
                    mom_bar_chain = bulk_cot[ords][:, :, chain, :]  # (K,ng,nx,2)
                    mom_bar_sweep = _reframe(
                        mom_bar_chain, frame_signs, is_moment_valued=True,
                    )
                    psi_bar_bar = mom_bar_sweep[..., AVERAGE_MOMENT]
                    psi_hat_bar = mom_bar_sweep[..., 1]          # (K, ng, nx)

                    # The SAME d=1 closed form ``_run`` builds (ONE LD
                    # algebra handle; its transpose rides _geom_fold).
                    abs_mu_c = geom.abs_mu[ords][:, None, None]  # (K, 1, 1)
                    A_down_c = geom.face_area_downstream[ords][:, None, :]     # (K, 1, nx)
                    V_c = geom.volume[ords][:, None, :]          # (K, 1, nx)
                    sig_t_chain = sigma[:, chain][None, :, :]    # (1, ng, nx)
                    cf = scheme.moment_scan_closure(
                        abs_mu=abs_mu_c, face_area_downstream=A_down_c,
                        volume=V_c,
                        reaction_xs=sig_t_chain,
                    )

                    # scan_reconstructᵀ: (ψ̄†, ψ̂†) → (s̄†, ŝ†, ψ_in-cell†).
                    s_bar_bar, s_hat_bar, psi_in_cell_bar = (
                        cf.scan_reconstruct_transpose(
                            V_c, psi_bar_bar, psi_hat_bar,
                        )
                    )                                            # (K, ng, nx)

                    # shiftᵀ: psi_in_cell[0] = ψ_in; psi_in_cell[i] =
                    # face[i−1] — plus the boundary outflow persistence on
                    # the last face (mirror of the scalar arm).
                    pic_scan = np.transpose(psi_in_cell_bar, (2, 0, 1))
                    psi_in_bar = pic_scan[0].copy()              # (K, ng)
                    psi_face_bar = np.zeros_like(pic_scan)
                    psi_face_bar[:-1] += pic_scan[1:]
                    psi_face_bar[-1] += out_cot[ords]            # (K, ng)

                    # reverse the affine scan (the SAME keystone op — the
                    # LD face chain is scalar, w-generic).
                    b_scan_bar, psi_in_bar_scan = ordinate_scan_transpose(
                        a_scan, psi_face_bar,
                    )
                    psi_in_bar += psi_in_bar_scan                # (K, ng)

                    # bᵀ: b = source_emission(s̄, inv, w) +
                    # scan_slope_face_source(V, ŝ, inv, w) — BOTH diagonal
                    # in their source moment, so each transposes as itself
                    # applied to the face-source cotangent ``b̄``.
                    b_bar = np.transpose(b_scan_bar, (1, 2, 0))  # (K, ng, nx)
                    s_bar_bar = s_bar_bar + scheme.source_emission(
                        b_bar, inv_denom_chain, w_chain,
                    )
                    s_hat_bar = s_hat_bar + cf.scan_slope_face_source(
                        V_c, b_bar, inv_denom_chain, w_chain,
                    )

                    # reframe the moment-source cotangent back (involution)
                    # + ×V (QV = Q·V ⟹ Q̄ = QV̄·V) + scatter to cell order.
                    QV_bar_sweep = np.stack([s_bar_bar, s_hat_bar], axis=-1)
                    QV_bar_chain = _reframe(
                        QV_bar_sweep, frame_signs, is_moment_valued=True,
                    )
                    Q_bar_chain = QV_bar_chain * V[chain][None, None, :, None]
                    Q_bar[ords] = Q_bar_chain[:, :, inv, :]      # scatter → cells

                # boundary inflow-slot cotangent: the identity passthrough of
                # the given inflow trace PLUS the scan-seed cotangent (the
                # inflow slot is read as ψ_in AND passed through to the output).
                in_cot = xmin_cot if direction_sign > 0 else xmax_cot
                in_bar = xmin_bar if direction_sign > 0 else xmax_bar
                in_bar[ords] = in_cot[ords] + psi_in_bar
            return Q_bar, m_boundary

        # ── CURVILINEAR reverse-scan (sphere AND cylinder both carrying) ──
        # The transpose of ``_run``'s unified curvilinear body: the per-level
        # Morel–Montry thread reversed; the CARRYING Carlson ψ½ march
        # transposed into a starting-direction cotangent (the sphere's single
        # level, and since Q5.6.3 every level of the admitted folded
        # cylinder); and the pure-azimuthal DEGENERATE ordinates as slot-local
        # diagonal transposes.
        # ⛔ Until Q5.6.3 the cylinder arm also carried a NON-carrying m0 seed
        # folded into the cell diagonal (#280 2.5b-cyl-fwd) and transposed as
        # the seed-ordinate's own-average routing (no carrier, ``m_seed =
        # None``).  The admission flip made that configuration
        # unconstructible and the fold was retired with it — see the
        # level-structure comment below and ``_run``'s HISTORY note.
        from ..angular.closure import MorelMontryAngularSweep

        is_sphere = coord is CoordSystem.SPHERICAL
        closure = self.angular_closure
        if not isinstance(closure, MorelMontryAngularSweep):
            raise TypeError(
                "_OneDimScanWalk._run_transpose curvilinear scan requires the "
                f"Morel-Montry closure; got {type(closure).__name__}."
            )
        weights = quad.weights
        sigma_gx = sigma                                         # (ng, nx)
        mirror = self._ensure_pole_mirror()
        bc_outer_cot = boundary_cot.face_view("xmax")            # (N, ng)
        bc_outer_bar = m_boundary.face_view("xmax")              # (N, ng) — written
        pole_outflow_bar = np.zeros((mu.size, ng))               # reverse coupled-pole

        # ── level structure (mirror _run) ──
        # Sphere: the single carrying level, all N ordinates in μ-increasing
        # order, ψ½ carried (route (a)).  Cylinder: multi-level, every level
        # carrying (Q5.6.3 admission) — the seed cotangents land on the
        # output composite's radial_characteristic block via the closure's
        # carrying branch.  (The retired #280 2.5b seed-fold transpose lived
        # here until Q5.6.3 leg 5 — see ``_run``'s HISTORY note.)
        if is_sphere:
            levels: "list[int | None]" = [None]
            level_ordinates_list = [list(range(N))]
        else:
            level_indices = quad.level_indices
            levels = list(range(len(level_indices)))
            level_ordinates_list = [list(li) for li in level_indices]

        for p_idx, level in enumerate(levels):
            ordinates_in_level = level_ordinates_list[p_idx]
            # ψ½ recurrence-thread cotangent, accumulated in cell order.
            psi_angle_bar = np.zeros((ng, nx))
            # ── reverse the ordinate loop (reverse-μ order) ──
            for global_n in reversed(ordinates_in_level):
                # ── degenerate pure-azimuthal ord (cylinder): slot-local ──
                # No radial streaming (μ_r=0 ⇒ face_area_downstream=0), so each cell is an
                # INDEPENDENT diagonal solve threaded by the M-M recurrence:
                #   ψ̄ = inv_denom·(QV + delta_A_over_w·c_in·ψ_ang);
                #   ψ_ang_out = tau_inv·ψ̄ − mm_a_in·ψ_ang.
                # The caches are VALID here (face_area_downstream=0 ⇒ inverse_denom =
                # 1/(delta_A_over_w·c_out + Σ_t·V); probe-confirmed 0-ULP), so there is no
                # scan and no recompute — the transpose is the diagonal's adjoint
                # in cell order (degenerate ords carry no chain, no face, no
                # pole/BC coupling; the forward's dedicated dag-walk branch).
                if not is_sphere and geom.is_degenerate[global_n]:
                    out_ang_bar = psi_angle_bar                  # (ng, nx) cell order
                    psi_avg_bar = tau_inv_per_ordinate[global_n] * out_ang_bar
                    psi_ang_bar = -march_a_in_per_ordinate[global_n] * out_ang_bar
                    # angular_flux[global_n] = ψ̄ (cell order, every cell).
                    psi_avg_bar = psi_avg_bar + bulk_cot[global_n]
                    # ψ̄ = inv_denom·(QV + |μ|·face_area_total·ψ_in + κ·ψ_ang);  QV = Q·V.
                    # (|μ|·face_area_total·ψ_in is the residual spatial coupling the
                    # forward's dag-walk carries even at face_area_downstream=0 — |μ|≈0-weighted,
                    # but its boundary cotangent must be threaded for an EXACT
                    # transpose; ψ_in is shared across cells, so its bar sums.)
                    u_bar = coll.inverse_denom[global_n] * psi_avg_bar   # (ng, nx)
                    Q_bar[global_n] += u_bar * V[None, :]
                    kappa = geom.delta_A_over_w[global_n] * c_in_per_ordinate[global_n]    # (nx,)
                    psi_ang_bar = psi_ang_bar + kappa[None, :] * u_bar
                    psi_angle_bar = psi_ang_bar                  # overwrite (cell order)
                    # Boundary: the degenerate ord does NOT overwrite its outflow
                    # slot (no face march), so the forward passes q.boundary[n] →
                    # sol.boundary[n] identically — the transpose mirrors that on
                    # the cotangent.  A μ<0 degenerate ord ALSO reads that slot as
                    # its spatial upstream (|μ|·face_area_total, summed over cells).
                    if mu[global_n] < 0:
                        spatial_sens = (
                            geom.abs_mu[global_n] * geom.face_area_total[global_n]
                        )[None, :] * u_bar                       # (ng, nx)
                        bc_outer_bar[global_n] = (
                            bc_outer_cot[global_n] + spatial_sens.sum(axis=1)
                        )
                    else:
                        bc_outer_bar[global_n] = bc_outer_cot[global_n]
                    continue

                mu_n = mu[global_n]
                chain = geom.chain_idx[global_n]
                inv = geom.chain_idx_inv[global_n]
                w_p = coll.face_blend_weight[global_n]
                # Every ordinate: cache coeffs + the M-M thread read/write
                # (the level's first ordinate reads the marched ψ½ seed
                # placed on the thread — route (a)).
                inv_denom_p = coll.inverse_denom[global_n]        # (ng, nx)
                a_atten_p = coll.a_attenuation[global_n]

                # reverse: psi_angle[:,chain] = psi_angle_out_chain_p.
                out_bar_chain = psi_angle_bar[:, chain]          # (ng, nx)
                # M-M thread: out = tau_inv·psi_avg_p − mm_a_in·psi_a_in.
                mm_bar = march_a_in_per_ordinate[global_n] * out_bar_chain
                psi_avg_chain_p_bar = tau_inv_per_ordinate[global_n] * out_bar_chain
                psi_a_in_chain_bar = -mm_bar
                # angular_flux[global_n] = psi_avg_chain_p[:, inv].
                psi_avg_chain_p_bar = (
                    psi_avg_chain_p_bar + bulk_cot[global_n][:, chain]
                )
                psi_avg_chain_bar = psi_avg_chain_p_bar.T        # (nx, ng)
                # cell_averageᵀ (w = w_p.T).
                w_pT = w_p.T                                     # (nx, ng)
                psi_face_in_chain_bar = (1.0 - w_pT) * psi_avg_chain_bar
                psi_face_chain_bar = w_pT * psi_avg_chain_bar
                # shiftᵀ.
                psi_in_bar = psi_face_in_chain_bar[0].copy()     # (ng,)
                psi_face_chain_bar[:-1] += psi_face_in_chain_bar[1:]
                # coupled-pole capture (μ<0) / outer-outflow persist (μ≥0).
                if mu_n < 0:
                    psi_face_chain_bar[-1] += pole_outflow_bar[global_n]
                else:
                    psi_face_chain_bar[-1] += bc_outer_cot[global_n]
                # reverse the per-ordinate affine scan (folded a on the seed ord).
                b_scan_bar, psi_in_bar_scan = ordinate_scan_transpose(
                    a_atten_p.T, psi_face_chain_bar,
                )
                psi_in_bar += psi_in_bar_scan
                # source_emissionᵀ (diagonal): s = QV_chain + ang_contrib.
                b_chain_bar = b_scan_bar.T                       # (ng, nx)
                s_bar = scheme.source_emission(b_chain_bar, inv_denom_p, w_p)
                # QV_chain = QV_full[:,chain]; QV_full = Q·V.
                Q_bar[global_n] += s_bar[:, inv] * V[None, :]
                # ang_contrib = (delta_A_over_w·c_in)·psi_a_in_chain.
                ang_coeff = geom.delta_A_over_w[global_n] * c_in_per_ordinate[global_n]  # (nx,)
                psi_a_in_chain_bar = (
                    psi_a_in_chain_bar + ang_coeff[None, :] * s_bar
                )
                # psi_a_in_chain = psi_angle[:,chain]; chain is a full perm,
                # so the OVERWRITE means the previous-state cotangent IS this
                # read's (scatter to cell order).
                psi_angle_bar = psi_a_in_chain_bar[:, inv]
                # inflow: μ<0 = outer trace (passthrough + ψ_in cot); μ>0 = the
                # mirror pole-continuation (reverse coupled-pole).
                if mu_n < 0:
                    bc_outer_bar[global_n] = bc_outer_cot[global_n] + psi_in_bar
                else:
                    pole_outflow_bar[mirror[global_n]] += psi_in_bar

            # The M-M thread cotangent (``psi_angle_bar``) is DISCARDED on
            # a carrying level (step 6): the zero thread was a fixed input
            # of the decoupled map, so its cotangent propagates nowhere —
            # the joint reversal (the fused A_ABᵀ feed + A_BB's
            # solve_transpose) is the M grid's transposed substitution.

        return Q_bar, m_boundary


def _sweep_scheduled(
    Q: np.ndarray,
    sig_t: np.ndarray,
    sn_mesh: "SNMesh",
    boundary_flux: "AngularBoundaryFlux",
    *,
    spatial_closure: "DiscretizationSchemeBase",
    angular_closure: "AngularClosureBase",
    schedule: "SweepSchedule",
    reflect: "Callable[[AngularBoundaryFlux, tuple[str, ...]], None] | None" = None,
    moment_frame: "FrameBase | None" = None,
    interior: "Callable" ,
) -> "tuple[np.ndarray, np.ndarray | None]":
    r"""Polymorphic schedule-driven 2-D sweep (Phase 3 sub-step 3c; S6.4(b)
    kernel-parameterized).

    ONE uniform sweep-and-reflect loop parameterized by ``schedule`` (the
    Jacobi / Gauss-Seidel splitting — there is NO ``if jacobi/gs`` branch;
    the splitting IS the schedule) **and by the representation's solve
    ``interior`` kernel** (S6.4(b) — the per-group octant frame is the shared
    :meth:`~orpheus.sn.loss_representation._OctantWalk.sweep_group`; this
    loop does not know HOW an octant's interior is traversed).  Because the
    inter-group reflect is kernel-agnostic, any representation's kernel
    composes with any schedule — e.g. the scan-march gains Gauss-Seidel for
    free.

    1. The GIVEN inflow ``boundary_flux`` (carrying ``B·ψₙ`` — the lagged
       reflection of the previous iterate, prepared by the caller) is read
       per-octant by the walk; there is no separate whole-trace seed.
    2. ``for group in schedule.groups``: walk the group's octants
       (:meth:`_OctantWalk.sweep_group` × ``interior``), sheding each octant's
       outflow into ``boundary_flux`` (the ι* absorb, per-octant during the
       walk). Then if ``reflect`` is given AND the group has reflective
       outgoing faces, apply ``reflect`` (the ``−B`` outflow→inflow
       reflection, in place, face-restricted) so a LATER group reads the
       fresh current-iterate inflow directly off ``boundary_flux`` (the
       ``(L+C−B_lower)⁻¹`` forward substitution) — no re-seed needed, the
       next walk reads the trace fresh.

    * **Jacobi** (``reflect=None``, one all-octants group) — every octant reads
      the frozen seed; the inter-group reflect never fires. This is exactly the
      bare sweep :func:`_sweep_jacobi` passes.
    * **Gauss-Seidel** (``reflect`` = the face-restricted ``−B``, one group per
      in-plane octant) — later groups see earlier groups' fresh reflected
      outflow. The SI scheduled resolvent supplies both: its ``.solve`` seeds
      ``B·ψₙ`` onto ``boundary_flux`` then calls this with the G-S schedule +
      the reflect closure. The walk's per-octant shed populates the fresh
      outflow that ``reflect`` then maps to inflow (replacing the storage-A
      ``absorb``-before-``reflect`` step).

    The converged fixed point is INVARIANT under ``schedule`` (any consistent
    splitting of ``(L+C−S−B)ψ=q`` shares ψ\*); only the SI spectral rate
    changes. NOTE (Phase 3 spike, issue #2/#215): this folds the BOUNDARY
    coupling ``B`` only — a modest reflective-SI rate gain. The dominant
    within-group SCATTERING ``c``-mode is NOT folded here (it cannot be folded
    into a directional sweep); that is consistent DSA / Krylov territory.

    Storage: the windowed walk (:meth:`SweepDependencyGraph.walk_windowed`)
    carries only the rolling ``(d-1)``-frontier cochain; the full-field walk
    (:meth:`SweepDependencyGraph.walk_full`) is retained as the bit-identity
    verification oracle (the ``window ≡ full-field`` test).  Both give the same
    converged solution — the two buffer policies are documented at
    ``docs/theory/methods/sn/loss_representation.rst §loss-rep-four``.

    Moment output: when ``moment_frame`` is given (the 2-D
    Cartesian windowed-SI path), the walk accumulates the harmonic moment tensor
    ``(L+1, 2L+1, ng, nx, ny)`` per anti-diagonal directly — the full
    per-ordinate angular OUTPUT ``(N, ng, nx, ny)`` is never materialized (the
    ~3× linear peak-memory win; the persistent SI iterate is already moments,
    5a).  Returns ``(moment_buf, None)`` — the scalar flux is :math:`\phi_0^0`
    = ``moment_buf[0, 0]`` (``Y_0^0 = 1``), read off the tensor, NOT returned
    separately (the angular-mode scalar is an independent array; ``None`` keeps
    the modes' second slot from being mistaken).  Principled-equivalence, NOT
    bit-identity: the cross-octant ``+=`` reorders the ordinate sum vs the
    post-sweep flat :attr:`~orpheus.numerics.frame.FrameBase.analysis` projection
    reduce (≤ 4 ULP de-risk).  ``moment_frame is None`` (every full-angular
    consumer — reconstruction, Krylov, 1-D) returns ``(angular_flux,
    scalar_flux)`` exactly as before.
    """
    # d-generic buffer setup (S6.4(d): the full-field SPINE routes through
    # this loop too, so the orchestrator must admit d = 1 as well as d = 2;
    # the historical ``ng, nx, ny`` unpack was a 2-D hardcode).
    ng = sig_t.shape[0]
    spatial = sig_t.shape[1:]
    N = sn_mesh.quad.N
    weights = sn_mesh.quad.weights
    # The output buffers carry the trailing 2^d spatial-moment axis at a
    # multi-moment closure (the φ̂ iterate accumulated by ``_CellSolve``; #240
    # D5b-S3) — both the angular field (FFW oracle / non-windowed) and the
    # harmonic-moment tensor (windowed production).  DD/Step (per_axis == 1) →
    # ``()`` tail, every buffer byte-identical (the negative control).
    moment_tail = face_moment_tail(
        cell_moment_count(spatial_closure.spatial_basis_per_axis, sn_mesh.ndim)
    )
    emit: "_SweepEmitAngular | _SweepEmitMoment"
    if moment_frame is None:
        emit = _SweepEmitAngular(
            weights=weights,
            angular_flux=np.zeros((N, ng, *spatial, *moment_tail)),
            scalar_flux=np.zeros((ng, *spatial, *moment_tail)),
        )
    else:
        # The moment buffer carries the frame table's (L+1, 2L+1) harmonic
        # block (table shape (N, L+1, 2L+1)); size it from there so the buffer
        # and the projection table agree by construction — and the sweep stays
        # basis-agnostic (it reads the order off the table it already consumes,
        # not a basis-specific attribute).
        n_l, n_m = moment_frame.table.shape[1:3]
        emit = _SweepEmitMoment(
            weights=weights,
            moment_buf=np.zeros((n_l, n_m, ng, *spatial, *moment_tail)),
            Y=moment_frame.table,
        )

    operands = _SolveOperands(
        Q=Q, sig_t=sig_t,
        str_axes=tuple(sn_mesh.streaming(a) for a in range(sn_mesh.ndim)),
    )
    walk = _OctantWalk(sn_mesh, spatial_closure, angular_closure)
    for group in schedule.groups:
        walk.sweep_group(
            group,
            operands=operands,
            emit=emit,
            boundary_flux=boundary_flux,
            interior=interior,
        )
        if reflect is not None and group.reflect_faces:
            # G-S inter-group reflect (a no-op for the Jacobi schedule, whose
            # sole group carries no reflect_faces): the walk already shed this
            # group's fresh outflow into ``boundary_flux``; reflect ``−B``
            # (outflow→inflow, in place, face-restricted) so the NEXT group
            # reads the fresh current-iterate reflected inflow off the trace.
            reflect(boundary_flux, group.reflect_faces)

    if isinstance(emit, _SweepEmitAngular):
        return emit.angular_flux, emit.scalar_flux
    # Moment mode: (moments, None).  The scalar IS φ_0^0 = ``moments[0, 0]``
    # (Y_0^0 = 1), read off the tensor by the caller — NOT returned separately
    # (returning the live ``moment_buf[0, 0]`` view invites aliasing; the
    # angular-mode scalar is an independent array, so a None here keeps the two
    # modes' second slot from being mistaken for the same kind of value).
    return emit.moment_buf, None


def _sweep_jacobi(
    Q: np.ndarray,
    sig_t: np.ndarray,
    sn_mesh: "SNMesh",
    boundary_flux: "AngularBoundaryFlux",
    *,
    spatial_closure: "DiscretizationSchemeBase",
    angular_closure: "AngularClosureBase",
    moment_frame: "FrameBase | None" = None,
    interior: "Callable",
) -> "tuple[np.ndarray, np.ndarray | None]":
    r"""The bare multi-D sweep = the **Jacobi** octant schedule × one
    interior kernel.  The JACOBI spelling (not a wavefront-specific one): all
    three multi-D representations' ``sweep`` doors route through here, each
    supplying its own interior.

    ONE group (all octants), NO inter-group reflect — delegates to the
    polymorphic :func:`_sweep_scheduled` with ``reflect=None``.  All octants
    read the same frozen inflow seed (**BARE**, Wave O #208 O.4b E1: the
    octant-incoming face slots come from the GIVEN ``boundary_flux``
    trace, no ``bc.apply`` — the reflective coupling ``B`` is delivered
    externally between sweeps, so this is the pure bulk solve
    ``ψ = (L+C)^{-1} q``).  The Gauss-Seidel SI resolvent calls the SAME
    ``_sweep_scheduled`` orchestrator with the per-in-plane-octant
    schedule + a ``−B`` reflect closure; Jacobi and G-S differ ONLY in
    the schedule object (the splitting is selected once, never by an
    ``if`` in the loop).

    S6.5 (#222): ``interior`` is REQUIRED — every caller names the
    representation instance whose kernel runs (production threads the
    operator's ONE instance; a defaulted kernel here would be a
    construction door outside ``default_for``).  Direct test consumers
    use the first-class ``MovingFrontierWindow(mesh).sweep(...)`` /
    ``ScanMarch(mesh).sweep(...)`` instead of this bare entry.

    Layout / history: ``Q`` is principled ``(N, ng, *spatial)``, ``sig_t``
    ``(ng, *spatial)`` (R-1 A1: producer-side-projected magnitude, no
    internal ``/W``; #196 PR-INDEX-5: principled face buffers,
    principled-equivalent to the legacy layout per vv-principles
    § bit-identity-vs-principled).
    """
    return _sweep_scheduled(
        Q, sig_t, sn_mesh, boundary_flux,
        spatial_closure=spatial_closure,
        angular_closure=angular_closure,
        schedule=SweepSchedule.jacobi(sn_mesh.ndim, sn_mesh.quad.octants),
        reflect=None,
        moment_frame=moment_frame,
        interior=interior,
    )


__all__ = [
    # the selection layer
    "Compatibility",
    "IncompatibleRepresentation",
    "LossRepresentation",
    "LOSS_REPRESENTATIONS",
    "default_for",
    # the concrete representations
    "CumprodScan",
    "FullFieldWavefront",
    "MovingFrontierWindow",
    "ScanMarch",
    # (No sweep entry point.  ``transport_sweep`` — the operator-free verb
    # this list once relocated here from the dissolved ``sweep.py`` — was
    # itself retired: the sweep is ``(L + C).solve(q)``, spelled through the
    # operator algebra, and there is no free function to export.)
]
