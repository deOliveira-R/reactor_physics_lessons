"""Unified SN (Discrete Ordinates) eigenvalue solver — operator-algebra form.

Wave E Round 2 (Issue #164): :class:`SNSolver` now constructs the
operator triple :math:`(L, S, F)` at ``__init__`` and consumes the
Wave E Round 1 iteration primitives.  The legacy BiCGSTAB FD-operator
path is replaced by Krylov-on-:meth:`L.apply` with the sweep as
preconditioner — the symmetric closure that closes ERR-026 for
curvilinear geometries.

Inner solver dispatch
=====================

* ``inner_solver="source_iteration"`` (default).  Sweep-driven within-
  group fixed-point iteration.  The closure is the WDD asymmetric
  closure that the curvilinear sweep ships (ERR-026 affected).  This
  path is bit-identical to the Wave A-D source iteration, by
  construction — the loop math is preserved character-for-character so
  the 11 frozen regression snapshots stay green.
* ``inner_solver="krylov"``.  GMRES on the symmetric closure carried
  by the algebraic composition
  :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` (= ``L + C``).
  This is the Wave E reconciliation that makes the curvilinear
  ``solve_sn_fixed_source`` path discretization-correct (closes
  ERR-026).  On Cartesian meshes it is bit-identical math to the
  legacy BiCGSTAB FD path.

Boundary conditions default to reflective (infinite lattice) but are
configurable via :class:`~orpheus.geometry.mesh.BC` on the mesh.

.. seealso:: :ref:`theory-discrete-ordinates` — Key Facts, equations, gotchas.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import replace
from functools import reduce
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast

import numpy as np
from scipy.sparse.linalg import gmres

from orpheus.data.macro_xs.cell_xs import assemble_cell_xs
from orpheus.data.macro_xs.mixture import Mixture
from orpheus.geometry import BC, Mesh1D, Mesh2D
from orpheus.numerics.convergence import (
    IterationRecord,
    StoppingCriterion,
    resolve_iteration_budget,
    warn_if_unconverged,
)
from orpheus.numerics.eigenvalue import power_iteration
from orpheus.numerics.face_layout import face_normal
from orpheus.sn.operators.loss_kernel_gauge import warn_if_gauge_freedom
from orpheus.transport.operators.isotropic_transfer import IsotropicFission
from orpheus.transport.reaction_rate_functional import IntegratedReactionRate
from .coupled_system import (
    WithinGroupSystem,
    _system_a_member,
    _system_b_member,
    build_within_group_system,
)
from orpheus.numerics.coupled_system import CoupledField, CoupledOperator
from orpheus.transport.radial_characteristic_field import (
    RadialCharacteristicField,
)
from .mesh.augmented_mesh import SNMesh
from orpheus.transport.spatial.scheme import DiscretizationSchemeBase
from .sweep.cache import CollisionCache, StreamingCoefficientCache
from orpheus.numerics.moment_layout import (
    AVERAGE_MOMENT,
    cell_moment_count,
    face_moment_tail,
    is_moment_valued_by_flat_rank,
)
from orpheus.numerics.quadrature import Quadrature
from orpheus.transport.operators.n2n import N2NOperator
from orpheus.transport.operators.scattering import ScatteringOperator
from orpheus.transport.mesh.axis import Axis1D
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
from orpheus.transport.full_field import FullField
from orpheus.transport.timed_full_field import TimedFullField

if TYPE_CHECKING:
    # Annotation-only names for the late-imported operator/driver types
    # (their runtime imports stay inside the function bodies — the
    # boundary/iteration modules are one-way late imports here).
    from orpheus.numerics.iteration import (
        KrylovAcceleration,
        SourceIteration,
        SupportsSeededApply,
    )
    from orpheus.numerics.operator import LinearOperator
    from orpheus.transport.fields.angular_flux import AngularFlux
    from orpheus.transport.fields.scalar_flux import ScalarFlux
    from orpheus.transport.source_sinks.angular_boundary_source_sink import (
        AngularBoundarySourceSink,
    )
    from .operators.boundary import SNBoundaryOperator
    from .operators.scheduled_invertible import ScheduledInvertibleOperator
    from .operators.streaming import StreamingCollisionOperator
    from .operators.sweep_operator import SweepOperator
    from .operators.windowing import WindowedSweep


def _apply_default_bcs(
    geometry: "Mesh1D | Mesh2D | tuple[Axis1D, ...]",
    boundary_condition: str,
) -> "Mesh1D | Mesh2D | tuple[Axis1D, ...]":
    """Apply *boundary_condition* string to all faces that lack explicit BCs.

    Returns the original declaration unchanged when it already carries
    ANY explicit :class:`~orpheus.geometry.mesh.BC`, so user-set BCs
    always take precedence over the ``boundary_condition`` parameter.

    C5.5 (#225): handles BOTH entry-surface geometry declarations — a
    legacy :class:`Mesh1D` / :class:`Mesh2D` (per-face dataclass
    fields) and an axis tuple (per-endpoint ``bc`` slots on each
    :class:`~orpheus.transport.mesh.axis.AxisMesh` /
    :class:`~orpheus.transport.mesh.axis.RadialAxisMesh`). The all-or-nothing
    semantics are identical on both representations.
    """
    bc = BC(boundary_condition)
    if isinstance(geometry, Mesh1D):
        if geometry.bc_left is None and geometry.bc_right is None:
            return replace(geometry, bc_left=bc, bc_right=bc)
        return geometry
    if isinstance(geometry, Mesh2D):
        faces = ("bc_xmin", "bc_xmax", "bc_ymin", "bc_ymax")
        if all(getattr(geometry, f) is None for f in faces):
            return replace(geometry, **{f: bc for f in faces})
        return geometry
    axes = tuple(geometry)
    if any(b is not None for ax in axes for b in ax.bc.values()):
        return axes
    return tuple(ax.with_uniform_bc(bc) for ax in axes)


def _as_sn_mesh(
    geometry: "Mesh1D | Mesh2D | tuple[Axis1D, ...]",
    quadrature: "Quadrature",
    materials: "dict[int, Mixture]",
    boundary_condition: "str | None" = None,
    mat_map: "np.ndarray | None" = None,
    *,
    scheme: "DiscretizationSchemeBase | None" = None,
) -> "SNMesh":
    r"""Normalize the entry-surface geometry declaration into an SNMesh.

    The single inbound seam for both ``solve_sn`` entries (C5.5,
    #225): ``geometry`` is a legacy :class:`Mesh1D` / :class:`Mesh2D`
    (the d≤2 user-facing declaration) or an axis tuple — the
    axis-native surface and the ONLY 3-D entry
    (:meth:`SNMesh.from_axes`). ``boundary_condition`` (the
    fixed-source vacuum convention) fills faces only when the
    declaration carries no explicit BC, on either representation;
    ``None`` (the eigenvalue entry) leaves the declaration verbatim —
    unset faces then resolve to the SNMesh-level reflective default
    (the infinite-lattice eigenvalue convention). ``mat_map`` is the
    axes-entry material-assignment channel (shape ``spatial_shape``;
    defaults to single-material id 0) — a legacy mesh carries its own
    and combining the two raises.
    """
    if boundary_condition is not None:
        geometry = _apply_default_bcs(geometry, boundary_condition)
    if isinstance(geometry, (Mesh1D, Mesh2D)):
        if mat_map is not None:
            raise ValueError(
                "mat_map is the axes-entry material channel; a legacy "
                "Mesh1D/Mesh2D carries its own mat_ids/mat_map — "
                "declare the assignment on the mesh."
            )
        return SNMesh(geometry, quadrature, materials, scheme=scheme)
    return SNMesh.from_axes(
        geometry, quadrature, materials, mat_map=mat_map, scheme=scheme,
    )


from .solution import AdjointSolution, IterationHistory, Solution, SolutionBase


# The within-group decomposition every solve consumes — the loss grid AND
# its splitting ``A = M − N`` (a splitting, NOT a Varga "regular" one — see
# the warning on that builder's module docstring, #341) — is constructed ONCE by
# :func:`orpheus.sn.coupled_system.build_within_group_system` and shipped as
# the :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record.  Its
# composition contract (why ``B`` is a separate first-class gain, RULING P1
# on the G-S split, the producer-side ``/W`` note) lives on that builder's
# docstring.  Within-group fission is zero (it enters as ``q_ext``), so
# there is no ``F = 0`` slot.


def _system_a_residual(lhs: "FullField", q_ext: "FullField") -> "FullField":
    r"""System A's typed balance defect — the named bulk/trace ``from_balance`` pair.

    The shared body of :func:`_typed_balance`'s two arms (bare /
    the coupled pair's A-member): parses the angular family at the composite
    boundary (the #289 F2-sibling role erasure) and mints the 2-block
    residual composite.
    """
    from orpheus.transport.fields._bases import AngularField, AngularBoundaryField
    from orpheus.transport.residuals import AngularResidual, AngularBoundaryResidual

    # Role parse at the composite boundary: ``AngularResidual.from_balance``
    # demands the angular family, but the ``FullField.interior`` slot erases the
    # role (the F2-sibling erasure — #289).
    # A scalar-bulk composite here is a caller error worth raising loudly —
    # on BOTH sides of the balance.
    lhs_bulk = lhs.interior
    q_bulk = q_ext.interior
    if not isinstance(lhs_bulk, AngularField) or not isinstance(
        q_bulk, AngularField
    ):
        raise TypeError(
            f"_typed_balance: both composites must carry angular-family "
            f"per-ordinate bulks; got lhs {type(lhs_bulk).__name__}, "
            f"q_ext {type(q_bulk).__name__}."
        )
    # Same parse on the trace legs: the widened ``FullField.boundary`` slot
    # (a BoundaryField since #290 P2) erases the family; the SN residual builder
    # demands the ANGULAR trace on both sides.
    lhs_boundary = lhs.boundary
    q_boundary = q_ext.boundary
    if not isinstance(lhs_boundary, AngularBoundaryField) or not isinstance(
        q_boundary, AngularBoundaryField
    ):
        raise TypeError(
            f"_typed_balance: both composites must carry angular "
            f"(AngularBoundaryField-family) traces; got lhs "
            f"{type(lhs_boundary).__name__}, rhs {type(q_boundary).__name__}."
        )
    # A residual is a one-shot balance defect, not an iterate — it carries no
    # history, so it is the timeless FullField (the history_depth=0 degenerate
    # of TimedFullField; W-C confines the timed type to the driver iterate).
    return FullField(
        interior=AngularResidual.from_balance(lhs=lhs_bulk, rhs=q_bulk),
        boundary=AngularBoundaryResidual.from_balance(
            lhs=lhs_boundary, rhs=q_boundary,
        ),
    )


def _typed_balance(
    loss_op: "LinearOperator",
    # ``FullField``, not ``TimedFullField``: the rolling-window history is
    # irrelevant to a single apply, and the #340 N6b exit defect evaluates
    # this on a reconstruction sweep's bare output.  Widened to the type
    # the body actually requires (``TimedFullField`` is a subclass, so
    # every existing caller still type-checks).
    psi: "FullField | CoupledField",
    q_ext: "FullField | CoupledField",
) -> "FullField | CoupledField":
    r"""The typed balance defect :math:`r = A\,\psi - q` of ONE GIVEN operator.

    The ARM PRIMITIVE (CS4b S4 — the F5 split): it evaluates whatever
    equation it is handed, with **no pose knowledge and no pose claim** —
    the deliberate arm-level consumers are the eigenvalue exit's bare
    System-A fission-defect projection (its carrying-mesh exemption is the
    CALL SITE's own honest reasoning) and the adjoint path's hand-daggered
    ``M.H − N.H`` equation, neither of which any
    :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record
    describes. The FULL-system claim — "the residual of the problem as
    POSED" — lives one level up on :func:`evaluate_residual`, which
    derives the required state shape from the posed system's arity;
    consumers outside this module go through that entry.

    Evaluates via the named ``from_balance`` compositions (NOT a bare
    cross-class ``−``, which would mis-type the defect as a source). On a
    bare pair ``A = L + C - S - B`` and the result is the typed 2-block
    composite ``FullField(interior=AngularResidual,
    boundary=AngularBoundaryResidual)``. On a coupled pair (B.2d — the
    coupled arm) ``loss_op`` is the 2×2 coupled loss grid, ``psi``/
    ``q_ext`` the coupled pairs, and the result is the coupled residual
    ``CoupledField[r_A, r_B]`` with ``r_B`` the split ψ½ pair
    (:class:`~orpheus.transport.residuals.radial_characteristic_interior_residual.RadialCharacteristicInteriorResidual`
    ⊕ :class:`~orpheus.transport.residuals.radial_characteristic_boundary_residual.RadialCharacteristicBoundaryResidual`).
    """
    lhs = loss_op.apply(psi)  # A·ψ — source-role members
    if isinstance(psi, CoupledField):
        from orpheus.transport.residuals import (
            RadialCharacteristicBoundaryResidual,
            RadialCharacteristicInteriorResidual,
        )

        if not isinstance(lhs, CoupledField) or not isinstance(
            q_ext, CoupledField
        ):
            raise TypeError(
                "_typed_balance: a coupled ψ requires the coupled loss "
                "grid and a coupled q_ext — got "
                f"lhs {type(lhs).__name__}, q_ext {type(q_ext).__name__}."
            )
        lhs_a = _system_a_member(lhs)
        q_a = _system_a_member(q_ext)
        lhs_b = _system_b_member(lhs)
        q_b = _system_b_member(q_ext)
        if lhs_b is None or q_b is None:
            raise TypeError(
                "_typed_balance: the coupled arm requires System-B "
                "members on both the loss output and q_ext."
            )
        r_b = RadialCharacteristicField(
            interior=RadialCharacteristicInteriorResidual.from_balance(
                lhs=lhs_b.interior, rhs=q_b.interior,
            ),
            boundary=RadialCharacteristicBoundaryResidual.from_balance(
                lhs=lhs_b.boundary, rhs=q_b.boundary,
            ),
        )
        return CoupledField(
            systems=(_system_a_residual(lhs_a, q_a), r_b),
        )
    if not isinstance(q_ext, FullField):
        raise TypeError(
            f"_typed_balance: a bare System-A ψ takes a bare FullField "
            f"q_ext; got {type(q_ext).__name__}."
        )
    return _system_a_residual(lhs, q_ext)


def evaluate_residual(
    system: "WithinGroupSystem",
    psi: "FullField | CoupledField",
    q_ext: "FullField | CoupledField",
) -> "FullField | CoupledField":
    r"""The FULL-system residual :math:`r = A\,\psi - q` of the POSED problem.

    .. math::

        r \;=\; A\,\psi \;-\; q

    with :math:`A` the posed system's loss — never a caller-supplied
    operator. Takes the
    :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record (CS4b S4
    — the F5 ruling): "does this problem carry System B?" is the POSE's
    property, and the pose is decided exactly once, at
    :func:`~orpheus.sn.coupled_system.build_within_group_system`, which
    reads the carrier and returns the 2×2 grid (carrying) or the 1×1 grid
    (seedless). So the required state shape is DERIVED from the system's
    arity, and the Mode-12 (b) hazard the pre-S4 mesh-reading guard
    patched — a bare System-A call on a carrying problem silently
    dropping System B's defect (a DSA consumer feeding
    ``solution.angular_flux`` alone would get a residual blind to a wrong
    seed row) — is refused by STRUCTURE: you cannot obtain a seedless
    system from a carrying mesh, and a 2×2 system refuses bare state by
    arity. The scheme-binding doctrine executed: the augmented object
    binds what the method needs, so the schemeless call is unspellable by
    currency rather than patrolled by a mesh read.

    On a 1×1 system the state is the bare 2-block composite pair and the
    result is ``FullField(interior=AngularResidual,
    boundary=AngularBoundaryResidual)``; on a 2×2 system the state is the
    coupled pair and the result is ``CoupledField[r_A, r_B]`` with
    ``r_B`` the split ψ½ residual pair — System B's defect is a typed
    member that cannot be silently dropped.

    A diagnostic (``balance_map`` / :func:`boundary_vs_interior_split` /
    ``relative_to``) AND the substrate the consistent-DSA low-order
    correction (`#2`) will consume (``r`` is the transport residual the
    diffusion solve corrects). NOT in the convergence path — evaluated on
    a (typically converged) flux, additive. Arm-level equations that no
    system record describes (the eigenvalue exit's deliberate System-A
    fission projection, the adjoint path's daggered ``M.H − N.H``) use
    the module-private :func:`_typed_balance` primitive, which carries no
    full-system claim.

    Parameters
    ----------
    system : WithinGroupSystem
        The posed within-group system (loss + splitting, one construction
        site). The residual is evaluated against ``system.loss``.
    psi : FullField or CoupledField
        The FULL-angular flux state, in the shape the system's arity
        demands (``bulk`` an ``AngularFlux``; for a windowed 2-D solve
        pass the reconstructed ``Solution.angular_flux``, NOT the moment
        iterate — the operators consume per-ordinate flux).
    q_ext : FullField or CoupledField
        The external source (source-role members), matching ``psi``'s
        carrier shape.
    """
    if not isinstance(system, WithinGroupSystem):
        raise TypeError(
            "evaluate_residual takes the POSED system (WithinGroupSystem) "
            "— the full-system residual is a claim about the problem as "
            "posed, so the pose travels with the call; got "
            f"{type(system).__name__}. Build it with "
            "build_within_group_system; for an arm-level equation no "
            "system describes, use the module-private _typed_balance."
        )
    arity = system.loss.n_cols
    if arity == 2:
        if not isinstance(psi, CoupledField) or not isinstance(
            q_ext, CoupledField
        ):
            raise ValueError(
                "evaluate_residual: this system is 2×2 — the mesh carries "
                "starting-direction levels (R12a) — pass the coupled pair "
                "[ψ_A, ψ_B] and coupled q_ext for the FULL-system "
                "residual; a bare System-A call would silently drop "
                "System B's defect (Mode-12 (b))."
            )
        return _typed_balance(system.loss, psi, q_ext)
    if isinstance(psi, CoupledField) or isinstance(q_ext, CoupledField):
        raise ValueError(
            "evaluate_residual: this system is 1×1 (seedless) — pass the "
            "bare FullField pair, not a coupled wrapper."
        )
    return _typed_balance(_bare_loss_arm(system), psi, q_ext)


def boundary_vs_interior_split(
    residual: "FullField | CoupledField",
) -> tuple[float, float]:
    r"""Split a typed System-A residual into ``(boundary, interior)`` L2 norms.

    Returns the flat-L2 norm of the boundary residual and of the interior
    (bulk) residual, so :math:`\sqrt{b^2 + i^2} = \lVert r_A\rVert` (System
    A's composite flat norm — EXACT since B.2d made ``FullField`` 2-block;
    the pre-eviction 3-block silently excluded the seed rows from this
    identity, the closed diagnostic gap). Discriminates a BC-realizer /
    reflective-trace defect (large ``boundary``) from an interior-streaming
    defect (large ``interior``). A coupled residual funnels to its System-A
    member; System B's defect is its own typed member — read
    ``residual.systems[1]`` directly (its flat norm is the ray-row defect).
    """
    member = _system_a_member(residual)
    interior = float(np.linalg.norm(np.asarray(member.interior.values).ravel()))
    boundary = float(np.linalg.norm(np.asarray(member.boundary.values).ravel()))
    return boundary, interior


class ConvergenceCertificateError(RuntimeError):
    r"""A within-group solve CLAIMED convergence but the honest equation
    residual disagrees — the production lag-death classifier (step 5,
    R-5.2).

    The running stop is the FREE-IDENTITY residual ``r = rhs_{n−1} −
    rhs_n`` (:class:`~orpheus.numerics.iteration.SourceIteration`), which
    equals the true ``Aψ − q`` ONLY when the step operator is an exact
    inverse of the splitting's ``M``. An in-``M`` inconsistency — the
    #282 class: a lagged seed, a stale block, a walk whose fixed point
    does not solve the equation — leaves the identity (and any
    ``‖Δψ‖``-family test) reporting "converged" while the equation is
    violated O(1). The certificate is the ONE honest
    :func:`evaluate_residual` per solve that closes exactly that hole
    (the row-6 oracle is tautology-blind to it — an in-``M`` lag rides
    both sides of an assembled self-compare; TA step-5 memo refutation
    #8).
    """


#: The certificate's false-alarm guard: the free identity and the honest
#: residual agree to FP-reassociation grain when M is exact, so a genuine
#: lag-death (O(1) defect — #282 measured 5e5) clears this by orders of
#: magnitude while exact-M exits never trip it.
_CERTIFICATE_SAFETY = 10.0


def _residual_is_expressible(sn_mesh: "SNMesh") -> bool:
    r"""Can this mesh's iterate be turned into a typed equation residual?

    ``False`` for a **moment-tailed (LD) scheme**: the residual mint
    (:class:`~orpheus.transport.residuals.AngularResidual` ``from_balance``)
    does not admit the trailing ``2^d`` spatial-moment axis, so
    :func:`evaluate_residual` raises rather than returning a field.  That
    is an un-built widening of the residual family, NOT a threat gap — the
    carrying production family is DD — and it lands with the LD residual
    carve (#310's deferred-out list).

    Named because TWO consumers need the same precondition and must not
    drift apart: :func:`_certify_within_group_exit` (which skips its
    correctness assertion) and :func:`_exit_balance_defect` (which reports
    no number).  Spelled inline in the first until 2026-08-10; the second
    would have been a second copy of the same `> 1` test, one rename away
    from disagreeing about which schemes are exempt.
    """
    return sn_mesh.scheme.spatial_basis_per_axis == 1


def _angular_moment_values(
    field: "FullField | TimedFullField | CoupledField",
) -> np.ndarray:
    r"""The zeroth angular moment of a composite's bulk, as a raw array.

    :math:`\phi_g(\mathbf{r}) = \sum_n w_n \, x[n, g, \mathbf{r}]`, shape
    ``(ng, *spatial)``.  :func:`_system_a_member` handles the coupled/bare
    split first, so a carrying mesh's
    :class:`~orpheus.numerics.coupled_system.CoupledField` reduces its
    System A member.

    ⚠ **The private ``_integrate_angular_values`` is reached deliberately,
    and the reason is a real gap rather than convenience.**  It is the ONE
    angular-reduction body, but only two of the four angular ROLES wrap it
    in a public ``integrate_angular`` — the two that have a scalar sibling
    TYPE to return.  A residual and a source-sink have none, and minting
    ``ScalarResidual`` for a single consumer fails the type-vs-property
    test (one realization, no non-identity morphism applied).  The only
    alternative is hand-rolling ``tensordot(w, x)``, which is a second
    spelling of a body whose whole purpose is being single — strictly
    worse.  If a third consumer ever wants this, promote the base method
    rather than adding a third caller of the private one.

    The narrowing below is not defensive: ``FullField.interior`` is typed
    as the broader ``BulkField``, and the angular reduction is exactly what
    distinguishes a BALANCE projection from a bare volume integral.  A
    moment-tailed interior can only arrive here if
    :func:`_residual_is_expressible` was widened without widening this, so
    the error names that rather than dying on a missing attribute.
    """
    from orpheus.transport.fields._bases import AngularField

    interior = _system_a_member(field).interior
    if not isinstance(interior, AngularField):
        raise TypeError(
            f"_angular_moment_values needs a per-ordinate interior to "
            f"integrate over angle; got {type(interior).__name__}. If the "
            f"residual mint has grown a moment-tailed arm, this reduction "
            f"needs its matching angular counterpart before anything can "
            f"report on one."
        )
    return np.asarray(interior._integrate_angular_values())


def _balance_projection(
    field: "FullField | TimedFullField | CoupledField", *, sn_mesh: "SNMesh",
) -> np.ndarray:
    r"""Project a per-ordinate field onto the per-group BALANCE functional.

    .. math::

        R_g \;=\; \int_V \int_{4\pi} x(\mathbf{r}, \mathbf{\Omega})_g
                  \, d\Omega \, dV
              \;=\; \sum_n w_n \sum_i V_i \, x[n, g, i]

    Angle first (the quadrature weights), then space (the cell volumes) —
    returning ``(ng,)``, a rate per group.  Both reductions are the
    canonical ones: the angular contraction is
    ``AngularField._integrate_angular_values`` (the ONE reduction body,
    shared with :meth:`AngularFlux.integrate_angular`) and the spatial one
    is
    :meth:`~orpheus.transport.mesh.material_mesh.MaterialMesh.integrate_per_group`.

    ⚠ **The private access is deliberate.** Only two of the four angular
    ROLES ship a public ``integrate_angular`` — the ones with a scalar
    sibling type to wrap the result in.  A residual and a source-sink have
    none, and minting ``ScalarResidual`` for this single consumer would
    fail the type-vs-property test (one realization, no non-identity
    morphism).  The alternative is hand-rolling ``tensordot(w, x)``, which
    is a second spelling of a body whose whole point is being single.  So:
    reach for the shared body, and say why.

    :func:`_system_a_member` handles the coupled/bare split — on a carrying
    mesh the driver state is a
    :class:`~orpheus.numerics.coupled_system.CoupledField` whose System A
    member is the composite this projects.
    """
    return sn_mesh.integrate_per_group(_angular_moment_values(field))


def _exit_balance_defect(
    loss_op: "LinearOperator",
    psi: "FullField | TimedFullField | CoupledField",
    q: "FullField | CoupledField",
    *,
    sn_mesh: "SNMesh",
    record: IterationRecord,
) -> float | None:
    r"""The returned iterate's RELATIVE per-group neutron-balance defect.

    .. math::

        \frac{\lVert R_g(A\psi - q)\rVert}{\lVert R_g(q)\rVert},
        \qquad
        R_g(x) = \int_V \int_{4\pi} x_g \, d\Omega \, dV

    A dimensionless magnitude: the net per-group imbalance the returned
    iterate leaves in its own equation, as a fraction of the per-group
    source rate.

    ⭐ **The exact complement of :func:`_certify_within_group_exit`, and
    the pair is deliberate.**  Both take a ``record`` and one forward
    apply; the certificate fires when the solve CLAIMED convergence and
    *asserts* (raising on a defect beyond ``_CERTIFICATE_SAFETY × tol``),
    this fires when it did not and *reports*.  One equation, two verbs,
    complementary guards — so no solve pays for both, and the happy path
    keeps exactly the cost it had before N6b.

    ``None`` in three cases HERE, each meaning something different: the
    tree fully converged (the certificate has it), the scheme cannot
    express a residual at all (:func:`_residual_is_expressible`), or the
    source integrates to zero so the ratio is undefined.  Two further
    ``None`` cases live at CALL SITES rather than in this body, because
    they are about what the caller can assemble rather than what this can
    compute — a carrying mesh at :func:`solve_sn` (#354) and the daggered
    eigenvalue entry (#353).  The full list is on
    :attr:`~orpheus.sn.solution.IterationHistory.balance_defect`, which is
    what a reader holding a ``None`` will actually be looking at.

    **Why this projection and not the residual norm.**  `[M]` #340 N5: the
    raw defect :math:`\lVert r \rVert / \lVert q \rVert` cannot tell a
    truncation that corrupted :math:`k` from one that did not — the benign
    and corrupting populations overlap **634×** and a threshold admitting
    every benign case misses **15 of 16** corrupting ones.  The reason is
    structural, not statistical: up to **99.995 %** of :math:`\lVert r
    \rVert` is reflective-trace rows, and a reflective inflow-trace defect
    in a zero-leakage system carries **no net current**, so a balance-based
    :math:`k` is blind to it *by conservation*.  Projecting onto
    :math:`R_g` — the functional :math:`k` actually reads — annihilates
    exactly those rows, and `[M]` cuts the overlap to **4.64×**.

    ⚠ **4.64× is still an overlap.  This is a DIAGNOSTIC, never a gate.**
    Do not branch on it, do not threshold it, do not assert on its
    magnitude in a test.  It is reported so a reader can weigh a truncation
    they have been told about; the attempt to make it a verdict is the
    refuted N5, and the refutation is in the plan beside the text it
    refutes.

    ⛔ And do not reach for an adjoint weight to sharpen it without solving
    for one: `[M]` a spatially-flat 0-D adjoint makes it **worse**, 4.64× →
    **128.95×**, because a signed projection against a wrong weight
    manufactures near-cancellations, i.e. false negatives.  The weighting
    machinery already exists
    (:meth:`IntegratedReactionRate.evaluate` takes ``adjoint=``); what a
    real gate would need is the adjoint SOLVE (#350).

    The equation is whatever the calling entry solved: :math:`q` is the
    fission source :math:`F\phi(\psi)/k` at an eigenvalue exit and the
    given :math:`q_{\rm ext}` at a fixed-source one.  Only the eigenvalue
    form inherits the 4.64× figure above — that is the population N5
    measured.
    """
    if record.fully_converged:
        return None
    if not _residual_is_expressible(sn_mesh):
        return None
    source_rate = _balance_projection(q, sn_mesh=sn_mesh)
    denominator = float(np.linalg.norm(np.asarray(source_rate)))
    if denominator == 0.0:
        return None
    residual = _typed_balance(loss_op, psi, q)
    defect_rate = _balance_projection(residual, sn_mesh=sn_mesh)
    return float(np.linalg.norm(np.asarray(defect_rate))) / denominator


#: Either flavour of the System-A composite — both carry a ``.boundary``.
#: A TypeVar rather than a union so the gauged result keeps the caller's exact
#: type: the fixed-source arms hand their return straight to ``Solution``.
_TraceCarrier = TypeVar("_TraceCarrier", "FullField", "TimedFullField")


def _exit_gauge_trace(
    psi: _TraceCarrier,
    *,
    sn_mesh: "SNMesh",
) -> "tuple[_TraceCarrier, float | None]":
    r"""Return the CANONICAL member of the returned trace's solution manifold.

    On an all-reflective Cartesian box closed by diamond differencing,
    :math:`A = L + C - S - B` is **exactly singular** (#344), so a converged
    solve lands on an arbitrary member of a solution manifold rather than on a
    point.  This projects the kernel component out —
    :math:`\psi \mapsto \psi - \Pi\psi` — leaving the minimum-:math:`G`-norm
    member, which is where the exact solution sits (a theorem, not a
    convention: every kernel mode is mirror-ODD, so any mirror-even functional
    annihilates it, and :math:`\psi_{\rm exact}` is one).

    Returns the gauged composite and
    :math:`\lVert \Pi\psi \rVert / \lVert \psi \rVert` for
    :attr:`~orpheus.sn.solution.IterationHistory.gauge_correction` — ``None``
    when there was no freedom to measure, never *"measured and zero"*.

    ⭐ **The sibling of** :func:`_exit_balance_defect` **with one sharpening:
    that one REPORTS and this one MUTATES.**  A forgotten balance-defect site
    loses a diagnostic; a forgotten gauge site silently returns a
    non-physical answer.  The structural guarantee a single construction site
    would have given is not available (`[M]` the two fixed-source arms
    deliberately bypass :func:`_package_solution` to keep their DG slope
    structure, :ref:`the note below <no-label>` at the arms), so coverage is
    GATED instead — see
    ``tests/sn/solve/test_every_entry_gauges_its_trace.py``.

    ⛔ **Rebuilds, never mutates in place.**  `[M]` on the Krylov arm the bulk
    and trace are two views into ONE flat buffer
    (``psi_full.boundary.values.base is psi_full.interior.values.base`` →
    ``True``), which ``psi_typed`` also references and which is still read
    after this point; on the un-windowed SI arm ``angular_out IS psi_typed``,
    the very object :func:`_exit_balance_defect` already measured.  An in-place
    write would reach backwards through both.  ``dataclasses.replace`` also
    re-runs ``__post_init__``, so the leaf's block invariants re-fire — where
    ``Composite._recombine`` would silently drop ``_history``.

    **Residual-neutral by construction**, so it is safe at a converged exit:
    :math:`A(\psi - \Pi\psi) = A\psi` because :math:`\Pi\psi \in \ker A`.  `[M]`
    on a truncated SI solve ``_exit_balance_defect`` reads
    ``0.3111434602740818`` on both the raw and the gauged iterate while
    ``gauge_correction`` goes ``3.592e-2 → 4.91e-17``.  Call it AFTER the
    defect anyway, so the reported number describes the object the caller
    actually receives.

    ⚠ Takes the **System-A member**, not a ``CoupledField``: only the transport
    bulk⊕trace block carries a boundary at all (a ``CoupledField`` has no
    ``.boundary``), and every exit already unpacks it with
    :func:`_system_a_member` before reading the trace.
    """
    gauge = sn_mesh.loss_kernel_gauge
    if not gauge.blocks:
        return psi, None

    boundary = psi.boundary
    values = np.asarray(boundary.values, dtype=float)
    gauged = gauge.gauge(values)
    scale = float(np.linalg.norm(values))
    correction = (
        float(np.linalg.norm(values - gauged)) / scale if scale > 0.0 else 0.0
    )
    return replace(psi, boundary=replace(boundary, values=gauged)), correction


def _certify_within_group_exit(
    system: "WithinGroupSystem",
    psi: "TimedFullField | CoupledField",
    q_ext: "FullField | CoupledField",
    *,
    sn_mesh: "SNMesh",
    record: IterationRecord,
    where: str,
) -> None:
    r"""The end-of-solve convergence CERTIFICATE — one honest residual.

    No-op when the exit made NO claim (``max_iter`` hit without reaching
    ``tol`` — best-effort returns stay legal); when the driver's stop
    CLAIMED convergence, evaluates the true ``r = A·ψ − q`` through
    :func:`evaluate_residual` (a real forward apply — the only
    measurement an in-``M`` lag cannot fool) and raises
    :class:`ConvergenceCertificateError` on a defect beyond
    ``_CERTIFICATE_SAFETY × tol``.

    Wired on every FULL-ANGULAR arm (the coupled sphere, the seedless
    un-windowed SI, both Krylov paths). Two structural exemptions —
    honest scope, each with the threat model spelled:

    * **the windowed 2-D moment arm** (skipped at the call sites): (i)
      the #282 in-M lag class is CARRYING-only and windowing is 2-D
      Cartesian ⟹ seedless (the threat cannot exist there); (ii) its
      G-S fixed-point threat class (ERR-056) is value-gated by the
      C5.5 Mode-9 mixed-BC box; (iii) the moment iterate cannot feed
      the typed angular residual without a per-solve reconstruction
      (the r3 coisometry exemption — the moment free-identity keeps
      the STOP role only).
    * **moment-tailed (LD) schemes** (skipped HERE, one seam): the
      residual mint (:class:`~orpheus.transport.residuals.AngularResidual`
      ``from_balance``) does not yet admit the trailing ``2^d``
      spatial-moment axis — an un-built widening of the residual
      family, NOT a threat gap today: the carrying production family
      (where the #282 class lives) is DD.  (The LD reverse-scan and
      adjoint arms this note once leaned on landed at #310 C2/C5 — the
      remaining gap is the residual mint alone.)  The widening lands
      with the LD residual carve (step-5 close-out note; on #310's
      deferred-out list).
    """
    if not _residual_is_expressible(sn_mesh):
        return  # moment-tailed scheme — the residual mint's un-built widening
    if not record.converged:
        return  # no convergence claim — nothing to certify
    criterion = record.binding_criterion
    if criterion is None or criterion.last is None:
        return  # nothing was measured — see IterationRecord.iterated
    tol = criterion.tolerance
    residual = evaluate_residual(system, psi, q_ext)
    r_norm = float(np.linalg.norm(np.asarray(residual.to_flat())))
    q_norm = max(float(np.linalg.norm(np.asarray(q_ext.to_flat()))), 1e-30)
    defect = r_norm / q_norm
    if defect > _CERTIFICATE_SAFETY * tol:
        raise ConvergenceCertificateError(
            f"{where}: the within-group solve claimed convergence "
            f"(running residual {criterion.last:.3e} < tol {tol:.1e}) "
            f"but the honest equation residual is ‖Aψ − q‖/‖q‖ = "
            f"{defect:.3e} — the iteration's fixed point does not solve "
            f"the equation (the #282 lag-death class: a stale/lagged "
            f"block inside M; the free-identity stop is structurally "
            f"blind to it)."
        )




def _bare_loss_arm(system: "WithinGroupSystem") -> "LinearOperator":
    r"""The 1×1 grid's (A,A) entry — the bare ``L+C−S−B_a`` composition.

    The seedless system's equation, unwrapped from the arity-guarded grid
    (whose ``apply`` demands a ``CoupledField`` even at arity 1, while the
    seedless drivers carry bare composites). Consumed by
    :func:`evaluate_residual`'s seedless arm (CS4b S4 — the one place the
    unwrap is spelled for the full-system claim) and by the arm-level
    ``_exit_balance_defect`` call sites that deliberately evaluate the
    System-A equation alone (the eigenvalue exit's fission-defect
    projection)."""
    arm = system.loss.blocks[0][0]
    if arm is None:  # unreachable — the grid ctor guards diagonal presence
        raise RuntimeError(
            "_bare_loss_arm: the 1×1 loss grid carries no (A,A) block."
        )
    return arm


def _within_group_krylov(
    LC: "LinearOperator", *gains: "LinearOperator",
    n_dof: int, max_iter: int, tol: float,
    corrector: "LinearOperator | None" = None,
) -> "KrylovAcceleration[Any]":
    r"""GMRES driver on the within-group system ``(M − N)·ψ = q``.

    Single source of truth (Cardinal Rule 2 / Phase 1 R2) for the
    :class:`~orpheus.numerics.iteration.KrylovAcceleration` construction shared
    by the eigenvalue (:meth:`SNSolver._solve_krylov`) and fixed-source
    (:func:`_solve_fixed_source_krylov`) Krylov paths — they previously built
    byte-identical instances.  The call sites pass the
    :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record's splitting:
    ``LC`` is ``M`` (the fused ``(L+C)`` seedless; the triangular
    :class:`~orpheus.numerics.coupled_system.CoupledOperator` grid
    ``[[LC, Seeding], [None, march]]`` on a carrying mesh — step 5, the
    block matvec) and ``*gains`` is ``N`` (the ``(S, B_a)`` pair seedless;
    the ONE coupled gain grid carrying); the matvec is
    ``M.apply − Σ Nᵢ.apply``.

    Without a ``corrector``, GMRES is UNPRECONDITIONED (explicit identity)
    per `issue #200 <https://github.com/deOliveira-R/ORPHEUS/issues/200>`_
    (the block-inverse face preconditioner re-enablement).  With one (the
    consistent-DSA posture, issue #2), the left preconditioner is the
    **transport-corrected** :math:`M^{-1} \approx (A - \Sigma g)^{-1}` of
    Adams & Larsen §VI:

    .. math::

        M^{-1} v \;=\; t + \mathcal{C}\,t, \qquad t = (L+C)^{-1} v,

    one sweep followed by the DSA correction of the swept vector — the
    swept vector IS the increment from a zero iterate, so the SAME correction
    operator serves both the SI and Krylov postures (single source of
    truth on :math:`R, A_{\rm low}^{-1} G, P`).  The preconditioner
    changes the Krylov TRAJECTORY only, never the converged fixed point
    (gated by D4; its effectiveness is the paired rate gate D13).

    ``restart`` is sized to the FULL problem ``n_dof = N·ng·nx·ny`` — the
    legacy ``min(50, …)`` clamp left GMRES structurally truncated on any
    mesh with ``n_dof > 50`` (ERR-053).
    """
    from orpheus.numerics.iteration import KrylovAcceleration, seeded_inverse

    if corrector is None:
        # explicit identity — issue #200 tracks the face-preconditioner
        # re-enablement; the DSA posture below is the first re-enabled M.
        preconditioner = lambda q: q  # noqa: E731
    else:
        sweep = seeded_inverse(LC)

        def preconditioner(q):
            swept = sweep.apply(q)
            return swept + corrector.apply(swept)

    return KrylovAcceleration(
        LC, *gains,
        preconditioner=preconditioner,
        tol=tol, max_iter=max_iter,
        restart=n_dof,
        budget_name="max_inner",
    )


def _maybe_window(
    sweep: "SweepOperator", scattering_op: "ScatteringOperator",
    sn_mesh: "SNMesh",
) -> "tuple[WindowedSweep | SweepOperator, bool]":
    r"""Phase 5a — compose the 2-D Cartesian angular-windowing product over
    ``sweep`` (the inverse operator ``A.inverse()``), else passthrough.
    Returns ``(step, windowed)``.

    The SINGLE site of the windowing-eligibility gate AND the factory of the
    windowed product (``coding-elegance`` Pattern 7 — the convention lives in
    one place, shared by the eigenvalue and fixed-source SI drivers):
    genuinely 2-D Cartesian holds the SI iterate as harmonic moments via the
    typed composition ``P @ A.inverse()`` (#226 §17 W1) — the ``@`` dispatch
    on a :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` right
    factor fuses to :class:`~orpheus.sn.operators.windowing.WindowedSweep`;
    curvilinear (1-D) stays full-angular — the Morel–Montry Carlson seed
    reads the previous per-ordinate iterate at ``μ=−1`` (lesson L21), which
    the moment tensor does not carry.  ``P`` is sourced from the scattering
    operator's own MINTED flux-analysis face (F-1) ⇒ the stored moments
    match ``S``'s internal projection term-for-term.

    C5.4 (#225, vv Mode 9): the gate is the GENUINE condition
    ``is_cartesian and ndim == 2`` — the pre-C5.4 ``reduced is None``
    proxy was a coincidence that is ALSO true at d=3 Cartesian and would
    have silently moment-windowed a 3-D solve (the in-sweep moment
    emission is a 2-D kernel; ``FullFieldWavefront`` refuses moment mode).
    """
    if sn_mesh.is_cartesian and sn_mesh.ndim == 2:
        from .operators.windowing import BulkAnalysisOperator

        return (
            BulkAnalysisOperator(scattering_op.flux_analysis, sn_mesh.full_field_space) @ sweep,
            True,
        )
    return sweep, False


def _windowed_cold_start(scattering_op, sn_mesh, *, history_depth):
    r"""Zero windowed (moment-bulk) SI cold-start iterate.

    The moment representation the windowed resolvent emits and the
    moment-consuming ``S.apply`` / ``B.apply`` expect — shared by both SI
    drivers (``coding-elegance`` Pattern 2).
    """
    from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
    from orpheus.transport.fields.harmonic_moment_flux import (
        HarmonicMomentFlux,
    )
    from orpheus.transport.timed_full_field import TimedFullField

    return TimedFullField(
        interior=HarmonicMomentFlux.zeros_for_mesh_and_L(
            sn_mesh, scattering_op.legendre_order,
            spatial_moments=sn_mesh.scheme.spatial_basis_per_axis,
        ),
        boundary=AngularBoundaryFlux.zeros(sn_mesh.angular_trace),
        _history=(),
        history_depth=history_depth,
    )


def _unwindowed_cold_start(sn_mesh, *, history_depth):
    r"""Zero un-windowed (full-angular) SI cold-start iterate.

    The full-angular ``AngularFlux`` iterate the 1-D / curvilinear SI driver
    holds.  Selects the SpatialMomentSpace factor (#240 D5b-S3) so a
    multi-moment closure (LD) carries the φ̂ axis end-to-end — composing with the
    moment-carrying ``q_ext`` + ``S.apply(ψ)`` in the SI rhs.  DD/Step
    (per_axis == 1) → no factor (byte-identical to the prior ``TimedFullField.zeros``).
    The un-windowed sibling of :func:`_windowed_cold_start` (Pattern 2).
    On a seed-carrying mesh the COUPLED cold start wraps this System-A
    frame with a zero ψ_B via :func:`_coupled_flux_state` (B.2d)."""
    from orpheus.transport.fields.angular_flux import AngularFlux
    from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux
    from orpheus.transport.timed_full_field import TimedFullField

    return TimedFullField(
        interior=AngularFlux.zeros(sn_mesh.angular_trial_space),
        boundary=AngularBoundaryFlux.zeros(sn_mesh.angular_trace),
        _history=(),
        history_depth=history_depth,
    )


def _radial_characteristic_source_from_per_ordinate(
    per_ordinate_values: "np.ndarray", sn_mesh,
    *, boundary_trace: "AngularBoundarySourceSink | None" = None,
) -> "RadialCharacteristicField | None":
    r"""Fold a PER-ORDINATE source to its q½ composite —
    :meth:`RadialCharacteristicField.source_from_angular` (Legendre-project
    the per-ordinate source to moments, then fold at :math:`\mu=\pm 1`).

    The PER-ORDINATE typed entry to the one fold kernel
    (:func:`~orpheus.numerics.spaces.radial_characteristic_space.fold_moments_to_radial_characteristic`),
    used where the source is genuinely per-ordinate (possibly anisotropic):
    the fixed-source rhs external source (:func:`_build_fixed_source_rhs`, its
    ONE caller since #448 — the eigenvalue finalize's total-source fold
    retired when the finalize became one step of the driven iteration, whose
    ψ½ member arrives as the fission seed). The eigenvalue FISSION q½ seed uses the
    MOMENTS entry :func:`_radial_characteristic_fission_seed` instead (its ℓ=0
    emission is already a moment, so the direct fold skips this factory's
    per-ordinate round-trip). Both bottom out in the SAME kernel — no twin, a
    different typed input.

    ``boundary_trace`` (step 7, the prescribed-corner arm): the SAME source
    composite's boundary member, forwarded so the factory can deliver the
    prescribed-inflow r = R corner datum to System B's given-data slot —
    see the factory's three-arm inflow-corner law. Pass it wherever the
    per-ordinate source HAS a boundary member (the fixed-source rhs);
    omit for boundary-free folds (cold starts, reconstructions whose
    corner is populated separately from the converged state).
    """
    return RadialCharacteristicField.source_from_angular(
        per_ordinate_values, sn_mesh, boundary_trace=boundary_trace,
    )


def _radial_characteristic_fission_seed(
    fission_source: "np.ndarray", sn_mesh,
) -> "RadialCharacteristicField | None":
    r"""The ψ½ FISSION ray seed — the direct ℓ=0 moments-fold of the fission
    emission (``A_BA_fission = Fold ∘ F.isotropic_energy``, factored).

    The eigenvalue outer loop computes ``fission_source = χνΣf·φ/k`` from the
    SCALAR flux (:meth:`SNSolver.compute_fission_source`), so
    ``F.isotropic_energy ∘ integrate`` is already applied and only the fold
    remains: the isotropic ℓ=0
    emission is reconstructed at the closed :math:`\mu=\pm 1` rays through the
    migrated
    :class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicReconstruction`
    (the single fold operator). ``None`` on a seedless mesh (no ψ½ ray).

    REPLACES the ``from_isotropic → from_angular_source`` round-trip (campaign
    step 4c commit 2): the old path broadcast ``fission_source`` to a per-ordinate
    iso source then re-projected it back to the ℓ=0 moment; the direct moments-fold
    is one step. Both bottom out in the SAME
    :func:`~orpheus.numerics.spaces.radial_characteristic_space.fold_moments_to_radial_characteristic`
    kernel — principled-equivalent (~ULP), NOT bit-identical (the removed
    round-trip's per-ordinate ``·w`` reassociates). Carrying meshes are 1-D
    curvilinear (R12a) so ``fission_source`` is ``(ng, nx)``; the fold takes the
    unit-ℓ ``[None]`` axis (``RadialCharacteristicReconstruction.apply`` guards it).
    """
    if sn_mesh.radial_characteristic_field_space is None:
        return None
    from orpheus.sn.operators.radial_characteristic import (
        RadialCharacteristicReconstruction,
    )
    return RadialCharacteristicReconstruction(
        sn_mesh.radial_characteristic_field_space,
        coord=sn_mesh.coord,
        quadrature=sn_mesh.quad,
    ).apply(
        np.asarray(fission_source)[None],
    )


def _coupled_flux_state(
    psi_a: "TimedFullField", sn_mesh: "SNMesh",
) -> "CoupledField":
    r"""Pair a System-A FLUX iterate with a zero System-B flux composite.

    The coupled cold-start / iterate birth on a carrying mesh (B.2d — the
    pair is born native; there is no fused 3-block to split). ψ_B's zero
    flux composite comes from the presence-gated
    :meth:`~orpheus.transport.radial_characteristic_field.RadialCharacteristicField.flux_zeros`.
    """
    return CoupledField(
        systems=(psi_a, RadialCharacteristicField.flux_zeros(sn_mesh.radial_characteristic_field_space)),
    )


def _coupled_source_state(
    q_a: "FullField", q_half: "RadialCharacteristicField | None",
    sn_mesh: "SNMesh", *, context: str,
) -> "CoupledField":
    r"""Pair a System-A SOURCE composite with its q½ System-B member.

    The coupled rhs birth on a carrying mesh: ``q_half`` is the q½ fold
    composite (:func:`_radial_characteristic_source_from_per_ordinate` /
    :func:`_radial_characteristic_fission_seed`) — System B's member,
    paired directly. ``None`` refuses loudly — a carrying mesh's rhs MUST
    carry the true q½ (the direct ψ½ solve consumes it).
    """
    if q_half is None:
        raise ValueError(
            f"{context}: a carrying mesh's coupled rhs requires the q½ fold "
            f"(got None) — the joint solve consumes System B's true source."
        )
    return CoupledField(systems=(q_a, q_half))


class InnerSolve(NamedTuple):
    r"""What the LAST within-group solve left behind — the posed
    :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record, the
    forward ``M`` whose inverse the driver applied (the un-windowed
    ``base_implicit`` on the SI arm; the record's ``implicit_operator`` on
    Krylov), the lagged gains ``N`` it evaluated each step (moment-bound when
    the SI iterate was windowed; the coupled gain grid when carrying), and
    the converged iterate itself (System A's bulk ⊕ trace, paired with
    System B's ψ½ member on a carrying mesh — B.2d).

    Written by both eigenvalue inner solves as ``SNSolver._inner`` (``None``
    before the first inner solve — the truthful type of a state that does
    not exist yet), read as the next inner's warm start, by ``compute_keff``'s
    leakage term (the converged trace), and by the finalize
    (:func:`solve_sn`), which evaluates the SAME map once more —
    :func:`~orpheus.numerics.iteration.fixed_point_step` on the converged
    iterate with the converged fission source — through the operators the
    iteration actually converged against.  The record IS the history (#340
    N2b-ii): nothing re-selects a splitting the inner already chose, so the
    reconstruction cannot drift from the iteration (#448), and the four
    members travel as one fact — they are written at one site and never
    independently.
    """

    system: "WithinGroupSystem"
    implicit: "CoupledOperator | StreamingCollisionOperator | ScheduledInvertibleOperator"
    gains: "tuple[LinearOperator, ...]"
    iterate: "TimedFullField | CoupledField"


def _eigenvalue_driver_source(
    fission_source: np.ndarray, sn_mesh: SNMesh, *, context: str,
) -> "TimedFullField | CoupledField":
    r"""The eigenvalue solve's ``q_ext`` for ONE within-group solve: the
    fission source :math:`F\phi/k` lifted to the per-ordinate composite,
    paired with its ψ½ seed on a carrying mesh.

    The single construction site the three eigenvalue solves share — the SI
    inner (:meth:`SNSolver._solve_source_iteration`), the Krylov inner
    (:meth:`SNSolver._solve_krylov`) and the finalize (:func:`solve_sn`,
    which reconstructs the returned flux by ONE
    :func:`~orpheus.numerics.iteration.fixed_point_step` from the converged
    iterate).  What the drivers converge against IS what the finalize
    rebuilds from, by construction — #448 was exactly that drift (the
    finalize hand-built a P0-only source of its own).

    * bulk — the per-ordinate density via the canonical
      :meth:`~orpheus.transport.source_sinks.AngularSourceSink.from_isotropic`
      factory (the ``/W`` projection at the factory boundary — Pattern 7
      producer-side normalisation; the legacy ``(fission_source /
      sum_w)[None]`` broadcast is GONE);
    * trace — ZERO: the EXTERNAL boundary source.  The reflective coupling
      is NOT pre-staged here — it is the ``B`` gain the drivers apply each
      step (Wave O O.2a; since B.2d the record's ``explicit_gains``), so the
      inflow is a live solved unknown carried in ``ψ.boundary``;
    * on a carrying mesh (the mesh HAS a radial-characteristic space — the
      same partition :func:`_build_fixed_source_rhs` reads, derived here
      rather than passed as a flag) the pair with System B's member: the
      ℓ = 0 fold of the FISSION source as the ψ½ march's entry
      (:func:`_radial_characteristic_fission_seed`, #282 route (a)) — the
      gains carry everything else (the coupled gain grid's ``Emission`` and
      ``B_b`` blocks), so the seed folds the fission source alone, exactly
      as the inner solves pass it.
    """
    from orpheus.transport.source_sinks import (
        AngularBoundarySourceSink,
        AngularSourceSink,
    )
    from orpheus.transport.timed_full_field import TimedFullField

    q_composite = TimedFullField(
        interior=AngularSourceSink.from_isotropic(fission_source, sn_mesh),
        boundary=AngularBoundarySourceSink.zeros(sn_mesh.angular_trace),
        _history=(),
        history_depth=2,
    )
    if sn_mesh.radial_characteristic_field_space is None:
        return q_composite
    return _coupled_source_state(
        q_composite,
        _radial_characteristic_fission_seed(fission_source, sn_mesh),
        sn_mesh,
        context=context,
    )


def _select_si_splitting(
    LC: "StreamingCollisionOperator",
    B: "LinearOperator[FullField, FullField]",
    sn_mesh: "SNMesh", inner_schedule: str,
) -> "tuple[StreamingCollisionOperator | ScheduledInvertibleOperator, LinearOperator[FullField, FullField]]":
    r"""Pick the ``(implicit_operator, boundary_gain)`` for the SEEDLESS within-group SI
    driver per ``inner_schedule`` — the single source of truth for the
    Jacobi/G-S choice, which is a choice about the BOUNDARY coupling only.

    Seedless-only since B.2d: a seed-carrying mesh takes the coupled
    block-native arm in :func:`_within_group_si` and never reaches this
    selector (its M/N splitting is fixed by
    :func:`~orpheus.sn.coupled_system.build_within_group_system`).

    * ``"jacobi"`` (or any 1-D mesh) → ``(L+C, B_a)``: the whole boundary
      lagged as an external gain (inter-sweep Jacobi — every geometry).
    * ``"gauss_seidel"`` on a multi-D Cartesian mesh →
      ``((L+C) - B_lower, B_upper)``: the splitting
      ``(L+C−B) = M − B_upper`` (#226 §17 W2).  ``B`` splits under the
      octant-group schedule (:meth:`SNBoundaryOperator.split`); the
      strictly-lower half folds into the REIFIED forward
      :class:`~orpheus.sn.operators.scheduled_invertible.ScheduledInvertibleOperator`
      (whose ``solve`` is the octant-group forward substitution), and the
      complement lags as an ordinary external gain — structurally congruent
      with the Jacobi arm, so the driver needs no case split.  The collision
      gains are NOT this selector's business: ``S`` and ``N₂ₙ`` lag in BOTH
      arms (only the boundary coupling gets G-S; the sweep never re-scatters
      mid-sweep), so the caller names the gain triple
      ``(S, N₂ₙ, boundary_gain)`` — §14.1's order, ``B`` LAST — itself.
      (Until the CS4c step-5 review round the selector passed ``S`` and
      ``N₂ₙ`` through its return tuple, and the windowed driver rebuilt two
      of the three slots by index; that is the smell the round removed.)

    1-D falls back to Jacobi: boundary G-S is a no-op on the scattering-
    dominated 1-D regime AND the 1-D scan is not a wavefront.  The converged
    fixed point is identical either way — this only selects the SI spectral
    rate.

    C5.4 (#225): the G-S gate is the GENUINE condition ``is_cartesian and
    not is_1d`` — the pre-C5.4 ``reduced is None`` proxy was 2-D-Cartesian
    by coincidence only. ``SweepSchedule.gauss_seidel`` and the scheduled
    sweep are d-generic (C3); d=3 G-S FP-invariance is value-gated by the
    C5.5 Mode-9 mixed-BC box (vv Mode 9 — never trust a splitting on a
    degenerate regime alone).
    """
    if inner_schedule not in ("jacobi", "gauss_seidel"):
        raise ValueError(
            f"Unknown inner_schedule: {inner_schedule!r}. "
            f"Valid choices are 'gauss_seidel' (boundary G-S, multi-D "
            f"Cartesian) or 'jacobi' (the splitting-invariant control)."
        )
    if (
        inner_schedule == "gauss_seidel"
        and sn_mesh.is_cartesian
        and not sn_mesh.is_1d
    ):
        from .loss_representation.sweep_schedule import (
            SweepSchedule,
            reflective_faces,
        )
        from .operators.boundary import SNBoundaryOperator

        # Multi-D Cartesian ⟹ SEEDLESS ⟹ B is the plain SNBoundaryOperator
        # (B_a alone; no ray block). The schedule split lives on B_a, never the
        # B_a + B_b composite (RULING P1 corollary) — this narrowing asserts that
        # invariant (a seed-carrying composite would be curvilinear, not
        # Cartesian, so it never reaches here).
        if not isinstance(B, SNBoundaryOperator):
            raise TypeError(
                "boundary Gauss-Seidel split requires the plain "
                "SNBoundaryOperator (a seedless multi-D Cartesian mesh); got "
                f"{type(B).__name__} — a seed-carrying composite must not reach "
                "the G-S schedule path (RULING P1: gradings live on B_a)."
            )
        parts = B.split(SweepSchedule.gauss_seidel(
            sn_mesh.ndim, sn_mesh.quad.octants, reflective_faces(sn_mesh),
        ))
        return LC - parts.lower, parts.upper
    return LC, B


def _within_group_si(
    system: "WithinGroupSystem",
    sn_mesh: "SNMesh", *, inner_schedule: str, max_iter: int, tol: float,
    corrector: "LinearOperator | None" = None,
) -> "tuple[SourceIteration[Any], CoupledOperator | StreamingCollisionOperator | ScheduledInvertibleOperator, tuple[LinearOperator, ...], bool]":
    r"""SourceIteration driver on the within-group system ``A = M − N``.

    Single source of truth (Cardinal Rule 2 / Phase 1 R1) for the
    :class:`~orpheus.numerics.iteration.SourceIteration` construction shared by
    the eigenvalue (:meth:`SNSolver._solve_source_iteration`) and fixed-source
    (:func:`_solve_fixed_source_si`) paths — the SI sibling of
    :func:`_within_group_krylov` (both inner methods have ONE construction
    helper consumed by their eigenvalue + fixed-source sites). Consumes the
    :class:`~orpheus.sn.coupled_system.WithinGroupSystem` record — the loss
    and its splitting from the ONE construction site
    (:func:`~orpheus.sn.coupled_system.build_within_group_system`).

    Two structurally-dispatched arms (B.2d DP-seedless — the coupled
    carrier appears exactly where System B exists):

    * **coupled** (the record's ``implicit_operator`` is the triangular
      :class:`~orpheus.numerics.coupled_system.CoupledOperator` grid — a
      seed-carrying 1-D curvilinear mesh): the block-native driver
      ``ψ ← M⁻¹(q + N·ψ)`` on the ``CoupledField [ψ_A, ψ_B]`` iterate,
      ``M⁻¹`` the block back-substitution
      (:class:`~orpheus.numerics.coupled_system.CoupledSubstitutionOperator`,
      step 5 — System B's march, then the ray-decoupled bulk sweep on
      ``q_A − Seeding·ψ_B``).
      Never windowed (carrying ⟹ 1-D, R12a) and never schedule-split
      (G-S is multi-D Cartesian ⟹ seedless, RULING P1) — both machineries
      are bypassed structurally, and ``inner_schedule`` is inert here (the
      1-D Jacobi fallback the seedless arm spells explicitly).
    * **seedless**: exactly the pre-B.2d composition — the schedule
      splitting (:func:`_select_si_splitting`, Jacobi vs boundary-G-S),
      the INVERSE build (``base_implicit.inverse()`` — the
      :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`), and
      the Phase-5a angular-windowing composition (:func:`_maybe_window` —
      2-D Cartesian holds the iterate as harmonic moments via
      ``P @ A.inverse()``).

    Returns ``(si, base_implicit, gains, windowed)``:

    * ``si`` — the :class:`SourceIteration` primitive;
    * ``base_implicit`` — the un-inverted FORWARD ``M`` (both finalizes
      need it for the one-shot full-angular reconstruction of
      ``Solution.angular_flux`` — the fixed-source windowed arm directly,
      the eigenvalue finalize through the :class:`InnerSolve` record
      the inner solve leaves beside its iterate);
    * ``gains`` — the lagged couplings ``N`` actually driven (the record's,
      except the G-S arm's re-split ``(S, N₂ₙ, B_upper)``), the other half
      of that reconstruction (:func:`~orpheus.numerics.iteration.fixed_point_step`);
    * ``windowed`` — whether the iterate is the moment representation (2-D
      Cartesian) vs full-angular (curvilinear / 1-D).

    Both paths forward their caller's ``inner_schedule`` (default boundary
    Gauss-Seidel on 2-D Cartesian — `#218
    <https://github.com/deOliveira-R/ORPHEUS/issues/218>`_ closed the
    eigenvalue-inner gap; the eigenvalue path reads ``SNSolver.inner_schedule``,
    the fixed-source path its ``inner_schedule`` argument).
    """
    from orpheus.numerics.iteration import SourceIteration

    if isinstance(system.implicit_operator, CoupledOperator):
        # The ψ½ coupled block-native arm (B.2d): the record's splitting IS
        # the driver's — M⁻¹ = the joint sweep, N = the coupled gain grid.
        # A corrector never reaches here: consistent DSA's admission is
        # 1-D CARTESIAN (curvilinear = carrying is #282-blocked), enforced
        # at DSALowOrderSystem.from_sn_mesh before this builder runs.
        if corrector is not None:
            raise NotImplementedError(
                "_within_group_si: a synthetic-acceleration corrector on "
                "the coupled (curvilinear) arm has no stability theory "
                "(#282); the DSA admission should have refused upstream."
            )
        si = SourceIteration(
            system.implicit_operator.inverse(), *system.explicit_gains,
            max_iter=max_iter, tol=tol, budget_name="max_inner",
        )
        return si, system.implicit_operator, system.explicit_gains, False
    # Seedless: the record's explicit_gains are the (S, N2N, B_a) triple
    # (§14.1; B_a LAST) — loud on drift.
    S, n2n, B = system.explicit_gains
    if not isinstance(S, ScatteringOperator) or not isinstance(
        n2n, N2NOperator,
    ):
        raise TypeError(
            f"_within_group_si: the seedless record's gains must lead "
            f"(ScatteringOperator, N2NOperator) — the builder's "
            f"(S, N2N, B_a) convention; got "
            f"({type(S).__name__}, {type(n2n).__name__})."
        )
    base_implicit, boundary_gain = _select_si_splitting(
        system.implicit_operator, B, sn_mesh, inner_schedule,
    )
    step, windowed = _maybe_window(base_implicit.inverse(), S, sn_mesh)
    # The gains, by NAME (S, N₂ₙ, the boundary gain — §14.1's order). When
    # windowed the iterate is the MOMENT composite, so the two collision
    # gains that read it are bound on it (CS4c step 5): the same datum and
    # faces as the record's angular bindings, the domain's interior the
    # analysis face's codomain — each binding acts through the body its
    # ends select, and the moment operand is admitted by ITS operator
    # instead of being dispatched on by the angular one. The boundary
    # gain (or its G-S upper part) reads the trace and stays.
    gains = (
        (S.on_moment_domain(), n2n.on_moment_domain(), boundary_gain)
        if windowed else (S, n2n, boundary_gain)
    )
    if corrector is not None and windowed:
        raise NotImplementedError(
            "_within_group_si: the DSA corrector consumes the full-"
            "angular increment; the 2-D moment-windowed iterate is "
            "outside the arm-1 admission (the corner-moment follow-up)."
        )
    si = SourceIteration(
        step, *gains, max_iter=max_iter, tol=tol, corrector=corrector,
        budget_name="max_inner",
    )
    return si, base_implicit, gains, windowed


# ═══════════════════════════════════════════════════════════════════════
# Solver class (EigenvalueSolver protocol)
# ═══════════════════════════════════════════════════════════════════════

class SNSolver:
    """Unified SN eigenvalue solver satisfying the EigenvalueSolver protocol.

    Constructs the operator triple :math:`(L, S, F)` at construction
    time and routes ``solve_fixed_source`` through one of two inner-
    solver paths:

    * ``"source_iteration"`` — sweep-driven within-group fixed-point
      iteration (WDD asymmetric closure; ERR-026-affected for
      curvilinear).  Bit-identical to the Wave A-D path.
    * ``"krylov"`` — GMRES on
      :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` (the algebraic
      composition ``L + C``; symmetric closure) with the sweep as
      preconditioner.  Closes ERR-026 on curvilinear; bit-identical
      math to the legacy BiCGSTAB FD path on Cartesian.

    The legacy ``"bicgstab"`` value is no longer accepted — call sites
    must migrate to ``"krylov"``.

    Parameters
    ----------
    sn_mesh : SNMesh — augmented geometry (wraps Mesh1D or Mesh2D with
        precomputed streaming stencil + materials dict + ``ng``).
        Issue #197 PR-TYPED-0: ``sn_mesh.materials`` IS the single
        source of truth for cross sections and group count; the
        legacy ``materials`` / ``n_groups`` SNSolver constructor
        parameters were retired (aggressive retirement per
        ``feedback_aggressive_retirement``).
    inner_solver : "source_iteration" or "krylov".
    scattering_order : int — Legendre order for scattering (0 = P0).
    keff_tol, flux_tol : outer iteration convergence.
    inner_tol : the inner iteration's convergence tolerance.
    max_inner : the inner iteration's budget.  ``None`` (the default) DERIVES
        it from ``inner_tol`` at the served rate — see
        :func:`~orpheus.numerics.convergence.default_iteration_budget`.  An
        explicit int is a deliberate cap and is never second-guessed.
    """

    def __init__(
        self,
        sn_mesh: SNMesh,
        inner_solver: str = "source_iteration",
        scattering_order: int = 0,
        keff_tol: float = 1e-7,
        flux_tol: float = 1e-6,
        max_inner: int | None = None,
        inner_tol: float = 1e-8,
        inner_schedule: str = "jacobi",
    ):
        if inner_solver not in ("source_iteration", "krylov"):
            raise ValueError(
                f"Unknown inner solver: {inner_solver!r}. "
                f"Valid choices are 'source_iteration' or 'krylov'. "
                f"(The legacy 'bicgstab' alias was retired in Wave E "
                f"Round 2; use 'krylov' which routes through "
                f"StreamingCollisionOperator (L + C) with the sweep as "
                f"preconditioner.)"
            )
        if inner_schedule not in ("jacobi", "gauss_seidel"):
            raise ValueError(
                f"Unknown inner_schedule: {inner_schedule!r}. "
                f"Valid choices are 'gauss_seidel' (boundary G-S, multi-D "
                f"Cartesian — auto-falls-back to Jacobi on 1-D) or 'jacobi'."
            )
        self.sn_mesh = sn_mesh
        self.quad = sn_mesh.quad
        self.inner_solver = inner_solver
        # SI BOUNDARY splitting for the eigenvalue inner (#218 — the eigenvalue
        # SI now CAN reach the boundary-Gauss-Seidel accelerator the
        # fixed-source path got in Phase 3, via the shared ``_within_group_si``
        # builder; validated SI(G-S)≡Krylov≡k_inf).  DEFAULT stays ``"jacobi"``:
        # the eigenvalue inner is warm-started (the G-S rate benefit is modest
        # there), and a schedule change shifts the converged k_eff by ~inner_tol
        # (1e-10-scale — same fixed point, vv Mode 9; only the inner SI stopping
        # differs), which the keff_tol-tight regression snapshots cannot absorb.
        # ``"gauss_seidel"`` is opt-in (2-D Cartesian; ``_select_si_splitting``
        # auto-falls-back to Jacobi on 1-D / curvilinear).
        self.inner_schedule = inner_schedule
        self.scattering_order = scattering_order
        self.keff_tol = keff_tol
        self.flux_tol = flux_tol
        # Resolved ONCE, here, so `self.max_inner` is always a live int: every
        # downstream reader (the two drivers, the truncation warning) sees the
        # budget that actually bound rather than a `None` it would have to
        # re-resolve — Pattern 7, and the reason the warning can name a number.
        self.max_inner = resolve_iteration_budget(max_inner, inner_tol)
        self.inner_tol = inner_tol

        # ``materials`` + ``ng`` are the single source of truth on the mesh
        # (``sn_mesh.ng`` raises ``InconsistentMaterialsError`` if materials
        # disagree — not a constructor parameter).
        materials = sn_mesh.materials
        self.ng = sn_mesh.ng

        # The canonical XS state is ONE attribute — ``self.mat_xs``, a
        # :class:`MaterialXSField` wrapping both the per-material
        # :class:`Mixture` data and the per-cell typed views.  Every operator
        # (L, C, S, F) reads cross sections through this single source of
        # truth via ``self.mat_xs.*`` accessors (``total_cross_section`` /
        # ``absorption_cross_section`` / …).
        self.mat_xs = sn_mesh.material_xs_field()

        # __debug__ cell-flattening invariant pinning (formerly at
        # construction of self.sig_t — now exercised through the
        # mat_xs.total_cross_section accessor, populated lazily).
        if __debug__:
            xs_check = assemble_cell_xs(materials, sn_mesh.mat_map)
            _sig_t_old = xs_check.sig_t.reshape(*sn_mesh.spatial_shape, self.ng)
            assert np.array_equal(
                _sig_t_old,
                np.moveaxis(self.mat_xs.total_cross_section, 0, -1),
            ), "PR-INDEX-3 cell-flattening invariant broke"

        # Scattering order — clamp to the minimum Legendre count
        # available across all materials.
        L = min(
            scattering_order,
            min(len(m.SigS) - 1 for m in materials.values()),
        )
        self.scattering_order = L

        # Weight normalization (1/sum(w) — works for both GL and Lebedev)
        self.weight_norm = 1.0 / sn_mesh.quad.weights.sum()

        #: The last within-group solve — its posed system, the splitting it
        #: drove and its converged iterate (:class:`InnerSolve`); ``None``
        #: until the first inner solve runs.  Warm start for the next inner,
        #: the converged trace for ``compute_keff``'s leakage term, and the
        #: operand of the finalize's one reconstruction step (#448).  (The
        #: write-only ``_boundary_flux`` buffer that sat here since #197
        #: retired with the finalize's hand reflect — the trace is a live
        #: unknown carried in ``iterate.boundary`` since Wave O O.4a.2.)
        self._inner: InnerSolve | None = None

        #: Inner records, newest last — one per within-group solve this
        #: instance has run.  Appended in ``_solve_source_iteration`` /
        #: ``_solve_krylov``; this is what makes ``SNSolver`` a
        #: :class:`~orpheus.numerics.eigenvalue.RecordingSolver`, so the outer
        #: record carries a SUBTREE and "the outer stalled because its inner
        #: starved" is answerable from the returned solution.
        #:
        #: ⛔ Replaced the scalar ``_total_inner_iterations`` accumulator on
        #: 2026-08-09 (#340 N2b).  That counter summed ``record.n_iterations``
        #: as each record arrived and dropped the record — the same lossy
        #: projection the campaign removes everywhere else, and the reason the
        #: FORWARD eigenvalue path could report a total but not a tree while
        #: its adjoint twin (``KEigenvalue``) could.  The total is now DERIVED
        #: (``sum(r.n_iterations for r in inner_records)``), which also retires
        #: the second spelling of it.
        self.inner_records: list[IterationRecord] = []

        # Volume array for keff computation
        self.volume = sn_mesh.volumes

        # ── The two cached reaction operators ────────────────────────────
        # S and F are the only operators worth caching on the solver: they
        # are σ-read-through (both consume the single ``self.mat_xs``; the
        # per-material dispatch lives inside :class:`MaterialXSField`'s typed
        # verbs, not on the operators — #197 PR-TYPED-1), so they survive a
        # cross-section rebind untouched and are shared BY IDENTITY into
        # every within-group build (``scattering_op=`` on
        # :func:`build_within_group_system`).
        #
        # The loss composite ``L + C`` is deliberately NOT cached here.  The
        # ONE LC spelling is :func:`build_streaming_collision`, and every
        # production solve reaches it through
        # :func:`build_within_group_system`, which builds the composite it
        # actually inverts.  A second, solver-held copy would be a twin that
        # can silently drift from the one the sweep uses (it did: the former
        # ``self.L``/``self.S``/``self.F`` triple was production-dead and
        # misnamed — ``self.L`` held ``L + C`` while the codebase's ``L`` is
        # the σ-free streaming leaf).  Consumers needing the composite call
        # ``build_streaming_collision(sn_mesh, mat_xs)`` directly.
        self.scattering_op = ScatteringOperator.from_solver_data(
            mat_xs=self.mat_xs,
            scattering_order=self.scattering_order,
            space=sn_mesh.full_field_space,
        )
        # §14.1 — the (n,2n) channel is its own first-class operator; the
        # within-group algebra spells (L+C) − S − N₂ₙ − B explicitly.
        self.n2n_op = N2NOperator.from_solver_data(
            mat_xs=self.mat_xs,
            scattering_order=self.scattering_order,
            space=sn_mesh.full_field_space,
        )
        # The fission ENERGY binding on the scalar bulk space (CS4c
        # step 4 — the binding-arity table's F row made true): the
        # k-outer feeds bare (ng, *spatial) scalar arrays, and the
        # binding now says so. The ANGULAR composite binding
        # (FissionOperator, the frame's ℓ=0 conjugation) is minted
        # where it is consumed — the eigen-M posing below.
        self.fission_op = IsotropicFission.from_material_xs(
            self.mat_xs,
            space=self.mat_xs.mesh.bulk_space,
        )

        # ── Sweep cache (Issue #196 Phase G Step 2.5c) ───────────────
        # Two-stratum cache: StreamingCoefficientCache built once at __init__
        # (geometry × quadrature only); CollisionCache built once after
        # σ_t binding.  Hot path consumes (geom, coll) without per-cell
        # StreamingTerms allocation.  Only applicable to 1-D meshes with
        # ReducedStreamingOperator — 2-D Cartesian uses the wavefront path.
        self.geom_cache: StreamingCoefficientCache | None = None
        self.coll_cache: CollisionCache | None = None
        # The two-stratum scan cache feeds the DAG-FREE scan strategies
        # (CumprodScan / ScanMarch) ONLY — its σ_t stratum is the closed-form
        # affine recurrence ``affine_scan_coefficients`` (the scan-family
        # triple), which a NON-affine-scannable scheme (LinearDiscontinuous,
        # #158) does not supply.  Such schemes run on the DAG wavefront
        # (FullFieldWavefront), which consumes the per-cell ``cell_kernel_batch``
        # directly — never the scan cache.  Build the cache only when the scan
        # path can actually be selected (DD keeps its bit-identical cache).
        # Chain scan ⟺ 1-D (the honest predicate, P4.5); ``reduced``
        # presence is its ctor-guaranteed realization.
        if sn_mesh.is_1d and sn_mesh.scheme.is_affine_scannable:
            # Stratum 1 through the strategy layer's INTERN (P4.9b step 2c
            # — one build per mesh × closure pair, however many operators
            # a solve constructs; the retired eager build + the mesh-attr
            # ``_geom_cache`` stash both lived here).  The σ-stratum below
            # needs it NOW to pose CollisionCache — that eager σ posing
            # (and its ``_coll_cache`` stash the walk reads) is Campaign
            # 2's consumer-side territory, deliberately untouched.
            from orpheus.sn.loss_representation import geometry_cache_for

            self.geom_cache = geometry_cache_for(
                sn_mesh, sn_mesh.angular_closure,
            )
            # No bridge needed: ``mat_xs.total_cross_section`` is the
            # principled ``(ng, nx)`` 1-D layout the cache expects
            # (rank-d (N, ng, *spatial); no phantom ny axis to drop).
            sig_t_1d = self.mat_xs.total_cross_section  # (ng, nx)
            self.coll_cache = CollisionCache.from_geometry(
                self.geom_cache, sig_t_1d, sn_mesh.scheme,
                sn_mesh.angular_closure,
            )
            sn_mesh._coll_cache = self.coll_cache  # type: ignore[attr-defined]

    def rebind_cross_sections(self, new_sig_t: np.ndarray) -> None:
        """Rebind the total cross-section and rebuild only :class:`CollisionCache`.

        :class:`StreamingCoefficientCache` survives — Stratum 1 is geometry-only.
        Only the σ_t-dependent Stratum 2 rebuilds.  Used by depletion /
        thermal-feedback consumers.

        Parameters
        ----------
        new_sig_t
            New total cross-section in the principled ``(ng, nx, ny)``
            layout (Issue #196 PR-INDEX-3).

        Notes
        -----
        Issue #197 PR-TYPED-1 — ``rebind_cross_sections`` overrides
        ``self.mat_xs._sig_t_cell`` directly (without re-deriving from
        materials) because the depletion / thermal-feedback consumer
        adjusts σ_t per-cell without revisiting the per-material data.
        """
        # Override the lazy cache on mat_xs.  Force the dense
        # per-cell view to be populated first so the other cell views
        # (sig_a, sig_p, chi) match the rebind contract.
        _ = self.mat_xs.absorption_cross_section
        self.mat_xs._sig_t_cell = new_sig_t
        # No operator rebuild is needed for the rebound σ_t to take effect.
        # ``L`` is σ-free (#257 S8b), and the collision diagonal ``C =
        # M[σ_t]`` is constructed FRESH on every solve by
        # :func:`build_within_group_system` from the read-through
        # ``mat_xs.total_cross_section_field`` property — so the composite
        # that is actually inverted is built after this rebind and carries
        # the new σ_t.  (A ``MultiplicationOperator`` holds its coefficient
        # as a snapshot, which is exactly why caching one here would go
        # stale — hence no solver-held copy; see ``__init__``.)  Only the
        # materialised σ_t stratum of the sweep cache, which likewise
        # snapshots values, has to be rebuilt.
        if self.geom_cache is not None:
            sig_t_1d = self.mat_xs.total_cross_section
            self.coll_cache = CollisionCache.from_geometry(
                self.geom_cache, sig_t_1d, self.sn_mesh.scheme,
                self.sn_mesh.angular_closure,
            )
            self.sn_mesh._coll_cache = self.coll_cache  # type: ignore[attr-defined]

    def initial_flux_distribution(self) -> np.ndarray:
        """Initial scalar flux guess: ones(ng, nx, ny).

        Issue #196 PR-INDEX-5: principled layout.
        """
        return np.ones((self.ng, *self.sn_mesh.spatial_shape))

    def compute_fission_source(
        self, flux_distribution: np.ndarray, keff: float,
    ) -> np.ndarray:
        """Fission source: χ · (νΣ_f · φ) / k.

        Thin delegator to the fission ENERGY binding's ``apply``
        (:class:`~orpheus.transport.operators.isotropic_transfer.IsotropicFission`
        — CS4c step 4: the scalar dyad bound at the mesh's bulk space).
        The :math:`1/k` division stays at this level — the fission
        operator is a *linear* operator; the binding returns
        :math:`F\\,\\phi` and the eigenvalue scaling lives here.

        Issue #196 PR-INDEX-5: ``flux_distribution`` is principled
        ``(ng, nx, ny)``.  No bridges — the PR-INDEX-4 transpose pair
        is GONE.
        """
        # The bare-ndarray leg returns bare (the union carries the
        # composite arm's type; asarray is the zero-cost narrowing).
        return np.asarray(self.fission_op.apply(flux_distribution)) / keff

    def solve_fixed_source(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        """Solve the within-group transport equation for given fission source.

        Returns updated scalar flux ``(ng, nx, ny)``.
        """
        if self.inner_solver == "source_iteration":
            return self._solve_source_iteration(fission_source, flux_distribution)
        if self.inner_solver == "krylov":
            return self._solve_krylov(fission_source, flux_distribution)
        # Should be unreachable — __init__ validated the choice.
        raise ValueError(f"Unknown inner solver: {self.inner_solver}")

    def compute_group_production_rate(
        self, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        r"""Per-group volume-integrated neutron production rate, shape ``(ng,)``.

        Component :math:`r_g` is

        .. math::

            r_g \;=\; \int_V \nu \Sigma_{f,g}(\mathbf{r})\,\phi_g(\mathbf{r})\,dV
                       \;+\; 2 \int_V \sum_{g' } \Sigma_{2,g'\to g}(\mathbf{r})
                                                 \,\phi_{g'}(\mathbf{r})\,dV

        i.e. the per-group fission-neutron production plus the per-group
        ``(n, 2n)`` contribution (the factor of 2 accounts for the
        two-neutron-out yield).  Fission is integrated against
        ``mesh.volume_measure`` (Issue 9.6 wiring); ``(n, 2n)`` runs the
        existing per-material loop because the ``sig2`` matrices are
        keyed on material rather than cell.

        The output is the natural diagnostic intermediate for spectral
        analysis (per-group production rates are reactor-physics-meaningful
        quantities).  ``compute_keff`` consumes it via ``.sum()``.
        """
        # Fission production: ∫ νΣ_f · φ dV, vectorised over groups.
        # Issue #196 PR-INDEX-5: both ``mat_xs.fission_production`` and
        # ``flux_distribution`` are principled ``(ng, nx, ny)``.  The
        # named intermediate ``per_cell_per_group`` has units ``[1/s]``
        # per cell per group (a reactor-physics quantity — coding-
        # elegance Pattern 3);
        # :meth:`~orpheus.transport.mesh.material_mesh.MaterialMesh.integrate_per_group`
        # owns the volume integral and the flat-view reshape it needs.
        per_cell_per_group = np.einsum(
            "g...,g...->g...", self.mat_xs.fission_production, flux_distribution,
        )
        rate = self.sn_mesh.integrate_per_group(per_cell_per_group)

        # (n,2n) contribution — Issue #197 PR-TYPED-1: the per-material
        # dispatch loop (and the yield) lives ONLY inside
        # :meth:`TransferMaterialField.add_to_group_rate` (§14.1).
        self.n2n_op.isotropic_energy.transfer.add_to_group_rate(
            rate, flux_distribution, self.volume,
        )

        return rate

    def compute_group_absorption_rate(
        self, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        r"""Per-group volume-integrated absorption rate, shape ``(ng,)``.

        Component :math:`a_g = \int_V \Sigma_{a,g}(\mathbf{r})\,\phi_g(\mathbf{r})\,dV`.

        Volume-integrated via ``mesh.volume_measure`` (Issue 9.6 wiring).
        ``.sum()`` of this vector is the ABSORPTION term of the
        ``compute_keff`` denominator (net removal = absorption + leakage
        − (n,2n) emission since #291/R7).

        Issue #196 PR-INDEX-5: ``flux_distribution`` is principled
        ``(ng, nx, ny)``.
        """
        per_cell_per_group = np.einsum(
            "g...,g...->g...", self.mat_xs.absorption_cross_section, flux_distribution,
        )
        return self.sn_mesh.integrate_per_group(per_cell_per_group)

    def compute_production_rate(self, flux_distribution: np.ndarray) -> float:
        r"""Total volume-integrated neutron production rate (scalar).

        :math:`P(\phi) = \int_V \sum_g \nu \Sigma_{f,g} \phi_g\,dV
        + 2 \int_V \sum_g \sum_{g'} \Sigma_{2,g'\to g} \phi_{g'} \,dV`.

        The fission term is the typed volume-integrated reaction rate
        :class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`
        over :math:`\nu\Sigma_f` — ``∫_V ⟨νΣf, φ⟩ dV`` — the single source of the
        ``Σx·φ`` contraction and its volume integral. The :math:`(n,2n)` channel
        is an **explicit additive term** (a second neutron-multiplying reaction,
        NOT a ``⟨Σx,φ⟩`` rate); it reuses the single ``TransferMaterialField.add_to_group_rate``
        machinery and is exactly zero on a no-(n,2n) mixture.

        This is the canonical scale anchor for the SN eigenmode:
        :func:`orpheus.numerics.eigenvalue.power_iteration` renormalises
        :math:`\phi` to unit production rate at each outer step (ERR-052).

        Role split (R7, #259): this TOTAL physical production — fission
        plus the (n,2n) emission — is the renormalisation scale anchor
        ONLY. The k numerator in :meth:`compute_keff` is fission-only,
        because the posed eigenproblem scales only fission by
        :math:`1/k`; the (n,2n) gain sits on the net-removal side there.
        """
        fission = IntegratedReactionRate(
            self.mat_xs.fission_production_field
        ).evaluate(flux_distribution)
        n2n_rate = np.zeros(self.ng)
        self.n2n_op.isotropic_energy.transfer.add_to_group_rate(
            n2n_rate, flux_distribution, self.volume,
        )
        return float(fission + n2n_rate.sum())

    def compute_keff(self, flux_distribution: np.ndarray) -> float:
        r"""k of the POSED eigenproblem: fission production over net removal.

        .. math::

            k \;=\; \frac{R_{\nu\Sigma_f}(\phi)}
                    {R_{\Sigma_a}(\phi) \;+\; L \;-\; E_{2n}(\phi)}

        Every inner solve poses the eigenproblem with ONLY fission scaled
        by :math:`1/k` — scattering and the (n,2n) emission are plain
        gains inside :meth:`solve_fixed_source` — so the reported k must
        be the eigenvalue of exactly that problem (#291 leakage omission
        + the R7 (n,2n) convention, 2026-07-03; an estimator that
        disagrees with its own posed problem converges cleanly to a
        non-eigenvalue ratio):

        * **numerator** — the typed volume-integrated FISSION production
          :math:`R_{\nu\Sigma_f}(\phi)`
          (:class:`~orpheus.transport.reaction_rate_functional.IntegratedReactionRate`,
          the :math:`\phi^\dagger\!=\!1` degenerate of the homogenization
          PG bilinear). The (n,2n) emission is NOT production here —
          contrast :meth:`compute_production_rate`, the ERR-052 scale
          anchor, which keeps total physical production.
        * **denominator** — net removal, assembled so no term CAN be
          forgotten: absorption :math:`R_{\Sigma_a}` (``absorption_xs``
          counts the (n,2n) COLLISION once), **plus** the net
          vacuum-boundary leakage :math:`L` (#291 — the historically
          omitted term; see :meth:`_boundary_leakage_rate`), **minus**
          the (n,2n) EMISSION :math:`E_{2n}(\phi) = \int_V \sum_{g,g'}
          2\,\Sigma_{2,g'\to g}\,\phi_{g'}\,dV` (a gain reduces net
          removal).

        Balance identity at any converged eigenpair:
        :math:`R_{\nu\Sigma_f}/k = R_{\Sigma_a} + L - E_{2n}`.

        On an all-reflective (lattice) problem :math:`L` is a STRUCTURAL
        zero, and on a Σ₂-free mixture :math:`E_{2n}` is exactly ``0.0``
        — so this reduces **bit-identically** to the historical lattice
        functional ``production / absorption``.

        The per-group :meth:`compute_group_production_rate` /
        :meth:`compute_group_absorption_rate` remain available as
        spectral diagnostics (not on the keff path).
        """
        production = IntegratedReactionRate(
            self.mat_xs.fission_production_field
        ).evaluate(flux_distribution)
        absorption = IntegratedReactionRate(
            self.mat_xs.absorption_cross_section_field
        ).evaluate(flux_distribution)
        emission_n2n = np.zeros(self.ng)
        self.n2n_op.isotropic_energy.transfer.add_to_group_rate(
            emission_n2n, flux_distribution, self.volume,
        )
        leakage = self._boundary_leakage_rate(production)
        return production / (absorption + leakage - emission_n2n.sum())

    def _boundary_leakage_rate(self, fission_production: float) -> float:
        r"""Net neutron outflow rate through the vacuum boundary faces [1/s].

        .. math::

            L \;=\; \sum_{f\,\in\,\text{vacuum}} \oint_{f} dA\,
                    \sum_g J_g(\mathbf{r}_f)
            \,, \qquad
            J_g \;=\; \sum_m (\Omega_m\cdot\hat n_f)\, w_m\, \psi_{m,g}

        — the face-area integral of the boundary trace's net outward
        current (:meth:`AngularBoundaryFlux.net_current
        <orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux.net_current>`,
        the single source of the :math:`\Omega\cdot\hat n\,w`
        contraction), read from the trace of the last inner solve
        (``self._inner.iterate.boundary``). On the converged trace a vacuum
        face's inflow slots are zero, so net = outflow; the signed form
        stays honest if a prescribed-inflow law ever lands.

        Reflective faces are a **structural zero**: the reflective law
        equates inflow to the reflected outflow exactly, so their net
        current vanishes by construction — they are SKIPPED, never
        accumulated, which keeps all-reflective problems' float
        arithmetic bit-identical to the lattice functional (no
        ±cancelling angular-sum noise enters the denominator).

        Scale bridge: the stored trace belongs to the UN-renormalised
        last inner iterate, while the estimator's :math:`\phi` may be
        its renormalised multiple
        (:func:`~orpheus.numerics.eigenvalue.power_iteration` divides by
        the production rate between the solve and the k-update). Leakage
        is degree-1 homogeneous in :math:`\psi`, so it is rescaled by
        the fission-production ratio of the two — exactly ``1.0`` when
        the caller passes the returned flux itself. Contract: the flux
        handed to :meth:`compute_keff` must be a scalar multiple of the
        last inner solve's flux (true for ``power_iteration`` and for
        every manual solve-then-estimate loop).

        Raises
        ------
        RuntimeError
            If a vacuum face exists but no inner solve has stored a
            boundary trace yet — the leakage term cannot be answered
            honestly, and answering without it would silently reproduce
            the #291 omission (fail loud; never return a non-eigenvalue).
        """
        # A face LEAKS iff its law returns nothing: R = 0 means every particle
        # crossing outward is lost to the k denominator. That is
        # ``response_kernel.is_zero``, asked of the law directly; until
        # campaign phase B2 it read ``op.kind == "vacuum"``, the same question
        # spelled as a string because the pre-B2.0 shim discarded the law.
        # Agreement is exact on SN's admitted set (vacuum R = 0, reflective
        # R = 1) and on every law but ONE: a prescribed-inflow face also leaks
        # its whole outflow, so it now joins this list where the tag test
        # missed it. That is the correct answer and it is unreachable today
        # (SN's admission table is {reflective, vacuum}), but it IS a
        # divergence — recorded rather than left silent.
        #
        # Known incompleteness, PRE-EXISTING and unchanged here: a partially
        # reflecting face (R = α < 1) leaks (1 − α) of its outflow and is in
        # neither the old set nor this one. It is unreachable because
        # ``_law_from_tag`` hard-codes albedo = 1.0 for reflective, and the
        # filter is an optimization rather than a semantic gate — the term it
        # skips is ``trace.net_current(face)``, which is identically zero for
        # the perfect reflector this list is really excluding. The honest
        # predicate is "R != 1", and it becomes reachable the moment #189
        # admits partial reflectors.
        leaking_faces = [
            name for name, op in self.sn_mesh.bc.items()
            if op.law.response_kernel.is_zero
        ]
        if not leaking_faces:
            return 0.0
        psi = None if self._inner is None else self._inner.iterate
        phi_of_trace = getattr(self, "_phi_of_trace", None)
        if psi is None or phi_of_trace is None:
            raise RuntimeError(
                "compute_keff on a vacuum-bounded problem needs the "
                "boundary trace of an inner solve (call "
                "solve_fixed_source first): the leakage term of the k "
                "denominator is read from psi.boundary, and answering "
                "without it would silently drop leakage (#291)."
            )
        rate = 0.0
        trace = _system_a_member(psi).boundary  # System A carries the trace
        # #289-F2 role parse: the widened FullField.boundary slot erases the
        # family; the SN leakage read needs the ANGULAR trace's net_current.
        if not isinstance(trace, AngularBoundaryFlux):
            raise TypeError(
                f"compute_keff: the converged iterate's trace must be an "
                f"AngularBoundaryFlux; got {type(trace).__name__}."
            )
        for face in leaking_faces:
            net_current = trace.net_current(face)  # (ng, *face_spatial[, moments])
            face_area = self._face_area_of(face)
            if net_current.ndim - 1 != np.ndim(face_area):
                raise NotImplementedError(
                    f"boundary leakage on {face!r}: the trace carries a "
                    f"transverse face-moment tail (net current shape "
                    f"{net_current.shape} vs face measure shape "
                    f"{np.shape(face_area)}, #251). The face integral "
                    f"consumes ONLY the transverse-average moment — higher "
                    f"Legendre face moments integrate to zero over each "
                    f"face cell — so wire the slot-0 read when the first "
                    f"multi-moment vacuum eigenvalue consumer arrives."
                )
            rate += float(np.sum(net_current * face_area))
        reference = IntegratedReactionRate(
            self.mat_xs.fission_production_field
        ).evaluate(phi_of_trace)
        if reference <= 0.0:
            raise RuntimeError(
                "leakage scale bridge is degenerate: the last inner "
                "solve's flux carries non-positive fission production."
            )
        return rate * (fission_production / reference)

    def _face_area_of(self, face: str) -> "float | np.ndarray":
        r"""Spatial measure of a boundary face, matching ``volume_measure``.

        1-D: the scalar face area from
        :attr:`~orpheus.transport.mesh.material_mesh.MaterialMesh.areas`
        — ``1.0`` (slab, per unit cross-section), :math:`2\pi R`
        (cylinder, per unit height), :math:`4\pi R^2` (sphere) — the
        same geometric convention the cell volumes integrate under, so
        the balance identity closes.

        d ≥ 2 Cartesian: the per-face-cell transverse measure

        .. math::

            A_{\mathbf{c}} \;=\; \prod_{j \ne a} \Delta_j[c_j]

        — the outer product of the OTHER axes' edge widths in
        **ascending axis order**, the same codimension-1 enumeration as
        :func:`~orpheus.transport.mesh.axis.face_shape`, so the array
        broadcasts cell-for-cell against the ``(ng, *face_spatial)``
        net current (2-D: the single transverse width vector, unit
        depth; 3-D: the ``(n_t0, n_t1)`` transverse-area product —
        the #291 estimator's d=3 arm). Equivalent to the boundary
        layer's ``volumes / Δ_axis`` — the object-level pin in
        ``tests/sn/eigenvalue/test_keff_estimator_gate.py``.
        """
        mesh = self.sn_mesh
        # One parse of the face name yields BOTH halves of its outward normal.
        # Until **B3.4c** this read the axis off a hand-written ``{"x": 0, ...}``
        # literal and the endpoint off a ``face == "xmin"`` compare — two
        # transcriptions of a convention with a single home.
        axis_index, outward_sign = face_normal(face)
        if mesh.ndim == 1:
            areas = mesh.areas
            return float(areas[-1] if outward_sign > 0 else areas[0])
        transverse_widths = (
            np.diff(np.asarray(mesh.axes[j].edges, dtype=float))
            for j in range(mesh.ndim)
            if j != axis_index
        )
        return reduce(np.multiply.outer, transverse_widths)

    def measure_stopping_criteria(
        self, keff: float, keff_old: float,
        flux_distribution: np.ndarray, flux_old: np.ndarray,
    ) -> tuple[StoppingCriterion, ...]:
        r"""``|Δk|`` against ``keff_tol`` and relative ``‖Δφ‖₂`` against ``flux_tol``.

        ⛔ Until 2026-08-09 (#340 N2b) this was ``converged(...) -> bool``: it
        computed both magnitudes, compared both, and returned one bit.  ``dphi``
        died here — which is why the truncation warning could only ever project
        off ``|Δk|``, and on a solve whose ``|Δk|`` had cleared while ``dphi``
        alternated in sign forever it dutifully answered "you need 1 more
        iteration" (`[M]` the mutated heterogeneous slab, ``|Δk| = 3.3e-16``
        against ``tol = 1e-9``).  Reporting BOTH is what retires that guess.
        """
        return (
            StoppingCriterion.reading(
                "dk", float(abs(keff - keff_old)), self.keff_tol,
            ),
            StoppingCriterion.reading(
                "dphi",
                float(
                    np.linalg.norm(flux_distribution - flux_old)
                    / max(np.linalg.norm(flux_distribution), 1e-30)
                ),
                self.flux_tol,
            ),
        )

    # ── Inner solver: source iteration ────────────────────────────────

    def _solve_source_iteration(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        r"""Inner within-group solve via :class:`SourceIteration` on typed AngularFlux.

        Carved onto :class:`~orpheus.numerics.iteration.SourceIteration`
        consuming the same typed-flux operator triple as the Krylov path:

        .. math::

            A \;=\; L + C\,, \quad
            S \;=\; \tfrac{1}{W}\,\text{full multi-group scatter}\,, \quad
            F \;=\; 0_{\rm wg}

        on :class:`~orpheus.transport.fields.angular_flux.AngularFlux`.  The
        iteration step is

        .. math::

            \psi_{n+1} \;=\; (L + C)^{-1}\,\bigl(S\,\psi_n
                                                + F\,\psi_n
                                                + q_{\rm ext}\bigr)

        where the driver applies ``(L + C).inverse()`` — a
        :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` whose
        ``apply`` IS the WDD sweep (#226 taxonomy step 3; the solver
        builds the inverse, :class:`SourceIteration` applies it).  The
        previous iterate :math:`\psi_n` travels into the sweep via the
        explicit ``initial_guess`` kwarg on the inverse's ``apply``; the
        M-M closure's ``psi_half_seed`` strategy reads it to derive the
        curvilinear Carlson coupled-pole seed (pinned by the
        seed-threading spy).

        Scope
        =====

        ALL geometries — slab, sphere, cylinder, AND 2-D Cartesian.  The
        eigenvalue SI inner is geometry-agnostic: it is the structural twin
        of :meth:`_solve_krylov` — identical composite RHS, identical loss
        decomposition (``LC = StreamingOperator + MultiplicationOperator``,
        the collision multiplier ``C = M[σ_t]``, plus the scattering ``S``
        and boundary ``B`` coupling gains via
        :func:`~orpheus.sn.coupled_system.build_within_group_system`, zero
        within-group fission), identical
        ``psi_typed.interior.integrate_angular()`` reduction — differing
        ONLY in the driver (:class:`SourceIteration` vs
        :class:`KrylovAcceleration`), neither of which carries geometry
        dependence.  The reflective coupling rides the bare sweep via the
        ``B`` coupling gain on the 4-face
        :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
        (:class:`SNBoundaryOperator` is natively 4-face, the SAME operator
        the 2-D Krylov path uses).

        Verified SI ≡ Krylov ≡ closed-form ``k_inf`` in
        ``tests/sn/eigenvalue/test_keff_2d.py`` (the values + the
        ERR-026 / ERR-058 curvilinear-closure history are recorded in the
        SN verification theory page).
        """
        from orpheus.transport.fields.angular_flux import (
            AngularFlux,
        )
        from orpheus.transport.fields.angular_boundary_flux import (
            AngularBoundaryFlux,
        )

        # ── Build the within-group system (single source of truth —
        # :func:`~orpheus.sn.coupled_system.build_within_group_system`;
        # shared with the Krylov and fixed-source paths; the solver's
        # cached scattering operator injects through the cache seam).
        # ``#218``: the eigenvalue inner honours ``self.inner_schedule``
        # (default boundary-G-S on 2-D Cartesian; the coupled arm and 1-D
        # fall to Jacobi structurally).  Phase-5a angular-windowing folds
        # in via :func:`_maybe_window` inside the SI builder. ──────────
        system = build_within_group_system(
            self.sn_mesh, self.mat_xs, scattering_op=self.scattering_op,
            n2n_op=self.n2n_op,
        )
        si, _base, _gains, windowed = _within_group_si(
            system, self.sn_mesh,
            inner_schedule=self.inner_schedule,
            max_iter=self.max_inner, tol=self.inner_tol,
        )
        # B.2d DP-seedless: the coupled pair appears exactly where System B
        # exists; the seedless paths (windowed 2-D, G-S) stay fused.
        coupled = isinstance(system.implicit_operator, CoupledOperator)

        # ── Warm start (composite / coupled pair) ───────────────────
        # SourceIteration threads the previous iterate to the inverse
        # operator's ``apply`` via the explicit ``initial_guess`` kwarg
        # (accepted-and-dropped by the direct sweeps since #282/2.5d; the
        # previous iterate's boundary trace seeds the reflective-BC
        # partner-flux state — pinned by the seed-threading spy).
        # Post-B.2d the previous inner's iterate (``self._inner.iterate``)
        # is a CoupledField on a carrying mesh (the converged pair — already
        # split) and a :class:`TimedFullField` elsewhere; both propagate
        # through the iteration primitive via the ravellable protocol.
        initial_guess = None if self._inner is None else self._inner.iterate
        if initial_guess is None:
            # B.5.2: cold-start iterate is an all-zeros FLUX composite,
            # decoupled from q_ext's AngularSourceSink type.  Phase 5a: when
            # windowed the bulk is a zero HarmonicMomentFlux (single-sourced
            # in :func:`_windowed_cold_start`); else a zero AngularFlux —
            # paired NATIVE with a zero ψ_B on a carrying mesh (B.2d).
            if windowed:
                initial_guess = _windowed_cold_start(
                    self.scattering_op, self.sn_mesh, history_depth=2,
                )
            else:
                cold = _unwindowed_cold_start(self.sn_mesh, history_depth=2)
                initial_guess = (
                    _coupled_flux_state(cold, self.sn_mesh) if coupled else cold
                )

        # The driver's rhs — the ONE construction site the finalize shares
        # (:func:`_eigenvalue_driver_source`): the fission source lifted to
        # the composite, paired with its ψ½ fold on a carrying mesh (#282
        # route (a) — System B's member, never a composite block, B.2d).
        q_driver = _eigenvalue_driver_source(
            fission_source, self.sn_mesh,
            context="SNSolver._solve_source_iteration",
        )
        psi_typed, record = si.solve(
            q_driver, initial_guess=initial_guess,
        )
        # The end-of-solve CERTIFICATE (step 5, R-5.2) — full-angular arms
        # only (the windowed moment arm is structurally exempt: seedless ⟹
        # no in-M lag surface; see _certify_within_group_exit).
        if not windowed:
            _certify_within_group_exit(
                system, psi_typed, q_driver,
                sn_mesh=self.sn_mesh, record=record,
                where="SNSolver._solve_source_iteration",
            )
        # Keep this outer step's inner record whole.  It used to be reduced to
        # ``+= record.n_iterations`` right here — the count kept, the criteria,
        # the rate and the status thrown away one line after the driver had
        # gone to the trouble of measuring them (#340 F8).
        # ⭐ The count that survives is ``record.n_iterations``, the loop's OWN
        # pass count, not ``len(trajectory)`` — SI measures the difference
        # between successive iterates, so the trajectory is one short (F10).
        self.inner_records.append(record)
        # What this solve DROVE and what it converged to, as ONE record —
        # the next inner's warm start; the finalize evaluates the same map
        # once more on it (#448).
        self._inner = InnerSolve(system, _base, _gains, psi_typed)

        # Scalar flux for the eigenvalue outer's contract.  Windowed: the
        # ℓ=0 moment IS the scalar flux (Y_0^0 = 1 ⇒ bit-identical to
        # integrate_angular) via the typed ``scalar_flux`` accessor that
        # carries the convention; un-windowed: reduce the full angular bulk.
        # The isinstance parses reify the driver-template contract (the
        # solve echoes the initial_guess representation) — the static
        # iterate type is the operators' ``FullField`` carrier, so the
        # bulk's representation re-narrows here, loudly on mismatch.
        from orpheus.transport.fields.harmonic_moment_flux import (
            HarmonicMomentFlux,
        )

        bulk = _system_a_member(psi_typed).interior
        if windowed:
            if not isinstance(bulk, HarmonicMomentFlux):
                raise TypeError(
                    f"windowed SI iterate must carry a HarmonicMomentFlux "
                    f"bulk (the moment template); got {type(bulk).__name__}."
                )
            phi = bulk.scalar_flux().values
        else:
            if not isinstance(bulk, AngularFlux):
                raise TypeError(
                    f"un-windowed SI iterate must carry an AngularFlux bulk "
                    f"(the flux template); got {type(bulk).__name__}."
                )
            phi = bulk.integrate_angular().values
        # The scale partner of the stored boundary trace (#291): the
        # leakage term of ``compute_keff`` rescales the trace's net
        # current by the fission-production ratio of the estimator's
        # flux to THIS flux (exactly 1.0 when the caller passes it back).
        self._phi_of_trace = phi
        return phi

    # ── Inner solver: Krylov on (L+C-S)·ψ = q_ext (R-1 Step D carve) ──

    def _solve_krylov(
        self, fission_source: np.ndarray, flux_distribution: np.ndarray,
    ) -> np.ndarray:
        r"""Inner solve via GMRES on :math:`(L + C - S)\,\psi = q_{\rm ext}`.

        Carved onto :class:`~orpheus.numerics.iteration.KrylovAcceleration`
        consuming the operator triple

        .. math::

            A \;=\; L + C\,, \quad
            S \;=\; \text{full multi-group scatter}\,, \quad
            F \;=\; 0_{\rm wg}

        on typed :class:`~orpheus.transport.fields.angular_flux.AngularFlux`.  The
        composite ``L + C`` returns an
        :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` (R-1 Step C);
        its ``.solve`` IS the WDD sweep but R-1 ships GMRES
        UNPRECONDITIONED (``preconditioner=None``) per user direction
        ("consolidating the foundational architecture; the block-inverse
        face preconditioner is `issue #200
        <https://github.com/deOliveira-R/ORPHEUS/issues/200>`_").  The
        sweep-as-preconditioner reactivation lives there.

        The within-group fission ``F`` is zero — the fission source
        comes in as the external ``q_{\rm ext}`` per the eigenvalue
        outer/within-group inner decomposition (Lewis & Miller §6.4).

        Scope
        =====

        2-D Cartesian is supported: the 4-face
        :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
        face descriptor (xmin / xmax / ymin / ymax) and the L2-native
        ``loss_action`` 2-D matvec walk operate directly on it.

        Returns the updated scalar flux ``(ng, nx, ny)``.
        """

        from orpheus.transport.fields.angular_flux import (
            AngularFlux,
        )
        from orpheus.transport.fields.angular_boundary_flux import (
            AngularBoundaryFlux,
        )
        from orpheus.transport.timed_full_field import TimedFullField

        # ── Build the within-group system (single source of truth —
        # :func:`~orpheus.sn.coupled_system.build_within_group_system` /
        # ``_within_group_krylov``; shared with the SI and fixed-source
        # paths; the cached scattering operator injects through the cache
        # seam). ──────────────────────────────────────────────────────
        system = build_within_group_system(
            self.sn_mesh, self.mat_xs, scattering_op=self.scattering_op,
            n2n_op=self.n2n_op,
        )
        coupled = isinstance(system.implicit_operator, CoupledOperator)

        # ── Warm start (composite / coupled pair) — built BEFORE the
        # driver so the GMRES restart is sized from the FULL ravel. ───
        # Post-B.2d the previous inner's iterate (``self._inner.iterate``)
        # is a CoupledField on a carrying mesh and a TimedFullField
        # elsewhere; the Krylov ravellable protocol detects either via
        # ``to_flat`` / ``from_flat`` (D-H.1b.1) and threads it through the
        # matvec / unravel cycle natively.
        initial_guess = None if self._inner is None else self._inner.iterate
        if initial_guess is None:
            # B.5.2: cold-start iterate is a FLUX composite, decoupled from
            # q_ext's now-AngularSourceSink type.  x0 stays all-zeros
            # (bit-identical); the flux template fixes the Krylov return
            # type — paired NATIVE with a zero ψ_B on a carrying mesh (B.2d).
            cold = TimedFullField.zeros(
                interior=AngularFlux, boundary=AngularBoundaryFlux, space=self.sn_mesh.full_field_space,
            )
            initial_guess = (
                _coupled_flux_state(cold, self.sn_mesh) if coupled else cold
            )

        # ERR-053 (#282 route (a)): ``restart`` MUST cover the FULL ravel —
        # bulk ⊕ trace on System A plus BOTH System-B legs on the coupled
        # pair (``CoupledField.to_flat`` concatenates the systems, so the
        # sizing tracks automatically — the B.2a conformance closure; the
        # count is HONEST since B.2d, no dead padding).  A bulk-sized
        # restart re-truncates GMRES on the trace+seed DOFs (the sphere
        # Krylov stall).  Size it from the state the driver ravels.
        krylov = _within_group_krylov(
            system.implicit_operator, *system.explicit_gains,
            n_dof=int(initial_guess.to_flat().size),
            max_iter=self.max_inner, tol=self.inner_tol,
        )

        q_driver = _eigenvalue_driver_source(
            fission_source, self.sn_mesh,
            context="SNSolver._solve_krylov",
        )
        psi_typed, record = krylov.solve(
            q_driver, initial_guess=initial_guess,
        )
        # The end-of-solve CERTIFICATE (step 5, R-5.2) — the Krylov path is
        # always full-angular (windowing is SI-only), so it certifies
        # unconditionally: GMRES's own stop is residual-based, but the
        # certificate is the honest cross-check on the ASSEMBLED equation
        # (the ERR-053 truncation family's independent catcher).
        _certify_within_group_exit(
            system, psi_typed, q_driver,
            sn_mesh=self.sn_mesh, record=record,
            where="SNSolver._solve_krylov",
        )
        # Keep this outer step's inner record whole (see the SI arm above).
        # GMRES gets one callback per iteration, so here the count and the
        # trajectory length agree — the opposite of SI's offset, which is why
        # each driver states its own (#340 F11).
        self.inner_records.append(record)
        # What this solve DROVE and what it converged to, as ONE record —
        # GMRES on ``(M − N)ψ = q`` with the record's own splitting; the
        # finalize evaluates ``M⁻¹(q + N·ψ)`` once on it (#448).
        self._inner = InnerSolve(
            system, system.implicit_operator, system.explicit_gains, psi_typed,
        )

        # Reduce angular → scalar flux for the eigenvalue outer's contract.
        # The parse reifies the driver-template contract (the solve echoes
        # the flux initial_guess) loudly on mismatch.
        bulk = _system_a_member(psi_typed).interior
        if not isinstance(bulk, AngularFlux):
            raise TypeError(
                f"eigenvalue Krylov iterate must carry an AngularFlux bulk "
                f"(the flux template); got {type(bulk).__name__}."
            )
        # The scale partner of the stored boundary trace (#291) — see the
        # SI path's twin store.
        phi = bulk.integrate_angular().values
        self._phi_of_trace = phi
        return phi


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def solve_sn(
    materials: dict[int, Mixture],
    mesh: "Mesh1D | Mesh2D | tuple[Axis1D, ...]",
    quadrature: Quadrature,
    inner_solver: str = "source_iteration",
    scattering_order: int = 0,
    max_outer: int = 500,
    keff_tol: float = 1e-7,
    flux_tol: float = 1e-6,
    max_inner: int | None = None,
    inner_tol: float = 1e-8,
    inner_schedule: str = "jacobi",
    mat_map: "np.ndarray | None" = None,
) -> Solution:
    """Solve the multi-group SN eigenvalue problem.

    This is the **canonical entry point** for the production SN solver.
    Production callers consume ``(materials, mesh, quadrature, ...)``
    directly: materials are :class:`~orpheus.data.macro_xs.mixture.Mixture`
    objects keyed by material ID, ``mesh`` is a
    :class:`~orpheus.geometry.Mesh1D` / :class:`~orpheus.geometry.Mesh2D`
    (build via :meth:`Mesh1D.from_geometry` for multi-region 1-D cases)
    OR an axis tuple — the axis-native surface and the ONLY 3-D entry
    (C5.5, #225; per-axis BCs ride the axes, ``mat_map=`` carries the
    material assignment), and ``quadrature`` is an explicitly chosen
    :class:`~orpheus.numerics.quadrature.Quadrature` — Gauss-Legendre
    for slab, level-symmetric / product quadrature for curvilinear, or
    Lebedev for 2-D.

    The mesh's boundary conditions (``bc_left`` / ``bc_right`` for 1-D,
    ``bc_xmin`` / ``bc_xmax`` / ``bc_ymin`` / ``bc_ymax`` for 2-D) are
    honoured verbatim — the SN sweep handles ``vacuum`` and
    ``reflective``.

    Parameters
    ----------
    materials : dict mapping material ID to Mixture.
    mesh : Mesh1D or Mesh2D (base geometry).
    quadrature : Quadrature
        Explicitly chosen by the caller — Gauss-Legendre for slab,
        level-symmetric / product quadrature for curvilinear 1-D,
        Lebedev for 2-D. Mismatches between geometry and quadrature
        family are not silently coerced.
    inner_solver : "source_iteration" (default) or "krylov".
        Wave E Round 2 deviation from the campaign plan: ``solve_sn``
        keeps the ``source_iteration`` default for **all** geometries
        (Cartesian and curvilinear).  The eigenvalue is shape-
        independent (k = production / absorption is a volume-weighted
        ratio), so even on the ERR-026-affected curvilinear sweep the
        keff is correct to the eigenvalue's tolerance, even though the
        flux *shape* would have a closure-bias drift.  Preserving the
        default keeps the 6 curvilinear regression snapshots bit-
        identical.  ``solve_sn_fixed_source`` *does* auto-flip to
        ``"krylov"`` for curvilinear because shape correctness is the
        whole point of fixed-source MMS verification.
    scattering_order : Legendre order for scattering (0 = P0).
    max_outer : maximum outer (power) iterations.
    keff_tol, flux_tol : outer convergence.
    max_inner, inner_tol : inner solver parameters.
    inner_schedule : {"jacobi", "gauss_seidel"}
        Boundary splitting for the ``source_iteration`` inner (#218 — the
        eigenvalue inner CAN now reach the same boundary-G-S accelerator the
        fixed-source path got in Phase 3, via the shared ``_within_group_si``
        builder; validated SI(G-S)≡Krylov≡closed-form k_inf).  ``"jacobi"``
        (default) lags the reflective boundary ``B`` as an external gain;
        ``"gauss_seidel"`` (opt-in) folds ``B`` into an octant-group forward
        substitution on 2-D Cartesian (1-D / curvilinear auto-fall-back to
        Jacobi — G-S is 2-D-Cartesian-only and a no-op on the
        scattering-dominated 1-D regime).  The converged eigenvalue is
        identical either way to within ``inner_tol`` (vv-principles Mode 9 —
        same fixed point; only the inner SI stopping differs).  Default stays
        Jacobi: the eigenvalue inner is warm-started (modest G-S benefit) and a
        schedule change shifts k_eff by ~inner_tol, which the keff_tol-tight
        regression snapshots cannot absorb.  Ignored when
        ``inner_solver="krylov"`` (Krylov is splitting-invariant).

    Returns
    -------
    Solution
        Typed return carrying eigenvalue, typed
        :class:`~orpheus.transport.fields.angular_flux.AngularFlux` +
        :class:`~orpheus.transport.fields.scalar_flux.ScalarFlux` +
        :class:`~orpheus.transport.fields.angular_boundary_flux.AngularBoundaryFlux`
        fields plus an
        :class:`~orpheus.sn.solution.IterationHistory` carrying the
        eigenvalue trajectory.  The unified :class:`Solution` type covers
        both eigenvalue and fixed-source problems.
    """
    t_start = time.perf_counter()

    # Build augmented geometry (precomputes streaming stencil).
    # Issue #197 PR-TYPED-0: materials now lives on SNMesh — the
    # phase-space-as-such object. C5.5 (#225): the declaration may be a
    # legacy mesh or an axis tuple (the only 3-D entry); unset faces
    # resolve to the SNMesh reflective default (eigenvalue convention).
    sn_mesh = _as_sn_mesh(mesh, quadrature, materials, mat_map=mat_map)

    solver = SNSolver(
        sn_mesh,
        inner_solver=inner_solver,
        scattering_order=scattering_order,
        keff_tol=keff_tol, flux_tol=flux_tol,
        max_inner=max_inner, inner_tol=inner_tol,
        inner_schedule=inner_schedule,
    )

    outcome = power_iteration(solver, max_iter=max_outer, budget_name="max_outer")
    keff, keff_history, scalar_flux = (
        outcome.keff, outcome.keff_history, outcome.flux_distribution,
    )

    # ── The returned angular flux: ONE source-iteration step from the
    # converged iterate (#448). ─────────────────────────────────────────
    #
    # The power iteration converged ``(k, φ)`` on the inner solves' iterate
    # ``solver._inner.iterate`` — the full within-group state: System A's
    # (bulk ⊕ trace), paired with System B's ψ½ member on a carrying mesh
    # (B.2d), the bulk a ``HarmonicMomentFlux`` when the 2-D Cartesian inner
    # was windowed.  The flux the caller receives is that iterate polished
    # ONCE against the CONVERGED fission source: the splitting map
    # ``G(ψ) = M⁻¹(q_F(φ, k) + Σ Nᵢ·ψ)`` evaluated at ψ_conv through the
    # splitting the last inner solve DROVE (``solver._inner`` — the same
    # record, the same forward ``M``, the same gains ``(S, N₂ₙ, B)``:
    # moment-bound when the iterate is windowed, the coupled gain grid when
    # carrying), so the source the reconstruction sees IS the source the
    # iteration converged against, at every scattering order, by
    # construction.  ``M⁻¹`` here is the UN-windowed full-angular inverse
    # (``inner.implicit`` is the un-wrapped forward ``M``), which is what
    # turns a moment iterate back into per-ordinate ψ — the fixed-source
    # windowed arm's spelling (:func:`_solve_fixed_source_si`), now the one
    # body :func:`~orpheus.numerics.iteration.fixed_point_step`.
    #
    # ⛔ Until #448 this block hand-built the source as fission + P0
    # scattering + P0 (n,2n) and lifted it isotropically — the ℓ ≥ 1 half of
    # BOTH channels silently dropped at every ``scattering_order ≥ 1``, so
    # the returned ψ solved a different equation from the one the power
    # iteration converged and did not reduce to the ``scalar_flux`` shipped
    # beside it.  `[M]` on the Be-reflected U slab (421 g, GL-8, P2) the
    # returned ψ missed the converged iterate by 8.8e-2 and its own moments
    # missed the reported φ by 3.4e-2; this step reproduces the iterate to
    # 1.2e-10 and its moments reproduce φ to 3.2e-10
    # (``tests/sn/solve/test_eigenvalue_finalize_reconstruction.py``;
    # error-catalogue entry ERR-083).  The reflective coupling arrives as
    # the ``B`` gain, exactly as in every inner solve; the hand reflect of
    # the converged trace this block used to perform first was `[M]` INERT
    # on a converged exit (2.0e-13 / 2.3e-15 / bit-identical on a vacuum
    # arm — the pre-carve battery's M5), so its removal moves nothing a
    # value gate can witness; the trace's correctness is pinned by the
    # wrong-``B`` arm (M5b) instead.
    from orpheus.numerics.iteration import fixed_point_step

    inner = solver._inner
    if inner is None:
        # Reachable: ``max_outer=0`` runs the power loop zero times (`[M]`
        # the review round fired it), so this is a live refusal of a legal
        # call, not decoration.
        raise RuntimeError(
            "solve_sn finalize: the power iteration returned without a "
            "within-group solve to reconstruct from — the finalize is one "
            "step of the iteration, not a cold solve (max_outer must be ≥ 1)."
        )
    # The SYSTEM is kept, not just its splitting: the #340 N6b exit balance
    # below needs its LOSS arm (``L+C−S−N₂ₙ−B``).
    final_system = inner.system
    # The inverse's static type is the UNION of the three resolvent classes
    # while the iterate's is the union of the two state carriers, and the
    # correlation between them (coupled ↔ coupled) is not expressible
    # without a seeded-inverse Protocol (#453) — the cast states that the
    # record pairs them by construction (both written at one site).
    final_state = fixed_point_step(
        cast("SupportsSeededApply[Any]", inner.implicit.inverse()), inner.gains,
        _eigenvalue_driver_source(
            solver.compute_fission_source(scalar_flux, keff), sn_mesh,
            context="solve_sn finalize",
        ),
        inner.iterate,
    )
    final_psi_a = _system_a_member(final_state)
    # ``None`` on a bare seedless composite by construction — the reader
    # spells the partition, not a flag beside it.
    final_ray = _system_b_member(final_state)
    elapsed = time.perf_counter() - t_start

    # The shared packaging tail (:func:`_package_solution`) owns the
    # composite-carrier convention; the scalar member is the power
    # iteration's converged scalar (SCALAR-AGNOSTIC contract).  The
    # boundary trace is the converged one from the final resolvent solve
    # (Wave O #208 O.4a.2: inflow = B·ψ.outflow seeded through the rhs,
    # outflow = streamed); System B's converged state is its OWN member
    # (B.2d — the marched ψ½ composite; None on non-carrying meshes).
    from orpheus.transport.fields.scalar_flux import ScalarFlux
    # The record IS the history (#340 N2b-ii).  Everything this used to
    # spell out by hand — ``converged``, ``n_outer``,
    # ``total_inner_iterations`` — is now READ from the tree the loop built,
    # so no producer can write a verdict and none can disagree with another
    # about what an iteration count means.  ``keff_history`` stays explicit:
    # it is a physics output, not a stopping criterion.
    # #340 N6b — the exit balance defect of the iterate the user RECEIVES.
    #
    # The rhs is REBUILT from the returned ψ (``Fφ(ψ)/k``), not taken from
    # the fission source the reconstruction step consumed, and the
    # distinction is not cosmetic: the two answer different questions.  That
    # source came from the power iteration's converged scalar, so it would
    # measure how well the RECONSTRUCTED angular flux solves the equation the
    # reconstruction was GIVEN; recomputing from ``final_psi_a`` asks the
    # closed question — does the object I was handed satisfy its own
    # equation?  The second is what `scratch/n5_outer_cert_lib.py` measured
    # the 4.64× against, and taking the first would silently substitute its
    # ``defect_pi`` variant.  `[M]` the rebuild costs 0.05 ms.
    #
    # ⛔ NOT the step's full lagged source ``q_F + Σ Nᵢ·ψ``: that is the
    # TOTAL reconstruction source (fission + scattering + (n,2n) + B),
    # `[M]` 13.7× larger than the fission source alone, and against
    # ``A = L+C−S−B`` it double-counts scattering.
    # ⛔ CARRYING MESHES ARE EXEMPT HERE, and the reason is a real refusal
    # rather than a missing feature on our side.  What this site assembles
    # IS bare (a System-A ψ against a System-A fission rhs), which on a
    # carrying mesh would silently omit r_B — the Mode-12 blindness the
    # split-residual mint exists to prevent — so the honest answer is "no
    # number", not a residual missing a block.  (CS4b S4: the refusal used
    # to live inside `evaluate_residual` as a mesh read; the full-system
    # entry now refuses by the POSED system's arity, and this deliberate
    # arm-level evaluation routes through the guard-free `_typed_balance`
    # primitive — so the exemption below is this call site's OWN honest
    # reasoning, kept, not a downstream raise avoided.)
    #
    # `[M]` 2026-08-10, and it is why this exemption exists at all: without
    # it the slice went 9 → 25 reds, every one of the 16 a curvilinear
    # solve raising out of `evaluate_residual`.  The fixed-source arms are
    # unaffected — they already pass the COUPLED pair when the mesh carries.
    # Extending this entry to do the same needs the coupled rhs (the
    # fission source AND the seed source, through `_coupled_source_state`);
    # tracked as #354 rather than assembled here from plausibility.
    #
    # ⭐ Worth naming, because it is why no existing gate caught it in
    # review: `_certify_within_group_exit` calls the same function on the
    # same meshes and has never hit this, because it is guarded on
    # `record.converged` and returns early on exactly the truncated solves
    # this runs on.  The complement of a guard reaches the states its
    # partner never visits.
    from orpheus.transport.source_sinks import (
        AngularBoundarySourceSink,
        AngularSourceSink,
    )

    balance_defect = None
    if sn_mesh.radial_characteristic_field_space is None:
        exit_rhs = FullField(
            interior=AngularSourceSink.from_isotropic(
                solver.compute_fission_source(
                    _angular_moment_values(final_psi_a), keff,
                ),
                sn_mesh,
            ),
            boundary=AngularBoundarySourceSink.zeros(sn_mesh.angular_trace),
        )
        balance_defect = _exit_balance_defect(
            _bare_loss_arm(final_system),
            final_psi_a, exit_rhs, sn_mesh=sn_mesh, record=outcome.record,
        )
    # #344 — AFTER the balance defect, so the number reported describes the
    # object the caller receives (residual-neutral either way; see the helper).
    final_psi_a, gauge_correction = _exit_gauge_trace(
        final_psi_a, sn_mesh=sn_mesh,
    )
    history = IterationHistory(
        record=outcome.record, keff_history=tuple(keff_history),
        balance_defect=balance_defect,
        gauge_correction=gauge_correction,
    )
    warn_if_unconverged(
        history.record, where="solve_sn",
        balance_defect=history.balance_defect,
    )
    warn_if_gauge_freedom(
        sn_mesh, history.gauge_correction, where="solve_sn",
    )
    return _package_solution(
        _cell_average_angular(final_psi_a, sn_mesh),
        final_psi_a.boundary,
        final_ray,
        sn_mesh,
        scalar=ScalarFlux(values=scalar_flux, space=sn_mesh.bulk_space),
        keff=float(keff_history[-1]),
        history=history,
        cls=Solution,
    )


# ═══════════════════════════════════════════════════════════════════════
# The Solution packaging tail — ONE convention, forward + adjoint
# ═══════════════════════════════════════════════════════════════════════


def _cell_average_angular(
    field: "FullField", sn_mesh: SNMesh,
) -> "AngularFlux":
    r"""The :class:`Solution` angular carrier: the CELL-AVERAGE view.

    A multi-moment closure's φ̂ tail is iterate-internal within-cell DG
    structure (#240 D5b-S3) — the user-facing angular flux is the
    ``AVERAGE_MOMENT`` slot, extracted through the one moment-slot
    single source (:func:`_average_moment_scalar`, layout-generic over
    the trailing moment axis) and wrapped once into the typed field.
    """
    from orpheus.transport.fields.angular_flux import AngularFlux

    bulk = _average_moment_scalar(
        np.asarray(field.interior.values), sn_mesh,
    )
    return AngularFlux(values=bulk, space=sn_mesh.angular_bulk_space)


SolutionT = TypeVar("SolutionT", bound=SolutionBase)


def _package_solution(
    angular: "AngularFlux",
    boundary,
    ray,
    sn_mesh: SNMesh,
    *,
    scalar: "ScalarFlux",
    keff: "float | None",
    history: IterationHistory,
    cls: "type[SolutionT]",
) -> SolutionT:
    r"""The CELL-AVERAGE :class:`SolutionBase` construction convention.

    Where the eigenvalue and adjoint entries turn converged iterates into the
    typed return (#197 PR-TYPED-5): the cell-average angular view + the
    converged boundary trace wrap into the ``TimedFullField`` composite carrier
    (D-H.1c stage 2 — ``_history=()``, ``history_depth=2``), alongside the
    scalar member, eigenvalue, iteration history, and System B's ray member
    (``None`` on non-carrying meshes, B.2d).

    ⛔ **This docstring read "The ONE ``SolutionBase`` construction convention …
    spelled HERE and nowhere else" until 2026-08-15. That was present-tense
    FALSE**, and it is the kind of falsehood that costs a later change real
    work: it invites installing a cross-cutting hook here and believing every
    entry is covered.

    `[M]` **3 of the 4 public entries route through this** —
    :func:`solve_sn` directly, and both adjoints via
    :func:`_package_adjoint_solution`. The **fixed-source family bypasses it
    entirely, once per arm**, building ``Solution(...)`` inline in
    :func:`_solve_fixed_source_si` and :func:`_solve_fixed_source_krylov`.

    The bypass is **deliberate, not drift**, and unifying it would be a
    regression: this tail routes the bulk through :func:`_cell_average_angular`,
    which strips a multi-moment closure to its ``AVERAGE_MOMENT`` slot, whereas
    the fixed-source arms return ``angular_out`` **whole** — a DG closure's
    :math:`\hat\varphi` slopes are internal structure, not the scalar flux the
    ``Solution`` reports (see the note in ``_solve_fixed_source_si``, #240
    D5b-S3). Two conventions, because there are two different returns.

    ⟹ **anything that must reach every entry belongs at the entries, not
    here** — see :func:`_exit_balance_defect` (4 sites) and
    :func:`_exit_gauge_trace` (5, because the fixed-source family has two
    arms), each one named, single-sourced, and invoked per exit.

    SCALAR- and ROLE-AGNOSTIC by design: the caller supplies the scalar
    member (forward — the power iteration's converged scalar; adjoint —
    the w-reduction of the packaged angular) AND names the role leaf
    (``cls`` — :class:`Solution` forward, :class:`AdjointSolution`
    adjoint; the A5 ruling made the role a TYPE), so the carrier
    convention stays single-sourced while the role never branches
    inside this shared tail.
    """
    return cls(
        angular_flux=TimedFullField(
            interior=angular,
            boundary=boundary,
            _history=(),
            history_depth=2,
        ),
        scalar_flux=scalar,
        mesh=sn_mesh,
        keff=keff,
        history=history,
        radial_characteristic=ray,
    )


# ═══════════════════════════════════════════════════════════════════════
# The ADJOINT entry family (#276 A4) — the daggered posing
# ═══════════════════════════════════════════════════════════════════════


def _adjoint_posing_parts(sn_mesh: SNMesh, scattering_order: int):
    r"""Shared build for the adjoint entries: the daggerable parts.

    Returns ``(implicit_operator, gain, F_posed, template)`` — the invertible
    within-group implicit operator ``M``, the summed coupling gain, the fission operator
    posed on the system's carrier, and a ZERO composite of that carrier
    (the shape/mesh template for guesses via ``from_flat``).  Everything
    comes off :func:`~orpheus.sn.coupled_system.build_within_group_system`
    (the ONE construction site) and is daggered by the CALLER with ``.H``
    — the entries never spell adjoint physics; the operator algebra is
    the implementation (#276 A4).

    On a carrying mesh (System B present) the carrier is the coupled
    pair: ``F`` is lifted as the composition ``[[F], [E]] ∘ r_bulk`` — a
    rectangular prolongation stack after the
    :class:`~orpheus.numerics.coupled_system.SystemRestrictionOperator`
    onto the bulk member (fission annihilates the ray system, so the
    lift's ray INPUT column is the restriction's, not a zero block; the
    within-group ray-coupled fission emission ``A_BA`` rides the forward
    outer's ``q_ext`` assembly, NOT the gain — HAZARD 5), and the gain is
    the builder's own coupled gain grid ``N``.
    """
    from orpheus.transport.fields.angular_boundary_flux import (
        AngularBoundaryFlux,
    )
    from orpheus.transport.fields.angular_flux import AngularFlux as _AF
    from orpheus.transport.operators.fission import FissionOperator

    mat_xs = sn_mesh.material_xs_field()
    system = build_within_group_system(
        sn_mesh, mat_xs, scattering_order=scattering_order,
    )
    gain = system.explicit_gains[0]
    for extra in system.explicit_gains[1:]:
        gain = gain + extra
    F = FissionOperator.from_solver_data(
        mat_xs=mat_xs, space=sn_mesh.full_field_space,
    )
    full_field_zero = FullField(
        interior=_AF.zeros(sn_mesh.angular_trial_space),
        boundary=AngularBoundaryFlux.zeros(sn_mesh.angular_trace),
    )
    if sn_mesh.radial_characteristic_field_space is None:
        return system.implicit_operator, gain, F, full_field_zero
    # Carrying mesh: pose F as (prolongation stack) ∘ (bulk restriction) —
    # the S4-amendment un-weld.  Fission annihilates the ray system (the
    # w = 0 closed rays carry no quadrature weight, so they never source
    # fission), and the honest spelling of that fact is the composition
    # ``F_posed = [[F], [E]] ∘ r_bulk``: the SystemRestrictionOperator
    # projects onto the bulk member, then the rectangular stack emits
    # both rows — ``F`` (bulk → bulk) and the FISSION ray fold
    # ``A_BA_fission = Fold ∘ F.isotropic_energy ∘ integrate``, the kernel-generic
    # :class:`RadialCharacteristicEmission` (the operator spelling of
    # :func:`_radial_characteristic_fission_seed`'s q-assembly math; on
    # the eigen-M operator this row BELONGS in the posing — HAZARD 5
    # keeps it out of the WITHIN-GROUP gain, not out of M).  No zero
    # blocks anywhere: the pre-amendment (B, B) hook-carrying
    # ``ZeroOperator`` existed only because an annihilated column cannot
    # be spelled ``None`` on a standalone grid, and its dagger's ray zero
    # now falls out of the restriction's extension-by-zero (minted
    # through the space's own zeros seam — #276 A4's SOURCE-classed
    # closure retired with the hooks).
    from orpheus.numerics.coupled_system import SystemRestrictionOperator
    from orpheus.sn.operators.radial_characteristic import (
        RadialCharacteristicEmission,
    )

    space = system.space
    restrict_bulk = SystemRestrictionOperator(space, system=0)
    stack = CoupledOperator(
        [
            [F],
            [RadialCharacteristicEmission(
                F.isotropic_energy,
                field_space=sn_mesh.radial_characteristic_field_space,
                full_field_space=sn_mesh.full_field_space,
                angular_bulk_space=sn_mesh.angular_bulk_space,
                angular_trace=sn_mesh.angular_trace,
                quadrature=sn_mesh.quad,
                coord=sn_mesh.coord,
            )],
        ],
        domain=restrict_bulk.codomain,
        codomain=space,
    )
    F_posed = stack @ restrict_bulk
    return system.implicit_operator, gain, F_posed, space.zeros()


def solve_sn_adjoint(
    materials: dict[int, Mixture],
    mesh: "Mesh1D | Mesh2D | tuple[Axis1D, ...]",
    quadrature: Quadrature,
    scattering_order: int = 0,
    max_outer: int = 500,
    keff_tol: float = 1e-7,
    flux_tol: float = 1e-6,
    max_inner: int | None = None,
    inner_tol: float = 1e-8,
    mat_map: "np.ndarray | None" = None,
) -> AdjointSolution:
    r"""Solve the multi-group ADJOINT SN eigenvalue problem.

    The adjoint criticality problem

    .. math::

        A_{\rm loss}^\dagger\,\psi^* \;=\; \frac{1}{k}\,F^\dagger\,\psi^*
        \qquad (A_{\rm loss} = L + C - S - N_{2n} - B)

    posed purely by DAGGER-ing the forward operator triple — the
    :class:`~orpheus.numerics.iteration.KEigenvalue` triple becomes
    ``((L+C).H, (S+N2N+B).H, F.H)`` — the daggered RESOLVENT, gain, and
    fission (the loss dagger :math:`A_{\rm loss}^\dagger =
    (L{+}C).\mathtt{H} - (S{+}N_{2n}{+}B).\mathtt{H}` is formed inside
    the posing) — and runs through the UNCHANGED canonical
    :func:`~orpheus.numerics.eigenvalue.power_iteration` (the adjoint row
    of the eigenvalue-posing table, live since #276 A4).  There is no
    adjoint-specific loop or sweep code anywhere: ``.H`` is the exact
    discrete Hilbert (G-metric) adjoint of every leaf — the reverse-scan
    transpose sweeps of #280/#310 behind ``A.H.inverse()``, the
    group-transpose scattering :math:`S^T` (#118), the group-transpose
    :math:`(n,2n)` emission :math:`(\nu_{2n}\Sigma_{2n}^{T})^{T} =
    \nu_{2n}\Sigma_{2n}` (the multiplicity is a scalar and rides the
    dagger unchanged — CS4c step 3), the χ↔νΣf fission role swap
    :math:`F^T` — so ``k_{\rm adj} = k_{\rm fwd}`` is an exact
    algebraic identity and :math:`\psi^*` is the true discrete adjoint
    (importance) flux, verified against the closed-form
    :math:`(\mathbf{A}^T)^{-1}\mathbf{F}^T` spectrum (NOT
    :math:`\text{eig}(M^T)` — the factor-order trap documented at
    :func:`~orpheus.derivations.common.eigenvalue.kinf_and_adjoint_spectrum_homogeneous`).

    Signature mirrors :func:`solve_sn` (the forward sibling); the
    daggered path has ONE inner realization (the transpose-sweep
    :class:`~orpheus.numerics.iteration.SourceIteration`), so the
    forward's ``inner_solver`` / ``inner_schedule`` strategy selectors do
    not appear.  Boundary conditions ride the mesh declaration exactly as
    in :func:`solve_sn` (unset faces resolve to the reflective eigenvalue
    default); reflective and vacuum are handled by the transpose
    machinery structurally — an adjoint vacuum is the transpose of the
    forward vacuum, never a user-facing BC flip.

    Returns
    -------
    AdjointSolution
        The role-typed return (the A5 carrier ruling): ``keff`` = the
        adjoint eigenvalue (== the forward eigenvalue to convergence
        tolerance), ``angular_flux`` = the adjoint angular flux
        :math:`\psi^*` (cell-average view), ``scalar_flux`` = the
        adjoint scalar flux :math:`\varphi^* = \sum_n w_n \psi^*_n`
        (the importance map — also readable as
        :attr:`~orpheus.sn.solution.AdjointSolution.importance`).
    """
    sn_mesh = _as_sn_mesh(mesh, quadrature, materials, mat_map=mat_map)
    implicit_operator, gain, F_posed, template = _adjoint_posing_parts(
        sn_mesh, scattering_order,
    )

    from orpheus.numerics.iteration import KEigenvalue

    ke = KEigenvalue(
        implicit_operator.H, gain.H, F_posed.H,
        max_outer=max_outer, keff_tol=keff_tol, flux_tol=flux_tol,
        max_inner=max_inner, inner_tol=inner_tol,
    )
    ones = np.ones(template.to_flat().size)
    guess = (
        CoupledField.from_flat(ones, template)
        if isinstance(template, CoupledField)
        else FullField.from_flat(ones, template)
    )
    outcome = ke.solve(initial_guess=guess)
    k_adj, keff_history, psi_star = (
        outcome.keff, outcome.keff_history, outcome.flux_distribution,
    )

    # The coupled unpack rides the canonical member readers (B.2d).
    system_a = _system_a_member(psi_star)
    adjoint_ray = (
        _system_b_member(psi_star)
        if isinstance(psi_star, CoupledField)
        else None
    )
    # Identical to the forward entry's spelling, and that is the point: the
    # two paths used to hand-write the same three facts in two different
    # ways (this one INFERRED convergence from ``len(keff_history) <
    # max_outer`` — wrong for a solve that converges on its last allowed
    # outer — and reported no inner total at all, because ``KEigenvalue``
    # discarded its inner trajectory inside ``numerics/``).  Both now read
    # the one tree ``power_iteration`` built (#340 N2b-ii).
    # ⛔ #340 N6b: this entry carries NO balance defect, and the omission is
    # deliberate rather than overlooked.  Every other entry's rhs is either
    # already in hand or rebuilt through a factory whose convention is
    # measured; the daggered eigenvalue rhs ``F_posed.H ψ*/k_adj`` would
    # have to be assembled here for the first time — the operator is right
    # there at :func:`_adjoint_posing_parts` — and since campaign 1 CS3
    # (2026-08-19) flux lives in V, so the ``1/k`` field scaling the old
    # affine-torsor note here called illegal is ordinary arithmetic (it
    # always was: scalar scaling was legal even under the torsor). The
    # real blocker stands independently: N5 never measured
    # the adjoint population, so there is no reference to check the result
    # against.  Assembling it from plausibility is exactly the ERR-032
    # class.  Tracked as #353; until then this path warns with everything
    # EXCEPT the number, which is why the clause is omitted rather than
    # printed as "unavailable".
    # #344 — wired for uniformity, and `[M]` INERT on every configuration this
    # entry can run: the adjoint routes through (L+C)^H, whose transpose solve
    # is 1-D-scan-only (#280 Phase 2.5b), and a 1-D problem has at most ONE
    # reflective axis pair, so `gauge_freedom(...).present` is False and this is
    # exactly the identity. ⚠ Do NOT read a green adjoint test as evidence the
    # gauge works — that is `inert`, not `verified`; the acceptance gate lives
    # on the forward entries.
    system_a, gauge_correction = _exit_gauge_trace(system_a, sn_mesh=sn_mesh)
    history = IterationHistory(
        record=outcome.record, keff_history=tuple(keff_history),
        gauge_correction=gauge_correction,
    )
    warn_if_unconverged(
        history.record, where="solve_sn_adjoint",
        balance_defect=history.balance_defect,
    )
    warn_if_gauge_freedom(
        sn_mesh, history.gauge_correction, where="solve_sn_adjoint",
    )
    return _package_adjoint_solution(
        system_a, adjoint_ray, sn_mesh,
        keff=float(k_adj),
        history=history,
    )


def solve_sn_adjoint_fixed_source(
    materials: dict[int, Mixture],
    mesh: "Mesh1D | Mesh2D | tuple[Axis1D, ...]",
    quadrature: Quadrature,
    detector_response: "np.ndarray | FullField",
    boundary_condition: "str | None" = "vacuum",
    scattering_order: int = 0,
    max_inner: int | None = None,
    inner_tol: float = 1e-12,
    mat_map: "np.ndarray | None" = None,
    scheme: "DiscretizationSchemeBase | None" = None,
) -> AdjointSolution:
    r"""Solve the multi-group ADJOINT SN fixed-source (importance) problem.

    .. math::

        A_{\rm loss}^\dagger\,\psi^* \;=\; q^* ,

    the detector-importance problem: :math:`\psi^*(\vec r, \Omega, g)` is
    the expected detector response per unit neutron introduced at phase-
    space point :math:`(\vec r, \Omega, g)`.  Solved by the daggered
    within-group :class:`~orpheus.numerics.iteration.SourceIteration`
    (``seeded_inverse(A.H)`` + the daggered gain) — the exact transpose
    of the forward fixed-source system; no adjoint-specific solver code.

    Signature mirrors :func:`solve_sn_fixed_source` (the forward
    sibling) for every shared parameter (``materials``, ``mesh``,
    ``quadrature``, ``boundary_condition``, ``scattering_order``,
    ``max_inner``, ``inner_tol``, ``mat_map``, ``scheme``); only the
    adjoint source is new.

    Parameters
    ----------
    detector_response : (ng, \*spatial) ndarray OR FullField
        The adjoint source, in either of two forms:

        * ``np.ndarray`` of shape ``(ng, *spatial)`` — the DETECTOR
          RESPONSE function :math:`\Sigma_d(\vec r, g)` (the canonical
          adjoint source).  Lifted to the composite as the **angle-flat
          broadcast** — no quadrature weights, no ``1/W``: under the
          G-pairing (bulk metric :math:`V\,w_n`) the plain broadcast is
          exactly the dual of the scalar-flux extraction,
          :math:`\langle \mathbf{1}_\Omega\Sigma_d,\,\psi\rangle_G =
          \sum_{\rm cells} V\,\Sigma_d\,\varphi = \langle\Sigma_d,
          \varphi\rangle_V` — the detector-response functional.  (The
          FORWARD iso-source lift divides by :math:`W`; the adjoint lift
          must NOT — the two lifts are duals of different maps, the
          source injection vs the flux extraction.  This asymmetry is
          the P1.2 reciprocity gate's exact content.)
        * :class:`~orpheus.transport.full_field.FullField` — the full
          composite adjoint source (bulk per-ordinate + boundary member)
          for angularly-selective detectors / prescribed adjoint inflow.

    Returns
    -------
    AdjointSolution
        The role-typed return (the A5 carrier ruling): ``keff`` is
        ``None`` (fixed-source), ``scalar_flux`` = the importance map
        :math:`\varphi^*` (also readable as
        :attr:`~orpheus.sn.solution.AdjointSolution.importance`).

    Notes
    -----
    Carrying (System-B) meshes are REFUSED loud for this entry at #276
    A4 — the daggered coupled fixed-source arm has no consumer or gate
    yet (the eigenvalue entry :func:`solve_sn_adjoint` covers carrying
    meshes, gated by the P1.3 sphere leg); it lands with its first
    consumer rather than shipping unexercised.
    """
    # Resolve BEFORE anything reads it, so the truncation warning below can
    # name the budget that actually bound (`None` in a message is useless).
    max_inner = resolve_iteration_budget(max_inner, inner_tol)
    sn_mesh = _as_sn_mesh(
        mesh, quadrature, materials, boundary_condition, mat_map=mat_map,
        scheme=scheme,
    )
    if sn_mesh.radial_characteristic_field_space is not None:
        raise NotImplementedError(
            "solve_sn_adjoint_fixed_source: the daggered COUPLED "
            "fixed-source arm (carrying meshes — System B present) has no "
            "consumer or gate yet and lands with its first consumer "
            "(#276 A4 scope note); the eigenvalue entry solve_sn_adjoint "
            "covers carrying meshes."
        )
    implicit_operator, gain, _F, template = _adjoint_posing_parts(
        sn_mesh, scattering_order,
    )

    from orpheus.numerics.iteration import SourceIteration, seeded_inverse
    from orpheus.transport.source_sinks import (
        AngularBoundarySourceSink,
        AngularSourceSink,
    )

    if isinstance(detector_response, FullField):
        q_star = detector_response
        if q_star.interior.space != q_star.interior.space_on(sn_mesh):
            raise ValueError(
                "solve_sn_adjoint_fixed_source: a composite "
                "detector_response must agree with this entry's mesh in "
                "space content (space-content invariant) — build it on the "
                "mesh passed here, or pass the scalar (ng, *spatial) "
                "detector form."
            )
    else:
        sigma_d = np.asarray(detector_response, dtype=float)
        expected = (sn_mesh.ng, *sn_mesh.spatial_shape)
        if sigma_d.shape != expected:
            raise ValueError(
                f"solve_sn_adjoint_fixed_source: detector_response shape "
                f"{sigma_d.shape} != (ng, *spatial) = {expected}."
            )
        # The angle-flat dual lift (docstring above), moment-lifted
        # through the ONE external-source policy (Q̂ = 0).
        per_ord = np.broadcast_to(
            sigma_d[None], (sn_mesh.quad.N, *sigma_d.shape),
        )
        bulk, _ = _lift_external_source_to_moments(
            np.ascontiguousarray(per_ord), sn_mesh,
        )
        q_star = FullField(
            interior=AngularSourceSink(
                values=bulk, space=sn_mesh.angular_trial_space,
            ),
            boundary=AngularBoundarySourceSink.zeros(sn_mesh.angular_trace),
        )

    si = SourceIteration(
        seeded_inverse(implicit_operator.H), gain.H,
        max_iter=max_inner, tol=inner_tol, budget_name="max_inner",
    )
    # Flux-classed zero start (the template) — the daggered iterate is an
    # adjoint FLUX; a zeros-like-the-source start would be source-classed
    # and trip the typed cross-class guard on the first increment ψ − ψ_prev.
    psi_star, record = si.solve(q_star, initial_guess=template)
    # #340 N6b — the exit defect of the DAGGERED equation, which is the one
    # this entry solved: ``A^† ψ* − q*`` with the same operator the driver
    # was handed.  No reconstruction and no rebuilt rhs here; both are
    # already the driver's own arguments.
    balance_defect = _exit_balance_defect(
        implicit_operator.H - gain.H, psi_star, q_star,
        sn_mesh=sn_mesh, record=record,
    )
    # #344 — see the note at `solve_sn_adjoint`: structurally inert here too
    # (1-D-only transpose solve ⟹ at most one reflective axis pair), wired so
    # the seam cannot rot.
    psi_star, gauge_correction = _exit_gauge_trace(psi_star, sn_mesh=sn_mesh)
    history = IterationHistory(
        record=record, balance_defect=balance_defect,
        gauge_correction=gauge_correction,
    )
    warn_if_unconverged(
        history.record, where="solve_sn_adjoint_fixed_source",
        balance_defect=history.balance_defect,
    )
    warn_if_gauge_freedom(
        sn_mesh, history.gauge_correction,
        where="solve_sn_adjoint_fixed_source",
    )
    return _package_adjoint_solution(
        psi_star, None, sn_mesh,
        keff=None,
        history=history,
    )


def _package_adjoint_solution(
    system_a: "FullField",
    adjoint_ray,
    sn_mesh: SNMesh,
    *,
    keff: "float | None",
    history: IterationHistory,
) -> AdjointSolution:
    r"""Wrap a converged daggered iterate into an :class:`AdjointSolution`.

    The adjoint face of the shared packaging tail — routes through the
    forward's own :func:`_cell_average_angular` + :func:`_package_solution`
    (ONE carrier convention, zero adjoint fork; the role is the ``cls``
    leaf, per the A5 ruling).  The scalar member is
    :math:`\varphi^* = \sum_n w_n \psi^*_n` — the importance map, the
    same w-reduction as the forward scalar flux (the adjoint of the ISO
    source injection, NOT a new functional).
    """
    angular = _cell_average_angular(system_a, sn_mesh)
    return _package_solution(
        angular,
        system_a.boundary,
        adjoint_ray,
        sn_mesh,
        scalar=angular.integrate_angular(),
        keff=keff,
        history=history,
        cls=AdjointSolution,
    )


def _build_fixed_source_rhs(
    external_source: "np.ndarray | TimedFullField | CoupledField",
    sn_mesh: SNMesh,
) -> "TimedFullField | CoupledField":
    r"""Normalize the external source into the driver RHS.

    A fixed-source problem's RHS is the composite source
    :class:`~orpheus.transport.timed_full_field.TimedFullField` — a bulk
    :class:`~orpheus.transport.source_sinks.AngularSourceSink` paired with a
    boundary :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`
    (the prescribed inflow :math:`q` of the affine BC
    :math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q`) — paired, on a
    carrying mesh (B.2d), with System B's q½ member into the coupled
    ``CoupledField[q_A, q_B]``. This is the ONE object that represents a
    source everywhere in the solve; this helper is its single construction
    point (Cardinal Rule 2 — the SI and Krylov inner paths both consume
    what it returns, rather than each re-deriving it).

    ``external_source`` is accepted in three forms (the bulk array is a typed
    union of TWO ndarray ranks; see :func:`_lift_external_source_to_moments`):

    * **flat** ``np.ndarray`` of shape ``(N, ng, *spatial)`` — the
      per-ordinate-density BULK source only; the boundary is vacuum (all-zero).
      The original form; every pre-existing caller keeps working unchanged
      (the slope moments :math:`\hat Q` are zeroed by the lift — the honest
      default, exact for a region-uniform source).
    * **moment-resolved** ``np.ndarray`` of shape
      ``(N, ng, *spatial, per_axis**ndim)`` — a multi-moment closure (LD)
      external source whose trailing axis carries the per-cell tensor-Legendre
      moment vector (slot 0 = cell average, the slope rows = :math:`\hat Q`,
      d=2 Kronecker order ``[Q̄, Q̂_y, Q̂_x, Q̂_xy]``).  The CALLER projects
      :math:`Q^{\rm ext}` onto the moment vector (e.g. by Gauss quadrature
      ``∫q·Pₖ``); this entry threads the slope rows through unchanged so they
      join the moment-carrying scattering source ``Σ_s·φ̂`` in the SI rhs
      (#247 — the slope-SOURCE half of the LM-1989 trap).  Only meaningful for
      LD (``per_axis > 1``); a moment-resolved input on a DD/Step mesh
      (``per_axis == 1``, no moment axis) is rejected by the flat-shape check.
    * :class:`TimedFullField` — the full COMPOSITE source (bulk + a possibly
      non-vacuum prescribed-inflow boundary, e.g. from
      :meth:`AngularBoundarySourceSink.prescribed_inflow`). Its leaf values are
      re-homed onto ``sn_mesh``: the trace/grid layout is deterministic from
      ``(mesh, quadrature, materials)``, so this is an exact values-copy onto
      the solve's own mesh instance — required because the within-group
      operators are built on ``sn_mesh`` and :class:`TimedFullField` algebra
      enforces mesh identity.  Its bulk may be flat OR moment-resolved.
    * :class:`~orpheus.numerics.coupled_system.CoupledField` — the coupled
      pair with an EXPLICIT q½ member (B.2d): System A re-homes as above;
      System B's member values re-home onto the fold slot exactly (the
      caller controls the true q½ instead of the ℓ = 0 fold default).
    """
    from orpheus.transport.source_sinks import (
        AngularSourceSink,
        AngularBoundarySourceSink,
    )

    N = sn_mesh.quad.N
    ng = sn_mesh.ng
    expected = (N, ng, *sn_mesh.spatial_shape)

    explicit_seed: "RadialCharacteristicField | None" = None
    source_a: "np.ndarray | FullField" = (
        external_source if not isinstance(external_source, CoupledField)
        else _system_a_member(external_source)
    )
    if isinstance(external_source, CoupledField):
        # The coupled pair: System B's explicit q½ member is captured for
        # the seed slot below; System A continues through the composite arm.
        explicit_seed = _system_b_member(external_source)
    if isinstance(source_a, FullField):
        bulk_values = np.asarray(source_a.interior.values)
        trace_size = int(sn_mesh.angular_trace.layout.total_size)
        boundary_values = source_a.boundary.values
        if boundary_values.size != trace_size:
            raise ValueError(
                f"_build_fixed_source_rhs: composite boundary source has "
                f"{boundary_values.size} values, but sn_mesh.angular_trace expects "
                f"{trace_size} (layout mismatch — the composite must be built "
                f"on the same mesh / quadrature / materials)."
            )
        boundary = AngularBoundarySourceSink(values=boundary_values.copy(), space=sn_mesh.angular_trace)
        # A composite carries its own q_∂. If the MESH also declares a
        # prescribed inflow, there are two answers to one question — refuse
        # rather than pick, since silently adding double-counts and silently
        # overriding makes the declaration a no-op.
        declared = AngularBoundarySourceSink.from_mesh_laws(sn_mesh)
        if declared.linf > 0.0 and boundary.linf > 0.0:
            raise ValueError(
                "_build_fixed_source_rhs: the boundary source q_∂ is specified "
                "TWICE — the mesh declares a PrescribedInflow law AND the "
                "composite external_source carries a non-zero boundary leaf. "
                "These are two answers to one question. Supply the inflow "
                "either as the declared boundary condition (preferred — it is "
                "the path a user travels) or as the composite's boundary leaf, "
                "not both."
            )
        if declared.linf > 0.0:
            boundary = declared
    else:
        bulk_values = np.asarray(source_a)
        if bulk_values.dtype == object:
            # A stray non-array, non-TimedFullField object (e.g. a bare
            # AngularSourceSink) — np.asarray wraps it as a 0-d object array.
            # Reject explicitly rather than failing the shape check obscurely.
            raise TypeError(
                f"_build_fixed_source_rhs: external_source must be an "
                f"(N, ng, *spatial) array (bulk-only / vacuum) or a "
                f"TimedFullField composite source; got "
                f"{type(external_source).__name__}"
            )
        # ⭐ The bulk-array form is "bulk only", NOT "boundary vacuum": a face
        # that DECLARES a PrescribedInflow contributes its q here (P2′ of
        # `.claude/plans/archive/affine_boundary_source_channel.md`). Before this, a
        # declared inflow was realized into an affine operator that nothing
        # consumed, so the declaration was silently inert. Every other law is
        # q = 0, so this is a zero trace allocation for all of them.
        boundary = AngularBoundarySourceSink.from_mesh_laws(sn_mesh)

    # Issue #196 PR-INDEX-5 + #247: the bulk source is a typed union of TWO
    # principled ndarray ranks — flat ``(N, ng, *spatial)`` (the original path)
    # OR moment-resolved ``(N, ng, *spatial, per_axis**ndim)`` (LD only; the new
    # slope-SOURCE path).  Discriminate by RANK (NOT trailing-size — a
    # coincidental spatial dim could equal 2^d): a flat bulk has exactly
    # ``len(expected)`` axes; a moment-resolved bulk has ONE more (the trailing
    # 2^d moment axis).  Reject everything else, INCLUDING a moment axis whose
    # length ≠ per_axis**ndim, and (for DD/Step where there is no moment axis) a
    # moment-resolved input outright (only flat is valid at per_axis == 1).
    n_cell_moments = cell_moment_count(
        sn_mesh.scheme.spatial_basis_per_axis, sn_mesh.ndim
    )
    moment_expected = (*expected, n_cell_moments)
    is_flat = bulk_values.shape == expected
    is_moment_resolved = (
        n_cell_moments > 1 and bulk_values.shape == moment_expected
    )
    if not (is_flat or is_moment_resolved):
        if n_cell_moments > 1 and bulk_values.shape[:-1] == expected:
            # Right rank for a moment-resolved bulk, but the trailing moment
            # axis is the wrong width — name the expected 2^d so the relaxation
            # does not swallow a real shape bug (#247 negative pin).
            raise ValueError(
                f"fixed-source moment-resolved bulk shape {bulk_values.shape} "
                f"has trailing moment axis {bulk_values.shape[-1]}, expected "
                f"per_axis**ndim = {n_cell_moments} "
                f"(moment vector {moment_expected})"
            )
        raise ValueError(
            f"fixed-source bulk shape {bulk_values.shape} does not match "
            f"(N, ng, *spatial) = {expected}"
            + (
                f" or the moment-resolved {moment_expected}"
                if n_cell_moments > 1 else ""
            )
        )
    # The external source carries the trailing 2^d spatial-moment axis at a
    # multi-moment closure (#240 D5b-S3) so it composes with the moment-carrying
    # scattering source ``S.apply(ψ)`` in the SI rhs ``q_ext + S.apply(ψ)``.  A
    # FLAT external source is flat-in-moment (Q̂ = 0 — the slope rows are zero,
    # the honest default exact for a region-uniform source): lift onto slot 0
    # (average), rest zero.  A MOMENT-RESOLVED external source already carries
    # the slope rows Q̂ (the caller projected them — #247): thread them through
    # unchanged.  DD/Step (per_axis == 1) → no lift, byte-identical.
    bulk_values, _ = _lift_external_source_to_moments(bulk_values, sn_mesh)
    q_a = TimedFullField(
        interior=AngularSourceSink(
            values=bulk_values, space=sn_mesh.angular_trial_space,
        ),
        boundary=boundary,
    )
    if sn_mesh.radial_characteristic_field_space is None:
        return q_a
    # #282 route (a) → B.2d: the q½ member on carrying meshes — System B's
    # OWN composite, paired with q_A.  A coupled input's explicit member is
    # re-homed values-exactly (same deterministic-layout argument as the
    # trace copy above); otherwise the ℓ = 0 fold of the per-ordinate bulk
    # populates it.  Carrying meshes are 1-D curvilinear DD (never
    # moment-resolved), so the flat bulk is the only live shape here.
    if explicit_seed is not None:
        seed_src = RadialCharacteristicField.source_zeros(sn_mesh.radial_characteristic_field_space)
        seed_src.interior.values[...] = explicit_seed.interior.values
        seed_src.boundary.values[...] = explicit_seed.boundary.values
    else:
        # ``boundary_trace`` = the SAME rhs's boundary member: a prescribed
        # (non-vacuum) inflow delivers System B's r = R corner datum through
        # the source channel (the factory's three-arm inflow-corner law —
        # the step-7 regression fix; zero for vacuum/reflective rhs, so
        # those paths stay byte-identical).
        seed_src = _radial_characteristic_source_from_per_ordinate(
            bulk_values, sn_mesh, boundary_trace=boundary,
        )
    return _coupled_source_state(
        q_a, seed_src, sn_mesh, context="_build_fixed_source_rhs",
    )


def _lift_external_source_to_moments(
    bulk_values: "np.ndarray", sn_mesh: SNMesh,
) -> "tuple[np.ndarray, int]":
    r"""Lift / thread an external source onto the ``2^d`` cell-moment vector,
    returning ``(lifted, per_axis)``.

    Single source of the external-source moment lift for the fixed-source path
    (#240 D5b-S3 / #247 — the slope-SOURCE widening).  One production caller
    (:func:`_build_fixed_source_rhs`); kept as a single-source helper so a future
    eigenvalue external-source hook reuses the same lift/thread policy.
    ``bulk_values`` is a typed union of TWO ndarray ranks, discriminated by RANK
    (NOT trailing-size —
    :func:`~orpheus.numerics.moment_layout.is_moment_valued_by_flat_rank` against
    the flat ``(N, ng, *spatial)`` rank; a coincidental spatial dim could equal
    ``2^d``):

    * **DD/Step** (``per_axis == 1``, ``tail == ()``): no moment axis — return
      the input unchanged (byte-identical, the backward-compat negative
      control).
    * **flat** ``(N, ng, *spatial)``: zero the ``2^d`` buffer, copy the flat
      input onto slot 0 (average), leave the slope rows :math:`\hat Q` ZERO —
      the honest default (``q̂ = 0`` is exact with no sub-cell information).
    * **moment-resolved** ``(N, ng, *spatial, 2^d)``: the caller already
      projected the slope rows (#247) — thread the moment vector through
      UNCHANGED (validate its trailing axis == ``2^d``).  Joins the
      moment-carrying scattering source ``Σ_s·φ̂`` in the SI rhs; the per-octant
      slope-sign reframe (``sweep_graph._CellSolve`` /
      ``octant_moment_frame_signs``) re-signs the external slopes global→sweep
      EXACTLY as it does the scattering slopes — no new cell branch."""
    per_axis = sn_mesh.scheme.spatial_basis_per_axis
    n_cell_moments = cell_moment_count(per_axis, sn_mesh.ndim)
    tail = face_moment_tail(n_cell_moments)
    if tail == ():
        return bulk_values, per_axis
    flat_ndim = 2 + len(sn_mesh.spatial_shape)  # (N, ng, *spatial)
    if not is_moment_valued_by_flat_rank(bulk_values, flat_ndim):
        # Flat input → lift onto slot 0, slopes zero (the honest default).
        lifted = np.zeros((*bulk_values.shape, *tail), dtype=bulk_values.dtype)
        lifted[..., AVERAGE_MOMENT] = bulk_values
        return lifted, per_axis
    # Moment-resolved input → thread the slope rows through unchanged (#247).
    if bulk_values.shape[-1] != n_cell_moments:
        raise ValueError(
            f"_lift_external_source_to_moments: moment-resolved bulk has "
            f"trailing moment axis {bulk_values.shape[-1]}, expected "
            f"per_axis**ndim = {n_cell_moments}"
        )
    return bulk_values, per_axis


def _average_moment_scalar(phi: "np.ndarray", sn_mesh: SNMesh) -> "np.ndarray":
    r"""Reduce a (possibly moment-carrying) scalar flux to its cell-AVERAGE.

    The user-facing :class:`Solution` scalar flux is the cell-average moment
    (slot 0); a multi-moment closure's φ̂ slopes are internal within-cell DG
    structure (#240 D5b-S3).  ``phi`` from a multi-moment closure carries a
    trailing ``2^d`` axis — take slot ``AVERAGE_MOMENT``; DD/Step (per_axis ==
    1) → no axis → return unchanged."""
    per_axis = sn_mesh.scheme.spatial_basis_per_axis
    if face_moment_tail(cell_moment_count(per_axis, sn_mesh.ndim)) == ():
        return phi
    return phi[..., AVERAGE_MOMENT]


def solve_sn_fixed_source(
    materials: dict[int, Mixture],
    mesh: "Mesh1D | Mesh2D | tuple[Axis1D, ...]",
    quadrature: Quadrature,
    external_source: "np.ndarray | TimedFullField",
    boundary_condition: "str | None" = "vacuum",
    scattering_order: int = 0,
    max_inner: int | None = None,
    inner_tol: float = 1e-12,
    inner_solver: str | None = None,
    inner_schedule: str = "gauss_seidel",
    mat_map: "np.ndarray | None" = None,
    scheme: "DiscretizationSchemeBase | None" = None,
    acceleration: str | None = None,
) -> Solution:
    r"""Solve the multi-group SN fixed-source transport problem.

    Solves

    .. math::

        \mu_n \frac{\partial\psi_n}{\partial x}+\Sigma_t\psi_n
        = \frac{1}{W}\left(\Sigma_s\phi + Q^{\text{ext}}_n\right)

    for a prescribed per-ordinate external source ``external_source``,
    with vacuum or reflective boundary conditions. The fission source
    is zero — this is the pure transport operator. Source iteration
    converges geometrically at rate :math:`c = \max\Sigma_s/\Sigma_t`.

    Parameters
    ----------
    materials, mesh, quadrature, scattering_order :
        Same as :func:`solve_sn`.
    external_source : (N, ng, nx, ny) ndarray OR TimedFullField
        The fixed-source RHS, in either of two forms (normalised by
        :func:`_build_fixed_source_rhs`):

        * ``np.ndarray`` of shape ``(N, ng, nx, ny)`` — the per-ordinate
          volumetric BULK source :math:`Q^{\text{ext}}_n(x)` in
          **per-ordinate density magnitude** (R-1 Step 4 A1 convention),
          with a **vacuum** boundary. Callers with an iso scalar source
          :math:`Q(\vec r, g)` should project to per-ordinate via
          :meth:`~orpheus.transport.source_sinks.AngularSourceSink.from_isotropic`
          before passing (the :math:`1/W` projection lives at the producer
          boundary per Pattern 7). The sweep does NOT apply ``/W``
          internally. Issue #196 PR-INDEX-5: principled layout (``g`` axis
          after ``N``).
        * :class:`~orpheus.transport.timed_full_field.TimedFullField` — the
          full **composite** source ``q = q_bulk ⊕ q_∂`` (a bulk
          :class:`~orpheus.transport.source_sinks.AngularSourceSink` paired
          with a boundary
          :class:`~orpheus.transport.source_sinks.AngularBoundarySourceSink`). This
          is how a **non-vacuum prescribed inflow** is supplied — build the
          boundary via
          :meth:`~orpheus.transport.source_sinks.AngularBoundarySourceSink.prescribed_inflow`
          (the affine-BC inhomogeneous term :math:`q`, consumed by the sweep
          as the inflow seed). The legacy array form is exactly the
          bulk-only / vacuum special case of this composite.
    boundary_condition : {"vacuum", "reflective"} or None
        Applied to all faces when the mesh has no explicit BC
        declarations (``bc_left`` etc. are ``None``).  When the mesh
        carries explicit :class:`~orpheus.geometry.mesh.BC` fields,
        those take precedence and this parameter is ignored.
        Vacuum is the default because the intended consumer is
        Method of Manufactured Solutions verification on a finite slab.
    max_inner, inner_tol :
        Inner solver iteration limits.
    inner_solver : {"source_iteration", "krylov", None}
        Inner-solve strategy.  When ``None`` (default), all geometries
        use ``"source_iteration"``: post-unification the sweep and matvec
        are ONE O(h²)-consistent discrete system (the ERR-058 closure-seed
        fix), so SI and Krylov converge to bit-identical fixed points on
        the curvilinear MMS ladders, with SI ~10²× faster (no GMRES restart
        pathology, ERR-053).  ``"krylov"`` stays available as the opt-in
        cross-check — the SI ≡ Krylov fixed-point equivalence is a standing
        splitting-invariance gate (vv-principles Mode 9).
    inner_schedule : {"gauss_seidel", "jacobi"}
        Source-iteration BOUNDARY splitting (Phase 3, ``inner_solver=
        "source_iteration"`` only).  ``"gauss_seidel"`` (default) folds the
        reflective coupling ``B`` into an octant-group Gauss-Seidel resolvent
        (multi-D Cartesian) — re-reflecting each octant group's outgoing
        reflective faces between group sweeps so a later group reads the fresh
        current-iterate inflow.  ``"jacobi"`` lags ``B`` fully (the
        splitting-invariant control).  The converged fixed point is IDENTICAL
        for both — this selects only the SI spectral rate.  1-D meshes always
        fall back to Jacobi (boundary G-S is a no-op on the scattering-
        dominated 1-D regime, and the scan is not a wavefront).  The dominant
        within-group SCATTERING rate is unchanged either way — that is what
        ``acceleration="dsa"`` deflates (issue #2).

        ⚠ **The rate effect is NOT regime-independent, and it is not
        bounded in either direction** (the splitting is not a *regular*
        splitting, so no comparison theorem applies —
        :ref:`sn-boundary-gs-not-regular`).  G-S folds only ``B``, so its
        leverage is exactly the weight of the boundary coupling in the
        iteration — which is maximal at ZERO leakage (nothing escapes, so
        the boundary is the whole coupling) and collapses as soon as any
        face is vacuum.
        `[M]` 2026-08-08, SI sweeps to ``inner_tol=1e-13``, LS4, 2-group,
        ``n_GS / n_Jacobi``:

        =========================== ==== ====== =====
        configuration                G-S Jacobi ratio
        =========================== ==== ====== =====
        d=2 all-reflective           258    648  0.40
        d=2 all-reflective, c=0.5    259    645  0.40
        d=3 all-reflective          1631    838  1.95
        d=3 all-reflective, c=0.5   1598    832  1.92
        d=2 one vacuum axis, c=0.5    34     35  0.97
        d=3 one vacuum axis, c=0.5   208    214  0.97
        d=3 two vacuum axes, c=0.5    33     33  1.00
        =========================== ==== ====== =====

        So: a wash the moment anything leaks — at every dimension — and
        at zero leakage a large effect of EITHER sign.  Scattering does
        not change the picture (the ``c=0.5`` rows track the absorber
        rows), which is consistent with G-S touching only ``B``.

        ⛔ **REFUTED 2026-08-09 (#341).** This paragraph read *"a WIN at
        d=2 zero-leakage, a LOSS at d=3 zero-leakage"*, and the ⚠ above
        read *"its SIGN flips with dimension"*.  ``ndim`` is NOT the
        discriminating variable.  `[M]` same probe, same mixture/quad/tol,
        varying only the mesh: d=2 all-reflective at extents (1,2) cells
        (1,1) gives ``202 / 38 = 5.32`` and at extents (6,6) cells (2,2)
        gives ``54 / 47 = 1.15`` — both LOSSES at d=2, the first worse
        than the d=3 row the story was built on.  The effect is
        continuous in the per-cell optical thickness and the mesh, which
        merely correlate with ``ndim`` on the fixtures first measured.
        **Do not branch a default on** ``ndim``.  The structural reason
        an inversion is permitted at all — the splitting is a splitting
        but NOT a *regular* one, so no comparison theorem bounds it — is
        :ref:`sn-boundary-gs-not-regular`.

        ⚠ Only the SIGN and the leakage-dependence are robust; the
        MAGNITUDE is fixture-specific.  A second d=2 zero-leakage point
        (B-2g 8×8 ``product(2,4)``, in
        ``test_si_convergence_rate.py::test_boundary_gs_recovers_reflective_2d_si``)
        reads 641/697 = 0.92 against 0.40 here — same sign, >2× different
        magnitude.  That is why that gate asserts the strict inequality
        and not a ratio: **the inequality is the law, the ratio is a
        fixture reading.**  Which SIDE a zero-leakage configuration lands
        on is measured but not yet PREDICTED — see issue #341 and
        :ref:`sn-boundary-gs-rate-regime`.  Practical reading: with
        leakage the choice is immaterial, and at zero leakage neither
        schedule is safe to assume — if the sweep count of an
        all-reflective fixture matters, measure both arms (they share the
        fixed point, so the comparison is free).
    acceleration : {"dsa", None}
        Within-group synthetic acceleration (issue #2).  ``"dsa"`` wires
        the consistent-DSA correction operator
        (:class:`~orpheus.sn.acceleration.dsa.DSACorrection` — the
        derived edge-centered low-order system, R4 ruling) into whichever
        inner posture runs: the ``SourceIteration`` corrector step under
        ``"source_iteration"``, the transport-corrected GMRES left
        preconditioner under ``"krylov"``.  Admission (1-D Cartesian DD
        slab, vacuum/reflective walls) is enforced at the operator build
        with loud seams for everything else.  ``None`` (default) leaves
        both paths byte-untouched — the accelerator is additive, never a
        default change.  The converged fixed point is IDENTICAL with and
        without (correction→0 at convergence; the D3/D4 FP-invariance
        battery) — DSA buys the RATE: :math:`\rho \le 0.2247c` in place
        of SI's :math:`\rho \approx c` (Adams & Larsen (3.65); the
        measured scan in ``.claude/plans/archive/dsa_d2_characterization.md``).

    Notes
    -----
    This entry point exists for L1 verification via MMS, not for
    engineering problems — real fixed-source calculations should still
    build on :func:`solve_sn` with an appropriate external-source hook.
    See :mod:`orpheus.derivations.continuous.mms.sn` and the MMS verification
    section of the discrete-ordinates theory page.
    """
    t_start = time.perf_counter()

    # Resolve BEFORE anything reads it: the budget reaches both the SNSolver
    # and the two driver helpers, and the truncation warning must be able to
    # name the number that actually bound.
    max_inner = resolve_iteration_budget(max_inner, inner_tol)

    # Normalize the geometry declaration (legacy mesh OR axis tuple —
    # the only 3-D entry; C5.5 #225) into the SN phase space;
    # boundary_condition fills faces only when the declaration carries
    # no explicit BC.
    sn_mesh = _as_sn_mesh(
        mesh, quadrature, materials, boundary_condition, mat_map=mat_map,
        scheme=scheme,
    )

    # All geometries default to ``"source_iteration"`` — the unified
    # sweep/matvec system is O(h²)-consistent (ERR-058 closure-seed fix),
    # so SI ≡ Krylov at the fixed point and SI is ~10²× faster.
    if inner_solver is None:
        inner_solver = "source_iteration"

    # Synthetic acceleration (issue #2) — additive opt-in: ``None`` leaves
    # both inner paths byte-untouched; ``"dsa"`` builds the ONE correction
    # operator both postures consume (admission enforced at the build).
    if acceleration not in (None, "dsa"):
        raise ValueError(
            f"solve_sn_fixed_source: unknown acceleration "
            f"{acceleration!r}; supported: None, 'dsa'."
        )
    corrector = None
    if acceleration == "dsa":
        from orpheus.sn.acceleration import DSACorrection

        corrector = DSACorrection.from_sn_mesh(
            sn_mesh, scattering_order=scattering_order,
        )

    solver = SNSolver(
        sn_mesh,
        inner_solver=inner_solver,
        scattering_order=scattering_order,
        max_inner=max_inner, inner_tol=inner_tol,
    )

    # Normalise the external source (raw bulk array OR composite
    # TimedFullField) into the single composite RHS ``q = q_bulk ⊕ q_∂`` the
    # inner paths consume (Cardinal Rule 2 — one construction point; shape
    # validation lives inside the helper).
    q_ext_composite = _build_fixed_source_rhs(external_source, sn_mesh)

    if inner_solver == "source_iteration":
        solution = _solve_fixed_source_si(
            solver, sn_mesh, q_ext_composite,
            t_start, max_inner, inner_tol, inner_schedule=inner_schedule,
            corrector=corrector,
        )
    else:
        # Krylov path.  We solve T·ψ = b directly via GMRES, where b carries
        # the external per-ordinate source plus any in-scatter / (n,2n) terms
        # built from the converged scalar flux.  Wrapping that in an outer
        # source iteration converges scattering self-consistently.
        solution = _solve_fixed_source_krylov(
            solver, sn_mesh, q_ext_composite,
            t_start, max_inner, inner_tol,
            corrector=corrector,
        )

    # ⭐ #340 N4.7 (2026-08-11): the emission lives HERE, in the PUBLIC entry,
    # and NOT in the two private arms this dispatches to.
    #
    # ⛔ Until 2026-08-11 both arms called it themselves, and `[M]` that made
    # this the one entry of eight whose warning blamed ORPHEUS instead of the
    # caller: :func:`warn_if_unconverged` uses ``stacklevel=3`` (helper →
    # public entry → user code), so from inside a private arm frame 3 is this
    # function's own ``return _solve_fixed_source_si(`` dispatch line.  The
    # warning appeared, named the right level, and pointed the reader at
    # ``sn/solver.py`` — a file they did not write.  Reproduced at
    # ``max_inner`` in {1, 2, 5, 50} on two fixtures.
    #
    # ⚠ The tempting alternative — pass a per-call ``stacklevel`` — is the
    # defect one layer up: a frame COUNT is a fact about the call chain that
    # the call site asserts and that silently rots the moment a helper is
    # interposed.  Hoisting makes the depth structurally true instead.
    #
    # ⭐ And it is strictly better than the arms were: reading the record off
    # the Solution about to be RETURNED makes "the warning and the returned
    # object describe the same solve" a theorem rather than a convention, and
    # collapses two mirror emission points into one (Cardinal Rule 2).
    #
    # ``history`` is Optional on :class:`~orpheus.sn.solution.Solution`
    # because other producers build one without a solve; both arms above
    # always construct it.  When it is genuinely absent there is nothing to
    # say about convergence, so silence is the honest answer — the same
    # reading :attr:`~orpheus.sn.solution.SolutionBase.converged` takes.
    if solution.history is not None:
        warn_if_unconverged(
            solution.history.record, where="solve_sn_fixed_source",
            balance_defect=solution.history.balance_defect,
        )
        # #344 — HOISTED here on purpose. Both arms project, but neither may
        # warn: from inside an arm this sits two frames below the entry, so
        # `stacklevel=3` blames `orpheus/sn/solver.py` rather than the caller
        # (#340 N4.7, ⛔ above). The verdict needs only the mesh, and the
        # magnitude rides `history`, so the entry can say it for either arm.
        warn_if_gauge_freedom(
            sn_mesh, solution.history.gauge_correction,
            where="solve_sn_fixed_source",
        )
    return solution


def _solve_fixed_source_si(
    solver: SNSolver,
    sn_mesh: SNMesh,
    q_ext_composite: "TimedFullField | CoupledField",
    t_start: float,
    max_inner: int,
    inner_tol: float,
    inner_schedule: str = "gauss_seidel",
    corrector: "LinearOperator | None" = None,
) -> Solution:
    r"""Fixed-source path via the :class:`SourceIteration` primitive.

    Carved onto the SAME :class:`~orpheus.numerics.iteration.SourceIteration`
    primitive the eigenvalue inner :meth:`SNSolver._solve_source_iteration`
    uses, on the identical operator triple

    .. math::

        A \;=\; L + C\,, \quad
        N \;=\; S + N_{2n} + B\,, \quad
        F \;=\; 0_{\rm wg}

    differing ONLY in ``q_ext`` (the EXTERNAL source here vs the fission
    source in the eigenvalue inner) and the returned contract (a full typed
    :class:`Solution` here vs an angular-integrated scalar flux there).

    Scattering, the (n,2n) gain and the reflective boundary enter as the
    primitive's coupling gains ``S.apply(ψ_n) + N₂ₙ.apply(ψ_n) + B.apply(ψ_n)``
    (§14.1): :meth:`ScatteringOperator.apply` / :meth:`N2NOperator.apply` each
    recompute their channel's ``P0/W + Pℓ`` bulk emission, and the
    boundary gain ``B``
    (:class:`~orpheus.sn.operators.boundary.SNBoundaryOperator`, a first-class
    coupling in the within-group system record) delivers the reflective
    ``B·ψ.outflow`` through ``rhs.boundary`` which the bare ``(L + C).solve``
    sweep reads as the inflow seed (single source of truth — Cardinal Rule 2).
    No production path sets inflow slots by hand any more: since #448 the
    eigenvalue finalize is one :func:`~orpheus.numerics.iteration.fixed_point_step`
    of this same map (``B`` a gain there too); the octant-restricted
    Gauss-Seidel resolvent completes its lagged rows additively through
    :meth:`~orpheus.sn.operators.boundary.SNMaskedBoundaryOperator.reflect_rows_inplace`,
    and the whole-trace assignment is the sweep-tier gates' helper
    (``tests/sn/_test_helpers.py::reflect_outflow_into_inflow``).

    Geometry-agnostic (slab / sphere / cylinder / 2-D Cartesian): the
    within-group solve carries no geometry dependence, exactly as the
    eigenvalue SI inner (Wave O "2-D SI Phase A").

    Equivalence note (vv-principles §bit-identity): the converged fixed point
    is identical to the retired loop's (same operators, same ``S`` and ``B``
    coupling gains, same WDD sweep), but the iteration TRAJECTORY differs — the primitive
    stops on the composite ``‖ψ_{n+1} − ψ_n‖ / ‖ψ_{n+1}‖`` residual (the full
    angular + boundary iterate, the same metric the verified eigenvalue inner
    uses) rather than the scalar-flux ``‖φ − φ_prev‖ / ‖φ‖``.  Converged ``φ``
    therefore agrees to ``~inner_tol`` (principled-equivalence), and
    ``history.n_inner`` / ``flux_residuals`` reflect the composite metric.
    """
    from orpheus.transport.fields.angular_flux import (
        AngularFlux,
    )
    from orpheus.transport.fields.angular_boundary_flux import (
        AngularBoundaryFlux,
    )
    from orpheus.transport.fields.scalar_flux import ScalarFlux

    # ``q_ext_composite`` is the normalised composite RHS ``q = q_bulk ⊕ q_∂``
    # built once by :func:`_build_fixed_source_rhs` (Cardinal Rule 2 — the SI
    # and Krylov paths share it).  The bulk is the per-ordinate-density
    # external source; the boundary is the prescribed inflow (zero for
    # vacuum / reflective — the reflective inflow rides ``rhs.boundary`` via
    # the ``B`` coupling gain, NOT ``q_ext``; a NON-vacuum prescribed inflow
    # is carried in ``q_ext_composite.boundary``).  Scattering (P0 + Pℓ +
    # (n,2n)) is NOT pre-staged — the primitive's ``S`` operator recomputes
    # it each iterate.

    # ── Build the within-group system + SI via the SHARED builders (single
    # source of truth — :func:`~orpheus.sn.coupled_system.build_within_group_system`
    # / :func:`_within_group_si`; identical construction to the eigenvalue
    # inner).  ``inner_schedule`` selects Jacobi vs boundary-G-S; the SI
    # builder folds in the Phase-5a angular-windowing.  ``base_implicit``
    # (un-wrapped) + ``gains`` are kept for the final full-angular
    # reconstruction below. ────────────────────────────────────────────
    system = build_within_group_system(
        sn_mesh, solver.mat_xs, scattering_op=solver.scattering_op,
        n2n_op=solver.n2n_op,
    )
    si, base_implicit, gains, windowed = _within_group_si(
        system, sn_mesh,
        inner_schedule=inner_schedule, max_iter=max_inner, tol=inner_tol,
        corrector=corrector,
    )
    coupled = isinstance(system.implicit_operator, CoupledOperator)

    # Cold-start iterate (x0 = zeros).  Fixed-source is a single solve — no
    # eigenvalue outer to warm-start from (cf. the eigenvalue inner's
    # ``self._inner.iterate``).  Windowed → zero moments (single-sourced in
    # :func:`_windowed_cold_start`); else a zero AngularFlux — paired
    # NATIVE with a zero ψ_B on a carrying mesh (B.2d).
    q_a_ext = _system_a_member(q_ext_composite)
    # The rhs builder emits the TIMED System-A frame (its history_depth keys
    # the cold start); the parse reifies that producer contract loudly.
    if not isinstance(q_a_ext, TimedFullField):
        raise TypeError(
            f"fixed-source SI: the rhs's System-A member must be the timed "
            f"composite; got {type(q_a_ext).__name__}."
        )
    if windowed:
        initial_guess = _windowed_cold_start(
            solver.scattering_op, sn_mesh,
            history_depth=q_a_ext.history_depth,
        )
    else:
        cold = _unwindowed_cold_start(
            sn_mesh, history_depth=q_a_ext.history_depth,
        )
        initial_guess = _coupled_flux_state(cold, sn_mesh) if coupled else cold
    # ``q_ext_composite`` is already driver-ready (the coupled pair on a
    # carrying mesh — built once by :func:`_build_fixed_source_rhs`).
    psi_typed, record = si.solve(
        q_ext_composite, initial_guess=initial_guess,
    )
    # The end-of-solve CERTIFICATE (step 5, R-5.2) — full-angular arms only
    # (the windowed moment arm is structurally exempt; see
    # _certify_within_group_exit).
    if not windowed:
        _certify_within_group_exit(
            system, psi_typed, q_ext_composite,
            sn_mesh=sn_mesh, record=record,
            where="solve_sn_fixed_source[source_iteration]",
        )
    # System A's converged member feeds the Solution contract; System B's
    # rides ``Solution.radial_characteristic`` (B.2d DP-Solution).
    psi_full = _system_a_member(psi_typed)
    # The parse reifies the driver-template contract (the solve echoes the
    # TimedFullField initial_guess member) — the static iterate type is the
    # operators' carrier, re-narrowed here loudly.
    if not isinstance(psi_full, TimedFullField):
        raise TypeError(
            f"fixed-source SI: the converged iterate must echo the timed "
            f"template; got {type(psi_full).__name__}."
        )
    # Issue #197 PR-TYPED-5: build typed Solution at the boundary.
    # (The former mesh / quadrature / materials parameters retired in C4 —
    # Solution never consumed them; the typed fluxes carry the SNMesh
    # reference, which transitively exposes those handles via
    # ``.mesh.{mesh, quad, materials}``.)
    # ``Solution.angular_flux`` must carry the FULL per-ordinate angular flux.
    # Un-windowed: ``psi_typed`` already IS it (return directly, exactly as the
    # fixed-source Krylov path does; the boundary trace lives on
    # ``psi_typed.boundary`` — no legacy ``solver._boundary_flux`` writeback).
    # Windowed: ``psi_typed.interior`` is the moment iterate, so reconstruct the
    # full angular with ONE application of the splitting map — the converged
    # source ``q + Σ gains·ψ`` through the UN-wrapped base resolvent, the one
    # body :func:`~orpheus.numerics.iteration.fixed_point_step` the eigenvalue
    # finalize evaluates too (#448).  Bit-identical to the un-windowed
    # converged ψ: S/B consume the moments == the full angular's moments
    # (de-risk proven), so the source is the same, and one sweep of the
    # converged source reproduces the converged iterate by the fixed point.
    if windowed:
        # Windowed ⟹ 2-D Cartesian ⟹ seedless (R12a): the iterate is the
        # fused composite (``psi_full IS psi_typed`` here) and the resolvent
        # is never the coupled bridge — the parse states the structural
        # fact loudly instead of assuming it.
        if isinstance(base_implicit, CoupledOperator):
            raise TypeError(
                "windowed reconstruction reached a coupled resolvent — "
                "structurally unreachable (windowing is 2-D Cartesian, "
                "seedless; the coupled arm never windows)."
            )
        from orpheus.numerics.iteration import fixed_point_step

        angular_out = fixed_point_step(
            base_implicit.inverse(), gains, q_a_ext, psi_full,
        )
    else:
        # Un-windowed: the (re-fused) converged iterate IS the full
        # per-ordinate angular flux.
        angular_out = psi_full
    # Scalar flux from the RETURNED full angular flux → the Solution is exactly
    # self-consistent (``scalar == ∫ angular dΩ``), matching the un-windowed
    # contract.  (For the un-windowed path ``angular_out`` IS ``psi_typed``, so
    # this is bit-identical to the prior ``psi_typed.interior.integrate_angular``.)
    # The user-facing scalar flux is the cell-AVERAGE moment (slot 0) — a
    # multi-moment closure's φ̂ slopes are internal DG structure, not the scalar
    # flux the Solution reports (#240 D5b-S3).  The parse reifies the
    # full-angular contract of BOTH arms (the reconstruction sweep emits
    # angular; the un-windowed iterate echoes the flux template) loudly.
    angular_bulk = angular_out.interior
    if not isinstance(angular_bulk, AngularFlux):
        raise TypeError(
            f"fixed-source SI: Solution.angular_flux must carry an "
            f"AngularFlux bulk; got {type(angular_bulk).__name__}."
        )
    phi = _average_moment_scalar(
        angular_bulk.integrate_angular().values, sn_mesh,
    )
    # ⛔ The warning is emitted HERE, not before the reconstruction above,
    # and the reason is the balance defect (#340 N6b).  On the WINDOWED arm
    # ``psi_typed.interior`` is a ``HarmonicMomentFlux`` — an angular-moment
    # iterate, which the projection cannot integrate over angle — while
    # ``angular_out`` is the full-angular reconstruction.  Warning before
    # line ~3806 would have silently dropped the number on exactly one arm,
    # and it is the arm a reader would least suspect, because the sibling
    # Krylov path and the eigenvalue entry both work on the same 2-D mesh.
    #
    # Which iterate: ``angular_out`` when windowed, ``psi_typed`` otherwise.
    # The two cases are disjoint by construction — windowing is 2-D
    # Cartesian hence seedless, so a windowed solve is never coupled (the
    # guard above says so) — which is why the coupled composite and the
    # reconstruction never both apply.
    balance_defect = _exit_balance_defect(
        system.loss if coupled else _bare_loss_arm(system),
        angular_out if windowed else psi_typed,
        q_ext_composite,
        sn_mesh=sn_mesh, record=record,
    )
    # #344 — the PROJECTION fires here, because this is where the trace is; the
    # WARNING must NOT. This is a private arm, two frames below the public
    # entry, so `stacklevel=3` would blame `orpheus/sn/solver.py` instead of the
    # caller — verbatim the defect #340 N4.7 measured and fixed by hoisting.
    # It is emitted by `solve_sn_fixed_source` off `history.gauge_correction`.
    #
    # ⛔ `_exit_gauge_trace` REBUILDS rather than writing in place: on the
    # un-windowed path `angular_out IS psi_typed`, which `_exit_balance_defect`
    # has already measured and `_system_b_member` is about to read.
    angular_out, gauge_correction = _exit_gauge_trace(
        angular_out, sn_mesh=sn_mesh,
    )
    history = IterationHistory(
        record=record, balance_defect=balance_defect,
        gauge_correction=gauge_correction,
    )
    return Solution(
        angular_flux=angular_out,
        scalar_flux=ScalarFlux(values=phi, space=sn_mesh.bulk_space),
        mesh=sn_mesh,
        keff=None,
        history=history,
        radial_characteristic=_system_b_member(psi_typed),
    )


def _solve_fixed_source_krylov(
    solver: SNSolver,
    sn_mesh: SNMesh,
    q_ext_composite: "TimedFullField | CoupledField",
    t_start: float,
    max_inner: int,
    inner_tol: float,
    corrector: "LinearOperator | None" = None,
) -> Solution:
    r"""Curvilinear-default fixed-source path: typed :class:`KrylovAcceleration`.

    Carved onto :class:`~orpheus.numerics.iteration.KrylovAcceleration`
    consuming the operator triple

    .. math::

        A \;=\; L + C\,, \quad
        S \;=\; \text{full multi-group scatter}\,, \quad
        F \;=\; 0_{\rm wg}

    on typed :class:`~orpheus.transport.fields.angular_flux.AngularFlux`.  The
    composite ``L + C`` returns an
    :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`; its ``.solve``
    IS the WDD sweep, but R-1 ships GMRES UNPRECONDITIONED
    (explicit identity) — issue #200 tracks the block-inverse face
    preconditioner re-enablement.

    The outer Picard wrap on the scattering source dissolves at G1:
    typed ``KrylovAcceleration`` solves :math:`(L+C-S)\,\psi =
    q_{\rm ext}` directly via GMRES, putting scattering
    self-consistency INSIDE the Krylov polynomial.  Fission ``F`` is
    zero — this is the pure fixed-source path; the eigenvalue
    outer/within-group decomposition lives in :func:`solve_sn`.

    Scope
    =====

    ALL geometries — slab, sphere, cylinder, AND 2-D Cartesian.  The 2-D
    Cartesian fixed-source Krylov is the structural twin of the eigenvalue
    Krylov inner :meth:`SNSolver._solve_krylov`: identical within-group
    system (:func:`~orpheus.sn.coupled_system.build_within_group_system`)
    and identical :class:`~orpheus.numerics.iteration.KrylovAcceleration`
    driver (:func:`_within_group_krylov`), differing ONLY in ``q_ext`` (the
    external source here vs the fission source there) — and the twin of the
    geometry-agnostic 2-D fixed-source SI path.  Verified: the converged
    per-ordinate flux hits the closed-form streaming equilibrium ``q/Σ_t``
    on a homogeneous reflective box, and SI ≡ Krylov flux shape agrees on a
    heterogeneous non-flat case
    (``tests/sn/solve/test_fixed_source_2d_equivalence.py``).
    """
    from orpheus.transport.fields.angular_flux import (
        AngularFlux,
    )
    from orpheus.transport.fields.angular_boundary_flux import (
        AngularBoundaryFlux,
    )
    from orpheus.transport.fields.scalar_flux import ScalarFlux

    ng = solver.ng
    N = sn_mesh.quad.N

    # ``q_ext_composite`` is the normalised composite RHS ``q = q_bulk ⊕ q_∂``
    # built once by :func:`_build_fixed_source_rhs` (Cardinal Rule 2). B.5.2:
    # q_ext IS a source — bulk per-ordinate-density ``AngularSourceSink`` +
    # boundary ``AngularBoundarySourceSink`` prescribed inflow (zero for vacuum /
    # reflective — the reflective inflow rides ``initial_guess`` /
    # ``rhs.boundary``, not ``q_ext``; a NON-vacuum prescribed inflow IS
    # carried in ``q_ext_composite.boundary``). The Krylov matvec composes
    # operator-output sources; q_ext is raveled type-agnostically as the
    # GMRES rhs ``b``.

    # B.5.2: the FLUX initial_guess (built FIRST so the GMRES restart is sized
    # from the FULL composite ravel) fixes the Krylov solution_template (the
    # return type); x0 stays all-zeros (bit-identical to the prior cold start).
    # The template carries the φ̂ moment axis at a multi-moment closure AND the
    # ψ½ state on a carrying mesh (the ravel keys on its ``to_flat``).
    q_a_ext = _system_a_member(q_ext_composite)
    if not isinstance(q_a_ext, TimedFullField):
        raise TypeError(
            f"fixed-source Krylov: the rhs's System-A member must be the "
            f"timed composite; got {type(q_a_ext).__name__}."
        )
    krylov_cold_start = _unwindowed_cold_start(
        sn_mesh, history_depth=q_a_ext.history_depth,
    )

    # ── Build the within-group system + Krylov driver (single source of
    # truth — :func:`~orpheus.sn.coupled_system.build_within_group_system` /
    # ``_within_group_krylov``; shared with the eigenvalue Krylov + SI
    # paths; the cached scattering operator injects through the cache
    # seam). ──────────────────────────────────────────────────────────
    # ERR-053 (#282 route (a)): ``restart`` MUST cover the FULL ravel —
    # bulk ⊕ trace ⊕ ψ½ (both systems on the coupled pair — the
    # CoupledField ``to_flat`` concatenates them, the B.2a conformance
    # closure).  A bulk-sized restart re-truncates GMRES on the trace+seed
    # DOFs.  Size it from the state the driver ravels (the multi-moment φ̂
    # axis + the trace + the ψ½ state all track automatically).
    system = build_within_group_system(
        sn_mesh, solver.mat_xs, scattering_op=solver.scattering_op,
        n2n_op=solver.n2n_op,
    )
    coupled = isinstance(system.implicit_operator, CoupledOperator)
    if coupled:
        # The coupled pair is born native (B.2d): the flux template pairs
        # with a zero ψ_B; ``q_ext_composite`` is already the coupled rhs.
        krylov_cold_start = _coupled_flux_state(krylov_cold_start, sn_mesh)
    krylov = _within_group_krylov(
        system.implicit_operator, *system.explicit_gains,
        n_dof=int(krylov_cold_start.to_flat().size),
        max_iter=max_inner, tol=inner_tol,
        corrector=corrector,
    )

    psi_typed, record = krylov.solve(
        q_ext_composite, initial_guess=krylov_cold_start,
    )
    # The end-of-solve CERTIFICATE (step 5, R-5.2) — the Krylov path is
    # always full-angular; the honest cross-check on the assembled
    # equation (the ERR-053 truncation family's independent catcher).
    _certify_within_group_exit(
        system, psi_typed, q_ext_composite,
        sn_mesh=sn_mesh, record=record,
        where="solve_sn_fixed_source[krylov]",
    )
    # The same triple, one line later, answering the OTHER question (#340
    # N6b).  The certificate asserts when the solve CLAIMED convergence and
    # is a no-op otherwise; this measures when it did not, and reports.
    # They are not folded together because the certificate raises and this
    # returns a number — one guard, two verbs.
    balance_defect = _exit_balance_defect(
        system.loss if coupled else _bare_loss_arm(system),
        psi_typed, q_ext_composite,
        sn_mesh=sn_mesh, record=record,
    )
    # System A's converged member feeds the Solution contract; System B's
    # rides ``Solution.radial_characteristic`` (B.2d DP-Solution).
    psi_full = _system_a_member(psi_typed)
    # D-H.1c stage 2 — the Krylov ravellable protocol unravels back to the
    # SOLUTION TEMPLATE (the flux ``initial_guess``), so the driver's static
    # iterate type (the operators' carrier) re-narrows to the timed flux
    # composite here. The parse reifies that template contract loudly
    # instead of assuming it.
    if not isinstance(psi_full, TimedFullField):
        raise TypeError(
            f"fixed-source Krylov: the converged iterate must echo the "
            f"timed flux template; got {type(psi_full).__name__}."
        )
    bulk = psi_full.interior
    if not isinstance(bulk, AngularFlux):
        raise TypeError(
            f"fixed-source Krylov: the converged iterate must carry an "
            f"AngularFlux bulk (the flux template); got {type(bulk).__name__}."
        )
    # Read bulk for scalar reduction (cell-average moment).
    phi = _average_moment_scalar(
        bulk.integrate_angular().values, sn_mesh,
    )
    # Issue #197 PR-TYPED-5: build typed Solution at the boundary.
    # R-1 Step 4 G1 — ``psi_full`` is the Krylov-converged composite; reuse
    # directly. (The former mesh / quadrature / materials parameters retired
    # in C4 — Solution never consumed them.)
    #
    # ⛔ This comment used to read "with the matvec's B1'' face residual on its
    # boundary". `[M]` #344, three ways: GMRES unravels into the flux
    # `solution_template` whose boundary is a zero `AngularBoundaryFlux`;
    # on a reflective/vacuum slab the trace reads |·|max = 5.213675 against a
    # bulk max of 5.259936 with the VACUUM-face inflow rows exactly 0.0 and the
    # reflective ones not (a residual block would be ≈0 on the reflective
    # face); and `test_declared_inflow_reaches_the_rhs.py` asserts this arm's
    # γ₋(xmin) equals the DECLARED inflow 2.5 to 18 ULP. It is a FLUX TRACE.
    # The residual reading describes the boundary block of the matvec's OUTPUT
    # (Aψ, which by BlockRole is a face residual) — a different object from the
    # solution vector's. Left uncorrected it is the one sentence that would make
    # a reader exempt this arm from the gauge below.
    #
    # #344 — projection here, warning at the public entry (see the SI arm).
    # ⛔ Rebuild, never in-place: `[M]` this arm's bulk and trace are two VIEWS
    # into one flat buffer that `psi_typed` also references and `:_system_b_member`
    # still reads.
    psi_full, gauge_correction = _exit_gauge_trace(psi_full, sn_mesh=sn_mesh)
    #
    # ⛔ This site used to write ``n_inner = len(residuals) + 1`` while its
    # SI sibling wrote ``len(residuals)`` — two conventions for one field,
    # undocumented, and BACKWARDS: it is SI whose pass count exceeds its
    # trajectory (it measures differences), while GMRES gets one callback
    # per iteration.  Both now read the producer's own count (#340 F11).
    history = IterationHistory(
        record=record, balance_defect=balance_defect,
        gauge_correction=gauge_correction,
    )
    # D-H.1c stage 2 (2026-05-28): psi_full IS already a TimedFullField;
    # no adapter wrap at the Solution boundary.
    return Solution(
        angular_flux=psi_full,
        scalar_flux=ScalarFlux(values=phi, space=sn_mesh.bulk_space),
        mesh=sn_mesh,
        keff=None,
        history=history,
        radial_characteristic=_system_b_member(psi_typed),
    )
