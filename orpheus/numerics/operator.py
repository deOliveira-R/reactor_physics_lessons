r"""Linear-operator algebra for matrix-free transport solvers.

The neutron transport eigenvalue problem and its fixed-source cousin
both reduce to compositions of a small set of linear operators acting
on a discrete flux distribution :math:`\psi`:

.. math::

    \Bigl(A - \sum_i g_i\Bigr)\,\psi \;=\; q
    \qquad \text{(fixed source)}

.. math::

    \Bigl(A - \sum_i g_i\Bigr)\,\psi \;=\; \tfrac{1}{k}\,F\,\psi
    \qquad \text{(eigenvalue)}

where :math:`A` is the INVERTIBLE resolvent operand and the
:math:`g_i` are the lagged coupling gains — for SN the binding is
:math:`A = L + C`, streaming plus collision, with gains
:math:`(S,\ B)`, the honest within-group operator :math:`L+C-S-B`;
the letter matters: project-wide, ``L`` names the STREAMING LEAF
(alone not invertible) and the invertible left-hand-side operand is
``A`` — :math:`S` is the scattering source operator, :math:`F` is the
fission source operator (never a gain in the eigenvalue posing: the
outer loop scales it by :math:`1/k`), and :math:`q` is an external
source (Trefethen & Bau 1997, §3.2 frame the matrix-free Krylov
view). For an SN sweep, an MoC
ray-tracer, a CP collision-probability matrix, or a diffusion BiCGSTAB
solve, the *outer* algebra is identical even though the *implementation*
of each operator differs by orders of magnitude in cost and structure.

This module installs the **algebra** as runtime-checkable Protocols.
Any object providing ``apply(x) -> Lx`` participates. Each further
ability is a per-axis THREE-LAYER surface (#226, Design C):

* a **predicate** (:attr:`~LinearOperator.is_invertible` /
  :attr:`~LinearOperator.is_adjointable`) — the runtime,
  instance-accurate truth, reading structure AND values (a
  zero-coefficient multiplier reports ``False``; a sum reports its
  leading term), recursive on composites;
* an **operator-returning method** (``inverse()`` / :attr:`~LinearOperator.H`)
  — the canonical act; ``.H`` lives on the base (one generic wrapper
  realization exists) and refuses EAGERLY (:class:`MissingAdjoint`),
  while ``inverse()`` lives per-class: a structurally-non-invertible
  type simply does not declare it (misuse is a *static* error), and a
  value-dependent type declares it and raises :class:`NotInvertible`;
* a **realization verb** (``solve`` / ``apply_transpose``) — present
  exactly where a native realization exists (the wrap-delegate family
  delegates through ``solve``; the composer transpose laws recurse
  through ``apply_transpose``), never as an exists-but-raises stub.

The checked bridges :func:`invertible` / :func:`adjointable` (PEP-647
``TypeGuard``) convert the runtime predicate into the static permission
at guarded call sites — you cannot obtain the permission without
executing the check. Composition mismatches still fail at COMPOSITION
time, never mid-iteration: the composers guard ``apply`` eagerly
(``TypeError``), ``.H`` gates at construction, and the value-dependent
``inverse()`` guards raise before any inverse object exists — so a
downstream :class:`scipy.sparse.linalg.LinearOperator` consumer never
silently hits a broken stub. Many transport operators have no
efficient inverse action — the scattering source S is never inverted
directly; the fission source F is rank-deficient — and their honest
surface is METHOD ABSENCE, not an advertising flag. See
:ref:`operator-algebra` for the full design rationale.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Generic,
    Optional,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np

from orpheus.numerics.vector import Vector

if TYPE_CHECKING:
    from orpheus.numerics.assembled_operator import SparseAssembledOperator
    from orpheus.numerics.functional import Functional
    from orpheus.numerics.space import FunctionSpace


# ── Two-parameter operator typevars (#65 / P4.5) ──────────────────────
# The honest operator type is ``LinearOperator[Domain, Codomain]``:
# ``apply`` maps an input carrier ``Domain`` to a (possibly distinct)
# output carrier ``Codomain`` — the carrier's ``(Representation, Role)``
# grid cell IS the operator's ``(Domain, Codomain)`` (the double-category
# 1-morphism between cells; see :ref:`operator-algebra`). The names are
# spelled in full (NOT ``Din``/``Cout``) because ``Domain`` already reads
# as "in" and ``Codomain`` as "out" — the abbreviation said nothing.
#
# ONE invariant pair (#65): :class:`LinearOperator` is now a
# SINGLE base — a ``@runtime_checkable`` Protocol that ALSO carries the
# algebra dunders as default-method bodies — so there is no longer a
# separate variant read-Protocol and an invariant impl-Mixin to reconcile.
# ``Domain``/``Codomain`` are therefore **invariant**: the variance the old
# split needed never reached the leaves (every leaf inherits the one base
# and the static carrier collapses to ``Vector`` at the numerics layer, so
# co/contra-variance bought nothing — and a contravariant TypeVar cannot be
# passed to the invariant composer bases). PEP-696 default
# ``Codomain = Domain`` makes ``LinearOperator[V] ≡ LinearOperator[V, V]``,
# so the endomorphic majority — and every existing single-parameter
# subscript site — keeps one parameter. The native
# ``typing.TypeVar(default=…)`` requires-python ``>=3.13``.
Domain = TypeVar("Domain", bound=Vector)  # operator input carrier
Codomain = TypeVar("Codomain", bound=Vector, default=Domain)  # operator output carrier
Cmid = TypeVar("Cmid", bound=Vector)  # OperatorProduct intermediate carrier
D2 = TypeVar("D2", bound=Vector)  # __matmul__ other-operand domain

# Composition-leg type parameters (C4 F1 — the composition wrappers are
# generic over their LEG types, so a named composition subclass carries its
# legs' identities at the type level and its accessors need no casts:
# ``StreamingCollisionOperator = OperatorSum[FF, FF, StreamingOperator,
# MultiplicationOperator]`` reads ``self.a`` as a ``StreamingOperator``).
# COVARIANT because a pinned composition must upcast to the defaulted
# spelling (``StreamingOperator.__add__`` returns ``StreamingCollisionOperator``
# where the base dunder contract says ``OperatorSum[Domain, Codomain]``) —
# which is also why the legs are read-only properties over ``Final``
# storage: covariance is sound only without a setter. PEP-696 defaults
# keep every existing ``OperatorSum[D, C]`` / bare spelling valid.
SummandA = TypeVar(
    "SummandA", bound="LinearOperator", covariant=True,
    default="LinearOperator[Domain, Codomain]",
)
SummandB = TypeVar(
    "SummandB", bound="LinearOperator", covariant=True,
    default="LinearOperator[Domain, Codomain]",
)
FactorA = TypeVar(  # A of ``A @ B`` — maps the intermediate to the output
    "FactorA", bound="LinearOperator", covariant=True,
    default="LinearOperator[Any, Codomain]",
)
FactorB = TypeVar(  # B of ``A @ B`` — maps the input to the intermediate
    "FactorB", bound="LinearOperator", covariant=True,
    default="LinearOperator[Domain, Any]",
)
ScaledOperand = TypeVar(
    "ScaledOperand", bound="LinearOperator", covariant=True,
    default="LinearOperator[Domain, Codomain]",
)

__all__ = [
    "LinearOperator",
    "SupportsInverse",
    "SupportsAdjoint",
    "SupportsAssembly",
    "BlockRole",
    "BulkOperator",
    "FullOperator",
    "BoundaryOperator",
    "NotInvertible",
    "MissingAdjoint",
    "MissingAssembly",
    "invertible",
    "adjointable",
    "assemblable",
    "IncompatibleOperatorComposition",
    "MatrixTooLarge",
    "InverseWrapMixin",
    "OperatorSum",
    "OperatorProduct",
    "AdjointOperator",
    "RieszLowerOperator",
    "RieszRaiseOperator",
    "ScaledOperator",
    "IdentityOperator",
    "PointwiseOperator",
    "ZeroMorphism",
    "ZeroOperator",
    "PermutationOperator",
    "TraceRestrictionOperator",
    "InverseMetricOperator",
    "DiagonalOperator",
    "RankOneOperator",
    "outer",
    "TensorProductOperator",
    "SumOfTensorProductsOperator",
]


# ───────────────────────────────────────────────────────────────────────
# Block-role classification (Issue #208)
# ───────────────────────────────────────────────────────────────────────
#
# On the direct-sum transport state space ``V = V_bulk ⊕ V_boundary`` a
# linear operator is, by the biproduct theorem, a 2×2 block matrix::
#
#     A = [ A_bb  A_bs ]      A_bb : bulk → bulk        A_bs : boundary → bulk
#         [ A_sb  A_ss ]      A_sb : bulk → boundary    A_ss : boundary → boundary
#
# :class:`BlockRole` classifies a leaf by WHICH blocks its action touches —
# the single fact :meth:`OperatorSum.apply` dispatches on, and the adjoint
# composition routes by. The classification is a partition
# (each leaf is exactly one role), and it lives on the INSTANCE (via the
# :attr:`LinearOperator.block_role` attribute), NOT the class, because
# the same generic operator class can play different roles in different
# contexts — e.g. :class:`IdentityOperator` is the bulk identity in one
# composition and a realized vacuum boundary law in another.


class BlockRole(Enum):
    r"""Which bulk/boundary blocks an operator's action touches.

    * :attr:`BULK` — only ``A_bb`` (bulk → bulk). The collision ``C``,
      scattering ``S`` and fission ``F`` operators: they read the bulk
      flux and write a bulk source/sink, with no boundary action.
    * :attr:`FULL` — has off-diagonal coupling (``A_bs`` and/or ``A_sb``).
      The streaming operator ``L``: it reads the inflow trace to seed the
      sweep and writes the outflow trace, coupling bulk ↔ boundary. The
      only irreducibly-full primitive.
    * :attr:`BOUNDARY` — only ``A_ss`` (boundary → boundary). A realized
      boundary law ``B`` (vacuum / reflective / albedo / white / periodic):
      it maps the outflow trace to the inflow trace, with no bulk action.
      The :class:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer` stamps
      this role on its realized outputs. ``B`` becomes a first-class
      algebra leaf — a sibling of ``L`` in ``(L_full + C − S − F − B)`` —
      when the boundary conditions are extracted from the streaming sweep;
      until that wiring lands ``B`` carries the role but is still consumed
      inside the sweep.
    """

    BULK = "bulk"
    FULL = "full"
    BOUNDARY = "boundary"


class SystemRole(Enum):
    r"""Which of the two coupled systems an operator's action maps between.

    The curvilinear-S\ :sub:`N` within-group system is a 2×2 coupled block
    operator over two systems (see
    ``docs/theory/foundations/coupled_block_operator.rst §coupled-block-operator``):

    .. math::

        \begin{bmatrix} A_{AA} & A_{AB} \\ A_{BA} & A_{BB} \end{bmatrix}
        \begin{bmatrix} \text{transport} \\ \text{ray} \end{bmatrix}

    * **System A** — the transport bulk ⊕ trace (the angular-flux
      :class:`~orpheus.transport.full_field.FullField`: a bulk field ⊕ its
      spatial boundary trace), governed by ``A_AA = L + C − S − B``.
    * **System B** — the ψ½ radial-characteristic ray (the starting-direction
      cells at each radial cell), governed by ``A_BB`` (the radial
      straight-characteristic march).

    This role is the COARSE two-system partition — orthogonal to
    :class:`BlockRole`, which refines the bulk ↔ boundary structure *within*
    System A. An operator carries at most one:

    * :attr:`A` — acts within System A only.
    * :attr:`B` — acts within System B only: the self-block ``A_BB``
      (:class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicOperator`)
      and the ray boundary ``B_b``
      (:class:`~orpheus.sn.operators.boundary.RadialCharacteristicBoundaryOperator`).
    * :attr:`COUPLED` — maps BETWEEN the systems (an off-diagonal block, or the
      assembled 2×2): the ray→bulk seed ``A_AB``
      (:class:`~orpheus.sn.operators.radial_characteristic.RadialCharacteristicSeeding`),
      the bulk→ray fold ``A_BA``, and the assembled ``CoupledOperator``.

    Reading each role as the SET of systems its action touches (``A = {A}``,
    ``B = {B}``, ``COUPLED = {A, B}``), a sum touches the union — see
    :func:`_join_system_roles`. Operators outside the two-system decomposition
    — every model-generic family (diffusion / CP / MoC) AND the model-generic
    reaction leaves ``C`` / ``S`` / ``F`` that a curvilinear-S\ :sub:`N` context
    COMPOSES into System A but that carry no intrinsic two-system membership —
    leave :attr:`~LinearOperator.system_role` at its ``None`` default: the
    conservative "not part of the ψ½ augmentation" reading, exactly as an
    unclassified :attr:`~LinearOperator.block_role` is ``None``.
    """

    A = "system_a"
    B = "system_b"
    COUPLED = "coupled"


def _join_block_roles(
    a: Optional["BlockRole"], b: Optional["BlockRole"],
) -> Optional["BlockRole"]:
    r"""The block role of a SUM ``A + B`` — the union of the blocks touched.

    Reading a role as the *set* of blocks its action touches
    (:attr:`BlockRole.BULK` = ``{bulk}``, :attr:`BlockRole.BOUNDARY` =
    ``{boundary}``, :attr:`BlockRole.FULL` = ``{bulk, boundary}``), the sum
    touches the union: ``BULK ⊔ BULK = BULK``, ``BOUNDARY ⊔ BOUNDARY =
    BOUNDARY``, and any mix (or anything with ``FULL``) is ``FULL``. So the
    join is simply *"same role stays, anything different becomes FULL"*. If
    either operand is unclassified (``None`` — a generic operator outside the
    SN bulk/boundary partition) the sum is unclassified too: ``None``
    propagates (a conservative "don't know" rather than a guessed role).

    This is what lets ``(L + C - S - F - B)`` carry its role BY
    CONSTRUCTION (no hand-stamped tag): ``L`` is ``FULL``, ``C``/``S``/``F``
    are ``BULK``, ``B`` is ``BOUNDARY`` → every within-group loss sum joins
    to ``FULL``, exactly the irreducibly bulk↔boundary-coupling streaming
    role.

    Twin of :func:`_join_system_roles` (the two-system analogue — ``COUPLED``
    there plays the top-of-lattice role ``FULL`` plays here): both are the SAME
    union-lattice join (two atoms + a top + ``None`` propagation) on ORTHOGONAL
    role axes. Kept as a deliberate twin while only two axes exist — a generic
    ``RoleAxis`` join would need ``setattr``-driven dispatch that regresses the
    #226 pyright ratchet. **Collapse trigger:** a THIRD parallel role axis (a
    DSA / multiphysics role) makes the shared abstraction pay — unify then.
    """
    if a is None or b is None:
        return None
    return a if a is b else BlockRole.FULL


def _join_system_roles(
    a: Optional["SystemRole"], b: Optional["SystemRole"],
) -> Optional["SystemRole"]:
    r"""The system role of a SUM ``A + B`` — the union of the systems touched.

    Reading a role as the *set* of systems its action touches
    (:attr:`SystemRole.A` = ``{A}``, :attr:`SystemRole.B` = ``{B}``,
    :attr:`SystemRole.COUPLED` = ``{A, B}``), the sum touches the union:
    ``A ⊔ A = A``, ``B ⊔ B = B``, and any mix (or anything with ``COUPLED``) is
    ``COUPLED``. So the join is *"same role stays, anything different becomes
    COUPLED"* — the two-system analogue of :func:`_join_block_roles` with
    ``COUPLED`` playing the top-of-lattice role that ``FULL`` plays there. If
    either operand is unclassified (``None`` — an operator outside the
    two-system decomposition) the sum is unclassified too: ``None`` propagates
    (a conservative "don't know" rather than a guessed role).
    """
    if a is None or b is None:
        return None
    return a if a is b else SystemRole.COUPLED


def _agreed_space(
    ops: tuple, role: str, owner: str,
) -> Optional["FunctionSpace"]:
    r"""The space a COMMUTATIVE composite's operands agree on, or ``None``.

    The two composites whose operands commute — the sum :math:`A + B` and the
    tensor product :math:`A \otimes B` on disjoint axes — cannot resolve their
    spaces BY POSITION, because position is not part of their identity: if
    ``(A & B).domain`` were ``ops[0].domain``, the derived space of an
    order-INDEPENDENT operator would be order-DEPENDENT, contradicting the
    type's own defining law. So they resolve by **agreement** instead, and this
    is that law, written once:

    * every operand that declares the space must declare the SAME one, else
      the composite is ill-posed and this raises;
    * an operand that declares nothing contributes nothing — silence is not
      disagreement (the module-wide ``None`` semantics: spaces are optional
      while the tree migrates, and a skipped check is never a failed one);
    * all silent ⟹ ``None``, the composite is unbound like its operands.

    Contrast :class:`OperatorProduct`, the one NON-commutative composite:
    ``A @ B`` genuinely maps ``B.domain → A.codomain``, so it resolves by
    position and an unbound factor there poisons that end. The law follows the
    algebra in both cases.

    ⚠ **Why agreement is enough, and where it would stop being enough.** A
    factor's binding in this module is a WHOLE-space binding, not a per-leg
    one: a :class:`PermutationOperator` with ``axis=0`` acting on a
    ``(4, 3)`` trace declares ``domain.shape == (4, 3)`` — both axes — because
    it broadcasts on the rest. So the factors of a tensor product are not
    describing separate legs to be multiplied together; each bound factor is
    describing the WHOLE space, and at most one of them can be non-trivial.
    Should genuine per-leg bindings ever arrive (an energy-dependent group
    kernel bound on its own axis), agreement becomes the wrong law and a
    product-space constructor is what has to be built — which is what the
    docstrings on the two tensor-product classes say, rather than silently
    picking one leg.

    ⚠ **The message keeps the phrase** ``equal <role>s`` **deliberately.** It
    predates this helper (it was :class:`OperatorSum`'s own inline wording) and
    two gates in ``tests/sn/operators/test_typed_residual_evaluation.py`` pin it
    as the provenance marker that says *this* guard fired and not some
    incidental raise elsewhere. ``owner`` is prefixed so the marker still
    identifies WHICH composite refused now that the law is shared.
    """
    declared = [
        (op, space)
        for op in ops
        if (space := getattr(op, role, None)) is not None
    ]
    if not declared:
        return None
    first_op, agreed = declared[0]
    for op, space in declared[1:]:
        if space != agreed:
            raise IncompatibleOperatorComposition(
                f"{owner} requires equal {role}s; "
                f"{type(first_op).__name__} declares {agreed!r} while "
                f"{type(op).__name__} declares {space!r}."
            )
    return agreed


class _BlockRoleMeta(type):
    r"""Metaclass making ``isinstance(op, BulkOperator)`` read ``op.block_role``.

    The role markers (:class:`BulkOperator`, :class:`FullOperator`,
    :class:`BoundaryOperator`) are never instantiated and carry no
    state. They exist so the block-role classification reads like the
    domain (``isinstance(L, FullOperator)``) while the single source of
    truth stays the :attr:`~LinearOperator.block_role` enum on the
    operator instance. Exclusivity is automatic: an operator carries one
    ``block_role`` and therefore satisfies at most one marker.

    A value-based check is required (not a plain ``@runtime_checkable``
    :class:`Protocol`) because Protocols can only test attribute
    *presence*, never the *value* the partition discriminates on — every
    operator has a ``block_role`` attribute, so a structural Protocol would
    match them all.
    """

    _role: "BlockRole"

    def __instancecheck__(cls, obj: object) -> bool:
        return getattr(obj, "block_role", None) is cls._role


class BulkOperator(metaclass=_BlockRoleMeta):
    r"""``isinstance`` marker for a :attr:`BlockRole.BULK` operator (``A_bb`` only)."""

    _role = BlockRole.BULK


class FullOperator(metaclass=_BlockRoleMeta):
    r"""``isinstance`` marker for a :attr:`BlockRole.FULL` operator (off-diagonal coupling)."""

    _role = BlockRole.FULL


class BoundaryOperator(metaclass=_BlockRoleMeta):
    r"""``isinstance`` marker for a :attr:`BlockRole.BOUNDARY` operator (``A_ss`` only).

    The realized boundary laws produced by the functional method
    realizers —
    :meth:`~orpheus.sn.boundary.realizer.SNBoundaryRealizer.realize`
    (vacuum / reflective / white / albedo / periodic / prescribed inflow)
    and
    :meth:`~orpheus.diffusion.boundary_realizer.DiffusionBoundaryRealizer.realize`
    (the albedo family incl. zero-flux, #290) — carry
    :attr:`BlockRole.BOUNDARY` via
    :func:`~orpheus.geometry.boundary.stamp_boundary_role`.

    **EVERY realizable law is stamped, prescribed inflow included** (P3,
    2026-08-05). Until P3 this docstring named it as the one exception,
    on the reading that the rank-0 affine source is the boundary *source*
    ``q.boundary`` rather than a linear boundary operator ``B``. The
    affine split is real and unchanged — see
    :ref:`bc-affine-source-channel` — but it does not put the law outside
    this marker: the law is ``γ₋ψ = L γ₊ψ + q``, the realizer realizes
    ``L``, and for prescribed inflow ``L = 0``. A zero morphism is an
    ordinary linear boundary operator, and it is the same one vacuum
    realizes to.

    ⛔ The exception was also not doing the job it appeared to do. An
    unstamped leaf is NOT excluded from ``B``:
    :attr:`~orpheus.sn.operators.boundary.SNBoundaryOperator._face_laws`
    collects every face's law with no ``block_role`` filter, so the
    pre-P3 AFFINE operator reached the block regardless — measured
    ``|B(0)| = q`` and ``|B(2x) − 2B(x)| = q``, and on the Krylov path a
    raised ``ConvergenceCertificateError``, because an affine map breaks
    the Arnoldi relation GMRES's residual depends on. The stamp is
    honest metadata about a leaf's role; it is not, and never was, the
    fence.
    """

    _role = BlockRole.BOUNDARY


class NotInvertible(TypeError):
    r"""Asked for the inverse of an operator that cannot produce one.

    The INVERSE-axis refusal: raised **eagerly** by
    :meth:`inverse` overrides (and the inverse-family constructors) when
    the operator's :attr:`~LinearOperator.is_invertible` is ``False`` —
    the VALUE-dependent arm of the two-kinds split. A zero-coefficient
    :class:`DiagonalOperator`, a sum whose leading term is not
    invertible, a product with a singular factor: the TYPE supports
    inversion, this INSTANCE refuses, at construction of the inverse and
    never mid-iteration. (The STRUCTURAL arm — :class:`ZeroOperator`,
    masks, source dyads, for which no inverse exists mathematically —
    does not declare :meth:`inverse` at all, so misuse there is a
    *static* error, not this exception.) ``TypeError`` parentage carries
    the retired ``MissingCapability``'s public contract forward — no
    ``except`` clause written against the old gate changes meaning.
    """


class MissingAdjoint(TypeError):
    r"""Asked for the Hilbert adjoint of an operator that has none.

    The ADJOINT-axis refusal: raised **eagerly**
    by :meth:`LinearOperator.adjoint` / :attr:`LinearOperator.H` when
    :attr:`~LinearOperator.is_adjointable` is ``False`` — at wrapper
    CONSTRUCTION, never lazily at the first ``.apply`` (the pre-carve
    behaviour). Also the refusal of the raw-transpose realization verb
    (``apply_transpose``) on composites whose operands cannot all
    transpose. ``TypeError`` parentage mirrors :class:`NotInvertible`.
    """


class MissingAssembly(TypeError):
    r"""Asked for the sparse assembly of an operator that has none.

    The ASSEMBLY-axis refusal — the third sibling of
    :class:`NotInvertible` (inverse axis) and :class:`MissingAdjoint`
    (adjoint axis): raised **eagerly** by the composer ``assemble()``
    bodies when an operand is not :attr:`~LinearOperator.is_assemblable`
    — a structural emission exists only where a leaf declared one, and a
    composite can recurse (Sum → ``+``, Product → ``@``, Scaled →
    scalar ``*``) only when every leg emits. Operators without a
    stencil realization simply do not declare :meth:`assemble` (misuse
    is a *static* error); the probing
    :meth:`~LinearOperator.as_matrix` remains their total (dense)
    Mat-functor. ``TypeError`` parentage mirrors the sibling refusals.
    """


class IncompatibleOperatorComposition(ValueError):
    """A composition's operands carry incompatible function spaces.

    Raised at composition time when two operators with declared
    :attr:`domain`/:attr:`codomain` carry shapes that cannot be combined
    (Sum: ``a.domain != b.domain`` or ``a.codomain != b.codomain``;
    Product ``A @ B``: ``A.domain != B.codomain``). The check is
    skipped when either operand has ``None`` for its domain or codomain
    — backward-compatible with operators predating Issue 9.6 that
    carry no function-space metadata.
    """


class MatrixTooLarge(RuntimeError):
    r"""A :meth:`LinearOperator.as_matrix` materialization exceeds its size gate.

    A **resource effect on a TOTAL functor**: every
    linear operator on a finite-dimensional space *has* a matrix — the
    functor ``Op → Mat`` is total — but materializing it commits
    :math:`O(n^2)` memory and :math:`n` applications, which this
    environment may refuse. That is why this is a ``RuntimeError``
    (a refused resource commitment), NOT a ``TypeError``/``ValueError``
    (the request is neither ill-typed nor ill-valued), and why there is
    deliberately NO ``is_materializable`` predicate alongside
    :attr:`LinearOperator.is_invertible` / ``is_adjointable``: those are
    *structural restriction* predicates (they read the operator's
    structure and values), whereas the size gate is a pure resource
    precheck that reads nothing but a dimension. Callers that want the
    fallback pattern write ``try: A.as_matrix() except MatrixTooLarge:
    <iterative path>`` — or raise the per-call ``max_dimension``.
    """


def _resolve_basis_shape(
    op: "LinearOperator",
    basis_shape: "tuple[int, ...] | None",
) -> tuple[int, ...]:
    r"""Resolve the basis shape a materialization iterates over.

    The SINGLE SOURCE for the resolution rule shared by
    :meth:`LinearOperator.as_matrix` and the eager
    :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
    constructor (which must know the resolved shape to reshape solutions
    back into carriers): an explicit ``basis_shape`` wins; otherwise the
    operator's own :attr:`~LinearOperator.domain` supplies its ``shape``;
    an operator with neither is un-materializable *as posed* and the
    caller is told both remedies.
    """
    if basis_shape is not None:
        return tuple(int(d) for d in basis_shape)
    domain = op.domain
    if domain is None:
        raise ValueError(
            f"as_matrix on {type(op).__name__}: the operator carries no "
            f"domain FunctionSpace, so the basis shape cannot be derived. "
            f"Either construct the operator with a space, or pass an "
            f"explicit basis_shape= (the element shape apply consumes, "
            f"e.g. (ng, 1) for an infinite-medium group operator)."
        )
    return tuple(domain.shape)


@runtime_checkable
class LinearOperator(Protocol[Domain, Codomain]):
    r"""Contract for a matrix-free linear operator on a flux vector.

    Any object exposing :meth:`apply` participates. The further
    abilities are per-axis structural surfaces (the module docstring's
    three layers): the recursive predicates
    :attr:`is_invertible`/:attr:`is_adjointable` are the runtime truth,
    ``inverse()``/:attr:`H` the operator-returning acts, and
    ``solve``/``apply_transpose`` the per-class realization verbs —
    declared exactly where a native realization exists, never as
    stubs. There is no capability registry to keep in sync: the single
    source of truth for what an operator can do is the operator's own
    structure and values, read through the predicates.

    Composition operators (:class:`OperatorSum`, :class:`OperatorProduct`,
    :class:`ScaledOperator`) are wired through ``__add__``, ``__sub__``,
    ``__mul__`` (scalar), and ``__matmul__`` (operator product) so the
    typical algebra of the Boltzmann transport equation,
    :math:`(L + C - S - B)`, can be built with the natural Python syntax.
    The composites derive their predicates recursively per the closure
    laws documented in :ref:`operator-algebra`.

    Notes
    -----
    Shape and dtype are deliberately not part of the protocol. numpy
    duck-typing (broadcasting + dtype promotion) handles them at
    ``apply`` call time. Imposing a static shape would forbid operators
    whose action shape depends on the input (a multi-group transport
    sweep can output a different layout than its input vector). If a
    consumer needs shape information, it can probe ``op.apply(x)`` on a
    known-size probe vector once at setup.
    """

    #: Block-role classification (Issue #208) — see
    #: :class:`BlockRole`. A single enum value: the role is a
    #: *partition* (an operator is exactly one of bulk/full/boundary),
    #: so one enum makes the illegal "BULK and FULL at once" state
    #: unrepresentable; a set would not.
    #:
    #: ``None`` = unclassified — the default for the generic algebra
    #: (composition operators derive their role from operands).
    #: ``None`` satisfies none of the :class:`BulkOperator` /
    #: :class:`FullOperator` markers. Concrete leaves override with a
    #: **plain (unannotated) class attribute** ``block_role = BlockRole.X``
    #: — NOT a ``ClassVar[...]`` annotation, which under ``from __future__
    #: import annotations`` is mis-detected by the ``@dataclass`` machinery
    #: as a field (it became a string and the ClassVar heuristic missed
    #: it). The annotation HERE is a **plain instance attribute** (NOT
    #: ``ClassVar``) precisely because the composers
    #: (:class:`OperatorSum` / :class:`ScaledOperator` /
    #: :class:`AdjointOperator`) and the
    #: :func:`~orpheus.geometry.boundary.stamp_boundary_role` stamp assign
    #: ``self.block_role`` per-instance (the role is DERIVED from operands,
    #: not fixed by the class). A ``ClassVar`` would (correctly) reject
    #: that instance assignment. This base is not a ``@dataclass``, so the
    #: annotation is never field-processed; the leaves' unannotated
    #: class-attr override keeps the class-level read
    #: (``ScatteringOperator.block_role``) working.
    block_role: Optional[BlockRole] = None

    #: The COARSE two-system membership (:class:`SystemRole` — System A / System
    #: B / COUPLED) of a curvilinear-S_N augmented operator, orthogonal to
    #: :attr:`block_role` (which refines the bulk↔boundary structure *within*
    #: System A). ``None`` for every operator outside the ψ½ two-system
    #: decomposition. Derived through composition exactly as :attr:`block_role`
    #: is — the passthrough (:class:`AdjointOperator`, :class:`ScaledOperator`)
    #: and the :func:`_join_system_roles` union (:class:`OperatorSum`).
    system_role: Optional[SystemRole] = None

    @property
    @abstractmethod
    def domain(self) -> Optional["FunctionSpace"]:
        """The function space this operator consumes — DEMANDED of every
        subclass (the S4-amendment, 2026-08-22: an operator is not an
        operator without its two spaces, and this base no longer supplies
        a silent ``None`` default an implementer can inherit unawares).

        Every class must ANSWER, in one of four ways:

        * **bind** — return a space stored/threaded at construction (the
          :class:`~orpheus.transport.operators.fission.FissionOperator`
          precedent: space MANDATORY, non-Optional return);
        * **derive** — compute it from operands/held data (the
          composites' agreement/position laws, :class:`InverseWrapMixin`'s
          swap, a mesh-holding leaf reading its mesh's space);
        * **the pointwise law** — :class:`PointwiseOperator` members
          answer ``None`` BY LAW (space-polymorphic: the domain is the
          operand's space at operation time, discriminated by type);
        * **a documented Optional** — an explicit override returning
          ``FunctionSpace | None`` whose docstring names the campaign
          that owns its mandatory flip. ⛔ This arm is EMPTY since CS4c
          step 6 item 6.4 (2026-09-07): S/C/iso flipped at CS4c K2b /
          steps 2–5 (``BoundOperator``'s mandatory kw-only ends), L and
          the three boundary leaves at item 6.4 — the last documented
          Optionals in the tree. It stays listed so a future leaf that
          reaches for it knows the answer is "bind or derive", never a
          new Optional.

        The post-flip law: a BOUND leaf's ends are never ``None`` — only
        the pointwise-law members answer ``None``, BY LAW. What was never
        legal is SILENCE.
        """
        ...

    @property
    @abstractmethod
    def codomain(self) -> Optional["FunctionSpace"]:
        """The function space this operator produces — DEMANDED of every
        subclass. See :attr:`domain` for the four legitimate answers and
        the Optional-until-terminal semantics.
        """
        ...

    # ------------------------------------------------------------------
    # Per-axis structural predicates (#226 inverse-as-operator carve).
    # Each is the RUNTIME advertisement for one operator-returning
    # method (:meth:`inverse` / :meth:`H`); the propagation LAW lives in
    # the composer method bodies, and these predicates compute the
    # matching "does it work?" answer recursively from the operands —
    # never a cached registry that can drift. The static bridges are
    # :func:`invertible` / :func:`adjointable` (narrowing to
    # :class:`SupportsInverse` / :class:`SupportsAdjoint`).
    # ------------------------------------------------------------------

    @property
    def is_invertible(self) -> bool:
        r"""Whether this operator can produce its inverse OPERATOR (:meth:`inverse`).

        The RUNTIME, instance-accurate predicate. Unlike
        ``isinstance(op, SupportsInverse)`` — which
        sees only class-level method presence — this property reads the
        operator's actual structure and values, so it correctly reports a
        sum with a non-invertible LEADING term as non-invertible and a
        zero-coefficient
        :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
        as singular (``min|f| = 0``). Composites derive it recursively from
        their operands; the default is ``False`` — an operator is
        invertible only by explicit override.
        """
        return False

    @property
    def is_adjointable(self) -> bool:
        r"""Whether this operator exposes a Hilbert adjoint (:attr:`H` / transpose).

        The RUNTIME predicate for the adjoint axis. The
        transpose-of-a-sum law :math:`(A+B)^{\mathsf T} = A^{\mathsf T} +
        B^{\mathsf T}` is realised in the composer method bodies; this
        predicate is the matching *advertisement* —
        ``(A+B).is_adjointable == A.is_adjointable and B.is_adjointable`` —
        structurally computed rather than cached in a string set. Default
        ``False``; an operator with a working ``apply_transpose`` overrides.
        """
        return False

    @property
    def is_metric_free_adjoint(self) -> bool:
        r"""Whether this operator's Hilbert adjoint needs NO metric — i.e.
        its Euclidean transpose IS its Hilbert adjoint in every shipped
        (diagonal) inner product.

        ``True`` exactly on the pointwise/multiplier stratum and its
        compositions: a real multiplier commutes with every diagonal
        metric, and sums/products/scalings of metric-free operators stay
        metric-free (for a composite only the END metrics enter the
        adjoint sandwich — interior metrics cancel — so all-metric-free
        composites keep the Euclidean identity). The S4-amendment's
        unbound-``.H`` refusal reads this: an UNBOUND operator may take
        ``.H`` iff this is ``True``, because then and only then is the
        Euclidean fallback the honest answer rather than the R2 hazard.
        Default ``False``; :class:`PointwiseOperator` overrides ``True``;
        composites derive recursively (the module's predicate
        discipline — computed from operands, never cached).
        """
        return False

    @property
    def is_assemblable(self) -> bool:
        r"""Whether this operator can emit its sparse assembly (:meth:`assemble`).

        The ASSEMBLY axis — the third structural surface beside
        :attr:`is_invertible` / :attr:`is_adjointable`: ``True`` iff a
        structural ``(row, col, value)`` emission of this operator
        exists (the stencil-assembly third consumption mode of the
        per-cell closure algebra; see
        :class:`~orpheus.numerics.assembled_operator.SparseAssembledOperator`).
        Composites derive it recursively (a sum/product assembles iff
        both legs do — the homomorphism laws in their ``assemble()``
        bodies); the default is ``False`` — an operator is assemblable
        only by explicit override, and the probing :meth:`as_matrix`
        remains the total dense Mat-functor for everything else. The
        static bridge is :func:`assemblable` (narrowing to
        :class:`SupportsAssembly`).
        """
        return False

    def apply(self, x: Domain, /) -> Codomain:
        r"""Return :math:`L\,x`.

        Mandatory. Every concrete :class:`LinearOperator` must implement
        this (the body here is the Protocol contract stub); the
        composers guard its presence eagerly at composition time.

        The two type variables express the operator honestly: ``apply``
        maps an input carrier :data:`Domain` to a (possibly distinct)
        output carrier :data:`Codomain`. The endomorphic majority (``C``,
        the loss solve, the flat ``np.ndarray`` of the scipy
        serialization boundary) is the special case ``Codomain == Domain``,
        spelled with a single parameter via the PEP-696 default
        (``LinearOperator[V] ≡ LinearOperator[V, V]``); the
        source-producing operators ``S``/``F`` are the genuine
        ``Codomain ≠ Domain`` case (flux carrier → source/sink carrier).
        """
        ...

    # ------------------------------------------------------------------
    # Algebra dunders — default-method bodies (#65)
    #
    # These carry real bodies ON this Protocol, so an explicit subclass
    # ``class Foo(LinearOperator[A, B])`` inherits BOTH the ``apply``
    # contract AND the natural Python algebra (``+``, ``-``, ``*`` scalar,
    # ``@`` composition) with no separate mixin. The dunders delegate to
    # the composer constructors, which enforce the capability-closure laws.
    # ------------------------------------------------------------------

    def __add__(
        self, other: "LinearOperator[Domain, Codomain]",
    ) -> "OperatorSum[Domain, Codomain]":
        return OperatorSum(self, other)

    def __radd__(
        self, other: "LinearOperator[Domain, Codomain]",
    ) -> "OperatorSum[Domain, Codomain]":
        return OperatorSum(other, self)

    def __sub__(
        self, other: "LinearOperator[Domain, Codomain]",
    ) -> "OperatorSum[Domain, Codomain]":
        return OperatorSum(self, ScaledOperator(-1.0, other))

    def __rsub__(
        self, other: "LinearOperator[Domain, Codomain]",
    ) -> "OperatorSum[Domain, Codomain]":
        return OperatorSum(other, ScaledOperator(-1.0, self))

    def __mul__(self, other: float) -> "ScaledOperator[Domain, Codomain]":
        if not isinstance(other, (int, float, np.floating, np.integer)):
            return NotImplemented
        return ScaledOperator(float(other), self)

    def __rmul__(self, other: float) -> "ScaledOperator[Domain, Codomain]":
        return self.__mul__(other)

    def __neg__(self) -> "ScaledOperator[Domain, Codomain]":
        r"""Unary minus: return :math:`-A` as ``ScaledOperator(-1.0, A)``.

        Pythonic completion of the ``__sub__`` family — when ``A - B``
        works (which ``__sub__`` already provides via the ``A +
        ScaledOperator(-1.0, B)`` rewrite) Python's arithmetic
        convention is that ``-A`` should also work (adjoint-flux sign
        flips, residual corrections ``-L @ delta``, Jacobi splitting).
        """
        return ScaledOperator(-1.0, self)

    def __truediv__(self, scalar: float) -> "ScaledOperator[Domain, Codomain]":
        r"""Scalar division: ``A / α`` is ``(1/α) * A``.

        Reads more naturally than the reciprocal-multiply form
        ``(1.0 / α) * A`` — eigenvalue/Krylov normalisation
        (``F / k_eff``), homogenisation averages.

        Raises :class:`TypeError` if ``scalar`` is not numeric.
        Division by zero raises :class:`ZeroDivisionError` per the
        standard Python convention (handled by ``1.0 / scalar``).
        """
        if not isinstance(scalar, (int, float, np.floating, np.integer)):
            return NotImplemented
        return ScaledOperator(1.0 / float(scalar), self)

    def __matmul__(
        self, other: "LinearOperator[D2, Domain]",
    ) -> "OperatorProduct[D2, Codomain]":
        # ``self`` (Domain → Codomain) ∘ ``other`` (D2 → Domain): the
        # intermediate carrier is ``self``'s domain ``Domain`` =
        # ``other``'s codomain — captured as ``OperatorProduct``'s
        # ``Cmid``, giving the honest ``D2 → Codomain``.
        return OperatorProduct(self, other)

    def __and__(self, other: "LinearOperator[Domain]") -> "TensorProductOperator":
        r"""Return :math:`A \otimes B` — the per-axis tensor-product operator.

        For two operators acting on independent tensor axes, ``A & B``
        produces the operator whose action is "apply A on its axis, apply
        B on its axis" (sequentially; commutative because axes are
        disjoint). See
        ``docs/theory/foundations/operator_tensor_network.rst §tensor-network-decomposition``.

        If either operand is already a :class:`TensorProductOperator`,
        the result is flattened so ``(A & B) & C`` and ``A & (B & C)``
        produce the same instance ``TensorProductOperator((A, B, C))``.
        """
        return TensorProductOperator._build(self, other)

    def __rand__(self, other: "LinearOperator[Domain]") -> "TensorProductOperator":
        return TensorProductOperator._build(other, self)

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`apply`. Lets user code write ``A(x)``.

        Accepts ``*args, **kwargs`` so any multi-argument ``apply``
        composes ergonomically (``op(x, y)`` reads as math); the generic
        forwarding is retained for future multi-argument operators.
        """
        return self.apply(*args, **kwargs)

    def __pow__(
        self: "LinearOperator[Domain, Domain]", n: int,
    ) -> "LinearOperator[Domain, Domain]":
        r"""Return :math:`A^n` for non-negative integer ``n``.

        Only an *endomorphic* operator is powerable (``A @ A`` requires
        ``A``'s codomain to equal its domain) — the precondition lives in
        the ``self`` annotation, so ``S**2`` on a flux→source ``S`` is a
        call-site type error, not a runtime surprise.

        ``n == 0`` returns :class:`IdentityOperator`. ``n == 1``
        returns ``self`` unchanged. ``n >= 2`` builds the composition
        ``A @ A @ ... @ A`` via repeated :meth:`__matmul__`. Negative
        powers raise :class:`ValueError` — operator inverse construction
        is not part of this API; use the operator's :meth:`solve`
        capability directly when an inverse is needed.
        """
        if not isinstance(n, (int, np.integer)):
            return NotImplemented
        if n < 0:
            raise ValueError(
                "operator inverse not constructed via __pow__; "
                "consult the operator's solve() capability for inverse "
                "actions."
            )
        if n == 0:
            return IdentityOperator()
        if n == 1:
            return self
        result: "LinearOperator[Domain, Domain]" = self
        for _ in range(n - 1):
            result = result @ self
        return result

    # ------------------------------------------------------------------
    # Adjoint
    # ------------------------------------------------------------------

    def adjoint(self) -> "LinearOperator[Codomain, Domain]":
        r"""Return the Hilbert adjoint :math:`A^*`.

        The adjoint SWAPS the carriers: for ``A : Domain → Codomain`` the
        adjoint is ``A^* : Codomain → Domain`` (it maps the codomain back
        to the domain), so the return type is the swapped
        ``[Codomain, Domain]``.

        For an operator :math:`A : V \to W` with diagonal inner-product
        weights :math:`w_V` (on the domain) and :math:`w_W` (on the
        codomain), the Hilbert adjoint satisfies

        .. math::

           \langle A x, y \rangle_W \;=\; \langle x, A^* y \rangle_V

        which gives :math:`A^* y = (1/w_V) \odot
        \mathrm{apply\_transpose}(w_W \odot y)`. When both weight
        arrays are ``None`` (Euclidean inner product on both sides)
        the adjoint reduces to the representation transpose.

        The returned wrapper preserves :meth:`apply` (= adjoint
        action) and swaps :attr:`domain` ↔ :attr:`codomain`.

        Raises
        ------
        MissingAdjoint
            **Eagerly, here at construction** when
            this operator is not :attr:`is_adjointable` — a wrapper that
            could only fail at its first ``.apply`` is the broken-stub
            anti-pattern this module refuses. The :func:`adjointable`
            guard doubles as the static bridge: the wrapper's
            constructor consumes the narrowed :class:`SupportsAdjoint`.
        """
        if not adjointable(self):
            raise MissingAdjoint(
                f"{type(self).__name__} is not adjointable — .H/.adjoint() "
                f"requires is_adjointable=True (a working apply_transpose "
                f"on every constituent). The Hilbert adjoint of this "
                f"operator does not exist as posed."
            )
        return AdjointOperator(self)

    @property
    def H(self) -> "LinearOperator[Codomain, Domain]":
        """Alias for :meth:`adjoint` — the Hilbert-adjoint vocabulary
        (``A.H`` reads as "A dagger"). Swaps the carriers
        ``[Domain, Codomain] → [Codomain, Domain]`` (see
        :meth:`adjoint`)."""
        return self.adjoint()

    def dual(self) -> "LinearOperator[Codomain, Domain]":
        r"""Return the dual arrow :math:`A^{\mathsf T} : W^* \to V^*`.

        The METRIC-FREE half of the Hilbert adjoint: the representation
        transpose, carried between the dual spaces. For ``A : V → W`` the
        dual maps ``W* → V*`` — the same array arithmetic as
        :meth:`apply_transpose`, with the arrow bookkeeping made explicit
        (:meth:`~orpheus.numerics.space.FunctionSpace.dual` on each end).
        The Hilbert adjoint is then the theorem-shaped composition

        .. math::

           A^{*} \;=\; \sharp_V \circ A^{\mathsf T} \circ \flat_W
           \qquad\text{(``domain.riesz_raise ∘ A.dual() ∘
           codomain.riesz_lower``)}

        realized by :class:`AdjointOperator` — the dual is the middle
        factor, and the metrics live entirely in the Riesz legs
        (:class:`RieszLowerOperator` / :class:`RieszRaiseOperator`).

        Involution: ``A.dual().dual() is A`` (object identity — the
        wrapper's own :meth:`_DualOperator.dual` returns the inner).

        Raises
        ------
        MissingAdjoint
            Eagerly, when this operator is not :attr:`is_adjointable`
            (no working ``apply_transpose``) — same broken-stub-refusing
            style as :meth:`adjoint`.
        """
        if not adjointable(self):
            raise MissingAdjoint(
                f"{type(self).__name__} is not adjointable — .dual() "
                f"requires is_adjointable=True (a working apply_transpose); "
                f"the dual arrow IS the transpose, carried between the "
                f"dual spaces."
            )
        return _DualOperator(self)

    # ------------------------------------------------------------------
    # Materialization — the functor OUT of the operator category
    # ------------------------------------------------------------------

    def as_matrix(
        self,
        *,
        basis_shape: tuple[int, ...] | None = None,
        max_dimension: int = 4096,
    ) -> np.ndarray:
        r"""Materialize the explicit matrix :math:`[A]` of this operator.

        Where :meth:`inverse` / :attr:`H` / composition are
        *endofunctors* (``Op → Op``), ``as_matrix`` is the **functor OUT
        of the operator category** (``Op → Mat``) — the
        serialization boundary. Column :math:`j` is the operator applied
        to the :math:`j`-th basis element:

        .. math::

            [A]_{:,j} \;=\; \operatorname{ravel}\bigl(A\,e_j\bigr),
            \qquad e_j = \operatorname{unravel}(\delta_j),

        with basis elements enumerated in **C-order** over
        ``basis_shape`` and outputs raveled the same way — so for a
        group-leading ``(ng, 1)`` carrier, column ``j`` is the response
        to a unit source in group ``j``, and ``[A] @ x.ravel() ==
        A.apply(x).ravel()`` exactly. The matrix is
        ``(prod(output shape), prod(basis_shape))`` — RECTANGULAR when
        the operator is not endomorphic; the output dimension emerges
        from :meth:`apply` itself, never from declared metadata.

        This default is the apply-to-basis pattern. Structured operators
        MAY override with a direct assembly (the future per-octant
        sparse-triangular streaming assembly noted at
        ``sweep_graph.py:66`` — DEFERRED with its 3-D consumer;
        :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
        overrides with one batched LU backsolve). Until a sparse
        consumer exists, the return is a DENSE :class:`numpy.ndarray` —
        keyed by the operator's structural override, with dense the only
        realization built (defer-until-consumer).

        Parameters
        ----------
        basis_shape : tuple[int, ...], optional
            The element shape :meth:`apply` consumes. Default: derived
            from :attr:`domain` (``domain.shape``); REQUIRED explicitly
            for operators carrying no space (bare/test-constructed
            operators on the legal-until-CS4 ``None`` path — the
            infinite-medium production operators thread the
            problem's pose since campaign 1 CS1 and derive).
        max_dimension : int, optional
            The size gate: refuse (``MatrixTooLarge``) when
            ``prod(basis_shape) > max_dimension``. Default ``4096`` —
            a 4096² float64 is 134 MB and 4096 applications, generous
            for every dense-by-construction consumer (0-D energy
            spectra, CP ``[P]``) and prohibitive for none of them;
            a meshed SN full-field operator is refused by design.
            Per-call configurable — a RESOURCE knob, not structure
            (see :class:`MatrixTooLarge`).

        Raises
        ------
        ValueError
            No ``basis_shape`` given and the operator carries no
            :attr:`domain` to derive one from.
        MatrixTooLarge
            The resolved basis dimension exceeds ``max_dimension``.

        Notes
        -----
        **Honest scope**: the default serves ndarray-carrier operators
        (the energy/scattering blocks, small compositions, quadrature
        maps). Typed-carrier operators (``FullField`` SN composites)
        are not constructible from ndarray basis columns — and sit far
        above any sane gate; they stay matrix-free.

        **Assembly delegation**: when
        this operator is :func:`assemblable`, the densification routes
        through the structural sparse emission
        (``assemble().as_matrix(...)`` — same gate contract, same
        dimension checks) instead of :math:`n` probing applications.
        The probing loop is RETAINED as :meth:`_as_matrix_by_probing` —
        the fallback for assembly-less operators AND the permanent
        fuller-view oracle the probed≡assembled equivalence gates pin
        (an assembly bug must never be able to hide inside its own
        densification).
        """
        shape = _resolve_basis_shape(self, basis_shape)
        n = int(np.prod(shape)) if shape else 1
        if n > max_dimension:
            raise MatrixTooLarge(
                f"as_matrix on {type(self).__name__}: basis dimension "
                f"{n} (= prod{shape}) exceeds max_dimension="
                f"{max_dimension}. Materializing would commit ~"
                f"{8 * n * n / 1e6:.0f} MB and {n} operator "
                f"applications. Raise max_dimension= if this size is "
                f"intended, or keep the operator matrix-free."
            )
        if assemblable(self):
            # The assembly delegation: densified structural assembly, with
            # the assembled column dimension checked against the resolved
            # basis shape (SparseAssembledOperator.as_matrix enforces it).
            return self.assemble().as_matrix(
                basis_shape=shape, max_dimension=max_dimension,
            )
        return self._as_matrix_by_probing(shape)

    def _as_matrix_by_probing(self, shape: tuple[int, ...]) -> np.ndarray:
        r"""The apply-to-basis probing loop — the RETAINED fuller-view pathway.

        Column :math:`j` = ``apply(e_j)`` raveled (the pre-assembly
        ``as_matrix`` body, byte-identical). Kept as its own named
        method — NOT inlined in :meth:`as_matrix` — for two consumers:
        the delegation fallback (assembly-less operators), and the
        probed≡assembled equivalence gates, which MUST be able to force
        this pathway on an *assemblable* operator (otherwise
        ``as_matrix ≡ assemble().to_dense()`` is assembly compared with
        itself — vacuous). Size/shape gating is the caller's job
        (:meth:`as_matrix` resolves and gates before delegating here).
        """
        n = int(np.prod(shape)) if shape else 1
        columns = []
        for j in range(n):
            e_flat = np.zeros(n)
            e_flat[j] = 1.0
            column = self.apply(cast(Domain, e_flat.reshape(shape)))
            columns.append(np.asarray(column).ravel())
        return np.column_stack(columns)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cls = type(self).__name__
        d = getattr(self, "domain", None)
        c = getattr(self, "codomain", None)
        d_name = repr(d.name) if d is not None else "'?'"
        c_name = repr(c.name) if c is not None else "'?'"
        # The two-axis surface, tokens present iff True.
        axes = "".join(
            f" {token}"
            for token, on in (
                ("invertible", getattr(self, "is_invertible", False)),
                ("adjointable", getattr(self, "is_adjointable", False)),
            )
            if on
        )
        return f"<{cls} domain={d_name} codomain={c_name}{axes}>"


# ───────────────────────────────────────────────────────────────────────
# Narrowing targets + checked bridges (#226 — Design C)
# ───────────────────────────────────────────────────────────────────────
#
# Each capability axis has THREE layers, and each layer carries the truth
# it alone can express:
#
#   1. the PREDICATE (``is_invertible`` / ``is_adjointable``) — runtime,
#      instance-accurate, value- and structure-aware; the polymorphic
#      override point every class defines its truth through;
#   2. the NARROWING TARGET below (``SupportsInverse``/``SupportsAdjoint``,
#      a Protocol EXTENDING :class:`LinearOperator`) — the static type a
#      guarded branch may treat the operand as;
#   3. the CHECKED BRIDGE (:func:`invertible` / :func:`adjointable`, a
#      PEP-647 ``TypeGuard``) — the ONE construct that converts the
#      runtime predicate into the static permission. You cannot obtain
#      the permission without executing the check (contrast the retired
#      ``cast(...)`` bridge, which asserted without checking).
#
# The Protocols are deliberately NOT ``runtime_checkable``: an
# ``isinstance`` check reads class-level method presence, which is
# class-uniform on composites (every ``OperatorSum`` defines
# ``apply_transpose`` even when a summand cannot transpose) and blind to
# value-dependent leaves (a zero-coefficient multiplier still has an
# ``inverse`` method). The bridge functions are the only sanctioned
# runtime→static conversion.


class SupportsInverse(LinearOperator[Domain, Codomain], Protocol):
    r"""Narrowing target: an operator whose :meth:`inverse` may be called.

    Extends :class:`LinearOperator`, so a branch narrowed by
    :func:`invertible` keeps the WHOLE algebra (``apply``, ``H``,
    composition dunders) alongside the licensed :meth:`inverse`. Never
    ``isinstance`` this (see the section comment); never annotate a
    parameter with it to *demand* invertibility — the static layer can
    certify only SPELLING (the method exists), never SOLVABILITY (the
    value-level predicate) — guard with :func:`invertible` instead.
    """

    def inverse(self) -> "LinearOperator[Codomain, Domain]": ...


class SupportsAdjoint(LinearOperator[Domain, Codomain], Protocol):
    r"""Narrowing target: an operator whose ``apply_transpose`` may be called.

    Extends :class:`LinearOperator`; the branch narrowed by
    :func:`adjointable` may call the raw Euclidean transpose verb
    ``apply_transpose`` (the realization the metric-aware
    :attr:`~LinearOperator.H` wrapper delegates to — two DIFFERENT
    objects: :math:`T^{\mathsf T}` vs :math:`G^{-1}T^{\mathsf T}G`).
    ``.H`` itself needs no narrowing — it lives on the base with an
    eager :class:`MissingAdjoint` gate.
    """

    def apply_transpose(self, x: Codomain, /) -> Domain: ...


class SupportsAssembly(LinearOperator[Domain, Codomain], Protocol):
    r"""Narrowing target: an operator whose :meth:`assemble` may be called.

    Extends :class:`LinearOperator`; the branch narrowed by
    :func:`assemblable` may call the structural sparse emission
    :meth:`assemble` — the ASSEMBLY-axis sibling of
    :class:`SupportsInverse` / :class:`SupportsAdjoint`, with the same
    discipline: never ``isinstance`` this (class-level method presence
    is class-uniform on composites — every ``OperatorSum`` defines
    ``assemble`` even when a summand cannot emit); never annotate a
    parameter with it to *demand* assemblability — guard with
    :func:`assemblable` instead.
    """

    def assemble(self) -> "SparseAssembledOperator": ...


def invertible(
    op: "LinearOperator[Domain, Codomain]",
) -> "TypeGuard[SupportsInverse[Domain, Codomain]]":
    r"""Checked bridge: narrow ``op`` to :class:`SupportsInverse` iff invertible.

    The runtime check and the static permission are ONE construct: a
    branch guarded by this function may call ``op.inverse()`` with no
    ``cast`` — and deleting the guard un-narrows the call, so CLI
    pyright REDs (the guard is type-load-bearing).

    Deliberately ``TypeGuard``, NOT ``TypeIs``: the predicate is
    VALUE-dependent (a zero-coefficient multiplier structurally *has*
    ``inverse()`` while reporting ``False``), so only the one-directional
    promise is honest — ``True`` licenses the call; ``False`` makes no
    static claim. A free function because PEP 647 narrowing applies only
    through a call expression and a method form narrows its first
    *explicit* argument, never ``self`` — no property spelling exists.
    Guard at ``LinearOperator``-typed sites only: ``TypeGuard`` REPLACES
    (does not intersect) the declared type, so guarding an
    already-concrete operand would widen it.
    """
    return op.is_invertible


def adjointable(
    op: "LinearOperator[Domain, Codomain]",
) -> "TypeGuard[SupportsAdjoint[Domain, Codomain]]":
    r"""Checked bridge: narrow ``op`` to :class:`SupportsAdjoint` iff adjointable.

    The adjoint-axis twin of :func:`invertible` — same one-directional
    ``TypeGuard`` semantics, same free-function necessity, same
    guard-at-``LinearOperator``-typed-sites discipline. Licenses the raw
    ``apply_transpose`` realization verb in composer law bodies
    (:math:`(A+B)^{\mathsf T} = A^{\mathsf T} + B^{\mathsf T}`) and
    gates the eager :attr:`~LinearOperator.H` construction.
    """
    return op.is_adjointable


def assemblable(
    op: "LinearOperator[Domain, Codomain]",
) -> "TypeGuard[SupportsAssembly[Domain, Codomain]]":
    r"""Checked bridge: narrow ``op`` to :class:`SupportsAssembly` iff assemblable.

    The assembly-axis member of the bridge family — same one-directional
    ``TypeGuard`` semantics, same free-function necessity, same
    guard-at-``LinearOperator``-typed-sites discipline as
    :func:`invertible` / :func:`adjointable`. Licenses the structural
    :meth:`~SupportsAssembly.assemble` emission in the composer
    homomorphism-law bodies (Sum → ``+``, Product → ``@``, Scaled →
    scalar ``*``) and the
    :meth:`~LinearOperator.as_matrix` densification delegation (R2).
    """
    return op.is_assemblable


# ───────────────────────────────────────────────────────────────────────
# Composition primitives
# ───────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Adjoint wrapper
# ---------------------------------------------------------------------------


class RieszLowerOperator(LinearOperator[Domain, Domain], Generic[Domain]):
    r"""The Riesz LOWERING leg :math:`\flat : V \to V^*` — apply the metric.

    ``♭ x = G x``: the isomorphism a Hilbert space's inner product induces
    between the space and its dual (the Riesz representation theorem,
    realized). Delegates to
    :meth:`~orpheus.numerics.space.FunctionSpace.apply_metric`, so the
    metric ARITHMETIC stays single-sourced in the space's resolved
    :class:`~orpheus.numerics.metric.HilbertMetric` (or its per-axis
    path) — this class contributes only the ARROW: ``domain`` is the
    primal ``V``, ``codomain`` is ``V.dual()``, both non-Optional by
    construction (the leg is born bound).

    Constructed by :attr:`~orpheus.numerics.space.FunctionSpace.riesz_lower`
    and by :class:`AdjointOperator` (whose codomain-side factor this is).

    ⛔ **PRIMAL spaces only.** A :class:`~orpheus.numerics.space.DualSpace`
    deliberately carries its PRIMAL's metric (L²-Riesz threading, P7 S2),
    so a lowering leg built on ``V*`` would apply ``G`` where the honest
    dual-side map applies ``G⁻¹`` — the measured ``G²`` trap
    (`[M]` 2026-08-30: ``lower_{V*}(lower_V(x)) = [0.25, 4, 16]`` for
    ``w = [0.5, 2, 4]``, where the double-Riesz involution must return
    ``x``). The constructor REFUSES a dual space, making the wrong
    composition unspellable; dual-side adjoints route through the
    dagger–dual commutation ``(A.dual()).H = (A.H).dual()`` instead
    (:meth:`_DualOperator.adjoint`).

    ``apply_transpose`` is ``apply``: every shipped metric realization is
    symmetric (a diagonal weight; a :class:`DenseMetric` admitted only
    through its symmetry guard), so ``♭ᵀ = ♭`` under the reflexive
    identification ``V** = V``.
    """

    def __init__(self, space: "FunctionSpace") -> None:
        from orpheus.numerics.space import DualSpace

        if isinstance(space, DualSpace):
            raise TypeError(
                f"RieszLowerOperator on a DualSpace ({space.name!r}) — the "
                f"Riesz legs live on the PRIMAL space only. A DualSpace "
                f"carries its primal's metric (L²-Riesz), so ♭ on V* would "
                f"apply G where the dual-side map needs G⁻¹ (the G² trap). "
                f"Compose the primal's riesz_lower/riesz_raise, or use the "
                f"dagger–dual commutation (A.dual()).H = (A.H).dual()."
            )
        self.space: Final = space

    @property
    def domain(self) -> "FunctionSpace":
        return self.space

    @property
    def codomain(self) -> "FunctionSpace":
        return self.space.dual()

    def apply(self, x: Domain) -> Domain:
        return self.space.apply_metric(x)

    def apply_transpose(self, x: Domain) -> Domain:
        # ♭ᵀ = ♭ (symmetric metric, V** = V) — see the class docstring.
        return self.space.apply_metric(x)

    @property
    def is_adjointable(self) -> bool:
        return True


class RieszRaiseOperator(LinearOperator[Domain, Domain], Generic[Domain]):
    r"""The Riesz RAISING leg :math:`\sharp : V^* \to V` — apply the
    pseudo-inverse metric.

    ``♯ f = G⁺ f``: the inverse Riesz map, Moore–Penrose everywhere by
    the metric family's doctrine (the reciprocal on the metric's range,
    zero on its kernel — the tangential ``|Ω·n| = 0`` trace slots).
    Delegates to
    :meth:`~orpheus.numerics.space.FunctionSpace.apply_inverse_metric`;
    this class contributes only the arrow: ``domain`` is ``V.dual()``,
    ``codomain`` the primal ``V``.

    The round trip is the honest law of the pseudo-inverse, NOT the
    identity: ``♯ ∘ ♭ = P_range(G)`` — the identity exactly when the
    metric is strictly positive, the tangential-zeroing projector on a
    singular trace block (`[M]` 2026-08-30: a legal 2-D
    ``Quadrature.product(4,4)`` mesh has 32/64 tangential trace slots and
    round-trip trace defect ``2.87``; every strictly-positive fixture
    reads ``≤ 4.4e-16``).

    Same primal-only refusal, same symmetric-transpose identity, and the
    same construction sites as :class:`RieszLowerOperator` (its exact
    mirror — the domain-side factor of :class:`AdjointOperator`).
    """

    def __init__(self, space: "FunctionSpace") -> None:
        from orpheus.numerics.space import DualSpace

        if isinstance(space, DualSpace):
            raise TypeError(
                f"RieszRaiseOperator on a DualSpace ({space.name!r}) — the "
                f"Riesz legs live on the PRIMAL space only (see "
                f"RieszLowerOperator: the G² trap). Compose the primal's "
                f"legs, or use (A.dual()).H = (A.H).dual()."
            )
        self.space: Final = space

    @property
    def domain(self) -> "FunctionSpace":
        return self.space.dual()

    @property
    def codomain(self) -> "FunctionSpace":
        return self.space

    def apply(self, x: Domain) -> Domain:
        return self.space.apply_inverse_metric(x)

    def apply_transpose(self, x: Domain) -> Domain:
        # ♯ᵀ = ♯ (symmetric pseudo-inverse of a symmetric metric).
        return self.space.apply_inverse_metric(x)

    @property
    def is_adjointable(self) -> bool:
        return True


class _DualOperator(LinearOperator[Codomain, Domain], Generic[Domain, Codomain]):
    r"""The dual arrow :math:`A^{\mathsf T} : W^* \to V^*` — the
    representation transpose with honest dual-space bookkeeping.

    The metric-free middle factor of the Hilbert adjoint (see
    :meth:`LinearOperator.dual`). ``apply`` IS ``inner.apply_transpose``;
    what this wrapper adds is the ARROW: domain/codomain are the DUALS of
    the inner's codomain/domain, so a composition chain can track bras vs
    kets through :class:`~orpheus.numerics.space.DualSpace` (its minted
    consumer — the §1 non-endomorphism made physical).

    Same explicit ``Generic[Domain, Codomain]`` pinning as
    :class:`AdjointOperator` (the PEP-696 parameter-order note there).

    Laws, as structure:

    * involution — ``A.dual().dual() is A`` (:meth:`dual` returns the
      inner: object identity);
    * transpose of the dual is the inner's action —
      :meth:`apply_transpose` delegates to ``inner.apply``;
    * dagger–dual commutation — ``(A.dual()).H = (A.H).dual()``
      (:meth:`adjoint` routes there), which is what makes a dual-side
      Hilbert adjoint expressible WITHOUT Riesz legs on dual spaces
      (those are refused — the G² trap).
    """

    def __init__(self, inner: "SupportsAdjoint[Domain, Codomain]") -> None:
        self.inner: Final = inner

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        c = getattr(self.inner, "codomain", None)
        return c.dual() if c is not None else None

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        d = getattr(self.inner, "domain", None)
        return d.dual() if d is not None else None

    def apply(self, y: Codomain) -> Domain:
        return self.inner.apply_transpose(y)

    def apply_transpose(self, x: Domain) -> Codomain:
        # (Aᵀ)ᵀ = A — the inner's forward action, no metric anywhere.
        return self.inner.apply(x)

    @property
    def is_adjointable(self) -> bool:
        return True

    def dual(self) -> "LinearOperator[Domain, Codomain]":
        # Involution as OBJECT IDENTITY: (Aᵀ)ᵀ = A.
        return self.inner  # type: ignore[return-value]

    def adjoint(self) -> "LinearOperator[Domain, Codomain]":
        r"""The dagger–dual commutation: ``(A.dual()).H = (A.H).dual()``.

        A dual arrow's Hilbert adjoint under the honest dual metrics
        (``G⁻¹`` on each dual space) is `[M]`-checkably the dual of the
        primal adjoint: :math:`(A^{\mathsf T})^{*} = G_W A G_V^{+} =
        ((A^{*})^{\mathsf T})`. Routing through the primal side keeps the
        Riesz legs primal-only (their constructors refuse dual spaces)
        rather than teaching :class:`~orpheus.numerics.space.DualSpace` a
        second metric.
        """
        return self.inner.adjoint().dual()  # type: ignore[return-value]


class AdjointOperator(LinearOperator[Codomain, Domain], Generic[Domain, Codomain]):
    r"""The Hilbert-adjoint ARROW :math:`A^{*} : W \to V` — first-class.

    Public since CS4c step 1 (the dagger-arrow promotion, R2 ruling —
    until then the private ``_AdjointOperator``): the realization of the
    dagger functor, constructed by :meth:`LinearOperator.adjoint` /
    ``A.H`` (the canonical door — direct construction is legal for an
    :func:`adjointable`-narrowed inner and repeats the same eager gates).

    Presents the SWAPPED carriers: an inner ``A : Domain → Codomain``
    becomes ``A^* : Codomain → Domain``. The explicit
    ``Generic[Domain, Codomain]`` pins the class's type-parameter order to
    ``[Domain, Codomain]`` (the non-defaulted ``Domain`` first) even though
    the base is the swapped ``LinearOperator[Codomain, Domain]`` — without
    it ``Codomain`` (which carries ``default=Domain``) would land before
    the non-defaulted ``Domain`` in the inferred parameter list, which
    PEP-696 forbids. This is the ONLY composer with ``[Codomain, Domain]``
    order.

    Constructed by :meth:`LinearOperator.adjoint` (and its alias
    ``A.H``). Domain/codomain are swapped relative to the inner operator;
    :meth:`apply` performs the weight-aware adjoint action.

    Construction is gated EAGERLY by :meth:`LinearOperator.adjoint`:
    only an :func:`adjointable`-narrowed operator
    reaches this constructor, so ``inner`` is statically a
    :class:`SupportsAdjoint` — there is no lazy capability gate left to
    fail at call time.

    **The arrow is a composition of three first-class factors, built at
    construction** (CS4c R2 ruling, 2026-08-30): the domain/codomain
    exchange is realized in code — for ``A : V → W``,

    .. math::

       A^{*} \;=\; \underbrace{\sharp_V}_{V^* \to V}
       \circ \underbrace{A^{\mathsf T}}_{W^* \to V^*}
       \circ \underbrace{\flat_W}_{W \to W^*}

    i.e. ``inner.domain.riesz_raise ∘ inner.dual() ∘
    inner.codomain.riesz_lower`` (the formula reserved at
    ``metric.py``'s CS4c-compatibility note). The metric arithmetic
    lives in the legs (single-sourced through the space's
    :class:`~orpheus.numerics.metric.HilbertMetric`); this class holds
    the composition plus what only the named arrow can carry: the
    eager refusals, the #280 swap law (:meth:`inverse` — an object
    identity), the role passthrough, and the dagger laws as structure:

    * involution — ``A.H.H is A`` (:meth:`adjoint` returns the inner);
    * transpose of the adjoint — a THEOREM of the legs,
      :math:`(A^{*})^{\mathsf T} = \flat_W \circ A \circ \sharp_V`
      (metrics symmetric by admission), which closed #375's dead-end:
      ``A.H`` is adjointable, and ``A.H.H`` is reachable — and IS ``A``.

    An UNBOUND end (a ``None`` space on a metric-free-exempt inner)
    contributes an :class:`IdentityOperator` leg — the Euclidean metric
    as the neutral element, bit-identical to the pre-CS4c skip branch.
    """

    def __init__(self, inner: "SupportsAdjoint[Domain, Codomain]") -> None:
        # The S4-amendment's unbound-.H refusal (user ruling, 2026-08-22):
        # the Hilbert adjoint is defined BY the two inner products, so an
        # operator that has not declared its spaces has no Hilbert adjoint
        # to take — and the pre-amendment behaviour (apply the Euclidean
        # transpose and skip the metric sandwiches) was the catalogued R2
        # hazard: "a bare Euclidean transpose wearing the Hilbert
        # adjoint's name". Eagerly, here at construction, in this class's
        # own broken-stub-refusing style. The metric-free families never
        # reach this constructor: a PointwiseOperator's adjoint() returns
        # itself, ZeroMorphism's the swapped zero map.
        if (
            inner.domain is None or inner.codomain is None
        ) and not inner.is_metric_free_adjoint:
            raise MissingAdjoint(
                f"{type(inner).__name__} is UNBOUND (domain/codomain "
                f"None) — the Hilbert adjoint needs the two spaces' "
                f"metrics; declare both, or use apply_transpose for the "
                f"bare representation transpose. (A space-polymorphic "
                f"multiplier belongs in PointwiseOperator, whose adjoint "
                f"is metric-free.)"
            )
        self.inner = inner
        # The G-adjoint transposes the 2×2 block matrix (A_bs ↔ A_sb^T),
        # which preserves WHICH blocks are touched — so the role is the
        # inner operator's role: ``L.H`` is FULL, ``B.H`` is BOUNDARY,
        # ``C.H`` is BULK.
        self.block_role = getattr(inner, "block_role", None)
        # The G-adjoint also preserves which SYSTEMS are touched: A_AB.H
        # (bulk→ray) is still COUPLED, A_BB.H still System B.
        self.system_role = getattr(inner, "system_role", None)
        # The three factors, built HERE — the domain/codomain exchange
        # realized at construction (R2 ruling): the codomain-side ♭ and
        # the domain-side ♯ are the swapped ends' Riesz legs; the dual
        # arrow is the metric-free middle. A None end (metric-free
        # exemption above) gets the Euclidean neutral element — the same
        # arithmetic the pre-leg skip branch performed, bit-identically.
        inner_domain = getattr(inner, "domain", None)
        inner_codomain = getattr(inner, "codomain", None)
        self._lower: "LinearOperator" = (
            RieszLowerOperator(inner_codomain)
            if inner_codomain is not None
            else IdentityOperator()
        )
        self._dual: "_DualOperator[Domain, Codomain]" = _DualOperator(inner)
        self._raise: "LinearOperator" = (
            RieszRaiseOperator(inner_domain)
            if inner_domain is not None
            else IdentityOperator()
        )

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        # Adjoint of A: V → W is A.H: W → V — domain swaps with inner.codomain.
        return getattr(self.inner, "codomain", None)

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        return getattr(self.inner, "domain", None)

    def apply(self, y: Codomain) -> Domain:
        # The Hilbert-adjoint action, as the leg composition built at
        # construction:
        #   (A^* y)_V = ♯_V(Aᵀ(♭_W y)) = G_V⁺ ⊙ apply_transpose(G_W ⊙ y)
        # Same call order, same delegation targets as the pre-leg inline
        # spelling (each leg delegates to the space's apply_metric /
        # apply_inverse_metric, so the SAME composition serves a
        # flat-ndarray metric AND a composite bulk ⊕ trace metric with
        # its pseudo-inverse on the singular partial-current trace) —
        # bit-identical by construction, gated by G-A1. The legs are the
        # individually-mutable seams the ledger's per-leg battery reads.
        return self._raise.apply(self._dual.apply(self._lower.apply(y)))

    def apply_transpose(self, x: Domain) -> Codomain:
        r"""The representation transpose of the adjoint — a THEOREM of
        the legs (#375's four-line composition, landed).

        :math:`(A^{*})^{\mathsf T} = (G_V^{+} A^{\mathsf T}
        G_W)^{\mathsf T} = G_W\, A\, G_V^{+} = \flat_W \circ A \circ
        \sharp_V` — the metrics transpose to themselves because every
        shipped realization is symmetric (diagonal weights; a
        :class:`~orpheus.numerics.metric.DenseMetric` is admitted only
        through its symmetry guard). Until CS4c this raised a stub with
        zero witnesses; the capability replaces it, and
        :attr:`is_adjointable` advertises it honestly.
        """
        return self._lower.apply(self.inner.apply(self._raise.apply(x)))

    @property
    def is_adjointable(self) -> bool:
        # True by the theorem above: apply_transpose needs only the legs
        # (always built) and the inner's forward action (always present).
        return True

    def adjoint(self) -> "LinearOperator[Domain, Codomain]":
        r"""The dagger involution as an OBJECT IDENTITY: ``A.H.H is A``.

        :math:`(A^{*})^{*} = A` — no double wrapper, no arithmetic: the
        inner IS the answer. (`[M]` pre-CS4c, ``A.H.H`` was unreachable —
        #375's headline: ``is_adjointable`` read ``False`` and the
        transpose raised.)
        """
        return self.inner  # type: ignore[return-value]

    def dual(self) -> "LinearOperator[Domain, Codomain]":
        r"""``(A.H).dual() = (A.dual()).H`` — the dagger–dual commutation,
        routed as the adjoint of the dual (see
        :meth:`_DualOperator.adjoint`, the other direction of the same
        square). Spelled here to keep the two spellings one object family
        rather than a generic :class:`_DualOperator` over an adjoint."""
        return _DualOperator(self)  # type: ignore[return-value]

    @property
    def is_invertible(self) -> bool:
        r"""Whether the adjoint's inverse operator exists — the swap law (#280).

        The inverse of the adjoint IS the adjoint of the inverse:
        :math:`(A^{*})^{-1} = (A^{-1})^{*}`. Honest
        iff the inner :math:`A` is invertible AND its inverse operator is
        adjointable (so ``.H`` on that inverse is well-posed) — spelled
        generally over the inner, no leaf specifics. :func:`invertible`
        narrows ``self.inner`` to :class:`SupportsInverse` for the RHS call;
        the ``and`` short-circuits so ``inner.inverse()`` is never built for a
        non-invertible inner.
        """
        return invertible(self.inner) and adjointable(self.inner.inverse())

    def inverse(self) -> "LinearOperator[Domain, Codomain]":
        r"""The inverse of the adjoint = the adjoint of the inverse (#280).

        :math:`(A^{*})^{-1} = (A^{-1})^{*}` — the operator-algebra swap law,
        an OBJECT IDENTITY here (not a computed numerical equivalence): this
        wrapper IS ``A.H``, so its inverse routes to ``A.inverse().H``. Gated
        by :attr:`is_invertible` (``A`` invertible and ``A.inverse()``
        adjointable). The metric adjoint-solve
        :math:`A^{-1\,*} b = G_V^{+}\,\mathrm{apply\_transpose}(G_W\,b)` then
        falls out of :meth:`apply` (which already routes
        ``inner.apply_transpose``) FOR FREE — no ``AdjointOperator.solve`` /
        no metric code enters the sweep.
        """
        if not invertible(self.inner):
            raise NotInvertible(
                f"AdjointOperator.inverse(): the inner "
                f"{type(self.inner).__name__} is not invertible, so the "
                f"adjoint-inverse swap law (A.H).inverse() = (A.inverse()).H "
                f"does not apply (is_invertible is False)."
            )
        inner_inverse = self.inner.inverse()
        if not adjointable(inner_inverse):
            # The swap law needs the inverse to be adjointable so ``.H`` exists
            # — matches :attr:`is_invertible`'s second clause. Raise
            # NotInvertible (NOT MissingAdjoint from the ``.H`` below): the
            # adjoint-INVERSE is what is absent here.
            raise NotInvertible(
                f"AdjointOperator.inverse(): the inner's inverse "
                f"{type(inner_inverse).__name__} is not adjointable, so "
                f"(A.inverse()).H does not exist — the swap law needs an "
                f"adjointable inverse (e.g. an SN SweepOperator, #280 2.5c). "
                f"is_invertible is False."
            )
        return inner_inverse.H


class OperatorSum(
    LinearOperator[Domain, Codomain],
    Generic[Domain, Codomain, SummandA, SummandB],
):
    r"""Sum of two linear operators: :math:`(A + B)\,x = A\,x + B\,x`.

    Generic over its SUMMAND types: a named composition subclass
    pins them — ``StreamingCollisionOperator = OperatorSum["FullField",
    "FullField", StreamingOperator, MultiplicationOperator]`` — so its
    leg accessors are typed by construction, no casts. The PEP-696
    defaults (``LinearOperator[Domain, Codomain]``) keep every
    ``OperatorSum[D, C]`` / bare spelling valid; the legs are covariant,
    read-only properties so a pinned composition upcasts to the
    defaulted spelling (the ``__add__`` contract).

    Structural closure laws (realized in the method bodies; the
    predicates are the matching advertisements):

    * ``apply`` requires **both** operands to act — guarded eagerly at
      construction (``TypeError``), never at the first call.
    * Invertibility does NOT propagate operand-wise: there is no
      algorithm for :math:`(A + B)^{-1}` from :math:`A^{-1}` and
      :math:`B^{-1}` alone — Sherman–Morrison–Woodbury applies only
      under low-rank structure (which the boundary block B has — rank
      ≤ N/2 per face, Issue #300 — and the bulk C, L do not).  What a
      sum DOES have, when its LEADING
      (left-spine head) term is invertible, is a
      preconditioned-SPLITTING inverse: :meth:`inverse` returns a
      :class:`~orpheus.numerics.green_operator.GreenOperator` iterating
      :math:`x_{n+1} = A^{-1}(q - B\,x_n)`.  A generic sum carries no
      ``solve`` verb — solving with it IS applying that inverse OBJECT
      (the sweep-invertible ``(L+C)`` subclass overrides with its own
      direct-sweep ``solve``).  See :attr:`is_invertible` for the
      canonical-ordering contract.
    * ``apply_transpose`` requires **both** operands to transpose
      (:math:`(A + B)^T = A^T + B^T`) — guarded in the verb body with
      :class:`MissingAdjoint`; :attr:`is_adjointable` is the recursion.

    Raises
    ------
    TypeError
        If either operand has no callable ``apply`` at construction
        time. Catch the failure here, not at the first ``apply`` call,
        so downstream Krylov consumers don't see a stub failure
        mid-iteration.
    """

    def __init__(self, a: SummandA, b: SummandB) -> None:
        if not callable(getattr(a, "apply", None)):
            raise TypeError(
                f"OperatorSum requires apply on both operands; left "
                f"operand {type(a).__name__} lacks 'apply'."
            )
        if not callable(getattr(b, "apply", None)):
            raise TypeError(
                f"OperatorSum requires apply on both operands; right "
                f"operand {type(b).__name__} lacks 'apply'."
            )
        # Domain/codomain agreement, eager (skipped per-operand when one
        # lacks function-space metadata — backward-compatible with operators
        # that pre-date Issue 9.6). The law is :func:`_agreed_space`, shared
        # with the other commutative composite (the tensor product): a sum
        # commutes, so its spaces are what its summands AGREE on, never a
        # function of which one was written first.
        _agreed_space((a, b), "domain", "OperatorSum")
        _agreed_space((a, b), "codomain", "OperatorSum")
        self._a: Final = a
        self._b: Final = b
        # Block role DERIVED from the operands: the sum touches the union
        # of the blocks its summands touch, so ``(L+C)`` and the whole
        # ``(L+C-S-F-B)`` loss carry FULL by construction (no hand-stamp).
        self.block_role = _join_block_roles(
            getattr(a, "block_role", None), getattr(b, "block_role", None),
        )
        # System role DERIVED the same way — the sum touches the union of the
        # systems its summands touch (``A ⊔ B = COUPLED``); the two-system
        # analogue of the block-role join.
        self.system_role = _join_system_roles(
            getattr(a, "system_role", None), getattr(b, "system_role", None),
        )

    @property
    def a(self) -> SummandA:
        """The left summand (read-only — covariant leg typing)."""
        return self._a

    @property
    def b(self) -> SummandB:
        """The right summand (read-only — covariant leg typing)."""
        return self._b

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        # The summands agreed at construction, so "the agreed one" and "the
        # first that speaks" coincide — but only the former is the LAW, and
        # spelling it twice is how the two drift.
        return _agreed_space((self.a, self.b), "domain", "OperatorSum")

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        return _agreed_space((self.a, self.b), "codomain", "OperatorSum")

    def apply(self, x: Domain, /) -> Codomain:
        return self.a.apply(x) + self.b.apply(x)

    def apply_transpose(self, x: Codomain, /) -> Domain:
        # (A+B)^T = A^T + B^T — the guard-narrow licenses the operand
        # calls (Design C: the runtime check IS the static permission).
        if not adjointable(self.a) or not adjointable(self.b):
            raise MissingAdjoint(
                f"OperatorSum.apply_transpose requires both summands to "
                f"transpose ((A+B)^T = A^T + B^T); got "
                f"{type(self.a).__name__} / {type(self.b).__name__} with "
                f"is_adjointable {self.a.is_adjointable} / "
                f"{self.b.is_adjointable}."
            )
        return self.a.apply_transpose(x) + self.b.apply_transpose(x)

    @property
    def is_adjointable(self) -> bool:
        # (A+B)^T = A^T + B^T (the law in :meth:`apply_transpose`) — the
        # sum is adjointable iff BOTH summands are.
        return self.a.is_adjointable and self.b.is_adjointable

    @property
    def is_metric_free_adjoint(self) -> bool:
        # A sum of metric-free operators is metric-free (derived).
        return (
            self.a.is_metric_free_adjoint and self.b.is_metric_free_adjoint
        )

    @property
    def is_invertible(self) -> bool:
        r"""``True`` iff the LEADING (left-spine head) term is invertible.

        There is no operand-wise law for a sum's inverse — but a sum
        whose leading term :math:`A` is invertible CAN produce its
        inverse OPERATOR: the preconditioned-splitting
        :class:`~orpheus.numerics.green_operator.GreenOperator`
        (:math:`x_{n+1} = A^{-1}(q - B\,x_n)`).
        The recursion ``self.a.is_invertible`` designates the left-spine
        head as the splitting's preconditioner — the CANONICAL-ORDERING
        contract: spell the invertible operator FIRST (``A - S``,
        mirroring the ``L + C`` fusion rule of
        :meth:`~orpheus.sn.operators.streaming.StreamingOperator.__add__`,
        #261).  Whether the splitting CONVERGES is a spectral
        (value-level) property no construction-time predicate can read —
        a divergent split raises
        :class:`~orpheus.numerics.green_operator.ConvergenceFailure`
        loudly at apply time, never a silent wrong answer.  (The
        sweep-invertible
        :class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`
        subclass shadows this by MRO with its own ``True`` +
        direct-sweep :meth:`inverse` — the type-as-structure dispatch.)
        """
        return self.a.is_invertible

    def inverse(self) -> "LinearOperator[Codomain, Domain]":
        r"""Return the preconditioned-splitting inverse — a
        :class:`~orpheus.numerics.green_operator.GreenOperator`.

        The annotation is the factory's honest STATIC face — "an inverse
        operator on the swapped spaces" — because subclass overrides
        return their own structure's inverse (the sweep-invertible
        composite returns a ``SweepOperator``; type-as-structure) and the
        family members are siblings, not a hierarchy.

        Late import: ``green_operator`` is a LEAF module wrapping the
        iteration drivers, which import THIS module — the same one-way
        late-import pattern as
        :meth:`~orpheus.sn.operators.streaming.StreamingCollisionOperator.inverse`
        → ``SweepOperator``.
        """
        from orpheus.numerics.green_operator import GreenOperator

        return GreenOperator(self)

    # NO ``solve`` on a generic sum: its inverse action is DRIVER-realized
    # (the GreenOperator), not a substrate verb — solving is
    # ``.inverse().apply(b)``. The sweep-invertible ``(L+C)`` subclass
    # overrides with its own direct sweep ``solve`` (streaming.py).

    @property
    def is_assemblable(self) -> bool:
        # [A+B] = [A] + [B] (the law in :meth:`assemble`) — the sum
        # assembles iff BOTH summands emit.
        return self.a.is_assemblable and self.b.is_assemblable

    def assemble(self) -> "SparseAssembledOperator":
        r"""Return :math:`[A+B] = [A] + [B]` — the additive homomorphism law.

        The assembly functor is additive-monoidal (the ``as_matrix``
        docstring's ``Op → Mat``, sparse carrier): a sum's structural
        emission is the SPARSE SUM of its summands' emissions —
        realized by the carrier's own CSR addition, never a re-walk of
        the stencils. The guard-narrow licenses the operand calls
        (Design C) and raises the assembly-axis refusal eagerly.
        """
        if not assemblable(self.a) or not assemblable(self.b):
            raise MissingAssembly(
                f"OperatorSum.assemble requires both summands to emit "
                f"([A+B] = [A] + [B]); got {type(self.a).__name__} / "
                f"{type(self.b).__name__} with is_assemblable "
                f"{self.a.is_assemblable} / {self.b.is_assemblable}."
            )
        from orpheus.numerics.assembled_operator import SparseAssembledOperator

        return SparseAssembledOperator(
            self.a.assemble().matrix + self.b.assemble().matrix,
            domain=self.domain,
            codomain=self.codomain,
        )


class OperatorProduct(
    LinearOperator[Domain, Codomain],
    Generic[Domain, Codomain, FactorA, FactorB],
):
    r"""Composition of two linear operators: :math:`(A\,B)\,x = A(B\,x)`.

    Generic over its FACTOR types (as :class:`OperatorSum` over
    its summands): a named composition pins them — ``WindowedSweep =
    OperatorProduct["FullField", "TimedFullField", BulkAnalysisOperator,
    SweepOperator]`` — so its factor accessors are typed by
    construction. The PEP-696 defaults keep ``OperatorProduct[D, C]``
    valid; the legs are covariant read-only properties. The
    intermediate-space coupling (``A.domain == B.codomain``) is the
    RUNTIME guard below — the leg parameters do not re-encode it.

    Structural closure laws (method bodies; predicates advertise):

    * ``apply`` requires **both** operands (function composition) —
      guarded eagerly at construction (``TypeError``).
    * Invertibility propagates iff **both** factors are invertible,
      with the order reversed: :math:`(A\,B)^{-1} = B^{-1}\,A^{-1}`.
      The product IS a wrap-delegate conformer — :meth:`solve` is its
      native realization verb, re-routed through the factors' CANONICAL
      surface (``.inverse().apply``) so factor kinds whose own
      ``solve`` retired (algebra-closed permutations/scalings, Green-
      invertible sums) compose without one.
    * ``apply_transpose`` requires **both**, order reversed
      (:math:`(A\,B)^T = B^T\,A^T`) — :class:`MissingAdjoint` in the
      verb body; :attr:`is_adjointable` is the recursion.

    Raises
    ------
    TypeError
        If either operand has no callable ``apply`` at construction.
    """

    def __init__(self, a: FactorA, b: FactorB) -> None:
        # ``A @ B``: ``B`` maps the input ``V`` to the intermediate,
        # ``A`` maps the intermediate to the output ``W`` — so the
        # product is honestly ``V → W``, the intermediate guarded below.
        if not callable(getattr(a, "apply", None)):
            raise TypeError(
                f"OperatorProduct requires apply on both operands; "
                f"left operand {type(a).__name__} lacks 'apply'."
            )
        if not callable(getattr(b, "apply", None)):
            raise TypeError(
                f"OperatorProduct requires apply on both operands; "
                f"right operand {type(b).__name__} lacks 'apply'."
            )
        # Domain/codomain compatibility check for ``A @ B``: A.domain
        # must equal B.codomain. Skipped when either is None.
        a_dom = getattr(a, "domain", None)
        b_cod = getattr(b, "codomain", None)
        if (
            a_dom is not None and b_cod is not None and a_dom != b_cod
        ):
            raise IncompatibleOperatorComposition(
                f"OperatorProduct A @ B requires A.domain == B.codomain; "
                f"got A.domain={a_dom!r}, B.codomain={b_cod!r}."
            )
        self._a: Final = a
        self._b: Final = b

    @property
    def a(self) -> FactorA:
        """The left factor ``A`` of ``A @ B`` (read-only — covariant leg)."""
        return self._a

    @property
    def b(self) -> FactorB:
        """The right factor ``B`` of ``A @ B`` (read-only — covariant leg)."""
        return self._b

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        # A @ B: input space is B.domain — with the pointwise-conforming
        # arm (S4-amendment): a PointwiseOperator B acts AT its operand's
        # space, so the product's input space is whatever A consumes.
        d = getattr(self.b, "domain", None)
        if d is None and isinstance(self.b, PointwiseOperator):
            return getattr(self.a, "domain", None)
        return d

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        # A @ B: output space is A.codomain — with the pointwise-conforming
        # arm (S4-amendment): a PointwiseOperator A emits at its operand's
        # space, which is B's output space.
        c = getattr(self.a, "codomain", None)
        if c is None and isinstance(self.a, PointwiseOperator):
            return getattr(self.b, "codomain", None)
        return c

    def apply(self, x: Domain, /) -> Codomain:
        return self.a.apply(self.b.apply(x))

    def solve(self, b_vec: Codomain) -> Domain:
        r"""Solve :math:`(AB)\,x = b` — :math:`B^{-1}(A^{-1}\,b)`, factor-wise.

        The product's native realization verb (the wrap-delegate family
        wraps it: :meth:`inverse` returns ``InverseOperator(self)`` whose
        ``apply`` delegates here). The recursion goes
        through each factor's CANONICAL surface — ``.inverse().apply`` —
        not a factor ``solve``: bit-identical for every factor kind (the
        inverse objects delegate to the same realizations) and total
        over the kinds whose own ``solve`` retired (a permutation's
        inverse is a first-class forward; a Green-invertible sum's is
        the GreenOperator). The guard-narrow licenses the calls and
        raises the value-dependent refusal eagerly.
        """
        if not invertible(self.a) or not invertible(self.b):
            raise NotInvertible(
                f"OperatorProduct.solve requires both factors to be "
                f"invertible ((AB)^{{-1}} = B^{{-1}}A^{{-1}}); got "
                f"{type(self.a).__name__} / {type(self.b).__name__} with "
                f"is_invertible {self.a.is_invertible} / "
                f"{self.b.is_invertible}."
            )
        return self.b.inverse().apply(self.a.inverse().apply(b_vec))

    def apply_transpose(self, x: Codomain, /) -> Domain:
        # (AB)^T = B^T A^T — the guard-narrow licenses the operand calls.
        if not adjointable(self.a) or not adjointable(self.b):
            raise MissingAdjoint(
                f"OperatorProduct.apply_transpose requires both factors "
                f"to transpose ((AB)^T = B^T A^T); got "
                f"{type(self.a).__name__} / {type(self.b).__name__} with "
                f"is_adjointable {self.a.is_adjointable} / "
                f"{self.b.is_adjointable}."
            )
        return self.b.apply_transpose(self.a.apply_transpose(x))

    @property
    def is_invertible(self) -> bool:
        # (AB)^{-1} = B^{-1} A^{-1} (the law in :meth:`solve`) — the product
        # is invertible iff BOTH factors are.
        return self.a.is_invertible and self.b.is_invertible

    def inverse(self) -> "InverseOperator":
        r"""Return :math:`(AB)^{-1}` — the generic family member wrapping this product.

        The functoriality law :math:`(AB)^{-1} = B^{-1}A^{-1}` holds
        BEHAVIORALLY through the wrapper: ``inverse().apply(q)`` delegates
        to this product's own :meth:`solve` ``= b.solve(a.solve(q))``.
        What the family wrapper adds is the CONTRACT (#285): a raw
        ``OperatorProduct`` of inverses carries no ``initial_guess``
        keyword, so a driver seeding it raised ``TypeError`` at iteration
        time; :class:`InverseOperator` carries the family's canonical
        seeded ``apply`` (accept-and-ignore — the solve path never
        threaded seeds either, so behavior is unchanged) and every
        ``.inverse()`` in the system now returns a seeded-apply
        conformer. The involution is object identity —
        ``(A@B).inverse().inverse() is (A@B)`` via the mixin. The factors
        stay reachable as ``.inner.a`` / ``.inner.b``.

        (Contrast the ALGEBRA-CLOSED inverses — a
        :class:`PermutationOperator`'s inverse IS a permutation, an
        :class:`IdentityOperator` is self-inverse, a
        :class:`ScaledOperator`'s is a scaled inverse: those inverses
        are first-class FORWARD operators in their own closed structure,
        the other kind of inverse, and stay unwrapped.)
        """
        if not self.is_invertible:
            raise NotInvertible(
                "OperatorProduct.inverse requires both factors to be "
                "invertible ((AB)^{-1} = B^{-1}A^{-1})."
            )
        return InverseOperator(self)

    @property
    def is_adjointable(self) -> bool:
        # (AB)^T = B^T A^T (the law in :meth:`apply_transpose`) — adjointable
        # iff BOTH factors are.
        return self.a.is_adjointable and self.b.is_adjointable

    @property
    def is_metric_free_adjoint(self) -> bool:
        # A product of metric-free operators is metric-free (derived).
        return (
            self.a.is_metric_free_adjoint and self.b.is_metric_free_adjoint
        )

    @property
    def is_assemblable(self) -> bool:
        # [AB] = [A] @ [B] (the law in :meth:`assemble`) — the product
        # assembles iff BOTH factors emit.
        return self.a.is_assemblable and self.b.is_assemblable

    def assemble(self) -> "SparseAssembledOperator":
        r"""Return :math:`[AB] = [A]\,[B]` — the multiplicative homomorphism law.

        The composition's structural emission is the SPARSE PRODUCT of
        its factors' emissions (dimension compatibility enforced by the
        carrier's own matmul). Same eager guard-narrow discipline as
        :meth:`OperatorSum.assemble`.
        """
        if not assemblable(self.a) or not assemblable(self.b):
            raise MissingAssembly(
                f"OperatorProduct.assemble requires both factors to emit "
                f"([AB] = [A][B]); got {type(self.a).__name__} / "
                f"{type(self.b).__name__} with is_assemblable "
                f"{self.a.is_assemblable} / {self.b.is_assemblable}."
            )
        from orpheus.numerics.assembled_operator import SparseAssembledOperator

        return SparseAssembledOperator(
            self.a.assemble().matrix @ self.b.assemble().matrix,
            domain=self.domain,
            codomain=self.codomain,
        )


class ScaledOperator(
    LinearOperator[Domain, Codomain],
    Generic[Domain, Codomain, ScaledOperand],
):
    r"""Scalar multiple of a linear operator: :math:`(\alpha L)\,x = \alpha\,(L\,x)`.

    Generic over its OPERAND type (as the other composition
    wrappers over their legs): ``ScaledOperator["FullField",
    "FullField", SNMaskedBoundaryOperator]`` reads ``.op`` as the masked
    boundary leaf — the ``-1·B_lower`` leg of the G-S splitting needs no
    cast. The PEP-696 default keeps ``ScaledOperator[D, C]`` valid; the
    operand is a covariant read-only property.

    Scaling (:math:`\alpha \neq 0`, caught at composition time) passes
    both structural axes through unchanged: the operand's
    invertibility/adjointability ARE the scaled operator's, and the
    algebra is closed — :meth:`inverse` is a
    :class:`ScaledOperator` (:math:`(\alpha L)^{-1} = (1/\alpha)L^{-1}`)
    and the transpose scales (:math:`(\alpha L)^T = \alpha L^T`). No
    ``solve`` verb: an algebra-closed inverse is a first-class forward,
    so solving is ``.inverse().apply(b)``.
    """

    def __init__(self, scalar: float, op: ScaledOperand) -> None:
        if not callable(getattr(op, "apply", None)):
            raise TypeError(
                f"ScaledOperator requires apply on its operand; "
                f"{type(op).__name__} lacks 'apply'."
            )
        if scalar == 0.0:
            # Zero scaling is a degenerate case: the result behaves as
            # a ZeroOperator (singular, structurally), not as the
            # underlying operator. The user should construct
            # ZeroOperator explicitly to make the intent clear.
            raise ValueError(
                "ScaledOperator with zero scalar is degenerate; "
                "use ZeroOperator explicitly."
            )
        self.scalar = float(scalar)
        self._op: Final = op
        # Scaling preserves which blocks the action touches.
        self.block_role = getattr(op, "block_role", None)
        # Scaling preserves which systems the action touches, too.
        self.system_role = getattr(op, "system_role", None)

    @property
    def op(self) -> ScaledOperand:
        """The scaled operand ``L`` (read-only — covariant leg typing)."""
        return self._op

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        return getattr(self.op, "domain", None)

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        return getattr(self.op, "codomain", None)

    def apply(self, x: Domain, /, *extra, **kwextra) -> Codomain:
        return self.scalar * self.op.apply(x, *extra, **kwextra)

    # NO ``solve``: the inverse is ALGEBRA-CLOSED —
    # :meth:`inverse` returns the first-class forward
    # ``ScaledOperator(1/α, op.inverse())`` — so there is no wrapped
    # realization verb to keep; solving is ``.inverse().apply(b)``.

    def apply_transpose(self, x: Codomain, /, *extra, **kwextra) -> Domain:
        # (αL)^T = α L^T — the guard-narrow licenses the operand call.
        if not adjointable(self.op):
            raise MissingAdjoint(
                f"ScaledOperator.apply_transpose requires an adjointable "
                f"operand ((αL)^T = αL^T); {type(self.op).__name__}."
                f"is_adjointable is False."
            )
        return self.scalar * self.op.apply_transpose(x, *extra, **kwextra)

    @property
    def is_invertible(self) -> bool:
        # (αL)^{-1} = (1/α) L^{-1} — α ≠ 0 is enforced at construction, so
        # the scaled operator is invertible iff the operand is.
        return self.op.is_invertible

    def inverse(self) -> "ScaledOperator[Codomain, Domain]":
        r"""Return :math:`(\alpha L)^{-1} = (1/\alpha)\,L^{-1}`.

        The natural structural inverse: a scaled operator's inverse IS a
        scaled operator — on the SWAPPED carriers (an inverse maps the
        forward's codomain back to its domain), so the return type is
        ``ScaledOperator[Codomain, Domain]``. ``1/α`` is exact whenever
        ``α`` is a power of two
        (the dominant −1.0 case); the action is bit-identical to
        :meth:`solve` given the operand's own ``inverse().apply ≡ solve``
        identity (both spell ``(1/α) * op_solve(b)``).
        """
        if not invertible(self.op):
            raise NotInvertible(
                "ScaledOperator.inverse requires an invertible operand "
                "((αL)^{-1} = (1/α)L^{-1})."
            )
        return ScaledOperator(1.0 / self.scalar, self.op.inverse())

    @property
    def is_adjointable(self) -> bool:
        # (αL)^T = α L^T — scaling preserves adjointability.
        return self.op.is_adjointable

    @property
    def is_metric_free_adjoint(self) -> bool:
        # A real scalar is a multiplier — scaling preserves metric-freeness.
        return self.op.is_metric_free_adjoint

    @property
    def is_assemblable(self) -> bool:
        # [αL] = α[L] (the law in :meth:`assemble`) — scaling preserves
        # assemblability.
        return self.op.is_assemblable

    def assemble(self) -> "SparseAssembledOperator":
        r"""Return :math:`[\alpha L] = \alpha\,[L]` — the scalar homomorphism law.

        The scaled emission is the carrier's own scalar multiply of the
        operand's emission. Same eager guard-narrow discipline as the
        other composer laws.
        """
        if not assemblable(self.op):
            raise MissingAssembly(
                f"ScaledOperator.assemble requires an assemblable operand "
                f"([αL] = α[L]); {type(self.op).__name__}.is_assemblable "
                f"is False."
            )
        from orpheus.numerics.assembled_operator import SparseAssembledOperator

        return SparseAssembledOperator(
            self.scalar * self.op.assemble().matrix,
            domain=self.domain,
            codomain=self.codomain,
        )


class PointwiseOperator(LinearOperator[Domain, Domain]):
    r"""The space-polymorphic POINTWISE (multiplier) family — an
    endomorphism at EVERY admissible space, acting identically.

    The natural stratum of the multiplier algebra the transport layer
    documents at
    :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`
    (``M[1] = I``, ``M[0] = 0``, ``M[f]`` — that class is the BOUND,
    typed-carrier realization; this base is the polymorphic engine
    stratum). A member is *multiplication by a coefficient* — the
    constant ``1`` (:class:`IdentityOperator`), the constant ``0``
    (:class:`ZeroOperator`), a field ``f`` (:class:`DiagonalOperator`) —
    so each output entry depends only on the SAME input entry: nothing
    couples, nothing is transposed across.

    The two laws the type declares (the S4-amendment):

    * **No stored space pair.** ``domain``/``codomain`` answer ``None``
      BY LAW — the member acts at the operand's space, obtained at
      operation time, and ``domain == codomain == that space`` for every
      apply. This is a *typed* statement (discriminated by
      ``isinstance``), not the legacy silent default: a bound operator
      answering ``None`` is a declared migration debt; a pointwise
      operator answering ``None`` is permanently, correctly
      space-polymorphic. Composition handles it through the agreement
      law unchanged ("an operand that declares nothing contributes
      nothing" — :func:`_agreed_space`): a pointwise operand CONFORMS to
      its neighbours' space, which is precisely its naturality.
    * **The Hilbert adjoint is metric-free:** ``A.H is A``. A real
      multiplier commutes with every diagonal metric (pointwise
      multiplications commute), so the metric sandwich cancels
      identically and :meth:`adjoint` returns ``self`` — algebra-closed,
      no :class:`AdjointOperator` wrapper, no spaces needed. This is
      what makes the family's exemption from the unbound-``.H`` refusal
      exact rather than permissive. (Real coefficients throughout —
      the complexification thread revisits at Campaign 2's resolvent.)

    The boundary of the family is the commutation theorem, not
    endomorphy: :class:`PermutationOperator` is endomorphic and acts on
    many spaces, yet permuting does NOT commute with a general diagonal
    metric, so its adjoint genuinely needs the spaces — it is correctly
    NOT a member. Bound multipliers (``MultiplicationOperator``,
    :class:`InverseMetricOperator`) are pointwise in ACTION but defined
    by a space's data — they stay bound classes.
    """

    @property
    def domain(self) -> None:
        r"""``None`` BY LAW — see the class docstring (space-polymorphic:
        the domain is the operand's space, at operation time)."""
        return None

    @property
    def codomain(self) -> None:
        r"""``None`` BY LAW — an endomorphism at the operand's space."""
        return None

    @property
    def is_adjointable(self) -> bool:
        return True  # every real multiplier is self-adjoint (class law)

    @property
    def is_metric_free_adjoint(self) -> bool:
        return True  # the family law — commutes with every diagonal metric

    def adjoint(self) -> "PointwiseOperator[Domain]":
        r"""``A^* = A`` — a real multiplier commutes with every diagonal
        metric, so the Hilbert adjoint IS the operator, in every shipped
        inner product. Algebra-closed (returns ``self``); the generic
        metric-sandwich wrapper is never built for a pointwise member."""
        return self


class IdentityOperator(PointwiseOperator[Domain]):
    r"""The identity operator :math:`I\,x = x` — multiplication by ``1``,
    the pointwise family's unit (:class:`PointwiseOperator`).

    Both axes hold trivially — :math:`I^{-1} = I` and :math:`I^* = I` —
    and both are ALGEBRA-CLOSED: :meth:`inverse` and the inherited
    pointwise :meth:`~PointwiseOperator.adjoint` each return this very
    instance, so there is no ``solve`` verb to keep (solving with the
    identity IS applying its inverse, itself).
    """

    def apply(self, x: Domain, /) -> Domain:
        return x

    def apply_transpose(self, x: Domain, /) -> Domain:
        return x

    @property
    def is_invertible(self) -> bool:
        return True  # I^{-1} = I

    def inverse(self) -> "IdentityOperator[Domain]":
        r"""Return :math:`I^{-1} = I` — this very instance (stateless)."""
        return self


class ZeroOperator(PointwiseOperator[Domain]):
    r"""The ENDOMORPHIC zero :math:`0\,x = 0` — multiplication by ``0``,
    the pointwise family's absorbing element (:class:`PointwiseOperator`).

    Stateless and space-polymorphic: the action routes through ``0.0 * x``,
    echoing the operand's type — the zero OF the operand's own space,
    which is correct precisely because a pointwise member is an
    endomorphism there (``domain == codomain == the operand's space``,
    by the family law). ``apply_transpose`` is the same map
    (:math:`0^{\mathsf T} = 0`), and the inherited pointwise
    :meth:`~PointwiseOperator.adjoint` returns ``self``.

    STRUCTURALLY non-invertible — the singular map par excellence — so it
    declares no ``inverse()`` at all: misuse is a static error, the
    honest surface for a type whose inverse does not exist mathematically
    (Design C; a raising stub would be the harmful-stub anti-pattern this
    module is designed against).

    The zero MAP between two DIFFERENT spaces is a different object —
    :class:`ZeroMorphism`, born bound to its pair. Until the
    S4-amendment (2026-08-22) this class straddled both roles through
    per-site ``codomain_zero``/``transpose_zero`` closures and optional
    space fields (#330); the un-weld split them: the closures' two
    production consumers dissolved (the fission pencil member became
    stack ∘ restriction at A2; the vacuum law binds a
    :class:`ZeroMorphism`), and the hooks retired with them.
    """

    def apply(self, x: Domain, /) -> Domain:
        # Endomorphic by the family law: the zero of the codomain IS the
        # zero of the operand's own space — ``0.0 * x`` echoes it (bare
        # ``np.ndarray`` → ``np.zeros_like(x)`` bit-exact; a typed
        # carrier → a fresh same-class zero via its scalar dunder).
        return cast("Domain", 0.0 * x)

    def apply_transpose(self, x: Domain, /) -> Domain:
        # 0^T = 0 — the same endomorphic echo.
        return cast("Domain", 0.0 * x)

    # is_invertible inherits the base ``False`` — the zero map is singular.


class ZeroMorphism(LinearOperator):
    r"""The zero MAP between two DECLARED spaces:
    :math:`0 : \mathcal D \to \mathcal C`.

    An operator :math:`A : \mathcal D \to \mathcal C` maps the domain
    to the codomain; its action — including the zero action — produces an
    element of :math:`\mathcal C`. The zero map is the ONE operator whose
    action cannot reveal which spaces it connects (every probe returns
    zero), so unlike every other operator it cannot even be *probed* into
    a pair — the pair must be DECLARED, and this class demands both at
    construction (the S4-amendment's base demand, in its sharpest
    instance). The canonical production member is the vacuum boundary
    law's :math:`R = 0 : \Gamma_+ \to \Gamma_-` — the α → 0 member of
    the albedo family ``α · (geometric link)``, realized structurally
    (the link is never built just to be zeroed).

    The action mints the codomain's zero from the BOUND shape, with the
    seam's payload convention: the space describes the STRUCTURAL axes,
    and any trailing axes the operand carries beyond the domain's rank
    (group/spatial payload at the trace seam) ride through unchanged —
    ``zeros(codomain.shape + payload)``. The transpose mints the
    domain's the same way (:math:`0^{\mathsf T} : \mathcal C \to
    \mathcal D` — under #276 A4 duality typing it consumes the
    codomain-cotangent and emits the domain-cotangent, and a zero is a
    zero on either side). This is
    :class:`TraceRestrictionOperator`'s trailing-axes-intact convention,
    stated on a bound pair.
    The pre-amendment spelling reached these shapes through per-site
    ``codomain_zero``/``transpose_zero`` closures, which duplicated what
    the bound spaces already knew; deriving from the binding retires the
    closures AND their caller-side consistency check (the hook-vs-space
    agreement is now unspellable — single source).

    :meth:`adjoint` is algebra-closed and metric-free: the Hilbert
    adjoint of the zero map is the zero map of the swapped pair, in any
    inner product — no sandwich, no wrapper.

    Deliberately UNPARAMETERIZED (the :class:`PermutationOperator`
    precedent): the ndarray-seam consumers (the boundary trace algebra)
    do not satisfy the ``Vector`` protocol statically, and the two
    ``FunctionSpace``\ s — not static type parameters — are this
    operator's identity.
    """

    def __init__(
        self,
        *,
        domain: "FunctionSpace",
        codomain: "FunctionSpace",
    ) -> None:
        if domain is None or codomain is None:  # a loud guard, not typing
            raise TypeError(
                "ZeroMorphism demands BOTH spaces at construction — the "
                "zero map is the one operator whose action cannot reveal "
                "its pair, so the pair must be declared. For the"
                " endomorphic zero at the operand's own space use"
                " ZeroOperator()."
            )
        self._domain = domain
        self._codomain = codomain

    @property
    def domain(self) -> "FunctionSpace":
        return self._domain

    @property
    def codomain(self) -> "FunctionSpace":
        return self._codomain

    def apply(self, x: object, /) -> "np.ndarray":
        r"""The zero of the CODOMAIN — bound structural axes + the
        operand's payload tail, in the operand's dtype."""
        arr = np.asarray(x)
        payload = arr.shape[len(self._domain.shape):]
        return np.zeros(self._codomain.shape + payload, dtype=arr.dtype)

    def apply_transpose(self, y: object, /) -> "np.ndarray":
        r"""The zero of the DOMAIN (the transpose lands there), payload
        riding through as in :meth:`apply`."""
        arr = np.asarray(y)
        payload = arr.shape[len(self._codomain.shape):]
        return np.zeros(self._domain.shape + payload, dtype=arr.dtype)

    @property
    def is_adjointable(self) -> bool:
        return True  # 0^* exists in any metric: the swapped zero map

    def adjoint(self) -> "ZeroMorphism":
        r"""``0^* : \mathcal C \to \mathcal D`` — the swapped pair's zero
        map (metric-free: the zero is self-dual up to the swap)."""
        return ZeroMorphism(domain=self._codomain, codomain=self._domain)

    # is_invertible inherits ``False``; no ``inverse()`` is declared —
    # the same Design-C surface as the endomorphic ZeroOperator.


class _WrappedForward(Protocol):
    r"""The MINIMAL structural contract the wrap-delegate back-half consumes.

    Exactly what :class:`InverseWrapMixin` itself reads of its wrapped
    forward :math:`A`: the function-space pair the inverse SWAPS
    (``domain``/``codomain``) and the forward matvec its ``solve``
    un-inverts through (``apply``). Nothing more — the
    :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
    sibling inverts the MATERIALIZATION and never touches ``inner.solve``
    or ``inner.is_invertible`` (it reads values, not structure), so the
    minimal contract is these three members only (the tighter
    :class:`_InvertibleForward` bound fits the solve-backed siblings).

    Each sibling NARROWS ``_ForwardT`` to what its own ctor guard and
    algorithm need: :class:`InverseOperator` to
    :class:`_InvertibleForward`;
    :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` to the
    schedule-triangular union;
    :class:`~orpheus.numerics.green_operator.GreenOperator` to
    :class:`OperatorSum`; ``MatrixInverseOperator`` to
    :class:`LinearOperator` (its guard needs :meth:`~LinearOperator.as_matrix`).
    """

    @property
    def domain(self) -> Optional["FunctionSpace"]: ...

    @property
    def codomain(self) -> Optional["FunctionSpace"]: ...

    def apply(self, x: Any, /) -> Any: ...


class _InvertibleForward(_WrappedForward, Protocol):
    r"""A solve-backed invertible forward — :class:`InverseOperator`'s narrowing.

    Extends the family-minimal :class:`_WrappedForward` with the two
    members the GENERIC sibling consumes: ``is_invertible`` (its ctor
    guard is the leaf's own value check) and :meth:`solve` — the
    forward's NATIVE inverse-action realization, the permanent face the
    family wrapper delegates through, so the verb lives exactly on the
    conformers the family wraps (value leaves, the product, the sweep
    composites). Delegating through one contract keeps the inverse OBJECT
    and the realization on ONE body (``coding-elegance`` Pattern 2 — no
    reciprocal twin path that could drift by a rounding).
    """

    @property
    def is_invertible(self) -> bool: ...

    def solve(self, b: Any, /) -> Any: ...


_ForwardT = TypeVar("_ForwardT", bound=_WrappedForward)


class InverseWrapMixin(Generic[_ForwardT], metaclass=ABCMeta):
    r"""The wrap-delegate back-half shared by every inverse-family sibling.

    An inverse operator in this codebase is a thin typed wrapper around
    its own FORWARD operator :math:`A` (:attr:`inner`): the wrapper's
    :meth:`apply` realizes :math:`A^{-1}` by the sibling's algorithm, and
    everything else is delegation — the byte-identical back-half shared
    by every inverse-family sibling (:class:`InverseOperator`,
    :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`,
    :class:`~orpheus.numerics.green_operator.GreenOperator`), extracted
    once the third sibling appeared (defer-until-≥2):

    * :attr:`domain` / :attr:`codomain` — the SWAP of the forward's: an
      inverse maps the forward's codomain back to its domain.
    * :meth:`solve` — solving :math:`A^{-1}\,y = b` IS applying
      :math:`A`: the forward matvec ``inner.apply``, delegated.
    * ``is_invertible`` is ``True`` and :meth:`inverse` returns the
      wrapped forward ITSELF — the involution :math:`(A^{-1})^{-1} = A`
      holds by OBJECT IDENTITY, typed per sibling
      through ``_ForwardT``.

    **The canonical seeded-apply signature is part of the back-half**
    (#285, resolved STRUCTURAL): the abstract :meth:`apply` declares
    ``apply(x, /, *, initial_guess=None)`` — the
    :class:`~orpheus.numerics.iteration.SupportsSeededApply` contract
    the iteration drivers consume — so a new sibling CANNOT forget the
    keyword: pyright rejects an override that drops it (LSP), and
    ``ABCMeta`` blocks instantiating a sibling that fails to implement
    it.  Members with no use for a start accept-and-ignore, documented
    per type (an exact inverse has nothing to seed; the sweep threads it
    into the curvilinear Carlson closure; the Green threads it as its
    splitting iteration's start).

    Siblings keep exactly three things of their own: the constructor
    GUARD (what makes their ``inner`` invertible — a value check, a
    type, a derivable splitting), the :meth:`apply` body (the inversion
    algorithm), and ``__repr__``.

    The ADJOINT axis is NOT part of the back-half: ``is_adjointable`` /
    ``.H`` stay at the base defaults — the adjoint-inverse is the #280
    family, deferred (free for the iterative branch, a reverse-DAG
    ``sweep_transpose`` for the direct sweep).

    (This wrap-delegate family is ONE of two kinds of inverse in the
    algebra: ALGEBRA-CLOSED structures invert into themselves — a
    permutation's inverse IS a permutation, a scaled operator's a scaled
    operator — and stay unwrapped as first-class forwards. The canonical
    statement of the split lives on :meth:`OperatorProduct.inverse`.)
    """

    def __init__(self, inner: _ForwardT) -> None:
        #: The forward operator :math:`A` this is the inverse of.
        self.inner = inner

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        # An inverse maps the forward's CODOMAIN back to its DOMAIN.
        return self.inner.codomain

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        return self.inner.domain

    @abstractmethod
    def apply(self, x: Any, /, *, initial_guess: Any | None = None) -> Any:
        r"""Return :math:`A^{-1}\,x` by this sibling's inversion algorithm.

        ``initial_guess`` is the inverse family's canonical driver
        signature: iterative drivers thread the
        previous iterate uniformly, with no per-type signature probes.
        """
        ...

    def solve(self, b: Any, /) -> Any:
        r"""Solve :math:`A^{-1}\,y = b`, i.e. return :math:`A\,b` (the forward).

        The un-invert face: an inverse object IS invertible, and its
        realization verb is the forward matvec — keeping the involution
        web closed (``is_invertible ⟺ a working solve`` on every family
        member).
        """
        return self.inner.apply(b)

    @property
    def is_invertible(self) -> bool:
        return True  # (A^{-1})^{-1} = A — the wrapped forward itself

    def inverse(self) -> _ForwardT:
        r"""Return :math:`(A^{-1})^{-1} = A` — the wrapped forward, by identity.

        The involution law holds as an OBJECT-IDENTITY
        fact: ``A.inverse().inverse() is A``.
        """
        return self.inner


class InverseOperator(InverseWrapMixin[_InvertibleForward], LinearOperator):
    r"""The inverse operator :math:`A^{-1}` of a solve-backed leaf, in operator form.

    The GENERIC member of the #226 inverse family — the name is earned by
    exactly the universal contract and nothing more ("round-trip
    alone earns only *InverseOperator*"): :meth:`apply` inverts, the
    round-trip :math:`A^{-1}(A\,x) = x` holds to the forward's own ``solve``
    precision, and no fancier invariant (S-direct seed-independence,
    G-Neumann, M-materialise) is promised. Structured inverses with a
    distinguishing invariant get their own named types
    (:class:`~orpheus.sn.operators.sweep_operator.SweepOperator` for the
    triangular sweep;
    :class:`~orpheus.numerics.green_operator.GreenOperator` for the
    preconditioned-splitting sum;
    :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`
    for the dense direct factorization) — this class serves any
    solve-backed forward with NO more specific named inverse: the
    value-bearing LEAVES (:class:`DiagonalOperator`,
    :class:`~orpheus.transport.operators.multiplication_operator.MultiplicationOperator`),
    whose inverse action is an exact pointwise division, AND the
    invertible COMPOSITES: :meth:`OperatorProduct.inverse` returns
    ``InverseOperator(self)`` (#285), so the wrapped inverse action there
    is the product's own ``solve``, :math:`B^{-1}(A^{-1}\,q)`, not a
    division.

    **One realization, not a reciprocal twin.** :meth:`apply` DELEGATES to
    the forward's own :meth:`solve`, bit-identical to today's gated call —
    it does NOT re-derive the inverse action. For a value-bearing LEAF the
    delegation matters doubly: a reciprocal twin would (a) differ from
    ``solve`` by a rounding (:math:`(1/c)\cdot b \neq b/c` in FP), and
    (b) for a cross-section multiplier mint a units-dishonest "reciprocal
    cross-section" field (:math:`1/\Sigma` is a mean free path, a
    DIFFERENT named quantity). The division realization carries the
    inverse semantics without either lie.

    The wrap-delegate back-half (domain↔codomain swap /
    ``solve→inner.apply`` / ``is_invertible`` / ``inverse()→inner``) is
    inherited from :class:`InverseWrapMixin`. This class keeps only its
    ctor guard (the leaf's own ``is_invertible`` value check),
    :meth:`apply`, and ``__repr__``.
    """

    def __init__(self, inner: _InvertibleForward) -> None:
        if not inner.is_invertible:
            raise NotInvertible(
                f"InverseOperator requires an invertible leaf; "
                f"{type(inner).__name__}.is_invertible is False."
            )
        super().__init__(inner)

    def apply(self, x: Any, /, *, initial_guess: Any | None = None) -> Any:
        r"""Return :math:`A^{-1}\,x` — the leaf's own ``solve`` (bit-identical).

        ``initial_guess`` is the inverse family's CANONICAL driver
        signature: iterative drivers thread the previous
        iterate uniformly, with no per-type signature probes.  An EXACT
        pointwise inverse has no use for a starting point — the argument is
        accepted and unused (contrast
        :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`, whose
        sweep threads it into the curvilinear Carlson closure).
        """
        del initial_guess  # exact inverse — no iterative start to seed
        return self.inner.solve(x)

    def __repr__(self) -> str:
        return f"InverseOperator({self.inner!r})"


class PermutationOperator(LinearOperator):
    r"""Index permutation along a configurable axis: :math:`(P x)_i = x_{\pi(i)}`.

    For a permutation :math:`\pi : \{0, \ldots, N-1\} \to \{0, \ldots, N-1\}`
    represented as an integer array ``perm`` of length :math:`N`, the
    apply action gathers entries along ``axis`` according to ``perm``:

    .. math::

        (P\,x)_{i_0 \ldots i_{a-1}\,j\,i_{a+1} \ldots} \;=\;
        x_{i_0 \ldots i_{a-1}\,\pi(j)\,i_{a+1} \ldots}

    The transpose is the inverse permutation :math:`\pi^{-1}`, computed
    once at construction via ``np.argsort(perm)``.

    ⭐ **Ask the algebra whether it is an involution; there is no flag.**
    Until G6.3 step 5 this class carried an ``is_involution`` attribute,
    set at construction from ``perm[perm] == np.arange(N)``. It is retired,
    and the retirement is the *point* rather than a tidy-up: an involution
    is a claim about :math:`P \circ P`, so it needs that composition to
    exist, and the composition is exactly what the operator algebra already
    knows how to form and to refuse. Spell it::

        (P @ P).apply(x)        # same-space: compare against x
        P @ P                   # cross-space: IncompatibleOperatorComposition

    The second line is the whole argument. A permutation bound
    :math:`\Gamma_+ \to \Gamma_-` — every specular kernel the SN realizer
    builds — has no square, and a stored ``bool`` had to answer *something*
    anyway. `[M]` for ONE physical law (a mirror about ``x`` on ``xmin``)
    the raw index test answered ``True`` on ``gauss_legendre(4/8)``,
    ``product(4,4)`` and ``level_symmetric(6)`` and ``False`` on
    ``lebedev(17)`` — an answer tracking the quadrature's local index
    ordering rather than the mirror, and unfalsifiable at every row because
    the flag's documented purpose (self-adjointness in the unweighted inner
    product) is undefined between two different spaces. Routing the
    question through ``@`` replaces a value that could be wrong with a
    composition that cannot be formed. A full-space rule like the mirror
    pairing
    :meth:`~orpheus.numerics.quadrature.Quadrature.ordinate_permutation`
    derives for a reflection IS an involution on
    :math:`\{0, \ldots, N-1\}`, and asks and answers as one; periodic
    shifts and rotational reorderings are not.

    A permutation is always invertible (:math:`P^{-1} = P^T`), and its
    inverse is ALGEBRA-CLOSED: :meth:`inverse` returns the inverse
    permutation as a first-class :class:`PermutationOperator` (#226
    taxonomy step 1) whose ``apply`` is the same
    ``np.take(·, inverse_perm)`` gather as :meth:`apply_transpose`.

    Parameters
    ----------
    perm
        1-D integer array of length :math:`N` whose entries are a
        permutation of :math:`\{0, \ldots, N-1\}`. Validated at
        construction; rejecting duplicates and out-of-range entries
        with :class:`ValueError`.
    axis
        Tensor axis along which the permutation acts. The operator
        broadcasts on every other axis.
    domain, codomain : FunctionSpace, optional
        The spaces this permutation maps between. For the SN deck
        transformation — a specular mirror realized as a **length-1 chain**
        (campaign step G6.3, issue **#330**) — these are
        :math:`\Gamma_+(f)` and :math:`\Gamma_-(f)`, and binding them is
        what makes ``@`` refuse a mis-composed chain and ``.H`` the
        metric-aware Hilbert adjoint rather than the bare transpose.
        Both extents are checked against :attr:`n` along :attr:`axis` at
        construction, because a mis-bound space is SILENT at apply-time.

    Attributes
    ----------
    perm : np.ndarray
        Forward permutation :math:`\pi`, as 1-D ``intp`` array.
    inverse_perm : np.ndarray
        Inverse permutation :math:`\pi^{-1}`, precomputed via
        :func:`numpy.argsort`.
    axis : int
        The tagged tensor axis.
    n : int
        Length of the permuted axis.
    """

    def __init__(
        self, perm: np.ndarray, axis: int = 0,
        *,
        domain: Optional[FunctionSpace] = None,
        codomain: Optional[FunctionSpace] = None,
    ) -> None:
        perm = np.asarray(perm, dtype=np.intp)
        if perm.ndim != 1:
            raise ValueError(
                f"PermutationOperator perm must be 1-D; got shape {perm.shape}"
            )
        n = perm.size
        # Validate: perm is a true permutation of {0, ..., n-1}.
        if n == 0 or not (
            perm.min() == 0
            and perm.max() == n - 1
            and np.unique(perm).size == n
        ):
            raise ValueError(
                "PermutationOperator perm must be a permutation of "
                f"{{0, ..., {n - 1}}}; got {perm!r}."
            )
        self.perm = perm
        self.inverse_perm = np.argsort(perm)
        self.axis = int(axis)
        self.n = n
        self._domain = checked_space_extent(
            domain, n, axis=self.axis,
            owner="PermutationOperator", role="domain",
        )
        self._codomain = checked_space_extent(
            codomain, n, axis=self.axis,
            owner="PermutationOperator", role="codomain",
        )

    @property
    def domain(self) -> Optional[FunctionSpace]:
        return self._domain

    @property
    def codomain(self) -> Optional[FunctionSpace]:
        return self._codomain

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.take(x, self.perm, axis=self.axis)

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        return np.take(x, self.inverse_perm, axis=self.axis)

    # NO ``solve``: the inverse is ALGEBRA-CLOSED —
    # :meth:`inverse` returns the inverse permutation as a first-class
    # forward whose ``apply`` is the SAME ``np.take(·, inverse_perm)``
    # gather (P^{-1} = P^T, bit-identical) — so solving is
    # ``.inverse().apply(b)``; ``apply_transpose`` keeps the gather as
    # the Euclidean-transpose verb.

    @property
    def is_invertible(self) -> bool:
        return True  # P^{-1} = P^T — a permutation is always invertible

    def inverse(self) -> "PermutationOperator":
        r"""Return :math:`P^{-1}` as a first-class :class:`PermutationOperator`.

        Built on the precomputed :attr:`inverse_perm` — its :meth:`apply`
        is the SAME integer gather as this operator's :meth:`solve` /
        :meth:`apply_transpose` (bit-identical: no arithmetic at all), and
        ``argsort`` of a permutation is exactly involutive in integer math,
        so :math:`(P^{-1})^{-1}` reproduces :attr:`perm` EXACTLY (§13 I2).

        ⭐ **The binding is INVERTED, not carried and not dropped.**
        :math:`P : V \to W` has :math:`P^{-1} : W \to V`, so the spaces
        swap. Dropping them instead — which is what this returned before
        the ends could be bound (G6.3 step 5) — would be the quieter bug of
        the two: the inverse would compose with anything, and the
        deck transformation's return leg would lose exactly the typing the
        forward leg had just been given.
        """
        return PermutationOperator(
            self.inverse_perm, axis=self.axis,
            domain=self._codomain, codomain=self._domain,
        )

    @property
    def is_adjointable(self) -> bool:
        return True


def checked_space_extent(
    space: Optional[FunctionSpace],
    expected: int,
    *,
    axis: int = 0,
    owner: str,
    role: str,
) -> Optional[FunctionSpace]:
    r"""Refuse a bound space whose extent contradicts a stored length.

    ⭐ **Why this exists as a shared primitive.** An operator that carries BOTH
    a length (``n_total``, ``n``, ``n_inflow``, ``len(cos_w)``) and a bound
    space is describing the SAME fact twice, and the two can disagree. That
    disagreement
    is **invisible at apply-time** — the arrays still broadcast, so the operator
    computes a plausible wrong answer — which makes construction the only place
    it can be caught.

    The redundancy is transitional: the tree-wide mandate (**#330**) retires
    the lengths in favour of MANDATORY spaces — G6.5 (2026-08-07) measured why
    the retirement cannot land sooner (binding is optional this era, and the
    space-less arms must still name their extents). Until then this keeps the
    pair honest, and it is one routine rather than four because four operators
    now need it
    (:class:`TraceRestrictionOperator`, :class:`PermutationOperator`,
    :class:`~orpheus.sn.boundary.angular.PartialCurrentOperator`,
    :class:`~orpheus.sn.boundary.angular.IsotropicEmissionOperator`) — well
    past the threshold at which a repeated check earns extraction.

    Parameters
    ----------
    space : FunctionSpace or None
        The bound space, or ``None`` while binding remains optional.
    expected : int
        The length the operator stores for that end.
    axis : int
        Which of the space's axes the length describes.
    owner, role : str
        Names for the error message — the class and which end (``"domain"`` /
        ``"codomain"``).

    Returns
    -------
    FunctionSpace or None
        ``space`` unchanged when consistent, so callers can assign the result.
    """
    if space is None:
        return None
    if not 0 <= axis < len(space.shape):
        raise ValueError(
            f"{owner} {role}={space!r} has {len(space.shape)} axes, so "
            f"axis={axis} is out of range."
        )
    actual = space.shape[axis]
    if actual != expected:
        raise ValueError(
            f"{owner} {role}={space!r} has extent {actual} along axis {axis}, "
            f"but this operator's {role} is {expected} rows. The space and the "
            f"length describe the SAME fact and disagree; a mis-bound space is "
            f"SILENT at apply-time because the arrays still broadcast."
        )
    return space


class TraceRestrictionOperator(LinearOperator):
    r"""Restriction onto an index subset along an axis:
    :math:`(\gamma_S x)_i = x_{S(i)}`.

    Given a **sorted, unique** index set
    :math:`S \subset \{0, \ldots, N-1\}` of size :math:`m < N`, the apply
    action gathers those entries along ``axis``, producing an array whose
    length along that axis is :math:`m` rather than :math:`N`:

    .. math::

        \gamma_S : \mathbb{R}^N \to \mathbb{R}^m, \qquad
        (\gamma_S\,x)_j \;=\; x_{S(j)}.

    The transpose is the **scatter** — zeros everywhere, the input written
    back into the selected rows:

    .. math::

        \iota_S = \gamma_S^{\mathsf T} : \mathbb{R}^m \to \mathbb{R}^N,
        \qquad
        (\iota_S\,y)_i \;=\;
        \begin{cases} y_j & i = S(j) \\ 0 & i \notin S. \end{cases}

    Relation to :class:`PermutationOperator` — a **sibling, not a
    subclass**. The mechanism is the same ``np.take`` gather with a
    non-square index array, but the algebra is different in kind: a
    permutation is a bijection (invertible, with an algebra-closed
    :meth:`~PermutationOperator.inverse`), whereas a
    restriction is **rank-deficient by construction** and has a scatter
    transpose rather than an inverse. Inheriting would have promised
    guarantees this type cannot honour.

    Why it exists
    -------------

    These are the **trace operators** :math:`\gamma_\pm` of the affine
    boundary form :math:`\gamma_-\psi = R\,G\,\gamma_+\psi + q` — the very
    maps the theory page names, which the codebase had spelled three
    different ways and typed as none of them. Every one of those spellings
    is a composition of this operator and its transpose:

    ================================================  ==================
    spelling found in the boundary subsystem          is
    ================================================  ==================
    a slice-write ``out[sel] = full[sel]``             :math:`\iota_S \circ \gamma_S`
    a dense diagonal multiply by an inflow mask        :math:`\iota_S \circ \gamma_S`
    a sparse tensor zeroing the inflow rows            :math:`I - \iota_S \circ \gamma_S`
    ================================================  ==================

    (That third spelling was a class of its own, ``IncomingOrdinateMaskTensor``,
    until campaign phase **B3.3** retired it: once B3.2 narrowed the SN law's
    domain to :math:`\Gamma_+`, the rows it preserved left the operator's
    domain and it lost its last construction site. Note it is
    :math:`I - \iota_S \circ \gamma_S`, whose range is
    :math:`\Gamma_+ \oplus \Gamma_{\text{tan}}` — the face partition is
    THREE-way, so "not inflow" was never "outflow".)

    and the observation that ``P_in ∘ P_out = 0`` stops being a curiosity:
    it is :math:`\gamma_- \circ \iota_+ = 0`, true because two disjoint
    index sets have nothing to hand each other.

    Guards, and what each one prevents
    ----------------------------------

    **Sorted** is canonical form, and it is load-bearing one tier up: the
    local↔global remap (``to_local``) lives on the half-trace SPACE since
    G6.5 — the embedding data is the space's — and its ``searchsorted``
    haystack carries, on the canonical trace-builder path, the same VALUES
    as this gather's index array (two objects; the agreement is CHECKED
    elementwise at binding — ERR-077 — not assumed). One ascending
    spelling per subspace means the gather, the space's row order, and the
    metric restriction are three views of one selection that cannot drift.
    (The 1-D-prefix-vs-2-D lesson that used to justify the remap living
    here travels with it — see
    :meth:`~orpheus.numerics.spaces.angular_trace_space.AngularFaceTraceSpace.to_local`.)

    **Unique** because a repeated row is not a restriction — the transpose
    would silently drop all but one contribution, and the pair would stop
    satisfying :math:`\gamma_S \circ \iota_S = I`.

    **The shape guard on apply** is the one a hand-rolled reduced
    permutation does not have: fed a full-length input it returns a
    same-shaped array of *wrong values*, with no raise. Measured. Here a
    domain mismatch is a loud ``ValueError`` naming both lengths.

    Structurally NON-invertible — the restriction discards rows — so it
    declares no ``inverse()``; misuse is a static error rather than a
    silent wrong answer. (A restriction covering *every* row is the
    identity in disguise; use :class:`IdentityOperator` for that.)

    Parameters
    ----------
    indices : np.ndarray
        Sorted, unique row indices in ``[0, n_total)`` — the subspace this
        operator restricts onto.
    n_total : int
        Length of the FULL axis, i.e. the domain's extent along ``axis``.
    axis : int
        Axis to gather along. Trailing axes broadcast, so this composes as
        a :class:`TensorProductOperator` factor.
    domain, codomain : FunctionSpace, optional
        The spaces this restriction maps between — for a boundary trace,
        :math:`\gamma_\pm : \Gamma(f) \to \Gamma_\pm(f)`. Binding them is what
        makes ``.H`` the **Hilbert** adjoint :math:`G_V^{-1}\gamma^{\mathsf T}G_W`
        rather than the bare Euclidean transpose, and what lets ``@`` refuse a
        mis-composed chain; unbound, both metric applications are silently
        skipped (campaign step G6.3, issue **#330**).

        ⭐ **Checked against** ``n_total`` **and** ``indices`` **at
        construction**, so the space and the length cannot disagree. That
        matters because they are redundant *today*: ``n_total`` /
        :attr:`n_restricted` are lengths standing where a space belongs, and
        the guard is what keeps the duplication honest until the tree-wide
        mandate (#330) retires them in favour of MANDATORY spaces. (G6.5,
        2026-08-07, measured why the retirement cannot land sooner: binding
        is optional this era, and the trace-less arm must still name its
        extent. What G6.5 did move is the local↔global remap — ``to_local``
        now lives on the half-trace space, where the index data does.) A
        binding that contradicts its own lengths is the one failure this
        class cannot detect any other way — the shapes still broadcast, so
        the wrong answer would be silent.
    """

    def __init__(
        self, indices: np.ndarray, n_total: int, axis: int = 0,
        *,
        domain: Optional[FunctionSpace] = None,
        codomain: Optional[FunctionSpace] = None,
    ) -> None:
        idx = np.asarray(indices, dtype=np.intp)
        if idx.ndim != 1:
            raise ValueError(
                f"TraceRestrictionOperator indices must be 1-D; got shape "
                f"{idx.shape}."
            )
        n_total = int(n_total)
        if n_total <= 0:
            raise ValueError(
                f"TraceRestrictionOperator n_total must be positive; got "
                f"{n_total}."
            )
        if idx.size and (idx.min() < 0 or idx.max() >= n_total):
            raise ValueError(
                f"TraceRestrictionOperator indices must lie in "
                f"[0, {n_total}); got min={idx.min()}, max={idx.max()}."
            )
        if np.unique(idx).size != idx.size:
            raise ValueError(
                "TraceRestrictionOperator indices must be unique — a repeated "
                "row is not a restriction: the scatter transpose would drop "
                f"all but one contribution. Got {idx!r}."
            )
        if idx.size > 1 and not np.all(idx[1:] > idx[:-1]):
            raise ValueError(
                "TraceRestrictionOperator indices must be SORTED ascending — "
                "one canonical spelling per subspace, matching the half-trace "
                "space's own row order (whose `to_local` searchsorts the same "
                f"array). Got {idx!r}; pass `np.sort(indices)`."
            )
        self.indices = idx
        self.n_total = n_total
        self.axis = int(axis)
        self._domain = self._checked_space(domain, n_total, "domain")
        self._codomain = self._checked_space(codomain, idx.size, "codomain")

    def _checked_space(
        self, space: Optional[FunctionSpace], expected: int, role: str,
    ) -> Optional[FunctionSpace]:
        """Refuse a space whose extent along :attr:`axis` contradicts the length.

        For the CODOMAIN, extent alone is not enough (**ERR-077**): the gather
        emits rows in :attr:`indices` order, and a space that declares its own
        ambient rows (``ordinate_indices``) must declare the SAME ones — or
        the space's ``to_local`` answers a row order this gather never
        produces. Pre-G6.5 the operator-side ``to_local`` was implicitly
        closed under the gather's own array; relocating the remap to the
        space made the agreement a separate fact, so it is CHECKED where the
        pair is bound rather than assumed. Duck-typed deliberately: numerics
        cannot import the spaces module (circular), and any space declaring
        ambient rows owes the agreement regardless of its class.
        """
        space = checked_space_extent(
            space, expected,
            axis=self.axis, owner="TraceRestrictionOperator", role=role,
        )
        declared = getattr(space, "ordinate_indices", None)
        if role == "codomain" and declared is not None:
            declared = np.asarray(declared, dtype=np.intp)
            if not np.array_equal(declared, self.indices):
                raise ValueError(
                    f"TraceRestrictionOperator: the bound codomain "
                    f"{getattr(space, 'name', space)!r} declares "
                    f"ordinate_indices {declared.tolist()} but this gather "
                    f"reads rows {self.indices.tolist()} — same extent, "
                    f"different rows. The space's to_local would answer a row "
                    f"order the gather does not emit (ERR-077)."
                )
        return space

    @property
    def domain(self) -> Optional[FunctionSpace]:
        r""":math:`\Gamma(f)` — the full space restricted FROM, when bound."""
        return self._domain

    @property
    def codomain(self) -> Optional[FunctionSpace]:
        r""":math:`\Gamma_\pm(f)` — the subspace restricted ONTO, when bound."""
        return self._codomain

    @property
    def n_restricted(self) -> int:
        """Extent of the codomain along :attr:`axis`."""
        return int(self.indices.size)

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.shape[self.axis] != self.n_total:
            raise ValueError(
                f"TraceRestrictionOperator.apply: input has "
                f"{x.shape[self.axis]} rows along axis {self.axis}, but this "
                f"restriction's DOMAIN is the full space of {self.n_total}. "
                f"Passing the restricted space back in ("
                f"{self.n_restricted} rows) is the likely mistake — the "
                f"restriction consumes the full trace and emits the subspace, "
                f"not the reverse."
            )
        return np.take(x, self.indices, axis=self.axis)

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        r"""The scatter :math:`\iota_S` — zeros, then write the selected rows."""
        x = np.asarray(x)
        if x.shape[self.axis] != self.n_restricted:
            raise ValueError(
                f"TraceRestrictionOperator.apply_transpose: input has "
                f"{x.shape[self.axis]} rows along axis {self.axis}, but the "
                f"scatter's DOMAIN is the restricted space of "
                f"{self.n_restricted}. Passing the full space in "
                f"({self.n_total} rows) is the likely mistake."
            )
        full_shape = list(x.shape)
        full_shape[self.axis] = self.n_total
        out = np.zeros(full_shape, dtype=x.dtype)
        sel: list = [slice(None)] * x.ndim
        sel[self.axis] = self.indices
        out[tuple(sel)] = x
        return out

    # NO ``to_local``: the local↔global remap moved to the half-trace SPACE
    # at G6.5 (`AngularFaceTraceSpace.to_local`) — the embedding data is the
    # space's, and on the canonical builder path the space's row order and
    # this operator's gather are the same array by construction.

    # NO ``inverse()``: a restriction is rank-deficient by construction.
    # The transpose is the scatter, NOT an inverse — ``ι ∘ γ`` is the
    # projector onto the subspace, not the identity on the full space.

    @property
    def is_invertible(self) -> bool:
        return False

    @property
    def is_adjointable(self) -> bool:
        return True


class _AxisMarginalBase(LinearOperator):
    r"""Shared retained state of the axis collapse pair (CS4b S6.0b).

    The two arrows of one axis collapse — the retraction
    :math:`R = \pi_*` (:class:`AxisRetractionOperator`) and its section
    :math:`E` (:class:`AxisSectionOperator`) — are the **forgetful
    retention** of a stage-2 generator's output: the single-region
    indicator frame over the axis's index set
    (``GalerkinFrame(IndicatorBasis, axis measure)``), built eagerly at
    the mint site (:func:`orpheus.numerics.frame._collapse_pair`), read
    for its induced data, and discarded. The ruled discipline (user,
    2026-08-24): *"a stage-2 generator induces structure on both the
    space and the operator, and the two inductions must be minted
    together, at one site … forgetting = retaining the induced parts;
    accessors are provenance."*

    What the operators retain IS the induced data and nothing else: the
    bound product spaces, the ndarray dims the axis occupies, the axis's
    flat weights (the frame measure's diagonal), and — on the section —
    the frame's 1×1 ``discrete_gram`` entry (the rank-one Parseval
    metric :math:`\Sigma w`). No frame and no faces: a frame FACE is a
    view holding ``frame:``, so retaining one would retain the
    generator; true forgetting copies the induced parts out.

    Construction is the mint's internal — the public path is
    :meth:`FunctionSpace.retraction
    <orpheus.numerics.space.FunctionSpace.retraction>` /
    :meth:`FunctionSpace.section
    <orpheus.numerics.space.FunctionSpace.section>`, which memoize one
    mint per space per axis label. Admission (axis-built space, exactly
    one label hit, NODAL kind, the typed-``EnergyAxis`` clause-2
    refusal, a surviving marginal) lives at the mint, once. Both
    realizations are **born bound** (the S4-amendment lens: an operator
    is not an operator without its two spaces — there is no unbound arm
    to refuse).

    An axis may span SEVERAL ndarray dims (the 2-D spatial axis is one
    factor of shape ``(nx, ny)`` carrying the 2-D ``V_cell`` measure);
    the pair contracts/broadcasts over all of them with the axis's own
    weights. ``weights is None`` IS the counting measure (the Axis
    canonicalization), realized at the mint as the frame measure's
    explicit ones.
    """

    def __init__(
        self,
        *,
        full_space: "FunctionSpace",
        marginal_space: "FunctionSpace",
        axis_shape: tuple[int, ...],
        dims: tuple[int, ...],
        flat_weights: np.ndarray,
    ) -> None:
        self._full_space = full_space
        self._marginal_space = marginal_space
        self._axis_shape = axis_shape
        self._dims = dims
        self._flat_weights = flat_weights

    # ── shared kernels (one spelling per direction, Pattern 2) ────────

    def _contract(self, x: np.ndarray) -> np.ndarray:
        r"""``Σ_axis w · x`` — the measure contraction over the axis dims.

        The analysis-face CONTENT of the mint's rank-one frame, spelled
        to be BIT-IDENTICAL with the shipped angular reduction
        (``AngularField._integrate_angular_values``'s
        ``einsum("n,ng...->g...", w, values)``) on its case — a leading
        1-dim axis: ``moveaxis`` to the front is then the identity view
        and the einsum program normalizes to the same contraction
        (gated ``np.array_equal``, G6.5; the frame-content equivalence
        is the tightness gate). A multi-dim axis is fused to one flat
        dim first (reshape of a moved view).
        """
        x = np.asarray(x)
        if len(self._dims) == 1:
            moved = np.moveaxis(x, self._dims[0], 0)
        else:
            moved = np.moveaxis(x, self._dims, range(len(self._dims)))
            moved = moved.reshape(-1, *moved.shape[len(self._dims):])
        return np.einsum("n,n...->...", self._flat_weights, moved)

    def _broadcast_scaled(self, x: np.ndarray, scale: np.ndarray) -> np.ndarray:
        r"""``scale_axis ⊗ x`` — scatter ``x`` across the axis dims, each
        slice scaled by that slot's ``scale`` entry (flat over the axis)."""
        x = np.asarray(x)
        out = np.multiply.outer(scale.reshape(self._axis_shape), x)
        nd = len(self._dims)
        return np.moveaxis(out, range(nd), self._dims)

    # ── the bound carriers ────────────────────────────────────────────

    @property
    def is_invertible(self) -> bool:
        return False

    @property
    def is_adjointable(self) -> bool:
        return True


class AxisRetractionOperator(_AxisMarginalBase):
    r"""The retraction :math:`R = \pi_*` — fiber integration over one
    named axis: :math:`(R\,\psi)(\cdot) = \sum_n w_n\, \psi(n, \cdot)`.

    **Canonical names.** :math:`R \circ E = \mathrm{id}` (`[M]`
    bit-exact) makes the pair a split epi/mono pair: :math:`R` is the
    *retraction* (split epimorphism) and :math:`E` its *section* (Mac
    Lane, CWM §I.5) — the collapse doctrine's own "retract rule".
    Content-wise :math:`R` is the pushforward :math:`\pi_*` (fiber
    integration) along the projection that forgets the axis, and its
    Hilbert adjoint is the pullback :math:`R^\dagger = \pi^*` — the
    plain broadcast (`[M]` ``np.array_equal``): the
    :math:`(\pi_*, \pi^*)` adjunction realized on the discrete product.

    **Frame-induced** (S6.0b): this operator is the analysis-face
    content of the single-region indicator frame over the axis's index
    set, minted by :func:`orpheus.numerics.frame._collapse_pair` via
    :meth:`FunctionSpace.retraction
    <orpheus.numerics.space.FunctionSpace.retraction>` — the frame is
    built eagerly there, its induced weights are copied out, and it is
    discarded (the forgetful-map discipline; see
    :class:`_AxisMarginalBase`). The tightness gate pins this
    operator's einsum against the frame's own
    :meth:`~orpheus.numerics.basis.base.Basis.analyze` content.

    The space-level realization of the angular flux reduction
    :math:`\phi = \int \psi\, \mathrm{d}\Omega \approx \sum_n w_n \psi_n`
    (`[M]` bit-identical with the shipped einsum, G6.5) and,
    axis-generically, of any factor-measure marginal (the spatial
    volume integral; a 2-D spatial axis contracts both dims). Domain =
    the minting space; codomain = the same product with the axis
    dropped (its OTHER factors keep their measures, so the marginal's
    metric stays physical).

    The two arrows differ by exactly the total weight:
    :math:`R^\dagger = \Sigma w \cdot E` (`[M]` ``np.array_equal``).
    Naming BOTH arrows canonically is the anti-ERR-051 move: a single
    undiscriminated verb would have had to choose a convention, and a
    re-pointed call site would have silently changed a source by
    :math:`\Sigma w`.

    Structurally rank-deficient (the marginal discards the axis) — no
    ``inverse()``; the transpose is the weighted scatter
    :math:`(R^{\mathsf T}\phi)(n, \cdot) = w_n\,\phi(\cdot)`, and the
    HILBERT adjoint rides the bound spaces' metrics through ``.H``.
    """

    @property
    def domain(self) -> "FunctionSpace":
        r"""The full space — the product carrying the contracted axis.

        Narrowed non-Optional: the pair is born bound (the S4-amendment
        lens), so a minted retraction ALWAYS has its two spaces.
        """
        return self._full_space

    @property
    def codomain(self) -> "FunctionSpace":
        r"""The marginal space — the remaining axes, measures intact."""
        return self._marginal_space

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.shape != self._full_space.shape:
            raise ValueError(
                f"AxisRetractionOperator.apply: input shape {x.shape} "
                f"does not match the full space {self._full_space.shape} "
                f"(the retraction consumes the full product and emits "
                f"the marginal, not the reverse)."
            )
        return self._contract(x)

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        r"""The weighted scatter :math:`w_n \phi` — the Euclidean transpose."""
        x = np.asarray(x)
        if x.shape != self._marginal_space.shape:
            raise ValueError(
                f"AxisRetractionOperator.apply_transpose: input shape "
                f"{x.shape} does not match the marginal space "
                f"{self._marginal_space.shape}."
            )
        return self._broadcast_scaled(x, self._flat_weights)


class AxisSectionOperator(_AxisMarginalBase):
    r"""The measure-normalized section of the axis retraction:
    :math:`(E\,\phi)(n, \cdot) = \phi(\cdot) / \Sigma w`.

    **Canonical name.** :math:`E` is DEFINED by
    :math:`R \circ E = \mathrm{id}` (`[M]` bit-exact) — the right
    inverse of the retraction, i.e. the *section* of the split pair
    (split monomorphism; Mac Lane CWM §I.5). "Embedding" was rejected
    as non-canonical for this object (ratified 2026-08-24): any
    injective structure-preserving map is an embedding — the pullback
    :math:`\pi^* = R^\dagger` is one too — so that name cannot
    discriminate the two arrows this two-type design exists to
    discriminate; "embedding" survives only as a generic adjective. The
    composite :math:`P = E \circ R` is the conditional expectation onto
    axis-constant functions — the :math:`w`-mean projector (`[M]`
    idempotent bit-exact, G6.2).

    **Frame-induced** (S6.0b): the reconstruction-face content composed
    with the inverse Gram, :math:`E = R_{\text{frame}} \circ G^{-1}` —
    the divisor IS the mint frame's 1×1 ``discrete_gram`` entry, the
    rank-one **Parseval metric** (F-0's theorem at :math:`K = 1`),
    induced at the mint and never a hand convention. ⚠ The induced read
    is ULP-equivalent — NOT universally bit-identical — to the old
    ``weights.sum()`` spelling: `[M]` 2026-08-24 (post-landing
    correction), exact at ``GL{2,4,5,6,12,16,32,64}`` and **1 ULP off
    at GL8** (gram ``1.9999999999999998`` vs sum ``2.0``; the section
    then differs from the pre-S6.0b iso kernel by ``2.07e-16`` max
    rel). The first probe's 8 fixtures skipped GL8, and its universal
    consequence was refuted by the S7 docs audit. Ruled acceptable:
    principled-over-bit-identical — the divisor's SOURCE is the frame's
    induction; the gram-derivation gate pins the value at its honest
    tier.

    On the angular axis this is the isotropic-source projection
    :math:`Q/\Sigma w` broadcast across the ordinates
    (``AngularSourceSink.from_isotropic``'s kernel — gated
    ``np.array_equal`` on the GL4 fixture, G6.6), and the iso column of
    the harmonic frame's physical adjoint WHEN the frame's discrete
    Gram is DIAGONAL (`[M]` slab L=1: ``face.H(e₀φ) == E(φ)`` to
    5.6e-17; a DENSE Gram — slab or sphere at L=2 — breaks it: pre-P7
    by the undressed continuum-metric factor (the then-recorded
    F-0/CS4c debt), and still post-P7 under the honest dense dressing,
    because :math:`G^{+}` couples the modes and admits no per-ℓ scalar
    collapse. The discriminator is Gram diagonality, not geometry —
    unchanged). The metric-free
    form ``reconstruction(e₀φ)/W == E(φ)`` is bit-exact regardless
    (``scratch/probe_s6_q5_dissolution.py`` carries the DENSE arm).

    NOT the adjoint of :class:`AxisRetractionOperator` — that is the
    plain broadcast :math:`R^\dagger = \Sigma w \cdot E` (`[M]` exact).
    The two arrows carry different names and different types precisely
    so the :math:`\Sigma w` convention cannot be silently swapped at a
    call site (the ERR-051 class becomes unspellable).

    Domain = the marginal space; codomain = the full space. An axis
    whose SIGNED measure sums to zero has NO section — the rank-one
    Gram is singular, so the frame has no canonical dual: the mint
    leaves this arm unminted and :meth:`FunctionSpace.section
    <orpheus.numerics.space.FunctionSpace.section>` refuses, while the
    retraction over the same axis stays legal.
    """

    def __init__(
        self,
        *,
        full_space: "FunctionSpace",
        marginal_space: "FunctionSpace",
        axis_shape: tuple[int, ...],
        dims: tuple[int, ...],
        flat_weights: np.ndarray,
        total_weight: float,
    ) -> None:
        super().__init__(
            full_space=full_space,
            marginal_space=marginal_space,
            axis_shape=axis_shape,
            dims=dims,
            flat_weights=flat_weights,
        )
        self._total_weight = total_weight

    @property
    def total_weight(self) -> float:
        r""":math:`\Sigma w` — the mint frame's 1×1 ``discrete_gram`` entry."""
        return self._total_weight

    @property
    def domain(self) -> "FunctionSpace":
        r"""The marginal space the section lifts FROM (born bound — non-Optional)."""
        return self._marginal_space

    @property
    def codomain(self) -> "FunctionSpace":
        r"""The full space the section lifts INTO."""
        return self._full_space

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.shape != self._marginal_space.shape:
            raise ValueError(
                f"AxisSectionOperator.apply: input shape {x.shape} "
                f"does not match the marginal space "
                f"{self._marginal_space.shape} (the section lifts the "
                f"marginal into the full product, not the reverse)."
            )
        # Divide THEN broadcast — the same float ops as the shipped
        # from_isotropic kernel (÷Σw first, then the axis broadcast), so
        # the equivalence gate can pin np.array_equal rather than a ULP
        # bound. The leading-1-dim case is literally its spelling.
        scaled = x / self._total_weight
        expanded = np.expand_dims(scaled, self._dims)
        return np.broadcast_to(expanded, self._full_space.shape).copy()

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        r"""The unweighted axis sum over :math:`\Sigma w` — the Euclidean
        transpose of the broadcast-and-scale."""
        x = np.asarray(x)
        if x.shape != self._full_space.shape:
            raise ValueError(
                f"AxisSectionOperator.apply_transpose: input shape "
                f"{x.shape} does not match the full space "
                f"{self._full_space.shape}."
            )
        return np.add.reduce(x, axis=self._dims) / self._total_weight


class TensorProductOperator(LinearOperator):
    r"""Per-axis tensor product :math:`A \otimes B \otimes \cdots`.

    Given a tuple of linear operators :math:`A_1, A_2, \ldots, A_k`
    acting on **independent** tensor axes (i.e. each carries an
    ``axis`` attribute and broadcasts on the rest), the tensor product
    operator's action is the sequential per-axis application

    .. math::

        (A_1 \otimes A_2 \otimes \cdots \otimes A_k)\,x
        \;=\; A_k\bigl(\cdots A_2(A_1\,x) \cdots\bigr).

    Because the constituents act on disjoint axes, the order does not
    matter (the operators commute on the joint tensor). Both structural
    axes are factor-wise INTERSECTIONS — invertible iff every factor
    is, adjointable iff every factor is — computed recursively by the
    predicates, and the inverse is ALGEBRA-CLOSED (a tensor product of
    the factor inverses), so there is no ``solve`` verb.

    Algebraic laws (verified by tests):

    * **Adjoint distributivity**:
      :math:`(A \otimes B)^* = A^* \otimes B^*`.
    * **Per-axis composition**:
      :math:`(A \otimes B) \circ (C \otimes D) = (A \circ C) \otimes (B \circ D)`
      when ``A``/``C`` share an axis and ``B``/``D`` share an axis.
    * **Inverse on every axis**:
      :math:`(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}` when both
      factors are invertible.
    * **Spaces by AGREEMENT, not position** (:attr:`domain`): since the
      factors commute, a position-based rule would give an
      order-independent operator order-dependent spaces. Every factor
      that declares a space must declare the same one; silence
      contributes nothing; disagreement raises
      :class:`IncompatibleOperatorComposition`. This is the law
      :class:`OperatorSum` — the other commutative composite — already
      obeyed, shared as :func:`_agreed_space`.

    Parameters
    ----------
    ops : tuple of LinearOperator
        The tensor-product factors. Each MUST advertise an ``axis``
        attribute (or accept an ``axis`` kwarg in :meth:`apply`) and
        broadcast on every other axis. :class:`IdentityOperator`,
        :class:`DiagonalOperator`, and any
        :class:`OperatorProduct`/:class:`OperatorSum` of such operators
        satisfy the contract.

    Notes
    -----

    Relation to numpy: :func:`numpy.kron`, :func:`numpy.tensordot`,
    :func:`numpy.einsum` are array primitives — the *implementation*
    layer. :class:`TensorProductOperator` is the *operator algebra
    type* — it carries axis tags, capability set, and the algebraic
    laws above. Its :meth:`apply` routes through each constituent's
    :meth:`apply`, which is itself typically a single ``np.einsum`` or
    broadcast-multiply. Different abstraction layers, complementary.

    Operators with non-axis-preserving signatures (e.g. an angular
    moment projection that consumes one ordinate axis and produces
    two harmonic-coefficient axes) do not fit this contract — their
    action changes tensor rank. Use them directly via their own
    :meth:`apply`; do not wrap in :class:`TensorProductOperator`.
    """

    def __init__(self, ops: tuple) -> None:
        if len(ops) < 1:
            raise ValueError("TensorProductOperator requires at least one factor")
        # Eager apply-guard per factor (composition time, never at call).
        for op in ops:
            if not callable(getattr(op, "apply", None)):
                raise TypeError(
                    f"TensorProductOperator factor must expose 'apply'; "
                    f"{type(op).__name__} lacks it."
                )
        self.ops: tuple = tuple(ops)
        # Eager space agreement, for the same reason :class:`OperatorSum`
        # checks it and by the same shared law: the factors COMMUTE, so the
        # product's spaces cannot be a function of factor order.
        _agreed_space(self.ops, "domain", "TensorProductOperator")
        _agreed_space(self.ops, "codomain", "TensorProductOperator")

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        r"""The space the factors agree on — see :func:`_agreed_space`.

        A factor that declares nothing (the group-axis
        :class:`IdentityOperator` every shipped boundary law carries) leaves
        the binding to the factor that does, so ``K_ω ⊗ I`` is bound exactly
        where ``K_ω`` is. Before G6.3 step 8.0 this returned the base's
        ``None``, which meant the binding was real at the inner factor and
        INVISIBLE at the object a realizer hands out — and, because
        :class:`AdjointOperator` reads the spaces to apply the metrics, it
        also meant ``(K_ω ⊗ I).H`` silently degraded to the Euclidean
        transpose (`[M]` 87 % relative error against the weighted adjoint on
        the Lambertian, exact only for the specular mirror, whose metric
        cancels).
        """
        return _agreed_space(self.ops, "domain", "TensorProductOperator")

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        """The space the factors agree on — see :attr:`domain`."""
        return _agreed_space(self.ops, "codomain", "TensorProductOperator")

    @staticmethod
    def _build(a: "LinearOperator", b: "LinearOperator") -> "TensorProductOperator":
        """Construct a flattened ``A & B`` instance.

        If either operand is itself a :class:`TensorProductOperator`,
        absorb its factors so ``(A & B) & C`` and ``A & (B & C)`` both
        produce ``TensorProductOperator((A, B, C))``.
        """
        a_ops = a.ops if isinstance(a, TensorProductOperator) else (a,)
        b_ops = b.ops if isinstance(b, TensorProductOperator) else (b,)
        return TensorProductOperator(a_ops + b_ops)

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = x
        for op in self.ops:
            out = op.apply(out)
        return out

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        out = x
        # Adjoint of tensor product is tensor product of adjoints.
        # Apply transposes of factors (order irrelevant for disjoint
        # axes); the per-factor guard-narrow licenses each call.
        for op in self.ops:
            if not adjointable(op):
                raise MissingAdjoint(
                    f"TensorProductOperator.apply_transpose requires "
                    f"every factor to transpose ((A⊗B)^T = A^T⊗B^T); "
                    f"{type(op).__name__}.is_adjointable is False."
                )
            out = op.apply_transpose(out)
        return out

    # NO ``solve``: the inverse is ALGEBRA-CLOSED —
    # :meth:`inverse` returns the tensor product of the factor inverses,
    # a first-class forward — so solving is ``.inverse().apply(b)``.

    @property
    def is_invertible(self) -> bool:
        # (A⊗B)^{-1} = A^{-1}⊗B^{-1} — invertible iff every factor is
        # (recursive over the factors, like every composite predicate).
        return all(op.is_invertible for op in self.ops)

    def inverse(self) -> "TensorProductOperator":
        r"""Return :math:`(A \otimes B \otimes \cdots)^{-1} = A^{-1} \otimes B^{-1} \otimes \cdots`.

        The factor-wise structural inverse (the docstring's "inverse on
        every axis" law). Factor ORDER is preserved —
        the factors act on disjoint axes and commute, exactly as
        :meth:`solve` applies them in stored order — so the action is
        bit-identical to :meth:`solve` given each factor's own
        ``inverse().apply ≡ solve`` identity.
        """
        factor_inverses = []
        for op in self.ops:
            if not invertible(op):
                raise NotInvertible(
                    f"TensorProductOperator.inverse requires every factor "
                    f"to be invertible ((A⊗B)^{{-1}} = A^{{-1}}⊗B^{{-1}}); "
                    f"{type(op).__name__}.is_invertible is False."
                )
            factor_inverses.append(op.inverse())
        return TensorProductOperator(tuple(factor_inverses))

    @property
    def is_adjointable(self) -> bool:
        # (A⊗B)^T = A^T⊗B^T — adjointable iff every factor is.
        return all(op.is_adjointable for op in self.ops)

    @property
    def is_metric_free_adjoint(self) -> bool:
        # A tensor product of metric-free operators is metric-free (derived).
        return all(op.is_metric_free_adjoint for op in self.ops)


class SumOfTensorProductsOperator(LinearOperator):
    r"""Sum of tensor products :math:`\sum_k A_k \otimes B_k \otimes \cdots`.

    The §15.2 / §15A.2 canonical form for scattering and streaming in
    the operator-algebra view:

    * **Streaming** (§15.1):
      :math:`L = D_x \otimes \Omega_x \otimes I_g + D_y \otimes \Omega_y \otimes I_g`.
    * **Scattering** (§15.2):
      :math:`S = \sum_\ell P_\ell \otimes \Sigma_{s,\ell}` (per-:math:`\ell`
      block-diagonal on moment space).

    Algebraically just :class:`OperatorSum` over
    :class:`TensorProductOperator` summands, but exposed as a named
    type because the structure carries V&V invariants worth checking
    explicitly:

    * Each summand IS a :class:`TensorProductOperator` —
      :meth:`assert_separable`.
    * (Future) common-axis factorisation — when many summands share
      an axis-factor, refactoring saves work.

    Parameters
    ----------
    summands : tuple of TensorProductOperator
        The tensor-product summands. Each MUST be a
        :class:`TensorProductOperator`; mixing in non-separable
        operators makes the type label misleading.

    Notes
    -----

    The implementation backs onto :class:`OperatorSum` —
    :meth:`apply` simply sums each summand's action — so all the
    algebra of :class:`OperatorSum` (composition, scaling, capability
    intersection) is inherited by delegation. The named subclass
    exists for the type signal and the assertion methods.
    """

    def __init__(self, summands: tuple) -> None:
        if len(summands) < 1:
            raise ValueError(
                "SumOfTensorProductsOperator requires at least one summand"
            )
        for s in summands:
            if not isinstance(s, TensorProductOperator):
                raise TypeError(
                    f"SumOfTensorProductsOperator summands must be "
                    f"TensorProductOperator instances; got {type(s).__name__}. "
                    f"Use OperatorSum for general operator addition."
                )
        self.summands: tuple = tuple(summands)
        # A sum of tensor products is a sum: same agreement law as
        # :class:`OperatorSum`, one spelling (G6.3 step 8.0).
        _agreed_space(self.summands, "domain", "SumOfTensorProductsOperator")
        _agreed_space(self.summands, "codomain", "SumOfTensorProductsOperator")

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        """The space the summands agree on — see :func:`_agreed_space`."""
        return _agreed_space(self.summands, "domain", "SumOfTensorProductsOperator")

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        """The space the summands agree on — see :func:`_agreed_space`."""
        return _agreed_space(self.summands, "codomain", "SumOfTensorProductsOperator")

    def apply(self, x: np.ndarray) -> np.ndarray:
        out = self.summands[0].apply(x)
        for s in self.summands[1:]:
            out = out + s.apply(x)
        return out

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        for s in self.summands:
            if not adjointable(s):
                raise MissingAdjoint(
                    f"SumOfTensorProductsOperator.apply_transpose requires "
                    f"every summand to transpose; "
                    f"{type(s).__name__}.is_adjointable is False."
                )
        out = self.summands[0].apply_transpose(x)
        for s in self.summands[1:]:
            out = out + s.apply_transpose(x)
        return out

    def assert_separable(self) -> None:
        """Assert every summand is a :class:`TensorProductOperator`.

        Holds by construction (the constructor enforces it), so this
        method is a no-op contract-validator. Useful as documentation
        and as a hook for subclasses or future invariant checks.
        """
        for s in self.summands:
            if not isinstance(s, TensorProductOperator):
                raise AssertionError(
                    f"SumOfTensorProductsOperator summand is not "
                    f"separable: {type(s).__name__}"
                )

    @property
    def is_adjointable(self) -> bool:
        # ∑ A_k⊗B_k transposes summand-wise — adjointable iff every summand
        # is. (Solve does not propagate through sums, so is_invertible
        # inherits the base ``False``.)
        return all(s.is_adjointable for s in self.summands)

    @property
    def is_metric_free_adjoint(self) -> bool:
        # Summand-wise, like the transpose (derived).
        return all(s.is_metric_free_adjoint for s in self.summands)


class InverseMetricOperator(LinearOperator):
    r"""A :class:`~orpheus.numerics.space.FunctionSpace`'s inverse metric, as an OPERATOR.

    :math:`G^{+} : V^{*} \to V` — the adapter that lets a space's metric
    enter the operator algebra instead of only being *applied* to arrays.

    **Why this exists.** A space's metric enters the operator algebra
    through this adapter — the trace metrics' inverses, a degenerate
    metric's Moore–Penrose face — wherever :math:`G^{+}` of a SPACE is the
    operator wanted. (Until CS4c step 6 item 6.2c-ii the frame's projector
    was spelled through it too, as
    ``frame.conjugate(InverseMetricOperator(frame.gram))`` over a metric-twin
    of the test space; the frame now owns that factor as a typed arrow,
    :attr:`~orpheus.numerics.frame.FrameBase.gram_inverse` — an endomorphism
    of a twin space cannot compose with the faces once the metric enters
    space identity.)

    **The arithmetic is the SPACE's, not ours** (Cardinal Rule 2): every
    apply delegates to
    :meth:`~orpheus.numerics.space.FunctionSpace.apply_inverse_metric`, so
    the Moore–Penrose masking on a degenerate metric —
    :math:`1/G` where :math:`G \neq 0`, **0** on
    :math:`\ker G` — is single-sourced there and cannot drift.  That
    masking is not academic: `[M]` the SN trace metric
    :math:`G = |\Omega\cdot\hat n|\,w_n` is **exactly zero** on tangential
    ordinates — 50 % of rows under ``product(4,4)``, 16 % under
    ``lebedev(11)``, 0 % under ``level_symmetric``.

    ⛔ **Not invertible, and it says so by ABSENCE.** On a degenerate
    metric this is a pseudo-inverse, so ``G⁺G ≠ I`` and there is no
    ``inverse()`` method to call — the
    :class:`TraceRestrictionOperator` spelling, not a raising stub
    (:attr:`is_invertible` stays ``False``).  ⚠ Consequently
    ``InverseMetricOperator(space)`` composed with a forward metric does
    **not** cancel; if you want the round trip, assert it on a field you
    know is off the null space.

    Self-adjoint: the metric is a symmetric positive-semi-definite form —
    a diagonal weight, or since P7 a dense
    :class:`~orpheus.numerics.metric.DenseMetric` (symmetry guarded at its
    construction) — so :math:`(G^{+})^{\mathsf T} = G^{+}` and
    ``apply_transpose`` is ``apply`` for every realization.

    Parameters
    ----------
    space :
        The space whose metric to invert.  Serves as BOTH ``domain`` and
        ``codomain`` — the metric is an endomorphism of the carrier's
        shape (it re-weights, it does not move between spaces), and
        binding both ends is what lets
        :class:`OperatorProduct`'s compatibility guard check the
        composition.
    """

    def __init__(self, space: FunctionSpace) -> None:
        self._space = space

    @property
    def space(self) -> FunctionSpace:
        """The space whose metric this inverts."""
        return self._space

    @property
    def domain(self) -> Optional[FunctionSpace]:
        return self._space

    @property
    def codomain(self) -> Optional[FunctionSpace]:
        return self._space

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self._space.apply_inverse_metric(x)

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        # Self-adjoint: a real diagonal weight is its own transpose.
        return self._space.apply_inverse_metric(x)

    @property
    def is_adjointable(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"InverseMetricOperator({self._space!r})"


class DiagonalOperator(PointwiseOperator):
    r"""Diagonal (pointwise) multiplication by a coefficient field.

    The operator multiplies a carrier tensor :math:`x` by a coefficient
    array :math:`c` that occupies a **sub-product** of the carrier's
    axes and is **constant** over the complementary ``broadcast_axes``:

    .. math::

        (D x)_{\mathbf{i}} \;=\;
        c_{\,\mathbf{i}\setminus\mathrm{bcast}} \; x_{\mathbf{i}}

    i.e. ``D.apply(x) == np.expand_dims(c, broadcast_axes) * x``. The
    coefficient's rank equals ``x.ndim - len(broadcast_axes)`` and its
    axes map, in order, onto the carrier axes NOT in ``broadcast_axes``.

    This is the canonical "diagonal in some basis" / pointwise-multiply
    operator. Two regimes it must express:

    * **1-D special case** — a 1-D coefficient on ONE carrier axis,
      broadcast over all others. This is the Grand Report v3 §9
      :math:`W` (``AngularWeightMatrix``) and the "multiply-by-weights
      along one axis" primitive (MoC track-weight diagonal, CP
      region-volume weighting, MC importance weighting). Spell it
      ``DiagonalOperator(w, axis=k)``; the action is rank-agnostic
      (broadcasts over however many other axes the carrier has).
    * **The multigroup-collision case** — a coefficient of shape
      ``(ng, *spatial)`` broadcast over the LEADING ordinate axis of a
      ``(N, ng, *spatial)`` angular flux, so that
      ``D.apply(psi) == sigma[None] * psi``. This is the broadcast
      engine the transport-layer ``MultiplicationOperator(σ_t)``
      delegates to. Spell it
      ``DiagonalOperator(sigma, broadcast_axes=(0,))``.

    Self-adjoint by construction (real-valued coefficient), so
    :meth:`apply_transpose` is the same code path as :meth:`apply`.
    Invertible iff every coefficient entry is non-zero — the
    VALUE-dependent arm of the split: :meth:`inverse` is declared and
    :meth:`solve` divides, both refusing eagerly
    (:class:`NotInvertible`) on a zero entry.

    Parameters
    ----------
    coefficient : np.ndarray
        The coefficient field :math:`c`. Its rank determines how many
        carrier axes it occupies (``x.ndim - len(broadcast_axes)``).
    broadcast_axes : tuple of int, optional
        The carrier axes over which the coefficient is constant — the
        positions :func:`numpy.expand_dims` inserts singleton dims.
        When omitted, the **1-D special case** applies: ``coefficient``
        MUST be 1-D and ``axis`` selects the single carrier axis it
        occupies (rank-agnostic broadcast over the rest).
    axis : int, default 0
        Used ONLY in the 1-D special case (``broadcast_axes is None``):
        the single carrier axis the 1-D coefficient occupies. Ignored
        when ``broadcast_axes`` is given.

    Notes
    -----

    Construction does NOT materialise a dense diagonal matrix; the
    action is a single broadcast-multiply
    (``self._broadcast(x.ndim) * x``) so memory cost is
    :math:`O(\mathrm{size}(c))` regardless of the carrier's shape.

    The two construction modes are unified through one broadcast helper
    (:meth:`_broadcast`): the 1-D ``axis`` mode is rank-agnostic (the
    carrier's rank is read at apply-time, so the same operator acts on
    a 1-D, 2-D, or N-D carrier), whereas the explicit ``broadcast_axes``
    mode pins both the coefficient rank and the complement, which a
    multi-axis coefficient requires.

    The ``weights``/``axis`` attributes remain available in the 1-D
    case (``coefficient`` is exposed for both modes) as the back-compat
    alias for ``from_measure`` ergonomics and the existing 1-D call
    sites. Composition (``&`` / :class:`TensorProductOperator`,
    ``@`` / :class:`OperatorProduct`) does NOT read them — every composer
    routes purely through ``apply`` / ``solve``.

    Use :meth:`from_measure` when a 1-D coefficient lives on a
    :class:`~orpheus.numerics.measure.DiscreteMeasure` — common for the
    angular axis of an SN field, where the operator is built from
    ``quad.weights``.
    """

    def __init__(
        self,
        coefficient: np.ndarray,
        broadcast_axes: tuple[int, ...] | None = None,
        *,
        axis: int = 0,
    ) -> None:
        coeff = np.asarray(coefficient, dtype=float)

        if broadcast_axes is None:
            # 1-D special case: a single-axis coefficient broadcasting,
            # rank-agnostically, over every other carrier axis. The
            # carrier rank is unknown at construction, so the broadcast
            # placement is deferred to apply-time (see _broadcast).
            if coeff.ndim != 1:
                raise ValueError(
                    f"DiagonalOperator without broadcast_axes is the 1-D "
                    f"special case; coefficient must be 1-D, got shape "
                    f"{coeff.shape}. For an N-D coefficient pass "
                    f"broadcast_axes=(...)."
                )
            self.broadcast_axes: tuple[int, ...] | None = None
        else:
            # General case: an N-D coefficient pinned onto an explicit
            # complement of carrier axes.
            bcast = tuple(int(a) for a in broadcast_axes)
            if len(set(bcast)) != len(bcast):
                raise ValueError(
                    f"DiagonalOperator broadcast_axes must be distinct, "
                    f"got {bcast}."
                )
            self.broadcast_axes = bcast

        # ``axis`` is consulted ONLY in the 1-D special case; storing it
        # as a plain int (default 0) keeps the attribute well-typed and
        # harmless in broadcast mode.
        self.axis = int(axis)
        self.coefficient = coeff

    @classmethod
    def from_measure(
        cls, measure, axis: int = 0,
    ) -> "DiagonalOperator":
        """Construct from the weights of a :class:`DiscreteMeasure`.

        Convenience constructor for the canonical 1-D case where the
        diagonal IS the discrete measure's weights — e.g.
        ``DiagonalOperator.from_measure(quad.measure, axis=0)`` is the
        Grand Report v3 §9 ``AngularWeightMatrix``.
        """
        return cls(measure.weights, axis=axis)

    @property
    def weights(self) -> np.ndarray:
        """The 1-D coefficient vector (the historical ``weights`` name).

        Available ONLY in the 1-D special case (``broadcast_axes is
        None``); it is the back-compat alias for ``from_measure`` and
        the existing 1-D call sites. Reading it on a multi-axis-
        coefficient instance is an illegal state and raises (Pattern 4)
        rather than returning an N-D array under a 1-D name.
        """
        if self.broadcast_axes is not None:
            raise AttributeError(
                "DiagonalOperator.weights is the 1-D special case's "
                "coefficient vector; this operator has an N-D coefficient "
                f"of shape {self.coefficient.shape} on broadcast_axes="
                f"{self.broadcast_axes}. Use .coefficient instead."
            )
        return self.coefficient

    def _broadcast(self, ndim: int) -> np.ndarray:
        """Reshape the coefficient to broadcast over an ``ndim`` carrier.

        Single source of truth for both construction modes:

        * 1-D ``axis`` mode — return a view of shape
          ``(1, ..., 1, N, 1, ..., 1)`` with ``N`` at ``self.axis``
          (rank-agnostic: built fresh for the carrier's actual ``ndim``).
        * explicit ``broadcast_axes`` mode — return
          ``np.expand_dims(coefficient, broadcast_axes)``, inserting a
          singleton at each broadcast axis so the coefficient occupies
          the complementary axes in order.
        """
        if self.broadcast_axes is None:
            shape = [1] * ndim
            shape[self.axis] = -1
            return self.coefficient.reshape(shape)
        return np.expand_dims(self.coefficient, self.broadcast_axes)

    def _check_shape(self, x: np.ndarray) -> None:
        """Validate the carrier's rank/axis sizes against the coefficient."""
        if self.broadcast_axes is None:
            if x.shape[self.axis] != self.coefficient.shape[0]:
                raise ValueError(
                    f"DiagonalOperator(axis={self.axis}) expects axis size "
                    f"{self.coefficient.shape[0]}; got {x.shape[self.axis]} "
                    f"in input of shape {x.shape}."
                )
            return
        expected_rank = x.ndim - len(self.broadcast_axes)
        if self.coefficient.ndim != expected_rank:
            raise ValueError(
                f"DiagonalOperator(broadcast_axes={self.broadcast_axes}) "
                f"expects a rank-{expected_rank} coefficient for a "
                f"{x.ndim}-D carrier; got rank-{self.coefficient.ndim} "
                f"coefficient of shape {self.coefficient.shape}."
            )

    def apply(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x)
        self._check_shape(x_arr)
        return self._broadcast(x_arr.ndim) * x_arr

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        # Real-valued diagonal is self-adjoint.
        return self.apply(x)

    def solve(self, b_vec: np.ndarray) -> np.ndarray:
        # The value-dependent guard (Pattern 4 heritage: never a silent
        # IEEE NaN on a σ=0 division — the legacy bare-σ path had no gate).
        if not self.is_invertible:
            raise NotInvertible(
                "DiagonalOperator.solve requires non-zero coefficient "
                "entries; this operator has at least one zero entry."
            )
        b_arr = np.asarray(b_vec)
        self._check_shape(b_arr)
        return b_arr / self._broadcast(b_arr.ndim)

    @property
    def is_invertible(self) -> bool:
        # Invertible iff every coefficient entry is non-zero (D^{-1} = 1/c).
        return bool(np.all(self.coefficient != 0.0))

    def inverse(self) -> "InverseOperator":
        r"""Return :math:`D^{-1}` as an :class:`InverseOperator` over this leaf.

        Delegation, NOT a reciprocal-coefficient twin: the returned
        object's ``apply`` IS :meth:`solve` (the division ``b / c``),
        bit-identical — whereas ``DiagonalOperator(1/c)`` would multiply
        by a rounded reciprocal and drift by an ulp. The generic name is
        the honest one (round-trip alone earns exactly
        "InverseOperator"; a diagonal division carries no distinguishing
        invariant beyond it).
        """
        if not self.is_invertible:
            raise NotInvertible(
                "DiagonalOperator.inverse requires non-zero coefficient "
                "entries; this operator has at least one zero entry."
            )
        return InverseOperator(self)



class RankOneOperator(LinearOperator):
    r"""The rank-1 dyad :math:`|v\rangle\langle w|` — a reconstruction column ⊗ a functional row.

    A :class:`RankOneOperator` is the outer product of a **reconstruction**
    vector :math:`v` (the column, an output direction) and a
    :class:`~orpheus.numerics.functional.Functional` :math:`\langle w|` (the
    row, the contraction). Its action on a carrier :math:`x` is

    .. math::

        (\,|v\rangle\langle w|\,)\,x \;=\; v \,\cdot\, \langle w, x\rangle ,

    i.e. ``reconstruction * functional.evaluate(x)``: the functional contracts
    :math:`x` to the inner product :math:`\langle w, x\rangle` (with
    ``keepdims`` on the contracted axis), and the reconstruction broadcasts back
    over that length-1 axis. The functional OWNS the contraction (its weight and
    axis); the operator only broadcasts — so the matvec routes THROUGH
    ``functional.evaluate`` and there is no parallel inline reduction to drift
    from it.

    Build instances with :func:`outer` (the readable verb,
    ``outer(reconstruction, functional)``). A genuine ``M × K`` rank-1 operator
    (``v ∈ ℝ^M``, ``w ∈ ℝ^K``, ``M ≠ K``) is legal — there is no same-shape
    constraint between the column and the row (the old ``left.shape ==
    right.shape`` check was an artifact of the square-only legacy form).

    Native to the multigroup fission emission
    :math:`F = |\chi\rangle\langle\nu\Sigma_f| =
    \texttt{outer}(\chi,\ \mathrm{ReactionRateFunctional}(\nu\Sigma_f))`
    (Grand Report v3 §15): the production-rate co-vector
    :math:`\langle\nu\Sigma_f, \phi\rangle = \sum_{g'}\nu\Sigma_{f,g'}\phi_{g'}`
    is the
    :class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
    row, and the emission spectrum :math:`\chi` is the reconstruction column.
    Fission is the **rank-1 (single-mode) degenerate** of the multi-mode
    scattering kernel ``R∘Λ∘M`` (a :class:`~orpheus.numerics.frame.FrameBase`
    manages the analogous *stack* of dyads); see
    :mod:`orpheus.transport.operators.integral_kernel_operator`.

    Relation to :class:`TensorProductOperator`
    -------------------------------------------

    A :class:`RankOneOperator` satisfies the TP-factor contract (it acts on the
    functional's contracted axis and broadcasts on the others), so it composes
    as a TP factor when the algebra wants the type-visible separable form:

    .. code-block:: python

        F_kernel = outer(chi, InnerProductFunctional(nu_sigma_f, axis=0)) & IdentityOperator()

    The :class:`IdentityOperator` factor advertises the spatial-axis broadcast;
    the TP fold reduces to :meth:`RankOneOperator.apply` bit-identically
    (``IdentityOperator.apply`` returns ``x``).

    Adjointable exactly when the row is an
    :class:`~orpheus.numerics.functional.InnerProductFunctional` (the usual
    case, including its
    :class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
    specialisation) — the VALUE-dependent arm on the adjoint axis. Rank-1
    operators are **structurally non-invertible** (no ``inverse()`` declared —
    the kernel is the orthogonal complement of the row along the contracted axis),
    but they DO have a **transpose**: :meth:`apply_transpose` is the dual dyad
    :math:`|w\rangle\langle v|` — swap the column :math:`v` with the row's
    weight :math:`w`, contracting :math:`\langle v,\cdot\rangle` over the same
    axis. This is the Euclidean transpose :math:`A^{T}`; the metric-correct
    Hilbert adjoint :math:`A^\dagger = G^{-1}A^{T}G` is the
    :attr:`~LinearOperator.H` wrapper's job. The fission adjoint
    :math:`F^\dagger\psi^* = \nu\Sigma_f\,(\chi\cdot\psi^*)` is exactly this
    dyad-swap (#276). A nonlinear / opaque functional has no dual
    column, so such a dyad advertises ``apply`` only. See
    :ref:`operator-adjoint`.

    Parameters
    ----------
    reconstruction : Vector | numpy.ndarray
        The column :math:`v` — the output direction the inner product is
        broadcast against. Aligns with the carrier on the complement of the
        functional's contracted axis; its size on that axis is the output
        dimension ``M``.
    functional : Functional
        The row co-vector :math:`\langle w|` — contracts the carrier to
        :math:`\langle w, x\rangle` over its own axis (typically the leading
        group axis for the multigroup reaction rate). Usually an
        :class:`~orpheus.numerics.functional.InnerProductFunctional` (generic)
        or a
        :class:`~orpheus.transport.reaction_rate_functional.ReactionRateFunctional`
        (the domain-typed reaction rate).

    Notes
    -----
    Bit-identity with the legacy ``(right * x).sum(axis, keepdims) * left``
    formulation is preserved because
    :meth:`~orpheus.numerics.functional.InnerProductFunctional.evaluate` performs
    that exact ``(w * x).sum(axis, keepdims=True)`` reduction — the same numpy
    primitive, the same axis, the same order — and the reconstruction broadcast
    is elementwise. IEEE-754 pairwise-reduction order is preserved.
    """

    def __init__(
        self,
        reconstruction: "Vector | np.ndarray",
        functional: "Functional",
    ) -> None:
        # The dyad |v⟩⟨w|: ``reconstruction`` is the column (output) vector v;
        # ``functional`` is the row co-vector ⟨w| that OWNS the contraction (its
        # weight and axis). NO same-shape guard — a genuine M×K rank-1 operator
        # (M ≠ K) is legal; the functional validates its own contraction against
        # the carrier at apply time (the old left.shape == right.shape check was
        # an artifact of the square-only legacy form, not a real constraint).
        self.reconstruction = reconstruction
        self.functional = functional

    @property
    def domain(self) -> Optional["FunctionSpace"]:
        r"""``None`` — a DOCUMENTED Optional (the S4-amendment's fourth
        answer): the dyad's two vectors are bare arrays that carry no
        spaces today, so there is nothing honest to derive. The owning
        flip is CS4c's kernel binding (``FissionKernel``'s dyad bound to
        its Energy space), where the row/column gain their spaces and
        this override narrows.
        """
        return None

    @property
    def codomain(self) -> Optional["FunctionSpace"]:
        r"""``None`` — see :attr:`domain` (the CS4c-owned debt)."""
        return None

    @property
    def is_adjointable(self) -> bool:
        # The VALUE-dependent arm on the ADJOINT axis: the dual dyad
        # |w⟩⟨v| exists iff the row ⟨w| is a genuine co-vector whose
        # weight IS the dual column — an InnerProductFunctional (the
        # ReactionRateFunctional specialisation included). A nonlinear /
        # opaque functional has no dual column. Mirrors — and gates —
        # the apply_transpose realization below.
        from orpheus.numerics.functional import InnerProductFunctional

        return isinstance(self.functional, InnerProductFunctional)

    def apply(self, x: np.ndarray) -> np.ndarray:
        # |v⟩⟨w| x = v · ⟨w, x⟩. The functional IS the contraction — it returns
        # the inner product ⟨w, x⟩ with ``keepdims`` on the contracted axis, and
        # the reconstruction broadcasts back over that length-1 axis. Routing the
        # matvec THROUGH ``functional.evaluate`` is what makes the row-factor a
        # first-class object (no parallel inline reduction to drift from it).
        recon = np.asarray(getattr(self.reconstruction, "values", self.reconstruction))
        return recon * self.functional.evaluate(x)

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        # The dual dyad: (|v⟩⟨w|)ᵀ = |w⟩⟨v|. The transpose swaps the column and
        # the row — the new column is the old row's weight w (the dual column),
        # the new row is ⟨v| (the old reconstruction as a co-vector on the SAME
        # contracted axis). So Aᵀx = w · ⟨v, x⟩, the Euclidean transpose (the .H
        # wrapper adds the metric). Reuses the InnerProductFunctional contraction
        # primitive — single source of truth with the forward `apply` row-factor.
        from orpheus.numerics.functional import InnerProductFunctional

        # The isinstance IS the narrowing (the same fact is_adjointable
        # advertises): the body reads the IPF-typed row's .weight/.axis.
        if not isinstance(self.functional, InnerProductFunctional):
            raise MissingAdjoint(
                "RankOneOperator.apply_transpose requires the row functional to "
                "be an InnerProductFunctional (a co-vector with a dual column); "
                f"got {type(self.functional).__name__} — a nonlinear functional "
                "has no dual column."
            )
        column = np.asarray(
            getattr(self.functional.weight, "values", self.functional.weight)
        )
        dual_row = InnerProductFunctional(
            np.asarray(getattr(self.reconstruction, "values", self.reconstruction)),
            axis=self.functional.axis,
        )
        return column * dual_row.evaluate(x)


def outer(
    reconstruction: "Vector | np.ndarray",
    functional: "Functional",
) -> RankOneOperator:
    r"""Build the rank-1 dyad :math:`|v\rangle\langle w|` from a column and a co-vector.

    The universal constructor for a rank-1 :class:`LinearOperator`: a
    :class:`~orpheus.numerics.vector.Vector` (or ``ndarray``) ``reconstruction``
    :math:`v` (the column, the output direction) tensored with a
    :class:`~orpheus.numerics.functional.Functional` ``functional`` :math:`\langle w|`
    (the row, the contraction). The action is

    .. math::

        (\,|v\rangle\langle w|\,)\,x \;=\; v \,\cdot\, \langle w, x\rangle ,

    i.e. ``reconstruction * functional.evaluate(x)``. Every separable rank-1
    operator in the algebra is one of these; the multi-mode generalisation is a
    sum of dyads managed by a :class:`~orpheus.numerics.frame.FrameBase` (the
    spectral / block-term decomposition). The canonical transport instance is
    the fission emission kernel
    :math:`F = \texttt{outer}(\chi,\ \mathrm{ReactionRateFunctional}(\nu\Sigma_f))`
    (see :class:`~orpheus.transport.operators.fission.FissionOperator`).
    """
    return RankOneOperator(reconstruction, functional)

