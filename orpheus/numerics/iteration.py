r"""Iteration primitives for the operator-algebra :math:`(A - \sum_i g_i)`.

The neutron transport equation, in its operator-algebra form, is

.. math::

    \Bigl(A - \sum_i g_i\Bigr)\,\psi = q_{\rm ext}
    \qquad\text{(fixed source)}

.. math::

    \Bigl(A - \sum_i g_i\Bigr)\,\psi = \tfrac{1}{k}\,F\,\psi
    \qquad\text{(eigenvalue)}

where :math:`A` is the INVERTIBLE resolvent operand — the left-hand
side whose inverse the iteration applies — and the :math:`g_i` are the
lagged coupling gains, handed to the variadic drivers as
``(A, *gains)``.  For SN the binding is :math:`A = L + C` (streaming
:math:`L = \Omega\cdot\nabla` plus collision :math:`C = \Sigma_t\cdot`)
with gains :math:`(S,\ B)` — the honest within-group operator
:math:`A - \sum_i g_i = L+C-S-B` — but this layer never sees the
leaves; a matrix method hands its full assembled loss matrix as
:math:`A` with a shorter gain tuple.  Fission :math:`F` is never a
gain in the eigenvalue posing — the outer loop scales it by
:math:`1/k` (Lewis & Miller §6.4 frame the decomposition; Trefethen &
Bau 1997 §3.2 give the matrix-free Krylov view).  The letter matters:
project-wide, ``L`` names the STREAMING LEAF (which alone is not
invertible), and invertible left-hand-side operands are the ``A``
family (the resolvent operand ``A``; the k-posing's
``A_loss = A - S``) — this module follows that convention.

This module installs the iteration primitives that consume the Wave A
:class:`~orpheus.numerics.operator.LinearOperator` Protocol and operate on
the resolvent operand :math:`A`, its lagged gains, and — at the
eigenvalue layer — the fission operator :math:`F`:

* :class:`SourceIteration` solves the within-group fixed-source
  problem :math:`(A - \sum_i g_i)\,\psi = q_{\rm ext}` by a
  fixed-point iteration

  .. math::

      \psi_{n+1} \;=\; A^{-1}\,\Bigl(\sum_i g_i\,\psi_n + q_{\rm ext}\Bigr).

  The convergence rate is bounded by the spectral radius
  :math:`\rho(A^{-1}\sum_i g_i) \le \max_{\rm cell}\,\Sigma_s/\Sigma_t`
  for the SN binding.  Trefethen & Bau §3.2.

* :class:`KEigenvalue` poses the k-eigenvalue problem
  :math:`(A - S)\,\psi = F\,\psi/k` from its operator triple and
  **delegates the outer loop** to the canonical
  :func:`~orpheus.numerics.eigenvalue.power_iteration` algorithm.  It
  realizes the method-agnostic
  :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` boundary
  (``compute_fission_source`` :math:`= F\psi/k`; ``solve_fixed_source``
  = the inner :class:`SourceIteration` realization of
  :math:`(A - S)^{-1}`, warm-started; the keff / production
  estimators), so there is ONE power-iteration loop in the codebase
  (Cardinal Rule 2).  Convergence is governed by the dominance ratio
  :math:`|k_1/k_0|`.

The primitives are deliberately kept **shape-agnostic**.  They make
no assumption about the rank or layout of :math:`\psi` — only that
the operator triple acts linearly on it and that
:func:`numpy.linalg.norm` returns a scalar that orders by relative
size.  The L0 synthetic tests use 4×4 dense matrices acting on
``(4,)`` flat vectors; the L1 SN gate uses
:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator` (the
composite ``A = L + C``),
:class:`~orpheus.transport.operators.scattering.ScatteringOperator`, and
:class:`~orpheus.transport.operators.fission.FissionOperator` acting on
:class:`~orpheus.transport.timed_full_field.TimedFullField` composite
carriers.

The driver APPLIES the inverse operator; a preconditioner is a different concept
=================================================================================

#226 taxonomy step 3 (2026-07-01, superseding the R-1 Step B contract
whose then-``L``-named operand carried a ``.solve`` surface) — the
SOLVER layer builds the inverse operator once (``A.inverse()``: a
:class:`~orpheus.sn.operators.sweep_operator.SweepOperator` for the
schedule-triangular SN family, an
:class:`~orpheus.numerics.operator.InverseOperator` for value-bearing
leaves, or the windowed product ``P @ A.inverse()``), and the iteration
primitives APPLY it:

* :class:`SourceIteration` receives the inverse-application operator
  ``A_inv`` directly and calls ``A_inv.apply(rhs, initial_guess=psi_prev)``
  each step — the inverse family's canonical seeded-apply signature
  (:class:`SupportsSeededApply`).  There is NO per-type signature probe:
  the former ``inspect.signature`` dispatch on the operand's ``solve``
  retired with the duck-typed resolvents (every family member accepts
  the keyword; members with no use for a start accept and ignore it,
  documented per type).

* :class:`KrylovAcceleration` takes an explicit ``preconditioner``
  Callable hook (a GMRES left preconditioner approximating the FULL
  system inverse is a different concept from the inverse step — the
  previous ``inverter`` name was a category mistake).  Default
  behaviour: if ``A`` is invertible
  (:attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`), use
  ``A.inverse().apply`` as the preconditioner; otherwise run
  unpreconditioned.

Carlson seed threading
======================

The previous iterate travels to the inverse application as the EXPLICIT
``initial_guess`` keyword each step — the load-bearing plumbing for
curvilinear sweeps, where the previous iterate IS the Morel–Montry
Carlson coupled-pole seed (the sweep reads the level's :math:`\psi` at
:math:`\mu = -1` from it; lesson L21).  A dropped / zeroed / stale seed
is a WRONG-FIXED-POINT bug there, not a rate change.  The fast always-on
catcher is the Mode-11 path-spy
``tests/sn/solve/test_seed_threading_spy.py`` (route-invariant across the
step-3 rewire); the value catcher is the ``@slow`` het-2G sphere
SI≡Krylov equivalence gate.  Bare-ndarray rhs paths ignore the seed
downstream (the synthetic L0 tests have no seed dependency).

Forward references
==================

* :func:`orpheus.numerics.eigenvalue.power_iteration` — the
  **canonical** power-method algorithm over the method-agnostic
  :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` boundary.
  :class:`KEigenvalue` is the operator-triple realization of that
  boundary and delegates its loop there (one engine); the 5 solver
  families (SN, CP, diffusion, MoC, homogeneous) satisfy the same
  boundary directly.  The full layered architecture (leaves → posing
  → resolvent → algorithm; the generalized eigenproblem
  :math:`A_{\rm loss}\psi = \lambda M\psi`) is documented at
  :ref:`eigenvalue-posing`.
* :class:`~orpheus.sn.solver.SNSolver` — consumes
  :class:`SourceIteration` / :class:`KrylovAcceleration` for its
  within-group resolvent and satisfies the ``EigenvalueSolver``
  boundary directly (its eigenvalue outer IS
  :func:`~orpheus.numerics.eigenvalue.power_iteration`).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Protocol,
    TypeGuard,
    cast,
)

import numpy as np
import scipy.sparse.linalg as spla

if TYPE_CHECKING:
    # Annotation only — the runtime import stays local inside ``solve`` so
    # the "eigenvalue.py does not import iteration.py" acyclicity note there
    # keeps holding in both directions.
    from .eigenvalue import PowerIterationOutcome

from .convergence import (
    ConvergenceWarning,
    IterationBudget,
    IterationRecord,
    StoppingCriterion,
    resolve_iteration_budget,
)
from .operator import (
    InverseWrapMixin,
    LinearOperator,
    NotInvertible,
    invertible,
)
from .vector import V


__all__ = [
    "SourceIteration",
    "SupportsSeededApply",
    "KrylovAcceleration",
    "KEigenvalue",
    "fixed_point_step",
    "lagged_source",
    "seeded_inverse",
]


class SupportsSeededApply(Protocol[V]):
    r"""Static contract: the inverse-application operator an iterative
    driver steps through — the inverse family's CANONICAL apply signature
    (#226 taxonomy step 3).

    ``apply(rhs, *, initial_guess=None)``: the driver threads the previous
    iterate as ``initial_guess`` on EVERY step, uniformly — no per-type
    signature probes.  Members with no use for a starting point accept and
    ignore the keyword, documented per type (an exact
    :class:`~orpheus.numerics.operator.InverseOperator`; the windowed
    product, whose multi-D walk has no bulk-seed consumer).  The members
    that CONSUME it are the curvilinear
    :class:`~orpheus.sn.operators.sweep_operator.SweepOperator`, whose
    Morel–Montry Carlson closure reads the previous iterate at
    :math:`\mu = -1`, and the
    :class:`~orpheus.numerics.green_operator.GreenOperator`, which seeds
    its splitting iteration's start.

    Since taxonomy §12 step 4 the signature is STRUCTURAL on the
    wrap-delegate family: the abstract
    :meth:`~orpheus.numerics.operator.InverseWrapMixin.apply` declares it,
    so a new inverse sibling cannot forget the keyword (#285).

    Like :class:`~orpheus.numerics.operator.SupportsInverse`, this is a
    STATIC contract only (an annotation target) — deliberately not
    ``runtime_checkable``.
    """

    def apply(self, rhs: V, /, *, initial_guess: V | None = None) -> V: ...


# Type alias for the GMRES left-preconditioner hook M ≈ A⁻¹.
Preconditioner = Callable[[np.ndarray], np.ndarray]


class _SeededExactApply:
    r"""Adapt an ALGEBRA-CLOSED inverse to the driver's seeded contract.

    The two-kinds split (taxonomy §12 step 5, canonical statement on
    :meth:`~orpheus.numerics.operator.OperatorProduct.inverse`): the
    wrap-delegate family carries the canonical seeded ``apply``
    structurally, but an algebra-closed inverse — a permutation's inverse
    IS a permutation, the identity is self-inverse, a scaled operator's
    is a scaled operator — is a first-class FORWARD whose ``apply`` is
    the plain positional signature.  Placed in a seeded slot (the Green
    splitting's preconditioner, a warm-started Richardson step) it must
    accept the driver's uniformly-threaded ``initial_guess`` — and these
    inverses are EXACT, so the seed carries no information: accept and
    drop, the same seed-independence the M-direct invariant pins on
    :class:`~orpheus.numerics.matrix_inverse_operator.MatrixInverseOperator`.
    """

    def __init__(self, exact_inverse: LinearOperator) -> None:
        self._exact_inverse = exact_inverse

    def apply(self, rhs: Any, /, *, initial_guess: Any | None = None) -> Any:
        del initial_guess  # exact algebra-closed inverse — nothing to seed
        return self._exact_inverse.apply(rhs)


def _wrap_delegate_member(inv: "LinearOperator") -> "TypeGuard[SupportsSeededApply[Any]]":
    r"""Checked bridge: family membership ⟹ the canonical seeded ``apply``.

    :class:`~orpheus.numerics.operator.InverseWrapMixin`'s abstract
    ``apply`` declares the seeded keyword, so membership STRUCTURALLY
    implies :class:`SupportsSeededApply` conformance (#285) — the
    ``isinstance`` here is type-as-structure dispatch on the two-kinds
    split, not a signature probe. ``TypeGuard`` (replace-not-intersect)
    hands the branch the contract type directly.
    """
    return isinstance(inv, InverseWrapMixin)


def seeded_inverse(A: "LinearOperator") -> "SupportsSeededApply[Any]":
    r"""Build ``A.inverse()`` conforming to the driver's seeded contract —
    the ONE home of the inverse→driver adaptation.  PUBLIC since #276 A4:
    the third consumer (``solve_sn_adjoint_fixed_source``'s daggered
    ``SourceIteration``) joined :class:`KEigenvalue` and the
    :class:`~orpheus.numerics.green_operator.GreenOperator` builder, which
    had already been importing the underscore name cross-module.

    Two kinds of inverse arrive here (taxonomy §12 step 5), keyed by the
    STRUCTURAL family membership, never a signature probe:

    * **wrap-delegate members** (:class:`~orpheus.numerics.operator.InverseWrapMixin`
      siblings — ``SweepOperator``/``InverseOperator``/``GreenOperator``/
      ``MatrixInverseOperator``): the abstract mixin ``apply`` declares
      the seeded keyword, so conformance holds by construction (#285) —
      returned as-is;
    * **algebra-closed inverses** (identity → itself, permutation →
      permutation, scaled → scaled, tensor-product → factor-wise): plain
      forwards with no seed slot — wrapped in :class:`_SeededExactApply`
      (accept-and-drop; an exact inverse has nothing to seed).

    The :func:`~orpheus.numerics.operator.invertible` guard is the
    checked narrowing bridge (Design C): the runtime predicate check IS
    the static permission for the ``.inverse()`` call — no cast.
    """
    if not invertible(A):
        raise NotInvertible(
            f"seeded_inverse requires an invertible operator; "
            f"{type(A).__name__}.is_invertible is False."
        )
    inv = A.inverse()
    if _wrap_delegate_member(inv):
        return inv  # canonical seeded apply, structurally (#285)
    return _SeededExactApply(inv)


# ───────────────────────────────────────────────────────────────────────
# Ravellable protocol — bridges typed flux containers to scipy's
# flat-vector requirement.  :class:`KrylovAcceleration` and
# :class:`SourceIteration` accept typed flux containers
# (:class:`~orpheus.transport.timed_full_field.TimedFullField`) via
# duck-typing on the pair of methods (``to_flat()`` instance method +
# class-level ``from_flat(flat, template)`` factory).
#
# Keeping the protocol duck-typed here (not an ABC import) preserves
# the deliberate decoupling of ``orpheus.numerics`` from
# ``orpheus.transport`` — the iteration primitives still know nothing
# about transport-specific shapes; they just consume any object that
# advertises the ravel pair.
#
# Bare ndarrays match neither protocol and fall through to the
# numpy reshape path in the helpers below.
# ───────────────────────────────────────────────────────────────────────


def _is_ravellable(x: object) -> bool:
    """Detect the ravellable protocol (template-based).

    Matches any object exposing ``to_flat()`` instance method +
    ``from_flat(flat, template)`` classmethod.  The canonical instance
    is :class:`~orpheus.transport.timed_full_field.TimedFullField`.
    """
    return (
        hasattr(x, "to_flat")
        and hasattr(type(x), "from_flat")
    )


def _ravel(x):
    """Ravel typed flux or bare ndarray to a 1-D ``float64`` ndarray."""
    if _is_ravellable(x):
        return np.asarray(x.to_flat(), dtype=float)
    return np.asarray(x, dtype=float).ravel()


def _unravel_like(template, flat: np.ndarray):
    """Reconstruct the typed flux (or reshape the bare ndarray) from ``flat``.

    Uses ``template`` only to recover the shape / mesh / factory —
    ``flat`` is the new numeric content.
    """
    if _is_ravellable(template):
        return type(template).from_flat(flat, template)
    return flat.reshape(template.shape)


class _CarrierMatvecOperator(spla.LinearOperator):
    """Flat scipy face of a carrier-space matvec (see :func:`_as_scipy_linop`).

    Subclassing is scipy's recommended construction API (define ``_matvec``,
    call ``super().__init__(dtype, shape)``) — the ``LinearOperator(shape,
    matvec=...)`` factory form is a runtime-only ``__new__`` dispatch the
    stubs don't model.
    """

    def __init__(self, carrier_matvec, template, n: int) -> None:
        super().__init__(dtype=float, shape=(n, n))
        self._carrier_matvec = carrier_matvec
        self._template = template

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return _ravel(self._carrier_matvec(_unravel_like(self._template, x)))


def _as_scipy_linop(carrier_matvec, template, n: int) -> spla.LinearOperator:
    """The single ORPHEUS↔scipy Krylov boundary.

    Wrap a carrier-space matvec ``carrier_matvec: V -> V`` as a flat
    ``scipy.sparse.linalg.LinearOperator`` of shape ``(n, n)``.  The flat
    vector scipy hands in is lifted to the typed carrier via ``template``
    (the ravellable protocol, :func:`_unravel_like`), the carrier-space
    matvec runs, and the result is ravelled back to 1-D (:func:`_ravel`).
    For a bare-ndarray ``template`` the lift/ravel reduce to a reshape, so
    the L0 path is unchanged.

    This is the SOLE site that constructs a scipy ``LinearOperator`` for the
    Krylov accelerator; both the system matvec and the preconditioner route
    through it (single source of truth, Cardinal Rule 2).
    """
    return _CarrierMatvecOperator(carrier_matvec, template, n)


def _zeros_like(template):
    """Zero typed flux or bare ndarray matching ``template``'s shape/mesh."""
    if _is_ravellable(template):
        flat_size = template.to_flat().size
        return type(template).from_flat(
            np.zeros(flat_size, dtype=float), template,
        )
    return np.zeros_like(template)


def _l2_norm(x) -> float:
    """L2 norm — ravels typed flux via the protocol before delegating to numpy."""
    if _is_ravellable(x):
        return float(np.linalg.norm(x.to_flat()))
    return float(np.linalg.norm(np.asarray(x)))


# The iterate-increment diagnostic leaf is the carrier's OWN answer
# (campaign 1 CS3-R): every field type exposes ``principal_bulk_leaf`` —
# a leaf returns itself, a two-block composite its ``interior``, a coupled
# block vector its first system's — so this layer reads one duck attribute
# and knows nothing of any carrier's anatomy (a bare ndarray has none →
# no diagnostics, as before). The norm CONVENTION (interior-leaf space
# ``l2``, not the whole-composite flat norm) lives with its chooser,
# ``Composite.principal_bulk_leaf``, and is pinned by
# :mod:`tests.numerics.test_si_diagnostic_trajectory`.


# ───────────────────────────────────────────────────────────────────────
# The splitting iteration's MAP, named once — ``G(ψ) = M⁻¹(q + N·ψ)``
# ───────────────────────────────────────────────────────────────────────


def lagged_source(
    q_ext: V, gains: "Sequence[LinearOperator]", psi: V,
) -> V:
    r"""The right-hand side of one fixed-point step — :math:`q + \sum_i N_i\,\psi`.

    The external source plus every lagged coupling evaluated on the given
    iterate: the ONE spelling of *the source the splitting iteration sees*
    (Hackbusch 2016 §11 — ``A = M − N``, the drivers iterate
    ``ψ ← M⁻¹(q + N·ψ)``).  :meth:`SourceIteration.solve` composes it to
    convergence and reads the increment ``rhs_{n−1} − rhs_n`` as its free
    equation residual — which is WHY the assembly is a separate body from
    the inverse apply: the stop inspects the rhs BEFORE the sweep it would
    trigger, so on a break no sweep is wasted.  :func:`fixed_point_step` is
    this assembly followed by the inverse — the map evaluated once.
    """
    rhs = q_ext
    for g in gains:
        rhs = rhs + g.apply(psi)
    return rhs


def fixed_point_step(
    inverse: "SupportsSeededApply[V]",
    gains: "Sequence[LinearOperator]",
    q_ext: V,
    psi: V,
) -> V:
    r"""One application of the splitting iteration's map
    :math:`G(\psi) = M^{-1}\,(q + N\,\psi)` — the ``M⁻¹`` apply seeded by the
    iterate it is evaluated at (the canonical seeded-apply signature; the
    curvilinear ψ½ seed and the reflective partner trace travel through
    ``initial_guess``).

    At a fixed point :math:`G(\psi^*) = \psi^*`, so applied ONCE to a
    converged iterate it RECONSTRUCTS that iterate through whatever
    representative of ``M⁻¹`` it is handed — which need not be the one the
    iteration ran.  The SN finalizes use exactly that freedom: after a
    moment-windowed inner solve they hand it the UN-windowed full-angular
    sweep, so a moment iterate comes back as per-ordinate ψ; after a power
    iteration they hand it the CONVERGED eigenvalue's fission source, so
    the returned flux solves the equation the caller is told about (k, φ)
    to the inverse's own residual.  This is the ONE spelling of
    *reconstruct from a converged iterate*.  A finalize that hand-rolls the
    source instead is a twin path, and the twin drifts: until #448 the
    eigenvalue finalize rebuilt fission + P0 scattering + P0 (n,2n) by hand
    — `[M]` at ``scattering_order = 2`` on a 421-group slab the returned ψ
    missed the converged iterate by 8.8e-2 and its own moments missed the
    reported φ by 3.4e-2, while this step reproduces the iterate to 1.2e-10.
    """
    return inverse.apply(lagged_source(q_ext, gains, psi), initial_guess=psi)


# ───────────────────────────────────────────────────────────────────────
# SourceIteration
# ───────────────────────────────────────────────────────────────────────


class SourceIteration(Generic[V]):
    r"""Fixed-point iteration for :math:`\bigl(A - \sum_i g_i\bigr)\,\psi =
    q_{\rm ext}`.

    Solves the loss equation in its operator-algebra form: the invertible
    loss operator :math:`A` minus a sum of lagged coupling operators
    :math:`g_i`.  Each iteration applies :math:`A^{-1}` to a
    right-hand-side built from the current iterate's couplings plus the
    external source:

    .. math::

        \psi_{n+1} \;=\; A^{-1}\Bigl(\sum_i g_i\,\psi_n + q_{\rm ext}\Bigr).

    The driver is problem-type-AGNOSTIC — it sees only the inverse
    application and a homogeneous bag of couplings.  WHICH operators are
    couplings is a posing-layer decision (see :ref:`eigenvalue-posing`):
    SN within-group passes the scattering ``S`` and the boundary
    reflection ``B`` (the within-group fission is zero, entering via
    ``q_ext``); a scattering-only synthetic problem passes a single gain.

    The :math:`A^{-1}` action arrives AS AN OPERATOR (#226 taxonomy
    step 3): the solver layer builds it once — ``(L+C).inverse()`` /
    ``M.inverse()`` (a
    :class:`~orpheus.sn.operators.sweep_operator.SweepOperator` whose
    ``apply`` IS the WDD sweep), the windowed product ``P @ A.inverse()``,
    or a leaf :class:`~orpheus.numerics.operator.InverseOperator` — and
    this driver only ever calls
    ``A_inv.apply(rhs, initial_guess=psi_prev)``, the family's canonical
    seeded-apply signature (:class:`SupportsSeededApply`).  The former
    duck-typed resolvents (a ``solve`` that inverted a DIFFERENT operator
    than ``apply`` applied) dissolved into these typed objects.

    Convergence test — **the ρ-honest equation residual** (step 5,
    R-5.2/R-5.3; supersedes the historical iterate-increment norm
    ``‖Δψ‖/‖ψ‖``, a DELIBERATE re-interpretation of ``tol``):

    .. math::

        r_{n} \;=\; A\,\psi_{n} - q_{\rm ext}
              \;=\; \mathrm{rhs}_{n-1} - \mathrm{rhs}_{n}
              \;=\; \textstyle\sum_i g_i\,(\psi_{n-1} - \psi_{n}),
        \qquad
        {\rm res}_n = \frac{\lVert r_n\rVert_2}
                           {\max(\lVert q_{\rm ext}\rVert_2, 10^{-30})}

    and the iteration breaks when :math:`{\rm res}_n < {\rm tol}`. The
    FREE-IDENTITY spelling (``rhs_{n-1} − rhs_n``, retained from the
    loop's own bookkeeping — zero marginal cost) is EXACT when the step
    operator is an exact inverse of the splitting's ``M``: then
    ``M ψ_n = rhs_{n-1}`` and ``A ψ_n − q = rhs_{n-1} − N ψ_n − q =
    N(ψ_{n-1} − ψ_n)``. Why the residual and not ``‖Δψ‖``: the increment
    understates the true error by ``1/(1 − ρ)`` (Banach) AND is blind to
    an iteration whose fixed point does not solve the EQUATION (the #282
    lag-death class); the residual claim is contraction-rate-independent.
    The identity itself assumes exact-``M`` — the hole the driver-level
    end-of-solve CERTIFICATE closes (one honest ``evaluate_residual``
    per solve, ``orpheus.sn.solver._certify_within_group_exit``).
    Normalization is EQUATION-relative (``‖q_ext‖``, a source is the
    residual's natural scale); ``q_ext ≈ 0`` degrades to absolute via
    the guard (a zero source has the zero solution — a zero cold start
    exits at the first comparison with ``res = 0`` exactly). For a
    windowed MOMENT-space iterate the same spelling stops on the
    moment-equation increment class (the coisometry ``M`` is not an
    exact inverse there, so the "≡ true residual" claim is NOT made —
    the r3 exemption; the stop ROLE is retained, one law for every arm).

    Parameters
    ----------
    A_inv : SupportsSeededApply
        The inverse-application operator :math:`A^{-1}` — must expose
        ``apply``; the iteration step is
        ``psi = A_inv.apply(rhs, initial_guess=psi_prev)``.  For SN this
        is ``(L+C).inverse()`` / ``M.inverse()``
        (:class:`~orpheus.sn.operators.sweep_operator.SweepOperator`) or
        the windowed product
        :class:`~orpheus.sn.operators.windowing.WindowedSweep`; the L0
        synthetic case passes ``MatrixOperator(...).inverse()``.  Note an
        apply-only object is acceptable BY DESIGN — the windowed product
        carries no round-trip promise (its ``P`` factor is a coisometry) —
        so "is this a faithful inverse of the intended forward?" is the
        CALLER's obligation, discharged where the operator is built.
    *gains : LinearOperator
        The lagged coupling operators :math:`g_i` — each must expose
        ``apply``.  They are applied to the current iterate and
        summed with ``q_ext`` every step, realising
        :math:`\bigl(A - \sum_i g_i\bigr)\psi = q_{\rm ext}`.  For SN
        within-group these are the scattering ``S`` and the boundary
        reflection ``B`` (Jacobi) or ``B_upper`` (boundary Gauss–Seidel).
        Zero gains solves ``A\,\psi = q_{\rm ext}`` outright.
    max_iter : int, optional
        Maximum fixed-point iterations.  Default ``1000``.
    tol : float, optional
        Convergence tolerance on the relative residual norm.  Default
        ``1e-8``.
    corrector : LinearOperator or None, optional
        A synthetic-acceleration correction operator (consistent DSA,
        issue #2 — e.g.
        :class:`~orpheus.sn.acceleration.dsa.DSACorrection`). When
        present, each iteration becomes the accelerated two-step

        .. math::

            \psi_{n+1/2} = A^{-1}\Bigl(\sum_i g_i\,\psi_n + q\Bigr),
            \qquad
            \psi_{n+1} = \psi_{n+1/2}
                + \mathcal{C}\,(\psi_{n+1/2} - \psi_n),

        the correction consuming the sweep DISPLACEMENT.  ``None``
        (default) is byte-identical to the un-accelerated loop.

        The corrector is **correctness-safe by construction** only when
        its correction vanishes with the increment (a synthetic
        accelerator's defining property — at the fixed point
        :math:`\psi_{n+1/2} = \psi_n` so :math:`\mathcal{C}` receives
        zero): then the corrected iteration shares the un-accelerated
        fixed point exactly (vv-principles Mode 9; gated for DSA by the
        FP-invariance battery D3).  Stop-identity note: with a corrector
        the free identity ``rhs_{n−1} − rhs_n = Σ g_i(ψ_{n−1} − ψ_n)``
        still holds verbatim, but it equals the EQUATION residual of
        ``ψ_n`` only up to :math:`A\,c_n` (the correction's image under
        the loss) — near convergence the correction itself → 0, so the
        stop remains ρ-honest in the accelerated metric, and the
        end-of-solve CERTIFICATE (one honest ``evaluate_residual``)
        closes the gap exactly as it does for the exact-``M`` arm.

    Raises
    ------
    TypeError
        At construction time if ``A_inv``, any gain, or the corrector
        has no callable ``apply`` (the eager composition-time guard,
        carve P4).

    Notes
    -----
    The primitive is shape-agnostic: ``q_ext`` is whatever shape the
    operators consume.  The convergence test uses
    :func:`numpy.linalg.norm` on the flattened arrays.  Both the L0
    synthetic case (flat ``(N,)`` vector) and the L1 SN case
    (structured :class:`~orpheus.transport.fields.angular_flux.AngularFlux`) are
    handled by the same :func:`_l2_norm` call routed through the
    ravellable protocol.

    The previous iterate travels to the inverse application as the
    explicit ``initial_guess`` keyword — the Carlson-seed plumbing for
    curvilinear sweeps (see the module-level "Carlson seed threading"
    section; pinned by the seed-threading spy).
    """

    def __init__(
        self,
        A_inv: SupportsSeededApply[V],
        *gains: LinearOperator[V],
        max_iter: int = 1000,
        tol: float = 1e-8,
        corrector: "LinearOperator | None" = None,
        budget_name: str = "max_iter",
    ) -> None:
        # Apply-guards at construction so a downstream caller never sees
        # a stub failure mid-iteration (Wave A philosophy, kept through
        # carve P4 as an eager callable check — no frozenset read). The
        # step operator arrives pre-inverted, so an apply IS the whole
        # contract (#226 taxonomy step 3).
        if not callable(getattr(A_inv, "apply", None)):
            raise TypeError(
                f"SourceIteration requires 'apply' on A_inv (the "
                f"inverse-application operator — build it via "
                f"A.inverse()); {type(A_inv).__name__} has none."
            )
        for i, g in enumerate(gains):
            if not callable(getattr(g, "apply", None)):
                raise TypeError(
                    f"SourceIteration requires 'apply' on every coupling "
                    f"operator; gain {i} ({type(g).__name__}) has none."
                )
        if corrector is not None and not callable(
            getattr(corrector, "apply", None)
        ):
            raise TypeError(
                f"SourceIteration requires 'apply' on the corrector; "
                f"{type(corrector).__name__} has none."
            )

        self.A_inv = A_inv
        self.gains = gains
        self.corrector = corrector
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        # What the CALLER calls ``max_iter``.  Only the construction site knows
        # whether this loop is somebody's ``max_inner`` or its own thing, and
        # the advice in a ConvergenceWarning has to name a knob the reader can
        # actually type (#340 N6).
        self.budget_name = str(budget_name)

    def solve(
        self,
        q_ext: V,
        initial_guess: V | None = None,
    ) -> "tuple[V, IterationRecord]":
        r"""Run fixed-point iteration to convergence.

        Parameters
        ----------
        q_ext : np.ndarray
            External source.  Shape determined by what the operator
            triple consumes — for SN this is ``(ng, nx, ny)``
            (principled storage; see :ref:`theory-sn-index-convention`);
            for the L0 synthetic case it is a flat ``(N,)`` vector.
        initial_guess : np.ndarray or None, optional
            Initial iterate.  When ``None`` (default), the iteration
            starts from :func:`np.zeros_like` of ``q_ext``.

        Returns
        -------
        psi : np.ndarray
            Converged iterate (or the final iterate if ``max_iter``
            was hit before tolerance was reached).
        record : IterationRecord
            What this level wanted, what it got, and why it stopped —
            the relative-residual trajectory, the iterate-increment norm
            trajectory (from which ``contraction_ratios`` and the c→1
            ``true_error_estimate`` derive — #208, relocated at CS3), the
            budget, and the loop's OWN pass count, from which ``converged``
            / ``truncated`` / ``rate`` / ``projected_iterations`` all
            derive.

            ⚠ ``record.n_iterations`` is NOT the trajectory length here.
            The stop compares SUCCESSIVE iterates, so ``P`` passes yield
            ``P - 1`` residuals; the record carries the pass count because
            only this loop knows that offset (#340 F10).

            ⛔ Returned a bare ``list[float]`` until 2026-08-09.  Every
            consumer then re-derived convergence from it — five sites, one
            of which read an EMPTY history as "not converged" when it means
            "returned on the initial guess".  The fact travels with the
            answer now, so there is nothing left to re-derive.
        """
        # R-1 Step 4a — ravellable protocol: typed flux containers
        # (:class:`AngularFlux`) and bare ndarrays both work.  ``psi``
        # carries the same type as ``q_ext``; arithmetic, norm, and
        # zeros routing through the protocol helpers.
        if initial_guess is None:
            psi = _zeros_like(q_ext)
        elif _is_ravellable(initial_guess):
            psi = initial_guess  # typed: trust frozen-arithmetic contract
        else:
            # The bare-ndarray L0 arm — the ravellable protocol blesses
            # ndarrays through the same V slots (module intro); the cast
            # states that here, where the union would otherwise leak.
            psi = cast(V, np.asarray(initial_guess).copy())
        residual_history: list[float] = []
        # Iterate-increment diagnostics (#208, relocated onto the record at
        # CS3): the space norm of Δψ's principal bulk leaf, one entry per
        # typed pass; the record derives ρ and the c→1 geometric-tail
        # estimate from these. Additive — NOT in the convergence path (the
        # stop is the equation residual below). O(1) memory: only floats
        # survive a pass.
        increment_norms: list[float] = []

        # The ρ-honest stop's fixed scale (step 5): the equation residual
        # is measured against the SOURCE's norm — computed once; the guard
        # degrades q ≈ 0 to an absolute test (a zero source has the zero
        # solution, caught at the first comparison).
        q_norm = max(_l2_norm(q_ext), 1e-30)
        rhs_prev = None
        # The loop's OWN pass count.  It is NOT ``len(residual_history)``:
        # the stop measures the DIFFERENCE between successive iterates, so
        # P passes yield P-1 residuals and an exhausted ``max_iter=50`` run
        # records 49.  Only this loop knows that offset, which is why the
        # record takes it rather than inferring it (#340 F10/F11 — the same
        # `n_inner` field used to be written with both conventions, one of
        # them an undocumented `+1`).
        iterations_run = 0

        for _ in range(self.max_iter):
            psi_prev = psi
            iterations_run += 1

            # The RHS of the fixed-point step: the external source plus
            # every lagged coupling ``g·ψ_n`` (for SN within-group:
            # ``S·ψ + N₂ₙ·ψ + B·ψ``) — the ONE body :func:`lagged_source`,
            # shared with :func:`fixed_point_step` (the map the SN finalizes
            # evaluate once to reconstruct from a converged iterate).  The
            # gains are LinearOperators; their ``.apply`` contracts are the
            # only thing this loop touches.
            rhs = lagged_source(q_ext, self.gains, psi)

            # ── the ρ-honest STOP (step 5): the equation residual of the
            # PREVIOUS iterate via the free identity r_n = rhs_{n−1} − rhs_n
            # (= Σ g_i (ψ_{n−1} − ψ_n) = A ψ_n − q under exact-M; see the
            # class docstring). Checked BEFORE the next inverse apply — on
            # a break, ``psi`` (the last applied iterate) is the converged
            # state and no sweep is wasted. Zero marginal cost: the rhs is
            # the loop's own bookkeeping.
            if rhs_prev is not None:
                res = _l2_norm(rhs_prev - rhs) / q_norm
                residual_history.append(res)
                if res < self.tol:
                    break
            rhs_prev = rhs

            # Apply the inverse OPERATOR.  The curvilinear Carlson
            # coupled-pole seed and the reflective-BC partner-flux trace
            # travel through ``initial_guess`` explicitly (the M-M closure
            # reads the level's ψ at μ = −1 from this argument) — the
            # canonical seeded-apply signature, threaded UNCONDITIONALLY
            # (pinned by the seed-threading spy).  Members with no seed
            # consumer ignore the kwarg downstream.
            psi = self.A_inv.apply(rhs, initial_guess=psi_prev)

            # The synthetic-acceleration correction (issue #2): the
            # corrector consumes the SWEEP increment ψ_{n+1/2} − ψ_n
            # and returns the additive correction (→ 0 at the fixed
            # point, so the un-accelerated FP is preserved — see the
            # class docstring's corrector entry). Absent corrector ⇒
            # this block is byte-inert.
            if self.corrector is not None:
                psi = psi + self.corrector.apply(psi - psi_prev)

            # The iterate increment Δψ = ψ⁽ⁱ⁾ − ψ⁽ⁱ⁻¹⁾ — DIAGNOSTICS only
            # (the record derives ρ ≈ ‖Δψ⁽ⁱ⁾‖/‖Δψ⁽ⁱ⁻¹⁾‖ and the c→1
            # geometric-tail estimate from the norm trajectory) — the STOP
            # rides the residual above. Bare-ndarray (L0) iterates record
            # nothing, as before.
            leaf = getattr(psi - psi_prev, "principal_bulk_leaf", None)
            if leaf is not None:
                increment_norms.append(leaf.l2)

        return psi, IterationRecord(
            label="inner(source-iteration)",
            criteria=(
                StoppingCriterion(
                    name="residual",
                    trajectory=tuple(residual_history),
                    tolerance=self.tol,
                ),
            ),
            increment_norms=tuple(increment_norms),
            # No conversion: one unit of ``max_iter`` IS one fixed-point
            # pass, so the identity ``iterations_per_unit`` is the honest
            # statement and the comparison against the trajectory is sound.
            budget=IterationBudget(self.max_iter, self.budget_name),
            iterations_run=iterations_run,
        )


# ───────────────────────────────────────────────────────────────────────
# KrylovAcceleration
# ───────────────────────────────────────────────────────────────────────


class KrylovAcceleration(Generic[V]):
    r"""GMRES on :math:`\bigl(A - \sum_i g_i\bigr)\,\psi = q_{\rm ext}` —
    sibling of :class:`SourceIteration` for the same algebra.

    Both primitives solve the same loss equation; they differ in
    algorithm.  :class:`SourceIteration` lags :math:`\sum_i g_i\,\psi` as
    the right-hand side and inverts :math:`A` at every step (geometric
    convergence at rate :math:`\rho(A^{-1}\sum_i g_i) \le
    \max\Sigma_s/\Sigma_t`).  :class:`KrylovAcceleration` builds the
    composed matvec :math:`\bigl(A - \sum_i g_i\bigr)\cdot` as a single
    linear operator and solves it with GMRES, optionally preconditioned
    by :math:`A^{-1}` (the sweep).  When the scattering ratio :math:`c =
    \Sigma_s/\Sigma_t` approaches 1, GMRES converges in
    :math:`\mathcal{O}(\sqrt{\kappa})` matvecs vs source iteration's
    :math:`\mathcal{O}(1/(1-c))` — the standard transport-Krylov win
    documented in Adams & Larsen 2002 (the SAILOR / preconditioned-
    Krylov framework).

    Algebra-of-record:

    .. math::

        \Bigl(A - \sum_i g_i\Bigr)\,\psi \;=\; q_{\rm ext}.

    The composed matvec is realised as ``A.apply(psi) - Σ gᵢ.apply(psi)``
    per call — no intermediate :class:`OperatorSum` allocation.  For SN
    within-group the gains are the scattering ``S`` and the boundary
    reflection ``B``, so the matvec IS the honest ``(L+C − S − B)·ψ``
    (with ``A = L + C``).  The right-hand side is whatever shape the
    operators consume; scipy GMRES requires a flat 1-D view internally,
    so the primitive ravels at the boundary and reshapes the solution
    back to ``q_ext.shape`` on return.

    The ``preconditioner`` parameter
    ================================

    The GMRES PRECONDITIONER approximates the FULL within-group system
    inverse, :math:`M \approx \bigl(A - \sum_i g_i\bigr)^{-1}` (for SN
    within-group, :math:`(L+C-S-B)^{-1}`).  The natural choice for
    transport problems is :math:`M = A^{-1}` (the sweep) — this is the
    "transport-corrected" preconditioner from Adams & Larsen 2002 §III.
    When :math:`c` is small, the sweep is an excellent preconditioner;
    when :math:`c` is near unity, the sweep is diffusion-like and GMRES
    needs more iterations.  This left preconditioner (an approximation to
    the FULL inverse :math:`\bigl(A - \sum_i g_i\bigr)^{-1}`) is a
    DIFFERENT object from :class:`SourceIteration`'s exact inverse of the
    single operand :math:`A` — hence the ``preconditioner`` name, not
    ``inverter`` (which would conflate the GMRES left preconditioner with
    the iteration's inverse step).

    * ``preconditioner = None`` (default): if ``A`` is invertible
      (:attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`),
      use ``A.inverse().apply`` as the preconditioner; otherwise, no
      preconditioner (identity ``M = I``).
    * ``preconditioner = lambda q: sweep_preconditioner(q)``:
      caller-supplied preconditioner.  Typically wraps a sweep that
      consumes the same packed/structured layout the operators
      consume.

    Parameters
    ----------
    A : LinearOperator
        The FORWARD loss operator — the GMRES matvec applies it, so it
        must expose ``apply`` (contrast
        :class:`SourceIteration`, which consumes the pre-built INVERSE).
        If ``A.is_invertible`` and no ``preconditioner`` is supplied,
        ``A.inverse().apply`` (the sweep) becomes the default GMRES
        preconditioner.
    *gains : LinearOperator
        The coupling operators :math:`g_i` subtracted from ``A`` in the
        matvec (each must expose ``apply``).  For SN within-
        group these are the scattering ``S`` and the boundary reflection
        ``B`` (within-group fission is zero — it enters as the EXTERNAL
        :math:`q_{\rm ext}` per the eigenvalue outer / within-group
        decomposition).  Zero gains solves ``A\,\psi = q_{\rm ext}``.
    preconditioner : callable or None, optional
        GMRES left preconditioner.  See above.  When ``None`` and
        ``A`` is not invertible, runs GMRES without preconditioner.
    max_iter : int, optional
        Maximum GMRES **restart cycles** (``maxiter`` in scipy).  Default
        ``1000``.

        ⛔ It read *"Maximum GMRES iterations"* until 2026-08-13, and that
        one missing word is the likeliest origin of #349: scipy runs an
        OUTER loop of ``maxiter`` restart cycles, each an INNER Arnoldi loop
        of up to ``restart`` steps, so this bounds ITERATIONS only when
        ``restart == 1``.  The two are related by ``restart`` exactly —
        `[M]` 2026-08-13, scipy 1.17.1, 12 of 12 non-converging rows gave
        ``callbacks == maxiter * restart``.  The record states that exchange
        rate on its
        :class:`~orpheus.numerics.convergence.IterationBudget` so the
        comparison it feeds is dimensionally sound.
    tol : float, optional
        GMRES relative residual tolerance (``rtol`` in scipy).
        Default ``1e-8``.
    restart : int, optional
        GMRES restart length — the Arnoldi steps ONE ``max_iter`` unit
        buys.  Default ``50``.  Clamped to ``n`` at :meth:`solve` time, and
        the SN caller passes ``n_dof`` (``sn/solver.py:721``, the ERR-053
        fix), so on that path one cycle admits the FULL problem dimension
        and ``max_iter`` effectively never binds.

    Raises
    ------
    TypeError
        At construction time if ``A`` or any gain has no callable
        ``apply`` (the eager composition-time guard, carve P4).

    Notes
    -----
    The primitive is shape-agnostic at the operator level — it only
    requires that ``A`` and the gains all consume and return arrays of
    the same shape as ``q_ext``.  Internally it ravels to 1-D for
    scipy's GMRES requirement and reshapes the solution to
    ``q_ext.shape`` on return.
    """

    def __init__(
        self,
        A: LinearOperator[V],
        *gains: LinearOperator[V],
        preconditioner: Preconditioner | None = None,
        max_iter: int = 1000,
        tol: float = 1e-8,
        restart: int = 50,
        budget_name: str = "max_iter",
    ) -> None:
        if not callable(getattr(A, "apply", None)):
            raise TypeError(
                f"KrylovAcceleration requires 'apply' on A; "
                f"{type(A).__name__} has none."
            )
        for i, g in enumerate(gains):
            if not callable(getattr(g, "apply", None)):
                raise TypeError(
                    f"KrylovAcceleration requires 'apply' on every "
                    f"coupling operator; gain {i} ({type(g).__name__}) "
                    f"has none."
                )

        self.A = A
        self.gains = gains
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.restart = int(restart)
        # See SourceIteration.__init__ — the caller's name for ``max_iter``,
        # so the ConvergenceWarning names a knob the reader can type (#340 N6).
        self.budget_name = str(budget_name)

        # Pin the preconditioner choice at construction.  If caller
        # supplied one, use it.  Otherwise, fall back to applying A's
        # inverse OPERATOR when A is invertible (the runtime,
        # instance-accurate query; the narrowing rationale lives on
        # :func:`seeded_inverse`); if not, run GMRES without
        # preconditioner.
        if preconditioner is not None:
            self._preconditioner: Preconditioner | None = preconditioner
        elif A.is_invertible:
            self._preconditioner = seeded_inverse(A).apply
        else:
            self._preconditioner = None

    def solve(
        self,
        q_ext: V,
        initial_guess: V | None = None,
    ) -> "tuple[V, IterationRecord]":
        r"""Run GMRES on :math:`(A - \sum_i g_i)\,\psi = q_{\rm ext}` to convergence.

        Parameters
        ----------
        q_ext : np.ndarray
            External source.  Whatever shape the operator triple
            consumes — ravelled to 1-D internally for scipy GMRES,
            reshaped back to ``q_ext.shape`` on return.
        initial_guess : np.ndarray or None, optional
            GMRES initial iterate.  When ``None`` (default), starts
            from :func:`np.zeros_like` of ``q_ext``.

        Returns
        -------
        psi : np.ndarray
            Converged solution, shape ``q_ext.shape``.
        record : IterationRecord
            What this level wanted, what it got, and why it stopped.  Its
            criterion is the preconditioned residual at every GMRES inner
            iteration (scipy's ``callback_type='pr_norm'``, which is
            RELATIVE to ``‖b‖`` — measured 2026-08-09, so judging it against
            ``rtol`` is dimensionally right).

            An EMPTY trajectory means GMRES returned in zero iterations,
            i.e. the initial guess already satisfied the tolerance.  The
            record reports that as ``iterated=False`` and ``converged=True``
            — two separate questions, because conflating them is what
            turned 44 such rows into phantom truncations in #340's audit.
        """
        # R-1 Step 4a — ravellable protocol: when ``q_ext`` is a typed
        # flux container (:class:`AngularFlux`), the ravel/unravel goes
        # through ``to_flat_with_traces`` / ``from_flat_with_traces`` so
        # the flat vector for scipy carries the FULL state including the
        # boundary face block.  Bare-ndarray inputs route through the
        # legacy reshape path unchanged.
        b = _ravel(q_ext)
        n = b.size

        # B.5.2: the iterate ψ and the returned solution live in the SOLUTION
        # (flux) space, NOT q_ext's source space.  Template their typed
        # reconstruction on a flux ``initial_guess`` when supplied (the SN
        # solver always passes a flux composite); fall back to ``q_ext`` only
        # for callers that pass none (bare-ndarray L0 tests, where domain ==
        # codomain so the type label is irrelevant).  ``b`` / the
        # preconditioner input stay templated on ``q_ext`` (source space).
        solution_template = (
            initial_guess
            if initial_guess is not None and _is_ravellable(initial_guess)
            else q_ext
        )

        def loss_minus_gains(psi: V) -> V:
            # The honest within-group system matvec (A − S − B)·ψ with
            # A = L+C: the invertible loss minus the in-scatter +
            # boundary gains.  Operator arithmetic propagates via
            # dunders to ``.boundary`` (typed AngularFlux) or just the
            # ndarray (bare).
            out = self.A.apply(psi)
            for g in self.gains:
                out = out - g.apply(psi)
            return out

        A_scipy = _as_scipy_linop(loss_minus_gains, solution_template, n)

        M_scipy: spla.LinearOperator | None = (
            _as_scipy_linop(self._preconditioner, q_ext, n)
            if self._preconditioner is not None
            else None
        )

        x0 = (
            _ravel(initial_guess)
            if initial_guess is not None
            else np.zeros_like(b)
        )

        residual_history: list[float] = []

        def callback(rk: object) -> None:
            # scipy GMRES with callback_type='pr_norm' passes the
            # preconditioned-residual norm (a scalar).  Older versions
            # may pass the residual vector — handle both defensively.
            r = np.asarray(rk)
            if r.ndim == 0:
                residual_history.append(float(r))
            else:
                residual_history.append(float(np.linalg.norm(r)))

        # No try/except around the solve: a TypeError raised from inside the
        # wrapped carrier matvec (``loss_minus_gains`` / the preconditioner,
        # via :func:`_as_scipy_linop`) or the callback must surface directly.
        # (A retired scipy<1.14 ``tol=`` fallback arm once lived here; its
        # over-broad ``except TypeError`` masked a B.5.2 matvec cross-class
        # regression as a misleading "tol" error. The ``scipy>=1.14`` floor
        # makes ``rtol`` the only spelling.)
        # The Arnoldi steps ONE ``max_iter`` unit buys, named once and read
        # three times below (the call, the warning, and the record's budget).
        # It is the exchange rate between this method's two currencies, and
        # leaving it as a thrice-spelled ``min(...)`` is what let #349 ship:
        # an un-named conversion is one nobody thinks to apply.
        iterations_per_cycle = min(self.restart, n)

        solution, info = spla.gmres(
            A_scipy, b, x0=x0, M=M_scipy,
            rtol=self.tol, atol=0.0,
            maxiter=self.max_iter,
            restart=iterations_per_cycle,
            callback=callback,
            callback_type='pr_norm',
        )

        # D-H.1e (2026-05-28) — surface GMRES non-convergence.  Pre-fix
        # the ``info`` flag was discarded; an unconverged ``solution``
        # would silently be consumed as if it were the true inverse.
        # Conjunction with the legacy ``restart=min(50, full_size)``
        # clamp at the caller produced the ERR-053 keff drift on
        # curvilinear meshes (the GMRES iteration was structurally
        # truncated, then the failure was hidden by this discard).
        # scipy convention: ``info > 0`` means "not converged within
        # ``maxiter``"; ``info < 0`` means illegal-input.  Both
        # surface as warnings — raising would break long-standing
        # callers that tolerate slow convergence and need the
        # best-effort iterate.  See ERR-053.
        # Exact-breakdown carve-out — a PERMANENT invariant of this
        # boundary, not a special case: a final preconditioned residual of
        # LITERAL 0.0 means the Krylov space collapsed AT the solution
        # (``M⁻¹(b − Ax) = 0`` with a nonsingular preconditioner —
        # identity / exact inverses — implies ``Ax = b`` exactly), yet
        # scipy's breakdown path then stagnates to ``maxiter`` and stamps
        # ``info > 0``.  That is CONVERGENCE, the opposite of the ERR-053
        # restart truncation this warning exists to surface — so it does
        # not warn.  The general trigger is any warm-started solve of a
        # singular-but-consistent system; the first caller to exercise it
        # was the B.2d transitional coupled matvec (dead ψ_A ray padding,
        # gone at the d2 eviction) — that caller motivated the guard, it
        # is not the reason the guard exists.
        exact_breakdown = bool(residual_history) and residual_history[-1] == 0.0
        if info != 0 and not exact_breakdown:
            warnings.warn(
                f"KrylovAcceleration.solve: scipy.sparse.linalg.gmres "
                f"returned info={info} (not converged within "
                f"maxiter={self.max_iter}; restart={iterations_per_cycle}; "
                f"rtol={self.tol}).  Returning best-effort iterate; "
                f"residual_history tail = "
                f"{residual_history[-3:] if residual_history else '[]'}.  "
                f"Tighten ``restart`` to ``n`` (full size) if the Krylov "
                f"subspace is being truncated; see ERR-053.",
                # ⛔ A bare ``RuntimeWarning`` until 2026-08-10 (#340 R3),
                # which put the tree's ONLY non-convergence announcement from
                # inside ``numerics`` outside the escalation net: the published
                # recipe is ``-W error::…convergence.ConvergenceWarning``, and
                # a bare RuntimeWarning does not match it.  A CI run could
                # therefore be configured to make truncation fatal, pass, and
                # still have swallowed this one — the exact
                # "the gate does not cover what it claims" defect #340 exists
                # to remove.  ``ConvergenceWarning`` subclasses
                # ``RuntimeWarning``, so every existing consumer that filters
                # on the base class (the ERR-053 gates match with
                # ``issubclass(w.category, RuntimeWarning)``) is unaffected;
                # the category is strictly narrowed, never widened.
                ConvergenceWarning,
                stacklevel=2,
            )

        return _unravel_like(solution_template, solution), IterationRecord(
            label="inner(gmres)",
            criteria=(
                StoppingCriterion(
                    # scipy's ``callback_type='pr_norm'`` reports the
                    # preconditioned residual RELATIVE to ``‖b‖`` — measured
                    # 2026-08-09 across three ‖b‖ scales, where the callback
                    # value matched ``‖b − Ax‖/‖b‖`` exactly.  So judging it
                    # against ``self.tol`` (an ``rtol``) is dimensionally
                    # right, and the long-standing ``_claims_convergence``
                    # comparison at the SN call sites was never a units bug.
                    name="pr_residual",
                    trajectory=tuple(residual_history),
                    tolerance=self.tol,
                ),
            ),
            # ⭐ The ONE producer in the tree whose knob and trajectory count
            # different things: ``max_iter`` is scipy's restart-CYCLE cap,
            # while the callback fires per inner Arnoldi step.  Stating the
            # exchange rate is what makes ``exhausted_budget`` answerable —
            # before it, a converged 91-callback solve under ``max_inner=5``
            # reported that it had run out (#349).
            budget=IterationBudget(
                self.max_iter,
                self.budget_name,
                iterations_per_unit=iterations_per_cycle,
            ),
            # One callback per Arnoldi step here, so this matches the
            # TRAJECTORY — the opposite convention from SourceIteration
            # above, which is why neither can be inferred by the record.
            # (It says nothing about the BUDGET; that is the budget's job,
            # and reading this line as covering both is exactly the
            # misreading #349 rode in on.)
            iterations_run=len(residual_history),
        )


# ───────────────────────────────────────────────────────────────────────
# KEigenvalue
# ───────────────────────────────────────────────────────────────────────


class KEigenvalue(Generic[V]):
    r"""The k-eigenvalue problem :math:`(A - S)\,\psi = F\psi/k`, posed from an
    operator triple and solved by the canonical ``power_iteration`` loop.

    ``KEigenvalue`` is the **operator-triple realization** of the
    method-agnostic
    :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` boundary (the
    Layer-2 k-posing :math:`A_{\rm loss} = A - S`, :math:`M = F`,
    :math:`k = \mu`): it exposes the boundary methods
    (``compute_fission_source`` :math:`= F\psi/k`; ``solve_fixed_source``
    = the inner :class:`SourceIteration` realization of
    :math:`(A - S)^{-1}`, warm-started; the keff / production estimators;
    ``converged``) and **delegates the outer loop** to
    :func:`~orpheus.numerics.eigenvalue.power_iteration` — the SINGLE
    power-iteration loop in the codebase (Cardinal Rule 2).  It is ONE
    implementer of the boundary alongside the five solver families
    (SN / CP / diffusion / MoC / homogeneous); it owns no parallel loop.  See
    :ref:`eigenvalue-posing` for the full layered architecture.

    Each outer step (run by ``power_iteration``) builds the fission source
    :math:`q_n = F\,\psi_n/k_n`, then drives the inner
    :class:`SourceIteration` with operator triple :math:`(A, S, 0)` and
    external source :math:`q_n`:

    .. math::

        \psi_{n+1} \;=\; (A - S)^{-1}\,F\,\psi_n / k_n

    .. math::

        k_{n+1} \;=\; {\rm keff\_estimator}(A, F, \psi_{n+1})

    Convergence test:  both :math:`|k_{n+1} - k_n| < {\rm keff\_tol}`
    AND the relative flux residual :math:`\|\psi_{n+1} -
    \psi_n\|_2 / \|\psi_{n+1}\|_2 < {\rm flux\_tol}`.

    The convergence rate is governed by the dominance ratio
    :math:`|k_1/k_0|` (Trefethen & Bau §27 power iteration analysis).

    Parameters
    ----------
    A, S, F : LinearOperator
        Operator triple.  ``A`` (the FORWARD invertible loss operator)
        MUST be invertible
        (:attr:`~orpheus.numerics.operator.LinearOperator.is_invertible`)
        — this posing layer builds ``A.inverse()`` once and hands it to
        the inner :class:`SourceIteration`, which only APPLIES it (#226
        taxonomy step 3).  ``S`` and ``F`` must expose
        ``apply``.  ``F`` is non-trivial for an eigenvalue
        solve (no degenerate zero-fission case — without fission the
        spectrum is empty).
    max_outer : int, optional
        Maximum outer (power) iterations.  Default ``500``.
    keff_tol, flux_tol : float, optional
        Outer convergence tolerances.  Defaults match
        :class:`~orpheus.sn.solver.SNSolver` (``1e-7`` / ``1e-6``).
    max_inner : int or None, optional
        Inner :class:`SourceIteration` ``max_iter`` budget.  ``None`` (the
        default) DERIVES it from ``inner_tol`` via
        :func:`~orpheus.numerics.convergence.default_iteration_budget`; an
        explicit int is a deliberate cap.

        ⛔ This was the **sixth** hardcoded budget in the tree and the only
        one outside SN — a third ``(1000, 1e-8)`` combination that no SN
        entry spelled, reachable whenever ``KEigenvalue`` is constructed
        directly.  #340's plan counted five and missed it; a constant that
        does not move with its tolerance cannot be right for both.
    inner_tol : float, optional
        Inner :class:`SourceIteration` ``tol``.  Default ``1e-8``.
    eigenvalue_method : str, optional
        Forward hook for FEAST-style contour-integral methods (Issue
        #163 acceptance criterion).  Currently only ``"power"`` is
        implemented; other values raise :class:`NotImplementedError`
        at construction time.
    Notes
    -----
    The pre-R8 ``keff_estimator`` / ``production_estimator`` injection
    kwargs were retired at #259 P1 (2026-07-03): the estimators are
    hardwired methods (see :meth:`compute_keff` /
    :meth:`compute_production_rate`). At a converged eigenpair every
    estimator CONSISTENT with the posed problem agrees, so injection
    could only introduce an inconsistent functional; the method-layer
    solvers implement the :class:`EigenvalueSolver` protocol directly
    by design and never routed through this class.

    Raises
    ------
    NotInvertible
        At construction when ``A`` is not invertible (this posing layer
        builds ``A.inverse()``).
    TypeError
        Apply-guard conditions as :class:`SourceIteration`.
    NotImplementedError
        If ``eigenvalue_method`` is not ``"power"`` — the FEAST hook
        is reserved for a future wave.
    """

    def __init__(
        self,
        A: LinearOperator,
        S: LinearOperator,
        F: LinearOperator,
        *,
        max_outer: int = 500,
        keff_tol: float = 1e-7,
        flux_tol: float = 1e-6,
        max_inner: int | None = None,
        inner_tol: float = 1e-8,
        eigenvalue_method: str = "power",
    ) -> None:
        if eigenvalue_method != "power":
            raise NotImplementedError(
                f"KEigenvalue currently only supports "
                f"eigenvalue_method='power'; got "
                f"{eigenvalue_method!r}.  FEAST-style contour-integral "
                f"methods are a forward hook reserved for a future wave."
            )

        # Construct-time validation of A happens HERE (the posing layer
        # builds the inverse — taxonomy step 3): a non-invertible A must
        # fail with a domain message, not an AttributeError from a
        # missing ``.inverse``.  S's apply-guard stays deferred to
        # the inner SourceIteration (one source of truth).
        if not A.is_invertible:
            raise NotInvertible(
                f"KEigenvalue requires an INVERTIBLE A — the inner "
                f"SourceIteration applies A.inverse() each step; "
                f"{type(A).__name__}.is_invertible is False."
            )

        self.A = A
        self.S = S
        self.F = F
        self.max_outer = int(max_outer)
        self.keff_tol = float(keff_tol)
        self.flux_tol = float(flux_tol)
        self.inner_tol = float(inner_tol)
        self.max_inner = resolve_iteration_budget(max_inner, self.inner_tol)
        self.eigenvalue_method = eigenvalue_method
        #: Inner records, newest last — one per outer iteration.  Reset by
        #: :meth:`solve` so this accessor reports the current solve.
        #:
        #: ⛔ #340 F12 measured the double-count on the scalar
        #: ``SNSolver._total_inner_iterations``, which had no reset; that
        #: counter was retired onto ``SNSolver.inner_records`` on 2026-08-09.
        #: The reset here is no longer what protects the outer record's
        #: subtree — :func:`~orpheus.numerics.eigenvalue.power_iteration`
        #: slices off only what was appended during its own loop, so no
        #: realizer's reset hygiene can corrupt the tree.  It still keeps
        #: THIS attribute honest for anyone reading it directly.
        self.inner_records: list[IterationRecord] = []

        # Build the inner fixed-source step ONCE: the SOLVER (this posing
        # layer) builds the inverse operator, the driver applies it (#226
        # taxonomy step 3).  Its single coupling gain is the scattering
        # ``S`` — the within-group fission is zero at the inner level
        # because F·ψ/k is the EXTERNAL source the outer power iteration
        # feeds in, NOT a within-group fixed-point term.  Constructing it
        # here validates S's apply at construction time, NEVER
        # mid-iteration.  The seeded-apply narrow (and its SCOPE — the
        # reachable inverses, not the whole family) is single-sourced on
        # :func:`seeded_inverse`; the ``is_invertible`` guard above is
        # its runtime precondition.
        self._inner = SourceIteration(
            seeded_inverse(self.A), self.S,
            max_iter=self.max_inner, tol=self.inner_tol,
            budget_name="max_inner",
        )
        # F (the outer eigen-operator F·ψ) needs apply.
        if not callable(getattr(self.F, "apply", None)):
            raise TypeError(
                f"KEigenvalue requires 'apply' on F (the outer fission "
                f"source F·ψ); {type(self.F).__name__} has none."
            )
        # The initial flux guess is supplied to .solve() and stashed for the
        # EigenvalueSolver.initial_flux_distribution boundary method.
        self._initial_guess: V | None = None

    # ── EigenvalueSolver boundary (consumed by power_iteration) ──────────
    #
    # KEigenvalue realizes the method-agnostic
    # :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` boundary from its
    # (A, S, F) operator triple — the k-eigenvalue posing A_loss = A − S (the
    # invertible loss ``A`` minus the scattering coupling ``S``),
    # eigen-operator M = F, k = μ — then delegates the outer loop to the
    # canonical ``power_iteration`` algorithm.  (The SN production path poses
    # the same standard form via
    # :func:`~orpheus.sn.coupled_system.build_within_group_system`, whose
    # record carries the boundary reflection ``B`` as a coupling gain so,
    # with A = L+C,
    # A_loss = L+C − S − B.)  There is ONE power-iteration loop in the codebase
    # (Cardinal Rule 2): KEigenvalue and SNSolver are both implementers of this
    # boundary, not parallel engines.

    def initial_flux_distribution(self) -> V:
        """Return the caller-supplied initial flux guess (set by :meth:`solve`).

        The :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` boundary
        is consumed by ``power_iteration`` *inside* :meth:`solve`, which
        stashes the guess (after its own None-rejection) before delegating —
        so an unset guess here means the boundary was invoked outside
        :meth:`solve`, a caller error.
        """
        if self._initial_guess is None:
            raise RuntimeError(
                "initial_flux_distribution() reads the guess stashed by "
                "solve(); the EigenvalueSolver boundary is not meaningful "
                "on an unsolved KEigenvalue."
            )
        return self._initial_guess

    def compute_fission_source(
        self, flux_distribution: V, keff: float,
    ) -> V:
        """Outer eigen-source ``F·ψ / k`` (the k-posing's eigen-operator M = F)."""
        return self.F.apply(flux_distribution) / keff

    def solve_fixed_source(
        self, fission_source: V, flux_distribution: V,
    ) -> V:
        r"""Resolvent ``A_loss⁻¹ q`` via the inner :class:`SourceIteration`.

        Warm-started from the previous outer iterate (``flux_distribution``) to
        amortise the inner cost across outer iterations — the same pattern
        :meth:`SNSolver._solve_source_iteration` uses.  The inner solve has a
        single coupling gain ``S`` (zero within-group fission — the eigen-source
        ``F·ψ/k`` is the EXTERNAL ``q``, not a within-group term).

        ⛔ Until 2026-08-09 this method did ``psi, _inner_residuals =
        ...; return psi`` — discarding the inner trajectory INSIDE a shared
        numerics primitive, so the adjoint eigenvalue path's inner level was
        unrecoverable from ``orpheus/sn/`` at any level of effort.  `[M]`
        ``solve_sn_adjoint`` reported ``total_inner_iterations is None``
        where the forward path reported 1470.  The records now accumulate on
        the instance (reset per :meth:`solve`), which is the same shape
        ``SNSolver`` already uses for its inner counter.

        ⚠ An accumulator, not a return value, is a way-station: the honest
        home is the outer's own record, and that arrives with #340 N2b when
        ``PowerIterationOutcome`` learns to carry children.  It is recorded
        here rather than deferred because a discarded trajectory cannot be
        recovered later, while a mis-homed one can be moved.
        """
        psi, record = self._inner.solve(
            fission_source, initial_guess=flux_distribution,
        )
        self.inner_records.append(record)
        return psi

    def compute_production_rate(self, flux_distribution: V) -> float:
        r"""Production-rate normalisation: :math:`P(\psi) = \sum (F\,\psi)`.

        Power iteration renormalises ψ to unit production each outer step so the
        iterate stays at :math:`O(1)` regardless of super/subcriticality
        (a subcritical iterate would otherwise decay to denormalised FP and the
        keff ratio become 0/0 — ERR-052).  Production is scale-invariant in
        ``keff`` so the converged eigenvalue is unchanged; the converged ``ψ``
        carries the canonical convention :math:`\int \nu\Sigma_f\,\phi\,dV = 1`,
        which makes rescaling to absolute flux at a target power a single
        multiplication by :math:`P_{\text{target}} / \kappa`.

        The ``F`` operator already carries any volume weights its domain
        advertises; the unweighted sum over array entries is the discrete
        :math:`\int \nu\Sigma_f\,\phi\,dV` when the operator's action
        absorbs the measure (as ORPHEUS's typed operators do).  Hardwired
        since #259 P1 / R8 — this operator-level adapter's estimator is
        not injectable (arithmetic identical to the retired default).

        Carrier-honest (#276 A4): a typed composite iterate ravels through
        the same protocol the inner drivers use (:func:`_ravel`); a bare
        ndarray reduces identically to the pre-A4 ``np.sum``.
        """
        return float(_ravel(self.F.apply(flux_distribution)).sum())

    def compute_keff(self, flux_distribution: V) -> float:
        r"""Operator-form Rayleigh :math:`k` estimator (hardwired; #259 P1 / R8).

        .. math::

            k \;=\; \frac{\sum (F\,\psi)}{\sum (A\,\psi) - \sum (S\,\psi)}

        — the operator-level spelling of the unified k discipline
        (fission production over net removal): when ``A`` carries
        streaming + collision, :math:`\sum(A\psi) - \sum(S\psi)` is
        absorption + leakage − scattering-family gains, term-for-term
        the method-layer functional
        (:meth:`orpheus.sn.solver.SNSolver.compute_keff` / diffusion's
        loss action) with the volume measure absorbed into the
        operators' action.  This spelling is leakage-inclusive through
        ``A`` — it never had the #291 omission.

        Contract: requires an HONEST ``A.apply`` — an adapter whose
        ``apply`` is a stub yields a non-eigenvalue here.  The retired
        injection seam used to let such adapters substitute their own
        functional; post-R8 the posed triple IS the estimator's source,
        and at a converged eigenpair every consistent estimator agrees
        with this one.

        Carrier-honest (#276 A4): typed composites ravel through
        :func:`_ravel` (bare ndarrays reduce identically).  On a DAGGERED
        triple ``(A.H, S.H, F.H)`` this same spelling is the adjoint
        Rayleigh estimator — at the converged adjoint eigenpair
        :math:`(A^\dagger - S^\dagger)\psi^* = F^\dagger\psi^*/k` holds
        exactly, so any linear functional (here the coordinate sum)
        recovers the SAME ``k`` as the forward problem
        (:math:`\text{eig}(A^\dagger) = \text{eig}(A)`).
        """
        num = _ravel(self.F.apply(flux_distribution)).sum()
        den = (
            _ravel(self.A.apply(flux_distribution)).sum()
            - _ravel(self.S.apply(flux_distribution)).sum()
        )
        return float(num / den)

    def measure_stopping_criteria(
        self, keff: float, keff_old: float,
        flux_distribution: V, flux_old: V,
    ) -> tuple[StoppingCriterion, ...]:
        """``dk`` against ``keff_tol`` and relative ``dφ`` against ``flux_tol``.

        Carrier-honest (#276 A4): norms go through :func:`_l2_norm` (the
        ravellable protocol; bit-identical ``np.linalg.norm`` on a bare
        ndarray), and the iterate difference is the carrier's own
        :class:`~orpheus.numerics.vector.Vector` ``__sub__``.

        ⛔ Was ``converged(...) -> bool`` with a leading ``if iteration <= 2``
        until 2026-08-09 (#340 N2b).  That guard is a property of power
        iteration, not of this solver, and now lives once at
        :data:`~orpheus.numerics.eigenvalue.MINIMUM_OUTER_ITERATIONS`; the
        magnitudes it used to consume and discard are the return value.
        """
        dk = abs(keff - keff_old)
        norm = _l2_norm(flux_distribution)
        dphi = (
            _l2_norm(flux_distribution - flux_old) / max(norm, 1e-30)
            if norm > 0.0
            else _l2_norm(flux_distribution - flux_old)
        )
        return (
            StoppingCriterion.reading("dk", float(dk), self.keff_tol),
            StoppingCriterion.reading("dphi", float(dphi), self.flux_tol),
        )

    def solve(
        self,
        initial_guess: V | None = None,
    ) -> "PowerIterationOutcome[V]":
        r"""Run the eigenvalue solve via the canonical ``power_iteration`` loop.

        KEigenvalue realizes the
        :class:`~orpheus.numerics.eigenvalue.EigenvalueSolver` boundary (the
        methods above) from its ``(A, S, F)`` triple and delegates the outer
        iteration to :func:`~orpheus.numerics.eigenvalue.power_iteration` — the
        SINGLE power-iteration loop in the codebase (Cardinal Rule 2).  The
        ``eigenvalue_method`` selector (validated at construction) reserves the
        full-spectrum / shift-invert seam; only ``"power"`` is implemented.

        Parameters
        ----------
        initial_guess : np.ndarray
            Initial flux guess.  REQUIRED — the operator triple does not expose
            its action shape, so the iterate shape is inferred from here.

        Returns
        -------
        keff : float
            Converged dominant eigenvalue.
        keff_history : list[float]
            Eigenvalue at every outer iteration.
        psi : np.ndarray
            Converged fundamental-mode iterate (unit production-rate
            normalisation).
        """
        if initial_guess is None:
            raise ValueError(
                "KEigenvalue.solve requires initial_guess for shape "
                "inference; the operator triple is not constrained to "
                "expose its action shape.  Pass np.ones(...) of the "
                "appropriate shape (or a typed composite carrier), or use "
                "the SNSolver wrapper that already builds the initial guess."
            )
        # Carrier-honest guess stash (#276 A4), the SourceIteration idiom:
        # a typed composite is frozen — the alias IS a faithful stash; a
        # bare-ndarray/sequence guess coerces + copies exactly as before
        # (the cast states the module-intro blessing of ndarrays through
        # the Carrier slots).
        self._initial_guess = (
            initial_guess  # typed: trust frozen-arithmetic contract
            if _is_ravellable(initial_guess)
            else cast(V, np.asarray(initial_guess).copy())
        )
        # Delegate the loop to the canonical algorithm.  No import cycle:
        # eigenvalue.py does not import iteration.py.
        from .eigenvalue import power_iteration
        # Reset before the loop, so a reused instance reports THIS solve's
        # inners rather than every solve it has ever run.  (#340 F12 measured
        # the un-reset variant on the retired scalar
        # ``SNSolver._total_inner_iterations``, where two power_iteration
        # calls on one solver silently double-counted.)
        self.inner_records = []
        return power_iteration(
            self, max_iter=self.max_outer, budget_name="max_outer"
        )
