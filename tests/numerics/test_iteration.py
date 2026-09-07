"""Foundation + L1 tests for ``orpheus.numerics.iteration``.

Wave E Round 1 (Issue #163) ships :class:`SourceIteration` and
:class:`KEigenvalue` as stand-alone iteration primitives that consume
the Wave A :class:`LinearOperator` Protocol triple :math:`(L, S, F)`.

Tests in this file:

* **Foundation (synthetic):** L0 dense-matrix fixtures where the
  ground truth is :func:`numpy.linalg.solve` /
  :func:`numpy.linalg.eig`.  Pin the algorithmic correctness of the
  primitives in isolation from any transport solver.
* **Foundation (apply-guards):** the constructors raise
  :class:`TypeError` when their argument operators lack the
  required Protocol surface.
* **L1 (SN integration gate):** build an actual SN operator triple
  (:class:`~orpheus.sn.operators.streaming.StreamingCollisionOperator`
  (= ``A = L + C``) /
  :class:`ScatteringOperator` / :class:`FissionOperator`) for a
  2-group homogeneous slab and assert that :class:`KEigenvalue`
  recovers the same :math:`k_{\\rm eff}` as :func:`solve_sn`.  This
  is the gate test that the new primitives compose with the existing
  SN operator algebra.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from orpheus.derivations.common.xs_library import get_mixture
from orpheus.geometry import Mesh1D
from orpheus.numerics.iteration import (
    KEigenvalue,
    KrylovAcceleration,
    SourceIteration,
)
from orpheus.numerics.operator import (
    InverseOperator,
    LinearOperator,
    NotInvertible,
    ZeroOperator,
)
from orpheus.transport.fields.angular_boundary_flux import AngularBoundaryFlux


# ───────────────────────────────────────────────────────────────────────
# Synthetic fixtures (L0 ground truth)
# ───────────────────────────────────────────────────────────────────────


class MatrixOperator(LinearOperator):
    """Test operator backed by a dense numpy matrix.

    Same shape as the fixture in ``test_operator.py`` — kept independent
    here so tests in this file are self-contained.
    """
    # S4-amendment: the base DEMANDS an answer from every subclass; this
    # double is a deliberately-unbound probe, so it DECLARES the unbound
    # state instead of inheriting a silent default (which no longer exists).
    domain = None
    codomain = None

    def __init__(
        self,
        matrix: np.ndarray,
        *,
        can_solve: bool = False,
        can_transpose: bool = False,
    ) -> None:
        self.matrix = np.asarray(matrix, dtype=float)
        self._can_solve = bool(can_solve)
        self._can_transpose = bool(can_transpose)

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.matrix @ x

    def solve(self, b: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.matrix, b)

    def apply_transpose(self, x: np.ndarray) -> np.ndarray:
        return self.matrix.T @ x

    @property
    def is_invertible(self) -> bool:
        return self._can_solve

    @property
    def is_adjointable(self) -> bool:
        return self._can_transpose

    def inverse(self) -> InverseOperator:
        # The #226 step-3 driver contract: the caller builds the inverse
        # OPERATOR and SourceIteration applies it.  The generic
        # InverseOperator delegates apply → this leaf's solve
        # (bit-identical to the pre-step-3 ``L.solve`` step).
        return InverseOperator(self)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260509)


# ───────────────────────────────────────────────────────────────────────
# Foundation: synthetic SourceIteration
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_source_iteration_recovers_direct_solve(rng):
    """L0 ground truth: SourceIteration on (A − S) matches np.linalg.solve.

    Build a 4×4 SPD-ish ``A`` and a contraction ``S`` (spectral radius
    well below 1).  Solve ``(A − S)·ψ = q`` two ways:

    * Directly: ``np.linalg.solve(A − S, q)``.
    * By SourceIteration with ``F = ZeroOperator()``.

    The fixed-point iteration converges geometrically at rate
    :math:`\\rho(A^{-1}\\,S)`; the two answers must agree to
    1e-10 absolute.
    """
    n = 4
    # Diagonal-dominant matrix → well-conditioned solve.
    A_mat = np.eye(n) * 4.0 + 0.1 * rng.standard_normal((n, n))
    A_mat = 0.5 * (A_mat + A_mat.T) + n * np.eye(n)  # symmetric, dominant
    # Small contraction: spectral radius of A^{-1} S < 0.1.
    S_mat = 0.05 * rng.standard_normal((n, n))

    A = MatrixOperator(A_mat, can_solve=True)
    S = MatrixOperator(S_mat)
    F = ZeroOperator()

    q = rng.standard_normal(n)
    expected = np.linalg.solve(A_mat - S_mat, q)

    si = SourceIteration(A.inverse(), S, F, max_iter=1000, tol=1e-14)
    psi, record = si.solve(q)
    residuals = _trajectory(record)

    np.testing.assert_allclose(psi, expected, atol=1e-10, rtol=1e-10)
    # Residual history must be monotonically (or near-monotonically)
    # decreasing — a contraction map produces geometric decay.
    assert residuals[-1] < 1e-10, (
        f"Residual did not reach 1e-10; final={residuals[-1]:.2e}"
    )


@pytest.mark.foundation
def test_source_iteration_with_fission_term(rng):
    """SourceIteration on full (A − S − F) recovers direct solve.

    Same construction as above but with a small ``F`` term.  Solves
    ``(A − S − F)·ψ = q`` by fixed-point iteration; compares to
    ``np.linalg.solve(A − S − F, q)``.
    """
    n = 4
    A_mat = np.eye(n) * 5.0 + 0.05 * rng.standard_normal((n, n))
    A_mat = 0.5 * (A_mat + A_mat.T) + n * np.eye(n)
    S_mat = 0.05 * rng.standard_normal((n, n))
    F_mat = 0.05 * rng.standard_normal((n, n))

    A = MatrixOperator(A_mat, can_solve=True)
    S = MatrixOperator(S_mat)
    F = MatrixOperator(F_mat)

    q = rng.standard_normal(n)
    expected = np.linalg.solve(A_mat - S_mat - F_mat, q)

    si = SourceIteration(A.inverse(), S, F, max_iter=2000, tol=1e-14)
    psi, _ = si.solve(q)

    np.testing.assert_allclose(psi, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.foundation
def test_source_iteration_with_explicit_solve_realisation():
    r"""The caller controls the inverse step by BUILDING the inverse operator.

    #226 taxonomy step 3 (superseding the R-1 Step B ``solve`` contract):
    the solver layer builds ``A.inverse()`` — whose ``apply`` delegates to
    the leaf's ``solve``, bit-identical — and :class:`SourceIteration`
    only APPLIES it.  This test exercises the pattern with a dense matrix
    wrapped in a ``MatrixOperator``: the caller's chosen inverse action
    (``np.linalg.solve`` under the hood) reaches the driver as the
    inverse-application operator, not as a duck-typed ``.solve`` surface.
    """
    n = 3
    A_mat = np.diag([5.0, 6.0, 7.0])
    S_mat = 0.1 * np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])

    # The caller-controlled inverse: the leaf carries the solve
    # realisation; .inverse() lifts it to the operator the driver applies.
    A = MatrixOperator(A_mat, can_solve=True)
    S = MatrixOperator(S_mat)
    F = ZeroOperator()

    q = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.solve(A_mat - S_mat, q)

    si = SourceIteration(A.inverse(), S, F, max_iter=500, tol=1e-14)
    psi, _ = si.solve(q)
    np.testing.assert_allclose(psi, expected, atol=1e-12)


# ───────────────────────────────────────────────────────────────────────
# Foundation: synthetic KrylovAcceleration (R-1 sibling of SourceIteration)
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_krylov_acceleration_recovers_direct_solve(rng):
    """L0 ground truth: KrylovAcceleration on (A − S) matches np.linalg.solve.

    Same algebraic setup as
    :func:`test_source_iteration_recovers_direct_solve`.  GMRES on
    the composed (A − S) matvec, with ``A.inverse().apply`` as the
    default preconditioner.  Convergence to 1e-10 should take far fewer
    matvecs than source iteration because A − S is well-conditioned.
    """
    n = 4
    A_mat = np.eye(n) * 4.0 + 0.1 * rng.standard_normal((n, n))
    A_mat = 0.5 * (A_mat + A_mat.T) + n * np.eye(n)
    S_mat = 0.05 * rng.standard_normal((n, n))

    A = MatrixOperator(A_mat, can_solve=True)
    S = MatrixOperator(S_mat)
    F = ZeroOperator()

    q = rng.standard_normal(n)
    expected = np.linalg.solve(A_mat - S_mat, q)

    krylov = KrylovAcceleration(A, S, F, max_iter=200, tol=1e-12)
    psi, record = krylov.solve(q)
    residuals = _trajectory(record)

    np.testing.assert_allclose(psi, expected, atol=1e-10, rtol=1e-10)
    assert record.iterated, "GMRES callback never fired"
    assert residuals[-1] < 1e-8, (
        f"GMRES residual did not reach 1e-8; final={residuals[-1]:.2e}"
    )


@pytest.mark.foundation
def test_krylov_acceleration_with_fission_term(rng):
    """KrylovAcceleration on full (A − S − F) recovers direct solve."""
    n = 4
    A_mat = np.eye(n) * 5.0 + 0.05 * rng.standard_normal((n, n))
    A_mat = 0.5 * (A_mat + A_mat.T) + n * np.eye(n)
    S_mat = 0.05 * rng.standard_normal((n, n))
    F_mat = 0.05 * rng.standard_normal((n, n))

    A = MatrixOperator(A_mat, can_solve=True)
    S = MatrixOperator(S_mat)
    F = MatrixOperator(F_mat)

    q = rng.standard_normal(n)
    expected = np.linalg.solve(A_mat - S_mat - F_mat, q)

    krylov = KrylovAcceleration(A, S, F, max_iter=200, tol=1e-12)
    psi, _ = krylov.solve(q)

    np.testing.assert_allclose(psi, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.foundation
def test_krylov_acceleration_explicit_preconditioner():
    """Caller-supplied ``preconditioner`` shadows the default inverse choice.

    R-1 Step B (2026-05-19) — the parameter name is ``preconditioner``
    (not ``inverter``).  Pass an ``A`` that is NOT invertible and
    supply ``preconditioner`` — construction must succeed and GMRES
    must converge using the supplied preconditioner.
    """
    n = 3
    A_mat = np.diag([5.0, 6.0, 7.0])
    S_mat = 0.1 * np.array([[0.0, 1.0, 0.0],
                            [1.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])

    A = MatrixOperator(A_mat, can_solve=False)
    S = MatrixOperator(S_mat)
    F = ZeroOperator()

    inv_A = np.linalg.inv(A_mat)
    preconditioner = lambda q: inv_A @ q

    q = np.array([1.0, 2.0, 3.0])
    expected = np.linalg.solve(A_mat - S_mat, q)

    krylov = KrylovAcceleration(
        A, S, F, preconditioner=preconditioner, max_iter=100, tol=1e-12,
    )
    psi, _ = krylov.solve(q)
    np.testing.assert_allclose(psi, expected, atol=1e-10)


@pytest.mark.foundation
def test_krylov_acceleration_works_without_preconditioner():
    """KrylovAcceleration runs unpreconditioned when A is not invertible.

    No ``preconditioner`` supplied, ``A`` not invertible — GMRES
    still converges, just with more iterations (M = I, the identity
    preconditioner).
    """
    n = 5
    # Well-conditioned diagonal-dominant A so unpreconditioned GMRES
    # still converges quickly.
    A_mat = np.eye(n) * 10.0
    A = MatrixOperator(A_mat, can_solve=False)
    S = ZeroOperator()
    F = ZeroOperator()

    q = np.arange(1.0, n + 1.0)
    expected = q / 10.0

    krylov = KrylovAcceleration(A, S, F, max_iter=50, tol=1e-12)
    assert krylov._preconditioner is None, (
        "Expected no preconditioner when A is not invertible and no "
        "preconditioner is supplied."
    )
    psi, _ = krylov.solve(q)
    np.testing.assert_allclose(psi, expected, atol=1e-10)


@pytest.mark.foundation
def test_krylov_acceleration_high_scattering_beats_source_iteration():
    """At c → 1, GMRES converges in many fewer matvecs than SI.

    The whole point of the KrylovAcceleration sibling is the
    spectral-radius win when the scattering ratio is high.  Pin the
    qualitative win at c ≈ 0.9 (~ρ(A⁻¹S) ≈ 0.9 ⇒ SI needs
    ~log(tol)/log(0.9) ≈ 220 iterations to reach 1e-10 vs GMRES at
    well under that).
    """
    n = 8
    # Diagonal A; nearly-uniform S to make ρ(A⁻¹·S) ≈ 0.9.
    A_mat = np.eye(n) * 1.0
    # All entries = 0.9/n so the row-sum is 0.9 and A⁻¹·S has spectral
    # radius exactly 0.9.
    S_mat = np.full((n, n), 0.9 / n)
    A = MatrixOperator(A_mat, can_solve=True)
    S = MatrixOperator(S_mat)
    F = ZeroOperator()
    q = np.arange(1.0, n + 1.0)

    si = SourceIteration(A.inverse(), S, F, max_iter=500, tol=1e-10)
    _, si_record = si.solve(q)

    krylov = KrylovAcceleration(A, S, F, max_iter=500, tol=1e-10)
    _, kr_record = krylov.solve(q)

    # GMRES should converge in well under SI's iteration count.  The
    # exact ratio is problem-dependent; pin the qualitative gap at 5×.
    # ⭐ Counts, not trajectory lengths.  The two drivers use DIFFERENT
    # conventions (SI measures differences, so its pass count exceeds its
    # trajectory by one; GMRES gets one callback per iteration), and this
    # comparison was silently mixing them until each driver began stating
    # its own count (#340 F11).
    assert kr_record.n_iterations < si_record.n_iterations / 5, (
        f"KrylovAcceleration ({kr_record.n_iterations} iters) was not "
        f"meaningfully faster than SourceIteration "
        f"({si_record.n_iterations} iters) at c=0.9 — the algorithmic win "
        f"that motivates the sibling primitive is missing."
    )


@pytest.mark.foundation
def test_krylov_acceleration_requires_apply_on_A():
    class BrokenA:
        pass  # genuinely no apply — the eager guard rejects

    with pytest.raises(TypeError, match=r"requires 'apply' on A"):
        KrylovAcceleration(BrokenA(), ZeroOperator(), ZeroOperator())


@pytest.mark.foundation
def test_krylov_acceleration_requires_apply_on_first_coupling():
    """A coupling gain without apply → TypeError at construction.

    Wave O #208 O.2a: the drivers take the variadic ``(A, *gains)`` shape; the
    per-gain apply check names the offending gain by index (the legacy
    ``S``/``F`` named slots are retired).
    """
    class BrokenS:
        pass

    A = MatrixOperator(np.eye(3), can_solve=True)
    with pytest.raises(
        TypeError,
        match=r"requires 'apply' on every coupling operator; gain 0",
    ):
        KrylovAcceleration(A, BrokenS(), ZeroOperator())


@pytest.mark.foundation
def test_krylov_acceleration_requires_apply_on_later_coupling():
    """A broken gain at a non-zero index is caught and named by its index."""
    class BrokenF:
        pass

    A = MatrixOperator(np.eye(3), can_solve=True)
    with pytest.raises(
        TypeError,
        match=r"requires 'apply' on every coupling operator; gain 1",
    ):
        KrylovAcceleration(A, ZeroOperator(), BrokenF())


# ───────────────────────────────────────────────────────────────────────
# Foundation: synthetic KEigenvalue
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_keigenvalue_recovers_dominant_eigenvalue(rng):
    """L0 ground truth: KEigenvalue matches numpy.linalg.eig dominant root.

    Build ``A`` (full apply + solve) and ``F`` (apply only).  Solve
    the generalised eigenvalue problem :math:`A\\,\\psi = (1/k)\\,F\\,\\psi`
    two ways:

    * Directly: largest eigenvalue of ``A^{-1}·F`` from
      :func:`numpy.linalg.eig`.
    * By :class:`KEigenvalue` with ``S = ZeroOperator``.

    Power iteration converges to the dominant eigenvalue; the two
    must agree to 1e-9 absolute.
    """
    n = 4
    # Diagonally-dominant L with positive diagonal.
    A_mat = np.diag([3.0, 5.0, 7.0, 11.0]) + 0.05 * rng.standard_normal((n, n))
    A_mat = 0.5 * (A_mat + A_mat.T) + 5.0 * np.eye(n)
    # F: small dominance ratio (k_0 / k_1 ≈ 2 for fast convergence).
    F_mat = np.diag([2.0, 1.0, 0.5, 0.25])

    A = MatrixOperator(A_mat, can_solve=True)
    S = ZeroOperator()
    F = MatrixOperator(F_mat)

    # Reference: largest k = eig(A^{-1} F) (the K = A⁻¹F multiplication
    # operator's dominant root).
    A_inv_F = np.linalg.solve(A_mat, F_mat)
    eigvals = np.linalg.eigvals(A_inv_F)
    expected_keff = float(np.max(np.real(eigvals)))

    initial = np.ones(n)
    ke = KEigenvalue(
        A, S, F,
        max_outer=500, keff_tol=1e-12, flux_tol=1e-12,
        max_inner=500, inner_tol=1e-14,
    )
    _o = ke.solve(initial_guess=initial)
    keff, keff_history, psi = _o.keff, _o.keff_history, _o.flux_distribution

    assert abs(keff - expected_keff) < 1e-9, (
        f"KEigenvalue keff={keff!r} expected≈{expected_keff!r}; "
        f"history[-3:]={keff_history[-3:]}"
    )

    # The eigenvector should satisfy (A⁻¹F)·ψ = k·ψ to high precision.
    np.testing.assert_allclose(A_inv_F @ psi, keff * psi, atol=1e-7)


# ───────────────────────────────────────────────────────────────────────
# Foundation: eager apply-guards (carve P4 — TypeError, no registry)
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.foundation
def test_source_iteration_requires_apply_on_A_inv():
    """A_inv without apply → TypeError at construction."""
    class BrokenAInv:
        pass  # genuinely no apply

    A_inv = BrokenAInv()
    S = MatrixOperator(np.eye(2))
    F = ZeroOperator()

    with pytest.raises(TypeError, match="apply"):
        SourceIteration(A_inv, S, F)


@pytest.mark.foundation
def test_source_iteration_requires_apply_on_S():
    """S without apply → TypeError at construction."""
    class BrokenS:
        pass

    A = MatrixOperator(np.eye(2), can_solve=True)
    S = BrokenS()
    F = ZeroOperator()

    with pytest.raises(TypeError, match="apply"):
        SourceIteration(A.inverse(), S, F)


@pytest.mark.foundation
def test_source_iteration_requires_apply_on_F():
    """F without apply → TypeError at construction."""
    class BrokenF:
        pass

    A = MatrixOperator(np.eye(2), can_solve=True)
    S = MatrixOperator(np.eye(2))
    F = BrokenF()

    with pytest.raises(TypeError, match="apply"):
        SourceIteration(A.inverse(), S, F)


@pytest.mark.foundation
def test_invertibility_obligation_lives_at_the_inverse_builder():
    """The R-1 Step B "L must solve" gate MIGRATED to the builder (#226 step 3).

    :class:`SourceIteration` no longer demands ``CAP_SOLVE`` — its step
    operator arrives pre-inverted, and an APPLY-ONLY step operator is
    acceptable BY DESIGN (the windowed product ``P @ A.inverse()`` is
    exactly that shape: no round-trip promise, just the family's canonical
    seeded-apply signature).  The "can this be inverted?" obligation now
    discharges where the inverse is BUILT: ``.inverse()`` on a
    non-invertible leaf raises with the domain message.
    """
    # The obligation fires at the builder …
    with pytest.raises(NotInvertible, match="invertible"):
        MatrixOperator(np.eye(2), can_solve=False).inverse()

    # … and the driver runs an apply-only, seeded-signature step operator
    # END-TO-END (the windowed-product shape): apply-only, the
    # inverse action baked into ``apply``.  Zero gains → one exact step.
    class ApplyOnlyStep:
        def apply(self, rhs, *, initial_guess=None):
            return rhs / 2.0  # the exact inverse of L = 2·I

    si = SourceIteration(ApplyOnlyStep(), max_iter=5, tol=1e-14)
    psi, _ = si.solve(np.array([2.0, 4.0]))
    np.testing.assert_allclose(psi, np.array([1.0, 2.0]))


@pytest.mark.foundation
def test_keigenvalue_requires_invertible_A():
    """KEigenvalue (the posing layer that BUILDS A.inverse()) guards
    invertibility at construction with a domain message — not an
    AttributeError from a missing ``.inverse`` (#226 step 3)."""
    A = MatrixOperator(np.eye(2), can_solve=False)
    S = MatrixOperator(np.eye(2))
    F = MatrixOperator(np.eye(2))

    with pytest.raises(NotInvertible, match="INVERTIBLE"):
        KEigenvalue(A, S, F)


@pytest.mark.foundation
def test_keigenvalue_rejects_non_power_method():
    """eigenvalue_method != 'power' raises NotImplementedError."""
    A = MatrixOperator(np.eye(2), can_solve=True)
    S = ZeroOperator()
    F = MatrixOperator(np.eye(2))

    with pytest.raises(NotImplementedError, match="FEAST"):
        KEigenvalue(A, S, F, eigenvalue_method="feast")


# ───────────────────────────────────────────────────────────────────────
# L1: SN integration gate
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.l1
@pytest.mark.verifies("multigroup")
def test_keigenvalue_matches_solve_sn_2g_slab():
    """L1 gate: KEigenvalue on the SN operator triple matches solve_sn.

    Build a 2-group homogeneous-material 1-D slab, run both:

    * :func:`solve_sn` (the legacy power_iteration-based path), and
    * :class:`KEigenvalue` directly on
      ``(StreamingCollisionOperator (= L + C), ScatteringOperator, FissionOperator)``
      with adapter shims that present scalar-flux shapes consistent
      across operators.

    Assert recovered keff agrees to 1e-9.  Both paths use the SAME
    underlying operators — the only difference is whether the iteration
    primitive is the legacy ``power_iteration(solver)`` (Wave-E pre-
    Round-1) or the new ``KEigenvalue`` (Wave-E Round 1).

    R8 rewire (#259 P1, 2026-07-03 — the estimator-injection seam
    retired): the pre-R8 version injected ``solver.compute_keff`` as
    KEigenvalue's ``keff_estimator``.  Post-R8 the k assertion routes
    through the THEOREM instead: the eigen-FLUX fixed point is
    estimator-independent (unit-production renormalisation cancels the
    k scaling), and at a converged eigenpair every CONSISTENT estimator
    agrees — so evaluating the SN method-layer functional
    ``solver.compute_keff`` at KEigenvalue's converged flux must
    reproduce ``solve_sn``'s reported k.  KEigenvalue's OWN reported k
    is deliberately NOT asserted on this stack: the scalar-level
    ``A_inv_adapter`` cannot advertise an honest ``.apply`` (the true
    within-group loss action is angular), so the hardwired Rayleigh
    denominator ``Σ(Aψ)−Σ(Sψ)`` is off-contract here — a limitation the
    injection seam used to paper over, exposed (by design) at R8.  The
    honest-triple version of the reported-k leg lives with the
    operator-level pins in ``test_estimators_as_functionals.py``.
    """
    # Suppress the eigenvalue.py deprecation warning for this test —
    # we are deliberately exercising the legacy path as a reference.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        from orpheus.transport.operators.fission import FissionOperator
        from orpheus.sn.mesh.augmented_mesh import SNMesh
        from orpheus.numerics.quadrature import Quadrature
        from orpheus.transport.operators.scattering import ScatteringOperator
        from orpheus.sn.solver import SNSolver, solve_sn
        from tests.sn._test_helpers import reflect_outflow_into_inflow, sweep_once

    # 2-group homogeneous 1-D slab — the same canonical fixture
    # ``test_solver_components.py`` uses for component checks.
    mix = get_mixture("A", "2g")
    materials = {0: mix}
    mesh = Mesh1D(
        edges=np.linspace(0.0, 5.0, 11),
        mat_ids=np.zeros(10, dtype=int),
    )
    quad = Quadrature.gauss_legendre(n_ordinates=8)

    # Reference: solve_sn (legacy power_iteration path).
    ref = solve_sn(
        materials, mesh, quad,
        inner_solver="source_iteration",
        scattering_order=0,
        max_outer=500,
        keff_tol=1e-9,
        flux_tol=1e-8,
        max_inner=500,
        inner_tol=1e-10,
    )
    expected_keff = ref.keff
    if expected_keff is None:
        pytest.fail("solve_sn reference returned no eigenvalue.")

    # Build the SN operator triple from the same precomputed solver
    # data used for the reference run.  ``A_inv_adapter`` (defined
    # below) wraps ``sweep_once`` directly; the
    # :class:`StreamingCollisionOperator` (= ``L + C``) on the SNSolver is
    # unused here.  Solver instance retained to provide
    # ``solver.scattering_op`` / ``solver.fission_op`` / ``solver.mat_xs``.
    sn_mesh = SNMesh(mesh, quad, materials)
    solver = SNSolver(sn_mesh, scattering_order=0)
    # The canonical S, F operators built directly from solver state.
    S = solver.scattering_op
    F = solver.fission_op

    # ── Adapter shims to keep the iteration primitive scalar-flux-only.
    #
    # The SN operator triple has an internal shape-mismatch for
    # historical reasons (Wave D Round 3 docstring §"Vector layout"):
    #   • F.apply takes scalar phi (nx,ny,ng), returns scalar Q (nx,ny,ng)
    #   • S.apply takes angular psi (N,nx,ny,ng), returns angular Q (N,nx,ny,ng)
    #   • L.solve takes Q+psi_bc+Q_aniso, returns (psi, phi) tuple.
    # Round 2 normalises this; for the L1 gate test we wrap each
    # operator into a thin scalar-in/scalar-out facade.
    # Issue #197 PR-TYPED-2 — typed AngularBoundaryFlux replaces psi_bc: dict.
    boundary_flux = AngularBoundaryFlux.zeros(sn_mesh.angular_trace)

    class A_inv_adapter(LinearOperator):
        """Adapter: rhs (ng, nx, ny) → phi via the unified sweep.

        Issue #196 PR-INDEX-5: principled layout throughout.  Returns
        scalar flux (drops angular flux for shape consistency with
        F.apply / S.apply scalar facade).

        #226 step 3: carries the invertibility pair (``is_invertible`` +
        ``inverse()``) so :class:`KEigenvalue` — which now BUILDS the
        inverse and hands it to the inner driver — can lift this leaf's
        ``solve`` through the generic :class:`InverseOperator`.
        """
        # S4-amendment: the base DEMANDS an answer from every subclass; this
        # double is a deliberately-unbound probe, so it DECLARES the unbound
        # state instead of inheriting a silent default (which no longer exists).
        domain = None
        codomain = None

        @property
        def is_invertible(self) -> bool:
            return True

        def inverse(self) -> InverseOperator:
            return InverseOperator(self)

        def apply(self, phi):  # not used by the iteration primitive
            return phi

        def solve(self, rhs):
            # R-1 Step 4 A1: single per-ordinate source carrier.
            # ``rhs`` is bare ndarray (ng, nx, ny) — wrap via the
            # canonical iso → per-ord factory at the adapter boundary.
            from orpheus.transport.source_sinks import AngularSourceSink
            source = AngularSourceSink.from_isotropic(rhs, sn_mesh)
            # Wave O (#208) O.4a.2 — the bare ``transport_sweep`` no longer
            # re-applies the reflective BC at entry; drive the −B coupling
            # explicitly (reflect the persisted outflow — ``boundary_flux``
            # is the closure-scoped partner-flux carrier — into the inflow
            # slots) before each sweep — the sweep-tier gates' inter-sweep −B
            # (the drivers deliver it as the ``B`` gain; #448).
            reflect_outflow_into_inflow(boundary_flux, sn_mesh)
            _angular, scalar = sweep_once(
                source, solver.mat_xs.total_cross_section, sn_mesh,
                boundary_flux,
            )
            return scalar

    class S_scalar_adapter(LinearOperator):
        """Adapter: phi (ng, nx, ny) → P0 scattering source (ng, nx, ny).

        Issue #196 PR-INDEX-5: principled end-to-end.
        """
        # S4-amendment: the base DEMANDS an answer from every subclass; this
        # double is a deliberately-unbound probe, so it DECLARES the unbound
        # state instead of inheriting a silent default (which no longer exists).
        domain = None
        codomain = None

        def apply(self, phi):
            Q = np.zeros_like(phi)
            S.transfer.add_p0_source(Q, phi)
            # §14.1: the (n,2n) verb lives on the solver-held N2N binding.
            solver.n2n_op.isotropic_energy.transfer.add_p0_source(Q, phi)
            return Q

    class F_scalar_adapter(LinearOperator):
        """Adapter: phi (ng, nx, ny) → fission source (ng, nx, ny).

        Issue #196 PR-INDEX-5: principled end-to-end.
        """
        # S4-amendment: the base DEMANDS an answer from every subclass; this
        # double is a deliberately-unbound probe, so it DECLARES the unbound
        # state instead of inheriting a silent default (which no longer exists).
        domain = None
        codomain = None

        def apply(self, phi):
            return F.apply(phi)

    A_adapt = A_inv_adapter()
    S_adapt = S_scalar_adapter()
    F_adapt = F_scalar_adapter()

    ng = solver.ng
    # Issue #196 PR-INDEX-5: principled initial guess.
    initial = np.ones((ng, *sn_mesh.spatial_shape))

    ke = KEigenvalue(
        A_adapt, S_adapt, F_adapt,
        max_outer=500, keff_tol=1e-9, flux_tol=1e-8,
        max_inner=500, inner_tol=1e-10,
    )
    _o = ke.solve(initial_guess=initial)
    _keff_rayleigh, keff_history, phi_ke = _o.keff, _o.keff_history, _o.flux_distribution

    # The theorem-form assertion (R8): the flux fixed point is shared, so
    # the SN method-layer functional evaluated at KEigenvalue's converged
    # flux reproduces solve_sn's reported k.  (All-reflective slab —
    # solver.compute_keff needs no boundary trace; the functional is a
    # scale-invariant ratio, so KEigenvalue's unit-production
    # normalisation drops out.)
    keff_at_ke_flux = solver.compute_keff(phi_ke)
    assert abs(keff_at_ke_flux - expected_keff) < 1e-9, (
        f"SN functional at KEigenvalue's converged flux "
        f"keff={keff_at_ke_flux!r} differs from solve_sn "
        f"keff={expected_keff!r} by {abs(keff_at_ke_flux-expected_keff):.2e}; "
        f"Rayleigh history[-3:]={keff_history[-3:]}"
    )


def _trajectory(record) -> list[float]:
    """The single criterion's trajectory — what ``solve`` returned as a bare
    list before #340 N2a gave the drivers an
    :class:`~orpheus.numerics.convergence.IterationRecord`.

    Not a compatibility shim: these rows are ABOUT the residual sequence, so
    naming the extraction keeps them reading that way.  Rows that are about
    the iteration COUNT read ``record.n_iterations`` directly instead —
    the two are not interchangeable across drivers.
    """
    return list(record.criteria[0].trajectory)


def _sn_composite_triple():
    r"""The HONEST typed-composite SN operator triple + solve_sn reference.

    #276 A4 activation fixture: ``build_within_group_system`` supplies the
    production splitting — ``LC`` (the invertible resolvent whose ``solve``
    is the sweep), the gains ``(S, N2N, B_a)`` summed into KEigenvalue's single
    coupling slot, and the production ``FissionOperator`` — all acting on
    typed :class:`FullField` composites (no scalar adapter shims; contrast
    the legacy-shim gate above, kept as the pre-A4 record).
    """
    from orpheus.numerics.quadrature import Quadrature
    from orpheus.sn.coupled_system import build_within_group_system
    from orpheus.sn.mesh.augmented_mesh import SNMesh
    from orpheus.sn.solver import SNSolver, solve_sn
    from orpheus.transport.fields.angular_flux import AngularFlux
    from orpheus.transport.full_field import FullField

    mix = get_mixture("A", "2g")
    materials = {0: mix}
    mesh = Mesh1D(
        edges=np.linspace(0.0, 5.0, 11), mat_ids=np.zeros(10, dtype=int),
    )
    quad = Quadrature.gauss_legendre(n_ordinates=8)
    ref = solve_sn(
        materials, mesh, quad,
        inner_solver="source_iteration", scattering_order=0,
        max_outer=500, keff_tol=1e-9, flux_tol=1e-8,
        max_inner=500, inner_tol=1e-10,
    )
    if ref.keff is None:  # explicit narrow — fires under -O (Mode 8)
        pytest.fail("solve_sn returned no eigenvalue — the reference leg "
                    "of the A4 activation fixture is broken.")
    sn = SNMesh(mesh, quad, materials)
    solver = SNSolver(sn, scattering_order=0)
    system = build_within_group_system(
        sn, solver.mat_xs, scattering_op=solver.scattering_op,
    )
    from functools import reduce
    from operator import add

    # ALL lagged gains summed into KEigenvalue's single coupling slot —
    # (S, N2N, B_a) since §14.1; a positional pair-sum here silently
    # dropped B_a when the tuple grew (caught by the A4 smoke's own
    # eig(A†)=eig(A) gate).
    S_total = reduce(add, system.explicit_gains)
    # CS4c step 4: the composite triple's F is the ANGULAR binding — the
    # frame-conjugated FissionOperator on the SAME composite space the
    # loss members ride (the pencil peer; its .H composes the Riesz legs
    # around the reversed full_fission_kernel product). The solver-held
    # fission_op is the scalar ENERGY binding the bare-array k-outer
    # feeds — a different, deliberate binding of the same datum.
    from orpheus.transport.operators.fission import FissionOperator

    F_composite = FissionOperator.from_solver_data(
        mat_xs=solver.mat_xs, space=sn.full_field_space,
    )
    guess = FullField(
        interior=AngularFlux(values=np.ones((sn.quad.N, sn.ng, *sn.spatial_shape)), space=sn.angular_bulk_space),
        boundary=AngularBoundaryFlux.zeros(sn.angular_trace),
    )
    return float(ref.keff), system.implicit_operator, S_total, F_composite, guess, mix


@pytest.mark.l1
@pytest.mark.verifies("multigroup")
def test_keigenvalue_honest_composite_triple_matches_solve_sn():
    r"""#276 A4 activation CONTROL: the honest typed-composite triple.

    ``KEigenvalue(LC, S+B_a, F)`` on :class:`FullField` composites — the
    production operators, NO adapter shims — converges to ``solve_sn``'s
    eigenvalue, asserted on KEigenvalue's OWN hardwired Rayleigh estimator
    (on-contract for the first time: ``A.apply`` is the TRUE composite
    ``L+C`` action, which the legacy scalar shim above could not provide).
    This is the FORWARD control leg the daggered smoke below rests on: any
    carrier-honesty regression in the posing layer (the ``_ravel`` sums,
    the ``_l2_norm`` convergence test, the frozen-alias guess stash) reds
    here before any adjoint claim is evaluated.
    """
    ref_keff, LC, S_total, F, guess, _mix = _sn_composite_triple()
    ke = KEigenvalue(
        LC, S_total, F,
        max_outer=500, keff_tol=1e-9, flux_tol=1e-8,
        max_inner=500, inner_tol=1e-10,
    )
    _o = ke.solve(initial_guess=guess)
    keff, history, psi = _o.keff, _o.keff_history, _o.flux_distribution
    np.testing.assert_allclose(
        keff, ref_keff, rtol=0, atol=1e-9,
        err_msg=f"honest-composite KEigenvalue k={keff!r} differs from "
        f"solve_sn k={ref_keff!r}; history[-3:]={history[-3:]}",
    )
    from orpheus.transport.full_field import FullField as _FF

    if not isinstance(psi, _FF):
        pytest.fail(
            f"the honest-composite drive must return a typed composite "
            f"iterate; got {type(psi).__name__}."
        )


@pytest.mark.l1
def test_keigenvalue_daggered_triple_adjoint_smoke():
    r"""#276 A4 activation: ``KEigenvalue(LC.H, (S+B_a).H, F.H)`` — the
    daggered posing through the UNCHANGED ``power_iteration``.

    The adjoint eigenproblem ``A_loss† ψ* = F† ψ*/k`` posed purely by
    dagger-ing the forward triple's members (the operator algebra IS the
    implementation — zero adjoint-specific loop code).  Gates, per Mode-12
    (k is NEVER sole evidence — ``eig(Aᵀ) = eig(A)`` makes every k-level
    functional blind to the whole adjoint mutation class):

    * ``k_adj == k_fwd == solve_sn`` (the exact algebraic equality);
    * the ∞-medium adjoint SPECTRUM equals the corrected closed form
      (``kinf_and_adjoint_spectrum_homogeneous`` — the dominant eigenvector
      of ``(Aᵀ)⁻¹Fᵀ``; on the flat reflective mesh the discrete daggered
      solve reproduces the 0-D energy shape).  This vector-level leg is
      what caught the reference's original ``eig(Mᵀ)`` factor-order
      degeneracy (≡ ν̂Σf, zero A-physics) on this machinery's first run —
      and it is asserted NOT to be that degenerate vector here.

    The full P1.3/P1.4 batteries (4G, heterogeneous slab, sphere, F†/S†
    mutations) land with the solver entries; this row is the posing-layer
    activation smoke.
    """
    from orpheus.derivations.common.eigenvalue import (
        kinf_and_adjoint_spectrum_homogeneous,
    )

    ref_keff, LC, S_total, F, guess, mix = _sn_composite_triple()
    ke_adj = KEigenvalue(
        LC.H, S_total.H, F.H,
        max_outer=500, keff_tol=1e-9, flux_tol=1e-8,
        max_inner=500, inner_tol=1e-10,
    )
    _o = ke_adj.solve(initial_guess=guess)
    k_adj, history, psi_star = _o.keff, _o.keff_history, _o.flux_distribution
    np.testing.assert_allclose(
        k_adj, ref_keff, rtol=0, atol=1e-9,
        err_msg=f"daggered-triple k_adj={k_adj!r} differs from the forward "
        f"solve_sn k={ref_keff!r} — eig(A†)=eig(A) is violated by the "
        f"posing; history[-3:]={history[-3:]}",
    )

    k_cf, phi_star_cf = kinf_and_adjoint_spectrum_homogeneous(
        np.asarray(mix.SigT),
        np.asarray(mix.SigS[0].todense()),
        np.asarray(mix.SigP),
        np.asarray(mix.chi),
    )
    np.testing.assert_allclose(
        k_adj, k_cf, rtol=0, atol=1e-9,
        err_msg="daggered k does not match the closed-form k∞ anchor.",
    )
    bulk = np.asarray(psi_star.interior.values)  # (N, ng, *spatial)
    spec = bulk.mean(axis=(0, *range(2, bulk.ndim)))
    spec = spec / np.linalg.norm(spec)
    np.testing.assert_allclose(
        spec, phi_star_cf, rtol=1e-8,
        err_msg="the SN daggered solve's ∞-medium energy spectrum does not "
        "match the closed-form adjoint eigenvector of (Aᵀ)⁻¹Fᵀ — the "
        "flux-shape leg (the k-blind Mode-12 catcher).",
    )
    nsf_hat = np.asarray(mix.SigP) / np.linalg.norm(np.asarray(mix.SigP))
    if np.allclose(spec, nsf_hat, rtol=1e-6):
        pytest.fail(
            "the daggered spectrum equals ν̂Σf — the eig(Mᵀ) factor-order "
            "degenerate; either the solve or the reference regressed to "
            "the wrong resolvent ordering."
        )


# ───────────────────────────────────────────────────────────────────────
# Foundation: the GMRES exact-breakdown carve-out (the ERR-053 warn
# boundary) — B.2d d3 NIT-2 anchor
# ───────────────────────────────────────────────────────────────────────
#
# The warn branch in :meth:`KrylovAcceleration.solve` carries a PERMANENT
# exact-breakdown carve-out: a final preconditioned residual of LITERAL
# ``0.0`` means the Krylov space collapsed AT the solution (``M⁻¹(b − Ax)
# = 0`` with a nonsingular preconditioner implies ``Ax = b`` exactly) —
# that is CONVERGENCE, the opposite of the ERR-053 restart-truncation
# stall the warning surfaces, so it must NOT warn even when scipy stamps
# ``info > 0``.  The guard's first production caller (the B.2d
# transitional dead-ray GMRES padding) dissolved at the d2 eviction,
# leaving it caller-less-but-permanent — these anchors keep both branch
# arms exercised.  scipy's ``info`` stamping on breakdown is
# version-dependent (current scipy stamps ``info == 0`` on every small
# reachable case because its convergence test is ``<=``), so the branch
# pair is pinned through a deterministic gmres STUB while the real-scipy
# arm pins the caller-visible contract.


def _singular_consistent() -> tuple[MatrixOperator, np.ndarray]:
    r"""``A = diag(1, 0)`` with ``b ∈ range(A)`` — the minimal
    singular-but-CONSISTENT system.  GMRES from ``x0 = 0`` solves it
    exactly in one iteration (``K₁ = span{b}`` contains the solution), so
    the ``pr_norm`` history ends at LITERAL ``0.0`` — the exact-breakdown
    signature."""
    return MatrixOperator(np.diag([1.0, 0.0])), np.array([3.0, 0.0])


def _krylov_warnings(krylov: KrylovAcceleration, b: np.ndarray):
    """Run ``krylov.solve(b)`` recording only this boundary's RuntimeWarnings.

    Returns the pr_norm TRAJECTORY (not the record): all three callers are
    ERR-053 gates asserting on the residual tail, which is the trajectory's
    job.  ``solve`` began returning an
    :class:`~orpheus.numerics.convergence.IterationRecord` at #340 N2a.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x, record = krylov.solve(b)
        history = _trajectory(record)
    ours = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "KrylovAcceleration.solve" in str(w.message)
    ]
    return x, history, ours


@pytest.mark.foundation
@pytest.mark.catches("ERR-053")
def test_singular_consistent_exact_breakdown_solves_clean() -> None:
    r"""REAL scipy: the singular-consistent exact-breakdown solve returns
    the EXACT solution with a literal-``0.0`` residual tail and NO
    ERR-053 warning — the caller-visible contract of the carve-out
    (``tol=0.0`` = "solve to exactness" makes the breakdown the only
    possible exit)."""
    A_op, b = _singular_consistent()
    krylov = KrylovAcceleration(
        A_op, preconditioner=lambda q: q, tol=0.0, max_iter=30, restart=2,
    )
    x, history, ours = _krylov_warnings(krylov, b)
    if not history or history[-1] != 0.0:
        pytest.fail(
            f"the singular-consistent probe no longer ends at a LITERAL 0.0 "
            f"pr_norm (tail = {history[-3:] if history else []}) — the "
            f"exact-breakdown fixture premise broke; redesign the anchor."
        )
    np.testing.assert_allclose(
        A_op.matrix @ x, b, atol=5e-16,
        err_msg="exact breakdown did not return the exact solution",
    )
    if ours:
        pytest.fail(
            f"exact breakdown WARNED ({ours[0].message}) — the ERR-053 "
            f"carve-out regressed: a converged solve is being reported as "
            f"a restart stall."
        )


def _stub_gmres_returning(info: int, pr_norm_tail: float):
    """A deterministic ``spla.gmres`` stand-in: feeds the callback one
    ``pr_norm`` value and stamps ``info`` — pins the guard BRANCH
    independent of scipy's version-dependent breakdown stamping."""

    def _stub(A, b, x0=None, M=None, rtol=None, atol=None, maxiter=None,
              restart=None, callback=None, callback_type=None):
        if callback is not None:
            callback(pr_norm_tail)
        x = np.zeros_like(np.asarray(b))
        return x, info

    return _stub


@pytest.mark.foundation
@pytest.mark.catches("ERR-053")
def test_exact_breakdown_guard_suppresses_the_info_warning(monkeypatch) -> None:
    r"""GUARD arm: ``info > 0`` WITH a literal-``0.0`` tail (the d1-observed
    breakdown stamping) must NOT warn — the carve-out recognizes the
    collapsed-at-the-solution Krylov space as convergence."""
    import orpheus.numerics.iteration as iteration_mod

    monkeypatch.setattr(
        iteration_mod.spla, "gmres", _stub_gmres_returning(info=7, pr_norm_tail=0.0),
    )
    A_op, b = _singular_consistent()
    krylov = KrylovAcceleration(
        A_op, preconditioner=lambda q: q, tol=1e-12, max_iter=30, restart=2,
    )
    _x, history, ours = _krylov_warnings(krylov, b)
    if history != [0.0]:
        pytest.fail(f"stub plumbing broke: history = {history}")
    if ours:
        pytest.fail(
            f"info=7 with a literal-0.0 tail WARNED ({ours[0].message}) — "
            f"the exact-breakdown guard branch regressed."
        )


@pytest.mark.foundation
@pytest.mark.catches("ERR-053")
def test_info_warning_fires_on_genuine_nonconvergence(monkeypatch) -> None:
    r"""TEETH arm: the SAME stub with a NONZERO tail (a genuine ERR-053
    stall signature) MUST warn — proving the guard arm's silence above is
    the carve-out biting, not dead warn machinery."""
    import orpheus.numerics.iteration as iteration_mod

    monkeypatch.setattr(
        iteration_mod.spla, "gmres",
        _stub_gmres_returning(info=7, pr_norm_tail=0.37),
    )
    A_op, b = _singular_consistent()
    krylov = KrylovAcceleration(
        A_op, preconditioner=lambda q: q, tol=1e-12, max_iter=30, restart=2,
    )
    _x, _history, ours = _krylov_warnings(krylov, b)
    if not ours:
        pytest.fail(
            "info=7 with a NONZERO residual tail did not warn — the "
            "ERR-053 non-convergence surface went silent (the guard has "
            "no teeth)."
        )
    if "ERR-053" not in str(ours[0].message):
        pytest.fail(f"the warning lost its ERR-053 pointer: {ours[0].message}")


# ═══════════════════════════════════════════════════════════════════════
# Step 5 (R-5.2/R-5.3) — the ρ-honest free-identity stop (C1/C5/r5)
# ═══════════════════════════════════════════════════════════════════════
#
# The SI stop is the equation residual via the free identity
# ``r_n = rhs_{n−1} − rhs_n = Σ g_i (ψ_{n−1} − ψ_n)`` normalized by
# ``‖q_ext‖`` — EXACT (= ``Aψ_n − q``) when the step operator is an exact
# inverse of M. These rows pin the identity ELEMENT-WISE against
# explicitly-computed references on a dense toy (Mode-12-safe: the object,
# not just a norm), the zero-gain honest exit, and the q ≈ 0 guard.


class _RecordingInverse:
    """A dense exact inverse that RECORDS every (rhs, ψ) pair — the
    reference sequence the free-identity rows recompute against."""

    def __init__(self, A: np.ndarray) -> None:
        self._inv = np.linalg.inv(A)
        self.rhs_seen: list[np.ndarray] = []
        self.psi_out: list[np.ndarray] = []

    def apply(self, rhs, /, *, initial_guess=None):
        del initial_guess
        psi = self._inv @ np.asarray(rhs, dtype=float)
        self.rhs_seen.append(np.asarray(rhs, dtype=float).copy())
        self.psi_out.append(psi.copy())
        return psi


@pytest.mark.foundation
def test_stop_is_the_free_identity_residual_elementwise(rng):
    """C1 — the recorded history ≡ ‖rhs_{n−1} − rhs_n‖/‖q‖ ≡
    ‖(S+F)·Δψ‖/‖q‖ ≡ ‖Aψ − q‖/‖q‖ (three independently-computed
    spellings; the last is the HONEST residual — exact-M toy).
    Discrimination is BY VALUE: the fixture's ‖ψ‖ and ‖q‖ scales differ
    O(10²), so a ψ-normalized mutant (the historical stop) sits O(10²)
    off every reference; a gain dropped from the rhs bookkeeping breaks
    the (S+F)·Δψ match O(1)."""
    n = 6
    A_mat = np.diag(rng.uniform(2.0, 3.0, n))
    S_mat = 0.35 * rng.random((n, n))
    F_mat = 0.25 * rng.random((n, n))
    A_inv = _RecordingInverse(A_mat)
    S, F = MatrixOperator(S_mat), MatrixOperator(F_mat)
    q = 200.0 * (rng.random(n) + 0.5)  # ‖q‖ scale ≫ tol, ≠ ‖ψ‖ scale
    si = SourceIteration(A_inv, S, F, max_iter=200, tol=1e-10)
    psi, record = si.solve(q)
    history = _trajectory(record)

    if not history or not history[-1] < 1e-10:
        pytest.fail("fixture did not converge — the rows below assume the "
                    "break path")
    q_norm = np.linalg.norm(q)
    # The BREAK pass computes one final rhs that never reaches the inverse
    # (the stop fires before the apply) — reconstruct it from the returned
    # iterate; the recorded applies plus that tail ARE the comparison pairs.
    # Reconstructed with the loop's own accumulation ORDER
    # ((q + S·ψ) + F·ψ — associativity matters at the 1e-12 bar):
    psi_arr = np.asarray(psi)
    rhs_seq = A_inv.rhs_seen + [(q + S_mat @ psi_arr) + F_mat @ psi_arr]
    psi_seq = [np.zeros(n)] + A_inv.psi_out
    if len(history) != len(rhs_seq) - 1:
        pytest.fail(
            f"history length {len(history)} ≠ #rhs-comparisons "
            f"{len(rhs_seq) - 1} — the stop is not the rhs-difference"
        )
    # The residual is a DIFFERENCE of O(‖q‖)-scale vectors, so near
    # convergence every independent re-spelling is cancellation-limited
    # (absolute noise ~ eps·‖rhs‖). The identity rows therefore run on
    # the iterations where r is MEASURABLE (well above that floor); the
    # stop-VALUE row (the same subtraction the production loop performs)
    # is bit-safe and runs on every iteration.
    cancellation_floor = 1e-12 * q_norm
    measurable = 0
    for k, res in enumerate(history):
        r_rhs = rhs_seq[k] - rhs_seq[k + 1]
        np.testing.assert_allclose(
            res, np.linalg.norm(r_rhs) / q_norm, rtol=1e-12,
            err_msg=f"iter {k}: the stop value is not ‖r‖/‖q_ext‖ — "
                    f"a ψ-normalized mutant would sit O(‖ψ‖/‖q‖) off",
        )
        if np.linalg.norm(r_rhs) < 1e4 * cancellation_floor:
            continue
        measurable += 1
        r_gain = (S_mat + F_mat) @ (psi_seq[k] - psi_seq[k + 1])
        np.testing.assert_allclose(
            r_rhs, r_gain, rtol=1e-9, atol=cancellation_floor,
            err_msg=f"iter {k}: rhs-difference ≠ (S+F)·Δψ — a gain "
                    f"dropped from the rhs bookkeeping",
        )
        # The HONEST residual of the iterate the comparison certifies:
        r_true = A_mat @ psi_seq[k + 1] - (
            q + (S_mat + F_mat) @ psi_seq[k + 1]
        )
        np.testing.assert_allclose(
            np.linalg.norm(r_rhs), np.linalg.norm(r_true),
            rtol=1e-6, atol=cancellation_floor,
            err_msg=f"iter {k}: the free identity is not the true "
                    f"residual on an exact-M toy",
        )
    if measurable < 10:
        pytest.fail(
            f"only {measurable} iterations above the cancellation floor — "
            f"the identity rows lost their teeth (fixture drift)"
        )
    # And the exit claim is honest: the converged iterate's true residual.
    r_final = A_mat @ psi - (q + (S_mat + F_mat) @ psi)
    if not np.linalg.norm(r_final) / q_norm < 1e-10:
        pytest.fail("the exit claim is dishonest on an exact-M toy")


@pytest.mark.foundation
def test_zero_gain_exits_after_one_apply_with_exact_residual(rng):
    """C5 — zero gains ⟹ rhs ≡ q is constant ⟹ the first comparison sees
    r = 0 EXACTLY: one inverse apply, history == [0.0], and the returned
    iterate is the exact solve (a zero-gain path that iterates or exits
    non-zero is a bookkeeping bug)."""
    n = 5
    A_mat = np.diag(rng.uniform(1.0, 2.0, n))
    A_inv = _RecordingInverse(A_mat)
    q = rng.random(n) + 0.5
    si = SourceIteration(A_inv, max_iter=50, tol=1e-12)
    psi, record = si.solve(q)
    history = _trajectory(record)
    if len(A_inv.rhs_seen) != 1:
        pytest.fail(
            f"zero-gain SI ran {len(A_inv.rhs_seen)} inverse applies — "
            f"expected exactly ONE (ψ₁ = A⁻¹q is exact when A = M)"
        )
    if history != [0.0]:
        pytest.fail(f"zero-gain history {history} — expected [0.0] exactly")
    np.testing.assert_allclose(psi, np.linalg.solve(A_mat, q), rtol=1e-13)


@pytest.mark.foundation
def test_zero_source_zero_start_exits_clean():
    """r5 — ``q_ext = 0`` with the zero cold start: the zero solution is
    found at the first comparison (res = 0/1e-30 = 0.0 exactly), no
    division blow-up, no spurious non-convergence."""
    n = 4
    A_inv = _RecordingInverse(np.eye(n) * 2.0)
    S = MatrixOperator(0.3 * np.eye(n))
    si = SourceIteration(A_inv, S, max_iter=50, tol=1e-12)
    psi, record = si.solve(np.zeros(n))
    history = _trajectory(record)
    np.testing.assert_array_equal(np.asarray(psi), 0.0)
    if history != [0.0]:
        pytest.fail(f"zero-source history {history} — expected [0.0]")
    if not np.isfinite(history).all():
        pytest.fail("the q≈0 guard leaked a non-finite residual")


@pytest.mark.foundation
@pytest.mark.catches("ERR-053")
def test_the_gmres_nonconvergence_warning_is_ESCALATABLE(monkeypatch) -> None:
    r"""#340 R3 — the tree's only non-convergence announcement from inside
    ``numerics`` must answer to the PUBLISHED escalation flag.

    ``ESCALATION_FLAG`` is
    ``-W error::orpheus.numerics.convergence.ConvergenceWarning``, and a
    filter names a category: a bare ``RuntimeWarning`` does not match it.
    So until 2026-08-10 a CI run could be configured to make truncation
    fatal, PASS, and still have swallowed this one — a gate not covering
    what its own recipe claims, which is the defect class #340 exists to
    remove.

    ⭐ The teeth are in the SECOND leg, and only there.  Asserting that the
    warning is caught by a ``RuntimeWarning`` filter is satisfied by the
    pre-fix code (``ConvergenceWarning`` IS a ``RuntimeWarning``), so that
    leg cannot discriminate and is not written.  Asserting the category is
    ``ConvergenceWarning`` reds the moment anyone widens it back — which is
    the regression this row exists to catch, and the one a reviewer would
    wave through as "it's only a warning category".

    The narrowing is one-directional and safe by construction: every
    existing consumer filters on the BASE class (``_krylov_warnings`` above
    matches ``issubclass(w.category, RuntimeWarning)``), so it keeps
    matching.  Widening back would break only this row.
    """
    import orpheus.numerics.iteration as iteration_mod
    from orpheus.numerics.convergence import ConvergenceWarning

    monkeypatch.setattr(
        iteration_mod.spla, "gmres",
        _stub_gmres_returning(info=7, pr_norm_tail=0.37),
    )
    A_op, b = _singular_consistent()

    def _fresh_krylov():
        return KrylovAcceleration(
            A_op, preconditioner=lambda q: q,
            tol=1e-12, max_iter=30, restart=2,
        )

    # leg 1 — the category IS the escalatable one, not merely a RuntimeWarning.
    with pytest.warns(ConvergenceWarning) as caught:
        _fresh_krylov().solve(b)
    assert any("ERR-053" in str(w.message) for w in caught), (
        "the escalatable warning must still carry its ERR-053 pointer — the "
        "category is what CI filters on, the pointer is what a human follows"
    )

    # leg 2 — the escalation the published flag performs, done in-process:
    # the SAME solve must RAISE once that category is an error.  This is the
    # leg with teeth; it reds on any widening of the category.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        with pytest.raises(ConvergenceWarning):
            _fresh_krylov().solve(b)


# ───────────────────────────────────────────────────────────────────────
# ERR-079 / #349 — the budget and the trajectory count the SAME thing
# ───────────────────────────────────────────────────────────────────────
#
# ⭐ These are the gates whose ABSENCE let ERR-079 ship.  `[M]` 2026-08-13:
# every one of the 58 `IterationRecord`s in the suite was hand-built, so no
# gate had ever read `budget` / `n_iterations` / `exhausted_budget` off a
# record produced by a real `KrylovAcceleration.solve` — including the
# fixture LABELLED `inner(gmres)`, which omits `iterations_run` and so does
# not even reproduce the producer's shape.  A synthetic record cannot catch a
# producer's units error, because the test author picks both numbers.


@pytest.mark.foundation
@pytest.mark.catches("ERR-079")
def test_a_CONVERGED_gmres_solve_did_not_exhaust_its_budget():
    """A healthy Krylov solve must not report that it ran out (#349).

    ⛔ The defect, verbatim: ``exhausted_budget`` was
    ``n_iterations >= budget`` over a pair in different units — scipy's
    ``maxiter`` counts restart CYCLES while the ``pr_norm`` callback fires per
    inner ARNOLDI STEP.  Raising the knob did not help, because the two
    numbers were never commensurable.

    ⭐ The fixture is sized so the OLD spelling would say the opposite, and
    that precondition is ASSERTED rather than assumed — a row that silently
    stopped discriminating would otherwise go on reading green forever
    (``vv`` #19: only the reading under the wrong structure carries
    information).  `[M]` 30 distinct eigenvalues need 29 Arnoldi steps
    against a budget of 5 cycles, so ``n_iterations >= limit`` holds by a
    factor of ~6 while the honest ceiling is 150.
    """
    n = 30
    A = MatrixOperator(np.diag(np.linspace(1.0, 10.0, n)), can_solve=True)
    max_iter, restart = 5, 30

    krylov = KrylovAcceleration(
        A, ZeroOperator(), preconditioner=lambda q: q,
        max_iter=max_iter, tol=1e-12, restart=restart,
    )
    _, record = krylov.solve(np.ones(n))
    budget = record.budget

    # The producer states the exchange rate, and it is scipy's own.
    assert budget.iterations_per_unit == min(restart, n)
    assert budget.limit == max_iter
    assert budget.in_iterations == max_iter * min(restart, n)

    # ⚠ The anti-dud precondition: without this, the row below could pass on a
    # fixture that never entered the defect's regime at all.
    assert record.n_iterations >= budget.limit, (
        f"fixture no longer discriminates — it ran "
        f"{record.n_iterations} steps against a limit of {budget.limit}, so "
        f"the retired `n_iterations >= budget` spelling would have agreed "
        f"with the correct one and this gate would be vacuous"
    )

    assert record.converged is True
    assert record.exhausted_budget is False, (
        "a solve whose criterion cleared did not stop because it ran out"
    )
    assert record.truncated is False
    assert record.status == "CONVERGED"


@pytest.mark.foundation
@pytest.mark.catches("ERR-079")
def test_a_gmres_solve_that_really_DID_run_out_still_says_so():
    """The negative leg — without it, ``exhausted_budget = False`` passes.

    ``vv`` #11: the row above validates that the property stops crying wolf;
    only this one validates that it can still cry.  ``restart=1`` collapses
    the cycle and the step to the same thing, so the honest ceiling IS the
    knob and a hard system exhausts it.
    """
    n = 30
    rng_local = np.random.default_rng(0)
    hard = np.eye(n) + 0.9 * rng_local.standard_normal((n, n)) / np.sqrt(n)
    A = MatrixOperator(hard, can_solve=True)
    max_iter = 4

    krylov = KrylovAcceleration(
        A, ZeroOperator(), preconditioner=lambda q: q,
        max_iter=max_iter, tol=1e-14, restart=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, record = krylov.solve(np.ones(n))

    assert record.budget.iterations_per_unit == 1
    assert record.budget.in_iterations == max_iter
    assert record.n_iterations == max_iter
    assert record.converged is False
    assert record.exhausted_budget is True
    assert record.truncated is True
    assert record.status == "TRUNCATED"


@pytest.mark.foundation
@pytest.mark.catches("ERR-079")
def test_the_advice_names_a_setting_in_the_KNOBs_units_not_the_trajectorys():
    """*"set max_inner=N"* must be typeable into ``max_inner`` (#349).

    ``projected_iterations`` fits the observed rate over the TRAJECTORY, so
    its answer is in Arnoldi steps; the knob takes cycles.  Before the fix
    both halves of *"needs about N iterations: set X=N"* printed the same N,
    which on this arm over-states the required setting by ``restart``.

    The two numbers below must DIFFER for the gate to mean anything, so that
    is asserted too — on a unit-consistent producer they coincide by design
    and the row would be a tautology.
    """
    n = 30
    rng_local = np.random.default_rng(1)
    hard = np.eye(n) + 0.9 * rng_local.standard_normal((n, n)) / np.sqrt(n)
    A = MatrixOperator(hard, can_solve=True)

    krylov = KrylovAcceleration(
        A, ZeroOperator(), preconditioner=lambda q: q,
        max_iter=2, tol=1e-14, restart=8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, record = krylov.solve(np.ones(n))

    projected = record.projected_iterations()
    assert projected is not None, "the fixture must be projectable"
    setting = record.budget.covering(projected)

    assert setting != projected, (
        "this arm's knob and trajectory are in different units, so a gate "
        "where the two coincide is not testing the conversion"
    )
    # The recommendation must actually BUY what was projected...
    assert setting * record.budget.iterations_per_unit >= projected
    # ...and not a cycle more than needed.
    assert (setting - 1) * record.budget.iterations_per_unit < projected
