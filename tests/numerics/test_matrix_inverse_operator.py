r"""Step-5 gates: ``as_matrix()`` + ``MatrixInverseOperator`` (#226 taxonomy §12 step 5).

Spec: ``issue_226_inverse_operator_verification.md`` PART IV §28–§30.

**Claim layers.** ``as_matrix()`` value gates are closed-form (hand-built
matrices for ``Diagonal``/``Permutation``; the STRUCTURALLY-INDEPENDENT
``dense_per_material`` storage transpose for the energy leaves — §28.2
spells the independence argument). ``MatrixInverseOperator`` is a DIRECT
dense inverse: its tolerances are **machine·cond** grain (NOT driver-tol,
NOT nulp) — and per §27.A the machine grain IS the name-earner, proven
DISTINGUISHING by the in-gate Green contrast (an iterative inverse
satisfies M-materialise only to its driver tolerance).

**Config-blindness (§0.6).** Non-uniform non-±1 ``c`` (uniform is blind to
per-column scaling); a NON-symmetric 4-cycle permutation and asymmetric-
``SigS`` ≥2G mixture (symmetric matrices are blind to the transpose-
assembly mutation M-ASM-TRANSPOSE — the Mode-6 ``SigSᵀ`` class); a
``(2,3)`` index-stamp basis for the C-order convention (a single-column
``(n,1)`` basis has identical C/F enumeration and is blind to
M-ASM-RAVEL); a TRUE ``0.0`` singular coefficient (``1e-300`` passes
``lu_factor``); a non-zero junk seed ≠ q (a zero seed cannot red
M-MINV-SEED-CONSUME).

**Mode-8.** Every gate asserts via ``np.testing.*`` / ``pytest.raises``
(fire under the canonical ``python -O``).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from orpheus.numerics.matrix_inverse_operator import MatrixInverseOperator
from orpheus.numerics.operator import (
    DiagonalOperator,
    LinearOperator,
    MatrixTooLarge,
    NotInvertible,
    PermutationOperator,
    ScaledOperator,
)
from orpheus.numerics.space import FunctionSpace

pytestmark = pytest.mark.foundation

_RNG = np.random.default_rng(285)
_EPS = np.finfo(float).eps

# Non-uniform, non-±1 coefficient (cond = max/min = 10) — §0.6.
_C4 = np.array([2.0, 5.0, 0.5, 3.0])
_COND4 = _C4.max() / _C4.min()
# 4-cycle: NON-involution, NON-symmetric gather (transpose-blind guard).
_P4 = np.array([1, 2, 3, 0])


# ───────────────────────────────────────────────────────────────────────
# Test-local operators (§35.1) — the ndarray-carrier fixtures the
# production tree does not supply (every production ndarray leaf is
# domain=None; see the §29.1 honest-scope note).
# ───────────────────────────────────────────────────────────────────────


class _DenseActionOperator(LinearOperator):
    """``apply(x) = M @ x.ravel()`` — a fixed dense action, APPLY-only.

    ``is_invertible`` stays the base ``False`` — the §30.7 witness's
    non-invertible leading leaf, and the §28.4 rectangular fixture.
    """
    # S4-amendment: the base DEMANDS an answer from every subclass; this
    # double is a deliberately-unbound probe, so it DECLARES the unbound
    # state instead of inheriting a silent default (which no longer exists).
    domain = None
    codomain = None

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=float)

    def apply(self, x, /):
        return self.matrix @ np.asarray(x).ravel()


class _SpaceCarrying(LinearOperator):
    """Identity action CARRYING a domain :class:`FunctionSpace`.

    The §29 resolution fixture: every production ndarray leaf is
    ``domain=None`` (verified — the domain-default arm has NO production
    exerciser in step 5; this test-local operator tests the RULE
    faithfully; honest-scope §35.3).
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._space = FunctionSpace("test_space", shape)

    @property
    def domain(self):
        return self._space

    @property
    def codomain(self):
        return self._space

    def apply(self, x, /):
        return np.asarray(x)


class _IndexStampOperator(LinearOperator):
    """``(2,3) → (2,3)`` endomorphism with the all-distinct C-raveled stamp
    ``M6[i, j] = 10·i + j + 1`` — NON-symmetric, so any C↔F swap in the
    basis enumeration OR the output ravel permutes the matrix and is
    O(1)-visible (§28.3). A single-column ``(n, 1)`` basis would be
    BLIND (C and F enumeration coincide) — hence ``(2, 3)``.
    """
    # S4-amendment: the base DEMANDS an answer from every subclass; this
    # double is a deliberately-unbound probe, so it DECLARES the unbound
    # state instead of inheriting a silent default (which no longer exists).
    domain = None
    codomain = None

    SHAPE = (2, 3)

    def __init__(self) -> None:
        n = int(np.prod(self.SHAPE))
        self.stamp = 10.0 * np.arange(n)[:, None] + np.arange(n)[None, :] + 1.0

    def apply(self, x, /):
        return (self.stamp @ np.asarray(x).ravel()).reshape(self.SHAPE)


def _off_diagonal_nilpotent(n: int) -> np.ndarray:
    """Strictly-superdiagonal ones: nilpotent, so ``det(D − N) = ∏c ≠ 0``
    (the §30.7 witness sum is non-singular as a MATRIX by construction)."""
    N = np.zeros((n, n))
    N[np.arange(n - 1), np.arange(1, n)] = 1.0
    return N


def _asymmetric_2g_mat_xs():
    """2G mixture with ASYMMETRIC ``SigS`` AND non-zero asymmetric ``Sig2``
    on a unit one-cell carrier (Mode-6: a symmetric matrix is
    blind to the transpose-assembly mutation; zero ``Sig2`` would vacuum
    the ``IsotropicN2N`` oracle row)."""
    from orpheus.derivations.common.xs_library import make_mixture
    from tests.transport._carrier_helpers import unit_cell_carrier

    sig_s0 = np.array([[0.30, 0.08], [0.01, 0.45]])  # [g_from, g_to], asymmetric
    sig_2 = np.array([[0.020, 0.006], [0.001, 0.030]])  # asymmetric, non-zero
    m = make_mixture(
        sig_t=np.array([1.0, 1.5]), sig_c=np.array([0.01, 0.02]),
        sig_f=np.array([0.0, 0.0]), nu=np.array([0.0, 0.0]),
        chi=np.zeros(2), sig_s=sig_s0,
    )
    m.SigS = [csr_matrix(sig_s0)]
    m.Sig2 = [csr_matrix(sig_2)]
    mat_xs = unit_cell_carrier({0: m}).material_xs_field()
    return mat_xs, sig_s0, sig_2, np.array([1.0, 1.5])


# ───────────────────────────────────────────────────────────────────────
# §28 — as_matrix() L0 correctness
# ───────────────────────────────────────────────────────────────────────


def test_as_matrix_diagonal_exact():
    """§28.1 ASM-STRUCTURED: apply-to-basis gathers single columns with no
    accumulation → BIT-exact against the hand-built matrix."""
    np.testing.assert_array_equal(
        DiagonalOperator(_C4).as_matrix(basis_shape=(4,)), np.diag(_C4),
        err_msg="as_matrix(Diagonal) ≠ diag(c)",
    )


def test_as_matrix_permutation_exact():
    """§28.1: the EXACT permutation-matrix convention the apply produces.

    ``(P x)_i = x[perm[i]]`` (a gather), so column j = apply(e_j) has its 1
    where ``perm[i] == j`` — the ROW-indexed ``np.eye(n)[perm, :]``. Pinning
    the exact convention (not just "a permutation matrix") is what makes
    M-ASM-TRANSPOSE bite on this NON-symmetric 4-cycle.
    """
    np.testing.assert_array_equal(
        PermutationOperator(_P4).as_matrix(basis_shape=(4,)), np.eye(4)[_P4, :],
        err_msg="as_matrix(Permutation) ≠ the gather matrix I[perm, :]",
    )


def test_as_matrix_energy_leaves_vs_storage_oracle():
    """§28.2 ASM-ORACLE — structural independence, spelled out.

    ``as_matrix(basis_shape=(ng,1))`` drives ``apply`` → the
    ``add_p0_source`` einsum kernel of either channel (per-column matvec
    accumulation over g'); ``dense_per_material()[mid]`` reads
    ``sig_s_legendre(mid)[0].T`` — a direct storage TRANSPOSE-copy. Neither
    computes the other: they agree ONLY if the apply is faithful to the
    stored cross sections — exactly the L0 claim. The mixture is ≥2G with
    ASYMMETRIC SigS/Sig2 (Mode-6: symmetric would null the transpose
    mutation).
    """
    from orpheus.transport.operators.isotropic_transfer import (
        IsotropicN2N,
        IsotropicScattering,
    )
    from orpheus.transport.operators.multiplication_operator import (
        MultiplicationOperator,
    )

    mat_xs, sig_s0, sig_2, sig_t = _asymmetric_2g_mat_xs()
    ng = 2
    for op in (
        IsotropicScattering.from_material_xs(
            mat_xs, space=mat_xs.mesh.bulk_space,
        ),
        IsotropicN2N.from_material_xs(
            mat_xs, space=mat_xs.mesh.bulk_space,
        ),
    ):
        got = op.as_matrix(basis_shape=(ng, 1))
        ref = op.dense_per_material()[0]  # the single material
        np.testing.assert_allclose(
            got, ref, rtol=0, atol=4 * _EPS * np.abs(ref).max(),
            err_msg=f"{type(op).__name__}: apply-to-basis ≠ storage transpose",
        )
    # The OperatorSum path — C − K_iso materialises as the fused storage oracle.
    loss = MultiplicationOperator.from_mesh(
        mat_xs.total_cross_section_field, mat_xs.mesh,
    ) - (
        IsotropicScattering.from_material_xs(
            mat_xs, space=mat_xs.mesh.bulk_space,
        )
        + IsotropicN2N.from_material_xs(
            mat_xs, space=mat_xs.mesh.bulk_space,
        )
    )
    ref_sum = np.diag(sig_t) - (sig_s0 + 2.0 * sig_2).T
    np.testing.assert_allclose(
        loss.as_matrix(basis_shape=(ng, 1)), ref_sum, atol=1e-12,
        err_msg="OperatorSum as_matrix ≠ diag(Σt) − (Σs0 + 2Σ2)ᵀ",
    )


def test_as_matrix_c_order_column_convention():
    """§28.3 ASM-COLUMN: flat index i (C-order over basis_shape) → column
    j = i, output raveled C-order into the rows — pinned on the (2,3)
    all-distinct stamp where any C↔F swap permutes the matrix."""
    A = _IndexStampOperator()
    np.testing.assert_array_equal(
        A.as_matrix(basis_shape=(2, 3)), A.stamp,
        err_msg="as_matrix column/row convention (C-order) is transposed",
    )


def test_as_matrix_rectangular_output_emerges_from_apply():
    """§28.4 ASM-RECTANGULAR: a (2,)→(3,) action materialises to (3,2) —
    as_matrix never assumes square; the output dim comes from apply."""
    M = _RNG.standard_normal((3, 2))
    got = _DenseActionOperator(M).as_matrix(basis_shape=(2,))
    assert got.shape == (3, 2), f"rectangular as_matrix shape {got.shape} ≠ (3, 2)"
    np.testing.assert_allclose(got, M, atol=1e-14, err_msg="rectangular as_matrix ≠ M")


def test_as_matrix_equals_retired_as_dense_loop():
    """HOMOG-EQUIV (#226 step 5, spec §32.2): the promoted ``as_matrix``
    reproduces the retired ``_as_dense`` apply-to-basis loop BYTE-for-byte
    on the homogeneous solver's own loss composition — the retirement is a
    pure relocation, not a re-derivation (the fuller-view-oracle
    discipline: the retired loop lives on here as the test-local
    reference, guarding a later ``as_matrix`` refactor from silently
    moving the homogeneous basis columns). Keep through the merge cycle,
    then reassess.

    HOME delta vs spec §35.1: hosted HERE (foundation-marked) rather than
    in ``test_homogeneous.py``, whose file-level ``l1 + verifies(...)``
    marks would have written FALSE equation-TESTS edges for this pure
    software invariant (the registry warned on the marker conflict).
    """
    from orpheus.derivations import get
    from tests.transport._carrier_helpers import unit_cell_carrier
    from orpheus.transport.operators.isotropic_transfer import (
        IsotropicN2N,
        IsotropicScattering,
    )
    from orpheus.transport.operators.multiplication_operator import (
        MultiplicationOperator,
    )

    case = get("homo_2eg_n2n")  # asymmetric SigS + non-zero Sig2 (Mode-6)
    mix = next(iter(case.materials.values()))
    mat_xs = unit_cell_carrier({0: mix}).material_xs_field()
    ng = mix.ng
    loss = MultiplicationOperator.from_mesh(
        mat_xs.total_cross_section_field, mat_xs.mesh,
    ) - (
        IsotropicScattering.from_material_xs(
            mat_xs, space=mat_xs.mesh.bulk_space,
        )
        + IsotropicN2N.from_material_xs(
            mat_xs, space=mat_xs.mesh.bulk_space,
        )
    )
    # local oracle = the retired _as_dense loop, verbatim
    cols = []
    for i in range(ng):
        e_i = np.zeros((ng, 1))
        e_i[i, 0] = 1.0
        cols.append(np.asarray(loss.apply(e_i)).ravel())
    oracle = np.column_stack(cols)
    np.testing.assert_array_equal(
        loss.as_matrix(basis_shape=(ng, 1)), oracle,
        err_msg="as_matrix diverged from the retired _as_dense loop",
    )


# ───────────────────────────────────────────────────────────────────────
# §29 — basis-shape resolution + the MatrixTooLarge boundary
# ───────────────────────────────────────────────────────────────────────


def test_basis_shape_explicit_wins_over_domain():
    """§29.1: explicit basis_shape OVERRIDES a carried domain (the shapes
    deliberately differ so M-ASM-RESOLVE — always-domain — reddens)."""
    A = _SpaceCarrying(shape=(3,))
    assert A.as_matrix(basis_shape=(6,)).shape == (6, 6)


def test_basis_shape_domain_default():
    """§29.1: no explicit shape → resolved from ``domain.shape`` (a shape
    that differs from any hard-coded default, proving the space is READ)."""
    assert _SpaceCarrying(shape=(3,)).as_matrix().shape == (3, 3)


def test_basis_shape_none_domain_raises_valueerror():
    """§29.1/§27.C: NEITHER domain NOR basis_shape → ``ValueError`` naming
    both remedies. Class-discriminated: this is the ILL-POSED arm
    (``ValueError``), NOT the resource-refused arm (``MatrixTooLarge``)."""
    bare = DiagonalOperator(np.array([1.0, 2.0]))
    assert bare.domain is None  # fixture verification: a genuine None-domain leaf
    with pytest.raises(ValueError, match="basis_shape") as excinfo:
        bare.as_matrix()
    assert not isinstance(excinfo.value, MatrixTooLarge)


def test_at_threshold_materialises():
    """§29.2 designed-GREEN control: prod == max_dimension PASSES (strict >).
    M-ASM-GATE-OFFBYONE (>=) reds THIS gate — the off-by-one tooth."""
    got = _SpaceCarrying(shape=(4,)).as_matrix(basis_shape=(4,), max_dimension=4)
    assert got.shape == (4, 4)


def test_one_above_threshold_raises_matrix_too_large():
    """§29.2/§27.C: the resource-refused arm — ``MatrixTooLarge``, not a
    generic ValueError."""
    with pytest.raises(MatrixTooLarge, match="max_dimension=4"):
        _SpaceCarrying(shape=(5,)).as_matrix(basis_shape=(5,), max_dimension=4)


def test_default_gate_refuses_above_4096():
    """§29.2: the DEFAULT gate (4096) refuses a 4097-element basis — the
    refusal is a size precheck (raises BEFORE any apply, so this is cheap)."""
    with pytest.raises(MatrixTooLarge, match="4096"):
        _SpaceCarrying(shape=(4097,)).as_matrix()


def test_per_call_max_dimension_tightens_and_lifts():
    """§29.2: max_dimension is a PER-CALL resource knob — both directions."""
    A = _SpaceCarrying(shape=(5,))
    with pytest.raises(MatrixTooLarge):
        A.as_matrix(max_dimension=4)  # tighten below the size → refuse
    assert A.as_matrix(max_dimension=5).shape == (5, 5)  # lift exactly to size


# ───────────────────────────────────────────────────────────────────────
# §30 — MatrixInverseOperator: universal contract + the M-invariants
# ───────────────────────────────────────────────────────────────────────


def test_minv_roundtrip_and_closed_form():
    """§30.1 MINV-I1: round-trip both ways + the HAND-built closed form
    (diag(1/c) — structurally independent of as_matrix; §30.0 caveat), at
    machine·cond grain."""
    A = DiagonalOperator(_C4)
    minv = MatrixInverseOperator(A, basis_shape=(4,))
    q = _RNG.standard_normal(4)
    tol = 32 * _EPS * _COND4
    np.testing.assert_allclose(
        minv.apply(A.apply(q)), q, atol=tol, err_msg="MINV: A⁻¹(Ax) ≠ x")
    np.testing.assert_allclose(
        A.apply(minv.apply(q)), q, atol=tol, err_msg="MINV: A(A⁻¹x) ≠ x")
    np.testing.assert_allclose(
        minv.apply(q), q / _C4, atol=tol,
        err_msg="MINV ≠ its closed-form inverse (diag(1/c))")


def test_minv_involution_is_object_identity():
    """§30.2 MINV-I2: (A⁻¹)⁻¹ IS the wrapped forward (mixin back-half)."""
    A = DiagonalOperator(_C4)
    assert MatrixInverseOperator(A, basis_shape=(4,)).inverse() is A


def test_minv_materialise_at_machine_grain_with_green_contrast():
    """§30.3 MINV-MATERIALISE — the name-earner (§27.A sharpened).

    ``as_matrix()`` is a UNIVERSAL base method, so an iterative Green also
    satisfies ``G·A ≈ I`` — to its DRIVER tolerance. What earns the
    ``MatrixInverseOperator`` name is the MACHINE·cond grain (the batched
    LU backsolve has no iteration floor). The Green CONTRAST proves the
    invariant is DISTINGUISHING, not merely satisfied; its sanity leg
    (driver-tol satisfaction) is a designed-green proving the Green works.
    """
    A = DiagonalOperator(_C4)
    minv = MatrixInverseOperator(A, basis_shape=(4,))
    mach = 64 * _EPS * _COND4
    Ainv_mat, A_mat = minv.as_matrix(), A.as_matrix(basis_shape=(4,))
    np.testing.assert_allclose(
        Ainv_mat @ A_mat, np.eye(4), atol=mach,
        err_msg="M-materialise: A⁻¹·A ≠ I at machine precision")
    np.testing.assert_allclose(
        A_mat @ Ainv_mat, np.eye(4), atol=mach,
        err_msg="M-materialise: A·A⁻¹ ≠ I at machine precision")
    np.testing.assert_allclose(
        Ainv_mat, np.diag(1.0 / _C4), atol=mach,
        err_msg="MINV.as_matrix() ≠ diag(1/c)")
    # The DISTINGUISHING contrast: an iterative Green on a convergent
    # splitting (ρ ≈ 0.3/min(c) = 0.6) inherits the base apply-to-basis
    # as_matrix — n driver solves at driver-tol, NOT machine grain.
    from orpheus.numerics.green_operator import GreenOperator

    sum_op = A - ScaledOperator(0.3, PermutationOperator(_P4))
    green = sum_op.inverse()
    assert isinstance(green, GreenOperator)  # the §24.4 dispatch narrow
    green_resid = np.abs(
        green.as_matrix(basis_shape=(4,)) @ sum_op.as_matrix(basis_shape=(4,))
        - np.eye(4)
    ).max()
    mach_sum = 64 * _EPS * np.linalg.cond(sum_op.as_matrix(basis_shape=(4,)))
    assert green_resid > mach_sum, (
        f"the Green as_matrix met MACHINE grain ({green_resid:.2e}) — "
        f"the name-earner is not distinguishing")
    assert green_resid < 1e3 * green.tol, (  # designed-green sanity: Green WORKS
        f"Green residual {green_resid:.2e} ≫ driver-tol — broken Green, "
        f"not a distinguishing gap")


def test_minv_direct_residual_and_seed_bit_identity():
    """§30.4 MINV-DIRECT: machine·cond true residual + BIT-identical output
    under a junk seed (M-direct IS seed-independence; array_equal, not
    allclose — the correct ``del initial_guess`` is byte-exact). The second
    leg runs a NON-symmetric inner so M-MINV-LUTRANS (trans=1 solves Aᵀx=q)
    reds HERE, per the §34 activating-config column."""
    q = _RNG.standard_normal(4)
    junk = _RNG.standard_normal(4) * 1e6  # non-zero, ≠ q (§0.6)

    A = DiagonalOperator(_C4)
    minv = MatrixInverseOperator(A, basis_shape=(4,))
    resid = np.linalg.norm(np.asarray(A.apply(minv.apply(q))).ravel() - q)
    assert resid <= 32 * _EPS * _COND4 * np.linalg.norm(q), (
        f"M-direct residual {resid:.2e} exceeds machine·cond")
    np.testing.assert_array_equal(
        minv.apply(q), minv.apply(q, initial_guess=junk),
        err_msg="MINV.apply consumed the seed — M-direct is NOT seed-independent")

    # NON-symmetric leg (D − N, N nilpotent): a transposed backsolve is
    # O(1)-wrong here while invisible on the symmetric diagonal above.
    hand = np.diag(_C4) - _off_diagonal_nilpotent(4)
    B = _DenseActionOperator(hand)
    binv = MatrixInverseOperator(B, basis_shape=(4,))
    resid_ns = np.linalg.norm(hand @ np.asarray(binv.apply(q)).ravel() - q)
    assert resid_ns <= 64 * _EPS * np.linalg.cond(hand) * np.linalg.norm(q), (
        f"M-direct (non-symmetric) residual {resid_ns:.2e} exceeds machine·cond")


def test_minv_backhalf_solve_is_the_forward_matvec():
    """§30.5 MINV-BACKHALF: mixin ``solve`` on the inverse = the FORWARD
    action, anchored on the hand form c⊙x (never via inverse().apply — a
    definition tautology)."""
    minv = MatrixInverseOperator(DiagonalOperator(_C4), basis_shape=(4,))
    x = _RNG.standard_normal(4)
    np.testing.assert_allclose(
        np.asarray(minv.solve(x)).ravel(), _C4 * x, rtol=1e-14,
        err_msg="mixin solve ≠ the forward matvec")


def test_minv_non_square_raises():
    """§30.6: the ctor's OWN squareness guard — matched on its domain-language
    message ("SQUARE materialization"), NOT scipy's generic "expected square
    matrix", so M-MINV-NOGUARD-SQ (guard deleted → scipy raises instead)
    reddens THIS gate."""
    A = _DenseActionOperator(_RNG.standard_normal((3, 2)))  # (2,)→(3,)
    with pytest.raises(ValueError, match="SQUARE materialization"):
        MatrixInverseOperator(A, basis_shape=(2,))


def test_minv_singular_raises_linalgerror():
    """§30.6: a TRUE 0.0 coefficient (1e-300 would pass) — the zero-LU-pivot
    check raises ``LinAlgError`` at CONSTRUCTION, never an inf/nan solve."""
    with pytest.raises(np.linalg.LinAlgError, match="exactly singular"):
        MatrixInverseOperator(
            DiagonalOperator(np.array([2.0, 0.0, 0.5])), basis_shape=(3,))


def test_minv_too_large_propagates():
    """§30.6: the ctor's eager materialization propagates MatrixTooLarge."""
    with pytest.raises(MatrixTooLarge):
        MatrixInverseOperator(_SpaceCarrying(shape=(4097,)))


def test_minv_nonsingular_constructs():
    """§30.6 designed-GREEN POSITIVE control (anti-#11: guards need a
    correct instance that does NOT raise)."""
    MatrixInverseOperator(DiagonalOperator(_C4), basis_shape=(4,))


def test_minv_as_matrix_override_honors_base_contract():
    """§11.4: the override accepts the base kwargs — a consistent explicit
    basis_shape passes; a contradicting one raises; a tighter per-call
    max_dimension still gates."""
    minv = MatrixInverseOperator(DiagonalOperator(_C4), basis_shape=(4,))
    np.testing.assert_allclose(
        minv.as_matrix(basis_shape=(4,)), np.diag(1.0 / _C4),
        atol=64 * _EPS * _COND4)
    with pytest.raises(ValueError, match="contradicts"):
        minv.as_matrix(basis_shape=(5,))
    with pytest.raises(MatrixTooLarge):
        minv.as_matrix(max_dimension=3)


def test_matrix_inverts_what_green_refuses():
    """§30.7 MINV-WITNESS — VALUE-vs-STRUCTURE (the §3 strategy-override
    witness; ndarray analog of (−S)+(L+C), §27.B — the FullField original
    is the motivation, this split the realisable proof).

    The sum's LEFT-SPINE HEAD ``ScaledOperator(−1, S_ao)`` is non-invertible
    (S_ao is apply-only), so the STRUCTURAL factory refuses — Green's
    splitting has no preconditioner. The MATRIX realization reads VALUES:
    ``D − S_ao`` is non-singular (S_ao strictly-superdiagonal nilpotent ⇒
    det = ∏c), and the dense strategy inverts it, anchored on the
    structurally-independent hand matrix.
    """
    S_ao = _DenseActionOperator(_off_diagonal_nilpotent(4))
    assert not S_ao.is_invertible  # fixture verification (else vacuous)
    D = DiagonalOperator(_C4)
    sum_op = (-1.0 * S_ao) + D
    assert not sum_op.is_invertible
    with pytest.raises(NotInvertible, match="canonical ordering"):
        sum_op.inverse()  # Green REFUSES (leading term not invertible)
    minv = MatrixInverseOperator(sum_op, basis_shape=(4,))
    q = _RNG.standard_normal(4)
    A_hand = np.diag(_C4) - _off_diagonal_nilpotent(4)
    np.testing.assert_allclose(
        np.asarray(minv.apply(q)).ravel(), np.linalg.solve(A_hand, q), atol=1e-12,
        err_msg="MatrixInverse did not invert the leading-non-invertible sum")


# ───────────────────────────────────────────────────────────────────────
# S5 (step 5) — the transpose backsolve on the SAME LU factors
# ───────────────────────────────────────────────────────────────────────
#
# ``apply_transpose = lu_solve(lu, b, trans=1)`` is the arm the
# operator-algebra swap law rides (#280 R5/R11): ``grid.H.inverse()``
# exists only because ``grid.inverse()`` is adjointable
# (``AdjointOperator.is_invertible`` clause 2). Mode-12 mandate: every
# value row runs on an ASYMMETRIC matrix — a symmetric fixture is blind
# to a forgotten ``trans`` flag (``A⁻¹b ≡ A⁻ᵀb`` there).


def _asymmetric_5x5() -> np.ndarray:
    rng = np.random.default_rng(20260712)
    matrix = rng.random((5, 5)) + 4.0 * np.eye(5)
    assert not np.allclose(matrix, matrix.T)  # fixture verification
    return matrix


def test_apply_transpose_matches_the_dense_transpose_solve():
    """``A⁻ᵀb`` against the structurally-independent ``np.linalg.solve(Aᵀ, b)``
    (a FRESH factorization of the transpose vs the stored-factor
    ``trans=1`` backsolve — reduction-tree independent)."""
    matrix = _asymmetric_5x5()
    minv = MatrixInverseOperator(
        _DenseActionOperator(matrix), basis_shape=(5,),
    )
    b = np.random.default_rng(3).standard_normal(5)
    np.testing.assert_allclose(
        np.asarray(minv.apply_transpose(b)).ravel(),
        np.linalg.solve(matrix.T, b),
        rtol=1e-12, atol=1e-14,
        err_msg="the trans=1 backsolve is off the dense-Aᵀ reference",
    )


def test_apply_transpose_is_not_apply_on_an_asymmetric_matrix():
    """The forget-``trans`` discriminator: a mutation returning ``A⁻¹b``
    from ``apply_transpose`` sits O(1) off the transpose reference on the
    asymmetric fixture (and EXACTLY on it on a symmetric one — the
    Mode-12 reason the fixture asserts its own asymmetry)."""
    matrix = _asymmetric_5x5()
    minv = MatrixInverseOperator(
        _DenseActionOperator(matrix), basis_shape=(5,),
    )
    b = np.random.default_rng(4).standard_normal(5)
    forward = np.asarray(minv.apply(b)).ravel()
    transposed = np.asarray(minv.apply_transpose(b)).ravel()
    if np.allclose(forward, transposed, rtol=1e-6):
        pytest.fail(
            "apply and apply_transpose coincide on an asymmetric matrix — "
            "the trans flag is not reaching the backsolve"
        )


def test_transpose_face_advertised_and_seed_free():
    """``is_adjointable`` is unconditionally True (a materialized matrix
    always transposes — same LU factors, no re-factorization), closing the
    swap-law precondition ``adjointable(grid.inverse())``."""
    minv = MatrixInverseOperator(
        _DenseActionOperator(_asymmetric_5x5()), basis_shape=(5,),
    )
    if not minv.is_adjointable:
        pytest.fail("MatrixInverseOperator must advertise its free transpose")
    b = np.random.default_rng(5).standard_normal(5)
    np.testing.assert_array_equal(  # M-direct seed-independence, both faces
        np.asarray(minv.apply(b, initial_guess=b)),
        np.asarray(minv.apply(b)),
    )


def test_typed_operand_without_a_space_zeros_exemplar_raises():
    """A ravellable (typed) operand needs the carrier space's zeros()
    exemplar to mint the typed result — absent, the seam refuses loudly
    instead of guessing a template (role honesty: a solution is a state,
    never the rhs's source role)."""

    class _RavelToy:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=float)

        def to_flat(self):
            return self.values.ravel()

        @classmethod
        def from_flat(cls, flat, template):
            del template
            return cls(flat)

    minv = MatrixInverseOperator(
        _DenseActionOperator(_asymmetric_5x5()), basis_shape=(5,),
    )
    with pytest.raises(ValueError, match=r"zeros\(\) exemplar"):
        minv.apply(_RavelToy(np.ones(5)))
    with pytest.raises(ValueError, match=r"zeros\(\) exemplar"):
        minv.apply_transpose(_RavelToy(np.ones(5)))
