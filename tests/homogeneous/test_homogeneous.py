"""Verify the infinite-medium eigenvalue solver against SymPy analytical solutions."""

import dataclasses

import numpy as np
import pytest

import orpheus.numerics.eigenvalue as _eig
from orpheus.derivations import get
from orpheus.homogeneous.solver import solve_homogeneous_infinite

# File-level verifies marker: every test in this file exercises the
# homogeneous eigenvalue chain end-to-end by asserting k_inf matches
# the SymPy-derived analytical reference to 1e-12. That tolerance is
# tight enough to pin every step of the derivation — if any of the
# labelled equations below were implemented incorrectly the k mismatch
# would be far larger than 1e-12, so a passing test is equation-level
# (L1) verification for every link in the chain.
#
# Declared explicitly here (rather than inherited from
# VerificationCase) so the Nexus AST pass picks it up via decorator
# parsing and writes TESTS edges.
#
# The 2G labels (two-group-*) and the power-iteration step labels
# (fission-source, fixed-source-solve, keff-update) are all exercised
# by the homo_2eg / homo_4eg cases — the analytical k_inf is derived
# symbolically via exactly those equations, so a solver k that matches
# to 1e-12 implies every link in the chain is correct. The absorption-
# xs label is the derived property used inside keff-update.
pytestmark = [pytest.mark.l1, pytest.mark.verifies(
    "one-group-kinf",
    "inf-hom-balance",
    "matrix-eigenvalue",
    "removal-matrix",
    "fission-matrix",
    "mg-balance",
    # B.1 additions (issue #87): the full 2G analytical chain and the
    # power-iteration step labels, all verified end-to-end by the
    # homo_2eg and homo_4eg parametric cases.
    "two-group-A",
    "two-group-F",
    "two-group-Ainv",
    "two-group-M",
    "two-group-charpoly",
    "two-group-roots",
    "fission-source",
    "fixed-source-solve",
    "keff-update",
    "absorption-xs",
)]


@pytest.mark.parametrize("case_name", [
    "homo_1eg",
    "homo_2eg",
    "homo_4eg",
    # Non-trivial, asymmetric (n,2n): de-vacuums the n2n-in-A convention.
    # Every other case has Sig2=0, so the 2·Σ₂ᵀ loss term is never
    # exercised; here it moves k_inf ~0.6 (a drop/double-count reds).
    "homo_2eg_n2n",
])
def test_kinf_exact(case_name):
    """Eigenvalue must match analytical solution to machine precision.

    The refounded solver assembles A = C − K_iso from the transport
    operators (a different FP reduction tree than the oracle's fused
    ``(Σ_s + 2Σ_2)ᵀ``), so the tolerance is now principled-equivalence
    (FP-non-associativity, ~1 ULP), not bit-identity — still ≪ 1e-12.
    """
    case = get(case_name)
    mix = next(iter(case.materials.values()))
    result = solve_homogeneous_infinite(mix)
    assert abs(result.k_inf - case.k_inf) < 1e-12, (
        f"k_inf mismatch: solver={result.k_inf:.10f} "
        f"analytical={case.k_inf:.10f}"
    )


@pytest.mark.verifies("normalisation")
def test_post_solve_production_rate_is_100():
    """L1: post-convergence flux is normalised to 100 n/cm^3/s production.

    After :func:`solve_homogeneous_infinite` solves, the flux is rescaled
    so the **fission** production rate

    .. math::

       \\nu\\Sigma_f \\cdot \\boldsymbol{\\phi} = 100

    (see Eq. ``normalisation`` in docs/theory/foundations/infinite_medium.rst).
    Production is :math:`\\nu\\Sigma_f` only — the (n,2n) reaction is a
    loss-side transfer folded into the loss matrix as
    :math:`2\\Sigma_2^T`, NOT a production channel (it does NOT enter
    the numerator).  The ``homo_2eg_n2n`` case (non-zero, asymmetric
    :math:`\\Sigma_2`) makes this non-vacuous: under the retired
    ``(\\Sigma_p + 2\\cdot\\text{colsum}(\\Sigma_2))`` formula the
    production would not equal 100.  The 1G case is a degenerate
    one-scalar normalisation a bug could accidentally satisfy, so it is
    excluded.
    """
    for case_name in ("homo_2eg", "homo_4eg", "homo_2eg_n2n"):
        case = get(case_name)
        mix = next(iter(case.materials.values()))
        result = solve_homogeneous_infinite(mix)

        # Production = νΣ_f @ φ  (fission only; n2n lives in A, not F).
        production = mix.SigP @ result.flux

        assert abs(production - 100.0) < 1e-9, (
            f"{case_name}: production rate = {production:.6e}, "
            f"expected 100.0 (normalisation constraint)"
        )


# ── Operator-algebra assembly: A-level oracle + Mode-11 liveness (#276) ──


@pytest.mark.verifies("removal-matrix")
def test_assemble_loss_operator_matches_fused_oracle():
    """The operator-composed A = C − K_iso, materialized apply-to-basis,
    matches the fused ``diag(Σ_t) − (Σ_s0 + 2Σ_2)ᵀ`` on the
    non-trivial-(n,2n) case.

    A SHARP procedural pin at the A level — it localises a sign/term/omission
    bug in the operator assembly faster than the end-to-end eig. It shares
    ``mat_xs`` data with the fused form (so it is NOT structurally
    independent), and therefore PAIRS with the SymPy ``case.k_inf`` anchor
    in :func:`test_kinf_exact` rather than replacing it.  (Step 5b: the
    assembly now returns the UN-materialized OperatorSum — in production the
    MatrixInverseOperator ctor densifies it; the oracle comparison here
    materializes explicitly.)
    """
    from orpheus.homogeneous.solver import HomogeneousProblem, _pose_space
    from orpheus.transport.mesh.material_mesh import MaterialMesh

    case = get("homo_2eg_n2n")
    mix = next(iter(case.materials.values()))
    mat_xs = MaterialMesh.from_materials({0: mix}).material_xs_field()

    # CS1 3b: bare as_matrix() — the shape derives from the threaded domain
    # (the explicit basis_shape idiom is retired on this path; D9 pins the
    # derivation). CS4a K2: the space is the mixture-minted pose.
    A = HomogeneousProblem(mix).loss.as_matrix()
    sig_t = mat_xs.total_cross_section[:, 0]
    sig_s0 = mat_xs.sig_s_legendre(0)[0]  # (ng, ng), [g_from, g_to]
    sig_2 = mat_xs.n2n_matrix(0)
    A_fused = np.diag(sig_t) - (sig_s0 + 2.0 * sig_2).T
    np.testing.assert_allclose(A, A_fused, atol=1e-12, rtol=0)


def test_kinf_gate_executes_the_plain_multiplier_assembly(monkeypatch):
    """Mode-11: the homogeneous k∞ gate actually EXECUTES the PLAIN-bound
    ``MultiplicationOperator``'s emission.

    The loss matrix ``A = C − K_iso`` is materialised through
    ``as_matrix``, which delegates to ``assemble()`` on an assemblable
    operator — and since CS4c step 5 the plain binding IS assemblable
    (the bulk diagonal on its own ends), so the collision diagonal reaches
    the k∞ gate through ``_bulk_assembly``, not through ``apply``
    (`[M]` 2026-09-04: perturbing ``apply`` alone left k_inf unmoved —
    this row's pre-step-5 spelling had gone vacuous-green). Perturbing
    ONLY the emission (×1.5 on the diagonal) moves k_inf O(1) — proving
    that path is on the gate's call graph and load-bearing.
    (``-O``-safe: the monkeypatch is an in-process attribute swap,
    reverted by the fixture; never a ``git checkout``.)

    ⚠ Coverage shift, stated: ``test_kinf_exact`` no longer witnesses
    ``MultiplicationOperator.apply`` at all (`[M]` 2026-09-05, perturbing
    ``apply`` ×1.5 over this tree reds two STRUCTURAL rows in
    ``test_operator_spaces.py`` and no k∞ value gate). The plain binding's
    ``apply`` is pinned by the 33-cell ends→body fence
    (``tests/transport/test_kernels.py``) and by those two rows — do not read
    the k∞ anchor as covering it.
    """
    from orpheus.transport.operators.multiplication_operator import (
        MultiplicationOperator,
    )

    raw = MultiplicationOperator.__dict__["_bulk_assembly"]

    def perturbed(self, bulk_space):
        emitted = raw.__get__(self, type(self))(bulk_space)
        return type(emitted)(
            emitted.matrix * 1.5, domain=emitted.domain, codomain=emitted.codomain,
        )

    monkeypatch.setattr(MultiplicationOperator, "_bulk_assembly", perturbed)

    case = get("homo_2eg_n2n")
    mix = next(iter(case.materials.values()))
    result = solve_homogeneous_infinite(mix)
    assert abs(result.k_inf - case.k_inf) > 1e-3, (
        f"perturbing the plain M[Σ_t] emission left k_inf at {result.k_inf:.6f} "
        f"(oracle {case.k_inf:.6f}) — the homogeneous gate does NOT execute "
        f"the multiplier's assembly (Mode-11 vacuous-green)"
    )


# ── The eigen-solve call path: #276 P4-D routed it through direct_eigenvalue;
#    taxonomy step 5b re-spelled it as the K-operator composition
#    ``K = MatrixInverseOperator(loss) @ production`` + dominant_eigenpair ──
#
# These gates pin the production call path of the eigen-solve. Equivalence-
# class history:
#
#   * P4-D (direct_eigenvalue rewire) — k_inf BIT-IDENTICAL: the eig
#     computation was structurally unchanged (same ``eig(solve(A, F))``).
#   * Step 5b (the K-operator spelling) — k_inf is PRINCIPLED-EQUIVALENCE,
#     gated at rtol=1e-12: the resolvent formation deliberately changed
#     LAPACK call sequence (one batched ``gesv`` → a held ``lu_factor`` +
#     one ``lu_solve`` backsolve per basis column), so drift up to ~κ(A)·ULP
#     is admissible on another BLAS build (measured bit-identical on this
#     host). All three re-baseline criteria hold: the resolvent is formed by
#     NAMED operators (MatrixInverseOperator = A⁻¹, the fission dyad = F,
#     OperatorProduct = A⁻¹F); the structurally-INDEPENDENT anchor stays
#     ``test_kinf_exact`` (SymPy ``case.k_inf``, 1e-12), into which
#     dominant_eigenpair must NOT be wired; the FP drift (~1e-14) is ≪ any
#     rewire bug (O(1e-3)+).
#   * rates / flux via IntegratedReactionRate — BIT-IDENTICAL on the shipped
#     cases (ng ≤ 4): V_cell = 1 and the short Σ_g νΣf_g φ_g reduction matches
#     ``νΣf @ φ`` exactly, comparing two computations on the SAME φ (whatever
#     the solver produced). (A larger group structure would relax this to ≤ few
#     ULP — a reduction-tree change per vv-principles bit-identity criterion 3.)


def _require(condition: bool, message: str) -> None:
    """A ``-O``-firing assertion (NOT a bare assert) for the liveness sentinel."""
    if not condition:
        pytest.fail(message)


def test_dominant_eigenpair_is_on_the_homogeneous_call_path(monkeypatch):
    """Mode-11: ``solve_homogeneous_infinite`` actually CALLS
    ``dominant_eigenpair`` (the K-path eigenvalue primitive) — not a
    routed-around inline ``eig``.

    In-process WRAP sentinel (the gold-standard Mode-11 proof): spies the
    symbol in the SOLVER's own namespace (the module-level ``from ... import``
    binding is the thing wrapped) and asserts the counter fires during the
    solve. HARD requirement — post-carve, symbol absence is a REGRESSION
    (the step-5b predecessor of this gate silently SKIPPED when its spied
    symbol left the namespace; never again).
    """
    import orpheus.homogeneous.solver as hsolver

    _require(
        hasattr(hsolver, "dominant_eigenpair"),
        "solver.py does not import dominant_eigenpair — the K-spelling rewire "
        "regressed (or the solver re-binds a different eigenvalue engine; "
        "re-target this spy, never let it skip).",
    )

    calls: list[int] = []

    def _wrap(ns: object, name: str) -> None:
        original = getattr(ns, name)

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(ns, name, spy)

    _wrap(hsolver, "dominant_eigenpair")
    if hasattr(_eig, "dominant_eigenpair"):  # definition site too (in-function import)
        _wrap(_eig, "dominant_eigenpair")

    case = get("homo_2eg_n2n")
    mix = next(iter(case.materials.values()))
    result = solve_homogeneous_infinite(mix)
    _require(
        len(calls) >= 1,
        "solve_homogeneous_infinite did NOT call dominant_eigenpair — the "
        "K-path eigenvalue primitive is not on the call graph (Mode-11 "
        "vacuous green).",
    )
    np.testing.assert_allclose(result.k_inf, case.k_inf, atol=1e-12, rtol=0)


def test_kinf_matches_direct_eigenvalue_engine_of_the_assembled_pair():
    """CROSS-ENGINE equivalence (principled, NOT byte): solver ``k_inf``
    (the K-path — MatrixInverseOperator ``lu_solve`` resolvent) vs
    ``direct_eigenvalue(A, F)`` (``np.linalg.solve`` resolvent) on the SAME
    assembled pair.

    Both engines extract through the SAME ``dominant_eigenpair``, so this
    localizes a REWIRE regression (factor swap, wrong basis_shape, transposed
    resolvent — all O(1) on k) to the resolvent-FORMATION boundary. It is NOT
    structurally independent on the eig side and PAIRS with ``test_kinf_exact``
    (the SymPy anchor). rtol=1e-12 per the step-5b equivalence-class note
    above: bit-identical on this host, κ(A)·ULP-portable across BLAS builds.
    """
    from orpheus.homogeneous.solver import HomogeneousProblem, _pose_space
    from orpheus.transport.mesh.material_mesh import MaterialMesh
    from orpheus.transport.operators.isotropic_transfer import (
        IsotropicFission,
    )

    case = get("homo_2eg_n2n")
    mix = next(iter(case.materials.values()))
    mat_xs = MaterialMesh.from_materials({0: mix}).material_xs_field()
    # CS4a K2: mirror the production spelling — the mixture-minted pose
    # threads every arm and every as_matrix derives its shape from it.
    space = _pose_space(mix)
    A = HomogeneousProblem(mix).loss.as_matrix()
    F = IsotropicFission.from_material_xs(mat_xs, space=space).as_matrix()
    k_engine = _eig.direct_eigenvalue(A, F)[0]

    result = solve_homogeneous_infinite(mix)
    np.testing.assert_allclose(result.k_inf, k_engine, rtol=1e-12, atol=0)


def test_matrix_inverse_operator_apply_is_on_the_homogeneous_call_path(monkeypatch):
    """Mode-11 liveness for the FIRST production consumer claim: the
    ``MatrixInverseOperator.apply`` LU backsolve executes during
    ``solve_homogeneous_infinite`` (once per resolvent column).

    The value gates (fused oracle, cross-engine, SymPy anchor) are
    structurally BLIND to WHICH code formed the resolvent — only a fired
    sentinel proves the inverse operator is the genuine producer.
    ``K.as_matrix()`` *structurally must* route ``Minv.apply(F.apply(e_j))``
    through the generic apply-to-basis loop, but "structurally must" is
    exactly the assumption Mode-11 says to verify, never assume. The floor
    is ``>= 1`` (honest liveness), not ``== ng`` — a future batched
    ``as_matrix`` override on the product would change the count without
    changing the claim.
    """
    from orpheus.numerics.matrix_inverse_operator import MatrixInverseOperator

    calls: list[int] = []
    raw = MatrixInverseOperator.apply

    def spy(self, x, /, *, initial_guess=None):
        calls.append(1)
        return raw(self, x, initial_guess=initial_guess)

    monkeypatch.setattr(MatrixInverseOperator, "apply", spy)

    case = get("homo_2eg_n2n")
    mix = next(iter(case.materials.values()))
    result = solve_homogeneous_infinite(mix)
    _require(
        len(calls) >= 1,
        "MatrixInverseOperator.apply never fired during the solve — "
        "K.as_matrix() routed around the inverse operator (Mode-11 vacuous "
        "green for the first-production-consumer claim).",
    )
    np.testing.assert_allclose(result.k_inf, case.k_inf, atol=1e-12, rtol=0)


@pytest.mark.verifies("resolvent-object-gate")
def test_K_operator_as_matrix_is_the_resolvent():
    """The genuinely-new step-5b structural element:
    ``OperatorProduct(MatrixInverseOperator(A), F).as_matrix()`` == the
    resolvent ``A⁻¹F``.

    Reference ``np.linalg.solve(A_dense, F_dense)`` — a DIFFERENT primitive
    than the operator-algebra path (procedurally independent), pinning the
    ``@``-composition of an inverse with the rank-1 fission dyad at the
    operator boundary (a factor swap ``F·A⁻¹``, a dropped factor, or a
    ``basis_shape`` threading bug reds HERE, faster and sharper than at the
    end-to-end eig). Successful construction of ``K`` doubles as the
    None-tolerant space-guard assertion for the meshless pair (an
    ``IncompatibleOperatorComposition`` here means a FunctionSpace leaked
    onto a meshless operand — investigate before touching the guard).
    """
    from orpheus.homogeneous.solver import HomogeneousProblem, _pose_space
    from orpheus.numerics.matrix_inverse_operator import MatrixInverseOperator
    from orpheus.transport.mesh.material_mesh import MaterialMesh
    from orpheus.transport.operators.isotropic_transfer import (
        IsotropicFission,
    )

    case = get("homo_2eg_n2n")
    mix = next(iter(case.materials.values()))
    mat_xs = MaterialMesh.from_materials({0: mix}).material_xs_field()

    # CS4a K2: the line-for-line production mirror (solver.py builds K
    # exactly this way — the mixture-minted pose threaded everywhere, no
    # explicit basis_shape anywhere; the shapes derive from the domain).
    space = _pose_space(mix)
    loss = HomogeneousProblem(mix).loss
    production = IsotropicFission.from_material_xs(mat_xs, space=space)
    K = MatrixInverseOperator(loss) @ production

    A = loss.as_matrix()
    F = production.as_matrix()
    np.testing.assert_allclose(
        K.as_matrix(),
        np.linalg.solve(A, F),
        rtol=1e-12,
        atol=0,
    )


def test_rates_via_integrated_reaction_rate_are_bit_identical():
    r"""Bit-identity of the rate rerouting: ``IntegratedReactionRate(νΣf).evaluate(φ)``
    == ``νΣf @ φ`` on the meshless unit-volume cell, for every shipped case.

    ``V_cell = 1`` and the short ``Σ_g νΣf_g φ_g`` reduction matches the dot
    bit-for-bit for ng ∈ {1, 2, 4}. Pins the production-rate rerouting
    independently of the eig; runs green TODAY (both paths exist) and guards
    the rewire. (No PRE-IMPL skip — it protects the rate-side claim regardless
    of the eigenvalue-side rewire state.)
    """
    from orpheus.transport.mesh.material_mesh import MaterialMesh
    from orpheus.transport.reaction_rate_functional import IntegratedReactionRate

    for case_name in ("homo_1eg", "homo_2eg", "homo_4eg", "homo_2eg_n2n"):
        case = get(case_name)
        mix = next(iter(case.materials.values()))
        mat_xs = MaterialMesh.from_materials({0: mix}).material_xs_field()
        ng = mix.ng
        phi = solve_homogeneous_infinite(mix).flux

        legacy_prod = float(mat_xs.fission_production[:, 0] @ phi)
        irr_prod = float(
            IntegratedReactionRate(mat_xs.fission_production_field).evaluate(
                phi.reshape(ng, 1)
            )
        )
        _require(
            legacy_prod == irr_prod,
            f"{case_name}: IntegratedReactionRate production {irr_prod!r} != "
            f"νΣf@φ {legacy_prod!r} — the rate rerouting is not bit-identical.",
        )


# ── #276 P4-F: energy-grid diagnostics folded onto EnergyGrid ──


def test_eg_block_wires_geometric_grid_diagnostics():
    """L1: the eg-block diagnostics are the mixture's EnergyGrid properties — the
    GEOMETRIC group centre + energy/lethargy widths — wired through, NOT
    re-derived as an arithmetic midpoint. (The eg-block was previously untested;
    #276 P4-F closes that gap and pins the geometric switch end-to-end.)
    """
    base = next(iter(get("homo_2eg").materials.values()))  # 2-group, eg=None
    eg = np.array([1.0e7, 1.0e3, 1.0e-3])  # descending edges → 2 groups, each 4 decades
    mix = dataclasses.replace(base, eg=eg)
    result = solve_homogeneous_infinite(mix)
    grid = mix.energy_grid
    rep, ew, lw = result.representative_energy, result.energy_widths, result.lethargy_widths
    assert rep is not None and ew is not None and lw is not None  # eg set ⟹ populated

    # GEOMETRIC centre √(E_up·E_lo), NOT 0.5·(E_up+E_lo):
    # √(1e7·1e3)=1e5, √(1e3·1e-3)=1 (the arithmetic midpoints would be ~5e6, ~500).
    np.testing.assert_array_equal(rep, grid.representative_energy)
    np.testing.assert_allclose(rep, [1.0e5, 1.0], rtol=1e-12)
    np.testing.assert_array_equal(ew, grid.energy_widths)
    np.testing.assert_array_equal(lw, grid.lethargy_widths)
    # the flux densities divide by the EnergyGrid widths
    np.testing.assert_allclose(result.flux_per_energy, result.flux / grid.energy_widths)
    np.testing.assert_allclose(result.flux_per_lethargy, result.flux / grid.lethargy_widths)


def test_synthetic_mixture_diagnostics_none_and_flux_densities_raise():
    """L1: a synthetic mixture (eg=None) leaves the energy-grid diagnostics None;
    flux_per_energy / flux_per_lethargy raise (undefined without a grid)."""
    result = solve_homogeneous_infinite(next(iter(get("homo_2eg").materials.values())))
    _require(result.representative_energy is None, "representative_energy must be None (eg=None)")
    _require(result.energy_widths is None, "energy_widths must be None (eg=None)")
    _require(result.lethargy_widths is None, "lethargy_widths must be None (eg=None)")
    with pytest.raises(ValueError):
        _ = result.flux_per_energy
    with pytest.raises(ValueError):
        _ = result.flux_per_lethargy